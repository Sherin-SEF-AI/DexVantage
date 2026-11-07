"""What the optimiser and the design agent share: scoring a design through core.metrics (with a memo and a result
cache), the settings both read, choosing the options to show from the Pareto front, and the plain-English
explanation of each option's trade-offs, computed from its figures.

    Evaluator(objectives, constraints, screen, cancel)   .score(project) -> Score (figures, f, g, cv)
    choose(entries, objectives, k)                       up to k options spread along the front
    explain(option, options, current, objectives, constraints) -> str
    fmt(value, unit) -> str                              a figure as the page shows it

Known limits: see core/optimise/__init__.py.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import numpy as np

from .. import jobs
from .. import metrics as M
from . import nsga2 as N
from . import objectives as OB

# The studies' settings while searching (labelled defaults; settings 'screen' replaces them, {} = the project's own):
# the grasp study on 40 grasps with no sensitivity, gallery or pre-contact steps and 200 bootstrap resamples (about
# 0.5–1 s instead of 40 s on the UR5e sample), brackets without meshing on 12 joint samples. The figures of a design
# are its figures at these settings; 'confirm' re-scores the options shown at the project's own settings.
SCREEN = {'grasp_visibility': {'limit': 40, 'sensitivity': False, 'gallery': 0, 'bootstrap': 200,
                               'precontact_steps': 0},
          'brackets': {'samples': 12, 'mesh': False}}
NOT_STATED = math.inf          # a figure a study does not state ranks below every stated one (sorting only)
_CACHE = None
_CACHE_LOCK = threading.Lock()


def shared_cache() -> jobs.ResultCache:
    """The study results the searches keep between runs (a design scored before is not scored again)."""
    global _CACHE
    with _CACHE_LOCK:
        if _CACHE is None:
            _CACHE = jobs.ResultCache(size=256)
        return _CACHE


def clear_cache():
    global _CACHE
    with _CACHE_LOCK:
        _CACHE = None


# ====================================================================== settings
def _int(v, default, lo, hi):
    try:
        x = int(round(float(v)))
    except (TypeError, ValueError, OverflowError):
        return default
    return min(max(x, lo), hi)


def _float(v, default, lo, hi):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(x):
        return default
    return min(max(x, lo), hi)


COMMON = {'objectives': list(OB.DEFAULT_OBJECTIVES), 'budget': 96, 'population': 12, 'seed': 0, 'options': 5,
          'confirm': False, 'constraints': [], 'screen': None, 'eta_c': 15.0, 'eta_m': 20.0, 'pc': 0.9}
LIMITS = {'budget': (4, 20000), 'population': (4, 400), 'seed': (0, 2 ** 31 - 1), 'options': (1, 12),
          'eta_c': (1.0, 100.0), 'eta_m': (1.0, 100.0), 'pc': (0.0, 1.0)}


def read_settings(text: str) -> dict:
    """Settings from JSON text or a .json file (batch --optimise-settings); StudyError in a sentence when they cannot
    be read."""
    import json
    src = str(text or '')
    try:
        if not src.lstrip().startswith(('{', '[', '"')) and not src.strip().lstrip('-').replace('.', '').isdigit():
            with open(src, encoding='utf-8') as fh:
                src = fh.read()
        doc = json.loads(src)
    except OSError as e:
        raise jobs.StudyError(f'--optimise-settings: could not read {text} ({e.strerror or e}).') from None
    except ValueError as e:
        raise jobs.StudyError(f'--optimise-settings is not JSON ({e}).') from None
    if not isinstance(doc, dict):
        raise jobs.StudyError('--optimise-settings must be a JSON object, e.g. {"budget": 200, "seed": 3}.')
    return doc


def common_settings(given: dict, defaults: dict) -> dict:
    """The settings both searches read, tolerant of junk (a bad value falls back to its default, said in
    'defaults'); every number a labelled default unless given."""
    given = given if isinstance(given, dict) else {}
    out = dict(defaults)
    used = []
    for k, dv in defaults.items():
        g = given.get(k)
        if g is None:
            used.append(k)
            continue
        if k in LIMITS and isinstance(dv, int) and not isinstance(dv, bool):
            out[k] = _int(g, dv, *LIMITS[k])
        elif k in LIMITS:
            out[k] = _float(g, dv, *LIMITS[k])
        elif isinstance(dv, bool):
            out[k] = bool(g)
        elif k == 'objectives':
            ids = [str(x) for x in g] if isinstance(g, (list, tuple)) else []
            out[k] = list(dict.fromkeys(ids)) or list(dv)
        elif k == 'constraints':
            out[k] = [c for c in g if isinstance(c, dict)] if isinstance(g, list) else []
        elif k == 'screen':
            out[k] = {str(s): dict(v) for s, v in g.items() if isinstance(v, dict)} if isinstance(g, dict) else None
        else:
            out[k] = g
    out['budget'] = max(out['budget'], out['population'])
    if out['screen'] is None:
        out['screen'] = {k: dict(v) for k, v in SCREEN.items()}
    out['defaults'] = used
    return out


# ====================================================================== scoring
@dataclass
class Score:
    figures: dict                 # metric id -> value (natural units) or None
    status: dict                  # metric id -> 'ok' | 'not stated' | 'refused'
    reasons: dict                 # metric id -> the study's sentence when refused
    f: np.ndarray                 # minimised objectives (+inf where not stated)
    g: np.ndarray                 # constraint violations
    cv: float
    violations: list = field(default_factory=list)   # [{'name', 'metric', 'value', 'threshold', 'op', 'met'}]
    bounds: dict = field(default_factory=dict)       # metric id -> 'at_least' | 'at_most' (a figure stated as a bound)


class _Cancelled(Exception):
    pass


class Evaluator:
    """Scores projects on the objectives and constraints through core.metrics.evaluate, every study run with the
    screening settings, results from the shared cache, and identical designs from a memo."""

    def __init__(self, objectives: list, constraints: list, screen: dict, cancel=None, cache=None):
        OB.ensure_metrics()
        self.objectives = list(objectives)
        self.constraints = list(constraints)
        self.screen = screen or {}
        self.cancel = cancel
        self.cache = cache if cache is not None else shared_cache()
        self.memo: dict = {}
        self.reused = 0
        self.scored = 0

    def ids(self) -> list:
        return list(dict.fromkeys(self.objectives + [c['metric'] for c in self.constraints]))

    def screened(self, project, screen=None):
        screen = self.screen if screen is None else screen
        for sid, st in screen.items():
            project.analysis.studies[sid] = {**(project.analysis.studies.get(sid) or {}), **st}
        return project

    def figures(self, project, screen=None) -> tuple[dict, dict, dict]:
        """(values, statuses, reasons) of every metric on the project. Each study is evaluated on its own, so a study
        that stops with an unexpected error (a bug, logged with its traceback) leaves its metrics 'refused' with a
        plain reason instead of stopping the search."""
        return self._figures(project, screen)[:3]

    def _figures(self, project, screen=None) -> tuple[dict, dict, dict, dict]:
        """figures() and {metric id: bound} of the figures a study states only as a bound."""
        import traceback
        p = self.screened(project, screen)
        ctx = jobs.Ctx(cancel=self.cancel)
        groups: dict = {}
        for mid in self.ids():
            groups.setdefault(M.get(mid).study, []).append(mid)
        rows = {}
        for sid, ids in groups.items():
            try:
                rows.update(M.evaluate(p, ids, cache=self.cache, ctx=ctx))
            except Exception as e:                    # noqa: BLE001 — said in words, the traceback logged
                traceback.print_exc()
                title = jobs.STUDIES[sid].title if sid in jobs.STUDIES else sid
                for mid in ids:
                    rows[mid] = {'value': None, 'status': 'refused',
                                 'reason': f'{title} stopped with an unexpected error ({type(e).__name__}) on this '
                                           'design; the details are in the log'}
            if ctx.cancelled():
                raise _Cancelled()
        return ({k: r['value'] for k, r in rows.items()}, {k: r['status'] for k, r in rows.items()},
                {k: r.get('reason', '') for k, r in rows.items() if r.get('reason')},
                {k: r['bound'] for k, r in rows.items() if r.get('bound') and r.get('value') is not None})

    def score(self, project, key=None, screen=None) -> Score:
        if key is not None and key in self.memo:
            self.reused += 1
            return self.memo[key]
        figs, status, reasons, bounds = self._figures(project, screen)
        self.scored += 1
        s = self.make(figs, status, reasons, bounds)
        if key is not None:
            self.memo[key] = s
        return s

    def make(self, figs, status, reasons, bounds=None) -> Score:
        bounds = bounds or {}
        f = np.array([NOT_STATED if figs.get(m) is None else OB.minimised(m, figs[m]) for m in self.objectives],
                     float)
        g, viol = [], []
        for c in self.constraints:
            v = figs.get(c['metric'])
            x = OB.violation(c, v, bounds.get(c['metric']))
            g.append(x)
            viol.append({'name': c['name'], 'metric': c['metric'], 'op': c['op'], 'threshold': c['threshold'],
                         'unit': c['unit'], 'value': v, 'met': x <= 0.0, 'is_default': c.get('is_default', False)})
        g = np.array(g, float)
        return Score(figs, status, reasons, f, g, float(g.sum()) if len(g) else 0.0, viol, dict(bounds))


def metric_info(mid: str) -> dict:
    m = M.get(mid)
    return {'id': mid, 'title': m.title, 'unit': m.unit, 'better': m.better, 'study': m.study}


# ====================================================================== options
def choose(entries: list, n_obj: int, k: int, reference: dict | None = None) -> list:
    """Up to k entries spread along the front: entries are dicts with 'f' (minimised, +inf = not stated) and 'cv';
    the feasible non-dominated ones (else the least-violating front), the same figures counted once, then the best on
    each objective first and the rest by crowding distance. With a reference (the current rig: 'f', 'cv'), it takes
    part in the dominance comparison and is never chosen: a design it dominates is not an option. Each chosen entry
    gets 'on_front'."""
    if not entries:
        return []
    F = np.array([e['f'] for e in entries], float).reshape(len(entries), n_obj)
    CV = np.array([e['cv'] for e in entries], float)
    if reference is not None:
        F = np.vstack([F, np.asarray(reference['f'], float).reshape(1, n_obj)])
        CV = np.r_[CV, float(reference['cv'])]
    front = [int(i) for i in N.non_dominated_sort(F, CV)[0] if i < len(entries)]
    seen, uniq = set(), []
    for i in front:
        key = tuple(None if not math.isfinite(x) else round(float(x), 6) for x in F[i])
        if key not in seen:
            seen.add(key)
            uniq.append(int(i))
    pick = []
    for j in range(n_obj):
        col = F[uniq, j]
        if np.isfinite(col).any():
            best = uniq[int(np.nanargmin(np.where(np.isfinite(col), col, np.nan)))]
            if best not in pick:
                pick.append(best)
    if len(uniq) > len(pick):
        cd = N.crowding_distance(F[uniq])
        for i in [uniq[t] for t in np.argsort(-cd, kind='stable')]:
            if i not in pick:
                pick.append(i)
    pick = pick[:k]
    out = []
    for i in pick:
        e = dict(entries[i])
        e['on_front'] = True
        out.append(e)
    # sorted by the first objective (best first), then the others
    out.sort(key=lambda e: tuple(np.nan_to_num(np.asarray(e['f'], float), posinf=1e300)))
    return out


def finite_or_none(f) -> list:
    """Minimised objective values for the JSON result: a not-stated +inf becomes null."""
    return [float(x) if x is not None and math.isfinite(float(x)) else None for x in f]


def dominates(a, b) -> bool:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool((a <= b).all() and (a < b).any())


# ====================================================================== words
def fmt(v, unit: str) -> str:
    if v is None:
        return 'not stated'
    if unit == '%':
        return f'{v:.1f} %'
    if unit == 'count':
        return f'{v:.0f}'
    if unit == 'tokens':
        return f'{v:.0f} tokens'
    if len(unit) == 3 and unit.isalpha() and unit.isupper():      # a currency code
        return f'{v:,.2f} {unit}'
    if unit in ('Mbit/s', 'mm', 'ms', 'px', 'g', 'W'):
        return f'{v:,.1f} {unit}'
    return f'{v:.3g} {unit}'.strip()


NUMBER = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five'}


def delta_words(o: dict, d: float) -> str:
    """A difference d (option − other) of metric o in words: '4.5 points more contact visibility', 'one camera
    fewer', '120.00 USD less cost'."""
    unit, mag = o['unit'], abs(d)
    more = d > 0
    title = o['title'][0].lower() + o['title'][1:]
    if unit == '%':
        return f'{mag:.1f} points {"more" if more else "less"} {title}'
    if unit == 'count':
        n = int(round(mag))
        noun = _singular(title) if n == 1 else title
        return f'{NUMBER.get(n, str(n))} {"more" if more else "fewer"} {noun}'
    if unit == 'tokens':
        return f'{mag:.0f} token{"s" if round(mag) != 1 else ""} {"more" if more else "fewer"} ({title})'
    return f'{fmt(mag, unit)} {"more" if more else "less"} {title}'


def _singular(title: str) -> str:
    """'sensors without a ready bracket' -> 'sensor without a ready bracket' (the first word only)."""
    first, _, rest = title.partition(' ')
    if first.endswith('s') and not first.endswith('ss') and len(first) > 3:
        first = first[:-1]
    return f'{first} {rest}'.strip()


def best_words(o: dict) -> str:
    """'the fewest cameras', 'the highest contact visibility' for the best value of metric o."""
    title = o['title'][0].lower() + o['title'][1:]
    if o['id'] == 'rig.camera_count':
        return 'the fewest cameras'
    if o['unit'] == 'count':
        return f'the fewest {title}' if o['better'] == 'lower' else f'the most {title}'
    return f'the {"highest" if o["better"] == "higher" else "lowest"} {title}'


def _gain(o: dict, a, b):
    """How much better a is than b on metric o (positive = better), or None."""
    if a is None or b is None:
        return None
    return (a - b) if o['better'] == 'higher' else (b - a)


def _significant(o: dict, g) -> bool:
    if g is None:
        return False
    return abs(g) >= {'%': 0.05, 'count': 0.5, 'tokens': 0.5}.get(o['unit'], 1e-6)


def explain(opt: dict, options: list, current: dict | None, objs: list, constraints: list) -> str:
    """The option's trade-offs in plain sentences, computed from the figures (no figure is invented here)."""
    figs = opt['figures']
    parts = []
    lead = opt.get('summary') or ''
    if lead:
        parts.append(lead.rstrip('.') + '.')
    # against the current rig
    if current is not None:
        cf = current['figures']
        up, down = [], []
        for o in objs:
            g = _gain(o, figs.get(o['id']), cf.get(o['id']))
            if not _significant(o, g):
                continue
            d = figs[o['id']] - cf[o['id']]
            (up if g > 0 else down).append(delta_words(o, d) if o['unit'] == 'count' else
                                           f'{delta_words(o, d)} ({fmt(figs[o["id"]], o["unit"])})')
        if up and down:
            parts.append(f'Against the current rig it has {_join(up)}, at the price of {_join(down)}.')
        elif up:
            unk = any(figs.get(o['id']) is None or cf.get(o['id']) is None for o in objs)
            parts.append(f'Against the current rig it has {_join(up)}, and is no worse on the other objectives'
                         + (' that are stated for both.' if unk else '.'))
        elif down:
            parts.append(f'Against the current rig it has {_join(down)}, and is better on none.')
        else:
            parts.append('Its figures are those of the current rig.')
    # best of the set
    best = []
    for o in objs:
        vals = [x['figures'].get(o['id']) for x in options if x['figures'].get(o['id']) is not None]
        v = figs.get(o['id'])
        if v is None or len(vals) < 2:
            continue
        top = max(vals) if o['better'] == 'higher' else min(vals)
        if abs(v - top) < 1e-9 and sum(abs(x - top) < 1e-9 for x in vals) == 1:      # the only one
            best.append(best_words(o))
    if best:
        parts.append(f'Of the {len(options)} designs shown it has {_join(best)}.')
    # against its nearest neighbour on the front
    nb = _neighbour(opt, options, objs)
    if nb is not None:
        better, worse = [], []
        for o in objs:
            g = _gain(o, figs.get(o['id']), nb['figures'].get(o['id']))
            if not _significant(o, g):
                continue
            d = figs[o['id']] - nb['figures'][o['id']]
            (better if g > 0 else worse).append(delta_words(o, d))
        if better and worse:
            parts.append(f'Against design {nb["label"]} it trades {_join(worse)} for {_join(better)}.')
        elif better:
            parts.append(f'Against design {nb["label"]} it has {_join(better)}.')
    # requirements
    if constraints:
        miss = [v for v in opt.get('violations') or [] if not v['met']]
        if not miss:
            parts.append(f'It meets every requirement ({len(constraints)}).')
        else:
            parts.append('It misses ' + _join([f'{v["name"]} ({fmt(v["value"], v["unit"])} against '
                                               f'{v["op"]} {fmt(v["threshold"], v["unit"])})' for v in miss]) + '.')
    gone = [o['title'] for o in objs if figs.get(o['id']) is None]
    if gone:
        parts.append(f'{_join(gone)} {"is" if len(gone) == 1 else "are"} not stated for it.')
    return ' '.join(parts)


def _neighbour(opt, options, objs):
    """The other option nearest in the objectives scaled by their range over the options (None alone)."""
    others = [x for x in options if x is not opt and x.get('label') != opt.get('label')]
    if not others:
        return None
    ids = [o['id'] for o in objs]
    rng = {}
    for k in ids:
        vals = [x['figures'].get(k) for x in options if x['figures'].get(k) is not None]
        rng[k] = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
    best, bd = None, math.inf
    for x in others:
        d = 0.0
        for k in ids:
            a, b = opt['figures'].get(k), x['figures'].get(k)
            if a is None or b is None or rng[k] <= 0:
                continue
            d += ((a - b) / rng[k]) ** 2
        if d < bd:
            best, bd = x, d
    return best


def _join(items) -> str:
    items = list(items)
    if len(items) <= 1:
        return ''.join(items)
    return ', '.join(items[:-1]) + ' and ' + items[-1]


def headline(options: list, current: dict | None, constraints: list, cancelled: bool, n_obj: int) -> dict:
    """The page's figure and STATUS word: how many designs the front offers and what they do for this rig."""
    feas = [o for o in options if o['feasible']]
    better = [o for o in feas if current is not None and dominates(o['f'], current['f'])]
    if cancelled and not options:
        return {'value': 0, 'unit': 'designs', 'status': 'warning', 'status_word': 'cancelled',
                'text': 'Cancelled before any design was scored.'}
    if constraints and not feas:
        st, word = 'critical', 'misses the requirements'
        text = ('No design found meets every requirement: the ones shown come closest. Loosen a requirement, widen '
                'the search (more evaluations, larger moves) or let it change the devices.')
    elif better:
        st, word = 'good', 'better rig found'
        n = len(better)
        text = (f'{n} of the {len(options)} designs {"is" if n == 1 else "are"} at least as good as the current rig on '
                'every objective and better on one' + ('; the others trade one objective for another.'
                                                        if n < len(options) else '.'))
    elif options:
        st, word = 'warning', 'trade-offs only'
        text = ('No design beats the current rig on every objective at once: each one shown gains on one objective '
                'and gives up on another. Choose the trade-off you want.')
    elif current is not None and not cancelled:
        st, word = 'good', 'current rig is best'
        text = ('Every design scored is no better than the current rig on any objective without being worse on '
                'another: keep the rig, or widen the search (more evaluations, larger moves, other devices).')
    else:
        st, word = 'warning', 'no designs'
        text = 'The search scored no design.'
    if cancelled:
        word += ' (cancelled)'
        text = 'Cancelled: the designs scored so far. ' + text
    if not constraints:
        text += ' No requirement is set, so every design counts as meeting them.'
    return {'value': len(options), 'unit': 'designs', 'status': st, 'status_word': word, 'text': text,
            'better_than_current': [o['label'] for o in better]}
