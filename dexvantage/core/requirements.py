"""Requirements and the readiness scorecard (spec §7.11, §11, §13): the rig's written requirements as measurable
pass/fail checks, evaluated against the studies' own results in one place, with one verdict and a CI gate
(batch --check requirements.yaml).

Ported from Vantage core/requirements.py: the comparison with a signed margin, the verdict over the items, the
scorecard document and the CI gate. Stripped: Vantage's automotive metric registry (sectors, object classes, lidar
and GNSS figures), priorities, the brief parser and the review pack. DexVantage's metrics are the named figures of
core/metrics.py (definitions in core/metric_defs.py), so a requirement is a metric id, a comparison and a threshold:

    {'name': 'Contact visibility', 'metric': 'grasp.contact_visibility', 'op': '>=', 'threshold': 95, 'unit': '%',
     'study': 'grasp_visibility', 'is_default': True, 'note': ...}       (core.project.Requirement)

The margin is signed so that positive is good: value - threshold for >= and >, threshold - value for <= and <, and
-|value - threshold| for ==. An item passes or fails on its study's number; it is "not evaluated" when the study
refuses (its sentence is the reason), when the result does not state the figure, or when the study states only a
bound that does not decide it (a time-sync skew of "at least 1 ms" against "< 2 ms"). The verdict passes when every
item passes; one failure fails it; otherwise an item not evaluated leaves it "not evaluated". The CI gate exits 0 on
pass, 1 on fail or not evaluated (a gate cannot certify what it did not measure) and 2 on an error (a requirements
file that cannot be read, an unknown metric, a threshold that is not a number, a study that crashed).

The defaults (DEFAULTS, labelled as defaults, never as standards): contact visibility >= 95 %; camera to joint-state
skew < 2 ms; no controller over budget; no cable pull or snag; no sensory blind gap before contact.

Every figure is the study's own: the scorecard computes nothing beyond reading, comparing and reporting.

Known limits: the scorecard evaluates the active variant (compare_variants every variant) with each study's stored
settings; the studies' own Known limits apply unchanged; a bound decides an item only in the direction it bounds;
"no blind gap" is the multi-finger study's handoff scan, sampled at its refine step (2 mm by default), so a gap
shorter than a step reads 0; there are no priorities (every requirement is decisive); thresholds compare to the
study's rounded figure (the precision each study writes).
"""
from __future__ import annotations

import math
import os
import time
import traceback

from . import jobs
from . import metrics as M
from .jobs import StudyError
from .project import Requirement

SCHEMA = 'dexvantage.requirements/1'
VARIANTS_SCHEMA = 'dexvantage.variants_compare/1'
FORMAT = 'dexvantage.requirements'
OPS = ('>=', '<=', '<', '>', '==')
OP_TEXT = {'>=': '≥', '<=': '≤', '==': '=', '<': '<', '>': '>'}
OP_WORDS = {'>=': 'at least', '<=': 'at most', '==': 'exactly', '<': 'less than', '>': 'more than'}
OP_ALIASES = {'≥': '>=', '=>': '>=', 'at least': '>=', 'min': '>=', 'ge': '>=', 'gte': '>=',
              '≤': '<=', '=<': '<=', 'at most': '<=', 'max': '<=', 'le': '<=', 'lte': '<=',
              'less than': '<', 'lt': '<', 'below': '<', 'more than': '>', 'gt': '>', 'above': '>',
              '=': '==', 'exactly': '==', 'eq': '==', 'equals': '=='}
STATUSES = ('pass', 'fail', 'not_evaluated')
STATUS_TEXT = {'pass': 'pass', 'fail': 'fail', 'not_evaluated': 'not evaluated'}
STATUS_LEVEL = {'pass': 'good', 'fail': 'critical', 'not_evaluated': 'warning'}       # ui/theme STATUS colours
VERDICT_TEXT = {'pass': 'ready', 'fail': 'not ready', 'incomplete': 'not evaluated', 'empty': 'no requirements'}
VERDICT_LEVEL = {'pass': 'good', 'fail': 'critical', 'incomplete': 'warning', 'empty': 'warning'}
EXIT_PASS, EXIT_FAIL, EXIT_ERROR = 0, 1, 2
UNIT_ALIASES = {'percent': '%', 'pct': '%', 'deg': '°', 'degrees': '°', 'millimetres': 'mm', 'millimeters': 'mm',
                'milliseconds': 'ms', 'pixels': 'px', 'mbit/s': 'Mbit/s', 'mbps': 'Mbit/s', 'count': 'count', '': ''}
DEFAULT_NOTE = 'DexVantage default (a documented starting point, not a standard): change it to your own figure.'
ASSUMPTIONS = ('The scorecard reads every figure off the studies\' own results and compares it with the threshold; it '
               'models nothing itself. Each study\'s assumptions and Known limits apply unchanged. A figure a study '
               'states only as a bound decides a requirement only in the direction it bounds.')

# the defaults of spec §11 (labelled as defaults: is_default True and the note says so)
DEFAULTS = (
    {'name': 'Contact visibility', 'metric': 'grasp.contact_visibility', 'op': '>=', 'threshold': 95.0},
    {'name': 'Camera to joint-state skew', 'metric': 'sync.skew_ms', 'op': '<', 'threshold': 2.0},
    {'name': 'No controller over budget', 'metric': 'bandwidth.over_budget', 'op': '<=', 'threshold': 0.0},
    {'name': 'No cable pull or snag', 'metric': 'cables.pull_snag', 'op': '<=', 'threshold': 0.0},
    {'name': 'No sensory blind gap before contact', 'metric': 'dexhand.blind_gap_mm', 'op': '<=', 'threshold': 0.0},
)

_CACHE = jobs.ResultCache(24)        # the studies' results shared by the scorecard, the variants compare and --check


class RequirementsError(StudyError):
    """A requirement or a requirements file that cannot be used; the message says why in plain words."""


# ====================================================================== items
def _num(v):
    if isinstance(v, bool):
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def norm_op(op) -> str | None:
    s = str(op if op is not None else '').strip()
    if s in OPS:
        return s
    return OP_ALIASES.get(s.lower())


def norm_unit(u) -> str:
    s = str(u or '').strip()
    return UNIT_ALIASES.get(s.lower(), s)


def default_requirements() -> list:
    """The labelled defaults of spec §11 as Requirement objects (new uids)."""
    out = []
    for d in DEFAULTS:
        m = M.get(d['metric'])
        out.append(Requirement(name=d['name'], metric=d['metric'], op=d['op'], threshold=float(d['threshold']),
                               unit=m.unit, study=m.study, is_default=True, note=DEFAULT_NOTE))
    return out


def clean(d, where: str = 'the requirement') -> Requirement:
    """A Requirement from a mapping as written in a requirements file or by the page; RequirementsError says what is
    wrong (an unknown metric, a comparison that is not one of OPS, a threshold that is not a number, a unit or study
    that is not the metric's)."""
    M.load_metrics()
    if isinstance(d, Requirement):
        d = d.to_dict()
    if not isinstance(d, dict):
        raise RequirementsError(f'{where} is not a mapping with metric, comparison and threshold.')
    mid = str(d.get('metric') or '').strip()
    if not mid:
        raise RequirementsError(f'{where} names no metric.')
    m = M.METRICS.get(mid)
    if m is None:
        raise RequirementsError(f'{where} names the metric "{mid}", which this build does not have (the metrics are '
                                'listed by dexvantage.core.requirements.catalogue()).')
    raw_op = d.get('op', d.get('comparison', '>=' if m.better == 'higher' else '<='))
    op = norm_op(raw_op)
    if op is None:
        raise RequirementsError(f'{where}: the comparison "{raw_op}" is not one of {", ".join(OPS)}.')
    t = _num(d.get('threshold', d.get('target')))
    if t is None:
        raise RequirementsError(f'{where}: the threshold {d.get("threshold", d.get("target"))!r} is not a number.')
    unit = norm_unit(d.get('unit'))
    if unit and unit != m.unit:
        raise RequirementsError(f'{where}: {m.title.lower()} is in {m.unit}, not {unit}: give the threshold in '
                                f'{m.unit}.')
    study = str(d.get('study') or '').strip()
    if study and study != m.study:
        raise RequirementsError(f'{where}: {mid} is measured by the {m.study} study, not {study}.')
    is_default = d.get('is_default', d.get('default', False))
    r = Requirement(name=str(d.get('name') or m.title)[:200], metric=mid, op=op, threshold=t, unit=m.unit,
                    study=m.study, is_default=bool(is_default), note=str(d.get('note') or '')[:500])
    if d.get('uid'):
        r.uid = str(d['uid'])
    return r


def parse(text: str, name: str = 'The requirements file') -> list:
    """Requirements from YAML (or JSON) text: a list of items, or {'requirements': [...]} with optional 'format'."""
    import yaml
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as e:
        mark = getattr(e, 'problem_mark', None)
        at = f' (line {mark.line + 1})' if mark is not None else ''
        raise RequirementsError(f'{name} is not valid YAML{at}: {getattr(e, "problem", None) or e}.') from None
    if isinstance(doc, dict):
        fmt = doc.get('format')
        if fmt not in (None, FORMAT):
            raise RequirementsError(f'{name} is a "{fmt}" file, not a DexVantage requirements file.')
        items = doc.get('requirements')
    else:
        items = doc
    if not isinstance(items, list):
        raise RequirementsError(f'{name} has no list of requirements (a "requirements:" list of items with metric, '
                                'comparison and threshold).')
    if not items:
        raise RequirementsError(f'{name} names no requirement.')
    return [clean(it, f'Requirement {i + 1}' + (f' ("{it.get("name")}")' if isinstance(it, dict) and it.get('name')
                                                 else '')) for i, it in enumerate(items)]


def load(path: str) -> list:
    """Requirements from a .yaml/.yml/.json file; 'defaults' (no such file) gives the labelled defaults."""
    if path == 'defaults' and not os.path.exists(path):
        return default_requirements()
    base = os.path.basename(path)
    try:
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
    except OSError as e:
        raise RequirementsError(f'{base} could not be read: {e.strerror or e}.') from None
    return parse(text, base)


def dumps(reqs) -> str:
    """The requirements as a YAML requirements file (the format parse() reads)."""
    import yaml
    items = []
    for r in reqs:
        r = r if isinstance(r, Requirement) else clean(r)
        it = {'name': r.name, 'metric': r.metric, 'comparison': r.op, 'threshold': r.threshold, 'unit': r.unit,
              'study': r.study, 'default': bool(r.is_default)}
        if r.note:
            it['note'] = r.note
        items.append(it)
    head = ('# DexVantage readiness scorecard: dexvantage PROJECT.dexv --check requirements.yaml\n'
            '# Each requirement: a metric (core/metric_defs.py), a comparison (>=, <=, <, >, ==), a threshold in the\n'
            '# metric\'s unit, and the study that measures it. "default: true" marks DexVantage\'s documented\n'
            '# defaults (starting points, not standards).\n')
    return head + yaml.safe_dump({'format': FORMAT, 'version': 1, 'requirements': items}, sort_keys=False,
                                 allow_unicode=True, width=200)


def describe(r: Requirement) -> str:
    m = M.METRICS.get(r.metric)
    unit = (r.unit or (m.unit if m else '')).replace('count', '').strip()
    return f'{r.name or r.metric} {OP_TEXT.get(r.op, r.op)} {r.threshold:g}{(" " + unit) if unit else ""}'


def catalogue() -> list:
    """Every metric a requirement may name: [{'id', 'title', 'unit', 'better', 'study', 'study_title', 'note'}]."""
    M.load_metrics()
    jobs.load_studies()
    return [{'id': m.id, 'title': m.title, 'unit': m.unit, 'better': m.better, 'study': m.study,
             'study_title': jobs.STUDIES[m.study].title if m.study in jobs.STUDIES else m.study, 'note': m.note,
             'headline': 'headline' in m.tags} for m in M.METRICS.values() if not m.id.startswith('_')]


# ====================================================================== comparing
def compare(value: float, op: str, threshold: float) -> tuple:
    """(passes, margin): the margin is signed so positive is good (ported from Vantage)."""
    eps = 1e-9 * max(1.0, abs(threshold))
    if op in ('>=', '>'):
        margin = value - threshold
        ok = margin >= -eps if op == '>=' else margin > eps
    elif op in ('<=', '<'):
        margin = threshold - value
        ok = margin >= -eps if op == '<=' else margin > eps
    else:
        margin = -abs(value - threshold)
        ok = margin >= -eps
    return bool(ok), float(margin)


def decide(value, bound, op: str, threshold: float) -> tuple:
    """(status, margin, reason) of one requirement. bound 'at_least': the true figure is at least value; 'at_most':
    at most value. A bound decides only when it settles the comparison."""
    if value is None:
        return 'not_evaluated', None, ''
    ok, margin = compare(value, op, threshold)
    if bound not in ('at_least', 'at_most'):
        return ('pass' if ok else 'fail'), margin, ''
    word = 'at least' if bound == 'at_least' else 'at most'
    up = bound == 'at_least'            # the true figure can only be larger than the value
    if op == '==':
        decisive_fail = (value > threshold) if up else (value < threshold)
        if not ok and decisive_fail:
            return 'fail', margin, ''
    elif (op in ('>=', '>')) == up:     # the bound goes the way that helps: a pass on the bound is a pass
        if ok:
            return 'pass', margin, ''
    elif not ok:                        # the bound goes the other way: a failure on the bound is a failure
        return 'fail', margin, ''
    return 'not_evaluated', None, (f'The study states only that the figure is {word} {_fmt(value)}, which does not '
                                   f'settle {OP_WORDS[op]} {_fmt(threshold)}: give the figures it lacks.')


def _fmt(v, unit=''):
    """A figure for the table and the texts: plain digits, then the unit (none for a count)."""
    if v is None:
        return '—'
    a = abs(v)
    if float(v).is_integer() or a >= 1000:
        s = f'{v:.0f}'
    elif a >= 100:
        s = f'{v:.1f}'
    elif a < 0.01:
        s = f'{v:.3g}'
    else:
        s = f'{v:.2f}'.rstrip('0').rstrip('.')
    if unit in ('', 'count'):
        return s
    return f'{s}{unit}' if unit == '°' else f'{s} {unit}'


# ====================================================================== the scorecard
def _with_variant(project, variant):
    p = project.snapshot()
    if variant is None:
        return p
    uid = variant.uid if hasattr(variant, 'uid') else str(variant)
    v = p.variant_by_uid(uid) or next((x for x in p.variants if x.name == uid), None)
    if v is None:
        raise RequirementsError(f'The project has no variant "{uid}".')
    p.active_uid = v.uid
    return p


def _sub(ctx, k: int, n: int, text: str):
    if ctx is None:
        return None
    lo, span = k / max(n, 1), 1.0 / max(n, 1)
    return jobs.Ctx(progress=lambda f, t='': ctx.report(lo + span * f, f'{text}: {t}' if t else text),
                    cancel=ctx.cancel)


def check(project, reqs=None, cache=None, ctx=None, variant=None) -> dict:
    """The scorecard of reqs (default: the project's own requirements) on the project (the active variant, or the
    variant named by uid or name). JSON-able; every study runs at most once (shared result cache)."""
    jobs.load_studies()
    M.load_metrics()
    cache = _CACHE if cache is None else cache
    raw = list(project.requirements if reqs is None else reqs)
    p = _with_variant(project, variant)
    items, errors = [], []
    for i, r in enumerate(raw):
        try:
            items.append(clean(r, f'Requirement {i + 1}'))
        except RequirementsError as e:
            r0 = r if isinstance(r, Requirement) else Requirement.from_dict(r if isinstance(r, dict) else {})
            items.append(r0)
            errors.append({'uid': r0.uid, 'message': str(e)})
    bad = {e['uid']: e['message'] for e in errors}
    by_study: dict = {}
    for r in items:
        if r.uid not in bad:
            by_study.setdefault(r.study, [])
            if r.metric not in by_study[r.study]:
                by_study[r.study].append(r.metric)
    values, studies = {}, {}
    for k, (sid, ids) in enumerate(by_study.items()):
        title = jobs.STUDIES[sid].title if sid in jobs.STUDIES else sid
        if ctx is not None and ctx.cancelled():
            studies[sid] = {'title': title, 'status': 'not run', 'reason': 'Cancelled.'}
            continue
        if ctx is not None:
            ctx.report(k / max(len(by_study), 1), title)
        t0 = time.perf_counter()
        try:
            rows = M.evaluate(p, ids, cache=cache, ctx=_sub(ctx, k, len(by_study), title))
        except Exception as e:                       # noqa: BLE001 — explained in the scorecard, the traceback logged
            traceback.print_exc()
            first = (str(e).splitlines() or [''])[0][:200]
            msg = f'{title} stopped with an unexpected error ({type(e).__name__}: {first}).'
            studies[sid] = {'title': title, 'status': 'error', 'reason': msg,
                            'seconds': round(time.perf_counter() - t0, 3)}
            errors.append({'study': sid, 'message': msg})
            continue
        values.update(rows)
        refused = next((x for x in rows.values() if x['status'] == 'refused'), None)
        studies[sid] = {'title': title, 'status': 'refused' if refused else 'ran',
                        'reason': refused['reason'] if refused else '', 'seconds': round(time.perf_counter() - t0, 3)}
    out_items = []
    for r in items:
        m = M.METRICS.get(r.metric)
        row = {'uid': r.uid, 'name': r.name, 'metric': r.metric, 'title': m.title if m else r.metric,
               'study': r.study, 'study_title': studies.get(r.study, {}).get('title', r.study), 'op': r.op,
               'op_text': OP_TEXT.get(r.op, r.op), 'threshold': r.threshold, 'unit': r.unit,
               'is_default': r.is_default, 'note': r.note, 'value': None, 'bound': None, 'margin': None,
               'status': 'not_evaluated', 'reason': '', 'figure_note': ''}
        if r.uid in bad:
            row['reason'] = bad[r.uid]
        else:
            ev = values.get(r.metric)
            st = studies.get(r.study, {})
            if ev is None:
                row['reason'] = st.get('reason') or 'The study did not run.'
            else:
                status, margin, why = decide(ev['value'], ev.get('bound'), r.op, r.threshold)
                row.update(value=ev['value'], bound=ev.get('bound'), margin=margin, status=status,
                           figure_note=ev.get('note') or '',
                           reason=why or (ev.get('reason') or ('The result does not state this figure.'
                                                               if ev['value'] is None else '')))
        row['status_text'] = STATUS_TEXT[row['status']]
        row['level'] = STATUS_LEVEL[row['status']]
        row['text'] = _item_text(row)
        out_items.append(row)
    verdict = verdict_of(out_items)
    counts = {s: sum(1 for x in out_items if x['status'] == s) for s in STATUSES}
    counts['total'] = len(out_items)
    v = p.variant
    return {'variant': v.name, 'variant_uid': v.uid, 'verdict': verdict, 'verdict_text': VERDICT_TEXT[verdict],
            'level': VERDICT_LEVEL[verdict], 'counts': counts, 'headline': _headline(verdict, counts, out_items),
            'items': out_items, 'studies': studies, 'errors': errors,
            'requirements': [r.to_dict() for r in items], 'assumptions': ASSUMPTIONS}


def _item_text(row) -> str:
    val = _fmt(row['value'], row['unit'])
    if row['bound']:
        val = ('≥ ' if row['bound'] == 'at_least' else '≤ ') + val
    thr = f'{row["op_text"]} {_fmt(row["threshold"], row["unit"])}'
    dflt = ' (default)' if row['is_default'] else ''
    if row['status'] == 'not_evaluated':
        return f'{row["name"]}: {thr}{dflt}: not evaluated. {row["reason"]}'.strip()
    return f'{row["name"]}: {val} against {thr}{dflt}: {row["status_text"]}.'


def verdict_of(items) -> str:
    if not items:
        return 'empty'
    if any(r['status'] == 'fail' for r in items):
        return 'fail'
    if any(r['status'] != 'pass' for r in items):
        return 'incomplete'
    return 'pass'


def _headline(verdict, counts, items) -> dict:
    n = counts['total']
    if verdict == 'empty':
        text = 'There are no requirements: add the defaults or your own on the Requirements page.'
    elif verdict == 'pass':
        text = f'All {n} requirement{"s" if n != 1 else ""} pass.'
    else:
        bits = []
        if counts['fail']:
            bits.append(f'{counts["fail"]} of {n} fail')
        if counts['not_evaluated']:
            bits.append(f'{counts["not_evaluated"]} not evaluated')
        first = next((r for r in items if r['status'] == 'fail'), None) or \
            next((r for r in items if r['status'] == 'not_evaluated'), None)
        text = '; '.join(bits) + '. ' + (first['text'] if first else '')
    return {'pass': counts['pass'], 'total': n, 'value': f'{counts["pass"]} / {n}', 'word': VERDICT_TEXT[verdict],
            'level': VERDICT_LEVEL[verdict], 'text': text.strip()}


def exit_code(card: dict) -> int:
    """The CI gate: 2 on an error, 0 when every requirement passes, else 1 (a failure or a figure not evaluated)."""
    if card.get('errors'):
        return EXIT_ERROR
    return EXIT_PASS if card.get('verdict') == 'pass' else EXIT_FAIL


def table(card: dict) -> str:
    """The scorecard as a plain-text table (batch --check prints it)."""
    rows = [('Requirement', 'Metric', 'Value', 'Threshold', 'Margin', 'Status')]
    for r in card['items']:
        val = _fmt(r['value'], r['unit'])
        if r['bound'] and r['value'] is not None:
            val = ('≥ ' if r['bound'] == 'at_least' else '≤ ') + val
        thr = f'{r["op_text"]} {_fmt(r["threshold"], r["unit"])}' + (' (default)' if r['is_default'] else '')
        mg = '—' if r['margin'] is None else ('+' if r['margin'] >= 0 else '−') + _fmt(abs(r['margin']), r['unit'])
        rows.append((r['name'][:44], r['metric'], val, thr, mg, r['status_text']))
    w = [max(len(x[i]) for x in rows) for i in range(len(rows[0]))]
    lines = ['  '.join(c.ljust(w[i]) for i, c in enumerate(x)).rstrip() for x in rows]
    lines.insert(1, '  '.join('-' * k for k in w))
    out = [f'Readiness scorecard · {card["variant"]}'] + lines
    for r in card['items']:
        if r['status'] == 'not_evaluated' and r['reason']:
            out.append(f'  {r["name"]}: {r["reason"]}')
    for e in card.get('errors') or []:
        out.append(f'  Error: {e["message"]}')
    c = card['counts']
    out.append(f'Verdict: {card["verdict_text"]} ({c["pass"]} pass, {c["fail"]} fail, {c["not_evaluated"]} not '
               'evaluated)')
    return '\n'.join(out)


# ====================================================================== variants compare
def compare_variants(project, reqs=None, cache=None, ctx=None) -> dict:
    """Every variant against every requirement: the value and status per cell, the best variant per requirement (by
    the metric's better direction; == by the smallest distance to the threshold; ties all named), and a ranking by
    passes (then fewest failures)."""
    cache = _CACHE if cache is None else cache
    reqs = list(project.requirements if reqs is None else reqs)
    variants = list(project.variants)
    cards = []
    for k, v in enumerate(variants):
        if ctx is not None and ctx.cancelled():
            break
        cards.append(check(project, reqs, cache, _sub(ctx, k, len(variants), v.name), variant=v.uid))
    first = cards[0] if cards else {'items': [], 'errors': []}
    req_rows = [{k: it[k] for k in ('uid', 'name', 'metric', 'title', 'study', 'op', 'op_text', 'threshold', 'unit',
                                    'is_default')} for it in first['items']]
    cells = [[{k: c['items'][j][k] for k in ('value', 'bound', 'status', 'margin', 'reason')}
              for j in range(len(req_rows))] for c in cards]
    best = {}
    for j, rq in enumerate(req_rows):
        cand = [(i, cells[i][j]['value']) for i in range(len(cards)) if cells[i][j]['value'] is not None]
        if not cand:
            best[rq['uid']] = []
            continue
        m = M.METRICS.get(rq['metric'])
        if rq['op'] == '==':
            key = [abs(v - rq['threshold']) for _, v in cand]
            target = min(key)
        else:
            key = [(-v if m is not None and m.better == 'higher' else v) for _, v in cand]
            target = min(key)
        tol = 1e-9 * max(1.0, abs(target))
        best[rq['uid']] = [cards[cand[i][0]]['variant_uid'] for i in range(len(cand)) if key[i] <= target + tol]
    rank = sorted(range(len(cards)), key=lambda i: (-cards[i]['counts']['pass'], cards[i]['counts']['fail'], i))
    return {'variants': [{'uid': c['variant_uid'], 'name': c['variant'], 'verdict': c['verdict'],
                          'verdict_text': c['verdict_text'], 'level': c['level'], 'counts': c['counts']}
                         for c in cards],
            'requirements': req_rows, 'cells': cells, 'best': best,
            'ranking': [cards[i]['variant_uid'] for i in rank],
            'errors': [dict(e, variant=c['variant']) for c in cards for e in c['errors']],
            'active_uid': project.active_uid, 'assumptions': ASSUMPTIONS}

