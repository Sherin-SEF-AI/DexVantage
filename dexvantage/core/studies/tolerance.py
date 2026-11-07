"""Monte Carlo tolerances (spec §7.11): every sensor's mount tolerance (or calibration covariance) drawn n times and
carried through the requirement metrics (core/metrics.py) into distributions; the model is core/tolerance.py
(written up in docs/models/tolerance.md).

    @jobs.study('tolerance', 'Monte Carlo tolerances', quick=False, batch='tolerance-mc')   -> --tolerance-mc OUT.json

(The batch flag is --tolerance-mc because --tolerance is --reproduce's comparison tolerance.)

Settings, project.analysis.studies['tolerance'] (every number a labelled default unless the user sets it):

    n             draws (default 64)
    seed          their seed (default 0)
    scale         a factor on every spread (default 1: the tolerances as stated)
    distribution  'uniform' (default) or 'normal' (the ± bound at 2 σ, truncated there) inside a mount tolerance
    metrics       the metric ids to carry (default: those the project's requirements name; with none,
                  DEFAULT_METRICS: the headline figures of coverage, depth validity, motion blur and VLA tokens)
    sensors       sensor uids or names to perturb (default: every enabled sensor)

The result: per metric the nominal figure, mean ± σ, P5 / P50 / P95, the share of draws meeting each requirement
(Wilson 95 % interval), the contribution of each mount (Spearman rank correlation, core/tolerance.py) and the values
of every draw; the draws themselves; and the headline: the requirement met by the smallest share of draws, as its
pessimistic 5 % figure (P5 of a higher-is-better metric, P95 of a lower-is-better one), with a status word from that
share (holds from 95 %, marginal from 80 %, often fails from 50 %, else fails: DexVantage defaults). The suggestion
(the page's Apply) halves the mount tolerance (position or rotation, whichever contributes more) of the unlocked
mount that contributes most to the headline metric, when its contribution is above the noise level.

Known limits: as core/tolerance.py: independent components, no shared-bracket correlation, the calibration
covariance's own optimism, and n bounded by the slowest metric's study (each draw re-runs every study the metrics
need). The suggestion's effect is not predicted: run the study again after Apply. A metric that gives no value for
some draws (its study refuses on the moved mount) is counted on the draws that gave one, and the count is shown.
"""
from __future__ import annotations

import math

from .. import jobs
from .. import tolerance as TL

SID = 'tolerance'
DEFAULTS = {'n': 64, 'seed': 0, 'scale': 1.0, 'distribution': 'uniform'}
MAX_N = 5000
STATUS_SHARES = ((0.95, 'good', 'holds'), (0.80, 'warning', 'marginal'), (0.50, 'serious', 'often fails'),
                 (0.0, 'critical', 'fails'))           # DexVantage defaults: the share of draws meeting a requirement
TIGHTEN = 0.5                                          # the suggestion halves a tolerance (a default)
# without requirements: the headline figures of the studies that depend on the mounts and take under a second on the
# samples (measured 2026-11-07 on the UR5e sample: coverage 0.8 s, depth 0.3 s, motion blur 0.2 s, VLA tokens 0.1 s;
# grasp visibility 30 s and the dexterous hand 4 s per draw are left to a requirement or the settings). A default.
DEFAULT_METRICS = ('coverage.covered', 'depth.valid', 'motion_blur.worst_px', 'vla.min_tokens')


def settings_of(project) -> dict:
    given = dict(project.analysis.studies.get(SID) or {})
    out = dict(DEFAULTS)
    used_default = []
    for k in ('n', 'seed', 'scale'):
        g = given.get(k)
        if g is None:
            used_default.append(k)
            continue
        try:
            x = float(g)
        except (TypeError, ValueError):
            raise jobs.StudyError(f'The tolerance setting {k} must be a number; it is {g!r}.') from None
        if not math.isfinite(x) or x < 0:
            raise jobs.StudyError(f'The tolerance setting {k} must be a positive number; it is {g!r}.')
        out[k] = int(round(x)) if k in ('n', 'seed') else x
    if out['n'] < 2 or out['n'] > MAX_N:
        raise jobs.StudyError(f'The tolerance study takes 2 to {MAX_N} draws; it is set to {out["n"]}.')
    dist = given.get('distribution')
    if dist is None:
        used_default.append('distribution')
    elif dist not in TL.DISTRIBUTIONS:
        raise jobs.StudyError(f'The tolerance distribution must be uniform or normal; it is {dist!r}.')
    else:
        out['distribution'] = dist
    out['metrics'] = [str(m) for m in given.get('metrics') or [] if m]
    out['sensors'] = [str(s) for s in given.get('sensors') or [] if s]
    out['defaults'] = used_default
    return out


def metric_ids(project, st: dict) -> tuple[list, str]:
    """(ids, where they came from): the settings', else the requirements', else every quick study's metrics."""
    from .. import metrics as M
    M.load_metrics()
    if st['metrics']:
        unknown = [m for m in st['metrics'] if m not in M.METRICS]
        if unknown:
            raise jobs.StudyError(f'There is no metric called {", ".join(unknown)}.')
        ids, why = st['metrics'], 'the study\'s settings'
    else:
        req = [r.metric for r in project.requirements if r.metric]
        known = [m for m in dict.fromkeys(req) if m in M.METRICS]
        if known:
            ids, why = known, 'the project\'s requirements'
        else:
            ids = [m for m in DEFAULT_METRICS if m in M.METRICS]
            why = ('the headline figures of the fast geometric studies, a default (the project has no requirement '
                   'with a known metric)')
            if not ids:
                jobs.load_studies()
                ids = [m.id for m in M.METRICS.values() if m.study in jobs.STUDIES and jobs.STUDIES[m.study].quick]
                why = 'every metric of a quick study (the project has no requirement with a known metric)'
    ids = [m for m in ids if M.METRICS[m].study != SID]
    if not ids:
        raise jobs.StudyError('There is no figure to carry through the tolerances: add a requirement (its metric is '
                              'the figure) or name metrics in the tolerance study\'s settings.')
    return ids, why


def status_of(share) -> tuple[str, str]:
    if share is None:
        return 'warning', 'no value'
    for lo, sev, word in STATUS_SHARES:
        if share >= lo:
            return sev, word
    return 'critical', 'fails'


def _fmt(v, unit=''):
    if v is None:
        return '—'
    a = abs(v)
    s = f'{v:.3g}' if a >= 100 or a == 0 else f'{v:.3f}' if a < 1 else f'{v:.2f}'
    return f'{s}{unit}' if unit == '°' else f'{s} {unit}'.strip()


def _margin(m: dict, r: dict) -> float:
    """How far the pessimistic 5 % figure is on the good side of the threshold, in σ (ties of the share: the
    smallest margin is the worst requirement)."""
    v, sd = m['pessimistic']['value'], m.get('std') or 0.0
    if v is None:
        return -math.inf
    d = (v - r['threshold']) if r['op'] in ('>=', '>') else (r['threshold'] - v) if r['op'] in ('<=', '<') \
        else -abs(v - r['threshold'])
    return d / sd if sd > 0 else (math.inf if d > 0 else -math.inf if d < 0 else 0.0)


def headline(doc: dict) -> dict:
    """The requirement met by the fewest draws, as its pessimistic 5 % figure, with its status; without
    requirements, the metric with the widest relative spread (no threshold to judge it by)."""
    rows, unanswered = [], []
    for mid, m in doc['metrics'].items():
        for r in m['requirements']:
            if r['share'] is not None:
                rows.append((r['share'], _margin(m, r), mid, r))
            else:
                unanswered.append((mid, r))
    if rows:
        share, _z, mid, r = min(rows, key=lambda x: (x[0], x[1], x[2]))
        m = doc['metrics'][mid]
        sev, word = status_of(share)
        pq = m['pessimistic']
        text = (f'{r["name"]} ({r["op"]} {_fmt(r["threshold"], m["unit"])}'
                f'{", a default" if r["is_default"] else ""}): {100 * share:.0f} % of {r["of"]} draws meet it '
                f'[{100 * r["share_ci95"][0]:.0f}–{100 * r["share_ci95"][1]:.0f} %, Wilson 95 %]; '
                f'{pq["quantile"]} {_fmt(pq["value"], m["unit"])}, nominal {_fmt(m["nominal"], m["unit"])}.')
        return {'value': pq['value'], 'unit': m['unit'], 'quantile': pq['quantile'], 'metric': mid,
                'requirement': r['uid'], 'share': share, 'status': sev, 'word': word, 'text': text}
    if unanswered:
        mid, r = unanswered[0]
        m = doc['metrics'][mid]
        return {'value': None, 'unit': m['unit'], 'quantile': m['pessimistic']['quantile'], 'metric': mid,
                'requirement': r['uid'], 'share': None, 'status': 'critical', 'word': 'no value',
                'text': f'{r["name"]}: no draw gave a value' + (f' ({m["reason"]})' if m.get('reason') else '.')}
    best = None
    for mid, m in doc['metrics'].items():
        if m.get('std') is None:
            continue
        rel = m['std'] / max(abs(m['mean']), 1e-12)
        if best is None or rel > best[0]:
            best = (rel, mid)
    if best is None:
        reasons = [m.get('reason') for m in doc['metrics'].values() if m.get('reason')]
        return {'value': None, 'unit': '', 'quantile': '', 'metric': None, 'requirement': None, 'share': None,
                'status': 'critical', 'word': 'no value',
                'text': 'No figure gave a value: ' + (reasons[0] if reasons else 'the studies stated none.')}
    mid = best[1]
    m = doc['metrics'][mid]
    pq = m['pessimistic']
    return {'value': pq['value'], 'unit': m['unit'], 'quantile': pq['quantile'], 'metric': mid, 'requirement': None,
            'share': None, 'status': 'warning', 'word': 'no requirement',
            'text': (f'{m["title"]}: {_fmt(m["mean"], m["unit"])} ± {_fmt(m["std"], m["unit"])} (σ over '
                     f'{m["valid"]} draws), {pq["quantile"]} {_fmt(pq["value"], m["unit"])}. No requirement names '
                     'it: add one to judge it.')}


def suggestion(project, doc: dict, head: dict) -> dict | None:
    """Halve the position or rotation tolerance of the unlocked mount (not a calibrated one) that contributes most to
    the headline metric, when its contribution is above the noise."""
    mid = head.get('metric')
    if not mid:
        return None
    for c in doc['metrics'][mid]['contributions']:
        if c['source'] != TL.MOUNT or not c['significant']:
            continue
        if c['locked']:
            continue
        s = project.variant.sensor(c['sensor_uid'])
        if s is None:
            continue
        fld = 'position_mm' if c['position_rho2'] >= c['rotation_rho2'] else 'rotation_deg'
        cur = float(getattr(s.mount.tolerance, fld))
        if cur <= 0:
            continue
        unit = ' mm' if fld == 'position_mm' else '°'
        return {'sensor_uid': s.uid, 'name': s.name, 'field': fld, 'from': cur, 'to': cur * TIGHTEN,
                'share': c['share'], 'metric': mid,
                'text': (f'Tighten {s.name}\'s {"position" if fld == "position_mm" else "rotation"} tolerance from '
                         f'± {cur:g}{unit} to ± {cur * TIGHTEN:g}{unit}: it carries {100 * c["share"]:.0f} % of the '
                         f'spread of {doc["metrics"][mid]["title"].lower()} (rank correlation). Run again to see the '
                         'effect.')}
    return None


@jobs.study(SID, 'Monte Carlo tolerances', quick=False, batch='tolerance-mc', needs_robot=False)
def run(project, ctx: jobs.Ctx) -> dict:
    st = settings_of(project)
    ids, why = metric_ids(project, st)
    doc = TL.monte_carlo(project, ids, st['n'], st['seed'], st['scale'], st['distribution'], ctx=ctx,
                         sensors=st['sensors'] or None)
    head = headline(doc)
    notes = [f'The figures carried are those of {why}.',
             'Mount tolerances are ± bounds per axis (' + ('uniform within them' if st['distribution'] == 'uniform'
                                                          else 'normal with the bound at 2 σ, truncated there') +
             '); a calibrated sensor draws from its hand-eye covariance instead.',
             f'A rank correlation below {doc["noise_rho"]:.2f} (2 / √(n − 1)) is not distinguishable from noise.']
    for s in doc['sources']:
        if s.get('note'):
            notes.append(f'{s["name"]}: {s["note"]}.')
    for mid, m in doc['metrics'].items():
        if m['no_value']:
            notes.append(f'{m["title"]}: {m["no_value"]} of {doc["done"]} draws gave no value'
                         + (f' ({m["reason"]})' if m.get('reason') else '') + '.')
    if doc['cancelled']:
        notes.insert(0, f'Stopped after {doc["done"]} of {doc["n"]} draws: the figures cover those.')
    sg = suggestion(project, doc, head)
    doc.update(headline=head, suggestion=sg, notes=notes, metric_source=why,
               settings={k: st[k] for k in ('n', 'seed', 'scale', 'distribution', 'metrics', 'sensors')},
               defaults=st['defaults'],
               status_rule={'shares': [[lo, sev, word] for lo, sev, word in STATUS_SHARES], 'default': True},
               advice=sg['text'] if sg else ('No unlocked mount stands out above the noise: more draws (n) would '
                                             'resolve smaller contributions.' if head.get('metric') else ''))
    ctx.report(1.0, head['text'])
    return doc
