"""The optimiser's objectives and constraints, as core.metrics figures of merit.

The objectives are metric ids of core/metric_defs.py (the vocabulary the requirements scorecard, the tolerance Monte
Carlo and the experiments share); the defaults are spec §7.11's list:

    grasp.contact_visibility   %        higher   the Contact Visibility Score (grasp study)
    depth.valid                %        higher   the depth-valid share of the task volume (depth validity)
    rig.camera_count           count    lower    enabled camera and depth sensors (rig inventory)
    rig.cost                   USD      lower    the stated device prices added up (rig inventory; not stated unless
                                                 every device's library entry states a price)
    bandwidth.margin           Mbit/s   higher   the smallest controller margin (bandwidth)
    brackets.not_ready         count    lower    sensors without a ready bracket (brackets: bracket printability)
    vla.min_tokens             tokens   higher   the fewest ViT tokens on a critical object (VLA tokens)

Any other registered metric can be named in the settings. Constraints come from the project's requirements
(project.requirements: metric, op, threshold) and the settings' own 'constraints': constraint_rows() reads them and
violation() turns each into a normalised violation; a requirement whose metric is unknown is listed, not used.

Known limits: a metric is the study's own figure at the settings the optimiser runs it with (the screening settings
reduce the grasp study's sample, said in the result); a figure a study does not state for a design is not a number:
the optimiser ranks that design below every design that states it, and counts a requirement it cannot show to hold
as missed (a figure not stated, or stated only as a bound that does not settle the test), rather than guess the
figure.
"""
from __future__ import annotations

import math

from .. import metrics as M

DEFAULT_OBJECTIVES = ('grasp.contact_visibility', 'depth.valid', 'rig.camera_count', 'rig.cost', 'bandwidth.margin',
                      'brackets.not_ready', 'vla.min_tokens')
COST = 'rig.cost'
CAMERAS = 'rig.camera_count'
# the relative scale of a constraint's violation: |threshold|, or this floor for a threshold near zero
UNIT_FLOOR = {'%': 1.0, 'mm': 1.0, 'm': 0.01, 'ms': 0.1, 's': 0.001, 'px': 0.1, 'px/mm': 0.1, '°': 0.1,
              'count': 1.0, 'tokens': 1.0, 'W': 1.0, 'g': 1.0, 'Mbit/s': 1.0, 'USD': 1.0, 'ratio': 0.01}
OPS = ('>=', '>', '<=', '<', '==')
_DONE = False


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def ensure_metrics():
    """Load the metric definitions and the studies they read (once)."""
    global _DONE
    M.load_metrics()
    if _DONE:
        return
    from .. import jobs
    jobs.load_studies()
    _DONE = True


def available() -> list:
    """[{'id', 'title', 'unit', 'better', 'study'}] of every registered metric, the defaults first."""
    ensure_metrics()
    ids = [k for k in DEFAULT_OBJECTIVES if k in M.METRICS]
    ids += sorted(k for k in M.METRICS if k not in ids and not k.startswith('_'))
    return [{'id': k, 'title': M.METRICS[k].title, 'unit': M.METRICS[k].unit, 'better': M.METRICS[k].better,
             'study': M.METRICS[k].study} for k in ids]


def minimised(mid: str, value):
    """The value as a figure to minimise (higher-is-better metrics negated); None stays None."""
    if value is None:
        return None
    return -float(value) if M.get(mid).better == 'higher' else float(value)


def price_stated(model_id: str) -> bool:
    """True when the device's library entry states a price (the rig inventory's rule, core/studies/rig_inventory)."""
    from .. import library as LB
    from ..studies import rig_inventory as RI
    if not model_id:
        return False
    return RI.stated_price(LB.get(model_id))[0] is not None


# ====================================================================== constraints
def constraint_rows(project, extra=()) -> tuple[list, list]:
    """([{'name', 'metric', 'op', 'threshold', 'unit', 'is_default', 'source'}], [skipped: {'name', 'why'}]) from the
    project's requirements and the extra constraints given (dicts with the same keys)."""
    ensure_metrics()
    rows, skipped = [], []
    items = [dict(r.to_dict(), source='requirement') for r in getattr(project, 'requirements', None) or []]
    items += [dict(c, source='setting') for c in extra or () if isinstance(c, dict)]
    for it in items:
        name = str(it.get('name') or it.get('metric') or 'requirement')
        mid = str(it.get('metric') or '')
        op = str(it.get('op') or '>=')
        thr = _num(it.get('threshold'))
        if mid not in M.METRICS:
            skipped.append({'name': name, 'why': f'there is no metric called {mid or "(none)"}, so nothing measures '
                                                 'it here'})
            continue
        if op not in OPS or thr is None:
            skipped.append({'name': name, 'why': f'its test "{op} {it.get("threshold")}" is not a comparison with a '
                                                 'number'})
            continue
        rows.append({'name': name, 'metric': mid, 'op': op, 'threshold': thr,
                     'unit': str(it.get('unit') or M.METRICS[mid].unit),
                     'is_default': bool(it.get('is_default', False)), 'source': it['source']})
    return rows, skipped


def violation(row: dict, value, bound=None) -> float:
    """The constraint's violation: 0 when met, else the shortfall relative to max(|threshold|, the unit's floor) (so
    constraints in different units weigh alike); 1 when the figure is not stated (it cannot be shown to hold), and 1
    when the study states the figure only as a bound ('at_least' / 'at_most') that does not settle the test (the
    requirements scorecard's "not evaluated", core/requirements.decide). A strict test (<, >) that is missed exactly at
    the threshold counts a shortfall of 10⁻⁶ of the scale."""
    if value is None:
        return 1.0
    if bound in ('at_least', 'at_most'):
        from ..requirements import decide
        if decide(float(value), bound, row['op'], float(row['threshold']))[0] == 'not_evaluated':
            return 1.0
    v, t, op = float(value), float(row['threshold']), row['op']
    scale = max(abs(t), UNIT_FLOOR.get(row.get('unit', ''), 1.0), 1e-9)
    if op == '>=':
        short = t - v
    elif op == '>':
        short = max(t - v, 1e-6 * scale) if v <= t else 0.0
    elif op == '<=':
        short = v - t
    elif op == '<':
        short = max(v - t, 1e-6 * scale) if v >= t else 0.0
    else:
        short = abs(v - t) if abs(v - t) > 1e-9 * scale else 0.0
    return max(0.0, short) / scale


def holds(row: dict, value, bound=None) -> bool:
    return value is not None and violation(row, value, bound) <= 0.0
