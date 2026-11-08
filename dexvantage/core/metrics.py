"""Named figures of merit read from study results: the one vocabulary the tolerance Monte Carlo, designed
experiments, the optimiser, the design agent, requirements (--check), the experiment tracker and the Python API share.

A metric is a number one study's result yields, with a unit and the direction that is better:

    @metric('grasp.contact_visibility', 'Contact visibility', '%', 'higher', study='grasp_visibility')
    def _(doc):  return doc['headline']['value']          # None when the result does not state it

    evaluate(project, ['grasp.contact_visibility', 'bandwidth.margin'])
      -> {'grasp.contact_visibility': {'value': 97.2, 'unit': '%', 'better': 'higher', 'study': 'grasp_visibility',
                                       'title': ..., 'status': 'ok'},
          'bandwidth.margin':         {'value': None, ..., 'status': 'refused', 'reason': 'plain sentence'}}

Each study is run at most once per evaluate() call (through core.jobs.run_study and its result cache). A study that
refuses (StudyError) gives value None with status 'refused' and its sentence; a figure the result does not state
gives value None with status 'not stated'. Never a guessed number.

Metric definitions for the studies live in core/metric_defs.py (loaded by load_metrics()); tests may register their
own with @metric.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:                                    # names for the type hints only
    from .jobs import Ctx, ResultCache
    from .project import Project

BETTER = ('higher', 'lower')


@dataclass
class Metric:
    id: str
    title: str
    unit: str
    better: str                         # 'higher' | 'lower'
    study: str                          # the core.jobs study id that yields it
    extract: Callable                   # (result doc) -> float | None
    note: str = ''
    tags: list = field(default_factory=list)


METRICS: dict = {}
_LOADED = False


def metric(mid: str, title: str, unit: str, better: str, study: str, note: str = '',
           tags: tuple | list = ()) -> Callable[[Callable], Callable]:
    """Register a metric; the decorated function reads the value from the study's result doc (None when absent)."""
    if better not in BETTER:
        raise ValueError(f'better must be one of {BETTER}')

    def deco(fn: Callable) -> Callable:
        METRICS[mid] = Metric(mid, title, unit, better, study, fn, note, list(tags))
        return fn
    return deco


def load_metrics() -> None:
    """Import core/metric_defs.py once (it registers the metrics of every study)."""
    global _LOADED
    if _LOADED:
        return
    try:
        from . import metric_defs  # noqa: F401
    except ImportError:
        pass
    _LOADED = True


def get(mid: str) -> Metric:
    load_metrics()
    if mid not in METRICS:
        from .jobs import StudyError
        raise StudyError(f'There is no metric called {mid}.')
    return METRICS[mid]


def evaluate(project: Project, ids: list | tuple, cache: ResultCache | None = None, ctx: Ctx | None = None) -> dict:
    """{metric id: {'value', 'unit', 'better', 'study', 'title', 'status', 'reason'?, 'bound'?, 'note'?}} for the
    project (read() says what bound and note mean)."""
    from .jobs import StudyError, run_study
    load_metrics()
    docs: dict = {}
    out: dict = {}
    for mid in ids:
        m = get(mid)
        row = {'value': None, 'unit': m.unit, 'better': m.better, 'study': m.study, 'title': m.title}
        if m.study not in docs:
            try:
                docs[m.study] = ('ok', run_study(m.study, project, ctx=ctx, cache=cache))
            except StudyError as e:
                docs[m.study] = ('refused', str(e))
        kind, doc = docs[m.study]
        if kind == 'refused':
            row.update(status='refused', reason=doc)
        else:
            row.update(read(m, doc))
        out[mid] = row
    return out


def read(m: Metric, doc: dict) -> dict:
    """{'value', 'status', 'bound'?, 'reason'?, 'note'?} of metric m on a study's result doc. An extractor returns a
    number, None, or {'value', 'bound': 'at_least' | 'at_most' (the true figure is at least / at most the value: the
    study states a bound), 'reason' (why it is not stated), 'note'}; a non-finite number or a result shaped otherwise
    than the extractor expects reads 'not stated', never a guess."""
    import math
    extra: dict = {}
    try:
        v = m.extract(doc)
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return {'value': None, 'status': 'not stated',
                'reason': f'The {m.study} result does not state {m.title.lower()}.'}
    if isinstance(v, dict):
        extra = {k: v[k] for k in ('bound', 'reason', 'note') if v.get(k)}
        v = v.get('value')
    try:
        v = None if v is None or isinstance(v, bool) else float(v)
    except (TypeError, ValueError):
        v = None
    if v is not None and not math.isfinite(v):
        v = None
    if v is None:
        extra.pop('bound', None)
    return {'value': v, 'status': 'ok' if v is not None else 'not stated', **extra}


def better_of(mid: str, a: float, b: float) -> bool:
    """True when a is better than b for this metric."""
    m = get(mid)
    return a > b if m.better == 'higher' else a < b


def delta_unit(unit: str | None) -> str:
    """The unit of a difference of two figures given in unit. A difference of two percentages is in percentage
    points: 94.52 % against 94.32 % is a change of −0.20 pp (a change of −0.21 % would be relative). Every other unit
    is its own difference's unit ('' stays '')."""
    u = (unit or '').strip()
    return 'pp' if u == '%' else u
