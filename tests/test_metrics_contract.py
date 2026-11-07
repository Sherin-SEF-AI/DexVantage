"""The metric contract (core/metrics.py) that Phase 9 and the research tools share."""
import pytest

from dexvantage.core import jobs, metrics as M
from dexvantage.core.project import Project


def test_evaluate_reads_a_registered_metric_and_states_refusals():
    @jobs.study('_metric_ok', 'ok')
    def _ok(project, ctx):
        return {'figure': 3.5}

    @jobs.study('_metric_no', 'no')
    def _no(project, ctx):
        raise jobs.StudyError('This study needs a robot.')

    M.metric('_t.ok', 'OK', 'mm', 'lower', study='_metric_ok')(lambda d: d['figure'])
    M.metric('_t.absent', 'Absent', 'mm', 'lower', study='_metric_ok')(lambda d: d.get('nope'))
    M.metric('_t.no', 'No', '%', 'higher', study='_metric_no')(lambda d: d['x'])
    try:
        out = M.evaluate(Project(), ['_t.ok', '_t.absent', '_t.no'])
        assert out['_t.ok']['value'] == 3.5 and out['_t.ok']['status'] == 'ok'
        assert out['_t.absent']['value'] is None and out['_t.absent']['status'] == 'not stated'
        assert out['_t.no']['status'] == 'refused' and 'robot' in out['_t.no']['reason']
        assert M.better_of('_t.ok', 1.0, 2.0) and not M.better_of('_t.no', 1.0, 2.0)
        with pytest.raises(jobs.StudyError):
            M.get('_t.missing')
        with pytest.raises(ValueError):
            M.metric('_t.bad', 'Bad', '', 'sideways', study='_metric_ok')
    finally:
        for k in ('_metric_ok', '_metric_no'):
            jobs.STUDIES.pop(k, None)
        for k in ('_t.ok', '_t.absent', '_t.no'):
            M.METRICS.pop(k, None)
