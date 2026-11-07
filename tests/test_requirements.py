"""Requirements and the readiness scorecard (core/requirements.py, ported from Vantage core/requirements.py and its
tests): items and files, the signed margin, bounds, the verdict, the table, the variants compare, and the studies."""
import os

import pytest

from dexvantage.core import jobs
from dexvantage.core import metrics as M
from dexvantage.core import requirements as RQ
from dexvantage.core.project import Project, Requirement, Sensor, Variant

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


@pytest.fixture
def fake():
    """Two cheap studies and their metrics: one reads the variant's sensor count, one refuses."""
    calls = []

    @jobs.study('_rq_count', 'Sensor count')
    def _count(project, ctx):
        calls.append(project.variant.name)
        n = len(project.variant.sensors)
        return {'n': n, 'skew': 1.5 + n, 'complete': n < 2}

    @jobs.study('_rq_no', 'Refuser')
    def _no(project, ctx):
        raise jobs.StudyError('This study needs a robot.')

    M.metric('_rq.n', 'Sensors', 'count', 'lower', study='_rq_count')(lambda d: d['n'])
    M.metric('_rq.skew', 'Skew', 'ms', 'lower', study='_rq_count')(
        lambda d: d['skew'] if d['complete'] else {'value': d['skew'], 'bound': 'at_least'})
    M.metric('_rq.no', 'Refused', '%', 'higher', study='_rq_no')(lambda d: d['x'])
    try:
        yield calls
    finally:
        for k in ('_rq_count', '_rq_no'):
            jobs.STUDIES.pop(k, None)
        for k in ('_rq.n', '_rq.skew', '_rq.no'):
            M.METRICS.pop(k, None)


def _project(n=1, variants=1):
    p = Project('Req cell')
    p.variants = [Variant(name=f'V{i + 1}') for i in range(variants)]
    p.active_uid = p.variants[0].uid
    for i, v in enumerate(p.variants):
        for k in range(n + i):
            v.sensors.append(Sensor(name=f'S{k}', kind='camera', slot=k))
    return p


# ====================================================================== comparing (ported from Vantage's tests)
@pytest.mark.parametrize('value, op, target, ok, margin', [
    (96, '>=', 95, True, 1), (95, '>=', 95, True, 0), (94, '>=', 95, False, -1), (95, '>', 95, False, 0),
    (1.5, '<', 2, True, 0.5), (2, '<', 2, False, 0), (2, '<=', 2, True, 0), (3, '<=', 2, False, -1),
    (2, '==', 2, True, 0), (2.5, '==', 2, False, -0.5)])
def test_compare_margin_is_positive_when_good(value, op, target, ok, margin):
    assert RQ.compare(value, op, target) == (ok, pytest.approx(margin))


def test_compare_tolerates_rounding_at_the_threshold():
    assert RQ.compare(95.0 - 1e-12, '>=', 95.0)[0] and not RQ.compare(95.0 + 1e-12, '>', 95.0 + 1e-12)[0]


@pytest.mark.parametrize('value, bound, op, t, status', [
    (54.4, 'at_least', '<', 2, 'fail'),           # at least 54 ms cannot be under 2 ms
    (1.0, 'at_least', '<', 2, 'not_evaluated'),   # at least 1 ms might be over 2 ms
    (96, 'at_least', '>=', 95, 'pass'), (90, 'at_least', '>=', 95, 'not_evaluated'),
    (38, 'at_most', '<=', 2, 'not_evaluated'), (1, 'at_most', '<=', 2, 'pass'),
    (90, 'at_most', '>=', 95, 'fail'), (96, 'at_most', '>=', 95, 'not_evaluated'),
    (3, 'at_least', '==', 2, 'fail'), (2, 'at_least', '==', 2, 'not_evaluated'), (1, 'at_most', '==', 2, 'fail'),
    (None, None, '>=', 1, 'not_evaluated')])
def test_a_bound_decides_only_in_its_direction(value, bound, op, t, status):
    st, _m, why = RQ.decide(value, bound, op, t)
    assert st == status
    assert bool(why) == (status == 'not_evaluated' and value is not None)


# ====================================================================== items and files
def test_defaults_are_the_spec_list_and_labelled():
    ds = RQ.default_requirements()
    got = {(r.metric, r.op, r.threshold) for r in ds}
    assert got == {('grasp.contact_visibility', '>=', 95.0), ('sync.skew_ms', '<', 2.0),
                   ('bandwidth.over_budget', '<=', 0.0), ('cables.pull_snag', '<=', 0.0),
                   ('dexhand.blind_gap_mm', '<=', 0.0)}
    assert all(r.is_default and 'default' in r.note and 'not a standard' in r.note for r in ds)
    assert all(r.study == M.get(r.metric).study and r.unit == M.get(r.metric).unit for r in ds)


def test_example_file_is_the_defaults():
    reqs = RQ.load(os.path.join(ROOT, 'docs', 'examples', 'requirements.yaml'))
    assert [(r.metric, r.op, r.threshold, r.is_default) for r in reqs] == \
        [(r.metric, r.op, r.threshold, r.is_default) for r in RQ.default_requirements()]


def test_yaml_round_trip_and_aliases(tmp_path):
    text = RQ.dumps(RQ.default_requirements())
    assert text.startswith('# DexVantage readiness scorecard') and 'default: true' in text
    back = RQ.parse(text)
    assert [(r.metric, r.op, r.threshold) for r in back] == [(r.metric, r.op, r.threshold)
                                                            for r in RQ.default_requirements()]
    r = RQ.parse('- {metric: sync.skew_ms, comparison: less than, threshold: 1.5, unit: milliseconds}')[0]
    assert (r.op, r.threshold, r.unit, r.study, r.is_default) == ('<', 1.5, 'ms', 'time_sync', False)
    assert RQ.parse('[{"metric": "coverage.covered", "op": "≥", "threshold": 90}]')[0].op == '>='
    assert RQ.load('defaults')[0].metric == 'grasp.contact_visibility'


@pytest.mark.parametrize('text, words', [
    ('requirements: [{metric: nope.x, threshold: 1}]', 'does not have'),
    ('- {metric: sync.skew_ms, comparison: sideways, threshold: 1}', 'not one of'),
    ('- {metric: sync.skew_ms, threshold: fast}', 'not a number'),
    ('- {metric: sync.skew_ms, threshold: 1, unit: s}', 'in ms, not s'),
    ('- {metric: sync.skew_ms, threshold: 1, study: coverage}', 'measured by the time_sync study'),
    ('- {threshold: 1}', 'names no metric'),
    ('requirements: []', 'names no requirement'),
    ('format: something.else\nrequirements: [{metric: sync.skew_ms, threshold: 1}]', 'not a DexVantage'),
    ('a: [1, 2', 'not valid YAML'),
    ('just text', 'no list of requirements')])
def test_bad_files_are_explained(text, words):
    with pytest.raises(RQ.RequirementsError, match=words):
        RQ.parse(text, 'req.yaml')


def test_missing_file_is_a_sentence(tmp_path):
    with pytest.raises(RQ.RequirementsError, match='could not be read'):
        RQ.load(str(tmp_path / 'none.yaml'))


def test_catalogue_lists_every_metric_with_its_study():
    cat = {c['id']: c for c in RQ.catalogue()}
    assert cat['grasp.contact_visibility']['study_title'] == 'Grasp self-occlusion'
    assert cat['rig.cost']['unit'] == 'USD' and cat['sync.skew_ms']['better'] == 'lower'


# ====================================================================== the scorecard
def test_scorecard_pass_fail_not_evaluated_and_one_run_per_study(fake):
    p = _project(n=1)
    reqs = [RQ.clean({'metric': '_rq.n', 'op': '<=', 'threshold': 2}),
            RQ.clean({'metric': '_rq.skew', 'op': '<', 'threshold': 2}),
            RQ.clean({'metric': '_rq.no', 'op': '>=', 'threshold': 50})]
    card = RQ.check(p, reqs, cache=jobs.ResultCache())
    st = [i['status'] for i in card['items']]
    assert st == ['pass', 'fail', 'not_evaluated'] and card['verdict'] == 'fail' and len(fake) == 1
    assert card['items'][0]['margin'] == pytest.approx(1) and card['items'][1]['margin'] == pytest.approx(-0.5)
    assert 'needs a robot' in card['items'][2]['reason'] and card['studies']['_rq_no']['status'] == 'refused'
    assert RQ.exit_code(card) == 1 and card['level'] == 'critical' and card['verdict_text'] == 'not ready'
    assert card['headline']['value'] == '1 / 3' and '1 of 3 fail' in card['headline']['text']
    import json
    json.dumps(card, allow_nan=False)


def test_bound_on_the_scorecard(fake):
    p = _project(n=2)                                  # complete False: the skew 3.5 is "at least"
    card = RQ.check(p, [RQ.clean({'metric': '_rq.skew', 'op': '<', 'threshold': 5})], cache=jobs.ResultCache())
    it = card['items'][0]
    assert it['status'] == 'not_evaluated' and it['bound'] == 'at_least' and 'at least 3.5' in it['reason']
    assert card['verdict'] == 'incomplete' and RQ.exit_code(card) == 1
    card = RQ.check(p, [RQ.clean({'metric': '_rq.skew', 'op': '<', 'threshold': 2})], cache=jobs.ResultCache())
    assert card['items'][0]['status'] == 'fail'
    assert '≥ 3.50 ms' in RQ.table(card) or '≥ 3.5 ms' in RQ.table(card)


def test_all_pass_and_empty_and_errors(fake):
    p = _project(n=0)
    card = RQ.check(p, [RQ.clean({'metric': '_rq.n', 'op': '==', 'threshold': 0})], cache=jobs.ResultCache())
    assert card['verdict'] == 'pass' and RQ.exit_code(card) == 0 and card['headline']['word'] == 'ready'
    empty = RQ.check(p, [], cache=jobs.ResultCache())
    assert empty['verdict'] == 'empty' and RQ.exit_code(empty) == 1 and 'no requirements' in empty['verdict_text']
    bad = Requirement(name='Typo', metric='nope.x', op='>=', threshold=1)
    card = RQ.check(p, [bad], cache=jobs.ResultCache())
    assert card['items'][0]['status'] == 'not_evaluated' and 'does not have' in card['items'][0]['reason']
    assert card['errors'] and RQ.exit_code(card) == 2


def test_a_study_that_crashes_is_an_error_not_a_crash(fake, capsys):
    @jobs.study('_rq_boom', 'Boom')
    def _boom(project, ctx):
        raise RuntimeError('kaput')
    M.metric('_rq.boom', 'Boom', 'mm', 'lower', study='_rq_boom')(lambda d: 1)
    try:
        card = RQ.check(_project(), [RQ.clean({'metric': '_rq.boom', 'threshold': 1})], cache=jobs.ResultCache())
        assert card['studies']['_rq_boom']['status'] == 'error'
        assert 'RuntimeError: kaput' in card['errors'][0]['message']
        assert RQ.exit_code(card) == 2
    finally:
        jobs.STUDIES.pop('_rq_boom', None)
        M.METRICS.pop('_rq.boom', None)


def test_table_lists_every_item_and_the_verdict(fake):
    card = RQ.check(_project(), [RQ.clean({'metric': '_rq.n', 'op': '<=', 'threshold': 2, 'name': 'Few sensors',
                                           'default': True})], cache=jobs.ResultCache())
    t = RQ.table(card)
    assert 'Few sensors' in t and '≤ 2 (default)' in t and 'pass' in t
    assert t.splitlines()[-1].startswith('Verdict: ready')


def test_project_requirements_round_trip_through_the_file(tmp_path):
    p = _project()
    p.requirements = RQ.default_requirements()
    path = str(tmp_path / 'r.dexv')
    p.save(path)
    q = Project.load(path)
    assert [r.to_dict() for r in q.requirements] == [r.to_dict() for r in p.requirements]


# ====================================================================== variants compare
def test_every_variant_against_every_requirement_and_the_best(fake):
    p = _project(n=1, variants=3)                       # 1, 2 and 3 sensors
    reqs = [RQ.clean({'metric': '_rq.n', 'op': '<=', 'threshold': 2}),
            RQ.clean({'metric': '_rq.skew', 'op': '<', 'threshold': 4})]
    d = RQ.compare_variants(p, reqs, cache=jobs.ResultCache())
    v1, v2, v3 = (v['uid'] for v in d['variants'])
    assert [v['name'] for v in d['variants']] == ['V1', 'V2', 'V3'] and len(d['cells']) == 3
    assert [c['status'] for c in d['cells'][0]] == ['pass', 'pass']
    assert [c['status'] for c in d['cells'][1]] == ['pass', 'not_evaluated']       # 3.5 "at least"
    assert [c['status'] for c in d['cells'][2]] == ['fail', 'fail']                # at least 4.5 ms
    assert d['best'][reqs[0].uid] == [v1] and d['best'][reqs[1].uid] == [v1]
    assert d['ranking'] == [v1, v2, v3] and d['active_uid'] == v1
    assert len(fake) == 3                               # one run of the study per variant


def test_best_for_equality_is_the_closest_and_ties_are_all_named(fake):
    p = _project(n=1, variants=3)
    p.variants[2].sensors = p.variants[2].sensors[:1]   # V3 has 1 sensor, like V1
    d = RQ.compare_variants(p, [RQ.clean({'metric': '_rq.n', 'op': '==', 'threshold': 2})], cache=jobs.ResultCache())
    assert d['best'][d['requirements'][0]['uid']] == [d['variants'][1]['uid']]
    d = RQ.compare_variants(p, [RQ.clean({'metric': '_rq.n', 'op': '<=', 'threshold': 5})], cache=jobs.ResultCache())
    assert set(d['best'][d['requirements'][0]['uid']]) == {d['variants'][0]['uid'], d['variants'][2]['uid']}


# ====================================================================== the studies
def test_the_scorecard_and_compare_are_registered_studies_with_batch_flags(fake):
    jobs.load_studies()
    assert jobs.STUDIES['requirements'].batch == 'scorecard'
    assert jobs.STUDIES['variants_compare'].batch == 'variants-compare'
    assert jobs.STUDIES['rig'].batch == 'rig-inventory'
    p = _project(n=1, variants=2)
    p.requirements = [RQ.clean({'metric': '_rq.n', 'op': '<=', 'threshold': 1})]
    doc = jobs.run_study('requirements', p)
    assert doc['schema'] == 'dexvantage.requirements/1' and doc['verdict'] == 'pass'
    assert doc['requirements'][0]['metric'] == '_rq.n'
    doc = jobs.run_study('variants_compare', p)
    assert [v['verdict'] for v in doc['variants']] == ['pass', 'fail']


def test_scorecard_on_the_ur5e_sample_with_the_defaults():
    """The labelled defaults on a real sample: the sync skew (at least 54.4 ms) and the cable snags fail, the
    bandwidth is not evaluated (no host), each with its study's own figure."""
    from dexvantage.core.robot.samples import sample_project
    p = sample_project('ur5e')
    p.analysis.studies['grasp_visibility'] = {'sensitivity': False, 'bootstrap': 200, 'cube_n': 256}
    card = RQ.check(p, RQ.default_requirements())
    by = {i['metric']: i for i in card['items']}
    assert by['grasp.contact_visibility']['status'] == 'pass'
    assert by['sync.skew_ms']['status'] == 'fail' and by['sync.skew_ms']['bound'] == 'at_least'
    assert by['sync.skew_ms']['value'] == pytest.approx(54.4258, abs=1e-3)
    bw = by['bandwidth.over_budget']
    assert bw['status'] == 'not_evaluated' and 'no host' in bw['reason']
    assert by['cables.pull_snag']['status'] == 'fail' and by['cables.pull_snag']['value'] >= 1
    assert card['verdict'] == 'fail' and RQ.exit_code(card) == 1
