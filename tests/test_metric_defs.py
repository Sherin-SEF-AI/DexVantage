"""The metric definitions (core/metric_defs.py): every metric read from a real result of its study on a sample, against
the figure the result states (read here again by hand), with its unit, bound and the plain reason when not stated."""
import math
import os

import pytest

from dexvantage.core import jobs
from dexvantage.core import metrics as M
from dexvantage.core.robot.samples import sample_project

FAST = {'grasp_visibility': {'sensitivity': False, 'bootstrap': 200, 'cube_n': 256},
        'groundtruth': {'grasps': 8, 'thumbs': 0, 'trajectory_frames': 2}}
SPEC_IDS = ('grasp.contact_visibility', 'coverage.covered', 'depth.valid', 'bandwidth.margin', 'sync.skew_ms',
            'cables.worst', 'vla.min_tokens', 'action_chunk.air_grasp_mm', 'teleop.overlap', 'lighting.depth_loss',
            'motion_blur.worst_px', 'handeye.sigma_mm', 'groundtruth.agreement', 'rig.camera_count', 'rig.cost')


def _fast(p):
    for k, v in FAST.items():
        p.analysis.studies.setdefault(k, {}).update(v)
    return p


class Run:
    """Each study run once on a project; evaluate() reads from the same cache, so the docs are the ones measured."""

    def __init__(self, project):
        self.p = project
        self.cache = jobs.ResultCache(64)
        self.docs = {}

    def doc(self, sid):
        if sid not in self.docs:
            self.docs[sid] = jobs.run_study(sid, self.p, cache=self.cache)
        return self.docs[sid]

    def ev(self, mid):
        m = M.get(mid)
        try:
            self.doc(m.study)
        except jobs.StudyError:
            pass                                      # evaluate() states the refusal
        return M.evaluate(self.p, [mid], cache=self.cache)[mid]


@pytest.fixture(scope='module')
def lit():
    """The UR5e sample with a can and lights (the lighting checks' scene) and the suggested host."""
    from dexvantage.core.studies import bandwidth as BW
    from dexvantage.validation.checks_lighting import lit_sample
    p = _fast(lit_sample())
    BW.apply_suggestion(p, BW.budget(p)['suggestion'])
    return Run(p)


@pytest.fixture(scope='module')
def so100():
    return Run(_fast(sample_project('so100')))


@pytest.fixture(scope='module')
def dexhand():
    return Run(sample_project('dexhand'))


def _ok(row, want, unit=None, rel=1e-9):
    assert row['status'] == 'ok', row
    assert row['value'] == pytest.approx(float(want), rel=rel, abs=1e-12), row
    if unit is not None:
        assert row['unit'] == unit


def test_every_spec_id_and_every_study_headline_is_registered():
    M.load_metrics()
    jobs.load_studies()
    for mid in SPEC_IDS:
        assert mid in M.METRICS, mid
    with_headline = {m.study for m in M.METRICS.values() if 'headline' in m.tags}
    for sid in jobs.STUDIES:
        if any(m.study == sid for m in M.METRICS.values()):
            assert sid in with_headline, sid
    for m in M.METRICS.values():
        if m.id.startswith('_'):
            continue
        assert m.better in ('higher', 'lower') and m.unit and m.title and m.study in jobs.STUDIES, m.id


def test_grasp_coverage_depth_on_the_lit_ur5e(lit):
    g = lit.doc('grasp_visibility')['summary']
    _ok(lit.ev('grasp.contact_visibility'), g['cvs_pct'], '%')
    _ok(lit.ev('grasp.contact_visibility_low'), g['ci_low'] * 100)
    _ok(lit.ev('grasp.both_fingertips'), g['both_fingertips_rate'] * 100)
    _ok(lit.ev('grasp.reachable'), 100.0 * g['feasible'] / g['grasps'])
    c = lit.doc('coverage')['summary']
    _ok(lit.ev('coverage.covered'), c['seen_ge1'] * 100, '%')
    _ok(lit.ev('coverage.seen_twice'), c['seen_ge2'] * 100)
    _ok(lit.ev('coverage.blind'), c['blind'] * 100)
    _ok(lit.ev('coverage.ppmm_p5'), c['best_ppmm_p5'], 'px/mm')
    d = lit.doc('depth_validity')
    _ok(lit.ev('depth.valid'), d['rig']['valid_pct'], '%')
    _ok(lit.ev('depth.worst_sensor_valid'), min(s['valid_pct'] for s in d['sensors']))
    _ok(lit.ev('depth.sigma_z_p95_mm'), max(s['sigma_z']['p95_m'] for s in d['sensors']) * 1000, 'mm')


def test_bandwidth_with_the_suggested_host(lit):
    b = lit.doc('bandwidth')
    (c,) = b['controllers']
    _ok(lit.ev('bandwidth.margin'), c['margin_mbps'], 'Mbit/s')
    assert c['margin_mbps'] == pytest.approx(c['capacity_mbps'] - c['load_mbps'])
    _ok(lit.ev('bandwidth.busiest_load'), b['headline']['busiest_load_pct'], '%')
    over = lit.ev('bandwidth.over_budget')
    _ok(over, int(c['load'] > 1.0), 'count')
    assert 'bound' not in over                      # every stream of the sample has a stated rate
    _ok(lit.ev('bandwidth.payload_mbps'), b['headline']['total_payload_mbps'])


def test_a_rig_without_a_host_states_why(so100):
    r = so100.ev('bandwidth.margin')
    assert r['value'] is None and r['status'] == 'not stated' and 'no host controller' in r['reason']


def test_sync_skew_is_a_lower_bound_when_figures_are_missing(lit):
    s = lit.doc('time_sync')['summary']
    r = lit.ev('sync.skew_ms')
    _ok(r, s['worst_skew_s'] * 1000, 'ms')
    assert s['complete'] is False and r['bound'] == 'at_least' and 'at least' in r['note']
    _ok(lit.ev('sync.label_error_mm'), s['label_error_m'] * 1000, 'mm')


def test_cables_counts(lit):
    s = lit.doc('cables')['summary']
    _ok(lit.ev('cables.worst'), s['cables'] - s['clear'], 'count')
    r = lit.ev('cables.pull_snag')
    _ok(r, s['flags'].get('pull', 0) + s['flags'].get('snag', 0))
    assert s['flags'].get('unknown') and r['bound'] == 'at_least'
    _ok(lit.ev('cables.bend'), s['flags'].get('bend', 0))


def test_vla_tokens_on_the_can(lit):
    s = lit.doc('vla_tokens')['summary']
    assert s['fewest'] is not None and s['fewest']['object'] == 'Soda can'
    _ok(lit.ev('vla.min_tokens'), s['fewest']['tokens'], 'tokens')
    _ok(lit.ev('vla.fingertip_tokens'), s['fingertips_fewest']['tokens'])
    _ok(lit.ev('vla.warnings'), s['warnings'])


def test_vla_without_a_task_object_says_why(so100):
    r = so100.ev('vla.min_tokens')
    assert r['value'] is None and 'critical task object' in r['reason']


def test_action_chunk_on_the_ur5e(lit):
    h = lit.doc('action_chunk_sync')['headline']
    _ok(lit.ev('action_chunk.air_grasp_mm'), h['air_grasp_mm'], 'mm')
    _ok(lit.ev('action_chunk.worst_mm'), h['worst_mm'])
    _ok(lit.ev('action_chunk.overshoot_mm'), h['overshoot_mm'])
    _ok(lit.ev('action_chunk.reaction_mm'), h['reaction_mm'])


def test_action_chunk_refusal_is_its_sentence(so100):
    r = so100.ev('action_chunk.air_grasp_mm')
    assert r['status'] == 'refused' and 'inference time is not stated' in r['reason']


def test_teleop_overlap_on_the_so100(so100):
    _ok(so100.ev('teleop.overlap'), so100.doc('teleop')['summary']['teleoperable_pct'], '%')


def test_lighting_on_the_lit_scene(lit):
    h = lit.doc('lighting')['headline']
    _ok(lit.ev('lighting.glare'), h['value'], '%')
    _ok(lit.ev('lighting.depth_loss'), h['depth_loss_pct'], '%')


def test_lighting_without_lights_says_so(so100):
    r = so100.ev('lighting.glare')
    assert r['value'] is None and 'no lights' in r['reason']


def test_motion_blur_is_an_upper_bound_over_the_maximum_exposure(lit):
    s = lit.doc('motion_blur')['summary']
    r = lit.ev('motion_blur.worst_px')
    _ok(r, s['smear_px'], 'px')
    assert s['smear_bound'] and r['bound'] == 'at_most'
    r = lit.ev('motion_blur.skew_px')
    _ok(r, s['skew_px'], 'px')
    assert r.get('bound') == ('at_most' if s['skew_bound'] else None)


def test_handeye_plan_sigma(lit):
    d = lit.doc('handeye_plan')
    sx = d['sigma']['X']
    _ok(lit.ev('handeye.sigma_mm'), max(sx['x_mm'], sx['y_mm'], sx['z_mm']), 'mm')
    _ok(lit.ev('handeye.sigma_deg'), max(sx['rx_deg'], sx['ry_deg'], sx['rz_deg']), '°')
    _ok(lit.ev('handeye.ratio'), d['worst']['ratio'])


def test_groundtruth_agreement(lit):
    s = lit.doc('groundtruth')['summary']
    _ok(lit.ev('groundtruth.agreement'), s['agreement_pct'], '%')
    _ok(lit.ev('groundtruth.depth_agreement'), s['depth_agreement_pct'])


def test_dexhand_and_the_blind_gap(dexhand, lit):
    d = dexhand.doc('dex_occlusion')
    _ok(dexhand.ev('dexhand.fingertips_seen'), d['rig']['fingertips_seen_pct'], '%')
    _ok(dexhand.ev('dexhand.all_fingertips'), d['rig']['all_fingertips_rate_pct'])
    r = dexhand.ev('dexhand.blind_gap_mm')
    _ok(r, d['handoff']['blind_gap_m'] * 1000, 'mm')
    assert 'bound' not in r
    _ok(dexhand.ev('dexhand.unsensed'), sum(1 for a in d['handoff']['approach_visibility'] for f in a['fingers']
                                            if not f['sensed_at_contact']), 'count')
    # the parallel gripper of the UR5e: blind from the start of the scan, so "at least"
    h = lit.doc('dex_occlusion')['handoff']
    r = lit.ev('dexhand.blind_gap_mm')
    _ok(r, h['blind_gap_m'] * 1000)
    censored = any(f['censored'] for a in h['approach_visibility'] for f in a['fingers']
                   if f['blind_gap_m'] == h['blind_gap_m'])
    assert r.get('bound') == ('at_least' if censored else None)


def test_brackets(lit):
    r = lit.doc('brackets')['rig']
    _ok(lit.ev('brackets.not_ready'), r['brackets'] - r['ready'] + r['unset'], 'count')


def test_rig_inventory_cost_sums_the_cited_prices(lit, dexhand):
    from dexvantage.core import library as LB
    d = lit.doc('rig')
    on = [s for s in lit.p.variant.sensors if s.enabled]
    _ok(lit.ev('rig.camera_count'), sum(s.kind in ('camera', 'depth') for s in on), 'count')
    _ok(lit.ev('rig.sensor_count'), len(on))
    cost = lit.ev('rig.cost')           # every sensor of the sample states a cited price (core/prices.py)
    _ok(cost, sum(LB.get(s.model_id).data['price']['amount'] for s in on), 'USD')
    assert not cost.get('bound') and d['cost']['missing'] == []
    _ok(lit.ev('rig.mass_g'), sum(LB.get(s.model_id).data['mass_g'] for s in on), 'g')
    _ok(lit.ev('rig.power_w'), sum(LB.get(s.model_id).data['power_w'] for s in on), 'W')


def test_rig_cost_sums_stated_prices_and_refuses_mixed_currencies(lit):
    from dexvantage.core import library as LB
    from dexvantage.core.studies import rig_inventory as RI
    entries = {s.model_id: LB.get(s.model_id) for s in lit.p.variant.sensors}
    saved = {k: e.data for k, e in entries.items()}
    try:
        for i, e in enumerate(entries.values()):
            e.data = dict(e.data, price={'value': 100.0 + 50 * i, 'currency': 'USD', 'source': 'test figure'})
        inv = RI.inventory(lit.p)
        assert inv['cost']['value'] == pytest.approx(sum(100.0 + 50 * i for i in range(len(entries))))
        assert inv['cost']['unit'] == 'USD'
        k0 = next(iter(entries))
        entries[k0].data = dict(entries[k0].data, price={'value': 90.0, 'currency': 'EUR'})
        inv = RI.inventory(lit.p)
        assert inv['cost']['value'] is None and 'different currencies' in inv['cost']['reason']
    finally:
        for k, e in entries.items():
            e.data = saved[k]


def test_data_studies_on_a_small_synthetic_campaign(tmp_path):
    from dexvantage.core.sim import episodes as SE
    from dexvantage.core.studies.audit import use_dataset
    p = sample_project('so100')
    root = str(tmp_path / 'camp')
    SE.generate(p, root, {'episodes': 2, 'frames': 60, 'size': [192, 108], 'seed': 2,
                          'faults': ['drop', 'mount_slip']})
    for sid, checks in (('audit', {'audit.pass_rate': ('summary', 'pass_pct')}),
                        ('plan_vs_recording', {'plan.match': ('summary', 'match_pct')}),
                        ('silhouette_calib', {'silhouette.offset_mm': ('headline', 'offset_mm'),
                                              'silhouette.offset_deg': ('headline', 'offset_deg')})):
        s = p.snapshot()
        use_dataset(s, root, sid)
        run = Run(s)
        doc = run.doc(sid)
        for mid, (a, b) in checks.items():
            _ok(run.ev(mid), doc[a][b])
    assert os.path.isdir(root)


def test_extractor_on_an_unexpected_shape_reads_not_stated():
    m = M.get('coverage.covered')
    assert M.read(m, {'summary': 'junk'})['status'] == 'not stated'
    assert M.read(m, {'summary': {'seen_ge1': math.nan}})['value'] is None
    r = M.read(M.get('motion_blur.worst_px'), {'summary': {'smear_px': None, 'word': 'exposure not set'}})
    assert r['value'] is None and 'bound' not in r and 'exposure not set' in r['reason']
