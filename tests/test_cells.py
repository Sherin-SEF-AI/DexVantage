"""The multi-cell comparison (core/cells.py, core/cells_pool.py, the 'cells' study, its exporter, batch --cells and
dv.cells): the six samples and edited copies; the geometric gap zero against itself and symmetric, each component
following its change, wrist cameras compared in the contact frame; dataset pooling on the plans and on synthetic
LeRobot datasets read back through the readers; drift of calibrated copies (hand-computed robust z, the systematic
offset and the as-built variant); the figures table with the tolerance spread; the files, batch and API.

Ported in part from Vantage tests/test_rig_gap.py (zero against itself, symmetry, components following their change,
a missing camera) and tests/test_fleet.py (robust z against the hand computation, the systematic offset and the
fleet-median variant)."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from dexvantage.core import cells as CE
from dexvantage.core import cells_pool as CP
from dexvantage.core import jobs
from dexvantage.core.frames import Pose
from dexvantage.core.jobs import StudyError
from dexvantage.core.robot.samples import sample_project
from dexvantage.validation import checks_cells as CC

SAMPLES = ('so100', 'ur5e', 'dexhand', 'aloha', 'panda-bimanual', 'g1')
WRIST, SCENE = 'Wrist camera (sample placement)', 'Third-person camera (sample placement)'
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
FAST = ['rig.camera_count']


def _gap(a, b):
    return CC._gap(a, b)


def _have_lerobot():
    try:
        import av  # noqa: F401
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


# ====================================================================== the six samples
def test_six_samples_side_by_side():
    cells = CE.make_cells([f'sample:{n}' for n in SAMPLES])
    doc = CE.compare(cells)                                   # the default figures (the quick studies)
    json.dumps(doc, allow_nan=False)
    assert [c['source'] for c in doc['cells']] == ['sample'] * 6 and doc['reference'] == 0
    M = np.array(doc['gap']['matrix'])
    assert M.shape == (6, 6) and np.allclose(np.diag(M), 0) and np.allclose(M, M.T, atol=1e-12)
    assert (M[~np.eye(6, dtype=bool)] > 0).all() and (M <= 1).all()
    assert len(doc['gap']['pairs']) == 5
    h = doc['headline']
    assert h['value'] == pytest.approx(M[0].max()) and h['word'].endswith('gap')
    assert h['status'] in CE.VERDICT_SEV.values()
    # every metric in one table, best and worst among the stated, plain values
    ids = [r['id'] for r in doc['metrics']['rows']]
    assert ids == list(CE.DEFAULT_METRICS)
    for r in doc['metrics']['rows']:
        assert len(r['values']) == 6
        plain = [v for v, b in zip(r['values'], r['bounds']) if v is not None and not b]
        if r['best'] is not None:
            hi = r['better'] == 'higher'
            assert r['values'][r['best']] == (max(plain) if hi else min(plain))
            assert r['values'][r['worst']] == (min(plain) if hi else max(plain))
        for v, why in zip(r['values'], r['reasons']):
            assert v is not None or why                       # a figure not stated says why
    cov = next(r for r in doc['metrics']['rows'] if r['id'] == 'rig.camera_count')
    assert cov['values'] == [2.0, 2.0, 3.0, 4.0, 3.0, 1.0]
    # pooling: different embodiments cannot be pooled by renaming
    p = doc['pooling']
    assert sum(p['counts'].values()) == 6 and p['counts']['not poolable'] == 5 and not p['poolable']
    for c in p['cells'][1:]:
        assert {'observation.state', 'action'} <= {u['what'] for u in c['unmappable']}
    g1 = p['cells'][5]
    assert 'observation.images.wrist_camera' in {u['what'] for u in g1['unmappable']}     # the G1 has no wrist view
    # roles: the G1's head camera rides on a link no joint moves, so it is a scene camera
    g1cams = doc['cells'][5]['cameras_detail']
    assert [c['role'] for c in g1cams] == ['scene']
    assert all('contact frame' in c['frame'] for c in doc['cells'][3]['cameras_detail'] if c['role'] == 'wrist')


def test_identical_is_zero_and_symmetric():
    for n in ('ur5e', 'aloha'):
        p = sample_project(n)
        g = _gap(p, p.snapshot())
        assert g['total'] == 0.0 and all(v == 0.0 for v in g['components'].values())
        assert [(m['a'], m['b']) for m in g['matches']] == [(m['a'], m['a']) for m in g['matches']]
    a, b = sample_project('so100'), sample_project('panda-bimanual')
    g1, g2 = _gap(a, b), _gap(b, a)
    assert g1['total'] == pytest.approx(g2['total'], abs=1e-12)
    for k in g1['components']:
        assert g1['components'][k] == pytest.approx(g2['components'][k], abs=1e-12), k


def test_each_component_follows_its_change():
    p = sample_project('so100')
    up = CC.turn_camera(p, SCENE, np.zeros(3), (0.0, 0.0, 0.1))
    g = _gap(p, up)
    assert g['components']['position'] == pytest.approx(0.5 * 0.1 / CE.POSITION_SCALE_M, abs=1e-12)
    assert g['components']['viewpoint'] == pytest.approx(0.0, abs=1e-12)
    th = math.radians(30)
    turned = _gap(p, CC.turn_camera(p, SCENE, (th, 0.0, 0.0)))
    assert turned['components']['viewpoint'] == pytest.approx(0.5 * 30 / CE.ANGLE_SCALE_DEG, abs=1e-9)
    assert turned['components']['coverage'] > 0                       # the table moves in the image
    # another device on the scene camera: another FOV and resolution
    q = p.snapshot()
    next(s for s in q.variant.sensors if s.name == SCENE).model_id = 'intel-realsense-d435'
    dev = _gap(p, q)
    m = next(x for x in dev['matches'] if x['a'] == SCENE)
    assert dev['components']['fov'] == pytest.approx(0.5 * abs(m['hfov_diff_deg']) / 90, abs=1e-12)
    assert dev['components']['resolution'] == pytest.approx(0.5 * abs(math.log2(m['resolution_ratio'])) / 2,
                                                            abs=1e-12) and m['resolution_ratio'] != 1
    # a camera missing: topology ½(1/2 + 1/3) and that camera unmatched
    r = p.snapshot()
    r.variant.sensors = [s for s in r.variant.sensors if s.name != SCENE]
    few = _gap(p, r)
    assert few['components']['topology'] == pytest.approx(0.5 * (1 / 2 + 1 / 3)) and few['unmatched_a'] == [SCENE]
    assert few['cameras'] == [2, 1]


def test_wrist_cameras_compare_in_the_contact_frame():
    p = sample_project('so100')
    q = p.snapshot()
    q.robots[0].base = Pose(np.array([0.3, -0.2, 0.05]))              # the robot stands elsewhere in the world
    g = _gap(p, q)
    wrist = next(m for m in g['matches'] if m['role'] == 'wrist')
    scene = next(m for m in g['matches'] if m['role'] == 'scene')
    assert wrist['distance_m'] == pytest.approx(0.0, abs=1e-9) and wrist['viewpoint_deg'] == pytest.approx(0, abs=1e-7)
    assert scene['distance_m'] == pytest.approx(0.0, abs=1e-12)          # the tripod did not move
    # which pad comes first is arbitrary: a camera half a turn about the approach axis reads as the same place
    import dataclasses
    rec = next(r for r in CE.cameras(p)[0] if r.role == 'wrist')
    mirrored = dataclasses.replace(rec, T=CE.FLIP @ rec.T)
    f = CE.pair_figures(rec, mirrored)
    assert f['closing_flipped'] and f['rotation_deg'] == pytest.approx(0, abs=1e-9) and f['distance_m'] < 1e-12
    assert not CE.pair_figures(rec, rec)['closing_flipped']
    # two arms: each wrist camera matched on its own arm
    a = sample_project('aloha')
    ga = _gap(a, a.snapshot())
    assert {(m['a'], m['b']) for m in ga['matches'] if m['role'] == 'wrist'} == {
        ('cam_left_wrist', 'cam_left_wrist'), ('cam_right_wrist', 'cam_right_wrist')}


def test_ray_box_and_coverage_basics():
    O = np.zeros((3, 3))
    D = np.array([[0, 0, 1.0], [0, 0, -1.0], [1.0, 0, 0]])
    hits = CE.ray_box_hits(O, D, np.array([-0.1, -0.1, 1.0]), np.array([0.1, 0.1, 2.0]))
    assert hits.tolist() == [True, False, False]
    inside = CE.ray_box_hits(np.zeros((1, 3)), np.array([[0, 1.0, 0]]), -np.ones(3), np.ones(3))
    assert inside.tolist() == [True]
    from dexvantage.core.optics import CameraModel
    cam = CameraModel(64, 48, 40.0, 40.0, 31.5, 23.5)
    full = CE.task_coverage(cam, np.eye(4), [(np.array([-5, -5, 0.5]), np.array([5, 5, 3.0]))])
    assert full['image_pct'] == 100.0 and full['in_view_pct'] < 100.0
    none = CE.task_coverage(cam, np.eye(4), [(np.array([-1, -1, -3]), np.array([1, 1, -2.0]))])
    assert none['image_pct'] == 0.0 and none['in_view_pct'] == 0.0
    assert CE.task_coverage(cam, np.eye(4), [])['image_pct'] is None


# ====================================================================== pooling
def test_pooling_on_plans_rename_resize_and_as_is():
    p = sample_project('ur5e')
    q = p.snapshot()
    w = next(s for s in q.variant.sensors if s.name == 'Overhead D435')
    w.name = 'Top camera'
    w.stream = [1280, 720]
    pool = CP.pooling([CP.plan_schema(p), CP.plan_schema(q), CP.plan_schema(p.snapshot())], ['a', 'b', 'c'], 0)
    b = pool['cells'][1]
    ops = {(o['op'], o['feature']) for o in b['operations']}
    assert ('rename', 'observation.images.top_camera') in ops and ('resize', 'observation.images.top_camera') in ops
    rs = next(o for o in b['operations'] if o['op'] == 'resize')
    assert rs['from'] == [1280, 720] and rs['to'] == [1920, 1080] and not rs['aspect_changes']
    assert b['verdict'] == 'poolable after mapping' and b['camera_map']['observation.images.top_camera'] == \
        'observation.images.overhead_d435'
    assert pool['cells'][2]['verdict'] == 'poolable as is' and pool['counts']['poolable as is'] == 2
    sc = CP.plan_schema(p)
    assert sc['state']['units'][:6] == ['rad'] * 6 and sc['action']['space']['value'] == 'joint position'
    assert sc['cameras'][0]['codec'] == 'av1' and 'default' in sc['cameras'][0]['codec_source']
    assert {c['role'] for c in sc['cameras']} == {'wrist', 'scene'} and sc['gripper']['joints']


def test_pooling_rules_unit_cases():
    def sc(**kw):
        base = {'source': 'recorded', 'fps': {'value': 10.0}, 'control_hz': {'value': 10.0},
                'cameras': [{'key': 'observation.images.wrist', 'role': 'wrist', 'width': 64, 'height': 48,
                             'fps': 10.0, 'codec': 'av1'}],
                'state': {'names': ['a', 'b'], 'count': 2, 'units': ['rad', 'm']},
                'action': {'names': ['a', 'b'], 'count': 2, 'units': ['rad', 'm'],
                           'space': {'value': 'joint position'}},
                'gripper': {'joints': ['b'], 'type': 'prismatic', 'range': [0.0, 0.08]}}
        base.update(kw)
        return base
    t = sc()
    assert CP.rules(t, sc())['verdict'] == 'poolable as is'
    r = CP.rules(t, sc(state={'names': ['a', 'b'], 'count': 2, 'units': ['rad', 'rad']}))
    assert r['verdict'] == 'not poolable' and 'another kind' in r['unmappable'][0]['why']
    r = CP.rules(t, sc(action={'names': ['x', 'y'], 'count': 2, 'units': None, 'space': {'value': 'end effector'}}))
    assert any(u['what'] == 'action space' for u in r['unmappable'])
    r = CP.rules(t, sc(cameras=[{'key': 'observation.images.wrist', 'role': 'wrist', 'width': 64, 'height': 64,
                                 'fps': 10.0, 'codec': 'h264'}]))
    ops = {o['op']: o for o in r['operations']}
    assert ops['resize']['aspect_changes'] and ops['reencode']['from'] == 'h264' and r['verdict'] == \
        'poolable after mapping'
    r = CP.rules(t, sc(cameras=[]))
    assert r['unmappable'][0]['what'] == 'observation.images.wrist'
    r = CP.rules(t, sc(state={'names': ['a', 'b'], 'count': 2, 'units': None,
                              'units_hint': {'joints': ['a'], 'source': 'values reach 90'}}))
    assert [o['op'] for o in r['operations']] == ['convert', 'convert']
    r = CP.rules(t, sc(state={'names': ['c', 'd'], 'count': 2, 'units': ['rad', 'm']}))
    assert any(o.get('by') == 'position' and 'assumption' in o['why'] for o in r['operations'])


@pytest.mark.skipif(not _have_lerobot(), reason='pyarrow and av (the lerobot extra) are needed')
def test_pooling_recorded_datasets_through_the_readers(tmp_path):
    paths = CC.write_pooling_datasets(str(tmp_path))
    sa, sb = CP.recorded_schema(paths['a']), CP.recorded_schema(paths['b'])
    assert sa['version'] == 'v3.0' and sb['version'] == 'v2.1' and sa['source'] == 'recorded'
    assert [(c['key'], c['width'], c['height'], c['fps']) for c in sb['cameras']] == [
        (CC.LR_KEY_B, 80, 48, 5.0), (CC.LR_TOP, 64, 48, 5.0)]
    assert sb['state']['names'] == ['gripper', 'j1', 'j2'] and sb['gripper']['range'] == pytest.approx([0, 0.04])
    pool = CP.pooling([sa, sb, CP.recorded_schema(paths['c'])], ['a', 'b', 'c'])
    got = {(o['op'], o['feature']) for o in pool['cells'][1]['operations']}
    assert got == CC.EXPECTED_B
    assert pool['cells'][2]['verdict'] == 'not poolable'
    # the cells' own datasets: 'auto' reads them, 'plan' does not; a missing dataset falls back with a note
    p = sample_project('so100')
    p.data['datasets'] = [{'uid': 'd1', 'name': 'run 1', 'path': paths['b']}]
    assert CP.cell_schema(p, 'auto')['source'] == 'recorded' and CP.cell_schema(p, 'plan')['source'] == 'plan'
    p.data['datasets'] = [{'uid': 'd1', 'name': 'gone', 'path': str(tmp_path / 'nothing')}]
    fb = CP.cell_schema(p, 'auto')
    assert fb['source'] == 'plan' and 'could not be read' in fb['notes'][0]
    with pytest.raises(StudyError):
        CP.cell_schema(p, 'recorded')
    with pytest.raises(StudyError, match='no recorded dataset'):
        CP.cell_schema(sample_project('so100'), 'recorded')


def test_schema_from_info_names_and_shapes():
    info = {'codebase_version': 'v2.1', 'fps': 15, 'features': {
        'observation.images.cam_wrist': {'dtype': 'video', 'shape': [480, 640, 3],
                                         'names': ['height', 'width', 'channels'],
                                         'info': {'video.fps': 15.0, 'video.codec': 'h264'}},
        'observation.images.front': {'dtype': 'image', 'shape': [3, 96, 128], 'names': ['channels', 'height',
                                                                                         'width']},
        'observation.state': {'dtype': 'float32', 'shape': [3], 'names': {'motors': ['a', 'b', 'gripper']}},
        'action': {'dtype': 'float32', 'shape': [3], 'names': {'motors': ['a', 'b', 'gripper']}}}}
    sc = CP.schema_from_info(info)
    c = {x['key']: x for x in sc['cameras']}
    assert (c['observation.images.cam_wrist']['width'], c['observation.images.cam_wrist']['height']) == (640, 480)
    assert c['observation.images.cam_wrist']['role'] == 'wrist' and c['observation.images.front']['role'] == 'scene'
    assert (c['observation.images.front']['width'], c['observation.images.front']['height']) == (128, 96)
    assert sc['state']['names'] == ['a', 'b', 'gripper'] and sc['action']['space']['value'] == 'joint position'
    assert sc['fps']['value'] == 15.0 and sc['gripper']['joints'] == ['gripper']


# ====================================================================== drift
def test_drift_robust_z_against_the_hand_computation():
    p = sample_project('ur5e')
    xs = [0.10, -0.20, 0.05, 2.30, 0.00]
    copies = [CC.calibrated_copy(p, 'Wrist D405', (x, 0, 0), name=f'V{k + 1}') for k, x in enumerate(xs)]
    dr = CE.drift(CE.make_cells(copies, open_project=p), 0)
    c = next(c for c in dr['consistency'] if c['sensor'] == 'Wrist D405')
    med = float(np.median(xs))
    sigma = 1.4826 * float(np.median(np.abs(np.array(xs) - med)))
    spread = max(sigma, CE.SIGMA_FLOOR * 1.0)
    assert c['median'][0] == pytest.approx(med) and c['sigma'][0] == pytest.approx(sigma)
    assert [o['name'] for o in c['outliers']] == ['V4'] and c['outliers'][0]['component'] == 'x'
    assert c['outliers'][0]['z'] == pytest.approx((2.30 - med) / spread) and not c['systematic']
    v4 = next(e for r in dr['cameras'] for e in r['copies'] if e['name'] == 'V4' and r['sensor'] == 'Wrist D405')
    assert v4['kind'] == 'measured' and v4['verdict'] == 'adjust' and v4['ratio'] == pytest.approx(2.30)
    assert dr['worst']['name'] == 'V4' and dr['measured_cells'] == [1, 2, 3, 4, 5]
    # the overhead camera has no calibration: compared as designed (the same design: no offset)
    oh = next(r for r in dr['cameras'] if r['sensor'] == 'Overhead D435')
    assert {e['kind'] for e in oh['copies']} == {'as designed'} and all(e['ratio'] == 0 for e in oh['copies'])


def test_systematic_offset_and_the_as_built_variant():
    """Every copy's wrist camera is built 3 mm further along the link's z than planned: the plan should follow."""
    p = sample_project('ur5e')
    copies = [CC.calibrated_copy(p, 'Wrist D405', (0, 0, 3.0 + 0.1 * k), (0.02, 0, 0), name=f'B{k}')
              for k in range(4)]
    doc = CE.compare(CE.make_cells(copies, open_project=p), metrics=FAST)
    dr = doc['drift']
    c = next(c for c in dr['consistency'] if c['sensor'] == 'Wrist D405')
    assert c['systematic'] and c['near_median'] == 4 and c['median'][2] == pytest.approx(3.15)
    sg = doc['suggestion']
    assert sg['kind'] == 'as_built_variant' and sg['sensors'] == ['Wrist D405'] and 'median of 4' in sg['name']
    # the as-built poses put the camera at the median: against them the copies are within tolerance
    q = p.snapshot()
    s = next(s for s in q.variant.sensors if s.name == 'Wrist D405')
    s.mount.pose = Pose.from_dict(sg['poses'][s.uid])
    dr2 = CE.drift(CE.make_cells(copies, open_project=q), 0)
    c2 = next(c for c in dr2['consistency'] if c['sensor'] == 'Wrist D405')
    assert not c2['systematic'] and c2['median_ratio'] < 0.2
    assert all(e['verdict'] == 'ok' for r in dr2['cameras'] for e in r['copies'] if e['kind'] == 'measured')


def test_drift_names_ignore_case_and_other_mounts_are_not_comparable():
    p = sample_project('ur5e')
    q = CC.calibrated_copy(p, 'Wrist D405', (0.5, 0, 0), name='case')
    next(s for s in q.variant.sensors if s.name == 'Wrist D405').name = 'WRIST d405'
    q.calibration['sessions'][-1]['result']['cameras'] = {
        'WRIST d405': q.calibration['sessions'][-1]['result']['cameras']['Wrist D405']}
    r = p.snapshot()
    r.name = 'moved'
    s = next(s for s in r.variant.sensors if s.name == 'Wrist D405')
    s.mount.parent_frame = 'wrist_2_link'
    dr = CE.drift(CE.make_cells([q, r], open_project=p), 0)
    rows = {e['name']: e for e in next(x for x in dr['cameras'] if x['sensor'] == 'Wrist D405')['copies']}
    assert rows['case']['kind'] == 'measured' and rows['case']['offset'][0] == pytest.approx(0.5)
    assert rows['moved']['kind'] == 'not comparable' and 'wrist_2_link' in rows['moved']['why']


def test_log_so3_round_trip_and_offsets():
    rng = np.random.default_rng(3)
    for _ in range(20):
        w = rng.normal(size=3)
        w *= rng.uniform(0, math.pi * 0.999) / np.linalg.norm(w)
        assert CE.log_so3(CE.exp_so3(w)) == pytest.approx(w, abs=1e-9)
    assert np.linalg.norm(CE.log_so3(np.diag([1.0, -1.0, -1.0]))) == pytest.approx(math.pi)
    X = np.eye(4)
    Y = np.eye(4)
    Y[:3, :3] = CE.exp_so3([0, 0, math.radians(2)])
    Y[:3, 3] = [0.001, 0, -0.002]
    assert CE.offset(X, Y) == pytest.approx([1, 0, -2, 0, 0, 2])
    assert CE.share_of_tolerance(CE.offset(X, Y), (1.0, 0.5)) == pytest.approx(4.0)
    assert [CE.verdict_of(x) for x in (0.5, 1.0, 2.9, 3.5)] == ['ok', 'ok', 'adjust', 're-mount']


# ====================================================================== figures with the tolerance spread
def test_meaningful_differences_from_the_tolerance_spread():
    def ev(project, ids, ctx):                          # a figure that follows the scene camera's height (mm)
        from dexvantage.core.scenegeom import sensor_world_pose
        s = next(s for s in project.variant.sensors if s.name == SCENE)
        z = float(sensor_world_pose(project, s).position[2]) * 1000.0
        return {'t.z': {'value': z, 'unit': 'mm', 'better': 'higher', 'title': 'Height', 'status': 'ok'},
                'rig.camera_count': {'value': 2.0, 'status': 'ok'}}
    from dexvantage.core import metrics as M
    M.metric('t.z', 'Height', 'mm', 'higher', study='rig')(lambda d: None)
    try:
        p = sample_project('so100')
        far = CC.turn_camera(p, SCENE, np.zeros(3), (0, 0, 0.010))           # 10 mm: beyond ±1 mm draws
        near = CC.turn_camera(p, SCENE, np.zeros(3), (0, 0, 0.0001))         # 0.1 mm: within them
        cells = CE.make_cells([far, near], open_project=p)
        mt = CE.metrics_table(cells, ['t.z', 'rig.camera_count'], 0, draws=12, seed=2, evaluator=ev)
        r = mt['rows'][0]
        d = {x['cell']: x for x in r['differences']}
        assert d[1]['meaningful'] is True and d[1]['delta'] == pytest.approx(10.0, abs=1e-6)
        assert d[2]['meaningful'] is False and d[2]['z'] < CE.MEANINGFUL_K
        assert d[1]['sigma'] > 0 and r['tolerance'][0]['std'] > 0
        same = {x['cell']: x for x in mt['rows'][1]['differences']}
        assert same[1]['meaningful'] is False and 'do not move' in same[1]['why']
        assert mt['meaningful'] == 1 and mt['assessed'] == 4
        try:
            import dexvantage.core.stats  # noqa: F401
            assert mt['tolerance']['stats']['used'] and d[1]['test']['significant']
            assert d[1]['test']['p_adjusted'] < 0.05
            assert 'g' not in d[1]['test'] and d[1]['test']['name'].startswith('permutation')
        except ImportError:
            assert mt['tolerance']['stats'] is None
        off = CE.metrics_table(cells, ['t.z'], 0, draws=0, evaluator=ev)
        assert all(x['meaningful'] is None and x['why'] == 'no tolerance draws' for x in off['rows'][0]['differences'])
        assert 'not assessed' in off['tolerance']['rule']
    finally:
        M.METRICS.pop('t.z', None)


def test_bounds_are_not_judged_and_unknown_metrics_refused():
    vals = [1.0, 2.0]
    d = CE._difference(vals, ['at_least', None], [None, None], 'm', 0, 1, 8)
    assert d['meaningful'] is None and 'bound' in d['why'] and d['delta'] == 1.0
    with pytest.raises(StudyError, match='no metric'):
        CE.metric_ids(['no.such.metric'])
    assert 'cells.gap_max' not in CE.metric_ids('all') and 'coverage.covered' in CE.metric_ids('all')


# ====================================================================== the study, the suggestion, refusals
def test_study_settings_variants_and_refusals():
    jobs.load_studies()
    info = jobs.STUDIES['cells']
    assert info.batch == 'cells-compare' and not info.quick and not info.needs_robot
    p = sample_project('so100')
    with pytest.raises(StudyError, match='at least two cells'):
        jobs.run_study('cells', p)
    from dexvantage.core.project import Variant
    v = Variant.from_dict(p.variant.to_dict())
    v.uid, v.name = 'v2', 'Rig B'
    p.variants.append(v)
    p.analysis.studies['cells'] = {'metrics': FAST, 'cells': ['sample:ur5e']}
    doc = jobs.run_study('cells', p)
    assert [c['source'] for c in doc['cells']] == ['open project', 'variant', 'sample']
    assert doc['gap']['pairs'][0]['total'] == 0.0                     # the variant is a copy of the rig
    p.analysis.studies['cells'] = {'metrics': FAST, 'variants': False, 'cells': [{'path': '/no/such/cell.dexv'}]}
    with pytest.raises(StudyError, match='could not be opened'):
        jobs.run_study('cells', p)
    p.analysis.studies['cells'] = {'metrics': FAST, 'reference': 7}
    with pytest.raises(StudyError, match='not in the set'):
        jobs.run_study('cells', p)
    p.analysis.studies['cells'] = {'metrics': FAST, 'pooling': 'maybe'}
    with pytest.raises(StudyError, match='Pooling compares'):
        jobs.run_study('cells', p)


def test_match_suggestion_puts_the_camera_at_the_reference_pose():
    p = sample_project('so100')
    turned = CC.turn_camera(p, WRIST, (0.0, math.radians(12), 0.0), (0.004, 0.0, 0.0))
    doc = CE.compare(CE.make_cells([p], open_project=turned), reference=1, metrics=FAST)
    sg = doc['suggestion']
    assert sg['kind'] == 'match_mount' and sg['sensor'] == WRIST and sg['turn_deg'] == pytest.approx(12, abs=1e-6)
    assert sg['move_m'] > 0 and not sg['locked']
    q = turned.snapshot()
    q.variant.sensor(sg['sensor_uid']).mount.pose = Pose.from_dict(sg['pose'])
    m = next(x for x in _gap(p, q)['matches'] if x['a'] == WRIST)
    assert m['rotation_deg'] == pytest.approx(0, abs=1e-6) and m['distance_m'] == pytest.approx(0, abs=1e-9)
    turned.variant.sensor(sg['sensor_uid']).locked = True
    doc2 = CE.compare(CE.make_cells([p], open_project=turned), reference=1, metrics=FAST)
    assert doc2['suggestion']['locked']


# ====================================================================== files, exporter, batch, API, assistant
def test_files_exporter_and_html(tmp_path):
    from dexvantage.core import exporters as X
    p = sample_project('ur5e')
    man = X.export(p, str(tmp_path), ['cells'], {'cells': {'cells': ['sample:so100'], 'metrics': FAST}})
    row = man['formats']['cells']
    assert row['status'] == 'ok', row
    assert row['files'] == ['cells/cells.json', 'cells/cells_pooling.json', 'cells/cells_comparison.html']
    pool = json.loads((tmp_path / 'cells' / 'cells_pooling.json').read_text())
    assert pool['schema'] == CP.SCHEMA and pool['provenance'] and len(pool['cells']) == 2
    assert pool['cells'][1]['verdict'] == 'not poolable' and pool['cells'][1]['operations']
    page = (tmp_path / 'cells' / 'cells_comparison.html').read_text()
    assert page.startswith('<!doctype html>') and '<script' not in page and 'Provenance' in page
    assert 'Dataset pooling' in page and 'Geometric gap' in page and str(tmp_path) not in page
    full = json.loads((tmp_path / 'cells' / 'cells.json').read_text())
    assert full['schema'] == CE.SCHEMA and full['headline']['value'] > 0
    with pytest.raises(StudyError, match='at least two'):
        X.EXPORTERS['cells'].fn(p, str(tmp_path / 'x'), {'variants': False}, None)


def _run(*args, cwd=None):
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    return subprocess.run([sys.executable, '-m', 'dexvantage', *args], capture_output=True, text=True, env=env,
                          cwd=cwd or ROOT, timeout=300)


def test_batch_cells(tmp_path):
    a = sample_project('so100')
    b = CC.turn_camera(a, SCENE, (math.radians(10), 0, 0))
    b.name = 'Bench B'
    pa, pb = str(tmp_path / 'a.dexv'), str(tmp_path / 'b.dexv')
    a.save(pa)
    b.save(pb)
    st = json.dumps({'metrics': FAST})
    out, pool, page = tmp_path / 'cells.json', tmp_path / 'pool.json', tmp_path / 'cells.html'
    r = _run('--cells', pa, pb, 'sample:ur5e', '--cells-out', str(out), '--cells-pooling', str(pool),
             '--cells-html', str(page), '--cells-settings', st)
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text())
    assert [c['name'] for c in doc['cells']] == ['SO-100 sample', 'Bench B', 'UR5e + Robotiq 2F-85 sample']
    m = next(x for x in doc['gap']['pairs'][0]['matches'] if x['a'] == SCENE)
    assert m['rotation_deg'] == pytest.approx(10, abs=1e-6)
    assert doc['provenance']['interface'] == 'batch' and str(tmp_path) not in out.read_text()
    assert json.loads(pool.read_text())['schema'] == CP.SCHEMA and page.read_text().startswith('<!doctype html>')
    # a project given first is the first cell; the stored study runs with --cells-compare
    r2 = _run(pa, '--cells', pb, '--cells-settings', st)
    assert r2.returncode == 0 and json.loads(r2.stdout)['cells'][1]['name'] == 'Bench B'
    r3 = _run('--cells', pa, '--cells-settings', st)
    assert r3.returncode == 2 and 'at least two cells' in r3.stderr and 'Traceback' not in r3.stderr
    r4 = _run('--cells-html', str(page))
    assert r4.returncode == 2 and 'go with --cells' in r4.stderr
    r5 = _run('--cells', pa, pb, '--cells-settings', '[1]')
    assert r5.returncode == 2 and 'JSON object' in r5.stderr
    # the cell set stored in the project: --cells-compare (the study's own flag)
    a.analysis.studies['cells'] = {'cells': [pb], 'metrics': FAST}
    pc = str(tmp_path / 'c.dexv')
    a.save(pc)
    r6 = _run(pc, '--cells-compare', str(tmp_path / 'stored.json'))
    assert r6.returncode == 0, r6.stderr
    stored = json.loads((tmp_path / 'stored.json').read_text())
    assert stored['study'] == 'cells' and [c['source'] for c in stored['cells']] == ['open project', 'file']


def test_api_compare(tmp_path):
    import dexvantage as dv
    res = dv.cells.compare(dv.sample('so100'), 'sample:ur5e', metrics=FAST)
    assert isinstance(res, dv.api.CellComparison) and res.study == 'cells'
    assert res.headline['metric'] == 'cells.gap_max'
    assert res.headline['value'] == pytest.approx(res['headline']['value'])
    rows = res.metric_rows()
    assert rows[0]['metric'] == 'rig.camera_count' and rows[0]['SO-100 sample'] == 2.0
    assert {r['role'] for r in res.gap_rows()} == {'wrist', 'scene'}
    assert res.pooling_mapping()['schema'] == CP.SCHEMA and res.html().startswith('<!doctype html>')
    files = res.save_files(str(tmp_path))
    assert [os.path.basename(f) for f in files] == ['cells.json', 'cells_pooling.json', 'cells_comparison.html']
    with pytest.raises(dv.DexVantageError):
        dv.cells.compare()
    with pytest.raises(dv.DexVantageError):
        dv.cells.compare(dv.sample('so100'), metrics=FAST)
    same = dv.cells.compare([dv.sample('so100'), dv.sample('so100')], metrics=FAST)
    assert [c['name'] for c in same['cells']] == ['SO-100 sample', 'SO-100 sample (2)'] and same.headline['value'] == 0
    assert 'cells' in dir(dv)


def test_assistant_tool_reads_the_comparison():
    from dexvantage.assistant import tools as T
    from dexvantage.assistant.session import FileSession
    p = sample_project('so100')
    p.analysis.studies['cells'] = {'metrics': FAST, 'cells': ['sample:aloha']}
    out = T.call(FileSession(p), 'get_cells', {})
    assert 'error' not in out, out
    assert out['gap'][0]['cell'] == 'ALOHA 2 sample' and out['pooling']['cells'][0]['verdict'] == 'not poolable'
    refused = T.call(FileSession(p, allow_long=False), 'get_cells', {})
    assert 'propose_plan' in refused['error']
    assert 'cells' in T.LONG
