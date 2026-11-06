"""Hand-eye pose planning, the coach, offline capture and the calibration records (core/calib/planner.py, coach.py,
capture.py, records.py, core/studies/handeye.py, project.calibration)."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rig_from_scene(prob, truth, info):
    from dexvantage.core.calib import planner as PL
    c, b = prob.cameras[0], prob.boards[0]
    mode = 'eye_in_hand' if c.robot else 'eye_to_hand'
    link = c.link or b.link
    return PL.Rig(mode, info['model'], link, c.camera, truth['X'][c.name], truth['Z'][b.name], b.board, 'ur5e')


# ====================================================================== the Fisher information
@pytest.mark.parametrize('kind', ['eye_in_hand', 'eye_to_hand'])
def test_view_jacobian_is_the_bundle_adjustments(kind):
    """The planner's chain-rule Jacobian (written out) against handeye.py's central differences of its residual, in
    the same parameterisation, at the truth."""
    from dexvantage.core.calib import handeye as HE
    from dexvantage.core.calib import planner as PL
    from dexvantage.core.calib import synthetic as S
    prob, truth, info = S.scene(kind, n_poses=4, noise_px=0.0, seed=3)
    rig = _rig_from_scene(prob, truth, info)
    c, b = prob.cameras[0].name, prob.boards[0].name
    model = HE._Model(prob, HE.Settings(), prob.observations, truth['X'], truth['Z'], 1.0)
    J_ba = model.jacobian(np.zeros(model.L.n))
    for j, o in enumerate(prob.observations):
        q = prob.joints[o.pose]['ur5e']
        uv, J = PL.view_jacobian(rig, q, o.ids)
        assert np.allclose(uv, o.uv, atol=1e-9)
        rows = model.L.rows[j]
        ref = np.c_[J_ba[rows, model.L.cam[c]], J_ba[rows, model.L.brd[b]]]
        assert np.max(np.abs(J - ref)) <= 1e-4 * np.max(np.abs(ref))


def test_information_inverse_is_the_covariance_the_solver_reports():
    from dexvantage.core.calib import handeye as HE
    from dexvantage.core.calib import planner as PL
    from dexvantage.core.calib import synthetic as S
    prob, truth, info = S.scene('eye_in_hand', n_poses=12, noise_px=0.0, seed=5)
    rig = _rig_from_scene(prob, truth, info)
    sigma = 0.4
    H = PL.information(rig, [(prob.joints[o.pose]['ur5e'], o.ids) for o in prob.observations], sigma)
    C, det = PL.covariance(H)
    assert det
    res = HE.solve(prob, HE.Settings(pixel_sigma=sigma))
    assert res.ok
    c, b = prob.cameras[0].name, prob.boards[0].name
    s_ba = np.r_[np.sqrt(np.diag(res.cov_X[c])), np.sqrt(np.diag(res.cov_Z[b]))]
    assert np.allclose(np.sqrt(np.diag(C)), s_ba, rtol=1e-3)


def test_singular_information_is_not_determined():
    from dexvantage.core.calib import planner as PL
    H = np.diag([1.0] * 11 + [0.0])
    C, det = PL.covariance(H)
    assert not det
    s = PL.sigma_dict(C[6:, 6:])
    assert s['z_mm'] == math.inf and math.isfinite(s['x_mm'])
    assert PL.verdict(math.inf) == ('critical', 'not determined')
    assert PL.verdict(0.5)[0] == 'good' and PL.verdict(1.5)[0] == 'warning' and PL.verdict(3)[0] == 'serious'


def test_counting_rules():
    """Corners behind the camera, out of view, seen from behind, and AprilGrid tags kept only whole."""
    from dexvantage.core.calib import planner as PL
    from dexvantage.core.calib import synthetic as S
    from dexvantage.core.calib.axxb import inv_T
    from dexvantage.core.calib.boards import AprilGrid
    prob, truth, info = S.scene('eye_in_hand', n_poses=2, noise_px=0.0, seed=1)
    rig = _rig_from_scene(prob, truth, info)
    st = PL.PlanSettings()
    q = prob.joints[0]['ur5e']
    ids, why = PL.counted(rig, q, st)
    assert why == '' and len(ids) >= 6
    T = rig.T_cam_board(q)
    # the board turned over (seen from behind)
    flip = np.eye(4)
    flip[:3, :3] = np.diag([1.0, -1.0, -1.0])
    rig2 = PL.Rig(rig.mode, rig.model, rig.link, rig.camera, rig.X, rig.Z @ flip, rig.board, 'ur5e')
    assert PL.counted(rig2, q, st)[1] in ('board seen from behind', 'board out of view')
    # far off to the side: out of view
    far = np.eye(4)
    far[:3, 3] = [5.0, 0, 0]
    rig3 = PL.Rig(rig.mode, rig.model, rig.link, rig.camera, rig.X, far @ rig.Z, rig.board, 'ur5e')
    assert PL.counted(rig3, q, st)[1] == 'board out of view'
    # an AprilGrid: whole tags only, at least 4
    g = AprilGrid(6, 5, 0.03)
    Z = inv_T(inv_T(rig.model.link_pose(q, rig.link) @ rig.X)) @ S.look_at_board(0.35, (10.0, 5.0), board=g)
    rig4 = PL.Rig(rig.mode, rig.model, rig.link, rig.camera, rig.X, Z, g, 'ur5e')
    ids4, why4 = PL.counted(rig4, q, st)
    assert why4 == '' and len(ids4) % 4 == 0 and len(ids4) >= 16
    tags = rig4.groups[ids4]
    assert all(np.sum(tags == t) == 4 for t in np.unique(tags))
    assert T.shape == (4, 4)


def test_settings_are_clamped_and_defaults_labelled():
    from dexvantage.core.calib import planner as PL
    st = PL.PlanSettings.from_dict({'n_poses': 1000, 'pixel_sigma': -1, 'criterion': 'nonsense', 'need_mm': 'x',
                                    'tilts_deg': [0, 120]})
    assert st.n_poses == 60 and st.pixel_sigma == 0.01 and st.criterion == 'weakest' and st.need_mm == PL.NEED_MM
    assert st.tilts_deg == (0.0, 80.0)
    d = PL.PlanSettings().to_dict()
    assert 'n_poses' in d['defaults'] and d['n_poses'] == 20


# ====================================================================== the plan on the sample
@pytest.fixture(scope='module')
def sample_plan():
    from dexvantage.core.calib import planner as PL
    from dexvantage.core.calib import records as RC
    from dexvantage.core.robot.samples import sample_project
    p = sample_project('ur5e')
    p.calibration['boards'].append(RC.default_board_record(p, 'eye_in_hand', p.robots[0].uid))
    rig = PL.rig_from_project(p)
    pl = PL.plan(rig, PL.PlanSettings(n_poses=15))
    return p, rig, pl


def test_plan_poses_are_reachable_collision_free_and_see_the_board(sample_plan):
    from dexvantage.core.calib import planner as PL
    from dexvantage.core.robot import collision as CO
    p, rig, pl = sample_plan
    doc = pl.doc
    assert doc['mode'] == 'eye_in_hand' and doc['link'] == 'wrist_3_link' and len(doc['poses']) == 15
    st = PL.PlanSettings()
    m = rig.model
    r0 = CO.obstacle_distance(p, rig.robot_uid, m.home, st.obstacle_margin_m)
    base = {(a, b): d for a, b, d in r0.pairs} if r0.in_collision else {}
    for ps in doc['poses']:
        q = np.asarray(ps['q'])
        assert np.all(q >= m.lower - 1e-9) and np.all(q <= m.upper + 1e-9)
        assert not CO.self_collision(m, q).in_collision
        r = CO.obstacle_distance(p, rig.robot_uid, q, st.obstacle_margin_m)
        assert all(d >= st.obstacle_margin_m or d >= base.get((a, b), 1e9) - 1e-3 for a, b, d in r.pairs)
        # the pose the IK found reaches the hemisphere target: the camera is as far from the board as planned
        C = rig.camera_world(q)
        from dexvantage.core.calib.synthetic import board_centre
        bc = rig.Z[:3, :3] @ board_centre(rig.board) + rig.Z[:3, 3]
        assert abs(np.linalg.norm(C[:3, 3] - bc) - ps['distance_m']) < 1e-3
        ids = np.asarray(ps['ids'])
        assert len(ids) >= 6 and ps['corners'] == len(ids)
        T = rig.T_cam_board(q)
        uv, ok = rig.camera.project(rig.points[ids] @ T[:3, :3].T + T[:3, 3])
        assert ok.all()
        assert PL.occlusion_visible(rig, q, ids, st).all()      # the hidden corners were taken out
    assert len({ps['candidate'] for ps in doc['poses']}) == 15
    json.dumps(doc)


def test_plan_predicts_sigma_and_the_next_gain(sample_plan):
    _p, _rig, pl = sample_plan
    doc = pl.doc
    sx = doc['sigma']['X']
    assert all(v is not None and 0 < v < 1 for v in sx.values())
    assert doc['determined'] and doc['status'] == 'good' and doc['worst']['ratio'] < 1
    nxt = doc['next']
    assert nxt is not None and nxt['worst_after'] <= nxt['worst_before'] and nxt['worst_gain_pct'] >= 0
    assert all(ps['worst_increase_pct'] is None or ps['worst_increase_pct'] >= -1e-6 for ps in doc['poses'])
    assert doc['covariance_order'][0] == 'X rx_rad' and len(doc['covariance']) == 12
    assert 'lower bound' in doc['assumptions']
    assert any('default' in n for n in doc['notes'])


def test_greedy_d_optimal_beats_random_sets(sample_plan):
    from dexvantage.core.calib import planner as PL
    _p, _rig, pl = sample_plan
    cs = pl.candidates
    st = PL.PlanSettings()
    prior = np.eye(12) / PL.PRIOR_SIGMA ** 2
    ld_plan = np.linalg.slogdet(pl.H + prior)[1]
    rng = np.random.default_rng(0)
    pool = np.array(sorted(cs.checked))           # the candidates whose occlusion was checked
    worse = 0
    for _ in range(20):
        pick = rng.choice(len(cs), 15, replace=False)
        H = sum(cs.A[c] for c in pick) / st.pixel_sigma ** 2
        worse += np.linalg.slogdet(H + prior)[1] < ld_plan
    assert worse == 20 and len(pool) >= 15


def test_more_poses_pin_the_camera_better(sample_plan):
    from dexvantage.core.calib import planner as PL
    _p, rig, pl = sample_plan
    st = PL.PlanSettings(n_poses=6)
    few, H6 = PL.select(rig, pl.candidates, 6, st)
    r6 = PL.worst(PL.sigmas(H6)[0], st.need_mm, st.need_deg)[0]
    r15 = pl.doc['worst']['ratio']
    assert len(few) == 6 and r6 > r15


def test_tour_starts_near_the_current_pose(sample_plan):
    from dexvantage.core.calib import planner as PL
    _p, rig, pl = sample_plan
    Q = np.array([ps['q'] for ps in pl.doc['poses']])
    order = PL.tour(Q, rig.q_hold, rig.chain)
    assert order[0] == 0 and sorted(order) == list(range(len(Q)))


def test_unreachable_board_is_explained():
    from dexvantage.core.calib import planner as PL
    from dexvantage.core.calib import records as RC
    from dexvantage.core.frames import Pose
    from dexvantage.core.robot.samples import sample_project
    p = sample_project('ur5e')
    rec = RC.default_board_record(p, 'eye_in_hand', p.robots[0].uid)
    rec['mount']['pose'] = Pose([4.0, 4.0, 0.0], [1.0, 0.0, 0.0, 0.0]).to_dict()
    p.calibration['boards'].append(rec)
    rig = PL.rig_from_project(p)
    with pytest.raises(ValueError) as e:
        PL.plan(rig, PL.PlanSettings(n_poses=5, tilts_deg=(0.0,), rolls_deg=(0.0,)))
    assert 'out of reach' in str(e.value) and 'Traceback' not in str(e.value)


def test_rig_refusals_are_plain():
    from dexvantage.core.calib import planner as PL
    from dexvantage.core.calib import records as RC
    from dexvantage.core.jobs import StudyError
    from dexvantage.core.robot.samples import sample_project
    p = sample_project('ur5e')
    overhead = p.variant.sensors[1]
    rec = RC.default_board_record(p, 'eye_in_hand', p.robots[0].uid)     # a board in the world
    p.calibration['boards'].append(rec)
    with pytest.raises(StudyError, match='both fixed'):
        PL.rig_from_project(p, overhead.uid, rec['uid'])
    p.variant.sensors = []
    with pytest.raises(StudyError, match='needs a camera'):
        PL.rig_from_project(p)


def test_eye_to_hand_plan_with_the_overhead_camera():
    from dexvantage.core.calib import planner as PL
    from dexvantage.core.robot.samples import sample_project
    p = sample_project('ur5e')
    overhead = p.variant.sensors[1]
    rig = PL.rig_from_project(p, overhead.uid)
    assert rig.mode == 'eye_to_hand' and rig.board_default and rig.link == p.robots[0].gripper_frame
    pl = PL.plan(rig, PL.PlanSettings(n_poses=8, rolls_deg=(-45.0, 0.0, 45.0)))
    assert len(pl.doc['poses']) == 8 and pl.doc['determined']
    assert pl.doc['board']['frame'].startswith('board frame in link')


# ====================================================================== the coach
def test_coach_advises_the_largest_reduction(sample_plan):
    from dexvantage.core.calib import coach as CO
    from dexvantage.core.calib import planner as PL
    _p, rig, pl = sample_plan
    st = PL.PlanSettings(occlusion=False)
    cs = PL.candidates(rig, st)
    taken = pl.doc['poses'][:4]
    adv = CO.advise(rig, taken, st, cs)
    nx = adv['next']
    assert nx is not None and adv['taken'] == 4 and adv['advice']
    # brute force: no other candidate leaves the camera's worst axis lower
    views = CO.taken_views(rig, taken, st)
    H = PL.information(rig, views, st.pixel_sigma)
    dup = CO._duplicates(rig, cs, views)
    best = min(PL.worst(PL.sigmas(H + cs.A[c] / st.pixel_sigma ** 2)[0], st.need_mm, st.need_deg)[0]
               for c in range(len(cs)) if not dup[c])
    assert nx['worst_after'] <= best * (1 + 1e-6) + 1e-9
    assert nx['worst_after'] < adv['worst']['ratio']
    assert nx['moves'] and {'joint', 'from_deg', 'to_deg', 'delta_deg'} <= set(nx['moves'][0])
    json.dumps(adv)


def test_coach_skips_duplicates_and_says_enough(sample_plan):
    from dexvantage.core.calib import coach as CO
    from dexvantage.core.calib import planner as PL
    _p, rig, pl = sample_plan
    st = PL.PlanSettings(occlusion=False)
    cs = PL.candidates(rig, st)
    taken = [{'q': list(cs.q[0])}]
    dup = CO._duplicates(rig, cs, CO.taken_views(rig, taken, st))
    assert dup[0]
    adv = CO.advise(rig, pl.doc['poses'], st, cs)
    assert adv['enough'] and adv['advice'].startswith('Enough')
    first = CO.advise(rig, [], st, cs)
    assert first['taken'] == 0 and first['next'] is not None and first['advice'].startswith('Start')


# ====================================================================== capture
def test_joint_states_csv_json_and_ros_style():
    from dexvantage.core.calib import capture as CAP
    names = ['a', 'b', 'c']
    js = CAP.read_joint_states('t,a,b,extra\n2.0,0.2,0.3,9\n1.0,0.1,0.2,9\n', names, [0, 0, 7.0])
    assert np.allclose(js.t, [1.0, 2.0]) and np.allclose(js.Q, [[0.1, 0.2, 7.0], [0.2, 0.3, 7.0]])
    assert any('held at home' in n for n in js.notes) and any('ignored' in n for n in js.notes)
    js2 = CAP.read_joint_states(json.dumps({'t': [0, 1], 'a': [10, 20], 'b': [0, 0], 'c': [0, 0]}), names,
                                unit='deg')
    assert np.allclose(js2.Q[1, 0], math.radians(20))
    js3 = CAP.read_joint_states(json.dumps([{'t': 0.5, 'name': ['c', 'a'], 'position': [1.0, 2.0]}]), names)
    assert np.allclose(js3.Q[0], [2.0, 0.0, 1.0])
    with pytest.raises(CAP.CaptureError, match='time column'):
        CAP.read_joint_states('a,b\n1,2\n', names)
    with pytest.raises(CAP.CaptureError, match='None of the joint'):
        CAP.read_joint_states('t,x\n1,2\n', names)
    with pytest.raises(CAP.CaptureError, match='not a number'):
        CAP.read_joint_states('t,a\n1,abc\n', names)


def test_image_times_and_matching(tmp_path):
    from dexvantage.core.calib import capture as CAP
    assert CAP.time_from_name('pose_07_1730900000.250.png') == pytest.approx(1730900000.25)
    assert CAP.time_from_name('frame_1730900000250.png') == pytest.approx(1730900000.25)
    assert CAP.time_from_name('img_1730900000250000000.png') == pytest.approx(1730900000.25)
    assert CAP.time_from_name('pose_07.png') is None
    js = CAP.read_joint_states('t,a\n10.00,0\n10.01,0\n20.0,0.0\n20.5,1.0\n30.0,0\n', ['a'])
    pairs, notes = CAP.match_times([('x.png', 10.004), ('y.png', 20.03), ('z.png', 25.0), ('w.png', 20.49)], js, 0.05)
    assert pairs[0][1] == 0 and abs(pairs[0][2] + 0.004) < 1e-9
    assert pairs[1][1] == 2
    assert pairs[2][1] is None and 'no joint sample within 50 ms' in pairs[2][4]
    assert pairs[3][1] == 3 and 'moving' in pairs[3][4]
    for n in ('b.png', 'a.png'):
        (tmp_path / n).write_bytes(b'')
    (tmp_path / 'images.csv').write_text('file,t\na.png,1.5\nb.png,2.5\n')
    times, how = CAP.image_times(str(tmp_path))
    assert times == [('a.png', 1.5), ('b.png', 2.5)] and 'images.csv' in how
    (tmp_path / 'images.csv').unlink()
    times, how = CAP.image_times(str(tmp_path))
    assert how == 'no image times'
    js2 = CAP.read_joint_states('t,a\n0,0\n1,0\n', ['a'])
    pairs, notes = CAP.match_times(times, js2)
    assert [p[1] for p in pairs] == [0, 1] and 'paired' in notes[0]
    with pytest.raises(CAP.CaptureError, match='cannot be paired'):
        CAP.match_times(times + [('c.png', None)], js2)


def test_ros2_and_lerobot_arrive_in_phase_8():
    from dexvantage.core.calib import capture as CAP
    for k in ('ros2', 'lerobot'):
        with pytest.raises(CAP.CaptureError, match='Phase 8'):
            CAP.check_source(k)
    CAP.check_source('folder')
    assert 'Phase 8' in CAP.SOURCE_LABELS['ros2'] and 'Phase 8' in CAP.SOURCE_LABELS['lerobot']


def render_capture(project, rig, poses, folder, joints_path, noise=1.0):
    """Board images rendered at the poses and their joint_states CSV (times in the file names)."""
    import cv2

    from dexvantage.core.calib import synthetic as S
    os.makedirs(folder, exist_ok=True)
    rows = ['t,' + ','.join(rig.model.joint_names)]
    for k, ps in enumerate(poses):
        q = np.asarray(ps['q'])
        img = S.render_view(rig.board, rig.camera, rig.T_cam_board(q), supersample=2, noise=noise, seed=k)
        t = 100.0 + 5 * k
        cv2.imwrite(os.path.join(folder, f'pose_{k:02d}_{t:.3f}.png'), img)
        rows.append(f'{t + 0.004},' + ','.join(f'{v:.10f}' for v in q))
    with open(joints_path, 'w') as fh:
        fh.write('\n'.join(rows) + '\n')


def test_capture_solve_and_coach_end_to_end(sample_plan, tmp_path):
    from dexvantage.core import jobs
    from dexvantage.core.calib import capture as CAP
    from dexvantage.core.calib import records as RC
    p, rig, pl = sample_plan
    p = p.snapshot()
    render_capture(p, rig, pl.doc['poses'][:9], str(tmp_path / 'img'), str(tmp_path / 'joints.csv'))
    sensor = p.variant.sensors[0]
    sess = CAP.capture_folder(p, sensor, str(tmp_path / 'img'), str(tmp_path / 'joints.csv'))
    assert sess['source']['detected'] == 9 and sess['mode'] == 'eye_in_hand' and sess['link'] == 'wrist_3_link'
    assert all(abs(ps['dt_s'] - 0.004) < 1e-6 for ps in sess['poses'])
    assert str(tmp_path) not in json.dumps(sess)                    # no folders stored, only names
    sess.pop('board', None)
    p.calibration['sessions'].append(sess)
    doc = jobs.run_study('handeye_solve', p)
    assert doc['ok'] and doc['status'] == 'good' and len(doc['views']) == 9
    ap = doc['apply']
    assert ap['change_mm'] < 0.2 and ap['change_deg'] < 0.02 and ap['parent_frame'] == 'wrist_3_link'
    from dexvantage.core.frames import Pose
    body = Pose.from_dict(ap['pose']).matrix()
    assert np.allclose(body[:3, 3], sensor.mount.pose.position, atol=3e-4)
    assert doc['coach'] and doc['coach']['next'] and doc['coach']['taken'] == 9
    json.dumps(doc)
    # the planned session refuses to solve, in words
    plan_sess = RC.session_record(sensor, RC.sensor_imager(sensor), 'eye_in_hand', '', rig.robot_uid, rig.link,
                                  [{'q': pl.doc['poses'][0]['q']}], {'kind': 'plan'})
    p.calibration['sessions'].append(plan_sess)
    p.analysis.studies['handeye_solve'] = {'session': plan_sess['uid']}
    with pytest.raises(jobs.StudyError, match='is a plan'):
        jobs.run_study('handeye_solve', p)


def test_solve_without_a_session_is_explained():
    from dexvantage.core import jobs
    from dexvantage.core.robot.samples import sample_project
    with pytest.raises(jobs.StudyError, match='no capture session'):
        jobs.run_study('handeye_solve', sample_project('ur5e'))


# ====================================================================== records in the project
def test_calibration_section_round_trips_and_validates(tmp_path):
    from dexvantage.core import project as P
    from dexvantage.core import provenance as PV
    from dexvantage.core.calib import records as RC
    from dexvantage.core.robot.samples import sample_project
    p = sample_project('ur5e')
    d0 = PV.project_digest(p)['sha256']
    assert p.section('calibration') == {'boards': [], 'sessions': []}
    rec = RC.default_board_record(p, 'eye_in_hand', p.robots[0].uid)
    s = p.variant.sensors[0]
    sess = RC.session_record(s, RC.sensor_imager(s), 'eye_in_hand', rec['uid'], p.robots[0].uid, 'wrist_3_link',
                             [{'name': 'a', 'q': [0.0] * 14, 'ids': [1, 2], 'uv': [[1, 2], [3, 4]], 'dt_s': 0.01}],
                             {'kind': 'folder'})
    p.restore('calibration', {'boards': [rec], 'sessions': [sess], 'junk': 1})
    assert set(p.calibration) == {'boards', 'sessions'}
    assert PV.project_digest(p)['sha256'] != d0
    path = str(tmp_path / 'x.dexv')
    p.save(path)
    q = P.Project.load(path)
    assert q.calibration == p.calibration
    jsonschema = pytest.importorskip('jsonschema')
    with open(os.path.join(ROOT, 'docs', 'project.schema.json')) as fh:
        schema = json.load(fh)
    jsonschema.validate(json.loads(json.dumps(q.to_dict())), schema)
    assert 'calibration' in P.SECTIONS
    p.restore('calibration', None)
    assert p.calibration == {'boards': [], 'sessions': []} and PV.project_digest(p)['sha256'] == d0


def test_default_board_records_lie_where_they_say():
    from dexvantage.core.calib import records as RC
    from dexvantage.core.calib.synthetic import board_centre
    from dexvantage.core.robot.model import get_model
    from dexvantage.core.robot.samples import sample_project
    p = sample_project('ur5e')
    r = p.robots[0]
    rec = RC.default_board_record(p, 'eye_in_hand', r.uid)
    assert rec['default'] and 'a default' in rec['where']
    rb, link, Z = RC.board_mount(rec)
    b = RC.board_of(rec)
    c = Z[:3, :3] @ board_centre(b) + Z[:3, 3]
    v = p.workspace.volumes[0]
    assert rb == '' and np.allclose(c[:2], v.center[:2]) and abs(c[2]) < 1e-12
    assert np.allclose(Z[:3, :3] @ [0, 0, -1.0], [0, 0, 1])           # printed face (ChArUco -z) up
    held = RC.default_board_record(p, 'eye_to_hand', r.uid)
    rb, link, Z = RC.board_mount(held)
    assert rb == r.uid and link == r.gripper_frame
    m = get_model(p, r.uid)
    pad = (np.linalg.inv(m.link_pose(m.home, link)) @ m.link_pose(m.home, r.fingertip_frames[0]))[2, 3]
    assert (Z[:3, :3] @ board_centre(b) + Z[:3, 3])[2] == pytest.approx(pad + RC.HELD_CLEARANCE_M)


def test_mount_pose_from_X_round_trips():
    from dexvantage.core.calib import records as RC
    from dexvantage.core.robot.samples import sample_project
    p = sample_project('ur5e')
    s = p.variant.sensors[0]
    im = RC.sensor_imager(s)
    _r, _l, X = RC.camera_mount(p, s, im)
    body = RC.mount_pose_from_X(s, im, X)
    assert np.allclose(body.matrix(), s.mount.pose.matrix(), atol=1e-12)


# ====================================================================== study registry, batch, validation
def test_studies_are_registered_with_batch_flags():
    from dexvantage.core import jobs
    jobs.load_studies()
    assert jobs.STUDIES['handeye_plan'].batch == 'handeye-plan' and not jobs.STUDIES['handeye_plan'].quick
    assert jobs.STUDIES['handeye_solve'].batch == 'handeye-solve' and jobs.STUDIES['handeye_solve'].needs_robot


def test_batch_flag_writes_the_plan(tmp_path):
    out = tmp_path / 'plan.json'
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    r = subprocess.run([sys.executable, '-m', 'dexvantage', '--sample', 'ur5e', '--handeye-plan', str(out)],
                       capture_output=True, text=True, env=env, cwd=ROOT, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    doc = json.loads(out.read_text())
    assert doc['study'] == 'handeye_plan' and 'provenance' in doc and len(doc['poses']) == 20
    assert str(tmp_path) not in out.read_text()


def test_planner_validation_check_passes_quick():
    from dexvantage.validation import suite
    suite.load_checks()
    spec = next(s for s in suite.CHECKS if s.id == 'handeye.planner_vs_ba')
    assert spec.model == 'handeye' and spec.cost == 'study'
    ctx = suite.Ctx(None, None, None, suite.ValidationSettings(quick=True))
    try:
        c = suite.run_one(spec, ctx)
    finally:
        ctx.close()
    assert c.passed, (c.error, c.exception, c.detail)
    assert c.detail['shape_ln_ratio'] < 0.01
