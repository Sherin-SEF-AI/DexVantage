"""Spec-completeness review of Phases 3, 6 and 7 (spec §7.3, §7.5, §7.7, §8.1, §13): grasps taken where the gripper
closes in a trajectory, the brackets' clearance to the project's cables, several hand-eye sessions in one bundle
adjustment (a fixed camera located through a board a wrist camera sees), and the ghosted swept volume of a
trajectory on the Tasks page."""
import json

import numpy as np
import pytest

from conftest import pump


def _run_check(cid):
    from dexvantage.validation import suite
    spec = next(s for s in suite.specs() if s.id == cid)
    ctx = suite.Ctx(None, None, None, suite.ValidationSettings(quick=True))
    try:
        return suite.run_one(spec, ctx)
    finally:
        ctx.close()


# ====================================================================== grasps where the gripper closes (§7.3)
@pytest.fixture(scope='module')
def pick():
    from dexvantage.core.robot.model import get_model
    from dexvantage.core.robot.samples import sample_project
    from dexvantage.validation.checks_grasps import scripted_pick
    p = sample_project('ur5e')
    p.tasks.grasp_sets = []
    r = p.robots[0]
    m = get_model(p, r.uid)
    tr, want = scripted_pick(m)
    tr['robot_uid'] = r.uid
    p.tasks.trajectories = [tr]
    return p, m, tr, want


def test_closure_events_rule():
    from dexvantage.core import grasps as GR
    mm = 1e-3
    # open, close 20 mm with a pause in the middle, hold, open, a 2 mm twitch, close 6 mm
    d = np.array([80, 80, 75, 70, 70, 65, 60, 60, 60, 80, 80, 78, 78, 80, 74, 74], float) * mm
    ev = GR.closure_events(d)
    assert [(a, b) for a, b, _ in ev] == [(1, 6), (13, 14)]
    assert ev[0][2] == pytest.approx(20 * mm) and ev[1][2] == pytest.approx(6 * mm)
    assert GR.closure_events(d, close_m=7 * mm) == [ev[0]]
    assert GR.closure_events(np.full(5, 0.05)) == []
    # noise below the flat band is a pause, not a move
    assert GR.closure_events(0.05 + 1e-5 * np.sin(np.arange(50))) == []


def test_grasps_where_the_gripper_closes(pick):
    from dexvantage.core import grasps as GR
    p, m, tr, want = pick
    gs = GR.from_trajectory(p, tr['uid'], mode='closures')
    assert gs['source'] == 'trajectory' and len(gs['poses']) == 2
    t = tr['t']
    assert [x['t'] for x in gs['poses']] == [t[a] for a, _b in want]
    assert [e['t_end'] for e in gs['settings']['events']] == [t[b] for _a, b in want]
    assert gs['settings']['defaults'] == ['close_mm', 'approach_mm'] and gs['settings']['closures'] == 2
    geom = GR.gripper_geometry(p, gs['robot_uid'])
    Q = np.asarray(tr['q'], float)
    i = m.link_index(geom['frame'])
    T = m.fk_batch(Q)[:, i]
    tcp = T[:, :3, :3] @ geom['tcp'] + T[:, :3, 3]
    for x, (a, _b) in zip(gs['poses'], want):
        assert np.allclose(GR.grasp_matrix(x), T[a], atol=1e-9)
        # the approach is the way the tool centre came in over its last 20 mm, written out here
        k = max(j for j in range(a + 1) if np.linalg.norm(tcp[j] - tcp[a]) >= 0.02)
        v = (tcp[a] - tcp[k]) / np.linalg.norm(tcp[a] - tcp[k])
        assert np.allclose(x['approach'], v, atol=1e-9)
    json.dumps(gs)
    # every Nth sample is still there, and the defaults can be changed
    assert len(GR.from_trajectory(p, tr['uid'], every=10)['poses']) == 9
    big = GR.from_trajectory(p, tr['uid'], mode='closures', close_mm=60.0)    # drops 74 and 44 mm
    assert len(big['poses']) == 1 and big['settings']['defaults'] == ['approach_mm']


def test_trajectory_closures_explain_what_they_cannot_do(pick):
    from dexvantage.core import grasps as GR
    p, m, tr, _w = pick
    q = p.snapshot()
    still = dict(tr, uid='still', q=[list(m.home)] * len(tr['t']))
    q.tasks.trajectories = [still]
    with pytest.raises(GR.GraspError, match='never closes'):
        GR.from_trajectory(q, 'still', mode='closures')
    with pytest.raises(GR.GraspError, match='mode must be'):
        GR.from_trajectory(q, 'still', mode='sometimes')
    with pytest.raises(GR.GraspError, match='closing distance'):
        GR.from_trajectory(q, 'still', mode='closures', close_mm=0.0)
    q.robots[0].fingertip_frames = []
    q.robots[0].contact_pads = []
    with pytest.raises(GR.GraspError, match='two or more fingertips'):
        GR.from_trajectory(q, 'still', mode='closures')


def test_trajectory_closures_check_passes():
    r = _run_check('grasp.trajectory_closures')
    assert r.passed, (r.error, r.detail, r.exception)


# ====================================================================== brackets against the cables (§7.7)
def test_bracket_clearance_reads_the_project_cables():
    from dexvantage.core import brackets as BR
    from dexvantage.core import jobs
    from dexvantage.core.robot.samples import sample_project
    from dexvantage.core.studies import cables as CS
    p = sample_project('ur5e')
    assert CS.cable_polylines(p) == [] and BR.cable_polylines(p) == []
    p.cables = CS.apply_suggested([], jobs.run_study('cables', p))
    lines = CS.cable_polylines(p)
    assert len(lines) == 1 and lines[0]['points'].shape[1] == 3 and lines[0]['radius_m'] == 0.0
    # the polyline runs from the first attachment to the last, in the world
    from dexvantage.core.robot.model import get_model
    c = p.cables[0]
    m = get_model(p, c['attachments'][0]['robot_uid'])
    T = m.fk(m.home)
    for a, P in ((c['attachments'][0], lines[0]['points'][0]), (c['attachments'][-1], lines[0]['points'][-1])):
        Tl = T[a['link']]
        assert np.allclose(P, Tl[:3, :3] @ a['point'] + Tl[:3, 3], atol=1e-9)
    assert len(BR.cable_polylines(p)) == 1
    sen = next(s for s in p.variant.sensors if s.mount.on_link)
    sen.mount.bracket = 'flange'
    doc = jobs.run_study('brackets', p)
    row = next(s for s in doc['sensors'] if s.get('uid') == sen.uid)
    cab = row['clearance']['cables']
    assert cab['checked'] == 1 and cab['min_mm'] is not None and cab['cable'] == c['name']


# ====================================================================== several sessions, one adjustment (§8.1)
@pytest.fixture(scope='module')
def joint():
    from dexvantage.validation.checks_handeye import joint_sessions_project
    return joint_sessions_project(10, 0.2, 3, True)


def test_two_sessions_solve_together(joint):
    from dexvantage.core import jobs
    from dexvantage.core.calib import capture as CAP
    p, truth = joint
    p = p.snapshot()
    ss = p.calibration['sessions']
    prob, info = CAP.sessions_problem(p, ss)
    assert len(prob.cameras) == 2 and len(prob.boards) == 1 and len(prob.joints) == 10      # poses merged
    assert {c['mode'] for c in info['cameras'].values()} == {'eye_in_hand', 'static'}
    # alone, the overhead session cannot be solved: nothing moves between it and its board
    p.analysis.studies['handeye_solve'] = {'session': ss[1]['uid'], 'coach': False}
    alone = jobs.run_study('handeye_solve', p)
    assert not alone['ok'] and 'fixed' in alone['message']
    p.analysis.studies['handeye_solve'] = {'sessions': 'all'}
    doc = jobs.run_study('handeye_solve', p)
    assert doc['ok'] and doc['joint'] and len(doc['cameras']) == 2 and len(doc['applies']) == 2
    json.dumps(doc)
    for c in doc['cameras']:
        X = np.asarray(doc['result']['cameras'][c['name']]['matrix'], float)
        e = np.abs(X[:3, 3] - truth[c['sensor_uid']][:3, 3]) * 1000
        sig = np.array([c['sigma'][k] for k in ('x_mm', 'y_mm', 'z_mm')])
        assert (e <= 5 * sig).all(), (c['name'], e, sig)
    over = next(c for c in doc['cameras'] if c['mode'] == 'static')
    assert over['apply']['change_mm'] > 5.0                             # it started 10 mm off
    assert {v['session'] for v in doc['views']} == {'Wrist', 'Overhead'}
    assert any('cannot help' in n for n in doc['notes'])
    assert doc['coach'] and doc['coach']['camera'] != over['name']
    # the explained refusals
    p.analysis.studies['handeye_solve'] = {'sessions': [ss[0]['uid']]}
    with pytest.raises(jobs.StudyError, match='two or more'):
        jobs.run_study('handeye_solve', p)
    p.analysis.studies['handeye_solve'] = {'sessions': [ss[0]['uid'], 'gone']}
    with pytest.raises(jobs.StudyError, match='not in the project'):
        jobs.run_study('handeye_solve', p)


def test_a_fixed_camera_no_robot_camera_links_is_named(joint):
    from dexvantage.core import jobs
    p, _t = joint
    p = p.snapshot()
    ss = p.calibration['sessions']
    other = dict(p.calibration['boards'][0], uid='other-board', name='Second board')
    p.calibration['boards'].append(other)
    ss[1]['board_uid'] = 'other-board'
    p.analysis.studies['handeye_solve'] = {'sessions': 'all', 'coach': False}
    doc = jobs.run_study('handeye_solve', p)
    assert not doc['ok'] and 'Overhead D435' in doc['message'] and 'not linked' in doc['message']


def test_joint_sessions_check_passes():
    r = _run_check('handeye.sessions_joint')
    assert r.passed, (r.error, r.detail, r.exception)


def test_solve_page_applies_every_camera_in_one_step(win, app):
    from dexvantage.core.calib import records as RC
    from dexvantage.validation.checks_handeye import joint_sessions_project
    assert win.open_sample('ur5e')
    p0, _t = joint_sessions_project(8, 0.2, 5, True, project=win.store.project.snapshot())
    over_uid = next(s.uid for s in p0.variant.sensors if not s.mount.on_link)

    def fn(p):
        p.calibration = RC.copy_calibration(p0)
        p.variant.sensor(over_uid).mount.pose = p0.variant.sensor(over_uid).mount.pose
    win.store.edit(['calibration', 'variant'], fn, 'Sessions, the overhead camera moved')
    pump(app, 0.02)
    page = win.pages['handeye_solve']
    win.open_page('handeye_solve')
    pump(app, 0.05)
    page.together.setChecked(True)
    assert not page.session_box.isEnabled()
    page.run()
    assert win.ctx.analysis.wait(180)
    pump(app, 0.1)
    doc = page.doc
    assert doc and doc['ok'] and doc['joint'], doc
    assert page.sig_tree.topLevelItemCount() == 3                     # two cameras and the board
    p = win.store.project
    before = {s.uid: s.mount.pose.to_dict() for s in p.variant.sensors}
    n = win.store.undo.count()
    assert page.b_apply.isEnabled() and page.apply()
    assert win.store.undo.count() == n + 1
    p = win.store.project
    assert all(p.variant.sensor(a['sensor_uid']).mount.pose.to_dict() != before[a['sensor_uid']]
               for a in doc['applies'])
    assert all(RC.find_session(p, u)['result']['applied'] for u in doc['session']['uids'])
    win.store.undo.undo()
    p = win.store.project
    assert {s.uid: s.mount.pose.to_dict() for s in p.variant.sensors} == before


# ====================================================================== the swept volume (§13)
def test_swept_ghosts_follow_the_trajectory(pick):
    from dexvantage.core.robot import swept as SW
    p, m, tr, _w = pick
    tri_l, link = SW.ghost_mesh(m)
    assert 0 < len(tri_l) < m.n_triangles / 5 and SW.is_cached(m)
    g = SW.swept_ghosts(m, tr, n=4)
    assert g['t'] == [tr['t'][0], *np.linspace(tr['t'][0], tr['t'][-1], 4)[1:-1].tolist(), tr['t'][-1]]
    assert g['tri'].shape == (4 * len(tri_l), 3, 3) and g['triangles_per_pose'] == len(tri_l)
    # each ghost is its link's decimated mesh carried by FK at that time (the last ghost: the last sample)
    q = m.clip(np.asarray(tr['q'][-1], float))
    T = m.fk_array(q)
    k = int(link[0])
    sel = np.nonzero(link == k)[0]
    want = tri_l[sel].reshape(-1, 3) @ T[k, :3, :3].T + T[k, :3, 3]
    got = g['tri'][3 * len(tri_l) + sel].reshape(-1, 3)
    assert np.allclose(got, want, atol=1e-5)
    # within a cell of the full meshes' bounds at that pose
    lo, hi = m.bounds(q)
    V = g['tri'][3 * len(tri_l):].reshape(-1, 3)
    assert (V >= lo - 0.011).all() and (V <= hi + 0.011).all()


def test_tasks_page_trajectory_modes_and_swept_volume(win, app, pick):
    from dexvantage.core import grasps as GR
    _p, _m, tr, _w = pick
    assert win.open_sample('ur5e')
    r = win.store.project.robots[0]
    win.store.edit('tasks', lambda p: p.tasks.trajectories.append(dict(tr, robot_uid=r.uid)), 'Add a trajectory')
    pump(app, 0.02)
    win.open_page('tasks')
    pump(app, 0.02)
    page = win.pages['tasks']
    assert page.traj_mode.value() == 'closures' and not page.every.isEnabled()
    uid = page.add_from_trajectory(tr['uid'])
    gs = GR.get_set(win.store.project, uid)
    assert len(gs['poses']) == 2 and 'closes 2 times' in page.msg.text()
    page.traj_mode.buttons['every'].click()
    assert page.every.isEnabled()
    # the swept volume: drawn (the first time through a background job), cleared when unticked
    page.swept.setChecked(True)
    for _ in range(200):
        pump(app, 0.02)
        if page._ghost:
            break
    assert page._ghost and any(getattr(o, 'tri', None) is not None and len(o.tri) for o in page.overlays(win.viewport))
    a = page._ghost[0].colors[:, 3]
    assert 0.0 < a.min() < a.max() < 1.0
    page.swept.setChecked(False)
    assert not page._ghost
