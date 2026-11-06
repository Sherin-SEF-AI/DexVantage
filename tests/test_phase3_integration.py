"""Integration of Phases 3, 6 and 7: what the integrator changed between the pieces, each with a test that fails on
the code before the change."""
import numpy as np

from dexvantage.core import jobs
from dexvantage.core.robot import ik, manipulability
from dexvantage.core.robot.model import RobotModel
from dexvantage.core.robot.samples import sample_project


def test_solve_batch_takes_one_start_per_target():
    """ik.solve_batch with q0 of shape (N, nq) called RobotModel.clip on the whole stack, which takes one joint vector
    ("has 14 moving joints; got 42 joint values"), so per-target starts always raised."""
    p = sample_project('ur5e')
    r = p.robots[0]
    m = RobotModel(r, p.sources)
    rng = np.random.default_rng(0)
    arm = manipulability.chain_mask(m, r.gripper_frame)          # the arm's joints; the fingers stay at home
    Q = np.tile(m.home, (3, 1))
    Q[:, arm] += rng.normal(0, 0.2, (3, int(arm.sum())))
    Q = np.clip(Q, m.lower, m.upper)
    T = np.stack([m.link_pose(q, r.gripper_frame) for q in Q])
    res = ik.solve_batch(m, r.gripper_frame, T, q0=Q, seeds=2, rng=1)
    assert res.ok.all() and res.q.shape == Q.shape


def test_samples_carry_a_grasp_set_and_a_policy():
    for name, policy in (('ur5e', 'pi0'), ('so100', 'smolvla-base')):
        p = sample_project(name)
        (gs,) = p.tasks.grasp_sets
        assert gs['robot_uid'] == p.robots[0].uid and gs['poses'] and 'sample' in gs['name']
        assert [x.library_id for x in p.policies] == [policy]
    a, b = sample_project('ur5e'), sample_project('ur5e')        # the cached poses get each project's own uids
    assert a.tasks.grasp_sets[0]['uid'] != b.tasks.grasp_sets[0]['uid']
    assert b.tasks.grasp_sets[0]['robot_uid'] == b.robots[0].uid
    assert a.tasks.grasp_sets[0]['poses'] == b.tasks.grasp_sets[0]['poses']


def test_grasp_and_vla_studies_run_on_the_ur5e_sample_as_it_opens():
    p = sample_project('ur5e')
    p.analysis.studies['grasp_visibility'] = {'limit': 12, 'sensitivity': False, 'gallery': 0,
                                              'precontact_steps': 0, 'bootstrap': 200, 'cube_n': 128}
    doc = jobs.run_study('grasp_visibility', p)
    assert doc['summary']['grasps'] == 12 and doc['summary']['cvs_pct'] is not None
    doc = jobs.run_study('vla_tokens', p)
    assert doc['policies'] and doc['policies'][0]['cameras']


def test_tasks_menu_holds_the_tasks_and_policies_pages(win):
    m = win.menus['Tasks']
    ids = [(a.data() or {}).get('id') for a in m.actions()]
    assert 'page_tasks' in ids and 'page_policies' in ids
    assert 'page_tasks' not in [(a.data() or {}).get('id') for a in win.menus['Analysis'].actions()]
    from dexvantage.ui.mainwindow import WORKSPACES
    assert all('Phase' not in WORKSPACES[k][2] for k in ('grasp', 'calibration'))


def test_handeye_import_button_detects_on_a_worker_thread(win, app, tmp_path):
    """The Import button ran the board detection of every image on the UI thread (seconds for a capture); it now runs
    as a window job and adds the session (one undo step) when it is done."""
    import time

    from conftest import pump
    from test_calib_planner import render_capture

    from dexvantage.core.calib import planner as PL
    assert win.open_sample('ur5e')
    p = win.store.project
    rig = PL.rig_from_project(p)
    plan = PL.plan(rig, PL.PlanSettings(n_poses=6))
    render_capture(p, rig, plan.doc['poses'], str(tmp_path / 'img'), str(tmp_path / 'joints.csv'))
    page = win.pages['handeye_solve']
    page.source.setCurrentIndex(page.source.findData('folder'))
    n = win.store.undo.count()
    t0 = time.perf_counter()
    assert page.import_capture(str(tmp_path / 'img'), str(tmp_path / 'joints.csv'), background=True) is None
    assert time.perf_counter() - t0 < 0.1 and 'Detecting' in page.import_note.text()
    t0 = time.time()
    while win._jobs and time.time() - t0 < 60:
        pump(app, 0.05)
    sessions = p.calibration['sessions']
    assert len(sessions) == 1 and sessions[0]['source']['detected'] == 6 and win.store.undo.count() == n + 1
    assert '6 images' in page.import_note.text()
    assert page.import_capture(str(tmp_path / 'nope'), str(tmp_path / 'joints.csv'), background=True) is None
    t0 = time.time()
    while win._jobs and time.time() - t0 < 30:
        pump(app, 0.05)
    assert 'Detecting' not in page.import_note.text() and len(p.calibration['sessions']) == 1
