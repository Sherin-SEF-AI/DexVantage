"""The hand-eye pages (ui/pages/handeye.py) on the UR5e sample: plan, save the plan, record (import a folder of rendered
board images and a joint_states file), solve, apply the calibrated mount (one undo step each), the coach, the
overlays and the explained failures."""
import json
import os

import numpy as np
import pytest

from conftest import pump
from test_calib_planner import render_capture


@pytest.fixture
def ur5e(win, app):
    assert win.open_sample('ur5e')
    win.store.undo.setClean()
    win.ctx.analysis.wait(5)
    pump(app, 0.02)
    return win


def _run(win, app, page, timeout=180):
    page.run()
    assert win.ctx.analysis.wait(timeout)
    pump(app, 0.1)


def test_pages_are_registered_in_the_calibrate_menu(win):
    from dexvantage.ui.pages import PAGES
    for pid in ('handeye_plan', 'handeye_solve'):
        assert pid in win.pages
        assert PAGES[pid].workspace == 'calibration' and PAGES[pid].menu == 'Calibrate'
        assert win.actions_by_id[pid].isEnabled()
    win.workspace('calibration')
    assert win.bottom.tabs.currentWidget() is win.pages['handeye_plan']
    win.actions_by_id['handeye_solve'].trigger()
    assert win.current_page() is win.pages['handeye_solve']


def test_plan_page_plans_saves_and_previews(app, ur5e, tmp_path):
    win = ur5e
    page = win.pages['handeye_plan']
    win.open_page('handeye_plan')
    pump(app, 0.05)
    assert page.head.status.text.text() == 'not run yet'
    assert page.sensor_box.count() == 2 and 'Wrist' in page.sensor_box.currentText()
    assert 'Default' in page.board_box.currentText() and 'a default' in page.board_note.text()
    # a user setting: one undo step, labelled
    sb, dl = page._spin['n_poses']
    assert dl.text() == 'default'
    sb.committed.emit(10.0)
    assert win.store.project.analysis.studies['handeye_plan']['n_poses'] == 10 and 'yours' in dl.text()
    _run(win, app, page)
    doc = page.doc
    assert doc is not None and len(doc['poses']) == 10
    assert page.head.status.text.text() == doc['word'] and page.head.figure.text() not in ('', '—')
    assert page.sig_tree.topLevelItemCount() == 2 and page.pose_tree.topLevelItemCount() == 10
    assert 'mm' in page.sig_tree.topLevelItem(0).text(1)
    assert page.b_apply.isEnabled() and page.b_preview.isEnabled()
    assert 'feasible' in page.cand_label.text() and 'One more pose' in page.next_label.text()
    # the 3D overlay: planned poses, the path and the board
    items = page.overlays(win.viewport)
    assert len(items) > 3 * 10 and win.page_overlays(win.viewport)
    # Preview steps the robot through the poses (view state, not an edit)
    n_undo = win.store.undo.count()
    uid = win.store.project.robots[0].uid
    page.preview()
    assert np.allclose(win.store.q[uid], doc['poses'][0]['q'])
    page.preview()
    assert np.allclose(win.store.q[uid], doc['poses'][1]['q'])
    assert win.store.undo.count() == n_undo
    # the Camera view overlay: the board's outline seen from the wrist camera at that pose
    from dexvantage.core import scenegeom as SG
    from dexvantage.core.calib import records as RC
    from dexvantage.core.frames import Pose
    s = win.store.project.variant.sensors[0]
    im = RC.sensor_imager(s)
    pose = Pose.from_matrix(SG.sensor_world_pose(win.store.project, s, win.store.q).matrix() @
                            im.pose_in_body.matrix())
    cv = page.camview_overlays(im.camera, pose)
    assert cv and cv[0][0] and cv[0][0][0].shape[1] == 2
    # Apply saves the plan as a session (and the default board it used), one undo step
    sid = page.apply()
    cal = win.store.project.calibration
    assert sid and cal['sessions'][-1]['uid'] == sid and len(cal['sessions'][-1]['poses']) == 10
    assert cal['sessions'][-1]['source']['kind'] == 'plan' and len(cal['boards']) == 1
    win.store.undo.undo()
    assert win.store.project.calibration == {'boards': [], 'sessions': []}
    # export with provenance
    out = page.export(str(tmp_path / 'plan.json'))
    exp = json.loads(open(out, encoding='utf-8').read())
    assert 'provenance' in exp and exp['sigma'] == doc['sigma']
    assert 'D-optimal' in page.explain_text()


def test_plan_page_saves_and_prints_the_board(app, ur5e, tmp_path):
    win = ur5e
    page = win.pages['handeye_plan']
    win.open_page('handeye_plan')
    page._place['x'].set_value(-150.0)
    page.b_size.set_value(25.0)
    page.b_marker.set_value(18.0)
    uid = page.save_board()
    p = win.store.project
    rec = p.calibration['boards'][0]
    assert uid == rec['uid'] and not rec['default'] and rec['board']['square_m'] == pytest.approx(0.025)
    assert rec['mount']['pose']['position'][0] == pytest.approx(-0.15)
    assert p.analysis.studies['handeye_plan']['board'] == uid
    assert page.board_box.currentData() == uid and page.board_note.text() == ''
    win.store.undo.undo()
    assert p.calibration['boards'] == [] and 'board' not in (p.analysis.studies.get('handeye_plan') or {})
    pdf = page.print_board(str(tmp_path / 'b.pdf'))
    assert pdf and os.path.getsize(pdf) > 1000
    page.b_marker.set_value(40.0)                        # a marker larger than its square
    assert page.save_board() is None and 'cannot be built' in page.board_note.text()


def test_solve_page_imports_solves_applies_and_coaches(app, ur5e, tmp_path):
    from dexvantage.core.calib import planner as PL
    from dexvantage.core.calib import records as RC
    win = ur5e
    p = win.store.project
    rig = PL.rig_from_project(p)
    plan = PL.plan(rig, PL.PlanSettings(n_poses=8))
    render_capture(p, rig, plan.doc['poses'], str(tmp_path / 'img'), str(tmp_path / 'joints.csv'))
    page = win.pages['handeye_solve']
    win.open_page('handeye_solve')
    pump(app, 0.05)
    # the sources that are not here yet say so
    page.source.setCurrentIndex(page.source.findData('ros2'))
    assert page.import_capture(str(tmp_path / 'img'), str(tmp_path / 'joints.csv')) is None
    assert 'Phase 8' in page.import_note.text()
    page.source.setCurrentIndex(page.source.findData('folder'))
    sid = page.import_capture(str(tmp_path / 'img'), str(tmp_path / 'joints.csv'))
    assert sid and RC.find_session(p, sid)['source']['detected'] == 8
    assert '8 images' in page.import_note.text() and page.session_box.currentData() == sid
    assert len(p.calibration['boards']) == 1                     # the default board it used is recorded
    _run(win, app, page)
    doc = page.doc
    assert doc is not None and doc['ok'], doc
    assert page.view_tree.topLevelItemCount() == 8 and page.sig_tree.topLevelItemCount() == 2
    assert page.head.status.text.text() == doc['word'] and 'px' in page.view_tree.topLevelItem(0).text(3)
    assert page.coach_label.text() and doc['coach']['next'] and page.b_preview.isEnabled()
    assert page.overlays(win.viewport)
    # the coach's pose, shown in the view
    uid = p.robots[0].uid
    page.preview()
    assert np.allclose(win.store.q[uid], doc['coach']['next']['q'])
    # Apply: the mount pose and the session's result, one undo step
    s = p.variant.sensors[0]
    before = s.mount.pose.to_dict()
    n = win.store.undo.count()
    assert page.b_apply.isEnabled() and page.apply()
    assert win.store.undo.count() == n + 1
    s = p.variant.sensors[0]
    assert s.mount.pose.to_dict() != before and s.mount.parent_frame == 'wrist_3_link'
    assert np.allclose(s.mount.pose.position, before['position'], atol=3e-4)
    assert RC.find_session(p, sid)['result']['applied']
    win.store.undo.undo()
    s = win.store.project.variant.sensors[0]
    assert s.mount.pose.to_dict() == before and not RC.find_session(win.store.project, sid)['result']
    assert 'bundle adjustment' in page.explain_text()


def test_solve_page_explains_a_missing_session(app, ur5e):
    win = ur5e
    page = win.pages['handeye_solve']
    win.open_page('handeye_solve')
    _run(win, app, page, 60)
    assert page.head.status.text.text() == 'could not run'
    assert 'StudyError' not in page.head.explain.text() and 'session' in page.head.explain.text()
    assert page.import_capture() is None and 'Choose the images folder' in page.import_note.text()
