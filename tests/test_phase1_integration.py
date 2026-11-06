"""Phase 1 integration: the samples in the main window and the Robot page, the wizard's guesses on the bundled files,
the summary's joint count, framing and the robots self-test line."""
import os

import numpy as np
import pytest

from conftest import pump


@pytest.fixture(scope='module')
def samples():
    from dexvantage.core.robot.samples import sample_project
    return {n: sample_project(n) for n in ('so100', 'ur5e')}


def _window(app, settings, project):
    from dexvantage.ui.mainwindow import MainWindow
    w = MainWindow(project, soft_3d=True, settings=settings)
    w.resize(1400, 900)
    w.show()
    w.ctx.analysis.wait(5)
    pump(app, 0.05)
    return w


def _close(app, w):
    w.cancel_jobs()
    w.store.undo.setClean()
    w.hide()
    w.deleteLater()
    pump(app, 0.02)


def test_summary_counts_the_joints_in_q(samples):
    from dexvantage.core import jobs
    from dexvantage.core.robot.model import RobotModel
    for p in samples.values():
        r = p.robots[0]
        doc = jobs.run_study('summary', p)
        assert doc['joints'] == RobotModel(r, p.sources).nq          # fixed joints are not counted
    assert jobs.run_study('summary', samples['ur5e'])['joints'] == 14   # 6 arm + 8 gripper linkage hinges


def test_samples_tag_their_gripper_joints(samples):
    so = [j['name'] for j in samples['so100'].robots[0].joints if j.get('gripper')]
    ur = [j['name'] for j in samples['ur5e'].robots[0].joints if j.get('gripper')]
    assert so == ['Jaw']
    assert len(ur) == 8 and all(n.startswith('gripper_') for n in ur)


@pytest.mark.parametrize('name', ['so100', 'ur5e'])
def test_sample_opens_and_the_robot_page_reads_out(app, settings, samples, name):
    from dexvantage.core.robot.model import get_model
    p = samples[name]
    w = _window(app, settings, p)
    try:
        r = p.robots[0]
        m = get_model(w.store.project, r.uid)
        assert w.ctx.analysis.results.get('summary') is not None
        hud = w.scene.hud_label(w.viewport)
        assert f'{m.nq} joints' in hud
        # framing fits the robot and the volume, not the 1.2 m empty-cell default around a 0.4 m arm
        lo, hi, _ = w.scene.home_box(w.viewport)
        blo, bhi = m.bounds(w.store.q[r.uid])
        assert (lo <= blo + 1e-9).all() and (hi >= bhi - 1e-9).all()
        w.open_page('robot')
        pump(app, 0.05)
        page = w.pages['robot']
        out = page.refresh_readouts()[r.uid]
        assert out['link'] == r.gripper_frame
        assert out['w'] is not None and out['w'] > 0
        # at home the open gripper sits at its stops, which is not the arm's problem
        assert out['near'] == []
        assert page.head.status.text.text() == 'clear'
        img = w.viewport.render_image(scale=0.5)
        assert not img.isNull()
    finally:
        _close(app, w)


def test_so100_framing_is_tighter_than_the_empty_cell(app, settings, samples):
    w = _window(app, settings, samples['so100'])
    try:
        lo, hi, _ = w.scene.home_box(w.viewport)
        assert float(np.max(hi - lo)) < 1.2
    finally:
        _close(app, w)


def test_wizard_guesses_the_gripper_frame_not_a_finger(app, win):
    from dexvantage.core.robot.samples import ROBOTS_DIR
    from dexvantage.ui.dialogs.import_wizard import ImportWizard
    cases = {os.path.join(ROBOTS_DIR, 'trs_so_arm100', 'so_arm100.xml'): 'Fixed_Jaw',
             os.path.join(ROBOTS_DIR, 'ur5e_robotiq_2f85.xml'): 'gripper_base'}
    for path, frame in cases.items():
        wiz = ImportWizard(win.ctx, win, soft_3d=True)
        wiz.load(path, background=False)
        assert wiz.gripper_frame == frame
        wiz.go(2)
        assert '−0 ' not in wiz.s_box.text() and '−0)' not in wiz.s_box.text()
        wiz.hide()
        wiz.deleteLater()
    pump(app, 0.02)


def test_guess_frame_skips_excluded_links():
    from dexvantage.ui.dialogs.import_wizard import guess_frame
    links = ['base', 'wrist', 'gripper_base', 'gripper_left_pad']
    assert guess_frame(links) == 'gripper_left_pad'
    assert guess_frame(links, {'gripper_left_pad'}) == 'gripper_base'
    assert guess_frame(['a', 'b'], {'a', 'b'}) == 'b'           # everything excluded: fall back to the last link


def test_robots_self_test_line():
    from dexvantage.selftest import OK, _check_robots
    status, msg = _check_robots({})
    assert status == OK, msg
    assert 'so100' in msg and 'ur5e' in msg


def test_stretched_ur5e_reads_singular(app, settings):
    from dexvantage.core.robot.samples import sample_project
    p = sample_project('ur5e')
    w = _window(app, settings, p)
    try:
        r = p.robots[0]
        w.open_page('robot')
        page = w.pages['robot']
        m = page.panel(r.uid).model
        q = page.panel(r.uid).q()
        q[m.joint_names.index('elbow_joint')] = 0.0              # upper arm and forearm in line
        w.store.set_q(r.uid, q)
        out = page.refresh_readouts()[r.uid]
        assert out['w'] < 1e-12 and page.head.status.text.text() == 'singular'
        assert page.panel(r.uid).cond.v.text() == '∞'
    finally:
        _close(app, w)
