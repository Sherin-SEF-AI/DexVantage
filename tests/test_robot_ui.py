"""The robot UI: the import wizard run headless through its steps, the joint sliders, the trajectory CSV import and the
pose scrubber, and IK drag in the viewport, on a small URDF written here.

References: joint values against the radians/metres the sliders and fields must produce (degrees and millimetres
converted by hand below); the end-effector read-out against fk; manipulability against numpy's SVD of the Jacobian
rows (an independent route from core/robot/manipulability.py); scrubbed poses against linear interpolation written
out again here; IK drag targets against the plane-ray intersection and fk of the solved pose."""
import math
import struct
import time

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QKeyEvent, QMouseEvent

from conftest import pump
from dexvantage.core.robot import get_model
from dexvantage.core.robot.trajectory import TrajectoryError, read_csv, sample

L1, L2, H = 0.5, 0.4, 0.1          # the fixture's link lengths (m)


def stl_box(hx, hy, hz) -> bytes:
    """A binary STL of a box with half-extents hx, hy, hz centred on the origin (12 triangles)."""
    c = [(x, y, z) for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)]
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = [t for a, b, cc, d in quads for t in ((a, b, cc), (a, cc, d))]
    out = b'\0' * 80 + struct.pack('<I', len(tris))
    for t in tris:
        out += struct.pack('<3f', 0, 0, 0) + b''.join(struct.pack('<3f', *c[i]) for i in t) + b'\0\0'
    return out


INERTIAL = ('<inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>'
            '</inertial>')


def urdf(mesh='../meshes/link1.stl', scale='1 1 1'):
    return f'''<?xml version="1.0"?>
<robot name="ui_arm">
  <link name="base"><visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual></link>
  <link name="link1">{INERTIAL}
    <visual><origin xyz="0.25 0 0"/><geometry><mesh filename="{mesh}" scale="{scale}"/></geometry></visual></link>
  <link name="link2">{INERTIAL}
    <visual><origin xyz="0.2 0 0"/><geometry><box size="0.4 0.03 0.03"/></geometry></visual></link>
  <link name="tool0"/>
  <link name="finger_tip">{INERTIAL}<visual><geometry><box size="0.02 0.02 0.02"/></geometry></visual></link>
  <joint name="shoulder" type="revolute"><parent link="base"/><child link="link1"/><origin xyz="0 0 {H}"/>
    <axis xyz="0 0 1"/><limit lower="-2.0" upper="2.0" velocity="3.1" effort="12"/></joint>
  <joint name="elbow" type="revolute"><parent link="link1"/><child link="link2"/><origin xyz="{L1} 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-2.5" upper="2.5" velocity="3.1" effort="12"/></joint>
  <joint name="flange" type="fixed"><parent link="link2"/><child link="tool0"/><origin xyz="{L2} 0 0"/></joint>
  <joint name="finger_slide" type="prismatic"><parent link="tool0"/><child link="finger_tip"/><axis xyz="1 0 0"/>
    <limit lower="0.0" upper="0.04" velocity="0.2" effort="5"/></joint>
</robot>
'''


@pytest.fixture
def urdf_path(tmp_path):
    d = tmp_path / 'arm'
    (d / 'urdf').mkdir(parents=True)
    (d / 'meshes').mkdir()
    (d / 'meshes' / 'link1.stl').write_bytes(stl_box(0.25, 0.02, 0.02))
    (d / 'urdf' / 'arm.urdf').write_text(urdf())
    return str(d / 'urdf' / 'arm.urdf')


def wizard(win):
    from dexvantage.ui.dialogs.import_wizard import ImportWizard
    return ImportWizard(win.ctx, win)


# ====================================================================== the import wizard
def test_wizard_runs_headless_through_its_steps(app, win, urdf_path):
    st = win.store
    w = wizard(win)
    assert w.go(3) == 0                                       # nothing read yet: the later steps are closed
    res = w.load(urdf_path, background=False)
    assert res['error'] == '' and w.robot is not None and w.name_edit.text() == 'ui_arm'
    assert 'URDF' in w.f_meta.text() and w.f_status.text.text().startswith('Read:')
    assert w.vp is not None and w.scene.mesh() is not None     # the embedded preview draws the robot
    # 2 · meshes
    assert w.next() and w.step == 1
    assert w.m_table.rowCount() == 1 and w.m_table.item(0, 2).text() == 'found'
    assert w.m_table.item(0, 3).text() == 'link1.stl' and 'The required file was found' in w.m_status.text.text()
    # 3 · scale and units: the box at home in mm (x from the base box's −50 to the tip's +910, as written above)
    w.next()
    m = w.model
    lo, hi = m.bounds()
    assert w.s_box.text().startswith(f'{(hi - lo)[0] * 1000:,.0f} × '.replace(',', ' '))
    assert abs((hi - lo)[0] - (0.05 + L1 + L2 + 0.01)) < 1e-6
    assert 'Looks like metres' in w.s_status.text.text() and w.s_table.item(0, 2).text() == 'plausible'
    # 4 · joints, read-only, limits in degrees and mm; the gripper joint guessed from its name and ticked
    w.next()
    rows = {w.j_table.item(r, 0).text(): r for r in range(w.j_table.rowCount())}
    assert set(rows) == {'shoulder', 'elbow', 'flange', 'finger_slide'}
    r = rows['shoulder']
    assert w.j_table.item(r, 2).text() == f'−{math.degrees(2.0):.1f}°' and w.j_table.item(r, 3).text() == '114.6°'
    assert w.j_table.item(r, 4).text() == f'{math.degrees(3.1):.0f} °/s' and w.j_table.item(r, 5).text() == '12.0 N·m'
    r = rows['finger_slide']
    assert w.j_table.item(r, 3).text() == '40.0 mm' and w.j_table.item(r, 4).text() == '200 mm/s'
    assert w.j_table.item(rows['flange'], 2).text() == '—'
    assert w.gripper_joints == ['finger_slide']
    assert w.j_table.item(r, 6).checkState() == Qt.CheckState.Checked
    assert not (w.j_table.item(r, 2).flags() & Qt.ItemFlag.ItemIsEditable)
    w.j_table.item(rows['elbow'], 6).setCheckState(Qt.CheckState.Checked)   # ticking in the table
    assert set(w.gripper_joints) == {'finger_slide', 'elbow'}
    w.set_gripper_joints(['finger_slide'])
    with pytest.raises(KeyError):
        w.set_gripper_joints(['flange'])                       # a fixed joint is not a gripper joint
    w.j_table.selectRow(rows['shoulder'])
    assert w.selected_joint()['name'] == 'shoulder'
    # 5 · gripper frame, fingertips and pads (guessed from the names, then set)
    w.next()
    assert w.gripper_frame == 'tool0' and w.fingertips == ['finger_tip']
    assert w.t_frame.currentData() == 'tool0'
    w.set_gripper_frame('link2')
    assert w.t_frame.currentData() == 'link2'
    w.set_gripper_frame('tool0')
    i = w.add_pad('link2')
    pad = w.pads[i]
    assert pad['name'] == 'Pad 1' and np.allclose(pad['box']['center'], [0.2, 0, 0], atol=1e-6)
    assert np.allclose(pad['box']['size'], [0.4, 0.03, 0.03], atol=1e-6)    # the link's visual box
    assert w.t_pads.rowCount() == 1 and w.pad_row() == 0 and w.pad_editor.isEnabled()
    w.update_pad(0, name='Inner pad', size=[0.02, 0.01, 0.005])
    w.p_c[0].set_value(390.0)
    w.p_c[0].committed.emit(390.0)                            # editing the centre in mm
    assert w.pads[0]['name'] == 'Inner pad' and abs(w.pads[0]['box']['center'][0] - 0.39) < 1e-9
    w.add_pad('tool0', 'Spare')
    w.remove_pad(1)
    assert [p['name'] for p in w.pads] == ['Inner pad']
    assert any(len(d.pts) == 12 for d in w.scene._tag_lines(w.vp))      # the pad's box is drawn in the preview
    # 6 · summary, then Import: one undo step
    w.next()
    assert w.b_next.text() == 'Import' and w.b_next.isEnabled() and 'Ready' in w.sum_status.text.text()
    n = st.undo.count()
    assert w.next() is True and w.imported is not None
    assert st.undo.count() == n + 1 and w.vp is None            # the preview is released
    r = st.project.robots[0]
    assert r.primary and r.name == 'ui_arm' and st.sel == ('robot', r.uid)
    assert r.gripper_frame == 'tool0' and r.fingertip_frames == ['finger_tip']
    assert r.contact_pads == [{'link': 'link2', 'name': 'Inner pad',
                               'box': {'center': pytest.approx([0.39, 0, 0], abs=1e-6),
                                       'size': [0.02, 0.01, 0.005]}}]
    assert [j['name'] for j in r.joints if j.get('gripper')] == ['finger_slide']
    assert r.source.file in st.project.sources and r.uid in st.q
    st.undo.undo()
    assert not st.project.robots and not st.project.sources
    st.undo.redo()
    assert st.project.robots[0].contact_pads[0]['name'] == 'Inner pad'
    w.deleteLater()


def test_wizard_lists_a_missing_mesh_and_will_not_import(win, tmp_path):
    p = tmp_path / 'lone.urdf'
    p.write_text(urdf('meshes/gone.stl'))
    w = wizard(win)
    w.load(str(p), background=False)
    assert w.robot is None and 'gone.stl' in w.f_status.text.text()
    assert w.go(1) == 1 and w.go(4) == 1                       # the mesh check opens; nothing after it
    assert w.m_table.item(0, 2).text() == 'missing' and 'gone.stl' in w.m_status.text.text()
    assert w.m_status.text.text().startswith('1 of 1 file is missing.')
    assert not w.b_next.isEnabled() and not w.can_import() and w.finish() is False
    assert not win.store.project.robots
    w.deleteLater()


def test_wizard_flags_millimetres_and_blocks_import(win, tmp_path):
    d = tmp_path / 'mm'
    d.mkdir()
    (d / 'link1.stl').write_bytes(stl_box(0.25, 0.02, 0.02))
    (d / 'mm.urdf').write_text(urdf('link1.stl', scale='1000 1000 1000'))
    w = wizard(win)
    w.load(str(d / 'mm.urdf'), background=False)
    assert w.robot is not None and w.units_problem()
    w.go(2)
    assert w.s_status.text.text().startswith('The mesh link1.stl is 500 m across')
    assert 'scale="0.001 0.001 0.001"' in w.s_status.text.text()
    assert w.s_table.item(0, 2).text() == 'too large: millimetres?'
    w.go(5)
    assert w.b_next.text() == 'Import' and not w.b_next.isEnabled() and w.finish() is False
    w.deleteLater()


def test_wizard_usd_says_what_to_install(win, tmp_path, monkeypatch):
    import importlib.util as iu
    real = iu.find_spec
    monkeypatch.setattr(iu, 'find_spec', lambda n, *a: None if n == 'pxr' else real(n, *a))
    p = tmp_path / 'arm.usda'
    p.write_text('#usda 1.0\n')
    w = wizard(win)
    w.load(str(p), background=False)
    assert w.robot is None and 'optional: install usd-core' in w.f_status.text.text()
    w.deleteLater()


def test_wizard_reads_in_the_background(app, win, urdf_path):
    w = wizard(win)
    assert w.load(urdf_path) is None and w.loading
    assert w.wait(20) and w.robot is not None and w.step == 0
    w.deleteLater()


def test_import_command_opens_the_wizard(app, win, urdf_path, monkeypatch):
    from dexvantage.ui.dialogs.import_wizard import ImportWizard
    seen = []

    def fake_exec(self):
        seen.append(self)
        self.load(urdf_path, background=False)
        self.go(5)
        self.finish()
        return 1
    monkeypatch.setattr(ImportWizard, 'exec', fake_exec)
    win.actions_by_id['import_robot'].trigger()
    assert seen and len(win.store.project.robots) == 1 and win.store.undo.count() == 1
    assert win.import_robot() is True and len(win.store.project.robots) == 2
    assert not win.store.project.robots[1].primary


# ====================================================================== joint sliders
def _imported(win, path):
    assert win.import_robot(path) is True
    r = win.store.project.robots[-1]
    return r, get_model(win.store.project, r.uid)


def test_joint_sliders_drive_q(app, win, urdf_path):
    st = win.store
    r, m = _imported(win, urdf_path)
    assert win.actions_by_id['joint_sliders'].isEnabled()
    win.actions_by_id['joint_sliders'].trigger()
    pump(app)
    page = win.pages['robot']
    assert win.bottom.tabs.currentWidget() is page
    panel = page.panel(r.uid)
    assert m.joint_names == ['shoulder', 'elbow', 'finger_slide'] and len(panel.sliders) == 3
    seen = []
    st.pose_changed.connect(seen.append)
    # the shoulder's slider spans the file's limits: its last step is the upper limit, its middle zero
    panel.sliders[0].setValue(panel.sliders[0].maximum())
    assert abs(st.q[r.uid][0] - 2.0) < 1e-12 and seen == [r.uid]
    assert panel.spins[0].text().replace('−', '-').startswith(f'{math.degrees(2.0):.2f}')
    panel.sliders[0].setValue(panel.sliders[0].maximum() // 2)
    assert abs(st.q[r.uid][0]) < 1e-12
    # typing degrees and millimetres
    panel.spins[1].set_value(-30.0)
    panel.spins[1].committed.emit(-30.0)
    assert abs(st.q[r.uid][1] - math.radians(-30.0)) < 1e-12
    panel.spins[2].committed.emit(25.0)
    assert abs(st.q[r.uid][2] - 0.025) < 1e-12
    assert panel.sliders[2].value() == round(0.025 / 0.04 * panel.sliders[2].maximum())   # synced back
    panel.spins[2].committed.emit(90.0)                       # past the limit: clipped to 40 mm
    assert abs(st.q[r.uid][2] - 0.04) < 1e-12
    # the read-outs: the gripper frame (none set: its last link) against fk
    out = page.refresh_readouts()[r.uid]
    q = st.q[r.uid]
    T = m.link_pose(q, out['link'])
    assert np.allclose(out['position_mm'], T[:3, 3] * 1000)
    assert out['near'] == ['finger_slide'] and page.head.status.text.text() == 'near a limit'
    if out['w'] is not None:                                  # manipulability: the position rows' SVD, by numpy
        s = np.linalg.svd(m.jacobian(q, out['link'])[:3], compute_uv=False)
        assert out['w'] == pytest.approx(float(np.prod(s)), abs=1e-12)
    if out['collision'] is not None:
        assert out['collision'][1] in ('clear', 'in collision')
    # Home puts the file's home back (zero clipped to the limits)
    panel.b_home.click()
    assert np.allclose(st.q[r.uid], m.home) and np.allclose(m.home, [0, 0, 0])
    page.refresh_readouts()
    assert page.head.status.text.text() == 'near a limit'      # the finger's home is its lower limit (0 mm)
    # the sliders follow a pose set elsewhere (the scrubber, IK drag)
    st.set_q(r.uid, [0.5, -0.5, 0.02])
    assert panel.sliders[2].value() == panel.sliders[2].maximum() // 2
    out = page.refresh_readouts()[r.uid]
    # a planar arm (both hinges about z, the slide along x) cannot move its tip in z: its position Jacobian has
    # rank 2, so the page says "singular" rather than "clear"
    assert page.head.status.text.text() == 'singular'
    # a second robot gets its own panel; undo of its import drops it again
    _imported(win, urdf_path)
    assert len(page._panels) == 2
    st.undo.undo()
    page.rebuild()
    assert len(page._panels) == 1


# ====================================================================== trajectories and the scrubber
CSV = 't,shoulder,elbow\n0.0,0.0,0.0\n1.0,0.5,-0.4\n2.0,1.0,-0.8\n'


def _robots(win):
    out = []
    for r in win.store.project.robots:
        m = get_model(win.store.project, r.uid)
        out.append((r.uid, m.joint_names, m.lower, m.upper, m.joint_types))
    return out


def test_trajectory_csv_import_and_scrub(app, win, urdf_path, tmp_path):
    from dexvantage.ui.pages.robot import import_trajectory
    st = win.store
    r, m = _imported(win, urdf_path)
    sc = win.scrubber
    assert not sc.slider.isEnabled() and 'Import trajectory' in sc.toolTip()
    assert 'import_trajectory' in win.actions_by_id
    assert win.actions_by_id['import_trajectory'] in win.menus['Robot'].actions()
    f = tmp_path / 'reach.csv'
    f.write_text(CSV)
    st.set_q(r.uid, [0.0, 0.0, 0.03])
    n = st.undo.count()
    assert import_trajectory(win.ctx, str(f)) is True
    assert st.undo.count() == n + 1
    tr = st.project.tasks.trajectories[0]
    assert tr['name'] == 'reach' and tr['robot_uid'] == r.uid and tr['joints'] == ['shoulder', 'elbow']
    assert tr['t'] == [0.0, 1.0, 2.0] and tr['q'][2] == [1.0, -0.8]
    assert sc.slider.isEnabled() and sc.combo.isVisible() and sc.combo.currentText() == 'reach'
    assert sc.slider.maximum() == 2000                        # 1 ms steps
    # scrubbing: linear between samples; the finger, not in the file, keeps its value
    q = sc.scrub_to(0.25)
    assert np.allclose(q, [0.125, -0.1, 0.03]) and np.allclose(st.q[r.uid], q)
    sc.slider.setValue(1500)
    assert abs(sc.t - 1.5) < 1e-9 and np.allclose(st.q[r.uid][:2], [0.75, -0.6])
    assert sc.where.text().strip().startswith('t   1.500 / 2.000 s')
    sc.step(1)
    assert sc.t == 2.0 and np.allclose(st.q[r.uid][:2], [1.0, -0.8])
    sc.step(-1)
    assert sc.t == 1.0
    sc.scrub_to(99)
    assert sc.t == 2.0                                        # clamped to the trajectory
    # play at real time: the trajectory clock follows the wall clock
    sc.scrub_to(0.0)
    w0 = time.monotonic()
    sc.play()
    assert sc.playing
    pump(app, 0.3)
    el = time.monotonic() - w0
    assert 0.05 < sc.t <= el + 0.02
    assert np.allclose(st.q[r.uid][:2], sample(tr, sc.t), atol=1e-9) or sc.playing
    sc.stop()
    assert not sc.playing
    # one undo step takes the trajectory away; the scrubber closes
    st.undo.undo()
    assert not st.project.tasks.trajectories and not sc.slider.isEnabled()


def test_trajectory_csv_errors_are_plain(win, urdf_path, tmp_path, monkeypatch):
    from dexvantage.ui.pages import robot as RP
    _imported(win, urdf_path)
    shown = []
    monkeypatch.setattr(RP.QMessageBox, 'warning', lambda *a: shown.append(a[1:3]))
    cases = [
        ('', 'The file is empty'),
        ('x,shoulder\n0,1\n', 'first column is "x"'),
        ('t,shoulder,wrist\n0,1,2\n', 'joint named wrist'),
        ('t,shoulder\n0,abc\n', 'Row 2, column shoulder: "abc" is not a number'),
        ('t,shoulder\n0,1\n0,2\n', 'times must increase'),
        ('t,shoulder\n0,1,2\n', 'Row 2 has 3 values'),
        ('t,shoulder,shoulder\n0,1,1\n', 'names joint shoulder twice'),
        ('t\n0\n', 'names no joints'),
        ('t,shoulder\n', 'no rows of numbers'),
    ]
    for k, (text, want) in enumerate(cases):
        f = tmp_path / f'bad{k}.csv'
        f.write_text(text)
        assert RP.import_trajectory(win.ctx, str(f)) is False
        assert shown[-1][0] == 'Could not import the trajectory' and want in shown[-1][1], (text, shown[-1])
    assert not win.store.project.tasks.trajectories and len(shown) == len(cases)


def test_trajectory_reader_warnings_and_sampling(win, urdf_path):
    _imported(win, urdf_path)
    robots = _robots(win)
    rec, warn = read_csv('t;shoulder\n0;0\n1;90\n', robots)     # semicolons; degrees by mistake; past the limit
    assert rec['joints'] == ['shoulder'] and any('look like degrees' in w for w in warn)
    assert any('leaves its limits in 1 of 2 samples' in w for w in warn)
    rec, warn = read_csv('# a comment\nt,finger_slide\n0,0.0\n0.5,0.02\n', robots)
    assert warn == [] and sample(rec, 0.25)[0] == pytest.approx(0.01) and sample(rec, -1)[0] == 0.0
    with pytest.raises(TrajectoryError, match='no robot'):
        read_csv('t,shoulder\n0,0\n', [])


# ====================================================================== IK drag
def send(w, e):
    """Deliver an event as Qt does (the viewport's handlers are found through Qt's virtual dispatch)."""
    from PyQt6.QtWidgets import QApplication
    QApplication.sendEvent(w, e)


def _mouse(kind, x, y, button=Qt.MouseButton.LeftButton):
    btns = Qt.MouseButton.NoButton if kind == QEvent.Type.MouseButtonRelease else button
    p = QPointF(x, y)
    return QMouseEvent(kind, p, p, button, btns, Qt.KeyboardModifier.NoModifier)


def test_ik_drag_calls_solve_and_updates_q(app, win, urdf_path, monkeypatch):
    from dexvantage.ui import ikdrag
    st = win.store
    r, m = _imported(win, urdf_path)
    st.edit('robots', lambda p: setattr(p.robots[0], 'gripper_frame', 'tool0'), 'Tagged the gripper frame')
    m = get_model(st.project, r.uid)
    calls = []
    try:
        real = ikdrag.solver()                                # the real IK solver (core/robot/ik.py)
    except ImportError:                                       # not in this build: a stand-in reaching anything
        real = None

    def solve(model, link, target, q0=None, **kw):
        calls.append((link, np.array(target), np.array(q0), kw))
        if real is not None:
            return real(model, link, target, q0=q0, **kw)

        class R:
            ok, q, pos_err, reason = True, np.clip(np.asarray(q0) + 0.1, model.lower, model.upper), 0.0, ''
        return R()
    monkeypatch.setattr(ikdrag, 'solver', lambda: solve)
    act = win.actions_by_id['ik_drag']
    assert act.isEnabled() and act.isCheckable()
    act.setChecked(True)
    ik = win.ctx.ik_drag
    assert ik.on
    vp = win.viewport
    vp.set_view('top')
    vp.frame_box(*win.scene.home_box(vp))
    pump(app)
    h = ik.handle()
    assert np.allclose(h, m.link_pose(st.q[r.uid], 'tool0')[:3, 3])
    assert any(len(d.tri) == 8 for d in ik.draws(vp) if hasattr(d, 'tri'))       # the handle
    sc, _d, ok = vp.project([h])
    hx, hy = sc[0]
    assert ok[0]
    # a press away from the handle is not a drag (navigation keeps it)
    assert not win.scene.mouse_press(vp, _mouse(QEvent.Type.MouseButtonPress, hx + 80, hy + 80))
    # press on the handle, move to a reachable point on the view plane (z of the handle: the view is from the top)
    send(vp, _mouse(QEvent.Type.MouseButtonPress, hx, hy))
    assert ik.dragging
    goal = np.array([0.6, 0.3, h[2]])
    gx, gy = vp.project([goal])[0][0]
    send(vp, _mouse(QEvent.Type.MouseMove, gx, gy))
    o, dv = vp.ray(gx, gy)                                   # the view-aligned plane through the handle, again
    n = vp.cam.view()[2, :3]
    want = o + dv * ((h - o) @ n) / (dv @ n)
    assert calls and calls[-1][0] == 'tool0' and np.allclose(calls[-1][1], want, atol=1e-9)
    assert abs((calls[-1][1] - h) @ n) < 1e-9 and np.linalg.norm(calls[-1][1] - goal) < 1e-3
    assert calls[-1][3].get('rule') == 'nearest_q0'
    assert ik.last[0] is True and ik.last[1] == 'reachable'
    q1 = st.q[r.uid].copy()
    if real is not None:
        assert np.linalg.norm(m.link_pose(q1, 'tool0')[:3, 3] - calls[-1][1]) < 1e-3
    assert 'reachable' in win.msg_label.text()
    assert ik.draws(vp)[0].colors[0][1] > ik.draws(vp)[0].colors[0][0]   # green
    # out of reach: the pose stays, the handle turns red, the reason goes to the status bar
    far = np.array([3.0, 0.0, h[2]])
    fx, fy = vp.project([far])[0][0]
    if real is None:
        monkeypatch.setattr(ikdrag, 'solver', lambda: lambda *a, **k: type(
            'R', (), {'ok': False, 'q': None, 'pos_err': 2.1, 'reason': 'out of reach'})())
    send(vp, _mouse(QEvent.Type.MouseMove, fx, fy))
    assert ik.last[0] is False and np.allclose(st.q[r.uid], q1)
    assert 'not reachable' in win.msg_label.text() and 'out of reach' in win.msg_label.text()
    d = ik.draws(vp)
    assert d[0].colors[0][0] > d[0].colors[0][1] and len(d) == 2             # red, with a dashed line to the target
    # Esc puts the pose back where the drag started
    send(vp, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))
    assert not ik.dragging and np.allclose(st.q[r.uid], m.home)
    # a drag that ends with the release keeps its pose
    monkeypatch.setattr(ikdrag, 'solver', lambda: solve)
    send(vp, _mouse(QEvent.Type.MouseButtonPress, hx, hy))
    send(vp, _mouse(QEvent.Type.MouseMove, gx, gy))
    send(vp, _mouse(QEvent.Type.MouseButtonRelease, gx, gy))
    assert not ik.dragging and not np.allclose(st.q[r.uid], m.home)
    assert st.undo.count() == 2                               # joint state is view state: no undo steps
    act.setChecked(False)
    assert not ik.on and ik.handle() is None and ik.draws(vp) == []
    assert not win.scene.mouse_press(vp, _mouse(QEvent.Type.MouseButtonPress, hx, hy))
