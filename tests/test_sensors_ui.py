"""Phase 2 sensors UI: the Camera view (core/camrender.py, ui/camview.py), the Sensors and mounts page, the Library
panel, Library ▸ New custom device… and the lens-model frusta in the 3D view, on the UR5e sample."""
import math

import numpy as np
import pytest

from conftest import pump


@pytest.fixture
def ur5e(win, app):
    assert win.open_sample('ur5e')
    win.store.undo.setClean()
    win.ctx.analysis.wait(5)
    pump(app, 0.02)
    return win


def _sensor(win, name):
    return next(s for s in win.store.project.variant.sensors if s.name == name)


def _robot_q(win):
    r = win.store.project.robots[0]
    return r.uid, win.store.q[r.uid].copy()


# ====================================================================== core renderer
def test_raster_matches_the_plane_exactly():
    """A big square 1 m in front of a pinhole camera fills the image; every pixel's distance is that of its ray to
    the plane (1 / cos of the ray angle), written out again here."""
    from dexvantage.core import camrender as CR
    from dexvantage.core.frames import Pose
    from dexvantage.core.optics import CameraModel
    cam = CameraModel.from_fov(64, 48, 90.0)
    a, b, c, d = [-5, -5, 1.0], [5, -5, 1.0], [5, 5, 1.0], [-5, 5, 1.0]
    tris = np.array([[a, b, c], [a, c, d]], float)
    z, tri = CR.raster(tris, cam.fx, cam.fy, cam.cx, cam.cy, 64, 48)
    assert (tri >= 0).all() and np.allclose(z, 1.0)
    # the same square in the world, seen through render() from a pose whose optical axis is world +x
    from dexvantage.core.frames import OPTICAL_TO_BODY, quat_from_matrix
    world = tris[:, :, [2, 0, 1]] * np.array([1, -1, -1])          # optical (x right, y down, z fwd) -> body
    geom = CR.Geometry(world.astype(np.float32), np.ones((2, 3), np.float32), CR._normals(world),
                       np.full(2, CR.KIND_OBJECT, np.int8), np.zeros(2, np.int32), ['object:a'])
    r = CR.render(geom, cam, Pose([0, 0, 0], quat_from_matrix(OPTICAL_TO_BODY)), floor=False)
    us, vs = np.meshgrid(np.arange(64.0), np.arange(48.0))
    ray_z = 1.0 / np.sqrt(((us - cam.cx) / cam.fx) ** 2 + ((vs - cam.cy) / cam.fy) ** 2 + 1)
    assert np.allclose(r.dist, 1.0 / ray_z, atol=1e-9)
    assert r.coverage['object'] == 1.0


def test_render_paths_agree_on_the_sample():
    """The three lens paths (pinhole, a pinhole base image for a mild Brown lens, cube faces for a fisheye) see the
    same scene: a Brown lens with zero distortion and an equidistant lens give the robot where the pinhole does."""
    from dexvantage.core import camrender as CR
    from dexvantage.core import library
    from dexvantage.core.optics import CameraModel
    from dexvantage.core.robot.samples import sample_project
    p = sample_project('ur5e')
    s = next(s for s in p.variant.sensors if s.name == 'Wrist D405')
    im = next(i for i in library.sensor_imagers(s) if i.role == 'rgb')
    pose = CR.optical_pose(p, s, im)
    g = CR.without_seat(CR.gather(p), p, s, [pose.position])
    pin = CameraModel(160, 90, 80.0, 80.0, 79.5, 44.5)
    brown = CameraModel(160, 90, 80.0, 80.0, 79.5, 44.5, 'brown', (0, 0, 0, 0, 0))
    r0 = CR.render(g, pin, pose)
    r1 = CR.render(g, brown, pose)
    assert (r0.path, r1.path) == ('pinhole', 'base')
    assert np.mean(r0.kind == r1.kind) > 0.97
    both = np.isfinite(r0.dist) & np.isfinite(r1.dist) & (r0.tri == r1.tri)
    assert np.allclose(r0.dist[both], r1.dist[both], atol=1e-6)
    fish = CameraModel.from_fov(200, 200, 190.0, model='equidistant', max_theta=math.radians(95))
    rf = CR.render(g, fish, pose)
    assert rf.path == 'cube' and rf.coverage['robot'] > 0.05
    assert not rf.valid[0, 0] and rf.valid[100, 100]            # the image circle's corner has no ray


def test_render_against_exact_ray_casting():
    """Independent reference: core/raycast.py's Möller–Trumbore first hit along the same pixel rays. Pixels away from
    a silhouette agree to 1 mm."""
    from dexvantage.core import camrender as CR
    from dexvantage.core import library
    from dexvantage.core import raycast as RC
    from dexvantage.core.frames import quat_to_matrix
    from dexvantage.core.robot.samples import sample_project
    p = sample_project('ur5e')
    s = next(s for s in p.variant.sensors if s.name == 'Wrist D405')
    im = next(i for i in library.sensor_imagers(s) if i.role == 'rgb')
    pose = CR.optical_pose(p, s, im)
    g = CR.without_seat(CR.gather(p), p, s, [pose.position])
    r = CR.render(g, im.camera, pose, 128, 72, floor=False)
    cam = r.camera
    rng = np.random.default_rng(7)
    vv, uu = rng.integers(0, 72, 600), rng.integers(0, 128, 600)
    rays = cam.unproject(np.c_[uu, vv].astype(float))
    R = quat_to_matrix(pose.quaternion)
    d = rays @ R.T
    near = CR.NEAR_M / rays[:, 2]
    grid = RC.Grid.build(g.tris)
    t = RC.first_hit(pose.position + d * near[:, None], d, grid) + near
    mine = r.dist[vv, uu].astype(float)
    sil = RC.near_silhouette(pose.position + d * np.maximum(np.where(np.isfinite(t), t, 1.0), 0.01)[:, None],
                             pose.position, grid, 2.0 / cam.fx)
    with np.errstate(invalid='ignore'):
        agree = (np.isinf(t) & np.isinf(mine)) | (np.abs(t - mine) <= 1e-3)
    assert agree[~sil].mean() >= 0.995, (agree[~sil].mean(), (~sil).sum())


def test_frustum_from_the_lens_model():
    from dexvantage.core import camrender as CR
    from dexvantage.core.frames import OPTICAL_TO_BODY, Pose, quat_from_matrix
    from dexvantage.core.optics import CameraModel
    cam = CameraModel.from_fov(640, 480, 90.0)
    apex, border, spokes = CR.frustum_body(cam, Pose([0, 0, 0], quat_from_matrix(OPTICAL_TO_BODY)), 0.1)
    assert len(spokes) == 4 and np.allclose(border[:, 0], 0.1)            # the plane 0.1 m along body x
    assert math.isclose(float(np.max(border[:, 1])), 0.1 * math.tan(math.radians(45)), rel_tol=1e-3)
    fish = CameraModel.from_fov(400, 400, 200.0, model='equidistant')
    _a, b2, _s = CR.frustum_body(fish, Pose([0, 0, 0], quat_from_matrix(OPTICAL_TO_BODY)), 0.1)
    ok = np.isfinite(b2).all(axis=1)
    assert np.allclose(np.linalg.norm(b2[ok], axis=1), 0.1)                 # a wide lens: on the sphere
    assert (b2[ok, 0] < 0).any()                                            # and it reaches behind the imager


# ====================================================================== camera view
def test_camera_view_renders_the_wrist_d405_and_follows_q(ur5e):
    win = ur5e
    s = _sensor(win, 'Wrist D405')
    win.store.select(('sensor', s.uid))
    res = win.camview.render_sync()
    assert res is not None and res['imager'].name in ('RGB', 'IR left', 'IR right')
    r = res['render']
    assert r.coverage['robot'] > 0.05 and r.rgb.std() > 5            # the gripper fills part of a non-flat image
    assert res['image'].width() == r.rgb.shape[1]
    before = r.rgb.copy()
    uid, q = _robot_q(win)
    q[5] += 0.6                                                      # turn the wrist: the camera moves with it
    win.store.set_q(uid, q)
    res2 = win.camview.render_sync()
    assert res2['render'].rgb.shape == before.shape
    assert np.abs(res2['render'].rgb.astype(int) - before.astype(int)).mean() > 2


def test_camera_view_imager_choice_and_background_render(ur5e):
    win = ur5e
    s = _sensor(win, 'Overhead D435')
    win.store.select(('sensor', s.uid))
    view = win.camview.view
    names = [view.imager_box.itemText(i) for i in range(view.imager_box.count())]
    assert {'RGB', 'IR left', 'IR right'} <= set(names)
    view.imager_box.setCurrentIndex(names.index('IR left'))
    view._imager_picked(names.index('IR left'))
    win.camview.show()
    view.schedule()
    assert win.camview.wait(20)
    assert view.result is not None and view.result['imager'].name == 'IR left'
    assert 'IR left' in view.info.text()


def test_camera_view_page_overlay_hook(ur5e):
    from dexvantage.ui.pages.base import StudyPage
    win = ur5e
    s = _sensor(win, 'Overhead D435')
    win.store.select(('sensor', s.uid))
    win.camview.render_sync()
    page = next(iter(win.pages.values()))
    assert StudyPage.camview_overlays(page, None, None) == []
    seen = {}

    def hook(cam, pose):
        seen['cam'] = cam
        return [([np.array([[0.0, 0.0], [100.0, 100.0]])], '#ff0000', 'test line')]
    page.camview_overlays = hook
    win.open_page(next(iter(win.pages)))
    win.camview.view.refresh_overlays()
    ov = win.camview.view.page_overlays()
    assert ov and ov[0][2] == 'test line' and seen['cam'].width == win.camview.view.result['camera'].width
    win.camview.view.image.grab()                                    # paints the overlay without error


def test_default_camera_is_labelled(win):
    win.add_static_camera()
    res = win.camview.render_sync()
    assert res['imager'].is_default
    assert win.camview.view.info.text().startswith('Default camera (no device chosen)')


# ====================================================================== sensors page
def test_sensors_page_edits_the_mount_and_undo_restores(ur5e):
    win = ur5e
    page = win.pages['sensors']
    win.open_page('sensors')
    s = _sensor(win, 'Overhead D435')
    win.store.select(('sensor', s.uid))
    from dexvantage.core.scenegeom import sensor_world_pose
    before = s.to_dict()
    w0 = sensor_world_pose(win.store.project, s, win.store.q)
    r = win.store.project.robots[0]
    assert page.set_mount(s.uid, 'wrist_3_link', r.uid, keep_place=True)
    s2 = win.store.project.variant.sensor(s.uid)
    assert s2.mount.on_link and s2.mount.parent_frame == 'wrist_3_link'
    w1 = sensor_world_pose(win.store.project, s2, win.store.q)
    assert np.allclose(w0.position, w1.position, atol=1e-9)         # it stayed where it was
    assert 'wrist_3_link' in page.table.topLevelItem(1).text(2)
    win.store.undo.undo()
    assert win.store.project.variant.sensor(s.uid).to_dict() == before
    # the mount combo lists the static frames and every link of the robot
    texts = [page.mount_box.itemText(i) for i in range(page.mount_box.count())]
    assert 'Tripod (static)' in texts and any(t.endswith('▸ wrist_3_link') for t in texts)


def test_sensors_page_pose_stream_device_and_enabled(ur5e):
    win = ur5e
    page = win.pages['sensors']
    s = _sensor(win, 'Overhead D435')
    win.store.select(('sensor', s.uid))
    before = s.to_dict()
    n0 = win.store.undo.count()
    page.set_position(s.uid, [0.1, 0.2, 0.9])
    page.set_rpy(s.uid, [0.0, 90.0, 0.0])
    page.set_stream(s.uid, (640, 480))
    page.set_enabled(s.uid, False)
    page.set_device(s.uid, 'intel-realsense-d455')
    o = win.store.project.variant.sensor(s.uid)
    assert np.allclose(o.mount.pose.position, [0.1, 0.2, 0.9])
    assert o.model_id == 'intel-realsense-d455' and not o.enabled and o.kind == 'depth'
    assert win.store.undo.count() == n0 + 5                         # one undo step each
    for _ in range(5):
        win.store.undo.undo()
    assert win.store.project.variant.sensor(s.uid).to_dict() == before
    # the editor's position fields are in millimetres
    page.refresh()
    assert page.pos_boxes[2].value() == pytest.approx(before['mount']['pose']['position'][2] * 1000)


def test_sensors_page_add_duplicate_remove(ur5e):
    win = ur5e
    page = win.pages['sensors']
    n = len(win.store.project.variant.sensors)
    a = page.add_sensor('intel-realsense-d435')
    assert a.model_id == 'intel-realsense-d435' and a.kind == 'depth'
    d = page.duplicate(a.uid)
    assert d.uid != a.uid and d.model_id == a.model_id and d.name != a.name
    assert len(win.store.project.variant.sensors) == n + 2
    assert page.remove(d.uid)
    assert len(win.store.project.variant.sensors) == n + 1
    win.store.undo.undo()
    assert len(win.store.project.variant.sensors) == n + 2
    assert page.table.topLevelItemCount() == n + 2


def test_stream_mode_is_saved_and_used(ur5e, tmp_path):
    from dexvantage.core import library
    from dexvantage.core.project import Project
    win = ur5e
    s = _sensor(win, 'Wrist D405')
    win.pages['sensors'].set_stream(s.uid, (848, 480))
    path = str(tmp_path / 'p.dexv')
    win.store.project.save(path)
    p = Project.load(path)
    s2 = next(x for x in p.variant.sensors if x.name == 'Wrist D405')
    assert s2.stream == [848, 480]
    ims = library.sensor_imagers(s2, tuple(s2.stream))
    assert ims[0].mode == (848, 480)


# ====================================================================== inspector
def test_inspector_shows_figures_and_not_stated(ur5e):
    from PyQt6.QtWidgets import QLabel
    from dexvantage.core.library import NOT_STATED
    win = ur5e
    s = _sensor(win, 'Wrist D405')
    win.store.select(('sensor', s.uid))
    labels = win.inspector.scroll.widget().findChildren(QLabel)
    texts = {lb.text(): lb for lb in labels}
    dash = [lb for t, lb in texts.items() if t == '—']
    assert dash and all(lb.toolTip() == NOT_STATED for lb in dash)      # D405: pixel size not stated
    assert any('18 mm' == t for t in texts)                             # the stated baseline
    assert any('href="https://' in t for t in texts)                    # the source link


# ====================================================================== library panel
def test_library_panel_lists_devices_and_adds_a_sensor(ur5e):
    from dexvantage.core import library
    win = ur5e
    lib = win.library
    it = lib.item_for('intel-realsense-d435')
    assert it is not None and 'Source:' in it.toolTip(0) and '(read 20' in it.toolTip(0)
    n = len(win.store.project.variant.sensors)
    lib._activated(it)
    ss = win.store.project.variant.sensors
    assert len(ss) == n + 1 and ss[-1].model_id == 'intel-realsense-d435' and not ss[-1].mount.on_link
    assert win.store.sel == ('sensor', ss[-1].uid)
    # it looks at the task volume
    from dexvantage.core.frames import quat_to_matrix
    c = np.asarray(win.store.project.workspace.volumes[0].center)
    x = quat_to_matrix(ss[-1].mount.pose.quaternion)[:, 0]
    d = c - ss[-1].mount.pose.position
    assert np.dot(x, d / np.linalg.norm(d)) > 0.999
    # a host is not placed
    host = next(e for e in library.entries('host'))
    assert lib.add_device(host.id) is None and len(win.store.project.variant.sensors) == n + 1
    # search filters
    lib.search.setText('d405')
    assert lib.item_for('intel-realsense-d405') is not None and lib.item_for('intel-realsense-d435') is None
    lib.search.setText('')


def test_library_drag_drop_into_the_view(ur5e):
    from PyQt6.QtCore import QMimeData, QPointF, Qt
    from PyQt6.QtGui import QDropEvent
    from dexvantage.ui.rigedit import MIME
    win = ur5e
    md = win.library.tree.mimeData([win.library.item_for('intel-realsense-d405')])
    assert bytes(md.data(MIME)).decode() == 'intel-realsense-d405'
    vp = win.viewport
    pos = QPointF(vp.width() / 2, vp.height() * 0.75)
    m = QMimeData()
    m.setData(MIME, b'intel-realsense-d405')
    e = QDropEvent(pos, Qt.DropAction.CopyAction, m, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    n = len(win.store.project.variant.sensors)
    assert win.scene.drag_enter(vp, e)
    win.scene.drop(vp, e)
    ss = win.store.project.variant.sensors
    assert len(ss) == n + 1 and ss[-1].model_id == 'intel-realsense-d405'
    g = vp.ground_at(pos.x(), pos.y(), 0.0)
    if g is not None:
        assert np.allclose(ss[-1].mount.pose.position[:2], g[:2])


# ====================================================================== custom device
def test_custom_device_dialog_saves_and_appears(ur5e):
    from dexvantage.core import library
    from dexvantage.ui.dialogs.custom_device import CustomDeviceDialog
    win = ur5e
    a = win.actions_by_id['new_device']
    assert a.isEnabled()
    d = CustomDeviceDialog(win.ctx, win, kind='camera')
    d.editor('maker').setText('Test maker')
    d.editor('name').setText('Wide cam 1')
    assert d.editor('id').text() == 'test-maker-wide-cam-1'
    assert not d.b_save.isEnabled()                                 # no source yet
    d.editor('src_url').setText('measured by me with a checkerboard')
    for k, v in (('hfov', '120'), ('vfov', '90'), ('resolution', '1280, 960'), ('fps', '30, 60')):
        d.set_given(k, True)
        d.editor(k).setText(v)
    d.set_given('shutter', True)
    d.set_value('shutter', 'global')
    d.set_given('pixel_um', True)
    d.editor('pixel_um').setText('abc')
    assert not d.b_save.isEnabled() and 'Pixel size' in d.msg.text()   # a plain error, saving blocked
    d.set_given('pixel_um', False)
    assert d.b_save.isEnabled(), d.msg.text()
    e = d.save()
    assert e is not None and e.is_custom and library.get('test-maker-wide-cam-1') is not None
    data = e.data
    assert data['imagers'][0]['pixel_um'] is None and data['mass_g'] is None      # not given: null, not a guess
    assert data['imagers'][0]['hfov_deg'] == 120 and data['streams'][0]['fps'] == [30, 60]
    assert 'Not given:' in data['note'] and 'Pixel size' in data['note']
    assert win.library.item_for('test-maker-wide-cam-1') is not None
    # it works as a sensor device: a 120 x 90 degree pinhole at 1280 x 960
    s = win.library.add_device('test-maker-wide-cam-1')
    ims = library.sensor_imagers(s)
    assert ims[0].mode == (1280, 960)
    assert ims[0].camera.fov_deg()['h'] == pytest.approx(120, abs=0.01)
    d.close()
    library.remove_custom('test-maker-wide-cam-1')


def test_custom_device_from_a_library_depth_camera(win):
    from dexvantage.core import library
    from dexvantage.ui.dialogs.custom_device import CustomDeviceDialog
    d = CustomDeviceDialog(win.ctx, win, start_from='intel-realsense-d435')
    assert d.kind() == 'depth' and d.editor('id').text() == 'intel-realsense-d435-custom'
    assert d.editor('baseline_mm').text() == '50'
    d.editor('src_url').setText('my own measurement of one unit')
    d.editor('hfov').setText('88')
    assert d.b_save.isEnabled(), d.msg.text()
    e = d.save()
    base = library.get('intel-realsense-d435')
    assert e.data['imagers'][0]['hfov_deg'] == 88
    assert len(e.data['imagers']) == len(base.data['imagers'])           # the other imagers kept
    assert e.data['sources'][0]['url'] == 'my own measurement of one unit'
    assert e.data['sources'][1:] == base.data['sources']
    # a built-in id is refused in plain words
    d.editor('id').setText('intel-realsense-d435')
    assert not d.b_save.isEnabled() and 'built-in' in d.msg.text()
    d.close()
    library.remove_custom('intel-realsense-d435-custom')


# ====================================================================== frusta
def test_frusta_follow_the_lens_and_the_arm(ur5e):
    from dexvantage.ui.viewport3d import LineDraw
    win = ur5e
    sc = win.scene
    fr = sc.frusta()
    wrist = [f for f in fr if f['sensor'].name == 'Wrist D405']
    assert {f['imager'] for f in wrist} == {'IR left', 'IR right', 'RGB'}
    assert all(f['segments'].shape[1:] == (2, 3) for f in fr)
    lines = sc.lines(win.viewport)
    segs0 = wrist[0]['segments']
    assert any(isinstance(d, LineDraw) and d.segments and np.array_equal(d.pts, segs0) for d in lines)
    # the reference imager's outline matches its modelled FOV: 84 x 58 deg (D405 16:9 depth FOV)
    from dexvantage.core import library
    im = library.sensor_imagers(wrist[0]['sensor'])[0]
    assert im.camera.fov_deg()['h'] == pytest.approx(84.0, abs=0.1)
    uid, q = _robot_q(win)
    q[0] += 0.4
    win.store.set_q(uid, q)
    moved = next(f for f in sc.frusta() if f['sensor'].name == 'Wrist D405' and f['index'] == 0)['segments']
    assert np.abs(moved - segs0).max() > 0.01
    over = next(f for f in sc.frusta() if f['sensor'].name == 'Overhead D435' and f['index'] == 0)['segments']
    over0 = next(f for f in fr if f['sensor'].name == 'Overhead D435' and f['index'] == 0)['segments']
    assert np.allclose(over, over0)                                    # a static mount does not move


def test_camera_view_small_while_dragging_full_after_and_ui_not_blocked(ur5e, app):
    """Frame budget: renders run on a worker thread; while the scene is dragged the image is rendered at half size,
    and at full size once the drag ends. The UI thread's event loop keeps turning (no gap over 100 ms)."""
    import time
    from dexvantage.ui.camview import DRAG_SCALE
    win = ur5e
    view = win.camview.view
    s = _sensor(win, 'Wrist D405')
    win.store.select(('sensor', s.uid))
    win.camview.show()
    full = win.camview.render_sync()['render'].rgb.shape
    uid, q = _robot_q(win)
    win.viewport.hide()             # the offscreen tests' software 3D view takes ~0.8 s a frame itself (not on test)
    win.store.dragging = True
    shapes, worst, t_end = set(), 0.0, time.time() + 1.5
    while time.time() < t_end:
        q[0] += 0.02
        t0 = time.perf_counter()
        win.store.set_q(uid, q)
        app.processEvents()
        worst = max(worst, time.perf_counter() - t0)
        if view.result:
            shapes.add((view.result['job']['scale'], view.result['render'].rgb.shape))
        time.sleep(0.01)
    assert worst < 0.1, worst
    small = [sh for k, sh in shapes if k == DRAG_SCALE]
    assert small and all(sh[1] <= full[1] * DRAG_SCALE + 2 for sh in small), (shapes, full)
    win.store.dragging = False
    assert view.wait(20)
    assert view.result['job']['scale'] == 1.0 and view.result['render'].rgb.shape[1] > small[0][1] * 1.5
    win.viewport.show()
