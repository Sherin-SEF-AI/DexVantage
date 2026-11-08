"""The MuJoCo-render checks (validation/mjrender.py, checks_optics_mujoco.py, checks_occlusion_mujoco.py): the render
worker, the camera conventions, the explained failure without EGL, and the checks themselves (quick runs)."""
import math
import subprocess

import numpy as np
import pytest


def _scene_xml(cam_attrs='fovy="60"', W=320, H=240, extra=''):
    return (f'<mujoco><visual><global offwidth="{W}" offheight="{H}"/></visual><worldbody>'
            f'<camera name="c" pos="0 0 1" xyaxes="1 0 0 0 1 0" {cam_attrs}/>'
            '<geom name="wall" type="box" size="2 2 0.01" pos="0 0 -0.01"/>'
            '<geom name="up" type="sphere" size="0.03" pos="0 0.2 0.3"/>' + extra + '</worldbody></mujoco>')


@pytest.fixture(scope='module')
def rendered():
    from render_support import require
    require()
    import mujoco
    from dexvantage.validation import mjrender as MR
    m = mujoco.MjModel.from_xml_string(_scene_xml())
    imgs, info = MR.render([{'model': m, 'camera': 'c', 'width': 320, 'height': 240,
                             'outputs': ('seg', 'depth', 'rgb')}])
    return m, imgs[0], info


def test_worker_renders_segmentation_depth_and_rgb(rendered):
    m, im, info = rendered
    assert im['seg'].shape == (240, 320) and im['seg'].dtype == np.int32
    assert im['depth'].shape == (240, 320) and im['rgb'].shape == (240, 320, 3)
    assert info['gl'] == 'egl' and info['mujoco'] and info['renderer']
    wall, up = m.geom('wall').id, m.geom('up').id
    assert set(np.unique(im['seg'])) <= {wall, up, -1}
    # the wall's top face is 1 m below the camera, straight along its axis: depth is the distance from the camera
    # plane, so every wall pixel reads 1 m, the corners too
    d = im['depth'][im['seg'] == wall]
    assert np.allclose(d, 1.0, atol=1e-4)
    assert abs(float(im['depth'][0, 0]) - 1.0) < 1e-4


def test_row_zero_is_the_top_of_the_image(rendered):
    m, im, _ = rendered
    vv, uu = np.nonzero(im['seg'] == m.geom('up').id)
    # the sphere sits at MuJoCo camera +y (image up): its pixels are in the upper half, centred left-right
    assert vv.mean() < 120 - 30
    assert abs(uu.mean() - 159.5) < 0.5


def test_camera_frames_and_models():
    import mujoco
    from dexvantage.validation import mjrender as MR
    m = mujoco.MjModel.from_xml_string(_scene_xml(extra='<camera name="s" pos="0 0 1" xyaxes="1 0 0 0 1 0" '
                                                        'focalpixel="300 310" principalpixel="10 -6" '
                                                        'sensorsize="0.0032 0.0024" resolution="320 240"/>'))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    R, p = MR.mujoco_camera(m, d, 'c')
    # MuJoCo looks down its −z axis (here world −z) with +y up: optical z is world −z, optical y (down) world −y
    assert np.allclose(R[:, 2], [0, 0, -1]) and np.allclose(R[:, 1], [0, -1, 0]) and np.allclose(R[:, 0], [1, 0, 0])
    assert np.allclose(MR.world_to_optical(R, p, [[0.1, 0.2, 0.5]]), [[0.1, -0.2, 0.5]])
    cm = MR.camera_model(m, 'c', 320, 240)
    assert cm.fy == pytest.approx(120 / math.tan(math.radians(30))) and cm.fx == pytest.approx(cm.fy)
    assert (cm.cx, cm.cy) == (159.5, 119.5)
    assert cm.fov_deg()['v'] == pytest.approx(60.0)
    s = MR.camera_model(m, 's', 320, 240)
    assert (s.fx, s.fy) == pytest.approx((300, 310))
    assert (s.cx, s.cy) == pytest.approx((159.5 - 10, 119.5 + 6))          # the measured sign convention
    # look_at: z towards the target, y down (image up towards world +z)
    L = MR.look_at([0, 0, 1], [1, 0, 1])
    assert np.allclose(L[:, 2], [1, 0, 0]) and np.allclose(L[:, 1], [0, 0, -1]) and np.allclose(L @ L.T, np.eye(3))
    D = MR.look_at([0, 0, 1], [0, 0, 0])                                   # straight down: still a rotation
    assert np.allclose(D[:, 2], [0, 0, -1]) and np.isclose(np.linalg.det(D), 1.0)


def test_add_camera_reproduces_the_optical_frame():
    import mujoco
    from dexvantage.validation import mjrender as MR
    spec = mujoco.MjSpec()
    R = MR.look_at([0.3, -0.2, 0.8], [0.0, 0.1, 0.0])
    MR.add_camera(spec, 'k', R, [0.3, -0.2, 0.8], 45.0, 200, 100)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    R2, p2 = MR.mujoco_camera(m, d, 'k')
    assert np.allclose(R2, R, atol=1e-9) and np.allclose(p2, [0.3, -0.2, 0.8])
    assert float(m.cam_fovy[0]) == 45.0 and list(m.cam_resolution[0]) == [200, 100]


def test_without_egl_the_check_fails_and_says_why(monkeypatch):
    from dexvantage.validation import mjrender as MR
    from dexvantage.validation import suite as S

    def fake_run(*a, **k):
        return subprocess.CompletedProcess(a, 3, b'error\tRuntimeError: cannot create an EGL display\n', b'')
    monkeypatch.setattr(MR.subprocess, 'run', fake_run)
    import mujoco
    with pytest.raises(MR.RenderUnavailable, match='EGL'):
        MR.render([{'model': mujoco.MjModel.from_xml_string(_scene_xml()), 'camera': 'c', 'width': 32,
                    'height': 24}])
    S.load_checks()
    ctx = S.Ctx(None, None, None, S.ValidationSettings(quick=True))
    try:
        for cid in ('optics.mujoco_fiducial_masks', 'occlusion.mujoco_box_spheres'):
            c = S.run_one(next(s for s in S.CHECKS if s.id == cid), ctx)
            assert not c.passed and 'RenderUnavailable' in c.exception and 'EGL display' in c.exception
    finally:
        ctx.close()


def test_worker_that_dies_is_explained(monkeypatch):
    from dexvantage.validation import mjrender as MR

    def fake_run(*a, **k):
        return subprocess.CompletedProcess(a, -11, b'', b'Segmentation fault\n')
    monkeypatch.setattr(MR.subprocess, 'run', fake_run)
    import mujoco
    with pytest.raises(MR.RenderUnavailable, match='Segmentation fault'):
        MR.render([{'model': mujoco.MjModel.from_xml_string(_scene_xml()), 'camera': 'c', 'width': 32,
                    'height': 24}])


@pytest.fixture(scope='module')
def quick_results():
    from dexvantage.validation import suite as S
    S.load_checks()
    ids = ('optics.mujoco_fiducial_masks', 'optics.mujoco_fiducial_centres', 'occlusion.mujoco_box_spheres',
           'occlusion.mujoco_ur5e', 'occlusion.samples_vs_raycast')
    ctx = S.Ctx(None, None, None, S.ValidationSettings(quick=True))
    try:
        return {cid: S.run_one(next(s for s in S.CHECKS if s.id == cid), ctx) for cid in ids}
    finally:
        ctx.close()


def test_checks_are_registered_with_their_models():
    from dexvantage.validation import catalogue
    cat = {c['id']: c for c in catalogue()}
    assert cat['optics.mujoco_fiducial_masks']['model'] == 'optics'
    assert cat['occlusion.mujoco_ur5e']['model'] == 'occlusion'
    assert cat['occlusion.samples_vs_raycast']['cost'] == 'study'
    assert cat['kinematics.obstacle_distance']['model'] == 'kinematics'
    from dexvantage.validation.references import REFERENCES
    for cid in ('optics.mujoco_fiducial_masks', 'occlusion.mujoco_box_spheres'):
        assert all(k in REFERENCES for k in cat[cid]['references'])


@pytest.mark.mujoco_render
def test_optics_checks_pass(quick_results):
    m = quick_results['optics.mujoco_fiducial_masks']
    assert m.passed, (m.error, m.exception)
    assert m.detail['missing_masks'] == 0 and m.detail['min_mask_iou'] >= 0.95
    assert set(m.detail['worst_px_by_camera']) == {'fovy', 'focal', 'principal'}
    c = quick_results['optics.mujoco_fiducial_centres']
    assert c.passed, (c.error, c.exception)
    # the uncorrected centre is off by more than the corrected one: the ellipse correction is doing something
    assert c.detail['perspective_bias_max_px'] > c.detail['median_px']


@pytest.mark.mujoco_render
def test_occlusion_mujoco_checks_pass(quick_results):
    for cid in ('occlusion.mujoco_box_spheres', 'occlusion.mujoco_ur5e'):
        c = quick_results[cid]
        assert c.passed, (cid, c.error, c.exception)
        assert c.detail['pixels'] > 1000 and c.detail['agree_exact_pct'] > 99.9
        assert 0.0 < c.detail['mujoco_visible_fraction'] < 1.0       # something is occluded, something seen
        assert any(k.startswith('agree_cube_') for k in c.detail)
    assert quick_results['occlusion.mujoco_ur5e'].detail['robot_pixels'] > 0


def test_samples_check_passes(quick_results):
    c = quick_results['occlusion.samples_vs_raycast']
    assert c.passed, (c.error, c.exception)
    assert len(c.detail['backends']) == 1 and c.detail['face_size'] == 256        # quick: the first backend
    assert c.tolerance == pytest.approx(max(c.detail['half_texel_band_fraction'], 1e-4))
    assert c.detail['disagree_outside_silhouette_band'] == 0
    names = ' '.join(c.detail['maps_listed'])
    assert 'so100' in names and 'ur5e' in names
