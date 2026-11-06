"""core/optics: the camera models (ported tests from Vantage tests/test_core.py and test_econ_modules.py, adapted to
pixels and OpenCV's pixel convention), pixel density, defocus and the optics validation checks."""
import math

import numpy as np
import pytest

from dexvantage.core.optics import CameraModel, blur_circle_px, depth_of_field, fit_fov, incidence_angle
from dexvantage.core.optics import lens as L

RADIAL = ('pinhole', 'equidistant', 'equisolid', 'stereographic', 'kb')
KB = (0.03, -0.012, 0.002, -0.0002)


def _ray(theta, phi=0.0):
    return np.array([[math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)]])


def _cams():
    return [CameraModel(1280, 720, 640.0, 641.0, 639.0, 361.0, 'pinhole'),
            CameraModel(1280, 720, 640.0, 641.0, 639.0, 361.0, 'brown', (-0.12, 0.03, 4e-4, -3e-4, -0.004)),
            CameraModel(1280, 720, 640.0, 641.0, 639.0, 361.0, 'brown',
                        (0.1, -0.02, 1e-4, 2e-4, 0.001, 0.05, -0.01, 0.002)),
            CameraModel(1920, 1080, 600.0, 600.0, 959.5, 539.5, 'kb', KB),
            CameraModel(1920, 1080, 600.0, 600.0, 959.5, 539.5, 'equidistant'),
            CameraModel(1920, 1080, 600.0, 600.0, 959.5, 539.5, 'equisolid'),
            CameraModel(1920, 1080, 600.0, 600.0, 959.5, 539.5, 'stereographic')]


# ----------------------------------------------------------------------------- radial functions (ported)
@pytest.mark.parametrize('model', RADIAL)
def test_theta_of_r_inverts_r_of_theta(model):
    d = KB if model == 'kb' else ()
    th = np.radians([0.0, 5, 20, 40, 60, 80])
    r = L.r_of_theta(th, model, d)
    assert np.allclose(L.theta_of_r(r, model, d), th, atol=1e-12)
    # the derivative is the derivative
    h = 1e-6
    fd = (L.r_of_theta(th[1:] + h, model, d) - L.r_of_theta(th[1:] - h, model, d)) / (2 * h)
    assert np.allclose(L.dr_dtheta(th[1:], model, d), fd, rtol=1e-6)


def test_kb_inverse_refuses_radii_beyond_the_valid_angle():
    t = L.theta_of_r(np.array([0.5, 3.0]), 'kb', (0, 0, 0, 0), max_theta=1.0)
    assert t[0] == pytest.approx(0.5) and np.isnan(t[1])


# ----------------------------------------------------------------------------- projection and unprojection
@pytest.mark.parametrize('k', range(7))
def test_unproject_then_project_is_the_pixel(k):
    cam = _cams()[k]
    rng = np.random.default_rng(k)
    uv = np.c_[rng.uniform(-0.5, cam.width - 0.5, 4000), rng.uniform(-0.5, cam.height - 0.5, 4000)]
    uv[:4] = [[-0.5, -0.5], [cam.width - 0.5, cam.height - 0.5], [cam.cx, cam.cy], [0, cam.height - 1]]
    rays, ok = cam.unproject(uv, with_ok=True)
    assert ok.all()
    assert np.allclose(np.linalg.norm(rays, axis=1), 1.0)
    back, okb = cam.project(rays * 2.5)
    assert okb.all() and np.abs(back - uv).max() < 1e-7


def test_project_matches_opencv():
    cv2 = pytest.importorskip('cv2')
    rng = np.random.default_rng(1)
    P = np.c_[rng.uniform(-0.5, 0.5, (500, 2)), rng.uniform(0.5, 2, 500)]
    for cam in _cams()[:4]:
        K, D, model = cam.to_opencv()
        uv, _ = cam.project(P)
        if model == 'kb':
            ref = cv2.fisheye.projectPoints(P.reshape(-1, 1, 3), np.zeros((3, 1)), np.zeros((3, 1)), K, D)[0]
        else:
            ref = cv2.projectPoints(P.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)[0]
        assert np.abs(uv - ref.reshape(-1, 2)).max() < 1e-8, cam.model


def test_shapes_follow_the_input():
    cam = _cams()[0]
    uv, ok = cam.project(np.array([0.0, 0.0, 1.0]))
    assert uv.shape == (2,) and ok.shape == () and bool(ok)
    assert uv == pytest.approx([cam.cx, cam.cy])
    grid = np.zeros((4, 5, 3))
    grid[..., 2] = 1.0
    uv, ok = cam.project(grid)
    assert uv.shape == (4, 5, 2) and ok.shape == (4, 5)
    assert cam.unproject(uv).shape == (4, 5, 3)
    assert cam.angle_to_axis(grid).shape == (4, 5)
    with pytest.raises(ValueError, match='shape'):
        cam.project(np.zeros((3, 2)))


def test_points_behind_or_beyond_the_valid_angle_are_never_wrapped():
    cam = _cams()[0]
    # behind a pinhole camera: dividing by z would land it on the image
    uv, ok = cam.project(np.array([[-0.1, -0.1, -1.0], [0.0, 0.0, 0.0]]))
    assert not ok.any() and np.isnan(uv).all()
    # a Kannala-Brandt curve that folds back at about 70°: a ray at 100° would map inside the image if wrapped
    fold = CameraModel(1000, 1000, 300.0, 300.0, 499.5, 499.5, 'kb', (-0.2, 0.0, 0.0, 0.0))
    assert math.degrees(fold.max_theta) == pytest.approx(math.degrees(math.sqrt(1 / 0.6)), abs=1e-6)
    uv, ok = fold.project(_ray(math.radians(100)))
    assert not ok[0] and np.isnan(uv).all()
    naive = 300 * L.r_of_theta(math.radians(100), 'kb', fold.dist) + 499.5
    assert 0 <= naive < 1000                    # (what wrapping would have given: a pixel inside the image)
    # a fisheye sees behind its own plane
    fe = _cams()[4]
    uv, ok = fe.project(_ray(math.radians(95)))
    assert np.isfinite(uv).all()


def test_valid_angle_beyond_the_fold_is_refused_with_a_reason():
    with pytest.raises(ValueError, match='folds back'):
        CameraModel(1000, 1000, 300.0, 300.0, 499.5, 499.5, 'kb', (-0.2, 0.0, 0.0, 0.0), max_theta=math.radians(80))
    with pytest.raises(ValueError, match='in front'):
        CameraModel(640, 480, 500.0, 500.0, 319.5, 239.5, 'pinhole', max_theta=math.radians(100))


def test_brown_fold_counts_the_tangential_terms():
    # tangential terms grow as r² against the radial r: a model with only p1 still folds before 90°
    assert L.brown_fold_angle((0.0, 0.0, 0.01, 0.0, 0.0)) < math.radians(89)
    assert L.brown_fold_angle(()) == pytest.approx(math.pi / 2)
    # strong barrel distortion folds back inside the image: those corner pixels have no ray (NaN), and say so
    cam = CameraModel(1280, 720, 640, 642, 639.2, 359.7, 'brown', (-0.28, 0.09, 0.001, -0.0005, -0.012, 0.01, 0.002,
                                                                    0.0003))
    rays, ok = cam.unproject(np.array([[0.0, 0.0], [640.0, 360.0]]), with_ok=True)
    assert list(ok) == [False, True] and np.isnan(rays[0]).all()


def test_circular_fisheye_has_no_rays_in_the_corners():
    cam = CameraModel(1920, 1080, 600.0, 600.0, 959.5, 539.5, 'equidistant', max_theta=math.radians(95))
    rays = cam.pixel_rays(stride=8)
    assert rays.shape == (math.ceil(1080 / 8), 1920 // 8, 3)
    assert np.isnan(rays[0, 0]).all() and np.isfinite(rays[rays.shape[0] // 2, rays.shape[1] // 2]).all()
    f = cam.fov_deg()
    assert f['d'] == pytest.approx(190.0)       # corners beyond the image circle count at the valid angle
    assert f['h'] == pytest.approx(math.degrees(960 / 600) * 2)


def test_pixel_rays_are_the_pixel_centres():
    cam = CameraModel.from_fov(64, 48, 90.0)
    rays = cam.pixel_rays()
    assert rays.shape == (48, 64, 3)
    uv, ok = cam.project(rays)
    U, V = np.meshgrid(np.arange(64.0), np.arange(48.0))
    assert ok.all() and np.allclose(uv[..., 0], U) and np.allclose(uv[..., 1], V)
    with pytest.raises(ValueError):
        cam.pixel_rays(0)


# ----------------------------------------------------------------------------- fields of view and fitting
def test_from_fov_uses_opencv_pixel_centres():
    cam = CameraModel.from_fov(1280, 720, 90.0)
    assert (cam.cx, cam.cy) == (639.5, 359.5)
    assert cam.fx == pytest.approx(640.0) and cam.fy == cam.fx            # the edge u = −0.5 is 640 px out
    left = cam.unproject(np.array([[-0.5, 359.5]]))[0]
    assert math.degrees(math.atan2(-left[0], left[2])) == pytest.approx(45.0)
    f = cam.fov_deg()
    assert f['h'] == pytest.approx(90.0) and f['v'] == pytest.approx(2 * math.degrees(math.atan(360 / 640)))
    cam = CameraModel.from_fov(1280, 720, 87.0, 58.0)                    # both given: fy from the vertical
    assert cam.fov_deg()['v'] == pytest.approx(58.0) and cam.fy != pytest.approx(cam.fx)


@pytest.mark.parametrize('model', ('pinhole', 'brown', 'equidistant', 'equisolid', 'stereographic', 'kb'))
def test_from_fov_reproduces_the_horizontal_fov(model):
    for h in (40.0, 75.0, 120.0):
        cam = CameraModel.from_fov(848, 480, h, model=model)
        assert cam.fov_deg()['h'] == pytest.approx(h, abs=1e-9)
        assert cam.fx == pytest.approx(cam.fy)
    assert CameraModel.from_fov(848, 480, 90, model='brown').dist == (0.0,) * 5


def test_kb_fit_passes_through_a_stated_hvd():
    """Vantage's wide-lens figures (H 128.7°, V 98.7°, D 170.1° on 1920 × 1536): reproduced to rounding (Vantage's
    weighted fit came within 1°; the constrained fit passes through the points)."""
    cam, rep = fit_fov(1920, 1536, 128.7, 98.7, 'kb', dfov_deg=170.1)
    f = cam.fov_deg()
    assert (f['h'], f['v'], f['d']) == pytest.approx((128.7, 98.7, 170.1), abs=1e-8)
    assert rep['fit'] == 'exact' and rep['base'] in L.FIT_BASES and rep['max_error_deg'] < 1e-8
    assert 'model, not a measurement' in rep['note']
    assert L.kb_monotonic(cam.dist, cam.max_theta)
    # the valid angle reaches FIT_MARGIN_DEG past the corner
    assert math.degrees(cam.max_theta) == pytest.approx(170.1 / 2 + L.FIT_MARGIN_DEG, abs=1e-6)


def test_kb_fit_falls_back_and_from_fov_refuses_figures_no_lens_has():
    """H 100°, V 50° on 4:3 square pixels would need the curve to fold back: fit_fov reports its best effort and how
    far off it is; from_fov refuses with a reason."""
    cam, rep = fit_fov(640, 480, 100.0, 50.0, 'kb')
    assert rep['fit'].startswith('weighted') and rep['max_error_deg'] > 1.0
    assert 'do not fit one such lens' in rep['note']
    with pytest.raises(ValueError, match='cannot be reproduced'):
        CameraModel.from_fov(640, 480, 100.0, 50.0, model='kb')


def test_the_fit_helpers_ported_from_vantage():
    hs = (960.0, 540.0, math.hypot(1920, 1080) / 2)
    # an equisolid lens's own FOV makes equisolid the best base, with its focal length
    f0 = 700.0
    sheet = tuple(2 * math.degrees(2 * math.asin(h / f0 / 2)) for h in hs)
    b, f, rel = L.best_base(sheet, hs)
    assert b == 'equisolid' and f == pytest.approx(f0) and rel < 1e-12
    # the weighted fit at Vantage's strongest weight comes close; the exact one passes through
    kw = L.fit_kb_to_fov(f, sheet, hs, base='stereographic', weight=50.0)
    ke = L.fit_kb_exact(f, sheet, hs, base='stereographic')
    th = np.radians(np.array(sheet) / 2)
    assert np.abs(f * L.r_of_theta(th, 'kb', ke) - hs).max() < 1e-8
    assert np.abs(f * L.r_of_theta(th, 'kb', kw) - hs).max() < 5.0


def test_bad_fields_of_view_are_explained():
    with pytest.raises(ValueError, match='between 0° and 180°'):
        CameraModel.from_fov(640, 480, 190.0)
    with pytest.raises(ValueError, match='diagonal'):
        fit_fov(640, 480, 90.0, None, 'pinhole', dfov_deg=100.0)
    with pytest.raises(ValueError, match='unknown lens model'):
        CameraModel.from_fov(640, 480, 90.0, model='fishy')


# ----------------------------------------------------------------------------- pixel density and angles
def test_pixel_density_on_the_axis_is_f_over_z():
    cam = CameraModel(1280, 720, 900.0, 900.0, 639.5, 359.5)
    P = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 2.0]])
    assert cam.pixels_per_metre(P) == pytest.approx([1800.0, 450.0])
    assert cam.pixels_per_mm(P) == pytest.approx([1.8, 0.45])
    # off axis a facing patch is seen at R and cos³θ: fx fy / (R² cos³ θ)
    th = math.radians(30)
    p = 1.2 * _ray(th)
    assert cam.pixels_per_metre(p)[0] == pytest.approx(900 / 1.2 / math.cos(th) ** 1.5)
    # a surface tilted 60° from the line of sight keeps √cos 60° of it
    n = np.array([[0.0, math.sin(math.radians(60)), -math.cos(math.radians(60))]])
    assert cam.pixels_per_metre(np.array([[0.0, 0.0, 1.0]]), n)[0] == pytest.approx(900 * math.sqrt(0.5))
    assert incidence_angle(np.array([[0.0, 0.0, 1.0]]), n)[0] == pytest.approx(math.radians(60))
    # outside the valid angle: no density
    assert np.isnan(cam.pixels_per_metre(np.array([[0.0, 0.0, -1.0]])))[0]


@pytest.mark.parametrize('k', range(7))
def test_pixel_density_matches_finite_differences(k):
    cam = _cams()[k]
    rng = np.random.default_rng(10 + k)
    P = np.c_[rng.uniform(-0.6, 0.6, (50, 2)), rng.uniform(0.6, 1.5, 50)]
    d = P / np.linalg.norm(P, axis=1, keepdims=True)
    e1 = np.cross(d, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(d, e1)
    h = 1e-5
    J = np.stack([(cam.project(P + h * e)[0] - cam.project(P - h * e)[0]) / (2 * h) for e in (e1, e2)], axis=-1)
    assert np.allclose(cam.pixels_per_metre(P), np.sqrt(np.abs(np.linalg.det(J))), rtol=1e-6)


def test_angle_to_axis():
    cam = _cams()[3]
    assert cam.angle_to_axis(_ray(math.radians(37), 1.0))[0] == pytest.approx(math.radians(37))
    assert cam.angle_to_axis(np.array([[0.0, 0.0, -1.0]]))[0] == pytest.approx(math.pi)


# ----------------------------------------------------------------------------- conversions
def test_opencv_round_trip_and_refusals():
    for cam in _cams()[:5]:
        K, D, model = cam.to_opencv()
        back = CameraModel.from_opencv(K, D, cam.width, cam.height, model)
        expected = cam if cam.model != 'equidistant' else CameraModel(cam.width, cam.height, cam.fx, cam.fy, cam.cx,
                                                                      cam.cy, 'kb', (0.0,) * 4)
        assert back.to_opencv()[0] == pytest.approx(expected.to_opencv()[0])
        assert back.dist == expected.dist
    K = np.array([[500.0, 1.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match='skew'):
        CameraModel.from_opencv(K, None, 640, 480, 'pinhole')
    K[0, 1] = 0.0
    assert CameraModel.from_opencv(K, [0.1, 0, 0, 0, 0], 640, 480, 'pinhole').model == 'brown'
    assert CameraModel.from_opencv(K, [0.1, 0, 0, 0], 640, 480, 'fisheye').model == 'kb'
    with pytest.raises(ValueError, match='8 distortion terms'):
        CameraModel.from_opencv(K, np.zeros(14), 640, 480, 'brown')


def test_equisolid_has_a_close_kannala_brandt_form():
    cam = CameraModel.from_fov(1920, 1080, 180.0, model='equisolid')
    k, err = cam.kb_approximation()
    assert err < 0.5                                   # px over the image's angles
    K, D, model = cam.to_opencv()
    assert model == 'kb' and tuple(D) == pytest.approx(k)
    assert _cams()[0].to_opencv()[2] == 'pinhole' and len(_cams()[0].to_opencv()[1]) == 5


def test_dict_round_trip_equality_and_errors():
    for cam in _cams():
        d = cam.to_dict()
        assert d['pixel_centres'] == 'opencv'
        back = CameraModel.from_dict(d)
        assert back == cam and hash(back) == hash(cam)
    with pytest.raises(ValueError, match='lacks fx'):
        CameraModel.from_dict({'width': 10, 'height': 10, 'fy': 1, 'cx': 1, 'cy': 1})
    with pytest.raises(ValueError, match='positive'):
        CameraModel(640, 480, -1.0, 500.0, 319.5, 239.5)
    with pytest.raises(ValueError, match='no distortion terms'):
        CameraModel(640, 480, 500.0, 500.0, 319.5, 239.5, 'equisolid', (0.1,))
    with pytest.raises(ValueError, match='4 distortion terms'):
        CameraModel(640, 480, 500.0, 500.0, 319.5, 239.5, 'kb', (0.1, 0.2))
    with pytest.raises(ValueError, match='whole pixels'):
        CameraModel(640.5, 480, 500.0, 500.0, 319.5, 239.5)


def test_scaled_keeps_the_opencv_convention():
    cam = _cams()[1]
    half = cam.scaled(640, 360)
    assert half.cx == pytest.approx((cam.cx + 0.5) / 2 - 0.5) and half.fx == pytest.approx(cam.fx / 2)
    P = np.array([[0.2, -0.1, 1.0], [-0.3, 0.2, 0.8]])
    uv, _ = cam.project(P)
    uvh, _ = half.project(P)
    assert np.allclose(uvh, (uv + 0.5) / 2 - 0.5)
    assert half.fov_deg() == pytest.approx(cam.fov_deg())


def test_summary_names_the_model():
    s = CameraModel.from_fov(1280, 720, 90.0).summary()
    assert s.startswith('Pinhole, 1280 × 720 px') and '90.0°' in s


# ----------------------------------------------------------------------------- defocus
def test_depth_of_field_hand_example():
    # f 4 mm, f/2, c 0.006 mm: H = 16 / 0.012 + 4 = 1337.33 mm; focused at 0.5 m
    near, far = depth_of_field(4.0, 2.0, 0.5, 0.006)
    H = 16 / 0.012 + 4
    assert near == pytest.approx(0.5 * (H - 4) / (H + 500 - 8))
    assert far == pytest.approx(0.5 * (H - 4) / (H - 500))
    near, far = depth_of_field(4.0, 2.0, 2.0, 0.006)        # beyond the hyperfocal distance
    assert math.isinf(far) and near < 2.0
    near, far = depth_of_field(4.0, 2.0, math.inf, 0.006)
    assert near == pytest.approx((H - 4) / 1000) and math.isinf(far)
    assert depth_of_field(4.0, None, 0.5, 0.006) == (None, None)        # aperture not stated
    with pytest.raises(ValueError, match='f-number'):
        depth_of_field(4.0, -2.0, 0.5, 0.006)
    with pytest.raises(ValueError, match='cannot focus'):
        depth_of_field(4.0, 2.0, 0.003, 0.006)


def test_blur_circle():
    assert blur_circle_px(4.0, 2.0, 0.5, 0.5, 2.0) == pytest.approx(0.0, abs=1e-12)
    z = np.array([0.25, 0.5, 1.0])
    b = blur_circle_px(4.0, 2.0, 0.5, z, 2.0)
    assert b.shape == (3,) and b[1] == pytest.approx(0.0, abs=1e-12) and b[0] > b[2] > 0
    near, far = depth_of_field(4.0, 2.0, 0.5, 0.006)
    assert blur_circle_px(4.0, 2.0, 0.5, near, 2.0) == pytest.approx(3.0)      # 0.006 mm on 2 µm pixels
    assert blur_circle_px(4.0, 2.0, 0.5, far, 2.0) == pytest.approx(3.0)
    assert blur_circle_px(4.0, 2.0, math.inf, 1.0, 2.0) == pytest.approx(2.0 * 4.0 / 1000 / 0.002)
    assert math.isinf(blur_circle_px(4.0, 2.0, 0.5, 0.002, 2.0))               # nearer than the focal length
    assert blur_circle_px(4.0, 2.0, 0.5, 1.0, None) is None
    assert L.DEFAULT_COC_PX == 2.0


# ----------------------------------------------------------------------------- validation
def test_optics_checks_pass_and_cite_their_references():
    from dexvantage import validation as VA
    res = VA.run(settings=VA.ValidationSettings(models=('optics',), quick=True))
    ids = {c.id for c in res.checks}
    assert {'optics.opencv_pinhole_brown', 'optics.opencv_fisheye', 'optics.round_trip', 'optics.from_fov',
            'optics.pixel_density', 'optics.depth_of_field', 'optics.blur_circle'} <= ids
    assert res.ok, [(c.id, c.error, c.exception) for c in res.failed]
    for c in res.checks:
        assert c.refs and all(r in VA.REFERENCES for r in c.refs) and c.note
