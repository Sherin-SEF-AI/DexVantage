"""The Gaussian rasteriser (core/splat_raster.py), ported from Vantage's tests/test_splat_reproject.py (the rasteriser
part) and extended.

References: the closed form of one Gaussian's footprint (EWA splatting with the 0.3 px² dilation) and of
front-to-back alpha compositing, the numpy reference written separately from the torch path (no tiles, the projection
through core.optics.CameraModel.project itself), the torch projector against CameraModel.project for every lens model,
the gradients against finite differences (torch.autograd.gradcheck in double precision), the SSIM against a numpy and
scipy.ndimage version. The CUDA tests compare the CUDA path with the numpy reference on fixed Gaussians and skip, with
the reason, where there is no CUDA; the gsplat test skips where gsplat is not installed.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from dexvantage.core import splat_raster as SR
from dexvantage.core import splat_reproject as RP
from dexvantage.core.optics import CameraModel

NO_TORCH = 'PyTorch is not installed (the splat extra): the torch rasteriser is not tested'
NO_CUDA = 'no CUDA device here: the CUDA path of the rasteriser is not tested'
NO_GSPLAT = 'gsplat is not installed or has no CUDA here (scripts/build_gsplat.sh builds it from source)'


def _torch():
    return pytest.importorskip('torch', reason=NO_TORCH)


def _one(P, sigma, colour, opacity):
    P = np.asarray(P, float).reshape(-1, 3)
    n = len(P)
    R = np.repeat(np.eye(3)[None], n, 0)
    s = np.broadcast_to(np.asarray(sigma, float).reshape(n, -1), (n, 3))
    return P, SR.covariance(R, s), np.asarray(colour, float).reshape(n, -1), np.asarray(opacity, float).reshape(n)


def _fixed(n=300, seed=0) -> RP.GaussianSet:
    rng = np.random.default_rng(seed)
    return RP.GaussianSet(rng.uniform([-0.3, -0.2, 0.8], [0.3, 0.2, 1.6], (n, 3)), rng.uniform(0.005, 0.04, (n, 3)),
                          rng.normal(size=(n, 4)), rng.uniform(0.2, 0.95, n), rng.uniform(0, 1, (n, 3)))


CAMS = {'pinhole': CameraModel(64, 48, 60, 60, 31.5, 23.5),
        'brown': CameraModel(64, 48, 60, 58, 30.0, 24.0, 'brown', (-0.1, 0.02, 0.001, -0.001, 0.0)),
        'kb': CameraModel(64, 48, 40, 40, 31.5, 23.5, 'kb', (0.02, -0.01, 0.0, 0.0)),
        'equidistant': CameraModel(64, 48, 30, 30, 31.5, 23.5, 'equidistant'),
        'equisolid': CameraModel(64, 48, 30, 30, 31.5, 23.5, 'equisolid'),
        'stereographic': CameraModel(64, 48, 30, 30, 31.5, 23.5, 'stereographic')}


# ----------------------------------------------------------------------------- closed forms (numpy reference)
def test_footprint_matches_the_closed_form():
    """One isotropic Gaussian on the axis of an equidistant lens: alpha = o·exp(−½ dᵀ Σ₂⁻¹ d), Σ₂ = (fσ/z)² I + 0.3 I
    (the projection's Jacobian at the axis is f/z), at every pixel centre (integers: OpenCV's convention), cut where
    the alpha falls below 1/255 (o = 0.3 < 0.35: the 1/255 contour lies inside 3σ)."""
    f, z, sig, op = 100.0, 10.0, 0.08, 0.3
    cam = CameraModel(64, 64, f, f, 32.0, 32.0, 'equidistant')
    out = SR.splat_numpy(*_one([0, 0, z], [sig], [1.0], [op]), cam)
    a = out['alpha'].reshape(64, 64)
    s2 = (f * sig / z) ** 2 + SR.LOWPASS
    yy, xx = np.mgrid[0:64, 0:64].astype(float)
    ref = op * np.exp(-0.5 * ((xx - 32) ** 2 + (yy - 32) ** 2) / s2)
    ref = np.where(ref >= SR.ALPHA_MIN, np.minimum(ref, SR.ALPHA_MAX), 0.0)
    assert np.max(np.abs(a - ref)) < 2e-4
    assert out['image'].reshape(64, 64)[32, 32] == pytest.approx(a[32, 32], abs=1e-12)


def test_footprint_is_cut_at_three_sigma_for_a_dense_gaussian():
    """o = 0.9: the cut-off is the 3σ ellipse (module docstring, Footprint), so a pixel 3.1σ away gets nothing and
    one 2.9σ away its full alpha."""
    f, z, sig = 100.0, 10.0, 0.2
    cam = CameraModel(64, 64, f, f, 32.0, 32.0)
    a = SR.splat_numpy(*_one([0, 0, z], [sig], [1.0], [0.9]), cam)['alpha'].reshape(64, 64)
    s = math.sqrt((f * sig / z) ** 2 + SR.LOWPASS)
    for d in range(0, 12):
        assert (a[32, 32 + d] > 0) == (d / s <= 3.0), d
    assert a[32, 32 + int(3 * s)] == pytest.approx(0.9 * math.exp(-0.5 * (int(3 * s) / s) ** 2), rel=1e-9)


def test_compositing_is_front_to_back_whatever_the_input_order():
    cam = CameraModel(32, 32, 50, 50, 16.0, 16.0, 'kb', (0.01, 0.0, 0.0, 0.0))
    near = ([0, 0, 5.0], [0.3], [1.0, 0.0, 0.0], [0.6])
    far = ([0, 0, 10.0], [0.6], [0.0, 1.0, 0.0], [0.8])
    bg = np.tile([0.0, 0.0, 1.0], (32 * 32, 1))
    outs = []
    for order in ((near, far), (far, near)):
        P, cov, col, o = (np.concatenate(x) for x in zip(*[_one(*g) for g in order]))
        outs.append(SR.splat_numpy(P, cov, col, o, cam, bg=bg)['image'].reshape(32, 32, 3))
    assert np.allclose(outs[0], outs[1])
    a1, a2 = 0.6, 0.8                                     # the centre pixel is on the axis: d = 0, alpha = o
    assert outs[0][16, 16] == pytest.approx([a1, (1 - a1) * a2, (1 - a1) * (1 - a2)], abs=1e-9)


# ----------------------------------------------------------------------------- the torch path
@pytest.mark.parametrize('name', sorted(CAMS))
def test_torch_projector_is_the_camera_model(name):
    torch = _torch()
    cam = CAMS[name]
    rng = np.random.default_rng(3)
    D = rng.normal(size=(500, 3))
    D[:, 2] = np.abs(D[:, 2]) + 0.4
    uv, _ok = cam.project(D)
    fin = np.isfinite(uv).all(axis=1)
    u, v, ok = SR.TorchProjector(cam)(torch.tensor(D, dtype=torch.float64))
    assert np.array_equal(ok.numpy(), fin)
    assert np.allclose(u.numpy()[fin], uv[fin, 0], atol=1e-9) and np.allclose(v.numpy()[fin], uv[fin, 1], atol=1e-9)


@pytest.mark.parametrize('name', ['pinhole', 'brown', 'kb', 'equisolid'])
def test_torch_cpu_matches_the_numpy_reference(name):
    """The tiled torch path (CPU, float64) against the untiled numpy reference on 300 fixed Gaussians: colour,
    alpha, the dominant-Gaussian index and depth agree to round-off (measured 2·10⁻¹⁴)."""
    torch = _torch()
    gs, cam = _fixed(), CAMS[name]
    a = RP.render(gs, cam, np.eye(4), backend='numpy', want_index=True, depth=True)
    b = RP.render(gs, cam, np.eye(4), backend='torch', device=torch.device('cpu'), dtype='float64',
                  want_index=True, depth=True)
    assert a['alpha'].mean() > 0.05                      # the Gaussians are in view
    assert np.abs(a['rgb'] - b['rgb']).max() < 1e-9 and np.abs(a['alpha'] - b['alpha']).max() < 1e-9
    assert np.abs(a['depth'] - b['depth']).max() < 1e-9 and np.array_equal(a['index'], b['index'])


@pytest.mark.parametrize('name', ['pinhole', 'brown', 'kb'])
def test_cuda_matches_the_numpy_reference(name):
    """The CUDA path (float32) against the numpy reference (float64) on fixed Gaussians: within 10⁻⁴ (measured
    1.3·10⁻⁵ on an RTX 5080: float32 round-off of the finite-difference Jacobian)."""
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip(NO_CUDA)
    gs, cam = _fixed(), CAMS[name]
    a = RP.render(gs, cam, np.eye(4), backend='numpy', want_index=True)
    c = RP.render(gs, cam, np.eye(4), backend='torch', device=torch.device('cuda', 0), want_index=True)
    assert np.abs(a['rgb'] - c['rgb']).max() < 1e-4 and np.abs(a['alpha'] - c['alpha']).max() < 1e-4
    assert (a['index'] != c['index']).mean() < 0.01


def test_gsplat_agrees_with_the_torch_path():
    """gsplat's rasteriser against the torch path on fixed Gaussians through a pinhole: the mean difference is small;
    the largest differs where gsplat's 3σ tile rule and z order differ from this module's (module docstring)."""
    torch = _torch()
    if not SR.gsplat_ok():
        pytest.skip(NO_GSPLAT)
    gs, cam = _fixed(), CAMS['pinhole']
    a = RP.render(gs, cam, np.eye(4), backend='torch', device=torch.device('cuda', 0))
    g = RP.render(gs, cam, np.eye(4), backend='gsplat', device=torch.device('cuda', 0))
    assert g['backend'] == 'gsplat'
    assert np.abs(a['rgb'] - g['rgb']).mean() < 0.01


def test_gradients_match_finite_differences():
    torch = _torch()
    cam = CameraModel(24, 24, 40, 40, 11.5, 11.5, 'equisolid')
    P0 = torch.tensor([[0.2, -0.1, 4.0], [-0.3, 0.2, 6.0]], dtype=torch.float64)
    q = torch.tensor([[0.1, 0.2, 0.0, 1.0], [0.0, -0.3, 0.2, 0.9]], dtype=torch.float64)
    col0 = torch.tensor([[0.8, 0.2, 0.1], [0.1, 0.5, 0.9]], dtype=torch.float64)

    def f(P, log_s, op, col):
        cov = SR.covariance(SR.quat_to_rot(q), torch.exp(log_s))
        return SR.splat(P, cov, col, op, cam)['image'].sum(0)
    args = (P0.clone().requires_grad_(), torch.log(torch.tensor([[0.2, 0.3, 0.1], [0.4, 0.2, 0.3]],
                                                                dtype=torch.float64)).requires_grad_(),
            torch.tensor([0.3, 0.32], dtype=torch.float64, requires_grad=True), col0.clone().requires_grad_())
    assert torch.autograd.gradcheck(f, args, eps=1e-6, atol=1e-5, rtol=1e-3)


def test_quaternions_are_xyzw_and_round_trip():
    rng = np.random.default_rng(1)
    q = rng.normal(size=(50, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    R = SR.quat_to_rot_np(q)
    from scipy.spatial.transform import Rotation
    assert np.allclose(R, Rotation.from_quat(q).as_matrix(), atol=1e-12)        # scipy: scalar last
    back = SR.rot_to_quat_np(R)
    assert np.allclose(np.abs(np.einsum('ij,ij->i', back, q)), 1.0, atol=1e-12) and (back[:, 3] >= 0).all()
    torch = pytest.importorskip('torch', reason=NO_TORCH)
    assert np.allclose(SR.quat_to_rot(torch.tensor(q)).numpy(), R, atol=1e-12)


def test_ssim_torch_matches_numpy_and_psnr():
    torch = _torch()
    rng = np.random.default_rng(2)
    a = rng.random((20, 24, 3))
    b = np.clip(a + rng.normal(0, 0.05, a.shape), 0, 1)
    t = float(SR.ssim_map(torch.tensor(a), torch.tensor(b)).mean())
    assert t == pytest.approx(SR.ssim_np(a, b), abs=1e-9)
    assert SR.ssim_np(a, a) == pytest.approx(1.0) and SR.psnr(a, a) == 99.0
    assert SR.psnr(np.zeros(4), np.full(4, 0.1)) == pytest.approx(20.0)


def test_backend_choice_and_refusals():
    assert RP.backend_for('numpy', CAMS['kb']) == 'numpy'
    with pytest.raises(ValueError, match='back end'):
        RP.backend_for('vulkan', CAMS['kb'])
    if SR.torch_ok():
        assert RP.backend_for('auto', CAMS['kb']) == 'torch'      # gsplat is used for pinhole cameras only


def test_the_projection_jacobian_is_the_closed_form_and_the_ewa_covariance_the_sampled_one():
    """The footprint's Jacobian (central differences, step 10⁻³ of the range) against the pinhole's written out,
    J = [[fx/z, 0, −fx x/z²], [0, fy/z, −fy y/z²]] (off the axis, where the closed-form footprint test cannot see
    it), and the EWA covariance J Σ Jᵀ of a small rotated anisotropic Gaussian against the covariance of 200,000 of
    its samples projected through the lens model itself (pinhole and an equidistant fisheye): an independent
    Monte Carlo reference. Tolerances: 10⁻⁵ relative (the difference step's truncation, measured 2.9 × 10⁻⁶) and 2 %
    (the sampling error of a covariance from 200,000 samples is about 0.3 %; the affine approximation's own error
    for a Gaussian of a few millimetres at 1.2 m is far below that)."""
    torch = _torch()
    rng = np.random.default_rng(5)
    cam = CameraModel(640, 480, 500, 480, 319.5, 239.5)
    P = np.c_[rng.uniform(-0.5, 0.5, (100, 2)), rng.uniform(0.3, 3.0, 100)]
    Pt = torch.tensor(P)
    J = SR._jacobian(SR.TorchProjector(cam), Pt, torch.linalg.norm(Pt, dim=1)).numpy()
    x, y, z = P.T
    ref = np.zeros((100, 2, 3))
    ref[:, 0, 0], ref[:, 0, 2] = 500 / z, -500 * x / z ** 2
    ref[:, 1, 1], ref[:, 1, 2] = 480 / z, -480 * y / z ** 2
    assert (np.abs(J - ref).max(axis=(1, 2)) / np.abs(ref).max(axis=(1, 2))).max() < 1e-5
    for c in (cam, CameraModel(640, 480, 200, 200, 319.5, 239.5, 'equidistant')):
        mu = np.array([0.3, -0.2, 1.2])
        R = SR.quat_to_rot_np([0.3, -0.5, 0.2, 0.8])[0]
        S = R @ np.diag(np.array([0.004, 0.002, 0.0008]) ** 2) @ R.T
        uv, ok = c.project(rng.multivariate_normal(mu, S, 200000))
        Jm = SR._jacobian(SR.TorchProjector(c), torch.tensor(mu[None]), torch.tensor([np.linalg.norm(mu)])).numpy()[0]
        mc = np.cov(uv[ok].T)
        assert np.abs(Jm @ S @ Jm.T - mc).max() / np.abs(mc).max() < 0.02, c.model


def test_an_image_background_is_not_sent_to_gsplat(monkeypatch):
    """gsplat takes one background colour: with an image as the background, 'auto' renders through torch (gsplat
    would have used the image's first pixel everywhere)."""
    _torch()
    monkeypatch.setattr(SR, 'gsplat_ok', lambda: True)
    cam = CAMS['pinhole']
    assert RP.backend_for('auto', cam) == 'gsplat'
    bg = np.zeros((cam.height, cam.width, 3))
    bg[:, cam.width // 2:] = 1.0
    gs = RP.GaussianSet(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 4)), np.zeros(0), np.zeros((0, 3)))
    out = RP.render(gs, cam, np.eye(4), backend='auto', bg=bg)
    assert out['backend'] == 'torch' and np.allclose(out['rgb'], bg)
