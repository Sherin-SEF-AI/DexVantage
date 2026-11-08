"""Checks of the robot-aligned 3D Gaussian Splatting model (core/splat_raster.py, core/splat_reproject.py,
core/sim/splat_rig.py, core/sim/splat_views.py) against independent references.

- splat.footprint_closed_form: one isotropic Gaussian (σ 8 cm, opacity 0.9) 10 m along the axis of a 100 px pinhole,
  rendered by the numpy reference rasteriser, against its footprint written out here: α = o · exp(−½ d² / s²) with
  s² = (f σ / z)² + 0.3 px² (EWA: the projection's Jacobian at the axis is f / z; the 0.3 px² dilation of 3DGS), at
  every pixel centre (OpenCV: integers) inside the 3 s cut-off. The error is the largest |α − closed form|.
- splat.raster_torch_cpu: the torch rasteriser (tiled, 16 × 16 tiles, float64, on the CPU) against the numpy reference
  (no tiles, CameraModel.project itself) on 300 fixed Gaussians through a pinhole, a Brown-Conrady and a
  Kannala-Brandt lens (64 × 48 px). Needs PyTorch (the splat extra). Tolerance 1e-9: float64 round-off (measured
  3.3 × 10⁻¹⁶ to 2.0 × 10⁻¹⁴ when the model was built).
- splat.raster_torch_cuda: the same on the CUDA device in float32 against the numpy reference in float64. Tolerance
  1e-4: float32 round-off in the finite-difference Jacobian (measured 1.1 to 1.3 × 10⁻⁵). Needs a CUDA device; left
  out with the reason where there is none.
- splat.reproject_disparity: splat_reproject.reproject of 500 random pixels and depths from the left to the right
  pinhole of a rectified stereo pair (baseline 6 cm, f 400 px) against the disparity in closed form,
  u_r = u_l − f b / z, v_r = v_l, z_r = z. The error is the largest pixel difference.
- splat.repose_rigid: a rig of 60 Gaussians attached to the SO-100's moving jaw link, posed at 5 random joint states
  and rendered (numpy reference) through a fixed camera, against the same Gaussians left in the link frame and
  rendered through the camera moved into that frame, T_link(q)⁻¹ · T_world_optical, with T_link from the robot model's
  link_pose (a separate call from the rig's forward kinematics). A rigid motion of the scene and the opposite motion
  of the camera give the same image; the error is the largest colour difference over the pixels. Tolerance 1e-5:
  a GaussianSet holds float32 arrays (the rig file's precision), so the posed centres, about 0.3 m from the camera,
  round to about 3 × 10⁻⁸ m, a shift of about 10⁻⁵ px (measured: 5.9 × 10⁻⁷).
- splat.ssim_closed_form: splat_raster.ssim_np of two constant images (levels a and b in [0, 1], 3 channels) against
  SSIM's closed form for constant patches, (2ab + C1) / (a² + b² + C1) with C1 = (0.01)² (the contrast-structure
  term is C2 / C2 = 1), over 40 pairs; and psnr of a constant difference d against −20 log10 d.
- splat.ssim_vs_skimage: splat_raster.ssim_np against scikit-image's structural_similarity set to Wang et al.'s
  implementation (gaussian_weights=True, sigma 1.5, use_sample_covariance False, data_range 1, channel_axis −1,
  full=True) on 20 random image pairs (64 × 48, noise, blur and offsets), compared on the interior 5 px from the
  border (scikit-image pads by reflection and crops, DexVantage replicates the border: the interior does not
  depend on it). Needs scikit-image (the test extra). Tolerance 1e-9. Both filter with scipy.ndimage's Gaussian (the
  same 11-tap kernel), so the interior agrees to the bit (measured 0): the check holds the window, the constants and
  the population covariance to Wang et al.'s, not the filter's arithmetic.
"""
from __future__ import annotations

import math

import numpy as np

from .suite import CUDA_NEED, Outcome, check

TORCH = (('torch', 'splat'),)
REFS = ('kerbl2023_3dgs', 'zwicker2001_ewa')


def _cams():
    from ..core.optics import CameraModel
    return {'pinhole': CameraModel(64, 48, 60, 60, 31.5, 23.5),
            'brown': CameraModel(64, 48, 60, 58, 30.0, 24.0, 'brown', (-0.1, 0.02, 0.001, -0.001, 0.0)),
            'kb': CameraModel(64, 48, 40, 40, 31.5, 23.5, 'kb', (0.02, -0.01, 0.0, 0.0))}


def _fixed(rng, n=300):
    from ..core import splat_reproject as RP
    return RP.GaussianSet(rng.uniform([-0.3, -0.2, 0.8], [0.3, 0.2, 1.6], (n, 3)), rng.uniform(0.005, 0.04, (n, 3)),
                          rng.normal(size=(n, 4)), rng.uniform(0.2, 0.95, n), rng.uniform(0, 1, (n, 3)))


@check('splat.footprint_closed_form', 'splat', 'One Gaussian\'s footprint against the closed form',
       'largest |α − o·exp(−½ d²/s²)| over the pixels of one Gaussian on the axis (numpy reference rasteriser)',
       'the EWA footprint of an isotropic Gaussian written out: s² = (fσ/z)² + 0.3 px²', 1e-12, 'alpha', refs=REFS)
def _footprint(ctx):
    from ..core import splat_raster as SR
    from ..core.optics import CameraModel
    f, z, sig, op = 100.0, 10.0, 0.08, 0.9
    cam = CameraModel(64, 64, f, f, 32.0, 32.0)
    out = SR.splat_numpy(np.array([[0.0, 0.0, z]]), (sig ** 2) * np.eye(3)[None], np.ones((1, 1)), np.array([op]),
                         cam)
    a = out['alpha'].reshape(64, 64)
    s2 = (f * sig / z) ** 2 + 0.3
    v, u = np.mgrid[0:64, 0:64]
    d2 = (u - 32.0) ** 2 + (v - 32.0) ** 2
    m = min(3.0, math.sqrt(2 * math.log(255 * op)))
    inside = d2 / s2 <= m * m * (1 - 1e-9)
    ref = np.minimum(op * np.exp(-0.5 * d2 / s2), 0.99)
    ref = np.where(ref >= 1 / 255, ref, 0.0)
    err = float(np.abs(a[inside] - ref[inside]).max())
    outside = float(np.abs(a[d2 / s2 > m * m * (1 + 1e-9)]).max())
    return Outcome(max(err, outside), {'pixels_inside': int(inside.sum()), 's_px': math.sqrt(s2), 'cutoff_sigma': m,
                                       'largest_outside': outside})


def _torch_vs_numpy(ctx, device: str, dtype: str):
    import torch

    from ..core import splat_reproject as RP
    gs = _fixed(np.random.default_rng(0))
    T = np.eye(4)
    per = {}
    dev = torch.device(device)
    for name, cam in _cams().items():
        a = RP.render(gs, cam, T, 'torch', device=dev, dtype=dtype)['rgb']
        b = RP.render(gs, cam, T, 'numpy')['rgb']
        per[name] = float(np.abs(np.asarray(a, float) - b).max())
    return Outcome(max(per.values()), {'per_lens': per, 'gaussians': gs.n, 'size_px': [64, 48], 'device': device,
                                       'dtype': dtype})


@check('splat.raster_torch_cpu', 'splat', 'Torch rasteriser (CPU, float64) against the numpy reference',
       'largest colour difference over 300 fixed Gaussians through three lens models', 'the numpy rasteriser written '
       'again without tiles (splat_raster.splat_numpy)', 1e-9, 'colour (0–1)', refs=REFS, needs=TORCH)
def _torch_cpu(ctx):
    return _torch_vs_numpy(ctx, 'cpu', 'float64')


@check('splat.raster_torch_cuda', 'splat', 'Torch rasteriser (CUDA, float32) against the numpy reference',
       'largest colour difference over 300 fixed Gaussians through three lens models', 'the numpy rasteriser written '
       'again without tiles, in float64', 1e-4, 'colour (0–1)', refs=REFS, needs=TORCH + ((CUDA_NEED, 'splat'),))
def _torch_cuda(ctx):
    out = _torch_vs_numpy(ctx, 'cuda', 'float32')
    import torch
    out.detail['gpu'] = torch.cuda.get_device_name(0)
    return out


@check('splat.reproject_disparity', 'splat', 'Reprojection between a rectified stereo pair',
       'largest pixel difference of 500 reprojected points against the closed-form disparity',
       'u_r = u_l − f b / z, v_r = v_l (rectified pinholes)', 1e-9, 'px', refs=('kerbl2023_3dgs',))
def _disparity(ctx):
    from ..core import splat_reproject as RP
    from ..core.optics import CameraModel
    rng = ctx.rng('splat.reproject_disparity')
    f, b = 400.0, 0.06
    cam = CameraModel(640, 480, f, f, 319.5, 239.5)
    Ta = np.eye(4)
    Tb = np.eye(4)
    Tb[0, 3] = b                                       # the right camera, b along the left camera's x (right)
    uv = rng.uniform([0, 0], [639, 479], (500, 2))
    z = rng.uniform(0.3, 5.0, 500)
    uvb, zb, ok = RP.reproject(uv, z, cam, Ta, cam, Tb)
    ref = np.c_[uv[:, 0] - f * b / z, uv[:, 1]]
    sel = ok & (ref[:, 0] >= 0)
    err = float(max(np.abs(uvb[sel] - ref[sel]).max(), np.abs(zb[sel] - z[sel]).max() * f))
    return Outcome(err, {'points': int(sel.sum()), 'baseline_m': b, 'f_px': f})


@check('splat.repose_rigid', 'splat', 'Re-posing a link\'s Gaussians with the joint state',
       'largest colour difference: the rig posed at q against the link-frame Gaussians through the camera moved '
       'into the link frame, 5 joint states', 'the rigid-motion identity render(T · G, C) = render(G, T⁻¹ · C), T from '
       'the robot model\'s link_pose', 1e-5, 'colour (0–1)', refs=('qureshi2025_splatsim', 'lou2025_robogs'))
def _repose(ctx):
    from ..core import splat_reproject as RP
    from ..core.optics import CameraModel
    from ..core.robot.model import get_model
    from ..core.robot.samples import sample_project
    from ..core.sim import splat_rig as SRG
    p = sample_project('so100')
    uid = p.robots[0].uid
    m = get_model(p, uid)
    rng = ctx.rng('splat.repose_rigid')
    links = [(uid, ln) for ln in m.link_names]
    k = len(links) - 1
    ln = links[k][1]
    n = 60
    gs = RP.GaussianSet(rng.uniform(-0.03, 0.03, (n, 3)), rng.uniform(0.002, 0.008, (n, 3)), rng.normal(size=(n, 4)),
                        rng.uniform(0.3, 0.9, n), rng.uniform(0, 1, (n, 3)))
    rig = SRG.SplatRig(gs, np.full(n, k, np.int32), links)
    cam = CameraModel(64, 48, 50, 50, 31.5, 23.5)
    worst, moved = 0.0, []
    for j, q in enumerate(SRG.random_qs(p, 6, int(rng.integers(1 << 30)), spread_rad=0.6)[1:]):
        Tl = m.link_pose(q[uid], ln)
        target = Tl[:3, 3]
        T = SRG.look_at(target + np.array([0.25, 0.12, 0.15]), target)
        a = rig.render(p, cam, T, q, backend='numpy')['rgb']
        b = RP.render(gs, cam, np.linalg.inv(Tl) @ T, 'numpy')['rgb']
        worst = max(worst, float(np.abs(a - b).max()))
        moved.append(float(np.linalg.norm(Tl[:3, 3] - m.link_pose(m.home, ln)[:3, 3])))
    return Outcome(worst, {'link': ln, 'gaussians': n, 'joint_states': 5,
                           'link_moved_m': [round(x, 4) for x in moved]})


@check('splat.ssim_closed_form', 'splat', 'SSIM and PSNR of constant images against their closed forms',
       'largest |SSIM − (2ab + C1)/(a² + b² + C1)| over 40 constant pairs, and |PSNR − (−20 log10 d)|',
       'SSIM\'s luminance term for constant patches (contrast and structure terms are 1); PSNR\'s definition',
       1e-12, 'SSIM / dB', refs=('wang2004_ssim',))
def _ssim_closed(ctx):
    from ..core import splat_raster as SR
    rng = ctx.rng('splat.ssim_closed_form')
    C1 = 0.01 ** 2
    err_s, err_p = 0.0, 0.0
    for _ in range(40):
        a, b = rng.uniform(0, 1, 2)
        A = np.full((24, 32, 3), a)
        B = np.full((24, 32, 3), b)
        err_s = max(err_s, abs(SR.ssim_np(A, B) - (2 * a * b + C1) / (a * a + b * b + C1)))
        d = abs(a - b)
        if d > 1e-4:
            err_p = max(err_p, abs(SR.psnr(A, B) + 20 * math.log10(d)))
    return Outcome(max(err_s, err_p), {'pairs': 40, 'ssim_error': err_s, 'psnr_error_db': err_p})


@check('splat.ssim_vs_skimage', 'splat', 'SSIM against scikit-image',
       'largest |SSIM − scikit-image\'s| on the interior of 20 random image pairs',
       'skimage.metrics.structural_similarity (gaussian_weights, σ 1.5, population covariance, data_range 1)', 1e-9,
       'SSIM', refs=('wang2004_ssim', 'skimage_ssim'), needs=(('skimage', 'test'),))
def _ssim_skimage(ctx):
    from scipy.ndimage import gaussian_filter
    from skimage.metrics import structural_similarity

    from ..core import splat_raster as SR
    rng = ctx.rng('splat.ssim_vs_skimage')
    err = 0.0
    inner = np.zeros((48, 64), bool)
    inner[5:-5, 5:-5] = True
    for k in range(20):
        a = gaussian_filter(rng.random((48, 64, 3)), (1.0 + k % 3, 1.0 + k % 3, 0))
        b = np.clip(a * rng.uniform(0.6, 1.2) + rng.uniform(-0.1, 0.1) + rng.normal(0, 0.05 * (1 + k % 4), a.shape),
                    0, 1)
        _, S = structural_similarity(a, b, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                                     data_range=1.0, channel_axis=-1, full=True)
        ref = float(S.mean(-1)[inner].mean())
        err = max(err, abs(SR.ssim_np(a, b, inner) - ref))
    import skimage
    return Outcome(err, {'pairs': 20, 'size_px': [64, 48], 'interior_margin_px': 5, 'skimage': skimage.__version__})
