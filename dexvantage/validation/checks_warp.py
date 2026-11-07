"""Checks of the canonical view warp (core/exporters/canonical_warp.py, spec §7.16) and its export.

Every case is synthetic: two cameras posed in a shared contact frame, looking at the contact plane z = 0 from 0.10 to
0.20 m with random tilts (seeded), with a pinhole pair and a pair with a strong Brown-Conrady and a Kannala-Brandt
fisheye lens. The references share nothing with the model but the cameras' figures.

vla.warp_plane_exact — points drawn on the plane are projected into both cameras by K (R X + t) written out here;
the model's homography must carry each source pixel onto its canonical pixel. Tolerance 1e-9 px: H is a closed form,
so only double rounding remains (about 1e-13 px measured at image coordinates of hundreds of pixels).

vla.warp_lut_lens_cv2 — the full remap with both lenses: for plane points projected into the canonical camera by
OpenCV (cv2.projectPoints for Brown-Conrady, cv2.fisheye.projectPoints for Kannala-Brandt), the LUT must name the
source pixel OpenCV projects the same point to. Tolerance 1e-6 px: the LUT unprojects through the lens model's
iterative inverse (run to convergence; measured about 1e-13 px), and OpenCV's projection is the forward map.

vla.warp_parallax_bruteforce — the residual parallax. Reference 1: points drawn off the plane, projected by OpenCV
into the canonical camera directly and through the warp (the plane point on the source ray through the point, then
into the canonical camera): Δ = true − warped. The closed plane-plus-parallax form Δ = h / (d Z_A) ((K_A t)_xy −
p (K_A t)_z) [Irani and Anandan 1996] written out here is compared with it on the pinhole pair, and the model's
parallax_at (which walks the source ray from the canonical pixel) on both pairs. Tolerance 1e-6 px for the same
reason as above (measured about 1e-13 px).

vla.warp_lut_vs_warpperspective — the remap maps of an undistorted pair against OpenCV's own homography warp: the
float64 LUT against H⁻¹ applied to every canonical pixel centre (reported), and the image cv2.remap(img, map_x, map_y)
against cv2.warpPerspective(img, H) on a smooth test image (gradient at most 9 grey levels per pixel), over the pixels
both fill away from the source border. The error is the largest grey-level difference; tolerance 1 grey level: both
quantise each sample position to 1/32 px (OpenCV's INTER_TAB_SIZE) from the same coordinate, which the float32 maps
can round to the neighbouring 1/32 step (at most 9/32 of a level here), and both round the result to an integer.

formats.canonical_warp_files — the export written and read back: the .yml through cv2.FileStorage must hold the
.npy maps bit for bit, the grid_sample .npy must equal (2u + 1)/W − 1 of the maps, the JSON's H must equal the
model's and its canonical view must load back as a view with the same camera and pose. Tolerance 0 (bit-exact
arrays; 1e-12 for numbers through JSON's decimal text).
"""
from __future__ import annotations

import json
import os

import numpy as np

from .suite import Outcome, check


def _look(pos, rpy):
    """An optical frame at pos in the contact frame, looking down the contact z axis, tilted by rpy."""
    from ..core.frames import Pose, quat_from_rpy
    M = Pose(np.asarray(pos, float), quat_from_rpy(*rpy)).matrix()
    M[:3, :3] = M[:3, :3] @ np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    return M


def _pairs(rng, lens: bool):
    from ..core.exporters import canonical_warp as CW
    from ..core.optics import CameraModel
    if lens:
        cA = CameraModel(640, 480, 420.0, 418.0, 322.0, 241.0, 'brown', (-0.28, 0.09, 0.0012, -0.0007, -0.012))
        cB = CameraModel(800, 600, 330.0, 331.0, 401.0, 299.0, 'kb', (0.021, -0.008, 0.0021, -0.0003))
    else:
        cA = CameraModel(640, 480, 520.0, 515.0, 321.3, 238.9)
        cB = CameraModel(800, 600, 610.0, 612.0, 401.2, 297.5)
    out = []
    for _ in range(4):
        TA = _look([*rng.uniform(-0.03, 0.03, 2), rng.uniform(0.10, 0.20)], rng.uniform(-0.25, 0.25, 3))
        TB = _look([*rng.uniform(-0.03, 0.03, 2), rng.uniform(0.10, 0.20)], rng.uniform(-0.25, 0.25, 3))
        A, B = CW.View('canonical', cA, TA), CW.View('source', cB, TB)
        out.append((A, B, CW.make_warp(B, A, 0.0)))
    return out


def _to_cam(V, P):
    Ti = np.linalg.inv(V.T_contact_cam)
    return P @ Ti[:3, :3].T + Ti[:3, 3]


def _cv2_project(camera, X):
    import cv2
    K, D, m = camera.to_opencv()
    X = np.ascontiguousarray(X, np.float64).reshape(-1, 1, 3)
    if m == 'kb':
        return cv2.fisheye.projectPoints(X, np.zeros((3, 1)), np.zeros((3, 1)), K, D)[0].reshape(-1, 2)
    return cv2.projectPoints(X, np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)


def _pinhole(K, X):
    x = X @ np.asarray(K, float).T
    return x[:, :2] / x[:, 2:]


def _in_image(cam, uv, X):
    return (X[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] <= cam.width - 1) & (uv[:, 1] >= 0) & \
        (uv[:, 1] <= cam.height - 1)


@check('vla.warp_plane_exact', 'vla', 'Canonical warp homography exact on the contact plane',
       'largest distance between H applied to a plane point\'s source pixel and its canonical pixel, over four random '
       'pinhole camera pairs and 400 plane points each', 'both pixels projected by K (R X + t) written out in the '
       'check from the cameras\' poses in the contact frame', 1e-9, 'px', refs=('hartley2004',))
def _plane_exact(ctx):
    rng = ctx.rng('vla.warp_plane_exact')
    worst, used = 0.0, 0
    for A, B, w in _pairs(rng, False):
        P = np.c_[rng.uniform(-0.05, 0.05, (400, 2)), np.zeros(400)]
        XA, XB = _to_cam(A, P), _to_cam(B, P)
        xa, xb = _pinhole(A.K, XA), _pinhole(B.K, XB)
        keep = _in_image(A.camera, xa, XA) & _in_image(B.camera, xb, XB)
        h = np.c_[xb[keep], np.ones(int(keep.sum()))] @ w.H.T
        worst = max(worst, float(np.abs(h[:, :2] / h[:, 2:] - xa[keep]).max(initial=0.0)))
        used += int(keep.sum())
    return Outcome(worst, {'pairs': 4, 'points_compared': used}, ok=used > 400)


@check('vla.warp_lut_lens_cv2', 'vla', 'Canonical warp remap through two distorted lenses against OpenCV',
       'largest distance between the source pixel the LUT gives for a plane point\'s canonical pixel and the source '
       'pixel OpenCV projects the point to (a Brown-Conrady canonical camera, a Kannala-Brandt source), four random '
       'pairs, 400 plane points each', 'cv2.projectPoints and cv2.fisheye.projectPoints of the plane points in each '
       'camera', 1e-6, 'px', refs=('opencv_calib3d', 'opencv_fisheye', 'kannala_brandt2006', 'brown1966'))
def _lut_lens(ctx):
    from ..core.exporters import canonical_warp as CW
    rng = ctx.rng('vla.warp_lut_lens_cv2')
    worst, used = 0.0, 0
    for A, B, w in _pairs(rng, True):
        P = np.c_[rng.uniform(-0.05, 0.05, (400, 2)), np.zeros(400)]
        XA, XB = _to_cam(A, P), _to_cam(B, P)
        pa, pb = _cv2_project(A.camera, XA), _cv2_project(B.camera, XB)
        keep = _in_image(A.camera, pa, XA) & _in_image(B.camera, pb, XB)
        uv, ok = CW.lut_at(w, pa[keep])
        if not ok.all():
            return Outcome(float('nan'), {'lut_missed': int((~ok).sum())}, ok=False)
        worst = max(worst, float(np.abs(uv - pb[keep]).max(initial=0.0)))
        used += int(keep.sum())
    return Outcome(worst, {'pairs': 4, 'points_compared': used}, ok=used > 400)


@check('vla.warp_parallax_bruteforce', 'vla', 'Residual parallax off the plane against brute-force projection',
       'largest difference of the parallax Δ (px) between the brute-force projection of off-plane points and (a) the '
       'closed plane-plus-parallax form on pinhole pairs, (b) the model\'s parallax_at on pinhole and distorted '
       'pairs; 300 points per pair at −0.2 to +0.03 m from the plane', 'each point projected by OpenCV into the '
       'canonical camera directly and through the plane point on its source ray; the closed form written out here',
       1e-6, 'px', refs=('irani_anandan1996', 'hartley2004', 'opencv_calib3d', 'opencv_fisheye'))
def _parallax(ctx):
    from ..core.exporters import canonical_warp as CW
    rng = ctx.rng('vla.warp_parallax_bruteforce')
    worst_cf, worst_model, used, biggest = 0.0, 0.0, 0, 0.0
    for lens in (False, True):
        for A, B, w in _pairs(rng, lens):
            P = np.c_[rng.uniform(-0.06, 0.06, (300, 2)), rng.uniform(-0.2, 0.03, 300)]
            XB = _to_cam(B, P)
            XA = XB @ w.R.T + w.t
            h = XB @ w.n_B + w.d_B
            X0B = XB * (-w.d_B / (XB @ w.n_B))[:, None]            # the plane point on the source ray
            X0A = X0B @ w.R.T + w.t
            p = _cv2_project(A.camera, X0A)
            true = _cv2_project(A.camera, XA)
            keep = (XA[:, 2] > 0.01) & (X0A[:, 2] > 0.01) & (XB[:, 2] > 0.01) & _in_image(A.camera, p, X0A)
            brute = (true - p)[keep]
            biggest = max(biggest, float(np.linalg.norm(brute, axis=1).max(initial=0.0)))
            if not lens:
                e = A.K @ w.t
                ZA = XA[keep, 2]
                cf = (h[keep] / (w.d_B * ZA))[:, None] * (e[:2][None, :] - p[keep] * e[2])
                worst_cf = max(worst_cf, float(np.abs(cf - brute).max(initial=0.0)))
            pk, hk = p[keep], h[keep]
            mine = np.array([CW.parallax_at(w, pk[i:i + 1], float(hk[i]))[0][0] for i in range(len(pk))])
            if np.isnan(mine).any():
                return Outcome(float('nan'), {'undefined': int(np.isnan(mine).any(axis=1).sum())}, ok=False)
            worst_model = max(worst_model, float(np.abs(mine - brute).max(initial=0.0)))
            used += int(keep.sum())
    return Outcome(max(worst_cf, worst_model), {'closed_form_vs_brute_px': worst_cf,
                                                'model_vs_brute_px': worst_model, 'points': used,
                                                'largest_parallax_px': biggest}, ok=used > 1000 and biggest > 1.0)


@check('vla.warp_lut_vs_warpperspective', 'vla', 'Canonical warp remap maps against cv2.warpPerspective',
       'largest grey-level difference between cv2.remap through the exported maps and cv2.warpPerspective with H, on '
       'an undistorted pair and a smooth test image, over the pixels both fill (the LUT against H⁻¹ is reported)',
       'OpenCV\'s homography warp (dst(x) = src(H⁻¹ x)) and H⁻¹ applied to every pixel centre', 1.0, 'grey level',
       refs=('opencv_remap', 'opencv_warp_perspective'))
def _vs_warp_perspective(ctx):
    import cv2

    from ..core.exporters import canonical_warp as CW
    rng = ctx.rng('vla.warp_lut_vs_warpperspective')
    A, B, w = _pairs(rng, False)[0]
    cA, cB = A.camera, B.camera
    G = CW.pixel_grid(cA)
    uv, ok = CW.lut_at(w, G)
    q = np.c_[G.reshape(-1, 2), np.ones(G.shape[0] * G.shape[1])] @ np.linalg.inv(w.H).T
    q = (q[:, :2] / q[:, 2:]).reshape(uv.shape)
    coord = float(np.abs(uv - q)[ok].max(initial=0.0))
    mx, my, valid = CW.remap_maps(w)
    yy, xx = np.mgrid[0:cB.height, 0:cB.width]
    img = (127.5 + 60 * np.sin(xx / 9.0) * np.cos(yy / 11.0) + 40 * np.sin((xx + yy) / 23.0)).astype(np.uint8)
    r1 = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    r2 = cv2.warpPerspective(img, w.H, (cA.width, cA.height), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    inner = valid & (mx > 1) & (my > 1) & (mx < cB.width - 2) & (my < cB.height - 2)
    d = np.abs(r1.astype(int) - r2.astype(int))[inner]
    return Outcome(float(d.max(initial=0)), {'lut_vs_H_inverse_px': coord, 'pixels_compared': int(inner.sum()),
                                             'mean_grey_difference': float(d.mean()) if d.size else None,
                                             'pixels_differing': int((d > 0).sum())},
                   ok=coord < 1e-9 and inner.sum() > 1000)


@check('formats.canonical_warp_files', 'formats', 'Canonical warp export read back',
       'largest difference between the arrays and numbers written and those read back: the .yml maps through '
       'cv2.FileStorage against the .npy maps, the grid_sample grid against (2u + 1)/W − 1 of the maps, the JSON '
       'homography against the model\'s, the canonical view file loaded back', 'OpenCV\'s FileStorage reader, '
       'numpy.load, json and the grid formula written out here', 1e-12, 'px', refs=('opencv_filestorage',),
       cost='study')
def _export_files(ctx):
    import cv2

    from ..core import exporters as X
    from ..core.exporters import canonical_warp as CW
    from ..core.robot.samples import sample_project
    p = sample_project('ur5e')
    out = os.path.join(ctx.tmp(), 'canonical_warp_export')
    man = X.export(p, out, ['canonical_warp'])
    row = man['formats']['canonical_warp']
    if row['status'] != 'ok':
        return Outcome(float('nan'), {'refused': row.get('reason')}, ok=False)
    f = {os.path.basename(r).rsplit('_', 1)[-1]: os.path.join(out, r) for r in row['files']}
    m1, m2 = np.load(f['map1.npy']), np.load(f['map2.npy'])
    fs = cv2.FileStorage(f['remap.yml'], cv2.FILE_STORAGE_READ)
    y1, y2 = fs.getNode('map1').mat(), fs.getNode('map2').mat()
    Hy = fs.getNode('homography').mat()
    fs.release()
    e_yml = float(max(np.abs(y1 - m1).max(), np.abs(y2 - m2).max()))
    hdoc = json.load(open(f['homography.json'], encoding='utf-8'))
    W, H = hdoc['source']['camera']['width'], hdoc['source']['camera']['height']
    grid, valid = np.load(f['grid.npy']), np.load(f['valid.npy']).astype(bool)
    gx = ((2.0 * m1.astype(np.float64) + 1.0) / W - 1.0).astype(np.float32)
    gy = ((2.0 * m2.astype(np.float64) + 1.0) / H - 1.0).astype(np.float32)
    e_grid = float(max(np.abs(grid[..., 0] - gx)[valid].max(initial=0), np.abs(grid[..., 1] - gy)[valid].max(initial=0),
                       np.abs(grid[~valid] - CW.GRID_INVALID).max(initial=0)))
    w, _st = CW.build(p, None)
    e_h = float(max(np.abs(np.asarray(hdoc['homography']) - w.H).max(), np.abs(Hy - w.H).max()))
    v = CW.view_from_dict(json.load(open(f['view.json'], encoding='utf-8')))
    e_view = float(max(np.abs(v.T_contact_cam - w.canonical.T_contact_cam).max(),
                       abs(v.camera.fx - w.canonical.camera.fx), abs(v.camera.cx - w.canonical.camera.cx)))
    sha_ok = all(r in man['files'] for r in row['files'])
    return Outcome(max(e_yml, e_grid, e_h, e_view), {'yml_vs_npy': e_yml, 'grid_formula': e_grid,
                                                      'homography_json_and_yml': e_h, 'canonical_view_file': e_view,
                                                      'files': [os.path.basename(r) for r in row['files']],
                                                      'manifest_sha256_for_every_file': sha_ok},
                   ok=sha_ok and bool(valid.any()))
