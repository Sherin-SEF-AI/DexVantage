"""Checks of core/optics (the camera models): projection against OpenCV's own implementations (cv2.projectPoints and
cv2.fisheye.projectPoints), unprojection against projection, the fields of view of models fitted to a stated FOV
against the stated angles, the closed-form pixel density against finite differences of the projection, and the
thin-lens depth of field and blur disc against the hyperfocal formulae and the thin-lens equation written out again
here."""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from .suite import Outcome, check, figure, series

_SIZES = ((1280, 720), (1920, 1080), (640, 480), (848, 480))


def _random_K(rng, w, h, hfov_lo=50.0, hfov_hi=110.0):
    f = (w / 2) / math.tan(math.radians(rng.uniform(hfov_lo, hfov_hi)) / 2)
    return f * rng.uniform(0.99, 1.01), f * rng.uniform(0.99, 1.01), (w - 1) / 2 + rng.uniform(-20, 20), \
        (h - 1) / 2 + rng.uniform(-20, 20)


def _brown_dists(rng, n):
    """Brown-Conrady terms of the size real calibrations give (moderate barrel, small tangential), five and eight
    terms."""
    out = []
    for i in range(n):
        d = [rng.uniform(-0.3, 0.05), rng.uniform(-0.05, 0.12), rng.uniform(-2e-3, 2e-3), rng.uniform(-2e-3, 2e-3),
             rng.uniform(-0.02, 0.02)]
        if i % 2:
            d += [rng.uniform(-0.05, 0.05), rng.uniform(-0.01, 0.01), rng.uniform(-1e-3, 1e-3)]
        out.append(tuple(d))
    return out


def _cone_points(rng, n, theta_max, rmin=0.05, rmax=3.0):
    """n points at uniformly random directions within theta_max of the optical axis (z), random ranges."""
    ct = rng.uniform(math.cos(theta_max), 1.0, n)
    th = np.arccos(ct)
    ph = rng.uniform(0, 2 * math.pi, n)
    d = np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]
    return d * rng.uniform(rmin, rmax, (n, 1))


@check('optics.opencv_pinhole_brown', 'optics', 'Pinhole and Brown-Conrady projection',
       'largest pixel difference between CameraModel.project and cv2.projectPoints for the same K and D (no '
       'distortion, five-term and eight-term rational Brown-Conrady), random points in front of the camera',
       'OpenCV cv2.projectPoints (calib3d), an independent implementation of the same model', 1e-6, 'px',
       refs=('opencv_calib3d', 'brown1966'))
def _opencv_brown(ctx):
    import cv2
    from ..core.optics import CameraModel
    rng = ctx.rng('optics.opencv_pinhole_brown')
    n_cams, n_pts = (4, 300) if ctx.quick else (16, 2000)
    worst, counted, per = 0.0, 0, []
    dists = [()] * (n_cams // 4) + _brown_dists(rng, n_cams - n_cams // 4)
    for i, d in enumerate(dists):
        w, h = _SIZES[i % len(_SIZES)]
        fx, fy, cx, cy = _random_K(rng, w, h)
        cam = CameraModel(w, h, fx, fy, cx, cy, 'brown' if d else 'pinhole', d)
        P = _cone_points(rng, n_pts, min(cam.max_theta, math.radians(60)) * 0.95)
        uv, ok = cam.project(P)
        K, D, _ = cam.to_opencv()
        ref = cv2.projectPoints(P.reshape(-1, 1, 3), np.zeros((3, 1)), np.zeros((3, 1)), K, D)[0].reshape(-1, 2)
        fin = np.isfinite(uv).all(axis=1)
        if fin.mean() < 0.999:
            return Outcome(float('inf'), {'camera': i, 'finite': float(fin.mean())}, ok=False,
                           note='CameraModel refused points inside its valid angle.')
        e = float(np.abs(uv - ref).max())
        per.append(e)
        worst = max(worst, e)
        counted += len(P)
    return Outcome(worst, {'cameras': len(dists), 'points': counted, 'median_camera_error_px': float(np.median(per)),
                           'distortion': 'none, 5-term and 8-term rational'},
                   figure=figure('Largest error per camera', 'camera', 'px',
                                 [series('|uv − OpenCV|', np.arange(len(per)), np.maximum(per, 1e-18), 'points')],
                                 logy=True, hline=1e-6, hline_label='tolerance'),
                   note='Both evaluate the same polynomial in double precision; they differ only in the order of '
                        'operations (about 1e-13 px at 1000 px). 1e-6 px is far below any convention error: a '
                        'half-pixel shift, swapped p1/p2 or a mis-ordered k3 shows as 1e-3 px or more.')


@check('optics.opencv_fisheye', 'optics', 'Kannala-Brandt (fisheye) projection',
       'largest pixel difference between CameraModel.project (model kb) and cv2.fisheye.projectPoints for the same '
       'K and D, random points up to 85° off the axis',
       'OpenCV cv2.fisheye.projectPoints, an independent implementation of the same model', 1e-6, 'px',
       refs=('opencv_fisheye', 'kannala_brandt2006'))
def _opencv_fisheye(ctx):
    import cv2
    from ..core.optics import CameraModel
    rng = ctx.rng('optics.opencv_fisheye')
    n_cams, n_pts = (4, 300) if ctx.quick else (16, 2000)
    worst, per = 0.0, []
    for i in range(n_cams):
        w, h = _SIZES[i % len(_SIZES)]
        f = (w / 2) / math.radians(rng.uniform(150, 200) / 2)
        d = (rng.uniform(-0.05, 0.05), rng.uniform(-0.02, 0.02), rng.uniform(-5e-3, 5e-3), rng.uniform(-1e-3, 1e-3))
        if i == 0:
            d = (0.0, 0.0, 0.0, 0.0)
        cam = CameraModel(w, h, f, f * rng.uniform(0.995, 1.005), (w - 1) / 2 + rng.uniform(-10, 10),
                          (h - 1) / 2 + rng.uniform(-10, 10), 'kb', d)
        # OpenCV divides by Z, so the comparison stays in front of the camera (CameraModel itself goes past 90°)
        P = _cone_points(rng, n_pts, min(cam.max_theta, math.radians(85)))
        P[0] = (0.0, 0.0, 1.0)                    # on the axis
        uv, ok = cam.project(P)
        K, D, model = cam.to_opencv()
        assert model == 'kb'
        ref = cv2.fisheye.projectPoints(P.reshape(-1, 1, 3), np.zeros((3, 1)), np.zeros((3, 1)), K, D)[0]
        e = float(np.abs(uv - ref.reshape(-1, 2)).max())
        per.append(e)
        worst = max(worst, e)
    return Outcome(worst, {'cameras': n_cams, 'points_per_camera': n_pts, 'median_camera_error_px':
                           float(np.median(per))},
                   note='The same polynomial in θ evaluated twice in double precision; 1e-6 px leaves six orders of '
                        'magnitude above rounding and catches any convention error (θ against atan(r), the order of '
                        'the terms).')


def _models_for_round_trip(rng):
    from ..core.optics import CameraModel
    cams = []
    for i, (w, h) in enumerate(_SIZES):
        fx, fy, cx, cy = _random_K(rng, w, h, 60, 100)
        cams.append(CameraModel(w, h, fx, fy, cx, cy, 'pinhole'))
        cams.append(CameraModel(w, h, fx, fy, cx, cy, 'brown', _brown_dists(rng, 2)[i % 2]))
        f = (w / 2) / math.radians(rng.uniform(140, 190) / 2)
        cams.append(CameraModel(w, h, f, f, cx, cy, 'kb', (rng.uniform(-0.04, 0.04), rng.uniform(-0.01, 0.01),
                                                         rng.uniform(-2e-3, 2e-3), rng.uniform(-3e-4, 3e-4))))
        for m in ('equidistant', 'equisolid', 'stereographic'):
            cams.append(CameraModel(w, h, f, f, cx, cy, m))
    return cams


@check('optics.round_trip', 'optics', 'Unprojection inverts projection',
       'largest pixel error of project(unproject(pixel)) over random pixels of the image, for every model '
       '(pinhole, Brown-Conrady, Kannala-Brandt, equidistant, equisolid, stereographic); the ray angle error of '
       'unproject(project(ray)) for random rays landing on the image must also be below 1e-9 rad',
       'the pixels themselves (and the rays themselves): an inverse is exact by definition', 1e-6, 'px',
       refs=('hartley2004', 'kannala_brandt2006'))
def _round_trip(ctx):
    rng = ctx.rng('optics.round_trip')
    n = 500 if ctx.quick else 5000
    worst_px, worst_rad, lost, per = 0.0, 0.0, 0, {}
    for cam in _models_for_round_trip(rng):
        uv = np.c_[rng.uniform(-0.5, cam.width - 0.5, n), rng.uniform(-0.5, cam.height - 0.5, n)]
        rays, ok = cam.unproject(uv, with_ok=True)
        lost += int((~ok).sum())
        uv2, ok2 = cam.project(rays[ok])
        e = float(np.abs(uv2 - uv[ok]).max()) if ok.any() else 0.0
        P = _cone_points(rng, n, cam.max_theta * 0.999)
        uvp, okp = cam.project(P)
        r2 = cam.unproject(uvp[okp])             # rays that land on the image: the pixels unproject serves
        d = P[okp] / np.linalg.norm(P[okp], axis=1, keepdims=True)
        ang = np.arctan2(np.linalg.norm(np.cross(r2, d), axis=1), np.einsum('ij,ij->i', r2, d))
        a = float(np.nanmax(ang)) if len(ang) else 0.0
        per[cam.model] = max(per.get(cam.model, 0.0), e)
        worst_px, worst_rad = max(worst_px, e), max(worst_rad, a)
    return Outcome(worst_px, {'worst_ray_angle_rad': worst_rad, 'per_model_px': per,
                              'pixels_without_a_ray': lost},
                   ok=worst_rad < 1e-9 and math.isfinite(worst_rad),
                   note='The Brown-Conrady and Kannala-Brandt inverses are iterative (Newton, converged to 1e-12 in '
                        'normalised units, about 1e-9 px); 1e-6 px allows for the image scale. Pixels with no ray '
                        '(a Brown-Conrady model folding back before the corner) are counted, not compared.')


@check('optics.from_fov', 'optics', 'A model fitted to a stated FOV reproduces it',
       'largest difference between the field of view a CameraModel.from_fov model shows (fov_deg: edge to edge from '
       'its own unprojection) and the stated horizontal, vertical and diagonal angles, over every model and a range '
       'of image sizes and FOVs',
       'the stated angles (for Kannala-Brandt, consistent H/V/D triples generated from pinhole, equisolid, '
       'equidistant and stereographic lenses written out in the check)', 1e-6, 'deg',
       refs=('kannala_brandt2006', 'miyamoto1964'))
def _from_fov(ctx):
    from ..core.optics import fit_fov
    rng = ctx.rng('optics.from_fov')
    worst, rows, closed_worst = 0.0, 0, 0.0
    n = 6 if ctx.quick else 30
    for m in ('pinhole', 'brown', 'equidistant', 'equisolid', 'stereographic', 'kb'):
        for _ in range(n):
            w, h = _SIZES[int(rng.integers(len(_SIZES)))]
            hi = 150.0 if m in ('pinhole', 'brown') else 220.0
            hf = rng.uniform(30.0, hi)
            # 'kb' with a vertical angle fits square pixels (below, with consistent figures); here only H
            if rng.uniform() < 0.5 or m == 'kb':
                cam, _ = fit_fov(w, h, hf, None, m)
                ref = {'h': hf}
                if m in ('pinhole', 'brown'):        # square pixels: vertical from tan(v/2) = (H/W) tan(h/2)
                    ref['v'] = 2 * math.degrees(math.atan(h / w * math.tan(math.radians(hf / 2))))
            else:
                vf = hf * rng.uniform(0.4, 0.9)
                cam, _ = fit_fov(w, h, hf, vf, m)
                ref = {'h': hf, 'v': vf}
            got = cam.fov_deg()
            worst = max(worst, max(abs(got[k] - v) for k, v in ref.items()))
            if m in ('pinhole', 'brown') and not any(cam.dist or ()):
                # independent of fov_deg (which reads the angles back through the model's own unprojection): the
                # pinhole closed form from the fitted intrinsics, image edges at −0.5 and W − 0.5 px
                cf = {'h': math.degrees(math.atan((cam.cx + 0.5) / cam.fx) + math.atan((w - 0.5 - cam.cx) / cam.fx)),
                      'v': math.degrees(math.atan((cam.cy + 0.5) / cam.fy) + math.atan((h - 0.5 - cam.cy) / cam.fy))}
                closed_worst = max(closed_worst, max(abs(cf[k] - v) for k, v in ref.items()))
            rows += 1
    worst = max(worst, closed_worst)
    # Kannala-Brandt through all three stated angles, generated from a classical lens (consistent figures)
    g = {'equidistant': (lambda t: t, lambda r: r),
         'equisolid': (lambda t: 2 * math.sin(t / 2), lambda r: 2 * math.asin(r / 2)),
         'stereographic': (lambda t: 2 * math.tan(t / 2), lambda r: 2 * math.atan(r / 2))}
    g['pinhole'] = (math.tan, math.atan)
    kb_worst = 0.0
    for i in range(n):
        w, h = _SIZES[int(rng.integers(len(_SIZES)))]
        base = ('equidistant', 'equisolid', 'stereographic', 'pinhole')[i % 4]
        fwd, inv = g[base]
        hf = rng.uniform(60.0, 140.0) if base == 'pinhole' else rng.uniform(100.0, 200.0)
        f = (w / 2) / fwd(math.radians(hf / 2))
        vf = 2 * math.degrees(inv((h / 2) / f))
        df = 2 * math.degrees(inv(math.hypot(w, h) / 2 / f))
        with_d = i % 3 != 2                      # H, V and D; or H and V only
        cam, rep = fit_fov(w, h, hf, vf, 'kb', dfov_deg=df if with_d else None)
        got = cam.fov_deg()
        e = max(abs(got['h'] - hf), abs(got['v'] - vf), abs(got['d'] - df) if with_d else 0.0)
        kb_worst = max(kb_worst, e)
        rows += 1
    worst = max(worst, kb_worst)
    return Outcome(worst, {'models_fitted': rows, 'kb_hvd_worst_deg': kb_worst,
                           'pinhole_closed_form_worst_deg': closed_worst},
                   note='The closed forms invert r(θ) exactly and the Kannala-Brandt fit passes through the stated '
                        'points as constraints, so only rounding and the iterative inverse remain (about 1e-12°); '
                        '1e-6° is far below a datasheet\'s 0.1° rounding and catches an edge off by half a pixel '
                        '(about 0.05° at 1000 px).')


@check('optics.pixel_density', 'optics', 'Pixel density on a patch facing the camera',
       'largest relative difference between CameraModel.pixels_per_metre (closed form) and √|det J| of the 2 × 2 '
       'Jacobian of the projection of a small patch facing the camera, by central finite differences of project(); '
       'every model, points across the field, and tilted surfaces (normals)',
       'finite differences of the projection itself (an independent derivative)', 1e-6, '(relative)',
       refs=('hartley2004', 'kannala_brandt2006'))
def _pixel_density(ctx):
    rng = ctx.rng('optics.pixel_density')
    n = 100 if ctx.quick else 600
    worst, per = 0.0, {}
    for cam in _models_for_round_trip(rng):
        P = _cone_points(rng, n, min(cam.max_theta * 0.95, math.radians(80)), 0.1, 2.0)
        normals = rng.normal(size=(n, 3))
        d = P / np.linalg.norm(P, axis=1, keepdims=True)
        normals[: n // 2] = -d[: n // 2]          # half facing the camera, half tilted at random
        a = np.cross(d, [0.0, 1.0, 0.0])
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b = np.cross(normals, a)
        bad = np.linalg.norm(b, axis=1) < 1e-6
        b[bad] = np.cross(normals[bad], [1.0, 0.0, 0.0])
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        e1 = np.cross(b, normals)
        e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(normals / np.linalg.norm(normals, axis=1, keepdims=True), e1)
        R = np.linalg.norm(P, axis=1, keepdims=True)
        step = 1e-5 * R
        J = np.empty((n, 2, 2))
        for j, e in enumerate((e1, e2)):
            up, _ = cam.project(P + step * e)
            um, _ = cam.project(P - step * e)
            J[:, :, j] = (up - um) / (2 * step)
        fd = np.sqrt(np.abs(np.linalg.det(J)))
        ours = cam.pixels_per_metre(P, normals)
        rel = np.abs(ours - fd) / fd
        e = float(np.nanmax(rel))
        if not np.isfinite(rel).all():
            return Outcome(float('inf'), {'model': cam.model}, ok=False, note='a density was not finite')
        per[cam.model] = max(per.get(cam.model, 0.0), e)
        worst = max(worst, e)
    return Outcome(worst, {'per_model': per, 'points_per_camera': n},
                   note='Central differences with a step of 1e-5 of the range: truncation about 1e-10 relative, '
                        'rounding about 1e-16 × 1000 px / 1e-5 ≈ 1e-8; 1e-6 leaves margin for both while any error '
                        'in the closed form (a missing cos θ, the sin θ of the tangential direction) is 1e-2 or more '
                        'away from the axis.')


@check('optics.depth_of_field', 'optics', 'Thin-lens depth of field',
       'largest relative difference between depth_of_field (solved from the blur-disc equation) and the near and '
       'far limits from the hyperfocal distance H = f²/(N c) + f: near = s (H − f)/(H + s − 2f), far = s (H − f)/'
       '(H − s) (infinite from s ≥ H)',
       'the hyperfocal-distance formulae of photographic optics, written out again in the check', 1e-10, '(relative)',
       refs=('ray2002',))
def _dof(ctx):
    from ..core.optics import depth_of_field
    rng = ctx.rng('optics.depth_of_field')
    n = 200 if ctx.quick else 3000
    worst, infinite_far, rows = 0.0, 0, []
    for _ in range(n):
        f = rng.uniform(1.0, 50.0)                    # mm
        N = rng.uniform(1.0, 16.0)
        c = rng.uniform(0.002, 0.03)                  # mm
        Hh = f * f / (N * c) + f
        s = rng.uniform(1.5 * f, 3 * Hh)              # mm, both sides of the hyperfocal distance
        near_ref = s * (Hh - f) / (Hh + s - 2 * f)
        far_ref = s * (Hh - f) / (Hh - s) if s < Hh else math.inf
        near, far = depth_of_field(f, N, s / 1000.0, c)
        e = abs(near * 1000 - near_ref) / near_ref
        if math.isinf(far_ref) or math.isinf(far):
            if math.isinf(far_ref) != math.isinf(far):
                # at s within rounding of H either answer is right; elsewhere it is a failure
                e = max(e, 0.0 if abs(s - Hh) <= 1e-9 * Hh else math.inf)
            infinite_far += 1
        else:
            e = max(e, abs(far * 1000 - far_ref) / far_ref)
        worst = max(worst, e)
        rows.append(e)
    # focused at infinity: near = H − f [the hyperfocal formula's limit as s → ∞], far = ∞
    f, N, c = 4.0, 2.0, 0.006
    near_inf, far_inf = depth_of_field(f, N, math.inf, c)
    e_inf = abs(near_inf * 1000 - f * f / (N * c)) / (f * f / (N * c))
    worst = max(worst, e_inf, 0.0 if math.isinf(far_inf) else math.inf)
    return Outcome(worst, {'cases': n, 'far_limit_infinite': infinite_far, 'focus_at_infinity_rel_error': e_inf},
                   note='Two algebraic forms of the same thin-lens result; only rounding separates them '
                        '(about 1e-15 relative). 1e-10 catches any slip in either.')


@check('optics.blur_circle', 'optics', 'Thin-lens defocus blur disc',
       'largest relative difference between blur_circle_px and the blur disc from the thin-lens equation 1/f = 1/z + '
       '1/v written out here (image distances v_s, v_z; disc A |v_z − v_s| / v_z on the sensor at v_s, A = f/N); and '
       'the blur at the depth-of-field limits must equal the circle of confusion',
       'the Gaussian thin-lens equation and similar triangles, written out again in the check and evaluated in exact '
       'rational arithmetic', 1e-11, '(relative)', refs=('ray2002', 'hecht2017'))
def _blur(ctx):
    from ..core.optics import blur_circle_px, depth_of_field
    rng = ctx.rng('optics.blur_circle')
    n = 200 if ctx.quick else 3000
    worst, limit_worst = 0.0, 0.0
    for _ in range(n):
        f = rng.uniform(1.0, 50.0)
        N = rng.uniform(1.0, 16.0)
        p_um = rng.uniform(1.0, 6.0)
        s = rng.uniform(2 * f, 5000.0)
        z = rng.uniform(1.2 * f, 10000.0)
        if abs(z - s) < 1e-3 * s:                 # |s − z| amplifies the rounding of s and z themselves (below)
            z = s * (1 + math.copysign(1e-3, z - s))
        s_m, z_m = s / 1000.0, z / 1000.0         # what the function is given
        # the reference in exact rational arithmetic on the same inputs: no rounding in the reference at all
        F, Nq, Sq, Zq = Fraction(f), Fraction(N), Fraction(s_m) * 1000, Fraction(z_m) * 1000
        vs = 1 / (1 / F - 1 / Sq)
        vz = 1 / (1 / F - 1 / Zq)
        ref = float((F / Nq) * abs(vz - vs) / vz / (Fraction(p_um) / 1000))
        got = blur_circle_px(f, N, s_m, z_m, p_um)
        if ref > 0:
            worst = max(worst, abs(got - ref) / ref)
        c = rng.uniform(0.002, 0.03)
        near, far = depth_of_field(f, N, s / 1000.0, c)
        for zl in (near, far):
            if math.isfinite(zl):
                b = blur_circle_px(f, N, s / 1000.0, zl, p_um) * p_um / 1000.0
                limit_worst = max(limit_worst, abs(b - c) / c)
    return Outcome(max(worst, limit_worst), {'cases': n, 'blur_rel_error': worst,
                                             'blur_at_dof_limits_rel_error': limit_worst},
                   note='The reference is computed in exact rational arithmetic (fractions.Fraction) from the same '
                        'floating-point inputs, so the error is DexVantage\'s own rounding: a few 1e-16, amplified by '
                        's / |s − z| because the focus distance passes through a metre-to-millimetre conversion. '
                        'Points within 0.1 % of the focus distance are moved to 0.1 % from it, bounding that to about '
                        '1e-13; 1e-11 leaves margin.')
