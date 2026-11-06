"""Checks of the depth-validity model (core/studies/depth_validity.py) against independent references.

- depth.sigma_z_monte_carlo: the stereo error model σ_z = z² σ_d / (f B) against a Monte Carlo simulation: a
  rectified stereo pair (projection matrices written out in the check), Gaussian noise of σ_d px on the matched
  right-image coordinate, linear (DLT) triangulation [Hartley and Zisserman 2004, §12.2] and the spread of the
  triangulated depth. Shares no code with the model.
- depth.d400_tables: the library's Intel RealSense D4xx entries, as the study reads them (min-Z at each depth
  resolution through depth_validity.setup, baseline, depth FOV, projector FOV), against the same figures typed again
  in this check from Intel's datasheets (D400 series datasheet Tables 3-49 and 4-12; D457 datasheet Tables 3-19
  and 4-4; URLs below, read 2026-11-06). This is an independent transcription by a second reading of the tables,
  not a second source: an error in the datasheet itself would pass.
- depth.fov_overlap_formula: condition 1 (inside both stereo imagers' fields of view) against Intel's closed form for
  the depth field of view at distance Z, Depth FOV = HFOV/2 + atan(tan(HFOV/2) − B/Z) (D400 datasheet §4.4), with
  HFOV the left imager's horizontal field of view as the model has it.
- depth.finger_shadow_raycast: condition 3 (seen by both imagers: cube maps from each imager's centre) against exact
  ray casting (core/raycast.visible_exact, Möller–Trumbore) from both imager centres, on the posed UR5e sample
  (random arm poses, the gripper partly closed), with points in the wrist D405's field of view where its fingers
  cast their shadows.
"""
from __future__ import annotations

import math

import numpy as np

from .suite import Outcome, check, figure, series

D400_URL = 'https://www.realsenseai.com/wp-content/uploads/2026/03/RealSense-D400-Series-Datasheet-Mar-2026.pdf'
D457_URL = 'https://realsenseai.com/wp-content/uploads/2025/08/RealSense-D457-Datasheet-Aug_2025.pdf'

# Typed again from the datasheets for this check (read 2026-11-06), NOT copied from the library files.
# D400 series datasheet (337029-017, March 2026), Table 4-12 "Minimum-Z Depth", columns D410/D415, D430/D43x,
# D450/D455/D455f/D456 and D401/D405, in mm; "-" (no figure) is None. D457 datasheet (337029-011, August 2025),
# Table 4-4 "Minimum-Z Depth", D450 column.
_RES = ((1280, 720), (848, 480), (640, 480), (640, 360), (480, 270), (424, 240))
_MINZ_D415 = (450, 310, 310, 240, 180, 160)
_MINZ_D43X = (280, 195, 175, 150, 120, 105)
_MINZ_D45X = (520, 350, 320, 260, 200, 180)
_MINZ_D405 = (100, 70, None, 55, 45, 40)
_MINZ_D457 = (520, 350, 320, 260, 200, 180)
# Table 3-49 "Depth Camera SKU Properties" (D457: its Table 3-19): baseline (mm), Depth FOV HD (16:9) H/V/D,
# Depth FOV VGA (4:3) H/V/D (None: "N/A"), IR projector FOV H/V/D (None: no projector).
DATASHEET = {
    'intel-realsense-d415': dict(url=D400_URL, minz=_MINZ_D415, baseline_mm=55, hd=(65, 40, 72), vga=(50, 40, 61),
                                 proj=(67, 41, 75)),
    'intel-realsense-d435': dict(url=D400_URL, minz=_MINZ_D43X, baseline_mm=50, hd=(87, 58, 95), vga=(75, 62, 89),
                                 proj=(90, 63, 99)),
    'intel-realsense-d435i': dict(url=D400_URL, minz=_MINZ_D43X, baseline_mm=50, hd=(87, 58, 95), vga=(75, 62, 89),
                                  proj=(90, 63, 99)),
    'intel-realsense-d455': dict(url=D400_URL, minz=_MINZ_D45X, baseline_mm=95, hd=(87, 58, 95), vga=(75, 62, 89),
                                 proj=(90, 63, 99)),
    'intel-realsense-d405': dict(url=D400_URL, minz=_MINZ_D405, baseline_mm=18, hd=(84, 58, 92), vga=None,
                                 proj=None),
    'intel-realsense-d457': dict(url=D457_URL, minz=_MINZ_D457, baseline_mm=95, hd=(87, 58, 95), vga=(75, 62, 89),
                                 proj=(90, 63, 99)),
}


def _sensor(model_id: str, res=None):
    from ..core.project import Project, Sensor
    p = Project('depth check')
    s = Sensor(name=model_id, kind='depth', model_id=model_id)
    p.variant.sensors = [s]
    if res is not None:
        p.analysis.studies['depth_validity'] = {'sensors': {s.uid: {'resolution': list(res)}}}
    return p, s


# ====================================================================== 1. the error model against Monte Carlo
def _triangulate_z(f, cx, cy, B, uL, uR, v):
    """Depth of each match by linear triangulation (DLT, Hartley and Zisserman §12.2) with the rectified pair
    P_L = K [I | 0], P_R = K [I | (−B, 0, 0)]."""
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    PL = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    PR = K @ np.hstack([np.eye(3), np.array([[-B], [0.0], [0.0]])])
    A = np.stack([uL[:, None] * PL[2] - PL[0], v[:, None] * PL[2] - PL[1],
                  uR[:, None] * PR[2] - PR[0], v[:, None] * PR[2] - PR[1]], axis=1)
    _u, _s, vt = np.linalg.svd(A)
    X = vt[:, -1, :]
    return X[:, 2] / X[:, 3]


@check('depth.sigma_z_monte_carlo', 'depth', 'Stereo depth error σ_z = z² σ_d / (f B) against Monte Carlo',
       'largest relative difference between the model\'s σ_z and the standard deviation of depths triangulated from '
       'matches with Gaussian disparity noise; D405, D435 and D455 geometry (f at 1280 × 720 from the library), '
       'σ_d 0.08 and 0.3 px, depths from min-Z to the maximum range',
       'Monte Carlo: a rectified stereo pair written out in the check, Gaussian noise on the right-image match, '
       'linear (DLT) triangulation', 0.02, 'relative', refs=('hartley2004', 'gallup2008'))
def _sigma_mc(ctx):
    from ..core.studies import depth_validity as DV
    rng = ctx.rng('depth.sigma_z_monte_carlo')
    n = 6000 if ctx.quick else 40000
    rows, worst, xs, ys, bias = [], 0.0, [], [], 0.0
    for mid in ('intel-realsense-d405', 'intel-realsense-d435', 'intel-realsense-d455'):
        p, s = _sensor(mid, (1280, 720))
        ds = DV.setup(p, s)
        f, B = ds.f_px, float(ds.baseline_m)
        cx, cy = ds.ref.camera.cx, ds.ref.camera.cy
        for z in np.geomspace(ds.min_z_m, ds.max_range_m, 4):
            for sd in (0.08, 0.3):
                x, y = 0.1 * z, -0.05 * z
                uL = np.full(n, f * x / z + cx)
                vv = np.full(n, f * y / z + cy)
                uR = f * (x - B) / z + cx + rng.normal(0.0, sd, n)
                zt = _triangulate_z(f, cx, cy, B, uL, uR, vv)
                mc = float(np.std(zt, ddof=1))
                model = float(DV.sigma_z(z, f, B, sd))
                e = abs(mc / model - 1.0)
                d = f * B / z
                bias = max(bias, float(3.0 * (sd / d) ** 2))
                worst = max(worst, e)
                rows.append({'model': mid, 'z_m': float(z), 'sigma_d_px': sd, 'disparity_px': d, 'sigma_z_model_m':
                             model, 'sigma_z_mc_m': mc, 'rel_diff': e})
                xs.append(z)
                ys.append(e)
    se = 1.0 / math.sqrt(2.0 * (n - 1))
    tol = 4.5 * se + bias
    return Outcome(worst, {'trials_per_case': n, 'cases': len(rows), 'standard_error_rel': se,
                           'second_order_allowance': bias, 'rows': rows},
                   tolerance=tol,
                   figure=figure('Relative difference, model against Monte Carlo', 'z (m)', 'relative difference',
                                 [series('cases', xs, np.maximum(ys, 1e-9), 'points')], logy=True, hline=tol,
                                 hline_label='tolerance'),
                   note='The tolerance is 4.5 standard errors of a standard deviation estimated from the trials '
                        '(1/√(2(n − 1)) relative) plus 3 (σ_d / d)², the size of the second-order term the '
                        'first-order model leaves out (z = f B / d is not linear in d).')


# ====================================================================== 2. the D4xx tables typed again
@check('depth.d400_tables', 'depth', 'Intel D4xx min-Z, baseline and FOV against the datasheet tables',
       'figures that differ between the library as the study reads it (min-Z at each depth resolution through '
       'depth_validity.setup, baseline, depth FOV HD and VGA, projector FOV) and the datasheet tables typed again '
       'in the check; D405, D415, D435, D435i, D455, D457',
       f'Intel RealSense D400 series datasheet, Tables 3-49 and 4-12 ({D400_URL}), and D457 datasheet, Tables 3-19 '
       f'and 4-4 ({D457_URL}), read 2026-11-06, transcribed again', 0.0, 'figures',
       refs=('intel_d400', 'intel_d457'))
def _tables(ctx):
    from ..core import library as L
    from ..core.studies import depth_validity as DV
    wrong, total, rows = [], 0, []
    for mid, ds_ in DATASHEET.items():
        e = L.get(mid)
        if e is None:
            wrong.append(f'{mid}: not in the library')
            continue
        dp = e.data['depth']
        for (w, h), mm in zip(_RES, ds_['minz']):
            p, s = _sensor(mid, (w, h))
            got = DV.setup(p, s).min_z_m
            exp = None if mm is None else mm / 1000.0
            total += 1
            ok = (got is None and exp is None) or (got is not None and exp is not None and abs(got - exp) < 1e-9)
            rows.append({'model': mid, 'what': f'min-Z {w}x{h}', 'library': got, 'datasheet': exp, 'ok': ok})
            if not ok:
                wrong.append(f'{mid} min-Z {w}x{h}: library {got}, datasheet {exp}')
        checks = [('baseline (m)', dp.get('baseline_m'), ds_['baseline_mm'] / 1000.0)]
        hd = (dp.get('depth_fov') or {})
        checks += [(f'depth FOV HD {k}', hd.get(k), v) for k, v in zip('hvd', ds_['hd'])]
        ref = e.imager('depth') or {}
        fm = {tuple(m['aspect']): m for m in ref.get('fov_modes') or []}
        vga = fm.get((4, 3))
        if ds_['vga'] is None:
            checks.append(('depth FOV VGA (none stated)', vga is None, True))
        else:
            checks += [(f'depth FOV VGA {k}', (vga or {}).get(f'{k}fov_deg'), v) for k, v in zip('hvd', ds_['vga'])]
        pj = e.data.get('projector')
        if ds_['proj'] is None:
            checks.append(('no projector', pj is None, True))
        else:
            checks += [(f'projector FOV {k}', (pj or {}).get(f'{k}fov_deg'), v) for k, v in zip('hvd', ds_['proj'])]
        for what, got, exp in checks:
            total += 1
            ok = got == exp if isinstance(exp, bool) else (got is not None and abs(float(got) - float(exp)) < 1e-9)
            rows.append({'model': mid, 'what': what, 'library': got, 'datasheet': exp, 'ok': bool(ok)})
            if not ok:
                wrong.append(f'{mid} {what}: library {got}, datasheet {exp}')
    return Outcome(float(len(wrong)), {'figures': total, 'differ': wrong, 'rows': rows},
                   note='A second reading of the same tables, typed in this file without copying from the library '
                        'files: it finds transcription errors and a broken lookup (resolution to min-Z), not errors '
                        'in the datasheet. The D405 has no 640 × 480 min-Z ("-" in Table 4-12), so the study must '
                        'report it as not stated.')


# ====================================================================== 3. the stereo overlap against Intel's formula
def _sweep_extent(ds, z: float, step_deg: float):
    """(left edge, right edge) in degrees of condition 1 along the horizontal line y = 0 at depth z in the
    reference imager's optical frame (negative angles to the left)."""
    from ..core.frames import Pose
    from ..core.studies import depth_validity as DV
    a = np.radians(np.arange(-75.0, 75.0 + step_deg / 2, step_deg))
    Q = np.stack([z * np.tan(a), np.zeros_like(a), np.full_like(a, z)], 1)
    body = Pose()
    P = DV.imager_world(body, ds.ref).apply(Q)
    ev = DV.evaluate(ds, body, P)
    inside = np.degrees(a[ev['in_fov']])
    return float(inside.min()), float(inside.max())


@check('depth.fov_overlap_formula', 'depth', 'Stereo overlap (condition 1) against Intel\'s depth FOV at distance',
       'largest difference (degrees) between the horizontal extent where a point is inside both stereo imagers\' '
       'fields of view and Intel\'s Depth FOV = HFOV/2 + atan(tan(HFOV/2) − B/Z); D405, D435, D455 at 1280 × 720 '
       'and 848 × 480, Z from min-Z to 2 m', 'Intel RealSense D400 series datasheet §4.4, "Depth Field of View at '
       'Distance (Z)", the closed form typed again in the check', 0.01, '°', refs=('intel_d400',))
def _overlap(ctx):
    from ..core.studies import depth_validity as DV
    step = 0.002 if ctx.quick else 0.0005
    worst, rows, xs, ys = 0.0, [], [], []
    for mid in ('intel-realsense-d405', 'intel-realsense-d435', 'intel-realsense-d455'):
        for res in ((1280, 720), (848, 480)):
            p, s = _sensor(mid, res)
            ds = DV.setup(p, s)
            H = ds.ref.camera.fov_deg()['h']
            B = float(ds.baseline_m)
            for z in (ds.min_z_m, 0.5, 1.0, 2.0):
                lo, hi = _sweep_extent(ds, z, step)
                t = math.tan(math.radians(H / 2))
                formula = H / 2 + math.degrees(math.atan(t - B / z))
                got = hi - lo
                e = abs(got - formula)
                worst = max(worst, e)
                rows.append({'model': mid, 'resolution': list(res), 'z_m': z, 'hfov_deg': H, 'measured_deg': got,
                             'formula_deg': formula, 'diff_deg': e})
                xs.append(z)
                ys.append(e)
    return Outcome(worst, {'sweep_step_deg': step, 'rows': rows},
                   figure=figure('Depth FOV, condition 1 against Intel\'s formula', 'Z (m)', '|difference| (°)',
                                 [series('cases', xs, np.maximum(ys, 1e-9), 'points')], logy=True, hline=0.01,
                                 hline_label='tolerance'),
                   note='The extent is found by a sweep at the stated step, so it can be off by up to two steps; '
                        '0.01° leaves room for that and the pixel-edge tolerance of the projection. The formula '
                        'takes the left imager as the reference and the right imager one baseline to its right, '
                        'as the datasheet does.')


# ====================================================================== 4. finger shadows against ray casting
@check('depth.finger_shadow_raycast', 'depth', 'Finger shadows from both imagers against exact ray casting',
       'fraction of points inside both imagers\' fields of view whose "seen by imager k" answer from the study '
       '(cube maps from each imager centre) differs from exact ray casting from the same centre, not counting '
       'points within two texels of a silhouette; UR5e sample, wrist D405 and overhead D435, random arm poses with '
       'the gripper partly closed', 'core/raycast.visible_exact (Möller–Trumbore through a uniform grid) and '
       'raycast.near_silhouette, from each imager centre', 1e-3, 'fraction',
       refs=('moller1997', 'amanatides_woo1987', 'greene1986'), cost='study')
def _shadows(ctx):
    from ..core import occlusion as OC
    from ..core import raycast as RC
    from ..core import scenegeom as SG
    from ..core.robot.model import get_model
    from ..core.robot.samples import sample_project
    from ..core.studies import depth_validity as DV
    rng = ctx.rng('depth.finger_shadow_raycast')
    n_q, n_pts, n = (1, 1500, 256) if ctx.quick else (3, 12000, 512)
    p = sample_project('ur5e')
    r = p.robots[0]
    m = get_model(p, r.uid)
    lo, hi = p.workspace.volumes[0].bounds()
    grip = np.array([j.get('gripper', False) for j in r.joints if j.get('type') in ('revolute', 'continuous',
                                                                                   'prismatic')], bool)
    total, dis, elsewhere, one_only, per = 0, 0, 0, 0, []
    for _ in range(n_q):
        q = m.home.copy()
        q[:6] += rng.normal(0, 0.15, 6)
        if len(grip) == m.nq and grip.any():
            q[grip] = m.lower[grip] + rng.uniform(0.2, 0.6) * (m.upper[grip] - m.lower[grip])
        qd = {r.uid: m.clip(q)}
        occ = SG.scene_occluders(p, qd)
        for s in p.variant.sensors:
            ds = DV.setup(p, s)
            body = SG.sensor_world_pose(p, s, qd)
            origins = [DV.imager_world(body, im).position for im in ds.needed]
            occ_s = SG.sensor_occluders(p, s, qd, origins=origins, base=occ)
            # points in the reference imager's frustum (near the camera, where the fingers are) and in the volume
            cam = ds.ref.camera
            uv = np.stack([rng.uniform(0, cam.width - 1, n_pts), rng.uniform(0, cam.height - 1, n_pts)], 1)
            rays = cam.unproject(uv)
            zz = rng.uniform(0.03, 0.6, n_pts)
            Pa = DV.imager_world(body, ds.ref).apply(rays * (zz / rays[:, 2])[:, None])
            P = np.concatenate([Pa, rng.uniform(lo, hi, (n_pts, 3))])
            ev = DV.evaluate(ds, body, P, occluders=occ_s, n=n)
            inf = ev['in_fov']
            grid = RC.Grid.build(occ_s.tris)
            for k, o in enumerate(origins):
                Pk = P[inf]
                exact = RC.visible_exact(Pk, o, grid)
                got = ev['visible'][inf, k]
                bad = np.nonzero(exact != got)[0]
                sil = RC.near_silhouette(Pk[bad], o, grid, 2.0 / n) if len(bad) else np.zeros(0, bool)
                total += len(Pk)
                dis += len(bad)
                elsewhere += int((~sil).sum())
                per.append(len(bad) / max(len(Pk), 1))
            vis = ev['visible'][inf]
            one_only += int((vis.any(axis=1) & ~vis.all(axis=1)).sum())
    backend = OC.resolve_backend('auto')
    frac = dis / max(total, 1)
    return Outcome(elsewhere / max(total, 1),
                   {'point_imager_pairs': total, 'disagree_fraction': frac, 'disagree_near_silhouette': dis - elsewhere,
                    'disagree_elsewhere': elsewhere, 'points_seen_by_one_imager_only': one_only, 'face_size': n,
                    'backend': backend, 'poses': n_q},
                   ok=frac < 0.02 and one_only > 0,
                   figure=figure('Points where the study and ray casting disagree, per imager and pose', 'map',
                                 'fraction', [series('disagree', np.arange(len(per)), np.maximum(per, 1e-9),
                                                     'points')], logy=True),
                   note='The cube maps answer per texel, so points within about a texel of a silhouette may differ '
                        '(counted, held under 2 %); elsewhere the two must agree. The check also requires some '
                        'points to be seen by one imager only, so the parallax and finger-shadow case is exercised.')
