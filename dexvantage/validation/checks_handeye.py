"""Checks of hand-eye calibration (core/calib: boards, detection and PnP, the closed-form AX = XB and AX = ZB solvers,
the bundle adjustment and its covariance) against independent references.

The ground truth is synthetic (core/calib/synthetic.py): the bundled UR5e sample's kinematics, a camera where the
sample's wrist camera sits (or a static camera in front of a board the arm holds), board corners projected through a
synthetic 640 × 480 camera model (not a product's figures) with Gaussian pixel noise. The truth is the pose the
corners were made from, so every check compares a solver with the transform that generated its input.

- handeye.closed_form_exact: every closed-form method (Tsai-Lenz, Park-Martin, Horaud-Dornaika, Andreff, Daniilidis
  for AX = XB; Shah and Li for AX = ZB) on exact board poses, eye-in-hand and eye-to-hand: recovers X (and Z) to
  rounding. OpenCV's calibrateHandEye / calibrateRobotWorldHandEye are used when the installed OpenCV has them; the
  detail names the backend (OpenCV 5.0.0's Python package has only the method constants, so the NumPy versions of
  core/calib/axxb.py run there).
- handeye.closed_form_noisy: the same methods on board poses from PnP on noisy corners (0.3 px).
- handeye.ba_truth: the bundle adjustment against the truth, in units of its own predicted σ, on eye-in-hand,
  eye-to-hand and two-camera scenes, with intrinsics estimated, with encoder noise, and with joint offsets.
- handeye.ba_covariance_mc: the bundle adjustment's covariance against the spread of its estimates over Monte Carlo
  noise draws (the same poses), the ratio within the χ² band of the number of draws.
- handeye.axzb_two_arms: AX = ZB with both sides moving (the camera on one UR5e, the board held by a second): the
  bundle adjustment against the closed-form robot-world solvers (OpenCV's calibrateRobotWorldHandEye when installed,
  else the NumPy Shah and Li) and against the truth.
- handeye.charuco_render / handeye.aprilgrid_render: a board rendered at a known pose through a Brown-Conrady camera
  model (core/calib/synthetic.render_view), detected (core/calib/detect.py) and solved by PnP: corner error against
  the projected truth, and the pose error.
- handeye.numpy_vs_opencv: registered only when the installed OpenCV has the hand-eye functions: the NumPy solvers
  against OpenCV's on the same poses.
- handeye.planner_vs_ba: the pose planner's predicted covariance of X and Z (core/calib/planner.py: its Fisher
  information, the Jacobian written out by the chain rule) against the bundle adjustment's covariance (handeye.py:
  central differences of its own residual, σ estimated from the residuals) on corners simulated at the planned poses
  of the UR5e sample, eye-in-hand with the sample's wrist camera and the default board.
- handeye.sessions_joint: two capture sessions of the UR5e sample solved together (core/calib/capture.py
  sessions_problem and the handeye_solve study's joint path): the wrist camera's eye-in-hand views of the table board
  and the overhead camera's views of the same board taken at the same robot poses, the overhead camera's mount moved
  10 mm before the solve; both calibrated mounts against the truth the corners were made from, in units of their own
  predicted σ.
"""
from __future__ import annotations

import math

import numpy as np

from .suite import Outcome, check, figure, series

_REFS_HE = ('tsai_lenz1989', 'park_martin1994', 'horaud_dornaika1995', 'andreff1999', 'daniilidis1999',
            'opencv_handeye')
_REFS_RW = ('shah2013', 'li2010', 'opencv_handeye')


def _exact_inputs(prob, truth):
    """Exact board poses (truth) for every view of a single-camera, single-board problem."""
    from ..core.calib.axxb import inv_T
    c, b = prob.cameras[0], prob.boards[0]
    mdl = prob.robots['ur5e']
    Mc, Mb, C = [], [], []
    for o in prob.observations:
        q = prob.joints[o.pose]['ur5e']
        mc = mdl.link_pose(q, c.link) if c.robot else np.eye(4)
        mb = mdl.link_pose(q, b.link) if b.robot else np.eye(4)
        Mc.append(mc)
        Mb.append(mb)
        C.append(inv_T(mc @ truth['X'][c.name]) @ mb @ truth['Z'][b.name])
    return Mc, Mb, C


def _closed_form(prob, Mc, Mb, C, backend='auto'):
    """{method: (X, Z, backend)} for one camera-board pair (handeye._pair_candidates' algebra, written out)."""
    from ..core.calib import axxb
    from ..core.calib.axxb import inv_T, mean_transform
    b = prob.boards[0]
    B = [inv_T(a) @ m for a, m in zip(Mc, Mb)]
    out = {}
    G = Mc if b.static else [inv_T(m) for m in Mb]
    for meth in axxb.HAND_EYE_METHODS:
        X, bk = axxb.hand_eye(G, C, meth, backend)
        Z = mean_transform([inv_T(bb) @ X @ cc for bb, cc in zip(B, C)])
        out[meth] = (X, Z, bk)
    for meth in axxb.ROBOT_WORLD_METHODS:
        X, Z, bk = axxb.robot_world([inv_T(bb) for bb in B], [inv_T(cc) for cc in C], meth, backend)
        out[meth] = (X, Z, bk)
    return out


@check('handeye.closed_form_exact', 'handeye', 'Closed-form hand-eye solvers on exact poses',
       'largest translation error (m) of X and Z from each closed-form method (five AX = XB, two AX = ZB) on exact '
       'board poses, eye-in-hand and eye-to-hand on the UR5e sample; every rotation error must be below 1e-9 rad',
       'the ground-truth transforms the poses were generated from', 1e-9, 'm', refs=_REFS_HE + _REFS_RW[:2])
def _closed_exact(ctx):
    from ..core.calib import synthetic as S
    from ..core.calib.axxb import transform_error
    rows, worst_t, worst_r, backends = [], 0.0, 0.0, set()
    for kind in ('eye_in_hand', 'eye_to_hand'):
        prob, truth, _ = S.scene(kind, n_poses=8 if ctx.quick else 15, noise_px=0.0, seed=11)
        Mc, Mb, C = _exact_inputs(prob, truth)
        for meth, (X, Z, bk) in _closed_form(prob, Mc, Mb, C).items():
            ex = transform_error(X, truth['X'][prob.cameras[0].name])
            ez = transform_error(Z, truth['Z'][prob.boards[0].name])
            worst_t = max(worst_t, ex[0], ez[0])
            worst_r = max(worst_r, ex[1], ez[1])
            backends.add(bk)
            rows.append((f'{kind} {meth}', max(ex[0], ez[0])))
    return Outcome(worst_t, {'worst_rotation_rad': worst_r, 'backend': sorted(backends),
                             'per_method_m': {k: v for k, v in rows}},
                   ok=worst_r <= 1e-9,
                   figure=figure('Translation error per method (exact poses)', 'method', 'm',
                                 [series('error', np.arange(len(rows)), [max(v, 1e-18) for _, v in rows], 'points')],
                                 logy=True, hline=1e-9, hline_label='tolerance', kind='bars',
                                 categories=[k for k, _ in rows]),
                   note='Every method is exact on noise-free motions; what is left is rounding (about 1e-15 m). '
                        '1e-9 m fails any convention error (a transform inverted or A and B swapped gives errors of '
                        'centimetres to metres). Backend "numpy": the NumPy versions in core/calib/axxb.py, because '
                        'this OpenCV has no calibrateHandEye in its Python bindings.')


@check('handeye.closed_form_noisy', 'handeye', 'Closed-form hand-eye solvers on noisy views',
       'largest translation error (mm) of X from each closed-form method on board poses from PnP on corners with '
       '0.3 px noise (20 poses, eye-in-hand and eye-to-hand, several seeds); every rotation error must be below 1.5°',
       'the ground-truth camera pose', 15.0, 'mm', refs=_REFS_HE + _REFS_RW[:2] + ('hartley2004',))
def _closed_noisy(ctx):
    from ..core.calib import handeye as HE
    from ..core.calib import synthetic as S
    from ..core.calib.axxb import transform_error
    per = {}
    seeds = range(2) if ctx.quick else range(5)
    for kind in ('eye_in_hand', 'eye_to_hand'):
        for sd in seeds:
            prob, truth, _ = S.scene(kind, n_poses=20, noise_px=0.3, seed=int(ctx.rng('he.noisy', sd).integers(10**6)))
            ini = HE.initialise(prob, prob.observations, HE.Settings())
            for m in ini['methods']:
                t, r = transform_error(np.array(m['X']), truth['X'][m['camera']])
                k = f'{kind} {m["method"]}'
                a, b = per.get(k, (0.0, 0.0))
                per[k] = (max(a, 1000 * t), max(b, math.degrees(r)))
    worst_t = max(v[0] for v in per.values())
    worst_r = max(v[1] for v in per.values())
    keys = list(per)
    return Outcome(worst_t, {'worst_rotation_deg': worst_r, 'seeds': len(seeds), 'noise_px': 0.3,
                             'per_method_mm_deg': {k: [round(a, 4), round(b, 5)] for k, (a, b) in per.items()}},
                   ok=worst_r <= 1.5,
                   figure=figure('Worst translation error per method (noisy views)', 'method', 'mm',
                                 [series('error', np.arange(len(keys)), [per[k][0] for k in keys], 'points')],
                                 hline=15.0, hline_label='tolerance', kind='bars', categories=keys),
                   note='The closed-form methods solve algebraic errors on PnP poses that are each a few tenths of a '
                        'millimetre off, so they are not exact; on these scenes their errors were 3–8 mm and up to '
                        '0.6° when the tolerance was set. 15 mm and 1.5° leave margin over that spread and still '
                        'fail a convention error (tens of millimetres to tens of degrees). Tsai-Lenz is singular '
                        'for a camera rotation of 180°; the eye-to-hand scene\'s camera is at 151°.')


def _norm_errors(res, truth, names) -> list:
    from ..core.calib.handeye import component_errors
    out = []
    for k in names:
        T = res.X.get(k, res.Z.get(k))
        C = res.cov_X.get(k, res.cov_Z.get(k))
        ref = truth['X'].get(k, truth['Z'].get(k))
        e = component_errors(T, ref)
        s = np.sqrt(np.clip(np.diag(C), 1e-30, None))
        out.append((k, e, s))
    return out


@check('handeye.ba_truth', 'handeye', 'Bundle adjustment against the truth',
       'largest error of any estimated quantity (X, Z, intrinsics, determined joint offsets) in units of its own '
       'predicted σ, on eye-in-hand, eye-to-hand, two-camera, intrinsics, encoder-noise and joint-offset scenes '
       '(0.3 px noise); the translation errors must also stay below 1 mm for the wrist and held-board transforms',
       'the ground truth the corners were generated from', 5.0, 'σ',
       refs=('triggs2000', 'hartley2004'), cost='study')
def _ba_truth(ctx):
    from ..core.calib import handeye as HE
    from ..core.calib import synthetic as S
    rng = ctx.rng('handeye.ba_truth')
    scenes = [('eye-in-hand, two cameras', dict(kind='eye_in_hand', two_cameras=True), HE.Settings()),
              ('eye-to-hand, two cameras', dict(kind='eye_to_hand', two_cameras=True), HE.Settings())]
    if not ctx.quick:
        cam = S.synthetic_camera(model='brown', dist=(-0.12, 0.05, 0.0005, -0.0003, 0.0))
        guess = S.synthetic_camera(f=470.0, model='brown', dist=(0.0,) * 5)
        scenes += [('eye-in-hand, intrinsics estimated', dict(kind='eye_in_hand', camera=cam,
                                                              estimate_intrinsics=True,
                                                              intrinsics_guess={'wrist': guess}), HE.Settings()),
                   ('eye-to-hand, encoder noise 0.02°', dict(kind='eye_to_hand', encoder_noise_deg=0.02),
                    HE.Settings(encoder_sigma_deg=0.02)),
                   ('eye-in-hand, joint offsets', dict(kind='eye_in_hand', two_cameras=True, n_poses=25,
                                                       joint_offsets_deg={'ur5e': [0.0, -0.4, 0.25, 0.5, -0.3, 0.0]
                                                                          + [0.0] * 8}),
                    HE.Settings(joint_offsets=('ur5e',)))]
    worst, rows, abs_worst = 0.0, [], 0.0
    for label, kw, st in scenes:
        kw = dict(kw)
        kw.setdefault('n_poses', 20)
        cam_true = kw.get('camera')
        prob, truth, _ = S.scene(noise_px=0.3, seed=int(rng.integers(10**6)), **kw)
        res = HE.solve(prob, st)
        if not res.ok:
            return Outcome(float('inf'), {'scene': label, 'message': res.message}, ok=False,
                           note=f'The solver failed on "{label}".')
        z = []
        names = [c.name for c in prob.cameras] + [b.name for b in prob.boards]
        if 'joint_offsets_deg' in kw:
            names = []             # X and Z absorb the undetermined offsets (first and last joint); see below
        for k, e, s in _norm_errors(res, truth, names):
            z.append(float(np.max(np.abs(e) / s)))
            if k in ('wrist', 'board') and (k != 'board' or prob.boards[0].robot):
                abs_worst = max(abs_worst, 1000 * float(np.linalg.norm(e[3:])))
        if cam_true is not None:
            est = res.intrinsics['wrist']
            vals = np.r_[est['model'].fx, est['model'].fy, est['model'].cx, est['model'].cy, est['model'].dist]
            tru = np.r_[cam_true.fx, cam_true.fy, cam_true.cx, cam_true.cy, cam_true.dist]
            z.append(float(np.max(np.abs(vals - tru) / np.maximum(est['sigma'], 1e-30))))
        if 'joint_offsets_deg' in kw:
            jo = res.joint_offsets['ur5e']
            true = np.degrees(truth['joint_offsets']['ur5e'])[:len(jo['value_deg'])]
            for v, s, d, t in zip(jo['value_deg'], jo['sigma_deg'], jo['determined'], true):
                if d:
                    z.append(abs(v - t) / s)
        zz = max(z) if z else 0.0
        worst = max(worst, zz)
        rows.append((label, zz, res.rms_px))
    return Outcome(worst, {'scenes': {k: {'worst_sigma': round(v, 3), 'rms_px': round(r, 4)} for k, v, r in rows},
                           'worst_abs_translation_mm': abs_worst},
                   ok=abs_worst <= 1.0,
                   figure=figure('Largest error over predicted σ per scene', 'scene', 'σ',
                                 [series('|error| / σ', np.arange(len(rows)), [v for _, v, _ in rows], 'points')],
                                 hline=5.0, hline_label='tolerance', kind='bars', categories=[k for k, _, _ in rows]),
                   note='With Gaussian corner noise an unbiased estimate is within about 3σ in each component almost '
                        'always; 5σ over the few tens of components checked is passed by chance with probability '
                        'below 1e-5, while a biased model (a frame convention, a wrong FK link, a pixel-centre '
                        'error) is off by tens to thousands of σ. In the joint-offset scene the first and last '
                        'joints\' offsets are not determined by the views (they trade off against Z and X about '
                        'the same axes); only the determined offsets are compared there.')


@check('handeye.ba_covariance_mc', 'handeye', 'Bundle-adjustment covariance against Monte Carlo',
       'largest share of the χ² band used by the ratio empirical σ / predicted σ over the 12 components of X and Z '
       '(ln ratio over ln of the band edge on its side; 1 = at the edge), the empirical σ the RMS error about the '
       'truth over noise draws (same poses, 0.5 px), the predicted σ the RMS of the per-draw Gauss-Newton σ, the '
       'band the 99 % χ² interval of the number of draws (Bonferroni over the 12 components)',
       'a Monte Carlo simulation of the corner noise', 1.0, 'of the band', refs=('triggs2000', 'hartley2004'),
       cost='study')
def _ba_mc(ctx):
    from scipy.stats import chi2

    from ..core.calib import handeye as HE
    from ..core.calib import synthetic as S
    from ..core.calib.handeye import component_errors
    n = 12 if ctx.quick else 40
    base, truth, _ = S.scene('eye_in_hand', n_poses=15, noise_px=0.0, seed=int(ctx.rng('he.mc').integers(10**6)))
    cname, bname = base.cameras[0].name, base.boards[0].name
    E, P = [], []
    for k in range(n):
        prob = S.with_noise(base, 0.5, int(ctx.rng('he.mc', k + 1).integers(10**9)))
        res = HE.solve(prob)
        if not res.ok:
            return Outcome(float('inf'), {'draw': k, 'message': res.message}, ok=False)
        E.append(np.r_[component_errors(res.X[cname], truth['X'][cname]),
                       component_errors(res.Z[bname], truth['Z'][bname])])
        P.append(np.r_[np.sqrt(np.diag(res.cov_X[cname])), np.sqrt(np.diag(res.cov_Z[bname]))])
    E, P = np.array(E), np.array(P)
    emp = np.sqrt(np.mean(E ** 2, axis=0))
    pred = np.sqrt(np.mean(P ** 2, axis=0))
    ratio = emp / pred
    alpha = 0.01 / 12
    lo = math.sqrt(chi2.ppf(alpha / 2, n) / n)
    hi = math.sqrt(chi2.ppf(1 - alpha / 2, n) / n)
    lr = np.log(ratio)
    err = float(np.max(np.where(lr >= 0, lr / math.log(hi), lr / math.log(lo))))
    labels = [f'X {a}' for a in ('rx', 'ry', 'rz', 'x', 'y', 'z')] + [f'Z {a}' for a in ('rx', 'ry', 'rz', 'x', 'y',
                                                                                           'z')]
    return Outcome(err, {'draws': n, 'ratios': {k: round(float(v), 4) for k, v in zip(labels, ratio)},
                         'band': [lo, hi], 'noise_px': 0.5, 'poses': 15,
                         'predicted_sigma': {k: float(v) for k, v in zip(labels, pred)}},
                   figure=figure(f'Empirical over predicted σ ({n} draws; band {lo:.2f}–{hi:.2f})', 'component',
                                 'ratio', [series('ratio', np.arange(12), ratio, 'points')], hline=1.0,
                                 hline_label='1', kind='bars', categories=labels),
                   note='With n draws the empirical σ of an unbiased Gaussian estimate scatters about the true σ by '
                        'the χ² law with n degrees of freedom; the band holds 99 % of that scatter for each of the 12 '
                        'components together. A covariance missing the residual variance, or J built at the wrong '
                        'point, is off by factors of 2 or more.')


@check('handeye.axzb_two_arms', 'handeye', 'AX = ZB with both sides moving',
       'largest translation difference (mm) between the bundle adjustment\'s X and Z and the closed-form robot-world '
       'solvers\' (Shah, Li) on a camera on one UR5e seeing a board held by a second (20 poses, 0.3 px); the bundle '
       'adjustment must also be within 5σ of the truth',
       'OpenCV calibrateRobotWorldHandEye when installed, else the NumPy Shah and Li solvers; and the ground truth',
       15.0, 'mm', refs=_REFS_RW + ('triggs2000',), cost='study')
def _axzb(ctx):
    from ..core.calib import axxb
    from ..core.calib import handeye as HE
    from ..core.calib import synthetic as S
    prob, truth, _ = S.two_arm_scene(n_poses=12 if ctx.quick else 20, noise_px=0.3,
                                     seed=int(ctx.rng('handeye.axzb').integers(10**6)))
    res = HE.solve(prob)
    if not res.ok:
        return Outcome(float('inf'), {'message': res.message}, ok=False)
    cname, bname = prob.cameras[0].name, prob.boards[0].name
    diffs, detail = [], {}
    for m in res.initial['methods']:
        dx = axxb.transform_error(np.array(m['X']), res.X[cname])
        dz = axxb.transform_error(np.array(m['Z']), res.Z[bname])
        ex = axxb.transform_error(np.array(m['X']), truth['X'][cname])
        diffs.append(1000 * max(dx[0], dz[0]))
        detail[m['method']] = {'vs_ba_mm': 1000 * max(dx[0], dz[0]), 'vs_ba_deg': math.degrees(max(dx[1], dz[1])),
                               'X_vs_truth_mm': 1000 * ex[0]}
    z = max(float(np.max(np.abs(e) / s)) for _k, e, s in _norm_errors(res, truth, [cname, bname]))
    ba = HE.errors_against(res, {**truth['X'], **truth['Z']})
    backend = 'opencv' if axxb.opencv_has_robot_world() else 'numpy'
    return Outcome(max(diffs) if diffs else float('inf'),
                   {'closed_form': detail, 'backend': backend, 'ba_vs_truth_mm_deg': ba, 'ba_worst_sigma': z,
                    'mode': prob.mode()},
                   ok=bool(diffs) and z <= 5.0,
                   note='The closed-form solvers start from PnP poses and minimise algebraic errors, so they differ '
                        'from the bundle adjustment by the order of their own error (a few millimetres here); the '
                        'tolerance is that of handeye.closed_form_noisy. The bundle adjustment itself is held to the '
                        'truth within 5σ, as in handeye.ba_truth.')


def _render_check(ctx, board, cid):
    from ..core.calib import detect as D
    from ..core.calib import synthetic as S
    from ..core.calib.axxb import transform_error
    cam = S.synthetic_camera(model='brown', dist=(-0.12, 0.05, 0.0005, -0.0003, 0.0))
    rng = ctx.rng(cid)
    poses = [(0.40, (0.0, 0.0), (0.0, 0.0)), (0.45, (25.0, -15.0), (0.05, -0.04)), (0.5, (-35.0, 20.0), (-0.1, 0.06)),
             (0.6, (15.0, 40.0), (0.12, 0.08)), (0.38, (-10.0, -30.0), (-0.05, -0.1))]
    if ctx.quick:
        poses = poses[:3]
    worst, rows, pose_worst = 0.0, [], (0.0, 0.0)
    min_share = 1.0
    for k, (d, tilt, c) in enumerate(poses):
        T = S.look_at_board(d, tilt, c, roll_deg=float(rng.uniform(-30, 30)), board=board)
        img = S.render_view(board, cam, T, noise=1.0, seed=int(rng.integers(10**6)))
        det = D.detect(img, board, cam)
        if det is None:
            return Outcome(float('inf'), {'view': k, 'pose': [d, tilt, c]}, ok=False, note='The board was not found.')
        P = det.points @ T[:3, :3].T + T[:3, 3]
        uv, _ = cam.project(P)
        e = np.linalg.norm(det.corners - uv, axis=1)
        r = float(np.sqrt(np.mean(e ** 2)))
        worst = max(worst, r)
        pnp = D.solve_pnp(det.points, det.corners, cam)
        te = transform_error(pnp.T_cam_board, T)
        pose_worst = (max(pose_worst[0], 1000 * te[0]), max(pose_worst[1], math.degrees(te[1])))
        share = det.n / board.n_points
        min_share = min(min_share, share)
        rows.append({'distance_m': d, 'tilt_deg': list(tilt), 'corners': det.n, 'rms_px': r,
                     'max_px': float(e.max()), 'pnp_rms_px': pnp.rms_px, 'pose_error_mm': 1000 * te[0],
                     'pose_error_deg': math.degrees(te[1])})
    return Outcome(worst, {'views': rows, 'worst_pose_error_mm': pose_worst[0],
                           'worst_pose_error_deg': pose_worst[1], 'least_share_of_corners': min_share,
                           'camera': cam.summary(), 'board': board.describe()},
                   ok=pose_worst[0] <= 1.0 and pose_worst[1] <= 0.2 and min_share >= 0.5,
                   figure=figure('Corner RMS per view', 'view', 'px',
                                 [series('RMS', np.arange(len(rows)), [r['rms_px'] for r in rows], 'points')],
                                 hline=0.25, hline_label='tolerance'))


@check('handeye.charuco_render', 'handeye', 'ChArUco detection on rendered boards',
       'largest per-view RMS (px) between the detected ChArUco corners and the true corners projected through the '
       'camera model, on boards rendered at known poses (0.38–0.6 m, up to 40° of tilt, 1 grey level of noise) '
       'through a Brown-Conrady camera; the PnP pose must be within 1 mm and 0.2° and half the corners found',
       'the known board pose, projected through the same camera model', 0.25, 'px',
       refs=('garrido2014', 'opencv_charuco', 'opencv_calib3d', 'hartley2004'))
def _charuco(ctx):
    from ..core.calib.boards import CharucoBoard
    out = _render_check(ctx, CharucoBoard(9, 7, 0.03, 0.022), 'handeye.charuco_render')
    out.note = ('Sub-pixel corner detectors locate sharp chessboard corners to about a tenth of a pixel; 0.25 px '
                'fails a half-pixel convention error, a wrong board layout or a wrong corner order (a pixel or more). '
                'The renderer integrates each pixel over 3 × 3 rays, so the corners carry the area coverage a '
                'sensor sees.')
    return out


@check('handeye.aprilgrid_render', 'handeye', 'AprilGrid detection on rendered boards',
       'largest per-view RMS (px) between the detected AprilGrid tag corners (Kalibr layout, corner squares) and the '
       'true corners projected through the camera model, as handeye.charuco_render',
       'the known board pose, projected through the same camera model', 0.25, 'px',
       refs=('wang_olson2016', 'kalibr', 'opencv_charuco', 'hartley2004'))
def _aprilgrid(ctx):
    from ..core.calib.boards import AprilGrid
    out = _render_check(ctx, AprilGrid(6, 5, 0.04), 'handeye.aprilgrid_render')
    out.note = ('With Kalibr\'s corner squares every tag corner is a chessboard-like X corner, which the sub-pixel '
                'step locates without bias; 0.25 px as for ChArUco.')
    return out


SHAPE_TOL = 0.01      # ln ratio: the planner's and the adjustment's covariance shapes (σ̂ / σ taken out)


@check('handeye.planner_vs_ba', 'handeye', 'Planned covariance against the bundle adjustment',
       'largest |ln(σ_BA / σ_plan)| over the 12 components of X and Z: σ_plan the pose planner\'s predicted 1σ at '
       'the planned poses (15, UR5e sample, eye-in-hand, the sample\'s wrist camera, the default ChArUco board, '
       '0.5 px), σ_BA the bundle adjustment\'s 1σ from corners simulated at the same poses with 0.5 px Gaussian noise '
       '(the same corner ids), σ estimated from its residuals',
       'the bundle adjustment of core/calib/handeye.py (an independent Jacobian, by central differences of its '
       'residuals, linearised at its own estimate)', math.log(1.1), 'ln ratio', refs=('triggs2000', 'hartley2004'),
       cost='study')
def _planner_vs_ba(ctx):
    from ..core.calib import handeye as HE
    from ..core.calib import planner as PL
    from ..core.calib import records as RC
    from ..core.calib.axxb import inv_T
    from ..core.robot.samples import sample_project
    proj = sample_project('ur5e')
    proj.calibration['boards'].append(RC.default_board_record(proj, 'eye_in_hand', proj.robots[0].uid))
    rig = PL.rig_from_project(proj)
    sigma = 0.5
    n = 10 if ctx.quick else 15
    # the quick run plans on a coarser grid without the occlusion check (it changes which corners count, not the
    # model compared)
    pst = PL.PlanSettings(n_poses=n, pixel_sigma=sigma, occlusion=not ctx.quick,
                          rolls_deg=(-45.0, 0.0, 45.0) if ctx.quick else PL.ROLLS_DEG)
    pl = PL.plan(rig, pst)
    if len(pl.doc['poses']) < n:
        return Outcome(float('inf'), {'planned': len(pl.doc['poses'])}, ok=False, note='Too few poses planned.')
    # the corners at the planned poses, projected from the truth (the nominal mounts) and made noisy, written out
    rng = ctx.rng('handeye.planner_vs_ba')
    uid, link = rig.robot_uid, rig.link
    P = rig.board.points()
    joints, obs = [], []
    for k, ps in enumerate(pl.doc['poses']):
        q = np.asarray(ps['q'], float)
        ids = np.asarray(ps['ids'], int)
        T = inv_T(rig.model.link_pose(q, link) @ rig.X) @ rig.Z
        uv, _ok = rig.camera.project(P[ids] @ T[:3, :3].T + T[:3, 3])
        joints.append({uid: q})
        obs.append(HE.Observation('camera', 'board', k, ids, uv + rng.normal(0.0, sigma, uv.shape)))
    prob = HE.Problem([HE.CameraSpec('camera', rig.camera, uid, link)], [HE.BoardSpec('board', P, board=rig.board)],
                      {uid: rig.model}, joints, obs)
    res = HE.solve(prob)
    if not res.ok:
        return Outcome(float('inf'), {'message': res.message}, ok=False)
    s_ba = np.r_[np.sqrt(np.diag(res.cov_X['camera'])), np.sqrt(np.diag(res.cov_Z['board']))]
    C = np.array([[np.nan if v is None else v for v in row] for row in pl.doc['covariance']], float)
    s_pl = np.sqrt(np.diag(C))
    ratio = s_ba / s_pl
    err = float(np.max(np.abs(np.log(ratio))))
    # the shape: the ratio with the bundle adjustment's σ̂ / σ taken out (what remains is the Jacobians' difference)
    shape = float(np.max(np.abs(np.log(ratio * sigma / res.sigma_px))))
    labels = [f'X {a}' for a in PL.AXES] + [f'Z {a}' for a in PL.AXES]
    m = res.dof
    return Outcome(err, {'ratios': {k: round(float(v), 4) for k, v in zip(labels, ratio)},
                         'shape_ln_ratio': shape, 'shape_tolerance': SHAPE_TOL,
                         'sigma_plan': pl.doc['sigma'], 'sigma_ba': {'X': res.sigma('camera'),
                                                                     'Z': res.sigma('board')},
                         'poses': len(obs), 'corners': int(sum(len(o.ids) for o in obs)), 'dof': m,
                         'sigma_hat_px': res.sigma_px, 'noise_px': sigma, 'occlusion_checked': pst.occlusion,
                         'candidates': pl.doc['candidates'], 'camera': rig.camera.summary(),
                         'board': rig.board.describe()},
                   ok=shape <= SHAPE_TOL,
                   figure=figure(f'σ_BA / σ_plan ({len(obs)} planned poses)', 'component', 'ratio',
                                 [series('ratio', np.arange(12), ratio, 'points')], hline=1.0, hline_label='1',
                                 kind='bars', categories=labels),
                   note='Both are the Gauss-Newton covariance of the same estimator, so they agree when the planner\'s '
                        'Jacobian is right; they differ by the bundle adjustment\'s σ̂ (its relative scatter is '
                        f'1/√(2·dof), {100 / math.sqrt(2 * max(m, 1)):.1f} % here, 3 × that within the tolerance) and '
                        'by linearising at its estimate instead of the truth (far smaller at 0.5 px). With σ̂ / σ taken '
                        f'out the ratios must agree within {SHAPE_TOL:g} in ln (the shape of the covariance). A wrong '
                        'sign, frame or missing rotation coupling in the planner gives factors of 2 or more. The '
                        'bundle adjustment\'s covariance is itself held to a Monte Carlo spread by '
                        'handeye.ba_covariance_mc.')



def joint_sessions_project(n_poses: int = 12, noise_px: float = 0.2, seed: int = 0, quick: bool = True,
                           project=None):
    """(project, truth {sensor uid: X 4 × 4, the optical frame in its mount frame}): the UR5e sample with the default
    table board and two capture sessions made from the truth: the wrist camera at n_poses planned eye-in-hand poses,
    and the overhead camera seeing the same board at the same joint vectors (images taken together); corners
    projected through each camera's model with N(0, noise_px²) noise on each axis. The overhead camera's mount is
    then moved 10 mm along its body x. project: a UR5e sample project to add them to (changed in place), else a
    new one."""
    from ..core.calib import planner as PL
    from ..core.calib import records as RC
    from ..core.calib import synthetic as S
    from ..core.frames import Pose
    from ..core.robot.samples import sample_project
    p = sample_project('ur5e') if project is None else project
    brec = RC.default_board_record(p, 'eye_in_hand', p.robots[0].uid)
    p.calibration['boards'].append(brec)
    wrist = next(x for x in p.variant.sensors if x.mount.on_link)
    over = next(x for x in p.variant.sensors if not x.mount.on_link)
    rig = PL.rig_from_project(p, wrist.uid, brec['uid'])
    pst = PL.PlanSettings(n_poses=n_poses, occlusion=not quick, rolls_deg=(-45.0, 0.0, 45.0) if quick else PL.ROLLS_DEG)
    pl = PL.plan(rig, pst)
    rng = np.random.default_rng(seed)
    im_o = RC.sensor_imager(over)
    _r, _l, Xo = RC.camera_mount(p, over, im_o)
    To = np.linalg.inv(Xo) @ RC.board_mount(brec)[2]
    mo, uvo = S.visible(im_o.camera, To, rig.points)
    pw, po = [], []
    for k, ps in enumerate(pl.doc['poses']):
        q = np.asarray(ps['q'], float)
        mk, uvk = S.visible(rig.camera, rig.T_cam_board(q), rig.points)
        pw.append({'name': f'pose {k + 1}', 'q': q.tolist(), 'ids': np.nonzero(mk)[0].tolist(),
                   'uv': (uvk[mk] + rng.normal(0.0, noise_px, (int(mk.sum()), 2))).tolist()})
        po.append({'name': f'pose {k + 1}', 'q': q.tolist(), 'ids': np.nonzero(mo)[0].tolist(),
                   'uv': (uvo[mo] + rng.normal(0.0, noise_px, (int(mo.sum()), 2))).tolist()})
    uid = p.robots[0].uid
    p.calibration['sessions'] += [
        RC.session_record(wrist, RC.sensor_imager(wrist), 'eye_in_hand', brec['uid'], uid, rig.link, pw,
                          {'kind': 'folder'}, name='Wrist'),
        RC.session_record(over, im_o, 'eye_to_hand', brec['uid'], uid, '', po, {'kind': 'folder'}, name='Overhead')]
    truth = {wrist.uid: RC.camera_mount(p, wrist, RC.sensor_imager(wrist))[2], over.uid: Xo}
    shift = np.eye(4)
    shift[0, 3] = 0.01
    over.mount.pose = Pose.from_matrix(over.mount.pose.matrix() @ shift)
    return p, truth


@check('handeye.sessions_joint', 'handeye', 'Two capture sessions in one bundle adjustment',
       'largest error of a calibrated mount (the wrist camera eye-in-hand; the overhead camera, fixed, seeing the same '
       'fixed board, located only through it) against the truth, per translation axis in units of its own predicted '
       'σ; the overhead camera starts 10 mm off',
       'the ground-truth mounts the corners were generated from (0.2 px noise)', 5.0, 'σ',
       refs=('triggs2000', 'hartley2004'), cost='study')
def _sessions_joint(ctx):
    from ..core import jobs
    p, truth = joint_sessions_project(10 if ctx.quick else 15, 0.2, int(ctx.rng('handeye.sessions_joint')
                                                                         .integers(0, 2 ** 31 - 1)), ctx.quick)
    p.analysis.studies['handeye_solve'] = {'sessions': 'all', 'coach': False}
    doc = jobs.run_study('handeye_solve', p)
    if not doc.get('ok'):
        return Outcome(float('inf'), {'message': doc.get('message')}, ok=False)
    worst, per, mm = 0.0, {}, {}
    for c in doc['cameras']:
        X = np.asarray(doc['result']['cameras'][c['name']]['matrix'], float)
        e = np.abs(X[:3, 3] - truth[c['sensor_uid']][:3, 3])          # along the mount frame's axes, as σ
        sig = np.array([c['sigma'][k] for k in ('x_mm', 'y_mm', 'z_mm')], float) / 1000.0
        z = e / np.maximum(sig, 1e-12)
        per[c['name']] = [round(float(v), 3) for v in z]
        mm[c['name']] = [round(1000 * float(v), 4) for v in e]
        worst = max(worst, float(z.max()))
    return Outcome(worst, {'errors_sigma': per, 'errors_mm': mm, 'views': len(doc['views']),
                           'modes': {c['name']: c['mode'] for c in doc['cameras']}},
                   ok=len(doc['cameras']) == 2,
                   note='Errors of an unbiased estimator are within 5σ with probability above 99.9 % per component; '
                        'a camera located through the wrong board pose, or a shared board estimated twice, shows as '
                        'its 10 mm start offset, many σ away.')

def _register_opencv_comparison():
    from ..core.calib import axxb
    if not (axxb.opencv_has_hand_eye() and axxb.opencv_has_robot_world()):
        return

    @check('handeye.numpy_vs_opencv', 'handeye', 'NumPy closed-form solvers against OpenCV',
           'largest translation difference (m) between the NumPy closed-form solvers and OpenCV\'s calibrateHandEye '
           '/ calibrateRobotWorldHandEye on the same exact poses',
           'OpenCV calibrateHandEye and calibrateRobotWorldHandEye', 1e-9, 'm', refs=_REFS_HE + _REFS_RW[:2])
    def _np_vs_cv(ctx):
        from ..core.calib import synthetic as S
        from ..core.calib.axxb import transform_error
        worst, noisy = 0.0, {}
        for kind in ('eye_in_hand', 'eye_to_hand'):
            prob, truth, _ = S.scene(kind, n_poses=12, noise_px=0.0, seed=21)
            Mc, Mb, C = _exact_inputs(prob, truth)
            a = _closed_form(prob, Mc, Mb, C, 'numpy')
            b = _closed_form(prob, Mc, Mb, C, 'opencv')
            for m in a:
                worst = max(worst, transform_error(a[m][0], b[m][0])[0], transform_error(a[m][1], b[m][1])[0])
        return Outcome(worst, {'noisy': noisy},
                       note='Both recover the truth exactly on noise-free poses; with noise they may differ by the '
                            'order of the noise (axxb.py says where the NumPy versions follow the papers).')


_register_opencv_comparison()
