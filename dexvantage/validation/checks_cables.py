"""Checks of the cable model (core/cables.py) and the cable study (core/studies/cables.py) against independent
references.

cables.routed_arc — the routed length over a revolute joint against the circle arc and the helix written out here.
A scripted robot (a static link "a" and a link "b" on a hinge about z, range ±2.8 rad) stands at a moved and rotated
base pose (so Robot.base must be applied). Attachments at the same distance r from the axis, at heights differing by
dz, at an angle α₀ apart at home: with the default wrap radius (= r) the cable lies on the cylinder of radius r, so
its length is the helix's, L(q) = √((r |q − α₀|)² + dz²) (dz = 0: the circle arc r |q − α₀|). A third case with
unequal radii and a smaller wrap radius ρ is compared with the belt-over-a-pulley length written out again
(tangent lengths √(d² − ρ²) plus the arc ρ (|θ| − arccos(ρ/d_a) − arccos(ρ/d_b)), or the straight chord when the line
clears the cylinder) and the unrolled height. The joint is swept over its range through the study's own path
(crossed joints, the posed model, the unwrapped angle); the angle the study unwraps must also agree with the
geometric angle modulo 2π at every sample.

cables.helix_bend — the bend radius of a taut run on the cylinder (the study's estimate, and the drawn 49-point
polyline's) against the radius of curvature of the helix, c² / a with a = r and c² = r² + b², b = dz / θ [do Carmo
1976, §1-5].

cables.loop_arc — the slack loop on a straight run (the study's radius estimate, and the drawn bow laid over the
chord) against the textbook circular arc of length s over the chord c: its radius R = s / (2β) with
sin β / β = c / s solved by scipy's brentq.

cables.snag_bruteforce — the snag test (ray casting of the loop's segments against the links' triangles, plus a
parity test of the loop's middle point) against brute-force sampling: the same loop polylines resampled every
0.5 mm, each point tested for lying inside each box of the scripted robot in closed form (the box's own frame from
the base pose and Rz(q) written out here). Poses where the deepest point lies within 1 mm of a box's surface are
grazing and not counted; every other pose must agree. Along the trajectory the share of time snagged by the study's
sampling (2° steps, the default) is also compared with brute force at 0.2° steps.
"""
from __future__ import annotations

import os

import numpy as np

from .suite import Outcome, check, figure, series

R_SAME = 0.08                     # m, attachment radius of the circle-arc and helix cases
ALPHA0 = 0.5                      # rad, the angle from B to A at home
RANGE = 2.8                       # rad, the hinge's range either side of zero
POSTS = ((1.2, 0.12), (-1.5, 0.11))     # (angle rad, radius m) of the two posts on link "a"
POST_HALF = (0.012, 0.012, 0.1)
BASE_HALF, BASE_POS = (0.06, 0.06, 0.02), (0.0, 0.0, -0.04)
ARM_HALF, ARM_POS = (0.15, 0.02, 0.02), (0.25, 0.0, 0.0)


def _rz(a) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot(axis, a) -> np.ndarray:
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def scripted(folder: str):
    """(project, robot, model, base rotation, base position) of the scripted hinge robot with two posts."""
    from ..core.frames import Pose, quat_from_matrix
    from ..core.project import Project
    from ..core.robot.model import get_model
    from .checks_ik import model_from_mjcf
    posts = ''.join(f'<body name="post{i}" pos="{r * np.cos(a)} {r * np.sin(a)} 0" euler="0 0 {a}">'
                    f'<geom type="box" size="{POST_HALF[0]} {POST_HALF[1]} {POST_HALF[2]}"/></body>'
                    for i, (a, r) in enumerate(POSTS))
    xml = ('<mujoco model="cable_check"><compiler angle="radian"/><worldbody><body name="a">'
           f'<geom type="box" size="{BASE_HALF[0]} {BASE_HALF[1]} {BASE_HALF[2]}" pos="0 0 {BASE_POS[2]}"/>' + posts +
           f'<body name="b"><joint name="hinge" type="hinge" axis="0 0 1" range="{-RANGE} {RANGE}"/>'
           f'<geom type="box" size="{ARM_HALF[0]} {ARM_HALF[1]} {ARM_HALF[2]}" pos="{ARM_POS[0]} 0 0"/></body>'
           '</body></worldbody></mujoco>')
    _m, robot, sources = model_from_mjcf(xml, folder, 'cable_check.xml')
    Rb = _rot([0.3, -0.2, 1.0], 0.7)
    tb = np.array([0.2, -0.1, 0.05])
    robot.base = Pose(tb, quat_from_matrix(Rb))
    p = Project('Cable check')
    p.robots, p.sources = [robot], dict(sources)
    return p, robot, get_model(p, robot.uid), Rb, tb


def _att(robot, link, point, laid=None) -> dict:
    return {'robot_uid': robot.uid, 'link': link, 'point': [float(v) for v in point], 'length_to_next_m': laid}


def _sweep(model, a, b, wrap_radius, n):
    from ..core import cables as CB
    from ..core.studies.cables import _Run
    run = _Run(model, a, b, wrap_radius)
    Q = np.linspace(-RANGE, RANGE, n)[:, None]
    T, anc, ax = CB.posed(model, Q)
    return run, Q[:, 0], run.evaluate(T, anc, ax, Q)


def _belt(da, db, dz, theta, rho):
    """The belt-over-a-pulley length written out again (independent of core/cables.py)."""
    out = []
    for th in np.abs(theta):
        aa, ab = np.arccos(rho / da), np.arccos(rho / db)
        if th > aa + ab:
            lp = np.sqrt(da * da - rho * rho) + np.sqrt(db * db - rho * rho) + rho * (th - aa - ab)
        else:
            lp = np.sqrt(da * da + db * db - 2 * da * db * np.cos(th))
        out.append(np.hypot(lp, dz))
    return np.array(out)


@check('cables.routed_arc', 'cables', 'Routed cable length over a revolute joint',
       'largest difference between the study\'s routed length (swept over the hinge\'s range, robot base moved and '
       'rotated) and the closed form, in three cases: the circle arc (equal radii, dz = 0), the helix (dz = 40 mm) '
       'and unequal radii over a smaller wrap radius; the unwrapped angle must equal the geometric angle modulo 2π',
       'the circle arc r·|q − α₀|, the helix √((r·|q − α₀|)² + dz²) and the belt-over-a-pulley length, written out in '
       'the check', 1e-9, 'm', refs=('docarmo1976',))
def _routed_arc(ctx):
    _p, robot, model, _Rb, _tb = scripted(os.path.join(ctx.tmp(), 'cables.routed_arc'))
    n = 181 if ctx.quick else 721
    errs, angle_err = {}, 0.0
    ref_all, got_all = [], []
    pa = [R_SAME * np.cos(ALPHA0), R_SAME * np.sin(ALPHA0), 0.03]
    for name, dz in (('circle arc', 0.0), ('helix', 0.04)):
        run, q, ev = _sweep(model, _att(robot, 'a', pa), _att(robot, 'b', [R_SAME, 0.0, 0.03 + dz]), None, n)
        ref = np.hypot(R_SAME * np.abs(q - ALPHA0), dz)
        errs[name] = float(np.max(np.abs(ev['L'] - ref)))
        geo = ev['cyl']['angle']
        angle_err = max(angle_err, float(np.max(np.abs(np.angle(np.exp(1j * (ev['theta'] - geo)))))))
        ref_all.append(ref)
        got_all.append(ev['L'])
    da, db, rho, dz = 0.10, 0.07, 0.05, 0.02
    run, q, ev = _sweep(model, _att(robot, 'a', [da * np.cos(-0.3), da * np.sin(-0.3), 0.0]),
                        _att(robot, 'b', [db, 0.0, dz]), rho, n)
    ref = _belt(da, db, dz, q + 0.3, rho)
    errs['belt over a pulley'] = float(np.max(np.abs(ev['L'] - ref)))
    geo = ev['cyl']['angle']
    angle_err = max(angle_err, float(np.max(np.abs(np.angle(np.exp(1j * (ev['theta'] - geo)))))))
    ref_all.append(ref)
    got_all.append(ev['L'])
    err = max(errs.values())
    return Outcome(err, {'samples_per_case': n, 'max_error_m': errs, 'angle_error_rad': angle_err,
                         'wraps_beyond_half_turn': bool(np.any(np.abs(q - ALPHA0) > np.pi))},
                   ok=angle_err < 1e-9,
                   figure=figure('Routed length over the hinge range', 'closed form (m)', 'DexVantage (m)',
                                 [series('circle arc', ref_all[0], got_all[0], 'points'),
                                  series('helix', ref_all[1], got_all[1], 'points'),
                                  series('belt', ref_all[2], got_all[2], 'points')], diag=True),
                   note='Both sides compute the same closed form in float64 from link poses MuJoCo gives (with the '
                        'base pose applied), so they agree to rounding, about 1e-16 m; 1e-9 m leaves room for that '
                        'while a wrong side sign, a wrong home angle or a length reduced to a half turn shows as '
                        'centimetres.')


@check('cables.helix_bend', 'cables', 'Bend radius of a taut run on the joint cylinder',
       'largest relative difference between the helix\'s radius of curvature and, for a taut run wrapped on the '
       'cylinder, both the study\'s bend radius estimate and the drawn 49-point polyline\'s (the smallest circle '
       'through three consecutive points), over the hinge\'s range where the run wraps more than 0.5 rad',
       'the helix\'s curvature a / c² (radius c² / a, a = r, c² = r² + b², b = dz / θ) [do Carmo 1976]', 5e-3,
       'relative', refs=('docarmo1976',))
def _helix_bend(ctx):
    from ..core import cables as CB
    _p, robot, model, _Rb, _tb = scripted(os.path.join(ctx.tmp(), 'cables.helix_bend'))
    pa = [R_SAME * np.cos(ALPHA0), R_SAME * np.sin(ALPHA0), 0.03]
    worst, rows = 0.0, []
    for dz in (0.0, 0.04, 0.10):
        run, q, ev = _sweep(model, _att(robot, 'a', pa), _att(robot, 'b', [R_SAME, 0.0, 0.03 + dz]), None, 121)
        sel = np.abs(q - ALPHA0) > 0.5
        P = run.route(ev, 49)[sel]
        got = CB.bend_radius(P)                                    # the drawn polyline
        est = CB.bend_estimate(ev['bend_route'], ev['L'], ev['L'])[sel]      # the study's figure, taut
        th = np.abs(q[sel] - ALPHA0)
        b = dz / th
        ref = (R_SAME ** 2 + b ** 2) / R_SAME
        rel = np.maximum(np.abs(got - ref), np.abs(est - ref)) / ref
        worst = max(worst, float(rel.max()))
        rows.append((dz, ref, got))
    return Outcome(worst, {'cases_dz_m': [0.0, 0.04, 0.10], 'polyline_points': 49},
                   figure=figure('Bend radius of a taut wrapped run', 'helix radius of curvature (m)',
                                 'DexVantage estimate (m)',
                                 [series(f'dz = {dz * 1000:.0f} mm', ref, got, 'points') for dz, ref, got in rows],
                                 diag=True),
                   note='The estimate is the circle through three consecutive polyline points; on a helix that circle '
                        'differs from the osculating one by a share of order (Δφ)² of the angle step Δφ between '
                        'points (here up to 0.1 rad), a few parts in ten thousand; 0.5 % bounds that.')


@check('cables.loop_arc', 'cables', 'Slack loop on a straight run against the circular arc',
       'largest relative difference between the radius of the circular arc of length s over c and both the study\'s '
       'slack-loop radius estimate and the drawn 49-point loop polyline\'s (laid length s over a straight run of '
       'chord c), for 200 random (c, s) with s / c from 1.02 to 6',
       'R = s / (2β), sin β / β = c / s, β found with scipy.optimize.brentq', 5e-3, 'relative',
       refs=('scipy_brentq',))
def _loop_arc(ctx):
    from scipy.optimize import brentq

    from ..core import cables as CB
    rng = ctx.rng('cables.loop_arc')
    n = 60 if ctx.quick else 200
    c = rng.uniform(0.02, 0.5, n)
    s = c * rng.uniform(1.02, 6.0, n)
    A = rng.normal(0, 0.3, (n, 3))
    d = rng.normal(0, 1, (n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    B = A + d * c[:, None]
    P = CB.chord_polyline(A, B, 49)
    loop = CB.bulge(P, s, CB.outward(P))
    got = CB.bend_radius(loop)
    ref = np.array([si / (2 * brentq(lambda b, r=ci / si: np.sin(b) / b - r, 1e-12, np.pi)) for ci, si in zip(c, s)])
    rel = np.abs(got - ref) / ref
    lens = np.abs(CB.poly_length(loop) - s) / s
    est = CB.bend_estimate(np.full(n, np.inf), c, s)               # the study's figure
    rel_est = np.abs(est - ref) / ref
    rel = np.maximum(rel, rel_est)
    return Outcome(float(rel.max()), {'cases': n, 'median_rel': float(np.median(rel)),
                                      'estimate_rel_max': float(rel_est.max()),
                                      'loop_length_rel_error_max': float(lens.max())},
                   ok=float(lens.max()) < 1e-6,
                   figure=figure('Slack-loop bend radius', 'circular arc (m)', 'DexVantage (m)',
                                 [series('loops', ref, got, 'points')], diag=True),
                   note='The loop\'s points lie on the circle, so the three-point circle is the arc itself; what '
                        'remains is the polyline being slightly shorter than the arc it samples (the bisection '
                        'matches the polyline\'s length to s), well under 0.5 % at 49 points.')


def _sd_box(p, half) -> np.ndarray:
    d = np.abs(p) - np.asarray(half)
    return np.linalg.norm(np.maximum(d, 0.0), axis=-1) + np.minimum(d.max(axis=-1), 0.0)


def _brute_sd(X, q, Rb, tb) -> np.ndarray:
    """(N,) the smallest signed distance of world points X (N, 3) to the scripted robot's boxes at hinge angle q,
    every box placed in closed form (base pose, then the post's own rotation or Rz(q))."""
    L = (X - tb) @ Rb                                   # into the base frame
    out = [_sd_box(L - BASE_POS, BASE_HALF)]
    for a, r in POSTS:
        c = np.array([r * np.cos(a), r * np.sin(a), 0.0])
        out.append(_sd_box((L - c) @ _rz(a), POST_HALF))
    out.append(_sd_box(L @ _rz(q) - ARM_POS, ARM_HALF))
    return np.min(out, axis=0)


def _resample(P, step) -> np.ndarray:
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    t = np.arange(0.0, cum[-1], step)
    return np.stack([np.interp(t, cum, P[:, k]) for k in range(3)], axis=1)


@check('cables.snag_bruteforce', 'cables', 'Snag detection against brute-force sampling',
       'poses (of the hinge\'s trajectory over its range, the robot base moved and rotated, a slack loop laid with '
       'room to swing into two posts) where the study\'s snag flag differs from brute force, grazing poses (the '
       'deepest point within 1 mm of a surface) left out; the share of time snagged at the study\'s 2° sampling must '
       'be within two steps of brute force at 0.2°',
       'the loop resampled every 0.5 mm, each point tested against every box in its own frame in closed form', 0,
       'poses', refs=('moller1997', 'amanatides_woo1987'), cost='study')
def _snag_bruteforce(ctx):
    from ..core import cables as CB
    from ..core.studies.cables import _Run, trajectory_poses
    _p, robot, model, Rb, tb = scripted(os.path.join(ctx.tmp(), 'cables.snag_bruteforce'))
    seat, s_laid, step = 0.02, 0.30, 0.0005
    run = _Run(model, _att(robot, 'a', [0.08, 0.0, 0.03]), _att(robot, 'b', [0.12, 0.0, 0.03]), None)
    traj = {'name': 'sweep', 'robot_uid': robot.uid, 'joints': ['hinge'], 't': [0.0, 5.6],
            'q': [[-RANGE], [RANGE]]}

    def flags(max_step_deg):
        Q, t, _note = trajectory_poses(model, traj, model.home, max_step_deg, 100000)
        T, anc, ax = CB.posed(model, Q)
        ev = run.evaluate(T, anc, ax, Q)
        P = run.loop(ev, s_laid, 49)
        hits = CB.snag_hits(model, T, P, ev['A'], ev['B'], seat)
        return Q[:, 0], t, P, ev, np.array([bool(h) for h in hits])

    q, t, P, ev, got = flags(2.0)
    want, depth = np.zeros(len(q), bool), np.zeros(len(q))
    for k in range(len(q)):
        X = _resample(P[k], step)
        far = (np.linalg.norm(X - ev['A'][k], axis=1) > seat) & (np.linalg.norm(X - ev['B'][k], axis=1) > seat)
        sd = _brute_sd(X[far], q[k], Rb, tb) if far.any() else np.array([np.inf])
        depth[k] = sd.min()
        want[k] = depth[k] < 0
    grazing = np.abs(depth) < 1e-3
    bad = int(np.sum((got != want) & ~grazing))
    # the share of the trajectory snagged: the study's sampling against brute force ten times finer
    qf, tf, Pf, evf, _g = flags(0.2)
    fine = np.zeros(len(qf), bool)
    for k in range(len(qf)):
        X = _resample(Pf[k], step)
        far = (np.linalg.norm(X - evf['A'][k], axis=1) > seat) & (np.linalg.norm(X - evf['B'][k], axis=1) > seat)
        fine[k] = far.any() and _brute_sd(X[far], qf[k], Rb, tb).min() < 0
    share_study, share_fine = float(got.mean()), float(fine.mean())
    bound = 2.0 / max(len(q) - 1, 1)
    return Outcome(float(bad), {'poses': int(len(q)), 'snagged_study': int(got.sum()),
                                'snagged_bruteforce': int(want.sum()), 'grazing_left_out': int(grazing.sum()),
                                'share_snagged_study': share_study, 'share_snagged_fine_bruteforce': share_fine,
                                'share_bound': bound, 'laid_length_m': s_laid, 'seat_m': seat},
                   ok=abs(share_study - share_fine) <= bound,
                   figure=figure('Deepest point of the loop in the robot\'s boxes', 'hinge angle (rad)',
                                 'signed distance (m)', [series('brute force', q, depth),
                                                         series('study flags (1 = snag)', q, got.astype(float) * 0.01,
                                                                'points')], hline=0.0, hline_label='surface'),
                   note='Both sides test the same loop polylines; the study by ray casting segments against triangles '
                        'and a parity test, brute force by points 0.5 mm apart in closed-form boxes. A loop that only '
                        'grazes a box (less than 1 mm deep or clear) can fall between the brute-force points, so '
                        'those poses are not counted; every other pose must agree. Between the sampled poses the '
                        'swept volume is not tested: the share-of-time comparison shows how much that sampling '
                        'misses.')
