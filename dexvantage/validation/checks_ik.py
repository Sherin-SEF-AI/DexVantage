"""Checks of core/robot/ik.py, manipulability.py and collision.py on scripted MJCF robots written here.

- IK round trips: random joint vectors (collision-free, inside the limits) → forward kinematics → IK → the solution's
  pose recomputed with a product of exponentials written out again in this module (no MuJoCo) and compared with the
  target. A 6-joint arm solves the full pose; a 5-joint arm solves position-only.
- Manipulability of a planar 2R arm against the closed form |l₁ l₂ sin q₂| [Yoshikawa 1985].
- Collision distances against the analytic distance of two spheres (|c₁ − c₂| − r₁ − r₂) and two capsules (the
  closest points of two segments [Ericson 2005, §5.1.9], minus the radii), with overlap (negative distances) included.

The scripted arms are test fixtures with round link lengths, not models of any real robot.
"""
from __future__ import annotations

import os

import numpy as np

from .suite import Outcome, check, figure, series

# ---------------------------------------------------------------------------------------------- scripted robots
# (name, body offset from the parent (m), hinge axis, limits (rad)) root to tip; bodies keep the parent's orientation
ARM6 = (('link1', (0, 0, 0.10), (0, 0, 1), (-3.0, 3.0)),
        ('link2', (0, 0, 0.10), (0, 1, 0), (-2.0, 2.0)),
        ('link3', (0, 0, 0.40), (0, 1, 0), (-2.5, 2.5)),
        ('link4', (0, 0, 0.35), (0, 0, 1), (-3.0, 3.0)),
        ('link5', (0, 0, 0.10), (0, 1, 0), (-2.0, 2.0)),
        ('link6', (0, 0, 0.08), (0, 0, 1), (-3.0, 3.0)))
ARM5 = tuple(j for j in ARM6 if j[0] != 'link4')
TOOL = (0.0, 0.0, 0.06)                    # the tool frame's offset from the last link (m)
L1_2R, L2_2R = 0.45, 0.30                  # the planar 2R arm's link lengths (m)


def arm_mjcf(chain=ARM6, tool=TOOL, name='scripted_arm') -> str:
    """An MJCF serial arm: capsules along each link (colliding), a body 'tool' at the end."""
    xml = [f'<mujoco model="{name}"><compiler angle="radian"/><worldbody>',
           '<body name="base"><geom type="cylinder" size="0.06 0.02" pos="0 0 0.02"/>']
    nxt = [j[1] for j in chain[1:]] + [tool]
    for (nm, pos, axis, (lo, hi)), off in zip(chain, nxt):
        xml.append(f'<body name="{nm}" pos="{pos[0]} {pos[1]} {pos[2]}">'
                   f'<joint name="{nm}_joint" type="hinge" axis="{axis[0]} {axis[1]} {axis[2]}" range="{lo} {hi}"/>')
        if np.linalg.norm(off) > 0.03:
            xml.append(f'<geom type="capsule" size="0.025" fromto="0 0 0.01 {off[0]} {off[1]} {off[2] * 0.85}"/>')
    xml.append(f'<body name="tool" pos="{tool[0]} {tool[1]} {tool[2]}"/>')
    xml.append('</body>' * len(chain) + '</body></worldbody></mujoco>')
    return ''.join(xml)


def planar_2r_mjcf(l1=L1_2R, l2=L2_2R) -> str:
    return (f'<mujoco model="planar_2r"><compiler angle="radian"/><worldbody><body name="base">'
            f'<body name="link1"><joint name="q1" axis="0 0 1" range="-3.1 3.1"/>'
            f'<geom type="capsule" size="0.01" fromto="0 0 0 {l1} 0 0" contype="0" conaffinity="0"/>'
            f'<body name="link2" pos="{l1} 0 0"><joint name="q2" axis="0 0 1" range="-3.1 3.1"/>'
            f'<geom type="capsule" size="0.01" fromto="0 0 0 {l2} 0 0" contype="0" conaffinity="0"/>'
            f'<body name="tip" pos="{l2} 0 0"/></body></body></body></worldbody></mujoco>')


def xyz_pair_mjcf(geom_a: str, geom_b: str) -> str:
    """A static base carrying geom_a and, three slide joints (x, y, z) further, a body carrying geom_b: the two are
    not adjacent, so their distance is checked."""
    return ('<mujoco model="xyz_pair"><worldbody><body name="base">' + geom_a +
            '<body name="sx"><joint name="x" type="slide" axis="1 0 0" range="-1 1"/>'
            '<body name="sy"><joint name="y" type="slide" axis="0 1 0" range="-1 1"/>'
            '<body name="sz"><joint name="z" type="slide" axis="0 0 1" range="-1 1"/>' + geom_b +
            '</body></body></body></body></worldbody></mujoco>')


def model_from_mjcf(text: str, folder: str, name: str = 'robot.xml'):
    """(RobotModel, Robot, sources) of an MJCF text written to folder/name and imported as the app imports it."""
    from ..core.robot import RobotModel, import_robot
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    robot, sources = import_robot(path)
    return RobotModel(robot, sources), robot, sources


def _rot(axis, a) -> np.ndarray:
    """Rodrigues' formula, written out."""
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def poe_fk(chain, q, tool=TOOL) -> np.ndarray:
    """The tool's 4 × 4 pose: Π Trans(offset_i) · Rot(axis_i, q_i), then Trans(tool) (independent of MuJoCo)."""
    T = np.eye(4)
    for (_n, pos, axis, _lim), a in zip(chain, q):
        S = np.eye(4)
        S[:3, :3] = _rot(axis, a)
        S[:3, 3] = pos
        T = T @ S
    E = np.eye(4)
    E[:3, 3] = tool
    return T @ E


def segment_distance(p1, q1, p2, q2) -> float:
    """The distance between segments p1q1 and p2q2 (Ericson 2005, Real-Time Collision Detection, §5.1.9)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    eps = 1e-15
    if a <= eps and e <= eps:
        return float(np.linalg.norm(r))
    if a <= eps:
        s, t = 0.0, np.clip(f / e, 0, 1)
    else:
        c = d1 @ r
        if e <= eps:
            t, s = 0.0, np.clip(-c / a, 0, 1)
        else:
            b = d1 @ d2
            den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0, 1) if den > eps else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, np.clip(-c / a, 0, 1)
            elif t > 1:
                t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def _random_free_q(model, rng, n, inset=0.1):
    """n joint vectors drawn uniformly inside the limits (inset rad from each), each collision-free by at least 5 mm."""
    from ..core.robot.collision import self_collision
    out = []
    while len(out) < n:
        q = rng.uniform(model.lower + inset, model.upper - inset)
        if not self_collision(model, q, margin=0.005).in_collision:
            out.append(q)
    return np.array(out)


def _round_trip(ctx, cid, chain, n):
    from ..core.robot import ik
    model, _r, _s = model_from_mjcf(arm_mjcf(chain), os.path.join(ctx.tmp(), cid))
    rng = ctx.rng(cid)
    Q = _random_free_q(model, rng, n)
    T = np.stack([poe_fk(chain, q) for q in Q])
    mj = np.stack([model.link_pose(q, 'tool') for q in Q])
    fk_err = float(np.abs(T - mj).max())
    res = ik.solve_batch(model, 'tool', T, rng=ctx.rng(cid, 1))
    back = np.stack([poe_fk(chain, q) for q in res.q])
    pos = np.linalg.norm(back[:, :3, 3] - T[:, :3, 3], axis=1)
    rel = np.einsum('nji,njk->nik', back[:, :3, :3], T[:, :3, :3])
    ang = np.arccos(np.clip(0.5 * (np.trace(rel, axis1=1, axis2=2) - 1), -1, 1))
    return model, res, pos, ang, fk_err


@check('kinematics.ik_round_trip', 'kinematics', 'IK round trip, 6-joint arm (full pose)',
       'largest tool position error of the IK solution, recomputed independently, over random reachable targets; '
       'every target must be solved and its rotation error be within 1 mrad',
       'the target itself: the pose of a random collision-free joint vector, by a product of exponentials written '
       'out in the check (Rodrigues\' formula; no MuJoCo), which also recomputes the solution\'s pose',
       1e-4, 'm', refs=('wampler1986', 'sugihara2011', 'liegeois1977', 'lynch_park2017'))
def _ik_round_trip(ctx):
    n = 25 if ctx.quick else 150
    model, res, pos, ang, fk_err = _round_trip(ctx, 'kinematics.ik_round_trip', ARM6, n)
    solved = int(res.ok.sum())
    err = float(pos.max())
    return Outcome(err, {'targets': n, 'solved': solved, 'reasons': res.counts(), 'median_m': float(np.median(pos)),
                         'max_rotation_rad': float(ang.max()), 'median_iterations': float(np.median(res.iterations)),
                         'mujoco_vs_poe_fk': fk_err},
                   figure=figure('Round-trip position error', 'target', 'error (m)',
                                 [series('position error', np.arange(n), np.maximum(pos, 1e-12), 'points')],
                                 logy=True, hline=1e-4, hline_label='tolerance (the solver\'s 0.1 mm)'),
                   ok=solved == n and float(ang.max()) <= 1e-3 and fk_err < 1e-12,
                   note='The tolerance is the solver\'s convergence tolerance (0.1 mm, 1 mrad): a solution reported '
                        'as converged must meet it when its pose is computed by an independent route. The targets '
                        'come from joint vectors inside the limits, so all are reachable and must be solved; MuJoCo\'s '
                        'forward kinematics must also agree with the product of exponentials to 1e-12.')


@check('kinematics.ik_position_only', 'kinematics', 'IK round trip, 5-joint arm (position only)',
       'largest tool position error of the position-only IK solution, recomputed independently, over random '
       'reachable targets; every target must be solved',
       'the target position, from a product of exponentials written out in the check (no MuJoCo)', 1e-4, 'm',
       refs=('nakamura1986', 'lynch_park2017'))
def _ik_position_only(ctx):
    n = 25 if ctx.quick else 150
    model, res, pos, ang, _fk = _round_trip(ctx, 'kinematics.ik_position_only', ARM5, n)
    solved = int(res.ok.sum())
    return Outcome(float(pos.max()), {'targets': n, 'solved': solved, 'reasons': res.counts(), 'mode': res.mode,
                                      'orientation_reached': int(res.orientation_ok.sum()),
                                      'median_rotation_deg': float(np.degrees(np.median(ang)))},
                   ok=solved == n and res.mode == 'position',
                   note='A 5-joint arm cannot set all six pose coordinates; the solver reaches the position and gets '
                        'the orientation as close as it can (as the target came from a real pose of the arm, the '
                        'orientation is often reached too, which the detail counts).')


@check('kinematics.manipulability_2r', 'kinematics', 'Manipulability of a planar 2R arm',
       'largest difference between √det(J Jᵀ) of the tip\'s planar (x, y) Jacobian and the closed form, over random '
       'joint angles including the stretched and folded singularities',
       'the closed form w = |l₁ l₂ sin q₂| (Yoshikawa 1985), written out in the check', 1e-12, 'm²',
       refs=('yoshikawa1985',))
def _manip_2r(ctx):
    from ..core.robot import manipulability as M
    model, _r, _s = model_from_mjcf(planar_2r_mjcf(), os.path.join(ctx.tmp(), 'planar_2r'))
    rng = ctx.rng('kinematics.manipulability_2r')
    n = 100 if ctx.quick else 1000
    Q = np.c_[rng.uniform(-3, 3, n), rng.uniform(-3, 3, n)]
    Q[:4, 1] = [0.0, np.pi / 2, -np.pi / 2, 3.1]
    J = np.stack([model.jacobian(q, 'tip')[:2] for q in Q])
    w = M.yoshikawa(J)
    ref = np.abs(L1_2R * L2_2R * np.sin(Q[:, 1]))
    err = np.abs(w - ref)
    # the condition number of the closed-form Jacobian, from the eigenvalues of J Jᵀ written out
    c1, s1 = np.cos(Q[:, 0]), np.sin(Q[:, 0])
    c12, s12 = np.cos(Q.sum(1)), np.sin(Q.sum(1))
    Jc = np.stack([np.stack([-L1_2R * s1 - L2_2R * s12, -L2_2R * s12], 1),
                   np.stack([L1_2R * c1 + L2_2R * c12, L2_2R * c12], 1)], 1)
    A = Jc @ np.transpose(Jc, (0, 2, 1))
    tr, det = A[:, 0, 0] + A[:, 1, 1], A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
    disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    lmax, lmin = tr / 2 + disc, tr / 2 - disc
    good = lmin > 1e-6
    kref = np.sqrt(lmax[good] / lmin[good])
    kerr = float(np.max(np.abs(M.condition(J[good]) - kref) / kref)) if good.any() else 0.0
    return Outcome(float(err.max()), {'poses': n, 'condition_rel_error': kerr, 'l1_m': L1_2R, 'l2_m': L2_2R},
                   ok=kerr < 1e-6,
                   figure=figure('Manipulability against q₂', 'q₂ (rad)', 'w (m²)',
                                 [series('closed form', Q[:, 1], ref, 'points'),
                                  series('DexVantage', Q[:, 1], w, 'points')]),
                   note='MuJoCo\'s Jacobian is analytic, so agreement is to rounding (1e-12 m² for w of order '
                        '0.1 m²). The condition number is checked as well (relative error < 1e-6, away from the '
                        'singularity, where it is infinite).')


def _distance_check(ctx, cid, geom_a, geom_b, ref_fn, n):
    from ..core.robot import collision as C
    model, _r, _s = model_from_mjcf(xyz_pair_mjcf(geom_a, geom_b), os.path.join(ctx.tmp(), cid))
    rng = ctx.rng(cid)
    Q = rng.uniform(-0.3, 0.3, (n, 3))
    hit, d, _p = C.self_collision_batch(model, Q, margin=0.0, distmax=1.0)
    ref = np.array([ref_fn(q) for q in Q])
    err = np.abs(d - ref)
    wrong = int(np.count_nonzero(hit != (ref < 0)))
    return err, ref, d, wrong


@check('kinematics.collision_spheres', 'kinematics', 'Self-collision distance, two spheres',
       'largest difference between the signed distance of two spheres on non-adjacent links and the analytic one, '
       'over random placements (overlapping ones included); the in-collision flag must agree with the sign',
       'the analytic distance |c₁ − c₂| − r₁ − r₂, written out in the check', 1e-9, 'm',
       refs=('mujoco_geomdistance',))
def _spheres(ctx):
    r1, r2, c1, c2off = 0.10, 0.07, np.array([0.0, 0.0, 0.0]), np.array([0.2, 0.0, 0.1])
    a = f'<geom type="sphere" size="{r1}" pos="0 0 0"/>'
    b = f'<geom type="sphere" size="{r2}" pos="{c2off[0]} {c2off[1]} {c2off[2]}"/>'
    n = 100 if ctx.quick else 1000
    err, ref, d, wrong = _distance_check(ctx, 'kinematics.collision_spheres', a, b,
                                         lambda q: np.linalg.norm(c2off + q - c1) - r1 - r2, n)
    return Outcome(float(err.max()), {'placements': n, 'overlapping': int((ref < 0).sum()), 'flag_mismatches': wrong},
                   ok=wrong == 0,
                   figure=figure('Sphere distance', 'analytic (m)', 'DexVantage (m)',
                                 [series('pairs', ref, d, 'points')], diag=True),
                   note='Sphere–sphere distance is closed form in MuJoCo as here, so agreement is to rounding.')


@check('kinematics.collision_capsules', 'kinematics', 'Self-collision distance, two capsules',
       'largest difference between the signed distance of two capsules on non-adjacent links and the analytic one, '
       'over random placements (overlapping ones included); the in-collision flag must agree with the sign',
       'the closest points of the two axis segments (Ericson 2005, §5.1.9) written out in the check, minus the '
       'radii', 1e-9, 'm', refs=('mujoco_geomdistance', 'ericson2005'))
def _capsules(ctx):
    ra, rb = 0.04, 0.03
    pa, qa = np.array([-0.15, 0.0, 0.0]), np.array([0.15, 0.0, 0.0])
    pb0, qb0 = np.array([0.05, -0.12, 0.08]), np.array([0.10, 0.12, 0.02])
    a = f'<geom type="capsule" size="{ra}" fromto="{" ".join(map(str, pa))} {" ".join(map(str, qa))}"/>'
    b = f'<geom type="capsule" size="{rb}" fromto="{" ".join(map(str, pb0))} {" ".join(map(str, qb0))}"/>'
    n = 100 if ctx.quick else 1000
    err, ref, d, wrong = _distance_check(ctx, 'kinematics.collision_capsules', a, b,
                                         lambda q: segment_distance(pa, qa, pb0 + q, qb0 + q) - ra - rb, n)
    return Outcome(float(err.max()), {'placements': n, 'overlapping': int((ref < 0).sum()), 'flag_mismatches': wrong,
                                      'median_m': float(np.median(err))},
                   ok=wrong == 0,
                   figure=figure('Capsule distance', 'analytic (m)', 'DexVantage (m)',
                                 [series('pairs', ref, d, 'points')], diag=True),
                   note='MuJoCo\'s capsule–capsule routine is a closed form as well (the closest points of the two '
                        'axis segments), so agreement is to rounding; 1e-9 m, as for the spheres, flags any wrong '
                        'branch of the segment-parameter clamping, which shows as millimetres.')
