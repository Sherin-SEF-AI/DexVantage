"""Checks of core/studies/teleop.py: the follower's dexterity measures and the mapping of a leader's comfortable
workspace into the follower's frame.

- Dexterity of a planar 2R arm (scripted here, round link lengths, not a model of any robot) through the study's own
  path (Arm: the tool point's Jacobian over the chain, the singular values) against the closed forms written out
  again: w = |l₁ l₂ sin q₂| [Yoshikawa 1985] and 1/κ from the eigenvalues of J Jᵀ of the analytic Jacobian, at random
  joint angles and at the IK solutions the study finds for random reachable points.
- Dexterity of the bundled robots against a finite-difference Jacobian of the tool point computed with the
  independent product-of-exponentials chain (core/robot/kinematics.py: no MuJoCo): w and 1/κ.
- The leader-workspace mapping (a leader arm of the follower's own model, joint to joint, the comfortable 0.8 of each
  range) against a brute-force sample: every voxel the study calls comfortable must hold a joint vector inside the
  comfortable box whose tool point the PoE chain puts on the voxel centre, and every voxel centre that a uniform
  sample of the comfortable box (posed by the PoE chain) passes within 5 mm of must be called comfortable.
- The cartesian mapping of a device volume (box and cylinder, a rotated offset, a scaled mapping, a moved and turned
  follower base): points drawn in the device frame and mapped into the world are classified by the study's inverse
  mapping, and the mapped volume measured by Monte Carlo is compared with scale³ × the device volume.
"""
from __future__ import annotations

import os

import numpy as np

from .checks_ik import L1_2R, L2_2R, model_from_mjcf, planar_2r_mjcf
from .suite import Outcome, check

FD_STEP = 1e-6                     # rad: central differences, truncation error ~ step² ≈ 1e-12 of the entries
NEAR_M = 0.005                     # m: a brute-force sample this close to a voxel centre marks it as reached


def _study_settings(**kw) -> dict:
    from ..core.studies import teleop as T
    st = dict(T.DEFAULTS)
    st.update(kw)
    return st


def _planar_arm(ctx):
    from ..core.project import Project
    from ..core.studies import teleop as T
    _m, robot, sources = model_from_mjcf(planar_2r_mjcf(), os.path.join(ctx.tmp(), 'teleop_2r'))
    robot.gripper_frame = 'tip'
    robot.primary = True
    p = Project('2R')
    p.robots, p.sources = [robot], dict(sources)
    return p, T.Arm(p, robot.uid, _study_settings())


def _closed_2r(Q):
    c1, s1 = np.cos(Q[:, 0]), np.sin(Q[:, 0])
    c12, s12 = np.cos(Q.sum(1)), np.sin(Q.sum(1))
    Jc = np.stack([np.stack([-L1_2R * s1 - L2_2R * s12, -L2_2R * s12], 1),
                   np.stack([L1_2R * c1 + L2_2R * c12, L2_2R * c12], 1)], 1)
    A = Jc @ np.transpose(Jc, (0, 2, 1))
    tr, det = A[:, 0, 0] + A[:, 1, 1], A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
    disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    lmax, lmin = tr / 2 + disc, np.maximum(tr / 2 - disc, 0)
    return np.abs(L1_2R * L2_2R * np.sin(Q[:, 1])), np.sqrt(lmin / lmax)


@check('teleop.manipulability_2r', 'teleop', 'Teleop dexterity of a planar 2R arm',
       'largest difference between the study\'s w (the product of the tool Jacobian\'s singular values) and the '
       'closed form, at random joint angles (singular poses included) and at the study\'s IK solutions for random '
       'reachable points; 1/κ must agree within 1e-9',
       'w = |l₁ l₂ sin q₂| (Yoshikawa 1985) and 1/κ = √(λ_min / λ_max) of J Jᵀ of the analytic 2R Jacobian, written '
       'out in the check', 1e-12, 'm²', refs=('yoshikawa1985',))
def _manip_2r(ctx):
    _p, arm = _planar_arm(ctx)
    rng = ctx.rng('teleop.manipulability_2r')
    n = 100 if ctx.quick else 1000
    Q = np.c_[rng.uniform(-3, 3, n), rng.uniform(-3, 3, n)]
    Q[:3, 1] = [0.0, np.pi / 2, 3.1]
    # points the arm reaches (an annulus in its plane), solved by the study's IK
    r = rng.uniform(abs(L1_2R - L2_2R) + 0.05, L1_2R + L2_2R - 0.05, n // 4)
    a = rng.uniform(-np.pi, np.pi, n // 4)
    P = np.c_[r * np.cos(a), r * np.sin(a), np.zeros_like(r)]
    b = arm.solve(P)
    Qs = b.q[b.ok]
    Qa = np.r_[Q, Qs]
    d = arm.dexterity(Qa)
    w_ref, inv_ref = _closed_2r(Qa)
    err = float(np.max(np.abs(d['w'] - w_ref)))
    inv_err = float(np.max(np.abs(d['inverse_condition'] - inv_ref)))
    return Outcome(err, {'joint_vectors': int(len(Q)), 'ik_points': int(len(P)), 'ik_solved': int(b.ok.sum()),
                         'inverse_condition_error': inv_err, 'w_max_m2': float(w_ref.max())},
                   ok=inv_err <= 1e-9 and bool(b.ok.all()),
                   note='' if b.ok.all() else f'{int((~b.ok).sum())} reachable points were not solved by the IK.')


@check('teleop.jacobian_fd', 'teleop', 'Teleop dexterity against a finite-difference Jacobian',
       'largest relative difference between the study\'s w and 1/κ (from MuJoCo\'s mj_jac at the tool point) and '
       'those of a central finite-difference Jacobian of the tool point, over random joint vectors of each robot',
       'the tool point posed by the independent product-of-exponentials chain (core/robot/kinematics.py, no MuJoCo), '
       'differenced with a step of 1e-6 rad', 1e-6, 'relative', refs=('yoshikawa1985', 'lynch_park2017'))
def _jac_fd(ctx):
    from ..core.robot import manipulability as M
    from ..core.robot.kinematics import PoEChain
    from ..core.studies import teleop as T
    from .checks_kinematics import random_q
    p = ctx.project
    n = 10 if ctx.quick else 60
    worst, per = 0.0, {}
    for k, r in enumerate(p.robots):
        arm = T.Arm(p, r.uid, _study_settings())
        chain = PoEChain(r, p.sources)
        names = arm.model.joint_names
        ci = np.flatnonzero(arm.chain)
        Q = random_q(arm.model, ctx.rng('teleop.jacobian_fd', k), n)
        d = arm.dexterity(Q)
        e = 0.0
        for i, q in enumerate(Q):
            def tool(qq):
                T_ = chain.fk(dict(zip(names, qq)))[arm.link]
                return T_[:3, :3] @ arm.tcp + T_[:3, 3]
            J = np.zeros((3, len(ci)))
            for c, j in enumerate(ci):
                dq = np.zeros(len(q))
                dq[j] = FD_STEP
                J[:, c] = (tool(q + dq) - tool(q - dq)) / (2 * FD_STEP)
            s = M.singular_values(J)
            w, inv = float(np.prod(s)), float(s[-1] / s[0])
            e = max(e, abs(d['w'][i] - w) / max(abs(w), 1e-12), abs(d['inverse_condition'][i] - inv))
        per[r.name] = e
        worst = max(worst, e)
    return Outcome(worst, {'per_robot': per, 'joint_vectors_per_robot': n, 'step_rad': FD_STEP})


@check('teleop.leader_workspace', 'teleop', 'Leader workspace mapped into the follower: against a brute-force sample',
       'the share of voxel centres a brute-force sample of the leader\'s comfortable box reaches (within 5 mm) that '
       'the study does not call comfortable; every voxel it calls comfortable must also hold a joint vector inside '
       'the box whose tool point the PoE chain puts within 0.2 mm of the centre',
       'uniform joint vectors in the comfortable box (0.8 of each SO-100 joint\'s range), the tool point posed by the '
       'independent product-of-exponentials chain', 0.05, 'share', refs=('lerobot_so100', 'lynch_park2017'),
       cost='study')
def _leader_ws(ctx):
    from scipy.spatial import cKDTree
    from ..core.robot.kinematics import PoEChain
    from ..core.robot.samples import sample_project
    from ..core.studies import teleop as T
    p = sample_project('so100')
    r = p.robots[0]
    st = _study_settings(voxel_mm=25.0, search_voxels=1)
    dev = T.new_device(p, 'leader_arm')
    arm = T.Arm(p, r.uid, st)
    grid = T.Grid(p, st['voxel_mm'], st['search_voxels'])
    fe = T.evaluate_follower(arm, grid)
    ev = T.evaluate_device(p, dev, arm, fe, grid, st, [])
    lo, hi = T.comfort_box(arm.model, st['comfort_fraction'])
    chain = PoEChain(r, p.sources)
    names = arm.model.joint_names

    def tool(q):
        T_ = chain.fk(dict(zip(names, q)))[arm.link]
        return T_[:3, :3] @ arm.tcp + T_[:3, 3]
    comf = ev['comf']
    bad = 0
    for k in np.flatnonzero(comf):
        q = ev['q'][k]
        inside = bool(np.all(q[arm.chain] >= lo[arm.chain] - 1e-9) and np.all(q[arm.chain] <= hi[arm.chain] + 1e-9))
        if not inside or np.linalg.norm(tool(q) - grid.eval[k]) > 2e-4:
            bad += 1
    rng = ctx.rng('teleop.leader_workspace')
    n = 20_000 if ctx.quick else 120_000
    Q = np.tile(arm.model.home, (n, 1))
    Q[:, arm.chain] = rng.uniform(lo[arm.chain], hi[arm.chain], (n, int(arm.chain.sum())))
    # the brute-force samples land near the evaluation lattice only where it is: keep those inside its box first
    elo, ehi = grid.eval.min(axis=0) - 0.01, grid.eval.max(axis=0) + 0.01
    pts = []
    tree = cKDTree(grid.eval)
    for q in Q:
        x = tool(q)
        if np.all(x >= elo) and np.all(x <= ehi):
            pts.append(x)
    pts = np.array(pts).reshape(-1, 3)
    dist, idx = tree.query(pts, k=1) if len(pts) else (np.zeros(0), np.zeros(0, int))
    hit = np.unique(idx[dist <= NEAR_M])
    missed = int((~comf[hit]).sum()) if len(hit) else 0
    # of the voxels missed: how many a wider IK search (32 seeds) finds inside the box (a local-IK miss) and how many
    # it does not (the centre is outside the comfortable workspace, within 5 mm of it: a boundary voxel)
    lost = hit[~comf[hit]] if len(hit) else np.zeros(0, int)
    wide = 0
    if len(lost):
        wide_st = dict(st, seeds=32)
        b = T.Arm(p, r.uid, wide_st).solve(grid.eval[lost], T.proxy(arm.model, lo, hi))
        wide = int(b.ok.sum())
    err = missed / max(len(hit), 1)
    return Outcome(err, {'voxels': int(len(grid.eval)), 'comfortable': int(comf.sum()), 'samples': n,
                         'samples_near_the_grid': int(len(pts)), 'voxels_hit_by_samples': int(len(hit)),
                         'missed': missed, 'missed_found_with_32_seeds': wide,
                         'missed_on_the_boundary': missed - wide, 'comfortable_without_a_valid_pose': bad},
                   ok=bad == 0 and len(hit) >= 20,
                   note='The tolerance allows for voxel centres within 5 mm of the comfortable workspace\'s boundary '
                        'and for targets the local IK misses from its seeds.')


@check('teleop.cartesian_mapping', 'teleop', 'Device volume mapped into the follower\'s frame',
       'the Monte Carlo volume of the mapped region against scale³ × the device volume, in standard errors over 3 '
       '(pass ≤ 1); no point drawn in the device frame inside or outside its box or cylinder, mapped into the world, '
       'may be misclassified by the study\'s inverse mapping, and the round trip must hold to 1e-12 m',
       'the classification in the device frame where the points were drawn, and the analytic volume', 1.0,
       '3 standard errors',
       refs=('lynch_park2017',))
def _cart(ctx):
    from ..core.frames import Pose, quat_from_axis_angle
    from ..core.studies import teleop as T
    rng = ctx.rng('teleop.cartesian_mapping')
    n = 4000 if ctx.quick else 40000
    worst, per, bad = 0.0, {}, False
    for k, vol in enumerate(({'kind': 'box', 'center': [0.1, -0.05, 0.2], 'size': [0.3, 0.2, 0.15]},
                             {'kind': 'cylinder', 'center': [0.0, 0.1, -0.1], 'size': [0.12, 0.25]})):
        ax = rng.normal(size=3)
        off = Pose(rng.uniform(-0.3, 0.3, 3), quat_from_axis_angle(ax / np.linalg.norm(ax), rng.uniform(-3, 3)))
        ax2 = rng.normal(size=3)
        base = Pose(rng.uniform(-0.5, 0.5, 3), quat_from_axis_angle(ax2 / np.linalg.norm(ax2), rng.uniform(-3, 3)))
        scale = float(rng.uniform(0.4, 2.5))
        dev = T.clean_device({'kind': 'vr', 'follower_uid': 'x', 'volume': vol,
                              'mapping': {'kind': 'cartesian', 'scale': scale, 'offset': off.to_dict()}})
        B = base.matrix()
        c, s = np.asarray(vol['center']), np.asarray(vol['size'])
        ext = (s if vol['kind'] == 'box' else np.r_[2 * s[0], 2 * s[0], s[1]]) * 0.75
        Pd = c + rng.uniform(-1, 1, (n, 3)) * ext
        d = Pd - c
        truth = np.all(np.abs(d) <= s / 2, 1) if vol['kind'] == 'box' else \
            (np.hypot(d[:, 0], d[:, 1]) <= s[0]) & (np.abs(d[:, 2]) <= s[1] / 2)
        Pw = T.from_device(dev, B, Pd)
        got = T.inside_volume(dev['volume'], T.to_device(dev, B, Pw))
        wrong = int((got != truth).sum())
        # Monte Carlo volume in the world: a box round the mapped region, points classified by the study
        W = T.from_device(dev, B, c + np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * ext)
        lo, hi = W.min(0), W.max(0)
        X = rng.uniform(lo, hi, (n, 3))
        f = T.inside_volume(dev['volume'], T.to_device(dev, B, X)).mean()
        vbox = float(np.prod(hi - lo))
        est = f * vbox
        true = scale ** 3 * (np.prod(s) if vol['kind'] == 'box' else np.pi * s[0] ** 2 * s[1])
        se = np.sqrt(max(f * (1 - f), 1e-12) / n) * vbox
        z = abs(est - true) / se / 3.0
        rt = float(np.max(np.abs(T.from_device(dev, B, T.to_device(dev, B, X)) - X)))
        e = z
        bad = bad or wrong > 0 or rt >= 1e-12
        per[vol['kind']] = {'misclassified': wrong, 'volume_true_m3': float(true), 'volume_mc_m3': float(est),
                            'z_over_3': float(z), 'round_trip_m': rt, 'scale': scale}
        worst = max(worst, e)
    return Outcome(worst, {'per_volume': per, 'points': n}, ok=not bad)
