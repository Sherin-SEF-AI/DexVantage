"""Checks of the grasp self-occlusion study (core/studies/grasp_visibility.py).

- grasp.visibility_vs_raycast: the study's per-point visibility of the contact locations (the fingertip pads' points
  lifted 2 mm in front of the pads, and the object's grasp region) from both cameras of the UR5e sample, at IK
  solutions of sampled grasps on a box, against exact ray casting (core/raycast.visible_exact) of the full scene
  built again by core/scenegeom.sensor_occluders: no culling, no cube map, no q-bucket. Disagreements are sorted into
  those within two texels of an edge (a point whose exact visibility changes when it is moved by two texels' angle
  within its own surface's plane) and the rest.
- grasp.bootstrap_coverage: the percentile bootstrap interval of the Contact Visibility Score on synthetic binomial
  samples of known p (n = 100, 300, 1000 grasps; p = 0.3, 0.7, 0.9), its coverage of p over many seeded samples
  against the nominal 95 %, with Wilson's score interval computed alongside (its coverage and the endpoint gap).
- grasp.qbucket_vs_exact: the q-bucket cache: the visible fraction of each fingertip patch reported for a grasp
  whose q falls into a bucket already evaluated (the bucket's first q, its maps reused) against the fraction at the
  grasp's own q, maps drawn there; pairs of joint vectors drawn in one bucket (the default 0.1°) on the UR5e sample.
"""
from __future__ import annotations

import math

import numpy as np

from .suite import Outcome, check, figure, series


def _ur5e_with_box(ctx, cid: str, n: int):
    """The UR5e sample with a 50 mm box on the table in its task volume, a sampled grasp set and its IK."""
    from ..core import grasps as GR
    from ..core.frames import Pose
    from ..core.project import TaskObject
    from ..core.robot.samples import sample_project
    from ..core.studies import grasp_visibility as GV
    p = sample_project('ur5e')
    p.tasks.grasp_sets = []                     # the sample's own set left out: the check samples its own
    r = p.robots[0]
    v = p.workspace.volumes[0]
    p.workspace.objects.append(TaskObject(name='Box (check)', shape='box', size=(0.05, 0.05, 0.05),
                                          pose=Pose(np.array([v.center[0], v.center[1], 0.025]),
                                                    np.array([0.0, 0.0, 0.0, 1.0]))))
    seed = int(ctx.rng(cid).integers(0, 2 ** 31 - 1))
    gs = GR.sample(p, r.uid, kinds=('top_down', 'side', 'angled'), n=n, rng_seed=seed)
    GR.add_set(p, gs)
    st = GV.settings_of(p)
    notes: list = []
    ev = GV.Evaluator(p, st, notes)
    m = ev.robots[r.uid].model
    pads = GV.snapped_pads(p, r.uid, st['snap_mm'] / 1000.0, notes)
    link, off, T = GR.ik_targets(p, gs)
    b = GV.solve_grasps(p, r, m, link, T, off, st, m.home)
    return p, r, gs, st, ev, pads, T, b


def _near_edge(P, N, o, grid, angle: float, eps: float, k: int = 8) -> np.ndarray:
    """(K,) True where a point's exact visibility changes when it moves by `angle` (seen from o) within the plane
    of its own surface (normal N), the move lengthened by 1/|cos| of the viewing angle (at most 5×) so it spans
    the same angle on an oblique surface."""
    from ..core import raycast as RC
    v = P - o
    d = np.linalg.norm(v, axis=1)
    a = np.where(np.abs(N[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    t1 = np.cross(N, a)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(N, t1)
    cosi = np.abs(np.einsum('ij,ij->i', N, v / d[:, None]))
    rr = d * math.tan(angle) / np.maximum(cosi, 0.2)
    base = RC.visible_exact(P, o, grid, eps)
    out = np.zeros(len(P), bool)
    for j in range(k):
        th = 2 * np.pi * j / k
        Q = P + rr[:, None] * (np.cos(th) * t1 + np.sin(th) * t2)
        out |= RC.visible_exact(Q, o, grid, eps) != base
    return out


@check('grasp.visibility_vs_raycast', 'grasp', 'Contact visibility against exact ray casting',
       'fraction of the tested contact locations (fingertip pads and the object\'s grasp region, both cameras, IK '
       'solutions of sampled grasps on a box) whose visibility from the study\'s culled cube maps differs from exact '
       'ray casting of the full scene, leaving out those within two texels of an edge',
       'core/raycast.py visible_exact (Möller–Trumbore through a uniform grid) against the scene built again by '
       'core/scenegeom.sensor_occluders: no culling, no cube map, no q-bucket', 1e-3, 'fraction',
       refs=('moller1997', 'amanatides_woo1987', 'greene1986'), cost='study')
def _visibility(ctx):
    from ..core import camrender as CR
    from ..core import raycast as RC
    from ..core import scenegeom as SG
    from ..core.studies import grasp_visibility as GV
    cid = 'grasp.visibility_vs_raycast'
    n_grasps = 3 if ctx.quick else 10
    p, r, gs, st, ev, pads, T, b = _ur5e_with_box(ctx, cid, 40)
    surf = GV._object_surface(p, p.workspace.objects[0].uid, st['object_points'], st['rng_seed'] + 7)
    fi = np.flatnonzero(b.ok)[:n_grasps]
    lift, eps = st['lift_mm'] / 1000.0, st['eps_mm'] / 1000.0
    angle = 2 * 2.0 / st['cube_n']
    tested = disagree = edge = 0
    per, backend, renderer = [], '', ''
    for i in fi:
        qd = {r.uid: b.q[i]}
        fkd = ev.fk(qd)
        Pp, Np, _seg = GV.pad_world(pads, fkd[r.uid])
        Po, No = GV.object_region(surf, T[i, :3, 3], st['region_radius_m'])
        P, N = np.r_[Pp, Po], np.r_[Np, No]
        for j, c in enumerate(ev.cams):
            _in, _fac, unocc, _front = ev.states(j, fkd, qd, P, N, None, False, full=True)
            backend, renderer = ev.backend, ev.renderer
            pose = CR.optical_pose(p, c.sensor, c.imager, qd)
            occ = SG.sensor_occluders(p, c.sensor, qd, origins=pose.position[None])
            grid = RC.Grid.build(occ.tris)
            Pl = P + lift * N
            ref = RC.visible_exact(Pl, pose.position, grid, eps)
            bad = unocc != ref
            ne = _near_edge(Pl, N, pose.position, grid, angle, eps) if bad.any() else np.zeros(len(P), bool)
            tested += len(P)
            disagree += int(bad.sum())
            edge += int((bad & ne).sum())
            per.append(float(bad.mean()))
    away = disagree - edge
    return Outcome(away / max(tested, 1),
                   {'points': tested, 'grasps': int(len(fi)), 'cameras': len(ev.cams), 'face_size': st['cube_n'],
                    'backend': backend, 'renderer': renderer, 'disagree': disagree, 'disagree_near_edge': edge,
                    'disagree_elsewhere': away, 'disagree_fraction': disagree / max(tested, 1),
                    'infeasible_of_40': int((~b.ok).sum())},
                   ok=disagree / max(tested, 1) < 0.02 and tested > 0,
                   figure=figure('Contact locations where the study and exact ray casting disagree, per view',
                                 'grasp × camera', 'fraction', [series('disagree', np.arange(len(per)),
                                                                      np.maximum(per, 1e-9), 'points')], logy=True),
                   note='The cube map answers per texel, so a point within about a texel of an edge may be judged '
                        'either way; those are counted but not held against it (all disagreements must stay under '
                        '2 %). Elsewhere the two must agree, which also shows that culling the triangles outside '
                        'the cone around the points changes nothing.')


@check('grasp.bootstrap_coverage', 'grasp', 'Bootstrap interval of the Contact Visibility Score: coverage',
       'largest distance of the percentile bootstrap interval\'s coverage of the true p from the nominal 95 %, over '
       'synthetic binomial samples (n = 100, 300, 1000; p = 0.3, 0.7, 0.9; many seeded samples each)',
       'the known p of the synthetic samples; Wilson\'s score interval (written out in grasp_visibility.'
       'wilson_interval) computed on the same samples for comparison', 0.03, 'probability',
       refs=('efron1979', 'wilson1927'))
def _bootstrap(ctx):
    from ..core.studies.grasp_visibility import bootstrap_ci, wilson_interval
    trials = 300 if ctx.quick else 1000
    g = ctx.rng('grasp.bootstrap_coverage')
    rows, worst, gap_max = [], 0.0, 0.0
    xs, cov_b, cov_w = [], [], []
    for n in (100, 300, 1000):
        for p in (0.3, 0.7, 0.9):
            cb = cw = 0
            gaps = []
            for _t in range(trials):
                x = (g.random(n) < p).astype(float)
                lo, hi = bootstrap_ci(x, 2000, 0.95, g)
                wl, wh = wilson_interval(int(x.sum()), n, 0.95)
                cb += lo <= p <= hi
                cw += wl <= p <= wh
                gaps.append(max(abs(lo - wl), abs(hi - wh)))
            rows.append({'n': n, 'p': p, 'coverage_bootstrap': cb / trials, 'coverage_wilson': cw / trials,
                         'endpoint_gap_mean': float(np.mean(gaps))})
            worst = max(worst, abs(cb / trials - 0.95))
            gap_max = max(gap_max, float(np.mean(gaps)))
            xs.append(len(xs))
            cov_b.append(cb / trials)
            cov_w.append(cw / trials)
    se = math.sqrt(0.95 * 0.05 / trials)
    tol = 0.01 + 3 * se
    return Outcome(worst, {'trials_per_case': trials, 'cases': rows, 'standard_error_of_coverage': se,
                           'largest_mean_endpoint_gap_to_wilson': gap_max},
                   tolerance=tol, ok=gap_max <= 0.02,
                   figure=figure('Coverage of the true p (nominal 95 %)', 'case (n, p)', 'coverage',
                                 [series('bootstrap', xs, cov_b, 'points'), series('Wilson', xs, cov_w, 'points')],
                                 hline=0.95, hline_label='nominal'),
                   note=f'Tolerance: three standard errors of a coverage estimated from {trials} samples '
                        f'({3 * se:.3f}) plus 0.01 for the binomial\'s discreteness (the exact coverage of any '
                        'interval of a proportion moves by about a percent between neighbouring n and p). The mean '
                        'gap between the bootstrap and Wilson endpoints must stay under 0.02.')


@check('grasp.qbucket_vs_exact', 'grasp', 'q-bucket cache against evaluation at each q',
       'mean difference in a fingertip patch\'s visible fraction (both cameras) between the value the study reports '
       'for a joint vector in an evaluated bucket (the bucket\'s first q, its maps reused) and the value at the '
       'joint vector itself (maps drawn there); UR5e sample, pairs drawn in one bucket of the default size',
       'the same visibility evaluated at each joint vector without the cache', 0.02, 'fraction', cost='study')
def _qbucket(ctx):
    from ..core.studies import grasp_visibility as GV
    cid = 'grasp.qbucket_vs_exact'
    n_pairs = 4 if ctx.quick else 16
    p, r, gs, st, ev, pads, T, b = _ur5e_with_box(ctx, cid, 30)
    g = ctx.rng(cid, 1)
    m = ev.robots[r.uid].model
    step = math.radians(st['q_bucket_deg'])
    prism = GV._prismatic(m)
    diffs, hits0 = [], ev.hits
    for i in np.flatnonzero(b.ok)[:n_pairs]:
        cell = np.rint(b.q[i] / step)
        q1 = m.clip((cell + g.uniform(-0.5, 0.5, m.nq)) * step)
        q2 = m.clip((cell + g.uniform(-0.5, 0.5, m.nq)) * step)
        k1 = GV._bucket(q1, step, st['q_bucket_mm'] / 1000.0, prism)
        if k1 != GV._bucket(q2, step, st['q_bucket_mm'] / 1000.0, prism):
            continue                                       # clipped onto a joint limit across a bucket edge
        ev.maps.clear()
        fk1 = ev.fk({r.uid: q1})
        P1, N1, s1 = GV.pad_world(pads, fk1[r.uid])
        fk2 = ev.fk({r.uid: q2})
        P2, N2, s2 = GV.pad_world(pads, fk2[r.uid])
        for j in range(len(ev.cams)):
            ev.states(j, fk1, {r.uid: q1}, P1, N1, k1, True)                  # the bucket's first grasp
            a = ev.states(j, fk1, {r.uid: q1}, P1, N1, k1, True)              # what the study reports for q2
            u = ev.states(j, fk2, {r.uid: q2}, P2, N2, None, False)           # q2 evaluated at q2
            va, vu = a[0] & a[1] & a[2], u[0] & u[1] & u[2]
            for f in range(len(pads)):
                diffs.append(abs(float(va[s1 == f].mean()) - float(vu[s2 == f].mean())))
    hits = ev.hits - hits0
    err = float(np.mean(diffs)) if diffs else float('nan')
    worst = max(diffs) if diffs else float('nan')
    return Outcome(err, {'pairs': len(diffs) // max(1, len(pads) * len(ev.cams)) if diffs else 0,
                         'comparisons': len(diffs), 'largest_difference': worst, 'bucket_deg': st['q_bucket_deg'],
                         'cache_hits': hits, 'points_per_patch': [pd['points'] for pd in pads]},
                   ok=bool(diffs) and hits > 0 and worst <= 0.15,
                   figure=figure('Visible fraction: bucket\'s q against the grasp\'s own q', 'comparison',
                                 '|difference|', [series('difference', np.arange(len(diffs)),
                                                         np.maximum(diffs, 1e-6), 'points')], logy=True),
                   note='Within a 0.1° bucket the gripper moves by up to about a millimetre at the UR5e\'s reach. '
                        'That is a real change of the view; on average it may move a patch\'s visible fraction by '
                        'at most 2 % (2.5 of 128 points; measured 0.05 to 0.8 % over four seeds). Where an '
                        'occluder\'s edge crosses a patch a whole row of its points can flip at once (a row of the '
                        '2F-85 pad is about 11 of its 128 points), so the '
                        'largest single difference may reach a row or two: it must stay at most 0.15. The cache '
                        'must have been used (hits > 0).')
