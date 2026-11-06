"""Checks of core/occlusion.py (the articulated cube-map occlusion) against exact ray casting (core/raycast.py:
Möller–Trumbore intersection through a uniform grid, which shares no code with either rasteriser).

- occlusion.texel_depth_exact: the distance stored in each cube-map texel against the exact first hit along the same
  texel-centre ray, from the cameras of the UR5e sample at random arm poses.
- occlusion.cube_vs_raycast: point visibility from the cube map against exact visibility of the same points, with the
  disagreements sorted into those within two texels of a silhouette (expected: the map answers per texel) and those
  elsewhere (a fault).

Both use the bundled UR5e + Robotiq 2F-85 sample, built in the check, with its own cameras and task volume; the arm's
six joints are drawn around home (σ = 0.15 rad, the gripper at home) with the check's seeded generator.
"""
from __future__ import annotations

import numpy as np

from .suite import Outcome, check, figure, series


def _scene(ctx, cid: str, n_q: int):
    from ..core import scenegeom as SG
    from ..core.robot.model import get_model
    from ..core.robot.samples import sample_project
    p = sample_project('ur5e')
    r = p.robots[0]
    m = get_model(p, r.uid)
    rng = ctx.rng(cid)
    out = []
    for _ in range(n_q):
        q = m.home.copy()
        q[:6] += rng.normal(0, 0.15, 6)
        qd = {r.uid: m.clip(q)}
        occ = SG.scene_occluders(p, qd)
        for s in p.variant.sensors:
            out.append((s.name, SG.sensor_world_pose(p, s, qd).position, SG.sensor_occluders(p, s, qd, base=occ)))
    return p, rng, out


@check('occlusion.texel_depth_exact', 'occlusion', 'Cube-map texel depth against exact ray casting',
       'fraction of sampled texels whose stored distance differs from the exact first hit along the texel-centre ray '
       'by more than 1 mm (or where only one of them finds a surface); UR5e sample, both cameras, random arm poses',
       'core/raycast.py: Möller–Trumbore ray–triangle intersection through a uniform grid (Amanatides–Woo traversal)',
       1e-3, 'fraction', refs=('moller1997', 'amanatides_woo1987', 'greene1986'))
def _texel_depth(ctx):
    from ..core import occlusion as OC
    from ..core import raycast as RC
    n_q, n_tex, n = (1, 6000, 256) if ctx.quick else (4, 30000, 512)
    _p, rng, scenes = _scene(ctx, 'occlusion.texel_depth_exact', n_q)
    bad, total, diffs, backend, renderer, per = 0, 0, [], '', '', []
    for name, o, occ in scenes:
        cube = OC.CubeMap.build(occ, o, n)
        backend, renderer = cube.backend, cube.renderer
        f = rng.integers(0, 6, n_tex)
        r = rng.integers(0, n, n_tex)
        c = rng.integers(0, n, n_tex)
        d = OC.face_dirs(n)[f, r, c]
        fw = np.array([OC.FACES[k][0] for k in f])
        # the ray starts at the near plane, as the map's near clip does
        s0 = cube.near / np.einsum('ij,ij->i', d, fw)
        t = RC.first_hit(o + d * s0[:, None], d, occ.tris) + s0
        got = cube.faces[f, r, c].astype(float)
        fin_a, fin_b = np.isfinite(got), np.isfinite(t)
        both = fin_a & fin_b
        gap = np.zeros(n_tex)
        gap[both] = np.abs(got[both] - t[both])
        wrong = (fin_a != fin_b) | (gap > 1e-3)
        bad += int(wrong.sum())
        total += n_tex
        diffs.append(np.abs(got[both] - t[both]))
        per.append(float(wrong.mean()))
    dd = np.concatenate(diffs) if diffs else np.zeros(0)
    return Outcome(bad / max(total, 1), {'texels': total, 'face_size': n, 'maps': len(scenes), 'backend': backend,
                                         'renderer': renderer, 'median_abs_diff_m': float(np.median(dd)) if len(dd)
                                         else 0.0, 'p99_abs_diff_m': float(np.percentile(dd, 99)) if len(dd) else 0.0},
                   figure=figure('Texels off by more than 1 mm, per map', 'map', 'fraction',
                                 [series('wrong', np.arange(len(per)), np.maximum(per, 1e-9), 'points')], logy=True,
                                 hline=1e-3, hline_label='tolerance'),
                   note='Both compute the distance to the triangle plane along the same ray; a texel can differ only '
                        'where its centre lies on an edge shared by two triangles at different depths (the '
                        'rasteriser\'s edge rule against the ray test\'s), or by float32 rounding of the stored '
                        'distance. 1e-3 of the texels leaves room for those edge ties and nothing else.')


@check('occlusion.cube_vs_raycast', 'occlusion', 'Cube-map visibility against exact ray casting',
       'fraction of points (random in the task volume) whose cube-map visibility differs from exact ray casting and '
       'that are not within two texels of a silhouette; UR5e sample, both cameras, random arm poses',
       'core/raycast.visible_exact (Möller–Trumbore through a uniform grid) and raycast.near_silhouette',
       1e-3, 'fraction', refs=('moller1997', 'amanatides_woo1987', 'greene1986'))
def _cube_vs_ray(ctx):
    from ..core import occlusion as OC
    from ..core import raycast as RC
    n_q, n_pts, n = (1, 2000, 256) if ctx.quick else (4, 10000, 512)
    p, rng, scenes = _scene(ctx, 'occlusion.cube_vs_raycast', n_q)
    lo, hi = p.workspace.volumes[0].bounds()
    dis, elsewhere, total, per, vis, backend = 0, 0, 0, [], [], ''
    for name, o, occ in scenes:
        P = rng.uniform(lo, hi, (n_pts, 3))
        grid = RC.Grid.build(occ.tris)
        cube = OC.CubeMap.build(occ, o, n)
        backend = cube.backend
        ve = RC.visible_exact(P, o, grid)
        vc = cube.visible(P)
        bad = np.nonzero(ve != vc)[0]
        sil = RC.near_silhouette(P[bad], o, grid, 2.0 / n) if len(bad) else np.zeros(0, bool)
        dis += len(bad)
        elsewhere += int((~sil).sum())
        total += n_pts
        per.append(len(bad) / n_pts)
        vis.append(float(ve.mean()))
    frac = dis / max(total, 1)
    return Outcome(elsewhere / max(total, 1),
                   {'points': total, 'face_size': n, 'maps': len(scenes), 'backend': backend,
                    'disagree_fraction': frac, 'disagree_near_silhouette': dis - elsewhere,
                    'disagree_elsewhere': elsewhere, 'visible_fraction_exact_mean': float(np.mean(vis))},
                   ok=frac < 0.02,
                   figure=figure('Points where the cube map and ray casting disagree, per map', 'map', 'fraction',
                                 [series('disagree', np.arange(len(per)), np.maximum(per, 1e-9), 'points')],
                                 logy=True),
                   note='The cube map answers per texel (2/n rad at a face centre), so a point within about a texel '
                        'of a silhouette may be judged by a ray that passes the edge on the other side; those are '
                        'counted but not held against it (they must stay under 2 % of the points). Anywhere else '
                        'the two must agree: 1e-3 leaves room for rounding at the near clip and nothing else.')
