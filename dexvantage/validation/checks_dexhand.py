"""Checks of the multi-finger occlusion study (core/studies/dex_hand_tactile.py, spec §7.14).

- dexhand.matrix_vs_analytic: a scripted two-finger scene whose occlusion is known in closed form. Finger A's pad is
  the top face of a box; finger B, a box above it, casts a shadow on it by central projection from a camera above;
  a palm plate hides a corner of A's pad from a low side camera. The study's matrix entries (share of each pad not
  seen per camera, split by the owner of the first hit) are compared with the closed form.
- dexhand.owner_vs_bruteforce: random two-finger-and-palm scenes and cameras; the owner of the first hit per patch
  point from the study's kernel against an all-pairs Möller–Trumbore ray cast written out again here (no grid, no
  code shared with core/raycast.py) that keeps the index of the nearest triangle.
- dexhand.cube_owner_union: on the dexhand sample at the power grasp, the cube-map path's d = min over the owner
  groups' maps against one cube map of the whole scene, along random directions (a z-buffer of a union keeps the
  nearest of its parts, so they must be equal); the detail also gives the cube-map path's matrix against exact ray
  casting on the sample.
"""
from __future__ import annotations

import numpy as np

from .suite import Outcome, check, figure, series

H_CAM = 0.30                  # camera 1 height above A's pad plane (m)
GRID = 20                     # pad points per side (cell centres)


def _box(lo, hi) -> np.ndarray:
    """12 triangles of an axis-aligned box."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                  [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
    f = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7),
         (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return c[np.array(f)]


def _grid(x0, x1, y0, y1, z) -> np.ndarray:
    xs = x0 + (np.arange(GRID) + 0.5) / GRID * (x1 - x0)
    ys = y0 + (np.arange(GRID) + 0.5) / GRID * (y1 - y0)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    return np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])


def _look(cam_pos, target):
    """(R, t): optical frame (z forward, x right, y down) at cam_pos looking at target."""
    p, t = np.asarray(cam_pos, float), np.asarray(target, float)
    z = t - p
    z /= np.linalg.norm(z)
    up = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    if abs(z @ np.array([0.0, 0.0, 1.0])) < 0.9:
        up = np.array([0.0, 0.0, 1.0])
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z]), p


def scripted_scene():
    """The scene of dexhand.matrix_vs_analytic and its closed-form answer: {'tris', 'group', 'groups', 'pads',
    'cams', 'expected': {(pad, cam): {owner: fraction}}}. Groups: 0 = finger A, 1 = finger B, 2 = palm."""
    cx, cy = 0.002, 0.001                       # camera 1 above A, inside B's footprint
    zb0, zb1 = 0.02, 0.03                       # B's bottom (its pad, facing A) and top
    k = H_CAM / (H_CAM - zb1)                   # central projection of B's top face onto A's plane (z = 0)
    X0, X1, Y0, Y1 = -0.006, 0.017, -0.005, 0.006     # B's shadow on A's plane: edges on whole millimetres, so
    # every pad point (1 mm grid, centres on half millimetres) is 0.5 mm from an edge
    bx0, bx1 = cx + (X0 - cx) / k, cx + (X1 - cx) / k
    by0, by1 = cy + (Y0 - cy) / k, cy + (Y1 - cy) / k
    A = _box([-0.01, -0.01, -0.01], [0.01, 0.01, 0.0])
    B = _box([bx0, by0, zb0], [bx1, by1, zb1])
    # camera 2: low at the side, 10 mm up, 40 mm out along +x; the palm plate at x = 0.025 hides A's points with
    # x >= XE and y >= 0: a ray from c2 to (x, y, 0) crosses x = 0.025 at s = 0.015 / (0.04 - x) of the way, at
    # height 0.01 (1 - s) and y = s y; it is blocked when that height is below the plate's top
    c2 = np.array([0.04, 0.0, 0.01])
    XE = 0.002
    s_edge = (c2[0] - 0.025) / (c2[0] - XE)
    plate_top = c2[2] * (1 - s_edge)
    C = _box([0.0249, 0.0, -0.01], [0.0251, 0.02, plate_top])
    tris = np.concatenate([A, B, C])
    group = np.r_[np.zeros(len(A), int), np.ones(len(B), int), np.full(len(C), 2)]
    pads = {'A': _grid(-0.01, 0.01, -0.01, 0.01, 0.0), 'B': _grid(bx0, bx1, by0, by1, zb0)}
    R1, t1 = _look([cx, cy, H_CAM], [cx, cy, 0.0])
    R2, t2 = _look(c2, [0.0, 0.0, 0.01])
    PA = pads['A']
    in_shadow = (PA[:, 0] >= X0) & (PA[:, 0] <= X1) & (PA[:, 1] >= Y0) & (PA[:, 1] <= Y1)
    behind_plate = (PA[:, 0] >= XE) & (PA[:, 1] >= 0.0)
    expected = {('A', 0): {1: float(in_shadow.mean())}, ('A', 1): {2: float(behind_plate.mean())},
                ('B', 0): {1: 1.0}, ('B', 1): {}}
    point_owner = {('A', 0): np.where(in_shadow, 1, -1), ('A', 1): np.where(behind_plate, 2, -1),
                   ('B', 0): np.ones(len(pads['B']), int), ('B', 1): np.full(len(pads['B']), -1)}
    return {'tris': tris, 'group': group, 'groups': ['finger A', 'finger B', 'palm'], 'pads': pads,
            'cams': [(R1, t1), (R2, t2)], 'expected': expected, 'point_owner': point_owner}


def _camera():
    from ..core.optics import CameraModel
    return CameraModel.from_fov(640, 640, 120.0, 120.0)


def brute_owner(points, origin, tris, group, eps=1e-3, near=0.005) -> np.ndarray:
    """Owner group of the first hit (−1: seen) by testing every ray against every triangle (Möller–Trumbore, written
    out here) and keeping the nearest triangle's index."""
    P = np.asarray(points, float)
    o = np.asarray(origin, float)
    v = P - o
    dist = np.linalg.norm(v, axis=1)
    d = v / dist[:, None]
    T = np.asarray(tris, float)
    v0, e1, e2 = T[:, 0], T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
    best = np.full(len(P), np.inf)
    idx = np.full(len(P), -1)
    for i in range(len(P)):
        ro = o + d[i] * near
        p = np.cross(d[i], e2)
        det = np.einsum('ij,ij->i', e1, p)
        okd = np.abs(det) > 1e-14
        inv = np.where(okd, 1.0 / np.where(okd, det, 1.0), 0.0)
        s = ro - v0
        u = np.einsum('ij,ij->i', s, p) * inv
        q = np.cross(s, e1)
        w = (q @ d[i]) * inv
        t = np.einsum('ij,ij->i', e2, q) * inv
        hit = okd & (u >= 0) & (w >= 0) & (u + w <= 1) & (t > 0)
        if hit.any():
            j = int(np.argmin(np.where(hit, t, np.inf)))
            best[i], idx[i] = t[j] + near, j
    owner = np.where(idx >= 0, np.asarray(group)[np.maximum(idx, 0)], -1)
    owner[dist <= best + eps] = -1
    return owner


@check('dexhand.matrix_vs_analytic', 'dexhand', 'Finger × camera occlusion matrix against a closed form',
       'largest difference between the study\'s share of a pad hidden by each owner (or seen) and the closed form, '
       'over the 2 pads × 2 cameras × owners of a scripted two-finger scene with a palm plate; the default '
       '(ray-cast) path and the cube-map path at n = 2048',
       'central projection of a box finger\'s top face onto the other finger\'s pad plane, and of a palm plate\'s '
       'edge seen from a low side camera, written out in the check (point counts on a 20 × 20 grid)',
       1e-3, 'fraction of a pad', refs=('hartley2004', 'moller1997'))
def _matrix_vs_analytic(ctx):
    from ..core.studies import dex_hand_tactile as D
    sc = scripted_scene()
    cam = _camera()
    G = len(sc['groups'])
    rows, worst, per = [], {}, {}
    for path in ('raycast', 'cube'):
        err = 0.0
        for (pad, j), exp in sc['expected'].items():
            R, t = sc['cams'][j]
            P = sc['pads'][pad]
            if path == 'raycast':
                vis, own, _i = D.visibility_owner_raycast(P, t, sc['tris'], sc['group'], G)
            else:
                vis, own, _i = D.visibility_owner(P, t, sc['tris'], sc['group'], G, n=2048, backend='auto')
            _uv, iv = cam.project((P - t) @ R)
            c = D.patch_counts(iv, vis, own, G) / len(P)
            want = np.zeros(G + 2)
            for g, f in exp.items():
                want[g] = f
            want[G + 1] = 1.0 - sum(exp.values())
            e = float(np.abs(c - want).max())
            err = max(err, e)
            rows.append({'path': path, 'pad': pad, 'camera': j + 1, 'got': [round(float(x), 6) for x in c],
                         'expected': [round(float(x), 6) for x in want]})
        worst[path] = err
        per[path] = err
    return Outcome(max(worst.values()), {'cells': rows, 'error_raycast': worst['raycast'],
                                         'error_cube_2048': worst['cube'], 'groups': sc['groups'] +
                                         ['outside the image', 'seen'],
                                         'expected_A_cam1_hidden_by_B': sc['expected'][('A', 0)][1],
                                         'expected_A_cam2_hidden_by_palm': sc['expected'][('A', 1)][2]},
                   figure=figure('Largest matrix-entry error per path', 'path', 'fraction of a pad',
                                 [series('error', [0, 1], [max(per['raycast'], 1e-12), max(per['cube'], 1e-12)],
                                         'points')], logy=True, hline=1e-3, hline_label='tolerance'),
                   note='Every pad point lies 0.5 mm (on the pad plane) from a shadow edge, which is 1.7 mrad from '
                        'camera 1 and more from camera 2, against a texel of 1 mrad at n = 2048: no point is '
                        'ambiguous for either path, so both must reproduce the closed form point for point. 1e-3 '
                        '(less than one of the 400 points of a pad) allows nothing else.')


@check('dexhand.owner_vs_bruteforce', 'dexhand', 'Owner of the first hit against an all-pairs ray cast',
       'fraction of patch points whose first-hit owner (or "seen") from the study\'s default kernel differs from an '
       'all-pairs Möller–Trumbore cast that keeps the nearest triangle; random two-finger and palm box scenes, random '
       'camera positions',
       'Möller–Trumbore ray–triangle intersection against every triangle, nearest hit kept by index, written out in '
       'the check (no grid; no code shared with core/raycast.py)',
       1e-3, 'fraction', refs=('moller1997',))
def _owner_vs_brute(ctx):
    from ..core.studies import dex_hand_tactile as D
    rng = ctx.rng('dexhand.owner_vs_bruteforce')
    n_scenes = 6 if ctx.quick else 30
    bad, total, per, owners_seen = 0, 0, [], set()
    for _ in range(n_scenes):
        boxes, group = [], []
        for g in range(3):
            c = rng.uniform(-0.03, 0.03, 3)
            h = rng.uniform(0.004, 0.015, 3)
            boxes.append(_box(c - h, c + h))
            group.append(np.full(12, g))
        tris = np.concatenate(boxes)
        grp = np.concatenate(group)
        # the patch: points on the top face of the first box, a few millimetres proud of the others' faces
        b0 = boxes[0]
        lo, hi = b0.reshape(-1, 3).min(axis=0), b0.reshape(-1, 3).max(axis=0)
        P = np.column_stack([rng.uniform(lo[0], hi[0], 150), rng.uniform(lo[1], hi[1], 150), np.full(150, hi[2])])
        o = rng.normal(0, 1, 3)
        o = o / np.linalg.norm(o) * rng.uniform(0.12, 0.4)
        o[2] = abs(o[2]) + 0.05
        _vis, own, _i = D.visibility_owner_raycast(P, o, tris, grp, 3)
        ref = brute_owner(P, o, tris, grp)
        wrong = int((own != ref).sum())
        bad += wrong
        total += len(P)
        per.append(wrong / len(P))
        owners_seen |= set(np.unique(ref).tolist())
    return Outcome(bad / max(total, 1), {'scenes': n_scenes, 'points': total, 'disagree': bad,
                                         'owners_met': sorted(int(x) for x in owners_seen)},
                   ok=len(owners_seen) >= 3,
                   figure=figure('Points whose owner differs, per scene', 'scene', 'fraction',
                                 [series('differ', np.arange(len(per)), np.maximum(per, 1e-9), 'points')],
                                 logy=True, hline=1e-3, hline_label='tolerance'),
                   note='Both are exact ray–triangle tests; they can differ only for a ray through a shared edge or '
                        'corner (rounding) or a hit within rounding of the 1 mm visibility margin. 1e-3 leaves room '
                        'for those and nothing else. The scenes must show at least three owners (seen, its own '
                        'finger and another group) for the check to count.')


@check('dexhand.cube_owner_union', 'dexhand', 'Owner-group cube maps against one cube map of the scene',
       'largest difference (m) between min over the owner groups\' cube-map distances and one cube map of the whole '
       'scene, along random directions; dexhand sample, power grasp, every camera',
       'core/occlusion.py CubeMap of the whole scene (a z-buffer of the union of the groups)', 1e-6, 'm',
       refs=('greene1986',))
def _cube_union(ctx):
    from ..core import dexhand as DH
    from ..core import occlusion as OC
    from ..core import scenegeom as SG
    from ..core.robot.model import get_model
    from ..core.robot.samples import sample_project
    from ..core.studies import dex_hand_tactile as D
    p = sample_project('dexhand')
    r = p.robots[0]
    m = get_model(p, r.uid)
    st = D.settings_of(p)
    syn = DH.synergies(p, r.uid, st, m)
    q = DH.posture(m, syn[0], 1.0)
    qd = {r.uid: q}
    hand = DH.structure(p, r.uid, m)
    rng = ctx.rng('dexhand.cube_owner_union')
    n = 256 if ctx.quick else 512
    occ = SG.scene_occluders(p, qd, objects=False)
    worst, backend, maps = 0.0, '', 0
    from ..core import library as LB
    for s in p.variant.sensors:
        im = LB.sensor_imagers(s)[0]
        t = (SG.sensor_world_pose(p, s, qd).matrix() @ im.pose_in_body.matrix())[:3, 3]
        oc = SG.sensor_occluders(p, s, qd, base=occ, origins=t[None])
        keys, gidx = D.owner_groups(oc, hand, r.uid)
        dirs = rng.normal(0, 1, (20000, 3))
        full_map = OC.CubeMap.build(oc, t, n)
        backend = full_map.backend
        full = full_map.depth(dirs)
        best = np.full(len(dirs), np.inf)
        for g in range(len(keys)):
            sel = gidx == g
            if sel.any():
                best = np.minimum(best, OC.CubeMap.build(oc.tris[sel], t, n).depth(dirs))
                maps += 1
        both = np.isfinite(full) & np.isfinite(best)
        if (np.isfinite(full) != np.isfinite(best)).any():
            worst = float('inf')
        elif both.any():
            worst = max(worst, float(np.abs(full[both] - best[both]).max()))
    # the cube-map path against exact ray casting on the sample (reported, not held against it)
    st_c = dict(st, backend='auto', cube_n=n)
    st_r = dict(st, backend='raycast')
    a = D.visibility_at(p, qd, st_c)
    b = D.visibility_at(p, qd, st_r)
    diff = {}
    for ca, cb, s in zip(a['cameras'], b['cameras'], p.variant.sensors):
        diff[s.name] = {f: round(abs(float((ca['in_view'][f] & ca['visible'][f]).mean())
                                     - float((cb['in_view'][f] & cb['visible'][f]).mean())) * 100, 2)
                        for f in ca['visible']}
    return Outcome(worst, {'face_size': n, 'maps': maps, 'backend': backend,
                           'cube_vs_raycast_seen_points_diff_pct': diff},
                   note='The cube map of a union keeps, per texel, the nearest of the triangles drawn into it, so the '
                        'minimum over maps of disjoint groups must equal it exactly (float32 distances on the GPU, '
                        'the same arithmetic per group); 1e-6 m allows float32 rounding of a distance near 1 m. The '
                        'detail lists, per camera and finger, how many points (percent of the patch) the cube-map '
                        'path sees differently from exact ray casting at this face size: the texel effect that makes '
                        'ray casting the study\'s default.')
