"""Copy the MuJoCo Menagerie models of the ALOHA 2, bimanual Panda and Unitree G1 samples into
dexvantage/resources/robots/ (spec §1.2, §14) and write the robot files the samples open.

    git clone --filter=blob:none --no-checkout https://github.com/google-deepmind/mujoco_menagerie.git
    cd mujoco_menagerie && git checkout 98a70c569c4c1b6da467a20012f766ee1f8dc521
    git sparse-checkout set --no-cone /aloha/ /franka_emika_panda/ /unitree_g1/
    .venv/bin/pip install fast-simplification==0.2.0          # the optional extra "bundle"; only this script uses it
    python -I scripts/bundle_samples.py <that checkout> [dexvantage/resources/robots]

Run it only after reading each model's own LICENSE (Menagerie licences differ per model; aloha BSD-3-Clause,
franka_emika_panda Apache-2.0, unitree_g1 BSD-3-Clause, read 2026-11-07) and recording it in THIRD_PARTY.md. The
script only reads the checkout; it never changes it. What it writes, and what it changes:

- aloha/: aloha.xml, joint_position_actuators.xml, keyframe_ctrl.xml (the two files aloha.xml includes), LICENSE and
  the 12 STL meshes aloha.xml uses, byte for byte (8.3 MB). scene.xml (the table and the aluminium frame, 13.2 MB with
  their meshes) is not bundled; its two frame cameras "overhead_cam" and "worms_eye_cam" and its table's collision box
  "table" are written to aloha/scene_parts.xml with the attributes scene.xml gives them (checked against scene.xml
  compiled by MuJoCo).
- aloha_left.xml, aloha_right.xml: one ViperX 300 arm of aloha.xml each, written by MuJoCo (MjSpec, to_xml): the
  other arm's bodies, every actuator, the gripper equality constraints, the keyframes and the world cameras are
  deleted, the arm's base_link is moved to the file's origin (the sample places each robot with its base pose, read
  from aloha.xml when the sample is built), and a keyframe "home" is written with this arm's values of aloha.xml's
  "neutral_pose". Every body, joint, geom and the wrist camera keep aloha.xml's names and numbers (MuJoCo's to_xml
  writes the numbers back to its own precision). Checked here: each arm's bodies at random joint values against
  aloha.xml compiled by MuJoCo, in the arm's base frame (printed).
- franka_emika_panda/: panda.xml and LICENSE; the 59 OBJ meshes panda.xml uses (34.3 MB of text) rewritten as
  binary STL with the same triangles (6.8 MB), named <stem>_obj.stl because the folder has STL collision meshes of
  the same stem (link1.stl beside link1.obj); the STL meshes are copied byte for byte. The <mesh file> names in
  panda.xml change accordingly and a comment at its top says so. Checked: every triangle of every geom at the home
  keyframe against the original (printed; MuJoCo keeps float32 vertices either way).
- unitree_g1/: g1_with_hands.xml and LICENSE, and its 49 STL meshes (31.5 MB) SIMPLIFIED (quadric edge collapse,
  fast-simplification 0.2.0) so the model fits the package budget: each mesh is welded (its STL triangle soup turned
  into shared vertices) and reduced as far as its two-sided Hausdorff distance from the original stays within
  G1_TOLERANCE_M (measured here on the original's vertices and area-sampled points against the exact surface, both
  ways; the largest is printed and recorded in THIRD_PARTY.md and the sample's note). The meshes are named
  <stem>.simplified.stl and g1_with_hands.xml's <mesh file> names say so, with a comment at its top. Nothing else in
  the file changes.
- g1_upper_body.xml: g1_with_hands.xml with its legs and waist held at the file's "stand" keyframe: the free joint
  "floating_base_joint" is deleted and the pelvis placed at the keyframe's base pose (0, 0, 0.79 m, upright); the 12
  leg joints and 3 waist joints are deleted (at "stand" they are all 0, so the bodies stay exactly where the
  keyframe puts them); every actuator, sensor and keyframe is deleted and a keyframe "home" written with the arms'
  and hands' "stand" values. What remains in q: both arms (7 joints each) and both hands (7 each), 28 joints. Checked:
  every body at random arm and hand values against g1_with_hands.xml compiled by MuJoCo at "stand" with the same
  values (printed; 0 m expected). Contact is excluded between each palm (wrist_yaw_link) and the thumb's middle link
  (hand_thumb_1_link): Menagerie's own collision meshes of the two overlap by 0.27 mm at "stand", so without it every
  pose of the arm counts as a self-collision. ('g1-upper' as the third argument rewrites only this file.)
"""
from __future__ import annotations

import os
import re
import shutil
import struct
import sys
import warnings

import numpy as np

MENAGERIE_COMMIT = '98a70c569c4c1b6da467a20012f766ee1f8dc521'
G1_TOLERANCE_M = 0.5e-3           # the largest two-sided Hausdorff distance a simplified G1 mesh may have (a choice)
SEARCH_SAMPLES = 5_000            # area-weighted surface samples per side while searching for the reduction
FINAL_SAMPLES = 40_000            # ... and for the figure recorded
AGGRESSIVENESS = (1.0, 3.0)       # fast-simplification's agg values tried (7, its default, lost 1-11 mm): 1 keeps
#                                   the shape best but stops early on some meshes; the fewest triangles within tol win
ALOHA_FILES = ('aloha.xml', 'joint_position_actuators.xml', 'keyframe_ctrl.xml', 'LICENSE')
ALOHA_FRAME_CAMERAS = ('overhead_cam', 'worms_eye_cam')
SCENE_PARTS = 'scene_parts.xml'
G1_LEG_WAIST = tuple(f'{s}_{j}_joint' for s in ('left', 'right')
                     for j in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')) + \
    ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint')
NOTE_PANDA = ('<!-- DexVantage: the OBJ <mesh> files are the Menagerie OBJ meshes rewritten as binary STL (the same '
              'triangles, float32) named <stem>_obj.stl, and a <mesh> that had no name is given the name MuJoCo gave '
              'it (its old file stem), by scripts/bundle_samples.py; nothing else in this file was changed. -->\n')
NOTE_G1 = ('<!-- DexVantage: the <mesh> files are the Menagerie STL meshes SIMPLIFIED by scripts/bundle_samples.py '
           '(quadric edge collapse, each within {tol} mm two-sided Hausdorff distance of the original; measured '
           'largest {err} mm), named <stem>.simplified.stl, each <mesh> named as MuJoCo named it before (its old '
           'file stem); nothing else in this file was changed. -->\n')


# ====================================================================== meshes
def read_stl(path: str) -> tuple[np.ndarray, np.ndarray]:
    """(V (n, 3) float64, F (m, 3) int) of a binary or ASCII STL, welded: corners at the same float32 point share
    one vertex."""
    raw = open(path, 'rb').read()
    n = struct.unpack('<I', raw[80:84])[0] if len(raw) >= 84 else 0
    if len(raw) == 84 + 50 * n:
        rec = np.frombuffer(raw, dtype=[('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')], count=n, offset=84)
        T = rec['v'].astype(np.float32)
    else:
        pts = [ln.split()[1:4] for ln in raw.decode('ascii', 'replace').splitlines() if ln.strip().startswith('vertex')]
        T = np.asarray(pts, np.float32).reshape(-1, 3, 3)
    return weld(T)


def read_obj(path: str) -> tuple[np.ndarray, np.ndarray]:
    """(V, F) of an OBJ's v and f records (polygons fanned), as scripts/bundle_menagerie.py reads them."""
    verts, tris = [], []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            if line.startswith('v '):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '):
                idx = [int(t.split('/')[0]) for t in line.split()[1:]]
                idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                for k in range(1, len(idx) - 1):
                    tris.append((idx[0], idx[k], idx[k + 1]))
    return np.asarray(verts, np.float64), np.asarray(tris, int).reshape(-1, 3)


def weld(T) -> tuple[np.ndarray, np.ndarray]:
    T = np.asarray(T, np.float32).reshape(-1, 3)
    V, inv = np.unique(T, axis=0, return_inverse=True)
    F = inv.reshape(-1, 3)
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    return V.astype(np.float64), F.astype(int)


def write_stl(path: str, V, F, header: bytes) -> int:
    T = np.asarray(V, np.float64)[np.asarray(F, int)]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)
    rec = np.zeros(len(T), dtype=[('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
    rec['n'], rec['v'] = n, T
    with open(path, 'wb') as fh:
        fh.write(header.ljust(80, b' ')[:80])
        fh.write(struct.pack('<I', len(T)))
        fh.write(rec.tobytes())
    return len(T)


def point_triangle_distance(P, A, B, C) -> np.ndarray:
    """Exact distance from points P (n, 3) to triangles (A, B, C) (n, 3 each), pairwise (Ericson, Real-Time Collision
    Detection, 5.1.5: the closest point by Voronoi region)."""
    ab, ac, ap = B - A, C - A, P - A
    d1, d2 = np.einsum('ij,ij->i', ab, ap), np.einsum('ij,ij->i', ac, ap)
    bp = P - B
    d3, d4 = np.einsum('ij,ij->i', ab, bp), np.einsum('ij,ij->i', ac, bp)
    cp = P - C
    d5, d6 = np.einsum('ij,ij->i', ab, cp), np.einsum('ij,ij->i', ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = va + vb + vc
        v = np.where(denom != 0, vb / denom, 0.0)
        w = np.where(denom != 0, vc / denom, 0.0)
        Q = A + ab * v[:, None] + ac * w[:, None]                       # inside the face
        # edges
        t_ab = np.clip(np.where(d1 - d3 != 0, d1 / (d1 - d3), 0.0), 0, 1)
        t_ac = np.clip(np.where(d2 - d6 != 0, d2 / (d2 - d6), 0.0), 0, 1)
        e = d4 - d3 + d5 - d6
        t_bc = np.clip(np.where(e != 0, (d4 - d3) / e, 0.0), 0, 1)
    cand = np.stack([Q, A + ab * t_ab[:, None], A + ac * t_ac[:, None], B + (C - B) * t_bc[:, None]], 1)
    inside = (va >= 0) & (vb >= 0) & (vc >= 0) & (denom > 0)
    d = np.linalg.norm(cand - P[:, None], axis=2)
    edge = d[:, 1:].min(axis=1)
    return np.where(inside, np.minimum(d[:, 0], edge), edge)


def surface_samples(V, F, n: int, seed: int = 0) -> np.ndarray:
    """The vertices plus n area-weighted uniform samples of the surface (seed fixed)."""
    rng = np.random.default_rng(seed)
    T = V[F]
    area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    if n <= 0 or area.sum() <= 0:
        return V.copy()
    k = rng.choice(len(F), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    P = (1 - s)[:, None] * T[k, 0] + (s * (1 - r2))[:, None] * T[k, 1] + (s * r2)[:, None] * T[k, 2]
    return np.concatenate([V, P])


def distance_to_mesh(P, V, F) -> np.ndarray:
    """(n,) exact distance from each point to the triangle mesh (V, F). The k triangles with the nearest centroids
    are tested; the result is certified exact when it is no more than c_k − R (c_k the k-th centroid's distance, R the
    largest centroid-to-corner distance of any triangle: a triangle further down the list is at least c_k − R away);
    points not certified are tried again with 4 k, then against every triangle."""
    from scipy.spatial import cKDTree
    T = V[F]
    cen = T.mean(axis=1)
    rad = np.linalg.norm(T - cen[:, None], axis=2).max(axis=1)
    # triangles in classes of similar size (radius within a factor 2), each with its own centroid tree, so one long
    # sliver does not weaken the bound of every other triangle
    cls = np.floor(np.log2(np.maximum(rad, 1e-12) / max(rad.max(), 1e-12))).astype(int)
    classes = []
    for c in np.unique(cls):
        sel = np.flatnonzero(cls == c)
        classes.append((sel, cKDTree(cen[sel]), float(rad[sel].max())))
    best = np.full(len(P), np.inf)
    pending = [(i, np.arange(len(P))) for i in range(len(classes))]
    k = 16
    while pending:
        nxt = []
        for ci, ids in pending:
            sel, tree, R = classes[ci]
            kk = min(k, len(sel))
            bound = np.zeros(len(ids))
            step = max(1, 400_000 // kk)
            for s in range(0, len(ids), step):
                q = ids[s:s + step]
                ck, idx = tree.query(P[q], k=kk)
                ck, idx = ck.reshape(len(q), -1), idx.reshape(len(q), -1)
                own = np.repeat(np.arange(len(q)), idx.shape[1])
                fl = sel[idx.ravel()]
                d = point_triangle_distance(P[q][own], T[fl, 0], T[fl, 1], T[fl, 2]).reshape(len(q), -1)
                best[q] = np.minimum(best[q], d.min(axis=1))
                bound[s:s + step] = (ck[:, -1] - R) if kk < len(sel) else np.inf
            nxt.append((ci, ids, bound))
        pending = [(ci, ids[best[ids] > bound]) for ci, ids, bound in nxt]
        pending = [(ci, ids) for ci, ids in pending if len(ids)]
        k *= 4
    return best


def hausdorff(V1, F1, V2, F2, n: int = FINAL_SAMPLES, seed: int = 0) -> float:
    """The two-sided Hausdorff distance between two meshes, measured on each one's vertices and n area samples of
    each surface against the other's exact surface."""
    a = distance_to_mesh(surface_samples(V1, F1, n, seed), V2, F2).max()
    b = distance_to_mesh(surface_samples(V2, F2, n, seed + 1), V1, F1).max()
    return float(max(a, b))


def simplify(V, F, tol: float) -> tuple[np.ndarray, np.ndarray, float]:
    """The smallest quadric-edge-collapse simplification (fast-simplification, each agg of AGGRESSIVENESS) of (V, F)
    found by bisection on the reduction whose two-sided Hausdorff distance from the original is at most tol: (V', F',
    its distance measured with FINAL_SAMPLES samples a side). The original is returned when nothing passes."""
    try:
        import fast_simplification as FS
    except ImportError:
        raise SystemExit('This script needs fast-simplification 0.2.0: .venv/bin/pip install '
                         'fast-simplification==0.2.0 (the optional extra "bundle").') from None

    def run(r, agg):
        Vs, Fs = FS.simplify(V.astype(np.float32), F.astype(np.int32), target_reduction=r, agg=agg)
        return np.asarray(Vs, np.float64), np.asarray(Fs, int)
    best = (V, F, 0.0)
    for agg in AGGRESSIVENESS:
        lo, hi, good = 0.0, 0.98, None
        for _ in range(7):
            mid = 0.5 * (lo + hi)
            Vs, Fs = run(mid, agg)
            if len(Fs) >= 4 and hausdorff(V, F, Vs, Fs, SEARCH_SAMPLES) <= tol:
                good, lo = mid, mid
            else:
                hi = mid
        while good is not None and good > 0.01:
            Vs, Fs = run(good, agg)
            if len(Fs) >= len(best[1]):
                break
            err = hausdorff(V, F, Vs, Fs)
            if err <= tol:
                best = (Vs, Fs, err)
                break
            good -= 0.05                                # the denser measure found more: step back
    return best


# ====================================================================== world triangles (the comparison)
def world_triangles(path: str, key: str | None = None) -> list:
    """Per mesh geom, its triangles in the world at the keyframe `key` (default the first), rows sorted."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    if m.nkey:
        k = 0 if key is None else m.key(key).id
        d.qpos[:] = m.key_qpos[k]
    mujoco.mj_kinematics(m, d)
    out = []
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        k = m.geom_dataid[g]
        va, vn, fa, fn = m.mesh_vertadr[k], m.mesh_vertnum[k], m.mesh_faceadr[k], m.mesh_facenum[k]
        V = m.mesh_vert[va:va + vn].astype(float)
        Fc = m.mesh_face[fa:fa + fn]
        T = V[Fc] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        key_ = T.round(7)
        o = np.lexsort((key_[:, :, 2], key_[:, :, 1], key_[:, :, 0]), axis=-1)
        T = np.take_along_axis(T, o[:, :, None], axis=1).reshape(-1, 9)
        out.append(T[np.lexsort(T.round(7).T[::-1])])
    return out


def compare_triangles(a: str, b: str) -> float:
    A, B = world_triangles(a), world_triangles(b)
    if len(A) != len(B) or any(x.shape != y.shape for x, y in zip(A, B)):
        raise SystemExit(f'{os.path.basename(b)}: a different number of geoms or triangles than the original')
    return max(float(np.abs(x - y).max()) for x, y in zip(A, B))


def _pose_of(d, name: str, kind: str = 'body') -> np.ndarray:
    T = np.eye(4)
    el = d.body(name) if kind == 'body' else d.cam(name)
    T[:3, :3] = el.xmat.reshape(3, 3)
    T[:3, 3] = el.xpos
    return T


def compare_bodies(ref_path: str, ref_key: str | None, sub_path: str, root: str | None = None, rng_seed: int = 0,
                   n: int = 20) -> float:
    """The largest difference (position, m, and rotation-matrix entry) of every body of the file sub_path against
    the same body of ref_path, at n random values of sub_path's joints (the rest of ref_path at its keyframe); with
    root, ref_path's bodies are taken in the frame of its body root (sub_path's world)."""
    import mujoco
    ref = mujoco.MjModel.from_xml_path(ref_path)
    sub = mujoco.MjModel.from_xml_path(sub_path)
    dr, ds = mujoco.MjData(ref), mujoco.MjData(sub)
    rng = np.random.default_rng(rng_seed)
    names = [sub.body(b).name for b in range(1, sub.nbody)]
    worst = 0.0
    for _ in range(n):
        if ref.nkey:
            dr.qpos[:] = ref.key_qpos[0 if ref_key is None else ref.key(ref_key).id]
        for j in range(sub.njnt):
            nm = sub.joint(j).name
            lo, hi = sub.jnt_range[j] if sub.jnt_limited[j] else (-np.pi, np.pi)
            v = rng.uniform(lo, hi)
            ds.qpos[sub.jnt_qposadr[j]] = v
            dr.qpos[ref.jnt_qposadr[ref.joint(nm).id]] = v
        mujoco.mj_kinematics(ref, dr)
        mujoco.mj_kinematics(sub, ds)
        Ti = np.linalg.inv(_pose_of(dr, root)) if root else np.eye(4)
        for nm in names:
            worst = max(worst, float(np.abs(Ti @ _pose_of(dr, nm) - _pose_of(ds, nm)).max()))
    return worst


def _spec_in(dst_root: str, rel: str):
    """An MjSpec of dst_root/rel with its mesh paths made relative to dst_root."""
    import mujoco
    folder = os.path.dirname(rel)
    # from a string, so file names resolve against the folder the spec is compiled in (dst_root), not rel's folder;
    # <include> files are put in place first (their <mujoco> wrapper dropped), as MuJoCo itself would
    text = open(os.path.join(dst_root, rel), encoding='utf-8').read()

    def inline(mt):
        inc = open(os.path.join(dst_root, folder, mt.group(1)), encoding='utf-8').read()
        return re.sub(r'^\s*<mujoco[^>]*>|</mujoco>\s*$', '', inc.strip())
    text = re.sub(r'<include\s+file="([^"]+)"\s*/>', inline, text)
    spec = mujoco.MjSpec.from_string(text)
    md = spec.meshdir or ''
    for mesh in spec.meshes:
        mesh.file = '/'.join(x for x in (folder, md, mesh.file) if x)
    spec.meshdir = ''
    return spec


def _to_xml(spec, dst_root: str, compile_only: bool = False):
    """spec compiled in dst_root (its mesh paths are relative to it): the XML, or the MjModel when compile_only."""
    cwd = os.getcwd()
    try:
        os.chdir(dst_root)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = spec.compile()
            return m if compile_only else spec.to_xml()
    finally:
        os.chdir(cwd)
        log = os.path.join(dst_root, 'MUJOCO_LOG.TXT')
        if os.path.exists(log):
            os.remove(log)


def _delete_all(spec, items) -> None:
    for it in list(items):
        spec.delete(it)


# ====================================================================== ALOHA 2
def bundle_aloha(src_root: str, dst_root: str) -> list:
    import mujoco
    src, dst = os.path.join(src_root, 'aloha'), os.path.join(dst_root, 'aloha')
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(os.path.join(dst, 'assets'))
    for f in ALOHA_FILES:
        shutil.copyfile(os.path.join(src, f), os.path.join(dst, f))
    used = {m.file for m in mujoco.MjSpec.from_file(os.path.join(src, 'aloha.xml')).meshes}
    for f in sorted(used):
        shutil.copyfile(os.path.join(src, 'assets', f), os.path.join(dst, 'assets', f))
    ref = mujoco.MjModel.from_xml_path(os.path.join(dst, 'aloha.xml'))
    neutral = ref.key_qpos[ref.key('neutral_pose').id]
    out = []
    # scene.xml's two frame cameras and its table's collision box, alone in a file (aloha.xml's world frame)
    scene = mujoco.MjSpec.from_file(os.path.join(src, 'scene.xml'))
    cams = mujoco.MjSpec()
    cams.modelname = 'aloha_scene_parts'
    t0 = scene.geom('table')
    cams.worldbody.add_geom(name='table', type=t0.type, size=list(t0.size), pos=list(t0.pos), quat=list(t0.quat))
    for c0 in scene.cameras:
        if c0.name in ALOHA_FRAME_CAMERAS:
            c = cams.worldbody.add_camera(name=c0.name, pos=list(c0.pos), quat=list(c0.quat))
            c.mode = c0.mode
            c.sensor_size = list(c0.sensor_size)
            c.resolution = list(c0.resolution)
            c.focal_length = list(c0.focal_length)
    cams.compile()
    path = os.path.join(dst, SCENE_PARTS)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('<!-- The cameras "overhead_cam" and "worms_eye_cam" and the table\'s collision box "table" of MuJoCo '
                 'Menagerie aloha/scene.xml (ALOHA 2, BSD-3-Clause: see LICENSE) at commit ' + MENAGERIE_COMMIT[:12] +
                 ', in aloha.xml\'s world frame, copied with their attributes by scripts/bundle_samples.py '
                 '(scene.xml\'s table, frame and camera-mount meshes are not bundled). -->\n' + cams.to_xml())
    a, b = mujoco.MjModel.from_xml_path(path), mujoco.MjModel.from_xml_path(os.path.join(src, 'scene.xml'))
    da, db = mujoco.MjData(a), mujoco.MjData(b)
    for mm, dd in ((a, da), (b, db)):
        mujoco.mj_kinematics(mm, dd)
        mujoco.mj_camlight(mm, dd)                    # camera poses are computed here, not by mj_kinematics
    cam_err = max(float(np.abs(_pose_of(da, n, 'cam') - _pose_of(db, n, 'cam')).max()) for n in ALOHA_FRAME_CAMERAS)
    tab_err = max(float(np.abs(a.geom('table').size - b.geom('table').size).max()),
                  float(np.abs(da.geom('table').xpos - db.geom('table').xpos).max()))
    out.append(f'aloha/{SCENE_PARTS}: {a.ncam} cameras and the table box; against scene.xml {cam_err:.3g} and '
               f'{tab_err:.3g}')
    for side, other in (('left', 'right'), ('right', 'left')):
        spec = _spec_in(dst_root, 'aloha/aloha.xml')
        _delete_all(spec, spec.actuators)
        _delete_all(spec, spec.equalities)
        _delete_all(spec, spec.keys)
        _delete_all(spec, [c for c in spec.cameras if c.parent.name == spec.worldbody.name])
        spec.delete(spec.body(f'{other}/base_link'))
        base = spec.body(f'{side}/base_link')
        base.pos = [0.0, 0.0, 0.0]
        base.quat = [1.0, 0.0, 0.0, 0.0]
        spec.modelname = f'aloha_{side}'
        m = _to_xml(spec, dst_root, compile_only=True)
        qpos = [float(neutral[ref.jnt_qposadr[ref.joint(m.joint(j).name).id]]) for j in range(m.njnt)]
        spec.add_key(name='home', qpos=qpos)
        xml = _to_xml(spec, dst_root)
        head = (f'<!-- The {side} ViperX 300 arm of MuJoCo Menagerie aloha/aloha.xml (ALOHA 2, BSD-3-Clause: see '
                'aloha/LICENSE) at commit ' + MENAGERIE_COMMIT[:12] + '. Written by scripts/bundle_samples.py with '
                f'mujoco.MjSpec: the {other} arm, the actuators, the gripper equality constraints, the keyframes and '
                f'the world cameras deleted; the body "{side}/base_link" moved to this file\'s origin (aloha.xml '
                'places it in its world frame; DexVantage places the robot there with its base pose, read from '
                'aloha.xml); keyframe "home" is this arm\'s part of aloha.xml\'s "neutral_pose". Every other name '
                'and number is aloha.xml\'s. -->\n')
        path = os.path.join(dst_root, f'aloha_{side}.xml')
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(head + xml)
        err = compare_bodies(os.path.join(dst, 'aloha.xml'), 'neutral_pose', path, root=f'{side}/base_link')
        mm = mujoco.MjModel.from_xml_path(path)
        out.append(f'aloha_{side}.xml: {mm.nbody - 1} bodies, {mm.njnt} joints; bodies against aloha.xml in the '
                   f'base frame {err:.3g}')
    return out


# ====================================================================== Franka Panda
def bundle_panda(src_root: str, dst_root: str) -> list:
    import mujoco
    folder = 'franka_emika_panda'
    src, dst = os.path.join(src_root, folder), os.path.join(dst_root, folder)
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(os.path.join(dst, 'assets'))
    shutil.copyfile(os.path.join(src, 'LICENSE'), os.path.join(dst, 'LICENSE'))
    text = open(os.path.join(src, 'panda.xml'), encoding='utf-8').read()
    used = sorted({m.file for m in mujoco.MjSpec.from_file(os.path.join(src, 'panda.xml')).meshes})
    n_obj = 0
    for f in used:
        s = os.path.join(src, 'assets', f)
        if f.endswith('.obj'):
            stl = f[:-4] + '_obj.stl'
            V, F = read_obj(s)
            write_stl(os.path.join(dst, 'assets', stl), V, F, b'binary STL from the MuJoCo Menagerie OBJ (DexVantage)')
            if f'file="{f}"' not in text:
                raise SystemExit(f'panda.xml does not name {f} as file="{f}"')
            # a <mesh> without a name is named after its file: keep that name
            text = text.replace(f'<mesh file="{f}"', f'<mesh name="{f[:-4]}" file="{f}"')
            text = text.replace(f'file="{f}"', f'file="{stl}"')
            n_obj += 1
        else:
            shutil.copyfile(s, os.path.join(dst, 'assets', f))
    with open(os.path.join(dst, 'panda.xml'), 'w', encoding='utf-8') as fh:
        fh.write(NOTE_PANDA + text)
    err = compare_triangles(os.path.join(src, 'panda.xml'), os.path.join(dst, 'panda.xml'))
    return [f'{folder}: {len(used)} meshes ({n_obj} OBJ as STL); largest triangle-vertex difference from the original '
            f'{err:.3g} m']


# ====================================================================== Unitree G1
def _simplify_file(job) -> tuple[int, int, float]:
    """(triangles before, after, Hausdorff distance) of one G1 mesh simplified into the folder job[1]."""
    path, out_dir = job
    V, F = read_stl(path)
    Vs, Fs, err = simplify(V, F, G1_TOLERANCE_M)
    stem = os.path.splitext(os.path.basename(path))[0]
    write_stl(os.path.join(out_dir, f'{stem}.simplified.stl'), Vs, Fs,
              b'MuJoCo Menagerie unitree_g1 mesh, simplified (DexVantage)')
    print(f'  {os.path.basename(path)}: {len(F)} -> {len(Fs)} triangles, Hausdorff {err * 1e3:.3f} mm', flush=True)
    return len(F), len(Fs), err


def bundle_g1(src_root: str, dst_root: str) -> list:
    import mujoco
    folder = 'unitree_g1'
    src, dst = os.path.join(src_root, folder), os.path.join(dst_root, folder)
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(os.path.join(dst, 'assets'))
    shutil.copyfile(os.path.join(src, 'LICENSE'), os.path.join(dst, 'LICENSE'))
    text = open(os.path.join(src, 'g1_with_hands.xml'), encoding='utf-8').read()
    used = sorted({m.file for m in mujoco.MjSpec.from_file(os.path.join(src, 'g1_with_hands.xml')).meshes})
    rows, worst, before, after = [], 0.0, 0, 0
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max(1, min(6, (os.cpu_count() or 2) - 2))) as pool:
        done = list(pool.map(_simplify_file, [(os.path.join(src, 'assets', f), os.path.join(dst, 'assets'))
                                              for f in used]))
    for f, (n0, n1, err) in zip(used, done):
        stem = os.path.splitext(f)[0]
        out = f'{stem}.simplified.stl'
        if f'file="{f}"' not in text:
            raise SystemExit(f'g1_with_hands.xml does not name {f} as file="{f}"')
        text = text.replace(f'<mesh file="{f}"', f'<mesh name="{stem}" file="{f}"')
        text = text.replace(f'file="{f}"', f'file="{out}"')
        worst = max(worst, err)
        before += n0
        after += n1
        rows.append(f'  {f}: {n0} -> {n1} triangles, Hausdorff {err * 1e3:.3f} mm')
    with open(os.path.join(dst, 'g1_with_hands.xml'), 'w', encoding='utf-8') as fh:
        fh.write(NOTE_G1.format(tol=f'{G1_TOLERANCE_M * 1e3:g}', err=f'{worst * 1e3:.3f}') + text)
    return rows + [f'{folder}: {len(used)} meshes, {before} -> {after} triangles; largest Hausdorff distance '
                   f'{worst * 1e3:.3f} mm (tolerance {G1_TOLERANCE_M * 1e3:g} mm)'] + write_g1_upper_body(dst_root)


def write_g1_upper_body(dst_root: str) -> list:
    """g1_upper_body.xml from the bundled unitree_g1/g1_with_hands.xml: legs and waist held at "stand" (module
    docstring). Run alone with the selection 'g1-upper' (it does not touch the meshes)."""
    import mujoco
    folder = 'unitree_g1'
    dst = os.path.join(dst_root, folder)
    full = mujoco.MjModel.from_xml_path(os.path.join(dst, 'g1_with_hands.xml'))
    stand = full.key_qpos[full.key('stand').id]
    spec = _spec_in(dst_root, f'{folder}/g1_with_hands.xml')
    _delete_all(spec, spec.actuators)
    _delete_all(spec, spec.sensors)
    _delete_all(spec, spec.keys)
    for j in list(spec.joints):
        if j.type == mujoco.mjtJoint.mjJNT_FREE:
            pelvis = j.parent
            spec.delete(j)
            pelvis.pos = [float(x) for x in stand[0:3]]
            pelvis.quat = [float(x) for x in stand[3:7]]
    for name in G1_LEG_WAIST:
        j = spec.joint(name)
        a = full.jnt_qposadr[full.joint(name).id]
        if abs(float(stand[a])) > 0:
            raise SystemExit(f'{name} is not 0 at "stand": deleting it would move its body')
        spec.delete(j)
    # Menagerie's collision meshes of the palm (on the wrist yaw link) and the thumb's middle link overlap by
    # 0.27 mm at "stand" (measured with mj_geomDistance on the original files): they are hinged together through the
    # thumb's first link, so the pair is excluded from contact as MuJoCo excludes a parent and child
    for side in ('left', 'right'):
        spec.add_exclude(bodyname1=f'{side}_wrist_yaw_link', bodyname2=f'{side}_hand_thumb_1_link')
    spec.modelname = 'g1_upper_body'
    m = _to_xml(spec, dst_root, compile_only=True)
    qpos = [float(stand[full.jnt_qposadr[full.joint(m.joint(j).name).id]]) for j in range(m.njnt)]
    spec.add_key(name='home', qpos=qpos)
    xml = _to_xml(spec, dst_root)
    head = ('<!-- Unitree G1 with hands, upper body, from MuJoCo Menagerie unitree_g1/g1_with_hands.xml (BSD-3-Clause: '
            'see unitree_g1/LICENSE) at commit ' + MENAGERIE_COMMIT[:12] + ', meshes simplified (see the comment at '
            'the top of unitree_g1/g1_with_hands.xml). Written by scripts/bundle_samples.py with mujoco.MjSpec: the '
            'free joint deleted and the pelvis placed at the "stand" keyframe\'s base pose; the 12 leg joints and 3 '
            'waist joints deleted (all 0 at "stand"), so the legs and waist are held where "stand" puts them; the '
            'actuators, sensors and keyframes deleted; keyframe "home" is the arms\' and hands\' part of "stand"; '
            'contact excluded between each wrist_yaw_link (the palm) and hand_thumb_1_link, whose collision meshes '
            'overlap by 0.27 mm at "stand" in Menagerie\'s own files. Every other name and number is the file\'s. '
            '-->\n')
    path = os.path.join(dst_root, 'g1_upper_body.xml')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(head + xml)
    err = compare_bodies(os.path.join(dst, 'g1_with_hands.xml'), 'stand', path)
    mb = mujoco.MjModel.from_xml_path(path)
    return [f'g1_upper_body.xml: {mb.nbody - 1} bodies, {mb.njnt} joints; bodies against g1_with_hands.xml at '
            f'"stand" {err:.3g}']


def main(argv) -> int:
    if not argv:
        print(__doc__)
        return 2
    src = os.path.abspath(argv[0])
    here = os.path.dirname(os.path.abspath(__file__))
    dst = os.path.abspath(argv[1]) if len(argv) > 1 else os.path.join(here, '..', 'dexvantage', 'resources', 'robots')
    only = set(argv[2].split(',')) if len(argv) > 2 else {'aloha', 'panda', 'g1'}
    os.makedirs(dst, exist_ok=True)
    lines = []
    if 'aloha' in only:
        lines += bundle_aloha(src, dst)
    if 'panda' in only:
        lines += bundle_panda(src, dst)
    if 'g1' in only:
        lines += bundle_g1(src, dst)
    elif 'g1-upper' in only:
        lines += write_g1_upper_body(dst)
    for ln in lines:
        print(ln)
    for folder in ('aloha', 'franka_emika_panda', 'unitree_g1'):
        p = os.path.join(dst, folder)
        if os.path.isdir(p):
            sizes = [os.path.getsize(os.path.join(r, f)) for r, _d, fs in os.walk(p) for f in fs]
            print(f'{folder}: {sum(sizes) / 1e6:.2f} MB, largest file {max(sizes) / 1e6:.2f} MB')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
