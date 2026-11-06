"""Copy the bundled MuJoCo Menagerie robots into dexvantage/resources/robots/ (spec §1.2, §14).

    git clone --filter=blob:none --no-checkout --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
    cd mujoco_menagerie && git sparse-checkout set --no-cone /trs_so_arm100/ /universal_robots_ur5e/ /robotiq_2f85/
    git checkout
    python -I scripts/bundle_menagerie.py <that checkout> [dexvantage/resources/robots]

Run it only after reading each model's own LICENSE (Menagerie licences differ per model) and recording it in
THIRD_PARTY.md. What it writes:

- trs_so_arm100/, robotiq_2f85/: the model's MJCF, its LICENSE and its meshes, byte for byte.
- universal_robots_ur5e/: the MJCF and LICENSE, with the 20 OBJ meshes (30 MB of text) rewritten as binary STL (6 MB)
  so the package stays small and no file is over 2 MB. The triangles are the same: the script compiles the original
  and the converted model with MuJoCo and compares every triangle of every geom at the home pose (the result is
  printed; it was 0 m for all three folders on 2026-11-06: MuJoCo keeps mesh vertices as float32 either way). The
  MJCF's <mesh file> names change from .obj to .stl and a comment at its top says so. Nothing else in the file changes.
- ur5e_robotiq_2f85.xml: the UR5e with the 2F-85 on its flange, written by MuJoCo itself: MjSpec.attach(gripper,
  site=ur5e's "attachment_site", prefix="gripper_") and MjSpec.to_xml(), with the mesh paths pointing into the two
  model folders. The gripper's names get the prefix (its "base" body would clash with the UR5e's). The arm's
  <option> is kept (MuJoCo reports that the gripper's cone and impratio differ; kinematics uses neither). The home
  keyframe is the UR5e's, with the gripper's joints at 0 (open).

The script only reads the checkout; it never changes it.
"""
from __future__ import annotations

import os
import shutil
import struct
import sys
import warnings

import numpy as np

MODELS = {                          # folder -> (main MJCF, files copied besides the LICENSE and assets/)
    'trs_so_arm100': 'so_arm100.xml',
    'universal_robots_ur5e': 'ur5e.xml',
    'robotiq_2f85': '2f85.xml',
}
COMBINED = 'ur5e_robotiq_2f85.xml'
PREFIX = 'gripper_'
NOTE_UR5E = ('<!-- DexVantage: the <mesh> files are the Menagerie OBJ meshes rewritten as binary STL (the same '
             'triangles, float32), by scripts/bundle_menagerie.py; nothing else in this file was changed. -->\n')


def obj_to_stl(src: str, dst: str) -> int:
    """Write the triangles of an OBJ file (v and f records; polygons fanned) as binary STL; returns the count."""
    verts, tris = [], []
    with open(src, encoding='utf-8') as fh:
        for line in fh:
            if line.startswith('v '):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '):
                idx = [int(t.split('/')[0]) for t in line.split()[1:]]
                idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                for k in range(1, len(idx) - 1):
                    tris.append((idx[0], idx[k], idx[k + 1]))
    V = np.asarray(verts, np.float64)
    T = V[np.asarray(tris, int)]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)
    rec = np.zeros(len(T), dtype=[('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
    rec['n'], rec['v'] = n, T
    with open(dst, 'wb') as fh:
        fh.write(b'binary STL from the MuJoCo Menagerie OBJ (DexVantage)'.ljust(80, b' '))
        fh.write(struct.pack('<I', len(T)))
        fh.write(rec.tobytes())
    return len(T)


def world_triangles(path: str) -> list:
    """Per mesh geom, its triangles in the world at the model's home keyframe (rows sorted, for comparison)."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    if m.nkey:
        d.qpos[:] = m.key_qpos[0]
    mujoco.mj_kinematics(m, d)
    out = []
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        k = m.geom_dataid[g]
        va, vn, fa, fn = m.mesh_vertadr[k], m.mesh_vertnum[k], m.mesh_faceadr[k], m.mesh_facenum[k]
        V = m.mesh_vert[va:va + vn].astype(float)
        F = m.mesh_face[fa:fa + fn]
        T = V[F] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]          # (n, 3, 3)
        key = T.round(7)                                                     # order vertices, then triangles
        o = np.lexsort((key[:, :, 2], key[:, :, 1], key[:, :, 0]), axis=-1)
        T = np.take_along_axis(T, o[:, :, None], axis=1).reshape(-1, 9)
        out.append(T[np.lexsort(T.round(7).T[::-1])])
    return out


def compare(a: str, b: str) -> float:
    A, B = world_triangles(a), world_triangles(b)
    if len(A) != len(B) or any(x.shape != y.shape for x, y in zip(A, B)):
        raise SystemExit(f'{os.path.basename(b)}: a different number of geoms or triangles than the original')
    return max(float(np.abs(x - y).max()) for x, y in zip(A, B))


def copy_model(src_root: str, dst_root: str, folder: str) -> None:
    src, dst = os.path.join(src_root, folder), os.path.join(dst_root, folder)
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(os.path.join(dst, 'assets'))
    shutil.copyfile(os.path.join(src, 'LICENSE'), os.path.join(dst, 'LICENSE'))
    main = MODELS[folder]
    text = open(os.path.join(src, main), encoding='utf-8').read()
    for f in sorted(os.listdir(os.path.join(src, 'assets'))):
        s = os.path.join(src, 'assets', f)
        if f.endswith('.obj'):
            stl = f[:-4] + '.stl'
            obj_to_stl(s, os.path.join(dst, 'assets', stl))
            text = text.replace(f'file="{f}"', f'file="{stl}"')
        elif f.endswith('.stl'):
            shutil.copyfile(s, os.path.join(dst, 'assets', f))
    if folder == 'universal_robots_ur5e':
        text = NOTE_UR5E + text
    with open(os.path.join(dst, main), 'w', encoding='utf-8') as fh:
        fh.write(text)


def write_combined(dst_root: str) -> str:
    import mujoco
    arm_dir, g_dir = 'universal_robots_ur5e', 'robotiq_2f85'
    def read(folder):                              # from a string: file names resolve against the working folder
        return mujoco.MjSpec.from_string(open(os.path.join(dst_root, folder, MODELS[folder]), encoding='utf-8').read())
    arm, grip = read(arm_dir), read(g_dir)
    for mesh in arm.meshes:
        mesh.file = f'{arm_dir}/assets/{mesh.file}'
    for mesh in grip.meshes:
        mesh.file = f'{g_dir}/assets/{mesh.file}'
    arm.meshdir = ''
    grip.meshdir = ''
    cwd = os.getcwd()
    try:
        os.chdir(dst_root)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')        # the option conflict (cone, impratio) described in the docstring
            arm.attach(grip, site=arm.site('attachment_site'), prefix=PREFIX)
            arm.modelname = 'ur5e_robotiq_2f85'
            arm.compile()
            xml = arm.to_xml()
    finally:
        os.chdir(cwd)
        log = os.path.join(dst_root, 'MUJOCO_LOG.TXT')       # MuJoCo logs the attach warning into the working folder
        if os.path.exists(log):
            os.remove(log)
    head = ('<!-- UR5e with the Robotiq 2F-85 on its flange, from MuJoCo Menagerie (universal_robots_ur5e, '
            'BSD-3-Clause; robotiq_2f85, BSD-2-Clause: see the LICENSE files in the two folders). Written by '
            'scripts/bundle_menagerie.py with mujoco.MjSpec.attach(gripper, site="attachment_site", '
            f'prefix="{PREFIX}") and MjSpec.to_xml(); the gripper\'s names carry the prefix. -->\n')
    path = os.path.join(dst_root, COMBINED)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(head + xml)
    return path


def main(argv) -> int:
    if not argv:
        print(__doc__)
        return 2
    src = os.path.abspath(argv[0])
    here = os.path.dirname(os.path.abspath(__file__))
    dst = os.path.abspath(argv[1]) if len(argv) > 1 else os.path.join(here, '..', 'dexvantage', 'resources', 'robots')
    os.makedirs(dst, exist_ok=True)
    for folder in MODELS:
        copy_model(src, dst, folder)
        err = compare(os.path.join(src, folder, MODELS[folder]), os.path.join(dst, folder, MODELS[folder]))
        print(f'{folder}: copied; largest triangle-vertex difference from the original {err:.3g} m')
    path = write_combined(dst)
    import mujoco
    m = mujoco.MjModel.from_xml_path(path)
    print(f'{COMBINED}: {m.nbody - 1} bodies, {m.njnt} joints, {m.nmesh} meshes')
    total = sum(os.path.getsize(os.path.join(r, f)) for r, _d, fs in os.walk(dst) for f in fs)
    print(f'{dst}: {total / 1e6:.1f} MB')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
