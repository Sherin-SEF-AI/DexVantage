"""Copy the MuJoCo Menagerie Shadow Hand into dexvantage/resources/robots/ and join it to the bundled UR5e (spec §1.2,
§6 dexterous hands, §14).

    git clone --filter=blob:none --no-checkout --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
    cd mujoco_menagerie && git sparse-checkout set --no-cone /shadow_hand/ && git checkout
    python -I scripts/bundle_dexhand.py <that checkout> [dexvantage/resources/robots]

Run it only after reading shadow_hand/LICENSE (Apache-2.0 at commit 98a70c5, read 2026-11-06) and recording it in
THIRD_PARTY.md. The UR5e folder must already be bundled (scripts/bundle_menagerie.py). What it writes:

- shadow_hand/: right_hand.xml, keyframes.xml (the folder's named hand postures, read by core/dexhand.py), LICENSE
  and the 13 OBJ meshes right_hand.xml uses, byte for byte (3.4 MB; no conversion: OBJ is small here). The folder's
  mounting_plate.obj is used by no model file and is left out, as are left_hand.xml, the scenes, the README, the
  CHANGELOG and the gallery picture.
- ur5e_shadow_hand.xml: the UR5e with the right Shadow Hand on its flange, written by MuJoCo itself:
  MjSpec.attach(hand, site=<the UR5e's "attachment_site">, prefix="hand_") and MjSpec.to_xml(), with the mesh paths
  pointing into the two model folders. The prefix is needed: both files name meshes "forearm_0". One change is made
  before attaching, and the file's top comment says so: the hand file turns its forearm body by quat="0 1 0 1" (90°
  about y, so the stand-alone hand lies along x); here that rotation is set to identity, so the forearm's own z axis
  continues the flange's z axis and the hand points out of the flange. The arm's <option> is kept (MuJoCo reports
  that the hand's cone and impratio differ; kinematics uses neither). The home keyframe is the UR5e's, with the hand's
  joints at 0 (the "open hand" keyframe of keyframes.xml).

The triangles are checked: the stand-alone hand's meshes, compiled by MuJoCo at q = 0, are compared with the joined
model's hand meshes at the UR5e's home pose after removing the flange transform (the largest corner difference is
printed; float32 rounding only). The script only reads the checkout; it never changes it.
"""
from __future__ import annotations

import os
import shutil
import sys
import warnings

import numpy as np

HAND = 'shadow_hand'
ARM = 'universal_robots_ur5e'
FILES = ('right_hand.xml', 'keyframes.xml', 'LICENSE')
COMBINED = 'ur5e_shadow_hand.xml'
PREFIX = 'hand_'


def copy_hand(src_root: str, dst_root: str) -> None:
    src, dst = os.path.join(src_root, HAND), os.path.join(dst_root, HAND)
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(os.path.join(dst, 'assets'))
    for f in FILES:
        shutil.copyfile(os.path.join(src, f), os.path.join(dst, f))
    import mujoco
    used = {m.file for m in mujoco.MjSpec.from_file(os.path.join(src, 'right_hand.xml')).meshes}
    for f in sorted(used):
        shutil.copyfile(os.path.join(src, 'assets', f), os.path.join(dst, 'assets', f))


def write_combined(dst_root: str) -> str:
    import mujoco

    def read(folder, main):
        return mujoco.MjSpec.from_string(open(os.path.join(dst_root, folder, main), encoding='utf-8').read())
    arm, hand = read(ARM, 'ur5e.xml'), read(HAND, 'right_hand.xml')
    for mesh in arm.meshes:
        mesh.file = f'{ARM}/assets/{mesh.file}'
    for mesh in hand.meshes:
        mesh.file = f'{HAND}/assets/{mesh.file}'
    arm.meshdir = ''
    hand.meshdir = ''
    hand.body('rh_forearm').quat = [1.0, 0.0, 0.0, 0.0]
    cwd = os.getcwd()
    try:
        os.chdir(dst_root)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')        # the option conflict (cone, impratio) described in the docstring
            arm.attach(hand, site=arm.site('attachment_site'), prefix=PREFIX)
            arm.modelname = 'ur5e_shadow_hand'
            arm.compile()
            xml = arm.to_xml()
    finally:
        os.chdir(cwd)
        log = os.path.join(dst_root, 'MUJOCO_LOG.TXT')
        if os.path.exists(log):
            os.remove(log)
    head = ('<!-- UR5e with the right Shadow Hand E3M5 on its flange, from MuJoCo Menagerie (universal_robots_ur5e, '
            'BSD-3-Clause; shadow_hand, Apache-2.0: see the LICENSE files in the two folders). Written by '
            'scripts/bundle_dexhand.py with mujoco.MjSpec.attach(hand, site="attachment_site", '
            f'prefix="{PREFIX}") and MjSpec.to_xml(); the hand\'s names carry the prefix. One change before '
            'attaching: the hand file\'s forearm rotation quat="0 1 0 1" is set to identity, so the forearm axis '
            'continues the flange axis. -->\n')
    path = os.path.join(dst_root, COMBINED)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(head + xml)
    return path


def _hand_triangles(m, d, prefix: str, inv=None) -> dict:
    """mesh-geom triangles of the hand's bodies (names without the prefix) in the hand's forearm frame."""
    import mujoco
    out = {}
    fa = next(b for b in range(m.nbody) if m.body(b).name == prefix + 'rh_forearm')
    Tf = np.eye(4)
    Tf[:3, :3], Tf[:3, 3] = d.xmat[fa].reshape(3, 3), d.xpos[fa]
    Ti = np.linalg.inv(Tf)
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        body = m.body(int(m.geom_bodyid[g])).name
        if not body.startswith(prefix + 'rh_'):
            continue
        k = m.geom_dataid[g]
        va, vn, fa_, fn = m.mesh_vertadr[k], m.mesh_vertnum[k], m.mesh_faceadr[k], m.mesh_facenum[k]
        V = m.mesh_vert[va:va + vn].astype(float)
        F = m.mesh_face[fa_:fa_ + fn]
        T = V[F] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        T = T @ Ti[:3, :3].T + Ti[:3, 3]
        out.setdefault(body[len(prefix):], []).append(T)
    return out


def compare(dst_root: str, combined: str) -> float:
    import mujoco
    a = mujoco.MjModel.from_xml_path(os.path.join(dst_root, HAND, 'right_hand.xml'))
    da = mujoco.MjData(a)
    mujoco.mj_kinematics(a, da)
    b = mujoco.MjModel.from_xml_path(combined)
    db = mujoco.MjData(b)
    db.qpos[:] = b.key_qpos[0]
    mujoco.mj_kinematics(b, db)
    A, B = _hand_triangles(a, da, ''), _hand_triangles(b, db, PREFIX)
    if sorted(A) != sorted(B):
        raise SystemExit('the joined model has different hand bodies than the hand file')
    worst = 0.0
    for k in A:
        for x, y in zip(A[k], B[k]):
            worst = max(worst, float(np.abs(x - y).max()))
    return worst


def main(argv) -> int:
    if not argv:
        print(__doc__)
        return 2
    src = os.path.abspath(argv[0])
    here = os.path.dirname(os.path.abspath(__file__))
    dst = os.path.abspath(argv[1]) if len(argv) > 1 else os.path.join(here, '..', 'dexvantage', 'resources', 'robots')
    if not os.path.isfile(os.path.join(dst, ARM, 'ur5e.xml')):
        raise SystemExit(f'{ARM} is not bundled in {dst}: run scripts/bundle_menagerie.py first.')
    copy_hand(src, dst)
    path = write_combined(dst)
    import mujoco
    m = mujoco.MjModel.from_xml_path(path)
    print(f'{COMBINED}: {m.nbody - 1} bodies, {m.njnt} joints, {m.nmesh} meshes')
    print(f'hand triangles against the stand-alone hand file: largest corner difference {compare(dst, path):.3g} m')
    total = sum(os.path.getsize(os.path.join(r, f)) for r, _d, fs in os.walk(os.path.join(dst, HAND)) for f in fs)
    print(f'{HAND}: {total / 1e6:.1f} MB')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
