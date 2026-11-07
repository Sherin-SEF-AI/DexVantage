"""Checks of robot import and export (model 'robot_io'): the exported URDF loads in MuJoCo as written and poses every
sensor frame like DexVantage, with the robots' inertials; the USD import round trip of three bundled robots and a
third party's UR5e against DexVantage's kinematics; the OpenVLA encoding choice against the pinned sources' rules;
the Isaac export's off-centre principal point against Isaac Lab's intrinsics rule.

    urdf_mujoco_errors(folder, rng, samples, draws) -> dict
    usd_roundtrip_errors(folder, rng, samples, draws) -> dict
    usd_third_party_errors(rng, draws) -> dict
    openvla_errors(folder) -> dict
    isaac_principal_point_errors(folder) -> dict

Each reference is independent of the code it checks: MuJoCo's own URDF compiler (not DexVantage's) poses the URDF;
the robots' MuJoCo models compiled from their own files give the kinematics and inertials the round trips must
reproduce; the third-party USD was written by another program (the mujoco-usd-converter); the OpenVLA rules are
restated here from the pinned source lines, and the written entry is executed against the enum classes copied
verbatim from them; Isaac Lab's intrinsics rule is written out again from its camera.py. The USD checks are
registered only when usd-core (pxr) is installed, as formats.bracket_step is only with gmsh.
"""
from __future__ import annotations

import importlib.util
import math
import os

import numpy as np

from .suite import Outcome, check

HAVE_PXR = importlib.util.find_spec('pxr') is not None
URDF_SAMPLES = ('so100', 'ur5e', 'panda-bimanual')
USD_SAMPLES = ('so100', 'ur5e', 'panda-bimanual')
DRAWS = 50
THIRD_PARTY = ('usd_robots', 'newton_ur5e', 'usd_structured', 'ur5e.usda')
UR5E_ARM = ('base', 'shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link')

# OpenVLA's enums at commit c8f03f4, prismatic/vla/datasets/rlds/oxe/configs.py L33-50, copied verbatim (OpenVLA-OFT's
# configs.py has the same lines at e4287e9); the written entry is executed against them
OPENVLA_ENUMS = '''
from enum import IntEnum


class StateEncoding(IntEnum):
    # fmt: off
    NONE = -1               # No Proprioceptive State
    POS_EULER = 1           # EEF XYZ (3) + Roll-Pitch-Yaw (3) + <PAD> (1) + Gripper Open/Close (1)
    POS_QUAT = 2            # EEF XYZ (3) + Quaternion (4) + Gripper Open/Close (1)
    JOINT = 3               # Joint Angles (7, <PAD> if fewer) + Gripper Open/Close (1)
    JOINT_BIMANUAL = 4      # Joint Angles (2 x [ Joint Angles (6) + Gripper Open/Close (1) ])
    # fmt: on


class ActionEncoding(IntEnum):
    # fmt: off
    EEF_POS = 1             # EEF Delta XYZ (3) + Roll-Pitch-Yaw (3) + Gripper Open/Close (1)
    JOINT_POS = 2           # Joint Delta Position (7) + Gripper Open/Close (1)
    JOINT_POS_BIMANUAL = 3  # Joint Delta Position (2 x [ Joint Delta Position (6) + Gripper Open/Close (1) ])
    EEF_R6 = 4              # EEF Delta XYZ (3) + R6 (6) + Gripper Open/Close (1)
    # fmt: on
'''
# the loaders' accepted action encodings (OpenVLA materialize.py L32-33; OpenVLA-OFT materialize.py L32-33)
LOADER_ACCEPTS = {'openvla': {'EEF_POS', 'EEF_R6'}, 'openvla-oft': {'EEF_POS', 'EEF_R6', 'JOINT_POS_BIMANUAL'}}


def _pose_err(A, B) -> tuple[float, float]:
    A, B = np.asarray(A, float), np.asarray(B, float)
    R = A[:3, :3].T @ B[:3, :3]
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 4
    ang = 2 * math.atan2(float(np.linalg.norm(v)), w) if w > 1e-6 else \
        math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    return float(np.linalg.norm(A[:3, 3] - B[:3, 3])), ang


def _random_q(rm, rng) -> np.ndarray:
    lo = np.where(np.isfinite(rm.lower), rm.lower, -math.pi)
    hi = np.where(np.isfinite(rm.upper), rm.upper, math.pi)
    return lo + (hi - lo) * rng.uniform(0.0, 1.0, rm.nq)


# ====================================================================== the URDF in MuJoCo
def urdf_mujoco_errors(folder: str, rng, samples=URDF_SAMPLES, draws: int = 10) -> dict:
    """Each sample's rig.urdf loaded by mujoco.MjModel.from_xml_path as written; every sensor body and optical frame
    posed by mj_kinematics at random joint states against DexVantage (camrender.optical_pose,
    scenegeom.sensor_world_pose); each robot body's mass, centre of mass and inertia tensor (link frame) against the
    robot's own compiled model (relative), the minimal placeholders against their stated value."""
    import mujoco

    from ..core import camrender as CR
    from ..core import exporters as X
    from ..core.exporters import _common as C
    from ..core.exporters import ros2 as R2
    from ..core.robot.model import get_model
    from ..core.robot.samples import sample_project
    from ..core.scenegeom import sensor_world_pose
    out = {'position_m': 0.0, 'rotation_rad': 0.0, 'mass_rel': 0.0, 'com_m': 0.0, 'inertia_rel': 0.0,
           'placeholder_mass_err': 0.0, 'load_errors': [], 'samples': {}}
    for name in samples:
        p = sample_project(name)
        sub = os.path.join(folder, f'urdf_{name}')
        man = X.export(p, sub, ['urdf'])
        if man['formats']['urdf']['status'] != 'ok':
            out['load_errors'].append(f'{name}: export refused: {man["formats"]["urdf"].get("reason")}')
            continue
        path = os.path.join(sub, 'ros2', 'rig.urdf')
        try:
            m = mujoco.MjModel.from_xml_path(path)
        except Exception as e:                         # noqa: BLE001 — the failure is the finding
            out['load_errors'].append(f'{name}: MuJoCo refused rig.urdf: {str(e).splitlines()[0]}')
            continue
        d = mujoco.MjData(m)
        many = len(p.robots) > 1
        frames = 0
        for _k in range(draws):
            q = {}
            d.qpos[:] = 0
            for r in p.robots:
                rm = get_model(p, r.uid)
                q[r.uid] = _random_q(rm, rng)
                pre = f'{C.frame_name(r.name)}_' if many else ''
                for jn, v in zip(rm.joint_names, q[r.uid]):
                    d.qpos[m.joint(pre + jn).qposadr[0]] = v
            mujoco.mj_kinematics(m, d)
            for c in C.rig_cameras(p, q):
                for frame, ref in ((c.optical_frame, CR.optical_pose(p, c.sensor, c.imager, q).matrix()),
                                   (c.sensor_frame, sensor_world_pose(p, c.sensor, q).matrix())):
                    b = m.body(frame).id
                    T = np.eye(4)
                    T[:3, :3] = d.xmat[b].reshape(3, 3)
                    T[:3, 3] = d.xpos[b]
                    e = _pose_err(ref, T)
                    out['position_m'] = max(out['position_m'], e[0])
                    out['rotation_rad'] = max(out['rotation_rad'], e[1])
                    frames += 1
        for r in p.robots:
            rm = get_model(p, r.uid)
            src = R2.inertial_sources(p, r, rm)
            pre = f'{C.frame_name(r.name)}_' if many else ''
            for b in range(1, rm.m.nbody):
                bb = m.body(pre + rm.body_link[b]).id
                if src[b] == 'none':
                    out['placeholder_mass_err'] = max(out['placeholder_mass_err'], abs(m.body_mass[bb] - R2.MIN_MASS))
                    continue
                ma, mb = float(rm.m.body_mass[b]), float(m.body_mass[bb])
                out['mass_rel'] = max(out['mass_rel'], abs(ma - mb) / ma)
                out['com_m'] = max(out['com_m'], float(np.linalg.norm(rm.m.body_ipos[b] - m.body_ipos[bb])))

                def tensor(mm, k):
                    from ..core.frames import quat_from_mujoco, quat_to_matrix
                    R = quat_to_matrix(quat_from_mujoco(mm.body_iquat[k]))
                    return R @ np.diag(mm.body_inertia[k]) @ R.T
                Ia, Ib = tensor(rm.m, b), tensor(m, bb)
                out['inertia_rel'] = max(out['inertia_rel'], float(np.linalg.norm(Ia - Ib) / np.linalg.norm(Ia)))
        out['samples'][name] = {'bodies': int(m.nbody), 'frames_compared': frames,
                                'subtree_mass_kg': float(m.body_subtreemass[0])}
    return out


@check('robot_io.urdf_mujoco', 'robot_io', 'Exported URDF loads in MuJoCo as written, with the robots\' inertials',
       'largest distance (m) between every sensor body and optical frame of the exported rig.urdf, loaded '
       'unchanged by mujoco.MjModel.from_xml_path and posed by mj_kinematics at 10 random joint states, and the '
       'same frame from DexVantage; also every rotation within 10⁻⁶ rad, the file loads, and each robot body\'s '
       'mass and inertia tensor within 10⁻⁹ (relative) and centre of mass within 10⁻⁹ m of the robot\'s own '
       'MuJoCo model; the SO-100, UR5e and bimanual Panda samples',
       'camrender.optical_pose and scenegeom.sensor_world_pose; MuJoCo\'s URDF compiler; the robots\' MuJoCo '
       'models compiled from their own files (body_mass, body_ipos, body_iquat, body_inertia)', 1e-6, 'm',
       refs=('mujoco_urdf', 'urdfdom', 'urdf'), cost='study')
def _urdf_mujoco(ctx):
    d = urdf_mujoco_errors(os.path.join(ctx.tmp(), 'robot_io_urdf'), ctx.rng('robot_io.urdf_mujoco'),
                           URDF_SAMPLES if not ctx.quick else URDF_SAMPLES[:2], 10 if not ctx.quick else 4)
    ok = (not d['load_errors'] and d['rotation_rad'] <= 1e-6 and d['mass_rel'] <= 1e-9 and d['com_m'] <= 1e-9
          and d['inertia_rel'] <= 1e-9 and d['placeholder_mass_err'] <= 1e-12)
    return Outcome(d['position_m'] if not d['load_errors'] else float('inf'), d, ok=ok,
                   note='10⁻⁶ m is the round trip\'s stated requirement; the file writes 17 significant digits, '
                        'so the error is rounding (measured 2026-11-07: 5.3 × 10⁻¹⁶ m) and a wrong joint anchor or '
                        'frame would be millimetres. '
                        'Inertials: MuJoCo\'s URDF compiler diagonalises the written full tensor again, so 10⁻⁹ '
                        'relative leaves room for its eigen-decomposition only. Before this export wrote '
                        'inertials, MuJoCo refused the file ("mass and inertia of moving bodies must be larger '
                        'than mjMINVAL").')


# ====================================================================== USD round trip
def usd_roundtrip_errors(folder: str, rng, samples=USD_SAMPLES, draws: int = DRAWS) -> dict:
    """Each sample's first robot written by core/robot/usd_writer and imported back by the USD importer
    (core/robot/importer.import_robot): every MuJoCo body of the source against the imported link of the same name
    over draws random joint states (base removed); the joint limits against the float32 the schema stores (exact) and
    the source's radians (within that float32's spacing); the writer's body poses as pxr composes them against
    DexVantage at the reference state; masses (relative)."""
    from pxr import Usd, UsdGeom, UsdPhysics

    from ..core.robot import import_robot
    from ..core.robot.model import RobotModel, get_model
    from ..core.robot.samples import sample_project
    from ..core.robot.usd_writer import usd_name, write_robot_usd
    out = {'position_m': 0.0, 'rotation_rad': 0.0, 'limit_rad': 0.0, 'limit_bound_rad': 0.0, 'limit_mismatches': [],
           'writer_pose_m': 0.0, 'mass_rel': 0.0, 'missing': [], 'samples': {}}
    for name in samples:
        p = sample_project(name)
        r = p.robots[0]
        rm = get_model(p, r.uid)
        path = os.path.join(folder, f'{name}.usda')
        os.makedirs(folder, exist_ok=True)
        if os.path.exists(path):
            os.remove(path)
        info = write_robot_usd(p, r.uid, path)
        # the writer's poses as pxr composes them, against DexVantage at the reference state (q = qpos0 = 0 here)
        st = Usd.Stage.Open(path)
        xc = UsdGeom.XformCache()
        T0 = rm.fk(np.zeros(rm.nq))
        Bi = np.linalg.inv(rm.base)
        for ln, prim_name in info['bodies'].items():
            M = np.asarray(xc.GetLocalToWorldTransform(st.GetPrimAtPath(f'{info["root"]}/{prim_name}')), float).T
            out['writer_pose_m'] = max(out['writer_pose_m'], _pose_err(Bi @ T0[ln], M)[0])
        stored = {}
        for prim in st.Traverse():
            if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
                stored[prim.GetName()] = (prim.GetAttribute('physics:lowerLimit').Get(),
                                          prim.GetAttribute('physics:upperLimit').Get(),
                                          prim.IsA(UsdPhysics.RevoluteJoint))
        robot, sources = import_robot(path)
        im = RobotModel(robot, sources)
        jmap = []
        for jn in rm.joint_names:
            k = usd_name(jn)
            if k not in im.joint_names:
                out['missing'].append(f'{name}: joint {jn}')
                jmap.append(None)
            else:
                jmap.append(im.joint_names.index(k))
        links = [ln for ln in rm.body_link[1:] if usd_name(ln) in im.link_names]
        out['missing'] += [f'{name}: link {ln}' for ln in rm.body_link[1:] if usd_name(ln) not in im.link_names]
        if None in jmap:
            continue
        for _k in range(draws):
            q = _random_q(rm, rng)
            q2 = np.zeros(im.nq)
            q2[jmap] = q
            A, B = rm.fk(q), im.fk(q2)
            for ln in links:
                e = _pose_err(Bi @ A[ln], B[usd_name(ln)])
                out['position_m'] = max(out['position_m'], e[0])
                out['rotation_rad'] = max(out['rotation_rad'], e[1])
        for i, jn in enumerate(rm.joint_names):
            lo, hi, rev = stored.get(usd_name(jn), (None, None, True))
            f = math.pi / 180 if rev else 1.0
            for src_v, got, file_v in ((rm.lower[i], im.lower[jmap[i]], lo), (rm.upper[i], im.upper[jmap[i]], hi)):
                if not np.isfinite(src_v):
                    if np.isfinite(got):
                        out['limit_mismatches'].append(f'{name}: {jn} gained a limit')
                    continue
                if file_v is None or float(file_v) * f != got:
                    out['limit_mismatches'].append(f'{name}: {jn} {got!r} is not the stored float32 {file_v!r}')
                out['limit_rad'] = max(out['limit_rad'], abs(float(got) - float(src_v)))
                spacing = float(np.spacing(np.float32(abs(src_v) / f))) * f
                out['limit_bound_rad'] = max(out['limit_bound_rad'], spacing)
                if abs(float(got) - float(src_v)) > spacing:
                    out['limit_mismatches'].append(f'{name}: {jn} {got!r} vs {src_v!r} beyond float32 rounding')
        for ln in links:
            b, b2 = rm.m.body(ln).id, im.m.body(usd_name(ln)).id
            if rm.m.body_mass[b] > 0:
                out['mass_rel'] = max(out['mass_rel'], abs(rm.m.body_mass[b] - im.m.body_mass[b2]) /
                                      rm.m.body_mass[b])
        out['samples'][name] = {'links': len(links), 'joints': rm.nq, 'geoms_written': info['geoms'],
                                'geoms_imported': int(im.m.ngeom), 'skipped': info['skipped'], 'notes': im.notes}
    return out


def usd_third_party_errors(rng, draws: int = DRAWS) -> dict:
    """The bundled newton-assets UR5e (written by the mujoco-usd-converter from Menagerie's ur5e.xml) imported by
    DexVantage's USD importer, against DexVantage's UR5e sample (Menagerie's ur5e.xml through MuJoCo): the seven arm
    bodies over draws random arm states (the gripper at zero), the six joint limits and the arm bodies' masses."""
    from ..core.robot import import_robot
    from ..core.robot.model import RobotModel, get_model
    from ..core.robot.samples import sample_project
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, 'resources', *THIRD_PARTY)
    robot, sources = import_robot(path)
    im = RobotModel(robot, sources)
    p = sample_project('ur5e')
    rm = get_model(p, p.robots[0].uid)
    Bi = np.linalg.inv(rm.base)
    arm_j = [rm.joint_names.index(j) for j in im.joint_names]
    out = {'position_m': 0.0, 'rotation_rad': 0.0, 'limit_rad': 0.0, 'mass_rel': 0.0, 'joints': im.joint_names,
           'notes': robot.source.note}
    for _k in range(draws):
        q = _random_q(im, rng)
        qa = np.zeros(rm.nq)
        qa[arm_j] = q
        A, B = rm.fk(qa), im.fk(q)
        for ln in UR5E_ARM:
            e = _pose_err(Bi @ A[ln], B[ln])
            out['position_m'] = max(out['position_m'], e[0])
            out['rotation_rad'] = max(out['rotation_rad'], e[1])
    for i, j in enumerate(arm_j):
        out['limit_rad'] = max(out['limit_rad'], abs(im.lower[i] - rm.lower[j]), abs(im.upper[i] - rm.upper[j]))
    for ln in UR5E_ARM:
        a, b = rm.m.body_mass[rm.m.body(ln).id], im.m.body_mass[im.m.body(ln).id]
        out['mass_rel'] = max(out['mass_rel'], abs(a - b) / a)
    return out


if HAVE_PXR:
    @check('robot_io.usd_roundtrip', 'robot_io', 'USD robot import: bundled robots written to USD and read back',
           'largest distance (m) between every link of the SO-100, UR5e (with the 2F-85) and Panda samples and the '
           'same link of the robot written to a UsdPhysics articulation (core/robot/usd_writer.py) and imported '
           'back (core/robot/usd_import.py), over 50 random joint states; also every rotation within 10⁻⁶ rad, the '
           'joint limits equal to the float32 the file stores and within its spacing of the source, the writer\'s '
           'body poses as pxr composes them within 10⁻⁶ m, masses within 10⁻⁶ relative',
           'the robots\' MuJoCo models compiled from their own files (mj_kinematics); pxr\'s XformCache for the '
           'written stage', 1e-6, 'm', refs=('openusd_physics', 'mujoco_menagerie'), cost='study')
    def _usd_roundtrip(ctx):
        d = usd_roundtrip_errors(os.path.join(ctx.tmp(), 'robot_io_usd'), ctx.rng('robot_io.usd_roundtrip'),
                                 USD_SAMPLES, DRAWS if not ctx.quick else 10)
        ok = (d['rotation_rad'] <= 1e-6 and not d['limit_mismatches'] and not d['missing']
              and d['writer_pose_m'] <= 1e-6 and d['mass_rel'] <= 1e-6)
        return Outcome(d['position_m'], d, ok=ok,
                       note='UsdPhysics stores joint frames as point3f and quatf and limits as float (32-bit): about '
                            '6 × 10⁻⁸ relative, so a chain of a dozen joints within a metre lands within a few '
                            '10⁻⁸ m (measured 2026-11-07: 2.5 × 10⁻⁸ m, 1.0 × 10⁻⁷ rad); 10⁻⁶ m and rad leave that '
                            'margin and sit far below a convention error (a sign of the axis, an anchor or a frame '
                            'swap is millimetres or degrees). "Exact" limits: the imported limit equals the stored '
                            'float32 decoded (degrees × π/180), which is within one float32 spacing of the source.')

    @check('robot_io.usd_third_party', 'robot_io', 'USD robot import: a third party\'s UR5e against DexVantage\'s',
           'largest distance (m) between the seven arm bodies of the newton-assets UR5e (a UsdPhysics articulation '
           'written by the mujoco-usd-converter from Menagerie\'s ur5e.xml), imported by DexVantage, and the same '
           'bodies of DexVantage\'s UR5e sample over 50 random arm states; also rotations within 10⁻⁶ rad, the six '
           'joint limits within 10⁻⁶ rad and the arm masses within 10⁻⁶ relative',
           'Menagerie\'s ur5e.xml compiled by MuJoCo (the UR5e sample); the third party\'s converter, not '
           'DexVantage\'s writer, made the file', 1e-6, 'm', refs=('newton_assets_ur5e', 'openusd_physics'),
           cost='study')
    def _usd_third_party(ctx):
        d = usd_third_party_errors(ctx.rng('robot_io.usd_third_party'), DRAWS if not ctx.quick else 10)
        ok = d['rotation_rad'] <= 1e-6 and d['limit_rad'] <= 1e-6 and d['mass_rel'] <= 1e-6
        return Outcome(d['position_m'], d, ok=ok,
                       note='An independent writer: a sign, axis or frame convention the importer and DexVantage\'s '
                            'writer shared by mistake would show here. The file stores float32 frames and limits '
                            '(degrees), hence 10⁻⁶. Its mesh library (6.3 MB) is not bundled, so its visual meshes are '
                            'left out and said; the bodies, joints, limits, masses and collision primitives are read. '
                            'Measured 2026-11-07: 2.6 × 10⁻⁸ m, 1.9 × 10⁻¹⁵ rad.')


# ====================================================================== OpenVLA encodings
def _expected(arms: list, action_space: str, target: str) -> tuple[str, str, bool]:
    """The pinned sources' rules restated (configs.py L33-50, materialize.py L32-33; OpenVLA-OFT L32-33, L43-45):
    arms [(arm joints, gripper joints)]."""
    if action_space == 'eef':
        return 'POS_EULER', 'EEF_POS', len(arms) == 1 and 'EEF_POS' in LOADER_ACCEPTS[target]
    if len(arms) == 2:
        layout = all(a == 6 and g >= 1 for a, g in arms)
        return 'JOINT_BIMANUAL', 'JOINT_POS_BIMANUAL', layout and 'JOINT_POS_BIMANUAL' in LOADER_ACCEPTS[target]
    return 'JOINT', 'JOINT_POS', 'JOINT_POS' in LOADER_ACCEPTS[target]


def openvla_errors(folder: str) -> dict:
    """The encoding of six rigs (SO-100 joint and end-effector, UR5e, ALOHA 2 with and without an OpenVLA-OFT policy,
    the bimanual Panda) against _expected; each written oxe_dataset_config.py executed against the enums copied from
    the pinned source; the OFT ALOHA image keys; every case that does not fit says so in the export notes."""
    import sys
    import types

    from ..core import exporters as X
    from ..core import project as P
    from ..core.exporters import lerobot as L
    from ..core.robot.samples import sample_project
    enums = types.ModuleType('prismatic.vla.datasets.rlds.oxe.configs')
    exec(compile(OPENVLA_ENUMS, 'openvla_configs_L33_50', 'exec'), enums.__dict__)    # noqa: S102 — fixed text
    cases = [('so100', None, False), ('so100', 'eef', False), ('ur5e', None, False), ('aloha', None, False),
             ('aloha', None, True), ('panda-bimanual', None, False)]
    bad, rows = [], []
    for name, space, oft in cases:
        p = sample_project(name)
        if oft:
            p.policies.append(P.Policy(name='OpenVLA-OFT', library_id='openvla-oft-plus-aloha'))
        enc = L.openvla_encoding(p, space)
        arms = [(len(a), len(g)) for _r, a, g in L._arm_split(p)]
        want = _expected(arms, space or 'joint', 'openvla-oft' if oft else 'openvla')
        got = (enc['state_encoding'], enc['action_encoding'], enc['fits'])
        tag = f'{name}{" eef" if space else ""}{" +OFT" if oft else ""}'
        if got != want:
            bad.append(f'{tag}: {got} where the sources give {want}')
        sub = os.path.join(folder, f'openvla_{len(rows)}')
        man = X.export(p, sub, ['lerobot'], {'lerobot': {'action_space': space} if space else {}})
        row = man['formats']['lerobot']
        if row['status'] != 'ok':
            bad.append(f'{tag}: export refused: {row.get("reason")}')
            continue
        saved = sys.modules.get('prismatic.vla.datasets.rlds.oxe.configs')
        sys.modules['prismatic.vla.datasets.rlds.oxe.configs'] = enums
        try:
            ns: dict = {}
            with open(os.path.join(sub, 'openvla', 'oxe_dataset_config.py'), encoding='utf-8') as fh:
                exec(compile(fh.read(), 'oxe_dataset_config.py', 'exec'), ns)          # noqa: S102 — our file
        except Exception as e:                         # noqa: BLE001 — the failure is the finding
            bad.append(f'{tag}: the entry does not run against the pinned enums ({type(e).__name__}: {e})')
            continue
        finally:
            if saved is None:
                sys.modules.pop('prismatic.vla.datasets.rlds.oxe.configs', None)
            else:
                sys.modules['prismatic.vla.datasets.rlds.oxe.configs'] = saved
        entry = next(iter(next(v for k, v in ns.items() if k.isupper() and isinstance(v, dict)).values()))
        if (entry['state_encoding'].name, entry['action_encoding'].name) != want[:2]:
            bad.append(f'{tag}: the file holds {entry["state_encoding"].name} / {entry["action_encoding"].name}')
        keys = set(entry['image_obs_keys'])
        want_keys = {'primary', 'secondary', 'left_wrist', 'right_wrist'} if oft and len(arms) == 2 else \
            {'primary', 'secondary', 'wrist'}
        if keys != want_keys:
            bad.append(f'{tag}: image keys {sorted(keys)}')
        if oft and len(arms) == 2:
            left = entry['image_obs_keys'].get('left_wrist') or ''
            if 'left' not in left:
                bad.append(f'{tag}: left_wrist is {left!r}, not the left arm\'s camera')
        said = ' '.join(row['notes'])
        if not want[2] and 'No encoding of the pinned source fits' not in said:
            bad.append(f'{tag}: the notes do not say that no encoding fits')
        rows.append({'case': tag, 'state': got[0], 'action': got[1], 'fits': got[2], 'target': enc['target']})
    return {'mismatches': bad, 'cases': rows}


@check('robot_io.openvla_encoding', 'robot_io', 'OpenVLA export picks its encodings from the rig',
       'number of mismatches over six rigs (SO-100 with joint and end-effector actions, UR5e, ALOHA 2 with and '
       'without an OpenVLA-OFT policy, the bimanual Panda) between the exported state and action encodings, whether '
       'they fit, the image keys and the notes, and the pinned sources\' rules restated in the check; each written '
       'entry executed against OpenVLA\'s enum classes copied verbatim from configs.py L33-50',
       'OpenVLA configs.py L33-50 and materialize.py L32-33 at c8f03f4; OpenVLA-OFT materialize.py L32-33, L43-45 '
       'and configs.py L681-687 at e4287e9', 0, 'mismatches', refs=('openvla_materialize', 'openvla_oft',
                                                                   'openvla_oxe'), cost='study')
def _openvla_encoding(ctx):
    d = openvla_errors(os.path.join(ctx.tmp(), 'robot_io_openvla'))
    return Outcome(len(d['mismatches']), d,
                   note='The rules are a table of the sources\' statements (which action encodings each loader '
                        'accepts, the 7 + 1 and 2 × (6 + 1) layouts); a zero means every rig got the encoding those '
                        'lines give and the export says when none fits. Not run in OpenVLA itself (not installed).')


# ====================================================================== Isaac principal point
def _isaac_project():
    """The UR5e sample with its first camera given an off-centre pinhole calibration (no distortion) and its
    second a Brown-Conrady one with an off-centre principal point (a flat calibration applies to a device's first
    imager)."""
    from ..core.robot.samples import sample_project
    p = sample_project('ur5e').snapshot()
    cams = [s for s in p.variant.sensors if s.enabled and s.kind in ('camera', 'depth')]
    cams[0].calibration = {'K': [[610.0, 0.0, 331.75], [0.0, 610.0, 228.25], [0.0, 0.0, 1.0]], 'D': [0.0] * 5,
                           'width': 640, 'height': 480, 'model': 'brown'}
    if len(cams) > 1:
        cams[1].calibration = {'K': [[600.0, 0.0, 300.25], [0.0, 640.0, 260.5], [0.0, 0.0, 1.0]],
                               'D': [0.1, -0.05, 0.001, 0.002, 0.01], 'width': 640, 'height': 480, 'model': 'brown'}
    return p


def isaac_v3_intrinsics(prim, width: int, height: int) -> tuple:
    """Isaac Lab v3.0.0's initial intrinsics of a camera prim, written again from sensors/camera/camera.py L828-857
    and L879-904: the authored OpenCV fx, fy, cx, cy when the prim has omni:lensdistortion:model and all four;
    otherwise fx = fy = width · focalLength / horizontalAperture, cx = width / 2, cy = height / 2."""
    model = prim.GetAttribute('omni:lensdistortion:model').Get() if prim.GetAttribute(
        'omni:lensdistortion:model') else None
    if model:
        vals = tuple(prim.GetAttribute(f'omni:lensdistortion:{model}:{k}').Get() for k in ('fx', 'fy', 'cx', 'cy'))
        if None not in vals:
            return tuple(float(v) for v in vals)
    f = float(prim.GetAttribute('focalLength').Get())
    ha = float(prim.GetAttribute('horizontalAperture').Get())
    fx = width * f / ha
    return fx, fx, width * 0.5, height * 0.5


def isaac_principal_point_errors(folder: str) -> dict:
    """Export the Isaac files of _isaac_project; re-open cameras.usda with pxr; each camera's v3.0.0 intrinsics
    (isaac_v3_intrinsics) against the imager's fx, fy, cx, cy; every off-centre camera named with its offset in the
    export notes and in the dialog's lines; its JSON record says v2.3.2 cannot show it."""
    import json

    from ..core import exporters as X
    from ..core.exporters import _common as C
    from ..core.exporters import isaac as I
    p = _isaac_project()
    man = X.export(p, folder, ['isaac'])
    row = man['formats']['isaac']
    out = {'px': 0.0, 'mismatches': [], 'cameras': [], 'reader': 'pxr' if I.have_pxr() else 'json'}
    if row['status'] != 'ok':
        out['mismatches'].append(f'refused: {row.get("reason")}')
        return out
    with open(os.path.join(folder, I.JSON_FILE), encoding='utf-8') as fh:
        recs = {r['name']: r for r in json.load(fh)['cameras']}
    cams = {C.frame_name(rc.key): rc for rc in C.rig_cameras(p)}
    stage = None
    if I.have_pxr():
        from pxr import Usd, UsdGeom
        stage = Usd.Stage.Open(os.path.join(folder, I.USD_FILE))
    notes = ' '.join(row['notes'])
    dialog = ' '.join(I.principal_point_lines(p))
    for name, rec in recs.items():
        rc = cams.get(name)
        cam = rc.camera if rc is not None else None
        if cam is None:
            out['mismatches'].append(f'{name}: no imager')
            continue
        if stage is not None:
            prim = next((x for x in stage.Traverse() if x.IsA(UsdGeom.Camera) and x.GetName() == name), None)
            if prim is None:
                out['mismatches'].append(f'{name}: no camera prim')
                continue
            K = isaac_v3_intrinsics(prim, cam.width, cam.height)
            if not prim.GetAttribute('omni:lensdistortion:model'):
                # Isaac's width / 2 is the image centre with pixel edges at integers; OpenCV's (W - 1) / 2 with
                # centres at integers (the half pixel isaac.py documents): compared in OpenCV's convention
                K = (K[0], K[1], K[2] - 0.5, K[3] - 0.5)
        else:
            dd = rec['distortion'] or {}
            K = (dd.get('fx', cam.fx), dd.get('fy', cam.fx), dd.get('cx', cam.width / 2),
                 dd.get('cy', cam.height / 2))
        dx, dy = cam.cx - (cam.width - 1) / 2, cam.cy - (cam.height - 1) / 2
        off = max(abs(dx), abs(dy)) > I.CENTRE_TOL_PX
        err = max(abs(K[0] - cam.fx), abs(K[1] - cam.fy), abs(K[2] - cam.cx), abs(K[3] - cam.cy))
        out['px'] = max(out['px'], err)
        title = f'{rec["sensor"]} · {rec["imager"]}'
        if off:
            if f'{title}: principal point' not in notes:
                out['mismatches'].append(f'{title}: off centre but not named in the notes')
            if f'{title}: principal point' not in dialog:
                out['mismatches'].append(f'{title}: off centre but not named in the dialog lines')
            if not str(rec['principal_point'].get('v2', '')).startswith('not representable'):
                out['mismatches'].append(f'{title}: the JSON does not say v2.3.2 cannot show it')
        out['cameras'].append({'camera': title, 'offset_px': [dx, dy], 'v3_error_px': err})
    if not any(c['offset_px'] != [0.0, 0.0] for c in out['cameras']):
        out['mismatches'].append('no off-centre camera was exercised')
    return out


@check('robot_io.isaac_principal_point', 'robot_io', 'Isaac export: an off-centre principal point reaches Isaac Lab',
       'largest difference (px) between each camera\'s fx, fy, cx, cy and the intrinsics Isaac Lab v3.0.0 would '
       'build from the exported prim (its camera.py rule written again, applied to the prim as pxr re-opens it), '
       'two cameras given off-centre principal points (one a pinhole without distortion); also every off-centre '
       'camera named with its offset in the export notes, in the Export dialog\'s lines and as not representable '
       'at v2.3.2 in isaac_cameras.json',
       'the imagers\' lens models; Isaac Lab v3.0.0 camera.py L828-857, L879-904 and sensors.py L162-165', 1e-3,
       'px', refs=('isaaclab_intrinsics', 'isaaclab_camera_v300', 'isaaclab_camera_v232', 'openusd_camera'),
       cost='study')
def _isaac_principal_point(ctx):
    d = isaac_principal_point_errors(os.path.join(ctx.tmp(), 'robot_io_isaac'))
    return Outcome(d['px'], d, ok=not d['mismatches'],
                   note='USD stores the OpenCV attributes as 32-bit floats (about 3 × 10⁻⁵ px at 640 px); 10⁻³ px '
                        'leaves margin and is far below the effect checked: without the OpenCV attributes Isaac '
                        'Lab centres the principal point (cx = width / 2), 16.6 and 28.5 px off on these cameras. '
                        'Measured 2026-11-07: 4.8 × 10⁻⁵ px. Not rendered in Isaac Sim (not installed).')


# ====================================================================== the self-test line
def selftest_robot_io(folder: str) -> tuple[bool, str]:
    """(ok, text): the SO-100 sample's rig.urdf loads in MuJoCo as written with its sensor frames where DexVantage
    puts them (two draws), and, with usd-core, the SO-100 written to USD and imported back (five draws). Under 2 s."""
    import time
    t0 = time.perf_counter()
    rng = np.random.default_rng(5)
    u = urdf_mujoco_errors(os.path.join(folder, 'urdf'), rng, ('so100',), 2)
    ok = not u['load_errors'] and u['position_m'] <= 1e-6 and u['mass_rel'] <= 1e-9
    text = ('SO-100 rig.urdf ' + (f'loads in MuJoCo, sensor frames within {u["position_m"]:.0e} m'
                                  if not u['load_errors'] else u['load_errors'][0]))
    if HAVE_PXR:
        d = usd_roundtrip_errors(os.path.join(folder, 'usd'), rng, ('so100',), 5)
        ok = ok and d['position_m'] <= 1e-6 and not d['limit_mismatches'] and not d['missing']
        text += f'; USD round trip within {d["position_m"]:.0e} m'
    else:
        text += '; USD import not checked (usd-core not installed)'
    return ok, text + f'; {(time.perf_counter() - t0) * 1000:.0f} ms'


__all__ = ['urdf_mujoco_errors', 'usd_roundtrip_errors', 'usd_third_party_errors', 'openvla_errors',
           'isaac_principal_point_errors', 'isaac_v3_intrinsics', 'OPENVLA_ENUMS', 'LOADER_ACCEPTS',
           'selftest_robot_io']
