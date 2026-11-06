"""Checks of the robot kinematics (core/robot/model.py, MuJoCo; core/robot/kinematics.py, product of exponentials).

1. MuJoCo's forward kinematics against the independent product-of-exponentials chain, on random joint vectors, for
   every robot the project holds and for the robots written in this module (a UR5e and a Panda built from their
   makers' DH tables, and two models that use every pose convention of MJCF and URDF the readers handle).
2. Both engines against the closed-form standard DH product for the UR5e, from Universal Robots' published table.
3. Both engines against the closed-form modified DH (Craig) product for the Franka Panda, from Franka's published
   table.
4. The PoE Jacobian against central finite differences of the PoE forward kinematics, and against MuJoCo's mj_jac.
5. Pinocchio's forward kinematics (registered only when pinocchio is installed).

The DH robots are written as URDF and MJCF files here, from the published tables, so the checks do not depend on any
downloaded model. Their DH figures, with source and date read:

- UR5e (standard DH; a, d in m, alpha in rad): Universal Robots, "DH Parameters for calculations of kinematics and
  dynamics", https://www.universal-robots.com/articles/ur/application-installation/
  dh-parameters-for-calculations-of-kinematics-and-dynamics/ (UR5e/UR7e table), read 2026-11-06.
- Panda (modified DH, Craig's convention; a, d in m, alpha in rad): Franka Emika, "Robot and interface
  specifications", docs source file source/control_parameters.rst of github.com/frankarobotics/docs (formerly
  frankaemika/docs) at commit 476a2ed2c1f457b87ba2fbb9213ebcd13825382d (2021-11-11, "the Panda's kinematic chain"),
  read 2026-11-06. The page itself (https://frankaemika.github.io/docs/control_parameters.html) returned HTTP 404 on
  that date; its successor "Control Interface Specification and Robot Limits"
  (https://frankarobotics.github.io/docs/robot_specifications.html, read 2026-11-06) gives the same figures for the
  Franka Research 3.
"""
from __future__ import annotations

import importlib.util
import json
import math

import numpy as np

from .suite import Outcome, check

PI = math.pi

# Universal Robots UR5e, standard DH (a [m], d [m], alpha [rad]) per joint 1..6 (source in the module docstring)
UR5E_DH = ((0.0, 0.1625, PI / 2), (-0.425, 0.0, 0.0), (-0.3922, 0.0, 0.0), (0.0, 0.1333, PI / 2),
           (0.0, 0.0997, -PI / 2), (0.0, 0.0996, 0.0))
# Franka Panda, modified DH, Craig's convention (a [m], d [m], alpha [rad]) per row Joint 1..7, then the flange
# (theta = 0) (source in the module docstring)
PANDA_MDH = ((0.0, 0.333, 0.0), (0.0, 0.0, -PI / 2), (0.0, 0.316, PI / 2), (0.0825, 0.0, PI / 2),
             (-0.0825, 0.384, -PI / 2), (0.0, 0.0, PI / 2), (0.088, 0.0, PI / 2))
PANDA_FLANGE = (0.0, 0.107, 0.0)

POS_TOL = 1e-9        # m: both sides are double-precision products of about 20 transforms
ROT_TOL = 1e-9        # rad
JAC_FD_TOL = 1e-7     # m/rad or rad/rad: central differences with h = 1e-6
FD_STEP = 1e-6


# ====================================================================== the closed forms, written out again
def dh_standard(a, d, alpha, theta) -> np.ndarray:
    """Rz(theta) Tz(d) Tx(a) Rx(alpha), Denavit and Hartenberg (1955) as Spong et al. write it."""
    ct, st, ca, sa = math.cos(theta), math.sin(theta), math.cos(alpha), math.sin(alpha)
    return np.array([[ct, -st * ca, st * sa, a * ct],
                     [st, ct * ca, -ct * sa, a * st],
                     [0.0, sa, ca, d],
                     [0.0, 0.0, 0.0, 1.0]])


def dh_modified(a, d, alpha, theta) -> np.ndarray:
    """Rx(alpha) Tx(a) Rz(theta) Tz(d), Craig's modified DH (Introduction to Robotics, chapter 3)."""
    ct, st, ca, sa = math.cos(theta), math.sin(theta), math.cos(alpha), math.sin(alpha)
    return np.array([[ct, -st, 0.0, a],
                     [st * ca, ct * ca, -sa, -sa * d],
                     [st * sa, ct * sa, ca, ca * d],
                     [0.0, 0.0, 0.0, 1.0]])


def ur5e_closed_form(q) -> list:
    """World poses of DH frames 1..6 of the UR5e (frame 6 is the tool flange)."""
    T, out = np.eye(4), []
    for (a, d, al), th in zip(UR5E_DH, q):
        T = T @ dh_standard(a, d, al, th)
        out.append(T)
    return out


def panda_closed_form(q) -> list:
    """World poses of the Panda's frames 1..7 and the flange."""
    T, out = np.eye(4), []
    for (a, d, al), th in zip(PANDA_MDH, q):
        T = T @ dh_modified(a, d, al, th)
        out.append(T)
    out.append(T @ dh_modified(*PANDA_FLANGE, 0.0))
    return out


# ====================================================================== the robots written here
def _f(v) -> str:
    return ' '.join(repr(float(x)) for x in np.ravel(v))


def _rx(al):
    c, s = math.cos(al), math.sin(al)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def make_robot(uid, name, kind, files: dict, main: str, base=None):
    """(Robot, sources) from files {relative path: text}; MJCF includes get a manifest entry, as the importer
    writes."""
    from ..core.project import Robot, RobotSource
    from ..core.robot.importer import MANIFEST
    src = {f'{uid}/{rel}': txt.encode() for rel, txt in files.items()}
    if kind == 'mjcf':
        refs = {f'include:{rel}': rel for rel in files if rel != main}
        src[f'{uid}/{MANIFEST}'] = json.dumps({'format': 'dexvantage.robot-refs', 'version': 1, 'kind': kind,
                                              'main': main, 'refs': refs}).encode()
    r = Robot(uid=uid, name=name, source=RobotSource(kind=kind, file=f'{uid}/{main}'))
    if base is not None:
        r.base = base
    return r, src


INERTIAL = ('<inertial><mass value="0.5"/><inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/>'
            '</inertial>')


def ur5e_mjcf_text() -> str:
    """The UR5e as MJCF: body i is DH frame i at q = 0 (pos (a, 0, d), Rx(alpha) as euler in degrees); the hinge of
    joint i turns about frame i−1's z axis, written in frame i's coordinates with its pos at frame i−1's origin."""
    out = ['<mujoco model="ur5e_dh">', '  <compiler angle="degree"/>',
           '  <default><geom type="sphere" size="0.02"/></default>', '  <worldbody>', '    <body name="frame0">',
           '      <geom/>']
    ind = '      '
    for i, (a, d, al) in enumerate(UR5E_DH, 1):
        R = _rx(al)
        anchor = -R.T @ np.array([a, 0.0, d])
        axis = R.T @ np.array([0.0, 0.0, 1.0])
        out.append(f'{ind}<body name="frame{i}" pos="{_f([a, 0, d])}" euler="{_f([math.degrees(al), 0, 0])}">')
        out.append(f'{ind}  <joint name="joint{i}" pos="{_f(anchor)}" axis="{_f(axis)}" range="-360 360"/><geom/>')
        ind += '  '
    for _ in UR5E_DH:
        ind = ind[:-2]
        out.append(f'{ind}</body>')
    out += ['    </body>', '  </worldbody>', '</mujoco>', '']
    return '\n'.join(out)


def ur5e_urdf_text() -> str:
    """The UR5e as URDF: joint i's origin is DH row i−1's fixed transform Tz(d) Tx(a) Rx(alpha), its axis z; a fixed
    joint then places frame i (the DH frame) on link i."""
    out = ['<?xml version="1.0"?>', '<robot name="ur5e_dh">', f'  <link name="frame0">{INERTIAL}</link>']
    prev = (0.0, 0.0, 0.0)
    for i, (a, d, al) in enumerate(UR5E_DH, 1):
        pa, pd, pal = prev
        out.append(f'  <link name="link{i}">{INERTIAL}</link><link name="frame{i}"/>')
        parent = 'frame0' if i == 1 else f'link{i - 1}'
        out.append(f'  <joint name="joint{i}" type="revolute"><parent link="{parent}"/><child link="link{i}"/>'
                   f'<origin xyz="{_f([pa, 0, pd])}" rpy="{_f([pal, 0, 0])}"/><axis xyz="0 0 1"/>'
                   f'<limit lower="-6.2831853" upper="6.2831853" effort="1" velocity="1"/></joint>')
        out.append(f'  <joint name="frame{i}_fixed" type="fixed"><parent link="link{i}"/><child link="frame{i}"/>'
                   f'<origin xyz="{_f([a, 0, d])}" rpy="{_f([al, 0, 0])}"/></joint>')
        prev = (a, d, al)
    out += ['</robot>', '']
    return '\n'.join(out)


def panda_urdf_text() -> str:
    """The Panda as URDF from the modified DH rows: joint i's origin is Rx(alpha) Tx(a) Tz(d) (xyz = (a, −d sin
    alpha, d cos alpha), rpy = (alpha, 0, 0)), its axis z; the flange a fixed joint."""
    out = ['<?xml version="1.0"?>', '<robot name="panda_mdh">', f'  <link name="frame0">{INERTIAL}</link>']
    for i, (a, d, al) in enumerate(PANDA_MDH, 1):
        out.append(f'  <link name="frame{i}">{INERTIAL}</link>')
        out.append(f'  <joint name="joint{i}" type="revolute"><parent link="frame{i - 1}"/><child link="frame{i}"/>'
                   f'<origin xyz="{_f([a, -d * math.sin(al), d * math.cos(al)])}" rpy="{_f([al, 0, 0])}"/>'
                   f'<axis xyz="0 0 1"/><limit lower="-3.0" upper="3.0" effort="1" velocity="1"/></joint>')
    a, d, al = PANDA_FLANGE
    out.append('  <link name="flange"/>')
    out.append(f'  <joint name="flange_fixed" type="fixed"><parent link="frame7"/><child link="flange"/>'
               f'<origin xyz="{_f([a, -d * math.sin(al), d * math.cos(al)])}" rpy="{_f([al, 0, 0])}"/></joint>')
    out += ['</robot>', '']
    return '\n'.join(out)


FEATURE_MJCF = '''<mujoco model="features">
  <compiler angle="degree" eulerseq="zyX"/>
  <include file="parts/defaults.xml"/>
  <worldbody>
    <body name="base" pos="0.1 -0.2 0.05" quat="2 0 0 0.6">
      <geom/>
      <body name="shoulder" pos="0 0 0.2" euler="10 -20 35" childclass="arm">
        <joint name="j1"/><geom/>
        <body name="upper" pos="0.05 0.02 0.3" axisangle="1 1 0 40">
          <joint name="j2" class="tilted"/><geom/>
          <body pos="0.3 0 0" xyaxes="0 1 0 -1 0.2 0">
            <joint name="j3" pos="-0.1 0 0.02" axis="0.3 -1 0.2" ref="15"/><geom/>
            <frame pos="0.1 0.05 0" euler="0 90 0">
              <body name="wrist" pos="0 0 0.12" zaxis="0.2 0.3 1">
                <joint name="j4"/>
                <joint name="j5" axis="1 0 0" pos="0 0.03 0"/>
                <joint type="slide" axis="0 1 1" ref="0.02" range="-0.1 0.1"/><geom/>
                <body name="flip" pos="0 0 0.05" zaxis="0 0 -1">
                  <joint name="ball" type="ball" class="main" pos="0.01 0 0"/><geom/>
                  <body name="tip" pos="0 0.04 0.03" euler="0 0 -30"><geom/></body>
                </body>
              </body>
            </frame>
          </body>
        </body>
      </body>
    </body>
    <body name="puck" pos="0.4 0 0.02"><freejoint name="puck_free"/><geom/></body>
  </worldbody>
  <keyframe><key name="home" qpos="0 0 0.2617993877991494 0 0 0.02  0.9238795325112867 0 0.3826834323650898 0
    0.5 0.1 0.3 0.7071067811865476 0 0 0.7071067811865476"/></keyframe>
</mujoco>
'''
FEATURE_DEFAULTS = '''<mujoco>
  <default>
    <geom type="sphere" size="0.02"/>
    <default class="arm">
      <joint axis="0 1 0" range="-170 170"/>
      <default class="tilted"><joint axis="1 0 1" pos="0 0 -0.05"/></default>
    </default>
  </default>
</mujoco>
'''


def feature_urdf_text() -> str:
    """A branching URDF with a 'world' root, rpy on every origin, axes that are not unit length, a continuous joint,
    prismatic fingers and a chain of fixed joints."""
    return f'''<?xml version="1.0"?>
<robot name="features_urdf">
  <link name="world"/>
  <link name="base">{INERTIAL}</link>
  <link name="l1">{INERTIAL}</link><link name="l2">{INERTIAL}</link><link name="l3">{INERTIAL}</link>
  <link name="mount"/><link name="palm">{INERTIAL}</link>
  <link name="finger_a">{INERTIAL}</link><link name="finger_b">{INERTIAL}</link><link name="tcp"/>
  <joint name="world_fixed" type="fixed"><parent link="world"/><child link="base"/>
    <origin xyz="0.2 0.1 0.03" rpy="0 0 0.4"/></joint>
  <joint name="a1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0.15" rpy="0.1 -0.2 0.3"/>
    <axis xyz="0 0 2"/><limit lower="-3" upper="3" effort="1" velocity="1"/></joint>
  <joint name="a2" type="continuous"><parent link="l1"/><child link="l2"/><origin xyz="0.02 0.01 0.2"
    rpy="1.5707963267948966 0 -0.3"/><axis xyz="0 1 1"/></joint>
  <joint name="a3" type="revolute"><parent link="l2"/><child link="l3"/><origin xyz="0.3 0 0" rpy="0 0.7 0"/>
    <axis xyz="-1 0.5 0"/><limit lower="-2" upper="2" effort="1" velocity="1"/></joint>
  <joint name="m1" type="fixed"><parent link="l3"/><child link="mount"/><origin xyz="0.1 0 0.02" rpy="3.0 0 1.2"/>
    </joint>
  <joint name="m2" type="fixed"><parent link="mount"/><child link="palm"/><origin xyz="0 0.05 0" rpy="0 -1.1 0"/>
    </joint>
  <joint name="fa" type="prismatic"><parent link="palm"/><child link="finger_a"/><origin xyz="0 0.02 0.06"/>
    <axis xyz="0 1 0"/><limit lower="0" upper="0.04" effort="1" velocity="1"/></joint>
  <joint name="fb" type="prismatic"><parent link="palm"/><child link="finger_b"/><origin xyz="0 -0.02 0.06"
    rpy="0 0 3.141592653589793"/><axis xyz="0 3 0"/><limit lower="0" upper="0.04" effort="1" velocity="1"/></joint>
  <joint name="tcp_fixed" type="fixed"><parent link="palm"/><child link="tcp"/><origin xyz="0 0 0.1" rpy="0.2 0 0"/>
    </joint>
</robot>
'''


def builtin_robots() -> list:
    """[(Robot, sources)] of the robots written in this module."""
    from ..core.frames import Pose, quat_from_rpy
    return [
        make_robot('kin-ur5e-mjcf', 'UR5e (DH, MJCF)', 'mjcf', {'ur5e.xml': ur5e_mjcf_text()}, 'ur5e.xml'),
        make_robot('kin-ur5e-urdf', 'UR5e (DH, URDF)', 'urdf', {'ur5e.urdf': ur5e_urdf_text()}, 'ur5e.urdf'),
        make_robot('kin-panda-urdf', 'Panda (modified DH, URDF)', 'urdf', {'panda.urdf': panda_urdf_text()},
                   'panda.urdf'),
        make_robot('kin-features-mjcf', 'MJCF conventions', 'mjcf',
                   {'robot/features.xml': FEATURE_MJCF, 'robot/parts/defaults.xml': FEATURE_DEFAULTS},
                   'robot/features.xml', base=Pose([0.3, -0.1, 0.7], quat_from_rpy(0.2, 0.1, -2.0))),
        make_robot('kin-features-urdf', 'URDF conventions', 'urdf', {'features.urdf': feature_urdf_text()},
                   'features.urdf', base=Pose([-0.5, 0.25, 0.0], quat_from_rpy(0, 0, 1.0))),
    ]


# ====================================================================== shared helpers
def rotation_angle(Ra, Rb) -> float:
    """The angle (rad) of Ra^T Rb, full precision near zero (atan2 of the sine and cosine parts)."""
    R = np.asarray(Ra).T @ np.asarray(Rb)
    s = 0.5 * math.sqrt((R[2, 1] - R[1, 2]) ** 2 + (R[0, 2] - R[2, 0]) ** 2 + (R[1, 0] - R[0, 1]) ** 2)
    c = 0.5 * (np.trace(R) - 1.0)
    return math.atan2(s, c)


def pose_errors(Ta, Tb) -> tuple[float, float]:
    """(position error m, rotation error rad) between two 4×4 poses."""
    return float(np.linalg.norm(np.asarray(Ta)[:3, 3] - np.asarray(Tb)[:3, 3])), rotation_angle(Ta[:3, :3],
                                                                                                    Tb[:3, :3])


def random_q(model, rng, n) -> np.ndarray:
    """n joint vectors uniform within the limits; [−π, π] or [−0.2, 0.2] m where a joint has none."""
    lo, hi = np.array(model.lower, float), np.array(model.upper, float)
    rev = np.array([t != 'prismatic' for t in model.joint_types])
    span = np.where(rev, PI, 0.2)
    lo = np.where(np.isfinite(lo), lo, -span)
    hi = np.where(np.isfinite(hi), hi, span)
    return rng.uniform(lo, hi, (n, model.nq))


def _pairs(ctx):
    """[(name, RobotModel, PoEChain or the reason it is missing)] for the project's robots and the built-in ones."""
    from ..core.robot.kinematics import KinematicsError, PoEChain
    from ..core.robot.model import RobotModel, RobotModelError
    out = []
    items = [(r, ctx.project.sources, 'project') for r in getattr(ctx.project, 'robots', [])]
    items += [(r, s, 'built in') for r, s in builtin_robots()]
    for r, src, where in items:
        try:
            m = RobotModel(r, src)
        except RobotModelError as e:
            out.append((f'{r.name} ({where})', None, f'MuJoCo could not compile it: {e}'))
            continue
        try:
            out.append((f'{r.name} ({where})', m, PoEChain(r, src)))
        except KinematicsError as e:
            out.append((f'{r.name} ({where})', m, f'the independent reader could not read it: {e}'))
    return out


def _to_poe(model, chain, q) -> np.ndarray:
    """q in RobotModel's joint order to PoEChain's order (matched by name)."""
    d = dict(zip(model.joint_names, q))
    return np.array([d[n] for n in chain.joint_names], float)


# ====================================================================== 1. MuJoCo against the PoE chain
@check('kinematics.fk_poe_mujoco', 'kinematics', 'Forward kinematics: MuJoCo against product of exponentials',
       'largest link position difference (m) between MuJoCo\'s forward kinematics (RobotModel.fk) and the '
       'independent product-of-exponentials chain (PoEChain.fk) over random joint vectors, every link of every robot; '
       'the rotation difference (rad) must also be within 1e-9',
       'core/robot/kinematics.py: its own XML reading (xml.etree), tree building and twist exponentials, no MuJoCo; '
       'the robots of the project and the five written in the check', POS_TOL, 'm',
       refs=('lynch_park2017', 'mujoco_todorov2012', 'mujoco_xml', 'urdf'))
def _fk_poe_mujoco(ctx):
    n = 20 if ctx.quick else 200
    per, skipped, worst_p, worst_r = {}, {}, 0.0, 0.0
    for k, (name, model, chain) in enumerate(_pairs(ctx)):
        if model is None or isinstance(chain, str):
            skipped[name] = chain if isinstance(chain, str) else 'not compiled'
            continue
        missing = sorted(set(model.link_names) - set(chain.link_names))
        joints = sorted(set(model.joint_names) ^ set(chain.joint_names))
        if missing or joints:
            skipped[name] = ('the two readers disagree: ' +
                             '; '.join(x for x in ((f'links only MuJoCo has: {", ".join(missing[:5])}' if missing
                                                     else ''),
                                                    (f'joints only one reader has: {", ".join(joints[:5])}' if joints
                                                     else '')) if x))
            worst_p = math.inf
            continue
        rng = ctx.rng('kinematics.fk_poe_mujoco', k)
        ep = er = 0.0
        for q in random_q(model, rng, n):
            A, B = model.fk(q), chain.fk(_to_poe(model, chain, q))
            for ln in model.link_names:
                p, r = pose_errors(A[ln], B[ln])
                ep, er = max(ep, p), max(er, r)
        per[name] = {'links': len(model.link_names), 'joints': model.nq, 'position_m': ep, 'rotation_rad': er}
        worst_p, worst_r = max(worst_p, ep), max(worst_r, er)
    return Outcome(worst_p, {'joint_vectors_per_robot': n, 'robots': per, 'not_compared': skipped,
                             'rotation_rad': worst_r}, ok=bool(per) and worst_r <= ROT_TOL,
                   note='Both are double-precision products of about twenty rigid transforms, so rounding stays near '
                        '1e-15; 1e-9 leaves room for that while any convention error (an rpy order, a quaternion '
                        'order, degrees for radians, a joint anchor) shows as 1e-3 or more. Robots the independent '
                        'reader cannot read are listed, not compared.')


# ====================================================================== 2. UR5e standard DH
def _dh_errors(ctx, cid, robots, closed, nj, frames, n):
    from ..core.robot.kinematics import PoEChain
    from ..core.robot.model import RobotModel
    per, worst_p, worst_r = {}, 0.0, 0.0
    for k, (r, src) in enumerate(robots):
        model, chain = RobotModel(r, src), PoEChain(r, src)
        rng = ctx.rng(cid, k)
        names = [f'joint{i + 1}' for i in range(nj)]
        res = {'mujoco': [0.0, 0.0], 'poe': [0.0, 0.0]}
        for q in rng.uniform(-PI, PI, (n, len(names))):
            ref = closed(q)
            qd = dict(zip(names, q))
            A = model.fk(np.array([qd[j] for j in model.joint_names]))
            B = chain.fk(qd)
            for fname, T in zip(frames, ref):
                for eng, F in (('mujoco', A), ('poe', B)):
                    p, rr = pose_errors(F[fname], T)
                    res[eng] = [max(res[eng][0], p), max(res[eng][1], rr)]
        per[r.name] = {e: {'position_m': v[0], 'rotation_rad': v[1]} for e, v in res.items()}
        worst_p = max(worst_p, res['mujoco'][0], res['poe'][0])
        worst_r = max(worst_r, res['mujoco'][1], res['poe'][1])
    return per, worst_p, worst_r


@check('kinematics.dh_ur5e', 'kinematics', 'UR5e against its standard DH closed form',
       'largest position difference (m) of DH frames 1–6 between the closed-form product of Rz(θ) Tz(d) Tx(a) Rx(α) '
       'and both engines (MuJoCo and PoE) on a UR5e written from the same table as MJCF and as URDF; rotations within '
       '1e-9 rad', 'Universal Robots\' published UR5e DH table, the standard DH product written out in the check',
       POS_TOL, 'm', refs=('ur_dh', 'dh1955', 'spong2006'))
def _dh_ur5e(ctx):
    rob = builtin_robots()[:2]
    n = 20 if ctx.quick else 200
    frames = [f'frame{i}' for i in range(1, 7)]
    per, wp, wr = _dh_errors(ctx, 'kinematics.dh_ur5e', rob, ur5e_closed_form, 6, frames, n)
    q0 = ur5e_closed_form(np.zeros(6))[-1][:3, 3]
    return Outcome(wp, {'joint_vectors': n, 'robots': per, 'rotation_rad': wr,
                        'flange_at_zero_m': [float(v) for v in q0], 'dh_table': [list(r) for r in UR5E_DH]},
                   ok=wr <= ROT_TOL,
                   note='The models are written from the DH figures in the check, so the only difference is the route '
                        '(a hinge with an offset anchor and tilted axis in MJCF, a joint origin and fixed frames in '
                        'URDF, against the DH product): agreement is to rounding, and 1e-9 m flags any convention '
                        'slip.')


# ====================================================================== 3. Panda modified DH
@check('kinematics.dh_panda', 'kinematics', 'Franka Panda against its modified DH (Craig) closed form',
       'largest position difference (m) of frames 1–7 and the flange between the closed-form product of '
       'Rx(α) Tx(a) Rz(θ) Tz(d) and both engines on a Panda written from the same table as URDF; rotations within '
       '1e-9 rad', 'Franka\'s published Panda DH table (Craig\'s convention), the modified DH product written out in '
       'the check', POS_TOL, 'm', refs=('franka_dh', 'craig2005'))
def _dh_panda(ctx):
    rob = [builtin_robots()[2]]
    n = 20 if ctx.quick else 200
    frames = [f'frame{i}' for i in range(1, 8)] + ['flange']
    per, wp, wr = _dh_errors(ctx, 'kinematics.dh_panda', rob, panda_closed_form, 7, frames, n)
    f0 = panda_closed_form(np.zeros(7))[-1][:3, 3]
    return Outcome(wp, {'joint_vectors': n, 'robots': per, 'rotation_rad': wr,
                        'flange_at_zero_m': [float(v) for v in f0]}, ok=wr <= ROT_TOL,
                   note='Joint ranges are not part of this check: θ is drawn in [−π, π] for every joint. The flange at '
                        'q = 0 is (0.088, 0, 0.926) m: x = 0.0825 − 0.0825 + 0.088, z = 0.333 + 0.316 + 0.384 − 0.107.')


# ====================================================================== 4. Jacobians
def fd_jacobian(chain, q, link, h=FD_STEP) -> np.ndarray:
    """(6, nq) [linear; angular] Jacobian of the link origin by central differences of chain.fk."""
    q = np.asarray(q, float)
    J = np.zeros((6, chain.nq))
    for i in range(chain.nq):
        dq = np.zeros(chain.nq)
        dq[i] = h
        Tp, Tm = chain.fk(q + dq)[link], chain.fk(q - dq)[link]
        J[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * h)
        A = Tp[:3, :3] @ Tm[:3, :3].T                   # ≈ exp([ω] 2h)
        v = 0.5 * np.array([A[2, 1] - A[1, 2], A[0, 2] - A[2, 0], A[1, 0] - A[0, 1]])
        s = float(np.linalg.norm(v))
        ang = math.atan2(s, 0.5 * (np.trace(A) - 1.0))
        J[3:, i] = (v * (ang / s) if s > 0 else v) / (2 * h)
    return J


@check('kinematics.jacobian_fd', 'kinematics', 'Jacobian against finite differences',
       'largest element difference between the PoE chain\'s analytic Jacobian (twists moved by the adjoint) and '
       'central finite differences of its own forward kinematics, every link of the built-in robots',
       'central differences, step 1e-6 rad or m, of PoEChain.fk; rotation by the log of R(q+h) R(q−h)ᵀ', JAC_FD_TOL,
       'm/rad', refs=('lynch_park2017',))
def _jac_fd(ctx):
    from ..core.robot.kinematics import PoEChain
    from ..core.robot.model import RobotModel
    n = 3 if ctx.quick else 10
    worst, per = 0.0, {}
    for k, (r, src) in enumerate(builtin_robots()):
        model, chain = RobotModel(r, src), PoEChain(r, src)
        e = 0.0
        for q in random_q(model, ctx.rng('kinematics.jacobian_fd', k), n):
            qp = _to_poe(model, chain, q)
            for ln in chain.link_names:
                e = max(e, float(np.abs(chain.jacobian(qp, ln) - fd_jacobian(chain, qp, ln)).max()))
        per[r.name] = e
        worst = max(worst, e)
    return Outcome(worst, {'joint_vectors_per_robot': n, 'robots': per, 'step': FD_STEP},
                   note='Central differences carry a truncation error of about h²/6 times the third derivative '
                        '(1e-13 here) and a rounding error of about 1e-16/h = 1e-10; 1e-7 is well above both and '
                        'well below any wrong column (of order the link lengths, 0.1 m).')


@check('kinematics.jacobian_mujoco', 'kinematics', 'Jacobian: PoE against MuJoCo',
       'largest element difference between the PoE chain\'s world-frame Jacobian and MuJoCo\'s mj_jac '
       '(RobotModel.jacobian), at each link origin and at a random point fixed in each link, columns matched by joint '
       'name, every robot',
       'MuJoCo mj_jac (another implementation of the same Jacobian)', 1e-9, 'm/rad',
       refs=('mujoco_todorov2012', 'lynch_park2017'))
def _jac_mujoco(ctx):
    n = 5 if ctx.quick else 40
    worst, per, skipped = 0.0, {}, {}
    for k, (name, model, chain) in enumerate(_pairs(ctx)):
        if model is None or isinstance(chain, str):
            skipped[name] = chain if isinstance(chain, str) else 'not compiled'
            continue
        if set(model.link_names) - set(chain.link_names) or set(model.joint_names) != set(chain.joint_names):
            skipped[name] = 'the two readers disagree on the links or joints (see kinematics.fk_poe_mujoco)'
            worst = math.inf
            continue
        idx = [chain.joint_names.index(j) for j in model.joint_names]
        rng = ctx.rng('kinematics.jacobian_mujoco', k)
        e = 0.0
        for q in random_q(model, rng, n):
            qp = _to_poe(model, chain, q)
            for ln in model.link_names:
                pt = rng.uniform(-0.2, 0.2, 3)          # a point fixed in the link, link coordinates (m)
                e = max(e, float(np.abs(model.jacobian(q, ln) - chain.jacobian(qp, ln)[:, idx]).max()),
                        float(np.abs(model.jacobian(q, ln, pt) - chain.jacobian(qp, ln, pt)[:, idx]).max()))
        per[name] = e
        worst = max(worst, e)
    return Outcome(worst, {'joint_vectors_per_robot': n, 'robots': per, 'not_compared': skipped}, ok=bool(per),
                   note='Both are exact formulas in double precision; 1e-9 allows for rounding only. The offset point '
                        '(up to 0.2 m from the link origin on each axis) exercises the point argument and the '
                        'link-to-world rotation that the origin alone does not.')


# ====================================================================== 5. pinocchio (only when installed)
if importlib.util.find_spec('pinocchio') is not None:
    @check('kinematics.fk_pinocchio', 'kinematics', 'Forward kinematics against pinocchio',
           'largest link position difference (m) between pinocchio\'s forward kinematics and both DexVantage engines '
           'on the UR5e and Panda URDFs written from the DH tables; rotations within 1e-9 rad',
           'pinocchio (buildModelFromXML, forwardKinematics, updateFramePlacements), an independent rigid-body '
           'library', POS_TOL, 'm', refs=('pinocchio2019', 'urdf'))
    def _fk_pinocchio(ctx):
        import pinocchio as pin
        from ..core.robot.kinematics import PoEChain
        from ..core.robot.model import RobotModel
        texts = {'kin-ur5e-urdf': ur5e_urdf_text(), 'kin-panda-urdf': panda_urdf_text()}
        n = 20 if ctx.quick else 200
        per, wp, wr = {}, 0.0, 0.0
        for k, (r, src) in enumerate(builtin_robots()):
            if r.uid not in texts:
                continue
            pm = pin.buildModelFromXML(texts[r.uid])
            pd = pm.createData()
            model, chain = RobotModel(r, src), PoEChain(r, src)
            ep = er = 0.0
            for q in random_q(model, ctx.rng('kinematics.fk_pinocchio', k), n):
                d = dict(zip(model.joint_names, q))
                qp = np.zeros(pm.nq)
                for jn in model.joint_names:
                    qp[pm.joints[pm.getJointId(jn)].idx_q] = d[jn]
                pin.forwardKinematics(pm, pd, qp)
                pin.updateFramePlacements(pm, pd)
                A, B = model.fk(q), chain.fk(d)
                for ln in model.link_names:
                    if not pm.existFrame(ln):
                        continue
                    T = pd.oMf[pm.getFrameId(ln)].homogeneous
                    for F in (A, B):
                        p, rr = pose_errors(F[ln], T)
                        ep, er = max(ep, p), max(er, rr)
            per[r.name] = {'position_m': ep, 'rotation_rad': er}
            wp, wr = max(wp, ep), max(wr, er)
        return Outcome(wp, {'joint_vectors_per_robot': n, 'robots': per, 'rotation_rad': wr,
                            'pinocchio': getattr(pin, '__version__', '')}, ok=bool(per) and wr <= ROT_TOL,
                       note='Three double-precision implementations of the same URDF; 1e-9 allows for rounding only.')
