"""ROS 2 export (spec §9.2): a URDF of the rig with sensor links and joints, a Python launch file of
static_transform_publisher nodes for the sensor frames, and a tf2 tree (YAML) of the camera frames.

    @exporter('urdf', ...)   options: q ({robot uid: joints}, for the tree's world poses only), package (''),
                             meshes (True)

Files (under ros2/):
- rig.urdf: a root link 'world' (the project's world, the primary robot's base, REP-103), every robot's links and
  joints, each sensor's body link '<sensor>_link' and each imager's '<sensor>_<imager>_optical_frame' (REP-103's
  optical frame: z forward, x right, y down) on fixed joints. With one robot, its link and joint names are those of
  its file; with several, each is prefixed '<robot>_' (said in the notes).
- meshes/<robot>_<link>.stl: each link's visual meshes in its frame (binary STL, metres), referenced as
  'meshes/…' or 'package://<package>/meshes/…' when the package option is set.
- static_transforms.launch.py: one tf2_ros static_transform_publisher Node per sensor frame (parent → '<sensor>_link'
  and '<sensor>_link' → each optical frame), with the arguments of tf2_ros's program (--x --y --z --qx --qy --qz --qw
  --frame-id --child-frame-id; ros2/geometry2 tf2_ros/src/static_transform_broadcaster_program.cpp at commit
  183399c, read 2026-11-07). Use it beside your own robot description instead of rig.urdf.
- tf_tree.yaml: every sensor frame with its parent, the static transform, whether it moves with a robot link, and
  its pose in the world at the export's joint state.

How the robot becomes URDF. The robot is read from the MuJoCo model DexVantage compiled from its stored file
(core/robot/model.py), so an MJCF robot exports too. Each MuJoCo body is a link at DexVantage's link frame. A body's
hinge or slide joint j, at anchor p and axis a in the body frame, with MuJoCo's reference qpos0, moves the body as
X_parent · T0 · Trans(p) · Rot(a, q − qpos0) · Trans(−p) (MuJoCo's mj_kinematics; T0 the body's pos and quat). URDF
moves a child as X_parent · origin · Rot(axis, q) about the child frame's origin [urdfdom joint and pose parsers], so
the joint's origin is T0 · Trans(p) · Rot(a, −qpos0) and, when p is not zero, an extra link '<body>__<joint>' carries
the joint and a fixed joint with origin Trans(−p) brings the body back to its own frame. A body with free or ball
joints (held at the home keyframe in DexVantage) is fixed at its home pose. origin rpy are urdfdom's fixed-axis
roll, pitch, yaw (R = Rz(yaw) Ry(pitch) Rx(roll); urdf_model/pose.h setFromRPY at ros/urdfdom_headers master, read
2026-11-07). Joint limits are DexVantage's (the robot's records or the file's ranges); a joint without finite limits
is 'continuous' (a slide without limits cannot be written in URDF and is refused). urdfdom refuses a <limit> without
effort and velocity; when the robot's records do not state them they are written as 0 and the notes say so (set them
before giving the file to a controller or a planner).

Known limits: no inertials (kinematics only: DexVantage does not use mass, and the figures are not stated by every
file) and no collision geometry; MuJoCo's equality constraints and URDF mimic joints are not written (each joint is
independent, as in DexVantage); materials are one colour per link; a URDF link that MuJoCo merged into its parent is
written on a fixed joint at its recorded offset.
"""
from __future__ import annotations

import math
from xml.sax.saxutils import quoteattr

import numpy as np

from ..frames import quat_from_matrix, quat_from_mujoco, quat_to_matrix
from ..jobs import StudyError
from . import _common as C
from . import exporter

URDF = 'ros2/rig.urdf'
LAUNCH = 'ros2/static_transforms.launch.py'
TREE = 'ros2/tf_tree.yaml'
MESHES = 'ros2/meshes'


def _T(R=None, t=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def _axis_angle(a, ang) -> np.ndarray:
    a = np.asarray(a, float)
    a = a / (np.linalg.norm(a) or 1.0)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def rpy_of(R) -> tuple[float, float, float]:
    """urdfdom's (roll, pitch, yaw) of a rotation matrix, R = Rz(yaw) Ry(pitch) Rx(roll), stable at and near
    ±90° pitch: pitch = atan2(−R20, hypot(R00, R10)) (asin would lose half the digits there), yaw = atan2(R10, R00)
    (any yaw at the lock), and roll from what is left, Ry(pitch)ᵀ Rz(yaw)ᵀ R, so an ill-conditioned yaw near the
    lock is absorbed by the roll and the rebuilt rotation stays within rounding of R."""
    R = np.asarray(R, float)
    c = math.hypot(R[0, 0], R[1, 0])
    pitch = math.atan2(-R[2, 0], c)
    yaw = math.atan2(R[1, 0], R[0, 0]) if c > 0 else 0.0
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    M = Ry.T @ Rz.T @ R
    return math.atan2(M[2, 1], M[2, 2]), pitch, yaw


def _origin(T) -> str:
    T = np.asarray(T, float)
    r, p, y = rpy_of(T[:3, :3])
    return (f'<origin xyz="{" ".join(C.fmt(v) for v in T[:3, 3])}" '
            f'rpy="{C.fmt(r)} {C.fmt(p)} {C.fmt(y)}"/>')


class _Urdf:
    def __init__(self):
        self.links, self.joints, self.names = [], [], set()

    def link(self, name: str, visual: str = ''):
        if name in self.names:
            raise StudyError(f'Two frames of the rig would be named {name} in the URDF: rename a sensor.')
        self.names.add(name)
        self.links.append(f'  <link name={quoteattr(name)}>{visual}</link>' if visual else
                          f'  <link name={quoteattr(name)}/>')

    def joint(self, name, kind, parent, child, T, axis=None, limit=None):
        body = [f'    <parent link={quoteattr(parent)}/>', f'    <child link={quoteattr(child)}/>', f'    {_origin(T)}']
        if axis is not None:
            body.append(f'    <axis xyz="{" ".join(C.fmt(v) for v in axis)}"/>')
        if limit is not None:
            body.append('    <limit ' + ' '.join(f'{k}="{C.fmt(v)}"' for k, v in limit.items()) + '/>')
        self.joints.append(f'  <joint name={quoteattr(name)} type="{kind}">\n' + '\n'.join(body) + '\n  </joint>')

    def text(self, name: str, header: str) -> str:
        return (f'<?xml version="1.0"?>\n{header}\n<robot name={quoteattr(name)}>\n' + '\n'.join(self.links) + '\n'
                + '\n'.join(self.joints) + '\n</robot>\n')


def _records(robot) -> dict:
    return {j.get('name'): j for j in robot.joints if isinstance(j, dict) and j.get('name')}


def robot_urdf(U: _Urdf, project, robot, prefix: str, out_dir: str, options: dict, files: list, notes: list):
    """Add the robot's links and joints (module docstring) to U; its link-name map {DexVantage link: URDF link}."""
    import mujoco
    from ..meshing import stl_bytes
    from ..robot.model import get_model
    rm = get_model(project, robot.uid)
    m = rm.m
    J = mujoco.mjtJoint
    act = {int(j): i for i, j in enumerate(rm._jid)}
    recs = _records(robot)
    # the kinematics with every moving joint at qpos0 (zero rotation) and free / ball joints at home
    d = mujoco.MjData(m)
    d.qpos[:] = rm._q0
    for j in rm._jid:
        d.qpos[m.jnt_qposadr[j]] = m.qpos0[m.jnt_qposadr[j]]
    mujoco.mj_kinematics(m, d)

    def X(b):
        return _T(d.xmat[b].reshape(3, 3), d.xpos[b])

    lname = {0: C.WORLD}
    for b in range(1, m.nbody):
        lname[b] = prefix + rm.body_link[b]
    # visual meshes per link
    meshes = {}
    if options.get('meshes', True):
        for ln, V, F, rgba in rm.visual_meshes():
            meshes.setdefault(ln, []).append((V, F, rgba))
    pkg = str(options.get('package') or '').strip()
    no_lim, no_eff = [], []
    for b in range(1, m.nbody):
        own = rm.body_link[b]
        vis = ''
        if own in meshes:
            parts = meshes[own]
            Vs, Fs, off = [], [], 0
            for V, F, _c in parts:
                Vs.append(V)
                Fs.append(F + off)
                off += len(V)
            fn = f'{C.frame_name(robot.name)}_{C.frame_name(own)}.stl'
            files.append(C.write_bytes(out_dir, f'{MESHES}/{fn}', stl_bytes(np.concatenate(Vs), np.concatenate(Fs),
                                                                         f'{robot.name} {own}: m, link frame')))
            uri = f'package://{pkg}/meshes/{fn}' if pkg else f'meshes/{fn}'
            c = parts[0][2]
            vis = (f'\n    <visual>\n      <origin xyz="0 0 0" rpy="0 0 0"/>\n      <geometry><mesh filename='
                   f'{quoteattr(uri)}/></geometry>\n      <material name={quoteattr(lname[b] + "_colour")}><color '
                   f'rgba="{" ".join(C.fmt(round(float(v), 4)) for v in c)}"/></material>\n    </visual>\n  ')
        U.link(lname[b], vis)
        pb = int(m.body_parentid[b])
        base = robot.base.matrix() if pb == 0 else np.eye(4)
        jids = [int(m.body_jntadr[b]) + k for k in range(int(m.body_jntnum[b]))] if m.body_jntnum[b] > 0 else []
        held = [j for j in jids if int(m.jnt_type[j]) in (int(J.mjJNT_FREE), int(J.mjJNT_BALL))]
        if held or not jids:
            T0 = (np.linalg.inv(X(pb)) @ X(b)) if held else _T(quat_to_matrix(quat_from_mujoco(m.body_quat[b])),
                                                              m.body_pos[b])
            if held:
                notes.append(f'{robot.name}: {own} has a free or ball joint, so it is fixed at its home pose in the '
                             'URDF (DexVantage holds such joints at home too).')
            U.joint(f'{lname[pb]}_to_{lname[b]}', 'fixed', lname[pb], lname[b], base @ T0)
            continue
        T0 = _T(quat_to_matrix(quat_from_mujoco(m.body_quat[b])), m.body_pos[b])
        pend = base @ T0
        parent = lname[pb]
        for k, j in enumerate(jids):
            p = np.asarray(m.jnt_pos[j], float)
            a = np.asarray(m.jnt_axis[j], float)
            q0 = float(m.qpos0[m.jnt_qposadr[j]])
            hinge = int(m.jnt_type[j]) == int(J.mjJNT_HINGE)
            last = k == len(jids) - 1
            needs_anchor = hinge and np.linalg.norm(p) > 1e-12
            child = lname[b] if last and not needs_anchor else f'{lname[b]}__{C.frame_name(rm.joint_name(j))}'
            if hinge:
                O = pend @ _T(None, p) @ _T(_axis_angle(a, -q0))
                after = _T(None, -p)
            else:
                O = pend @ _T(None, -a * q0 / (np.linalg.norm(a) or 1.0))
                after = np.eye(4)
            if child != lname[b]:
                U.link(child)
            i = act.get(j)
            name = prefix + rm.joint_name(j)
            lo, hi = (rm.lower[i], rm.upper[i]) if i is not None else (-np.inf, np.inf)
            rec = recs.get(rm.joint_name(j)) or {}
            eff, vel = rec.get('effort'), rec.get('velocity')
            if eff is None or vel is None:
                no_eff.append(rm.joint_name(j))
            lim = {'effort': float(eff or 0.0), 'velocity': float(vel or 0.0)}
            if np.isfinite(lo) and np.isfinite(hi):
                kind = 'revolute' if hinge else 'prismatic'
                lim = {'lower': float(lo), 'upper': float(hi), **lim}
            elif hinge:
                kind = 'continuous'
                no_lim.append(rm.joint_name(j))
            else:
                raise StudyError(f'{robot.name}\'s slide joint {rm.joint_name(j)} has no limits, which URDF cannot '
                                 'write for a prismatic joint: give it limits on the Robot page.')
            U.joint(name, kind, parent, child, O, axis=a / (np.linalg.norm(a) or 1.0), limit=lim)
            parent, pend = child, after
        if parent != lname[b]:
            U.joint(f'{parent}_to_{lname[b]}', 'fixed', parent, lname[b], pend)
    # URDF links MuJoCo merged into a body (a fixed offset from it)
    for i, ln in enumerate(rm.link_names):
        if ln in rm.body_link[1:]:
            continue
        b = int(rm._lb[i])
        U.link(prefix + ln)
        U.joint(f'{lname[b]}_to_{prefix + ln}', 'fixed', lname[b], prefix + ln, rm._loff[i])
    if no_eff:
        notes.append(f'{robot.name}: the effort and velocity limits of {", ".join(no_eff)} are not stated by its file '
                     'and are written as 0 (urdfdom needs the attributes): set them before giving the file to a '
                     'controller or a planner.')
    if no_lim:
        notes.append(f'{robot.name}: {", ".join(no_lim)} have no finite limits and are written as continuous joints.')
    return {ln: prefix + ln for ln in rm.link_names}


def sensor_frames(project, cams, link_map: dict) -> list:
    """[(parent, child, T_parent_child 4 × 4, sensor, kind)] for every sensor body frame and optical frame."""
    out, seen = [], set()
    for c in cams:
        parent = link_map.get((c.robot_uid, c.parent), c.parent) if c.robot_uid else C.WORLD
        if c.sensor.uid not in seen:
            seen.add(c.sensor.uid)
            out.append((parent, c.sensor_frame, c.T_parent_sensor, c, 'sensor'))
        out.append((c.sensor_frame, c.optical_frame, c.T_sensor_optical, c, 'optical'))
    return out


def launch_text(frames: list, header: str) -> str:
    nodes = []
    for parent, child, T, _c, _k in frames:
        p = T[:3, 3]
        qx, qy, qz, qw = quat_from_matrix(T[:3, :3])
        args = ['--x', p[0], '--y', p[1], '--z', p[2], '--qx', qx, '--qy', qy, '--qz', qz, '--qw', qw]
        a = ', '.join(f"'{v}'" if isinstance(v, str) else f"'{C.fmt(v)}'" for v in args)
        nodes.append(f"        Node(package='tf2_ros', executable='static_transform_publisher',\n"
                     f"             name='static_tf_{child}',\n"
                     f"             arguments=[{a},\n"
                     f"                        '--frame-id', '{parent}', '--child-frame-id', '{child}']),")
    return (f'"""{header}"""\nfrom launch import LaunchDescription\nfrom launch_ros.actions import Node\n\n\n'
            'def generate_launch_description():\n    return LaunchDescription([\n' + '\n'.join(nodes) + '\n    ])\n')


def tree_text(frames: list, project, q_text: str, header: str) -> str:
    def vec(v):
        return '[' + ', '.join(C.fmt(x) for x in v) + ']'

    lines = [f'# {line}' for line in header.splitlines()] + ['frames:']
    for parent, child, T, c, kind in frames:
        moves = bool(c.robot_uid)
        Tw = c.T_world_optical @ np.linalg.inv(c.T_sensor_optical) if kind == 'sensor' else c.T_world_optical
        lines += [f'  - child: {child}', f'    parent: {parent}', f'    kind: {kind}',
                  f'    sensor: "{c.sensor.name}"' + (f'\n    imager: "{c.imager.name}"' if kind == 'optical' else ''),
                  f'    moves_with_robot: {"true" if moves else "false"}',
                  f'    translation: {vec(T[:3, 3])}', f'    rotation_xyzw: {vec(quat_from_matrix(T[:3, :3]))}',
                  f'    world_translation_at_export: {vec(Tw[:3, 3])}',
                  f'    world_rotation_xyzw_at_export: {vec(quat_from_matrix(Tw[:3, :3]))}']
    lines.append(f'joint_state: "{q_text}"')
    return '\n'.join(lines) + '\n'


@exporter('urdf', 'ROS 2: URDF, static transforms and tf2 tree', needs_robot=True,
          description='A URDF of the robots with every sensor\'s link and optical frames on fixed joints, a Python '
                      'launch file of static_transform_publisher nodes and a tf2 tree (YAML) of the camera frames.')
def export_urdf(project, out_dir, options, ctx):
    q, q_text = C.joint_state(project, options)
    cams = C.rig_cameras(project, q)
    files, notes = [], []
    U = _Urdf()
    U.link(C.WORLD)
    many = len(project.robots) > 1
    link_map = {}
    from ..robot.model import RobotModelError
    for k, r in enumerate(project.robots):
        ctx.report(0.1 + 0.6 * k / max(1, len(project.robots)), f'Writing {r.name}')
        prefix = f'{C.frame_name(r.name)}_' if many else ''
        try:
            mp = robot_urdf(U, project, r, prefix, out_dir, options, files, notes)
        except RobotModelError as e:
            notes.append(f'{r.name} is not in the URDF: {e}')
            continue
        link_map.update({(r.uid, ln): v for ln, v in mp.items()})
    if many:
        notes.append('The project has several robots, so each robot\'s links and joints are prefixed with its name '
                     '(for example ' + ', '.join(f'{C.frame_name(r.name)}_…' for r in project.robots) + ').')
    frames = sensor_frames(project, cams, link_map)
    for parent, child, T, _c, _k in frames:
        if child not in U.names:
            U.link(child)
        U.joint(f'{parent}_to_{child}', 'fixed', parent, child, T)
    header = (f'<!-- DexVantage rig of the project "{project.name}", rig variant "{project.variant.name}": the robots '
              'and every sensor\'s link and optical frames (REP-103). Kinematics only: no inertials, no collision '
              'geometry. -->')
    files.insert(0, C.write_text(out_dir, URDF, U.text(C.frame_name(project.name) or 'dexvantage_rig', header)))
    if frames:
        h = (f'Static transforms of the DexVantage rig "{project.name}" ({project.variant.name}): every sensor body '
             'frame on its parent link (or the world) and each imager\'s optical frame (REP-103). Run beside your own '
             'robot description: ros2 launch static_transforms.launch.py')
        files.append(C.write_text(out_dir, LAUNCH, launch_text(frames, h)))
        files.append(C.write_text(out_dir, TREE, tree_text(frames, project, q_text,
                                                           f'tf2 tree of the DexVantage rig "{project.name}": each '
                                                           'sensor frame, its parent and the static transform\n'
                                                           'translation in metres, rotation as a quaternion (x, y, z, '
                                                           'w); the world pose is at the export\'s joint state')))
    else:
        notes.append('No camera or depth sensor is in use, so the URDF has the robots only and no launch file or tf2 '
                     'tree is written.')
    if not options.get('meshes', True):
        notes.append('Meshes left out (option meshes off): the links have no visual geometry.')
    elif not str(options.get('package') or '').strip():
        notes.append('Meshes are referenced as meshes/… relative to rig.urdf; set the package option to write '
                     'package://<package>/meshes/… for RViz.')
    return {'files': files, 'notes': list(dict.fromkeys(notes))}
