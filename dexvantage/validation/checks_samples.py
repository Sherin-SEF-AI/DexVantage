"""Checks of the ALOHA 2, bimanual Panda and Unitree G1 samples (core/robot/bimanual_samples.py) against the bundled
model files compiled by MuJoCo directly (not through DexVantage's importer, its RobotModel or its sensor mounts).

- The ALOHA 2 arms: each arm is a robot of its own (its own file, written by scripts/bundle_samples.py, its base pose
  read from aloha.xml); every link of both at random joint values against aloha.xml itself (both arms in one file),
  in the left arm's base frame.
- The ALOHA 2 cameras: the four sensors' world poses (DexVantage's FK and mount chain) against MuJoCo's camera frames
  of aloha.xml (the wrist cameras) and scene_parts.xml (the overhead and worms-eye cameras of scene.xml).
- The G1's held legs and waist: every link of the upper-body file at random arm and hand values against
  g1_with_hands.xml with its free joint, legs and waist at the "stand" keyframe.
- The G1's head camera against Unitree's URDF joint d435_joint composed on MuJoCo's torso_link pose.
- The two Panda instances against panda.xml, each moved by its base pose.
"""
from __future__ import annotations

import numpy as np

from .suite import Outcome, check

TOL_TO_XML = 1e-6        # m (and rotation-matrix entries): MuJoCo's to_xml writes numbers to about 6 digits


def _T(xmat, xpos) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(xmat, float).reshape(3, 3)
    T[:3, 3] = xpos
    return T


def _set(m, d, names, q):
    for n, v in zip(names, q):
        d.qpos[m.jnt_qposadr[m.joint(n).id]] = v


def _mj_frame(project, sensor, q=None):
    """A sensor's world pose (core/scenegeom.py) as a MuJoCo camera frame: x right, y up, looking along −z."""
    from ..core.frames import quat_to_matrix
    from ..core.scenegeom import sensor_world_pose
    P = sensor_world_pose(project, sensor, q)
    R = quat_to_matrix(P.quaternion)
    return _T(np.c_[-R[:, 1], R[:, 2], -R[:, 0]], P.position)


@check('samples.aloha_arms', 'samples', 'ALOHA 2: the two arm robots against aloha.xml',
       'largest difference of any link pose (position in m, rotation-matrix entries) of the two ALOHA 2 arm robots, '
       'each its own file at its base pose, from the same body of aloha.xml at the same joint values, in the left '
       'arm\'s base frame', 'MuJoCo Menagerie aloha.xml (both arms in one file) compiled and posed by MuJoCo '
       'directly (mj_kinematics)', TOL_TO_XML, 'm', refs=('menagerie_98a70c5', 'mujoco_todorov2012'))
def _aloha_arms(ctx):
    import mujoco

    from ..core.robot import get_model, samples as S
    p = S.sample_project('aloha')
    left, right = p.robots
    ml, mr = get_model(p, left.uid), get_model(p, right.uid)
    ref = mujoco.MjModel.from_xml_path(S.robot_file('aloha', 'aloha.xml'))
    d = mujoco.MjData(ref)
    rng = ctx.rng('samples.aloha_arms')
    n = 5 if ctx.quick else 50
    worst, links = 0.0, 0
    for _ in range(n):
        ql, qr = rng.uniform(ml.lower, ml.upper), rng.uniform(mr.lower, mr.upper)
        d.qpos[:] = 0
        _set(ref, d, ml.joint_names, ql)
        _set(ref, d, mr.joint_names, qr)
        mujoco.mj_kinematics(ref, d)
        Ti = np.linalg.inv(_T(d.body('left/base_link').xmat, d.body('left/base_link').xpos))
        for m, q in ((ml, ql), (mr, qr)):
            T = m.fk(q)
            for ln in m.link_names:
                worst = max(worst, float(np.abs(T[ln] - Ti @ _T(d.body(ln).xmat, d.body(ln).xpos)).max()))
                links += 1
    return Outcome(worst, {'joint_vectors': n, 'link_poses': links, 'robots': [left.name, right.name]},
                   note='The arm files are aloha.xml with the other arm deleted and the base moved to the origin, '
                        'written back by MuJoCo\'s to_xml, which rounds numbers to about six significant digits '
                        '(measured 4.5e-7 when the files were written); 1e-6 allows that rounding and nothing more.')


@check('samples.aloha_cameras', 'samples', 'ALOHA 2: the four cameras against the model\'s cameras',
       'largest difference between each ALOHA 2 sensor\'s world pose (DexVantage forward kinematics and mount) and '
       'the MuJoCo camera it is read from (positions in m and rotation-matrix entries), at random joint values',
       'MuJoCo\'s camera frames (mj_camlight) of aloha.xml (wrist_cam_left, wrist_cam_right) and scene_parts.xml '
       '(overhead_cam, worms_eye_cam, copied from scene.xml)', TOL_TO_XML, 'm',
       refs=('aloha2_2024', 'menagerie_98a70c5', 'mujoco_camera'))
def _aloha_cameras(ctx):
    import mujoco

    from ..core.robot import get_model, samples as S
    p = S.sample_project('aloha')
    left, right = p.robots
    ml, mr = get_model(p, left.uid), get_model(p, right.uid)
    ref = mujoco.MjModel.from_xml_path(S.robot_file('aloha', 'aloha.xml'))
    sc = mujoco.MjModel.from_xml_path(S.robot_file('aloha', 'scene_parts.xml'))
    d, ds = mujoco.MjData(ref), mujoco.MjData(sc)
    mujoco.mj_kinematics(sc, ds)
    mujoco.mj_camlight(sc, ds)
    rng = ctx.rng('samples.aloha_cameras')
    n = 5 if ctx.quick else 30
    worst = 0.0
    names = {'cam_high': ('scene', 'overhead_cam'), 'cam_low': ('scene', 'worms_eye_cam'),
             'cam_left_wrist': ('arm', 'wrist_cam_left'), 'cam_right_wrist': ('arm', 'wrist_cam_right')}
    for _ in range(n):
        ql, qr = rng.uniform(ml.lower, ml.upper), rng.uniform(mr.lower, mr.upper)
        d.qpos[:] = 0
        _set(ref, d, ml.joint_names, ql)
        _set(ref, d, mr.joint_names, qr)
        mujoco.mj_kinematics(ref, d)
        mujoco.mj_camlight(ref, d)
        Ti = np.linalg.inv(_T(d.body('left/base_link').xmat, d.body('left/base_link').xpos))
        for s in p.variant.sensors:
            where, cam = names[s.name.split(' ')[0]]
            dd = ds if where == 'scene' else d
            want = Ti @ _T(dd.cam(cam).xmat, dd.cam(cam).xpos)
            worst = max(worst, float(np.abs(_mj_frame(p, s, {left.uid: ql, right.uid: qr}) - want).max()))
    return Outcome(worst, {'joint_vectors': n, 'cameras': [s.name for s in p.variant.sensors]},
                   ok=len(p.variant.sensors) == 4,
                   note='The wrist cameras ride on the arm files (to_xml rounding, as for the arms); the two frame '
                        'cameras are read from scene_parts.xml through aloha.xml\'s base pose. 1e-6 allows the '
                        'rounding only.')


@check('samples.g1_held_legs', 'samples', 'G1: the upper body against the full model at "stand"',
       'largest difference of any link pose (position in m, rotation-matrix entries) of the G1 upper-body robot (free '
       'joint, legs and waist deleted) from the same body of g1_with_hands.xml at "stand" with the same arm and '
       'hand values', 'MuJoCo Menagerie g1_with_hands.xml (as bundled, meshes simplified) compiled and posed by '
       'MuJoCo directly, its free joint, legs and waist at the "stand" keyframe', TOL_TO_XML, 'm',
       refs=('menagerie_98a70c5', 'mujoco_todorov2012'))
def _g1_held_legs(ctx):
    import mujoco

    from ..core.robot import get_model, samples as S
    p = S.sample_project('g1')
    r = p.robots[0]
    m = get_model(p, r.uid)
    ref = mujoco.MjModel.from_xml_path(S.robot_file('unitree_g1', 'g1_with_hands.xml'))
    d = mujoco.MjData(ref)
    stand = ref.key_qpos[ref.key('stand').id]
    rng = ctx.rng('samples.g1_held_legs')
    n = 5 if ctx.quick else 50
    worst = 0.0
    for q in [m.home] + [rng.uniform(m.lower, m.upper) for _ in range(n)]:
        d.qpos[:] = stand
        _set(ref, d, m.joint_names, q)
        mujoco.mj_kinematics(ref, d)
        T = m.fk(q)
        for ln in m.link_names:
            worst = max(worst, float(np.abs(T[ln] - _T(d.body(ln).xmat, d.body(ln).xpos)).max()))
    return Outcome(worst, {'joint_vectors': n + 1, 'joints': m.nq, 'links': len(m.link_names)},
                   ok=m.nq == 28,
                   note='The upper-body file deletes joints that are 0 at "stand", so their bodies stay where the '
                        'keyframe puts them; what remains is MuJoCo\'s to_xml rounding (1e-6).')


@check('samples.g1_head_camera', 'samples', 'G1: the head camera against Unitree\'s URDF',
       'largest difference between the G1 head camera\'s world pose (DexVantage) and Unitree\'s d435_joint origin '
       'composed on torso_link (positions in m and rotation-matrix entries)', 'Unitree\'s g1_29dof_rev_1_0.urdf '
       'joint d435_joint (xyz, rpy as URDF writes them: R = Rz Ry Rx) on the torso_link pose of g1_with_hands.xml '
       'posed by MuJoCo at "stand"', 1e-12, 'm', refs=('unitree_g1_urdf', 'menagerie_98a70c5'))
def _g1_head(ctx):
    import mujoco

    from ..core.frames import quat_to_matrix
    from ..core.robot import samples as S
    from ..core.scenegeom import sensor_world_pose
    p = S.sample_project('g1')
    head = p.variant.sensors[0]
    ref = mujoco.MjModel.from_xml_path(S.robot_file('unitree_g1', 'g1_with_hands.xml'))
    d = mujoco.MjData(ref)
    d.qpos[:] = ref.key_qpos[ref.key('stand').id]
    mujoco.mj_kinematics(ref, d)
    a = 0.8307767239493009
    Ry = np.array([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]])
    T = _T(d.body('torso_link').xmat, d.body('torso_link').xpos)
    want = T @ _T(Ry, [0.0576235, 0.01753, 0.42987])
    P = sensor_world_pose(p, head)
    got = _T(quat_to_matrix(P.quaternion), P.position)
    return Outcome(float(np.abs(got - want).max()), {'sensor': head.name, 'model_id': head.model_id},
                   ok=head.mount.parent_frame == 'torso_link',
                   note='Both sides compose the same figures by different routes (a quaternion through the mount '
                        'against the URDF\'s rotation written out here); only floating-point rounding remains.')


@check('samples.panda_instances', 'samples', 'Bimanual Panda: both instances against panda.xml',
       'largest difference of any link pose (position in m, rotation-matrix entries) of the two Panda robots from '
       'panda.xml\'s bodies at the same joint values, moved by each robot\'s base pose',
       'MuJoCo Menagerie franka_emika_panda/panda.xml (as bundled: OBJ meshes as STL, bodies unchanged) compiled and '
       'posed by MuJoCo directly', 1e-9, 'm', refs=('menagerie_98a70c5', 'mujoco_todorov2012'))
def _panda(ctx):
    import mujoco

    from ..core.robot import get_model, samples as S
    from ..core.robot.bimanual_samples import PANDA_FILE
    p = S.sample_project('panda-bimanual')
    ref = mujoco.MjModel.from_xml_path(S.robot_file(*PANDA_FILE))
    d = mujoco.MjData(ref)
    rng = ctx.rng('samples.panda_instances')
    n = 5 if ctx.quick else 50
    worst = 0.0
    for r in p.robots:
        m = get_model(p, r.uid)
        B = r.base.matrix()
        for q in [m.home] + [rng.uniform(m.lower, m.upper) for _ in range(n)]:
            _set(ref, d, m.joint_names, q)
            mujoco.mj_kinematics(ref, d)
            T = m.fk(q)
            for ln in m.link_names:
                worst = max(worst, float(np.abs(T[ln] - B @ _T(d.body(ln).xmat, d.body(ln).xpos)).max()))
    return Outcome(worst, {'robots': [r.name for r in p.robots], 'joint_vectors': 2 * (n + 1)},
                   ok=len(p.robots) == 2 and p.robots[0].source.sha256 == p.robots[1].source.sha256,
                   note='The file\'s bodies are not rewritten (only its mesh file names), so only the base pose\'s '
                        'floating-point composition remains; 1e-9 m is far below any figure the studies report.')
