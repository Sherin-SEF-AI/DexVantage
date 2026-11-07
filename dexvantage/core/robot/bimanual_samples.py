"""The bimanual and humanoid samples (spec §11 --sample aloha|panda-bimanual|g1, §14.1, §14.4, §14.5).

    build_aloha() -> Project          (through core.robot.samples.sample_project('aloha'))
    build_panda() -> Project          ('panda-bimanual')
    build_g1() -> Project             ('g1')

The robots come from MuJoCo Menagerie at commit 98a70c5 (each folder's own LICENSE read 2026-11-07, kept beside the
files, recorded in THIRD_PARTY.md and samples.BUNDLED), copied and prepared by scripts/bundle_samples.py, which says
exactly what it changes. Joint limits, link geometry and home poses are the files' own; nothing is typed in again.

ALOHA 2 ('aloha'). Menagerie's aloha.xml is both ViperX 300 arms with the ALOHA 2 grippers in one file; the sample
opens each arm as a robot of its own (resources/robots/aloha_left.xml and aloha_right.xml: one arm each, its base at
the file's origin), so the studies that work per gripper (grasps, teleop, the hand-eye plan) see two grippers, and the
bimanual parts of the teleop study (reach overlap, handover, arm-to-arm margin) and the grasp sampler's bimanual
handoff see two arms. Both bases are placed where aloha.xml puts them, read from aloha.xml as the sample is built: the
left arm is the primary robot (the world is its base frame, REP-103), the right arm stands 0.938 m along its x axis,
turned half a turn. Joint names are aloha.xml's ("left/waist", "right/waist", ...), so a recording with both arms in
one state vector (ALOHA's 14 values) is read onto each arm by name (core/datasets/joints.py). The cameras are the four
of the ALOHA 2 hardware: "We upgrade the cameras used in the ALOHA system to 4 RealSense D405 cameras" and "All four
cameras record 848 x 480 rgb images" (ALOHA 2 paper, arXiv 2405.02292v1 §2.5, read 2026-11-07), placed as Menagerie's
model places them ("We precisely model all cameras and robot positions of the ALOHA 2 specification", the same paper,
on its MuJoCo model): the wrist cameras of aloha.xml and the overhead and worms-eye cameras of scene.xml (copied to
aloha/scene_parts.xml with the table's collision box). They carry the names of the ALOHA recordings ('camera_names':
['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist'], tonyzhaozh/aloha aloha_scripts/constants.py, read
2026-11-07). The teleop devices are ALOHA's leader arms, from the library's entry aloha-widowx-250-leader
(core/teleop_devices.py): "two ViperX 6-DoF arms ... along with 2 smaller WidowX arms ... The WidowX contains the same
kinematic structure as the ViperX ... The follower joints are synchronized with the leader arms" (paper §1); each
leads its follower joint to joint, with the follower's joint ranges (the WidowX's are not stated; the record says so).

Franka Panda bimanual ('panda-bimanual'). There is no official bimanual Franka model: the cell is two instances of
Menagerie's single-arm franka_emika_panda/panda.xml (with the Franka Hand), side by side 0.8 m apart, both facing +x
(a sample choice). Each hand carries a wrist RealSense D405 and a RealSense D435 looks at the cell from the front
(sample placements). The teleop devices are the library's GELLO entry, which has "the same kinematic structure as the
target arm" (Wu et al. 2023).

Unitree G1 upper body ('g1'). Menagerie's g1_with_hands.xml (both arms with a three-finger hand each; Menagerie does
not name the hand, so it is not called Dex3 here). Its meshes are SIMPLIFIED to fit the package (31.5 MB of STL):
scripts/bundle_samples.py reduces each mesh by quadric edge collapse while its two-sided Hausdorff distance from the
original stays within 0.5 mm (G1_SIMPLIFICATION records the measured figures). The legs and the waist are held at
the file's "stand" keyframe: resources/robots/g1_upper_body.xml has the free joint, the 12 leg joints and the 3
waist joints deleted (all at 0 in "stand", the pelvis at the keyframe's 0.79 m), so q is both arms (7 joints each)
and both hands (7 each), 28 values in one state vector; moving one arm leaves the other put (no shared waist joint).
The head camera is where Unitree's own URDF puts it: g1_29dof_rev_1_0.urdf (unitreerobotics/unitree_ros, commit
7c40519, read 2026-11-07) has a frame "d435_link" on torso_link at xyz (0.0576235, 0.01753, 0.42987) m, rpy
(0, 0.8307767239493009, 0) rad; Unitree's G1 page states only "Depth Camera+3D LiDAR", so the RealSense D435 is the
model the frame's name gives, and the frame is taken as the camera's body frame (x along the optical axis; the URDF
does not say which point of the camera it marks). The gripper frame and fingertips are the right hand's (one gripper
per robot; the palm is a geom of right_wrist_yaw_link, so that link is the gripper frame); the left hand is in the
model and moves with q. The upper-body file excludes contact between each palm and the thumb's middle link: Menagerie's
own collision meshes of the two overlap by 0.27 mm at "stand", which would make every arm pose a self-collision. The
teleop device is the library's Apple Vision Pro entry, as Unitree's xr_teleoperate drives the G1; no source states its
tracking volume, so none is set.

Every sample carries task volumes, a scripted trajectory, grasp sets and a policy, all labelled "(sample)": sample
choices so every study has something to analyse, not published figures.

Known limits: the ALOHA 2 grippers' two fingers are coupled by an equality constraint in aloha.xml that the arm files
leave out (kinematics treats each finger joint on its own, as for the 2F-85); the WidowX leader is modelled with the
follower's own joint ranges; the G1's simplified meshes differ from Menagerie's by up to 0.5 mm, its legs and waist do
not move, and its pelvis is fixed in the world (a humanoid that steps or bends at the waist is not modelled); the Panda
cell's layout and every camera placement without a citation are sample choices.
"""
from __future__ import annotations

import json
import os

import numpy as np

from ..frames import BODY_TO_OPTICAL, Pose, quat_from_matrix, quat_from_mujoco, quat_from_rpy, quat_to_matrix
from ..project import Mount, Policy, Project, Sensor, Volume, new_uid
from . import samples as S
from .model import get_model

OPT_FROM_MUJOCO = np.diag([1.0, -1.0, -1.0])         # MuJoCo camera (−z forward, y up) -> optical (z forward, y down)
ALOHA_LEFT, ALOHA_RIGHT = 'aloha_left.xml', 'aloha_right.xml'
PANDA_FILE = ('franka_emika_panda', 'panda.xml')
G1_FILE = 'g1_upper_body.xml'

SOURCES = {
    'aloha2_paper': {'url': 'https://arxiv.org/html/2405.02292v1', 'read': '2026-11-07',
                     'title': 'ALOHA 2 Team, "ALOHA 2: An Enhanced Low-Cost Hardware for Bimanual Teleoperation", 2024',
                     'quotes': ['We upgrade the cameras used in the ALOHA system to 4 RealSense D405 cameras.',
                                'All four cameras record 848 x 480 rgb images.',
                                'We precisely model all cameras and robot positions of the ALOHA 2 specification',
                                'The WidowX contains the same kinematic structure as the ViperX, in a smaller form '
                                'factor. The follower joints are synchronized with the leader arms',
                                'The robots are mounted on a 48" x 30" table']},
    'aloha_constants': {'url': 'https://github.com/tonyzhaozh/aloha/blob/main/aloha_scripts/constants.py',
                        'read': '2026-11-07',
                        'quote': "'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']"},
    'menagerie_aloha': {'url': f'{S.MENAGERIE}/tree/{S.MENAGERIE_COMMIT}/aloha', 'read': '2026-11-07',
                        'quote': 'Added 4 cameras: overhead camera, worms-eye camera and left and right wrist cameras. '
                                 'The intrinsic parameters of the simulated cameras are matched with an Intel '
                                 'RealSense D405 (aloha/README.md)'},
    'unitree_urdf': {'url': 'https://github.com/unitreerobotics/unitree_ros/blob/'
                            '7c40519e02d7dd16c06b25fe3fa3b67fdeb7cd74/robots/g1_description/g1_29dof_rev_1_0.urdf',
                     'read': '2026-11-07',
                     'quote': '<joint name="d435_joint" type="fixed"> <origin xyz="0.0576235 0.01753 0.42987" '
                              'rpy="0 0.8307767239493009 0"/> <parent link="torso_link"/> <child link="d435_link"/>'},
    'unitree_g1_page': {'url': 'https://www.unitree.com/g1/', 'read': '2026-11-07',
                        'quote': 'Sensing Sensor: Depth Camera+3D LiDAR'},
    'xr_teleoperate': {'url': 'https://github.com/unitreerobotics/xr_teleoperate', 'read': '2026-11-07',
                       'quote': 'Wear your XR headset (e.g. Apple Vision Pro, PICO4, etc.)'},
}

# the G1 head camera frame in torso_link, from Unitree's URDF (SOURCES['unitree_urdf'])
G1_D435_XYZ = (0.0576235, 0.01753, 0.42987)
G1_D435_RPY = (0.0, 0.8307767239493009, 0.0)
# measured by scripts/bundle_samples.py when the meshes were simplified (2026-11-07; THIRD_PARTY.md)
G1_SIMPLIFICATION = {'tolerance_mm': 0.5, 'largest_mm': 0.497, 'triangles_before': 629338, 'triangles_after': 216325,
                     'megabytes_before': 31.47, 'megabytes_after': 10.82}

_CACHE: dict = {}          # (sample, set) -> (grasp set JSON, {uid placeholder: uid}) of the first build


# ====================================================================== helpers
def mujoco_camera_mount(m, name: str) -> tuple[str, Pose]:
    """(the body a camera of the compiled MjModel m rides on, the sensor body frame (x forward, y left, z up) in that
    body's frame) for a camera of mode 'fixed'. MuJoCo cameras look along their −z with +y up."""
    cid = m.camera(name).id
    body = m.body(int(m.cam_bodyid[cid])).name or 'world'
    R = quat_to_matrix(quat_from_mujoco(m.cam_quat[cid])) @ OPT_FROM_MUJOCO @ BODY_TO_OPTICAL
    return body, Pose(np.array(m.cam_pos[cid], float), quat_from_matrix(R))


def _matrix(pos, quat_wxyz) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(quat_from_mujoco(np.asarray(quat_wxyz, float)))
    T[:3, 3] = np.asarray(pos, float)
    return T


def aloha_frames() -> dict:
    """aloha.xml's two base_link poses and scene_parts.xml's cameras and table, in aloha.xml's world frame (4 × 4),
    read with MuJoCo from the bundled files: {'left', 'right', 'cam:<name>', 'table': (centre, half sizes)}."""
    import mujoco
    spec = mujoco.MjSpec.from_file(S.robot_file('aloha', 'aloha.xml'))       # parsed only: no mesh is read
    out = {}
    for side in ('left', 'right'):
        b = spec.body(f'{side}/base_link')
        if b.alt.type != mujoco.mjtOrientation.mjORIENTATION_QUAT:
            raise S.SampleError('aloha.xml gives the base orientation in a form this sample does not read.')
        out[side] = _matrix(b.pos, b.quat)
    m = mujoco.MjModel.from_xml_path(S.robot_file('aloha', 'scene_parts.xml'))
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    mujoco.mj_camlight(m, d)                           # camera poses are computed here, not by mj_kinematics
    for c in range(m.ncam):
        T = np.eye(4)
        T[:3, :3] = d.cam_xmat[c].reshape(3, 3)
        T[:3, 3] = d.cam_xpos[c]
        out[f'cam:{m.camera(c).name}'] = T
    g = m.geom('table').id
    out['table'] = (np.array(d.geom_xpos[g], float), np.array(m.geom_size[g], float))
    return out


def _import(path: str, name: str, license_text: str, note: str, primary: bool = True, base=None):
    """(Robot, sources) as samples._import reads them, without compiling a third time (the project's get_model
    compiles once per file content and shares it)."""
    from . import import_robot
    robot, sources = import_robot(path, name)
    robot.primary = primary
    robot.source.license = license_text
    robot.source.note = ' '.join(x for x in (robot.source.note, note) if x)
    if base is not None:
        robot.base = Pose.from_matrix(base)
    return robot, sources


def _copy_robot(robot, sources: dict, name: str, base) -> tuple:
    """A second instance of an imported robot: the same records and files under a new uid (no second import)."""
    from ..project import Robot
    uid = new_uid()
    old = robot.uid + '/'
    text = json.dumps(robot.to_dict()).replace(json.dumps(old)[1:-1], uid + '/')
    twin = Robot.from_dict(json.loads(text))
    twin.uid, twin.name, twin.primary = uid, name, False
    twin.base = Pose.from_matrix(base)
    return twin, {uid + '/' + k[len(old):]: v for k, v in sources.items() if k.startswith(old)}


def _gripper_q(model, q, names, fraction: float) -> np.ndarray:
    """q with the named gripper joints at `fraction` of their range from the lower limit."""
    q = np.array(q, float)
    for n in names:
        i = model.joint_names.index(n)
        q[i] = model.lower[i] + fraction * (model.upper[i] - model.lower[i])
    return q


def _reach(model, link: str, tcp_local, goals, t, grip, grip_joints, robot_uid: str, name: str, start=None,
           others=None) -> dict:
    """A scripted trajectory (a sample choice): the TCP (tcp_local in `link`) goes through the world points `goals`
    (None: stay), each solved by position-only IK from the waypoint before (core/robot/ik.py, nearest to it,
    self-collision off as for the UR5e's pick); the gripper joints at `grip` (fraction of their range) at each
    waypoint. others: extra (link, tcp, goals) chains of the same robot solved in turn at each waypoint (the G1's
    second arm), so one state vector moves both arms."""
    from .ik import solve
    names = [str(n) for n in model.joint_names]
    chains = [(link, tcp_local, goals)] + list(others or [])
    rows, q = [], np.array(model.home if start is None else start, float)
    for k, g in enumerate(grip):
        for ln, tl, gs in chains:
            goal = gs[k]
            if goal is None:
                continue
            T = np.eye(4)
            T[:3, 3] = tl
            r = solve(model, ln, np.asarray(goal, float), q0=q, tcp=T, rule='nearest_q0', rng=0, collision=False)
            if not r.ok:
                raise S.SampleError(f'The sample trajectory "{name}" could not be solved ({r.reason}).')
            q = np.array(r.q, float)
        q = _gripper_q(model, q, grip_joints, g) if grip_joints else q
        rows.append([float(v) for v in model.clip(q)])
    return {'uid': new_uid(), 'name': name, 'robot_uid': robot_uid, 'joints': names, 't': [float(x) for x in t],
            'q': rows}


def _tcp(project, robot_uid: str):
    """(gripper frame, TCP in it) of core/grasps.gripper_geometry, kept per robot file and frames for this process."""
    from .. import grasps as GR
    r = project.robot(robot_uid)
    key = ('tcp', r.source.sha256, r.gripper_frame, tuple(r.fingertip_frames), json.dumps(r.contact_pads))
    if key not in _CACHE:
        g = GR.gripper_geometry(project, robot_uid)
        _CACHE[key] = (g['frame'], np.asarray(g['tcp'], float))
    link, tcp = _CACHE[key]
    return link, tcp.copy()


def _trajectory(key: str, robot_uid: str, make) -> dict:
    """make() once per process (the scripted IK depends only on the bundled robot and the sample's volumes); later
    openings get a copy with their own uids."""
    hit = _CACHE.get(('traj', key))
    if hit is None:
        tr = make()
        _CACHE[('traj', key)] = (json.dumps(tr), tr['robot_uid'])
        return tr
    tr = json.loads(hit[0])
    tr['uid'], tr['robot_uid'] = new_uid(), robot_uid
    return tr


def _tcp_world(model, link, tcp, q=None) -> np.ndarray:
    T = model.link_pose(model.home if q is None else q, link)
    return T[:3, :3] @ tcp + T[:3, 3]


GRASP_HEIGHT_M = 0.06       # the TCP above the table for grasps without an object (a sample choice: the middle of
#                             a 12 cm object; at the sampler's 30 mm default the ALOHA and Panda hands meet the table)


def _grasps(p: Project, key: str, robot, kinds: tuple, n: int, name: str, volume=None, handover=None):
    """A sampled grasp set (core/grasps.sample, seed 0) added to the project; sampled once per process and given the
    new project's uids after that (the poses depend only on the bundled robots and the volumes)."""
    from .. import grasps as GR
    uids = {'robot': robot.uid, 'volume': volume.uid if volume else '', 'handover': handover.uid if handover else ''}
    uids.update({f'r{i}': r.uid for i, r in enumerate(p.robots)})
    hit = _CACHE.get((key, name))
    if hit is None:
        gs = GR.sample(p, robot.uid, kinds=kinds, n=n, rng_seed=0, volume_uid=volume.uid if volume else None,
                       handover_uid=handover.uid if handover else None, name=name, height_m=GRASP_HEIGHT_M)
        _CACHE[(key, name)] = (json.dumps(gs), uids)
    else:
        text, old = hit
        for k, v in old.items():
            if v and uids.get(k):
                text = text.replace(v, uids[k])
        gs = json.loads(text)
        gs['uid'] = new_uid()
    GR.add_set(p, gs)


def _library_devices(p: Project, library_id: str, followers: list, why: str = '') -> None:
    """Teleop devices from a library entry (core/teleop_devices.device_from_library: its stated figures with their
    sources), one per (follower robot, name); why says where the sample's choice of device comes from."""
    from .. import teleop_devices as TD
    for robot, name in followers:
        d = TD.device_from_library(p, library_id, robot.uid)
        d['name'] = name
        if why:
            d['source']['sample'] = why
        p.tasks.teleop_devices.append(d)


def _surface(name: str, z: float, size, center, note: str = '') -> dict:
    s = {'uid': new_uid(), 'name': name, 'z': float(z), 'size': [float(v) for v in size],
         'center': [float(v) for v in center], 'material': 'matte'}
    if note:
        s['note'] = note
    return s


def _vol(name: str, kind: str, center, size) -> Volume:
    return Volume(name=name, kind=kind, center=tuple(float(v) for v in np.round(center, 3) + 0.0),
                  size=tuple(float(v) for v in size))


# ====================================================================== ALOHA 2
def build_aloha() -> Project:
    F = aloha_frames()
    TL_inv = np.linalg.inv(F['left'])
    lic = 'BSD-3-Clause (Menagerie aloha/LICENSE, © 2023 Trossen Robotics)'
    note = (S._source_note(['aloha']) + ' One ViperX 300 arm of aloha.xml (scripts/bundle_samples.py writes it with '
            'mujoco.MjSpec: the other arm, actuators, equality constraints and keyframes deleted, its base at the '
            'file\'s origin); the robot\'s base pose is aloha.xml\'s.')
    left, src_l = _import(S.robot_file(ALOHA_LEFT), 'ALOHA 2 left arm', lic, note)
    right, src_r = _import(S.robot_file(ALOHA_RIGHT), 'ALOHA 2 right arm', lic, note, False, TL_inv @ F['right'])
    p = Project('ALOHA 2 sample')
    p.robots, p.sources = [left, right], {**src_l, **src_r}
    for r, side in ((left, 'left'), (right, 'right')):
        r.gripper_frame = f'{side}/gripper_base'
        r.fingertip_frames = [f'{side}/left_finger_link', f'{side}/right_finger_link']
        S._tag_gripper(r, {f'{side}/left_finger', f'{side}/right_finger'})
    ml, mr = get_model(p, left.uid), get_model(p, right.uid)
    # the table: scene.xml's collision box (48" x 30" in the paper), in the left arm's base frame
    tc, th = F['table']
    top = TL_inv @ np.r_[tc + [0.0, 0.0, th[2]], 1.0]
    p.workspace.surfaces.append(_surface('Table (ALOHA 2 scene.xml)', top[2], 2 * th[:2], top[:2],
                                         'the collision box "table" of Menagerie aloha/scene.xml'))
    # task volumes (sample choices): the table between the arms and a handover zone above its middle
    mid = TL_inv @ np.r_[tc[0], tc[1], tc[2] + th[2], 1.0]
    table_vol = _vol('Tabletop between the arms (sample)', 'tabletop', mid[:3] + [0.0, 0.0, 0.10], (0.50, 0.40, 0.20))
    hand_vol = _vol('Handover zone (sample)', 'handover', mid[:3] + [0.0, 0.0, 0.25], (0.20, 0.20, 0.15))
    p.workspace.volumes += [table_vol, hand_vol]
    # the four ALOHA 2 cameras: D405s at 848 x 480, posed as Menagerie's model poses them
    cams = []
    for slot, (cam, title) in enumerate((('overhead_cam', 'cam_high (overhead)'),
                                         ('worms_eye_cam', 'cam_low (worms-eye)'))):
        T = TL_inv @ F[f'cam:{cam}'] @ np.c_[np.r_[OPT_FROM_MUJOCO @ BODY_TO_OPTICAL, np.zeros((1, 3))],
                                             [0, 0, 0, 1]]
        cams.append(Sensor(name=title, kind='depth', slot=slot, model_id='intel-realsense-d405', stream=[848, 480],
                           mount=Mount('cage', '', Pose.from_matrix(T))))
    for slot, (r, m, side) in enumerate(((left, ml, 'left'), (right, mr, 'right')), start=2):
        body, pose = mujoco_camera_mount(m.m, f'wrist_cam_{side}')
        cams.append(Sensor(name=f'cam_{side}_wrist', kind='depth', slot=slot, model_id='intel-realsense-d405',
                           stream=[848, 480], mount=Mount(body, r.uid, pose)))
    p.variant.name = 'ALOHA 2 four cameras'
    p.variant.sensors = cams
    p.variant.note = ('The four ALOHA 2 cameras (four RealSense D405, 848 × 480, ALOHA 2 paper §2.5) where MuJoCo '
                      'Menagerie\'s model places them; named as the ALOHA recordings name them.')
    # leader arms: the library's ALOHA leader (WidowX-250, core/teleop_devices.py), which leads the ViperX joint to
    # joint; the paper adds that "The WidowX contains the same kinematic structure as the ViperX"
    _library_devices(p, 'aloha-widowx-250-leader', [(left, 'ALOHA 2 leader arm, left'),
                                                    (right, 'ALOHA 2 leader arm, right')],
                     'ALOHA 2 paper, ' + SOURCES['aloha2_paper']['url'] + ' (read 2026-11-07): "The WidowX contains '
                     'the same kinematic structure as the ViperX, in a smaller form factor. The follower joints are '
                     'synchronized with the leader arms"')
    # a handover (sample choreography): each arm brings its TCP to the handover zone's centre, 50 mm short of it
    # along the line between the bases, the right gripper closes, both go home
    hc = np.asarray(hand_vol.center, float)
    axis = np.asarray(right.base.position, float) - np.asarray(left.base.position, float)
    axis[2] = 0.0
    axis /= np.linalg.norm(axis)
    for r, m, side, sgn, grip in ((left, ml, 'left', -1.0, [0.6, 0.6, 0.6, 0.6, 0.6]),
                                  (right, mr, 'right', 1.0, [0.6, 0.6, 0.6, 0.1, 0.1])):
        link, tcp = _tcp(p, r.uid)
        meet = hc + sgn * 0.05 * axis
        pre = meet + sgn * 0.10 * axis + [0.0, 0.0, 0.05]
        p.tasks.trajectories.append(_trajectory(f'aloha-{side}', r.uid, lambda: _reach(
            m, link, tcp, [None, pre, meet, None, _tcp_world(m, link, tcp)], [0.0, 1.0, 2.0, 2.5, 4.0], grip,
            [f'{side}/left_finger', f'{side}/right_finger'], r.uid, f'Handover, {side} arm (sample)')))
    _grasps(p, 'aloha', left, ('top_down', 'angled'), 100, 'Left arm, tabletop grasps (sample)', table_vol)
    _grasps(p, 'aloha', right, ('top_down', 'angled'), 100, 'Right arm, tabletop grasps (sample)', table_vol)
    _grasps(p, 'aloha', left, ('handoff',), 60, 'Left arm, handoff to the right arm (sample)', handover=hand_vol)
    p.policies.append(Policy(name='ACT (sample)', library_id='act'))
    p.notes = ('Sample (spec §14.1): ALOHA 2 from MuJoCo Menagerie aloha (BSD-3-Clause), each ViperX 300 arm a robot '
               'of its own at aloha.xml\'s base poses. The four cameras are the ALOHA 2 hardware\'s (four RealSense '
               'D405 at 848 × 480, the ALOHA 2 paper) where Menagerie\'s model places them; the leader arms are '
               'ALOHA\'s, modelled with the followers\' own joints. The table is scene.xml\'s; the task volumes, the '
               'handover trajectory, the grasp sets and ACT are sample choices.')
    p.add_log('sample', 'Opened the ALOHA 2 sample')
    return p


# ====================================================================== Franka Panda bimanual
PANDA_SPACING_M = 0.8                  # the second arm's base along −y (a sample choice)


def build_panda() -> Project:
    lic = 'Apache-2.0 (Menagerie franka_emika_panda/LICENSE)'
    note = (S._source_note(['franka_emika_panda']) + ' The OBJ meshes are bundled as binary STL with the same '
            'triangles (scripts/bundle_samples.py). There is no official bimanual Franka model: this is one of two '
            'instances of the single-arm panda.xml.')
    path = S.robot_file(*PANDA_FILE)
    a, src_a = _import(path, 'Panda left (sample cell)', lic, note)
    base_b = np.eye(4)
    base_b[1, 3] = -PANDA_SPACING_M
    b, src_b = _copy_robot(a, src_a, 'Panda right (sample cell)', base_b)
    p = Project('Franka Panda bimanual sample')
    p.robots, p.sources = [a, b], {**src_a, **src_b}
    for r in (a, b):
        r.gripper_frame = 'hand'
        r.fingertip_frames = ['left_finger', 'right_finger']
        S._tag_gripper(r, {'finger_joint1', 'finger_joint2'})
    ma, mb = get_model(p, a.uid), get_model(p, b.uid)
    mid_y = -PANDA_SPACING_M / 2
    p.workspace.surfaces.append(_surface('Table (sample)', 0.0, (0.8, 1.6), (0.55, mid_y)))
    table_vol = _vol('Tabletop in front of both arms (sample)', 'tabletop', (0.55, mid_y, 0.12), (0.40, 0.80, 0.24))
    hand_vol = _vol('Handover zone (sample)', 'handover', (0.50, mid_y, 0.40), (0.20, 0.20, 0.15))
    p.workspace.volumes += [table_vol, hand_vol]
    # wrist D405s: on the hand, 0.07 m off the tool axis along the hand's x (across the fingers' closing line), on
    # the side away from the other arm, 0.02 m along it, looking at the pinch point (the TCP measured from the finger
    # pads); a front D435 0.85 m in front of the table
    sensors = []
    for slot, (r, m, side, sx) in enumerate(((a, ma, 'left', 1.0), (b, mb, 'right', -1.0))):
        link, tcp = _tcp(p, r.uid)
        sensors.append(Sensor(name=f'Wrist D405, {side} (sample placement)', kind='depth', slot=slot,
                              model_id='intel-realsense-d405',
                              mount=Mount(link, r.uid, S.look_at([sx * 0.07, 0.0, 0.02], tcp, up=(sx, 0.0, 0.0)))))
    look = np.array([0.55, mid_y, 0.15])
    sensors.append(Sensor(name='Front D435 (sample placement)', kind='depth', slot=2, model_id='intel-realsense-d435',
                          mount=Mount('tripod', '', S.look_at(look + [0.85, 0.0, 0.50], look))))
    p.variant.name = 'Two wrist D405 + front D435 (sample)'
    p.variant.sensors = sensors
    p.variant.note = S.SAMPLE_NOTE
    _library_devices(p, 'gello', [(a, 'GELLO, left arm (sample)'), (b, 'GELLO, right arm (sample)')])
    hc = np.asarray(hand_vol.center, float)
    for r, m, side, sgn, grip in ((a, ma, 'left', 1.0, [1.0, 1.0, 1.0, 1.0, 1.0]),
                                  (b, mb, 'right', -1.0, [1.0, 1.0, 1.0, 0.3, 0.3])):
        link, tcp = _tcp(p, r.uid)
        meet = hc + [0.0, sgn * 0.05, 0.0]
        pre = meet + [0.0, sgn * 0.10, 0.05]
        p.tasks.trajectories.append(_trajectory(f'panda-{side}', r.uid, lambda: _reach(
            m, link, tcp, [None, pre, meet, None, _tcp_world(m, link, tcp)], [0.0, 1.0, 2.0, 2.5, 4.0], grip,
            ['finger_joint1', 'finger_joint2'], r.uid, f'Handover, {side} arm (sample)')))
    _grasps(p, 'panda', a, ('top_down', 'side', 'angled'), 100, 'Left arm, tabletop grasps (sample)', table_vol)
    _grasps(p, 'panda', b, ('top_down', 'side', 'angled'), 100, 'Right arm, tabletop grasps (sample)', table_vol)
    _grasps(p, 'panda', a, ('handoff',), 60, 'Left arm, handoff to the right arm (sample)', handover=hand_vol)
    # Diffusion Policy's real-world setting takes two camera images (its library entry, n_cameras): the two wrist
    # views here (a sample choice; the front D435 still counts for coverage and depth)
    p.policies.append(Policy(name='Diffusion Policy (sample)', library_id='diffusion-policy-cnn',
                             cameras=[s.uid for s in sensors if s.name.startswith('Wrist')]))
    p.notes = ('Sample (spec §14.4): two Franka Panda arms with the Franka Hand from MuJoCo Menagerie '
               'franka_emika_panda (Apache-2.0) in one cell. There is no official bimanual Franka model: these are '
               'two instances of the single-arm panda.xml, 0.8 m apart, both facing +x (a sample layout). '
               f'{S.SAMPLE_NOTE} The cameras name their library devices (Intel RealSense D405 and D435); the GELLO '
               'leaders, the task volumes, the handover trajectory, the grasp sets and Diffusion Policy (fed the two '
               'wrist views) are sample choices.')
    p.add_log('sample', 'Opened the Franka Panda bimanual sample')
    return p


# ====================================================================== Unitree G1 upper body
G1_RIGHT_TIPS = ['right_hand_thumb_2_link', 'right_hand_index_1_link', 'right_hand_middle_1_link']
G1_LEFT_TIPS = ['left_hand_thumb_2_link', 'left_hand_index_1_link', 'left_hand_middle_1_link']


def g1_hand_joints(model, side: str) -> list:
    return [n for n in model.joint_names if n.startswith(f'{side}_hand_')]


def _g1_reach(model, robot_uid: str, rl: str, rtcp, table_vol) -> dict:
    """Both palms to points over the table (the right 0.1 m to the right of the volume's centre, the left 0.1 m to
    the left, 80 mm above them first), the right hand closing along its power posture (core/dexhand.py
    default_synergies), back home: one trajectory, both arms in one state vector (a sample choice)."""
    from .. import dexhand as DH
    ltcp = rtcp * np.array([1.0, -1.0, 1.0])            # the left hand mirrors the right one in its palm frame
    ll = 'left_wrist_yaw_link'
    r0, l0 = _tcp_world(model, rl, rtcp), _tcp_world(model, ll, ltcp)
    c = np.asarray(table_vol.center, float)
    r_goal, l_goal = c + [0.0, -0.10, 0.0], c + [0.0, 0.10, 0.0]
    tr = _reach(model, rl, rtcp, [None, r_goal + [0, 0, 0.08], r_goal, None, r0], [0.0, 1.0, 2.0, 2.5, 4.0],
                [None] * 5, [], robot_uid, 'Both hands reach over the table (sample)',
                others=[(ll, ltcp, [None, l_goal + [0, 0, 0.08], l_goal, None, l0])])
    try:
        syn = DH.default_synergies(model, DH.structure_of(model, robot_uid, G1_RIGHT_TIPS))
        power = next((s for s in syn if s.id == 'power'), syn[0] if syn else None)
    except Exception:                                  # noqa: BLE001 — no posture: the hand stays open
        power = None
    if power is not None:
        for k, closing in ((3, 1.0), (4, 0.0)):
            tr['q'][k] = [float(v) for v in DH.posture(model, power, closing, np.asarray(tr['q'][k], float))]
    return tr


def build_g1() -> Project:
    lic = 'BSD-3-Clause (Menagerie unitree_g1/LICENSE, © 2016-2023 Unitree Robotics)'
    simp = G1_SIMPLIFICATION
    note = (S._source_note(['unitree_g1']) + ' g1_with_hands.xml with the legs and waist held at its "stand" '
            'keyframe (scripts/bundle_samples.py deletes the free joint, the 12 leg joints and the 3 waist joints, '
            'all 0 at "stand"). Its meshes are simplified to fit the package: each within '
            f'{simp["tolerance_mm"]:g} mm (two-sided Hausdorff distance) of Menagerie\'s, measured largest '
            f'{simp["largest_mm"]} mm.')
    robot, sources = _import(S.robot_file(G1_FILE), 'Unitree G1 upper body', lic, note)
    robot.gripper_frame = 'right_wrist_yaw_link'
    robot.fingertip_frames = list(G1_RIGHT_TIPS)
    S._tag_gripper(robot, {j['name'] for j in robot.joints if str(j.get('name', '')).startswith(
        ('right_hand_', 'left_hand_')) and j.get('type') != 'fixed'})
    p = Project('Unitree G1 upper body sample')
    p.robots, p.sources = [robot], dict(sources)
    model = get_model(p, robot.uid)
    # a chest-high table where the head camera's view (47.6° down) and the hands' reach meet: its top 0.08 m above
    # the wrists at "stand" (the arms hang beside the hips there), its near edge 0.20 m in front of the pelvis
    # (sample choices; the arms reach about 0.45 m from the shoulders)
    T = model.fk(model.home)
    wrists = np.array([T['right_wrist_yaw_link'][:3, 3], T['left_wrist_yaw_link'][:3, 3]])
    z_table = float(np.round(wrists[:, 2].max() + 0.08, 2))
    p.workspace.surfaces.append(_surface('Table (sample)', z_table, (0.40, 0.90), (0.40, 0.0)))
    table_vol = _vol('Tabletop in front of the robot (sample)', 'tabletop', (0.32, 0.0, z_table + 0.08),
                     (0.20, 0.50, 0.16))
    hand_vol = _vol('Handover zone between the hands (sample)', 'handover', (0.30, 0.0, z_table + 0.22),
                    (0.15, 0.20, 0.12))
    # where the right hand can hold an angled grasp (its reach with a set orientation is short: over the whole
    # tabletop volume about 80 % of angled grasps are out of reach)
    reach_vol = _vol('Right hand\'s reach (sample)', 'tabletop', (0.25, -0.12, z_table + 0.08), (0.15, 0.20, 0.10))
    p.workspace.volumes += [table_vol, hand_vol, reach_vol]
    # the head camera: Unitree's URDF frame d435_link on torso_link
    head = Sensor(name='Head D435 (Unitree URDF d435_link)', kind='depth', slot=0, model_id='intel-realsense-d435',
                  mount=Mount('torso_link', robot.uid, Pose(np.array(G1_D435_XYZ, float), quat_from_rpy(*G1_D435_RPY))))
    p.variant.name = 'Head D435 (Unitree)'
    p.variant.sensors = [head]
    p.variant.note = ('The head camera where Unitree\'s g1_29dof_rev_1_0.urdf puts its d435_link (unitree_ros commit '
                      '7c40519); Unitree\'s G1 page states only "Depth Camera+3D LiDAR".')
    # XR hand tracking, as Unitree's xr_teleoperate drives the G1 (its README names Apple Vision Pro): the library's
    # entry, whose tracking volume no source states, so the teleop study says so instead of guessing one
    _library_devices(p, 'apple-vision-pro-m5', [(robot, 'Apple Vision Pro hand tracking (xr_teleoperate)')],
                     f'{SOURCES["xr_teleoperate"]["url"]} (read 2026-11-07): "Wear your XR headset (e.g. Apple '
                     'Vision Pro, PICO4, etc.)"')
    # both arms reach in one state vector: each palm to a point over the table, the right hand closes, back home
    rl, rtcp = _tcp(p, robot.uid)
    tr = _trajectory('g1', robot.uid, lambda: _g1_reach(model, robot.uid, rl, rtcp, table_vol))
    p.tasks.trajectories.append(tr)
    # the multi-finger study poses the hand at its postures with the arms at the trajectory's reach (both hands over
    # the table, open), not at "stand", where the hands hang beside the hips out of the head camera's view
    p.analysis.studies['dex_occlusion'] = {'robot_uid': robot.uid, 'q': {robot.uid: list(tr['q'][2])}}
    _grasps(p, 'g1', robot, ('angled',), 100, 'Right hand, angled grasps in its reach (sample)', reach_vol)
    p.policies.append(Policy(name='GR00T N1.5 (sample)', library_id='gr00t-n1.5-3b'))
    p.notes = ('Sample (spec §14.5): the Unitree G1 with hands from MuJoCo Menagerie unitree_g1 (BSD-3-Clause), '
               'upper body: the legs and waist held at the file\'s "stand" keyframe, q = both arms and both '
               'three-finger hands (28 joints; Menagerie does not name the hand, so it is not called Dex3). The '
               'meshes are simplified to fit the package, each within '
               f'{simp["tolerance_mm"]:g} mm of Menagerie\'s (measured largest {simp["largest_mm"]} mm; '
               f'{simp["megabytes_before"]:g} MB of meshes became {simp["megabytes_after"]} MB): import the full model '
               'with Robot ▸ Import from MuJoCo Menagerie (unitree_g1) when that matters. The head D435 is where '
               'Unitree\'s URDF puts its d435_link. The gripper frame and fingertips are the right hand\'s; the '
               'table, task volumes, trajectory, grasp set, the Apple Vision Pro device and GR00T are sample choices.')
    p.add_log('sample', 'Opened the Unitree G1 upper body sample')
    return p


def g1_files_present() -> bool:
    return os.path.isfile(S.robot_file(G1_FILE))
