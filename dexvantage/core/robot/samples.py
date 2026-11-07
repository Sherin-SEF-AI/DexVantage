"""The sample projects (spec §14) and the robot models bundled for them.

    sample_project('so100')    LeRobot SO-100 arm (MuJoCo Menagerie trs_so_arm100): a wrist camera on the gripper and
                               a third-person camera on a tripod, a tabletop task volume
    sample_project('ur5e')     UR5e with a Robotiq 2F-85 on its flange: a wrist depth camera ("Wrist D405") and an
                               overhead one ("Overhead D435"), a tabletop task volume and a scripted pick trajectory
                               (a sample motion for the motion blur and time sync studies)
    sample_project('dexhand')  UR5e with the right Shadow Hand on its flange: a palm micro camera, an angled dorsal
                               wrist camera and a head camera (core/robot/dexhand_sample.py), for the multi-finger
                               occlusion study (spec §7.14)
    sample_project('aloha')    ALOHA 2: the two ViperX 300 arms of MuJoCo Menagerie aloha as two robots at aloha.xml's
                               base poses, its four cameras (four RealSense D405), leader arms, a handover
                               (core/robot/bimanual_samples.py)
    sample_project('panda-bimanual')  two Franka Panda arms (Menagerie franka_emika_panda) in one cell, a wrist D405
                               on each hand and a front D435, GELLO leaders, a handover
    sample_project('g1')       Unitree G1 with hands (Menagerie unitree_g1), upper body: legs and waist held at its
                               "stand" keyframe, both arms and hands in q, the head D435 where Unitree's URDF puts it;
                               its meshes simplified to fit the package (within 0.5 mm, measured)
    list_samples()             every sample with whether this build has it

The robots are read with core.robot.import_robot from dexvantage/resources/robots/, which holds only the MuJoCo
Menagerie models whose own LICENSE allows redistribution; each folder keeps its LICENSE file and THIRD_PARTY.md lists
the source, licence, commit and the date it was checked (BUNDLED below says the same). Joint limits, link geometry
and the home pose are the model files' own; nothing is typed in again. What the files are, by their own READMEs
(Menagerie folders at commit 98a70c5, read 2026-11-06): both are "simplified" MJCF descriptions derived from public
URDFs. The SO-ARM100's joint limits are ones Menagerie "added" as "approximate joint limits" (its derivation step 7),
not The Robot Studio's figures; the robot's source note says so (SO100_LIMITS_NOTE). The UR5e's link offsets are
rounded to the millimetre (0.163 m against Universal Robots' DH d1 = 0.1625 m, for example): after the best rigid
alignment its flange is up to 1.3 mm (0.9 mm RMS, 500 random poses) from the flange of Universal Robots' DH table.

The UR5e and the gripper are two Menagerie models. ur5e_robotiq_2f85.xml joins them the way MuJoCo itself does:
scripts/bundle_menagerie.py calls mujoco.MjSpec.attach(gripper, site=<the UR5e's "attachment_site">,
prefix="gripper_") and writes the result with MjSpec.to_xml(), so the gripper's frame is the flange site of the arm
file and the gripper's names carry the prefix (its "base" body would clash with the arm's).

Camera placements are sample placements chosen for these samples so the rig has something to analyse; they are not
published figures and the sensor names or the project notes say so. The UR5e sample's depth cameras name their
library entries (model_id 'intel-realsense-d405' and 'intel-realsense-d435', core/library, figures from Intel's D400
series datasheet); the sensor body frame is the device's (origin at the left imager's optical centre). The SO-100
sample's cameras have no model_id: no library camera entry is the camera a SO-100 build is stated to use.

Known limits: the 2F-85's linkage joints are independent entries of q (core/robot/model.py treats every hinge as a
joint of its own), so a q that moves one finger joint alone poses the linkage in a way the real gripper cannot.
"""
from __future__ import annotations

import json
import os

import numpy as np

from ..frames import Pose, quat_from_matrix
from ..project import Mount, Policy, Project, ProjectFormatError, Sensor, Volume, new_uid

ROBOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          'resources', 'robots')
MENAGERIE = 'https://github.com/google-deepmind/mujoco_menagerie'
MENAGERIE_COMMIT = '98a70c569c4c1b6da467a20012f766ee1f8dc521'     # the checkout the bundled files were copied from
CHECKED = '2026-11-06'

# folder under resources/robots -> what THIRD_PARTY.md records (each folder holds its own LICENSE file)
BUNDLED = {
    'trs_so_arm100': {'title': 'SO-ARM100 (The Robot Studio)', 'main': 'so_arm100.xml', 'license': 'Apache-2.0',
                      'folder_commit': '71f066ad0be9cd271f7ed58c030243ef157af9f4'},
    'universal_robots_ur5e': {'title': 'Universal Robots UR5e', 'main': 'ur5e.xml', 'license': 'BSD-3-Clause',
                              'folder_commit': 'fb36886b3023db08c198264e1e9956ab202c3e3b'},
    'robotiq_2f85': {'title': 'Robotiq 2F-85', 'main': '2f85.xml', 'license': 'BSD-2-Clause',
                     'folder_commit': '71f066ad0be9cd271f7ed58c030243ef157af9f4'},
    'shadow_hand': {'title': 'Shadow Hand E3M5 (right)', 'main': 'right_hand.xml', 'license': 'Apache-2.0',
                    'folder_commit': '71f066ad0be9cd271f7ed58c030243ef157af9f4'},
    # the ALOHA 2, bimanual Panda and G1 samples (scripts/bundle_samples.py; LICENSE files read 2026-11-07)
    'aloha': {'title': 'ALOHA 2 (Trossen ViperX 300 arms)', 'main': 'aloha.xml', 'license': 'BSD-3-Clause',
              'folder_commit': 'fb36886b3023db08c198264e1e9956ab202c3e3b', 'checked': '2026-11-07'},
    'franka_emika_panda': {'title': 'Franka Emika Panda with the Franka Hand', 'main': 'panda.xml',
                           'license': 'Apache-2.0', 'folder_commit': 'fb36886b3023db08c198264e1e9956ab202c3e3b',
                           'checked': '2026-11-07'},
    'unitree_g1': {'title': 'Unitree G1 with hands (meshes simplified)', 'main': 'g1_with_hands.xml',
                   'license': 'BSD-3-Clause', 'folder_commit': '98a70c569c4c1b6da467a20012f766ee1f8dc521',
                   'checked': '2026-11-07'},
}
COMBINED_UR5E = 'ur5e_robotiq_2f85.xml'
COMBINED_DEXHAND = 'ur5e_shadow_hand.xml'          # scripts/bundle_dexhand.py
GENERATED = ('aloha_left.xml', 'aloha_right.xml', 'g1_upper_body.xml')   # scripts/bundle_samples.py

SO100_LIMITS_NOTE = ('Its joint limits are the ones Menagerie added to this model as "approximate joint limits" '
                     '(Menagerie trs_so_arm100/README.md, "URDF → MJCF derivation steps", step 7), not figures stated '
                     'by The Robot Studio.')
SAMPLE_NOTE = ('Camera placements in this sample are sample placements chosen for it, not published figures: move '
               'them to where your rig has its cameras.')


class SampleError(ProjectFormatError):
    """A sample this build cannot open; the message says why in plain words."""


# name -> (title, phase it arrives in or None when this build has it)
SAMPLES = {
    'so100': ('LeRobot SO-100 arm with a wrist and a third-person camera', None),
    'ur5e': ('UR5e and Robotiq 2F-85 with a wrist D405 and an overhead D435', None),
    'dexhand': ('UR5e and Shadow Hand with a palm, a dorsal wrist and a head camera', None),
    'aloha': ('ALOHA 2 bimanual with its four cameras', None),
    'panda-bimanual': ('Two Franka Panda arms in one cell with wrist and front cameras', None),
    'g1': ('Unitree G1 humanoid upper body with its head camera', None),
}


def list_samples() -> list:
    """[{'name', 'title', 'available', 'phase'}] for every sample of the specification."""
    return [{'name': k, 'title': t, 'available': ph is None, 'phase': ph} for k, (t, ph) in SAMPLES.items()]


def robot_file(*parts: str) -> str:
    return os.path.join(ROBOTS_DIR, *parts)


# ====================================================================== placement helpers
def look_at(position, target, up=(0.0, 0.0, 1.0)) -> Pose:
    """The pose of a sensor body frame (x forward along the optical axis, y left, z up; REP-103) at position, looking
    at target, its z axis as close to up as the view allows. Both points in the parent frame."""
    p, t = np.asarray(position, float), np.asarray(target, float)
    f = t - p
    n = np.linalg.norm(f)
    if n < 1e-9:
        raise ValueError('a sensor cannot look at its own position')
    f = f / n
    u = np.asarray(up, float)
    if abs(float(np.dot(f, u / np.linalg.norm(u)))) > 0.999:   # looking along up: take another hint
        u = np.array([1.0, 0.0, 0.0]) if abs(f[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    left = np.cross(u, f)
    left /= np.linalg.norm(left)
    R = np.column_stack([f, left, np.cross(f, left)])
    return Pose(p, quat_from_matrix(R))


def _site_in_body(model, site: str, body: str) -> np.ndarray:
    """The position of an MJCF site in the frame of the link `body`, from the compiled model at its home pose."""
    m = model.m
    sid = next(i for i in range(m.nsite) if m.site(i).name == site)
    T_site_body = model.link_pose(model.home, model.body_link[int(m.site_bodyid[sid])])
    p_world = T_site_body[:3, :3] @ m.site_pos[sid] + T_site_body[:3, 3]
    T = model.link_pose(model.home, body)
    return np.linalg.inv(T) @ np.r_[p_world, 1.0]


def _import(path: str, name: str, license_text: str, note: str):
    from . import import_robot
    from .model import RobotModel
    robot, sources = import_robot(path, name)
    robot.primary = True
    robot.source.license = license_text
    robot.source.note = ' '.join(x for x in (robot.source.note, note) if x)
    return robot, sources, RobotModel(robot, sources)


def _source_note(folders) -> str:
    return (f'MuJoCo Menagerie ({MENAGERIE}) at commit {MENAGERIE_COMMIT[:12]}, folder(s) {", ".join(folders)}; '
            f'licence read from each folder\'s LICENSE on {CHECKED} (THIRD_PARTY.md).')


def _tag_gripper(robot, names: set):
    """Mark the gripper's joints as the import wizard's joint tagger does ('gripper': True on the joint record)."""
    for j in robot.joints:
        if j['name'] in names:
            j['gripper'] = True


_GRASPS: dict = {}          # sample name -> (grasp set, robot uid, volume uid) of the first build in this process


def _add_sample_grasps(p: Project, key: str, kinds: tuple, n: int, name: str):
    """A sampled grasp set over the sample's task volume (core/grasps.sample, seed 0), so the grasp self-occlusion
    and VLA token studies run on the sample as it opens. The poses depend only on the bundled robot and the volume,
    so they are sampled once per process and given the new project's uids after that."""
    from .. import grasps as GR
    robot, vol = p.robots[0], p.workspace.volumes[0]
    hit = _GRASPS.get(key)
    if hit is None:
        gs = GR.sample(p, robot.uid, kinds=kinds, n=n, rng_seed=0, volume_uid=vol.uid, name=name)
        _GRASPS[key] = (json.dumps(gs), robot.uid, vol.uid)
    else:
        text, ru, vu = hit
        gs = json.loads(text.replace(ru, robot.uid).replace(vu, vol.uid))
        gs['uid'] = new_uid()
    GR.add_set(p, gs)


# ====================================================================== the samples
def _so100() -> Project:
    robot, sources, model = _import(
        robot_file('trs_so_arm100', 'so_arm100.xml'), 'SO-ARM100', 'Apache-2.0 (Menagerie trs_so_arm100/LICENSE)',
        _source_note(['trs_so_arm100']) + ' ' + SO100_LIMITS_NOTE)
    # the model's own bodies: the wrist-roll body carries the fixed jaw, the moving jaw is its child
    robot.gripper_frame = 'Fixed_Jaw'
    robot.fingertip_frames = ['Fixed_Jaw', 'Moving_Jaw']
    _tag_gripper(robot, {'Jaw'})
    p = Project('SO-100 sample')
    p.robots, p.sources = [robot], dict(sources)
    # The arm reaches along the base frame's -y axis (the Menagerie file's own orientation). At the file's home
    # keyframe the jaw tips are about 0.07 m above the base plane, 0.24 m out.
    tips = model.link_pose(model.home, 'Fixed_Jaw') @ np.array([0.0, -0.10, 0.0, 1.0])
    centre = np.round([tips[0], tips[1], 0.06], 3) + 0.0      # to the millimetre: a sample choice
    p.workspace.volumes.append(Volume(name='Tabletop (sample)', kind='tabletop', center=tuple(float(v) for v in centre),
                                      size=(0.30, 0.20, 0.12)))
    p.workspace.surfaces.append({'uid': 'table', 'name': 'Table (sample)', 'z': 0.0, 'size': [0.8, 0.8],
                                 'material': 'matte'})
    # wrist camera: on a bracket 0.06 m behind the jaws' plane on the wrist side (the jaw meshes span z = ±0.024 m
    # in the link, the jaws reach to y = -0.106 m), looking down past the jaw tips; a sample placement. It was on the
    # back of the fixed jaw before, where the jaw itself hid 39 % of the task volume from it (11 % here, coverage
    # study at 10 mm, home pose).
    wrist = Sensor(name='Wrist camera (sample placement)', kind='camera', slot=0,
                   mount=Mount('Fixed_Jaw', robot.uid, look_at([-0.01, 0.0, -0.06], [-0.01, -0.14, 0.0])))
    third = Sensor(name='Third-person camera (sample placement)', kind='camera', slot=1,
                   mount=Mount('tripod', '', look_at(centre + [0.40, -0.25, 0.30], centre)))
    p.variant.name = 'Wrist + third-person (sample)'
    p.variant.sensors = [wrist, third]
    p.variant.note = SAMPLE_NOTE
    # a grasp set and a policy, so the grasp and VLA token studies run as the sample opens (sample choices: SmolVLA
    # because its paper evaluates on SO-100 rigs ("For SO100, we use top and wrist cameras", quoted in the library
    # entry); top-down pinches because the five-joint arm cannot hold most side grasps)
    _add_sample_grasps(p, 'so100', ('top_down',), 100, 'Top-down pinches over the tabletop (sample)')
    p.policies.append(Policy(name='SmolVLA (sample)', library_id='smolvla-base'))
    # the kit's leader arm (core/studies/teleop.py): an SO-100 of the same joints through the wrist, mapped joint to
    # joint (LeRobot's SO-100 guide, read 2026-11-07); its comfortable joint share is the study's labelled default
    p.tasks.teleop_devices.append({
        'uid': new_uid(), 'name': 'SO-100 leader arm (sample)', 'kind': 'leader_arm', 'follower_uid': robot.uid,
        'leader': {'robot_uid': robot.uid},
        'mapping': {'kind': 'joint', 'scale': 1.0, 'offset': Pose().to_dict(), 'joint_map': {}},
        'volume': None,
        'source': {'leader': 'LeRobot SO-100 guide, https://huggingface.co/docs/lerobot/so100 (read 2026-11-07): the '
                             'leader arm repeats the follower\'s assembly steps 1-23 and both arms are calibrated to '
                             'the same joint positions, so the leader is the follower\'s own model, joint to joint'}})
    p.notes = (f'Sample (spec §14.2): the LeRobot SO-100 arm from MuJoCo Menagerie trs_so_arm100 (Apache-2.0). '
               f'{SAMPLE_NOTE} The cameras have no library device: no maker in the library states a camera for '
               'the SO-100, so both use the labelled default camera until you choose one. The SO-101 is in '
               'Menagerie too (robotstudio_so101): import it with Robot ▸ Import from MuJoCo Menagerie.')
    p.add_log('sample', 'Opened the SO-100 sample')
    return p


def _ur5e() -> Project:
    robot, sources, model = _import(
        robot_file(COMBINED_UR5E), 'UR5e + Robotiq 2F-85',
        'BSD-3-Clause (Menagerie universal_robots_ur5e/LICENSE) and BSD-2-Clause (Menagerie robotiq_2f85/LICENSE)',
        _source_note(['universal_robots_ur5e', 'robotiq_2f85']) + ' The gripper is attached at the UR5e file\'s '
        '"attachment_site" with mujoco.MjSpec.attach (scripts/bundle_menagerie.py).')
    robot.gripper_frame = 'gripper_base'            # the gripper body that holds the file's "pinch" site
    robot.fingertip_frames = ['gripper_right_pad', 'gripper_left_pad']
    _tag_gripper(robot, {j['name'] for j in robot.joints if j['name'].startswith('gripper_') and j['type'] != 'fixed'})
    p = Project('UR5e + Robotiq 2F-85 sample')
    p.robots, p.sources = [robot], dict(sources)
    # at the file's home keyframe the gripper points down about 0.49 m out along the base's +y axis
    pinch_gb = _site_in_body(model, 'gripper_pinch', 'gripper_base')[:3]
    pinch_w = (model.link_pose(model.home, 'gripper_base') @ np.r_[pinch_gb, 1.0])[:3]
    centre = np.round([pinch_w[0], pinch_w[1], 0.15], 3) + 0.0
    p.workspace.volumes.append(Volume(name='Tabletop (sample)', kind='tabletop', center=tuple(float(v) for v in centre),
                                      size=(0.50, 0.40, 0.30)))
    p.workspace.surfaces.append({'uid': 'table', 'name': 'Table (sample)', 'z': 0.0, 'size': [1.2, 1.2],
                                 'material': 'matte'})
    # wrist depth camera: on a flange bracket 0.07 m off the tool axis, 0.02 m along it from the gripper's base
    # (outside its 0.0375 m half-width), looking at the gripper's pinch point; posed in the wrist_3 link frame, so
    # it rides with the flange. A sample placement.
    T_w3_gb = np.linalg.inv(model.link_pose(model.home, 'wrist_3_link')) @ model.link_pose(model.home, 'gripper_base')
    cam_gb = look_at([0.07, 0.0, 0.02], pinch_gb, up=(1.0, 0.0, 0.0))
    wrist = Sensor(name='Wrist D405', kind='depth', slot=0, model_id='intel-realsense-d405',
                   mount=Mount('wrist_3_link', robot.uid, Pose.from_matrix(T_w3_gb @ cam_gb.matrix())))
    over = Sensor(name='Overhead D435', kind='depth', slot=1, model_id='intel-realsense-d435',
                  mount=Mount('cage', '', look_at(centre + [0.0, 0.0, 0.85], centre, up=(1.0, 0.0, 0.0))))
    p.variant.name = 'Wrist D405 + overhead D435 (sample)'
    p.variant.sensors = [wrist, over]
    p.variant.note = SAMPLE_NOTE
    # a grasp set and a policy, so the grasp and VLA token studies run as the sample opens (sample choices)
    _add_sample_grasps(p, 'ur5e', ('top_down', 'side', 'angled'), 200, 'Tabletop grasps (sample)')
    p.policies.append(Policy(name='π0 (sample)', library_id='pi0'))
    p.tasks.trajectories.append(_ur5e_pick(robot.uid, model))
    p.notes = ('Sample (spec §14.3): the UR5e and Robotiq 2F-85 from MuJoCo Menagerie (BSD-3-Clause and '
               f'BSD-2-Clause). {SAMPLE_NOTE} "Wrist D405" and "Overhead D435" name the intended Intel RealSense '
               'models; their lenses, depth range and minimum Z come from the library entries (Phase 2), read from '
               'Intel\'s D400 series datasheet; the entries\' notes say what the datasheet does not state.')
    p.add_log('sample', 'Opened the UR5e + Robotiq 2F-85 sample')
    return p


def _ur5e_pick(robot_uid: str, model) -> dict:
    """A short scripted pick for the UR5e sample (a sample choice, not a recorded motion), so the motion blur and
    time sync studies have a motion to analyse. The gripper keeps its home orientation; its TCP (between the pads)
    goes from home down to 0.17 m above the base plane over 1 s, the 2F-85 closes (spring links 0 to 0.5 rad over
    0.5 s), the arm lifts it to 0.30 m (1 s) and carries it 0.15 m along −y and 0.15 m along −x (1.5 s). Each waypoint
    is solved by IK from the one before (core/robot/ik.py, nearest to it; its self-collision rule is off because the
    2F-85's own linkage links already touch at home); the trajectory is linear in the joints between waypoints
    (core/robot/trajectory.py), so the TCP path bows a little between them."""
    from .ik import solve
    names = [str(n) for n in model.joint_names]      # (this module rebinds list: see the end)
    J = {n: i for i, n in enumerate(names)}
    link = 'gripper_base'
    tcp = np.eye(4)
    tcp[:3, 3] = _site_in_body(model, 'gripper_pinch', link)[:3]
    T0 = model.link_pose(model.home, link) @ tcp
    p0 = T0[:3, 3]
    goals = [None, (p0[0], p0[1], 0.17), None, None, (p0[0], p0[1], 0.30), (p0[0] - 0.15, p0[1] - 0.15, 0.30)]
    t = [0.0, 1.0, 1.5, 2.0, 3.0, 4.5]
    grip = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5]
    rows, q = [], np.array(model.home, float)
    for goal, g in zip(goals, grip):
        if goal is not None:
            T = T0.copy()
            T[:3, 3] = goal
            r = solve(model, link, T, q0=q, tcp=tcp, rule='nearest_q0', rng=0, collision=False)
            if not r.ok:
                raise SampleError(f'The UR5e sample\'s scripted pick could not be solved ({r.reason}).')
            q = np.array(r.q, float)
        q = q.copy()
        for side in ('right', 'left'):
            q[J[f'gripper_{side}_spring_link_joint']] = g
        rows.append([float(v) for v in model.clip(q)])
    return {'uid': new_uid(), 'name': 'Scripted pick (sample)', 'robot_uid': robot_uid, 'joints': names, 't': t,
            'q': rows}


def _dexhand() -> Project:
    from .dexhand_sample import build
    return build()


def _aloha() -> Project:
    from .bimanual_samples import build_aloha
    return build_aloha()


def _panda() -> Project:
    from .bimanual_samples import build_panda
    return build_panda()


def _g1() -> Project:
    from .bimanual_samples import build_g1
    return build_g1()


_BUILD = {'so100': _so100, 'ur5e': _ur5e, 'dexhand': _dexhand, 'aloha': _aloha, 'panda-bimanual': _panda, 'g1': _g1}


def sample_project(name: str) -> Project:
    """A new Project for the named sample (see SAMPLES); raises SampleError for a name this build cannot open."""
    key = str(name or '').strip().lower()
    if key not in SAMPLES:
        raise SampleError(f'There is no sample named "{name}". The samples are: {", ".join(SAMPLES)}.')
    title, phase = SAMPLES[key]
    if phase is not None:
        raise SampleError(f'The {title} sample arrives in Phase {phase}.')
    return _BUILD[key]()


list = list_samples                                  # samples.list(), as the specification names it
