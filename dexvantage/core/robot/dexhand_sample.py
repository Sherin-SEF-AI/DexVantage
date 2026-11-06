"""The 'dexhand' sample (spec §6 dexterous hands, §7.14, §14): the UR5e with the right Shadow Hand on its flange and
three candidate camera mounts for the multi-finger occlusion study.

    build() -> Project            (through core.robot.samples.sample_project('dexhand'))

The robot is resources/robots/ur5e_shadow_hand.xml, written by scripts/bundle_dexhand.py with
mujoco.MjSpec.attach(hand, site=<the UR5e's "attachment_site">, prefix="hand_") from the MuJoCo Menagerie folders
universal_robots_ur5e (BSD-3-Clause) and shadow_hand (Apache-2.0); THIRD_PARTY.md and samples.BUNDLED record both.
Joint limits, the link geometry and the home pose are the files' own (home: the UR5e's keyframe, the hand open).
At home the hand points straight down from the flange with its palm facing the world's +x axis.

What the sample adds, each labelled as a sample choice:
- the fingertip frames (the five distal links), the gripper frame (the palm) and the hand's joints tagged as gripper
  joints, as the Robot page's taggers would;
- the contact pads: the palmar side of each distal link's collision mesh (core/dexhand.py pad_faces, its rule and
  its 45° default), stored as Robot.contact_pads faces;
- the grasp postures: power, precision and tripod sweeps between the hand folder's own keyframes
  (core/dexhand.py shadow_synergies), in the study's settings;
- three cameras at sample placements (not published figures), each with a library device whose figures come from
  its maker: "Palm micro camera" (OmniVision OVM6948, a palm-embeddable 0.65 mm module) set into the palm's palmar
  face and looking out of it, tilted 25° towards the fingertips; "Dorsal wrist camera" (e-con Systems See3CAM_24CUG)
  on the back of the hand's forearm 65 mm off its axis, 35 mm before the wrist joint, angled to look past the
  knuckles at the space in front of the fingertips; "Head camera" (Intel RealSense D435, its RGB imager) static,
  0.55 m in front of the palm and 0.4 m above it, looking at the fingers, roughly where a humanoid's head camera
  would see its own hand;
- a tabletop task volume under the fingertips and the table.

Known limits: the hand's mass (the forearm alone is 3 kg in the file) is not checked against the UR5e's payload;
the cameras' housings are not occluders (core/scenegeom.py); the Shadow Hand's coupled middle and distal joints are
independent entries of q (core/dexhand.py).
"""
from __future__ import annotations

import numpy as np

from ..project import Mount, Project, Sensor, Volume
from . import samples as S

STUDY_ID = 'dex_occlusion'
PREFIX = 'hand_'
TIPS = [f'{PREFIX}rh_{f}distal' for f in ('ff', 'mf', 'rf', 'lf', 'th')]
PALM = f'{PREFIX}rh_palm'
FOREARM = f'{PREFIX}rh_forearm'
NOTE = ('The hand is the right Shadow Hand E3M5 from MuJoCo Menagerie shadow_hand (Apache-2.0), on the UR5e '
        '(universal_robots_ur5e, BSD-3-Clause) with mujoco.MjSpec.attach (scripts/bundle_dexhand.py). The forearm '
        'rotation of the hand file is set to identity so the hand continues the flange axis.')


def build() -> Project:
    from .. import dexhand as DH
    robot, sources, model = S._import(
        S.robot_file(S.COMBINED_DEXHAND), 'UR5e + Shadow Hand',
        'BSD-3-Clause (Menagerie universal_robots_ur5e/LICENSE) and Apache-2.0 (Menagerie shadow_hand/LICENSE)',
        S._source_note(['universal_robots_ur5e', 'shadow_hand']) + ' ' + NOTE)
    robot.gripper_frame = PALM
    robot.fingertip_frames = list(TIPS)
    hand = DH.structure_of(model, robot.uid, TIPS)
    S._tag_gripper(robot, set(hand.hand_joints(model)))
    robot.contact_pads = DH.contact_pads_for(model, TIPS)
    p = Project('UR5e + Shadow Hand sample')
    p.robots, p.sources = [robot], dict(sources)
    T = model.fk(model.home)
    tips = np.array([T[t][:3, 3] for t in TIPS])
    lo = tips.min(axis=0)
    centre = np.round([tips[:, 0].mean() + 0.03, tips[:, 1].mean(), 0.06], 3) + 0.0
    p.workspace.volumes.append(Volume(name='Tabletop (sample)', kind='tabletop', center=tuple(float(v) for v in centre),
                                      size=(0.30, 0.30, 0.12)))
    p.workspace.surfaces.append({'uid': 'table', 'name': 'Table (sample)', 'z': 0.0, 'size': [1.2, 1.2],
                                 'material': 'matte'})
    # palm camera: in the palm's palmar face (the palm frame's -y side; its collision box face is at y = -0.012 m),
    # on the hand's centre line 50 mm up from the wrist, looking out of the palm tilted 25° towards the fingers
    tilt = np.radians(25.0)
    palm_pos = np.array([0.0, -0.013, 0.05])
    palm_cam = S.look_at(palm_pos, palm_pos + np.array([0.0, -np.cos(tilt), np.sin(tilt)]), up=(0.0, 0.0, 1.0))
    # dorsal wrist camera: on the forearm's back (its -x side, the palm faces its +x), 65 mm off the axis and 35 mm
    # before the wrist joint (z = 0.213 m in the forearm frame), looking 0.20 m along the hand and 30 mm palmar
    dors_pos = np.array([-0.065, 0.0, 0.178])
    dors_cam = S.look_at(dors_pos, [0.03, 0.0, 0.42], up=(-1.0, 0.0, 0.0))
    # head camera: static, in front of the palm and above it, looking at the middle of the fingers
    look = tips.mean(axis=0) * 0.5 + T[PALM][:3, 3] * 0.5
    head_pos = look + np.array([0.55, 0.0, 0.40])
    sensors = [
        Sensor(name='Palm micro camera (sample placement)', kind='camera', slot=0, model_id='omnivision-ovm6948',
               mount=Mount(PALM, robot.uid, palm_cam)),
        Sensor(name='Dorsal wrist camera (sample placement)', kind='camera', slot=1, model_id='e-con-see3cam-24cug',
               mount=Mount(FOREARM, robot.uid, dors_cam)),
        Sensor(name='Head camera (sample placement)', kind='depth', slot=2, model_id='intel-realsense-d435',
               mount=Mount('cage', '', S.look_at(head_pos, look))),
    ]
    p.variant.name = 'Palm + dorsal wrist + head (sample)'
    p.variant.sensors = sensors
    p.variant.note = S.SAMPLE_NOTE
    syn = DH.shadow_synergies(S.robot_file('shadow_hand'), PREFIX)
    p.analysis.studies[STUDY_ID] = {'robot_uid': robot.uid, 'synergies': [s.to_dict() for s in syn]}
    p.notes = ('Sample (spec §7.14): the UR5e with the right Shadow Hand from MuJoCo Menagerie (BSD-3-Clause and '
               f'Apache-2.0). {S.SAMPLE_NOTE} The three cameras are the candidate mounts the multi-finger occlusion '
               'study compares: a palm micro camera (OmniVision OVM6948), an angled dorsal wrist camera (e-con '
               'Systems See3CAM_24CUG) and a head camera (Intel RealSense D435); their figures come from the '
               'library entries. The grasp postures are sweeps between the hand file\'s own keyframes (open hand to '
               'grasp sphere, two finger pinch and three finger pinch), sample sweeps rather than measured '
               f'synergies. The lowest fingertip is {lo[2] * 1000:.0f} mm above the table at home.')
    p.add_log('sample', 'Opened the UR5e + Shadow Hand sample')
    return p

