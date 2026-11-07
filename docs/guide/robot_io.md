# Robots in and out: URDF for MuJoCo, USD robots, OpenVLA and Isaac exports

The model, the equations and the validation are in [docs/models/robot_io.md](../models/robot_io.md).

## Export a URDF that MuJoCo loads

1. Open the SO-100 sample (File ▸ Open sample ▸ SO-100).
2. File ▸ Export…, tick **ROS 2: URDF, static transforms and tf2 tree**, choose a folder, Export.
3. `ros2/rig.urdf` now has an `<inertial>` in every link. A comment in each says where its figures come from: the
   robot's file, MuJoCo's computation from the file's geoms (the SO-100's `Base` has no inertial in its file), or a
   minimal placeholder (10⁻⁶ kg) for the sensor and optical frames, labelled as such.
4. Load it in MuJoCo as written:

```python
import mujoco
m = mujoco.MjModel.from_xml_path('out/ros2/rig.urdf')    # 12 bodies, 1.1714 kg in all on the SO-100 sample
m.body('wrist_camera_sample_placement_rgb_optical_frame')   # every sensor frame is a body
```

The file's `<mujoco><compiler …/></mujoco>` element tells MuJoCo where the meshes are and to keep the sensor frames;
ROS tools ignore it.

## Import a USD robot

Needs the optional usd-core package (`pip install "dexvantage[usd]"`); without it the wizard says so.

1. Robot ▸ Import robot…, drop or browse to a `.usd`, `.usda`, `.usdc` or `.usdz` file with a UsdPhysics
   articulation (rigid bodies and revolute, prismatic or fixed joints, as Isaac Sim or the mujoco-usd-converter write
   them).
2. The File step reads it and shows "USD, converted to MJCF"; the status line lists what the conversion left out.
3. Meshes: the layers the stage uses, and any it names that are missing (not required: what they hold is left out).
4. Continue as for a URDF: units, joints (tick the gripper joints), gripper frame and pads, Import (one undo step).

What to expect, on the bundled third-party UR5e (`dexvantage/resources/usd_robots/newton_ur5e/usd_structured/
ur5e.usda`, BSD-3-Clause): 6 hinges and a fixed joint to the world, 9 collision primitives (8 capsules, 1 cylinder),
the stated masses, and two notes: its mesh library `GeometryLibrary.usdc` is not bundled (so the 20 visual meshes are
left out), and drives and joint friction are not imported. Its arm poses like DexVantage's UR5e sample within
2.6 × 10⁻⁸ m (check `robot_io.usd_third_party`).

From Python, headless:

```python
import dexvantage as dv
rig = dv.sample('so100')
dv.robot_io.write_usd(rig, 'so100.usda')            # 7 bodies, 6 joints, 31 geoms as a UsdPhysics articulation
cell = dv.new('USD import')
arm = dv.robot_io.add_robot(cell, 'so100.usda', 'SO-100 from USD')
arm.source.note                                      # 'Converted from the USD file so100.usda (SHA-256 …) …'
dv.robot_io.convert_usd('so100.usda')['notes']       # what the conversion left out, in sentences
```

## OpenVLA encodings

The LeRobot export's `openvla/oxe_dataset_config.py` now picks the state and action encodings from the rig:

```python
dv.robot_io.openvla_encoding(dv.sample('aloha'))
# {'target': 'openvla', 'state_encoding': 'JOINT_BIMANUAL', 'action_encoding': 'JOINT_POS_BIMANUAL', 'fits': False, …}
dv.robot_io.openvla_encoding(dv.sample('so100'), 'eef')
# {'state_encoding': 'POS_EULER', 'action_encoding': 'EEF_POS', 'fits': True, …}
```

ALOHA 2 has two arms of 6 joints and a gripper, OpenVLA's bimanual layout, but OpenVLA's own loader takes only
end-effector actions (`materialize.py` L32-33), so `fits` is False and the notes say to use OpenVLA-OFT, whose
loader reads `JOINT_POS_BIMANUAL` as absolute joint positions: add an OpenVLA-OFT policy (Policies ▸ Add) and the
export targets OpenVLA-OFT, with its `left_wrist` and `right_wrist` image keys. To record end-effector deltas, export
with the LeRobot option `{"action_space": "eef"}` (the Export dialog's Options field). When no encoding fits, the
file says "NO ENCODING FITS" and why.

## Isaac: an off-centre principal point

A calibrated camera's principal point is rarely at the image centre. A unit calibration comes from the Hand-eye
calibration page (Apply stores the solved intrinsics on the sensor) or from Python:

```python
rig = dv.sample('ur5e')
wrist = rig.sensors[0]                               # Wrist D405: a flat calibration applies to its first imager
wrist.calibration = {'K': [[610.0, 0, 331.75], [0, 610.0, 228.25], [0, 0, 1]], 'D': [0.0] * 5,
                     'width': 640, 'height': 480, 'model': 'brown'}
rig.export('out', ['isaac'])['formats']['isaac']['notes']
```

With such a project open in the window, File ▸ Export… says under **Isaac Lab and Isaac Sim cameras**, per camera and
before anything is exported,

> Wrist D405 · IR left: principal point +12.2, -11.2 px from the centre. Isaac Lab v3.0.0 gets it through the OpenCV
> lens attributes …; Isaac Lab v2.3.2 … its render is centred: 16.6 px off this camera's principal point.

The v3.0.0 config and `cameras.usda` carry it as `omni:lensdistortion:opencvPinhole:cx, cy`, which Isaac Lab's
renderer projects through; Omniverse ignores the USD aperture offsets. For v2.3.2 there is no way to show it: use
v3.0.0, or crop and re-centre the rendered images.
