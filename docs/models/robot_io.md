# Robot import and export: URDF inertials, USD import, OpenVLA encodings, Isaac principal point

Four fixes on the import and export side, each with a validation check of model `robot_io`
(`dexvantage/validation/checks_robot_io.py`) and a self-test line (`robot io`).

| Part | Code | Check |
|---|---|---|
| URDF with inertials, loadable by MuJoCo as written | `core/exporters/ros2.py` | `robot_io.urdf_mujoco` |
| USD robot import (spec §13) | `core/robot/usd_import.py`, `core/robot/importer.py`, the import wizard | `robot_io.usd_roundtrip`, `robot_io.usd_third_party` |
| A small UsdPhysics writer (for the round trip) | `core/robot/usd_writer.py` | `robot_io.usd_roundtrip` |
| OpenVLA encodings chosen from the rig | `core/exporters/lerobot.py` (`openvla_encoding`) | `robot_io.openvla_encoding` |
| Isaac: off-centre principal point | `core/exporters/isaac.py` (`principal_point_lines`), Export dialog | `robot_io.isaac_principal_point` |

Python: `dv.robot_io.convert_usd(path)`, `dv.robot_io.add_robot(rig, path)`, `dv.robot_io.write_usd(rig, path)`,
`dv.robot_io.openvla_encoding(rig, action_space=None)` (headless, no Qt).

## 1. URDF inertials

Each link that is a MuJoCo body gets `<inertial>` from the robot's compiled MuJoCo model (DexVantage's link frame is
the MuJoCo body frame):

- origin `xyz = body_ipos` (centre of mass in the link frame), `rpy = 0 0 0`;
- `mass = body_mass`;
- the full tensor about the centre of mass in the link axes, **I = R diag(body_inertia) Rᵀ**, R the rotation of
  `body_iquat` (MuJoCo's principal axes). MuJoCo's URDF compiler diagonalises it again when it reads the file.

A comment in each `<inertial>` says where the figures come from (`inertial_sources`, read from the robot's stored
file with `mujoco.MjSpec`, parsing only):

| Source | When |
|---|---|
| "from the robot's file" | the body has an explicit inertial (URDF `<inertial>`, MJCF `<inertial>`) |
| "computed by MuJoCo from the robot file's geoms" | an MJCF body without one, `inertiafromgeom` auto or true: the geoms' mass or density as the file states, else MuJoCo's default density 1000 kg/m³ (the SO-100's `Base`, the 2F-85's mount and pads) |
| "a minimal placeholder" | no mass in the robot's files: a URDF link without `<inertial>`, the joint-anchor links `<body>__<joint>`, the sensor and optical frames, links MuJoCo merged into a parent |

The placeholder is `MIN_MASS = 1e-6 kg` and `MIN_INERTIA = 1e-9 kg·m²` on the diagonal: a stated minimum, not a
figure of the robot. MuJoCo refuses a moving body without mass ("mass and inertia of moving bodies must be larger
than mjMINVAL"), which is why the file could not be loaded before.

The file also carries MuJoCo's URDF extension element (MuJoCo modeling guide, "URDF extensions", read 2026-11-07):

```xml
<mujoco>
  <compiler angle="radian" meshdir="meshes/" strippath="true" discardvisual="false" fusestatic="false"/>
</mujoco>
```

With the element present the compiler attributes are stated rather than left to the URDF defaults (measured: without
`strippath="true"` the mesh path became `meshes/meshes/…`). `fusestatic="false"` keeps every sensor and optical
frame as a body, so `mujoco.MjModel.from_xml_path('ros2/rig.urdf')` loads the file unchanged with the camera frames
findable by name. urdfdom ignores elements it does not know.

## 2. USD robot import

`core/robot/importer.import_robot(path)` converts a `.usd`, `.usda`, `.usdc` or `.usdz` file to MJCF
(`usd_import.convert_to_folder`, a temporary folder) and imports that MJCF as for any MuJoCo file. The stored files
are the converted ones: `robot.xml`, `meshes/*.stl` and `.dexvantage/usd_source.json` (the USD layers used with
their SHA-256, the units, the up axis, the counts and the conversion notes). `Robot.source.note` names the USD file
and repeats the notes. usd-core (`pip install "dexvantage[usd]"`, `usd-core==26.8`) is optional: without it the
importer and the wizard say what to install.

**What is read** (pxr; the OpenUSD UsdPhysics schemas, read 2026-11-07). The stage is opened with payloads loaded and
traversed with instance proxies (Isaac's instanceable meshes). Bodies: prims with `PhysicsRigidBodyAPI` under the
first `PhysicsArticulationRootAPI` prim (every rigid body when there is none). Joints: `UsdPhysicsJoint` prims that
link two of those bodies or one of them and the world (a `body0`/`body1` that is not a rigid body is the world frame
of that prim). pxr's composition warnings are collected (`UsdUtils.CoalescingDiagnosticDelegate`) and turned into one
sentence naming the layers that could not be opened.

**Frames.** Every prim's world map is pxr's `XformCache.GetLocalToWorldTransform`, scaled to metres by
`metersPerUnit` and turned to Z up for a Y-up stage by Rx(+90°). A body frame is that map's rotation and translation
(polar decomposition; the scale is kept for points, `localPos` and primitives). UsdPhysics states a joint by its
frame in each body, `F0 = X_body0 · (localPos0, localRot0)` and `F1 = X_body1 · (localPos1, localRot1)`, with

  X_body0 · F0 · M(q) = X_body1 · F1,   M(q) = Rot(axis, q) (revolute) or Trans(axis · q) (prismatic),

`axis` the token X, Y or Z of the joint frame. With the joint frames in each body, J0 = X_p⁻¹ F0 and J1 = X_c⁻¹ F1,
the child's MJCF body is placed at **B = J0 · J1⁻¹** and its hinge or slide at `pos = t(J1)`, `axis = R(J1) e_axis`.
MuJoCo then moves the child as X_p · B · Trans(pos) · Rot(axis, q) · Trans(−pos) = X_p · J0 · M(q) · J1⁻¹, the
UsdPhysics constraint exactly. The joints rule the kinematics; an authored body pose that disagrees with them at the
authored state is named in the notes. A joint written the other way round (its `body0` is the child in the tree) is
turned: frames swapped, axis negated, the same motion for the same q. Revolute limits are degrees (converted to
radians), prismatic limits stage units (converted to metres); a limit that is not finite or beyond ±10⁶ (the float
sentinels some writers use), or lower above upper, is no limit.

**Tree.** Breadth-first from the body jointed to the world (else the articulation root, said to be fixed at its
authored pose: DexVantage places a robot by its base pose). A joint that would close a loop, a disabled joint, one
excluded from the articulation and a generic (D6) or distance joint (imported as fixed) are named in the notes.
Spherical joints become ball joints (held at home, as for every ball joint in DexVantage).

**Geometry.** Every `UsdGeom.Mesh`, `Cube`, `Sphere`, `Capsule` and `Cylinder` below a body (and not below another
body): with `PhysicsCollisionAPI` a collision geom, otherwise (visible, purpose not guide or proxy) a visual geom
(`contype = conaffinity = 0`, group 1, which `core/robot/model.py` reads as visual-only). Meshes are triangulated
(fans; `leftHanded` flipped), moved into the body frame with their full transform and written as binary STL in metres.
Cube: box of half size `size/2 ×` scale; Sphere: sphere, or ellipsoid under a non-uniform scale; Capsule and
Cylinder: radius × the scale across the axis, half length `height/2 ×` the scale along it, turned so the axis token
is the geom's z.

**Mass.** A body whose `PhysicsMassAPI` states mass and diagonal inertia gets `<inertial pos quat mass
diaginertia>`: mass × `kilogramsPerUnit`, `centerOfMass` in metres in the body frame, `principalAxes`, inertia ×
`kilogramsPerUnit × metersPerUnit²`. The MJCF says `inertiafromgeom="false"`, so a body without stated mass gets none
(MuJoCo does not invent one from the geoms' volume); `core/robot/model.py` then compiles with its stated `boundmass`,
as for a kinematics-only URDF.

**The writer** (`core/robot/usd_writer.write_robot_usd`) writes a project robot as `/<robot>` (ArticulationRootAPI),
one Xform per MuJoCo body with RigidBodyAPI and MassAPI, its geoms under `visuals/` and `collisions/`, and
`/<robot>/joints/*`. With R_a a rotation taking +X to MuJoCo's axis a, anchor p and reference qpos0:
F1 = Trans(p) R_a and F0 = T0 Trans(p) Rot(a, −qpos0) R_a (slide: F0 = T0 Trans(p − a·qpos0) R_a), so the USD joint
position equals DexVantage's q. A body with several hinges or slides gets massless intermediate bodies. MuJoCo
3.17's own `mujoco.usd.exporter.USDExporter` was checked and not used: it writes render frames from
`mujoco.Renderer` (meshes and animated transforms, no UsdPhysics bodies, joints or limits) and needs OpenGL; MuJoCo
3.17's `MjSpec.from_file` has no USD decoder in the pip wheel ("could not find decoder for resource").

## 3. OpenVLA encodings

`openvla_encoding(project, action_space=None)` chooses the OXE `state_encoding` and `action_encoding` from the rig,
citing (read 2026-11-07):

- OpenVLA at `c8f03f4`: `prismatic/vla/datasets/rlds/oxe/configs.py` L33-40 (`StateEncoding`: `JOINT` = "Joint
  Angles (7, <PAD> if fewer) + Gripper Open/Close (1)", `JOINT_BIMANUAL` = "2 x [ Joint Angles (6) + Gripper
  Open/Close (1) ]"), L44-50 (`ActionEncoding`: `EEF_POS`, `JOINT_POS` "Joint Delta Position", `JOINT_POS_BIMANUAL`,
  `EEF_R6`), `materialize.py` L32-33 (`make_oxe_dataset_kwargs` refuses anything but `EEF_POS` and `EEF_R6`:
  "only EEF_POS & EEF_R6 actions supported!").
- OpenVLA-OFT at `e4287e9`: the same enums (configs.py L39, L48), its ALOHA entries L681-687 (`JOINT_BIMANUAL`,
  `JOINT_POS_BIMANUAL`, image keys `primary`, `secondary`, `left_wrist`, `right_wrist`), `materialize.py` L32-33
  (also accepts `JOINT_POS_BIMANUAL`) and L43-45 (`absolute_action_mask = [True] * 14`: absolute joint positions).

Inputs: the **target** is OpenVLA-OFT when a project policy is an OpenVLA-OFT library entry, else OpenVLA; the
**action space** is the export option `action_space` (`'joint'` or `'eef'`), else the OpenVLA policy's stored
`overrides['action_space']`, else `'joint'` (a default: the LeRobot export records joint positions); per robot, the
**arm joints** are its actuated joints minus those marked `gripper` (the import wizard's Joints step), and its gripper
is one open/close value.

| Rig | Action space | Encodings | Fits when |
|---|---|---|---|
| one arm | eef | `POS_EULER` / `EEF_POS` | always (both loaders accept `EEF_POS`) |
| two arms | eef | `POS_EULER` / `EEF_POS` (nearest) | never: no bimanual end-effector encoding |
| two arms | joint | `JOINT_BIMANUAL` / `JOINT_POS_BIMANUAL` | each arm 6 arm joints + a gripper, and the target is OpenVLA-OFT |
| one arm | joint | `JOINT` / `JOINT_POS` (nearest) | never: neither loader accepts `JOINT_POS`, and it is deltas |

When nothing fits, the nearest is written, the file's docstring says "NO ENCODING FITS" with the reasons, and the
export notes say so. With an OFT policy on a two-arm rig the image keys are OFT's (`left_wrist` is the first camera on
the arm named left, else the one whose base is further along +y). `oxe_dataset_config.json` carries the whole
choice under `encoding`. The dataset's own arrays are not converted (one gripper value per arm, end-effector deltas).

## 4. Isaac: an off-centre principal point

Isaac Lab's spawner warns that "Camera aperture offsets are not supported by Omniverse. These parameters will be
ignored" (v3.0.0 `sim/spawners/sensors/sensors.py` L162-165; v2.3.2 L108-111). At v3.0.0 the OpenCV lens model is
authored as `omni:lensdistortion:<model>:fx, fy, cx, cy` (sensors.py L52-58, L68-108, L190-192: "the RTX/OVRTX
renderer honors these attributes natively"), and `Camera._initialize_intrinsics` uses them (camera.py L828-857,
L879-904: "these intrinsics may be non-square (fx != fy) or off-center"; without them `cx = width * 0.5`). So any
imager whose principal point is not at ((W − 1)/2, (H − 1)/2) gets an `opencvPinhole` model (zero distortion for a
plain pinhole) carrying its cx, cy; the aperture offsets stay in the USD for other readers. v2.3.2 has no OpenCV
fields (sensors_cfg.py) and its camera.py L589-599 sets `c_x = width * 0.5`: its render is centred. Each off-centre
camera is then named with its offset:

- in the export notes and `isaac_cameras.json` (`principal_point`: `offset_px`, `off_centre`, what v3 and v2 do);
- in its comment in `isaaclab_cameras.py` and `isaaclab_v2_cameras.py`;
- in the Export dialog, under the Isaac row, before anything is exported (`principal_point_lines`, no forward
  kinematics).

## Validation (measured 2026-11-07, full run)

| Check | Quantity | Reference | Tolerance | Measured |
|---|---|---|---|---|
| `robot_io.urdf_mujoco` | sensor body and optical frames of `rig.urdf` loaded unchanged by `MjModel.from_xml_path`, 10 random states, SO-100, UR5e, bimanual Panda; inertials | DexVantage's poses; MuJoCo's URDF compiler; the robots' own MuJoCo models | 10⁻⁶ m (rotations 10⁻⁶ rad; mass, inertia 10⁻⁹ relative; COM 10⁻⁹ m) | 5.3 × 10⁻¹⁶ m, 1.9 × 10⁻¹⁵ rad; mass 0, COM 0, inertia 2.3 × 10⁻¹⁵ |
| `robot_io.usd_roundtrip` | every link of SO-100, UR5e + 2F-85, Panda written to USD and imported back, 50 random states; limits; writer poses through pxr | the robots' MuJoCo models; pxr's XformCache | 10⁻⁶ m (10⁻⁶ rad; limits equal to the stored float32 decoded and within its spacing; masses 10⁻⁶) | 2.5 × 10⁻⁸ m, 1.0 × 10⁻⁷ rad; limits 1.0 × 10⁻⁷ rad (spacing bound 5.3 × 10⁻⁷); writer 0 m; mass 4.5 × 10⁻⁸ |
| `robot_io.usd_third_party` | the newton-assets UR5e (written by the mujoco-usd-converter from Menagerie's ur5e.xml) imported, 7 arm bodies, 50 random states | DexVantage's UR5e sample | 10⁻⁶ m (10⁻⁶ rad, limits 10⁻⁶ rad, masses 10⁻⁶) | 2.6 × 10⁻⁸ m, 1.9 × 10⁻¹⁵ rad; limits 1.0 × 10⁻⁷ rad; mass 4.2 × 10⁻⁸ |
| `robot_io.openvla_encoding` | encodings, fit, image keys and notes of six rigs; the written entry executed against OpenVLA's enum classes copied verbatim | the pinned sources' rules restated in the check | 0 mismatches | 0 |
| `robot_io.isaac_principal_point` | fx, fy, cx, cy against Isaac Lab v3.0.0's intrinsics rule (written again) applied to the re-opened prims; notes, dialog, JSON | the imagers' lens models | 10⁻³ px | 4.8 × 10⁻⁵ px |

Why these tolerances: the URDF writes 17 significant digits, so anything above rounding is a wrong frame (millimetres
or degrees). UsdPhysics stores joint frames and limits as 32-bit floats (about 6 × 10⁻⁸ relative), which sets the
USD figures; 10⁻⁶ m is far below an axis sign, anchor or frame-swap error. The third-party file was written by another
program, so a convention the importer and DexVantage's writer might share by mistake would show there. The OpenCV
attributes are 32-bit floats too (about 3 × 10⁻⁵ px at 640 px). The USD checks are registered only when usd-core is
installed (as `formats.bracket_step` is only with gmsh). Isaac Lab, Isaac Sim and OpenVLA are not installed: the
exports are checked against the pinned sources, not by running those programs.

## Known limits

- URDF: no collision geometry; mimic joints and MuJoCo equality constraints are not written; sensor links carry the
  minimal placeholder, not the device's mass; links MuJoCo merged into a parent carry the placeholder (their mass is
  in that parent).
- USD import: one articulation per file; loops (four-bars, parallel grippers' second links), mimic and tendon
  couplings, drives, friction, damping, armature, effort and velocity limits are not imported; materials become one
  colour per geom; textures, cones, planes, height fields, point instancers and subdivision are not read; a body's
  non-uniform scale is applied to its points and primitives but not to its joint-frame rotations or its inertia;
  mesh-only collision shapes are used as MuJoCo meshes (convex hulls in MuJoCo's collision); time samples are read
  at the earliest time; the wizard's mesh step lists the stage's layers, not each mesh.
- The bundled third-party UR5e (`dexvantage/resources/usd_robots/newton_ur5e/`, BSD-3-Clause, THIRD_PARTY.md) lacks
  its 6.3 MB `GeometryLibrary.usdc` (over the 5 MB file budget), so its visual meshes are left out (said in the
  notes); its bodies, joints, limits, masses and collision primitives are read.
- OpenVLA: the choice is per rig, not per recorded dataset; the dataset's arrays are not converted; the policy's
  `action_space` is not offered on the Policies page yet (set it as the export option or in the project file).
- Isaac: v2.3.2 cannot show an off-centre principal point at all (stated per camera); the OpenCV attributes' pixel
  convention is taken to be OpenCV's (the source does not state it), half a pixel from Isaac's pinhole `width / 2`.
