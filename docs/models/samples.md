# The ALOHA 2, bimanual Panda and G1 samples {#samples}

Modules: `dexvantage/core/robot/bimanual_samples.py` (the three builders), `dexvantage/core/robot/samples.py`
(`SAMPLES`, `BUNDLED`, `sample_project`). Bundling: `scripts/bundle_samples.py`. Validation:
`dexvantage/validation/checks_samples.py` (model `samples`). Tests: `tests/test_samples_bimanual.py`. Self-test line:
`samples`. Batch: `dexvantage --sample aloha|panda-bimanual|g1 …` with any study flag.

Spec §11 names five samples (`--sample aloha|so100|ur5e|panda-bimanual|g1`) and §14 says what each holds. This page
describes how the three bimanual and humanoid ones are built from their official model files, what was changed in
those files and why, and how each change was checked.

## Sources and licences

All three come from MuJoCo Menagerie at commit `98a70c569c4c1b6da467a20012f766ee1f8dc521` (2026-11-02; the same
commit as the SO-100, UR5e and Shadow Hand). Each folder's own `LICENSE` was read on 2026-11-07, is kept beside the
bundled files and is recorded in `THIRD_PARTY.md` and `samples.BUNDLED`:

| Folder | Licence | Folder commit | Bundled | Files changed |
|---|---|---|---|---|
| `aloha` | BSD-3-Clause, © 2023 Trossen Robotics | `fb36886b30` | 8.3 MB | none (byte for byte); two arm files and `scene_parts.xml` written beside them |
| `franka_emika_panda` | Apache-2.0 | `fb36886b30` | 6.9 MB | the 59 OBJ meshes as binary STL with the same triangles (34.3 MB of text → 6.8 MB) |
| `unitree_g1` | BSD-3-Clause, © 2016-2023 Unitree Robotics | `98a70c569c` | see below | the 49 STL meshes **simplified** (measured below); an upper-body file written beside it |

`franka_fr3` (Apache-2.0) was not used: its Menagerie model has no hand. Every figure that is not in a model file has
a cited source (`bimanual_samples.SOURCES`, URL and date read) or is labelled a sample choice.

## ALOHA 2 (`aloha`)

**Two robots, not one.** `aloha.xml` holds both ViperX 300 arms in one file. The studies that work per gripper (the
grasp study, the teleop study's follower, the hand-eye plan, the grasp sampler's handoff partner) take one gripper
frame per robot, so the sample opens each arm as a robot of its own. `scripts/bundle_samples.py` writes
`aloha_left.xml` and `aloha_right.xml` with `mujoco.MjSpec`: the other arm, the actuators, the gripper equality
constraints, the keyframes and the world cameras deleted, the arm's `base_link` moved to the file's origin, and a
keyframe `home` written with this arm's values of `aloha.xml`'s `neutral_pose`. The sample places the arms with the
base poses of `aloha.xml` (parsed from the bundled file as the sample is built): the left arm is the primary robot (the
world is its base frame, REP-103) and the right arm stands at (0.938, 0, 0) m turned half a turn about z.

**One state vector for recordings.** Each arm keeps `aloha.xml`'s joint names (`left/waist` … `left/right_finger`,
`right/waist` …), so an ALOHA recording that stores both arms in one vector is read onto each arm by name
(`core/datasets/joints.py`: a dataset column a model does not name is ignored).

**Cameras.** The ALOHA 2 paper (arXiv 2405.02292v1, read 2026-11-07): "We upgrade the cameras used in the ALOHA
system to 4 RealSense D405 cameras" and "All four cameras record 848 x 480 rgb images" (§2.5); of the Menagerie model,
"We precisely model all cameras and robot positions of the ALOHA 2 specification". The four sensors are therefore
library D405s (`intel-realsense-d405`) with the 848 × 480 stream, posed from Menagerie's own cameras: the wrist
cameras `wrist_cam_left` / `wrist_cam_right` of `aloha.xml` (riding on `*/gripper_base`), and `overhead_cam` and
`worms_eye_cam` of `scene.xml` (static). `scene.xml` itself is not bundled (its table and frame meshes add 4.9 MB):
its two frame cameras and the table's collision box are copied with their attributes into `aloha/scene_parts.xml`.
A MuJoCo camera looks along its −z with +y up; the sensor body frame is x forward, y left, z up, so
R_body = R_cam · diag(1, −1, −1) · B, B the body-to-optical rotation of `core/frames.py`. The sensors carry the names
of the ALOHA recordings, `cam_high`, `cam_low`, `cam_left_wrist`, `cam_right_wrist` (tonyzhaozh/aloha
`aloha_scripts/constants.py`, read 2026-11-07).

**Table.** `scene.xml`'s collision box `table` (1.22 × 0.74 m, its top 0.9 mm below `aloha.xml`'s origin; the paper's
"48" x 30" table"), 20.9 mm below the arms' base plane.

**Teleop.** Two leader arms from the library's ALOHA leader entry (`aloha-widowx-250-leader`, its figures with
their sources): "two ViperX 6-DoF arms … along with 2 smaller WidowX arms … The WidowX contains the same kinematic
structure as the ViperX … The follower joints are synchronized with the leader arms" (the paper, §1). Each leads its
follower joint to joint; the WidowX's own joint ranges are not stated, so the follower's are used (the device record
says so: an assumption). With two followers the teleop study's bimanual part runs: reach overlap, the
handover zone's quality and the arm-to-arm margin.

**Sample choices** (labelled "(sample)"): a 0.50 × 0.40 × 0.20 m tabletop volume between the arms and a
0.20 × 0.20 × 0.15 m handover zone above the table's middle; a handover trajectory per arm (each TCP to 50 mm short
of the handover zone's centre along the line between the bases, the right gripper closes, both go home; position-only
IK from the waypoint before); grasp sets: 100 top-down and angled grasps per arm (TCP 60 mm above the table) and 60 bimanual handoffs for the left
arm (its partner the right arm); the policy ACT (Zhao et al. 2023, trained on ALOHA).

## Franka Panda bimanual (`panda-bimanual`)

There is no official bimanual Franka model (spec §14.4). The cell is two instances of Menagerie's single-arm
`franka_emika_panda/panda.xml` with the Franka Hand: the file is imported once and the second robot is a copy of its
records and files under a new uid (one compiled model serves both). The left arm is the primary robot; the right arm
stands 0.8 m along −y, both facing +x (a sample layout). Each hand carries a wrist RealSense D405 0.07 m off the tool
axis looking at the pinch point, and a RealSense D435 stands 0.85 m in front of the table looking back at the arms
(sample placements). The teleop devices are the library's GELLO entry ("we build a GELLO controller device that has the
same kinematic structure as the target arm", Wu et al. 2023), one per arm. Volumes, the handover trajectories, the grasp sets (100 top-down, side
and angled grasps per arm, 60 handoffs) and Diffusion Policy are sample choices.

**Mesh conversion.** `panda.xml` uses 59 OBJ visual meshes (34.3 MB of text). They are rewritten as binary STL with
the same triangles (float32), named `<stem>_obj.stl` because the folder already has STL collision meshes of the same
stem (`link1.stl` beside `link1.obj`); a `<mesh>` that had no name is given its old file stem, the name MuJoCo gave
it. The script compares every triangle of every geom at the home keyframe with the original: 0 m.

## Unitree G1 upper body (`g1`)

**The model.** Menagerie's `g1_with_hands.xml`: both arms (7 joints each) with a three-finger hand each (7 joints:
thumb 3, index 2, middle 2). Menagerie does not name the hand, so it is not called Dex3.

**Legs and waist held.** `g1_upper_body.xml` is `g1_with_hands.xml` with the free joint `floating_base_joint`
deleted and the pelvis placed at the `stand` keyframe's base pose (0, 0, 0.79 m, upright), and its 12 leg joints and 3
waist joints deleted. At `stand` all 15 are 0, so deleting them leaves every body exactly where `stand` puts it (the
script refuses if one is not 0). q is the 28 arm and hand joints, in one state vector; no joint is shared by the two
arms, so IK on one arm leaves the other put (`manipulability.chain_mask`). `stand` puts the thumbs' second joint at
±1.05 rad, 0.0028 rad past its ±1.0472 rad limit; DexVantage's home clips it to the limit.

**Mesh simplification (measured).** The 49 meshes `g1_with_hands.xml` uses are 31.5 MB of STL (629 338 triangles,
CAD exports: a 4 cm finger link has 30 434 triangles); the package budget is 12 MB per robot. They are simplified by
quadric edge collapse (fast-simplification 0.2.0, optional extra `bundle`; aggressiveness 1 and 3 tried, the fewest
triangles within tolerance kept) with a bisection on the reduction, each mesh as far as its **two-sided Hausdorff
distance** from the original stays within **0.5 mm** (a DexVantage choice: under 1.5 px of the head D435's
1920 × 1080 colour stream at 0.5 m). The distance is measured on each mesh's vertices plus 40 000 area-weighted samples
of its surface against the other's exact surface (point-to-triangle distance, Ericson 5.1.5, with an exact candidate
bound), both ways. Result: 629 338 triangles became 216 325 (10.82 MB of STL; the folder with its XML and LICENSE 10.9 MB); the largest distance of any mesh is 0.497 mm (the hand links, which were reduced most: a 9 518-triangle finger base to 2 376), the median 0.45 mm; one mesh (`pelvis_contour_link`) could not be reduced within the tolerance and is kept as it was. The simplified files are named `<stem>.simplified.stl`, the comment at the
top of `unitree_g1/g1_with_hands.xml` says so, and so do the robot's source note and the project notes. Where the
geometry matters to a millimetre, import the full model with Robot ▸ Import from MuJoCo Menagerie (`unitree_g1`).

**Head camera.** Unitree's own URDF `g1_29dof_rev_1_0.urdf` (unitreerobotics/unitree_ros, commit `7c40519`, read
2026-11-07) has `d435_joint`: torso_link → `d435_link`, xyz (0.0576235, 0.01753, 0.42987) m, rpy
(0, 0.8307767239493009, 0) rad (47.6° down). Its `torso_link` and `head_link` offsets equal Menagerie's (the head's
(0.0039635, 0, −0.044) m), so the frame is placed on Menagerie's `torso_link` as written. Unitree's G1 page states only
"Depth Camera+3D LiDAR", so the RealSense D435 is the model the frame's name gives, and the frame is taken as the
camera's body frame (x along the optical axis); the URDF does not say which point of the camera it marks.

**Palm and thumb contact.** Menagerie's own collision meshes of the palm (on `*_wrist_yaw_link`) and the thumb's middle
link (`*_hand_thumb_1_link`) overlap by 0.27 mm at `stand` (`mj_geomDistance` on the original files; the simplified
meshes give the same). The two are hinged together through the thumb's first link, so `g1_upper_body.xml` excludes the
pair from contact (`<contact><exclude>`, as MuJoCo excludes a parent and child); without it every arm pose counted as a
self-collision and the hand-eye planner found no pose.

**Hand frames.** The palm mesh is a geom of `right_wrist_yaw_link` (there is no palm body), so the gripper frame is
`right_wrist_yaw_link` and the fingertips `right_hand_thumb_2_link`, `right_hand_index_1_link`,
`right_hand_middle_1_link`; one gripper per robot, so the left hand moves with q but is not the grasp study's gripper.

**Sample choices.** A chest-high table 0.40 × 0.90 m, its top 0.08 m above the wrists at `stand` (0.80 m; at `stand`
the arms hang beside the hips), its near edge 0.20 m in front of the pelvis: there the head camera's view (47.6° down)
and the hands' reach meet (at a 0.62 m table the camera saw the volume but the right hand reached few angled grasps,
and the hands were out of view). A 0.20 × 0.50 × 0.16 m tabletop volume, a handover zone between the hands and
"Right hand's reach (sample)", 0.15 × 0.20 × 0.10 m, where the right hand can hold an angled grasp (over the whole
tabletop about 80 % of angled grasps are out of its reach); one trajectory moving both arms (each wrist TCP to a point
over the table, the right hand closing along its sample power posture, `core/dexhand.py default_synergies`); 100
angled grasps for the right hand in its reach volume; the multi-finger study's joint state set to the trajectory's
reach (the hands over the table, open), not `stand`; GR00T N1.5. The teleop device
is the library's Apple Vision Pro entry, as Unitree's xr_teleoperate drives the G1 ("Wear your XR headset (e.g. Apple
Vision Pro, PICO4, etc.)"); no source states its tracking volume, so it is left empty and the teleop study says so.

## Validation

`checks_samples.py`, every reference MuJoCo compiling the bundled files directly (not DexVantage's importer, model or
mounts):

| Check | Reference | Tolerance (why) | Measured |
|---|---|---|---|
| `samples.aloha_arms` | every link of both arm robots against `aloha.xml` (both arms in one file), left base frame, 50 random joint vectors | 10⁻⁶ m (`to_xml` writes about six significant digits) | 4.6 × 10⁻⁷ m |
| `samples.aloha_cameras` | the four sensors' world poses against MuJoCo's camera frames (`mj_camlight`) of `aloha.xml` and `scene_parts.xml` | 10⁻⁶ m (same rounding) | 5.1 × 10⁻⁷ m |
| `samples.g1_held_legs` | every link of the upper body against `g1_with_hands.xml` at `stand`, 51 joint vectors | 10⁻⁶ m | 0 m |
| `samples.g1_head_camera` | the head camera against Unitree's `d435_joint` composed on MuJoCo's `torso_link` (R = Rz Ry Rx written out) | 10⁻¹² m (rounding only) | 2.2 × 10⁻¹⁶ m |
| `samples.panda_instances` | both Pandas against `panda.xml`'s bodies moved by each base pose | 10⁻⁹ m | 0 m |

The bundling script measures what cannot be checked without the originals: the arm files against `aloha.xml`
(4.5 × 10⁻⁷), the frame cameras and table against `scene.xml` (0), the Panda's triangles against the OBJ files (0 m),
the G1 meshes' Hausdorff distances (above) and the upper body against the full model (0). The robot catalogue check
(`robots.catalogue_dof`) recounts the bundled `aloha.xml`, `panda.xml` and `g1_with_hands.xml` against its records.

## Known limits

- ALOHA 2: the grippers' two fingers are coupled by an equality constraint in `aloha.xml` that the arm files leave
  out (kinematics treats each finger joint on its own, as for the 2F-85); the WidowX leader arms are modelled with the
  followers' own joints and ranges; the aluminium frame and the camera mounts of `scene.xml` are not occluders (only
  the table is, as a surface).
- Panda: the cell layout, the camera placements and the GELLO leaders are sample choices; the two arms share one
  compiled model, so a change to one robot's file does not apply to the other.
- G1: the simplified meshes differ from Menagerie's by up to the measured Hausdorff distance; the legs and waist do not
  move and the pelvis is fixed in the world (a humanoid that steps, squats or bends at the waist is not modelled); the
  head camera's model is inferred from its frame's name; the stand keyframe's thumb value is clipped to its limit.
- Every study that works per gripper sees one gripper per robot (the G1's right hand).
- The G1's head camera rides on `torso_link`, which no joint moves here: the hand-eye planner and solve treat it as a
  static camera (`core/calib/records.py camera_mount`, a link no joint moves counts as the world), but the silhouette
  calibration still asks for a camera not on a robot link and refuses it.
- The synthetic episodes (`--synth-episodes`) move one robot, so an ALOHA or Panda campaign records one arm's joints.
