# Robots: import, the MuJoCo model and the samples {#robots}

Modules: `dexvantage/core/robot/importer.py`, `model.py`, `filecheck.py`, `trajectory.py`, `samples.py`,
`menagerie.py`, `catalogue.py` ([the robot catalogue](#catalogue)). Tests: `tests/test_robot_core.py`,
`test_samples.py`, `test_menagerie.py`, `test_robot_ui.py`, `test_phase1_integration.py`, `test_robot_catalogue.py`. Forward kinematics is checked in [kinematics.md](kinematics.md); IK, manipulability and
collision distances in [ik.md](ik.md).

## Import

`import_robot(path)` reads a URDF or MJCF file and every file it names (meshes, included XML, textures) into memory.
The project keeps them byte for byte (the `.dexv` archive's `sources/`), so a robot opens on any machine.

- **Where the files go.** Under `<robot uid>/<path relative to the robot's folder>`; a file outside that folder under
  `<uid>/_ext/<name>`. A manifest `<uid>/.dexvantage/refs.json` records which stored file each reference means, so the
  model compiles again from memory exactly as imported. It holds no uid, so the content hash is stable.
- **How references resolve.** URDF `package://pkg/…` is looked for by walking up from the URDF to a folder named
  `pkg`; `file://` and absolute paths are used as written; anything else is relative to the URDF. MJCF includes and
  `meshdir`, `texturedir` and `assetdir` are resolved as MuJoCo does: the list of files comes from MuJoCo's own parser
  (`mujoco.MjSpec`), not a hand-written scan.
- **Figures.** Joint limits, velocities and efforts are the file's own (radians, metres, rad/s or m/s, N·m or N); a
  figure the file does not state is `None` and shows as "—". The home pose is the MJCF keyframe `home`, else zero
  clipped to the limits.
- **Units check.** A mesh more than 10 m across (`MESH_MAX_EXTENT`, a plausibility default of this build) is almost
  certainly in millimetres: the import stops and says to set `scale="0.001 0.001 0.001"`. The import wizard also
  flags a robot under 20 mm or over 5 m across (a plausibility default, labelled as one).
- **Errors** are sentences that name the file and what to pick; a missing MJCF texture is not fatal and is recorded in
  `Robot.source.note`.

## The kinematics model

`RobotModel(robot, sources)` compiles the stored files with MuJoCo from an in-memory asset dictionary (nothing is read
from disk). URDF is compiled with `discardvisual="false"` and `fusestatic="false"` so visual meshes and links joined by
fixed joints keep their own bodies and names. Visual and collision geoms are told apart by MuJoCo's collision masks
(`contype = conaffinity = 0` is visual only).

- `fk(q)`, `link_pose(q, link)`, `fk_batch(Q)`: 4 × 4 world transforms, world = `Robot.base` applied to MuJoCo's frame.
- `jacobian(q, link, point)`: the (6, nq) world-frame [linear; angular] Jacobian from `mj_jac`.
- `posed_triangles(q)`: the visual meshes at q, for drawing and (Phase 3) occlusion.
- q holds the one-degree-of-freedom joints (hinge, slide) in MuJoCo's tree order. Fixed joints are not in q; the
  project summary, the Outliner and the 3D view's caption count the joints in q.

When MuJoCo refuses a model only because a moving link has no mass (common in kinematics-only URDFs), it is compiled
again with mass and inertia bounds of 10⁻⁶ kg and 10⁻⁹ kg·m², and the model's notes say so. Kinematics does not
depend on mass.

`get_model(project, uid)` is a thread-safe cache shared by the UI and the studies, keyed by the SHA-256 of the
robot's files, its records and its base pose.

## Robot records the UI writes

- A gripper joint has `"gripper": true` on its record in `Robot.joints` (the wizard's joint tagger; the samples tag
  theirs). IK leaves joints that do not move the target link alone either way (`manipulability.chain_mask`).
- `Robot.gripper_frame` and `Robot.fingertip_frames` name links. The wizard guesses the gripper frame by name (tcp,
  tool0, ee, gripper, hand, flange, …) and never guesses a link that a gripper joint moves.
- A contact pad is `{"link", "name", "box": {"center": [m], "size": [m]}}`, an axis-aligned box in the link frame.
- A trajectory is `{"uid", "name", "robot_uid", "joints": [names], "t": [s], "q": [[per joint] per sample]}` in
  `project.tasks.trajectories`, interpolated linearly by the pose scrubber.

## Bundled models and samples

Three MuJoCo Menagerie models are bundled after their own licences were read (`THIRD_PARTY.md`): `trs_so_arm100`
(Apache-2.0), `universal_robots_ur5e` (BSD-3-Clause) and `robotiq_2f85` (BSD-2-Clause), from Menagerie commit
`98a70c569c4c`. The UR5e's OBJ meshes are stored as binary STL (the same triangles as MuJoCo compiles them, a
difference of 0 m; 31.0 MB of text became 6.1 MB). `ur5e_robotiq_2f85.xml` is the arm with the gripper attached at
its `attachment_site`, written by MuJoCo's `MjSpec.attach` (`scripts/bundle_menagerie.py`).

| Sample | Robot | Joints in q | Gripper frame | Sensors (sample placements) |
|---|---|---|---|---|
| `so100` | SO-ARM100 | 6 (5 arm, 1 jaw) | `Fixed_Jaw` | wrist camera on `Fixed_Jaw`, third-person camera on a tripod |
| `ur5e` | UR5e + Robotiq 2F-85 | 14 (6 arm, 8 linkage hinges) | `gripper_base` | "Wrist D405" on `wrist_3_link`, "Overhead D435" 0.85 m above the volume |

The camera placements are chosen for the samples, not published figures, and say so in their names or notes. The
sensors get their datasheet models in Phase 2. Robot ▸ Import from MuJoCo Menagerie downloads any other Menagerie
model, shows its own LICENSE and imports it only after the user accepts it ([guide](../user-guide.md#robots)).

## The robot catalogue {#catalogue}

Module: `dexvantage/core/robot/catalogue.py`; entries `dexvantage/resources/library/robot/<id>.json` (library kind
`robot`, the `robot` block of `library.schema.json`); UI: the Library dock's Robots group and
`ui/dialogs/robot_catalogue.py`; check `validation/checks_robots.py`; tests `tests/test_robot_catalogue.py`.

One entry per robot: arms, a bimanual rig, hands, grippers, humanoids and mobile manipulators. Each entry holds

- **the maker's figures**: `dof`, `reach_mm`, `payload_kg`, `repeatability_mm` (the ± half-width), `mass_kg`,
  `control_hz`, `interfaces` and `joint_ranges` (per joint min and max, in the `unit` the maker uses). Each is
  `{value, source, quote}`: `source` indexes the entry's `sources` (URL, title, the date read, what it covers) and
  `quote` is the source's words. A figure no source read states is null and shows as "—" ("not stated by the
  maker"). `dof` is an integer, or `[low, high]` where the maker states a range for configurable versions (Unitree
  G1 EDU: "23 - 43"). Nothing is filled in from memory or from the model file; joint limits used for kinematics
  always come from the imported file (§6), the catalogue's `joint_ranges` are the datasheet's, for comparison.
- **`model`**: the MuJoCo Menagerie folder (one of `menagerie.OFFLINE_MODELS`), the MJCF file the catalogue imports,
  the commit (`98a70c569c4c1b6da467a20012f766ee1f8dc521`) and the licence `menagerie.license_name()` recognised in that
  folder's own LICENSE, read on 2026-11-07. Every model the catalogue names carries Apache-2.0, BSD-3-Clause,
  BSD-2-Clause, BSD-3-Clause-Clear or MIT, and none restricts commercial use. Nothing new is bundled: Import downloads
  the folder and shows its LICENSE again.
- **`verified`**: what DexVantage measured when the folder was downloaded once at that commit (a partial clone into
  a throw-away folder) and imported through `core/robot/importer.py` and `RobotModel`: joints in q (hinge and slide
  joints), actuators (MuJoCo `nu`), links, free joints, the joint names, the import time, and the DoF under a stated
  rule. `dof_rule = joints` counts joints in q minus `excluded_joints` (the finger or linkage joints of an arm file
  that includes its gripper: Panda, xArm 7, ViperX, WidowX, PiPER, ALOHA, ...); `dof_rule = actuators` counts
  actuators (the Robotiq 2F-85, whose eight linkage hinges are driven by one actuator). A model that does not import
  says so in plain words; all 53 imported. `dof_differs` explains a model DoF that differs from the maker's; no entry
  needs one at this commit.

### Import from the Library

The Library dock's Robots group lists the catalogue under its two import commands, with type, DoF, reach and payload
as stated; hovering shows every figure with its source and date, the model and what was verified. Double-click (or
right-click ▸ Import from MuJoCo Menagerie…) opens the Menagerie dialog narrowed to the robot's folder, starts the
download, picks the catalogue's file and shows the folder's own LICENSE. Import stays disabled until "I accept the
licence of this model" is ticked; the import is one undo step and records the licence on the robot
(`Robot.source.license`) with the LICENSE text in the project. `catalogue.import_entry(entry, opener=...)` is the same
flow without the UI (the caller shows the licence).

### Entries

DoF (maker) is the stated figure; "joints / actuators / links" and DoF (model) are the verified record at
`98a70c569c4c`.

| Robot | Type | Menagerie folder | Licence | DoF (maker) | Joints / actuators / links | DoF (model) |
|---|---|---|---|---|---|---|
| AgileX Robotics PiPER | Arm | `agilex_piper` | MIT | 6 | 8 / 7 / 9 | 6 (−2 gripper) |
| Alexander Koch v1.1 (Low-Cost Robot Arm, LeRobot update by Jess Moss) | Arm | `low_cost_robot_arm` | Apache-2.0 | — | 6 / 6 / 7 | 6 |
| Apptronik Apollo | Humanoid | `apptronik_apollo` | Apache-2.0 | — | 32 / 32 / 36 | 32 |
| ARX L5 | Arm | `arx_l5` | BSD-3-Clause | — | 8 / 7 / 9 | 6 (−2 gripper) |
| Booster Robotics T1 | Humanoid | `booster_t1` | Apache-2.0 | — | 23 / 23 / 24 | 23 |
| Carnegie Mellon University LEAP Hand | Hand | `leap_hand` | MIT | 16 | 16 / 16 / 17 | 16 |
| FANUC CRX-5iA | Arm | `fanuc_crx5ia` | Apache-2.0 | 6 | 6 / 6 / 7 | 6 |
| Flexiv Rizon 4 | Arm | `flexiv_rizon4` | Apache-2.0 AND BSD-3-Clause (one LICENSE file, both texts) | — | 7 / 7 / 8 | 7 |
| Flexiv Rizon 4s | Arm | `flexiv_rizon4s` | Apache-2.0 | — | 7 / 7 / 8 | 7 |
| Fourier N1 | Humanoid | `fourier_n1` | Apache-2.0 | — | 23 / 23 / 29 | 23 |
| Franka Emika Panda | Arm | `franka_emika_panda` | Apache-2.0 | 7 | 9 / 8 / 11 | 7 (−2 gripper) |
| Franka Robotics Franka Research 3 | Arm | `franka_fr3` | Apache-2.0 AND BSD-3-Clause (one LICENSE file, both texts) | 7 | 7 / 7 / 9 | 7 |
| Franka Robotics Franka Research 3 (Menagerie fr3_v2 model) | Arm | `franka_fr3_v2` | Apache-2.0 AND BSD-3-Clause (one LICENSE file, both texts) | 7 | 7 / 7 / 10 | 7 |
| Google Robot (Everyday Robots, RT-1) | Mobile manipulator | `google_robot` | Apache-2.0 | — | 9 / 9 / 12 | 9 |
| Google DeepMind ALOHA 2 | Bimanual | `aloha` | BSD-3-Clause | — | 16 / 14 / 20 | 12 (−4 gripper) |
| Hello Robot Stretch 2 | Mobile manipulator | `hello_robot_stretch` | BSD-3-Clause-Clear | — | 17 / 8 / 26 | 17 |
| Hello Robot Stretch 3 | Mobile manipulator | `hello_robot_stretch_3` | Apache-2.0 | — | 20 / 10 / 37 | 20 |
| I2RT Robotics YAM | Arm | `i2rt_yam` | MIT | — | 8 / 7 / 13 | 6 (−2 gripper) |
| Kinova Gen3 (7 DoF) | Arm | `kinova_gen3` | BSD-3-Clause | 7 | 7 / 7 / 8 | 7 |
| KUKA LBR iiwa 14 R820 | Arm | `kuka_iiwa_14` | BSD-3-Clause | 7 | 7 / 7 / 8 | 7 |
| PAL Robotics TALOS | Humanoid | `pal_talos` | Apache-2.0 | — | 44 / 0 / 45 | 44 |
| PAL Robotics TIAGo | Mobile manipulator | `pal_tiago` | Apache-2.0 | — | 22 / 0 / 23 | 22 |
| PAL Robotics TIAGo (dual arm) | Mobile manipulator | `pal_tiago_dual` | Apache-2.0 | — | 25 / 0 / 26 | 25 |
| PNDbotics Adam Lite | Humanoid | `pndbotics_adam_lite` | MIT | — | 25 / 25 / 26 | 25 |
| Rainbow Robotics RB-Y1 | Mobile manipulator | `rainbow_robotics_rby1` | Apache-2.0 | — | 28 / 26 / 32 | 28 |
| Rethink Robotics Sawyer | Arm | `rethink_robotics_sawyer` | Apache-2.0 | 7 | 7 / 7 / 9 | 7 |
| Robotiq 2F-85 | Gripper | `robotiq_2f85` | BSD-2-Clause | — | 8 / 1 / 14 | 1 (actuators) |
| Robotiq 2F-85 (Menagerie v4 model) | Gripper | `robotiq_2f85_v4` | BSD-2-Clause | — | 6 / 1 / 11 | 1 (actuators) |
| ROBOTIS OMY-F3M | Arm | `robotis_omy_3m` | Apache-2.0 | 6 | 6 / 6 / 7 | 6 |
| ROBOTIS OP3 | Humanoid | `robotis_op3` | Apache-2.0 | 20 | 20 / 20 / 21 | 20 |
| Seeed Studio reBot DevArm | Arm | `seeed_rebot_devarm` | MIT | — | 8 / 8 / 10 | 6 (−2 gripper) |
| Shadow Robot Company DEX-EE | Hand | `shadow_dexee` | Apache-2.0 | — | 12 / 12 / 19 | 12 |
| Shadow Robot Company Shadow Dexterous Hand (E3M5) | Hand | `shadow_hand` | Apache-2.0 | 24 | 24 / 20 / 25 | 24 |
| SharpaWave | Hand | `sharpa_wave` | Apache-2.0 | — | 22 / 22 / 23 | 22 |
| Stanford University TidyBot++ | Mobile manipulator | `stanford_tidybot` | MIT | — | 18 / 11 / 22 | 18 |
| Stanford University ToddlerBot 2.0 (2xc) | Humanoid | `toddlerbot_2xc` | MIT | — | 44 / 30 / 45 | 44 |
| Stanford University ToddlerBot 2.0 (2xm) | Humanoid | `toddlerbot_2xm` | MIT | — | 44 / 30 / 45 | 44 |
| Stanford, Columbia and TRI UMI gripper (Universal Manipulation Interface) | Gripper | `umi_gripper` | MIT | — | 8 / 7 / 3 | 8 |
| TetherIA Aero Hand Open | Hand | `tetheria_aero_hand_open` | Apache-2.0 | — | 16 / 7 / 22 | 16 |
| The Robot Studio SO-101 | Arm | `robotstudio_so101` | Apache-2.0 | — | 6 / 6 / 8 | 6 |
| The Robot Studio SO-ARM100 | Arm | `trs_so_arm100` | Apache-2.0 | — | 6 / 6 / 7 | 6 |
| Trossen Robotics ViperX-300 6DOF | Arm | `trossen_vx300s` | BSD-3-Clause | 6 | 8 / 7 / 10 | 6 (−2 gripper) |
| Trossen Robotics WidowX AI | Arm | `trossen_wxai` | BSD-3-Clause | — | 8 / 7 / 13 | 6 (−2 gripper) |
| Trossen Robotics WidowX-250 6DOF | Arm | `trossen_wx250s` | BSD-3-Clause | 6 | 8 / 7 / 9 | 6 (−2 gripper) |
| UC Berkeley Humanoid | Humanoid | `berkeley_humanoid` | BSD-3-Clause | — | 12 / 12 / 13 | 12 |
| UFACTORY Lite 6 | Arm | `ufactory_lite6` | BSD-3-Clause | 6 | 6 / 6 / 7 | 6 |
| UFACTORY xArm 7 | Arm | `ufactory_xarm7` | BSD-3-Clause | 7 | 13 / 8 / 15 | 7 (−6 gripper) |
| Unitree G1 | Humanoid | `unitree_g1` | BSD-3-Clause | 23–43 | 29 / 29 / 30 | 29 |
| Unitree H1 | Humanoid | `unitree_h1` | BSD-3-Clause | — | 19 / 19 / 20 | 19 |
| Unitree Z1 | Arm | `unitree_z1` | BSD-3-Clause | 6 | 6 / 6 / 7 | 6 |
| Universal Robots UR10e | Arm | `universal_robots_ur10e` | BSD-3-Clause | 6 | 6 / 6 / 7 | 6 |
| Universal Robots UR5e | Arm | `universal_robots_ur5e` | BSD-3-Clause | 6 | 6 / 6 / 7 | 6 |
| Wonik Robotics Allegro Hand V4 | Hand | `wonik_allegro` | BSD-2-Clause | 16 | 16 / 16 / 21 | 16 |

Where the figures come from: the makers' current datasheets or product pages (Franka FR3 datasheet R02212 2.4,
Universal Robots' UR5e and UR10e datasheets, Kinova's Gen3 one-pager 2024, KUKA's LBR iiwa page, UFACTORY's xArm and
Lite 6 pages, Unitree's Z1, G1 and H1 pages, Trossen's Interbotix specification pages, Wonik's Allegro Hand V4 page,
Shadow's Dexterous Hand series table, Robotiq's adaptive-gripper product sheet, ROBOTIS's OMY and OP3 e-Manual pages,
AgileX's PiPER page, FANUC America's CRX-5iA page, PAL Robotics' TALOS and TIAGo pages); for products no longer sold,
the maker's last datasheet (the Panda's May 2019 datasheet through the Internet Archive's copy of Franka's download,
Rethink's 2015 Sawyer datasheet through a distributor's copy, each named in its source); for research hardware, the
paper (LEAP Hand, UMI). Twenty-two entries have no maker figure at all: their maker's page could not be read (scripts,
404, no answer) or was not read for this build, and the entry note says which. Their figures stay null; the entry
still carries its model, licence and verified record.

### Validation

`robots.catalogue_dof` (model `robots`, **a data-consistency check, labelled as one**): for every entry, the verified
DoF must follow from its own counts under its rule, the model and verified commits and files must agree, and the
verified DoF must equal the maker's stated DoF (inside a stated range) unless `dof_differs` explains it. Independently,
the Menagerie models bundled with DexVantage (`core/robot/samples.py` BUNDLED, copied from the same commit) whose
folder and file the catalogue names are imported again now and their joints in q, actuators and links are counted
with MuJoCo and compared with the catalogue's records; a bundled file the catalogue does not name (the combined
`unitree_g1/g1_with_hands.xml`) is listed, not compared. Tolerance 0 entries: every quantity is a count, so a
difference of one is a data error, not a numerical one.

Measured (2026-11-07): 0 disagreements over 53 entries; 21 entries state a DoF and all 21 agree with the imported
model (the G1's 29 inside the EDU's 23–43); the six bundled models counted again give exactly the recorded counts
(SO-ARM100 6 / 6 / 7, UR5e 6 / 6 / 7, 2F-85 8 / 1 / 14, Shadow Hand 24 / 20 / 25, ALOHA 16 / 14 / 20, Panda
9 / 8 / 11). The self-test line "robot
catalogue" repeats the per-entry part (0.7 s).

### Known limits (catalogue)

The figures describe the product, not the Menagerie file, which its own README calls a "simplified" description;
makers state payload, reach and repeatability under their own conditions (full or mid reach, ISO 9283 or not), which
the quotes keep. A DoF is the maker's count: the Shadow Hand's 24 joints include 4 under-actuated ones (20 actuated);
the LEAP Hand's 16 is the paper's count of motor joint angles. Unitree's Z1 page gives AIR and PRO figures and the
Menagerie README does not say which the model is; the entry uses the AIR's. Two entries describe the same product
through two Menagerie models (FR3 and FR3 v2; 2F-85 and 2F-85 v4). `verified` records one download at one commit; a
later Menagerie commit may differ until the catalogue is checked again (the Import dialog always shows the licence of
the version actually downloaded). Robots without a Menagerie model (no maker URDF was needed for the list the
specification names) would carry `model.source = "urdf"`; none does yet.

## Validation

The kinematics checks run on the bundled samples as well as on the robots written in the checks: with no project
given, the validation suite's built-in project holds both sample robots. Measured with the default seed (full run):

| Check | Robots | Error | Tolerance |
|---|---|---|---|
| `kinematics.fk_poe_mujoco` | both samples and five robots written in the check, 200 random q each, every link | 8.7 × 10⁻¹⁵ m | 10⁻⁹ m |
| `kinematics.jacobian_mujoco` | the same robots, 40 random q each, at each link's origin and a random point fixed in the link | 8.8 × 10⁻¹⁵ | 10⁻⁹ |

`dexvantage --self-test` repeats the comparison quickly ("robots": both samples import, MuJoCo and the
product-of-exponentials chain agree within 10⁻⁹ m at five random poses each).

## Known limits

xacro must be expanded to plain URDF first; USD is not read yet (the wizard says so).
URDF mimic joints and the 2F-85's linkage move independently: MuJoCo's equality constraints are not applied by
kinematics, so moving one finger joint alone poses the linkage in a way the real gripper cannot. Free and ball
joints are held at the home keyframe and are not part of q. Planes and height fields are not drawn. MJCF `<model>`
(attach) assets and plugins are not collected, and the independent reader stops with a plain error on `replicate`,
`composite` and `flexcomp`. Frames are links: MJCF sites
cannot be the gripper frame yet. The SO-101 is not bundled (18 MB of meshes); import it from Menagerie.

What the bundled files are, by Menagerie's own READMEs (read 2026-11-06): "simplified" MJCF descriptions derived
from public URDFs. The SO-ARM100's joint limits are ones Menagerie added as "approximate joint limits" (its derivation
step 7), not The Robot Studio's figures; the robot's source note says so. The UR5e's link offsets are rounded to the
millimetre (d1 = 0.163 m against Universal Robots' 0.1625 m, d5 = d6 = 0.100 m against 0.0997 m and 0.0996 m): after
the best rigid alignment its flange is up to 1.3 mm (0.9 mm RMS over 500 random poses) from the flange of Universal
Robots' DH table (`tests/test_samples.py`). Plan with the maker's figures where a millimetre matters.
