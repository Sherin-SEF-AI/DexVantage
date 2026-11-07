# Sample projects

Three samples open in this build (SO-100, UR5e + Robotiq 2F-85, and the UR5e with a Shadow Hand for the multi-finger
occlusion study, [dexhand.md](dexhand.md)); the others of the specification arrive in Phase 11 and say so.

    dexvantage --sample so100                    # in the window (or File ▸ Open sample in a running window)
    dexvantage --sample ur5e --analyse -         # in batch mode: the project summary as JSON on standard output

The robots are MuJoCo Menagerie models bundled with DexVantage after their own licences were read
(`THIRD_PARTY.md`; each model folder under `dexvantage/resources/robots/` keeps its `LICENSE`). Joint limits, link
geometry and the home pose are the model files' own. **Camera placements are sample placements** chosen so the rig has
something to analyse, not published figures: move them to where your cameras are.

## SO-100 (`--sample so100`)

- **Robot:** the SO-ARM100 from Menagerie `trs_so_arm100` (Apache-2.0), at the file's `home` keyframe. The arm
  reaches along the base frame's −y axis, as the file defines it; the world frame is the robot's base (REP-103).
  Its joint limits are the ones Menagerie added as "approximate joint limits" (the model's README), not The Robot
  Studio's figures; the robot's source note says so.
- **Gripper frame:** the `Fixed_Jaw` body; fingertips `Fixed_Jaw` and `Moving_Jaw`.
- **Cameras:** "Wrist camera (sample placement)" on `Fixed_Jaw`, on a bracket 0.06 m behind the jaws on the wrist
  side, looking down past the jaw tips; "Third-person camera (sample placement)" on a tripod in front of and beside
  the table, looking at the task volume. Neither has a library device: no maker in the library states a camera for
  the SO-100, so both use the labelled default camera (pinhole, 1280 × 720 px, 90° × 60°) until you choose one on
  the Sensors page.
- **Workspace:** a 0.30 × 0.20 × 0.12 m tabletop volume under the jaw tips at home.
- **Tasks and policy (sample choices):** the grasp set "Top-down pinches over the tabletop (sample)" (100 top-down
  pinches sampled over the volume, seed 0; the five-joint arm cannot hold most side grasps) and the policy "SmolVLA
  (sample)" (its paper evaluates on SO-100 rigs), so the grasp self-occlusion and VLA token studies run as the sample
  opens (`--grasp`, `--vla-tokens`). SmolVLA's inference time is not stated by its source, so `--action-chunk`
  refuses until you enter a measured one on the Policies page.
- **Teleop device (sample choice):** "SO-100 leader arm (sample)": the leader is the follower's own model, joint to
  joint, as the LeRobot SO-100 guide builds it, so `--teleop` runs as the sample opens (96.7 % teleoperable).

Tour: open the sample and move a joint of the arm; the wrist camera rides with the gripper
while the tripod camera stays put. The SO-101 is in Menagerie too (`robotstudio_so101`, Apache-2.0, 18 MB of
meshes, not bundled): Robot ▸ Import from MuJoCo Menagerie downloads it and shows its licence first.

## UR5e + Robotiq 2F-85 (`--sample ur5e`)

- **Robot:** the UR5e (`universal_robots_ur5e`, BSD-3-Clause) with the 2F-85 (`robotiq_2f85`, BSD-2-Clause) on its
  flange. `ur5e_robotiq_2f85.xml` was written by MuJoCo itself: `MjSpec.attach` of the gripper at the UR5e file's
  `attachment_site`, names prefixed `gripper_` (`scripts/bundle_menagerie.py`). A test checks the gripper's pose
  against the two separate files compiled by MuJoCo.
- **Gripper frame:** `gripper_base` (the body with the file's `pinch` site); fingertips `gripper_right_pad` and
  `gripper_left_pad`.
- **Sensors:** "Wrist D405", a depth camera on `wrist_3_link`, 0.07 m off the tool axis, looking at the pinch point;
  "Overhead D435", a depth camera on the cage 0.85 m above the task volume, looking straight down. Both name
  their library entries (`intel-realsense-d405`, `intel-realsense-d435`): lenses, depth FOV, baseline, the
  minimum-Z table and the stated RMS error come from Intel's D400 series datasheet (Rev 023, read 2026-11-06). Where
  Intel states no offset (the D435's RGB imager, the projector) the entry places it as a labelled assumption.
- **Workspace:** a 0.50 × 0.40 × 0.30 m tabletop volume under the gripper at home.
- **Tasks and policy (sample choices):** the grasp set "Tabletop grasps (sample)" (200 grasps: top-down, side and
  angled, seed 0) and the policy "π0 (sample)", so every Phase 3, 6 and 7 batch flag runs on it: `--grasp`,
  `--vla-tokens`, `--bandwidth`, `--cables`, `--brackets`, `--handeye-plan`. There is no task object: add one to
  count its tokens and its grasp region. A scripted pick trajectory ("Scripted pick (sample)", 4.5 s) gives the motion
  blur, time sync and action-chunk studies their motion (`--motion-blur`, `--time-sync`, `--action-chunk`).
- **Not in the samples:** lights (`--lighting` says "no lights": the lighting guide adds a key light and a window),
  a set exposure (smear figures are upper bounds) and, on the UR5e and dexhand samples, a teleop device.
- **Geometry:** the Menagerie UR5e rounds its link offsets to the millimetre, so its flange is up to 1.3 mm from
  the flange of Universal Robots' DH table ([robots.md](../models/robots.md#known-limits)).

Tour: open the sample and run Analysis ▸ Run the project summary (F5): one robot, two depth sensors, one on a link.
Turn the shoulder: the wrist D405 follows the flange, the overhead D435 does not.

Known limits: the 2F-85's linkage joints are separate entries of the joint vector, so moving one finger joint alone
poses the linkage in a way the real gripper cannot (the coupling is an equality constraint in the file, which
kinematics does not apply).

## Other Menagerie robots

Robot ▸ Import from MuJoCo Menagerie… lists Menagerie's models (from GitHub, or a built-in list of names when
offline), downloads the one you pick into `XDG_DATA_HOME/DexVantage/menagerie/`, shows that model's own LICENSE and
imports it only after you tick "I accept the licence of this model". The robot records the licence, the Menagerie
commit and the date you accepted it, and the project keeps the LICENSE text.
