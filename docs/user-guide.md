# DexVantage user guide

How to use DexVantage, from the first ten minutes to a paper: one chapter per part of the window and per kind of
study, each with a worked example on a bundled sample. Every figure in a worked example is what this build computes
when you do the same: `scripts/guide_examples.py` runs each example (marked in this file's source by an invisible
`<!-- example: … -->` line) and `tests/test_guide_examples.py` fails when a figure printed here is no longer the one
the code gives, within the rounding printed. The ten-minute tour is also driven through the window by
`tests/test_ten_minute_tour.py`. A figure that depends on the GPU's rasteriser against numpy's may differ in its last
digit on another machine (the occlusion model's own agreement, [overview](overview.md#occlusion)).

The models behind each figure, with their equations, references, validation and known limits, are in
[overview.md](overview.md); the longer worked example of each feature is in `docs/guide/` (each chapter links its
own). The project file is described in [project-format.md](project-format.md).

**Contents.** 1 [The ten-minute tour](#ten-minute-tour) · 2 [Getting started](#getting-started) ·
3 [The window](#window) · 4 [Robots](#robots) · 5 [Sensors and coverage](#sensors) · 6 [The studies](#studies) ·
7 [Calibration](#calibration) · 8 [Data](#data) · 9 [Optimisation and requirements](#requirements) ·
10 [Export](#export) · 11 [The assistant and MCP](#assistant) · 12 [Research workflows](#research) ·
13 [Batch mode](#batch) · 14 [The Python API](#python-api) · 15 [Troubleshooting](#troubleshooting) ·
the [sample tours](guide/samples.md) · reference: [command line](reference/cli.md), [Python API](reference/api.md).

## 1. The ten-minute tour {#ten-minute-tour}

The specification's definition of done: install DexVantage, open the ALOHA 2 sample, and within ten minutes see
which grasps lose fingertip visibility and why, move a camera and see the score change, check depth validity, VLA
token coverage and bandwidth, export the cameras to MuJoCo and LeRobot, and export a printable bracket. On the
machine the figures come from (an RTX 5080 through EGL) the compute of the whole tour takes about two and a half
minutes; the rest is reading.

<!-- example: tour -->
1. **Open the sample.** `dexvantage --sample aloha` (or File ▸ Open sample ▸ ALOHA 2 bimanual with its four cameras).
   The window shows 2 robots (the two ViperX 300 arms of Menagerie's `aloha.xml`; the left one's base is the world
   frame) and 4 cameras: `cam_high (overhead)`, `cam_low (worms-eye)`, `cam_left_wrist` and `cam_right_wrist`, four
   RealSense D405s where the ALOHA 2 project puts them. Select a camera in the Outliner to see its image in the Camera
   view (right).
2. **Which grasps lose fingertip visibility.** Analysis ▸ Grasp self-occlusion opens the page with the left arm's
   tabletop grasps chosen (100 grasps, a sample set). Press **Run** (30.9 s from the page on a lightly loaded RTX 5080
   machine, [performance.md](performance.md); 60–62 s in batch mode when its load average was 18). The arm is solved for every
   grasp: 86 reachable, 14 out of reach (listed by reason). The headline reads **100.0 [95.7–100]** % of reachable
   grasps, "well seen": every reachable grasp has every fingertip at least 50 % visible in at least one camera (the
   requirement and its 50 % are labelled defaults; the interval is Wilson's, because every bootstrap resample gives
   100 %). The score is 100 % but the margin is thin: the camera table shows `cam_left_wrist` seeing 61 % and 56 % of
   the two fingertips' contact patches, just above the 50 % line, and the gallery of the weakest grasps starts at
   56 %. Click a grasp in the table or the gallery and the arm takes that pose; Robot ▸ Joint sliders ▸ Home all puts
   it back.
3. **Why.** The camera table splits each camera's contact points by what happens to them. `cam_high (overhead)` sees
   16 % of them: 39 % are behind their own finger and 46 % are hidden by the scene (the other arm, the gripper body).
   `cam_right_wrist` has 55 % outside its image: it looks at the right gripper. So the left wrist camera carries the
   score alone.
4. **Move a camera and see the score change.** Select `cam_left_wrist`, open Sensors ▸ Sensors and mounts and set
   its pitch from −65° to −75° (10° further down, one undo step). The grasp page says the project changed since the
   run; **Run** again: **4.7 [1–9]**, "blind at contact": 4 of 86 grasps meet the requirement, because 43 % of the
   left wrist camera's contact points are now outside its image. Ctrl+Z puts the camera back.
5. **Depth validity.** Analysis ▸ Depth validity ▸ **Run** (the arms at home): "Depth-valid task volume: 100.0 %".
   The bar chart gives, per D405, the first condition each voxel fails (field of view, minimum Z, occlusion, the
   projector cone); the translucent pyramid in front of each camera in the 3D view is its min-Z blind zone.
6. **VLA token coverage.** Analysis ▸ VLA token resolution ▸ **Run**: "no task object", because there is nothing to
   count tokens on. Workspace ▸ Add box object adds Box 1, a 60 mm cube at (450, 0, 30) mm on the table between the
   arms; **Run** again: "Fewest ViT tokens on a critical object: 3 tokens (too few tokens)" for the sample's ACT
   policy, under the 2 × 2 tokens a critical object should span (a labelled default). The page draws the patch grid in
   the Camera view and suggests a stream or a camera move.
7. **Bandwidth.** Analysis ▸ Bandwidth, hosts and power: the four D405s send 1 563.0 Mbit/s of payload, but the
   sample has no host, so the headline reads "—, no hosts". **Apply** adds the suggested host with one USB 3
   controller (an assumption the topology says) and plugs every camera in: the busiest controller is at 46 %, within
   budget, a margin of 1958.8 Mbit/s.
8. **Export the cameras to MuJoCo and LeRobot.** File ▸ Export…, tick MuJoCo MJCF and LeRobot, ACT, Diffusion Policy
   and OpenVLA configs, choose a folder, **Export**: "2 of 2 formats written; 38 files." The folder holds the camera
   and site snippets for the arms' own MJCF and a full MuJoCo scene, the LeRobot dataset features and the policy
   train configs, and `export_manifest.json` with every file's SHA-256 and the project's provenance.
9. **Export a printable bracket.** Analysis ▸ Brackets ▸ **Run**, then **Apply**: `cam_high` and `cam_low` get a 2020
   T-slot camera clamp (a default guess the page says; change it to your profile). The wrist cameras keep no type: the
   page suggests none for them. The headline reads 0/2 ready, "check": each clamp draws the D405's mounting holes at
   the middle of the face, since Intel's drawing does not dimension where they sit, so check them against the camera
   before printing (batch mode counts the same as "Sensors without a ready bracket: 4": the two clamps to check and
   the two wrist cameras without a type). Pick `cam_high (overhead)` and **Export STL…**: a watertight binary STL with a JSON sidecar
   (dimensions, print orientation, provenance). The batch export of the brackets format writes 2 STL files, one per
   clamp.

The same tour in batch mode:

```bash
dexvantage --sample aloha --grasp grasp.json --depth-validity depth.json --vla-tokens vla.json --bandwidth bw.json
dexvantage --sample aloha --export out/aloha --formats mjcf,lerobot
```

## 2. Getting started {#getting-started}

1. **Install.** The [README](../README.md#install) has the bundles and the source install
   (`pip install -e .`, Python 3.10 or newer) with the optional extras (`lerobot`, `bags`, `usd`, `pin`, `step`,
   `rerun`, `assistant`, `research`, `tracking`).
2. **Check the installation.** `dexvantage --self-test` checks, without a window, the project file, frames, MuJoCo,
   OpenCV, fonts, OpenGL, the main window and a quick validation, one line per check. `dexvantage --validate
   out/validation` runs the whole validation suite and writes its HTML report (Help ▸ Validate the models does the
   same from the window). After a plain `pip install -e .` the checks that need an optional extra cannot run: the
   self-test marks their lines as a warning naming the extra, and the validation run leaves them out and says, in one
   sentence, how many and which extra runs them (13 need the `lerobot` extra, one the `test` extra's jsonschema).
3. **Open a sample.** `dexvantage --sample NAME` with `so100`, `ur5e`, `dexhand`, `aloha`, `panda-bimanual` or `g1`,
   or File ▸ Open sample. Each has a [tour](guide/samples.md).
4. **Frames and units.** SI units; joints in radians, shown in degrees; the world is the primary robot's base frame
   (REP-103: x forward, y left, z up); a camera's optical frame has z forward, x right, y down. Help ▸ Coordinate
   frames draws them.

## 3. The window {#window}

1. **Layout.** Left: the Outliner (the project, robots, the active rig variant's sensors, the workspace, tasks) and
   the Library. Centre: the 3D view with the pose scrubber under it. Right: the Inspector and the Camera view; Ctrl+J
   opens the Assistant column. Bottom: Analysis, one tab per page.
2. **Navigation.** Orbit with the middle button or a left drag on empty space (about the point under the cursor);
   Shift pans; the wheel zooms towards the cursor. 7, 1 and 3 look from the top, front and side; 5 toggles
   orthographic; Home frames everything. The control in the top right does the same with a mouse or touchpad.
3. **Commands.** Ctrl+K opens the command palette, Ctrl+F finds an object, `?` lists every shortcut. Every edit is
   one undo step (Ctrl+Z, Ctrl+Shift+Z).
4. **Workspaces.** Window ▸ Placement, Grasp study, Teleop, Calibration, Data audit and Review switch the panel layout;
   Window ▸ Save workspace layout keeps your arrangement, Reset puts the default back. View ▸ Overlays and View ▸
   Theme (dark, light, or the system's) change the 3D view and the colours.
5. **Files.** File ▸ Save writes a `.dexv` file; Save as folder (for Git) writes the unpacked form; Open recent and
   Open sample are in the File menu; File ▸ Render image (F12) writes a high-resolution render; Export ▸ Export the
   project summary writes JSON with its provenance block.

<!-- example: window -->
**Worked example: the dense-voxel limit.** File ▸ New starts an empty project with a default tabletop task volume of
500 × 600 × 300 mm (labelled default). Select the volume in the Outliner and type `1.5` into each Size field (or
`1500 mm`), then Analysis ▸ Run the project summary (F5): at the default 5 mm the volume is 27 × 10⁶ voxels, still
feasible, while a dense 1 mm grid would need 3.4 × 10⁹, above the limit of 5 × 10⁷ (a documented default), so the
studies refine to 1–2 mm only inside refinement regions. Ctrl+Z undoes each change.

## 4. Robots {#robots}

The models: [robots](overview.md#robots), [kinematics](overview.md#kinematics), [IK](overview.md#ik), the
[catalogue](overview.md#catalogue). Longer examples: [guide/robots.md](guide/robots.md) (the catalogue) and
[guide/robot_io.md](guide/robot_io.md) (USD import, URDF inertials).

### Import a robot

1. **Robot ▸ Import robot…** opens the wizard. Drop a URDF, MJCF or USD file on it (USD needs the `usd` extra), or
   Browse…. A preview of the robot at its home pose appears on the right.
2. **File:** the format, the files, their size and the main file's SHA-256; rename the robot here.
3. **Meshes:** every file the model names, found or missing. A missing mesh stops the import and says which file to
   pick. A xacro file must be expanded first (`ros2 run xacro xacro in.urdf.xacro > out.urdf`); MuJoCo reads STL and
   OBJ, not COLLADA, so convert `.dae` meshes.
4. **Scale and units:** the bounding box at home and each mesh's largest edge. A mesh over 10 m across means a file
   in millimetres: the wizard blocks the import and says to set `scale="0.001 0.001 0.001"`.
5. **Joints:** each joint's type, limits, velocity and effort as the file states them ("—" where it states none);
   tick the gripper's joints (names with finger, gripper, jaw or claw are ticked for you).
6. **Gripper and pads:** the gripper frame (what IK drags and the grasp studies aim), the fingertip frames and the
   contact pads (boxes on a link where the fingers touch, sized at first to the link's mesh).
7. **Summary**, then Import: the robot and its files are one undo step. The first robot is the primary one; its base
   is the world frame.

**Robot ▸ Import from MuJoCo Menagerie…** downloads a Menagerie model (one pinned commit), shows that model's own
LICENSE and imports it only after you tick "I accept the licence of this model". The Library's **Robots** group lists
the robot catalogue: each robot's maker figures with their sources, and a double-click imports its Menagerie model.

### Joint sliders, IK and trajectories (the Robot page)

**Robot ▸ Joint sliders** opens the Robot page: a slider and a field per joint, over the file's limits. The headline is
the gripper frame's manipulability w (Yoshikawa's measure) with a word: clear, near a limit (within 2 % of a range, a
default), singular (condition number above 10¹²) or in collision. **Robot ▸ Drag the end effector (IK)** shows a
handle on the gripper frame: drag it and IK follows (green: reachable; red with the solver's reason: out of reach,
joint limit, self-collision…); Esc puts the pose back. **Robot ▸ Import trajectory (CSV)…** reads `t,<joint>,…` rows
in seconds and radians; the pose scrubber under the 3D view plays it.

<!-- example: robot -->
**Worked example on the UR5e (`dexvantage --sample ur5e`).** The robot has 14 joints (six arm joints and the 2F-85's
eight linkage hinges). On the Robot page at home the gripper frame `gripper_base` is at (−134.0, +492.0, +477.2) mm,
roll, pitch, yaw (180, 0, 90)°, w = 0.0820 from the 6 × n Jacobian, condition number 8.3, "clear". Type `0` into
`elbow_joint`: the arm stands straight up, the frame goes to (−134.0, +110.8, +1080.0) mm and the headline says
"singular" (the gripper cannot move along the arm). Home all puts it back.

## 5. Sensors and coverage {#sensors}

The models: [library](overview.md#library), [lens models](overview.md#optics), [occlusion](overview.md#occlusion),
[coverage](overview.md#coverage), [depth validity](overview.md#depth), [the Camera view](overview.md#camview),
[prices](overview.md#prices), [the workspace check](overview.md#workspace). Longer examples:
[guide/sensors.md](guide/sensors.md), [guide/library.md](guide/library.md),
[guide/library_systems.md](guide/library_systems.md), [guide/coverage.md](guide/coverage.md),
[guide/depth.md](guide/depth.md), [guide/workspace.md](guide/workspace.md), [guide/prices.md](guide/prices.md).

1. **The library.** The Library panel lists the hardware by kind (cameras, depth sensors, tactile sensors, hosts,
   lights, cables, policies, teleop devices, robots), each figure from the maker's datasheet or page with the URL and
   the date read. A figure the maker does not state shows "—" with the tooltip "not stated by the maker".
   **Sensors ▸ New custom device…** makes your own entry, every figure ticked "given" or "not given", with its source.
2. **Add and mount.** Double-click a device in the Library (or drag it into the 3D view); Sensors ▸ Add camera on a
   tripod adds a camera with the labelled default lens. **Sensors ▸ Sensors and mounts** opens the Sensors and mounts
   page: per sensor its device, what it is mounted on (World, Cage, Tripod, Table or any link of any robot: it rides
   with the link), its position and roll, pitch, yaw, and its stream mode. Each change is one undo step.
3. **The Camera view** shows the selected sensor's image through its lens model, with the robots posed, the objects
   and the task volumes' edges; the line under it states the model and "offset assumed" where the maker gives no
   imager position.
4. **Coverage and resolution** (Analysis ▸ Coverage and resolution ▸ Run): per 5 mm voxel (a default) and camera, in
   the image, visible past the posed robot, pixels per millimetre, viewing angle and redundancy.
5. **Depth validity** (Analysis ▸ Depth validity ▸ Run): a voxel is valid for a depth camera when both imagers see it,
   it is beyond the minimum Z of the chosen stream and within range, no link shadows it, it is inside the projector
   cone where texture is needed, and no transparent or IR-absorbing object is tagged there.
6. **The Workspace page** (Workspace ▸ Workspace) checks that every task object lies inside a task volume and
   suggests a volume around those outside. Workspace ▸ Add task volume and Add box object add them.

<!-- example: coverage -->
**Worked example on the UR5e.** Coverage ▸ Run: "Task volume seen: 91.9 % (mostly single or blind)". The camera
table: the Wrist D405 sees 46.4 % of the volume at a median 2.13 px/mm, with 39.5 % outside its image; the Overhead
D435 sees 71.9 % at 1.68 px/mm (the arm hides the rest). Depth validity ▸ Run: "Depth-valid task volume: 84.6 %".
Raise the overhead D435 by 10 cm (its z on the Sensors page, or the factor `sensor:Overhead D435.dz` in Python) and
run both again: 92.0 % seen and 87.0 % depth-valid, so the higher mount buys a little coverage and more valid depth.
The rig's cited list prices add up to 700 USD (`rig.cost`, the two RealSense cameras).

## 6. The studies {#studies}

Each study is a page under Analysis (or Tasks, Simulate, Calibrate, Data, Export) with the same pattern: a headline
figure with its status word, **Run**, **Apply** (the suggested change, one undo step), **Preview**, **Explain** (the
model and inputs) and **Export…** (the result as JSON with provenance). Every study has a batch flag (chapter 13).

### Tasks, grasp sets and policies

The **Tasks** page (Tasks ▸ Tasks) samples a grasp set over a task volume (top-down pinch, side, angled, bimanual
handoff; count and seed), previews it as glyphs, and adds it (one undo step); or imports a CSV or JSON of gripper
poses, takes grasps from a trajectory, or from each time the gripper closes in a recorded dataset. The **Policies**
page (Tasks ▸ Policies) names the policy the data is for (OpenVLA, π0, SmolVLA, ACT, Diffusion Policy, GR00T and more,
each figure cited); the Camera view outlines what its input crop keeps. Longer examples:
[guide/grasps.md](guide/grasps.md), [guide/policies.md](guide/policies.md).

### Grasp self-occlusion {#grasp}

The **Grasp self-occlusion** page solves the arm for every grasp, measures how much of each fingertip's contact patch
each camera sees at contact and on the approach, and gives the Contact Visibility Score with its interval, the
cameras, "what to move" (each camera moved 2 cm or turned 5°, one at a time) and a gallery of the weakest grasps.
Model: [overview](overview.md#grasp); longer example: [guide/grasp.md](guide/grasp.md).

<!-- example: grasp -->
**Worked example on the UR5e's 200 tabletop grasps.** Run: 133 of 200 grasps are reachable (64 obstacle collision,
2 out of reach, 1 no convergence), and 133 of 133 meet the requirement: 100.0 %, "well seen". The best single change
the page finds is "move Wrist D405 +2 cm along y of wrist_3_link"; Apply makes it.

### VLA token resolution {#vla}

The **VLA token resolution** page counts the ViT tokens each critical task object and each fingertip covers in every
camera the policy takes, at the pre-grasp and contact poses, warns under 2 × 2 tokens (a default) or when the crop
cuts an object away, and exports Plücker ray maps. Model: [overview](overview.md#vla); longer example:
[guide/vla.md](guide/vla.md).

<!-- example: vla -->
**Worked example on the UR5e's π0.** Run on the sample as it opens: "no task object". Add the soda can of the
lighting example (a cylinder of radius 33 mm and height 115 mm at the task volume's centre) and run again: "Fewest
ViT tokens on a critical object: 1 token (too few tokens)": in the camera that sees it least the can spans about one
patch of π0's input, so move that camera closer or choose a narrower lens.

### Multi-finger occlusion and the tactile handoff {#dexhand}

The **Multi-finger occlusion** page (the dexhand sample: a Shadow Hand on a UR5e with a palm, a dorsal wrist and a head
camera) gives a finger × camera matrix of how much of each fingertip's contact patch each camera does not see over
grasp postures, what hides it, the mounts ranked and a suggested turn; under it the visual-to-tactile handoff chart and
the sensory blind gap. Models: [overview](overview.md#dexhand); longer examples: [guide/dexhand.md](guide/dexhand.md),
[guide/handoff.md](guide/handoff.md).

<!-- example: dexhand -->
**Worked example (`dexvantage --sample dexhand`).** Run: "Fingertips seen: 93.8 %" of the fingertip-cases; the matrix
shows which finger each camera loses and to what (the palm, a neighbouring finger, the thumb, or outside the image).

### Teleop overlap {#teleop}

The **Teleop overlap** page (the Teleop workspace) maps the teleop device (a leader arm or GELLO joint to joint, or a
VR or space-mouse volume you measured) onto the task volume: where the operator is comfortable and the follower
reaches well, and the dead zones (near a singularity, a joint limit or itself, or out of reach). Apply moves the
follower's base by the suggested step; with two arms, the overlap and the handover quality. Model:
[overview](overview.md#teleop); longer example: [guide/teleop.md](guide/teleop.md).

<!-- example: teleop -->
**Worked example on the SO-100 and its leader arm.** Run: "Teleoperable task volume: 96.7 % (clear)".

### Lighting and glare {#lighting}

The **Lighting and glare** page takes the lab's lights (from the library or with your measured figures) and the
objects' materials and gives each camera's share of object and gripper area in specular saturation, a glare map in
the Camera view and, for active-stereo cameras, the depth washed out by ambient IR, with each fix's figure before and
after. Model: [overview](overview.md#lighting); longer example: [guide/lighting.md](guide/lighting.md).

<!-- example: lighting -->
**Worked example on the UR5e.** Add a metallic soda can at the task volume's centre and **Add light ▸ Elgato Key
Light (library)**, which goes 1.2 m above the volume facing down; Run: "Specular saturation (worst camera): 34.9 %
(heavy glare)", relative to an auto-exposure (no maker in the library states a sensor's sensitivity). The
recommendations (a polariser, moving or dimming the light, tilting the camera) each show their figure.

### Motion blur, time sync and action-chunk latency {#timing}

The **Motion blur and rolling shutter** page follows the trajectories and gives the smear in one exposure and the
rolling-shutter skew per camera; **Time sync** gives each stream's timestamp error against the joint states and the
action-label error it causes; **Action-chunk latency** gives the policy's sensor-to-action loop, when the gripper
closes relative to the demonstrated contact and the delay offset to train with. A term no maker states makes a figure
a bound (≤, ≥). Models: [overview](overview.md#timing); longer examples: [guide/motion_blur.md](guide/motion_blur.md),
[guide/time_sync.md](guide/time_sync.md), [guide/action_chunk.md](guide/action_chunk.md).

<!-- example: timing -->
**Worked example on the UR5e's scripted pick.** Motion blur: "Worst motion smear: ≤ 38.0 px (may be blurred)" (no
exposure is set, so the frame period bounds it). Time sync: "Worst camera to joint-state skew: ≥ 54.4 ms (out of
sync)"; Apply sets the D435's hardware trigger. Action-chunk latency for π0: "Worst closing error: 0 mm (on contact)".

### Bandwidth, cables and brackets {#bandwidth}

The **Bandwidth, hosts and power** page gives every stream's rate, the load and margin of each controller and link in
the host topology, disk per episode and bus power. The **Cable flex across joints** page routes cables across joints
and checks slack, bends, pulls and snags, with a harness sheet. The **Brackets** page makes printable brackets
(ISO 9409-1 flange adapters, T-slot clamps, a neck plate, palm and dorsal sleeves) and checks their clearance. Models:
[bandwidth](overview.md#bandwidth), [cables](overview.md#cables), [brackets](overview.md#brackets); longer examples:
[guide/bandwidth.md](guide/bandwidth.md), [guide/cables.md](guide/cables.md), [guide/brackets.md](guide/brackets.md).

<!-- example: hardware -->
**Worked example on the UR5e.** Bandwidth: 2 322.4 Mbit/s of payload and "no hosts"; Apply adds the suggested host:
"Smallest controller margin: 1161.4 Mbit/s (within budget)". Cables: "Cables not clear: 1 (at risk)". Brackets:
"No sensor has a bracket type yet: Apply sets the suggested ones."; after Apply (a flange adapter for the wrist D405,
a 2020 T-slot clamp for the overhead D435) and Run: "Sensors without a ready bracket: 2", both because Intel's
drawings do not place the mounting holes on the face, which the page says to check before printing.

### Canonical view {#canonical}

The **Canonical view** page (Export ▸ Canonical view) warps robot B's wrist image to robot A's viewpoint through the
fingertip contact plane and reports the residual parallax off the plane; Export warp maps… writes the homography and
remap LUT. Model: [overview](overview.md#canonical_warp); longer example:
[guide/canonical_warp.md](guide/canonical_warp.md).

<!-- example: canonical -->
**Worked example on the UR5e (Wrist D405 to Overhead D435).** "Canonical view warp error off the contact plane (P95):
675.1 px (large parallax)": the default depth range is a pessimistic bound; narrow it to the depths your objects take.

### MuJoCo ground truth and synthetic episodes {#groundtruth}

**Simulate ▸ Rendered against predicted visibility…** opens the **Rendered against predicted** page: MuJoCo renders
through every camera's lens model, compared with the grasp study's contact visibility and the depth study's voxels.
Simulate ▸ Build the MuJoCo scene… writes the scene; Simulate ▸ Generate synthetic episodes… writes LeRobot v2.1 or
v3.0 episodes with known faults and their ground truth. Models: [overview](overview.md#groundtruth),
[synthetic episodes](overview.md#synth_episodes); longer examples: [guide/groundtruth.md](guide/groundtruth.md),
[guide/synth_episodes.md](guide/synth_episodes.md).

<!-- example: groundtruth -->
**Worked example on the UR5e.** Run: "Rendered against predicted agreement: 92.8 % (agrees)".

## 7. Calibration {#calibration}

Models: [hand-eye](overview.md#handeye), [silhouette calibration](overview.md#render_calib),
[data links](overview.md#datalinks). Longer examples: [guide/handeye.md](guide/handeye.md),
[guide/render_calib.md](guide/render_calib.md).

1. **Calibrate ▸ Plan hand-eye poses…** opens the **Hand-eye pose plan** page: 20 poses (a default) the arm reaches
   without collision with the board in view, chosen to pin the camera down best, with the predicted 1σ in mm and
   degrees; Print board… writes the ChArUco board as a PDF; Save as session keeps the plan.
2. **Calibrate ▸ Solve hand-eye…** opens the **Hand-eye solve** page: import a folder of board images with a
   `joint_states` file (or take frames from a LeRobot dataset or ROS 2 bag), solve (closed form, then a bundle
   adjustment), read the per-view residuals and the covariance, take the pose the coach asks for next, and Apply the
   calibrated mount.
3. **Calibrate ▸ Silhouette calibration** opens the **Silhouette calibration** page: a static camera's mount from the
   robot's own silhouette in recorded frames, per episode, with its σ and whether it is observable.

<!-- example: handeye -->
**Worked example: hand-eye on the UR5e's wrist D405.** Plan with the defaults: "Predicted camera position σ (worst
axis): 0.0117 mm (meets the requirement)". Render 12 board images at 12 planned poses with 1 grey level of noise
(Simulate a capture…, or `core/calib/synthetic.py`), import them with their joint states: 12 images detected; Solve:
status good, and Apply would move the mount by 0.01 mm and 0.002°, the rendered truth recovered.

<!-- example: silhouette -->
**Worked example: silhouette calibration of the UR5e's overhead D435.** Generate the synthetic episodes with their
defaults (4 episodes, 640 × 360, every fault on, among them a mount slip of the overhead camera from episode 2), then
choose the dataset on the page and Run: "Mount offset found by the silhouettes: 4.21 mm (mount moved)": the slip, found
from the robot's outline alone.

## 8. Data {#data}

Models: [readers, the auditor and plan against recording](overview.md#data), [sim-to-real](overview.md#sim2real).
Longer examples: [guide/datasets.md](guide/datasets.md), [guide/bags_rlds.md](guide/bags_rlds.md),
[guide/audit.md](guide/audit.md), [guide/sim2real.md](guide/sim2real.md).

DexVantage reads LeRobot v2.1 and v3.0 (and writes them), ROS 2 bags (MCAP and SQLite) and RLDS / Open X-Embodiment
TFRecords. **Data ▸ Audit a dataset…** opens the **Dataset audit** page (the Data audit workspace): per episode and
camera the dropped and duplicated frames, frame rate, jitter, USB gaps, camera-joint offset, clock drift, exposure
steps, blur and a hidden gripper, mount slip across the campaign, and a verdict (pass, fix, recalibrate, drop); the
report exports as JSON, HTML and a FiftyOne dataset. **Data ▸ Plan against recording…** opens the **Plan against
recording** page. **Data ▸ Sim-to-real gap** opens the **Sim-to-real gap** page: a recorded episode rendered again
through the rig and compared camera by camera, with the causes ranked and the change that closes each.

<!-- example: data -->
**Worked example on the SO-100.** Simulate ▸ Generate synthetic episodes… with 2 episodes of 48 frames at 160 px wide
(seed 7, every fault on with its default size) writes a Recording LeRobot v3.0 of 2 episodes. Audit it: "Episodes
that pass the audit: 0 % (not usable)", which is the point of a dataset with every fault in it: each episode names
its faults and the fix. Plan against recording: "Plan against recording: comparisons that match: 80 % (mostly
matches)".

<!-- example: sim2real -->
**Worked example: the sim-to-real gap on the UR5e.** `rig.gap_dataset(…)` renders one episode of 24 frames at 320 ×
180 with the overhead D435 made 20 grey levels brighter than the render; the page (or `rig.sim2real`) reads:
"Sim-to-real image gap (worst camera): 18.6 DN (very large gap)", with the causes ranked and the change that closes
the first one offered by Apply.

## 9. Optimisation and requirements {#requirements}

Models: [requirements](overview.md#requirements), [tolerances](overview.md#tolerance),
[designed experiments](overview.md#experiments), [the optimiser](overview.md#optimise). Longer examples:
[guide/requirements.md](guide/requirements.md), [guide/tolerance.md](guide/tolerance.md),
[guide/experiments.md](guide/experiments.md), [guide/optimise.md](guide/optimise.md).

1. **Analysis ▸ Requirements and variants** opens the **Requirements and variants** page (the Review workspace): each
   requirement a named metric, an operator and a threshold; Run gives the scorecard (pass, fail, not evaluated, each
   margin); Compare variants evaluates every requirement on every rig variant. The batch CI gate is `--check`.
2. **Analysis ▸ Tolerances** opens the **Tolerances** page: each mount's ± tolerance (or its calibration covariance),
   draws and seed; each metric's spread and the share of draws meeting each requirement. Export perturbations writes
   the draws for augmentation.
3. **Analysis ▸ Designed experiments** opens the **Designed experiments** page: factors, responses and a design (full
   factorial, fractional, Latin hypercube, Sobol, Saltelli); Sobol indices, main effects and the best run.
4. **Analysis ▸ Optimise** opens the **Optimise** page: improve this rig (mounts within a box and a cone) or design new
   rigs (devices, mount sites, brackets); the Pareto front, each design explained; Accept adds it as a variant.

<!-- example: requirements -->
**Worked example on the UR5e: a requirement and the tolerances.** Add the requirement "Task volume seen" ≥ 90 % (the
metric `coverage.covered`; 90 is this example's figure, not a standard) and Run: "Readiness scorecard: 1 / 1 (ready).
All 1 requirement pass." On the Tolerances page with 16 draws, seed 1 and that one metric (the mounts' default ±1 mm
and ±0.5°): "Monte Carlo tolerances: P5 91.7 % (no requirement). Task volume seen: 91.88 % ± 0.120 % (σ over 16
draws), P5 91.70 %. No requirement names it: add one to judge it."

<!-- example: experiment -->
**Worked example: a designed experiment.** Factors the overhead D435's height (±5 cm) and pitch (±5°), responses
task volume seen and depth-valid volume, a Latin hypercube of 8 runs, seed 2: 8 runs, "better than now": "Best of 8
runs: task volume seen 92 % at run 1, against 91.9 % for the rig as it is. Apply sets that run's factors (one undo
step)."

<!-- example: optimise -->
**Worked example: the optimiser.** Improve this rig with a budget of 12 designs, seed 1: "Optimiser: 3 designs
(better rig found). 1 of the 3 designs is at least as good as the current rig on every objective and better on one;
the others trade one objective for another. No requirement is set, so every design counts as meeting them."

## 10. Export {#export}

Models: [exporters](overview.md#exporters), [formats](overview.md#formats), [Rerun](overview.md#rerun),
[the paper kit](overview.md#paper_kit). Longer examples: [guide/exporters_a.md](guide/exporters_a.md),
[guide/exporters_b.md](guide/exporters_b.md), [guide/rerun.md](guide/rerun.md).

**File ▸ Export…** (Ctrl+E, and Export ▸ Export to MuJoCo, Isaac, ROS 2, LeRobot…): tick formats (MuJoCo MJCF; ROS 2
URDF and tf2; OpenCV YAML and the Kalibr camchain; self-occlusion and depth masks; Plücker ray maps; bracket STL and
STEP; Isaac Lab and Isaac Sim cameras; LeRobot, ACT, Diffusion Policy and OpenVLA configs; the HTML and PDF report;
perturbed extrinsics; the canonical warp; FiftyOne; a synthetic LeRobot dataset; the campaign plan; the cell
comparison; the paper kit), the pose and a folder. The export runs in the background and shows its manifest.
**Export ▸ Stream to Rerun…** sends the scene to an `.rrd` file or a Rerun viewer (the `rerun` extra).

<!-- example: export -->
**Worked example on the UR5e.** Export 7 formats at once (mjcf, urdf, opencv_yaml, masks, plucker, isaac, lerobot):
97 files, each listed in `export_manifest.json` with its SHA-256.

## 11. The assistant and MCP {#assistant}

Models: [the tools and the MCP server](overview.md#assistant), [the dock and the Copilot](overview.md#copilot).
Longer examples: [guide/assistant_dock.md](guide/assistant_dock.md), [guide/mcp.md](guide/mcp.md).

**Window ▸ Assistant** (Ctrl+J) asks Claude (the `assistant` extra) or Groq about the page on screen. Keys are entered
in the dock's settings and kept in the system keyring only. Every number in an answer is checked against what the
tools and the page returned (an unmatched number is shown to you); an edit comes as a proposal you accept as one undo
step; long studies run only from a plan you approve. `python -m dexvantage.mcp` serves the same tools to an MCP client
over standard input and output: with a DexVantage window open it attaches to it, so changes show at once and can be
undone; otherwise it works on `--project cell.dexv` or `--sample NAME`.

<!-- example: mcp -->
**Worked example.** `python -m dexvantage.mcp --sample aloha` (or in process, `dexvantage.mcp.server.Server`):
`initialize` answers protocol 2025-11-25 and `tools/list` lists 40 tools.

## 12. Research workflows {#research}

A research team's day, from a notebook to a paper. Models: [tracker](overview.md#tracking),
[statistics](overview.md#stats), [campaign](overview.md#campaign), [cells](overview.md#cells),
[cluster](overview.md#cluster), [versioning](overview.md#versioning), [paper kit](overview.md#paper_kit). The whole
day on ALOHA 2: [guide/research-workflow.md](guide/research-workflow.md) and the review of it,
[guide/product-review.md](guide/product-review.md); fixes of the review round: [guide/ui-fixes.md](guide/ui-fixes.md).

1. **Keep every run.** **Analysis ▸ Runs** opens the **Runs** page: Apply turns tracking on (one undo step), each
   study run is logged with its settings, figures, seeds and provenance in `<name>.runs/` beside the project; select
   runs and Compare. Batch mode logs with `--track DIR` ([guide/tracking.md](guide/tracking.md)).
2. **On all cores or a cluster.** Run on… ▸ This machine (`--jobs N`) or a Slurm job folder (`--slurm FOLDER`, then
   `--run-shard` and `--gather`) for experiments, tolerance Monte Carlo and optimiser generations; the gathered result
   is the single-process one ([guide/cluster.md](guide/cluster.md)).
3. **Is the difference real?** **Analysis ▸ Statistics** opens the **Statistics** page: success counts, variants over
   their tolerance draws, an experiment's levels or tracker runs, each comparison naming its test and why, and the
   power calculator ([guide/stats.md](guide/stats.md)).
4. **Plan the recording.** **Data ▸ Campaign** opens the **Campaign** page: conditions, the demonstrations a power
   analysis asks for, operators counterbalanced, storage and a calendar ([guide/campaign.md](guide/campaign.md)).
5. **Several cells.** **File ▸ Compare cells…** opens the **Cells** page: other projects, samples or variants beside
   the open one ([guide/cells.md](guide/cells.md)).
6. **What changed.** File ▸ Compare with… lists every change between two projects; File ▸ Save as folder (for Git)
   and `--textconv` make a project diffable; `--merge` merges three ways; `--cite` writes BibTeX; Export figure… on a
   chart writes a publication figure ([guide/versioning.md](guide/versioning.md),
   [guide/publication.md](guide/publication.md)).
7. **Write it up.** **Export ▸ Paper kit…** opens the **Publication** page: the methods paragraph, LaTeX tables,
   figures, BibTeX and a reproducibility appendix, every number traced ([guide/paper_kit.md](guide/paper_kit.md)).
8. **Extend it.** Plugins add studies, metrics, exporters, readers, library folders and pages through the
   `dexvantage.plugins` entry point (Help ▸ Plugins…, `--list-plugins`, `--no-plugins`;
   [guide/plugins.md](guide/plugins.md)).

<!-- example: stats -->
**Worked example: two success rates.** ACT 14 of 20 against Diffusion Policy 19 of 20 (`dv.stats.compare_rates`):
"no clear difference" at this sample size; telling 70 % from 95 % at the default α 0.05 and 80 % power takes 39
episodes per condition (`dv.stats.episodes_needed(0.7, 0.95)`).

<!-- example: campaign -->
**Worked example: the ALOHA 2 handover campaign** of [guide/campaign.md](guide/campaign.md) (three operators, two
lighting presets, demonstrations from a 50 % against 70 % power analysis, 30 s episodes, a 4 TB and ten-day budget):
"Campaign episodes recorded: 216 (within budget)".

<!-- example: cells -->
**Worked example: three cells.** The UR5e sample as the reference with the SO-100 and ALOHA 2 samples
(`dv.cells.compare(rig, 'sample:so100', 'sample:aloha')`): "Largest geometric gap to the reference cell: 0.35 index
(large gap)".

<!-- example: versioning -->
**Worked example: a diff and the citations.** The UR5e sample against a copy with the overhead D435 raised 10 cm: 1
change, "Sensor Overhead D435: mount pose moved 100.0 mm (x +0.0, y +0.0, z +100.0 mm) and turned 0.00°". Its BibTeX
(`--cite refs.bib`, `dv.bibtex(rig)`) holds 236 BibTeX entries: every source the project and the checks depend on.

<!-- example: paper -->
**Worked example: the paper kit of the SO-100.** "ready": 48 of 48 numbers traced. "All 48 numbers in the methods
paragraph and 52 in the tables trace to the project or a result; every citation has its BibTeX entry and the LaTeX is
balanced."

## 13. Batch mode {#batch}

Everything runs without a window: `dexvantage [PROJECT.dexv | --sample NAME] [OPTION …]`. The full list of options,
generated from the parser itself, is the [command-line reference](reference/cli.md); exit codes are 0 done, 1 a
failed check or a result that does not reproduce, 2 an error said in one sentence.

```bash
dexvantage --sample ur5e --analyse -                       # the project summary as JSON on standard output
dexvantage cell.dexv --coverage cov.json                   # any study: its flag writes the result with provenance
dexvantage cell.dexv --reproduce cov.json                  # re-run it and compare every number
dexvantage --sample ur5e --check defaults scorecard.json   # the readiness scorecard as a CI gate
dexvantage --sample ur5e --export out/ --formats mjcf,urdf,opencv_yaml,isaac,lerobot,report
dexvantage --sample so100 --synth-episodes out/synth       # synthetic LeRobot episodes with known faults
dexvantage --sample so100 --audit audit.json --dataset out/synth
dexvantage --sample ur5e --experiment design.json          # a designed experiment (docs/guide/experiments.md)
dexvantage --sample ur5e --optimise opt.json --optimise-settings '{"budget": 24}'
dexvantage --sample aloha --campaign spec.json             # a campaign plan
dexvantage a.dexv --diff b.dexv diff.json                  # what changed (exit 1 when they differ)
dexvantage --validate out/validation                       # the validation suite
dexvantage --self-test                                     # this installation
```

<!-- example: reference -->
The reference lists 106 options (with `--no-plugins`; installed plugins add their own study flags), every study's
flag among them; its companion, the [Python API reference](reference/api.md), documents the 37 names `dv` exports.

## 14. The Python API {#python-api}

`import dexvantage as dv` gives DexVantage without the window and without Qt: open or make a rig, run any study,
read the named metrics, sweep factors, check requirements, export, read datasets, track runs and compare. It runs the
same code as the window and batch mode, so every figure is the study's own. The worked examples are in
[guide/api.md](guide/api.md) and the notebooks in `examples/`; the docstrings of the 37 names `dv` exports are the
[Python API reference](reference/api.md).

```python
import dexvantage as dv
rig = dv.sample('ur5e')                                   # or dv.open('cell.dexv'), dv.new('My cell')
res = rig.run('coverage')                                 # a dict plus .headline, .metrics(), .to_frame()
res.headline['text']                                      # 'Task volume seen: 91.9 % (mostly single or blind)'
taller = rig.with_factors({'sensor:Overhead D435.dz': 0.10})   # a new Rig; rig is unchanged
taller.metrics(['coverage.covered', 'depth.valid']).figures()
rig.check('requirements.yaml'); rig.export('out/', ['opencv_yaml', 'mjcf']); rig.save('cell.dexv')
```

## 15. Troubleshooting {#troubleshooting}

| What you see | Why, and what to do |
|---|---|
| The import wizard refuses a `.xacro` file, or a `.urdf` with `$(…)` or `${…}` | xacro macros must be expanded first: `ros2 run xacro xacro in.urdf.xacro > out.urdf`. |
| The wizard blocks a mesh over 10 m across | the meshes are in millimetres: set `scale="0.001 0.001 0.001"` on them in the model file. |
| A COLLADA (`.dae`) mesh is not read | MuJoCo reads STL and OBJ: convert the mesh and point the model at the converted file. |
| `dexvantage file.urdf` says it is a robot model file | start DexVantage and import it with Robot ▸ Import robot…; the command line takes `.dexv` projects. |
| A figure shows "—" with "not stated by the maker" | the datasheet does not state it; DexVantage never guesses. Enter your measured figure with its source (a custom device, or the field's own override). |
| A study refuses in one sentence (for example Action-chunk latency on the SO-100: SmolVLA's inference time is not stated by its source) | the input it needs is missing and the sentence says which; enter it where it says (the Policies page here). |
| "no hosts" on the bandwidth page | the rig has no host topology: Apply adds an assumed host, or add your own from the library. |
| "no task object" on the VLA page | add a task object (Workspace ▸ Add box object); every task object counts as critical by default. |
| Studies are slow | the GPU path needs Linux EGL and OpenGL 3.3; elsewhere the exact numpy rasteriser runs (the same figures within the occlusion model's agreement, slower). |
| The 3D view is blank or flickers | start with `--software-3d` (QPainter instead of OpenGL). It is slow, and slower the more triangles the robots have: 160 ms (SO-100) to 586 ms (UR5e) a frame in Phase 2 ([models/performance.md](models/performance.md)); on 2026-11-08, in a 1600 × 1000 window on a busy machine (load average about 30), 0.19 s (SO-100, 60 682 triangles), 0.73 s (UR5e), 0.91 s (UR5e + Shadow Hand), 0.93 s (G1), 1.1 s (two Pandas) and 1.4 s (ALOHA 2, 331 392 triangles) a median frame. The view lags while you drag; the studies still run on their own threads. |
| The status bar says "OpenGL is not available here" | DexVantage could not create an OpenGL 2.1 context (a virtual machine, a remote X session, or a missing GPU driver), so it uses the QPainter renderer above. `dexvantage --self-test` says on its `opengl` line what it tried; install the GPU's OpenGL driver (or Mesa) to get the fast view back. |
| A plugin breaks start-up | `dexvantage --no-plugins`, then Help ▸ Plugins… or `--list-plugins` shows which plugin failed and why. |
| The paper kit's LaTeX compile is skipped | no LaTeX engine is on PATH; the `.tex` files are written and checked anyway. |
| The Menagerie list is short and says it is offline | the list falls back to the folder names; connect to download a model. |
| The Assistant asks for a key | keys live in the system keyring only (the `assistant` extra installs `keyring`). |
| Something failed with "details in the log" | the window logs the traceback and shows a sentence; run the same step in batch mode for its exit code and the sentence on standard error. |
