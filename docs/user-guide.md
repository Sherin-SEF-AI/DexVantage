# DexVantage user guide

A step-by-step guide with a worked example for every feature. Each phase adds its chapter.

## 1. The window (Phase 0)

1. Run `dexvantage`. An empty project opens with a default tabletop task volume (500 × 600 × 300 mm, labelled
   default).
2. **Left:** the Outliner (project, robots, sensors of the active rig variant, workspace, tasks) and the Library (it
   fills in Phase 1 and 2). **Centre:** the 3D view with the pose scrubber under it. **Right:** the Inspector and the
   Camera view; Ctrl+J opens the Assistant column. **Bottom:** Analysis.
3. Orbit with the middle button or a left drag on empty space (about the point under the cursor); Shift pans; the
   wheel zooms towards the cursor. 7, 1 and 3 look from the top, the front and the side; 5 toggles orthographic;
   Home frames everything. The control in the top right does the same with a mouse or touchpad only.
4. **Worked example.** Sensors ▸ Add camera on a tripod adds a camera looking at the task volume. Select the task
   volume in the Outliner and type `1.5` into each Size field (or `1500 mm`): the Inspector warns that a dense 1 mm grid
   would need 2.7 × 10⁹ voxels, and the Analysis summary updates. Ctrl+Z undoes each change as one step.
5. Ctrl+K opens the command palette; Ctrl+F finds an object; `?` lists every shortcut.
6. Window ▸ workspaces switch the panel layout (Placement, Grasp study, Teleop, Calibration, Data audit, Review);
   Window ▸ Save workspace layout keeps your arrangement, Reset puts the default back.
7. File ▸ Save writes a `.dexv` file; File ▸ Render image (F12) writes a 2× render; Export ▸ Export the project
   summary writes JSON with its provenance block.
8. Help ▸ Coordinate frames shows the world and optical frames; Help ▸ Validate the models runs the validation suite
   and opens its report.

## 2. Batch mode

```bash
dexvantage cell.dexv --analyse summary.json        # the summary, with provenance
dexvantage --sample ur5e --analyse -               # a bundled sample's summary as JSON on standard output
dexvantage cell.dexv --reproduce summary.json      # re-run and compare: "Reproduces exactly …"
dexvantage --validate report/                      # validation report (exit 0 pass, 1 fail, 2 error)
dexvantage --self-test                             # check this installation
```

Each study has its own flag writing its result as JSON with provenance ("-" for standard output). On the UR5e sample
every one of them runs as the sample opens:

```bash
dexvantage --sample ur5e --grasp grasp.json              # grasp self-occlusion (200 grasps, about 30 s on a GPU)
dexvantage --sample ur5e --vla-tokens vla.json           # VLA token resolution for the sample's π0
dexvantage --sample dexhand --dex-occlusion dex.json     # the multi-finger occlusion matrix
dexvantage --sample ur5e --bandwidth bw.json             # bandwidth, hosts and power
dexvantage --sample ur5e --cables cables.json            # cable flex (suggested routes when there are no cables)
dexvantage --sample ur5e --brackets brackets.json        # brackets and their clearance
dexvantage --sample ur5e --handeye-plan plan.json        # a hand-eye calibration pose plan
dexvantage cell.dexv --handeye-solve solve.json          # solve a recorded hand-eye session of the project
dexvantage --sample ur5e --groundtruth gt.json           # rendered against predicted visibility and depth (MuJoCo)
dexvantage --sample so100 --teleop teleop.json           # teleop overlap and dead zones (the sample's leader arm)
dexvantage cell.dexv --lighting light.json               # glare and IR wash-out (needs lights in the project)
dexvantage --sample ur5e --motion-blur blur.json         # smear and rolling-shutter skew along the scripted pick
dexvantage --sample ur5e --time-sync sync.json           # stamp skew and the action-label error
dexvantage --sample ur5e --action-chunk chunk.json       # the policy's loop latency, overshoot and delay offset
dexvantage --sample so100 --synth-episodes out/synth     # synthetic LeRobot episodes with known faults
dexvantage --sample so100 --audit audit.json --dataset out/synth              # audit a recorded dataset
dexvantage --sample so100 --plan-vs-recording pvr.json --dataset out/synth    # the plan against the recording
dexvantage --sample so100 --silhouette-calib sil.json --dataset out/synth     # a static camera's mount from frames
```

`--synth-settings` takes the generator's settings as JSON text or a file, e.g.
`--synth-settings '{"episodes": 4, "frames": 120, "version": "v2.1"}'`. A study that cannot work with its inputs says
why in one sentence and exits 2 (for example `--action-chunk` on the SO-100 sample: SmolVLA's inference time is not
stated by its source, so enter a measured figure on the Policies page).

Phases 9 and 10 and the research tools add (each measured on the samples on 2026-11-07; times include start-up):

```bash
dexvantage --sample ur5e --tolerance-mc tol.json          # Monte Carlo mount tolerances (64 draws, about 80 s)
dexvantage --sample ur5e --experiment design.json         # a designed experiment (docs/models/experiments.md)
dexvantage --sample ur5e --optimise opt.json --optimise-settings '{"budget": 24}'      # improve this rig
dexvantage --sample ur5e --design-agent da.json --optimise-settings '{"budget": 24}'   # design new rigs
dexvantage --sample ur5e --check defaults scorecard.json  # the readiness scorecard (exit 0 pass, 1 fail, 2 error)
dexvantage --sample ur5e --check requirements.yaml        # your own requirements (docs/examples/requirements.yaml)
dexvantage --sample ur5e --reproduce scorecard.json       # any result re-run from its provenance
dexvantage --sample ur5e --run-plan plan.json out/        # studies, checks, exports and reproductions in order
dexvantage --sample ur5e --export out/ --formats mjcf,urdf,opencv_yaml,isaac,lerobot,report   # exporters
dexvantage --sample ur5e --canonical-view canon.json      # the cross-embodiment canonical view warp
dexvantage --sample ur5e --rerun scene.rrd                # the scene for Rerun (optional extra rerun)
dexvantage --sample aloha --coverage cov.json             # the new samples: aloha, panda-bimanual, g1
dexvantage a.dexv --diff b.dexv diff.json --diff-html diff.html   # what changed (exit 1 when they differ)
dexvantage cell.dexv --unpack cell.dexv.d                 # a Git-friendly folder; --pack writes it back
dexvantage --textconv cell.dexv                           # a stable text for git diff
dexvantage cell.dexv --cite refs.bib                      # BibTeX of every source the project depends on
dexvantage --figure spec.json fig.pdf --figure-width double      # a publication figure from a chart's data
python -m dexvantage.mcp cell.dexv                        # the MCP server on a project file
```

## 3. Robots (Phase 1) {#robots}

The models behind this chapter are in [models/robots.md](models/robots.md), [models/kinematics.md](models/kinematics.md)
and [models/ik.md](models/ik.md).

### Import a robot

1. Robot ▸ Import robot… opens the import wizard. Drop a URDF or MJCF file on it, or press Browse…. A preview of the
   robot at its home pose appears on the right.
2. **File:** the format, the number of files, their size and the main file's SHA-256. Rename the robot here.
3. **Meshes:** every file the model names, found or missing. A missing mesh stops the import and says which file to
   pick; a missing texture does not. A xacro file (or a .urdf that still holds xacro macros or `$(…)` / `${…}`
   expressions) must be expanded to URDF first (`ros2 run xacro xacro in.urdf.xacro > out.urdf`); USD is not read
   yet. MuJoCo reads STL and OBJ meshes but not COLLADA (`.dae`, common in ROS packages): convert those to STL or OBJ
   and point the URDF at the converted files.
4. **Scale and units:** the bounding box at home in millimetres and each mesh's largest edge. A mesh over 10 m
   across means the file is in millimetres: the wizard blocks the import and says to set
   `scale="0.001 0.001 0.001"`. A robot under 20 mm or over 5 m across gets a warning (a plausibility default).
5. **Joints:** each joint's type and the file's limits, velocity and effort ("—" where the file states none). Tick
   the gripper's joints; names with finger, gripper, jaw or claw are ticked for you.
6. **Gripper and pads:** the gripper frame (what IK drags and the grasp studies aim; the wizard's guess never picks
   a finger link), the fingertip frames, and contact pads: boxes on a link where the fingers touch, sized at first to
   the link's mesh.
7. **Summary**, then Import. The robot and its files are one undo step (Ctrl+Z removes both). The first robot in a
   project is the primary one and its base is the world frame.

Select a robot in the Outliner to see its file, licence, joints and tags in the Inspector. Edit ▸ Delete selection
removes it with its stored files and trajectories (one undo step); while a sensor is mounted on one of its links, the
delete is refused and says which sensor to move or delete first.

Robot ▸ Import from MuJoCo Menagerie… lists Menagerie's robots, downloads the one you pick (one pinned commit),
shows that model's own LICENSE and imports it only after you tick "I accept the licence of this model". Licences
differ per model; the robot records the licence, the commit and the date you accepted, and the project keeps the
LICENSE text. Offline, the list falls back to the folder names and says so.

### Joint sliders and the read-outs

Robot ▸ Joint sliders opens the Robot tab of the Analysis dock: a slider and a number field per joint, in degrees
(revolute) or millimetres (prismatic), over the file's limits. The 3D view, the link-mounted cameras and the read-outs
follow as you drag. Type a value into a field; a value past a limit is clipped to it. Home puts one
robot back at its home pose, Home all every robot.

The headline is the gripper frame's manipulability w (Yoshikawa's measure: 0 at a singularity; in m³ from the
position rows when fewer than six joints move the frame) with a status word:

| Word | Meaning | What to do |
|---|---|---|
| clear | every joint that moves the frame is inside its limits and no two links touch | — |
| near a limit | a joint that moves the frame is within 2 % of its range (a default) of a limit | move it away, or the arm cannot follow a motion in that direction |
| singular | the condition number is above 10¹² (the Jacobian has lost rank to double precision) | bend a joint away: a straight arm, or two wrist axes in line |
| in collision | two links touch (MuJoCo's signed distance) | change the pose |

Below the sliders: the frame's position (mm) and roll, pitch, yaw (°) in the world frame, w, the condition number,
and self-collision with the closest pair of links and their distance.

### Drag the end effector (IK)

1. Robot ▸ Drag the end effector (IK) turns the mode on: the gripper frame of the selected (else the primary) robot
   shows a handle.
2. Drag the handle. The target moves on the plane through the handle that faces the camera, and inverse kinematics
   follows, choosing the solution nearest the current pose so the arm does not jump between elbow-up and elbow-down.
3. A reachable target turns the handle green ("reachable"). An unreachable one turns it red and the status bar gives
   the solver's reason (out of reach, joint limit, orientation unreachable, self-collision, no convergence); the arm
   stays at the last reachable pose.
4. Esc during a drag puts the pose back where the drag started. The pose is view state, not an undo step.

The drag moves the frame's position and leaves its orientation free.

### Trajectories and the pose scrubber

Robot ▸ Import trajectory (CSV)… reads q(t): a header row `t,<joint>,<joint>,…` (seconds, then joint names as in the
model file), one row per sample, times increasing, radians or metres. Joints the file does not name keep their
current value. Values that look like degrees, or leave the joint limits, are warned about; the import is one undo
step. Columns may be separated by commas, semicolons or tabs; a semicolon or tab file may write decimal commas
(`0,25`). The pose scrubber under the 3D view then scrubs the trajectory: drag along it, play it at real time,
or step between samples; the arm follows by linear interpolation. Trajectories are listed under Tasks in the
Outliner: select one to play it in the scrubber, Delete removes it.

```csv
t,shoulder_pan_joint,shoulder_lift_joint,elbow_joint
0.0,-1.5708,-1.5708,1.5708
1.0,-1.2,-1.4,1.3
```

### Sample tour: SO-100 (`dexvantage --sample so100`)

1. **Open.** The SO-ARM100 (Menagerie `trs_so_arm100`, Apache-2.0) stands at the file's home keyframe, reaching
   along the base's −y axis, with a wrist camera on the fixed jaw, a third-person camera on a tripod and a
   300 × 200 × 120 mm tabletop volume. Both camera placements are sample placements, not published figures.
2. **Read the headline figures.** The 3D view's caption reads "1 robot · 6 joints · 2 sensors"; the Summary tab
   57 600 voxels at 5 mm, dense grid feasible. Robot ▸ Joint sliders: the gripper frame `Fixed_Jaw` is at (0.0, −238.8, +176.6) mm,
   w = 0.00407 m³ from the position rows (five arm joints cannot set all six pose coordinates), condition number
   2.1, status "clear"; the closest links are the wrist and the moving jaw, 12.1 mm apart.
3. **Change one thing.** Type `0` into the Elbow field. The jaw rises to (0.0, −158.9, +366.5) mm, w drops to
   0.00195 m³ and the condition number rises to 5.2: the arm is closer to straight and moves less freely. The wrist
   camera's frustum rides with the jaw; the tripod camera stays put. Home all puts it back.

### Sample tour: UR5e + Robotiq 2F-85 (`dexvantage --sample ur5e`)

1. **Open.** The UR5e (BSD-3-Clause) with the 2F-85 gripper (BSD-2-Clause) on its flange, from MuJoCo Menagerie, at
   the file's home keyframe. "Wrist D405" is a depth camera on the flange looking at the pinch point; "Overhead D435"
   looks down from 0.85 m above the 500 × 400 × 300 mm tabletop volume. Both take their lenses, depth FOV and
   minimum Z from the library's Intel entries (chapter 4).
2. **Read the headline figures.** The caption reads "1 robot · 14 joints · 2 sensors" (6 arm joints, 8 gripper
   linkage hinges); the Summary tab 480 000 voxels at 5 mm, and a dense 1 mm grid would need 6 × 10⁷ voxels, marked
   not feasible.
   Joint sliders: `gripper_base` at (−134.0, +492.0, +477.2) mm, roll, pitch, yaw (180, 0, 90)°, w = 0.0820 from the
   6 × n Jacobian, condition number 8.3, "clear". The closest pair is two links of the gripper's linkage, 3.6 mm
   apart (they move independently here; see the known limits).
3. **Change one thing.** Type `0` into elbow_joint: the arm stands straight up, the frame goes to
   (−134.0, +110.8, +1080.0) mm, w falls to about 10⁻²², the condition number reads ∞ and the headline says "singular": the
   gripper cannot move along the arm. Type `90` again (or Home all) and it is "clear". With Robot ▸ Drag the end
   effector (IK) on, drag the gripper's handle far beyond the arm: it turns red with "out of reach".

**Known limits of the samples:** the 2F-85's linkage joints are independent entries of q, so moving one alone poses
the linkage in a way the real gripper cannot. The SO-100's cameras have no library device (no maker in the library
states a camera for it), so they use the labelled default camera until you choose one (chapter 4).

## 4. Sensors and coverage (Phase 2) {#sensors}

The models behind this chapter: [models/library.md](models/library.md), [models/optics.md](models/optics.md),
[models/occlusion.md](models/occlusion.md), [models/voxels.md](models/voxels.md),
[models/coverage.md](models/coverage.md), [models/depth.md](models/depth.md) and
[models/camview.md](models/camview.md). Longer worked examples: [guide/sensors.md](guide/sensors.md),
[guide/coverage.md](guide/coverage.md) and [guide/depth.md](guide/depth.md).

### The library

The **Library** panel (left) lists the hardware library by kind: 9 cameras, 21 depth sensors, 2 tactile sensors and
8 hosts in this build, each from its maker's datasheet or product page. Hover over a device for its figures and its
sources (URL, title, the date read). A figure the maker does not state shows "—" with the tooltip "not stated by the
maker"; nothing is filled in. Where an entry needs an assumption (the D435's RGB imager position, which Intel does
not give, or a stream whose FOV is derived by a centred crop) the figure is marked and the entry's note says why.

**Sensors ▸ New custom device…** (also on the Sensors page and in the Library panel's right-click menu) opens a form in which every figure is ticked "given" or
left "not given"; a source and the date you read it are required. *Start from a library device…* copies an entry
for you to change (a unit you measured, say). Custom devices are saved in your own library folder, never in the
built-in one.

### Adding sensors and mounting them on links

1. **Add a device:** double-click it in the Library (or right-click ▸ Add to the rig), or drag it into the 3D view:
   it is placed above the floor point where you drop it. Sensors ▸ Add camera on a tripod adds a camera with no device (the labelled default camera).
2. **Sensors ▸ Sensors and mounts** opens the Sensors page: one row per sensor (device, what it is mounted on,
   position in mm, stream, in use). Select a row to edit it; every change is one undo step.
3. **Mounted on:** World, Cage, Tripod, Table (static) or any link of any robot. Choose `UR5e + Robotiq 2F-85 ▸
   wrist_3_link` and the sensor rides with that link at every joint state; with *Keep its place* ticked it stays
   where it is and its pose is recomputed in the link's frame.
4. **Position** and **Roll, pitch, yaw** are in mm and degrees in the mount's frame; drag a field's axis letter to scrub (Shift fine,
   Ctrl coarse).
   **Stream** picks one of the device's stream modes (default: the largest listed); the lens model follows it.

### The Camera view

The **Camera view** (right) shows the selected sensor's image through its imager's lens model, posed with the robot:
the robots, task objects, tables and floor, with the task volumes as dashed edges where the image sees them. The
combo above the image picks the imager (RGB, IR left, IR right). The line under it says the model
(`RGB · 1280 × 720 px · 84° × 53°`) and "offset assumed" when the imager's position in the device is not stated.
While you drag a joint the image is drawn at half size and sharpens when you let go; it never blocks the window.
The 3D view draws each sensor's frustum from the same lens models.

### Coverage and resolution

**Analysis ▸ Coverage and resolution**, then **Run**. Every task volume is cut into 5 mm voxels (2 mm around the
objects; both defaults) and for every voxel and camera the study asks: is it in the image, is it hidden by the robot
at the pose in the 3D view (a cube map around the camera), how many pixels per millimetre, at what angle, how
blurred (only where the lens's focus and f-number are stated or entered). The headline is the share of the volume
seen by two or more cameras, with a status from your thresholds (good from 90 %, warning from 60 %, serious from 30 %:
defaults). The camera table says how much each camera sees, misses (outside its image) and has hidden; the slice
(x, y or z, with a slider) colours the voxels in the 3D view. Batch: `dexvantage cell.dexv --coverage cov.json`.

### Depth validity

**Analysis ▸ Depth validity**, then **Run**. For each depth sensor a voxel is depth-valid when both stereo imagers
see it, it is beyond the minimum Z for the chosen resolution (Intel's table) and within the stated range, no link
hides it from either imager (finger shadows, stereo parallax), it is inside the projector cone where the sensor needs
texture, and no transparent, IR-absorbing or glossy object is tagged there (a warning). The bar chart gives the
first condition each voxel fails. **Resolution** changes the depth stream (and so the minimum Z); the expected depth
noise σ_z = z² σ_d / (f B) is shown beside the maker's own RMS statement, with σ_d = 0.08 px labelled as a default.
The translucent pyramid in front of each depth camera in the 3D view is its min-Z blind zone; it moves with the
wrist. Batch: `dexvantage cell.dexv --depth-validity dv.json`.

### Sample tour: SO-100 (`dexvantage --sample so100`)

1. **Open.** Two cameras, both "(sample placement)": one on a bracket 60 mm behind the jaws on the wrist side,
   looking down past the jaw tips, and one on a tripod. Neither has a device, so the Camera view's imager combo reads
   "Default camera (no device chosen)" and the Inspector's Model row "no device chosen".
2. **Coverage.** Run it: **47.3 %** of the 300 × 200 × 120 mm volume is seen by two or more cameras ("gaps"), 95 %
   by at least one, 5 % by none (57 600 voxels). The wrist camera sees 54 % at 4.7 px/mm (35 % is outside its image,
   11 % hidden by the jaws); the tripod camera sees 88 % at 1.2 px/mm, and the arm hides the other 12 %.
3. **Change one thing.** On the Sensors page set the wrist camera's **Device** to *e-con Systems See3CAM_24CUG*
   (104.6° × 61.6°, as e-con states for 1920 × 1200) and run again: two or more cameras now see **53.7 %**; the wrist
   camera sees 61 % at 6.8 px/mm.
4. **Depth.** Set the wrist camera's device to *Intel RealSense Depth Camera D405* and run Depth validity:
   **35.9 %** of the volume has valid depth. 44.9 % is outside the field of view of one of its two imagers, 6.0 % is
   nearer than its 100 mm minimum Z at 1280 × 720 and 13.2 % is occluded (7.6 % hidden from one imager only). The
   page suggests 480 × 270, whose minimum Z is 45 mm: about 5 % more of the volume. Ctrl+Z puts the sample back.

### Sample tour: UR5e + Robotiq 2F-85 (`dexvantage --sample ur5e`)

1. **Open.** Select *Wrist D405*: the Inspector shows Intel's figures (stated FOV 90° × 64° × 98°, modelled at the
   stream 84.0° × 58.0°, global shutter, pixel size "—": not stated by the maker). The Camera view looks down at the
   two fingers; pick *IR right* to see the other half of the 18 mm stereo pair.
2. **Coverage.** Run it: **26.4 %** of the volume is seen by both cameras ("mostly single or blind"), 92 % by at
   least one (480 000 voxels). The overhead D435 sees 72 % at 1.7 px/mm and the arm hides 28 %; the wrist D405 sees
   46 % at 2.1 px/mm, and 40 % of the volume is outside its image.
3. **Depth validity.** Run it: **84.6 %** of the volume has valid depth from at least one camera ("good"): the wrist
   D405 45.8 %, the overhead D435 61.3 %. For the D405 36.6 % is outside a field of view and 17.6 % is occluded,
   7.2 % of it from one imager only. Both fingertip pads are 114 mm from it, beyond its 100 mm minimum Z.
4. **Change one thing.** Move the elbow on the Robot page: the D405's frustum, its min-Z pyramid and the Camera view
   follow the wrist; press Run again to see the coverage at the new pose (the study uses the pose in the 3D view).

## 5. Grasps, policies and dexterous hands (Phase 3)

The models behind this chapter: [models/grasps.md](models/grasps.md), [models/grasp.md](models/grasp.md),
[models/policies.md](models/policies.md), [models/vla.md](models/vla.md) and [models/dexhand.md](models/dexhand.md).

### Grasp sets (Tasks ▸ Tasks) {#grasps}

A grasp set is the input of the grasp studies: gripper poses at contact, each with its approach direction. On the
**Tasks** page sample one over a task volume (top-down pinch, side grasp, angled, bimanual handoff in a handover
volume; count and seed), **Preview** it as glyphs in the 3D view and the Camera view, then **Add set** (one undo
step); or **Import file…** (CSV or JSON of gripper poses), take one from a trajectory (a grasp wherever the gripper
closes), or **From a dataset…** (each time the gripper closes in a recorded LeRobot, RLDS or ROS 2 episode; the
recording's joints are matched to the model by name, then by order). The grasp centre and the gripper's axes are
measured from the contact pads, not typed in. The UR5e and SO-100 samples come with a set. Worked example:
[guide/grasps.md](guide/grasps.md); the dataset link: [models/datalinks.md](models/datalinks.md).

### Grasp self-occlusion (Analysis ▸ Grasp self-occlusion) {#grasp}

Can the cameras see the fingertips at the moment the fingers close? **Run** solves the arm for every grasp (infeasible
grasps are counted by reason), measures which share of each fingertip's contact patch every camera sees, and gives
the **Contact Visibility Score**: the share of reachable grasps where your requirement holds (default: one camera sees
every fingertip at 50 % or more), with its bootstrap 95 % interval and Wilson's interval. The page lists the cameras,
"what to move" (each camera moved 2 cm or turned 5°, one at a time; **Apply** makes the best move, one undo step), a
gallery of the worst grasps as Camera view renders (click one to pose the robot there) and every grasp. Sets above
2 000 grasps run in the background with partial results. On the UR5e sample: 133 of 200 grasps reachable, CVS 100 %
(Wilson 97.2–100 %); the Overhead D435 alone 34.6 % [26.3–42.9]. Worked example: [guide/grasp.md](guide/grasp.md).

### Policies (Tasks ▸ Policies) {#policies}

Name the policy your data is for: **Add policy** picks one of eight cited entries (OpenVLA 7B, π0, π0-FAST, SmolVLA,
ACT, Diffusion Policy, Octo, GR00T N1). Each figure shows its source and quote; a figure no source states is "—" and
can be replaced by your own with its source. Tick the cameras it takes. The Camera view outlines what part of the image
the policy's input keeps. Worked example: [guide/policies.md](guide/policies.md).

### VLA token resolution (Analysis ▸ VLA token resolution) {#vla}

How many ViT tokens does each task object and fingertip cover in each camera the policy takes, at the pre-grasp and
contact poses? The page warns below 2 × 2 tokens (a default) or when the crop cuts an object away, draws the patch
grid and the crop in the Camera view and the patch footprints on the objects in 3D, suggests a change, and exports
Plücker ray maps (`.npy` or `.safetensors`) at the policy's input resolution. Worked example: [guide/vla.md](guide/vla.md).

### Multi-finger occlusion (Analysis ▸ Multi-finger occlusion) {#dexhand}

For a dexterous hand (`dexvantage --sample dexhand`: a Shadow Hand on a UR5e with a palm micro camera, a dorsal
wrist camera and a head camera): a finger × camera matrix of how much of each fingertip's contact patch the camera does
not see over power, precision and tripod grasp sweeps (or a grasp set), what hides it, the mounts ranked, and a
suggested turn of a camera. Below the matrix, the **visual-to-tactile handoff** chart follows each fingertip's contact
patch from 50 mm out to contact: how much of it some camera sees, and how much a tactile pad covers at contact; the
**sensory blind gap** is the distance over which a contact is neither seen nor touched. Worked examples:
[guide/dexhand.md](guide/dexhand.md) and [guide/handoff.md](guide/handoff.md).

## 6. Hardware engineering (Phase 6)

The models behind this chapter: [models/bandwidth.md](models/bandwidth.md), [models/cables.md](models/cables.md) and
[models/brackets.md](models/brackets.md).

### Bandwidth, hosts and power (Analysis ▸ Bandwidth, hosts and power) {#bandwidth}

Every stream's data rate from its stated mode (width × height × fps × bits per pixel, plus a labelled 5 % protocol
overhead); the host topology (add hosts, controllers and hubs, plug each sensor in) with the load and margin of each
controller and link; disk per episode and campaign; USB bus power. On the UR5e sample **Apply** adds a host with one
USB 3 controller (an assumption, labelled) and plugs both cameras in: 68 % of its capacity. A stream whose maker states
no bits per pixel has no rate until you enter a measured one. Worked example: [guide/bandwidth.md](guide/bandwidth.md).

### Cable flex across joints (Analysis ▸ Cable flex across joints) {#cables}

Route a cable between attachment points on links (or Apply the suggested routes, one per link-mounted sensor): for
each joint crossed, the slack it needs over the joint's range and along trajectories, the tightest bend against the
cable's stated minimum, pulls and snags; a harness sheet (JSON, CSV, HTML) with cable, length, slack loops, connectors
and strain relief. Worked example: [guide/cables.md](guide/cables.md).

### Brackets (Analysis ▸ Brackets) {#brackets}

Printable, watertight STL brackets: wrist-flange adapters on ISO 9409-1 patterns, 2020 and 4040 T-slot clamps, a
pan-tilt neck plate, palm and dorsal micro-camera sleeves, with the camera's stated mounting holes. **Apply** sets the
suggested bracket per sensor; **Run** meshes them and checks clearance against the robot over sampled joint states;
**Export STL…** writes the STL with a JSON sidecar (dimensions, print orientation, provenance). Worked example:
[guide/brackets.md](guide/brackets.md).

## 7. Hand-eye calibration (Phase 7) {#handeye}

The model: [models/handeye.md](models/handeye.md). **Calibrate ▸ Plan hand-eye poses…** picks 20 poses (a default)
that the arm reaches without collision with the board in view, chosen to pin the camera down best, and predicts the
1σ of the result in mm and degrees; **Save as session** keeps the plan, **Next pose** steps the robot through it, and
**Print board…** writes the ChArUco or AprilGrid board as a PDF. **Calibrate ▸ Solve hand-eye…** imports a folder of
board images with a `joint_states` file, detects the corners, solves (closed form, then one bundle adjustment; optional
joint zero offsets and intrinsics), shows per-view residuals, outliers and the covariance, says which pose to take next
(the coach), and **Apply** writes the calibrated mount (one undo step). **From a LeRobot dataset or ROS 2 bag…**
takes the frames with the board in view and the joints at their times from a recording; **Simulate a capture…**
renders board images at the planned poses with a known offset that the solve should recover. A 5-joint arm such as
the SO-100 is planned position-first; several sessions can be solved in one bundle adjustment. Worked example:
[guide/handeye.md](guide/handeye.md).

### Silhouette calibration (Calibrate ▸ Silhouette calibration) {#render_calib}

No board: a static camera's mount from the robot's own silhouette in recorded frames. Choose the dataset and the
camera, **Run**: per episode the mount's offset from the project's (with its 1σ and whether every direction is
observable), the steps between episodes (a bracket knocked), and **Apply** writes the recovered mount (one undo
step). The Camera view draws the robot through the recovered mount. It needs the robot to move and to look like its
model, and the frames' timing to be right (audit it first). Worked example: [guide/render_calib.md](guide/render_calib.md).

## 8. MuJoCo ground truth (Phase 4) {#groundtruth}

The models: [models/groundtruth.md](models/groundtruth.md) and [models/synth_episodes.md](models/synth_episodes.md).
**Simulate ▸ Rendered against predicted visibility…** renders the scene in MuJoCo through every camera's lens model
and compares it with the predictions: the grasp study's contact points (a confusion matrix, the worst disagreements
side by side, why they disagree: at an edge, the rule, occlusion) and the depth study's valid voxels. On the UR5e
sample 92.8 % of 20 992 contact points agree over 41 poses, "agrees". **Simulate ▸ Build the MuJoCo scene…** writes the scene as a
folder MuJoCo loads. **Simulate ▸ Generate synthetic episodes…** writes LeRobot v2.1 or v3.0 episodes rendered from
the project with known faults (dropped frames, desync, blur, mount slip…) and their ground truth beside them, to test
the auditor or a training pipeline. Worked examples: [guide/groundtruth.md](guide/groundtruth.md) and
[guide/synth_episodes.md](guide/synth_episodes.md).

## 9. Teleop, lighting, blur and sync (Phase 5)

### Teleop overlap (Analysis ▸ Teleop overlap, the Teleop workspace) {#teleop}

Add the teleop device (a leader arm or GELLO, a VR headset or a space mouse with the volume you measured): the page
shows where in the task volume the operator is comfortable and the follower reaches well, and the **dead zones**
where the operator is comfortable but the follower is near a singularity, a joint limit or itself, or out of reach
(click one to pose the follower there). **Apply** moves the follower's base by the suggested step. The SO-100 sample
has its leader arm: 96.7 % teleoperable, "clear". Model: [models/teleop.md](models/teleop.md); worked example:
[guide/teleop.md](guide/teleop.md).

### Lighting and glare (Analysis ▸ Lighting and glare) {#lighting}

Add the lab's lights (from the library or with the figures you measured, each with its source) and the objects'
materials; **Run** gives each camera's share of the object and gripper in specular saturation (relative to an
auto-exposure unless you enter the sensor's sensitivity), a glare map in the Camera view, and for active-stereo depth
cameras the share of depth the ambient IR washes out. Each recommendation (a polariser, moving or dimming a light,
tilting the camera, blinding a window) shows the figure before and after; **Apply** makes it. Model:
[models/lighting.md](models/lighting.md); worked example: [guide/lighting.md](guide/lighting.md).

### Motion blur and time sync (Analysis ▸ Motion blur and rolling shutter, Analysis ▸ Time sync) {#timing}

Along the project's trajectories: how far the image moves during one exposure (smear) and across a rolling-shutter
readout (skew) per camera, the longest exposure that keeps the smear to 1 px (**Apply** records those exposures), and
the smear vectors in the Camera view. Time sync gives each stream's timestamp error against the joint states by clock
and trigger, and the action-label error it causes; **Apply** sets the suggested hardware trigger. A figure the maker
does not state makes the result a bound, said as "at most" or "≥". Models: [models/motion_blur.md](models/motion_blur.md),
[models/time_sync.md](models/time_sync.md); worked examples: [guide/motion_blur.md](guide/motion_blur.md),
[guide/time_sync.md](guide/time_sync.md).

### Action-chunk latency (Analysis ▸ Action-chunk latency) {#action_chunk}

For the project's policy (ACT, Diffusion Policy, π0…): the sensor-to-action loop per camera term by term with each
term's source, when the gripper closes relative to the demonstrated contact (overshoot or air-grasp), an overshoot
map over the task volume, and the delay offset and temporal-ensemble weight to train with; **Apply** stores them on
the policy. Model: [models/action_chunk.md](models/action_chunk.md); worked example:
[guide/action_chunk.md](guide/action_chunk.md).

## 10. Data (Phase 8) {#data}

The models: [models/datasets.md](models/datasets.md), [models/audit.md](models/audit.md) and
[models/datalinks.md](models/datalinks.md). DexVantage reads LeRobot v2.1 and v3.0 (and writes them), ROS 2 bags
(MCAP and SQLite) and RLDS / Open X-Embodiment TFRecords.

- **Data ▸ Audit a dataset…** (the Data audit workspace): add a dataset folder or bag, **Run**: per episode and camera
  the dropped and duplicated frames, frame rate, jitter, USB gaps, camera-joint offset, clock drift, exposure steps,
  blur and a hidden gripper, mount slip across the campaign, and a verdict (pass, fix, recalibrate, drop) against your
  requirements (defaults labelled). **Export report…** writes JSON and HTML; **Export to FiftyOne…** writes a dataset
  with per-frame tags.
- **Data ▸ Plan against recording…**: each recorded camera against the plan (resolution, frame rate, mount, sync,
  what the grasp study predicted the camera would see); the Camera view draws the recorded frame's outline.

Worked examples: [guide/datasets.md](guide/datasets.md), [guide/bags_rlds.md](guide/bags_rlds.md) and
[guide/audit.md](guide/audit.md).

## 11. Requirements, tolerances, experiments and the optimiser (Phase 9) {#requirements}

The models: [models/requirements.md](models/requirements.md), [models/tolerance.md](models/tolerance.md),
[models/experiments.md](models/experiments.md) and [models/optimise.md](models/optimise.md).

- **Analysis ▸ Requirements and variants** (the Review workspace): list requirements as a metric, an operator and a
  threshold (each edit one undo step), or load a YAML file; **Run** gives the scorecard (pass, fail, not evaluated,
  with each margin; a bound shown as "≥" or "≤" decides only in its direction). **Compare variants** evaluates every
  requirement on every rig variant. The same scorecard is the batch CI gate `--check`.
- **Analysis ▸ Tolerances**: each mount's ± tolerance (or its calibration covariance), the number of draws and the
  seed; the result is each metric's spread, the share of draws that meet each requirement with its interval, and the
  mount components that drive it. **Export perturbations** writes the draws for augmentation or Isaac randomisation.
- **Analysis ▸ Designed experiments**: add factors (a mount coordinate, a device, a stream, a study setting), choose
  responses and a design (full factorial, fractional, Latin hypercube, Sobol, Saltelli); the result has the runs, the
  Sobol indices as bars with their intervals, main effects and the best run; **Apply** sets it as one undo step.
- **Analysis ▸ Optimise**: improve this rig (mounts within a box and a cone) or design new rigs (devices, mount sites
  and brackets) over the objectives you choose, with requirements as constraints; the Pareto front, each design
  explained; **Accept** adds it as a variant.

Worked examples: [guide/requirements.md](guide/requirements.md), [guide/tolerance.md](guide/tolerance.md),
[guide/experiments.md](guide/experiments.md) and [guide/optimise.md](guide/optimise.md).

## 12. Export, the assistant and MCP (Phase 10) {#export}

The models: [models/exporters_a.md](models/exporters_a.md), [models/exporters_b.md](models/exporters_b.md),
[models/canonical_warp.md](models/canonical_warp.md), [models/rerun.md](models/rerun.md),
[models/mcp.md](models/mcp.md) and [models/assistant_dock.md](models/assistant_dock.md).

- **File ▸ Export…** (and the Export menu): tick formats (MuJoCo, ROS 2 URDF and tf, OpenCV and Kalibr calibration,
  masks, Plücker maps, brackets, Isaac Lab and Isaac Sim, LeRobot and policy configs, the HTML and PDF report,
  perturbations, the canonical warp, FiftyOne, a synthetic LeRobot dataset), the pose (home or the one on screen) and
  a folder; the export runs in the background and shows its manifest (every file's SHA-256 and the provenance). The
  report reuses the results already computed in the window and, by default, runs the studies not yet computed.
- **Export ▸ Canonical view**: robot B's wrist image warped to robot A's viewpoint through the contact plane, with the
  residual parallax off the plane; **Export warp maps…** writes the homography and remap LUT.
- **Export ▸ Stream to Rerun…**: the scene to an `.rrd` file or a new Rerun viewer (`pip install "dexvantage[rerun]"`).
- **Window ▸ Assistant** (Ctrl+J): ask about the page on screen. Keys are entered in the dock's settings and kept in
  the system keyring only. Numbers the assistant cannot trace to a tool or the page are shown to you; edits come as
  proposals you accept (one undo step); long studies run only from a plan you approve.
- **MCP:** `python -m dexvantage.mcp [cell.dexv]` serves the same tools to an MCP client (Claude Desktop, Claude Code);
  with the window open it attaches to it, so changes show at once and can be undone.

Worked examples: [guide/exporters_a.md](guide/exporters_a.md), [guide/exporters_b.md](guide/exporters_b.md),
[guide/canonical_warp.md](guide/canonical_warp.md), [guide/rerun.md](guide/rerun.md), [guide/mcp.md](guide/mcp.md)
and [guide/assistant_dock.md](guide/assistant_dock.md).

## 13. Samples, the robot catalogue and the larger library {#samples}

- **File ▸ Open sample**: ALOHA 2 (`--sample aloha`), two Franka Pandas (`--sample panda-bimanual`) and the Unitree G1
  upper body (`--sample g1`) join the SO-100, UR5e and dexhand samples ([guide/samples.md](guide/samples.md)).
- **Workspace ▸ Workspace**: whether each task object lies inside a task volume, with a one-step suggestion
  ([guide/workspace.md](guide/workspace.md)).
- **Library ▸ Robots**: 53 robots with the maker's figures and their sources; activating one imports its Menagerie
  model with that model's licence shown ([guide/robots.md](guide/robots.md)).
- The library holds 138 hardware entries: cameras, depth cameras and tactile sensors
  ([guide/library.md](guide/library.md)); Jetson hosts, capture cards, policies, lights, cables and the new teleop
  devices, which the Teleop page adds straight from the library ([guide/library_systems.md](guide/library_systems.md)).

## 14. Research tools (part 1) {#research}

- **File ▸ Compare with…**: every change between the open project and another file, with pose deltas in mm and
  degrees and, when both have results, metric deltas; export as JSON or HTML. `--merge BASE THEIRS OUT` merges three
  ways ([guide/versioning.md](guide/versioning.md)).
- **File ▸ Save as folder (for Git)…** and **Open folder…**: the project as one JSON file per section, with
  `--textconv` for `git diff`.
- **Export figure…** (right-click a chart): a publication figure, one or two columns, SVG or PDF; `--cite` writes
  BibTeX for the project's sources ([guide/publication.md](guide/publication.md)).
- **Python:** `import dexvantage as dv; rig = dv.sample('ur5e'); rig.run('coverage').headline`
  ([guide/api.md](guide/api.md)); plugins add studies, metrics, exporters, readers, library folders and pages
  (Help ▸ Plugins…, [guide/plugins.md](guide/plugins.md)).

## 15. Research workflows (research tools part 2) {#research2}

A research team's day, from a notebook to a paper, on the bundled samples. Each step links its worked example; the
models and their validation are in the overview ([tracker](overview.md#tracking), [statistics](overview.md#stats),
[campaign](overview.md#campaign), [sim-to-real](overview.md#sim2real), [paper kit](overview.md#paper_kit),
[cluster](overview.md#cluster), [cells](overview.md#cells), [robot import and export](overview.md#robot_io)).

1. **Start in a notebook and keep every run.** Save the project, turn tracking on and run studies; each run is logged
   with its settings, figures, seeds and provenance in `<name>.runs/` beside the file:

   ```python
   import dexvantage as dv
   rig = dv.sample('so100'); rig.save('so100.dexv'); rig = dv.open('so100.dexv')
   rig.factors['setting:tracking.enabled'] = True
   rig.run('coverage')                                   # logged
   dv.runs(rig)                                          # a DataFrame of the runs, units in .attrs
   ```

   In the window the same is **Analysis ▸ Runs** (Apply turns tracking on, one undo step): select two runs and
   **Compare** for the figures side by side and what differs. Batch mode logs with `--track RUNS_DIR`
   ([guide/tracking.md](guide/tracking.md)).
2. **Ask a designed question, on all cores or a cluster.** A designed experiment or a tolerance Monte Carlo runs the
   same studies many times: **Run on… ▸ This machine** (or `--jobs N`) spreads them over processes, **Write a Slurm
   job folder** (or `--slurm FOLDER`, then `--gather`) over a cluster; the result is the one a single process gives,
   and it is logged like any other run ([guide/cluster.md](guide/cluster.md)).
3. **Is the difference real?** **Analysis ▸ Statistics** compares success counts (Fisher's exact test, the risk
   difference with Newcombe's interval, corrected across comparisons), two variants over their tolerance draws, an
   experiment's factor levels or tracker runs, and says which test ran and why. A difference that is not clear comes
   with the episodes or draws that would settle it, and the power calculator draws power against n
   ([guide/stats.md](guide/stats.md)).
4. **Plan the recording.** **Data ▸ Campaign** turns the conditions (tasks, variants, lighting, objects, teleop
   devices), the demonstrations a power analysis asks for, the operators and the budget into a counterbalanced
   schedule with storage, a calendar (`.ics`) and checklists (`--campaign SPEC.json`,
   [guide/campaign.md](guide/campaign.md)).
5. **After recording: how far is simulation from the real frames?** **Data ▸ Sim-to-real gap** renders a recorded
   episode again through the rig and compares it camera by camera (exposure, blur, noise, silhouette overlap, mount
   offset, latency), ranks the causes and offers the change that closes each one (`--sim2real OUT.json --dataset
   PATH`, [guide/sim2real.md](guide/sim2real.md)).
6. **Several cells.** **File ▸ Compare cells…** puts other labs' projects, samples or variants beside the open one:
   figures of merit, the geometric gap of each camera to the reference, whether their recordings pool for training
   and how the built copies drift (`--cells A.dexv B.dexv …`, [guide/cells.md](guide/cells.md)).
7. **Write it up.** **Export ▸ Paper kit…** writes the methods paragraph, LaTeX tables (with the statistical
   comparisons, the cell comparison and the tracked runs when they are present), figures, BibTeX and a
   reproducibility appendix with a run plan, every number traced to where it came from (`--export OUT --formats
   paper_kit`, [guide/paper_kit.md](guide/paper_kit.md)).

Also new: **Robot ▸ Import robot…** reads USD robots (`pip install "dexvantage[usd]"`) and the URDF export carries
inertials MuJoCo loads ([guide/robot_io.md](guide/robot_io.md)); 81 library entries carry cited prices, so `rig.cost`
is a figure on the samples ([guide/prices.md](guide/prices.md)); hidden pages wait until they are shown, so an edit
costs a few milliseconds on the UI thread ([guide/ui-fixes.md](guide/ui-fixes.md)).
