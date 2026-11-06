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
   looks down from 0.85 m above the 500 × 400 × 300 mm tabletop volume. The camera models' datasheet figures arrive
   with the Phase 2 library.
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
the linkage in a way the real gripper cannot; the cameras have no lens or depth model until Phase 2.
