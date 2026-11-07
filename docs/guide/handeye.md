# Hand-eye calibration: a worked example

Where exactly is the wrist camera on the flange? This walk-through calibrates the UR5e sample's wrist camera (the
D405's RGB imager on `wrist_3_link`) against a printed ChArUco board on the table, eye-in-hand: plan the poses, record
them, solve, apply, with the coach saying what to take next. The model, its equations and its checks are in
`docs/models/handeye.md` (the planner, the recording and the coach under "Pose planning, recording and the coach").

The images in the numbers below are rendered (`core/calib/synthetic.py`) so that the true answer is known; on a real
rig they come from the camera and the joint vectors from the robot at the moment each image is taken.

## A. Plan the poses (Calibrate ▸ Plan hand-eye poses…)

Open the sample (File ▸ Open sample ▸ "UR5e and Robotiq 2F-85 with a wrist D405 and an overhead D435") and switch
to the Calibration workspace (Window ▸ Calibration workspace): the Analysis dock opens on **Hand-eye pose plan**.

1. **Camera**: "Wrist D405 (on wrist_3_link)", eye-in-hand. (Choosing the fixed "Overhead D435" plans eye-to-hand,
   with the board held on the gripper.)
2. **Board and where it is**: until you save one, the page uses DexVantage's default and says so: a ChArUco board of
   9 × 7 squares of 30 mm (22 mm markers, `DICT_5X5_100`) flat on the table at the task volume's centre, printed face up.
   Enter the board you print and where you lay it (the fields are the board frame's origin, a corner of the pattern,
   and its roll, pitch and yaw), then **Save board** (one undo step). **Print board…** writes a PDF at true size: print
   it at 100 % ("actual size"), glue it flat, measure a square with a ruler and enter what you measured.
3. **Poses** (default 20), **Corner noise** (default 0.3 px) and the **Requirement** (default 0.5 mm and 0.05°) are
   yours to set; each field says "default" until you change it.
4. **Run**. On the sample (default board and settings), 304 of the 375 candidate poses are feasible (63 come within
   10 mm of the table, 5 are out of reach, 2 make the arm touch itself, 1 did not converge), and the plan of 20 poses
   predicts for the camera

   | | σ x | σ y | σ z | σ rx | σ ry | σ rz |
   |---|---|---|---|---|---|---|
   | X: wrist camera in `wrist_3_link` | 0.0117 mm | 0.0100 mm | 0.0099 mm | 0.0028° | 0.0029° | 0.0088° |
   | Z: board in the world | 0.0173 mm | 0.0212 mm | 0.0165 mm | 0.0060° | 0.0049° | 0.0088° |

   The headline reads 0.00882° (the rotation about the optical axis, the loosest) and "meets the requirement". The
   pose table lists the 20 poses in the order to take them (each "0.19 m from the board, tilted 50° (towards 90°)" and
   so on), how many corners each sees, how many the gripper's fingers hide, and how much the loosest σ grows without
   it (0.5–5.9 %); one more pose would cut the loosest axis by 3.5 %. The 3D view draws the planned gripper frames,
   the path between them and a line from each camera position to the board; the Camera view outlines the board.
5. **Next pose** moves the robot in the view through the poses (view only). **Save as session** stores the plan as a
   calibration session (one undo step) so you can record it pose by pose.

These σ are predictions at 0.3 px of independent corner noise: real detected corners are somewhat correlated, so treat
them as a lower bound (2 to 2.5 times lower than the error measured on rendered images).

## B. Record

Drive the robot to each planned pose (the session's joint vectors, or the pose table), let it settle, and take an
image. Save the images in one folder and the robot's joint states as a CSV with a time column `t` (seconds) and one
column per joint (radians; the first row of the example below, with the gripper's joints at 0 after the dots):

    t,shoulder_pan_joint,shoulder_lift_joint,elbow_joint,wrist_1_joint,wrist_2_joint,wrist_3_joint,...
    100.004,-1.5707874644,-1.0634923917,1.6950665100,-1.8402133477,-1.5707808154,0.0000013386,...

The image times come from an `images.csv` next to the images (columns `file`, `t`), or from the file names
(`pose_00_100.000.png`: the last decimal number, in seconds).

**From a recording.** If the calibration was recorded as a LeRobot dataset (v2.1 or v3.0) or a ROS 2 bag (MCAP or
SQLite), use **From a LeRobot dataset or ROS 2 bag…** on the Hand-eye solve page instead. Hold the robot still at each
pose for a second or so while recording. The dialog asks for the dataset, the camera to calibrate, its stream
(empty: the stream named like the camera, `observation.images.wrist_d405` or `/wrist_d405/image_raw` for "Wrist
D405"), the episodes, every Nth frame, the joint time offset (default 0 ms: the camera and the joints on one clock)
and the joints' unit. On a worker thread it then
- reads each frame's joint state by interpolating the recorded rows linearly at the frame's time (+ the offset);
- leaves out frames taken while the joints moved faster than 0.5°/s (a default; tick **Use frames taken while the
  robot moves** to keep them, marked) and frames outside the joint rows' time span;
- looks for the board in the remaining frames; a stream recorded at a smaller size than the sensor's (the same lens,
  resampled) is handled by scaling the camera model and carrying the corners back to the sensor's pixels;
- makes one view of each held pose (still frames whose joints agree within 0.05°, a default; the frame with the most
  corners), and adds the session in one undo step.
The note under the buttons says how many frames were looked at, in how many the board was found and how many views
that gave, then the timing: "Joint states interpolated linearly at each frame's time. The camera and joint times are
taken as one clock (offset 0.0 ms, a default). The nearest joint row is 7.0 ms from a frame (median)."

Worked example on the UR5e sample (the test `tests/test_dataset_links.py`, measured 2026-11-07): 10 planned poses held
1 s each and joined by 0.8 s moves, recorded at 5 fps through the wrist D405 moved 2.69 mm and 0.62° from its mount,
at 640 × 360 px (half the stream), as a LeRobot v3.0 episode with AV1 video: 90 frames looked at, the board found in
the 52 still ones, 10 views (0.19 s on the CPU). The solve finds the camera 0.05 mm and 0.011° from where the frames
were rendered (the solve's own 1σ: 0.014 mm and 0.007°; the same frames uncoded give 0.049 mm, and full-size frames 0.020 mm:
rendered corners err alike within a view, which the σ does not assume). The same
capture as an MCAP bag (joint states at 50 Hz, images stamped 7 ms after a joint row) gives the same 10 views, and
says the 7 ms.

**No rig yet?** On the Hand-eye solve page, **Simulate a capture…** renders the board at the plan's poses with the
camera moved a known 2.69 mm and 0.62° from its mount (defaults), saves the images, `joint_states.csv` and `truth.json`
in a new `simulated-capture-<time>` folder inside the folder you choose, and imports them as a session named
"(simulated)". Solve it as below: on the UR5e sample the headline says the camera moves 2.69 mm and 0.619°, and the
notes say the session is simulated and how far the solve is from the truth. It is a test of the pipeline, not a
measurement: Apply on it moves the camera in the project to the simulated mount (Edit ▸ Undo takes it back). The
SO-100 (five arm joints) plans and simulates the same way; its plan notes that each pose is reached position first.

## C. Solve (Calibrate ▸ Solve hand-eye…)

1. **Images folder…** and **Joint states…**, **Match within** (default 50 ms), **Joints in** rad or deg, then
   **Import**: each image takes the nearest joint sample, the board is detected, and the session is saved (one undo
   step). The note says how many images were matched and detected; an image with no joint sample within the tolerance,
   or taken while the robot moved, is listed with the reason.
2. **Run** solves the session. On the 20 rendered views of the plan above: 20 of 20 views used, RMS 0.127 px (per view
   0.04–0.19 px), corner σ̂ 0.090 px; the camera's σ 0.0032–0.0038 mm and 0.0008–0.0021°, and its error against the
   truth 0.0077 mm and 0.0020°. (The solve used every corner of the rendered boards, while the plan had left out the
   corners the fingers would hide, so its σ is a little below the plan's prediction at the same corner noise.)
3. The **Coach** names the next pose and why: "Enough: every axis of the camera is within 0.5 mm and 0.05°. The loosest
   axis now is the camera's rotation about its z axis at 0.0021° …". With fewer poses taken it names the pose that
   most reduces the loosest σ, and **Show next pose** moves the robot there in the view.
4. **Apply** writes the calibrated mount to the sensor (its body pose on `wrist_3_link`; the change, here 0.008 mm, is
   in the status line) and marks the session applied: one undo step. With **Refine the intrinsics** ticked the refined
   lens model is written as the sensor's calibration too.

5. **Two cameras at once.** Import a second session for the other camera (choose it on the plan page first, or as
   the session's sensor) and tick **Every session together**, then **Run**: one bundle adjustment over both. On the
   UR5e sample, with the wrist D405 seeing the table board at 10 planned poses and the overhead D435 seeing the same
   board at the same poses (`validation/checks_handeye.joint_sessions_project`, the overhead mount moved 10 mm on
   purpose), the covariance table lists both cameras and the board; the overhead camera, which alone is refused ("every
   camera and board is fixed in the world"), comes back about 0.8 mm from its true mount (its σ about 1 mm across the
   image), and **Apply** writes both mounts in one undo step. The notes say why the coach cannot help the overhead
   camera.

Batch: `dexvantage --sample ur5e --handeye-plan plan.json` writes the plan; `--handeye-solve out.json` solves the
project's newest session (or, with the study settings `folder` and `joints`, captures a folder at run time).

## D. The same with the Python API

The steps below are what the pages do, with `dexvantage.core.calib`.

### 1. Make and print a board

```python
from dexvantage.core.calib import boards
board = boards.CharucoBoard(9, 7, square_m=0.030, marker_m=0.022)     # DICT_5X5_100, a default
boards.save_board(board, 'charuco-9x7-30mm.pdf')                       # true scale, one-square margin
boards.save_board(board, 'charuco-9x7-30mm.png', dpi=300)
```

Print the PDF at 100 % ("actual size"), glue it to something flat, and measure a square: if it is 29.8 mm, define
the board with `square_m=0.0298` (and the marker scaled alike). An AprilGrid in Kalibr's layout is
`boards.AprilGrid(6, 6, tag_m=0.088, spacing=0.3)`; keep its corner squares (the default) for the best corners.

### 2. Detect the board in each image

```python
from dexvantage.core.calib import detect
det = detect.detect(image, board, camera)          # camera: the core.optics.CameraModel of the unit
pnp = detect.solve_pnp(det.points, det.corners, camera)
print(det.n, pnp.rms_px)
```

On the 20 rendered views of this example (640 × 480, f = 480 px, 1 grey level of noise) the board was found in
every view, with 8 to 48 of its 48 corners (in some poses part of the board is out of the image; ChArUco corners
count one by one, so a partial board still gives a pose).

From a recorded dataset, `capture.capture_dataset` does the detection and the joint reading for every frame and
returns a session; `capture.session_problem` turns it into the problem below:

```python
from dexvantage.core.calib import capture
sess = capture.capture_dataset(project, 'path/to/lerobot_dataset', sensor_uid, episodes=None, every_n=1)
problem, info = capture.session_problem(project, sess)
```

### 3. Solve

```python
from dexvantage.core.calib import handeye as HE
problem = HE.Problem(
    cameras=[HE.CameraSpec('wrist', camera, robot='ur5e', link='wrist_3_link')],
    boards=[HE.BoardSpec('board', board.points(), board=board)],          # static: in the world
    robots={'ur5e': model},                 # core.robot.model.get_model(project, robot_uid)
    joints=[{'ur5e': q} for q in joint_readings],
    observations=[HE.Observation('wrist', 'board', p, det.ids, det.corners) for p, det in enumerate(dets)])
result = HE.solve(problem)
print(result.message)
print(result.to_dict()['cameras']['wrist'])
```

`synthetic.scene('eye_in_hand', ...)` builds the same problem with corners projected from the truth instead of
detected; the numbers below are from rendered and detected images (scene seed 7, the sample's home pose ± 0.35 rad
on the six arm joints):

| | Result | σ | Error against the truth |
|---|---|---|---|
| Views used | 20 of 20, RMS 0.078 px (per view 0.04–0.12 px), corner σ̂ 0.055 px | | |
| Wrist camera in `wrist_3_link` | (0.0, 130.8, 70.0) mm | 0.009–0.011 mm, 0.001° | 0.023 mm, 0.0013° |
| Board in the world | (−266.7, 461.5, 65.0) mm | 0.007–0.014 mm, 0.001–0.004° | 0.025 mm, 0.0083° |

The closed-form initialisers on the same views (pixel RMS of each start): Tsai-Lenz 0.247 px (chosen), Park-Martin
0.256, Horaud-Dornaika 0.255, Andreff 0.255, Daniilidis 0.412, Shah 0.338, Li 0.336. The adjustment brings the RMS to
0.078 px.

The errors are about 2.5 times the predicted σ: corners detected in rendered images are not perfectly independent
from one corner to the next, while the covariance assumes they are. With real images, treat σ as a lower bound and
look at the per-view RMS (`result.views`) for views to drop or retake.

### 4. Options

- **Eye-to-hand:** mount the camera in the world (`robot=''`) and the board on the link that holds it
  (`BoardSpec(..., robot='ur5e', link='wrist_3_link')`); X is then the camera in the world, Z the board on the link.
- **Several cameras, several arms:** list them all in one problem; each view names its camera, board and pose.
- **Intrinsics:** `CameraSpec(..., intrinsics=True)` refines fx, fy, cx, cy and the distortion terms of the camera's
  lens model; give 20 or more views spread over the image.
- **Joint offsets:** `Settings(joint_offsets=('ur5e',))`. The result marks each offset "determined" or not; with a
  board on the table and a camera on the last link, the first and last joints' offsets cannot be told from the
  board's yaw and the camera's roll and stay at their prior (1°, a default).
- **Encoder noise:** `Settings(encoder_sigma_deg=0.02)` treats the joint readings as measurements with that σ.
- **Outliers:** views over max(3 × the median view RMS, 1 px) are dropped and the problem solved again (defaults;
  `Settings.outlier_k`, `outlier_floor_px`).

## E. Check the installation

    dexvantage --validate out/validation

runs the hand-eye checks with the rest (`handeye.*`, among them `handeye.planner_vs_ba`, the planner's predicted
covariance against the bundle adjustment's); the report gives each check's measured error and tolerance.
