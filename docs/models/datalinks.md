# Data links: grasp sets and hand-eye captures from recorded datasets

Phase 8 (spec §7.3 "taken from a recorded dataset", §8.1 "recording and solving: ROS 2 topics, a LeRobot episode").
Two features read a recorded dataset through the shared reader layer (`core/datasets`: LeRobot v2.1 and v3.0, ROS 2
bags in MCAP or SQLite, RLDS) and turn it into a project record:

| What | Function | Page |
|---|---|---|
| A grasp set: one grasp each time the gripper closes in an episode | `core/grasps.from_dataset` | Tasks ▸ From a dataset… |
| A hand-eye session: the frames with the board in view, with the joints at their times | `core/calib/capture.capture_dataset` | Hand-eye solve ▸ From a LeRobot dataset or ROS 2 bag… |

Both run on a window job (a worker thread) from the pages, and both explain a dataset they cannot use in a plain
sentence (a `GraspError` or `CaptureError`, never a traceback). Neither stores a full path: a record names the
dataset's folder or file only.

## The recorded joints on the robot's model (`core/datasets/joints.py`)

`on_model(episode, model, unit)` puts an episode's state vector (`Episode.q`, `Episode.joint_names`) into the model's
joint order:

1. the same name;
2. the same name after normalising both (lower case; a trailing `.pos`, `_pos`, `.position`, `_position`, `_joint`
   or `.joint` and a leading `main_` or `follower_` removed; `-`, space and `.` read as `_`), when unambiguous. This
   matches LeRobot names such as `shoulder_pan.pos` or `main_shoulder_pan` to `shoulder_pan_joint`;
3. by order, only when no name matches and the counts are equal (RLDS `state_0` …), said in the notes.

A model joint the dataset lacks is held at the model's home; unused dataset columns are listed. When fewer than half
of the model's joints are found and the dataset names more other joints than were found, the recording is refused
as another robot's. Unit `deg` converts revolute joints (prismatic joints stay in metres); a revolute column beyond
a full turn (2π + 0.01 rad) while the joint's limits are not is reported as "looks like degrees", never converted on
its own.

`interpolate(t, Q, x)` is linear between rows:

    q(x) = Q[k] (1 − w) + Q[k+1] w,   w = (x − t[k]) / (t[k+1] − t[k]),   t[k] ≤ x ≤ t[k+1]

NaN outside [t₀, t_N] and next to a NaN row (a bag joint not yet published). A time within 10⁻³ of the median row
spacing (at least 1 µs) outside the span takes the end row: LeRobot stores timestamps as float32, whose rounding at
20 s is about 2 µs.

## Grasps from a dataset (`core/grasps.from_dataset`)

Each episode is read as a trajectory: the joints on the model, clipped to the limits, rows with a NaN joint left out
(counted), then the rules of `from_trajectory` at the episode's rows (`docs/models/grasps.md`): the gripper opening
from the pads' centroids (`pad_opening`), closing strokes of 5 mm or more (`closure_events`, CLOSE_MM, a default),
the gripper frame at the stroke's onset (the last row before the fingers move) and the approach over the last 20 mm
(APPROACH_MM, a default). Mode `every` takes every Nth row. Pose ids name the episode (`e3c1`: episode 3, its first
closing); `t` is the time in the episode; the settings list each episode's rows, strokes and opening range, the
format and version, the joint-matching rule and the unit. The poses' kind is `trajectory` (a recorded trajectory);
the set's source is `dataset`.

## Hand-eye capture from a dataset (`core/calib/capture.capture_dataset`)

For the sensor's camera stream (the key given; else the only camera; else the key ending with, or holding, the
sensor's name in lower case without its parenthesised part), for every Nth frame i of each episode:

- **Joint reading.** q_i = q(t_i + δ) by the interpolation above, t_i the frame's stored time (LeRobot's frame
  timestamp, a bag's header stamp carried onto the joint clock by the reader), δ the joint time offset (default 0 s:
  one clock, a default). The pose record keeps t_image = t_i, t_joints = t_i + δ and dt_s = t_row − t_i for the
  nearest joint row (the timestamp offset, said per episode as a median and as the camera's first time minus the
  joints' first time).
- **Still or moving.** The speed is the fastest joint's |ΔQ| / Δt over the row interval holding t_i + δ (and the one
  before it at a row's own time). Above MOVING_DEG_S (0.5°/s, a default) the frame is moving: left out by default
  (the solve assumes still poses, and a constant clock offset only shows in moving frames), or kept and marked.
- **Detection.** The board is detected at the frame's size. A frame of W' × H' from a camera model of W × H with the
  same aspect ratio (within 1 %) is the same lens resampled: the model is scaled (`CameraModel.scaled`: f' = f s,
  c' = s (c + ½) − ½) for the detector and the corners are carried back to the model's pixels,
  u = (W / W') (u' + ½) − ½ (OpenCV pixel centres). Another shape is refused (a crop cannot be undone).
- **One view per held pose.** Consecutive still frames of an episode whose joints all lie within STILL_SAME_DEG
  (0.05°, a default) of the run's first frame are one view: the frame with the most corners, the middle one among
  equals. A robot holding a pose for 2 s at 30 fps then gives one view, not 60 views whose errors are correlated (the
  solver's σ treats views as independent). At most MAX_VIEWS (500, a default) views, evenly spread.
- **The session.** Only frames with the board found become poses (`episode E frame I`); the source records the
  dataset, the camera key, per episode the frames looked at, with joints, still and with the board, and the notes
  above. A dataset camera's own intrinsics (a bag's CameraInfo) are compared with the sensor's lens model (fx, 1 %)
  and the difference said; they are not used.

The session then solves as any other (`docs/models/handeye.md`: `session_problem`, the bundle adjustment).

## References

- LeRobot dataset format v2.1 and v3.0 (`lerobot_dataset_format`; read from the LeRobot repository, see
  `docs/models/datasets.md`); ROS 2 bags and their message definitions (`docs/models/datasets.md`, ROS 2 bags).
- OpenCV ChArUco detection (`opencv_charuco`) and pixel centres (`opencv_calib3d`); hand-eye calibration
  (`park_martin1994` and the solver's references in `docs/models/handeye.md`).

## Validation (measured 2026-11-07)

| Check | Reference | Tolerance | Measured |
|---|---|---|---|
| `grasp.dataset_closures` | the scripted pick's own closing ranges (`checks_grasps.scripted_pick`), written as a LeRobot v3.0 dataset of two episodes, the second 1.5 times slower; the gripper frame against MuJoCo forward kinematics of the script's joints at each onset | 10⁻⁶ m, and every stroke row for row | 6.2 × 10⁻⁸ m (float32 state column); strokes (rows 25–35, 70–75; slow episode 38–53, 106–113) all found, the 0.01 rad twitch not |
| `handeye.dataset_capture` (cost: study) | the mount the board frames were rendered through: the UR5e sample's wrist D405 moved by simcapture's default offset (2.69 mm, 0.62°); 12 planned poses (8 in a quick run) held 1 s, 0.8 s minimum-jerk moves, 5 fps, 640 × 360 px (half the stream), AV1-coded | 0.2 mm and 0.05° | full: 0.046 mm, 0.0097° (σ 0.010–0.014 mm, 0.0035–0.0060°); quick: 0.060 mm, 0.015° |

Why the hand-eye tolerance is wider than the solver's σ (measured on the 10-pose capture of the tests): the error is
3–4 times the solver's own σ even without video coding, and at full size (uncoded 1280 × 720 frames: 0.020 mm and
0.0058° against σ 0.005–0.006 mm and 0.0014–0.0034°), because the corner errors of a rendered view are not
independent, which the σ assumes. Half the stream size doubles the corner error in the sensor's pixels (RMS 0.28 px
against 0.15 px) and the error with it (0.049 mm uncoded); AV1 coding adds under 0.002 mm (0.051 mm). The camera
starts 2.7 mm and 0.62° from the truth; corners left in the stream's pixels or joints read at the wrong frame miss by
millimetres or tenths of a degree, far outside 0.2 mm.

Tests (`tests/test_dataset_links.py`, 15): the joint matching (names, LeRobot names in degrees, partial, by order,
refused) and the interpolation against the closed form; grasps from LeRobot v2.1 and v3.0 against the script's
closing times and the forward kinematics at the onset (10⁻⁶ m), from episodes generated by `core/sim/episodes.py`
(rendered on the RTX 5080, EGL device 0: one grasp per episode within one row of 1.5 s, within 1 mm of the
trajectory's grasp), and their refusals; hand-eye captures from LeRobot v2.1 and v3.0 (0.05 mm, 0.011° from the
truth) and from an MCAP bag with its own 2026-epoch nanosecond clock, joints at 50 Hz and images 7 ms after a joint
row (the 7 ms reported, the same 10 views); the offset and moving-frame rules on a ramp, exactly; the refusals; the
two pages' commands (synchronously and as window jobs, one undo step) and the dialog's parsing.

Performance (this machine's CPU): grasps from 200 episodes of 136 rows, 0.87 s (31 000 rows/s); a hand-eye capture
of 90 frames at 640 × 360 (52 decoded and searched), 0.19 s.

## Known limits

- Joints: one robot per state vector; joint values are taken in the model's convention (no zero offsets, gear
  ratios or gripper percentages converted); matching by order is only as right as the dataset's order.
- Grasps: the limits of `from_trajectory` (a closing that is not a grasp still counts; fingers closing by less than
  0.1 mm between rows are not seen to close); the onset is the last row before the fingers move, so its time is
  quantised to the row spacing.
- Hand-eye: linear interpolation between joint rows (a curved path between rows is cut short); no clock offset is
  estimated, only the one given is applied; motion blur, rolling shutter and the exposure time are not modelled in
  moving frames (which is why they are left out by default); a dataset's own intrinsics are compared, never used;
  live ROS 2 topics are not read (record a bag); a stream of another shape than the sensor's (a crop) is refused.
- The hand-eye solve study's run-time capture from settings (`core/studies/handeye.py`, `folder` and `joints`) reads a
  folder only; a dataset is captured on the page (or with `capture_dataset` in a script) and stored as a session.
