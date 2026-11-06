# Hand-eye calibration {#handeye}

Modules: `dexvantage/core/calib/boards.py` (ChArUco and AprilGrid boards, printing), `detect.py` (corner detection,
sub-pixel refinement, PnP), `axxb.py` (closed-form AX = XB and AX = ZB solvers), `handeye.py` (the bundle adjustment
and its covariance), `synthetic.py` (rendered board images and synthetic hand-eye problems for the checks and tests).
Checks: `dexvantage/validation/checks_handeye.py` (model `handeye`). Tests: `tests/test_calib.py`. Spec §8.1 (board
detection and solvers) and the hand-eye checks of §10. Ported from Vantage `core/targets.py` (board points) and
`core/calib.py` (grey conversion, PnP on lens-undistorted rays, per-view reprojection error, the outlier rule); the
ChArUco and AprilGrid detectors, the hand-eye solvers and the bundle adjustment are new.

The pose planner, the recording and the coach of §8.1 (`planner.py`, `capture.py`, `coach.py`, `records.py`, the
studies of `dexvantage/core/studies/handeye.py` and the pages of `dexvantage/ui/pages/handeye.py`) are described in
[Pose planning, recording and the coach](#handeye-planner) below. Tests: `tests/test_calib_planner.py`,
`tests/test_calib_ui.py`. The planner ports the Fisher-information model of Vantage `core/observability.py` and the
next-best-view idea of Vantage `core/calib_coach.py`.

## Boards

| | ChArUco | AprilGrid (Kalibr layout) |
|---|---|---|
| Definition | `CharucoBoard(squares_x, squares_y, square_m, marker_m, dictionary='DICT_5X5_100', legacy=False)` | `AprilGrid(tag_cols, tag_rows, tag_m, spacing=0.3, first_id=0, corner_squares=True)` |
| Points | the (squares_x − 1)(squares_y − 1) inner chessboard corners, OpenCV's `getChessboardCorners` order | 4 corners per tag, Kalibr's order: a (2 rows) × (2 cols) grid, index R · 2 cols + C |
| Board frame | origin at the pattern's top-left outer corner, x right, y down, z into the board (OpenCV's) | origin at the bottom-left corner of tag 0, x right, y up, z out of the face (Kalibr's) |
| Detector | `cv2.aruco.CharucoDetector.detectBoard` | `cv2.aruco.ArucoDetector`, `DICT_APRILTAG_36h11`, `markerBorderBits = 2`, contour corners, then `cornerSubPix` |
| Minimum per view | 6 corners (`detect.MIN_POINTS`, a default) | 4 tags (`APRILGRID_MIN_TAGS`, Kalibr's `minTagsForValidObs` default) |

The AprilGrid layout was read from Kalibr's sources on 2026-11-06 (`kalibr` in `validation/references.py`): the wiki
page "Calibration targets" (tagSize "edge to edge", tagSpacing the ratio of the gap to tagSize, 0.3 in the example),
`GridCalibrationTargetAprilgrid.cpp` and its header (the point order, `blackTagBorder` 2, `minTagsForValidObs` 4) and
`kalibr_create_target_pdf` (ids row by row from the bottom left, a 2-bit black border, a black square of side
spacing × tag at each tag corner). A tag is 6 × 6 data bits in a 2-bit black border, so a bit is tag_m / 10. The
dictionary default `DICT_5X5_100` is a default, not a recommendation from a source; any predefined dictionary with
enough markers works.

**Printing.** `save_board(board, 'board.pdf')` writes one page of exactly the pattern's size plus a white margin
(default one square or one tag, after Kalibr's "a white border around the calibration target of min. the size of one
grid element"), drawn in vectors with a caption in the margin; `save_board(board, 'board.png', dpi=300)` writes a
grey PNG with its resolution in the file (pHYs). `render_board(board, dpi)` returns the image. Print at 100 % and
measure a square or a tag with a ruler: the measured size is what the solver must be given.

## Detection and PnP

ChArUco corners are OpenCV's (markers, then the chessboard corners between them interpolated and refined by
`cornerSubPix`). A second `cornerSubPix` pass was tried and dropped: on rendered boards it made the corners worse at
25–40° of tilt (0.09 → 0.10–0.13 px).

AprilGrid tag corners come from OpenCV's contour corners, then `cornerSubPix` with a half-window of 0.33 of the gap
between tags in pixels (2–8 px; defaults), so the window stays inside the gap. With Kalibr's corner squares each tag
corner is a chessboard-like X corner; measured on rendered grids with 1 grey level of noise, 0.08–0.12 px RMS
against the projected truth, against 0.13–0.26 px without the squares (L corners are biased inwards). OpenCV 5.0.0's
`CORNER_REFINE_APRILTAG` was measured on the same renders at about +0.5 px in both axes (a half-pixel convention
difference) and lost tags next to the corner squares, so it is not used.

`solve_pnp(points, corners, camera)` turns the corners into rays with the camera model (`core.optics.CameraModel`,
any of its lens models), solves the planar pose on the normalised plane z = 1 (`cv2.solvePnPGeneric` with IPPE, the
better of its two solutions), refines it with `cv2.solvePnPRefineLM`, then once more in pixels through the camera
model (`scipy.optimize.least_squares`). The per-corner error is `|uv_detected − project(T · P)|` and the view RMS
is `√(mean e²)`.

## Closed-form solvers

With `T_g2b` the gripper (or any mount link) in the base frame from forward kinematics and `T_t2c` the board in the
camera frame from PnP:

- **AX = XB** (`axxb.hand_eye`, OpenCV's `calibrateHandEye` conventions): for every pair i < j,
  `A = T_g2b[j]⁻¹ T_g2b[i]`, `B = T_t2c[j] T_t2c[i]⁻¹`, `X = T_cam2gripper`. Eye-to-hand passes `T_base2gripper` and
  gets `T_cam2base`. Methods: Tsai-Lenz (modified Rodrigues vectors), Park-Martin (log map, (MᵀM)^(−½) Mᵀ),
  Horaud-Dornaika (quaternions, smallest eigenvector), Andreff (Kronecker linear system), Daniilidis (dual
  quaternions). The four rotation-first methods share the translation step `(R_A − I) t = R_X t_B − t_A`.
- **AX = ZB** (`axxb.robot_world`, OpenCV's `calibrateRobotWorldHandEye` conventions): `A = T_world2cam`,
  `B = T_base2gripper`, `X = T_base2world`, `Z = T_gripper2cam`. Methods: Shah (the leading singular pair of
  Σ R_B ⊗ R_A, then the translations in least squares) and Li (one linear system in the rotations and translations).

OpenCV's functions are used when the installed OpenCV has them. **OpenCV 5.0.0's Python package (the `.venv` of this
build) exposes only the method constants, not the functions**, so the NumPy versions in `axxb.py`, written from the
papers, run here; every result names its backend. The check `handeye.numpy_vs_opencv` compares the two on exact
poses and is registered only where OpenCV has the functions; it has not run in the suite of this build.

**Measured once against OpenCV 4.10.0** (2026-11-07, `opencv-python-headless` 4.10.0.84 loaded from a scratch folder
ahead of the `.venv`'s 5.0.0, not installed; the synthetic UR5e scenes of `core/calib/synthetic.py`, 15 poses, 5
seeds, eye-in-hand and eye-to-hand; noise added to the board poses as Gaussian 1σ per axis). Largest difference of X or
Z between the NumPy and the OpenCV result of the same method:

| Method | exact poses | 0.5 mm, 0.05° | 2 mm, 0.2° |
|---|---|---|---|
| Park-Martin, Horaud-Dornaika, Daniilidis, Shah | ≤ 2.6 × 10⁻¹² mm | ≤ 3.1 × 10⁻¹² mm | ≤ 7.7 × 10⁻¹² mm |
| Tsai-Lenz | 6.4 × 10⁻¹³ mm | 0.11 mm, 0.005° | 0.40 mm, 0.022° |
| Andreff | 2.3 × 10⁻¹² mm | 2.3 mm, 0.030° | 13.6 mm, 0.11° |
| Li | 2.9 × 10⁻¹² mm | 2.4 mm (rotation identical) | 14.0 mm (rotation identical) |

Four methods are OpenCV's to rounding. Andreff and Li differ by several times the pose noise, as their design says
(OpenCV keeps the linear system's translation; these re-solve it with the projected rotation): against the truth the
NumPy versions were the closer ones (worst 6.2 and 6.0 mm at 2 mm noise, OpenCV's 14.0 and 14.5 mm), so a NumPy
result labelled Andreff or Li is not what OpenCV's method of that name returns. Tsai-Lenz differs by a few tenths of a
millimetre under noise for a reason not traced here (the solution is not OpenCV's to rounding, though both are within
0.25 mm of each other's error against the truth). All these are initialisers; the bundle adjustment starts from one
and does not depend on which.

## The bundle adjustment

`handeye.solve(Problem, Settings)` estimates, in one sparse non-linear least-squares problem
(`scipy.optimize.least_squares`, trust-region reflective):

- X_c for every camera: its optical frame in its mount frame (a robot link, or the world for a static camera);
- Z_b for every board: its frame in its mount frame (the world, or a link when the robot holds it);
- optionally each camera's intrinsics (fx, fy, cx, cy and its lens model's distortion terms);
- optionally joint offsets δ (encoder zero errors) of the joints that move the mount links of the robots named in
  `Settings.joint_offsets`, with a prior δ ~ N(0, σ_δ²), σ_δ = 1° by default;
- optionally per-pose joint corrections ε when the encoders are given a noise σ_q (`Settings.encoder_sigma_deg`):
  the readings then enter as measurements, ε ~ N(0, σ_q²).

For a corner k seen in view j (camera c, board b, robot pose p):

    M(link, p)   = FK_world_link(q_p + δ + ε_p)      (identity for a world mount; RobotModel.fk_array)
    T_cam_board  = (M_c(p) X_c)⁻¹ M_b(p) Z_b
    uv_k         = project_c(T_cam_board · P_k)
    cost         = Σ |uv_k − uv_detected|² / σ_px² + Σ |δ|² / σ_δ² + Σ |ε|² / σ_q²

Several cameras, several boards and several robots enter the same problem (multi-camera, multi-arm). Eye-in-hand,
eye-to-hand and the general case (a camera on one arm, a board held by another: AX = ZB with both sides moving) are
the same model with different mounts.

**Parameterisation.** Each transform is updated as `T = [R₀ Exp(ω) | t₀ + v]`: ω (rad) about the transform's own
axes, v (m) along its parent's.

**Sparse Jacobian.** A view's residuals depend only on its camera's X and intrinsics, its board's Z, its robots'
offsets and the joints of its pose. The Jacobian is formed by central differences over groups of columns that share
no row (a greedy colouring of that pattern), so all poses' joint columns cost one evaluation per joint together;
`tests/test_calib.py` checks the grouped Jacobian against a column-by-column difference.

**Initial values.** Every view's board pose from PnP; then, for each camera-board pair not yet placed, every
applicable closed-form method (the five AX = XB methods when exactly one of the two is static, Shah and Li always);
the one with the smallest pixel RMS starts the adjustment (`Settings.initialiser = 'best'`, or a named method).
Cameras and boards linked only through others are placed from the ones already known. The result lists every method's
X, Z and RMS, so the closed-form results can be compared with the adjustment.

**Covariance.** At the optimum the problem is re-linearised (the estimates become the origins, the deltas zero) and

    Σ = (Jᵀ J)⁻¹,   pixel rows weighted by 1/σ̂,   σ̂² = Σ r² / (m − n)

with m the pixel residuals and n the parameters (the Gauss-Newton covariance with the residual variance; the priors
keep their stated σ). Directions with eigenvalues below 10⁻¹² of the largest are reported as not determined. σ of X
and Z are given in mm (x, y, z along the parent's axes) and degrees (rx, ry, rz about the frame's own axes), with the
full 6 × 6 blocks in the order (rx, ry, rz, x, y, z) in rad and m. A joint offset whose σ stays above half its prior
σ (a default) is marked "not determined": the views cannot tell it from X or Z.

**Outliers.** After a solve, a view whose RMS exceeds max(3 × the median view RMS, 1 px) is dropped and the problem
solved again, up to 5 rounds, while at least 3 views per camera remain (the rule and its numbers are defaults, ported
from Vantage's per-unit lens calibration). Dropped views are listed with the reason.

**Output** (`HandEyeResult.to_dict()`, schema `dexvantage.handeye/1`): per camera and board the pose, matrix, frame
("wrist optical frame in link wrist_3_link of UR5e + Robotiq 2F-85"), σ and covariance; estimated intrinsics with σ;
joint offsets with σ and "determined"; per view the corners, RMS and whether it was used; the overall RMS, σ̂, degrees
of freedom, the initialisers tried and the settings with their defaults.

## Validation

Measured with the suite's default seed (20261027) on 2026-11-06, OpenCV 5.0.0 (backend `numpy` for the closed-form
solvers, see above), full run of the seven checks 16 s (quick run 10 s):

| Check | Measured | Tolerance | Why that tolerance |
|---|---|---|---|
| `handeye.closed_form_exact`: all 7 closed-form methods on exact poses, eye-in-hand and eye-to-hand | 1.8 × 10⁻¹⁵ m (rotation 3.5 × 10⁻¹⁵ rad) | 10⁻⁹ m, 10⁻⁹ rad | exact on noise-free motions; a convention error gives centimetres |
| `handeye.closed_form_noisy`: the same on PnP poses from 0.3 px corners, 5 seeds | 5.87 mm, 0.47° | 15 mm, 1.5° | measured spread 3–8 mm, up to 0.6°, when set; a convention error gives tens of mm and degrees |
| `handeye.ba_truth`: the adjustment against the truth in units of its σ (two cameras eye-in-hand and eye-to-hand, intrinsics, encoder noise, joint offsets) | 3.11 σ; translation of the wrist and held-board transforms 0.18 mm | 5 σ and 1 mm | an unbiased estimate is within about 3σ per component; a model error is off by tens of σ or more |
| `handeye.ba_covariance_mc`: empirical / predicted σ over 40 noise draws (0.5 px, 15 poses), 12 components of X and Z | ratios 0.94–1.20; 0.56 of the band | the 99 % χ² band for 40 draws, Bonferroni over 12: 0.65–1.39 | the χ² spread of a σ estimated from n draws |
| `handeye.axzb_two_arms`: camera on one UR5e, board held by a second; the adjustment against Shah and Li | 3.50 mm (Shah); the adjustment 0.08 mm and 1.18 σ from the truth | 15 mm, and 5 σ to the truth | as `closed_form_noisy` and `ba_truth` |
| `handeye.charuco_render`: rendered ChArUco (Brown-Conrady camera, 0.38–0.6 m, up to 40° tilt) detected, corner RMS against the projected truth | 0.113 px; PnP 0.21 mm, 0.044° | 0.25 px; 1 mm, 0.2° | sub-pixel detectors reach about 0.1 px; a half-pixel or layout error fails |
| `handeye.aprilgrid_render`: the same for an AprilGrid with corner squares | 0.118 px; PnP 0.046 mm, 0.019° | 0.25 px; 1 mm, 0.2° | as above |

A larger Monte Carlo run during development (150 draws on one geometry) gave ratios of 0.91–1.13. End to end from
rendered images (20 poses of the eye-in-hand scene, 1 grey level of noise, `docs/guide/handeye.md`), the adjustment
recovered the wrist camera to 0.023 mm and 0.0013°, about 2.5 times its predicted σ: rendered-corner errors are
not perfectly independent from corner to corner, so treat the σ as a lower bound when the corner noise is not white.

## Known limits

- **OpenCV 5.0.0** has no `calibrateHandEye` or `calibrateRobotWorldHandEye` in its Python bindings; the NumPy
  solvers stand in and follow the papers, not OpenCV's source. Compared once with OpenCV 4.10.0 (above): Park-Martin,
  Horaud-Dornaika, Daniilidis and Shah agree to rounding; Andreff and Li re-solve the translation and differ from
  OpenCV's by several times the pose noise; Tsai-Lenz by a few tenths of a millimetre under noise, not traced.
- **Tsai-Lenz** is singular for a camera rotation (in the static frame) near 180° (its modified Rodrigues vector
  grows without bound): an eye-to-hand camera looking straight back along the base's axes gave 181 mm and 41° of
  error in development. With `initialiser='best'` another method starts the adjustment.
- Robot base poses are taken as exact; a second arm's base is not estimated. Link lengths and joint-axis errors are
  not modelled, only zero offsets. The first joint's offset with a world board, and the last joint's with a camera
  on the last link, trade off against Z and X and stay at their prior.
- Corner noise is modelled as independent and equal on every corner; images and joint readings are taken at the same
  instant (still poses); boards are flat and printed true to the size entered.
- Detection has been measured on rendered boards only (no blur, vignetting, glare or real print texture); a board
  printed by Kalibr's own `kalibr_create_target_pdf` (which turns its code matrix by 180° before drawing) was not
  tested for the corner order OpenCV reports.
- ChArUco corner interpolation uses the lens model only for pinhole and Brown-Conrady cameras; a strongly distorted
  fisheye may lose corners near the image edge.
- The planner, the coach and the capture have their own limits, listed in their section below.

## Pose planning, recording and the coach {#handeye-planner}

### The planner (`planner.py`, study `handeye_plan`, batch `--handeye-plan OUT.json`)

A calibration pose set for one camera and one board, exactly one of them on a robot link: eye-in-hand (the camera on
the link, the board in the world) or eye-to-hand (the camera fixed, the board held on the link). The number of poses
is the user's (default 20; the specification asks for 15–25).

**The model** (ported from Vantage `core/observability.py`: corner pixels with independent noise σ on each axis,
differentiated with respect to the unknowns, the information summed, the covariance its inverse, σ per axis in mm and
degrees, a direction nothing pins reported as not determined). The unknowns are the bundle adjustment's X and Z in its
own parameterisation `T = [R₀ Exp(ω) | t₀ + v]` (12 parameters). A corner P of the board seen at joint vector q:

    P_c = Rxᵀ (Rcᵀ (M_b(q) Z P − t_c) − t_x),    M_c(q) = [Rc | t_c], M_b(q) the mount frames in the world
    ∂P_c/∂ω_X = [P_c]ₓ     ∂P_c/∂v_X = −Rxᵀ     ∂P_c/∂ω_Z = −G Rz [P]ₓ     ∂P_c/∂v_Z = G,   G = Rxᵀ Rcᵀ Rb
    J = ∂uv/∂P_c · ∂P_c/∂θ        (∂uv/∂P_c by central differences of the lens model, any core.optics model)
    H = Σ_views Σ_corners Jᵀ J / σ²,      Σ = H⁻¹

Σ is the Gauss-Newton covariance the bundle adjustment reports when its residual σ̂ equals σ (`tests/test_calib_planner.py`
checks the Jacobian against the adjustment's own, column for column, and H⁻¹ against its covariance at a fixed σ).

**Which corners count** (each a setting with a default): inside the image 3 px from its border; the camera on the
printed side; the line of sight within 60° of the board's normal; the squares (an AprilGrid's tags) at least 8 px wide
in the image along both board axes (Vantage's `min_square_px` rule); an AprilGrid tag only whole, at least 4 tags
(Kalibr's `minTagsForValidObs`); at least 6 ChArUco corners (`detect.MIN_POINTS`); and none hidden by the posed robot,
the task objects or the tables (a cube map around the camera's centre, `core/occlusion.py`, the camera's own seat left
out by `scenegeom.sensor_occluders`). Occlusion is checked for each pose the selection is about to take; its hidden
corners leave its information and it is scored again.

**Candidates**: the camera on a hemisphere around the board's centre on its printed side, at three distances (the
board's longer side filling 80, 55 and 35 % of the image width when faced squarely, from the lens model's horizontal
FOV), tilted 0, 20, 35 and 50° from the board's normal towards 8 directions, rolled −90, −45, 0, 45 and 90° about the
optical axis, looking at the board's centre: 375 candidates by default. The moving link's target follows (eye-in-hand
`Z · T_board_cam · X⁻¹`, eye-to-hand `X · T_cam_board · Z⁻¹`) and is solved by `core.robot.ik.solve_batch` (full pose,
4 seeds, self-collision checked; a chain of fewer than six joints, such as the SO-100's five, is solved position first
with the orientation as close as its joints allow, and the candidate is judged where the camera then looks, its
corners and information computed at the pose reached, and the plan's notes say so); a solution is kept when it keeps 10 mm (a default) from the project's obstacles
(`collision.obstacle_distance`). A link already within that margin at the robot's own pose (the UR5e's shoulder sits
3 mm above the table it stands on) counts only if a candidate brings it closer still.

**Selection**: greedy D-optimal. From a vague prior (1 rad and 1 m on every parameter, so the log determinant is
finite before every axis is pinned; it is not part of the reported covariance), each step adds the candidate that
maximises `log det(H + J_cᵀ J_c / σ²)` — the classical D-optimal design criterion (the volume of the uncertainty
ellipsoid; Kiefer and Wolfowitz, Fedorov's *Theory of Optimal Experiments*, 1972). log det does not depend on the units
of the parameters (a change of units adds a constant), so millimetres and degrees need no weights. Greedy selection is
not guaranteed optimal; the test `test_greedy_d_optimal_beats_random_sets` checks it beats 20 random sets of the same
size from the same candidates. The chosen poses are ordered by a nearest-neighbour tour in joint space from the
robot's current pose.

**Reported** (`dexvantage.handeye_plan/1`): per pose its joint vector, the camera's, the link's and the board's world
poses, the corner ids, how many were hidden, and how much the camera's loosest σ grows without it; σ of X and Z (mm and
degrees), the 12 × 12 covariance; the camera's loosest axis against the requirement (0.5 mm and 0.05°, DexVantage
defaults, not standards) with Vantage's status rule (good within it, warning to twice it, serious beyond, critical when
not determined); the best next pose and what it would gain; the candidates kept and why the others were left out.

### Recording (`capture.py`)

Offline capture from a folder of board images and a joint_states file. The joint file is CSV (a time column `t`, one
column per joint, radians or degrees; in degrees only the revolute joints are converted, prismatic ones stay in metres) or JSON (columns, rows, or ROS-style `{t, name, position}` samples). Image times
come from a sidecar `images.csv` / `timestamps.csv` / `images.json`, else from the file names (the last decimal number in
seconds; a 19-, 16-, 13- or 10-digit integer in ns, µs, ms or s), else images and samples are paired in order when their
counts are equal. Each image takes the nearest joint sample within the tolerance (50 ms, a default); an image taken
while the joints moved faster than 0.5°/s (a default) is flagged. Every matched image is detected with the sensor's
camera model (`detect.detect`). The session (`project.calibration.sessions`) stores the file names (no folders), the
times, the joint vectors and the detected corners. ROS 2 topics and LeRobot episodes arrive in Phase 8 and say so.

**A simulated capture** (`simcapture.py`, the solve page's **Simulate a capture…**). Board images rendered at the
pose plan's joint vectors through the sensor's lens model (`synthetic.render_view`, 2 × 2 supersampling, image noise σ
1 grey level), with the camera's true mount the project's X followed by a known offset in its optical frame
(2.0, −1.5, 1.0 mm and 0.3°, −0.2°, 0.5°, defaults: 2.69 mm and 0.62°), written as `images/pose_NN_<t>.png`,
`joint_states.csv` (each pose read 4 ms after its image and again 0.5 s later, the arm still) and `truth.json`, then
imported as a session named "(simulated)" whose source records the offset and X_true. It tests the whole loop before the rig exists; it is never a
measurement. The solve of such a session says it is simulated and how far it lands from the truth
(`doc['simulated']`). Measured on the UR5e sample (20 planned poses): the solve moves the camera 2.69 mm and 0.619°
(truth 2.69 mm and 0.62°); on the SO-100 (19 of 20 detected) 2.69 mm and 0.616°; `tests/test_phase3_product_review.py`
requires the solve within 0.2 mm and 0.05° of the truth on 12 poses. Known limits: no lighting, blur, rolling shutter,
defocus or encoder error, and a perfectly flat board printed at its nominal size, so a simulated solve is tighter than
a real one.

### Solving and applying (study `handeye_solve`, batch `--handeye-solve OUT.json`)

`handeye.solve` on the session's views (optionally joint offsets and intrinsics); the output adds per view its image,
time offset, RMS and whether it was used, the status of the camera's σ against the requirement, and the mount Apply
writes: the sensor body pose in its parent frame, `X · (pose of the imager in the body)⁻¹`, its change from the current
mount, and the refined lens model as the sensor's per-imager calibration when the intrinsics were estimated. Apply is
one undo step (the sensor and the session's result). For batch runs the settings may name a folder and a joint file to
capture at run time.

**Several sessions together** (settings `sessions`: a list of session uids, or `'all'`; the page's **Every session
together**): `capture.sessions_problem` reads each session as above and merges them into one problem for one
adjustment. A camera (a sensor) in several sessions is one camera with one X; a board record used by several sessions
is one board with one Z; images taken at the same joint vector (to 10⁻⁹) of the same robot are one robot pose; every
robot of every session takes part with its own readings (so two arms are calibrated together, and `joint_offsets`
estimates the offsets of each). This calibrates a camera that alone cannot be: a static camera that sees a board
fixed in the world (nothing moves between them, so its own session is refused) is located through that board's Z,
which a wrist camera's eye-in-hand views of the same board determine:

    M_c X_c C_j = Z      (wrist camera, view j)          X_s C'_j = Z      (static camera)

The closed-form start skips a camera-board pair that does not move relative to each other (both fixed, or on one
link) and takes such a camera from the other side once that is known (`handeye.initialise`). The output adds per
camera its σ, status against the requirement and the mount Apply writes (`cameras`, `applies`), the boards, and the
views with their session; `camera` and `apply` are the camera furthest from the requirement. Apply writes every
camera's mount and marks every session applied, one undo step. A camera no view links to the others is named in the
refusal. The coach advises the session of the robot-mounted camera (or held board) furthest from the requirement; for
a static camera seeing a static board the notes say it cannot help (its σ depends on how much of the image the board
fills and on the board's own σ).

### The coach (`coach.py`)

Ported from the next-best-view idea of Vantage `core/calib_coach.py`. The information of the poses taken (their detected
corner ids; for planned poses the corners the planner counts) at the solve's σ̂ and linearised at the solve's X and Z;
every feasible candidate that is not a duplicate of a pose taken (camera within 20 mm and turned less than 5°, Vantage's
rule) is scored, and the coach names the one that leaves the camera's loosest σ smallest (D-optimal while not every axis
is pinned, and between ties), says which axis is loosest now, what that pose brings it to, and the joint moves from the
current pose. It says "Enough" when every axis meets the requirement.

### Validation

`handeye.planner_vs_ba` (cost "study"): the planner on the UR5e sample (eye-in-hand, the sample's wrist D405 RGB
imager, the default ChArUco 9 × 7 board of 30 mm squares on the table), corners simulated at the planned poses with
0.5 px Gaussian noise (the same corner ids, projected in the check from the nominal mounts), solved by the bundle
adjustment (σ estimated from its residuals). Measured on 2026-11-06 with the suite's default seed:

| Run | Poses, corners, dof | σ_BA / σ_plan over the 12 components | max \|ln ratio\| (tolerance ln 1.1 = 0.095) | Shape (σ̂/σ taken out; tolerance 0.01) |
|---|---|---|---|---|
| quick (coarser grid: rolls −45, 0, 45°; no occlusion check) | 10, 422, 832 | 0.9855–0.9858 | 0.0146 | 2.2 × 10⁻⁴ |
| full | 15, 485, 958 | 0.9913–0.9915 | 0.0087 | 3.4 × 10⁻⁴ |

The ratios are all σ̂/σ (0.493 and 0.496 px against 0.5 px): the planner's covariance has the adjustment's shape to
3 × 10⁻⁴ and its scale differs only by the residual σ̂, whose relative scatter is 1/√(2·dof) (2.5 % and 2.3 %). The
quick run takes 1.1 s, the full one 6.6 s. A
wrong sign, frame or a missing rotation coupling in the Jacobian gives factors of 2 or more. The adjustment's own
covariance is held to a Monte Carlo spread by `handeye.ba_covariance_mc`.

End to end (`docs/guide/handeye.md`): 20 planned poses rendered (1 grey level of noise), captured and solved recovered the
wrist camera to 0.0077 mm and 0.0020°; its σ was 0.0032–0.0038 mm and 0.0008–0.0021°, so the error is about 2 times
σ, as with the rendered views above (corner errors in rendered images are not independent).

`handeye.sessions_joint` (cost "study"): two sessions on the UR5e sample with the default table board: the wrist
D405's eye-in-hand views at planned poses and the overhead D435 seeing the same board at the same joint vectors,
corners projected from the truth with 0.2 px noise, the overhead camera's mount moved 10 mm before the solve; both
calibrated X against the truth per translation axis in units of their own σ (tolerance 5 σ: an unbiased estimate is
within it with probability above 99.9 % per component; a camera located through the wrong board pose, or a shared
board estimated twice, shows its 10 mm start offset). Measured on 2026-11-07 with the suite's default seed:

| Run | Views | Wrist D405 error (σ) | Overhead D435 error | Worst |
|---|---|---|---|---|
| quick (10 poses) | 20 | 0.43, 0.07, 0.30 σ (4 µm at most) | 0.71, 0.80, 0.04 mm (0.59, 0.52, 0.53 σ) | 0.59 σ |
| full (15 poses) | 30 | 0.11, 0.84, 0.25 σ | 1.07, 0.78, 0.08 mm (1.10, 0.62, 1.28 σ) | 1.28 σ |

The overhead camera's σ is about 1 mm across the image plane: it sees the 9 × 7 board of 30 mm squares from about a
metre, the same view at every pose, so more poses of the wrist camera do not help it.

Timing (UR5e sample, 375 candidates, 20 poses, occlusion on EGL device 0, an NVIDIA GeForce RTX 5080): 6–7 s, of which
inverse kinematics 0.8 s and the occlusion checks of the poses the selection considered 5 s (133 checks).

### Known limits

- One camera and one board per plan, exactly one of them on the robot; multi-camera and two-arm problems are solved
  (`handeye.solve`, and from the project's sessions with **Every session together**) but not planned.
- A joint solve merges poses by their joint vectors only: two sessions recorded at different times but at the same
  joint vector are taken as one pose (as they should be for still poses), and sessions recorded together but whose
  joint files differ by rounding stay separate poses (harmless: the adjustment couples them through X and Z).
- The prediction takes the joint readings and the intrinsics as exact; estimating joint offsets or intrinsics in the
  solve loosens X and Z beyond it.
- Corner noise is modelled as independent and equal on every corner, so the prediction is a lower bound for detected
  corners (about 2 to 2.5 times on rendered images).
- A linearisation at the nominal X and Z (the project's mounts); the candidates are a fixed grid and the selection is
  greedy.
- Collision: the camera body and the board are not collision geometry; the motion between poses is not checked (only
  the poses); IK solutions are those nearest the robot's home, not the ones with the most clearance.
- Occlusion by sensor housings, cables and the board's holder (beyond the robot's meshes) is not modelled; the focus
  range of the camera is not checked (makers rarely state it).
- Capture: the nearest joint sample, not an interpolation; image and joint clocks taken as one clock; ROS 2 and
  LeRobot sources arrive in Phase 8.
