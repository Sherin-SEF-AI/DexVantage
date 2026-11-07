# Motion blur and rolling shutter {#motion_blur}

Module: `dexvantage/core/studies/motion_blur.py` (study id `motion_blur`, batch flag `--motion-blur OUT.json`). Page:
`dexvantage/ui/pages/motion_blur.py` (Analysis ▸ Motion blur and rolling shutter, Placement workspace). Checks:
`dexvantage/validation/checks_blur.py` (model `timing`). Tests: `tests/test_motion_blur.py`. Spec §7.9. Written for
DexVantage (Vantage's `core/timing.py` models a vehicle driving past static objects; only its convention that a
rolling shutter's middle row is exposed around trigger + readout/2 + exposure/2 is shared, in the time sync study).

## What is computed

Along each trajectory of the project (or, without one, the approaches of the first grasp set), for every camera
imager of the active variant that streams (cameras and depth sensors switched on): how fast the image of what the
camera sees moves, the smear that gives during one exposure, and the skew across a rolling-shutter readout.

## Image-plane velocity

A camera's optical frame (z forward, x right, y down) is at `o` in the world with rotation `R` (columns = the optical
axes in world coordinates). A camera on a robot link moves with the link's twist at the trajectory's joint velocity:

    v_c = J_v(q) q̇,  ω_c = J_ω(q) q̇

`J(q)` is MuJoCo's world-frame Jacobian of the camera's optical centre held fixed in the link
(`RobotModel.jacobian(q, link, point)`, `mj_jac`). A camera on a tripod or the cage has `v_c = ω_c = 0`. A world
point `P` moving at `v_P` sits at `P_c = Rᵀ(P − o)` in the optical frame. Since `Ṙ = [ω_c]× R`,

    Ṗ_c = Rᵀ (v_P − v_c − ω_c × (P − o))
    (u̇, v̇) = ∂π/∂P_c · Ṗ_c

where `π` is the imager's lens model (`core/optics`). For a pinhole, `∂π/∂P_c` is written out (`fx/z`, `−fx·x/z²`,
`fy/z`, `−fy·y/z²`). For Brown-Conrady and the fisheye models it is a central difference of `π` in the optical frame,
with a step of 10⁻⁶ of the distance.

The points:
- the scene: a `volume_grid`³ grid (4³ by default) over each task volume, and `object_points` (40) points on each
  task object's surface. These do not move (`v_P = 0`);
- the trajectory's robot: its TCP and up to nine points on each fingertip pad (`core/grasps.py`). These move at
  `v_P = v_o + ω × (P − o_link)`, from one Jacobian per link.

A point counts for a camera while it projects inside the image and in front of the camera. Occlusion is not tested.

The joint velocity `q̇` is the slope between the trajectory's samples. Trajectories are linear between samples
(`core/robot/trajectory.py`), so `q̇` is constant over each interval. The study samples every `step_s` (10 ms), at
interval mid-times.

## Smear, skew and capture position

    smear = |(u̇, v̇)| × exposure        (pixels, first order in the exposure)
    skew  = |(u̇, v̇)| × readout         (rolling shutter; 0 for a global shutter)

The skew is how far a point's image moves between the reading of the first row and the last (Liang et al. 2008). The
study also gives where a rolling shutter records a moving point (`rolling_capture`). Rows are read at
`t(r) = readout · r/(H − 1)`. A point at `(u₀, v₀)` when the first row is read, moving at `(u̇, v̇)`, is caught when
its row is read:

    t* = v₀ / ((H − 1)/readout − v̇),   recorded at (u₀, v₀) + (u̇, v̇)·t*

## Where exposure and readout come from

1. Your figure, if you gave one. Settings `exposure_s` / `readout_s`, as `{sensor uid: {value, source}}`: what the
   driver is set to, or a measurement.
2. Otherwise the library entry's figure (cited). The D400 series datasheet gives the depth imagers' manual exposure
   range for module 430: 1 to 166 ms (Table 4-23, entered for the D435). The RGB camera's "Manual Exposure Time" is
   listed as 1 to 10000 with no unit, so it is not entered. No readout time is stated for any imager in the library.
3. Otherwise a stated upper bound. The exposure is at most one frame period, `1/fps`, or the maker's stated maximum
   when that is shorter. A rolling-shutter readout is at most one frame period, because the rows of one frame are
   read within one frame period at a steady frame rate. The result labels both "upper bound". Neither is a device
   figure.
4. A sensor with no library device has no frame rate, so no exposure is known. Its smear needs your exposure; the
   page still gives its image speed and the longest exposure that keeps the limit.

## Report

- **Per trajectory segment:** the trajectory's intervals, merged into at most `max_segments` (24) equal spans. For
  each segment and camera imager: the worst image speed, smear and skew, the time, the point, and a status.
- **Per camera imager:** its peak image speed, smear and skew, the longest exposure that keeps the limit
  (`max_exposure = smear_limit / peak speed`), and the worst moment.
- **Worst frames:** the `frames_kept` (3) frames with the largest image speed, each at least 5 steps from the others.
  Each frame keeps `q` and the smear vector of up to `vectors_kept` (150) points; the Camera view draws them.
- **Status:** "sharp" when the smear is within the limit (`smear_limit_px`, 1 px by default; a DexVantage default,
  not a standard); "some blur" up to twice the limit; "blurred" beyond. When no exposure is set the smear is an upper
  bound, and over the limit the words are "may blur" and "may be blurred" (the severity is that of the bound).
  "exposure not set" when no exposure is known.
- **Recommendation:** for each camera over the limit, the longest exposure that keeps the limit, rounded down to two
  significant figures, with the predicted smear. Apply records it as your exposure (one undo step); set the same
  exposure on the camera. When that is shorter than the shortest exposure the maker states, the study suggests
  slowing the motion by the factor needed instead. When a rolling shutter's skew is over the limit, it suggests a
  global-shutter camera (skew 0) and lists library entries whose imagers are all global shutter.
- **Without a trajectory:** up to `max_grasps` (8) grasps of the first set. The arm moves joint-linearly from the
  pre-grasp pose (`pre_grasp_m`, 0.10 m back along the approach) to contact at `approach_speed_m_s` (0.10 m/s).
  Both poses are solved by IK, and both figures are defaults.

The 3D view draws the TCP's path coloured by each segment's status.

## Validation (model `timing`, measured 2026-11-07, full run)

| Check | Reference | Measured | Tolerance |
|---|---|---|---|
| `blur.image_velocity_fd` | central differences (h = 10 µs) of the points projected at q(t ± h), forward kinematics only; UR5e sample, both cameras, 92 frames | 2.8 × 10⁻⁹ of the frame's peak speed | 10⁻⁵ |
| `blur.smear_analytic` | f·v·E/Z for a point moving parallel to the image plane, f·ω·E for a camera turning about its y axis; pinhole and Kannala-Brandt | 2.4 × 10⁻¹¹ (relative) | 10⁻⁶ |
| `blur.rolling_shutter_bruteforce` | row-by-row readout simulation along the true motion (forward kinematics at every row time), readouts 10 and 30 ms, 2 002 points | 2.4 % of the point's motion (worst 0.29 px, wrist camera, 30 ms) | 5 % |

The rolling-shutter difference is the model's own limit. The study takes the image motion as linear over the readout.
The real motion turns and changes perspective, which shows at second order: a few per cent of the displacement for a
wrist camera over 30 ms.

## Worked example

The UR5e sample's scripted pick is in [../guide/motion_blur.md](../guide/motion_blur.md).

## Known limits

- **First order in the exposure:** the image velocity is taken as constant over one exposure. It jumps at the
  trajectory's samples, where `q̇` changes at once.
- **No occlusion test:** a point counted may be hidden behind the gripper.
- **The exposure bound overstates:** when no exposure is set, the bound is the frame period. That overstates the
  smear of a camera whose auto-exposure runs shorter.
- **Light is not modelled:** a shorter exposure needs more light or gain (the lighting study).
- **Other motion:** the robot's other joints and other robots stay at home. A carried object's points are counted as
  scene points, not moved with the gripper.
- **Rolling shutter:** rows are read top to bottom at a constant rate.
- **Lens derivative:** the distortion of non-pinhole lenses enters through a numerical derivative.
- **Vibration:** the mount's vibration is not modelled.
