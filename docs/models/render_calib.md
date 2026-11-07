# Targetless link-silhouette calibration (spec §8.4)

`dexvantage/core/calib/render_calib.py` (the method), `dexvantage/core/studies/silhouette_calib.py` (the study,
`--silhouette-calib`), `dexvantage/ui/pages/silhouette.py` (Calibrate ▸ Silhouette calibration),
`dexvantage/validation/checks_render_calib.py` (two checks), `tests/test_render_calib.py`.

A static camera (one not on a robot link) is calibrated against the robot base with no board. The robot is rendered
at the joint states logged with the frames, through MuJoCo's offscreen segmentation (`core/sim/render.Renderer`, the
mount perturbed through `extrinsic_delta`). The rendered silhouette is then aligned with the recorded frame's edges and
foreground mask. Run per episode, the estimate shows the mount's drift (a bracket knocked between episodes, a slow
sag). The dataset auditor uses `estimate` for its mount-slip check.

## API

    estimate(dataset_or_episode, project, sensor_uid, frames=None, settings=None, *, renderer=None, progress=None,
             cancel=None) -> dict
    drift(dataset, project, sensor_uid, episodes=None, settings=None, ...) -> {'episodes', 'steps', 'largest_step',
                                                                             'trend', 'step_rule'}
    settings_of(project, given=None) -> dict
    static_cameras(project), static_sensor(project, uid), match_stream(episode, sensor, key=None), status_of(doc, st)

`dataset_or_episode` takes a `core.datasets` Dataset (settings `episode` picks one; the default is the first), an
Episode, or a path. `frames` lists indices into the camera's stored frames to choose from. Pass a `renderer` to reuse
one MuJoCo worker across calls. The result is JSON-able:

| key | meaning |
|---|---|
| `pose` | the recovered optical frame in the world (the primary robot's base), position and quaternion (x, y, z, w) |
| `pose_before` | the same frame at the project's mount |
| `mount` | `{'parent_frame', 'robot_uid', 'pose'}`: the recovered sensor body pose, which Apply writes |
| `delta` | ΔT in the sensor body frame: `position_m`, `rotation_deg` (a rotation vector), `norm_mm`, `angle_deg`. The pose is T_world_sensor · ΔT · T_sensor_optical, the convention of the synthetic episodes' `mount_slip` fault and of `extrinsic_delta` |
| `delta_optical` | the same correction in the optical frame |
| `sigma` | `position_mm`, `rotation_deg` per axis (the largest of the three estimates below); `hessian`, `cluster_robust`, `bootstrap`; `covariance` (6 × 6, m and rad) |
| `frames_used`, `frame_times_s` | the frames the estimate used |
| `observable`, `why`, `observability` | the observability test (below) in words and figures |
| `fit` | IoU and chamfer before and after, the edge RMS, residual counts, per-frame figures |
| `scale` | one pixel of the working image at the robot's median contour depth (`mm_per_px`, `deg_per_px`) |
| `overlay` | a thumbnail (base64 PNG) of the frame with the largest silhouette, with outlines before and after |
| `render` | the renderer (GPU), MuJoCo version, render count |
| `method`, `settings`, `defaults`, `notes`, `seconds` | |

The estimate never reads the dataset's ground-truth sidecar.

## Method

**Frames.** Up to `max_frames` (default 12) frames are chosen by farthest-point sampling (max-norm) on the joint
vectors q(t) at the frames' timestamps. q(t) is interpolated linearly per joint from the state rows. Joints the
episode does not record stay at home.

**Image features**, at each level of a pyramid (default scales 1/4, 1/2 and 1 of a working image at most
`max_width` = 640 px wide; levels narrower than 160 px are left out):
- Canny edges of the colour image after a 1-σ Gaussian (thresholds 50 and 150, L2 gradient; OpenCV takes the largest
  channel gradient). Each edge pixel moves to the vertex of a parabola through the gradient magnitude at −1, 0 and
  +1 px along its normal, so e = p + δ n with δ = ½ (m₋ − m₊) / (m₋ − 2m₀ + m₊), clipped to ±½.
- A foreground mask F: pixels that differ from the per-pixel median of the frames used by more than `fg_threshold`
  (default 25 grey levels) in some channel, opened with a 3 × 3 square. Its boundary is a second edge set: the inner
  boundary pixels, moved ½ px outward along the smoothed mask's normal.
- Directional chamfer: each edge set keeps one Euclidean distance transform per orientation bin (6 bins of 30° on the
  normal's angle modulo 180°; `cv2.distanceTransformWithLabels`, DIST_L2, 5 × 5 mask, DIST_LABEL_PIXEL). A point is
  matched to the nearest edge pixel in its own bin or the two next to it (Liu et al. 2010), so it does not snap to a
  nearby edge running across it.

**Rendered features** at the current estimate: the robot mask S (segmentation owners that are robot links) and its
contour points:
- the outer contour: a mask pixel with a 4-neighbour outside S, at the pixel edge midpoint (p + ½ d), with the
  smoothed mask's outward normal;
- depth jumps inside S larger than `inner_jump_m` (default 2 cm): one link in front of another.

Points next to a nearer body (an object in front of the robot) or within 2 px of the image border are left out. Each
point is back-projected with its rendered depth along the lens model's ray to a world point X. One point per
`point_spacing_px` cell is kept (default 3 px), at most `max_points` (400) per frame. For the foreground residuals,
only outer points away from the majority mask M are used (M = pixels inside S in more than half of the frames,
dilated by 2 px). There the observed F's boundary is the robot's own outline; elsewhere the median background is the
robot itself.

**Residuals.** A point projecting to u, matched to edge pixel e with normal n_e, has the point-to-tangent residual
r = n_e · (u − e). It counts only when the distance transform at u is within the truncation τ. τ is `truncate_px`
(10 px) at the coarsest level and tightens as the levels refine, τ_L = max(3, 10 · s₀ / s_L): 10, 5 and 3 px at
1/4, 1/2 and 1. The loss is Huber with threshold `huber_px` (1 px). An out-of-range point costs as much as a residual
at τ, so pushing points out of range is not free. The foreground residuals are weighted by `mask_weight` (1).

**Optimisation of ΔT**, ξ = (ρ, φ) composed on the right, D ← D · Exp(ξ):
1. Coarse search on the coarsest level: a pattern search over the twelve moves ±step along and about the sensor body
   axes, each move rendered (no gradient). The cost is J = mean truncated chamfer / τ + `iou_weight` · (1 − IoU),
   where the IoU compares the observed F with the predicted foreground S_i XOR M. Steps start at 4 coarse pixels at the
   robot's median depth and halve down to ½ px.
2. Levenberg–Marquardt on every level, coarse to fine. The contour points are re-rendered at each outer iteration
   (default 4) and held fixed for the inner iterations (default 8), because the contour generator slides over curved
   links as the pose changes. The Jacobian is analytic in the pose and the edge:
   ∂p/∂ρ = −R_soᵀ, ∂p/∂φ = R_soᵀ [y]ₓ (y the point in the current sensor body frame, p in the optical frame) and
   ∂r/∂u = n_eᵀ. The lens projection's ∂u/∂p is by central differences of `CameraModel.project`, so every lens model
   (pinhole, Brown–Conrady, Kannala–Brandt, the fisheyes) works the same way.

**Uncertainty**, at the solution on the finest level:
- (a) Gauss–Newton: C = σ̂² (JᵀJ)⁻¹, σ̂² = Σr² / (n − 6). Frames of the same pose (no joint differing by
  `min_motion_deg`, 5°) are counted once, since their silhouettes repeat the same errors.
- (b) Cluster-robust (Liang & Zeger 1986): C = B⁻¹ (Σ_c s_c s_cᵀ) B⁻¹ · c/(c − 1), with bread B = JᵀWJ and the scores
  s_c = Σ_{i∈c} w_i r_i J_i summed per cluster c = (distinct pose, link). To this is added
  B⁻¹ (Σ_c g_c g_cᵀ) B⁻¹ · σ_floor², with g_c = Σ_{i∈c} w_i J_i: a common offset of each link's contour along its
  normals of σ_floor = `edge_floor_px` (0.25 px, a default).
- (c) A bootstrap over frames: `bootstrap` (30) resamples with replacement, each re-solved by LM from the solution on
  the resampled frames' fixed contour points.

`sigma` is the largest of the three per axis.

**Observability.** I = σ̂⁻² JᵀJ (deduplicated) is scaled by the requirement:
N = S I S, S = diag(need_mm × 3 in m, need_deg × 3 in rad) (defaults 5 mm and 0.5°). The estimate is observable when
all of these hold:
- the smallest eigenvalue of N is at least 1 (every direction's σ within the requirement);
- the robot covers at least `min_robot_frac` of the image (0.5 %) with a median of `min_points` contour points per
  frame (30);
- the frames hold at least `min_poses` distinct joint states (3).

`why` names the test that failed: "too small", "too static" (with the largest joint range over the frames), or the
weakest eigenvector, named by its dominant component in the optical frame (for example "the camera's rotation about
the image's x axis (tilt)").

**Drift.** `drift` estimates each episode independently. Between consecutive episodes the change Δ is compared with
the combined σ: z = max |Δ| / √(σ_a² + σ_b²). It is a step when z > `step_z` (3) and the change is beyond the
tolerance (`tolerance_mm` 2 mm or `tolerance_deg` 0.2°). A least-squares trend per episode is reported from three
episodes on.

**Status** (the page's headline):
- `mount holds` (good): within the tolerance.
- `mount moved` (serious): beyond it by more than 3 σ.
- `uncertain` (warning): beyond the tolerance but within 3 σ.
- `not observable` (warning).

All of these thresholds are DexVantage defaults, not standards.

## Validation

| check | reference | tolerance | measured (2026-11-07, RTX 5080 through EGL device 0) |
|---|---|---|---|
| `handeye.silhouette_recovery` | the generator's sidecar (read after the estimate), synthetic LeRobot episodes (640 × 360, AV1) of the SO-100 third-person and UR5e overhead cameras, a 5 mm and 1° slip from episode 1 (only `mount_slip` on) | 2 px of the working image; every estimate observable; the slip found as a step | 1.66 px: SO-100 2.39 and 2.16 mm, 0.30°; UR5e 1.28 and 0.37 mm, 0.03°. Steps found: SO-100 5.5 mm and 0.96° (truth 5.0 mm and 1.0°), UR5e 5.9 mm and 1.00° |
| `handeye.silhouette_vs_board` | the board-based bundle adjustment (`core/calib/handeye.py`) of a simulated ChArUco capture (15 planned poses) of the same SO-100 rig with the same slip | 2 px; the board solve within 0.5 mm and 0.05° of the truth | 1.65 px (2.12 mm, 0.30°); the board solve 0.039 mm and 0.005° from the truth |

The tolerance is in pixels of the working image so every rig is held to the same image-space standard. One pixel is
1.9 mm sideways at the SO-100's contour depth through its third-person camera (fx 320 px at 640 px wide) and 0.18°.
For the UR5e's overhead D435 (fx 466 px) it is 1.1–1.8 mm and 0.12°. In quick mode the recovery check runs the UR5e
only (1.13 px, 16 s); the full checks take 35 s and 8 s.

Other measurements (in-memory renders, no codec, 640 × 360, 12 frames, 2026-11-07):
- UR5e overhead: recovered to 0.03–0.08 mm and 0.014–0.017°, with the cluster-robust σ within 1.6 σ of the truth.
- SO-100 third-person: 0.6–1.3 mm and 0.07–0.15°, but 3–7 cluster-robust σ along the lateral/yaw direction (below).

The board-based solve is 20–50 times sharper than the silhouette. The silhouette's value is that it needs no board
and runs on every episode.

`tests/test_render_calib.py` covers the following:
- the Jacobian against finite differences;
- sub-pixel and directional edge residuals on a drawn square;
- recovery of a 5 mm / 1° slip within one pixel at 320 × 180 from six frames;
- the static-robot and too-small refusals of observability;
- drift and the study on a synthetic LeRobot dataset;
- the page (Apply is one undo step, the Camera view overlay).

## Known limits

- The robot must look like its model. Covers, cables and tools the model lacks make edges that are outliers at best
  and pull the fit at worst.
- The intrinsics are taken as exact: the stream's own when the dataset stores them, else the project's imager scaled
  to the stream's size (a different aspect ratio is refused).
- The joint readings are taken as exact at each frame's timestamp. A camera–joint desync, clock drift or rolling
  shutter on a moving robot shifts the silhouette and shows as a mount error: in the worked example, an episode with a
  50 ms desync and 5000 ppm drift was 2.65 mm and 0.23° off. Correct the timing (the auditor's sync checks) first.
- The median background needs the robot to move. Parts that never move contribute edges but no foreground.
- A small, distant robot leaves sideways shift and yaw nearly interchangeable. On the SO-100 sample's third-person
  camera (the robot about 4 % of the image, light grey partly on a light grey table) the estimate carries a bias of
  about 2.4 mm and 0.3°, the same in every episode. Steps between episodes are recovered to about 1 mm. σ is a lower
  bound along that direction (measured up to 7 σ). On the UR5e overhead camera (the robot about 25 % of the image)
  σ holds.
- The bootstrap re-solves on the final contour points without re-rendering, so it measures the frame-to-frame scatter
  at the solution, not the capture range.
- The coarse search can stop in a local minimum when the start is more than about the truncation distance away: tens
  of millimetres or a few degrees at a metre.
- One static camera per run; episodes are estimated independently.
- Run time: about 4–5 s per episode at 640 × 360 with 12 frames on the RTX 5080 (about 1 000–2 500 renders,
  mostly the bootstrap's LM and the coarse search). The study runs as a background analysis, never on the UI thread.

## References

Barrow et al. 1977 (chamfer matching); Borgefors 1988 (hierarchical chamfer matching); Canny 1986; Liu, Tuzel,
Veeraraghavan and Chellappa 2010 (fast directional chamfer matching); Efron 1979 (the bootstrap); Liang and Zeger 1986
(cluster-robust covariance); OpenCV `distanceTransformWithLabels`; MuJoCo 3.17.0 cameras (`docs/models/groundtruth.md`).
Keys in `dexvantage/validation/references.py`: barrow1977, borgefors1988, canny1986, liu2010_fdcm,
opencv_distance_transform, efron1979, liang_zeger1986, mujoco_camera_317.
