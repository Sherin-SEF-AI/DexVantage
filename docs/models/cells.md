# Multi-cell comparison

Several robot cells side by side: other labs' or benches' rigs (`.dexv` files or folders), samples, or variants of
the open project. Code: `dexvantage/core/cells.py` (the cell set, the figures table, the geometric gap, the drift,
the files), `dexvantage/core/cells_pool.py` (dataset pooling), `dexvantage/core/studies/cells.py` (the study `cells`),
`dexvantage/core/exporters/cells.py` (the export `cells`), `dexvantage/ui/pages/cells.py` (File ▸ Compare cells…),
`dexvantage/batch_cells.py` (`--cells`), `dv.cells.compare` in `dexvantage/api.py`, the assistant's `get_cells`.

The gap index is ported from Vantage `core/rig_gap.py` (the Rig Contrastive Distance) and the drift from Vantage
`core/fleet.py` and the measurement idea of `core/fleet_twin.py`; the automotive parts (viewing directions around a
vehicle, the ground in bird's-eye view, body occlusion, nuScenes and KITTI readers, Kalman drift forecasts) are not
ported.

## The cell set

The open project is cell 0 (its active variant); with `variants` on (the default) each of its other variants is a
cell; then the cells the settings name, in order: `{'path': ...}` (a `.dexv` file or an unpacked folder),
`{'sample': name}` or `'sample:<name>'`, `{'variant': uid or name}`. The **reference** (setting `reference`, default
0) is the cell the others are compared with, and the plan for the drift. Names are made unique with `(2)`, `(3)`.

## 1. Figures of merit

Every metric asked for (`metrics`: a list of ids, `'all'` for every metric whose study reads no recorded dataset and
runs no other study, or the default quick set `DEFAULT_METRICS`: coverage seen, seen twice and P5 resolution, depth
validity, the sync skew, the payload, the camera and sensor counts, the rig mass and power — a default) is evaluated
on every cell through `core.metrics.evaluate` (one result cache for the set). Per metric the table gives every cell's
value (with its bound and the reason when it is not stated), the best and the worst cell among the plainly stated
values (none when they are all equal), the spread, and per cell its difference Δ from the reference.

With `tolerance_draws` n ≥ 2 the tolerance Monte Carlo (`core/tolerance.monte_carlo`, seed `seed`) runs on every
cell and a difference is called **meaningful** when

    |Δ| > k · √(σ_ref² + σ_cell²),   k = 2 (MEANINGFUL_K, a default: about the two-sided 95 % band of a normal difference)

with σ the standard deviation of the metric over the cell's own draws (the spread the build tolerances alone give).
When neither cell's figure moves under the draws (σ = 0) any non-zero difference is meaningful. A figure stated only as
a bound is not judged. Without draws nothing is judged ("not assessed"). With `core/stats` installed, such a
difference must also be significant over the draws: `stats.compare_samples` on the two cells' draws (the permutation
test of the mean difference, 1 999 resamples here, with Hedges' g), the p-values Holm-adjusted across the whole table
at α = 0.05 (core/stats' defaults); a difference beyond the spread but not significant is not meaningful (said).

## 2. The geometric gap

Each enabled camera or depth sensor contributes its picked imager (the RGB one of a depth camera, at the stream size),
posed at the home pose, with a **role**:

- **wrist**: mounted on a link a joint moves. Its optical frame is expressed in its robot's **contact frame**
  (`core/exporters/canonical_warp.contact_frame`: origin between the fingertips, z the approach axis, x the closing
  axis), so wrist cameras of different arms compare relative to the fingers. A robot without a contact frame falls
  back to the mount link (said in the camera's frame text). Which pad comes first is arbitrary, so the comparison
  also tries the other camera half a turn about the approach axis (F = diag(−1, −1, 1)) and keeps the closer
  reading (`closing_flipped`).
- **scene**: static (world, cage, tripod, table, or a link no joint moves, such as the G1's torso). Its optical frame
  in the world (the primary robot's base).

Cameras are matched one to one within a role by the Hungarian method (`scipy.optimize.linear_sum_assignment`) on

    cost = θ_axis / 90° + |c_a − c_b| / 0.5 m + |ΔHFOV| / 90° + 1 (different arms) + θ_rot / 1800°

Per matched pair:

| Figure | Definition |
|---|---|
| viewpoint angle θ_axis | atan2(‖z_a × z_b‖, z_a · z_b) of the optical axes (exact near 0) |
| rotation θ_rot | atan2 of the sine and cosine parts of R_aᵀ R_b (roll included) |
| distance | ‖c_a − c_b‖ of the centres in the role's frame |
| target distance ratio | wrist: distance to the fingertip centre; scene: to the task volumes' centre |
| HFOV difference and ratio | the lens model's edge-to-edge field through the principal point (`CameraModel.fov_deg`) |
| resolution ratio | pixels per degree at the centre, f_x · π / 180, cell over reference |
| task volume in the image | share of the image whose rays pass through a task volume: rays through a grid of pixel centres at most 128 wide, unprojected through the lens, slab-tested (Kay and Kajiya 1986) against each volume's box |
| task volume in view | share of the volumes' interior points (9 per edge) inside the image |

Components, each in 0..1, means over the matched pairs:

    viewpoint  = min(1, θ_axis / 90°)                 weight 0.30
    position   = min(1, distance / 0.5 m)             weight 0.20   (Vantage matches on distance / 2 m around a vehicle)
    resolution = min(1, |log2 ratio| / 2)             weight 0.15   (two octaves = 1)
    coverage   = |Δ image share|                      weight 0.15
    fov        = min(1, |ΔHFOV| / 90°)                weight 0.10
    topology   = ½ (|n_a − n_b| / max(n_a, n_b) + unmatched cameras / (n_a + n_b))   weight 0.10

    gap = Σ w_k c_k / Σ w_k ;   band: < 0.15 small (good), < 0.35 moderate (warning), else large (serious)

The weights, scales and bands are DexVantage's defaults. The bands (0.15, 0.35), the two-octave resolution scale
and the 90° angle and FOV scales are Vantage's (core/rig_gap.py); the components and their weights are DexVantage's
own (Vantage weights coverage 0.30, resolution 0.20, height and pitch 0.15, bird's-eye view 0.15, FOV 0.10 and
topology 0.10 for vehicle rigs, with no viewpoint or position component; its 2 m is a matching cost only). The bands
are **not calibrated against any policy's success**. The gap is symmetric and 0 for identical cells. A full matrix of every
pair is in the result (`gap.matrix`).

## 3. Dataset pooling

Each cell's recording is a **schema** read from a LeRobot `meta/info.json` (the v2.1 / v3.0 feature schema of
`core/datasets/lerobot.py`, from the LeRobot source at commit d490e47, read 2026-11-07): the cell's first recorded
dataset (`project.data.datasets`, opened through `core/datasets`; an RLDS folder or a ROS 2 bag through its first
episode) under `pooling: auto` (the default) or `recorded`, else the plan: the info.json its LeRobot export would
write (`core/exporters/lerobot.info_json` over `rig_streams` and `state_names`), with the joint types and limits of
the robot models, the policy's control rate and the streams' frame rates. The plan's codec is LeRobot's default
encoder (AV1 through libsvtav1), labelled a default, not a recording.

The rules (cell against the reference's schema) and their mapping steps:

| Feature | Same | Mapping step | Cannot be mapped |
|---|---|---|---|
| camera key | paired by key, then by role in order | `rename` | a target view with no counterpart |
| image size | equal | `resize` (aspect change said: pad or crop) | |
| camera rate | equal | `resample` (drop or repeat frames) | |
| codec | equal | `reencode` | |
| extra camera | | `drop` | |
| dataset rate (rows) | equal | `resample` rows | |
| joint count | equal | | another embodiment |
| joint names | equal | `reorder` (index list), or `rename` by position (an assumption) | |
| joint units | equal (rad / m) | `convert` deg → rad (a ±2π heuristic on stored stats) | revolute against prismatic |
| action space | equal | | another space (joint against end effector) |
| gripper encoding | equal type and range | `rescale` the [closed, open] range linearly (an assumption) | |
| control rate | equal | `resample` actions | |

Verdict: poolable as is, poolable after mapping, not poolable. The pooling mapping JSON (`cells_pooling.json`,
schema `dexvantage.cells_pooling/1`) holds the target's schema and per cell its verdict, steps, what cannot be mapped,
warnings (figures not stated) and the camera map.

## 4. Drift of one design built several times

The reference cell is the plan; a cell is a copy of it for each camera whose name it shares (case-insensitive). For
each plan camera X_plan is its optical frame in its mount frame (`core/calib/records.camera_mount`: the moving link,
else the world). A copy's camera is **measured** when the cell holds a solved hand-eye session for it (the latest by
date; its X and σ from `HandEyeResult.to_dict`), else **as designed** (its own mount: a design difference, not drift).
A copy on another mount frame is "not comparable".

    Δ = [ (t_meas − t_plan) · 1000 mm ,  log(R_meas R_planᵀ) in degrees ]          (mount-frame axes)
    share of tolerance  r = max( max|Δ_xyz| / tol_mm ,  ‖ω‖ / tol_deg )             (the plan camera's mount tolerance)
    verdict: ok r ≤ 1, adjust r ≤ 3, re-mount above (Vantage core/fleet.py)

Across the measured copies of a camera (two or more): the median m and robust spread s = 1.4826 · MAD per component
(Hampel 1974), floored at a quarter of the tolerance; with three or more copies, a copy whose robust z = (Δ − m) / s
exceeds 3.5 in any component is an outlier; the build is **systematic** when the median is beyond the tolerance
while more than half the copies sit within tolerance of the median. Then the suggestion adds a variant with those
cameras at plan ⊕ median (X = [Exp(ω_m) R_plan | t_plan + m_xyz]); the plan is kept and stays active.

## The suggestion

With the open project not the reference: mount its least-alike matched camera at the reference camera's pose in the
role's frame (`T_world = T_world_role · T_ref`, `F · T_ref` when the pair is flipped), so that pair's viewpoint and
position gaps are zero. With the open project the reference and a systematic drift: add the as-built variant. Apply
is one undo step; a locked camera is not moved.

## Validation

`dexvantage --validate out/ --models cells` (2026-11-07, all pass):

| Check | Reference | Measured | Tolerance |
|---|---|---|---|
| `cells.gap_identical` | a cell against its exact copy (SO-100 sample; a synthetic three-camera rig) | 0 | 10⁻¹² |
| `cells.gap_rotation` | a camera turned by θ (1–60°, SciPy rotation vectors) about its centre: rotation θ, viewpoint θ or 0, distance 0 | 3.6 × 10⁻¹⁴ ° | 10⁻⁹ ° |
| `cells.gap_shift` | a camera moved by d: distance ‖d‖ (wrist in the contact frame, scene in the world) | 2.8 × 10⁻¹⁴ mm | 10⁻⁹ mm |
| `cells.coverage_polygon` | the exact share of a box's projected convex hull (Andrew 1979) clipped to the image (Sutherland and Hodgman 1974), 40 random boxes | 0.0144 | 0.02 (grid boundary cells) |
| `cells.pooling_lerobot` | three LeRobot datasets written by `write_lerobot` (v3.0, v2.1) with known differences, read back through the reader: steps found, unmappable joint count, sizes and rates against the decoded frames | 0 | 0 |
| `cells.drift_known` | copies with known offsets as hand-eye results: offsets, median, 1.4826 · MAD and the outlier's z by hand; a shared offset called systematic | 1.2 × 10⁻¹⁵ | 10⁻⁹ |

The coverage tolerance is the grid's: at most 128 columns of rays, so the cells along the polygon's edge carry about
1–2 % of a share (measured 1.44 % at worst). The self-test's `cells` line runs the first, second and last checks
(about 1 s).

## Known limits

The gap is geometric (viewpoint, distance, field of view, resolution, coverage of the task volume), not appearance
(lighting, background, blur, colour) and not a predicted success rate; its bands are a heuristic. Wrist cameras are
related at the home pose through the contact frame the grasp code measures (an assumed one for a robot without
pads); occlusion of the task volume by the robot is not in the coverage share; scene cameras of two cells compare in
their own world frames, so robots standing differently relative to their tables differ for that reason too. The gap
compares the cells' designs (a calibration applied on the Hand-eye page writes the measured mount into the design);
the drift reads the calibrations themselves, applied or not. Pooling reads the feature schema, not the data (the stats only for the units
heuristic and the gripper range); joints correspond by name or position, never by kinematics; a recorded camera's
role comes from its key name; LeRobot's own multi-dataset loader is not run (the package is not a dependency). The
drift compares the latest calibration per copy and camera (earlier ones are listed), with small-angle rotation
vectors, matching copies by camera name; trends over time, knocks and forecasts (Vantage `fleet_twin.py`) are not
ported. The tolerance spread costs n evaluations of every study named per cell.

## References

Vantage `core/rig_gap.py`, `core/fleet.py`, `core/fleet_twin.py` (the ported models); H. W. Kuhn, "The Hungarian
method for the assignment problem", Naval Research Logistics Quarterly 2, 1955; T. L. Kay and J. T. Kajiya, "Ray
tracing complex scenes", SIGGRAPH '86; I. E. Sutherland and G. W. Hodgman, "Reentrant polygon clipping", CACM 17(1),
1974; A. M. Andrew, "Another efficient algorithm for convex hulls in two dimensions", IPL 9(5), 1979; F. R. Hampel,
"The influence curve and its role in robust estimation", JASA 69, 1974; LeRobot dataset format v2.1 and v3.0,
https://github.com/huggingface/lerobot at commit d490e472a5d1a207cfe0255caaaf74b6d66e4656 (read 2026-11-07).
