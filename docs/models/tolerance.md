# Monte Carlo tolerances and the perturbation export

Spec §7.11 (Monte Carlo tolerances) and §8.5 (synthetic camera perturbation). Code: `dexvantage/core/tolerance.py`
(the model, ported from Vantage `core/tolerance.py`), `dexvantage/core/studies/tolerance.py` (the study `tolerance`,
batch flag `--tolerance-mc OUT.json`), `dexvantage/core/exporters/perturbations.py` (the exporter `perturbations`),
`dexvantage/ui/pages/tolerance.py` (the Tolerances page). Checks: `dexvantage/validation/checks_tolerance.py`
(model `tolerance`).

The batch flag is `--tolerance-mc`, not `--tolerance`: `--tolerance` already exists as `--reproduce`'s comparison
tolerance.

## What is drawn

Each draw moves every enabled sensor of the active variant (locked ones too: a locked mount is still built with a
tolerance). A draw of one sensor is six numbers, the factors of `core/factors.py`:

    d = [dx, dy, dz]            metres along the mount parent frame's axes
        [droll, dpitch, dyaw]   degrees about the sensor body's own x, y, z axes, in that order:  R' = R · Rx · Ry · Rz

applied to a copy of the project through `factors.apply(..., allow_locked=True)`.

### Mount tolerance: a ± bound

`Mount.tolerance` (`position_mm`, `rotation_deg`) is a **± bound per axis**, the way the Inspector and the Sensors
page show it ("± 1 mm · ± 0.5°", the project's labelled defaults). Each of the three shifts lies within ± position_mm
and each of the three turns within ± rotation_deg, independently. Inside the bound the distribution is a setting:

| `distribution` | draw | σ of one component |
|---|---|---|
| `uniform` (default, labelled) | U(−b, b) | b / √3 |
| `normal` | N(0, (b/2)²) truncated at ± b (the bound is 2 σ) | (b/2) √(1 − 2·2 φ(2) / (2 Φ(2) − 1)) = 0.4398 b |

Uniform is the default because a drawing tolerance says only that the part is inside the bound: every value within it
is taken as equally likely (Vantage's default too). The truncated normal is for a process known to centre.

### Calibration covariance

A sensor with a hand-eye session (`project.calibration`, `core/calib`) whose result is solved (`ok`) and applied
draws from that result's 6 × 6 covariance instead. The bundle adjustment parameterises the imager's optical frame X
in the mount frame as X' = [R_X Exp(ω) | t_X + v] (ω about X's own axes, v along the parent's: `core/calib/handeye.py`).
A draw is (ω, v) ~ N(0, C), turned exactly into the body's factors. With the imager's pose in the body
T_im = [R_im | t_im] and the body's nominal rotation R_B:

    δ     = R_im ω                                   (the turn about the body's axes: Exp(R_im ω) = R_im Exp(ω) R_imᵀ)
    shift = v − R_B (Exp(δ) − I) t_im                (the body origin moves when X turns about the imager)
    (droll, dpitch, dyaw) = the X-Y-Z decomposition of Exp(δ):  b = asin R₀₂, a = atan2(−R₁₂, R₂₂), c = atan2(−R₀₁, R₀₀)

so the moved project's X is exactly [R_X Exp(ω) | t_X + v]. The calibration replaces the mount tolerance only while
the sensor is where the calibration put it: if its current X differs from the stored one by more than 1 µm or 1 µrad
(the sensor was moved since), the mount tolerance is used and the source says why.

### Bounds

`bounds(project)` gives per sensor `position_m` and `rotation_deg` as `[lo, hi]` per axis and the `source`:
a mount tolerance's ± bound (× scale), or a calibration covariance's ± 2 σ of each marginal of the linearised body
covariance J C Jᵀ, with J mapping (ω, v) to (shift, δ): ∂shift/∂ω = R_B [t_im]× R_im, ∂shift/∂v = I, ∂δ/∂ω = R_im.

## Monte Carlo

`monte_carlo(project, metric_ids, n, seed)` evaluates the named metrics (`core/metrics.evaluate`: each study re-run on
the moved project, no cache) on the project as it is (the nominal) and on n draws from one `numpy.random.default_rng
(seed)`, sensors in the variant's order. Per metric:

- nominal, mean, σ (n − 1), P5, P50, P95, min, max over the draws that gave a value; the count that gave none;
- the pessimistic 5 % figure: P5 for a higher-is-better metric, P95 for a lower-is-better one;
- per project requirement on the metric: the share of draws meeting it, with its Wilson score 95 % interval
  (Wilson 1927), and whether the nominal meets it. Each draw is decided as the scorecard decides it
  (core/requirements.decide: a tolerance of 10⁻⁹ relative at the threshold; a figure the study states only as a
  bound, such as a time-sync skew of "at least 1 ms", meets or misses only when the bound settles it); a draw whose
  bound does not settle the requirement counts as not meeting it and is counted in `undecided`;
- per mount, its contribution: for each component x the Spearman rank correlation ρ(x, f) and ρ(|x|, f) (a figure
  that falls whichever way the camera turns has ρ ≈ 0 and ρ(|x|, f) large); a component counts max(ρ², ρ_abs²), a
  mount the sum over its six (split into its position and rotation parts), and the shares are those sums normalised
  over the mounts. A mount whose largest |ρ| is below 2 / √(n − 1) (two standard errors of ρ under no association) is
  flagged as not distinguishable from noise.

The study's headline is the requirement met by the smallest share of draws (ties: the smallest margin of its
pessimistic figure over the threshold, in σ), as that pessimistic figure, with a status word from the share: **holds**
from 95 %, **marginal** from 80 %, **often fails** from 50 %, else **fails** (DexVantage defaults). Without
requirements it is the figure with the widest relative spread, word **no requirement**.

Which figures: the study's `metrics` setting; else the metrics the project's requirements name; else (a default) the
headline metrics of the fast geometric studies, `coverage.covered`, `depth.valid`, `motion_blur.worst_px`,
`vla.min_tokens` (under a second per draw on the samples, measured 2026-11-07; grasp visibility takes about 30 s per
draw on the UR5e sample and is left to a requirement or the settings).

The suggestion (the page's Apply) halves the position or rotation tolerance (whichever part contributes more) of the
unlocked mount-tolerance sensor that contributes most to the headline figure, if above noise. Its effect is not
predicted; the study is run again.

## The perturbation export (§8.5)

`perturbed_extrinsics(project, n, seed)` draws as above and gives arrays (n draws, S sensors):

| array | meaning |
|---|---|
| `d` (n, S, 6) | the factors above |
| `delta_body` (n, S, 4, 4) | T_body' = T_body · Δ, Δ = [Rx Ry Rz \| R_Bᵀ shift] (`core/sim/render` `extrinsic_delta`) |
| `T_parent_body` | the moved body in its mount parent frame |
| `T_parent_optical` | the primary imager's REP-103 optical frame (z forward, x right, y down) in the mount parent frame |
| `T_world_optical` | the same in the world at the joint state given (home by default) |

The exporter writes `perturbations/perturbations.json` (per draw and sensor: `d`, `delta_body`, the mount pose, an
Isaac Lab `CameraCfg.OffsetCfg` with `pos`, `rot_xyzw`, `rot_wxyz` and `convention: "ros"`, and a 3DGS camera with
`position` and camera-to-world `rotation`) and `perturbations/perturbations.npz` (the arrays, sensor uids and names,
seed, scale, distribution), both listed with SHA-256 in the export manifest and the JSON stamped with provenance.

Conventions, read from the targets' source on 2026-11-07: Isaac Lab's `OffsetCfg.rot` is (w, x, y, z) at tag v2.3.0
and (x, y, z, w) at tag v3.0.0 and on main (commit d4be263), `source/isaaclab/isaaclab/sensors/camera/camera_cfg.py`
— both orders are written, named; convention "ros" is forward +Z, up −Y, the optical frame here. The 3DGS reference
code (`utils/camera_utils.py` `camera_to_JSON`, commit 54c035f) writes a camera as its centre `position` and a
camera-to-world `rotation` in the COLMAP/OpenCV axes, the optical frame here.

## Validation (measured 2026-11-07)

The rig of the checks: three static sensors, "Mount A" ± 2 mm ± 1°, "Mount B" ± 1 mm ± 0.25°, and a RealSense D435
calibrated by a hand-eye result written in the check with a correlated 6 × 6 covariance (σ 1.5 mrad, 0.8 mm), so the
RGB imager's offset in the body matters.

| check | reference | measured (full / quick) | tolerance | why that tolerance |
|---|---|---|---|---|
| `tolerance.linear_propagation` | GUM law of propagation, σ² = Σ (∂f/∂x)² σ²ₓ + wᵀ C w, written out (JCGM 100:2008 eq. 10, 13) | 0.0060 / 0.0083 relative error of σ (n = 20 000 / 3 000; uniform 0.0030, normal 0.0060) | 4 standard errors of a sample σ, 4 √(2 / 4n): 0.020 / 0.052 | sampling error of σ̂ for a kurtosis up to 3; the mean is also within 4 σ/√n |
| `tolerance.sampler_ks` | scipy.stats.kstest against uniform, truncnorm and norm; positions, SciPy intrinsic XYZ Euler angles and rotation vectors read back from the moved projects | largest D 0.0235 / 0.032 over 18 marginals (n = 4 000 / 1 500) | critical D at α = 0.001 / 18: 0.037 / 0.061 | Bonferroni over the marginals; the calibration draws' sample covariance is within 4.1 % (Frobenius) of C |
| `tolerance.seed_reproducible` | the same run repeated; the export's separate draw loop | 0 | 0 | the same seed must give the same numbers |
| `tolerance.spearman` | scipy.stats.spearmanr, with ties | 5.6 × 10⁻¹⁷ | 10⁻¹² | rounding only |
| `tolerance.export_roundtrip` | the project moved through core.factors per draw, composed with the imager offset; SciPy reads `rot_wxyz` | 1.5 × 10⁻⁹ m or rad | 10⁻⁸ | the JSON's 9 decimals |

The self-test runs the last three (about 0.8 s, mostly imports).

## Known limits

- The six components of a mount tolerance are independent, and no two sensors are correlated: a shared bracket that
  bends moves its cameras together, which is not modelled.
- The turns are applied in a fixed order (x, y, z about the moving body axes); a rotation-vector reading of the same
  bound differs only to second order in the angle.
- The calibration covariance is the bundle adjustment's Gauss-Newton estimate (optimistic when the residuals are not
  Gaussian, `docs/models/handeye.md`); intrinsics' covariance, joint encoders and link lengths are not drawn.
- The contribution shares treat the components as independent; for a calibration covariance (correlated components)
  they are indicative. Rank correlation finds monotone dependence on a component or its size, not interactions.
- Each draw re-runs every study the metrics need, so n is bounded by the slowest study; a share of 95 % from 64 draws
  has a Wilson interval about ± 6 % wide.
- The export gives optical frames for each sensor's primary imager only (others: apply `delta_body`); the 3DGS world
  pose of a link-mounted camera holds at the joint state of the export only.

## References

`jcgm100_2008`, `jcgm101_2008`, `kolmogorov1933`, `smirnov1948`, `spearman1904`, `wilson1927`, `scipy_stats`,
`scipy_rotation`, `isaaclab_camera_cfg`, `kerbl2023_3dgs` (`dexvantage/validation/references.py`).
