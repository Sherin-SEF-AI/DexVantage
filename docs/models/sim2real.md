# The sim-to-real gap (research tools part 2)

`dexvantage/core/sim2real/` (`gap.py` the analysis and Apply, `measures.py` the image, depth and distribution
measures, `inject.py` recordings with known gaps), `dexvantage/core/studies/sim2real.py` (the study, `--sim2real`),
`dexvantage/ui/pages/sim2real.py` (Data ▸ Sim-to-real gap), `dexvantage/validation/checks_sim2real.py` (seven checks,
model `sim2real`), `tests/test_sim2real.py`.

A recorded episode is rendered again in MuJoCo through the project's rig: each camera's lens model and mount, the
robot at the recorded joint states (`core/sim/render.Renderer`, the same renderer the ground truth, the synthetic
episodes and the silhouette calibration use). The renders are compared with the recorded frames camera by camera. The
result says how large the gap is, which corrections explain how much of it, and which change closes each part.

## API

    gap.analyse(project, dataset, settings=None, *, progress=None, cancel=None) -> dict   (dataset: a path or a Dataset)
    gap.settings_of(project, given=None) -> dict
    gap.apply_change(project, change) -> [store sections]      gap.describe_change(project, change) -> str
    inject.make_gap_dataset(project, folder, gaps, settings=None, *, progress=None) -> dict
    measures.*                                                  (no rendering; listed in measures.py's docstring)

    import dexvantage as dv
    rig = dv.sample('ur5e')
    res = rig.sim2real('path/to/dataset')            # a Result; res['causes'], res['apply'], res.headline
    rig2 = rig.apply_sim2real(res['apply'])           # a NEW Rig with the top suggestion made
    rig.gap_dataset('out/gaps', {'brightness_dn': 20, 'latency_s': 0.04})   # a recording with known gaps

Batch: `dexvantage project.dexv --sim2real OUT.json --dataset PATH` (the specification's `--sim2real dataset_path`;
the study registry's flags take the output file, and `--dataset` names the recording, as for the other data studies).
Without `--dataset` the study reads its setting `dataset`, else the audit's, else the project's first dataset entry.
The assistant reads it with `get_sim2real` (a long study: it is proposed as a plan step in the app).

## Method (per camera)

Cameras are matched by `render_calib.match_stream` (the stream's `sensor_uid`, else its name). Frames are compared at
a working width (default 320 px), resized with area averaging; the render is made at the recording's size (up to
640 px) and resized the same way, so both images carry the same resampling.

1. **Latency** τ (content time − stamp). At the 8 frames where the joints move fastest the robot is rendered at
   stamp + τ for τ in [−150, 150] ms in 10 ms steps. The score at τ is the mean zero-mean normalised
   cross-correlation of the gradient magnitudes (Sobel of the image smoothed with σ = 1 px). It is taken over the
   pixels the robot covers at some τ, dilated 5 px, for a static camera, and over the whole image for a camera on the
   robot. A normalised correlation of gradients does not depend on gain or offset, so an exposure gap does not move
   the peak. A parabola through the peak gives the sub-step value; σ_τ is the spread of the peak over 200 bootstrap
   resamples of the frames. A latency the time-sync settings already hold (`latency_s`, stamp minus exposure) is the
   plan's: frames are rendered at stamp − e, and the search finds what is left.
2. **Mount offset** (static cameras): the silhouette calibration (`render_calib.estimate`: chamfer and IoU alignment,
   docs/models/render_calib.md) on the episode with the stamps moved by τ. A desync otherwise reads as a mount error,
   which is render_calib's own known limit. When the mount is observable, the latency search runs again through the
   recovered mount, because a mount offset biases it (measured −8.9 ms on a 3 mm, 1° offset, 0.0 ms after).
3. **The gap and its causes.** The frames (8, farthest apart in joint space) are rendered with every combination of
   the geometric corrections found (latency, mount). For each combination S of all corrections the image difference is

       d(S) = mean over frames, pixels and channels of |recorded − f_S(render_S)|      (grey levels, DN)

   where f_S applies the photometric corrections in S: a Gaussian blur σ_b (only when the recording's sharpness is
   under 0.9 of the render's), then a per-channel gain and offset. Each correction's share is its Shapley value
   (Shapley 1953):

       φ_i = Σ_{S ⊆ P∖{i}} |S|! (n − |S| − 1)! / n! · (d(S) − d(S ∪ {i})),    Σ φ_i = d(∅) − d(P)

   This is what correction i removes, averaged over every order of applying the corrections. With four corrections
   it takes 4 render sets (latency × mount) and 16 image evaluations.
4. **Photometric figures** (recording against the geometry-corrected render): the mean per channel, RMS contrast,
   the Earth mover's distance of each channel's 256-bin histogram (W₁ = Σ_k |F_a(k) − F_b(k)| in DN; Rubner et al.
   2000) and sharpness. Sharpness is var(∇²I) − 20 σ_n²: the 4-neighbour Laplacian (Pech-Pacheco et al. 2000), less
   the share 20 σ² that white noise adds to it. Noise is σ_n = √(π/2) · mean|I ∗ N| / 6 with N = [[1, −2, 1],
   [−2, 4, −2], [1, −2, 1]] (Immerkær 1996), over the pixels below the 90th percentile of the Sobel magnitude
   (Tai and Yang 2008's edge exclusion). The factors come from the operator: on white noise of σ the response is
   Gaussian with σ√36 = 6σ, and E|r| = 6σ √(2/π).
   The blur σ_b is found by bisection, so the render's summed sharpness after blurring equals the recording's.
   The gain and offset are a least-squares line through the two images' quantile pairs (2nd to 98th percentile),
   leaving out pairs where either quantile is clipped. The fit is repeated three times on the corrected and clipped
   render. A pixel-wise fit is biased towards gain 0 when the images are not aligned or not equally sharp
   (regression dilution: measured gain 0.86 against a true 1.0 on a blurred recording). The quantile fit is not:
   1.00 there, and the offset +20.0, +20.0, +20.1 DN for an injected +20. The brightness change is reported at mid
   grey, g·128 + o − 128 DN, and in stops, log₂((g·128 + o)/128).
5. **Silhouette IoU** (static cameras). No DexVantage reader exposes segmentation masks, so the robot is segmented
   by background difference: pixels more than 25 DN from the median of the compared frames, opened 3 × 3
   (`render_calib.foreground`). The same rule is applied to the recorded frames and to the rendered frames, so both
   masks hold the robot where it moves, and they are compared like for like. The figure is the mean IoU before (as
   planned) and after every correction. A camera on the robot has no static background, so it gets no IoU.
6. **Depth** (depth sensors). The predicted valid-pixel share is the rendered depth within the device's stated
   minimum Z at the stream and its maximum range, each only when the library states it (else "not stated by the
   maker"). When the episode carries a decoded depth stream for the camera (a `CameraStream` with meta
   `{'depth_of': <colour key>, 'depth_scale_m': s}`, which a plugin reader can provide), the study reports the
   measured share and, on the robot's pixels where both are valid, the error measured − predicted (median, RMS, P95).
   DexVantage's own readers list depth streams in `meta['depth_streams']` but do not decode them, so this side reads
   "not measured" and names the streams.
7. **Timing**: the stamps' jitter (`audit.episode.timing`: 1.4826 × MAD of the residuals about a fitted regular
   grid) and the latency, both against the time-sync study's prediction for the stream
   (`plan_vs_recording.sync_prediction`: the predicted offset is the negated skew interval; an incomplete budget is
   a lower bound and gets no verdict).
8. **Distribution distance**: the unbiased kernel MMD² (Gretton et al. 2012, eq. 3) with
   k(x, y) = exp(−‖x − y‖²/(2σ²)) on 32 × 24 RGB thumbnails in [0, 1]. σ comes from the median heuristic, and the
   permutation p-value from 200 permutations (seeded). It is reported as planned and with every correction. It is
   **not FID**. FID needs a trained Inception network's features over thousands of images; torch and those weights
   are not part of DexVantage, so no `fid` extra was added and the result says FID is not computed.

### Causes and the changes that close them

Ranked over the cameras by the DN each removes. A cause under 5 % of its camera's gap, or removing under 1 DN, gets
no action (both are defaults).

| cause | Apply (one undo step) | the page that owns it |
|---|---|---|
| latency | `analysis.studies.time_sync.latency_s[uid] = −(planned + τ)` with its source (stamp minus exposure, relative to joint_states) | Time sync |
| mount | the sensor's mount pose from the silhouette calibration (a locked sensor is refused) | Silhouette calibration |
| exposure | the render's per-channel gain and offset for the camera (`analysis.studies.sim2real.appearance[uid]`) | Sim-to-real gap |
| lighting | the same gain (within 10 %) in every camera with an exposure cause: the scene's light. The project's lights' `dim` is scaled when it has lights, the scale fits (dim ≤ 1) and the cameras share no offset of `min_reduction_dn` (1 DN) or more (a scale of the lights gives a gain, not a constant offset); otherwise each camera's render gain and offset are set | Lighting |
| blur | the render's blur σ (composed in quadrature) | Motion blur (focus, exposure time) |
| noise | the render's noise σ, √(σ_rec² − σ_ren²); listed, not ranked: noise adds to d rather than closing it | Sim-to-real gap |

`appearance` is the render's camera response, applied before every comparison: blur, then gain and offset, then
seeded Gaussian noise. After an Apply, the part of the gap that the change closes is gone on the next run. A latency
entered in the time-sync settings is likewise rendered as planned. Status: close to real up to 4 DN, a small gap up to
8 DN, a large gap up to 16 DN, a very large gap beyond (DexVantage defaults, not standards).

## Validation (measured 2026-11-07, `--validate --models sim2real`)

| check | reference | tolerance | measured |
|---|---|---|---|
| `sim2real.emd_vs_scipy` | `scipy.stats.wasserstein_distance`, 200 random pairs | 1e-9 DN | 1.7e-13 DN |
| `sim2real.mmd_hand_case` | X = {0, 1}, Y = {2, 4}, σ = 1 written out | 1e-12 | 1.1e-16 |
| `sim2real.mmd_gaussians` | the closed form for two 4-D Gaussians by eigenvalues; 400 unbiased estimates (m = n = 40) | 4 standard errors (0.0068) | 0.00063 (the module's own closed form agrees to 1e-10) |
| `sim2real.noise_known` | σ = 0.5–8 DN added to a structured image | 5 % | 1.2 % |
| `sim2real.blur_known` | `scipy.ndimage.gaussian_filter` σ = 0.8, 1.5, 3 px | 0.1 px | 0.0005 px |
| `sim2real.shapley_brute_force` | the mean marginal over all 24 orders | 1e-12 DN | 8.9e-16 DN |
| `sim2real.injected_recovery` | four UR5e recordings, one gap each on the overhead D435 (LeRobot v3.0, AV1 codec) | error ÷ tolerance ≤ 1, each ranked first | 0.68 |

The injected-recovery tolerances and what was measured:

- **Brightness +20 DN.** Within 3 DN at mid grey, because clipping at 255 and the codec move the fit. Measured
  +20.03, +19.97, +20.12 DN per channel, gain 0.99. Exposure ranks first.
- **Blur σ 1.5 px.** Within 0.3 px, because the codec blurs too. Measured 1.42 px; the clean recording reads
  sharpness ratio 0.96, under the 0.9 threshold for no blur. Blur ranks first.
- **Latency 40 ms.** Within 5 ms (half the grid step). Measured 39.97 ms, σ 0.4 ms. Latency ranks first.
- **Mount 3 mm along x, 1° about y.** Within 0.3° and 3 mm: the silhouette calibration's accuracy on this camera
  (3 mm is about 2 px of the 320 × 180 image at the robot's depth; `render_calib.md` gives 1.1–1.8 mm per 2 px at
  640 × 360). Measured 0.13° and 2.0 mm off (recovered 1.08°). Mount ranks first. The 3 mm allowance equals the
  injected 3 mm, so the position part alone would pass with no offset found: it is the angle (0.13° off an
  injected 1°) that shows the mount was recovered. The 3 mm translation is not resolved at this image size.

The checks take 14 s together (the recovery check renders and encodes four recordings), and the self-test line runs
the five cheap ones in under 1 s.

## Measured on the UR5e sample

All measurements are on synthetic recordings, so they test the method, not a real camera.

**Clean recording** (`inject.make_gap_dataset` with no gap; 40 rows, 320 × 180, AV1). The gap is 1.8 DN on the
overhead D435 and 1.1 DN on the wrist D405 ("close to real"). This is what the renderer, the generator's 1 DN pixel
noise and the video codec leave. Latency reads +0.15 and −0.26 ms, the overhead silhouette IoU 0.94, and MMD² −0.105
(p 0.99). Nothing is suggested. The analysis takes 2.1–2.9 s (about 330 renders and two cameras).

**All four gaps on the overhead camera at once** (+20 DN, blur 1.5 px, 40 ms, 3 mm and 1°). The gap is 24.4 DN.
The Shapley shares are exposure 14.6, mount 3.2, blur 2.4 and latency 2.1 DN, with 2.1 DN left after every
correction (91 % explained). The latency reads 39.3 ms ± 0.5, the blur 1.43 px and the offset +18.7 DN. The mount
reads 6.7 mm and 1.31° against a true 3 mm and 1°: the blur and the offset degrade the silhouette edges. The IoU goes
from 0.53 to 0.89, and MMD² from 0.142 (p 0.015) to −0.084. The run takes 7–8 s with the silhouette calibration,
and its largest process uses 1.27 GB.

Applying the top suggestion five times (the API's `apply_sim2real`, each a new rig) took the gap from 24.4 DN to
10.0, 7.6, 4.6 and 2.1 DN. The changes, in order, were exposure, mount, latency (entered in the time-sync settings)
and blur, and then nothing was left to suggest.

## Known limits

- The render is MuJoCo's OpenGL shading of the project's model of the cell, so a real recording differs by much that
  no correction here models: textures, shadows, cables, the background, objects the project does not have. The
  corrections explain the part they can, and the rest is reported as left.
- One episode per run, and a constant latency over it (no drift). The latency needs the robot to move.
- The mount is the silhouette calibration's, with its limits: static cameras only, a robot that looks like its model,
  degraded by blur.
- A static camera's observed silhouette is only the moving part of the robot. There is no IoU for cameras on the
  robot.
- The exposure fit is global per channel (no vignetting or tone curve), and the blur is one isotropic Gaussian (no
  motion-blur direction).
- Noise is measured after the codec, which removes some of it.
- Depth is compared only where a reader decodes it, which DexVantage's own readers do not yet.
- Shapley shares are exact attributions over the corrections tried, but corrections interact (a mount fitted under a
  wrong exposure), so a share is not a measurement of a physical cause.
- MMD on eight frames per side has little power (its p-value says so), and it is not FID.

## References

Immerkær 1996 (`immerkaer1996`); Tai and Yang 2008 (`tai_yang2008`); Gretton et al. 2012 (`gretton2012`); Rubner et
al. 2000 (`rubner2000`); Shapley 1953 (`shapley1953`); Pech-Pacheco et al. 2000 (`pech_pacheco2000`); SciPy's
`wasserstein_distance` (`scipy_wasserstein`). The full entries are in `dexvantage/validation/references.py`. The
Immerkær paper's full text is paywalled: its bibliographic record and abstract were read on 2026-11-07, and the
estimator's constants are derived again in `measures.py` and checked on images of known noise.
