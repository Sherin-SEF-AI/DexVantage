# Dataset and episode auditor, plan against recording

Spec §8.2 (the auditor), §8.3 (plan against recording) and the FiftyOne bridge of §9.4. Code: `core/audit/`
(`episode.py` per-episode checks, `campaign.py` the campaign and mount slip, `slip.py` the step-against-gradual
classifier ported from Vantage `core/online_calib.py`, `plan.py` what the plan says about a recorded camera,
`report.py` JSON, HTML and FiftyOne), the studies `core/studies/audit.py` and `core/studies/plan_vs_recording.py`, the
pages `ui/pages/audit.py` and `ui/pages/plan_vs_recording.py`, the checks `validation/checks_audit.py`.

Every check reads only the dataset (core/datasets: LeRobot v2.1 and v3.0, RLDS, ROS 2 bags) and, where it helps, the
project's plan. A synthetic dataset's ground-truth sidecar is never read by the auditor; the tests and the validation
check audit a copy of the dataset with the sidecar removed and compare afterwards.

## Requirements

Every limit is a DexVantage default, user-settable in `project.data['requirements']` (the page's settings, or the
project file). None is a standard.

| Requirement | Default | Compared |
|---|---|---|
| `max_dropped_frames` | 0 | dropped frames per camera |
| `max_duplicate_frames` | 0 | duplicated frames per camera |
| `fps_tolerance_pct` | 1 % | measured against stated frame rate |
| `max_jitter_ms` | 1 ms | robust spread of the stamps about a regular grid |
| `max_gaps` | 0 | gaps of `usb_gap_s` (0.1 s) or longer per camera |
| `max_offset_ms` | 25 ms | camera–joint offset, failing only beyond 3 σ too |
| `max_drift_ppm` | 2000 ppm | drift within the episode, failing only beyond 3 σ too |
| `max_exposure_changes` | 0 | exposure or gain steps over `exposure_step` (15 %) |
| `max_blur_frames` | 0 | frames under `blur_ratio` (0.85) |
| `max_gripper_hidden_frames` | 0 | wrist-camera frames whose gripper correlation is under `gripper_ncc` (0.5) |
| `action_lag_rows` | 0 … 3 rows | row lag of the action against the state |
| `max_mount_shift_deg` / `_mm` | 0.5° / 2 mm | mount change from the first episode's estimate |

An episode's verdict: **pass**; **fix** (only an offset or drift fails: re-stamping corrects it); **recalibrate**
(only the mount moved: the episodes need the new extrinsics); **drop** (drop it or re-record it).

An episode that cannot be read (a damaged Parquet, video or bag file) or that holds no joint states and no camera
frames (a bag with no readable messages) fails the requirement **Episode readable** (`readable`, no limit to set):
its verdict is drop, its flag `unreadable`, and the reader's sentence is kept in the episode's `error`; the other
episodes are audited as usual and it is left out of the mount series and the FiftyOne media. When no episode can be
read the study refuses with that sentence (product review, 2026-11-07: a truncated MCAP audited as "ready" and a
damaged Parquet stopped the audit with a traceback). A recording whose joint states do not match the project's robot
(by name or by count) is audited from the data alone: the plan's cameras are not posed (said in `plan_notes`).

## Per-episode checks (`core/audit/episode.py`)

**Timing.** P = 1 / the stated fps (else the median spacing). A spacing d ≥ 1.5 P is a gap of round(d/P) − 1 frames:
a dropped frame below `usb_gap_s`, a USB reset or stall from it on. With slots n_k = Σ round(d/P) the stamps are fitted
t_k = a + P′ n_k by least squares; the measured rate is 1/P′, the jitter 1.4826 · MAD of the residuals. With device
timestamps, (host − device) is fitted against the device time: the slope is the clock drift in ppm.

**Duplicates.** Exact: the frame's BLAKE2b digest equals the previous frame's. Near: on frames low-passed with a
Gaussian (σ 1.5 px), the change e_k = mean|I_k − I_{k−1}| under `dup_ratio` (0.3) of the median change of the eight
frames around it (itself and the next, which carries the double step, left out). Both are tested only where the scene
should change: the plan's sample points move at least 1 px between the two frames (or, without a plan, the joints move
at 10 % of their top speed). A stale frame re-encoded by LeRobot's codec (g = 2: every other frame a keyframe) is not
bit-identical (36–40 dB from the frame it repeats), so equality alone misses it, and on a still scene identical
decoded frames are normal (counted as `identical_still`, not as faults).

**Exposure and gain.** Per pair, the median over pixels (both in 16…200) of ln((I_k + 1)/(I_{k−1} + 1)): a global
gain multiplies every pixel, so the median is ln(gain) while the moving part of the scene is a minority. |median| >
ln(1 + `exposure_step`) is a step; steps of opposite sign pair into runs. A recorded exposure or gain column is used
instead when the episode has one for the camera.

**Motion blur.** The variance of the Laplacian (Pech-Pacheco et al. 2000), normalised by the mean squared intensity
(gain-invariant), over the pixels that move (|I_{k+1} − I_{k−1}| > 8 grey levels, dilated), divided by the smaller of
the same measure's mean over k ± 1 and over k ± 2. The second pair is there because frames between keyframes keep less
fine detail (about 0.6 of their neighbours' Laplacian energy, measured): each frame is compared with frames of its own
kind. Cameras on the robot are not tested by default (`blur_on_moving_cameras` 0): on synthetic wrist cameras the
codec's loss of detail on fast motion gave dozens of false alarms per episode.

**Camera–joint desync.** The offset δ that lines the image motion up with the joints'. Observed: e_k (above). Predicted
(with a plan camera): at points sampled on the robot's surface (and, for a camera on a link, the static scene), the
linearised brightness-constancy change |∇I_{k−1}(u) · Δu|, Δu the point's image displacement between the two frames'
times shifted by δ, posed at the recorded joint states through the plan's lens and mount. Without a plan: the joint
speed |q̇| × spacing. δ maximises Spearman's rank correlation of observed and predicted over a 4 ms grid within
±0.25 s, refined by a parabola. The rank correlation needs only a monotone relation (frame differences saturate and
depend on texture), which is the robust fit; time-delay estimation by cross-correlation is Knapp and Carter 1976.
σ_δ: the spread of the peak over a moving-block bootstrap (Künsch 1989; blocks of 10 pairs, 40 resamples). Drift:
δ(t) = δ0 + d (t − t̄) by the same correlation over a grid of d (±30 000 ppm, 1000 ppm steps), σ_d by the bootstrap but
at least σ_δ √12 / T (a drift cannot be known better than the offset over the episode's span T). Pairs next to a gap,
a duplicate, an exposure step or a blurred frame are left out.

**Gripper hidden.** For a camera on the robot with gripper links in the plan: the plan's gripper mask (core/camrender's
z-buffer of the gripper links alone, at the recorded joint state, cached by the gripper's pose relative to the camera)
and the zero-mean normalised cross-correlation of the frame inside it with the median of the frames within ±15. NCC
under `gripper_ncc` means something else is there. NCC is gain-invariant, so an exposure step does not read as
hidden. Which mask: core/camrender's, not core/occlusion's cube maps (the pixels are what is compared).

**Action alignment.** The row lag k in −10…10 minimising the RMS of action[t] − q[t + k] (joints matched by name, else
by order), refined by a parabola.

## Campaign checks and mount slip (`core/audit/campaign.py`, `core/audit/slip.py`)

One mount estimate per camera per episode (setting `mount_source`, default `auto`: a board when one is found, else
registration): (1) a calibration board in the frames (core/calib/detect.detect_and_solve on up to 8 frames, the board's
placement from its record); (2) with `mount_source` `silhouette`, for a static camera, the link-silhouette fit
`core/calib/render_calib.estimate`, called only through that API, absolute against the plan (about 3 s per episode and
camera on the SO-100 sample; it refuses frames whose shape differs from the plan's imager, and the registration is then
used, with the reason in the notes; measured in tests/test_audit.py: 1.15° recovered for an injected 1.5° pitch on 90
frames at 320 × 180); (3) image registration: the episode's median image over its static pixels (temporal σ under 6 grey levels:
for a static camera the background, for a wrist camera the gripper parts fixed in its view) registered to the first
episode's by a pure rotation A = Exp(ω) of the optical frame (each reference ray r looked up at project(Aᵀ r), any lens
model), with a gain and an offset, by Gauss-Newton on a three-level pyramid with Huber weights (Lucas and Kanade 1981;
Baker and Matthews 2004), over the pixels with a grey gradient over 1 per px; σ by a block bootstrap (16 × 16 px blocks,
20 resamples). The rotation is reported in the sensor's body frame, ΔR = R_so A R_soᵀ, as a rotation vector in degrees
(render_calib's and the synthetic mount_slip fault's convention). Registration measures rotation only, relative to the
first episode.

Each axis series (over the episodes, in days when the dataset records when each was taken, else in episode order)
goes through Vantage's model selection, ported unchanged but for two constants: y_i = level + Σ size_s [i ≥ j_s] +
slope (t_i − t_0) + e_i, terms added greedily while they lower BIC = n ln(RSS/n) + p ln n by more than 6 (Kass and
Raftery 1995, "strong"), a pair of steps tried when no single one pays, Huber-weighted final fit, terms within 3 σ (or
under 0.1°) pruned, slopes checked with each episode left out. A step back to the earlier level is an **excursion**,
one that stays a **step** (slip), a significant slope **gradual** drift. Changed from Vantage: two estimates on either
side of a step (MIN_SEG 2, Vantage 3: a campaign has few episodes), and the residual-scale floor as half the median
stated σ (Vantage: 0.5° fixed).

## Plan against recording (`core/studies/plan_vs_recording.py`)

Per recorded camera, matched to the plan by name (or the dataset entry's mapping): present; resolution (same, scaled
with the same aspect ratio within 1 %, or different); frame rate against the bandwidth study's stream choice (or "not
stated by the plan"); stored intrinsics; the mount (stored extrinsics against the plan's optical frame, else the
audit's estimates: absolute from a board or the silhouette, else the change over the campaign); sync (the audit's
offsets against the limit, and against the plan's own prediction: the time-sync study's skew interval of the stream,
negated because the audit's offset is content time minus stamp and the skew stamp minus exposure, widened by 2 σ_δ;
"as planned" or "outside the prediction" counts as a comparison when the plan's budget is complete, while an
incomplete budget, a lower bound, is shown without a verdict); visibility at up to 8 frames of up to 4 episodes: the grasp study's prediction (its Evaluator at the recorded
joint state: at least 10 % of the contact points visible) against the recorded frame (a wrist camera: the gripper
correlation; a static camera: at least 30 % of the plan's gripper-mask pixels differ from the episode's median image by
more than 25 grey levels). The headline is the share of these comparisons that match.

## FiftyOne export (`core/audit/report.py`)

From the page (Export to FiftyOne…) or in batch with the audit:
`dexvantage project.dexv --audit audit.json --dataset DATASET --audit-html audit.html --fiftyone fiftyone/` (the HTML
report and the FiftyOne folder; a folder that is not empty is refused in one sentence).

FiftyOne's importer for its own format (`FiftyOneDatasetImporter`) hands a folder whose `metadata.json` has no
`version` key to `LegacyFiftyOneDatasetImporter`, read from the FiftyOne source on 2026-11-07
(github.com/voxel51/fiftyone, branch develop: `fiftyone/utils/data/importers.py`, `fiftyone/core/sample.py`). The
export writes that layout: `metadata.json` {name, media_type "video", info}, `samples.json` {"samples": [...]}, one
`frames/<sample>.json` {"frames": {"1": {...}}} per sample, and `data/<clip>.mp4` (each camera of each episode re-encoded
as H.264 at its own timestamps, so frame k of the clip is frame k of the audit). Sample tags: `audit:<verdict>` and the
fault kinds found on that camera; sample fields: the episode's index and id, the camera, the offset and σ, drift,
jitter, measured fps, counts of dropped, gap, duplicated, blurred and gripper-hidden frames, exposure steps, the mount
shift, the action lag; frame fields: time, change, sharpness ratio, gain step, gripper correlation, the gap before,
and the per-frame flags. The `fiftyone` package is optional and not installed here, so FiftyOne's own importer has
not been run on these files (the test imports them with FiftyOne when it is installed); load with
`fiftyone.Dataset.from_dir(folder, dataset_type=fiftyone.types.FiftyOneDataset)`.

## Validation (measured 2026-11-07, `--validate --models data`, 53 s, 2.4 GB peak)

A seeded campaign (`checks_audit.SETTINGS`): the SO-100 sample, 6 episodes × 240 rows at 30 fps, 320 × 240, each fault
injected into two episodes (the other four clean for it), mount slip from episode 3 (body pitch 1.5°, 3 mm). Rendered by
MuJoCo through EGL on the RTX 5080 (device 0, the generator's `auto`); the audit itself runs on the CPU (numpy, OpenCV).

`audit.fault_recall` (recall over injected (episode, camera) faults, false-positive rate over the 10 clean pairs per
kind; floors stated in `RECALL_FLOOR`, ceilings in `FP_CEILING`): passes.

| Kind | Recall | FP rate | Frames found | Floor |
|---|---|---|---|---|
| drop | 1.0 | 0.0 | 4/4 | 1.0 |
| usb_reset | 1.0 | 0.0 | 2/2 | 1.0 |
| jitter | 1.0 | 0.0 | — | 1.0 |
| exposure_change | 1.0 | 0.0 | 120/120 | 1.0 |
| gripper_hidden | 1.0 | 0.0 | 6/6 | 1.0 |
| mount_slip | 1.0 | 0.0 | — | 1.0 |
| duplicate | 0.5 | 0.0 | 2/4 | 0.5 |
| desync | 0.5 | 0.0 | — | 0.5 |
| blur | 1.0 | 0.1 | 2/4 | 0.5 |
| drift | 0.0 | 0.0 | — | 0 (not resolved) |

Duplicates on the third-person camera are missed (the robot is small and the change drop is near the codec's noise);
desync is found on one of its two episodes (47 ± 17 ms is within 3 σ of the 25 ms limit); drift at 5000 ppm moves the
offset by 40 ms over 8 s, about the estimator's own scatter, so it is reported, never claimed.

`audit.desync_offset`: the largest error over the six static-camera estimates is **16.9 ms** against 20 ms (two thirds
of a frame period). Estimates against the truth's mean offset: 44.6/50.0, 25.3/19.9, 47.2/49.9, 5.2/0.1, −7.7/0.0,
36.8/19.9 ms (σ 9–20 ms). The wrist camera, on the robot, is listed and not counted: −46, −49, 1, −43, −44, 7 ms against
0 (σ 20–31 ms: its predicted motion comes from the few edges of a textureless synthetic scene). The 3 σ rule keeps
those from failing the requirement.

`audit.slip_vs_vantage`: twelve seeded series (steps, returning shifts, slopes, noise, a step with a slope, outliers)
through Vantage's `select_model` (recorded by `scripts/record_vantage_slip.py` from the Vantage checkout at commit
89eeab82, 2026-11-07; `validation/VANTAGE_SLIP.json`) and the port: the same terms on all twelve, levels, sizes and
slopes equal to **0** (bit for bit) against 10⁻⁹.

`audit.plan_vs_recording`: both cameras matched to their sensors; the slipped camera's recovered rotation change
(0.19, 1.47, −0.06)° against the truth's (0, 1.5, 0)° (rotation vectors in the sensor's body frame): **0.20°** apart,
against 0.3°. The sizes alone (1.48° against 1.5°) differ by 0.020°, which is what this check first compared; that hid
the 0.19° about the other axes. The registration fits a pure rotation, and the slip also moves the camera 3 mm, which at
the scene's 0.56 m subtends up to 0.31°: the tolerance states that bound.

## Known limits

LeRobot stores camera timestamps on the nominal grid (its frames are decoded at the rows' times), so drops and jitter
that were resampled away before writing do not show (a resampled drop shows as a duplicate). RLDS without an fps has
step counts for times. The desync estimate needs motion of varying speed; it is resolved to about ±17 ms on static
cameras and poorly on cameras on the robot over textureless scenes. In-episode drift under several thousand ppm is not
resolved (real clocks drift tens of ppm: the campaign's per-episode offsets show that). Blur is relative to the
neighbouring frames (a whole episode blurred alike is not flagged) and off for cameras on the robot. Duplicates of a
still scene cannot be told from stillness. The gripper check and the predicted motion need the plan's mount and lens to
be close to the real ones. Mount registration measures rotation only, relative to the first episode, and needs static
texture; a campaign of fewer than five episodes cannot be classified (step against gradual), only compared with the
limit. Plan-against-recording visibility from images is evidence, not segmentation (on the guide's worked example the
third-person camera agreed with the grasp study on 75 % of 32 frames, the wrist camera on 100 %: a static camera's median
image holds the gripper where it rests, and the grasp study's contact points are a stricter test than "the gripper is
in view").
