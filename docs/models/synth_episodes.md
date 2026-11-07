# Synthetic LeRobot episodes with ground truth {#synth_episodes}

Module: `dexvantage/core/sim/episodes.py` (`settings_of`, `plan`, `generate`, `batch_generate`). Command: Simulate ▸
Generate synthetic episodes… (`dexvantage/ui/dialogs/synth_episodes.py`, a window job with progress). Batch:
`--synth-episodes OUT_DIR [--synth-settings JSON]`. Check: `groundtruth.synthetic_roundtrip`
(`dexvantage/validation/checks_synth.py`, model `groundtruth`). Tests: `tests/test_synth_episodes.py`. Throughput:
`scripts/measure_synth.py`. Spec §9.1, last bullet: "Generate synthetic episodes in LeRobot format (v2.1 and v3.0)
with ground truth, so the auditor and the training pipelines can be tested."

The generator moves the project's robot, renders every camera through its own lens model with the MuJoCo ground-truth
renderer (`core/sim/render.py`: pinhole, Brown-Conrady and the fisheye models through the distortion remap), writes
the episodes with the LeRobot writer (`core/datasets/lerobot.py`, v2.1 or v3.0) and writes the ground truth beside
them as `dexvantage_truth.json` (`Dataset.truth()` returns it). Faults are injected at seeded, known places. Nothing
in the dataset files marks a fault: the auditor (§8.2) has to find them from the data and can then be scored against
the sidecar.

## Motion

| Source | When | How |
|---|---|---|
| Trajectories | `motion` `auto` and the project has any, or `trajectories` | episode e plays trajectory e mod n of the first trajectory's robot (`core/robot/trajectory.py`: linear in the joints between samples; joints it does not name at home), time-scaled to the episode when `frames` is set, else at its own speed |
| Grasp approaches | `auto` with no trajectory, or `grasps` | home → pre-grasp (`pre_grasp_m` = 80 mm back along the approach) → grasp → hold → pre-grasp. Each waypoint by IK (`ik.solve_batch`, 4 seeds, self-collision off as in the motion-blur study's approaches, the pre-grasp nearest the grasp). Grasps drawn without replacement from the first grasp set with the seed |

Minimum-jerk timing between approach waypoints:

    q(t) = q_k + (q_{k+1} − q_k) · s(τ),   s(τ) = 10τ³ − 15τ⁴ + 6τ⁵,   τ = (t − T_k) / (T_{k+1} − T_k)
    T_{k+1} − T_k = max(0.3 s, max_j |Δq_j| / v_max) · u,   v_max = 0.8 rad/s,  u ~ U(0.85, 1.15) (seeded)

so the joints stand still at every waypoint (tested: |q̇| < 10⁻³ rad/s there). Each episode adds a seeded constant
joint offset δ ~ N(0, 0.02 rad) (gripper joints left out) and is clipped to the limits. Rows: t_i = i / fps,
`observation.state` = q(t_i), `action` = q(t_i + L / fps) with L = `action_lead_rows` = 1 (the commanded next state;
the last rows hold). 0.3 s, 0.8 rad/s, 80 mm, 0.02 rad and L = 1 are DexVantage defaults, not measurements.

## Cameras and timelines

One camera per enabled camera or depth sensor: the scene's primary imager (the RGB imager of a depth camera) at its
stream size, scaled to at most `max_width` = 640 px wide with even sides (yuv420p needs them), or to `size` [w, h].
Frame rate: the stream's (the bandwidth study's stream choice, as `motion_blur.imaging` reads it), else `default_fps`
= 30 fps (a default, said in the sidecar's `fps_source`). The dataset's fps is the largest camera rate as a whole
number (LeRobot stores one). Key: `observation.images.<sensor name>`, lower case, the parenthesised part dropped.

Camera c has slots k = 0 … M − 1, M = ⌊D · fps_c⌋ + 1 over the rows' span D, at nominal times t_k = k / fps_c. Per
slot the generator keeps

    stamp      s_k = t_k + n_k                       (n_k = 0 unless the camera has jitter)
    content    c_k = t_k + Δ + ρ · t_k               (Δ: desync offset, ρ: drift rate; 0 unless injected)

The stored timestamp is s_k; the image shows the robot at q(c_k). Every image gets seeded Gaussian noise of
`noise_dn` = 1 grey level (a default, not a sensor model), so two real frames are never bit-identical while a
duplicate is.

## Faults

Each fault is switchable and has its own generator seeded with (seed, episode, fault), so turning one off leaves the
others where they were (tested). `episodes` (default every episode) and `camera` (default: a seeded choice among the
cameras frame faults have used least) can be set per fault. Frame faults go to interior slots (two from each end)
that no other frame fault uses, a duplicate's previous frame included; when an episode is too short, fewer are placed
and the notes say so. Every figure is a DexVantage default chosen so the fault shows in a short episode.

| Fault | Default | In the data |
|---|---|---|
| `drop` | 2 per episode | slots not stored: a stamp gap of two periods |
| `duplicate` | 2 | the stored frame is the previous stored frame's pixels exactly, at its own stamp; placed where the joints move fastest |
| `desync` | Δ = 0.05 s | c_k = t_k + Δ for one camera |
| `drift` | ρ = 5000 ppm | c_k = t_k + ρ t_k (exaggerated so it shows within seconds) |
| `jitter` | σ = 3 ms | n_k ~ N(0, σ), clipped to ±0.2 period (stamps stay ordered; a regular gap stays < 1.4 periods and a drop gap > 1.6) |
| `exposure_change` | gain 1.4 over 25 % of the frames | a contiguous run of frames times the gain, clipped at 255 |
| `blur` | 2 frames, 9 subframes, E = 1 / fps | the mean of 9 renders at c_k + E (j/8 − ½), j = 0 … 8 (temporal supersampling); placed where the joints move fastest. Other frames are one instant |
| `gripper_hidden` | 3 consecutive frames | in the wrist camera (the first camera on a link of the moving robot), a flat dark rectangle painted over the gripper's pixels grown by 5 % of the width |
| `mount_slip` | from episode ⌊n/2⌋, ΔT = (3 mm, 0, 0), (0, 1.5°, 0) | one camera (default the first static one) renders through T_body_sensor · ΔT; ΔT in the sensor body, its rotation a rotation vector in degrees (the §8.5 mount-tolerance defaults) |
| `usb_reset` | 0.25 s | a contiguous run of round(0.25 · fps) slots not stored (at most a fifth of the episode) |

## The ground-truth sidecar

`dexvantage_truth.json`, schema `dexvantage.truth/1`, in the dataset root:

- `project_digest`, `generator` (every setting, the seed, the defaults used, the codec, the renderer, the robot's joint
  names and gripper links, the conventions in words, the notes) and a `provenance` block;
- `cameras` {key: `sensor_uid`, `imager`, `fps`, `fps_source`, `wrist`, `intrinsics` {model, width, height, K, dist}
  of the stored images, `extrinsics` {frame: the mount's link or `world`, pose: the optical frame (z forward, x right,
  y down) in it, position and quaternion (x, y, z, w)}: the true mount};
- `episodes` {"e": `faults` [{kind, camera, frames, slots, and offset_s | drift_ppm | jitter_s | gain | exposure_s |
  delta | gap_s}], `extrinsics` {key: the mount actually used}, `visibility` {key: `gripper_visible` per stored frame,
  `gripper_px`, `min_px`}, `timing` {key: `slot`, `stamp_s`, `content_time_s` per stored frame}, `rows`, `motion`}.

`frames` are indices into the camera's stored frames (`CameraStream.frames`); for `drop` and `usb_reset` they are the
stored frames right after each gap and `slots` lists the missing slots. The gripper counts as visible when at least
`gripper_min_frac` = 0.2 % of the image (a default) is the gripper's links (the gripper frame's body and every body
below it) in the rendered segmentation, after the occluder.

## Validation

`groundtruth.synthetic_roundtrip` generates two SO-100 episodes (16 rows, 64 × 48 px, every fault on, seed 3) at v2.1
and v3.0, reads them back through `open_dataset` and compares them with the generator's own arrays and the sidecar it
returned. The LeRobot reader itself is checked independently (pyarrow and the ffmpeg command-line tools) by
`formats.lerobot_roundtrip`. Counted as mismatches, tolerance 0:

| Quantity | Tolerance and why | Measured 2026-11-07 |
|---|---|---|
| state, action, time columns | bit-exact after the float32 cast LeRobot stores (the writer casts once) | exact |
| frame counts per camera (reader, generator, sidecar timing) | equal | equal |
| camera stamps | half a 1/90000 s tick (irregular cameras are stored at that time base) | within |
| decoded frame against its generated frame | ≥ 30 dB PSNR, and not closer to a neighbour that differs by more than the codec error (3 dB margin) | worst 33.7 dB (v2.1 and v3.0, AV1) |
| duplicate against the frame it repeats, both decoded | ≥ 30 dB: LeRobot's g = 2 makes every other frame a keyframe, so a repeated picture is coded again | farthest pair 36.2 dB (v3.0), 39.8 dB (v2.1) |
| gaps (≥ 1.5 periods) | exactly before the frames the sidecar's drop and USB-reset faults list | exact |
| sidecar | the read-back document equals the returned one; one visibility entry per stored frame; fault frames in range | equal |

Result: 0 mismatches over 54 frames per version, 1.3 s on the RTX 5080. The tests also check the faults in the pixels
against fresh renders (noise off): a duplicate is bit-identical to its previous frame; a blurred frame has a lower
Laplacian variance than the sharp render at its content time; an exposure frame equals the sharp render times the
gain within 1 grey level; the occluder leaves no gripper pixel; the slipped camera's image differs from the unslipped
render; the desynced camera's content time is stamp + Δ (+ ρ t_k when it also drifts). The same seed gives the same
truth and bit-identical frames.

This check confirms that the files carry exactly the faults the generator injected. Detecting them from the data
alone is the auditor's job and its own checks.

## Throughput

`scripts/measure_synth.py`, every fault on, default settings unless stated (4 episodes, each motion at its own
length), 2026-11-07, Ryzen 9 9900X. "Render" is the wall time of the render calls (MuJoCo through EGL in the worker
process, the transfer and the lens remap); "other" is mostly AV1 encoding (libsvtav1, LeRobot's options), then the
per-frame noise and Parquet.

| Sample, images | GPU (EGL device) | Rows | Camera frames (renders) | Total | Render | Other | Camera frames/s |
|---|---|---|---|---|---|---|---|
| UR5e, 2 × 640 × 360 | RTX 5080 (0) | 544 | 1 048 (1 104) | 14.8 s | 6.2 s | 8.6 s | 70.9 |
| UR5e, 2 × 640 × 360 | Ryzen 9 9900X integrated Radeon (2) | 544 | 1 048 (1 104) | 18.9 s | 10.1 s | 8.8 s | 55.4 |
| UR5e, 2 × 640 × 360, no faults | RTX 5080 (0) | 544 | 1 088 (1 088) | 15.2 s | 6.1 s | 9.1 s | 71.7 |
| SO-100 (grasp approaches), 2 × 640 × 360 | RTX 5080 (0) | 324 | 608 (664) | 9.0 s | 3.4 s | 5.5 s | 67.6 |
| UR5e, 2 × 64 × 48 | RTX 5080 (0) | 544 | 1 048 (1 104) | 1.9 s | 0.9 s | 1.0 s | 545 |
| UR5e, stream size 1280 × 720 and 1920 × 1080, 2 episodes | RTX 5080 (0) | 272 | 524 (552) | 44.4 s | 20.2 s | 24.2 s | 11.8 |

The datasets are 5.8 MB (640 × 360), 0.8 MB (64 × 48) and 8.5 MB (stream size). Render calls hold at most 8 Mpx
(`CHUNK_PIXELS`), so stream-size renders fit in a 4 GB memory cap (48 renders of 1920 × 1080 per call did not).

## Known limits

Images are MuJoCo's OpenGL shading: no auto-exposure dynamics, rolling shutter or sensor noise model; the 1-grey-level
noise only keeps real frames distinct. Blur averages 8-bit images, not linear radiance. The gripper occluder is
painted in the image (no body, no shadow, no depth). The motion is kinematic: no dynamics and no tracking error
besides the action lead; grasp approaches are not checked against obstacles and do not close the gripper. One robot
moves and its joints are the state vector; other robots stay at home. One imager per sensor is recorded. Drift is a
constant rate and jitter Gaussian; faults are placed independently and several on one camera combine (desync, drift
and jitter add; an exposure run can include a blurred frame). The drift default (5000 ppm) is far above a real
oscillator's so that it shows within one episode.
