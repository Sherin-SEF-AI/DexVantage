# Exporters, part B: Isaac Lab and Isaac Sim, LeRobot and the policy codes, reports

`dexvantage/core/exporters/isaac.py` ('isaac'), `lerobot.py` ('lerobot'), `report.py` ('report') with `_pdf.py`;
the round trips in `dexvantage/validation/formats_isaac_lerobot.py`, registered in `checks_formats.py` as
`formats.isaac_usd`, `formats.isaac_config`, `formats.lerobot_export` and `formats.report`. Spec §9.2 (exporters),
§9.4 (bridges), §8.5 (randomisation bounds), §7.13 (the training-config recommendations). A worked example is in
[../guide/exporters_b.md](../guide/exporters_b.md).

Each runs through the export registry (`core/exporters/__init__.py`): File ▸ Export… and `export(project, out_dir,
['isaac', 'lerobot', 'report'], options)`; every JSON file gets the provenance block, except those a target parser
refuses extra keys in (below), and the manifest holds every file's SHA-256.

**Contract change (additive).** An exporter may return `'unstamped': [rel]`: those JSON files are hashed in the
manifest but not stamped, and the format's row lists them. LeRobot's `train_config.json` is read by draccus, which
refuses unknown keys, and LeRobot's `DatasetInfo.from_dict` warns on them, so the train config and the info.json
templates are unstamped.

## Isaac Lab and Isaac Sim — `isaac.py`

**Sources** (read 2026-11-07, https://github.com/isaac-sim/IsaacLab):

| Release | Commit | What it fixes |
|---|---|---|
| v3.0.0 (the current release) | `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4` | `CameraCfg.OffsetCfg.rot` is **(x, y, z, w)**; `TiledCameraCfg` is deprecated ("Use CameraCfg directly"); `PinholeCameraCfg.distortion` takes `OpenCvPinholeDistortionCfg` (fx, fy, cx, cy, image_size, k1–k6, p1, p2, s1–s4) or `OpenCvFisheyeDistortionCfg` (k1–k4, max_fov); `spawn_camera` writes them as `omni:lensdistortion:<model>:*` attributes with the `OmniLensDistortionOpenCv{Pinhole,Fisheye}API` schema; `Camera._initialize_intrinsics` gives a prim without such a model fx = fy = W·focalLength/horizontalAperture and cx = W/2, cy = H/2, and uses the authored fx, fy, cx, cy when it has one |
| v2.3.2 | `37ddf626871758333d6ed89cf64ad702aef127d0` | `OffsetCfg.rot` is **(w, x, y, z)**; `TiledCameraCfg` current; no OpenCV distortion fields |

**Frames.** A camera's offset is its imager's optical frame (REP-103: z forward, x right, y down) in its parent: the
robot link the sensor is mounted on, or the environment for a static mount; Isaac Lab's convention "ros" (forward
+Z, up −Y) is the same frame. The prim path is `{ENV_REGEX_NS}/Robot/<link>/<camera>` (the articulation's prim name
is an option) or `{ENV_REGEX_NS}/<camera>`.

**USD lens.** Isaac Lab's own rule (`utils/sensors.py convert_camera_intrinsics_to_usd`: pixel size 1/W when no
focal length is given): horizontalAperture = 1, focalLength = fx/W, verticalAperture = (H/W)(fx/fy). GfCamera's
frustum puts the window on the image plane at (aperture/2 ± offset)/focal, so with DexVantage's pixel convention
(OpenCV: pixel centres at integers, image from −0.5 to W − 0.5):

    fx = W f / hAp,   cx = (W − 1)/2 − W·hOff/hAp
    fy = H f / vAp,   cy = (H − 1)/2 + H·vOff/vAp        (USD's y is up, the image's v down)

so hOff = hAp((W − 1)/2 − cx)/W and vOff = vAp(cy − (H − 1)/2)/H. A USD camera looks along −Z with +Y up: its
transform is the optical frame times diag(1, −1, −1). The stage is metres, Z up, `defaultPrim` /DexVantage; static
cameras under /DexVantage/World, link-mounted ones under /DexVantage/<robot>/<link>, an Xform at the link's pose at
the export's joint state (home by default) so the file shows the rig. clippingRange (0.005, 50) m are DexVantage's
labelled defaults (the MuJoCo scene's). With usd-core (`pip install "dexvantage[usd]"`, pinned `usd-core==26.8`) pxr
writes the file; without it a text writer of the same prims and attributes does, and the notes say which.

**Lens models.**

| DexVantage lens | Isaac Lab v3.0.0 | Exact? |
|---|---|---|
| pinhole, square, centred | focal length and aperture alone | yes |
| pinhole, non-square or off-centre | `OpenCvPinholeDistortionCfg` with zero terms (carries fx, fy, cx, cy) | yes |
| Brown-Conrady (k1, k2, p1, p2, k3, rational k4–k6) | `OpenCvPinholeDistortionCfg`, the same terms | yes |
| Kannala-Brandt (k1–k4) | `OpenCvFisheyeDistortionCfg`, max_fov = 2 × the valid half angle | yes |
| equidistant | `OpenCvFisheyeDistortionCfg` with k = 0 (KB with k = 0 is equidistant) | yes |
| equisolid, stereographic | no match: a least-squares Kannala-Brandt fit to r(θ)/f over the image's angles | no; the largest pixel gap is stated (1.9 × 10⁻⁸ px for an equisolid 250 px lens on 640 × 480, 4.0 × 10⁻⁴ px stereographic) |

Isaac Lab v2.3.2 has no OpenCV fields: its file carries the square, centred pinhole of fx, and the notes give the
largest pixel gap from the real lens, or say that the lens sees rays a pinhole cannot (a fisheye beyond 90°).

**Imagers.** Every imager of every enabled camera and depth sensor is a camera: data_types ['rgb'], plus
'distance_to_image_plane' on a depth sensor's reference imager (the library entry's reference_imager). IR and ToF
imagers are visible-light RGB prims with that imager's geometry (Isaac renders no near-infrared): the notes say so.
update_period is 1/fps of the stream choice (core/studies/motion_blur.imaging), else 0.0 (every step), said in a
comment.

**Randomisation bounds** come from `core/tolerance.bounds` (the TOLERANCE study's ± bound per axis of the mount
tolerance, shifts along the mount parent's x, y, z and turns about the body's own x, y, z applied as R' = R·Rx·Ry·Rz;
or ± 2σ of a calibration covariance's marginals). Each config file carries `DR[name]` (the sensor body in its parent,
the imager in the body, the bounds) and pure-Python `perturbed_offset(name, d)`, `sample_offset(name, rng)` (uniform
within the bounds) and `randomized_cameras(seed)` (every CameraCfg with one draw, through configclass `replace`).

## LeRobot, ACT, Diffusion Policy, OpenVLA — `lerobot.py`

**Sources** (read 2026-11-07): LeRobot at commit `d490e472a5d1a207cfe0255caaaf74b6d66e4656` (the one
`core/datasets/lerobot.py` follows): `configs/train.py`, `configs/default.py`, `configs/policies.py`,
`configs/types.py`, `policies/act/configuration_act.py`, `policies/diffusion/configuration_diffusion.py`, the
`register_subclass` names of pi0, pi0_fast, smolvla and groot, `datasets/factory.py`,
`datasets/lerobot_dataset.py`, `datasets/utils.py` (`DatasetInfo.from_dict`); ACT at
`742c753c0d4a5d87076c8f69e5628c79a8cc5488` (`constants.py`, `imitate_episodes.py`); Diffusion Policy at
`5ba07ac6661db573af695b419a7947ecb704690f` (`config/task/real_pusht_image.yaml`,
`config/train_diffusion_unet_real_image_workspace.yaml`, `dataset/real_pusht_image_dataset.py`); OpenVLA at
`c8f03f48af692657d3060c19588038c7220e9af9` (`prismatic/vla/datasets/rlds/oxe/configs.py`).

**info.json** (v2.1 and v3.0) is what `write_lerobot` writes for a dataset recorded with the rig: observation.state
and action (float32, the robots' actuated joints by name), one video feature per camera the policies take (each
sensor's primary imager at its stream size, key `observation.images.<sensor name>` as `core/sim/episodes.py` names
them, sorted as the writer sorts them), the default features, zero counters, and the video fields of LeRobot's
default encoder (AV1 from libsvtav1, yuv420p, g 2, crf 30, preset 12 at v3.0). fps: the option, else the first
policy's control rate, else the fastest camera, else 30 fps (a default).

**Train config** (per policy with a LeRobot type: act, diffusion, pi0, pi0_fast, smolvla, groot):
`TrainPipelineConfig` keys dataset {repo_id, root}, policy {type, input_features (STATE and one VISUAL [3, H, W] per
camera the policy takes), output_features (ACTION), device, push_to_hub false, repo_id}, output_dir, job_name; ACT:
chunk_size from the policy entry, temporal_ensemble_coeff = m and n_action_steps 1 (ACTConfig: "n_action_steps must
be 1 when using temporal ensembling"); Diffusion: horizon, n_action_steps, resize_shape (H, W) from the entry's
pre_resize and crop_shape from its centre crop. `train.sh` runs `lerobot-train --config_path=train_config.json`.

**§7.13 recommendations** (Policy.recommendations, stored by the action-chunk study's Apply):

| Recommendation | Where it goes |
|---|---|
| temporal_ensemble_m (w_i = exp(−m·i), i = 0 the oldest) | ACTConfig.temporal_ensemble_coeff (the same convention: "w₀ is the oldest action"); ACT's `k` in `act/task_config.yaml` (the original code hard-codes k = 0.01) |
| delay_offset_s, δ, rounded to whole frames | LeRobot: the train config has no field (delta timestamps come from the policy's delta indices, `datasets/factory.py`), so `delta_timestamps.json` holds {"action": [δ + k/fps]} over the policy's window and `make_dataset.py` builds `LeRobotDataset(repo_id, root, delta_timestamps=...)`; Diffusion Policy: `n_latency_steps` = δ in steps (its dataset drops the first n actions of a window; a negative δ cannot be written there and is said) |

Without stored recommendations the frameworks' own defaults are written and labelled (ACT's k = 0.01, the entry's
temporal ensembling, n_latency_steps 0).

**ACT YAML** (`act/task_config.yaml`): the TASK_CONFIGS entry `{dataset_dir, num_episodes, episode_len,
camera_names}` with the camera names, temporal_agg_k and the delay in steps. **Diffusion Policy YAML**
(`diffusion_policy/task.yaml`): `shape_meta` (rgb cameras [3, H, W] at the entry's pre_resize, the low-dim state,
the action) and `n_latency_steps`. **OpenVLA** (`openvla/oxe_dataset_config.py` and `.json`): an
OXE_DATASET_CONFIGS entry: image_obs_keys primary (the first static camera), secondary (the next), wrist (the first
link-mounted camera); depth keys none; state_obs_keys ['joint_state']; StateEncoding.JOINT, ActionEncoding.JOINT_POS.

## Reports — `report.py`, `_pdf.py`

`report/report.html` (standalone: inline SVG figures from `validation/report.svg_figure`, light or dark),
`report/report.pdf` and `report/report.json`. The results are those given (`results`: the app's results; or
`results_dir`: batch result files), and the studies named in `studies` are run now ('all' when nothing is given:
every study except those that run other studies many times, the tolerance Monte Carlo, experiments, the optimiser,
the design agent, the variant comparison and the scorecard, which are reported when their results are given; on
the UR5e sample that is 18 results and 4 refusals in 66 s, mostly the ground-truth renders and the teleop and grasp
studies);
a study that refuses is listed under "Not run" with its sentence. Each section: the STATUS word with its colour and
the study's advice (read from the result's `headline` or `summary`), the study's named figures (`core/metric_defs.py`
through `core.metrics.read`; "not stated" when the result does not state one), one bar chart of the first per-camera
(per-policy, per-cable, …) list carrying a number with a unit suffix, the notes, and the result's own hash when it
came with a provenance block. The scorecard is `core/requirements.check` on the same results (shared through a result
cache, so no study runs twice). The provenance block is `core.provenance.block` of the report document.

The PDF is written by `_pdf.py`, pure Python (ISO 32000-1 file structure, Flate streams, the standard fonts with
WinAnsiEncoding), so no optional package is needed and nothing is refused; characters outside cp1252 are spelled
out (≥ as >=, σ as sigma).

## Validation

| Check | Quantity | Reference | Tolerance (why) | Measured 2026-11-07 |
|---|---|---|---|---|
| `formats.isaac_usd` | largest pixel gap of points projected through every re-opened camera prim (pxr- and text-written; the three samples, two cameras given a Brown-Conrady and a Kannala-Brandt calibration with an off-centre, non-square K) | pxr's Gf.Camera frustum against `optics.CameraModel`'s pinhole part; the prim's world transform against `camrender.optical_pose`; the OpenCV attributes against the lens | 10⁻³ px: USD stores the lens as 32-bit floats (10⁻⁴ px at a 1 400 px focal length); a sign error moves the principal point by tens of pixels | 2.6 × 10⁻⁵ px; transform 5.6 × 10⁻¹⁷ m, 5.5 × 10⁻¹⁶ rad; 0 mismatches |
| `formats.isaac_config` | position difference of each CameraCfg offset (v3.0.0 and v2.3.2 files executed against stubs that accept only the source's fields) | `camrender.optical_pose` taken to the link by `RobotModel.link_pose`; `tolerance.bounds`; `perturbed_offset` against `tolerance.apply_draw` through `core.factors` | 10⁻⁹ m and rad: 17-digit literals, so only composition rounding remains | 1.4 × 10⁻¹⁶ m (draws 1.4 × 10⁻¹⁶ m, 8.2 × 10⁻¹⁶ rad), 0 mismatches, 52 draws |
| `formats.lerobot_export` | mismatches of the exported info.json against a dataset written with the same features and read back; train-config keys against the pinned dataclass fields; YAML and OpenVLA entry parsed | `core/datasets/lerobot` (itself checked by `formats.lerobot_roundtrip`); the LeRobot, ACT, Diffusion Policy and OpenVLA sources | 0 | 0 |
| `formats.report` | mismatches when report.html is parsed with html.parser (every study present a section with its heading, refusals listed, scorecard, provenance with the content hash, figures) and report.pdf read by pdftotext | report.json and `provenance.project_digest`; html.parser; poppler | 0 | 0 (pdftotext 0 errors) |

A self-test line, "sim exports", runs the SO-100 versions of the first three (about 1.3 s).

## Known limits

Isaac Lab, Isaac Sim, LeRobot, ACT, Diffusion Policy and OpenVLA are not installed here: the configs are checked by
parsing and executing them against the field names of the pinned sources, not by those programs, and the USD is not
rendered in Isaac Sim. The OpenCV lens attributes' pixel convention is taken to be OpenCV's (centres at integers),
which the Isaac Lab source does not state; its pinhole K puts the centre at W/2, half a pixel from OpenCV's
(W − 1)/2 if its pixels are centred at integers. The USD aperture offsets are exact for pxr's `Gf.Camera`, but Isaac Lab's spawner warns "Camera aperture offsets are not supported by Omniverse. These parameters will be ignored": in an Isaac render an off-centre principal point comes only through the OpenCV lens attributes (the v2.3.2 file has none, so its render is centred). OpenVLA's `ActionEncoding.JOINT_POS` is joint *delta* positions (7 joints, padded, + gripper) in its `configs.py`: absolute joint actions need converting, and a two-arm dataset needs the `_BIMANUAL` encodings (the export's notes say so). A renderer that ignores the OpenCV attributes conforms non-square
pixels to the render's aspect. IR and ToF imagers are RGB prims; depth outputs have no noise model. The randomisation
is drawn once per config build (per-environment draws need `Camera.set_world_poses` at reset, not generated). The
LeRobot state and action are every actuated joint of the robot files (a gripper's linkage joints included; the UR5e
sample has 14); LeRobot's train config has no field for the delay offset (applied through delta timestamps or by
shifting the action column); a dataset rate above a camera's repeats frames (said). The report's figure is one list
per study chosen by a naming rule; results given with an export are reported as given. The PDF's line breaking uses
approximate glyph widths.
