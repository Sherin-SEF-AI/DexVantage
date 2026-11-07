# Isaac Lab, LeRobot and reports: a worked example on the UR5e sample

Export the UR5e sample's cameras to Isaac Lab, write the LeRobot dataset template and a train config that carries
the action-chunk study's recommendation, and write a report. The model, the sources and the checks are in
[../models/exporters_b.md](../models/exporters_b.md).

In the app: **File ▸ Export…** (or Export ▸ Export to MuJoCo, Isaac, ROS 2, LeRobot…), tick *Isaac Lab and Isaac Sim
cameras*, *LeRobot, ACT, Diffusion Policy and OpenVLA configs* and *Report (HTML and PDF)*, choose a folder. The
dialog lists each format's files and notes. From Python:

```python
from dexvantage.core import exporters as X, jobs
from dexvantage.core.robot.samples import sample_project
from dexvantage.core.studies import action_chunk_sync as AC

p = sample_project('ur5e')
AC.apply_recommendations(p, jobs.run_study('action_chunk_sync', p.snapshot()))   # the page's Apply
man = X.export(p, 'out/ur5e', ['isaac', 'lerobot', 'report'],
               {'report': {'studies': ['bandwidth', 'motion_blur', 'time_sync', 'action_chunk_sync']}})
```

## 1. Isaac Lab and Isaac Sim

`out/ur5e/isaac/` holds `cameras.usda` (written by pxr when usd-core is installed: `pip install "dexvantage[usd]"`),
`isaaclab_cameras.py` (Isaac Lab v3.0.0), `isaaclab_v2_cameras.py` (v2.3.2) and `isaac_cameras.json`. The sample has
six imagers (the D405 on `wrist_3_link` and the D435 on the cage, each with IR left, IR right and RGB):

| Camera | Parent | Size | fx, fy (px) | Lens in Isaac Lab | Outputs |
|---|---|---|---|---|---|
| wrist_d405_ir_left | wrist_3_link | 1280 × 720 | 710.79, 649.46 | OpenCV pinhole, zero terms (non-square) | rgb, distance_to_image_plane |
| wrist_d405_rgb | wrist_3_link | 1280 × 720 | 710.79, 721.62 | OpenCV pinhole, zero terms | rgb |
| overhead_d435_rgb | environment | 1920 × 1080 | 1396.81, 1406.75 | OpenCV pinhole, zero terms | rgb |

(The focal lengths come from the D405 and D435 entries' stated fields of view, so fx ≠ fy: Isaac Lab's plain pinhole
would use fx for both, so each camera carries an OpenCV model with zero distortion that holds fx, fy, cx, cy.)

A camera config, v3.0.0:

```python
WRIST_D405_RGB = CameraCfg(
    prim_path=ENV_NS + '/Robot/wrist_3_link/wrist_d405_rgb',
    update_period=0.03333333333333333,          # the D405's 30 fps stream
    width=1280, height=720, data_types=['rgb'],
    offset=CameraCfg.OffsetCfg(pos=(-0.0, 0.1308, 0.07), rot=(-0.8627, -0.0, 0.0, 0.5057), convention="ros"),
    spawn=sim_utils.PinholeCameraCfg(focal_length=0.5553..., horizontal_aperture=1.0, ..., distortion=...),
)
```

Add it to your scene config (`scene.wrist = CAMERAS["wrist_d405_rgb"]`, the robot's prim named `Robot`, or pass
`{'isaac': {'robot_prim': 'MyArm'}}`). The IR imagers are written as RGB prims with the IR imager's geometry (the
notes say so: Isaac renders visible light).

**Domain randomisation.** The sample's mounts carry DexVantage's labelled default tolerance, ± 1 mm and ± 0.5° per
axis (the notes say so); set your bracket's own on the Sensors page or the Tolerance page. Then, per training run:

```python
from isaaclab_cameras import randomized_cameras, sample_offset, DR
cams = randomized_cameras(seed=3)               # every CameraCfg with one draw of its mount
pos, rot, d = sample_offset('wrist_d405_rgb')   # one draw: d = (dx, dy, dz) m, (droll, dpitch, dyaw) degrees
```

The draw moves the sensor body exactly as DexVantage's tolerance study does (`formats.isaac_config` compares it).

## 2. LeRobot, ACT, Diffusion Policy, OpenVLA

The action-chunk study recommends a delay offset of 0.1 s for the sample's π0 (5 frames at its 50 Hz control rate)
and, for π0 (which executes chunks open loop), no ensemble weight. The export writes:

- `lerobot/v2.1/meta/info.json` and `lerobot/v3.0/meta/info.json`: 14 state and action names (the UR5e's six joints
  and the Robotiq's eight linkage joints), `observation.images.overhead_d435` (1080 × 1920 × 3) and
  `observation.images.wrist_d405` (720 × 1280 × 3), fps 50. The notes say the 30 fps cameras repeat frames at 50 fps:
  record at 30 fps or pass `{'lerobot': {'fps': 30}}`.
- `lerobot/pi0_sample/train_config.json`: policy type `pi0`, the two cameras as VISUAL [3, H, W] inputs,
  `push_to_hub: false`; `train.sh` runs `lerobot-train --config_path=train_config.json` (flags after it override,
  e.g. `--steps=20000`, or `--policy.path=<a pretrained π0 checkpoint>` to fine-tune).
  Set `{'lerobot': {'repo_id': '<you>/<dataset>'}}` for your dataset's name (default `local/<project>`).
- `lerobot/pi0_sample/delta_timestamps.json`: `{"action": [0.1, 0.12, …]}`, the 50-step window starting 0.1 s
  after each observation, and `make_dataset.py`, which builds the LeRobotDataset with it (LeRobot's train config has
  no field for the offset).
- `act/task_config.yaml`, `diffusion_policy/task.yaml` and `openvla/oxe_dataset_config.py`: the same cameras for the
  original codes (OpenVLA: primary `overhead_d435`, wrist `wrist_d405`).

With an ACT policy (Policies page, *ACT*), the train config gets `temporal_ensemble_coeff` = the study's m with
`n_action_steps: 1`, and the notes say that ACT needs one image size for every camera (here 1280 × 720 and
1920 × 1080 differ).

## 3. The report

`report/report.html` opens in any browser (it follows the light or dark setting); `report/report.pdf` is the same on
A4 pages. When the project has requirements (for example DexVantage's defaults,
`core/requirements.default_requirements()`), the report starts with the scorecard; each study then has its STATUS word (for example *on contact* for the action-chunk study, *no hosts* for
bandwidth, *may be blurred* for motion blur), its figures, a chart (payload per sensor, peak speed per camera) and its
notes; studies that cannot run on the project are listed under *Not run* with their reason. The provenance block at
the end gives the DexVantage version and code hash, the project's content hash and the time.

In batch, run the studies first and point the report at their files:

    dexvantage ur5e.dexv --motion-blur res/blur.json --time-sync res/sync.json
    python -c "from dexvantage.core import exporters as X; from dexvantage.core.project import Project; \
      X.export(Project.load('ur5e.dexv'), 'out', ['report'], {'report': {'results_dir': 'res'}})"
