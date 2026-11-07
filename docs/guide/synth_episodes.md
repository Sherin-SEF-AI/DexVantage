# Generating synthetic episodes with known faults

A synthetic dataset is a LeRobot dataset rendered from your project, with faults placed where you know they are. Use
it to test a dataset auditor, a data loader or a training pipeline before real recordings exist. The model, the
defaults and the validation are in `docs/models/synth_episodes.md`.

## In the app

1. Open a project with a robot and cameras, for example File ▸ Open sample ▸ UR5e + Robotiq 2F-85.
2. Simulate ▸ Generate synthetic episodes….
3. Set the number of episodes and the rows per episode (0 plays each motion at its own length), the LeRobot version
   (v3.0 or v2.1), the image width, and the seed. Tick the faults to inject; all ten are ticked at first.
4. Choose a new or empty folder and press Generate. The status bar shows the progress, and Cancel stops the job
   without leaving a half-written folder. When it ends, the status bar says what was written and which faults were
   injected.

The robot follows the project's trajectories. A project without any (the SO-100 sample) approaches the grasps of its
first grasp set instead.

## Worked example: the UR5e sample

    dexvantage --sample ur5e --synth-episodes ur5e-synth \
        --synth-settings '{"episodes": 2, "frames": 60, "max_width": 320}'

prints (on an RTX 5080):

    2 synthetic episodes (120 rows, 220 camera frames from 2 cameras) written as LeRobot v3.0 in 1.9 s (116 frames/s).
    Faults injected: dropped frames (4 frames in 2 episodes); duplicated frames (4 frames in 2 episodes); camera-joint
    offset (2 episodes); clock drift (2 episodes); timestamp jitter (2 episodes); exposure change (30 frames in 2
    episodes); motion-blurred frames (4 frames in 2 episodes); gripper hidden in the wrist camera (6 frames in 2
    episodes); mount slip (1 episode); USB reset gaps (16 frames in 2 episodes). Ground truth: dexvantage_truth.json;
    rendered on NVIDIA GeForce RTX 5080/PCIe/SSE2.

The folder is an ordinary LeRobot v3.0 dataset plus the sidecar:

    data/chunk-000/file-000.parquet
    meta/info.json, meta/stats.json, meta/tasks.parquet, meta/episodes/chunk-000/file-000.parquet
    videos/observation.images.wrist_d405/chunk-000/file-000.mp4
    videos/observation.images.overhead_d435/chunk-000/file-000.mp4
    dexvantage_truth.json

Both episodes play the sample's "Scripted pick", time-scaled to 60 rows at 30 fps. The 30 fps comes from the D405's
and D435's RGB streams (the sidecar's `fps_source`). The images are the streams scaled to 320 × 180. What episode 0 of
the sidecar says, in short:

| Fault | Camera | Where |
|---|---|---|
| gripper hidden | wrist D405 | stored frames 19–21 (an occluder over the gripper) |
| USB reset | overhead D435 | slots 9–16 missing; frame 9 is the first after the 0.3 s gap |
| drops | wrist D405 | slots 41 and 51 missing; frames 41 and 50 follow the gaps |
| duplicates | wrist D405 | frames 3 and 28 repeat frames 2 and 27 exactly |
| blur | overhead D435 | frames 3 and 29, 9 renders over 1/30 s |
| exposure change | overhead D435 | frames 20–34 × 1.4 |
| desync | overhead D435 | its images show the robot 0.05 s after their stamps |
| drift | wrist D405 | content time = stamp × (1 + 5000 × 10⁻⁶) |
| jitter | overhead D435 | stamps ± N(0, 3 ms), clipped to ±6.7 ms |

Episode 1 also carries the mount slip: from episode 1 on, the overhead D435 sees the cell through its mount moved
3 mm along x and turned 1.5° about y. `episodes["1"]["extrinsics"]` holds that pose, and `cameras[...]["extrinsics"]`
holds the true one. `visibility` says, per stored frame, whether the gripper shows (the wrist camera sees it in 55 of
58 frames, the 3 hidden ones excepted).

## Reading it back

    from dexvantage.core.datasets import open_dataset
    with open_dataset('ur5e-synth') as ds:
        ep = ds.load(0)                      # Episode: t, q, action, cameras {key: CameraStream}
        truth = ds.truth()                   # the sidecar, or None for a recorded dataset
        cam = ep.cameras['observation.images.wrist_d405']
        for i, rgb in cam.frames([19, 20]):  # decoded lazily
            ...

A test of an auditor should find the faults from `ep` alone and only then compare its findings with `truth`. Seed,
settings and the project fix every fault's place, so the same command writes the same dataset.

## Settings

Stored in `project.analysis.studies['synth_episodes']`; `--synth-settings` (JSON text or a .json file) goes over
them. `episodes` (4), `frames` (none: each motion's own length), `version` (`v3.0`), `seed` (0), `motion` (`auto`,
`trajectories`, `grasps`), `size` ([w, h]) or `max_width` (640), `default_fps` (30), `noise_dn` (1), `action_lead_rows`
(1), `q_offset_rad` (0.02), `pre_grasp_m` (0.08), `max_joint_speed` (0.8), `gripper_min_frac` (0.002), `device`
(`auto` or an EGL device index), and `faults`: a list of the faults to inject, or {fault: false | {its settings,
`episodes`, `camera`}}, for example

    {"faults": {"drift": {"drift_ppm": 200, "camera": "observation.images.wrist_d405"}, "blur": false}}

The figures in brackets are DexVantage defaults chosen so each fault shows in a short episode. They are not
measurements of any camera.
