# Silhouette calibration: a worked example

Has the overhead camera moved since the rig was calibrated? There is no board in the episodes, but the robot is in
every frame. This walk-through checks the UR5e sample's overhead D435 against the robot's own silhouette, episode by
episode. It finds the episode where the mount slipped and writes the recovered mount back. The method, its
equations, its checks and its limits are in `docs/models/render_calib.md`.

The episodes here are synthetic (Simulate ▸ Generate synthetic episodes…, `core/sim/episodes.py`), so the true mount is known.
On a real rig they are the LeRobot, RLDS or ROS 2 bag recordings of your data collection.

## A. A dataset to check

1. Open the sample: File ▸ Open sample ▸ "UR5e and Robotiq 2F-85 with a wrist D405 and an overhead D435".
2. Make a dataset: Simulate ▸ Generate synthetic episodes… with its defaults (4 episodes of the sample's pick trajectory,
   LeRobot v3.0, 640 × 360). Every fault is on by default. The one that matters here is the mount slip: from episode
   2 on, the overhead camera's mount is moved 3 mm along its body x axis and turned 1.5° about its y axis (defaults
   chosen to show, not measured knocks). The overhead camera also gets a 50 ms camera–joint desync in episodes 0–2
   and a 5000 ppm clock drift in episodes 2–3, among others. Generating took 16 s on the RTX 5080.

## B. Run it (Calibrate ▸ Silhouette calibration)

Switch to the Calibration workspace: the page is next to the hand-eye pages.

1. **Dataset…**: choose the dataset's folder. The page shows its name. The path is stored in the project's study
   settings, one undo step.
2. **Camera**: "Overhead D435", the only static camera. The wrist D405 rides on the arm, so it uses the hand-eye
   solve instead.
3. **Frames** (default 12 per episode, chosen where the joints differ most) and **Episodes** (default the first 8)
   are yours to set; the line under them lists the defaults in use.
4. **Run**. The four episodes take 17.7 s, about 4.5 s each (about 1 500 renders through EGL on the RTX 5080).

## C. What it says

The headline is the last observable episode:

    4.2 mm · 1.53°    ± 0.9 mm · 0.06° (1σ), episode 3        mount moved
    The camera is 4.2 mm and 1.53° from its mount: Apply writes the recovered mount.

The per-episode table (offset from the project's mount; the truth from the dataset's sidecar, which the page does not
read):

| episode | offset | 1σ | error against the truth | IoU before → after | edge RMS |
|---|---|---|---|---|---|
| 0 | 1.34 mm, 0.080° | 2.3 mm, 0.13° | 1.34 mm, 0.080° | 0.26 → 0.26 | 1.11 px |
| 1 | 0.68 mm, 0.062° | 2.3 mm, 0.15° | 0.68 mm, 0.062° | 0.23 → 0.23 | 1.13 px |
| 2 | 2.76 mm, 1.387° | 1.6 mm, 0.10° | 2.65 mm, 0.230° | 0.19 → 0.23 | 1.02 px |
| 3 | 4.21 mm, 1.534° | 0.9 mm, 0.06° | 1.35 mm, 0.056° | 0.14 → 0.17 | 0.80 px |

Below the table:

> Mount steps: between episodes 1 and 2, 2.5 mm and 1.42° (19.6 σ).

That is the slip: 1.5° about the camera's y axis is recovered as 1.37–1.53°. The change from episode 2 to 3
(3.9 mm, 0.28°) stays under 3 σ, so it is not called a step.

Read the errors with the faults in mind. Episode 2 is off by 2.65 mm and 0.23° because its overhead stream also
carries a 50 ms desync and a clock drift: the images show the arm where the joint log says it was earlier, and the
silhouette calibration takes that as a mount error. Fix the timing first (Analysis ▸ Time sync, and the dataset
auditor's camera–joint checks). The page's notes say which lens model it used: the project's imager, because the
LeRobot format stores no intrinsics.

The thumbnail shows the frame with the largest silhouette in the headline episode. The grey dashed outline is the
robot rendered through the project's mount; the green one is the robot through the recovered mount, which should hug
the arm's edges. With the Camera view on "Overhead D435", the page draws the robot's outline through the recovered
mount over the view in orange, at the joint state on screen.

## D. Apply

**Preview** shows both camera frames in the 3D view: grey is the mount now, coloured is the recovered one, with a
dashed line between them. **Apply** writes the recovered mount (the sensor body pose in its parent frame, 4.2 mm and
1.53° away) to the D435, one undo step. Every study that depends on the camera's pose re-runs against the new mount.

## When it says "not observable"

- **The robot is too small in the image**: it covers less than 0.5 % of the image (a default), or its outline gives
  fewer than 30 points per frame. Move the camera closer or record at a higher resolution.
- **The robot is too static**: the frames hold fewer than 3 distinct joint states (a default; two frames are distinct
  when a joint differs by 5°). An episode where the arm sits still cannot separate a mount error from the arm's own
  outline. Choose episodes with motion, or more frames.
- **The weakest direction is beyond the requirement**: the page names it, for example "the camera's position along
  the view (distance)", and says how far beyond 5 mm and 0.5° (defaults) its σ is. More frames with the arm spread
  across the image tighten it.

## Batch

    dexvantage project.dexv --silhouette-calib silhouette.json

The dataset is the one in the project's settings (`analysis.studies.silhouette_calib.dataset`, set by the page's
Dataset… button), together with `sensor`, `max_frames`, `episodes` and the other settings of
`docs/models/render_calib.md`. Without a dataset the run refuses in one sentence and exits 2. The result is JSON with
a provenance block.

## From code (the auditor's entry point)

    from dexvantage.core.calib import render_calib as RC
    from dexvantage.core.datasets import open_dataset
    with open_dataset(path) as ds:
        one = RC.estimate(ds, project, sensor_uid)          # the first episode (settings {'episode': k} for another)
        all_ = RC.drift(ds, project, sensor_uid)             # every episode, and the steps between them
    one['delta'], one['sigma'], one['observable'], one['why'], one['frames_used']
