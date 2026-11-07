# Looking at the rig in Rerun

Rerun is a viewer many robot-learning groups already keep open. DexVantage can hand it the rig: the robot moving
through its trajectory, every camera's frustum, the depth cameras' min-Z blind zones, the task volumes and, after a
dataset audit, which episodes failed and why. The model is in [docs/models/rerun.md](../models/rerun.md).

## Set-up (once)

    pip install "dexvantage[rerun]"          # rerun-sdk 0.42.0

Without the package, Export ▸ Stream to Rerun… says this in one sentence, and nothing else in DexVantage needs it.

## A worked example on the UR5e sample

1. File ▸ Open sample ▸ UR5e + Robotiq 2F-85.
2. Export ▸ Stream to Rerun… asks whether to save an `.rrd` file or open the Rerun viewer.

   - **Save an .rrd file…** writes the scene in the background. The status bar then reads: "Sent 1 robot (21 links,
     136 poses over Scripted pick (sample)), 6 camera frustums, 2 min-Z zones, 1 task volume to dexvantage.rrd."
     The file is about 2.8 MB and takes 0.3 s. Open it later with `rerun dexvantage.rrd`.
   - **Open the Rerun viewer** starts a viewer and streams the same scene into it.
3. In Rerun:
   - Scrub the `time` timeline: the arm runs the scripted pick at 30 samples per second.
   - The Wrist D405's frusta (RGB and IR) ride on `wrist_3_link`, and the Overhead D435's stay on the cage.
   - The red pyramids in front of each RealSense are the volumes closer than min-Z, where the camera gives no depth.
   - The blue box is the task volume.
   - Each camera's `lens` entity names its lens model and K.

## With a dataset audit

Run Data ▸ Audit a dataset… first, or a batch audit. Stream to Rerun… then adds the `episode` timeline:

- `audit/verdict` logs one line per episode, for example "episode 3: fix (desync)", at the WARN level.
- `audit/pass` is 1 or 0.
- `audit/flags/<flag>` (blur, desync, gripper_hidden, mount_slip, …) is the number of cameras failing that check.

In batch:

    dexvantage --sample so100 --synth-episodes out/synth
    dexvantage --sample so100 --audit out/audit.json --dataset out/synth --rerun out/scene.rrd
    # or, from an audit written before:
    dexvantage project.dexv --rerun out/scene.rrd --rerun-audit out/audit.json

An output name that does not end in `.rrd`, or a folder that does not exist, is refused in one sentence (exit 2).

## Reading the file back in Python

```python
from dexvantage.core.bridges import rerun as RB
rb = RB.read_back('out/scene.rrd')        # {entity path: {'static': {...}, 'rows': [...]}}
rb['/world/task_volumes/Tabletop_sample']['static']['Boxes3D:centers']
```
