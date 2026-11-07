# Datasets: a worked example on the SO-100 sample

Write a two-episode LeRobot dataset with the SO-100 sample's joints and a wrist camera, read it back, and see what
DexVantage reports for each episode. The format, the reader's mapping, the writer and the validation are in
[../models/datasets.md](../models/datasets.md). The LeRobot reader and writer need the `lerobot` extra (pyarrow and
PyAV): `pip install "dexvantage[lerobot]"`.

## 1. Write a dataset

```python
import numpy as np
from dexvantage.core.robot.samples import sample_project
from dexvantage.core.datasets import episode as E, lerobot as L, open_dataset

robot = next(r for r in sample_project('so100').robots if r.primary)
names = [j['name'] for j in robot.joints if j.get('type') in ('revolute', 'prismatic')]
# ['Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll', 'Jaw']

fps, n = 30, 90                                   # 3 s at 30 fps
t = np.arange(n) / fps
q = np.zeros((n, 6)); q[:, 0] = 0.6 * np.sin(2 * np.pi * t / 3.0)   # radians

def frames(indices):                              # any source of HxWx3 uint8 RGB frames
    yy, xx = np.mgrid[0:240, 0:320]
    for i in (range(n) if indices is None else indices):
        img = np.zeros((240, 320, 3), np.uint8)
        img[..., 0], img[..., 1], img[..., 2] = (xx + 4 * i) % 256, yy, 128
        yield i, img

cam = E.CameraStream('observation.images.wrist', 320, 240, fps, t.copy(), _decode=frames)
ep = E.Episode('demo', 0, t, q, names, action=q.copy(), action_names=names, fps=fps, robot_type='so100',
               task='wave the shoulder', cameras={cam.key: cam})
summary = L.write_lerobot('so100_demo', [ep, ep], 'v3.0', robot_type='so100')
```

`summary['codec']` is `AV1 (libsvtav1)` on a PyAV build with that encoder, `H.264 (libx264)` otherwise. The files are
`data/chunk-000/file-000.parquet` and `videos/observation.images.wrist/chunk-000/file-000.mp4`, plus `meta/`
(`info.json`, `stats.json`, `tasks.parquet`, `episodes/chunk-000/file-000.parquet`). Measured on the development
machine (CPU): 0.61 s for 180 frames at 320 × 240; 153 kB on disk. For one file per episode in LeRobot's older
layout, pass `'v2.1'`.

## 2. Read it back

```python
with open_dataset('so100_demo') as ds:
    print(ds.format, ds.version, [(r.index, r.length) for r in ds.episodes()])
    # lerobot v3.0 [(0, 90), (1, 90)]
    g = ds.load(1)
    wrist = g.cameras['observation.images.wrist']
    print(g.task, g.joint_names[:2], g.q.shape, wrist.n_frames, wrist.meta['from_timestamp'])
    # wave the shoulder ['Rotation', 'Pitch'] (90, 6) 90 3.0
    some = dict(wrist.frames([0, 45, 89]))        # decoded on demand: 7 ms for these three
```

Episode 1 starts 3.0 s into the packed video, because episode 0 is 90 frames at 30 fps; its camera times still read
0, 1/30, 2/30, … from the episode start. The three decoded frames measured 44.3, 40.5 and 39.7 dB PSNR against the
frames written: the video is lossy (crf 30, LeRobot's default), and the validation check requires 30 dB.

## 3. Datasets recorded elsewhere

Point `open_dataset` at a LeRobot dataset folder or at a Hub snapshot already on disk, for example
`~/.cache/huggingface/hub/datasets--lerobot--svla_so101_pickplace` (it follows `refs/main` to the snapshot). Or
name a public dataset on the Hub and let DexVantage stream it: `open_dataset('hf://datasets/lerobot/svla_so101_pickplace')`,
in batch `--dataset hf://datasets/lerobot/svla_so101_pickplace`, or **From the Hub…** on the Dataset audit page
(type `lerobot/svla_so101_pickplace`, optionally `@revision`). Only the metadata comes on open; each episode's data
and video files come when the episode is read, into `~/.cache/DexVantage/hub/` (XDG_CACHE_HOME), pinned to the commit
the revision pointed at. A dataset of another codebase version is refused with the sentence "… is a LeRobot
dataset of codebase version v2.0; DexVantage reads versions v2.1 and v3.0 only (LeRobot's own scripts convert
older ones: v2.0 to v2.1 in its release v0.3.3, v2.1 to v3.0 in lerobot.scripts.convert_dataset_v21_to_v30)." Joint values come back as stored: check the units, because
many SO-100 datasets on the Hub store degrees.
