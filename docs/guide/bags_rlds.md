# Reading ROS 2 bags and RLDS datasets

DexVantage reads recorded episodes from ROS 2 bags (MCAP or SQLite) and from RLDS / Open X-Embodiment datasets into
the same episode shape it reads LeRobot datasets into, with no ROS and no TensorFlow installed. The model, the formats
and the validation are in [../models/datasets.md](../models/datasets.md) (sections *ROS 2 bags* and *RLDS*).

## A ROS 2 bag, step by step

This example writes a two-second pick episode as a bag (the same way a recorder would: joint states at 100 Hz, a
30 fps wrist camera with its CameraInfo, the static transform of the camera on the tool flange and the task string)
and reads it back.

```python
import numpy as np
from dexvantage.core import rosbag as rb
from dexvantage.core.datasets import open_dataset

T0 = 1_762_500_000_000_000_000                     # ns since the epoch
names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint',
         'wrist_3_joint']
with rb.BagWriter('out/pick_001') as bag:          # out/pick_001/pick_001_0.mcap + metadata.yaml
    bag.add_topic('/joint_states', 'sensor_msgs/msg/JointState')
    bag.add_topic('/wrist/color/image_raw', 'sensor_msgs/msg/Image')
    bag.add_topic('/wrist/color/camera_info', 'sensor_msgs/msg/CameraInfo')
    bag.add_topic('/tf_static', 'tf2_msgs/msg/TFMessage')
    bag.add_topic('/task', 'std_msgs/msg/String')
    T = np.eye(4); T[:3, 3] = [0.0, 0.05, 0.03]    # the camera 5 cm to the side of tool0, 3 cm up
    bag.write('/tf_static', T0, {'transforms': [rb.transform_stamped(T, T0, 'tool0', 'wrist_optical')]})
    bag.write('/task', T0, {'data': 'pick the red cube'})
    for i in range(200):
        st = T0 + i * 10_000_000
        bag.write('/joint_states', st + 150_000, {'header': rb.header(st, 'base_link'), 'name': names,
                                                   'position': [0.1 * np.sin(0.02 * i + k) for k in range(6)]})
    for i in range(60):
        st = T0 + 4_000_000 + i * 33_333_333
        img = np.full((48, 64, 3), (i * 4) % 256, np.uint8)
        bag.write('/wrist/color/image_raw', st + 6_000_000, rb.image_msg(img, st, 'wrist_optical'))
        bag.write('/wrist/color/camera_info', st + 6_000_000,
                  rb.camera_info(64, 48, 'plumb_bob', [0.05, -0.01, 0, 0, 0],
                                 [[60.0, 0, 32], [0, 60, 24], [0, 0, 1]], t_ns=st, frame_id='wrist_optical'))

with open_dataset('out/pick_001') as ds:
    ep = ds.load(0)
    cam = ep.cameras['/wrist/color/image_raw']
```

What comes back (run on 2026-11-07):

| | value |
|---|---|
| `ds.format`, `ds.version`, episode ids | `mcap`, `v5`, `['pick_001']` |
| `ep.q.shape`, `ep.task`, `ep.duration_s` | `(200, 6)`, `pick the red cube`, `1.99` s |
| camera size and frames | 64 × 48, 60 frames |
| first `cam.timestamps` | 0.004, 0.0373, 0.0707 s (header stamps from the episode start) |
| first `cam.meta['log_times']` | 0.010, 0.0433, 0.0767 s (when the recorder received them: 6 ms later) |
| `cam.intrinsics` | model `brown`, K row 1 = [60, 0, 32] |
| `cam.extrinsics` | frame `tool0`, position [0, 0.05, 0.03], quaternion [0, 0, 0, 1] |
| `next(cam.frames([10]))` | frame 10, a 48 × 64 × 3 uint8 RGB array |

Things to know when you open your own bags:

- **Folders of bags.** Point `open_dataset` at a folder whose sub-folders are bags (or which holds loose `.mcap` /
  `.db3` files): each bag is one episode, in natural name order (`ep_2` before `ep_10`).
- **Compressed bags** need the bags extra: `pip install "dexvantage[bags]"`. Without it, loading says so in a
  sentence. A bag compressed as whole files must be decompressed first with `ros2 bag decompress`.
- **Device clocks.** If a camera driver stamps frames with the camera's own clock, DexVantage notices (its offset from
  the receive time differs from the joint states' by more than 10 s, a default) and moves those stamps to
  `cam.hw_timestamps`; `cam.meta['timestamp_source']` tells you what `timestamps` hold.
- **What is not read:** actions (bags have no standard action message), the dynamic `/tf`, depth images (listed in
  `ep.meta['depth_streams']`).

## An RLDS / Open X-Embodiment dataset

Download a dataset folder with its `dataset_info.json`, `features.json` and TFRecord shards (for example one Open
X-Embodiment dataset's `0.1.0` folder), then:

```python
from dexvantage.core.datasets.rlds import RldsDataset

ds = RldsDataset('datasets/berkeley_autolab_ur5', fps=5.0)   # fps: your control rate; None leaves t in steps
refs = ds.episodes()                    # 'train-000000', ... from the record headers only
ep = ds.load(0)                         # CRCs checked; JPEG frames decoded only when asked for
ep.q                                    # the observation's state vector (meta['state_feature'] names it)
ep.action, ep.action_names              # 'world_vector_0', ... in features.json order
ep.task                                 # the language instruction
ep.cameras['observation/image'].frames([0])
```

`open_dataset(path)` finds the format by itself; use `RldsDataset` directly when you want to give the fps. RLDS does
not store times, so DexVantage does not invent them: without an fps, `ep.t` counts steps and
`ep.meta['time_base']` is `'steps'`. A damaged record is refused with a sentence naming it ("Record 3 of
bridge-train.tfrecord-00012-of-01024 is damaged: its data checksum does not match, so it was not read"), and the other episodes still read.

## Checking the readers yourself

Help ▸ Validate the models… (or `dexvantage --validate out/validation`) runs `formats.mcap_roundtrip` and
`formats.tfrecord_crc` among the format checks; both must report 0 mismatches.
