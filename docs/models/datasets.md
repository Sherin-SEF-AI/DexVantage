# Datasets: readers, writers and the shared episode shape

`dexvantage/core/datasets/`: `episode.py` (the shared `Dataset` / `Episode` / `CameraStream` shape and
`open_dataset`), one module per format, the validation check `formats.lerobot_roundtrip` in
`dexvantage/validation/checks_formats.py`. A worked example is in [../guide/datasets.md](../guide/datasets.md).

Every reader opens a dataset into the same shape, so the auditor (§8.2), plan-against-recording (§8.3) and the
calibration capture work on any format:

```python
from dexvantage.core.datasets import open_dataset
with open_dataset(path) as ds:            # sniffs the format; a plain DatasetError when it cannot
    for ref in ds.episodes():             # cheap: index, id, length
        ep = ds.load(ref.index)           # t (N,), q (N, nq), joint_names, action, task, cameras, meta
        cam = ep.cameras['observation.images.wrist']
        for i, rgb in cam.frames([0, 10]):   # decoded lazily, HxWx3 uint8
            ...
    truth = ds.truth()                    # the ground-truth sidecar of a synthetic dataset, else None
```

A figure a format does not record is `None`, never a guess: LeRobot stores no intrinsics, extrinsics or hardware
timestamps, so those are `None` for a LeRobot camera.

## Formats

### LeRobot v2.1 and v3.0 — `core/datasets/lerobot.py`

**Source.** The format is taken from the LeRobot repository itself, https://github.com/huggingface/lerobot, read
2026-11-07: v3.0 at commit `d490e472a5d1a207cfe0255caaaf74b6d66e4656` (main, 2026-11-05), files
`src/lerobot/datasets/{utils, dataset_metadata, dataset_writer, io_utils, feature_utils, compute_stats,
video_utils}.py`, `src/lerobot/configs/video.py`, `src/lerobot/utils/constants.py`,
`src/lerobot/scripts/convert_dataset_v21_to_v30.py` and `docs/source/lerobot-dataset-v3.mdx`; v2.1 at tag `v0.3.3`
(commit `b883328e6c95681ca90a18b102e4ae5e1f91e2bf`, the last release whose `CODEBASE_VERSION` is `"v2.1"`), files
`src/lerobot/datasets/{utils, lerobot_dataset, compute_stats, video_utils}.py`. LeRobot is Apache-2.0; nothing of it
is bundled or imported (the layout is reimplemented from the source).

**Layouts.**

| | v2.1 | v3.0 |
|---|---|---|
| header | `meta/info.json`: `codebase_version`, `robot_type`, `total_episodes`, `total_frames`, `total_tasks`, `total_videos`, `total_chunks`, `chunks_size` (1000), `fps`, `splits`, `data_path`, `video_path`, `features` | `meta/info.json`: `codebase_version`, `fps` (integer), `features`, `total_episodes`, `total_frames`, `total_tasks`, `chunks_size`, `data_files_size_in_mb` (100), `video_files_size_in_mb` (200), `data_path`, `video_path`, `robot_type`, `splits` |
| data | `data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet`, one per episode; `episode_chunk = episode_index // chunks_size` | `data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet`, many episodes, one row group each |
| video | `videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4` | `videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4`, many episodes |
| episodes | `meta/episodes.jsonl`: `{episode_index, tasks, length}` | `meta/episodes/chunk-000/file-000.parquet`: `episode_index`, `tasks`, `length`, `data/chunk_index`, `data/file_index`, `dataset_from_index`, `dataset_to_index`, `videos/<key>/{chunk_index, file_index, from_timestamp, to_timestamp}`, `stats/<feature>/<stat>`, `meta/episodes/{chunk_index, file_index}` |
| tasks | `meta/tasks.jsonl`: `{task_index, task}` | `meta/tasks.parquet`: `task_index`, with `task` as the pandas index |
| stats | `meta/episodes_stats.jsonl`: `{episode_index, stats: {feature: {min, max, mean, std, count}}}` | per episode in the episodes table, aggregated in `meta/stats.json`; adds `q01, q10, q50, q90, q99` |

Every dataset has the default features `timestamp` (float32), `frame_index`, `episode_index`, `index`, `task_index`
(int64), each of shape `[1]`. A video feature has `dtype: "video"`, `shape: [h, w, 3]`, `names: ["height", "width",
"channels"]` and an `info` block (`video.height`, `video.width`, `video.codec`, `video.pix_fmt`, `video.fps`,
`video.channels`, `has_audio`; v2.1 `video.is_depth_map`, v3.0 `is_depth_map` and the encoder fields `video.g`,
`video.crf`, `video.preset`, `video.fast_decode`). The conversion script's comment block writes `file_000`; the
constants it imports and every path LeRobot builds give `file-000`, which is what DexVantage writes and reads.

**Which frame belongs to a row.** LeRobot looks a row's frame up by time: at v2.1 the frame at the row's `timestamp`
in the episode's own video; at v3.0 the frame at

    t_video = from_timestamp(episode, camera) + timestamp(row)

in the packed file, where `from_timestamp` is the summed duration of the episodes before it in that file
(`dataset_writer._save_episode_video`, `get_video_duration_in_s` = stream duration × time base).

**Reader.** `open_dataset(path)` takes the dataset folder, its `meta/info.json`, a Hub snapshot folder
(`…/snapshots/<revision>`) or a Hub cache folder (`datasets--org--name`, whose `refs/main` names the snapshot), or a
dataset on the Hugging Face Hub by name, `hf://datasets/<org>/<name>[@<revision>]` (or its huggingface.co page URL),
streamed file by file (`core/datasets/hub.py`, below). Any other `codebase_version` is refused with a sentence naming v2.1 and v3.0 (and the
conversion scripts LeRobot offers); so are v3.0's non-Parquet `storage_format` backends (Lance). The episode maps as:

| Episode | LeRobot |
|---|---|
| `t` | the `timestamp` column (float32 stored, float64 returned) |
| `q`, `joint_names` | `observation.state` and its feature `names` (a list, or a dict of lists such as `{"motors": […]}`; `[]` when not stated) |
| `action`, `action_names` | `action` and its names |
| `task` | the task of the episode's first row (`task_index`), all tasks in `meta['tasks']` |
| `meta['columns']` | every other data column (`frame_index`, `index`, `task_index`, extra features) |
| `cameras[key]` | each `video` feature, and `image` features (PNG bytes in the Parquet, decoded with OpenCV) |
| `CameraStream.timestamps` | the presentation times of the episode's frames in the video, `pts · time_base − from_timestamp`, read by demuxing only (no decoding) |

A camera's `timestamps` are the video's own times, not the rows', so a dropped frame shows as a gap and a late
camera as an offset against `Episode.t`, which is what the auditor compares. `frames(indices)` decodes with PyAV,
seeking to the keyframe before the first frame asked for, and matches frames by their exact pts; frames come back in
ascending index order, each once. Values are returned **as stored**: LeRobot records no units (DexVantage writes
radians and metres; many SO-100 datasets on the Hub store degrees), so `meta['units']` says so.

**Writer.** `write_lerobot(path, episodes, version='v3.0', fps=None, robot_type=None, features=None, *,
codec='auto', truth=None, overwrite=False, chunks_size=1000, data_file_size_mb=100, video_file_size_mb=200,
progress=None) -> dict` writes `observation.state` and `action` as float32 with the joint names, the default
features, extra features (`{key: {dtype, shape, names}}`, values in each episode's `meta['columns'][key]`), the
tasks, per-episode and aggregated stats as `compute_stats.py` defines them, and the videos:

- **Codec.** `codec='auto'` takes **AV1 with `libsvtav1`**, LeRobot's default, when the PyAV build has that encoder,
  otherwise **H.264 with `libx264`**; the summary's `codec` says which (`"AV1 (libsvtav1)"` or `"H.264
  (libx264)"`) and `info.json` records `video.codec`. Options are LeRobot's defaults (`configs/video.py`):
  yuv420p, `g=2`, `crf=30`, and `preset=12` for libsvtav1 at v3.0. The PyAV build in the development venv
  (av 19.0.2) has libsvtav1 (SVT-AV1 4.2.0, which maps preset 12 to 11) and decodes with libdav1d.
- **Frame times.** A camera's frames are written at **its own timestamps** (pts), so a synthetic camera that drops,
  duplicates or shifts frames keeps exactly those times. A camera whose every frame k sits at k/fps gets time base
  1/fps, as LeRobot's own files do; any other camera gets 1/90000 s (the MPEG 90 kHz clock), so its times come back
  within half a tick, 5.6 µs.
- **Packing at v3.0.** Each episode is encoded to its own MP4 first, as LeRobot does, and its packets are then
  remuxed, not re-encoded, into the camera's packed file, shifted by `from_timestamp`. An episode's duration is
  counted in whole ticks of the time base:

      duration = t_last + 1/fps,    from_timestamp(e+1) = from_timestamp(e) + duration(e)

  For regular cameras that is the duration LeRobot's `get_video_duration_in_s` measures (N/fps), so the file is the
  one LeRobot's concatenation produces. LeRobot concatenates with ffmpeg's concat demuxer, which subtracts each
  file's start time and would erase a late camera's offset. The remux keeps it (tested). H.264 at v3.0 is written
  without B-frames (`bf=0`, so pts = dts and packets shift cleanly). A new file starts when the packed file's bytes
  plus the episode's would reach `video_files_size_in_mb` (LeRobot's rule; the bytes counted are packet bytes),
  and a new data file when the uncompressed Arrow bytes would reach `data_files_size_in_mb`.
- **Stats.** Per feature, over the episode's rows: min, max, mean, std (population), count, and at v3.0 the
  quantiles q01–q99 (exact, `numpy.quantile`; LeRobot approximates them with 5000-bin histograms). Images: frames
  sampled as `sample_indices` does (`max(min(100, N), min(int(N^0.75), 10000))` evenly spaced), downsampled as
  `auto_downsample_height_width` does, per channel over (frame, height, width), divided by 255, shape (3, 1, 1); the
  count is the number sampled at v2.1 and N at v3.0. Aggregation is `aggregate_stats`: min of mins, max of maxes,
  the count-weighted mean, the pooled variance

      μ = Σ nᵢ μᵢ / Σ nᵢ,    σ² = Σ nᵢ (σᵢ² + (μᵢ − μ)²) / Σ nᵢ

  and for quantiles the envelope (min of q ≤ 50, max above), as LeRobot v3.0 does.
- **Ground truth.** `truth={'schema': 'dexvantage.truth/1', …}` is written as `dexvantage_truth.json` at the root and
  returned by `Dataset.truth()`; LeRobot ignores the file.
- **Refusals.** Each is a plain sentence: a non-whole fps (LeRobot stores an integer), odd frame sizes (yuv420p),
  feature names with `/`, episodes with different features or cameras, camera times that do not start at 0 s or
  later and increase, a non-empty folder (an existing LeRobot dataset is replaced only with `overwrite=True`; any
  other folder never is). A failed write removes the folder it created.

**Validation.** `formats.lerobot_roundtrip` writes three synthetic episodes (12, 9 and 8 rows, six joints, actions, a
two-column extra feature, two tasks, a 64 × 48 camera; the third camera drops a frame and runs one frame 13 ms late)
as v2.1 and as v3.0. It reads them back twice: with DexVantage's reader, and **independently** with `json` and
`pyarrow.parquet` on the files at the paths the format defines, the videos decoded by the `ffmpeg`/`ffprobe`
command-line tools (a separate FFmpeg build from PyAV's). Tolerances, and why: columns bit for bit after the float32
cast the format stores; camera times within half a 90 kHz tick (regular cameras are exact); frames at 30 dB PSNR or
better, because the videos are lossy (crf 30) and 30 dB, an RMS error of 8 grey levels, is below what was measured
with margin for other encoder builds; every frame closer to its own source than to a neighbour; info.json features,
names and totals as written. The `lerobot` package is not installed, so LeRobot's own loader is not run; the check
says so.

Measured 2026-11-07 (AV1, libsvtav1, av 19.0.2, CPU): 0 mismatches; worst-frame PSNR 35.65 dB (v2.1) and 33.24 dB
(v3.0, preset 12), identical from DexVantage's reader and from the ffmpeg CLI; the closest neighbouring source frame
was at least 15.3 dB (v2.1) and 12.7 dB (v3.0) worse than a frame's own; camera time error 0 s (v2.1) and
2.8 × 10⁻¹⁶ s (v3.0); the state against the float64 source 5.9 × 10⁻⁸ (float32 rounding). On a 64 × 48 / 128 × 96
test pattern, H.264 (libx264, crf 30) gave a worst frame of 36.7 dB. The check takes 0.5 s; no GPU is used.

**Known limits.** Compatibility is with the format as read from the pinned source; the `lerobot` package itself is
not a dependency and its loader is not run here. LeRobot records no units: values come back as stored. Intrinsics,
extrinsics and hardware timestamps are not part of the format (`None`). Depth videos (12-bit, `is_depth_map`) are
listed as cameras but their frames are not dequantised. v3.0 quantiles are exact where LeRobot's are histogram
estimates. Data files roll over by uncompressed Arrow bytes and video files by packet bytes, where LeRobot measures
the files on disk, so file boundaries can differ slightly from LeRobot's own for the same limits. The episodes
table is written as one file. The writer writes videos, not PNG image features (the reader reads both). Video bytes
streamed from the Hub are not kept on disk (a second pass reads them again unless they are still among the blocks
in memory).

**Hub streaming (`core/datasets/hub.py`).** The Hub's HTTP interface as `huggingface_hub` uses it (read from
github.com/huggingface/huggingface_hub at commit fd5dccb, main, 2026-11-05, read 2026-11-07: `constants.py`,
`file_download.hf_hub_url`, `hf_api.dataset_info` and `_iter_repo_tree_pages`, `utils/_pagination.py`):
`{endpoint}/api/datasets/{repo}/revision/{rev}` (the commit `sha`, `private`, `gated`, the card's licence),
`{endpoint}/api/datasets/{repo}/tree/{sha}/{path}?recursive=true&expand=false` (the file list, paged by a
`Link: <…>; rel="next"` header) and `{endpoint}/datasets/{repo}/resolve/{sha}/{path}` (a file); the endpoint is
`HF_ENDPOINT` or https://huggingface.co. Opening pins the revision to its commit and fetches `meta/info.json` and
the episode metadata (v2.1 `episodes.jsonl` and `tasks.jsonl`; v3.0 every file under `meta/episodes/` and
`tasks.parquet`); loading an episode fetches its data file, and its cameras' videos are read in place by HTTP range
requests (4 MiB blocks, the last 24 kept in memory: defaults), so a packed v3.0 video file is not downloaded whole (a
server that ignores ranges gets the whole file fetched instead); stats and the ground-truth sidecar only when asked
for. Measured on `lerobot/svla_so100_pickplace` (commit 728583b, 2026-11-07, a home connection): open 2.3 s (50
episodes); episode 10's two cameras 11.9 s and 33 MB by range requests, where their packed files are 449 MB; its 364
frames of one camera then decoded in 0.6 s. Files go to `XDG_CACHE_HOME/DexVantage/hub/<org>--<name>/<commit>/` (".part" then
renamed), so a second read is local. Only public datasets: no token is read or sent, and a gated or private dataset
is refused in one sentence, as are a missing dataset or revision and a network that cannot reach the Hub. The
dataset's `source` (repo, revision, commit, the card's licence) goes into the audit's `dataset` block. Tested against
a fake Hub that serves a synthetic dataset through the same URLs (`tests/test_spec_review_p458.py`: only metadata on
open, the episode's files on load, identical arrays and frames to the local read, a paged file list, a second read
from the cache); the real Hub was not contacted by the tests.

## ROS 2 bags

Modules: `dexvantage/core/rosbag.py` (the CDR codec, the MCAP writer, the MCAP and SQLite readers; ported from
Vantage `core/rosbag.py`) and `dexvantage/core/datasets/bag.py` (a bag as a `Dataset`; formats `mcap` and
`rosbag2-sqlite`). Check `formats.mcap_roundtrip`. Tests `tests/test_bag_reader.py`. Worked example:
[../guide/bags_rlds.md](../guide/bags_rlds.md).

**What is read.** A rosbag2 bag (a folder with `metadata.yaml` and its `.mcap` or `.db3` files, split bags merged in
time order), one `.mcap` or `.db3` file, or a folder of bags (each sub-folder bag and each loose bag file is one
episode, in natural name order). No ROS installation: numpy and the standard library, PyYAML for `metadata.yaml`,
OpenCV for compressed images, and `zstandard` / `lz4` (the `bags` extra, `pip install "dexvantage[bags]"`) only for
compressed MCAP chunks; without them the reader says so in a sentence.

**Wire formats.** MCAP as specified at https://mcap.dev/spec (read 2026-11-07): chunks streamed one at a time and
merged into log-time order, their CRC-32 checked; a file without a summary (a recorder that was killed) is read by a
linear scan that stops at the first incomplete record and says so. SQLite bags: the `topics` and `messages` tables
rosbag2's sqlite3 plugin writes, with `message_definitions` (Iron and later) when present. Messages are XCDR1 "plain
CDR" (OMG DDS-XTypes 1.3 §7.4): a 4-byte encapsulation header, every primitive aligned to its own size measured from
the end of the header, strings as a uint32 length that counts the NUL, sequences as a uint32 count. A bag's own
message definitions (MCAP schema, `message_definitions`) take precedence, so vendor types and other distributions
decode as recorded. The types a robot-learning bag holds are built in: `sensor_msgs/JointState`, `Image`,
`CompressedImage`, `CameraInfo`, `TimeReference`, `tf2_msgs/TFMessage`, `std_msgs/Header`, `String`, the multi-array
types, and the geometry types they use.

**Episode mapping.**

| Episode field | From |
|---|---|
| `t`, `q`, `joint_names` | `/joint_states` (else the JointState topic with the most messages, command topics last). `t` is `header.stamp`, or the log time when the driver leaves the stamp at zero (`meta['t_source']`). `q` holds the union of joint names in first-seen order; a message that leaves a joint out holds its previous value (`meta['q_held']` counts them), a joint not yet seen is NaN. |
| `cameras[topic]` | every `Image` / `CompressedImage` topic with 8-bit colour, mono, Bayer or YUV 4:2:2 pixels. `timestamps` = header stamps, `meta['log_times']` = receive times. Depth, 16-bit and float images go to `meta['depth_streams']`. |
| `hw_timestamps` | a `sensor_msgs/TimeReference` in the camera's namespace (`time_ref` of the frame whose stamp is `header.stamp`), or the header stamps themselves when they run on another clock than the joint states (see below). Seconds from that camera's first device time; `meta['hw_t0_s']` keeps the raw first value. |
| `intrinsics` | the first `CameraInfo` beside the image topic (`<ns>/camera_info`), or with the same `frame_id`: `plumb_bob` and `rational_polynomial` → `brown` (`pinhole` when every coefficient is zero), `equidistant` → `kb`; an all-zero K (uncalibrated) → `None`. |
| `extrinsics` | `/tf_static`: the pose of the image's `frame_id` in the top frame reachable through static transforms (`meta['tf_chain']`): the base or world frame of a fixed camera, the link a wrist camera is bolted to. |
| `task`, `robot_type` | a `String` topic named `task`, `instruction`, `language_instruction` or `prompt` (else `metadata.yaml` `custom_data`); the robot name in `/robot_description`. |

All times are seconds from the episode start t₀ (the earliest joint-state or camera time), computed in integer
nanoseconds before the conversion, `t = (stamp_ns − t₀) / 10⁹`, so an epoch time keeps nanosecond resolution.

**Two clocks.** A camera driver may stamp frames with the device's clock (seconds since power-on) rather than the
host's. With o = median(header stamp − log time) of a stream, a camera whose o differs from the joint states' o_j by
more than `CLOCK_SPLIT_S` (default 10 s, a labelled default; transport delays are milliseconds, clock-domain offsets
are hours to decades) has its header stamps moved to `hw_timestamps`, and its `timestamps` become log time + o_j:
the receive time carried onto the joint-state clock (`meta['timestamp_source']` says which was used).

**Validation** (`formats.mcap_roundtrip`, tolerance 0 mismatches: a container either carries every byte or is
wrong). A two-camera robot episode (321 messages in the full run, 65 in the quick run) is written with every chunk
compression installed (none, zstd, lz4) and read back by (1) an MCAP walk written again in the check from the
specification, checking every chunk's CRC-32, the data-section CRC and the summary CRC; (2) the `mcap` package
(Foxglove, version 1.6.0) with CRC validation, when installed; (3) one JointState laid out byte by byte with `struct`
in XCDR1; and the dataset reader must return the joint states, stamps and static camera pose exactly. Measured
2026-11-07: 0 mismatches, 0 CRC failures, hand CDR layout identical, in 0.06 s. A reader that drops a stamp's
nanoseconds fails it (a test). The tests also cross-check the codec with the `mcap` package both ways, read SQLite
bags written with `sqlite3` alone, rebuild the static camera poses from the quaternions by hand, and decode JPEG,
PNG, rgb8, bgr8, mono8 and Bayer frames.

**Speed** (AMD Ryzen 9 9900X, files in the page cache): an episode of 1000 joint states and 300 rgb8 640×480 frames
(277 MB) loads in 0.09 s (headers only: the pixel data is not decoded) and all 300 frames decode in 0.12 s.

**Known limits.** One robot state vector from one JointState topic; actions are not read from bags (no standard
action message; command topics are listed in `meta['other_joint_topics']`); dynamic `/tf` is not used for
extrinsics; vendor frame-metadata topics (exposure, gain) are not decoded; a `frame_id` that is the camera's body
frame rather than its optical frame gives that frame's pose (`meta['frame_id']`); the camera `fps` is `None` (a bag
does not state a nominal rate); rosbag2 file compression (a whole `.mcap.zstd`) is refused with the command that
undoes it (`ros2 bag decompress`); XCDR2 and `wstring` are not read; a bag with no JointState topic has
no reference clock, so each camera keeps its header stamps as recorded.

## RLDS

Module: `dexvantage/core/datasets/rlds.py` (format `rlds`). Check `formats.tfrecord_crc`. Tests
`tests/test_rlds_reader.py` (fixtures written by `tests/tfrecord_writer.py`, which shares no code with the reader).

**What is read.** An RLDS / Open X-Embodiment dataset in TensorFlow Datasets' on-disk form — a folder of
`<name>-<split>.tfrecord-XXXXX-of-YYYYY` shards with `dataset_info.json` and `features.json` (or its parent, whose
highest version folder is used) — or a single TFRecord file. No TensorFlow and no protobuf runtime: the framing,
the CRC and the protobuf messages are decoded by hand from their definitions.

**Sources** (read 2026-11-07): TFRecord framing, `xla/tsl/lib/io/record_writer.h` and the mask in
`xla/tsl/lib/hash/crc32c.h` (https://github.com/openxla/xla) and https://www.tensorflow.org/tutorials/load_data/tfrecord;
`tensorflow/core/example/example.proto` and `feature.proto` (https://github.com/tensorflow/tensorflow); the protobuf
wire format, https://protobuf.dev/programming-guides/encoding/; TFDS's serialisation,
`tensorflow_datasets/core/example_serializer.py` and `core/features/{image_feature, tensor_feature, dataset_feature,
sequence_feature}.py` (https://github.com/tensorflow/datasets); RLDS's episode and step fields, the README of
https://github.com/google-research/rlds (Ramos et al. 2021, arXiv:2111.02767); and the published `features.json` /
`dataset_info.json` of three Open X-Embodiment datasets (`fractal20220817_data`, `bridge`, `berkeley_autolab_ur5`, at
`https://storage.googleapis.com/gresearch/robotics/<name>/0.1.0/`), whose layout the test fixtures follow.

**TFRecord and CRC-32C.** A record is `uint64 length | uint32 masked_crc(length) | data | uint32 masked_crc(data)`,
little-endian, with masked(c) = ((c ≫ 15) | (c ≪ 17)) + 0xa282ead8 mod 2³². The CRC is CRC-32C (Castagnoli,
reflected polynomial 0x82f63b78, initial value and final XOR 0xffffffff). Both CRCs of every record read are
verified; listing episodes reads only the 12-byte headers. The CRC is linear over GF(2), which the implementation
uses to run in numpy: a buffer of n bytes is cut into B blocks of L ≈ √n bytes whose registers advance together
(one table look-up per byte position for all blocks), then combined in order with r ← Z_L(r) ⊕ c_k, where Z_L, "the
register after L zero bytes", is a fixed linear map tabulated once per L. 100 MB take 1.0 s on a Ryzen 9 9900X (the
table-driven byte-by-byte Python loop, measured on 4 MB, would take 4.9 s).
GZIP-compressed TFRecords are read through Python's `gzip`.

**Example and SequenceExample.** `Example{Features features = 1}`, `SequenceExample{Features context = 1;
FeatureLists feature_lists = 2}`, `Features{map<string, Feature> feature = 1}` (map entries: key = 1, value = 2),
`Feature{oneof: BytesList = 1, FloatList = 2, Int64List = 3}`. Packed and unpacked repeated encodings are both read
(the encoding guide requires parsers to accept either); int64 varints are two's complement.

**TFDS features.** TFDS flattens the nested feature dictionary with `/` and stores the RLDS `steps` sequence stacked:
a `(3,)` float32 action over N steps is one FloatList of 3N values, an image is one encoded JPEG or PNG per step,
`bool` is int64, `uint64` is bit-cast to int64, a Tensor with encoding `bytes` or `zlib` is one (zlib-compressed)
raw buffer per step, and a float32 image is a 4-channel PNG of its bytes. `features.json` gives each leaf's class,
shape, dtype and encoding; without it shapes are inferred from N (from `steps/is_first`).

**Episode mapping.** One record is one episode, id `<split>-<k>`. `q` is the observation's state vector — the first
of `joint_position(s)`, `joint_pos`, `joint_state(s)`, `qpos`, `state`, `robot_state`, `proprio` present — with
components named `<feature>_<i>`, because RLDS stores no joint names (Open X-Embodiment `state` vectors are often
end-effector poses, not joint angles; `meta['state_feature']` names the source). `action` is the `action` tensor, or
the `action/*` sub-features concatenated in `features.json` order. `task` is the language instruction (a step
feature or episode metadata). Every 8-bit image feature is a `CameraStream`, decoded lazily with OpenCV to RGB; float
and 16-bit images are listed in `meta['depth_streams']`. `meta['rlds']` holds `is_first`, `is_last`, `is_terminal`,
`reward`, `discount`; a flag not set on exactly the first / last step is a warning in `meta['warnings']`.
`meta['episode_metadata']` holds the episode-level features.

**Time.** RLDS records no times in general. `t` is the step index divided by the fps passed to the reader
(`RldsDataset(path, fps=...)`, a user setting, never a guess); without it `t` counts steps and
`meta['time_base'] = 'steps'`. A float step feature named `timestamp` is used instead, read as seconds
(`meta['time_source']` says so).

**Validation** (`formats.tfrecord_crc`, tolerance 0 mismatches: a checksum is exact). The reader's CRC-32C against
the five test vectors of RFC 3720 appendix B.4 and the CRC catalogue's check value crc32c("123456789") = 0xe3069283;
against a bit-at-a-time Castagnoli CRC written in the check on random buffers on both sides of the vectorised path's
8 KiB threshold (up to 150 kB); the masked CRC against TensorFlow's formula on 200 random values; and TFRecord framing
written in the check, read back, with a damaged copy that must be refused. Measured 2026-11-07: 0 mismatches in
0.11 s. CRC-32 (zlib's polynomial) in its place fails it (a test). The tests also parse messages serialised by
Google's protobuf runtime (descriptors built from `example.proto` / `feature.proto` at run time, no TensorFlow),
check that runtime reads the test writer's bytes, and read whole TFDS-style datasets back exactly (PNG frames bit for
bit, JPEG within 3 grey levels).

**Known limits.** float64 tensors that TFDS stored in a FloatList keep float32 precision; float32 and uint16 images
(depth) are not exposed as cameras; ragged (multi-level) sequences are left out with a note; intrinsics, extrinsics
and hardware timestamps are `None` (RLDS has no standard place for them); a GZIP TFRecord is decompressed from the
start for every episode, so it is slow to browse; Hub / GCS datasets are read only from a local copy.
