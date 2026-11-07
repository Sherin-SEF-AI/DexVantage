"""LeRobot datasets, codebase versions v2.1 and v3.0: a reader (format 'lerobot') into the shared Episode shape and a
writer, write_lerobot(), whose output LeRobot's own loader conventions accept.

The format is taken from the LeRobot source itself, read on 2026-11-07:

- v3.0: https://github.com/huggingface/lerobot at commit d490e472a5d1a207cfe0255caaaf74b6d66e4656 (main,
  2026-11-05): src/lerobot/datasets/utils.py (path templates, DatasetInfo), dataset_metadata.py (CODEBASE_VERSION
  "v3.0", the episodes table), dataset_writer.py (packed data and video files, from/to timestamps), io_utils.py
  (tasks.parquet, stats.json), compute_stats.py (per-episode stats with quantiles), video_utils.py (PyAV encode and
  decode, get_video_info), configs/video.py (encoder defaults: libsvtav1, yuv420p, g=2, crf=30, preset 12),
  utils/constants.py (DEFAULT_FEATURES), scripts/convert_dataset_v21_to_v30.py and docs/source/lerobot-dataset-v3.mdx.
- v2.1: the same repository at tag v0.3.3 (commit b883328e6c95681ca90a18b102e4ae5e1f91e2bf, the last release whose
  CODEBASE_VERSION is "v2.1"): src/lerobot/datasets/utils.py, lerobot_dataset.py, compute_stats.py, video_utils.py.

Layouts, exactly as those files define them:

    v2.1  meta/info.json            codebase_version "v2.1", fps, features, total_* counters, chunks_size,
                                    data_path "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                                    video_path "videos/chunk-{episode_chunk:03d}/{video_key}/
                                                episode_{episode_index:06d}.mp4" (one line in the file)
          meta/episodes.jsonl       {"episode_index", "tasks": [str], "length"} per line
          meta/tasks.jsonl          {"task_index", "task"} per line
          meta/episodes_stats.jsonl {"episode_index", "stats": {feature: {min, max, mean, std, count}}} per line
          episode_chunk = episode_index // chunks_size; one Parquet and one MP4 per camera per episode; the video's
          frame for a row is the one at the row's timestamp.
    v3.0  meta/info.json            codebase_version "v3.0", data_files_size_in_mb, video_files_size_in_mb,
                                    data_path "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                                    video_path "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
          meta/tasks.parquet        task_index, indexed by task (pandas index metadata)
          meta/stats.json           aggregated stats (min, max, mean, std, count, q01, q10, q50, q90, q99)
          meta/episodes/chunk-XXX/file-XXX.parquet  one row per episode: episode_index, tasks, length,
                                    data/chunk_index, data/file_index, dataset_from_index, dataset_to_index,
                                    videos/<key>/{chunk_index, file_index, from_timestamp, to_timestamp},
                                    stats/<feature>/<stat>, meta/episodes/{chunk_index, file_index}
          many episodes per data and video file; an episode's frame for a row is at from_timestamp + its timestamp.

(The conversion script's comment block writes "file_000"; the constants it imports, and every path LeRobot builds,
give "file-000". The constants are followed here.)

Reader. Episode.t is the data's `timestamp` column; q the `observation.state` column with the feature's names as
joint_names; action the `action` column; every other column is kept in Episode.meta['columns']. Each video feature is
a CameraStream whose timestamps are the presentation times of the episode's frames in its video file (seconds from
the episode start: pts·time_base − from_timestamp), read by demuxing only; frames() decodes lazily with PyAV, seeking
to the keyframe before the first frame asked for, and matches frames by their exact pts. A camera stored as PNG
images in the Parquet (dtype 'image') is decoded with OpenCV. A Hub dataset already downloaded is read as a local
snapshot folder (the dataset root, a `snapshots/<revision>` folder or the `datasets--org--name` cache folder).
Any codebase_version other than v2.1 and v3.0, and v3.0's non-Parquet storage backends, are refused with a sentence.
A dataset on the Hugging Face Hub is read by name, 'hf://datasets/<org>/<name>[@<revision>]' (or its huggingface.co
page URL), streamed file by file into a local cache as the reader needs them (core/datasets/hub.py: the metadata on
open, an episode's data and video files when it is loaded; public datasets only; the commit is pinned at open and
reported in `source`).

Writer. write_lerobot(path, episodes, version, fps, robot_type, features) writes info.json (features with dtype, shape
and names, the default timestamp/frame_index/episode_index/index/task_index features, video info), the tasks, the
per-episode and aggregated stats, the Parquet data (float32 state and action, one row group per episode) and the
videos: AV1 with libsvtav1 (LeRobot's default) when PyAV has that encoder, else H.264 with libx264, with LeRobot's
default options (yuv420p, g=2, crf=30; preset 12 for libsvtav1 at v3.0); the codec used is returned and stored in
info.json. A camera's frames are written at its own timestamps (pts), so a synthetic camera that drops, duplicates
or shifts frames keeps exactly those times; regular cameras (frame k at k/fps) get time base 1/fps as LeRobot's own
files do, irregular ones 1/90000 s. Each episode is encoded on its own, as LeRobot does. At v3.0 its packets are then
remuxed (not re-encoded) into the camera's packed file, shifted by from_timestamp = the summed durations of the episodes
before it in the file (a duration being the last frame's time + 1/fps, which is what LeRobot's get_video_duration_in_s
gives for its regular per-episode files). For regular cameras that is the file LeRobot's concatenation produces; unlike
ffmpeg's concat demuxer, which LeRobot uses, it does not erase the start offset of a camera that starts late. At v3.0
H.264 is written without B-frames (bf=0, so pts = dts and the packets shift cleanly); the files roll over when the
packed file's bytes plus the episode's would reach video_files_size_in_mb, LeRobot's rule. A ground-truth sidecar
(dexvantage_truth.json) is written when given.

Known limits: the real `lerobot` package is not a dependency, so compatibility is with the format as read from the
pinned source, not tested against LeRobot's loader here; LeRobot stores no units, so joint values are returned as
stored (DexVantage writes radians and metres; many SO-100 datasets store degrees); intrinsics, extrinsics and
hardware timestamps are not part of the format (None); depth videos (12-bit, is_depth_map) are listed but their
frames are not dequantised; v3.0 quantiles are exact (numpy) where LeRobot approximates them with histograms; data
files roll over by the uncompressed Arrow bytes and video files by packet bytes (LeRobot measures the files on disk);
the episodes table is written as one file; image (PNG) features are read but not written (the writer writes videos).
"""
from __future__ import annotations

import json
import math
import os
import shutil
from fractions import Fraction

import numpy as np

from .episode import CameraStream, Dataset, DatasetError, Episode, EpisodeRef, register_reader

SUPPORTED = ('v2.1', 'v3.0')
SOURCE_COMMIT = 'd490e472a5d1a207cfe0255caaaf74b6d66e4656'
SOURCE_COMMIT_V21 = 'b883328e6c95681ca90a18b102e4ae5e1f91e2bf'
SOURCE_URL = 'https://github.com/huggingface/lerobot'
SOURCE_READ = '2026-11-07'

INFO_PATH = 'meta/info.json'
STATS_PATH = 'meta/stats.json'
TRUTH_FILE = 'dexvantage_truth.json'
TRUTH_SCHEMA = 'dexvantage.truth/1'
# v2.1 (tag v0.3.3, datasets/utils.py)
V21_EPISODES = 'meta/episodes.jsonl'
V21_TASKS = 'meta/tasks.jsonl'
V21_EPISODES_STATS = 'meta/episodes_stats.jsonl'
V21_DATA_PATH = 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet'
V21_VIDEO_PATH = 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'
# v3.0 (main at SOURCE_COMMIT, datasets/utils.py)
CHUNK_FILE_PATTERN = 'chunk-{chunk_index:03d}/file-{file_index:03d}'
V30_TASKS = 'meta/tasks.parquet'
V30_EPISODES_DIR = 'meta/episodes'
V30_EPISODES_PATH = V30_EPISODES_DIR + '/' + CHUNK_FILE_PATTERN + '.parquet'
V30_DATA_PATH = 'data/' + CHUNK_FILE_PATTERN + '.parquet'
V30_VIDEO_PATH = 'videos/{video_key}/' + CHUNK_FILE_PATTERN + '.mp4'
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_DATA_FILE_SIZE_IN_MB = 100
DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200
DEFAULT_FEATURES = {                       # utils/constants.py (v3.0) and datasets/utils.py (v2.1): identical
    'timestamp': {'dtype': 'float32', 'shape': [1], 'names': None},
    'frame_index': {'dtype': 'int64', 'shape': [1], 'names': None},
    'episode_index': {'dtype': 'int64', 'shape': [1], 'names': None},
    'index': {'dtype': 'int64', 'shape': [1], 'names': None},
    'task_index': {'dtype': 'int64', 'shape': [1], 'names': None},
}
QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)  # compute_stats.DEFAULT_QUANTILES (v3.0)
STATE_KEY, ACTION_KEY = 'observation.state', 'action'
ENCODER_DEFAULTS = {'pix_fmt': 'yuv420p', 'g': 2, 'crf': 30, 'preset_svtav1_v30': 12}   # configs/video.py
FINE_TIME_BASE = Fraction(1, 90000)        # irregular cameras (the MPEG 90 kHz clock)


# ---------------------------------------------------------------------------------------------------------------
# optional packages


def _pa():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise DatasetError('Reading or writing LeRobot datasets needs the Python package pyarrow (install the lerobot '
                           'extra: pip install "dexvantage[lerobot]").') from None
    return pa, pq


def _av():
    try:
        import av
    except ImportError:
        raise DatasetError('LeRobot videos need the Python package av (PyAV; install the lerobot extra: pip install '
                           '"dexvantage[lerobot]").') from None
    return av


# ---------------------------------------------------------------------------------------------------------------
# finding the dataset root (a plain folder or a local Hugging Face Hub snapshot)


def resolve_root(path: str) -> str | None:
    """The dataset root (the folder holding meta/info.json) for a dataset folder, its meta/info.json, a Hub snapshot
    folder or a Hub cache folder (datasets--org--name, whose refs/main names the snapshot); None when there is none."""
    p = os.path.abspath(path)
    if os.path.isfile(p) and os.path.basename(p) == 'info.json':
        p = os.path.dirname(os.path.dirname(p))
    if os.path.isfile(os.path.join(p, INFO_PATH)):
        return p
    snaps = os.path.join(p, 'snapshots')
    if os.path.isdir(snaps):
        ref = os.path.join(p, 'refs', 'main')
        if os.path.isfile(ref):
            with open(ref, encoding='utf-8') as fh:
                rev = fh.read().strip()
            if os.path.isfile(os.path.join(snaps, rev, INFO_PATH)):
                return os.path.join(snaps, rev)
        cands = [os.path.join(snaps, d) for d in os.listdir(snaps) if os.path.isfile(os.path.join(snaps, d, INFO_PATH))]
        if cands:
            return max(cands, key=lambda d: os.path.getmtime(os.path.join(d, INFO_PATH)))
    return None


def sniff(path: str) -> bool:
    """A folder with meta/info.json that states features (any codebase_version: the opener refuses the unsupported
    ones with a sentence)."""
    root = resolve_root(path)
    if root is None:
        from .hub import is_hub_ref
        return is_hub_ref(path)                 # a Hub dataset: the opener fetches and checks its info.json
    try:
        with open(os.path.join(root, INFO_PATH), encoding='utf-8') as fh:
            info = json.load(fh)
    except ValueError:                         # a damaged info.json: the opener says so in words
        return True
    except OSError:
        return False
    return isinstance(info, dict) and isinstance(info.get('features'), dict)


# ---------------------------------------------------------------------------------------------------------------
# Arrow to numpy


def _to_numpy(col):
    """An Arrow column as numpy: scalars (N,), fixed or equal-length lists (N, k, ...); strings, structs and ragged
    lists as Python lists."""
    pa, _ = _pa()
    arr = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    t = arr.type
    if pa.types.is_fixed_size_list(t) or pa.types.is_list(t) or pa.types.is_large_list(t):
        n = len(arr)
        if pa.types.is_fixed_size_list(t):
            width = t.list_size
        else:
            lens = np.diff(np.asarray(arr.offsets))
            if n and not np.all(lens == lens[0]):
                return arr.to_pylist()
            width = int(lens[0]) if n else 0
        inner = _to_numpy(arr.flatten())
        if isinstance(inner, list):
            return arr.to_pylist()
        return np.asarray(inner).reshape((n, width) + np.asarray(inner).shape[1:])
    if pa.types.is_struct(t) or pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_binary(t):
        return arr.to_pylist()
    return arr.to_numpy(zero_copy_only=False)


def _names(ft: dict, width: int) -> list:
    """A feature's names as a flat list (names may be a list, a dict of lists such as {'motors': [...]}, or None);
    [] when they are not stated or do not match the width."""
    n = ft.get('names') if isinstance(ft, dict) else None
    if isinstance(n, dict):
        flat = [x for v in n.values() if isinstance(v, (list, tuple)) for x in v]
    elif isinstance(n, (list, tuple)):
        flat = list(n)
    else:
        flat = []
    flat = [str(x) for x in flat]
    return flat if len(flat) == width else []


# ---------------------------------------------------------------------------------------------------------------
# video: presentation times and lazy decoding


def video_pts(file: str, from_ts: float = 0.0, to_ts: float | None = None, fps: float | None = None):
    """(pts as int64 array, time base) of the frames whose presentation time lies in [from_ts, to_ts) in a video file,
    read by demuxing only (no decoding)."""
    av = _av()
    eps = 1e-4 if not fps else min(1e-4, 0.25 / float(fps))
    margin = max(1.0, 32.0 / float(fps or 30.0))      # B-frame reordering: dts runs ahead of pts by < this
    out = []
    with av.open(file() if callable(file) else file) as c:
        s = c.streams.video[0]
        tb = s.time_base
        if from_ts > 0:
            c.seek(max(0, int(math.floor(from_ts / tb)) - 1), backward=True, any_frame=False, stream=s)
        for p in c.demux(s):
            if p.pts is None or p.size == 0:
                continue
            t = float(p.pts * tb)
            if to_ts is not None and p.dts is not None and float(p.dts * tb) > to_ts + margin:
                break
            if t >= from_ts - eps and (to_ts is None or t < to_ts - eps):
                out.append(int(p.pts))
    return np.unique(np.asarray(out, dtype=np.int64)), tb


def _video_decoder(file: str, pts: np.ndarray, key: str):
    """decode(indices or None) -> iterator of (frame index, HxWx3 uint8 RGB), frames matched by exact pts, in
    ascending index order."""
    def decode(indices):
        av = _av()
        n = len(pts)
        want_idx = list(range(n)) if indices is None else sorted({int(i) for i in indices})
        bad = [i for i in want_idx if not 0 <= i < n]
        if bad:
            raise DatasetError(f'Camera {key} has {n} frames in this episode; frame {bad[0]} does not exist.')
        if not want_idx:
            return
        with av.open(file() if callable(file) else file) as c:
            s = c.streams.video[0]
            s.thread_type = 'AUTO'
            tb = s.time_base
            groups, cur = [], [want_idx[0]]                 # re-seek across gaps of more than 2 s
            for i in want_idx[1:]:
                if float((pts[i] - pts[cur[-1]]) * tb) > 2.0:
                    groups.append(cur)
                    cur = [i]
                else:
                    cur.append(i)
            groups.append(cur)
            for g in groups:
                want = {int(pts[i]): i for i in g}
                last = int(pts[g[-1]])
                c.seek(max(0, int(pts[g[0]]) - 1), backward=True, any_frame=False, stream=s)
                for fr in c.decode(s):
                    if fr.pts is None:
                        continue
                    if fr.pts in want:
                        yield want.pop(fr.pts), fr.to_ndarray(format='rgb24')
                    if fr.pts >= last or not want:
                        break
                if want:
                    raise DatasetError(f'{len(want)} frame(s) of camera {key} could not be decoded from '
                                       f'{os.path.basename(getattr(file, "name", file))} (the file may be '
                                       'truncated).')
    return decode


def _image_decoder(cells: list, key: str):
    """decode for a camera stored as PNG/JPEG bytes in the Parquet (dtype 'image': {'bytes', 'path'} cells)."""
    def decode(indices):
        try:
            import cv2
        except ImportError:
            raise DatasetError(f'Camera {key} is stored as images in the Parquet; decoding them needs OpenCV '
                               '(opencv-python).') from None
        n = len(cells)
        for i in (range(n) if indices is None else sorted({int(i) for i in indices})):
            if not 0 <= i < n:
                raise DatasetError(f'Camera {key} has {n} frames in this episode; frame {i} does not exist.')
            cell = cells[i] or {}
            data = cell.get('bytes') if isinstance(cell, dict) else None
            if not data:
                raise DatasetError(f'Frame {i} of camera {key} holds no image bytes.')
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise DatasetError(f'Frame {i} of camera {key} is not an image OpenCV can decode.')
            yield i, img[:, :, ::-1].copy()
    return decode


# ---------------------------------------------------------------------------------------------------------------
# the reader


def _read_json(path: str):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def _read_jsonl(path: str) -> list:
    out = []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class _Reopen:
    """A video file streamed from the Hub: each call gives a fresh RemoteFile at the start (av.open takes it)."""

    def __init__(self, hub, rel: str, size: int):
        self.hub, self.name, self.size = hub, rel, size

    def __call__(self):
        from .hub import RemoteFile
        return RemoteFile(self.hub, self.name, self.size)


class LeRobotDataset(Dataset):
    """A LeRobot dataset folder (v2.1 or v3.0) opened for reading. Additive to the Dataset shape: tasks (index ->
    text), stats() (the dataset's aggregated stats as stored) and episode_stats(i)."""
    format = 'lerobot'

    def __init__(self, path: str, *, opener=None, cache: str | None = None):
        from .hub import HubMirror, parse_hub_ref
        self.hub = None
        ref = parse_hub_ref(path) if resolve_root(path) is None else None
        if ref is not None:                     # a Hub dataset, fetched file by file (core/datasets/hub.py)
            self.hub = HubMirror(ref[0], ref[1], cache=cache, opener=opener)
            self.hub.fetch(INFO_PATH)
            root, name = self.hub.root, ref[0]
        else:
            root = resolve_root(path)
            name = os.path.basename(os.path.normpath(path)) or path
        if root is None:
            raise DatasetError(f'{name} is not a LeRobot dataset (there is no meta/info.json).')
        try:
            info = _read_json(os.path.join(root, INFO_PATH))
        except (OSError, ValueError):
            raise DatasetError(f'The meta/info.json of {name} cannot be read as JSON.') from None
        version = info.get('codebase_version') if isinstance(info, dict) else None
        if not version:
            raise DatasetError(f'{name} does not state its LeRobot codebase_version in meta/info.json; DexVantage '
                               f'reads versions {" and ".join(SUPPORTED)}.')
        if version not in SUPPORTED:
            raise DatasetError(f'{name} is a LeRobot dataset of codebase version {version}; DexVantage reads versions '
                               f'{" and ".join(SUPPORTED)} only (LeRobot\'s own scripts convert older ones: v2.0 '
                               'to v2.1 in its release v0.3.3, v2.1 to v3.0 in '
                               'lerobot.scripts.convert_dataset_v21_to_v30).')
        if info.get('storage_format') not in (None, 'parquet'):
            raise DatasetError(f'{name} stores its data as {info.get("storage_format")}; DexVantage reads the Parquet '
                               'and MP4 layout only.')
        _pa()
        self.path, self.info, self.version, self.root = root, info, version, root
        if self.hub is not None:
            self.path = self.hub.ref                # the dataset's name, not the cache folder
        self.features = info.get('features') or {}
        self.fps = float(info['fps']) if info.get('fps') else None
        self._episodes: list[dict] = []
        self.tasks: dict[int, str] = {}
        self._ep_stats: dict[int, dict] | None = None
        try:
            if version == 'v2.1':
                self._load_v21_meta()
            else:
                self._load_v30_meta()
        except DatasetError:
            raise
        except (OSError, ValueError, KeyError) as e:
            raise DatasetError(f'The metadata of {name} is incomplete or damaged ({type(e).__name__}: {e}).') from None

    # --- metadata
    def _need(self, rel: str, optional: bool = False):
        """A Hub dataset fetches the file rel now (a local folder has it already)."""
        if self.hub is not None:
            self.hub.fetch(rel, optional=optional)

    @property
    def source(self) -> dict:
        """Where the dataset was read from: a folder, or a Hub repository at a pinned commit (with its licence as
        the dataset card states it)."""
        if self.hub is None:
            return {'kind': 'folder'}
        return {'kind': 'hub', 'repo_id': self.hub.repo_id, 'revision': self.hub.revision, 'commit': self.hub.sha,
                'license': self.hub.license}

    def _load_v21_meta(self):
        r = self.root
        for f in (V21_EPISODES, V21_TASKS):
            self._need(f, optional=True)
        for f in (V21_EPISODES, V21_TASKS):
            if not os.path.isfile(os.path.join(r, f)):
                raise DatasetError(f'This LeRobot v2.1 dataset has no {f}.')
        eps = sorted(_read_jsonl(os.path.join(r, V21_EPISODES)), key=lambda d: d['episode_index'])
        self._episodes = eps
        self.tasks = {int(d['task_index']): str(d['task']) for d in _read_jsonl(os.path.join(r, V21_TASKS))}

    def _load_v30_meta(self):
        pa, pq = _pa()
        r = self.root
        if self.hub is not None:
            for rel in self.hub.list(V30_EPISODES_DIR):
                if rel.endswith('.parquet'):
                    self._need(rel)
            self._need(V30_TASKS, optional=True)
        d = os.path.join(r, V30_EPISODES_DIR)
        files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(d) for f in fs if f.endswith('.parquet'))
        if not files:
            raise DatasetError('This LeRobot v3.0 dataset has no meta/episodes/*/*.parquet.')
        rows = []
        for f in files:
            names = pq.read_schema(f).names
            t = pq.read_table(f, columns=[c for c in names if not c.startswith('stats/')])
            rows += t.to_pylist()
        self._episodes = sorted(rows, key=lambda e: e['episode_index'])
        tp = os.path.join(r, V30_TASKS)
        if os.path.isfile(tp):
            t = pq.read_table(tp)
            if 'task' in t.column_names:
                texts = t.column('task').to_pylist()
            else:                                        # index stored without pandas metadata
                texts = t.column(t.column_names[-1]).to_pylist()
            self.tasks = {int(i): str(s) for i, s in zip(t.column('task_index').to_pylist(), texts)}

    def episodes(self) -> list[EpisodeRef]:
        return [EpisodeRef(int(e['episode_index']), f'episode_{int(e["episode_index"]):06d}',
                           int(e['length']) if e.get('length') is not None else None,
                           {'tasks': list(e.get('tasks') or [])}) for e in self._episodes]

    def _row(self, index: int) -> dict:
        for e in self._episodes:
            if int(e['episode_index']) == int(index):
                return e
        raise DatasetError(f'This dataset has no episode {index} (it has {len(self._episodes)}).')

    def stats(self) -> dict | None:
        """The dataset's stats as stored (v3.0 meta/stats.json; v2.1 aggregated from episodes_stats.jsonl the way
        LeRobot's aggregate_stats does), as nested lists."""
        p = os.path.join(self.root, STATS_PATH)
        if self.version == 'v3.0':
            self._need(STATS_PATH, optional=True)
        if self.version == 'v3.0' and os.path.isfile(p):
            return _read_json(p)
        eps = self._all_episode_stats()
        if not eps:
            return None
        return _jsonable(aggregate_stats([_np_stats(s) for s in eps.values()]))

    def episode_stats(self, index: int) -> dict | None:
        return (self._all_episode_stats() or {}).get(int(index))

    def _all_episode_stats(self) -> dict:
        if self._ep_stats is not None:
            return self._ep_stats
        out = {}
        if self.version == 'v2.1':
            self._need(V21_EPISODES_STATS, optional=True)
            p = os.path.join(self.root, V21_EPISODES_STATS)
            if os.path.isfile(p):
                out = {int(d['episode_index']): d['stats'] for d in _read_jsonl(p)}
        else:
            _, pq = _pa()
            d = os.path.join(self.root, V30_EPISODES_DIR)
            for dp, _, fs in os.walk(d):
                for f in sorted(fs):
                    if f.endswith('.parquet'):
                        for row in pq.read_table(os.path.join(dp, f)).to_pylist():
                            st = {}
                            for k, v in row.items():
                                if k.startswith('stats/'):
                                    _, feat, stat = k.split('/', 2) if k.count('/') >= 2 else (None, k, '')
                                    if v is not None:
                                        st.setdefault(feat, {})[stat] = v
                            out[int(row['episode_index'])] = st
        self._ep_stats = out
        return out

    # --- one episode
    def load(self, index: int) -> Episode:
        _, pq = _pa()
        e = self._row(index)
        i = int(e['episode_index'])
        if self.version == 'v2.1':
            chunk = i // int(self.info.get('chunks_size') or DEFAULT_CHUNK_SIZE)
            rel = (self.info.get('data_path') or V21_DATA_PATH).format(episode_chunk=chunk, episode_index=i)
            loc = {'chunk': chunk}
        else:
            ci, fi = int(e['data/chunk_index']), int(e['data/file_index'])
            rel = (self.info.get('data_path') or V30_DATA_PATH).format(chunk_index=ci, file_index=fi)
            loc = {'chunk': ci, 'file': fi}
        self._need(rel, optional=True)
        f = os.path.join(self.root, rel)
        if not os.path.isfile(f):
            raise DatasetError(f'Episode {i} has no data file: {rel} is missing.')
        try:
            if self.version == 'v2.1':
                table = pq.read_table(f)
            else:
                table = pq.read_table(f, filters=[('episode_index', '=', i)])
        except (OSError, ValueError) as e:     # pyarrow's ArrowInvalid is a ValueError: a truncated or foreign file
            first = str(e).splitlines()[0] if str(e) else type(e).__name__
            first = first.replace(f"'{f}'", rel).replace(f, rel)
            raise DatasetError(f'The data file {rel} of episode {i} is damaged and cannot be read ({first}).') \
                from None
        cols = {c: _to_numpy(table.column(c)) for c in table.column_names}
        n = table.num_rows
        if n == 0:
            raise DatasetError(f'Episode {i} has no rows in {rel}.')
        t = np.asarray(cols.pop('timestamp'), dtype=np.float64).reshape(-1) if 'timestamp' in cols else \
            (np.arange(n) / self.fps if self.fps else np.arange(n, dtype=np.float64))
        q, joint_names = np.zeros((n, 0)), []
        if STATE_KEY in cols and not isinstance(cols[STATE_KEY], list):
            q = np.asarray(cols.pop(STATE_KEY), dtype=np.float64).reshape(n, -1)
            joint_names = _names(self.features.get(STATE_KEY, {}), q.shape[1])
        action, action_names = None, []
        if ACTION_KEY in cols and not isinstance(cols[ACTION_KEY], list):
            action = np.asarray(cols.pop(ACTION_KEY), dtype=np.float64).reshape(n, -1)
            action_names = _names(self.features.get(ACTION_KEY, {}), action.shape[1])
        tasks = list(e.get('tasks') or [])
        task = ''
        ti = cols.get('task_index')
        if ti is not None and len(ti):
            task = self.tasks.get(int(np.asarray(ti)[0]), '')
        if not task and tasks:
            task = str(tasks[0])
        cams = {}
        for key, ft in self.features.items():
            if not isinstance(ft, dict) or ft.get('dtype') not in ('video', 'image'):
                continue
            cams[key] = self._camera(key, ft, e, i, cols.pop(key, None) if ft['dtype'] == 'image' else None, t)
        meta = {'format': 'lerobot', 'version': self.version, 'data_file': rel, 'tasks': tasks, 'columns': cols,
                'units': 'as stored (LeRobot records no units)', **loc}
        return Episode(id=f'episode_{i:06d}', index=i, t=t, q=q, joint_names=joint_names, action=action,
                       action_names=action_names, fps=self.fps, robot_type=str(self.info.get('robot_type') or ''),
                       task=task, cameras=cams, meta=meta)

    def _camera(self, key, ft, e, i, cells, t) -> CameraStream:
        vinfo = ft.get('info') or ft.get('video_info') or {}
        shape = list(ft.get('shape') or [])
        names = [str(x) for x in (ft.get('names') or [])] if isinstance(ft.get('names'), list) else []
        h = w = None
        if len(shape) == 3 and 'height' in names and 'width' in names:
            h, w = int(shape[names.index('height')]), int(shape[names.index('width')])
        h = int(vinfo['video.height']) if vinfo.get('video.height') else h
        w = int(vinfo['video.width']) if vinfo.get('video.width') else w
        fps = float(vinfo['video.fps']) if vinfo.get('video.fps') else self.fps
        meta = {'dtype': ft['dtype'], 'codec': vinfo.get('video.codec'), 'pix_fmt': vinfo.get('video.pix_fmt'),
                'is_depth_map': bool(vinfo.get('is_depth_map', vinfo.get('video.is_depth_map', False)))}
        if ft['dtype'] == 'image':
            cells = cells if isinstance(cells, list) else []
            return CameraStream(key, w, h, fps, np.asarray(t, dtype=np.float64)[:len(cells)], None, None, None,
                                {**meta, 'source': 'parquet images'}, _image_decoder(cells, key))
        if self.version == 'v2.1':
            chunk = i // int(self.info.get('chunks_size') or DEFAULT_CHUNK_SIZE)
            rel = (self.info.get('video_path') or V21_VIDEO_PATH).format(episode_chunk=chunk, video_key=key,
                                                                         episode_index=i)
            lo, hi = 0.0, None
        else:
            ci, fi = e.get(f'videos/{key}/chunk_index'), e.get(f'videos/{key}/file_index')
            if ci is None or fi is None:
                raise DatasetError(f'Episode {i} does not say which video file holds camera {key}.')
            rel = (self.info.get('video_path') or V30_VIDEO_PATH).format(video_key=key, chunk_index=int(ci),
                                                                         file_index=int(fi))
            lo = float(e.get(f'videos/{key}/from_timestamp') or 0.0)
            hi = e.get(f'videos/{key}/to_timestamp')
            hi = float(hi) if hi is not None else None
        f = os.path.join(self.root, rel)
        if self.hub is not None and not os.path.isfile(f):      # read in place by HTTP range requests
            try:
                rf = self.hub.remote(rel)
            except DatasetError:
                raise DatasetError(f'Episode {i} has no video for camera {key}: {rel} is missing.') from None
            if rf is not None:
                f = _Reopen(self.hub, rel, rf.size)
            else:
                self._need(rel)                     # the server ignores range requests: the whole file
        elif not os.path.isfile(f):
            raise DatasetError(f'Episode {i} has no video for camera {key}: {rel} is missing.')
        pts, tb = video_pts(f, lo, hi, fps)
        ts = np.asarray([float(p * tb) for p in pts], dtype=np.float64) - lo
        meta.update({'file': rel, 'from_timestamp': lo, 'to_timestamp': hi, 'time_base': str(tb)})
        return CameraStream(key, w, h, fps, ts, None, None, None, meta, _video_decoder(f, pts, key))

    def truth(self) -> dict | None:
        self._need(TRUTH_FILE, optional=True)
        p = os.path.join(self.root, TRUTH_FILE)
        if not os.path.isfile(p):
            return None
        try:
            d = _read_json(p)
        except (OSError, ValueError):
            return None
        return d if isinstance(d, dict) and d.get('schema') == TRUTH_SCHEMA else None


@register_reader('lerobot', sniff)
def open_lerobot(path: str) -> LeRobotDataset:
    return LeRobotDataset(path)


# ---------------------------------------------------------------------------------------------------------------
# stats (compute_stats.py of both versions)


def _sample_indices(n: int) -> list:
    """compute_stats.sample_indices: estimate_num_samples(min 100, max 10000, power 0.75), evenly spaced."""
    lo = min(100, n)
    k = max(lo, min(int(n ** 0.75), 10_000))
    return np.round(np.linspace(0, n - 1, k)).astype(int).tolist()


def _downsample(img_chw: np.ndarray) -> np.ndarray:
    """compute_stats.auto_downsample_height_width (target 150, threshold 300)."""
    _, h, w = img_chw.shape
    if max(w, h) < 300:
        return img_chw
    f = int(w / 150) if w > h else int(h / 150)
    return img_chw[:, ::f, ::f]


def _feature_stats(a: np.ndarray, axis, keepdims: bool, quantiles: bool, count: int | None = None) -> dict:
    a = np.asarray(a, dtype=np.float64)
    st = {'min': np.min(a, axis=axis, keepdims=keepdims), 'max': np.max(a, axis=axis, keepdims=keepdims),
          'mean': np.mean(a, axis=axis, keepdims=keepdims), 'std': np.std(a, axis=axis, keepdims=keepdims),
          'count': np.array([len(a) if count is None else count])}
    if quantiles:
        for q in QUANTILES:
            key = f'q{int(q * 100):02d}'
            st[key] = (np.quantile(a, q, axis=axis, keepdims=keepdims) if len(a) >= 2 else st['mean'].copy())
    return st


def _image_stats(samples: list, n_frames: int, version: str) -> dict:
    arr = np.stack(samples)                                     # (k, 3, h, w) uint8
    st = _feature_stats(arr, (0, 2, 3), True, version == 'v3.0', None if version == 'v2.1' else n_frames)
    return {k: (v if k == 'count' else np.squeeze(v / 255.0, axis=0)) for k, v in st.items()}


def aggregate_stats(stats_list: list) -> dict:
    """compute_stats.aggregate_stats: min of mins, max of maxes, count-weighted mean, pooled variance; quantiles as
    the envelope (min for q <= 50, max above), as LeRobot v3.0 does."""
    out = {}
    for key in {k for s in stats_list for k in s}:
        fs = [s[key] for s in stats_list if key in s]
        means = np.stack([np.asarray(s['mean'], float) for s in fs])
        var = np.stack([np.asarray(s['std'], float) ** 2 for s in fs])
        counts = np.stack([np.asarray(s['count'], float) for s in fs])
        total = counts.sum(axis=0)
        while counts.ndim < means.ndim:
            counts = np.expand_dims(counts, -1)
        mean = (means * counts).sum(axis=0) / total
        variance = ((var + (means - mean) ** 2) * counts).sum(axis=0) / total
        agg = {'min': np.min(np.stack([np.asarray(s['min'], float) for s in fs]), axis=0),
               'max': np.max(np.stack([np.asarray(s['max'], float) for s in fs]), axis=0),
               'mean': mean, 'std': np.sqrt(variance), 'count': total.astype(np.int64)}
        for qk in [k for k in fs[0] if k.startswith('q') and k[1:].isdigit()]:
            if all(qk in s for s in fs):
                qs = np.stack([np.asarray(s[qk], float) for s in fs])
                agg[qk] = np.min(qs, axis=0) if int(qk[1:]) <= 50 else np.max(qs, axis=0)
        out[key] = agg
    return out


def _np_stats(st: dict) -> dict:
    return {f: {k: np.asarray(v) for k, v in d.items()} for f, d in st.items()}


def _jsonable(st: dict) -> dict:
    return {f: {k: np.asarray(v).tolist() for k, v in d.items()} for f, d in st.items()}


# ---------------------------------------------------------------------------------------------------------------
# the writer


def available_codec(codec: str = 'auto') -> tuple[str, str]:
    """(PyAV encoder name, label) for the requested codec: 'auto' takes AV1 (libsvtav1, LeRobot's default) when the
    encoder exists, else H.264 (libx264); 'av1'/'libsvtav1' and 'h264'/'libx264' ask for one of them."""
    av = _av()

    def has(name):
        try:
            av.codec.Codec(name, 'w')
            return True
        except Exception:                          # noqa: BLE001 — UnknownCodecError and friends
            return False
    c = (codec or 'auto').lower()
    order = {'auto': ['libsvtav1', 'libx264'], 'av1': ['libsvtav1'], 'libsvtav1': ['libsvtav1'],
             'h264': ['libx264'], 'libx264': ['libx264']}.get(c)
    if order is None:
        raise DatasetError(f'The video codec {codec} is not one the LeRobot writer offers (auto, av1, h264).')
    for name in order:
        if has(name):
            return name, ('AV1 (libsvtav1)' if name == 'libsvtav1' else 'H.264 (libx264)')
    raise DatasetError(f'This PyAV build has no {" or ".join(order)} encoder, so the videos cannot be written.')


def _arrow_type(dtype: str, shape: list):
    pa, _ = _pa()
    base = {'float32': pa.float32(), 'float64': pa.float64(), 'int64': pa.int64(), 'int32': pa.int32(),
            'int16': pa.int16(), 'int8': pa.int8(), 'uint8': pa.uint8(), 'bool': pa.bool_(),
            'string': pa.string()}.get(dtype)
    if base is None:
        raise DatasetError(f'The feature dtype {dtype} is not one the LeRobot writer writes.')
    shape = [int(s) for s in shape]
    if shape == [1] or dtype == 'string':
        return base
    t = base
    for s in reversed(shape):
        t = pa.list_(t, s)
    return t


def _arrow_column(values, dtype: str, shape: list):
    pa, _ = _pa()
    typ = _arrow_type(dtype, shape)
    if dtype == 'string':
        return pa.array([str(v) for v in values], type=typ)
    a = np.ascontiguousarray(np.asarray(values).astype(dtype))
    n = a.shape[0]
    if [int(s) for s in shape] == [1]:
        return pa.array(a.reshape(n), type=typ)
    arr = pa.array(a.reshape(-1))
    for s in reversed([int(s) for s in shape]):
        arr = pa.FixedSizeListArray.from_arrays(arr, s)
    return arr.cast(typ)


def _pandas_index_metadata(index_col: str, data_cols: list) -> bytes:
    """The schema metadata pandas writes for a DataFrame with a named index, so pd.read_parquet (LeRobot's
    load_tasks) restores 'task' as the index."""
    cols = [{'name': c, 'field_name': c, 'pandas_type': p, 'numpy_type': n, 'metadata': None}
            for c, p, n in data_cols]
    cols.append({'name': index_col, 'field_name': index_col, 'pandas_type': 'unicode', 'numpy_type': 'object',
                 'metadata': None})
    doc = {'index_columns': [index_col], 'column_indexes': [{'name': None, 'field_name': None,
                                                             'pandas_type': 'unicode', 'numpy_type': 'object',
                                                             'metadata': {'encoding': 'UTF-8'}}],
           'columns': cols, 'creator': {'library': 'pyarrow', 'version': _pa()[0].__version__},
           'pandas_version': '2.2.3'}
    return json.dumps(doc).encode('utf-8')


def _write_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, indent=4, ensure_ascii=False)


def _regular(ts: np.ndarray, fps: float) -> bool:
    k = np.arange(len(ts))
    return bool(len(ts) == 0 or np.max(np.abs(ts - k / fps)) < 1e-6)


class _VideoFile:
    """One episode's MP4 being encoded (PyAV), frames given with their presentation time in seconds."""

    def __init__(self, path, fps, width, height, codec, version, regular):
        av = _av()
        os.environ.setdefault('SVT_LOG', '1')          # SVT-AV1: errors only on stderr
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path, self.fps, self.regular = path, fps, regular
        self.tb = Fraction(1, int(round(fps))) if regular else FINE_TIME_BASE
        opts = {'g': str(ENCODER_DEFAULTS['g']), 'crf': str(ENCODER_DEFAULTS['crf'])}
        if codec == 'libsvtav1' and version == 'v3.0':
            opts['preset'] = str(ENCODER_DEFAULTS['preset_svtav1_v30'])
        if codec == 'libx264' and version == 'v3.0':
            opts['bf'] = '0'                           # no B-frames: pts = dts, so packets shift into a packed file
        self.options = {k: (int(v) if v.isdigit() else v) for k, v in opts.items()}
        self.c = av.open(path, 'w', options={'movflags': 'faststart'})
        self.s = self.c.add_stream(codec, rate=int(round(fps)), options=opts)
        self.s.pix_fmt = ENCODER_DEFAULTS['pix_fmt']
        self.s.width, self.s.height = int(width), int(height)
        self.s.codec_context.time_base = self.tb
        self._last = None
        self.bytes = 0                                  # encoded bytes so far (the muxer buffers file writes)

    def add(self, t_s: float, img: np.ndarray, key_frame: bool = False):
        av = _av()
        pts = int(round(t_s * self.fps)) if self.regular else int(round(t_s / self.tb))
        if self._last is not None and pts <= self._last:
            raise DatasetError('Two frames of a camera fall on the same video timestamp; camera timestamps must '
                               'increase.')
        self._last = pts
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format='rgb24')
        vf.pts, vf.time_base = pts, self.tb
        if key_frame:
            try:
                vf.pict_type = av.video.frame.PictureType.I
            except Exception:                          # noqa: BLE001 — older PyAV: keyframes then follow g
                pass
        self._mux(self.s.encode(vf))

    def _mux(self, packets):
        for p in packets:
            self.bytes += int(p.size)
            self.c.mux(p)

    def size(self) -> int:
        return self.bytes

    def close(self):
        self._mux(self.s.encode())
        self.c.close()


def _encode_episode(path, k, j, cam, ts, fps, vcodec, version, regular, cam_shape):
    """Encode one camera of one episode to path; (the closed _VideoFile, the camera's stats)."""
    m = len(ts)
    sample = set(_sample_indices(m))
    samples, seen, vf = [], 0, None
    try:
        for i, img in cam.frames():
            img = np.asarray(img)
            if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
                raise DatasetError(f'Frame {i} of camera {k} is not an HxWx3 uint8 RGB image.')
            if i != seen:
                raise DatasetError(f'Camera {k} of episode {j} gave frame {i} where frame {seen} was expected.')
            if i >= m:
                raise DatasetError(f'Camera {k} of episode {j} gave more frames than its {m} timestamps.')
            if k not in cam_shape:
                if img.shape[0] % 2 or img.shape[1] % 2:
                    raise DatasetError(f'Camera {k} is {img.shape[1]}x{img.shape[0]} px; the yuv420p videos need an '
                                       'even width and height.')
                cam_shape[k] = img.shape
            elif img.shape != cam_shape[k]:
                raise DatasetError(f'Camera {k} changes size from {cam_shape[k][1]}x{cam_shape[k][0]} to '
                                   f'{img.shape[1]}x{img.shape[0]} px; LeRobot features have one shape.')
            if vf is None:
                vf = _VideoFile(path, fps, img.shape[1], img.shape[0], vcodec, version, regular)
            vf.add(float(ts[i]), img, key_frame=(i == 0))
            if i in sample:
                samples.append(_downsample(np.transpose(img, (2, 0, 1))))
            seen += 1
    finally:
        if vf is not None:
            vf.close()
    if seen != m:
        raise DatasetError(f'Camera {k} of episode {j} gave {seen} frames for {m} timestamps.')
    return vf, _image_stats(samples, m, version)


class _Packer:
    """A v3.0 video file: the episodes' own MP4s remuxed one after another, each shifted by its from_timestamp, which
    is the summed durations (last frame time + 1/fps, in whole ticks of the time base) of the episodes before it."""

    def __init__(self, path, ci, fi):
        self.path, self.ci, self.fi = path, ci, fi
        self.c = self.s = self.tb = None
        self.bytes, self.ticks = 0, 0

    def append(self, episode_file: str, last_t: float, fps: int) -> tuple[float, float]:
        av = _av()
        with av.open(episode_file) as src:
            ist = src.streams.video[0]
            if self.c is None:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                self.c = av.open(self.path, 'w', options={'movflags': 'faststart'})
                self.s = self.c.add_stream_from_template(template=ist, opaque=True)
                self.s.time_base = ist.time_base
                self.tb = ist.time_base
            elif ist.time_base != self.tb:
                raise DatasetError('The episodes of a camera were encoded with different time bases and cannot share '
                                   'a video file.')
            off = self.ticks
            for p in src.demux(ist):
                if p.dts is None or p.pts is None:
                    continue
                p.pts += off
                p.dts += off
                p.stream = self.s
                self.bytes += int(p.size)
                self.c.mux(p)
        self.ticks = off + int(round((last_t + 1.0 / fps) / self.tb))
        return float(off * self.tb), float(self.ticks * self.tb)

    def close(self):
        if self.c is not None:
            self.c.close()
            self.c = None


def _video_info(path: str, version: str, encoder_options: dict, fps: int, regular: bool) -> dict:
    """video_utils.get_video_info for the first file of a camera (plus, at v3.0, the encoder fields). video.fps is
    the probed base rate, as LeRobot reads it, for regular cameras; for irregular ones (time base 1/90000, where the
    probed rate is a guess from the pts) it is the nominal rate the file was encoded at."""
    av = _av()
    with av.open(path) as c:
        s = c.streams.video[0]
        rate = float(s.base_rate) if (regular and s.base_rate) else float(fps)
        codec = getattr(getattr(s, 'codec', None), 'canonical_name', None) or s.codec_context.name
        info = {'video.height': s.height, 'video.width': s.width, 'video.codec': codec, 'video.pix_fmt': s.pix_fmt}
        if version == 'v2.1':
            info['video.is_depth_map'] = False
            info['video.fps'] = int(rate)
        else:
            info['video.fps'] = int(rate) if rate.is_integer() else rate
        info['video.channels'] = 3
        info['has_audio'] = len(c.streams.audio) > 0
    if version == 'v3.0':
        for k in ('g', 'crf', 'preset'):
            info.setdefault(f'video.{k}', encoder_options.get(k))
        info.setdefault('video.fast_decode', 0)
        info['is_depth_map'] = False
    return info


def _check_episodes(episodes: list, fps: float, extra: dict):
    if not episodes:
        raise DatasetError('There are no episodes to write.')
    e0 = episodes[0]
    nq = int(np.asarray(e0.q).reshape(len(e0.t), -1).shape[1]) if len(e0.t) else 0
    na = None if e0.action is None else int(np.asarray(e0.action).reshape(len(e0.t), -1).shape[1])
    cam_keys = sorted(e0.cameras)
    for k in list(extra) + cam_keys:
        if '/' in k:
            raise DatasetError(f'The feature name {k} contains "/", which LeRobot does not allow.')
        if k in DEFAULT_FEATURES or k in (STATE_KEY, ACTION_KEY):
            raise DatasetError(f'The feature name {k} is one the writer fills itself.')
    for j, e in enumerate(episodes):
        n = len(e.t)
        if n == 0:
            raise DatasetError(f'Episode {j} has no rows; LeRobot episodes need at least one frame.')
        q = np.asarray(e.q)
        if q.shape[0] != n or (q.reshape(n, -1).shape[1] if q.size else 0) != nq:
            raise DatasetError(f'Episode {j} has a joint state of a different shape from episode 0 ({q.shape}).')
        if (e.action is None) != (na is None) or (e.action is not None and np.asarray(e.action).reshape(n, -1)
                                                  .shape[1] != na):
            raise DatasetError(f'Episode {j} has actions of a different shape from episode 0; every episode needs '
                               'the same features.')
        if sorted(e.cameras) != cam_keys:
            raise DatasetError(f'Episode {j} has cameras {sorted(e.cameras)}, episode 0 has {cam_keys}; every '
                               'episode needs the same cameras.')
        for k in extra:
            col = (e.meta or {}).get('columns', {}).get(k)
            if col is None or len(col) != n:
                raise DatasetError(f'Episode {j} has no column {k} of {n} rows (put it in episode.meta["columns"]).')
        if not np.all(np.isfinite(np.asarray(e.t, float))) or np.any(np.diff(np.asarray(e.t, float)) < 0):
            raise DatasetError(f'The timestamps of episode {j} are not finite and non-decreasing.')
    return nq, na, cam_keys


def write_lerobot(path: str, episodes: list, version: str = 'v3.0', fps: float | None = None,
                  robot_type: str | None = None, features: dict | None = None, *, codec: str = 'auto',
                  truth: dict | None = None, overwrite: bool = False, chunks_size: int = DEFAULT_CHUNK_SIZE,
                  data_file_size_mb: float = DEFAULT_DATA_FILE_SIZE_IN_MB,
                  video_file_size_mb: float = DEFAULT_VIDEO_FILE_SIZE_IN_MB, progress=None) -> dict:
    """Write episodes as a LeRobot dataset of codebase version 'v2.1' or 'v3.0' at path (a new or empty folder; an
    existing LeRobot dataset only with overwrite=True).

    episodes: [Episode] with t (N,), q (N, nq) -> observation.state (float32, names = joint_names), action (N, na) ->
    action (float32), task -> the task text, cameras {feature key: CameraStream with timestamps (M,) >= 0 s from the
    episode start and frames() giving (i, HxWx3 uint8 RGB) for i = 0..M-1} -> video features. features: extra
    non-video features {key: {'dtype', 'shape', 'names'}} whose values each episode holds in meta['columns'][key].
    fps: the dataset's whole-number frame rate (default: the first episode's). truth: a dexvantage.truth/1 document
    written as dexvantage_truth.json. Returns a summary (root, version, codec, files, frames)."""
    pa, pq = _pa()
    if version not in SUPPORTED:
        raise DatasetError(f'The LeRobot writer writes versions {" and ".join(SUPPORTED)}, not {version}.')
    episodes = list(episodes or [])
    fps = fps if fps is not None else (episodes[0].fps if episodes else None)
    if fps is None or not np.isfinite(float(fps)) or float(fps) <= 0:
        raise DatasetError('The dataset needs a frame rate (fps) above zero.')
    if abs(float(fps) - round(float(fps))) > 1e-9:
        raise DatasetError(f'LeRobot stores the frame rate as a whole number; {fps} fps is not one.')
    fps = int(round(float(fps)))
    extra = dict(features or {})
    for k, ft in extra.items():
        if not isinstance(ft, dict) or 'dtype' not in ft or 'shape' not in ft:
            raise DatasetError(f'The extra feature {k} needs a dtype and a shape.')
        if ft['dtype'] in ('video', 'image'):
            raise DatasetError(f'The extra feature {k} is a {ft["dtype"]}; cameras go in episode.cameras.')
    if truth is not None and (not isinstance(truth, dict) or truth.get('schema') != TRUTH_SCHEMA):
        raise DatasetError(f'The ground truth must be a {TRUTH_SCHEMA} document.')
    nq, na, cam_keys = _check_episodes(episodes, fps, extra)
    root = os.path.abspath(path)
    created = False
    if os.path.exists(root):
        if not os.path.isdir(root):
            raise DatasetError(f'{os.path.basename(root)} exists and is not a folder.')
        if os.listdir(root):
            if not overwrite:
                raise DatasetError(f'The folder {os.path.basename(root)} is not empty; choose a new folder or allow '
                                   'overwriting.')
            if not os.path.isfile(os.path.join(root, INFO_PATH)):
                raise DatasetError(f'The folder {os.path.basename(root)} is not empty and is not a LeRobot dataset, so '
                                   'it is not overwritten.')
            shutil.rmtree(root)
            os.makedirs(root)
            created = True
    else:
        os.makedirs(root)
        created = True
    vcodec, label = available_codec(codec) if cam_keys else (None, None)
    try:
        summary = _write(root, episodes, version, fps, robot_type, extra, nq, na, cam_keys, vcodec,
                         chunks_size, data_file_size_mb, video_file_size_mb, progress)
        if truth is not None:
            _write_json(truth, os.path.join(root, TRUTH_FILE))
    except BaseException:
        if created:
            shutil.rmtree(root, ignore_errors=True)
        raise
    summary.update({'root': root, 'version': version, 'codec': label, 'encoder': vcodec, 'fps': fps})
    return summary


def _write(root, episodes, version, fps, robot_type, extra, nq, na, cam_keys, vcodec, chunks_size, data_mb, video_mb,
           progress):
    pa, pq = _pa()
    e0 = episodes[0]
    feats: dict = {}
    if nq:
        feats[STATE_KEY] = {'dtype': 'float32', 'shape': [nq], 'names': list(e0.joint_names) or None}
    if na is not None:
        feats[ACTION_KEY] = {'dtype': 'float32', 'shape': [na], 'names': list(e0.action_names) or None}
    for k, ft in extra.items():
        feats[k] = {'dtype': ft['dtype'], 'shape': [int(s) for s in ft['shape']], 'names': ft.get('names')}
    cam_shape = {}
    for k in cam_keys:
        feats[k] = {'dtype': 'video', 'shape': None, 'names': ['height', 'width', 'channels'], 'info': None}
    feats.update({k: dict(v) for k, v in DEFAULT_FEATURES.items()})
    numeric = [k for k, ft in feats.items() if ft['dtype'] != 'video']
    # one time base per camera (v3.0 files hold many episodes): 1/fps when every episode's frame k is at k/fps
    cam_regular = {k: all(_regular(np.asarray(e.cameras[k].timestamps, float).reshape(-1), fps) for e in episodes)
                   for k in cam_keys}
    tasks: dict[str, int] = {}
    ep_rows, ep_stats, files = [], [], set()
    total = 0
    # v3.0 file state
    data_ci = data_fi = 0
    writer, writer_bytes = None, 0
    packers: dict = {}                  # v3.0: camera key -> its open packed video file
    options: dict = {}                  # camera key -> (encoder options, regular)
    tmp_dir = os.path.join(root, '.dexvantage-tmp')
    video_info: dict = {}
    data_schema = None
    for j, e in enumerate(episodes):
        n = len(e.t)
        task = str(e.task or '')
        if task not in tasks:
            tasks[task] = len(tasks)
        cols = {'timestamp': np.asarray(e.t, dtype=np.float64), 'frame_index': np.arange(n),
                'episode_index': np.full(n, j), 'index': np.arange(total, total + n), 'task_index':
                np.full(n, tasks[task])}
        if nq:
            cols[STATE_KEY] = np.asarray(e.q, dtype=np.float64).reshape(n, nq)
        if na is not None:
            cols[ACTION_KEY] = np.asarray(e.action, dtype=np.float64).reshape(n, na)
        for k in extra:
            cols[k] = e.meta['columns'][k]
        arrays = [_arrow_column(cols[k], feats[k]['dtype'], feats[k]['shape']) for k in numeric]
        table = pa.Table.from_arrays(arrays, names=numeric)
        if data_schema is None:
            data_schema = table.schema
        st = {}
        for k in numeric:
            if feats[k]['dtype'] == 'string':
                continue
            a = np.asarray(cols[k]).astype(feats[k]['dtype'])
            a = a.reshape(n) if feats[k]['shape'] == [1] else a.reshape((n,) + tuple(feats[k]['shape']))
            st[k] = _feature_stats(a, 0, a.ndim == 1, version == 'v3.0')
        meta = {}
        if version == 'v2.1':
            chunk = j // chunks_size
            rel = V21_DATA_PATH.format(episode_chunk=chunk, episode_index=j)
            os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
            pq.write_table(table, os.path.join(root, rel), compression='snappy')
            files.add(rel)
        else:
            if writer is not None and writer_bytes + table.nbytes >= data_mb * 1024 ** 2:
                writer.close()
                writer, writer_bytes = None, 0
                data_ci, data_fi = (data_ci + 1, 0) if data_fi == chunks_size - 1 else (data_ci, data_fi + 1)
            rel = V30_DATA_PATH.format(chunk_index=data_ci, file_index=data_fi)
            if writer is None:
                os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
                writer = pq.ParquetWriter(os.path.join(root, rel), data_schema, compression='snappy',
                                          use_dictionary=True)
                files.add(rel)
            writer.write_table(table, row_group_size=max(1, n))
            writer_bytes += table.nbytes
            meta.update({'data/chunk_index': data_ci, 'data/file_index': data_fi, 'dataset_from_index': total,
                         'dataset_to_index': total + n})
        # cameras: each episode encoded on its own, as LeRobot does; at v3.0 its packets are then appended to the
        # camera's packed file, shifted by the episode's from_timestamp (no re-encoding)
        for k in cam_keys:
            cam = e.cameras[k]
            ts = np.asarray(cam.timestamps, dtype=np.float64).reshape(-1)
            if len(ts) == 0:
                raise DatasetError(f'Camera {k} of episode {j} has no frame timestamps.')
            if ts[0] < 0 or np.any(np.diff(ts) <= 0) or not np.all(np.isfinite(ts)):
                raise DatasetError(f'The frame timestamps of camera {k} in episode {j} must start at 0 s or later and '
                                   'increase.')
            if version == 'v2.1':
                rel_v = V21_VIDEO_PATH.format(episode_chunk=j // chunks_size, video_key=k, episode_index=j)
                target = os.path.join(root, rel_v)
                regular = _regular(ts, fps)
            else:
                os.makedirs(tmp_dir, exist_ok=True)
                target = os.path.join(tmp_dir, f'episode_{j:06d}_{cam_keys.index(k)}.mp4')
                regular = cam_regular[k]
            vf, st[k] = _encode_episode(target, k, j, cam, ts, fps, vcodec, version, regular, cam_shape)
            options.setdefault(k, (vf.options, regular))
            if version == 'v2.1':
                files.add(rel_v)
                if k not in video_info:
                    video_info[k] = _video_info(target, version, vf.options, fps, regular)
                continue
            pk = packers.get(k)
            ep_bytes = os.path.getsize(target)
            if pk is not None and pk.bytes + ep_bytes >= video_mb * 1024 ** 2:     # dataset_writer._save_episode_video
                pk.close()
                ci, fi = (pk.ci + 1, 0) if pk.fi == chunks_size - 1 else (pk.ci, pk.fi + 1)
                pk = None
            elif pk is None:
                ci, fi = 0, 0
            if pk is None:
                rel_v = V30_VIDEO_PATH.format(video_key=k, chunk_index=ci, file_index=fi)
                pk = packers[k] = _Packer(os.path.join(root, rel_v), ci, fi)
                files.add(rel_v)
            lo, hi = pk.append(target, float(ts[-1]), fps)
            os.remove(target)
            meta.update({f'videos/{k}/chunk_index': pk.ci, f'videos/{k}/file_index': pk.fi,
                         f'videos/{k}/from_timestamp': lo, f'videos/{k}/to_timestamp': hi})
        ep_rows.append({'episode_index': j, 'tasks': [task], 'length': n, **meta})
        ep_stats.append(st)
        total += n
        if progress is not None:
            progress((j + 1) / len(episodes), f'Episode {j + 1} of {len(episodes)} written')
    if writer is not None:
        writer.close()
    for k, pk in packers.items():
        pk.close()
        video_info[k] = _video_info(os.path.join(root, V30_VIDEO_PATH.format(video_key=k, chunk_index=0,
                                                                              file_index=0)), version,
                                    options[k][0], fps, options[k][1])
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    for k in cam_keys:
        h, w, c = cam_shape[k]
        feats[k]['shape'] = [int(h), int(w), int(c)]
        feats[k]['info'] = video_info[k]
    task_list = sorted(tasks, key=tasks.get)
    _write_meta(root, version, fps, robot_type, feats, ep_rows, ep_stats, task_list, total, cam_keys, chunks_size,
                data_mb, video_mb)
    return {'episodes': len(episodes), 'frames': total, 'files': sorted(files), 'features': sorted(feats)}


def _write_meta(root, version, fps, robot_type, feats, ep_rows, ep_stats, task_list, total, cam_keys, chunks_size,
                data_mb, video_mb):
    pa, pq = _pa()
    n_eps = len(ep_rows)
    splits = {'train': f'0:{n_eps}'}
    if version == 'v2.1':
        info = {'codebase_version': 'v2.1', 'robot_type': robot_type, 'total_episodes': n_eps, 'total_frames': total,
                'total_tasks': len(task_list), 'total_videos': n_eps * len(cam_keys),
                'total_chunks': (n_eps - 1) // chunks_size + 1, 'chunks_size': chunks_size, 'fps': fps,
                'splits': splits, 'data_path': V21_DATA_PATH, 'video_path': V21_VIDEO_PATH if cam_keys else None,
                'features': feats}
        _write_json(info, os.path.join(root, INFO_PATH))
        with open(os.path.join(root, V21_EPISODES), 'w', encoding='utf-8') as fh:
            for r in ep_rows:
                fh.write(json.dumps({'episode_index': r['episode_index'], 'tasks': r['tasks'], 'length': r['length']},
                                    ensure_ascii=False) + '\n')
        with open(os.path.join(root, V21_TASKS), 'w', encoding='utf-8') as fh:
            for i, t in enumerate(task_list):
                fh.write(json.dumps({'task_index': i, 'task': t}, ensure_ascii=False) + '\n')
        with open(os.path.join(root, V21_EPISODES_STATS), 'w', encoding='utf-8') as fh:
            for r, st in zip(ep_rows, ep_stats):
                fh.write(json.dumps({'episode_index': r['episode_index'], 'stats': _jsonable(st)}) + '\n')
        return
    info = {'codebase_version': 'v3.0', 'fps': fps, 'features': feats, 'total_episodes': n_eps,
            'total_frames': total, 'total_tasks': len(task_list), 'chunks_size': chunks_size,
            'data_files_size_in_mb': data_mb, 'video_files_size_in_mb': video_mb, 'data_path': V30_DATA_PATH,
            'video_path': V30_VIDEO_PATH if cam_keys else None, 'robot_type': robot_type, 'splits': splits}
    _write_json(info, os.path.join(root, INFO_PATH))
    _write_json(_jsonable(aggregate_stats([_np_stats(s) for s in ep_stats])), os.path.join(root, STATS_PATH))
    # tasks.parquet: task_index with 'task' as the pandas index (io_utils.write_tasks)
    tt = pa.table({'task_index': pa.array(range(len(task_list)), pa.int64()), 'task': pa.array(task_list, pa.string())})
    tt = tt.replace_schema_metadata({b'pandas': _pandas_index_metadata('task', [('task_index', 'int64', 'int64')])})
    pq.write_table(tt, os.path.join(root, V30_TASKS))
    # the episodes table, one file
    rows = []
    for r, st in zip(ep_rows, ep_stats):
        row = dict(r)
        for f, d in _jsonable(st).items():
            for k, v in d.items():
                row[f'stats/{f}/{k}'] = v
        row['meta/episodes/chunk_index'] = 0
        row['meta/episodes/file_index'] = 0
        rows.append(row)
    cols = list(rows[0])
    for r in rows[1:]:
        cols += [c for c in r if c not in cols]
    arrays = {}
    for c in cols:
        vals = [r.get(c) for r in rows]
        if c == 'tasks':
            arrays[c] = pa.array(vals, pa.list_(pa.string()))
        elif c.endswith('_timestamp'):
            arrays[c] = pa.array(vals, pa.float64())
        elif c.startswith('stats/'):
            arrays[c] = pa.array(vals, _stats_type(c, vals))
        else:
            arrays[c] = pa.array(vals, pa.int64())
    p = os.path.join(root, V30_EPISODES_PATH.format(chunk_index=0, file_index=0))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    pq.write_table(pa.table(arrays), p)


def _stats_type(col: str, vals: list):
    pa, _ = _pa()
    v = next((x for x in vals if x is not None), [])
    depth = 0
    while isinstance(v, list):
        depth += 1
        v = v[0] if v else 0
    t = pa.int64() if col.endswith('/count') else pa.float64()
    for _ in range(depth):
        t = pa.list_(t)
    return t
