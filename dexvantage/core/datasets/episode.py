"""The one shape every recorded or synthetic dataset is read into, whatever its format (LeRobot v2.1 and v3.0, RLDS /
Open X-Embodiment, ROS 2 bags in MCAP or SQLite, DexVantage's synthetic episodes).

A reader module registers itself with @register_reader('format', sniff) on a function open(path) -> Dataset; the
dataset lists its episodes cheaply (Dataset.episodes()) and loads one at a time (Dataset.load(i)) so a 500-episode
campaign never sits in memory at once. Camera frames are decoded lazily (CameraStream.frames()).

Times are seconds, float64. Joint states are radians (revolute) or metres (prismatic), in the order of joint_names.
A figure the dataset does not record is None, never a guess: a reader that has no hardware timestamps leaves
hw_timestamps None; no intrinsics, intrinsics None.

Known limits: one robot state vector per episode (a bimanual dataset's two arms are one vector with both arms' joint
names); per-camera intrinsics and extrinsics only when the format stores them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np

from ..jobs import StudyError


class DatasetError(StudyError):
    """A dataset that cannot be read, with a plain sentence saying why (unknown format or version, a missing file)."""


@dataclass
class CameraStream:
    key: str                                   # the dataset's camera key, e.g. 'observation.images.wrist'
    width: int | None = None
    height: int | None = None
    fps: float | None = None                   # nominal, as the dataset states it
    timestamps: np.ndarray = field(default_factory=lambda: np.zeros(0))   # (M,) s from the episode start, as stored
    hw_timestamps: np.ndarray | None = None    # (M,) device timestamps (s) when the format records them
    intrinsics: dict | None = None             # {'model': 'pinhole'|..., 'K': 3x3 list, 'dist': [...]} when stored
    extrinsics: dict | None = None             # {'frame': parent, 'pose': {'position', 'quaternion' (x,y,z,w)}}
    meta: dict = field(default_factory=dict)   # codec, pixel format, file name, exposure / gain columns if any
    _decode: Callable | None = None            # (indices or None) -> iterator of (i, HxWx3 uint8); set by the reader

    @property
    def n_frames(self) -> int:
        return int(len(self.timestamps))

    def frames(self, indices=None) -> Iterator[tuple[int, np.ndarray]]:
        """(frame index, HxWx3 uint8 RGB) for the given frame indices (all when None), decoded lazily."""
        if self._decode is None:
            raise DatasetError(f'The frames of camera {self.key} cannot be decoded here (no video decoder: install '
                               'the lerobot extra, pip install "dexvantage[lerobot]").')
        return self._decode(None if indices is None else list(indices))


@dataclass
class Episode:
    id: str                                    # unique in its dataset
    index: int
    t: np.ndarray                              # (N,) s from the episode start: the state rows' times
    q: np.ndarray                              # (N, nq) joint state
    joint_names: list = field(default_factory=list)
    action: np.ndarray | None = None           # (N, na) when recorded
    action_names: list = field(default_factory=list)
    fps: float | None = None                   # nominal
    robot_type: str = ''
    task: str = ''                             # the language instruction, when recorded
    cameras: dict = field(default_factory=dict)   # key -> CameraStream
    meta: dict = field(default_factory=dict)   # format-specific extras (episode file, chunk, data columns)

    @property
    def duration_s(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0


@dataclass
class EpisodeRef:
    index: int
    id: str
    length: int | None = None                  # rows, when the index states it
    meta: dict = field(default_factory=dict)


class Dataset:
    """Subclass per format: set format, version, path and info; implement episodes() and load()."""
    format: str = ''                           # 'lerobot' | 'rlds' | 'mcap' | 'rosbag2-sqlite' | 'synthetic'
    version: str = ''                          # e.g. 'v2.1', 'v3.0'
    path: str = ''
    info: dict = {}                            # the dataset's own header (LeRobot info.json, bag metadata, ...)

    def episodes(self) -> list[EpisodeRef]:
        raise NotImplementedError

    def load(self, index: int) -> Episode:
        raise NotImplementedError

    def truth(self) -> dict | None:
        """Ground truth stored beside a synthetic dataset (dexvantage_truth.json), else None."""
        return None

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


READERS: list = []                             # [(format, sniff(path) -> bool, open(path) -> Dataset)]


def register_reader(fmt: str, sniff: Callable[[str], bool]):
    """Decorator for a reader's open(path) -> Dataset; sniff(path) says whether the path is in this format."""
    def deco(fn):
        READERS[:] = [r for r in READERS if r[0] != fmt] + [(fmt, sniff, fn)]
        return fn
    return deco


def open_dataset(path: str) -> Dataset:
    """Open a dataset in any registered format, or raise DatasetError with a plain sentence."""
    import os
    from . import load_readers
    load_readers()
    if not os.path.exists(path):
        raise DatasetError(f'There is no dataset at {os.path.basename(path) or path}.')
    for fmt, sniff, opener in READERS:
        try:
            ok = sniff(path)
        except Exception:                      # noqa: BLE001 — a sniffer must not stop the others
            ok = False
        if ok:
            return opener(path)
    names = ', '.join(r[0] for r in READERS) or 'none'
    raise DatasetError(f'{os.path.basename(path) or path} is not a dataset format DexVantage reads (it reads: '
                       f'{names}).')
