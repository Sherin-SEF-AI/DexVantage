"""Recorded and synthetic datasets: readers (one module per format, each registering with
episode.register_reader), the LeRobot writer and the shared Episode shape (episode.py)."""
from __future__ import annotations

import importlib
import os
import pkgutil

from .episode import (READERS, CameraStream, Dataset, DatasetError, Episode, EpisodeRef, open_dataset,  # noqa: F401
                      register_reader)

_LOADED = False


def load_readers():
    """Import every reader module in this package once (each registers itself)."""
    global _LOADED
    if _LOADED:
        return
    here = os.path.dirname(os.path.abspath(__file__))
    for m in pkgutil.iter_modules([here]):
        if m.name not in ('episode',) and not m.name.startswith('_'):
            importlib.import_module(f'{__package__}.{m.name}')
    _LOADED = True
