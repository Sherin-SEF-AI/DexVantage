"""Studies and the result cache.

A study is a function study(project, ctx) -> dict that reads a project snapshot and returns a JSON-able result. It
reports progress through ctx.progress(fraction, text), hands over partial results through ctx.partial(doc) and stops
early (returning what it has, marked cancelled) when ctx.cancelled() turns true. Studies register with @study.

Results are cached by content digest: the project's content hash (core.provenance) plus the study's id and its
settings, so re-running an unchanged study is instant and any change to the project or the settings misses the cache.

Phase 0 registers one study, the project summary, which also states the dense-voxel limit of §7.1 of the
specification: how many voxels a dense grid over the task volumes would need at the chosen size and at 1 mm.
"""
from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

STUDIES: dict = {}
CACHE_SIZE = 32
DENSE_LIMIT = 50_000_000         # voxels: above this a dense grid is not built (memory and time); adaptive refinement


@dataclass
class StudyInfo:
    id: str
    title: str
    fn: object
    quick: bool = True            # quick enough to re-run after each edit (live)


def study(sid: str, title: str, quick: bool = True):
    def deco(fn):
        STUDIES[sid] = StudyInfo(sid, title, fn, quick)
        return fn
    return deco


@dataclass
class Ctx:
    progress: object = None       # fn(fraction, text)
    partial: object = None        # fn(doc)
    cancel: object = None         # fn() -> bool
    t0: float = field(default_factory=time.perf_counter)

    def cancelled(self) -> bool:
        return bool(self.cancel and self.cancel())

    def report(self, frac: float, text: str = ''):
        if self.progress:
            self.progress(max(0.0, min(1.0, float(frac))), text)


class ResultCache:
    """A small LRU of study results keyed by (study, project content hash, settings hash); thread-safe."""

    def __init__(self, size=CACHE_SIZE):
        self.size = size
        self._d: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def key(self, sid: str, project) -> tuple:
        from . import provenance as PV
        settings = project.analysis.studies.get(sid, {})
        return sid, PV.project_digest(project)['sha256'], PV.sha256(PV.canonical(settings))

    def get(self, key):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                self.hits += 1
                return self._d[key]
            self.misses += 1
            return None

    def put(self, key, value):
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.size:
                self._d.popitem(last=False)


def run_study(sid: str, project, ctx: Ctx | None = None, cache: ResultCache | None = None) -> dict:
    """Run study sid on project (a snapshot), from the cache when it has the same inputs. The result carries
    'schema', 'study', 'seconds', 'cancelled' and 'cached'."""
    info = STUDIES[sid]
    ctx = ctx or Ctx()
    key = cache.key(sid, project) if cache is not None else None
    if key is not None:
        hit = cache.get(key)
        if hit is not None:
            return dict(hit, cached=True)
    t0 = time.perf_counter()
    doc = info.fn(project, ctx)
    doc = {'schema': f'dexvantage.{sid}/1', 'study': sid, **doc}
    doc['seconds'] = round(time.perf_counter() - t0, 4)
    doc['cancelled'] = ctx.cancelled()
    doc['cached'] = False
    if key is not None and not doc['cancelled']:
        cache.put(key, doc)
    return doc


def voxel_count(size_m, voxel_m: float) -> int:
    """Voxels of a dense axis-aligned grid over a box: the product of ceil(edge / voxel) per axis."""
    return int(np.prod([max(1, math.ceil(float(s) / voxel_m - 1e-9)) for s in size_m]))


@study('summary', 'Project summary')
def summary(project, ctx: Ctx) -> dict:
    """What the project holds, and the dense voxel counts of its task volumes."""
    a = project.analysis
    vols = []
    for k, v in enumerate(project.workspace.volumes):
        if ctx.cancelled():
            break
        n = voxel_count(v.size, a.voxel_mm / 1000.0)
        n1 = voxel_count(v.size, 0.001)
        vols.append({'uid': v.uid, 'name': v.name, 'size_m': list(v.size), 'volume_m3': float(np.prod(v.size)),
                     'voxels_dense': n, 'voxels_dense_1mm': n1, 'dense_feasible': n <= DENSE_LIMIT,
                     'dense_1mm_feasible': n1 <= DENSE_LIMIT})
        ctx.report((k + 1) / max(1, len(project.workspace.volumes)), v.name)
    sensors = project.variant.sensors
    return {
        'project': project.name,
        'robots': len(project.robots),
        'joints': sum(len(r.joints) for r in project.robots),
        'sensors': {'total': len(sensors), **{k: sum(s.kind == k for s in sensors) for k in ('camera', 'depth',
                                                                                              'tactile')}},
        'sensors_on_links': sum(s.mount.on_link for s in sensors),
        'variants': len(project.variants),
        'volumes': vols,
        'objects': len(project.workspace.objects),
        'requirements': len(project.requirements),
        'voxel_mm': a.voxel_mm,
        'dense_limit_voxels': DENSE_LIMIT,
    }
