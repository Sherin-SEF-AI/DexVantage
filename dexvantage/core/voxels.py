"""Sparse voxel grids over the task volumes, with adaptive refinement (spec §7.1).

    SparseGrid.from_project(project, voxel_mm=None, refine_mm=None, margin_m=REFINE_MARGIN_M)
        .centers (N, 3) float32 world (m), .size (N,) float32 edge (m), .volume_index (N,) int32 into .volumes,
        .refined (N,) bool, .count, .chunks(n) -> Chunk(start, stop, centers, size, volume_index), .volumes [Volume]
    estimate(project, voxel_mm=None, refine_mm=None) -> {'count', 'dense_count', 'feasible', 'message', ...}
    dense_message(size_m, voxel_m) -> "A dense 1 mm grid over 1.5 × 1.5 × 1.2 m is 2.7 × 10⁹ voxels…"

The grid: each task volume (project.workspace.volumes, an axis-aligned box) is covered by cubic base voxels of edge
voxel_mm (default project.analysis.voxel_mm, 5 mm by default), ceil(edge / voxel) per axis as core.jobs.voxel_count
counts them, the grid centred on the volume (so it overhangs each face by less than half a voxel). A base voxel is
refined, split into k³ voxels with k = ceil(voxel / refine), so the refined edge voxel / k is at most refine_mm
(default project.analysis.refine_mm, 2 mm: 5 mm / 3 = 1.67 mm), when
- its volume is a refinement region (Volume.refine), or
- it overlaps a task object's bounding box grown by margin_m (default REFINE_MARGIN_M = 20 mm, a default): the
  object and its contact zone.
Voxels are listed per volume, so where two volumes overlap their voxels appear once for each (each volume's
statistics stay whole).

Memory: estimate() counts the voxels before anything is built (a boolean mask over the base cells of each volume,
one byte per base cell, only when the base grid itself is under the limit) and from_project refuses a grid over
core.jobs.DENSE_LIMIT (5 × 10⁷) voxels with core.jobs.StudyError carrying the plain-English message; chunks(n)
hands the voxels to a study n at a time so per-voxel work never holds more than n voxels of temporaries.

Known limits: refinement is one level (base → k³), not a full octree; the refined zone is the objects' bounding
boxes (not their shapes) grown by the margin; volumes are axis-aligned boxes; overlapping volumes count shared space
twice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .jobs import DENSE_LIMIT, StudyError

REFINE_MARGIN_M = 0.02          # default margin around task objects inside which voxels are refined
_SUP = str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻')


def sci(n: float) -> str:
    """2700000000 -> '2.7 × 10⁹'; numbers under a million are written out ('480,000')."""
    n = float(n)
    if abs(n) < 1e6:
        return f'{int(round(n)):,}'
    e = int(math.floor(math.log10(abs(n))))
    m = n / 10 ** e
    if round(m, 1) >= 10:
        m, e = m / 10, e + 1
    ms = f'{m:.1f}'.rstrip('0').rstrip('.')
    return f'{ms} × 10{str(e).translate(_SUP)}'


def _mm(v_m: float) -> str:
    mm = v_m * 1000
    return f'{mm:.0f} mm' if abs(mm - round(mm)) < 1e-6 else f'{mm:.2g} mm'


def _m(x: float) -> str:
    return f'{x:.2f}'.rstrip('0').rstrip('.')


def dims(size_m, voxel_m: float) -> np.ndarray:
    """Base voxels per axis over a box: ceil(edge / voxel), at least 1 (as core.jobs.voxel_count)."""
    return np.array([max(1, math.ceil(float(s) / voxel_m - 1e-9)) for s in size_m], np.int64)


def dense_message(size_m, voxel_m: float, limit: int = DENSE_LIMIT) -> str:
    """The spec's sentence for a dense grid over one box: 'A dense 1 mm grid over 1.5 × 1.5 × 1.2 m is 2.7 × 10⁹
    voxels, which is over the limit of 5 × 10⁷: …' (or 'which fits' when it is under)."""
    n = int(np.prod(dims(size_m, voxel_m)))
    over = (f', which is over the limit of {sci(limit)} and is not built: use a coarser voxel size and keep the fine '
            'size to refinement regions around the objects or the contact zone.') if n > limit else \
        f', within the limit of {sci(limit)}.'
    return (f'A dense {_mm(voxel_m)} grid over {" × ".join(_m(s) for s in size_m)} m is {sci(n)} voxels' + over)


@dataclass
class Chunk:
    start: int
    stop: int
    centers: np.ndarray
    size: np.ndarray
    volume_index: np.ndarray


def _settings(project, voxel_mm, refine_mm):
    a = project.analysis
    vm = float(voxel_mm if voxel_mm is not None else a.voxel_mm)
    rm = float(refine_mm if refine_mm is not None else a.refine_mm)
    if not (vm > 0 and math.isfinite(vm)):
        raise StudyError(f'The voxel size must be a positive number of millimetres; it is {vm}.')
    if not (rm > 0 and math.isfinite(rm)):
        raise StudyError(f'The refinement voxel size must be a positive number of millimetres; it is {rm}.')
    h = vm / 1000.0
    k = max(1, math.ceil(vm / rm - 1e-9))
    return h, k


def _object_boxes(project, margin_m: float) -> list:
    from .scenegeom import object_triangles
    out = []
    for o in project.workspace.objects:
        t = object_triangles(o, project)
        if len(t):
            v = t.reshape(-1, 3)
            out.append((v.min(axis=0) - margin_m, v.max(axis=0) + margin_m))
    return out


def _volume_plan(v, h: float, k: int, boxes: list, limit: int):
    """(lo corner of the base grid, dims, refine mask (base cells, bool, flat x-fastest) or None = all or nothing,
    whole-volume refine flag, base count)."""
    size = np.asarray(v.size, float)
    d = dims(size, h)
    c = np.asarray(v.center, float)
    lo = c - d * h / 2
    nb = int(np.prod(d))
    if v.refine or k == 1:
        return lo, d, None, bool(v.refine and k > 1), nb
    if nb > limit:
        return lo, d, None, False, nb                 # too big to mask; counted as base only (and refused)
    mask = np.zeros(tuple(d[::-1]), bool)             # (z, y, x)
    for blo, bhi in boxes:
        i0 = np.floor((blo - lo) / h).astype(np.int64)
        i1 = np.ceil((bhi - lo) / h).astype(np.int64) - 1
        i0, i1 = np.maximum(i0, 0), np.minimum(i1, d - 1)
        if (i1 < i0).any():
            continue
        mask[i0[2]:i1[2] + 1, i0[1]:i1[1] + 1, i0[0]:i1[0] + 1] = True
    return lo, d, mask.reshape(-1), False, nb


def estimate(project, voxel_mm: float | None = None, refine_mm: float | None = None,
             margin_m: float = REFINE_MARGIN_M) -> dict:
    """How many voxels the adaptive grid has, and how many a dense grid at the refined size would need; with
    feasible = count <= DENSE_LIMIT and a plain-English message."""
    h, k = _settings(project, voxel_mm, refine_mm)
    hr = h / k
    boxes = _object_boxes(project, margin_m)
    per, count, dense = [], 0, 0
    for v in project.workspace.volumes:
        lo, d, mask, whole, nb = _volume_plan(v, h, k, boxes, DENSE_LIMIT)
        nref = nb if whole else (int(mask.sum()) if mask is not None else 0)
        n = (nb - nref) + nref * k ** 3
        nd = int(np.prod(dims(v.size, hr)))
        per.append({'uid': v.uid, 'name': v.name, 'size_m': [float(s) for s in v.size], 'base': nb,
                    'refined_cells': nref, 'count': n, 'dense_count': nd})
        count += n
        dense += nd
    feasible = count <= DENSE_LIMIT
    vols = project.workspace.volumes
    if not vols:
        msg = 'There is no task volume to voxelise: add one (Workspace ▸ Add task volume).'
    elif not feasible:
        if len(vols) == 1 and not any(p['refined_cells'] for p in per):
            msg = dense_message(vols[0].size, h)
        else:
            what = f'over {" × ".join(_m(s) for s in vols[0].size)} m' if len(vols) == 1 else \
                f'over the {len(vols)} task volumes'
            msg = (f'The grid {what} at {_mm(h)}, refined to {_mm(hr)}, is {sci(count)} voxels, which is over the '
                   f'limit of {sci(DENSE_LIMIT)} and is not built: use a coarser voxel size or smaller refinement '
                   'regions.')
    else:
        ref = sum(p['refined_cells'] for p in per)
        msg = (f'{sci(count)} voxels: {_mm(h)}' + (f', refined to {_mm(hr)} in {sci(ref)} of the {_mm(h)} cells '
                                                    '(refinement regions and around the objects)' if ref else '')
               + f'. A dense {_mm(hr)} grid over the same volumes would be {sci(dense)} voxels'
               + ('.' if dense <= DENSE_LIMIT else f', over the limit of {sci(DENSE_LIMIT)}.'))
    return {'count': int(count), 'dense_count': int(dense), 'feasible': bool(feasible), 'message': msg,
            'voxel_mm': h * 1000, 'refine_mm': hr * 1000, 'refine_k': k, 'limit': DENSE_LIMIT,
            'margin_m': margin_m, 'volumes': per}


class SparseGrid:
    """Voxel centres, edges and volumes (see the module docstring)."""

    def __init__(self, centers, size, volume_index, refined, volumes, voxel_m: float, refine_m: float, info: dict):
        self.centers = centers
        self.size = size
        self.volume_index = volume_index
        self.refined = refined
        self.volumes = volumes
        self.voxel_m = voxel_m
        self.refine_m = refine_m
        self.info = info

    @property
    def count(self) -> int:
        return int(len(self.centers))

    def __len__(self):
        return self.count

    def chunks(self, n: int = 200_000):
        """Chunk(start, stop, centers, size, volume_index) of at most n voxels each (views, not copies)."""
        n = max(1, int(n))
        for a in range(0, self.count, n):
            b = min(self.count, a + n)
            yield Chunk(a, b, self.centers[a:b], self.size[a:b], self.volume_index[a:b])

    def volume_mask(self, i: int) -> np.ndarray:
        return self.volume_index == i

    @property
    def volume_m3(self) -> np.ndarray:
        """(N,) each voxel's volume (m³)."""
        return self.size.astype(np.float64) ** 3

    @staticmethod
    def from_project(project, voxel_mm: float | None = None, refine_mm: float | None = None,
                     margin_m: float = REFINE_MARGIN_M) -> 'SparseGrid':
        est = estimate(project, voxel_mm, refine_mm, margin_m)
        if not est['feasible']:
            raise StudyError(est['message'])
        h, k = _settings(project, voxel_mm, refine_mm)
        hr = h / k
        boxes = _object_boxes(project, margin_m)
        sub = (np.stack(np.meshgrid(*[np.arange(k)] * 3, indexing='ij'), -1).reshape(-1, 3)[:, ::-1] + 0.5) * hr \
            - h / 2                                    # sub-voxel centre offsets in a base cell, x fastest
        C, S, V, R = [], [], [], []
        for i, v in enumerate(project.workspace.volumes):
            lo, d, mask, whole, nb = _volume_plan(v, h, k, boxes, DENSE_LIMIT)
            zz, yy, xx = np.meshgrid(np.arange(d[2]), np.arange(d[1]), np.arange(d[0]), indexing='ij')
            base = lo + (np.stack([xx.ravel(), yy.ravel(), zz.ravel()], 1) + 0.5) * h
            ref = np.ones(nb, bool) if whole else (mask if mask is not None else np.zeros(nb, bool))
            keep = base[~ref]
            fine = (base[ref][:, None, :] + sub[None]).reshape(-1, 3)
            C += [keep.astype(np.float32), fine.astype(np.float32)]
            S += [np.full(len(keep), h, np.float32), np.full(len(fine), hr, np.float32)]
            V += [np.full(len(keep) + len(fine), i, np.int32)]
            R += [np.zeros(len(keep), bool), np.ones(len(fine), bool)]
        if not C:
            z = np.zeros((0, 3), np.float32)
            return SparseGrid(z, np.zeros(0, np.float32), np.zeros(0, np.int32), np.zeros(0, bool), [], h, hr, est)
        return SparseGrid(np.concatenate(C), np.concatenate(S), np.concatenate(V), np.concatenate(R),
                          list(project.workspace.volumes), h, hr, est)
