"""Exact ray casting against triangles: the independent reference for the cube-map occlusion (core/occlusion.py).

    first_hit(origins, dirs, tris, tmax=None) -> (N,) distance along each ray to the nearest triangle (inf: none)
    visible_exact(points, origin, tris, eps=1e-3) -> (N,) bool: nothing lies between origin and the point
    Grid.build(tris)                           the uniform grid both use; pass it instead of tris to reuse it

The intersection test is Möller and Trumbore's [Möller 1997]: for a ray o + t·d and a triangle (v0, v1, v2), with
e1 = v1 − v0, e2 = v2 − v0, p = d × e2, det = e1 · p, s = o − v0, q = s × e1, the barycentric coordinates are
u = (s · p) / det, v = (d · q) / det and the distance t = (e2 · q) / det; a hit needs u ≥ 0, v ≥ 0, u + v ≤ 1 and
t > 0. Triangles are two-sided (no back-face culling) and a triangle seen exactly edge-on (|det| < 1e-12 times the
squared edge lengths) is not hit. Distances are metres when dirs are unit vectors; dirs are normalised here, so the
result is always a distance in metres.

Acceleration: a uniform grid over the triangles' bounding box [Fujimoto 1986], each triangle listed in every cell its
axis-aligned bounding box overlaps (conservative: a cell may list a triangle that does not touch it, never the other
way round), traversed cell by cell with the 3D digital differential analyser of Amanatides and Woo [1987], all rays at
once (numpy, one step of every active ray per iteration). A ray stops at the first cell whose own interval contains
the nearest hit found so far, so the answer is the exact nearest hit, not an approximation: the grid only skips
triangles that cannot be hit first. The cell size is chosen so the grid has about GRID_DENSITY cells per triangle
(capped at MAX_CELLS cells and MAX_DIM cells per axis); 200 000 triangles and 100 000 rays take seconds.

This module shares no code with the rasterisers in core/occlusion.py: they find the nearest triangle per cube-map
texel by rasterisation and a z-buffer, this module by ray–triangle intersection per ray. The validation check
occlusion.cube_vs_raycast compares the two.

Known limits: a ray that grazes a shared edge of two triangles can miss both by rounding (a hole of about 1e-12 in
barycentric terms); triangles of zero area are never hit; the grid is rebuilt for every new set of triangles (build
it once with Grid.build and pass it in to reuse it). Distances are float64; the triangles may be float32.
"""
from __future__ import annotations

import numpy as np

GRID_DENSITY = 2.0            # cells per triangle (Wald et al. and pbrt use 2–4 for uniform grids)
MAX_CELLS = 4_000_000
MAX_DIM = 256
MAX_REFS = 40_000_000         # triangle-cell references; above this the grid is coarsened
PAIR_CHUNK = 4_000_000        # ray-triangle pairs tested at once (memory: about 0.5 GB of float64 temporaries)


def _as_tris(tris) -> np.ndarray:
    t = np.asarray(tris, np.float64)
    if t.ndim != 3 or t.shape[1:] != (3, 3):
        raise ValueError(f'triangles must be an (M, 3, 3) array; got shape {t.shape}')
    return t


def moller_trumbore(o: np.ndarray, d: np.ndarray, v0: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """Distances t (K,) of K ray-triangle pairs (inf where the ray misses or t <= 0): o, d, v0, e1, e2 all (K, 3)."""
    p = np.cross(d, e2)
    det = np.einsum('ij,ij->i', e1, p)
    scale = np.einsum('ij,ij->i', e1, e1) * np.einsum('ij,ij->i', e2, e2)
    ok = np.abs(det) > 1e-12 * np.sqrt(scale)
    with np.errstate(divide='ignore', invalid='ignore'):
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = o - v0
        u = np.einsum('ij,ij->i', s, p) * inv
        q = np.cross(s, e1)
        v = np.einsum('ij,ij->i', d, q) * inv
        t = np.einsum('ij,ij->i', e2, q) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    return np.where(hit, t, np.inf)


class Grid:
    """A uniform grid of triangle references (see the module docstring)."""

    def __init__(self, tris, lo, h, dims, cell_start, refs):
        self.tris = tris                      # (M, 3, 3) float64
        self.v0 = tris[:, 0]
        self.e1 = tris[:, 1] - tris[:, 0]
        self.e2 = tris[:, 2] - tris[:, 0]
        self.lo = lo                          # (3,) the grid's lower corner
        self.h = h                            # (3,) cell edge lengths
        self.dims = dims                      # (3,) int cells per axis
        self.hi = lo + h * dims
        self.cell_start = cell_start          # (ncells + 1,) int64 CSR offsets into refs
        self.refs = refs                      # triangle indices, grouped by cell

    @property
    def n_triangles(self) -> int:
        return len(self.tris)

    @staticmethod
    def build(tris, density: float = GRID_DENSITY) -> 'Grid':
        tris = _as_tris(tris)
        M = len(tris)
        if M == 0:
            z = np.zeros(3)
            return Grid(tris, z, np.ones(3), np.ones(3, np.int64), np.zeros(2, np.int64), np.zeros(0, np.int64))
        tlo, thi = tris.min(axis=1), tris.max(axis=1)
        lo, hi = tlo.min(axis=0), thi.max(axis=0)
        ext = hi - lo
        pad = 1e-6 * max(1.0, float(ext.max()))
        lo, hi = lo - pad, hi + pad
        ext = hi - lo
        target = min(MAX_CELLS, max(1.0, density * M))
        while True:
            cell = (float(np.prod(ext)) / target) ** (1 / 3)
            dims = np.clip(np.ceil(ext / max(cell, 1e-12)), 1, MAX_DIM).astype(np.int64)
            h = ext / dims
            i0 = np.clip(np.floor((tlo - lo) / h).astype(np.int64), 0, dims - 1)
            i1 = np.clip(np.floor((thi - lo) / h).astype(np.int64), 0, dims - 1)
            span = i1 - i0 + 1
            cnt = np.prod(span, axis=1)
            total = int(cnt.sum())
            if total <= MAX_REFS or target <= 1:
                break
            target /= 4                    # big triangles over a fine grid: coarsen
        # every (triangle, cell) of the bounding-box overlaps, without a Python loop
        tri_of = np.repeat(np.arange(M, dtype=np.int64), cnt)
        start = np.cumsum(cnt) - cnt
        local = np.arange(total, dtype=np.int64) - np.repeat(start, cnt)
        sx, sy = span[tri_of, 0], span[tri_of, 1]
        cx = i0[tri_of, 0] + local % sx
        cy = i0[tri_of, 1] + (local // sx) % sy
        cz = i0[tri_of, 2] + local // (sx * sy)
        cid = (cz * dims[1] + cy) * dims[0] + cx
        order = np.argsort(cid, kind='stable')
        refs = tri_of[order]
        ncell = int(np.prod(dims))
        counts = np.bincount(cid, minlength=ncell)
        cell_start = np.zeros(ncell + 1, np.int64)
        np.cumsum(counts, out=cell_start[1:])
        return Grid(tris, lo, h, dims, cell_start, refs)

    # ---------------------------------------------------------------- traversal
    def first_hit(self, origins, dirs, tmax=None) -> np.ndarray:
        d = np.asarray(dirs, np.float64).reshape(-1, 3)
        N = len(d)
        o = np.broadcast_to(np.asarray(origins, np.float64).reshape(-1, 3), (N, 3)) if N else np.zeros((0, 3))
        nrm = np.linalg.norm(d, axis=1)
        good = nrm > 1e-300
        d = np.where(good[:, None], d / np.where(good, nrm, 1.0)[:, None], np.array([1.0, 0.0, 0.0]))
        lim = np.full(N, np.inf) if tmax is None else np.broadcast_to(np.asarray(tmax, np.float64), (N,)).copy()
        best = np.full(N, np.inf)
        if N == 0 or self.n_triangles == 0:
            return best
        # clip each ray to the grid's box (slab method)
        with np.errstate(divide='ignore', invalid='ignore'):
            inv = 1.0 / d
            ta = (self.lo - o) * inv
            tb = (self.hi - o) * inv
        tn = np.where(np.isnan(ta), -np.inf, np.minimum(ta, tb))
        tf = np.where(np.isnan(ta), np.inf, np.maximum(ta, tb))
        # an axis-parallel ray outside the slab never enters it
        par = d == 0
        outside = par & ((o < self.lo) | (o > self.hi))
        tn = np.where(par, -np.inf, tn)
        tf = np.where(par, np.inf, tf)
        t_in = np.maximum(tn.max(axis=1), 0.0)
        t_out = np.minimum(tf.min(axis=1), lim)
        act = np.nonzero(good & ~outside.any(axis=1) & (t_in <= t_out))[0]
        if not len(act):
            return best
        p = o[act] + d[act] * t_in[act, None]
        cell = np.clip(np.floor((p - self.lo) / self.h).astype(np.int64), 0, self.dims - 1)
        da = d[act]
        step = np.where(da >= 0, 1, -1).astype(np.int64)
        with np.errstate(divide='ignore'):
            tdelta = np.where(da != 0, self.h / np.abs(da), np.inf)
            nxt = self.lo + (cell + (step > 0)) * self.h
            tnext = np.where(da != 0, (nxt - o[act]) / np.where(da != 0, da, 1.0), np.inf)
        tend = t_out[act]
        tri_v0, tri_e1, tri_e2 = self.v0, self.e1, self.e2
        dx, dy = int(self.dims[0]), int(self.dims[1])
        while len(act):
            cid = (cell[:, 2] * dy + cell[:, 1]) * dx + cell[:, 0]
            s0 = self.cell_start[cid]
            cnt = self.cell_start[cid + 1] - s0
            has = np.nonzero(cnt)[0]
            if len(has):
                c = cnt[has]
                cum = np.cumsum(c)
                a = 0
                while a < len(has):              # chunks of about PAIR_CHUNK pairs, to bound memory
                    base = int(cum[a - 1]) if a else 0
                    b = max(a + 1, int(np.searchsorted(cum, base + PAIR_CHUNK, 'right')))
                    hs, cc = has[a:b], c[a:b]
                    k = int(cc.sum())
                    ray = np.repeat(hs, cc)
                    st = np.repeat(s0[hs] - (np.cumsum(cc) - cc), cc)
                    tri = self.refs[st + np.arange(k, dtype=np.int64)]
                    gr = act[ray]
                    t = moller_trumbore(o[gr], d[gr], tri_v0[tri], tri_e1[tri], tri_e2[tri])
                    t = np.where(t <= lim[gr], t, np.inf)
                    np.minimum.at(best, gr, t)
                    a = b
            # the cell's own interval ends at the nearest next boundary
            texit = tnext.min(axis=1)
            done = (best[act] <= texit) | (texit > tend)
            ax = np.argmin(tnext, axis=1)
            r = np.arange(len(act))
            cell[r, ax] += step[r, ax]
            tnext[r, ax] += tdelta[r, ax]
            done |= (cell[r, ax] < 0) | (cell[r, ax] >= self.dims[ax])
            keep = ~done
            act, cell, step, tdelta, tnext, tend = act[keep], cell[keep], step[keep], tdelta[keep], tnext[keep], \
                tend[keep]
        return best


def first_hit(origins, dirs, tris, tmax=None) -> np.ndarray:
    """(N,) distance (m) along each ray origins[i] + t·dirs[i] to the nearest triangle, inf where none is hit (or
    none nearer than tmax). origins (N, 3) or one (3,) for every ray; tris an (M, 3, 3) array or a Grid."""
    grid = tris if isinstance(tris, Grid) else Grid.build(tris)
    return grid.first_hit(origins, dirs, tmax)


def visible_exact(points, origin, tris, eps: float = 1e-3) -> np.ndarray:
    """(N,) True where no triangle lies on the segment from origin to the point, closer than the point's distance
    less eps (m): a point on a surface itself counts as visible when nothing else is in front of it."""
    P = np.asarray(points, np.float64).reshape(-1, 3)
    o = np.asarray(origin, np.float64).reshape(3)
    v = P - o
    dist = np.linalg.norm(v, axis=1)
    lim = np.maximum(dist - eps, 0.0)
    t = first_hit(o, v, tris, tmax=lim)
    return ~(t < lim)


def near_silhouette(points, origin, tris, angle_rad: float, eps: float = 1e-3, k: int = 8) -> np.ndarray:
    """(N,) True where the exact visibility of a point differs from that of a point at the same distance angle_rad
    away from it (k directions around the line of sight): the point lies within angle_rad of an occluder's
    silhouette as seen from origin. Used to judge where a per-texel answer may legitimately differ."""
    grid = tris if isinstance(tris, Grid) else Grid.build(tris)
    P = np.asarray(points, np.float64).reshape(-1, 3)
    o = np.asarray(origin, np.float64).reshape(3)
    v = P - o
    dist = np.linalg.norm(v, axis=1, keepdims=True)
    w = v / np.maximum(dist, 1e-300)
    a = np.where(np.abs(w[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    e1 = np.cross(w, a)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(w, e1)
    base = visible_exact(P, o, grid, eps)
    out = np.zeros(len(P), bool)
    r = dist * np.tan(angle_rad)
    for j in range(k):
        th = 2 * np.pi * j / k
        Q = P + r * (np.cos(th) * e1 + np.sin(th) * e2)
        out |= visible_exact(Q, o, grid, eps) != base
    return out


def brute_first_hit(origins, dirs, tris, chunk: int = 2_000_000) -> np.ndarray:
    """first_hit without the grid: every ray against every triangle (for tests on small inputs)."""
    tris = _as_tris(tris)
    d = np.asarray(dirs, np.float64).reshape(-1, 3)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    N, M = len(d), len(tris)
    o = np.broadcast_to(np.asarray(origins, np.float64).reshape(-1, 3), (N, 3))
    best = np.full(N, np.inf)
    if not N or not M:
        return best
    per = max(1, chunk // M)
    v0, e1, e2 = tris[:, 0], tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    for a in range(0, N, per):
        b = min(N, a + per)
        r = np.repeat(np.arange(a, b), M)
        k = np.tile(np.arange(M), b - a)
        t = moller_trumbore(o[r], d[r], v0[k], e1[k], e2[k]).reshape(b - a, M)
        best[a:b] = t.min(axis=1)
    return best
