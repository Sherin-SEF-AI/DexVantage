"""Gaussian scenes in DexVantage's frames: a set of 3D Gaussians, rendered from any camera pose through the project's
lens models (core/splat_raster.py), depth images lifted into points and re-projected into another camera, the
Gaussians' shapes from point neighbourhoods, and the standard 3D Gaussian Splatting .ply.

Ported from Vantage core/splat_reproject.py: the scene arrays and their torch counterpart (Scene, TorchScene), one
camera's render (render_view), the shapes from the eight-neighbour principal components (_shapes), voxel thinning
(_voxel_centroids), the image scores, and the INRIA .ply reader and writer (read_ply, ply_bytes, ply_scene).
Stripped (automotive): the recorded drive, its ego trajectory and lidar sweeps, the moving actors in boxes, the sky
texture, the rolling-shutter bands, the A/B rig comparison and the detector. Robots are core/sim/splat_rig.py's: it
attaches Gaussians to robot links and poses them by forward kinematics.

Frames and conventions (core/frames.py, docs/DEVELOPING.md): SI units; a pose is child -> parent,
p_parent = R p_child + t; a camera pose is T_world_optical, the camera's optical frame (REP-103: z forward, x right,
y down; OpenCV's camera axes) in the world; quaternions (x, y, z, w). Pixels are OpenCV's (core/optics/lens.py: the
centre of pixel (i, j) is u = i, v = j). The .ply stores rot_0..3 as (w, x, y, z), opacity as a logit, scale_0..2 as
natural logarithms and colour as the degree-0 spherical-harmonics coefficient f_dc = (c − ½) / C0 (the layout the
INRIA reference code, gsplat and nerfstudio write; Kerbl et al. 2023); the reader and writer convert.

Reprojection (backproject, reproject): a pixel (u, v) with depth z along the optical axis is the point
X_a = z · r(u, v) / r_z, r the model's unit ray of the pixel (CameraModel.unproject: the lens's distortion included);
in another camera b it is X_b = T_b⁻¹ T_a X_a, at pixel CameraModel.project(X_b). For two pinhole cameras with the
same intrinsics, b displaced by a baseline B along a's x axis, a point at depth z moves by the disparity fx·B / z
(tests/test_splat_reproject.py checks this case).

Known limits: as core/splat_raster.py (no view-dependent colour; spherical-harmonics terms of a .ply are dropped on
reading and written as zero); the shapes from neighbours are a starting point for the optimisation, not a surface
reconstruction.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from . import splat_raster as SR

SCHEMA = 'dexvantage.gaussians/1'
_PLY_TYPES = {'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8', 'uchar': 'u1', 'uint8': 'u1',
              'char': 'i1', 'int8': 'i1', 'short': 'i2', 'int16': 'i2', 'ushort': 'u2', 'uint16': 'u2', 'int': 'i4',
              'int32': 'i4', 'uint': 'u4', 'uint32': 'u4'}


# =========================================================================== the Gaussians
@dataclass
class GaussianSet:
    """N Gaussians in one frame: means (N, 3) m, scales (N, 3) σ in m, quats (N, 4) (x, y, z, w), opacity (N,) in
    [0, 1], colours (N, 3) RGB in [0, 1]. float32 arrays."""
    means: np.ndarray
    scales: np.ndarray
    quats: np.ndarray
    opacity: np.ndarray
    colours: np.ndarray
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        f = np.float32
        self.means = np.asarray(self.means, f).reshape(-1, 3)
        n = len(self.means)
        self.scales = np.asarray(self.scales, f).reshape(n, 3)
        self.quats = np.asarray(self.quats, f).reshape(n, 4)
        self.opacity = np.asarray(self.opacity, f).reshape(n)
        self.colours = np.asarray(self.colours, f).reshape(n, 3)

    @property
    def n(self) -> int:
        return int(len(self.means))

    def subset(self, idx) -> 'GaussianSet':
        return GaussianSet(self.means[idx], self.scales[idx], self.quats[idx], self.opacity[idx], self.colours[idx],
                           dict(self.info))

    def copy(self) -> 'GaussianSet':
        return self.subset(slice(None))

    @staticmethod
    def concat(sets) -> 'GaussianSet':
        sets = list(sets)
        if not sets:
            return GaussianSet(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 4)), np.zeros(0), np.zeros((0, 3)))
        return GaussianSet(*(np.concatenate([getattr(s, k) for s in sets]) for k in
                             ('means', 'scales', 'quats', 'opacity', 'colours')))

    def rotations(self) -> np.ndarray:
        return SR.quat_to_rot_np(self.quats)

    def transformed(self, T) -> 'GaussianSet':
        """The Gaussians moved rigidly by T (4 × 4, or (N, 4, 4) one per Gaussian): means T·μ, rotations R·Rᵢ."""
        T = np.asarray(T, float)
        R, t = T[..., :3, :3], T[..., :3, 3]
        if T.ndim == 2:
            P = self.means.astype(float) @ R.T + t
            Rg = np.einsum('ij,njk->nik', R, self.rotations())
        else:
            P = np.einsum('nij,nj->ni', R, self.means.astype(float)) + t
            Rg = R @ self.rotations()
        return GaussianSet(P, self.scales, SR.rot_to_quat_np(Rg), self.opacity, self.colours, dict(self.info))


def to_camera(means, rots, T_world_optical):
    """World centres (N, 3) and rotations (N, 3, 3) -> the camera's optical frame (numpy or torch)."""
    T = np.asarray(T_world_optical, float)
    W2C = np.linalg.inv(T)
    if isinstance(means, np.ndarray):
        return means @ W2C[:3, :3].T + W2C[:3, 3], np.einsum('ij,njk->nik', W2C[:3, :3], rots)
    import torch
    R = torch.as_tensor(W2C[:3, :3], dtype=means.dtype, device=means.device)
    t = torch.as_tensor(W2C[:3, 3], dtype=means.dtype, device=means.device)
    return means @ R.T + t, R[None] @ rots


def backend_for(backend: str, cam) -> str:
    """The back end a render uses: 'auto' is gsplat for a pinhole camera when gsplat runs here, else torch when it
    is installed, else numpy."""
    if backend not in SR.BACKENDS:
        raise ValueError(f'unknown rasteriser back end {backend!r}: use one of {", ".join(SR.BACKENDS)}')
    if backend != 'auto':
        return backend
    if cam.model == 'pinhole' and SR.gsplat_ok():
        return 'gsplat'
    return 'torch' if SR.torch_ok() else 'numpy'


def render(gs: GaussianSet, cam, T_world_optical, backend: str = 'auto', device=None, bg=(0.0, 0.0, 0.0),
           want_index: bool = False, depth: bool = False, dtype: str = 'float32') -> dict:
    """World-frame Gaussians through cam (optics.CameraModel) at T_world_optical (4 × 4 or frames.Pose).
    Returns numpy {'rgb' (H, W, 3), 'alpha' (H, W), 'depth' (H, W) expected range along the ray in m where alpha >
    0 (depth), 'index' (H, W) the dominant Gaussian, −1 none (want_index; torch and numpy back ends), 'backend'}.
    bg: the background colour (3,) or an (H, W, 3) image."""
    T = T_world_optical.matrix() if hasattr(T_world_optical, 'matrix') else np.asarray(T_world_optical, float)
    be = backend_for(backend, cam)
    if be == 'gsplat' and (want_index or depth):
        be = 'torch'
    H, W = cam.height, cam.width
    bgi = np.broadcast_to(np.asarray(bg, float), (H, W, 3)).reshape(H * W, 3)
    if be == 'numpy':
        Pc, Rc = to_camera(gs.means.astype(float), gs.rotations(), T)
        cov = SR.covariance(Rc, gs.scales.astype(float))
        feats = gs.colours.astype(float)
        bgf = bgi
        if depth:
            feats = np.c_[feats, np.linalg.norm(Pc, axis=1)]
            bgf = np.c_[bgi, np.zeros(H * W)]
        out = SR.splat_numpy(Pc, cov, feats, gs.opacity, cam, bg=bgf, want_index=want_index)
        img, alpha = out['image'], out['alpha']
        res = {'rgb': img[:, :3].reshape(H, W, 3), 'alpha': alpha.reshape(H, W), 'backend': be}
        if depth:
            with np.errstate(invalid='ignore', divide='ignore'):
                res['depth'] = np.where(alpha > 1e-6, img[:, 3] / np.maximum(alpha, 1e-12), 0.0).reshape(H, W)
        if want_index:
            res['index'] = out['index'].reshape(H, W)
        return res
    import torch
    dev = device if device is not None else SR.device_for('auto')
    dt = getattr(torch, dtype)

    def t(a):
        return torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=dev)
    with torch.no_grad():
        if be == 'gsplat':
            out = SR.splat_gsplat(t(gs.means), t(gs.quats), t(gs.scales), t(gs.opacity), t(gs.colours), cam, T,
                                  bg=np.asarray(bg, float).reshape(-1)[:3])
            return {'rgb': out['image'].cpu().numpy().astype(float), 'alpha': out['alpha'].cpu().numpy().astype(float),
                    'backend': be}
        Pc, Rc = to_camera(t(gs.means), SR.quat_to_rot(t(gs.quats)), T)
        cov = SR.covariance(Rc, t(gs.scales))
        feats = t(gs.colours)
        bgf = t(bgi)
        if depth:
            feats = torch.cat([feats, torch.linalg.norm(Pc, dim=1, keepdim=True)], 1)
            bgf = torch.cat([bgf, torch.zeros_like(bgf[:, :1])], 1)
        out = SR.splat(Pc, cov, feats, t(gs.opacity), cam, bg=bgf, want_index=want_index)
        img = out['image'].cpu().numpy().astype(float)
        alpha = out['alpha'].cpu().numpy().astype(float)
    res = {'rgb': img[:, :3].reshape(H, W, 3), 'alpha': alpha.reshape(H, W), 'backend': be}
    if depth:
        res['depth'] = np.where(alpha > 1e-6, img[:, 3] / np.maximum(alpha, 1e-12), 0.0).reshape(H, W)
    if want_index:
        res['index'] = out['index'].cpu().numpy().reshape(H, W)
    return res


# =========================================================================== reprojection
def backproject(depth, cam, T_world_optical, stride: int = 1, mask=None):
    """World points of the pixels of a depth image (H, W) (metres along the optical z axis, 0 = no hit) through cam
    at T_world_optical: (points (K, 3), pixel indices (K, 2) as (row, column)). stride keeps every stride-th row and
    column; mask (H, W) bool keeps only those pixels."""
    T = T_world_optical.matrix() if hasattr(T_world_optical, 'matrix') else np.asarray(T_world_optical, float)
    D = np.asarray(depth, float)
    H, W = D.shape
    rr, cc = np.mgrid[0:H:max(int(stride), 1), 0:W:max(int(stride), 1)]
    rr, cc = rr.ravel(), cc.ravel()
    z = D[rr, cc]
    keep = np.isfinite(z) & (z > 0)
    if mask is not None:
        keep &= np.asarray(mask, bool)[rr, cc]
    rr, cc, z = rr[keep], cc[keep], z[keep]
    rays = cam.unproject(np.c_[cc, rr].astype(float))
    ok = np.isfinite(rays).all(axis=1) & (rays[:, 2] > 1e-9)
    X = rays[ok] * (z[ok] / rays[ok, 2])[:, None]
    return X @ T[:3, :3].T + T[:3, 3], np.c_[rr[ok], cc[ok]]


def reproject(uv, z, cam_a, T_world_a, cam_b, T_world_b):
    """Pixels uv (K, 2) of camera a with depths z (K,) (optical z, m) -> (uv_b (K, 2), z_b (K,), ok (K,)) in camera b
    (module docstring, Reprojection). ok: the point is in front of b within its model and on its image."""
    Ta = T_world_a.matrix() if hasattr(T_world_a, 'matrix') else np.asarray(T_world_a, float)
    Tb = T_world_b.matrix() if hasattr(T_world_b, 'matrix') else np.asarray(T_world_b, float)
    uv = np.asarray(uv, float).reshape(-1, 2)
    z = np.asarray(z, float).reshape(-1)
    rays = cam_a.unproject(uv)
    Xa = rays * (z / rays[:, 2])[:, None]
    M = np.linalg.inv(Tb) @ Ta
    Xb = Xa @ M[:3, :3].T + M[:3, 3]
    uvb, ok = cam_b.project(Xb)
    return uvb, Xb[:, 2], ok & np.isfinite(Xa).all(axis=1)


# =========================================================================== shapes from points
def voxel_centroids(P, size, values=None):
    """Centroids of the points in each voxel of the given size (one, or one per point): (centroids (M, 3), counts (M,),
    the mean of values (N, k) per voxel or None, the voxel of every point (N,))."""
    P = np.asarray(P, float).reshape(-1, 3)
    if not len(P):
        mv = None if values is None else np.zeros((0,) + np.shape(values)[1:])
        return np.zeros((0, 3)), np.zeros(0, int), mv, np.zeros(0, int)
    size = np.broadcast_to(np.asarray(size, float), (len(P),))
    key = np.floor(P / size[:, None]).astype(np.int64)
    key = np.concatenate([key, np.round(np.log2(size) * 4).astype(np.int64)[:, None]], 1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    out = np.zeros((len(cnt), 3))
    np.add.at(out, inv, P)
    mv = None
    if values is not None:
        V = np.asarray(values, float).reshape(len(P), -1)
        mv = np.zeros((len(cnt), V.shape[1]))
        np.add.at(mv, inv, V)
        mv /= cnt[:, None]
    return out / cnt[:, None], cnt, mv, inv


def thin_to_budget(P, base: float, budget: int, values=None, grow: float = 1.2):
    """Voxel centroids of P at the voxel size base, grown by `grow` until at most budget remain: (centroids, voxel
    size used, mean values)."""
    for _ in range(60):
        C, _cnt, mv, _inv = voxel_centroids(P, base, values)
        if len(C) <= budget:
            return C, base, mv
        base *= grow
    return C[:budget], base, (None if mv is None else mv[:budget])


def shapes_from_points(P, voxel) -> tuple:
    """(scales (n, 3), quats (n, 4) (x, y, z, w)) from the eight-neighbour principal components: a flat
    neighbourhood (smallest eigenvalue under a tenth of the middle one) gives a disc along the surface, σ along it half
    the mean distance to the 3 nearest neighbours (at most three quarters of its voxel, at least 1 mm), across it a
    tenth of that (at least 0.3 mm); otherwise a sphere of that σ. voxel: the thinning voxel size (one, or one per
    point). Ported from Vantage splat_reproject._shapes; the floors (1 mm, 0.3 mm) are DexVantage defaults for
    robot-scale scenes (Vantage's were 4 mm and 3 mm for streets)."""
    from scipy.spatial import cKDTree
    P = np.asarray(P, float).reshape(-1, 3)
    n = len(P)
    if n == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 4), np.float32)
    k = min(9, n)
    vox = np.broadcast_to(np.asarray(voxel, float), (n,))
    scales_floor, normal_floor = 1e-3, 3e-4
    if k == 1:
        sig = np.maximum(0.5 * vox, scales_floor)
        return np.repeat(sig[:, None], 3, 1).astype(np.float32), np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)).astype(
            np.float32)
    tree = cKDTree(P)
    dist, idx = tree.query(P, k=k)
    dist, idx = np.atleast_2d(dist), np.atleast_2d(idx)
    nn = dist[:, 1:min(4, k)].mean(1)
    sig = np.clip(0.5 * nn, scales_floor, np.maximum(0.75 * vox, scales_floor))
    scales = np.repeat(sig[:, None], 3, 1)
    quats = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
    if k >= 4:
        Q = P[idx] - P[idx].mean(1, keepdims=True)
        C = np.einsum('nki,nkj->nij', Q, Q) / k
        lam, vec = np.linalg.eigh(C)                     # ascending
        flat = lam[:, 0] < 0.1 * np.maximum(lam[:, 1], 1e-12)
        R = vec.copy()
        R[np.linalg.det(R) < 0, :, 0] *= -1              # right-handed, the normal last
        R = R[:, :, [1, 2, 0]]
        scales[flat, 2] = np.maximum(0.1 * sig[flat], normal_floor)
        if flat.any():
            quats[flat] = SR.rot_to_quat_np(R[flat])
    return scales.astype(np.float32), quats.astype(np.float32)


# =========================================================================== the .ply
def read_ply(path) -> dict:
    """The vertex element of a .ply file (binary little-endian or ASCII) as {property: array}."""
    with open(path, 'rb') as fh:
        head = b''
        while not head.endswith(b'end_header\n'):
            line = fh.readline()
            if not line:
                raise ValueError(f'{os.path.basename(path)} has no PLY header.')
            head += line.replace(b'\r\n', b'\n')
        lines = head.decode('ascii', 'replace').splitlines()
        if not lines or lines[0].strip() != 'ply':
            raise ValueError(f'{os.path.basename(path)} is not a PLY file.')
        fmt = next((ln.split()[1] for ln in lines if ln.startswith('format')), '')
        n, props, cur = 0, [], None
        for ln in lines:
            p = ln.split()
            if p and p[0] == 'element':
                cur = p[1]
                if cur == 'vertex':
                    n = int(p[2])
                elif n == 0:
                    raise ValueError(f'{os.path.basename(path)}: elements before the vertices are not supported.')
            elif p and p[0] == 'property' and cur == 'vertex':
                if p[1] == 'list':
                    raise ValueError(f'{os.path.basename(path)}: list properties on the vertices are not supported.')
                if p[1] not in _PLY_TYPES:
                    raise ValueError(f'{os.path.basename(path)}: the property type {p[1]} is not supported.')
                props.append((p[2], _PLY_TYPES[p[1]]))
        dt = np.dtype([(k, '<' + t) for k, t in props])
        if fmt == 'binary_little_endian':
            raw = fh.read(n * dt.itemsize)
            if len(raw) < n * dt.itemsize:
                raise ValueError(f'{os.path.basename(path)} is shorter than its header says.')
            a = np.frombuffer(raw, dt, count=n)
        elif fmt == 'ascii':
            a = np.loadtxt(fh, max_rows=n, ndmin=2)
            a = np.rec.fromarrays([a[:, k].astype(dt[k]) for k in range(len(props))], dtype=dt)
        else:
            raise ValueError(f'PLY format {fmt} is not supported (binary_little_endian or ascii).')
    return {k: np.asarray(a[k]) for k in dt.names}


def gaussians_from_ply(path, T=None, scale: float = 1.0, max_gaussians: int | None = None) -> GaussianSet:
    """A standard 3DGS .ply as a GaussianSet, placed by p' = scale · R p + t (T 4 × 4, default identity); the
    spherical-harmonics terms beyond degree 0 are dropped. A file without scales gets shapes from its points'
    neighbours, one without rotations identity ones, without opacity 0.9, without colour grey."""
    v = read_ply(path)
    if not all(k in v for k in ('x', 'y', 'z')):
        raise ValueError(f'{os.path.basename(path)} has no x, y, z.')
    P = np.stack([v['x'], v['y'], v['z']], 1).astype(float)
    n = len(P)
    T = np.eye(4) if T is None else np.asarray(T, float)
    R, t = T[:3, :3], T[:3, 3]
    k = float(scale)
    Pg = k * P @ R.T + t
    if all(f'scale_{i}' in v for i in range(3)):
        scales = np.exp(np.stack([v[f'scale_{i}'] for i in range(3)], 1).astype(float)) * k
    else:
        scales = shapes_from_points(P, 0.01)[0] * k
    if all(f'rot_{i}' in v for i in range(4)):
        q = np.stack([v[f'rot_{i}'] for i in range(4)], 1).astype(float)[:, [1, 2, 3, 0]]
        quats = SR.rot_to_quat_np(np.einsum('ij,njk->nik', R, SR.quat_to_rot_np(q)))
    else:
        quats = SR.rot_to_quat_np(np.repeat(R[None], n, 0))
    opacity = 1 / (1 + np.exp(-v['opacity'].astype(float))) if 'opacity' in v else np.full(n, 0.9)
    if all(f'f_dc_{i}' in v for i in range(3)):
        col = 0.5 + SR.C0 * np.stack([v[f'f_dc_{i}'] for i in range(3)], 1).astype(float)
    elif all(c in v for c in ('red', 'green', 'blue')):
        col = np.stack([v['red'], v['green'], v['blue']], 1).astype(float)
        col = col / (255.0 if col.max() > 1.5 else 1.0)
    else:
        col = np.full((n, 3), 0.5)
    keep = np.all(np.isfinite(Pg), 1) & np.all(np.isfinite(scales), 1) & (opacity > 0.004)
    idx = np.flatnonzero(keep)
    if max_gaussians is not None and len(idx) > max_gaussians:
        idx = idx[np.sort(np.argsort(-opacity[idx], kind='stable')[:int(max_gaussians)])]
    return GaussianSet(Pg[idx], np.clip(scales[idx], 1e-5, 50), quats[idx], np.clip(opacity[idx], 0, 1),
                       np.clip(col[idx], 0, 1), {'gaussians_in_file': int(n), 'file': os.path.basename(str(path))})


PLY_NAMES = (['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2'] + [f'f_rest_{i}' for i in range(45)] +
             ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'])


def ply_bytes(gs: GaussianSet, comment: str = 'DexVantage 3D Gaussians, metres') -> bytes:
    """The Gaussians as a standard 3DGS .ply (INRIA layout, degree-0 colour, f_rest zero, binary little-endian)."""
    n = gs.n
    a = np.zeros((n, len(PLY_NAMES)), np.float32)
    a[:, 0:3] = gs.means
    a[:, 6:9] = (gs.colours - 0.5) / SR.C0
    op = np.clip(gs.opacity.astype(float), 1e-6, 1 - 1e-6)
    a[:, 54] = np.log(op / (1 - op))
    a[:, 55:58] = np.log(np.maximum(gs.scales, 1e-8))
    a[:, 58:62] = gs.quats[:, [3, 0, 1, 2]]
    comment = ' '.join(str(comment).split())
    head = (f'ply\nformat binary_little_endian 1.0\ncomment {comment}\nelement vertex {n}\n' +
            ''.join(f'property float {k}\n' for k in PLY_NAMES) + 'end_header\n')
    return head.encode('ascii', 'replace') + a.tobytes()


# =========================================================================== scores
def image_scores(pred, target, mask=None) -> dict:
    """{'psnr_db', 'ssim'} of a render against an image (both (H, W, 3) in [0, 1]) over the mask's pixels."""
    return {'psnr_db': SR.psnr(pred, target, None if mask is None else np.broadcast_to(
        np.asarray(mask, bool)[..., None], np.shape(pred))), 'ssim': SR.ssim_np(pred, target, mask)}
