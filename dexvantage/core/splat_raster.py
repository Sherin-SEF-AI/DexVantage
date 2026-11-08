"""A differentiable rasteriser of anisotropic 3D Gaussians through DexVantage's camera models, with three back ends:
plain torch (CUDA when there is one, else the CPU), gsplat (CUDA, pinhole only, when it is installed) and a numpy
reference for tests and small renders without torch. core/splat_reproject.py and core/sim/splat_rig.py render with it.

Ported from Vantage core/splat_raster.py (the torch rasteriser, its tile binning and compositing, the SSIM and PSNR).
Adapted: DexVantage's lens models (core/optics/lens.py: pinhole, Brown-Conrady, Kannala-Brandt and the classical
fisheye projections) and its pixel convention (OpenCV's: the centre of pixel (i, j) is u = i, v = j; Vantage put
centres at i + ½), quaternions in DexVantage's order (x, y, z, w; core/frames.py); stripped: the lidar's spherical
projector and the planar-depth extension used for lidar ranges. Added: an explicit per-pixel cut-off (below), which
makes the result independent of the tiling, the numpy reference (splat_numpy) and the gsplat back end (splat_gsplat).

The method is 3D Gaussian splatting's (Kerbl, Kopanas, Leimkühler and Drettakis, "3D Gaussian Splatting for Real-Time
Radiance Field Rendering", ACM TOG 42(4), 2023; references key kerbl2023_3dgs):

* Each Gaussian has a centre μ, a covariance Σ = R S Sᵀ Rᵀ (R from a unit quaternion (x, y, z, w); S the diagonal of
  its three scales, metres), an opacity o in [0, 1] and features (colour, depth, …).
* Projection: the centre goes through the camera model (u = fx·xd + cx, v = fy·yd + cy, (xd, yd) the model's
  normalised image coordinates, core/optics/lens.py) and the covariance through the projection's Jacobian J at the
  centre (the local affine approximation of EWA splatting, Zwicker, Pfister, van Baar and Gross 2001; key
  zwicker2001_ewa): Σ₂ = J Σ_c Jᵀ + LOWPASS·I (px²). J is taken by central differences (step 10⁻³ of the range), so
  every lens model works, including a fisheye beyond 90°.
* Footprint: m = min(3, √(2 ln(255 o))) standard deviations, the Mahalanobis radius where the alpha falls to 1/255
  (or 3σ, whichever is smaller). A Gaussian contributes to a pixel only inside that ellipse (dᵀ Σ₂⁻¹ d ≤ m²). Vantage
  (and 3DGS) let a Gaussian reach every pixel of the 16 × 16 tiles its bounding square touches; the explicit cut-off
  here makes the image the same whatever the tiling, which the numpy reference relies on. It changes a pixel by less
  than the 1/255 alpha threshold wherever o ≤ 0.35 and, above that, only in the ring between 3σ and the 1/255 contour.
* Compositing: per tile the Gaussians whose footprint square reaches it, sorted by range from the camera centre (not
  by z: a fisheye sees beyond 90°), front to back: αᵢ = min(0.99, oᵢ·exp(−½ dᵀ Σ₂⁻¹ d)), dropped below 1/255;
  C = Σ cᵢ αᵢ Tᵢ + T_N · background, Tᵢ = Π_{j<i} (1 − αⱼ). Without gradients a tile stops when every pixel's
  transmittance is below 10⁻⁴; with gradients a tile composites at most max_per_tile Gaussians (the front-most; the
  rest is cut and counted).
* Gradients flow to the centres, covariances (scales, rotations), opacities, features and the background through
  torch's autograd. The binning (tiles, depth order, the footprint) is not differentiable, as in 3DGS.

gsplat back end (splat_gsplat): gsplat.rasterization (Ye et al., "gsplat: An Open-Source Library for Gaussian
Splatting", JMLR 26(34), 2025; key ye2025_gsplat) for pinhole cameras. gsplat sorts by the camera z, assigns
Gaussians to tiles by a 3σ square and evaluates every pixel of those tiles, and puts pixel centres at i + ½: the
principal point is passed as cx + ½. Its images agree with the torch path's to the differences those rules make
(tests/test_splat_raster.py measures them). gsplat publishes no wheel for Python 3.14 or torch 2.14:
scripts/build_gsplat.sh builds 1.5.3 from source (docs/models/splat.md, "Environment"). The fit always uses the torch
path, and so do the splat views study's renders; 'auto' takes gsplat only for other pinhole renders.

Known limits: the affine approximation of the projection is poor for a Gaussian that spans a large angle (close to
the camera, on a fisheye's rim); the depth order is the centres' (overlapping Gaussians of different sizes can pop);
no view-dependent colour (spherical-harmonics degree 0); CUDA backward passes use atomic adds, so CUDA training is not
bit-for-bit repeatable (the CPU path is).
"""
from __future__ import annotations

import math

import numpy as np

TILE = 16
LOWPASS = 0.3                       # px², added to every projected covariance (3DGS's dilation; a 3DGS default)
ALPHA_MIN = 1.0 / 255.0
ALPHA_MAX = 0.99
T_STOP = 1e-4
MAX_SIGMA = 3.0                     # the footprint never reaches beyond 3σ
C0 = 0.28209479177387814            # the degree-0 spherical-harmonics constant (the .ply colour convention)
BACKENDS = ('auto', 'torch', 'gsplat', 'numpy')

TORCH_MISSING = ('{what} needs PyTorch, which is not installed: install the splat extra (pip install '
                 '"dexvantage[splat]" --extra-index-url https://download.pytorch.org/whl/cu130), or use the numpy '
                 'reference for small renders.')


def torch_ok() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def gsplat_ok() -> bool:
    """True when gsplat imports and has CUDA to run on."""
    try:
        import gsplat  # noqa: F401
        import torch
        return bool(torch.cuda.is_available())
    except Exception:                                    # noqa: BLE001 — an import that fails to build is "not here"
        return False


def cuda_ok() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def device_for(pref: str = 'auto', mem_gb: float = 3.0):
    """The torch device: CUDA when asked for ('auto' or 'cuda'), present and with at least min(mem_gb, 1) GB free,
    its share of the GPU capped at mem_gb; else the CPU."""
    import torch
    if pref in ('auto', 'cuda') and torch.cuda.is_available():
        try:
            free, _ = torch.cuda.mem_get_info(0)
        except RuntimeError:                              # pragma: no cover — a busy or broken driver
            free = 0
        if free < min(mem_gb, 1.0) * 1e9:                 # the GPU is busy with other programs: the CPU
            return torch.device('cpu')
        try:
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(max(mem_gb * 1e9 / total, 0.02), 0.95), 0)
        except (RuntimeError, AssertionError):            # pragma: no cover — a driver that refuses the cap
            pass
        return torch.device('cuda', 0)
    _cpu_threads(torch)
    return torch.device('cpu')


def _cpu_threads(torch):
    """At most 8 CPU threads (half the cores): torch's spinning worker threads slow to a crawl when the machine is
    shared with the user interface and other work."""
    import os
    n = max(1, min(8, (os.cpu_count() or 2) // 2))
    if torch.get_num_threads() > n:
        torch.set_num_threads(n)


# =========================================================================== geometry
def quat_to_rot(q):
    """(N, 4) quaternions (x, y, z, w) (normalised here) -> (N, 3, 3) rotation matrices (torch)."""
    import torch
    q = q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1e-12)
    x, y, z, w = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


def quat_to_rot_np(q) -> np.ndarray:
    """(N, 4) (x, y, z, w) -> (N, 3, 3) (numpy)."""
    q = np.asarray(q, float).reshape(-1, 4)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    x, y, z, w = q.T
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)


def rot_to_quat_np(R) -> np.ndarray:
    """(N, 3, 3) rotation matrices -> (N, 4) unit quaternions (x, y, z, w) with w ≥ 0 (numpy; Shepperd's method)."""
    R = np.asarray(R, float).reshape(-1, 3, 3)
    n = len(R)
    q = np.zeros((n, 4))                                  # w, x, y, z while computing
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cases = np.stack([tr, R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]], 1).argmax(1)
    for c in range(4):
        m = cases == c
        if not m.any():
            continue
        r = R[m]
        if c == 0:
            s = np.sqrt(1.0 + tr[m]) * 2
            q[m] = np.stack([0.25 * s, (r[:, 2, 1] - r[:, 1, 2]) / s, (r[:, 0, 2] - r[:, 2, 0]) / s,
                             (r[:, 1, 0] - r[:, 0, 1]) / s], 1)
        elif c == 1:
            s = np.sqrt(1.0 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2]) * 2
            q[m] = np.stack([(r[:, 2, 1] - r[:, 1, 2]) / s, 0.25 * s, (r[:, 0, 1] + r[:, 1, 0]) / s,
                             (r[:, 0, 2] + r[:, 2, 0]) / s], 1)
        elif c == 2:
            s = np.sqrt(1.0 + r[:, 1, 1] - r[:, 0, 0] - r[:, 2, 2]) * 2
            q[m] = np.stack([(r[:, 0, 2] - r[:, 2, 0]) / s, (r[:, 0, 1] + r[:, 1, 0]) / s, 0.25 * s,
                             (r[:, 1, 2] + r[:, 2, 1]) / s], 1)
        else:
            s = np.sqrt(1.0 + r[:, 2, 2] - r[:, 0, 0] - r[:, 1, 1]) * 2
            q[m] = np.stack([(r[:, 1, 0] - r[:, 0, 1]) / s, (r[:, 0, 2] + r[:, 2, 0]) / s,
                             (r[:, 1, 2] + r[:, 2, 1]) / s, 0.25 * s], 1)
    q *= np.where(q[:, :1] < 0, -1.0, 1.0)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    return q[:, [1, 2, 3, 0]]


def covariance(rot, scales):
    """Σ = R diag(s²) Rᵀ: (N, 3, 3) rotations and (N, 3) scales (metres) -> (N, 3, 3) covariances (torch or numpy)."""
    M = rot * scales[:, None, :]
    return M @ M.transpose(0, 2, 1) if isinstance(M, np.ndarray) else M @ M.transpose(1, 2)


# =========================================================================== projection (torch)
def _r_norm(theta, model: str, dist):
    """Normalised image height r(θ)/f of a radial model (torch; core.optics.lens.r_of_theta)."""
    if model == 'equidistant':
        return theta
    if model == 'equisolid':
        import torch
        return 2 * torch.sin(theta / 2)
    if model == 'stereographic':
        import torch
        return 2 * torch.tan(theta / 2)
    if model == 'kb':
        k1, k2, k3, k4 = (list(dist) + [0.0] * 4)[:4]
        t2 = theta * theta
        return theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    raise ValueError(f'the {model} model is not radial')


class TorchProjector:
    """A CameraModel (core/optics/lens.py) in torch: optical-frame points (x right, y down, z forward) -> (u, v, ok),
    OpenCV pixels (pixel (i, j) centred on u = i, v = j). ok where the model holds: within max_theta of the axis, in
    front of a pinhole or Brown-Conrady camera, as CameraModel.project's valid set (not clipped to the image)."""

    def __init__(self, cam):
        self.cam = cam
        self.model = cam.model
        self.dist = tuple(float(d) for d in cam.dist)
        self.fx, self.fy, self.cx, self.cy = float(cam.fx), float(cam.fy), float(cam.cx), float(cam.cy)
        self.max_theta = float(cam.max_theta)

    def __call__(self, P):
        import torch
        x, y, z = P.unbind(-1)
        rho = torch.sqrt(x * x + y * y + 1e-30)
        th = torch.atan2(rho, z)
        ok = th <= self.max_theta
        if self.model in ('pinhole', 'brown'):
            ok = ok & (z > 1e-9)
            zz = torch.where(ok, z, torch.ones_like(z))
            xn, yn = x / zz, y / zz
            if self.model == 'brown':
                d = (list(self.dist) + [0.0] * 8)[:8]
                k1, k2, p1, p2, k3, k4, k5, k6 = d
                r2 = xn * xn + yn * yn
                c = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
                xd = xn * c + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
                yd = yn * c + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
                xn, yn = xd, yd
        else:
            g = _r_norm(th, self.model, self.dist)
            s = g / rho
            xn, yn = x * s, y * s
        return self.fx * xn + self.cx, self.fy * yn + self.cy, ok


# =========================================================================== the torch rasteriser
def _footprint(a, b, c, op):
    """(Mahalanobis radius m (n,), square half-size rad px (n,)) of projected covariances [[a, b], [b, c]] and opacities
    (module docstring, Footprint)."""
    import torch
    det = torch.clamp(a * c - b * b, min=1e-12)
    mid = 0.5 * (a + c)
    lam = mid + torch.sqrt(torch.clamp(mid * mid - det, min=0.0))
    m = torch.sqrt(torch.clamp(2.0 * torch.log(torch.clamp(op * 255.0, min=1.0 + 1e-6)), min=0.0))
    m = m.clamp(max=MAX_SIGMA)
    return m, torch.ceil(m * torch.sqrt(lam))


def _tile_lists(u, v, rad, keep, W, H):
    """The (tile, Gaussian) pairs sorted by tile, then by the Gaussians' order (they arrive sorted by range): returns
    (tile id per pair, Gaussian per pair, tiles across). u, v in OpenCV pixels; pixel i covers [i − ½, i + ½)."""
    import torch
    dev = u.device
    TW = (W + TILE - 1) // TILE
    TH = (H + TILE - 1) // TILE
    ue, ve = u + 0.5, v + 0.5                              # pixel-edge coordinates: pixel i covers [i, i + 1)
    tx0 = torch.clamp(torch.floor((ue - rad) / TILE), 0, TW - 1).long()
    tx1 = torch.clamp(torch.floor((ue + rad) / TILE), 0, TW - 1).long()
    ty0 = torch.clamp(torch.floor((ve - rad) / TILE), 0, TH - 1).long()
    ty1 = torch.clamp(torch.floor((ve + rad) / TILE), 0, TH - 1).long()
    inside = keep & (ue + rad >= 0) & (ue - rad < W) & (ve + rad >= 0) & (ve - rad < H)
    nw = (tx1 - tx0 + 1)
    cnt = torch.where(inside, nw * (ty1 - ty0 + 1), torch.zeros_like(nw))
    total = int(cnt.sum())
    if total == 0:
        z = torch.zeros(0, dtype=torch.long, device=dev)
        return z, z, TW
    gid = torch.repeat_interleave(torch.arange(len(u), device=dev), cnt)
    first = torch.cumsum(cnt, 0) - cnt
    k = torch.arange(total, device=dev) - torch.repeat_interleave(first, cnt)
    w = nw[gid]
    tile = (ty0[gid] + k // w) * TW + tx0[gid] + k % w
    order = torch.sort(tile, stable=True).indices          # gid ascends within a tile: the range order
    return tile[order], gid[order], TW


def _jacobian(proj, P, r):
    """(n, 2, 3) Jacobian of the projection at P by central differences (step 10⁻³ of the range)."""
    import torch
    h = (1e-3 * r.detach()).clamp(min=1e-5)
    eye = torch.eye(3, device=P.device, dtype=P.dtype)
    Q = torch.cat([P[None] + h[None, :, None] * eye[:, None, :], P[None] - h[None, :, None] * eye[:, None, :]])
    uq, vq, _ = proj(Q.reshape(-1, 3))
    uq, vq = uq.reshape(6, -1), vq.reshape(6, -1)
    return torch.stack([torch.stack([(uq[k] - uq[k + 3]) / (2 * h) for k in range(3)], -1),
                        torch.stack([(vq[k] - vq[k + 3]) / (2 * h) for k in range(3)], -1)], 1)


def splat(Pc, cov_c, feats, opacity, cam, bg=None, near: float = 0.01, block: int = 512, max_per_tile: int = 4096,
          want_index: bool = False, chunk_elems: int | None = None, lowpass: float = LOWPASS) -> dict:
    """Render Gaussians already in the camera's optical frame (torch).

    Pc (N, 3) centres, cov_c (N, 3, 3) covariances, feats (N, F), opacity (N,) in [0, 1] (torch, one device and
    dtype); cam an optics.CameraModel (or a TorchProjector of one); bg (H·W, F) background per pixel (row-major) or
    None (zeros). Returns {'image' (H·W, F): Σ fᵢ αᵢ Tᵢ + T·bg, 'alpha' (H·W,): 1 − T, 'index' (H·W,) long: the
    Gaussian with the largest weight αᵢTᵢ per pixel (−1 none; want_index), 'cut' int: tile entries left out
    (max_per_tile, with gradients only), 'pairs' int}."""
    import torch
    from torch.utils.checkpoint import checkpoint
    proj = cam if isinstance(cam, TorchProjector) else TorchProjector(cam)
    W, H = int(proj.cam.width), int(proj.cam.height)
    dev = Pc.device
    F = feats.shape[1]
    TH = (H + TILE - 1) // TILE
    grad = torch.is_grad_enabled() and (Pc.requires_grad or cov_c.requires_grad or feats.requires_grad or
                                        opacity.requires_grad or (bg is not None and bg.requires_grad))
    if chunk_elems is None:
        cuda = dev.type == 'cuda'
        chunk_elems = (1 << 23 if cuda else 1 << 21) if grad else (1 << 24 if cuda else 1 << 22)
    bgf = bg if bg is not None else torch.zeros(H * W, F, device=dev, dtype=feats.dtype)
    out = {'image': bgf, 'alpha': torch.zeros(H * W, device=dev, dtype=feats.dtype),
           'index': torch.full((H * W,), -1, dtype=torch.long, device=dev) if want_index else None,
           'cut': 0, 'pairs': 0}
    if len(Pc) == 0:
        return out
    rng = torch.linalg.norm(Pc, dim=-1)
    u, v, ok = proj(Pc)
    ok = ok & (rng > near) & (opacity > ALPHA_MIN)
    idx = torch.nonzero(ok).squeeze(1)
    if len(idx) == 0:
        return out
    P = Pc[idx]
    r = rng[idx]
    J = _jacobian(proj, P, r)
    S2 = J @ cov_c[idx] @ J.transpose(1, 2)
    a = S2[:, 0, 0] + lowpass
    b = S2[:, 0, 1]
    c = S2[:, 1, 1] + lowpass
    det = torch.clamp(a * c - b * b, min=1e-12)
    con = torch.stack([c / det, -b / det, a / det], -1)
    uu, vv = u[idx], v[idx]
    op = opacity[idx]
    with torch.no_grad():
        msig, rad = _footprint(a.detach(), b.detach(), c.detach(), op.detach())
        m2 = msig * msig
        good = torch.isfinite(rad) & torch.isfinite(uu) & torch.isfinite(vv) & (rad < 4 * max(W, H))
        order = torch.argsort(r.detach(), stable=True)                  # front to back
        tile, gid_s, TW = _tile_lists(uu.detach()[order], vv.detach()[order], rad[order], good[order], W, H)
    gid = order[gid_s]
    out['pairs'] = int(len(gid))
    if len(gid) == 0:
        return out
    with torch.no_grad():
        ntiles = TW * TH
        counts = torch.bincount(tile, minlength=ntiles)
        starts = torch.cumsum(counts, 0) - counts
        live = torch.nonzero(counts).squeeze(1)
        live = live[torch.argsort(counts[live], descending=True, stable=True)]
        cnt_live = counts[live].tolist()
    fv = feats[idx]
    lx = torch.arange(TILE, device=dev, dtype=P.dtype)
    gy, gx = torch.meshgrid(lx, lx, indexing='ij')
    gx, gy = gx.reshape(-1), gy.reshape(-1)
    pieces_i, pieces_v, pieces_a, pieces_x = [], [], [], []
    i = 0
    nl = len(live)
    while i < nl:
        nmax = cnt_live[i]
        kmax = min(nmax, max_per_tile) if grad else nmax
        kb = min(block, kmax)
        n_c = max(1, min(nl - i, chunk_elems // (TILE * TILE * kb)))
        tiles = live[i:i + n_c]
        i += n_c
        tx, ty = tiles % TW, tiles // TW
        px = (tx[:, None] * TILE + gx[None]).to(P.dtype)                 # pixel indices = OpenCV pixel centres
        py = (ty[:, None] * TILE + gy[None]).to(P.dtype)
        pix_ok = (px < W) & (py < H)
        Tcur = torch.ones(len(tiles), TILE * TILE, device=dev, dtype=P.dtype)
        acc = torch.zeros(len(tiles), TILE * TILE, F, device=dev, dtype=feats.dtype)
        best_w = torch.zeros(len(tiles), TILE * TILE, device=dev, dtype=P.dtype) if want_index else None
        best_g = torch.full((len(tiles), TILE * TILE), -1, dtype=torch.long, device=dev) if want_index else None
        tc = counts[tiles]
        if grad:
            out['cut'] += int(torch.clamp(tc - max_per_tile, min=0).sum())
            tc = torch.clamp(tc, max=max_per_tile)
        ts = starts[tiles]
        for k0 in range(0, kmax, kb):
            kk = min(kb, kmax - k0)
            ar = torch.arange(k0, k0 + kk, device=dev)
            valid = ar[None, :] < tc[:, None]
            pos = torch.clamp(ts[:, None] + ar[None, :], max=len(gid) - 1)
            g = gid[pos]
            if grad:
                add, Tb_last = checkpoint(_block, px, py, uu, vv, con, op, m2, fv, g, valid, Tcur, False,
                                          use_reentrant=False)
            else:
                add, Tb_last, wgt = _block(px, py, uu, vv, con, op, m2, fv, g, valid, Tcur, True)
                if want_index:
                    bw, bk = wgt.max(-1)
                    better = bw > best_w
                    best_w = torch.where(better, bw, best_w)
                    best_g = torch.where(better, torch.gather(g, 1, bk), best_g)
            acc = acc + add
            Tcur = Tcur * Tb_last
            if not grad and bool((Tcur < T_STOP).all()):
                break
        flat = (py * W + px).long()
        pieces_i.append(flat[pix_ok])
        pieces_v.append(acc[pix_ok])
        pieces_a.append(Tcur[pix_ok])
        if want_index:
            pieces_x.append(best_g[pix_ok])
    pi = torch.cat(pieces_i)
    T = torch.cat(pieces_a)
    out['image'] = bgf.index_copy(0, pi, torch.cat(pieces_v) + T[:, None] * bgf[pi])
    out['alpha'] = out['alpha'].index_copy(0, pi, 1.0 - T)
    if want_index:
        bx = torch.cat(pieces_x)
        out['index'] = out['index'].index_copy(0, pi, torch.where(bx >= 0, idx[bx.clamp(min=0)], bx))
    return out


def _block(px, py, uu, vv, con, op, m2, fv, g, valid, Tcur, keep_w=False):
    """One block of a chunk of tiles: the Gaussians g (tiles × k, front to back) composited over the tiles' pixels
    with the transmittance Tcur carried in. Returns (Σ f α T, the block's own transmittance product[, the weights])."""
    import torch
    dx = px[:, :, None] - uu[g][:, None, :]
    dy = py[:, :, None] - vv[g][:, None, :]
    cg = con[g]
    power = -0.5 * (cg[:, None, :, 0] * dx * dx + cg[:, None, :, 2] * dy * dy) - cg[:, None, :, 1] * dx * dy
    inside = (-2.0 * power.detach()) <= m2[g][:, None, :]
    al = torch.clamp(op[g][:, None, :] * torch.exp(torch.clamp(power, max=0.0)), max=ALPHA_MAX)
    al = torch.where(valid[:, None, :] & inside & (al >= ALPHA_MIN), al, torch.zeros_like(al))
    Tb = torch.cumprod(1.0 - al, -1)
    Tex = torch.cat([torch.ones_like(Tb[..., :1]), Tb[..., :-1]], -1) * Tcur[..., None]
    wgt = al * Tex
    add = torch.einsum('cpk,ckf->cpf', wgt, fv[g])
    if keep_w:
        return add, Tb[..., -1], wgt
    return add, Tb[..., -1]


# =========================================================================== the numpy reference
def splat_numpy(Pc, cov_c, feats, opacity, cam, bg=None, near: float = 0.01, want_index: bool = False,
                lowpass: float = LOWPASS) -> dict:
    """The same image as splat, written again in numpy without tiles (a reference for the tests and small renders
    without torch): every Gaussian, front to back by range, composited over the pixels of its footprint square.
    The projection is core.optics.CameraModel.project itself (float64), its Jacobian by the same central
    differences. Inputs numpy; returns {'image' (H·W, F), 'alpha' (H·W,), 'index' (H·W,) (want_index)}."""
    W, H = cam.width, cam.height
    Pc = np.asarray(Pc, float).reshape(-1, 3)
    cov_c = np.asarray(cov_c, float).reshape(-1, 3, 3)
    feats = np.asarray(feats, float).reshape(len(Pc), -1)
    opacity = np.asarray(opacity, float).reshape(-1)
    F = feats.shape[1]
    img = np.zeros((H, W, F))
    T = np.ones((H, W))
    best_w = np.zeros((H, W))
    best = np.full((H, W), -1, np.int64)
    rng = np.linalg.norm(Pc, axis=1)
    uv, _ = cam.project(Pc)
    ok = np.isfinite(uv).all(axis=1) & (rng > near) & (opacity > ALPHA_MIN)
    for i in np.argsort(np.where(ok, rng, np.inf), kind='stable'):
        if not ok[i]:
            break
        P, r = Pc[i], rng[i]
        h = max(1e-3 * r, 1e-5)
        J = np.zeros((2, 3))
        for k in range(3):
            e = np.zeros(3)
            e[k] = h
            up, _ = cam.project((P + e)[None])
            um, _ = cam.project((P - e)[None])
            J[:, k] = (up[0] - um[0]) / (2 * h)
        S2 = J @ cov_c[i] @ J.T + lowpass * np.eye(2)
        a, b, c = S2[0, 0], S2[0, 1], S2[1, 1]
        det = max(a * c - b * b, 1e-12)
        lam = 0.5 * (a + c) + math.sqrt(max(0.25 * (a + c) ** 2 - det, 0.0))
        o = opacity[i]
        m = min(math.sqrt(max(2.0 * math.log(max(o * 255.0, 1.0 + 1e-6)), 0.0)), MAX_SIGMA)
        rad = math.ceil(m * math.sqrt(lam))
        u0, v0 = uv[i]
        if not (np.isfinite(rad) and rad < 4 * max(W, H)):
            continue
        x0, x1 = max(int(math.floor(u0 - rad)), 0), min(int(math.ceil(u0 + rad)), W - 1)
        y0, y1 = max(int(math.floor(v0 - rad)), 0), min(int(math.ceil(v0 + rad)), H - 1)
        if x1 < x0 or y1 < y0:
            continue
        yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1].astype(float)
        dx, dy = xx - u0, yy - v0
        q = (c * dx * dx - 2 * b * dx * dy + a * dy * dy) / det             # dᵀ Σ₂⁻¹ d
        al = np.minimum(o * np.exp(-0.5 * q), ALPHA_MAX)
        al = np.where((q <= m * m) & (al >= ALPHA_MIN), al, 0.0)
        Ts = T[y0:y1 + 1, x0:x1 + 1]
        w = al * Ts
        img[y0:y1 + 1, x0:x1 + 1] += w[..., None] * feats[i]
        if want_index:
            bw = best_w[y0:y1 + 1, x0:x1 + 1]
            better = w > bw
            bw[better] = w[better]
            best[y0:y1 + 1, x0:x1 + 1][better] = i
        T[y0:y1 + 1, x0:x1 + 1] = Ts * (1.0 - al)
    bgf = np.zeros((H * W, F)) if bg is None else np.asarray(bg, float).reshape(H * W, F)
    out = {'image': img.reshape(-1, F) + T.reshape(-1, 1) * bgf, 'alpha': 1.0 - T.reshape(-1)}
    if want_index:
        out['index'] = best.reshape(-1)
    return out


# =========================================================================== the gsplat back end
def splat_gsplat(means, quats_xyzw, scales, opacity, colours, cam, T_world_optical, bg=None, near: float = 0.01,
                 lowpass: float = LOWPASS) -> dict:
    """gsplat.rasterization of world-frame Gaussians (torch CUDA tensors) through a pinhole CameraModel at
    T_world_optical (4 × 4): {'image' (H, W, 3), 'alpha' (H, W)}. Raises ValueError for a model gsplat is not used
    for here (anything but pinhole), ImportError when gsplat is not installed."""
    import gsplat
    import torch
    if cam.model != 'pinhole':
        raise ValueError(f'the gsplat back end renders pinhole cameras only, not {cam.model}; use the torch back end')
    dev = means.device
    W2C = torch.tensor(np.linalg.inv(np.asarray(T_world_optical, float)), dtype=means.dtype, device=dev)
    K = torch.tensor([[cam.fx, 0.0, cam.cx + 0.5], [0.0, cam.fy, cam.cy + 0.5], [0.0, 0.0, 1.0]],
                     dtype=means.dtype, device=dev)                      # gsplat: pixel centres at i + ½
    q = quats_xyzw[:, [3, 0, 1, 2]]                                     # gsplat wants (w, x, y, z)
    bgt = None if bg is None else torch.as_tensor(bg, dtype=means.dtype, device=dev).reshape(1, -1)
    img, alpha, _meta = gsplat.rasterization(means, q, scales, opacity, colours, W2C[None], K[None], cam.width,
                                             cam.height, near_plane=near, eps2d=lowpass, backgrounds=bgt,
                                             rasterize_mode='classic', packed=False)
    # packed=False: gsplat 1.5.3's packed path checks a per-camera background against an empty batch shape and
    # refuses it; the unpacked path takes backgrounds of shape (cameras, channels) as documented.
    return {'image': img[0], 'alpha': alpha[0, ..., 0]}


# =========================================================================== image quality
def ssim_map(a, b):
    """SSIM (Wang, Bovik, Sheikh and Simoncelli 2004, key wang2004_ssim: 11 × 11 Gaussian window, σ 1.5, K1 0.01,
    K2 0.03, values in [0, 1], replicated borders) of two (H, W, C) torch images: the per-pixel map averaged over
    channels, (H, W)."""
    import torch
    import torch.nn.functional as Fn
    x = a.permute(2, 0, 1)[None]
    y = b.permute(2, 0, 1)[None]
    C = x.shape[1]
    g = torch.exp(-(torch.arange(11, device=a.device, dtype=a.dtype) - 5) ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None].repeat(C, 1, 1, 1)

    def f(z):
        return Fn.conv2d(Fn.pad(z, (5, 5, 5, 5), mode='replicate'), w, groups=C)
    mx, my = f(x), f(y)
    sxx = f(x * x) - mx * mx
    syy = f(y * y) - my * my
    sxy = f(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return s[0].mean(0)


def ssim_np(a, b, mask=None) -> float | None:
    """Mean SSIM of two (H, W, C) images in [0, 1] (numpy and scipy.ndimage, written again: the same window, constants
    and borders as ssim_map) over the mask's pixels (all when None)."""
    from scipy.ndimage import gaussian_filter
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.ndim == 2:
        a, b = a[..., None], b[..., None]

    def f(z):
        return np.stack([gaussian_filter(z[..., k], 1.5, mode='nearest', truncate=5 / 1.5)
                         for k in range(z.shape[-1])], -1)
    mx, my = f(a), f(b)
    sxx = f(a * a) - mx * mx
    syy = f(b * b) - my * my
    sxy = f(a * b) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = (((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))).mean(-1)
    if mask is not None:
        s = s[np.asarray(mask, bool)]
    return float(s.mean()) if s.size else None


def psnr(a, b, mask=None) -> float | None:
    """PSNR in dB of two images in [0, 1] over the mask's pixels (all when None); 99 dB for identical images."""
    d = (np.asarray(a, float) - np.asarray(b, float)) ** 2
    if mask is not None:
        d = d[np.asarray(mask, bool)]
    if not d.size:
        return None
    mse = float(d.mean())
    return 99.0 if mse <= 1e-10 else float(10 * math.log10(1.0 / mse))
