"""The image, depth and distribution measures of the sim-to-real gap (core/sim2real/gap.py), on arrays alone: no
rendering, no dataset. Every function is deterministic for its inputs (the permutation test takes a seed).

    to_grey(img) -> (H, W) float32              ITU-R BT.601 luma (0.299 R + 0.587 G + 0.114 B), OpenCV's weights
    channel_histograms(imgs, masks=None) -> (3, 256) int64
    emd_1d(h_a, h_b) -> float                   Earth mover's distance (Wasserstein-1) of two 256-bin histograms, DN
    emd_channels(imgs_a, imgs_b) -> [r, g, b]
    photometric(imgs) -> {'mean_rgb', 'mean', 'contrast', 'saturated_pct', 'dark_pct'}
    noise_sigma(grey, edge_pct=90) -> float     Immerkær's estimator over the non-edge pixels, DN
    sharpness(grey, noise=None) -> float        variance of the Laplacian less its noise share, DN²
    blur(img, sigma_px) -> img                  Gaussian blur (σ in pixels; 0: the image)
    blur_to_match(sims, target) -> σ px         the Gaussian blur that brings the renders' sharpness to the target
    fit_affine(sims, reals) -> (gain (3,), offset (3,))   real ≈ gain · sim + offset per channel, on quantiles
    apply_affine(img, gain, offset) -> float32 image
    mad(a, b) -> float                          mean absolute difference over pixels and channels, DN
    iou(a, b) -> float | None
    features(img, size=(32, 24)) -> (d,)        a thumbnail, RGB in [0, 1], flattened (the MMD's feature)
    median_bandwidth(X, Y) -> σ                 the median heuristic: the median distance between pooled samples
    mmd2_unbiased(X, Y, sigma) -> float         Gretton et al. (2012) eq. (3), RBF kernel
    mmd_test(X, Y, sigma=None, permutations=200, seed=0) -> dict   MMD² with a permutation p-value
    mmd2_gaussians(mu1, S1, mu2, S2, sigma) -> float  the population MMD² of two Gaussians under the RBF kernel
    depth_gap(pred, meas, robot=None, z_min=None, z_max=None) -> dict
    shapley(players, value) -> {player: φ}      Shapley (1953) values of a cost function over subsets

Earth mover's distance. For histograms on the same unit-spaced bins (grey levels 0…255), the 1-D optimal-transport
distance is W₁ = Σ_k |F_a(k) − F_b(k)| (F the normalised cumulative histograms; Rubner et al. 2000 for the EMD, which
in one dimension is the Wasserstein-1 distance). It is in grey levels (DN): a pure brightness offset of b DN, with no
clipping, has W₁ = b. Checked against scipy.stats.wasserstein_distance (validation sim2real.emd_vs_scipy).

Noise. Immerkær (1996): σ = √(π/2) · mean |I ∗ N| / 6 with N = [[1, −2, 1], [−2, 4, −2], [1, −2, 1]], the difference
of two Laplacians, which cancels a plane and most smooth structure. The factors follow from the operator itself: on
white Gaussian noise of σ the response is Gaussian with σ_r = σ √(Σ N²) = 6σ, and E|r| = σ_r √(2/π). Edges still
leak into the response (the paper's own known weakness on texture), so pixels whose Sobel gradient magnitude (of the
image smoothed with σ = 1 px) is above its edge_pct percentile (default 90, a DexVantage default after Tai and Yang
2008's adaptive edge exclusion) are left out. Checked on images of known noise (validation sim2real.noise_known).

Sharpness. The variance of the 4-neighbour Laplacian (Pech-Pacheco et al. 2000, the measure the audit's blur test
uses), less 20 σ² (the Laplacian kernel's Σ k² = 20 times the noise variance σ² that white noise adds to it), so
two images of the same detail but different noise read alike; clipped at 0.

Kernel MMD. With k(x, y) = exp(−‖x − y‖² / (2σ²)), the unbiased estimate of the squared maximum mean discrepancy is
MMD²_u = Σ_{i≠j} k(x_i, x_j) / (m(m−1)) + Σ_{i≠j} k(y_i, y_j) / (n(n−1)) − 2 Σ_{i,j} k(x_i, y_j) / (mn)
(Gretton et al. 2012, Lemma 6, eq. 3); it can be negative when the two samples are alike. σ by the median heuristic
(Gretton et al. 2012, section 8) unless given. The p-value is the share of label permutations (seeded) whose MMD²_u
is at least the observed one, (1 + #) / (1 + permutations). It is a distance between small sets of downsampled
frames: NOT the Fréchet Inception Distance (FID), which needs a trained network's features and many more images.
For two Gaussians N(μ₁, Σ₁), N(μ₂, Σ₂) the population value is E k(X, X') + E k(Y, Y') − 2 E k(X, Y), with
E exp(−‖Z‖²/(2σ²)) = det(I + S/σ²)^(−½) exp(−½ mᵀ (S + σ² I)⁻¹ m) for Z ~ N(m, S) (validation sim2real.mmd_gaussians
writes the expectation again by eigenvalues).

Shapley values. For a cost v(S) over the subsets S of the players (here the image difference left after applying the
corrections in S), φ_i = Σ_{S ⊆ P∖{i}} |S|! (n − |S| − 1)! / n! · (v(S) − v(S ∪ {i})): what correction i removes,
averaged over every order the corrections could be applied in (Shapley 1953). Σ φ_i = v(∅) − v(P) exactly.

Known limits: the noise estimate is of the image as stored (after the codec, which removes noise as it compresses);
Immerkær's operator reads fine texture as noise where the edge exclusion misses it; the sharpness depends on the
scene's content, so it is only compared between images of the same scene (a recording and its render); the affine fit
is global per channel (no vignetting, no tone curve); MMD on a handful of frames has little power (the p-value says
so).
"""
from __future__ import annotations

import itertools
import math

import numpy as np

LUMA = (0.299, 0.587, 0.114)            # ITU-R BT.601, OpenCV's RGB2GRAY weights
IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
LAPLACE_NOISE_GAIN = 20.0               # Σ k² of the 4-neighbour Laplacian [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
EDGE_PCT = 90.0                         # default: pixels above this gradient percentile are left out of the noise
SAT_HI, SAT_LO = 254, 1                 # a pixel at or beyond these grey levels is clipped (affine fit, shares)
MAX_BLUR_PX = 8.0                       # blur_to_match searches σ up to this (px)


def to_grey(img) -> np.ndarray:
    a = np.asarray(img, np.float32)
    if a.ndim == 2:
        return a
    return a[..., 0] * LUMA[0] + a[..., 1] * LUMA[1] + a[..., 2] * LUMA[2]


# ====================================================================== histograms and EMD
def channel_histograms(imgs, masks=None) -> np.ndarray:
    """(3, 256) pixel counts of the frames' R, G, B values (uint8 images; masks: per frame, the pixels counted)."""
    h = np.zeros((3, 256), np.int64)
    for k, img in enumerate(imgs):
        a = np.asarray(img)
        if a.dtype != np.uint8:
            a = np.clip(np.rint(a), 0, 255).astype(np.uint8)
        m = None if masks is None else np.asarray(masks[k], bool)
        for c in range(3):
            v = a[..., c] if a.ndim == 3 else a
            h[c] += np.bincount((v[m] if m is not None else v).ravel(), minlength=256)[:256]
    return h


def emd_1d(h_a, h_b) -> float | None:
    """Wasserstein-1 distance (DN) of two histograms on bins 0, 1, …, len − 1 (module docstring); None when either
    is empty."""
    a = np.asarray(h_a, float)
    b = np.asarray(h_b, float)
    if a.sum() <= 0 or b.sum() <= 0:
        return None
    Fa = np.cumsum(a / a.sum())
    Fb = np.cumsum(b / b.sum())
    return float(np.sum(np.abs(Fa - Fb)[:-1]))


def emd_channels(imgs_a, imgs_b, masks_a=None, masks_b=None) -> list:
    ha, hb = channel_histograms(imgs_a, masks_a), channel_histograms(imgs_b, masks_b)
    return [emd_1d(ha[c], hb[c]) for c in range(3)]


def photometric(imgs) -> dict:
    """Mean per channel and of the grey image, RMS contrast (standard deviation of the grey image) and the clipped
    shares, over the frames (DN)."""
    st = np.stack([np.asarray(i, np.float32) for i in imgs])
    g = to_grey(st.reshape(-1, st.shape[-2], 3)) if st.ndim == 4 else st
    return {'mean_rgb': [float(st[..., c].mean()) for c in range(3)], 'mean': float(g.mean()),
            'contrast': float(g.std()), 'saturated_pct': float(100.0 * np.mean(np.max(st, axis=-1) >= SAT_HI)),
            'dark_pct': float(100.0 * np.mean(np.max(st, axis=-1) <= SAT_LO))}


# ====================================================================== noise and sharpness
def noise_sigma(grey, edge_pct: float | None = EDGE_PCT) -> float:
    """Immerkær's noise σ (DN) over the pixels that are not edges (module docstring)."""
    import cv2
    g = np.asarray(grey, np.float32)
    if g.ndim == 3:
        g = to_grey(g)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return float('nan')
    r = np.abs(cv2.filter2D(g, cv2.CV_32F, IMMERKAER, borderType=cv2.BORDER_REPLICATE))[1:-1, 1:-1]
    if edge_pct is not None and edge_pct < 100:
        s = cv2.GaussianBlur(g, (0, 0), 1.0)
        gx = cv2.Sobel(s, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(s, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)[1:-1, 1:-1]
        keep = mag <= np.percentile(mag, edge_pct)
        if keep.sum() >= 16:
            r = r[keep]
    return float(math.sqrt(math.pi / 2.0) * float(np.mean(r)) / 6.0)


def sharpness(grey, noise: float | None = None) -> float:
    """Variance of the 4-neighbour Laplacian less 20 noise² (DN², module docstring)."""
    import cv2
    g = np.asarray(grey, np.float32)
    if g.ndim == 3:
        g = to_grey(g)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=1)[1:-1, 1:-1]
    v = float(lap.var())
    if noise is not None and math.isfinite(noise):
        v -= LAPLACE_NOISE_GAIN * float(noise) ** 2
    return max(0.0, v)


def blur(img, sigma_px: float) -> np.ndarray:
    import cv2
    a = np.asarray(img, np.float32)
    if not sigma_px or sigma_px <= 0:
        return a
    return cv2.GaussianBlur(a, (0, 0), float(sigma_px), borderType=cv2.BORDER_REPLICATE)


def blur_to_match(sims, target: float, max_px: float = MAX_BLUR_PX) -> float:
    """σ (px) of the Gaussian blur after which the renders' summed sharpness equals target (the recordings' summed
    sharpness), by bisection (sharpness falls monotonically with σ); 0 when the renders are already no sharper,
    max_px when even that is not enough."""
    greys = [to_grey(np.asarray(s, np.float32)) for s in sims]

    def f(s):
        return sum(sharpness(blur(g, s)) for g in greys)
    if f(0.0) <= target:
        return 0.0
    if f(max_px) > target:
        return float(max_px)
    lo, hi = 0.0, float(max_px)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if f(mid) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-3:
            break
    return 0.5 * (lo + hi)


# ====================================================================== photometric fit
def fit_affine(sims, reals, masks=None, passes: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Per channel gain and offset with real ≈ gain · sim + offset, fitted by least squares on the two images'
    quantiles (the 2nd to the 98th percentile in 2 % steps), the pairs where neither quantile is clipped: the affine
    map that best lines up the two histograms (clipping is monotone, so quantiles pass through it). Matching quantiles
    rather than pixels keeps the fit unbiased when the images are not aligned or not equally sharp (a pixel-wise fit
    shrinks the gain towards 0 then: regression dilution). Gain 1 and the median difference when the render has no
    spread. The fit is repeated on the render as corrected and clipped (passes, default 3), composing the maps,
    so the pixels the correction itself clips are handled as the recording's are."""
    S = np.stack([np.asarray(s, np.float32) for s in sims])
    R = np.stack([np.asarray(r, np.float32) for r in reals])
    M = None if masks is None else np.stack([np.asarray(m, bool) for m in masks])
    g, o = np.ones(3), np.zeros(3)
    for _ in range(max(1, int(passes))):
        cur = np.clip(S * g.astype(np.float32) + o.astype(np.float32), 0.0, 255.0)
        g1, o1 = _fit_quantiles(cur, R, M)
        g, o = g1 * g, g1 * o + o1
        if np.all(np.abs(g1 - 1) < 1e-4) and np.all(np.abs(o1) < 1e-2):
            break
    return g, o


def _fit_quantiles(S, R, M) -> tuple[np.ndarray, np.ndarray]:
    """One quantile fit (fit_affine)."""
    g, o = np.ones(3), np.zeros(3)
    pct = np.linspace(2.0, 98.0, 49)
    for c in range(3):
        x, y = S[..., c], R[..., c]
        xv, yv = (x[M], y[M]) if M is not None and M.sum() >= 16 else (x.ravel(), y.ravel())
        qx, qy = np.percentile(xv, pct), np.percentile(yv, pct)
        ok = (qx > SAT_LO) & (qx < SAT_HI) & (qy > SAT_LO) & (qy < SAT_HI)
        qx, qy = qx[ok], qy[ok]
        vx = float(np.var(qx)) if len(qx) >= 3 else 0.0
        if vx < 1e-6:
            g[c], o[c] = 1.0, float(np.median(yv) - np.median(xv))
            continue
        g[c] = float(np.mean((qx - qx.mean()) * (qy - qy.mean())) / vx)
        o[c] = float(qy.mean() - g[c] * qx.mean())
    return g, o


def apply_affine(img, gain, offset) -> np.ndarray:
    a = np.asarray(img, np.float32)
    return np.clip(a * np.asarray(gain, np.float32) + np.asarray(offset, np.float32), 0.0, 255.0)


def mad(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32))))


def iou(a, b) -> float | None:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    u = np.logical_or(a, b).sum()
    return None if u == 0 else float(np.logical_and(a, b).sum() / u)


# ====================================================================== kernel MMD
def features(img, size=(32, 24)) -> np.ndarray:
    import cv2
    a = np.asarray(img, np.float32)
    t = cv2.resize(a, (int(size[0]), int(size[1])), interpolation=cv2.INTER_AREA)
    return (t / 255.0).ravel().astype(np.float64)


def _sqdist(X, Y) -> np.ndarray:
    X, Y = np.atleast_2d(np.asarray(X, float)), np.atleast_2d(np.asarray(Y, float))
    d = (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * X @ Y.T
    return np.maximum(d, 0.0)


def median_bandwidth(X, Y) -> float:
    Z = np.vstack([np.atleast_2d(np.asarray(X, float)), np.atleast_2d(np.asarray(Y, float))])
    d = np.sqrt(_sqdist(Z, Z))[np.triu_indices(len(Z), 1)]
    d = d[d > 0]
    return float(np.median(d)) if len(d) else 1.0


def _mmd2_from(K, m) -> float:
    n = len(K) - m
    Kxx, Kyy, Kxy = K[:m, :m], K[m:, m:], K[:m, m:]
    sxx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    syy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
    return float(sxx + syy - 2.0 * Kxy.sum() / (m * n))


def mmd2_unbiased(X, Y, sigma: float) -> float:
    """Gretton et al. (2012) eq. (3) with the RBF kernel of bandwidth sigma (module docstring); needs m, n ≥ 2."""
    X, Y = np.atleast_2d(np.asarray(X, float)), np.atleast_2d(np.asarray(Y, float))
    if len(X) < 2 or len(Y) < 2:
        raise ValueError('The unbiased MMD needs at least two samples on each side.')
    Z = np.vstack([X, Y])
    K = np.exp(-_sqdist(Z, Z) / (2.0 * float(sigma) ** 2))
    return _mmd2_from(K, len(X))


def mmd_test(X, Y, sigma: float | None = None, permutations: int = 200, seed: int = 0) -> dict:
    """MMD²_u, its bandwidth and a permutation p-value (module docstring)."""
    X, Y = np.atleast_2d(np.asarray(X, float)), np.atleast_2d(np.asarray(Y, float))
    m, n = len(X), len(Y)
    if m < 2 or n < 2:
        return {'mmd2': None, 'sigma': None, 'p_value': None, 'm': m, 'n': n,
                'why': 'at least two frames on each side are needed'}
    s = float(sigma) if sigma else median_bandwidth(X, Y)
    Z = np.vstack([X, Y])
    K = np.exp(-_sqdist(Z, Z) / (2.0 * s * s))
    obs = _mmd2_from(K, m)
    rng = np.random.default_rng(seed)
    hits = 0
    P = max(0, int(permutations))
    for _ in range(P):
        p = rng.permutation(m + n)
        if _mmd2_from(K[np.ix_(p, p)], m) >= obs:
            hits += 1
    return {'mmd2': obs, 'mmd': float(math.sqrt(max(obs, 0.0))), 'sigma': s,
            'p_value': (1 + hits) / (1 + P) if P else None, 'permutations': P, 'm': m, 'n': n,
            'kernel': 'RBF exp(-|x - y|^2 / (2 sigma^2)), sigma by the median heuristic' if not sigma else
                      'RBF exp(-|x - y|^2 / (2 sigma^2)), sigma given'}


def _gauss_rbf_mean(m, S, sigma) -> float:
    d = len(m)
    A = np.eye(d) + np.asarray(S, float) / sigma ** 2
    sign, logdet = np.linalg.slogdet(A)
    q = float(m @ np.linalg.solve(np.asarray(S, float) + sigma ** 2 * np.eye(d), m))
    return float(math.exp(-0.5 * logdet - 0.5 * q))


def mmd2_gaussians(mu1, S1, mu2, S2, sigma: float) -> float:
    """The population MMD² of N(mu1, S1) and N(mu2, S2) under the RBF kernel (module docstring)."""
    mu1, mu2 = np.asarray(mu1, float), np.asarray(mu2, float)
    S1, S2 = np.asarray(S1, float), np.asarray(S2, float)
    z = np.zeros_like(mu1)
    return (_gauss_rbf_mean(z, 2 * S1, sigma) + _gauss_rbf_mean(z, 2 * S2, sigma)
            - 2.0 * _gauss_rbf_mean(mu1 - mu2, S1 + S2, sigma))


# ====================================================================== depth
def depth_gap(pred, meas, robot=None, z_min: float | None = None, z_max: float | None = None) -> dict:
    """Valid-pixel shares of a predicted (rendered) and a measured depth image (m; 0, NaN or out of [z_min, z_max]:
    no depth) and, on the robot's pixels where both are valid, the depth error measured − predicted (mm)."""
    P = np.asarray(pred, float)
    Mz = np.asarray(meas, float)
    if P.shape != Mz.shape:
        raise ValueError(f'The predicted depth is {P.shape[1]} × {P.shape[0]} and the measured '
                         f'{Mz.shape[1]} × {Mz.shape[0]}: they must be the same size.')

    def valid(D):
        ok = np.isfinite(D) & (D > 0)
        if z_min is not None:
            ok &= D >= z_min
        if z_max is not None:
            ok &= D <= z_max
        return ok
    vp, vm = valid(P), valid(Mz)
    out = {'predicted_valid_pct': float(100 * vp.mean()), 'measured_valid_pct': float(100 * vm.mean()),
           'valid_gap_pct': float(100 * (vm.mean() - vp.mean())),
           'agree_pct': float(100 * np.mean(vp == vm)), 'z_min_m': z_min, 'z_max_m': z_max}
    if robot is not None:
        both = vp & vm & np.asarray(robot, bool)
        out['robot_px'] = int(both.sum())
        if both.any():
            e = (Mz[both] - P[both]) * 1000.0
            out['robot_error_mm'] = {'mean': float(e.mean()), 'median': float(np.median(e)),
                                     'rms': float(np.sqrt(np.mean(e * e))),
                                     'p95_abs': float(np.percentile(np.abs(e), 95))}
        else:
            out['robot_error_mm'] = None
    return out


# ====================================================================== attribution
def shapley(players, value) -> dict:
    """{player: φ}: the Shapley values of the reduction of the cost value(frozenset) (module docstring)."""
    P = list(players)
    n = len(P)
    cache = {}

    def v(S):
        k = frozenset(S)
        if k not in cache:
            cache[k] = float(value(k))
        return cache[k]
    out = {}
    for i in P:
        rest = [p for p in P if p != i]
        phi = 0.0
        for r in range(len(rest) + 1):
            w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for S in itertools.combinations(rest, r):
                phi += w * (v(S) - v(set(S) | {i}))
        out[i] = phi
    return out
