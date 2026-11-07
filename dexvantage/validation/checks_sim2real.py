"""Checks of the sim-to-real gap (core/sim2real) against independent references.

- sim2real.emd_vs_scipy: the Earth mover's distance of 256-bin histograms (measures.emd_1d) against
  scipy.stats.wasserstein_distance on the same bins and weights, 200 random pairs.
- sim2real.mmd_hand_case: the unbiased MMD² (measures.mmd2_unbiased) of X = {0, 1}, Y = {2, 4}, σ = 1, against its
  terms written out by hand: e^(−1/2) + e^(−2) − (e^(−2) + e^(−8) + e^(−1/2) + e^(−9/2)) / 2.
- sim2real.mmd_gaussians: the mean of 400 unbiased estimates (m = n = 40, 4-D Gaussians of different means and
  covariances, RBF σ = 1.5) against the population MMD² in closed form, written again here by the eigenvalues of the
  covariances (Π (1 + λ_k/σ²)^(−½) and the quadratic form in the eigenbasis); the estimator is unbiased, so the
  tolerance is 4 standard errors of the mean of the estimates. The module's own closed form must agree with this one
  to 1e-10.
- sim2real.noise_known: Immerkær's estimate (measures.noise_sigma, edge pixels left out) on a structured image
  (sinusoids, a ramp and a step) with Gaussian noise of σ = 0.5 to 8 DN added (numpy's generator), against σ.
- sim2real.blur_known: the Gaussian σ that blur_to_match finds (its blur is OpenCV's) on a textured image blurred by
  scipy.ndimage.gaussian_filter with σ = 0.8, 1.5 and 3 px (an independent implementation), against σ.
- sim2real.shapley_brute_force: the Shapley values (measures.shapley, the subset formula) of a random cost over four
  players against the average marginal reduction over all 24 orders of the players.
- sim2real.injected_recovery: synthetic LeRobot recordings of the UR5e sample (core/sim2real/inject.py: the synthetic
  episode generator, one episode of 40 rows at 320 × 180 px, the video codec) with ONE known gap each on the overhead
  camera: a brightness offset of 20 DN, a Gaussian blur of σ 1.5 px, a latency of 40 ms, a mount offset of 3 mm and 1°;
  each recovered by the study's analysis (core/sim2real/gap.analyse) and ranked first among the camera's causes. The
  error is the largest recovery error over the four as a share of its own tolerance (the brightness change at mid
  grey, gain · 128 + offset − 128, averaged over the channels, within 3 DN: the clean render
  and recording differ by up to 1 DN per channel, and clipping at 255 moves the fit; blur 0.3 px: the codec blurs too,
  measured 1.39–1.42 px recovered for 1.5; latency 5 ms: half the 10 ms grid step, measured 0.03 ms; mount 0.3° and
  3 mm: the silhouette calibration's accuracy on this camera, measured about 0.1° and 2 mm). The 3 mm is about 2 px
  of the 320 × 180 working image at the robot's depth (docs/models/render_calib.md: 1.1–1.8 mm per 2 px at
  640 × 360), and it equals the injected 3 mm, so the position part alone would pass with no offset found: the angle
  part (0.3° of an injected 1°) is what shows the mount was recovered; the 3 mm translation itself is not resolved
  at this image size (2 mm off).
"""
from __future__ import annotations

import math

import numpy as np

from .suite import Outcome, check


@check('sim2real.emd_vs_scipy', 'sim2real', 'Histogram EMD against SciPy',
       'largest |EMD − SciPy\'s Wasserstein-1| over 200 random histogram pairs',
       'scipy.stats.wasserstein_distance on the bin values 0…255 with the histograms as weights', 1e-9, 'DN',
       refs=('rubner2000', 'scipy_wasserstein'))
def _emd(ctx):
    from scipy.stats import wasserstein_distance
    from ..core.sim2real import measures as MS
    rng = ctx.rng('sim2real.emd_vs_scipy')
    v = np.arange(256, dtype=float)
    err = 0.0
    for k in range(200):
        a = rng.integers(0, 60, 256) * (rng.random(256) < 0.6)
        b = rng.integers(0, 60, 256) * (rng.random(256) < 0.6)
        a[k % 256] += 1
        b[(k * 7) % 256] += 1
        err = max(err, abs(MS.emd_1d(a, b) - wasserstein_distance(v, v, a, b)))
    return Outcome(err, {'pairs': 200, 'bins': 256})


@check('sim2real.mmd_hand_case', 'sim2real', 'Unbiased MMD² on a case worked by hand',
       '|MMD²_u − its terms written out| for X = {0, 1}, Y = {2, 4}, σ = 1', 'the estimator\'s sums by hand',
       1e-12, 'MMD² (unitless)', refs=('gretton2012',))
def _mmd_hand(ctx):
    from ..core.sim2real import measures as MS
    e = math.exp
    ref = e(-0.5) + e(-2.0) - (e(-2.0) + e(-8.0) + e(-0.5) + e(-4.5)) / 2.0
    got = MS.mmd2_unbiased([[0.0], [1.0]], [[2.0], [4.0]], 1.0)
    return Outcome(abs(got - ref), {'computed': got, 'by_hand': ref})


def _gauss_term(m, S, sigma):
    lam, V = np.linalg.eigh(np.asarray(S, float))
    y = V.T @ np.asarray(m, float)
    return float(np.prod((1.0 + lam / sigma ** 2) ** -0.5) * math.exp(-0.5 * float(np.sum(y * y / (lam + sigma ** 2)))))


@check('sim2real.mmd_gaussians', 'sim2real', 'Unbiased MMD² against the closed form for two Gaussians',
       '|mean of 400 unbiased estimates − the population MMD²| (4-D Gaussians, RBF σ = 1.5)',
       'the closed form E k(X,X\') + E k(Y,Y\') − 2 E k(X,Y), by eigenvalues; tolerance 4 standard errors', 0.0,
       'MMD² (unitless)', refs=('gretton2012',))
def _mmd_gauss(ctx):
    from ..core.sim2real import measures as MS
    rng = ctx.rng('sim2real.mmd_gaussians')
    d, sig = 4, 1.5
    mu1, mu2 = np.zeros(d), np.array([0.6, -0.3, 0.0, 0.2])
    A = rng.normal(size=(d, d))
    S1 = 0.4 * np.eye(d)
    S2 = 0.2 * np.eye(d) + 0.1 * A @ A.T
    z = np.zeros(d)
    ref = _gauss_term(z, 2 * S1, sig) + _gauss_term(z, 2 * S2, sig) - 2 * _gauss_term(mu1 - mu2, S1 + S2, sig)
    mine = MS.mmd2_gaussians(mu1, S1, mu2, S2, sig)
    L1, L2 = np.linalg.cholesky(S1), np.linalg.cholesky(S2)
    R = 200 if ctx.quick else 400
    est = []
    for _ in range(R):
        X = mu1 + rng.normal(size=(40, d)) @ L1.T
        Y = mu2 + rng.normal(size=(40, d)) @ L2.T
        est.append(MS.mmd2_unbiased(X, Y, sig))
    est = np.asarray(est)
    se = float(est.std(ddof=1) / math.sqrt(R))
    return Outcome(abs(float(est.mean()) - ref), {'closed_form': ref, 'module_closed_form': mine,
                                                  'mean_estimate': float(est.mean()), 'standard_error': se,
                                                  'repeats': R, 'm': 40, 'n': 40},
                   ok=abs(mine - ref) <= 1e-10, tolerance=4.0 * se)


@check('sim2real.noise_known', 'sim2real', 'Noise σ (Immerkær) on images of known noise',
       'largest |σ̂ − σ| / σ over σ = 0.5, 1, 2, 4 and 8 DN on a structured image',
       'the σ of the Gaussian noise added (numpy)', 0.05, 'relative', refs=('immerkaer1996', 'tai_yang2008'))
def _noise(ctx):
    from ..core.sim2real import measures as MS
    rng = ctx.rng('sim2real.noise_known')
    y, x = np.mgrid[0:240, 0:320].astype(np.float32)
    base = 100 + 0.3 * x + 40 * np.sin(x / 20.0) * np.cos(y / 25.0)
    base[60:120, 80:200] += 60
    rows, err = [], 0.0
    for s in (0.5, 1.0, 2.0, 4.0, 8.0):
        est = MS.noise_sigma(base + rng.normal(0, s, base.shape).astype(np.float32))
        rows.append({'sigma': s, 'estimate': est})
        err = max(err, abs(est - s) / s)
    return Outcome(err, {'cases': rows, 'note': 'edge pixels (Sobel magnitude above its 90th percentile) left out; '
                                                'Immerkær\'s operator reads residual structure as noise, so the '
                                                'tolerance allows 5 %'})


@check('sim2real.blur_known', 'sim2real', 'Blur σ that matches the sharpness, on images of known blur',
       'largest |σ̂ − σ| over σ = 0.8, 1.5 and 3 px', 'scipy.ndimage.gaussian_filter with the σ given (an independent '
       'Gaussian blur)', 0.1, 'px', refs=('pech_pacheco2000', 'scipy_stats'))
def _blur(ctx):
    import cv2
    from scipy.ndimage import gaussian_filter
    from ..core.sim2real import measures as MS
    rng = ctx.rng('sim2real.blur_known')
    tex = cv2.resize(rng.uniform(0, 255, (60, 80)).astype(np.float32), (320, 240), interpolation=cv2.INTER_NEAREST)
    rows, err = [], 0.0
    for s in (0.8, 1.5, 3.0):
        b = gaussian_filter(tex, s, mode='nearest', truncate=4.0)
        est = MS.blur_to_match([tex], MS.sharpness(b))
        rows.append({'sigma': s, 'estimate': est})
        err = max(err, abs(est - s))
    return Outcome(err, {'cases': rows, 'note': 'a blocky texture (8 px squares), no noise; the two libraries '
                                                'truncate their kernels differently'})


@check('sim2real.shapley_brute_force', 'sim2real', 'Shapley shares against every order of the corrections',
       'largest |φ_i − mean marginal reduction over the 24 orders| for a random cost of four players',
       'brute force over all orders', 1e-12, 'DN', refs=('shapley1953',))
def _shapley(ctx):
    import itertools
    from ..core.sim2real import measures as MS
    rng = ctx.rng('sim2real.shapley_brute_force')
    P = ['latency', 'mount', 'exposure', 'blur']
    cost = {frozenset(S): float(rng.uniform(0, 20)) for r in range(5) for S in itertools.combinations(P, r)}
    phi = MS.shapley(P, lambda S: cost[frozenset(S)])
    ref = {p: 0.0 for p in P}
    orders = list(itertools.permutations(P))
    for o in orders:
        cur = set()
        for p in o:
            ref[p] += (cost[frozenset(cur)] - cost[frozenset(cur | {p})]) / len(orders)
            cur.add(p)
    err = max(abs(phi[p] - ref[p]) for p in P)
    tot = abs(sum(phi.values()) - (cost[frozenset()] - cost[frozenset(P)]))
    return Outcome(err, {'shapley': phi, 'brute_force': ref, 'sum_minus_total': tot}, ok=tot < 1e-9)


CASES = (('brightness', {'brightness_dn': 20.0}, 'exposure'),
         ('blur', {'blur_px': 1.5}, 'blur'),
         ('latency', {'latency_s': 0.04}, 'latency'),
         ('mount', {'mount': {'position_m': [0.003, 0.0, 0.0], 'rotation_deg': [0.0, 1.0, 0.0]}}, 'mount'))
TOL = {'brightness': 3.0, 'blur': 0.3, 'latency': 0.005, 'mount_deg': 0.3, 'mount_mm': 3.0}


def recovery_errors(cam: dict, case: str) -> dict:
    """{name: (error, tolerance)} of one case's recovered figure (the test file uses it too)."""
    ph = cam['photometric']
    if case == 'brightness':
        mid = float(np.mean(np.asarray(ph['gain']) * 128.0 + np.asarray(ph['offset_dn']) - 128.0))
        return {'brightness': (abs(mid - 20.0), TOL['brightness'])}
    if case == 'blur':
        return {'blur': (abs(ph['blur_px_full'] - 1.5), TOL['blur'])}
    if case == 'latency':
        lat = cam['latency']
        return {'latency': (abs(lat['latency_s'] - 0.04) if lat.get('observable') else math.inf, TOL['latency'])}
    m = cam['mount']
    if not m.get('observable'):
        return {'mount_deg': (math.inf, TOL['mount_deg']), 'mount_mm': (math.inf, TOL['mount_mm'])}
    r = np.asarray(m['delta']['rotation_deg']) - [0.0, 1.0, 0.0]
    p = (np.asarray(m['delta']['position_m']) - [0.003, 0.0, 0.0]) * 1000
    return {'mount_deg': (float(np.linalg.norm(r)), TOL['mount_deg']),
            'mount_mm': (float(np.linalg.norm(p)), TOL['mount_mm'])}


def top_factor(cam: dict):
    """The factor of the camera's cause that removes the most."""
    rows = [c for c in cam['causes'] if c.get('reduction_dn') is not None]
    return max(rows, key=lambda c: c['reduction_dn'])['factor'] if rows else None


@check('sim2real.injected_recovery', 'sim2real', 'Known gaps injected into synthetic recordings, recovered',
       'largest recovery error / its tolerance over a brightness offset, a blur, a latency and a mount offset, each '
       'alone on the UR5e sample\'s overhead camera; each must also rank first among the camera\'s causes',
       'the gaps injected (core/sim2real/inject.py through the synthetic episode generator)', 1.0, 'ratio',
       refs=('mujoco_todorov2012', 'lerobot_dataset_format', 'shapley1953'), cost='study')
def _recovery(ctx):
    import os
    from ..core.robot.samples import sample_project
    from ..core.sim2real import gap as G
    from ..core.sim2real import inject as IJ
    p = sample_project('ur5e')
    key = 'observation.images.overhead_d435'
    worst, rows, ranked = 0.0, [], True
    for name, gaps, factor in CASES:
        root = os.path.join(ctx.tmp(), f'sim2real_{name}')
        IJ.make_gap_dataset(p, root, dict(gaps, camera=key), {'episodes': 1, 'frames': 40, 'size': [320, 180],
                                                               'seed': 4})
        doc = G.analyse(p, root, {'cameras': ['Overhead D435'], 'mount': name == 'mount'})
        cam = doc['cameras'][0]
        errs = recovery_errors(cam, name)
        top = top_factor(cam)
        ranked &= top == factor
        for k, (e, t) in errs.items():
            worst = max(worst, e / t)
            rows.append({'case': name, 'figure': k, 'error': e, 'tolerance': t, 'top_cause': top})
    return Outcome(worst, {'cases': rows, 'project': 'the built-in UR5e sample (not the project given)',
                           'dataset': '1 episode, 40 rows, 320 × 180 px, LeRobot v3.0'}, ok=ranked)
