"""Checks of the Monte Carlo tolerances (core/tolerance.py) and the perturbation export (core/exporters/
perturbations.py) against independent references.

The rig they use (tol_rig, built here): three static sensors in the world, "Mount A" (± 2 mm, ± 1°), "Mount B"
(± 1 mm, ± 0.25°) and "Calibrated D435" (the library's RealSense D435, calibrated by a hand-eye session whose result
is written here with a correlated 6 × 6 covariance, σ about 1.5 mrad and 0.8 mm, so its RGB imager's offset from the
body matters).

- tolerance.linear_propagation: the Monte Carlo mean and σ of a linear metric (a weighted sum of the two mounts'
  positions, the calibrated imager's optical-frame position and its rotation vector, each read back from the moved
  project by code written out here, the rotation vector through SciPy) against the law of propagation of
  uncertainty, σ² = Σ (∂f/∂x)² σ²ₓ + the covariance terms (JCGM 100:2008, eq. 13): uniform ± b gives b² / 3, a
  normal with ± b at 2 σ truncated there σ² (1 − 2·2 φ(2) / (2 Φ(2) − 1)) (written out with math.erf), the calibrated
  sensor wᵀ C w. Both distributions are run. Error: |σ_MC / σ − 1| (the worst); tolerance 4 standard errors of a
  sample σ (√(2 / (4 n)) for a kurtosis up to 3), with the mean within 4 σ / √n as a further condition.
- tolerance.sampler_ks: the marginals of the draws, read back from the moved projects (positions, and the turns
  through SciPy's intrinsic "XYZ" Euler angles; the calibrated imager's rotation vector), against the stated
  distribution with the Kolmogorov-Smirnov test (scipy.stats.kstest against uniform, truncnorm and norm). Error: the
  largest KS statistic D; tolerance the critical D at α = 0.001 / 18 (Bonferroni over the 18 marginals),
  √(−ln(α / 2) / (2 n)).
- tolerance.seed_reproducible: two Monte Carlo runs with one seed give identical draws and values, another seed
  different ones, and the perturbation export's draws (a separate code path) are the Monte Carlo's for the same
  seed. Error: the largest difference; tolerance 0.
- tolerance.spearman: core.tolerance.spearman against scipy.stats.spearmanr on data with ties; tolerance 1e-12.
- tolerance.export_roundtrip: the exported JSON's Isaac Lab offsets (pos, rot_wxyz read with SciPy scalar_first) and
  the .npz's T_parent_optical against the optical frame of the project moved by each draw's components (core.factors,
  composed here with the imager's offset): error the largest position (m) and rotation (rad) difference; tolerance
  1e-8 (the JSON's 9 decimals).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from .suite import Outcome, check, figure, series

N_FULL, N_QUICK = 20000, 3000
KS_FULL, KS_QUICK = 4000, 1500
TRUNC_K = 2.0


def tol_rig(rng=None):
    """The checks' rig (module docstring): (project, {'A', 'B', 'C': sensor uid}, covariance of C)."""
    from ..core.calib import records as RC
    from ..core.frames import Pose, quat_from_rpy
    from ..core.project import Mount, Project, Sensor, Tolerance
    rng = rng if rng is not None else np.random.default_rng(7)
    p = Project('Tolerance checks (built in)')
    a = Sensor(name='Mount A', kind='camera', slot=0,
               mount=Mount('world', '', Pose(np.array([0.5, 0.1, 0.6]), quat_from_rpy(0.1, 0.6, -2.8)),
                           Tolerance(2.0, 1.0)))
    b = Sensor(name='Mount B', kind='camera', slot=1,
               mount=Mount('world', '', Pose(np.array([0.2, -0.5, 0.4]), quat_from_rpy(-0.2, 0.3, 1.4)),
                           Tolerance(1.0, 0.25)))
    c = Sensor(name='Calibrated D435', kind='depth', slot=2, model_id='intel-realsense-d435',
               mount=Mount('world', '', Pose(np.array([0.6, 0.4, 0.5]), quat_from_rpy(0.05, 0.5, -2.2)),
                           Tolerance(5.0, 2.0)))
    p.variant.sensors[:] = [a, b, c]
    T_im = RC.sensor_imager(c).pose_in_body.matrix()
    X = c.mount.pose.matrix() @ T_im
    s = np.array([1.5e-3, 1.2e-3, 2.0e-3, 0.8e-3, 0.6e-3, 1.0e-3])
    Rm = rng.uniform(-0.6, 0.6, (6, 6))
    Rm = Rm @ Rm.T + 6 * np.eye(6)
    d = np.sqrt(np.diag(Rm))
    corr = Rm / np.outer(d, d)
    C = corr * np.outer(s, s)
    p.calibration['sessions'].append({
        'uid': 'tolsession', 'name': 'check session', 'sensor_uid': c.uid, 'mode': 'eye_to_hand', 'poses': [],
        'result': {'ok': True, 'applied': True, 'applied_at': '2026-11-07T00:00:00Z',
                   'covariance_order': ['rx_rad', 'ry_rad', 'rz_rad', 'x_m', 'y_m', 'z_m'],
                   'cameras': {c.name: {'matrix': X.tolist(), 'covariance': C.tolist()}}}})
    return p, {'A': a.uid, 'B': b.uid, 'C': c.uid}, C


def _optical(project, uid, T_im=None):
    """The sensor's calibration imager's optical frame in its mount frame (T_im: the imager in the body, when known)."""
    from ..core.calib import records as RC
    s = project.variant.sensor(uid)
    T_im = RC.sensor_imager(s).pose_in_body.matrix() if T_im is None else T_im
    return s.mount.pose.matrix() @ T_im


def _imager(project, uid):
    from ..core.calib import records as RC
    return RC.sensor_imager(project.variant.sensor(uid)).pose_in_body.matrix()


def _rotvec(R0, R1):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R0.T @ R1).as_rotvec()


def _trunc_var(b, k=TRUNC_K):
    """Variance of a normal with σ = b / k truncated at ± k σ, written out: σ² (1 − 2 k φ(k) / (2 Φ(k) − 1))."""
    sig = b / k
    phi = math.exp(-k * k / 2) / math.sqrt(2 * math.pi)
    big = 0.5 * (1 + math.erf(k / math.sqrt(2)))
    return sig * sig * (1 - 2 * k * phi / (2 * big - 1))


WA = np.array([0.7, -1.3, 0.4])          # weights on Mount A's position (per m)
WB = np.array([-0.5, 0.9, 2.0])
WC_V = np.array([1.1, 0.3, -0.8])        # on the calibrated imager's optical-frame position
WC_W = np.array([0.4, -0.2, 0.6])        # on its rotation vector (per rad)


def _linear_metric(uids, X0, T_im):
    def ev(pr, ids, _ctx):
        a = pr.variant.sensor(uids['A']).mount.pose.position
        b = pr.variant.sensor(uids['B']).mount.pose.position
        X = _optical(pr, uids['C'], T_im)
        w = _rotvec(X0[:3, :3], X[:3, :3])
        f = float(WA @ a + WB @ b + WC_V @ (X[:3, 3]) + WC_W @ w)
        return {'lin': {'value': f, 'unit': '', 'better': 'higher', 'title': 'Linear test metric', 'status': 'ok',
                        'study': ''}}
    return ev


@check('tolerance.linear_propagation', 'tolerance', 'Monte Carlo moments of a linear metric',
       'relative error of the Monte Carlo σ of a linear metric of three mounts', 'the law of propagation of '
       'uncertainty σ² = Σ (∂f/∂x)² σ²ₓ + covariance terms (JCGM 100:2008), written out here', 0.05, 'relative',
       refs=('jcgm100_2008', 'jcgm101_2008'), cost='study')
def _linear(ctx):
    from ..core import tolerance as TL
    p, uids, C = tol_rig()
    T_im = _imager(p, uids['C'])
    X0 = _optical(p, uids['C'], T_im)
    n = N_QUICK if ctx.quick else N_FULL
    wc = np.r_[WC_W, WC_V]                                # C's order: ω then v
    var_c = float(wc @ C @ wc)
    rows, worst, ok = [], 0.0, True
    nominal = _linear_metric(uids, X0, T_im)(p, ['lin'], None)['lin']['value']
    for dist in ('uniform', 'normal'):
        pa, pb = 2e-3, 1e-3
        va = (pa * pa / 3) if dist == 'uniform' else _trunc_var(pa)
        vb = (pb * pb / 3) if dist == 'uniform' else _trunc_var(pb)
        var = float(WA @ WA) * va + float(WB @ WB) * vb + var_c
        sd = math.sqrt(var)
        res = TL.monte_carlo(p, ['lin'], n, int(ctx.rng('tolerance.linear_propagation').integers(2 ** 31)),
                             distribution=dist, evaluator=_linear_metric(uids, X0, T_im))
        m = res['metrics']['lin']
        rel = abs(m['std'] / sd - 1)
        mean_z = abs(m['mean'] - nominal) / (sd / math.sqrt(n))
        ok &= mean_z <= 4
        worst = max(worst, rel)
        rows.append({'distribution': dist, 'sigma_analytic': sd, 'sigma_mc': m['std'], 'rel_error': rel,
                     'mean_offset_in_se': mean_z})
    tol = 4 * math.sqrt(2 / (4 * n))
    fig = figure('σ of the linear metric', 'distribution (0 uniform, 1 normal)', 'σ',
                 [series('propagation', [0, 1], [r['sigma_analytic'] for r in rows], 'points'),
                  series('Monte Carlo', [0, 1], [r['sigma_mc'] for r in rows], 'points')])
    return Outcome(worst, {'draws': n, 'runs': rows}, fig, ok=ok, tolerance=tol,
                   note=f'tolerance 4 standard errors of a sample σ at n = {n}')


@check('tolerance.sampler_ks', 'tolerance', 'Perturbation sampler marginals',
       'largest Kolmogorov-Smirnov statistic D of 18 marginals read back from the moved projects',
       'the stated distributions (uniform, truncated normal, the calibration normal) through scipy.stats.kstest',
       0.05, 'KS statistic D', refs=('kolmogorov1933', 'smirnov1948', 'scipy_stats', 'scipy_rotation'))
def _ks(ctx):
    from scipy import stats
    from scipy.spatial.transform import Rotation

    from ..core import tolerance as TL
    p, uids, C = tol_rig()
    n = KS_QUICK if ctx.quick else KS_FULL
    rng = ctx.rng('tolerance.sampler_ks')
    out, worst = [], 0.0
    T_im = _imager(p, uids['C'])
    X0 = _optical(p, uids['C'], T_im)
    cal = []
    for dist in ('uniform', 'normal'):
        samples = {k: [] for k in ('A', 'B')}
        for _ in range(n):
            q = TL.perturb(p, rng, distribution=dist)
            for k in ('A', 'B'):
                s0, s1 = p.variant.sensor(uids[k]).mount.pose, q.variant.sensor(uids[k]).mount.pose
                R0 = Rotation.from_quat(s0.quaternion)        # SciPy reads (x, y, z, w)
                R1 = Rotation.from_quat(s1.quaternion)
                ang = (R0.inv() * R1).as_euler('XYZ')           # intrinsic: Rx(a) Ry(b) Rz(c)
                samples[k].append(np.r_[s1.position - s0.position, np.degrees(ang)])
            if dist == 'uniform':
                X = _optical(q, uids['C'], T_im)
                cal.append(np.r_[_rotvec(X0[:3, :3], X[:3, :3]), X[:3, 3] - X0[:3, 3]])
        for k, (pb, rb) in (('A', (2e-3, 1.0)), ('B', (1e-3, 0.25))):
            S = np.array(samples[k])
            for j in range(6):
                b = pb if j < 3 else rb
                if dist == 'uniform':
                    D = stats.kstest(S[:, j], stats.uniform(loc=-b, scale=2 * b).cdf).statistic
                else:
                    D = stats.kstest(S[:, j], stats.truncnorm(-TRUNC_K, TRUNC_K, scale=b / TRUNC_K).cdf).statistic
                worst = max(worst, float(D))
                out.append({'sensor': k, 'component': TL.COMPONENTS[j], 'distribution': dist, 'D': float(D)})
    Cs = np.array(cal)
    for j in range(6):
        D = stats.kstest(Cs[:, j], stats.norm(scale=math.sqrt(C[j, j])).cdf).statistic
        worst = max(worst, float(D))
        out.append({'sensor': 'C', 'component': ['wx', 'wy', 'wz', 'vx', 'vy', 'vz'][j], 'distribution': 'calibration',
                    'D': float(D)})
    alpha = 0.001 / len(out)
    tol = math.sqrt(-math.log(alpha / 2) / (2 * n))
    cov_err = float(np.linalg.norm(np.cov(Cs.T) - C) / np.linalg.norm(C))
    fig = figure('KS statistic per marginal', 'marginal', 'D',
                 [series('D', np.arange(len(out)), [r['D'] for r in out], 'points')], hline=tol,
                 hline_label='critical D')
    return Outcome(worst, {'draws': n, 'marginals': out, 'calibration_covariance_rel_error': cov_err}, fig,
                   tolerance=tol, note=f'critical D at α = 0.001 over {len(out)} marginals (Bonferroni), n = {n}')


@check('tolerance.seed_reproducible', 'tolerance', 'Seed reproducibility',
       'largest difference between two runs with one seed, and between the Monte Carlo and the export draws',
       'the same run repeated; the export\'s separate draw loop', 0.0, 'm or rad', refs=())
def _seed(ctx):
    from ..core import tolerance as TL
    p, uids, _C = tol_rig()
    T_im = _imager(p, uids['C'])
    X0 = _optical(p, uids['C'], T_im)
    ev = _linear_metric(uids, X0, T_im)
    a = TL.monte_carlo(p, ['lin'], 50, 11, evaluator=ev)
    b = TL.monte_carlo(p, ['lin'], 50, 11, evaluator=ev)
    c = TL.monte_carlo(p, ['lin'], 50, 12, evaluator=ev)
    E = TL.perturbed_extrinsics(p, 50, 11)
    err = 0.0
    for uid in a['draws']:
        err = max(err, float(np.max(np.abs(np.array(a['draws'][uid]) - np.array(b['draws'][uid])))))
    err = max(err, float(np.max(np.abs(np.array(a['metrics']['lin']['values']) -
                                       np.array(b['metrics']['lin']['values'])))))
    order = [s['uid'] for s in E['sensors']]
    exp = np.stack([np.array(a['draws'][u]) for u in order], axis=1)
    err = max(err, float(np.max(np.abs(exp - E['d']))))
    differs = any(not np.allclose(a['draws'][u], c['draws'][u]) for u in a['draws'])
    return Outcome(err, {'same_seed_identical': err == 0.0, 'other_seed_differs': differs}, ok=differs)


@check('tolerance.spearman', 'tolerance', 'Rank correlation',
       'largest difference of core.tolerance.spearman from scipy.stats.spearmanr', 'scipy.stats.spearmanr', 1e-12,
       'rank correlation', refs=('spearman1904', 'scipy_stats'))
def _spearman(ctx):
    from scipy import stats

    from ..core import tolerance as TL
    rng = ctx.rng('tolerance.spearman')
    err = 0.0
    for k in range(20):
        n = int(rng.integers(5, 400))
        x = rng.normal(size=n)
        y = 0.3 * x + rng.normal(size=n)
        if k % 2:
            x = np.round(x, 1)                  # ties
            y = np.round(y, 0)
        r = TL.spearman(x, y)
        ref = stats.spearmanr(x, y).statistic
        err = max(err, abs(r - float(ref)))
    return Outcome(err, {'cases': 20})


@check('tolerance.export_roundtrip', 'tolerance', 'Perturbation export round trip',
       'largest difference of the exported optical frames from the moved project\'s (m and rad)',
       'the project moved through core.factors by each draw\'s components, composed here; SciPy reads the '
       'quaternions', 1e-8, 'm or rad', refs=('isaaclab_camera_cfg', 'scipy_rotation'))
def _export(ctx):
    from scipy.spatial.transform import Rotation

    from ..core import tolerance as TL
    from ..core.exporters import perturbations as PX
    p, uids, _C = tol_rig()
    out = os.path.join(ctx.tmp(), 'perturbations-check')
    PX.write(p, out, {'n': 12, 'seed': 5})
    doc = json.load(open(os.path.join(out, 'perturbations', 'perturbations.json'), encoding='utf-8'))
    Z = np.load(os.path.join(out, 'perturbations', 'perturbations.npz'))
    err = 0.0
    for i, per in enumerate(doc['draws']):
        for j, s in enumerate(doc['sensors']):
            q = TL.apply_draw(p, {s['uid']: np.array(per[j]['d'])})
            X = _optical(q, s['uid'])
            il = per[j]['isaac_lab']
            R = Rotation.from_quat(il['rot_wxyz'], scalar_first=True).as_matrix()
            err = max(err, float(np.max(np.abs(np.array(il['pos']) - X[:3, 3]))),
                      float(np.linalg.norm(Rotation.from_matrix(R.T @ X[:3, :3]).as_rotvec())),
                      float(np.max(np.abs(Z['T_parent_optical'][i, j] - X))))
    return Outcome(err, {'draws': len(doc['draws']), 'sensors': len(doc['sensors'])})
