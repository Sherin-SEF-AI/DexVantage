"""Monte Carlo tolerances (spec §7.11, §8.5): the uncertainty of every sensor's mount, drawn many times and carried
through the named figures of merit (core/metrics.py) into distributions.

Ported from Vantage core/tolerance.py (the sampler: uniform or normal draws scaled by ± bounds, a fixed seed, the
samples kept for the worst cases and a per-parameter sensitivity). The automotive parts are gone (the blind-ground
grid, seams, sliding along a body surface): here a draw is a change of each mount, applied to a copy of the project
through core/factors.py, and the figures are whatever metrics the caller names.

What a mount's uncertainty is, per sensor of the active variant (enabled sensors; locked ones too, because a locked
mount is still built with tolerances):

- **Mount tolerance** (project Mount.tolerance, `position_mm`, `rotation_deg`): a **± bound per axis**, as the
  Inspector and the Sensors page label it ("± 1 mm · ± 0.5°", the project's defaults, labelled defaults). Each of
  the three shifts (along the mount parent frame's x, y, z) lies within ± position_mm and each of the three turns
  (about the sensor body's own x, y, z, applied in that order: R' = R · Rx · Ry · Rz) within ± rotation_deg,
  independently. The distribution inside the bound is a setting: 'uniform' (the default, a labelled default: every
  value within the bound equally likely, the conservative reading of a drawing tolerance, as Vantage's default) or
  'normal' (the bound is 2 σ and the draw is truncated at the bound: about 95.4 % of an untruncated normal lies
  within it).
- **Calibration covariance** (project.calibration: a hand-eye session whose result is solved and applied to this
  sensor, core/calib): the bundle adjustment's 6 × 6 covariance of the imager's optical frame X in the mount frame,
  in its own parameterisation X' = [R Exp(ω) | t + v] (ω about X's own axes, rad; v along the parent's axes, m).
  A draw is (ω, v) ~ N(0, C), turned exactly into the body's shift and turns (the imager's offset in the body
  included: δ = R_im ω, shift = v − R_B (Exp(δ) − I) t_im, the turns the X-Y-Z decomposition of Exp(δ)). It replaces
  the mount tolerance while the sensor stays where the calibration put it; a sensor moved since is back on its mount
  tolerance (said in the source's note).

    sources(project, scale, distribution)        -> [Source] (what is drawn for each sensor)
    bounds(project, scale, distribution)         -> {uid: {'position_m': [[lo, hi]] × 3, 'rotation_deg': [[lo, hi]] × 3,
                                                   'source': 'mount tolerance' | 'calibration covariance', ...}}
                                                   (calibration: ± 2 σ of each marginal, linearised; a mount: ± bound)
    draw(project, rng, scale, distribution)      -> {uid: [dx, dy, dz (m), droll, dpitch, dyaw (deg)]}
    perturb(project, rng, scale=1.0)             -> a NEW Project (core.factors.apply, allow_locked=True)
    monte_carlo(project, metric_ids, n, seed)    -> per metric: nominal, mean, σ, P5 / P50 / P95, the share of draws
                                                   meeting each requirement (with its Wilson 95 % interval; decided
                                                   as the scorecard decides it, so a draw whose figure is only a
                                                   bound that does not settle the requirement counts as not meeting
                                                   it and is counted in 'undecided') and the contribution of each
                                                   mount (Spearman rank correlation)
    perturbed_extrinsics(project, n, seed)       -> the draws as arrays for the perturbation export (§8.5)

Contribution of a mount: for each of its six components, the Spearman rank correlation ρ of the metric with the
component and ρ_abs with its magnitude (a figure that falls whichever way a camera turns has ρ ≈ 0 and ρ_abs large);
the component counts max(ρ², ρ_abs²), the mount the sum over its six, and the shares are those sums normalised over
the mounts. A correlation below 2 / √(n − 1) (two standard errors of ρ under no association) is not distinguishable
from noise and is flagged so.

Known limits: the six components of a mount tolerance are independent and none is correlated between sensors (a
shared bracket that bends moves its cameras together; that is not modelled); the turns are applied in a fixed order
(x, y, z), which differs from a rotation vector only to second order; the calibration covariance is the bundle
adjustment's Gauss-Newton estimate (core/calib/handeye.py says when it is optimistic), and the intrinsics' covariance
is not drawn; a robot's joint encoders and link lengths are not perturbed; the rank-correlation shares treat the
components as independent, which a calibration covariance's correlated components are not (its shares are
indicative); the metrics are re-evaluated by their studies for every draw, so n is limited by the slowest study
(a default of 64 draws resolves a share of 95 % to about ± 6 %).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import factors as F
from .frames import Pose, quat_to_matrix
from .jobs import Ctx, StudyError

COMPONENTS = ('dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw')     # m, m, m, deg, deg, deg
UNITS = ('m', 'm', 'm', '°', '°', '°')
DISTRIBUTIONS = ('uniform', 'normal')
NORMAL_K = 2.0                    # 'normal': the ± bound is NORMAL_K σ and draws are truncated there (documented)
CAL_K = 2.0                       # bounds() of a calibration covariance: ± CAL_K σ of each marginal
STALE_M, STALE_RAD = 1e-6, 1e-6   # a calibrated sensor moved by more than this is back on its mount tolerance
MOUNT = 'mount tolerance'
CALIB = 'calibration covariance'


# ====================================================================== small maths
def exp_so3(w) -> np.ndarray:
    """Rodrigues: the rotation matrix of the rotation vector w (rad)."""
    w = np.asarray(w, float)
    th = float(np.linalg.norm(w))
    K = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    if th < 1e-12:
        return np.eye(3) + K
    return np.eye(3) + math.sin(th) / th * K + (1 - math.cos(th)) / th ** 2 * (K @ K)


def decompose_xyz(R) -> tuple[float, float, float]:
    """(a, b, c) rad with R = Rx(a) · Ry(b) · Rz(c) (turns about the body's own x, then y, then z: the order
    core.factors applies droll, dpitch, dyaw in)."""
    R = np.asarray(R, float)
    b = math.asin(max(-1.0, min(1.0, R[0, 2])))
    a = math.atan2(-R[1, 2], R[2, 2])
    c = math.atan2(-R[0, 1], R[0, 0])
    return a, b, c


def _skew(t) -> np.ndarray:
    return np.array([[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]])


def ranks(x) -> np.ndarray:
    """1-based ranks with ties given their average rank."""
    x = np.asarray(x, float)
    _u, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    avg = np.cumsum(cnt) - (cnt - 1) / 2.0
    return avg[inv.reshape(-1)]


def spearman(x, y) -> float | None:
    """Spearman's rank correlation (Pearson's of the average ranks); None when either is constant or n < 3."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or len(x) != len(y):
        return None
    rx, ry = ranks(x), ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = math.sqrt(float(rx @ rx) * float(ry @ ry))
    if d <= 0:
        return None
    return float(rx @ ry) / d


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """The Wilson score interval (95 % by default) of a proportion k / n."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


# ====================================================================== what is drawn per sensor
@dataclass
class Source:
    uid: str
    name: str
    kind: str                              # MOUNT | CALIB
    position_mm: float = 0.0               # MOUNT: the ± bounds (before scale)
    rotation_deg: float = 0.0
    distribution: str = 'uniform'
    cov: np.ndarray | None = None          # CALIB: 6 × 6 of (ω rad, v m), the BA order
    R_B: np.ndarray = field(default_factory=lambda: np.eye(3))     # the body's rotation in its parent (nominal)
    T_im: np.ndarray = field(default_factory=lambda: np.eye(4))    # CALIB: the imager's optical frame in the body
    locked: bool = False
    note: str = ''
    session: str = ''

    def body_cov(self) -> np.ndarray:
        """CALIB: the linearised 6 × 6 covariance of (shift m, turns rad) of the body: J C Jᵀ."""
        R_im, t_im = self.T_im[:3, :3], self.T_im[:3, 3]
        J = np.zeros((6, 6))
        J[:3, :3] = self.R_B @ _skew(t_im) @ R_im
        J[:3, 3:] = np.eye(3)
        J[3:, :3] = R_im
        return J @ self.cov @ J.T

    def to_dict(self, scale: float = 1.0) -> dict:
        d = {'sensor_uid': self.uid, 'name': self.name, 'source': self.kind, 'locked': self.locked}
        if self.kind == MOUNT:
            d.update(position_mm=self.position_mm, rotation_deg=self.rotation_deg, distribution=self.distribution,
                     meaning=('± bound per axis, ' + ('uniform within it' if self.distribution == 'uniform' else
                                                      f'normal with the bound at {NORMAL_K:g} σ, truncated there')))
        else:
            C = self.body_cov() * scale * scale
            s = np.sqrt(np.clip(np.diag(C), 0, None))
            d.update(session=self.session, sigma_mm=[1000 * float(v) for v in s[:3]],
                     sigma_deg=[math.degrees(float(v)) for v in s[3:]],
                     meaning='normal, the hand-eye bundle adjustment\'s covariance (shift along the parent\'s axes, '
                             'turns about the body\'s own)')
        if self.note:
            d['note'] = self.note
        return d


def _sensors(project, uids=None) -> list:
    out = [s for s in project.variant.sensors if s.enabled]
    if uids:
        want = set(uids)
        out = [s for s in out if s.uid in want or s.name in want]
    return out


def calibration_source(project, sensor):
    """(covariance 6 × 6 (ω, v), T_im 4 × 4, session name, note) of the latest solved and applied hand-eye session of
    this sensor, or (None, None, '', why not) — a sensor moved since its calibration has no calibration source."""
    cal = getattr(project, 'calibration', None) or {}
    sess = [s for s in cal.get('sessions') or [] if isinstance(s, dict) and s.get('sensor_uid') == sensor.uid]
    sess = [s for s in sess if isinstance(s.get('result'), dict) and s['result'].get('ok') and
            s['result'].get('applied')]
    if not sess:
        return None, None, '', ''
    s = max(sess, key=lambda x: str(x['result'].get('applied_at') or ''))
    cams = s['result'].get('cameras') or {}
    entry = cams.get(sensor.name) if len(cams) != 1 else next(iter(cams.values()))
    if not isinstance(entry, dict) or entry.get('covariance') is None or entry.get('matrix') is None:
        return None, None, '', f'its calibration "{s.get("name", "")}" states no covariance for it'
    C = np.asarray(entry['covariance'], float)
    X = np.asarray(entry['matrix'], float)
    if C.shape != (6, 6) or X.shape != (4, 4) or not np.isfinite(C).all():
        return None, None, '', f'its calibration "{s.get("name", "")}" has no usable covariance'
    try:
        from .calib import records as RC
        T_im = RC.sensor_imager(sensor).pose_in_body.matrix()
    except Exception as e:                      # noqa: BLE001 — no imager: the mount tolerance, said why
        return None, None, '', f'its imager could not be read ({e})'
    Xn = sensor.mount.pose.matrix() @ T_im
    dt = float(np.linalg.norm(Xn[:3, 3] - X[:3, 3]))
    dr = float(np.arccos(np.clip((np.trace(Xn[:3, :3].T @ X[:3, :3]) - 1) / 2, -1, 1)))
    if dt > STALE_M or dr > STALE_RAD:
        return None, None, '', (f'it moved {1000 * dt:.2f} mm and {math.degrees(dr):.3f}° since its calibration '
                                f'"{s.get("name", "")}", so its mount tolerance is used')
    return 0.5 * (C + C.T), T_im, str(s.get('name') or ''), ''


def sources(project, scale: float = 1.0, distribution: str = 'uniform', sensors=None) -> list:
    """[Source] for every enabled sensor of the active variant (sensors: uids or names to keep; None = all)."""
    if distribution not in DISTRIBUTIONS:
        raise StudyError(f'The tolerance distribution must be one of {", ".join(DISTRIBUTIONS)}; it is '
                         f'{distribution!r}.')
    out = []
    for s in _sensors(project, sensors):
        C, T_im, name, why = calibration_source(project, s)
        R_B = quat_to_matrix(s.mount.pose.quaternion)
        if C is not None:
            out.append(Source(s.uid, s.name, CALIB, cov=C, R_B=R_B, T_im=T_im, locked=s.locked, session=name))
            continue
        t = s.mount.tolerance
        out.append(Source(s.uid, s.name, MOUNT, float(t.position_mm), float(t.rotation_deg), distribution,
                          R_B=R_B, locked=s.locked, note=why))
    return out


def bounds(project, scale: float = 1.0, distribution: str = 'uniform', sensors=None) -> dict:
    """{sensor uid: {'position_m': [[lo, hi]] × 3, 'rotation_deg': [[lo, hi]] × 3, 'source', 'meaning'}}: a mount
    tolerance's ± bound, a calibration covariance's ± 2 σ per marginal (linearised to the body's shift and turns)."""
    out = {}
    for src in sources(project, scale, distribution, sensors):
        if src.kind == MOUNT:
            p = src.position_mm / 1000.0 * scale
            r = src.rotation_deg * scale
            pos = [[-p, p]] * 3
            rot = [[-r, r]] * 3
            meaning = '± bound'
        else:
            s = np.sqrt(np.clip(np.diag(src.body_cov()), 0, None)) * scale
            pos = [[-CAL_K * float(v), CAL_K * float(v)] for v in s[:3]]
            rot = [[-CAL_K * math.degrees(float(v)), CAL_K * math.degrees(float(v))] for v in s[3:]]
            meaning = f'± {CAL_K:g} σ of each marginal'
        out[src.uid] = {'position_m': [list(x) for x in pos], 'rotation_deg': [list(x) for x in rot],
                        'source': src.kind, 'meaning': meaning}
    return out


def _truncated_normal(rng, k: int, cut: float) -> np.ndarray:
    """k standard-normal draws truncated at ± cut (rejection)."""
    z = rng.standard_normal(k)
    bad = np.abs(z) > cut
    while bad.any():
        z[bad] = rng.standard_normal(int(bad.sum()))
        bad = np.abs(z) > cut
    return z


def _cov_root(C) -> np.ndarray:
    w, V = np.linalg.eigh(np.asarray(C, float))
    return V * np.sqrt(np.clip(w, 0, None))


def draw_source(src: Source, rng, scale: float = 1.0) -> np.ndarray:
    """One draw [dx, dy, dz (m), droll, dpitch, dyaw (deg)] of a sensor's mount."""
    if src.kind == MOUNT:
        if src.distribution == 'normal':
            u = _truncated_normal(rng, 6, NORMAL_K) / NORMAL_K
        else:
            u = rng.uniform(-1.0, 1.0, 6)
        b = np.array([src.position_mm / 1000.0] * 3 + [src.rotation_deg] * 3) * scale
        return u * b
    y = _cov_root(src.cov) @ rng.standard_normal(6) * scale
    w, v = y[:3], y[3:]
    R_im, t_im = src.T_im[:3, :3], src.T_im[:3, 3]
    Rd = exp_so3(R_im @ w)
    shift = v - src.R_B @ ((Rd - np.eye(3)) @ t_im)
    a, b, c = decompose_xyz(Rd)
    return np.r_[shift, math.degrees(a), math.degrees(b), math.degrees(c)]


def draw(project, rng, scale: float = 1.0, distribution: str = 'uniform', srcs=None, sensors=None) -> dict:
    """{sensor uid: [dx, dy, dz (m), droll, dpitch, dyaw (deg)]}, the sensors in the variant's order."""
    srcs = srcs if srcs is not None else sources(project, scale, distribution, sensors)
    return {s.uid: draw_source(s, rng, scale) for s in srcs}


def apply_draw(project, d: dict):
    """A NEW project with each sensor's mount moved by its draw (core.factors, locked mounts included)."""
    values = {}
    for uid, x in d.items():
        for c, v in zip(COMPONENTS, x):
            values[f'sensor:{uid}.{c}'] = float(v)
    return F.apply(project, values, allow_locked=True)


def perturb(project, rng, scale: float = 1.0, distribution: str = 'uniform', sensors=None):
    """A NEW project whose mounts are one draw of their tolerances (or calibration covariance)."""
    return apply_draw(project, draw(project, rng, scale, distribution, sensors=sensors))


def delta_body(src_pose: Pose, x) -> np.ndarray:
    """The 4 × 4 Δ in the sensor body frame of a draw x: T_body' = T_body · Δ (core/sim/render extrinsic_delta)."""
    R = quat_to_matrix(src_pose.quaternion)
    a, b, c = (math.radians(float(v)) for v in x[3:])
    from .frames import rot_x, rot_y, rot_z
    D = np.eye(4)
    D[:3, :3] = rot_x(a) @ rot_y(b) @ rot_z(c)
    D[:3, 3] = R.T @ np.asarray(x[:3], float)
    return D


# ====================================================================== Monte Carlo
OPS = {'>=': lambda v, t: v >= t, '<=': lambda v, t: v <= t, '>': lambda v, t: v > t, '<': lambda v, t: v < t,
       '==': lambda v, t: abs(v - t) <= 1e-9 * max(1.0, abs(t))}


def meets(op: str, v, t, bound=None) -> bool | None:
    """Whether the figure v meets 'op t' as the requirements scorecard decides it (core/requirements.decide: the
    same tolerance of 10⁻⁹ relative at the threshold; a figure a study states only as a bound ('at_least' /
    'at_most') meets or misses only when the bound settles it). None: no figure, or a bound that does not settle it
    (a draw that cannot be shown to meet the requirement, counted as not meeting it)."""
    if v is None:
        return None
    from .requirements import decide
    status = decide(float(v), bound, op if op in OPS else '>=', float(t))[0]
    return None if status == 'not_evaluated' else status == 'pass'


def _evaluate(project, ids, ctx):
    from . import metrics as M
    return M.evaluate(project, ids, cache=None, ctx=ctx)


def monte_carlo(project, metric_ids, n: int, seed: int, scale: float = 1.0, distribution: str = 'uniform',
                ctx: Ctx | None = None, evaluator=None, sensors=None) -> dict:
    """The metrics over n draws of the mounts. evaluator(project, ids, ctx) -> {id: {'value', 'unit', 'better',
    'title', 'status', ...}} (default core.metrics.evaluate); it is called once on the project as it is (the nominal)
    and once per draw. Stops early when ctx is cancelled (the statistics then cover the draws made)."""
    ctx = ctx or Ctx()
    ids = list(dict.fromkeys(metric_ids))
    n = int(n)
    if n < 2:
        raise StudyError('The tolerance Monte Carlo needs at least 2 draws.')
    if not ids:
        raise StudyError('Name at least one figure to carry through the tolerances.')
    ev = evaluator or _evaluate
    srcs = sources(project, scale, distribution, sensors)
    if not srcs:
        raise StudyError('There is no enabled sensor in this rig to perturb: add one (Sensors) first.')
    inner = Ctx(cancel=ctx.cancel)
    ctx.report(0.0, 'The design as it is')
    nominal = ev(project, ids, inner)
    rng = np.random.default_rng(int(seed))
    D = np.zeros((n, len(srcs), 6))
    vals = {m: [None] * n for m in ids}
    bnds = {m: [None] * n for m in ids}              # 'at_least' / 'at_most' where the study states only a bound
    done = 0
    for i in range(n):
        if ctx.cancelled():
            break
        d = {s.uid: draw_source(s, rng, scale) for s in srcs}
        D[i] = np.array([d[s.uid] for s in srcs])
        res = ev(apply_draw(project, d), ids, inner)
        for m in ids:
            v = (res.get(m) or {}).get('value')
            vals[m][i] = None if v is None or not math.isfinite(float(v)) else float(v)
            bnds[m][i] = (res.get(m) or {}).get('bound') if vals[m][i] is not None else None
        done = i + 1
        ctx.report(done / n, f'Draw {done} of {n}')
    D = D[:done]
    noise = 2.0 / math.sqrt(max(done - 1, 1))
    reqs = list(getattr(project, 'requirements', None) or [])
    out = {}
    for m in ids:
        nom = nominal.get(m) or {}
        v = vals[m][:done]
        ok = np.array([x is not None for x in v], bool)
        arr = np.array([x if x is not None else np.nan for x in v], float)
        good = arr[ok]
        row = {'title': nom.get('title', m), 'unit': nom.get('unit', ''), 'better': nom.get('better', 'higher'),
               'study': nom.get('study', ''), 'nominal': nom.get('value'), 'nominal_status': nom.get('status', ''),
               'values': [None if x is None else float(x) for x in v], 'valid': int(ok.sum()),
               'no_value': int(done - ok.sum())}
        if nom.get('reason'):
            row['reason'] = nom['reason']
        if len(good):
            p5, p50, p95 = (float(x) for x in np.percentile(good, [5, 50, 95]))
            row.update(mean=float(good.mean()), std=float(good.std(ddof=1)) if len(good) > 1 else 0.0,
                       p5=p5, p50=p50, p95=p95, min=float(good.min()), max=float(good.max()))
            hi = row['better'] == 'higher'
            row['pessimistic'] = {'quantile': 'P5' if hi else 'P95', 'value': p5 if hi else p95}
        else:
            row.update(mean=None, std=None, p5=None, p50=None, p95=None, min=None, max=None,
                       pessimistic={'quantile': 'P5' if row['better'] == 'higher' else 'P95', 'value': None})
        rq = []
        for r in reqs:
            if r.metric != m:
                continue
            dec = [meets(r.op, x, r.threshold, b) for x, b in zip(v, bnds[m][:done]) if x is not None]
            k = sum(1 for x in dec if x)
            lo, hi_ = wilson(k, len(good))
            rq.append({'uid': r.uid, 'name': r.name or row['title'], 'op': r.op, 'threshold': float(r.threshold),
                       'is_default': bool(r.is_default), 'met': k, 'of': int(len(good)),
                       'undecided': sum(1 for x in dec if x is None),
                       'share': k / len(good) if len(good) else None, 'share_ci95': [lo, hi_],
                       'nominal_meets': meets(r.op, nom.get('value'), r.threshold, nom.get('bound'))})
        row['requirements'] = rq
        row['contributions'] = _contributions(srcs, D[ok], good, noise) if len(good) >= 3 else []
        out[m] = row
    return {'n': n, 'done': done, 'seed': int(seed), 'scale': float(scale), 'distribution': distribution,
            'sources': [s.to_dict(scale) for s in srcs], 'bounds': bounds(project, scale, distribution, sensors),
            'metrics': out, 'noise_rho': noise, 'cancelled': done < n,
            'draws': {s.uid: D[:, j, :].tolist() for j, s in enumerate(srcs)}, 'components': list(COMPONENTS),
            'component_units': list(UNITS)}


def _contributions(srcs, D, y, noise) -> list:
    rows = []
    for j, s in enumerate(srcs):
        comps = {}
        tot = 0.0
        best = 0.0
        for k, c in enumerate(COMPONENTS):
            x = D[:, j, k]
            r = spearman(x, y)
            ra = spearman(np.abs(x), y)
            r2 = max((r or 0.0) ** 2, (ra or 0.0) ** 2)
            tot += r2
            best = max(best, abs(r or 0.0), abs(ra or 0.0))
            comps[c] = {'rho': r, 'rho_abs': ra}
        pos = sum(max((comps[c]['rho'] or 0) ** 2, (comps[c]['rho_abs'] or 0) ** 2) for c in COMPONENTS[:3])
        rows.append({'sensor_uid': s.uid, 'name': s.name, 'source': s.kind, 'locked': s.locked, 'rho2': tot,
                     'position_rho2': pos, 'rotation_rho2': tot - pos, 'max_rho': best,
                     'significant': best > noise, 'components': comps})
    total = sum(r['rho2'] for r in rows)
    for r in rows:
        r['share'] = r['rho2'] / total if total > 0 else 0.0
    rows.sort(key=lambda r: -r['rho2'])
    return rows


# ====================================================================== the perturbation export (§8.5)
def perturbed_extrinsics(project, n: int, seed: int, scale: float = 1.0, distribution: str = 'uniform',
                         q: dict | None = None, sensors=None) -> dict:
    """n draws of every enabled sensor's mount as arrays (S sensors, the variant's order):

        d                (n, S, 6)    [dx, dy, dz (m, parent axes), droll, dpitch, dyaw (deg, body axes, x-y-z)]
        delta_body       (n, S, 4, 4) T_body' = T_body · Δ (core/sim/render extrinsic_delta)
        T_parent_body    (n, S, 4, 4) the perturbed sensor body in its mount parent frame
        T_parent_optical (n, S, 4, 4) the primary imager's optical frame (REP-103: z forward, x right, y down) in the
                                      mount parent frame (NaN for a sensor without an imager)
        T_world_optical  (n, S, 4, 4) the same in the world at the joint state q (home when not given; NaN when the
                                      mount's link cannot be posed)
    plus 'sensors' [{'uid', 'name', 'parent_frame', 'robot_uid', 'imager', 'width', 'height', 'fx', 'fy', 'cx',
    'cy'}] and 'sources' (what was drawn)."""
    from . import scenegeom as SG
    srcs = sources(project, scale, distribution, sensors)
    if not srcs:
        raise StudyError('There is no enabled sensor in this rig to perturb: add one (Sensors) first.')
    n = int(n)
    if n < 1:
        raise StudyError('The perturbation export needs at least 1 draw.')
    rng = np.random.default_rng(int(seed))
    S = len(srcs)
    d = np.zeros((n, S, 6))
    for i in range(n):
        for j, s in enumerate(srcs):
            d[i, j] = draw_source(s, rng, scale)
    info, T_im, T_wp = [], [], []
    for s in srcs:
        sen = project.variant.sensor(s.uid)
        row = {'uid': sen.uid, 'name': sen.name, 'parent_frame': sen.mount.parent_frame,
               'robot_uid': sen.mount.robot_uid, 'imager': None}
        Ti = None
        try:
            from .calib import records as RC
            im = RC.sensor_imager(sen)
            Ti = im.pose_in_body.matrix()
            cam = im.camera
            row.update(imager=im.name, width=int(cam.width), height=int(cam.height))
            for k in ('fx', 'fy', 'cx', 'cy'):
                v = getattr(cam, k, None)
                row[k] = float(v) if isinstance(v, (int, float)) and math.isfinite(float(v)) else None
            row['lens_model'] = str(getattr(cam, 'model', '') or getattr(cam, 'kind', '') or '')
        except Exception:                       # noqa: BLE001 — a sensor without an imager: no optical frames
            pass
        try:
            Tw = SG.sensor_world_pose(project, sen, q).matrix() @ np.linalg.inv(sen.mount.pose.matrix())
        except Exception:                       # noqa: BLE001 — its link cannot be posed: no world frames
            Tw = None
        info.append(row)
        T_im.append(Ti)
        T_wp.append(Tw)
    nanT = np.full((4, 4), np.nan)
    dB = np.zeros((n, S, 4, 4))
    Tpb = np.zeros((n, S, 4, 4))
    Tpo = np.zeros((n, S, 4, 4))
    Two = np.zeros((n, S, 4, 4))
    for j, s in enumerate(srcs):
        pose = project.variant.sensor(s.uid).mount.pose
        B = pose.matrix()
        for i in range(n):
            D = delta_body(pose, d[i, j])
            dB[i, j] = D
            Tpb[i, j] = B @ D
            Tpo[i, j] = Tpb[i, j] @ T_im[j] if T_im[j] is not None else nanT
            Two[i, j] = T_wp[j] @ Tpo[i, j] if T_wp[j] is not None and T_im[j] is not None else nanT
    return {'n': n, 'seed': int(seed), 'scale': float(scale), 'distribution': distribution, 'd': d,
            'delta_body': dB, 'T_parent_body': Tpb, 'T_parent_optical': Tpo, 'T_world_optical': Two,
            'sensors': info, 'sources': [s.to_dict(scale) for s in srcs], 'components': list(COMPONENTS),
            'component_units': list(UNITS)}
