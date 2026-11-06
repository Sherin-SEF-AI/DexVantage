"""Inverse kinematics of a robot link: damped least squares with joint-limit avoidance, several seeds and a rule to
choose among the solutions.

    solve(model, link, target, q0=None, seeds=8, ...)          -> IKResult(q, ok, pos_err, rot_err, iterations, reason)
    solve_batch(model, link, targets (N, 4, 4), ...)           -> IKBatch (arrays over the targets, reason per target)

The model. The link's pose x(q) (a 4 × 4 world transform, optionally followed by a fixed tool offset `tcp` in link
coordinates) is driven to the target by iterating

    e  = [p* − p ;  w · log(R* Rᵀ)]                     position error (m) and rotation error (rad, world axis-angle)
    Jw = [J_lin ; w · J_ang]                            MuJoCo's world Jacobian of the tool point, rows weighted alike
    Δq = (Jwᵀ Jw + λ² I)⁻¹ Jwᵀ e,   λ² = ½|e|² + δ²     damped least squares [Wampler 1986; Nakamura and Hanafusa
                                                        1986] with Sugihara's error-dependent damping [Sugihara 2011]
    Δq += −k (I − Jw⁺ Jw) ∇H(q),   H = Σ ((q − q_mid)/(q_hi − q_lo))²
                                                        a null-space gradient away from the joint limits [Liégeois
                                                        1977] (it moves only joints the task leaves free)
    q  ← clip(q + Δq, lower, upper)                     the step is capped at max_step and the joints clamped

until |p* − p| ≤ pos_tol and the rotation angle ≤ rot_tol (default 0.1 mm and 1 mrad), the error score has not
dropped by PROGRESS (0.1 %) for STALL iterations, or max_iter. w = weight_rot (m per rad), δ = damping and
k = limit_gain are solver defaults, not figures of any robot.

Only the joints that move the link are solved (manipulability.chain_mask): a gripper's finger joints, which do not
move the gripper's base, or the other arm of a bimanual robot stay at q0 (or home) and do not count as joints of the
chain. The SO-100's jaw, for instance, does not move its fixed jaw, so that link's chain has five joints.

Position-only mode. A 5-joint arm (the SO-100) cannot set all six pose coordinates, so mode='auto' (the default)
solves chains with fewer than 6 joints position-first: the position task by damped least squares, and the
orientation as a second task inside its null space N_p = I − J_lin⁺ J_lin [Nakamura and Hanafusa 1986], so the arm
gets the target's position exactly and the orientation as close as its joints allow. `ok` then means the position
was reached; `orientation_ok` says whether the orientation was too, and `note` says by how much it was missed.
mode='pose' asks for the full pose on any robot (a 5-joint arm then fails with the reason 'orientation unreachable'
when the position alone could be reached); mode='position' ignores the orientation's rank and solves position-first.
A target given as a 3-vector position leaves the orientation free (position task only; rot_err is nan).

Seeds and the rule. Each target is solved from `seeds` starting points in lock step (the Jacobians come from MuJoCo
one at a time; the linear algebra is batched over every seed and target with numpy): q0 when given, the home pose,
then joint vectors drawn uniformly within the limits from a seeded generator (rng; ±π about home for unlimited
revolute joints). Of the converged seeds, `rule` picks: 'nearest_home' (smallest Euclidean joint distance to home,
unlimited revolute joints wrapped to within π of home), 'nearest_q0' (the same, to q0, and wrapped to within π of q0:
smooth for dragging, never a jump of a full turn) or 'most_manipulable' (largest Yoshikawa measure √det(J Jᵀ) of the
task Jacobian, manipulability.py). With collision=True the chosen solution is the first in rule order whose
self-collision distance (collision.py) is at least `margin`. Targets no seed solved get retry_seeds (default 2 × seeds)
fresh random seeds before their reason is decided.

Why a target fails (IKResult.reason, IKBatch.reason), checked in this order:

    'out of reach'            the target is farther from the first moving joint than the chain's reach bound (the sum
                              of the distances between consecutive joint anchors and to the tool point at home,
                              plus each prismatic joint's travel from home: the triangle inequality), or every seed
                              stopped improving short of it with and without the joint limits
    'joint limit'             a solve without the limits converges, and whole turns of the revolute joints do not
                              bring that solution inside them
    'orientation unreachable' mode='pose' on a chain with fewer than 6 joints: the position can be reached
                              but not the orientation
    'self-collision'          every converged solution has the robot touching itself (closer than margin)
    'no convergence'          none of the above: the seeds ran out of iterations (more seeds or iterations may help)

References: C. W. Wampler 1986; Y. Nakamura and H. Hanafusa 1986; T. Sugihara 2011; A. Liégeois 1977; T. Yoshikawa
1985 (validation/references.py).

Known limits: a local method: a reachable target can be missed when every seed falls into a local minimum (reported
as 'no convergence' or, if the seeds also stalled without limits, 'out of reach'); the reach bound is an upper bound
(a target inside it may still be unreachable); unlimited revolute joints are drawn within ±π of home only; mimic
joints of a URDF move independently (as in model.py); the self-collision test uses convex hulls of the meshes
(collision.py); environment obstacles are not avoided.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import collision as _col
from .manipulability import chain_mask, yoshikawa

REASONS = ('out of reach', 'joint limit', 'orientation unreachable', 'self-collision', 'no convergence')
RULES = ('nearest_home', 'most_manipulable', 'nearest_q0')
MODES = ('auto', 'pose', 'position')

# solver defaults (labelled defaults: tuning of the method, not figures of any robot)
POS_TOL = 1e-4                    # m
ROT_TOL = 1e-3                    # rad
MAX_ITER = 200
DAMPING = 0.01                    # δ: λ² = ½|e|² + δ²
WEIGHT_ROT = 0.3                  # m per rad: how much one radian of rotation error weighs against one metre
LIMIT_GAIN = 0.05                 # k of the null-space gradient away from the limits
MAX_STEP = 0.5                    # rad (or m) per iteration, largest joint change
STALL = 15                        # iterations without PROGRESS before a seed gives up
PROGRESS = 1e-3                   # relative decrease of the error score that counts as progress
SEED = 20261106                   # the default generator seed (results are reproducible)
CHUNK = 512                       # problems (targets × seeds) iterated in lock step


# ====================================================================== results
@dataclass
class IKResult:
    q: np.ndarray                 # (nq,) the chosen solution, or the closest attempt when not ok
    ok: bool
    pos_err: float                # m
    rot_err: float                # rad
    iterations: int               # of the chosen (or closest) seed
    reason: str = ''              # '' when ok, else one of REASONS
    mode: str = 'pose'            # what was solved: 'pose' or 'position' (position-first)
    orientation_ok: bool = True   # the orientation is within rot_tol (always True for an ok pose solve)
    manipulability: float = float('nan')   # Yoshikawa measure of the task Jacobian at q (nan when not ok)
    min_distance: float = float('nan')     # self-collision distance at q, m (nan when not checked)
    solutions: int = 0            # seeds that converged
    note: str = ''                # a plain sentence when there is something to say


@dataclass
class IKBatch:
    q: np.ndarray                 # (N, nq)
    ok: np.ndarray                # (N,) bool
    pos_err: np.ndarray           # (N,) m
    rot_err: np.ndarray           # (N,) rad
    iterations: np.ndarray        # (N,) int
    reason: list                  # N strings, '' when ok
    orientation_ok: np.ndarray    # (N,) bool
    manipulability: np.ndarray    # (N,)
    min_distance: np.ndarray      # (N,) m (nan when not checked)
    solutions: np.ndarray         # (N,) int
    mode: str = 'pose'
    settings: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.ok)

    def counts(self) -> dict:
        """{'ok': n, reason: n, ...} for every reason that occurred."""
        out = {'ok': int(np.count_nonzero(self.ok))}
        for r in REASONS:
            n = sum(1 for x in self.reason if x == r)
            if n:
                out[r] = n
        return out

    def result(self, i: int) -> IKResult:
        note = ''
        if self.ok[i] and not self.orientation_ok[i]:
            note = _orientation_note(self.rot_err[i])
        return IKResult(self.q[i].copy(), bool(self.ok[i]), float(self.pos_err[i]), float(self.rot_err[i]),
                        int(self.iterations[i]), self.reason[i], self.mode, bool(self.orientation_ok[i]),
                        float(self.manipulability[i]), float(self.min_distance[i]), int(self.solutions[i]), note)


def _orientation_note(rot_err) -> str:
    return (f'Position only: this arm has fewer than six moving joints, so it cannot reach every orientation; the '
            f'orientation is {np.degrees(rot_err):.1f}° from the target.')


# ====================================================================== the chain
class _Chain:
    """The tool point of one link: its world pose and Jacobian from one MuJoCo kinematics pass."""

    def __init__(self, model, link: str, tcp=None):
        import mujoco
        self.model = model
        self.i = model.link_index(link)
        self.body = int(model._lb[self.i])
        self.off = model._loff[self.i] @ (np.eye(4) if tcp is None else np.asarray(tcp, float).reshape(4, 4))
        self.base = model.base
        self.nq = model.nq
        self.active = chain_mask(model, link)        # the joints that move the link; the others are held
        self.jp = np.zeros((3, model.m.nv))
        self.jr = np.zeros((3, model.m.nv))
        self._mj = mujoco

    def eval(self, q):
        mj, model = self._mj, self.model
        d = model._kin(q)
        mj.mj_comPos(model.m, d)
        Tb = np.eye(4)
        Tb[:3, :3] = d.xmat[self.body].reshape(3, 3)
        Tb[:3, 3] = d.xpos[self.body]
        T = Tb @ self.off
        mj.mj_jac(model.m, d, self.jp, self.jr, T[:3, 3].copy(), self.body)
        R = self.base[:3, :3]
        J = np.r_[R @ self.jp[:, model._dadr], R @ self.jr[:, model._dadr]]
        return self.base @ T, J

    def reach(self) -> tuple[np.ndarray, float]:
        """(the first moving joint's anchor in the world, an upper bound of the tool point's distance from it)."""
        import mujoco
        model, m = self.model, self.model.m
        chain, b = set(), self.body
        while b > 0:
            chain.add(b)
            b = int(m.body_parentid[b])
        js = [k for k, j in enumerate(model._jid) if int(m.jnt_bodyid[j]) in chain]
        T, _J = self.eval(model.home)
        tool = T[:3, 3]
        if not js:
            return tool, 0.0
        d = model._kin(model.home)
        anchors = (self.base[:3, :3] @ np.asarray(d.xanchor)[model._jid[js]].T).T + self.base[:3, 3]
        r = float(np.linalg.norm(np.diff(anchors, axis=0), axis=1).sum() + np.linalg.norm(tool - anchors[-1]))
        for k in js:
            if int(m.jnt_type[model._jid[k]]) == int(mujoco.mjtJoint.mjJNT_SLIDE):
                # the anchors were taken at home, so a slide can lengthen the chain by its travel from home, which
                # is at most max(upper − home, home − lower) (not max(|lower|, |upper|): with lower < 0 < upper and
                # home at one end the travel is upper − lower)
                h = float(model.home[k])
                r += float(max(model.upper[k] - h, h - model.lower[k], 0.0))
        return anchors[0], r


def _rotvec(R) -> np.ndarray:
    """(P, 3) axis-angle vectors of rotation matrices (P, 3, 3), accurate near 0 and near π."""
    v = 0.5 * np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], axis=1)
    c = np.clip(0.5 * (np.trace(R, axis1=1, axis2=2) - 1.0), -1.0, 1.0)
    s = np.linalg.norm(v, axis=1)
    th = np.arctan2(s, c)
    with np.errstate(divide='ignore', invalid='ignore'):
        f = np.where(s > 1e-12, th / np.where(s > 1e-12, s, 1.0), 1.0)
    out = v * f[:, None]
    near = th > np.pi - 1e-3
    if near.any():
        for p in np.flatnonzero(near):
            # the symmetric part of R is cos θ I + (1 − cos θ) a aᵀ, so a aᵀ = (sym(R) − cos θ I) / (1 − cos θ)
            # exactly; ½ (R + I) carries the skew part sin θ [a] / 2 as well, an axis error of order π − θ
            B = (0.5 * (R[p] + R[p].T) - c[p] * np.eye(3)) / max(1.0 - c[p], 1e-300)
            k = int(np.argmax(np.diag(B)))
            a = B[:, k] / np.sqrt(max(B[k, k], 1e-300))
            a /= np.linalg.norm(a)
            if a @ v[p] < 0:
                a = -a
            out[p] = a * th[p]
    return out


# ====================================================================== the lock-step solver
class _Cfg:
    def __init__(self, pos_tol, rot_tol, max_iter, damping, weight_rot, limit_gain, max_step):
        self.pt, self.rt = float(pos_tol), float(rot_tol)
        self.max_iter = int(max_iter)
        self.d2 = float(damping) ** 2
        self.w = float(weight_rot)
        self.k = float(limit_gain)
        self.max_step = float(max_step)
        self.no_rot = False               # position targets: the orientation is free


def _iterate(chain: _Chain, targets, Q0, cfg: _Cfg, position: bool, clamp: bool, cancel=None):  # noqa: C901
    """Iterate P problems (targets (P,4,4), starts Q0 (P,nq)) in lock step. Returns (best q, pos err, rot err,
    iterations, converged, stalled, J at the best q)."""
    model = chain.model
    P, n = Q0.shape
    lo, hi = (model.lower, model.upper) if clamp else (np.full(n, -np.inf), np.full(n, np.inf))
    fin = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
    still = ~chain.active
    mid = np.where(fin, 0.5 * (np.where(fin, lo, 0.0) + np.where(fin, hi, 0.0)), 0.0)
    rng2 = np.where(fin, (np.where(fin, hi, 1.0) - np.where(fin, lo, 0.0)) ** 2, 1.0)
    q = np.clip(Q0.astype(float), lo, hi)
    best_q, best_s = q.copy(), np.full(P, np.inf)
    best_pe, best_re = np.full(P, np.inf), np.full(P, np.inf)
    best_J = np.zeros((P, 6, n))
    iters, since = np.zeros(P, int), np.zeros(P, int)
    mark = np.full(P, np.inf)                         # the score progress is measured against
    done, conv, stalled = np.zeros(P, bool), np.zeros(P, bool), np.zeros(P, bool)
    prev_re = np.full(P, np.inf)
    tp, tR = targets[:, :3, 3], targets[:, :3, :3]
    I = np.eye(n)
    for it in range(cfg.max_iter + 1):
        act = np.flatnonzero(~done)
        if not len(act) or (cancel is not None and cancel()):
            break
        T = np.empty((len(act), 4, 4))
        J = np.empty((len(act), 6, n))
        for k, p in enumerate(act):
            T[k], J[k] = chain.eval(q[p])
        ep = tp[act] - T[:, :3, 3]
        er = np.zeros((len(act), 3)) if cfg.no_rot else _rotvec(tR[act] @ np.transpose(T[:, :3, :3], (0, 2, 1)))
        pe, re = np.linalg.norm(ep, axis=1), np.linalg.norm(er, axis=1)
        if position:
            s = np.where(pe <= cfg.pt, 1e3, 1e3 * pe / cfg.pt) + re / cfg.rt
            ok = (pe <= cfg.pt) & ((re <= cfg.rt) | (prev_re[act] - re < 1e-7))
        else:
            s = np.maximum(pe / cfg.pt, re / cfg.rt)
            ok = (pe <= cfg.pt) & (re <= cfg.rt)
        prev_re[act] = re
        better = s < best_s[act] * (1 - 1e-9)
        upd = act[better | ok]
        sel = better | ok
        best_q[upd], best_s[upd], best_pe[upd], best_re[upd], best_J[upd] = q[upd], s[sel], pe[sel], re[sel], J[sel]
        iters[act] = it
        moved = s < mark[act] * (1 - PROGRESS)
        mark[act[moved]] = s[moved]
        since[act] = np.where(moved, 0, since[act] + 1)
        conv[act[ok]] = True
        st = (~ok) & (since[act] >= STALL)
        stalled[act[st]] = True
        fin_now = ok | st | (it >= cfg.max_iter)
        done[act[fin_now]] = True
        keep = ~fin_now
        if not keep.any():
            continue
        act, J, ep, er, pe = act[keep], J[keep], ep[keep], er[keep], pe[keep]
        if position:
            Jp, Jr = J[:, :3], J[:, 3:]
            lam = 0.5 * pe ** 2 + cfg.d2
            A = np.transpose(Jp, (0, 2, 1)) @ Jp + lam[:, None, None] * I
            dq1 = np.linalg.solve(A, (np.transpose(Jp, (0, 2, 1)) @ ep[..., None]))[..., 0]
            if cfg.no_rot:
                dq, Jtask = dq1, Jp
            else:
                Np = I - np.linalg.pinv(Jp, rcond=1e-10) @ Jp
                Jrn = Jr @ Np
                er2 = (er - (Jr @ dq1[..., None])[..., 0]) * cfg.w
                Jrn_w = Jrn * cfg.w
                lam2 = 0.5 * np.einsum('pi,pi->p', er2, er2) + cfg.d2
                A2 = np.transpose(Jrn_w, (0, 2, 1)) @ Jrn_w + lam2[:, None, None] * I
                dq2 = np.linalg.solve(A2, np.transpose(Jrn_w, (0, 2, 1)) @ er2[..., None])[..., 0]
                dq = dq1 + (Np @ dq2[..., None])[..., 0]
                Jtask = J
        else:
            Jw = J.copy()
            Jw[:, 3:] *= cfg.w
            e = np.concatenate([ep, cfg.w * er], axis=1)
            lam = 0.5 * np.einsum('pi,pi->p', e, e) + cfg.d2
            A = np.transpose(Jw, (0, 2, 1)) @ Jw + lam[:, None, None] * I
            dq = np.linalg.solve(A, np.transpose(Jw, (0, 2, 1)) @ e[..., None])[..., 0]
            Jtask = Jw
        if clamp and cfg.k > 0 and fin.any() and n > 0:
            N = I - np.linalg.pinv(Jtask, rcond=1e-10) @ Jtask
            g = np.where(fin & chain.active, 2.0 * (q[act] - mid) / rng2, 0.0)
            dq -= cfg.k * (N @ g[..., None])[..., 0]
        dq[:, still] = 0.0
        big = np.abs(dq).max(axis=1) if n else np.zeros(len(act))
        scale = np.where(big > cfg.max_step, cfg.max_step / np.maximum(big, 1e-300), 1.0)
        q[act] = np.clip(q[act] + dq * scale[:, None], lo, hi)
    return best_q, best_pe, best_re, iters, conv, stalled, best_J


# ====================================================================== seeds, wrapping and the rule
def _rng(rng):
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(SEED if rng is None else rng)


def _seeds(model, q0, seeds: int, rng) -> np.ndarray:
    """(S, nq): q0 (when given), home, then uniform draws within the limits."""
    import mujoco
    n = model.nq
    out = []
    if q0 is not None:
        out.append(model.clip(q0))
    out.append(model.home.copy())
    k = max(int(seeds) - len(out), 0)
    if k:
        g = _rng(rng)
        lo, hi = model.lower, model.upper
        fin = np.isfinite(lo) & np.isfinite(hi)
        slide = np.array([int(model.m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_SLIDE) for j in model._jid], bool)
        u = g.random((k, n))
        R = np.where(fin, lo + u * np.where(fin, hi - lo, 0.0),
                     np.where(slide, model.home, model.home + (2 * u - 1) * np.pi))
        out += list(R)
    return np.array(out[:max(int(seeds), 1)], float).reshape(-1, n)


def _revolute_unlimited(model) -> np.ndarray:
    import mujoco
    return np.array([int(model.m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE) for j in model._jid], bool) & \
        ~(np.isfinite(model.lower) & np.isfinite(model.upper))


def _wrap_to(model, Q, ref, mask=None) -> np.ndarray:
    """Unlimited revolute joints (those in mask, default all) wrapped to within π of ref (nq,): whole turns that do
    not change the pose."""
    cont = _revolute_unlimited(model)
    if mask is not None:
        cont = cont & np.asarray(mask, bool)
    if not cont.any():
        return Q
    Q = np.array(Q, float, copy=True)
    r = np.asarray(ref, float)[cont]
    d = Q[..., cont] - r
    Q[..., cont] = r + (d + np.pi) % (2 * np.pi) - np.pi
    return Q


def _into_limits(model, q) -> np.ndarray | None:
    """q with each limited revolute joint shifted by whole turns into its limits, or None when that is impossible."""
    import mujoco
    q = q.copy()
    for k, j in enumerate(model._jid):
        lo, hi = model.lower[k], model.upper[k]
        if lo - 1e-9 <= q[k] <= hi + 1e-9:
            continue
        if int(model.m.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_HINGE) or not (np.isfinite(lo) and np.isfinite(hi)):
            return None
        t = q[k] - 2 * np.pi * np.ceil((q[k] - hi - 1e-9) / (2 * np.pi))
        if lo - 1e-9 <= t <= hi + 1e-9:
            q[k] = min(max(t, lo), hi)
        else:
            return None
    return q


# ====================================================================== the public API
def _mode_of(n_active: int, mode: str) -> bool:
    if mode not in MODES:
        raise ValueError(f'mode must be one of {", ".join(MODES)}; got {mode!r}.')
    return mode == 'position' or (mode == 'auto' and n_active < 6)


def solve_batch(model, link: str, targets, q0=None, seeds: int = 8, rng=None, pos_tol: float = POS_TOL,
                rot_tol: float = ROT_TOL, max_iter: int = MAX_ITER, damping: float = DAMPING,
                weight_rot: float = WEIGHT_ROT, *, mode: str = 'auto', rule: str = 'nearest_home',
                collision: bool = True, margin: float = 0.0, tcp=None, retry_seeds: int | None = None,
                limit_gain: float = LIMIT_GAIN, max_step: float = MAX_STEP, progress=None, cancel=None) -> IKBatch:
    """Solve IK of `link` (a RobotModel link name; tcp: an optional 4 × 4 tool offset in link coordinates) for
    targets (N, 4, 4) world poses, or (N, 3) world positions (the orientation left free; rot_err is then nan).
    q0: (nq,) or (N, nq) warm starts (the first seed), or None. Targets no seed solved are tried again from
    retry_seeds fresh random seeds (default 2 × seeds) before their reason is decided. Returns an IKBatch; see the
    module docstring for the method, the rules and the reasons. progress(f) gets the fraction done; cancel()
    returning True stops early (the targets not reached are 'no convergence')."""
    if rule not in RULES:
        raise ValueError(f'rule must be one of {", ".join(RULES)}; got {rule!r}.')
    chain = _Chain(model, link, tcp)
    n_act = int(chain.active.sum())
    position = _mode_of(n_act, mode)
    raw = np.asarray(targets, float)
    free = raw.shape[-1] == 3 and raw.ndim <= 2
    if free:
        P3 = raw.reshape(-1, 3)
        T = np.tile(np.eye(4), (len(P3), 1, 1))
        T[:, :3, 3] = P3
        position = True
    else:
        try:
            T = raw.reshape(-1, 4, 4)
        except ValueError:
            raise ValueError(f'IK targets must be 4 × 4 poses or 3-vector positions; got an array of shape '
                             f'{raw.shape}.') from None
    N, n = len(T), model.nq
    seeds = max(int(seeds), 1)
    retry = 2 * seeds if retry_seeds is None else max(int(retry_seeds), 0)
    cfg = _Cfg(pos_tol, rot_tol, max_iter, damping, weight_rot, limit_gain, max_step)
    cfg.no_rot = free
    g = _rng(rng)
    Q0 = None if q0 is None else np.asarray(q0, float)
    per_target = Q0 is not None and Q0.ndim == 2
    base_seeds = _seeds(model, None if (Q0 is None or per_target) else Q0, seeds - (1 if per_target else 0), g)
    if per_target:
        if Q0.shape != (N, n):
            raise ValueError(f'Per-target IK starts q0 must have shape ({N}, {n}), one joint vector per target; got '
                             f'{Q0.shape}.')
    elif Q0 is not None and Q0.size != n:
        raise ValueError(f'{model.name} has {n} moving joints; the IK start q0 has {Q0.size} values.')
    hold = model.clip(Q0.reshape(N, n)) if per_target else \
        np.tile(model.home if Q0 is None else model.clip(Q0), (N, 1))
    still = ~chain.active
    out_q = hold.copy()
    out_ok = np.zeros(N, bool)
    out_pe, out_re = np.full(N, np.inf), np.full(N, np.inf)
    out_it = np.zeros(N, int)
    out_reason = ['no convergence'] * N
    out_ook = np.zeros(N, bool)
    out_w = np.full(N, np.nan)
    out_d = np.full(N, np.nan)
    out_sol = np.zeros(N, int)
    stalled_all = np.ones(N, bool)
    best_s = np.full(N, np.inf)
    rows = 3 if position else 6
    stopped = (lambda: cancel is not None and cancel())

    def ref_of(t):
        """The joint vector solutions are measured from (and unlimited revolute joints wrapped to) for target t."""
        if rule == 'nearest_q0' and Q0 is not None:
            return model.clip(Q0[t] if per_target else Q0)
        return model.home

    def score_of(pe, re):
        if position:
            return np.where(pe <= cfg.pt, 0.0, pe / cfg.pt) * 1e3 + re / cfg.rt
        return np.maximum(pe / cfg.pt, re / cfg.rt)

    def run(idx, seeds_of, S, f0, f1):
        """Solve targets idx from seeds_of(idx) -> (len(idx), S, nq), in chunks; record the best per target."""
        per = max(CHUNK // max(S, 1), 1)
        for c0 in range(0, len(idx), per):
            if stopped():
                return
            sub = idx[c0:c0 + per]
            m = len(sub)
            QS = np.array(seeds_of(sub), float)
            QS[:, :, still] = hold[sub][:, None, still]     # joints that do not move the link stay where they were
            bq, pe, re, it, cv, st, BJ = _iterate(chain, np.repeat(T[sub], S, axis=0), QS.reshape(-1, n), cfg,
                                                  position, True, cancel)
            bq, pe, re, it, cv, st = (x.reshape((m, S) + x.shape[1:]) for x in (bq, pe, re, it, cv, st))
            BJ = BJ.reshape(m, S, 6, n)
            for a, t in enumerate(sub):
                # whole turns of the chain's unlimited revolute joints, to within π of the reference (home, or q0
                # for 'nearest_q0', so a drag never jumps a full turn); the held joints keep their values
                bq[a] = _wrap_to(model, bq[a], ref_of(t), chain.active)
                sc = score_of(pe[a], re[a])
                b = int(np.argmin(sc))
                stalled_all[t] &= bool(st[a].all())
                cands = np.flatnonzero(cv[a])
                out_sol[t] += len(cands)
                if sc[b] < best_s[t] and not len(cands):
                    best_s[t] = sc[b]
                    out_q[t], out_pe[t], out_re[t], out_it[t] = bq[a, b], pe[a, b], re[a, b], it[a, b]
                if not len(cands):
                    continue
                if position and not cfg.no_rot:
                    # position-first: the seeds that got the orientation closest (within rot_tol) go first
                    cands = cands[re[a, cands] <= re[a, cands].min() + cfg.rt]
                if rule != 'most_manipulable':
                    dd = bq[a, cands] - ref_of(t)
                    order = cands[np.argsort(np.einsum('ij,ij->i', dd, dd), kind='stable')]
                else:
                    w = np.asarray(yoshikawa(BJ[a, cands, :rows])).reshape(-1)
                    order = cands[np.argsort(-w, kind='stable')]
                chosen, dmin = None, np.nan
                for c in order:
                    if collision:
                        r = _col.self_collision(model, bq[a, c], margin)
                        if r.in_collision:
                            dmin = r.min_distance if np.isnan(dmin) else max(dmin, r.min_distance)
                            continue
                        dmin = r.min_distance
                    chosen = int(c)
                    break
                b = int(order[0]) if chosen is None else chosen
                out_q[t], out_pe[t], out_re[t], out_it[t] = bq[a, b], pe[a, b], re[a, b], it[a, b]
                out_d[t] = dmin
                best_s[t] = -1.0
                if chosen is None:
                    out_reason[t] = 'self-collision'
                    continue
                out_ok[t], out_reason[t] = True, ''
                out_ook[t] = bool(re[a, b] <= cfg.rt)
                out_w[t] = float(yoshikawa(BJ[a, b, :rows]))
            if progress is not None:
                progress(f0 + (f1 - f0) * min(c0 + per, len(idx)) / max(len(idx), 1))

    S1 = len(base_seeds) + (1 if per_target else 0)
    if per_target:
        run(np.arange(N), lambda sub: np.concatenate([Q0[sub][:, None, :], np.broadcast_to(
            base_seeds, (len(sub),) + base_seeds.shape)], axis=1), S1, 0.0, 0.8)
    else:
        run(np.arange(N), lambda sub: np.broadcast_to(base_seeds, (len(sub),) + base_seeds.shape), S1, 0.0, 0.8)
    # ------------------------------------------------------------ fresh seeds for the targets not solved
    fail = np.flatnonzero(~out_ok)
    if len(fail) and retry and not stopped():
        extra = _seeds(model, None, retry + 1, g)[1:]           # without home, which round one had
        if len(extra):
            run(fail, lambda sub: np.broadcast_to(extra, (len(sub),) + extra.shape), len(extra), 0.8, 0.9)
    # ------------------------------------------------------------ why the others failed
    fail = np.flatnonzero(~out_ok & np.array([r == 'no convergence' for r in out_reason], bool))
    if len(fail) and not stopped():
        a0, rmax = chain.reach()
        far = np.linalg.norm(T[fail, :3, 3] - a0, axis=1) > rmax + cfg.pt
        for t in fail[far]:
            out_reason[t] = 'out of reach'
        rest = fail[~far]
        if len(rest):
            home = np.where(still, hold[rest], model.home)
            starts = np.stack([out_q[rest], home], axis=1).reshape(-1, n)
            uq, upe, ure, _ui, ucv, ust, uJ = _iterate(chain, np.repeat(T[rest], 2, axis=0), starts, cfg, position,
                                                       False, cancel)
            uq, ucv, ust = uq.reshape(len(rest), 2, n), ucv.reshape(len(rest), 2), ust.reshape(len(rest), 2)
            upe, ure = upe.reshape(len(rest), 2), ure.reshape(len(rest), 2)
            uJ = uJ.reshape(len(rest), 2, 6, n)
            for a, t in enumerate(rest):
                hits = np.flatnonzero(ucv[a])
                accepted = False
                for b in hits:
                    qi = _into_limits(model, uq[a, b])
                    if qi is None:
                        continue
                    if collision:
                        r = _col.self_collision(model, qi, margin)
                        out_d[t] = r.min_distance
                        if r.in_collision:
                            out_reason[t] = 'self-collision'
                            continue
                    out_q[t], out_pe[t], out_re[t] = _wrap_to(model, qi, ref_of(t), chain.active), upe[a, b], \
                        ure[a, b]
                    out_ok[t], out_reason[t] = True, ''
                    out_ook[t] = bool(ure[a, b] <= cfg.rt)
                    out_w[t] = float(yoshikawa(uJ[a, b, :rows]))
                    out_sol[t] = 1
                    accepted = True
                    break
                if accepted or out_reason[t] == 'self-collision':
                    continue
                if len(hits):
                    out_reason[t] = 'joint limit'
                elif stalled_all[t] and ust[a].all():
                    out_reason[t] = 'out of reach'
        if not position and n_act < 6:
            for t in [t for t in fail if not out_ok[t] and out_reason[t] in ('out of reach', 'no convergence')]:
                pcv = _iterate(chain, T[t][None], out_q[t][None], cfg, True, True)[4]
                if pcv[0]:
                    out_reason[t] = 'orientation unreachable'
    if free:
        out_re[:] = np.nan
        out_ook[:] = True
    if progress is not None:
        progress(1.0)
    settings = {'joints': [nm for nm, a in zip(model.joint_names, chain.active) if a],
                'seeds': S1, 'retry_seeds': retry, 'pos_tol': cfg.pt, 'rot_tol': cfg.rt, 'max_iter': cfg.max_iter,
                'damping': float(damping), 'weight_rot': cfg.w, 'limit_gain': cfg.k, 'max_step': cfg.max_step,
                'rule': rule, 'mode': 'position' if position else 'pose', 'collision': bool(collision),
                'margin': float(margin), 'free_orientation': bool(free)}
    return IKBatch(out_q, out_ok, out_pe, out_re, out_it, out_reason, out_ook, out_w, out_d, out_sol,
                   'position' if position else 'pose', settings)


def solve(model, link: str, target, q0=None, seeds: int = 8, rng=None, pos_tol: float = POS_TOL,
          rot_tol: float = ROT_TOL, max_iter: int = MAX_ITER, damping: float = DAMPING,
          weight_rot: float = WEIGHT_ROT, **kw) -> IKResult:
    """Solve IK of `link` for one world target: a 4 × 4 pose, or a 3-vector position (orientation free); keyword
    options as solve_batch (mode, rule, collision, margin, tcp, retry_seeds, limit_gain, max_step)."""
    t = np.asarray(target, float)
    t = t.reshape(1, 3) if t.size == 3 else t.reshape(1, 4, 4)
    b = solve_batch(model, link, t, None if q0 is None else np.asarray(q0, float).reshape(-1), seeds, rng, pos_tol,
                    rot_tol, max_iter, damping, weight_rot, **kw)
    return b.result(0)
