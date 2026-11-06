"""Hand-eye calibration: eye-in-hand (AX = XB), eye-to-hand and robot-world/hand-eye (AX = ZB), and one sparse
bundle adjustment over every board corner of every view.

The problem (Problem): cameras, boards and robots, and per robot pose p the joint encoder readings q_p of every
robot. A camera is mounted either in the world (static) or on a robot link; so is a board. A view (Observation) is
one camera's detection of one board at one pose: corner ids and pixels. The unknowns are
- X_c, the pose of camera c's optical frame in its mount frame (the link, or the world for a static camera),
- Z_b, the pose of board b's frame in its mount frame (the world, or the link for a board held by the robot),
- optionally camera c's intrinsics (fx, fy, cx, cy and the lens model's distortion terms),
- optionally joint offsets δ_r (encoder zero errors) for the joints that move the mount links of robot r,
- optionally per-pose joint corrections ε_p,r, when the encoders are given a noise σ_q (the readings then enter as
  measurements q_p + ε with the prior ε ~ N(0, σ_q²)).
The model of a corner k seen in view j (camera c, board b, pose p):

    M(link, p) = FK_world_link(q_p + δ + ε_p)        (identity for a world mount; FK through core.robot RobotModel)
    T_cam_board = (M_c(p) X_c)⁻¹ M_b(p) Z_b,         uv_k = project_c(T_cam_board · P_k)

and the bundle adjustment minimises Σ |uv_k − uv_detected|² / σ_px² + Σ |δ|² / σ_δ² + Σ |ε|² / σ_q² with
scipy.optimize.least_squares. Each transform is updated as T = [R₀ Exp(ω) | t₀ + v] (ω about the transform's own
axes, v along its parent's). The Jacobian is formed by central differences over column groups that share no row
(the sparse pattern: a view's rows depend only on its camera's X and intrinsics, its board's Z, and the joints of its
pose), so per-pose joint columns cost one evaluation per joint for all poses together. The result is re-linearised
at the optimum, and the covariance is the Gauss-Newton (JᵀJ)⁻¹ with the pixel rows weighted by the residual
variance σ̂² = Σ r² / (m − n) (m pixel residuals, n parameters); priors keep their stated σ. σ of X and Z are
reported in mm (along the parent's axes) and degrees (about the frame's own axes), with the full 6 × 6 blocks.

Initial values: every view's board pose by PnP (detect.solve_pnp), then for each camera-board pair the closed-form
solvers of axxb.py (OpenCV's calibrateHandEye and calibrateRobotWorldHandEye when installed, NumPy otherwise): the
five AX = XB methods when one of the two is static, Shah and Li always. With initialiser 'best' (the default), every
applicable method is tried and the one with the smallest pixel RMS starts the adjustment; their results are reported
so they can be compared.

Outlier rejection (a default rule, ported from Vantage core/calib.py calibrate): after a solve, a view whose RMS
exceeds max(OUTLIER_K × the median view RMS, OUTLIER_FLOOR_PX) is dropped and the problem solved again, up to
max_rounds times, as long as MIN_VIEWS views per camera remain.

Known limits: the robots' base poses (Robot.base) are taken as exact (a second arm's base is not estimated); joint
offsets that the views cannot tell apart from X or Z (the first joint's offset with a world board, the last joint's
with a wrist camera, about the same axis) are held by their prior and reported "not determined"; link lengths and
joint-axis errors are not modelled (only zero offsets); corner noise is taken as independent and equal on every
corner; time offsets between images and joint readings are not modelled (capture still poses); the covariance is a
linearisation, checked against a Monte Carlo spread by validation/checks_handeye.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import axxb
from .axxb import SolverError, T_of, exp_so3, inv_T, log_so3, mean_transform

SCHEMA = 'dexvantage.handeye/1'
OUTLIER_K = 3.0                  # a default (Vantage's calibrate rule): × the median view RMS
OUTLIER_FLOOR_PX = 1.0           # a default: px, a view below this is never an outlier
MIN_VIEWS = 3                    # views per camera the outlier rejection keeps
PIXEL_SIGMA_START = 0.5          # a default: px, the corner σ of the first solve (then estimated from the residuals)
JOINT_OFFSET_PRIOR_DEG = 1.0     # a default: σ of the joint-offset prior (bounds the offsets the views cannot see)
DETERMINED_RATIO = 0.5           # a default: an offset whose σ is below this share of its prior σ is "determined"
MODES = ('eye_in_hand', 'eye_to_hand', 'general')


# =========================================================================== problem records
@dataclass
class CameraSpec:
    """A camera: its lens model (core.optics.CameraModel), its mount (robot key and link, or '' for the world), an
    optional initial X (4 × 4, optical frame in the mount frame) and whether to estimate its intrinsics."""
    name: str
    camera: object
    robot: str = ''
    link: str = ''
    T0: np.ndarray | None = None
    intrinsics: bool = False

    @property
    def static(self) -> bool:
        return not self.robot


@dataclass
class BoardSpec:
    """A board: its points (N, 3) in the board frame, its mount (robot key and link, or '' for the world) and an
    optional initial Z (4 × 4, board frame in the mount frame). board: the boards.py record, for the report."""
    name: str
    points: np.ndarray
    robot: str = ''
    link: str = ''
    T0: np.ndarray | None = None
    board: object = None

    @property
    def static(self) -> bool:
        return not self.robot


@dataclass
class Observation:
    """One camera's view of one board at robot pose `pose`: point ids into the board's points and pixels (N, 2)."""
    camera: str
    board: str
    pose: int
    ids: np.ndarray
    uv: np.ndarray
    name: str = ''


@dataclass
class Problem:
    cameras: list
    boards: list
    robots: dict                    # key -> RobotModel (fk_array(q), link_index(name), nq, joint_names)
    joints: list                    # per pose: {robot key: (nq,) encoder readings, rad / m}
    observations: list

    def camera(self, name) -> CameraSpec:
        return next(c for c in self.cameras if c.name == name)

    def board(self, name) -> BoardSpec:
        return next(b for b in self.boards if b.name == name)

    def mode(self) -> str:
        """'eye_in_hand' (every camera on a link, every board static), 'eye_to_hand' (every camera static, every
        board on a link) or 'general'."""
        if all(not c.static for c in self.cameras) and all(b.static for b in self.boards):
            return 'eye_in_hand'
        if all(c.static for c in self.cameras) and all(not b.static for b in self.boards):
            return 'eye_to_hand'
        return 'general'


@dataclass
class Settings:
    """Solver settings; every number is a default the user can change."""
    joint_offsets: tuple = ()               # robot keys whose joint zero offsets are estimated
    joint_offset_prior_deg: float = JOINT_OFFSET_PRIOR_DEG
    encoder_sigma_deg: float | None = None  # None: joint readings exact; else per-pose corrections with this σ
    initialiser: str = 'best'               # 'best' or one of axxb's method names
    backend: str = 'auto'                   # closed-form solvers: 'auto' (OpenCV when installed), 'opencv', 'numpy'
    outlier_k: float = OUTLIER_K
    outlier_floor_px: float = OUTLIER_FLOOR_PX
    max_rounds: int = 5
    pixel_sigma: float | None = None        # None: estimated from the residuals


# =========================================================================== kinematics helper
def _chain_mask(model, link: str) -> np.ndarray:
    try:
        from ..robot.manipulability import chain_mask
        return chain_mask(model, link)
    except Exception:                       # noqa: BLE001 — a model without MuJoCo internals: every joint
        return np.ones(model.nq, bool)


class _Kin:
    """World poses of the mount links at each pose, cached by joint vector."""

    def __init__(self, problem: Problem):
        self.p = problem
        self._cache: dict = {}
        self.links: dict = {}                # robot -> sorted list of links used
        for s in list(problem.cameras) + list(problem.boards):
            if s.robot:
                if s.robot not in problem.robots:
                    raise SolverError(f'{s.name} is mounted on robot "{s.robot}", which is not in the problem')
                self.links.setdefault(s.robot, set()).add(s.link)
        self.links = {r: sorted(v) for r, v in self.links.items()}
        self.lidx = {r: {ln: problem.robots[r].link_index(ln) for ln in v} for r, v in self.links.items()}
        self.chain = {}
        for r, links in self.links.items():
            m = np.zeros(problem.robots[r].nq, bool)
            for ln in links:
                m |= _chain_mask(problem.robots[r], ln)
            self.chain[r] = np.flatnonzero(m)

    def pose(self, robot: str, link: str, q) -> np.ndarray:
        key = (robot, np.asarray(q, float).tobytes())
        T = self._cache.get(key)
        if T is None:
            if len(self._cache) > 200_000:
                self._cache.clear()
            T = self.p.robots[robot].fk_array(q)
            self._cache[key] = T
        return T[self.lidx[robot][link]]


def _mount0(kin: _Kin, spec, p: int) -> np.ndarray:
    """The mount frame's world pose at pose p from the encoder readings (no offsets): identity for a world mount."""
    if not spec.robot:
        return np.eye(4)
    return kin.pose(spec.robot, spec.link, kin.p.joints[p][spec.robot])


# =========================================================================== camera projection with free intrinsics
def _intr_vector(cam) -> np.ndarray:
    return np.r_[cam.fx, cam.fy, cam.cx, cam.cy, np.asarray(cam.dist, float)]


def _project(cam, x, P) -> np.ndarray:
    """uv (N, 2) of optical-frame points through cam's model with intrinsics x = (fx, fy, cx, cy, dist...); NaN
    where the model has no image."""
    from ..optics.lens import _brown, r_of_theta
    fx, fy, cx, cy = x[:4]
    dist = tuple(x[4:])
    X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
    rho = np.hypot(X, Y)
    theta = np.arctan2(rho, Z)
    valid = theta <= cam.max_theta
    with np.errstate(divide='ignore', invalid='ignore'):
        if cam.model in ('pinhole', 'brown'):
            valid &= Z > 0
            zz = np.where(valid, Z, 1.0)
            xn, yn = X / zz, Y / zz
            if cam.model == 'brown':
                xn, yn = _brown(xn, yn, dist)
        else:
            g = r_of_theta(theta, cam.model, dist)
            sc = np.where(rho > 0, g / np.where(rho > 0, rho, 1.0), 0.0)
            xn, yn = X * sc, Y * sc
    uv = np.stack([fx * xn + cx, fy * yn + cy], -1)
    uv[~valid] = np.nan
    return uv


def _camera_with(cam, x):
    from ..optics import CameraModel
    return CameraModel(cam.width, cam.height, float(x[0]), float(x[1]), float(x[2]), float(x[3]), cam.model,
                       tuple(float(v) for v in x[4:]), None)


# =========================================================================== parameter layout
class _Layout:
    def __init__(self, problem: Problem, settings: Settings, kin: _Kin, obs: list):
        self.p, self.s, self.kin = problem, settings, kin
        n = 0
        self.cam = {}
        for c in problem.cameras:
            self.cam[c.name] = slice(n, n + 6)
            n += 6
        self.brd = {}
        for b in problem.boards:
            self.brd[b.name] = slice(n, n + 6)
            n += 6
        self.intr = {}
        for c in problem.cameras:
            if c.intrinsics:
                k = 4 + len(c.camera.dist)
                self.intr[c.name] = slice(n, n + k)
                n += k
        self.off = {}
        for r in settings.joint_offsets:
            if r in kin.chain:
                k = len(kin.chain[r])
                self.off[r] = slice(n, n + k)
                n += k
        self.enc = {}
        poses = sorted({o.pose for o in obs})
        if settings.encoder_sigma_deg:
            for p in poses:
                for r in kin.chain:
                    k = len(kin.chain[r])
                    self.enc[(p, r)] = slice(n, n + k)
                    n += k
        self.n = n
        # residual rows
        self.rows = []
        m = 0
        for o in obs:
            self.rows.append(slice(m, m + 2 * len(o.ids)))
            m += 2 * len(o.ids)
        self.m_pix = m
        self.prior_rows = {}
        for r, sl in self.off.items():
            self.prior_rows[('off', r)] = slice(m, m + sl.stop - sl.start)
            m += sl.stop - sl.start
        for key, sl in self.enc.items():
            self.prior_rows[('enc',) + key] = slice(m, m + sl.stop - sl.start)
            m += sl.stop - sl.start
        self.m = m

    def sparsity(self, obs) -> np.ndarray:
        S = np.zeros((self.m, self.n), bool)
        for o, rs in zip(obs, self.rows):
            c, b = self.p.camera(o.camera), self.p.board(o.board)
            cols = [self.cam[c.name], self.brd[b.name]]
            if c.name in self.intr:
                cols.append(self.intr[c.name])
            for r in {c.robot, b.robot} - {''}:
                if r in self.off:
                    cols.append(self.off[r])
                if (o.pose, r) in self.enc:
                    cols.append(self.enc[(o.pose, r)])
            for sl in cols:
                S[rs, sl] = True
        for key, rs in self.prior_rows.items():
            sl = self.off[key[1]] if key[0] == 'off' else self.enc[key[1:]]
            S[rs, sl] = np.eye(sl.stop - sl.start, dtype=bool)
        return S


def _groups(S: np.ndarray) -> list:
    """Columns grouped greedily so that no two columns of a group share a row."""
    groups, used = [], []
    for j in range(S.shape[1]):
        col = S[:, j]
        for g, u in zip(groups, used):
            if not (u & col).any():
                g.append(j)
                u |= col
                break
        else:
            groups.append([j])
            used.append(col.copy())
    return groups


# =========================================================================== the model
class _Model:
    def __init__(self, problem: Problem, settings: Settings, obs: list, X0: dict, Z0: dict, sigma_px: float):
        self.p, self.s, self.obs = problem, settings, obs
        self.kin = _Kin(problem)
        self.L = _Layout(problem, settings, self.kin, obs)
        self.X0 = {k: np.array(v, float) for k, v in X0.items()}
        self.Z0 = {k: np.array(v, float) for k, v in Z0.items()}
        self.intr0 = {c.name: _intr_vector(c.camera) for c in problem.cameras}
        self.sigma_px = float(sigma_px)
        self.s_off = math.radians(settings.joint_offset_prior_deg)
        self.s_enc = math.radians(settings.encoder_sigma_deg) if settings.encoder_sigma_deg else None
        self.pts = {b.name: np.asarray(b.points, float) for b in problem.boards}
        self.S = self.L.sparsity(obs)
        self.groups = _groups(self.S)
        self.obs_pts = [self.pts[o.board][o.ids] for o in obs]
        self.by_cam = {}
        for j, o in enumerate(obs):
            self.by_cam.setdefault(o.camera, []).append(j)
        self.cam_rows = {c: np.concatenate([np.arange(self.L.rows[j].start, self.L.rows[j].stop) for j in js])
                         for c, js in self.by_cam.items()}
        self.cam_uv = {c: np.concatenate([obs[j].uv for j in js]).ravel() for c, js in self.by_cam.items()}

    @staticmethod
    def apply(T0, d) -> np.ndarray:
        return T_of(T0[:3, :3] @ exp_so3(d[:3]), T0[:3, 3] + d[3:6])

    def transforms(self, x) -> tuple[dict, dict]:
        X = {k: self.apply(T, x[self.L.cam[k]]) for k, T in self.X0.items()}
        Z = {k: self.apply(T, x[self.L.brd[k]]) for k, T in self.Z0.items()}
        return X, Z

    def intrinsics(self, x, name) -> np.ndarray:
        sl = self.L.intr.get(name)
        return self.intr0[name] + (x[sl] if sl is not None else 0.0)

    def joints(self, x, p: int, r: str) -> np.ndarray:
        q = np.array(self.p.joints[p][r], float)
        ch = self.kin.chain[r]
        if r in self.L.off:
            q[ch] += x[self.L.off[r]]
        sl = self.L.enc.get((p, r))
        if sl is not None:
            q[ch] += x[sl]
        return q

    def mount(self, x, spec, p: int) -> np.ndarray:
        if not spec.robot:
            return np.eye(4)
        return self.kin.pose(spec.robot, spec.link, self.joints(x, p, spec.robot))

    def view_residual(self, x, o: Observation, X, Z) -> np.ndarray:
        c, b = self.p.camera(o.camera), self.p.board(o.board)
        T = inv_T(self.mount(x, c, o.pose) @ X[c.name]) @ self.mount(x, b, o.pose) @ Z[b.name]
        P = self.pts[b.name][o.ids] @ T[:3, :3].T + T[:3, 3]
        if c.name in self.L.intr:
            uv = _project(c.camera, self.intrinsics(x, c.name), P)
        else:
            uv = c.camera.project(P)[0]
        r = (uv - o.uv).ravel()
        return r

    def residual(self, x, scaled: bool = True) -> np.ndarray:
        X, Z = self.transforms(x)
        out = np.empty(self.L.m)
        for cname, js in self.by_cam.items():
            c = self.p.camera(cname)
            parts = []
            for j in js:
                o = self.obs[j]
                b = self.p.board(o.board)
                T = inv_T(self.mount(x, c, o.pose) @ X[cname]) @ self.mount(x, b, o.pose) @ Z[b.name]
                parts.append(self.obs_pts[j] @ T[:3, :3].T + T[:3, 3])
            P = np.concatenate(parts)
            uv = _project(c.camera, self.intrinsics(x, cname), P) if cname in self.L.intr else c.camera.project(P)[0]
            r = uv.ravel() - self.cam_uv[cname]
            out[self.cam_rows[cname]] = np.where(np.isfinite(r), r, 0.0) / (self.sigma_px if scaled else 1.0)
        for key, rs in self.L.prior_rows.items():
            if key[0] == 'off':
                out[rs] = x[self.L.off[key[1]]] / self.s_off
            else:
                out[rs] = x[self.L.enc[key[1:]]] / self.s_enc
        return out

    def jacobian(self, x, h: float = 1e-6):
        """Central differences by column group (the sparse pattern), as a dense (m, n) array."""
        J = np.zeros((self.L.m, self.L.n))
        for g in self.groups:
            e = np.zeros(self.L.n)
            e[g] = h
            d = (self.residual(x + e) - self.residual(x - e)) / (2 * h)
            for j in g:
                rows = self.S[:, j]
                J[rows, j] = d[rows]
        return J

    def view_rms(self, x) -> np.ndarray:
        X, Z = self.transforms(x)
        out = []
        for o in self.obs:
            r = self.view_residual(x, o, X, Z).reshape(-1, 2)
            e = np.linalg.norm(r, axis=1)
            out.append(float(np.sqrt(np.mean(np.where(np.isfinite(e), e, 1e6) ** 2))))
        return np.array(out)


def _solve(model: _Model, x0=None):
    from scipy.optimize import least_squares
    x0 = np.zeros(model.L.n) if x0 is None else x0
    jac = (lambda x: model.jacobian(x))
    sol = least_squares(model.residual, x0, jac=jac, method='trf', tr_solver='exact', x_scale='jac',
                        xtol=1e-12, ftol=1e-12, gtol=1e-10, max_nfev=200)
    return sol


# =========================================================================== initialisation
def _pnp_all(problem: Problem, obs: list) -> dict:
    from .detect import solve_pnp
    out = {}
    for j, o in enumerate(obs):
        c, b = problem.camera(o.camera), problem.board(o.board)
        r = solve_pnp(np.asarray(b.points, float)[o.ids], o.uv, c.camera)
        if r is not None:
            out[j] = r
    return out


def _pair_candidates(problem, kin, obs, pnp, cam: CameraSpec, brd: BoardSpec, backend: str) -> list:
    """[(method, X, Z)] from the closed-form solvers for one camera-board pair. With M_c, M_b the mount poses and C
    the PnP board pose: X C = B Z, B = M_c⁻¹ M_b."""
    js = [j for j, o in enumerate(obs) if o.camera == cam.name and o.board == brd.name and j in pnp]
    if len(js) < 3:
        return []
    Mc = [_mount0(kin, cam, obs[j].pose) for j in js]
    Mb = [_mount0(kin, brd, obs[j].pose) for j in js]
    C = [pnp[j].T_cam_board for j in js]
    B = [inv_T(a) @ b for a, b in zip(Mc, Mb)]
    out = []
    if brd.static != cam.static:
        if brd.static:            # eye-in-hand: M_c X C = Z
            G = Mc
        else:                     # eye-to-hand: M_b⁻¹ X C = Z
            G = [inv_T(b) for b in Mb]
        for meth in axxb.HAND_EYE_METHODS:
            try:
                X, _bk = axxb.hand_eye(G, C, meth, backend)
            except Exception:         # noqa: BLE001 — a method that cannot solve these poses is skipped
                continue
            Z = mean_transform([inv_T(b) @ X @ c for b, c in zip(B, C)])
            out.append((meth, X, Z))
    for meth in axxb.ROBOT_WORLD_METHODS:
        try:     # B⁻¹ X = Z C⁻¹: OpenCV's A X = Z B with A = B⁻¹, B = C⁻¹
            X, Z, _bk = axxb.robot_world([inv_T(b) for b in B], [inv_T(c) for c in C], meth, backend)
        except Exception:             # noqa: BLE001 — as above
            continue
        out.append((meth, X, Z))
    return out


def initialise(problem: Problem, obs: list, settings: Settings, pnp: dict | None = None) -> dict:
    """Initial X and Z for every camera and board: {'X': {name: T}, 'Z': {name: T}, 'methods': [...], 'pnp': ...}.
    Raises SolverError when a camera or board is never linked to the others by views."""
    pnp = _pnp_all(problem, obs) if pnp is None else pnp
    kin = _Kin(problem)
    X = {c.name: np.array(c.T0, float) for c in problem.cameras if c.T0 is not None}
    Z = {b.name: np.array(b.T0, float) for b in problem.boards if b.T0 is not None}
    tried = []

    def count(c, b):
        return sum(1 for j, o in enumerate(obs) if o.camera == c and o.board == b and j in pnp)

    for _ in range(len(problem.cameras) + len(problem.boards) + 1):
        if len(X) == len(problem.cameras) and len(Z) == len(problem.boards):
            break
        progress = False
        # a pair with one side known: the other from the PnP poses
        for c in problem.cameras:
            for b in problem.boards:
                js = [j for j, o in enumerate(obs) if o.camera == c.name and o.board == b.name and j in pnp]
                if not js:
                    continue
                if c.name in X and b.name not in Z:
                    Z[b.name] = mean_transform([inv_T(_mount0(kin, b, obs[j].pose)) @
                                                _mount0(kin, c, obs[j].pose) @ X[c.name] @
                                                pnp[j].T_cam_board for j in js])
                    progress = True
                elif b.name in Z and c.name not in X:
                    X[c.name] = mean_transform([inv_T(_mount0(kin, c, obs[j].pose)) @
                                                _mount0(kin, b, obs[j].pose) @ Z[b.name] @
                                                inv_T(pnp[j].T_cam_board) for j in js])
                    progress = True
        if progress:
            continue
        # a closed form needs the pair to move relative to each other: a camera and a board both fixed in the world
        # (or on the same link) start only from the other side once it is known (above)
        pairs = sorted(((count(c.name, b.name), c, b) for c in problem.cameras for b in problem.boards
                        if c.name not in X and b.name not in Z
                        and not (c.static and b.static) and not (c.robot == b.robot and c.link == b.link)),
                       key=lambda t: -t[0])
        if not pairs or pairs[0][0] < 3:
            break
        _n, c, b = pairs[0]
        cands = _pair_candidates(problem, kin, obs, pnp, c, b, settings.backend)
        if not cands:
            raise SolverError(f'No closed-form solver could start {c.name} with {b.name}: the poses need rotations '
                              'about at least two different axes')
        scored = []
        for meth, Xc, Zb in cands:
            e = _pair_rms(kin, obs, c, b, Xc, Zb)
            scored.append((e, meth, Xc, Zb))
            tried.append({'camera': c.name, 'board': b.name, 'method': meth, 'label': axxb.METHOD_LABEL[meth],
                          'rms_px': e, 'X': Xc.tolist(), 'Z': Zb.tolist()})
        want = settings.initialiser
        pick = [s for s in scored if s[1] == want] if want != 'best' else []
        e, meth, Xc, Zb = (pick or sorted(scored, key=lambda s: s[0] if math.isfinite(s[0]) else 1e300))[0]
        X[c.name], Z[b.name] = Xc, Zb
        for t in tried:
            if t['camera'] == c.name and t['board'] == b.name:
                t['chosen'] = t['method'] == meth
    missing = [c.name for c in problem.cameras if c.name not in X] + [b.name for b in problem.boards
                                                                      if b.name not in Z]
    if missing:
        raise SolverError(f'{", ".join(missing)}: not linked to the rest by at least 3 views with a board pose; '
                          'add views in which these cameras see a board that another camera also sees')
    return {'X': X, 'Z': Z, 'methods': tried, 'pnp': pnp}


def _pair_rms(kin, obs, c, b, Xc, Zb) -> float:
    errs = []
    for j, o in enumerate(obs):
        if o.camera != c.name or o.board != b.name:
            continue
        T = inv_T(_mount0(kin, c, o.pose) @ Xc) @ _mount0(kin, b, o.pose) @ Zb
        P = np.asarray(b.points, float)[o.ids] @ T[:3, :3].T + T[:3, 3]
        uv = c.camera.project(P)[0]
        e = np.linalg.norm(uv - o.uv, axis=1)
        errs.append(np.where(np.isfinite(e), e, 1e6))
    if not errs:
        return float('inf')
    e = np.concatenate(errs)
    return float(np.sqrt(np.mean(e ** 2)))


# =========================================================================== result
@dataclass
class HandEyeResult:
    ok: bool
    message: str
    mode: str
    X: dict = field(default_factory=dict)              # camera name -> 4 × 4 (optical frame in its mount frame)
    Z: dict = field(default_factory=dict)              # board name -> 4 × 4 (board frame in its mount frame)
    cov_X: dict = field(default_factory=dict)          # name -> 6 × 6 (ω rad about own axes, v m), the BA order
    cov_Z: dict = field(default_factory=dict)
    intrinsics: dict = field(default_factory=dict)     # camera name -> {'model': CameraModel, 'sigma': [...]}
    joint_offsets: dict = field(default_factory=dict)  # robot -> {'joints': names, 'value_deg', 'sigma_deg', ...}
    views: list = field(default_factory=list)          # per observation: name, camera, board, pose, n, rms, used
    rms_px: float = float('nan')
    sigma_px: float = float('nan')
    dof: int = 0
    rounds: int = 0
    initial: dict = field(default_factory=dict)
    nfev: int = 0
    settings: dict = field(default_factory=dict)
    frames: dict = field(default_factory=dict)

    def sigma(self, name: str) -> dict:
        """σ of X (a camera) or Z (a board): x, y, z in mm and rx, ry, rz in degrees."""
        C = self.cov_X.get(name)
        if C is None:
            C = self.cov_Z[name]
        return sigma_dict(C)

    def to_dict(self) -> dict:
        from ..frames import Pose

        def pose(T):
            return Pose.from_matrix(T).to_dict()
        return {
            'schema': SCHEMA, 'ok': self.ok, 'message': self.message, 'mode': self.mode,
            'cameras': {k: {'pose': pose(T), 'matrix': np.asarray(T).tolist(), 'frame': self.frames.get(k, ''),
                            'sigma': sigma_dict(self.cov_X[k]) if k in self.cov_X else None,
                            'covariance': np.asarray(self.cov_X[k]).tolist() if k in self.cov_X else None}
                        for k, T in self.X.items()},
            'boards': {k: {'pose': pose(T), 'matrix': np.asarray(T).tolist(), 'frame': self.frames.get(k, ''),
                           'sigma': sigma_dict(self.cov_Z[k]) if k in self.cov_Z else None,
                           'covariance': np.asarray(self.cov_Z[k]).tolist() if k in self.cov_Z else None}
                       for k, T in self.Z.items()},
            'covariance_order': ['rx_rad', 'ry_rad', 'rz_rad', 'x_m', 'y_m', 'z_m'],
            'intrinsics': {k: {'model': v['model'].to_dict(), 'sigma': v['sigma'], 'names': v['names']}
                           for k, v in self.intrinsics.items()},
            'joint_offsets': self.joint_offsets, 'views': self.views, 'rms_px': self.rms_px,
            'sigma_px': self.sigma_px, 'dof': self.dof, 'rounds': self.rounds, 'nfev': self.nfev,
            'initial': {'methods': self.initial.get('methods', [])}, 'settings': self.settings}


def sigma_dict(C) -> dict:
    s = np.sqrt(np.clip(np.diag(np.asarray(C, float)), 0, None))
    return {'x_mm': 1000 * s[3], 'y_mm': 1000 * s[4], 'z_mm': 1000 * s[5], 'rx_deg': math.degrees(s[0]),
            'ry_deg': math.degrees(s[1]), 'rz_deg': math.degrees(s[2])}


def _frame_text(spec, problem) -> str:
    what = 'optical frame' if isinstance(spec, CameraSpec) else 'board frame'
    if not spec.robot:
        return f'{spec.name} {what} in the world frame'
    return f'{spec.name} {what} in link {spec.link} of {getattr(problem.robots[spec.robot], "name", spec.robot)}'


# =========================================================================== the solve
def solve(problem: Problem, settings: Settings | None = None, init: dict | None = None,
          progress=None) -> HandEyeResult:
    """The bundle adjustment (module docstring). init: initialise()'s output to start from (computed when None).
    Never raises for a problem it cannot solve: the result says why (ok False, message)."""
    st = settings or Settings()
    mode = problem.mode()
    res = HandEyeResult(False, '', mode, settings=_settings_dict(st))
    try:
        _validate(problem)
        obs = [o for o in problem.observations if len(o.ids) >= 4]
        if init is None:
            init = initialise(problem, obs, st)
        res.initial = init
        used = list(range(len(obs)))
        dropped: dict = {}
        sigma = st.pixel_sigma or PIXEL_SIGMA_START
        x = None
        model = None
        for rnd in range(1, max(1, int(st.max_rounds)) + 1):
            cur = [obs[j] for j in used]
            X0 = dict(init['X']) if model is None else model.transforms(x)[0]
            Z0 = dict(init['Z']) if model is None else model.transforms(x)[1]
            I0 = None if model is None else {c.name: model.intrinsics(x, c.name) for c in problem.cameras}
            off0 = None if model is None else {r: x[sl].copy() for r, sl in model.L.off.items()}
            for _pass in range(3):
                model = _Model(problem, st, cur, X0, Z0, sigma)
                if I0:
                    model.intr0 = I0
                x0 = np.zeros(model.L.n)
                if off0:
                    for r, sl in model.L.off.items():
                        if r in off0:
                            x0[sl] = off0[r]
                sol = _solve(model, x0)
                x = sol.x
                res.nfev += int(sol.nfev)
                r = model.residual(x, scaled=False)[:model.L.m_pix]
                dof = max(model.L.m_pix - model.L.n, 1)
                s_hat = float(np.sqrt(np.sum(r ** 2) / dof))
                if st.pixel_sigma or abs(s_hat / sigma - 1) < 0.05 or s_hat <= 0:
                    break
                X0, Z0 = model.transforms(x)
                I0 = {c.name: model.intrinsics(x, c.name) for c in problem.cameras}
                off0 = {r: x[sl].copy() for r, sl in model.L.off.items()}
                sigma = s_hat
            if progress:
                progress(min(0.95, rnd / max(st.max_rounds, 1)), f'Round {rnd}: {len(cur)} views')
            vr = model.view_rms(x)
            med = float(np.median(vr))
            bad = [k for k, v in enumerate(vr) if v > max(st.outlier_k * med, st.outlier_floor_px)]
            per_cam = {}
            for k, o in enumerate(cur):
                per_cam[o.camera] = per_cam.get(o.camera, 0) + (0 if k in bad else 1)
            res.rounds = rnd
            if not bad or rnd == st.max_rounds or min(per_cam.values(), default=0) < MIN_VIEWS:
                break
            limit = max(st.outlier_k * med, st.outlier_floor_px)
            for k in bad:
                dropped[used[k]] = f'dropped in round {rnd}: {vr[k]:.2f} px (over {limit:.2f} px)'
            used = [j for k, j in enumerate(used) if k not in bad]
        _finish(res, problem, st, model, x, obs, used, dropped, sigma)
    except SolverError as e:
        res.ok, res.message = False, str(e)
    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as e:
        import traceback
        traceback.print_exc()
        res.ok, res.message = False, (f'The hand-eye solve could not finish ({type(e).__name__}: {e}). Check that '
                                      'the views, joint readings and camera models belong together.')
    if progress:
        progress(1.0, res.message)
    return res


def _settings_dict(st: Settings) -> dict:
    return {'joint_offsets': list(st.joint_offsets), 'joint_offset_prior_deg': st.joint_offset_prior_deg,
            'encoder_sigma_deg': st.encoder_sigma_deg, 'initialiser': st.initialiser, 'backend': st.backend,
            'outlier_k': st.outlier_k, 'outlier_floor_px': st.outlier_floor_px, 'max_rounds': st.max_rounds,
            'pixel_sigma': st.pixel_sigma,
            'defaults': {'outlier_k': OUTLIER_K, 'outlier_floor_px': OUTLIER_FLOOR_PX,
                         'joint_offset_prior_deg': JOINT_OFFSET_PRIOR_DEG}}


def _validate(problem: Problem):
    if not problem.cameras:
        raise SolverError('There is no camera in the problem.')
    if not problem.boards:
        raise SolverError('There is no board in the problem.')
    if not problem.observations:
        raise SolverError('There are no views: detect the board in the captured images first.')
    names = [c.name for c in problem.cameras] + [b.name for b in problem.boards]
    if len(set(names)) != len(names):
        raise SolverError('Cameras and boards need distinct names.')
    if all(c.static for c in problem.cameras) and all(b.static for b in problem.boards):
        raise SolverError('Every camera and board is fixed in the world, so the robot motion tells nothing: mount '
                          'a camera or a board on the robot.')
    for o in problem.observations:
        if not 0 <= o.pose < len(problem.joints):
            raise SolverError(f'view {o.name or o.pose} refers to robot pose {o.pose}, but there are '
                              f'{len(problem.joints)} poses')
        if len(o.ids) != len(o.uv):
            raise SolverError(f'view {o.name or o.pose}: {len(o.ids)} corner ids but {len(o.uv)} pixels')


def _finish(res, problem, st, model, x, obs, used, dropped, sigma):
    # re-linearise at the optimum: the transforms become the new origins, the deltas zero
    X, Z = model.transforms(x)
    I0 = {c.name: model.intrinsics(x, c.name) for c in problem.cameras}
    offs = {r: x[sl].copy() for r, sl in model.L.off.items()}
    encs = {k: x[sl].copy() for k, sl in model.L.enc.items()}
    cur = [obs[j] for j in used]
    lin = _Model(problem, st, cur, X, Z, sigma)
    lin.intr0 = I0
    x1 = np.zeros(lin.L.n)
    for r, sl in lin.L.off.items():
        x1[sl] = offs.get(r, 0.0)
    for k, sl in lin.L.enc.items():
        x1[sl] = encs.get(k, 0.0)
    r_pix = lin.residual(x1, scaled=False)[:lin.L.m_pix]
    dof = max(lin.L.m_pix - lin.L.n, 1)
    s_hat = float(np.sqrt(np.sum(r_pix ** 2) / dof)) if st.pixel_sigma is None else float(st.pixel_sigma)
    lin.sigma_px = max(s_hat, 1e-12)
    J = lin.jacobian(x1)
    H = J.T @ J
    w, V = np.linalg.eigh(H)
    tiny = w <= w.max() * 1e-12
    winv = np.where(tiny, 0.0, 1.0 / np.where(tiny, 1.0, w))
    cov = (V * winv) @ V.T
    res.dof = int(dof)
    res.sigma_px = s_hat
    e = np.linalg.norm(r_pix.reshape(-1, 2), axis=1)
    res.rms_px = float(np.sqrt(np.mean(e ** 2)))
    res.X, res.Z = X, Z
    for c in problem.cameras:
        sl = lin.L.cam[c.name]
        res.cov_X[c.name] = cov[sl, sl]
        res.frames[c.name] = _frame_text(c, problem)
    for b in problem.boards:
        sl = lin.L.brd[b.name]
        res.cov_Z[b.name] = cov[sl, sl]
        res.frames[b.name] = _frame_text(b, problem)
    for c in problem.cameras:
        if c.name in lin.L.intr:
            sl = lin.L.intr[c.name]
            names = ['fx', 'fy', 'cx', 'cy'] + [f'd{i + 1}' for i in range(len(c.camera.dist))]
            try:
                cm = _camera_with(c.camera, I0[c.name])
            except ValueError as ex:
                raise SolverError(f'{c.name}: the estimated intrinsics are not a valid lens ({ex})') from None
            res.intrinsics[c.name] = {'model': cm, 'sigma': [float(v) for v in np.sqrt(np.clip(np.diag(
                cov[sl, sl]), 0, None))], 'names': names}
    prior = math.radians(st.joint_offset_prior_deg)
    for r, sl in lin.L.off.items():
        mdl = problem.robots[r]
        ch = lin.kin.chain[r]
        s = np.sqrt(np.clip(np.diag(cov[sl, sl]), 0, None))
        res.joint_offsets[r] = {
            'joints': [mdl.joint_names[i] for i in ch], 'value_deg': [math.degrees(v) for v in offs[r]],
            'sigma_deg': [math.degrees(v) for v in s],
            'determined': [bool(v < DETERMINED_RATIO * prior) for v in s],
            'note': (f'σ near the prior\'s {st.joint_offset_prior_deg:g}° means the views do not determine that '
                     'offset (it trades off against X or Z)')}
    vr = lin.view_rms(x1)
    k = 0
    for j, o in enumerate(obs):
        row = {'name': o.name or f'pose {o.pose}', 'camera': o.camera, 'board': o.board, 'pose': int(o.pose),
               'n': int(len(o.ids)), 'used': j in used}
        if j in used:
            row['rms_px'] = float(vr[k])
            k += 1
        else:
            row['rms_px'] = None
            row['note'] = dropped.get(j, '')
        res.views.append(row)
    for j, o in enumerate(problem.observations):
        if len(o.ids) < 4:
            res.views.append({'name': o.name or f'pose {o.pose}', 'camera': o.camera, 'board': o.board,
                              'pose': int(o.pose), 'n': int(len(o.ids)), 'used': False, 'rms_px': None,
                              'note': 'fewer than 4 corners'})
    n_used = len(used)
    rank_note = '' if not tiny.any() else f' {int(tiny.sum())} direction(s) of the parameters are not determined.'
    res.ok = True
    res.message = (f'{n_used} of {len(obs)} views used · RMS {res.rms_px:.3f} px · corner σ {s_hat:.3f} px'
                   + rank_note)


# =========================================================================== conveniences
def errors_against(res: HandEyeResult, truth: dict) -> dict:
    """{name: (translation error mm, rotation error deg)} of res.X / res.Z against truth {name: 4 × 4}."""
    out = {}
    for k, T in truth.items():
        E = res.X.get(k, res.Z.get(k))
        if E is None:
            continue
        t, r = axxb.transform_error(E, T)
        out[k] = (1000 * t, math.degrees(r))
    return out


def component_errors(T, T_ref) -> np.ndarray:
    """(ω, v) of T against T_ref in the BA's parameterisation (ω about T_ref's own axes, v along the parent's)."""
    return np.r_[log_so3(T_ref[:3, :3].T @ T[:3, :3]), T[:3, 3] - T_ref[:3, 3]]
