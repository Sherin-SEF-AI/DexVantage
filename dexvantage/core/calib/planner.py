"""Hand-eye calibration pose planning: before anything is captured, a set of robot poses (15 to 25 by default, the
number is the user's) that the robot can reach without touching itself or the scene, with the board in the camera's
view, chosen so that the calibration pins the camera's pose X (and the board's Z) as firmly as possible; the
covariance of X and Z the plan predicts, in mm and degrees; and what one more pose would gain.

Ported from Vantage core/observability.py: the Fisher-information model of board corners (each counted corner's two
pixel coordinates with independent noise σ on each, differentiated with respect to the pose parameters, the
information summed over corners and views, the covariance its inverse, 1σ per axis in mm and degrees, a direction
the information does not pin reported as not determined), its corner-counting rules (in the image, not hidden by the
scene, the squares at least min_square_px wide in the image) and its requirement ratio and status words; and the
next-best-view idea of Vantage core/calib_coach.py (candidate poses scored by what their information would add,
duplicates of a view already taken left out). What changed: the unknowns are the hand-eye transforms of
core/calib/handeye.py, the boards are held or laid where the project puts them, and every candidate is a robot pose
found by inverse kinematics.

The model (one camera, one board; exactly one of them on a robot link):
- Unknowns: X (the camera's optical frame in its mount frame: the link eye-in-hand, the world eye-to-hand) and Z
  (the board frame in its mount frame: the world eye-in-hand, the link eye-to-hand), in handeye.py's
  parameterisation T = [R₀ Exp(ω) | t₀ + v] (ω about the frame's own axes, v along its parent's): 12 parameters.
  Joint readings are taken as exact and the intrinsics as known (see Known limits).
- A corner P (board frame) seen at robot pose q: P_c = Rxᵀ (Rcᵀ (M_b(q) Z P − t_c) − t_x), M_c(q) = [Rc | t_c] and
  M_b(q) the mount frames' world poses (identity for a world mount). Its derivatives, written out:
      ∂P_c/∂ω_X = [P_c]ₓ,   ∂P_c/∂v_X = −Rxᵀ,   ∂P_c/∂ω_Z = −G Rz [P]ₓ,   ∂P_c/∂v_Z = G,   G = Rxᵀ Rcᵀ Rb
  and ∂uv/∂P_c by central differences of the camera model's projection (core.optics, any lens model). The view's
  Jacobian J (2n × 12) is their product; its information is Jᵀ J / σ², σ the corner noise (PIXEL_SIGMA, a default).
- The plan's information H = Σ Jᵀ J / σ² over its views, its predicted covariance Σ = H⁻¹ (the Gauss-Newton
  covariance the bundle adjustment reports when its residual σ̂ equals σ).

Which corners count (defaults, each a setting): inside the image EDGE_PX from its border; the camera on the printed
side; the line of sight within MAX_INCIDENCE_DEG of the board's normal; the board's squares (an AprilGrid's tags) at
least MIN_SQUARE_PX wide in the image along both board axes (the foreshortened side, Vantage's rule); an AprilGrid
tag only with all four of its corners; at least MIN_CORNERS ChArUco corners (detect.MIN_POINTS) or
APRILGRID_MIN_TAGS tags; and none hidden by the robot, the task objects or the tables at that pose (a cube map of
distances around the camera's centre, core.occlusion, with the camera's own seat on its link left out by
scenegeom.sensor_occluders' rule). Occlusion is checked for the poses the selection takes (each is checked before it
is accepted, and its hidden corners are taken out of its information), not for every candidate.

Candidates: the camera on a hemisphere around the board's centre, on its printed side: at each distance (from the
camera's horizontal FOV, so the board's longer side spans FILLS of the image width when faced squarely), tilted from
the board's normal by TILTS_DEG towards AZIMUTHS directions, rolled about its optical axis by ROLLS_DEG, looking at the
board's centre. Eye-in-hand the camera's world pose follows from the board's, eye-to-hand the board's from the
camera's; the moving link's target is that pose times the mount transform's inverse, solved by core.robot.ik
solve_batch (full pose, several seeds, self-collision checked with SELF_MARGIN_M) and kept when the solution keeps
OBSTACLE_MARGIN_M from the project's obstacles (core.robot.collision obstacle_distance: workspace objects, tables and
the robot file's static geoms). A link already within the margin at the robot's own pose (a shoulder just above the
table the robot stands on) counts only when a candidate brings it closer still.

Selection: greedy D-optimal. Starting from a vague prior (PRIOR_SIGMA, 1 rad and 1 m on every parameter, so log det
is finite before the views pin every axis; it is left out of the reported covariance), each step adds the candidate
whose information maximises log det(H + J_cᵀ J_c / σ²), the volume of the information ellipsoid (the D-optimal
design criterion of Kiefer and Wolfowitz; Fedorov, Theory of Optimal Experiments, 1972). log det does not depend on
the units of the parameters (a change of units adds a constant), so millimetres and degrees need no weighting. Greedy
selection is not guaranteed optimal; log det is submodular over views, so it keeps a large share of the optimum.
The chosen poses are ordered by a nearest-neighbour tour in joint space from the robot's current pose, so the
robot moves little between them.

Reported: σ of X and Z (x, y, z mm along the parent's axes; rx, ry, rz degrees about the frame's own axes), the 12 × 12
covariance, the camera's worst axis against the requirement (NEED_MM and NEED_DEG, defaults), the status word, per
pose how much the worst axis grows without it, and the best next pose with the σ it would give.

Known limits: one camera and one board, exactly one of them on the robot (multi-camera and two-arm plans are not
planned, though handeye.solve solves them); joint readings exact and the intrinsics known (the bundle adjustment's
optional joint offsets and intrinsics would loosen the prediction); corner noise independent and equal on every
corner, so the predicted σ is a lower bound for detected corners (docs/models/handeye.md measured about 2.5 times
on rendered images); a linearisation at the nominal X and Z; candidates on a fixed grid, the selection greedy;
occlusion by sensor housings and by the board's own holder is not modelled beyond the robot's meshes; the camera
body is not part of the collision check (only the robot's own geoms); IK solutions are chosen nearest the home pose,
not for clearance; the camera's focus range is not checked (makers rarely state it).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

import numpy as np

from .axxb import inv_T

SCHEMA = 'dexvantage.handeye_plan/1'
N_POSES = 20                     # a default (the specification asks for 15-25, user-set)
PIXEL_SIGMA = 0.3                # a default: px 1σ per image axis of a detected corner (Vantage's default)
NEED_MM = 0.5                    # a default: 1σ the camera's position should reach
NEED_DEG = 0.05                  # a default: 1σ its rotation should reach
EDGE_PX = 3.0                    # a default: corners this close to the image border are not counted
MAX_INCIDENCE_DEG = 60.0         # a default: corners seen more obliquely than this are not counted
MIN_SQUARE_PX = 8.0              # a default (Vantage's min_square_px): squares narrower than this are not found
TILTS_DEG = (0.0, 20.0, 35.0, 50.0)          # defaults: the candidates' tilts from the board's normal
AZIMUTHS = 8                     # a default: tilt directions around the normal
ROLLS_DEG = (-90.0, -45.0, 0.0, 45.0, 90.0)  # defaults: rolls about the optical axis
FILLS = (0.8, 0.55, 0.35)        # defaults: the board's longer side over the image width, faced squarely
OBSTACLE_MARGIN_M = 0.01         # a default: the clearance a pose keeps from the project's obstacles
SELF_MARGIN_M = 0.0              # a default: self-collision margin of the IK solutions
IK_SEEDS = 4                     # a default
PRIOR_SIGMA = 1.0                # rad and m: the selection's vague prior (not part of the reported covariance)
OCCLUSION_N = 256                # a default: cube-map face size of the occlusion check
UNDETERMINED = 100.0             # a σ this many times the requirement: not determined (Vantage's rule)
DUP_M = 0.02                     # a default: a candidate this close to a pose taken (camera position) ...
DUP_DEG = 5.0                    # ... and turned less than this from it is a duplicate (Vantage's coach)
CRITERIA = ('weakest', 'd_optimal')
AXES = ('rx', 'ry', 'rz', 'x', 'y', 'z')
SIGMA_KEYS = ('rx_deg', 'ry_deg', 'rz_deg', 'x_mm', 'y_mm', 'z_mm')
ORDER = ['X rx_rad', 'X ry_rad', 'X rz_rad', 'X x_m', 'X y_m', 'X z_m',
         'Z rx_rad', 'Z ry_rad', 'Z rz_rad', 'Z x_m', 'Z y_m', 'Z z_m']
AXIS_WORDS = {'rx': 'rotation about its x axis', 'ry': 'rotation about its y axis', 'rz': 'rotation about its z axis',
              'x': 'position along x', 'y': 'position along y', 'z': 'position along z'}
ASSUMPTIONS = (
    'Predictions from the Fisher information of the planned geometry, not measurements: the camera model and the robot '
    'kinematics are taken as exact, the joint readings as exact, corner noise as independent and equal on every '
    'corner (detected corners are somewhat correlated, so treat the predicted σ as a lower bound), the board as flat '
    'and printed true, and the camera and board where the project puts them (the linearisation point).')
LIMITS = {'n_poses': (3, 60), 'pixel_sigma': (0.01, 10.0), 'need_mm': (0.001, 1000.0), 'need_deg': (0.0001, 30.0),
          'edge_px': (0.0, 200.0), 'max_incidence_deg': (1.0, 89.0), 'min_square_px': (0.5, 200.0),
          'min_corners': (4, 1000), 'obstacle_margin_m': (0.0, 0.5), 'self_margin_m': (0.0, 0.2),
          'ik_seeds': (1, 32), 'occlusion_n': (32, 2048), 'azimuths': (1, 72)}


# =========================================================================== settings
@dataclass
class PlanSettings:
    """Every number is a default the user can change (LIMITS clamps them)."""
    n_poses: int = N_POSES
    pixel_sigma: float = PIXEL_SIGMA
    need_mm: float = NEED_MM
    need_deg: float = NEED_DEG
    edge_px: float = EDGE_PX
    max_incidence_deg: float = MAX_INCIDENCE_DEG
    min_square_px: float = MIN_SQUARE_PX
    min_corners: int = 6
    tilts_deg: tuple = TILTS_DEG
    azimuths: int = AZIMUTHS
    rolls_deg: tuple = ROLLS_DEG
    fills: tuple = FILLS
    distances_m: tuple = ()          # explicit camera-board distances (m) instead of the fills
    obstacle_margin_m: float = OBSTACLE_MARGIN_M
    self_margin_m: float = SELF_MARGIN_M
    ik_seeds: int = IK_SEEDS
    occlusion: bool = True
    occlusion_n: int = OCCLUSION_N
    backend: str = 'auto'            # occlusion backend (core.occlusion)
    criterion: str = 'weakest'       # the coach's and the next-pose rule: 'weakest' or 'd_optimal'

    @classmethod
    def from_dict(cls, d) -> 'PlanSettings':
        d = d if isinstance(d, dict) else {}
        s = cls()
        for f in fields(cls):
            if f.name not in d or d[f.name] is None:
                continue
            v = d[f.name]
            cur = getattr(s, f.name)
            try:
                if isinstance(cur, bool):
                    v = bool(v)
                elif isinstance(cur, int):
                    v = int(v)
                elif isinstance(cur, float):
                    v = float(v)
                    if not math.isfinite(v):
                        continue
                elif isinstance(cur, tuple):
                    v = tuple(float(x) for x in v)
                elif isinstance(cur, str):
                    v = str(v)
            except (TypeError, ValueError):
                continue
            if f.name in LIMITS and not isinstance(v, tuple):
                lo, hi = LIMITS[f.name]
                v = type(cur)(min(max(v, lo), hi))
            setattr(s, f.name, v)
        if s.criterion not in CRITERIA:
            s.criterion = 'weakest'
        s.tilts_deg = tuple(min(max(t, 0.0), 80.0) for t in s.tilts_deg) or TILTS_DEG
        s.fills = tuple(f for f in s.fills if 0.02 <= f <= 2.0) or FILLS
        s.distances_m = tuple(x for x in s.distances_m if 0.01 <= x <= 20.0)
        return s

    def to_dict(self) -> dict:
        out = {}
        defaults = []
        base = PlanSettings()
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = list(v) if isinstance(v, tuple) else v
            if v == getattr(base, f.name):
                defaults.append(f.name)
        out['defaults'] = defaults
        return out


# =========================================================================== the rig
@dataclass
class Rig:
    """One camera and one board, one of them on a robot link. X: the camera's optical frame in its mount frame (the
    link when on the robot, else the world); Z: the board frame in its mount frame. q_hold: the joints the plan does
    not move (the other joints of the robot), default its home."""
    mode: str
    model: object
    link: str
    camera: object
    X: np.ndarray
    Z: np.ndarray
    board: object
    robot_uid: str = ''
    project: object = None
    sensor: object = None
    q_hold: np.ndarray | None = None
    camera_name: str = 'camera'
    board_name: str = 'board'
    imager: str = ''
    board_uid: str = ''
    board_default: bool = False
    notes: list = field(default_factory=list)

    def __post_init__(self):
        self.X = np.asarray(self.X, float)
        self.Z = np.asarray(self.Z, float)
        self.points = np.asarray(self.board.points(), float)
        if self.q_hold is None:
            self.q_hold = np.asarray(self.model.home, float).copy()
        from ..robot.manipulability import chain_mask
        self.chain = chain_mask(self.model, self.link)
        kind = getattr(self.board, 'kind', 'charuco')
        # the printed face's outward normal in the board frame (ChArUco z points into the board)
        self.out = np.array([0.0, 0.0, 1.0]) if kind == 'aprilgrid' else np.array([0.0, 0.0, -1.0])
        if kind == 'aprilgrid':
            n = len(self.points)
            idx = np.arange(n)
            R, C = idx // (2 * self.board.tag_cols), idx % (2 * self.board.tag_cols)
            self.groups = (R // 2) * self.board.tag_cols + C // 2
            self.feature_m = float(self.board.tag_m)
        else:
            self.groups = np.arange(len(self.points))
            self.feature_m = float(self.board.square_m)

    @property
    def eye_in_hand(self) -> bool:
        return self.mode == 'eye_in_hand'

    def mounts(self, q) -> tuple[np.ndarray, np.ndarray]:
        """(M_c, M_b): the camera's and the board's mount frames in the world at joint vector q."""
        L = self.model.link_pose(np.asarray(q, float), self.link)
        return (L, np.eye(4)) if self.eye_in_hand else (np.eye(4), L)

    def camera_world(self, q) -> np.ndarray:
        Mc, _ = self.mounts(q)
        return Mc @ self.X

    def board_world(self, q) -> np.ndarray:
        _, Mb = self.mounts(q)
        return Mb @ self.Z

    def T_cam_board(self, q) -> np.ndarray:
        Mc, Mb = self.mounts(q)
        return inv_T(Mc @ self.X) @ Mb @ self.Z


# =========================================================================== the Fisher information
def _skew(P) -> np.ndarray:
    P = np.asarray(P, float)
    S = np.zeros(P.shape[:-1] + (3, 3))
    S[..., 0, 1], S[..., 0, 2] = -P[..., 2], P[..., 1]
    S[..., 1, 0], S[..., 1, 2] = P[..., 2], -P[..., 0]
    S[..., 2, 0], S[..., 2, 1] = -P[..., 1], P[..., 0]
    return S


def projection_jacobian(camera, Pc, h: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """(uv (n, 2), D (n, 2, 3)): pixels of optical-frame points and ∂uv/∂P by central differences (step h × |P|)."""
    Pc = np.asarray(Pc, float)
    uv, _ = camera.project(Pc)
    s = h * np.maximum(np.linalg.norm(Pc, axis=1), 1e-3)
    D = np.empty((len(Pc), 2, 3))
    for k in range(3):
        d = np.zeros_like(Pc)
        d[:, k] = s
        a, _ = camera.project(Pc + d)
        b, _ = camera.project(Pc - d)
        D[:, :, k] = (a - b) / (2 * s[:, None])
    return uv, D


def view_jacobian(rig: Rig, q, ids) -> tuple[np.ndarray, np.ndarray]:
    """(uv (n, 2), J (2n, 12)) of the board corners `ids` at joint vector q: pixel derivatives with respect to
    (ω_X, v_X, ω_Z, v_Z) (module docstring). Rows u₀, v₀, u₁, v₁, ... as handeye.py orders its residuals."""
    ids = np.asarray(ids, int)
    Mc, Mb = rig.mounts(q)
    X, Z = rig.X, rig.Z
    T = inv_T(Mc @ X) @ Mb @ Z
    P = rig.points[ids]
    Pc = P @ T[:3, :3].T + T[:3, 3]
    Rx, Rz = X[:3, :3], Z[:3, :3]
    G = Rx.T @ Mc[:3, :3].T @ Mb[:3, :3]
    n = len(ids)
    dP = np.zeros((n, 3, 12))
    dP[:, :, 0:3] = _skew(Pc)
    dP[:, :, 3:6] = -Rx.T
    dP[:, :, 6:9] = -np.einsum('ij,njk->nik', G @ Rz, _skew(P))
    dP[:, :, 9:12] = G
    uv, D = projection_jacobian(rig.camera, Pc)
    J = np.einsum('nij,njk->nik', D, dP).reshape(2 * n, 12)
    return uv, J


def information(rig: Rig, views, sigma_px: float) -> np.ndarray:
    """H (12 × 12) of views [(q, ids)] at corner noise sigma_px."""
    H = np.zeros((12, 12))
    for q, ids in views:
        if len(ids):
            _uv, J = view_jacobian(rig, q, ids)
            H += J.T @ J
    return H / float(sigma_px) ** 2


def covariance(H) -> tuple[np.ndarray, bool]:
    """(Σ, determined): H⁻¹, or for a singular H the pseudo-inverse plus 10¹² along its null space (Vantage's rule:
    what the views do not pin is reported as not determined)."""
    H = (np.asarray(H, float) + np.asarray(H, float).T) / 2
    w, V = np.linalg.eigh(H)
    if w.max() <= 0:
        return np.eye(len(H)) * 1e12, False
    tiny = w <= w.max() * 1e-13
    winv = np.where(tiny, 0.0, 1.0 / np.where(tiny, 1.0, w))
    C = (V * winv) @ V.T
    if tiny.any():
        N = V[:, tiny]
        C = C + N @ N.T * 1e12
    return C, not tiny.any()


def sigma_dict(C6) -> dict:
    """1σ of a 6 × 6 block (ω rad, v m): rx, ry, rz degrees and x, y, z mm; beyond 10⁶ inf (not determined)."""
    s = np.sqrt(np.clip(np.diag(np.asarray(C6, float)), 0, None))
    v = [math.degrees(s[0]), math.degrees(s[1]), math.degrees(s[2]), 1000 * s[3], 1000 * s[4], 1000 * s[5]]
    return {k: (float(x) if math.isfinite(x) and x < 1e6 else math.inf) for k, x in zip(SIGMA_KEYS, v)}


def worst(sig: dict, need_mm: float, need_deg: float) -> tuple[float, str]:
    """(the largest σ over its requirement, its key) of a sigma_dict."""
    best = (0.0, '')
    for k, v in sig.items():
        r = v / (need_deg if k.endswith('_deg') else need_mm)
        if not math.isfinite(r):
            return math.inf, k
        if r > best[0]:
            best = (r, k)
    return best


def verdict(ratio: float) -> tuple[str, str]:
    """(STATUS severity, word) of a requirement ratio (Vantage's rule)."""
    if not math.isfinite(ratio) or ratio >= UNDETERMINED:
        return 'critical', 'not determined'
    if ratio <= 1.0 + 1e-9:
        return 'good', 'meets the requirement'
    if ratio <= 2.0:
        return 'warning', 'close'
    return 'serious', 'too loose'


# =========================================================================== which corners count
def counted(rig: Rig, q, st: PlanSettings) -> tuple[np.ndarray, str]:
    """(ids of the corners that count at q, '' or the reason none do) by the rules of the module docstring, before
    occlusion."""
    T = rig.T_cam_board(q)
    P = rig.points
    Pc = P @ T[:3, :3].T + T[:3, 3]
    cam = rig.camera
    uv, ok = cam.project(Pc)
    W, H = cam.width, cam.height
    e = st.edge_px
    with np.errstate(invalid='ignore'):
        inside = ok & (uv[:, 0] >= e) & (uv[:, 0] <= W - 1 - e) & (uv[:, 1] >= e) & (uv[:, 1] <= H - 1 - e)
    if not inside.any():
        return np.zeros(0, int), 'board out of view'
    c_board = inv_T(T)[:3, 3]                         # the camera centre in the board frame
    if (c_board - P[0]) @ rig.out <= 0:
        return np.zeros(0, int), 'board seen from behind'
    n_c = T[:3, :3] @ rig.out
    with np.errstate(invalid='ignore', divide='ignore'):
        cosi = -(Pc @ n_c) / np.linalg.norm(Pc, axis=1)
    inc_ok = cosi >= math.cos(math.radians(st.max_incidence_deg))
    s = rig.feature_m
    ax = Pc + s * T[:3, 0]
    ay = Pc + s * T[:3, 1]
    ua, _ = cam.project(ax)
    ub, _ = cam.project(ay)
    with np.errstate(invalid='ignore'):
        feat = np.minimum(np.linalg.norm(ua - uv, axis=1), np.linalg.norm(ub - uv, axis=1))
        sq_ok = np.nan_to_num(feat) >= st.min_square_px
    use = inside & inc_ok & sq_ok
    ng = int(rig.groups.max()) + 1
    bad = np.bincount(rig.groups, weights=(~use).astype(float), minlength=ng) > 0
    if getattr(rig.board, 'kind', '') == 'aprilgrid':
        use &= ~bad[rig.groups]
        from .boards import APRILGRID_MIN_TAGS
        need = APRILGRID_MIN_TAGS * 4
    else:
        need = st.min_corners
    ids = np.flatnonzero(use)
    if len(ids) >= need:
        return ids, ''
    if not (inside & inc_ok).any():
        return np.zeros(0, int), 'board too oblique'
    if not (inside & sq_ok).any():
        return np.zeros(0, int), 'squares too small in the image'
    return np.zeros(0, int), 'too few corners in view'


def occlusion_visible(rig: Rig, q, ids, st: PlanSettings) -> np.ndarray:
    """(len(ids),) True for the corners nothing of the scene hides at q (module docstring); every one True without a
    project or with st.occlusion off."""
    ids = np.asarray(ids, int)
    if rig.project is None or not st.occlusion or not len(ids):
        return np.ones(len(ids), bool)
    from .. import occlusion as OC
    from .. import scenegeom as SG
    qd = {rig.robot_uid: np.asarray(q, float)}
    occ = SG.scene_occluders(rig.project, qd)
    origin = rig.camera_world(q)[:3, 3]
    if rig.sensor is not None:
        occ = SG.sensor_occluders(rig.project, rig.sensor, qd, base=occ, origins=origin[None])
    if not len(occ):
        return np.ones(len(ids), bool)
    cube = OC.CACHE.get(occ, origin, st.occlusion_n, st.backend)
    Bw = rig.board_world(q)
    Pw = rig.points[ids] @ Bw[:3, :3].T + Bw[:3, 3]
    return np.asarray(cube.visible(Pw), bool)


# =========================================================================== candidates
def _ik_mode(rig: Rig) -> str:
    """'pose' for a chain of six or more joints; 'auto' (position first, the orientation as close as the joints allow,
    core/robot/ik.py) for a 5-joint arm such as the SO-100, which cannot set every view's roll and tilt exactly. The
    candidate is then judged at the pose the arm reaches: its corners in view and its information come from that q."""
    n = int(np.count_nonzero(rig.chain)) if getattr(rig, 'chain', None) is not None else 6
    return 'pose' if n >= 6 else 'auto'


def view_poses(rig: Rig, st: PlanSettings) -> list:
    """[(T_board_cam 4 × 4, meta)] on the hemisphere of the module docstring."""
    from .synthetic import board_centre
    bc = board_centre(rig.board)
    w, h = rig.board.size_m
    side = max(w, h)
    if st.distances_m:
        dists = list(st.distances_m)
    else:
        hfov = math.radians(min(max(rig.camera.fov_deg()['h'], 5.0), 150.0))
        d_ref = side / (2 * math.tan(hfov / 2))
        dists = [d_ref / f for f in st.fills]
    out_n = rig.out
    a = np.array([1.0, 0.0, 0.0])
    b = np.cross(out_n, a)
    res = []
    for d in dists:
        for tilt in st.tilts_deg:
            az_list = [0.0] if tilt == 0 else [360.0 * k / st.azimuths for k in range(st.azimuths)]
            for az in az_list:
                t, p = math.radians(tilt), math.radians(az)
                dirn = math.sin(t) * (math.cos(p) * a + math.sin(p) * b) + math.cos(t) * out_n
                c = bc + d * dirn
                z = -dirn
                x0 = a - (a @ z) * z
                if np.linalg.norm(x0) < 1e-6:
                    x0 = np.array([0.0, 1.0, 0.0]) - z[1] * z
                x0 /= np.linalg.norm(x0)
                for roll in st.rolls_deg:
                    r = math.radians(roll)
                    x = math.cos(r) * x0 + math.sin(r) * np.cross(z, x0)
                    y = np.cross(z, x)
                    T = np.eye(4)
                    T[:3, :3] = np.stack([x, y, z], 1)
                    T[:3, 3] = c
                    res.append((T, {'distance_m': float(d), 'tilt_deg': float(tilt), 'azimuth_deg': float(az),
                                    'roll_deg': float(roll)}))
    return res


@dataclass
class Candidates:
    """The feasible candidate poses: q (C, nq), ids (counted corners, before occlusion), A (C, 12, 12) information
    per view at σ = 1 px, meta per candidate, and the rejection tally."""
    q: np.ndarray
    ids: list
    A: np.ndarray
    meta: list
    rejected: dict
    total: int
    checked: dict = field(default_factory=dict)      # index -> occlusion-checked ids

    def __len__(self):
        return len(self.ids)


def candidates(rig: Rig, st: PlanSettings | None = None, progress=None, cancel=None) -> Candidates:
    """Every candidate pose of the hemisphere that the robot reaches without a collision with the board in view."""
    from ..robot import collision as CO
    from ..robot import ik
    st = st or PlanSettings()
    rel = view_poses(rig, st)
    targets = []
    for Tbc, _m in rel:
        if rig.eye_in_hand:                   # the board is static: the camera's world pose, then the link's
            targets.append(rig.Z @ Tbc @ inv_T(rig.X))
        else:                                 # the camera is static: the board's world pose, then the link's
            targets.append(rig.X @ inv_T(Tbc) @ inv_T(rig.Z))
    rejected: dict = {}

    def tally(why):
        rejected[why] = rejected.get(why, 0) + 1
    if progress:
        progress(0.05, f'Inverse kinematics of {len(targets)} candidate poses')
    ikb = ik.solve_batch(rig.model, rig.link, np.array(targets), q0=rig.q_hold, seeds=st.ik_seeds,
                         rng=np.random.default_rng(0), mode=_ik_mode(rig), collision=True, margin=st.self_margin_m,
                         progress=(lambda f: progress(0.05 + 0.6 * f, 'Inverse kinematics')) if progress else None,
                         cancel=cancel)
    Q, ids_l, A, meta = [], [], [], []
    check_obs = rig.project is not None and bool(rig.robot_uid)
    base = {}
    if check_obs:
        # pairs already within the margin at the robot's own pose (a shoulder just above the table it stands on)
        # count only when a candidate brings them closer still
        r0 = CO.obstacle_distance(rig.project, rig.robot_uid, rig.q_hold, st.obstacle_margin_m, model=rig.model)
        if r0.in_collision:
            base = {(a, b): d for a, b, d in r0.pairs}
    for i, (Tbc, m) in enumerate(rel):
        if cancel is not None and cancel():
            break
        if not ikb.ok[i]:
            tally(ikb.reason[i] or 'no convergence')
            continue
        q = np.where(rig.chain, ikb.q[i], rig.q_hold)
        if check_obs:
            r = CO.obstacle_distance(rig.project, rig.robot_uid, q, st.obstacle_margin_m, model=rig.model)
            hit = [(a, b, d) for a, b, d in r.pairs if d < st.obstacle_margin_m and d < base.get((a, b), 1e9) - 1e-3]
            if r.in_collision and hit:
                tally('too close to ' + hit[0][1])
                continue
        ids, why = counted(rig, q, st)
        if why:
            tally(why)
            continue
        _uv, J = view_jacobian(rig, q, ids)
        Q.append(q)
        ids_l.append(ids)
        A.append(J.T @ J)
        meta.append(dict(m, ik_pos_err_mm=float(1000 * ikb.pos_err[i]), ik_rot_err_deg=math.degrees(ikb.rot_err[i])))
        if progress and i % 25 == 0:
            progress(0.65 + 0.2 * i / max(len(rel), 1), 'Corners in view')
    nq = rig.model.nq
    return Candidates(np.array(Q).reshape(-1, nq), ids_l, np.array(A).reshape(-1, 12, 12), meta, rejected, len(rel))


def _check(rig: Rig, cs: Candidates, c: int, st: PlanSettings) -> bool:
    """Occlusion-check candidate c once: drop its hidden corners from its information; False when too few stay."""
    if c in cs.checked:
        return len(cs.checked[c]) > 0
    ids = cs.ids[c]
    vis = occlusion_visible(rig, cs.q[c], ids, st)
    hidden = int((~vis).sum())
    cs.meta[c]['hidden'] = hidden
    if hidden:
        keep = ids[vis]
        if getattr(rig.board, 'kind', '') == 'aprilgrid':
            g = rig.groups[ids]
            bad = np.unique(g[~vis])
            keep = ids[~np.isin(g, bad)]
            from .boards import APRILGRID_MIN_TAGS
            need = 4 * APRILGRID_MIN_TAGS
        else:
            need = st.min_corners
        if len(keep) < need:
            cs.checked[c] = np.zeros(0, int)
            cs.A[c] = 0.0
            return False
        _uv, J = view_jacobian(rig, cs.q[c], keep)
        cs.A[c] = J.T @ J
        cs.ids[c] = keep
    cs.checked[c] = cs.ids[c]
    return True


# =========================================================================== selection
def _logdet(M) -> np.ndarray:
    sign, ld = np.linalg.slogdet(M)
    return np.where(sign > 0, ld, -np.inf)


def _worst_ratios(Hs: np.ndarray, st: PlanSettings) -> np.ndarray:
    """(K,) the camera's worst σ over its requirement for each information matrix in Hs (K, 12, 12)."""
    try:
        C = np.linalg.inv(Hs)
    except np.linalg.LinAlgError:
        C = np.stack([covariance(h)[0] for h in Hs])
    d = np.sqrt(np.clip(np.einsum('kii->ki', C)[:, :6], 0, None))
    need = np.r_[np.full(3, math.radians(st.need_deg)), np.full(3, st.need_mm / 1000.0)]
    r = (d / need).max(axis=1)
    bad = ~np.isfinite(r) | (np.linalg.cond(Hs) > 1e13)
    return np.where(bad, np.inf, r)


def rank(H: np.ndarray, cs: Candidates, st: PlanSettings, mask=None, criterion: str = 'd_optimal') -> np.ndarray:
    """Candidate indices best first by criterion: 'd_optimal' (largest log det of H + A_c / σ², with the vague prior)
    or 'weakest' (the smallest worst-axis ratio of the camera after adding it; D-optimal among ties and while the
    views do not pin every axis)."""
    if not len(cs):
        return np.zeros(0, int)
    prior = np.eye(12) / PRIOR_SIGMA ** 2
    Hs = H[None] + prior[None] + cs.A / st.pixel_sigma ** 2
    ld = _logdet(Hs)
    if mask is not None:
        ld = np.where(mask, ld, -np.inf)
    if criterion == 'weakest':
        r = _worst_ratios(H[None] + cs.A / st.pixel_sigma ** 2, st)
        if mask is not None:
            r = np.where(mask, r, np.inf)
        if np.isfinite(r).any():
            order = np.lexsort((-ld, np.round(r, 9)))
            return order[np.isfinite(ld[order])]
    order = np.argsort(-ld, kind='stable')
    return order[np.isfinite(ld[order])]


def select(rig: Rig, cs: Candidates, n: int, st: PlanSettings, H0=None, taken_mask=None, progress=None,
           cancel=None) -> tuple[list, np.ndarray]:
    """(chosen indices, H of the views alone): greedy D-optimal (module docstring), each pick occlusion-checked."""
    H = np.zeros((12, 12)) if H0 is None else np.array(H0, float)
    avail = np.ones(len(cs), bool) if taken_mask is None else ~np.asarray(taken_mask, bool)
    chosen: list = []
    guard = 0
    while len(chosen) < n and avail.any() and guard < 20 * n + len(cs):
        guard += 1
        if cancel is not None and cancel():
            break
        order = rank(H, cs, st, avail, 'd_optimal')
        if not len(order):
            break
        c = int(order[0])
        if c not in cs.checked:
            if not _check(rig, cs, c, st):
                avail[c] = False
                cs.meta[c]['rejected'] = 'hidden by the scene'
            continue                         # scored again with its visible corners
        chosen.append(c)
        avail[c] = False
        H = H + cs.A[c] / st.pixel_sigma ** 2
        if progress:
            progress(0.85 + 0.1 * len(chosen) / max(n, 1), f'Pose {len(chosen)} of {n}')
    return chosen, H


def next_best(rig: Rig, cs: Candidates, H: np.ndarray, st: PlanSettings, avail, criterion: str | None = None,
              tries: int = 6) -> int | None:
    """The best candidate to add to H by st.criterion (occlusion-checked), or None."""
    crit = criterion or st.criterion
    avail = np.array(avail, bool)
    for _ in range(max(tries, 1) * 4):
        order = rank(H, cs, st, avail, crit)
        if not len(order):
            return None
        c = int(order[0])
        if c in cs.checked and len(cs.checked[c]):
            return c
        if not _check(rig, cs, c, st):
            avail[c] = False
    return None


def tour(Q: np.ndarray, start, chain) -> list:
    """A nearest-neighbour order of joint vectors Q from `start` over the chain joints."""
    left = list(range(len(Q)))
    cur = np.asarray(start, float)
    out = []
    while left:
        d = [float(np.linalg.norm((Q[i] - cur)[chain])) for i in left]
        k = left.pop(int(np.argmin(d)))
        out.append(k)
        cur = Q[k]
    return out


# =========================================================================== the plan
def _pose(T) -> dict:
    from ..frames import Pose
    return Pose.from_matrix(np.asarray(T, float)).to_dict()


def _r(v, nd=4):
    if v is None:
        return None
    v = float(v)
    return round(v, nd) if math.isfinite(v) else None


def _sig_doc(sig: dict) -> dict:
    return {k: _r(v, 5) for k, v in sig.items()}


def sigmas(H: np.ndarray) -> tuple[dict, dict, np.ndarray, bool]:
    C, det = covariance(H)
    return sigma_dict(C[:6, :6]), sigma_dict(C[6:, 6:]), C, det


def axis_text(key: str, what: str) -> str:
    ax = key.split('_')[0]
    return f'{what} {AXIS_WORDS.get(ax, ax)}'


def fmt_sigma(key: str, v: float) -> str:
    if not math.isfinite(v):
        return 'not determined'
    return f'{v:.3f} mm' if key.endswith('_mm') else f'{v:.4f}°'


def gain_doc(rig: Rig, cs: Candidates, H: np.ndarray, c: int, st: PlanSettings) -> dict:
    """What adding candidate c to H gives: σ before and after, the worst ratio before and after and its gain."""
    sx0, sz0, _C, _d = sigmas(H)
    H1 = H + cs.A[c] / st.pixel_sigma ** 2
    sx1, sz1, _C1, _d1 = sigmas(H1)
    r0, k0 = worst(sx0, st.need_mm, st.need_deg)
    r1, k1 = worst(sx1, st.need_mm, st.need_deg)
    pct = None
    if math.isfinite(r0) and r0 > 0 and math.isfinite(r1):
        pct = 100.0 * (1 - r1 / r0)
    per = {}
    for k in SIGMA_KEYS:
        a, b = sx0[k], sx1[k]
        per[k] = _r(100.0 * (1 - b / a), 2) if math.isfinite(a) and a > 0 and math.isfinite(b) else None
    return {'index': int(c), 'q': [float(v) for v in cs.q[c]], 'meta': cs.meta[c], 'corners': int(len(cs.ids[c])),
            'sigma_X_before': _sig_doc(sx0), 'sigma_X_after': _sig_doc(sx1), 'sigma_Z_after': _sig_doc(sz1),
            'worst_before': _r(r0), 'worst_axis_before': k0, 'worst_after': _r(r1), 'worst_axis_after': k1,
            'worst_gain_pct': _r(pct, 2), 'gain_pct': per,
            'logdet_gain': _r(float(_logdet((H1 + np.eye(12) / PRIOR_SIGMA ** 2)[None])[0] -
                                    _logdet((H + np.eye(12) / PRIOR_SIGMA ** 2)[None])[0]), 4),
            'camera_world': _pose(rig.camera_world(cs.q[c])), 'link_world': _pose(rig.model.link_pose(cs.q[c],
                                                                                                        rig.link))}


def describe_view(meta: dict) -> str:
    """A candidate's geometry in words."""
    t = meta.get('tilt_deg', 0.0)
    s = f'{meta.get("distance_m", 0.0):.2f} m from the board'
    s += ', facing it squarely' if t == 0 else f', tilted {t:.0f}° (towards {meta.get("azimuth_deg", 0.0):.0f}°)'
    r = meta.get('roll_deg', 0.0)
    if r:
        s += f', rolled {r:+.0f}°'
    return s


@dataclass
class Plan:
    doc: dict
    candidates: Candidates
    H: np.ndarray
    chosen: list


def plan(rig: Rig, st: PlanSettings | None = None, start=None, progress=None, cancel=None) -> Plan:
    """The pose plan (module docstring). start: the robot's current joint vector (the tour starts there; default
    rig.q_hold). Raises ValueError (plain sentence) when no candidate is feasible."""
    st = st or PlanSettings()
    cs = candidates(rig, st, progress, cancel)
    if not len(cs):
        top = ', '.join(f'{k} ({v})' for k, v in sorted(cs.rejected.items(), key=lambda t: -t[1])[:3])
        raise ValueError(f'None of the {cs.total} candidate poses works: {top}. Move the board within reach of the '
                         'robot and in front of the camera, or loosen the settings (tilts, distances, margins).')
    chosen, H = select(rig, cs, int(st.n_poses), st, progress=progress, cancel=cancel)
    start = rig.q_hold if start is None else np.asarray(start, float)
    order = tour(cs.q[chosen], start, rig.chain) if chosen else []
    chosen = [chosen[k] for k in order]
    sx, sz, C, det = sigmas(H)
    ratio, key = worst(sx, st.need_mm, st.need_deg)
    sev, word = verdict(ratio)
    poses = []
    names = list(rig.model.joint_names)
    for k, c in enumerate(chosen):
        H_wo = H - cs.A[c] / st.pixel_sigma ** 2
        r_wo = worst(sigmas(H_wo)[0], st.need_mm, st.need_deg)[0]
        inc = (100.0 * (r_wo / ratio - 1)) if math.isfinite(r_wo) and math.isfinite(ratio) and ratio > 0 else None
        m = cs.meta[c]
        poses.append({'index': k + 1, 'candidate': int(c), 'q': [float(v) for v in cs.q[c]], 'joint_names': names,
                      'corners': int(len(cs.ids[c])), 'ids': [int(i) for i in cs.ids[c]],
                      'hidden': int(m.get('hidden', 0)), 'distance_m': _r(m['distance_m']),
                      'tilt_deg': m['tilt_deg'], 'azimuth_deg': m['azimuth_deg'], 'roll_deg': m['roll_deg'],
                      'what': describe_view(m), 'worst_without': _r(r_wo),
                      'worst_increase_pct': _r(inc, 2) if inc is not None else None,
                      'link_world': _pose(rig.model.link_pose(cs.q[c], rig.link)),
                      'camera_world': _pose(rig.camera_world(cs.q[c])),
                      'board_world': _pose(rig.board_world(cs.q[c]))})
    avail = np.ones(len(cs), bool)
    avail[chosen] = False
    nb = next_best(rig, cs, H, st, avail)
    nxt = gain_doc(rig, cs, H, nb, st) if nb is not None else None
    if nxt is not None:
        nxt['what'] = describe_view(cs.meta[nb])
    unit_need = f'{st.need_mm:g} mm and {st.need_deg:g}°'
    if not math.isfinite(ratio):
        advice = ('The planned poses do not pin every axis of the camera: add poses with rotations about different '
                  'axes, or allow larger tilts.')
    elif ratio <= 1.0:
        advice = (f'{len(chosen)} poses pin the camera within {unit_need} (1σ, predicted at {st.pixel_sigma:g} px '
                  'corner noise).')
    else:
        advice = (f'The camera\'s {AXIS_WORDS[key.split("_")[0]]} is the loosest at {fmt_sigma(key, sx[key])}, '
                  f'{ratio:.1f}× the requirement: add poses, allow larger tilts, or bring the camera closer.')
    if nxt is not None and nxt.get('worst_gain_pct') is not None:
        advice += f' One more pose would cut the worst axis by {nxt["worst_gain_pct"]:.1f} %.'
    notes = list(rig.notes)
    if len(chosen) < st.n_poses:
        notes.append(f'Only {len(chosen)} of the {st.n_poses} poses asked for could be planned: the other candidates '
                     'were unreachable, in collision or did not see the board.')
    if rig.board_default:
        notes.append('The board is at DexVantage\'s default place (a default, not yours): set where you will put it '
                     'on the plan page for a plan that holds.')
    if rig.project is None or not st.occlusion:
        notes.append('Occlusion by the scene was not checked.')
    if _ik_mode(rig) != 'pose':
        notes.append(f'The arm moves {int(np.count_nonzero(rig.chain))} joints to {rig.link}, fewer than the six a '
                     'free camera pose needs: each candidate is reached position first and judged where the camera '
                     'then looks.')
    doc = {
        'mode': rig.mode, 'robot_uid': rig.robot_uid, 'link': rig.link,
        'camera': {'sensor_uid': getattr(rig.sensor, 'uid', ''), 'name': rig.camera_name, 'imager': rig.imager,
                   'model': rig.camera.summary(), 'X_nominal': _pose(rig.X),
                   'frame': ('optical frame in link ' + rig.link) if rig.eye_in_hand else 'optical frame in the world'},
        'board': {'uid': rig.board_uid, 'name': rig.board_name, 'what': rig.board.describe(),
                  'definition': rig.board.to_dict(), 'default': rig.board_default, 'Z_nominal': _pose(rig.Z),
                  'frame': 'board frame in the world' if rig.eye_in_hand else 'board frame in link ' + rig.link,
                  'corners': int(len(rig.points))},
        'poses': poses, 'n_asked': int(st.n_poses), 'n_planned': len(poses),
        'sigma': {'X': _sig_doc(sx), 'Z': _sig_doc(sz)}, 'determined': bool(det),
        'covariance': [[float(v) if abs(v) < 1e11 else None for v in row] for row in C], 'covariance_order': ORDER,
        'worst': {'ratio': _r(ratio), 'key': key, 'value': _r(sx.get(key, math.inf), 5),
                  'axis': AXIS_WORDS.get(key.split('_')[0], key) if key else ''},
        'status': sev, 'word': word, 'advice': advice, 'next': nxt,
        'requirement': {'need_mm': st.need_mm, 'need_deg': st.need_deg,
                        'default': 'need_mm' in st.to_dict()['defaults'] and 'need_deg' in st.to_dict()['defaults']},
        'candidates': {'total': cs.total, 'feasible': len(cs), 'rejected': dict(sorted(cs.rejected.items(),
                                                                                      key=lambda t: -t[1]))},
        'settings': st.to_dict(), 'notes': notes, 'assumptions': ASSUMPTIONS}
    if progress:
        progress(1.0, advice)
    return Plan(doc, cs, H, chosen)


# =========================================================================== from the project
def rig_from_project(project, sensor_uid: str | None = None, board_uid: str | None = None, q=None) -> Rig:
    """The planning rig of a project: the sensor (default: the first camera or depth sensor on a robot link, else the
    first static one), the board record (default: the first of project.calibration's boards, else DexVantage's
    default placement for the mode), the robot that moves. Raises jobs.StudyError (a plain sentence)."""
    from ..jobs import StudyError
    from ..robot.model import RobotModelError, get_model
    from . import records as RC
    v = project.variant
    cams = [s for s in v.sensors if s.kind in ('camera', 'depth') and s.enabled]
    if sensor_uid:
        sensor = v.sensor(sensor_uid)
        if sensor is None:
            raise StudyError('The camera chosen for the hand-eye plan is not in this rig any more: choose another.')
    else:
        sensor = next((s for s in cams if s.mount.on_link), cams[0] if cams else None)
    if sensor is None:
        raise StudyError('Hand-eye planning needs a camera: add one (Sensors ▸ Add camera).')
    try:
        im = RC.sensor_imager(sensor)
    except ValueError as e:
        raise StudyError(f'{sensor.name} cannot be planned: {e}.') from None
    try:
        c_robot, c_link, X = RC.camera_mount(project, sensor, im)
    except (RobotModelError, KeyError) as e:
        raise StudyError(f'{sensor.name} could not be placed: {e}') from None
    mode = 'eye_in_hand' if c_robot else 'eye_to_hand'
    rec = RC.find_board(project, board_uid)
    if board_uid and rec is None:
        raise StudyError('The board chosen for the hand-eye plan is not in the project any more: choose another.')
    notes = []
    if rec is None:
        try:
            rec = RC.default_board_record(project, mode, c_robot)
        except (ValueError, RobotModelError) as e:
            raise StudyError(f'There is no board to plan with and the default could not be placed: {e}.') from None
    try:
        board = RC.board_of(rec)
    except ValueError as e:
        raise StudyError(f'The board "{rec.get("name", "")}" cannot be used: {e}.') from None
    b_robot, b_link, Z = RC.board_mount(rec)
    if bool(c_robot) == bool(b_robot):
        if c_robot:
            raise StudyError(f'{sensor.name} and the board are both on a robot: the planner plans one moving side '
                             '(eye-in-hand: camera on the robot, board fixed; eye-to-hand: the other way round). '
                             'Put the board in the world for eye-in-hand.')
        raise StudyError(f'{sensor.name} and the board are both fixed, so moving the robot tells nothing: hold the '
                         'board on the robot (eye-to-hand) or mount the camera on it (eye-in-hand).')
    robot_uid, link = (c_robot, c_link) if c_robot else (b_robot, b_link)
    if project.robot(robot_uid) is None:
        raise StudyError('The robot the calibration moves is not in the project.')
    try:
        mdl = get_model(project, robot_uid)
    except RobotModelError as e:
        raise StudyError(str(e)) from None
    if link not in mdl.link_names:
        raise StudyError(f'{link} is not a link of {mdl.name}.')
    hold = mdl.home.copy()
    if q is not None and robot_uid in q and len(np.asarray(q[robot_uid]).reshape(-1)) == mdl.nq:
        hold = mdl.clip(np.asarray(q[robot_uid], float))
    return Rig(mode, mdl, link, im.camera, X, Z, board, robot_uid, project, sensor, hold,
               camera_name=sensor.name, board_name=str(rec.get('name') or 'board'), imager=im.name,
               board_uid=str(rec.get('uid') or ''), board_default=bool(rec.get('default')), notes=notes)
