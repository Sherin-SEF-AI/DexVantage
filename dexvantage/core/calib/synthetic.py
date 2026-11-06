"""Synthetic calibration data with known ground truth: board images rendered through a camera model, and hand-eye
problems (robot poses, the corners each camera sees, pixel noise, joint offsets) for the checks and the tests.

render_view: every image pixel is supersampled (SUPERSAMPLE × SUPERSAMPLE rays by default, a default), each ray from
CameraModel.unproject is intersected with the board plane, and the board's printed pattern (boards.rasterise at a
resolution finer than one ray's footprint) is sampled bilinearly there; the samples are averaged per pixel, so edges
carry the area coverage a real sensor integrates. Optional Gaussian blur (σ px) and Gaussian noise (grey levels).
The board's surroundings are a flat grey (BACKGROUND, a default); the white margin is part of the print.

simulate: a hand-eye problem from ground-truth mounts. Corners outside the image (an EDGE_PX margin, a default) or
seen at more than MAX_INCIDENCE_DEG off the board normal (a default) are not observed; pixels get N(0, σ²) noise on
each axis; joint readings are the true joints minus the true offsets, plus optional N(0, σ_q²) noise.

sample_poses: random joint vectors around a centre pose, spread ± `spread` rad on the joints that move the link,
kept when every camera sees at least `min_points` corners of a board (so the views are usable).

Known limits: no lens vignetting, motion blur, rolling shutter, defocus or sensor non-linearity; the board is
perfectly flat and printed true; the background has no texture (a detector's false positives are not exercised).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .axxb import exp_so3, inv_T
from .boards import Board, rasterise
from .handeye import BoardSpec, CameraSpec, Observation, Problem

SUPERSAMPLE = 3                 # a default: rays per pixel side
BACKGROUND = 120                # a default: grey level around the board
EDGE_PX = 3.0                   # a default: corners this close to the image edge are not observed
MAX_INCIDENCE_DEG = 70.0        # a default: corners seen more obliquely than this are not observed
TEX_MAX_PX_PER_M = 12000.0      # the board texture's finest resolution (0.083 mm)


_RAYS: dict = {}


def _subpixel_rays(camera, s: int) -> np.ndarray:
    """(H·s·W·s, 3) rays through s × s sub-pixel points per pixel, row-major; cached per camera (the iterative
    Brown-Conrady inverse is the slow part of a render)."""
    key = (camera, s)
    r = _RAYS.get(key)
    if r is None:
        W, H = camera.width, camera.height
        off = (np.arange(s) + 0.5) / s - 0.5
        us = (np.arange(W)[:, None] + off[None, :]).ravel()
        vs = (np.arange(H)[:, None] + off[None, :]).ravel()
        U, V = np.meshgrid(us, vs)
        r = camera.unproject(np.stack([U, V], -1).reshape(-1, 2))
        r.setflags(write=False)
        if len(_RAYS) >= 4:
            _RAYS.pop(next(iter(_RAYS)))
        _RAYS[key] = r
    return r


def render_view(board: Board, camera, T_cam_board, supersample: int = SUPERSAMPLE, noise: float = 0.0,
                blur_px: float = 0.0, seed: int = 0, background: int = BACKGROUND,
                margin_m: float | None = None) -> np.ndarray:
    """A grey uint8 image (camera.height × camera.width) of the board posed at T_cam_board (board frame → camera
    optical frame) through the camera model (any core.optics model)."""
    import cv2
    from .boards import default_margin
    T = np.asarray(T_cam_board, float)
    R, t = T[:3, :3], T[:3, 3]
    s = max(int(supersample), 1)
    W, H = camera.width, camera.height
    rays = _subpixel_rays(camera, s)
    n = R[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        lam = (n @ t) / (rays @ n)
    hit = np.isfinite(lam) & (lam > 0) & np.isfinite(rays).all(axis=1)
    P = rays * np.where(hit, lam, 0.0)[:, None]
    B = (P - t) @ R                                        # board-frame coordinates
    m = default_margin(board) if margin_m is None else float(margin_m)
    # texture finer than a ray's footprint at the nearest board point
    zmin = float(np.min(lam[hit])) if hit.any() else 1.0
    f = max(camera.fx, camera.fy)
    ppm = float(np.clip(3.0 * f * s / max(zmin, 1e-3), 2000.0, TEX_MAX_PX_PER_M))
    tex = rasterise(board, ppm, m).astype(np.float32)
    x0, y0, _x1, _y1 = board.page_extent()
    page = board.board_to_page(B[:, :2])
    mx = ((page[:, 0] - x0 + m) * ppm - 0.5).astype(np.float32)
    my = ((page[:, 1] - y0 + m) * ppm - 0.5).astype(np.float32)
    mx[~hit] = -10.0
    my[~hit] = -10.0
    samp = cv2.remap(tex, mx.reshape(H * s, W * s), my.reshape(H * s, W * s), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=float(background)).reshape(-1)
    img = samp.reshape(H, s, W, s).mean(axis=(1, 3))
    if blur_px > 0:
        img = cv2.GaussianBlur(img, (0, 0), float(blur_px))
    if noise > 0:
        img = img + np.random.default_rng(seed).normal(0.0, float(noise), img.shape)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def look_at_board(distance: float, tilt_deg: tuple = (0.0, 0.0), centre=(0.0, 0.0), roll_deg: float = 0.0,
                  board: Board | None = None) -> np.ndarray:
    """T_cam_board (board frame → camera optical frame) that puts the board's centre `distance` m along the ray
    through the normalised image point `centre`, its printed side towards the camera, tilted by tilt_deg (about the
    camera's x, then y axis) and rolled by roll_deg about the optical axis. A ChArUco board's z axis points into the
    board (away from the camera), an AprilGrid's out of its face (towards it); facing() turns it."""
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler('xyz', [math.radians(tilt_deg[0]), math.radians(tilt_deg[1]),
                                    math.radians(roll_deg)]).as_matrix()
    F = facing(board)[:3, :3] if board is not None else np.eye(3)
    bc = board_centre(board) if board is not None else np.zeros(3)
    d = np.array([centre[0], centre[1], 1.0])
    p = d / np.linalg.norm(d) * float(distance)
    T = np.eye(4)
    T[:3, :3] = R @ F
    T[:3, 3] = p - R @ F @ bc
    return T


def facing(board: Board | None) -> np.ndarray:
    """The rotation (4 × 4) that turns the board frame so its printed side faces a camera looking along +z: identity
    for ChArUco (z into the board), 180° about x for an AprilGrid (z out of its face)."""
    T = np.eye(4)
    if getattr(board, 'kind', '') == 'aprilgrid':
        T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    return T


def board_centre(board: Board) -> np.ndarray:
    P = board.points()
    return 0.5 * (P.min(0) + P.max(0))


# =========================================================================== hand-eye problems
@dataclass
class SimCamera:
    name: str
    camera: object
    robot: str = ''
    link: str = ''
    T: np.ndarray = field(default_factory=lambda: np.eye(4))     # truth: optical frame in the mount frame


@dataclass
class SimBoard:
    name: str
    board: Board
    robot: str = ''
    link: str = ''
    T: np.ndarray = field(default_factory=lambda: np.eye(4))     # truth: board frame in the mount frame


def _mount(robots, robot, link, q) -> np.ndarray:
    if not robot:
        return np.eye(4)
    mdl = robots[robot]
    return mdl.fk_array(q[robot])[mdl.link_index(link)]


def visible(camera, T_cam_board, points, edge_px: float = EDGE_PX,
            max_incidence_deg: float = MAX_INCIDENCE_DEG) -> tuple[np.ndarray, np.ndarray]:
    """(mask (N,), uv (N, 2)): the board points the camera images inside the image (edge_px from the edge) and
    within max_incidence_deg of the board normal, in front of the board's printed side or its back alike."""
    P = np.asarray(points, float) @ T_cam_board[:3, :3].T + T_cam_board[:3, 3]
    uv, ok = camera.project(P)
    n = T_cam_board[:3, 2]
    with np.errstate(invalid='ignore', divide='ignore'):
        c = np.abs(P @ n) / np.linalg.norm(P, axis=1)
    inc = np.degrees(np.arccos(np.clip(c, 0, 1)))
    W, H = camera.width, camera.height
    inside = (uv[:, 0] >= edge_px) & (uv[:, 0] <= W - 1 - edge_px) & (uv[:, 1] >= edge_px) & \
        (uv[:, 1] <= H - 1 - edge_px)
    return ok & inside & (inc <= max_incidence_deg) & (P[:, 2] > 0), uv


def simulate(robots: dict, cameras: list, boards: list, q_true: list, noise_px: float = 0.3, seed: int = 0,
             joint_offsets: dict | None = None, encoder_noise_rad: float = 0.0, min_points: int = 6,
             estimate_intrinsics: bool = False, intrinsics_guess: dict | None = None) -> tuple[Problem, dict]:
    """A hand-eye Problem from ground truth, and the truth {'X': {camera: T}, 'Z': {board: T}, 'joint_offsets':
    {robot: (nq,)}, 'q_true': ...}. q_true: per pose {robot: true joint vector}. The recorded readings are
    q_true − offset (+ noise). intrinsics_guess: {camera: CameraModel} to start the solver from instead of the
    true model (with estimate_intrinsics)."""
    rng = np.random.default_rng(seed)
    offs = {r: np.asarray(v, float) for r, v in (joint_offsets or {}).items()}
    joints = []
    for q in q_true:
        rec = {}
        for r, v in q.items():
            v = np.asarray(v, float)
            rec[r] = v - offs.get(r, 0.0) + (rng.normal(0.0, encoder_noise_rad, v.shape) if encoder_noise_rad
                                            else 0.0)
        joints.append(rec)
    obs = []
    for p, q in enumerate(q_true):
        for c in cameras:
            Tc = _mount(robots, c.robot, c.link, q) @ c.T
            for b in boards:
                Tb = _mount(robots, b.robot, b.link, q) @ b.T
                T = inv_T(Tc) @ Tb
                pts = b.board.points()
                vis, uv = visible(c.camera, T, pts)
                if vis.sum() < min_points:
                    continue
                ids = np.flatnonzero(vis)
                z = uv[ids] + rng.normal(0.0, noise_px, (len(ids), 2))
                obs.append(Observation(c.name, b.name, p, ids, z, f'pose {p + 1:02d} {c.name}'))
    guess = intrinsics_guess or {}
    prob = Problem(
        [CameraSpec(c.name, guess.get(c.name, c.camera), c.robot, c.link, intrinsics=estimate_intrinsics)
         for c in cameras],
        [BoardSpec(b.name, b.board.points(), b.robot, b.link, board=b.board) for b in boards],
        dict(robots), joints, obs)
    truth = {'X': {c.name: np.array(c.T) for c in cameras}, 'Z': {b.name: np.array(b.T) for b in boards},
             'joint_offsets': {r: v.copy() for r, v in offs.items()}, 'q_true': q_true,
             'cameras': {c.name: c.camera for c in cameras}}
    return prob, truth


def sample_poses(robots: dict, cameras: list, boards: list, centre: dict, n: int, rng, spread: float = 0.35,
                 min_points: int = 12, max_tries: int = 4000) -> list:
    """n joint vectors {robot: q} around `centre` (every robot's centre pose), the joints that move a mount link
    spread uniformly ± spread rad, each kept when every camera sees min_points corners of some board."""
    from ..robot.manipulability import chain_mask
    masks = {}
    for s in list(cameras) + list(boards):
        if s.robot:
            m = masks.get(s.robot, np.zeros(robots[s.robot].nq, bool))
            masks[s.robot] = m | chain_mask(robots[s.robot], s.link)
    out = []
    tries = 0
    while len(out) < n and tries < max_tries:
        tries += 1
        q = {}
        for r, c in centre.items():
            c = np.asarray(c, float)
            d = rng.uniform(-spread, spread, c.shape) * masks.get(r, np.zeros(c.shape, bool))
            q[r] = robots[r].clip(c + d)
        good = True
        for c in cameras:
            Tc = _mount(robots, c.robot, c.link, q) @ c.T
            seen = 0
            for b in boards:
                T = inv_T(Tc) @ _mount(robots, b.robot, b.link, q) @ b.T
                seen = max(seen, int(visible(c.camera, T, b.board.points())[0].sum()))
            if seen < min_points:
                good = False
                break
        if good:
            out.append(q)
    return out


# =========================================================================== the built-in scenes
def ur5e_model():
    """The bundled UR5e + Robotiq 2F-85 sample's robot (key 'ur5e') and its RobotModel."""
    from ..robot.model import get_model
    from ..robot.samples import sample_project
    p = sample_project('ur5e')
    r = p.robots[0]
    return p, get_model(p, r.uid)


def synthetic_camera(width: int = 640, height: int = 480, f: float = 480.0, model: str = 'pinhole', dist=()):
    """A synthetic camera for the checks (not a product's figures): f px, the principal point near the centre."""
    from ..optics import CameraModel
    return CameraModel(width, height, f, f * 1.0005, (width - 1) / 2 + 3.2, (height - 1) / 2 - 2.1, model, dist)


def scene(kind: str = 'eye_in_hand', n_poses: int = 20, noise_px: float = 0.3, seed: int = 0, board=None,
          joint_offsets_deg: dict | None = None, encoder_noise_deg: float = 0.0, two_cameras: bool = False,
          camera=None, estimate_intrinsics: bool = False, intrinsics_guess=None, spread: float = 0.35,
          pose_seed: int | None = None):
    """A ready hand-eye problem on the UR5e sample: 'eye_in_hand' (a camera on wrist_3_link where the sample's
    wrist camera is, a board on the table in front of it) or 'eye_to_hand' (the board on wrist_3_link, a static
    camera 0.6 m in front of it). two_cameras adds a second static camera that sees the same board (multi-camera).
    Returns (Problem, truth, info). pose_seed: the seed of the robot poses (default: seed), so Monte Carlo trials
    can keep the poses and redraw only the noise."""
    from .boards import CharucoBoard
    from ..frames import optical_pose
    proj, mdl = ur5e_model()
    robots = {'ur5e': mdl}
    cam = camera or synthetic_camera()
    board = board or CharucoBoard(9, 7, 0.03, 0.022)
    home = mdl.home.copy()
    wrist = proj.variant.sensors[0]
    T_w3_cam = optical_pose(wrist.mount.pose).matrix()           # the sample's wrist camera, optical frame
    cams, boards = [], []
    flip = np.diag([-1.0, 1.0, -1.0, 1.0])                      # 180° about y: looks back along the optical axis
    if kind == 'eye_in_hand':
        cams.append(SimCamera('wrist', cam, 'ur5e', 'wrist_3_link', T_w3_cam))
        Tw_cam = mdl.link_pose(home, 'wrist_3_link') @ T_w3_cam
        Tw_b = Tw_cam @ look_at_board(0.42, (8.0, -6.0), board=board)
        boards.append(SimBoard('board', board, '', '', Tw_b))
        if two_cameras:
            cams.append(SimCamera('static', cam, '', '', Tw_b @ inv_T(look_at_board(0.7, (-15.0, 10.0),
                                                                                    board=board))))
    elif kind == 'eye_to_hand':
        # a viewer 0.72 m out along the wrist camera's axis, looking back; the board 0.6 m in front of it (0.12 m
        # from the flange-side camera place), printed side away from the flange
        # (rolled 30° about its axis: looking straight back would put the camera's rotation in the world at 180°,
        # where the Tsai-Lenz parameterisation is singular; axxb's known limits)
        roll = np.eye(4)
        roll[:3, :3] = exp_so3([0.0, 0.0, math.radians(30.0)])
        C2 = T_w3_cam @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.72], [0, 0, 0, 1.0]]) @ flip @ roll
        T_w3_board = C2 @ look_at_board(0.6, (10.0, 5.0), board=board)
        boards.append(SimBoard('board', board, 'ur5e', 'wrist_3_link', T_w3_board))
        Tw_w3 = mdl.link_pose(home, 'wrist_3_link')
        cams.append(SimCamera('static', cam, '', '', Tw_w3 @ C2))
        if two_cameras:
            cams.append(SimCamera('static 2', cam, '', '', Tw_w3 @ T_w3_board @
                                  inv_T(look_at_board(0.75, (-20.0, -8.0), board=board))))
    else:
        raise ValueError(f'unknown scene kind {kind!r}')
    rng = np.random.default_rng(seed if pose_seed is None else pose_seed)
    Q = sample_poses(robots, cams, boards, {'ur5e': home}, n_poses, rng, spread)
    if len(Q) < n_poses:
        raise RuntimeError(f'only {len(Q)} of {n_poses} poses keep the board in view')
    offs = None
    if joint_offsets_deg:
        offs = {r: np.radians(np.asarray(v, float)) for r, v in joint_offsets_deg.items()}
    prob, truth = simulate(robots, cams, boards, Q, noise_px, seed, offs, math.radians(encoder_noise_deg),
                           estimate_intrinsics=estimate_intrinsics, intrinsics_guess=intrinsics_guess)
    return prob, truth, {'project': proj, 'model': mdl, 'cameras': cams, 'boards': boards, 'poses': Q}


def two_arm_scene(n_poses: int = 20, noise_px: float = 0.3, seed: int = 0, board=None, camera=None,
                  spread: float = 0.2, pose_seed: int | None = None):
    """A robot-world/hand-eye problem with both sides moving (multi-arm, the general AX = ZB case): the UR5e sample's
    arm 'a' carries the camera on wrist_3_link (where the sample's wrist camera is), a second UR5e 'b' (the same
    model, its base 0.98 m along y, turned 180° about z) holds the board on its wrist_3_link, 0.42 m in front of the
    camera at the home poses. Returns (Problem, truth, info)."""
    import copy

    from scipy.spatial.transform import Rotation

    from ..frames import Pose, optical_pose
    from ..robot.model import RobotModel, compile_robot
    from .boards import CharucoBoard
    proj, mdl_a = ur5e_model()
    r1 = proj.robots[0]
    r2 = copy.deepcopy(r1)
    r2.uid = r1.uid + '-b'
    r2.name = r1.name + ' (second arm)'
    r2.base = Pose([0.0, 0.98, 0.0], Rotation.from_euler('z', math.pi).as_quat())
    mdl_b = RobotModel(r2, proj.sources, compile_robot(r1, proj.sources))
    robots = {'a': mdl_a, 'b': mdl_b}
    cam = camera or synthetic_camera()
    board = board or CharucoBoard(9, 7, 0.03, 0.022)
    T_w3_cam = optical_pose(proj.variant.sensors[0].mount.pose).matrix()
    Tw_cam = mdl_a.link_pose(mdl_a.home, 'wrist_3_link') @ T_w3_cam
    Tw_board = Tw_cam @ look_at_board(0.42, (8.0, -6.0), board=board)
    T_b3_board = inv_T(mdl_b.link_pose(mdl_b.home, 'wrist_3_link')) @ Tw_board
    cams = [SimCamera('wrist a', cam, 'a', 'wrist_3_link', T_w3_cam)]
    boards = [SimBoard('board b', board, 'b', 'wrist_3_link', T_b3_board)]
    rng = np.random.default_rng(seed if pose_seed is None else pose_seed)
    Q = sample_poses(robots, cams, boards, {'a': mdl_a.home.copy(), 'b': mdl_b.home.copy()}, n_poses, rng, spread)
    if len(Q) < n_poses:
        raise RuntimeError(f'only {len(Q)} of {n_poses} poses keep the board in view')
    prob, truth = simulate(robots, cams, boards, Q, noise_px, seed)
    return prob, truth, {'project': proj, 'models': robots, 'cameras': cams, 'boards': boards, 'poses': Q}


def with_noise(problem: Problem, noise_px: float, seed: int) -> Problem:
    """A copy of a problem with N(0, noise_px²) added to every observed pixel (the poses and ids unchanged)."""
    rng = np.random.default_rng(seed)
    obs = [Observation(o.camera, o.board, o.pose, o.ids, o.uv + rng.normal(0.0, noise_px, o.uv.shape), o.name)
           for o in problem.observations]
    return Problem(problem.cameras, problem.boards, problem.robots, problem.joints, obs)
