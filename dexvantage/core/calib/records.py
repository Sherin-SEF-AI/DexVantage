"""Calibration records in the project (project.calibration) and the helpers that turn them into the planner's and
the solver's inputs.

    project.calibration = {'boards': [board record], 'sessions': [session record]}

A board record: {'uid', 'name', 'board': boards' to_dict() (its kind and figures), 'mount': {'parent_frame',
'robot_uid', 'pose': {'position', 'quaternion'}} (the board frame in its mount frame: the world, a static frame read
in the world, or a robot link), 'default': True when the placement is DexVantage's default and not the user's}.

A session record: {'uid', 'name', 'sensor_uid', 'imager', 'mode' ('eye_in_hand' | 'eye_to_hand'), 'board_uid',
'robot_uid', 'link' (the robot link that moves: the camera's mount eye-in-hand, the board's eye-to-hand), 'created',
'source': {'kind': 'plan' | 'folder', ...}, 'settings': {...}, 'poses': [pose record], 'result': {} or the solve's
HandEyeResult.to_dict() with 'applied': bool}.

A pose record: {'name', 'q': [...] (the robot's joint vector in its model's joint order, rad / m), 'joint_names',
'image' (file name, no folder), 't_image', 't_joints', 'dt_s' (s, or None), 'ids': [...], 'uv': [[u, v], ...]
(detected corners, px, OpenCV pixel centres; empty for a planned pose), 'note'}.

Defaults used here (labelled as such wherever they show): the board DEFAULT_BOARD (a ChArUco 9 × 7 board of 30 mm
squares, the size the synthetic checks use), laid printed face up at the first task volume's centre on the surface
under it (eye-in-hand), or held on the robot's gripper frame HELD_CLEARANCE_M beyond its fingertip frames, printed
face along the frame's z axis (eye-to-hand).
"""
from __future__ import annotations

import copy
import datetime as _dt
import math

import numpy as np

from ..frames import Pose
from .axxb import inv_T

MODES = ('eye_in_hand', 'eye_to_hand')
DEFAULT_BOARD = {'kind': 'charuco', 'squares_x': 9, 'squares_y': 7, 'square_m': 0.03, 'marker_m': 0.022,
                 'dictionary': 'DICT_5X5_100', 'legacy': False}
HELD_CLEARANCE_M = 0.05        # a default: a held board's centre this far beyond the fingertip frames
HELD_OFFSET_M = 0.15           # a default: along the holding link's z when the robot names no fingertip frames
STATIC = ('world', 'cage', 'tripod', 'table')


def now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def calibration_of(project) -> dict:
    """project.calibration, or an empty one for a project object that has none."""
    c = getattr(project, 'calibration', None)
    if not isinstance(c, dict):
        return {'boards': [], 'sessions': []}
    return c


# =========================================================================== boards
def board_of(rec: dict):
    """The boards.py board of a board record (BoardError says what is wrong)."""
    from .boards import board_from_dict
    return board_from_dict(dict((rec or {}).get('board') or DEFAULT_BOARD))


def board_mount(rec: dict) -> tuple[str, str, np.ndarray]:
    """(robot uid or '', link or '', Z 4 × 4): the board frame in its mount frame (the world for a static mount)."""
    m = (rec or {}).get('mount') or {}
    parent = str(m.get('parent_frame') or 'world')
    robot = str(m.get('robot_uid') or '')
    Z = Pose.from_dict(m.get('pose')).matrix()
    if not robot or parent in STATIC:
        return '', '', Z
    return robot, parent, Z


def find_board(project, uid: str | None) -> dict | None:
    boards = calibration_of(project).get('boards') or []
    if uid:
        return next((b for b in boards if isinstance(b, dict) and b.get('uid') == uid), None)
    return next((b for b in boards if isinstance(b, dict)), None)


def _surface_z(project, x: float, y: float) -> float:
    best = None
    for s in getattr(project.workspace, 'surfaces', []) or []:
        if not isinstance(s, dict):
            continue
        c = np.asarray(s.get('center') or (0.0, 0.0), float).reshape(-1)[:2]
        sz = np.asarray(s.get('size') or (0.0, 0.0), float).reshape(-1)[:2]
        if len(sz) == 2 and abs(x - c[0]) <= sz[0] / 2 and abs(y - c[1]) <= sz[1] / 2:
            z = float(s.get('z', 0.0))
            best = z if best is None else max(best, z)
    return 0.0 if best is None else best


def default_board_record(project, mode: str, robot_uid: str = '', board: dict | None = None) -> dict:
    """A board record with DexVantage's default placement (module docstring), marked default=True."""
    from .boards import board_from_dict
    from .synthetic import board_centre
    from ..project import new_uid
    bd = dict(board or DEFAULT_BOARD)
    b = board_from_dict(dict(bd))
    bc = board_centre(b)
    # the printed face's outward normal in the board frame is +z for an AprilGrid and -z for ChArUco (its z points
    # into the board), so R below turns it to the target direction
    if mode == 'eye_to_hand':
        rob = project.robot(robot_uid) if robot_uid else project.primary_robot
        if rob is None:
            raise ValueError('an eye-to-hand board needs a robot to hold it')
        from ..robot.model import get_model
        mdl = get_model(project, rob.uid)
        link = rob.gripper_frame if rob.gripper_frame in mdl.link_names else mdl.link_names[-1]
        Tl = mdl.link_pose(mdl.home, link)
        tips = [f for f in rob.fingertip_frames if f in mdl.link_names]
        if tips:
            reach = max(float((inv_T(Tl) @ mdl.link_pose(mdl.home, f))[2, 3]) for f in tips)
            off = max(reach, 0.0) + HELD_CLEARANCE_M
        else:
            off = HELD_OFFSET_M
        R = np.eye(3) if b.kind == 'aprilgrid' else np.diag([1.0, -1.0, -1.0])   # printed face along the link's z
        centre = np.array([0.0, 0.0, off])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = centre - R @ bc
        mount = {'parent_frame': link, 'robot_uid': rob.uid, 'pose': Pose.from_matrix(T).to_dict()}
        where = f'held on {link}, {1000 * off:.0f} mm along its z axis, printed face outwards (a default)'
    else:
        vols = getattr(project.workspace, 'volumes', []) or []
        if vols:
            cx, cy = float(vols[0].center[0]), float(vols[0].center[1])
        else:
            cx, cy = 0.5, 0.0
        z = _surface_z(project, cx, cy)
        # printed face up: its outward normal along the world's +z
        R = np.eye(3) if b.kind == 'aprilgrid' else np.diag([1.0, -1.0, -1.0])     # R @ out = +z
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.array([cx, cy, z]) - R @ bc
        mount = {'parent_frame': 'world', 'robot_uid': '', 'pose': Pose.from_matrix(T).to_dict()}
        where = (f'flat on the table at ({1000 * cx:.0f}, {1000 * cy:.0f}, {1000 * z:.0f}) mm, printed face up '
                 '(a default: the first task volume\'s centre)')
    return {'uid': new_uid(), 'name': 'Calibration board', 'board': bd, 'mount': mount, 'default': True,
            'where': where}


# =========================================================================== the camera
def sensor_imager(sensor):
    """The imager a sensor is calibrated by: the coverage study's rule (a depth sensor's RGB imager when it has one,
    else its reference imager), at the sensor's chosen stream (core.studies.coverage pick_imager, sensor_mode)."""
    from .. import library as LB
    from ..studies.coverage import pick_imager, sensor_mode
    ims = LB.sensor_imagers(sensor, sensor_mode(sensor, {}))
    if not ims:
        raise ValueError(f'{sensor.name} has no imager to calibrate')
    return pick_imager(sensor, ims, {})[0]


def camera_mount(project, sensor, imager) -> tuple[str, str, np.ndarray]:
    """(robot uid or '', link or '', X 4 × 4): the imager's optical frame in its mount frame (the link for a sensor
    on a link, else the world)."""
    from ..scenegeom import sensor_world_pose
    Tbo = imager.pose_in_body.matrix()
    if sensor.mount.on_link:
        return sensor.mount.robot_uid, sensor.mount.parent_frame, sensor.mount.pose.matrix() @ Tbo
    return '', '', sensor_world_pose(project, sensor).matrix() @ Tbo


def mount_pose_from_X(sensor, imager, X) -> Pose:
    """The sensor body pose in its mount's parent frame that puts the imager's optical frame at X (in the mount
    frame: the link, or the world for a static sensor; static frames are read in the world)."""
    return Pose.from_matrix(np.asarray(X, float) @ inv_T(imager.pose_in_body.matrix()))


def camera_calibration_from(model, source: str) -> dict:
    """sensor.calibration's per-imager record of an estimated lens model (core.library reads K, D, width, height and
    model)."""
    K, D, m = model.to_opencv()
    return {'K': np.asarray(K).tolist(), 'D': [float(v) for v in np.asarray(D).ravel()], 'width': model.width,
            'height': model.height, 'model': m, 'source': source}


# =========================================================================== sessions
def session_record(sensor, imager, mode: str, board_uid: str, robot_uid: str, link: str, poses: list, source: dict,
                   name: str = '', settings: dict | None = None) -> dict:
    from ..project import new_uid
    return {'uid': new_uid(), 'name': name or f'{sensor.name} · {mode.replace("_", "-")}', 'sensor_uid': sensor.uid,
            'imager': imager.name, 'mode': mode, 'board_uid': board_uid, 'robot_uid': robot_uid, 'link': link,
            'created': now_utc(), 'source': dict(source), 'settings': dict(settings or {}),
            'poses': [clean_pose(p) for p in poses], 'result': {}}


def clean_pose(p: dict) -> dict:
    p = p if isinstance(p, dict) else {}

    def num(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) else None
    uv = np.asarray(p.get('uv') or [], float).reshape(-1, 2)
    ids = [int(i) for i in p.get('ids') or []]
    if len(ids) != len(uv):
        ids, uv = [], np.zeros((0, 2))
    return {'name': str(p.get('name') or ''), 'q': [float(v) for v in p.get('q') or []],
            'joint_names': [str(v) for v in p.get('joint_names') or []], 'image': str(p.get('image') or ''),
            't_image': num(p.get('t_image')), 't_joints': num(p.get('t_joints')), 'dt_s': num(p.get('dt_s')),
            'ids': ids, 'uv': uv.tolist(), 'note': str(p.get('note') or '')}


def find_session(project, uid: str | None = None, detected: bool = False) -> dict | None:
    """The session with that uid, else the newest one (with detections when detected=True)."""
    ss = [s for s in calibration_of(project).get('sessions') or [] if isinstance(s, dict)]
    if uid:
        return next((s for s in ss if s.get('uid') == uid), None)
    if detected:
        ss = [s for s in ss if any(p.get('ids') for p in s.get('poses') or [])]
    return ss[-1] if ss else None


def copy_calibration(project) -> dict:
    return copy.deepcopy(calibration_of(project))
