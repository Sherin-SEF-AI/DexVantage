"""Recording for hand-eye calibration: offline capture from a folder of board images and a joint_states file, matched
by time, detected, and turned into a session (records.py) and a hand-eye Problem (handeye.py).

Sources (SOURCES): 'folder' (this module); 'ros2' (ROS 2 topics) and 'lerobot' (a LeRobot episode) arrive in Phase 8
with the dataset readers, and say so (CaptureError) instead of failing in an unexplained way.

The joint_states file (read_joint_states):
- CSV: a header row with a time column ('t', 'time', 'stamp', 'timestamp' or 'time_s') and one column per joint,
  named as the robot's joints; one row per sample. Values in radians (metres for prismatic joints); 'unit' = 'deg'
  reads the revolute joints in degrees (prismatic joints stay in metres).
- JSON: {"t": [...], "<joint>": [...], ...} (columns); [{"t": ..., "<joint>": ..., ...}, ...] (rows); or ROS-style
  [{"t": ..., "name": [...], "position": [...]}, ...] (sensor_msgs/JointState samples with a time in seconds).
Joints of the robot the file does not name are held at the robot's home (said in the notes); columns that name no
joint of the robot are ignored (said too).

Image times (image_times), in this order: a sidecar file in the folder ('images.csv' or 'timestamps.csv' with columns
file and t, or 'images.json' {file: t}); else a time in each file name: the last number of its stem with a decimal
point is seconds ('pose_07_1730900000.250.png'), an integer of 19 digits nanoseconds, 16 microseconds, 13
milliseconds, 10 seconds; else none, and then images and joint samples are paired in order when their counts are equal
(said in the notes), else the capture is refused.

Matching (match_times): each image takes the joint sample nearest in time, when within tolerance_s (TOLERANCE_S, a
default); an image with none that near is left out with the reason. The joint speed at the match (the difference to
the neighbouring samples) is recorded: an image taken while the robot moved faster than MOVING_DEG_S (a default) is
flagged, since the solver assumes still poses.

Several sessions together (sessions_problem): one bundle adjustment over every camera, board and robot of the
sessions, a camera or board shared between sessions estimated once; this is how a static camera that only sees a
board fixed in the world is calibrated (through a robot-mounted camera's views of the same board).

Known limits: the nearest sample, not an interpolation (calibration poses are taken still); one camera and one robot per
session (several are solved together with sessions_problem); image and joint clocks are taken as the same clock (no
offset estimated); images are read as files OpenCV can decode.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import re

import numpy as np

SOURCES = ('folder', 'ros2', 'lerobot')
SOURCE_LABELS = {'folder': 'Image folder and joint states file', 'ros2': 'ROS 2 topics (arrives in Phase 8)',
                 'lerobot': 'LeRobot episode (arrives in Phase 8)'}
TOLERANCE_S = 0.05             # a default: an image takes a joint sample this close in time
MOVING_DEG_S = 0.5             # a default: faster joint motion at the match flags the image
IMAGE_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.pgm', '.ppm')
TIME_COLUMNS = ('t', 'time', 'stamp', 'timestamp', 'time_s')
SIDECARS = ('images.csv', 'timestamps.csv', 'images.json')


class CaptureError(ValueError):
    """A capture that cannot be read; the message says why in plain words."""


def check_source(kind: str):
    if kind == 'ros2':
        raise CaptureError('Recording from ROS 2 topics arrives in Phase 8 (with the dataset readers). For now, save '
                           'the board images and a joint_states file and use "Image folder and joint states file".')
    if kind == 'lerobot':
        raise CaptureError('Reading a LeRobot episode arrives in Phase 8 (with the dataset readers). For now, export '
                           'the frames and the joint states and use "Image folder and joint states file".')
    if kind != 'folder':
        raise CaptureError(f'Unknown capture source "{kind}": use one of {", ".join(SOURCES)}.')


# =========================================================================== joint states
class JointStates:
    """t (N,) seconds, Q (N, nq) in the robot's joint order, names (the robot's), found (the joints the file names),
    notes."""

    def __init__(self, t, Q, names, found, notes):
        self.t, self.Q, self.names, self.found, self.notes = t, Q, names, found, notes

    def __len__(self):
        return len(self.t)


def _num(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise CaptureError(f'"{v}" in the joint states is not a number.') from None
    if not math.isfinite(x):
        raise CaptureError(f'The joint states hold a value that is not finite ({v}).')
    return x


def _rows_from_json(doc) -> list:
    if isinstance(doc, dict):
        if isinstance(doc.get('samples'), list):
            return _rows_from_json(doc['samples'])
        cols = {k: v for k, v in doc.items() if isinstance(v, list)}
        if not cols:
            raise CaptureError('The joint states JSON holds no columns: give {"t": [...], "<joint>": [...]}.')
        n = {len(v) for v in cols.values()}
        if len(n) != 1:
            raise CaptureError('The joint states JSON columns have different lengths.')
        return [{k: v[i] for k, v in cols.items()} for i in range(n.pop())]
    if isinstance(doc, list):
        rows = []
        for r in doc:
            if not isinstance(r, dict):
                raise CaptureError('Each joint states sample must be an object with a time and joint values.')
            if isinstance(r.get('name'), list) and isinstance(r.get('position'), list):
                row = {k: v for k, v in r.items() if k not in ('name', 'position', 'velocity', 'effort')}
                if len(r['name']) != len(r['position']):
                    raise CaptureError('A joint states sample has a different number of names and positions.')
                row.update(dict(zip(r['name'], r['position'])))
                rows.append(row)
            else:
                rows.append(r)
        return rows
    raise CaptureError('The joint states JSON must be an object of columns or a list of samples.')


def read_joint_states(source, joint_names, home=None, unit: str = 'rad', prismatic=None) -> JointStates:
    """The joint states of a CSV or JSON file (a path, or text) for a robot with joint_names (its model's order);
    home: the values of joints the file does not name (default 0). prismatic: (nq,) True for the sliding joints,
    whose values stay in metres when unit is 'deg' (only angles are converted)."""
    names = [str(n) for n in joint_names]
    if isinstance(source, str) and os.path.isfile(source):
        with open(source, encoding='utf-8-sig') as fh:
            text = fh.read()
        is_json = source.lower().endswith('.json') or text.lstrip()[:1] in ('{', '[')
    else:
        text = str(source)
        is_json = text.lstrip()[:1] in ('{', '[')
    if is_json:
        try:
            rows = _rows_from_json(json.loads(text))
        except ValueError as e:
            if isinstance(e, CaptureError):
                raise
            raise CaptureError(f'The joint states file is not valid JSON: {e}.') from None
    else:
        rd = csv.DictReader(io.StringIO(text))
        if not rd.fieldnames:
            raise CaptureError('The joint states CSV has no header row (a time column and one column per joint).')
        rows = [r for r in rd if any((v or '').strip() for v in r.values())]
    if not rows:
        raise CaptureError('The joint states file holds no samples.')
    keys = list(rows[0].keys())
    tcol = next((k for k in keys if str(k).strip().lower() in TIME_COLUMNS), None)
    if tcol is None:
        raise CaptureError('The joint states have no time column: name it "t" (seconds).')
    lookup = {str(k).strip(): k for k in keys}
    found = [n for n in names if n in lookup]
    if not found:
        raise CaptureError('None of the joint states columns names a joint of the robot (it has '
                           + ', '.join(names[:8]) + ('…' if len(names) > 8 else '') + ').')
    extra = [str(k) for k in keys if k != tcol and str(k).strip() not in names]
    h = np.zeros(len(names)) if home is None else np.asarray(home, float).reshape(-1)
    scale = math.pi / 180.0 if str(unit).lower().startswith('deg') else 1.0
    slide = np.zeros(len(names), bool) if prismatic is None else np.asarray(prismatic, bool).reshape(-1)
    t = np.array([_num(r.get(tcol)) for r in rows])
    Q = np.tile(h, (len(rows), 1))
    for j, n in enumerate(names):
        if n in lookup:
            k = 1.0 if j < len(slide) and slide[j] else scale
            Q[:, j] = [_num(r.get(lookup[n])) * k for r in rows]
    order = np.argsort(t, kind='stable')
    notes = []
    missing = [n for n in names if n not in lookup]
    if missing:
        notes.append(f'{len(missing)} joint(s) of the robot are not in the file and are held at home: '
                     + ', '.join(missing[:6]) + ('…' if len(missing) > 6 else '') + '.')
    if extra:
        notes.append('Columns that name no joint of the robot are ignored: ' + ', '.join(extra[:6])
                     + ('…' if len(extra) > 6 else '') + '.')
    return JointStates(t[order], Q[order], names, found, notes)


# =========================================================================== images
_NUM = re.compile(r'\d+(?:\.\d+)?')


def time_from_name(name: str) -> float | None:
    """A time (s) in a file name by the rule of the module docstring, or None."""
    stem = os.path.splitext(os.path.basename(name))[0]
    nums = _NUM.findall(stem)
    if not nums:
        return None
    dec = [x for x in nums if '.' in x]
    if dec:
        return float(dec[-1])
    s = nums[-1]
    scale = {19: 1e-9, 16: 1e-6, 13: 1e-3, 10: 1.0}.get(len(s))
    return None if scale is None else int(s) * scale


def list_images(folder: str) -> list:
    try:
        names = sorted(os.listdir(folder))
    except OSError as e:
        raise CaptureError(f'The image folder cannot be read: {e.strerror or e}.') from None
    return [n for n in names if n.lower().endswith(IMAGE_EXT) and os.path.isfile(os.path.join(folder, n))]


def image_times(folder: str, files: list | None = None) -> tuple[list, str]:
    """([(file, t or None)], how the times were found)."""
    files = list_images(folder) if files is None else files
    for sc in SIDECARS:
        p = os.path.join(folder, sc)
        if not os.path.isfile(p):
            continue
        try:
            with open(p, encoding='utf-8-sig') as fh:
                text = fh.read()
            if sc.endswith('.json'):
                m = {str(k): _num(v) for k, v in json.loads(text).items()}
            else:
                rd = csv.DictReader(io.StringIO(text))
                fk = next((k for k in rd.fieldnames or [] if k.strip().lower() in ('file', 'image', 'name')), None)
                tk = next((k for k in rd.fieldnames or [] if k.strip().lower() in TIME_COLUMNS), None)
                if fk is None or tk is None:
                    raise CaptureError(f'{sc} needs a "file" and a "t" column.')
                m = {os.path.basename(r[fk].strip()): _num(r[tk]) for r in rd if r.get(fk)}
        except (ValueError, AttributeError) as e:
            if isinstance(e, CaptureError):
                raise
            raise CaptureError(f'{sc} cannot be read: {e}.') from None
        return [(f, m.get(f)) for f in files], f'times from {sc}'
    ts = [(f, time_from_name(f)) for f in files]
    if ts and all(t is not None for _f, t in ts):
        return ts, 'times from the file names'
    return [(f, None) for f in files], 'no image times'


def match_times(times: list, js: JointStates, tolerance_s: float = TOLERANCE_S) -> tuple[list, list]:
    """([(file, joint row or None, dt s or None, moving deg/s or None, note)], notes): each image to its nearest
    joint sample (module docstring)."""
    notes = []
    out = []
    if times and all(t is None for _f, t in times):
        if len(times) != len(js):
            raise CaptureError(f'The images carry no times (no images.csv and no time in the file names) and there '
                               f'are {len(times)} images but {len(js)} joint samples, so they cannot be paired. Add '
                               'an images.csv with columns file and t.')
        notes.append('The images carry no times: image k is paired with joint sample k (their counts are equal).')
        for k, (f, _t) in enumerate(times):
            out.append((f, k, None, None, ''))
        return out, notes
    for f, t in times:
        if t is None:
            out.append((f, None, None, None, 'no time for this image'))
            continue
        k = int(np.argmin(np.abs(js.t - t)))
        dt = float(js.t[k] - t)
        speed = None
        nb = [i for i in (k - 1, k + 1) if 0 <= i < len(js)]
        sp = [np.max(np.abs(js.Q[i] - js.Q[k])) / abs(js.t[i] - js.t[k]) for i in nb if js.t[i] != js.t[k]]
        if sp:
            speed = math.degrees(float(min(sp)))
        if abs(dt) > tolerance_s:
            out.append((f, None, dt, speed, f'no joint sample within {1000 * tolerance_s:g} ms (nearest '
                                            f'{1000 * dt:+.0f} ms)'))
            continue
        note = ''
        if speed is not None and speed > MOVING_DEG_S:
            note = f'the robot was moving ({speed:.2f}°/s at the match): retake it still'
        out.append((f, k, dt, speed, note))
    return out, notes


# =========================================================================== a session from a folder
def capture_folder(project, sensor, folder: str, joints, board_rec: dict | None = None,
                   tolerance_s: float = TOLERANCE_S, unit: str = 'rad', progress=None, cancel=None) -> dict:
    """A session record (records.py) from a folder of board images and a joint_states file (path or text): times
    matched, every image detected with the sensor's camera model. Raises CaptureError (a plain sentence)."""
    import cv2

    from ..robot.model import RobotModelError, get_model
    from . import records as RC
    from .detect import detect
    if not os.path.isdir(folder):
        raise CaptureError('The image folder does not exist.')
    try:
        im = RC.sensor_imager(sensor)
        c_robot, c_link, _X = RC.camera_mount(project, sensor, im)
    except (ValueError, RobotModelError, KeyError) as e:
        raise CaptureError(f'{sensor.name} cannot be calibrated: {e}') from None
    mode = 'eye_in_hand' if c_robot else 'eye_to_hand'
    rec = board_rec or RC.find_board(project, None)
    if rec is None:
        rec = RC.default_board_record(project, mode, c_robot)
    board = RC.board_of(rec)
    b_robot, b_link, _Z = RC.board_mount(rec)
    robot_uid, link = (c_robot, c_link) if c_robot else (b_robot, b_link)
    if not robot_uid or project.robot(robot_uid) is None:
        raise CaptureError('The board record must be held on a robot link for an eye-to-hand calibration (the '
                           'camera is fixed).' if not c_robot else 'The robot the camera is on is not in the project.')
    mdl = get_model(project, robot_uid)
    js = read_joint_states(joints, mdl.joint_names, mdl.home, unit,
                           prismatic=[t == 'prismatic' for t in mdl.joint_types])
    files = list_images(folder)
    if not files:
        raise CaptureError('The folder holds no images (PNG, JPEG, BMP, TIFF, PGM or PPM).')
    times, how = image_times(folder, files)
    pairs, notes = match_times(times, js, tolerance_s)
    notes = list(js.notes) + [f'Image times: {how}.'] + notes
    cam = im.camera
    poses = []
    for k, (f, row, dt, speed, why) in enumerate(pairs):
        if cancel is not None and cancel():
            break
        if progress:
            progress(k / max(len(pairs), 1), f'Detecting the board in {f}')
        t_img = dict(times).get(f)
        rec_p = {'name': os.path.splitext(f)[0], 'image': f, 't_image': t_img, 'dt_s': dt, 'note': why}
        if row is None:
            poses.append(dict(rec_p, q=[], ids=[], uv=[]))
            continue
        rec_p.update(q=[float(v) for v in js.Q[row]], joint_names=list(mdl.joint_names), t_joints=float(js.t[row]))
        img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            poses.append(dict(rec_p, ids=[], uv=[], note='OpenCV cannot read this image'))
            continue
        if (img.shape[1], img.shape[0]) != (cam.width, cam.height):
            poses.append(dict(rec_p, ids=[], uv=[], note=f'the image is {img.shape[1]} × {img.shape[0]} px, the '
                                                         f'camera model {cam.width} × {cam.height} px'))
            continue
        det = detect(img, board, cam)
        if det is None:
            poses.append(dict(rec_p, ids=[], uv=[], note=(why + '; ' if why else '') + 'board not found'))
            continue
        poses.append(dict(rec_p, ids=[int(i) for i in det.ids], uv=np.asarray(det.corners, float).tolist()))
    src = {'kind': 'folder', 'folder': os.path.basename(os.path.normpath(folder)),
           'joints': os.path.basename(joints) if isinstance(joints, str) and os.path.isfile(joints) else '',
           'images': len(files), 'matched': sum(1 for p in poses if p.get('q')),
           'detected': sum(1 for p in poses if p.get('ids')), 'times': how, 'notes': notes}
    sess = RC.session_record(sensor, im, mode, str(rec.get('uid') or ''), robot_uid, link, poses, src,
                             settings={'tolerance_s': tolerance_s, 'unit': unit,
                                       'tolerance_default': tolerance_s == TOLERANCE_S})
    # a board that is not (yet) one of the project's records travels with the session
    sess['board'] = None if RC.find_board(project, rec.get('uid')) else rec
    if progress:
        progress(1.0, f'{src["detected"]} of {len(files)} images with the board')
    return sess


# =========================================================================== a session as a problem
def session_problem(project, session: dict):
    """(handeye.Problem, info) of a session with detections: the sensor's camera model, the board record, the robot
    model; info names the camera, the board, the pose index of each view and the imager. Raises CaptureError."""
    from ..robot.model import RobotModelError, get_model
    from . import handeye as HE
    from . import records as RC
    sensor = project.variant.sensor(session.get('sensor_uid'))
    if sensor is None:
        raise CaptureError('The session\'s camera is not in this rig any more.')
    try:
        im = RC.sensor_imager(sensor)
        c_robot, c_link, X0 = RC.camera_mount(project, sensor, im)
    except (ValueError, RobotModelError, KeyError) as e:
        raise CaptureError(f'{sensor.name} cannot be calibrated: {e}') from None
    rec = RC.find_board(project, session.get('board_uid')) or session.get('board')
    if rec is None:
        rec = RC.default_board_record(project, session.get('mode', 'eye_in_hand'), c_robot)
    board = RC.board_of(rec)
    b_robot, b_link, Z0 = RC.board_mount(rec)
    robot_uid = session.get('robot_uid') or c_robot or b_robot
    if project.robot(robot_uid) is None:
        raise CaptureError('The session\'s robot is not in the project.')
    mdl = get_model(project, robot_uid)
    cname = sensor.name or 'camera'
    bname = str(rec.get('name') or 'board')
    if bname == cname:
        bname += ' (board)'
    cams = [HE.CameraSpec(cname, im.camera, robot_uid if c_robot else '', c_link if c_robot else '')]
    brds = [HE.BoardSpec(bname, board.points(), robot_uid if b_robot else '', b_link if b_robot else '',
                         board=board)]
    joints, obs, index = [], [], []
    for k, p in enumerate(session.get('poses') or []):
        q = np.asarray(p.get('q') or [], float)
        ids = np.asarray(p.get('ids') or [], int)
        if len(q) != mdl.nq or len(ids) < 4:
            continue
        uv = np.asarray(p.get('uv'), float).reshape(-1, 2)
        joints.append({robot_uid: q})
        obs.append(HE.Observation(cname, bname, len(joints) - 1, ids, uv, p.get('name') or f'pose {k + 1}'))
        index.append(k)
    if len(obs) < 3:
        raise CaptureError(f'The session has {len(obs)} usable views (a board found and its joint reading matched); '
                           'the solve needs at least 3, and 15 or more for a good result.')
    prob = HE.Problem(cams, brds, {robot_uid: mdl}, joints, obs)
    return prob, {'camera': cname, 'board': bname, 'pose_index': index, 'imager': im.name, 'sensor': sensor,
                  'board_record': rec, 'X0': X0, 'Z0': Z0, 'robot_uid': robot_uid,
                  'link': c_link if c_robot else b_link, 'mode': 'eye_in_hand' if c_robot else 'eye_to_hand'}


def sessions_problem(project, sessions: list):
    """(handeye.Problem, info) of several sessions solved together in one bundle adjustment (spec §8.1: several
    cameras and arms, eye-in-hand and eye-to-hand at once). Each session is read as session_problem reads it; then
    - a camera (a sensor) seen in several sessions is one camera, with one X;
    - a board (a board record, by uid) used in several sessions is one board, with one Z: a static camera that sees
      a board fixed in the world, which alone tells nothing (nothing moves between them), is located through that
      board's Z, which a robot-mounted camera's views of it determine;
    - robot poses with the same joint vector (to 10⁻⁹) of the same robot are one pose (images taken together);
    - each robot of any session is in the problem with its own joint readings.
    info: {'sessions': [{'uid', 'name', 'mode', 'robot_uid', 'camera', 'board'}], 'cameras': {name:
    session_problem's camera info and the indices of its sessions}, 'boards': {name: {'record', 'Z0'}}, 'views': {view
    name: (session index, session pose index)}, 'robots': [uids]}.
    Raises CaptureError (fewer than two sessions, one unusable)."""
    from . import handeye as HE
    sessions = [s for s in sessions or [] if isinstance(s, dict)]
    if len(sessions) < 2:
        raise CaptureError('A joint solve needs two or more capture sessions with detections.')
    cams, brds, robots, joints, obs = {}, {}, {}, [], []
    info = {'sessions': [], 'cameras': {}, 'boards': {}, 'views': {}, 'robots': []}
    pose_of: dict = {}
    names: set = set()

    def unique(base: str) -> str:
        n, k = base, 2
        while n in names:
            n, k = f'{base} ({k})', k + 1
        names.add(n)
        return n

    cam_by_sensor, brd_by_key = {}, {}
    for si, sess in enumerate(sessions):
        try:
            prob, inf = session_problem(project, sess)
        except CaptureError as e:
            raise CaptureError(f'Session "{sess.get("name", si + 1)}": {e}') from None
        info['sessions'].append({'uid': sess.get('uid'), 'name': sess.get('name', f'session {si + 1}'),
                                 'mode': inf['mode'], 'robot_uid': inf['robot_uid']})
        suid = inf['sensor'].uid
        if suid not in cam_by_sensor:
            c = prob.cameras[0]
            cam_by_sensor[suid] = unique(c.name)
            cams[cam_by_sensor[suid]] = HE.CameraSpec(cam_by_sensor[suid], c.camera, c.robot, c.link)
            ci = {k: inf[k] for k in ('sensor', 'imager', 'X0', 'robot_uid', 'link', 'mode')}
            if not c.robot and not prob.boards[0].robot:
                ci['mode'] = 'static'          # fixed camera, fixed board: located through the board's Z
            ci['sessions'] = []
            info['cameras'][cam_by_sensor[suid]] = ci
        cname = cam_by_sensor[suid]
        info['cameras'][cname]['sessions'].append(si)
        rec = inf['board_record']
        bkey = rec.get('uid') or f'session:{sess.get("uid")}'
        if bkey not in brd_by_key:
            b = prob.boards[0]
            brd_by_key[bkey] = unique(b.name)
            brds[brd_by_key[bkey]] = HE.BoardSpec(brd_by_key[bkey], b.points, b.robot, b.link, board=b.board)
            info['boards'][brd_by_key[bkey]] = {'record': rec, 'Z0': inf['Z0']}
        bname = brd_by_key[bkey]
        info['sessions'][-1].update(camera=cname, board=bname)
        robots.update(prob.robots)
        for o, k in zip(prob.observations, inf['pose_index']):
            q_by = prob.joints[o.pose]
            key = tuple((r, tuple(np.round(np.asarray(q, float) / 1e-9).astype(np.int64))) for r, q in
                        sorted(q_by.items()))
            if key not in pose_of:
                pose_of[key] = len(joints)
                joints.append({r: np.asarray(q, float) for r, q in q_by.items()})
            vname = f'{info["sessions"][-1]["name"]} · {o.name}'
            while vname in info['views']:
                vname += "'"
            info['views'][vname] = (si, k)
            obs.append(HE.Observation(cname, bname, pose_of[key], o.ids, o.uv, vname))
    info['robots'] = list(robots)
    return HE.Problem(list(cams.values()), list(brds.values()), robots, joints, obs), info
