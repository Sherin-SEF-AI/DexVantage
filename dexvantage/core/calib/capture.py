"""Recording for hand-eye calibration: offline capture from a folder of board images and a joint_states file, matched
by time, or from a recorded dataset (a LeRobot episode, ROS 2 topics in a bag), detected, and turned into a session
(records.py) and a hand-eye Problem (handeye.py).

Sources (SOURCES): 'folder' (capture_folder) and 'dataset' (capture_dataset: LeRobot v2.1 / v3.0, ROS 2 bags in MCAP
or SQLite, RLDS, through core/datasets); the older names 'ros2' and 'lerobot' mean 'dataset' (check_source). Live
capture from running ROS 2 topics is not done here: record a bag and read it.

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

A recorded dataset (capture_dataset(project, dataset, sensor_uid, episodes, every_n)):
- the camera stream: the camera key given, else the dataset's only camera, else the one whose key ends with (or
  holds) the sensor's name in lower case with its parenthesised part left out ('Wrist D405' ->
  'observation.images.wrist_d405' or '/wrist_d405/image_raw'); otherwise the capture asks which, listing the keys;
- the joints: each episode's state put on the robot's model by name (core/datasets/joints.on_model; unit 'rad' or
  'deg');
- every every_n-th frame of the stream is looked at. Its joint reading is the state interpolated linearly at the
  frame's time t + offset_s (default 0 s: the camera and the joints on one clock, a default; the timestamps the
  dataset stores, LeRobot's frame timestamps or a bag's header stamps carried onto the joint clock by the reader, not
  device clocks). A frame outside the joint rows' time span, or before every joint is recorded, is left out. The
  joints' speed there is the fastest joint's over the row interval holding the frame; with still_only (default) a
  frame taken faster than MOVING_DEG_S is left out, since the solve assumes still poses (and a moving frame's
  interpolated reading carries the timing error), otherwise it is kept and marked;
- the board is detected in the frame at the frame's own size: when the stream is a resampled copy of the sensor's
  image (the same shape within ASPECT_TOL, 1 %), the camera model is scaled to it (CameraModel.scaled) and the
  corners are carried back to the model's pixels, u = s (u' + ½) − ½; a stream of another shape is refused;
- with merge_still (default), consecutive still frames of one episode whose joints all lie within STILL_SAME_DEG
  (0.05°, a default) of the run's first frame are one pose: the frame with the most corners is kept (the middle one
  among equals), so a robot that holds a pose for two seconds at 30 fps gives one view, not sixty correlated ones; at
  most MAX_VIEWS (500, a default) views are kept, evenly spread;
- only frames with the board found become poses (named 'episode E frame I'); the source records the dataset's folder
  or file name (never its full path), its format and version, the camera key, per episode the frames looked at, with
  joints, still and with the board, the camera's first time minus the joints' first time and the median distance of
  a frame from its nearest joint row (the timestamp offset, said in the notes), and every rule above that applied.

Several sessions together (sessions_problem): one bundle adjustment over every camera, board and robot of the
sessions, a camera or board shared between sessions estimated once; this is how a static camera that only sees a
board fixed in the world is calibrated (through a robot-mounted camera's views of the same board).

Known limits: a folder capture takes the nearest sample, not an interpolation (calibration poses are taken still); a
dataset capture interpolates linearly between joint rows (a joint path that curves between rows is cut short); one
camera and one robot per session (several are solved together with sessions_problem); image and joint clocks are taken
as the same clock up to the offset given (no offset is estimated; a constant offset shows only in moving frames, which
are left out by default); images are read as files OpenCV can decode; a dataset camera's own intrinsics are compared
with the sensor's lens model and said when they differ, never used; motion blur and rolling shutter in moving frames
are not modelled.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import re

import numpy as np

SOURCES = ('folder', 'dataset')
SOURCE_LABELS = {'folder': 'Image folder and joint states file', 'dataset': 'LeRobot dataset or ROS 2 bag'}
SOURCE_ALIASES = {'ros2': 'dataset', 'lerobot': 'dataset', 'bag': 'dataset', 'rlds': 'dataset'}
TOLERANCE_S = 0.05             # a default: an image takes a joint sample this close in time
STILL_SAME_DEG = 0.05          # a default: still dataset frames whose joints agree this closely are one view
ASPECT_TOL = 0.01              # frames and camera model of the same shape within this share
MAX_VIEWS = 500                # a default: at most this many views from a dataset
MOVING_DEG_S = 0.5             # a default: faster joint motion at the match flags the image
IMAGE_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.pgm', '.ppm')
TIME_COLUMNS = ('t', 'time', 'stamp', 'timestamp', 'time_s')
SIDECARS = ('images.csv', 'timestamps.csv', 'images.json')


class CaptureError(ValueError):
    """A capture that cannot be read; the message says why in plain words."""


def check_source(kind: str) -> str:
    """The capture source a name stands for ('folder' or 'dataset'; 'ros2', 'lerobot', 'bag' and 'rlds' are
    'dataset': ROS 2 topics are read from a recorded bag); CaptureError for any other name."""
    kind = SOURCE_ALIASES.get(str(kind), str(kind))
    if kind not in SOURCES:
        raise CaptureError(f'Unknown capture source "{kind}": use one of {", ".join(SOURCES)}.')
    return kind


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
def _session_context(project, sensor, board_rec):
    """(imager, mode, board record, board, robot uid, link, robot model) of a capture of sensor; CaptureError."""
    from ..robot.model import RobotModelError, get_model
    from . import records as RC
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
    try:
        mdl = get_model(project, robot_uid)
    except RobotModelError as e:
        raise CaptureError(str(e)) from None
    return im, mode, rec, board, robot_uid, link, mdl


def capture_folder(project, sensor, folder: str, joints, board_rec: dict | None = None,
                   tolerance_s: float = TOLERANCE_S, unit: str = 'rad', progress=None, cancel=None) -> dict:
    """A session record (records.py) from a folder of board images and a joint_states file (path or text): times
    matched, every image detected with the sensor's camera model. Raises CaptureError (a plain sentence)."""
    import cv2

    from . import records as RC
    from .detect import detect
    if not os.path.isdir(folder):
        raise CaptureError('The image folder does not exist.')
    im, mode, rec, board, robot_uid, link, mdl = _session_context(project, sensor, board_rec)
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


# =========================================================================== a session from a dataset
def _key_words(text: str) -> str:
    t = re.sub(r'\([^)]*\)', ' ', str(text).lower())
    return '_'.join(w for w in re.split(r'[^a-z0-9]+', t) if w)


def camera_key_for(sensor, keys) -> str | None:
    """The dataset camera key that names the sensor (module docstring), or None."""
    keys = list(keys)
    if len(keys) == 1:
        return keys[0]
    want = _key_words(getattr(sensor, 'name', ''))
    if not want:
        return None
    hits = [k for k in keys if _key_words(k).endswith(want)]
    if len(hits) == 1:
        return hits[0]
    hits = [k for k in keys if want in _key_words(k)]
    return hits[0] if len(hits) == 1 else None


def _speeds(t, Q, x) -> np.ndarray:
    """(M,) deg/s: the fastest joint's speed at times x, from the rows' finite differences (the segment holding x
    and, at a row's own time, the segment before it too)."""
    t = np.asarray(t, float)
    if len(t) < 2:
        return np.zeros(len(x))
    dt = np.diff(t)
    with np.errstate(divide='ignore', invalid='ignore'):
        V = np.where(dt[:, None] > 0, np.diff(Q, axis=0) / np.where(dt > 0, dt, 1.0)[:, None], 0.0)
    v = np.degrees(np.nanmax(np.abs(V), axis=1)) if V.shape[1] else np.zeros(len(dt))
    k = np.clip(np.searchsorted(t, x, side='right') - 1, 0, len(dt) - 1)
    sp = v[k]
    at = np.isclose(x, t[np.clip(k, 0, len(t) - 1)], rtol=0, atol=1e-9) & (k > 0)
    sp[at] = np.maximum(sp[at], v[k[at] - 1])
    return sp


def _scaled_model(cam, w: int, h: int):
    """(the camera model at the frames' size, (sx, sy)) or CaptureError when the shapes differ."""
    if (w, h) == (cam.width, cam.height):
        return cam, (1.0, 1.0)
    if abs((w / h) / (cam.width / cam.height) - 1.0) > ASPECT_TOL:
        raise CaptureError(f'The dataset\'s frames are {w} × {h} px and the camera model is {cam.width} × '
                           f'{cam.height} px: a different shape (a crop?), so the board corners cannot be placed in '
                           'the model\'s pixels. Choose the camera stream of this sensor, or set the sensor\'s stream '
                           'size to the recording\'s.')
    return cam.scaled(w, h), (cam.width / w, cam.height / h)


def _merge_still(views: list) -> tuple[list, int]:
    """One view per run of still frames at one pose (module docstring): (kept views, frames merged away)."""
    out, run = [], []

    def flush():
        if run:
            best = max(len(v['ids']) for v in run)
            cand = [i for i, v in enumerate(run) if len(v['ids']) == best]
            out.append(run[min(cand, key=lambda i: abs(i - (len(run) - 1) / 2))])
    for v in views:
        if run and v['still'] and run[-1]['still'] and v['episode'] == run[-1]['episode'] and \
                np.degrees(np.max(np.abs(np.asarray(v['q']) - np.asarray(run[0]['q'])))) <= STILL_SAME_DEG:
            run.append(v)
            continue
        flush()
        run = [v]
    flush()
    return out, len(views) - len(out)


def capture_dataset(project, dataset, sensor_uid: str, episodes=None, every_n: int = 1, *, camera: str | None = None,
                    board_rec: dict | None = None, unit: str = 'rad', offset_s: float = 0.0, still_only: bool = True,
                    merge_still: bool = True, max_views: int = MAX_VIEWS, progress=None, cancel=None) -> dict:
    """A session record (records.py) from a recorded dataset (spec §8.1 'a LeRobot episode', 'ROS 2 topics'): the
    frames of the sensor's camera stream with the board in view and the joint states at the frame times
    (interpolated; module docstring). dataset: a core.datasets Dataset or a path open_dataset reads (LeRobot v2.1 /
    v3.0, a ROS 2 bag in MCAP or SQLite, RLDS). episodes: the dataset's episode indices (None: all); every_n: look at
    every Nth frame. Raises CaptureError (a plain sentence)."""
    from ..datasets import DatasetError, open_dataset
    from ..datasets.joints import interpolate, on_model
    from . import records as RC
    from .detect import detect, to_gray
    sensor = project.variant.sensor(sensor_uid) if sensor_uid else None
    if sensor is None:
        raise CaptureError('The camera to calibrate is not in this rig: choose one of its cameras.')
    try:
        every_n, offset_s, max_views = int(every_n), float(offset_s), int(max_views)
    except (TypeError, ValueError):
        raise CaptureError('"Every N frames" and the view limit must be whole numbers and the time offset a number '
                           'of seconds.') from None
    if every_n < 1 or max_views < 3 or not math.isfinite(offset_s):
        raise CaptureError('"Every N frames" must be 1 or more, the view limit 3 or more and the time offset finite.')
    im, mode, rec, board, robot_uid, link, mdl = _session_context(project, sensor, board_rec)
    own = isinstance(dataset, str)
    try:
        ds = open_dataset(dataset) if own else dataset
    except DatasetError as e:
        raise CaptureError(str(e)) from None
    try:
        refs = ds.episodes()
        have = [r.index for r in refs]
        if not have:
            raise CaptureError('The dataset has no episodes.')
        want = have if episodes is None else [int(e) for e in (episodes if isinstance(episodes, (list, tuple, set,
                                                                                                  range))
                                                                else [episodes])]
        bad = [e for e in want if e not in set(have)]
        if bad:
            raise CaptureError(f'The dataset has no episode {bad[0]} (its episodes are {have[0]} to {have[-1]}).')
        base = os.path.basename(os.path.normpath(str(getattr(ds, 'path', '') or ''))) or 'dataset'
        cam0 = im.camera
        views, notes, ep_docs, map_notes = [], [], [], []
        looked = matched = found = moving = outside = 0
        key = camera
        det_cam, scale = None, (1.0, 1.0)
        steps = False
        for n_ep, e in enumerate(want):
            if cancel is not None and cancel():
                notes.append(f'Stopped after {n_ep} of {len(want)} episodes.')
                break
            ep = ds.load(e)
            if key is None:
                key = camera_key_for(sensor, ep.cameras)
                if key is None:
                    keys = sorted(ep.cameras)
                    raise CaptureError(
                        'The dataset records no camera.' if not keys else
                        f'Which camera stream is {sensor.name}? The dataset has {", ".join(keys[:6])}'
                        f'{"…" if len(keys) > 6 else ""}: choose one.')
            if key not in ep.cameras:
                raise CaptureError(f'Episode {e} has no camera stream {key} (it has '
                                   f'{", ".join(sorted(ep.cameras)) or "none"}).')
            cs = ep.cameras[key]
            steps = steps or (ep.meta or {}).get('time_base') == 'steps'
            jm = on_model(ep, mdl, unit)
            for t_ in jm.notes:
                if t_ not in map_notes:
                    map_notes.append(t_)
            ts = np.asarray(cs.timestamps, float).reshape(-1)
            rows_t = np.asarray(ep.t, float).reshape(-1)
            idx = np.arange(0, len(ts), every_n)
            tq = ts[idx] + offset_s
            Qf = interpolate(rows_t, jm.Q, tq)
            sp = _speeds(rows_t, jm.Q, tq)
            near = np.array([float(rows_t[np.argmin(np.abs(rows_t - x))] - x) for x in tq]) if len(rows_t) else \
                np.full(len(tq), np.nan)
            ok = np.isfinite(Qf).all(axis=1)
            outside += int((~ok).sum())
            still = sp <= MOVING_DEG_S
            take = ok & (still | (not still_only))
            moving += int((ok & ~still).sum())
            looked += len(idx)
            matched += int(take.sum())
            ep_doc = {'episode': int(e), 'id': str(ep.id), 'frames': int(len(ts)), 'looked_at': int(len(idx)),
                      'with_joints': int(ok.sum()), 'still': int((ok & still).sum()), 'board': 0,
                      'camera_start_minus_joints_s': float(ts[0] - rows_t[0]) if len(ts) and len(rows_t) else None,
                      'nearest_row_ms_median': float(1000 * np.median(np.abs(near[ok]))) if ok.any() else None}
            sel = [int(i) for i in idx[take]]
            pos = {int(i): j for j, i in enumerate(idx)}
            if cs.intrinsics and cs.intrinsics.get('K') is not None and n_ep == 0:
                try:
                    fx = float(np.asarray(cs.intrinsics['K'], float).reshape(3, 3)[0, 0])
                    w0 = int(cs.intrinsics.get('width') or cs.width or cam0.width)
                    fx_m = cam0.fx * w0 / cam0.width
                    if fx_m > 0 and abs(fx / fx_m - 1.0) > 0.01:
                        notes.append(f'The dataset\'s own intrinsics give fx {fx:.1f} px at {w0} px wide; the '
                                     f'sensor\'s lens model {fx_m:.1f} px. The solve uses the lens model: refine the '
                                     'intrinsics, or enter the calibration on the Sensors page.')
                except (TypeError, ValueError, KeyError):
                    pass
            for k_, (i, img) in enumerate(cs.frames(sel) if sel else []):
                if cancel is not None and cancel():
                    break
                if progress and k_ % 10 == 0:
                    progress((n_ep + k_ / max(len(sel), 1)) / len(want),
                             f'Episode {e}: looking for the board in frame {i}')
                h, w = img.shape[:2]
                if det_cam is None or (det_cam.width, det_cam.height) != (w, h):
                    det_cam, scale = _scaled_model(cam0, w, h)
                    if scale != (1.0, 1.0):
                        notes.append(f'The frames are {w} × {h} px, the camera model {cam0.width} × {cam0.height} '
                                     'px: the board is found in the frames and its corners are scaled to the model\'s '
                                     'pixels (the same lens, resampled).')
                det = detect(to_gray(img), board, det_cam)
                if det is None:
                    continue
                uv = np.asarray(det.corners, float).reshape(-1, 2)
                uv = np.stack([(uv[:, 0] + 0.5) * scale[0] - 0.5, (uv[:, 1] + 0.5) * scale[1] - 0.5], axis=1)
                j = pos[int(i)]
                found += 1
                ep_doc['board'] += 1
                note = '' if still[j] else f'the robot was moving ({sp[j]:.2f}°/s at this frame)'
                views.append({'name': f'episode {e} frame {i}', 'image': f'{key} frame {i}',
                              't_image': float(ts[i]), 't_joints': float(tq[j]), 'dt_s': float(near[j]),
                              'q': [float(v) for v in Qf[j]], 'joint_names': list(mdl.joint_names),
                              'ids': [int(x) for x in det.ids], 'uv': uv.tolist(), 'note': note,
                              'episode': int(e), 'still': bool(still[j])})
            ep_docs.append(ep_doc)
        if progress:
            progress(1.0, f'The board found in {found} frames')
        if not views:
            why = []
            if outside:
                why.append(f'{outside} had no joint reading around them')
            if moving and still_only:
                why.append(f'{moving} were taken while the robot moved (look at moving frames too, or record still '
                           'poses)')
            raise CaptureError(f'The board was not found in any of the {looked} frames of {key} looked at'
                               + (f' ({"; ".join(why)})' if why else '') + '. Check the camera stream, the board '
                               'definition and that the board is in view.')
        merged = 0
        if merge_still:
            views, merged = _merge_still(views)
        if len(views) > max_views:
            pick = np.unique(np.linspace(0, len(views) - 1, max_views).round().astype(int))
            notes.append(f'{len(views)} views with the board; {len(pick)} evenly spread ones are kept (at most '
                         f'{max_views}, {"a default" if max_views == MAX_VIEWS else "yours"}).')
            views = [views[i] for i in pick]
        same_clock = ('The camera and joint times are taken as one clock' if offset_s == 0 else
                      f'Joint times are read {1000 * offset_s:+.1f} ms from the frame times')
        timing = (f'Joint states interpolated linearly at each frame\'s time. {same_clock} (offset '
                  f'{1000 * offset_s:.1f} ms{", a default" if offset_s == 0 else ", yours"}).')
        med = [d['nearest_row_ms_median'] for d in ep_docs if d['nearest_row_ms_median'] is not None]
        if med:
            timing += f' The nearest joint row is {float(np.median(med)):.1f} ms from a frame (median).'
        notes.insert(0, timing)
        if merged:
            notes.append(f'{merged + len(views)} frames with the board; still frames at one pose are one view (joints '
                         f'within {STILL_SAME_DEG:g}°, a default): {len(views)} views.')
        if moving:
            notes.append(f'{moving} frames were taken while the robot moved faster than {MOVING_DEG_S:g}°/s (a '
                         'default)' + (': left out (the solve assumes still poses).' if still_only else
                                       ': kept and marked.'))
        if outside:
            notes.append(f'{outside} frames had no joint reading around them (outside the joint states\' time span, '
                         'or joints not yet recorded) and were left out.')
        if steps:
            notes.append('The dataset stores no times (RLDS): frames and joint rows are paired by step.')
        notes += map_notes
        poses = [{k_: v for k_, v in p.items() if k_ not in ('episode', 'still')} for p in views]
        src = {'kind': 'dataset', 'folder': base, 'dataset': base, 'format': str(getattr(ds, 'format', '')),
               'version': str(getattr(ds, 'version', '')), 'camera': key, 'episodes': ep_docs, 'every_n': every_n,
               'images': looked, 'matched': matched, 'board_frames': found, 'detected': len(poses),
               'offset_s': offset_s, 'times': 'joint states interpolated at the frame times', 'notes': notes}
        sess = RC.session_record(sensor, im, mode, str(rec.get('uid') or ''), robot_uid, link, poses, src,
                                 name=f'{sensor.name} · {base}',
                                 settings={'every_n': every_n, 'unit': unit, 'offset_s': offset_s,
                                           'still_only': bool(still_only), 'merge_still': bool(merge_still),
                                           'offset_default': offset_s == 0.0})
        sess['board'] = None if RC.find_board(project, rec.get('uid')) else rec
        return sess
    except DatasetError as e:
        raise CaptureError(str(e)) from None
    finally:
        if own:
            ds.close()


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
