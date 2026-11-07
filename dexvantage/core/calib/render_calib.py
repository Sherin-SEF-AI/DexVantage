"""Targetless link-silhouette extrinsic calibration of a static camera (spec §8.4): no board, only the robot's own
links. The robot is rendered at the logged joint states q(t) through MuJoCo's offscreen segmentation
(core/sim/render.Renderer, the mount perturbed by extrinsic_delta) and the rendered silhouette is aligned with the
recorded image's edges and foreground mask, optimising the mount correction ΔT of the camera against the robot base
(the world).

    estimate(dataset_or_episode, project, sensor_uid, frames=None, settings=None, *, renderer=None, progress=None,
             cancel=None) -> dict
    drift(dataset, project, sensor_uid, episodes=None, settings=None, ...) -> dict   (one estimate per episode, and
                                                                                     the steps between them)
    settings_of(project, given=None) -> dict          (the labelled defaults filled in)

The estimate's document (JSON-able): 'pose' (the recovered optical frame in the world, position and quaternion x, y,
z, w), 'mount' (the recovered sensor body pose in its parent frame: what Apply writes), 'delta' {'position_m',
'rotation_deg'} (ΔT in the sensor body frame, T_world_sensor · ΔT, the rotation a rotation vector in degrees: the
same convention as the synthetic episodes' mount_slip fault and Renderer's extrinsic_delta), 'delta_optical' (the same
correction in the optical frame), 'sigma' {'position_mm', 'rotation_deg' (the larger of the two estimates per axis),
'hessian', 'bootstrap'}, 'frames_used', 'observable', 'why', 'observability' (the eigen test), 'fit' (IoU and chamfer
before and after, per frame), 'robot', 'camera', 'settings', 'notes'.

Method
- Frames: up to max_frames (default 12) of the camera's frames, chosen by farthest-point sampling on the joint
  states at the frames' timestamps (q(t) linearly interpolated from the state rows), so the poses differ as much as
  the episode allows.
- Observed features, per frame, at each level of an image pyramid (default scales 1/4, 1/2, 1 of a working size at
  most max_width = 640 px wide; levels under 160 px wide are left out): Canny edges of the 1-σ smoothed colour image
  (thresholds 50 and 150, defaults), each edge pixel moved to its sub-pixel position along the image gradient (a
  parabola through the gradient magnitude at −1, 0, +1 px) with the gradient direction as its normal; and a foreground
  mask F: pixels that differ from the median of the frames used by more than fg_threshold (default 25 grey levels) in
  some colour channel, opened with a 3 × 3 square, whose boundary (half a pixel out of the mask, along its smoothed
  normal) is a second edge set. The median is the static background where the robot moves; where it does not move
  the median is the robot itself.
- Rendered features, at the current estimate: the robot's mask S (every pixel whose segmentation owner is a robot
  link), its contour (inner boundary pixels moved half a pixel towards the outside neighbour, with the smoothed mask's
  outward normal) and, inside the mask, depth discontinuities larger than inner_jump_m (default 2 cm: one link in
  front of another). A contour pixel next to a nearer body (an object in front of the robot) or within 2 px of the
  image's border is left out (the edge there is not the robot's). Each contour point is back-projected with its
  rendered depth to a world point X; at most max_points per frame (default 400), one per point_spacing_px cell
  (default 3 px). Only outer contour points away from M (the pixels inside S in most frames, dilated 2 px) are matched
  to F's boundary: there F's boundary is the robot's own outline.
- Cost: a truncated directional chamfer distance (Liu et al. 2010) through distance transforms
  (cv2.distanceTransformWithLabels, Euclidean, one per 30° bin of edge orientation): a point projecting to u is matched
  to the nearest edge pixel e whose normal n_e is within its own bin or the next ones, and its residual is
  r = n_e · (u − e) (the point-to-tangent distance). Points whose match is beyond the truncation are outliers: it is
  truncate_px (default 10 px) at the coarsest level and tightens with the levels (10, 5, 3 px at 1/4, 1/2, 1; never
  under 3 px), so clutter edges and edges the robot does not make do not pull. The loss is Huber (huber_px, default
  1 px); F's boundary residuals are weighted by mask_weight (default 1). The foreground term of the coarse search is
  the IoU of F with the mask the render predicts for it, S_i XOR M: pixels whose robot coverage differs from the
  majority are the ones that differ from the median.
- Optimisation: ΔT is a 6-vector ξ = (ρ, φ) composed on the right of the current mount, T_world_sensor · ΔT · Exp(ξ).
  (1) Coarse search on the coarsest level: a pattern search over the twelve moves ±step along and about the sensor
  body's axes, every move rendered (a finite-difference search, no gradient), on J = mean truncated chamfer /
  truncate_px + iou_weight (default 1) × (1 − IoU); the steps start at 4 coarse pixels at the robot's median depth and
  halve when no move improves J, down to 0.5 px. (2) Gauss-Newton with Levenberg-Marquardt damping on every level,
  coarse to fine: the contour points are re-rendered at each outer iteration (default 4) and held fixed for the inner
  iterations (default 8). The Jacobian is analytic in the pose (∂p/∂ρ = −R_so^T, ∂p/∂φ = R_so^T [y]_×, y the point in
  the sensor body frame) and in the edge (∂r/∂u = n_e); the lens projection's Jacobian ∂u/∂p is by central differences
  of optics.CameraModel.project (any lens model).
- Uncertainty: (a) the Gauss-Newton covariance σ̂² (JᵀJ)⁻¹ at the solution on the finest level, σ̂² the inliers'
  residual variance, with frames of the same pose (no joint differing by min_motion_deg, default 5°) counted once (their
  silhouettes repeat the same errors); (b) a cluster-robust (sandwich) covariance with the scores summed per (distinct
  pose, link), plus a systematic floor: each cluster's contour offset along its normals by edge_floor_px (default
  0.25 px); (c) a bootstrap over frames: bootstrap (default 30) resamples of the frames with replacement, each
  re-solved from the solution on the resampled frames' fixed contour points. The headline σ is the largest of the
  three, per axis. 'scale' gives one pixel of the working image at the robot's median depth (mm and degrees).
- Observability: the information matrix I = σ̂⁻² JᵀJ (pose-deduplicated) scaled by the requirement, N = S I S with
  S = diag(need_mm × 3 in metres, need_deg × 3 in radians) (defaults 5 mm and 0.5°); the estimate is observable when
  the smallest eigenvalue of N is at least 1 (every direction's σ within the requirement), the robot covers at least
  min_robot_frac of the image (default 0.5 %) with at least min_points contour points per frame (default 30), and the
  frames hold at least min_poses distinct joint states (default 3). 'why' says which test failed in words: the robot
  too small in the image, too static (insufficient joint motion), or the weakest direction.

Known limits: the robot's links must look like its model (no covers, cables or tools that the model lacks; these edges
are outliers at best and pull the fit at worst); the intrinsics are taken as known (the stream's own when the dataset
stores them, else the project's imager scaled to the stream size); the joint readings are taken as exact at the
frame's timestamp (a camera–joint offset or a moving robot under a rolling shutter shifts the silhouette, and a
desync shows as a mount error); the median background needs the robot to move; the edge test uses the robot's outer
contour and depth jumps, not its texture; a small, distant robot leaves sideways shift and yaw nearly
interchangeable, and σ can then be a lower bound along that direction (measured up to 7 σ on the SO-100 sample's
third-person camera, where the estimate carries a bias of about 2.4 mm and 0.3° while steps between episodes are
recovered to about 1 mm; docs/models/render_calib.md); the bootstrap re-solves on the final contour points (it does
not re-render), so it measures the frame-to-frame scatter at the solution, not the capture range; the coarse search
can stop in a local minimum when the start is more than about the truncation distance away (tens of millimetres or a
few degrees at a metre).
"""
from __future__ import annotations

import base64
import math

import numpy as np

from ..jobs import StudyError

SID = 'silhouette_calib'
DEFAULTS = {
    'dataset': None,               # the study: a recorded dataset's path
    'sensor': None,                # the study: the static camera's uid (None: the first)
    'episode': None,               # estimate on a Dataset: which episode (None: the first)
    'episodes': None,              # the study's drift: episode indices (None: the first max_episodes)
    'max_episodes': 8,             # default: episodes the study estimates at most
    'camera': None,                # the dataset's camera key (None: matched to the sensor by name)
    'max_frames': 12,              # default: frames used per estimate
    'max_width': 640,              # default: the working image's width at most (px)
    'levels': [0.25, 0.5, 1.0],    # default: pyramid scales of the working image, coarse to fine
    'truncate_px': 10.0,           # default: chamfer truncation (px of each level)
    'canny': [50.0, 150.0],        # default: Canny thresholds
    'fg_threshold': 25.0,          # default: foreground threshold (grey levels)
    'inner_jump_m': 0.02,          # default: depth jump that makes an inner contour (m)
    'point_spacing_px': 3.0,       # default: one contour point per cell of this size (px)
    'max_points': 400,             # default: contour points per frame at most
    'outer_iters': 4,              # default: re-renders per level
    'inner_iters': 8,              # default: Gauss-Newton steps per render
    'coarse_search': True,         # default: the pattern search before Gauss-Newton
    'iou_weight': 1.0,             # default: weight of 1 − IoU in the coarse search's cost
    'mask_weight': 1.0,            # default: weight of the foreground-boundary residuals against the edges'
    'huber_px': 1.0,               # default: Huber loss threshold of the residuals (px)
    'edge_floor_px': 0.25,         # default: systematic edge placement error per link contour (px), in the σ
    'bootstrap': 30,               # default: bootstrap resamples over frames
    'need_mm': 5.0,                # default: observability requirement, position σ (mm)
    'need_deg': 0.5,               # default: observability requirement, rotation σ (°)
    'min_robot_frac': 0.005,       # default: the robot's share of the image at least
    'min_points': 30,              # default: contour points per frame at least
    'min_poses': 3,                # default: distinct joint states at least
    'min_motion_deg': 5.0,         # default: two frames are distinct poses when a joint differs by this much
    'tolerance_mm': 2.0,           # default: the page's status: a mount within this holds
    'tolerance_deg': 0.2,          # default: the page's status: a mount within this holds
    'step_z': 3.0,                 # default: a change between episodes over this many σ is a step
    'objects': True,               # render the project's objects (they hide the robot where they are in front)
    'device': 'auto',
    'seed': 0,
}
EDGE_BORDER_PX = 2                 # contour pixels this close to the image border are left out
COARSE_START_PX = 4.0              # the coarse search's first step (coarse pixels)
COARSE_END_PX = 0.5                # ... and its last
COARSE_SWEEPS = 24                 # ... at most this many sweeps of twelve moves
MIN_LEVEL_W = 160                  # pyramid levels narrower than this are left out (px)
MIN_TRUNC_PX = 3.0                 # the truncation never goes below this (px of the level)
MAX_VIEWS_PER_CALL = 96            # renders per worker call at most
ORIENT_BINS = 6                    # directional chamfer: orientation bins of edge normals (30° each)
EDGE_BLUR = 1.0                    # Gaussian σ (px) before Canny
LM_LAMBDA = 1e-3                   # initial Levenberg-Marquardt damping (relative to the diagonal)
OVERLAY_W = 320                    # the page's thumbnail width (px)
AXES = ('x', 'y', 'z')


class CalibError(StudyError):
    """The silhouette calibration cannot run on these inputs; the message says why."""


# ====================================================================== settings
def settings_of(project, given: dict | None = None) -> dict:
    """The settings (project.analysis.studies['silhouette_calib'], then given over them) with the defaults filled in;
    'defaults' lists the keys left at their default."""
    stored = dict((project.analysis.studies.get(SID) or {}) if project is not None else {})
    st = dict(DEFAULTS)
    for src in (stored, given or {}):
        for k, v in src.items():
            if k in ('defaults',):
                continue
            if k not in DEFAULTS:
                raise CalibError(f'The silhouette calibration has no setting {k}; it has '
                                 f'{", ".join(sorted(DEFAULTS))}.')
            st[k] = v
    for k in ('max_frames', 'max_points', 'outer_iters', 'inner_iters', 'bootstrap', 'min_points', 'min_poses',
              'max_episodes', 'max_width', 'seed'):
        try:
            st[k] = int(st[k])
        except (TypeError, ValueError):
            raise CalibError(f'The setting {k} must be a whole number; it is {st[k]!r}.') from None
        if st[k] < (0 if k in ('bootstrap', 'seed', 'min_points') else 1):
            raise CalibError(f'The setting {k} must be at least {0 if k in ("bootstrap", "seed") else 1}.')
    if st['max_frames'] < 1:
        raise CalibError('The setting max_frames must be at least 1.')
    for k in ('truncate_px', 'fg_threshold', 'inner_jump_m', 'point_spacing_px', 'iou_weight', 'mask_weight',
              'huber_px', 'edge_floor_px', 'need_mm', 'need_deg',
              'min_robot_frac', 'min_motion_deg', 'tolerance_mm', 'tolerance_deg', 'step_z'):
        try:
            st[k] = float(st[k])
        except (TypeError, ValueError):
            raise CalibError(f'The setting {k} must be a number; it is {st[k]!r}.') from None
        if not math.isfinite(st[k]) or st[k] < 0:
            raise CalibError(f'The setting {k} must be a positive number; it is {st[k]!r}.')
    for k in ('truncate_px', 'need_mm', 'need_deg', 'point_spacing_px'):
        if st[k] <= 0:
            raise CalibError(f'The setting {k} must be more than 0.')
    lv = st['levels']
    try:
        lv = [float(x) for x in (lv if isinstance(lv, (list, tuple)) else [lv])]
    except (TypeError, ValueError):
        raise CalibError('The setting levels must be a list of scales such as [0.25, 0.5, 1].') from None
    if not lv or any(not (0 < x <= 1) for x in lv):
        raise CalibError('Each pyramid level must be a scale above 0 and at most 1 of the working image.')
    st['levels'] = sorted(lv)
    c = st['canny']
    if not isinstance(c, (list, tuple)) or len(c) != 2:
        raise CalibError('The setting canny must be two thresholds, such as [50, 150].')
    st['canny'] = [float(c[0]), float(c[1])]
    st['defaults'] = [k for k in DEFAULTS if k not in stored and k not in (given or {})]
    return st


# ====================================================================== small geometry
def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _rotvec_to_R(w):
    w = np.asarray(w, float)
    th = float(np.linalg.norm(w))
    if th < 1e-14:
        return np.eye(3) + _skew(w)
    k = w / th
    K = _skew(k)
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def _R_to_rotvec(R):
    R = np.asarray(R, float)
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    th = math.acos(c)
    if th < 1e-9:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    if math.pi - th < 1e-6:
        w, V = np.linalg.eigh((R + R.T) / 2)
        k = V[:, int(np.argmax(w))]
        return k * th
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return v * th


def exp_se3(xi) -> np.ndarray:
    """4 × 4 of ξ = (ρ, φ): rotation by the rotation vector φ, translation ρ (a first-order exponential: the
    translation is not coupled to the rotation, which is how ΔT is reported: position, then rotation vector)."""
    T = np.eye(4)
    T[:3, :3] = _rotvec_to_R(xi[3:])
    T[:3, 3] = xi[:3]
    return T


def log_se3(T) -> np.ndarray:
    """The inverse of exp_se3: (position, rotation vector)."""
    T = np.asarray(T, float)
    return np.r_[T[:3, 3], _R_to_rotvec(T[:3, :3])]


def compose(D, xi) -> np.ndarray:
    """D · Exp(ξ), with the step's translation applied in D's frame and its rotation about the frame's origin."""
    E = exp_se3(xi)
    return D @ E


def _pose_doc(T) -> dict:
    from ..frames import Pose
    return Pose.from_matrix(np.asarray(T, float)).to_dict()


# ====================================================================== inputs
def _episode_of(obj, st: dict):
    """(Episode, dataset or None, close?) from a Dataset, an Episode or a path."""
    from ..datasets.episode import Dataset, Episode, open_dataset
    if isinstance(obj, Episode):
        return obj, None
    if isinstance(obj, str):
        obj = open_dataset(obj)
    if isinstance(obj, Dataset):
        refs = obj.episodes()
        if not refs:
            raise CalibError('The dataset has no episodes.')
        want = st.get('episode')
        idx = refs[0].index if want is None else int(want)
        if idx not in [r.index for r in refs]:
            raise CalibError(f'The dataset has no episode {idx}; it has {refs[0].index} to {refs[-1].index}.')
        return obj.load(idx), obj
    raise CalibError('The silhouette calibration needs a dataset, an episode or a dataset path.')


def static_sensor(project, sensor_uid: str):
    """The sensor, refused in words when it is missing, not a camera, or on a robot link."""
    s = project.variant.sensor(sensor_uid) if sensor_uid else None
    if s is None:
        raise CalibError('The camera chosen for the silhouette calibration is not in this rig: choose another.')
    if s.kind not in ('camera', 'depth'):
        raise CalibError(f'{s.name} is not a camera, so it has no image to align.')
    if s.mount.on_link:
        raise CalibError(f'{s.name} rides on {s.mount.parent_frame}: the silhouette calibration is for static '
                         'cameras (the robot would move with the camera). Use the hand-eye solve for it.')
    return s


def static_cameras(project) -> list:
    """The enabled camera and depth sensors that are not on a robot link."""
    return [s for s in project.variant.sensors if s.enabled and s.kind in ('camera', 'depth') and not s.mount.on_link]


def match_stream(episode, sensor, key: str | None = None):
    """The episode's CameraStream of the sensor: the key given, else meta sensor_uid, else the key the synthetic
    episodes give the sensor's name, else a key containing the name's words."""
    import re
    cams = episode.cameras or {}
    if not cams:
        raise CalibError('The episode has no camera streams.')
    if key:
        if key not in cams:
            raise CalibError(f'The episode has no camera {key}; it has {", ".join(cams)}.')
        return cams[key]
    for c in cams.values():
        if (c.meta or {}).get('sensor_uid') == sensor.uid:
            return c
    base = re.sub(r'\([^)]*\)', '', str(sensor.name or '')).strip().lower()
    base = re.sub(r'[^a-z0-9]+', '_', base).strip('_')
    for k, c in cams.items():
        if k == f'observation.images.{base}' or k.split('.')[-1] == base:
            return c
    hits = [c for k, c in cams.items() if base and base in re.sub(r'[^a-z0-9]+', '_', k.lower())]
    if len(hits) == 1:
        return hits[0]
    raise CalibError(f'No camera of the episode matches {sensor.name}: it has {", ".join(cams)}. Choose the camera '
                     'key in the settings.')


def robot_map(project, episode):
    """(robot uid, RobotModel, index map): the robot whose joint names the episode's state carries most of; index
    map [(model q index, episode column)]."""
    from ..robot.model import RobotModelError, get_model
    names = [str(n) for n in (episode.joint_names or [])]
    best = None
    for r in project.robots:
        try:
            m = get_model(project, r.uid)
        except RobotModelError:
            continue
        mj = list(m.joint_names)
        pairs = [(mj.index(n), k) for k, n in enumerate(names) if n in mj]
        if best is None or len(pairs) > len(best[2]):
            best = (r.uid, m, pairs)
    if best is None:
        raise CalibError('The project has no robot to render.')
    uid, m, pairs = best
    q = np.asarray(episode.q, float)
    if not pairs:
        if q.ndim == 2 and q.shape[1] == m.nq:
            return uid, m, [(k, k) for k in range(m.nq)], ['The episode names no joint of the robot: its state '
                                                           'columns are read in the model\'s joint order.']
        raise CalibError(f'The episode\'s joint states ({", ".join(names[:4])}…) name no joint of '
                         f'{project.robot(uid).name}, and their count differs from its {m.nq} joints.')
    notes = []
    if len(pairs) < m.nq:
        miss = [n for n in m.joint_names if n not in names]
        notes.append(f'The episode does not record {", ".join(miss[:6])}: held at home.')
    return uid, m, pairs, notes


def q_at(episode, model, pairs, times) -> np.ndarray:
    """(M, nq) model joint vectors at the given times: each recorded joint linearly interpolated (held at the ends),
    the others at home."""
    t = np.asarray(episode.t, float)
    Q = np.asarray(episode.q, float)
    out = np.tile(np.asarray(model.home, float), (len(times), 1))
    for mi, k in pairs:
        out[:, mi] = np.interp(times, t, Q[:, k])
    return out


def choose_frames(Q: np.ndarray, n: int, candidates=None) -> list:
    """Farthest-point sampling of up to n frames on the joint vectors Q (M, nq): the first is the frame farthest from
    the mean pose, each next the one farthest (max-norm) from those chosen. Sorted."""
    M = len(Q)
    idx = list(range(M)) if candidates is None else [int(i) for i in candidates if 0 <= int(i) < M]
    if len(idx) <= n:
        return sorted(idx)
    P = Q[idx]
    first = int(np.argmax(np.max(np.abs(P - P.mean(0)), axis=1)))
    chosen = [first]
    d = np.max(np.abs(P - P[first]), axis=1)
    while len(chosen) < n:
        k = int(np.argmax(d))
        if d[k] <= 0:
            k = next(i for i in range(len(idx)) if i not in chosen)
        chosen.append(k)
        d = np.minimum(d, np.max(np.abs(P - P[k]), axis=1))
    return sorted(idx[k] for k in chosen)


def pose_clusters(Q: np.ndarray, min_motion_rad: float) -> list:
    """Cluster labels of joint vectors: frames whose joints all agree within min_motion_rad of a cluster's first frame
    share it."""
    lab, heads = [], []
    for q in Q:
        for c, h in enumerate(heads):
            if np.max(np.abs(q - h)) < min_motion_rad:
                lab.append(c)
                break
        else:
            heads.append(q)
            lab.append(len(heads) - 1)
    return lab


def camera_model_of(stream, scene_cam, notes: list):
    """The lens model of the recorded images: the stream's intrinsics when stored, else the scene camera's (the
    project's imager) scaled to the stream's size."""
    from ..optics import CameraModel
    W, H = stream.width, stream.height
    intr = stream.intrinsics
    if isinstance(intr, dict) and intr.get('K') is not None:
        try:
            m = str(intr.get('model') or 'pinhole')
            K = np.asarray(intr['K'], float)
            w = int(intr.get('width') or W or scene_cam.model.width)
            h = int(intr.get('height') or H or scene_cam.model.height)
            d = tuple(float(x) for x in (intr.get('dist') or ()))
            if m in ('pinhole', 'brown', 'kb', 'plumb_bob', 'rational_polynomial', 'fisheye'):
                cam = CameraModel.from_opencv(K, d, w, h, m)
            else:
                cam = CameraModel(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2], m, d)
            notes.append('The lens model is the dataset\'s own intrinsics for this camera.')
            return cam
        except (ValueError, KeyError, TypeError) as e:
            notes.append(f'The dataset\'s intrinsics could not be read ({e}); the project\'s imager is used.')
    cam = scene_cam.model
    if W and H and (W, H) != (cam.width, cam.height):
        a0, a1 = cam.width / cam.height, W / H
        if abs(a0 / a1 - 1) > 0.02:
            raise CalibError(f'The recorded images are {W} × {H} px but the project\'s imager is {cam.width} × '
                             f'{cam.height} px, a different shape, so its lens model cannot be matched to them: give '
                             'the dataset\'s intrinsics or choose the imager\'s stream.')
        cam = cam.scaled(int(W), int(H))
    notes.append(f'The lens model is the project\'s imager ({cam.summary()}); the dataset stores no intrinsics.')
    return cam


# ====================================================================== image features
class Edges:
    """Edge pixels of one image at one level, for the chamfer residuals: the distance transform to the nearest edge
    pixel and, per edge pixel, its sub-pixel position and unit normal. Edges.image(rgb, canny) are Canny edges of
    the colour image; Edges.mask(F) the boundary of a binary mask."""

    def __init__(self, E: np.ndarray, ex, ey, nx, ny):
        import cv2
        H, W = E.shape
        self.shape = (H, W)
        self.n_edges = int(E.sum())
        self.ex, self.ey = np.asarray(ex, np.float64), np.asarray(ey, np.float64)
        self.nx, self.ny = np.asarray(nx, np.float64), np.asarray(ny, np.float64)
        # directional chamfer: one distance transform per orientation bin (the normal's angle modulo 180°)
        ys, xs = np.nonzero(E)
        b = _orientation_bin(self.nx, self.ny)
        self.dt, self.lab = [], []
        for k in range(ORIENT_BINS):
            sel = np.nonzero(b == k)[0]
            if not len(sel):
                self.dt.append(None)
                self.lab.append(None)
                continue
            src = np.full((H, W), 255, np.uint8)
            src[ys[sel], xs[sel]] = 0
            dt, lab = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
            # labels number the zero pixels in row-major order (1-based): the order of sel (np.nonzero's order)
            idx = np.r_[-1, sel].astype(np.int32)
            self.dt.append(dt)
            self.lab.append(idx[lab])
        # the plain (all-orientation) distance transform, for the coarse cost
        if self.n_edges:
            src = np.where(E, 0, 255).astype(np.uint8)
            self.dt_all = cv2.distanceTransform(src, cv2.DIST_L2, 5)
        else:
            self.dt_all = np.full((H, W), np.inf, np.float32)

    @classmethod
    def image(cls, rgb: np.ndarray, canny) -> 'Edges':
        """Canny edges of the 1-σ smoothed colour image (OpenCV takes the largest channel gradient), each moved to
        the peak of a parabola through the gradient magnitude at −1, 0, +1 px along its normal."""
        import cv2
        im = cv2.GaussianBlur(np.ascontiguousarray(rgb, np.uint8), (0, 0), EDGE_BLUR)
        E = cv2.Canny(im, float(canny[0]), float(canny[1]), L2gradient=True) > 0
        f = im.astype(np.float32)
        gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
        if gx.ndim == 3:                                  # per pixel, the channel with the largest gradient
            m2 = gx * gx + gy * gy
            k = np.argmax(m2, axis=-1)[..., None]
            gx = np.take_along_axis(gx, k, -1)[..., 0]
            gy = np.take_along_axis(gy, k, -1)[..., 0]
        mag = np.hypot(gx, gy)
        ys, xs = np.nonzero(E)
        nx, ny = gx[ys, xs], gy[ys, xs]
        nn = np.hypot(nx, ny)
        nn[nn == 0] = 1.0
        nx, ny = nx / nn, ny / nn
        m0 = mag[ys, xs]
        mp = _bilinear(mag, xs + nx, ys + ny)
        mm = _bilinear(mag, xs - nx, ys - ny)
        den = mm - 2 * m0 + mp
        with np.errstate(divide='ignore', invalid='ignore'):
            off = np.where(np.abs(den) > 1e-6, 0.5 * (mm - mp) / den, 0.0)
        off = np.clip(np.nan_to_num(off), -0.5, 0.5)
        return cls(E, xs + off * nx, ys + off * ny, nx, ny)

    @classmethod
    def mask(cls, F: np.ndarray) -> 'Edges':
        """The boundary of a binary mask: each inner boundary pixel, moved half a pixel outward along the smoothed
        mask's normal."""
        import cv2
        F = np.asarray(F, bool)
        er = cv2.erode(F.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_REPLICATE) > 0
        E = F & ~er
        Ff = cv2.GaussianBlur(F.astype(np.float32), (0, 0), 1.0)
        gx = cv2.Sobel(Ff, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(Ff, cv2.CV_32F, 0, 1, ksize=3)
        ys, xs = np.nonzero(E)
        nx, ny = -gx[ys, xs], -gy[ys, xs]                 # outward
        nn = np.hypot(nx, ny)
        good = nn > 1e-6
        nx = np.where(good, nx / np.where(good, nn, 1), 0.0)
        ny = np.where(good, ny / np.where(good, nn, 1), 0.0)
        return cls(E, xs + 0.5 * nx, ys + 0.5 * ny, nx, ny)

    def residuals(self, uv: np.ndarray, trunc: float, normals: np.ndarray):
        """(r (N,), edge normals (N, 2), inlier (N,) bool, distance (N,)) of points projected to uv (N, 2) whose
        contour normals are normals (N, 2): each is matched to the nearest edge pixel of its own orientation bin or
        the two next to it (±1.5 bins, 45° at 6 bins), within trunc px; r = n_e · (u − e)."""
        H, W = self.shape
        N = len(uv)
        r = np.zeros(N)
        n = np.zeros((N, 2))
        best = np.full(N, np.inf)
        lab = np.full(N, -1, np.int64)
        ok = np.isfinite(uv).all(axis=1)
        ui = np.zeros(N, np.int64)
        vi = np.zeros(N, np.int64)
        ui[ok] = np.rint(uv[ok, 0]).astype(np.int64)
        vi[ok] = np.rint(uv[ok, 1]).astype(np.int64)
        ok &= (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        if not self.n_edges or not ok.any():
            return r, n, np.zeros(N, bool), best
        b = _orientation_bin(normals[:, 0], normals[:, 1])
        sel = np.nonzero(ok)[0]
        for k in range(ORIENT_BINS):
            if self.dt[k] is None:
                continue
            # points whose bin is k or a neighbour of k
            m = sel[((b[sel] - k) % ORIENT_BINS <= 1) | ((k - b[sel]) % ORIENT_BINS <= 1)]
            if not len(m):
                continue
            d = self.dt[k][vi[m], ui[m]]
            better = d < best[m]
            mm = m[better]
            best[mm] = d[better]
            lab[mm] = self.lab[k][vi[mm], ui[mm]]
        ok &= (best <= trunc) & (lab >= 0)
        s_ = np.nonzero(ok)[0]
        L_ = lab[s_]
        n[s_, 0], n[s_, 1] = self.nx[L_], self.ny[L_]
        r[s_] = n[s_, 0] * (uv[s_, 0] - self.ex[L_]) + n[s_, 1] * (uv[s_, 1] - self.ey[L_])
        return r, n, ok, best


def _orientation_bin(nx, ny) -> np.ndarray:
    """The orientation bin (0 … ORIENT_BINS − 1) of normals: the angle modulo 180°."""
    a = np.mod(np.arctan2(ny, nx), np.pi)
    return np.minimum((a / np.pi * ORIENT_BINS).astype(np.int64), ORIENT_BINS - 1)


def _bilinear(img, x, y):
    H, W = img.shape
    x = np.clip(x, 0, W - 1.001)
    y = np.clip(y, 0, H - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy) + img[y0 + 1, x0] * (1 - fx) * fy +
            img[y0 + 1, x0 + 1] * fx * fy)


def foreground(imgs: list, thr: float) -> list | None:
    """Foreground masks of a static camera's frames: the pixels that differ from the frames' median by more than thr
    in some channel, opened with a 3 × 3 square. None with fewer than 3 frames (no median background)."""
    import cv2
    if len(imgs) < 3:
        return None
    st = np.stack([np.asarray(i, np.int16) for i in imgs])
    bg = np.median(st, axis=0)
    k = np.ones((3, 3), np.uint8)
    out = []
    for i in st:
        m = (np.max(np.abs(i - bg), axis=-1) > thr).astype(np.uint8)
        out.append(cv2.morphologyEx(m, cv2.MORPH_OPEN, k) > 0)
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float | None:
    u = np.logical_or(a, b).sum()
    if u == 0:
        return None
    return float(np.logical_and(a, b).sum() / u)


# ====================================================================== rendered contours
class Contour:
    """A frame's rendered contour points: world points X (K, 3), their pixels uv at the render (K, 2), their image
    normals nrm (K, 2), outer (K,) bool (on the robot's outer contour; False: a depth jump inside it), and the robot
    mask S."""

    def __init__(self, X, uv, outer, S, nrm=None, owner=None):
        self.X, self.uv, self.outer, self.S = X, uv, outer, S
        self.nrm = np.zeros((len(X), 2)) if nrm is None else nrm
        self.owner = np.zeros(len(X), np.int64) if owner is None else owner
        self.mask_ok = np.zeros(len(X), bool)

    def __len__(self):
        return len(self.X)


def contour_points(seg, depth, robot_ids, cam, T_world_opt, st: dict, rng=None) -> Contour:
    """The robot's rendered contour (module docstring) as world points."""
    H, W = seg.shape
    S = np.isin(seg, robot_ids)
    empty = Contour(np.zeros((0, 3)), np.zeros((0, 2)), np.zeros(0, bool), S)
    if not S.any():
        return empty
    import cv2
    d = np.asarray(depth, np.float64)
    pts_u, pts_v, dep, out, nxs, nys, own = [], [], [], [], [], [], []
    Sf = cv2.GaussianBlur(S.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(Sf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Sf, cv2.CV_32F, 0, 1, ksize=3)
    jump = float(st['inner_jump_m'])
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        # pixel p in S whose neighbour p + (dx, dy) is outside S (outer contour) or a farther robot surface (a jump)
        Sn = np.zeros_like(S)
        dn = np.zeros_like(d)
        ys0, ys1 = max(0, -dy), H - max(0, dy)
        xs0, xs1 = max(0, -dx), W - max(0, dx)
        Sn[ys0:ys1, xs0:xs1] = S[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
        dn[ys0:ys1, xs0:xs1] = d[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
        inside = np.zeros_like(S)
        inside[ys0:ys1, xs0:xs1] = True
        outer = S & inside & ~Sn
        outer &= ~((dn > 0) & (dn < d - jump))           # a nearer body in front: its edge, not the robot's
        inner = (S & inside & Sn & (dn > d + jump) & (d > 0)) if jump > 0 else np.zeros_like(S)
        for m, flag in ((outer, True), (inner, False)):
            ys, xs = np.nonzero(m)
            if len(xs):
                pts_u.append(xs + 0.5 * dx)
                pts_v.append(ys + 0.5 * dy)
                dep.append(d[ys, xs])
                own.append(np.asarray(seg[ys, xs], np.int64))
                out.append(np.full(len(xs), flag))
                if flag:                                 # the mask's outward normal
                    a, c = -gx[ys, xs].astype(float), -gy[ys, xs].astype(float)
                    nn = np.hypot(a, c)
                    bad = nn < 1e-6
                    a = np.where(bad, dx, a / np.where(bad, 1, nn))
                    c = np.where(bad, dy, c / np.where(bad, 1, nn))
                else:
                    a, c = np.full(len(xs), float(dx)), np.full(len(xs), float(dy))
                nxs.append(a)
                nys.append(c)
    if not pts_u:
        return empty
    u, v, z, o, nx, ny, ow = (np.concatenate(x) for x in (pts_u, pts_v, dep, out, nxs, nys, own))
    b = EDGE_BORDER_PX
    keep = (z > 0) & (u >= b) & (u <= W - 1 - b) & (v >= b) & (v <= H - 1 - b)
    sp = float(st['point_spacing_px'])                    # one point per spacing cell
    cell = np.floor(u / sp).astype(np.int64) * 100003 + np.floor(v / sp).astype(np.int64)
    idx = np.nonzero(keep)[0]
    if not len(idx):
        return empty
    _, first = np.unique(cell[idx], return_index=True)
    idx = idx[first]
    if len(idx) > st['max_points']:
        rng = rng or np.random.default_rng(0)
        idx = np.sort(rng.choice(idx, st['max_points'], replace=False))
    u, v, z, o, nx, ny, ow = u[idx], v[idx], z[idx], o[idx], nx[idx], ny[idx], ow[idx]
    rays = cam.unproject(np.c_[u, v])
    ok = np.isfinite(rays).all(axis=1) & (rays[:, 2] > 1e-6)
    rays, z, u, v, o, nx, ny, ow = rays[ok], z[ok], u[ok], v[ok], o[ok], nx[ok], ny[ok], ow[ok]
    P = rays * (z / rays[:, 2])[:, None]
    Xw = P @ T_world_opt[:3, :3].T + T_world_opt[:3, 3]
    return Contour(Xw, np.c_[u, v], o, S, np.c_[nx, ny], ow)


# ====================================================================== the problem
class _Problem:
    """The renders, features and residuals of one estimate."""

    def __init__(self, R, sensor, cam_full, imgs_full, Qs, robot_uid, st, notes):
        import cv2
        self.R, self.sensor, self.st = R, sensor, st
        self.notes = notes
        self.robot_uid = robot_uid
        self.Qs = Qs
        self.scam = R.camera(sensor.uid)
        self.T_bs = np.asarray(self.scam.T_body_sensor, float)
        self.T_so = np.asarray(self.scam.T_sensor_optical, float)
        self.robot_ids = [i for i, o in enumerate(R.build.owners) if o[0] == 'robot']
        Wf = cam_full.width
        s0 = min(1.0, st['max_width'] / Wf)
        scales = list(st['levels'])
        kept = [s for s in scales if Wf * s0 * s >= MIN_LEVEL_W] or [scales[-1]]
        if len(kept) < len(scales):
            notes.append(f'Pyramid levels narrower than {MIN_LEVEL_W} px are left out (the robot would be a few '
                         'pixels there).')
        self.levels = []
        for s in kept:
            w = max(16, int(round(Wf * s0 * s)))
            h = max(12, int(round(cam_full.height * w / Wf)))
            cam = cam_full.scaled(w, h)
            ims = [cv2.resize(np.ascontiguousarray(im), (w, h), interpolation=cv2.INTER_AREA) for im in imgs_full]
            # the truncation tightens as the levels refine: truncate_px at the coarsest, truncate_px × (coarsest
            # scale / this scale) at the others (so 10, 5, 3 px at 1/4, 1/2, 1), never under MIN_TRUNC_PX
            tr = max(MIN_TRUNC_PX, st['truncate_px'] * kept[0] / s)
            self.levels.append({'scale': s, 'cam': cam, 'imgs': ims, 'edges': None, 'fg': None, 'fg_edges': None,
                                'trunc': float(tr)})
        self.renders = 0

    def level_features(self, L):
        lv = self.levels[L]
        if lv['edges'] is None:
            lv['edges'] = [Edges.image(im, self.st['canny']) for im in lv['imgs']]
            lv['fg'] = foreground(lv['imgs'], self.st['fg_threshold'])
            lv['fg_edges'] = None if lv['fg'] is None else [Edges.mask(F) for F in lv['fg']]
        return lv

    def T_world_opt(self, D):
        return self.T_bs @ D @ self.T_so

    def render(self, L, Ds):
        """Renders of every frame at each mount correction in Ds (one worker call): [[{'seg','depth'}] per D]."""
        lv = self.levels[L]
        views = []
        for D in Ds:
            for q in self.Qs:
                views.append({'sensor_uid': self.sensor.uid, 'q': {self.robot_uid: q}, 'outputs': ('seg', 'depth'),
                              'camera_model': lv['cam'], 'extrinsic_delta': D})
        out = []
        n = len(self.Qs)
        per = max(1, MAX_VIEWS_PER_CALL // n) * n
        for a in range(0, len(views), per):
            out += self.R.render(views[a:a + per])
        self.renders += len(views)
        return [out[k * n:(k + 1) * n] for k in range(len(Ds))]

    # ------------------------------------------------------------ the coarse cost (rendered at each D)
    def coarse_cost(self, L, imgs_r):
        """(J, chamfer / truncation, IoU or None) of renders at one mount correction (module docstring)."""
        import cv2
        lv = self.level_features(L)
        tr = lv['trunc']
        masks = [np.isin(im['seg'], self.robot_ids) for im in imgs_r]
        M = np.mean(masks, axis=0) > 0.5 if masks else None
        cham, ious = [], []
        b = EDGE_BORDER_PX
        for k, S in enumerate(masks):
            e = lv['edges'][k]
            er = cv2.erode(S.astype(np.uint8), np.ones((3, 3), np.uint8))
            B = S & ~(er > 0)
            B[:b], B[-b:], B[:, :b], B[:, -b:] = False, False, False, False
            cham.append(1.0 if (B.sum() == 0 or not e.n_edges) else float(np.mean(np.minimum(e.dt_all[B], tr))) / tr)
            if lv['fg'] is not None:
                v = iou(S ^ M, lv['fg'][k])
                if v is not None:
                    ious.append(v)
        c = float(np.mean(cham))
        i = float(np.mean(ious)) if ious else None
        return c + (self.st['iou_weight'] * (1 - i) if i is not None else 0.0), c, i

    # ------------------------------------------------------------ Gauss-Newton on fixed contour points
    def points(self, L, D, imgs_r=None) -> list:
        """[Contour] of every frame rendered at D, each with .mask_ok: outer points away from the majority mask M
        (where the observed foreground's boundary is the robot's own contour)."""
        import cv2
        lv = self.levels[L]
        if imgs_r is None:
            imgs_r = self.render(L, [D])[0]
        T = self.T_world_opt(D)
        rng = np.random.default_rng(self.st['seed'])
        cs = [contour_points(im['seg'], im['depth'], self.robot_ids, lv['cam'], T, self.st, rng) for im in imgs_r]
        M = np.mean([c.S for c in cs], axis=0) > 0.5
        Md = cv2.dilate(M.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        H, W = M.shape
        for c in cs:
            if len(c):
                ui = np.clip(np.rint(c.uv[:, 0]).astype(int), 0, W - 1)
                vi = np.clip(np.rint(c.uv[:, 1]).astype(int), 0, H - 1)
                c.mask_ok = c.outer & ~Md[vi, ui]
            else:
                c.mask_ok = np.zeros(0, bool)
        return cs

    def project(self, cam, D, X):
        """uv (N, 2) and points in the optical frame (N, 3) of world points X at mount correction D."""
        Ti = np.linalg.inv(self.T_world_opt(D))
        Pc = X @ Ti[:3, :3].T + Ti[:3, 3]
        uv, _ok = cam.project(Pc)
        return uv, Pc

    def jacobian(self, cam, D, X):
        """(uv (N, 2), J (N, 2, 6)): ∂u/∂ξ of the points at D · Exp(ξ), ξ = 0."""
        Ti = np.linalg.inv(self.T_bs @ D)
        Y = X @ Ti[:3, :3].T + Ti[:3, 3]                     # points in the (current) sensor body frame
        Rso = self.T_so[:3, :3]
        Tos = np.linalg.inv(self.T_so)
        Pc = Y @ Tos[:3, :3].T + Tos[:3, 3]
        uv, _ok = cam.project(Pc)
        h = 1e-6 * np.maximum(np.abs(Pc[:, 2]), 0.01)       # ∂u/∂p: central differences of the lens projection
        Jp = np.zeros((len(X), 2, 3))
        for a in range(3):
            dP = np.zeros_like(Pc)
            dP[:, a] = h
            up, _ = cam.project(Pc + dP)
            um, _ = cam.project(Pc - dP)
            Jp[:, :, a] = (up - um) / (2 * h)[:, None]
        Jx = np.zeros((len(X), 3, 6))
        Jx[:, :, :3] = -Rso.T                                # ∂p/∂ρ
        Ysk = np.zeros((len(X), 3, 3))                       # [y]×
        Ysk[:, 0, 1], Ysk[:, 0, 2] = -Y[:, 2], Y[:, 1]
        Ysk[:, 1, 0], Ysk[:, 1, 2] = Y[:, 2], -Y[:, 0]
        Ysk[:, 2, 0], Ysk[:, 2, 1] = -Y[:, 1], Y[:, 0]
        Jx[:, :, 3:] = np.einsum('ij,njk->nik', Rso.T, Ysk)  # ∂p/∂φ = R_soᵀ [y]×
        return uv, np.einsum('nij,njk->nik', Jp, Jx)

    def residual_sets(self, L, k):
        """[(Edges, which points ('all' | 'mask'), weight)] of frame k at level L."""
        lv = self.level_features(L)
        out = [(lv['edges'][k], 'all', 1.0)]
        if lv['fg_edges'] is not None and self.st['mask_weight'] > 0:
            out.append((lv['fg_edges'][k], 'mask', float(self.st['mask_weight'])))
        return out

    def gn_system(self, L, D, P, frames=None):
        """(A = JᵀWJ, g = JᵀWr, cost, inliers, per-frame [(Σ r², n)], J rows, r, frame of each row, weights); the
        segmentation owner of each row's point is left in self.last_owners."""
        lv = self.level_features(L)
        cam = lv['cam']
        tr = lv['trunc']
        k_h = float(self.st['huber_px'])
        A = np.zeros((6, 6))
        g = np.zeros(6)
        cost, n = 0.0, 0
        per, rows, res, fr, wts, owners = [], [], [], [], [], []
        for k in (range(len(P)) if frames is None else frames):
            c = P[k]
            if not len(c):
                per.append((0.0, 0))
                continue
            uv, Ju = self.jacobian(cam, D, c.X)
            s_f, n_f = 0.0, 0
            for E, which, wset in self.residual_sets(L, k):
                sel = np.ones(len(c), bool) if which == 'all' else c.mask_ok
                if not sel.any():
                    continue
                r, nrm, ok, _dt = E.residuals(uv[sel], tr, c.nrm[sel])
                Jr = np.einsum('ni,nij->nj', nrm[ok], Ju[sel][ok])
                rr = r[ok]
                ow = c.owner[sel][ok]
                fin = np.isfinite(Jr).all(axis=1) & np.isfinite(rr)
                Jr, rr, ow = Jr[fin], rr[fin], ow[fin]
                a = np.abs(rr)
                w = wset * np.where(a <= k_h, 1.0, k_h / np.maximum(a, 1e-12))      # Huber weights
                rho = np.where(a <= k_h, 0.5 * rr * rr, k_h * (a - 0.5 * k_h))
                A += (Jr * w[:, None]).T @ Jr
                g += (Jr * w[:, None]).T @ rr
                # an outlier costs as much as a residual at the truncation, so moving points out of range is not free
                n_out = int(sel.sum()) - len(rr)
                cost += wset * (float(rho.sum()) + n_out * k_h * (tr - 0.5 * k_h))
                n += len(rr)
                s_f += float(rr @ rr)
                n_f += len(rr)
                rows.append(Jr)
                res.append(rr)
                fr.append(np.full(len(rr), k))
                wts.append(w)
                owners.append(ow)
            per.append((s_f, n_f))
        self.last_owners = owners
        return A, g, cost, n, per, rows, res, fr, wts

    def gn(self, L, D, P, iters, frames=None):
        """Levenberg-Marquardt (iteratively reweighted for the Huber loss) from D on fixed points; the new D."""
        lam = LM_LAMBDA
        A, g, cost, n, *_ = self.gn_system(L, D, P, frames)
        for _ in range(iters):
            if n < 6:
                break
            dg = np.diag(np.diag(A)) + 1e-12 * np.eye(6)
            try:
                step = -np.linalg.solve(A + lam * dg, g)
            except np.linalg.LinAlgError:
                break
            D2 = compose(D, step)
            A2, g2, c2, n2, *_ = self.gn_system(L, D2, P, frames)
            if c2 < cost:
                D, A, g, cost, n = D2, A2, g2, c2, n2
                lam = max(lam / 3, 1e-7)
                if np.max(np.abs(step[:3])) < 1e-6 and np.max(np.abs(step[3:])) < 1e-6:
                    break
            else:
                lam *= 5
                if lam > 1e6:
                    break
        return D


# ====================================================================== the estimate
def _sig(v6) -> dict:
    v = np.asarray(v6, float)
    return {'position_mm': [float(x) * 1000 for x in v[:3]], 'rotation_deg': [math.degrees(float(x)) for x in v[3:]]}


def _delta_doc(D) -> dict:
    x = log_se3(D)
    return {'position_m': [float(v) for v in x[:3]], 'rotation_deg': [math.degrees(float(v)) for v in x[3:]],
            'norm_mm': float(np.linalg.norm(x[:3]) * 1000), 'angle_deg': math.degrees(float(np.linalg.norm(x[3:]))),
            'frame': 'the sensor body: T_world_sensor · ΔT; rotation_deg a rotation vector (axis × angle)'}


def _direction_words(v, T_so) -> str:
    """The dominant component of a 6-vector direction in the sensor body, in the optical frame's words."""
    v = np.asarray(v, float)
    R = T_so[:3, :3].T                                   # body → optical
    t, w = R @ v[:3], R @ v[3:]
    names = ('sideways (the image\'s x)', 'up and down (the image\'s y)', 'along the view (distance)')
    rnames = ('about the image\'s x axis (tilt)', 'about the image\'s y axis (pan)', 'about the view axis (roll)')
    if np.linalg.norm(t) >= np.linalg.norm(w):
        return 'the camera\'s position ' + names[int(np.argmax(np.abs(t)))]
    return 'the camera\'s rotation ' + rnames[int(np.argmax(np.abs(w)))]


def _overlay(img, cam_full, masks_before, masks_after) -> dict:
    """A thumbnail of the frame (base64 PNG) and the rendered silhouette outlines before and after, in thumbnail
    pixels (OpenCV pixel centres)."""
    import cv2
    h, w = img.shape[:2]
    tw = min(OVERLAY_W, w)
    th = max(1, int(round(h * tw / w)))
    small = cv2.resize(np.ascontiguousarray(img), (tw, th), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.png', cv2.cvtColor(small, cv2.COLOR_RGB2BGR))

    def outlines(S):
        if S is None:
            return []
        sc = tw / S.shape[1]
        cs, _h = cv2.findContours(S.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cs:
            if len(c) < 3:
                continue
            p = (c.reshape(-1, 2).astype(float) + 0.5) * sc - 0.5
            out.append([[round(float(a), 2), round(float(b), 2)] for a, b in np.vstack([p, p[:1]])])
        return out[:200]
    return {'width': tw, 'height': th, 'png_base64': base64.b64encode(buf.tobytes()).decode('ascii') if ok else '',
            'before': outlines(masks_before), 'after': outlines(masks_after)}


def estimate(dataset_or_episode, project, sensor_uid: str, frames=None, settings: dict | None = None, *,
             renderer=None, progress=None, cancel=None) -> dict:
    """The mount of a static camera from the robot's silhouette in its recorded frames (module docstring). frames:
    indices into the camera's stored frames to choose from (None: all). Raises CalibError (a plain sentence)."""
    import time

    st = settings_of(project, settings)
    report = progress or (lambda f, t='': None)
    cancel = cancel or (lambda: False)
    t0 = time.perf_counter()
    sensor = static_sensor(project, sensor_uid)
    ep, _ds = _episode_of(dataset_or_episode, st)
    stream = match_stream(ep, sensor, st.get('camera'))
    notes = []
    if not len(ep.t) or np.asarray(ep.q).ndim != 2 or not len(ep.q):
        raise CalibError(f'Episode {ep.index} has no joint states, so the robot cannot be posed.')
    if (ep.meta or {}).get('time_base') == 'steps':
        notes.append('The dataset has no timestamps: frames are paired with state rows by step.')
    ruid, model, pairs, rn = robot_map(project, ep)
    notes += rn
    ts = np.asarray(stream.timestamps, float)
    if not len(ts):
        raise CalibError(f'Camera {stream.key} of episode {ep.index} has no frames.')
    Qall = q_at(ep, model, pairs, ts)
    cand = None if frames is None else list(frames)
    chosen = choose_frames(Qall, st['max_frames'], cand)
    if not chosen:
        raise CalibError('None of the frames asked for is in the camera\'s stream.')
    report(0.02, f'Decoding {len(chosen)} frames of {stream.key}')
    imgs = {}
    for i, im in stream.frames(chosen):
        imgs[int(i)] = np.asarray(im, np.uint8)
    chosen = [i for i in chosen if i in imgs]
    if not chosen:
        raise CalibError(f'No frame of {stream.key} could be decoded.')
    imgs_l = [imgs[i] for i in chosen]
    Qs = Qall[chosen]
    own = renderer is None
    if own:
        from ..sim.render import Renderer
        try:
            renderer = Renderer(project, device=st['device'], objects=bool(st['objects']))
        except RuntimeError as e:
            raise CalibError(f'The MuJoCo scene could not be built: {e}') from None
    try:
        try:
            scam = renderer.camera(sensor.uid)
        except KeyError as e:
            raise CalibError(str(e).strip('"\'')) from None
        H0, W0 = imgs_l[0].shape[:2]
        if stream.width is None or stream.height is None:
            stream.width, stream.height = W0, H0
        cam_full = camera_model_of(stream, scam, notes)
        if (W0, H0) != (cam_full.width, cam_full.height):
            cam_full = cam_full.scaled(W0, H0)
        prob = _Problem(renderer, sensor, cam_full, imgs_l, Qs, ruid, st, notes)
        if not prob.robot_ids:
            raise CalibError('The robot has no visual geometry in the MuJoCo scene, so it has no silhouette.')
        D = np.eye(4)
        # before: the project's mount
        L0, Lf = 0, len(prob.levels) - 1
        before = prob.render(Lf, [D])[0]
        c_before = prob.coarse_cost(Lf, before)
        frac = float(np.mean([np.isin(b['seg'], prob.robot_ids).mean() for b in before]))
        masks_before = [np.isin(b['seg'], prob.robot_ids) for b in before]
        # (1) the coarse search
        if st['coarse_search']:
            report(0.1, 'Coarse search')
            D = _coarse(prob, L0, D, cancel)
        # (2) Gauss-Newton, coarse to fine
        nl = len(prob.levels)
        for L in range(nl):
            for it in range(st['outer_iters']):
                if cancel():
                    break
                report(0.2 + 0.6 * (L + it / st['outer_iters']) / nl,
                       f'Aligning at {prob.levels[L]["cam"].width} × {prob.levels[L]["cam"].height} px')
                P = prob.points(L, D)
                D_new = prob.gn(L, D, P, st['inner_iters'])
                step = log_se3(np.linalg.inv(D) @ D_new)
                D = D_new
                if np.max(np.abs(step[:3])) < 2e-5 and np.max(np.abs(step[3:])) < 2e-5:
                    break
        # the solution: features, fit, uncertainty
        report(0.85, 'Uncertainty')
        after = prob.render(Lf, [D])[0]
        P = prob.points(Lf, D, after)
        masks_after = [c.S for c in P]
        c_after = prob.coarse_cost(Lf, after)
        A, g, cost, n, per, rows, res, fr, wts = prob.gn_system(Lf, D, P)
        npts = [len(x) for x in P]
        unc = _uncertainty(prob, Lf, D, P, rows, res, fr, wts, prob.last_owners, Qs, st, cancel)
        obs = _observability(unc, frac, npts, Qs, st, prob.T_so)
        fit = {'iou_before': c_before[2], 'iou_after': c_after[2],
               'chamfer_before_px': c_before[1] * st['truncate_px'], 'chamfer_after_px': c_after[1] * st['truncate_px'],
               'rms_px': float(math.sqrt(sum(p[0] for p in per) / max(1, sum(p[1] for p in per)))),
               'residuals': int(n), 'points': int(sum(npts)),
               'inlier_share': float(n / max(1, sum(npts) + sum(int(c.mask_ok.sum()) for c in P))),
               'level': {'width': prob.levels[Lf]['cam'].width, 'height': prob.levels[Lf]['cam'].height},
               'per_frame': [{'frame': int(f), 'points': int(npts[k]), 'inliers': int(per[k][1]),
                              'rms_px': float(math.sqrt(per[k][0] / per[k][1])) if per[k][1] else None,
                              'iou': (iou(masks_after[k] ^ (np.mean(masks_after, axis=0) > 0.5),
                                          prob.levels[Lf]['fg'][k]) if prob.levels[Lf]['fg'] is not None else None)}
                             for k, f in enumerate(chosen)]}
        if prob.levels[Lf]['fg'] is None:
            notes.append('Fewer than 3 frames: no median background, so the foreground IoU is not used.')
        T_new = prob.T_world_opt(D)
        Ti = np.linalg.inv(T_new)
        zs = np.concatenate([c.X @ Ti[2, :3] + Ti[2, 3] for c in P if len(c)] or [np.zeros(0)])
        z_med = float(np.median(zs)) if len(zs) else None
        camf = prob.levels[Lf]['cam']
        f_px = 0.5 * (camf.fx + camf.fy)
        scale = {'robot_depth_m': z_med, 'focal_px': f_px, 'width_px': camf.width,
                 'mm_per_px': (1000 * z_med / f_px) if z_med else None, 'deg_per_px': math.degrees(1 / f_px),
                 'note': 'one pixel of the working image, sideways at the robot\'s median contour depth'}
        mount = prob.T_bs @ D                                 # the sensor body in the world (static: its parent)
        D_opt = np.linalg.inv(prob.T_so) @ D @ prob.T_so
        info = renderer.info
        kb = int(np.argmax([m.sum() for m in masks_after]))
        doc = {
            'sensor_uid': sensor.uid, 'sensor': sensor.name, 'camera_key': stream.key, 'episode': int(ep.index),
            'pose': _pose_doc(T_new), 'pose_before': _pose_doc(prob.T_world_opt(np.eye(4))),
            'mount': {'parent_frame': sensor.mount.parent_frame, 'robot_uid': '', 'pose': _pose_doc(mount)},
            'delta': _delta_doc(D), 'delta_optical': dict(_delta_doc(D_opt), frame='the optical frame (z forward, x '
                                                                                 'right, y down)'),
            'sigma': unc['sigma'], 'frames_used': [int(i) for i in chosen],
            'frame_times_s': [float(ts[i]) for i in chosen],
            'observable': obs['observable'], 'why': obs['why'], 'observability': obs['doc'],
            'fit': fit, 'scale': scale, 'robot': {'uid': ruid, 'name': project.robot(ruid).name, 'share_of_image': frac,
                                  'joint_motion_deg': obs['doc']['joint_motion_deg']},
            'camera': {'model': cam_full.to_dict(), 'summary': cam_full.summary()},
            'overlay': _overlay(imgs_l[kb], cam_full, masks_before[kb], masks_after[kb]) | {'frame': int(chosen[kb])},
            'render': {**info, 'renders': prob.renders},
            'method': {'coarse': 'pattern search over ±steps along and about the sensor body axes, each move '
                                 'rendered (no gradient)' if st['coarse_search'] else 'off',
                       'fine': 'Gauss-Newton / Levenberg-Marquardt on point-to-tangent chamfer residuals: analytic '
                               'pose and edge derivatives, lens projection derivative by central differences; '
                               'contour points re-rendered at each outer iteration',
                       'levels': [{'width': lv['cam'].width, 'height': lv['cam'].height} for lv in prob.levels]},
            'settings': {k: v for k, v in st.items() if k != 'defaults'}, 'defaults': list(st['defaults']),
            'notes': notes, 'seconds': round(time.perf_counter() - t0, 3),
        }
        report(1.0, 'Done')
        return doc
    finally:
        if own:
            renderer.close()


def _coarse(prob: _Problem, L: int, D, cancel):
    """The pattern search on the coarse cost (module docstring)."""
    lv = prob.level_features(L)
    cam = lv['cam']
    base = prob.render(L, [D])[0]
    zs = np.concatenate([b['depth'][np.isin(b['seg'], prob.robot_ids)] for b in base] or [np.zeros(0)])
    zs = zs[zs > 0]
    if not len(zs):
        return D
    z = float(np.median(zs))
    f = 0.5 * (cam.fx + cam.fy)
    step_t, step_r = COARSE_START_PX * z / f, COARSE_START_PX / f
    end_r = COARSE_END_PX / f
    best, *_ = prob.coarse_cost(L, base)
    for _ in range(COARSE_SWEEPS):
        if cancel() or step_r < end_r:
            break
        moves = []
        for a in range(6):
            for sgn in (1, -1):
                xi = np.zeros(6)
                xi[a] = sgn * (step_t if a < 3 else step_r)
                moves.append(compose(D, xi))
        outs = prob.render(L, moves)
        costs = [prob.coarse_cost(L, o)[0] for o in outs]
        k = int(np.argmin(costs))
        if costs[k] < best - 1e-9:
            best, D = costs[k], moves[k]
        else:
            step_t, step_r = step_t / 2, step_r / 2
    return D


def _uncertainty(prob, L, D, P, rows, res, fr, wts, owners, Qs, st, cancel) -> dict:
    """The Gauss-Newton covariance (pose-deduplicated), the cluster-robust covariance and the bootstrap over frames
    (module docstring)."""
    out = {}
    if rows:
        J, r, f = np.concatenate(rows), np.concatenate(res), np.concatenate(fr)
        w, ow = np.concatenate(wts), np.concatenate(owners)
    else:
        J, r, f = np.zeros((0, 6)), np.zeros(0), np.zeros(0, int)
        w, ow = np.zeros(0), np.zeros(0, int)
    n = len(r)
    s2 = float(r @ r / (n - 6)) if n > 6 else math.nan
    lab = pose_clusters(Qs, math.radians(st['min_motion_deg']))
    counts = np.bincount(lab) if lab else np.zeros(0)
    # (a) the Hessian of the cost: Gauss-Newton information, frames of one pose counted once
    I = np.zeros((6, 6))
    for k in range(len(P)):
        m = f == k
        if m.any():
            I += (J[m].T @ J[m]) / counts[lab[k]]
    I = I / s2 if math.isfinite(s2) and s2 > 0 else I * math.nan
    C = _inv(I)
    sh = _sd(C)
    # (b) cluster-robust (sandwich) covariance: the scores summed per (distinct pose, link) cluster, so residuals
    # that share an error along one link's contour, or repeat over frames of one pose, count as one
    Cs, nclu = np.full((6, 6), math.nan), 0
    if n > 6:
        Bread = (J * w[:, None]).T @ J
        key = np.array([lab[int(k)] for k in f], np.int64) * 100003 + ow
        uniq, inv_ = np.unique(key, return_inverse=True)
        nclu = len(uniq)
        sc = np.zeros((nclu, 6))
        np.add.at(sc, inv_, J * (w * r)[:, None])
        gs = np.zeros((nclu, 6))                         # the score of a common offset along the normals
        np.add.at(gs, inv_, J * w[:, None])
        if nclu > 6:
            Bi = _inv(Bread)
            Cs = Bi @ (sc.T @ sc) @ Bi * (nclu / (nclu - 1))
            # the systematic floor: each cluster's contour offset along its normals by N(0, edge_floor_px²)
            Cs = Cs + Bi @ (gs.T @ gs) @ Bi * float(st['edge_floor_px']) ** 2
    sc_ = _sd(Cs)
    out['info'] = I
    out['cov'] = C
    out['sigma_px'] = math.sqrt(s2) if math.isfinite(s2) else None
    out['poses'] = int(len(counts))
    # (c) the bootstrap over frames
    B = int(st['bootstrap'])
    used = [k for k in range(len(P)) if len(P[k])]
    sb = np.full(6, math.nan)
    xs = []
    if B > 0 and len(used) >= 2:
        rng = np.random.default_rng(int(st['seed']) + 7)
        for _b in range(B):
            if cancel():
                break
            pick = list(rng.choice(used, len(used), replace=True))
            Db = prob.gn(L, D, P, max(4, st['inner_iters'] // 2), frames=pick)
            xs.append(log_se3(np.linalg.inv(D) @ Db))
        if len(xs) >= 2:
            sb = np.std(np.asarray(xs), axis=0, ddof=1)
    comb = np.nanmax(np.vstack([sh, sc_, sb]), axis=0) if np.isfinite(np.vstack([sh, sc_, sb])).any() else sh
    Cuse = Cs if np.isfinite(Cs).all() else C
    out['sigma'] = {**_sig(comb),
                    'rule': 'per axis the largest of the Gauss-Newton, cluster-robust and bootstrap σ (sensor body '
                            'axes)',
                    'hessian': {**_sig(sh), 'sigma_px': out['sigma_px'], 'distinct_poses': out['poses'],
                                'note': 'Gauss-Newton σ̂² (JᵀJ)⁻¹, frames of one pose counted once: a lower bound, '
                                        'residuals along a contour share errors'},
                    'cluster_robust': {**_sig(sc_), 'clusters': int(nclu),
                                       'note': 'sandwich covariance with the scores summed per (distinct pose, link)'}
                    if np.isfinite(sc_).all() else None,
                    'bootstrap': {**_sig(sb), 'resamples': len(xs),
                                  'note': 'resampled frames re-solved on the final contour points (no re-render)'}
                    if len(xs) >= 2 else None,
                    'covariance': [[float(x) for x in row] for row in Cuse] if np.isfinite(Cuse).all() else None,
                    'covariance_kind': 'cluster-robust' if np.isfinite(Cs).all() else 'Gauss-Newton',
                    'covariance_units': 'm and rad, (x, y, z, rx, ry, rz) of ΔT in the sensor body'}
    return out


def _inv(A) -> np.ndarray:
    if not np.isfinite(A).all():
        return np.full((6, 6), math.nan)
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)


def _sd(C) -> np.ndarray:
    return np.sqrt(np.clip(np.diag(C), 0, None)) if np.isfinite(C).all() else np.full(6, math.nan)


def _observability(unc, frac, npts, Qs, st, T_so) -> dict:
    """The observability test (module docstring)."""
    I = unc['info']
    S = np.diag([st['need_mm'] / 1000.0] * 3 + [math.radians(st['need_deg'])] * 3)
    lam, vec, weakest = None, None, ''
    if np.isfinite(I).all():
        N = S @ I @ S
        w, V = np.linalg.eigh((N + N.T) / 2)
        lam, vec = float(w[0]), V[:, 0]
        weakest = _direction_words(S @ vec, T_so)
    motion = float(np.degrees(np.max(np.ptp(Qs, axis=0)))) if len(Qs) > 1 else 0.0
    poses = unc['poses']
    small = frac < st['min_robot_frac'] or (np.median(npts) if npts else 0) < st['min_points']
    static = poses < st['min_poses']
    weak = lam is None or not math.isfinite(lam) or lam < 1.0
    why = []
    if small:
        why.append(f'the robot is too small in the image: it covers {100 * frac:.2f} % of it with a median of '
                   f'{int(np.median(npts)) if npts else 0} contour points per frame (at least '
                   f'{100 * st["min_robot_frac"]:g} % and {st["min_points"]} points, defaults)')
    if static:
        why.append(f'the robot is too static: the frames hold {poses} distinct joint state{"" if poses == 1 else "s"} '
                   f'(at least {st["min_poses"]}, a default; two frames are distinct when a joint differs by '
                   f'{st["min_motion_deg"]:g}°), its largest joint range over them {motion:.1f}°')
    if weak:
        if lam is None or not math.isfinite(lam):
            why.append('the alignment has too few edge matches to estimate an uncertainty')
        else:
            why.append(f'the weakest direction, mostly {weakest}, has σ {1 / math.sqrt(max(lam, 1e-300)):.2f} × the '
                       f'requirement ({st["need_mm"]:g} mm and {st["need_deg"]:g}°, defaults)')
    ok = not (small or static or weak)
    text = ('observable: every direction within the requirement '
            f'({st["need_mm"]:g} mm, {st["need_deg"]:g}°; defaults)' if ok else
            'not observable: ' + '; '.join(why))
    doc = {'min_eigenvalue': lam, 'threshold': 1.0, 'weakest_direction': weakest,
           'weakest_vector': None if vec is None else [float(x) for x in vec],
           'scaling': f'information scaled by the requirement: {st["need_mm"]:g} mm, {st["need_deg"]:g}°',
           'robot_share': frac, 'median_points': int(np.median(npts)) if npts else 0, 'distinct_poses': poses,
           'joint_motion_deg': motion, 'too_small': bool(small), 'too_static': bool(static), 'weak': bool(weak)}
    return {'observable': ok, 'why': text, 'doc': doc}


# ====================================================================== drift over episodes
def drift(dataset, project, sensor_uid: str, episodes=None, settings: dict | None = None, *, renderer=None,
          progress=None, cancel=None) -> dict:
    """One estimate per episode (episodes: indices, None: the first max_episodes) and the changes between them:
    {'episodes': [estimate or {'episode', 'error'}], 'steps': [{'from', 'to', 'position_mm', 'rotation_deg', 'z'}],
    'largest_step', 'trend'}."""
    from ..datasets.episode import open_dataset
    st = settings_of(project, settings)
    report = progress or (lambda f, t='': None)
    cancel = cancel or (lambda: False)
    if isinstance(dataset, str):
        dataset = open_dataset(dataset)
    refs = dataset.episodes()
    if not refs:
        raise CalibError('The dataset has no episodes.')
    idx = [r.index for r in refs]
    if episodes is None:
        episodes = st['episodes']
    if episodes is None:
        episodes = idx[:st['max_episodes']]
    episodes = [int(e) for e in episodes if int(e) in idx]
    if not episodes:
        raise CalibError('None of the episodes asked for is in the dataset.')
    own = renderer is None
    if own:
        from ..sim.render import Renderer
        try:
            renderer = Renderer(project, device=st['device'], objects=bool(st['objects']))
        except RuntimeError as e:
            raise CalibError(f'The MuJoCo scene could not be built: {e}') from None
    static_sensor(project, sensor_uid)
    out = []
    try:
        for k, e in enumerate(episodes):
            if cancel():
                break
            sub = (lambda f, t='', k=k, e=e: report((k + f) / len(episodes), f'Episode {e}: {t}'))
            try:
                ep = dataset.load(e)
                out.append(estimate(ep, project, sensor_uid, None, dict(settings or {}), renderer=renderer,
                                    progress=sub, cancel=cancel))
            except CalibError as err:                    # one episode that cannot be used: said, the others go on
                out.append({'episode': e, 'error': str(err)})
    finally:
        if own:
            renderer.close()
    good = [d for d in out if 'delta' in d]
    steps = []
    for a, b in zip(good, good[1:]):
        da, db = np.r_[a['delta']['position_m'], np.radians(a['delta']['rotation_deg'])], \
            np.r_[b['delta']['position_m'], np.radians(b['delta']['rotation_deg'])]
        sa = np.r_[np.asarray(a['sigma']['position_mm']) / 1000, np.radians(a['sigma']['rotation_deg'])]
        sb = np.r_[np.asarray(b['sigma']['position_mm']) / 1000, np.radians(b['sigma']['rotation_deg'])]
        s = np.sqrt(np.nan_to_num(sa) ** 2 + np.nan_to_num(sb) ** 2)
        with np.errstate(divide='ignore', invalid='ignore'):
            z = np.where(s > 0, np.abs(db - da) / s, np.inf)
        d = db - da
        steps.append({'from': a['episode'], 'to': b['episode'], 'position_mm': [float(x) * 1000 for x in d[:3]],
                      'rotation_deg': [math.degrees(float(x)) for x in d[3:]],
                      'norm_mm': float(np.linalg.norm(d[:3]) * 1000),
                      'angle_deg': math.degrees(float(np.linalg.norm(d[3:]))), 'z': float(np.max(z)),
                      'step': bool(np.max(z) > st['step_z'] and (np.linalg.norm(d[:3]) * 1000 > st['tolerance_mm'] or
                                                                 math.degrees(np.linalg.norm(d[3:])) >
                                                                 st['tolerance_deg']))})
    largest = max(steps, key=lambda s: s['z'], default=None)
    trend = None
    if len(good) >= 3:
        x = np.array([d['episode'] for d in good], float)
        Y = np.array([np.r_[np.asarray(d['delta']['position_m']) * 1000, d['delta']['rotation_deg']] for d in good])
        A = np.c_[x, np.ones_like(x)]
        coef = np.linalg.lstsq(A, Y, rcond=None)[0][0]
        trend = {'position_mm_per_episode': [float(v) for v in coef[:3]],
                 'rotation_deg_per_episode': [float(v) for v in coef[3:]]}
    return {'episodes': out, 'steps': steps, 'largest_step': largest, 'trend': trend,
            'step_rule': f'a change over {st["step_z"]:g} σ (combined) and beyond {st["tolerance_mm"]:g} mm or '
                         f'{st["tolerance_deg"]:g}° (defaults)'}


def status_of(doc: dict, st: dict) -> tuple[str, str, str]:
    """(severity, word, advice) of one estimate for the page's headline."""
    if not doc.get('observable'):
        return 'warning', 'not observable', doc.get('why', '')
    d, s = doc['delta'], doc['sigma']
    smm = float(np.linalg.norm(np.nan_to_num(s['position_mm'])))
    sdeg = float(np.linalg.norm(np.nan_to_num(s['rotation_deg'])))
    within = d['norm_mm'] <= st['tolerance_mm'] and d['angle_deg'] <= st['tolerance_deg']
    signif = d['norm_mm'] > 3 * smm or d['angle_deg'] > 3 * sdeg
    if within:
        return 'good', 'mount holds', (f'The camera is within {st["tolerance_mm"]:g} mm and {st["tolerance_deg"]:g}° '
                                       'of its mount (defaults).')
    if signif:
        return 'serious', 'mount moved', (f'The camera is {d["norm_mm"]:.1f} mm and {d["angle_deg"]:.2f}° from its '
                                          'mount: Apply writes the recovered mount.')
    return 'warning', 'uncertain', ('The offset is beyond the tolerance but within 3 σ: use more frames or episodes '
                                    'where the robot moves more.')
