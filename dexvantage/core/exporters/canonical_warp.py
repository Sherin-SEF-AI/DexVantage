"""Cross-embodiment canonical view warp (spec §7.16): warp robot B's wrist-camera image to robot A's canonical virtual
viewpoint, exactly for the fingertip contact plane, and say how far off the plane it is wrong.

When datasets are pooled across embodiments (an ALOHA, a Franka, an SO-101), each wrist camera sits somewhere else
relative to its gripper. Aligning the two grippers' contact frames (below) gives the pose of camera B relative to
camera A, and every point of the contact plane then maps from B's image to A's image by one plane-induced homography.
Off the plane the warp is wrong by the plane-plus-parallax term; this module reports that residual parallax over the
task volume's depth range, so the user sees where the warp is exact and where it is not.

    view_of_sensor(project, sensor_uid, q=None, robot_uid=None, closing_sign=1) -> View
    view_from_dict(doc) -> View                         a virtual canonical view (canonical_view.json)
    build(project, options) -> Warp                     source and canonical views, the plane, H
    lut_at(warp, uv_A) -> (uv_B, ok)                    the full remap: canonical pixel -> source pixel, lenses included
    remap_maps(warp) -> (map_x, map_y, valid)           OpenCV remap maps (float32, the canonical image's size)
    parallax_at(warp, uv_A, h) -> (N, 2) px             residual parallax at canonical pixels for a surface at h
    parallax_map(warp, h_lo, h_hi, stride) -> dict      the map, its max and P95 at each depth extreme
    analyse(project, options, ctx) -> dict              the study's result (core/studies/canonical_view.py)
    export_warp(project, out_dir, options, ctx) -> {'files', 'notes'}   the exporter 'canonical_warp'

The contact frame (each robot's own). Origin: the tool centre point between the fingertips; z: the approach axis;
x: the closing axis (times closing_sign, a user setting, because which pad comes first is arbitrary: −1 turns the
frame half a turn about the approach); y = z × x. All three are measured from the robot's contact pads at its home
pose by core/grasps.py gripper_geometry (its docstring has the rule) and posed at q with the gripper frame. The two
robots' contact frames are taken to be the same frame: that identification is the meaning of "canonical".

The contact plane: z = plane_offset_m in the contact frame (default 0, the plane through the fingertip centre, normal
along the approach axis; a labelled default).

Conventions. Camera optical frames are OpenCV's (x right, y down, z forward); pixel centres are at integers (u = 0 is
the centre of the first column). (R, t) is canonical-from-source: X_A = R X_B + t. In source camera coordinates the
plane is n_Bᵀ X_B + d_B = 0, with n_B the unit normal pointing from the plane towards camera B and d_B > 0 B's
distance to the plane. Then for every point of the plane

    x_A ~ H x_B,   H = K_A (R − t n_Bᵀ / d_B) K_B⁻¹          [Hartley and Zisserman 2004, ch. 13]

with x_A, x_B homogeneous pixels of the two cameras' pinhole (undistorted) images and K their camera matrices
[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]. H is scaled so H[2, 2] = 1. H maps source to canonical; cv2.warpPerspective(
img_B, H, (W_A, H_A)) applies it (dst(x) = src(H⁻¹ x)). With lenses that are not pinhole, H alone is not the warp:
the full remap LUT runs, for every canonical pixel, unproject through A's lens -> intersect the plane -> into B's
frame -> project through B's lens, so map_x, map_y give the source pixel each canonical pixel reads
(cv2.remap(img_B, map_x, map_y, cv2.INTER_LINEAR)); pixels whose ray misses the plane, meets it behind either camera,
or lands outside B's image hold −1 (OpenCV's remap then gives the border value).

Residual parallax. The warp shows at canonical pixel p what B sees along its ray through the plane point X(p). When
the surface there is not on the plane but at signed distance h from it along that ray (h > 0 towards camera B,
h < 0 beyond the plane), the true canonical image shows it at π_A(X_h), so the warp is wrong by
Δ(p, h) = π_A(X_h) − p (pixels of the canonical image, A's real lens). For pinhole cameras this is the
plane-plus-parallax term Δ = h / (d_B Z_A) · ((K_A t)_xy − p (K_A t)_z) [Irani and Anandan 1996], derived in
docs/models/canonical_warp.md, with Z_A the point's depth in A: it vanishes on the plane (h = 0) and when the cameras
share a centre (t = 0). Along one ray the image of X_h runs monotonically along the epipolar line, so over a depth
range [h_lo, h_hi] the largest |Δ| is at one of the two ends: the map is max(|Δ(p, h_lo)|, |Δ(p, h_hi)|).

The depth range (defaults, labelled): at a grasp the contact frame's origin can be anywhere in the task volume, so a
point of the volume can lie up to D = Σᵢ |aᵢ| sᵢ beyond the plane, a the approach axis in the world (at the source
robot's q) and s the volume's edge lengths (with several volumes the largest D). Towards the camera, an object held
between the pads comes no nearer than the pads' far end (the gripper fills the rest), so the near end is the largest
distance of a contact-pad point from the plane on the camera's side (core/grasps.py contact_points), at most D and at
most d_B − near_m (a point must stay in front of camera B): the range is [−D, h_hi]. The user can set the range
instead (depth_range_m).

Ported in part from Vantage core/viz_bridge.py's camera handling (OpenCV optical axes and K); the homography, the
LUT and the parallax are new in DexVantage.

Known limits: the contact frames of two different grippers are identified by their measured TCP, approach and
closing axes; a gripper whose pads are not tagged gets its gripper frame's axes (said in the result), and the
closing direction's sign is a user choice. The warp is exact only for points on the contact plane; elsewhere the
parallax report bounds the error over the depth range but nothing is done about it (no depth-based reprojection, no
occlusion: what B sees behind the fingers is warped as if it were on the plane, and parts of A's view that B cannot
see stay empty). A camera that is not fixed in the gripper frame (a static or forearm camera) is related at the q
given (default home). The image content is resampled once (bilinear by default in OpenCV), with no anti-aliasing
when the warp minifies. Rolling shutter, exposure and colour differences between the cameras are not modelled.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field

import numpy as np

from ..jobs import StudyError
from . import exporter

SCHEMA = 'dexvantage.canonical_warp/1'
VIEW_SCHEMA = 'dexvantage.canonical_view/1'
SID = 'canonical_view'
NEAR_M = 0.005                      # default: a point must be this far in front of a camera (as camrender.NEAR_M)
MAP_CELLS = 64                      # the study's parallax map: at most this many cells along the canonical width
INVALID = -1.0                      # map value of a canonical pixel with no source pixel
GRID_INVALID = -2.0                 # grid_sample value of such a pixel (outside [-1, 1]: zeros with padding 'zeros')
DEFAULTS = {
    'plane_offset_m': 0.0,          # the contact plane through the TCP (default)
    'closing_sign_source': 1,
    'closing_sign_canonical': 1,
    'near_m': NEAR_M,
    'thresholds_px': {'good': 1.0, 'warning': 4.0, 'serious': 16.0},     # DexVantage defaults, not standards
}
WORDS = {'good': 'aligned', 'warning': 'mild parallax', 'serious': 'parallax', 'critical': 'large parallax'}
FORMATS = ('npy', 'yml', 'grid', 'json')
CONVENTION = ('Optical frames: OpenCV (x right, y down, z forward); pixel centres at integers. (R, t): canonical (A) '
              'from source (B), X_A = R X_B + t. Plane in B: n_B . X_B + d_B = 0, n_B the unit normal from the plane '
              'towards camera B, d_B > 0. H = K_A (R - t n_B^T / d_B) K_B^-1 maps source pixels to canonical pixels '
              'of the undistorted (pinhole) images, scaled so H[2][2] = 1 (cv2.warpPerspective(img_B, H, (W_A, '
              'H_A))). The remap maps include both lenses: cv2.remap(img_B, map_x, map_y, cv2.INTER_LINEAR); -1 '
              'marks a canonical pixel with no source pixel.')


class WarpError(StudyError):
    """A warp that cannot be made; the message says why in plain words."""


# ====================================================================== views
@dataclass
class View:
    """One camera seen from its gripper: its lens and its optical frame in its robot's contact frame."""
    name: str
    camera: object                   # optics.CameraModel at the stream size
    T_contact_cam: np.ndarray        # 4 × 4: camera optical frame -> contact frame (X_contact = T @ X_cam)
    robot_uid: str = ''
    robot_name: str = ''
    sensor_uid: str = ''
    imager: str = ''
    virtual: bool = False
    rigid: bool = True               # fixed in the gripper frame (independent of q)
    contact_rule: str = ''
    T_world_contact: np.ndarray | None = None     # the robot's contact frame in the world at q (None for virtual)
    notes: list = field(default_factory=list)

    @property
    def K(self) -> np.ndarray:
        c = self.camera
        return np.array([[c.fx, 0.0, c.cx], [0.0, c.fy, c.cy], [0.0, 0.0, 1.0]])

    def to_dict(self) -> dict:
        from ..frames import Pose
        return {'name': self.name, 'robot_uid': self.robot_uid, 'robot': self.robot_name,
                'sensor_uid': self.sensor_uid, 'imager': self.imager, 'virtual': self.virtual, 'rigid': self.rigid,
                'camera': self.camera.to_dict(), 'K': _lists(self.K),
                'pose_in_contact': Pose.from_matrix(self.T_contact_cam).to_dict(),
                'T_contact_cam': _lists(self.T_contact_cam), 'contact_rule': self.contact_rule,
                'T_world_contact': None if self.T_world_contact is None else _lists(self.T_world_contact)}


def _lists(a):
    a = np.asarray(a, float)
    return [_lists(x) for x in a] if a.ndim > 1 else [_num(x) for x in a]


def _num(x):
    x = float(x)
    return x if math.isfinite(x) else None


def _q_of(model, q, uid):
    if q is None:
        return model.home
    if isinstance(q, dict):
        v = q.get(uid)
        return model.home if v is None else model.clip(np.asarray(v, float))
    return model.clip(np.asarray(q, float))


def contact_frame(project, robot_uid: str, q=None, closing_sign: int = 1) -> tuple[np.ndarray, dict]:
    """(4 × 4 contact frame in the world at q, the gripper geometry used) of a robot: origin at the TCP, z the
    approach axis, x the closing axis × closing_sign, y = z × x (module docstring)."""
    from .. import grasps as GR
    from ..robot.model import RobotModelError, get_model
    try:
        g = GR.gripper_geometry(project, robot_uid)
        m = get_model(project, robot_uid)
    except (GR.GraspError, RobotModelError) as e:
        raise WarpError(str(e)) from None
    z = np.asarray(g['approach'], float)
    x = np.asarray(g['closing'], float) * (1.0 if int(closing_sign) >= 0 else -1.0)
    y = np.cross(z, x)
    T_gc = np.eye(4)
    T_gc[:3, :3] = np.stack([x, y, z], axis=1)
    T_gc[:3, 3] = np.asarray(g['tcp'], float)
    T_wg = m.link_pose(_q_of(m, q, robot_uid), g['frame'])
    return T_wg @ T_gc, g


def _camera_robot(project, sensor, robot_uid):
    if robot_uid:
        if project.robot(robot_uid) is None:
            raise WarpError('The robot named for the camera is not in the project.')
        return robot_uid
    if sensor.mount.on_link and project.robot(sensor.mount.robot_uid) is not None:
        return sensor.mount.robot_uid
    r = project.primary_robot
    if r is None:
        raise WarpError('The canonical view warp needs a robot whose gripper defines the contact frame.')
    return r.uid


def _other_q(model):
    """A second joint vector, away from home, to test whether a camera is fixed in the gripper frame."""
    q = np.array(model.home, float) + 0.37
    return model.clip(q)


def view_of_sensor(project, sensor_uid: str, q=None, robot_uid: str | None = None, closing_sign: int = 1) -> View:
    """The View of a project camera: its picked imager (the RGB one, as the Camera view) at its stream size, its
    optical frame in the contact frame of the robot it is mounted on (or robot_uid; a static camera: the primary
    robot) at q ({robot uid: q}; None = home)."""
    from .. import camrender as CR
    from .. import plucker as PK
    from ..robot.model import RobotModelError, get_model
    s = project.variant.sensor(sensor_uid)
    if s is None:
        raise WarpError('That camera is not in the active rig.')
    im = PK.pick_imager(s, PK.sensor_stream(s))
    if im is None:
        raise WarpError(f'{s.name} has no imager to warp.')
    ru = _camera_robot(project, s, robot_uid)
    rob = project.robot(ru)
    T_wc, g = contact_frame(project, ru, q, closing_sign)
    try:
        T_wo = CR.optical_pose(project, s, im, q).matrix()
    except (KeyError, RobotModelError) as e:
        raise WarpError(f'{s.name} cannot be posed: {e}') from None
    T = np.linalg.inv(T_wc) @ T_wo
    notes = []
    rigid = False
    try:
        m = get_model(project, ru)
        q2 = dict(q or {})
        q2[ru] = _other_q(m)
        T2 = np.linalg.inv(contact_frame(project, ru, q2, closing_sign)[0]) @ CR.optical_pose(project, s, im,
                                                                                               q2).matrix()
        rigid = bool(np.allclose(T, T2, atol=1e-9))
    except (KeyError, RobotModelError, WarpError):
        pass
    if not rigid:
        notes.append(f'{s.name} is not fixed in {rob.name}\'s gripper frame: it is related to the contact frame at '
                     + ('the joint state given.' if q and ru in (q or {}) else 'the home pose.'))
    if g.get('assumed'):
        notes.append(f'{rob.name}: {g["rule"]}.')
    return View(s.name, im.camera, T, ru, rob.name, s.uid, im.name, False, rigid, g.get('rule', ''), T_wc, notes)


def view_from_dict(doc: dict) -> View:
    """A virtual view from a canonical_view.json document (or the 'canonical' entry of an export's JSON):
    {'name', 'camera': CameraModel.to_dict(), 'pose_in_contact': {'position', 'quaternion'}}."""
    from ..frames import Pose
    from ..optics import CameraModel
    if not isinstance(doc, dict):
        raise WarpError('A canonical view must be a JSON object with "camera" and "pose_in_contact".')
    if doc.get('schema') not in (None, VIEW_SCHEMA):
        raise WarpError(f'This is a {doc.get("schema")} file, not a canonical view ({VIEW_SCHEMA}).')
    try:
        cam = CameraModel.from_dict(doc.get('camera'))
        pose = Pose.from_dict(doc.get('pose_in_contact'))
    except (ValueError, TypeError) as e:
        raise WarpError(f'The canonical view cannot be read: {e}.') from None
    if not isinstance(doc.get('pose_in_contact'), dict):
        raise WarpError('The canonical view has no "pose_in_contact" (the camera in the contact frame).')
    return View(str(doc.get('name') or 'Canonical view'), cam, pose.matrix(), str(doc.get('robot_uid') or ''),
                str(doc.get('robot') or ''), '', str(doc.get('imager') or ''), True, True,
                str(doc.get('contact_rule') or 'as recorded in the canonical view file'), None, [])


def view_document(v: View) -> dict:
    """What canonical_view.json holds: the camera and its pose in the contact frame (loadable by view_from_dict)."""
    from ..frames import Pose
    return {'schema': VIEW_SCHEMA, 'name': v.name, 'robot': v.robot_name, 'imager': v.imager,
            'camera': v.camera.to_dict(), 'pose_in_contact': Pose.from_matrix(v.T_contact_cam).to_dict(),
            'contact_rule': v.contact_rule,
            'contact_frame': 'origin at the TCP between the fingertips, z the approach axis, x the closing axis, '
                             'y = z × x (core/exporters/canonical_warp.py)'}


# ====================================================================== the warp
@dataclass
class Warp:
    source: View                     # B
    canonical: View                  # A
    R: np.ndarray                    # canonical from source
    t: np.ndarray
    n_B: np.ndarray
    d_B: float
    n_A: np.ndarray
    d_A: float
    H: np.ndarray
    plane_offset_m: float
    notes: list = field(default_factory=list)

    @property
    def baseline_m(self) -> float:
        return float(np.linalg.norm(self.t))


def homography(K_A, R, t, n_B, d_B, K_B) -> np.ndarray:
    """H = K_A (R − t n_Bᵀ / d_B) K_B⁻¹, scaled so H[2, 2] = 1 when it is not zero (module docstring)."""
    H = np.asarray(K_A, float) @ (np.asarray(R, float) - np.outer(t, n_B) / float(d_B)) @ np.linalg.inv(K_B)
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def make_warp(source: View, canonical: View, plane_offset_m: float = 0.0) -> Warp:
    """The warp between two views: their relative pose through the shared contact frame and the contact plane."""
    T_AB = np.linalg.inv(canonical.T_contact_cam) @ source.T_contact_cam
    R, t = T_AB[:3, :3], T_AB[:3, 3]
    Rc, tc = source.T_contact_cam[:3, :3], source.T_contact_cam[:3, 3]
    n_B = Rc.T @ np.array([0.0, 0.0, 1.0])
    d_B = float(tc[2] - float(plane_offset_m))
    if not math.isfinite(d_B) or abs(d_B) < 1e-6:
        raise WarpError(f'{source.name} lies in the contact plane, so the plane has no image in it: move the plane '
                        '(plane offset) or the camera.')
    if d_B < 0:
        n_B, d_B = -n_B, -d_B
    n_A = R @ n_B
    d_A = float(d_B - n_A @ t)
    notes = []
    if abs(d_A) < 1e-6:
        raise WarpError(f'{canonical.name} lies in the contact plane, so the plane has no image in it.')
    if d_A < 0:
        notes.append(f'{canonical.name} looks at the contact plane from the side away from {source.name}: the warp '
                     'shows the plane from behind (it is still exact on the plane).')
    H = homography(canonical.K, R, t, n_B, d_B, source.K)
    return Warp(source, canonical, R, t, n_B, d_B, n_A, d_A, H, float(plane_offset_m), notes)


def _rows(a, k):
    a = np.asarray(a, float)
    return a.reshape(-1, k), a.shape[:-1]


def plane_points_A(warp: Warp, uv_A):
    """(X_A (N, 3) plane points on the canonical pixels' rays, in A's optical frame, ok (N,)): unprojected through A's
    lens and intersected with the plane n_A · X + d_A = 0 in front of A (NaN where not)."""
    uv, shape = _rows(uv_A, 2)
    r, ok = warp.canonical.camera.unproject(uv, with_ok=True)
    den = r @ warp.n_A
    with np.errstate(divide='ignore', invalid='ignore'):
        s = -warp.d_A / den
    ok = ok & np.isfinite(s) & (s > 0)
    X = np.where(ok[:, None], r * np.where(ok, s, 0.0)[:, None], np.nan)
    return X.reshape(shape + (3,)), ok.reshape(shape)


def lut_at(warp: Warp, uv_A):
    """The full remap at canonical pixels uv_A (N, 2): (uv_B (N, 2) the source pixel each reads, ok (N,)). ok is false
    (uv NaN) where the ray misses the plane or meets it behind a camera, and false (uv finite) where the source pixel
    lies outside B's image."""
    X_A, ok = plane_points_A(warp, uv_A)
    Xa, shape = _rows(X_A, 3)
    okf = ok.reshape(-1)
    X_B = (Xa - warp.t) @ warp.R                     # Rᵀ (X_A − t), row-wise
    okf = okf & (X_B[:, 2] > 0 if warp.source.camera.model in ('pinhole', 'brown') else okf)
    uv, inside = warp.source.camera.project(np.where(okf[:, None], X_B, np.nan))
    uv = np.where(okf[:, None], uv, np.nan)
    return uv.reshape(shape + (2,)), (okf & inside).reshape(shape)


def pixel_grid(camera, stride: int = 1):
    us = np.arange(0, camera.width, stride, dtype=float)
    vs = np.arange(0, camera.height, stride, dtype=float)
    U, V = np.meshgrid(us, vs)
    return np.stack([U, V], axis=-1)


def remap_maps(warp: Warp):
    """(map_x, map_y float32 (H_A, W_A), valid bool (H_A, W_A)): for cv2.remap(img_B, map_x, map_y, …). Invalid
    pixels hold INVALID (−1)."""
    uv, ok = lut_at(warp, pixel_grid(warp.canonical.camera))
    mx = np.where(ok, uv[..., 0], INVALID).astype(np.float32)
    my = np.where(ok, uv[..., 1], INVALID).astype(np.float32)
    return mx, my, ok


def grid_sample_grid(warp: Warp, map_x, map_y, valid) -> np.ndarray:
    """(H_A, W_A, 2) float32 for torch.nn.functional.grid_sample(img_B, grid, align_corners=False): x_n = (2u + 1)/W_B
    − 1, y_n = (2v + 1)/H_B − 1 (OpenCV integer pixel centres; ±1 are the outer edges of the image); invalid pixels
    GRID_INVALID (−2, outside: zero with padding_mode='zeros')."""
    W, H = warp.source.camera.width, warp.source.camera.height
    gx = (2.0 * map_x.astype(np.float64) + 1.0) / W - 1.0
    gy = (2.0 * map_y.astype(np.float64) + 1.0) / H - 1.0
    g = np.stack([gx, gy], axis=-1)
    g[~valid] = GRID_INVALID
    return g.astype(np.float32)


def parallax_at(warp: Warp, uv_A, h: float, near_m: float = NEAR_M):
    """(Δ (N, 2) px, ok (N,)): the residual parallax at canonical pixels uv_A for a surface at signed distance h (m)
    from the plane along B's ray (h > 0 towards B): π_A(X_h) − uv_A, A's real lens (module docstring). ok is false
    where the pixel has no plane point, X_h is not near_m in front of both cameras, or A cannot project it."""
    X_A, ok = plane_points_A(warp, uv_A)
    Xa, shape = _rows(X_A, 3)
    uv, _ = _rows(uv_A, 2)
    okf = ok.reshape(-1)
    X_B = (Xa - warp.t) @ warp.R
    nrm = np.linalg.norm(X_B, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = X_B / nrm[:, None]
        den = r @ warp.n_B
        s = (float(h) - warp.d_B) / den
    okf = okf & np.isfinite(s) & (s > near_m)
    Xh_B = r * np.where(okf, s, 0.0)[:, None]
    Xh_A = Xh_B @ warp.R.T + warp.t
    okf = okf & (np.linalg.norm(Xh_A, axis=1) > near_m)
    if warp.canonical.camera.model in ('pinhole', 'brown'):
        okf = okf & (Xh_A[:, 2] > near_m)
    p, _inside = warp.canonical.camera.project(np.where(okf[:, None], Xh_A, np.nan))
    okf = okf & np.isfinite(p).all(axis=1)
    D = np.where(okf[:, None], p - uv, np.nan)
    return D.reshape(shape + (2,)), okf.reshape(shape)


def _stats(mag, ok) -> dict:
    v = mag[ok]
    if not v.size:
        return {'max_px': None, 'p95_px': None, 'median_px': None, 'pixels': 0}
    return {'max_px': float(v.max()), 'p95_px': float(np.percentile(v, 95)), 'median_px': float(np.median(v)),
            'pixels': int(v.size)}


def parallax_map(warp: Warp, h_lo: float, h_hi: float, stride: int | None = None, near_m: float = NEAR_M) -> dict:
    """The residual parallax over canonical pixels covered by the warp (lut_at ok): at the cell centres of a grid of
    stride px (default: at most MAP_CELLS cells across), |Δ| at h_lo and h_hi and their maximum; the max and P95 of
    each over the covered pixels. Arrays: 'u' (w,), 'v' (h,), 'at_lo', 'at_hi', 'map' (h, w) px (NaN where not
    covered or not defined)."""
    cam = warp.canonical.camera
    if stride is None:
        stride = max(1, int(math.ceil(cam.width / MAP_CELLS)))
    us = (np.arange(0, cam.width, stride) + 0.5 * stride - 0.5).clip(0, cam.width - 1)
    vs = (np.arange(0, cam.height, stride) + 0.5 * stride - 0.5).clip(0, cam.height - 1)
    U, V = np.meshgrid(us, vs)
    uv = np.stack([U, V], axis=-1)
    _, cover = lut_at(warp, uv)
    out = {'u': us, 'v': vs, 'stride': int(stride), 'covered': cover}
    mags = []
    for key, h in (('at_lo', h_lo), ('at_hi', h_hi)):
        D, ok = parallax_at(warp, uv, h, near_m)
        ok = ok & cover
        mag = np.where(ok, np.linalg.norm(np.nan_to_num(D), axis=-1), np.nan)
        out[key] = mag
        out[key + '_stats'] = _stats(mag, ok)
        mags.append(mag)
    both = np.fmax(mags[0], mags[1])
    out['map'] = both
    out['stats'] = _stats(both, np.isfinite(both))
    return out


# ====================================================================== the depth range and the options
def depth_range(project, warp: Warp, settings: dict) -> dict:
    """{'h_lo_m', 'h_hi_m', 'D_m', 'rule', 'default': bool, 'clamped': bool} (module docstring)."""
    near = float(settings.get('near_m', NEAR_M))
    hmax_ok = warp.d_B - near
    given = settings.get('depth_range_m')
    if given is not None:
        try:
            lo, hi = (float(x) for x in given)
        except (TypeError, ValueError):
            raise WarpError('The depth range must be two numbers of metres: [beyond the plane (negative), towards '
                            'the camera (positive)].') from None
        if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
            raise WarpError('The depth range must run from its lower to its upper end, in metres.')
        clamped = hi > hmax_ok
        return {'h_lo_m': lo, 'h_hi_m': min(hi, hmax_ok), 'D_m': None, 'rule': 'your setting', 'default': False,
                'clamped': bool(clamped)}
    vols = list(project.workspace.volumes)
    if not vols:
        raise WarpError('The residual parallax is measured over the task volume\'s depth range: add a task volume, or '
                        'set the depth range on the Canonical view page.')
    if warp.source.T_world_contact is None:
        raise WarpError('The source camera must be a project camera (its robot sets the depth range).')
    a = warp.source.T_world_contact[:3, 2]
    D = max(float(np.abs(a) @ np.asarray(v.size, float)) for v in vols)
    reach, why = pad_reach(project, warp, settings.get('q'))
    hi = min(D, hmax_ok, reach) if reach is not None else min(D, hmax_ok)
    vol = 'the task volumes\'' if len(vols) > 1 else 'the task volume\'s'
    rule = (f'beyond the plane: {vol} extent along the approach axis, {D * 1000:.0f} mm (the TCP may be anywhere in '
            'the volume); towards the camera: ')
    if reach is not None and hi == reach:
        rule += f'{why}, {hi * 1000:.0f} mm'
    elif hi == hmax_ok:
        rule += f'cut to {hi * 1000:.0f} mm, {near * 1000:.0f} mm in front of the camera'
    else:
        rule += f'the same extent, {hi * 1000:.0f} mm'
    return {'h_lo_m': -D, 'h_hi_m': hi, 'D_m': D, 'pad_reach_m': reach, 'default': True,
            'clamped': bool(hi == hmax_ok and hmax_ok < D), 'rule': rule + ' (DexVantage defaults)'}


def pad_reach(project, warp: Warp, q=None) -> tuple[float | None, str]:
    """(how far the contact pads reach from the plane towards the source camera (m), the rule) or (None, why not):
    an object held between the pads comes no nearer the camera than the pads' far end (the gripper's palm and fingers
    fill the rest)."""
    from .. import grasps as GR
    B = warp.source
    if B.T_world_contact is None or not B.robot_uid:
        return None, 'no robot'
    try:
        pts = GR.contact_points(project, B.robot_uid, q)
    except GR.GraspError:
        return None, 'no contact pads'
    if not pts:
        return None, 'no contact pads'
    P = np.concatenate(list(pts.values()))
    Ti = np.linalg.inv(B.T_world_contact)
    z = P @ Ti[:3, :3].T + Ti[:3, 3]
    side = 1.0 if float(B.T_contact_cam[2, 3]) >= warp.plane_offset_m else -1.0
    h = side * (z[:, 2] - warp.plane_offset_m)
    return max(0.0, float(h.max())), 'up to the contact pads\' far end (an object between the pads comes no nearer)'


def settings_of(project, options: dict | None = None) -> dict:
    st = dict(DEFAULTS)
    st['thresholds_px'] = dict(DEFAULTS['thresholds_px'])
    st.update((project.analysis.studies.get(SID) or {}) if project is not None else {})
    st.update(options or {})
    return st


def candidate_cameras(project) -> list:
    """The active rig's enabled sensors that have an imager, in rig order."""
    from .. import plucker as PK
    out = []
    for s in project.variant.sensors:
        if s.enabled and s.kind in ('camera', 'depth') and PK.pick_imager(s, PK.sensor_stream(s)) is not None:
            out.append(s)
    return out


def default_pair(project) -> tuple[str, str | None]:
    """(source uid, canonical uid or None): with two robots, a camera on a non-primary robot's link to the primary
    robot's; with one, the first camera on a link to the next camera."""
    cams = candidate_cameras(project)
    if not cams:
        raise WarpError('The canonical view warp needs a camera: add one on the Sensors and mounts page.')
    prim = project.primary_robot.uid if project.primary_robot is not None else ''
    on_link = [s for s in cams if s.mount.on_link]
    other = [s for s in on_link if s.mount.robot_uid != prim]
    mine = [s for s in on_link if s.mount.robot_uid == prim]
    if other and mine:
        return other[0].uid, mine[0].uid
    src = (on_link or cams)[0]
    rest = [s for s in cams if s.uid != src.uid]
    return src.uid, (rest[0].uid if rest else None)


def build(project, options: dict | None = None) -> tuple[Warp, dict]:
    """(Warp, settings used) from the settings: 'source' (a sensor uid), 'canonical' (a sensor uid, or a canonical view
    document {'camera', 'pose_in_contact', ...}), 'q' ({robot uid: q}), 'plane_offset_m', 'closing_sign_source',
    'closing_sign_canonical', 'depth_range_m', 'near_m', 'thresholds_px'."""
    if not project.robots:
        raise WarpError('The canonical view warp needs a robot: its gripper defines the contact plane.')
    st = settings_of(project, options)
    src, can = st.get('source'), st.get('canonical')
    if not src or (can is None or can == ''):
        dsrc, dcan = default_pair(project)
        src = src or dsrc
        if can is None or can == '':
            can = dcan
    if can is None:
        raise WarpError('The canonical view warp needs a second camera (the canonical view) besides the source: add '
                        'one, or load a canonical view file on the Canonical view page.')
    if isinstance(can, str) and can == src:
        raise WarpError('The source and the canonical camera are the same camera: choose two.')
    q = st.get('q')
    B = view_of_sensor(project, src, q, st.get('source_robot') or None, st.get('closing_sign_source', 1))
    A = view_from_dict(can) if isinstance(can, dict) else view_of_sensor(project, can, q,
                                                                         st.get('canonical_robot') or None,
                                                                         st.get('closing_sign_canonical', 1))
    try:
        off = float(st.get('plane_offset_m', 0.0))
    except (TypeError, ValueError):
        raise WarpError('The plane offset must be a number of metres.') from None
    w = make_warp(B, A, off)
    w.notes = B.notes + A.notes + w.notes
    st['source'] = src
    st['canonical'] = can if isinstance(can, str) else {'virtual': A.name}
    return w, st


# ====================================================================== the study's result
def status_of(px, thresholds: dict) -> tuple[str, str]:
    if px is None:
        return 'critical', 'no overlap'
    th = thresholds or DEFAULTS['thresholds_px']
    for sev in ('good', 'warning', 'serious'):
        if px <= float(th.get(sev, DEFAULTS['thresholds_px'][sev])):
            return sev, WORDS[sev]
    return 'critical', WORDS['critical']


def suggestion(project, warp: Warp, q=None) -> dict | None:
    """Move the source camera to the canonical pose in its own contact frame: the warp is then a pure homography with
    no parallax at any depth (t = 0). {'sensor_uid', 'parent_frame', 'robot_uid', 'pose', 'locked', 'text',
    'parallax_after_px': 0.0} or None (a virtual source, or a camera already there)."""
    from .. import camrender as CR
    from .. import plucker as PK
    from ..frames import Pose
    from ..robot.model import get_model
    B, A = warp.source, warp.canonical
    if B.virtual or B.T_world_contact is None or warp.baseline_m < 1e-4:
        return None
    s = project.variant.sensor(B.sensor_uid)
    if s is None:
        return None
    im = PK.pick_imager(s, PK.sensor_stream(s))
    T_wo = B.T_world_contact @ A.T_contact_cam
    T_wb = T_wo @ np.linalg.inv(im.pose_in_body.matrix())
    if s.mount.on_link:
        m = get_model(project, s.mount.robot_uid)
        T_wl = m.link_pose(_q_of(m, q, s.mount.robot_uid), s.mount.parent_frame)
        T_pose = np.linalg.inv(T_wl) @ T_wb
    else:
        T_pose = T_wb
    old = CR.sensor_world_pose(project, s, q).matrix()
    move = float(np.linalg.norm(T_wb[:3, 3] - old[:3, 3]))
    return {'sensor_uid': s.uid, 'sensor': s.name, 'parent_frame': s.mount.parent_frame,
            'robot_uid': s.mount.robot_uid, 'pose': Pose.from_matrix(T_pose).to_dict(), 'locked': bool(s.locked),
            'move_m': move, 'parallax_after_px': 0.0,
            'text': (f'Mount {s.name} at {A.name}\'s pose relative to the fingertips (moves it {move * 1000:.0f} mm): '
                     'the two views then share a centre, the warp is a pure homography and the parallax is 0 px at '
                     'every depth.' + (' The camera is locked: unlock it to apply.' if s.locked else ''))}


def analyse(project, options: dict | None = None, ctx=None, built: tuple | None = None) -> dict:
    """The study's result (JSON-able): views, H, the plane, the depth range, the parallax summary and map, coverage,
    the headline and the suggestion. built: (Warp, settings) from build() when the caller has them."""
    w, st = built if built is not None else build(project, options)
    if ctx is not None:
        ctx.report(0.2, 'the warp')
    rng = depth_range(project, w, st)
    near = float(st.get('near_m', NEAR_M))
    pm = parallax_map(w, rng['h_lo_m'], rng['h_hi_m'], near_m=near)
    if ctx is not None:
        ctx.report(0.8, 'the parallax map')
    cover = pm['covered']
    lo, hi = pm['at_lo_stats'], pm['at_hi_stats']
    p95 = max([x for x in (lo['p95_px'], hi['p95_px']) if x is not None], default=None)
    mx = max([x for x in (lo['max_px'], hi['max_px']) if x is not None], default=None)
    sev, word = status_of(p95, st.get('thresholds_px'))
    th = st.get('thresholds_px') or DEFAULTS['thresholds_px']
    share = float(cover.mean()) if cover.size else 0.0
    sg = suggestion(project, w, st.get('q'))
    if p95 is None:
        advice = (f'{w.source.name} sees none of the contact plane that {w.canonical.name} sees: no pixel of the '
                  'canonical view can be filled. Check the two cameras look at the fingertips.')
    elif sev == 'good':
        advice = (f'Off the contact plane the warp is at most {p95:.2f} px wrong (P95) over the depth range: pooled '
                  'images line up.')
    else:
        advice = (f'Points {abs(rng["h_lo_m"]) * 1000:.0f} mm beyond the plane or {rng["h_hi_m"] * 1000:.0f} mm in '
                  f'front of it land up to {p95:.1f} px (P95) from where {w.canonical.name} would see them; the warp '
                  f'is exact only on the contact plane. {sg["text"] if sg else ""}').strip()
    width = int(w.canonical.camera.width)
    beyond = p95 is not None and p95 > width          # the bound is right but past the image: say so, not 11 878 px
    if beyond:
        advice = (f'The residual parallax is larger than {w.canonical.name}\'s image (bound {p95:.0f} px P95 against '
                  f'a {width} px wide image): off the contact plane the warped view does not resemble the canonical '
                  'one. ' + advice)
    mp = pm['map']
    doc = {
        'convention': CONVENTION,
        'source': w.source.to_dict(), 'canonical': w.canonical.to_dict(),
        'homography': _lists(w.H), 'homography_inverse': _lists(np.linalg.inv(w.H)),
        'relative': {'R': _lists(w.R), 't_m': _lists(w.t), 'baseline_m': w.baseline_m},
        'plane': {'offset_m': w.plane_offset_m, 'n_B': _lists(w.n_B), 'd_B_m': w.d_B, 'n_A': _lists(w.n_A),
                  'd_A_m': w.d_A, 'in_contact_frame': 'z = offset (normal along the approach axis)',
                  'offset_default': float(st.get('plane_offset_m', 0.0)) == 0.0},
        'depth_range': rng,
        'parallax': {'at_lo': lo, 'at_hi': hi, 'p95_px': p95, 'max_px': mx,
                     'map': {'stride': pm['stride'], 'u': _lists(pm['u']), 'v': _lists(pm['v']),
                             'values': [[None if not math.isfinite(x) else round(float(x), 3) for x in r] for r in mp]},
                     'rule': 'max(|Δ(h_lo)|, |Δ(h_hi)|) over canonical pixels the warp fills'},
        'coverage': {'share': share, 'text': f'{share * 100:.1f} % of {w.canonical.name}\'s image reads a source '
                                             'pixel (on the sampled grid)'},
        'headline': {'value': p95, 'unit': 'px', 'status': sev, 'word': word, 'advice': advice,
                     'max_px': mx, 'thresholds_px': th,
                     'thresholds_default': th == DEFAULTS['thresholds_px'],
                     'image_width_px': width, 'beyond_image': bool(beyond)},
        'suggestion': sg,
        'notes': list(w.notes) + ([f'The near end of the depth range is cut to {rng["h_hi_m"] * 1000:.0f} mm: a '
                                   'point must stay in front of the source camera.'] if rng['clamped'] else []),
        'settings': json.loads(json.dumps(st, default=_jsonable)),
    }
    return doc


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


# ====================================================================== overlays (Camera view and 3D view)
def _grid_lines(camera, n: int = 8, per: int = 48) -> list:
    """Polylines of an n × n grid over an image (its edges included), as (K, 2) pixel arrays."""
    W, H = camera.width - 0.5, camera.height - 0.5
    xs = np.linspace(-0.5, W, n + 1)
    ys = np.linspace(-0.5, H, n + 1)
    t = np.linspace(0.0, 1.0, per)
    lines = [np.c_[np.full(per, x), -0.5 + t * (H + 0.5)] for x in xs]
    lines += [np.c_[-0.5 + t * (W + 0.5), np.full(per, y)] for y in ys]
    return lines


def source_to_canonical(warp: Warp, uv_B):
    """(uv_A (N, 2), ok): source pixels carried through the plane into the canonical image (the forward warp)."""
    uv, shape = _rows(uv_B, 2)
    r, ok = warp.source.camera.unproject(uv, with_ok=True)
    den = r @ warp.n_B
    with np.errstate(divide='ignore', invalid='ignore'):
        s = -warp.d_B / den
    ok = ok & np.isfinite(s) & (s > 0)
    X_B = r * np.where(ok, s, 0.0)[:, None]
    X_A = X_B @ warp.R.T + warp.t
    if warp.canonical.camera.model in ('pinhole', 'brown'):
        ok = ok & (X_A[:, 2] > 0)
    p, _ = warp.canonical.camera.project(np.where(ok[:, None], X_A, np.nan))
    ok = ok & np.isfinite(p).all(axis=1)
    return p.reshape(shape + (2,)), ok.reshape(shape)


def _split(P, ok) -> list:
    out, cur = [], []
    for p, k in zip(P, ok):
        if k:
            cur.append(p)
        elif cur:
            out.append(np.array(cur))
            cur = []
    if cur:
        out.append(np.array(cur))
    return [c for c in out if len(c) >= 2]


def warped_grid(warp: Warp, into: str = 'canonical', n: int = 8) -> list:
    """Polylines for the Camera view: into='canonical' carries the source image's n × n grid into the canonical
    image (where B's pixels land); into='source' carries the canonical image's grid into the source image (which of
    B's pixels the warp reads). Only the parts inside the target image are kept."""
    if into == 'canonical':
        cam_from, cam_to, fn = warp.source.camera, warp.canonical.camera, source_to_canonical
    else:
        cam_from, cam_to = warp.canonical.camera, warp.source.camera
        fn = lut_at
    out = []
    for line in _grid_lines(cam_from, n):
        p, ok = fn(warp, line)
        if into == 'source':
            ok = np.isfinite(p).all(axis=1)
        inside = ok & (p[:, 0] >= -0.5) & (p[:, 0] <= cam_to.width - 0.5) & (p[:, 1] >= -0.5) & \
            (p[:, 1] <= cam_to.height - 0.5)
        out += _split(p, inside)
    return out


def world_geometry(warp: Warp, frustum_m: float = 0.08, half_m: float = 0.04) -> dict:
    """World-frame geometry at the source robot for the 3D view: the contact plane's square (half_m about the TCP),
    the source camera's frustum and the canonical camera placed at its pose relative to the source robot's
    fingertips. {'plane': (5, 3), 'source': [polylines], 'canonical': [polylines]} or {} for a virtual source."""
    from .. import camrender as CR
    from ..frames import Pose
    B, A = warp.source, warp.canonical
    if B.T_world_contact is None:
        return {}
    Tc = B.T_world_contact
    o = warp.plane_offset_m
    sq = np.array([[-half_m, -half_m, o], [half_m, -half_m, o], [half_m, half_m, o], [-half_m, half_m, o],
                   [-half_m, -half_m, o]])
    plane = sq @ Tc[:3, :3].T + Tc[:3, 3]
    out = {'plane': plane}
    for key, v in (('source', B), ('canonical', A)):
        T = Tc @ v.T_contact_cam
        apex, border, spokes = CR.frustum_body(v.camera, Pose.from_matrix(T), frustum_m)
        out[key] = [b for b in _split(border, np.isfinite(border).all(axis=1))] + list(spokes)
        out[key + '_centre'] = apex
    return out


# ====================================================================== the export
def _tag(text: str) -> str:
    t = re.sub(r'[^A-Za-z0-9_-]+', '_', str(text)).strip('_')
    return t[:40] or 'camera'


def write_yml(path: str, map_x, map_y, H, meta: dict):
    """An OpenCV FileStorage YAML with map1 (x), map2 (y) as float32 matrices, H and the sizes."""
    import cv2
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise OSError(f'cannot open {os.path.basename(path)} for writing')
    try:
        fs.write('map1', map_x)
        fs.write('map2', map_y)
        fs.write('homography', np.asarray(H, np.float64))
        for k, v in meta.items():
            fs.write(k, v)
    finally:
        fs.release()


def export_warp(project, out_dir: str, options: dict | None = None, ctx=None) -> dict:
    """Write the warp of the settings (plus options) under out_dir/canonical_warp/: <stem>_map1.npy and
    <stem>_map2.npy (float32 x and y for cv2.remap), <stem>_remap.yml (the same in OpenCV FileStorage YAML),
    <stem>_grid.npy (torch grid_sample, see grid_sample_grid) with <stem>_valid.npy (uint8), <stem>_parallax.npy
    (float32 |Δ| px at full canonical resolution, the larger of the two depth extremes; NaN where not covered),
    <stem>_homography.json (H, the conventions, both views, the plane, the depth range, the parallax summary, the files)
    and <stem>_canonical_view.json (the canonical camera, loadable as a virtual canonical view). options 'formats':
    any of npy, yml, grid, json (default all; json always). Returns {'files' (relative to out_dir), 'notes'}."""
    opts = dict(options or {})
    formats = tuple(opts.pop('formats', FORMATS) or FORMATS)
    bad = [f for f in formats if f not in FORMATS]
    if bad:
        raise WarpError(f'The canonical warp formats are {", ".join(FORMATS)}; not {", ".join(bad)}.')
    w, st = build(project, opts)
    doc = analyse(project, opts, None, built=(w, st))
    rng = doc['depth_range']
    if ctx is not None:
        ctx.report(0.3, 'the remap maps')
    mx, my, ok = remap_maps(w)
    sub = 'canonical_warp'
    folder = os.path.join(out_dir, sub)
    try:
        os.makedirs(folder, exist_ok=True)
    except OSError as e:
        raise WarpError(f'Could not make the folder {sub}: {e.strerror or e}.') from None
    stem = f'{_tag(w.source.name)}_to_{_tag(w.canonical.name)}'
    files, notes = {}, []

    def rel(name):
        return f'{sub}/{stem}_{name}'
    try:
        if 'npy' in formats:
            np.save(os.path.join(out_dir, rel('map1.npy')), mx)
            np.save(os.path.join(out_dir, rel('map2.npy')), my)
            files['map1'], files['map2'] = rel('map1.npy'), rel('map2.npy')
        if 'yml' in formats:
            write_yml(os.path.join(out_dir, rel('remap.yml')), mx, my, w.H,
                      {'canonical_width': w.canonical.camera.width, 'canonical_height': w.canonical.camera.height,
                       'source_width': w.source.camera.width, 'source_height': w.source.camera.height,
                       'invalid_value': INVALID, 'convention': CONVENTION})
            files['yml'] = rel('remap.yml')
        if 'grid' in formats:
            np.save(os.path.join(out_dir, rel('grid.npy')), grid_sample_grid(w, mx, my, ok))
            np.save(os.path.join(out_dir, rel('valid.npy')), ok.astype(np.uint8))
            files['grid'], files['valid'] = rel('grid.npy'), rel('valid.npy')
        if ctx is not None:
            ctx.report(0.7, 'the parallax map')
        uv = pixel_grid(w.canonical.camera)
        near = float(st.get('near_m', NEAR_M))
        mags = []
        for h in (rng['h_lo_m'], rng['h_hi_m']):
            D, okp = parallax_at(w, uv, h, near)
            mags.append(np.where(okp & ok, np.linalg.norm(np.nan_to_num(D), axis=-1), np.nan))
        par = np.fmax(mags[0], mags[1]).astype(np.float32)
        np.save(os.path.join(out_dir, rel('parallax.npy')), par)
        files['parallax'] = rel('parallax.npy')
        full = _stats(par.astype(np.float64), np.isfinite(par))
        vdoc = view_document(w.canonical)
        p = os.path.join(out_dir, rel('canonical_view.json'))
        with open(p, 'w', encoding='utf-8') as fh:
            json.dump(vdoc, fh, indent=1, ensure_ascii=False)
            fh.write('\n')
        files['canonical_view'] = rel('canonical_view.json')
        hdoc = {'schema': SCHEMA, 'convention': CONVENTION, 'homography': doc['homography'],
                'homography_inverse': doc['homography_inverse'], 'relative': doc['relative'], 'plane': doc['plane'],
                'source': doc['source'], 'canonical': doc['canonical'], 'depth_range': rng,
                'parallax': {k: v for k, v in doc['parallax'].items() if k != 'map'},
                'parallax_full_resolution': full, 'coverage_full_resolution': float(ok.mean()),
                'invalid_value': INVALID, 'grid_invalid_value': GRID_INVALID,
                'files': {k: os.path.basename(v) for k, v in files.items()},
                'usage': {'opencv': 'map1 = np.load("…_map1.npy"); map2 = np.load("…_map2.npy"); '
                                    'canonical = cv2.remap(img_B, map1, map2, cv2.INTER_LINEAR, '
                                    'borderMode=cv2.BORDER_CONSTANT)',
                          'torch': 'grid = torch.from_numpy(np.load("…_grid.npy"))[None]; canonical = '
                                   'torch.nn.functional.grid_sample(img_B, grid, mode="bilinear", '
                                   'padding_mode="zeros", align_corners=False)',
                          'homography_only': 'cv2.warpPerspective(img_B, H, (W_A, H_A)) is exact on the plane only '
                                             'for undistorted images; the maps include both lenses'},
                'notes': doc['notes']}
        p = os.path.join(out_dir, rel('homography.json'))
        with open(p, 'w', encoding='utf-8') as fh:
            json.dump(hdoc, fh, indent=1, ensure_ascii=False, allow_nan=False)
            fh.write('\n')
        files['json'] = rel('homography.json')
    except OSError as e:
        raise WarpError(f'Could not write the canonical warp: {e.strerror or e}.') from None
    notes.append(f'Exact on the contact plane only; off it up to {_fmt(doc["parallax"]["p95_px"])} px (P95) over the '
                 f'depth range {rng["h_lo_m"] * 1000:.0f} to {rng["h_hi_m"] * 1000:.0f} mm.')
    if w.source.camera.model != 'pinhole' or w.canonical.camera.model != 'pinhole':
        notes.append('A lens is not pinhole: use the remap maps; the homography alone holds for undistorted images.')
    notes += doc['notes']
    return {'files': list(files.values()), 'notes': notes, 'result': doc}


def _fmt(x):
    return '—' if x is None else f'{x:.2f}'


def self_test() -> tuple[float, float]:
    """(homography error px, lens remap error px) on one synthetic pair (the self-test's line; the validation checks
    vla.warp_plane_exact and vla.warp_lut_lens_cv2 do the same over more cases)."""
    import cv2

    from ..frames import Pose, quat_from_rpy
    from ..optics import CameraModel
    flip = np.diag([1.0, -1.0, -1.0])

    def pose(p, rpy):
        M = Pose(np.asarray(p, float), quat_from_rpy(*rpy)).matrix()
        M[:3, :3] = M[:3, :3] @ flip
        return M

    def proj(cam, X):
        K, D, m = cam.to_opencv()
        f = cv2.fisheye.projectPoints if m == 'kb' else cv2.projectPoints
        z = np.zeros((3, 1))
        return f(np.ascontiguousarray(X).reshape(-1, 1, 3), z, z, K, D)[0].reshape(-1, 2)
    TA, TB = pose([0.01, 0.02, 0.16], [0.05, -0.08, 0.3]), pose([-0.03, 0.01, 0.12], [-0.1, 0.07, -0.4])
    P = np.array([[x, y, 0.0] for x in np.linspace(-0.03, 0.03, 7) for y in np.linspace(-0.02, 0.02, 5)])
    pairs = ((CameraModel(640, 480, 520.0, 515.0, 321.3, 238.9),
              CameraModel(800, 600, 610.0, 612.0, 401.2, 297.5)),
             (CameraModel(640, 480, 420.0, 418.0, 322.0, 241.0, 'brown', (-0.28, 0.09, 0.0012, -0.0007, -0.012)),
              CameraModel(800, 600, 330.0, 331.0, 401.0, 299.0, 'kb', (0.021, -0.008, 0.0021, -0.0003))))
    errs = []
    for cA, cB in pairs:
        w = make_warp(View('B', cB, TB), View('A', cA, TA))
        iA, iB = np.linalg.inv(TA), np.linalg.inv(TB)
        pa, pb = proj(cA, P @ iA[:3, :3].T + iA[:3, 3]), proj(cB, P @ iB[:3, :3].T + iB[:3, 3])
        if cA.model == 'pinhole':
            x = np.c_[pb, np.ones(len(pb))] @ w.H.T
            errs.append(float(np.abs(x[:, :2] / x[:, 2:] - pa).max()))
        else:
            uv, ok = lut_at(w, pa)
            errs.append(float(np.abs(uv - pb).max()) if ok.all() else float('inf'))
    return errs[0], errs[1]


@exporter('canonical_warp', 'Canonical view warp (OpenCV remap maps, GPU grid, homography JSON)', needs_robot=True,
          description='Robot B\'s wrist camera warped to the canonical viewpoint through the fingertip contact plane '
                      '(spec §7.16): cv2.remap maps (.npy and .yml), a grid_sample .npy and H as JSON.')
def _export(project, out_dir, options, ctx):
    r = export_warp(project, out_dir, options, ctx)
    return {'files': r['files'], 'notes': r['notes']}
