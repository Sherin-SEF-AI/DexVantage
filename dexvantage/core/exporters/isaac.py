"""Isaac Lab and Isaac Sim export (spec §9.2, §8.5, §9.4): USD camera prims, Isaac Lab camera sensor configs as Python,
and the extrinsic domain-randomisation bounds of the tolerance study.

    @exporter('isaac', 'Isaac Lab and Isaac Sim cameras', ...)
    options: q ({robot uid: joint vector}: the pose the USD file shows link-mounted cameras at; default every robot at
             home), env_ns (default '{ENV_REGEX_NS}'), robot_prim (default 'Robot': the articulation's prim name under
             the environment), scale (tolerance scale, default 1), usd ('auto' (default): pxr when usd-core is
             installed, else the text writer; 'text': always the text writer)

Files (under out_dir/isaac/):

    cameras.usda              one Camera prim per imager of every enabled camera and depth sensor (below)
    isaaclab_cameras.py       Isaac Lab v3.0.0: CameraCfg per imager, OpenCV lens models, the randomisation bounds
    isaaclab_v2_cameras.py    Isaac Lab v2.3.2: TiledCameraCfg per imager, (w, x, y, z) quaternions, pinhole only
    isaac_cameras.json        the same figures as data (dexvantage.isaac/1), with the notes

The sources, read on 2026-11-07 (https://github.com/isaac-sim/IsaacLab):
- v3.0.0 (tag, commit e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4): source/isaaclab/isaaclab/sensors/camera/camera_cfg.py
  (CameraCfg: OffsetCfg pos, rot "(x, y, z, w) w.r.t. the parent frame", convention 'ros' = forward +Z, up −Y;
  width, height, data_types, update_period from sensor_base_cfg.py; prim_path with {ENV_REGEX_NS}),
  tiled_camera_cfg.py ("TiledCameraCfg is deprecated. Use CameraCfg directly"), sim/spawners/sensors/sensors_cfg.py
  (PinholeCameraCfg focal_length, horizontal_aperture, vertical_aperture, offsets, clipping_range, distortion;
  OpenCvPinholeDistortionCfg: fx, fy, cx, cy, image_size, k1..k6, p1, p2, s1..s4; OpenCvFisheyeDistortionCfg: k1..k4,
  max_fov), sim/spawners/sensors/sensors.py (spawn_camera: the omni:lensdistortion:<model>:* attributes and the
  OmniLensDistortionOpenCvPinholeAPI / ...FisheyeAPI schemas; "Camera aperture offsets are not supported by
  Omniverse"), sensors/camera/camera.py (_initialize_intrinsics: a prim without a lens-distortion model gets
  fx = fy = width · focalLength / horizontalAperture and cx = width / 2, cy = height / 2; with one, its authored fx,
  fy, cx, cy), utils/sensors.py (convert_camera_intrinsics_to_usd: pixel size 1 / width when no focal length given).
- v2.3.2 (tag, commit 37ddf626871758333d6ed89cf64ad702aef127d0): camera_cfg.py (OffsetCfg rot (w, x, y, z)),
  tiled_camera_cfg.py (TiledCameraCfg, not deprecated), sensors_cfg.py (no OpenCV distortion fields).

**Camera prims (USD).** Stage metersPerUnit 1, upAxis Z, defaultPrim /DexVantage. A static sensor's imagers are under
/DexVantage/World; a link-mounted sensor's under /DexVantage/<robot>/<link>, an Xform at the link's world pose at the
export's joint state, so the file shows the rig as posed and each camera's own transform is its offset from the link
(what a robot's link prim needs). A camera's transform is its optical frame turned by diag(1, −1, −1) (a USD camera
looks along −Z with +Y up; the optical frame along +Z with +Y down), written as one xformOp:transform (USD's row-vector
matrix, the transpose of DexVantage's). Lens: Isaac Lab's own rule, pixel size 1 / width, so horizontalAperture = 1,
focalLength = fx / W, verticalAperture = (H / W)(fx / fy) and, from GfCamera's frustum (the window on the image plane
is the aperture ± offset over the focal length), horizontalApertureOffset = ((W − 1)/2 − cx) / W and
verticalApertureOffset = verticalAperture (cy − (H − 1)/2) / H, in DexVantage's pixel convention (OpenCV: pixel
centres at integers). USD then reproduces the pinhole part (fx, fy, cx, cy) exactly; the check formats.isaac_usd
projects points through pxr's Gf.Camera to verify it. clippingRange is (0.005, 50) m, the MuJoCo scene's labelled
defaults. Each prim also carries Isaac's cameraProjectionType token ('pinhole') and, where the lens needs it, the
OpenCV lens-distortion attributes Isaac Lab's spawner writes, and customData (sensor uid, imager, lens model, the
tolerance bounds). With usd-core (the 'usd' extra) the file is written by pxr; without it by a text writer of the
same content (the notes say which).

**Lens models.** Isaac Lab v3.0.0's projection is pinhole (fx = fy, centred) unless an OpenCV lens-distortion model is
authored, whose fx, fy, cx, cy then rule. So every imager whose pinhole part is not square and centred, or that
has distortion, gets one: Brown-Conrady (k1, k2, p1, p2, k3 and the rational k4..k6) -> OpenCvPinholeDistortionCfg,
exact; Kannala-Brandt -> OpenCvFisheyeDistortionCfg (k1..k4, max_fov = twice the valid half angle), exact;
equidistant -> OpenCvFisheyeDistortionCfg with k = 0, exact. Equisolid and stereographic have no match in Isaac Lab
(its FisheyeCameraCfg models are f-theta polynomials with undocumented coefficients): the export fits the four
Kannala-Brandt terms to the model's r(θ) by least squares over the image's angles and states the largest pixel gap
of the fit. v2.3.2 has no OpenCV fields: its file carries the pinhole part only (square, centred: what its spawner
can show), and the notes give the largest pixel gap from the real lens (or say the lens cannot be shown).

**Randomisation bounds** come from core/tolerance.bounds (the tolerance study's ± bound per axis: shifts along the
mount parent frame's x, y, z; turns about the sensor body's own x, y, z, applied as R' = R · Rx · Ry · Rz; or ± 2 σ of
a calibration covariance's marginals). The configs carry them per camera with perturbed_offset(name, d) (d = dx, dy,
dz in m, droll, dpitch, dyaw in degrees, applied to the sensor body exactly as core/tolerance applies a draw) and
sample_offset(name, rng) (uniform within the bounds), and randomized_cameras(seed): every CameraCfg with one draw.

Known limits: Isaac Lab and Isaac Sim are not installed here, so the configs are checked by compiling and executing
them against stub classes with the fields of the cited source (formats.isaac_config), not in Isaac Lab; the USD is
checked with pxr (formats.isaac_usd), not rendered in Isaac Sim; IR and ToF imagers are written as visible-light RGB
prims (Isaac renders no near-infrared); a depth sensor's reference imager gets 'distance_to_image_plane' and no noise
model; the OpenCV attributes' pixel convention is taken to be OpenCV's (pixel centres at integers), which the Isaac
Lab source does not state, and its pinhole K puts the centre at width / 2, half a pixel from OpenCV's (W − 1)/2 if
its pixels are centred at integers; the USD aperture offsets (an off-centre principal point) are exact for pxr's
Gf.Camera, but Isaac Lab's spawner warns "Camera aperture offsets are not supported by Omniverse. These parameters
will be ignored", so the off-centre cx, cy is written through the OpenCV lens attributes (an opencvPinhole model with
zero distortion when the lens has none), which v3.0.0 authors and projects through (PP_SOURCE['v3']); v2.3.2 has no
such attributes and ignores the offsets (PP_SOURCE['v2']), so its render is centred, and each such camera is named
with its offset in the notes, in its config comment, in isaac_cameras.json ('principal_point') and in the Export
dialog (principal_point_lines); non-square pixels conform to the render's aspect in a
renderer that does not read the OpenCV attributes; the randomisation is drawn once per config build (per-environment
draws need Camera.set_world_poses at reset, not generated); sensor housings and the robot itself are not in the file.
"""
from __future__ import annotations

import json
import math
import os
import re

import numpy as np

from ..frames import quat_from_matrix
from ..jobs import StudyError
from . import _common as C
from . import exporter

FOLDER = 'isaac'
USD_FILE = f'{FOLDER}/cameras.usda'
V3_FILE = f'{FOLDER}/isaaclab_cameras.py'
V2_FILE = f'{FOLDER}/isaaclab_v2_cameras.py'
JSON_FILE = f'{FOLDER}/isaac_cameras.json'
SCHEMA = 'dexvantage.isaac/1'
SOURCE_URL = 'https://github.com/isaac-sim/IsaacLab'
V3_TAG, V3_COMMIT = 'v3.0.0', 'e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4'
V2_TAG, V2_COMMIT = 'v2.3.2', '37ddf626871758333d6ed89cf64ad702aef127d0'
SOURCE_READ = '2026-11-07'
OPTICAL_TO_USD = np.diag([1.0, -1.0, -1.0])
CLIP_M = (0.005, 50.0)              # defaults of DexVantage (core/sim/scene NEAR_M, FAR_M), labelled so
CENTRE_TOL_PX = 1e-6                # a principal point this close to the centre counts as centred
SQUARE_TOL = 1e-9                   # |fx − fy| / fx below this counts as square pixels
ROOT = 'DexVantage'
DEFAULTS = {'env_ns': '{ENV_REGEX_NS}', 'robot_prim': 'Robot', 'scale': 1.0, 'usd': 'auto'}
DISTORTION_API = {'opencvPinhole': 'OmniLensDistortionOpenCvPinholeAPI',
                  'opencvFisheye': 'OmniLensDistortionOpenCvFisheyeAPI'}
# where an off-centre principal point can reach an Isaac render, at the pinned tags (read 2026-11-07)
PP_SOURCE = {
    'v3': 'Isaac Lab v3.0.0 (e3f7d9f) source/isaaclab/isaaclab/sim/spawners/sensors/sensors.py L52-58 and L68-108 (the '
          'OpenCV lens model authored as omni:lensdistortion:<model>:fx, fy, cx, cy, "the RTX/OVRTX renderer honors '
          'these attributes natively"), L162-165 ("Camera aperture offsets are not supported by Omniverse. These '
          'parameters will be ignored."), L190-192; sensors/camera/camera.py L828-857 '
          '(_read_authored_opencv_intrinsics: "these intrinsics may be non-square (fx != fy) or off-center") and '
          'L879-904 (_initialize_intrinsics: a prim without the model gets cx = width * 0.5, cy = height * 0.5)',
    'v2': 'Isaac Lab v2.3.2 (37ddf62) source/isaaclab/isaaclab/sim/spawners/sensors/sensors.py L108-111 (the same '
          'aperture-offset warning), sim/spawners/sensors/sensors_cfg.py (PinholeCameraCfg: no distortion or OpenCV '
          'fields), sensors/camera/camera.py L589-599 ("currently rendering does not use aperture offsets"; c_x = '
          'width * 0.5, c_y = height * 0.5)',
}
COEFFS = {'opencvPinhole': ('k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'p1', 'p2', 's1', 's2', 's3', 's4'),
          'opencvFisheye': ('k1', 'k2', 'k3', 'k4')}


# ====================================================================== the cameras as Isaac sees them
def prim_token(text: str) -> str:
    """A USD prim name that keeps the case of a robot link or prim name (the importers keep link names; only
    characters USD identifiers cannot hold become '_')."""
    s = re.sub(r'[^A-Za-z0-9_]+', '_', str(text or '')).strip('_') or 'x'
    return s if not s[0].isdigit() else '_' + s


def usd_name(text: str) -> str:
    """A USD prim name (identifier: a letter or underscore, then letters, digits, underscores)."""
    s = C.frame_name(text)
    return s if not s[0].isdigit() else '_' + s


def usd_lens(cam) -> dict:
    """USD camera lens attributes reproducing cam's pinhole part (module docstring; Isaac's pixel size 1 / width)."""
    W, H = cam.width, cam.height
    f = cam.fx / W
    v_ap = H / W * cam.fx / cam.fy
    return {'focalLength': f, 'horizontalAperture': 1.0, 'verticalAperture': v_ap,
            'horizontalApertureOffset': ((W - 1) / 2 - cam.cx) / W,
            'verticalApertureOffset': v_ap * (cam.cy - (H - 1) / 2) / H}


def approx_error_px(cam, approx, n: int = 41) -> float:
    """Largest distance (px) between where cam and approx put the same ray, over an n × n grid of cam's pixels that
    have a ray (corners and edges included); inf when approx cannot map one of them (a pinhole and a ray beyond 90°)."""
    u, v = np.meshgrid(np.linspace(-0.5, cam.width - 0.5, n), np.linspace(-0.5, cam.height - 0.5, n))
    uv = np.stack([u.ravel(), v.ravel()], 1)
    rays, ok = cam.unproject(uv, with_ok=True)
    if not ok.any():
        return float('nan')
    uv2, _in_image = approx.project(rays[ok])            # a ray it maps just outside the image still counts
    if not np.isfinite(uv2).all():
        return float('inf')
    return float(np.linalg.norm(uv2 - uv[ok], axis=1).max())


def _kb_fit(cam) -> tuple[tuple, float]:
    """Kannala-Brandt terms fitted to an equisolid or stereographic lens's r(θ)/f over the angles its image spans,
    and the largest pixel gap of the fit (approx_error_px)."""
    from ..optics import CameraModel
    from ..optics.lens import r_of_theta
    corners = np.array([[-0.5, -0.5], [cam.width - 0.5, -0.5], [-0.5, cam.height - 0.5],
                        [cam.width - 0.5, cam.height - 0.5]])
    rays, ok = cam.unproject(corners, with_ok=True)
    tmax = float(np.max(np.arctan2(np.hypot(rays[ok, 0], rays[ok, 1]), rays[ok, 2]))) if ok.any() else cam.max_theta
    tmax = min(max(tmax, 1e-3), cam.max_theta)
    th = np.linspace(1e-4, tmax, 400)
    g = r_of_theta(th, cam.model)
    A = np.stack([th ** 3, th ** 5, th ** 7, th ** 9], 1)
    k = np.linalg.lstsq(A, g - th, rcond=None)[0]
    try:
        approx = CameraModel(cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy, 'kb', tuple(float(x) for x in k))
        err = approx_error_px(cam, approx)
    except ValueError:
        err = float('inf')
    return tuple(float(x) for x in k), err


def lens_model(cam) -> dict:
    """{'distortion': None | {'model': 'opencvPinhole' | 'opencvFisheye', fx, fy, cx, cy, image_size, coefficients,
    'max_fov'?}, 'exact': bool, 'fit_error_px': float | None, 'text': what was done}."""
    W, H = cam.width, cam.height
    centred = abs(cam.cx - (W - 1) / 2) <= CENTRE_TOL_PX and abs(cam.cy - (H - 1) / 2) <= CENTRE_TOL_PX
    square = abs(cam.fx - cam.fy) <= SQUARE_TOL * cam.fx
    base = {'fx': cam.fx, 'fy': cam.fy, 'cx': cam.cx, 'cy': cam.cy, 'image_size': [W, H]}
    m, d = cam.model, tuple(cam.dist)
    if m == 'pinhole':
        if centred and square:
            return {'distortion': None, 'exact': True, 'fit_error_px': None,
                    'text': 'pinhole, square pixels, centred: the focal length and aperture alone'}
        return {'distortion': {'model': 'opencvPinhole', **base, 'coefficients': {}}, 'exact': True,
                'fit_error_px': None,
                'text': 'pinhole with ' + ('non-square pixels' if not square else 'an off-centre principal point')
                        + ': an OpenCV pinhole model with zero distortion carries fx, fy, cx, cy'}
    if m == 'brown':
        names = ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6')
        co = {n: float(v) for n, v in zip(names, d)}
        return {'distortion': {'model': 'opencvPinhole', **base, 'coefficients': co}, 'exact': True,
                'fit_error_px': None, 'text': 'Brown-Conrady: OpenCV pinhole model, the same terms (exact)'}
    if m in ('kb', 'equidistant'):
        k = d if m == 'kb' else (0.0, 0.0, 0.0, 0.0)
        co = {f'k{i + 1}': float(v) for i, v in enumerate(k)}
        return {'distortion': {'model': 'opencvFisheye', **base, 'coefficients': co,
                               'max_fov': float(min(2 * cam.max_theta, 2 * math.pi))}, 'exact': True,
                'fit_error_px': None,
                'text': ('Kannala-Brandt: OpenCV fisheye model, the same terms (exact)' if m == 'kb' else
                         'equidistant: OpenCV fisheye model with k1..k4 = 0, which is the equidistant projection '
                         '(exact)')}
    k, err = _kb_fit(cam)
    co = {f'k{i + 1}': v for i, v in enumerate(k)}
    return {'distortion': {'model': 'opencvFisheye', **base, 'coefficients': co,
                           'max_fov': float(min(2 * cam.max_theta, 2 * math.pi))}, 'exact': False,
            'fit_error_px': err,
            'text': (f'{m}: Isaac Lab has no {m} model; the OpenCV fisheye terms are a least-squares fit to it, '
                     f'largest gap {err:.3g} px over the image')}


def principal_point(cam) -> dict:
    """{'offset_px': [cx − (W−1)/2, cy − (H−1)/2] (DexVantage's pixel convention: centres at integers),
    'off_centre': beyond CENTRE_TOL_PX}."""
    d = [float(cam.cx - (cam.width - 1) / 2), float(cam.cy - (cam.height - 1) / 2)]
    return {'offset_px': d, 'off_centre': bool(max(abs(d[0]), abs(d[1])) > CENTRE_TOL_PX)}


def principal_point_text(title: str, pp: dict) -> str:
    """One sentence per camera: where its off-centre principal point goes in each Isaac Lab version."""
    dx, dy = pp['offset_px']
    return (f'{title}: principal point {dx:+.3g}, {dy:+.3g} px from the centre. Isaac Lab v3.0.0 gets it through the '
            'OpenCV lens attributes (omni:lensdistortion:opencvPinhole:cx, cy, which its renderer projects through); '
            'the USD aperture offsets carry it too, but Omniverse ignores them. Isaac Lab v2.3.2 has no OpenCV lens '
            'attributes and ignores aperture offsets, so its render is centred: '
            f'{math.hypot(dx, dy):.3g} px off this camera\'s principal point.')


def principal_point_lines(project) -> list:
    """principal_point_text for every imager in use with an off-centre principal point (no forward kinematics:
    cheap enough for the export dialog)."""
    from .. import library as LB
    from ..studies import coverage as CV
    out = []
    for s in project.variant.sensors:
        if not s.enabled or s.kind not in ('camera', 'depth'):
            continue
        try:
            ims = LB.sensor_imagers(s, CV.sensor_mode(s, {}))
        except LB.LibraryError:
            continue
        for im in ims or []:
            pp = principal_point(im.camera)
            if pp['off_centre']:
                out.append(principal_point_text(f'{s.name} · {im.name}', pp))
    return out


def _v2_error(cam) -> float:
    """Largest pixel gap between the lens and the square, centred pinhole Isaac Lab v2.3.2 can show (inf: the lens
    sees rays a pinhole cannot)."""
    from ..optics import CameraModel
    pin = CameraModel(cam.width, cam.height, cam.fx, cam.fx, (cam.width - 1) / 2, (cam.height - 1) / 2)
    if cam.model == 'pinhole' and pin == cam:
        return 0.0
    return approx_error_px(cam, pin)


def _finite(x):
    """x, or None when it is not a finite number (JSON has no infinity: None here means "cannot be shown")."""
    return float(x) if x is not None and math.isfinite(float(x)) else None


def _data_types(rc) -> tuple[list, str]:
    """Isaac Lab data types of an imager: 'rgb' for every imager; plus 'distance_to_image_plane' on a depth sensor's
    reference imager."""
    from .. import library as LB
    types, why = ['rgb'], ''
    role = {'ir_left': 'a left infrared', 'ir_right': 'a right infrared', 'tof': 'a time-of-flight',
            'mono': 'a monochrome'}.get(rc.imager.role)
    if role:
        why = (f'{rc.title} is {role} imager: Isaac renders visible light, so its prim is an RGB camera with that '
               'imager\'s geometry')
    entry = LB.get(rc.sensor.model_id) if rc.sensor.model_id else None
    ref = entry.reference_imager if entry is not None else None
    is_ref = (ref.get('name') == rc.imager.name) if ref is not None else rc.primary
    if rc.sensor.kind == 'depth' and is_ref:
        types.append('distance_to_image_plane')
    return types, why


def _fps(project) -> dict:
    """{'<sensor uid>|<imager>': (fps, (w, h))} from the stream choice (core/studies/motion_blur.imaging)."""
    from ..studies import motion_blur as MB
    try:
        rows = MB.imaging(project)
    except StudyError:
        return {}
    return {r['key']: (r.get('fps'), tuple(r.get('stream') or ())) for r in rows}


def _tolerance(project, scale: float) -> tuple[dict, list]:
    from .. import tolerance as TL
    try:
        return TL.bounds(project, scale=scale), []
    except StudyError as e:
        return {}, [f'The randomisation bounds are left out: {e}']


def camera_records(project, options: dict | None = None) -> tuple[list, list]:
    """([record], notes): one dict per imager with everything the files hold (JSON-able)."""
    options = {**DEFAULTS, **(options or {})}
    q, qtext = C.joint_state(project, options)
    cams = C.rig_cameras(project, q)
    C.need_cameras(cams, 'The Isaac export')
    scale = float(options.get('scale') or 1.0)
    if not math.isfinite(scale) or scale < 0:
        raise StudyError('The tolerance scale must be a number of zero or more.')
    tol, notes = _tolerance(project, scale)
    fps = _fps(project)
    from ..robot.model import get_model
    out, names = [], set()
    for rc in cams:
        cam = rc.camera
        name = usd_name(rc.key)
        while name in names:
            name += '_'
        names.add(name)
        robot = project.robot(rc.robot_uid) if rc.robot_uid else None
        if robot is not None:
            m = get_model(project, robot.uid)
            T_world_parent = np.asarray(m.link_pose(q.get(robot.uid, m.home), rc.parent), float)
        else:
            T_world_parent = np.eye(4)
        T_po = rc.T_parent_optical
        lens = lens_model(cam)
        types, why = _data_types(rc)
        f, mode = fps.get(f'{rc.sensor.uid}|{rc.imager.name}', (None, ()))
        if f and mode and tuple(mode) != (cam.width, cam.height):
            f = None
        b = tol.get(rc.sensor.uid)
        pp = principal_point(cam)
        rec = {
            'name': name, 'sensor_uid': rc.sensor.uid, 'sensor': rc.sensor.name, 'imager': rc.imager.name,
            'role': rc.imager.role, 'primary': rc.primary, 'locked': bool(rc.sensor.locked),
            'parent': rc.parent, 'robot_uid': rc.robot_uid, 'robot': robot.name if robot is not None else None,
            'parent_prim': usd_name(robot.name) + '/' + prim_token(rc.parent) if robot is not None else 'World',
            'width': cam.width, 'height': cam.height, 'fx': cam.fx, 'fy': cam.fy, 'cx': cam.cx, 'cy': cam.cy,
            'lens_model': cam.model, 'dist': list(cam.dist), 'max_theta_rad': float(cam.max_theta),
            'lens_label': rc.imager.label, 'default_camera': bool(rc.imager.is_default),
            'T_parent_body': np.asarray(rc.T_parent_sensor, float).tolist(),
            'T_body_optical': np.asarray(rc.T_sensor_optical, float).tolist(),
            'T_parent_optical': np.asarray(T_po, float).tolist(),
            'T_world_parent': T_world_parent.tolist(),
            'T_world_optical': (T_world_parent @ T_po).tolist(),
            'offset_ros': {'pos': [float(v) for v in T_po[:3, 3]],
                           'rot_xyzw': [float(v) for v in quat_from_matrix(T_po[:3, :3])]},
            'usd': usd_lens(cam), 'distortion': lens['distortion'], 'lens_exact': lens['exact'],
            'lens_fit_error_px': _finite(lens['fit_error_px']), 'lens_text': lens['text'],
            'v2_pinhole_error_px': _finite(_v2_error(cam)), 'data_types': types, 'role_note': why,
            'principal_point': {**pp, 'v3': 'OpenCV lens attributes (omni:lensdistortion:<model>:cx, cy)'
                                if pp['off_centre'] else 'centred',
                                'v2': 'not representable: the v2.3.2 render is centred' if pp['off_centre']
                                else 'centred'},
            'fps': f, 'update_period_s': (1.0 / f) if f else 0.0,
            'tolerance': None if b is None else {'position_m': b['position_m'], 'rotation_deg': b['rotation_deg'],
                                                 'source': b['source'], 'meaning': b['meaning']},
        }
        out.append(rec)
    notes.append(f'Link-mounted cameras are shown at {qtext} in the USD file; their offsets from the link do not '
                 'depend on the pose.')
    return out, notes


# ====================================================================== USD
def _usd_matrix(T) -> list:
    """USD's row-vector matrix (translation in the last row) of a column-vector 4 × 4."""
    return np.asarray(T, float).T.tolist()


def _camera_local(rec) -> np.ndarray:
    T = np.asarray(rec['T_parent_optical'], float).copy()
    T[:3, :3] = T[:3, :3] @ OPTICAL_TO_USD
    return T


def _custom(rec) -> dict:
    """customData under the 'dexvantage' dictionary (pxr nests 'dexvantage:<key>' that way)."""
    d = {'sensor_uid': rec['sensor_uid'], 'sensor': rec['sensor'], 'imager': rec['imager'],
         'lens_model': rec['lens_model'], 'lens': rec['lens_text']}
    if rec['tolerance']:
        d['tolerance_position_m'] = json.dumps(rec['tolerance']['position_m'])
        d['tolerance_rotation_deg'] = json.dumps(rec['tolerance']['rotation_deg'])
        d['tolerance_source'] = rec['tolerance']['source']
    return dict(sorted(d.items()))


def _groups(records) -> list:
    """[(group path relative to the root, T_world_group, [records])]."""
    groups: dict = {}
    for r in records:
        key = r['parent_prim']
        groups.setdefault(key, (np.asarray(r['T_world_parent'], float), []))[1].append(r)
    return [(k, T, rs) for k, (T, rs) in groups.items()]


def write_usd_pxr(path: str, records, project_name: str):
    """Write the camera prims with pxr (usd-core)."""
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt  # noqa: F401
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, f'/{ROOT}')
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey('dexvantage:project', project_name)
    for rel, T, rs in _groups(records):
        parts = rel.split('/')
        for i in range(1, len(parts) + 1):
            UsdGeom.Xform.Define(stage, f'/{ROOT}/' + '/'.join(parts[:i]))
        g = UsdGeom.Xform.Get(stage, f'/{ROOT}/{rel}')
        if not np.allclose(T, np.eye(4)):
            g.AddTransformOp().Set(Gf.Matrix4d(_usd_matrix(T)))
        for r in rs:
            cam = UsdGeom.Camera.Define(stage, f'/{ROOT}/{rel}/{r["name"]}')
            cam.AddTransformOp().Set(Gf.Matrix4d(_usd_matrix(_camera_local(r))))
            u = r['usd']
            cam.CreateProjectionAttr(UsdGeom.Tokens.perspective)
            cam.CreateFocalLengthAttr(float(u['focalLength']))
            cam.CreateHorizontalApertureAttr(float(u['horizontalAperture']))
            cam.CreateVerticalApertureAttr(float(u['verticalAperture']))
            cam.CreateHorizontalApertureOffsetAttr(float(u['horizontalApertureOffset']))
            cam.CreateVerticalApertureOffsetAttr(float(u['verticalApertureOffset']))
            cam.CreateClippingRangeAttr(Gf.Vec2f(*CLIP_M))
            p = cam.GetPrim()
            p.CreateAttribute('cameraProjectionType', Sdf.ValueTypeNames.Token).Set('pinhole')
            dd = r['distortion']
            if dd is not None:
                mdl = dd['model']
                p.AddAppliedSchema(DISTORTION_API[mdl])
                p.CreateAttribute('omni:lensdistortion:model', Sdf.ValueTypeNames.Token).Set(mdl)
                pre = f'omni:lensdistortion:{mdl}'
                p.CreateAttribute(f'{pre}:imageSize', Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(*dd['image_size']))
                for k in ('fx', 'fy', 'cx', 'cy'):
                    p.CreateAttribute(f'{pre}:{k}', Sdf.ValueTypeNames.Float).Set(float(dd[k]))
                for k in COEFFS[mdl]:
                    p.CreateAttribute(f'{pre}:{k}', Sdf.ValueTypeNames.Float).Set(float(dd['coefficients'].get(k, 0.0)))
                if 'max_fov' in dd:
                    p.CreateAttribute(f'{pre}:maxFov', Sdf.ValueTypeNames.Float).Set(float(dd['max_fov']))
            for k, v in _custom(r).items():
                p.SetCustomDataByKey(f'dexvantage:{k}', v)
    stage.GetRootLayer().Save()


def _q(s: str) -> str:
    return json.dumps(str(s), ensure_ascii=False)


def _num(x) -> str:
    return C.fmt(x)


def _mat_text(T) -> str:
    rows = _usd_matrix(T)
    return '( ' + ', '.join('(' + ', '.join(_num(v) for v in row) + ')' for row in rows) + ' )'


def usda_text(records, project_name: str) -> str:
    """The same stage as write_usd_pxr, as USD text (the writer used when usd-core is not installed)."""
    L = ['#usda 1.0', '(', f'    defaultPrim = "{ROOT}"', '    metersPerUnit = 1', '    upAxis = "Z"', ')', '',
         f'def Xform "{ROOT}" (', '    customData = {', '        dictionary dexvantage = {',
         f'            string project = {_q(project_name)}', '        }', '    }', ')', '{']

    def emit_cam(r, ind):
        dd = r['distortion']
        meta = []
        if dd is not None:
            meta.append(f'{ind}    prepend apiSchemas = ["{DISTORTION_API[dd["model"]]}"]')
        cd = [f'{ind}            string {k} = {_q(v)}' for k, v in _custom(r).items()]
        meta.append(f'{ind}    customData = {{\n{ind}        dictionary dexvantage = {{\n' + '\n'.join(cd)
                    + f'\n{ind}        }}\n{ind}    }}')
        out = [f'{ind}def Camera "{r["name"]}" (', '\n'.join(meta), f'{ind})', f'{ind}{{']
        u = r['usd']
        a = ind + '    '
        out += [f'{a}token cameraProjectionType = "pinhole"',
                f'{a}float2 clippingRange = ({_num(CLIP_M[0])}, {_num(CLIP_M[1])})',
                f'{a}float focalLength = {_num(u["focalLength"])}',
                f'{a}float horizontalAperture = {_num(u["horizontalAperture"])}',
                f'{a}float horizontalApertureOffset = {_num(u["horizontalApertureOffset"])}']
        if dd is not None:
            mdl = dd['model']
            pre = f'omni:lensdistortion:{mdl}'
            vals = [(k, dd[k]) for k in ('cx', 'cy', 'fx', 'fy')] + \
                   [(k, dd['coefficients'].get(k, 0.0)) for k in COEFFS[mdl]]
            if 'max_fov' in dd:
                vals.append(('maxFov', dd['max_fov']))
            out += [f'{a}int2 {pre}:imageSize = ({dd["image_size"][0]}, {dd["image_size"][1]})']
            out += [f'{a}float {pre}:{k} = {_num(v)}' for k, v in vals]
            out += [f'{a}uniform token omni:lensdistortion:model = "{mdl}"']
        out += [f'{a}token projection = "perspective"',
                f'{a}float verticalAperture = {_num(u["verticalAperture"])}',
                f'{a}float verticalApertureOffset = {_num(u["verticalApertureOffset"])}',
                f'{a}matrix4d xformOp:transform = {_mat_text(_camera_local(r))}',
                f'{a}uniform token[] xformOpOrder = ["xformOp:transform"]', f'{ind}}}']
        return out

    tree: dict = {}
    for rel, T, rs in _groups(records):
        tree[rel] = (T, rs)
    # nest robot/link groups: 'World' or 'robot/link'
    robots: dict = {}
    for rel, (T, rs) in tree.items():
        parts = rel.split('/')
        robots.setdefault(parts[0], []).append((parts[1] if len(parts) > 1 else None, T, rs))
    for top, items in robots.items():
        L.append(f'    def Xform "{top}"')
        L.append('    {')
        for link, T, rs in items:
            if link is None:
                for r in rs:
                    L += emit_cam(r, '        ')
                continue
            L.append(f'        def Xform "{link}"')
            L.append('        {')
            if not np.allclose(T, np.eye(4)):
                L.append(f'            matrix4d xformOp:transform = {_mat_text(T)}')
                L.append('            uniform token[] xformOpOrder = ["xformOp:transform"]')
            for r in rs:
                L += emit_cam(r, '            ')
            L.append('        }')
        L.append('    }')
    L.append('}')
    return '\n'.join(L) + '\n'


def have_pxr() -> bool:
    try:
        import pxr.Usd  # noqa: F401
    except ImportError:
        return False
    return True


# ====================================================================== Isaac Lab configs (Python)
def _py(v) -> str:
    """A Python literal: floats as repr (round trip), tuples kept."""
    if isinstance(v, bool) or v is None:
        return repr(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return C.fmt(float(v)) if float(v) != 0 else '0.0'
    if isinstance(v, tuple):
        return '(' + ', '.join(_py(x) for x in v) + (',)' if len(v) == 1 else ')')
    if isinstance(v, list):
        return '[' + ', '.join(_py(x) for x in v) + ']'
    if isinstance(v, dict):
        return '{' + ', '.join(f'{_py(k)}: {_py(x)}' for k, x in v.items()) + '}'
    return repr(v)


def _var(name: str) -> str:
    return name.upper()


_HELPERS = '''

# ---------------------------------------------------------------------------- domain randomisation (spec §8.5)
# core/tolerance.bounds: shifts along the mount parent frame's x, y, z (m); turns about the sensor body's own x, y, z
# (degrees), applied as R' = R · Rx · Ry · Rz. A draw moves the sensor body; the imager keeps its place in the body.


def _mat_mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def _rot(axis, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    if axis == 0:
        return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
    if axis == 1:
        return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def _quat_xyzw(R):
    tr = R[0][0] + R[1][1] + R[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [(R[2][1] - R[1][2]) / s, (R[0][2] - R[2][0]) / s, (R[1][0] - R[0][1]) / s, 0.25 * s]
    elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2
        q = [0.25 * s, (R[0][1] + R[1][0]) / s, (R[0][2] + R[2][0]) / s, (R[2][1] - R[1][2]) / s]
    elif R[1][1] > R[2][2]:
        s = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2
        q = [(R[0][1] + R[1][0]) / s, 0.25 * s, (R[1][2] + R[2][1]) / s, (R[0][2] - R[2][0]) / s]
    else:
        s = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2
        q = [(R[0][2] + R[2][0]) / s, (R[1][2] + R[2][1]) / s, 0.25 * s, (R[1][0] - R[0][1]) / s]
    n = math.sqrt(sum(x * x for x in q))
    q = [x / n for x in q]
    return q if q[3] >= 0 else [-x for x in q]


def perturbed_offset(name, d):
    """The camera's (pos, rot) offset from its parent when its sensor body moves by d = (dx, dy, dz) m along the
    parent frame's axes and turns by (droll, dpitch, dyaw) degrees about its own x, y, z (R' = R Rx Ry Rz)."""
    c = DR[name]
    Tb, To = c["T_parent_body"], c["T_body_optical"]
    R = [row[:3] for row in Tb[:3]]
    R = _mat_mul(_mat_mul(_mat_mul(R, _rot(0, d[3])), _rot(1, d[4])), _rot(2, d[5]))
    Tb2 = [R[i] + [Tb[i][3] + d[i]] for i in range(3)] + [[0.0, 0.0, 0.0, 1.0]]
    T = _mat_mul(Tb2, To)
    pos = (T[0][3], T[1][3], T[2][3])
    q = _quat_xyzw([row[:3] for row in T[:3]])
    return pos, (tuple(q) if QUAT_ORDER == "xyzw" else (q[3], q[0], q[1], q[2]))


def sample_offset(name, rng=None):
    """One uniform draw within the camera's bounds: (pos, rot, d). rng: a random.Random (default a new one)."""
    rng = rng or random.Random()
    b = DR[name]["bounds"]
    if b is None:
        return CAMERAS[name].offset.pos, CAMERAS[name].offset.rot, (0.0,) * 6
    d = [rng.uniform(lo, hi) for lo, hi in b["position_m"]] + [rng.uniform(lo, hi) for lo, hi in b["rotation_deg"]]
    pos, rot = perturbed_offset(name, d)
    return pos, rot, tuple(d)


def randomized_cameras(seed=None):
    """Every camera config with one draw of its mount within the bounds (one draw per call: per training run)."""
    rng = random.Random(seed)
    out = {}
    for name, cfg in CAMERAS.items():
        pos, rot, _ = sample_offset(name, rng)
        out[name] = cfg.replace(offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="ros"))
    return out
'''


def _header(project, v3: bool, env_ns: str) -> str:
    tag, commit = (V3_TAG, V3_COMMIT) if v3 else (V2_TAG, V2_COMMIT)
    lines = [
        f'"""Isaac Lab {tag} camera sensors of the DexVantage project {project.name!r}.',
        '',
        f'Written by DexVantage against the Isaac Lab source at tag {tag} (commit {commit},',
        f'{SOURCE_URL}, read {SOURCE_READ}). Each camera is one imager of a sensor of the active rig, its offset in',
        'the ROS convention (forward +Z, up -Y) from its parent prim: the robot link it is mounted on, or the',
        'environment for a static mount. Units: metres; the lens follows Isaac Lab\'s from_intrinsic_matrix rule',
        '(pixel size 1 / width: horizontal_aperture 1, focal_length fx / width).',
    ]
    if v3:
        lines += ['Quaternions are (x, y, z, w) (CameraCfg.OffsetCfg at v3.0.0). TiledCameraCfg is deprecated at this '
                  'tag', '("Use CameraCfg directly"), so these are CameraCfg. A lens that is not a square, centred',
                  'pinhole carries an OpenCV lens model, whose fx, fy, cx, cy Isaac Lab then uses.']
    else:
        lines += ['Quaternions are (w, x, y, z) (CameraCfg.OffsetCfg at v2.3.2). This tag has no OpenCV distortion',
                  'fields, so each camera is the square, centred pinhole of its lens (focal length from fx); see',
                  'isaac_cameras.json for the lens each one approximates and by how much.']
    lines += ['', 'Use: add the configs to your InteractiveSceneCfg, e.g. scene.wrist = CAMERAS["<name>"].',
              'DR holds each camera\'s mount and its randomisation bounds (core/tolerance.bounds).', '"""', '',
              'import math', 'import random', '',
              'import isaaclab.sim as sim_utils',
              'from isaaclab.sensors import CameraCfg' if v3 else 'from isaaclab.sensors import TiledCameraCfg',
              '' if v3 else 'CameraCfg = TiledCameraCfg   # the same fields at v2.3.2 (TiledCameraCfg subclasses it)',
              '', f'QUAT_ORDER = "{"xyzw" if v3 else "wxyz"}"',
              f'ENV_NS = {_py(env_ns)}', '']
    return '\n'.join(lines)


def _distortion_py(dd) -> str:
    cls = 'OpenCvPinholeDistortionCfg' if dd['model'] == 'opencvPinhole' else 'OpenCvFisheyeDistortionCfg'
    kw = [f'fx={_py(dd["fx"])}', f'fy={_py(dd["fy"])}', f'cx={_py(dd["cx"])}', f'cy={_py(dd["cy"])}',
          f'image_size={_py(tuple(dd["image_size"]))}']
    kw += [f'{k}={_py(float(v))}' for k, v in dd['coefficients'].items()]
    if 'max_fov' in dd:
        kw.append(f'max_fov={_py(dd["max_fov"])}')
    return f'sim_utils.{cls}(' + ', '.join(kw) + ')'


def config_text(project, records, options: dict, v3: bool = True) -> str:
    """The Isaac Lab Python config file for v3.0.0 (v3) or v2.3.2."""
    o = {**DEFAULTS, **(options or {})}
    env_ns = str(o.get('env_ns') or DEFAULTS['env_ns'])
    robot_prim = prim_token(o.get('robot_prim') or DEFAULTS['robot_prim'])
    out = [_header(project, v3, env_ns)]
    names = []
    for r in records:
        var = _var(r['name'])
        names.append((r['name'], var))
        if r['robot_uid']:
            path = f'ENV_NS + {_py("/" + robot_prim + "/" + prim_token(r["parent"]) + "/" + r["name"])}'
        else:
            path = f'ENV_NS + {_py("/" + r["name"])}'
        pos = tuple(r['offset_ros']['pos'])
        qx = r['offset_ros']['rot_xyzw']
        rot = tuple(qx) if v3 else (qx[3], qx[0], qx[1], qx[2])
        u = r['usd']
        spawn = [f'focal_length={_py(u["focalLength"])}', f'horizontal_aperture={_py(u["horizontalAperture"])}',
                 f'vertical_aperture={_py(u["verticalAperture"])}', f'clipping_range={_py(CLIP_M)}']
        if v3 and r['distortion'] is not None:
            spawn.append(f'distortion={_distortion_py(r["distortion"])}')
        if not v3:
            spawn[2] = f'vertical_aperture={_py(r["height"] / r["width"])}'
        why = r['lens_text'] if v3 else (
            'pinhole part only (square, centred); ' + (f'{r["v2_pinhole_error_px"]:.3g} px from the lens'
                                                        if r['v2_pinhole_error_px'] is not None
                                                        else 'the lens sees rays a pinhole cannot'))
        if r['principal_point']['off_centre']:
            dx, dy = r['principal_point']['offset_px']
            why += (f'; principal point {dx:+.3g}, {dy:+.3g} px off centre: carried by the OpenCV lens model' if v3
                    else f'; its principal point ({dx:+.3g}, {dy:+.3g} px off centre) cannot be shown at v2.3.2')
        out.append(f'# {r["sensor"]} · {r["imager"]} ({r["role"]}): {why}')
        if r['fps'] is None:
            out.append('# update_period 0.0: every step (the stream\'s frame rate is not known)')
        out.append(f'{var} = CameraCfg(\n    prim_path={path},\n    update_period={_py(r["update_period_s"])},\n'
                   f'    width={r["width"]},\n    height={r["height"]},\n    data_types={_py(r["data_types"])},\n'
                   f'    offset=CameraCfg.OffsetCfg(pos={_py(pos)}, rot={_py(rot)}, convention="ros"),\n'
                   f'    spawn=sim_utils.PinholeCameraCfg(\n        ' + ',\n        '.join(spawn) + ',\n    ),\n)\n')
    out.append('CAMERAS = {' + ', '.join(f'{_py(n)}: {v}' for n, v in names) + '}\n')
    dr = {}
    for r in records:
        dr[r['name']] = {'sensor': r['sensor'], 'imager': r['imager'], 'parent': r['parent'],
                         'T_parent_body': [[float(x) for x in row] for row in r['T_parent_body']],
                         'T_body_optical': [[float(x) for x in row] for row in r['T_body_optical']],
                         'bounds': None if r['tolerance'] is None else
                         {'position_m': [tuple(x) for x in r['tolerance']['position_m']],
                          'rotation_deg': [tuple(x) for x in r['tolerance']['rotation_deg']],
                          'source': r['tolerance']['source'], 'meaning': r['tolerance']['meaning']}}
    out.append('DR = {\n' + ',\n'.join(f'    {_py(k)}: {_py(v)}' for k, v in dr.items()) + ',\n}')
    out.append(_HELPERS)
    return '\n'.join(out)


# ====================================================================== the exporter
@exporter('isaac', 'Isaac Lab and Isaac Sim cameras (USD prims, camera configs, randomisation bounds)',
          needs_robot=False,
          description='USD camera prims, Isaac Lab v3.0.0 CameraCfg and v2.3.2 TiledCameraCfg Python files, and the '
                      'mount tolerance as domain-randomisation bounds.')
def export_isaac(project, out_dir: str, options: dict, ctx) -> dict:
    o = {**DEFAULTS, **(options or {})}
    if o.get('usd') not in ('auto', 'text'):
        raise StudyError('The USD writer must be "auto" or "text".')
    records, notes = camera_records(project, o)
    files = []
    use_pxr = o['usd'] == 'auto' and have_pxr()
    path = os.path.join(out_dir, USD_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if use_pxr:
        if os.path.exists(path):
            os.remove(path)
        try:
            write_usd_pxr(path, records, project.name)
        except Exception as e:                              # noqa: BLE001 — explained, the text writer instead
            notes.append(f'pxr could not write the USD file ({type(e).__name__}: {str(e).splitlines()[0][:160]}); '
                         'it is written as text instead.')
            use_pxr = False
    if not use_pxr:
        C.write_text(out_dir, USD_FILE, usda_text(records, project.name))
        if o['usd'] == 'auto' and not have_pxr():
            notes.append('usd-core is not installed, so cameras.usda is written as text by DexVantage (the same '
                         'prims and attributes); install the usd extra (pip install "dexvantage[usd]") to write it '
                         'with pxr and to run the USD round-trip check.')
    else:
        notes.append('cameras.usda is written with pxr (usd-core).')
    files.append(USD_FILE)
    files.append(C.write_text(out_dir, V3_FILE, config_text(project, records, o, v3=True)))
    files.append(C.write_text(out_dir, V2_FILE, config_text(project, records, o, v3=False)))
    for r in records:
        if r['principal_point']['off_centre']:
            notes.append(principal_point_text(f'{r["sensor"]} · {r["imager"]}', r['principal_point']))
        if not r['lens_exact']:
            notes.append(f'{r["sensor"]} · {r["imager"]}: {r["lens_text"]}.')
        if r['distortion'] is not None and r['lens_model'] != 'pinhole':
            e = r['v2_pinhole_error_px']
            notes.append(f'{r["sensor"]} · {r["imager"]}: Isaac Lab v2.3.2 has no OpenCV lens fields, so its file '
                         'has the pinhole part only (' + (f'{e:.3g} px from the lens at worst)' if e is not None
                                                          else 'the lens sees rays a pinhole cannot)') + '.')
        if r['role_note']:
            notes.append(r['role_note'] + '.')
        if r['tolerance'] is None:
            notes.append(f'{r["sensor"]}: no randomisation bounds (the tolerance study does not cover it).')
    if any(r['tolerance'] and r['tolerance']['source'] == 'mount tolerance' for r in records):
        notes.append('Randomisation bounds are the mount tolerances (Mount.tolerance; ± 1 mm and ± 0.5° unless you '
                     'set your own: those are DexVantage\'s labelled defaults).')
    doc = {'schema': SCHEMA, 'project': project.name,
           'source': {'url': SOURCE_URL, 'v3': {'tag': V3_TAG, 'commit': V3_COMMIT},
                      'v2': {'tag': V2_TAG, 'commit': V2_COMMIT}, 'read': SOURCE_READ},
           'conventions': {'offset': 'CameraCfg.OffsetCfg convention "ros" (forward +Z, up -Y) from the parent prim',
                           'quaternion_v3': 'xyzw', 'quaternion_v2': 'wxyz',
                           'usd_camera': 'looks along -Z with +Y up: optical frame · diag(1, -1, -1)',
                           'usd_lens': 'pixel size 1 / width (Isaac Lab utils/sensors.py)',
                           'principal_point': 'off-centre cx, cy: the OpenCV lens attributes at v3.0.0, not '
                                              'representable at v2.3.2 (' + PP_SOURCE['v3'] + '; ' + PP_SOURCE['v2']
                                              + ')',
                           'clipping_range_m': list(CLIP_M), 'clipping_range_is_default': True,
                           'tolerance': 'core/tolerance.bounds: shifts along the parent axes (m), turns about the '
                                        'body axes (deg), R\' = R Rx Ry Rz'},
           'usd_writer': 'pxr' if use_pxr else 'text', 'cameras': records, 'notes': notes}
    files.append(C.write_json(out_dir, JSON_FILE, doc))
    return {'files': files, 'notes': notes}
