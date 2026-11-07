"""Calibration export (spec §9.2): OpenCV YAML per imager (written by cv2.FileStorage), a Kalibr camchain.yaml, and
JSON intrinsics and extrinsics of every camera.

    @exporter('opencv_yaml', ...)   options: q ({robot uid: joints}; default home), imagers ('all' | 'primary')

Files:
- calibration/opencv/<sensor>__<imager>.yaml: image_width, image_height, camera_name, camera_matrix (3 × 3),
  distortion_model (ROS's names: plumb_bob, rational_polynomial, or fisheye for cv2.fisheye), distortion_coefficients
  (1 × N), opencv_functions ('calib3d' or 'fisheye'), lens_model (DexVantage's), exact (1 when the OpenCV form is the
  lens itself), approximation_error_px (over the pixels it maps; −1 when it maps none), unmapped_share (the share
  of the image's pixels whose rays it cannot map), parent_frame, T_parent_camera (4 × 4: the optical frame in the
  parent link or world, metres), T_world_camera (4 × 4 at the export's joint state).
- calibration/kalibr/camchain.yaml: cam0, cam1, … with camera_model, intrinsics [fu, fv, pu, pv], distortion_model,
  distortion_coeffs, resolution [w, h], rostopic and, from cam1 on, T_cn_cnm1 (Kalibr: "takes cam0 to cam1
  coordinates" for cam1, T_c1_c0) [Kalibr wiki "YAML formats" and "Supported models", and
  kalibr_common/ConfigReader.py at commit 1f60227, read 2026-11-07].
- calibration/cameras.json: every imager's intrinsics (CameraModel.to_dict, OpenCV's K and D), its frames and its
  extrinsics (T_parent_sensor, T_sensor_optical, T_parent_optical, T_world_optical), with the lens mapping of each
  target.

Lens mapping (lens_for): OpenCV takes pinhole and Brown-Conrady with 5 terms as plumb_bob and with 8 as
rational_polynomial (cv2.projectPoints), Kannala-Brandt as fisheye (cv2.fisheye), all exact; the equidistant fisheye
is Kannala-Brandt with zero terms, exact; the equisolid and stereographic fisheyes have no OpenCV form and are written
as the least-squares Kannala-Brandt curve (CameraModel.kb_approximation). Kalibr's pinhole camera takes 'none',
'radtan' [k1 k2 r1 r2] (OpenCV's k1, k2, p1, p2; aslam_cv RadialTangentialDistortion) and 'equidistant' [k1 k2 k3 k4]
(θd = θ (1 + k1 θ² + k2 θ⁴ + k3 θ⁶ + k4 θ⁸) with θ = atan(r), aslam_cv EquidistantDistortion: OpenCV's fisheye). A
Brown-Conrady lens with k3 or the rational terms has no radtan form: its radtan terms are fitted by linear least
squares to the lens's distorted coordinates over the image; the radial fisheyes other than Kannala-Brandt go to
equidistant by kb_approximation. Every approximation is said in the notes with its error, measured as the largest
distance between where the lens and the written model put the same ray over a 41 × 41 grid of the image's pixels.
Kalibr's and OpenCV's fisheye compute θ from X/Z, so they map only rays in front of the camera (θ < 90°): a lens
whose image reaches beyond that is said to have no exact equivalent there.

Pixel convention: DexVantage, OpenCV and ROS put pixel centres at integers; the principal point is written as
DexVantage holds it. Kalibr's projection is u = fu·x + pu (aslam_cv PinholeProjection); its documentation does not
state a pixel-centre convention, so the same numbers are written.

Known limits: one joint state per export, so a chain between a camera on a link and another camera
(T_cn_cnm1) holds at that pose only (the notes say which cameras move); no camera-IMU transforms (DexVantage models
no IMUs); Kalibr's omni, ds and eucm models (which could hold a fisheye beyond 180°) are not fitted.
"""
from __future__ import annotations

import os

import numpy as np

from ..jobs import StudyError
from . import _common as C
from . import exporter

OPENCV_DIR = 'calibration/opencv'
KALIBR = 'calibration/kalibr/camchain.yaml'
CAMERAS_JSON = 'calibration/cameras.json'
FIT_GRID = 41


def _radtan_fit(cam) -> tuple:
    """(k1, k2, p1, p2) fitted by linear least squares to the lens's distorted normalised coordinates at the rays of
    a FIT_GRID × FIT_GRID grid of its pixels."""
    u, v = np.meshgrid(np.linspace(-0.5, cam.width - 0.5, FIT_GRID), np.linspace(-0.5, cam.height - 0.5, FIT_GRID))
    rays, ok = cam.unproject(np.stack([u.ravel(), v.ravel()], 1), with_ok=True)
    rays, uu, vv = rays[ok], u.ravel()[ok], v.ravel()[ok]
    x, y = rays[:, 0] / rays[:, 2], rays[:, 1] / rays[:, 2]
    xd, yd = (uu - cam.cx) / cam.fx, (vv - cam.cy) / cam.fy
    r2 = x * x + y * y
    A = np.concatenate([np.stack([x * r2, x * r2 * r2, 2 * x * y, r2 + 2 * x * x], 1),
                        np.stack([y * r2, y * r2 * r2, r2 + 2 * y * y, 2 * x * y], 1)])
    b = np.concatenate([xd - x, yd - y])
    return tuple(float(t) for t in np.linalg.lstsq(A, b, rcond=None)[0])


def lens_for(cam, target: str) -> dict:
    """How a lens is written for target ('opencv' | 'kalibr'): {'model', 'coeffs', 'exact', 'error_px', 'note',
    'functions' (opencv)}."""
    from ..optics.lens import MODEL_LABEL, CameraModel
    m, dist = cam.model, tuple(cam.dist)
    label = MODEL_LABEL[m]
    front = ''
    if m in ('kb', 'equidistant', 'equisolid', 'stereographic'):
        fov = cam.fov_deg()
        if max(fov.get('d', 0.0), fov.get('h', 0.0)) >= 180.0:
            front = (f' Its image reaches {max(fov.get("d", 0.0), fov.get("h", 0.0)):.1f}° across, but the target '
                     'computes θ from X/Z, so rays 90° or more off the axis have no equivalent there.')

    def out(model, coeffs, approx_model, approx_dist, exact, why='', functions=''):
        approx = CameraModel(cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy, approx_model, approx_dist)
        err, lost = (0.0, 0.0) if exact else C.approx_error_px(cam, approx)
        size = f'largest error {err:.3g} px over the image' if np.isfinite(err) else 'no ray of the image maps'
        if lost > 0:
            size += (f'; {100 * lost:.1f} % of the image\'s pixels have rays the written model cannot map (it folds '
                     'back or stops before the image edge)')
        note = '' if exact and not front else (
            f'{label} has no exact {"OpenCV" if target == "opencv" else "Kalibr"} form: written as {model} {why}'
            f'({size}).' if not exact else front.strip())
        if not exact and front:
            note += front
        return {'model': model, 'coeffs': [float(c) for c in coeffs], 'exact': bool(exact and not front),
                'unmapped_share': float(lost),
                'error_px': float(err) if np.isfinite(err) else None, 'note': note, 'functions': functions}

    if target == 'opencv':
        if m == 'pinhole':
            return out('plumb_bob', (0.0,) * 5, 'pinhole', (), True, functions='calib3d')
        if m == 'brown':
            return out('plumb_bob' if len(dist) == 5 else 'rational_polynomial', dist, 'brown', dist, True,
                       functions='calib3d')
        if m == 'kb':
            return out('fisheye', dist, 'kb', dist, True, functions='fisheye')
        if m == 'equidistant':
            return out('fisheye', (0.0,) * 4, 'kb', (0.0,) * 4, True, functions='fisheye')
        k = cam.kb_approximation()[0]
        return out('fisheye', k, 'kb', k, False, '(the least-squares Kannala-Brandt curve) ', 'fisheye')
    if target == 'kalibr':
        if m == 'pinhole':
            return out('none', (), 'pinhole', (), True)
        if m == 'brown':
            if len(dist) == 5 and dist[4] == 0.0:
                return out('radtan', dist[:4], 'brown', dist[:4] + (0.0,), True)
            k = _radtan_fit(cam)
            return out('radtan', k, 'brown', k + (0.0,), False,
                       '(k1, k2, p1, p2 fitted by least squares: radtan has no k3 or rational terms) ')
        if m == 'kb':
            return out('equidistant', dist, 'kb', dist, True)
        if m == 'equidistant':
            return out('equidistant', (0.0,) * 4, 'kb', (0.0,) * 4, True)
        k = cam.kb_approximation()[0]
        return out('equidistant', k, 'kb', k, False, '(the least-squares Kannala-Brandt curve) ')
    raise ValueError(target)


def write_opencv(path: str, c, lens: dict, q_text: str):
    """One imager's OpenCV YAML through cv2.FileStorage."""
    import cv2
    K = np.array([[c.camera.fx, 0.0, c.camera.cx], [0.0, c.camera.fy, c.camera.cy], [0.0, 0.0, 1.0]])
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise StudyError(f'Could not write {os.path.basename(path)}.')
    try:
        fs.writeComment(f'DexVantage camera {c.title}: OpenCV intrinsics (pixel centres at integers) and extrinsics '
                        '(metres; T_a_b maps b coordinates to a).')
        fs.write('image_width', int(c.camera.width))
        fs.write('image_height', int(c.camera.height))
        fs.write('camera_name', c.key)
        fs.write('camera_matrix', K)
        fs.write('distortion_model', lens['model'])
        fs.write('distortion_coefficients', np.asarray(lens['coeffs'], float).reshape(1, -1))
        fs.write('opencv_functions', lens['functions'])
        fs.write('lens_model', c.camera.model)
        fs.write('exact', int(lens['exact']))
        fs.write('approximation_error_px', -1.0 if lens['error_px'] is None else float(lens['error_px']))
        fs.write('unmapped_share', float(lens['unmapped_share']))
        fs.write('parent_frame', c.parent)
        fs.write('camera_frame', c.optical_frame)
        fs.write('T_parent_camera', np.asarray(c.T_parent_optical, float))
        fs.write('T_world_camera', np.asarray(c.T_world_optical, float))
        fs.write('joint_state', q_text)
    finally:
        fs.release()


def _mat(T) -> list:
    return [[float(v) for v in row] for row in np.asarray(T, float)]


def kalibr_text(cams: list, lenses: list, project, q_text: str) -> str:
    def vec(v):
        return '[' + ', '.join(C.fmt(x) for x in v) + ']'

    lines = [f'# Kalibr camera chain of the DexVantage rig "{project.name}" ({project.variant.name}).',
             '# T_cn_cnm1 takes camera n-1 coordinates to camera n coordinates (Kalibr), metres, at '
             f'{q_text}.', '# rostopic names are suggestions: replace them with your driver\'s topics.']
    for n, (c, L) in enumerate(zip(cams, lenses)):
        cm = c.camera
        lines += [f'cam{n}:', '  camera_model: pinhole', f'  intrinsics: {vec([cm.fx, cm.fy, cm.cx, cm.cy])}',
                  f'  distortion_model: {L["model"]}', f'  distortion_coeffs: {vec(L["coeffs"])}',
                  f'  resolution: [{cm.width}, {cm.height}]',
                  f'  rostopic: /{C.frame_name(c.sensor.name)}/{c.key.split("__", 1)[1]}/image_raw']
        if n:
            T = np.linalg.inv(c.T_world_optical) @ cams[n - 1].T_world_optical
            lines.append('  T_cn_cnm1:')
            lines += [f'  - {vec(row)}' for row in T]
    return '\n'.join(lines) + '\n'


@exporter('opencv_yaml', 'OpenCV YAML, Kalibr camchain and JSON intrinsics', needs_robot=False,
          description='Per camera an OpenCV YAML (cv2.FileStorage), a Kalibr camchain.yaml, and JSON intrinsics and '
                      'extrinsics; each lens mapped to the target\'s model, saying when there is no exact equivalent.')
def export_opencv_yaml(project, out_dir, options, ctx):
    imagers = str(options.get('imagers') or 'all')
    if imagers not in ('all', 'primary'):
        raise StudyError('The imagers option must be "all" or "primary".')
    q, q_text = C.joint_state(project, options)
    cams = C.rig_cameras(project, q, imagers)
    C.need_cameras(cams, 'The calibration export')
    files, notes, docs, kal = [], [], [], []
    for k, c in enumerate(cams):
        ctx.report(k / len(cams), c.title)
        cv = lens_for(c.camera, 'opencv')
        kb = lens_for(c.camera, 'kalibr')
        kal.append(kb)
        rel = f'{OPENCV_DIR}/{c.key}.yaml'
        os.makedirs(os.path.join(out_dir, OPENCV_DIR), exist_ok=True)
        write_opencv(os.path.join(out_dir, rel), c, cv, q_text)
        files.append(rel)
        for t, L in (('OpenCV', cv), ('Kalibr', kb)):
            if L['note']:
                notes.append(f'{c.title} ({t}): {L["note"]}')
        K, D, model = c.camera.to_opencv()
        docs.append({'name': c.key, 'sensor': c.sensor.name, 'sensor_uid': c.sensor.uid, 'imager': c.imager.name,
                     'role': c.imager.role, 'primary': c.primary, 'label': c.imager.label,
                     'intrinsics': c.camera.to_dict(),
                     'opencv': {'K': _mat(K), 'D': [float(v) for v in D], 'model': model, 'mapping': cv},
                     'kalibr': kb,
                     'frames': {'parent': c.parent, 'robot_uid': c.robot_uid, 'sensor': c.sensor_frame,
                                'optical': c.optical_frame},
                     'extrinsics': {'T_parent_sensor': C.pose_record(c.T_parent_sensor),
                                    'T_sensor_optical': C.pose_record(c.T_sensor_optical),
                                    'T_parent_optical': C.pose_record(c.T_parent_optical),
                                    'T_world_optical': C.pose_record(c.T_world_optical)}})
    files.append(C.write_text(out_dir, KALIBR, kalibr_text(cams, kal, project, q_text)))
    moving = [c.title for c in cams if c.robot_uid]
    if moving and len(cams) > 1:
        notes.append(f'The Kalibr chain holds at {q_text} only: {", ".join(moving)} move with a robot link.')
    doc = {'schema': 'dexvantage.cameras/1',
           'convention': 'Pixel centres at integers (OpenCV). Optical frame: z forward, x right, y down (REP-103). '
                         'T_a_b maps b coordinates to a; metres; quaternions (x, y, z, w).',
           'world': 'the primary robot\'s base (REP-103: x forward, y left, z up)', 'joint_state': q_text,
           'cameras': docs}
    files.append(C.write_json(out_dir, CAMERAS_JSON, doc))
    return {'files': files, 'notes': list(dict.fromkeys(notes))}
