"""The 3D Gaussian Splatting bridge: a capture (posed images of the robot cell, core/sim/splat_rig.py) as the input
of the common 3DGS trainers (Nerfstudio's transforms.json, a COLMAP text model), their trained .ply read back, and
whether gsplat runs here.

Ported from Vantage core/gsplat_bridge.py: the camera-parameter conversion, the OpenGL camera-to-world and OpenCV
world-to-camera matrices, the Nerfstudio and COLMAP writers. Stripped (automotive): the vehicle frame, the ego
trajectories, the rig timing (rolling shutter, lidar sweeps), NCore, 3DGRUT's lidar input and HUGSIM. Adapted: the
poses come straight from DexVantage's T_world_optical (the world is the primary robot's base frame, REP-103, z up),
the camera models are core/optics/lens.py's.

Formats, as Vantage's port read them from their sources (2026-11-04) and checked again here (2026-11-08,
nerfstudio/data/dataparsers/nerfstudio_dataparser.py at main: per-frame fl_x, fl_y, cx, cy, w, h and k1, k2, k3, k4,
p1, p2 are read, missing terms are 0, camera_model is global):

- **Nerfstudio** ``transforms.json``: ``camera_model`` is global (OPENCV: Brown-Conrady k1, k2, k3, p1, p2;
  OPENCV_FISHEYE: Kannala-Brandt k1..k4); intrinsics per frame; ``transform_matrix`` is camera-to-world with OpenGL
  camera axes (+X right, +Y up, +Z back): c2w = [R_world_optical · diag(1, −1, −1) | C]. Nerfstudio puts pixel (0, 0)'s
  ray through its centre at (0.5, 0.5), so cx = OpenCV cx + ½ (likewise cy).
- **COLMAP** text model (doc/format.rst, doc/cameras.rst): ``cameras.txt`` (PINHOLE fx fy cx cy; OPENCV fx fy cx cy
  k1 k2 p1 p2; FULL_OPENCV fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6; OPENCV_FISHEYE fx fy cx cy k1 k2 k3 k4),
  ``images.txt`` with the world-to-camera pose (QW QX QY QZ TX TY TZ, OpenCV camera axes), ``points3D.txt``; pixel
  coordinates are corner-based (the top-left pixel's centre at (0.5, 0.5)): cx = OpenCV cx + ½.

Lens models without an exact form there: the equisolid and stereographic fisheyes are written as their least-squares
Kannala-Brandt approximation (CameraModel.kb_approximation), the largest error in pixels written with the camera.

Known limits: the images are written as PNG at the capture's resolution; the poses are the capture's (known from the
rig and the robot's joint states), not refined; a capture mixing perspective and fisheye lenses cannot be one
Nerfstudio dataset (its camera model is global) and is refused with a message.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .frames import quat_from_matrix

SCHEMA = 'dexvantage.gsplat_bridge/1'
GL_FROM_CV = np.diag([1.0, -1.0, -1.0])           # OpenCV camera axes -> OpenGL camera axes (and back)
PERSPECTIVE = ('pinhole', 'brown')


def gsplat_version() -> str | None:
    """gsplat's version when it imports here, else None."""
    try:
        import gsplat
        return str(getattr(gsplat, '__version__', '') or 'unknown')
    except Exception:                                    # noqa: BLE001 — a broken build is "not here"
        return None


# ---------------------------------------------------------------- cameras
def camera_params(cam) -> dict:
    """A CameraModel's parameters for the trainers (corner-based pixels): fx, fy, cx, cy, w, h, k1..k4, p1, p2, the
    COLMAP model and its parameter list, the Nerfstudio family ('OPENCV' or 'OPENCV_FISHEYE') and the approximation's
    largest error in pixels (0 when exact)."""
    fx, fy, cx, cy = float(cam.fx), float(cam.fy), float(cam.cx) + 0.5, float(cam.cy) + 0.5
    d = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'w': int(cam.width), 'h': int(cam.height), 'k1': 0.0, 'k2': 0.0,
         'k3': 0.0, 'k4': 0.0, 'p1': 0.0, 'p2': 0.0, 'approx_px': 0.0, 'model': cam.model}
    if cam.model == 'pinhole':
        d.update(colmap='PINHOLE', colmap_params=[fx, fy, cx, cy], nerfstudio='OPENCV')
    elif cam.model == 'brown':
        k1, k2, p1, p2, k3 = cam.dist[:5]
        rational = cam.dist[5:] if len(cam.dist) > 5 else (0.0, 0.0, 0.0)
        d.update(k1=k1, k2=k2, k3=k3, p1=p1, p2=p2, nerfstudio='OPENCV')
        if k3 == 0.0 and not any(rational):
            d.update(colmap='OPENCV', colmap_params=[fx, fy, cx, cy, k1, k2, p1, p2])
        else:
            d.update(colmap='FULL_OPENCV', colmap_params=[fx, fy, cx, cy, k1, k2, p1, p2, k3, *rational])
            if any(rational):
                d['nerfstudio'] = None                    # Nerfstudio's OPENCV has no rational terms
    else:
        k, err = cam.kb_approximation()
        d.update(k1=k[0], k2=k[1], k3=k[2], k4=k[3], approx_px=float(err), nerfstudio='OPENCV_FISHEYE',
                 colmap='OPENCV_FISHEYE', colmap_params=[fx, fy, cx, cy, *k])
    return d


def c2w_opengl(T_world_optical) -> np.ndarray:
    """Nerfstudio's 4 × 4 camera-to-world (OpenGL camera axes) of a camera at T_world_optical."""
    T = np.array(T_world_optical.matrix() if hasattr(T_world_optical, 'matrix') else T_world_optical, float)
    T[:3, :3] = T[:3, :3] @ GL_FROM_CV
    return T


def w2c_opencv(T_world_optical):
    """COLMAP's world-to-camera (R (3, 3), t (3,)) in OpenCV camera axes."""
    T = np.asarray(T_world_optical.matrix() if hasattr(T_world_optical, 'matrix') else T_world_optical, float)
    R = T[:3, :3].T
    return R, -R @ T[:3, 3]


# ---------------------------------------------------------------- writers
def _write_images(capture, folder) -> list:
    import cv2
    os.makedirs(folder, exist_ok=True)
    names = []
    for k, v in enumerate(capture.views):
        name = f'{k:05d}.png'
        rgb = np.asarray(v.rgb)
        if rgb.dtype != np.uint8:
            rgb = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
        if not cv2.imwrite(os.path.join(folder, name), rgb[..., ::-1]):
            raise OSError(f'the image {name} could not be written to {folder}.')
        names.append(name)
    return names


def export_nerfstudio(capture, out_dir, write_images: bool = True) -> list[str]:
    """nerfstudio/transforms.json (+ images/ and README.md) in out_dir for a capture: one frame per view, per-frame
    intrinsics, the capture's synthetic marking and provenance under 'dexvantage'."""
    pars = [camera_params(v.camera) for v in capture.views]
    fam = {p['nerfstudio'] for p in pars}
    if None in fam:
        raise ValueError('a Brown-Conrady lens with rational terms (k4..k6) has no Nerfstudio form; export the COLMAP '
                         'model instead')
    if len(fam) > 1:
        raise ValueError('the capture mixes perspective and fisheye lenses, and a Nerfstudio dataset has one camera '
                         'model: export the views of each kind separately')
    model = fam.pop() if fam else 'OPENCV'
    d = os.path.join(out_dir, 'nerfstudio')
    os.makedirs(d, exist_ok=True)
    names = _write_images(capture, os.path.join(d, 'images')) if write_images else \
        [f'{k:05d}.png' for k in range(len(capture.views))]
    frames = []
    for k, (v, p) in enumerate(zip(capture.views, pars)):
        f = {'file_path': f'images/{names[k]}', 'transform_matrix': np.round(c2w_opengl(v.T_world_optical), 9).tolist(),
             'fl_x': p['fx'], 'fl_y': p['fy'], 'cx': p['cx'], 'cy': p['cy'], 'w': p['w'], 'h': p['h'],
             'k1': p['k1'], 'k2': p['k2'], 'k3': p['k3'], 'k4': p['k4'],
             'dexvantage_view': v.name, 'dexvantage_q': {u: [float(x) for x in q] for u, q in (v.q or {}).items()}}
        if model == 'OPENCV':
            f.update({'p1': p['p1'], 'p2': p['p2']})
        frames.append(f)
    doc = {'camera_model': model, 'orientation_override': 'none', 'frames': frames,
           'dexvantage': {'schema': SCHEMA, 'synthetic': bool(capture.synthetic), 'source': capture.source,
                          'world': 'the primary robot\'s base frame (REP-103: x forward, y left, z up), metres',
                          'approx_px': {f['dexvantage_view']: p['approx_px'] for f, p in zip(frames, pars)}}}
    p1 = os.path.join(d, 'transforms.json')
    with open(p1, 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=1)
    p2 = os.path.join(d, 'README.md')
    with open(p2, 'w', encoding='utf-8') as fh:
        fh.write(f'# Nerfstudio dataset from DexVantage\n\n`transforms.json` holds {len(frames)} frames with '
                 f'per-frame intrinsics ({model}, corner-based pixels) and camera-to-world matrices (OpenGL camera '
                 'axes, the robot base frame, z up, metres). '
                 + ('These images are synthetic (rendered by MuJoCo), not a real capture. '
                    if capture.synthetic else '')
                 + 'The robot moves between views when their joint states differ (`dexvantage_q`): a trainer for a '
                 'static scene needs views at one joint state.\n\nTrain: `ns-train splatfacto --data .`\n')
    return [os.path.abspath(p1), os.path.abspath(p2)]


def export_colmap(capture, out_dir, points=None, colours=None, write_images: bool = True) -> list[str]:
    """colmap/: sparse/0/cameras.txt (one camera per view), images.txt, points3D.txt (points (K, 3) world and their
    colours (K, 3) in [0, 1], e.g. the capture's back-projected depth; none: an empty list) and images/."""
    d = os.path.join(out_dir, 'colmap')
    sp = os.path.join(d, 'sparse', '0')
    os.makedirs(sp, exist_ok=True)
    names = _write_images(capture, os.path.join(d, 'images')) if write_images else \
        [f'{k:05d}.png' for k in range(len(capture.views))]
    L = ['# Camera list with one line of data per camera:', '#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]',
         f'# Number of cameras: {len(capture.views)}  (DexVantage{" synthetic" if capture.synthetic else ""} capture)']
    for k, v in enumerate(capture.views, start=1):
        p = camera_params(v.camera)
        L.append(f'{k} {p["colmap"]} {p["w"]} {p["h"]} ' + ' '.join(f'{x:.12g}' for x in p['colmap_params']))
    with open(os.path.join(sp, 'cameras.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    L = ['# Image list with two lines of data per image:', '#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME',
         '#   POINTS2D[] as (X, Y, POINT3D_ID)', f'# Number of images: {len(capture.views)}']
    for k, v in enumerate(capture.views, start=1):
        R, t = w2c_opencv(v.T_world_optical)
        x, y, z, w = quat_from_matrix(R)
        L.append(f'{k} {w:.12g} {x:.12g} {y:.12g} {z:.12g} {t[0]:.12g} {t[1]:.12g} {t[2]:.12g} {k} {names[k - 1]}')
        L.append('')
    with open(os.path.join(sp, 'images.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    P = np.zeros((0, 3)) if points is None else np.asarray(points, float).reshape(-1, 3)
    C = np.full((len(P), 3), 0.5) if colours is None else np.asarray(colours, float).reshape(-1, 3)
    L = ['# 3D point list with one line of data per point:',
         '#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)',
         f'# Number of points: {len(P)}']
    rgb = np.clip(np.rint(C * 255), 0, 255).astype(int)
    L += [f'{i + 1} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 1.0' for i, ((x, y, z), (r, g, b)) in enumerate(zip(P, rgb))]
    with open(os.path.join(sp, 'points3D.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    return [os.path.abspath(os.path.join(sp, n)) for n in ('cameras.txt', 'images.txt', 'points3D.txt')]


def read_trained_ply(path, T=None, scale: float = 1.0, max_gaussians: int | None = None):
    """A .ply a trainer wrote (gsplat, Nerfstudio's splatfacto, the INRIA code) as a splat_reproject.GaussianSet in the
    robot base frame (p = scale · R p_file + t by T, default identity: the frame the export wrote)."""
    from .splat_reproject import gaussians_from_ply
    return gaussians_from_ply(path, T, scale, max_gaussians)
