"""Renders of the project's MuJoCo scene (core/sim/scene.py) through each sensor's own lens model (spec §9.1).

    with Renderer(project, variant=None, device='auto') as r:
        images = r.render(views)
        r.info          -> {'renderer', 'gl', 'mujoco', 'device'}
        r.stats         -> {'frames', 'render_seconds', 'calls', 'seconds', 'frames_per_s'}

    views: [{'sensor_uid', 'q': {robot uid: q} or None (home), 'outputs': ('rgb', 'depth', 'seg'), 'scale': 1.0,
             'extrinsic_delta': frames.Pose or 4 × 4 in the sensor body frame, or None,
             'imager': imager name (optional; default the sensor's primary imager, scene.SceneBuild.camera),
             'camera_model': optics.CameraModel (optional; another model of the same imager to render instead)}]
    images: [{'rgb': (H, W, 3) uint8, 'depth': (H, W) float32 metres along the optical z axis (0: no hit),
              'range': (H, W) float32 metres along the pixel's ray (0: no hit), 'seg': (H, W) int32 owner index
              (-1: nothing), 'seg_owner': {index: ('robot', uid, link) | ('object', uid) | ('surface', uid)},
              'model': the CameraModel the image follows, 'sensor_uid', 'imager', 'path': 'pinhole' | 'wide' |
              'cube'}]

Rendering runs in validation/mjrender.py's worker process (MUJOCO_GL=egl, multi-sampling off), kept up for the
Renderer's life (mjrender.Session): the compiled scene is written once and each call sends only joint states and
camera poses. extrinsic_delta perturbs the sensor's mount (T_body_sensor · delta), for a synthetic mount slip.

How an image is made (lens_plan):
- **pinhole**: the scene's camera, its intrinsics those of the CameraModel at the asked size; no resampling.
- **wide**: any other lens whose pixel rays all lie within BASE_MAX_DEG of the axis (a Brown-Conrady lens, a mild
  fisheye): one pinhole image (the same optical frame) that covers every pixel's ray is rendered, and each pixel of
  the lens image takes the base pixel its own ray lands on: for pixel (u, v) the ray r = CameraModel.unproject(u, v)
  goes to base pixel (fx_b r_x / r_z + cx_b, fy_b r_y / r_z + cy_b), and cv2.remap samples it (nearest for the
  segmentation and the depth, bilinear for RGB). That is the lens model's distortion, applied exactly at every pixel
  centre; nothing is fitted.
- **cube**: a lens beyond BASE_MAX_DEG (a wide fisheye, beyond 180° too): the cube faces around the optical frame that
  the rays need are rendered (90° pinholes), and each pixel takes the face its ray leaves through.
The base images are at least as dense, in pixels per radian, as the lens image anywhere (OVERSAMPLE times its
largest angular density, default 2.0, a default), up to MAX_BASE_PX per side; a cap that bites is said in the
plan's notes. The depth of a base pixel is along the base camera's axis; it is turned into the lens's optical z
(and the range along the ray) with the pixel's own ray: z = d · r_z / (r · a), range = d / (r · a) for a base axis a.

MuJoCo's camera conventions (looks along −z, +y up; the principal offset moves the axis's image to
(W − 1)/2 − px, (H − 1)/2 − py) are validation/mjrender.py's, and the intrinsics are set per render through
mjrender's camera_intrinsic (cam_sensorsize and cam_intrinsic in pixel units: sensorsize (W, H), intrinsic
(fx, fy, (W − 1)/2 − cx, (H − 1)/2 − cy)), which MuJoCo reads at every render (MuJoCo 3.17.0, measured in
tests/test_groundtruth.py).

Known limits: nearest-pixel sampling moves an edge by up to half a base pixel (at most half a lens pixel, less where
the base is denser); RGB is MuJoCo's OpenGL shading (no exposure, noise, blur or rolling shutter; Phase 5); the
depth is exact geometry, with no sensor noise (core/studies/groundtruth.py adds the depth sensor's noise model); a
cube face seam can show in the bilinear RGB within a pixel of it.
"""
from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from .scene import OPTICAL_TO_MUJOCO, VIEW_CAMERA, SceneBuild, _mjquat, build_scene

BASE_MAX_DEG = 60.0                # lenses within this of the axis: one wide pinhole base; beyond it: cube faces
OVERSAMPLE = 2.0                   # default: base density / the lens image's largest angular density
MAX_BASE_PX = 4096                 # base image side limit (px)
MAX_FACE_PX = 2048                 # cube face side limit (px)
PLAN_CACHE = 16


@dataclass
class Sub:
    """One base render: its optical frame in the lens's (R, 3 × 3), its pinhole CameraModel, and the lens pixels it
    feeds (sel, (H, W) bool; None: every pixel) with their base coordinates (map_u, map_v, (H, W) float32, −1 where
    not fed) and the cosine factor r · a per pixel."""
    R: np.ndarray
    K: object
    sel: np.ndarray | None = None
    map_u: np.ndarray | None = None
    map_v: np.ndarray | None = None
    cos: np.ndarray | None = None


@dataclass
class Plan:
    path: str
    cam: object
    subs: list
    rays: np.ndarray                   # (H, W, 3) unit rays of the lens's pixels (NaN: none)
    valid: np.ndarray                  # (H, W) bool
    notes: list = field(default_factory=list)


_PLANS: OrderedDict = OrderedDict()


def _pinhole(cam) -> bool:
    return cam.model == 'pinhole' or (cam.model == 'brown' and not any(cam.dist))


def _angular_density(rays: np.ndarray, valid: np.ndarray) -> float:
    """The largest angular density (px/rad) of a pixel grid of unit rays: 1 / the smallest angle between
    neighbouring pixel rays (the 99.9th percentile, so a single pixel at a fold does not set it)."""
    dens = []
    for a, b, ok in ((rays[:, 1:], rays[:, :-1], valid[:, 1:] & valid[:, :-1]),
                     (rays[1:], rays[:-1], valid[1:] & valid[:-1])):
        if ok.any():
            c = np.clip(np.einsum('...i,...i->...', a[ok], b[ok]), -1.0, 1.0)
            ang = np.arccos(c)
            ang = ang[ang > 0]
            if len(ang):
                dens.append(float(np.percentile(1.0 / ang, 99.9)))
    return max(dens) if dens else 1.0


def lens_plan(cam, oversample: float = OVERSAMPLE) -> Plan:
    """How a CameraModel's image is made from MuJoCo pinhole renders (module docstring); cached by model."""
    from ..optics import CameraModel
    key = (cam, float(oversample))
    hit = _PLANS.get(key)
    if hit is not None:
        _PLANS.move_to_end(key)
        return hit
    W, H = cam.width, cam.height
    rays, valid = cam.unproject(np.stack(np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float)), -1),
                                with_ok=True)
    rays = np.asarray(rays, float)
    valid = np.asarray(valid, bool) & np.isfinite(rays).all(axis=-1)
    notes = []
    if _pinhole(cam):
        plan = Plan('pinhole', cam, [Sub(np.eye(3), CameraModel(W, H, cam.fx, cam.fy, cam.cx, cam.cy))], rays, valid)
    else:
        dens = float(oversample) * _angular_density(rays, valid)
        theta = np.degrees(np.arccos(np.clip(rays[..., 2][valid], -1.0, 1.0)))
        tmax = float(theta.max()) if len(theta) else 0.0
        if tmax <= BASE_MAX_DEG:
            r = rays[valid]
            X, Y = r[:, 0] / r[:, 2], r[:, 1] / r[:, 2]
            pad = 1.5 / dens
            x0, x1, y0, y1 = X.min() - pad, X.max() + pad, Y.min() - pad, Y.max() + pad
            f = dens
            if max((x1 - x0), (y1 - y0)) * f > MAX_BASE_PX:
                f = MAX_BASE_PX / max((x1 - x0), (y1 - y0))
                notes.append(f'the base image is capped at {MAX_BASE_PX} px: {f / dens:.2f} of the lens image\'s '
                             'largest angular density')
            Wb, Hb = max(2, int(math.ceil((x1 - x0) * f))), max(2, int(math.ceil((y1 - y0) * f)))
            fx, fy = Wb / (x1 - x0), Hb / (y1 - y0)
            K = CameraModel(Wb, Hb, fx, fy, -0.5 - fx * x0, -0.5 - fy * y0)
            mu = np.full((H, W), -1.0, np.float32)
            mv = np.full((H, W), -1.0, np.float32)
            mu[valid] = K.fx * X + K.cx
            mv[valid] = K.fy * Y + K.cy
            cos = np.where(valid, rays[..., 2], np.nan)
            plan = Plan('wide', cam, [Sub(np.eye(3), K, valid, mu, mv, cos)], rays, valid, notes)
        else:
            subs = []
            dirs = np.array([[0, 0, 1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, -1]], float)
            dots = np.where(valid[..., None], np.nan_to_num(rays) @ dirs.T, -np.inf)
            face = np.argmax(dots, axis=-1)
            n = int(min(MAX_FACE_PX, max(16, math.ceil(2.0 * dens) + 2)))
            if 2.0 * dens + 2 > MAX_FACE_PX:
                notes.append(f'the cube faces are capped at {MAX_FACE_PX} px: {MAX_FACE_PX / (2 * dens + 2):.2f} of '
                             'the lens image\'s largest angular density')
            K = CameraModel(n, n, n / 2.0, n / 2.0, (n - 1) / 2.0, (n - 1) / 2.0)
            for k, z in enumerate(dirs):
                sel = valid & (face == k)
                if not sel.any():
                    continue
                hint = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.5 else np.array([0.0, 0.0, 1.0])
                x = np.cross(hint, z)
                x /= np.linalg.norm(x)
                y = np.cross(z, x)
                R = np.c_[x, y, z]
                r = rays[sel]
                c = r @ z
                mu = np.full((H, W), -1.0, np.float32)
                mv = np.full((H, W), -1.0, np.float32)
                mu[sel] = K.fx * (r @ x) / c + K.cx
                mv[sel] = K.fy * (r @ y) / c + K.cy
                cos = np.full((H, W), np.nan)
                cos[sel] = c
                subs.append(Sub(R, K, sel, mu, mv, cos))
            plan = Plan('cube', cam, subs, rays, valid, notes)
    _PLANS[key] = plan
    while len(_PLANS) > PLAN_CACHE:
        _PLANS.popitem(last=False)
    return plan


def sub_jobs(model, camera: str, T_body_optical, plan: Plan, outputs, qpos=None) -> list:
    """mjrender jobs for a plan: the MuJoCo camera `camera` (in its body) posed at each base render's optical frame
    T_body_optical · R and given its pinhole intrinsics (pixel units)."""
    T = np.asarray(T_body_optical, float)
    out = []
    for s in plan.subs:
        R = T[:3, :3] @ s.R @ OPTICAL_TO_MUJOCO
        K = s.K
        out.append({'model': model, 'qpos': qpos, 'camera': camera, 'width': K.width, 'height': K.height,
                    'outputs': tuple(outputs),
                    'camera_pose': np.r_[T[:3, 3], _mjquat(R)],
                    'camera_intrinsic': {'sensorsize': (K.width, K.height),
                                         'intrinsic': (K.fx, K.fy, (K.width - 1) / 2 - K.cx,
                                                       (K.height - 1) / 2 - K.cy)}})
    return out


def assemble(plan: Plan, base: list, outputs, far: float, geom_owner=None) -> dict:
    """The lens image from the base renders of sub_jobs (module docstring): 'seg' (geom ids, or owner indices through
    geom_owner), 'depth' (optical z), 'range', 'rgb'."""
    import cv2
    cam = plan.cam
    H, W = cam.height, cam.width
    out = {}
    want = set(outputs)

    def seg_of(img):
        s = np.asarray(img, np.int32)
        if geom_owner is not None:
            lut = np.asarray(geom_owner, np.int32)
            s = np.where(s >= 0, lut[np.clip(s, 0, max(len(lut) - 1, 0))], -1).astype(np.int32)
        return s

    def depth_of(img):
        d = np.asarray(img, np.float32).copy()
        d[~(d < far * 0.999)] = 0.0
        return d

    if plan.path == 'pinhole':
        b = base[0]
        if 'seg' in want:
            out['seg'] = seg_of(b['seg'])
        if 'depth' in want:
            d = depth_of(b['depth'])
            out['depth'] = d
            with np.errstate(invalid='ignore', divide='ignore'):
                rz = np.where(plan.valid, plan.rays[..., 2], np.nan)
                out['range'] = np.where(d > 0, d / rz, 0.0).astype(np.float32)
        if 'rgb' in want:
            out['rgb'] = np.asarray(b['rgb'], np.uint8)
        return out
    seg = np.full((H, W), -1, np.int32)
    dep = np.zeros((H, W), np.float32)
    rng = np.zeros((H, W), np.float32)
    rgb = np.zeros((H, W, 3), np.uint8)
    for s, b in zip(plan.subs, base):
        sel = s.sel
        if 'seg' in want:
            r = cv2.remap(np.asarray(b['seg'], np.float32), s.map_u, s.map_v, cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0)
            seg[sel] = seg_of(np.rint(r[sel]).astype(np.int32))
        if 'depth' in want:
            r = cv2.remap(depth_of(b['depth']), s.map_u, s.map_v, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0.0)
            d = r[sel].astype(float)
            c = s.cos[sel]
            with np.errstate(invalid='ignore', divide='ignore'):
                rr = np.where(d > 0, d / c, 0.0)
                zz = np.where(d > 0, rr * plan.rays[..., 2][sel], 0.0)
            rng[sel] = rr
            dep[sel] = zz
        if 'rgb' in want:
            r = cv2.remap(np.asarray(b['rgb'], np.uint8), s.map_u, s.map_v, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
            rgb[sel] = r[sel]
    if 'seg' in want:
        out['seg'] = seg
    if 'depth' in want:
        out['depth'], out['range'] = dep, rng
    if 'rgb' in want:
        out['rgb'] = rgb
    return out


def render_lens(session, model, camera: str, T_body_optical, cam, outputs=('seg', 'depth'), qpos=None,
                geom_owner=None, oversample: float = OVERSAMPLE) -> dict:
    """One lens image of a compiled model's camera (any MuJoCo model, for the checks): the camera posed at
    T_body_optical in its body, imaged through the CameraModel cam."""
    plan = lens_plan(cam, oversample)
    base, info = session.render(sub_jobs(model, camera, T_body_optical, plan, outputs, qpos))
    far = float(model.vis.map.zfar * model.stat.extent)
    out = assemble(plan, base, outputs, far, geom_owner)
    out.update(model=cam, path=plan.path, notes=list(plan.notes), info=info)
    return out


class Renderer:
    """The project's scene rendered through its sensors' lens models (module docstring). A context manager; close()
    stops the worker."""

    def __init__(self, project, variant=None, device='auto', objects: bool = True, lights: bool = True,
                 timeout: float = 180.0, oversample: float = OVERSAMPLE):
        from ...validation import mjrender as MR
        self.project = project
        self.build: SceneBuild = build_scene(project, variant, objects=objects, lights=lights, view_camera=True)
        self.model = self.build.model
        self.device = device
        self.oversample = float(oversample)
        self.session = MR.Session(None if device in (None, 'auto') else device, timeout)
        self._info = {}
        self.stats = {'frames': 0, 'render_seconds': 0.0, 'calls': 0, 'seconds': 0.0, 'frames_per_s': None}
        self.far = float(self.model.vis.map.zfar * self.model.stat.extent)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.session.close()

    @property
    def notes(self) -> list:
        return list(self.build.notes)

    @property
    def info(self) -> dict:
        """{'renderer', 'gl', 'mujoco', 'device'}: the OpenGL renderer string comes from the worker's first render
        (a 16 × 16 one is made when none was asked for yet)."""
        if not self._info and self.build.cameras:
            from ..optics import CameraModel
            c = next(iter(self.build.cameras.values()))
            self._info = render_lens(self.session, self.model, c.name, c.local_optical(),
                                     CameraModel(16, 16, 8, 8, 7.5, 7.5), ('seg',))['info']
        i = self._info
        return {'renderer': i.get('renderer', ''), 'gl': i.get('gl', 'egl'), 'mujoco': i.get('mujoco', ''),
                'device': i.get('device', '')}

    def camera(self, sensor_uid: str, imager: str | None = None):
        c = self.build.camera(sensor_uid, imager)
        if c is None:
            s = self.project.variant.sensor(sensor_uid)
            what = f'{s.name}' if s is not None else f'sensor {sensor_uid}'
            raise KeyError(f'{what} has no camera in the MuJoCo scene' + (f' for imager {imager}' if imager else '')
                           + (': ' + ' '.join(self.build.notes[-1:]) if self.build.notes else '.'))
        return c

    def render(self, views: list) -> list:
        """Render the views (module docstring) in one worker call."""
        t0 = time.perf_counter()
        jobs, parts = [], []
        for v in views:
            c = self.camera(v['sensor_uid'], v.get('imager'))
            cam = v.get('camera_model') or c.model
            sc = float(v.get('scale') or 1.0)
            if abs(sc - 1.0) > 1e-12:
                cam = cam.scaled(max(1, int(round(cam.width * sc))), max(1, int(round(cam.height * sc))))
            outs = tuple(o for o in (v.get('outputs') or ('rgb', 'depth', 'seg')) if o in ('rgb', 'depth', 'seg'))
            plan = lens_plan(cam, self.oversample)
            qpos = self.build.qpos(v.get('q'))
            js = sub_jobs(self.model, c.name, c.local_optical(v.get('extrinsic_delta')), plan, outs, qpos)
            parts.append((c, plan, outs, len(jobs), len(js)))
            jobs += js
        base, info = self.session.render(jobs) if jobs else ([], {})
        if not self._info and info:
            self._info = info
        out = []
        owners = dict(enumerate(self.build.owners))
        for c, plan, outs, a, n in parts:
            img = assemble(plan, base[a:a + n], outs, self.far, self.build.geom_owner)
            img.update(seg_owner=owners, model=plan.cam, sensor_uid=c.sensor_uid, imager=c.imager, path=plan.path,
                       notes=list(plan.notes))
            out.append(img)
        st = self.stats
        st['calls'] += 1
        st['frames'] += int(info.get('frames', 0))
        st['render_seconds'] += float(info.get('render_seconds', 0.0))
        st['seconds'] += time.perf_counter() - t0
        st['frames_per_s'] = st['frames'] / st['render_seconds'] if st['render_seconds'] > 0 else None
        return out

    def render_pose(self, cam_model, pose, q: dict | None = None, outputs=('seg', 'depth')) -> dict:
        """One image through cam_model from any optical frame in the world (frames.Pose or 4 × 4), the robots at q:
        the Camera view's pose, for its overlay of the rendered segmentation."""
        T = pose.matrix() if hasattr(pose, 'matrix') else np.asarray(pose, float).reshape(4, 4)
        plan = lens_plan(cam_model, self.oversample)
        outs = tuple(o for o in outputs if o in ('rgb', 'depth', 'seg'))
        base, _info = self.session.render(sub_jobs(self.model, VIEW_CAMERA, T, plan, outs, self.build.qpos(q)))
        img = assemble(plan, base, outs, self.far, self.build.geom_owner)
        img.update(seg_owner=dict(enumerate(self.build.owners)), model=plan.cam, path=plan.path)
        return img

    def owner_index(self, owner) -> int:
        """The seg value of an owner tuple (-1 when it is not in the scene)."""
        try:
            return self.build.owners.index(tuple(owner))
        except ValueError:
            return -1
