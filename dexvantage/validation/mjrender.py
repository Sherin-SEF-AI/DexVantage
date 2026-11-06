"""MuJoCo offscreen renders for the validation checks: segmentation, depth and RGB from a camera of a compiled model.

    render(jobs, timeout=180) -> (images, info)
        jobs: [{'model': mujoco.MjModel, 'qpos': array | None, 'camera': camera name, 'width': w, 'height': h,
                'outputs': ('seg', 'depth', 'rgb')}]
        images: one dict per job: 'seg' (H, W) int32 geom id (-1: no geom), 'depth' (H, W) float32 metres along the
                camera's viewing axis (MuJoCo's "distance from camera plane"), 'rgb' (H, W, 3) uint8; row 0 is the top
                of the image and column 0 its left, as mujoco.Renderer returns them.
        info: {'renderer': the OpenGL renderer string, 'mujoco': version, 'gl': 'egl', 'seconds'}
    mujoco_camera(m, d, name) -> (R_optical (3,3), position (3,)): the camera's optical frame (z forward, x right,
        y down) in the world, from MuJoCo's camera frame (it looks along its -z axis, +x right, +y up:
        R_optical = R_mujoco · diag(1, −1, −1)).
    camera_model(m, name, width, height) -> optics.CameraModel of a MuJoCo camera rendered at width × height
    add_camera(spec, name, R_optical, position, fovy_deg, width, height) -> the camera, posed from an optical frame
    look_at(position, target) -> R_optical looking from position at target (image up as near world +z as can be)
    robot_spec(project, robot_uid, folder) -> mujoco.MjSpec of the robot's stored files (written to folder)
    copy_qpos(model, q, m_scene) -> qpos of m_scene with the robot's joints at q (matched by joint name)

MuJoCo's camera intrinsics (XML reference, body/camera; doc/XMLreference.rst of github.com/google-deepmind/mujoco,
read 2026-11-06): fovy is the vertical field of view in degrees and the horizontal one follows from the image's aspect
(square pixels), so fy = fx = (H/2) / tan(fovy/2); with sensorsize set, fovy is ignored and the focal length
(focal in length units, or focalpixel) and the principal-point offset from the image centre (principal, principalpixel)
apply. MuJoCo stores them as cam_intrinsic = [fx, fy, px, py] in length units, so fx_px = fx · W / sensor width.
The image centre is at (W − 1)/2, (H − 1)/2 in OpenCV's integer pixel centres. The direction of the principal offset
is not stated in the reference; rendered on MuJoCo 3.17.0 (a sphere on the optical axis), principalpixel (px, py)
moves the image of the optical axis to (W − 1)/2 − px, (H − 1)/2 − py (row 0 at the top), and camera_model uses
that: the fiducial check (checks_optics_mujoco.py) confirms it holds over the whole image and with the focal lengths,
but the sign itself was read off a render, not from the documentation.

Rendering runs in a worker process (python -m dexvantage.validation.mjrender) with MUJOCO_GL=egl and
PYOPENGL_PLATFORM=egl set before mujoco is imported: MuJoCo picks its OpenGL platform when it is first imported, and
this process has imported mujoco (for kinematics) and may hold PyOpenGL for another platform (Qt's GLX). The models
travel as MuJoCo binary files (mj_saveModel) in a temporary folder.

Multi-sampling is switched off in the worker (visual/quality offsamples = 0): with anti-aliasing, the segmentation
colours of two geoms blend at their edges into ids of neither, and depth is averaged across them.

When EGL rendering is not available (no OpenGL driver with EGL, a headless machine without Mesa, a worker that
cannot start), render raises RenderUnavailable with the reason in plain words: the checks that need renders then fail
with that message, they are not skipped.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

OUTPUTS = ('seg', 'depth', 'rgb')


class RenderUnavailable(RuntimeError):
    """MuJoCo cannot render offscreen through EGL here; the message says why."""


def mujoco_camera(m, d, name: str) -> tuple[np.ndarray, np.ndarray]:
    """(R, p): the optical frame (z forward, x right, y down) of MuJoCo camera `name` in the world. MuJoCo's camera
    frame looks along its −z axis with +y up (XML reference, body/camera), so the optical frame's axes are MuJoCo's
    x, −y and −z: R_optical = R_mujoco · diag(1, −1, −1). Needs mj_kinematics (or mj_forward) on d."""
    cid = m.camera(name).id
    R = np.asarray(d.cam_xmat[cid], float).reshape(3, 3)
    return R @ np.diag([1.0, -1.0, -1.0]), np.asarray(d.cam_xpos[cid], float).copy()


def world_to_optical(R: np.ndarray, p: np.ndarray, X) -> np.ndarray:
    """World points X (N, 3) in the optical frame (R, p) of mujoco_camera."""
    return (np.asarray(X, float) - p) @ R


def camera_model(m, name: str, width: int, height: int):
    """The optics.CameraModel (pinhole, OpenCV conventions) of MuJoCo camera `name` rendered at width × height; see
    the module docstring for the conversion."""
    import math
    from ..core.optics import CameraModel
    cid = m.camera(name).id
    W, H = int(width), int(height)
    ss = np.asarray(m.cam_sensorsize[cid], float)
    if ss[0] > 0 and ss[1] > 0:
        fx_l, fy_l, px_l, py_l = (float(v) for v in m.cam_intrinsic[cid])
        fx, fy = fx_l / ss[0] * W, fy_l / ss[1] * H
        cx, cy = (W - 1) / 2 - px_l / ss[0] * W, (H - 1) / 2 - py_l / ss[1] * H
    else:
        fy = (H / 2) / math.tan(math.radians(float(m.cam_fovy[cid])) / 2)
        fx, cx, cy = fy, (W - 1) / 2, (H - 1) / 2
    return CameraModel(W, H, fx, fy, cx, cy)


def look_at(position, target) -> np.ndarray:
    """R_optical (columns: optical x right, y down, z forward in the world) looking from position at target, its
    image up as close to world +z as the view allows (world +x when looking straight up or down)."""
    z = np.asarray(target, float) - np.asarray(position, float)
    z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    return np.c_[x, np.cross(z, x), z]


def add_camera(spec, name: str, R_optical, position, fovy_deg: float, width: int, height: int):
    """Add a fixed camera to spec's world body whose optical frame is (R_optical, position): MuJoCo's camera frame
    is R_optical · diag(1, −1, −1) (it looks along −z with +y up)."""
    from ..core.frames import quat_from_matrix, quat_to_mujoco
    Rm = np.asarray(R_optical, float) @ np.diag([1.0, -1.0, -1.0])
    cam = spec.worldbody.add_camera(name=name, pos=np.asarray(position, float),
                                    quat=np.asarray(quat_to_mujoco(quat_from_matrix(Rm)), float))
    cam.fovy = float(fovy_deg)
    cam.resolution = [int(width), int(height)]
    return cam


def robot_spec(project, robot_uid: str, folder: str):
    """mujoco.MjSpec of the robot's stored model file, its referenced files written under folder as stored (MJCF
    only: the scripted scenes add cameras and targets to it)."""
    import mujoco
    r = project.robot(robot_uid)
    pre = r.uid + '/'
    for k, b in project.sources.items():
        if k.startswith(pre):
            path = os.path.join(folder, *k[len(pre):].split('/'))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                f.write(b)
    if (r.source.kind or 'urdf') != 'mjcf':
        raise ValueError(f'{r.name} is stored as {r.source.kind or "URDF"}; the scripted render scenes take MJCF.')
    return mujoco.MjSpec.from_file(os.path.join(folder, *r.source.file[len(pre):].split('/')))


def copy_qpos(model, q, m_scene) -> np.ndarray:
    """qpos of m_scene with every named one-value joint (hinge, slide) of the RobotModel at q (the others at the
    scene's qpos0), matched by joint name."""
    import mujoco
    d0 = model._kin(model._q(q))
    out = np.asarray(m_scene.qpos0, float).copy()
    J = mujoco.mjtJoint
    for j in range(model.m.njnt):
        if int(model.m.jnt_type[j]) not in (int(J.mjJNT_HINGE), int(J.mjJNT_SLIDE)):
            continue
        name = model.m.joint(j).name
        try:
            js = m_scene.joint(name).id
        except KeyError:
            continue
        out[int(m_scene.jnt_qposadr[js])] = float(d0.qpos[int(model.m.jnt_qposadr[j])])
    return out


def render(jobs: list, timeout: float = 180.0) -> tuple[list, dict]:
    """Render the jobs in a worker process (see the module docstring). Raises RenderUnavailable."""
    import mujoco
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='dexvantage-mjrender-') as tmp:
        spec, saved = [], {}
        for k, j in enumerate(jobs):
            m = j['model']
            if id(m) not in saved:
                path = os.path.join(tmp, f'model_{len(saved)}.mjb')
                m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), int(j['width']))
                m.vis.global_.offheight = max(int(m.vis.global_.offheight), int(j['height']))
                mujoco.mj_saveModel(m, path, None)
                saved[id(m)] = os.path.basename(path)
            q = j.get('qpos')
            if q is not None:
                np.save(os.path.join(tmp, f'qpos_{k}.npy'), np.asarray(q, float))
            outs = tuple(o for o in j.get('outputs', ('seg', 'depth')) if o in OUTPUTS)
            spec.append({'model': saved[id(m)], 'qpos': f'qpos_{k}.npy' if q is not None else None,
                         'camera': str(j['camera']), 'width': int(j['width']), 'height': int(j['height']),
                         'outputs': list(outs)})
        with open(os.path.join(tmp, 'jobs.json'), 'w', encoding='utf-8') as f:
            json.dump(spec, f)
        env = dict(os.environ, MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl')
        env.setdefault('EGL_LOG_LEVEL', 'fatal')
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env['PYTHONPATH'] = root + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        try:
            pr = subprocess.run([sys.executable, '-m', 'dexvantage.validation.mjrender', tmp], env=env, cwd=root,
                                capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RenderUnavailable(f'MuJoCo\'s offscreen renderer did not finish within {timeout:.0f} s (EGL '
                                    'rendering hung or is very slow on this machine).') from None
        except OSError as e:
            raise RenderUnavailable(f'the render worker could not start ({e}).') from None
        msg = pr.stdout.decode('utf-8', 'replace').strip().splitlines()
        last = msg[-1] if msg else ''
        if pr.returncode != 0 or not last.startswith('ok\t'):
            why = last[len('error\t'):] if last.startswith('error\t') else ''
            if not why:
                err = pr.stderr.decode('utf-8', 'replace').strip().splitlines()
                why = err[-1] if err else f'the worker stopped with exit code {pr.returncode}'
            raise RenderUnavailable('MuJoCo cannot render offscreen through EGL on this machine, so the rendered '
                                    f'reference is missing: {why}. Rendered checks need an OpenGL driver with EGL '
                                    '(a GPU driver, or Mesa for software rendering).')
        info = json.loads(last[len('ok\t'):])
        out = []
        with np.load(os.path.join(tmp, 'out.npz'), allow_pickle=False) as z:
            for k, j in enumerate(spec):
                out.append({o: z[f'{k}_{o}'] for o in j['outputs']})
    info['seconds'] = time.perf_counter() - t0
    return out, info


def _worker(tmp: str) -> int:
    """Render the jobs in folder tmp (jobs.json, *.mjb, qpos_*.npy) into tmp/out.npz; print 'ok\\t{info}' or
    'error\\t<reason>' as the last line."""
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    try:
        import mujoco
    except Exception as e:                  # noqa: BLE001 — reported to the parent in words
        print(f'error\tmujoco could not be imported with MUJOCO_GL=egl ({type(e).__name__}: {e})', flush=True)
        return 3
    with open(os.path.join(tmp, 'jobs.json'), encoding='utf-8') as f:
        jobs = json.load(f)
    models, res, renderer_name = {}, {}, ''
    renderers = {}
    try:
        for k, j in enumerate(jobs):
            if j['model'] not in models:
                m = mujoco.MjModel.from_binary_path(os.path.join(tmp, j['model']))
                m.vis.quality.offsamples = 0          # no multi-sampling: exact ids and depths at pixel centres
                models[j['model']] = (m, mujoco.MjData(m))
            m, d = models[j['model']]
            if j['qpos']:
                d.qpos[:] = np.load(os.path.join(tmp, j['qpos']))
            mujoco.mj_forward(m, d)
            key = (j['model'], j['width'], j['height'])
            r = renderers.get(key)
            if r is None:
                r = renderers[key] = mujoco.Renderer(m, j['height'], j['width'])
                if not renderer_name:
                    try:
                        from OpenGL import GL
                        renderer_name = (GL.glGetString(GL.GL_RENDERER) or b'').decode('utf-8', 'replace')
                    except Exception:       # noqa: BLE001 — the name is only reported
                        renderer_name = 'unknown'
            for o in j['outputs']:
                if o == 'seg':
                    r.enable_segmentation_rendering()
                elif o == 'depth':
                    r.enable_depth_rendering()
                r.update_scene(d, camera=j['camera'])
                img = r.render().copy()
                if o == 'seg':
                    r.disable_segmentation_rendering()
                    gid = np.where(img[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), img[..., 0], -1).astype(np.int32)
                    res[f'{k}_seg'] = gid
                elif o == 'depth':
                    r.disable_depth_rendering()
                    res[f'{k}_depth'] = img.astype(np.float32)
                else:
                    res[f'{k}_rgb'] = img.astype(np.uint8)
        np.savez(os.path.join(tmp, 'out.npz'), **res)
        for r in renderers.values():
            r.close()
    except Exception as e:                  # noqa: BLE001 — reported to the parent in words
        msg = str(e).strip().splitlines()
        print(f'error\t{type(e).__name__}: {msg[0] if msg else ""}', flush=True)
        os._exit(3)
    print('ok\t' + json.dumps({'renderer': renderer_name, 'mujoco': mujoco.__version__, 'gl': 'egl'}), flush=True)
    sys.stdout.flush()
    os._exit(0)                             # skip EGL teardown at exit (MuJoCo's deallocators print errors there)


if __name__ == '__main__':
    raise SystemExit(_worker(sys.argv[1]))
