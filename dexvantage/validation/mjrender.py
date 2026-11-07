"""MuJoCo offscreen renders for the validation checks: segmentation, depth and RGB from a camera of a compiled model.

    render(jobs, timeout=180, device=None) -> (images, info)
        jobs: [{'model': mujoco.MjModel, 'qpos': array | None, 'camera': camera name, 'width': w, 'height': h,
                'outputs': ('seg', 'depth', 'rgb'), 'camera_pose': None or (x, y, z, qw, qx, qy, qz),
                'camera_intrinsic': None or {'sensorsize': (w, h), 'intrinsic': (fx, fy, px, py)} (lengths)}]
                camera_pose and camera_intrinsic (optional) replace the camera's pose in its body (MuJoCo's cam_pos,
                cam_quat) and its intrinsics (cam_sensorsize, cam_intrinsic, which MuJoCo reads at every render) for
                that job only (core/sim/render.py: a perturbed mount, the wide pinhole or cube faces a distorted lens
                is remapped from)
        images: one dict per job: 'seg' (H, W) int32 geom id (-1: no geom), 'depth' (H, W) float32 metres along the
                camera's viewing axis (MuJoCo's "distance from camera plane"), 'rgb' (H, W, 3) uint8; row 0 is the top
                of the image and column 0 its left, as mujoco.Renderer returns them.
        info: {'renderer': the OpenGL renderer string, 'mujoco': version, 'gl': 'egl', 'seconds' (the call, with the
               worker's start), 'render_seconds' (the renders alone, in the worker), 'frames', 'device'}
    Session(device=None, timeout=180)   the same render(jobs) from a worker that stays up between calls (python -m
        dexvantage.validation.mjrender --serve): models are written once per session and stay loaded with their
        renderers (at most MAX_RENDERERS GL contexts, least recently used closed). A context manager.
    device: None or 'auto' is MuJoCo's own choice, the first EGL device that initialises (mujoco/egl/__init__.py,
        MuJoCo 3.17.0), unless DEXVANTAGE_EGL says 'EGL device N'; an int (or 'EGL device N') sets
        MUJOCO_EGL_DEVICE_ID for the worker.
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


def _job_spec(jobs: list, folder: str, saved: dict, keep: dict | None = None, absolute: bool = False) -> list:
    """Write the jobs' models (once per model object: saved maps id(model) to its file) and joint states into
    folder; the jobs as the worker reads them. keep holds the models saved (so their ids are not reused)."""
    import mujoco
    spec = []
    for k, j in enumerate(jobs):
        m = j['model']
        if id(m) not in saved:
            path = os.path.join(folder if not absolute else keep['__dir__'], f'model_{len(saved)}.mjb')
            mujoco.mj_saveModel(m, path, None)
            saved[id(m)] = path if absolute else os.path.basename(path)
            if keep is not None:
                keep[id(m)] = m
        q = j.get('qpos')
        if q is not None:
            np.save(os.path.join(folder, f'qpos_{k}.npy'), np.asarray(q, float))
        outs = tuple(o for o in j.get('outputs', ('seg', 'depth')) if o in OUTPUTS)
        cp, ci = j.get('camera_pose'), j.get('camera_intrinsic')
        spec.append({'model': saved[id(m)], 'qpos': f'qpos_{k}.npy' if q is not None else None,
                     'camera': str(j['camera']), 'width': int(j['width']), 'height': int(j['height']),
                     'outputs': list(outs),
                     'camera_pose': None if cp is None else [float(v) for v in np.asarray(cp, float).reshape(7)],
                     'camera_intrinsic': None if ci is None else
                     {'sensorsize': [float(v) for v in np.asarray(ci['sensorsize'], float).reshape(2)],
                      'intrinsic': [float(v) for v in np.asarray(ci['intrinsic'], float).reshape(4)]}})
    with open(os.path.join(folder, 'jobs.json'), 'w', encoding='utf-8') as f:
        json.dump(spec, f)
    return spec


def _read_out(folder: str, spec: list) -> list:
    out = []
    with np.load(os.path.join(folder, 'out.npz'), allow_pickle=False) as z:
        for k, j in enumerate(spec):
            out.append({o: z[f'{k}_{o}'] for o in j['outputs']})
    return out


def worker_env(device=None) -> tuple[dict, str]:
    """(environment, root) for a render worker: MUJOCO_GL=egl and PYOPENGL_PLATFORM=egl, the package on the path, and
    MUJOCO_EGL_DEVICE_ID when device names one (an int, or 'EGL device N'; also read from DEXVANTAGE_EGL when it says
    'EGL device N'). Without one MuJoCo takes the first EGL device that initialises (mujoco/egl/__init__.py)."""
    env = dict(os.environ, MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl')
    env.setdefault('EGL_LOG_LEVEL', 'fatal')
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env['PYTHONPATH'] = root + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    idx = _device_index(device)
    if idx is None and device in (None, 'auto'):
        idx = _device_index(os.environ.get('DEXVANTAGE_EGL', ''))
    if idx is not None:
        env['MUJOCO_EGL_DEVICE_ID'] = str(idx)
    return env, root


def _device_index(device):
    if isinstance(device, bool):
        return None
    if isinstance(device, int):
        return device if device >= 0 else None
    s = str(device or '').strip().lower()
    if s.startswith('egl device'):
        s = s[len('egl device'):].strip()
    return int(s) if s.isdigit() else None


def device_text(device=None) -> str:
    """Which EGL device a worker started with device uses, in words."""
    env, _root = worker_env(device)
    d = env.get('MUJOCO_EGL_DEVICE_ID')
    return f'EGL device {d}' if d is not None else 'the first EGL device that initialises'


def _unavailable(why: str) -> RenderUnavailable:
    return RenderUnavailable('MuJoCo cannot render offscreen through EGL on this machine, so the rendered reference is '
                             f'missing: {why}. Rendered checks need an OpenGL driver with EGL (a GPU driver, or Mesa '
                             'for software rendering).')


def render(jobs: list, timeout: float = 180.0, device=None) -> tuple[list, dict]:
    """Render the jobs in a worker process (see the module docstring). device: None or 'auto' (MuJoCo's choice, or
    DEXVANTAGE_EGL='EGL device N'), or an EGL device index. Raises RenderUnavailable."""
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='dexvantage-mjrender-') as tmp:
        spec = _job_spec(jobs, tmp, {})
        env, root = worker_env(device)
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
            raise _unavailable(why)
        info = json.loads(last[len('ok\t'):])
        out = _read_out(tmp, spec)
    info['seconds'] = time.perf_counter() - t0
    info['device'] = device_text(device)
    return out, info


class Session:
    """A render worker that stays up between calls (python -m dexvantage.validation.mjrender --serve): each model is
    written once per session and the worker keeps it loaded with its renderers, so a call costs the renders only.
    A context manager; render(jobs) as the module's render(). Raises RenderUnavailable when the worker cannot
    render (the message says why) and keeps the reason for every later call."""

    def __init__(self, device=None, timeout: float = 180.0):
        import queue
        import threading
        self.device = device
        self.timeout = float(timeout)
        self._dir = tempfile.TemporaryDirectory(prefix='dexvantage-mjsession-')
        self._saved: dict = {}
        self._keep: dict = {'__dir__': self._dir.name}
        self._calls = 0
        self._dead = ''
        self._err = open(os.path.join(self._dir.name, 'stderr.txt'), 'wb')
        env, root = worker_env(device)
        try:
            self._p = subprocess.Popen([sys.executable, '-m', 'dexvantage.validation.mjrender', '--serve'], env=env,
                                       cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err)
        except OSError as e:
            self._p = None
            self._dead = f'the render worker could not start ({e})'
            return
        self._lines: queue.Queue = queue.Queue()

        def pump(f, q):
            for ln in iter(f.readline, b''):
                q.put(ln.decode('utf-8', 'replace').rstrip('\n'))
            q.put(None)
        self._thread = threading.Thread(target=pump, args=(self._p.stdout, self._lines), daemon=True)
        self._thread.start()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _reply(self) -> str:
        import queue
        t_end = time.perf_counter() + self.timeout
        while True:
            left = t_end - time.perf_counter()
            if left <= 0:
                self._kill()
                self._dead = (f'MuJoCo\'s offscreen renderer did not finish within {self.timeout:.0f} s (EGL '
                              'rendering hung or is very slow on this machine)')
                raise RenderUnavailable(self._dead + '.')
            try:
                ln = self._lines.get(timeout=min(left, 1.0))
            except queue.Empty:
                continue
            if ln is None:
                return ''
            if ln.startswith(('ok\t', 'error\t')):
                return ln

    def render(self, jobs: list) -> tuple[list, dict]:
        if self._dead:
            raise _unavailable(self._dead)
        t0 = time.perf_counter()
        self._calls += 1
        folder = os.path.join(self._dir.name, f'call_{self._calls}')
        os.makedirs(folder)
        try:
            spec = _job_spec(jobs, folder, self._saved, self._keep, absolute=True)
            try:
                self._p.stdin.write(f'render\t{folder}\n'.encode('utf-8'))
                self._p.stdin.flush()
            except (OSError, ValueError):
                pass                                    # the worker is gone: its last words are read below
            last = self._reply()
            if not last.startswith('ok\t'):
                why = last[len('error\t'):] if last.startswith('error\t') else ''
                if not why:
                    self._err.flush()
                    try:
                        with open(self._err.name, 'rb') as f:
                            err = f.read().decode('utf-8', 'replace').strip().splitlines()
                    except OSError:
                        err = []
                    code = self._p.poll()
                    why = err[-1] if err else f'the worker stopped (exit code {code})'
                    self._dead = why
                raise _unavailable(why)
            info = json.loads(last[len('ok\t'):])
            out = _read_out(folder, spec)
        finally:
            import shutil
            shutil.rmtree(folder, ignore_errors=True)
        info['seconds'] = time.perf_counter() - t0
        info['device'] = device_text(self.device)
        return out, info

    def _kill(self):
        if self._p is not None and self._p.poll() is None:
            self._p.kill()
            try:
                self._p.wait(5)
            except subprocess.TimeoutExpired:
                pass

    def close(self):
        if self._p is not None:
            if self._p.poll() is None:
                try:
                    self._p.stdin.write(b'quit\n')
                    self._p.stdin.flush()
                    self._p.stdin.close()
                except (OSError, ValueError):
                    pass
                try:
                    self._p.wait(10)
                except subprocess.TimeoutExpired:
                    self._kill()
            self._p = None
            self._dead = self._dead or 'the render session is closed'
        try:
            self._err.close()
        except OSError:
            pass
        self._keep.clear()
        self._saved.clear()
        try:
            self._dir.cleanup()
        except OSError:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:                   # noqa: BLE001 — interpreter shutdown
            pass


MAX_RENDERERS = 8                           # renderers (one GL context each) a worker keeps, least recently used out


class _State:
    """What a worker keeps between calls: the loaded models and their renderers."""

    def __init__(self):
        from collections import OrderedDict
        self.models: dict = {}
        self.renderers: OrderedDict = OrderedDict()
        self.renderer_name = ''


def _render_folder(mujoco, tmp: str, state: _State) -> dict:
    """Render the jobs in folder tmp (jobs.json, models, qpos_*.npy) into tmp/out.npz; the info dict."""
    with open(os.path.join(tmp, 'jobs.json'), encoding='utf-8') as f:
        jobs = json.load(f)
    res = {}
    t_render, frames = 0.0, 0
    for k, j in enumerate(jobs):
        path = os.path.join(tmp, j['model'])
        if path not in state.models:
            m = mujoco.MjModel.from_binary_path(path)
            m.vis.quality.offsamples = 0          # no multi-sampling: exact ids and depths at pixel centres
            state.models[path] = (m, mujoco.MjData(m))
        m, d = state.models[path]
        d.qpos[:] = np.load(os.path.join(tmp, j['qpos'])) if j['qpos'] else m.qpos0
        cp, ci = j.get('camera_pose'), j.get('camera_intrinsic')
        cid = old = None
        if cp or ci:
            cid = m.camera(j['camera']).id
            old = (m.cam_pos[cid].copy(), m.cam_quat[cid].copy(), m.cam_sensorsize[cid].copy(),
                   m.cam_intrinsic[cid].copy())
            if cp:
                m.cam_pos[cid] = cp[:3]
                m.cam_quat[cid] = cp[3:]
            if ci:
                m.cam_sensorsize[cid] = ci['sensorsize']
                m.cam_intrinsic[cid] = ci['intrinsic']
        try:
            mujoco.mj_forward(m, d)
            key = (path, j['width'], j['height'])
            r = state.renderers.get(key)
            if r is None:
                m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), int(j['width']))
                m.vis.global_.offheight = max(int(m.vis.global_.offheight), int(j['height']))
                r = state.renderers[key] = mujoco.Renderer(m, j['height'], j['width'])
                while len(state.renderers) > MAX_RENDERERS:
                    _k, old_r = state.renderers.popitem(last=False)
                    old_r.close()
                if not state.renderer_name:
                    try:
                        from OpenGL import GL
                        state.renderer_name = (GL.glGetString(GL.GL_RENDERER) or b'').decode('utf-8', 'replace')
                    except Exception:       # noqa: BLE001 — the name is only reported
                        state.renderer_name = 'unknown'
            else:
                state.renderers.move_to_end(key)
            for o in j['outputs']:
                t0 = time.perf_counter()
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
                t_render += time.perf_counter() - t0
                frames += 1
        finally:
            if cid is not None:
                m.cam_pos[cid], m.cam_quat[cid], m.cam_sensorsize[cid], m.cam_intrinsic[cid] = old
    np.savez(os.path.join(tmp, 'out.npz'), **res)
    return {'renderer': state.renderer_name, 'mujoco': mujoco.__version__, 'gl': 'egl',
            'render_seconds': round(t_render, 6), 'frames': frames}


def _import_mujoco():
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    try:
        import mujoco
    except Exception as e:                  # noqa: BLE001 — reported to the parent in words
        print(f'error\tmujoco could not be imported with MUJOCO_GL=egl ({type(e).__name__}: {e})', flush=True)
        return None
    return mujoco


def _error_line(e: Exception) -> str:
    msg = str(e).strip().splitlines()
    return f'error\t{type(e).__name__}: {msg[0] if msg else ""}'


def _worker(tmp: str) -> int:
    """Render the jobs in folder tmp (jobs.json, *.mjb, qpos_*.npy) into tmp/out.npz; print 'ok\\t{info}' or
    'error\\t<reason>' as the last line."""
    mujoco = _import_mujoco()
    if mujoco is None:
        return 3
    state = _State()
    try:
        info = _render_folder(mujoco, tmp, state)
        for r in state.renderers.values():
            r.close()
    except Exception as e:                  # noqa: BLE001 — reported to the parent in words
        print(_error_line(e), flush=True)
        os._exit(3)
    print('ok\t' + json.dumps(info), flush=True)
    sys.stdout.flush()
    os._exit(0)                             # skip EGL teardown at exit (MuJoCo's deallocators print errors there)


def _serve() -> int:
    """The Session worker: one 'render\\t<folder>' per line on stdin, one 'ok\\t{info}' or 'error\\t<reason>' line per
    request on stdout; 'quit' or the end of stdin stops it."""
    mujoco = _import_mujoco()
    if mujoco is None:
        os._exit(3)
    state = _State()
    for line in sys.stdin:
        line = line.rstrip('\n')
        if line == 'quit':
            break
        if not line.startswith('render\t'):
            continue
        try:
            info = _render_folder(mujoco, line[len('render\t'):], state)
            print('ok\t' + json.dumps(info), flush=True)
        except Exception as e:              # noqa: BLE001 — reported to the parent in words
            print(_error_line(e), flush=True)
    sys.stdout.flush()
    os._exit(0)                             # skip EGL teardown at exit


if __name__ == '__main__':
    if sys.argv[1:2] == ['--serve']:
        raise SystemExit(_serve())
    raise SystemExit(_worker(sys.argv[1]))
