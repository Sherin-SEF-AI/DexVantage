"""`dexvantage --self-test`: checks an installation without showing a window and prints one line per check, in the
order a new user meets them.

    project      a project built, saved as .dexv, read back with the same content hash, and summarised
    frames       quaternion and rotation conversions agree with scipy's
    mujoco       MuJoCo loads a small model and computes its kinematics (the primary kinematics engine)
    robots       the bundled SO-100 and UR5e samples import, and MuJoCo and the independent product-of-exponentials
                 forward kinematics agree on every link
    library      every hardware library entry loads and validates against its schema; the UR5e sample's sensors
                 get their imagers from it
    occlusion    a cube map of a small scripted scene agrees with exact ray casting (says which backend ran: the
                 GPU or numpy)
    grasp        a 20-grasp run of the grasp self-occlusion study on the UR5e sample (batched IK, contact patches,
                 cube-map visibility) gives a Contact Visibility Score with its bootstrap interval
    brackets     the ISO 9409-1-50-4-M6 flange adapter disc meshes to one watertight, consistently wound STL that
                 reads back
    handeye      a synthetic AX = XB problem (exact poses) is solved back to the known camera-to-gripper transform
    opencv       the chessboard detector and the PNG codec
    fonts        the bundled IBM Plex fonts load
    opengl       the shader renderer draws a frame off-screen (a warning only: the 3D view falls back to QPainter)
    main window  the main window is built with a project and its summary study finishes (offscreen, never shown on
                 a desktop)
    software 3d  the QPainter 3D view, the fallback without OpenGL, draws the scene
    validation   the validation suite's quick run passes

Ported from Vantage selftest.py (the runner and the headless EGL context); the checks are DexVantage's. Nothing is
written outside a temporary folder and no network connection is made. The exit code is 0 when everything the app needs
works, 1 otherwise; a warning says what does not work here and what the app does instead.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import sys
import tempfile
import time
import traceback

OK, WARN, FAIL = 'ok', 'warning', 'FAILED'
# EGL platform enums (EGL_EXT_platform_device, EGL_MESA_platform_surfaceless); PyOpenGL has no constants for them
_EGL_PLATFORM_DEVICE = 0x313F
_EGL_PLATFORM_SURFACELESS = 0x31DD


# ------------------------------------------------------------------------------------------------------- OpenGL (EGL)
def _egl_try(get):
    """An EGL display from get() initialised with a pbuffer-capable OpenGL config, a context made current on it:
    (dpy, surf, ctx), or raises."""
    from OpenGL import EGL
    dpy = get()
    if not dpy or not EGL.eglInitialize(dpy, None, None):
        raise RuntimeError('no display')
    try:
        attrs = (EGL.EGLint * 13)(EGL.EGL_SURFACE_TYPE, EGL.EGL_PBUFFER_BIT, EGL.EGL_RED_SIZE, 8, EGL.EGL_GREEN_SIZE, 8,
                                 EGL.EGL_BLUE_SIZE, 8, EGL.EGL_DEPTH_SIZE, 24, EGL.EGL_RENDERABLE_TYPE,
                                 EGL.EGL_OPENGL_BIT, EGL.EGL_NONE)
        cfg, n = EGL.EGLConfig(), EGL.EGLint()
        if not EGL.eglChooseConfig(dpy, attrs, ctypes.pointer(cfg), 1, ctypes.pointer(n)) or n.value < 1:
            raise RuntimeError('no OpenGL config')
        surf = EGL.eglCreatePbufferSurface(dpy, cfg, (EGL.EGLint * 5)(EGL.EGL_WIDTH, 16, EGL.EGL_HEIGHT, 16,
                                                                       EGL.EGL_NONE))
        EGL.eglBindAPI(EGL.EGL_OPENGL_API)
        ctx = EGL.eglCreateContext(dpy, cfg, EGL.EGL_NO_CONTEXT, None)
        if not ctx or not EGL.eglMakeCurrent(dpy, surf, surf, ctx):
            raise RuntimeError('no OpenGL context')
        return dpy, surf, ctx
    except BaseException:
        EGL.eglTerminate(dpy)
        raise


def _egl_context():
    """A headless EGL OpenGL context (Linux, for the offscreen Qt platform, which has no OpenGL of its own). Tried in
    turn: the default display (a desktop session), Mesa's surfaceless platform (a GPU or, without one, the llvmpipe
    software renderer, as on a CI machine), then each EGL device (GPUs without a display server, NVIDIA's driver).
    Returns ((dpy, surf, ctx), how) or raises with what each attempt said."""
    from OpenGL import EGL
    tries = [('default display', lambda: EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)),
             ('surfaceless', lambda: EGL.eglGetPlatformDisplay(_EGL_PLATFORM_SURFACELESS, EGL.EGL_DEFAULT_DISPLAY,
                                                               None))]
    try:
        from OpenGL.EGL.EXT.device_enumeration import eglQueryDevicesEXT
        from OpenGL.EGL.EXT.platform_base import eglGetPlatformDisplayEXT
        devs, n = (EGL.EGLDeviceEXT * 8)(), EGL.EGLint()
        if eglQueryDevicesEXT(8, devs, ctypes.pointer(n)):
            tries += [(f'EGL device {i}', lambda d=devs[i]: eglGetPlatformDisplayEXT(_EGL_PLATFORM_DEVICE, d, None))
                      for i in range(n.value)]
    except Exception:                   # no device extension in this libEGL
        pass
    said = []
    for how, get in tries:
        try:
            return _egl_try(get), how
        except Exception as e:
            said.append(f'{how}: {_first_line(e)}')
    raise RuntimeError('; '.join(said) or 'no EGL')


def _egl_release(handles):
    from OpenGL import EGL
    dpy, surf, ctx = handles
    EGL.eglMakeCurrent(dpy, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT)
    EGL.eglDestroyContext(dpy, ctx)
    EGL.eglDestroySurface(dpy, surf)
    EGL.eglTerminate(dpy)


def _first_line(e) -> str:
    text = str(e).strip()
    if type(e).__name__ == 'EGLError':          # PyOpenGL's EGLError prints a multi-line repr: keep the error name
        import re
        m = re.search(r'err\s*=\s*(\w+)', text)
        return f'EGLError {m.group(1)}' if m else 'EGLError'
    return (text.splitlines() or [type(e).__name__])[0][:160]




# ------------------------------------------------------------------------------------------------------------ checks
def _check_project(S):
    from .core import jobs, provenance as PV
    from .validation.checks_formats import reference_project
    from .core.project import Project
    p = reference_project()
    path = os.path.join(S['tmp'], 'self-test.dexv')
    p.save(path)
    q = Project.load(path)
    a, b = PV.project_digest(p)['sha256'], PV.project_digest(q)['sha256']
    if a != b:
        return FAIL, 'a project read back from its .dexv file differs from the one saved'
    doc = jobs.run_study('summary', q)
    S['project'] = q
    return OK, (f'{len(q.robots)} robots, {doc["sensors"]["total"]} sensors, {len(doc["volumes"])} task volumes; '
                f'.dexv round trip identical ({os.path.getsize(path) / 1024:.0f} KB)')


def _check_frames(S):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from .core import frames as F
    rng = np.random.default_rng(7)
    Q = rng.normal(size=(200, 4))
    err = max(float(np.abs(F.quat_to_matrix(q) - Rotation.from_quat(q).as_matrix()).max()) for q in Q)
    if err > 1e-12:
        return FAIL, f'quaternion to matrix differs from scipy by {err:.2e}'
    return OK, f'200 rotations agree with scipy within {err:.1e}'


def _check_mujoco(S):
    import mujoco
    import numpy as np
    xml = ('<mujoco><worldbody><body name="l1"><joint name="j1" type="hinge" axis="0 0 1"/>'
           '<geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02"/><body name="l2" pos="0.3 0 0">'
           '<joint name="j2" type="hinge" axis="0 0 1"/><geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.02"/>'
           '<site name="tip" pos="0.25 0 0"/></body></body></worldbody></mujoco>')
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    d.qpos[:] = [0.4, -0.7]
    mujoco.mj_forward(m, d)
    tip = d.site('tip').xpos
    ref = np.array([0.3 * np.cos(0.4) + 0.25 * np.cos(-0.3), 0.3 * np.sin(0.4) + 0.25 * np.sin(-0.3), 0.0])
    err = float(np.abs(tip - ref).max())
    if err > 1e-9:
        return FAIL, f'MuJoCo {mujoco.__version__}: a planar 2R arm\'s tip is {err:.2e} m from the closed form'
    return OK, f'MuJoCo {mujoco.__version__}: 2R arm forward kinematics matches the closed form ({err:.0e} m)'


def _check_robots(S):
    """The bundled samples import from their files, MuJoCo compiles them, and the independent product-of-exponentials
    chain (core/robot/kinematics.py, its own XML reader) places every link where MuJoCo does at a few random poses."""
    import numpy as np
    from .core.robot.kinematics import PoEChain
    from .core.robot.model import RobotModel
    from .core.robot.samples import sample_project
    rng = np.random.default_rng(11)
    said, worst = [], 0.0
    for name in ('so100', 'ur5e'):
        p = sample_project(name)
        r = p.robots[0]
        model, chain = RobotModel(r, p.sources), PoEChain(r, p.sources)
        if set(model.joint_names) != set(chain.joint_names) or set(model.link_names) - set(chain.link_names):
            return FAIL, f'{r.name}: MuJoCo and the independent reader find different links or joints'
        lo, hi = np.maximum(model.lower, -np.pi), np.minimum(model.upper, np.pi)
        for q in rng.uniform(lo, hi, (5, model.nq)):
            d = dict(zip(model.joint_names, q))
            A, B = model.fk(q), chain.fk(np.array([d[n] for n in chain.joint_names]))
            worst = max(worst, max(float(np.abs(A[ln] - B[ln]).max()) for ln in model.link_names))
        said.append(f'{name} ({model.nq} joints)')
    if worst > 1e-9:
        return FAIL, f'MuJoCo and the product-of-exponentials chain differ by {worst:.1e} on a bundled sample'
    return OK, f'samples {", ".join(said)} import; MuJoCo and product-of-exponentials FK agree within {worst:.0e}'


def _check_library(S):
    """Every hardware library entry loads and validates against the library schema, and the UR5e sample's two
    RealSense sensors get their imagers from it (not the labelled default)."""
    from .core import library as L
    from .core.robot.samples import sample_project
    L.reload()
    if L.problems():
        return FAIL, f'{len(L.problems())} library files were skipped: {L.problems()[0]}'
    es = L.entries()
    bad = [(e.id, m) for e in es for m in L.validate(e.data)[:1]]
    if bad:
        return FAIL, f'{len(bad)} entries do not validate, first {bad[0][0]}: {bad[0][1]}'
    p = sample_project('ur5e')
    for sn in p.variant.sensors:
        ims = L.sensor_imagers(sn)
        if not ims or any(i.is_default for i in ims):
            return FAIL, f'the UR5e sample\'s "{sn.name}" has no library imagers ({sn.model_id or "no model id"})'
    by = {k: sum(e.kind == k for e in es) for k in L.KINDS}
    return OK, f'{len(es)} entries valid (' + ', '.join(f'{n} {k}' for k, n in by.items() if n) + \
        '); the UR5e sample\'s D405 and D435 resolve to their datasheet imagers'


def _check_occlusion(S):
    """A cube map of a small scripted scene (three boxes over a floor) against exact ray casting (core/raycast.py)
    on 20 000 random points: every disagreement must lie within two texels of a silhouette."""
    import numpy as np
    from .core import occlusion as OC, raycast as RC
    from .core.scenegeom import _box_tris

    def box(c, h):
        return _box_tris(np.asarray(h, float)) + np.asarray(c, float)
    tris = np.concatenate([box((0.4, 0.0, 0.15), (0.05, 0.08, 0.15)), box((0.3, 0.25, 0.1), (0.06, 0.06, 0.1)),
                           box((0.55, -0.2, 0.05), (0.1, 0.04, 0.05)), box((0.4, 0.0, -0.01), (0.6, 0.6, 0.01))])
    origin = np.array([0.0, 0.0, 0.6])
    rng = np.random.default_rng(5)
    pts = rng.uniform((0.1, -0.4, 0.0), (0.8, 0.4, 0.35), (20000, 3))
    n = 256
    t0 = time.perf_counter()
    cube = OC.CubeMap.build(tris, origin, n=n, backend='auto')
    ms = (time.perf_counter() - t0) * 1000
    got, ref = cube.visible(pts), RC.visible_exact(pts, origin, tris)
    diff = got != ref
    far = diff & ~RC.near_silhouette(pts, origin, tris, 2 * cube.texel_rad) if diff.any() else diff
    who = cube.backend if cube.backend == 'numpy' else f'gl on {cube.renderer}'
    if far.any():
        return FAIL, f'{int(far.sum())} of {len(pts)} points disagree with ray casting away from a silhouette ({who})'
    return OK, (f'cube map ({who}, n = {n}, {ms:.0f} ms) agrees with exact ray casting on {len(pts) - int(diff.sum())}'
                f' of {len(pts)} points; the {int(diff.sum())} others lie at silhouettes')


def _check_grasp(S):
    """A 20-grasp run of the grasp self-occlusion study on the UR5e sample: batched IK, the contact patches carried to
    each solution and their visibility from both cameras in cube maps, ending in a Contact Visibility Score with its
    bootstrap interval. The sensitivity sweep, the gallery and the pre-contact steps are left out to keep it fast."""
    from .core import grasps as GR, jobs
    from .core.robot.samples import sample_project
    p = sample_project('ur5e')
    p.tasks.grasp_sets = []
    GR.add_set(p, GR.sample(p, p.robots[0].uid, kinds=('top_down',), n=20, rng_seed=0))
    p.analysis.studies['grasp_visibility'] = {'sensitivity': False, 'gallery': 0, 'precontact_steps': 0,
                                              'bootstrap': 500, 'cube_n': 256}
    t0 = time.perf_counter()
    doc = jobs.run_study('grasp_visibility', p)
    ms = (time.perf_counter() - t0) * 1000
    s = doc.get('summary') or {}
    if s.get('cvs_pct') is None or not s.get('feasible'):
        return FAIL, f'no Contact Visibility Score ({s.get("feasible", 0)} of 20 grasps feasible)'
    oc = doc.get('occlusion') or {}
    who = oc.get('backend', '?') if oc.get('backend') != 'gl' else f'gl on {oc.get("renderer") or "the GPU"}'
    return OK, (f'{s["feasible"]} of 20 grasps feasible, Contact Visibility Score {s["cvs_pct"]:.0f} % '
                f'[{100 * s["ci_low"]:.0f}–{100 * s["ci_high"]:.0f}] (bootstrap 95 %); {oc.get("maps_built", 0)} '
                f'cube maps ({who}), {ms:.0f} ms')


def _check_handeye(S):
    """AX = XB on synthetic motions: a known camera-to-gripper transform X, 12 random gripper poses and the board poses
    they imply (exact), solved by Park-Martin (core/calib/axxb.py; OpenCV's calibrateHandEye when the installed
    OpenCV has it) must give X back within 10⁻⁶ m and 10⁻⁶ rad."""
    import numpy as np
    from .core.calib import axxb as AX
    rng = np.random.default_rng(3)
    X = AX.T_of(AX.exp_so3([0.3, -0.2, 0.9]), [0.04, -0.03, 0.11])               # camera in the gripper
    T_board = AX.T_of(AX.exp_so3([0.1, 0.4, -0.2]), [0.6, 0.1, 0.0])             # board in the base
    G = [AX.T_of(AX.exp_so3(rng.normal(0, 0.6, 3)), rng.uniform(-0.3, 0.3, 3) + [0.4, 0, 0.4]) for _ in range(12)]
    C = [AX.inv_T(g @ X) @ T_board for g in G]                                    # board in the camera
    t0 = time.perf_counter()
    Xs, backend = AX.hand_eye(G, C, 'park')
    ms = (time.perf_counter() - t0) * 1000
    dt, dr = AX.transform_error(Xs, X)
    if not (dt < 1e-6 and dr < 1e-6):
        return FAIL, f'Park-Martin ({backend}) is {dt:.1e} m and {dr:.1e} rad from the known X'
    return OK, f'Park-Martin ({backend}) gives X back within {dt:.1e} m and {dr:.1e} rad from 12 exact poses ' \
               f'({ms:.0f} ms)'


def _check_brackets(S):
    """One printable part: the ISO 9409-1-50-4-M6 flange adapter disc meshed at 0.5 mm must be a closed, consistently
    wound 2-manifold with positive volume (core/meshing.check), and its STL bytes must read back with the same
    triangle count."""
    from .core import brackets as BR, meshing as M
    t0 = time.perf_counter()
    solid, fig = BR.flange_plate()
    V, F = M.mesh(solid, 0.5)
    chk = M.check(V, F)
    data = M.stl_bytes(V, F, 'self-test')
    back = M.read_stl(data)
    ms = (time.perf_counter() - t0) * 1000
    n_back = len(back[1]) if isinstance(back, tuple) else len(back)
    if not chk['ok'] or n_back != chk['triangles']:
        return FAIL, (f'the flange disc is not watertight (watertight {chk["watertight"]}, consistent '
                      f'{chk["consistent"]}, {chk["degenerate"]} degenerate) or its STL does not read back')
    return OK, (f'ISO 9409-1-50-4-M6 flange disc: {chk["triangles"]} triangles, watertight, {chk["shells"]} shell, '
                f'{chk["volume_mm3"] / 1000:.1f} cm³, STL {len(data) / 1024:.0f} KB ({ms:.0f} ms)')


def _check_opencv(S):
    import cv2
    import numpy as np
    sq, nx, ny = 40, 9, 7
    img = np.full(((ny + 1) * sq + 80, (nx + 1) * sq + 80), 255, np.uint8)
    for r in range(ny + 1):
        for c in range(nx + 1):
            if (r + c) % 2 == 0:
                img[40 + r * sq:40 + (r + 1) * sq, 40 + c * sq:40 + (c + 1) * sq] = 0
    ok, corners = cv2.findChessboardCorners(img, (nx, ny))
    if not ok or len(corners) != nx * ny:
        return FAIL, f'OpenCV {cv2.__version__}: the chessboard detector found no board'
    good, buf = cv2.imencode('.png', img)
    back = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) if good else None
    if back is None or not np.array_equal(back, img):
        return FAIL, 'the PNG codec does not round-trip'
    return OK, f'OpenCV {cv2.__version__}: chessboard detector ({nx * ny} corners) and PNG codec'


def _check_fonts(S):
    from PyQt6.QtGui import QFontDatabase
    fams = set(QFontDatabase.families())
    miss = [f for f in ('IBM Plex Sans', 'IBM Plex Mono') if f not in fams]
    if miss:
        return FAIL, f'{", ".join(miss)} did not load'
    return OK, 'IBM Plex Sans and IBM Plex Mono loaded'


def _check_opengl(S):
    import numpy as np
    from PyQt6.QtGui import QGuiApplication
    plat = QGuiApplication.platformName()
    egl, how, qt = None, '', []
    if plat in ('offscreen', 'minimal'):
        if not sys.platform.startswith('linux'):
            return WARN, f'not checked: the {plat} platform has no OpenGL'
        try:
            egl, how = _egl_context()
        except Exception as e:
            return WARN, f'no headless OpenGL ({e}); the 3D view would use the software renderer'
    else:
        from PyQt6.QtGui import QOffscreenSurface, QOpenGLContext, QSurfaceFormat
        surf = QOffscreenSurface()
        surf.setFormat(QSurfaceFormat.defaultFormat())
        surf.create()
        ctx = QOpenGLContext()
        ctx.setFormat(QSurfaceFormat.defaultFormat())
        if not ctx.create() or not ctx.makeCurrent(surf):
            return WARN, 'no OpenGL context; the 3D view will use the software renderer'
        qt, how = [surf, ctx], 'Qt'
    try:
        from OpenGL import GL
        from .ui.render_gl import Frame, GLRenderer
        from .ui.scene import box_tris
        from .ui.viewport3d import MeshDraw, OrbitCam
        name = GL.glGetString(GL.GL_RENDERER)
        name = name.decode(errors='replace') if name else '?'
        r = GLRenderer()
        if not r.init():
            return WARN, f'{name}: shaders failed ({(r.error or "").splitlines()[0][:120]}); fixed-function GL is used'
        tri = box_tris((0.2, -0.2, 0.0), (0.6, 0.2, 0.4)).astype(np.float32)
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        md = MeshDraw(tri, n, np.full((len(tri), 3), 0.85), np.ones(len(tri)), version=1)
        cam = OrbitCam()
        W, H = 320, 200
        img = r.render_offscreen(Frame(cam.view(), cam.proj(W / H), W, H, target=cam.target.copy(), mesh=md,
                                       clear=(0.1, 0.1, 0.1, 1.0)))
        r.release()
    finally:
        if egl is not None:
            _egl_release(egl)
        if qt:
            qt[1].doneCurrent()
    lit = float((img[..., :3].mean(axis=2) > 120).mean())
    if lit < 0.02:
        return FAIL, f'{name}: the shader renderer drew an empty frame'
    S['gl_name'] = name
    return OK, f'shader renderer on {name} ({how})'


def _check_window(S):
    from .core.project import blank_project
    from .ui.mainwindow import MainWindow
    w = MainWindow(blank_project('Self-test'), soft_3d=True)
    S['window'] = w
    w.show()
    ok = w.ctx.analysis.wait(10.0)
    doc = w.ctx.analysis.results.get('summary')
    if not ok or doc is None:
        return FAIL, 'the summary study did not finish within 10 s'
    menus = [a.text().replace('&', '') for a in w.menuBar().actions()]
    return OK, f'{len(menus)} menus, {len(w.docks)} docks, {len(w.actions_by_id)} commands; summary in ' \
               f'{doc["seconds"] * 1000:.1f} ms'


def _check_software_3d(S):
    import numpy as np
    w = S.get('window')
    if w is None:
        return FAIL, 'no main window to draw'
    img = w.viewport.render_image(scale=1.0)
    a = np.frombuffer(img.constBits().asarray(img.sizeInBytes()), np.uint8).reshape(img.height(), img.bytesPerLine())
    spread = float(a[:, :img.width() * 4].std())
    if spread < 1.0:
        return FAIL, 'the QPainter 3D view drew a blank image'
    return OK, f'QPainter 3D view drew {img.width()} × {img.height()} px'


def _check_validation(S):
    from . import validation as VA
    res = VA.run(settings=VA.ValidationSettings(quick=True))
    if not res.ok:
        return FAIL, f'{len(res.failed)} of {len(res.checks)} checks failed: ' + ', '.join(c.id for c in res.failed[:4])
    return OK, f'{res.passed} of {len(res.checks)} checks passed (quick run, {res.seconds:.1f} s)'


CHECKS = (('project', _check_project), ('frames', _check_frames), ('mujoco', _check_mujoco), ('robots', _check_robots),
          ('library', _check_library), ('occlusion', _check_occlusion), ('grasp', _check_grasp),
          ('brackets', _check_brackets), ('handeye', _check_handeye), ('opencv', _check_opencv),
          ('fonts', _check_fonts), ('opengl', _check_opengl), ('main window', _check_window),
          ('software 3d', _check_software_3d), ('validation', _check_validation))
NAMES = tuple(n for n, _ in CHECKS)
BUNDLE_MAY_WARN = ('opengl',)


def run() -> int:
    from . import __version__
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors='replace')
        except (AttributeError, ValueError):
            pass
    plat = os.environ.get('QT_QPA_PLATFORM', '')
    if sys.platform.startswith('linux') and plat.startswith(('offscreen', 'minimal')):
        os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
        os.environ.setdefault('EGL_LOG_LEVEL', 'fatal')
    t0 = time.perf_counter()
    print(f'DexVantage {__version__} self-test · Python {sys.version.split()[0]} · {sys.platform}'
          + (' · bundled' if getattr(sys, 'frozen', False) else ''), flush=True)
    S = {'errors': [], 'tmp': tempfile.mkdtemp(prefix='dexvantage-self-test-')}
    from .app import make_app
    app, _ = make_app(settings_scope='self-test')
    hook = sys.excepthook
    sys.excepthook = lambda t, v, tb: S['errors'].append(''.join(traceback.format_exception(t, v, tb)))
    failed = False
    try:
        for name, fn in CHECKS:
            n0 = len(S['errors'])
            try:
                status, msg = fn(S)
            except Exception as e:
                status, msg = FAIL, f'{type(e).__name__}: {_first_line(e)}'
            if len(S['errors']) > n0 and status != FAIL:
                status, msg = FAIL, S['errors'][-1].strip().splitlines()[-1]
            failed |= status == FAIL
            print(f'  {name:<12} {status:<8} {msg}', flush=True)
    finally:
        sys.excepthook = hook
        w = S.pop('window', None)
        if w is not None:
            w.cancel_jobs()
            w.hide()                    # not close(): closing saves settings
            w.deleteLater()
            app.processEvents()
        shutil.rmtree(S['tmp'], ignore_errors=True)
    print(f'self-test {"FAILED" if failed else "passed"} in {time.perf_counter() - t0:.0f} s')
    sys.stdout.flush()
    del app
    return 1 if failed else 0
