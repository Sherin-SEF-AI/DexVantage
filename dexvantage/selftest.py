"""`dexvantage --self-test`: checks an installation without showing a window and prints one line per check, in the
order a new user meets them.

    project      a project built, saved as .dexv, read back with the same content hash, and summarised
    frames       quaternion and rotation conversions agree with scipy's
    mujoco       MuJoCo loads a small model and computes its kinematics (the primary kinematics engine)
    robots       the bundled SO-100 and UR5e samples import, and MuJoCo and the independent product-of-exponentials
                 forward kinematics agree on every link
    samples      the ALOHA 2, bimanual Panda and G1 samples open, and MuJoCo and the product-of-exponentials chain
                 agree on every link of their robots at home
    library      every hardware library entry loads and validates against its schema; the UR5e sample's sensors
                 get their imagers from it
    occlusion    a cube map of a small scripted scene agrees with exact ray casting (says which backend ran: the
                 GPU or numpy)
    grasp        a 20-grasp run of the grasp self-occlusion study on the UR5e sample (batched IK, contact patches,
                 cube-map visibility) gives a Contact Visibility Score with its bootstrap interval
    brackets     the ISO 9409-1-50-4-M6 flange adapter disc meshes to one watertight, consistently wound STL that
                 reads back
    groundtruth  the UR5e sample's MuJoCo scene builds, its cameras' intrinsics match the lens models
    handeye      a synthetic AX = XB problem (exact poses) is solved back to the known camera-to-gripper transform
    teleop       the teleop study's IK and dexterity on a planar 2R arm against the closed form
    lighting     the lighting study on the UR5e sample with a can, a library key light and a window
    timing       the motion blur, time sync and action-chunk studies on the UR5e sample give their headlines
    datasets     LeRobot v2.1 and v3.0 and an MCAP bag written and read back (independent readers)
    audit        the mount-slip classifier against Vantage's recorded outputs
    experiments  a Saltelli design's Sobol indices of the Ishigami function against the analytic ones, and a
                 resolution IV fraction's main effects of a linear model with interactions
    statistics   the Benjamini-Hochberg (1995) example's rejections and a G*Power paired-t power, and the
                 evaluation episodes to tell two success rates apart
    opencv       the chessboard detector and the PNG codec
    fonts        the bundled IBM Plex fonts load
    opengl       the shader renderer draws a frame off-screen (a warning only: the 3D view falls back to QPainter)
    main window  the main window is built with a project and its summary study finishes (offscreen, never shown on
                 a desktop)
    software 3d  the QPainter 3D view, the fallback without OpenGL, draws the scene
    validation   the validation suite's quick run of its cheap checks passes (--validate runs every check)
    splat        the splat views' rasteriser (numpy reference against its closed form, torch against numpy),
                 reprojection and SSIM against closed forms; a warning saying the splat extra is not installed where
                 PyTorch is not (the downloadable app leaves it out)

Ported from Vantage selftest.py (the runner and the headless EGL context); the checks are DexVantage's. Nothing is
written outside a temporary folder and no network connection is made. The exit code is 0 when everything the app needs
works, 1 otherwise; a warning says what does not work here and what the app does instead. A check that needs an
optional extra (OPTIONAL_PACKAGES) that is not installed is a warning naming the extra, not a failure: a plain
`pip install dexvantage` has none of them, and the downloadable bundles carry the ones these checks use, so there it
stays a failure (packaging/build.py allows warnings only in BUNDLE_MAY_WARN).
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
# optional packages some checks need, and the extra of pyproject.toml that installs each
OPTIONAL_PACKAGES = {'pyarrow': 'lerobot', 'av': 'lerobot', 'zstandard': 'bags', 'lz4': 'bags', 'jsonschema': 'test',
                     'mcap': 'test', 'google.protobuf': 'test'}
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


def _check_new_samples(S):
    """The ALOHA 2, bimanual Panda and G1 samples open (robots, cameras, tasks), and every link of their robots at
    home is where the independent product-of-exponentials chain puts it."""
    import numpy as np
    from .core.robot.kinematics import PoEChain
    from .core.robot.model import get_model
    from .core.robot.samples import sample_project
    said, worst = [], 0.0
    for name in ('aloha', 'panda-bimanual', 'g1'):
        p = sample_project(name)
        if not (p.variant.sensors and p.tasks.grasp_sets and p.tasks.trajectories):
            return FAIL, f'the {name} sample opened without its cameras, grasp sets or trajectories'
        for r in p.robots:
            model, chain = get_model(p, r.uid), PoEChain(r, p.sources)
            d = dict(zip(model.joint_names, model.home))
            A, B = model.fk(model.home), chain.fk(np.array([d[n] for n in chain.joint_names]))
            worst = max(worst, max(float(np.abs(A[ln] - B[ln]).max()) for ln in model.link_names))
        nr, ns = len(p.robots), len(p.variant.sensors)
        said.append(f'{name} ({nr} robot{"s" if nr != 1 else ""}, {ns} camera{"s" if ns != 1 else ""})')
    if worst > 1e-9:
        return FAIL, f'MuJoCo and the product-of-exponentials chain differ by {worst:.1e} on a new sample'
    return OK, f'{", ".join(said)} open; MuJoCo and product-of-exponentials FK agree within {worst:.0e}'


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


def _check_library_teleop(S):
    """The library's teleop devices are consistent, and the SO-100 leader arm entry makes a teleop device that leads
    the SO-100 sample's robot with its own joints (core/teleop_devices.py)."""
    from .core import teleop_devices as TD
    from .core.robot.samples import sample_project
    es = TD.entries()
    bad = [(e.id, m) for e in es for m in TD.entry_problems(e.data)[:1]]
    if not es or bad:
        return FAIL, (f'{bad[0][0]}: {bad[0][1]}' if bad else 'the library has no teleop devices')
    p = sample_project('so100')
    d = TD.device_from_library(p, 'so100-leader-arm')
    if (d.get('leader') or {}).get('robot_uid') != p.robots[0].uid or d.get('library_id') != 'so100-leader-arm':
        return FAIL, f'the SO-100 leader arm entry did not lead the SO-100 sample\'s robot: {d.get("leader")}'
    return OK, f'{len(es)} teleop devices consistent; the SO-100 leader arm leads the SO-100 sample joint to joint'


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


def _check_groundtruth(S):
    """The MuJoCo ground-truth scene of the UR5e sample (core/sim/scene.py), built only (no render, to stay fast):
    the robot attached from its stored files, one camera per imager whose intrinsics read back from the compiled model
    (validation/mjrender.camera_model) match the imager's lens model within 10⁻³ px."""
    from .core.robot.samples import sample_project
    from .core.sim.scene import build_scene
    from .validation import mjrender as MR
    p = sample_project('ur5e')
    t0 = time.perf_counter()
    b = build_scene(p)
    ms = (time.perf_counter() - t0) * 1000
    worst = 0.0
    for c in b.cameras.values():
        cm = MR.camera_model(b.model, c.name, c.model.width, c.model.height)
        worst = max(worst, abs(cm.fx - c.model.fx), abs(cm.fy - c.model.fy), abs(cm.cx - c.model.cx),
                    abs(cm.cy - c.model.cy))
    if not b.cameras or worst > 1e-3:
        return FAIL, f'{len(b.cameras)} scene cameras, intrinsics {worst:.1e} px from the lens models'
    return OK, (f'UR5e scene: {b.model.nbody} bodies, {b.model.ngeom} geoms, {len(b.cameras)} cameras within '
                f'{worst:.0e} px of their lens models ({ms:.0f} ms)')


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


def _run_checks(ids) -> list:
    """Run validation checks by id (quick settings, the built-in sample) and return their results."""
    from .validation import suite as VS
    VS.load_checks()
    by_id = {sp.id: sp for sp in VS.CHECKS}
    ctx = VS.Ctx(None, None, None, VS.ValidationSettings(quick=True))
    try:
        return [VS.run_one(by_id[i], ctx) for i in ids]
    finally:
        ctx.close()


def _optional_missing(*texts) -> str | None:
    """The optional package (OPTIONAL_PACKAGES) that a failure's texts say is missing, when it really is not
    installed here; else None."""
    import importlib.util
    import re
    text = ' '.join(str(t) for t in texts if t)
    for mod in OPTIONAL_PACKAGES:
        name = mod.split('.')[-1]
        if re.search(rf"needs the Python package {name}\b|No module named '{re.escape(mod)}|import of {re.escape(mod)}"
                     rf"\b|\b{name} is not installed", text):
            try:
                if importlib.util.find_spec(mod) is None:
                    return mod
            except (ImportError, ValueError):
                return mod
    return None


def _check_missing(c) -> str | None:
    """The optional package a validation check result failed for want of, or None."""
    d = c.detail if isinstance(c.detail, dict) else {}
    return _optional_missing(c.exception, c.note, d.get('skipped'))


def _not_checked(mods) -> tuple:
    mods = sorted(set(mods))
    extras = sorted({OPTIONAL_PACKAGES[m] for m in mods})
    return WARN, (f'not checked: needs {", ".join(mods)}, not installed here (pip install '
                  f'"dexvantage[{",".join(extras)}]")')


def _checks_line(ids, ok_text) -> tuple:
    t0 = time.perf_counter()
    res = _run_checks(ids)
    ms = (time.perf_counter() - t0) * 1000
    bad = [c for c in res if not c.passed]
    missing = [_check_missing(c) for c in bad]
    if bad and all(missing):
        return _not_checked(missing)
    bad = [c for c, m in zip(bad, missing) if not m]
    if bad:
        c = bad[0]
        why = c.exception or f'error {c.error:.3g} {c.unit} against {c.tolerance:.3g}'
        return FAIL, f'{c.id}: {why}'
    return OK, ok_text(res) + f' ({ms:.0f} ms)'


def _check_censored(S):
    """The experiments' best run with figures stated only as bounds, against brute force over the figures the bounds
    allow (validation check experiments.censored_best, quick)."""
    return _checks_line(['experiments.censored_best'],
                        lambda r: f'{r[0].detail["trials"]} designs, {r[0].detail["settled_picks"]} settled picks and '
                                  f'{r[0].detail["verdicts_decided"]} decided verdicts, none contradicted')


def _check_sim2real(S):
    """The sim-to-real gap's measures: the histogram EMD against SciPy, the unbiased MMD² by hand and against the
    closed form for two Gaussians, Immerkær's noise σ and the blur σ on images of known noise and blur (validation
    checks sim2real.*, quick, without the rendered recordings)."""
    return _checks_line(['sim2real.emd_vs_scipy', 'sim2real.mmd_hand_case', 'sim2real.mmd_gaussians',
                         'sim2real.noise_known', 'sim2real.blur_known'],
                        lambda r: f'EMD within {r[0].error:.0e} DN of SciPy; MMD² by hand {r[1].error:.0e}, two '
                                  f'Gaussians within {r[2].error:.1e}; noise σ within {100 * r[3].error:.1f} %; '
                                  f'blur σ within {r[4].error:.2f} px')


def _check_cells(S):
    """The multi-cell comparison's gap index on synthetic rigs (identical: 0; a camera turned by θ: θ) and the drift of
    copies with known offsets (validation checks cells.gap_identical, cells.gap_rotation, cells.drift_known, quick)."""
    return _checks_line(['cells.gap_identical', 'cells.gap_rotation', 'cells.drift_known'],
                        lambda r: f'identical cells gap {r[0].error:.0e}; turned cameras within {r[1].error:.0e}° of '
                                  f'the angle; drift offsets within {r[2].error:.0e}')


def _check_tracking(S):
    """The experiment tracker's run store round trip, its reading after writers that died mid-write and its CSV and
    Parquet tables (validation checks tracking.store_round_trip, tracking.crash_recovery,
    tracking.csv_parquet_identical, quick)."""
    return _checks_line(['tracking.store_round_trip', 'tracking.crash_recovery', 'tracking.csv_parquet_identical'],
                        lambda r: f'{r[0].detail["fields_compared"]} run fields read back exactly; '
                                  f'{len(r[1].detail["cases"])} crash cases read as they should; '
                                  f'{r[2].detail["cells"]} table cells identical in CSV and Parquet')


def _check_tolerance(S):
    """The tolerance Monte Carlo's seed reproducibility, its rank correlation against SciPy and the perturbation
    export's round trip (validation checks tolerance.seed_reproducible, tolerance.spearman, tolerance.export_roundtrip,
    quick)."""
    return _checks_line(['tolerance.seed_reproducible', 'tolerance.spearman', 'tolerance.export_roundtrip'],
                        lambda r: f'seeded draws reproduce, ρ within {r[1].error:.0e} of SciPy, exported extrinsics '
                                  f'within {r[2].error:.0e} of the moved mounts')


def _check_versioning(S):
    """The project diff lists exactly 24 seeded edits, the unpacked folder round trip is byte-stable and the SVG
    figure's data comes back through its tick labels (validation checks versioning.diff_lists_edits,
    versioning.unpack_round_trip, versioning.svg_data_recovery, quick)."""
    return _checks_line(['versioning.diff_lists_edits', 'versioning.unpack_round_trip', 'versioning.svg_data_recovery'],
                        lambda r: f'diff lists {r[0].detail["edits"]} of {r[0].detail["edits"]} edits, '
                                  f'{r[1].detail["files"]} folder files byte-stable, SVG data within '
                                  f'{r[2].error:.0e} of range')


def _check_publication(S):
    """The paper kit of the sample: every number of its methods paragraph found in the source data, and its cited
    keys equal to refs.bib's (validation checks publication.methods_numbers, publication.bibtex_keys, quick)."""
    return _checks_line(['publication.methods_numbers', 'publication.bibtex_keys'],
                        lambda r: f'{r[0].detail["numbers"]} numbers of the methods paragraph found in the source '
                                  f'data, {r[1].detail["cited"]} cited keys all in refs.bib')


def _check_campaign(S):
    """The campaign planner's Williams squares, its schedule balance and session limits on seeded random campaigns and
    its calendar read back by a strict RFC 5545 reader (validation checks campaign.williams_square,
    campaign.schedule_balance, campaign.session_limits, campaign.ics_roundtrip, quick)."""
    return _checks_line(['campaign.williams_square', 'campaign.schedule_balance', 'campaign.session_limits',
                         'campaign.ics_roundtrip'],
                        lambda r: f'Williams squares 1-12 balanced, {r[1].detail["campaigns"]} campaigns balanced '
                                  f'and within their limits, {r[3].detail["events"]} calendar events read back')


def _check_workspace(S):
    """The workspace check's object boxes against closed forms and its inside-or-outside verdicts on random scenes
    (validation checks workspace.object_extent, workspace.inside_classification, quick)."""
    return _checks_line(['workspace.object_extent', 'workspace.inside_classification'],
                        lambda r: f'object boxes within {r[0].error:.2f} of the tessellation bound; '
                                  f'{r[1].detail["objects"] - int(r[1].error)} of {r[1].detail["objects"]} objects '
                                  'placed inside or outside as they are')


def _check_requirements(S):
    """The readiness gate's exit codes on synthetic pass, fail and error cases and the bound rule against a Monte
    Carlo (validation checks requirements.gate_exit_codes, requirements.bound_decisions, quick)."""
    return _checks_line(['requirements.gate_exit_codes', 'requirements.bound_decisions'],
                        lambda r: f'--check exits 0, 1 and 2 as it should on {len(r[0].detail["cases"])} cases; '
                                  f'bounds decide as a Monte Carlo over {r[1].detail["cases"]} cases does')


def _check_teleop(S):
    """The teleop study's follower IK and dexterity (core/studies/teleop.Arm) on a planar 2R arm against the closed
    form w = |l₁ l₂ sin q₂| (validation check teleop.manipulability_2r, quick)."""
    return _checks_line(['teleop.manipulability_2r'],
                        lambda r: f'2R arm: manipulability within {r[0].error:.0e} m² of the closed form, IK solved '
                                  f'{r[0].detail.get("ik_solved", 0)} of {r[0].detail.get("ik_points", 0)} points')


def _check_lighting(S):
    """The lighting study on the UR5e sample with a metallic can, an Elgato Key Light (library figures) and a window
    (the validation checks' scene): light sources, the microfacet materials and cube-map visibility end in a headline
    glare figure, said to be relative when no camera states its sensitivity."""
    from .core import jobs
    from .validation.checks_lighting import lit_sample
    p = lit_sample()
    t0 = time.perf_counter()
    doc = jobs.run_study('lighting', p)
    ms = (time.perf_counter() - t0) * 1000
    h = doc.get('headline') or {}
    if h.get('value') is None:
        return FAIL, f'no glare figure: {h.get("text") or "the study gave no headline"}'
    rel = ' relative' if h.get('relative') else ''
    return OK, f'{len(p.workspace.lights)} lights, glare {h["value"]:.1f} %{rel} ({h.get("word")}), {ms:.0f} ms'


def _check_timing(S):
    """The motion blur, time sync and action-chunk latency studies on the UR5e sample (its scripted pick and π0):
    each gives a headline with a status word; figures not stated by a maker stay bounds."""
    from .core import jobs
    from .core.robot.samples import sample_project
    p = sample_project('ur5e')
    t0 = time.perf_counter()
    out = []
    for sid, key, fmt in (('motion_blur', 'smear_px', 'smear {:.1f} px'), ('time_sync', 'worst_skew_s', None),
                          ('action_chunk_sync', 'worst_mm', 'chunk {:.1f} mm')):
        doc = jobs.run_study(sid, p)
        h = doc.get('headline') or doc.get('summary') or {}
        word = h.get('word') or h.get('status_word')
        if not word:
            return FAIL, f'{jobs.STUDIES[sid].title} gave no headline'
        v = h.get(key)
        if sid == 'time_sync':
            out.append(f'skew {1000 * v:.1f} ms' if v is not None else 'skew not stated')
        else:
            out.append(fmt.format(v) if v is not None else f'{key} not stated')
        out[-1] += f' ({word})'
    ms = (time.perf_counter() - t0) * 1000
    return OK, 'UR5e: ' + ', '.join(out) + f'; {ms:.0f} ms'


def _check_datasets(S):
    """Dataset formats: three synthetic episodes written as LeRobot v2.1 and v3.0 and read back by DexVantage's reader
    and independently (pyarrow, ffmpeg), and an MCAP bag round trip (validation checks formats.lerobot_roundtrip and
    formats.mcap_roundtrip)."""
    return _checks_line(['formats.lerobot_roundtrip', 'formats.mcap_roundtrip'],
                        lambda r: 'LeRobot v2.1 and v3.0 and an MCAP bag read back with 0 mismatches')


def _check_optimiser(S):
    """NSGA-II (core/optimise/nsga2.py) on ZDT1, 30 genes, 6 000 evaluations, against the analytic front (IGD, the
    validation check optimise.zdt1_igd runs 25 000), and the same seed giving the same archive."""
    import numpy as np

    from .core.optimise import nsga2 as N
    t0 = time.perf_counter()
    p = N.benchmark_problem('zdt1')
    a, b = N.run(p, pop=60, budget=6000, seed=0), N.run(p, pop=60, budget=6000, seed=0)
    g = N.igd(a.F[a.front()], N.zdt_front('zdt1'))
    ms = (time.perf_counter() - t0) * 1000
    if not np.array_equal(a.archive_F, b.archive_F):
        return FAIL, 'the same seed gave two different searches'
    if g > 0.05:
        return FAIL, f'ZDT1 front {g:.4f} from the analytic front (IGD; 0.05 allowed at 6 000 evaluations)'
    return OK, f'ZDT1 IGD {g:.4f} at 6 000 evaluations, seed reproducible; {ms:.0f} ms'


def _check_audit(S):
    """The auditor's mount-slip classifier (core/audit/slip.py, ported from Vantage) against Vantage's original
    outputs on twelve recorded series (validation check audit.slip_vs_vantage)."""
    return _checks_line(['audit.slip_vs_vantage'],
                        lambda r: f'mount-slip classifier within {r[0].error:.0e} of Vantage\'s recorded outputs')


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


def _check_experiments(S):
    import math
    import time

    import numpy as np
    from .core import experiments as EX
    from .validation.checks_experiments import ishigami, ishigami_indices
    t0 = time.perf_counter()
    d = {'factors': [{'path': f'x{i}', 'lo': -math.pi, 'hi': math.pi} for i in (1, 2, 3)],
         'design': {'method': 'saltelli', 'n': 1024, 'seed': 1}, 'analysis': {'bootstrap': 100}}
    s = EX.run_function(ishigami, d)['analysis']['y']['sobol']
    S1, ST = ishigami_indices()
    e = float(max(np.abs(np.array(s['S1']) - S1).max(), np.abs(np.array(s['ST']) - ST).max()))
    coef = np.array([1.0, -2.0, 0.5, 3.0, 1.5])
    f2 = {'factors': [{'path': f'x{i}', 'lo': -1, 'hi': 1} for i in range(5)],
          'design': {'method': 'fractional', 'resolution': 'IV'}}
    r = EX.run_function(lambda v: float(sum(c * v[f'x{i}'] for i, c in enumerate(coef)) + v['x0'] * v['x1']), f2)
    e2 = float(np.abs(np.array([m['effect'] for m in r['analysis']['y']['main_effects']]) - 2 * coef).max())
    ms = (time.perf_counter() - t0) * 1000
    if e > 2 / math.sqrt(1024) or e2 > 1e-9:
        return FAIL, f'Sobol indices {e:.3f} from the analytic ones, main effects {e2:.2g} from the coefficients'
    return OK, (f'Ishigami Sobol indices within {e:.3f} of the analytic ones (N = 1024); a 16-run resolution IV '
                f'fraction recovers the main effects exactly ({ms:.0f} ms)')


def _check_stats(S):
    import time

    from .core import stats as ST
    t0 = time.perf_counter()
    p = (0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344, 0.0459, 0.3240, 0.4262, 0.5719, 0.6528,
         0.7590, 1.000)                                  # Benjamini and Hochberg (1995), p. 295
    k_bh, k_bonf = sum(ST.adjust(p, 'bh')['reject']), sum(ST.adjust(p, 'bonferroni')['reject'])
    pw = ST.power_t(0.421637, 50, 0.05, paired=True)       # the G*Power 3.1 manual, p. 48: 0.832114
    n = ST.episodes_needed(0.5, 0.8)
    ms = (time.perf_counter() - t0) * 1000
    if (k_bh, k_bonf) != (4, 3) or abs(pw - 0.832114) > 5e-6:
        return FAIL, (f'Benjamini-Hochberg example: {k_bh} and {k_bonf} rejections (published 4 and 3); paired t '
                      f'power {pw:.6f} (G*Power 0.832114)')
    return OK, (f'the Benjamini-Hochberg example rejects 4 (Bonferroni 3) as published; paired t power {pw:.6f} as '
                f'G*Power; {n["n_per_group"]} episodes per condition tell 50 % from 80 % apart ({ms:.0f} ms)')

def _check_mcp(S):
    import json
    import time

    from .assistant.session import FileSession
    from .core.project import blank_project
    from .mcp.server import PROTOCOL, Router, Server
    t0 = time.perf_counter()
    router = Router(mode='file')
    router.file = FileSession(project=blank_project('Self-test'))
    srv = Server(router)
    init = srv.handle({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'protocolVersion': PROTOCOL}})
    tools = srv.handle({'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'})['result']['tools']
    r = srv.handle({'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call', 'params': {'name': 'get_overview',
                                                                                 'arguments': {}}})['result']
    ov = json.loads(r['content'][0]['text'])
    ms = (time.perf_counter() - t0) * 1000
    if init['result']['protocolVersion'] != PROTOCOL or r['isError'] or ov.get('project') != 'Self-test':
        return FAIL, 'the MCP server did not answer initialize, tools/list and tools/call as it should'
    return OK, f'MCP {PROTOCOL} server: {len(tools)} tools listed, get_overview answered ({ms:.0f} ms)'


def _check_warp(S):
    """The canonical view warp (core/exporters/canonical_warp.py): H carries plane points exactly and the remap through
    a Brown-Conrady and a fisheye lens agrees with cv2.projectPoints; the Rerun bridge says whether it can run."""
    from .core.bridges import rerun as RB
    from .core.exporters import canonical_warp as CW
    e_h, e_lut = CW.self_test()
    if not (e_h < 1e-9 and e_lut < 1e-6):
        return FAIL, f'plane points off by {e_h:.2g} px (homography) and {e_lut:.2g} px (remap with lenses)'
    return OK, (f'homography on the plane {e_h:.1g} px, remap through two lenses {e_lut:.1g} px against OpenCV; '
                + ('Rerun bridge ready (rerun-sdk installed)' if RB.available() else
                   'Rerun bridge off (rerun-sdk not installed: optional)'))


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
    res = VA.run(settings=VA.ValidationSettings(quick=True, cheap_only=True))
    missing = {c.id: _check_missing(c) for c in res.failed}
    failed = [c for c in res.failed if not missing[c.id]]
    if failed:
        return FAIL, f'{len(failed)} of {len(res.checks)} checks failed: ' + ', '.join(c.id for c in failed[:4])
    text = (f'{res.passed} of {len(res.checks)} cheap checks passed (quick run, {res.seconds:.1f} s; the '
            f'study-scale checks run with --validate)')
    if missing:
        mods = sorted(set(missing.values()))
        extras = sorted({OPTIONAL_PACKAGES[m] for m in mods})
        return WARN, (f'{text}; {len(missing)} not run: they need {", ".join(mods)}, not installed here (pip install '
                      f'"dexvantage[{",".join(extras)}]")')
    return OK, text


def _check_assistant(S):
    """The Assistant dock's loop with the fake provider (no network, no key): a tool turn, an edit proposal that
    changes nothing until accepted, a long study refused without an approved plan, the number post-check."""
    import json as _json

    from .assistant import agent as A
    from .assistant import toolbox as TB
    from .assistant.fake import FakeReply
    from .core.robot.samples import sample_project
    t0 = time.perf_counter()
    p = sample_project('so100')
    tc = TB.ToolContext(project_name=p.name, variant_name=p.variant.name)
    tc.session = TB.file_session(p.snapshot(), tc)
    sid = p.variant.sensors[0].uid
    s = A.new_session('fake', None, A.Settings('fake-model'), script=[[
        FakeReply(calls=[('get_overview', {}), ('run_study', {'study': 'grasp_visibility'}),
                         ('propose_changes', {'title': 'self-test', 'changes': [{'factor': f'sensor:{sid}.z',
                                                                                 'value': 0.05}]})]),
        FakeReply(text='Proposed z = 0.05 m; 99.9 % is a made-up figure.')]])
    ev = []
    s.run_turn('self-test', tc, ev.append)
    done = [e.data for e in ev if e.kind == 'tool_done']
    ans = next((e.data for e in ev if e.kind == 'answer'), None)
    if len(done) != 3 or ans is None:
        return FAIL, f'the turn ended early ({[e.kind for e in ev][-3:]})'
    if not done[0]['ok'] or not _json.loads(done[1]['result']).get('plan_required') or len(tc.proposals) != 1:
        return FAIL, 'a tool, the plan gate or the proposal did not behave'
    if [u['text'] for u in ans['check']['unsupported']] != ['99.9']:
        return FAIL, f'the number check flagged {ans["check"]["unsupported"]} (99.9 expected)'
    return OK, (f'fake provider: 3 tools, 1 proposal, a long study refused, 1 invented figure flagged '
                f'({(time.perf_counter() - t0) * 1000:.0f} ms)')


def _check_exporters(S):
    """The ROS 2 and OpenCV exporters on the UR5e sample: the URDF compiled by MuJoCo poses every camera frame as
    DexVantage does, and an OpenCV YAML reads back exactly (validation checks formats.urdf_roundtrip and
    formats.opencv_yaml run the full comparison)."""
    from .validation.formats_exports import selftest_export
    ok, text = selftest_export(os.path.join(S['tmp'], 'export'))
    return (OK if ok else FAIL), text


def _check_isaac_lerobot(S):
    """The Isaac and LeRobot exporters on the SO-100 sample: the Isaac Lab configs executed against stubs of the cited
    source and compared with the cameras and the tolerance draws, the USD prims re-read, the LeRobot info.json read back
    (validation checks formats.isaac_usd, formats.isaac_config and formats.lerobot_export run the full comparison)."""
    from .validation.formats_isaac_lerobot import selftest_isaac_lerobot
    ok, text = selftest_isaac_lerobot(os.path.join(S['tmp'], 'export_b'))
    missing = None if ok else _optional_missing(text)
    if missing:
        return _not_checked([missing])
    return (OK if ok else FAIL), text


def _check_robot_io(S):
    """The exported URDF loads in MuJoCo as written and the USD import round-trips the SO-100 (validation checks
    robot_io.urdf_mujoco and robot_io.usd_roundtrip run the full comparison)."""
    from .validation.checks_robot_io import selftest_robot_io
    ok, text = selftest_robot_io(os.path.join(S['tmp'], 'robot_io'))
    return (OK if ok else FAIL), text


def _check_api(S):
    """The Python API (dexvantage/api.py) on the SO-100 sample: a study, a metric and a factor through dv, the same
    figures as core.jobs gives directly, and the plugins found at start-up (dexvantage/plugin_sdk.py)."""
    import time
    from . import api as dv
    from . import plugin_sdk as PL
    from .core import jobs
    t0 = time.perf_counter()
    rig = dv.sample('so100')
    res = rig.run('coverage')
    direct = jobs.run_study('coverage', rig.project.snapshot())
    same = res['summary'] == direct['summary']
    z = f'sensor:{rig.sensors[-1].name}.z'
    moved = rig.with_factors({z: rig.factors[z] + 0.05})
    ok = same and abs(moved.factors[z] - rig.factors[z] - 0.05) < 1e-12
    recs = PL.loaded()
    bad = [r for r in recs if r.status == 'error']
    plug = (f'{sum(r.status == "loaded" for r in recs)} of {len(recs)} plugins loaded' if recs else 'no plugins')
    ms = (time.perf_counter() - t0) * 1000
    text = (f'SO-100 coverage {res.headline["text"].split(": ", 1)[-1]} through dv, '
            f'{"the same as" if same else "DIFFERENT from"} core.jobs; {plug}; {ms:.0f} ms')
    return (OK if ok and not bad else WARN if ok else FAIL), text + (f'. {bad[0].error}' if bad else '')


def _check_robot_catalogue(S):
    """The robot catalogue (core/robot/catalogue.py): every entry loads, its Menagerie folder is a known one and its
    DoF agrees with its verified import record and its maker's figure (a data-consistency check; no network)."""
    from .core import library as L
    from .core.robot import catalogue as CAT, menagerie as MG
    L.reload()
    es = CAT.entries()
    if not es:
        return FAIL, 'the robot catalogue has no entries'
    unknown = [e.id for e in es if CAT.model(e).get('folder') not in MG.OFFLINE_MODELS]
    if unknown:
        return FAIL, f'{len(unknown)} entries name a Menagerie folder DexVantage does not know, first {unknown[0]}'
    bad = [p for e in es for p in CAT.dof_problems(e)]
    if bad:
        return FAIL, f'{len(bad)} DoF inconsistencies, first: {bad[0]}'
    stated = sum(CAT.dof_matches(CAT.figure(e, 'dof')[0], CAT.measured_dof(CAT.verified(e))) is True for e in es)
    return OK, f'{len(es)} robots; {stated} DoF as the maker states them agree with the imported model'


def _check_cluster(S):
    """Cluster sweeps (core/cluster): shards cover every item once, the disk cache refuses a torn file, the array
    script's #SBATCH lines, and a work of study jobs run as 3 shards and gathered equals the studies run directly."""
    import json
    import os
    import time

    from .core import cluster as C
    from .core import jobs
    from .core.cluster import cache as CA
    from .core.cluster import slurm as SL
    from .core.project import blank_project
    t0 = time.perf_counter()
    bad = sum(sorted(i for sh in C.partition(m, n) for i in sh) != list(range(m)) for m in range(60) for n in (1, 2, 7))
    folder = os.path.join(S['tmp'], 'cluster')
    c = C.DiskResultCache(os.path.join(folder, 'cache'))
    key = ('summary', 'p', 's', 'l', 0, '')
    c.put(key, {'x': [1.5, 2]})
    ok_cache = C.DiskResultCache(os.path.join(folder, 'cache')).get(key) == {'x': [1.5, 2]}
    with open(c.path(key), 'r+b') as fh:                 # a torn file must read as a miss
        fh.truncate(os.path.getsize(c.path(key)) - 3)
    ok_cache = ok_cache and CA.read_entry(c.path(key), key) is None
    d = dict(SL.directives(SL.sbatch_script(SL.read_settings({'mem': '6G', 'throttle': 2}), 5)))
    ok_sb = d.get('--array') == '0-4%2' and d.get('--mem') == '6G'
    projs = [blank_project(f'Cell {k}') for k in range(3)]
    w = C.studies_work([(p, 'summary', {}) for p in projs])
    C.slurm_folder(w, os.path.join(folder, 'job'), shards=3)
    for i in range(3):
        C.run_shard(os.path.join(folder, 'job', 'work.json'), i, 3)
    got = C.gather(os.path.join(folder, 'job'), write=False)['result']['jobs']
    want = [jobs.run_study('summary', p) for p in projs]
    same = all(json.dumps({k: v for k, v in g['doc'].items() if k != 'seconds'}, sort_keys=True) ==
               json.dumps({k: v for k, v in x.items() if k not in ('seconds', 'cached')}, sort_keys=True)
               for g, x in zip(got, want))
    ms = (time.perf_counter() - t0) * 1000
    if bad or not ok_cache or not ok_sb or not same:
        return FAIL, (f'partition failures {bad}, cache {"ok" if ok_cache else "wrong"}, sbatch '
                      f'{"ok" if ok_sb else "wrong"}, gathered {"equal" if same else "different"}')
    return OK, (f'shards cover every item once; a torn cache file reads as a miss; 3 shards gathered equal the '
                f'studies run directly ({ms:.0f} ms)')


def _check_splat(S):
    """Splat views (core/sim/splat_views.py): validation checks splat.footprint_closed_form, raster_torch_cpu,
    reproject_disparity and ssim_closed_form (quick); without PyTorch, a warning that the splat extra is not
    installed."""
    from .core.sim import splat_views as SV
    why = SV.requirement()
    if why is not None:
        from . import extras
        if extras.frozen():                 # the downloadable app: pip cannot add PyTorch to it (extras.missing)
            return WARN, 'splat extra not installed: ' + why[0].lower() + why[1:].rstrip('.')
        return WARN, f'splat extra not installed: splat views need PyTorch ({SV.INSTALL})'
    return _checks_line(['splat.footprint_closed_form', 'splat.raster_torch_cpu', 'splat.reproject_disparity',
                         'splat.ssim_closed_form'],
                        lambda r: f'footprint within {r[0].error:.0e} of the closed form, torch within '
                                  f'{r[1].error:.0e} of the numpy rasteriser, disparity within {r[2].error:.0e} px, '
                                  f'SSIM within {r[3].error:.0e}')


def _check_prices(S):
    """Cited prices: every built-in price block sound (core/prices.problems) and the rig cost of a two-camera rig
    against its two stated prices added up by hand (D405 + D435)."""
    from .core import library as LB
    from .core import prices as PR
    from .core.project import Project, Sensor
    from .core.studies import rig_inventory as RI
    t0 = time.perf_counter()
    blocks = [(e.id, PR.block(e)) for e in LB.entries() if not e.is_custom and PR.block(e) is not None]
    bad = [i for i, p in blocks if PR.problems(p)]
    p = Project('Self-test')
    p.variant.sensors = [Sensor(name='a', model_id='intel-realsense-d405'),
                         Sensor(name='b', model_id='intel-realsense-d435')]
    want = sum(LB.get(s.model_id).data['price']['amount'] for s in p.variant.sensors)
    c = RI.inventory(p)['cost']
    ms = (time.perf_counter() - t0) * 1000
    if bad or c['value'] is None or abs(c['value'] - want) > 0.005 or c['bound']:
        return FAIL, f'price problems in {", ".join(bad[:4]) or "none"}; rig cost {c["value"]} against {want}'
    return OK, f'{len(blocks)} cited prices sound; a D405 + D435 rig costs {c["value"]:.2f} {c["unit"]} ({ms:.0f} ms)'


CHECKS = (('project', _check_project), ('frames', _check_frames), ('mujoco', _check_mujoco), ('robots', _check_robots),
          ('samples', _check_new_samples),
          ('library', _check_library), ('robot catalogue', _check_robot_catalogue),
          ('library teleop', _check_library_teleop), ('prices', _check_prices),
          ('workspace', _check_workspace), ('occlusion', _check_occlusion),
          ('grasp', _check_grasp),
          ('groundtruth', _check_groundtruth), ('brackets', _check_brackets), ('handeye', _check_handeye),
          ('teleop', _check_teleop), ('lighting', _check_lighting), ('timing', _check_timing),
          ('datasets', _check_datasets), ('audit', _check_audit), ('tolerance', _check_tolerance),
          ('experiments', _check_experiments), ('censored bounds', _check_censored),
          ('optimiser', _check_optimiser), ('cluster', _check_cluster),
          ('requirements', _check_requirements), ('versioning', _check_versioning),
          ('publication', _check_publication),
          ('campaign', _check_campaign),
          ('cells', _check_cells),
          ('sim2real', _check_sim2real),
          ('statistics', _check_stats),
          ('mcp server', _check_mcp), ('assistant', _check_assistant), ('opencv', _check_opencv),
          ('exporters', _check_exporters), ('sim exports', _check_isaac_lerobot), ('warp', _check_warp),
          ('robot io', _check_robot_io),
          ('python api', _check_api), ('tracking', _check_tracking),
          ('fonts', _check_fonts), ('opengl', _check_opengl), ('main window', _check_window),
          ('software 3d', _check_software_3d), ('validation', _check_validation), ('splat', _check_splat))
NAMES = tuple(n for n, _ in CHECKS)
BUNDLE_MAY_WARN = ('opengl', 'splat')      # splat: the downloadable app leaves out the splat extra (PyTorch)


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
                missing = _optional_missing(f'{type(e).__name__}: {e}')
                status, msg = _not_checked([missing]) if missing else (FAIL, f'{type(e).__name__}: {_first_line(e)}')
            if len(S['errors']) > n0 and status != FAIL:
                status, msg = FAIL, S['errors'][-1].strip().splitlines()[-1]
            failed |= status == FAIL
            print(f'  {name:<15} {status:<8} {msg}', flush=True)
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
