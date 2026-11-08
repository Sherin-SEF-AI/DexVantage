"""Measure DexVantage's performance on this machine (spec §16 performance budgets) and write docs/performance.md.

    systemd-run --user --scope -q -p MemoryMax=7G env QT_QPA_PLATFORM=offscreen \\
        .venv/bin/python scripts/benchmarks.py [--only grasp,studies,...] [--samples ur5e,aloha] [--grasps 2000]
                                               [--egl 'EGL device 0'] [--out out/bench] [--doc docs/performance.md]
    .venv/bin/python scripts/benchmarks.py --doc-only        # remake docs/performance.md from out/bench/*.json

Each part runs in child processes, one at a time, so a figure's peak memory is its own: the largest resident set the
kernel reports for the child when it ends (os.wait4, ru_maxrss; it includes the few MB the parent's interpreter had
when it forked). The parts:

  environment  CPU, GPU (the EGL device the occlusion backend opens), RAM, OS, Python and package versions; never
               the host name or the user name
  import       'import dexvantage as dv' in a fresh interpreter from the source tree (no PyInstaller bundle), 7 runs
  startup      the window: process start to the main window's first paint (offscreen), a blank project and each
               sample, 3 runs each, through dexvantage.app.main as a user starts it
  grasp        a 2,000-grasp study on each sample's primary arm (grasps sampled with core/grasps.sample, top-down,
               side and angled, seed 0; the study's default settings, sensitivity and gallery on)
  profile      when a sample misses the grasp budget: the slowest one again under cProfile, where the time goes
  studies      coverage and depth validity on every sample: the first run in a fresh process and a repeat
  cubemap      the occlusion cube-map build for every camera of every sample, both backends, 256 and 512 texels
  ui           the slowest UI-thread events while opening each sample and running its studies from their pages (the
               watchdog and timing helpers of scripts/measure_ui_phase3.py)
  splat        splat views on the so100 and ur5e samples (core/sim/splat_views.py; needs the splat extra and CUDA):
               the study's fit with its default settings on a MuJoCo render of the sample, the render rate through
               the target camera at its stream size, and the GPU memory PyTorch allocated and reserved at its peak
  validate     a full 'dexvantage --validate out/validation' run; its report then makes docs/validation.md
               (scripts/gen_validation_doc.py)

Results go to out/bench/<part>.json; docs/performance.md is made from those files only. The machine may be shared:
each part records the 1-minute load average before and after it, and the page states it.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
SAMPLES = ('so100', 'ur5e', 'dexhand', 'aloha', 'panda-bimanual', 'g1')
PARTS = ('environment', 'import', 'startup', 'grasp', 'profile', 'studies', 'cubemap', 'ui', 'splat', 'validate')
UI_BUDGET_MS = 100.0
STALL_MS = 50.0                   # UI-thread stalls this long or longer are named with their stack (ui part)
# The 2,000-grasp budget used here (a project default, not a figure from the specification, which asks only for a
# measured, documented time): 180 s on this machine's GPU. Why: spec §18 asks that a researcher sees which grasps
# lose fingertip visibility, moves a camera and sees the score change within ten minutes of installing, which runs
# the grasp study at least twice beside the rest of the tour; three minutes a run leaves four minutes for the rest.
GRASP_BUDGET_S = 180.0
# the studies the ui part runs on each sample: every study except those that need inputs a sample does not carry or
# that run many studies in turn (said in docs/performance.md)
UI_SKIP = {
    'optimise': 'runs the coverage and grasp studies for every design (minutes; its page was timed in Phase 9)',
    'design_agent': 'runs the optimiser in rounds (minutes)',
    'tolerance': 'runs the studies once per Monte Carlo draw (minutes; its page was timed in Phase 9)',
    'experiments': 'needs factors and responses chosen on its page (timed in Phase 9 with a Saltelli design)',
    'groundtruth': 'renders every camera with MuJoCo (timed by the validation run below)',
    'audit': 'needs a recorded dataset', 'plan_vs_recording': 'needs a recorded dataset',
    'silhouette_calib': 'needs a recorded dataset', 'sim2real': 'needs a recorded dataset',
    'handeye_solve': 'needs captured board images',
    'cells': 'needs a second cell', 'stats': 'needs success counts', 'campaign': 'needs a campaign spec',
    'tracking': 'needs a run store', 'variants_compare': 'needs a second variant',
    'splat_views': 'fits a 3D Gaussian Splatting rig on the GPU (about a minute; timed by the splat part below)',
}
SECRET_ENV = ('ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'GROQ_API_KEY', 'HF_TOKEN', 'GITHUB_TOKEN')


# ============================================================================ machine (stdlib only, no host name)
def _read(path: str) -> str:
    try:
        with open(path, encoding='utf-8', errors='replace') as fh:
            return fh.read()
    except OSError:
        return ''


def machine() -> dict:
    """CPU, RAM, OS and the NVIDIA GPU if nvidia-smi answers; nothing that names the host or the user."""
    cpu = ''
    for ln in _read('/proc/cpuinfo').splitlines():
        if ln.lower().startswith('model name'):
            cpu = ln.split(':', 1)[1].strip()
            break
    cpu = cpu or platform.processor() or platform.machine()
    cores = None
    try:
        out = subprocess.run(['lscpu', '-p=core,socket'], capture_output=True, text=True, timeout=10).stdout
        cores = len({ln for ln in out.splitlines() if ln and not ln.startswith('#')}) or None
    except (OSError, subprocess.SubprocessError):
        pass
    mem = None
    for ln in _read('/proc/meminfo').splitlines():
        if ln.startswith('MemTotal:'):
            mem = round(int(ln.split()[1]) * 1024 / 1e9, 1)
    osname = ''
    try:
        osname = platform.freedesktop_os_release().get('PRETTY_NAME', '')
    except (OSError, AttributeError):
        pass
    gpus = []
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=20).stdout
        for ln in out.strip().splitlines():
            parts = [x.strip() for x in ln.split(',')]
            if len(parts) == 3:
                gpus.append({'name': parts[0], 'driver': parts[1], 'memory': parts[2]})
    except (OSError, subprocess.SubprocessError):
        pass
    return {'cpu': cpu, 'logical_cpus': os.cpu_count(), 'physical_cores': cores, 'ram_gb': mem,
            'os': osname or platform.system(), 'kernel': f'{platform.system()} {platform.release()}',
            'machine': platform.machine(), 'nvidia_smi': gpus}


def load() -> float | None:
    try:
        return round(os.getloadavg()[0], 2)
    except OSError:
        return None


# ============================================================================ running a child
def child_env(tmp: str, egl: str) -> dict:
    """The environment a child runs in: throw-away XDG folders, offscreen Qt, the EGL device, no API keys."""
    env = {k: v for k, v in os.environ.items() if k not in SECRET_ENV}
    for k, sub in (('XDG_CONFIG_HOME', 'c'), ('XDG_DATA_HOME', 'd'), ('XDG_CACHE_HOME', 'k')):
        env[k] = os.path.join(tmp, sub)
        os.makedirs(env[k], exist_ok=True)
    env['TMPDIR'] = tmp
    env['QT_QPA_PLATFORM'] = 'offscreen'
    env['PYTHONPATH'] = ROOT + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    if egl:
        env['DEXVANTAGE_EGL'] = egl
    return env


def run_child(cmd: list, log: str, egl: str, timeout: float, extra_env: dict | None = None) -> dict:
    """Run cmd to its end in a throw-away folder; {'exit', 'wall_s', 'peak_mb', 'timed_out', 'load_before',
    'load_after'}."""
    tmp = tempfile.mkdtemp(prefix='bench-', dir=os.environ.get('TMPDIR') or None)
    env = child_env(tmp, egl)
    env.update(extra_env or {})
    before = load()
    t0 = time.perf_counter()
    timed_out = False
    try:
        with open(log, 'a', encoding='utf-8') as fh:
            fh.write(f'\n$ {" ".join(os.path.basename(c) if os.path.isabs(c) else c for c in cmd)}\n')
            fh.flush()
            pr = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
            while True:
                pid, status, ru = os.wait4(pr.pid, os.WNOHANG)
                if pid:
                    break
                if time.perf_counter() - t0 > timeout:
                    timed_out = True
                    pr.kill()
                time.sleep(0.05)
            pr.returncode = os.waitstatus_to_exitcode(status)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {'exit': pr.returncode, 'wall_s': round(time.perf_counter() - t0, 3),
            'peak_mb': round(ru.ru_maxrss * 1024 / 1e6, 1), 'timed_out': timed_out,
            'load_before': before, 'load_after': load()}


def py_child(part: str, args: list, out_dir: str, egl: str, timeout: float, tag: str) -> tuple[dict, dict | None]:
    """Run one of this script's child parts; (the run's figures, the JSON the child wrote or None)."""
    res = os.path.join(out_dir, 'logs', f'{part}-{tag}.result.json')
    if os.path.exists(res):
        os.remove(res)
    cmd = [sys.executable, os.path.abspath(__file__), '--child', part, '--result', res, *args]
    run = run_child(cmd, os.path.join(out_dir, 'logs', f'{part}.log'), egl, timeout,
                    {'BENCH_SPAWN': repr(time.time())})
    doc = None
    if os.path.exists(res):
        with open(res, encoding='utf-8') as fh:
            doc = json.load(fh)
        os.remove(res)
    return run, doc


# ============================================================================ the children
def _write(path: str, doc: dict):
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=1, default=str)


def c_environment(a) -> dict:
    import numpy as np
    out = {'python': platform.python_version(), 'implementation': platform.python_implementation()}
    vers = {}
    for name, mod in (('numpy', 'numpy'), ('scipy', 'scipy'), ('mujoco', 'mujoco'), ('opencv', 'cv2'),
                      ('PyOpenGL', 'OpenGL'), ('trimesh', 'trimesh')):
        try:
            m = __import__(mod)
            vers[name] = getattr(m, '__version__', None)
        except Exception:                                  # noqa: BLE001 — reported as not installed
            vers[name] = None
    try:
        from PyQt6.QtCore import PYQT_VERSION_STR, QT_VERSION_STR
        vers['Qt'], vers['PyQt6'] = QT_VERSION_STR, PYQT_VERSION_STR
    except Exception:                                      # noqa: BLE001
        vers['Qt'] = vers['PyQt6'] = None
    import dexvantage
    vers['dexvantage'] = dexvantage.__version__
    out['packages'] = vers
    try:
        deps = np.show_config(mode='dicts').get('Build Dependencies', {})
        out['blas'] = ', '.join(sorted({f"{c.get('name', '?')} {c.get('version', '')}".strip()
                                        for c in deps.values() if isinstance(c, dict)})) or None
    except Exception:                                      # noqa: BLE001
        out['blas'] = None
    from dexvantage.core import occlusion as OC
    out['gl'] = OC.gl_status()
    try:
        r = subprocess.run(['git', 'rev-parse', '--short=10', 'HEAD'], cwd=ROOT, capture_output=True, text=True,
                           timeout=10)
        d = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, capture_output=True,
                           text=True, timeout=30)
        out['git'] = {'commit': r.stdout.strip() or None, 'modified': bool(d.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        out['git'] = None
    return out


def c_grasp(a) -> dict:
    from dexvantage.core import grasps as GR
    from dexvantage.core import jobs
    from dexvantage.core.robot.samples import sample_project
    from dexvantage.core.studies import grasp_visibility as GV
    p = sample_project(a.sample)
    p.tasks.grasp_sets = []                       # the sample's own set left out: a.grasps sampled grasps
    r = p.robots[0]
    t0 = time.perf_counter()
    gs = GR.sample(p, r.uid, kinds=('top_down', 'side', 'angled'), n=a.grasps, rng_seed=0)
    GR.add_set(p, gs)
    t_sample = time.perf_counter() - t0
    p.analysis.studies[GV.SID] = {}
    t0 = time.perf_counter()
    doc = jobs.run_study(GV.SID, p, jobs.Ctx())
    wall = time.perf_counter() - t0
    s, oc, th, tm = doc['summary'], doc['occlusion'], doc['throughput'], doc['timings']
    return {'sample': a.sample, 'robot': r.name, 'robots_in_sample': len(p.robots),
            'cameras': len(p.variant.sensors), 'grasps': a.grasps, 'seed': 0, 'renderer': oc['renderer'],
            'backend': oc['backend'], 'gl_how': oc.get('gl_how'), 'cube_n': oc['cube_n'],
            'feasible': s['feasible'], 'cvs': s['cvs'], 'ci': [s['ci_low'], s['ci_high']],
            'sampling_s': round(t_sample, 3), 'study_s': round(wall, 3), 'timings': tm,
            'grasps_per_s': th['grasps_per_s'], 'maps_built': oc['maps_built'], 'map_seconds': oc['map_seconds'],
            'maps_per_s': th['maps_per_s'], 'cache_hit_rate': oc['cache_hit_rate']}


def c_profile(a) -> dict:
    """c_grasp under cProfile: the functions with the most time of their own and the DexVantage functions with the
    most time in all (paths relative to the repository or to site-packages)."""
    import cProfile
    import pstats
    pr = cProfile.Profile()
    pr.enable()
    doc = c_grasp(a)
    pr.disable()
    st = pstats.Stats(pr)
    total = st.total_tt

    def where(k):
        fn, ln, name = k
        fn = os.path.normpath(fn) if os.path.isabs(fn) else fn
        if fn.startswith(ROOT + os.sep):
            fn = os.path.relpath(fn, ROOT)
        elif 'site-packages' + os.sep in fn:
            fn = fn.split('site-packages' + os.sep, 1)[1]
        elif fn.startswith(os.sep):
            fn = os.path.basename(fn)
        return f'{fn}:{ln} {name}'
    rows = [(k, v[2], v[3], v[1]) for k, v in st.stats.items()]       # key, own s, cumulative s, calls
    own = sorted(rows, key=lambda r: -r[1])[:15]
    ours = sorted((r for r in rows if os.path.normpath(r[0][0]).startswith(os.path.join(ROOT, 'dexvantage') + os.sep)),
                  key=lambda r: -r[2])[:20]
    return {'grasp': doc, 'profiled_s': round(total, 1),
            'own_time': [{'function': where(k), 'own_s': round(o, 2), 'cumulative_s': round(c, 2), 'calls': n}
                         for k, o, c, n in own],
            'cumulative': [{'function': where(k), 'own_s': round(o, 2), 'cumulative_s': round(c, 2), 'calls': n}
                           for k, o, c, n in ours]}


def c_studies(a) -> dict:
    from dexvantage.core import jobs
    from dexvantage.core.robot.samples import sample_project
    p = sample_project(a.sample)
    runs = []
    for k in range(2):
        t0 = time.perf_counter()
        doc = jobs.run_study(a.study, p, jobs.Ctx())
        runs.append(time.perf_counter() - t0)
    out = {'sample': a.sample, 'study': a.study, 'first_s': round(runs[0], 4), 'repeat_s': round(runs[1], 4),
           'cameras': len(p.variant.sensors)}
    if a.study == 'coverage':
        sm = doc.get('summary') or {}
        out['seen_ge1'] = sm.get('seen_ge1')
        out['voxels'] = sum(int(v.get('voxels', 0) or 0) for v in (doc.get('volumes') or [])
                            if isinstance(v, dict)) or None
        out['timings'] = doc.get('timings')
    else:
        sens = doc.get('sensors') or []
        out['depth_sensors'] = len(sens)
        out['skipped'] = len(doc.get('skipped') or [])
        out['message'] = doc.get('message') if not sens else ''
    return out


def c_cubemap(a) -> dict:
    import numpy as np
    from dexvantage.core import occlusion as OC
    from dexvantage.core import scenegeom as SG
    from dexvantage.core.robot.samples import sample_project
    st = OC.gl_status()
    bes = ('gl', 'numpy') if st['available'] else ('numpy',)
    rows = []
    for name in a.samples.split(','):
        p = sample_project(name)
        occ = SG.scene_occluders(p, {})
        for s in p.variant.sensors:
            o = SG.sensor_world_pose(p, s, {}).position
            so = SG.sensor_occluders(p, s, {}, base=occ)
            row = {'sample': name, 'camera': s.name, 'triangles': len(so)}
            for be in bes:
                for n in (256, 512):
                    t0 = time.perf_counter()
                    OC.CubeMap.build(so, o, n, be)           # the first: geometry upload, face directions
                    first = time.perf_counter() - t0
                    reps = []
                    for _ in range(10 if be == 'gl' else 3):
                        t0 = time.perf_counter()
                        OC.CubeMap.build(so, o, n, be)
                        reps.append(time.perf_counter() - t0)
                    row[f'{be}_{n}'] = {'first_ms': round(first * 1000, 2),
                                        'median_ms': round(float(np.median(reps)) * 1000, 2),
                                        'min_ms': round(min(reps) * 1000, 2), 'reps': len(reps)}
            rows.append(row)
    return {'gl': st, 'rows': rows}


def c_startup(a) -> dict:
    spawn = float(os.environ.get('BENCH_SPAWN', time.time()))
    entry = time.time()
    import dexvantage.app as A
    marks: dict = {}
    keep: list = []
    orig = A.make_app

    def make_app(*args, **kw):
        from PyQt6.QtCore import QEvent, QObject, QTimer
        from PyQt6.QtWidgets import QApplication, QWidget
        app, settings = orig(*args, **kw)
        marks['app'] = time.time()

        class FirstPaint(QObject):
            def eventFilter(self, obj, ev):
                if 'paint' not in marks and ev.type() == QEvent.Type.Paint and isinstance(obj, QWidget):
                    from dexvantage.ui.mainwindow import MainWindow
                    if isinstance(obj.window(), MainWindow):
                        marks['paint'] = time.time()
                        # the filter is taken off (a Python filter alive at teardown crashes sip), and the
                        # window closed as a user closes it
                        QApplication.instance().removeEventFilter(self)
                        QTimer.singleShot(0, obj.window().close)
                return False
        f = FirstPaint()
        keep.append(f)
        app.installEventFilter(f)
        QTimer.singleShot(120000, QApplication.instance().quit)      # no paint in two minutes: give up
        return app, settings
    A.make_app = make_app
    rc = A.main(['--sample', a.sample] if a.sample != 'blank' else [])
    out = {'sample': a.sample, 'exit': rc, 'interpreter_s': round(entry - spawn, 3),
           'app_s': round(marks['app'] - spawn, 3) if 'app' in marks else None,
           'first_paint_s': round(marks['paint'] - spawn, 3) if 'paint' in marks else None}
    return out


class StallSampler:
    """Names the UI-thread stalls: a 5 ms timer on the UI thread marks each tick; a thread looks at the UI thread's
    Python stack whenever a tick is 30 ms late, and a gap of STALL_MS or more between ticks (while active) is kept
    with the stack seen most often during it (the DexVantage frames innermost first, paths relative to the
    repository)."""

    def __init__(self):
        import collections
        import threading

        from PyQt6.QtCore import QTimer
        self._C = collections.Counter
        self.lock = threading.Lock()
        self.main = threading.main_thread().ident
        self.tick, self.cpu = time.perf_counter(), time.thread_time()
        self.active = False
        self.label = ''
        self.cur = None
        self.events: list = []
        self.stop = False
        self.gcs: list = []                       # (end time, ms, generation) of each cycle collection
        self._gc0 = None
        import gc
        gc.callbacks.append(self._gc)
        self.timer = QTimer()
        self.timer.setInterval(5)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        threading.Thread(target=self._run, daemon=True).start()

    def _gc(self, phase, info):
        if phase == 'start':
            self._gc0 = time.perf_counter()
        elif self._gc0 is not None:
            now = time.perf_counter()
            self.gcs.append((now, (now - self._gc0) * 1000, info.get('generation')))
            self._gc0 = None
            del self.gcs[:-200]

    def _tick(self):
        now, cpu = time.perf_counter(), time.thread_time()       # the UI thread's own CPU time
        gap = (now - self.tick) * 1000
        start, self.tick = self.tick, now
        cpu_ms, self.cpu = (cpu - self.cpu) * 1000, cpu
        with self.lock:
            cur, self.cur = self.cur, None
        if self.active and gap >= STALL_MS:
            where = cur.most_common(1)[0][0] if cur else 'not sampled'
            # the cycle collections that ran during the stall (on any thread: they hold the GIL)
            gcs = [(ms, g) for t, ms, g in self.gcs if start <= t <= now and ms >= 1.0]
            self.events.append({'ms': round(gap, 1), 'during': self.label, 'where': where,
                                'ui_cpu_ms': round(cpu_ms, 1), 'gc_ms': round(sum(ms for ms, _g in gcs), 1),
                                'gc_generations': sorted({g for _ms, g in gcs})})

    def _run(self):
        while not self.stop:
            time.sleep(0.005)
            if time.perf_counter() - self.tick > 0.03:
                f = sys._current_frames().get(self.main)
                s = describe_stack(f)
                with self.lock:
                    if self.cur is None:
                        self.cur = self._C()
                    self.cur[s] += 1

    def close(self):
        import gc
        self.stop = True
        self.timer.stop()
        if self._gc in gc.callbacks:
            gc.callbacks.remove(self._gc)


def describe_stack(f) -> str:
    """The innermost three DexVantage frames of a stack ('dexvantage/ui/x.py:12 fn') and the innermost UI frame
    that called them, else the innermost library frame; never an absolute path."""
    ours, lib = [], None
    pkg = os.path.join(ROOT, 'dexvantage') + os.sep
    while f is not None:
        fn = os.path.normpath(f.f_code.co_filename)       # a sys.path entry such as scripts/.. stays in the name
        if fn.startswith(pkg):
            ours.append(f'{os.path.relpath(fn, ROOT)}:{f.f_lineno} {f.f_code.co_name}')
        elif lib is None and 'site-packages' in fn:
            lib = f'{fn.split("site-packages" + os.sep, 1)[1]}:{f.f_lineno} {f.f_code.co_name}'
        f = f.f_back
    if ours:
        ui = next((x for x in ours if x.startswith('dexvantage/ui/')), None)   # the UI code that called it
        return (' < '.join(ours[:3]) + (f' … < {ui}' if ui and ui not in ours[:3] else '')
                + (f' (in {lib})' if lib else ''))
    return ('outside DexVantage\'s Python code (Qt\'s own work: layout, paint or delivering queued signals'
            + (f'; {lib}' if lib else '') + ')')


def c_ui(a) -> dict:  # noqa: C901 — the steps in order
    sys.path.insert(0, HERE)
    import traceback

    import measure_ui_phase3 as M            # the application, the watchdog and the timing helpers
    from PyQt6.QtCore import QEvent, QObject

    from dexvantage.core import jobs
    from dexvantage.core.robot.samples import sample_project
    from dexvantage.ui.mainwindow import MainWindow
    jobs.load_studies()
    name = a.sample
    S = StallSampler()
    painted = []

    class FirstPaint(QObject):
        def eventFilter(self, obj, ev):
            if not painted and ev.type() == QEvent.Type.Paint:
                painted.append(time.perf_counter())
            return False
    fp = FirstPaint()
    M.app.installEventFilter(fp)
    t0 = time.perf_counter()
    w = MainWindow(sample_project(name), soft_3d=True, settings=M.settings)
    w.resize(1680, 1000)
    w.show()
    M.record(name, 'open the sample: build and show the main window (start-up)', (time.perf_counter() - t0) * 1000,
             'start')
    t1 = time.perf_counter()
    while not painted and time.perf_counter() - t1 < 30:
        M.app.processEvents()
    M.app.removeEventFilter(fp)
    M.record(name, 'open the sample: show to the first paint (start-up)', (time.perf_counter() - t1) * 1000, 'start')
    vp = w.viewport
    type(vp).paintEvent = lambda self, e: None     # the QPainter 3D fallback is offscreen-only (see the docs)
    stall, _t = watch_until(M, S, w, None, 60, 'after the first paint, the live studies running')
    M.record(name, 'events after the first paint, the live studies running (watchdog stall)', stall)
    failures: dict = {}
    w.ctx.analysis.failed.connect(lambda sid, msg: failures.__setitem__(sid, msg.split('\n')[0][:300]))
    by_study = {pg.STUDY: pid for pid, pg in w.pages.items() if getattr(pg, 'STUDY', None)}
    reports = [0]                                  # progress reports a study sends (counted on its worker thread)
    report = jobs.Ctx.report

    def counted(self, frac, text=''):
        reports[0] += 1
        return report(self, frac, text)
    jobs.Ctx.report = counted
    ran, skipped = [], {}
    for sid, info in jobs.STUDIES.items():
        if sid in UI_SKIP:
            skipped[sid] = UI_SKIP[sid]
            continue
        if info.needs_robot and not w.store.project.robots:
            skipped[sid] = 'needs a robot'
            continue
        pid = by_study.get(sid)
        try:
            if pid is not None:
                M.record(name, f'{pid}: show the page', M.timed(lambda: (w.open_page(pid), M.app.processEvents())))
                stall, _t = watch_until(M, S, w, '', 0.3, f'{pid}: after showing the page')
                M.record(name, f'{pid}: events after showing (watchdog stall)', stall)
            w.ctx.analysis.results.pop(sid, None)
            reports[0] = 0
            t1 = time.perf_counter()
            if pid is not None:
                w.pages[pid].run()
            else:
                w.ctx.analysis.run(sid)
            stall, t_done = watch_until(M, S, w, sid, 900, f'{sid}: the study running and its result arriving')
            whole = (t_done - t1) * 1000
            M.record(name, f'{sid}: study, whole run', whole, 'worker')
            M.record(name, f'{sid}: events while the study runs and its result arrives (watchdog stall)', stall)
            doc = w.ctx.analysis.results.get(sid)
            if pid is not None and doc is not None:
                page = w.pages[pid]
                M.record(name, f'{pid}: the result arrives (show_result)', M.timed(lambda: page._finished(sid, doc)))
                M.record(name, f'{pid}: 3D overlays for one frame', M.timed(lambda: page.overlays(vp)))
            ran.append({'study': sid, 'page': pid, 'seconds': round(whole / 1000, 3), 'result': doc is not None,
                        'refused': failures.get(sid, ''), 'progress_reports': reports[0]})
        except Exception:                          # noqa: BLE001 — report and go on with the next study
            traceback.print_exc()
            M.record(name, f'{sid}: FAILED to measure', 0.0, 'error')
            ran.append({'study': sid, 'page': pid, 'seconds': None, 'result': False, 'refused': 'not measured'})
    S.close()
    w.cancel_jobs()
    w.hide()
    w.deleteLater()
    M.app.processEvents()
    return {'sample': name, 'budget_ms': M.BUDGET_MS, 'rows': M.ROWS, 'studies': ran, 'skipped': skipped,
            'stalls': sorted(S.events, key=lambda e: -e['ms'])[:40], 'stall_ms': STALL_MS}


def watch_until(M, S, w, sid, timeout: float, label: str) -> tuple[float, float]:
    """Process events with the watchdog and the stall sampler running until study sid is done (None: every study;
    '': for timeout seconds), then for half a second more (the events the result causes); (the longest stall in ms,
    the time the study was seen done)."""
    WD = M.WD
    WD.worst = 0.0
    WD.clock.invalidate()
    WD.t.start()
    with S.lock:
        S.tick, S.cur = time.perf_counter(), None      # what ran before the watch is not a stall
        S.cpu = time.thread_time()
    S.label, S.active = label, True
    t0 = time.perf_counter()
    an = w.ctx.analysis
    t_done = None
    try:
        while time.perf_counter() - t0 < timeout:
            M.app.processEvents()
            if sid is None:
                done = not an.busy() and not an.timer.isActive()
            elif sid == '':
                done = False
            else:
                done = not an.busy(sid) and (sid in an.results or time.perf_counter() - t0 > 0.5)
            if done:
                t_done = time.perf_counter()
                break
            time.sleep(0.001)
        t1 = time.perf_counter()
        while sid != '' and time.perf_counter() - t1 < 0.5:
            M.app.processEvents()
            time.sleep(0.001)
    finally:
        WD.t.stop()
        S.active = False
    return WD.worst, (t_done or time.perf_counter())


def c_splat(a) -> dict:
    import torch

    from dexvantage.core.robot.samples import sample_project
    from dexvantage.core.sim import splat_views as SV
    p = sample_project(a.sample)
    out = {'sample': a.sample, 'torch': torch.__version__, 'cuda': torch.version.cuda,
           'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    doc = SV.analyse(p, {'source': 'demo', 'lpips': 'off'})
    out['study_s'] = round(time.perf_counter() - t0, 2)
    out['fit_s'] = doc['rig']['fit_seconds']
    out['iterations'] = doc['rig']['iterations']
    out['gaussians'] = doc['rig']['gaussians']
    out['stopped_by'] = doc['rig']['stopped_by']
    out['fit_ms_per_step'] = round(1000 * out['fit_s'] / max(out['iterations'], 1), 2)
    out['capture'] = doc['capture']
    out['held_out_psnr_db'] = doc['headline']['value']
    out['fit_peak_allocated_mb'] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    out['fit_peak_reserved_mb'] = round(torch.cuda.max_memory_reserved() / 1e6, 1)
    st = SV.settings_of(p, {'source': 'demo'})
    rig, info = SV.get_rig(p, st)
    dev = SV._torch_device(info['device'])
    s, im = SV.target_sensor(p, st)
    cam = im.camera
    fn = SV.target_pose_fn(p, st, s, im, False)
    from dexvantage.core.sim import splat_rig as SRG
    qs = SRG.random_qs(p, 40, 5, 0.4)
    for q in qs[:3]:
        rig.render(p, cam, fn(q), q, backend='torch', device=dev)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for q in qs:
        rig.render(p, cam, fn(q), q, backend='torch', device=dev)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    out['render'] = {'sensor': s.name, 'width': cam.width, 'height': cam.height, 'frames': len(qs),
                     'fps': round(len(qs) / dt, 2), 'ms_per_frame': round(1000 * dt / len(qs), 2),
                     'peak_allocated_mb': round(torch.cuda.max_memory_allocated() / 1e6, 1),
                     'peak_reserved_mb': round(torch.cuda.max_memory_reserved() / 1e6, 1),
                     'includes': 'posing the Gaussians by forward kinematics, the CUDA rasteriser and the copy of '
                                 'the image back to the CPU'}
    out['gpu_total_mb'] = round(torch.cuda.get_device_properties(0).total_memory / 1e6, 1)
    return out


CHILDREN = {'environment': c_environment, 'grasp': c_grasp, 'profile': c_profile, 'studies': c_studies,
            'cubemap': c_cubemap, 'startup': c_startup, 'ui': c_ui, 'splat': c_splat}


# ============================================================================ the parts (parent)
def p_environment(a, out_dir) -> dict:
    run, doc = py_child('environment', [], out_dir, a.egl, 300, 'env')
    return {'machine': machine(), 'software': doc, 'egl_request': a.egl or None, 'run': run}


IMPORT_CODE = r"""
import json, os, sys, time
t0 = time.perf_counter()
import dexvantage as dv
t1 = time.perf_counter()
n1, qt1 = len(sys.modules), any(m.startswith('PyQt6') for m in sys.modules)
dv.Rig                                    # the first API name: imports dexvantage.api
t2 = time.perf_counter()
n2 = len(sys.modules)
rig = dv.sample('so100')                  # a first use: a sample project as a Rig
t3 = time.perf_counter()
print('BENCH ' + json.dumps({'import_s': t1 - t0, 'api_s': t2 - t1, 'sample_s': t3 - t2, 'modules_import': n1,
      'modules_api': n2, 'modules_sample': len(sys.modules), 'qt_import': qt1,
      'qt': any(m.startswith('PyQt6') for m in sys.modules), 'mujoco': 'mujoco' in sys.modules,
      'cv2': 'cv2' in sys.modules, 'scipy': 'scipy' in sys.modules,
      'from_source': os.path.abspath(dv.__file__).startswith(os.path.abspath(sys.argv[1]))}))
"""


def importtime_children(stderr: str, root: str) -> list:
    """The modules root imports directly, with their cumulative import time (python -X importtime's tree)."""
    rows = []
    for ln in stderr.splitlines():
        if not ln.startswith('import time:') or ln.count('|') != 2:
            continue
        _head, cum, mod = ln.split('|')
        level = (len(mod) - len(mod.lstrip(' ')) - 1) // 2
        name = mod.strip()
        if level == 1:
            try:
                rows.append({'module': name, 'cumulative_ms': round(int(cum) / 1000, 1)})
            except ValueError:
                pass
        elif level == 0:                     # the tree is printed children first: these were another root's
            if name == root:
                break
            rows = []
    return sorted(rows, key=lambda d: -d['cumulative_ms'])


def p_import(a, out_dir) -> dict:
    tmp = tempfile.mkdtemp(prefix='bench-import-', dir=os.environ.get('TMPDIR') or None)
    env = child_env(tmp, '')
    rows, base = [], []
    try:
        for _ in range(7):
            t0 = time.perf_counter()
            subprocess.run([sys.executable, '-c', 'pass'], cwd=ROOT, env=env, check=True)
            base.append(time.perf_counter() - t0)
        for _ in range(7):
            t0 = time.perf_counter()
            r = subprocess.run([sys.executable, '-c', IMPORT_CODE, ROOT], cwd=ROOT, env=env, capture_output=True,
                               text=True, check=True)
            wall = time.perf_counter() - t0
            line = next(ln for ln in r.stdout.splitlines() if ln.startswith('BENCH '))
            rows.append({**json.loads(line[6:]), 'process_s': wall})
        r = subprocess.run([sys.executable, '-X', 'importtime', '-c', 'import dexvantage.api'], cwd=ROOT, env=env,
                           capture_output=True, text=True, check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    first = rows[0]

    def med(k):
        return round(statistics.median(r[k] for r in rows) * 1000, 2)
    return {'runs': len(rows), 'import_ms_median': med('import_s'),
            'import_ms_min': round(min(r['import_s'] for r in rows) * 1000, 2),
            'api_ms_median': med('api_s'), 'sample_ms_median': med('sample_s'),
            'process_ms_median': med('process_s'), 'interpreter_ms_median': round(statistics.median(base) * 1000, 1),
            'modules_import': first['modules_import'], 'modules_api': first['modules_api'],
            'modules_sample': first['modules_sample'], 'imports_qt_on_import': first['qt_import'],
            'imports_qt': first['qt'], 'imports_mujoco': first['mujoco'], 'imports_opencv': first['cv2'],
            'imports_scipy': first['scipy'], 'from_source': first['from_source'],
            'slowest_api_imports': importtime_children(r.stderr, 'dexvantage.api')[:8], 'load_before': load()}


def p_startup(a, out_dir) -> dict:
    rows = []
    for name in ('blank',) + tuple(a.samples):
        reps = []
        for k in range(3):
            run, doc = py_child('startup', ['--sample', name], out_dir, a.egl, 300, f'{name}-{k}')
            reps.append({**(doc or {}), 'run': run})
        ok = [r['first_paint_s'] for r in reps if r.get('first_paint_s') is not None]
        rows.append({'sample': name, 'first_paint_s_median': round(statistics.median(ok), 3) if ok else None,
                     'first_paint_s_max': round(max(ok), 3) if ok else None,
                     'interpreter_s_median': round(statistics.median(r['interpreter_s'] for r in reps
                                                                     if 'interpreter_s' in r), 3) if ok else None,
                     'app_s_median': round(statistics.median(r['app_s'] for r in reps if r.get('app_s')), 3)
                     if ok else None,
                     'peak_mb': max(r['run']['peak_mb'] for r in reps), 'runs': reps})
    return {'rows': rows}


def p_grasp(a, out_dir) -> dict:
    rows = []
    for name in a.samples:
        run, doc = py_child('grasp', ['--sample', name, '--grasps', str(a.grasps)], out_dir, a.egl, 3600, name)
        rows.append({**(doc or {'sample': name, 'error': 'the run did not finish; see out/bench/logs/grasp.log'}),
                     'run': run, 'budget_s': GRASP_BUDGET_S,
                     'within_budget': bool(doc) and doc['study_s'] <= GRASP_BUDGET_S})
        print(f'grasp {name}: {doc and doc["study_s"]} s, {run["peak_mb"]} MB', flush=True)
    return {'budget_s': GRASP_BUDGET_S, 'grasps': a.grasps, 'rows': rows}


def p_profile(a, out_dir) -> dict:
    path = os.path.join(out_dir, 'grasp.json')
    if not os.path.exists(path):
        return {'skipped': 'no grasp results to look at'}
    with open(path, encoding='utf-8') as fh:
        gr = json.load(fh)
    miss = [r for r in gr.get('rows', []) if r.get('study_s') is not None and not r.get('within_budget')]
    if not miss:
        return {'skipped': 'every sample finished within the grasp budget'}
    worst = max(miss, key=lambda r: r['study_s'])
    run, doc = py_child('profile', ['--sample', worst['sample'], '--grasps', str(a.grasps)], out_dir, a.egl, 7200,
                        worst['sample'])
    return {'sample': worst['sample'], **(doc or {'error': 'the profiled run did not finish'}), 'run': run}


def p_studies(a, out_dir) -> dict:
    rows = []
    for name in a.samples:
        for sid in ('coverage', 'depth_validity'):
            run, doc = py_child('studies', ['--sample', name, '--study', sid], out_dir, a.egl, 1200, f'{name}-{sid}')
            rows.append({**(doc or {'sample': name, 'study': sid, 'error': 'did not finish'}), 'run': run})
            print(f'{sid} {name}: {doc and doc["first_s"]} s', flush=True)
    return {'rows': rows}


def p_cubemap(a, out_dir) -> dict:
    run, doc = py_child('cubemap', ['--samples', ','.join(a.samples)], out_dir, a.egl, 3600, 'all')
    return {**(doc or {'error': 'did not finish'}), 'run': run}


def p_ui(a, out_dir) -> dict:
    out = []
    for name in a.samples:
        run, doc = py_child('ui', ['--sample', name], out_dir, a.egl, 3600, name)
        out.append({**(doc or {'sample': name, 'rows': [], 'error': 'did not finish'}), 'run': run})
        n_over = sum(1 for r in (doc or {}).get('rows', []) if r.get('over_budget'))
        print(f'ui {name}: {n_over} steps over {UI_BUDGET_MS:g} ms', flush=True)
    return {'budget_ms': UI_BUDGET_MS, 'samples': out}


def p_validate(a, out_dir) -> dict:
    folder = os.path.join(ROOT, 'out', 'validation')
    cmd = [sys.executable, '-m', 'dexvantage', '--validate', folder]
    run = run_child(cmd, os.path.join(out_dir, 'logs', 'validate.log'), a.egl, 7200)
    summary = None
    path = os.path.join(folder, 'validation.json')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as fh:
            rep = json.load(fh)
        summary = {**rep.get('summary', {}), 'seconds_in_suite': rep.get('seconds'), 'created_utc':
                   rep.get('created_utc'), 'models': len(rep.get('models', []))}
    return {'command': 'dexvantage --validate out/validation', 'run': run, 'summary': summary}


def p_splat(a, out_dir) -> dict:
    rows = []
    for name in ('so100', 'ur5e'):
        run, doc = py_child('splat', ['--sample', name], out_dir, a.egl, 1800, name)
        rows.append({**(doc or {'sample': name, 'error': 'did not finish (the splat extra and CUDA are needed)'}),
                     'run': run})
        print(f'splat {name}: fit {doc and doc.get("fit_s")} s, render {doc and doc["render"]["fps"]} fps',
              flush=True)
    return {'rows': rows}


PART_FNS = {'environment': p_environment, 'import': p_import, 'startup': p_startup, 'grasp': p_grasp,
            'profile': p_profile,
            'studies': p_studies, 'cubemap': p_cubemap, 'ui': p_ui, 'splat': p_splat, 'validate': p_validate}


# ============================================================================ docs/performance.md
def _f(x, nd=1, unit=''):
    if x is None:
        return '—'
    if isinstance(x, float):
        s = f'{x:,.{nd}f}'.replace(',', ' ')
    else:
        s = f'{x:,}'.replace(',', ' ')
    return f'{s} {unit}'.strip()


def _verdict(ok) -> str:
    return '—' if ok is None else ('pass' if ok else 'FAIL')


def load_results(out_dir: str) -> dict:
    res = {}
    for part in PARTS + ('grasp_ab',):
        path = os.path.join(out_dir, f'{part}.json')
        if os.path.exists(path):
            with open(path, encoding='utf-8') as fh:
                res[part] = json.load(fh)
    return res


def ui_worst(ui: dict) -> list:
    """[(sample, worst UI-thread figure {'what', 'max_ms'}, steps over budget)] from the ui part: the timed steps
    and watchdog stalls on the UI thread, and the stalls the sampler named."""
    out = []
    budget = ui.get('budget_ms', UI_BUDGET_MS)
    for s in ui.get('samples', []):
        rows = [{'what': r['what'], 'max_ms': r['max_ms']} for r in s.get('rows', []) if r.get('thread') == 'UI']
        rows += [{'what': f'a stall, {e["during"]} ({"" if e["where"].startswith(("outside", "not")) else "at "}'
                          f'{e["where"]})', 'max_ms': e['ms']} for e in s.get('stalls', [])]
        worst = max(rows, key=lambda r: r['max_ms']) if rows else None
        over = sum(1 for r in s.get('rows', []) if r.get('thread') == 'UI' and r['max_ms'] > budget)
        over_stalls = sum(1 for e in s.get('stalls', []) if e['ms'] > budget)
        out.append((s.get('sample'), worst, max(over, over_stalls)))
    return out


def render_doc(res: dict) -> str:  # noqa: C901 — one section per part, in the order of the page
    env = res.get('environment', {})
    mach, sw = env.get('machine') or {}, env.get('software') or {}
    gl = sw.get('gl') or {}
    date = res.get('_date') or _dt.date.today().isoformat()
    L = ['# Performance', '',
         'Measured by `scripts/benchmarks.py` on one machine; every figure below comes from the JSON files it wrote to '
         '`out/bench/` and this page is made from them (`python scripts/benchmarks.py --doc-only`). Spec §16 asks for '
         'two budgets: no UI-thread work over 100 ms, and a 2,000-grasp study that finishes in a measured, documented '
         'time on a mid-range GPU. Only the GPU below has been measured, and this page does not class it as '
         'mid-range or otherwise: the figures are one machine\'s, a slower GPU or CPU scales them, and the machine '
         'was not idle (the load average is given beside each part). The UI timings of the earlier phases, and what '
         'was changed to meet the budget then, are in [models/performance.md](models/performance.md).', '',
         f'Measured on {date} (UTC).', '', '## The machine', '',
         '| | |', '|---|---|',
         f'| CPU | {mach.get("cpu", "—")} ({_f(mach.get("physical_cores"))} cores, '
         f'{_f(mach.get("logical_cpus"))} threads) |',
         f'| GPU (the occlusion and render backend) | {gl.get("renderer") or "none"}'
         f'{" through " + gl["how"] if gl.get("how") else ""}'
         f'{" (software)" if gl.get("software") else ""} |']
    for g in mach.get('nvidia_smi') or []:
        L.append(f'| NVIDIA driver, memory | {g["name"]}: driver {g["driver"]}, {g["memory"]} |')
    pk = sw.get('packages') or {}
    L += [f'| RAM | {_f(mach.get("ram_gb"), 1, "GB")} |',
          f'| OS | {mach.get("os", "—")}, {mach.get("kernel", "—")}, {mach.get("machine", "—")} |',
          f'| Python | {sw.get("python", "—")} ({sw.get("implementation", "")}) |',
          '| Packages | ' + ', '.join(f'{k} {v}' for k, v in pk.items() if v) + ' |']
    if sw.get('blas'):
        L.append(f'| BLAS | {sw["blas"]} |')
    gi = sw.get('git') or {}
    if gi.get('commit'):
        L.append(f'| Source | commit {gi["commit"]}{" with local changes" if gi.get("modified") else ""} |')
    L.append('')

    # ---------------------------------------------------------------- budgets
    L += ['## Budgets', '', '| Budget | Where it comes from | Measured | Result |', '|---|---|---|---|']
    gr = res.get('grasp') or {}
    budget_s = gr.get('budget_s', GRASP_BUDGET_S)
    for r in gr.get('rows', []):
        L.append(f'| 2,000-grasp study, {r.get("sample")}: at most {_f(budget_s, 0, "s")} | spec §16 (the time is '
                 f'this project\'s default; why below) | {_f(r.get("study_s"), 1, "s")} | '
                 f'{_verdict(r.get("within_budget") if "study_s" in r else None)} |')
    ui = res.get('ui') or {}
    for name, worst, over in ui_worst(ui):
        if worst is None:
            continue
        L.append(f'| No UI-thread step over {_f(ui.get("budget_ms", UI_BUDGET_MS), 0, "ms")}, {name} | spec §16 | '
                 f'slowest {_f(worst["max_ms"], 1, "ms")} ({worst["what"]}); {over} over | {_verdict(over == 0)} |')
    L += ['', f'The grasp budget of {_f(budget_s, 0, "s")} is a default chosen for this project, not a figure from the '
          'specification (which asks for a measured, documented time): spec §18 asks that a researcher sees which '
          'grasps lose fingertip visibility, moves a camera and sees the score change within ten minutes, which runs '
          'the grasp study at least twice beside the rest of that tour; three minutes a run leaves four for the rest.',
          '']

    # ---------------------------------------------------------------- grasp
    if gr.get('rows'):
        L += ['## A 2,000-grasp study on each sample', '',
              f'{_f(gr.get("grasps"))} grasps sampled on the primary arm of each sample (`core/grasps.sample`, '
              'top-down, side and angled in turn, seed 0), then the grasp self-occlusion study with its default '
              'settings: IK for every grasp, the pre-contact path, cube maps per camera and joint state, "what to '
              'move" (sensitivity, 12 one-at-a-time moves per camera on up to 250 grasps) and the gallery. Each '
              'sample in a fresh process; the peak is the process\'s largest resident set.', '',
              '| Sample | Arm | Cameras | Feasible grasps | Study (s) | Grasps/s | IK (s) | Visibility (s) | '
              'What to move (s) | Cube maps | Maps/s | Peak memory (MB) | Load before → after |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for r in gr['rows']:
            tm = r.get('timings') or {}
            run = r.get('run') or {}
            L.append(f'| {r.get("sample")} | {r.get("robot", "—")} | {_f(r.get("cameras"))} | '
                     f'{_f(r.get("feasible"))} of {_f(r.get("grasps"))} | {_f(r.get("study_s"), 1)} | '
                     f'{_f(r.get("grasps_per_s"), 1)} | {_f(tm.get("ik_seconds"), 1)} | '
                     f'{_f(tm.get("occlusion_seconds"), 1)} | {_f(tm.get("sensitivity_seconds"), 1)} | '
                     f'{_f(r.get("maps_built"))} | {_f(r.get("maps_per_s"), 0)} | {_f(run.get("peak_mb"), 0)} | '
                     f'{_f(run.get("load_before"), 1)} → {_f(run.get("load_after"), 1)} |')
        rend = next((r.get('renderer') for r in gr['rows'] if r.get('renderer')), None)
        L += ['', f'Cube maps rendered by {rend or "—"} at {_f(gr["rows"][0].get("cube_n"))} texels a face. '
              'Grasps/s is the study\'s own figure (grasps over the whole run).', '']
        ab = res.get('grasp_ab') or {}
        if ab.get('before') and ab.get('after'):
            b, a2 = ab['before'], ab['after']
            tb, ta = b.get('timings') or {}, a2.get('timings') or {}
            L += ['### The change made to the grasp study', '',
                  'The first measurement missed the budget on the ALOHA 2 sample, and a profile showed a share of the '
                  'time in the grasp study\'s own culling: every cube map tested each triangle of a link against the '
                  'cone around the points (two links of 94 730 triangles each), and each "what to move" step '
                  'recomputed the moved camera\'s seat mask once per grasp. The study now tests runs of 256 '
                  'consecutive triangles first and only the triangles of the runs that meet the cone (the triangles '
                  'kept are exactly the per-triangle test\'s: `tests/test_grasp_cull.py`), and keeps the seat masks '
                  'of the last 16 poses (`core/studies/grasp_visibility.py`). Measured with the'
                  f' same {ab.get("how")} ({ab.get("sample")}, {ab.get("measured_local_date")}):', '',
                  '| | Study (s) | Visibility (s) | What to move (s) | Cube maps | Load before → after |',
                  '|---|---|---|---|---|---|',
                  f'| Before | {_f(b.get("study_s"), 1)} | {_f(tb.get("occlusion_seconds"), 1)} | '
                  f'{_f(tb.get("sensitivity_seconds"), 1)} | {_f(b.get("maps"))} | {_f(b.get("load_before"), 1)} → '
                  f'{_f(b.get("load_after"), 1)} |',
                  f'| After | {_f(a2.get("study_s"), 1)} | {_f(ta.get("occlusion_seconds"), 1)} | '
                  f'{_f(ta.get("sensitivity_seconds"), 1)} | {_f(a2.get("maps"))} | {_f(a2.get("load_before"), 1)} → '
                  f'{_f(a2.get("load_after"), 1)} |', '',
                  ('The study\'s result (its summary and the maps drawn) is the same before and after.'
                   if ab.get('same_result') else '**The study\'s result differed before and after.**'), '']
        if res.get('_grasp_notes'):
            L += res['_grasp_notes'] + ['']
        prof = res.get('profile') or {}
        if prof.get('own_time'):
            g = prof.get('grasp') or {}
            L += [f'### Where the time goes ({prof.get("sample")})', '',
                  f'The slowest sample over the budget, run again under cProfile (`--only profile`): '
                  f'{_f(prof.get("profiled_s"), 1, "s")} profiled (the profiler adds its own overhead; '
                  f'{_f(g.get("study_s"), 1, "s")} for the study in that run). The DexVantage functions with the '
                  'most time in all:', '',
                  '| Function | Calls | Time in all (s) | Own time (s) |', '|---|---|---|---|']
            for r in prof.get('cumulative', [])[:12]:
                L.append(f'| `{r["function"]}` | {_f(r["calls"])} | {_f(r["cumulative_s"], 1)} | '
                         f'{_f(r["own_s"], 1)} |')
            def fn(path, name, key='cumulative'):
                rows = prof.get('cumulative', []) + prof.get('own_time', [])
                for r in rows:
                    f, _, nm = r['function'].rpartition(' ')
                    if f.rsplit(':', 1)[0] == path and nm == name:
                        return r
                return None
            tot = prof.get('profiled_s') or 0
            parts = []
            for path, name, what in (('dexvantage/core/occlusion.py', 'build', 'drawing the cube maps '
                                      '(`core/occlusion.py` `CubeMap.build`)'),
                                     ('dexvantage/core/studies/grasp_visibility.py', 'occluders', 'posing and culling '
                                      'the triangles for each map (`Evaluator.occluders`)'),
                                     ('dexvantage/core/robot/ik.py', 'solve_batch', 'the IK (`ik.solve_batch`)')):
                r = fn(path, name)
                if r and tot:
                    parts.append(f'{what} {_f(r["cumulative_s"], 1, "s")} ({_f(100 * r["cumulative_s"] / tot, 0)} %)')
            if parts:
                L += ['', f'Of the {_f(tot, 1, "s")} profiled: ' + '; '.join(parts) + '. What is left in the grasp '
                      'study\'s own code is posing each kept triangle on the CPU for every map; the larger share is '
                      'the shared occlusion code (`core/occlusion.py`, not changed here): its GL backend\'s per-map '
                      'work in Python, the read-back of the six faces and the far-plane distance over every vertex '
                      '(the own times below). Drawing the posed links on the GPU from their link transforms, as '
                      '`docs/models/occlusion.md` proposes, would remove both the posing and most of the upload.']
            L += ['', 'The functions with the most time of their own (any library):', '',
                  '| Function | Calls | Own time (s) | Time in all (s) |', '|---|---|---|---|']
            for r in prof['own_time'][:10]:
                L.append(f'| `{r["function"]}` | {_f(r["calls"])} | {_f(r["own_s"], 1)} | '
                         f'{_f(r["cumulative_s"], 1)} |')
            L.append('')

    # ---------------------------------------------------------------- coverage and depth
    stud = res.get('studies') or {}
    if stud.get('rows'):
        L += ['## Coverage and depth validity on every sample', '',
              'Each study in a fresh process with the sample\'s own settings: the first run (it builds the robot '
              'model, the scene\'s occluders and the cube maps) and a repeat in the same process (the compiled '
              'model and cube maps then come from the process caches).', '',
              '| Sample | Study | Cameras | First run (s) | Repeat (s) | Result | Peak memory (MB) |',
              '|---|---|---|---|---|---|---|']
        for r in stud['rows']:
            if r.get('study') == 'coverage':
                what = f'{_f((r.get("seen_ge1") or 0) * 100, 1, "%")} of the task volume seen' \
                    if r.get('seen_ge1') is not None else '—'
            else:
                n = r.get('depth_sensors')
                what = (f'{n} depth sensor{"s" if n != 1 else ""}' if n
                        else (r.get('message') or 'no depth sensor'))
            L.append(f'| {r.get("sample")} | {r.get("study")} | {_f(r.get("cameras"))} | {_f(r.get("first_s"), 2)} | '
                     f'{_f(r.get("repeat_s"), 3)} | {what} | {_f((r.get("run") or {}).get("peak_mb"), 0)} |')
        L.append('')

    # ---------------------------------------------------------------- cube maps
    cm = res.get('cubemap') or {}
    if cm.get('rows'):
        bes = [b for b in ('gl', 'numpy') if any(f'{b}_512' in r for r in cm['rows'])]
        L += ['## Occlusion cube-map build', '',
              'One cube map (six faces) of the scene seen from each camera at the home pose, `core/occlusion.CubeMap.'
              'build`; the median of 10 builds (GPU) or 3 (NumPy) after a first build that uploads the geometry. '
              f'GPU: {(cm.get("gl") or {}).get("renderer") or "not available"}.', '',
              '| Sample | Camera | Triangles | ' + ' | '.join(f'{b} {n} (ms)' for b in bes for n in (256, 512))
              + ' | GPU first build, 512 (ms) |',
              '|---|---|---|' + '---|' * (2 * len(bes)) + '---|']
        for r in cm['rows']:
            L.append(f'| {r["sample"]} | {r["camera"]} | {_f(r["triangles"])} | '
                     + ' | '.join(_f((r.get(f'{b}_{n}') or {}).get('median_ms'), 1) for b in bes for n in (256, 512))
                     + f' | {_f((r.get("gl_512") or {}).get("first_ms"), 1)} |')
        L.append('')

    # ---------------------------------------------------------------- UI
    if ui.get('samples'):
        L += ['## UI-thread events', '',
              'Each sample opened in a fresh process (offscreen), then every study its pages offer run from its '
              'page on the worker thread, as a user starts it. Timed on the UI thread: showing each page, the result '
              'arriving (show_result) and the page\'s 3D overlays (median of 3, the largest in the table); an event-'
              'loop watchdog (a 5 ms timer, `scripts/measure_ui_phase3.py`) records the longest stall while the study '
              'runs and its result arrives, so a slow handler no step names still shows. Offscreen the 3D view '
              'falls back to QPainter; its paint is left out, as in the earlier UI timings (a desktop draws it with '
              'the shader renderer).', '',
              '| Sample | Studies run | Slowest UI-thread step | Its time (ms) | Over budget | Building the window '
              '(ms) | Show to first paint (ms) |', '|---|---|---|---|---|---|---|']
        worst_by = {n: (wr, ov) for n, wr, ov in ui_worst(ui)}
        for s in ui['samples']:
            rows = s.get('rows', [])
            wr, n_over = worst_by.get(s.get('sample'), (None, 0))
            built = next((r['max_ms'] for r in rows if r['what'].startswith('open the sample: build')), None)
            first = next((r['max_ms'] for r in rows if r['what'].startswith('open the sample: show')), None)
            L.append(f'| {s.get("sample")} | {len(s.get("studies", []))} | {wr["what"] if wr else "—"} | '
                     f'{_f(wr["max_ms"] if wr else None, 1)} | {n_over} | {_f(built, 0)} | {_f(first, 0)} |')
        L += ['', 'Building the window and its first paint are start-up (the window is not yet on screen); they are '
              'listed here and in the start-up table below, not counted against the 100 ms budget, as in the earlier '
              'UI timings. "Over budget" counts the steps and the stalls above 100 ms after the first paint.', '',
              f'The stalls of {_f(ui["samples"][0].get("stall_ms", STALL_MS), 0, "ms")} or more on each sample, '
              'with where the UI thread was (a thread samples its Python stack while a 5 ms timer is late; the '
              'innermost DexVantage frames first):', '']
        for s in ui['samples']:
            ev = s.get('stalls', [])[:6]
            if ev:
                L.append(f'- **{s.get("sample")}**: ' + '; '.join(
                    f'{_f(e["ms"], 0, "ms")} {e["during"]} — `{e["where"]}`'
                    + (f' (cycle collector {_f(e["gc_ms"], 0, "ms")})' if (e.get('gc_ms') or 0) >= 1 else '')
                    for e in ev))
            else:
                L.append(f'- **{s.get("sample")}**: none')
        over = [(s.get('sample'), e) for s in ui['samples'] for e in s.get('stalls', [])
                if e['ms'] > ui.get('budget_ms', UI_BUDGET_MS)]
        if over:
            L += ['', 'The stalls and steps over the budget, and what the measurement says about them:', '']
            for name, e in over:
                sid = e['during'].split(':', 1)[0]
                st = next((x for s in ui['samples'] if s.get('sample') == name for x in s.get('studies', [])
                           if x.get('study') == sid), {})
                why = ''
                if (e.get('gc_ms') or 0) >= 0.3 * e['ms']:
                    gens = ', '.join(str(g) for g in e.get('gc_generations') or [])
                    why = (f' Python\'s cycle collector (generation {gens}) ran for {_f(e["gc_ms"], 0, "ms")} of it '
                           '(it holds the interpreter lock, so the UI thread waits whichever thread started it).')
                elif e.get('ui_cpu_ms') is not None and e['ui_cpu_ms'] < 0.5 * e['ms']:
                    why = (f' The UI thread used {_f(e["ui_cpu_ms"], 0, "ms")} of processor time in it: for the rest '
                           'it was not running (waiting for the interpreter lock or for the processor).')
                elif 'DexVantage\'s Python code' in e['where'] and (st.get('progress_reports') or 0) >= 1000:
                    why = (f' The study sent {_f(st["progress_reports"])} progress reports in '
                           f'{_f(st.get("seconds"), 1, "s")}, each a queued signal the UI thread delivers.')
                elif not e['where'].startswith('outside') and e['where'] != 'not sampled':
                    why = f' The UI thread was in `{e["where"]}`.'
                L.append(f'- {name}, {_f(e["ms"], 0, "ms")}, a stall: {e["during"]}.{why}')
        steps = [(s.get('sample'), r) for s in ui['samples'] for r in s.get('rows', [])
                 if r.get('thread') == 'UI' and r['max_ms'] > ui.get('budget_ms', UI_BUDGET_MS)
                 and 'watchdog stall' not in r['what']]
        if steps:
            L += ([] if over else ['', 'The steps over the budget:', ''])
            for name, r in steps:
                L.append(f'- {name}, {_f(r["max_ms"], 0, "ms")} (median of the runs '
                         f'{_f(r.get("median_ms"), 1, "ms")}), a timed step: {r["what"]}.')
        many = sorted(((s.get('sample'), x) for s in ui['samples'] for x in s.get('studies', [])
                       if (x.get('progress_reports') or 0) >= 1000), key=lambda t: -t[1]['progress_reports'])
        if many:
            L += ['', 'Studies that sent 1 000 or more progress reports in one run: ' + '; '.join(
                f'{n} {x["study"]} {_f(x["progress_reports"])} in {_f(x.get("seconds"), 1, "s")}' for n, x in many)
                + '.']
        L += ['', 'The five slowest timed UI-thread steps on each sample:', '']
        for s in ui['samples']:
            rows = sorted((r for r in s.get('rows', []) if r.get('thread') == 'UI'), key=lambda r: -r['max_ms'])[:5]
            if rows:
                L.append(f'- **{s.get("sample")}**: ' + '; '.join(f'{r["what"]} {_f(r["max_ms"], 1, "ms")}'
                                                               for r in rows))
        L.append('')
        refused = [(s.get('sample'), x) for s in ui['samples'] for x in s.get('studies', []) if x.get('refused')]
        if refused:
            L += ['Studies that refused on a sample (their refusal is the result the page shows; the events were '
                  'still timed): ' + '; '.join(f'{n} {x["study"]} ("{x["refused"]}")' for n, x in refused) + '.', '']
        sk = next((s.get('skipped') for s in ui['samples'] if s.get('skipped')), None)
        if sk:
            L += ['Not run in this part: ' + '; '.join(f'{k} ({v})' for k, v in sk.items()) + '.', '']
        slow = sorted(((s.get('sample'), x) for s in ui['samples'] for x in s.get('studies', [])
                       if x.get('seconds') is not None), key=lambda t: -t[1]['seconds'])[:6]
        if slow:
            L += ['The longest worker runs (the study on its thread, the UI responsive meanwhile): '
                  + '; '.join(f'{n} {x["study"]} {_f(x["seconds"], 1, "s")}' for n, x in slow) + '.', '']

    # ---------------------------------------------------------------- start-up and import
    su = res.get('startup') or {}
    if su.get('rows'):
        L += ['## Start-up to first paint', '',
              '`dexvantage.app.main` started in a fresh process as `dexvantage [--sample NAME]` (offscreen, throw-away '
              'settings), timed from the moment the parent starts the process to the first paint event of the main '
              'window; 3 runs each.', '',
              '| Project | First paint, median (s) | Largest (s) | Interpreter up (s) | Application made (s) | '
              'Peak memory (MB) |', '|---|---|---|---|---|---|']
        for r in su['rows']:
            L.append(f'| {r["sample"]} | {_f(r.get("first_paint_s_median"), 2)} | {_f(r.get("first_paint_s_max"), 2)}'
                     f' | {_f(r.get("interpreter_s_median"), 2)} | {_f(r.get("app_s_median"), 2)} | '
                     f'{_f(r.get("peak_mb"), 0)} |')
        L.append('')
    im = res.get('import') or {}
    if im:
        yn = {True: 'yes', False: 'no', None: '—'}
        L += ['## Import time', '',
              f'`import dexvantage as dv` in a fresh interpreter from the source tree (no bundle), {im.get("runs")} '
              'runs, medians. The package resolves its API names on first use, so the import itself loads almost '
              'nothing; the first API name loads `dexvantage.api`, and a first use opens a sample as a Rig.', '',
              '| Step | Time (ms) | Modules loaded after it |', '|---|---|---|',
              f'| `import dexvantage as dv` | {_f(im.get("import_ms_median"), 2)} | {_f(im.get("modules_import"))} |',
              f'| the first API name (`dv.Rig`) | {_f(im.get("api_ms_median"), 1)} | {_f(im.get("modules_api"))} |',
              f'| a first use (`dv.sample(\'so100\')`) | {_f(im.get("sample_ms_median"), 1)} | '
              f'{_f(im.get("modules_sample"))} |',
              f'| the whole process (an empty interpreter takes {_f(im.get("interpreter_ms_median"), 1)}) | '
              f'{_f(im.get("process_ms_median"), 1)} | |', '',
              f'Qt is imported by the import: {yn[im.get("imports_qt_on_import")]}; after the first use: Qt '
              f'{yn[im.get("imports_qt")]}, MuJoCo {yn[im.get("imports_mujoco")]}, OpenCV '
              f'{yn[im.get("imports_opencv")]}, SciPy {yn[im.get("imports_scipy")]}.', '']
        if im.get('slowest_api_imports'):
            L += ['The slowest modules `dexvantage.api` imports directly (`python -X importtime`, cumulative): '
                  + ', '.join(f'{t["module"]} {_f(t["cumulative_ms"], 1, "ms")}' for t in im['slowest_api_imports'])
                  + '.', '']

    # ---------------------------------------------------------------- splat views
    sp = res.get('splat') or {}
    srows = [r for r in sp.get('rows') or [] if r.get('fit_s') is not None]
    if srows:
        r0 = srows[0]
        L += ['## Splat views: fit time, render rate and GPU memory', '',
              'The splat views study (`core/sim/splat_views.py`) with its default settings on a MuJoCo render of each '
              'sample (a synthetic demonstration capture), in a fresh process: the fit (1,500 steps at most, a '
              '120 s budget, a 20,000-Gaussian budget, 48 views at 160 × 120 px, defaults), then 40 renders of the '
              'fitted rig through the target camera (the wrist camera) at its stream size at 40 joint states, after '
              f'3 warm-up renders. GPU: {r0.get("gpu")}, PyTorch {r0.get("torch")} (CUDA {r0.get("cuda")}), the '
              'plain-torch rasteriser (the study fits and renders through it whether or not gsplat is installed: '
              '[models/splat.md](models/splat.md)). GPU memory is '
              'what PyTorch allocated and reserved at its peak (`torch.cuda.max_memory_allocated` and '
              '`max_memory_reserved`), not what the driver reports for the process.', '',
              '| Sample | Gaussians | Fit (s) | ms per step | Study (s) | Fit GPU peak, allocated / reserved (MB) | '
              'Render size | Render fps | ms per frame | Render GPU peak, allocated / reserved (MB) | Process peak '
              '(MB) | Load before → after |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for r in srows:
            rd = r.get('render') or {}
            run = r.get('run') or {}
            L.append(f'| {r["sample"]} | {_f(r.get("gaussians"))} | {_f(r.get("fit_s"), 1)} | '
                     f'{_f(r.get("fit_ms_per_step"), 1)} | {_f(r.get("study_s"), 1)} | '
                     f'{_f(r.get("fit_peak_allocated_mb"), 0)} / {_f(r.get("fit_peak_reserved_mb"), 0)} | '
                     f'{rd.get("width")} × {rd.get("height")} | {_f(rd.get("fps"), 1)} | '
                     f'{_f(rd.get("ms_per_frame"), 1)} | {_f(rd.get("peak_allocated_mb"), 0)} / '
                     f'{_f(rd.get("peak_reserved_mb"), 0)} | {_f(run.get("peak_mb"), 0)} | '
                     f'{_f(run.get("load_before"), 1)} → {_f(run.get("load_after"), 1)} |')
        L += ['', 'The study\'s time includes the MuJoCo capture, the held-out scoring and the before/after preview '
                  'besides the fit. A render includes posing the Gaussians by forward kinematics and copying the '
                  'image back to the CPU; the replay and the augmentation also encode video.', '']

    # ---------------------------------------------------------------- validation
    va = res.get('validate') or {}
    if va.get('run'):
        sm = va.get('summary') or {}
        L += ['## A full validation run', '',
              f'`{va.get("command")}` in a fresh process: {_f(va["run"].get("wall_s"), 0, "s")} for the process, '
              f'{_f(sm.get("seconds_in_suite"), 0, "s")} in the suite, {_f(sm.get("passed"))} of '
              f'{_f(sm.get("checks"))} checks passed over {_f(sm.get("models"))} models, peak memory '
              f'{_f(va["run"].get("peak_mb"), 0, "MB")} (load {_f(va["run"].get("load_before"), 1)} → '
              f'{_f(va["run"].get("load_after"), 1)}). The checks and their measured agreement are in '
              '[validation.md](validation.md).', '']

    L += ['## How to reproduce', '',
          '```', 'cd <repository>',
          'systemd-run --user --scope -q -p MemoryMax=7G env QT_QPA_PLATFORM=offscreen \\',
          "    .venv/bin/python scripts/benchmarks.py --egl 'EGL device 0'",
          '.venv/bin/python scripts/gen_validation_doc.py out/validation/validation.json   # docs/validation.md',
          '```', '',
          '`--only grasp,ui` runs some parts, `--samples ur5e,aloha` some samples, `--doc-only` remakes this page '
          'from `out/bench/*.json`. `--egl` picks the EGL device or renderer of the GPU cube maps (DEXVANTAGE_EGL); '
          'each child runs offscreen with throw-away settings folders.', '',
          '## Known limits', '',
          '- One machine, one run of each figure (start-up and import take the median of a few); the machine was '
          'shared with other work, so the times are upper bounds of what it does idle.',
          '- Peak memory is the child process\'s largest resident set (`ru_maxrss`); a validation check\'s own child '
          'processes (cluster shards, MuJoCo renders in a separate process) are not added to it.',
          '- The UI timings are offscreen with the QPainter 3D fallback\'s paint left out; a desktop session adds '
          'the shader renderer\'s frames, timed in Phase 2 ([models/performance.md](models/performance.md)): '
          '7.8 ms (SO-100) and 11.9 ms (UR5e) median for a steady frame, and for a frame after a joint change '
          '26.5 ms and 67.6 ms median, 95.5 ms worst (UR5e, integrated GPU). The QPainter fallback itself '
          '(no OpenGL, or `--software-3d`) takes 160 ms and 586 ms a frame there, over the budget.',
          '- The 2,000-grasp set is sampled, not a task\'s own grasps; a set with more infeasible grasps is faster.',
          '']
    return '\n'.join(L)


def scrub_logs(folder: str):
    """The children's logs without the repository's or the home folder's path (they are kept for reading)."""
    home = os.path.expanduser('~')
    for name in os.listdir(folder) if os.path.isdir(folder) else ():
        path = os.path.join(folder, name)
        try:
            with open(path, encoding='utf-8', errors='replace') as fh:
                text = fh.read()
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(text.replace(ROOT, '<repository>').replace(home, '~'))
        except OSError:
            pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--only', default='', help='comma-separated parts: ' + ', '.join(PARTS))
    ap.add_argument('--samples', default=','.join(SAMPLES))
    ap.add_argument('--grasps', type=int, default=2000)
    ap.add_argument('--egl', default='EGL device 0', help="EGL device or renderer for the GPU ('' for the default)")
    ap.add_argument('--out', default=os.path.join(ROOT, 'out', 'bench'))
    ap.add_argument('--doc', default=os.path.join(ROOT, 'docs', 'performance.md'))
    ap.add_argument('--doc-only', action='store_true')
    ap.add_argument('--no-validation-doc', action='store_true')
    ap.add_argument('--child', default='')
    ap.add_argument('--result', default='')
    ap.add_argument('--sample', default='')
    ap.add_argument('--study', default='')
    a = ap.parse_args(argv)
    if a.child:
        doc = CHILDREN[a.child](a)
        _write(a.result, doc)
        return 0
    a.samples = tuple(s for s in a.samples.split(',') if s)
    unknown = [s for s in a.samples if s not in SAMPLES]
    if unknown:
        print(f'benchmarks: there is no sample named {unknown[0]}; the samples are {", ".join(SAMPLES)}.',
              file=sys.stderr)
        return 2
    parts = [p for p in (a.only.split(',') if a.only else PARTS) if p]
    bad = [p for p in parts if p not in PARTS]
    if bad:
        print(f'benchmarks: there is no part named {bad[0]}; the parts are {", ".join(PARTS)}.', file=sys.stderr)
        return 2
    os.makedirs(os.path.join(a.out, 'logs'), exist_ok=True)
    if not a.doc_only:
        for part in parts:
            t0 = time.perf_counter()
            print(f'== {part}', flush=True)
            doc = PART_FNS[part](a, a.out)
            doc['measured_utc'] = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
            doc['part_seconds'] = round(time.perf_counter() - t0, 1)
            _write(os.path.join(a.out, f'{part}.json'), doc)
            print(f'   {part}: {doc["part_seconds"]} s', flush=True)
        scrub_logs(os.path.join(a.out, 'logs'))
        if 'validate' in parts and not a.no_validation_doc:
            subprocess.run([sys.executable, os.path.join(HERE, 'gen_validation_doc.py'),
                            os.path.join(ROOT, 'out', 'validation', 'validation.json')], cwd=ROOT, check=False)
    res = load_results(a.out)
    dates = sorted(d['measured_utc'][:10] for d in res.values() if d.get('measured_utc'))
    if dates:
        res['_date'] = dates[0] if dates[0] == dates[-1] else f'{dates[0]} to {dates[-1]}'
    notes = os.path.join(a.out, 'grasp-notes.md')
    if os.path.exists(notes):
        with open(notes, encoding='utf-8') as fh:
            res['_grasp_notes'] = fh.read().rstrip().splitlines()
    text = render_doc(res)
    with open(a.doc, 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')
    print(f'wrote {os.path.relpath(a.doc, ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
