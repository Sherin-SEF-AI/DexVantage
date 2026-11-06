"""UI-thread timings of the Phase 2 pages on the samples (spec §16: no UI-thread work over 100 ms).

    QT_QPA_PLATFORM=offscreen python scripts/measure_ui.py [--json out/ui-timings.json]

For the SO-100 and UR5e samples it times, on the UI thread, each handler a user triggers: building the window,
showing each Phase 2 page, a study's result arriving (show_result), the page's 3D overlays (once per frame), a joint
change (the posed meshes and the frusta the 3D view redraws), selecting a sensor (Inspector, Camera view), the Camera
view's per-image work on the UI thread (snapshot, receiving the image, painting it with its overlays) and the
Library panel. Studies and Camera view images render on worker threads; their times are listed separately as
"worker". An event-loop watchdog (a 5 ms timer) also records the longest stall while the events of each step are
processed, so a slow handler this list does not name still shows. Each figure is the median of 5 runs, with the
largest in brackets. The 3D view itself is drawn by the shader renderer on a desktop; offscreen it falls back to
QPainter (SoftViewport), whose frame time is listed for the record, not as the budget.
"""
import json
import os
import statistics
import sys
import tempfile
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
for k in ('XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_CACHE_HOME'):
    os.environ[k] = tempfile.mkdtemp(prefix='dexvantage-measure-')

import numpy as np  # noqa: E402
from PyQt6.QtCore import QElapsedTimer, QTimer  # noqa: E402

from dexvantage.app import make_app  # noqa: E402
from dexvantage.core.robot.samples import sample_project  # noqa: E402

app, settings = make_app('dark', settings_scope='measure')
from dexvantage.ui.mainwindow import MainWindow  # noqa: E402

BUDGET_MS = 100.0
ROWS: list = []


class Watchdog:
    """The longest gap between ticks of a 5 ms timer while events are processed: a UI-thread stall."""

    def __init__(self):
        self.t = QTimer()
        self.t.setInterval(5)
        self.t.timeout.connect(self._tick)
        self.clock = QElapsedTimer()
        self.worst = 0.0

    def _tick(self):
        if self.clock.isValid():
            self.worst = max(self.worst, self.clock.nsecsElapsed() / 1e6)
        self.clock.restart()

    def pump(self, seconds: float) -> float:
        """Process events for seconds; the longest stall (ms) meanwhile."""
        self.worst = 0.0
        self.clock.invalidate()
        self.t.start()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            app.processEvents()
            time.sleep(0.001)
        self.t.stop()
        return self.worst


WD = Watchdog()


def timed(fn, n=5):
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000)
    return statistics.median(out), max(out)


def record(sample, what, ms, where='UI'):
    med, worst = ms if isinstance(ms, tuple) else (ms, ms)
    over = where == 'UI' and worst > BUDGET_MS
    ROWS.append({'sample': sample, 'what': what, 'median_ms': round(med, 1), 'max_ms': round(worst, 1),
                 'thread': where, 'over_budget': over})
    print(f'{sample:6s} {where:6s} {med:8.1f} [{worst:7.1f}] ms  {what}' + ('   OVER 100 ms' if over else ''),
          flush=True)


def wait_study(w, sid, timeout=120):
    t0 = time.perf_counter()
    w.ctx.analysis.run(sid)
    while time.perf_counter() - t0 < timeout:
        app.processEvents()
        if not w.ctx.analysis.busy(sid) and sid in w.ctx.analysis.results:
            return (time.perf_counter() - t0) * 1000
        time.sleep(0.002)
    raise RuntimeError(f'{sid} did not finish')


def measure(name):
    t0 = time.perf_counter()
    w = MainWindow(sample_project(name), soft_3d=True, settings=settings)
    w.resize(1680, 1000)
    w.show()
    record(name, 'build and show the main window (startup)', (time.perf_counter() - t0) * 1000, 'start')
    w.ctx.analysis.wait(10)
    WD.pump(0.5)
    st, vp, scene = w.ctx.store, w.viewport, w.scene
    # the 3D view's frame is the shader renderer's on a desktop; offscreen it is QPainter's (timed on its own at the
    # end), so its paint is left out of the UI handlers' figures here
    soft_paint = type(vp).paintEvent
    type(vp).paintEvent = lambda self, e: None
    sensors = st.project.variant.sensors

    for pid in ('sensors', 'coverage', 'depth'):
        record(name, f'show the {pid} page', timed(lambda pid=pid: (w.open_page(pid), app.processEvents())))

    # studies (worker) and their results arriving on the UI thread
    for pid, sid in (('coverage', 'coverage'), ('depth', 'depth_validity')):
        w.open_page(pid)
        page = w.pages[pid]
        record(name, f'{sid} study, whole run (worker thread)', wait_study(w, sid), 'worker')
        doc = w.ctx.analysis.results[sid]
        record(name, f'{pid} page: show the result (show_result)', timed(lambda: page._finished(sid, doc)))
        record(name, f'{pid} page: 3D overlays for one frame', timed(lambda: page.overlays(vp)))
        record(name, f'{pid} page: events after the result (watchdog stall)', WD.pump(1.0))

    # a joint change: what the 3D view rebuilds for the next frame (posed meshes, lines with frusta and overlays)
    r = st.project.robots[0]
    from dexvantage.core.robot.model import get_model
    m = get_model(st.project, r.uid)
    rng = np.random.default_rng(3)
    qs = [m.home + rng.normal(0, 0.05, m.nq) * (m.upper > m.lower) for _ in range(6)]
    it = iter(qs)

    def move():
        st.set_q(r.uid, next(it))
        scene.mesh()
        scene.lines(vp)
        scene.tris_extra(vp)
    w.open_page('coverage')
    record(name, 'joint change: posed meshes, frusta, volumes, page overlays (coverage page shown)', timed(move))
    record(name, 'joint change: events afterwards (watchdog stall, Camera view re-render requested)', WD.pump(1.5))

    # selecting each sensor: Inspector, Camera view header, frusta
    def select_each():
        for s in sensors:
            st.select(('sensor', s.uid))
            app.processEvents()
    record(name, f'select each of the {len(sensors)} sensors (Inspector, Camera view header)', timed(select_each))
    record(name, 'after selection (watchdog stall)', WD.pump(1.5))

    # the Camera view: the UI thread's part of one image, and the worker's
    view = w.camview.view
    record(name, 'Camera view: snapshot for a render (UI thread)', timed(lambda: view._job(1.0)))
    t0 = time.perf_counter()
    out = view.render_sync(1.0)
    record(name, 'Camera view: render one full-size image (worker thread in the app)',
           (time.perf_counter() - t0) * 1000, 'worker')
    record(name, 'Camera view: take the image (apply, describe)', timed(lambda: view._apply(out)))
    for pid in ('coverage', 'depth'):
        w.open_page(pid)
        view._overlays = None
        record(name, f'Camera view: page overlays ({pid} page), once per image',
               timed(lambda: (setattr(view, '_overlays', None), view.page_overlays())))
        record(name, f'Camera view: paint the image with the {pid} overlays',
               timed(lambda: view.image.grab()))

    record(name, 'Library panel: refill (40 entries)', timed(w.library.refill))
    from dexvantage.ui.dialogs.custom_device import CustomDeviceDialog
    record(name, 'New custom device dialog: open from a library device',
           timed(lambda: CustomDeviceDialog(w.ctx, w, kind='depth', start_from='intel-realsense-d435').deleteLater()))
    type(vp).paintEvent = soft_paint
    record(name, '3D view, QPainter fallback: one frame (offscreen only; the desktop uses the shader renderer)',
           timed(lambda: vp.render_image(scale=1.0), n=2), 'soft3d')
    gl_frames(name, w, r, qs)
    w.cancel_jobs()
    w.hide()
    w.deleteLater()
    app.processEvents()


def gl_frames(name, w, robot, qs):
    """The shader renderer's frame for this scene, headless (EGL, as the self-test makes it): the first frame
    (uploads the meshes), a steady frame, and a frame after a joint change (the posed meshes are uploaded again)."""
    if not sys.platform.startswith('linux'):
        return
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    from dexvantage.selftest import _egl_context, _egl_release
    try:
        egl, how = _egl_context()
    except Exception as e:                       # noqa: BLE001
        print('no headless OpenGL:', e)
        return
    try:
        from OpenGL import GL
        from dexvantage.ui.render_gl import Frame, GLRenderer
        vp, scene, st = w.viewport, w.scene, w.ctx.store
        gl_name = (GL.glGetString(GL.GL_RENDERER) or b'?').decode(errors='replace')
        rr = GLRenderer()
        if not rr.init():
            print('shaders failed:', rr.error)
            return
        W, H = vp.width(), vp.height()

        def frame():
            V, P, _ = vp.matrices()
            f = Frame(V, P, W, H, target=vp.cam.target.copy(), mesh=scene.mesh(), ground=scene.ground(),
                      lines=scene.lines(vp), tris=scene.tris_extra(vp), clear=(0.1, 0.1, 0.1, 1.0))
            rr.render_offscreen(f)
        t0 = time.perf_counter()
        frame()
        record(name, f'3D view, shader renderer: first frame ({W} × {H}, mesh upload; {gl_name})',
               (time.perf_counter() - t0) * 1000)
        record(name, '3D view, shader renderer: steady frame', timed(frame))
        it = iter(qs * 2)
        record(name, '3D view, shader renderer: frame after a joint change (pose, upload, draw)',
               timed(lambda: (st.set_q(robot.uid, next(it)), frame())))
        rr.release()
    finally:
        _egl_release(egl)


if __name__ == '__main__':
    for name in ('so100', 'ur5e'):
        measure(name)
    over = [r for r in ROWS if r['over_budget']]
    print(f'{len(over)} UI-thread steps over {BUDGET_MS:g} ms')
    if '--json' in sys.argv:
        path = sys.argv[sys.argv.index('--json') + 1]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump({'rows': ROWS, 'budget_ms': BUDGET_MS}, fh, indent=1)
    QTimer.singleShot(0, app.quit)
