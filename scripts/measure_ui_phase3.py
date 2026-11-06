"""UI-thread timings of the Phase 3, 6 and 7 pages on the samples (spec §16: no UI-thread work over 100 ms).

    QT_QPA_PLATFORM=offscreen python scripts/measure_ui_phase3.py [--json out/ui-timings-phase3.json]

For each page (grasp, vla, tasks and policies on the UR5e sample; dexhand on the dexhand sample; bandwidth, cables,
brackets and the hand-eye plan and solve on the UR5e sample; the solve after importing 12 synthetic board images
rendered at the plan's poses) it times on the UI thread: showing the page, the study's result
arriving (show_result), the page's 3D overlays and Camera view overlays (once per frame / image), Preview and Apply
(one undo step, undone afterwards). The study itself runs on a worker thread and is listed as "worker". An
event-loop watchdog (a 5 ms timer, scripts/measure_ui.py) records the longest stall while the events after each step
are processed. Figures are the median of 3 runs with the largest in brackets (Apply: one run). Offscreen the 3D view's
paint is left out, as in scripts/measure_ui.py.
"""
import json
import os
import statistics
import sys
import tempfile
import time
import traceback

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
for k in ('XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_CACHE_HOME'):
    os.environ[k] = tempfile.mkdtemp(prefix='dexvantage-measure-')

from PyQt6.QtCore import QElapsedTimer, QTimer  # noqa: E402

from dexvantage.app import make_app  # noqa: E402
from dexvantage.core.robot.samples import sample_project  # noqa: E402

app, settings = make_app('dark', settings_scope='measure')
from dexvantage.ui.mainwindow import MainWindow  # noqa: E402

BUDGET_MS = 100.0
ROWS: list = []


class Watchdog:
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


def timed(fn, n=3):
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
    print(f'{sample:8s} {where:6s} {med:8.1f} [{worst:7.1f}] ms  {what}' + ('   OVER 100 ms' if over else ''),
          flush=True)


def wait_page(w, page, timeout=300):
    sid = page.STUDY
    w.ctx.analysis.results.pop(sid, None)
    t0 = time.perf_counter()
    page.run()
    while time.perf_counter() - t0 < timeout:
        app.processEvents()
        if not w.ctx.analysis.busy(sid) and sid in w.ctx.analysis.results:
            return (time.perf_counter() - t0) * 1000
        time.sleep(0.002)
    raise RuntimeError(f'{sid} did not finish')


def page_steps(sample, w, pid):
    page = w.pages[pid]
    vp = w.viewport
    record(sample, f'{pid}: show the page', timed(lambda: (w.open_page(pid), app.processEvents())))
    record(sample, f'{pid}: events after showing (watchdog stall)', WD.pump(0.5))
    if page.STUDY:
        record(sample, f'{pid}: study {page.STUDY}, whole run', wait_page(w, page), 'worker')
        doc = w.ctx.analysis.results[page.STUDY]
        record(sample, f'{pid}: the result arrives (show_result)', timed(lambda: page._finished(page.STUDY, doc)))
        record(sample, f'{pid}: events after the result (watchdog stall)', WD.pump(1.0))
    record(sample, f'{pid}: 3D overlays for one frame', timed(lambda: page.overlays(vp)))
    view = w.camview.view
    view._overlays = None
    record(sample, f'{pid}: Camera view overlays, once per image',
           timed(lambda: (setattr(view, '_overlays', None), view.page_overlays())))
    if page.b_preview.isEnabled():
        record(sample, f'{pid}: Preview', timed(page.preview, n=1))
        record(sample, f'{pid}: events after Preview (watchdog stall)', WD.pump(0.5))
    if page.b_apply.isEnabled():
        n = w.store.undo.count()
        record(sample, f'{pid}: Apply (one undo step)', timed(page.apply, n=1))
        record(sample, f'{pid}: events after Apply (watchdog stall)', WD.pump(1.0))
        if w.store.undo.count() > n:
            w.store.undo.undo()
            WD.pump(0.3)


def synthetic_capture(w):
    """A capture to solve: board images rendered at 12 of the hand-eye plan's poses and their joint_states file
    (as tests/test_calib_planner.py renders them), imported on the solve page."""
    sys.path.insert(0, os.path.join(ROOT, 'tests'))
    from test_calib_planner import render_capture

    from dexvantage.core.calib import planner as PL
    p = w.ctx.store.project
    poses = (w.ctx.analysis.results.get('handeye_plan') or {}).get('poses') or []
    tmp = tempfile.mkdtemp(prefix='dexvantage-measure-capture-')
    render_capture(p, PL.rig_from_project(p), poses[:12], os.path.join(tmp, 'img'), os.path.join(tmp, 'joints.csv'))
    sv = w.pages['handeye_solve']
    sv.source.setCurrentIndex(sv.source.findData('folder'))
    t0 = time.perf_counter()
    sv.import_capture(os.path.join(tmp, 'img'), os.path.join(tmp, 'joints.csv'))
    record('ur5e', 'handeye_solve: import 12 images (detection runs here)', (time.perf_counter() - t0) * 1000,
           'import')


def measure(sample, pids):
    t0 = time.perf_counter()
    w = MainWindow(sample_project(sample), soft_3d=True, settings=settings)
    w.resize(1680, 1000)
    w.show()
    record(sample, 'build and show the main window (startup)', (time.perf_counter() - t0) * 1000, 'start')
    w.ctx.analysis.wait(10)
    WD.pump(0.5)
    vp = w.viewport
    soft_paint = type(vp).paintEvent
    type(vp).paintEvent = lambda self, e: None
    first = w.store.project.variant.sensors[0]
    w.store.select(('sensor', first.uid))
    WD.pump(0.5)
    for pid in pids:
        try:
            if pid == 'handeye_solve':
                synthetic_capture(w)
            page_steps(sample, w, pid)
        except Exception:                          # noqa: BLE001 — report and go on with the next page
            traceback.print_exc()
            record(sample, f'{pid}: FAILED to measure', 0.0, 'error')
    type(vp).paintEvent = soft_paint
    w.cancel_jobs()
    w.hide()
    w.deleteLater()
    app.processEvents()


if __name__ == '__main__':
    measure('ur5e', ('tasks', 'policies', 'grasp', 'vla', 'bandwidth', 'cables', 'brackets', 'handeye_plan',
                     'handeye_solve'))
    measure('dexhand', ('dexhand',))
    over = [r for r in ROWS if r['over_budget']]
    print(f'{len(over)} UI-thread steps over {BUDGET_MS:g} ms')
    if '--json' in sys.argv:
        path = sys.argv[sys.argv.index('--json') + 1]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump({'rows': ROWS, 'budget_ms': BUDGET_MS}, fh, indent=1)
    QTimer.singleShot(0, app.quit)
