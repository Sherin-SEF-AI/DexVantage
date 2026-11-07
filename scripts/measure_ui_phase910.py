"""UI-thread timings of the Phase 9 and 10 pages on the samples (spec §16: no UI-thread work over 100 ms).

    QT_QPA_PLATFORM=offscreen python scripts/measure_ui_phase910.py [--json out/ui-timings-phase910.json]

The same harness as scripts/measure_ui_phase3.py (imported from it: the watchdog, timed steps, page_steps): for each
page it times showing the page, the result arriving, the 3D and Camera view overlays, Preview and Apply (one undo
step, undone afterwards); the study runs on a worker thread and is listed as "worker". Pages on the UR5e sample:
tolerances (24 draws), designed experiments (a Saltelli design of 8 base samples over the Overhead D435's pitch and
height, response coverage.covered), the optimiser (24 designs), requirements and variants (the labelled defaults),
the canonical view. Also timed: building and showing the main window of each new sample (ALOHA 2, bimanual Panda,
G1), opening the Export dialog and the Plugins dialog, and the Compare-with window's diff arriving.
"""
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import measure_ui_phase3 as M  # noqa: E402  (creates the application and the harness)

from dexvantage.core.robot.samples import sample_project  # noqa: E402
from dexvantage.ui.mainwindow import MainWindow  # noqa: E402

ROOT = os.path.join(HERE, '..')


def window(label, project):
    t0 = time.perf_counter()
    w = MainWindow(project, soft_3d=True, settings=M.settings)
    w.resize(1680, 1000)
    w.show()
    M.record(label, 'build and show the main window (startup)', (time.perf_counter() - t0) * 1000, 'start')
    w.ctx.analysis.wait(10)
    M.WD.pump(0.5)
    return w


def close(w):
    w.cancel_jobs()
    w.hide()
    w.deleteLater()
    M.app.processEvents()


def prep(w):
    st = w.store
    st.edit('analysis', lambda p: p.analysis.studies.update({
        'tolerance': {'n': 24, 'seed': 1}, 'optimise': {'budget': 24, 'population': 8, 'seed': 1}}),
        'Measurement settings')
    page = w.pages['experiments']
    w.open_page('experiments')
    M.app.processEvents()
    over = next(s for s in st.project.variant.sensors if s.name == 'Overhead D435')
    page.add_factor(f'sensor:{over.uid}.dpitch')
    page.set_factor(f'sensor:{over.uid}.dpitch', lo=-10.0, hi=10.0)
    page.add_factor(f'sensor:{over.uid}.dz')
    page.set_factor(f'sensor:{over.uid}.dz', lo=-0.1, hi=0.1)
    page.set_responses(['coverage.covered'])
    page._set_design(method='saltelli', text='design type')
    page._set_design(n=8, text='n')


def dialogs(label, w):
    from dexvantage.core.project import Sensor
    from dexvantage.ui import versioning as V
    from dexvantage.ui.dialogs import export as DX
    from dexvantage.ui.dialogs import plugins as DP
    d = None

    def ex():
        nonlocal d
        d = DX.open_dialog(w)
    M.record(label, 'File ▸ Export… opens', M.timed(ex, n=1))
    M.record(label, 'events after the Export dialog opens (watchdog stall)', M.WD.pump(0.5))
    d.close()
    pd = None

    def pl():
        nonlocal pd
        pd = DP.open_dialog(w, modal=False)
    M.record(label, 'Help ▸ Plugins… opens', M.timed(pl, n=1))
    pd.close()
    other = w.store.project.snapshot()
    other.variant.sensors.append(Sensor(name='Side camera'))
    path = os.path.join(ROOT, 'out', 'p910', 'measure_other.dexv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    other.save(path)
    cw = None

    def cmp():
        nonlocal cw
        cw = V.compare(w, path)
    M.record(label, 'File ▸ Compare with… opens', M.timed(cmp, n=1))
    t0 = time.time()
    while cw.diff is None and time.time() - t0 < 30:
        M.WD.pump(0.05)
    M.record(label, 'events while the diff arrives (watchdog stall)', M.WD.pump(0.5))
    cw.close()


def measure(label, project, pids, before=None):
    w = window(label, project)
    vp = w.viewport
    soft_paint = type(vp).paintEvent
    type(vp).paintEvent = lambda self, e: None
    first = w.store.project.variant.sensors[0]
    w.store.select(('sensor', first.uid))
    M.WD.pump(0.5)
    if before is not None:
        before(w)
        M.WD.pump(0.5)
    for pid in pids:
        try:
            M.page_steps(label, w, pid)
        except Exception:                          # noqa: BLE001 — report and go on with the next page
            traceback.print_exc()
            M.record(label, f'{pid}: FAILED to measure', 0.0, 'error')
    try:
        dialogs(label, w)
    except Exception:                              # noqa: BLE001
        traceback.print_exc()
        M.record(label, 'dialogs: FAILED to measure', 0.0, 'error')
    type(vp).paintEvent = soft_paint
    close(w)


if __name__ == '__main__':
    import json
    measure('ur5e', sample_project('ur5e'), ('tolerance', 'experiments', 'optimise', 'requirements',
                                             'canonical_view'), prep)
    for name in ('aloha', 'panda-bimanual', 'g1'):
        close(window(name, sample_project(name)))
    over = [r for r in M.ROWS if r['over_budget']]
    print(f'{len(over)} UI-thread steps over {M.BUDGET_MS:g} ms')
    if '--json' in sys.argv:
        path = sys.argv[sys.argv.index('--json') + 1]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump({'rows': M.ROWS, 'budget_ms': M.BUDGET_MS}, fh, indent=1)
    M.QTimer.singleShot(0, M.app.quit)
