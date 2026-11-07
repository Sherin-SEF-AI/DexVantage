"""UI-thread timings of the Phase 4, 5 and 8 pages on the samples (spec §16: no UI-thread work over 100 ms).

    QT_QPA_PLATFORM=offscreen python scripts/measure_ui_phase458.py [--json out/ui-timings-phase458.json]

The same harness as scripts/measure_ui_phase3.py (imported from it: the watchdog, timed steps, page_steps): for each
page it times showing the page, the result arriving, the 3D and Camera view overlays, Preview and Apply (one undo
step, undone afterwards); the study runs on a worker thread and is listed as "worker". Pages: ground truth, motion
blur, time sync and action-chunk latency on the UR5e sample; lighting on the UR5e sample with a can, a key light and a
window (the lighting checks' scene); teleop on the SO-100 sample; the dataset audit, plan against recording and the
silhouette calibration on the SO-100 sample over the synthetic campaigns in out/ (made first when missing); the
multi-finger page (with the tactile handoff) on the dexhand sample.
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
DATA = os.path.join(ROOT, 'out', 'synth_so100')
DATA_W = os.path.join(ROOT, 'out', 'synth_so100w')


def campaigns():
    from dexvantage.core.sim import episodes as SE
    if not os.path.isdir(DATA):
        SE.generate(sample_project('so100'), DATA, {'episodes': 4, 'frames': 120, 'size': [320, 240], 'seed': 7,
                                                    'faults': {'mount_slip': {'from_episode': 2}}})
    if not os.path.isdir(DATA_W):
        SE.generate(sample_project('so100'), DATA_W, {'episodes': 2, 'frames': 90, 'max_width': 320, 'seed': 3,
                                                      'faults': {**{k: False for k in SE.FAULT_KINDS},
                                                                 'mount_slip': {'from_episode': 1}}})


def measure(label, project, pids, prep=None):
    t0 = time.perf_counter()
    w = MainWindow(project, soft_3d=True, settings=M.settings)
    w.resize(1680, 1000)
    w.show()
    M.record(label, 'build and show the main window (startup)', (time.perf_counter() - t0) * 1000, 'start')
    w.ctx.analysis.wait(10)
    M.WD.pump(0.5)
    vp = w.viewport
    soft_paint = type(vp).paintEvent
    type(vp).paintEvent = lambda self, e: None
    first = w.store.project.variant.sensors[0]
    w.store.select(('sensor', first.uid))
    M.WD.pump(0.5)
    if prep is not None:
        prep(w)
        M.WD.pump(0.5)
    for pid in pids:
        try:
            M.page_steps(label, w, pid)
        except Exception:                          # noqa: BLE001 — report and go on with the next page
            traceback.print_exc()
            M.record(label, f'{pid}: FAILED to measure', 0.0, 'error')
    type(vp).paintEvent = soft_paint
    w.cancel_jobs()
    w.hide()
    w.deleteLater()
    M.app.processEvents()


def datasets(w):
    w.pages['audit']._opened({'path': DATA, 'format': 'lerobot', 'version': 'v3.0', 'episodes': 4})
    w.pages['silhouette'].choose_dataset(DATA_W)


if __name__ == '__main__':
    import json

    from dexvantage.validation.checks_lighting import lit_sample
    campaigns()
    measure('ur5e', sample_project('ur5e'), ('groundtruth', 'motion_blur', 'time_sync', 'action_chunk'))
    measure('ur5e-lit', lit_sample(), ('lighting',))
    measure('so100', sample_project('so100'), ('teleop', 'audit', 'plan_vs_recording', 'silhouette'), datasets)
    measure('dexhand', sample_project('dexhand'), ('dexhand',))
    over = [r for r in M.ROWS if r['over_budget']]
    print(f'{len(over)} UI-thread steps over {M.BUDGET_MS:g} ms')
    if '--json' in sys.argv:
        path = sys.argv[sys.argv.index('--json') + 1]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump({'rows': M.ROWS, 'budget_ms': M.BUDGET_MS}, fh, indent=1)
    M.QTimer.singleShot(0, M.app.quit)
