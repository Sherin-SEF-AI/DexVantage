"""UI-thread timings of the research tools part 2 pages on the ALOHA 2 and G1 samples (spec §16: no UI-thread work
over 100 ms).

    QT_QPA_PLATFORM=offscreen python scripts/measure_ui_rt2.py [--json out/ui-timings-rt2.json]

The harness of scripts/measure_ui_phase3.py (imported: the watchdog, timed steps, page_steps): for each page it times
showing the page, the result arriving, the overlays, Preview and Apply (one undo step, undone afterwards); the study
runs on a worker thread and is listed as "worker". Each sample is saved to a temporary folder with tracking on (so
the Runs page has a store), success counts for the Statistics page, the SO-100 sample as a second cell, a campaign
with two lighting presets and, for the Sim-to-real page, a synthetic recording with a brightness gap rendered from
the sample. Also timed: building and showing the main window, an edit while every page is built (the hidden pages
hold their signals), the Run on… dialog of the Designed experiments and Tolerances pages, and the USD import
wizard's first step on the bundled UR5e (its conversion runs on a worker).
"""
import os
import shutil
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import measure_ui_phase3 as M  # noqa: E402  (creates the application and the harness)

from dexvantage.core.project import Project  # noqa: E402
from dexvantage.core.robot.samples import sample_project  # noqa: E402
from dexvantage.ui.mainwindow import MainWindow  # noqa: E402

ROOT = os.path.join(HERE, '..')
PAGES = ('runs', 'statistics', 'campaign', 'sim2real', 'cells', 'publication')


def saved(name, tmp):
    p = sample_project(name)
    p.analysis.studies['tracking'] = {'enabled': True}
    p.analysis.studies['stats'] = {'source': 'rates', 'rates': [
        {'condition': 'ACT', 'successes': 31, 'trials': 50}, {'condition': 'Diffusion Policy', 'successes': 42,
                                                               'trials': 50}]}
    p.analysis.studies['cells'] = {'cells': [{'sample': 'so100'}], 'metrics': ['rig.camera_count',
                                                                              'coverage.covered']}
    p.analysis.studies['campaign'] = {'operators': ['Ana', 'Ben'], 'lighting': [{'name': 'Bright'}, {'name': 'Dim'}],
                                      'demos': {'mode': 'power', 'p_base': 0.5, 'p_target': 0.7}, 'episode_s': 30}
    try:
        from dexvantage.core.sim2real import inject as IJ
        root = IJ.make_gap_dataset(p, os.path.join(tmp, f's2r-{name}'), {'brightness_dn': 20.0},
                                   {'episodes': 1, 'frames': 24, 'size': [320, 180], 'seed': 4})['root']
        p.analysis.studies['sim2real'] = {'dataset': root, 'mount': False}
    except Exception:                                  # noqa: BLE001 — the page is then measured on its refusal
        traceback.print_exc()
    path = os.path.join(tmp, f'{name}.dexv')
    p.save(path)
    return Project.load(path)


def window(label, project):
    t0 = time.perf_counter()
    w = MainWindow(project, soft_3d=True, settings=M.settings)
    w.resize(1680, 1000)
    w.show()
    M.record(label, 'build and show the main window (startup)', (time.perf_counter() - t0) * 1000, 'start')
    w.ctx.analysis.wait(10)
    M.WD.pump(0.5)
    return w


def extras(label, w):
    from dexvantage.ui import cluster_run as CR
    st = w.store
    uid = st.project.variant.sensors[0].uid

    def edit():
        st.edit('variant', lambda p: setattr(p.variant.sensor(uid).mount.pose, 'position',
                                             p.variant.sensor(uid).mount.pose.position + [0.0, 0.0, 0.001]),
                'Nudge a camera 1 mm (measurement)')
    M.record(label, 'an edit with every page built (store.edit)', M.timed(edit, n=3))
    M.record(label, 'events after the edits (watchdog stall)', M.WD.pump(1.0))
    for pid in ('experiments', 'tolerance'):
        w.open_page(pid)
        M.app.processEvents()
        d = None

        def dlg():
            nonlocal d
            d = CR.RunOnDialog(w, 'Designed experiments' if pid == 'experiments' else 'Monte Carlo tolerances')
        M.record(label, f'{pid}: Run on… dialog opens', M.timed(dlg, n=1))
        d.deleteLater()
    from dexvantage.ui.dialogs.import_wizard import ImportWizard
    path = os.path.join(ROOT, 'dexvantage', 'resources', 'usd_robots', 'newton_ur5e', 'usd_structured', 'ur5e.usda')
    wz = None

    def wizard():
        nonlocal wz
        wz = ImportWizard(w.ctx, w)
        wz.load(path, background=True)
    M.record(label, 'Import robot… wizard opens and starts the USD conversion (worker)', M.timed(wizard, n=1))
    t0 = time.time()
    while getattr(wz, 'robot', None) is None and time.time() - t0 < 60:
        M.WD.pump(0.05)
    M.record(label, 'events while the USD robot converts (watchdog stall)', M.WD.pump(0.5))
    wz.close()
    wz.deleteLater()


def measure(label, project):
    w = window(label, project)
    vp = w.viewport
    soft_paint = type(vp).paintEvent
    type(vp).paintEvent = lambda self, e: None
    w.store.select(('sensor', w.store.project.variant.sensors[0].uid))
    M.WD.pump(0.5)
    for pid in PAGES:
        try:
            M.page_steps(label, w, pid)
        except Exception:                          # noqa: BLE001 — report and go on with the next page
            traceback.print_exc()
            M.record(label, f'{pid}: FAILED to measure', 0.0, 'error')
    try:
        extras(label, w)
    except Exception:                              # noqa: BLE001
        traceback.print_exc()
        M.record(label, 'extras: FAILED to measure', 0.0, 'error')
    type(vp).paintEvent = soft_paint
    w.cancel_jobs()
    w.hide()
    w.deleteLater()
    M.app.processEvents()


if __name__ == '__main__':
    import json
    tmp = tempfile.mkdtemp(prefix='dexvantage-measure-rt2-')
    try:
        for name in ('aloha', 'g1'):
            measure(name, saved(name, tmp))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    over = [r for r in M.ROWS if r['over_budget']]
    print(f'{len(over)} UI-thread steps over {M.BUDGET_MS:g} ms')
    if '--json' in sys.argv:
        path = sys.argv[sys.argv.index('--json') + 1]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump({'rows': M.ROWS, 'budget_ms': M.BUDGET_MS}, fh, indent=1)
    M.QTimer.singleShot(0, M.app.quit)
