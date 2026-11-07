"""Offscreen screenshots for the review gates: out/screens/*.png. Phase 0: the window, dialogs and palette. Phase 1:
for the SO-100 and UR5e samples, the main window, the Robot page and the import wizard (each of its six steps).
Phase 2 (phase2-*): the Sensors and mounts page, the Coverage and Depth validity pages after a run, the Camera view
of the UR5e sample's Wrist D405 and Overhead D435 (and the SO-100's two cameras), the Library panel and the New
custom device dialog. Each Camera view image is also saved full size on its own (phase2-camview-*.png).
Phases 3, 6 and 7 (phase3-*): on the UR5e sample the Tasks page (its grasp set previewed), the Policies page, the
grasp self-occlusion page after a run (with its worst-case gallery), the VLA token page and the Camera view with the
patch grid, the bandwidth page with a host topology, the cables page, the brackets page with the generated bracket in
the 3D view, the hand-eye plan and solve pages (a synthetic capture rendered and imported); on the dexhand sample the
multi-finger occlusion matrix.
Phases 4, 5 and 8 (p458-*): on the UR5e sample the ground-truth page and its confusion matrix, motion blur, time
sync, action-chunk latency (its overshoot map and curve); the lighting page and its glare map in the Camera view on
the UR5e sample with a can and a key light (the lighting checks' scene); on the SO-100 sample teleop dead zones, the
dataset audit, plan against recording and the silhouette calibration over synthetic campaigns written into out/; on
the dexhand sample the visual-to-tactile handoff chart.
The ALOHA 2, bimanual Panda and G1 samples (sample-aloha-*, sample-panda-bimanual-*, sample-g1-*): the main window,
each camera's Camera view, the Robot page and, for the two bimanual cells, the teleop page after a run.
Phases 9 and 10 and research tools part 1 (p910-*): tolerances, designed experiments with Sobol bars, the optimiser's
Pareto front, the requirements scorecard and variants compare, the export dialog, the canonical view, the assistant
dock with a scripted fake-provider conversation, the plugins dialog, the Library's robots, compare-with (a project
diff) and each new sample in 3D.
Research tools part 2 (rt2-*): the Runs page with a comparison, Statistics with a power curve, the campaign plan on
ALOHA 2, the sim-to-real gap with recorded against rendered frames, the paper kit preview, the cells comparison of three
samples, the Run on… dialog and the USD import wizard on the bundled UR5e.

    QT_QPA_PLATFORM=offscreen python scripts/screens.py [--only phase3 | phase458 | samples | phase910 | research2]
"""
import atexit
import os
import sys
import tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
_TEMP = []                                              # throw-away folders, removed when the script ends


def _mkdtemp(prefix):
    d = tempfile.mkdtemp(prefix=prefix)
    _TEMP.append(d)
    return d


def _remove_temp():
    import shutil
    for d in _TEMP:
        shutil.rmtree(d, ignore_errors=True)


atexit.register(_remove_temp)
for k in ('XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_CACHE_HOME'):
    os.environ[k] = _mkdtemp('dexvantage-screens-')

from PyQt6.QtCore import Qt, QTimer  # noqa: E402

from dexvantage.app import make_app  # noqa: E402
from dexvantage.core import project as P  # noqa: E402
from dexvantage.core.frames import Pose  # noqa: E402

OUT = os.path.join(ROOT, 'out', 'screens')
os.makedirs(OUT, exist_ok=True)
app, settings = make_app('dark', settings_scope='screens')
from dexvantage.ui.mainwindow import MainWindow  # noqa: E402

p = P.blank_project('Tabletop cell')
p.workspace.objects.append(P.TaskObject(name='Mug', size=(0.08, 0.08, 0.1), pose=Pose([0.42, 0.05, 0.05]),
                                        materials=['glossy']))
w = MainWindow(p, soft_3d=True, settings=settings)
w.resize(1680, 1000)
w.show()
w.add_static_camera()
w.add_static_camera()
w.store.select(('volume', p.workspace.volumes[0].uid))
w.ctx.analysis.wait(5)


def pump(n=20):
    for _ in range(n):
        app.processEvents()


def shot(widget, name):
    pump()
    widget.grab().save(os.path.join(OUT, name))
    print(name)



# ---------------------------------------------------------------- Phases 3, 6 and 7: the new pages
def phase3():
    import time

    from dexvantage.core.robot.samples import sample_project as sp

    def wait_sid(win, sid, timeout=300.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            pump(5)
            if sid in win.ctx.analysis.results and not win.ctx.analysis.busy():
                return True
            time.sleep(0.02)
        return False

    def window(name):
        pw = MainWindow(sp(name), soft_3d=True, settings=settings)
        pw.resize(1680, 1000)
        pw.show()
        pw.ctx.analysis.wait(5)
        pw.resizeDocks([pw.docks['inspector'], pw.docks['camview']], [380, 460], Qt.Orientation.Vertical)
        pw.resizeDocks([pw.docks['inspector']], [430], Qt.Orientation.Horizontal)
        pw.resizeDocks([pw.docks['bottom']], [470], Qt.Orientation.Vertical)
        pump()
        pw.viewport.frame_box(*pw.scene.home_box(pw.viewport))
        return pw

    def run_page(pw, pid, tag, timeout=300.0):
        pw.open_page(pid)
        pump()
        page = pw.pages[pid]
        if page.STUDY:
            pw.ctx.analysis.results.pop(page.STUDY, None)
            page.run()
            ok = wait_sid(pw, page.STUDY, timeout)
        else:
            ok = True
        pump(40)
        shot(pw, f'phase3-{tag}.png' if ok else f'phase3-{tag}-NOT-FINISHED.png')
        return page

    def camview(pw, sensor_name, tag):
        s = next(s for s in pw.ctx.store.project.variant.sensors if s.name == sensor_name)
        pw.ctx.store.select(('sensor', s.uid))
        pump()
        pw.camview.render_sync()
        pw.camview.wait(20)
        pump(40)
        shot(pw.docks['camview'], f'phase3-camview-{tag}.png')

    pw = window('ur5e')
    # Tasks: the sample's grasp set, and a sampled set previewed in 3D
    tasks = pw.pages['tasks']
    pw.open_page('tasks')
    tasks.set_form(kinds=['top_down', 'side', 'angled'], n=60, rng_seed=1)
    tasks.preview()
    shot(pw, 'phase3-ur5e-tasks-page.png')
    tasks.preview()                                  # preview off again
    run_page(pw, 'policies', 'ur5e-policies-page')
    camview(pw, 'Overhead D435', 'policies-overhead')
    # the grasp study (the sample's 200 grasps) and its worst-case gallery
    g = run_page(pw, 'grasp', 'ur5e-grasp-page', timeout=600)
    pw.resizeDocks([pw.docks['bottom']], [900], Qt.Orientation.Vertical)
    pump()
    sc = g.body.parentWidget().parentWidget()
    sc.ensureWidgetVisible(g.gallery)
    pump()
    shot(g, 'phase3-ur5e-grasp-gallery.png')
    pw.resizeDocks([pw.docks['bottom']], [470], Qt.Orientation.Vertical)
    camview(pw, 'Wrist D405', 'grasp-wrist')
    run_page(pw, 'vla', 'ur5e-vla-page')
    camview(pw, 'Overhead D435', 'vla-overhead')
    camview(pw, 'Wrist D405', 'vla-wrist')
    # bandwidth: Apply the suggested host, then the topology and margins
    bw = run_page(pw, 'bandwidth', 'ur5e-bandwidth-page-no-host')
    if bw.b_apply.isEnabled():
        bw.apply()
        pump(20)
        bw.run()
        wait_sid(pw, bw.STUDY)
        bw.topo.expandAll()
        pump(40)
    shot(pw, 'phase3-ur5e-bandwidth-topology.png')
    cb = run_page(pw, 'cables', 'ur5e-cables-page')
    camview(pw, 'Wrist D405', 'cables-wrist')
    del cb
    # brackets: Apply the suggested bracket kinds, run (meshes them), the bracket in the 3D view
    br = run_page(pw, 'brackets', 'ur5e-brackets-page-before')
    if br.b_apply.isEnabled():
        br.apply()
        pump(20)
        br.run()
        wait_sid(pw, br.STUDY, 300)
        pump(40)
    pw.viewport.frame_box(*pw.scene.home_box(pw.viewport))
    shot(pw, 'phase3-ur5e-brackets-preview.png')
    # hand-eye: plan, save it, render a synthetic capture at the planned poses, import, solve
    hp = run_page(pw, 'handeye_plan', 'ur5e-handeye-plan')
    camview(pw, 'Wrist D405', 'handeye-plan-wrist')
    try:
        sys.path.insert(0, os.path.join(ROOT, 'tests'))
        from test_calib_planner import render_capture
        from dexvantage.core.calib import planner as PL
        p = pw.ctx.store.project
        rig = PL.rig_from_project(p)
        poses = (hp.doc or {}).get('poses') or []
        tmp = _mkdtemp('dexvantage-screens-capture-')
        render_capture(p, rig, poses[:12], os.path.join(tmp, 'img'), os.path.join(tmp, 'joints.csv'))
        sv = pw.pages['handeye_solve']
        pw.open_page('handeye_solve')
        pump()
        sv.source.setCurrentIndex(sv.source.findData('folder'))
        sv.import_capture(os.path.join(tmp, 'img'), os.path.join(tmp, 'joints.csv'))
        run_page(pw, 'handeye_solve', 'ur5e-handeye-solve')
    except Exception:                                   # noqa: BLE001 — the other screens still matter
        import traceback
        traceback.print_exc()
    pw.cancel_jobs()
    pw.hide()

    dw = window('dexhand')
    run_page(dw, 'dexhand', 'dexhand-matrix-page', timeout=300)
    first = dw.ctx.store.project.variant.sensors[0]
    camview(dw, first.name, 'dexhand-first-camera')
    dw.cancel_jobs()
    dw.hide()


# ---------------------------------------------------------------- Phases 4, 5 and 8: ground truth, timing, data
def phase458():
    """p458-*: the new pages of Phases 4, 5 and 8 after a run, with their Camera view overlays."""
    import time

    from dexvantage.core.robot.samples import sample_project as sp
    from dexvantage.core.sim import episodes as SE
    from dexvantage.validation.checks_lighting import lit_sample

    def wait_sid(win, sid, timeout=300.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            pump(5)
            if sid in win.ctx.analysis.results and not win.ctx.analysis.busy():
                return True
            time.sleep(0.02)
        return False

    def window(project):
        pw = MainWindow(project, soft_3d=True, settings=settings)
        pw.resize(1680, 1000)
        pw.show()
        pw.ctx.analysis.wait(5)
        pw.resizeDocks([pw.docks['inspector'], pw.docks['camview']], [380, 460], Qt.Orientation.Vertical)
        pw.resizeDocks([pw.docks['inspector']], [430], Qt.Orientation.Horizontal)
        pw.resizeDocks([pw.docks['bottom']], [470], Qt.Orientation.Vertical)
        pump()
        pw.viewport.frame_box(*pw.scene.home_box(pw.viewport))
        return pw

    def run_page(pw, pid, tag, timeout=600.0):
        pw.open_page(pid)
        pump()
        page = pw.pages[pid]
        pw.ctx.analysis.results.pop(page.STUDY, None)
        page.run()
        t0 = time.time()                         # pages that fingerprint a dataset first start the study a bit later
        while page.STUDY not in pw.ctx.analysis.results and not pw.ctx.analysis.busy() and time.time() - t0 < 30:
            pump(5)
            time.sleep(0.02)
        ok = wait_sid(pw, page.STUDY, timeout)
        pump(40)
        shot(pw, f'p458-{tag}.png' if ok else f'p458-{tag}-NOT-FINISHED.png')
        return page

    def camview(pw, sensor_name, tag):
        s = next(s for s in pw.ctx.store.project.variant.sensors if s.name == sensor_name)
        pw.ctx.store.select(('sensor', s.uid))
        pump()
        pw.camview.render_sync()
        pw.camview.wait(20)
        pump(60)
        time.sleep(0.5)                          # page overlays rendered in the background (ground truth, silhouette)
        pump(60)
        pw.camview.render_sync()
        pump(40)
        shot(pw.docks['camview'], f'p458-camview-{tag}.png')

    def tall(pw, page, widget, name):
        pw.resizeDocks([pw.docks['bottom']], [900], Qt.Orientation.Vertical)
        pump()
        page.body.parentWidget().parentWidget().ensureWidgetVisible(widget)
        pump(20)
        shot(page, name)
        pw.resizeDocks([pw.docks['bottom']], [470], Qt.Orientation.Vertical)
        pump()

    # the synthetic campaigns the data pages read (made once into out/)
    data = os.path.join(ROOT, 'out', 'synth_so100')
    data_w = os.path.join(ROOT, 'out', 'synth_so100w')
    if not os.path.isdir(data):
        SE.generate(sp('so100'), data, {'episodes': 4, 'frames': 120, 'size': [320, 240], 'seed': 7,
                                        'faults': {'mount_slip': {'from_episode': 2}}})
    if not os.path.isdir(data_w):
        SE.generate(sp('so100'), data_w, {'episodes': 2, 'frames': 90, 'max_width': 320, 'seed': 3,
                                          'faults': {**{k: False for k in SE.FAULT_KINDS},
                                                     'mount_slip': {'from_episode': 1}}})

    pw = window(sp('ur5e'))
    gt = run_page(pw, 'groundtruth', 'ur5e-groundtruth-page')
    tall(pw, gt, gt.matrix, 'p458-ur5e-groundtruth-matrix.png')
    camview(pw, 'Wrist D405', 'groundtruth-wrist')
    mb = run_page(pw, 'motion_blur', 'ur5e-motion-blur-page')
    if mb.show_frame():                          # the worst camera's kept frame: its smear vectors in the Camera view
        pw.camview.render_sync()
        pw.camview.wait(20)
        pump(40)
        shot(pw.docks['camview'], 'p458-camview-motion-blur-frame.png')
    camview(pw, 'Wrist D405', 'motion-blur-wrist')
    ts = run_page(pw, 'time_sync', 'ur5e-time-sync-page')
    ts.show_worst()
    camview(pw, 'Wrist D405', 'time-sync-wrist')
    ac = run_page(pw, 'action_chunk', 'ur5e-action-chunk-page')
    if ac.b_show_map.isEnabled():
        ac.b_show_map.click()
        pump(40)
    pw.viewport.frame_box(*pw.scene.home_box(pw.viewport))
    shot(pw, 'p458-ur5e-action-chunk-overshoot-map.png')
    tall(pw, ac, ac.curve, 'p458-ur5e-action-chunk-curve.png')
    pw.cancel_jobs()
    pw.hide()

    lw = window(lit_sample())
    run_page(lw, 'lighting', 'ur5e-lighting-page')
    camview(lw, 'Overhead D435', 'lighting-overhead')
    camview(lw, 'Wrist D405', 'lighting-wrist')
    lw.cancel_jobs()
    lw.hide()

    sw = window(sp('so100'))
    tp = run_page(sw, 'teleop', 'so100-teleop-page')
    tall(sw, tp, tp.zone_list, 'p458-so100-teleop-no-zones.png')

    def scaled(p):                               # a leader mapped at 1.25 × reaches past the follower's joint limits
        d = p.tasks.teleop_devices[0]
        p.tasks.teleop_devices[0] = dict(d, mapping=dict(d.get('mapping') or {}, scale=1.25))
    sw.ctx.store.edit('tasks', scaled, 'Teleop · leader scale 1.25 (screenshot)')
    run_page(sw, 'teleop', 'so100-teleop-dead-zones')
    tall(sw, tp, tp.zone_list, 'p458-so100-teleop-zones.png')
    sw.ctx.store.undo.undo()
    au = sw.pages['audit']
    sw.open_page('audit')
    au._opened({'path': data, 'format': 'lerobot', 'version': 'v3.0', 'episodes': 4})
    run_page(sw, 'audit', 'so100-audit-page')
    run_page(sw, 'plan_vs_recording', 'so100-plan-vs-recording-page')
    camview(sw, 'Third-person camera (sample placement)', 'plan-vs-recording-third-person')
    si = sw.pages['silhouette']
    sw.open_page('silhouette')
    si.choose_dataset(data_w)
    run_page(sw, 'silhouette', 'so100-silhouette-page')
    if si.b_preview.isEnabled():
        si.preview()
        pump(20)
    camview(sw, 'Third-person camera (sample placement)', 'silhouette-third-person')
    sw.cancel_jobs()
    sw.hide()

    dw = window(sp('dexhand'))
    dh = run_page(dw, 'dexhand', 'dexhand-page')
    tall(dw, dh, dh.handoff, 'p458-dexhand-handoff-chart.png')
    dw.cancel_jobs()
    dw.hide()


# ---------------------------------------------------------------- the ALOHA 2, bimanual Panda and G1 samples
def new_samples():
    """sample-<name>-*: the ALOHA 2, bimanual Panda and G1 samples: the main window, the Robot page, each camera's
    Camera view, and the teleop page (the bimanual handover) after a run."""
    import time

    from dexvantage.core.robot.samples import sample_project as sp

    def wait_sid(win, sid, timeout=600.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            pump(5)
            if sid in win.ctx.analysis.results and not win.ctx.analysis.busy():
                return True
            time.sleep(0.02)
        return False

    for name in ('aloha', 'panda-bimanual', 'g1'):
        sw = MainWindow(sp(name), soft_3d=True, settings=settings)
        sw.resize(1680, 1000)
        sw.show()
        sw.ctx.analysis.wait(5)
        sw.resizeDocks([sw.docks['inspector'], sw.docks['camview']], [380, 460], Qt.Orientation.Vertical)
        pump()
        sw.viewport.frame_box(*sw.scene.home_box(sw.viewport))
        shot(sw, f'sample-{name}-main.png')
        for s in sw.ctx.store.project.variant.sensors:
            sw.ctx.store.select(('sensor', s.uid))
            pump()
            sw.camview.render_sync()
            sw.camview.wait(20)
            pump(40)
            tag = ''.join(c if c.isalnum() else '-' for c in s.name.lower()).strip('-')
            while '--' in tag:
                tag = tag.replace('--', '-')
            shot(sw.docks['camview'], f'sample-{name}-camview-{tag}.png')
        sw.ctx.store.select(())
        sw.open_page('robot')
        sw.resizeDocks([sw.docks['bottom']], [520], Qt.Orientation.Vertical)
        pump()
        sw.viewport.frame_box(*sw.scene.home_box(sw.viewport))
        shot(sw, f'sample-{name}-robot-page.png')
        if name != 'g1':
            sw.open_page('teleop')
            pump()
            page = sw.pages['teleop']
            sw.ctx.analysis.results.pop(page.STUDY, None)
            page.run()
            ok = wait_sid(sw, page.STUDY)
            pump(40)
            shot(sw, f'sample-{name}-teleop-page.png' if ok else f'sample-{name}-teleop-page-NOT-FINISHED.png')
        sw.cancel_jobs()
        sw.hide()


# ---------------------------------------------------------------- Phases 9 and 10, samples, research tools part 1
def phase910():
    """p910-*: tolerances, designed experiments (Sobol bars), the optimiser's Pareto front, the requirements
    scorecard and variants compare, the export dialog, the canonical view, the assistant dock with a scripted
    (fake-provider) conversation, the plugins dialog, the robot catalogue, compare-with (project diff) and each new
    sample in 3D, on the UR5e sample unless named."""
    import time

    from dexvantage.assistant.fake import FakeReply
    from dexvantage.core.robot.samples import sample_project as sp

    def wait_sid(win, sid, timeout=600.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            pump(5)
            if sid in win.ctx.analysis.results and not win.ctx.analysis.busy():
                return True
            time.sleep(0.02)
        return False

    def window(project):
        pw = MainWindow(project, soft_3d=True, settings=settings)
        pw.resize(1680, 1000)
        pw.show()
        pw.ctx.analysis.wait(5)
        pw.resizeDocks([pw.docks['inspector'], pw.docks['camview']], [380, 460], Qt.Orientation.Vertical)
        pw.resizeDocks([pw.docks['inspector']], [430], Qt.Orientation.Horizontal)
        pw.resizeDocks([pw.docks['bottom']], [520], Qt.Orientation.Vertical)
        pump()
        pw.viewport.frame_box(*pw.scene.home_box(pw.viewport))
        return pw

    def run_page(pw, pid, tag, timeout=900.0):
        pw.open_page(pid)
        pump()
        page = pw.pages[pid]
        pw.ctx.analysis.results.pop(page.STUDY, None)
        page.run()
        t0 = time.time()
        while page.STUDY not in pw.ctx.analysis.results and not pw.ctx.analysis.busy() and time.time() - t0 < 30:
            pump(5)
            time.sleep(0.02)
        ok = wait_sid(pw, page.STUDY, timeout)
        pump(40)
        shot(pw, f'p910-{tag}.png' if ok else f'p910-{tag}-NOT-FINISHED.png')
        return page

    def tall(pw, page, name, widget=None):
        pw.resizeDocks([pw.docks['bottom']], [900], Qt.Orientation.Vertical)
        pump()
        if widget is not None:
            page.body.parentWidget().parentWidget().ensureWidgetVisible(widget)
        pump(20)
        shot(page, name)
        pw.resizeDocks([pw.docks['bottom']], [520], Qt.Orientation.Vertical)
        pump()

    def part(fn):
        try:
            fn()
        except Exception:                        # noqa: BLE001 — one broken shot must not stop the others
            import traceback
            traceback.print_exc()

    pw = window(sp('ur5e'))
    st = pw.ctx.store

    def tolerance():
        st.edit('analysis', lambda p: p.analysis.studies.__setitem__('tolerance', {'n': 24, 'seed': 1}),
                'Tolerance settings (screenshot)')
        page = run_page(pw, 'tolerance', 'ur5e-tolerance-page')
        tall(pw, page, 'p910-ur5e-tolerance-chart.png', getattr(page, 'chart', None))
        pw.viewport.frame_box(*pw.scene.home_box(pw.viewport))
        shot(pw.viewport, 'p910-ur5e-tolerance-3d.png')

    def experiments():
        page = pw.pages['experiments']
        pw.open_page('experiments')
        pump()
        over = next(s for s in st.project.variant.sensors if s.name == 'Overhead D435')
        page.add_factor(f'sensor:{over.uid}.dpitch')
        page.set_factor(f'sensor:{over.uid}.dpitch', lo=-10.0, hi=10.0)
        page.add_factor(f'sensor:{over.uid}.dz')
        page.set_factor(f'sensor:{over.uid}.dz', lo=-0.1, hi=0.1)
        page.set_responses(['coverage.covered'])
        page._set_design(method='saltelli', text='design type')
        page._set_design(n=32, text='n')       # 128 runs: with N = 8 the indices' intervals span the axis
        pump()
        run_page(pw, 'experiments', 'ur5e-experiments-page')
        tall(pw, page, 'p910-ur5e-experiments-sobol.png', getattr(page, 'sobol', None))

    def optimise():
        st.edit('analysis', lambda p: p.analysis.studies.__setitem__('optimise', {'budget': 24, 'population': 8,
                                                                              'seed': 1}),
                'Optimiser settings (screenshot)')
        page = run_page(pw, 'optimise', 'ur5e-optimise-pareto')
        tall(pw, page, 'p910-ur5e-optimise-chart.png', getattr(page, 'chart', None))

    def requirements():
        pw.open_page('requirements')
        pump()
        pw.pages['requirements'].add_defaults()             # the labelled defaults (one undo step)
        pump()
        page = run_page(pw, 'requirements', 'ur5e-requirements-scorecard')
        tall(pw, page, 'p910-ur5e-requirements-table.png', getattr(page, 'table', None))

        def second(p):
            v = P.Variant.from_dict(p.variant.to_dict())
            v.uid, v.name = P.new_uid(), 'Overhead raised 10 cm (screenshot)'
            for s in v.sensors:
                s.uid = P.new_uid()
                if s.name == 'Overhead D435':
                    s.mount.pose.position[2] += 0.1
            p.variants.append(v)
        st.edit('variants', second, 'Add a variant (screenshot)')
        page.compare()
        t0 = time.time()
        while time.time() - t0 < 600 and (pw.ctx.analysis.busy() or 'variants_compare' not in
                                          pw.ctx.analysis.results):
            pump(5)
            time.sleep(0.05)
        pump(40)
        tall(pw, page, 'p910-ur5e-variants-compare.png', getattr(page, 'b_compare', None))

    def export_dialog():
        from dexvantage.ui.dialogs import export as DX
        d = DX.open_dialog(pw)
        d.set_chosen(['mjcf', 'urdf', 'isaac', 'lerobot', 'report'])
        d.resize(900, 900)
        pump(20)
        shot(d, 'p910-export-dialog.png')
        out = _mkdtemp('dexvantage-export-')      # no personal path in the picture
        d.folder.setText(out)
        d.set_chosen(['mjcf', 'opencv_yaml', 'perturbations'])
        if d.start():
            t0 = time.time()
            while time.time() - t0 < 300 and getattr(pw, '_jobs', None):
                pump(5)
                time.sleep(0.05)
            pump(20)
            shot(d, 'p910-export-dialog-done.png')
        d.close()

    def canonical():
        page = run_page(pw, 'canonical_view', 'ur5e-canonical-view')
        tall(pw, page, 'p910-ur5e-canonical-view-map.png')

    def assistant():
        pw.docks['assistant'].show()
        pw.resizeDocks([pw.docks['assistant']], [460], Qt.Orientation.Horizontal)
        dock = pw.assistant
        pw.open_page('tolerance')
        pump()

        def final(results):
            h = dict(results)['get_page_context']['shown']['headline']
            return (f'The Tolerances page shows {h.get("figure")} {h.get("unit", "")} ({h.get("word", "")}). '
                    'Move the mount with the widest spread first.')
        dock.ctl.use_script([[FakeReply(calls=[('get_page_context', {})]), FakeReply(text=final)]])
        dock.input.setPlainText('What does this page say, and what should I change first?')
        dock.send()
        dock.ctl.wait(30)
        pump(40)
        shot(pw, 'p910-assistant-dock.png')
        shot(pw.docks['assistant'], 'p910-assistant-dock-only.png')
        pw.docks['assistant'].hide()

    def plugins():
        from dexvantage.ui.dialogs import plugins as DP
        d = DP.open_dialog(pw, modal=False)
        d.resize(820, 520)
        pump(20)
        shot(d, 'p910-plugins-dialog.png')
        d.close()

    def catalogue():
        pw.docks['library'].show()
        pw.resizeDocks([pw.docks['outliner'], pw.docks['library']], [200, 760], Qt.Orientation.Vertical)
        pw.resizeDocks([pw.docks['library']], [520], Qt.Orientation.Horizontal)
        tree = pw.library.tree
        for i in range(tree.topLevelItemCount()):
            tree.topLevelItem(i).setExpanded(tree.topLevelItem(i).text(0) == 'Robots')
        pump()
        shot(pw.docks['library'], 'p910-library-panel-robots.png')

    def compare():
        from dexvantage.ui import versioning as V
        other = st.project.snapshot()
        other.variant.sensors[-1].mount.pose.position[2] += 0.05
        other.variant.sensors.append(P.Sensor(name='Side camera'))
        other.notes = 'Overhead raised 5 cm and a side camera (screenshot)'
        path = os.path.join(ROOT, 'out', 'p910', 'compare_other.dexv')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        other.save(path)
        cw = V.compare(pw, path)
        t0 = time.time()
        while cw.diff is None and time.time() - t0 < 30:
            pump(5)
            time.sleep(0.02)
        cw.resize(1100, 700)
        pump(20)
        shot(cw, 'p910-compare-with.png')
        cw.close()

    def review_additions():                      # added by the reviews: the Workspace page, the Library's new groups
        pw.open_page('workspace')
        pump(20)
        shot(pw, 'p910-ur5e-workspace-page.png')
        pw.docks['library'].show()
        tree = pw.library.tree
        for i in range(tree.topLevelItemCount()):
            tree.topLevelItem(i).setExpanded(tree.topLevelItem(i).text(0) in ('Teleop devices', 'Policies'))
        pump()
        shot(pw.docks['library'], 'p910-library-panel-teleop-policies.png')

    for fn in (tolerance, experiments, optimise, requirements, export_dialog, canonical, assistant, plugins,
               catalogue, compare, review_additions):
        part(fn)
    pw.cancel_jobs()
    pw.hide()

    for name in ('aloha', 'panda-bimanual', 'g1'):
        def sample(name=name):
            sw = window(sp(name))
            shot(sw, f'p910-sample-{name}-main.png')
            shot(sw.viewport, f'p910-sample-{name}-3d.png')
            if name == 'g1':                     # a study page on the humanoid: its one head camera's coverage
                run_page(sw, 'coverage', 'sample-g1-coverage-page')
            sw.cancel_jobs()
            sw.hide()
        part(sample)


# ---------------------------------------------------------------- research tools part 2
def research2():
    """rt2-*: the Runs page with a comparison, Statistics with a power curve, the campaign plan, the sim-to-real gap
    with recorded against rendered frames, the paper kit preview, the cells comparison, the Run on… dialog and the USD
    import wizard."""
    import time

    from dexvantage.core.robot.samples import sample_project as sp

    tmp = tempfile.mkdtemp(prefix='dexvantage-screens-rt2-')       # no personal path in the pictures

    def wait_idle(win, timeout=600.0, page=None):
        t0 = time.time()
        pump(5)
        while time.time() - t0 < timeout:
            pump(5)
            busy = win.ctx.analysis.busy() or bool(getattr(win, '_jobs', None))
            if page is not None and hasattr(page, '_threads'):
                busy = busy or any(t.is_alive() for t in page._threads)
            if not busy:
                pump(20)
                return True
            time.sleep(0.03)
        return False

    def window(project):
        pw = MainWindow(project, soft_3d=True, settings=settings)
        pw.resize(1680, 1000)
        pw.show()
        pw.ctx.analysis.wait(5)
        pw.resizeDocks([pw.docks['inspector'], pw.docks['camview']], [380, 460], Qt.Orientation.Vertical)
        pw.resizeDocks([pw.docks['inspector']], [430], Qt.Orientation.Horizontal)
        pw.resizeDocks([pw.docks['bottom']], [560], Qt.Orientation.Vertical)
        pump()
        pw.viewport.frame_box(*pw.scene.home_box(pw.viewport))
        return pw

    def tall(pw, page, name, h=900, widget=None):
        pw.resizeDocks([pw.docks['bottom']], [h], Qt.Orientation.Vertical)
        pump(20)
        if widget is not None:
            page.body.parentWidget().parentWidget().ensureWidgetVisible(widget)
            pump(20)
        shot(page, name)
        pw.resizeDocks([pw.docks['bottom']], [560], Qt.Orientation.Vertical)
        pump()

    def part(fn):
        try:
            fn()
        except Exception:                        # noqa: BLE001 — one broken shot must not stop the others
            import traceback
            traceback.print_exc()

    def saved(name):
        p = sp(name)
        path = os.path.join(tmp, f'{name}.dexv')
        p.save(path)
        return P.Project.load(path)

    def runs():
        p = saved('so100')
        p.analysis.studies['tracking'] = {'enabled': True, 'quick': True, 'tags': ['screens']}
        pw = window(p)
        pw.open_page('runs')
        page = pw.pages['runs']
        st = pw.ctx.store
        pw.ctx.analysis.run('coverage')
        wait_idle(pw, page=page)
        uid = st.project.variant.sensors[0].uid
        st.edit('variant', lambda q: setattr(q.variant.sensor(uid).mount.pose, 'position',
                                             q.variant.sensor(uid).mount.pose.position + [0.0, 0.0, 0.03]),
                'Raise a camera 30 mm (screenshot)')
        wait_idle(pw, page=page)
        pw.ctx.analysis.run('coverage')
        wait_idle(pw, page=page)
        page.refresh()
        wait_idle(pw, page=page)
        from dexvantage.core import tracking as TR
        ids = [r['run_id'] for r in TR.store_for(st.project).runs(study='coverage')][-2:]
        page.select_ids(ids)
        page.compare_selected(ids)
        wait_idle(pw, page=page)
        shot(pw, 'rt2-so100-runs-page.png')
        tall(pw, page, 'rt2-so100-runs-compare.png')
        pw.cancel_jobs()
        pw.hide()

    def statistics():
        pw = window(sp('ur5e'))
        pw.open_page('statistics')
        page = pw.pages['statistics']
        page.set_rates([('ACT', 14, 20), ('Diffusion Policy', 19, 20), ('pi0', 17, 20)])
        wait_idle(pw)
        page.preview()
        t0 = time.time()
        while time.time() - t0 < 60 and not page.pw_answer.text():
            pump(5)
            time.sleep(0.03)
        pump(40)
        shot(pw, 'rt2-ur5e-statistics-page.png')
        tall(pw, page, 'rt2-ur5e-statistics-power.png', 980, page.pw_curve)
        pw.cancel_jobs()
        pw.hide()

    def campaign():
        import json
        spec = os.path.join(tmp, 'campaign.json')
        with open(spec, 'w', encoding='utf-8') as fh:
            json.dump({'schema': 'dexvantage.campaign_spec/1', 'name': 'ALOHA handover, two lighting presets',
                       'operators': ['Ana', 'Ben', 'Chen'], 'lighting': [{'name': 'Bright'}, {'name': 'Dim'}],
                       'demos': {'mode': 'power', 'p_base': 0.5, 'p_target': 0.7}, 'episode_s': 30,
                       'start_date': '2026-11-16', 'budget': {'disk_gb': 4000, 'days': 10, 'upload_mbps': 1000,
                                                              'upload_hours_per_day': 8}}, fh)
        pw = window(sp('aloha'))
        pw.open_page('campaign')
        page = pw.pages['campaign']
        page.open_spec(spec)
        wait_idle(pw)
        shot(pw, 'rt2-aloha-campaign-page.png')
        tall(pw, page, 'rt2-aloha-campaign-schedule.png', 980, page.sessions)
        pw.cancel_jobs()
        pw.hide()

    def sim2real():
        from dexvantage.core.sim2real import inject as IJ
        p = sp('ur5e')
        root = IJ.make_gap_dataset(p, os.path.join(tmp, 's2r'), {'brightness_dn': 20.0, 'blur_px': 1.5,
                                                                  'camera': 'observation.images.overhead_d435'},
                                   {'episodes': 1, 'frames': 40, 'size': [320, 180], 'seed': 4})['root']
        pw = window(p)
        pw.open_page('sim2real')
        page = pw.pages['sim2real']
        page.choose_dataset(root)
        wait_idle(pw)
        page.run()
        t0 = time.time()
        while time.time() - t0 < 10 and not pw.ctx.analysis.busy('sim2real'):
            pump(5)
        wait_idle(pw)
        ov = [i for i in range(page.cam_box.count()) if 'Overhead' in page.cam_box.itemText(i)]
        if ov:
            page.cam_box.setCurrentIndex(ov[0])
            page._show_camera()
        pump(40)
        shot(pw, 'rt2-ur5e-sim2real-page.png')
        tall(pw, page, 'rt2-ur5e-sim2real-frames.png', 980, page.causes)
        pw.cancel_jobs()
        pw.hide()

    def publication():
        pw = window(sp('so100'))
        for sid in ('rig', 'bandwidth', 'coverage'):
            pw.ctx.analysis.run(sid)
            pw.ctx.analysis.wait(120)
        pump(20)
        pw.open_page('publication')
        page = pw.pages['publication']
        page.run()
        t0 = time.time()
        while time.time() - t0 < 120 and page.kit is None:
            pump(5)
            time.sleep(0.03)
        pump(40)
        shot(pw, 'rt2-so100-publication-page.png')
        tall(pw, page, 'rt2-so100-publication-preview.png', 980, page.text)
        pw.cancel_jobs()
        pw.hide()

    def cells():
        pw = window(sp('so100'))
        pw.open_page('cells')
        page = pw.pages['cells']
        page.add_sample('ur5e')
        page.add_sample('aloha')
        wait_idle(pw)
        page.run()
        t0 = time.time()
        while time.time() - t0 < 10 and not pw.ctx.analysis.busy('cells'):
            pump(5)
        wait_idle(pw)
        shot(pw, 'rt2-so100-cells-page.png')
        tall(pw, page, 'rt2-so100-cells-tables.png', 980, page.gap_tab)
        pw.cancel_jobs()
        pw.hide()

    def run_on():
        from dexvantage.ui import cluster_run as CR
        pw = window(sp('so100'))
        pw.open_page('experiments')
        dlg = CR.RunOnDialog(pw, 'Designed experiments')
        dlg.resize(620, 520)
        pump(20)
        shot(dlg, 'rt2-run-on-dialog.png')
        dlg.slurm.setChecked(True)
        dlg.folder.setText(os.path.join('out', 'slurm-job'))
        pump(20)
        shot(dlg, 'rt2-run-on-dialog-slurm.png')
        dlg.close()
        pw.cancel_jobs()
        pw.hide()

    def usd_wizard():
        from dexvantage.ui.dialogs.import_wizard import ImportWizard
        pw = window(P.blank_project('USD import'))
        path = os.path.join(ROOT, 'dexvantage', 'resources', 'usd_robots', 'newton_ur5e', 'usd_structured',
                            'ur5e.usda')
        wz = ImportWizard(pw.ctx, pw)
        wz.resize(1100, 760)
        wz.show()
        wz.load(path, background=False)
        pump(60)
        shot(wz, 'rt2-usd-import-wizard.png')
        wz.close()
        pw.hide()

    for fn in (runs, statistics, campaign, sim2real, publication, cells, run_on, usd_wizard):
        part(fn)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


ONLY = {'phase3': phase3, 'phase458': phase458, 'samples': new_samples, 'phase910': phase910,
        'research2': research2}
if '--only' in sys.argv and sys.argv[sys.argv.index('--only') + 1] in ONLY:
    w.hide()
    ONLY[sys.argv[sys.argv.index('--only') + 1]]()
    QTimer.singleShot(0, app.quit)
    sys.exit(0)

w.viewport.frame_box(*w.scene.home_box(w.viewport))
shot(w, 'main-dark.png')
w.set_theme('light')
pump()
shot(w, 'main-light.png')
w.set_theme('dark')
from dexvantage.ui.dialogs.misc import FramesDialog, ShortcutsDialog  # noqa: E402
from dexvantage.ui.dialogs.palette import CommandPalette  # noqa: E402
for cls, name in ((FramesDialog, 'frames.png'), (ShortcutsDialog, 'shortcuts.png')):
    d = cls(w)
    d.show()
    shot(d, name)
    d.hide()
cp = CommandPalette(w.ctx, w, '')
cp.edit.setText('add')
cp.show()
shot(cp, 'palette.png')
cp.hide()
w.workspace('review')
shot(w, 'workspace-review.png')
w.hide()

# ---------------------------------------------------------------- Phase 1: the samples, the Robot page, the wizard
from dexvantage.core.robot.samples import ROBOTS_DIR, sample_project  # noqa: E402
from dexvantage.ui.dialogs.import_wizard import ImportWizard  # noqa: E402

SAMPLE_FILES = {'so100': os.path.join(ROBOTS_DIR, 'trs_so_arm100', 'so_arm100.xml'),
                'ur5e': os.path.join(ROBOTS_DIR, 'ur5e_robotiq_2f85.xml')}
for name, path in SAMPLE_FILES.items():
    sw = MainWindow(sample_project(name), soft_3d=True, settings=settings)
    sw.resize(1680, 1000)
    sw.show()
    sw.ctx.analysis.wait(5)
    pump()
    sw.viewport.frame_box(*sw.scene.home_box(sw.viewport))
    shot(sw, f'sample-{name}-main.png')
    sw.open_page('robot')
    sw.resizeDocks([sw.docks['bottom']], [520], Qt.Orientation.Vertical)     # room for the sliders
    pump()
    sw.viewport.frame_box(*sw.scene.home_box(sw.viewport))
    shot(sw, f'sample-{name}-robot-page.png')
    wiz = ImportWizard(sw.ctx, sw, soft_3d=True)
    wiz.show()
    wiz.load(path, background=False)
    for i in range(6):
        wiz.go(i)
        shot(wiz, f'sample-{name}-wizard-{i + 1}.png')
    wiz.hide()
    sw.hide()

# ---------------------------------------------------------------- Phase 2: sensors, coverage, depth, Camera view
from dexvantage.ui.dialogs.custom_device import CustomDeviceDialog  # noqa: E402


def wait_study(win, sid, timeout=120.0):
    import time
    t0 = time.time()
    while time.time() - t0 < timeout:
        pump(5)
        if sid in win.ctx.analysis.results and not win.ctx.analysis.busy():
            return True
        time.sleep(0.02)
    return False


def show_camview(win, name, tag):
    """Select the sensor named name, let the Camera view render at full size, save the window and the view."""
    s = next(s for s in win.ctx.store.project.variant.sensors if s.name == name)
    win.ctx.store.select(('sensor', s.uid))
    pump()
    win.camview.render_sync()
    win.camview.wait(20)
    pump(40)
    shot(win.docks['camview'], f'phase2-camview-{tag}.png')
    return s


for name in ('ur5e', 'so100'):
    pw = MainWindow(sample_project(name), soft_3d=True, settings=settings)
    pw.resize(1680, 1000)
    pw.show()
    pw.ctx.analysis.wait(5)
    pw.resizeDocks([pw.docks['inspector'], pw.docks['camview']], [380, 460], Qt.Orientation.Vertical)
    pw.resizeDocks([pw.docks['inspector']], [430], Qt.Orientation.Horizontal)
    pw.resizeDocks([pw.docks['bottom']], [430], Qt.Orientation.Vertical)
    pump()
    pw.viewport.frame_box(*pw.scene.home_box(pw.viewport))
    pw.open_page('sensors')
    first = pw.ctx.store.project.variant.sensors[0]
    pw.ctx.store.select(('sensor', first.uid))
    pump()
    shot(pw, f'phase2-{name}-sensors-page.png')
    for page, sid in (('coverage', 'coverage'), ('depth', 'depth_validity')):
        pw.open_page(page)
        pump()
        pg = pw.pages.get(page) if hasattr(pw, 'pages') else None
        if pg is not None and hasattr(pg, 'run'):
            pg.run()
        else:
            pw.ctx.analysis.run(sid)
        ok = wait_study(pw, sid)
        pump(40)
        pw.viewport.frame_box(*pw.scene.home_box(pw.viewport))
        pump(20)
        shot(pw, f'phase2-{name}-{page}-page.png' if ok else f'phase2-{name}-{page}-page-NOT-FINISHED.png')
    pw.open_page('coverage')
    for s in pw.ctx.store.project.variant.sensors:
        tag = f'{name}-' + ''.join(c if c.isalnum() else '-' for c in s.name.lower()).strip('-').replace('--', '-')
        show_camview(pw, s.name, tag)
    if name == 'ur5e':
        shot(pw, 'phase2-ur5e-main-camview.png')
        pw.resizeDocks([pw.docks['outliner'], pw.docks['library']], [200, 760], Qt.Orientation.Vertical)
        pw.resizeDocks([pw.docks['outliner']], [380], Qt.Orientation.Horizontal)
        tree = pw.library.tree
        for i in range(tree.topLevelItemCount()):
            it = tree.topLevelItem(i)
            it.setExpanded(it.text(0) in ('Depth sensors', 'Cameras'))
        pump()
        shot(pw.docks['library'], 'phase2-library-panel.png')
        d = CustomDeviceDialog(pw.ctx, pw, kind='depth', start_from='intel-realsense-d435')
        d.show()
        shot(d, 'phase2-custom-device-from-d435.png')
        d.hide()
        d = CustomDeviceDialog(pw.ctx, pw, kind='camera')
        d.show()
        shot(d, 'phase2-custom-device-blank.png')
        d.hide()
    pw.cancel_jobs()
    pw.hide()

# the Camera view's empty state in a squeezed dock (a blank project, no sensor)
ew = MainWindow(P.blank_project('Empty'), soft_3d=True, settings=settings)
ew.resize(1280, 800)
ew.show()
ew.resizeDocks([ew.docks['inspector'], ew.docks['camview']], [640, 110], Qt.Orientation.Vertical)
pump()
shot(ew.docks['camview'], 'phase2-camview-empty-squeezed.png')
ew.resizeDocks([ew.docks['inspector'], ew.docks['camview']], [420, 330], Qt.Orientation.Vertical)
pump()
shot(ew.docks['camview'], 'phase2-camview-empty.png')
ew.hide()

phase3()
phase458()

QTimer.singleShot(0, app.quit)
