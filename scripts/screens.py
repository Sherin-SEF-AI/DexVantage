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

    QT_QPA_PLATFORM=offscreen python scripts/screens.py [--only phase3 | --only phase458]
"""
import os
import sys
import tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
for k in ('XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_CACHE_HOME'):
    os.environ[k] = tempfile.mkdtemp(prefix='dexvantage-screens-')

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
        tmp = tempfile.mkdtemp(prefix='dexvantage-screens-capture-')
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


if '--only' in sys.argv and sys.argv[sys.argv.index('--only') + 1] in ('phase3', 'phase458'):
    w.hide()
    (phase3 if sys.argv[sys.argv.index('--only') + 1] == 'phase3' else phase458)()
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
