"""Offscreen screenshots for the review gates: out/screens/*.png. Phase 0: the window, dialogs and palette. Phase 1:
for the SO-100 and UR5e samples, the main window, the Robot page and the import wizard (each of its six steps).
Phase 2 (phase2-*): the Sensors and mounts page, the Coverage and Depth validity pages after a run, the Camera view
of the UR5e sample's Wrist D405 and Overhead D435 (and the SO-100's two cameras), the Library panel and the New
custom device dialog. Each Camera view image is also saved full size on its own (phase2-camview-*.png).

    QT_QPA_PLATFORM=offscreen python scripts/screens.py
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

QTimer.singleShot(0, app.quit)
