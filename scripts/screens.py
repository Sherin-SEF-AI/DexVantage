"""Offscreen screenshots for the review gates: out/screens/*.png. Phase 0: the window, dialogs and palette. Phase 1:
for the SO-100 and UR5e samples, the main window, the Robot page and the import wizard (each of its six steps).

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
QTimer.singleShot(0, app.quit)
