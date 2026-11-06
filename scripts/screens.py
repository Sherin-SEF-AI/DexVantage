"""Offscreen screenshots of the Phase 0 window for the review gate: out/screens/*.png.

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

from PyQt6.QtCore import QTimer  # noqa: E402

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
QTimer.singleShot(0, app.quit)
