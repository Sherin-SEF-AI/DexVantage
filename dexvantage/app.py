"""DexVantage entry point: the window, batch mode and the self-test.

Ported from Vantage app.py (the application object, fonts, theme and icon); the options are DexVantage's.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

APP_ICON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources', 'icons', 'dexvantage.png')


def make_app(theme=None, scale=None, settings_scope=None):
    """The QApplication with DexVantage's fonts, theme and icon; returns (app, settings). settings_scope: a name for
    throw-away settings in a temporary folder (the self-test), never the user's own."""
    from PyQt6.QtCore import QEvent, QSettings, Qt, pyqtSignal
    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QApplication
    from .ui.viewport3d import configure_default_format

    class _App(QApplication):
        """Keeps macOS's open-document events (a .dexv double-clicked in Finder arrives as a QFileOpenEvent)."""
        file_opened = pyqtSignal(str)

        def __init__(self, argv):
            super().__init__(argv)
            self.pending = []

        def event(self, e):
            if e.type() == QEvent.Type.FileOpen and e.file():
                self.pending.append(e.file())
                self.file_opened.emit(e.file())
                return True
            return super().event(e)

    app = QApplication.instance()
    if app is None:
        configure_default_format()
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        app = _App(sys.argv[:1])
    app.setApplicationName('DexVantage')
    app.setOrganizationName('DexVantage')
    app.setApplicationDisplayName('DexVantage')
    app.setDesktopFileName('dexvantage')
    from .ui import icons
    from .ui.theme import THEME, load_fonts
    load_fonts()
    if settings_scope:
        path = os.path.join(tempfile.mkdtemp(prefix=f'dexvantage-{settings_scope}-'), 'settings.ini')
        settings = QSettings(path, QSettings.Format.IniFormat)
    else:
        settings = QSettings('DexVantage', 'DexVantage')
    THEME.apply(app, theme or settings.value('ui/theme', 'dark'), scale or float(settings.value('ui/scale', 1.0)))
    app.setWindowIcon(QIcon(APP_ICON) if os.path.exists(APP_ICON) else icons.icon('robot-arm', 16, THEME.tok['ink']))
    return app, settings


def main(argv=None):
    from . import __version__
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors='replace')
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog='dexvantage', description='DexVantage · camera, depth and teleop planning for '
                                                                'robot-learning data collection')
    ap.add_argument('--version', action='version', version=f'dexvantage {__version__}')
    ap.add_argument('--self-test', action='store_true',
                    help='check this installation (project file, frames, MuJoCo, OpenCV, fonts, OpenGL, main window, '
                         'validation) without a window')
    ap.add_argument('path', nargs='?', help='a .dexv project')
    ap.add_argument('--theme', choices=['dark', 'light', 'system'])
    ap.add_argument('--scale', type=float, help='UI scale, e.g. 1.15')
    ap.add_argument('--software-3d', action='store_true', help='draw the 3D view with QPainter instead of OpenGL')
    from . import batch
    batch.add_arguments(ap)
    args = ap.parse_args(argv)
    args._parser = ap
    if args.self_test:
        from . import selftest
        return selftest.run()
    if batch.wanted(args):
        return batch.run(args)
    if args.software_3d:
        os.environ['DEXVANTAGE_SOFTWARE_3D'] = '1'
    app, settings = make_app(args.theme, args.scale)
    app.processEvents()
    if not args.path and getattr(app, 'pending', None):
        args.path = app.pending.pop()
    from .core.project import Project, ProjectFormatError, blank_project
    from .ui.mainwindow import MainWindow
    from .ui.viewport3d import gl_available
    project = None
    if args.path:
        try:
            project = Project.load(args.path)
        except ProjectFormatError as e:
            print(f'dexvantage: {e}', file=sys.stderr)
            return 2
    w = MainWindow(project or blank_project(), soft_3d=not gl_available(), settings=settings)
    w.show()
    if project is not None and project.path:
        w.add_recent(project.path)
    if hasattr(app, 'file_opened'):
        app.file_opened.connect(lambda p: w.open_project(p))
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
