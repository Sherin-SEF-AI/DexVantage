"""DexVantage entry point: the window, batch mode and the self-test.

Ported from Vantage app.py (the application object, fonts, theme and icon); the options are DexVantage's.
"""
from __future__ import annotations

import argparse
import atexit
import os
import shutil
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
        folder = tempfile.mkdtemp(prefix=f'dexvantage-{settings_scope}-')
        atexit.register(shutil.rmtree, folder, True)        # throw-away: removed when the process ends
        path = os.path.join(folder, 'settings.ini')
        settings = QSettings(path, QSettings.Format.IniFormat)
    else:
        settings = QSettings('DexVantage', 'DexVantage')
    THEME.apply(app, theme or settings.value('ui/theme', 'dark'), scale or float(settings.value('ui/scale', 1.0)))
    app.setWindowIcon(QIcon(APP_ICON) if os.path.exists(APP_ICON) else icons.icon('robot-arm', 16, THEME.tok['ink']))
    return app, settings


def main(argv=None):
    from . import __version__
    from .core.memlimit import limit_arenas
    limit_arenas()                  # before any worker thread: freed thread memory goes back to the system
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
    from .core.robot.samples import SAMPLES
    have = ', '.join(k for k, (_t, ph) in SAMPLES.items() if ph is None)
    later = ', '.join(k for k, (_t, ph) in SAMPLES.items() if ph is not None)
    ap.add_argument('--sample', metavar='NAME', help=f'open a sample project, in the window or in batch mode: {have}'
                                                     + (f' ({later} arrive in a later phase)' if later else ''))
    ap.add_argument('--no-plugins', action='store_true', help='start without loading the installed plugins (Help ▸ '
                    'Plugins lists them as off)')
    ap.add_argument('--list-plugins', action='store_true', help='list the installed plugins, what each added and any '
                    'error, then exit')
    from . import plugin_sdk                            # before the batch flags: a plugin study gets its flag too
    raw = list(sys.argv[1:] if argv is None else argv)
    plugin_sdk.load(enabled=False if '--no-plugins' in raw else None)
    from . import batch
    batch.add_arguments(ap)
    args = ap.parse_args(argv)
    args._parser = ap
    if args.list_plugins:
        lines = plugin_sdk.report_lines()
        print('\n'.join(lines) if lines else 'No DexVantage plugins are installed.')
        return 0
    if args.self_test:
        from . import selftest
        return selftest.run()
    if args.path and os.path.splitext(args.path)[1].lower() in ('.urdf', '.xacro', '.xml', '.usd', '.usda', '.usdc'):
        print(f'dexvantage: {os.path.basename(args.path)} is a robot model file, not a project (.dexv). Start '
              'DexVantage and import it with Robot ▸ Import robot…, or open a sample with --sample so100.',
              file=sys.stderr)
        return 2
    if args.sample:
        if args.path:
            print('dexvantage: give a project file or --sample, not both', file=sys.stderr)
            return 2
        from .core.project import ProjectFormatError
        from .core.robot.samples import sample_project
        try:
            args._project = sample_project(args.sample)        # batch._load and the window use it
        except ProjectFormatError as e:
            print(f'dexvantage: {e}', file=sys.stderr)
            return 2
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
    project = getattr(args, '_project', None)
    if args.path:
        try:
            project = Project.load(args.path)
        except ProjectFormatError as e:
            print(f'dexvantage: {e}', file=sys.stderr)
            return 2
    w = MainWindow(project or blank_project(), soft_3d=not gl_available(), settings=settings)
    w.show()
    from .ui.mcp_bridge import start_bridge          # Claude over MCP reaches this window (setting mcp/bridge)
    bridge = start_bridge(w)
    if bridge is not None:
        app.aboutToQuit.connect(bridge.stop)
    if project is not None and project.path:
        w.add_recent(project.path)
    if hasattr(app, 'file_opened'):
        app.file_opened.connect(lambda p: w.open_project(p))
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
