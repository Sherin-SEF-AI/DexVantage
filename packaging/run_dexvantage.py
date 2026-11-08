"""The downloadable app's entry script (packaging/dexvantage.spec builds both launchers from it).

Two things a frozen DexVantage needs before dexvantage.app.main():

1. multiprocessing.freeze_support(). The cluster's local pool (core/cluster/local.py) and the designed experiments'
   pool (core/experiments.py) use the spawn start method: in a bundle a worker is the bundle's own executable started
   with --multiprocessing-fork (and the resource tracker with -c "from multiprocessing.resource_tracker import
   main..."). PyInstaller's multiprocessing runtime hook makes freeze_support() recognise both; it must run first,
   before the argument parser sees those arguments.

2. "-m module [args]". DexVantage starts some of its own modules as worker processes with
   [sys.executable, '-m', 'dexvantage.…']: the MuJoCo render worker (validation/mjrender.py, MUJOCO_GL=egl), the
   occlusion GPU worker (core/occlusion.py --gl-worker), a cluster shard (dexvantage --run-shard) and the MCP batch
   runner. In a bundle sys.executable is this program, not Python, so it runs the module the same way python -m would.
   Only modules of the dexvantage package are run; anything else is refused in one sentence.

DEXVANTAGE_STARTUP_PROBE=1 (packaging/build.py's start-up measurement): once the event loop runs with the main window
shown, print "startup-probe <seconds since this script started>" (also into the file DEXVANTAGE_STARTUP_PROBE_OUT
names, if set, for a launcher started with its standard output closed) and quit. The window then uses throw-away
settings (make_app's settings_scope), as the self-test does, so the measurement never reads or writes the settings of
whoever runs the build, even without the smoke test's XDG folders.
"""
import multiprocessing
import os
import runpy
import sys
import time

_T0 = time.perf_counter()


def _run_module(argv):
    mod = argv[2] if len(argv) > 2 else ''
    if mod != 'dexvantage' and not mod.startswith('dexvantage.'):
        print(f'DexVantage: -m runs DexVantage\'s own modules only (dexvantage.…), not {mod or "nothing"}.',
              file=sys.stderr)
        return 2
    sys.argv = [argv[0]] + argv[3:]
    runpy.run_module(mod, run_name='__main__', alter_sys=True)     # its own sys.exit() ends the process as usual
    return 0


def _startup_probe():
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication, QMainWindow

    def check():
        app = QApplication.instance()
        if app is None:
            return
        if any(isinstance(w, QMainWindow) and w.isVisible() for w in app.topLevelWidgets()):
            line = f'startup-probe {time.perf_counter() - _T0:.3f}'
            if sys.stdout is not None:
                print(line, flush=True)
            out = os.environ.get('DEXVANTAGE_STARTUP_PROBE_OUT')
            if out:                         # a launcher started with its stdout closed
                with open(out, 'w', encoding='utf-8') as fh:
                    fh.write(line + '\n')
            app.quit()
        else:
            QTimer.singleShot(10, check)

    from dexvantage import app as A
    make_app = A.make_app

    def wrapped(*a, **kw):
        if len(a) < 3:                      # make_app(theme, scale, settings_scope)
            kw.setdefault('settings_scope', 'startup-probe')
        out = make_app(*a, **kw)
        QTimer.singleShot(0, check)
        return out
    A.make_app = wrapped


def main():
    multiprocessing.freeze_support()
    if len(sys.argv) > 1 and sys.argv[1] == '-m':
        return _run_module(sys.argv)
    if os.environ.get('DEXVANTAGE_STARTUP_PROBE') == '1':
        _startup_probe()
    from dexvantage.app import main as app_main
    return app_main()


if __name__ == '__main__':
    sys.exit(main())
