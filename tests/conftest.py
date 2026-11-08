"""One QApplication for the whole session, offscreen, with throw-away config, data and cache folders: a test never
reads or writes the user's real settings, keyring or keys. Ported from Vantage tests/conftest.py."""
import atexit
import os
import shutil
import sys
import tempfile
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
for _k in ('XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_CACHE_HOME'):
    os.environ[_k] = tempfile.mkdtemp(prefix=f'dexvantage-test-{_k[4:8].lower()}-')
    atexit.register(shutil.rmtree, os.environ[_k], True)    # thrown away when the session ends
os.environ['DEXVANTAGE_NO_KEYRING'] = '1'
for _k in ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL', 'GROQ_API_KEY'):
    os.environ.pop(_k, None)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dexvantage.core.memlimit import limit_arenas  # noqa: E402

limit_arenas()          # two malloc arenas: freed worker-thread memory goes back (seven modules: 1.64 GB -> 0.84 GB)

import pytest  # noqa: E402

# Run first: the quick validation run of every check now runs one model per child process, two at a time (up to about
# 6 GB together while they run, in the suite's cgroup); first, the suite's own process is still at about 0.3 GB.
RUN_FIRST = ('tests/test_validation.py::test_every_check_passes_and_names_its_reference',)


def pytest_collection_modifyitems(session, config, items):
    first = [it for it in items if it.nodeid in RUN_FIRST]
    if first:
        items[:] = first + [it for it in items if it.nodeid not in RUN_FIRST]


def pytest_configure(config):
    config.addinivalue_line('markers', 'mujoco_render: needs MuJoCo\'s offscreen renderer (EGL); skipped with the '
                                       'reason where there is none (tests/render_support.py)')


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """A test marked mujoco_render is skipped (or, with DEXVANTAGE_TEST_RENDER=require, failed) with the reason when
    MuJoCo cannot render offscreen here, before its fixtures render anything."""
    if item.get_closest_marker('mujoco_render') is not None:
        from render_support import require
        require()


@pytest.fixture(scope='session', autouse=True)
def app():
    from dexvantage.app import make_app
    a, _ = make_app('dark', settings_scope='tests')
    yield a


def pump(app, seconds=0.05):
    t = time.time()
    while time.time() - t < seconds:
        app.processEvents()
        time.sleep(0.005)


@pytest.fixture
def settings():
    from PyQt6.QtCore import QSettings
    folder = tempfile.mkdtemp(prefix='dexvantage-test-settings-')
    yield QSettings(os.path.join(folder, 'settings.ini'), QSettings.Format.IniFormat)
    shutil.rmtree(folder, ignore_errors=True)


@pytest.fixture
def win(app, settings):
    from dexvantage.core.project import blank_project
    from dexvantage.ui.mainwindow import MainWindow
    w = MainWindow(blank_project('Test cell'), soft_3d=True, settings=settings)
    w.resize(1400, 900)
    w.show()
    w.ctx.analysis.wait(5)
    pump(app, 0.05)
    yield w
    w.cancel_jobs()
    _settle(app, w)
    w.store.undo.setClean()
    w.hide()
    w.deleteLater()
    pump(app, 0.02)
    _deferred_deletes()


def _settle(app, w, timeout=60.0):
    """Let the window's background work end (studies, window jobs, Camera view renders), so no queued result
    reaches it after it is deleted."""
    w.ctx.analysis.wait(timeout)
    t0 = time.time()
    while getattr(w, '_jobs', None) and time.time() - t0 < timeout:
        pump(app, 0.05)
    cv = getattr(w, 'camview', None)
    if cv is not None and hasattr(cv, 'wait'):
        cv.wait(20)
    pump(app, 0.05)


def _deferred_deletes():
    """Run the deleteLater() calls now: processEvents() does not, so every window a test closed stayed in memory
    with its project, robot models and caches until the session ended."""
    from PyQt6.QtCore import QCoreApplication, QEvent
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


@pytest.fixture(scope='module', autouse=True)
def _free_caches_between_modules():
    """Each test module opens its own samples (each with new uids, so new cached robot models of some 35 MB): drop
    the shared caches after every module so a full run stays within its memory cap."""
    yield
    import gc
    from dexvantage.core import grasps
    from dexvantage.core.calib import synthetic
    from dexvantage.core.robot import model
    model.clear_cache()
    grasps._PAD_CACHE.clear()
    synthetic._RAYS.clear()
    from dexvantage.validation.suite import release_memory      # cube maps, lens remap plans, the audit campaign
    release_memory()
    _deferred_deletes()
    gc.collect()
    if sys.platform.startswith('linux'):            # hand freed heap pages back to the system (glibc keeps them)
        import ctypes
        try:
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except (OSError, AttributeError):
            pass
