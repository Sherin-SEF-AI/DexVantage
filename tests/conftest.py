"""One QApplication for the whole session, offscreen, with throw-away config, data and cache folders: a test never
reads or writes the user's real settings, keyring or keys. Ported from Vantage tests/conftest.py."""
import os
import sys
import tempfile
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
for _k in ('XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_CACHE_HOME'):
    os.environ[_k] = tempfile.mkdtemp(prefix=f'dexvantage-test-{_k[4:8].lower()}-')
os.environ['DEXVANTAGE_NO_KEYRING'] = '1'
for _k in ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL', 'GROQ_API_KEY'):
    os.environ.pop(_k, None)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import pytest  # noqa: E402


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
    path = os.path.join(tempfile.mkdtemp(prefix='dexvantage-test-settings-'), 'settings.ini')
    return QSettings(path, QSettings.Format.IniFormat)


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
    w.store.undo.setClean()
    w.hide()
    w.deleteLater()
    pump(app, 0.02)
