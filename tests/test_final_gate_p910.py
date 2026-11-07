"""The Phases 9 and 10 final gate's fixes.

- Throw-away settings folders (make_app(settings_scope=...)) and the test session's XDG and settings folders were never
  removed: two days of test and self-test runs left about 12 000 folders (4 GB) in the system temp folder, which ran
  out of space during a review's full suite ("Disk quota exceeded" from a MuJoCo render).
- The assistant's export tools refused the 'usd' alias that core/exporters.export and --formats accept.
- A ticked checkbox that is disabled (the Requirements page's read-only "DexVantage default") drew no tick: the
  theme's :disabled rule cleared the background the inverted tick icon needs.
"""
import json
import os
import subprocess
import sys

from dexvantage.assistant import tools as T
from dexvantage.assistant.session import FileSession
from dexvantage.core import exporters as X
from dexvantage.core.project import blank_project

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def test_settings_scope_folder_is_removed_at_exit(tmp_path):
    code = ('import os, sys\n'
            'from dexvantage.app import make_app\n'
            'app, s = make_app(settings_scope="gate-check")\n'
            's.setValue("x", 1); s.sync()\n'
            'print(os.path.dirname(s.fileName()))\n')
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen', TMPDIR=str(tmp_path))
    r = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    folder = r.stdout.strip().splitlines()[-1]
    assert os.path.basename(folder).startswith('dexvantage-gate-check-')
    assert not os.path.exists(folder)
    assert not [n for n in os.listdir(tmp_path) if n.startswith('dexvantage-gate-check-')]


def test_settings_fixture_folder_is_removed(request):
    s = request.getfixturevalue('settings')
    s.setValue('x', 1)
    s.sync()
    folder = os.path.dirname(s.fileName())
    assert os.path.isdir(folder)
    # the fixture removes its folder at teardown; checked by the next test through the recorded path
    test_settings_fixture_folder_is_removed.folder = folder


def test_settings_fixture_folder_is_gone_after_its_test():
    folder = getattr(test_settings_fixture_folder_is_removed, 'folder', None)
    assert folder and not os.path.exists(folder)


def test_assistant_export_takes_the_usd_alias(tmp_path):
    @X.exporter('gate_json', 'A test format')
    def _(project, out_dir, options, ctx):
        with open(os.path.join(out_dir, 'x.json'), 'w', encoding='utf-8') as fh:
            json.dump({'name': project.name}, fh)
        return {'files': ['x.json'], 'notes': []}
    X.ALIASES['gate-alias'] = 'gate_json'
    try:
        s = FileSession(project=blank_project('gate'))
        out = T.call(s, 'export_files', {'formats': ['gate-alias'], 'folder': str(tmp_path / 'out')})
        assert 'error' not in out, out
        assert out['formats']['gate_json']['status'] == 'ok'
        assert 'usd' in X.ALIASES and X.resolve(['usd', 'isaac']) == ['isaac']
    finally:
        X.EXPORTERS.pop('gate_json', None)
        X.ALIASES.pop('gate-alias', None)


def test_a_disabled_ticked_checkbox_shows_its_tick(app):
    from PyQt6.QtWidgets import QCheckBox
    from dexvantage.ui.theme import THEME

    def indicator(checked):
        c = QCheckBox('x')
        c.setChecked(checked)
        c.setEnabled(False)
        c.resize(60, 20)
        img = c.grab().toImage()
        return {img.pixel(x, y) for x in range(0, 16) for y in range(0, 20)}

    for theme in ('dark', 'light'):
        THEME.apply(app, theme, 1.0)
        assert len(indicator(True) - indicator(False)) > 3, theme     # the tick and its fill differ from an empty box
    THEME.apply(app, 'dark', 1.0)
