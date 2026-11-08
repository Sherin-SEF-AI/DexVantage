"""A plain pip install, as a new user has it after `pip install -e .` with no extras: the validation suite leaves out
the checks whose optional package is missing and says so in one sentence naming the extra, instead of failing them
(found by following the README in a new virtual environment: `--validate` failed 14 checks and exited 1 there, all
for want of pyarrow and av (the lerobot extra) or jsonschema (the test extra))."""
import importlib.util
import json

from dexvantage import validation as VA
from dexvantage.validation import suite as VS

LEROBOT_CHECKS = {'grasp.dataset_closures', 'groundtruth.synthetic_roundtrip', 'handeye.dataset_capture',
                  'handeye.silhouette_recovery', 'handeye.silhouette_vs_board', 'audit.fault_recall',
                  'audit.desync_offset', 'audit.plan_vs_recording', 'formats.lerobot_roundtrip',
                  'formats.lerobot_export', 'tracking.csv_parquet_identical', 'cells.pooling_lerobot',
                  'sim2real.injected_recovery'}


def _hide(monkeypatch, *mods):
    """Make importlib.util.find_spec say these top-level packages are not installed."""
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, 'find_spec',
                        lambda name, *a, **k: None if name.split('.')[0] in mods else real(name, *a, **k))


def test_checks_that_need_an_extra_are_left_out_and_listed(monkeypatch):
    VS.load_checks()
    # what this machine leaves out already (CI installs no PyTorch and has no CUDA device: the splat torch checks)
    here = {d['id'] for d in VS.not_run_specs()}
    assert not here & (LEROBOT_CHECKS | {'formats.schema'})
    _hide(monkeypatch, 'pyarrow')
    nr = {d['id']: d for d in VS.not_run_specs() if d['id'] not in here}
    assert set(nr) == LEROBOT_CHECKS
    assert {(d['module'], d['extra']) for d in nr.values()} == {('pyarrow', 'lerobot')}
    assert not LEROBOT_CHECKS & {s.id for s in VS.specs()}
    _hide(monkeypatch, 'av')                 # the other half of the extra: the same checks
    assert {d['id'] for d in VS.not_run_specs()} - here == LEROBOT_CHECKS
    monkeypatch.undo()
    _hide(monkeypatch, 'jsonschema')
    assert [(d['id'], d['extra']) for d in VS.not_run_specs() if d['id'] not in here] == [('formats.schema', 'test')]
    # the settings still select by model and cost
    assert VS.not_run_specs(VS.ValidationSettings(models=('stats',))) == []
    assert VS.not_run_specs(VS.ValidationSettings(cheap_only=True)) == [
        d for d in VS.not_run_specs() if VS.CHECKS[[s.id for s in VS.CHECKS].index(d['id'])].cost == 'cheap']


def test_not_run_sentence_names_the_extra():
    one = [{'id': 'formats.schema', 'module': 'jsonschema', 'extra': 'test'}]
    assert VS.not_run_sentence([]) == ''
    assert VS.not_run_sentence(one) == ('1 check not run because an optional package is not installed: 1 needs '
                                        'jsonschema (the test extra). Install it with pip install "dexvantage[test]" '
                                        'to run it.')
    two = one + [{'id': f'x{i}', 'module': 'pyarrow', 'extra': 'lerobot'} for i in range(2)]
    assert VS.not_run_sentence(two) == ('3 checks not run because an optional package is not installed: 1 needs '
                                        'jsonschema (the test extra); 2 need pyarrow (the lerobot extra). Install '
                                        'them with pip install "dexvantage[test,lerobot]" to run them.')


def test_batch_validate_passes_and_says_what_was_not_run(app, monkeypatch, tmp_path, capsys):
    """`dexvantage --validate DIR --models tracking --quick` without pyarrow: the other checks pass, exit 0, one
    sentence names the lerobot extra, and the report's JSON and HTML list the check left out."""
    from dexvantage.app import main
    _hide(monkeypatch, 'pyarrow')
    code = main(['--validate', str(tmp_path / 'v'), '--quick', '--models', 'tracking'])
    out = capsys.readouterr().out
    assert code == 0, out
    assert 'FAIL' not in out and 'tracking.csv_parquet_identical' not in out
    assert ('1 check not run because an optional package is not installed: 1 needs pyarrow (the lerobot extra). '
            'Install it with pip install "dexvantage[lerobot]" to run it.') in out
    doc = json.load(open(tmp_path / 'v' / 'validation.json', encoding='utf-8'))
    assert doc['summary']['ok'] and [d['id'] for d in doc['summary']['not_run']] == ['tracking.csv_parquet_identical']
    assert 'needs pyarrow (the lerobot extra)' in (tmp_path / 'v' / 'validation.html').read_text(encoding='utf-8')


def test_a_full_installation_reports_nothing_not_run(monkeypatch):
    """With every package a check needs installed, the result carries no not-run list (the report is unchanged)."""
    VS.load_checks()
    monkeypatch.setattr(VS, 'missing_need', lambda spec: None)
    assert VS.not_run_specs() == []
    res = VS.ValidationResult([], VS.ValidationSettings(), 'p', 'v', '', 0.0, {})
    assert res.not_run == [] and 'not_run' not in VA.summary(res) and 'not_run' not in VA.to_doc(res)['summary']


def test_window_says_when_opengl_is_missing(app, monkeypatch):
    """On a desktop without OpenGL (a VM, a remote X session) the window falls back to the QPainter 3D view, about a
    second a frame on the larger samples: the status bar says why the view is slow instead of leaving it unexplained.
    An explicit --software-3d, or an offscreen run, says nothing."""
    from PyQt6.QtGui import QGuiApplication

    from dexvantage import app as A
    from dexvantage.ui import mcp_bridge, viewport3d
    from dexvantage.ui.mainwindow import MainWindow
    shown = []
    monkeypatch.setattr(MainWindow, 'msg', lambda self, text, ms=3000: shown.append((self, text)))
    monkeypatch.setattr(viewport3d, 'gl_available', lambda: False)
    monkeypatch.setattr(mcp_bridge, 'start_bridge', lambda w: None)
    monkeypatch.setattr(type(app), 'exec', lambda self: 0)
    try:
        monkeypatch.setattr(QGuiApplication, 'platformName', staticmethod(lambda: 'xcb'))
        import signal
        before = signal.getsignal(signal.SIGINT)
        seen = []
        monkeypatch.setattr(type(app), 'exec', lambda self: seen.append(signal.getsignal(signal.SIGINT)) or 0)
        assert A.main(['--sample', 'so100']) == 0
        # Ctrl+C in the terminal closes the window while it runs; the handler before it is back afterwards
        assert seen and seen[0] is not before and signal.getsignal(signal.SIGINT) is before
        texts = [t for _, t in shown]
        assert any(t.startswith('OpenGL is not available here, so the 3D view is drawn by the slower QPainter') and
                   "Troubleshooting" in t for t in texts), texts
        shown.clear()
        assert A.main(['--sample', 'so100', '--software-3d']) == 0
        assert not any('OpenGL is not available' in t for _, t in shown)
    finally:
        monkeypatch.undo()
        import os
        os.environ.pop('DEXVANTAGE_SOFTWARE_3D', None)
        from conftest import _deferred_deletes, _settle, pump
        for w in [w for w in app.topLevelWidgets() if isinstance(w, MainWindow)]:
            w.cancel_jobs()
            _settle(app, w)                     # its studies and jobs end before it is deleted
            w.store.undo.setClean()
            w.hide()
            w.deleteLater()
        pump(app, 0.02)
        _deferred_deletes()


class _Tty:
    """A terminal stderr: what a user who typed the README's export command sees."""

    def __init__(self):
        self.text = ''

    def isatty(self):
        return True

    def write(self, s):
        self.text += s
        return len(s)

    def flush(self):
        pass


def test_batch_export_says_what_it_runs_on_a_terminal(monkeypatch, tmp_path, capsys):
    """The README's export command with the report format ran 89 s with nothing printed (the report runs every study):
    on a terminal each step is now named on stderr as it starts; in a pipe or a file (CI logs, the tests) nothing is
    added."""
    import sys

    from dexvantage import batch
    from dexvantage.core import exporters as X
    seen = []

    def fake(project, out_dir, formats, options=None, ctx=None):
        seen.append(ctx)
        if ctx is not None:
            for k, t in enumerate(('Coverage and resolution', 'Coverage and resolution', 'Depth validity')):
                ctx.report(k / 3, t)
        return {'formats': {'report': {'status': 'ok', 'files': ['report.html'], 'notes': []}}, 'files': {}}
    monkeypatch.setattr(X, 'export', fake)
    from dexvantage.core.robot.samples import sample_project
    args = type('A', (), {'export': str(tmp_path / 'x'), 'formats': 'report', 'export_options': None,
                          'dataset': None})()
    tty, err = _Tty(), sys.stderr
    sys.stderr = tty
    try:
        assert batch._export(args, sample_project('so100')) == 0
    finally:
        sys.stderr = err
    assert tty.text == '  Coverage and resolution (0 %)\n  Depth validity (67 %)\n'
    assert batch._export(args, sample_project('so100')) == 0
    assert seen[-1] is None and capsys.readouterr().err == ''
