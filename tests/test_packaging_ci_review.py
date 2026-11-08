"""The packaging and CI review of Phase 11: what it found and fixed, each with a regression test.

- The cluster's peak-memory figure imported the POSIX-only `resource` module at the top of core/cluster/shards.py, so
  on Windows every spawned pool worker (--jobs N, the bundle smoke test's --tolerance-mc --jobs 2) stopped with a
  ModuleNotFoundError; Windows now reads the peak working set instead.
- The bundle's start-up measurement (packaging/run_dexvantage.py DEXVANTAGE_STARTUP_PROBE) opened the window with the
  user's own settings: the smoke test's XDG folders redirect QSettings on Linux only, not the Windows registry or the
  macOS preferences. It also printed only to stdout, which the console-less Windows launcher may not have.
- build.py's home-folder scan matched the native spelling only (on Windows not the forward-slash one, nor JSON's
  doubled backslashes).
- Runner labels are versions, never -latest; the draft release of a release candidate is marked a pre-release.
No network; nothing is built.
"""
import importlib
import importlib.util
import json
import math
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, 'packaging')
WF = os.path.join(ROOT, '.github', 'workflows')


def _packaging_module(name):
    path = os.path.join(PKG, f'{name}.py')
    sys.path.insert(0, PKG)
    try:
        spec = importlib.util.spec_from_file_location(f'_dexvantage_pkgci_{name}', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(PKG)
    return mod


def _workflow(name):
    with open(os.path.join(WF, name), encoding='utf-8') as fh:
        return yaml.safe_load(fh)


# ------------------------------------------------------------------------------------------------ Windows: resource
def test_cluster_shards_imports_without_the_resource_module(monkeypatch):
    """Windows has no `resource` module: shards.py imports without it and peak_rss_mb still answers."""
    from dexvantage.core.cluster import shards
    monkeypatch.setitem(sys.modules, 'resource', None)          # `import resource` now raises ImportError
    mod = importlib.reload(shards)
    try:
        monkeypatch.setattr(sys, 'platform', 'win32')
        v = mod.peak_rss_mb()                                    # no WinDLL here: NaN, not an exception
        assert isinstance(v, float) and (math.isnan(v) or v > 0)
    finally:
        monkeypatch.undo()
        importlib.reload(shards)
    assert shards.peak_rss_mb() > 0                              # this platform's own figure, as before


def test_no_module_of_the_app_imports_a_posix_only_module_at_its_top():
    """A top-level `import resource` (or fcntl, termios, pwd, grp) breaks the module on Windows; such imports must
    be inside the function that needs them or guarded by try/except ImportError."""
    import ast
    posix = {'resource', 'fcntl', 'termios', 'pwd', 'grp', 'posix'}
    bad = []
    for d, subdirs, files in os.walk(os.path.join(ROOT, 'dexvantage')):
        subdirs[:] = [s for s in subdirs if s != '__pycache__']
        for f in files:
            if not f.endswith('.py'):
                continue
            path = os.path.join(d, f)
            with open(path, encoding='utf-8') as fh:
                tree = ast.parse(fh.read())
            for node in tree.body:                               # module level, outside any try
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name.split('.')[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    names = [node.module.split('.')[0]]
                bad += [f'{os.path.relpath(path, ROOT)}:{node.lineno} {n}' for n in names if n in posix]
    assert not bad, bad


# ------------------------------------------------------------------------------------------------ the start-up probe
def test_startup_probe_uses_throw_away_settings_and_can_write_a_file(app, monkeypatch, tmp_path):
    launcher = _packaging_module('run_dexvantage')
    from dexvantage import app as A
    seen = []

    def fake_make_app(theme=None, scale=None, settings_scope=None):
        seen.append(settings_scope)
        return app, None
    monkeypatch.setattr(A, 'make_app', fake_make_app)
    launcher._startup_probe()
    A.make_app('dark', 1.0)                                      # as app.main calls it
    assert seen == ['startup-probe']
    # the probe line also goes to DEXVANTAGE_STARTUP_PROBE_OUT (the console-less Windows launcher)
    from PyQt6.QtWidgets import QMainWindow
    out = tmp_path / 'probe.txt'
    monkeypatch.setenv('DEXVANTAGE_STARTUP_PROBE_OUT', str(out))
    quits = []
    monkeypatch.setattr(app, 'quit', lambda: quits.append(1))
    monkeypatch.setattr(sys, 'stdout', None)                     # pythonw-like: print must not be needed
    w = QMainWindow()
    w.show()
    try:
        for _ in range(200):
            app.processEvents()
            if quits:
                break
    finally:
        w.close()
    assert quits and out.read_text(encoding='utf-8').startswith('startup-probe ')


# ------------------------------------------------------------------------------------------------ build.py
def test_home_folder_scan_finds_every_spelling(tmp_path):
    build = _packaging_module('build')
    home = tmp_path / 'home'
    home.mkdir()
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'clean.json').write_text('{"a": 1}', encoding='utf-8')
    (bundle / 'native.txt').write_text(f'path={home}{os.sep}x', encoding='utf-8')
    (bundle / 'escaped.json').write_text(json.dumps({'p': str(home) + os.sep + 'y'}), encoding='utf-8')
    (bundle / 'forward.cfg').write_text(str(home).replace('\\', '/') + '/z', encoding='utf-8')
    (bundle / 'binary.so').write_bytes(f'{home}{os.sep}'.encode())        # not a text file: not scanned
    hits = sorted(build.personal_paths(str(bundle), home=str(home)))
    assert hits == ['escaped.json', 'forward.cfg', 'native.txt']


def test_smoke_environment_redirects_every_temporary_folder(tmp_path):
    build = _packaging_module('build')
    env = build.smoke_env(str(tmp_path))
    for k in ('TMPDIR', 'TMP', 'TEMP'):                          # TMP and TEMP are Windows' names
        assert env[k] == str(tmp_path)
    for k in ('XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_CACHE_HOME'):
        assert env[k].startswith(str(tmp_path)) and os.path.isdir(env[k])


# ------------------------------------------------------------------------------------------------ workflows
def _runner_labels():
    for name in sorted(os.listdir(WF)):
        if not name.endswith(('.yml', '.yaml')):
            continue
        doc = _workflow(name)
        for jid, job in doc['jobs'].items():
            ro = job.get('runs-on')
            labels = [ro] if isinstance(ro, str) and '${{' not in ro else []
            matrix = (job.get('strategy') or {}).get('matrix') or {}
            labels += [t['os'] for t in matrix.get('target', []) if isinstance(t, dict)] + list(matrix.get('os', []))
            yield name, jid, labels


def test_every_runner_is_named_by_its_version():
    """-latest labels move to a new image over one to two months (runner-images README), so a pinned matrix that
    passed can fail on its own; every runner here is named by its version."""
    seen = 0
    for name, jid, labels in _runner_labels():
        for lb in labels:
            seen += 1
            assert not lb.endswith('-latest'), f'{name} {jid}: {lb}'
    assert seen >= 10


def test_draft_release_marks_a_pre_release_and_finds_the_bundle_cli_with_forward_slashes():
    b = _workflow('bundles.yml')
    rel = ' '.join(s.get('run', '') for s in b['jobs']['draft-release']['steps'])
    assert 'release.is_prerelease' in rel and '--prerelease' in rel and '--draft' in rel
    release = _packaging_module('release')
    assert release.is_prerelease('0.1.0rc1') and not release.is_prerelease('0.1.0')
    own = ' '.join(s.get('run', '') for s in b['jobs']['bundle']['steps'])
    assert "replace(chr(92), '/')" in own and 'mktemp -d' not in own


def test_ci_lints_every_python_folder_and_runs_tests_in_utf8_mode():
    ci = _workflow('ci.yml')
    assert ci['env']['PYTHONUTF8'] == '1'
    lint = ' '.join(s.get('run', '') for s in ci['jobs']['lint']['steps'])
    assert 'ruff check dexvantage tests scripts examples packaging' in lint


def test_bundle_requirements_name_the_bundles_workflow():
    with open(os.path.join(PKG, 'requirements-bundle.txt'), encoding='utf-8') as fh:
        text = fh.read()
    assert '.github/workflows/bundles.yml' in text and 'build.yml' not in text.replace('bundles.yml', '')
    assert os.path.isfile(os.path.join(WF, 'bundles.yml'))


# ------------------------------------------------------------------------------------------------ the self-test
def test_self_test_warns_not_fails_when_an_optional_extra_is_missing(monkeypatch):
    """`pip install dexvantage` has no extras, and README's first command after it is --self-test (also CI's oldest
    Python job): a check that needs pyarrow or jsonschema, when they are not installed, is a warning naming the extra.
    Where the package is installed the same text stays a failure, so a real fault is never hidden."""
    from dexvantage import selftest as ST
    from dexvantage.validation import suite as VS
    pyarrow_msg = ('DatasetError: Reading or writing LeRobot datasets needs the Python package pyarrow (install the '
                   'lerobot extra: pip install "dexvantage[lerobot]").')
    monkeypatch.setitem(sys.modules, 'pyarrow', None)           # not installed, as far as find_spec can tell
    monkeypatch.setitem(sys.modules, 'jsonschema', None)
    assert ST._optional_missing(pyarrow_msg) == 'pyarrow'
    assert ST._optional_missing("ModuleNotFoundError: No module named 'jsonschema'") == 'jsonschema'
    assert ST._optional_missing('pyarrow is not installed: Parquet cannot be checked.') == 'pyarrow'
    assert ST._optional_missing('error 3 px against 1 px') is None
    status, msg = ST._not_checked(['pyarrow', 'jsonschema'])
    assert status == ST.WARN and 'pip install "dexvantage[lerobot,test]"' in msg

    def fake(ids):
        return [VS.Check(id=i, model='formats', title='t', quantity='q', reference='r', refs=(), unit='',
                         tolerance=0.0, error=float('nan'), passed=False, detail={}, figure=None, note='',
                         seconds=0.0, exception=pyarrow_msg) for i in ids]
    monkeypatch.setattr(ST, '_run_checks', fake)
    assert ST._checks_line(['a', 'b'], lambda r: 'fine')[0] == ST.WARN
    monkeypatch.delitem(sys.modules, 'pyarrow')                 # installed after all: the same failure is a failure
    if importlib.util.find_spec('pyarrow') is not None:
        assert ST._optional_missing(pyarrow_msg) is None
        assert ST._checks_line(['a'], lambda r: 'fine')[0] == ST.FAIL
