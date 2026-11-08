"""Packaging (spec §15): the PyInstaller spec parses and collects what the app needs, the resource lists cover every
resources/ subfolder, the launcher's freeze support and -m dispatch, the optional extras' bundling decision and the
sentence the downloadable app adds, release.py (checksums, notes, the gh command only printed), the Linux installer,
the Windows installer script, the macOS Info.plist and the icons. Ported from Vantage tests/test_packaging.py.

Building the bundle itself is not a test (minutes, gigabytes): packaging/build.py --smoke does it, and CI runs it on
every platform (.github/workflows/bundles.yml)."""
import ast
import hashlib
import importlib.util
import os
import plistlib
import re
import shutil
import struct
import subprocess
import sys
import tomllib

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, 'packaging')


def _packaging_module(name):
    """packaging/<name>.py as a module (packaging/ is not a package)."""
    path = os.path.join(PKG, f'{name}.py')
    sys.path.insert(0, PKG)
    try:
        spec = importlib.util.spec_from_file_location(f'_dexvantage_packaging_{name}', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(PKG)
    return mod


def _text(*parts):
    with open(os.path.join(ROOT, *parts), encoding='utf-8') as fh:
        return fh.read()


def _pyproject():
    with open(os.path.join(ROOT, 'pyproject.toml'), 'rb') as fh:
        return tomllib.load(fh)


def _req_name(line):
    return re.split(r'[;<>=!~\[ ]', line.split('#')[0].strip(), maxsplit=1)[0].lower()


def _env(tmp_path):
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen', PYTHONPATH=ROOT)
    for k in ('XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_CACHE_HOME'):
        env[k] = str(tmp_path / k.lower())
    return env


# ------------------------------------------------------------------------------------------------ the spec
def test_spec_parses_and_names_what_a_frozen_app_needs():
    """The spec is valid Python and carries the decisions docs/packaging.md describes (PyInstaller is not needed to
    read it)."""
    spec = _text('packaging', 'dexvantage.spec')
    tree = ast.parse(spec, 'dexvantage.spec')
    calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert {'Analysis', 'PYZ', 'EXE', 'COLLECT', 'BUNDLE'} <= calls
    # two launchers sharing one folder, both in Python's UTF-8 mode
    assert "name='DexVantage', console=False" in spec and "name='dexvantage-cli', console=True" in spec
    assert "('X utf8', None, 'OPTION')" in spec and spec.count('options, name=') == 2
    # the entry script with freeze support, every module of the package (and its .py files for the folder scans and
    # the provenance code digest), every data file
    assert "os.path.join(HERE, 'run_dexvantage.py')" in spec
    assert "collect_submodules('dexvantage')" in spec and "module_collection_mode={'dexvantage': 'pyz+py'}" in spec
    assert 'datas = release.data_files(ROOT)' in spec and 'datas=datas' in spec
    # SciPy's Sobol direction numbers, the metadata the provenance reads, and never the editable-install metadata
    assert "collect_data_files('scipy', includes=['**/*.npz']" in spec and 'copy_metadata(dist)' in spec
    assert "endswith('/direct_url.json')" in spec
    # MuJoCo: the plugin folder, the GL backends chosen by MUJOCO_GL, PyOpenGL's platforms, glfw's library
    for need in ("'mujoco/plugin'", "'mujoco.egl'", "'mujoco.osmesa'", "'mujoco.glfw'", "'mujoco.cgl'",
                 "'OpenGL.raw.EGL'", "'OpenGL.osmesa'", "collect_dynamic_libs('glfw')"):
        assert need in spec, need
    # the extras left out by decision are excluded even when installed, and the spec checks it
    assert 'excludes += sorted({m for m, _ in left_out})' in spec and 'left out by decision were collected' in spec
    for gone in ("'tkinter'", "'PyQt6.QtWebEngineCore'", "'PyQt5'", "'torch'", "'zstandard.backend_cffi'",
                 "'mujoco.experimental'", "'pyarrow.flight'", "'PIL'"):
        assert gone in spec, gone
    assert "'PyQt6/Qt6/translations/'" in spec and 'release.license_files(_top' in spec
    # macOS: the Info.plist file, its version set from the package
    assert "os.path.join(HERE, 'macos', 'Info.plist')" in spec
    assert 'CFBundleShortVersionString=VERSION' in spec


def test_resource_lists_cover_every_resources_subfolder():
    """Every subfolder of dexvantage/resources (fonts, icons, library, robots, usd_robots, ...) and every non-Python
    file of the package goes into the bundle, at its own place; nothing from __pycache__."""
    release = _packaging_module('release')
    files = release.data_files()
    dests = {d.replace(os.sep, '/') for _, d in files}
    src = {os.path.relpath(f, ROOT).replace(os.sep, '/') for f, _ in files}
    subs = release.resource_dirs()
    assert {'fonts', 'icons', 'library', 'robots', 'usd_robots'} <= set(subs)
    on_disk = sorted(d for d in os.listdir(os.path.join(ROOT, 'dexvantage', 'resources'))
                     if os.path.isdir(os.path.join(ROOT, 'dexvantage', 'resources', d)) and d != '__pycache__')
    assert subs == on_disk
    for sub in subs:
        assert any(d == f'dexvantage/resources/{sub}' or d.startswith(f'dexvantage/resources/{sub}/') for d in dests), \
            sub
    for name in ('project.schema.json', 'library.schema.json'):
        assert f'dexvantage/resources/{name}' in src, name
    assert not [f for f in src if '__pycache__' in f or f.endswith(('.py', '.pyc'))]
    for f, d in files:
        rel = os.path.relpath(f, ROOT)
        assert d == ('.' if os.sep not in rel else os.path.dirname(rel)), (f, d)
    for d, subdirs, names in os.walk(os.path.join(ROOT, 'dexvantage')):
        subdirs[:] = [x for x in subdirs if x != '__pycache__']
        for n in names:
            if not n.endswith(('.py', '.pyc')):
                assert os.path.relpath(os.path.join(d, n), ROOT).replace(os.sep, '/') in src, n
    # at the bundle's top: the licence, the citation core/cite.py reads, the third-party record
    assert {'LICENSE', 'CITATION.cff', 'THIRD_PARTY.md'} <= src


def test_every_resource_is_package_data():
    """A file under dexvantage/ that is not Python must match a package-data glob, or a pip (wheel) install leaves it
    out: the bundle and pip install the same resources (the hardware library was missing from the globs)."""
    import glob
    globs = _pyproject()['tool']['setuptools']['package-data']['dexvantage']
    pkg = os.path.join(ROOT, 'dexvantage')
    covered = {os.path.normpath(p) for g in globs for p in glob.glob(os.path.join(pkg, g), recursive=True)}
    files = []
    for d, subdirs, names in os.walk(pkg):
        subdirs[:] = [s for s in subdirs if s != '__pycache__']
        files += [os.path.normpath(os.path.join(d, n)) for n in names if not n.endswith(('.py', '.pyc'))]
    assert files and not [os.path.relpath(f, ROOT) for f in files if f not in covered]
    release = _packaging_module('release')
    for sub in release.resource_dirs():
        assert any(g.startswith(f'resources/{sub}/') for g in globs), sub


# ------------------------------------------------------------------------------------------------ the launcher
def test_launcher_freeze_support_first_and_module_dispatch(tmp_path):
    """run_dexvantage.py calls multiprocessing.freeze_support() before anything parses the arguments, runs
    DexVantage's own modules for "-m" (the render, GL and shard workers), and refuses any other module."""
    tree = ast.parse(_text('packaging', 'run_dexvantage.py'))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    first = main.body[0]
    assert isinstance(first, ast.Expr) and ast.unparse(first) == 'multiprocessing.freeze_support()'
    launcher = os.path.join(PKG, 'run_dexvantage.py')
    env = _env(tmp_path)
    r = subprocess.run([sys.executable, launcher, '-m', 'dexvantage', '--version'], capture_output=True, text=True,
                       env=env, timeout=120)
    from dexvantage import __version__
    assert r.returncode == 0 and r.stdout.strip() == f'dexvantage {__version__}', r.stderr
    r = subprocess.run([sys.executable, launcher, '-m', 'os'], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 2 and "DexVantage's own modules only" in r.stderr
    r = subprocess.run([sys.executable, launcher, '--version'], capture_output=True, text=True, env=env, timeout=120)
    assert r.stdout.strip() == f'dexvantage {__version__}', r.stderr


def test_spawned_workers_use_the_module_dispatch():
    """The worker processes DexVantage starts are '[sys.executable, "-m", "dexvantage…"]', which the launcher
    handles in a bundle; a new form of worker launch must be added there too."""
    pat = re.compile(r"\[sys\.executable,\s*([^\]]*)")
    found = []
    for d, subdirs, names in os.walk(os.path.join(ROOT, 'dexvantage')):
        subdirs[:] = [x for x in subdirs if x != '__pycache__']
        for n in names:
            if n.endswith('.py'):
                for m in pat.finditer(open(os.path.join(d, n), encoding='utf-8').read()):
                    found.append((n, m.group(1)))
    assert found
    for n, args in found:
        assert re.match(r"'-m',\s*'dexvantage(\.|')", args), (n, args)


# ------------------------------------------------------------------------------------------------ extras
def test_every_extra_has_a_bundling_decision():
    from dexvantage.extras import EXTRAS, TOOL_EXTRAS
    opt = _pyproject()['project']['optional-dependencies']
    assert set(opt) == set(EXTRAS) | set(TOOL_EXTRAS), set(opt) ^ (set(EXTRAS) | set(TOOL_EXTRAS))
    assert not set(EXTRAS) & set(TOOL_EXTRAS)
    assert {n for n, e in EXTRAS.items() if e.bundled} == {'lerobot', 'bags', 'assistant'}
    for name, e in EXTRAS.items():
        assert e.modules and e.what and e.why, name
        assert e.extra_mb is None or e.extra_mb > 0, name
    # the bundle's requirements carry the bundled extras and none of the others
    pins = {_req_name(ln) for ln in _text('packaging', 'requirements-bundle.txt').splitlines()
            if ln.split('#')[0].strip()}
    for name, e in EXTRAS.items():
        reqs = {_req_name(r) for r in opt[name]}
        assert (reqs <= pins) if e.bundled else not (reqs & pins), (name, reqs)


def test_bundle_note_only_in_the_downloadable_app(monkeypatch):
    from dexvantage import extras
    assert extras.bundle_note('rerun') == '' and extras.bundled('rerun')
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    note = extras.bundle_note('rerun')
    assert note.startswith(' This DexVantage download leaves out the Rerun bridge (it would add about ')
    assert note.endswith('554 MB): run DexVantage from a pip installation to use it.')
    assert extras.bundle_note('lerobot') == '' and extras.bundled('lerobot') and not extras.bundled('usd')
    assert extras.bundle_note('no-such-extra') == ''


def test_missing_extra_is_one_sentence(monkeypatch):
    """A feature whose extra is not installed gives one sentence naming the extra: its own pip command from a pip
    installation; in the downloadable app, which pip cannot add to, the download's sentence instead (not both)."""
    from dexvantage import extras
    from dexvantage.core.robot import usd_import
    pip = usd_import.missing_pxr_message('robot.usd')
    assert 'pip install "dexvantage[usd]"' in pip and 'download' not in pip
    assert extras.missing('lerobot', 'x') == 'x'                         # bundled: its own text, even when frozen
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    assert extras.missing('lerobot', 'x') == 'x'
    msg = usd_import.missing_pxr_message('robot.usd')
    assert msg.startswith('This DexVantage download leaves out OpenUSD import') and msg.count('. ') == 0
    assert 'the usd extra (pip install -e ".[usd]" in its folder)' in msg and msg.endswith('import that.')
    assert 'dexvantage[usd]' not in msg
    import builtins

    from dexvantage.core.exporters import bridges
    from dexvantage.core.jobs import StudyError
    real = builtins.__import__

    def no_gmsh(name, *a, **k):
        if name == 'gmsh':
            raise ImportError(name)
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, '__import__', no_gmsh)
    with pytest.raises(StudyError) as e:
        bridges.step_from_mesh([[0, 0, 0]], [[0, 0, 0]], 'x.step')
    assert str(e.value).startswith('This DexVantage download leaves out bracket STEP export')
    assert str(e.value).endswith('to use it, or export STL only.') and 'dexvantage[step]' not in str(e.value)


def test_missing_extra_messages_use_the_one_sentence():
    """Each feature of a left-out extra builds its "is not installed" message through extras.missing."""
    for parts, extra in ((('dexvantage', 'core', 'bridges', 'rerun.py'), 'rerun'),
                         (('dexvantage', 'core', 'robot', 'usd_import.py'), 'usd'),
                         (('dexvantage', 'core', 'exporters', 'isaac.py'), 'usd'),
                         (('dexvantage', 'ui', 'dialogs', 'import_wizard.py'), 'usd'),
                         (('dexvantage', 'core', 'exporters', 'bridges.py'), 'step')):
        assert f"missing('{extra}'" in _text(*parts), parts


# ------------------------------------------------------------------------------------------------ release.py
def _fake_artefacts(dist):
    dist.mkdir(parents=True)
    files = {'DexVantage-0.1.0.dev0-linux-x86_64.tar.gz': b'linux bundle',
             'DexVantage-0.1.0.dev0-windows-x64-setup.exe': b'windows installer',
             'DexVantage-0.1.0.dev0-macos-arm64.zip': b'mac app'}
    for n, data in files.items():
        (dist / n).write_bytes(data)
    (dist / 'smoke-report.json').write_text('{}')            # not a release file
    return files


def test_release_dry_run_writes_checksums_and_notes(tmp_path):
    """release.py --dry-run builds nothing, writes SHA256SUMS.txt over the artefacts and RELEASE_NOTES.md from the
    CHANGELOG into the output folder, and only prints the gh release command."""
    files = _fake_artefacts(tmp_path / 'dist')
    out = tmp_path / 'release'
    r = subprocess.run([sys.executable, os.path.join(PKG, 'release.py'), '--dry-run', '--dist', str(tmp_path / 'dist'),
                        '--out', str(out)], capture_output=True, text=True, timeout=120, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    sums = (out / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines()
    assert sums == [f'{hashlib.sha256(files[n]).hexdigest()}  {n}' for n in sorted(files)]
    notes = (out / 'RELEASE_NOTES.md').read_text(encoding='utf-8')
    release = _packaging_module('release')
    entries = {v: body for v, _, body in release.changelog()}
    ver = release.version()
    if ver in entries:                                       # a version with its own entry (0.1.0rc1)
        assert notes == entries[ver]
    else:                                                    # a development version: the [Unreleased] notes
        unreleased = next(body for v, body in entries.items() if v.lower() == 'unreleased')
        assert notes.endswith(unreleased) and notes.startswith('(Dry run: CHANGELOG.md has no entry for')
    assert 'Not published.' in r.stdout and 'gh release create v' in r.stdout and '--notes-file' in r.stdout
    assert ('--prerelease' in r.stdout) == release.is_prerelease(ver)        # a .dev or rc version
    assert sorted(os.listdir(tmp_path / 'dist')) == sorted(list(files) + ['smoke-report.json'])   # untouched


def test_release_never_runs_gh_without_publish(tmp_path, monkeypatch):
    release = _packaging_module('release')
    _fake_artefacts(tmp_path / 'dist')

    def no_run(*a, **k):
        raise AssertionError(f'release.py ran {a[0] if a else k}')
    monkeypatch.setattr(release.subprocess, 'run', no_run)
    dist = str(tmp_path / 'dist')
    assert release.release(dist, str(tmp_path / 'r'), build=False, dry_run=True) == 0
    assert (tmp_path / 'r' / 'SHA256SUMS.txt').is_file()
    dev = tmp_path / 'dev'
    (dev / 'dexvantage').mkdir(parents=True)
    (dev / 'dexvantage' / '__init__.py').write_text('__version__ = "0.3.0.dev0"\n')
    (dev / 'CHANGELOG.md').write_text('# Changelog\n\n## [Unreleased]\n- next\n', encoding='utf-8')
    # without --dry-run a development version has no CHANGELOG entry of its own: refused, nothing written or run
    assert release.release(dist, str(tmp_path / 'r1'), build=False, root=str(dev)) == 1
    assert not (tmp_path / 'r1').exists()
    # --publish of a development version is refused before anything runs; --publish with --dry-run too
    assert release.release(dist, str(tmp_path / 'r2'), build=False, publish=True, root=str(dev)) == 1
    assert release.release(dist, str(tmp_path / 'r3'), dry_run=True, publish=True) == 2
    # a release candidate is not a development version: it may be published, as a pre-release
    assert release.is_dev('0.3.0.dev0') and not release.is_dev('0.1.0rc1') and release.is_prerelease('0.1.0rc1')
    # a released version (a source tree whose version has its CHANGELOG entry): notes from that entry, the command
    # printed, and still nothing run without --publish
    src = tmp_path / 'src'
    (src / 'dexvantage').mkdir(parents=True)
    (src / 'dexvantage' / '__init__.py').write_text('__version__ = "0.2.0"\n')
    (src / 'CHANGELOG.md').write_text('# Changelog\n\n## [0.2.0] - 2026-12-01\n- the release\n', encoding='utf-8')
    assert release.release(dist, str(tmp_path / 'r4'), build=False, root=str(src)) == 0
    assert (tmp_path / 'r4' / 'RELEASE_NOTES.md').read_text(encoding='utf-8') == '- the release\n'
    cmd = release.gh_command('0.1.0', ['a.tar.gz'], 'notes.md')
    assert cmd[:4] == ['gh', 'release', 'create', 'v0.1.0'] and '--prerelease' not in cmd and '--verify-tag' in cmd


def test_release_notes_command():
    from dexvantage import __version__
    cmd = [sys.executable, os.path.join(PKG, 'release.py')]
    r = subprocess.run(cmd + ['version'], capture_output=True, text=True, timeout=60)
    assert r.stdout.strip() == __version__
    r = subprocess.run(cmd + ['notes', '--tag', 'v99.0.0'], capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and 'does not match' in r.stderr


def test_changelog_reader(tmp_path):
    release = _packaging_module('release')
    (tmp_path / 'CHANGELOG.md').write_text('# Changelog\n\n## [Unreleased]\n- next\n\n## [0.2.0] - 2026-11-01\n'
                                           '- two\n\n## 0.1.0\n- one\n', encoding='utf-8')
    assert [v for v, _, _ in release.changelog(str(tmp_path))] == ['Unreleased', '0.2.0', '0.1.0']
    assert release.notes('0.2.0', str(tmp_path)) == '- two\n'
    with pytest.raises(ValueError):
        release.notes('9.9.9', str(tmp_path))
    assert release.release_notes('0.3.0.dev0', str(tmp_path), allow_unreleased=True).endswith('- next\n')
    with pytest.raises(ValueError):
        release.release_notes('0.3.0.dev0', str(tmp_path))


# ------------------------------------------------------------------------------------------------ build.py
def test_self_test_problems():
    """build.py's reading of a bundle's self-test: a failure, a warning outside BUNDLE_MAY_WARN, a check that did not
    run or a missing verdict fails the smoke test; an OpenGL warning does not."""
    from dexvantage.selftest import BUNDLE_MAY_WARN, NAMES
    build = _packaging_module('build')

    def out(status=None, drop=(), end='self-test passed in 20 s'):
        status = status or {}
        lines = ['DexVantage 0.1.0 self-test · Python 3.14 · linux · bundled']
        lines += [f'  {n:<15} {status.get(n, "ok"):<8} fine' for n in NAMES if n not in drop]
        return '\n'.join(lines + [end]) + '\n'
    assert build.self_test_problems(out()) == []
    assert BUNDLE_MAY_WARN == ('opengl',)
    assert build.self_test_problems(out({'opengl': 'warning'})) == []
    bad = build.self_test_problems(out({'mujoco': 'warning'}))
    assert len(bad) == 1 and bad[0].startswith('mujoco')
    assert build.self_test_problems(out({'opengl': 'FAILED'}))
    assert build.self_test_problems(out(drop=('fonts',))) == ['fonts: did not run']
    assert build.self_test_problems(out(end='self-test FAILED in 3 s')) == ['no "self-test passed" line']


def test_archive_names_match_the_installers():
    build = _packaging_module('build')
    iss = _text('packaging', 'windows', 'dexvantage.iss')
    assert 'OutputBaseFilename=DexVantage-{#AppVersion}-windows-x64-setup' in iss
    assert "f'{tag}-windows-x64-setup.exe'" in _text('packaging', 'build.py')
    assert build.VERSION == _packaging_module('release').version()
    assert build.arch() in ('x86_64', 'arm64', 'x64') or build.arch()
    for f in build.LINUX_FILES:
        assert os.path.isfile(os.path.join(PKG, 'linux', f)), f
    assert build.cli_path('d').endswith(('dexvantage-cli', 'dexvantage-cli.exe'))


def test_bundle_requirements():
    """packaging/requirements-bundle.txt pins every requirement of the app (OpenCV as its headless build), the bundled
    extras and PyInstaller, each to one version; PyQt6 per machine (Jetson's glibc)."""
    deps = {_req_name(d) for d in _pyproject()['project']['dependencies']}
    pins, marked = {}, {}
    for ln in _text('packaging', 'requirements-bundle.txt').splitlines():
        ln = ln.split('#')[0].strip()
        if ln:
            spec, _, marker = ln.partition(';')
            name, _, ver = spec.partition('==')
            assert ver.strip() and all(c.isdigit() or c == '.' for c in ver.strip()), ln
            pins[name.strip().lower()] = ver.strip()
            if marker.strip():
                marked.setdefault(name.strip().lower(), set()).add((marker.strip(), ver.strip()))
    for name in ('pyqt6', 'pyqt6-qt6'):
        m = dict(marked[name])
        assert set(m) == {'platform_machine != "aarch64"', 'platform_machine == "aarch64"'}, m
        assert m['platform_machine == "aarch64"'].startswith('6.7.')
    want = {('opencv-python-headless' if d == 'opencv-python' else d) for d in deps}
    want |= {'pyinstaller', 'pyinstaller-hooks-contrib', 'pyqt6-qt6', 'glfw', 'pyarrow', 'av'}
    assert want <= set(pins), want - set(pins)
    assert 'opencv-python' not in pins
    bundle_extra = {_req_name(r): r.split('==')[1] for r in _pyproject()['project']['optional-dependencies']['bundle']
                    if '==' in r}
    assert bundle_extra['pyinstaller'] == pins['pyinstaller']
    assert bundle_extra['pyinstaller-hooks-contrib'] == pins['pyinstaller-hooks-contrib']


def test_license_notice(tmp_path):
    import importlib.metadata as md
    release = _packaging_module('release')
    notice = tmp_path / 'THIRD-PARTY.txt'
    mods = {'yaml', 'numpy', 'no_such_module_here'}
    files = release.license_files(mods, str(notice))
    by_mod = md.packages_distributions()
    dists = {d for m in mods for d in by_mod.get(m, ())}
    assert dists and (str(notice), 'licenses') in files
    assert all(os.path.isfile(src) for src, _ in files)
    text = notice.read_text(encoding='utf-8')
    assert text.startswith('DexVantage ') and 'THIRD_PARTY.md' in text
    for d in dists:
        assert f'{d} {md.version(d)} — ' in text, d


# ------------------------------------------------------------------------------------------------ installers
def test_inno_setup_script():
    """Per-user install without administrator rights, the .dexv association, and a self-test shortcut whose command
    line survives cmd.exe's quote stripping wherever the program is installed."""
    iss = _text('packaging', 'windows', 'dexvantage.iss')
    for line in ('PrivilegesRequired=lowest', 'DefaultDirName={autopf}\\DexVantage', 'ChangesAssociations=yes',
                 'ArchitecturesInstallIn64BitMode=x64compatible', 'SetupIconFile=..\\icons\\dexvantage.ico',
                 'VersionInfoVersion={#AppVersion}', 'LicenseFile=..\\..\\LICENSE'):
        assert line in iss, line
    assert 'Software\\Classes\\.dexv\\OpenWithProgids' in iss
    assert 'ValueData: """{app}\\DexVantage.exe"" ""%1"""' in iss
    assert 'Vantage.exe' not in iss.replace('DexVantage.exe', '')       # nothing of Vantage's left
    shortcut = next(ln for ln in iss.splitlines() if ln.startswith('Name: "{group}\\DexVantage self-test"'))
    quoted = shortcut.split('Parameters: "', 1)[1].rsplit('"; IconFilename', 1)[0]
    params = quoted.replace('""', '"')
    for app in (r'C:\Users\ana\AppData\Local\Programs\DexVantage', r'C:\Program Files\DexVantage',
                r'C:\Users\Sam Lee (work)\AppData\Local\Programs\DexVantage'):
        cmdline = params.replace('{app}', app)[3:]
        assert cmdline.count('"') > 2
        i, j = cmdline.index('"'), cmdline.rindex('"')
        assert cmdline[:i] + cmdline[i + 1:j] + cmdline[j + 1:] == f'"{app}\\dexvantage-cli.exe" --self-test'


def test_macos_info_plist():
    with open(os.path.join(PKG, 'macos', 'Info.plist'), 'rb') as fh:
        info = plistlib.load(fh)
    assert info['CFBundleIdentifier'] == 'io.github.sherin-sef-ai.dexvantage'
    assert info['LSMinimumSystemVersion'] == '14.0' and info['LSArchitecturePriority'] == ['arm64']
    assert info['LSBackgroundOnly'] is False and info['NSHighResolutionCapable'] is True
    uti = info['UTExportedTypeDeclarations'][0]
    assert uti['UTTypeTagSpecification']['public.filename-extension'] == ['dexv']
    assert info['CFBundleDocumentTypes'][0]['LSItemContentTypes'] == [uti['UTTypeIdentifier']]
    with open(os.path.join(PKG, 'macos', 'entitlements.plist'), 'rb') as fh:
        ent = plistlib.load(fh)
    assert ent == {'com.apple.security.cs.allow-unsigned-executable-memory': True,
                   'com.apple.security.cs.disable-library-validation': True}
    sh = _text('packaging', 'macos', 'sign_and_notarise.sh')
    assert '--options runtime' in sh and 'notarytool submit' in sh and 'stapler staple' in sh


def test_icons_are_valid():
    from dexvantage.app import APP_ICON
    with open(APP_ICON, 'rb') as fh:
        assert fh.read(8) == b'\x89PNG\r\n\x1a\n'
    with open(os.path.join(PKG, 'icons', 'dexvantage.ico'), 'rb') as fh:
        reserved, kind, count = struct.unpack('<HHH', fh.read(6))
        assert (reserved, kind) == (0, 1) and count >= 4
    with open(os.path.join(PKG, 'icons', 'dexvantage.icns'), 'rb') as fh:
        head = fh.read(8)
        assert head[:4] == b'icns' and struct.unpack('>I', head[4:])[0] == os.path.getsize(fh.name)


def _exec_argv(value):
    """A desktop entry's Exec value as the Desktop Entry spec reads it (string escapes, then quoting)."""
    s, i = '', 0
    while i < len(value):
        if value[i] == '\\':
            nxt = value[i + 1]
            assert nxt in 'sntr\\', f'invalid string escape \\{nxt} in {value!r}'
            s += {'s': ' ', 'n': '\n', 't': '\t', 'r': '\r', '\\': '\\'}[nxt]
            i += 2
        else:
            s += value[i]
            i += 1
    args, cur, quoted, i = [], None, False, 0
    while i < len(s):
        c = s[i]
        if quoted:
            if c == '\\':
                assert s[i + 1] in '"`$\\', s
                cur += s[i + 1]
                i += 1
            elif c == '"':
                quoted = False
            else:
                cur += c
        elif c == '"':
            quoted, cur = True, cur or ''
        elif c == ' ':
            if cur is not None:
                args.append(cur)
            cur = None
        else:
            cur = (cur or '') + c
        i += 1
    if cur is not None:
        args.append(cur)
    return args


@pytest.mark.skipif(os.name == 'nt', reason="the Linux installer's escaping: Windows hands the sh of Git or MSYS its "
                    'arguments through its own command-line quoting, which is not what install.sh sees on Linux')
@pytest.mark.skipif(not shutil.which('sh') or not shutil.which('sed'), reason='needs sh and sed')
def test_exec_line_escaping():
    script = _text('packaging', 'linux', 'install.sh')
    sed = next(ln for ln in script.splitlines() if ln.startswith('exe=$(')).split('|', 1)[1].rsplit(')', 1)[0]
    assert sed in _text('packaging', 'linux', 'uninstall.sh')
    for path in ('/opt/dexvantage/DexVantage', '/srv/a b/.local/opt/dexvantage/DexVantage', '/x/$HOME/V',
                 '/x/q"t/V', '/x/back\\slash/V', '/x/50% `tick`/V'):
        r = subprocess.run(['sh', '-c', f"printf '%s' \"$1\" | {sed}", 'sh', path], capture_output=True, text=True)
        line = f'Exec="{r.stdout}" %f'
        assert _exec_argv(line[5:]) == [path.replace('%', '%%'), '%f'], (path, line)


@pytest.mark.skipif(not sys.platform.startswith('linux') or not shutil.which('sh'), reason='Linux installer')
def test_install_and_uninstall_sh(tmp_path):
    """install.sh from an extracted bundle into a home with a space and a $ in its path, with XDG_DATA_HOME
    elsewhere: the program, the command, the menu entry, the icon and the file type; uninstall.sh removes them."""
    import configparser
    src = os.path.join(PKG, 'linux')
    bundle = tmp_path / 'DexVantage-0.1.0-linux-x86_64'
    icon_dir = bundle / '_internal' / 'dexvantage' / 'resources' / 'icons'
    icon_dir.mkdir(parents=True)
    shutil.copy(os.path.join(ROOT, 'dexvantage', 'resources', 'icons', 'dexvantage.png'), icon_dir / 'dexvantage.png')
    for exe in ('DexVantage', 'dexvantage-cli'):
        p = bundle / exe
        p.write_text('#!/bin/sh\necho "dexvantage 0.1.0"\n')
        p.chmod(0o755)
    for f in ('install.sh', 'uninstall.sh', 'dexvantage.desktop', 'dexvantage-mime.xml'):
        shutil.copy2(os.path.join(src, f), bundle / f)
    home = tmp_path / 'my home $USER'
    data = tmp_path / 'data home'
    home.mkdir()
    env = {'HOME': str(home), 'XDG_DATA_HOME': str(data), 'PATH': '/usr/bin:/bin'}
    r = subprocess.run(['sh', str(bundle / 'install.sh')], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    dest = home / '.local' / 'opt' / 'dexvantage'
    assert (dest / 'DexVantage').is_file() and (dest / '.dexvantage-install').is_file()
    link = home / '.local' / 'bin' / 'dexvantage'
    assert link.is_symlink() and os.readlink(link) == str(dest / 'dexvantage-cli')
    assert subprocess.run([str(link)], capture_output=True, text=True).stdout.startswith('dexvantage 0.1.0')
    entry = data / 'applications' / 'dexvantage.desktop'
    cp = configparser.ConfigParser(interpolation=None)
    cp.optionxform = str
    cp.read(entry)
    e = cp['Desktop Entry']
    assert e['Icon'] == 'dexvantage' and 'application/x-dexvantage-project' in e['MimeType']
    assert _exec_argv(e['Exec']) == [str(dest / 'DexVantage'), '%f'], e['Exec']
    if shutil.which('desktop-file-validate'):
        v = subprocess.run(['desktop-file-validate', str(entry)], capture_output=True, text=True)
        assert v.returncode == 0, v.stdout + v.stderr
    assert (data / 'icons' / 'hicolor' / '256x256' / 'apps' / 'dexvantage.png').is_file()
    assert '*.dexv' in (data / 'mime' / 'packages' / 'dexvantage.xml').read_text()
    assert not (home / '.local' / 'share').exists()
    assert subprocess.run(['sh', str(bundle / 'install.sh')], env=env, capture_output=True).returncode == 0
    r = subprocess.run(['sh', str(dest / 'uninstall.sh')], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert not dest.exists() and not link.exists() and not entry.exists()
    assert not (data / 'mime' / 'packages' / 'dexvantage.xml').exists()
    assert not (data / 'icons' / 'hicolor' / '256x256' / 'apps' / 'dexvantage.png').exists()
    dest.mkdir(parents=True)
    r = subprocess.run(['sh', str(bundle / 'install.sh')], env=env, capture_output=True, text=True)
    assert r.returncode == 1 and 'not made by this installer' in r.stderr


def test_desktop_entry_matches_the_app():
    """The menu entry's id and window class are the ones the app announces (app.setDesktopFileName('dexvantage'))."""
    import configparser
    cp = configparser.ConfigParser(interpolation=None)
    cp.optionxform = str
    cp.read(os.path.join(PKG, 'linux', 'dexvantage.desktop'))
    e = cp['Desktop Entry']
    assert e['Type'] == 'Application' and e['Name'] == 'DexVantage' and e['Exec'] == '@EXEC@ %f'
    assert "setDesktopFileName('dexvantage')" in _text('dexvantage', 'app.py')
    from dexvantage.core.project import EXTENSION
    assert f'<glob pattern="*{EXTENSION}"/>' in _text('packaging', 'linux', 'dexvantage-mime.xml')
