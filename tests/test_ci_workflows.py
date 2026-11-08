"""The CI machinery (.github/, scripts/ci_shards.py, scripts/get_actionlint.py, tests/render_support.py): the
workflows parse, every action is pinned by commit SHA, the Python versions agree with pyproject.toml, the draft
release never publishes, the shards cover every test file once, and the render skip says why. actionlint runs on the
workflows when its binary is here (python scripts/get_actionlint.py fetches it into out/tools). No network."""
import glob
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WF = os.path.join(ROOT, '.github', 'workflows')
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import ci_shards  # noqa: E402
import get_actionlint  # noqa: E402

SHA_PIN = re.compile(r'^[\w.-]+/[\w.-]+(/[\w./-]+)?@[0-9a-f]{40}$')


def _workflows():
    files = sorted(glob.glob(os.path.join(WF, '*.yml')) + glob.glob(os.path.join(WF, '*.yaml')))
    assert files, 'no workflow files'
    return files


def _load(name):
    with open(os.path.join(WF, name), encoding='utf-8') as f:
        doc = yaml.safe_load(f)
    doc['on'] = doc.pop('on', None) if 'on' in doc else doc.pop(True)      # YAML 1.1 reads the key `on` as True
    return doc


def _steps(doc):
    for job in doc['jobs'].values():
        yield from job.get('steps', [])


def _pyproject():
    try:
        import tomllib
    except ImportError:                     # Python 3.10
        pytest.skip('tomllib needs Python 3.11 or newer')
    with open(os.path.join(ROOT, 'pyproject.toml'), 'rb') as f:
        return tomllib.load(f)


@pytest.mark.parametrize('path', _workflows(), ids=os.path.basename)
def test_workflow_parses_and_every_action_is_pinned_by_sha(path):
    with open(path, encoding='utf-8') as f:
        text = f.read()
    doc = yaml.safe_load(text)
    assert isinstance(doc, dict) and doc.get('jobs') and doc.get('name')
    assert doc.get('permissions') == {'contents': 'read'}, 'the workflow token is read-only by default'
    uses = re.findall(r'^\s*(?:-\s*)?uses:\s*(\S+)(.*)$', text, re.M)
    assert uses
    for ref, rest in uses:
        if ref.startswith('./'):
            continue
        assert SHA_PIN.match(ref), f'{ref} is not pinned by a full commit SHA'
        assert re.match(r'\s*#\s*v\d+(\.\d+)*\s*$', rest), f'{ref} has no "# vX.Y.Z" comment naming its release'
    # the same action is at the same commit everywhere
    seen = {}
    for ref, rest in uses:
        name, _, sha = ref.partition('@')
        assert seen.setdefault(name, (sha, rest.strip())) == (sha, rest.strip()), f'{name} is pinned twice'


def test_every_workflow_uses_the_same_action_commits():
    pins = {}
    for path in _workflows():
        with open(path, encoding='utf-8') as f:
            for ref in re.findall(r'uses:\s*([^\s#]+@[0-9a-f]{40})', f.read()):
                name, sha = ref.split('@')
                assert pins.setdefault(name, sha) == sha, f'{name} has two different commits across the workflows'


def test_python_versions_agree_with_pyproject():
    pp = _pyproject()['project']
    floor = re.fullmatch(r'>=\s*3\.(\d+)', pp['requires-python'].strip())
    assert floor, 'requires-python should be a plain lower bound'
    listed = sorted(int(m.group(1)) for c in pp['classifiers']
                    if (m := re.fullmatch(r'Programming Language :: Python :: 3\.(\d+)', c)))
    assert listed and listed[0] == int(floor.group(1)), 'the oldest classifier is the requires-python floor'
    assert listed == list(range(listed[0], listed[-1] + 1)), 'no gaps in the classifiers'
    ci = _load('ci.yml')
    tested = {t['python'] for t in ci['jobs']['tests']['strategy']['matrix']['target']}
    assert tested == {'3.12', f'3.{listed[-1]}'}, 'CI tests 3.12 and the newest version pyproject claims'
    oldest = [s for s in ci['jobs']['oldest-python']['steps'] if 'setup-python' in s.get('uses', '')]
    assert oldest[0]['with']['python-version'] == f'3.{listed[0]}'
    for s in _steps(ci):
        if 'setup-python' in s.get('uses', ''):
            v = str(s['with']['python-version'])
            assert v.startswith('${{') or int(v.split('.')[1]) in listed, f'CI runs Python {v}, not a claimed one'


def test_ci_triggers_platforms_shards_and_steps():
    ci = _load('ci.yml')
    assert 'push' in ci['on'] and 'pull_request' in ci['on']
    assert ci['env']['QT_QPA_PLATFORM'] == 'offscreen'
    tests = ci['jobs']['tests']
    oses = {t['os'] for t in tests['strategy']['matrix']['target']}
    assert oses == {'ubuntu-24.04'}                                         # DexVantage supports Linux only
    shards = tests['strategy']['matrix']['shard']
    run = next(s['run'] for s in tests['steps'] if 'ci_shards.py' in s.get('run', ''))
    n = int(re.search(r'--shards (\d+)', run).group(1))
    assert shards == list(range(n)) and f'/{n}' in tests['name']
    assert tests['env']['DEXVANTAGE_TEST_RENDER'] == 'require'             # must render, never skip
    assert {t['os'] for t in ci['jobs']['checks']['strategy']['matrix']['target']} == {'ubuntu-24.04'}
    checks = ci['jobs']['checks']
    runs = [s.get('run', '') for s in checks['steps']]
    assert any(r.strip() == 'python -m dexvantage --self-test' for r in runs)
    assert any('--validate out/validation --quick' in r for r in runs)
    up = [s for s in checks['steps'] if 'upload-artifact' in s.get('uses', '')]
    assert up and up[0]['with']['path'] == 'out/validation'
    lint = [s.get('run', '') for s in ci['jobs']['lint']['steps']]
    assert any('ruff check' in r for r in lint) and any('get_actionlint.py' in r for r in lint)
    for job in ci['jobs'].values():
        assert job.get('timeout-minutes'), 'every job has a time limit'


def test_linux_installs_the_mesa_egl_and_qt_libraries():
    with open(os.path.join(ROOT, '.github', 'scripts', 'apt-gl-qt.sh'), encoding='utf-8') as f:
        text = f.read()
    pkgs = set(re.findall(r'\b(lib[\w.+-]+)', text.split('--no-install-recommends', 1)[1]))
    for p in ('libegl1', 'libegl-mesa0', 'libgl1-mesa-dri', 'libopengl0', 'libxkbcommon0', 'libfontconfig1',
              'libxcb-cursor0', 'libdbus-1-3',
              # gmsh's libgmsh.so (the extra "step"): the first CI run failed formats.bracket_step without libGLU
              'libglu1-mesa', 'libxrender1', 'libxcursor1', 'libxfixes3', 'libxext6', 'libxft2', 'libxinerama1',
              'libgomp1'):
        assert p in pkgs, p
    assert 'glib=libglib2.0-0t64' in text and '|| glib=libglib2.0-0' in text     # 24.04's name, else 22.04's
    for path in _workflows():
        with open(path, encoding='utf-8') as f:
            for ln in f:
                if 'apt-gl-qt.sh' in ln and 'run:' in ln:
                    assert 'bash .github/scripts/apt-gl-qt.sh' in ln     # needs no executable bit


def test_bundles_build_on_linux_only_and_the_release_stays_a_draft():
    b = _load('bundles.yml')
    assert set(b['on']) == {'push', 'workflow_dispatch'} and b['on']['push'] == {'tags': ['v*']}
    # Linux on 22.04 (glibc 2.35), so the bundles also run on 22.04 and on Jetson with JetPack 6
    assert b['jobs']['bundle']['strategy']['matrix']['os'] == ['ubuntu-22.04', 'ubuntu-22.04-arm']
    runs = ' '.join(s.get('run', '') for s in b['jobs']['bundle']['steps'])
    assert 'packaging/build.py --smoke' in runs and '--self-test' in runs
    rel = b['jobs']['draft-release']
    assert "startsWith(github.ref, 'refs/tags/v')" in rel['if'] and rel['needs'] == 'bundle'
    text = ' '.join(s.get('run', '') for s in rel['steps'])
    assert 'gh release create' in text and '--draft' in text
    for bad in ('--draft=false', 'gh release edit', 'gh release publish', 'git push', 'git tag'):
        assert bad not in text, bad
    # only the release job may write to the repository
    for name, job in b['jobs'].items():
        if name != 'draft-release':
            assert 'permissions' not in job


def test_constraints_pin_exact_versions():
    with open(os.path.join(ROOT, '.github', 'ci-constraints.txt'), encoding='utf-8') as f:
        lines = [ln.split('#')[0].strip() for ln in f]
    pins = [ln for ln in lines if ln]
    assert pins and all(re.fullmatch(r'[A-Za-z0-9_.-]+==[\w.]+', p) for p in pins), pins
    names = {p.split('==')[0].lower() for p in pins}
    assert {'mujoco', 'pyqt6', 'numpy', 'pytest', 'ruff'} <= names


def test_no_personal_paths_or_attribution_in_ci_files():
    files = (_workflows() + glob.glob(os.path.join(ROOT, '.github', '**', '*'), recursive=True)
             + [os.path.join(ROOT, 'CONTRIBUTING.md'), os.path.join(ROOT, 'scripts', 'ci_shards.py'),
                os.path.join(ROOT, 'scripts', 'get_actionlint.py'), os.path.join(ROOT, 'tests', 'render_support.py')])
    for p in sorted(set(f for f in files if os.path.isfile(f))):
        with open(p, encoding='utf-8') as f:
            text = f.read()
        assert not re.search(r'/home/\w|/Users/\w|C:\\Users\\', text), p
        assert 'Co-Authored-By' not in text and 'Generated with' not in text, p


# ------------------------------------------------------------------------------------------------ actionlint
def _actionlint():
    for cand in (os.environ.get('ACTIONLINT'), get_actionlint.binary_path(), shutil.which('actionlint')):
        if cand and os.path.isfile(cand):
            return cand
    return None


def test_actionlint_finds_nothing():
    exe = _actionlint()
    if exe is None:
        pytest.skip('actionlint is not here (python scripts/get_actionlint.py fetches the pinned binary); the YAML '
                    'and pin checks above ran without it')
    r = subprocess.run([exe, '-no-color'], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


def test_actionlint_download_is_refused_when_the_checksum_differs(tmp_path, monkeypatch):
    blob = io.BytesIO()
    with tarfile.open(fileobj=blob, mode='w:gz') as t:
        data = b'#!/bin/sh\necho fake\n'
        info = tarfile.TarInfo('actionlint')
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
    payload = blob.getvalue()

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(get_actionlint.urllib.request, 'urlopen', lambda url, timeout=60: Resp(payload))
    monkeypatch.setattr(get_actionlint, 'target', lambda: 'linux_amd64')
    monkeypatch.setattr(get_actionlint.sys, 'platform', 'linux')
    with pytest.raises(RuntimeError, match='not the published'):
        get_actionlint.fetch(str(tmp_path))
    assert not os.path.exists(get_actionlint.binary_path(str(tmp_path)))
    # the same archive with its own checksum pinned is unpacked and made executable
    monkeypatch.setitem(get_actionlint.SHA256, 'linux_amd64', hashlib.sha256(payload).hexdigest())
    path = get_actionlint.fetch(str(tmp_path))
    with open(path, 'rb') as f:
        assert f.read() == data
    assert get_actionlint.fetch(str(tmp_path)) == path                     # once there, nothing is fetched again


def test_actionlint_platform_keys():
    assert set(get_actionlint.SHA256) == {'linux_amd64', 'linux_arm64'}
    assert all(re.fullmatch(r'[0-9a-f]{64}', v) for v in get_actionlint.SHA256.values())
    assert re.fullmatch(r'[a-z]+_[a-z0-9]+', get_actionlint.target())


# ------------------------------------------------------------------------------------------------ shards
def test_shards_cover_every_test_file_once():
    files = ci_shards.test_files()
    assert 'test_ci_workflows.py' in files and len(files) > 50
    for n in (1, 2, 6, 9):
        shards = ci_shards.split(files, n)
        assert len(shards) == n
        flat = [f for s in shards for f in s]
        assert sorted(flat) == files and len(flat) == len(set(flat))
    six = ci_shards.split(files, 6)
    assert six[0] == ['test_validation.py']                                # alone: its memory, not its time
    assert six == ci_shards.split(list(reversed(files)), 6)                # the same split however listed


def test_shards_balance_the_measured_time():
    files = ci_shards.test_files()
    w = ci_shards.weights(files)
    assert all(v > 0 for v in w.values())
    loads = [sum(w[f] for f in s) for s in ci_shards.split(files, 6, w)[1:]]
    assert max(loads) - min(loads) <= max(w[f] for f in files if f != 'test_validation.py')
    # a file the table does not know counts as the median of the known ones
    w2 = ci_shards.weights(['test_new_one.py', 'test_validation.py'])
    assert w2['test_new_one.py'] == w2['test_validation.py']               # median of one known value
    with pytest.raises(ValueError):
        ci_shards.split(files, 0)


def test_ci_installs_no_cuda_and_every_cuda_test_skips_with_a_reason():
    """The runners have no GPU and CI installs no PyTorch (the extra "splat"), so a test that runs on CUDA must
    skip, with a sentence, where CUDA is not available, and must never be a test that fails there."""
    import ast
    ci = _load('ci.yml')
    assert 'splat' not in ci['env']['EXTRAS'].split(',')
    with open(os.path.join(ROOT, '.github', 'ci-constraints.txt'), encoding='utf-8') as f:
        names = {ln.split('==')[0].strip().lower() for ln in f if '==' in ln.split('#')[0]}
    assert not names & {'torch', 'torchvision', 'lpips', 'gsplat'}
    probes = ('cuda.is_available()', 'gsplat_ok()', 'cuda_ok()')
    found = 0
    for path in sorted(glob.glob(os.path.join(ROOT, 'tests', 'test_*.py'))):
        with open(path, encoding='utf-8') as f:
            text = f.read()
        if 'cuda' not in text:
            continue
        for node in ast.walk(ast.parse(text)):
            if not (isinstance(node, ast.FunctionDef) and node.name.startswith('test_')):
                continue
            src = ast.get_source_segment(text, node)
            if not re.search(r"""device\(\s*['"]cuda|device=['"]cuda""", src):
                continue
            found += 1
            where = f'{os.path.basename(path)}::{node.name}'
            assert 'pytest.skip(' in src and any(p in src for p in probes), f'{where} runs on CUDA without a skip'
            skips = [c for c in ast.walk(node) if isinstance(c, ast.Call) and ast.unparse(c.func) == 'pytest.skip']
            assert all(c.args for c in skips), f'{where} skips without saying why'
    assert found >= 3                                       # the rasteriser's, gsplat's and the fit's CUDA tests


def test_shard_cli(capsys):
    assert ci_shards.main(['--shards', '6', '--index', '0']) == 0
    assert capsys.readouterr().out.split() == [os.path.join('tests', 'test_validation.py')]
    assert ci_shards.main(['--shards', '6', '--index', '6']) == 2
    assert 'between 0 and 5' in capsys.readouterr().err
    assert ci_shards.main(['--shards', '3', '--list']) == 0
    assert capsys.readouterr().out.count('shard ') == 3


# ------------------------------------------------------------------------------------------------ render skip
def test_render_skip_modes(monkeypatch):
    import render_support as RS
    monkeypatch.setenv(RS.ENV, 'skip')
    what, why = RS.check()
    assert what == 'skip' and 'no offscreen renderer' in why
    monkeypatch.setattr(RS, '_PROBE', [(False, 'cannot create an EGL display')])
    monkeypatch.setenv(RS.ENV, 'require')
    what, why = RS.check()
    assert what == 'fail' and 'cannot create an EGL display' in why
    monkeypatch.delenv(RS.ENV)
    what, why = RS.check()
    assert what == 'skip' and 'needs MuJoCo' in why and 'EGL display' in why
    monkeypatch.setattr(RS, '_PROBE', [(True, 'llvmpipe')])
    assert RS.check() == ('run', '')


def test_render_probe_off_linux_says_why(monkeypatch):
    import render_support as RS
    monkeypatch.setattr(RS, '_PROBE', [])
    monkeypatch.setattr(RS.sys, 'platform', 'darwin')
    ok, why = RS.probe()
    assert not ok and 'EGL' in why and 'darwin' in why


def test_the_marker_skips_before_any_fixture(monkeypatch):
    import conftest
    import render_support as RS

    class Item:
        def get_closest_marker(self, name):
            return object() if name == 'mujoco_render' else None
    monkeypatch.setenv(RS.ENV, 'skip')
    with pytest.raises(pytest.skip.Exception, match='no offscreen renderer'):
        conftest.pytest_runtest_setup(Item())
    monkeypatch.setattr(RS, '_PROBE', [(False, 'no EGL')])
    monkeypatch.setenv(RS.ENV, 'require')
    with pytest.raises(pytest.fail.Exception, match='no EGL'):
        conftest.pytest_runtest_setup(Item())


@pytest.mark.mujoco_render
def test_a_marked_test_renders_when_it_runs():
    """Runs only where the conftest hook found a renderer (skipped with the reason otherwise)."""
    import mujoco
    from dexvantage.validation import mjrender as MR
    m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><camera name="c" pos="0 0 1"/>'
                                       '<geom type="sphere" size="0.2"/></worldbody></mujoco>')
    imgs, info = MR.render([{'model': m, 'camera': 'c', 'width': 8, 'height': 8, 'outputs': ('seg',)}])
    assert imgs[0]['seg'].shape == (8, 8) and info['gl'] == 'egl'
