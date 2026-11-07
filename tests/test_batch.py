import json
import os
import subprocess
import sys

import pytest

from dexvantage.core.project import blank_project

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def _run(*args, cwd=None):
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    return subprocess.run([sys.executable, '-m', 'dexvantage', *args], capture_output=True, text=True, env=env,
                          cwd=cwd or ROOT, timeout=120)


@pytest.fixture
def proj(tmp_path):
    p = blank_project('Batch cell')
    path = str(tmp_path / 'cell.dexv')
    p.save(path)
    return path


def test_analyse_writes_json_with_provenance(tmp_path, proj):
    out = tmp_path / 'summary.json'
    r = _run(proj, '--analyse', str(out))
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text())
    assert doc['volumes'][0]['voxels_dense'] == 720_000
    prov = doc['provenance']
    assert prov['interface'] == 'batch' and prov['command'][:2] == ['dexvantage', 'cell.dexv']
    assert str(tmp_path) not in out.read_text()


def test_reproduce_and_tampering(tmp_path, proj):
    out = tmp_path / 'summary.json'
    assert _run(proj, '--analyse', str(out)).returncode == 0
    r = _run(proj, '--reproduce', str(out))
    assert r.returncode == 0 and 'Reproduces exactly' in r.stdout, r.stdout + r.stderr
    doc = json.loads(out.read_text())
    doc['volumes'][0]['voxels_dense'] += 1
    out.write_text(json.dumps(doc))
    r = _run(proj, '--reproduce', str(out))
    assert r.returncode == 1 and 'changed after DexVantage wrote it' in r.stdout


def test_validate_exit_code_and_report(tmp_path):
    r = _run('--validate', str(tmp_path / 'v'), '--quick', '--models', 'frames,optics,library')   # the exit code
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / 'v' / 'validation.html').exists()
    assert _run('--validate', str(tmp_path / 'w'), '--models', 'nope').returncode == 2


def test_errors_are_explained_not_tracebacks(tmp_path):
    bad = tmp_path / 'bad.dexv'
    bad.write_text('hello')
    r = _run(str(bad), '--analyse', '-')
    assert r.returncode == 2 and 'not a DexVantage project' in r.stderr and 'Traceback' not in r.stderr
