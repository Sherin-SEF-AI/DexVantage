"""The example notebooks (examples/*.py, jupytext percent format) run headless on the samples with small settings, and
their cells are well formed."""
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
EXAMPLES = os.path.join(ROOT, 'examples')
NOTEBOOKS = ('01_rig_studies_factors.py', '02_datasets_audit.py')


def _run(name, tmp_path):
    env = dict(os.environ)
    for k in ('XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_CACHE_HOME'):
        env[k] = str(tmp_path / k.lower())
    env['DEXVANTAGE_EXAMPLE_OUT'] = str(tmp_path / 'out')
    env['PYTHONPATH'] = ROOT + os.pathsep + env.get('PYTHONPATH', '')
    env.pop('QT_QPA_PLATFORM', None)
    out = subprocess.run([sys.executable, os.path.join(EXAMPLES, name)], cwd=str(tmp_path), env=env,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-3000:]
    return out.stdout


@pytest.mark.parametrize('name', NOTEBOOKS)
def test_percent_format(name):
    text = open(os.path.join(EXAMPLES, name), encoding='utf-8').read()
    assert text.startswith('# ---\n# jupyter:\n#   jupytext:\n') and 'format_name: percent' in text
    cells = re.findall(r'^# %%( \[markdown\])?$', text, flags=re.M)
    assert sum(1 for c in cells if c) >= 3 and sum(1 for c in cells if not c) >= 3
    compile(text, name, 'exec')


def test_rig_notebook_runs(tmp_path):
    out = _run('01_rig_studies_factors.py', tmp_path)
    assert re.search(r'^Task volume seen: [0-9.]+ % \(', out, flags=re.M), out
    assert re.search(r'^(ready|not ready) · Task volume seen: ', out, flags=re.M), out
    assert re.search(r"^ok \['calibration/", out, flags=re.M), out
    assert 'ur5e_higher_overhead.dexv' in out
    assert os.path.isfile(tmp_path / 'out' / 'ur5e_higher_overhead.dexv')
    assert os.path.isfile(tmp_path / 'out' / 'export' / 'export_manifest.json')


def test_datasets_notebook_runs(tmp_path):
    out = _run('02_datasets_audit.py', tmp_path)
    if 'Could not render the episodes:' in out:
        pytest.skip('MuJoCo\'s offscreen renderer is not available here: ' + out.strip().splitlines()[-1])
    assert '2 episodes,' in out and "<Recording LeRobot v3.0 'so100_synthetic': 2 episodes>" in out
    assert '24 rows in episode 0' in out
    assert re.search(r'^Episodes that pass the audit: [0-9.]+ % \(', out, flags=re.M), out
