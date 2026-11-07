import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def test_self_test_passes_offscreen():
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    r = subprocess.run([sys.executable, '-m', 'dexvantage', '--self-test'], capture_output=True, text=True, env=env,
                       cwd=ROOT, timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr
    from dexvantage.selftest import NAMES
    for name in NAMES:
        assert f'  {name:<15} ' in r.stdout
    assert 'self-test passed' in r.stdout
