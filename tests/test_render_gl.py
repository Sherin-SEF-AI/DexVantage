"""The shader renderer (ported from Vantage) draws in a headless EGL context: a lit box, shadows darker than
without, an empty scene empty. Skipped, with the reason, on a machine without EGL OpenGL."""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope='module')
def drawn():
    r = subprocess.run([sys.executable, os.path.join(HERE, 'gl_harness.py')], capture_output=True, text=True,
                       timeout=120)
    line = (r.stdout.strip().splitlines() or ['{}'])[-1]
    out = json.loads(line) if line.startswith('{') else {'error': r.stderr[-400:]}
    if 'skip' in out:
        pytest.skip(f'no headless OpenGL here: {out["skip"]}')
    return out


def test_shader_renderer_draws(drawn):
    assert drawn.get('init'), drawn
    assert drawn['shape'] == [260, 400, 4]
    assert drawn['box_px'] > 0.01 and drawn['gl_error'] == 0


def test_shadows_darken_and_empty_is_empty(drawn):
    assert drawn['shadow_darkens'] > 0.0
    assert drawn['empty_max'] < 40
