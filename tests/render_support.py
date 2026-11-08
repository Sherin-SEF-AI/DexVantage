"""Whether MuJoCo can render offscreen here, for the tests marked `mujoco_render` (tests/conftest.py skips them, with
this module's reason, when it cannot).

DexVantage renders MuJoCo scenes in a worker process with MUJOCO_GL=egl (validation/mjrender.py): the ground-truth
study, the synthetic episodes (core/sim/episodes.py, which the dataset tests generate their recordings with), the
silhouette calibration, the sim-to-real gap and the rendered validation checks. EGL needs Linux with an OpenGL driver
that offers it: a GPU driver, or Mesa (llvmpipe, software rendering) on a machine without a GPU, as on a CI runner.
Windows and macOS have no EGL, so there these tests are skipped and say why.

DEXVANTAGE_TEST_RENDER (environment) chooses what a marked test does:
    unset or 'auto'   probe once (a 4 × 4 render in the worker); skip with the probe's reason when it fails
    'require'         probe; when it fails, the test fails with the reason (CI on Linux, where Mesa is installed, so
                      the rendered tests can never be skipped there unnoticed)
    'skip'            skip without probing (emulates a machine with no offscreen renderer)
"""
from __future__ import annotations

import os
import sys

ENV = 'DEXVANTAGE_TEST_RENDER'
_PROBE: list = []           # [(ok, reason)] once probed in this process


def probe() -> tuple[bool, str]:
    """(True, renderer string) when a tiny MuJoCo render in the worker succeeds here, else (False, why)."""
    if _PROBE:
        return _PROBE[0]
    if not sys.platform.startswith('linux'):
        res = (False, f'MuJoCo renders through EGL in DexVantage (validation/mjrender.py), which {sys.platform} does '
                      'not have')
    else:
        try:
            import mujoco
            from dexvantage.validation import mjrender as MR
            m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><camera name="c" pos="0 0 1"/>'
                                               '<geom type="sphere" size="0.1"/></worldbody></mujoco>')
            _imgs, info = MR.render([{'model': m, 'camera': 'c', 'width': 4, 'height': 4, 'outputs': ('seg',)}],
                                    timeout=120)
            res = (True, str(info.get('renderer') or 'EGL'))
        except Exception as e:                       # noqa: BLE001 — the reason is the skip message
            res = (False, str(e).strip().splitlines()[0][:400] if str(e).strip() else type(e).__name__)
    _PROBE.append(res)
    return res


def check() -> tuple[str, str]:
    """('run', '') when marked tests can run; ('skip', reason) or ('fail', reason) when they cannot."""
    mode = os.environ.get(ENV, 'auto').strip().lower() or 'auto'
    if mode == 'skip':
        return 'skip', f'{ENV}=skip: running as on a machine with no offscreen renderer'
    ok, why = probe()
    if ok:
        return 'run', ''
    if mode == 'require':
        return 'fail', f'{ENV}=require, but MuJoCo cannot render offscreen here: {why}'
    return 'skip', f'needs MuJoCo\'s offscreen renderer (EGL), which this machine does not offer: {why}'


def require() -> None:
    """Skip (or fail, under 'require') the calling test or fixture when MuJoCo cannot render here. A fixture that
    renders (a synthetic campaign, a ground-truth result) calls it first, so every test using it skips with the
    reason; a test that renders by itself takes the mujoco_render marker instead (tests/conftest.py)."""
    import pytest
    what, why = check()
    if what == 'skip':
        pytest.skip(why)
    if what == 'fail':
        pytest.fail(why, pytrace=False)
