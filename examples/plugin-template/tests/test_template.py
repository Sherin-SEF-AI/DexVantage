"""The template plugin's own tests (headless: no Qt). Run from the plugin's folder: python -m pytest tests.

The study's angle is checked against an independent closed form (atan2 of the perpendicular and parallel parts,
not the arccos of a dot product), with a tolerance of 1e-9 degrees: both are exact in double precision up to rounding.
"""
import math
from importlib.metadata import EntryPoint

import numpy as np
import pytest


def _project(yaw_deg: float, target=(1.0, 0.0, 0.0)):
    """A blank project with one static camera at the origin turned yaw_deg about z, and a task volume at target."""
    from dexvantage.core.frames import Pose, quat_from_rpy
    from dexvantage.core.project import Mount, Sensor, Volume, blank_project
    p = blank_project('Aim test')
    p.workspace.volumes = [Volume(name='Box', center=tuple(target), size=(0.1, 0.1, 0.1))]
    pose = Pose(np.zeros(3), quat_from_rpy(0.0, 0.0, math.radians(yaw_deg)))
    p.variant.sensors = [Sensor(name='Cam', kind='camera', mount=Mount('world', '', pose))]
    return p


@pytest.fixture
def plugin():
    from dexvantage import plugin_sdk as PL
    ep = EntryPoint(name='aim_template', value='dexvantage_plugin_template:register', group=PL.GROUP)
    recs = PL.load(entry_points=[ep])
    yield recs[0]
    PL.unload('aim_template')


def test_angle_against_closed_form():
    from dexvantage_plugin_template.aim import angle_deg
    rng = np.random.default_rng(0)
    for _ in range(200):
        o, a, t = rng.normal(size=3), rng.normal(size=3), rng.normal(size=3)
        d = t - o
        a_hat = a / np.linalg.norm(a)
        par = float(np.dot(d, a_hat))
        perp = float(np.linalg.norm(d - par * a_hat))
        assert abs(angle_deg(o, a, t) - math.degrees(math.atan2(perp, par))) < 1e-9


def test_plugin_loads_and_runs(plugin):
    from dexvantage.core import jobs, metrics
    assert plugin.status == 'loaded', plugin.error
    assert plugin.added['study'] == ['aim_check']
    assert plugin.added['metric'] == ['aim.worst_deg', 'aim.share_aimed']
    assert plugin.added['page'] == ['aim_check']
    doc = jobs.run_study('aim_check', _project(30.0))
    assert abs(doc['summary']['worst_deg'] - 30.0) < 1e-6
    assert doc['summary']['word'] == 'off target'          # 30° > 20° (template default) and <= 40°
    m = metrics.evaluate(_project(10.0), ['aim.worst_deg', 'aim.share_aimed'])
    assert abs(m['aim.worst_deg']['value'] - 10.0) < 1e-6 and m['aim.share_aimed']['value'] == 100.0


def test_apply_aims_the_camera(plugin):
    from dexvantage.core import jobs
    from dexvantage_plugin_template.aim import apply_suggestion
    p = _project(75.0, target=(0.5, 0.4, 0.3))
    doc = jobs.run_study('aim_check', p)
    assert doc['summary']['word'] == 'aimed away' and doc['suggestion']
    assert apply_suggestion(p, doc['suggestion'])
    assert jobs.run_study('aim_check', p)['summary']['worst_deg'] < 1e-6


def test_refuses_without_a_volume(plugin):
    from dexvantage.core import jobs
    p = _project(0.0)
    p.workspace.volumes = []
    with pytest.raises(jobs.StudyError, match='needs a task volume'):
        jobs.run_study('aim_check', p)
