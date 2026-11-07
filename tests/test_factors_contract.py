"""The factor contract (core/factors.py) shared by tolerances, experiments, the optimiser and the API."""
import math

import numpy as np
import pytest

from dexvantage.core import factors as F
from dexvantage.core.project import Project, Sensor


def _proj():
    p = Project()
    p.variant.sensors.append(Sensor(name='Wrist'))
    p.variant.sensors.append(Sensor(name='Top', locked=True))
    return p


def test_get_apply_round_trip_and_never_mutates():
    p = _proj()
    q = F.apply(p, {'sensor:Wrist.x': 0.3, 'sensor:Wrist.yaw': 30.0, 'setting:coverage.voxel_mm': 4})
    assert F.get(p, 'sensor:Wrist.x') == 0.0 and F.get(q, 'sensor:Wrist.x') == pytest.approx(0.3)
    assert F.get(q, 'sensor:Wrist.yaw') == pytest.approx(30.0)
    assert F.get(q, 'setting:coverage.voxel_mm') == 4 and F.get(p, 'setting:coverage.voxel_mm') is None
    r = F.apply(q, {'sensor:Wrist.dx': 0.1, 'sensor:Wrist.droll': 90.0})
    assert F.get(r, 'sensor:Wrist.x') == pytest.approx(0.4)
    s = next(s for s in r.variant.sensors if s.name == 'Wrist')
    R = s.mount.pose.matrix()[:3, :3]
    # a body-frame roll keeps the body x axis where yaw 30° put it
    assert np.allclose(R[:, 0], [math.cos(math.radians(30)), math.sin(math.radians(30)), 0], atol=1e-9)


def test_locked_and_bad_paths_are_plain_sentences():
    p = _proj()
    with pytest.raises(F.FactorError, match='locked'):
        F.apply(p, {'sensor:Top.z': 1.0})
    assert F.get(F.apply(p, {'sensor:Top.z': 1.0}, allow_locked=True), 'sensor:Top.z') == 1.0
    for bad in ('nope', 'sensor:Nobody.x', 'sensor:Wrist.colour', 'setting:x', 'json:a'):
        with pytest.raises(F.FactorError):
            F.get(p, bad)
    assert F.describe('sensor:Wrist.pitch', p)['unit'] == '°'
    assert F.describe('sensor:Top.model', p)['locked'] is True


def test_json_pointer_fallback():
    p = _proj()
    q = F.apply(p, {'json:/name': 'Renamed'})
    assert q.name == 'Renamed' and p.name != 'Renamed'
    assert F.get(q, 'json:/name') == 'Renamed'
