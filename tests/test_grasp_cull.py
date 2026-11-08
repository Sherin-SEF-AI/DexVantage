"""The grasp study's two-level cone cull (core/studies/grasp_visibility.py, _RobotGeom.cone_mask) keeps exactly the
triangles of the per-triangle test, and the "what to move" seat cache returns the seat mask of the pose asked for.
Added with the Phase 11 performance measurement (scripts/benchmarks.py, docs/performance.md)."""
import numpy as np
import pytest

from dexvantage.core.robot.samples import sample_project
from dexvantage.core.studies import grasp_visibility as GV


def _evaluator(name):
    p = sample_project(name)
    return GV.Evaluator(p, GV.settings_of(p))


@pytest.mark.parametrize('name', ['ur5e', 'aloha'])
def test_two_level_cull_equals_the_per_triangle_test(name):
    ev = _evaluator(name)
    rng = np.random.default_rng(11)
    checked = 0
    for g in ev.robots.values():
        m = g.model
        for _ in range(4):
            q = m.clip(m.home + rng.normal(0, 0.4, m.nq))
            T = m.fk_array(q)
            for _ in range(6):
                o = rng.uniform(-0.8, 0.8, 3) + [0, 0, 0.5]
                P = rng.uniform(-0.3, 0.3, (20, 3)) * rng.uniform(0.05, 1.0) + rng.uniform(-0.5, 0.5, 3)
                cone = GV._cone(o, P)
                if cone is None:
                    continue
                for k in g.links:
                    if len(g.local[k]) <= 64:
                        continue
                    Tk = T[k]
                    brute = GV._in_cone(g.tri_centre[k] @ Tk[:3, :3].T + Tk[:3, 3], g.tri_radius[k], cone)
                    assert np.array_equal(g.cone_mask(k, Tk, cone), brute)
                    checked += 1
    assert checked > 20


def test_run_spheres_hold_their_triangles():
    ev = _evaluator('ur5e')
    for g in ev.robots.values():
        for k in g.links:
            c, r = g.tri_centre[k], g.tri_radius[k]
            for i in range(len(g.run_radius[k])):
                sl = slice(i * GV.CULL_CHUNK, (i + 1) * GV.CULL_CHUNK)
                d = np.linalg.norm(c[sl] - g.run_centre[k][i], axis=1) + r[sl]
                assert (d <= g.run_radius[k][i]).all()


def test_seat_cache_returns_the_mask_of_the_pose():
    ev = _evaluator('ur5e')
    j = next(i for i, c in enumerate(ev.cams) if c.on_link)
    c = ev.cams[j]
    for dz in (0.0, 0.02, -0.02, 0.0):
        T = c.T_po.copy()
        T[2, 3] += dz
        got = ev.seat_for(j, T)
        want = ev.seat_mask(c, T)
        assert (got is None and want is None) or np.array_equal(got, want)
    assert len(ev._seats) == 3
