import pytest

from dexvantage.core import jobs
from dexvantage.core.project import Volume, blank_project


def test_voxel_count_matches_the_spec_example():
    # §7.1: a dense 1 mm grid over 1.5 × 1.5 × 1.2 m is 2.7 × 10⁹ voxels
    assert jobs.voxel_count((1.5, 1.5, 1.2), 0.001) == 2_700_000_000
    assert jobs.voxel_count((0.5, 0.6, 0.3), 0.005) == 100 * 120 * 60
    assert jobs.voxel_count((0.0051, 0.01, 0.01), 0.005) == 2 * 2 * 2


def test_summary_flags_infeasible_dense_grids():
    p = blank_project()
    p.workspace.volumes.append(Volume(name='Room', size=(1.5, 1.5, 1.2)))
    doc = jobs.run_study('summary', p)
    room = next(v for v in doc['volumes'] if v['name'] == 'Room')
    assert room['voxels_dense_1mm'] == 2_700_000_000 and not room['dense_1mm_feasible']
    assert room['dense_feasible']                      # 5 mm: 300 × 300 × 240 = 21.6 M
    assert doc['schema'] == 'dexvantage.summary/1'


def test_cache_hits_until_the_project_changes():
    cache = jobs.ResultCache()
    p = blank_project()
    a = jobs.run_study('summary', p, cache=cache)
    b = jobs.run_study('summary', p, cache=cache)
    assert not a['cached'] and b['cached'] and cache.hits == 1
    p.workspace.volumes[0].size = (0.2, 0.2, 0.2)
    c = jobs.run_study('summary', p, cache=cache)
    assert not c['cached']


def test_cancel_stops_early_and_is_not_cached():
    cache = jobs.ResultCache()
    p = blank_project()
    doc = jobs.run_study('summary', p, jobs.Ctx(cancel=lambda: True), cache)
    assert doc['cancelled'] and doc['volumes'] == []
    assert jobs.run_study('summary', p, cache=cache)['cached'] is False


def test_progress_is_reported():
    seen = []
    jobs.run_study('summary', blank_project(), jobs.Ctx(progress=lambda f, t: seen.append(f)))
    assert seen and seen[-1] == pytest.approx(1.0)
