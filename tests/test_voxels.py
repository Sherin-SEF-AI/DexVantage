"""core/voxels.py: base counts against core.jobs.voxel_count, refinement in regions and around objects, chunking,
and the plain-English limit message of spec §7.1."""
import numpy as np
import pytest

from dexvantage.core import jobs
from dexvantage.core import voxels as VX
from dexvantage.core.frames import Pose
from dexvantage.core.project import TaskObject, Volume, blank_project


def _project(*vols):
    p = blank_project()
    p.workspace.volumes = list(vols)
    return p


def test_base_grid_counts_and_extent():
    v = Volume(center=(0.4, 0.0, 0.15), size=(0.5, 0.4, 0.3))
    p = _project(v)
    g = VX.SparseGrid.from_project(p)
    assert g.count == jobs.voxel_count(v.size, 0.005) == 100 * 80 * 60
    assert np.allclose(g.size, 0.005) and not g.refined.any() and (g.volume_index == 0).all()
    lo, hi = v.bounds()
    assert (g.centers >= lo - 1e-6).all() and (g.centers <= hi + 1e-6).all()
    assert g.volume_m3.sum() == pytest.approx(0.5 * 0.4 * 0.3, rel=1e-5)
    assert len(np.unique(np.round(g.centers / 0.0025).astype(np.int64), axis=0)) == g.count     # no duplicates
    est = VX.estimate(p)
    assert est['count'] == g.count and est['feasible'] and est['voxel_mm'] == 5.0


def test_non_multiple_edge_overhangs_evenly():
    v = Volume(center=(0.0, 0.0, 0.0), size=(0.012, 0.01, 0.01))
    g = VX.SparseGrid.from_project(_project(v))
    assert g.count == 3 * 2 * 2
    assert np.allclose(np.sort(np.unique(g.centers[:, 0])), [-0.005, 0.0, 0.005], atol=1e-7)   # centred


def test_refinement_region():
    v = Volume(center=(0, 0, 0), size=(0.1, 0.1, 0.05), refine=True)
    p = _project(v)
    g = VX.SparseGrid.from_project(p, voxel_mm=5, refine_mm=2)
    k = 3                                                  # ceil(5 / 2): 1.67 mm voxels
    assert g.count == 20 * 20 * 10 * k ** 3 and g.refined.all()
    assert np.allclose(g.size, 0.005 / k)
    assert g.volume_m3.sum() == pytest.approx(0.1 * 0.1 * 0.05, rel=1e-5)
    assert VX.estimate(p, 5, 2)['refine_mm'] == pytest.approx(5 / 3)
    assert VX.SparseGrid.from_project(p, voxel_mm=4, refine_mm=2).size.min() == pytest.approx(0.002)


def test_refinement_around_objects():
    v = Volume(center=(0, 0, 0.1), size=(0.3, 0.3, 0.2))
    p = _project(v)
    p.workspace.objects.append(TaskObject(shape='box', size=(0.04, 0.04, 0.04), pose=Pose([0.05, 0.0, 0.1])))
    g = VX.SparseGrid.from_project(p)
    est = VX.estimate(p)
    assert g.count == est['count']
    fine = g.centers[g.refined]
    assert len(fine) and np.allclose(g.size[g.refined], 0.005 / 3)
    lo = np.array([0.03, -0.02, 0.08]) - VX.REFINE_MARGIN_M - 0.005
    hi = np.array([0.07, 0.02, 0.12]) + VX.REFINE_MARGIN_M + 0.005
    assert (fine >= lo).all() and (fine <= hi).all()       # only the object and its margin are refined
    assert g.volume_m3.sum() == pytest.approx(0.3 * 0.3 * 0.2, rel=1e-5)
    nref = est['volumes'][0]['refined_cells']
    assert g.count == est['volumes'][0]['base'] - nref + nref * 27
    assert 'refined to 1.7 mm' in est['message']


def test_estimate_message_dense_1mm():
    """Spec §7.1: 'A dense 1 mm grid over 1.5 × 1.5 × 1.2 m is 2.7 × 10⁹ voxels', stated, and the grid refused."""
    p = _project(Volume(center=(0, 0, 0.6), size=(1.5, 1.5, 1.2)))
    est = VX.estimate(p, voxel_mm=1)
    assert not est['feasible'] and est['count'] == 1500 * 1500 * 1200 > jobs.DENSE_LIMIT
    assert est['message'].startswith('A dense 1 mm grid over 1.5 × 1.5 × 1.2 m is 2.7 × 10⁹ voxels')
    assert 'not built' in est['message']
    with pytest.raises(jobs.StudyError) as e:
        VX.SparseGrid.from_project(p, voxel_mm=1)
    assert '2.7 × 10⁹' in str(e.value)
    ok = VX.estimate(p)                                    # 5 mm fits; the dense refined grid would not
    assert ok['feasible'] and 'over the limit' in ok['message'] and ok['dense_count'] > jobs.DENSE_LIMIT


def test_chunks_cover_everything():
    g = VX.SparseGrid.from_project(_project(Volume(size=(0.1, 0.1, 0.1)), Volume(size=(0.05, 0.05, 0.05))))
    seen = 0
    for ch in g.chunks(333):
        assert ch.start == seen and len(ch.centers) == ch.stop - ch.start <= 333
        seen = ch.stop
    assert seen == g.count == 20 ** 3 + 10 ** 3
    assert (g.volume_index == 1).sum() == 10 ** 3 and len(g.volumes) == 2


def test_no_volumes_and_bad_sizes():
    p = _project()
    assert VX.SparseGrid.from_project(p).count == 0
    assert 'no task volume' in VX.estimate(p)['message']
    with pytest.raises(jobs.StudyError):
        VX.estimate(_project(Volume()), voxel_mm=0)
    with pytest.raises(jobs.StudyError):
        VX.estimate(_project(Volume()), refine_mm=float('nan'))


def test_sci():
    assert VX.sci(2.7e9) == '2.7 × 10⁹'
    assert VX.sci(5e7) == '5 × 10⁷'
    assert VX.sci(480000) == '480,000'
    assert VX.sci(9.96e6) == '1 × 10⁷'
