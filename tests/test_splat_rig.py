"""Robot-aligned 3D Gaussian Splatting (core/sim/splat_rig.py) on the bundled SO-100 and UR5e samples.

References: a re-posed link's Gaussians against the link's forward kinematics (exact: T(q₂) T(q₁)⁻¹ applied to the
Gaussians at q₁, and the independent product-of-exponentials chain of core/robot/kinematics.py for the link poses),
per-link segmentation on a synthetic scene whose labels are known by construction (Gaussians sampled on each link's
visual mesh and on a plane away from the robot; a MuJoCo capture whose per-pixel owner segmentation gives every
starting Gaussian its true link), the sidecar format against its own reader. The capture for the fitting tests is
rendered by MuJoCo (synthetic) at 32 × 24 px with a few hundred Gaussians, so it runs on the CPU; the CUDA fit skips,
with the reason, where there is no CUDA.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from dexvantage.core import splat_reproject as RP
from dexvantage.core.frames import Pose, quat_from_axis_angle
from dexvantage.core.jobs import StudyError
from dexvantage.core.optics import CameraModel
from dexvantage.core.robot.kinematics import PoEChain
from dexvantage.core.robot.model import get_model
from dexvantage.core.robot.samples import sample_project
from dexvantage.core.sim import splat_rig as SRG

NO_TORCH = 'PyTorch is not installed (the splat extra): fitting is not tested'
NO_CUDA = 'no CUDA device here: the CUDA fit is not tested'


@pytest.fixture(scope='module')
def so100():
    return sample_project('so100')


def _mesh_rig(p, per_link=60, seed=0, world=40) -> SRG.SplatRig:
    """A rig whose robot Gaussians are sampled on each link's visual mesh (in the link frame: labels known) plus
    world Gaussians on a plane 0.5 m below the robot's base."""
    uid = p.robots[0].uid
    m = get_model(p, uid)
    links = [(uid, ln) for ln in m.link_names]
    P, F = [], []
    for k, (_u, ln) in enumerate(links):
        S, _ = SRG._mesh_samples(m, ln, per_link, seed + k)
        if len(S):
            P.append(S)
            F.append(np.full(len(S), k))
    rng = np.random.default_rng(seed)
    P.append(np.c_[rng.uniform(-0.5, 0.5, (world, 2)), np.full(world, -0.5)])
    F.append(np.full(world, -1))
    P, F = np.concatenate(P), np.concatenate(F)
    n = len(P)
    g = RP.GaussianSet(P, np.full((n, 3), 0.004), rng.normal(size=(n, 4)), np.full(n, 0.8), rng.uniform(0, 1, (n, 3)))
    return SRG.SplatRig(g, F, links, {uid: m.home.copy()}, [0.5, 0.5, 0.5],
                        {uid: {'name': p.robots[0].name, 'digest': SRG.robot_digest(p, uid)}}, True)


# ----------------------------------------------------------------------------- forward kinematics
@pytest.mark.parametrize('name', ['so100', 'ur5e'])
def test_reposed_link_gaussians_move_rigidly_with_fk(name):
    p = sample_project(name)
    rig = _mesh_rig(p)
    uid = p.robots[0].uid
    m = get_model(p, uid)
    q1 = m.clip(m.home + 0.3)
    q2 = m.clip(m.home - 0.4)
    g1, g2 = rig.posed(p, {uid: q1}), rig.posed(p, {uid: q2})
    T1, T2 = rig.link_transforms(p, {uid: q1}), rig.link_transforms(p, {uid: q2})
    chain = PoEChain(p.robot(uid), p.sources)
    for k, (_u, ln) in enumerate(rig.links):
        sel = rig.frame == k
        if not sel.any():
            continue
        M = T2[k] @ np.linalg.inv(T1[k])
        assert np.abs(g2.means[sel] - (g1.means[sel].astype(float) @ M[:3, :3].T + M[:3, 3])).max() < 1e-6
        assert np.abs(g2.rotations()[sel] - M[:3, :3] @ g1.rotations()[sel]).max() < 1e-5
        d1 = np.linalg.norm(g1.means[sel][:, None] - g1.means[sel][None], axis=-1)
        d2 = np.linalg.norm(g2.means[sel][:, None] - g2.means[sel][None], axis=-1)
        assert np.abs(d1 - d2).max() < 1e-6                                   # rigid: distances kept
        if ln in chain.link_names:                                            # the independent FK agrees
            assert np.abs(chain.fk(q2)[ln] - T2[k]).max() < 1e-9
    w = rig.frame < 0
    assert np.array_equal(g1.means[w], g2.means[w]) and np.array_equal(g1.means[w], rig.gaussians.means[w])


def test_robot_change_is_refused(so100):
    rig = _mesh_rig(so100)
    rig.check_robots(so100)
    rig.robots[so100.robots[0].uid]['digest'] = 'f' * 64
    with pytest.raises(StudyError, match='changed since the splat rig was fitted'):
        rig.check_robots(so100)


# ----------------------------------------------------------------------------- segmentation
def test_segmentation_by_meshes_on_a_synthetic_scene(so100):
    """Gaussians on each link's mesh and on a plane away from the robot: each gets its true frame except where two
    links' meshes meet (there both are right within a millimetre), also after a third of them were moved into wrong
    frames (relabel keeps their pose at q_ref)."""
    rig = _mesh_rig(so100, per_link=80)
    truth = rig.frame.copy()
    lab, rep = SRG.segment_links(rig, so100, None, 'meshes', max_dist_m=0.005, mesh_spacing_m=0.001)
    r = SRG.mislabel_rate(lab, truth, rig.links)
    assert rep['by_meshes'] == rig.n and r['rate'] <= 0.02, r
    # the few it gets wrong lie where two links' meshes meet: within 1 mm of both (measured 8 of 600, all < 0.8 mm)
    from scipy.spatial import cKDTree
    S, F = SRG.link_surface_samples(rig, so100, rig.q_ref, 0.001)
    P = rig.posed(so100, rig.q_ref).means
    for i in np.flatnonzero(lab != truth):
        assert lab[i] >= 0 and truth[i] >= 0
        assert cKDTree(S[F == lab[i]]).query(P[i])[0] < 1e-3 and cKDTree(S[F == truth[i]]).query(P[i])[0] < 1e-3
    rng = np.random.default_rng(5)
    wrong = truth.copy()
    flip = rng.random(rig.n) < 0.33
    wrong[flip] = (truth[flip] + 2) % len(rig.links)
    before = rig.posed(so100, rig.q_ref).means.copy()
    assert SRG.relabel(rig, so100, wrong) == int((wrong != truth).sum())
    assert np.abs(rig.posed(so100, rig.q_ref).means - before).max() < 1e-6     # same pose at the reference q
    lab, _ = SRG.segment_links(rig, so100, None, 'meshes', max_dist_m=0.005, mesh_spacing_m=0.001)
    assert SRG.mislabel_rate(lab, truth)['rate'] <= 0.02


def test_static_frames_and_moving_rate(so100):
    rig = _mesh_rig(so100)
    st = SRG.static_frames(rig, so100)
    assert [rig.links[k][1] for k in st] == ['Base']
    truth = rig.frame.copy()
    pred = truth.copy()
    base = rig.links.index((so100.robots[0].uid, 'Base'))
    pred[truth == base] = -1                                                  # base Gaussians called world
    r = SRG.mislabel_rate(pred, truth, rig.links, st)
    assert r['wrong'] == int((truth == base).sum()) and r['moving_rate'] == 0.0


@pytest.fixture(scope='module')
def capture(so100):
    from render_support import require
    require()
    qs = SRG.random_qs(so100, 2, seed=3)
    return SRG.synthetic_capture(so100, n_views=10, qs=qs, width=32, height=24, seed=1)


@pytest.mark.mujoco_render
def test_synthetic_capture_is_marked_and_posed(so100, capture):
    assert capture.synthetic and len(capture.views) == 10 and capture.source['robot_uid'] == so100.robots[0].uid
    v = capture.views[0]
    assert v.rgb.shape == (24, 32, 3) and v.rgb.dtype == np.uint8 and v.depth.shape == (24, 32)
    assert (v.depth > 0).mean() > 0.2
    robot = [i for i, o in enumerate(capture.owners) if o[0] == 'robot']
    assert np.isin(v.seg, robot).any()
    # every depth pixel back-projects inside the robot's box or onto the table, through the view's own pose
    X, _ = RP.backproject(v.depth, v.camera, v.T_world_optical)
    assert np.isfinite(X).all() and np.abs(X).max() < 3.0


@pytest.mark.mujoco_render
def test_init_labels_from_the_masks_are_recovered(so100, capture):
    """The starting labels come from MuJoCo's per-pixel owners (true by construction); flip 20 % of them and
    segmentation 'both' gives back at least 90 % of the flipped ones, 'masks' alone does not change the true ones
    by more than 2 %."""
    rig = SRG.init_rig(so100, capture, SRG.FitSettings(max_gaussians=600))
    truth = rig.frame.copy()
    assert (truth >= 0).any() and (truth < 0).any()
    lab, _ = SRG.segment_links(rig, so100, capture, 'masks')
    assert SRG.mislabel_rate(lab, truth)['rate'] <= 0.02
    rng = np.random.default_rng(9)
    flip = rng.random(rig.n) < 0.2
    wrong = truth.copy()
    wrong[flip] = np.where(truth[flip] < 0, 3, -1)
    SRG.relabel(rig, so100, wrong)
    lab, _ = SRG.segment_links(rig, so100, capture, 'both')
    assert (lab[flip] == truth[flip]).mean() >= 0.9


# ----------------------------------------------------------------------------- files
def test_save_and_load_round_trip(so100, tmp_path):
    rig = _mesh_rig(so100)
    rig.info = {'fit': {'iterations': 3}}
    paths = SRG.SplatRig.save(rig, tmp_path / 'cell.splat')
    assert paths[0].endswith('cell.splat.npz') and paths[1].endswith('cell.splat.json')
    side = json.loads((tmp_path / 'cell.splat.json').read_text())
    assert side['schema'] == SRG.SCHEMA and side['synthetic'] is True and side['quaternion_order'] == 'x, y, z, w'
    assert 'synthetic' in side['renders'] and side['links'][0] == list(rig.links[0])
    back = SRG.SplatRig.load(paths[0])
    assert back.links == rig.links and np.array_equal(back.frame, rig.frame) and back.synthetic
    assert np.array_equal(back.gaussians.means, rig.gaussians.means) and back.info['fit']['iterations'] == 3
    uid = so100.robots[0].uid
    q = {uid: get_model(so100, uid).home + 0.2}
    assert np.allclose(back.posed(so100, q).means, rig.posed(so100, q).means)
    ply = rig.ply_bytes(so100, q)
    (tmp_path / 'p.ply').write_bytes(ply)
    assert RP.read_ply(str(tmp_path / 'p.ply'))['x'].shape == (rig.n,)
    (tmp_path / 'bad.json').write_text('{"schema": "other"}')
    np.savez(tmp_path / 'bad.npz', means=np.zeros((1, 3)))
    with pytest.raises(ValueError, match='not a DexVantage splat rig'):
        SRG.SplatRig.load(tmp_path / 'bad.npz')


def test_render_from_a_pose_and_through_a_sensor(so100):
    """The numpy back end (no torch needed): a render is marked synthetic; a sensor's render moves with its mount
    when the extrinsic is perturbed (a §8.5 knock of 2°)."""
    rig = _mesh_rig(so100, per_link=40)
    cam = CameraModel(24, 18, 22, 22, 11.5, 8.5)
    T = SRG.look_at([0.6, -0.3, 0.4], [0.0, -0.1, 0.12])
    out = rig.render(so100, cam, T, None, backend='numpy')
    assert out['synthetic'] and out['rgb'].shape == (18, 24, 3) and out['alpha'].max() > 0.5
    s = next(s for s in so100.variant.sensors if not s.mount.on_link)
    a = rig.render_sensor(so100, s.uid, None, scale=0.05, backend='numpy')
    knock = Pose([0, 0, 0], quat_from_axis_angle([0, 0, 1], np.radians(2.0)))
    b = rig.render_sensor(so100, s.uid, None, extrinsic_delta=knock, scale=0.05, backend='numpy')
    assert not np.allclose(a['T_world_optical'], b['T_world_optical'])
    assert a['camera'].width == b['camera'].width
    with pytest.raises(StudyError, match='no sensor'):
        rig.render_sensor(so100, 'nope')


# ----------------------------------------------------------------------------- fitting
def _fit_settings(**kw) -> SRG.FitSettings:
    return SRG.FitSettings(**{'iterations': 12, 'device': 'cpu', 'max_gaussians': 300, 'holdout_every': 5,
                              'dtype': 'float64', **kw})


@pytest.mark.mujoco_render
def test_fit_on_the_cpu_is_deterministic_and_improves(so100, capture):
    pytest.importorskip('torch', reason=NO_TORCH)
    seen = []
    rig, rep = SRG.fit_splat_rig(so100, capture, _fit_settings(), progress=lambda f, t: seen.append((f, t)))
    assert rep['iterations'] == 12 and rep['device'] == 'cpu' and rep['deterministic'] and not rep['cancelled']
    assert rep['held_out_views'] == 2 and rep['held_out_after']['psnr_db'] > rep['held_out_before']['psnr_db']
    assert len(rep['loss_curve']) == 2 and rep['lpips'] is None and 'LPIPS' in rep['lpips_note']
    assert rep['segmentation']['method'] == 'masks' and rig.synthetic and rig.info['capture']['synthetic']
    assert seen and seen[-1][0] == 1.0 and all(0.0 <= f <= 1.0 for f, _ in seen)
    rig2, rep2 = SRG.fit_splat_rig(so100, capture, _fit_settings())
    assert np.array_equal(rig.gaussians.means, rig2.gaussians.means) and rep['loss_curve'] == rep2['loss_curve']


@pytest.mark.mujoco_render
def test_fit_cancels_and_refuses_plainly(so100, capture, monkeypatch):
    pytest.importorskip('torch', reason=NO_TORCH)
    calls = []

    def cancel():
        calls.append(1)
        return len(calls) > 3
    rig, rep = SRG.fit_splat_rig(so100, capture, _fit_settings(iterations=50), cancel=cancel)
    assert rep['cancelled'] and rep['stopped_by'] == 'cancelled' and rep['iterations'] < 50 and rig.n > 0
    with pytest.raises(StudyError, match='Every view is held out'):
        SRG.fit_splat_rig(so100, capture, _fit_settings(holdout_every=1))
    with pytest.raises(StudyError, match='no views'):
        SRG.fit_splat_rig(so100, SRG.Capture([], [], True), _fit_settings())
    from dexvantage.core import splat_raster as SR
    monkeypatch.setattr(SR, 'torch_ok', lambda: False)
    with pytest.raises(StudyError, match='needs PyTorch'):
        SRG.fit_splat_rig(so100, capture, _fit_settings())


@pytest.mark.mujoco_render
def test_fit_on_cuda(so100, capture):
    torch = pytest.importorskip('torch', reason=NO_TORCH)
    if not torch.cuda.is_available():
        pytest.skip(NO_CUDA)
    rig, rep = SRG.fit_splat_rig(so100, capture, _fit_settings(device='cuda', dtype='float32', iterations=30))
    assert rep['device'].startswith('cuda') and not rep['deterministic']
    assert rep['held_out_after']['psnr_db'] > rep['held_out_before']['psnr_db']
