"""Gaussian scenes and reprojection (core/splat_reproject.py), ported from Vantage's tests/test_splat_reproject.py (the
.ply round trip) and extended.

References: the stereo disparity fx·B/z of two pinhole cameras a baseline B apart (the analytic case of the
reprojection), a distorted lens's pixels back-projected and projected again (core.optics.CameraModel, its own
inverse), rigid transforms against direct matrix products, the INRIA .ply conventions (sigmoid opacity, exp scales,
SH DC colour, rot_0..3 = w, x, y, z) against the writer, and the neighbour shapes of a plane (its normal is the
thinnest axis).
"""
from __future__ import annotations

import numpy as np
import pytest

from dexvantage.core import splat_raster as SR
from dexvantage.core import splat_reproject as RP
from dexvantage.core.frames import Pose, quat_from_axis_angle
from dexvantage.core.optics import CameraModel


def _pose(R=None, t=(0, 0, 0)) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    T[:3, 3] = t
    return T


def test_reprojection_of_a_stereo_pair_is_the_disparity():
    """Camera b is camera a moved B = 0.12 m along a's x axis: a point at depth z seen at u in a is at u − fx·B/z in
    b, the same row, the same depth."""
    cam = CameraModel(640, 480, 500.0, 500.0, 319.5, 239.5)
    B = 0.12
    Ta = _pose(t=(0.3, -0.1, 0.5))
    Tb = Ta @ _pose(t=(B, 0, 0))
    rng = np.random.default_rng(0)
    uv = np.c_[rng.uniform(0, 639, 200), rng.uniform(0, 479, 200)]
    z = rng.uniform(0.3, 3.0, 200)
    uvb, zb, ok = RP.reproject(uv, z, cam, Ta, cam, Tb)
    assert np.allclose(uvb[:, 0], uv[:, 0] - 500.0 * B / z, atol=1e-9)
    assert np.allclose(uvb[:, 1], uv[:, 1], atol=1e-9) and np.allclose(zb, z, atol=1e-12)
    assert ok[(uvb[:, 0] > 0) & (uvb[:, 0] < 639)].all()


def test_reprojection_into_itself_and_through_a_distorted_lens():
    """The same camera and pose: every pixel comes back to itself (Brown-Conrady and Kannala-Brandt lenses); a pure
    rotation about the camera centre does not depend on the depth."""
    for cam in (CameraModel(320, 240, 260, 255, 161.0, 118.0, 'brown', (-0.2, 0.05, 0.001, 0.0005, 0.0)),
                CameraModel(320, 240, 120, 120, 159.5, 119.5, 'kb', (0.03, -0.01, 0.002, 0.0))):
        T = Pose([0.1, 0.2, 0.3], quat_from_axis_angle([0.3, 1.0, 0.2], 0.7)).matrix()
        rng = np.random.default_rng(1)
        uv = np.c_[rng.uniform(20, 300, 100), rng.uniform(20, 220, 100)]
        z = rng.uniform(0.2, 2.0, 100)
        back, zb, ok = RP.reproject(uv, z, cam, T, cam, T)
        assert ok.all() and np.abs(back - uv).max() < 1e-6 and np.allclose(zb, z)
        Tr = T @ _pose(Pose([0, 0, 0], quat_from_axis_angle([0, 1, 0], 0.05)).matrix()[:3, :3])
        u1, _, _ = RP.reproject(uv, z, cam, T, cam, Tr)
        u2, _, _ = RP.reproject(uv, 3 * z, cam, T, cam, Tr)
        assert np.abs(u1 - u2).max() < 1e-6


def test_backproject_lands_on_the_plane():
    """A plane z = 1 m in front of a camera at the world origin looking along +x (optical z = world x): every pixel
    back-projects to x = 1 m."""
    cam = CameraModel(32, 24, 30, 30, 15.5, 11.5, 'kb', (0.01, 0.0, 0.0, 0.0))
    from dexvantage.core.frames import OPTICAL_TO_BODY
    T = _pose(OPTICAL_TO_BODY)                                         # body = world: optical z along world x
    D = np.ones((24, 32))
    D[0, 0] = 0.0                                                      # no hit
    X, rc = RP.backproject(D, cam, T, stride=1)
    assert len(X) == 24 * 32 - 1 and np.allclose(X[:, 0], 1.0)
    assert not ((rc[:, 0] == 0) & (rc[:, 1] == 0)).any()
    X2, _ = RP.backproject(D, cam, T, stride=2, mask=D > 0)
    assert len(X2) == 12 * 16 - 1


def test_rigid_transform_of_gaussians():
    rng = np.random.default_rng(2)
    g = RP.GaussianSet(rng.normal(size=(20, 3)), rng.uniform(0.01, 0.1, (20, 3)), rng.normal(size=(20, 4)),
                       rng.uniform(0, 1, 20), rng.uniform(0, 1, (20, 3)))
    T = Pose([0.5, -0.2, 1.0], quat_from_axis_angle([1, 2, 3], 1.1)).matrix()
    m = g.transformed(T)
    assert np.allclose(m.means, g.means @ T[:3, :3].T + T[:3, 3], atol=1e-6)
    assert np.allclose(m.rotations(), T[:3, :3] @ g.rotations(), atol=1e-6)
    cov0 = SR.covariance(g.rotations(), g.scales.astype(float))
    cov1 = SR.covariance(m.rotations(), m.scales.astype(float))
    assert np.allclose(cov1, T[:3, :3] @ cov0 @ T[:3, :3].T, atol=1e-7)
    per = g.transformed(np.repeat(T[None], 20, 0))
    assert np.allclose(per.means, m.means) and np.allclose(per.quats, m.quats)


def test_shapes_of_a_plane_are_discs_along_it():
    g = np.arange(0, 0.1, 0.005)
    xx, yy = np.meshgrid(g, g)
    P = np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.size)]
    sc, q = RP.shapes_from_points(P, 0.005)
    R = SR.quat_to_rot_np(q)
    thin = np.argmin(sc, axis=1)
    normals = R[np.arange(len(R)), :, thin]
    assert np.allclose(np.abs(normals[:, 2]), 1.0, atol=1e-6)
    assert np.all(sc.min(1) < 0.2 * sc.max(1))


def test_voxel_thinning_fits_the_budget():
    rng = np.random.default_rng(3)
    P = rng.uniform(0, 1, (5000, 3))
    C, vox, mv = RP.thin_to_budget(P, 0.01, 300, values=P[:, :1])
    assert len(C) <= 300 and vox > 0.01 and np.allclose(mv[:, 0], C[:, 0])


def test_ply_round_trip_and_placement(tmp_path):
    rng = np.random.default_rng(4)
    n = 30
    g = RP.GaussianSet(rng.normal(size=(n, 3)), rng.uniform(0.01, 0.2, (n, 3)), rng.normal(size=(n, 4)),
                       np.linspace(0.2, 0.9, n), rng.uniform(0.05, 0.95, (n, 3)))
    path = tmp_path / 'scene.ply'
    path.write_bytes(RP.ply_bytes(g))
    v = RP.read_ply(str(path))
    assert len([k for k in v if k.startswith('f_rest_')]) == 45                  # the INRIA trainer's degree-3 layout
    assert v['opacity'][0] == pytest.approx(np.log(0.2 / 0.8), abs=1e-5)         # a logit
    assert np.allclose(np.c_[v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], g.quats[:, [3, 0, 1, 2]])  # w first
    back = RP.gaussians_from_ply(str(path))
    assert np.allclose(back.means, g.means, atol=1e-6) and np.allclose(back.colours, g.colours, atol=1e-5)
    assert np.allclose(back.opacity, g.opacity, atol=1e-5) and np.allclose(back.scales, g.scales, rtol=1e-5)
    assert np.allclose(SR.quat_to_rot_np(back.quats), g.rotations(), atol=1e-5)
    T = Pose([5.0, 0.0, 0.0], quat_from_axis_angle([0, 0, 1], np.pi / 2)).matrix()
    moved = RP.gaussians_from_ply(str(path), T, scale=2.0)
    assert moved.means[0] == pytest.approx([5.0 - 2 * g.means[0, 1], 2 * g.means[0, 0], 2 * g.means[0, 2]], abs=1e-5)
    assert moved.scales[0] == pytest.approx(2 * g.scales[0], rel=1e-5)
    bad = tmp_path / 'bad.ply'
    bad.write_bytes(b'not a ply\n')
    with pytest.raises(ValueError, match='PLY'):
        RP.read_ply(str(bad))


def test_numpy_render_of_a_wall_has_its_depth():
    """Flat Gaussians tiling the plane z = 2 m in front of a pinhole: the expected range at the centre is 2 m, the
    wall covers the image, the colour is the wall's."""
    g = np.arange(-1.5, 1.5 + 1e-9, 0.05)
    xx, yy = np.meshgrid(g, g)
    n = xx.size
    gs = RP.GaussianSet(np.c_[xx.ravel(), yy.ravel(), np.full(n, 2.0)], np.tile([0.03, 0.03, 0.002], (n, 1)),
                        np.tile([0, 0, 0, 1.0], (n, 1)), np.full(n, 0.95), np.tile([0.2, 0.6, 0.4], (n, 1)))
    cam = CameraModel(24, 18, 20, 20, 11.5, 8.5)
    out = RP.render(gs, cam, np.eye(4), backend='numpy', depth=True)
    assert out['alpha'].min() > 0.99
    assert out['depth'][9, 12] == pytest.approx(2.0, abs=0.01)
    assert np.allclose(out['rgb'][9, 12], [0.2, 0.6, 0.4], atol=0.01)
