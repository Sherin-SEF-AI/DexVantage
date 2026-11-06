import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dexvantage.core import frames as F


def test_quaternion_round_trip_and_sign():
    rng = np.random.default_rng(1)
    for q in rng.normal(size=(500, 4)):
        q = F.quat_normalize(q)
        assert q[3] >= 0
        assert F.quat_from_matrix(F.quat_to_matrix(q)) == pytest.approx(q, abs=1e-12)


def test_half_turns_take_every_shepperd_branch():
    for axis in np.eye(3):
        R = Rotation.from_rotvec(axis * math.pi).as_matrix()
        assert F.quat_to_matrix(F.quat_from_matrix(R)) == pytest.approx(R, abs=1e-12)


def test_rpy_is_urdf_fixed_axes():
    r, p, y = 0.3, -0.4, 1.2
    ref = Rotation.from_euler('xyz', [r, p, y]).as_matrix()
    assert F.quat_to_matrix(F.quat_from_rpy(r, p, y)) == pytest.approx(ref, abs=1e-12)
    assert F.rpy_from_quat(F.quat_from_rpy(r, p, y)) == pytest.approx((r, p, y), abs=1e-12)


def test_quat_angle_is_precise_for_tiny_rotations():
    q = F.quat_from_axis_angle([0, 0, 1], 1e-9)
    assert F.quat_angle([0, 0, 0, 1], q) == pytest.approx(1e-9, rel=1e-6)


def test_mujoco_order():
    q = F.quat_from_rpy(0.1, 0.2, 0.3)
    m = F.quat_to_mujoco(q)
    assert m[0] == pytest.approx(q[3]) and m[1:] == pytest.approx(q[:3])
    assert F.quat_from_mujoco(m) == pytest.approx(q)


def test_optical_frame_axes():
    body = F.Pose([0.1, 0.2, 0.3], F.quat_from_rpy(0, 0, 0))
    R = F.quat_to_matrix(F.optical_pose(body).quaternion)
    assert R[:, 2] == pytest.approx([1, 0, 0])      # optical z along body x
    assert R[:, 0] == pytest.approx([0, -1, 0])     # optical x along body −y
    assert R[:, 1] == pytest.approx([0, 0, -1])     # optical y along body −z


def test_pose_compose_inverse_and_dict():
    a = F.Pose([1, 2, 3], F.quat_from_rpy(0.3, 0.1, -0.2))
    b = F.Pose([-0.5, 0, 0.25], F.quat_from_rpy(-1, 0.4, 2))
    ab = a @ b
    assert (ab @ ab.inverse()).matrix() == pytest.approx(np.eye(4), abs=1e-12)
    assert F.Pose.from_dict(ab.to_dict()).matrix() == pytest.approx(ab.matrix())


def test_zero_quaternion_is_refused():
    with pytest.raises(ValueError):
        F.quat_normalize([0, 0, 0, 0])
