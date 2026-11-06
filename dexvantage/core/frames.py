"""Frames, poses and quaternions.

Conventions (docs/overview.md, Help ▸ Coordinate frames):

- Units are SI: metres, seconds, kilograms. Joint angles are stored in radians (URDF and MJCF native) and shown in
  degrees with the unit.
- The world frame is the base frame of the project's primary robot, REP-103: x forward, y left, z up.
- A camera's optical frame is REP-103's `_optical_frame`: z forward (the optical axis), x right, y down.
- Rotations are stored as unit quaternions in the order (x, y, z, w), as ROS geometry_msgs and tf2 write them.
  The scalar-last order is the same as scipy's Rotation.from_quat default; MuJoCo writes (w, x, y, z), so the MJCF
  reader and writer convert at the boundary (quat_to_mujoco, quat_from_mujoco).
- A pose is a rigid transform from a child frame to its parent: p_parent = R @ p_child + t.

The validation suite checks these conversions against scipy.spatial.transform.Rotation (an independent
implementation) and the optical-frame rotation against REP-103's written definition.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# body (x forward, y left, z up) -> optical (x right, y down, z forward), REP-103: a point's optical coordinates are
# BODY_TO_OPTICAL @ its body coordinates. The camera link's x axis is the optical z axis.
BODY_TO_OPTICAL = np.array([[0.0, -1.0, 0.0],
                            [0.0, 0.0, -1.0],
                            [1.0, 0.0, 0.0]])
OPTICAL_TO_BODY = BODY_TO_OPTICAL.T


def quat_normalize(q) -> np.ndarray:
    q = np.asarray(q, float).reshape(4)
    n = float(np.linalg.norm(q))
    if not math.isfinite(n) or n < 1e-12:
        raise ValueError('a quaternion of zero length is not a rotation')
    q = q / n
    return -q if q[3] < 0 else q            # one of the two signs: w >= 0, so equal rotations compare equal


def quat_to_matrix(q) -> np.ndarray:
    """Rotation matrix of a unit quaternion (x, y, z, w)."""
    x, y, z, w = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_from_matrix(R) -> np.ndarray:
    """Unit quaternion (x, y, z, w), w >= 0, of a rotation matrix (Shepperd's method: the largest diagonal term)."""
    m = np.asarray(R, float)
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    return quat_normalize(q)


def quat_multiply(a, b) -> np.ndarray:
    """The rotation a then applied after b (R(a) @ R(b)), both (x, y, z, w)."""
    ax, ay, az, aw = np.asarray(a, float)
    bx, by, bz, bw = np.asarray(b, float)
    return quat_normalize([aw * bx + ax * bw + ay * bz - az * by,
                           aw * by - ax * bz + ay * bw + az * bx,
                           aw * bz + ax * by - ay * bx + az * bw,
                           aw * bw - ax * bx - ay * by - az * bz])


def quat_from_axis_angle(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, float)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    s = math.sin(angle / 2) / n
    return quat_normalize([a[0] * s, a[1] * s, a[2] * s, math.cos(angle / 2)])


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF's rpy (radians): fixed axes x, then y, then z, i.e. R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    return quat_from_matrix(rot_z(yaw) @ rot_y(pitch) @ rot_x(roll))


def rpy_from_quat(q) -> tuple[float, float, float]:
    """Inverse of quat_from_rpy: (roll, pitch, yaw) radians, pitch in [-π/2, π/2]."""
    R = quat_to_matrix(q)
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(R[2, 0]) < 1 - 1e-9:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:                                    # gimbal lock: the yaw takes the whole turn
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return roll, pitch, yaw


def quat_conjugate(q) -> np.ndarray:
    x, y, z, w = quat_normalize(q)
    return np.array([-x, -y, -z, w])


def quat_angle(a, b) -> float:
    """The angle (radians) of the rotation that takes a to b: 2·atan2(|v|, |w|) of the relative quaternion, which
    keeps full precision for small angles (2·acos(w) loses half the digits near zero)."""
    r = quat_multiply(quat_conjugate(a), b)
    return 2.0 * math.atan2(float(np.linalg.norm(r[:3])), abs(float(r[3])))


def quat_to_mujoco(q) -> np.ndarray:
    """(x, y, z, w) -> MuJoCo's (w, x, y, z)."""
    x, y, z, w = quat_normalize(q)
    return np.array([w, x, y, z])


def quat_from_mujoco(q) -> np.ndarray:
    """MuJoCo's (w, x, y, z) -> (x, y, z, w)."""
    w, x, y, z = np.asarray(q, float)
    return quat_normalize([x, y, z, w])


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


@dataclass
class Pose:
    """A rigid transform child -> parent: position (m) and quaternion (x, y, z, w)."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternion: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))

    def __post_init__(self):
        self.position = np.asarray(self.position, float).reshape(3)
        self.quaternion = quat_normalize(self.quaternion)

    @staticmethod
    def from_matrix(T) -> 'Pose':
        T = np.asarray(T, float)
        return Pose(T[:3, 3].copy(), quat_from_matrix(T[:3, :3]))

    def matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = quat_to_matrix(self.quaternion)
        T[:3, 3] = self.position
        return T

    def inverse(self) -> 'Pose':
        return Pose.from_matrix(np.linalg.inv(self.matrix()))

    def __matmul__(self, other: 'Pose') -> 'Pose':
        return Pose.from_matrix(self.matrix() @ other.matrix())

    def apply(self, pts) -> np.ndarray:
        P = np.asarray(pts, float)
        return P @ quat_to_matrix(self.quaternion).T + self.position

    def to_dict(self) -> dict:
        return {'position': [float(v) for v in self.position], 'quaternion': [float(v) for v in self.quaternion]}

    @staticmethod
    def from_dict(d) -> 'Pose':
        d = d if isinstance(d, dict) else {}
        return Pose(d.get('position', (0, 0, 0)), d.get('quaternion', (0, 0, 0, 1)))


def optical_pose(body: Pose) -> Pose:
    """The optical frame of a camera whose body (link) frame is body: same origin, axes turned per REP-103."""
    T = body.matrix()
    T[:3, :3] = T[:3, :3] @ OPTICAL_TO_BODY
    return Pose.from_matrix(T)
