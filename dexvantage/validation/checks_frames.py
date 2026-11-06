"""Checks of core/frames.py: quaternion and matrix conversions against scipy's Rotation, URDF's rpy convention, the
REP-103 optical frame as REP-103 writes it, MuJoCo's own quaternion routine, and pose composition."""
from __future__ import annotations

import numpy as np

from .suite import Outcome, check


def _rotations(ctx, cid, n=2000):
    """n uniformly random unit quaternions (x, y, z, w) plus the awkward ones: identity, half turns about each axis
    and about diagonals (the w ≈ 0 branches of Shepperd's method)."""
    rng = ctx.rng(cid)
    n = 200 if ctx.quick else n
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    s = np.sqrt(0.5)
    special = np.array([[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [s, s, 0, 0], [0, s, s, 0],
                        [s, 0, s, 0], [1e-9, 0, 0, 1], [0.5, 0.5, 0.5, 0.5], [s, 0, 0, s]], float)
    return np.vstack([special / np.linalg.norm(special, axis=1, keepdims=True), q])


@check('frames.quat_to_matrix', 'frames', 'Quaternion to rotation matrix',
       'largest element difference between DexVantage\'s rotation matrix and scipy\'s for the same quaternion',
       'scipy.spatial.transform.Rotation.from_quat(...).as_matrix() (scalar-last, the same order DexVantage stores)',
       1e-12, '(unitless)', refs=('scipy_rotation', 'shoemake1985'))
def _quat_to_matrix(ctx):
    from scipy.spatial.transform import Rotation
    from ..core import frames as F
    Q = _rotations(ctx, 'frames.quat_to_matrix')
    ours = np.stack([F.quat_to_matrix(q) for q in Q])
    ref = Rotation.from_quat(Q).as_matrix()
    err = np.abs(ours - ref).max(axis=(1, 2))
    return Outcome(float(err.max()), {'rotations': len(Q), 'median': float(np.median(err))},
                   note='Double precision rounding is about 1e-16 per element; 1e-12 leaves room for the products.')


@check('frames.matrix_to_quat', 'frames', 'Rotation matrix to quaternion',
       'largest angle between DexVantage\'s quaternion and scipy\'s for the same matrix, including half turns',
       'scipy.spatial.transform.Rotation.from_matrix(...).as_quat()', 1e-12, 'rad',
       refs=('scipy_rotation', 'shepperd1978'))
def _matrix_to_quat(ctx):
    from scipy.spatial.transform import Rotation
    from ..core import frames as F
    Q = _rotations(ctx, 'frames.matrix_to_quat')
    M = Rotation.from_quat(Q).as_matrix()
    ours = np.stack([F.quat_from_matrix(m) for m in M])
    ref = Rotation.from_matrix(M).as_quat()
    rel = Rotation.from_quat(ref).inv() * Rotation.from_quat(ours)
    ang = rel.magnitude()
    w_neg = int((ours[:, 3] < 0).sum())
    return Outcome(float(ang.max()), {'rotations': len(Q), 'w_negative': w_neg}, ok=w_neg == 0,
                   note='The angle of the relative rotation (scipy\'s magnitude, full precision near zero). The '
                        'quaternion\'s sign is chosen with w ≥ 0, which the check also requires.')


@check('frames.urdf_rpy', 'frames', 'URDF roll, pitch, yaw',
       'largest element difference between the rotation of an rpy triple and scipy\'s extrinsic x-y-z Euler rotation; '
       'and the round-trip angle error rpy → quaternion → rpy → quaternion',
       'URDF\'s definition (roll about fixed x, then pitch about fixed y, then yaw about fixed z) as scipy '
       'from_euler(\'xyz\', …), lowercase = extrinsic axes', 1e-12, '(unitless)', refs=('urdf', 'scipy_rotation'))
def _urdf_rpy(ctx):
    from scipy.spatial.transform import Rotation
    from ..core import frames as F
    rng = ctx.rng('frames.urdf_rpy')
    n = 100 if ctx.quick else 1000
    rpy = np.c_[rng.uniform(-np.pi, np.pi, n), rng.uniform(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, n),
                rng.uniform(-np.pi, np.pi, n)]
    ours = np.stack([F.quat_to_matrix(F.quat_from_rpy(*r)) for r in rpy])
    ref = Rotation.from_euler('xyz', rpy).as_matrix()
    err = float(np.abs(ours - ref).max())
    trip = max(F.quat_angle(F.quat_from_rpy(*F.rpy_from_quat(F.quat_from_rpy(*r))), F.quat_from_rpy(*r)) for r in rpy)
    return Outcome(max(err, trip), {'triples': n, 'matrix_error': err, 'round_trip_rad': trip})


@check('frames.optical_rep103', 'frames', 'REP-103 optical frame',
       'largest difference between the optical frame\'s axes, expressed in the camera body frame, and REP-103\'s '
       'definition (z forward = body x, x right = body −y, y down = body −z)',
       'REP-103 §"Suffix Frames", written out in the check', 0.0, '(unitless)', refs=('rep103',))
def _optical(ctx):
    from ..core import frames as F
    ref = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)   # columns: optical x, y, z in body coordinates
    ours = F.OPTICAL_TO_BODY
    body = F.Pose(np.zeros(3), F.quat_from_rpy(0.3, -0.2, 1.1))
    opt = F.optical_pose(body)
    rel = body.inverse() @ opt
    err = max(float(np.abs(ours - ref).max()), float(np.abs(F.quat_to_matrix(rel.quaternion) - ref).max()))
    return Outcome(err if err > 1e-14 else 0.0, {'matrix': ours.tolist()},
                   note='Exact: the axes are a signed permutation, so any difference beyond rounding is a wrong axis.')


@check('frames.mujoco_quat', 'frames', 'MuJoCo quaternion order',
       'largest element difference between DexVantage\'s rotation matrix and MuJoCo\'s mju_quat2Mat after the '
       '(x, y, z, w) → (w, x, y, z) conversion',
       'MuJoCo\'s own C routine mju_quat2Mat (the library DexVantage exports to)', 1e-12, '(unitless)',
       refs=('mujoco_xml',))
def _mujoco(ctx):
    import mujoco
    from ..core import frames as F
    Q = _rotations(ctx, 'frames.mujoco_quat', 500)
    worst = 0.0
    back = 0.0
    for q in Q:
        m = np.zeros(9)
        mujoco.mju_quat2Mat(m, F.quat_to_mujoco(q))
        worst = max(worst, float(np.abs(m.reshape(3, 3) - F.quat_to_matrix(q)).max()))
        back = max(back, F.quat_angle(F.quat_from_mujoco(F.quat_to_mujoco(q)), q))
    return Outcome(max(worst, back), {'rotations': len(Q), 'mujoco': mujoco.__version__, 'round_trip_rad': back})


@check('frames.pose_compose', 'frames', 'Pose composition and inverse',
       'largest position difference (m) of points mapped through a chain of three poses and its inverse, against '
       'scipy rotations applied step by step', 'scipy Rotation.apply and vector addition, one link at a time', 1e-12,
       'm', refs=('scipy_rotation',))
def _compose(ctx):
    from scipy.spatial.transform import Rotation
    from ..core import frames as F
    rng = ctx.rng('frames.pose_compose')
    worst = 0.0
    for _ in range(20 if ctx.quick else 200):
        poses = [F.Pose(rng.uniform(-1, 1, 3), rng.normal(size=4)) for _ in range(3)]
        pts = rng.uniform(-0.5, 0.5, (16, 3))
        chain = poses[0] @ poses[1] @ poses[2]
        ref = pts
        for P in reversed(poses):
            ref = Rotation.from_quat(P.quaternion).apply(ref) + P.position
        worst = max(worst, float(np.abs(chain.apply(pts) - ref).max()),
                    float(np.abs(chain.inverse().apply(ref) - pts).max()))
    return Outcome(worst, {})
