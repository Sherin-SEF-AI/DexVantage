"""Closed-form hand-eye solvers: AX = XB (Tsai-Lenz, Park-Martin, Horaud-Dornaika, Andreff, Daniilidis) and the
robot-world/hand-eye AX = ZB (Shah, Li), with OpenCV's calibrateHandEye and calibrateRobotWorldHandEye used when the
installed OpenCV provides them.

The conventions are OpenCV's (calib3d, "calibrateHandEye" and "calibrateRobotWorldHandEye"):

- hand_eye(T_gripper2base, T_target2cam, method) -> T_cam2gripper. With X = T_cam2gripper and every pair of poses
  i < j, A = T_g2b[j]⁻¹ T_g2b[i] and B = T_t2c[j] T_t2c[i]⁻¹, so A X = X B (OpenCV uses every pair as well). Eye-to-hand
  (a fixed camera, the board on the gripper): pass T_base2gripper (the inverse FK) instead and the result is
  T_cam2base.
- robot_world(T_world2cam, T_base2gripper, method) -> (T_base2world, T_gripper2cam): A X = Z B with A = T_world2cam,
  X = T_base2world, Z = T_gripper2cam, B = T_base2gripper.

The NumPy versions here are written from the papers (validation/references.py: tsai_lenz1989, park_martin1994,
horaud_dornaika1995, andreff1999, daniilidis1999, shah2013, li2010):

- Tsai-Lenz: the rotation from the modified Rodrigues vectors P = 2 sin(θ/2) n of A and B, skew(P_A + P_B) P' =
  P_B − P_A in least squares, P = 2 P' / √(1 + |P'|²); the translation from (R_A − I) t = R_X t_B − t_A.
- Park-Martin: R_X = (MᵀM)^(−½) Mᵀ with M = Σ β αᵀ, α = log R_A, β = log R_B; the same translation step.
- Horaud-Dornaika: the unit quaternion q minimising Σ |q_A q − q q_B|², the eigenvector of the smallest eigenvalue of
  Σ (L(q_A) − R(q_B))ᵀ (L(q_A) − R(q_B)); the same translation step.
- Andreff: the linear system [I₉ − R_A ⊗ R_B] vec R_X = 0, (I₃ ⊗ t_Bᵀ) vec R_X + (I − R_A) t_X = t_A (row-major
  vec) solved in least squares, R_X projected onto SO(3); the translation step again with that rotation.
- Daniilidis: the dual quaternions of A and B (scalar parts made the same sign), the 6 × 8 blocks
  [[a − b, [a + b]×, 0, 0], [a' − b', [a' + b']×, a − b, [a + b]×]] stacked, the solution λ₁ v₇ + λ₂ v₈ in the span of
  the two last right singular vectors with |q| = 1 and q·q' = 0.
- Shah: vec R_Z = (R_B ⊗ R_A) vec R_X (column-major vec); R_X and R_Z from the leading singular pair of
  Σ R_B ⊗ R_A, projected onto SO(3); then [R_A, −I] (t_X, t_Z) = R_Z t_B − t_A in least squares.
- Li: the linear system in (vec R_X, vec R_Z, t_X, t_Z): (I ⊗ R_A) vec R_X − (R_Bᵀ ⊗ I) vec R_Z = 0 and
  R_A t_X − t_Z − (t_Bᵀ ⊗ I) vec R_Z = −t_A in least squares, the rotations projected onto SO(3), then the
  translations solved again with those rotations.

These are initialisers for the bundle adjustment (handeye.py); none is a maximum-likelihood estimate. The NumPy
versions follow the papers, not OpenCV's source. Measured once against OpenCV 4.10.0 (docs/models/handeye.md):
Park-Martin, Horaud-Dornaika, Daniilidis and Shah agree to rounding; Andreff and Li differ by several times the
pose noise (OpenCV's keep their linear translation; these re-solve it with the projected rotation, and were the
closer to the truth); Tsai-Lenz by a few tenths of a millimetre under noise, for a reason not traced.

Known limits: every method needs at least two motions with non-parallel rotation axes (three poses); the
translations need rotations of tens of degrees to be well conditioned. Motions with a rotation near 180° make the
modified Rodrigues vector of Tsai-Lenz large.
"""
from __future__ import annotations

import math

import numpy as np

HAND_EYE_METHODS = ('tsai', 'park', 'horaud', 'andreff', 'daniilidis')
ROBOT_WORLD_METHODS = ('shah', 'li')
_CV_HE = {'tsai': 'CALIB_HAND_EYE_TSAI', 'park': 'CALIB_HAND_EYE_PARK', 'horaud': 'CALIB_HAND_EYE_HORAUD',
          'andreff': 'CALIB_HAND_EYE_ANDREFF', 'daniilidis': 'CALIB_HAND_EYE_DANIILIDIS'}
_CV_RW = {'shah': 'CALIB_ROBOT_WORLD_HAND_EYE_SHAH', 'li': 'CALIB_ROBOT_WORLD_HAND_EYE_LI'}
METHOD_LABEL = {'tsai': 'Tsai-Lenz', 'park': 'Park-Martin', 'horaud': 'Horaud-Dornaika', 'andreff': 'Andreff',
                'daniilidis': 'Daniilidis', 'shah': 'Shah', 'li': 'Li'}


class SolverError(ValueError):
    """The poses cannot determine the transform (too few, or no rotation between them); said in plain words."""


def opencv_has_hand_eye() -> bool:
    """True when the installed OpenCV has cv2.calibrateHandEye (OpenCV 5.0.0's Python package has only the
    method constants)."""
    try:
        import cv2
    except ImportError:
        return False
    return hasattr(cv2, 'calibrateHandEye')


def opencv_has_robot_world() -> bool:
    try:
        import cv2
    except ImportError:
        return False
    return hasattr(cv2, 'calibrateRobotWorldHandEye')


# ----------------------------------------------------------------------------- small rotation helpers
def skew(v) -> np.ndarray:
    x, y, z = np.asarray(v, float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def project_so3(M) -> np.ndarray:
    """The rotation nearest to M (Frobenius), by SVD."""
    U, _, Vt = np.linalg.svd(np.asarray(M, float))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt)) or 1.0])
    return U @ D @ Vt


def log_so3(R) -> np.ndarray:
    """Rotation vector (axis · angle, rad) of R, stable near 0 and π."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(np.asarray(R, float)).as_rotvec()


def exp_so3(w) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.asarray(w, float)).as_matrix()


def quat_wxyz(R) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(np.asarray(R, float)).as_quat()
    return np.array([w, x, y, z])


def quat_to_R(q) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    w, x, y, z = np.asarray(q, float)
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def qmul(a, b) -> np.ndarray:
    aw, av = a[0], np.asarray(a[1:])
    bw, bv = b[0], np.asarray(b[1:])
    return np.r_[aw * bw - av @ bv, aw * bv + bw * av + np.cross(av, bv)]


def _qL(q) -> np.ndarray:
    """Left-multiplication matrix: q ⊗ p = L(q) p."""
    w, x, y, z = q
    return np.array([[w, -x, -y, -z], [x, w, -z, y], [y, z, w, -x], [z, -y, x, w]])


def _qR(q) -> np.ndarray:
    """Right-multiplication matrix: p ⊗ q = R(q) p."""
    w, x, y, z = q
    return np.array([[w, -x, -y, -z], [x, w, z, -y], [y, -z, w, x], [z, y, -x, w]])


def T_of(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, float).ravel()
    return T


def inv_T(T) -> np.ndarray:
    T = np.asarray(T, float)
    R, t = T[:3, :3], T[:3, 3]
    return T_of(R.T, -R.T @ t)


def _stack(Ts, name) -> np.ndarray:
    A = np.asarray(Ts, float)
    if A.ndim != 3 or A.shape[1:] != (4, 4):
        raise SolverError(f'{name} must be a list of 4 × 4 transforms')
    return A


# ----------------------------------------------------------------------------- AX = XB
def motion_pairs(T_g2b, T_t2c) -> tuple[np.ndarray, np.ndarray]:
    """(A, B) motions of every pair i < j: A = T_g2b[j]⁻¹ T_g2b[i], B = T_t2c[j] T_t2c[i]⁻¹ (A X = X B)."""
    G, C = _stack(T_g2b, 'the gripper poses'), _stack(T_t2c, 'the board poses')
    if len(G) != len(C):
        raise SolverError(f'{len(G)} gripper poses but {len(C)} board poses')
    if len(G) < 3:
        raise SolverError(f'hand-eye calibration needs at least 3 poses (two motions about different axes); got '
                          f'{len(G)}')
    A, B = [], []
    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            A.append(inv_T(G[j]) @ G[i])
            B.append(C[j] @ inv_T(C[i]))
    return np.array(A), np.array(B)


def _translation(A, B, R) -> np.ndarray:
    """t from (R_A − I) t = R t_B − t_A in least squares."""
    M = np.concatenate([a[:3, :3] - np.eye(3) for a in A])
    y = np.concatenate([R @ b[:3, 3] - a[:3, 3] for a, b in zip(A, B)])
    return np.linalg.lstsq(M, y, rcond=None)[0]


def _check_rotation_spread(A):
    w = np.array([log_so3(a[:3, :3]) for a in A])
    big = w[np.linalg.norm(w, axis=1) > 1e-6]
    if len(big) < 2 or np.linalg.matrix_rank(big, tol=1e-6 * max(1.0, np.abs(big).max())) < 2:
        raise SolverError('the gripper poses do not rotate about two different axes, so the hand-eye rotation is '
                          'not determined: add poses that tilt the gripper about other axes')


def _tsai(A, B) -> np.ndarray:
    M, y = [], []
    for a, b in zip(A, B):
        wa, wb = log_so3(a[:3, :3]), log_so3(b[:3, :3])
        ta, tb = np.linalg.norm(wa), np.linalg.norm(wb)
        pa = 2 * math.sin(ta / 2) * (wa / ta) if ta > 1e-12 else np.zeros(3)
        pb = 2 * math.sin(tb / 2) * (wb / tb) if tb > 1e-12 else np.zeros(3)
        M.append(skew(pa + pb))
        y.append(pb - pa)
    p1 = np.linalg.lstsq(np.concatenate(M), np.concatenate(y), rcond=None)[0]
    p = 2 * p1 / math.sqrt(1 + p1 @ p1)
    n2 = p @ p
    return (1 - n2 / 2) * np.eye(3) + 0.5 * (np.outer(p, p) + math.sqrt(max(4 - n2, 0.0)) * skew(p))


def _park(A, B) -> np.ndarray:
    M = np.zeros((3, 3))
    for a, b in zip(A, B):
        M += np.outer(log_so3(b[:3, :3]), log_so3(a[:3, :3]))
    w, V = np.linalg.eigh(M.T @ M)
    if w.min() <= 1e-15 * max(w.max(), 1e-300):
        raise SolverError('Park-Martin: the motions do not determine the rotation (rotation axes all parallel)')
    return project_so3(V @ np.diag(w ** -0.5) @ V.T @ M.T)


def _horaud(A, B) -> np.ndarray:
    S = np.zeros((4, 4))
    for a, b in zip(A, B):
        qa, qb = quat_wxyz(a[:3, :3]), quat_wxyz(b[:3, :3])
        if qa[0] * qb[0] < 0:
            qb = -qb
        D = _qL(qa) - _qR(qb)
        S += D.T @ D
    w, V = np.linalg.eigh(S)
    return quat_to_R(V[:, 0])


def _andreff(A, B) -> np.ndarray:
    rows = []
    for a, b in zip(A, B):
        Ra = a[:3, :3]
        Rb = b[:3, :3]
        rows.append(np.eye(9) - np.kron(Ra, Rb))          # row-major vec: vec(Ra X Rbᵀ) = (Ra ⊗ Rb) vec X
    M = np.concatenate(rows)
    _, s, Vt = np.linalg.svd(M)
    X = Vt[-1].reshape(3, 3)
    d = np.linalg.det(X)
    if abs(d) < 1e-12:
        raise SolverError('Andreff: the motions do not determine the rotation')
    X = X / (np.sign(d) * abs(d) ** (1 / 3))
    return project_so3(X)


def _dq(T) -> tuple[np.ndarray, np.ndarray]:
    q = quat_wxyz(T[:3, :3])
    qd = 0.5 * qmul(np.r_[0.0, T[:3, 3]], q)
    return q, qd


def _daniilidis(A, B) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for a, b in zip(A, B):
        qa, qad = _dq(a)
        qb, qbd = _dq(b)
        if qa[0] * qb[0] < 0:
            qb, qbd = -qb, -qbd
        av, adv, bv, bdv = qa[1:], qad[1:], qb[1:], qbd[1:]
        S = np.zeros((6, 8))
        S[:3, 0] = av - bv
        S[:3, 1:4] = skew(av + bv)
        S[3:, 0] = adv - bdv
        S[3:, 1:4] = skew(adv + bdv)
        S[3:, 4] = av - bv
        S[3:, 5:8] = skew(av + bv)
        rows.append(S)
    T = np.concatenate(rows)
    _, s, Vt = np.linalg.svd(T)
    v7, v8 = Vt[-2], Vt[-1]
    u1, w1, u2, w2 = v7[:4], v7[4:], v8[:4], v8[4:]
    a = u1 @ w1
    b = u1 @ w2 + u2 @ w1
    c = u2 @ w2
    if abs(a) < 1e-15:
        roots = [-c / b] if abs(b) > 1e-15 else [0.0]
    else:
        disc = b * b - 4 * a * c
        disc = max(disc, 0.0)
        roots = [(-b + math.sqrt(disc)) / (2 * a), (-b - math.sqrt(disc)) / (2 * a)]
    best = None
    for r in roots:
        val = r * r * (u1 @ u1) + 2 * r * (u1 @ u2) + u2 @ u2
        if best is None or val > best[0]:
            best = (val, r)
    val, r = best
    if val <= 0:
        raise SolverError('Daniilidis: no unit dual quaternion fits the motions')
    l2 = 1 / math.sqrt(val)
    l1 = r * l2
    q = l1 * v7 + l2 * v8
    qr, qd = q[:4], q[4:]
    R = quat_to_R(qr)
    t = 2 * qmul(qd, np.r_[qr[0], -qr[1:]])[1:]
    return R, t


def hand_eye_numpy(T_g2b, T_t2c, method: str = 'park') -> np.ndarray:
    """AX = XB by one of HAND_EYE_METHODS, in NumPy (see the module docstring): T_cam2gripper."""
    m = str(method).lower()
    if m not in HAND_EYE_METHODS:
        raise SolverError(f'unknown hand-eye method {method!r}: use one of {", ".join(HAND_EYE_METHODS)}')
    A, B = motion_pairs(T_g2b, T_t2c)
    _check_rotation_spread(A)
    if m == 'daniilidis':
        R, t = _daniilidis(A, B)
        return T_of(R, t)
    R = {'tsai': _tsai, 'park': _park, 'horaud': _horaud, 'andreff': _andreff}[m](A, B)
    return T_of(R, _translation(A, B, R))


def hand_eye_opencv(T_g2b, T_t2c, method: str = 'park') -> np.ndarray:
    """cv2.calibrateHandEye (raises SolverError when the installed OpenCV does not provide it)."""
    import cv2
    if not opencv_has_hand_eye():
        raise SolverError(f'OpenCV {cv2.__version__} has no calibrateHandEye in its Python bindings')
    G, C = _stack(T_g2b, 'the gripper poses'), _stack(T_t2c, 'the board poses')
    R, t = cv2.calibrateHandEye([g[:3, :3] for g in G], [g[:3, 3].reshape(3, 1) for g in G],
                                [c[:3, :3] for c in C], [c[:3, 3].reshape(3, 1) for c in C],
                                method=getattr(cv2, _CV_HE[str(method).lower()]))
    return T_of(R, t)


def hand_eye(T_g2b, T_t2c, method: str = 'park', backend: str = 'auto') -> tuple[np.ndarray, str]:
    """(T_cam2gripper, backend used): OpenCV's calibrateHandEye when it is installed ('auto' or 'opencv'), else the
    NumPy version ('numpy')."""
    use_cv = backend == 'opencv' or (backend == 'auto' and opencv_has_hand_eye())
    if use_cv:
        return hand_eye_opencv(T_g2b, T_t2c, method), 'opencv'
    return hand_eye_numpy(T_g2b, T_t2c, method), 'numpy'


# ----------------------------------------------------------------------------- AX = ZB
def _rw_inputs(T_w2c, T_b2g):
    A, B = _stack(T_w2c, 'the board-to-camera poses'), _stack(T_b2g, 'the base-to-gripper poses')
    if len(A) != len(B):
        raise SolverError(f'{len(A)} camera poses but {len(B)} robot poses')
    if len(A) < 3:
        raise SolverError(f'robot-world/hand-eye calibration needs at least 3 poses; got {len(A)}')
    return A, B


def _rw_translations(A, B, RX, RZ):
    M = np.concatenate([np.c_[a[:3, :3], -np.eye(3)] for a in A])
    y = np.concatenate([RZ @ b[:3, 3] - a[:3, 3] for a, b in zip(A, B)])
    t = np.linalg.lstsq(M, y, rcond=None)[0]
    return t[:3], t[3:]


def _shah(A, B):
    K = np.zeros((9, 9))
    for a, b in zip(A, B):
        K += np.kron(b[:3, :3], a[:3, :3])
    U, s, Vt = np.linalg.svd(K)
    X = Vt[0].reshape(3, 3, order='F')
    Z = U[:, 0].reshape(3, 3, order='F')
    sx = np.sign(np.linalg.det(X)) or 1.0
    sz = np.sign(np.linalg.det(Z)) or 1.0
    RX, RZ = project_so3(sx * X), project_so3(sz * Z)
    return RX, RZ


def _li(A, B):
    n = len(A)
    M = np.zeros((12 * n, 24))
    y = np.zeros(12 * n)
    I3 = np.eye(3)
    for i, (a, b) in enumerate(zip(A, B)):
        Ra, ta, Rb, tb = a[:3, :3], a[:3, 3], b[:3, :3], b[:3, 3]
        r = 12 * i
        M[r:r + 9, 0:9] = np.kron(I3, Ra)
        M[r:r + 9, 9:18] = -np.kron(Rb.T, I3)
        M[r + 9:r + 12, 9:18] = -np.kron(tb.reshape(1, 3), I3)
        M[r + 9:r + 12, 18:21] = Ra
        M[r + 9:r + 12, 21:24] = -I3
        y[r + 9:r + 12] = -ta
    w = np.linalg.lstsq(M, y, rcond=None)[0]
    X = w[0:9].reshape(3, 3, order='F')
    Z = w[9:18].reshape(3, 3, order='F')
    if abs(np.linalg.det(X)) < 1e-12 or abs(np.linalg.det(Z)) < 1e-12:
        raise SolverError('Li: the poses do not determine the rotations')
    return project_so3(X), project_so3(Z)


def robot_world_numpy(T_w2c, T_b2g, method: str = 'shah') -> tuple[np.ndarray, np.ndarray]:
    """AX = ZB by one of ROBOT_WORLD_METHODS, in NumPy: (T_base2world, T_gripper2cam)."""
    m = str(method).lower()
    if m not in ROBOT_WORLD_METHODS:
        raise SolverError(f'unknown robot-world method {method!r}: use one of {", ".join(ROBOT_WORLD_METHODS)}')
    A, B = _rw_inputs(T_w2c, T_b2g)
    _check_rotation_spread([inv_T(B[0]) @ b for b in B[1:]])
    RX, RZ = (_shah if m == 'shah' else _li)(A, B)
    tX, tZ = _rw_translations(A, B, RX, RZ)
    return T_of(RX, tX), T_of(RZ, tZ)


def robot_world_opencv(T_w2c, T_b2g, method: str = 'shah') -> tuple[np.ndarray, np.ndarray]:
    import cv2
    if not opencv_has_robot_world():
        raise SolverError(f'OpenCV {cv2.__version__} has no calibrateRobotWorldHandEye in its Python bindings')
    A, B = _rw_inputs(T_w2c, T_b2g)
    RX, tX, RZ, tZ = cv2.calibrateRobotWorldHandEye([a[:3, :3] for a in A], [a[:3, 3].reshape(3, 1) for a in A],
                                                    [b[:3, :3] for b in B], [b[:3, 3].reshape(3, 1) for b in B],
                                                    method=getattr(cv2, _CV_RW[str(method).lower()]))
    return T_of(RX, tX), T_of(RZ, tZ)


def robot_world(T_w2c, T_b2g, method: str = 'shah', backend: str = 'auto') -> tuple[np.ndarray, np.ndarray, str]:
    """(T_base2world, T_gripper2cam, backend used): OpenCV's calibrateRobotWorldHandEye when installed, else
    NumPy."""
    use_cv = backend == 'opencv' or (backend == 'auto' and opencv_has_robot_world())
    if use_cv:
        X, Z = robot_world_opencv(T_w2c, T_b2g, method)
        return X, Z, 'opencv'
    X, Z = robot_world_numpy(T_w2c, T_b2g, method)
    return X, Z, 'numpy'


def mean_transform(Ts) -> np.ndarray:
    """The chordal mean of rigid transforms: rotations by SVD of their sum, translations averaged."""
    A = _stack(Ts, 'the transforms')
    return T_of(project_so3(A[:, :3, :3].sum(0)), A[:, :3, 3].mean(0))


def transform_error(T, T_ref) -> tuple[float, float]:
    """(translation error m, rotation error rad) of T against T_ref."""
    T, T_ref = np.asarray(T, float), np.asarray(T_ref, float)
    dR = T_ref[:3, :3].T @ T[:3, :3]
    return float(np.linalg.norm(T[:3, 3] - T_ref[:3, 3])), float(np.linalg.norm(log_so3(dR)))
