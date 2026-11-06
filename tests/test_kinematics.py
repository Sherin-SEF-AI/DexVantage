"""The independent product-of-exponentials kinematics (core/robot/kinematics.py) and the kinematics validation checks
(validation/checks_kinematics.py).

References: the twist exponential against scipy.linalg.expm of the 4×4 twist matrix; rotations against scipy's
Rotation; link poses against MuJoCo (RobotModel) on models written here and imported from disk through the importer;
the UR5e and Panda flanges at q = 0 against the sums of the published DH lengths; the Jacobian against MuJoCo's mj_jac
and central differences. Two deliberate convention slips (URDF rpy taken as intrinsic, MJCF quat read scalar-last)
must make the MuJoCo comparison fail, so the check is shown to catch them."""
import importlib.util
import json
import math

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from dexvantage import validation as VA
from dexvantage.core import project as P
from dexvantage.core.frames import Pose, quat_from_rpy
from dexvantage.core.robot import RobotModel, import_robot
from dexvantage.core.robot import kinematics as KN
from dexvantage.core.robot.kinematics import KinematicsError, PoEChain
from dexvantage.core.robot.model import RobotModelError
from dexvantage.validation import checks_kinematics as CK
from dexvantage.validation import suite

RNG = np.random.default_rng(20261106)


def _compare(model, chain, n=25, rng=RNG):
    """(max position error m, max rotation error rad) over n random joint vectors and every MuJoCo link."""
    ep = er = 0.0
    for q in CK.random_q(model, rng, n):
        A, B = model.fk(q), chain.fk(CK._to_poe(model, chain, q))
        for ln in model.link_names:
            p, r = CK.pose_errors(A[ln], B[ln])
            ep, er = max(ep, p), max(er, r)
    return ep, er


# ====================================================================== the maths
def test_twist_exponential_matches_the_matrix_exponential():
    for _ in range(200):
        w = RNG.normal(size=3)
        w /= np.linalg.norm(w)
        v = RNG.normal(size=3)
        th = RNG.uniform(-7, 7)
        X = np.zeros((4, 4))
        X[:3, :3] = KN.hat(w)
        X[:3, 3] = v
        assert np.abs(KN.exp_twist(np.r_[w, v], th) - expm(X * th)).max() < 1e-12
        u = v / np.linalg.norm(v)                       # a pure translation
        X = np.zeros((4, 4))
        X[:3, 3] = u
        assert np.abs(KN.exp_twist(np.r_[0, 0, 0, u], th) - expm(X * th)).max() < 1e-14
        assert np.abs(KN.exp_rotation(w, th) - Rotation.from_rotvec(w * th).as_matrix()).max() < 1e-13


def test_adjoint_moves_twists_between_frames():
    """Ad_T S is the screw of T e^[S]θ T⁻¹: T exp(S θ) T⁻¹ = exp((Ad_T S) θ)."""
    for _ in range(50):
        w = RNG.normal(size=3)
        w /= np.linalg.norm(w)
        S = np.r_[w, RNG.normal(size=3)]
        T = np.eye(4)
        T[:3, :3] = Rotation.random(random_state=int(RNG.integers(1 << 30))).as_matrix()
        T[:3, 3] = RNG.normal(size=3)
        th = RNG.uniform(-3, 3)
        lhs = T @ KN.exp_twist(S, th) @ np.linalg.inv(T)
        assert np.abs(lhs - KN.exp_twist(KN.adjoint(T) @ S, th)).max() < 1e-12


def test_rotation_conventions_against_scipy():
    for _ in range(100):
        r, p, y = RNG.uniform(-3, 3, 3)
        # URDF rpy: fixed (extrinsic) x, y, z axes = scipy lowercase 'xyz'
        assert np.abs(KN.rot_rpy(r, p, y) - Rotation.from_euler('xyz', [r, p, y]).as_matrix()).max() < 1e-13
        # MJCF euler: lowercase letters rotate with the frame (intrinsic) = scipy uppercase
        a = RNG.uniform(-3, 3, 3)
        assert np.abs(KN.rot_euler(a, 'xyz') - Rotation.from_euler('XYZ', a).as_matrix()).max() < 1e-13
        assert np.abs(KN.rot_euler(a, 'ZYX') - Rotation.from_euler('zyx', a).as_matrix()).max() < 1e-13
        q = RNG.normal(size=4)
        ref = Rotation.from_quat(q / np.linalg.norm(q), scalar_first=True).as_matrix()
        assert np.abs(KN.rot_quat_wxyz(q) - ref).max() < 1e-13
        z = RNG.normal(size=3)
        Rz = KN.rot_zaxis(z)
        assert np.allclose(Rz[:, 2], z / np.linalg.norm(z)) and np.allclose(Rz.T @ Rz, np.eye(3))
    assert np.allclose(KN.rot_zaxis([0, 0, -2]), np.diag([1.0, -1.0, -1.0]))
    R = KN.rot_xyaxes([2, 0, 0, 1, 3, 0])
    assert np.allclose(R, np.eye(3))


# ====================================================================== against MuJoCo
@pytest.mark.parametrize('k', range(5))
def test_builtin_robots_match_mujoco(k):
    r, src = CK.builtin_robots()[k]
    model, chain = RobotModel(r, src), PoEChain(r, src)
    assert set(model.joint_names) == set(chain.joint_names)
    assert set(model.link_names) <= set(chain.link_names)
    if r.source.kind == 'mjcf':
        assert chain.joint_names == model.joint_names        # MuJoCo's tree order
    ep, er = _compare(model, chain)
    assert ep < 1e-12 and er < 1e-12, (r.name, ep, er)


def test_mjcf_features_are_read():
    r, src = CK.builtin_robots()[3]
    chain = PoEChain(r, src)
    assert 'body_4' in chain.link_names and 'joint_5' in chain.joint_names     # unnamed, MuJoCo's numbering
    assert chain.joint_types == ['revolute'] * 5 + ['prismatic']
    assert chain.ref[2] == pytest.approx(math.radians(15)) and chain.ref[5] == pytest.approx(0.02)
    # the dict form: joints not named sit at their ref, which is the file's pose
    T0, Tref = chain.fk({}), chain.fk(chain.ref)
    assert all(np.allclose(T0[n], Tref[n]) for n in chain.link_names)
    # the free puck is held at the home keyframe's pose, with the robot's base applied
    base = r.base.matrix()
    assert np.allclose(T0['puck'][:3, 3], base[:3, :3] @ [0.5, 0.1, 0.3] + base[:3, 3])


def test_held_ball_and_free_joints_follow_the_keyframe():
    r, src = CK.builtin_robots()[3]
    key = f'{r.uid}/robot/features.xml'
    txt = src[key].decode()
    txt = txt.replace('0.9238795325112867 0 0.3826834323650898 0', '0.5 0.5 0.5 0.5').replace(
        '0.5 0.1 0.3 0.7071067811865476 0 0 0.7071067811865476', '-0.2 0.3 0.1 0 1 0 0')
    src = dict(src)
    src[key] = txt.encode()
    model, chain = RobotModel(r, src), PoEChain(r, src)
    ep, er = _compare(model, chain, 10)
    assert ep < 1e-12 and er < 1e-12


def test_imported_urdf_and_mjcf_match_mujoco(tmp_path):
    """Through the real importer: a ROS package URDF and an MJCF with an include in a sub-folder (the manifest)."""
    pkg = tmp_path / 'ws' / 'feat'
    (pkg / 'urdf').mkdir(parents=True)
    (pkg / 'urdf' / 'f.urdf').write_text(CK.feature_urdf_text())
    d = tmp_path / 'mj'
    (d / 'parts').mkdir(parents=True)
    (d / 'parts' / 'defaults.xml').write_text(CK.FEATURE_DEFAULTS)
    (d / 'features.xml').write_text(CK.FEATURE_MJCF)
    for path in (pkg / 'urdf' / 'f.urdf', d / 'features.xml'):
        r, src = import_robot(str(path))
        r.base = Pose([0.1, 0.2, 0.3], quat_from_rpy(0.3, -0.4, 0.5))
        model, chain = RobotModel(r, src), PoEChain(r, src)
        ep, er = _compare(model, chain, 10)
        assert ep < 1e-12 and er < 1e-12, path.name
        if r.source.kind == 'urdf':                # the URDF's own "world" link is the base
            assert np.allclose(chain.fk(np.zeros(chain.nq))['world'], r.base.matrix())


def test_jacobian_matches_mujoco_and_finite_differences():
    for r, src in CK.builtin_robots():
        model, chain = RobotModel(r, src), PoEChain(r, src)
        idx = [chain.joint_names.index(j) for j in model.joint_names]
        pt = np.array([0.03, -0.02, 0.05])
        for q in CK.random_q(model, RNG, 4):
            qp = CK._to_poe(model, chain, q)
            for ln in model.link_names:
                Jm = model.jacobian(q, ln, point=pt)
                Jp = chain.jacobian(qp, ln, point=pt)[:, idx]
                assert np.abs(Jm - Jp).max() < 1e-12, (r.name, ln)
                assert np.abs(chain.jacobian(qp, ln) - CK.fd_jacobian(chain, qp, ln)).max() < 1e-7
        # space Jacobian: twist columns; joints off the path are zero
        J = chain.space_jacobian(np.zeros(chain.nq) + chain.ref, chain.link_names[0])
        assert np.allclose(J, 0)


# ====================================================================== the closed forms
def test_dh_closed_forms_at_zero():
    """The flanges at q = 0 from the sums of the published lengths, independent of the matrix products."""
    ur = CK.ur5e_closed_form(np.zeros(6))[-1]
    assert np.allclose(ur[:3, 3], [-(0.425 + 0.3922), -(0.1333 + 0.0996), 0.1625 - 0.0997], atol=1e-15)
    pa = CK.panda_closed_form(np.zeros(7))[-1]
    assert np.allclose(pa[:3, 3], [0.088, 0.0, 0.333 + 0.316 + 0.384 - 0.107], atol=1e-15)
    for r, src in CK.builtin_robots()[:3]:
        chain = PoEChain(r, src)
        ref = (CK.ur5e_closed_form if 'ur5e' in r.uid else CK.panda_closed_form)(np.zeros(chain.nq))
        frames = [f'frame{i}' for i in range(1, chain.nq + 1)] + ([] if 'ur5e' in r.uid else ['flange'])
        T = chain.fk(np.zeros(chain.nq))
        for f, Tr in zip(frames, ref):
            assert np.abs(T[f] - Tr).max() < 1e-12, (r.name, f)


def test_standard_and_modified_dh_rows_written_out():
    """One row of each against the four elementary transforms multiplied out."""
    def Rz(t):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler('z', t).as_matrix()
        return T

    def Rx(t):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler('x', t).as_matrix()
        return T

    def Tr(x, y, z):
        T = np.eye(4)
        T[:3, 3] = [x, y, z]
        return T
    a, d, al, th = 0.3, -0.2, 0.7, -1.1
    assert np.allclose(CK.dh_standard(a, d, al, th), Rz(th) @ Tr(0, 0, d) @ Tr(a, 0, 0) @ Rx(al))
    assert np.allclose(CK.dh_modified(a, d, al, th), Rx(al) @ Tr(a, 0, 0) @ Rz(th) @ Tr(0, 0, d))


# ====================================================================== errors and inputs
def _robot(kind, text, extra=None):
    files = {'m.' + ('xml' if kind == 'mjcf' else 'urdf'): text}
    files.update(extra or {})
    return CK.make_robot('kin-test', 'Test arm', kind, files, next(iter(files)))


def test_q_forms_and_wrong_lengths():
    r, src = CK.builtin_robots()[2]
    chain = PoEChain(r, src)
    q = RNG.uniform(-1, 1, chain.nq)
    A = chain.fk(q)
    B = chain.fk(dict(zip(chain.joint_names, q)))
    assert all(np.allclose(A[n], B[n]) for n in A)
    with pytest.raises(ValueError, match='7 moving joints'):
        chain.fk(np.zeros(3))
    with pytest.raises(KeyError, match='no link named'):
        chain.jacobian(q, 'nope')


@pytest.mark.parametrize('kind,text,words', [
    ('urdf', '<robot name="x"><link name="a"', 'not valid XML'),
    ('urdf', '<robot name="x" xmlns:xacro="http://www.ros.org/wiki/xacro"><xacro:macro name="m"/></robot>', 'xacro'),
    ('urdf', '<robot name="x"><link name="a"/><link name="b"/><joint name="p" type="planar"><parent link="a"/>'
             '<child link="b"/></joint></robot>', 'planar'),
    ('urdf', '<robot name="x"><link name="a"/><joint name="j" type="fixed"><parent link="a"/><child link="c"/>'
             '</joint></robot>', 'not in the file'),
    ('mjcf', '<mujoco><worldbody><body><joint axis="0 0 0"/></body></worldbody></mujoco>', 'zero axis'),
    ('mjcf', '<mujoco><worldbody><body><joint class="nope"/></body></worldbody></mujoco>', 'does not define'),
    ('mjcf', '<mujoco><worldbody><replicate count="2"><body/></replicate></worldbody></mujoco>', 'replicate'),
    ('mjcf', '<mujoco><include file="gone.xml"/><worldbody/></mujoco>', 'not stored'),
    ('mjcf', '<mujoco><worldbody><body pos="1 2"/></worldbody></mujoco>', 'should have 3 numbers'),
])
def test_plain_errors(kind, text, words):
    r, src = _robot(kind, text)
    with pytest.raises(KinematicsError, match=words):
        PoEChain(r, src)


def test_missing_file_and_error_class():
    r = P.Robot(name='Ghost', source=P.RobotSource(kind='urdf', file='ghost/x.urdf'))
    with pytest.raises(RobotModelError, match='not stored'):
        PoEChain(r, {})


def test_mjcf_radian_and_eulerseq_against_mujoco():
    text = '''<mujoco><compiler angle="radian" eulerseq="XzY"/>
      <worldbody><body name="a" euler="0.3 -1.2 2.0" pos="0.1 0 0"><joint name="h" axis="0.2 -1 0.4" pos="0.05 0 0"/>
        <geom size="0.02"/><body name="b" axisangle="0 1 1 1.1" pos="0 0.2 0"><joint name="s" type="slide"
        axis="1 2 3" range="-0.1 0.1"/><geom size="0.02"/></body></body></worldbody></mujoco>'''
    r, src = _robot('mjcf', text)
    ep, er = _compare(RobotModel(r, src), PoEChain(r, src), 20)
    assert ep < 1e-12 and er < 1e-12


# ====================================================================== the validation checks
def _run(cid, quick=True, project=None):
    suite.load_checks()
    spec = next(s for s in suite.CHECKS if s.id == cid)
    ctx = suite.Ctx(project, None, None, VA.ValidationSettings(models=('kinematics',), quick=quick))
    try:
        return suite.run_one(spec, ctx)
    finally:
        ctx.close()


@pytest.mark.parametrize('cid', ['kinematics.fk_poe_mujoco', 'kinematics.dh_ur5e', 'kinematics.dh_panda',
                                 'kinematics.jacobian_fd', 'kinematics.jacobian_mujoco'])
def test_checks_pass_with_references(cid):
    c = _run(cid)
    assert c.passed, (c.id, c.error, c.exception, c.detail)
    assert c.reference and c.quantity and c.unit and c.note
    assert c.refs and all(k in VA.REFERENCES for k in c.refs)
    json.dumps(c.detail)


def test_reference_texts_name_the_sources():
    assert 'dh-parameters-for-calculations-of-kinematics-and-dynamics' in VA.REFERENCES['ur_dh']
    assert '476a2ed2c1' in VA.REFERENCES['franka_dh'] and '2026-11-06' in VA.REFERENCES['franka_dh']


def test_pinocchio_check_only_when_installed():
    suite.load_checks()
    ids = {s.id for s in suite.CHECKS}
    assert ('kinematics.fk_pinocchio' in ids) == (importlib.util.find_spec('pinocchio') is not None)


def test_the_mujoco_check_catches_convention_slips(monkeypatch):
    """URDF rpy read as intrinsic axes, or an MJCF quaternion read scalar-last, must fail the comparison."""
    monkeypatch.setattr(KN, 'rot_rpy', lambda r, p, y: Rotation.from_euler('XYZ', [r, p, y]).as_matrix())
    c = _run('kinematics.fk_poe_mujoco')
    assert not c.passed and c.error > 1e-3
    monkeypatch.undo()
    monkeypatch.setattr(KN, 'rot_quat_wxyz', lambda q: Rotation.from_quat(np.asarray(q, float) /
                                                                           np.linalg.norm(q)).as_matrix())
    c = _run('kinematics.fk_poe_mujoco')
    assert not c.passed and c.error > 1e-3


def test_project_robots_are_compared():
    p = P.blank_project('Kinematics')
    r, src = CK.builtin_robots()[4]
    r.primary = True
    p.robots.append(r)
    p.sources.update(src)
    c = _run('kinematics.fk_poe_mujoco', project=p)
    assert c.passed and any(k.endswith('(project)') for k in c.detail['robots'])
