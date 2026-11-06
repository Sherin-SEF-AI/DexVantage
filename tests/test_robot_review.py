"""Regression tests from the Phase 1 correctness review of core/robot/: IK wrapping of unlimited revolute joints for
the 'nearest_q0' rule, the reach bound of slide joints, the rotation log near a half turn, batch shapes, and the
Jacobian check's offset points.

References: poses recomputed by the product-of-exponentials chain (core/robot/kinematics.py, no MuJoCo) or written
out here; rotation vectors from scipy.spatial.transform.Rotation."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dexvantage.core.robot import collision as C
from dexvantage.core.robot import ik
from dexvantage.core.robot import manipulability as M
from dexvantage.core.robot.kinematics import PoEChain
from dexvantage.core.robot.model import RobotModel
from dexvantage.validation import checks_kinematics as CK


def _features_urdf():
    r, src = CK.builtin_robots()[4]               # base rotated; joint a2 is continuous (unlimited)
    return RobotModel(r, src), PoEChain(r, src)


def _poe_q(model, chain, q):
    return CK._to_poe(model, chain, q)


# ---------------------------------------------------------------------------------------------- nearest_q0 wrapping
def test_nearest_q0_keeps_unlimited_joints_within_a_half_turn_of_q0():
    model, chain = _features_urdf()
    k = model.joint_names.index('a2')
    assert model.joint_types[k] == 'continuous'
    q0 = np.array([0.4, 3.0, 0.5, 0.01, 0.01])     # a2 near +π: one more step crosses it
    q1 = q0.copy()
    q1[k] = 3.25
    target = chain.fk(_poe_q(model, chain, q1))['tcp'][:3, 3]
    r = ik.solve(model, 'tcp', target, q0=q0, rule='nearest_q0', collision=False)
    assert r.ok
    assert abs(r.q[k] - q0[k]) < np.pi, r.q          # 3.25, not 3.25 − 2π (a full-turn jump of the slider)
    back = chain.fk(_poe_q(model, chain, r.q))['tcp'][:3, 3]
    assert np.linalg.norm(back - target) <= ik.POS_TOL
    # 'nearest_home' still wraps to within π of home
    h = ik.solve(model, 'tcp', target, q0=q0, rule='nearest_home', collision=False)
    assert h.ok and abs(h.q[k] - model.home[k]) <= np.pi + 1e-12


def test_held_unlimited_joints_are_not_wrapped():
    model, _chain = _features_urdf()
    k = model.joint_names.index('a2')
    q0 = model.home.copy()
    q0[k] = 4.0                                      # a2 moves the tcp, so use a link it does not move: l1
    r = ik.solve(model, 'l1', model.link_pose(q0, 'l1')[:3, 3], q0=q0, rule='nearest_home', collision=False)
    assert r.ok and r.q[k] == pytest.approx(4.0)     # off the chain: held exactly, not moved by a turn


# ---------------------------------------------------------------------------------------------- the reach bound
SLIDE = '''<mujoco><worldbody><body name="b"><geom size="0.01"/>
  <body name="s"><joint name="s" type="slide" axis="1 0 0" range="-0.1 0.1"/><geom size="0.01"/>
    <body name="t" pos="0.1 0 0"/></body></body></worldbody>
  <keyframe><key name="home" qpos="-0.1"/></keyframe></mujoco>'''


def test_reach_bound_counts_slide_travel_from_home():
    r, src = CK.make_robot('slide', 'Slide', 'mjcf', {'s.xml': SLIDE}, 's.xml')
    model = RobotModel(r, src)
    assert model.home[0] == pytest.approx(-0.1)
    a0, rmax = ik._Chain(model, 't').reach()
    # the tool reaches x = 0.1 + 0.1 = 0.2 m from the slide's fixed anchor at the origin (written out)
    assert rmax >= 0.2 - 1e-12, rmax
    # a reachable target the solver cannot finish in one iteration is not called out of reach
    res = ik.solve(model, 't', [0.19, 0.0, 0.0], seeds=1, retry_seeds=0, max_iter=1, collision=False)
    assert res.reason != 'out of reach'
    assert ik.solve(model, 't', [0.19, 0.0, 0.0], collision=False).ok
    assert ik.solve(model, 't', [0.25, 0.0, 0.0], collision=False).reason == 'out of reach'


# ---------------------------------------------------------------------------------------------- the rotation log
@pytest.mark.parametrize('gap', [1e-4, 5e-4, 9e-4, 2e-3, 0.3])
def test_rotvec_near_a_half_turn_against_scipy(gap):
    rng = np.random.default_rng(11)
    axes = rng.normal(size=(20, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    ref = axes * (np.pi - gap)
    R = Rotation.from_rotvec(ref).as_matrix()
    assert np.abs(ik._rotvec(R) - ref).max() < 1e-9


# ---------------------------------------------------------------------------------------------- batch shapes
def test_batches_refuse_rows_of_the_wrong_length():
    model, _chain = _features_urdf()
    assert model.nq == 5
    bad = np.zeros((6, 5 + 1))                       # 36 values: would reshape silently to (…, 5) only by accident
    bad2 = np.zeros((5, 6))                          # 30 values: reshape(-1, 5) would give 6 wrong joint vectors
    for Q in (bad, bad2):
        with pytest.raises(ValueError, match='moving joints'):
            model.fk_batch(Q)
        with pytest.raises(ValueError, match='moving joints'):
            C.self_collision_batch(model, Q)
        with pytest.raises(ValueError, match='moving joints'):
            M.measures_batch(model, Q, 'tcp')
    one = model.fk_batch(model.home)
    assert one.shape == (1, len(model.link_names), 4, 4)
    assert np.abs(one[0] - model.fk_array(model.home)).max() == 0.0
    assert M.measures_batch(model, np.tile(model.home, (3, 1)), 'tcp')['yoshikawa'].shape == (3,)


def test_ik_start_of_the_wrong_shape_is_explained():
    model, _chain = _features_urdf()
    T = np.tile(model.link_pose(model.home, 'tcp')[:3, 3], (3, 1))
    with pytest.raises(ValueError, match='one joint vector per target'):
        ik.solve_batch(model, 'tcp', T, q0=np.zeros((2, model.nq)))
    with pytest.raises(ValueError, match='moving joints'):
        ik.solve_batch(model, 'tcp', T, q0=np.zeros(model.nq + 1))


# ---------------------------------------------------------------------------------------------- Jacobian at a point
def test_jacobian_of_an_offset_point_matches_poe_and_the_velocity_rule():
    model, chain = _features_urdf()
    rng = np.random.default_rng(5)
    idx = [chain.joint_names.index(j) for j in model.joint_names]
    for q in CK.random_q(model, rng, 5):
        qp = _poe_q(model, chain, q)
        for ln in model.link_names:
            p = rng.uniform(-0.2, 0.2, 3)
            Jm, Jc = model.jacobian(q, ln, p), chain.jacobian(qp, ln, p)[:, idx]
            assert np.abs(Jm - Jc).max() < 1e-9
            # v_point = v_origin + ω × (R p), written out from the origin's Jacobian
            T = model.link_pose(q, ln)
            J0 = model.jacobian(q, ln)
            lin = J0[:3] + np.cross(J0[3:].T, T[:3, :3] @ p).T
            assert np.abs(Jm[:3] - lin).max() < 1e-12


# ---------------------------------------------------------------------------------------------- fused static bodies
FUSED = '''<mujoco><compiler fusestatic="true" angle="radian"/><worldbody><body name="base"><geom size="0.02"/>
  <body name="link1" pos="0 0 0.1"><joint name="j1" axis="0 0 1"/><geom size="0.02"/>
    <body name="bracket" pos="0.05 0 0.02" euler="0 0.4 0"><geom size="0.01"/>
      <body name="link2" pos="0.1 0 0"><joint name="j2" axis="0 1 0"/><geom size="0.02"/>
        <body name="tcp" pos="0 0 0.08"/></body></body></body></body></worldbody></mujoco>'''


def test_an_mjcf_files_fusestatic_keeps_jointless_bodies_as_frames():
    r, src = CK.make_robot('fused', 'Fused', 'mjcf', {'f.xml': FUSED}, 'f.xml')
    model, chain = RobotModel(r, src), PoEChain(r, src)
    assert {'base', 'bracket', 'tcp'} <= set(model.link_names)
    for q in CK.random_q(model, np.random.default_rng(2), 10):
        A, B = model.fk(q), chain.fk(_poe_q(model, chain, q))
        for ln in model.link_names:
            assert np.abs(A[ln] - B[ln]).max() < 1e-12, ln
