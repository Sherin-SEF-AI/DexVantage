"""Inverse kinematics, manipulability and collision distances (core/robot/ik.py, manipulability.py, collision.py) on
scripted MJCF robots written here (validation/checks_ik.py builds the same fixtures).

References: tool poses recomputed by the product of exponentials in checks_ik.poe_fk (no MuJoCo); the planar 2R
closed form |l1 l2 sin q2|; sphere distances |c1 - c2| - r1 - r2; a floor plane's distance z - r."""
import numpy as np
import pytest

from dexvantage.core.robot import collision as C
from dexvantage.core.robot import ik
from dexvantage.core.robot import manipulability as M
from dexvantage.validation import checks_ik as F


@pytest.fixture(scope='module')
def arm6(tmp_path_factory):
    return F.model_from_mjcf(F.arm_mjcf(F.ARM6), str(tmp_path_factory.mktemp('arm6')))[0]


@pytest.fixture(scope='module')
def arm5(tmp_path_factory):
    return F.model_from_mjcf(F.arm_mjcf(F.ARM5), str(tmp_path_factory.mktemp('arm5')))[0]


def mj(tmp_path, text, name='robot.xml'):
    return F.model_from_mjcf(text, str(tmp_path), name)[0]


def targets(model, chain, n, seed=0):
    rng = np.random.default_rng(seed)
    Q = F._random_free_q(model, rng, n)
    return Q, np.stack([F.poe_fk(chain, q) for q in Q])


def angle(Ra, Rb):
    return float(np.arccos(np.clip(0.5 * (np.trace(Ra.T @ Rb) - 1), -1, 1)))


# ---------------------------------------------------------------------------------------------- IK
def test_round_trip_full_pose_within_tolerance_and_limits(arm6):
    Q, T = targets(arm6, F.ARM6, 40)
    res = ik.solve_batch(arm6, 'tool', T)
    assert res.mode == 'pose' and res.ok.all(), res.counts()
    assert res.counts() == {'ok': 40} and all(r == '' for r in res.reason)
    for q, t in zip(res.q, T):
        back = F.poe_fk(F.ARM6, q)
        assert np.linalg.norm(back[:3, 3] - t[:3, 3]) <= ik.POS_TOL
        assert angle(back[:3, :3], t[:3, :3]) <= ik.ROT_TOL
    assert (res.q >= arm6.lower - 1e-12).all() and (res.q <= arm6.upper + 1e-12).all()
    assert (res.pos_err <= ik.POS_TOL).all() and (res.rot_err <= ik.ROT_TOL).all()
    assert np.isfinite(res.manipulability).all() and (res.manipulability > 0).all()
    assert (res.min_distance >= 0).all()
    one = res.result(3)
    assert isinstance(one, ik.IKResult) and one.ok and one.reason == '' and one.solutions >= 1


def test_single_solve_seeded_and_reproducible(arm6):
    _Q, T = targets(arm6, F.ARM6, 3, seed=1)
    a = ik.solve(arm6, 'tool', T[0], rng=5)
    b = ik.solve(arm6, 'tool', T[0], rng=5)
    assert a.ok and np.array_equal(a.q, b.q) and a.iterations == b.iterations
    assert a.pos_err <= 1e-4 and a.rot_err <= 1e-3


def test_rules_choose_among_solutions(arm6):
    Q, T = targets(arm6, F.ARM6, 12, seed=2)
    near = ik.solve_batch(arm6, 'tool', T, seeds=12, rule='nearest_home')
    most = ik.solve_batch(arm6, 'tool', T, seeds=12, rule='most_manipulable')
    assert near.ok.all() and most.ok.all()
    dn = np.linalg.norm(near.q - arm6.home, axis=1)
    dm = np.linalg.norm(most.q - arm6.home, axis=1)
    assert (dn <= dm + 1e-6).all()
    assert (most.manipulability >= near.manipulability - 1e-9).all()
    # nearest_q0: a target a few millimetres from q0's pose is solved next to q0 (smooth dragging)
    q0 = Q[0]
    T0 = arm6.link_pose(q0, 'tool')
    T0[:3, 3] += [0.003, -0.002, 0.001]
    r = ik.solve(arm6, 'tool', T0, q0=q0, rule='nearest_q0')
    h = ik.solve(arm6, 'tool', T0, q0=q0, rule='nearest_home')
    assert r.ok and h.ok and np.linalg.norm(r.q - q0) <= np.linalg.norm(h.q - q0) + 1e-9
    assert np.abs(r.q - q0).max() < 0.3


def test_position_only_for_a_five_joint_arm(arm5):
    Q, T = targets(arm5, F.ARM5, 20, seed=3)
    res = ik.solve_batch(arm5, 'tool', T)
    assert res.mode == 'position' and res.ok.all(), res.counts()
    for q, t in zip(res.q, T):
        assert np.linalg.norm(F.poe_fk(F.ARM5, q)[:3, 3] - t[:3, 3]) <= ik.POS_TOL
    # an orientation the arm cannot take at that position: the position is reached and the note says so
    t = T[0].copy()
    R = F._rot((1, 0, 0), 0.6)
    t[:3, :3] = t[:3, :3] @ R
    r = ik.solve(arm5, 'tool', t)
    assert r.ok and r.mode == 'position' and not r.orientation_ok
    assert r.pos_err <= ik.POS_TOL and r.rot_err > ik.ROT_TOL
    assert 'Position only' in r.note and '°' in r.note
    # asked for the full pose, the same target fails with the reason
    p = ik.solve(arm5, 'tool', t, mode='pose')
    assert not p.ok and p.reason == 'orientation unreachable'


def test_position_target_leaves_orientation_free(arm6):
    _Q, T = targets(arm6, F.ARM6, 2, seed=4)
    r = ik.solve(arm6, 'tool', T[0][:3, 3])
    assert r.ok and r.mode == 'position' and np.isnan(r.rot_err)
    assert np.linalg.norm(arm6.link_pose(r.q, 'tool')[:3, 3] - T[0][:3, 3]) <= ik.POS_TOL


def test_reasons(tmp_path, arm6):
    slide = mj(tmp_path, F.xyz_pair_mjcf('<geom type="sphere" size="0.1"/>',
                                         '<geom type="sphere" size="0.07" pos="0.3 0 0"/>'))
    T = np.tile(np.eye(4), (4, 1, 1))
    T[0, :3, 3] = [10.0, 0, 0]                  # beyond every joint's travel
    T[1, :3, 3] = [1.5, 0, 0]                   # x slide range is ±1
    T[2, :3, 3] = [-0.3, 0, 0]                  # the second sphere on the base's: the only solution overlaps
    T[3, :3, 3] = [0.5, 0.2, -0.3]              # reachable
    res = ik.solve_batch(slide, 'sz', T)
    assert res.reason == ['out of reach', 'joint limit', 'self-collision', ''], res.reason
    assert res.ok.tolist() == [False, False, False, True]
    assert res.counts() == {'ok': 1, 'out of reach': 1, 'joint limit': 1, 'self-collision': 1}
    assert res.min_distance[2] < 0                          # the closest solution's overlap is reported
    free = ik.solve_batch(slide, 'sz', T[2:3], collision=False)
    assert free.ok[0]
    _Q, Tt = targets(arm6, F.ARM6, 1, seed=6)
    r = ik.solve(arm6, 'tool', Tt[0], max_iter=1, retry_seeds=0, q0=arm6.home, seeds=1)
    assert not r.ok and r.reason == 'no convergence'
    assert set(res.reason) - {''} <= set(ik.REASONS)


def test_joints_off_the_chain_are_held(tmp_path):
    text = F.arm_mjcf(F.ARM6).replace(
        '<body name="tool"', '<body name="finger" pos="0 0.02 0"><joint name="finger" type="slide" axis="0 1 0" '
        'range="0 0.04"/><geom type="box" size="0.005 0.005 0.01" contype="0" conaffinity="0"/></body>'
        '<body name="tool"')
    model = mj(tmp_path, text)
    k = model.joint_names.index('finger')
    assert model.nq == 7
    assert M.chain_mask(model, 'tool').tolist() == [True] * 6 + [False]
    q = model.home.copy()
    q[:6] = [0.3, 0.4, 0.5, 0.2, -0.6, 0.1]
    q0 = model.home.copy()
    q0[k] = 0.03
    res = ik.solve(model, 'tool', model.link_pose(q, 'tool'), q0=q0)
    assert res.ok and res.mode == 'pose' and res.q[k] == pytest.approx(0.03)


def test_bad_inputs_are_explained(arm6):
    with pytest.raises(ValueError, match='rule'):
        ik.solve(arm6, 'tool', np.eye(4), rule='best')
    with pytest.raises(ValueError, match='mode'):
        ik.solve(arm6, 'tool', np.eye(4), mode='fast')
    with pytest.raises(ValueError, match='4 × 4'):
        ik.solve_batch(arm6, 'tool', np.zeros((2, 5)))
    with pytest.raises(KeyError):
        ik.solve(arm6, 'nope', np.eye(4))


def test_progress_and_cancel(arm6):
    _Q, T = targets(arm6, F.ARM6, 6, seed=7)
    seen = []
    res = ik.solve_batch(arm6, 'tool', T, progress=seen.append)
    assert res.ok.all() and seen[-1] == 1.0 and seen == sorted(seen)
    res = ik.solve_batch(arm6, 'tool', T, cancel=lambda: True)
    assert not res.ok.any() and set(res.reason) == {'no convergence'}


def test_sample_so100_is_solved_position_only():
    samples = pytest.importorskip('dexvantage.core.robot.samples',
                                  reason='the robot samples module does not import here')
    from dexvantage.core.robot import get_model
    try:
        p = samples.sample_project('so100')
    except Exception as e:                                   # noqa: BLE001 — the sample is another phase's
        pytest.skip(f'so100 sample not available: {e}')
    r = p.robots[0]
    m = get_model(p, r.uid)
    link = r.gripper_frame or m.link_names[-1]
    rng = np.random.default_rng(8)
    Q = rng.uniform(m.lower, m.upper, (10, m.nq))
    Q = Q[[not C.self_collision(m, q).in_collision for q in Q]]
    T = np.stack([m.link_pose(q, link) for q in Q])
    res = ik.solve_batch(m, link, T)
    assert res.mode == 'position' and res.ok.all(), res.counts()
    back = np.stack([m.link_pose(q, link) for q in res.q])
    assert np.linalg.norm(back[:, :3, 3] - T[:, :3, 3], axis=1).max() <= ik.POS_TOL


# ---------------------------------------------------------------------------------------------- manipulability
def test_manipulability_planar_2r_closed_form(tmp_path):
    model = mj(tmp_path, F.planar_2r_mjcf())
    for q2 in (0.0, 0.3, np.pi / 2, -2.0):
        q = np.array([0.7, q2])
        J = model.jacobian(q, 'tip')[:2]
        assert M.yoshikawa(J) == pytest.approx(abs(F.L1_2R * F.L2_2R * np.sin(q2)), abs=1e-12)
    m = M.measures(model, [0.2, np.pi / 2], 'tip', part='position')
    J3 = model.jacobian([0.2, np.pi / 2], 'tip')[:3]
    assert m['yoshikawa'] == 0.0                                 # 3 x 2: no motion along z
    assert m['condition'] == pytest.approx(np.linalg.cond(J3))   # numpy's convention: over min(m, n) values
    Q = np.c_[np.zeros(5), np.linspace(0.1, 3.0, 5)]
    b = M.measures_batch(model, Q, 'tip')
    assert b['yoshikawa'].shape == (5,) and b['limit_distance'].shape == (5, 2)


def test_manipulability_measures():
    J = np.diag([2.0, 1.0, 0.5])
    assert M.yoshikawa(J) == pytest.approx(1.0)
    assert M.condition(J) == pytest.approx(4.0) and M.inverse_condition(J) == pytest.approx(0.25)
    assert M.condition(J) == pytest.approx(np.linalg.cond(J))
    assert M.yoshikawa(np.ones((6, 5))) == 0.0                 # more rows than joints
    S = np.stack([J, 2 * J])
    assert np.allclose(M.yoshikawa(S), [1.0, 8.0]) and np.allclose(M.condition(S), [4.0, 4.0])
    sing = np.array([[1.0, 0.0], [0.0, 0.0]])
    assert M.condition(sing) == np.inf and M.inverse_condition(sing) == 0.0 and M.yoshikawa(sing) == 0.0
    lo, hi = np.array([-1.0, 0.0, -np.inf]), np.array([1.0, 2.0, np.inf])
    d = M.limit_distance([0.0, 2.0, 5.0], lo, hi)
    assert d.tolist() == [1.0, 0.0, 1.0]
    assert M.limit_distance([0.5, 0.5, 0.0], lo, hi)[:2].tolist() == [0.5, 0.5]
    assert M.min_limit_distance([0.5, 1.5, 0.0], lo, hi) == pytest.approx(0.5)
    with pytest.raises(ValueError, match='part'):
        M._rows(J, 'all')


# ---------------------------------------------------------------------------------------------- collision
def test_sphere_distance_and_batch(tmp_path):
    model = mj(tmp_path, F.xyz_pair_mjcf('<geom type="sphere" size="0.1"/>',
                                         '<geom type="sphere" size="0.05" pos="0.3 0 0"/>'))
    r = C.self_collision(model, [0.0, 0.0, 0.0])
    assert not r.in_collision and r.min_distance == pytest.approx(0.15, abs=1e-12)
    assert r.pairs == [('base', 'sz', pytest.approx(0.15, abs=1e-12))]
    r = C.self_collision(model, [-0.2, 0.0, 0.0])
    assert r.in_collision and r.min_distance == pytest.approx(-0.05, abs=1e-12) and r.pairs[0][:2] == ('base', 'sz')
    r = C.self_collision(model, [0.0, 0.0, 0.0], margin=0.2)
    assert r.in_collision                                       # nearer than the margin
    far = C.self_collision(model, [0.9, 0.9, 0.9])
    assert far.min_distance == C.DIST_MAX and not far.in_collision and far.pairs == []
    hit, d, pairs = C.self_collision_batch(model, [[0, 0, 0], [-0.2, 0, 0]])
    assert hit.tolist() == [False, True] and d == pytest.approx([0.15, -0.05], abs=1e-12) and len(pairs) == 2


def test_adjacent_links_are_not_checked(arm6):
    pairs = C.checked_pairs(arm6)
    adjacent = {('base', 'link1'), ('link1', 'link2'), ('link2', 'link3'), ('link3', 'link4'), ('link4', 'link5'),
                ('link5', 'link6')}
    assert pairs and not (set(pairs) & adjacent)
    assert ('base', 'link2') in pairs
    r = C.self_collision(arm6, arm6.home)
    assert not r.in_collision and r.min_distance > 0


def test_model_exclusions_and_explicit_pairs(tmp_path):
    a = '<geom name="ga" type="sphere" size="0.1"/>'
    b = '<geom name="gb" type="sphere" size="0.05" pos="0.3 0 0" contype="2" conaffinity="2"/>'
    model = mj(tmp_path / 'masks', F.xyz_pair_mjcf(a, b))
    assert C.checked_pairs(model) == []                         # contype/conaffinity say these never touch
    text = F.xyz_pair_mjcf(a, b.replace(' contype="2" conaffinity="2"', '')).replace(
        '</mujoco>', '<contact><exclude body1="base" body2="sz"/></contact></mujoco>')
    assert C.checked_pairs(mj(tmp_path / 'excl', text)) == []
    adj = ('<mujoco><worldbody><body name="b0"><geom name="g0" type="sphere" size="0.1"/>'
           '<body name="b1"><joint type="slide" axis="1 0 0" range="-1 1"/><geom name="g1" type="sphere" size="0.05" '
           'pos="0.3 0 0"/></body></body></worldbody><contact><pair geom1="g0" geom2="g1"/></contact></mujoco>')
    m = mj(tmp_path / 'pair', adj)
    assert C.checked_pairs(m) == [('b0', 'b1')]                 # adjacent, but the file names the pair
    assert C.self_collision(m, [0.0]).min_distance == pytest.approx(0.15, abs=1e-12)


def test_environment_distance_to_the_floor(tmp_path):
    text = ('<mujoco><worldbody><geom name="floor" type="plane" size="1 1 0.1"/><body name="base" pos="0 0 0.5">'
            '<body name="arm"><joint name="z" type="slide" axis="0 0 1" range="-1 1"/>'
            '<geom type="sphere" size="0.1"/></body></body></worldbody></mujoco>')
    model = mj(tmp_path, text)
    assert C.checked_pairs(model, 'environment') == [('arm', 'world')]
    r = C.environment_distance(model, [-0.3])
    assert r.min_distance == pytest.approx(0.1, abs=1e-12) and not r.in_collision
    hit, d, _p = C.environment_distance_batch(model, [[-0.3], [-0.45]])
    assert hit.tolist() == [False, True] and d == pytest.approx([0.1, -0.05], abs=1e-12)
    assert C.self_collision(model, [0.0]).pairs == []          # one link: nothing to collide with itself


# ---------------------------------------------------------------------------------------------- validation
def test_ik_checks_pass():
    from dexvantage.validation import REFERENCES, suite
    suite.load_checks()
    specs = [s for s in suite.CHECKS if s.fn.__module__.endswith('checks_ik')]
    assert {s.id for s in specs} == {'kinematics.ik_round_trip', 'kinematics.ik_position_only',
                                     'kinematics.manipulability_2r', 'kinematics.collision_spheres',
                                     'kinematics.collision_capsules'}
    ctx = suite.Ctx(None, None, None, suite.ValidationSettings(quick=True))
    try:
        for s in specs:
            c = suite.run_one(s, ctx)
            assert c.passed, (c.id, c.error, c.exception, c.detail)
            assert all(r in REFERENCES for r in c.refs) and c.model == 'kinematics'
    finally:
        ctx.close()


def test_segment_distance_reference():
    p = np.array
    assert F.segment_distance(p([0., 0, 0]), p([1., 0, 0]), p([0.5, 1, 0]), p([0.5, 2, 0])) == pytest.approx(1.0)
    assert F.segment_distance(p([0., 0, 0]), p([1., 0, 0]), p([2., 0, 0]), p([3., 0, 0])) == pytest.approx(1.0)
    assert F.segment_distance(p([0., 0, 0]), p([1., 0, 0]), p([0.5, -1, 1]), p([0.5, 1, 1])) == pytest.approx(1.0)
