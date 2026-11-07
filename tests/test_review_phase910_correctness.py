"""Correctness review of Phases 9 and 10 (tolerances, experiments, the optimiser and design agent, requirements and
--reproduce, the exporters' frame conventions, the canonical warp, project diff / merge / folders): regression tests
for what the review fixed, and independent checks of the conventions it confirmed.

Fixed here:
- a json: factor pointer could move, re-model, delete or unlock a locked sensor (core/factors.apply);
- the experiments and the optimiser took such a pointer as a factor (check_design, given_factors), and the optimiser's
  refused-design path returned a constraint vector of the wrong length;
- the tolerance Monte Carlo counted a draw stated only as a bound ("at least 1 ms") as meeting "< 2 ms", and
  compared without the scorecard's 10⁻⁹ tolerance (core/tolerance.meets now uses core/requirements.decide);
- the optimiser counted such an unsettled bound as a met constraint (objectives.violation);
- NSGA-II's crowding distance gave +inf to a design whose figure is not stated, keeping it ahead of stated designs;
- --reproduce of a scorecard made for a non-active variant re-ran it on the active one;
- the Plücker 'base' frame defaulted to the first robot instead of the primary one.
"""
import copy
import math

import numpy as np
import pytest

from dexvantage.core import factors as F
from dexvantage.core import jobs
from dexvantage.core import tolerance as TL
from dexvantage.core.frames import Pose, quat_from_rpy, rot_x, rot_y, rot_z
from dexvantage.core.project import Requirement, Variant, new_uid
from dexvantage.core.robot.samples import sample_project


@pytest.fixture
def so100():
    return sample_project('so100')


# ====================================================================== locked sensors and json: pointers
def _vi(p):
    return next(i for i, v in enumerate(p.variants) if v.uid == p.active_uid)


def test_json_pointer_cannot_change_a_locked_sensor(so100):
    p = so100
    s = p.variant.sensors[0]
    s.locked = True
    i = _vi(p)
    for path, value in ((f'json:/variants/{i}/sensors/0/mount/pose/position/0', 0.5),
                        (f'json:/variants/{i}/sensors/0/locked', False),
                        (f'json:/variants/{i}/sensors/0/model_id', 'something-else'),
                        (f'json:/variants/{i}/sensors', []),
                        ('json:/variants', [])):
        with pytest.raises(F.FactorError, match='locked'):
            F.apply(p, {path: value})
        assert F.locked_target(p, path) is s
    # the lock is the user's to lift: allow_locked (the tolerance Monte Carlo's) still works
    q = F.apply(p, {f'json:/variants/{i}/sensors/0/mount/pose/position/0': 0.5}, allow_locked=True)
    assert q.variant.sensors[0].mount.pose.position[0] == 0.5
    # an unlocked sensor and other parts of the project are unaffected
    q = F.apply(p, {f'json:/variants/{i}/sensors/1/mount/pose/position/0': 0.25, 'json:/name': 'R'})
    assert q.variant.sensors[1].mount.pose.position[0] == 0.25 and q.name == 'R'
    assert F.locked_target(p, f'json:/variants/{i}/sensors/1/name') is None
    assert F.locked_target(p, 'json:/name') is None
    assert F.locked_target(p, f'sensor:{s.uid}.x') is s


def test_experiment_and_optimiser_leave_out_json_into_locked(so100):
    from dexvantage.core import experiments as EX
    from dexvantage.core.optimise import rig as OR
    p = so100
    p.variant.sensors[0].locked = True
    path = f'json:/variants/{_vi(p)}/sensors/0/mount/pose/position/2'
    d = EX.load_design({'factors': [{'path': path, 'lo': 0.0, 'hi': 0.1}], 'responses': ['rig.camera_count'],
                        'method': 'lhs', 'n': 4}, p)
    assert any('locked' in m for m in EX.check_design(p, d))
    fs, notes = OR.given_factors(p, [{'path': path, 'lo': 0.0, 'hi': 0.1}])
    assert fs == [] and any('locked' in n for n in notes)


# ====================================================================== bounds decide as the scorecard does
def test_tolerance_meets_uses_the_scorecard_rule():
    from dexvantage.core import requirements as RQ
    assert TL.meets('<', 1.0, 2.0, 'at_least') is None          # at least 1 ms: does not settle < 2 ms
    assert TL.meets('<', 3.0, 2.0, 'at_least') is False          # at least 3 ms: fails < 2 ms
    assert TL.meets('>=', 96.0, 95.0, 'at_least') is True        # at least 96 %: passes >= 95 %
    assert TL.meets('<=', 1.0, 2.0, 'at_most') is True
    assert TL.meets('>=', 1.0, 2.0, 'at_most') is False
    for op, v, t in (('>=', 95 - 1e-12, 95), ('>', 95 + 1e-12, 95), ('==', 2.0 + 1e-12, 2.0), ('<', 2.0, 2.0)):
        assert TL.meets(op, v, t) == RQ.compare(v, op, t)[0]
    assert TL.meets('>=', None, 1.0) is None


def test_tolerance_shares_count_unsettled_bounds_as_not_met(so100):
    p = so100
    p.requirements = [Requirement(name='skew', metric='sync.skew_ms', op='<', threshold=2.0)]
    calls = {'n': 0}

    def ev(project, ids, ctx):
        calls['n'] += 1
        k = calls['n']
        # the nominal and odd draws: "at least 1 ms" (unsettled), even draws 0.5 ms exactly (a pass)
        row = {'value': 1.0, 'bound': 'at_least'} if k % 2 else {'value': 0.5}
        return {'sync.skew_ms': dict(row, title='skew', unit='ms', better='lower', status='ok')}
    doc = TL.monte_carlo(p, ['sync.skew_ms'], 8, 0, evaluator=ev)
    rq = doc['metrics']['sync.skew_ms']['requirements'][0]
    assert rq['of'] == 8 and rq['met'] == 4 and rq['undecided'] == 4 and rq['share'] == 0.5
    assert rq['nominal_meets'] is None


def test_optimiser_violation_of_an_unsettled_bound():
    from dexvantage.core.optimise import objectives as OB
    row = {'name': 'skew', 'metric': 'sync.skew_ms', 'op': '<', 'threshold': 2.0, 'unit': 'ms'}
    assert OB.violation(row, 1.0) == 0.0
    assert OB.violation(row, 1.0, 'at_least') == 1.0              # cannot be shown to hold
    assert OB.violation(row, 1.0, 'at_most') == 0.0
    assert OB.violation(row, 3.0, 'at_least') == pytest.approx(0.5)
    assert not OB.holds(row, 1.0, 'at_least') and OB.holds(row, 1.0)


# ====================================================================== NSGA-II
def test_crowding_gives_unstated_figures_no_advantage():
    from dexvantage.core.optimise import nsga2 as N
    F_ = np.array([[0.0, 1.0], [np.inf, 0.2], [1.0, 0.0], [0.4, 0.6], [0.7, 0.3]])
    cd = N.crowding_distance(F_)
    assert math.isinf(cd[0]) and math.isinf(cd[2])                # the stated extremes of objective 0
    assert np.isfinite(cd[1])                                     # unstated on objective 0: no +inf from it
    assert cd[1] < cd[3] and cd[1] < cd[4]
    # a front of finite points: Deb's distances, unchanged
    cd = N.crowding_distance(np.array([[0, 4], [1, 2], [2, 1], [4, 0]], float))
    assert math.isinf(cd[0]) and math.isinf(cd[3])
    assert cd[1] == pytest.approx(2 / 4 + 3 / 4) and cd[2] == pytest.approx(3 / 4 + 2 / 4)
    # survivor selection prefers the stated design of the same front
    keep = N.select(F_, np.zeros(len(F_)), 4)
    assert 1 not in keep


def test_constrained_domination_and_categorical_mutation():
    from dexvantage.core.optimise import nsga2 as N
    D = N.dominance_matrix([[5.0], [1.0], [0.0], [0.0]], [0.0, 1.0, 2.0, 2.0])
    assert D[0, 1] and D[0, 2] and D[1, 2] and not D[2, 3] and not D[3, 2] and not D[1, 0]
    rng = np.random.default_rng(1)
    x = rng.integers(0, 3, (3000, 2)).astype(float)
    y = N.categorical_mutation(x, [3, 1], 1.0, rng)
    assert (y[:, 0] != x[:, 0]).all() and (y[:, 1] == x[:, 1]).all() and set(np.unique(y[:, 0])) <= {0.0, 1.0, 2.0}
    steps = np.bincount(((y[:, 0] - x[:, 0]) % 3).astype(int), minlength=3)
    assert steps[0] == 0 and abs(steps[1] - steps[2]) < 200       # the other choices equally likely


# ====================================================================== --reproduce of a scorecard of a variant
def test_reproduce_scorecard_uses_its_variant(so100):
    from dexvantage.core import reproduce as RP
    from dexvantage.core import requirements as RQ
    p = so100
    v2 = Variant.from_dict(p.variant.to_dict())
    v2.uid, v2.name = new_uid(), 'one camera'
    for s in v2.sensors:
        s.uid = new_uid()
    v2.sensors[1].enabled = False
    p.variants.append(v2)
    reqs = [Requirement(name='cams', metric='rig.camera_count', op='<=', threshold=1.0)]
    card = RQ.check(p, reqs, variant=v2.uid)
    assert card['variant_uid'] == v2.uid and card['items'][0]['status'] == 'pass'
    assert RQ.check(p, reqs)['items'][0]['status'] == 'fail'      # the active variant has two
    doc = {'schema': RQ.SCHEMA, 'study': 'requirements', **card}
    again = RP.rerun(doc, p)
    assert again['variant_uid'] == v2.uid and again['items'][0]['value'] == card['items'][0]['value']


# ====================================================================== the result cache
def test_result_cache_key_follows_the_new_sections(so100):
    p = so100
    c = jobs.ResultCache(4)
    k0 = c.key('coverage', p)
    q = p.snapshot()
    q.requirements = [Requirement(name='x', metric='coverage.covered', op='>=', threshold=50.0)]
    assert c.key('coverage', q) != k0
    for sid, st in (('experiments', {'n': 9}), ('optimise', {'budget': 10}), ('tolerance', {'n': 3})):
        q = p.snapshot()
        q.analysis.studies[sid] = st
        assert c.key(sid, q) != c.key(sid, p)
    q = p.snapshot()
    q.variant.sensors[0].mount.tolerance.position_mm = 3.0       # the tolerance study reads it
    assert c.key('tolerance', q) != c.key('tolerance', p)


# ====================================================================== Plücker frame
def test_plucker_base_defaults_to_the_primary_robot(so100):
    from dexvantage.core import plucker as PL
    p = so100
    other = copy.deepcopy(p.robots[0])
    other.uid, other.primary = new_uid(), False
    other.base = Pose(np.array([0.5, 0.0, 0.0]), quat_from_rpy(0.0, 0.0, 1.0))
    p.robots[0].primary = True
    p.robots.insert(0, other)
    T, who = PL.frame_transform(p, 'base')
    assert who == p.robots[1].uid and np.allclose(T, np.eye(4))
    T, who = PL.frame_transform(p, 'base', other.uid)
    assert who == other.uid and np.allclose(T @ other.base.matrix(), np.eye(4))


# ====================================================================== conventions confirmed (independent checks)
def test_tolerance_draw_is_body_turn_and_parent_shift(so100):
    p = so100
    rng = np.random.default_rng(7)
    d = TL.draw(p, rng)
    q = TL.apply_draw(p, d)
    for s in p.variant.sensors:
        x = d[s.uid]
        B = s.mount.pose.matrix()
        R = B[:3, :3] @ rot_x(math.radians(x[3])) @ rot_y(math.radians(x[4])) @ rot_z(math.radians(x[5]))
        B2 = q.variant.sensor(s.uid).mount.pose.matrix()
        assert np.allclose(B2[:3, :3], R, atol=1e-12) and np.allclose(B2[:3, 3], B[:3, 3] + x[:3], atol=1e-12)
        assert np.allclose(B @ TL.delta_body(s.mount.pose, x), B2, atol=1e-12)
        b = TL.bounds(p)[s.uid]
        assert all(lo <= v <= hi for v, (lo, hi) in zip(x[:3], b['position_m']))
        assert all(lo <= v <= hi for v, (lo, hi) in zip(x[3:], b['rotation_deg']))


def test_calibration_draw_is_the_ba_parameterisation():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 6))
    C = A @ A.T * 1e-6
    T_im = np.eye(4)
    T_im[:3, :3] = rot_x(0.3) @ rot_z(-1.2)
    T_im[:3, 3] = [0.03, -0.02, 0.05]
    B = np.eye(4)
    B[:3, :3] = rot_y(0.7) @ rot_z(0.4)
    B[:3, 3] = [0.1, 0.2, 0.3]
    src = TL.Source('u', 'n', TL.CALIB, cov=C, R_B=B[:3, :3], T_im=T_im)
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    x = TL.draw_source(src, r1)
    y = TL._cov_root(C) @ r2.standard_normal(6)
    X0 = B @ T_im
    want = np.eye(4)
    want[:3, :3] = X0[:3, :3] @ TL.exp_so3(y[:3])
    want[:3, 3] = X0[:3, 3] + y[3:]
    assert np.allclose(B @ TL.delta_body(Pose.from_matrix(B), x) @ T_im, want, atol=1e-12)


def test_urdf_rpy_is_fixed_axis_xyz():
    from dexvantage.core.exporters.ros2 import rpy_of
    rng = np.random.default_rng(2)
    for r, pch, y in list(rng.uniform(-3, 3, (20, 3))) + [(0.3, math.pi / 2 - 1e-9, 1.0), (0.2, -math.pi / 2, -2.0)]:
        R = rot_z(y) @ rot_y(pch) @ rot_x(r)
        a, b, c = rpy_of(R)
        assert np.allclose(rot_z(c) @ rot_y(b) @ rot_x(a), R, atol=1e-9)


def _cam(w=640, h=480, fx=500.0, fy=510.0, cx=322.0, cy=236.0):
    from dexvantage.core.optics.lens import CameraModel
    return CameraModel(w, h, fx, fy, cx, cy, 'pinhole', ())


def test_canonical_homography_maps_source_to_canonical_on_the_plane():
    from dexvantage.core.exporters import canonical_warp as CW

    def T(R, t):
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = R, t
        return M
    # cameras above the contact plane z = 0 looking down-ish (optical z towards −z of the contact frame)
    down = rot_x(math.pi)
    src = CW.View('B', _cam(), T(down @ rot_x(0.2) @ rot_y(-0.1), [0.03, -0.02, 0.12]))
    can = CW.View('A', _cam(fx=450.0, fy=450.0, cx=319.5, cy=239.5), T(down @ rot_y(0.15), [-0.02, 0.01, 0.10]))
    w = CW.make_warp(src, can)
    assert w.H[2, 2] == pytest.approx(1.0) and w.d_B > 0
    rng = np.random.default_rng(3)
    P = np.c_[rng.uniform(-0.03, 0.03, (50, 2)), np.zeros(50)]                       # points on the plane
    def pix(view, X):
        Xc = (np.c_[X, np.ones(len(X))] @ np.linalg.inv(view.T_contact_cam).T)[:, :3]
        return np.c_[view.camera.fx * Xc[:, 0] / Xc[:, 2] + view.camera.cx,
                     view.camera.fy * Xc[:, 1] / Xc[:, 2] + view.camera.cy]
    uB, uA = pix(src, P), pix(can, P)
    h = np.c_[uB, np.ones(len(uB))] @ w.H.T
    assert np.allclose(h[:, :2] / h[:, 2:], uA, atol=1e-8)
    uvB, ok = CW.lut_at(w, uA)                                       # the remap reads the source pixel back
    assert ok.all() and np.allclose(uvB, uB, atol=1e-6)


def test_kalibr_chain_direction_and_usd_camera_axes():
    from dexvantage.core.exporters import calibration as CAL
    from dexvantage.core.exporters import isaac as IS

    class _C:
        def __init__(self, T, name):
            self.T_world_optical = T
            self.camera = _cam()
            self.key = f'{name}__rgb'
            self.sensor = type('S', (), {'name': name})()
    T0, T1 = np.eye(4), np.eye(4)
    T1[:3, :3], T1[:3, 3] = rot_z(0.3) @ rot_x(0.1), [0.2, -0.1, 0.05]
    lens = {'model': 'none', 'coeffs': []}
    text = CAL.kalibr_text([_C(T0, 'a'), _C(T1, 'b')], [lens, lens], type('P', (), {'name': 'p', 'variant':
                                                                                   type('V', (), {'name': 'v'})()})(),
                           'home')
    rows = [ln for ln in text.splitlines() if ln.startswith('  - [')]
    M = np.array([[float(x) for x in ln.split('[', 1)[1].rstrip(']').split(',')] for ln in rows])
    X0 = np.array([0.3, -0.2, 1.5, 1.0])                                            # a point in cam0
    assert np.allclose(M @ X0, np.linalg.inv(T1) @ T0 @ X0)                          # T_c1_c0 takes cam0 to cam1
    rec = {'T_parent_optical': T1.tolist()}
    U = IS._camera_local(rec)
    assert np.allclose(-U[:3, 2], T1[:3, 2]) and np.allclose(-U[:3, 1], T1[:3, 1]) and np.allclose(U[:3, 0],
                                                                                                    T1[:3, 0])


def test_merge_conflict_on_remove_against_change(so100):
    from dexvantage.core import projdiff as PD
    base = so100
    a, b = base.variant.sensors[0].uid, base.variant.sensors[1].uid
    ours = F.apply(base, {f'sensor:{a}.x': 0.3})
    theirs = F.apply(base, {f'sensor:{b}.z': 0.4})
    m = PD.merge3(base, ours, theirs)
    assert m.project.variant.sensor(a).mount.pose.position[0] == 0.3
    assert m.project.variant.sensor(b).mount.pose.position[2] == 0.4
    gone = base.snapshot()
    gone.variant.sensors = [s for s in gone.variant.sensors if s.uid != a]
    with pytest.raises(PD.MergeConflict) as e:
        PD.merge3(base, ours, gone)
    assert any('removed in theirs' in c for c in e.value.conflicts)
    both = PD.merge3(base, gone, gone.snapshot())
    assert [s.uid for s in both.project.variant.sensors] == [b]


def test_unpacked_folder_round_trip_is_exact(so100, tmp_path):
    from dexvantage.core import projfolder as PFD
    from dexvantage.core import provenance as PV
    from dexvantage.core.project import Project
    p = so100
    p.meshes['odd key/é'] = np.arange(6, dtype=np.float32).reshape(2, 3)
    p.meshes['odd_key_'] = np.arange(3, dtype=np.int64)
    p.meshes['Odd_key_'] = np.zeros(2, np.uint8)
    p.thumb = b'\x89PNG not really'
    folder = str(tmp_path / 'x.dexv.d')
    PFD.save_unpacked(p, folder)
    zip_path = str(tmp_path / 'x.dexv')
    PFD.pack(folder, zip_path)
    for q in (PFD.load_unpacked(folder), Project.load(zip_path)):
        assert PV.project_digest(q)['sha256'] == PV.project_digest(p)['sha256']
        assert sorted(q.meshes) == sorted(p.meshes)
        for k, v in p.meshes.items():
            assert q.meshes[k].dtype == v.dtype and np.array_equal(q.meshes[k], v)
        assert q.sources == p.sources and q.thumb == p.thumb
    assert PFD.save_unpacked(PFD.load_unpacked(folder), folder)['written'] == []
