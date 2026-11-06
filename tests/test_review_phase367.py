"""Regression tests from the correctness review of Phases 3, 6 and 7: grasp IK targets, the grasp study's occluder
digest, the result cache's library hash, cable records and the hand-eye joint-state reader."""
import math

import numpy as np
import pytest

from dexvantage.core import cables as CB
from dexvantage.core import grasps as GR
from dexvantage.core import jobs
from dexvantage.core import library as LB
from dexvantage.core.calib import capture as CAP
from dexvantage.core.frames import Pose
from dexvantage.core.robot.samples import sample_project
from dexvantage.core.studies import grasp_visibility as GV


# ---------------------------------------------------------------------- grasps.ik_targets
def test_ik_targets_use_the_tcp_the_set_was_sampled_with():
    """A set sampled with a TCP override places that TCP on the grasp centre; the IK target must be that point, not
    the TCP measured from the pads (69 mm apart on the UR5e sample before the fix)."""
    p = sample_project('ur5e')
    r = p.robots[0]
    tcp = np.array([0.0, 0.0, 0.2])
    gs = GR.sample(p, r.uid, kinds=('top_down',), n=4, rng_seed=1, tcp=tcp)
    link, off, T = GR.ik_targets(p, gs)
    assert link == gs['frame'] and np.allclose(off[:3, 3], tcp)
    for g, Tt in zip(gs['poses'], T):
        Tg = GR.grasp_matrix(g)
        assert np.allclose(Tt[:3, 3], Tg[:3, :3] @ tcp + Tg[:3, 3], atol=1e-9)
    # without an override the measured TCP is used, as before
    gs2 = GR.sample(p, r.uid, kinds=('top_down',), n=2, rng_seed=1)
    _l, off2, _T = GR.ik_targets(p, gs2)
    assert np.allclose(off2[:3, 3], GR.gripper_geometry(p, r.uid)['tcp'])
    # an imported set (no recorded gripper) gets the measured TCP
    imp = GR.normalise_set({'name': 'x', 'source': 'imported', 'robot_uid': r.uid, 'frame': gs['frame'],
                            'poses': gs['poses']})
    _l, off3, _T = GR.ik_targets(p, imp)
    assert np.allclose(off3[:3, 3], GR.gripper_geometry(p, r.uid)['tcp'])


# ---------------------------------------------------------------------- the grasp study's occluder digest
def test_occluder_digest_follows_the_robot_base():
    """The GL backend keeps uploaded geometry by the occluders' digest. The digest held the joint vector but not the
    robot's base pose, so a moved robot at the same q reused its old triangles."""
    p = sample_project('ur5e')
    r = p.robots[0]
    st = GV.settings_of(p)

    def digest_and_tris():
        ev = GV.Evaluator(p, st, [], cameras=[])
        m = ev.robots[r.uid].model
        qd = {r.uid: m.home}
        occ = ev.occluders(ev.fk(qd), qd, None)
        return occ.digest(), occ.tris.copy()
    d1, t1 = digest_and_tris()
    d1b, _ = digest_and_tris()
    assert d1 == d1b                                       # the same scene, the same digest
    r.base = Pose([0.25, -0.1, 0.0], r.base.quaternion.copy())
    d2, t2 = digest_and_tris()
    assert t1.shape == t2.shape and not np.allclose(t1, t2)   # the triangles moved ...
    assert d2 != d1                                           # ... so the digest must change


# ---------------------------------------------------------------------- the result cache's library hash
def test_result_cache_key_follows_policy_and_host_entries(monkeypatch):
    """A study result depends on the policy entry's input size and crop rule and on a host entry's controllers; a
    custom entry changed in place under the same id must not return the old cached result."""
    p = sample_project('ur5e')
    assert p.policies and p.policies[0].library_id
    cache = jobs.ResultCache()
    k1 = cache.key('vla_tokens', p)
    e = LB.get(p.policies[0].library_id)
    assert e is not None
    monkeypatch.setitem(e.data, 'note', str(e.data.get('note', '')) + ' changed in place')
    assert cache.key('vla_tokens', p) != k1
    monkeypatch.undo()
    assert cache.key('vla_tokens', p) == k1
    host = next((x for x in LB.entries() if x.kind == 'host'), None) if hasattr(LB, 'entries') else None
    if host is None:
        pytest.skip('no host entry in the library')
    p.rig = {'hosts': [{'uid': 'h1', 'name': 'Host', 'library_id': host.id, 'controllers': []}], 'hubs': [],
             'links': []}
    k2 = cache.key('bandwidth', p)
    monkeypatch.setitem(host.data, 'note', str(host.data.get('note', '')) + ' changed in place')
    assert cache.key('bandwidth', p) != k2
    assert host.id in jobs.library_ids(p) and p.policies[0].library_id in jobs.library_ids(p)


# ---------------------------------------------------------------------- cables
def test_negative_wrap_radius_is_the_default():
    assert CB.clean_cable({'wrap_radius_m': -0.01})['wrap_radius_m'] is None
    assert CB.clean_cable({'wrap_radius_m': 0.0})['wrap_radius_m'] == 0.0
    assert CB.clean_cable({'wrap_radius_m': 0.02})['wrap_radius_m'] == 0.02
    assert CB.clean_cable({})['wrap_radius_m'] is None


# ---------------------------------------------------------------------- hand-eye joint states
def test_joint_states_in_degrees_leave_prismatic_joints_in_metres():
    js = CAP.read_joint_states('t,a,s\n0,90,0.02\n1,-45,0.01\n', ['a', 's'], unit='deg', prismatic=[False, True])
    assert np.allclose(js.Q, [[math.pi / 2, 0.02], [-math.pi / 4, 0.01]])
    js2 = CAP.read_joint_states('t,a,s\n0,90,0.02\n', ['a', 's'], unit='deg')     # no mask: every joint an angle
    assert np.allclose(js2.Q, [[math.pi / 2, math.radians(0.02)]])
    js3 = CAP.read_joint_states('t,a,s\n0,1.5,0.02\n', ['a', 's'], prismatic=[False, True])   # radians: unchanged
    assert np.allclose(js3.Q, [[1.5, 0.02]])
