"""Grasp sets (core/grasps.py) and the Tasks page (ui/pages/tasks.py): samplers, imports, trajectories, contact pads,
object points, the project round trip and the page's undo steps."""
import json
import math

import numpy as np
import pytest

from conftest import pump


# ====================================================================== helpers
@pytest.fixture(scope='module')
def ur5e():
    return _fresh('ur5e')


@pytest.fixture(scope='module')
def so100():
    return _fresh('so100')


def _fresh(name):
    from dexvantage.core.robot.samples import sample_project
    p = sample_project(name)
    p.tasks.grasp_sets = []  # the sample's own grasp set and policy left out
    return p


def _R(g):
    from dexvantage.core.frames import quat_to_matrix
    return quat_to_matrix(g['pose']['quaternion'])


def _tcp(geom, g):
    return _R(g) @ geom['tcp'] + np.asarray(g['pose']['position'])


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding='utf-8')
    return str(p)


# ====================================================================== contact pads and the gripper geometry
def test_ur5e_pads_are_the_inner_faces_of_the_2f85_pad_boxes(ur5e):
    from dexvantage.core import grasps as GR
    r = ur5e.robots[0]
    pt = GR.contact_patches(ur5e, r.uid)
    assert set(pt) == {'gripper_right_pad', 'gripper_left_pad'}
    R, L = pt['gripper_right_pad'], pt['gripper_left_pad']
    assert R['points'].shape == (GR.PAD_POINTS, 3)
    # the inner faces face each other: right's normal points at left's points and vice versa
    d = L['points'].mean(0) - R['points'].mean(0)
    assert np.all(R['normals'] @ d > 0) and np.all(L['normals'] @ -d > 0)
    # one plane per pad (a face of the box), 22 mm by 37.5 mm (the Menagerie pad boxes, two per pad)
    for p in (R, L):
        n = p['normals'][0]
        assert np.allclose(p['normals'], n, atol=1e-6)
        assert np.ptp(p['points'] @ n) < 1e-6
        ext = np.sort(np.ptp(p['points'], axis=0))
        assert ext[-1] < 0.0376 and ext[-2] < 0.0221
    assert 'collision geoms' in R['rule']


def test_so100_pads_are_the_jaws_inner_faces(so100):
    from dexvantage.core import grasps as GR
    r = so100.robots[0]
    pt = GR.contact_patches(so100, r.uid)
    assert set(pt) == {'Fixed_Jaw', 'Moving_Jaw'}
    nf, nm = pt['Fixed_Jaw']['normals'].mean(0), pt['Moving_Jaw']['normals'].mean(0)
    assert nf @ nm < -0.95                                  # they face each other
    d = pt['Moving_Jaw']['points'].mean(0) - pt['Fixed_Jaw']['points'].mean(0)
    assert nf @ d > 0
    g = GR.gripper_geometry(so100, r.uid)
    assert not g['assumed'] and 0.01 < g['width_m'] < 0.04
    assert g['approach'] @ np.array([0.0, -1.0, 0.0]) > 0.99   # along the jaws (the link's -y axis)


def test_contact_points_ride_with_the_links(ur5e):
    from dexvantage.core import grasps as GR
    from dexvantage.core.robot.model import get_model
    r = ur5e.robots[0]
    m = get_model(ur5e, r.uid)
    q = m.home.copy()
    q[:6] += np.array([0.3, -0.4, 0.5, 0.2, -0.6, 0.9])
    home, moved = GR.contact_points(ur5e, r.uid), GR.contact_points(ur5e, r.uid, q)
    T0, T1 = m.fk(m.home), m.fk(q)
    for k in home:
        M = T1[k] @ np.linalg.inv(T0[k])
        assert np.allclose(home[k] @ M[:3, :3].T + M[:3, 3], moved[k], atol=1e-12)
    # a {robot uid: q} mapping means the same
    assert np.allclose(GR.contact_points(ur5e, r.uid, {r.uid: q})['gripper_left_pad'], moved['gripper_left_pad'])


def test_ur5e_tcp_is_on_the_pinch_axis(ur5e):
    from dexvantage.core import grasps as GR
    g = GR.gripper_geometry(ur5e, ur5e.robots[0].uid)
    assert g['frame'] == 'gripper_base'
    assert np.allclose(g['approach'], [0, 0, 1], atol=1e-6)
    assert np.linalg.norm(g['tcp'][:2]) < 1e-6 and 0.11 < g['tcp'][2] < 0.145
    assert abs(g['closing'] @ g['approach']) < 1e-12 and g['width_m'] == pytest.approx(0.0854, abs=1e-3)


def test_tagged_box_pads_take_the_face_towards_the_other_pad():
    from dexvantage.core import grasps as GR
    p = _fresh('ur5e')
    r = p.robots[0]
    r.contact_pads = [{'link': 'gripper_right_pad', 'name': 'right', 'box': {'center': [0, -0.003, 0.02],
                                                                              'size': [0.02, 0.006, 0.03]}},
                      {'link': 'gripper_left_pad', 'name': 'left', 'box': {'center': [0, -0.003, 0.02],
                                                                            'size': [0.02, 0.006, 0.03]}}]
    pt = GR.contact_patches(p, r.uid)
    assert set(pt) == {'right', 'left'}
    assert len(pt['right']['points']) == GR.PAD_GRID ** 2
    d = pt['left']['points'].mean(0) - pt['right']['points'].mean(0)
    assert np.all(pt['right']['normals'] @ d > 0) and 'towards the other pads' in pt['right']['rule']


def test_a_robot_without_fingertips_gets_assumed_axes():
    from dexvantage.core import grasps as GR
    p = _fresh('ur5e')
    r = p.robots[0]
    r.fingertip_frames = []
    assert GR.contact_points(p, r.uid) == {}
    g = GR.gripper_geometry(p, r.uid)
    assert g['assumed'] and np.allclose(g['tcp'], 0) and np.allclose(g['approach'], [0, 0, 1])
    gs = GR.sample(p, r.uid, ['top_down'], 5, 0)
    assert any('assumed' in n.lower() for n in gs['settings']['notes'])


# ====================================================================== the samplers
def test_sampling_is_seeded_and_round_robin(ur5e):
    from dexvantage.core import grasps as GR
    uid = ur5e.robots[0].uid
    a = GR.sample(ur5e, uid, ['top_down', 'side', 'angled'], 30, 7)
    b = GR.sample(ur5e, uid, ['top_down', 'side', 'angled'], 30, 7)
    c = GR.sample(ur5e, uid, ['top_down', 'side', 'angled'], 30, 8)
    assert a['poses'] == b['poses'] and a['poses'] != c['poses']
    assert GR.counts(a) == {'top_down': 10, 'side': 10, 'angled': 10}
    assert [p['kind'] for p in a['poses'][:3]] == ['top_down', 'side', 'angled']
    assert a['source'] == 'sampled' and a['frame'] == 'gripper_base' and a['robot_uid'] == uid
    assert json.loads(json.dumps(a)) == a                  # JSON-able as it is
    s = a['settings']
    assert s['rng_seed'] == 7 and set(s['defaults']) == set(GR.DEFAULTS) and s['gripper']['frame'] == 'gripper_base'


def test_volume_grasps_follow_their_closed_forms(ur5e):
    from dexvantage.core import grasps as GR
    r = ur5e.robots[0]
    vol = ur5e.workspace.volumes[0]
    lo, hi = (np.asarray(x) for x in vol.bounds())
    g = GR.gripper_geometry(ur5e, r.uid)
    gs = GR.sample(ur5e, r.uid, list(GR.KINDS[:3]), 300, 1, spread_deg=30.0, angled_tilt_deg=30.0, height_m=0.05)
    base = np.asarray(r.base.position)
    for x in gs['poses']:
        t = _tcp(g, x)
        a = np.asarray(x['approach'])
        assert np.allclose(_R(x) @ g['approach'], a, atol=1e-12)
        assert t[2] == pytest.approx(0.05, abs=1e-12) and np.all(t[:2] >= lo[:2]) and np.all(t[:2] <= hi[:2])
        c = _R(x) @ g['closing']
        assert abs(c[2]) < 1e-12 and abs(c @ a) < 1e-12
        if x['kind'] == 'top_down':
            assert np.allclose(a, [0, 0, -1])
        else:
            tilt = 90.0 if x['kind'] == 'side' else 30.0
            assert math.degrees(math.acos(-a[2])) == pytest.approx(tilt, abs=1e-9)
            az, az0 = math.atan2(a[1], a[0]), math.atan2(t[1] - base[1], t[0] - base[0])
            assert abs((az - az0 + math.pi) % (2 * math.pi) - math.pi) <= math.radians(30.0) + 1e-12
        assert x['width_m'] is None and x['object_uid'] == ''
    assert set(gs['settings']['defaults']) == set(GR.DEFAULTS) - {'spread_deg', 'angled_tilt_deg', 'height_m'}


def test_support_surface_inside_the_volume_lifts_the_grasps():
    from dexvantage.core import grasps as GR
    p = _fresh('ur5e')
    vol = p.workspace.volumes[0]
    c = np.asarray(vol.center)
    p.workspace.surfaces.append({'uid': 'shelf', 'name': 'shelf', 'z': 0.10, 'size': [0.1, 0.4],
                                 'center': [c[0] - 0.2, c[1]]})
    g = GR.gripper_geometry(p, p.robots[0].uid)
    gs = GR.sample(p, p.robots[0].uid, ['top_down'], 400, 3)
    T = np.array([_tcp(g, x) for x in gs['poses']])
    on = np.abs(T[:, 0] - (c[0] - 0.2)) <= 0.05
    assert on.any() and (~on).any()
    assert np.allclose(T[on, 2], 0.13) and np.allclose(T[~on, 2], 0.03)


def test_object_grasps_land_on_the_objects():
    from dexvantage.core import grasps as GR
    from dexvantage.core.frames import Pose
    from dexvantage.core.project import TaskObject
    p = _fresh('ur5e')
    c = np.asarray(p.workspace.volumes[0].center)
    box = TaskObject(name='Block', shape='box', size=(0.04, 0.08, 0.06), pose=Pose(c + [0.05, 0.0, -0.12]))
    far = TaskObject(name='Elsewhere', shape='box', size=(0.04, 0.04, 0.04), pose=Pose([3.0, 3.0, 0.02]))
    p.workspace.objects = [box, far]
    uid = p.robots[0].uid
    g = GR.gripper_geometry(p, uid)
    gs = GR.sample(p, uid, ['top_down', 'side'], 40, 5)
    assert {x['object_uid'] for x in gs['poses']} == {box.uid}
    top = box.pose.position[2] + 0.03
    for x in gs['poses']:
        t = _tcp(g, x)
        if x['kind'] == 'top_down':
            assert t[2] == pytest.approx(top - min(g['pad_length_m'] / 2, 0.03), abs=1e-9)
            assert np.allclose(t[:2], box.pose.position[:2], atol=1e-9)
            # closes across the narrower side (x, 40 mm), within the yaw jitter
            c_w = _R(x) @ g['closing']
            assert abs(c_w[0]) >= math.cos(math.radians(GR.DEFAULTS['yaw_jitter_deg'])) - 1e-9
            assert 0.04 - 1e-9 <= x['width_m'] <= 0.04 * abs(c_w[0]) + 0.08 * abs(c_w[1]) + 1e-9
        else:
            assert np.allclose(t, box.pose.position, atol=1e-9)
    assert any('task object' in n for n in gs['settings']['notes'])


def test_handoff_grasps_point_at_the_partner():
    from dexvantage.core import grasps as GR
    from dexvantage.core.frames import Pose
    from dexvantage.core.project import Robot, Volume
    p = _fresh('ur5e')
    a = p.robots[0]
    b = Robot.from_dict(a.to_dict())
    b.uid, b.name, b.primary = 'partner0001', 'Second arm', False
    b.base = Pose([0.0, 1.2, 0.0])
    b.source.file = a.source.file                           # same stored files (keyed by uid prefix: copy them)
    for k, v in list(p.sources.items()):
        if k.startswith(a.uid + '/'):
            p.sources[b.uid + k[len(a.uid):]] = v
    b.source.file = b.uid + a.source.file[len(a.uid):]
    p.robots.append(b)
    hv = Volume(name='Handover', kind='handover', center=(0.0, 0.6, 0.4), size=(0.2, 0.2, 0.2))
    p.workspace.volumes.append(hv)
    g = GR.gripper_geometry(p, a.uid)
    gs = GR.sample(p, a.uid, ['handoff'], 50, 2)
    assert gs['settings']['partner_robot_uid'] == b.uid and gs['settings']['handover_uid'] == hv.uid
    lo, hi = (np.asarray(x) for x in hv.bounds())
    for x in gs['poses']:
        t, ap = _tcp(g, x), np.asarray(x['approach'])
        assert np.all(t >= lo - 1e-12) and np.all(t <= hi + 1e-12)
        assert abs(ap[2]) < 1e-12
        assert math.degrees(math.acos(ap @ [0, 1, 0])) <= GR.DEFAULTS['handoff_spread_deg'] + 1e-9


def test_sampler_errors_are_sentences(ur5e):
    from dexvantage.core import grasps as GR
    uid = ur5e.robots[0].uid
    with pytest.raises(GR.GraspError, match='handover zone'):
        GR.sample(ur5e, uid, ['handoff'], 5, 0)
    with pytest.raises(GR.GraspError, match='Choose the kinds'):
        GR.sample(ur5e, uid, ['sideways'], 5, 0)
    with pytest.raises(GR.GraspError, match='between 1 and'):
        GR.sample(ur5e, uid, ['top_down'], 0, 0)
    with pytest.raises(GR.GraspError, match='must be between 0 and 90'):
        GR.sample(ur5e, uid, ['angled'], 5, 0, angled_tilt_deg=120)
    with pytest.raises(GR.GraspError, match='not a sampler setting'):
        GR.sample(ur5e, uid, ['angled'], 5, 0, tilt=10)
    with pytest.raises(GR.GraspError, match='no such robot'):
        GR.sample(ur5e, 'nope', ['top_down'], 5, 0)
    p = _fresh('ur5e')
    p.workspace.volumes = []
    with pytest.raises(GR.GraspError, match='no task volume'):
        GR.sample(p, p.robots[0].uid, ['top_down'], 5, 0)


def test_sampled_grasps_are_reachable_by_the_ur5e(ur5e):
    """Not a promise of the samplers (they do not check reach), but the sample's defaults should give a usable set."""
    from dexvantage.core import grasps as GR
    from dexvantage.core.robot import ik
    from dexvantage.core.robot.model import get_model
    gs = GR.sample(ur5e, ur5e.robots[0].uid, ['top_down', 'angled'], 20, 4)
    link, tcp, T = GR.ik_targets(ur5e, gs)
    b = ik.solve_batch(get_model(ur5e, gs['robot_uid']), link, T, seeds=4, collision=False, tcp=tcp, rng=0)
    assert b.ok.sum() >= 18
    # the solved grasp puts the pads' centre on the target
    i = int(np.flatnonzero(b.ok)[0])
    cp = GR.contact_points(ur5e, gs['robot_uid'], b.q[i])
    centre = np.mean([v.mean(0) for v in cp.values()], axis=0)
    assert np.linalg.norm(centre - T[i][:3, 3]) < 2e-3


# ====================================================================== imports
CSV_OK = """# grasps for the test
id,x,y,z,qx,qy,qz,qw,ax,ay,az,kind,object
a,0.1,0.4,0.05,1,0,0,0,0,0,-1,top-down,
b,0.2,0.4,0.05,0,0,0,2,,,,,cup
"""


def test_import_csv(tmp_path, ur5e):
    from dexvantage.core import grasps as GR
    gs = GR.import_file(_write(tmp_path, 'mine.csv', CSV_OK))
    assert gs['source'] == 'imported' and gs['name'] == 'mine' and len(gs['poses']) == 2
    a, b = gs['poses']
    assert a['kind'] == 'top_down' and a['approach'] == [0.0, 0.0, -1.0]
    assert b['object_uid'] == 'cup' and b['pose']['quaternion'] == [0.0, 0.0, 0.0, 1.0]
    assert b['approach'] == [0.0, 0.0, 1.0] and b['kind'] == 'imported'       # the frame's z axis: no robot given
    assert any('normalised' in n for n in gs['settings']['notes'])
    # with a robot: its gripper frame and approach axis
    gr = GR.import_file(_write(tmp_path, 'mine.csv', CSV_OK), ur5e, ur5e.robots[0].uid)
    assert gr['frame'] == 'gripper_base' and gr['robot_uid'] == ur5e.robots[0].uid


def test_import_csv_semicolons_and_decimal_commas(tmp_path):
    from dexvantage.core import grasps as GR
    text = 'ID;X;Y;Z;QX;QY;QZ;QW;width\n1;0,1;0,2;0,3;0;0;0;1;0,04\n'
    gs = GR.import_file(_write(tmp_path, 'semi.csv', text))
    p = gs['poses'][0]
    assert p['pose']['position'] == [0.1, 0.2, 0.3] and p['width_m'] == 0.04 and p['id'] == '1'


@pytest.mark.parametrize('text, match', [
    ('', 'empty'),
    ('id,x,y,z,qx,qy,qz\n1,0,0,0,0,0,0\n', 'no column qw'),
    ('id,x,y,z,qx,qy,qz,qw\n1,0,0,0,0,0,0,1\n2,0,abc,0,0,0,0,1\n', 'Line 3, column y: "abc" is not a number'),
    ('id,x,y,z,qx,qy,qz,qw\n1,0,0,0,0,0,0,0\n', 'all zeros'),
    ('id,x,y,z,qx,qy,qz,qw,ax,ay,az\n1,0,0,0,0,0,0,1,0,,1\n', 'all three approach'),
    ('id,x,y,z,qx,qy,qz,qw,ax,ay,az\n1,0,0,0,0,0,0,1,0,0,0\n', 'zero length'),
    ('id,x,y,z,qx,qy,qz,qw,kind\n1,0,0,0,0,0,0,1,pinchy\n', 'kind "pinchy"'),
    ('id,x,y,z,qx,qy,qz,qw\n1,0,0,0,0,0,0,1\n1,0,0,0,0,0,0,1\n', 'id 1 is used by more than one'),
    ('id,x,y,z,qx,qy,qz,qw\n1,0,0,0,0,0,1\n', 'Line 2 has 7 values'),
    ('id,x,y,z,qx,qy,qz,qw\n', 'no rows'),
    ('id,x,x,y,z,qx,qy,qz,qw\n', 'column x twice'),
    ('id,x,y,z,qx,qy,qz,qw\n,0,0,0,0,0,0,1\n', 'id is blank'),
])
def test_import_csv_errors_name_the_row(tmp_path, text, match):
    from dexvantage.core import grasps as GR
    with pytest.raises(GR.GraspError, match=match):
        GR.import_file(_write(tmp_path, 'bad.csv', text))


def test_import_warns_about_millimetres(tmp_path):
    from dexvantage.core import grasps as GR
    gs = GR.import_file(_write(tmp_path, 'mm.csv', 'id,x,y,z,qx,qy,qz,qw\n1,450,120,30,0,0,0,1\n'))
    assert any('millimetres' in n for n in gs['settings']['notes'])


def test_import_json(tmp_path):
    from dexvantage.core import grasps as GR
    rows = [{'id': 'g1', 'position': [0.1, 0.2, 0.3], 'quaternion': [0, 0, 0, 1], 'approach': [0, 0, -2],
             'kind': 'side', 'object': 'mug', 'width_m': 0.05},
            {'pose': {'position': [0.0, 0.1, 0.2], 'quaternion': [0, 0, 1, 0]}, 't': 1.5},
            {'id': 'g3', 'x': 0, 'y': 0, 'z': 0, 'qx': 0, 'qy': 0, 'qz': 0, 'qw': 1}]
    gs = GR.import_file(_write(tmp_path, 'g.json', json.dumps(rows)))
    a, b, c = gs['poses']
    assert a['approach'] == [0.0, 0.0, -1.0] and a['kind'] == 'side' and a['object_uid'] == 'mug'
    assert b['id'] == '2' and b['t'] == 1.5 and c['id'] == 'g3'
    # a whole grasp set record
    gs2 = GR.import_file(_write(tmp_path, 'set.json', json.dumps({'name': 'From the lab', 'poses': rows})))
    assert gs2['name'] == 'From the lab'


@pytest.mark.parametrize('doc, match', [
    ('{"x": 1}', 'no list of grasps'),
    ('[]', 'empty'),
    ('[1]', 'Item 1 is not an object'),
    ('[{"id": 1, "position": [0, 0], "quaternion": [0, 0, 0, 1]}]', 'Item 1: "position"'),
    ('[{"id": 1, "position": [0, 0, 0]}]', '"quaternion" must be'),
    ('[{"id": 1}]', 'Item 1 has no x'),
    ('[{"id": 1, "position": [0, 0, 0], "quaternion": [0, 0, 0, 1], "approach": 3}]', '"approach" must be'),
    ('[{"id": 1, "position": [0, "a", 0], "quaternion": [0, 0, 0, 1]}]', 'Item 1, column y'),
    ('[{"id": 1, ', 'not valid JSON'),
])
def test_import_json_errors(tmp_path, doc, match):
    from dexvantage.core import grasps as GR
    with pytest.raises(GR.GraspError, match=match):
        GR.import_file(_write(tmp_path, 'bad.json', doc))


def test_import_missing_file(tmp_path):
    from dexvantage.core import grasps as GR
    with pytest.raises(GR.GraspError, match='There is no file'):
        GR.import_file(str(tmp_path / 'nothing.csv'))


# ====================================================================== trajectories, pre-contact, objects
def _with_trajectory():
    from dexvantage.core.robot.model import get_model
    p = _fresh('so100')
    r = p.robots[0]
    m = get_model(p, r.uid)
    t = np.linspace(0, 2, 21)
    Q = m.home + np.outer(np.sin(t), np.r_[0.3, -0.2, 0.2, 0.1, 0.5, 0.0])
    p.tasks.trajectories.append({'uid': 'traj1', 'name': 'Pick', 'robot_uid': r.uid, 'joints': m.joint_names,
                                 't': t.tolist(), 'q': Q.tolist()})
    return p, m, t, Q


def test_from_trajectory():
    from dexvantage.core import grasps as GR
    p, m, t, Q = _with_trajectory()
    gs = GR.from_trajectory(p, 'traj1', every=5)
    assert gs['source'] == 'trajectory' and [x['t'] for x in gs['poses']] == [t[0], t[5], t[10], t[15], t[20]]
    g = GR.gripper_geometry(p, gs['robot_uid'])
    for x, k in zip(gs['poses'], (0, 5, 10, 15, 20)):
        T = m.link_pose(m.clip(Q[k]), 'Fixed_Jaw')          # played poses are clipped to the limits
        assert np.allclose(GR.grasp_matrix(x), T, atol=1e-9)
        assert np.allclose(x['approach'], T[:3, :3] @ g['approach'], atol=1e-12)
        assert x['kind'] == 'trajectory'
    with pytest.raises(GR.GraspError, match='1 or more'):
        GR.from_trajectory(p, 'traj1', every=0)
    with pytest.raises(GR.GraspError, match='no such trajectory'):
        GR.from_trajectory(p, 'gone')


def test_pre_contact_poses_back_off_along_the_approach(ur5e):
    from dexvantage.core import grasps as GR
    g = GR.sample(ur5e, ur5e.robots[0].uid, ['angled'], 1, 0)['poses'][0]
    T0 = GR.grasp_matrix(g)
    out = GR.pre_contact_poses(g, [0.0, 0.02, 0.10])
    assert len(out) == 3 and np.allclose(out[0], T0)
    for d, T in zip((0.0, 0.02, 0.10), out):
        assert np.allclose(T[:3, :3], T0[:3, :3])
        assert np.allclose(T0[:3, 3] - T[:3, 3], d * np.asarray(g['approach']), atol=1e-15)


def test_object_points_lie_on_the_object():
    from dexvantage.core import grasps as GR
    from dexvantage.core.frames import Pose
    from dexvantage.core.project import Project, TaskObject
    p = Project('t')
    o = TaskObject(name='Can', shape='cylinder', size=(0.03, 0.12, 0), pose=Pose([0.2, 0.0, 0.06]))
    p.workspace.objects = [o]
    P = GR.object_points(p, o.uid, 2000, 3)
    assert P.shape == (2000, 3)
    rel = P - o.pose.position
    rad = np.hypot(rel[:, 0], rel[:, 1])
    side = np.abs(np.abs(rel[:, 2]) - 0.06) > 1e-9
    # on the 24-gon side (between its inscribed and circumscribed radius) or on a cap
    assert np.all(rad[side] <= 0.03 + 1e-9) and np.all(rad[side] >= 0.03 * math.cos(math.pi / 24) - 1e-9)
    assert np.all(rad[~side] <= 0.03 + 1e-9)
    assert np.array_equal(P, GR.object_points(p, o.uid, 2000, 3))
    with pytest.raises(GR.GraspError, match='no such task object'):
        GR.object_points(p, 'nope', 5)


# ====================================================================== the project record
def test_sets_round_trip_through_the_project_file(tmp_path):
    from dexvantage.core import grasps as GR
    from dexvantage.core.project import Project
    p = _fresh('ur5e')
    gs = GR.sample(p, p.robots[0].uid, ['top_down', 'side'], 12, 1)
    uid = GR.add_set(p, gs)
    uid2 = GR.add_set(p, gs)                                # the same record again: a new uid and a unique name
    assert uid2 != uid and GR.get_set(p, uid2)['name'].endswith('(2)')
    path = str(tmp_path / 'g.dexv')
    p.save(path)
    q = Project.load(path)
    assert GR.get_set(q, uid) == GR.get_set(p, uid)
    assert GR.remove_set(q, uid) and GR.get_set(q, uid) is None and not GR.remove_set(q, uid)


def test_normalise_set_refuses_bad_records():
    from dexvantage.core import grasps as GR
    with pytest.raises(GR.GraspError):
        GR.normalise_set({'source': 'guessed', 'poses': []})
    with pytest.raises(GR.GraspError, match='no gripper pose'):
        GR.normalise_set({'poses': [{'id': 1}]})


def test_validation_checks_pass():
    from dexvantage.validation import suite
    suite.load_checks()
    specs = [s for s in suite.CHECKS if s.id.startswith('grasp.') and s.id in (
        'grasp.pads_vs_mujoco_boxes', 'grasp.tcp_vs_pinch_site', 'grasp.sampler_closed_form',
        'grasp.object_points_box')]
    assert len(specs) == 4
    ctx = suite.Ctx(None, None, None, suite.ValidationSettings(quick=True))
    try:
        for s in specs:
            res = suite.run_one(s, ctx)
            assert res.passed, (s.id, res.error, res.detail)
    finally:
        ctx.close()


# ====================================================================== the Tasks page
@pytest.fixture
def page(win, app):
    assert win.open_sample('ur5e')
    win.store.edit('tasks', lambda p: setattr(p.tasks, 'grasp_sets', []), 'Removed the sample\'s grasp set')
    win.store.undo.setClean()
    pump(app, 0.02)
    win.open_page('tasks')
    pump(app, 0.02)
    return win.pages['tasks']


def test_tasks_page_is_registered(win):
    from dexvantage.ui.pages import PAGES
    assert 'tasks' in PAGES and 'tasks' in win.pages
    assert 'page_tasks' in win.actions_by_id


def test_page_adds_a_sampled_set_as_one_undo_step(page, win, app):
    from dexvantage.core import grasps as GR
    p = win.store.project
    page.set_form(kinds=['top_down', 'angled'], n=40, rng_seed=3)
    page.preview()
    assert page.preview_set is not None and not GR.sets(p)            # preview adds nothing
    assert page.overlays(win.viewport)
    n0 = win.store.undo.count()
    uid = page.apply()
    assert uid and win.store.undo.count() == n0 + 1
    gs = GR.get_set(win.store.project, uid)
    assert len(gs['poses']) == 40 and GR.counts(gs) == {'top_down': 20, 'angled': 20}
    assert page.tree.topLevelItemCount() == 1 and page.selected_uid() == uid
    assert page.head.figure.text() == '40'
    segs = page.overlays(win.viewport)
    assert sum(len(s.pts) for s in segs) == 40 * 4
    win.store.undo.undo()
    pump(app, 0.02)
    assert not GR.sets(win.store.project) and page.tree.topLevelItemCount() == 0
    win.store.undo.redo()
    pump(app, 0.02)
    assert GR.get_set(win.store.project, uid) is not None


def test_page_import_trajectory_delete_and_export(page, win, app, tmp_path):
    from dexvantage.core import grasps as GR
    uid = page.import_path(_write(tmp_path, 'lab.csv', CSV_OK))
    assert uid and GR.get_set(win.store.project, uid)['frame'] == 'gripper_base'
    assert page.import_path(_write(tmp_path, 'bad.csv', 'id,x\n1,2\n')) is None
    assert 'no column' in page.msg.text()
    # a trajectory of the UR5e
    from dexvantage.core.robot.model import get_model
    r = win.store.project.robots[0]
    m = get_model(win.store.project, r.uid)
    win.store.edit('tasks', lambda p: p.tasks.trajectories.append(
        {'uid': 'tr1', 'name': 'Reach', 'robot_uid': r.uid, 'joints': m.joint_names[:6], 't': [0.0, 1.0, 2.0],
         'q': [list(m.home[:6]), list(m.home[:6] + 0.1), list(m.home[:6] + 0.2)]}), 'Add a trajectory')
    pump(app, 0.02)
    tuid = page.add_from_trajectory('tr1', 2)
    assert len(GR.get_set(win.store.project, tuid)['poses']) == 2
    page.select_set(uid)
    out = page.export(str(tmp_path / 'set.json'))
    doc = json.loads(open(out, encoding='utf-8').read())
    assert doc['grasp_set']['uid'] == uid and 'provenance' in doc
    n0 = win.store.undo.count()
    assert page.delete_set(uid)
    assert win.store.undo.count() == n0 + 1 and GR.get_set(win.store.project, uid) is None
    win.store.undo.undo()
    pump(app, 0.02)
    assert GR.get_set(win.store.project, uid) is not None


def test_page_explains_what_it_cannot_do(page, win, app):
    page.set_form(kinds=['handoff'])
    assert 'handover zone' in page.form_note.text() and not page.b_apply.isEnabled()
    page.set_form(kinds=[])
    assert 'Tick at least one' in page.form_note.text()
    assert page.apply() is None and 'Choose the kinds' in page.msg.text()


def test_page_camview_overlay(page, win):
    from dexvantage.core.frames import optical_pose
    from dexvantage.core.optics import CameraModel
    from dexvantage.core.robot.samples import look_at
    page.set_form(kinds=['top_down'], n=10, rng_seed=1)
    page.apply()
    c = np.asarray(win.store.project.workspace.volumes[0].center)
    body = look_at(c + [0, 0, 0.8], c, up=(1.0, 0.0, 0.0))
    cam = CameraModel.from_fov(320, 240, 80.0)
    out = page.camview_overlays(cam, optical_pose(body))
    assert out and out[0][2] == 'Top-down pinch' and all(np.asarray(pl).shape[1] == 2 for pl in out[0][0])
