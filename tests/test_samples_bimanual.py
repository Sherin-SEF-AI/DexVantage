"""The ALOHA 2, bimanual Panda and Unitree G1 samples (spec §11 --sample, §14.1, §14.4, §14.5).

Independent references: the bundled Menagerie files compiled directly by MuJoCo (aloha.xml with both arms in one
file, scene_parts.xml's cameras, panda.xml, g1_with_hands.xml with its free joint at "stand"), not through
DexVantage's importer, kinematics or mounts; Unitree's URDF figures as quoted in the sample's sources."""
import json
import os

import mujoco
import numpy as np
import pytest

from dexvantage.core.frames import quat_to_matrix
from dexvantage.core.project import Project
from dexvantage.core.robot import clear_cache, get_model
from dexvantage.core.robot import bimanual_samples as B
from dexvantage.core.robot import samples as S
from dexvantage.core.scenegeom import sensor_world_pose

NEW = ('aloha', 'panda-bimanual', 'g1')


class _Built(dict):
    def __missing__(self, name):
        self[name] = S.sample_project(name)
        return self[name]


@pytest.fixture(scope='module')
def built():
    return _Built()


def _mj(path):
    m = mujoco.MjModel.from_xml_path(path)
    return m, mujoco.MjData(m)


def _set(m, d, names, q):
    for n, v in zip(names, q):
        d.qpos[m.jnt_qposadr[m.joint(n).id]] = v


def _body_T(d, name):
    T = np.eye(4)
    T[:3, :3] = d.body(name).xmat.reshape(3, 3)
    T[:3, 3] = d.body(name).xpos
    return T


def _cam_T(d, name):
    T = np.eye(4)
    T[:3, :3] = d.cam(name).xmat.reshape(3, 3)
    T[:3, 3] = d.cam(name).xpos
    return T


def _sensor_mj_frame(project, sensor, q=None):
    """The sensor's world pose as a MuJoCo camera frame (x right, y up, looking along −z)."""
    P = sensor_world_pose(project, sensor, q)
    R = quat_to_matrix(P.quaternion)
    # body (x forward, y left, z up) -> MuJoCo camera: x_cam = −y_body, y_cam = z_body, −z_cam = x_body
    return np.c_[-R[:, 1], R[:, 2], -R[:, 0]], np.asarray(P.position)


# ====================================================================== every new sample
@pytest.mark.parametrize('name', NEW)
def test_sample_opens_with_what_the_studies_need(name, built):
    p = built[name]
    assert S.SAMPLES[name][1] is None and next(s for s in S.list() if s['name'] == name)['available']
    assert sum(r.primary for r in p.robots) == 1 and p.robots[0].primary
    assert np.allclose(p.robots[0].base.position, 0) and np.allclose(p.robots[0].base.quaternion, [0, 0, 0, 1])
    for r in p.robots:
        assert r.source.file in p.sources and r.source.license and 'Menagerie' in r.source.note
        m = get_model(p, r.uid)
        assert np.isfinite(m.lower).all() and np.isfinite(m.upper).all()
        for f in [r.gripper_frame, *r.fingertip_frames]:
            m.link_index(f)
        assert any(j.get('gripper') for j in r.joints)
    kinds = {v.kind for v in p.workspace.volumes}
    assert 'tabletop' in kinds and 'handover' in kinds
    assert p.variant.sensors and p.tasks.trajectories and p.tasks.grasp_sets and p.policies
    for tr in p.tasks.trajectories:
        m = get_model(p, tr['robot_uid'])
        assert tr['joints'] == list(m.joint_names) and len(tr['q']) == len(tr['t']) >= 3
        Q = np.array(tr['q'])
        assert np.all(Q >= m.lower - 1e-9) and np.all(Q <= m.upper + 1e-9)
        assert np.ptp(Q, axis=0).max() > 0.05                     # it moves
    for gs in p.tasks.grasp_sets:
        assert gs['poses'] and p.robot(gs['robot_uid']) is not None and '(sample)' in gs['name']
    assert '(sample)' in p.notes or 'sample' in p.notes


@pytest.mark.parametrize('name', NEW)
def test_round_trip_through_dexv(name, built, tmp_path):
    p = built[name]
    path = str(tmp_path / f'{name}.dexv')
    p.save(path)
    clear_cache()
    q = Project.load(path)
    assert [r.name for r in q.robots] == [r.name for r in p.robots]
    for a, b in zip(p.robots, q.robots):
        ma, mb = get_model(p, a.uid), get_model(q, b.uid)
        assert np.allclose(ma.fk_array(ma.home), mb.fk_array(mb.home), atol=1e-12)
        assert np.allclose(a.base.matrix(), b.base.matrix())
    assert [s.to_dict() for s in p.variant.sensors] == [s.to_dict() for s in q.variant.sensors]
    assert json.dumps(p.tasks.to_dict(), sort_keys=True) == json.dumps(q.tasks.to_dict(), sort_keys=True)
    for s, t in zip(p.variant.sensors, q.variant.sensors):
        assert np.allclose(sensor_world_pose(p, s).position, sensor_world_pose(q, t).position, atol=1e-12)


# ====================================================================== ALOHA 2
def test_aloha_fk_and_cameras_against_aloha_xml(built):
    """Both arms (two robots, bases from aloha.xml) against aloha.xml itself (both arms in one file) at the same joint
    values, in the left arm's base frame; the wrist cameras against aloha.xml's cameras and the frame cameras against
    scene_parts.xml, all compiled by MuJoCo directly."""
    p = built['aloha']
    left, right = p.robots
    ml, mr = get_model(p, left.uid), get_model(p, right.uid)
    ref, d = _mj(S.robot_file('aloha', 'aloha.xml'))
    rng = np.random.default_rng(5)
    worst = cam = 0.0
    for _ in range(6):
        ql, qr = rng.uniform(ml.lower, ml.upper), rng.uniform(mr.lower, mr.upper)
        d.qpos[:] = 0
        _set(ref, d, ml.joint_names, ql)
        _set(ref, d, mr.joint_names, qr)
        mujoco.mj_kinematics(ref, d)
        mujoco.mj_camlight(ref, d)
        Ti = np.linalg.inv(_body_T(d, 'left/base_link'))
        for m, q, side in ((ml, ql, 'left'), (mr, qr, 'right')):
            for link in (f'{side}/gripper_base', f'{side}/left_finger_link', f'{side}/wrist_link'):
                worst = max(worst, float(np.abs(m.link_pose(q, link) - Ti @ _body_T(d, link)).max()))
        qs = {left.uid: ql, right.uid: qr}
        for s in p.variant.sensors:
            if s.mount.on_link:
                R, t = _sensor_mj_frame(p, s, qs)
                want = Ti @ _cam_T(d, f'wrist_cam_{"left" if "left" in s.name else "right"}')
                cam = max(cam, float(np.abs(R - want[:3, :3]).max()), float(np.abs(t - want[:3, 3]).max()))
    assert worst < 1e-6 and cam < 1e-6          # MuJoCo's to_xml writes aloha.xml's numbers back to ~5e-7
    sc, ds = _mj(S.robot_file('aloha', 'scene_parts.xml'))
    mujoco.mj_kinematics(sc, ds)
    mujoco.mj_camlight(sc, ds)
    d.qpos[:] = 0
    mujoco.mj_kinematics(ref, d)
    Ti = np.linalg.inv(_body_T(d, 'left/base_link'))
    for s, cname in zip(p.variant.sensors[:2], ('overhead_cam', 'worms_eye_cam')):
        R, t = _sensor_mj_frame(p, s)
        want = Ti @ _cam_T(ds, cname)
        assert np.allclose(R, want[:3, :3], atol=1e-9) and np.allclose(t, want[:3, 3], atol=1e-9)


def test_aloha_rig_is_the_published_one(built):
    p = built['aloha']
    names = [s.name for s in p.variant.sensors]
    assert [n.split(' ')[0] for n in names] == ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    for s in p.variant.sensors:
        assert s.model_id == 'intel-realsense-d405' and s.stream == [848, 480]
    left, right = p.robots
    assert {s.mount.robot_uid for s in p.variant.sensors if s.mount.on_link} == {left.uid, right.uid}
    # the bases are aloha.xml's: 0.938 m apart, the right one turned half a turn
    assert np.allclose(right.base.position, [0.938, 0.0, 0.0], atol=1e-9)
    assert np.allclose(quat_to_matrix(right.base.quaternion), np.diag([-1.0, -1.0, 1.0]), atol=1e-9)
    # a two-arm recording names both arms' joints: each arm's joints are its own aloha.xml names
    ml, mr = get_model(p, left.uid), get_model(p, right.uid)
    ref = mujoco.MjModel.from_xml_path(S.robot_file('aloha', 'aloha.xml'))
    both = [ref.joint(j).name for j in range(ref.njnt)]
    assert list(ml.joint_names) + list(mr.joint_names) == both
    for m in (ml, mr):                                       # joint limits are the file's own
        for i, n in enumerate(m.joint_names):
            assert (m.lower[i], m.upper[i]) == tuple(float(v) for v in ref.jnt_range[ref.joint(n).id])
    devs = p.tasks.teleop_devices
    assert len(devs) == 2 and {d['follower_uid'] for d in devs} == {left.uid, right.uid}
    assert all(d['library_id'] == 'aloha-widowx-250-leader' and 'WidowX' in d['source']['sample'] for d in devs)
    assert all(d['leader'] == {'robot_uid': d['follower_uid']} for d in devs)     # led joint to joint
    hand = [g for g in p.tasks.grasp_sets if 'handoff' in g['name']]
    assert len(hand) == 1 and hand[0]['robot_uid'] == left.uid
    assert hand[0]['settings']['partner_robot_uid'] == right.uid      # the handoff approaches the right arm


# ====================================================================== Franka Panda
def test_panda_two_instances_fk_against_panda_xml(built):
    p = built['panda-bimanual']
    a, b = p.robots
    assert a.source.sha256 == b.source.sha256 and a.uid != b.uid
    assert 'no official bimanual Franka model' in p.notes
    ref, d = _mj(S.robot_file(*B.PANDA_FILE))
    rng = np.random.default_rng(2)
    worst = 0.0
    for r in (a, b):
        m = get_model(p, r.uid)
        for q in [m.home] + [rng.uniform(m.lower, m.upper) for _ in range(4)]:
            _set(ref, d, m.joint_names, q)
            mujoco.mj_kinematics(ref, d)
            for link in ('hand', 'link7', 'left_finger'):
                worst = max(worst, float(np.abs(m.link_pose(q, link) - r.base.matrix() @ _body_T(d, link)).max()))
    assert worst < 1e-9
    assert np.allclose(b.base.position, [0.0, -B.PANDA_SPACING_M, 0.0])
    k = ref.key('home').id
    m = get_model(p, a.uid)
    assert np.allclose(m.home, ref.key_qpos[k][[ref.jnt_qposadr[ref.joint(n).id] for n in m.joint_names]])


def test_panda_cameras(built):
    p = built['panda-bimanual']
    wl, wr, front = p.variant.sensors
    assert wl.mount.on_link and wr.mount.on_link and wl.mount.robot_uid != wr.mount.robot_uid
    assert not front.mount.on_link
    assert all('sample placement' in s.name for s in p.variant.sensors)
    assert [(d['kind'], d['library_id']) for d in p.tasks.teleop_devices] == [('gello', 'gello')] * 2


# ====================================================================== Unitree G1
def test_g1_upper_body_against_the_full_model_at_stand(built):
    """The held legs and waist: every link of the upper-body model at random arm and hand values equals the same body
    of g1_with_hands.xml (free joint, legs and waist at "stand") compiled by MuJoCo."""
    p = built['g1']
    r = p.robots[0]
    m = get_model(p, r.uid)
    assert m.nq == 28 and not any(n.startswith(('left_hip', 'right_knee', 'waist')) for n in m.joint_names)
    ref, d = _mj(S.robot_file('unitree_g1', 'g1_with_hands.xml'))
    stand = ref.key_qpos[ref.key('stand').id]
    rng = np.random.default_rng(9)
    worst = 0.0
    for q in [m.home] + [rng.uniform(m.lower, m.upper) for _ in range(4)]:
        d.qpos[:] = stand
        _set(ref, d, m.joint_names, q)
        mujoco.mj_kinematics(ref, d)
        for link in ('pelvis', 'left_ankle_roll_link', 'torso_link', 'right_wrist_yaw_link', 'left_hand_index_1_link',
                     'right_hand_thumb_2_link'):
            worst = max(worst, float(np.abs(m.link_pose(q, link) - _body_T(d, link)).max()))
    assert worst < 1e-6
    k = ref.key('stand').id
    stand_q = ref.key_qpos[k][[ref.jnt_qposadr[ref.joint(n).id] for n in m.joint_names]]
    # "stand" puts the thumbs' second joint at ±1.05 rad, just past the joint's ±1.0472 limit: home is clipped
    assert np.allclose(m.home, np.clip(stand_q, m.lower, m.upper)) and np.abs(m.home - stand_q).max() < 0.003


def test_g1_head_camera_is_unitrees_d435_link(built):
    """The head camera against Unitree's URDF figures, composed independently: torso_link from MuJoCo, the joint
    origin as URDF writes it (R = Rz(y) Ry(p) Rx(r))."""
    p = built['g1']
    head = p.variant.sensors[0]
    assert head.model_id == 'intel-realsense-d435' and head.mount.parent_frame == 'torso_link'
    ref, d = _mj(S.robot_file('unitree_g1', 'g1_with_hands.xml'))
    d.qpos[:] = ref.key_qpos[ref.key('stand').id]
    mujoco.mj_kinematics(ref, d)
    pitch = 0.8307767239493009
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    T = _body_T(d, 'torso_link')
    want_R, want_t = T[:3, :3] @ Ry, T[:3, :3] @ np.array([0.0576235, 0.01753, 0.42987]) + T[:3, 3]
    P = sensor_world_pose(p, head)
    assert np.allclose(quat_to_matrix(P.quaternion), want_R, atol=1e-12) and np.allclose(P.position, want_t)
    assert quat_to_matrix(P.quaternion)[2, 0] < -0.5           # it looks down, at the hands
    assert 'Depth Camera' in B.SOURCES['unitree_g1_page']['quote']


def test_g1_one_state_vector_moves_both_arms(built):
    p = built['g1']
    tr = p.tasks.trajectories[0]
    m = get_model(p, tr['robot_uid'])
    Q = np.array(tr['q'])
    moved = np.ptp(Q, axis=0) > 0.05
    names = np.array(m.joint_names)
    assert any(n.startswith('left_shoulder') for n in names[moved])
    assert any(n.startswith('right_shoulder') for n in names[moved])
    assert any(n.startswith('right_hand') for n in names[moved])


def test_g1_simplification_is_recorded_and_said(built):
    simp = B.G1_SIMPLIFICATION
    assert simp['largest_mm'] is not None and simp['largest_mm'] <= simp['tolerance_mm']
    text = open(S.robot_file('unitree_g1', 'g1_with_hands.xml'), encoding='utf-8').read()
    assert 'SIMPLIFIED' in text and '.simplified.stl' in text
    assert 'simplified' in built['g1'].notes and f'{simp["largest_mm"]}' in built['g1'].notes
    assert 'simplified' in built['g1'].robots[0].source.note


# ====================================================================== the bundle
def test_new_bundles_fit_the_budget():
    for folder, gen in (('aloha', ('aloha_left.xml', 'aloha_right.xml')), ('franka_emika_panda', ()),
                        ('unitree_g1', ('g1_upper_body.xml',))):
        files = [os.path.join(r, f) for r, _d, fs in os.walk(S.robot_file(folder)) for f in fs]
        files += [S.robot_file(g) for g in gen]
        assert max(os.path.getsize(f) for f in files) <= 5e6
        assert sum(os.path.getsize(f) for f in files) <= 12e6, folder
        assert os.path.isfile(S.robot_file(folder, 'LICENSE'))


def test_cli_opens_each_new_sample(tmp_path):
    from dexvantage.app import main
    for name in NEW:
        out = tmp_path / f'{name}.json'
        assert main(['--sample', name, '--analyse', str(out)]) == 0
        doc = json.loads(out.read_text())
        assert doc['sensors'] and doc['project']


def test_sample_in_the_window(app, win):
    from conftest import pump
    for name in NEW:
        assert win.open_sample(name)
        pump(app, 0.05)
        posed = {s.name for s, _R, _t in win.scene.posed_sensors()}
        assert posed == {s.name for s in win.store.project.variant.sensors}
        assert win.scene.mesh() is not None


def test_a_camera_on_a_link_no_joint_moves_is_calibrated_as_static(built):
    """The G1's head camera rides on torso_link, which no joint moves (legs and waist held): the hand-eye planner
    treats it as a static camera (eye-to-hand, the board in the hand), not as eye-in-hand with nothing to move."""
    from dexvantage.core.calib import records as RC
    from dexvantage.core.scenegeom import sensor_world_pose
    p = built['g1']
    head = p.variant.sensors[0]
    im = RC.sensor_imager(head)
    robot, link, X = RC.camera_mount(p, head, im)
    assert (robot, link) == ('', '')
    assert np.allclose(X, sensor_world_pose(p, head).matrix() @ im.pose_in_body.matrix())
    a = built['aloha']
    wrist = next(s for s in a.variant.sensors if s.name == 'cam_left_wrist')
    assert RC.camera_mount(a, wrist, RC.sensor_imager(wrist))[:2] == (a.robots[0].uid, 'left/gripper_base')


def test_g1_palm_and_thumb_do_not_count_as_a_collision_at_home(built):
    """Menagerie's palm and thumb-middle collision meshes overlap by 0.27 mm at "stand"; the upper-body file excludes
    the pair, so the arm's IK does not refuse every pose as a self-collision."""
    from dexvantage.core.robot import collision as C
    p = built['g1']
    m = get_model(p, p.robots[0].uid)
    assert not [pr for pr in C.self_collision(m, m.home, margin=0.0).pairs if pr[2] <= 0]
