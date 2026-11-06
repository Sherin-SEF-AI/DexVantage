"""The sample projects (spec §14.2, §14.3) and the robot models bundled for them (spec §1.2).

Independent references: the bundled Menagerie files compiled directly by MuJoCo (joint limits, the UR5e flange site,
the gripper's own base offset), not through DexVantage's importer."""
import json
import os
import re

import mujoco
import numpy as np
import pytest

from dexvantage.core.frames import quat_to_matrix
from dexvantage.core.project import Project
from dexvantage.core.robot import clear_cache, get_model
from dexvantage.core.robot import samples as S

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
ROBOTS = S.ROBOTS_DIR


@pytest.fixture(scope='module')
def so100():
    return S.sample_project('so100')


@pytest.fixture(scope='module')
def ur5e():
    return S.sample_project('ur5e')


def _world_pose(project, sensor, q=None):
    R, t = quat_to_matrix(sensor.mount.pose.quaternion), np.asarray(sensor.mount.pose.position)
    if not sensor.mount.on_link:
        return R, t
    m = get_model(project, sensor.mount.robot_uid)
    T = m.link_pose(m.home if q is None else q, sensor.mount.parent_frame)
    return T[:3, :3] @ R, T[:3, :3] @ t + T[:3, 3]


# ====================================================================== bundled models and their licences
def test_every_bundled_model_has_its_licence_file():
    folders = sorted(d for d in os.listdir(ROBOTS) if os.path.isdir(os.path.join(ROBOTS, d)))
    assert folders == sorted(S.BUNDLED)
    for d in folders:
        lic = os.path.join(ROBOTS, d, 'LICENSE')
        assert os.path.isfile(lic) and os.path.getsize(lic) > 500, d
        assert os.path.isfile(os.path.join(ROBOTS, d, S.BUNDLED[d]['main'])), d
    from dexvantage.core.robot.menagerie import license_name
    for d, info in S.BUNDLED.items():                  # the recorded licence is what the file itself says
        assert license_name(open(os.path.join(ROBOTS, d, 'LICENSE'), encoding='utf-8').read())[0] == info['license']


def test_third_party_lists_every_bundled_model():
    text = open(os.path.join(ROOT, 'THIRD_PARTY.md'), encoding='utf-8').read()
    assert S.MENAGERIE_COMMIT in text and S.MENAGERIE in text
    for d, info in S.BUNDLED.items():
        row = next((ln for ln in text.splitlines() if f'`{d}`' in ln and ln.startswith('|')), '')
        assert row, f'{d} is not in THIRD_PARTY.md'
        assert info['license'] in row and info['folder_commit'] in row and S.CHECKED in row
    assert S.COMBINED_UR5E in text


def test_bundled_files_are_only_model_files_and_small():
    total = 0
    for r, _d, files in os.walk(ROBOTS):
        for f in files:
            p = os.path.join(r, f)
            total += os.path.getsize(p)
            assert f == 'LICENSE' or f.endswith(('.xml', '.stl', '.obj')), p
            assert os.path.getsize(p) < 2_000_000, p
    assert total < 20e6


def test_every_bundled_mesh_is_used():
    used = set()
    for d, info in S.BUNDLED.items():
        spec = mujoco.MjSpec.from_file(os.path.join(ROBOTS, d, info['main']))
        used |= {os.path.normpath(os.path.join(ROBOTS, d, spec.meshdir or '', m.file)) for m in spec.meshes}
    on_disk = {os.path.normpath(os.path.join(r, f)) for r, _d, fs in os.walk(ROBOTS) for f in fs
               if f.endswith(('.stl', '.obj'))}
    assert on_disk == used


# ====================================================================== the samples
@pytest.mark.parametrize('name', ['so100', 'ur5e'])
def test_samples_build_and_fk_runs(name, so100, ur5e):
    p = {'so100': so100, 'ur5e': ur5e}[name]
    assert len(p.robots) == 1 and p.robots[0].primary
    r = p.robots[0]
    assert r.source.file in p.sources and r.source.license and r.source.kind == 'mjcf'
    m = get_model(p, r.uid)
    assert m.nq >= 6
    assert np.isfinite(m.lower).all() and np.isfinite(m.upper).all()     # every joint limited in these files
    rng = np.random.default_rng(3)
    for q in [m.home] + [rng.uniform(m.lower, m.upper) for _ in range(5)]:
        T = m.fk_array(q)
        assert T.shape == (len(m.link_names), 4, 4) and np.isfinite(T).all()
        assert np.allclose(T[:, :3, :3] @ np.transpose(T[:, :3, :3], (0, 2, 1)), np.eye(3), atol=1e-9)
    for f in [r.gripper_frame, *r.fingertip_frames]:
        m.link_index(f)                                # the frames are links of the model
    assert len(r.fingertip_frames) == 2
    assert len(p.workspace.volumes) == 1 and p.workspace.volumes[0].kind == 'tabletop'
    lo, hi = p.workspace.volumes[0].bounds()
    tip = m.link_pose(m.home, r.gripper_frame)[:3, 3]
    assert np.all(tip[:2] > lo[:2]) and np.all(tip[:2] < hi[:2])         # the volume is under the gripper at home
    assert 'sample placement' in p.notes and 'sample placement' in p.variant.note


def test_joint_limits_are_the_files_own(so100, ur5e):
    for p, path in ((so100, S.robot_file('trs_so_arm100', 'so_arm100.xml')), (ur5e, S.robot_file(S.COMBINED_UR5E))):
        ref = mujoco.MjModel.from_xml_path(path)
        m = get_model(p, p.robots[0].uid)
        for i, n in enumerate(m.joint_names):
            j = ref.joint(n).id
            assert (m.lower[i], m.upper[i]) == tuple(float(v) for v in ref.jnt_range[j])
        k = ref.key('home').id
        assert np.allclose(m.home, ref.key_qpos[k][[ref.jnt_qposadr[ref.joint(n).id] for n in m.joint_names]])


def test_so100_sensors(so100):
    wrist, third = so100.variant.sensors
    assert wrist.mount.on_link and wrist.mount.parent_frame == 'Fixed_Jaw'
    assert third.mount.parent_frame == 'tripod' and not third.mount.on_link
    assert 'sample placement' in wrist.name and 'sample placement' in third.name
    assert wrist.kind == third.kind == 'camera'


def test_ur5e_sensors_and_gripper(ur5e):
    r = ur5e.robots[0]
    wrist, over = ur5e.variant.sensors
    assert (wrist.name, wrist.kind, wrist.model_id) == ('Wrist D405', 'depth', '')
    assert (over.name, over.kind, over.model_id) == ('Overhead D435', 'depth', '')
    assert wrist.mount.on_link and wrist.mount.parent_frame == 'wrist_3_link' and wrist.mount.robot_uid == r.uid
    assert not over.mount.on_link
    assert 'Phase 2' in ur5e.notes
    assert r.gripper_frame == 'gripper_base' and r.fingertip_frames == ['gripper_right_pad', 'gripper_left_pad']
    # the wrist camera's optical axis passes through the gripper's pinch site (the file's own site), at any q
    m = get_model(ur5e, r.uid)
    q = m.clip(m.home + 0.3)
    ref = mujoco.MjModel.from_xml_path(S.robot_file(S.COMBINED_UR5E))
    d = mujoco.MjData(ref)
    d.qpos[[ref.jnt_qposadr[ref.joint(n).id] for n in m.joint_names]] = q
    mujoco.mj_kinematics(ref, d)
    pinch = d.site('gripper_pinch').xpos
    R, t = _world_pose(ur5e, wrist, q)
    v = pinch - t
    off_axis = np.linalg.norm(v - np.dot(v, R[:, 0]) * R[:, 0])
    assert np.dot(v, R[:, 0]) > 0.1 and off_axis < 1e-6
    # the overhead camera looks straight down at the volume centre
    R, t = _world_pose(ur5e, over)
    assert np.allclose(R[:, 0], [0, 0, -1], atol=1e-9)
    assert np.allclose(t[:2], ur5e.workspace.volumes[0].center[:2])


def test_combined_file_attaches_the_gripper_at_the_flange_site():
    """Independent reference: the separate UR5e and 2F-85 files compiled by MuJoCo. The gripper's base_mount body in
    the combined file sits at the UR5e's attachment_site composed with the 2F-85 file's own base_mount offset."""
    arm = mujoco.MjModel.from_xml_path(S.robot_file('universal_robots_ur5e', 'ur5e.xml'))
    grip = mujoco.MjModel.from_xml_path(S.robot_file('robotiq_2f85', '2f85.xml'))
    both = mujoco.MjModel.from_xml_path(S.robot_file(S.COMBINED_UR5E))
    rng = np.random.default_rng(7)
    for _ in range(4):
        q = rng.uniform(-2, 2, 6)
        da, db = mujoco.MjData(arm), mujoco.MjData(both)
        da.qpos[:6] = q
        db.qpos[:6] = q
        mujoco.mj_kinematics(arm, da)
        mujoco.mj_kinematics(both, db)
        Rs, ts = da.site('attachment_site').xmat.reshape(3, 3), da.site('attachment_site').xpos
        bm = grip.body('base_mount')
        Rb = np.empty(9)
        mujoco.mju_quat2Mat(Rb, bm.quat)
        want_t = Rs @ bm.pos + ts
        want_R = Rs @ Rb.reshape(3, 3)
        got = db.body('gripper_base_mount')
        assert np.allclose(got.xpos, want_t, atol=1e-12)
        assert np.allclose(got.xmat.reshape(3, 3), want_R, atol=1e-12)
        for b in ('base', 'shoulder_link', 'wrist_3_link'):             # the arm itself is unchanged
            assert np.allclose(db.body(b).xpos, da.body(b).xpos, atol=1e-12)


def test_link_mounted_sensor_follows_q_static_does_not(so100):
    wrist, third = so100.variant.sensors
    m = get_model(so100, so100.robots[0].uid)
    q = m.home.copy()
    q[m.joint_names.index('Rotation')] += 0.5
    q = m.clip(q)
    R0, t0 = _world_pose(so100, wrist, m.home)
    R1, t1 = _world_pose(so100, wrist, q)
    assert np.linalg.norm(t1 - t0) > 0.05
    # a base rotation turns the wrist camera about the joint axis by the same angle
    ang = np.arccos(np.clip((np.trace(R0.T @ R1) - 1) / 2, -1, 1))
    assert ang == pytest.approx(abs(q[0] - m.home[0]), abs=1e-9)
    assert np.allclose(_world_pose(so100, third)[1], third.mount.pose.position)


def test_sample_in_the_window_poses_sensors_with_q(app, win, ur5e):
    from conftest import pump
    win.store.set_project(S.sample_project('ur5e'))
    pump(app, 0.05)
    r = win.store.project.robots[0]
    before = {s.name: t.copy() for s, _R, t in win.scene.posed_sensors()}
    assert set(before) == {'Wrist D405', 'Overhead D435'}
    m = get_model(win.store.project, r.uid)
    q = win.store.q[r.uid].copy()
    q[0] += 0.4
    win.store.set_q(r.uid, q)
    after = {s.name: t for s, _R, t in win.scene.posed_sensors()}
    assert np.linalg.norm(after['Wrist D405'] - before['Wrist D405']) > 0.05
    assert np.allclose(after['Overhead D435'], before['Overhead D435'])
    assert win.scene.mesh() is not None and m.n_triangles > 0


def test_sample_save_load_compiles_from_the_file(tmp_path, so100):
    path = str(tmp_path / 'so100.dexv')
    so100.save(path)
    clear_cache()
    p = Project.load(path)
    r = p.robots[0]
    m = get_model(p, r.uid)
    assert m.nq == 6 and r.source.license.startswith('Apache-2.0')
    a = get_model(so100, so100.robots[0].uid)
    assert np.allclose(m.fk_array(m.home), a.fk_array(a.home))


def test_later_samples_say_their_phase_and_unknown_is_plain():
    names = {s['name']: s for s in S.list()}
    assert set(names) == {'so100', 'ur5e', 'aloha', 'panda-bimanual', 'g1'}
    for n in ('aloha', 'panda-bimanual', 'g1'):
        assert not names[n]['available']
        with pytest.raises(S.SampleError, match='arrives in Phase 11'):
            S.sample_project(n)
    with pytest.raises(S.SampleError, match='so100, ur5e'):
        S.sample_project('nope')


def test_cli_sample_batch(tmp_path, capsys):
    from dexvantage.app import main
    out = tmp_path / 'summary.json'
    assert main(['--sample', 'ur5e', '--analyse', str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc['project'] == 'UR5e + Robotiq 2F-85 sample' and doc['sensors']['depth'] == 2
    assert main(['--sample', 'aloha', '--analyse', str(out)]) == 2
    assert 'Phase 11' in capsys.readouterr().err


def test_no_personal_paths_in_bundled_xml():
    for r, _d, files in os.walk(ROBOTS):
        for f in files:
            if f.endswith('.xml'):
                text = open(os.path.join(r, f), encoding='utf-8').read()
                assert not re.search(r'/home/|/Users/|[A-Za-z]:\\\\', text), f


def test_so100_limits_said_to_be_approximate(so100):
    """Menagerie's README says it added "approximate joint limits" to trs_so_arm100: the robot's note must say so."""
    assert 'approximate joint limits' in so100.robots[0].source.note


def test_bundled_ur5e_against_ur_dh_table():
    """The bundled Menagerie UR5e rounds its link offsets to the millimetre; its flange stays within 2 mm of the flange
    of Universal Robots' DH table (the closed form in checks_kinematics) after the best rigid alignment, which is what
    the docs state (1.3 mm worst, 0.9 mm RMS). Independent reference: the DH product, Kabsch alignment by SVD."""
    from dexvantage.validation.checks_kinematics import ur5e_closed_form
    m = mujoco.MjModel.from_xml_path(os.path.join(ROBOTS, 'universal_robots_ur5e', 'ur5e.xml'))
    d = mujoco.MjData(m)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'attachment_site')
    P, Q = [], []
    for q in np.random.default_rng(1).uniform(-np.pi, np.pi, (500, 6)):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        P.append(d.site_xpos[sid].copy())
        Q.append(ur5e_closed_form(q)[-1][:3, 3])
    P, Q = np.array(P), np.array(Q)
    cp, cq = P.mean(0), Q.mean(0)
    U, _s, Vt = np.linalg.svd((Q - cq).T @ (P - cp))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    r = np.linalg.norm(P - ((Q - cq) @ R.T + cp), axis=1)
    assert r.max() < 2e-3 and 0.5e-3 < np.sqrt((r ** 2).mean()) < 1.2e-3
