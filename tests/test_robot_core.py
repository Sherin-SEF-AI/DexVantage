"""The robot core: import (URDF, MJCF), the MuJoCo model (FK, Jacobian, meshes), the store's joint state, the scene
and the window's Robot ▸ Import robot…, on small model files written here.

References: forward kinematics against the planar closed form written out again below; the Jacobian against central
finite differences of fk (an independent route: no mj_jac); posed triangle counts against the faces listed by
visual_meshes; source digests against hashlib."""
import hashlib
import math
import struct
import threading

import numpy as np
import pytest

from conftest import pump
from dexvantage.core import project as P
from dexvantage.core import provenance as PV
from dexvantage.core.frames import Pose, quat_from_rpy
from dexvantage.core.robot import RobotImportError, RobotModel, clear_cache, get_model, import_robot
from dexvantage.core.robot.model import primitive_mesh

L1, L2, D_TOOL, H_BASE = 0.5, 0.4, 0.4, 0.1        # the fixture's link lengths (m)


def stl_box(hx, hy, hz) -> bytes:
    """A binary STL of a box with half-extents hx, hy, hz centred on the origin (12 triangles)."""
    c = [(x, y, z) for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)]
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = [t for a, b, cc, d in quads for t in ((a, b, cc), (a, cc, d))]
    out = b'\0' * 80 + struct.pack('<I', len(tris))
    for t in tris:
        out += struct.pack('<3f', 0, 0, 0) + b''.join(struct.pack('<3f', *c[i]) for i in t) + b'\0\0'
    return out


INERTIAL = ('<inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>'
            '</inertial>')


def urdf_text(mesh_ref='package://arm/meshes/link1.stl', scale='1 1 1'):
    return f'''<?xml version="1.0"?>
<robot name="planar_arm">
  <link name="base"><visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual></link>
  <link name="link1">{INERTIAL}
    <visual><origin xyz="0.25 0 0"/><geometry><mesh filename="{mesh_ref}" scale="{scale}"/></geometry>
      <material name="red"><color rgba="1 0 0 1"/></material></visual>
    <collision><origin xyz="0.25 0 0" rpy="0 1.5707963 0"/><geometry><cylinder radius="0.02" length="0.5"/>
      </geometry></collision></link>
  <link name="link2">{INERTIAL}
    <visual><origin xyz="0.2 0 0"/><geometry><box size="0.4 0.03 0.03"/></geometry></visual></link>
  <link name="tool"/>
  <link name="slider">{INERTIAL}<visual><geometry><sphere radius="0.02"/></geometry></visual></link>
  <joint name="shoulder" type="revolute"><parent link="base"/><child link="link1"/><origin xyz="0 0 {H_BASE}"/>
    <axis xyz="0 0 1"/><limit lower="-2.0" upper="2.0" velocity="3.1" effort="12"/></joint>
  <joint name="elbow" type="revolute"><parent link="link1"/><child link="link2"/><origin xyz="{L1} 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-1.5" upper="1.5" velocity="3.1" effort="12"/></joint>
  <joint name="flange" type="fixed"><parent link="link2"/><child link="tool"/><origin xyz="{D_TOOL} 0 0"
    rpy="0 0 0.3"/></joint>
  <joint name="slide" type="prismatic"><parent link="tool"/><child link="slider"/><axis xyz="1 0 0"/>
    <limit lower="0.01" upper="0.1" velocity="0.2" effort="5"/></joint>
</robot>
'''


MJCF = f'''<mujoco model="mj_arm">
  <compiler angle="radian" meshdir="assets"/>
  <include file="parts/defaults.xml"/>
  <asset><mesh name="l1" file="link1.stl"/>
    <texture name="wood" type="2d" file="wood.png"/><material name="w" texture="wood" rgba="0 1 0 1"/></asset>
  <worldbody>
    <geom type="plane" size="1 1 0.1"/>
    <body name="base"><geom type="box" size="0.05 0.05 0.05" class="vis"/>
      <body name="link1" pos="0 0 {H_BASE}"><joint name="j1" axis="0 0 1" range="-2 2" actuatorfrcrange="-5 5"/>
        <geom type="mesh" mesh="l1" pos="0.25 0 0" class="vis" material="w"/>
        <geom type="capsule" fromto="0 0 0 {L1} 0 0" size="0.02"/>
        <body name="link2" pos="{L1} 0 0"><joint name="j2" axis="0 0 1"/>
          <geom type="capsule" fromto="0 0 0 {L2} 0 0" size="0.02" class="vis"/>
          <body name="tool" pos="{L2} 0 0"><site name="tcp"/></body>
        </body>
      </body>
    </body>
  </worldbody>
  <keyframe><key name="home" qpos="0.3 -0.2"/></keyframe>
</mujoco>
'''
MJCF_DEFAULTS = ('<mujoco><default><default class="vis"><geom contype="0" conaffinity="0" group="2"/></default>'
                 '</default></mujoco>')


@pytest.fixture
def urdf_path(tmp_path):
    """A ROS-style package: ws/arm/urdf/arm.urdf naming package://arm/meshes/link1.stl."""
    pkg = tmp_path / 'ws' / 'arm'
    (pkg / 'urdf').mkdir(parents=True)
    (pkg / 'meshes').mkdir()
    (pkg / 'meshes' / 'link1.stl').write_bytes(stl_box(0.25, 0.02, 0.02))
    (pkg / 'urdf' / 'arm.urdf').write_text(urdf_text())
    return str(pkg / 'urdf' / 'arm.urdf')


@pytest.fixture
def mjcf_path(tmp_path):
    d = tmp_path / 'mj'
    (d / 'assets').mkdir(parents=True)
    (d / 'parts').mkdir()
    (d / 'assets' / 'link1.stl').write_bytes(stl_box(0.25, 0.02, 0.02))
    (d / 'parts' / 'defaults.xml').write_text(MJCF_DEFAULTS)
    (d / 'arm.xml').write_text(MJCF)
    return str(d / 'arm.xml')


def project_with(path, base=None):
    p = P.blank_project('Robot test')
    r, s = import_robot(path)
    r.primary = True
    if base is not None:
        r.base = base
    p.robots.append(r)
    p.sources.update(s)
    return p, r


def closed_form(q, base=np.eye(4)):
    """World poses of tool and slider of the planar fixture: z-rotations by shoulder, elbow, then the fixed 0.3 rad
    flange, and the slide along the tool's x."""
    a1, a2, d = q
    def Rz(a):
        return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    p_elbow = np.array([0, 0, H_BASE]) + Rz(a1) @ [L1, 0, 0]
    p_tool = p_elbow + Rz(a1 + a2) @ [D_TOOL, 0, 0]
    R_tool = Rz(a1 + a2 + 0.3)
    p_slider = p_tool + R_tool @ [d, 0, 0]
    out = {}
    for k, (R, p) in {'tool': (R_tool, p_tool), 'slider': (R_tool, p_slider), 'link2': (Rz(a1 + a2), p_elbow)}.items():
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, p
        out[k] = base @ T
    return out


# ====================================================================== import
def test_urdf_import_reads_the_files_figures(urdf_path):
    r, s = import_robot(urdf_path)
    data = open(urdf_path, 'rb').read()
    assert r.name == 'planar_arm' and r.source.kind == 'urdf'
    assert r.source.file == f'{r.uid}/arm/urdf/arm.urdf' and s[r.source.file] == data
    assert r.source.sha256 == hashlib.sha256(data).hexdigest()
    assert f'{r.uid}/arm/meshes/link1.stl' in s
    assert all(k.startswith(r.uid + '/') for k in s)
    j = {x['name']: x for x in r.joints}
    assert [x['type'] for x in r.joints] == ['revolute', 'revolute', 'fixed', 'prismatic']
    assert (j['shoulder']['lower'], j['shoulder']['upper'], j['shoulder']['velocity'], j['shoulder']['effort']) == \
        (-2.0, 2.0, 3.1, 12.0)
    assert (j['slide']['lower'], j['slide']['upper']) == (0.01, 0.1)
    assert j['flange']['lower'] is None and j['flange']['velocity'] is None   # the file states none
    assert j['elbow']['origin']['position'] == [L1, 0.0, 0.0]
    assert np.allclose(j['flange']['origin']['quaternion'], quat_from_rpy(0, 0, 0.3))
    links = {x['name']: x for x in r.links}
    assert set(links) == {'base', 'link1', 'link2', 'tool', 'slider'}
    assert links['tool']['parent'] == 'link2' and links['tool']['parent_joint'] == 'flange'
    assert links['link1']['mesh_keys'] == [f'{r.uid}/arm/meshes/link1.stl']
    # no keyframe in a URDF: zeros clipped to the limits (the slide's lower limit is 0.01 m)
    assert r.home == {'shoulder': 0.0, 'elbow': 0.0, 'slide': 0.01}


def test_mjcf_import_with_include_meshdir_and_keyframe(mjcf_path):
    r, s = import_robot(mjcf_path)
    assert r.source.kind == 'mjcf' and r.name == 'mj_arm'
    assert {f'{r.uid}/arm.xml', f'{r.uid}/assets/link1.stl', f'{r.uid}/parts/defaults.xml'} <= set(s)
    assert 'wood.png' in r.source.note                    # the missing texture is said, not fatal
    j = {x['name']: x for x in r.joints}
    assert (j['j1']['type'], j['j1']['lower'], j['j1']['upper'], j['j1']['effort']) == ('revolute', -2.0, 2.0, 5.0)
    assert (j['j2']['type'], j['j2']['lower'], j['j2']['upper']) == ('continuous', None, None)
    assert j['tool_fixed']['type'] == 'fixed' and j['tool_fixed']['parent'] == 'link2'
    assert r.home == pytest.approx({'j1': 0.3, 'j2': -0.2})      # the file's keyframe "home"
    m = RobotModel(r, s)
    assert m.joint_names == ['j1', 'j2'] and m.nq == 2
    assert m.lower[0] == -2 and m.upper[0] == 2 and np.isinf(m.lower[1]) and np.isinf(m.upper[1])
    assert np.allclose(m.home, [0.3, -0.2])
    # visual: the contype-0 geoms (box, mesh, link2 capsule); collision: link1's capsule; the plane is not the robot's
    assert sorted(v[0] for v in m.visual_meshes()) == ['base', 'link1', 'link2']
    assert [c[0] for c in m.collision_geoms()] == ['link1']
    mesh_rgba = [v[3] for v in m.visual_meshes() if v[0] == 'link1'][0]
    assert np.allclose(mesh_rgba, [0, 1, 0, 1])           # the material's colour
    T = m.fk([0.0, 0.0])
    assert np.allclose(T['tool'][:3, 3], [L1 + L2, 0, H_BASE])


def test_missing_mesh_names_the_file_and_what_to_pick(urdf_path, tmp_path):
    import os
    os.remove(os.path.join(os.path.dirname(urdf_path), '..', 'meshes', 'link1.stl'))
    with pytest.raises(RobotImportError) as e:
        import_robot(urdf_path)
    msg = str(e.value)
    assert 'link1.stl' in msg and 'not in the folder you chose' in msg and 'pick' in msg.lower()
    assert 'Traceback' not in msg
    # a package the folder tree does not contain
    lone = tmp_path / 'lone.urdf'
    lone.write_text(urdf_text('package://other_pkg/meshes/link1.stl'))
    with pytest.raises(RobotImportError) as e:
        import_robot(str(lone))
    assert 'other_pkg' in str(e.value) and 'package folder' in str(e.value)


def test_relative_mesh_and_millimetre_check(tmp_path):
    (tmp_path / 'link1.stl').write_bytes(stl_box(250, 20, 20))           # a mesh drawn in millimetres
    (tmp_path / 'mm.urdf').write_text(urdf_text('link1.stl'))
    with pytest.raises(RobotImportError) as e:
        import_robot(str(tmp_path / 'mm.urdf'))
    assert 'millimetres' in str(e.value) and 'scale="0.001 0.001 0.001"' in str(e.value)
    (tmp_path / 'ok.urdf').write_text(urdf_text('link1.stl', scale='0.001 0.001 0.001'))
    r, s = import_robot(str(tmp_path / 'ok.urdf'))
    m = RobotModel(r, s)
    V = [v[1] for v in m.visual_meshes() if v[0] == 'link1'][0]
    assert np.allclose(V.max(axis=0) - V.min(axis=0), [0.5, 0.04, 0.04], atol=1e-6)   # <mesh scale> applied


def test_not_a_robot_file(tmp_path):
    f = tmp_path / 'x.xml'
    f.write_text('<scene/>')
    with pytest.raises(RobotImportError) as e:
        import_robot(str(f))
    assert '<robot>' in str(e.value) and '<mujoco>' in str(e.value)


# ====================================================================== kinematics
def test_fk_against_the_closed_form(urdf_path):
    base = Pose([0.2, -0.1, 0.05], quat_from_rpy(0.0, 0.0, 0.7))
    p, r = project_with(urdf_path, base)
    m = get_model(p, r.uid)
    assert m.joint_names == ['shoulder', 'elbow', 'slide']
    assert np.allclose(m.lower, [-2, -1.5, 0.01]) and np.allclose(m.upper, [2, 1.5, 0.1])
    assert m.link_names == ['base', 'link1', 'link2', 'tool', 'slider']
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.uniform(m.lower, m.upper)
        T = m.fk(q)
        ref = closed_form(q, base.matrix())
        for k in ref:
            assert np.allclose(T[k], ref[k], atol=1e-12), k
        assert np.allclose(m.link_pose(q, 'slider'), ref['slider'], atol=1e-12)
    with pytest.raises(ValueError):
        m.fk([0.0, 0.0])
    with pytest.raises(KeyError):
        m.link_pose(m.home, 'no_such_link')


def test_jacobian_against_finite_differences(urdf_path):
    p, r = project_with(urdf_path, Pose([0.1, 0.0, 0.0], quat_from_rpy(0.2, -0.1, 0.4)))
    m = get_model(p, r.uid)
    q = np.array([0.4, -0.7, 0.05])
    pt = np.array([0.01, 0.02, -0.03])
    J = m.jacobian(q, 'slider', pt)
    assert J.shape == (6, 3)
    h = 1e-6
    for i in range(3):
        dq = np.zeros(3)
        dq[i] = h
        Tp, Tm = m.link_pose(q + dq, 'slider'), m.link_pose(q - dq, 'slider')
        lin = ((Tp[:3, :3] @ pt + Tp[:3, 3]) - (Tm[:3, :3] @ pt + Tm[:3, 3])) / (2 * h)
        W = (Tp[:3, :3] - Tm[:3, :3]) / (2 * h) @ m.link_pose(q, 'slider')[:3, :3].T
        ang = np.array([W[2, 1], W[0, 2], W[1, 0]])
        assert np.allclose(J[:3, i], lin, atol=1e-8), i
        assert np.allclose(J[3:, i], ang, atol=1e-8), i
    # the prismatic joint moves the point along the tool's x and does not rotate it
    assert np.allclose(J[3:, 2], 0) and np.isclose(np.linalg.norm(J[:3, 2]), 1)
    assert np.allclose(m.jacobian(q, 'slider')[:3], m.jacobian(q, 'slider', [0, 0, 0])[:3])


def test_fk_batch_equals_fk_and_threads_agree(urdf_path):
    p, r = project_with(urdf_path)
    m = get_model(p, r.uid)
    Q = np.random.default_rng(2).uniform(m.lower, m.upper, (40, 3))
    B = m.fk_batch(Q)
    assert B.shape == (40, len(m.link_names), 4, 4)
    for k in (0, 17, 39):
        T = m.fk(Q[k])
        assert np.array_equal(B[k], np.array([T[n] for n in m.link_names]))
    errs = []

    def work():
        for k in range(40):
            if not np.array_equal(m.fk_batch(Q[k:k + 1])[0], B[k]):
                errs.append(k)
    ts = [threading.Thread(target=work) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errs


def test_posed_triangles_and_meshes(urdf_path):
    p, r = project_with(urdf_path)
    m = get_model(p, r.uid)
    vis = m.visual_meshes()
    assert [v[0] for v in vis] == ['base', 'link1', 'link2', 'slider']
    assert all(v[1].dtype == np.float32 and v[2].dtype == np.int32 and v[3].shape == (4,) for v in vis)
    assert np.allclose([v[3] for v in vis if v[0] == 'link1'][0], [1, 0, 0, 1])   # the URDF material colour
    assert [c[0] for c in m.collision_geoms()] == ['link1']
    q = np.array([0.5, 0.3, 0.06])
    tri, nrm, li = m.posed_triangles(q)
    assert len(tri) == sum(len(v[2]) for v in vis) == m.n_triangles == len(m.triangle_rgba())
    assert tri.shape[1:] == (3, 3) and tri.dtype == np.float32 and nrm.shape == (len(tri), 3)
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1, atol=1e-5)
    # the link1 mesh, posed: link-frame vertices through fk
    T1 = m.link_pose(q, 'link1')
    k = m.link_names.index('link1')
    V, F = [(v[1], v[2]) for v in vis if v[0] == 'link1'][0]
    assert np.allclose(tri[li == k], (V[F] @ T1[:3, :3].T + T1[:3, 3]), atol=1e-6)
    # outward normals on the convex link1 box
    c = T1[:3, :3] @ [0.25, 0, 0] + T1[:3, 3]
    assert (np.einsum('ij,ij->i', nrm[li == k], tri[li == k].mean(axis=1) - c) > 0).all()
    assert m.posed_triangles(q)[0] is tri                                  # cached per q
    lo, hi = m.bounds(q)
    assert (lo <= tri.reshape(-1, 3).min(axis=0) + 1e-9).all() and (hi >= tri.reshape(-1, 3).max(axis=0) - 1e-9).all()


@pytest.mark.parametrize('gtype,size,vol', [(6, (0.1, 0.2, 0.3), 8 * 0.006), (2, (0.1, 0, 0), 4 / 3 * math.pi * 1e-3),
                                            (5, (0.1, 0.2, 0), math.pi * 0.01 * 0.4),
                                            (3, (0.1, 0.2, 0), math.pi * 0.01 * 0.4 + 4 / 3 * math.pi * 1e-3)])
def test_primitives_are_closed_and_outward(gtype, size, vol):
    V, F = primitive_mesh(gtype, size)
    t = V[F]
    v = np.einsum('ij,ij->i', t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6     # divergence theorem
    # 24 segments: the inscribed polygon loses at most 1 - (sin(2π/24)·24/2π) = 1.1 % of the area per ring
    assert v == pytest.approx(vol, rel=0.03) and v > 0


def test_get_model_is_shared_and_keyed(urdf_path):
    p, r = project_with(urdf_path)
    a = get_model(p, r.uid)
    assert get_model(p, r.uid) is a
    assert get_model(p.snapshot(), r.uid) is a                 # a worker's snapshot shares it
    r.base = Pose([1.0, 0, 0])
    b = get_model(p, r.uid)
    assert b is not a and b.m is a.m                           # same compiled MuJoCo model, new base
    assert np.allclose(b.fk(b.home)['base'][:3, 3], [1, 0, 0])
    clear_cache()
    assert get_model(p, r.uid) is not b


# ====================================================================== store, scene, window
def test_store_q_clipping_and_reset(app, urdf_path):
    from dexvantage.ui.store import ProjectStore
    p, r = project_with(urdf_path)
    st = ProjectStore(p)
    assert np.allclose(st.q[r.uid], [0, 0, 0.01])              # home
    seen = []
    st.pose_changed.connect(seen.append)
    kept = st.set_q(r.uid, [5.0, -5.0, 0.05])
    assert np.allclose(kept, [2.0, -1.5, 0.05]) and np.allclose(st.q[r.uid], kept) and seen == [r.uid]
    st.set_q(r.uid, kept)
    assert seen == [r.uid]                                     # no change, no signal
    with pytest.raises(ValueError):
        st.set_q(r.uid, [0.0])
    st.edit('meta', lambda pr: setattr(pr, 'notes', 'x'), 'Note')
    assert np.allclose(st.q[r.uid], kept)                      # other edits keep the pose
    p2, r2 = project_with(urdf_path)
    st.set_project(p2)
    assert set(st.q) == {r2.uid} and np.allclose(st.q[r2.uid], [0, 0, 0.01])


def test_window_import_scene_undo(app, win, urdf_path):
    st = win.store
    assert 'no robot yet' in win.scene.hud_label(win.viewport)
    assert win.scene.mesh() is None
    assert win.actions_by_id['import_robot'].isEnabled()
    assert win.import_robot(urdf_path) is True
    pump(app)
    r = st.project.robots[0]
    assert r.primary and st.sel == ('robot', r.uid) and st.project.sources[r.source.file]
    md = win.scene.mesh()
    m = get_model(st.project, r.uid)
    assert md is not None and len(md.tri) == m.n_triangles and len(md.base) == len(md.tri)
    assert np.allclose(md.base[:12], m.triangle_rgba()[:12, :3])
    assert win.scene.mesh() is md                              # cached while nothing changes
    v = md.version
    st.set_q(r.uid, [1.0, 0.5, 0.05])
    md2 = win.scene.mesh()
    assert md2 is not md and md2.version > v
    assert np.allclose(md2.tri, m.posed_triangles([1.0, 0.5, 0.05])[0])
    hud = win.scene.hud_label(win.viewport)
    assert '1 robot ' in hud and '3 joints' in hud
    lo, hi, _ = win.scene.home_box(win.viewport)
    blo, bhi = m.bounds(st.q[r.uid])
    assert (lo <= blo).all() and (hi >= bhi).all()
    win.viewport.update()
    pump(app)
    # one undo step: the robot and its files go, and come back
    st.undo.undo()
    assert not st.project.robots and not st.project.sources and not st.q
    assert win.scene.mesh() is None
    st.undo.redo()
    assert st.project.robots[0].uid == r.uid and r.source.file in st.project.sources and r.uid in st.q
    # a second robot is not the primary one
    assert win.import_robot(urdf_path) and not st.project.robots[1].primary


def test_window_import_error_is_plain(win, tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr('dexvantage.ui.mainwindow.QMessageBox.warning', lambda *a: shown.append(a[1:3]))
    bad = tmp_path / 'broken.urdf'
    bad.write_text(urdf_text('meshes/nowhere.stl'))
    assert win.import_robot(str(bad)) is False
    assert shown and shown[0][0] == 'Could not import the robot' and 'nowhere.stl' in shown[0][1]
    assert not win.store.project.robots and win.store.undo.count() == 0


def test_sensor_on_link_follows_q(app, win, urdf_path):
    st = win.store
    win.import_robot(urdf_path)
    r = st.project.robots[0]
    mount = Pose([0.02, 0.0, 0.03], quat_from_rpy(0, 0.4, 0))
    st.edit('variant', lambda p: p.variant.sensors.append(
        P.Sensor(name='Wrist camera', mount=P.Mount('slider', r.uid, mount))), 'Added a wrist camera')
    s = st.project.variant.sensors[-1]
    assert s.mount.on_link and s not in win.scene.static_sensors()
    m = get_model(st.project, r.uid)
    for q in ([0.0, 0.0, 0.01], [0.8, -0.6, 0.09]):
        st.set_q(r.uid, q)
        (s2, R, t), = [x for x in win.scene.posed_sensors() if x[0].uid == s.uid]
        T = m.link_pose(q, 'slider') @ mount.matrix()
        assert np.allclose(R, T[:3, :3]) and np.allclose(t, T[:3, 3])
        # its frustum lines start at the sensor (the lens model's outline, Phase 2) and its body is drawn there
        starts = [ld.pts[0][0] for ld in win.scene.lines(win.viewport) if ld.pts.ndim == 3]
        assert any(np.allclose(p0, t) for p0 in starts)
        tris = win.scene.tris_extra(win.viewport)
        assert any(np.allclose(td.tri.reshape(-1, 3).mean(axis=0), T[:3, :3] @ [-0.01, 0, 0] + t, atol=1e-6)
                   for td in tris)
    # a mount on a link the robot does not have is skipped, not drawn at the origin
    st.edit('variant', lambda p: setattr(p.variant.sensors[-1].mount, 'parent_frame', 'no_link'), 'Moved')
    assert all(x[0].uid != s.uid for x in win.scene.posed_sensors())
    win.scene.lines(win.viewport)


# ====================================================================== the project file
def test_save_load_round_trip_and_content_hash(tmp_path, urdf_path, mjcf_path):
    p, r = project_with(urdf_path)
    r2, s2 = import_robot(mjcf_path)
    p.robots.append(r2)
    p.sources.update(s2)
    q = np.array([0.3, -0.2, 0.04])
    before = get_model(p, r.uid).fk(q)
    h0 = PV.project_digest(p)['sha256']
    f = str(tmp_path / 'rig.dexv')
    p.save(f)
    clear_cache()
    p2 = P.Project.load(f)
    assert p2.sources == p.sources
    assert [x.to_dict() for x in p2.robots] == [x.to_dict() for x in p.robots]
    assert PV.project_digest(p2)['sha256'] == h0
    after = get_model(p2, r.uid).fk(q)
    assert all(np.array_equal(before[k], after[k]) for k in before)
    assert np.allclose(get_model(p2, r2.uid).home, [0.3, -0.2])
    # the content hash sees the robot's files
    p2.sources[r.source.file] = p2.sources[r.source.file] + b'\n'
    assert PV.project_digest(p2)['sha256'] != h0


def test_mesh_outside_the_folder_and_no_manifest(tmp_path, urdf_path):
    from dexvantage.core.robot.importer import MANIFEST
    from dexvantage.core.robot.model import compile_robot
    (tmp_path / 'shared').mkdir()
    (tmp_path / 'shared' / 'link1.stl').write_bytes(stl_box(0.25, 0.02, 0.02))
    (tmp_path / 'robot').mkdir()
    (tmp_path / 'robot' / 'arm.urdf').write_text(urdf_text('../shared/link1.stl'))
    r, s = import_robot(str(tmp_path / 'robot' / 'arm.urdf'))
    assert f'{r.uid}/_ext/link1.stl' in s and r.source.file == f'{r.uid}/arm.urdf'   # no folders above the robot's
    assert RobotModel(r, s).n_triangles == RobotModel(*import_robot(urdf_path)).n_triangles
    # stored without the manifest (written by hand or by an older build): package paths resolve by their tail
    r2, s2 = import_robot(urdf_path)
    del s2[f'{r2.uid}/{MANIFEST}']
    assert compile_robot(r2, s2).mesh_keys == [f'{r2.uid}/arm/meshes/link1.stl']
