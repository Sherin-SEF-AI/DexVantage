"""Robot import and export fixes: the exported URDF's inertials (MuJoCo loads it as written), the OpenVLA encoding
choice, the Isaac export's off-centre principal point (notes, JSON, dialog) and the USD robot import (spec §13): a
hand-built stage (Y up, centimetres, a reversed prismatic joint, stated mass), the round trip of the bundled robots,
a third party's UR5e, the wizard and the API."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from dexvantage.core.jobs import StudyError
from dexvantage.core.project import ProjectFormatError
from dexvantage.core.robot.samples import sample_project
from dexvantage.validation import checks_robot_io as V

HAVE_PXR = importlib.util.find_spec('pxr') is not None
needs_pxr = pytest.mark.skipif(not HAVE_PXR, reason='usd-core (the usd extra) is not installed')


# ====================================================================== the URDF's inertials
def test_exported_urdf_loads_in_mujoco_and_poses_the_sensor_frames(tmp_path):
    d = V.urdf_mujoco_errors(str(tmp_path), np.random.default_rng(1), ('so100', 'panda-bimanual'), 3)
    assert not d['load_errors'], d['load_errors']
    assert d['position_m'] <= 1e-6 and d['rotation_rad'] <= 1e-6
    assert d['mass_rel'] <= 1e-9 and d['com_m'] <= 1e-9 and d['inertia_rel'] <= 1e-9
    assert d['placeholder_mass_err'] == 0.0
    assert all(s['frames_compared'] > 0 for s in d['samples'].values())


def test_every_link_has_a_labelled_inertial_and_the_world_none(tmp_path):
    from dexvantage.core import exporters as X
    from dexvantage.core.exporters import ros2 as R2
    from dexvantage.core.robot.model import get_model
    p = sample_project('so100')
    man = X.export(p, str(tmp_path), ['urdf'])
    text = (tmp_path / 'ros2' / 'rig.urdf').read_text(encoding='utf-8')
    root = ET.fromstring(text)
    links = {ln.get('name'): ln for ln in root.findall('link')}
    assert links['world'].find('inertial') is None
    assert all(ln.find('inertial') is not None for n, ln in links.items() if n != 'world')
    assert root.find('mujoco/compiler').get('meshdir') == 'meshes/'
    rm = get_model(p, p.robots[0].uid)
    src = R2.inertial_sources(p, p.robots[0], rm)
    assert src[rm.m.body('Base').id] == 'geoms' and src[rm.m.body('Moving_Jaw').id] == 'file'
    assert 'computed by MuJoCo from the robot file' in text and 'from the robot\'s file' in text
    assert 'a minimal placeholder' in text                 # the sensor and optical frames
    b = rm.m.body('Upper_Arm').id
    ine = links['Upper_Arm'].find('inertial')
    assert float(ine.find('mass').get('value')) == pytest.approx(float(rm.m.body_mass[b]), rel=1e-15)
    notes = ' '.join(man['formats']['urdf']['notes'])
    assert 'MuJoCo\'s computation from the file\'s geoms' in notes and 'minimal inertial placeholder' in notes


def test_inertia_tensor_is_the_principal_one_turned_into_the_link_axes():
    from dexvantage.core.exporters import ros2 as R2
    I = np.array([[2.0, 0.3, 0.0], [0.3, 1.0, 0.1], [0.0, 0.1, 3.0]])
    el = ET.fromstring(R2.inertial_xml(1.5, [0.1, 0, 0], I, 'file').strip())
    it = el.find('inertia')
    got = np.array([[float(it.get('ixx')), float(it.get('ixy')), float(it.get('ixz'))],
                    [float(it.get('ixy')), float(it.get('iyy')), float(it.get('iyz'))],
                    [float(it.get('ixz')), float(it.get('iyz')), float(it.get('izz'))]])
    assert np.array_equal(got, I) and el.find('origin').get('rpy') == '0 0 0'


# ====================================================================== OpenVLA
def test_openvla_encodings_follow_the_pinned_sources(tmp_path):
    d = V.openvla_errors(str(tmp_path))
    assert d['mismatches'] == []
    by = {c['case']: c for c in d['cases']}
    assert by['aloha +OFT']['fits'] and by['aloha +OFT']['action'] == 'JOINT_POS_BIMANUAL'
    assert by['so100 eef']['fits'] and not by['so100']['fits'] and not by['panda-bimanual']['fits']


def test_openvla_action_space_from_the_policy_and_refused_when_unknown(tmp_path):
    from dexvantage.core import exporters as X
    from dexvantage.core import project as P
    from dexvantage.core.exporters import lerobot as L
    p = sample_project('so100')
    pol = P.Policy(name='OpenVLA', library_id='openvla-7b',
                   overrides={'action_space': {'value': 'eef', 'source': 'measured by the user: EEF teleop'}})
    p.policies.append(pol)
    e = L.openvla_encoding(p)
    assert (e['state_encoding'], e['action_encoding'], e['fits']) == ('POS_EULER', 'EEF_POS', True)
    assert 'stored action_space' in e['action_space_source']
    with pytest.raises(StudyError, match='"joint" or "eef"'):
        L.openvla_encoding(p, 'cartesian')
    man = X.export(p, str(tmp_path), ['lerobot'], {'lerobot': {'action_space': 'joint'}})
    doc = json.loads((tmp_path / 'openvla' / 'oxe_dataset_config.json').read_text(encoding='utf-8'))
    assert doc['encoding']['action_encoding'] == 'JOINT_POS' and not doc['encoding']['fits']
    assert any('No encoding of the pinned source fits' in n for n in man['formats']['lerobot']['notes'])
    py = (tmp_path / 'openvla' / 'oxe_dataset_config.py').read_text(encoding='utf-8')
    assert 'materialize.py L32-33' in py and 'NO ENCODING FITS' in py


# ====================================================================== Isaac principal point
def test_isaac_off_centre_principal_point_reaches_v3_and_is_stated_for_v2(tmp_path):
    d = V.isaac_principal_point_errors(str(tmp_path))
    assert d['mismatches'] == [] and d['px'] <= 1e-3
    assert sum(1 for c in d['cameras'] if c['offset_px'] != [0.0, 0.0]) == 2
    v2 = (tmp_path / 'isaac' / 'isaaclab_v2_cameras.py').read_text(encoding='utf-8')
    assert 'cannot be shown at v2.3.2' in v2


def test_export_dialog_states_the_principal_point_per_camera(app, win):
    from dexvantage.ui.dialogs import export as DX
    win.store.set_project(V._isaac_project())
    d = DX.open_dialog(win)
    try:
        note = d.rows['isaac'].note
        assert not note.isHidden() and note.text().count('px from the centre') == 2
        assert 'Isaac Lab v2.3.2 has no OpenCV lens attributes' in note.text()
    finally:
        d.close()


# ====================================================================== USD import
def _hand_built_stage(path):
    """A Y-up stage in centimetres: A fixed to the world, B on a revolute joint about A's Z whose anchor is 10 cm
    from B's origin, C on a prismatic joint written the other way round (body0 = C, the child), and stated masses."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    st = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(st, 0.01)
    UsdPhysics.SetStageKilogramsPerUnit(st, 1.0)
    root = UsdGeom.Xform.Define(st, '/arm').GetPrim()
    st.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    pos = {'A': (0, 0, 0), 'B': (-10, 50, 0), 'C': (-10, 50, 30)}
    for n, t in pos.items():
        x = UsdGeom.Xform.Define(st, f'/arm/{n}')
        x.AddTranslateOp().Set(Gf.Vec3d(*t))
        UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
        m = UsdPhysics.MassAPI.Apply(x.GetPrim())
        m.CreateMassAttr(2.0)
        m.CreateDiagonalInertiaAttr(Gf.Vec3f(1.0, 2.0, 3.0))
        c = UsdGeom.Cube.Define(st, f'/arm/{n}/box')
        c.CreateSizeAttr(4.0)
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    f = UsdPhysics.FixedJoint.Define(st, '/arm/fix')
    f.CreateBody1Rel().SetTargets(['/arm/A'])
    r = UsdPhysics.RevoluteJoint.Define(st, '/arm/hinge')
    r.CreateBody0Rel().SetTargets(['/arm/A'])
    r.CreateBody1Rel().SetTargets(['/arm/B'])
    r.CreateAxisAttr('Z')
    r.CreateLocalPos0Attr(Gf.Vec3f(0, 50, 0))
    r.CreateLocalPos1Attr(Gf.Vec3f(10, 0, 0))
    r.CreateLowerLimitAttr(-90.0)
    r.CreateUpperLimitAttr(45.0)
    s = UsdPhysics.PrismaticJoint.Define(st, '/arm/slide')
    s.CreateBody0Rel().SetTargets(['/arm/C'])            # the child first: the importer turns it round
    s.CreateBody1Rel().SetTargets(['/arm/B'])
    s.CreateAxisAttr('X')
    s.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
    s.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 30))
    s.CreateLowerLimitAttr(0.0)
    s.CreateUpperLimitAttr(20.0)
    st.GetRootLayer().Save()


@needs_pxr
def test_hand_built_stage_converts_with_units_up_axis_and_a_reversed_joint(tmp_path):
    from dexvantage.core.robot import import_robot
    from dexvantage.core.robot.model import RobotModel
    path = str(tmp_path / 'arm.usda')
    _hand_built_stage(path)
    robot, sources = import_robot(path)
    rm = RobotModel(robot, sources)
    assert rm.joint_names == ['hinge', 'slide']
    assert rm.lower == pytest.approx([-math.pi / 2, 0.0], abs=1e-15)
    assert rm.upper == pytest.approx([math.pi / 4, 0.2], abs=1e-15)
    Rup = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])          # Y up to Z up

    def rz(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    for q in ([0.0, 0.0], [0.7, 0.05], [-1.2, 0.17]):
        T = rm.fk(np.array(q))
        # B turns about the joint at (0, 50, 0) cm (A's frame, Y up) about Z; its origin is 10 cm from the anchor
        Rb = rz(q[0])
        b = np.array([0, 50, 0]) + Rb @ np.array([-10.0, 0, 0])
        assert np.allclose(T['B'][:3, 3], 0.01 * Rup @ b, atol=1e-7)
        assert np.allclose(T['B'][:3, :3], Rup @ Rb, atol=1e-7)
        # C: X_C F0 T(q x) = X_B F1, F0 = I, F1 = T(0, 0, 30): X_C = X_B T(0, 0, 30) T(-q x)
        c = b + Rb @ (np.array([0, 0, 30.0]) - np.array([100.0 * q[1], 0, 0]))
        assert np.allclose(T['C'][:3, 3], 0.01 * Rup @ c, atol=1e-7)
        assert np.allclose(T['C'][:3, :3], Rup @ Rb, atol=1e-7)
    m = rm.m
    assert m.body_mass[m.body('B').id] == pytest.approx(2.0)
    assert sorted(m.body_inertia[m.body('B').id]) == pytest.approx([1e-4, 2e-4, 3e-4], rel=1e-6)
    assert m.ngeom == 3 and np.allclose(m.geom_size[0], 0.02)       # a 4 cm cube in metres
    assert 'Y up' in robot.source.note and 'Converted from the USD file arm.usda' in robot.source.note
    rec = json.loads(sources[f'{robot.uid}/.dexvantage/usd_source.json'])
    assert rec['up_axis'] == 'Y' and rec['meters_per_unit'] == 0.01 and rec['joints']['hinge'] == 1


@needs_pxr
def test_bundled_robots_round_trip_through_usd(tmp_path):
    d = V.usd_roundtrip_errors(str(tmp_path), np.random.default_rng(2), ('so100', 'ur5e', 'panda-bimanual'), 50)
    assert d['missing'] == [] and d['limit_mismatches'] == []
    assert d['position_m'] <= 1e-6 and d['rotation_rad'] <= 1e-6
    assert d['writer_pose_m'] <= 1e-6 and d['mass_rel'] <= 1e-6
    assert d['limit_rad'] <= d['limit_bound_rad']


@needs_pxr
def test_third_party_ur5e_poses_like_dexvantages():
    d = V.usd_third_party_errors(np.random.default_rng(3), 50)
    assert d['position_m'] <= 1e-6 and d['rotation_rad'] <= 1e-6 and d['limit_rad'] <= 1e-6 and d['mass_rel'] <= 1e-6
    assert 'GeometryLibrary.usdc' in d['notes']             # the mesh library is not bundled: said, not hidden


@needs_pxr
def test_usd_without_bodies_and_reference_check(tmp_path):
    from dexvantage.core.robot import import_robot
    from dexvantage.core.robot.filecheck import check_references
    p = tmp_path / 'empty.usda'
    p.write_text('#usda 1.0\n')
    with pytest.raises(ProjectFormatError, match='USD file with no UsdPhysics rigid bodies'):
        import_robot(str(p))
    path = os.path.join(os.path.dirname(V.__file__), '..', 'resources', *V.THIRD_PARTY)
    chk = check_references(path)
    assert chk['kind'] == 'usd'
    found = {x['ref'] for x in chk['items'] if x['found']}
    assert {'ur5e.usda', 'Physics.usda', 'Geometry.usda'} <= found
    missing = [x for x in chk['items'] if not x['found']]
    assert missing and all(not x['required'] for x in missing)


@needs_pxr
def test_wizard_imports_a_usd_robot(win, tmp_path):
    from dexvantage.ui.dialogs.import_wizard import ImportWizard
    from dexvantage.core.robot.usd_writer import write_robot_usd
    p = sample_project('so100')
    path = str(tmp_path / 'so100.usda')
    write_robot_usd(p, p.robots[0].uid, path)
    w = ImportWizard(win.ctx, win)
    w.load(path, background=False)
    assert w.robot is not None, w.f_status.text.text()
    assert 'USD, converted to MJCF' in w.f_meta.text()
    w.deleteLater()


@needs_pxr
def test_api_converts_imports_and_writes_usd(tmp_path):
    import dexvantage as dv
    rig = dv.sample('so100')
    info = dv.robot_io.write_usd(rig, str(tmp_path / 'arm.usda'))
    assert info['joints'] and info['geoms'] > 0
    c = dv.robot_io.convert_usd(str(tmp_path / 'arm.usda'))
    assert c['mjcf'].startswith('<mujoco') and c['meshes']
    blank = dv.new('USD cell')
    r = dv.robot_io.add_robot(blank, str(tmp_path / 'arm.usda'), 'Arm')
    assert r.primary and blank.robots[0].name == 'Arm'
    with pytest.raises(dv.DexVantageError, match='no such robot'):
        dv.robot_io.write_usd(rig, str(tmp_path / 'x.usda'), 'nope')
    assert dv.robot_io.openvla_encoding(rig)['state_encoding'] == 'JOINT'


def test_usd_without_pxr_says_what_to_install(tmp_path, monkeypatch):
    from dexvantage.core.robot import import_robot
    from dexvantage.core.robot import usd_import as U
    monkeypatch.setattr(U, 'have_pxr', lambda: False)
    p = tmp_path / 'arm.usda'
    p.write_text('#usda 1.0\n')
    with pytest.raises(ProjectFormatError, match='optional: install usd-core'):
        import_robot(str(p))
