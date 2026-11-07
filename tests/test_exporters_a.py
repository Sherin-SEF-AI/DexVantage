"""Exporters part A (core/exporters: mjcf, urdf, opencv_yaml, masks, plucker, brackets, fiftyone, lerobot_dataset),
the batch flag --export and the Export dialog.

References: a robot file written here with an off-origin hinge anchor, a non-zero reference angle, a slide and two
hinges in one body, posed by DexVantage, by an ElementTree walk with urdfdom's rules written again
(validation/formats_exports.urdf_fk) and by MuJoCo's URDF compiler; cv2.FileStorage and PyYAML reading the files
back; OpenCV's projectPoints and fisheye.projectPoints for the lens mappings; exact ray casting (core/raycast.py) for
the masks."""
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from conftest import pump
from dexvantage.core import exporters as X
from dexvantage.core import project as P
from dexvantage.core.exporters import _common as C
from dexvantage.core.exporters.calibration import lens_for
from dexvantage.core.frames import Pose, quat_from_rpy
from dexvantage.core.jobs import StudyError
from dexvantage.core.optics import CameraModel
from dexvantage.core.robot import import_robot
from dexvantage.core.robot.model import get_model
from dexvantage.core.robot.samples import sample_project
from dexvantage.validation import formats_exports as FE

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PART_A = ('mjcf', 'urdf', 'opencv_yaml', 'masks', 'plucker', 'brackets', 'fiftyone', 'lerobot_dataset')

ODD_MJCF = """<mujoco model="odd">
  <worldbody>
    <body name="base_link">
      <geom type="box" size="0.05 0.05 0.02" contype="0" conaffinity="0"/>
      <body name="arm" pos="0 0 0.1" quat="0.9659258 0 0 0.2588190">
        <joint name="j_anchor" type="hinge" axis="0 1 0" pos="0.1 0 0" ref="0.3" range="-2 2"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" contype="0" conaffinity="0"/>
        <body name="slider" pos="0.3 0 0">
          <joint name="j_slide" type="slide" axis="1 0 1" range="-0.05 0.1"/>
          <geom type="box" size="0.02 0.02 0.02" contype="0" conaffinity="0"/>
          <body name="wrist" pos="0.05 0 0" euler="0.2 0.1 -0.4">
            <joint name="j_a" type="hinge" axis="1 0 0" pos="0 0.02 0" range="-3 3"/>
            <joint name="j_b" type="hinge" axis="0 0 1" range="-3 3"/>
            <geom type="box" size="0.03 0.01 0.01" contype="0" conaffinity="0"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def odd_project(tmp_path):
    path = tmp_path / 'odd.xml'
    path.write_text(ODD_MJCF)
    p = P.blank_project('Odd arm')
    r, s = import_robot(str(path))
    r.primary = True
    p.robots.append(r)
    p.sources.update(s)
    p.variant.sensors.append(P.Sensor(name='Wrist cam', kind='camera',
                                      mount=P.Mount('wrist', r.uid, Pose([0.03, 0.01, 0.02],
                                                                         quat_from_rpy(0.3, -0.2, 1.1)))))
    p.variant.sensors.append(P.Sensor(name='Side cam', kind='camera', slot=1,
                                      mount=P.Mount('tripod', '', Pose([0.5, -0.4, 0.6],
                                                                       quat_from_rpy(0, np.pi / 2, 0.4)))))
    return p, r


def _mujoco_urdf(folder):
    import mujoco
    root = ET.fromstring(open(os.path.join(folder, 'ros2', 'rig.urdf'), encoding='utf-8').read())
    mj = ET.SubElement(root, 'mujoco')
    ET.SubElement(mj, 'compiler', fusestatic='false', discardvisual='true', boundmass='1e-6', boundinertia='1e-9')
    md = os.path.join(folder, 'ros2', 'meshes')
    assets = {f: open(os.path.join(md, f), 'rb').read() for f in os.listdir(md)} if os.path.isdir(md) else {}
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'), assets)


def test_part_a_formats_are_registered_with_descriptions():
    X.load_exporters()
    for fid in PART_A:
        assert fid in X.EXPORTERS, fid
        assert X.EXPORTERS[fid].title and len(X.EXPORTERS[fid].description) > 40


def test_urdf_of_a_robot_with_anchors_ref_slide_and_two_hinges(tmp_path):
    import mujoco
    p, r = odd_project(tmp_path)
    man = X.export(p, str(tmp_path / 'out'), ['urdf'])
    row = man['formats']['urdf']
    assert row['status'] == 'ok', row.get('reason')
    assert any('effort and velocity' in n for n in row['notes'])
    text = open(tmp_path / 'out' / 'ros2' / 'rig.urdf', encoding='utf-8').read()
    assert 'arm__j_anchor' in text            # the off-origin anchor gets its own link
    rm = get_model(p, r.uid)
    m = _mujoco_urdf(str(tmp_path / 'out'))
    d = mujoco.MjData(m)
    rng = np.random.default_rng(3)
    for _ in range(5):
        q = rm.lower + (rm.upper - rm.lower) * rng.uniform(0.1, 0.9, rm.nq)
        T_dv = rm.fk(q)
        T_xml, problems = FE.urdf_fk(text, dict(zip(rm.joint_names, q)))
        assert not problems
        d.qpos[:] = 0
        for jn, v in zip(rm.joint_names, q):
            d.qpos[m.joint(jn).qposadr[0]] = v
        mujoco.mj_kinematics(m, d)
        for link in ('base_link', 'arm', 'slider', 'wrist'):
            e_p, e_r = FE.pose_errors(T_dv[link], T_xml[link])
            assert e_p < 1e-12 and e_r < 1e-12, (link, e_p, e_r)
            b = m.body(link).id
            Tm = np.eye(4)
            Tm[:3, :3] = d.xmat[b].reshape(3, 3)
            Tm[:3, 3] = d.xpos[b]
            e_p, e_r = FE.pose_errors(T_dv[link], Tm)
            assert e_p < 1e-12 and e_r < 1e-12, (link, e_p, e_r)
        cams = C.rig_cameras(p, {r.uid: q})
        from dexvantage.core.camrender import optical_pose
        for c in cams:
            e_p, e_r = FE.pose_errors(optical_pose(p, c.sensor, c.imager, {r.uid: q}).matrix(), T_xml[c.optical_frame])
            assert e_p < 1e-12 and e_r < 1e-12
    # the launch file and the tree name every sensor frame, with the link or the world as parent
    nodes = FE._launch_nodes(open(tmp_path / 'out' / 'ros2' / 'static_transforms.launch.py', encoding='utf-8').read())
    parents = {c: pa for pa, c, _T in nodes}
    assert parents['wrist_cam_link'] == 'wrist' and parents['side_cam_link'] == 'world'
    import yaml
    tree = yaml.safe_load(open(tmp_path / 'out' / 'ros2' / 'tf_tree.yaml', encoding='utf-8'))
    assert {f['child'] for f in tree['frames']} == set(parents)
    assert {f['child']: f['moves_with_robot'] for f in tree['frames']}['wrist_cam_link'] is True


def test_urdf_refuses_an_unlimited_slide(tmp_path):
    (tmp_path / 'slide.xml').write_text(ODD_MJCF.replace(' range="-0.05 0.1"', ''))
    p = P.blank_project('Slide')
    r, s = import_robot(str(tmp_path / 'slide.xml'))
    r.primary = True
    p.robots.append(r)
    p.sources.update(s)
    for j in r.joints:
        if j.get('name') == 'j_slide':
            j['lower'] = j['upper'] = None
    row = X.export(p, str(tmp_path / 'o'), ['urdf'])['formats']['urdf']
    assert row['status'] == 'refused' and 'prismatic' in row['reason'] and 'Traceback' not in row['reason']


def test_rpy_writer_keeps_precision_near_gimbal_lock():
    from dexvantage.core.exporters.ros2 import rpy_of
    for pitch in (np.pi / 2, np.pi / 2 - 1e-9, -np.pi / 2 + 3e-8, 0.3):
        R = (Pose([0, 0, 0], quat_from_rpy(0.4, pitch, -1.2))).matrix()[:3, :3]
        r, pp, y = rpy_of(R)
        R2 = FE._qmat(FE.urdf_quat_rpy(r, pp, y))
        assert FE.rot_angle(R, R2) < 1e-12, pitch


def test_mjcf_snippets_and_scene(tmp_path):
    import mujoco
    p = sample_project('so100')
    out = str(tmp_path / 'm')
    row = X.export(p, out, ['mjcf'])['formats']['mjcf']
    assert row['status'] == 'ok'
    snip = ET.parse(os.path.join(out, 'mjcf', 'cameras_snippet.xml')).getroot()
    cams = C.rig_cameras(p)
    assert len(snip.findall('camera')) == len(cams) and len(snip.findall('site')) == 2
    for e in snip.findall('camera'):
        assert {'pos', 'quat', 'resolution', 'sensorsize', 'focalpixel', 'principalpixel'} <= set(e.attrib)
    m = mujoco.MjModel.from_xml_path(os.path.join(out, 'mjcf', 'scene', 'scene.xml'))
    assert m.ncam == len(cams)
    assert all(os.path.isfile(os.path.join(out, f)) for f in row['files'])
    with pytest.raises(StudyError):
        X.EXPORTERS['mjcf'].fn(p, out, {'mode': 'everything'}, None)


def test_mjcf_snippet_check_on_the_samples(tmp_path):
    from dexvantage.validation.suite import Ctx, ValidationSettings
    ctx = Ctx(None, None, None, ValidationSettings(quick=True))
    try:
        err, ok, det = FE.mjcf_snippets(ctx)
    finally:
        ctx.close()
    assert err < 1e-9 and ok, det


@pytest.mark.parametrize('model,dist,f,opencv,kalibr,exact_cv,exact_kb', [
    ('pinhole', (), 500.0, 'plumb_bob', 'none', True, True),
    ('brown', (-0.2, 0.05, 1e-3, -5e-4, 0.0), 500.0, 'plumb_bob', 'radtan', True, True),
    ('brown', (-0.2, 0.05, 1e-3, -5e-4, -0.01), 500.0, 'plumb_bob', 'radtan', True, False),
    ('brown', (0.1, -0.02, 1e-4, 2e-4, 0.001, 0.3, -0.01, 0.002), 500.0, 'rational_polynomial', 'radtan', True,
     False),
    ('kb', (0.05, -0.01, 0.002, -0.0003), 600.0, 'fisheye', 'equidistant', True, True),
    ('equidistant', (), 600.0, 'fisheye', 'equidistant', True, True),
    ('equisolid', (), 600.0, 'fisheye', 'equidistant', False, False),
    ('stereographic', (), 600.0, 'fisheye', 'equidistant', False, False),
])
def test_lens_mapping_per_target(model, dist, f, opencv, kalibr, exact_cv, exact_kb):
    cam = CameraModel(1280, 720, f, f, 639.5, 359.5, model, dist)
    a, b = lens_for(cam, 'opencv'), lens_for(cam, 'kalibr')
    assert (a['model'], b['model']) == (opencv, kalibr)
    assert (a['exact'], b['exact']) == (exact_cv, exact_kb)
    for L, exact in ((a, exact_cv), (b, exact_kb)):
        assert (L['note'] == '') == exact
        if not exact:
            assert 'no exact' in L['note'] and L['error_px'] is not None
    if model == 'brown' and not exact_kb:
        # the least-squares radtan beats dropping k3 (and the rational terms)
        drop = CameraModel(1280, 720, f, f, 639.5, 359.5, 'brown', tuple(dist[:4]) + (0.0,))
        assert b['error_px'] < C.approx_error_px(cam, drop)[0]


def test_a_fisheye_beyond_90_degrees_says_so():
    cam = CameraModel(1280, 720, 300.0, 300.0, 639.5, 359.5, 'kb', (0.0, 0.0, 0.0, 0.0))
    L = lens_for(cam, 'opencv')
    assert not L['exact'] and '90°' in L['note']


def test_opencv_yaml_kalibr_and_json_read_back(tmp_path):
    import cv2
    import yaml
    p = sample_project('ur5e')
    out = str(tmp_path / 'c')
    row = X.export(p, out, ['opencv_yaml'])['formats']['opencv_yaml']
    assert row['status'] == 'ok'
    cams = C.rig_cameras(p)
    for c in cams:
        fs = cv2.FileStorage(os.path.join(out, 'calibration', 'opencv', c.key + '.yaml'), cv2.FILE_STORAGE_READ)
        K = fs.getNode('camera_matrix').mat()
        T = fs.getNode('T_parent_camera').mat()
        assert fs.getNode('distortion_model').string() == 'plumb_bob'
        assert fs.getNode('parent_frame').string() == c.parent
        fs.release()
        assert K[0, 0] == c.camera.fx and K[1, 2] == c.camera.cy
        assert np.array_equal(T, c.T_parent_optical)
    chain = yaml.safe_load(open(os.path.join(out, 'calibration', 'kalibr', 'camchain.yaml'), encoding='utf-8'))
    assert sorted(chain) == [f'cam{i}' for i in range(len(cams))]
    T10 = np.asarray(chain['cam1']['T_cn_cnm1'])
    assert np.allclose(T10, np.linalg.inv(cams[1].T_world_optical) @ cams[0].T_world_optical, atol=1e-15)
    doc = json.load(open(os.path.join(out, 'calibration', 'cameras.json'), encoding='utf-8'))
    assert 'provenance' in doc and len(doc['cameras']) == len(cams)
    assert doc['cameras'][0]['extrinsics']['T_parent_optical']['quaternion_xyzw']


def test_calibration_round_trip_check_with_every_lens_model(tmp_path):
    from dexvantage.validation.suite import Ctx, ValidationSettings
    ctx = Ctx(None, None, None, ValidationSettings(quick=True))
    try:
        e1, ok1, d1 = FE.opencv_yaml(ctx)
        e2, ok2, d2 = FE.kalibr_camchain(ctx)
    finally:
        ctx.close()
    assert e1 == 0.0 and ok1, d1['problems']
    assert e2 == 0.0 and ok2, d2['problems']


def test_masks_export_matches_ray_casting(tmp_path):
    import cv2

    from dexvantage.core import raycast as RC
    from dexvantage.core.scenegeom import sensor_occluders
    p = sample_project('ur5e')
    out = str(tmp_path / 'k')
    row = X.export(p, out, ['masks'], {'masks': {'scale': 0.25, 'atlas_grasps': 2, 'atlas_scale': 0.1}})
    row = row['formats']['masks']
    assert row['status'] == 'ok', row.get('reason')
    idx = json.load(open(os.path.join(out, 'masks', 'masks.json'), encoding='utf-8'))
    assert len(idx['cameras']) == 2 and idx['atlas']['used'] == 2 and len(idx['depth']) == 2
    cams = {c.key: c for c in C.rig_cameras(p, None, 'primary')}
    for rec in idx['cameras']:
        c = cams[rec['name']]
        mask = cv2.imread(os.path.join(out, rec['mask']), cv2.IMREAD_UNCHANGED) > 0
        owner = cv2.imread(os.path.join(out, rec['owner']), cv2.IMREAD_UNCHANGED)
        assert owner.dtype == np.uint16 and owner.max() <= len(idx['owners'])
        robot_owner = np.array([False] + [o.startswith('robot:') for o in idx['owners']])
        assert np.array_equal(mask, robot_owner[owner])
        cam = c.camera.scaled(*rec['size'])
        vs, us = np.mgrid[0:cam.height:3, 0:cam.width:3]
        rays, ok = cam.unproject(np.stack([us.ravel(), vs.ravel()], 1).astype(float), with_ok=True)
        o = c.T_world_optical[:3, 3]
        occ = sensor_occluders(p, c.sensor, None, origins=o[None])
        rob = np.array([w.startswith('robot:') for w in occ.owners])[occ.owner]
        dw = rays[ok] @ c.T_world_optical[:3, :3].T
        dr, dn = RC.first_hit(o, dw, occ.tris[rob]), RC.first_hit(o, dw, occ.tris[~rob])
        ref = np.isfinite(dr) & (dr < dn)
        assert (ref != mask[vs.ravel()[ok], us.ravel()[ok]]).mean() < 1e-3
    for rec in idx['depth']:
        code = cv2.imread(os.path.join(out, rec['code']), cv2.IMREAD_UNCHANGED)
        assert set(np.unique(code)) <= {0, 1, 2, 3, 4, 5, 6, 255}
        assert abs(rec['fractions']['valid'] - (code == 0).mean()) < 1e-12
        valid = cv2.imread(os.path.join(out, rec['valid']), cv2.IMREAD_UNCHANGED)
        assert np.array_equal(valid > 0, code == 0)
    for a in idx['atlas']['cameras']:
        img = cv2.imread(os.path.join(out, a['atlas']), cv2.IMREAD_UNCHANGED)
        assert img.shape == (a['grid'][1] * a['tile'][1], a['grid'][0] * a['tile'][0])


def test_masks_refuse_bad_options_and_need_cameras(tmp_path):
    p = sample_project('so100')
    with pytest.raises(StudyError):
        X.EXPORTERS['masks'].fn(p, str(tmp_path), {'scale': 2.0}, None)
    for s in p.variant.sensors:
        s.enabled = False
    row = X.export(p, str(tmp_path / 'x'), ['masks'])['formats']['masks']
    assert row['status'] == 'refused' and 'camera' in row['reason']


def test_plucker_brackets_and_wrappers_refuse_in_words(tmp_path):
    from dexvantage.core import brackets as B
    p = sample_project('ur5e')
    man = X.export(p, str(tmp_path / 'w'), ['plucker', 'brackets', 'fiftyone'])
    f = man['formats']
    assert f['plucker']['status'] == 'ok' and any(x.endswith('.safetensors') for x in f['plucker']['files'])
    assert f['brackets']['status'] == 'refused' and 'bracket type' in f['brackets']['reason']
    assert f['fiftyone']['status'] == 'refused' and 'dataset' in f['fiftyone']['reason']
    s = p.variant.sensors[0]
    s.mount.bracket = B.suggest_kind(p, s)[0]
    row = X.export(p, str(tmp_path / 'b'), ['brackets'])['formats']['brackets']
    assert row['status'] == 'ok' and any(x.endswith('.stl') for x in row['files'])
    data = open(tmp_path / 'b' / row['files'][0], 'rb').read()
    from dexvantage.core.meshing import read_stl
    V, F, _n, _h = read_stl(data)
    assert len(F) > 1000
    os.makedirs(tmp_path / 'l' / 'lerobot_dataset')
    (tmp_path / 'l' / 'lerobot_dataset' / 'x').write_text('x')
    row = X.export(p, str(tmp_path / 'l'), ['lerobot_dataset'])['formats']['lerobot_dataset']
    assert row['status'] == 'refused' and 'not empty' in row['reason']


def test_step_without_gmsh_is_a_plain_refusal(tmp_path, monkeypatch):
    import builtins
    from dexvantage.core import brackets as B
    real = builtins.__import__

    def no_gmsh(name, *a, **k):
        if name == 'gmsh':
            raise ImportError('no gmsh')
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, '__import__', no_gmsh)
    p = sample_project('ur5e')
    s = p.variant.sensors[0]
    s.mount.bracket = B.suggest_kind(p, s)[0]
    row = X.export(p, str(tmp_path), ['brackets'], {'brackets': {'step': True}})['formats']['brackets']
    assert row['status'] == 'refused' and 'gmsh' in row['reason'] and 'dexvantage[step]' in row['reason']
    assert not os.path.exists(tmp_path / 'brackets')


def test_step_through_gmsh_reads_back(tmp_path):
    pytest.importorskip('gmsh')
    import gmsh

    from dexvantage.core.exporters.bridges import step_from_mesh
    V = np.array([(x, y, z) for x in (0, 10) for y in (0, 20) for z in (0, 5)], float)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    F = np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])
    path = str(tmp_path / 'box.step')
    step_from_mesh(V, F, path)
    gmsh.initialize([], readConfigFiles=False)
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        ents = gmsh.model.occ.importShapes(path)
        gmsh.model.occ.synchronize()
        vols = [t for d, t in ents if d == 3]
        assert len(vols) == 1 and abs(gmsh.model.occ.getMass(3, vols[0]) - 1000.0) < 1e-9
    finally:
        gmsh.finalize()


def _run(*args):
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    return subprocess.run([sys.executable, '-m', 'dexvantage', *args], capture_output=True, text=True, env=env,
                          cwd=ROOT, timeout=300)


def test_batch_export_flags(tmp_path):
    out = str(tmp_path / 'e')
    r = _run('--sample', 'so100', '--export', out, '--formats', 'urdf,opencv_yaml,brackets',
             '--export-options', '{"opencv_yaml": {"imagers": "primary"}}')
    assert r.returncode == 1, r.stderr                 # brackets refused (no bracket type), the others written
    assert 'brackets: refused' in r.stderr and 'Traceback' not in r.stderr
    man = json.load(open(os.path.join(out, 'export_manifest.json'), encoding='utf-8'))
    assert man['formats']['urdf']['status'] == 'ok' and 'provenance' in man
    r = _run('--sample', 'so100', '--export', out, '--formats', 'urdf,no_such_format')
    assert r.returncode == 2 and len(r.stderr.strip().splitlines()) == 1 and 'no_such_format' in r.stderr
    r = _run('--sample', 'so100', '--export', out)
    assert r.returncode == 2 and '--formats' in r.stderr
    r = _run('--sample', 'so100', '--export', out, '--formats', 'urdf', '--export-options', '{"mjcf": {}}')
    assert r.returncode == 2 and 'mjcf' in r.stderr


def test_export_dialog_runs_a_job_and_shows_the_manifest(app, win, tmp_path):
    win.open_sample('so100')
    pump(app, 0.1)
    a = win.actions_by_id['export_formats']
    assert a.isEnabled() and not (a.data() or {}).get('phase')
    assert win.actions_by_id['export'] in win.m_file.actions()
    d = win.export_dialog()
    assert {'mjcf', 'urdf', 'opencv_yaml', 'masks'} <= set(d.rows)
    assert not d.start()                                # nothing ticked: said in the hint
    d.set_chosen(['urdf', 'brackets'])
    d.rows['urdf'].extra.setText('{not json')
    d.folder.setText(str(tmp_path / 'dlg'))
    assert not d.start() and 'JSON' in d.hint.text()
    d.rows['urdf'].extra.setText('')
    d.pose.setCurrentIndex(1)
    assert 'q' in d.options()['urdf']
    assert d.start()
    for _ in range(600):
        pump(app, 0.05)
        if d.go.isEnabled():
            break
    text = d.result.toPlainText()
    assert 'urdf' in text and 'refused' in text and 'Folder: dlg' in text
    assert os.path.isfile(tmp_path / 'dlg' / 'export_manifest.json')
    d.close()


def test_lerobot_dataset_and_fiftyone_wrappers(tmp_path):
    p = sample_project('so100')
    out = str(tmp_path / 'd')
    row = X.export(p, out, ['lerobot_dataset'], {'lerobot_dataset': {'episodes': 1, 'frames': 20,
                                                                     'max_width': 160}})['formats']['lerobot_dataset']
    assert row['status'] == 'ok', row.get('reason')
    info = json.load(open(os.path.join(out, 'lerobot_dataset', 'meta', 'info.json'), encoding='utf-8'))
    assert 'provenance' not in info and info['codebase_version'] == 'v3.0'
    truth = json.load(open(os.path.join(out, 'lerobot_dataset', 'dexvantage_truth.json'), encoding='utf-8'))
    assert 'provenance' in truth
    ds = os.path.join(out, 'lerobot_dataset')
    row = X.export(p, str(tmp_path / 'f'), ['fiftyone'], {'fiftyone': {'dataset': ds}})
    assert row['formats']['fiftyone']['status'] == 'ok', row['formats']['fiftyone'].get('reason')
    assert row['formats']['fiftyone']['files']
