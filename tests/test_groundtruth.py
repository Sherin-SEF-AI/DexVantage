"""MuJoCo ground truth (core/sim/scene.py, core/sim/render.py), the 'Rendered against predicted' study
(core/studies/groundtruth.py), its page, its checks, its batch flag and its self-test line."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import pump
from dexvantage.core import jobs
from dexvantage.core.frames import Pose, quat_from_axis_angle
from dexvantage.core.optics import CameraModel
from dexvantage.core.project import TaskObject
from dexvantage.core.robot.samples import sample_project
from dexvantage.core.sim import render as RD
from dexvantage.core.sim import scene as SC
from dexvantage.core.studies import groundtruth as GT

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
FAST = {'grasps': 6, 'thumbs': 2, 'thumb_px': 160}


@pytest.fixture(scope='module')
def ur5e():
    return sample_project('ur5e')


@pytest.fixture(scope='module')
def ur5e_doc():
    p = sample_project('ur5e')
    p.analysis.studies[GT.SID] = dict(FAST)
    return jobs.run_study(GT.SID, p)


# ---------------------------------------------------------------------- MuJoCo 3.17 camera attributes
def test_mujoco_camera_attributes_as_documented():
    """MuJoCo 3.17.0: with sensorsize set, focalpixel and principalpixel are converted with the resolution into
    cam_intrinsic (lengths) and fovy is ignored (doc/XMLreference.rst, body/camera); the intrinsics are read again at
    every render, so a job can override them."""
    import mujoco
    s = mujoco.MjSpec()
    c = s.worldbody.add_camera(name='c')
    c.sensor_size = [848e-6, 480e-6]
    c.resolution = [848, 480]
    c.focal_pixel = [420.5, 421.25]
    c.principal_pixel = [3.25, -2.5]
    c.fovy = 10.0                                       # ignored: sensorsize is set
    m = s.compile()
    fx, fy, px, py = (float(v) for v in m.cam_intrinsic[0])
    assert fx == pytest.approx(420.5 * 848e-6 / 848, rel=1e-6) and fy == pytest.approx(421.25 * 480e-6 / 480, rel=1e-6)
    assert px == pytest.approx(3.25e-6, rel=1e-5) and py == pytest.approx(-2.5e-6, rel=1e-5)
    assert float(m.cam_fovy[0]) == pytest.approx(2 * math.degrees(math.atan(240 / 421.25)), rel=1e-5)
    # a camera whose intrinsic is set directly (not through focal) compiles to zero: the scene uses focalpixel
    s2 = mujoco.MjSpec()
    c2 = s2.worldbody.add_camera(name='c')
    c2.sensor_size = [320, 240]
    c2.intrinsic = [300, 300, 0, 0]
    assert not np.any(s2.compile().cam_intrinsic[0])


def test_scene_camera_matches_the_lens_model(ur5e):
    from dexvantage.validation import mjrender as MR
    b = SC.build_scene(ur5e)
    assert len(b.cameras) == 6 and set(b.camera_names) == {s.uid for s in ur5e.variant.sensors}
    for c in b.cameras.values():
        cm = MR.camera_model(b.model, c.name, c.model.width, c.model.height)
        for k in ('fx', 'fy', 'cx', 'cy'):
            assert getattr(cm, k) == pytest.approx(getattr(c.model, k), abs=1e-3)
    # the primary imager is the one the grasp study uses (RGB of a depth sensor)
    assert all(b.imagers[u][0] == 'RGB' for u in b.camera_names)
    assert b.model.vis.map.znear * b.model.stat.extent == pytest.approx(SC.NEAR_M, rel=1e-6)


# ---------------------------------------------------------------------- the scene
def test_scene_poses_follow_robot_model(ur5e):
    import mujoco

    from dexvantage.core import camrender as CR
    from dexvantage.core import library as LB
    from dexvantage.core.robot.model import get_model
    from dexvantage.validation import mjrender as MR
    b = SC.build_scene(ur5e)
    r = ur5e.robots[0]
    rm = get_model(ur5e, r.uid)
    rng = np.random.default_rng(3)
    q = np.clip(rm.home + rng.normal(0, 0.4, rm.nq), rm.lower, rm.upper)
    d = mujoco.MjData(b.model)
    d.qpos[:] = b.qpos({r.uid: q})
    mujoco.mj_forward(b.model, d)
    T = rm.fk(q)
    for ln in ('wrist_3_link', 'gripper_base', 'gripper_right_pad'):
        bid = b.model.body(b.robots[r.uid]['prefix'] + ln).id
        assert np.allclose(d.xpos[bid], T[ln][:3, 3], atol=1e-9)
        assert b.body_map[bid] == ('robot', r.uid, ln) and b.body_map[b.model.body(bid).name] == b.body_map[bid]
    # the wrist camera rides on the wrist: its optical frame is camrender's at q
    s = ur5e.variant.sensors[0]
    c = b.camera(s.uid)
    R, p = MR.mujoco_camera(b.model, d, c.name)
    im = next(i for i in LB.sensor_imagers(s) if i.name == c.imager)
    want = CR.optical_pose(ur5e, s, im, {r.uid: q})
    assert np.allclose(p, want.position, atol=1e-9) and np.allclose(R, want.matrix()[:3, :3], atol=1e-9)


def test_scene_hides_what_the_predictions_do_not_draw(ur5e):
    b = SC.build_scene(ur5e)
    m = b.model
    hidden = np.flatnonzero(m.geom_group == SC.HIDDEN_GROUP)
    assert len(hidden) > 0 and np.all(b.geom_owner[hidden] == -1)
    shown = np.flatnonzero(b.geom_owner >= 0)
    assert np.all(m.geom_group[shown] < 3)                    # drawn by MuJoCo's default visual options
    # collision geoms of the gripper (contype set) are hidden; the visual ones are owned
    assert all(m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 for g in shown
               if b.body_map.get(int(m.geom_bodyid[g]), ('',))[0] == 'robot')
    assert np.all(m.site_group == SC.HIDDEN_GROUP)
    assert ('surface', 'table') in b.owners
    assert any('group 5' in n for n in b.notes) and any('1 mm thick' in d for d in b.defaults)


def test_objects_surfaces_and_material_tags():
    p = sample_project('ur5e')
    p.workspace.objects = [
        TaskObject(name='Cube', shape='box', size=(0.05, 0.06, 0.07), pose=Pose(np.array([0.4, 0.0, 0.035])),
                   materials=['glossy', 'black']),
        TaskObject(name='Can', shape='cylinder', size=(0.03, 0.1, 0.0), pose=Pose(np.array([0.5, 0.1, 0.05])),
                   materials=['transparent']),
        TaskObject(name='Ball', shape='sphere', size=(0.02, 0, 0), pose=Pose(np.array([0.3, -0.1, 0.02])),
                   materials=['metallic']),
        TaskObject(name='Wedge', shape='mesh', mesh_key='wedge', pose=Pose(np.array([0.4, -0.2, 0.0]))),
        TaskObject(name='Nothing', shape='box', size=(0, 0, 0)),
    ]
    V = np.array([[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]], float)
    p.meshes['wedge'] = V
    p.meshes['wedge/faces'] = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    p.workspace.surfaces.append({'uid': 'shelf', 'name': 'Shelf', 'z': 0.4, 'size': [0.3, 0.2], 'center': [0.6, 0],
                                 'thickness': 0.02, 'material': 'black'})
    b = SC.build_scene(p)
    m = b.model
    for o in p.workspace.objects[:4]:
        assert ('object', o.uid) in b.owners
    assert any('Nothing is left out' in n for n in b.notes)
    cube = m.material(f'mat_object_{p.workspace.objects[0].uid}')
    assert np.allclose(cube.rgba[:3], SC.MATERIALS['black']['rgb']) and cube.specular == pytest.approx(0.8)
    can = m.material(f'mat_object_{p.workspace.objects[1].uid}')
    assert can.rgba[3] == pytest.approx(0.3)
    g = m.geom(f'object_{p.workspace.objects[1].uid}')
    assert g.size[0] == pytest.approx(0.03) and g.size[1] == pytest.approx(0.05)
    shelf = m.geom('surface_shelf')
    assert shelf.size[2] == pytest.approx(0.01)
    bid = m.body('surface_shelf').id
    assert m.body_pos[bid][2] + shelf.size[2] == pytest.approx(0.4)        # its top at z


def test_save_scene_is_self_contained(ur5e, tmp_path):
    import mujoco
    path = SC.save_scene(ur5e, str(tmp_path / 'scene'))
    assert os.path.basename(path) == 'scene.xml'
    m = mujoco.MjModel.from_xml_path(path)
    assert m.ncam == 6 and m.nmesh > 0
    xml = open(path, encoding='utf-8').read()
    assert '/home/' not in xml and str(tmp_path) not in xml and SC.VIEW_CAMERA not in xml
    assert os.path.isdir(tmp_path / 'scene' / 'assets')
    notes = open(tmp_path / 'scene' / 'scene_notes.txt', encoding='utf-8').read()
    assert 'Wrist_D405__RGB' in notes and 'Defaults:' in notes


# ---------------------------------------------------------------------- renders
def test_renderer_outputs(ur5e):
    from dexvantage.core.robot.model import get_model
    s0, s1 = ur5e.variant.sensors
    with RD.Renderer(ur5e) as r:
        info = r.info
        assert info['gl'] == 'egl' and info['renderer'] and info['mujoco'] == '3.17.0'
        imgs = r.render([{'sensor_uid': s0.uid, 'q': None, 'outputs': ('rgb', 'depth', 'seg'), 'scale': 0.25,
                          'extrinsic_delta': None},
                         {'sensor_uid': s1.uid, 'outputs': ('seg',), 'scale': 0.25},
                         {'sensor_uid': s1.uid, 'outputs': ('seg',), 'scale': 0.25,
                          'extrinsic_delta': Pose(np.zeros(3), quat_from_axis_angle([0, 0, 1], math.radians(2)))},
                         {'sensor_uid': s0.uid, 'outputs': ('seg',), 'scale': 0.25,
                          'q': {ur5e.robots[0].uid: get_model(ur5e, ur5e.robots[0].uid).home + 0.3}}])
        assert r.stats['frames'] == 6 and r.stats['frames_per_s'] > 0
    a = imgs[0]
    assert a['rgb'].shape == (180, 320, 3) and a['rgb'].dtype == np.uint8
    assert a['depth'].dtype == np.float32 and a['seg'].dtype == np.int32 and a['path'] == 'pinhole'
    assert np.all((a['depth'] > 0) == (a['seg'] >= 0))
    assert set(np.unique(a['seg'])) - {-1} <= set(a['seg_owner'])
    assert ('surface', 'table') in a['seg_owner'].values()
    assert a['model'].width == 320 and a['sensor_uid'] == s0.uid
    assert (imgs[1]['seg'] != imgs[2]['seg']).mean() > 0.01           # a 2° mount slip moves the image
    assert (a['seg'] != imgs[3]['seg']).mean() > 0.01                  # the wrist camera follows q


def test_renders_agree_with_the_camera_view(ur5e):
    from dexvantage.core import camrender as CR
    from dexvantage.core import library as LB
    with RD.Renderer(ur5e) as r:
        s = ur5e.variant.sensors[1]
        im = r.render([{'sensor_uid': s.uid, 'outputs': ('seg', 'depth'), 'scale': 0.25}])[0]
        c = r.build.camera(s.uid)
    g = CR.gather(ur5e, None)
    imager = next(i for i in LB.sensor_imagers(s) if i.name == c.imager)
    rr = CR.render(g, im['model'], CR.optical_pose(ur5e, s, imager, None), floor=False)
    names = np.array(list(g.owners) + [''])
    pred = names[np.where(rr.tri >= 0, g.owner[np.clip(rr.tri, 0, None)], -1)]
    mjn = np.array([':'.join(o) for _k, o in sorted(im['seg_owner'].items())] + [''])
    mj = mjn[np.where(im['seg'] >= 0, im['seg'], len(mjn) - 1)]
    assert (pred == mj).mean() > 0.999
    ok = (rr.tri >= 0) & (im['seg'] >= 0)
    assert np.median(np.abs(rr.dist[ok] - im['range'][ok])) < 1e-4


@pytest.mark.parametrize('lens', ['brown', 'equidistant'])
def test_distortion_remap_matches_the_lens_model(lens):
    """A sphere seen through a distorted lens: the rendered mask's centroid within 0.4 px of the predicted mask's."""
    import mujoco

    from dexvantage.validation import mjrender as MR
    cam = (CameraModel(320, 240, 240, 241, 160.0, 119.0, 'brown', (-0.25, 0.07, 0.0, 0.0, 0.0)) if lens == 'brown'
           else CameraModel(320, 240, 100, 100, 159.5, 119.5, 'equidistant'))
    T = np.eye(4)
    T[:3, :3] = MR.look_at([0, 0, 2], [0, 0, 0])
    T[:3, 3] = [0, 0, 2]
    rad = 0.012 if lens == 'brown' else 0.03
    pts = []
    for u, v in ((40, 30), (280, 200), (160, 120), (300, 40)):
        ray = cam.unproject(np.array([[u, v]], float))[0]
        pts.append(ray * 0.8)
    xml = ('<mujoco><visual><quality numslices="128" numstacks="64"/><map znear="0.001"/></visual><worldbody>'
           '<camera name="cam" pos="0 0 2"/>')
    W = np.array(pts) @ T[:3, :3].T + T[:3, 3]
    for k, c in enumerate(W):
        xml += f'<geom name="f{k}" type="sphere" size="{rad}" pos="{c[0]:.9g} {c[1]:.9g} {c[2]:.9g}"/>'
    m = mujoco.MjModel.from_xml_string(xml + '</worldbody></mujoco>')
    with MR.Session() as sess:
        out = RD.render_lens(sess, m, 'cam', T, cam, ('seg', 'depth'))
    assert out['path'] == ('wide' if lens == 'brown' else 'cube')
    rays = np.nan_to_num(cam.pixel_rays())
    for k, P in enumerate(pts):
        d = np.linalg.norm(P)
        pred = rays @ (P / d) > math.cos(math.asin(rad / d))
        vm, um = np.nonzero(out['seg'] == m.geom(f'f{k}').id)
        vp, up = np.nonzero(pred)
        assert len(um) > 20
        assert math.hypot(um.mean() - up.mean(), vm.mean() - vp.mean()) < 0.4
    # depth is the lens's optical z: the sphere's visible surface lies between its front and its centre's depth
    z = out['depth'][out['seg'] == m.geom('f2').id]
    assert np.all(z > pts[2][2] - rad - 1e-3) and np.all(z < pts[2][2] + 1e-3) and z.min() < pts[2][2] - 0.8 * rad


def test_lens_plans():
    assert RD.lens_plan(CameraModel(64, 48, 50, 50, 31.5, 23.5)).path == 'pinhole'
    assert RD.lens_plan(CameraModel(64, 48, 50, 50, 31.5, 23.5, 'brown', (0, 0, 0, 0, 0))).path == 'pinhole'
    pl = RD.lens_plan(CameraModel(64, 48, 15, 15, 31.5, 23.5, 'equisolid'))
    assert pl.path == 'cube' and len(pl.subs) >= 5
    assert sum(int(s.sel.sum()) for s in pl.subs) == int(pl.valid.sum())       # every pixel from one face
    w = RD.lens_plan(CameraModel(64, 48, 50, 50, 31.5, 23.5, 'brown', (-0.2, 0, 0, 0, 0)))
    assert w.path == 'wide' and w.subs[0].K.width >= 64 * RD.OVERSAMPLE * 0.9


def test_mjrender_session_overrides_are_per_job():
    import mujoco

    from dexvantage.validation import mjrender as MR
    m = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><camera name="c" pos="0 0 1" fovy="60"/><geom type="box" size="2 2 0.01" pos="0 0 -0.01"/>'
        '<geom name="s" type="sphere" size="0.05" pos="0 0 0.3"/></worldbody></mujoco>')
    with MR.Session() as s:
        a, info = s.render([{'model': m, 'camera': 'c', 'width': 64, 'height': 48, 'outputs': ('seg',)}])
        b, _ = s.render([{'model': m, 'camera': 'c', 'width': 64, 'height': 48, 'outputs': ('seg',),
                          'camera_pose': (0.2, 0, 1, 1, 0, 0, 0),
                          'camera_intrinsic': {'sensorsize': (64, 48), 'intrinsic': (80, 80, 0, 0)}}])
        c, _ = s.render([{'model': m, 'camera': 'c', 'width': 64, 'height': 48, 'outputs': ('seg',)}])
    assert info['frames'] == 1 and info['render_seconds'] >= 0 and info['device']
    assert np.array_equal(a[0]['seg'], c[0]['seg']) and not np.array_equal(a[0]['seg'], b[0]['seg'])


def test_render_unavailable_is_a_sentence():
    import mujoco

    from dexvantage.validation import mjrender as MR
    m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><camera name="c"/></worldbody></mujoco>')
    with pytest.raises(MR.RenderUnavailable) as e:
        with MR.Session(device=97, timeout=60) as s:
            s.render([{'model': m, 'camera': 'c', 'width': 8, 'height': 8, 'outputs': ('seg',)}])
    assert 'Traceback' not in str(e.value) and 'EGL' in str(e.value)


# ---------------------------------------------------------------------- the study
def test_study_result(ur5e_doc):
    doc = ur5e_doc
    json.dumps(doc, allow_nan=False)
    s = doc['summary']
    assert s['headline'] == 'contacts' and 0 <= s['agreement_pct'] <= 100
    assert s['word'] in GT.WORDS.values()
    c = doc['contacts']
    t = c['totals']
    assert t['tp'] + t['fp'] + t['fn'] + t['tn'] == t['n'] > 0
    n_traj = c['trajectories']['frames']
    assert n_traj == GT.DEFAULTS['trajectory_frames']           # the UR5e sample's one scripted pick
    assert c['grasp_set']['sampled'] == 6 and c['cases'] == 1 + c['grasp_set']['feasible'] + n_traj
    kinds = [r['kind'] for r in c['table']]
    assert kinds.count('home') == 1 and kinds.count('trajectory') == n_traj
    assert all('t' in r and r['trajectory_uid'] for r in c['table'] if r['kind'] == 'trajectory')
    assert sum(v['n'] for v in c['by_kind'].values()) == c['totals']['n']
    assert sum(r['n'] for r in c['table']) == t['n']
    assert sum(c['why'].values()) == t['fp'] + t['fn']
    assert c['table'][0]['case'] == 'home'
    assert len(c['cameras']) == 2 and c['grasp_level']['n'] == c['grasp_set']['feasible']
    dec = GT.decode(doc)
    assert dec['states'].shape[:2] == (c['cases'], 2) and dec['points'].shape[0] == c['cases']
    n_in = int(((dec['states'] != GT.ST_OUT)).sum())
    assert n_in == t['n']
    assert 0 < len(c['thumbs']) <= 2
    th = GT.decode_thumb(c['thumbs'][0])
    assert th['predicted_rgb'].shape == th['rendered_rgb'].shape and th['predicted_rgb'].shape[1] == 160
    assert len(th['uv']) == len(th['pred']) == len(th['rend'])
    d = doc['depth']
    assert len(d['sensors']) == 2 and all(x['compared_on'] == 'voxels' for x in d['sensors'])
    assert d['totals']['agreement'] > 0.99
    r = doc['render']
    assert r['renderer'] and r['frames'] > 0 and r['frames_per_s'] > 0
    assert len(doc['scene']['cameras']) == 6
    assert 'Known limits' in GT.__doc__


def test_contacts_agree_mostly_on_the_ur5e(ur5e_doc):
    c = ur5e_doc['contacts']
    assert c['totals']['agreement'] > 0.85
    assert c['why']['occlusion'] <= 0.02 * c['totals']['n']


def test_finger_links():
    from dexvantage.core.robot.model import get_model
    p = sample_project('ur5e')
    m = get_model(p, p.robots[0].uid)
    assert set(GT.finger_links(m, 'gripper_right_pad')) == {'gripper_right_pad', 'gripper_right_silicone_pad',
                                                             'gripper_right_follower'}
    q = sample_project('so100')
    assert GT.finger_links(get_model(q, q.robots[0].uid), 'Moving_Jaw') == ['Moving_Jaw']


def test_settings_and_status():
    p = sample_project('ur5e')
    st = GT.settings_of(p)
    assert st['grasps'] == 50 and 'grasps' in st['defaults'] and st['good_pct'] == 90.0
    p.analysis.studies[GT.SID] = {'depth_tol_mm': -1}
    with pytest.raises(jobs.StudyError):
        GT.settings_of(p)
    assert GT.status_for(95, GT.DEFAULTS) == ('good', 'agrees')
    assert GT.status_for(80, GT.DEFAULTS) == ('warning', 'mostly agrees')
    assert GT.status_for(10, GT.DEFAULTS) == ('critical', 'disagrees')


def test_without_grasps_or_depth_sensors():
    p = sample_project('dexhand')                       # no grasp set; a head depth camera
    p.analysis.studies[GT.SID] = {'grasps': 2}
    doc = jobs.run_study(GT.SID, p)
    assert 'grasp set' in doc['contacts']['skipped'] and doc['summary']['headline'] == 'depth'
    q = sample_project('so100')
    q.tasks.grasp_sets = []
    with pytest.raises(jobs.StudyError) as e:
        jobs.run_study(GT.SID, q)
    assert 'Nothing could be compared' in str(e.value) and 'grasp set' in str(e.value)


def test_trajectory_frames_without_a_grasp_set():
    """A project with a trajectory and no grasp set compares home and the trajectory's frames; the frames are posed
    where the trajectory says (§9.1: renders over grasp sets and trajectories)."""
    from dexvantage.core.robot import trajectory as TJ
    p = sample_project('ur5e')
    p.tasks.grasp_sets = []
    p.analysis.studies[GT.SID] = {'trajectory_frames': 3, 'thumbs': 0, 'depth': False}
    doc = jobs.run_study(GT.SID, p)
    c = doc['contacts']
    assert c['grasp_set'] is None and c['cases'] == 4 and c['grasp_level']['n'] == 0
    assert doc['summary']['headline'] == 'contacts' and c['totals']['n'] > 0
    tr = p.tasks.trajectories[0]
    rows = [r for r in c['table'] if r['kind'] == 'trajectory']
    t0, t1 = TJ.duration(tr)
    assert [r['t'] for r in rows] == pytest.approx([t0, (t0 + t1) / 2, t1])
    from dexvantage.core.robot.model import get_model
    m = get_model(p, tr['robot_uid'])
    q_mid = TJ.on_model(tr, m.joint_names, m.home, (t0 + t1) / 2)
    assert np.allclose(rows[1]['q'], q_mid)
    assert GT.trajectory_cases([tr], tr['robot_uid'], m.joint_names, m.home, 0) == []


# ---------------------------------------------------------------------- checks, self-test, batch
@pytest.mark.parametrize('cid', ['groundtruth.scene_intrinsics', 'groundtruth.lens_remap_fiducials'])
def test_checks_pass_quick(cid):
    from dexvantage import validation as VA
    from dexvantage.validation import suite
    suite.load_checks()
    spec = next(s for s in suite.CHECKS if s.id == cid)
    ctx = suite.Ctx(None, None, None, VA.ValidationSettings(quick=True))
    try:
        c = suite.run_one(spec, ctx)
    finally:
        ctx.close()
    assert c.passed, (c.error, c.tolerance, c.detail, c.exception)
    assert 'groundtruth' in suite.MODEL_IDS


def test_self_test_line():
    from dexvantage import selftest
    assert 'groundtruth' in selftest.NAMES
    status, msg = selftest._check_groundtruth({})
    assert status == selftest.OK and '6 cameras' in msg


def test_batch_flag(tmp_path):
    assert jobs.STUDIES.get(GT.SID) is not None or jobs.load_studies() is None
    assert jobs.STUDIES[GT.SID].batch == 'groundtruth'
    p = sample_project('ur5e')
    p.analysis.studies[GT.SID] = {'grasps': 3, 'thumbs': 0}
    path = str(tmp_path / 'cell.dexv')
    p.save(path)
    out = tmp_path / 'gt.json'
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    r = subprocess.run([sys.executable, '-m', 'dexvantage', path, '--groundtruth', str(out)], capture_output=True,
                       text=True, env=env, cwd=ROOT, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    doc = json.loads(out.read_text())
    assert doc['study'] == GT.SID and 'provenance' in doc and doc['contacts']['grasp_set']['sampled'] == 3


# ---------------------------------------------------------------------- the page
def test_page(app, win, tmp_path):
    assert 'groundtruth' in win.pages
    assert win.actions_by_id['mujoco_validate'].isEnabled() and win.actions_by_id['mujoco_scene'].isEnabled()
    assert win.open_sample('ur5e')
    page = win.pages['groundtruth']
    win.open_page('groundtruth')
    pump(app, 0.05)
    assert page.head.status.text.text() == 'not run yet'
    snap = win.store.project.snapshot()
    snap.analysis.studies[GT.SID] = dict(FAST)
    ur5e_doc = jobs.run_study(GT.SID, snap)
    page._finished(GT.SID, ur5e_doc)
    pump(app, 0.05)
    assert page.head.figure.text() not in ('', '—') and page.head.status.text.text() in GT.WORDS.values()
    assert page.matrix.topLevelItemCount() == 3 and page.why.topLevelItemCount() == 3
    assert page.cams.topLevelItemCount() == 2 and page.depth.topLevelItemCount() == 2
    assert page.cases.topLevelItemCount() == ur5e_doc['contacts']['cases']
    assert page.thumbs_grid.count() >= 3
    assert page.overlays(win.viewport)
    # a pose row puts the robot there (home first, then the grasps, lowest agreement first)
    assert page.cases.topLevelItem(0).data(0, 0x0100) == 0
    it = page.cases.topLevelItem(1)
    i = it.data(0, 0x0100)
    assert i > 0
    page.show_case(i)
    r = win.store.project.robots[0]
    assert np.allclose(win.store.q[r.uid], ur5e_doc['contacts']['table'][i]['q'], atol=1e-6)
    page.preview()
    # settings are undoable
    page.grasps.committed.emit(12.0)
    assert win.store.project.analysis.studies[GT.SID]['grasps'] == 12
    win.store.undo.undo()
    assert 'grasps' not in (win.store.project.analysis.studies.get(GT.SID) or {})
    # the Camera view overlay renders MuJoCo's segmentation in the background
    from dexvantage.core import camrender as CR
    from dexvantage.core import library as LB
    p = win.store.project
    s = p.variant.sensors[1]
    im = LB.sensor_imagers(s)[2]
    cm = im.camera.scaled(320, 180)
    pose = CR.optical_pose(p, s, im, {k: v for k, v in win.store.q.items()})
    assert page.camview_overlays(cm, pose) == []
    out = []
    for _ in range(300):
        pump(app, 0.05)
        out = page.camview_overlays(cm, pose)
        if out:
            break
    assert out and all(len(x) == 3 for x in out) and any('robot' in x[2] for x in out)
    # the scene folder
    folder = str(tmp_path / 'mj')
    assert page.save_scene_dialog(folder) == folder
    for _ in range(200):
        pump(app, 0.05)
        if os.path.exists(os.path.join(folder, 'scene.xml')) and 'mujoco_scene' not in win._jobs:
            break
    assert os.path.exists(os.path.join(folder, 'scene.xml'))
    exp = page.export(str(tmp_path / 'gt.json'))
    assert 'provenance' in json.loads(open(exp, encoding='utf-8').read())
    page._replaced()
    pump(app, 0.2)


def test_seg_outlines():
    from dexvantage.ui.pages.groundtruth import seg_outlines
    seg = np.full((40, 60), -1, np.int32)
    seg[10:20, 10:30] = 0
    seg[25:35, 30:50] = 1
    out = seg_outlines(seg, {0: ('robot', 'u', 'link'), 1: ('object', 'o')}, 2.0)
    assert {x[2] for x in out} == {'MuJoCo segmentation: robot', 'MuJoCo segmentation: object'}
    pl = out[1][0][0] if out[0][2].endswith('object') else out[0][0][0]
    assert pl.min() >= 0 and pl.max() < 120
