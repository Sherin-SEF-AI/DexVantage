"""Depth validity (spec §7.2): the five conditions, the error model, the per-sensor settings, the study on the UR5e
sample, the Depth page with its min-Z zones, and the depth validation checks.

References: σ_z and the focal length written out by hand here; min-Z figures typed from Intel's D400 datasheet
Table 4-12 (read 2026-11-06, the URL in validation/checks_depth.py); the stereo overlap against Intel's §4.4 formula;
the conditions of hand-placed points whose answer follows from the geometry; occlusion against exact ray casting."""
import copy
import json
import math

import numpy as np
import pytest

from conftest import pump
from dexvantage.core import jobs, library as L
from dexvantage.core.frames import Pose
from dexvantage.core.project import Project, Sensor, TaskObject
from dexvantage.core.studies import depth_validity as DV


def project_with(model_id, res=None, **cfg):
    p = Project('depth test')
    s = Sensor(name='Depth', kind='depth', model_id=model_id)
    p.variant.sensors = [s]
    c = dict(cfg)
    if res is not None:
        c['resolution'] = list(res)
    if c:
        p.analysis.studies['depth_validity'] = {'sensors': {s.uid: c}}
    return p, s


def optical_point(ds, x, y, z, body=Pose()):
    """World point at (x, y, z) in the reference imager's optical frame."""
    return DV.imager_world(body, ds.ref).apply(np.array([[x, y, z]], float))[0]


# ====================================================================== the error model
def test_sigma_z_closed_form():
    # z = 0.5 m, f = 700 px, B = 50 mm, σ_d = 0.08 px: 0.25 × 0.08 / 35 = 5.714 × 10⁻⁴ m
    assert DV.sigma_z(0.5, 700.0, 0.05, 0.08) == pytest.approx(0.25 * 0.08 / 35.0, rel=1e-12)
    z = np.array([0.2, 1.0, 2.0])
    np.testing.assert_allclose(DV.sigma_z(z, 600.0, 0.1, 0.1), z ** 2 * 0.1 / 60.0, rtol=1e-12)
    # the inverse: the σ_d that gives 5 mm at 0.5 m for f = 710.8 px, B = 18 mm
    assert DV.sigma_d_equivalent(0.005, 0.5, 710.8, 0.018) == pytest.approx(0.005 * 710.8 * 0.018 / 0.25)


def test_weighted_quantile():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    assert DV.weighted_quantile(v, np.ones(4), (0.5,))[0] == pytest.approx(2.5)
    # one value carrying almost all the weight is the median
    assert DV.weighted_quantile(v, np.array([1, 1, 100, 1.0]), (0.5,))[0] == pytest.approx(3.0, abs=0.02)
    assert DV.weighted_quantile([], [], (0.5,)) == [None]


# ====================================================================== the set-up of a sensor
def test_d405_defaults_are_labelled():
    p, s = project_with('intel-realsense-d405')
    ds = DV.setup(p, s)
    assert ds.resolution == (1280, 720) and ds.resolution_default and 'default' in ds.resolution_note
    assert ds.technology == 'passive_stereo' and [i.name for i in ds.needed] == ['IR left', 'IR right']
    # Intel Table 4-12, D401/D405 column: 1280 x 720 -> 100 mm
    assert ds.min_z_m == pytest.approx(0.100)
    # pinhole fitted to the stated 16:9 depth FOV, 84° wide: fx = 640 / tan(42°)
    assert ds.f_px == pytest.approx(640.0 / math.tan(math.radians(42.0)), rel=1e-9)
    assert ds.baseline_m == pytest.approx(0.018)
    assert ds.sigma_d_px == DV.DEFAULT_SIGMA_D_PX == 0.08 and ds.sigma_d_default and 'default' in ds.sigma_d_note
    assert not ds.need_projector and 'passive' in ds.projector_note


@pytest.mark.parametrize('res, mm', [((848, 480), 70), ((640, 360), 55), ((480, 270), 45), ((424, 240), 40)])
def test_d405_min_z_per_resolution(res, mm):
    p, s = project_with('intel-realsense-d405', res)
    ds = DV.setup(p, s)
    assert ds.resolution == res and not ds.resolution_default
    assert ds.min_z_m == pytest.approx(mm / 1000.0)          # Intel Table 4-12, typed again here
    assert ds.f_px < 640.0 / math.tan(math.radians(42.0))   # fewer pixels across the same FOV


def test_d405_vga_min_z_not_stated():
    p, s = project_with('intel-realsense-d405', (640, 480))
    ds = DV.setup(p, s)
    assert ds.min_z_m is None and 'not stated' in ds.min_z_note and 'not tested' in ds.min_z_note


def test_unknown_resolution_falls_back_with_a_note():
    p, s = project_with('intel-realsense-d435', (1000, 1000))
    ds = DV.setup(p, s)
    assert ds.resolution == (1280, 720) and any('not a depth mode' in n for n in ds.notes)


def test_projector_need_auto_and_setting():
    p, s = project_with('intel-realsense-d435')
    ds = DV.setup(p, s)
    assert ds.need_projector and 'default' in ds.projector_note and 'assumption' in ds.projector_note
    p, s = project_with('intel-realsense-d435', projector=False)
    assert not DV.setup(p, s).need_projector


def test_tof_and_min_range_devices():
    p, s = project_with('orbbec-femto-bolt', (640, 576))
    ds = DV.setup(p, s)
    assert ds.technology == 'tof' and len(ds.needed) == 1 and not ds.has_error_model
    assert ds.max_range_m == pytest.approx(3.86) and 'mode' in ds.max_range_note
    p, s = project_with('stereolabs-zed-mini')
    ds = DV.setup(p, s)
    assert ds.min_z_m == pytest.approx(0.1) and 'minimum range' in ds.min_z_note


def test_sensor_without_device_is_explained():
    p = Project('x')
    s = Sensor(name='Mystery', kind='depth')
    p.variant.sensors = [s]
    with pytest.raises(DV.DepthSetupError, match='no device is chosen'):
        DV.setup(p, s)
    s.model_id = 'e-con-see3cam-24cug'
    with pytest.raises(DV.DepthSetupError, match='not a depth device'):
        DV.setup(p, s)


# ====================================================================== the conditions at hand-placed points
def test_conditions_at_points():
    p, s = project_with('intel-realsense-d435')
    ds = DV.setup(p, s)
    H = ds.ref.camera.fov_deg()['h']
    edge = math.tan(math.radians(H / 2))
    pts = [optical_point(ds, 0.0, 0.0, 1.0),               # straight ahead at 1 m: valid
           optical_point(ds, 0.0, 0.0, 0.2),               # 200 mm < 280 mm min-Z
           optical_point(ds, 0.0, 0.0, 3.5),               # beyond the stated 3 m
           optical_point(ds, 2.0, 0.0, 1.0),               # far to the side
           # at 0.3 m just inside the left imager's left edge: the right imager (50 mm to the right) cannot see it
           optical_point(ds, -0.3 * edge * 0.995, 0.0, 0.3)]
    ev = DV.evaluate(ds, Pose(), np.array(pts))
    names = ['valid'] + list(DV.REASONS)
    assert [names[c] for c in ev['code']] == ['valid', 'min_z', 'max_range', 'fov', 'fov']
    assert ev['z'][0] == pytest.approx(1.0)


def test_stereo_overlap_matches_intel_formula():
    # D400 datasheet §4.4: Depth FOV = HFOV/2 + atan(tan(HFOV/2) − B/Z)
    p, s = project_with('intel-realsense-d435')
    ds = DV.setup(p, s)
    H = ds.ref.camera.fov_deg()['h']
    for z in (0.3, 1.0):
        a = np.radians(np.arange(-60, 60, 0.001))
        P = np.array([optical_point(ds, z * math.tan(t), 0.0, z) for t in a[::50]])
        inside = np.degrees(a[::50][DV.evaluate(ds, Pose(), P)['in_fov']])
        got = inside.max() - inside.min()
        want = H / 2 + math.degrees(math.atan(math.tan(math.radians(H / 2)) - ds.baseline_m / z))
        assert got == pytest.approx(want, abs=0.06)


def test_occlusion_from_either_imager():
    """A thin post in front of the right imager only: the point behind it is hidden from one imager, so it fails."""
    p, s = project_with('intel-realsense-d435')
    ds = DV.setup(p, s)
    from dexvantage.core.scenegeom import Occluders, object_triangles
    target = optical_point(ds, 0.02, 0.0, 1.0)
    right = DV.imager_world(Pose(), ds.needed[1]).position
    mid = right + 0.1 * (target - right)                    # 0.1 of the way along the right imager's line of sight
    post = TaskObject(name='post', shape='box', size=(0.01, 0.01, 0.01), pose=Pose(mid))
    occ = Occluders.concat([('object:post', object_triangles(post))])
    ev = DV.evaluate(ds, Pose(), target[None], occluders=occ, backend='numpy', n=256)
    assert ev['visible'][0].tolist() == [True, False] and ev['code'][0] == 1 + DV.REASONS.index('occluded')
    # the reference: exact ray casting from both centres
    from dexvantage.core import raycast as RC
    for k, im in enumerate(ds.needed):
        o = DV.imager_world(Pose(), im).position
        assert RC.visible_exact(target[None], o, occ.tris)[0] == ev['visible'][0, k]


def test_materials_warn_by_technology():
    p, s = project_with('intel-realsense-d405')               # passive: IR-absorbing does not warn
    target = optical_point(DV.setup(p, s), 0.0, 0.0, 0.4)
    for tags, passive, active in ((['transparent'], True, True), (['ir_absorbing'], False, True),
                                  (['glossy'], True, True), (['matte'], False, False)):
        o = TaskObject(name='cup', shape='box', size=(0.05, 0.05, 0.05), pose=Pose(target), materials=tags)
        for mid, want in (('intel-realsense-d405', passive), ('intel-realsense-d435', active)):
            p, s = project_with(mid)
            p.workspace.objects = [o]
            ds = DV.setup(p, s)
            mats = DV.material_objects(p, ds.technology)
            assert bool(mats) == want, (tags, mid)
            if want:
                ev = DV.evaluate(ds, Pose(), target[None] + [0.0, 0.0, 0.0], size=np.array([0.005]), mats=mats,
                                 project=p)
                assert ev['material'][0] == 1


def test_projector_cone_with_a_custom_device():
    d = copy.deepcopy(L.get('intel-realsense-d435').data)
    d['id'] = 'test-narrow-projector'
    d['name'] = 'Narrow projector test device'
    d['projector']['hfov_deg'], d['projector']['vfov_deg'] = 30.0, 20.0
    L.add_custom(d)
    try:
        p, s = project_with('test-narrow-projector')
        ds = DV.setup(p, s)
        assert ds.need_projector
        pts = np.array([optical_point(ds, 0.0, 0.0, 1.0), optical_point(ds, 0.5, 0.0, 1.0)])
        ev = DV.evaluate(ds, Pose(), pts)
        assert ev['in_fov'].all()
        assert ev['code'][0] == 0 and ev['code'][1] == 1 + DV.REASONS.index('projector')
    finally:
        L.remove_custom('test-narrow-projector')


def test_blind_zone_geometry():
    p, s = project_with('intel-realsense-d405')
    ds = DV.setup(p, s)
    tris, segs = DV.blind_zone_local(ds)
    Q = ds.ref.pose_in_body.inverse().apply(tris.reshape(-1, 3))
    assert Q[:, 2].max() == pytest.approx(ds.min_z_m) and Q[:, 2].min() == pytest.approx(0.0, abs=1e-12)
    far = Q[np.isclose(Q[:, 2], ds.min_z_m)]
    # the zone spans the image edge to edge: x/z = tan(42°) at the widest (84° depth FOV)
    assert np.abs(far[:, 0] / far[:, 2]).max() == pytest.approx(math.tan(math.radians(42.0)), rel=1e-6)
    assert segs.shape[1:] == (2, 3)
    p, s = project_with('intel-realsense-d405', (640, 480))
    assert DV.blind_zone_local(DV.setup(p, s)) is None      # min-Z not stated: no zone drawn


# ====================================================================== the study
@pytest.fixture(scope='module')
def ur5e_doc():
    from dexvantage.core.robot.samples import sample_project
    p = sample_project('ur5e')
    return p, jobs.run_study('depth_validity', p)


def test_study_registered():
    jobs.load_studies()
    info = jobs.STUDIES['depth_validity']
    assert info.batch == 'depth-validity' and not info.needs_robot and not info.quick


def test_study_on_the_ur5e_sample(ur5e_doc):
    p, doc = ur5e_doc
    json.dumps(doc)
    assert doc['schema'] == 'dexvantage.depth_validity/1' and doc['pose']['kind'] == 'home'
    names = [s['name'] for s in doc['sensors']]
    assert names == ['Wrist D405', 'Overhead D435'] and not doc['skipped']
    for s in doc['sensors']:
        assert sum(s['breakdown'].values()) == pytest.approx(100.0, abs=1e-6)
        assert s['breakdown']['valid'] == pytest.approx(s['valid_pct'])
        assert 0 < s['valid_pct'] < 100
        sz = s['sigma_z']
        assert sz['p5_m'] <= sz['median_m'] <= sz['p95_m']
        # every depth-valid voxel lies between min-Z and the maximum range, so σ_z lies between their values
        lo = DV.sigma_z(s['min_z_m'], s['f_px'], s['baseline_m'], s['sigma_d_px'])
        hi = DV.sigma_z(s['max_range_m'], s['f_px'], s['baseline_m'], s['sigma_d_px'])
        assert lo <= sz['p5_m'] and sz['p95_m'] <= hi
        assert s['occlusion']['one_imager_pct'] + s['occlusion']['all_imagers_pct'] == \
            pytest.approx(s['fails']['occluded'], abs=1e-6)
        assert len(s['overlay']['points_mm']) == len(s['overlay']['reason'])
    rig = doc['rig']['valid_pct']
    assert rig >= max(s['valid_pct'] for s in doc['sensors']) - 1e-9
    wrist = doc['sensors'][0]
    # the maker's figure for the D405: RMS ≤ 1 % at ≤ 0.5 m (Table 4-15) -> 5 mm, shown beside the model
    mk = wrist['maker_rms']
    assert mk['sigma_m'] == pytest.approx(0.005)
    assert mk['model_sigma_m'] == pytest.approx(0.25 * 0.08 / (wrist['f_px'] * 0.018))
    assert mk['sigma_d_equiv_px'] == pytest.approx(0.005 * wrist['f_px'] * 0.018 / 0.25)
    # the wrist camera sees the fingertips; their depth is reported against min-Z
    tips = wrist['fingertips']
    assert {t['link'] for t in tips} == {'gripper_right_pad', 'gripper_left_pad'}
    assert all(t['in_fov'] and t['z_m'] > 0 for t in tips)


def test_study_follows_the_pose():
    from dexvantage.core.robot.model import get_model
    from dexvantage.core.robot.samples import sample_project
    p = sample_project('ur5e')
    r = p.robots[0]
    m = get_model(p, r.uid)
    q = m.home.copy()
    q[0] += 0.4
    p.analysis.studies['depth_validity'] = {'q': {r.uid: q.tolist()}}
    doc = jobs.run_study('depth_validity', p)
    assert doc['pose']['kind'] == 'current'
    home = jobs.run_study('depth_validity', sample_project('ur5e'))
    assert doc['sensors'][0]['world_pose'] != home['sensors'][0]['world_pose']
    assert doc['sensors'][1]['world_pose'] == home['sensors'][1]['world_pose']       # the overhead camera is static
    assert doc['sensors'][0]['valid_pct'] != pytest.approx(home['sensors'][0]['valid_pct'], abs=1e-9)


def test_suggestion_when_min_z_bites():
    """A box 50 to 90 mm in front of a D405: min-Z at 1280 x 720 (100 mm) hides all of it, 640 x 360 (55 mm) part
    of it; 480 x 270 (45 mm) and 424 x 240 (40 mm) clear it all, and of those the study suggests the larger
    resolution (the smaller σ_z penalty)."""
    from dexvantage.core.project import Volume
    p, s = project_with('intel-realsense-d405')
    p.workspace.volumes = [Volume(name='near', center=(0.07, 0.0, 0.0), size=(0.04, 0.04, 0.04))]
    doc = jobs.run_study('depth_validity', p)
    s0 = doc['sensors'][0]
    assert s0['breakdown']['min_z'] > 50
    sug = doc['suggestion']
    assert sug is not None and sug['resolution'] == [480, 270]
    assert sug['gain_pct'] == pytest.approx(s0['breakdown']['min_z'])
    assert '480 × 270' in sug['text'] and 'σ_z rises 2.67×' in sug['text']


def test_fingertips_inside_min_z_and_the_suggestion():
    """The wrist D405 of the UR5e sample moved 30 mm along its optical axis towards the gripper: the fingertips'
    depth drops by 30 mm (the sensor moved along its own axis), below the 100 mm min-Z, and the study suggests the
    largest 16:9 mode whose min-Z clears them (848 x 480: 70 mm, Intel Table 4-12)."""
    from dexvantage.core.frames import quat_to_matrix
    from dexvantage.core.robot.samples import sample_project
    base = jobs.run_study('depth_validity', sample_project('ur5e'))['sensors'][0]['fingertips']
    p = sample_project('ur5e')
    s = p.variant.sensors[0]
    R = quat_to_matrix(s.mount.pose.quaternion)
    s.mount.pose = Pose(s.mount.pose.position + R @ np.array([0.03, 0.0, 0.0]), s.mount.pose.quaternion)
    doc = jobs.run_study('depth_validity', p)
    tips = doc['sensors'][0]['fingertips']
    for a, b in zip(base, tips):
        assert b['z_m'] == pytest.approx(a['z_m'] - 0.03, abs=1e-9)
    assert all(t['inside_min_z'] for t in tips) and min(t['z_m'] for t in tips) > 0.07
    sug = doc['suggestion']
    assert sug['resolution'] == [848, 480] and sug['fingertips'] and 'clears the fingertips' in sug['text']


def test_study_messages_without_sensors_or_volumes():
    p = Project('empty')
    doc = jobs.run_study('depth_validity', p)
    assert 'task volume' in doc['message'] and doc['sensors'] == []
    from dexvantage.core.project import blank_project
    p = blank_project()
    doc = jobs.run_study('depth_validity', p)
    assert 'no depth sensor' in doc['message'].lower()
    p.variant.sensors = [Sensor(name='Unknown', kind='depth')]
    doc = jobs.run_study('depth_validity', p)
    assert doc['skipped'] and 'no device is chosen' in doc['skipped'][0]['why']


def test_material_object_in_the_study():
    from dexvantage.core.robot.samples import sample_project
    p = sample_project('ur5e')
    c = p.workspace.volumes[0].center
    p.workspace.objects = [TaskObject(name='Glass', shape='cylinder', size=(0.04, 0.12, 0.0),
                                      pose=Pose((c[0] + 0.12, c[1] - 0.1, 0.06)), materials=['transparent'])]
    doc = jobs.run_study('depth_validity', p)
    for s in doc['sensors']:
        m = s['materials']
        assert len(m) == 1 and m[0]['object'] == 'Glass' and 'transparent' in m[0]['reason']
        assert s['breakdown']['material'] > 0 and m[0]['pct'] == pytest.approx(s['breakdown']['material'])
        assert s['valid_geometric_pct'] == pytest.approx(s['valid_pct'] + s['breakdown']['material'])


# ====================================================================== the page and the min-Z zones
def test_depth_page(win, app):
    from dexvantage.core.robot.samples import sample_project
    from dexvantage.ui.viewport3d import LineDraw, TriDraw
    win.store.set_project(sample_project('ur5e'))
    win.ctx.analysis.wait(10)
    page = win.pages['depth']
    a = win.actions_by_id['depth_validity']
    assert a.isEnabled()
    a.trigger()
    pump(app, 0.05)
    assert win.current_page() is page
    assert page.c_sensor.count() == 2 and page.c_res.count() == 6
    assert '(default)' in page.c_res.currentText()
    page.run()
    win.ctx.analysis.wait(60)
    pump(app, 0.05)
    assert page.doc is not None and page.doc['pose']['kind'] == 'current'
    assert page.head.figure.text() != '—' and page.head.unit.text() == '%'
    assert page.noise.topLevelItemCount() == 2 and page.sens.topLevelItemCount() == 2
    assert page.chart.rows[0][0] == 'valid' and len(page.chart.rows) == 7
    items = page.overlays(win.viewport)
    assert items and all(isinstance(i, LineDraw) for i in items)
    # the min-Z zones: one translucent volume per depth sensor, moving with the wrist
    z0 = [d for d in page.zones(win.viewport) if isinstance(d, TriDraw)]
    assert len(z0) == 2
    uid = win.store.project.robots[0].uid
    q = win.store.q[uid].copy()
    q[0] += 0.5
    win.store.set_q(uid, q)
    z1 = [d for d in page.zones(win.viewport) if isinstance(d, TriDraw)]
    assert not np.allclose(z0[0].tri, z1[0].tri) and np.allclose(z0[1].tri, z1[1].tri)
    # the zone's apex is the wrist D405's left imager centre at the new pose
    from dexvantage.core import scenegeom as SG
    s = win.store.project.variant.sensors[0]
    ds = DV.setup(win.store.project, s)
    o = DV.imager_world(SG.sensor_world_pose(win.store.project, s, win.store.q), ds.ref).position
    assert np.linalg.norm(z1[0].tri[:, 0] - o, axis=1).min() < 1e-9
    win.ctx.set_overlay('depthzones', False)
    assert page.zones(win.viewport) == []
    win.ctx.set_overlay('depthzones', True)
    assert 'ov_depthzones' in win.actions_by_id
    # the resolution is a setting: one undo step
    page.set_resolution(s.uid, (848, 480))
    assert DV.setup(win.store.project, s).min_z_m == pytest.approx(0.07)
    win.store.undo.undo()
    assert DV.setup(win.store.project, s).resolution == (1280, 720)
    win.ctx.analysis.wait(60)


def test_apply_the_suggestion(win, app):
    from dexvantage.core.project import Volume
    p, s = project_with('intel-realsense-d405')
    p.workspace.volumes = [Volume(name='near', center=(0.07, 0.0, 0.0), size=(0.04, 0.04, 0.04))]
    win.store.set_project(p)
    page = win.pages['depth']
    page.run()
    win.ctx.analysis.wait(60)
    pump(app, 0.05)
    assert page.b_apply.isEnabled()
    page.apply()
    win.ctx.analysis.wait(60)
    pump(app, 0.05)
    assert DV.setup(win.store.project, win.store.project.variant.sensors[0]).resolution == (480, 270)
    assert page.doc['sensors'][0]['resolution'] == [480, 270] and page.doc['sensors'][0]['breakdown']['min_z'] == 0
    win.store.undo.undo()
    assert DV.setup(win.store.project, win.store.project.variant.sensors[0]).resolution == (1280, 720)
    win.ctx.analysis.wait(60)


# ====================================================================== validation
def test_depth_checks_pass_quick():
    from dexvantage import validation as VA
    res = VA.run(settings={'models': ['depth'], 'quick': True})
    ids = {c.id for c in res.checks}
    assert ids == {'depth.sigma_z_monte_carlo', 'depth.d400_tables', 'depth.fov_overlap_formula',
                   'depth.finger_shadow_raycast'}
    for c in res.checks:
        assert c.passed, (c.id, c.error, c.tolerance, c.exception)
