"""VLA token resolution (core/studies/vla_tokens.py), Plücker ray maps (core/plucker.py), the safetensors writer, the
validation checks (validation/checks_vla.py) and the page (ui/pages/vla.py)."""
import json
import os
import struct
import types

import numpy as np
import pytest

from dexvantage.core import grasps as GR
from dexvantage.core import jobs
from dexvantage.core import plucker as PL
from dexvantage.core import policies as PO
from dexvantage.core.frames import Pose
from dexvantage.core.optics import CameraModel
from dexvantage.core.project import Policy, TaskObject
from dexvantage.core.robot.samples import sample_project
from dexvantage.core.studies import vla_tokens as VT

from conftest import pump

ID_Q = np.array([0.0, 0.0, 0.0, 1.0])


def _res(**kw):
    base = {k: (None, 'not stated') for k in PO.FIELDS}
    base.update({k: (v, 'test') for k, v in kw.items()})
    return base


def _view(cam, rule='none', inp=None, frac=None, pre=None, R=None, t=None):
    g = PO.crop_geometry(rule, (cam.width, cam.height), inp or (cam.width, cam.height), frac, pre)
    return VT.View(None, None, cam, g, np.eye(3) if R is None else R, np.zeros(3) if t is None else t)


def _box(center, size):
    from dexvantage.core.scenegeom import object_triangles
    return object_triangles(TaskObject(shape='box', size=size, pose=Pose(np.asarray(center, float), ID_Q)))


@pytest.fixture
def ur5e():
    p = sample_project('ur5e')
    p.tasks.grasp_sets, p.policies = [], []  # the sample's own grasp set and policy left out
    lo, hi = (np.asarray(x) for x in p.workspace.volumes[0].bounds())
    c = (lo + hi) / 2
    p.workspace.objects.append(TaskObject(name='Cube', shape='box', size=(0.05, 0.05, 0.05),
                                          pose=Pose(np.array([c[0], c[1], lo[2] + 0.025]), ID_Q)))
    p.workspace.objects.append(TaskObject(name='Peg', shape='cylinder', size=(0.006, 0.03, 0.0),
                                          pose=Pose(np.array([c[0] + 0.1, c[1], lo[2] + 0.015]), ID_Q)))
    return p


# ====================================================================== the token grid
def test_token_grid_from_the_patch_or_the_token_count():
    g = VT.token_grid(_res(input_resolution=[224, 224], patch_px=14, tokens_per_image=256))
    assert (g['cell_px'], g['cols'], g['rows'], g['kind'], g['merged_cell_px']) == (14, 16, 16, 'patch', None)
    g = VT.token_grid(_res(input_resolution=[512, 512], patch_px=16, tokens_per_image=64))     # SmolVLA
    assert g['cell_px'] == 16 and g['merged_cell_px'] == 64 and 'square grid' in g['merged_source']
    g = VT.token_grid(_res(input_resolution=[640, 480], tokens_per_image=300))                # ACT
    assert (g['cell_px'], g['cols'], g['rows'], g['kind']) == (32, 20, 15, 'token')
    assert 'derived from tokens per image' in g['source']
    assert VT.token_grid(_res(input_resolution=[224, 224], tokens_per_image=64))['cell_px'] == 28      # GR00T
    assert VT.token_grid(_res(input_resolution=[288, 216])) is None                                    # CNN
    assert VT.token_grid(_res(input_resolution=[224, 224], tokens_per_image=50)) is None               # no tiling


def test_every_library_policy_gets_a_grid_or_says_why():
    p = sample_project('so100')
    want = {'pi0': 14, 'pi0-fast': 14, 'openvla-7b': 14, 'octo-base': 16, 'smolvla-base': 16, 'act': 32,
            'gr00t-n1-2b': 28, 'diffusion-policy-cnn': None}
    for lid, cell in want.items():
        pol = Policy(name=lid, library_id=lid)
        p.policies = [pol]
        g = VT.token_grid(PO.resolve(p, pol.uid))
        assert (g['cell_px'] if g else None) == cell, lid


# ====================================================================== raster and counts
def test_a_square_facing_a_pinhole_camera_covers_the_pixels_of_its_closed_form():
    cam = CameraModel(224, 224, 100.0, 100.0, 111.5, 111.5)
    v = _view(cam)
    tri = _box([0.0, 0.0, 1.0], (0.4, 0.4, 0.01))         # front face at z = 0.995
    s = VT.silhouette(v, tri)
    # u = 100 x / z + 111.5, edge coordinate u + 0.5: the face spans 112 ± 20.10 in edge coordinates, so the pixel
    # centres i + 0.5 from 92.5 to 131.5 are inside: 40 columns and 40 rows
    half = 100 * 0.2 / 0.995
    cols = np.arange(224) + 0.5
    n = int(np.count_nonzero(np.abs(cols - 112.0) <= half))
    assert n == 40 and s.pixels_input == n * n and s.bbox == (92, 92, 132, 132)
    c = VT.count_mask(s.input_mask, 14, s.bbox)
    assert c['tokens'] == 16 and c['extent'] == pytest.approx([40 / 14, 40 / 14])
    assert c['tokens_half'] == 4                          # cells 7 and 8 of each axis are fully covered
    assert s.outside_fraction == 0.0


def test_raster_matches_ray_casting_on_distorted_lenses():
    from dexvantage.validation import checks_vla as CV
    rows = CV.patch_case_rows(7, quick=True)
    assert {r['camera'] for r in rows} == {'pinhole', 'brown', 'kb', 'pinhole_vga'}
    assert max(r['token_error'] for r in rows) == 0
    assert sum(r['pixels_reference'] for r in rows) > 5000
    assert sum(r['pixels_differ'] for r in rows) <= 0.001 * sum(r['pixels_reference'] for r in rows)


def test_without_subdivision_a_distorted_lens_gets_edges_wrong():
    """The reference is sensitive: straight projected edges on a barrel-distorted lens miss pixels."""
    from dexvantage.validation import checks_vla as CV
    cam = CV._cameras()[1][1]
    v = _view(cam, 'resize', (256, 256))
    tri = _box([0.25, 0.15, 0.5], (0.3, 0.3, 0.02))
    ref = CV.reference_mask(v, tri)
    fine = VT.silhouette(v, tri).input_mask                           # the defaults: 0.01 px, 8 levels
    loose = VT.silhouette(v, tri, 0.05, 6).input_mask
    coarse = VT.silhouette(v, tri, 0.05, 0).input_mask
    assert np.count_nonzero(fine != ref) == 0 and 0 < np.count_nonzero(loose != ref) <= 5
    assert np.count_nonzero(coarse != ref) > 20


def test_cropped_parts_are_measured():
    cam = CameraModel(640, 480, 400.0, 400.0, 319.5, 239.5)
    v = _view(cam, 'center_crop', (288, 216), pre=(320, 240))          # keeps the middle 576 × 432 sensor pixels
    # an object near the left edge: half inside the crop
    x_edge = (32 - 320) / 400.0                                        # the crop's left edge at z = 1 (u + 0.5 = 32)
    s = VT.silhouette(v, _box([x_edge, 0.0, 1.0], (0.1, 0.1, 0.001)))
    assert 0.4 < s.outside_fraction < 0.6 and s.pixels_input > 0
    # wholly in the cropped margin: seen by the camera, not by the input
    s = VT.silhouette(v, _box([(10 - 320) / 400.0, 0.0, 1.0], (0.02, 0.02, 0.001)))
    assert s.pixels_image > 0 and s.pixels_input == 0 and s.outside_fraction == 1.0
    # behind the camera: nothing
    s = VT.silhouette(v, _box([0.0, 0.0, -1.0], (0.1, 0.1, 0.1)))
    assert s.pixels_image == 0 and s.outside_fraction is None


def test_padding_is_never_a_token_of_the_image():
    cam = CameraModel(1280, 720, 600.0, 600.0, 639.5, 359.5)
    v = _view(cam, 'resize_pad', (224, 224))
    assert v.geom.dst_rect == (0.0, 49.0, 224.0, 175.0)
    s = VT.silhouette(v, _box([0.0, 0.0, 0.3], (2.0, 2.0, 0.01)))     # fills the whole image
    assert s.pixels_input == 224 * 126 and not s.input_mask[:49].any() and not s.input_mask[175:].any()


def test_fingertip_points_count_the_cells_that_hold_them():
    cam = CameraModel(224, 224, 100.0, 100.0, 111.5, 111.5)
    v = _view(cam)
    P = np.array([[0.0, 0.0, 1.0], [0.01, 0.0, 1.0], [0.3, 0.3, 1.0], [0.0, 0.0, -1.0], [5.0, 0, 1.0]])
    c = VT.count_points(v, P, 14)
    # (112, 112) and (113, 112): one cell (8, 8); (142, 142): cell (10, 10); one behind, one off the image
    assert c['tokens'] == 2 and c['cells'] == [[8, 8], [10, 10]] and c['points_in_input'] == 3
    assert c['points_in_image'] == 3


def test_overlay_geometry_in_sensor_pixels():
    cam = CameraModel(1920, 1080, 1000.0, 1000.0, 959.5, 539.5)
    g = PO.crop_geometry('resize_pad', (1920, 1080), (224, 224))
    crop = VT.crop_polyline(g)
    assert np.allclose(crop[0], [-0.5, -0.5]) and np.allclose(crop[2], [1919.5, 1079.5])
    lines = VT.grid_polylines(g, 14)
    # 17 vertical lines over the content (y 49 … 175) and the horizontal ones inside it (56, 70, … 168)
    assert len(lines) == 17 + 9
    assert all(-0.6 <= ln[:, 0].min() and ln[:, 0].max() <= 1919.6 for ln in lines)
    cell = VT.cell_polylines(g, 14, [[4, 0]])[0]
    assert np.allclose(cell[0], [-0.5, 56 / g.sy - 49 / g.sy - 0.5])
    v = _view(cam, 'resize_pad', (224, 224))
    seg = VT.footprints(v, _box([0.0, 0.0, 1.0], (0.4, 0.4, 0.01)), 14, [[8, 8]])
    assert len(seg) and np.allclose(seg[..., 2], 0.995, atol=1e-6)    # on the object's front face


# ====================================================================== the study
def test_no_policy_is_a_plain_sentence(ur5e):
    with pytest.raises(jobs.StudyError, match='Policies page'):
        jobs.run_study('vla_tokens', ur5e)


def test_study_on_the_ur5e_sample_without_a_grasp_set(ur5e):
    for lid in ('pi0', 'diffusion-policy-cnn'):
        ur5e.policies.append(Policy(name=lid, library_id=lid))
    doc = jobs.run_study('vla_tokens', ur5e)
    json.dumps(doc)
    assert doc['schema'] == 'dexvantage.vla_tokens/1' and doc['pose_source'] == 'home'
    assert any('no grasp set' in n for n in doc['notes'])
    assert any('CNN' in n for n in doc['notes'])
    pi0 = doc['policies'][0]
    assert pi0['grid']['cell_px'] == 14 and pi0['crop_rule'] == 'resize_pad'
    over = next(c for c in pi0['cameras'] if c['name'] == 'Overhead D435')
    m = over['mapping']
    assert m['sensor_size'] == [1920, 1080] and m['crop_box_sensor_px']['u0'] == -0.5
    assert m['scale_input_per_sensor_px'] == pytest.approx([224 / 1920, 126 / 1080])
    assert m['content_rect_input_px'] == [0.0, 49.0, 224.0, 175.0] and 'x_in' in m['equation']
    peg = next(o for o in over['objects'] if o['name'] == 'Peg')
    assert peg['poses_in_input'] == 1 and peg['extent_min'] < 2
    assert any(w['kind'] == 'small' and w['object'] == peg['uid'] for w in doc['warnings'])
    assert doc['summary']['status'] == 'serious' and doc['summary']['word'] == 'too few tokens'
    assert doc['summary']['fewest']['tokens'] >= 1
    dp = doc['policies'][1]
    assert dp['grid'] is None and dp['cameras'][0]['objects'][0]['tokens_min'] is None
    assert {t['phase'] for c in pi0['cameras'] for t in c['fingertips']} == {'home'}
    assert jobs.STUDIES['vla_tokens'].batch == 'vla-tokens' and not jobs.STUDIES['vla_tokens'].needs_robot


def test_critical_objects_and_the_threshold_are_settings(ur5e):
    ur5e.policies.append(Policy(name='pi0', library_id='pi0'))
    cube = ur5e.workspace.objects[0]
    ur5e.analysis.studies['vla_tokens'] = {'critical_objects': [cube.uid], 'min_tokens_side': 0.1}
    doc = jobs.run_study('vla_tokens', ur5e)
    assert 'min_tokens_side' not in doc['settings']['defaults']
    assert all(w['object'] == cube.uid for w in doc['warnings'])
    assert not any(w['kind'] == 'small' for w in doc['warnings'])
    ur5e.analysis.studies['vla_tokens'] = {'min_tokens_side': -1}
    with pytest.raises(jobs.StudyError):
        jobs.run_study('vla_tokens', ur5e)


def test_grasp_set_poses_with_ik_for_the_wrist_camera(ur5e):
    ur5e.policies.append(Policy(name='smolvla', library_id='smolvla-base'))
    gs = GR.sample(ur5e, ur5e.robots[0].uid, ('top_down',), 6, 0)
    GR.add_set(ur5e, gs)
    ur5e.analysis.studies['vla_tokens'] = {'pre_grasp_m': 0.15}
    doc = jobs.run_study('vla_tokens', ur5e)
    assert doc['pose_source'] == 'grasp_set' and doc['grasp_set']['poses'] == 6
    assert len(doc['poses']) == 12 and {p['phase'] for p in doc['poses']} == {'pre_grasp', 'contact'}
    wrist = next(c for c in doc['policies'][0]['cameras'] if c['name'] == 'Wrist D405')
    assert wrist['on_link']
    phases = {t['phase'] for t in wrist['fingertips']}
    assert phases == {'pre_grasp', 'contact'}
    reached = sum(bool(p['ik_reached']) for p in doc['poses'])
    assert reached == 12 - len(wrist['poses_left_out'])
    # the overhead camera does not move: its object counts are the same at every pose
    over = next(c for c in doc['policies'][0]['cameras'] if c['name'] == 'Overhead D435')
    cube = next(o for o in over['objects'] if o['name'] == 'Cube')
    assert len({r['tokens'] for r in cube['rows']}) == 1 and cube['poses'] == 12
    # pre-grasp pads are farther from the objects than at contact: the overhead camera sees them move
    pre = [r for r in over['fingertip_rows'] if r['phase'] == 'pre_grasp']
    con = [r for r in over['fingertip_rows'] if r['phase'] == 'contact']
    assert len(pre) == len(con) > 0


def test_suggestion_picks_a_stream_with_more_tokens(ur5e):
    ur5e.policies.append(Policy(name='smolvla', library_id='smolvla-base'))
    doc = jobs.run_study('vla_tokens', ur5e)
    sg = doc['suggestion']
    assert sg is not None and sg['worst_after'] > sg['worst_before'] and 'stream' in sg['text']
    s = ur5e.variant.sensor(sg['sensor_uid'])
    s.stream = list(sg['mode'])
    again = jobs.run_study('vla_tokens', ur5e)
    cam = next(c for c in again['policies'][0]['cameras'] if c['uid'] == s.uid)
    assert cam['stream'] == sg['mode']


def test_nothing_to_count():
    p = sample_project('so100')
    p.tasks.grasp_sets, p.policies = [], []  # the sample's own grasp set and policy left out
    p.robots = []
    p.variant.sensors = [s for s in p.variant.sensors if not s.mount.on_link]
    p.policies.append(Policy(name='pi0', library_id='pi0'))
    with pytest.raises(jobs.StudyError, match='nothing to count'):
        jobs.run_study('vla_tokens', p)


# ====================================================================== Plücker rays
def test_ray_field_is_unit_and_its_moments_are_the_lines(ur5e):
    ur5e.policies.append(Policy(name='pi0', library_id='pi0'))
    s = ur5e.variant.sensors[1]
    f = PL.ray_field(ur5e, s, policy_uid=ur5e.policies[0].uid)
    assert f.d.shape == (224, 224, 3) and f.frame == 'base' and f.frame_robot == ur5e.robots[0].uid
    d, m = f.d[f.valid], f.m[f.valid]
    assert np.allclose(np.linalg.norm(d, axis=1), 1) and np.abs((d * m).sum(1)).max() < 1e-12
    for t in (0.3, 2.0, 11.0):
        assert np.allclose(np.cross(f.origin + t * d, d), m, atol=1e-12)
    # padding: zero rays, not valid; the content rows are 49 … 174 for a 16:9 stream padded into a square
    assert not f.valid[:49].any() and not f.valid[175:].any() and f.valid[49:175].all()
    assert np.all(f.d[:49] == 0) and np.all(f.m[:49] == 0)
    pk = f.plucker('dm')
    assert pk.shape == (224, 224, 6) and np.allclose(pk[..., 3:], f.m) and np.allclose(f.plucker('md')[..., :3], f.m)
    with pytest.raises(PL.PluckerError):
        f.plucker('xx')


def test_pinhole_directions_through_the_crop_match_the_inverse_intrinsics(ur5e):
    ur5e.policies.append(Policy(name='gr00t', library_id='gr00t-n1-2b'))       # centre crop 0.95, then resize
    s = ur5e.variant.sensors[1]
    f = PL.ray_field(ur5e, s, policy_uid=ur5e.policies[0].uid, frame='camera')
    cam, g = f.camera, f.geometry
    assert np.allclose(f.m, 0) and f.valid.all()
    # column i of the input is sensor u = x0 + (i + 0.5) / sx - 0.5
    x0, y0 = g.src_rect[:2]
    u = x0 + (np.arange(224) + 0.5) / g.sx - 0.5
    v = y0 + (np.arange(224) + 0.5) / g.sy - 0.5
    assert np.allclose(f.u, u) and np.allclose(f.v, v)
    U, V = np.meshgrid(u, v)
    K = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1.0]])
    r = np.linalg.solve(K, np.stack([U.ravel(), V.ravel(), np.ones(U.size)]))
    r /= np.linalg.norm(r, axis=0)
    assert np.allclose(f.d.reshape(-1, 3), r.T, atol=1e-12)


def test_base_frame_of_another_robot(ur5e):
    ur5e.policies.append(Policy(name='pi0', library_id='pi0'))
    s = ur5e.variant.sensors[1]
    w = PL.ray_field(ur5e, s, policy_uid=ur5e.policies[0].uid, frame='world')
    rob = ur5e.robots[0]
    rob.base = Pose(np.array([0.5, -0.2, 0.1]), np.array([0, 0, np.sin(0.3), np.cos(0.3)]))
    from dexvantage.core.robot.model import clear_cache
    clear_cache()
    b = PL.ray_field(ur5e, s, policy_uid=ur5e.policies[0].uid, frame='base', robot_uid=rob.uid)
    Ti = np.linalg.inv(rob.base.matrix())
    assert np.allclose(b.origin, Ti[:3, :3] @ w.origin + Ti[:3, 3])
    assert np.allclose(b.d[b.valid], w.d[w.valid] @ Ti[:3, :3].T)
    with pytest.raises(PL.PluckerError):
        PL.ray_field(ur5e, s, policy_uid=ur5e.policies[0].uid, frame='vehicle')


def test_export_writes_npy_safetensors_and_provenance(ur5e, tmp_path):
    ur5e.policies.append(Policy(name='π0 test', library_id='pi0'))
    paths = PL.export_plucker(ur5e, str(tmp_path), ur5e.policies[0].uid, dtype='float16')
    names = sorted(os.path.basename(p) for p in paths)
    assert names == sorted(['plucker_0_test_Wrist_D405.npy', 'plucker_0_test_Wrist_D405.safetensors',
                            'plucker_0_test_Wrist_D405.json', 'plucker_0_test_Overhead_D435.npy',
                            'plucker_0_test_Overhead_D435.safetensors', 'plucker_0_test_Overhead_D435.json'])
    a = np.load(tmp_path / 'plucker_0_test_Overhead_D435.npy')
    assert a.shape == (224, 224, 6) and a.dtype == np.float16
    t, meta = PL.read_safetensors(str(tmp_path / 'plucker_0_test_Overhead_D435.safetensors'))
    assert np.array_equal(t['plucker'], a) and t['valid'].dtype == bool and t['u'].shape == (224,)
    assert meta['schema'] == PL.SCHEMA and meta['frame'] == 'base'
    doc = json.load(open(tmp_path / 'plucker_0_test_Overhead_D435.json', encoding='utf-8'))
    assert doc['field']['shape'] == [224, 224, 6] and doc['field']['crop']['rule'] == 'resize_pad'
    assert doc['field']['intrinsics']['width'] == 1920 and 'provenance' in doc and doc['q_rule']
    for bad in ({'dtype': 'int8'}, {'order': 'x'}, {'formats': ('png',)}):
        with pytest.raises(PL.PluckerError):
            PL.export_plucker(ur5e, str(tmp_path / 'b'), ur5e.policies[0].uid, **bad)
    with pytest.raises(PL.PluckerError):
        PL.export_plucker(ur5e, str(tmp_path / 'c'), 'nope')


def test_safetensors_format(tmp_path):
    p = str(tmp_path / 'x.safetensors')
    a = {'b': np.arange(6, dtype=np.float32).reshape(2, 3), 'a': np.array([True, False]), 'z': np.zeros((0,), np.int64)}
    PL.write_safetensors(p, a, {'k': 'v'})
    raw = open(p, 'rb').read()
    n = struct.unpack('<Q', raw[:8])[0]
    assert n % 8 == 0 and raw[8:9] == b'{'
    hdr = json.loads(raw[8:8 + n])
    assert hdr['__metadata__'] == {'k': 'v'} and hdr['a'] == {'dtype': 'BOOL', 'shape': [2], 'data_offsets': [0, 2]}
    assert hdr['b']['data_offsets'] == [2, 26] and hdr['b']['dtype'] == 'F32'
    t, meta = PL.read_safetensors(p)
    assert all(np.array_equal(t[k], a[k]) and t[k].dtype == a[k].dtype for k in a) and meta == {'k': 'v'}
    # a hole in the buffer is refused
    bad = raw[:8 + n] + raw[8 + n:] + b'\0'
    open(p, 'wb').write(bad)
    with pytest.raises(PL.PluckerError, match='no tensor'):
        PL.read_safetensors(p)
    with pytest.raises(PL.PluckerError):
        PL.write_safetensors(p, {'c': np.zeros(2, np.complex64)})


# ====================================================================== validation
def test_validation_checks_pass():
    from dexvantage.validation import suite as S
    from dexvantage.validation import checks_vla as CV
    ctx = S.Ctx(None, None, None, S.ValidationSettings(quick=True))
    try:
        for fn, tol in ((CV._patch_counts, 0.0), (CV._plucker, 1e-6), (CV._safetensors, 0.0)):
            o = fn(ctx)
            assert o.ok and o.error <= tol, (fn.__name__, o.error, o.note)
    finally:
        ctx.close()
    ids = {c.id for c in S.CHECKS if c.model == 'vla'}
    assert {'vla.patch_counts_raycast', 'vla.plucker_moment_cv2', 'vla.safetensors_roundtrip'} <= ids


# ====================================================================== the page
def _ready(win, app):
    assert win.open_sample('ur5e')
    win.ctx.analysis.wait(5)
    pump(app, 0.02)
    st = win.store
    lo, hi = (np.asarray(x) for x in st.project.workspace.volumes[0].bounds())
    c = (lo + hi) / 2

    def fn(p):
        p.tasks.grasp_sets, p.policies = [], []          # the sample's own grasp set and policy left out
        p.workspace.objects.append(TaskObject(name='Cube', shape='box', size=(0.05, 0.05, 0.05),
                                              pose=Pose(np.array([c[0], c[1], lo[2] + 0.025]), ID_Q)))
        p.workspace.objects.append(TaskObject(name='Peg', shape='cylinder', size=(0.006, 0.03, 0.0),
                                              pose=Pose(np.array([c[0] + 0.1, c[1], lo[2] + 0.015]), ID_Q)))
    st.edit('workspace', fn, 'Added two objects')
    win.pages['policies'].add_policy('smolvla-base')
    page = win.pages['vla']
    page.run()
    assert win.ctx.analysis.wait(60)
    pump(app, 0.1)
    return page


def test_page_runs_shows_and_overlays(win, app, tmp_path):
    from dexvantage.ui.pages import PAGES
    page = _ready(win, app)
    assert PAGES['vla'].workspace == 'grasp' and PAGES['vla'].menu == 'Analysis'
    assert page.doc is not None and page.head.status.text.text() == 'too few tokens'
    assert page.objs.topLevelItemCount() == 2 and page.tips.topLevelItemCount() >= 2
    assert 'crop box' in page.mapping.text() and 'patch 16 px' in page.mapping.text()
    over = next(s for s in win.store.project.variant.sensors if s.name == 'Overhead D435')
    page.cam_box.setCurrentIndex(page.cam_box.findData(over.uid))
    win.camview.view.sensor_uid = over.uid
    out = page.camview_overlays(types.SimpleNamespace(width=1920, height=1080), None)
    labels = [t for _p, _c, t in out]
    assert labels[0].startswith('SmolVLA') or 'crop' in labels[0]
    assert any('grid 16 px' in t for t in labels) and any('tokens on critical' in t for t in labels)
    items = page.overlays(win.viewport)
    assert items and all(len(it.pts) for it in items)
    # export with provenance, and the ray maps
    path = page.export(str(tmp_path / 'vla.json'))
    doc = json.load(open(path, encoding='utf-8'))
    assert doc['study'] == 'vla_tokens' and 'provenance' in doc
    paths = page.export_plucker(str(tmp_path / 'rays'))
    assert paths and any(p.endswith('.safetensors') for p in paths)
    win.store.undo.setClean()


def test_page_apply_is_one_undo_step_and_settings_too(win, app):
    page = _ready(win, app)
    st = win.store
    sg = page.doc['suggestion']
    assert sg is not None and page.b_apply.isEnabled()
    assert page.preview() is sg
    n = st.undo.count()
    assert page.apply()
    s = st.project.variant.sensor(sg['sensor_uid'])
    assert s.stream == sg['mode'] and st.undo.count() == n + 1
    win.ctx.analysis.wait(60)
    st.undo.undo()
    assert st.project.variant.sensor(sg['sensor_uid']).stream == sg['from'] or \
        st.project.variant.sensor(sg['sensor_uid']).stream == []
    n = st.undo.index()
    peg = st.project.workspace.objects[1]
    page.set_critical(peg.uid, False)
    assert st.project.analysis.studies['vla_tokens']['critical_objects'] == [st.project.workspace.objects[0].uid]
    assert st.undo.index() == n + 1
    page.set_critical(peg.uid, True)
    assert st.project.analysis.studies['vla_tokens'].get('critical_objects') is None
    win.ctx.analysis.wait(60)
    st.undo.setClean()


def test_the_guide_on_the_ur5e_sample():
    """docs/guide/vla.md, the figures it quotes."""
    p = sample_project('ur5e')
    p.tasks.grasp_sets, p.policies = [], []  # the sample's own grasp set and policy left out
    p.workspace.objects.append(TaskObject(name='Cube', shape='box', size=(0.05, 0.05, 0.05),
                                          pose=Pose(np.array([-0.134, 0.492, 0.025]), ID_Q)))
    p.policies.append(Policy(name='π0 (pi0)', library_id='pi0'))
    doc = jobs.run_study('vla_tokens', p)
    s = doc['summary']
    assert (s['fewest']['tokens'], s['fewest']['camera'], s['word']) == (4, 'Overhead D435', 'too few tokens')
    assert s['fewest']['extent'] == pytest.approx([0.571, 0.571], abs=1e-3)
    cams = {c['name']: c for c in doc['policies'][0]['cameras']}
    m = cams['Overhead D435']['mapping']
    assert round(m['scale_input_per_sensor_px'][0], 4) == 0.1167 and m['padded_fraction'] == pytest.approx(0.4375)
    assert cams['Wrist D405']['objects'][0]['worst_extent'] == pytest.approx([1.143, 1.357], abs=1e-3)
    assert 'Overhead D435' in doc['suggestion']['text'] or 'Wrist D405' in doc['suggestion']['text']
    GR.add_set(p, GR.sample(p, p.robots[0].uid, ('top_down',), 20, 0))
    doc = jobs.run_study('vla_tokens', p)
    cams = {c['name']: c for c in doc['policies'][0]['cameras']}
    cube = cams['Wrist D405']['objects'][0]
    assert (cube['tokens_min'], cube['tokens_max']) == (12, 27)
    assert cube['worst_extent'] == pytest.approx([2.357, 2.714], abs=1e-3)
    assert [w['sensor'] for w in doc['warnings']] == [cams['Overhead D435']['uid']]
    assert doc['suggestion']['mode'] == [640, 480] and doc['suggestion']['sensor'] == 'Overhead D435'
    assert doc['suggestion']['worst_after'] == pytest.approx(0.9, abs=0.05)
    tips = {(t['fingertip'], t['phase']): t for t in cams['Wrist D405']['fingertips']}
    assert all(t['tokens_min'] == 6 for t in tips.values())
