"""Phase 2 review, correctness lens: regression tests for what the review checked and fixed.

- the library's imager poses: optical frames in the body frame, the stereo baseline's sign (the second imager of a
  pair on the reference imager's right, disparity f·B/z > 0), the pair's separation equal to the stated baseline;
- a link-mounted sensor posed at q (scenegeom.sensor_world_pose against the link's FK composed with the mount);
- the sensor's own seat on its link (scenegeom.sensor_occluders) and the cube map's two backends on scenes with
  triangles that cross the near plane, away from silhouettes;
- the GPU worker's geometry LRU (the client must follow the worker's least-recently-used order);
- the sparse voxel grid: refinement without double counting, chunk boundaries;
- the coverage and depth studies: chunk size does not change a result; the stream chosen for a sensor
  (Sensor.stream) is the mode the studies model, as the Camera view and the Inspector do; σ_z in metres from f in
  pixels at the chosen depth resolution and B in metres, against the disparity relation z = f·B/d;
- the result cache: a change of q, of a sensor or of a library entry (a custom device changed under its id) misses.
"""
import math

import numpy as np
import pytest

from dexvantage.core import jobs
from dexvantage.core import library as L
from dexvantage.core import occlusion as OC
from dexvantage.core import scenegeom as SG
from dexvantage.core import voxels as VX
from dexvantage.core.frames import OPTICAL_TO_BODY, Pose, quat_to_matrix
from dexvantage.core.project import Mount, Sensor, TaskObject, Volume, blank_project
from dexvantage.core.robot.samples import look_at, sample_project
from dexvantage.core.studies import coverage as CV
from dexvantage.core.studies import depth_validity as DV

STEREO = [e for e in L.entries('depth') if (e.data.get('depth') or {}).get('stereo_pair')
          and (e.data['depth'].get('baseline_m') is not None)]


# ====================================================================== library frames
@pytest.mark.parametrize('entry', STEREO, ids=lambda e: e.id)
def test_stereo_pair_baseline_sign_and_length(entry):
    """The pair's optical centres are the stated baseline apart, the second imager on the reference imager's right
    (+x of its optical frame), axes parallel; a point ahead has disparity u_ref − u_other = f·B/z > 0."""
    dp = entry.data['depth']
    s = Sensor(name='S', kind='depth', model_id=entry.id)
    a, b = dp['stereo_pair']
    mode = DV.native_depth_mode(entry)[0]
    ims = {i.name: i for i in L.sensor_imagers(s, {a: mode, b: mode})}
    A, B = ims[a], ims[b]
    assert _norm(dp['reference_imager']) == _norm(a)
    off = A.pose_in_body.inverse().apply(B.pose_in_body.position[None])[0]       # B's centre in A's optical frame
    assert off[0] == pytest.approx(dp['baseline_m'], abs=1e-9)
    assert abs(off[1]) < 1e-9 and abs(off[2]) < 1e-9
    RA, RB = quat_to_matrix(A.pose_in_body.quaternion), quat_to_matrix(B.pose_in_body.quaternion)
    assert np.allclose(RA, RB, atol=1e-12)
    z = 1.3
    P_body = A.pose_in_body.apply(np.array([[0.05, -0.02, z]]))                   # a point ahead of A
    uA = A.camera.project(A.pose_in_body.inverse().apply(P_body))[0][0, 0]
    uB = B.camera.project(B.pose_in_body.inverse().apply(P_body))[0][0, 0]
    assert uA - uB == pytest.approx(A.camera.fx * dp['baseline_m'] / z, rel=1e-9)


def _norm(s):
    return ' '.join(str(s).lower().replace('_', ' ').split())


@pytest.mark.parametrize('entry', [e for e in L.entries() if e.imagers], ids=lambda e: e.id)
def test_imager_poses_are_optical_frames_looking_along_body_x(entry):
    """Every built-in imager looks along the device's body x (REP-103 body, optical z forward, x right, y down):
    its rotation in the body frame is OPTICAL_TO_BODY, and a point ahead of the body projects near the centre."""
    s = Sensor(name='S', kind=entry.kind if entry.kind in ('camera', 'depth', 'tactile') else 'camera',
               model_id=entry.id)
    for im in L.sensor_imagers(s):
        R = quat_to_matrix(im.pose_in_body.quaternion)
        assert np.allclose(R, OPTICAL_TO_BODY, atol=1e-9), im.name
        # body +y (left) lands left of the image centre, body +z (up) above it
        P = im.pose_in_body.position + np.array([[1.0, 0, 0], [1.0, 0.05, 0], [1.0, 0, 0.05]])
        uv, ok = im.camera.project(im.pose_in_body.inverse().apply(P))
        assert ok.all(), im.name
        assert uv[1, 0] < uv[0, 0] and uv[2, 1] < uv[0, 1]


def test_rgb_offsets_are_in_the_body_frame_and_labelled():
    """An RGB imager's offset is a body-frame position; a position the maker does not state is flagged."""
    for e in L.entries('depth'):
        for im in e.imagers:
            if im['role'] != 'rgb':
                continue
            s = Sensor(name='S', kind='depth', model_id=e.id)
            got = next(i for i in L.sensor_imagers(s) if i.name == im['name'])
            assert np.allclose(got.pose_in_body.position, im['pose']['position'])
            if not im.get('pose_stated', True):
                assert not got.pose_stated and 'assumption' in got.label


# ====================================================================== a link-mounted sensor at q
def test_link_mounted_sensor_pose_follows_q():
    from dexvantage.core.robot.model import get_model
    p = sample_project('ur5e')
    r = p.robots[0]
    m = get_model(p, r.uid)
    s = next(x for x in p.variant.sensors if x.mount.on_link)
    q = m.clip(m.home + np.r_[np.random.default_rng(3).normal(0, 0.3, 6), np.zeros(m.nq - 6)])
    T = SG.sensor_world_pose(p, s, {r.uid: q}).matrix()
    ref = m.link_pose(q, s.mount.parent_frame) @ s.mount.pose.matrix()
    assert np.allclose(T, ref, atol=1e-12)
    home = SG.sensor_world_pose(p, s, None).matrix()
    assert not np.allclose(T, home, atol=1e-3)                       # it moves with the arm
    # the robot's base pose is applied (a second robot on a moved base)
    r.base = Pose(np.array([0.3, -0.2, 0.1]), np.array([0.0, 0.0, math.sin(0.4), math.cos(0.4)]))
    T2 = SG.sensor_world_pose(p, s, {r.uid: q}).matrix()
    assert np.allclose(T2, r.base.matrix() @ T, atol=1e-9)


def test_own_seat_dropped_rest_of_link_kept():
    """The mount link's triangles wholly within own_radius of an imager centre are the seat (left out); the rest of
    the link stays an occluder (a wrist camera sees its own gripper); origins of every imager count."""
    from dexvantage.core.robot.model import get_model
    p = sample_project('ur5e')
    s = next(x for x in p.variant.sensors if x.mount.on_link)
    occ = SG.scene_occluders(p, None)
    own = f'robot:{s.mount.robot_uid}:{s.mount.parent_frame}'
    n_own = int(occ.owner_mask([own]).sum())
    body = SG.sensor_world_pose(p, s, None)
    m = get_model(p, s.mount.robot_uid)
    link_T = m.link_pose(m.home, s.mount.parent_frame)
    # put the origin right on a vertex of the mount link: its nearby triangles are a seat
    v = occ.tris[occ.owner_mask([own])].reshape(-1, 3).astype(float)
    o = v[len(v) // 2]
    out = SG.sensor_occluders(p, s, None, base=occ, origins=[o])
    near = (np.linalg.norm(occ.tris[occ.owner_mask([own])] - o, axis=2) <= SG.OWN_RADIUS_M).all(axis=1)
    assert near.any()
    assert int(out.owner_mask([own]).sum()) == n_own - int(near.sum())
    assert len(out) == len(occ) - int(near.sum())
    assert link_T.shape == (4, 4) and np.isfinite(body.position).all()
    # the whole link out on request; a static sensor keeps everything
    assert int(SG.sensor_occluders(p, s, None, base=occ, exclude_mount_link=True).owner_mask([own]).sum()) == 0
    st = next(x for x in p.variant.sensors if not x.mount.on_link)
    assert len(SG.sensor_occluders(p, st, None, base=occ)) == len(occ)


# ====================================================================== the cube map's two backends
def _gl():
    st = OC.gl_status()
    if not st['available']:
        pytest.skip(f'no OpenGL for the GPU backend here: {st["reason"]}')
    return st


def test_gl_and_numpy_agree_away_from_silhouettes_with_near_plane_crossings():
    """Random triangles, some straddling the 5 mm near plane and the origin: the two backends give the same depth
    (to 0.1 %) at every texel whose neighbours in the reference map do not jump (a silhouette, where the GPU's
    sub-pixel vertex snapping may decide a texel the other way)."""
    _gl()
    rng = np.random.default_rng(11)
    T = rng.uniform(-0.5, 0.5, (300, 1, 3)) + rng.normal(0, 0.08, (300, 3, 3))
    T2 = rng.normal(0, 0.012, (60, 3, 3))                            # around the centre: cross the near plane
    tris = np.concatenate([T, T2]).astype(np.float32)
    o = np.array([0.01, -0.02, 0.03])
    n = 128
    a = OC.CubeMap.build(tris, o, n, 'numpy').faces.astype(float)
    b = OC.CubeMap.build(tris, o, n, 'gl').faces.astype(float)
    edge = np.zeros(a.shape, bool)
    for m in (a, b):                       # a silhouette in either map (a sliver one map shows is one in that map)
        M = np.where(np.isfinite(m), m, 1e3)
        for ax in (1, 2):
            for sh in (1, -1):
                nb = np.roll(M, sh, axis=ax)
                edge |= np.abs(nb - M) > 1e-3 * np.minimum(nb, M) + 1e-4
    edge[:, 0], edge[:, -1], edge[:, :, 0], edge[:, :, -1] = True, True, True, True     # face borders
    with np.errstate(invalid='ignore'):
        same = (np.isinf(a) & np.isinf(b)) | (np.abs(a - b) <= 1e-3 * np.minimum(a, b))
    assert (~same & ~edge).sum() == 0
    assert (~same).mean() < 0.01


def test_gl_worker_follows_the_workers_lru_order():
    """The worker process keeps VBOS geometries least recently used first; reusing one moves it to the end. The
    client must keep the same order, else after a reuse it believes a dropped geometry is loaded and asks the worker
    to draw it without its triangles."""
    _gl()
    try:
        w = OC._GLWorkerClient()
    except OC.OcclusionError as e:
        pytest.skip(f'the GPU worker does not start here: {e}')
    try:
        # geometry i: one large triangle on the plane x = 1 + 0.1 i, filling the +x face (depth there = its plane)
        geos = [np.array([[[1 + 0.1 * i, -4, -4], [1 + 0.1 * i, 8, -4], [1 + 0.1 * i, -4, 8]]], np.float32)
                for i in range(OC._GLRaster.VBOS + 2)]
        digests = [OC.geometry_digest(g) for g in geos]
        o, n = np.zeros(3), 128                                        # GPU vertex snapping: 1/256 texel
        x0 = OC.face_dirs(n)[0][..., 0]                                # cos of each +x texel ray to the x axis

        def face0(i, got):
            assert np.allclose(got[0], (1 + 0.1 * i) / x0, rtol=1e-4), i     # float32 on the GPU

        for i, (d, g) in enumerate(zip(digests[:OC._GLRaster.VBOS], geos)):
            face0(i, w.render(d, g, o, n, OC.NEAR_M, 5.0))
        face0(0, w.render(digests[0], geos[0], o, n, OC.NEAR_M, 5.0))   # reuse the oldest: the worker moves it last
        face0(6, w.render(digests[6], geos[6], o, n, OC.NEAR_M, 5.0))   # a new one: the worker drops geometry 1
        assert w.has(digests[0]) and not w.has(digests[1])
        face0(1, w.render(digests[1], geos[1], o, n, OC.NEAR_M, 5.0))
        # a client that believes the worker holds a geometry it does not: uploaded again, not an error
        w.loaded[digests[7]] = True
        face0(7, w.render(digests[7], geos[7], o, n, OC.NEAR_M, 5.0))
    finally:
        w.close()


# ====================================================================== voxels
def _vox_project():
    p = blank_project()
    p.workspace.volumes = [Volume(name='A', center=(0.4, 0.0, 0.1), size=(0.13, 0.11, 0.07)),
                           Volume(name='B', center=(0.5, 0.05, 0.12), size=(0.06, 0.06, 0.04), refine=True)]
    p.workspace.objects = [TaskObject(name='Box', shape='box', size=(0.03, 0.02, 0.02),
                                      pose=Pose(np.array([0.41, 0.01, 0.09])))]
    p.analysis.voxel_mm, p.analysis.refine_mm = 10.0, 4.0
    return p


def test_refinement_does_not_double_count():
    p = _vox_project()
    g = VX.SparseGrid.from_project(p)
    h, k = 0.01, 3
    assert g.refine_m == pytest.approx(h / k)
    for i, v in enumerate(p.workspace.volumes):
        m = g.volume_index == i
        d = VX.dims(v.size, h)
        # the voxels of a volume fill exactly its base grid: refined cells are replaced, not added to
        assert g.volume_m3[m].sum() == pytest.approx(float(np.prod(d)) * h ** 3, rel=1e-6)
        c = g.centers[m].astype(float)
        q = np.round((c - (np.asarray(v.center) - d * h / 2)) / (h / k / 2)).astype(np.int64)
        assert len(np.unique(q, axis=0)) == len(q)                   # no voxel twice within a volume
        base = np.floor(q / (2 * k)).astype(np.int64)
        cells, counts = np.unique(base, axis=0, return_counts=True)
        assert len(cells) == int(np.prod(d))                         # every base cell present
        assert set(np.unique(counts)) <= {1, k ** 3}
    assert g.refined[g.volume_index == 1].all()                      # a refinement region: all fine
    a = g.volume_index == 0
    assert 0 < g.refined[a].sum() < a.sum()                          # around the object only
    est = VX.estimate(p)
    assert est['count'] == g.count


def test_chunks_cover_the_grid_once():
    g = VX.SparseGrid.from_project(_vox_project())
    for n in (1, 7, 1000, g.count, g.count + 5):
        parts = list(g.chunks(n))
        assert parts[0].start == 0 and parts[-1].stop == g.count
        assert all(a.stop == b.start for a, b in zip(parts, parts[1:]))
        assert np.array_equal(np.concatenate([c.centers for c in parts]), g.centers)


# ====================================================================== studies
def _cov_project():
    p = blank_project()
    p.workspace.volumes = [Volume(name='Table', center=(0.45, 0.0, 0.08), size=(0.3, 0.3, 0.16))]
    p.workspace.objects = [TaskObject(name='Block', shape='box', size=(0.06, 0.06, 0.06),
                                      pose=Pose(np.array([0.45, 0.02, 0.03])))]
    p.workspace.surfaces = [{'uid': 't', 'name': 'Table', 'z': 0.0, 'size': [1.0, 1.0]}]
    for k, eye in enumerate(((0.45, -0.5, 0.5), (0.9, 0.3, 0.45))):
        p.variant.sensors.append(Sensor(name=f'C{k}', slot=k, model_id='intel-realsense-d435', kind='depth',
                                        mount=Mount('cage', '', look_at(eye, (0.45, 0.0, 0.05)))))
    p.analysis.voxel_mm = p.analysis.refine_mm = 10.0
    return p


def test_coverage_does_not_depend_on_the_chunk_size():
    p = _cov_project()
    p.analysis.studies['coverage'] = {'backend': 'numpy', 'cube_n': 128}
    a = CV.decode(jobs.run_study('coverage', p))
    p.analysis.studies['coverage'] = {'backend': 'numpy', 'cube_n': 128, 'chunk': 1001}
    b = CV.decode(jobs.run_study('coverage', p))
    assert np.array_equal(a['redundancy'], b['redundancy'])
    for ca, cb in zip(a['cameras'], b['cameras']):
        assert np.array_equal(ca['state'], cb['state']) and np.array_equal(ca['ppmm'], cb['ppmm'])


def test_depth_validity_does_not_depend_on_the_chunk_size(monkeypatch):
    p = _cov_project()
    p.analysis.studies['depth_validity'] = {'backend': 'numpy', 'cube_n': 128}
    a = jobs.run_study('depth_validity', p)
    monkeypatch.setattr(DV, 'CHUNK', 777)
    b = jobs.run_study('depth_validity', p)
    for sa, sb in zip(a['sensors'], b['sensors']):
        assert sa['breakdown'] == sb['breakdown'] and sa['valid_pct'] == sb['valid_pct']


def test_coverage_models_the_stream_chosen_for_the_sensor():
    """Sensor.stream (the Sensors page) sets the imagers' image size in the Camera view and the frusta; coverage
    modelled every camera at its largest stream instead (2.26× the pixels per mm of an 848 × 480 stream)."""
    p = _cov_project()
    p.variant.sensors = p.variant.sensors[:1]
    p.analysis.studies['coverage'] = {'backend': 'numpy', 'cube_n': 64}
    big = jobs.run_study('coverage', p)['cameras'][0]
    assert big['mode'] == [1920, 1080]
    p.variant.sensors[0].stream = [848, 480]
    small = jobs.run_study('coverage', p)['cameras'][0]
    assert small['mode'] == [848, 480]
    assert small['ppmm_median'] == pytest.approx(big['ppmm_median'] * 848 / 1920, rel=0.02)
    # the study's own per-sensor setting still wins
    p.analysis.studies['coverage']['modes'] = {p.variant.sensors[0].uid: [640, 480]}
    assert jobs.run_study('coverage', p)['cameras'][0]['mode'] == [640, 480]


def test_depth_resolution_follows_the_sensor_stream_and_min_z_with_it():
    p = _cov_project()
    s = p.variant.sensors[0]
    ds = DV.setup(p, s)
    assert ds.resolution == (1280, 720) and ds.resolution_default and ds.min_z_m == pytest.approx(0.28)
    s.stream = [848, 480]
    ds = DV.setup(p, s)
    assert ds.resolution == (848, 480) and not ds.resolution_default and 'stream chosen' in ds.resolution_note
    assert ds.min_z_m == pytest.approx(0.195) == L.min_z(s, 848, 480)
    s.stream = [1920, 1080]                                          # an RGB-only stream: not a depth mode
    assert DV.setup(p, s).resolution == (1280, 720)
    s.stream = [848, 480]
    p.analysis.studies['depth_validity'] = {'sensors': {s.uid: {'resolution': [424, 240]}}}
    ds = DV.setup(p, s)
    assert ds.resolution == (424, 240) and ds.min_z_m == pytest.approx(0.105)


def test_sigma_z_units_and_the_disparity_relation():
    """σ_z = z² σ_d / (f B) with f in pixels at the chosen depth stream (from the stated 16:9 depth FOV, 87°) and B
    in metres gives metres; it matches the change of z = f B / d when d moves by σ_d."""
    p = _cov_project()
    s = p.variant.sensors[0]
    s.stream = [848, 480]
    ds = DV.setup(p, s)
    f = (848 / 2) / math.tan(math.radians(87.0 / 2))
    assert ds.f_px == pytest.approx(f, rel=1e-9) and ds.baseline_m == pytest.approx(0.05)
    z = 0.8
    d = f * ds.baseline_m / z
    numeric = 0.5 * (f * ds.baseline_m / (d - ds.sigma_d_px) - f * ds.baseline_m / (d + ds.sigma_d_px))
    assert float(DV.sigma_z(z, ds.f_px, ds.baseline_m, ds.sigma_d_px)) == pytest.approx(numeric, rel=1e-4)
    assert 1e-4 < float(DV.sigma_z(z, ds.f_px, ds.baseline_m, ds.sigma_d_px)) < 1e-2      # a millimetre, not km
    p.analysis.studies['depth_validity'] = {'backend': 'numpy', 'cube_n': 64}
    doc = jobs.run_study('depth_validity', p)
    rec = next(x for x in doc['sensors'] if x['uid'] == s.uid)
    assert rec['resolution'] == [848, 480] and rec['f_px'] == pytest.approx(f)
    assert rec['maker_rms']['model_sigma_m'] == pytest.approx(2.0 ** 2 * 0.08 / (f * 0.05))


# ====================================================================== the result cache
def test_result_cache_misses_when_q_sensor_or_library_entry_changes():
    cache = jobs.ResultCache()
    p = _cov_project()
    p.variant.sensors[0].model_id = 'review-test-camera'
    L.add_custom({'id': 'review-test-camera', 'kind': 'camera', 'maker': 'Test maker', 'name': 'Bench camera',
                  'sources': [{'url': 'measured on the bench', 'title': 'own measurement', 'read': '2026-11-06',
                               'covers': 'FOV'}],
                  'note': 'A test device.', 'imagers': [{'name': 'Main', 'role': 'mono', 'hfov_deg': 70.0,
                                                         'vfov_deg': None, 'resolution': [800, 600],
                                                         'lens_model': 'pinhole', 'shutter': 'global'}]})
    try:
        k0 = cache.key('coverage', p)
        assert cache.key('coverage', p) == k0
        p.analysis.studies['coverage'] = {'q': {'robot-x': [0.1, 0.2]}}
        k1 = cache.key('coverage', p)
        assert k1 != k0
        p.analysis.studies['coverage'] = {'q': {'robot-x': [0.1, 0.25]}}
        assert cache.key('coverage', p) != k1
        k2 = cache.key('coverage', p)
        p.variant.sensors[1].mount.pose = Pose(p.variant.sensors[1].mount.pose.position + 0.001,
                                               p.variant.sensors[1].mount.pose.quaternion)
        k3 = cache.key('coverage', p)
        assert k3 != k2
        g = L.generation()
        e = L.get('review-test-camera').data
        L.add_custom(dict(e, imagers=[dict(e['imagers'][0], hfov_deg=60.0)]))    # the same id, a new FOV
        assert L.generation() != g
        assert cache.key('coverage', p) != k3
        # the study itself: a cached result is not reused after the entry changed
        p.analysis.studies['coverage'] = {'backend': 'numpy', 'cube_n': 32}
        a = jobs.run_study('coverage', p, cache=cache)
        assert jobs.run_study('coverage', p, cache=cache)['cached']
        L.add_custom(dict(e, imagers=[dict(e['imagers'][0], hfov_deg=50.0)]))
        b = jobs.run_study('coverage', p, cache=cache)
        assert not b['cached']
        fa = next(c for c in a['cameras'] if c['model_id'] == 'review-test-camera')['fov_deg']['h']
        fb = next(c for c in b['cameras'] if c['model_id'] == 'review-test-camera')['fov_deg']['h']
        assert fa == pytest.approx(60.0, abs=0.01) and fb == pytest.approx(50.0, abs=0.01)
    finally:
        L.remove_custom('review-test-camera')


def test_depth_validity_leaves_the_inside_of_solid_objects_out_as_coverage_does():
    """A solid block filling 12 % of the volume read as 12 points of 'occluded' (its inside), so depth validity said
    79 % where coverage, which leaves a solid's inside out, said 91 % seen. Both now share one task volume."""
    p = _cov_project()
    p.variant.sensors = p.variant.sensors[:1]
    p.workspace.objects = [TaskObject(name='Block', shape='box', size=(0.12, 0.12, 0.12),
                                      pose=Pose(np.array([0.45, 0.0, 0.06])))]
    p.analysis.studies['depth_validity'] = {'backend': 'numpy', 'cube_n': 128}
    p.analysis.studies['coverage'] = {'backend': 'numpy', 'cube_n': 128}
    d = jobs.run_study('depth_validity', p)
    c = jobs.run_study('coverage', p)
    assert d['voxels']['inside_objects'] == c['grid']['inside_objects'] > 0
    assert d['voxels']['volume_m3'] == pytest.approx(c['grid']['task_volume_m3'], rel=1e-6)
    s = d['sensors'][0]
    assert sum(s['breakdown'].values()) == pytest.approx(100.0)
    # a depth-valid voxel needs both IR imagers, a covered one only the RGB imager: depth ≤ coverage, and close here
    assert s['valid_pct'] <= 100 * c['cameras'][0]['seen'] + 1.0
    assert s['valid_pct'] > 100 * c['cameras'][0]['seen'] - 5.0
    g = VX.SparseGrid.from_project(p)
    inside = CV._inside_objects(p, g.centers.astype(float))
    pts = np.asarray(s['overlay']['points_mm'], float).reshape(-1, 3) / 1000
    assert not CV._inside_objects(p, pts).any() and inside.any()      # the 3D overlay shows no voxel inside a solid
