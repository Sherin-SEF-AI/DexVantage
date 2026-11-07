"""core/raycast.py, core/occlusion.py and core/scenegeom.py: exact ray casting against brute force, the cube map
against closed forms and against exact ray casting on the posed UR5e sample, the two backends against each other,
the scene's occluders and the GPU fallbacks."""
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from dexvantage.core import occlusion as OC
from dexvantage.core import raycast as RC
from dexvantage.core import scenegeom as SG
from dexvantage.core.frames import Pose, quat_from_axis_angle
from dexvantage.core.project import Mount, Sensor, TaskObject, blank_project

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def _gl():
    st = OC.gl_status()
    if not st['available']:
        pytest.skip(f'no OpenGL for the GPU backend here: {st["reason"]}')
    return st


@pytest.fixture(scope='module')
def ur5e():
    """The UR5e sample posed away from home (a fixed q), its occluders and the two cameras' centres."""
    from dexvantage.core.robot.model import get_model
    from dexvantage.core.robot.samples import sample_project
    p = sample_project('ur5e')
    r = p.robots[0]
    m = get_model(p, r.uid)
    q = m.home.copy()
    q[:6] += np.random.default_rng(5).normal(0, 0.15, 6)         # the arm's six joints; the gripper at home
    q = m.clip(q)
    qd = {r.uid: q}
    occ = SG.scene_occluders(p, qd)
    origins = {s.name: SG.sensor_world_pose(p, s, qd).position for s in p.variant.sensors}
    lo, hi = p.workspace.volumes[0].bounds()
    pts = np.random.default_rng(11).uniform(lo, hi, (6000, 3))
    return dict(p=p, q=qd, occ=occ, origins=origins, pts=pts, grid=RC.Grid.build(occ.tris))


# ---------------------------------------------------------------------------------------------- ray casting
def test_moller_trumbore_single_triangle():
    tri = np.array([[[0.0, -1, -1], [0, 1, -1], [0, 0, 1]]])
    t = RC.brute_first_hit([[-2.0, 0, 0], [-2.0, 0, 0], [-2.0, 5, 0]], [[1.0, 0, 0], [-1.0, 0, 0], [1.0, 0, 0]], tri)
    assert t[0] == pytest.approx(2.0, abs=1e-12)
    assert np.isinf(t[1]) and np.isinf(t[2])            # behind the ray, and beside the triangle


def test_grid_raycast_matches_brute_force():
    rng = np.random.default_rng(1)
    c = rng.uniform(-1, 1, (2500, 3))
    tris = c[:, None, :] + rng.normal(0, 0.06, (2500, 3, 3))
    tris = np.concatenate([tris, [[[-3, -3, -1.2], [3, -3, -1.2], [3, 3, -1.2]]]])   # one large triangle too
    o = rng.uniform(-2, 2, (3000, 3))
    d = rng.normal(size=(3000, 3))
    a, b = RC.first_hit(o, d, tris), RC.brute_first_hit(o, d, tris)
    assert np.array_equal(np.isfinite(a), np.isfinite(b))
    fin = np.isfinite(a)
    assert np.abs(a[fin] - b[fin]).max() < 1e-12
    a = RC.first_hit(np.zeros(3), d, tris, tmax=0.7)    # one origin, a distance limit
    b = RC.brute_first_hit(np.zeros(3), d, tris)
    assert np.array_equal(np.isfinite(a), b <= 0.7)


def test_visible_exact_box():
    from dexvantage.ui.scene import box_tris
    box = box_tris((0.4, -0.1, -0.1), (0.6, 0.1, 0.1))
    v = RC.visible_exact([[1.0, 0, 0], [0.3, 0, 0], [1.0, 0.5, 0], [0.4, 0, 0]], np.zeros(3), box)
    assert v.tolist() == [False, True, True, True]      # behind, in front, beside, on the near face itself


# ---------------------------------------------------------------------------------------------- the cube map
def test_texel_index_round_trip():
    n = 32
    d = OC.face_dirs(n)
    f, r, c = OC.texel_index(d.reshape(-1, 3), n)
    F, R, C = np.meshgrid(np.arange(6), np.arange(n), np.arange(n), indexing='ij')
    assert np.array_equal(f, F.ravel()) and np.array_equal(r, R.ravel()) and np.array_equal(c, C.ravel())


@pytest.mark.parametrize('backend', ['numpy', 'gl'])
def test_plane_depth_is_exact(backend):
    """A wall at x = 1 m in front of the centre: every texel of the +x face holds 1 / cos(angle to the axis)."""
    if backend == 'gl':
        _gl()
    wall = np.array([[[1.0, -3, -3], [1, 3, -3], [1, 3, 3]], [[1.0, -3, -3], [1, 3, 3], [1, -3, 3]]])
    n = 64
    cube = OC.CubeMap.build(wall, np.zeros(3), n, backend)
    assert cube.backend == backend
    d = OC.face_dirs(n)[0]
    exact = 1.0 / d[..., 0]
    assert np.abs(cube.faces[0] - exact).max() < 1e-5
    assert np.isinf(cube.faces[1]).all()                # nothing behind


@pytest.mark.parametrize('backend', ['numpy', 'gl'])
def test_point_behind_box_is_hidden(backend):
    if backend == 'gl':
        _gl()
    from dexvantage.ui.scene import box_tris
    box = box_tris((0.4, -0.1, -0.1), (0.6, 0.1, 0.1))
    cube = OC.CubeMap.build(box, np.zeros(3), 128, backend)
    vis = cube.visible([[1.0, 0, 0], [0.3, 0, 0], [1.0, 0.5, 0], [0.4, 0.0, 0.0], [-1.0, 0, 0], [0.5, 0, 0]])
    assert vis.tolist() == [False, True, True, True, True, False]
    # the texel's centre ray is up to √2/n rad off the axis: its plane distance is 0.4 / cos of that
    assert cube.depth([[1.0, 0, 0]])[0] == pytest.approx(0.4 / np.cos(np.sqrt(2) / 128), abs=1e-5)


def test_near_clip_ignores_the_surface_the_sensor_sits_on():
    seat = np.array([[[0.002, -1, -1], [0.002, 1, -1], [0.002, 0, 1]]])    # 2 mm ahead: inside the near clip
    far = np.array([[[0.5, -1, -1], [0.5, 1, -1], [0.5, 0, 1]]])
    cube = OC.CubeMap.build(np.concatenate([seat, far]), np.zeros(3), 32, 'numpy')
    assert cube.depth([[1.0, 0, 0]])[0] == pytest.approx(0.5 / np.cos(np.sqrt(2) / 32), abs=1e-6)


def test_cube_matches_raycast_on_posed_ur5e(ur5e):
    """The cube map against exact ray casting from both cameras of the posed UR5e sample: few points disagree, and
    every disagreeing point is within two texels of a silhouette."""
    for name, o in ur5e['origins'].items():
        cube = OC.CubeMap.build(ur5e['occ'], o, 512)
        vc = cube.visible(ur5e['pts'])
        ve = RC.visible_exact(ur5e['pts'], o, ur5e['grid'])
        bad = np.nonzero(vc != ve)[0]
        frac = len(bad) / len(vc)
        assert 0.05 < ve.mean() < 0.999, name                       # the scene does occlude something
        assert frac < 0.01, (name, frac)
        sil = RC.near_silhouette(ur5e['pts'][bad], o, ur5e['grid'], 2.0 / cube.n)
        assert sil.all(), (name, int((~sil).sum()), 'disagreements away from any silhouette')


def test_backends_agree_on_posed_ur5e(ur5e):
    _gl()
    for o in ur5e['origins'].values():
        for n in (256, 512):
            a = OC.CubeMap.build(ur5e['occ'], o, n, 'gl')
            b = OC.CubeMap.build(ur5e['occ'], o, n, 'numpy')
            assert a.backend == 'gl' and b.backend == 'numpy'
            fa, fb = np.isfinite(a.faces), np.isfinite(b.faces)
            assert (fa != fb).mean() < 1e-4
            both = fa & fb
            diff = np.abs(a.faces[both].astype(float) - b.faces[both])
            assert (diff > 1e-3).mean() < 1e-3
            assert np.median(diff) < 1e-5
            pts = ur5e['pts']
            assert (a.visible(pts) != b.visible(pts)).mean() < 2e-3


def test_cache_and_convenience(ur5e):
    cache = OC.CubeCache()
    o = ur5e['origins']['Overhead D435']
    c1 = cache.get(ur5e['occ'], o, 64, 'numpy')
    c2 = cache.get(ur5e['occ'], o.copy(), 64, 'numpy')            # the same origin: a hit
    assert c1 is c2 and cache.hits == 1 and len(cache) == 1
    c3 = cache.get(ur5e['occ'], o + 1e-6, 64, 'numpy')            # 1 µm away: its own map (the key is exact; it
    assert c3 is not c1 and cache.misses == 2 and len(cache) == 2  # was rounded to 0.1 mm before)
    v = OC.visible(ur5e['pts'][:50], o, ur5e['occ'], n=64, backend='numpy')
    assert v.dtype == bool and v.shape == (50,)


def test_decimation_report(ur5e):
    o = ur5e['origins']['Overhead D435']
    dec = SG.decimate(ur5e['occ'], 0.005)
    assert 0 < len(dec) < len(ur5e['occ']) / 3
    assert set(dec.owners) == set(ur5e['occ'].owners) and dec.info['triangles_before'] == len(ur5e['occ'])
    rep = OC.decimation_report(ur5e['occ'], o, ur5e['pts'][:2000], 0.005, n=256, backend='numpy')
    assert rep['triangles_decimated'] == len(dec)
    assert 0 <= rep['changed_fraction'] < 0.05


def test_bad_inputs():
    with pytest.raises(ValueError):
        OC.CubeMap.build(np.zeros((3, 3)), np.zeros(3), 32)
    with pytest.raises(ValueError):
        OC.CubeMap.build(np.zeros((0, 3, 3)), np.zeros(3), 1)
    with pytest.raises(ValueError):
        OC.resolve_backend('vulkan')
    empty = OC.CubeMap.build(np.zeros((0, 3, 3)), np.zeros(3), 16)
    assert np.isinf(empty.faces).all() and empty.visible([[1.0, 2, 3]]).all()


# ---------------------------------------------------------------------------------------------- the scene
def test_scene_occluders_owners_and_exclusions(ur5e):
    p, occ = ur5e['p'], ur5e['occ']
    uid = p.robots[0].uid
    assert len(occ) > 100_000 and occ.tris.dtype == np.float32 and occ.owner.dtype == np.int32
    assert f'robot:{uid}:wrist_3_link' in occ.owners and 'surface:table' in occ.owners
    no_w3 = SG.scene_occluders(p, ur5e['q'], exclude_links=[(uid, 'wrist_3_link')])
    assert f'robot:{uid}:wrist_3_link' not in no_w3.owners
    assert len(no_w3) == len(occ) - int(occ.owner_mask([f'robot:{uid}:wrist_3_link']).sum())
    assert len(SG.scene_occluders(p, ur5e['q'], exclude_links=['wrist_3_link'])) == len(no_w3)
    only = SG.scene_occluders(p, ur5e['q'], robots=False)
    assert only.owners == ['surface:table'] and len(only) == 2
    home = SG.scene_occluders(p, {})
    assert not np.array_equal(home.tris, occ.tris)                  # q moves the arm


def test_sensor_world_pose_and_own_rule(ur5e):
    from dexvantage.core.robot.model import get_model
    p, q = ur5e['p'], ur5e['q']
    wrist = next(s for s in p.variant.sensors if s.mount.on_link)
    m = get_model(p, wrist.mount.robot_uid)
    T = m.link_pose(q[wrist.mount.robot_uid], wrist.mount.parent_frame) @ wrist.mount.pose.matrix()
    pose = SG.sensor_world_pose(p, wrist, q)
    assert np.allclose(pose.matrix(), T, atol=1e-12)
    static = next(s for s in p.variant.sensors if not s.mount.on_link)
    assert np.allclose(SG.sensor_world_pose(p, static, q).position, static.mount.pose.position)
    own = f'robot:{wrist.mount.robot_uid}:{wrist.mount.parent_frame}'
    occ = ur5e['occ']
    big = SG.sensor_occluders(p, wrist, q, own_radius=0.08, base=occ)
    dropped = len(occ) - len(big)
    assert dropped == big.info.get('own_dropped', 0)
    gone = occ.tris[occ.owner_mask([own])]
    if dropped:
        assert len(big.tris[big.owner_mask([own])]) == len(gone) - dropped
    without = SG.sensor_occluders(p, wrist, q, exclude_mount_link=True, base=occ)
    assert own not in without.owners and len(without) == len(occ) - len(gone)
    assert SG.sensor_occluders(p, static, q, base=occ) is occ


def test_objects_and_surfaces():
    p = blank_project()
    r = 0.04
    p.workspace.objects = [
        TaskObject(uid='bx', shape='box', size=(0.1, 0.2, 0.3), pose=Pose([1.0, 0, 0])),
        TaskObject(uid='cy', shape='cylinder', size=(r, 0.1, 0), pose=Pose([0, 1.0, 0], quat_from_axis_angle(
            [1, 0, 0], np.pi / 2))),
        TaskObject(uid='sp', shape='sphere', size=(r, 0, 0), pose=Pose([0, 0, 1.0])),
        TaskObject(uid='me', shape='mesh', mesh_key='tri', pose=Pose([0, 0, -1.0])),
        TaskObject(uid='bad', shape='box', size=(0, 0.1, 0.1)),
    ]
    p.meshes['tri'] = np.array([[0.0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]])
    p.workspace.surfaces = [{'uid': 'table', 'z': 0.7, 'size': [1.0, 0.5], 'center': [0.2, 0]},
                            {'uid': 'slab', 'z': 0.0, 'size': [0.2, 0.2], 'thickness': 0.03}]
    occ = SG.scene_occluders(p, {})
    assert {'object:bx', 'object:cy', 'object:sp', 'object:me', 'surface:table', 'surface:slab'} <= set(occ.owners)
    assert 'object:bad' not in occ.owners and any('bad' in n or 'Object' in n for n in occ.notes)
    bx = occ.tris[occ.owner_mask(['object:bx'])].reshape(-1, 3)
    assert np.allclose(bx.min(axis=0), [0.95, -0.1, -0.15]) and np.allclose(bx.max(axis=0), [1.05, 0.1, 0.15])
    cy = occ.tris[occ.owner_mask(['object:cy'])].reshape(-1, 3).astype(float) - [0, 1.0, 0]
    assert np.abs(cy[:, 1]).max() == pytest.approx(0.05, abs=1e-6)          # the axis turned onto y
    rho = np.hypot(cy[:, 0], cy[:, 2])
    assert rho.max() == pytest.approx(r, abs=1e-6)
    sp = occ.tris[occ.owner_mask(['object:sp'])].reshape(-1, 3).astype(float) - [0, 0, 1.0]
    assert np.linalg.norm(sp, axis=1).max() == pytest.approx(r, abs=1e-6)
    assert len(occ.tris[occ.owner_mask(['object:me'])]) == 1
    tb = occ.tris[occ.owner_mask(['surface:table'])].reshape(-1, 3)
    assert np.allclose(tb.min(axis=0), [-0.3, -0.25, 0.7]) and np.allclose(tb.max(axis=0), [0.7, 0.25, 0.7])
    sl = occ.tris[occ.owner_mask(['surface:slab'])].reshape(-1, 3)
    assert len(sl) == 36 and sl[:, 2].min() == pytest.approx(-0.03)
    # a point under the table is hidden from a camera above it
    cube = OC.CubeMap.build(occ, [0.2, 0.0, 1.5], 64, 'numpy')
    assert cube.visible([[0.2, 0.1, 0.5], [0.2, 0.1, 0.8]]).tolist() == [False, True]


def test_sensor_without_robot_pose_raises():
    from dexvantage.core.robot.model import RobotModelError
    p = blank_project()
    s = Sensor(mount=Mount('wrist_3_link', 'nobody'))
    with pytest.raises(RobotModelError):
        SG.sensor_world_pose(p, s, {})


# ---------------------------------------------------------------------------------------------- GPU fallbacks
def _run(code: str, env: dict) -> dict:
    e = dict(os.environ, QT_QPA_PLATFORM='offscreen', **env)
    r = subprocess.run([sys.executable, '-c', textwrap.dedent(code)], cwd=ROOT, env=e, capture_output=True,
                       text=True, timeout=180)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_numpy_fallback_with_reason():
    out = _run("""
        import json, numpy as np
        from dexvantage.core import occlusion as OC
        wall = np.array([[[1.0, -3, -3], [1, 3, -3], [1, 3, 3]]])
        c = OC.CubeMap.build(wall, np.zeros(3), 16)
        try:
            OC.CubeMap.build(wall, np.zeros(3), 16, 'gl'); err = ''
        except OC.OcclusionError as e:
            err = str(e)
        print(json.dumps({'backend': c.backend, 'status': OC.gl_status(), 'err': err}))
    """, {'DEXVANTAGE_NO_GL': '1'})
    assert out['backend'] == 'numpy'
    assert not out['status']['available'] and 'DEXVANTAGE_NO_GL' in out['status']['reason']
    assert 'not available' in out['err']


def test_worker_process_when_pyopengl_is_glx():
    """With PyOpenGL loaded for GLX (an X11 desktop), the GPU backend runs in its worker process and still agrees
    with numpy."""
    _gl()
    out = _run("""
        import json, numpy as np
        import OpenGL.GL  # loaded for GLX, as Qt on X11 would
        from dexvantage.core import occlusion as OC
        from dexvantage.ui.scene import box_tris
        box = box_tris((0.4, -0.1, -0.1), (0.6, 0.1, 0.1))
        g = OC.CubeMap.build(box, np.zeros(3), 64, 'auto')
        a = OC.CubeMap.build(box, np.zeros(3), 64, 'numpy')
        same = bool((np.isfinite(g.faces) == np.isfinite(a.faces)).all())
        print(json.dumps({'backend': g.backend, 'status': OC.gl_status(), 'same': same}))
    """, {'PYOPENGL_PLATFORM': 'glx'})
    if not out['status']['available']:
        pytest.skip(f'the GPU worker could not start here: {out["status"]["reason"]}')
    assert out['backend'] == 'gl' and out['status']['mode'] == 'worker process' and out['same']
