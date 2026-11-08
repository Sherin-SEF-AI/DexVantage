"""Phase 2 integration: the samples' sensors and the library, batch coverage and depth validity on the samples, the
library and occlusion self-test lines, the posing and smooth-normal speed-ups the 3D view relies on, and the SO-100
wrist camera's placement."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


@pytest.fixture(scope='module')
def samples():
    from dexvantage.core.robot.samples import sample_project
    return {n: sample_project(n) for n in ('so100', 'ur5e')}


def test_ur5e_sample_sensors_resolve_to_datasheet_imagers(samples):
    from dexvantage.core import library as L
    p = samples['ur5e']
    ids = {s.name: s.model_id for s in p.variant.sensors}
    assert ids == {'Wrist D405': 'intel-realsense-d405', 'Overhead D435': 'intel-realsense-d435'}
    for s in p.variant.sensors:
        ims = L.sensor_imagers(s)
        assert ims and not any(i.is_default for i in ims)
        assert L.depth_spec(s)['baseline_m'] > 0
    # the SO-100's cameras have no sourced device: the labelled default, said in the project notes
    so = samples['so100']
    assert all(s.model_id == '' for s in so.variant.sensors)
    assert all(L.sensor_imagers(s)[0].is_default for s in so.variant.sensors)
    assert 'labelled default camera' in so.notes and 'Phase 2' not in so.notes


def test_every_library_file_validates_against_the_schema():
    jsonschema = pytest.importorskip('jsonschema', reason='jsonschema (the test extra) is not installed')
    import glob
    from dexvantage.core import library as L
    v = jsonschema.Draft202012Validator(L.schema())
    files = sorted(glob.glob(os.path.join(L.BUILTIN_DIR, '**', '*.json'), recursive=True))
    assert len(files) >= 40
    for f in files:
        with open(f, encoding='utf-8') as fh:
            assert not list(v.iter_errors(json.load(fh))), f
    L.reload()
    assert L.problems() == [] and len(L.entries()) == len(files)


def test_so100_wrist_camera_is_not_hidden_by_its_own_jaw(samples):
    """The wrist camera sat on the back of the fixed jaw, which hid 39 % of the task volume from it; it now looks
    past the jaws from the wrist side."""
    from dexvantage.core import jobs
    from dexvantage.core.studies import coverage  # noqa: F401  (registers the study)
    p = samples['so100'].snapshot()
    wrist = p.variant.sensors[0]
    p.analysis.studies['coverage'] = {'voxel_mm': 10, 'refine_mm': 10, 'sensors': [wrist.uid], 'backend': 'numpy'}
    cam = jobs.run_study('coverage', p)['cameras'][0]
    assert cam['hidden'] < 0.2 and cam['seen'] > 0.4


@pytest.mark.parametrize('flag', ['coverage', 'depth-validity'])
def test_batch_flags_on_the_ur5e_sample(tmp_path, flag):
    out = tmp_path / f'{flag}.json'
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    r = subprocess.run([sys.executable, '-m', 'dexvantage', '--sample', 'ur5e', f'--{flag}', str(out)], cwd=ROOT,
                       env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    doc = json.loads(out.read_text())
    assert 'provenance' in doc
    if flag == 'coverage':
        assert {c['name'] for c in doc['cameras']} == {'Wrist D405', 'Overhead D435'}
        assert 0 < doc['summary']['seen_ge1'] <= 1
    else:
        assert {s['name'] for s in doc['sensors']} == {'Wrist D405', 'Overhead D435'}


@pytest.mark.parametrize('name', ['library', 'occlusion'])
def test_selftest_lines(app, name):
    from dexvantage import selftest as T
    status, msg = dict(T.CHECKS)[name]({})
    assert status == T.OK, msg
    assert name in T.NAMES and T.NAMES.index(name) < T.NAMES.index('opencv')


def test_posed_triangles_match_the_per_triangle_product(samples):
    from dexvantage.core.robot.model import get_model
    for p in samples.values():
        m = get_model(p, p.robots[0].uid)
        q = m.home + 0.1 * (m.upper > m.lower)
        tri, _n, li = m.posed_triangles(q)
        T = m.fk_array(q)
        ref = np.einsum('mij,mkj->mki', T[li, :3, :3], m.local_triangles()) + T[li, None, :3, 3]
        assert np.abs(tri - ref).max() < 1e-6
        V = np.random.default_rng(1).normal(size=(len(li), 3, 3))
        assert np.abs(m.pose_vectors(q, V) - np.einsum('mij,mkj->mki', T[li, :3, :3], V)).max() < 1e-5


def test_scene_smooth_normals_are_posed_not_recomputed(win, samples):
    """The scene computes each model's smooth normals once in link frames and rotates them; the result matches
    smoothing the posed mesh per link, except at the few corners where the 10 µm grid that merges coinciding
    vertices falls differently in the two frames (0.1 % of the SO-100's corners, 0.8 % of the UR5e's, measured)."""
    from dexvantage.ui.render_gl import vertex_normals
    win.store.set_project(samples['so100'].snapshot())
    sc = win.scene
    r, m, _q = sc.robot_models()[0]
    q = m.home + 0.2 * (m.upper > m.lower)
    win.store.set_q(r.uid, q)
    md = sc.mesh()
    q = sc.robot_models()[0][2]                        # the joint state the scene posed (the store clips to limits)
    assert md.vnormals is not None and md.vnormals.shape == md.tri.shape
    tri, n, li = m.posed_triangles(q)
    ref = vertex_normals(tri, n, group=li)
    off = np.abs(md.vnormals[:len(ref)] - ref).max(axis=2) > 1e-3
    assert off.mean() < 0.02
    assert np.allclose(np.linalg.norm(md.vnormals, axis=2), 1.0, atol=1e-4)
