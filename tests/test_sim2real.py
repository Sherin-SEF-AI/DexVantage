"""The sim-to-real gap (core/sim2real): the measures on arrays against closed forms, the settings and refusals, the
analysis on synthetic UR5e recordings with KNOWN injected gaps (a clean one, a brightness offset, a blur, a latency, a
mount offset) recovered and ranked first, Apply closing the gap it explains, the lighting rule, depth through a decoded
depth stream, the study with its metrics, batch flag, API and assistant tool, and the page (Apply in one undo step).
Small: one episode of 40 rows at 320 × 180 px."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import pump

pytest.importorskip('pyarrow', reason='pyarrow (the lerobot extra) is not installed')
pytest.importorskip('av', reason='av (the lerobot extra) is not installed')

from dexvantage.core import jobs  # noqa: E402
from dexvantage.core.robot.samples import sample_project  # noqa: E402
from dexvantage.core.sim2real import gap as G  # noqa: E402
from dexvantage.core.sim2real import inject as IJ  # noqa: E402
from dexvantage.core.sim2real import measures as MS  # noqa: E402
from dexvantage.validation import checks_sim2real as CS  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
GEN = {'episodes': 1, 'frames': 40, 'size': [320, 180], 'seed': 4}
OVER = 'observation.images.overhead_d435'


@pytest.fixture(scope='module')
def ur5e():
    return sample_project('ur5e')


@pytest.fixture(scope='module')
def made(ur5e, tmp_path_factory):
    """Datasets by case name, made on first use."""
    from render_support import require
    require()                               # renders MuJoCo scenes
    base = tmp_path_factory.mktemp('s2r')
    out = {}

    def get(name, gaps):
        if name not in out:
            out[name] = IJ.make_gap_dataset(ur5e, str(base / name), gaps, GEN)['root']
        return out[name]
    return get


# ---------------------------------------------------------------------- measures
def test_emd_and_histograms():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 200, (40, 50, 3)).astype(np.uint8)
    b = np.clip(a.astype(int) + 7, 0, 255).astype(np.uint8)
    assert MS.emd_channels([a], [a]) == [0.0, 0.0, 0.0]
    assert np.allclose(MS.emd_channels([a], [b]), 7.0)              # an offset without clipping: exactly the offset
    from scipy.stats import wasserstein_distance
    h1, h2 = rng.integers(0, 9, 256), rng.integers(0, 9, 256)
    v = np.arange(256.0)
    assert abs(MS.emd_1d(h1, h2) - wasserstein_distance(v, v, h1, h2)) < 1e-12
    assert MS.emd_1d(np.zeros(256), h2) is None


def test_affine_fit_is_unbiased_by_misalignment_and_blur():
    rng = np.random.default_rng(1)
    import cv2
    tex = cv2.resize(rng.uniform(20, 200, (30, 40, 3)).astype(np.float32), (320, 240),
                     interpolation=cv2.INTER_NEAREST)
    real = np.clip(1.2 * np.roll(tex, 9, axis=1) + 10, 0, 255)        # shifted 9 px: not aligned
    g, o = MS.fit_affine([tex], [real])
    assert np.allclose(g, 1.2, atol=0.03) and np.allclose(o, 10, atol=3), (g, o)
    out = MS.apply_affine(tex, g, o)
    assert out.max() <= 255 and out.min() >= 0


def test_noise_sharpness_and_blur():
    rng = np.random.default_rng(2)
    y, x = np.mgrid[0:180, 0:320].astype(np.float32)
    base = 120 + 50 * np.sin(x / 15.0) * np.cos(y / 18.0)
    for s in (1.0, 3.0):
        assert abs(MS.noise_sigma(base + rng.normal(0, s, base.shape)) - s) / s < 0.05
    noisy = base + rng.normal(0, 2.0, base.shape)
    # the noise share is taken off: the noisy image's sharpness is near the clean one's
    assert abs(MS.sharpness(noisy, MS.noise_sigma(noisy)) - MS.sharpness(base)) < 0.25 * MS.sharpness(noisy)
    assert MS.blur_to_match([base], MS.sharpness(base) * 2) == 0.0      # already no sharper
    s = MS.blur_to_match([base], MS.sharpness(MS.blur(base, 2.0)))
    assert abs(s - 2.0) < 0.02


def test_mmd_and_shapley():
    rng = np.random.default_rng(3)
    X, Y = rng.normal(0, 1, (30, 5)), rng.normal(0, 1, (30, 5))
    Z = rng.normal(1.5, 1, (30, 5))
    same, diff = MS.mmd_test(X, Y, permutations=100), MS.mmd_test(X, Z, permutations=100)
    assert diff['mmd2'] > same['mmd2'] and diff['p_value'] < 0.05 and same['p_value'] > 0.05
    assert 'RBF' in diff['kernel'] and MS.mmd_test(X[:1], Y)['mmd2'] is None
    with pytest.raises(ValueError):
        MS.mmd2_unbiased(X[:1], Y, 1.0)
    # the closed form: zero for identical distributions, positive otherwise
    assert abs(MS.mmd2_gaussians(np.zeros(3), np.eye(3), np.zeros(3), np.eye(3), 1.0)) < 1e-14
    assert MS.mmd2_gaussians(np.zeros(3), np.eye(3), np.ones(3), np.eye(3), 1.0) > 0
    cost = {frozenset(): 10.0, frozenset('a'): 4.0, frozenset('b'): 7.0, frozenset('ab'): 2.0}
    phi = MS.shapley('ab', lambda S: cost[frozenset(S)])
    assert abs(phi['a'] + phi['b'] - 8.0) < 1e-12 and abs(phi['a'] - 0.5 * (6 + 5)) < 1e-12


def test_depth_gap_on_arrays():
    pred = np.full((60, 80), 1.0)
    meas = pred + 0.01
    meas[:12] = 0.0                                                    # a fifth without depth
    robot = np.zeros_like(pred, bool)
    robot[20:40, 20:60] = True
    d = MS.depth_gap(pred, meas, robot, z_min=0.2, z_max=3.0)
    assert d['predicted_valid_pct'] == 100.0 and abs(d['measured_valid_pct'] - 80.0) < 1e-9
    assert abs(d['robot_error_mm']['median'] - 10.0) < 1e-6 and d['robot_px'] == 800
    with pytest.raises(ValueError):
        MS.depth_gap(pred, meas[:10], robot)


def test_settings_and_refusals(ur5e, tmp_path):
    st = G.settings_of(ur5e)
    assert st['frames'] == 8 and st['lag_step_s'] == 0.01 and 'gap_good_dn' in st['defaults']
    assert st['mount_settings']['max_frames'] == 12
    for bad, words in (({'colour': 1}, 'no setting colour'), ({'frames': 'x'}, 'must be a number'),
                       ({'frames': 2}, 'at least 3'), ({'lag_step_s': 0}, 'above zero'),
                       ({'appearance': [1]}, 'must be a dictionary')):
        with pytest.raises(G.GapError) as e:
            G.settings_of(ur5e, bad)
        assert words in str(e.value) and 'Traceback' not in str(e.value)
    with pytest.raises(G.GapError) as e:
        G.analyse(ur5e, str(tmp_path / 'nothing'))
    assert 'no dataset' in str(e.value).lower()
    with pytest.raises(jobs.StudyError) as e:
        IJ.make_gap_dataset(ur5e, str(tmp_path / 'x'), {'smudge': 1})
    assert 'no gap called smudge' in str(e.value)


# ---------------------------------------------------------------------- recordings with known gaps
def test_clean_recording_is_close(ur5e, made):
    root = made('clean', {})
    doc = G.analyse(ur5e, root)
    json.dumps(doc, allow_nan=False)
    assert doc['summary']['status'] == 'good' and doc['summary']['gap_dn'] < 4.0, doc['summary']
    assert {c['key'] for c in doc['cameras']} == {OVER, 'observation.images.wrist_d405'}
    assert not any(c.get('apply') for c in doc['causes']) and doc['apply'] is None
    for c in doc['cameras']:
        assert abs(c['latency']['latency_s']) < 0.005 and c['latency']['observable']
        assert c['thumbs'] and c['thumbs'][0]['real'] and c['mmd']['not_fid']
        assert c['depth']['measured'] is None and 'no depth stream' in c['depth']['why']
    over = next(c for c in doc['cameras'] if c['key'] == OVER)
    assert over['silhouette']['iou_after'] > 0.85 and 'background difference' in over['silhouette']['method']
    wrist = next(c for c in doc['cameras'] if c['key'] != OVER)
    assert wrist['silhouette']['iou_after'] is None and 'hand-eye' in wrist['mount']['why']
    assert 'not computed' in doc['fid']['why']


@pytest.mark.parametrize('name, gaps, factor', CS.CASES, ids=[c[0] for c in CS.CASES])
def test_known_gap_recovered_and_ranked_first(ur5e, made, name, gaps, factor):
    root = made(name, dict(gaps, camera=OVER))
    doc = G.analyse(ur5e, root, {'cameras': ['Overhead D435'], 'mount': name == 'mount'})
    cam = doc['cameras'][0]
    for k, (err, tol) in CS.recovery_errors(cam, name).items():
        assert err <= tol, (k, err, tol)
    assert CS.top_factor(cam) == factor
    top = doc['causes'][0]
    assert top['factor'] == factor and top['apply'] and top['page'] == G.PAGES[factor][0]
    assert doc['summary']['top']['factor'] == factor and doc['apply'] == top['apply']
    assert abs(sum(cam['gap']['shapley_dn'].values()) - (cam['gap']['none_dn'] - cam['gap']['all_dn'])) < 1e-6
    with IJ_truth(root) as t:
        assert OVER in t['sim2real_gaps']['injected']


class IJ_truth:
    def __init__(self, root):
        self.root = root

    def __enter__(self):
        from dexvantage.core.datasets import open_dataset
        self.ds = open_dataset(self.root)
        return self.ds.truth()

    def __exit__(self, *exc):
        self.ds.close()


def test_apply_closes_the_gap(ur5e, made):
    root = made('brightness', {'brightness_dn': 20.0, 'camera': OVER})
    p = ur5e.snapshot()
    st = {'cameras': ['Overhead D435'], 'mount': False}
    before = G.analyse(p, root, st)
    ch = before['apply']
    assert ch['kind'] == 'appearance' and G.sections_of(ch) == ['analysis']
    assert 'camera response' in G.describe_change(p, ch)
    G.apply_change(p, ch)
    assert p.analysis.studies['sim2real']['appearance'][ch['sensor_uid']]['gain']
    after = G.analyse(p, root, st)
    assert after['summary']['gap_dn'] < 0.5 * before['summary']['gap_dn'], (before['summary'], after['summary'])
    assert after['apply'] is None
    # latency: the measured value goes into the time-sync settings, stamp minus exposure
    q = ur5e.snapshot()
    uid = q.variant.sensors[0].uid
    G.apply_change(q, {'kind': 'latency', 'sensor_uid': uid, 'value_s': -0.04, 'source': 'test'})
    from dexvantage.core.studies import time_sync as TS
    assert TS.settings_of(q)['latency_s'][uid]['value'] == -0.04
    # a locked sensor's mount is refused
    q.variant.sensors[0].locked = True
    with pytest.raises(G.GapError) as e:
        G.apply_change(q, {'kind': 'mount', 'sensor_uid': uid, 'pose': q.variant.sensors[0].mount.pose.to_dict()})
    assert 'locked' in str(e.value)
    with pytest.raises(G.GapError):
        G.apply_change(q, {})


def test_lighting_rule(ur5e):
    def cam(uid, gain, red):
        return {'sensor_uid': uid, 'photometric': {'gain': [gain] * 3},
                'causes': [{'factor': 'exposure', 'camera': f'observation.images.{uid}', 'sensor': uid,
                            'sensor_uid': uid, 'reduction_dn': red, 'share_pct': 80.0, 'title': 'Exposure',
                            'apply': {'kind': 'appearance', 'sensor_uid': uid, 'set': {'gain': [gain] * 3}}}]}
    st = G.settings_of(ur5e)
    p = ur5e.snapshot()
    rows = G._rank(p, [cam('a', 1.30, 20.0), cam('b', 1.25, 15.0)], st)
    assert rows[0]['factor'] == 'lighting' and rows[0]['apply']['kind'] == 'appearance_many'   # no lights to scale
    from dexvantage.core import lights as LT
    p.workspace.lights = [LT.new_light('panel', [0.0, 0.0, 2.0])]
    p.workspace.lights[0]['dim'] = 0.5
    rows = G._rank(p, [cam('a', 1.30, 20.0), cam('b', 1.25, 15.0)], st)
    assert rows[0]['apply'] == {'kind': 'lights', 'scale': pytest.approx(1.275)}
    assert G.apply_change(p, rows[0]['apply']) == ['workspace'] and p.workspace.lights[0]['dim'] == pytest.approx(
        0.6375)
    rows = G._rank(p, [cam('a', 1.6, 20.0), cam('b', 1.0, 15.0)], st)          # different gains: per camera
    assert [r['factor'] for r in rows] == ['exposure', 'exposure']


def test_depth_through_a_decoded_depth_stream(ur5e, made):
    """A reader that decodes depth (here: the clean recording with the rendered depth + 10 mm and a fifth missing)
    gives the measured valid share and the error on the robot."""
    import dataclasses
    from dexvantage.core.datasets import open_dataset
    from dexvantage.core.datasets.episode import CameraStream, Dataset
    from dexvantage.core.sim.render import Renderer
    root = made('clean', {})
    with open_dataset(root) as ds:
        ep = ds.load(0)
    s = next(x for x in ur5e.variant.sensors if x.name == 'Overhead D435')
    cs = ep.cameras[OVER]
    from dexvantage.core.calib import render_calib as RC
    ruid, model, pairs, _ = RC.robot_map(ur5e, ep)
    Q = RC.q_at(ep, model, pairs, np.asarray(cs.timestamps))
    with Renderer(ur5e) as R:
        cam = R.camera(s.uid).model.scaled(320, 180)
        imgs = R.render([{'sensor_uid': s.uid, 'q': {ruid: q}, 'outputs': ('depth',), 'camera_model': cam}
                         for q in Q])
    depth = []
    for o in imgs:
        d = o['depth'].astype(np.float32) + 0.01
        d[(o['depth'] <= 0) | (d > 60.0)] = 0.0                        # no hit, or beyond a 16-bit millimetre value
        d[:36] = 0.0
        depth.append((d * 1000).astype(np.uint16))                    # millimetres, as a 16-bit depth image

    def dec(idx):
        for i in (range(len(depth)) if idx is None else idx):
            yield int(i), depth[int(i)]
    dstream = CameraStream('observation.depth.overhead', 320, 180, cs.fps, cs.timestamps.copy(),
                           meta={'depth_of': OVER, 'depth_scale_m': 0.001}, _decode=dec)
    ep2 = dataclasses.replace(ep, cameras={OVER: cs, dstream.key: dstream})

    class One(Dataset):
        format, version, path = 'test', '', root

        def episodes(self):
            from dexvantage.core.datasets.episode import EpisodeRef
            return [EpisodeRef(0, 'e')]

        def load(self, index):
            return ep2
    doc = G.analyse(ur5e, One(), {'cameras': ['Overhead D435'], 'mount': False})
    d = doc['cameras'][0]['depth']
    assert d['predicted_valid_pct'] > 30 and d['measured_valid_pct'] < d['predicted_valid_pct']
    assert abs(d['robot_error_mm']['median'] - 10.0) < 1.5, d['robot_error_mm']


# ---------------------------------------------------------------------- the study, metrics, batch, API, tool
def test_study_metrics_batch_api_and_tool(ur5e, made, tmp_path):
    root = made('latency', {'latency_s': 0.04, 'camera': OVER})
    snap = ur5e.snapshot()
    snap.analysis.studies['sim2real'] = {'dataset': root, 'mount': False}
    doc = jobs.run_study('sim2real', snap)
    assert doc['study'] == 'sim2real' and doc['schema'] == 'dexvantage.sim2real/1'
    assert jobs.STUDIES['sim2real'].batch == 'sim2real' and 'sim2real' in jobs.DATASET_STUDIES
    from dexvantage.core import metrics as M
    M.load_metrics()
    gap = M.read(M.METRICS['sim2real.image_gap_dn'], doc)
    assert gap['value'] == pytest.approx(doc['summary']['gap_dn'])
    assert M.read(M.METRICS['sim2real.latency_ms'], doc)['value'] == pytest.approx(40, abs=5)
    assert M.read(M.METRICS['sim2real.mount_offset_mm'], doc)['value'] is None
    # batch: --sim2real OUT.json --dataset PATH
    out = tmp_path / 's2r.json'
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    r = subprocess.run([sys.executable, '-m', 'dexvantage', '--sample', 'ur5e', '--sim2real', str(out), '--dataset',
                        root], capture_output=True, text=True, cwd=ROOT, env=env, timeout=600)
    assert r.returncode == 0, r.stderr
    res = json.loads(out.read_text(encoding='utf-8'))
    assert res['study'] == 'sim2real' and 'provenance' in res and res['causes'][0]['factor'] == 'latency'
    r = subprocess.run([sys.executable, '-m', 'dexvantage', '--sample', 'ur5e', '--sim2real', str(out), '--dataset',
                        str(tmp_path / 'nothing')], capture_output=True, text=True, cwd=ROOT, env=env, timeout=300)
    assert r.returncode == 2 and 'Traceback' not in r.stderr and 'not there' in r.stderr
    # the API (and no Qt on its path)
    import dexvantage as dv
    rig = dv.sample('ur5e')
    res = rig.sim2real(root, mount=False)
    assert res.headline['metric'] == 'sim2real.image_gap_dn' and res['causes'][0]['factor'] == 'latency'
    new = rig.apply_sim2real(res['apply'])
    assert new is not rig and new.project.analysis.studies['time_sync']['latency_s']
    again = new.sim2real(root, mount=False)                          # the plan now renders at stamp + 40 ms
    over = next(c for c in again['cameras'] if c['key'] == OVER)
    assert over['latency']['planned_s'] == pytest.approx(0.04, abs=0.005) and abs(over['latency']['latency_s']) < 0.005
    assert not any(c['factor'] == 'latency' and c.get('apply') for c in again['causes'])
    assert 'time_sync' not in (rig.project.analysis.studies or {}) or not \
        rig.project.analysis.studies['time_sync'].get('latency_s')
    q = subprocess.run([sys.executable, '-c', 'import sys, dexvantage as dv; import dexvantage.core.sim2real.gap, '
                        'dexvantage.core.studies.sim2real; print("PyQt6" in sys.modules)'], capture_output=True,
                       text=True, cwd=ROOT, timeout=120)
    assert q.stdout.strip() == 'False', q.stdout + q.stderr
    from dexvantage.assistant import tools as T
    assert 'get_sim2real' in T.TOOLS and 'sim2real' in T.LONG


def test_checks_registered():
    from dexvantage.validation import references as RF
    from dexvantage.validation import suite as S
    S.load_checks()
    ids = {c.id: c for c in S.CHECKS if c.model == 'sim2real'}
    assert set(ids) >= {'sim2real.emd_vs_scipy', 'sim2real.mmd_hand_case', 'sim2real.mmd_gaussians',
                        'sim2real.noise_known', 'sim2real.blur_known', 'sim2real.shapley_brute_force',
                        'sim2real.injected_recovery'}
    assert all(r in RF.REFERENCES for c in ids.values() for r in c.refs)
    assert ids['sim2real.injected_recovery'].cost == 'study'
    ctx = S.Ctx(None, None, None, S.ValidationSettings(quick=True))
    try:
        for cid in ('sim2real.emd_vs_scipy', 'sim2real.mmd_hand_case', 'sim2real.mmd_gaussians',
                    'sim2real.noise_known', 'sim2real.blur_known', 'sim2real.shapley_brute_force'):
            c = S.run_one(ids[cid], ctx)
            assert c.passed, (cid, c.error, c.tolerance, c.exception)
    finally:
        ctx.close()


# ---------------------------------------------------------------------- the page
def test_page(app, win, ur5e, made, tmp_path):
    root = made('brightness', {'brightness_dn': 20.0, 'camera': OVER})
    assert 'sim2real' in win.pages
    assert win.open_sample('ur5e')
    page = win.pages['sim2real']
    win.open_page('sim2real')
    pump(app, 0.05)
    assert page.head.status.text.text() == 'not run yet'
    page.choose_dataset(root)
    assert win.store.project.analysis.studies['sim2real']['dataset'] == root
    assert page.dataset_label.text() == os.path.basename(root)
    snap = win.store.project.snapshot()
    snap.analysis.studies['sim2real'] = {'dataset': root, 'mount': False}
    doc = jobs.run_study('sim2real', snap)
    page._finished('sim2real', doc)
    pump(app, 0.05)
    assert page.head.status.text.text() == doc['summary']['word'] and 'DN' in page.head.figure.text() + \
        page.head.unit.text()
    assert page.causes.topLevelItemCount() >= 2 and page.cam_box.count() == 2
    assert page.frame_box.count() >= 1 and not page.pics['real'].pixmap().isNull()
    page.overlay.setChecked(True)
    page.mode_box.setCurrentIndex(1)
    page._show_frames()
    assert not page.pics['diff'].pixmap().isNull() and page.figs.topLevelItemCount() >= 4
    assert page.b_apply.isEnabled()
    page.preview()
    n0 = win.store.undo.count()
    assert page.apply()
    assert win.store.undo.count() == n0 + 1
    ap = win.store.project.analysis.studies['sim2real'].get('appearance') or {}
    assert doc['apply']['sensor_uid'] in ap
    win.store.undo.undo()
    assert not (win.store.project.analysis.studies['sim2real'].get('appearance') or {})
    page._choose(0)
    page.open_linked_page()
    exp = page.export(str(tmp_path / 's2r.json'))
    assert 'provenance' in json.loads(open(exp, encoding='utf-8').read())
    ctx = page.page_context()
    assert ctx['study'] == 'sim2real' and ctx['headline']['word']
    page._replaced()
    pump(app, 0.1)
    assert page.causes.topLevelItemCount() == 0 and math.isfinite(1.0)
