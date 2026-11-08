"""Splat views (core/sim/splat_views.py, core/studies/splat_views.py, batch_splat.py, api_splat.py; spec §9.3,
Phase 12): the settings and their refusals, the fit from a MuJoCo demonstration and from a recorded LeRobot episode,
the held-out quality, novel camera pose replay (the video read back), perturbation augmentation (a LeRobot dataset
read back, every render marked synthetic), LPIPS against the lpips package's own model, and the batch, API, assistant
and self-test entry points.

The fits run on the CPU at a few dozen pixels with a few hundred Gaussians and a few steps (seconds); the recordings
are DexVantage's synthetic SO-100 episodes (core/sim/episodes.py), so MuJoCo must render offscreen (tests marked
mujoco_render skip with the reason otherwise). LPIPS needs AlexNet's weights in the cache, which a test never
downloads: the LPIPS comparison skips with the reason when they are absent.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from dexvantage.core import jobs
from dexvantage.core.jobs import StudyError
from dexvantage.core.robot.samples import sample_project
from dexvantage.core.sim import splat_views as SV

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NO_TORCH = 'PyTorch is not installed (the splat extra): splat views are not tested'
# tiny CPU settings: a fit of seconds (the figures are not the point here, the plumbing is)
TINY = {'device': 'cpu', 'iterations': 12, 'max_gaussians': 600, 'demo_views': 12, 'demo_width': 40,
        'demo_height': 30, 'demo_q_states': 2, 'holdout_q': 2, 'preview_frames': 2, 'thumb_width': 64,
        'views_per_camera': 10, 'width': 48, 'min_views': 6, 'lpips': 'off'}


def _torch():
    return pytest.importorskip('torch', reason=NO_TORCH)


@pytest.fixture(scope='module')
def so100():
    return sample_project('so100')


@pytest.fixture(scope='module')
def dataset(so100, tmp_path_factory):
    from render_support import require
    require()
    from dexvantage.core.sim import episodes as EPS
    root = str(tmp_path_factory.mktemp('splat') / 'so100_episodes')
    EPS.generate(so100, root, {'episodes': 1, 'frames': 36, 'max_width': 96, 'faults': [], 'codec': 'h264'})
    return root


@pytest.fixture(scope='module')
def demo_doc(so100):
    _torch()
    from render_support import require
    require()
    return SV.analyse(so100, dict(TINY, source='demo', target_offset_m=[0.0, 0.0, 0.04]))


# ====================================================================== settings and refusals
def test_settings_are_labelled_defaults_and_bad_ones_are_refused(so100):
    st = SV.settings_of(so100)
    assert st['source'] == 'auto' and st['iterations'] == 1500 and 'iterations' in st['defaults']
    st = SV.settings_of(so100, {'iterations': 40})
    assert st['iterations'] == 40 and 'iterations' not in st['defaults']
    for bad, words in (({'source': 'photo'}, 'source must be one of'), ({'augment_n': 0}, '1 to 100'),
                       ({'device': 'tpu'}, 'auto, cuda or cpu'), ({'width': 'wide'}, 'must be a number'),
                       ({'target_offset_m': [0, 1]}, 'three numbers'), ({'lpips': 'maybe'}, 'auto or off')):
        with pytest.raises(StudyError, match=words):
            SV.settings_of(so100, bad)


def test_without_torch_it_says_the_splat_extra_is_not_installed(so100, monkeypatch):
    import importlib.util
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, 'find_spec', lambda n, *a: None if n == 'torch' else real(n, *a))
    why = SV.requirement()
    assert why and 'the splat extra is not installed' in why and 'pip install "dexvantage[splat]"' in why
    with pytest.raises(StudyError, match='splat extra is not installed'):
        SV.analyse(so100, {'source': 'demo'})
    from dexvantage import selftest as ST
    status, msg = ST._check_splat({})
    assert status == ST.WARN and msg.startswith('splat extra not installed')


def test_in_the_downloadable_app_it_says_how_to_get_the_splat_extra(monkeypatch):
    """The bundle leaves PyTorch out (extras.EXTRAS['splat']): pip cannot add it there, so the sentence says to install
    from source with the extra and PyTorch's CUDA index, for the page, the study and the self-test line."""
    import importlib.util
    import sys
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, 'find_spec', lambda n, *a: None if n == 'torch' else real(n, *a))
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    why = SV.requirement()
    assert why.startswith('This DexVantage download leaves out PyTorch') and 'pip install -e ".[splat]"' in why
    assert 'https://download.pytorch.org/whl/cu130' in why and '\n' not in why
    from dexvantage import selftest as ST
    status, msg = ST._check_splat({})
    assert status == ST.WARN and msg.startswith('splat extra not installed: this DexVantage download leaves out')


def test_without_cuda_auto_refuses_and_cpu_is_allowed(so100, monkeypatch):
    _torch()
    from dexvantage.core import splat_raster as SR
    monkeypatch.setattr(SR, 'cuda_ok', lambda: False)
    with pytest.raises(StudyError, match='no CUDA device'):
        SV._device(SV.settings_of(so100))
    assert SV._device(SV.settings_of(so100, {'device': 'cpu'})) == 'cpu'


def test_distinct_views_counts_poses_apart():
    T = np.eye(4)
    near = np.eye(4)
    near[0, 3] = 0.005
    far = np.eye(4)
    far[0, 3] = 0.05
    from dexvantage.core.frames import rot_z
    turned = np.eye(4)
    turned[:3, :3] = rot_z(np.radians(10))
    assert SV.distinct_views([T, near, T, near], 0.02, 5.0) == 1
    assert SV.distinct_views([T, far, turned, near], 0.02, 5.0) == 3


def test_the_study_is_registered_with_its_batch_flag_and_settings(so100):
    from dexvantage.core import study_settings as SST
    jobs.load_studies()
    info = jobs.STUDIES['splat_views']
    assert info.batch == 'splat-views' and info.needs_robot and not info.quick
    assert 'splat_views' in jobs.DATASET_STUDIES
    assert set(SST.known('splat_views')) == set(SV.DEFAULTS)
    with pytest.raises(StudyError, match='no setting called iters'):
        SST.check('splat_views', ['iters'])


def test_the_assistant_and_mcp_have_its_tool():
    from dexvantage.assistant import page_guide as G
    from dexvantage.assistant import tools as T
    assert 'get_splat_views' in T.TOOLS and 'splat_views' in T.LONG and G.GUIDES['splat'].purpose


# ====================================================================== the demonstration fit
@pytest.mark.mujoco_render
def test_demo_fit_reports_held_out_quality_and_marks_everything_synthetic(demo_doc):
    d = demo_doc
    h = d['headline']
    assert d['synthetic'] and d['synthetic_capture'] and d['source']['kind'] == 'demo'
    assert 'demonstration' in d['source']['label'] and 'synthetic capture' in h['word']
    q = d['quality']
    assert q['measured'] and q['held_out_views']['views'] >= 1 and q['held_out_q']['views'] == 2
    assert h['value'] == q['held_out_views']['psnr_db'] and 5 < h['value'] < 60
    assert 0 < q['held_out_views']['ssim'] <= 1 and q['held_out_views']['lpips'] is None
    assert 'lpips is off' in q['lpips_note']
    assert h['status'] in ('good', 'warning', 'serious', 'critical') and h['text'].startswith('Held-out views: PSNR')
    pv = d['preview']
    assert pv['synthetic'] and len(pv['frames']) == 2 and 'moved by (+0, +0, +40) mm' in pv['proposal']
    f0 = pv['frames'][0]
    assert f0['synthetic'] and f0['current'] and f0['proposed'] and f0['current'] != f0['proposed']
    assert d['rig']['gaussians'] > 0 and d['rig']['device'] == 'cpu' and d['rig']['deterministic'] is True
    json.dumps(d, allow_nan=False)


@pytest.mark.mujoco_render
def test_a_second_run_reuses_the_cached_rig(so100, demo_doc):
    d = SV.analyse(so100, dict(TINY, source='demo', target_offset_m=[0.0, 0.0, 0.04]))
    assert d['rig']['cached'] and d['rig']['file'] == demo_doc['rig']['file']
    assert d['quality']['held_out_views']['psnr_db'] == pytest.approx(demo_doc['quality']['held_out_views']['psnr_db'],
                                                                      abs=1e-6)
    assert os.path.isfile(SV.cache_dir('splat', d['rig']['file']))


@pytest.mark.mujoco_render
def test_the_target_camera_pose_is_the_renderers(so100):
    from dexvantage.core.scenegeom import sensor_world_pose
    from dexvantage.core.sim.render import Renderer
    st = SV.settings_of(so100)
    s, im = SV.target_sensor(so100, st)
    assert s.mount.on_link                                           # the wrist camera by default
    q = {so100.robots[0].uid: np.array([0.2, -0.3, 0.4, 0.1, -0.2, 0.0][:len(so100.robots[0].joints)])}
    with Renderer(so100) as R:
        want = sensor_world_pose(so100, s, q).matrix() @ R.camera(s.uid).T_sensor_optical
    got = SV.target_pose_fn(so100, st, s, im, False)(q)
    assert np.abs(got - want).max() < 1e-9
    moved = SV.target_pose_fn(so100, SV.settings_of(so100, {'target_offset_m': [0.0, 0.0, 0.05]}), s, im, True)(q)
    body = sensor_world_pose(so100, s, q).matrix()
    assert np.linalg.norm(moved[:3, 3] - got[:3, 3]) == pytest.approx(0.05, abs=1e-9)
    assert np.allclose(np.linalg.inv(body)[:3, :3] @ (moved[:3, 3] - got[:3, 3]), [0, 0, 0.05], atol=1e-9)


# ====================================================================== a recorded episode
@pytest.mark.mujoco_render
def test_episode_fit_replay_and_augment(so100, dataset, tmp_path):
    _torch()
    st = dict(TINY, source='episode', dataset=dataset, target_offset_m=[0.0, 0.01, 0.02], replay_max_frames=12,
              replay_scale=0.25, augment_n=2, augment_frame_scale=0.5, codec='h264')
    d = SV.analyse(so100, st)
    assert d['source']['kind'] == 'episode' and d['source']['synthetic'] and 'synthetic dataset' in d['source']['label']
    assert d['quality']['held_out_q']['views'] >= 1 and 'farthest in joint space' in d['quality']['held_out_q']['how']
    assert d['preview']['frames'][0].get('recorded')                  # the episode records the wrist camera
    # replay: a video read back after writing, per-frame images, a manifest of SHA-256 digests
    r = SV.replay(so100, str(tmp_path), st)
    root = r['root']
    assert os.path.basename(root) == 'splat_replay_synthetic' and r['synthetic'] and r['frames'] == 12
    assert r['video']['read_back'] and r['video']['frames'] == 12 and 'synthetic' in r['video']['file']
    pngs = sorted(os.listdir(os.path.join(root, 'frames')))
    assert len(pngs) == 12 and all(n.endswith('_synthetic.png') for n in pngs)
    import av
    with av.open(os.path.join(root, r['video']['file'])) as c:
        assert sum(1 for _ in c.decode(video=0)) == 12
    man = json.load(open(os.path.join(root, SV.MANIFEST), encoding='utf-8'))
    assert man['synthetic'] and 'provenance' in man and len(man['files']) == 13
    for f in man['files']:
        assert SV._sha256(os.path.join(root, f['path'])) == f['sha256']
    with pytest.raises(StudyError, match='not empty'):
        SV.replay(so100, str(tmp_path), st)
    # augmentation: a LeRobot dataset of perturbed copies, read back, every render marked synthetic
    a = SV.augment(so100, str(tmp_path), st)
    root = a['root']
    assert os.path.basename(root) == 'splat_augment_synthetic' and a['synthetic'] and a['episodes'] == 2
    from dexvantage.core.datasets.episode import open_dataset
    ds = open_dataset(root)
    try:
        assert [e.index for e in ds.episodes()] == [0, 1]
        e = ds.load(1)
        assert set(e.cameras) == set(a['cameras'])
        cam = next(iter(e.cameras.values()))
        frames = [img for _i, img in cam.frames([0])]
        assert frames[0].shape[2] == 3
    finally:
        ds.close()
    info = json.load(open(os.path.join(root, 'meta', 'info.json'), encoding='utf-8'))
    assert info['dexvantage']['synthetic'] and info['dexvantage']['manifest'] == SV.MANIFEST
    vids = [f for f in info['features'].values() if f.get('dtype') == 'video']
    assert vids and all(f['info']['dexvantage.synthetic'] for f in vids)
    man = json.load(open(os.path.join(root, SV.MANIFEST), encoding='utf-8'))
    draws = man['perturbations']['draws']
    assert len(draws) == 2 and draws[0][0]['d'] != draws[1][0]['d'] and 'provenance' in man
    # the draws are core/tolerance's for the same seed (the §8.5 perturbation export's)
    from dexvantage.core import tolerance as TL
    A = TL.perturbed_extrinsics(so100, 2, 0, 1.0, 'uniform')
    j = [u['uid'] for u in A['sensors']].index(draws[0][0]['sensor_uid'])
    assert np.allclose(draws[1][0]['d'], A['d'][1, j], atol=1e-8)


@pytest.mark.mujoco_render
def test_an_episode_with_one_static_camera_has_too_few_views(so100, dataset):
    _torch()
    static = next(s.uid for s in so100.variant.sensors if not s.mount.on_link)
    with pytest.raises(StudyError, match='too few distinct views'):
        SV.analyse(so100, dict(TINY, source='episode', dataset=dataset, cameras=[static]))


def test_episode_source_without_a_dataset_says_so(so100):
    _torch()
    with pytest.raises(StudyError, match='Choose the recorded LeRobot dataset'):
        SV.get_rig(so100, dict(TINY, source='episode'))
    with pytest.raises(StudyError, match='not there'):
        SV.get_rig(so100, dict(TINY, source='episode', dataset=os.path.join(ROOT, 'no-such-dataset')))


# ====================================================================== LPIPS
def test_lpips_is_never_downloaded_unasked(so100, monkeypatch, tmp_path):
    _torch()
    pytest.importorskip('lpips', reason='the lpips package (the splat extra) is not installed')
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path))
    assert SV.lpips_state() == 'no weights'
    import torch.hub

    def no(*a, **k):
        raise AssertionError('a test must not download')
    monkeypatch.setattr(torch.hub, 'download_url_to_file', no)
    lp, note = SV._lpips_for(SV.settings_of(so100), 'cpu')
    assert lp is None and 'not in the cache' in note and '244 MB' in note


def test_lpips_matches_the_lpips_package(monkeypatch):
    torch = _torch()
    lpips = pytest.importorskip('lpips', reason='the lpips package (the splat extra) is not installed')
    given = os.environ.get('DEXVANTAGE_TEST_LPIPS_WEIGHTS')       # a copy already on disk (tests never download)
    if SV.lpips_state() != 'ready' and given and os.path.isfile(given):
        os.makedirs(os.path.dirname(SV.lpips_weights_path()), exist_ok=True)
        os.symlink(os.path.abspath(given), SV.lpips_weights_path())
    if SV.lpips_state() != 'ready':
        pytest.skip('AlexNet\'s weights for LPIPS are not in the cache (tests never download them: '
                    'dv.splat.fetch_lpips_weights() gets them, or DEXVANTAGE_TEST_LPIPS_WEIGHTS names a copy)')
    import warnings
    rng = np.random.default_rng(0)
    a = rng.random((48, 64, 3))
    b = np.clip(a + rng.normal(0, 0.1, a.shape), 0, 1)
    mine = SV._Lpips(torch.device('cpu'))
    monkeypatch.setenv('TORCH_HOME', SV.cache_dir('torch'))   # the package's own loader finds the same file
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref = lpips.LPIPS(net='alex', verbose=False)
    x, y = (torch.tensor(i, dtype=torch.float32).permute(2, 0, 1)[None] * 2 - 1 for i in (a, b))
    assert mine(a, b) == pytest.approx(float(ref(x, y)), abs=1e-6) and mine(a, a) == pytest.approx(0.0, abs=1e-7)


# ====================================================================== batch, API, validation
@pytest.mark.mujoco_render
def test_batch_splat_fit_and_its_refusals(tmp_path):
    _torch()
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    out = tmp_path / 'out'
    st = json.dumps(dict(TINY, source='demo'))
    r = subprocess.run([sys.executable, '-m', 'dexvantage', '--sample', 'so100', '--splat', 'fit', '--splat-out',
                        str(out), '--splat-settings', st], capture_output=True, text=True, env=env, cwd=ROOT,
                       timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    doc = json.load(open(out / 'splat_views.json', encoding='utf-8'))
    assert doc['study'] == 'splat_views' and doc['synthetic'] and 'provenance' in doc
    assert (out / 'splat_rig.npz').is_file() and (out / 'splat_rig.json').is_file()
    r = subprocess.run([sys.executable, '-m', 'dexvantage', '--sample', 'so100', '--splat', 'replay', '--splat-out',
                        str(out), '--splat-settings', st], capture_output=True, text=True, env=env, cwd=ROOT,
                       timeout=600)
    assert r.returncode == 2 and 'Replay needs a recorded LeRobot episode' in r.stderr and 'Traceback' not in r.stderr
    r = subprocess.run([sys.executable, '-m', 'dexvantage', '--sample', 'so100', '--splat', 'fit', '--splat-out',
                        str(out), '--splat-settings', '{"augment_n": 0}'], capture_output=True, text=True, env=env,
                       cwd=ROOT, timeout=600)
    assert r.returncode == 2 and '1 to 100' in r.stderr and 'Traceback' not in r.stderr


def test_batch_flags_are_listed():
    import argparse

    from dexvantage import batch
    ap = argparse.ArgumentParser()
    batch.add_arguments(ap)
    a = ap.parse_args(['--splat', 'augment', '--splat-out', 'x', '--splat-views', 'y.json'])
    assert a.splat == 'augment' and a.study__splat_views == 'y.json' and batch.wanted(a)
    with pytest.raises(SystemExit):
        ap.parse_args(['--splat', 'paint'])


@pytest.mark.mujoco_render
def test_the_python_api(so100):
    _torch()
    import dexvantage as dv
    rig = dv.sample('so100')
    res = dv.splat.fit(rig, **dict(TINY, source='demo', target_offset_m=[0.0, 0.0, 0.04]))
    assert res['study'] == 'splat_views' and res.headline['word'] and res['synthetic']
    assert dv.splat.available() is None and dv.splat.settings()['iterations'] == 1500
    srig = dv.splat.rig(rig, **dict(TINY, source='demo', target_offset_m=[0.0, 0.0, 0.04]))
    assert srig.n == res['rig']['gaussians']
    with pytest.raises(dv.DexVantageError, match='no setting called iters'):
        dv.splat.fit(rig, iters=3)


def test_the_validation_checks_of_the_splat_model_pass():
    from dexvantage import validation as VA
    res = VA.run(settings=VA.ValidationSettings(quick=True, models=('splat',)))
    ids = {c.id for c in res.checks} | {d['id'] for d in res.not_run}
    assert {'splat.footprint_closed_form', 'splat.reproject_disparity', 'splat.repose_rigid',
            'splat.ssim_closed_form'} <= ids
    assert all(c.passed for c in res.checks), [(c.id, c.error, c.exception) for c in res.checks if not c.passed]
    from dexvantage.validation import suite as VS
    s = VS.not_run_sentence([{'id': 'splat.raster_torch_cuda', 'model': 'splat', 'title': '', 'module': 'cuda',
                              'extra': 'splat'}])
    assert s == '1 check not run because there is no CUDA device here: 1 needs a CUDA device.'


def test_the_self_test_line_passes_here():
    _torch()
    from dexvantage import selftest as ST
    assert 'splat' in ST.NAMES and 'splat' in ST.BUNDLE_MAY_WARN
    status, msg = ST._check_splat({})
    assert status == ST.OK, msg


@pytest.mark.mujoco_render
def test_the_fit_reports_held_out_scores_of_the_rig_as_fitted(so100):
    """The fit report's held-out scores are those of the fitted rig's own renders: its colours and background are
    not clamped back into [0.01, 0.99] for scoring (that cost 0.1 to 0.3 of SSIM on a black background)."""
    torch = _torch()
    from dexvantage.core import splat_reproject as RP
    from dexvantage.core.sim import splat_rig as SRG
    cap = SRG.synthetic_capture(so100, n_views=16, qs=SRG.random_qs(so100, 2, 0), width=48, height=36)
    rig, rep = SRG.fit_splat_rig(so100, cap, {'iterations': 30, 'device': 'cpu', 'max_gaussians': 800})
    held = [v for k, v in enumerate(cap.views) if SRG.held_out(k, 8)]
    ps, ss = [], []
    for v in held:
        pred = np.clip(rig.render(so100, v.camera, v.T_world_optical, v.q, backend='torch',
                                  device=torch.device('cpu'))['rgb'], 0, 1)
        s = RP.image_scores(pred, np.asarray(v.rgb, float) / 255.0)
        ps.append(s['psnr_db'])
        ss.append(s['ssim'])
    assert rep['held_out_after']['psnr_db'] == pytest.approx(np.mean(ps), abs=0.02)
    assert rep['held_out_after']['ssim'] == pytest.approx(np.mean(ss), abs=0.002)
