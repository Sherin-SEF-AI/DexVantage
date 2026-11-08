"""Synthetic LeRobot episodes with ground truth (core/sim/episodes.py): the settings, the plan (motions, timelines and
the seeded faults), the generated datasets read back at v2.1 and v3.0 against the generator and its sidecar, the
faults in the pixels, the refusals, the batch flag, the Simulate command and its dialog, and the validation check.
Small: 2 episodes × 20 rows × 64 × 48 px."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip('pyarrow', reason='pyarrow (the lerobot extra) is not installed')
pytest.importorskip('av', reason='av (the lerobot extra) is not installed')

from conftest import pump  # noqa: E402
from dexvantage.core.datasets import open_dataset  # noqa: E402
from dexvantage.core.robot.samples import sample_project  # noqa: E402
from dexvantage.core.sim import episodes as SE  # noqa: E402
from dexvantage.validation import checks_synth as CS  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SMALL = {'episodes': 2, 'frames': 20, 'size': [64, 48], 'seed': 1}


@pytest.fixture(scope='module')
def so100():
    return sample_project('so100')


@pytest.fixture(scope='module')
def ur5e():
    return sample_project('ur5e')


@pytest.fixture(scope='module', params=['v2.1', 'v3.0'])
def made(request, so100, tmp_path_factory):
    from render_support import require
    require()                               # renders MuJoCo scenes
    root = str(tmp_path_factory.mktemp('synth') / f'so100_{request.param}')
    return SE.generate(so100, root, dict(SMALL, version=request.param), keep_frames=True)


@pytest.fixture(scope='module')
def clean(so100, tmp_path_factory):
    """No pixel noise, so the faults can be compared with fresh renders exactly."""
    from render_support import require
    require()                               # renders MuJoCo scenes
    root = str(tmp_path_factory.mktemp('synth') / 'so100_clean')
    return SE.generate(so100, root, dict(SMALL, noise_dn=0.0), keep_frames=True)


# ---------------------------------------------------------------------- settings
def test_settings_defaults_and_refusals(so100):
    st = SE.settings_of(so100)
    assert st['version'] == 'v3.0' and st['episodes'] == 4 and st['max_width'] == 640
    assert all(st['faults'][k]['enabled'] for k in SE.FAULT_KINDS) and set(st['defaults']) >= {'episodes', 'seed'}
    assert st['faults']['blur']['subframes'] % 2 == 1
    only = SE.settings_of(so100, {'faults': ['drop', 'blur']})['faults']
    assert [k for k in SE.FAULT_KINDS if only[k]['enabled']] == ['drop', 'blur']
    d = SE.settings_of(so100, {'faults': {'drift': {'drift_ppm': 100}, 'jitter': False}})['faults']
    assert d['drift']['drift_ppm'] == 100.0 and d['drift']['enabled'] and not d['jitter']['enabled']
    for bad, words in (({'faults': ['smudge']}, 'no fault called smudge'), ({'size': [63, 48]}, 'even width'),
                       ({'episodes': 0}, 'at least 1'), ({'version': 'v1.6'}, 'one of v2.1, v3.0'),
                       ({'colour': 1}, 'no setting colour'), ({'faults': {'drop': {'cnt': 2}}}, 'no setting cnt')):
        with pytest.raises(SE.SynthError) as e:
            SE.settings_of(so100, bad)
        assert words in str(e.value) and 'Traceback' not in str(e.value)


# ---------------------------------------------------------------------- the plan
def test_plan_grasp_approaches_and_trajectories(so100, ur5e):
    p = SE.plan(so100, SMALL)
    assert [m.motion.source for m in p.episodes] == ['grasp approach'] * 2      # the SO-100 has no trajectory
    m = p.episodes[0].motion
    # minimum-jerk timing: the joints stand still at every waypoint
    for tk in m.T:
        v = (m.q(tk + 1e-4) - m.q(max(tk - 1e-4, 0.0))) / 2e-4
        assert np.max(np.abs(v)) < 1e-3
    assert p.fps == 30 and all(c.fps == 30.0 and 'default' in c.fps_source for c in p.cameras)
    assert p.wrist == 'observation.images.wrist_camera'
    assert {c.key for c in p.cameras} == {'observation.images.wrist_camera', 'observation.images.third_person_camera'}
    assert set(p.gripper_links) == {'Fixed_Jaw', 'Moving_Jaw'}
    q = SE.plan(ur5e, SMALL)
    assert q.episodes[0].motion.source == 'trajectory' and q.episodes[0].motion.name == 'Scripted pick (sample)'
    assert all(c.fps == 30.0 and 'stream' in c.fps_source for c in q.cameras)       # the D405/D435 streams
    ep = q.episodes[0]
    assert ep.q.shape == (20, len(ep.motion.model.joint_names)) and ep.action.shape == ep.q.shape
    # the action is the state one row later (the default lead), held at the end
    np.testing.assert_allclose(ep.action[:-1], ep.q[1:], atol=1e-12)
    with pytest.raises(SE.SynthError, match='no trajectory'):
        SE.plan(so100, dict(SMALL, motion='trajectories'))


def test_plan_is_seeded_and_faults_are_consistent(so100):
    a, b = SE.plan(so100, SMALL), SE.plan(so100, SMALL)
    c = SE.plan(so100, dict(SMALL, seed=2))
    fa = [f for ep in a.episodes for f in ep.faults]
    assert fa == [f for ep in b.episodes for f in ep.faults]
    assert fa != [f for ep in c.episodes for f in ep.faults]
    kinds = {f['kind'] for f in fa}
    assert kinds == set(SE.FAULT_KINDS)
    for ep in a.episodes:
        for key, tl in ep.timelines.items():
            st = tl.stored
            M = len(st)
            stamps = tl.stamp[st]
            assert np.all(np.diff(stamps) > 0) and stamps[0] >= 0
            gaps = {i + 1 for i in np.flatnonzero(np.diff(stamps) >= 1.5 / tl.fps)}
            listed = {i for f in ep.faults if f['camera'] == key and f['kind'] in ('drop', 'usb_reset')
                      for i in f['frames']}
            assert gaps == listed
            for f in ep.faults:
                if f['camera'] == key and f.get('frames'):
                    assert all(0 <= i < M for i in f['frames'])
        # frame faults never share a slot (a duplicate's previous frame included)
        taken = {}
        for f in ep.faults:
            if f['kind'] in ('duplicate', 'blur', 'gripper_hidden', 'drop', 'usb_reset'):
                s = set(f['slots']) | ({k - 1 for k in f['slots']} if f['kind'] == 'duplicate' else set())
                assert not (taken.setdefault(f['camera'], set()) & s), f
                taken[f['camera']] |= s
    # switching one fault off leaves the others where they were
    d = SE.plan(so100, dict(SMALL, faults={'jitter': False}))
    keep = [f for ep in a.episodes for f in ep.faults if f['kind'] in ('desync', 'drift', 'mount_slip')]
    assert keep == [f for ep in d.episodes for f in ep.faults if f['kind'] in ('desync', 'drift', 'mount_slip')]
    # desync, drift and jitter in the timelines
    for ep in a.episodes:
        for f in ep.faults:
            tl = ep.timelines[f['camera']]
            if f['kind'] == 'desync':
                assert np.all(tl.content - tl.t_slot >= f['offset_s'] - 1e-12)
            if f['kind'] == 'jitter':
                j = tl.stamp - tl.t_slot
                assert np.any(j != 0) and np.max(np.abs(j)) <= 0.2 / tl.fps + 1e-12
            if f['kind'] == 'mount_slip':
                assert ep.index >= f['from_episode'] == 1 and tl.delta is not None
    assert all(ep.timelines[k].delta is None for ep in a.episodes[:1] for k in ep.timelines)


# ---------------------------------------------------------------------- generated and read back
def test_round_trip_against_generator_and_sidecar(made):
    bad, detail = CS.roundtrip(made)
    assert bad == [], bad
    assert detail['frames_compared'] == made['camera_frames'] and detail['worst_psnr_db'] >= 30
    assert made['codec'] in ('AV1 (libsvtav1)', 'H.264 (libx264)')
    assert made['rows'] == 40 and made['renders'] >= made['camera_frames'] - 8
    assert made['camera_frames_per_s'] > 0 and made['renderer']['renderer']


def test_sidecar_follows_the_contract(made, so100):
    t = made['truth']
    with open(os.path.join(made['root'], 'dexvantage_truth.json'), encoding='utf-8') as fh:
        assert json.load(fh) == t
    assert t['schema'] == 'dexvantage.truth/1' and isinstance(t['project_digest'], str) and 'provenance' in t
    assert t['generator']['seed'] == 1 and t['generator']['settings']['frames'] == 20
    for key, c in t['cameras'].items():
        assert key.startswith('observation.images.') and c['sensor_uid'] in {s.uid for s in so100.variant.sensors}
        assert set(c['intrinsics']) == {'model', 'width', 'height', 'K', 'dist'}
        assert (c['intrinsics']['width'], c['intrinsics']['height']) == (64, 48)
        assert set(c['extrinsics']) == {'frame', 'pose'} and len(c['extrinsics']['pose']['quaternion']) == 4
    assert t['cameras']['observation.images.wrist_camera']['extrinsics']['frame'] == 'Fixed_Jaw'
    assert t['cameras']['observation.images.third_person_camera']['extrinsics']['frame'] == 'world'
    kinds = set()
    for e, ep in t['episodes'].items():
        for f in ep['faults']:
            kinds.add(f['kind'])
            assert f['kind'] in SE.FAULT_KINDS and 'camera' in f and 'frames' in f
            if f['kind'] == 'desync':
                assert 'offset_s' in f
            if f['kind'] == 'drift':
                assert 'drift_ppm' in f
            if f['kind'] == 'jitter':
                assert 'jitter_s' in f
            if f['kind'] == 'mount_slip':
                assert set(f['delta']) == {'position', 'rotation_deg'}
        assert set(ep['extrinsics']) == set(t['cameras']) and set(ep['visibility']) == set(t['cameras'])
        for key, v in ep['visibility'].items():
            assert len(v['gripper_visible']) == len(ep['timing'][key]['slot'])
    assert kinds == set(SE.FAULT_KINDS)
    # the slipped mount differs from the true one by the delta, in the later episode only
    slip = next(f for f in t['episodes']['1']['faults'] if f['kind'] == 'mount_slip')
    k = slip['camera']
    p0 = np.array(t['cameras'][k]['extrinsics']['pose']['position'])
    assert t['episodes']['0']['extrinsics'][k] == t['cameras'][k]['extrinsics']
    p1 = np.array(t['episodes']['1']['extrinsics'][k]['pose']['position'])
    assert abs(np.linalg.norm(p1 - p0) - 0.003) < 1e-9
    with open_dataset(made['root']) as ds:
        assert ds.truth() == t and ds.version == made['version']


def test_faults_in_the_pixels(clean, so100):
    """Duplicates repeat their frame exactly; a blurred frame is the mean of renders over its exposure and less sharp
    than one instant; an exposure frame is the instant's render times the gain; the occluder hides the gripper;
    the slipped mount moves the image; the desynced camera shows the robot at stamp + offset."""
    import cv2

    from dexvantage.core.sim.render import Renderer
    t, arr = clean['truth'], clean['arrays']
    plan = SE.plan(so100, dict(SMALL, noise_dn=0.0))
    with Renderer(so100) as R:
        def sharp(e, key, i, delta=None):
            cp = next(c for c in plan.cameras if c.key == key)
            tm = t['episodes'][str(e)]['timing'][key]
            ruid = plan.episodes[e].motion.robot_uid
            v = {'sensor_uid': cp.sensor_uid, 'imager': cp.imager, 'outputs': ('rgb', 'seg'), 'camera_model': cp.model,
                 'q': {ruid: plan.episodes[e].motion.q(tm['content_time_s'][i])},
                 'extrinsic_delta': plan.episodes[e].timelines[key].delta if delta is None else delta}
            return R.render([v])[0]
        seen = set()
        for e, ep in t['episodes'].items():
            e = int(e)
            for f in ep['faults']:
                key = f['camera']
                fr = arr[e]['cameras'][key]['frames']
                if f['kind'] == 'duplicate':
                    for i in f['frames']:
                        assert np.array_equal(fr[i], fr[i - 1])
                    seen.add('duplicate')
                if f['kind'] == 'blur':
                    for i in f['frames']:
                        s = sharp(e, key, i)['rgb']
                        lap = [cv2.Laplacian(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for x in (fr[i], s)]
                        if not np.array_equal(fr[i], s):
                            assert lap[0] < lap[1]
                            seen.add('blur')
                if f['kind'] == 'exposure_change':
                    for i in f['frames']:
                        other = {j for g in ep['faults'] if g['camera'] == key and g['kind'] in
                                 ('blur', 'gripper_hidden', 'duplicate') for j in g['frames']}
                        if i in other:
                            continue
                        s = sharp(e, key, i)['rgb'].astype(float)
                        np.testing.assert_allclose(fr[i], np.clip(np.rint(s * f['gain']), 0, 255), atol=1)
                        seen.add('exposure_change')
                if f['kind'] == 'gripper_hidden':
                    vis = ep['visibility'][key]['gripper_visible']
                    for i in f['frames']:
                        assert not vis[i] and ep['visibility'][key]['gripper_px'][i] == 0
                        assert (fr[i] == np.array(SE.OCCLUDER_RGB, np.uint8)).all(axis=-1).sum() > 20
                    seen.add('gripper_hidden')
                if f['kind'] == 'mount_slip':
                    i = 0
                    a = sharp(e, key, i)['rgb']
                    assert np.array_equal(a, fr[i]) or np.abs(a.astype(int) - fr[i]).max() <= 1
                    b = sharp(e, key, i, delta=np.eye(4))['rgb']
                    assert np.abs(a.astype(float) - b).mean() > 1.0
                    seen.add('mount_slip')
                if f['kind'] == 'desync':
                    tm = ep['timing'][key]
                    np.testing.assert_allclose(np.array(tm['content_time_s']) - np.array(tm['slot']) / tm['fps'],
                                               f['offset_s'] + (np.array(tm['slot']) / tm['fps'] * next(
                                                   (g['drift_ppm'] * 1e-6 for g in ep['faults']
                                                    if g['kind'] == 'drift' and g['camera'] == key), 0.0)),
                                               atol=1e-12)
                    seen.add('desync')
    assert seen >= {'duplicate', 'blur', 'gripper_hidden', 'mount_slip', 'desync'}


def test_same_seed_same_dataset(so100, tmp_path, clean):
    s = SE.generate(so100, str(tmp_path / 'again'), dict(SMALL, noise_dn=0.0), keep_frames=True)
    strip = lambda d: {k: v for k, v in d.items() if k not in ('provenance', 'generator')}  # noqa: E731
    assert strip(s['truth']) == strip(clean['truth'])
    for a, b in zip(s['arrays'], clean['arrays']):
        for key in a['cameras']:
            assert all(np.array_equal(x, y) for x, y in zip(a['cameras'][key]['frames'], b['cameras'][key]['frames']))


# ---------------------------------------------------------------------- refusals
@pytest.mark.mujoco_render
def test_refusals_are_sentences(so100, tmp_path):
    from dexvantage.core.project import blank_project
    full = tmp_path / 'full'
    full.mkdir()
    (full / 'x.txt').write_text('x')
    with pytest.raises(SE.SynthError, match='not empty'):
        SE.generate(so100, str(full), SMALL)
    with pytest.raises(SE.SynthError, match='need a robot'):
        SE.generate(blank_project('Empty'), str(tmp_path / 'a'), SMALL)
    p = sample_project('so100')
    for s in p.variant.sensors:
        s.enabled = False
    with pytest.raises(SE.SynthError, match='no camera to record'):
        SE.plan(p, SMALL)
    # cancelled: a plain sentence and no half-written folder
    out = tmp_path / 'cancelled'
    with pytest.raises(SE.SynthError, match='cancelled'):
        SE.generate(so100, str(out), SMALL, cancel=lambda: True)
    assert not out.exists()
    # an empty folder the user chose is emptied again, not removed
    out.mkdir()
    calls = []
    with pytest.raises(SE.SynthError, match='cancelled'):
        SE.generate(so100, str(out), SMALL, cancel=lambda: calls.append(1) or len(calls) > 3)
    assert out.is_dir() and os.listdir(out) == []


# ---------------------------------------------------------------------- batch, the command, the check
@pytest.mark.mujoco_render
def test_batch_flag(tmp_path):
    out = str(tmp_path / 'b')
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    args = [sys.executable, '-m', 'dexvantage', '--sample', 'so100', '--synth-episodes', out, '--synth-settings',
            json.dumps(dict(SMALL, episodes=1, version='v2.1'))]
    r = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert 'LeRobot v2.1' in r.stdout and 'Faults injected' in r.stdout
    assert os.path.isfile(os.path.join(out, 'dexvantage_truth.json'))
    r = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 2 and 'not an empty folder' in r.stderr and 'Traceback' not in r.stderr
    r = subprocess.run(args[:-1] + ['{"episodes": "two"}'], cwd=ROOT, env=env, capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 2 and 'whole number' in r.stderr


@pytest.mark.mujoco_render
def test_simulate_command_and_dialog(win, app, tmp_path):
    from dexvantage.ui.dialogs import synth_episodes as D
    a = win.actions_by_id['synth_episodes']
    assert a.isEnabled() and a in win.menus['Simulate'].actions()
    assert win.open_sample('so100')
    dlg = D.open_dialog(win)
    st = dlg.settings()
    assert st['version'] == 'v3.0' and st['episodes'] == 4 and st['frames'] is None and len(st['faults']) == 10
    assert dlg.start() is False and 'Choose a folder' in dlg.hint.text()
    full = tmp_path / 'full'
    full.mkdir()
    (full / 'x').write_text('x')
    dlg.folder.setText(str(full))
    assert dlg.start() is False and 'not empty' in dlg.hint.text()
    dlg.n_episodes.setValue(1)
    dlg.n_frames.setValue(8)
    dlg.size.setCurrentIndex(2)                                  # up to 160 px wide
    dlg.faults['blur'].setChecked(False)
    out = str(tmp_path / 'ds')
    dlg.folder.setText(out)
    assert dlg.start() is True
    for _ in range(600):
        pump(app, 0.05)
        if 'synth_episodes' not in win._jobs:
            break
    t = json.load(open(os.path.join(out, 'dexvantage_truth.json'), encoding='utf-8'))
    assert t['generator']['settings']['max_width'] == 160 and not t['generator']['settings']['faults']['blur'][
        'enabled']
    assert all(c['intrinsics']['width'] <= 160 for c in t['cameras'].values())
    assert 'synthetic episode' in win.msg_label.text() or win.msg_label.text() == ''
    dlg.close()


@pytest.mark.mujoco_render
def test_validation_check_registers_and_passes(tmp_path):
    from dexvantage.validation import suite
    suite.load_checks()
    spec = next(s for s in suite.CHECKS if s.id == 'groundtruth.synthetic_roundtrip')
    assert spec.model == 'groundtruth' and spec.tolerance == 0 and spec.cost == 'study'

    class Ctx:
        def tmp(self):
            return str(tmp_path)
    o = spec.fn(Ctx())
    assert o.error == 0, o.detail['mismatches']
    assert o.detail['v2.1']['worst_psnr_db'] >= 30 and o.detail['v3.0']['worst_psnr_db'] >= 30
