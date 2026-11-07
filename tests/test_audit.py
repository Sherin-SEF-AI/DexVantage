"""The dataset and episode auditor (core/audit, core/studies/audit.py): a seeded synthetic campaign with injected
faults (core/sim/episodes.py) audited from its data alone (a copy of the dataset without the ground-truth sidecar),
then scored against the sidecar; the per-episode check functions on hand-made streams; the step-against-gradual
classifier ported from Vantage against Vantage's recorded outputs and its own adapted tests; the project's data
section; the reports, the FiftyOne export, the batch flags and the page."""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip('pyarrow')
pytest.importorskip('av')

from conftest import pump  # noqa: E402
from dexvantage.core import jobs  # noqa: E402
from dexvantage.core.audit import campaign as AC  # noqa: E402
from dexvantage.core.audit import episode as AE  # noqa: E402
from dexvantage.core.audit import report as RP  # noqa: E402
from dexvantage.core.audit import slip as SL  # noqa: E402
from dexvantage.core.datasets import CameraStream, Episode  # noqa: E402
from dexvantage.core.robot.samples import sample_project  # noqa: E402
from dexvantage.core.sim import episodes as SE  # noqa: E402
from dexvantage.validation import checks_audit as CA  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SMALL = {'episodes': 2, 'frames': 60, 'size': [160, 120], 'seed': 2, 'faults': ['drop', 'usb_reset']}


@pytest.fixture(scope='module')
def so100():
    return sample_project('so100')


@pytest.fixture(scope='module')
def camp(so100, tmp_path_factory):
    """The validation check's campaign: generated, copied without its sidecar, audited, then the sidecar read."""
    base = tmp_path_factory.mktemp('audit')
    root = str(base / 'campaign')
    SE.generate(so100, root, dict(CA.SETTINGS))
    blind = str(base / 'blind')
    shutil.copytree(root, blind)
    os.remove(os.path.join(blind, 'dexvantage_truth.json'))
    res = AC.audit_dataset(blind, so100)
    with open(os.path.join(root, 'dexvantage_truth.json'), encoding='utf-8') as fh:
        truth = json.load(fh)
    return {'res': res, 'truth': truth, 'root': root, 'blind': blind, 'base': str(base)}


# ---------------------------------------------------------------------- the campaign against its ground truth
def test_detection_from_the_data_alone(camp):
    res, truth = camp['res'], camp['truth']
    assert res['dataset']['episodes_audited'] == 6
    sc = CA.score(res, truth)
    for kind, d in sc.items():
        assert d['injected'] >= 1, kind
        assert d['recall'] >= CA.RECALL_FLOOR[kind], (kind, d)
        assert (d['fp_rate'] or 0.0) <= CA.FP_CEILING[kind], (kind, d)
    # exact kinds: every injected frame found
    for kind in ('drop', 'usb_reset', 'gripper_hidden', 'exposure_change'):
        assert sc[kind]['frames_found'] == sc[kind]['frames_injected'], kind


def test_timing_frames_match_the_truth_exactly(camp):
    res, truth = camp['res'], camp['truth']
    for ep in res['episodes']:
        te = truth['episodes'][str(ep['index'])]
        for key, cam in ep['cameras'].items():
            for kind, field in (('drop', 'drops'), ('usb_reset', 'usb_gaps')):
                want = sorted(k for f in te['faults'] if f['kind'] == kind and f['camera'] == key
                              for k in f['frames'])
                assert sorted(g['frame'] for g in cam['timing'][field]) == want, (ep['index'], key, kind)
            jit = any(f['kind'] == 'jitter' and f['camera'] == key for f in te['faults'])
            assert (cam['timing']['jitter_ms'] > 1.0) == jit
            if not jit:
                assert cam['timing']['jitter_ms'] < 0.05          # the stamps sit on the 1/90000 s time base


def test_desync_against_the_injected_offset(camp):
    rows = CA.desync_errors(camp['res'], camp['truth'])
    static = [r for r in rows if not r[5]]
    assert len(static) == 6
    for ep, key, est, true, sig, _on in static:
        assert abs(est - true) <= CA.DESYNC_TOL_MS, (ep, est, true)
        assert sig is not None and sig > 0
    # the requirement fails only where the offset is beyond 25 ms and 3 σ
    for ep in camp['res']['episodes']:
        for r in ep['requirements']:
            if r['id'] == 'max_offset_ms' and not r['pass']:
                assert abs(r['value']) > 25.0


def test_mount_slip_step_found_on_the_slipped_camera(camp):
    res, truth = camp['res'], camp['truth']
    slipped = {f['camera']: f for te in truth['episodes'].values() for f in te['faults'] if f['kind'] == 'mount_slip'}
    assert len(slipped) == 1
    key, f = next(iter(slipped.items()))
    m = res['mounts'][key]
    assert m['source'] == 'registration' and m['kind'] == 'slip'
    steps = [e for e in m['events'] if e['kind'] == 'step']
    pitch = next(e for e in steps if e['axis'] == 'rotation y')
    assert pitch['episode'] == f['from_episode'] == 3
    assert pitch['size'] == pytest.approx(f['delta']['rotation_deg'][1], abs=0.15)
    # the episodes after the step need new extrinsics (the slip is beyond the 0.5° default)
    for e in res['episodes']:
        shift = [r for r in e['requirements'] if r['id'] == 'max_mount_shift_deg' and r['camera'] == key]
        assert shift and shift[0]['pass'] == (e['index'] < 3)
    for other, mo in res['mounts'].items():
        if other != key:
            assert mo['kind'] == 'steady'


def test_mount_from_the_link_silhouette(camp, so100, tmp_path):
    """The second source: core/calib/render_calib.estimate through its API, absolute against the plan; where it
    refuses (the campaign's 320 × 240 frames do not have the plan's 16:9 shape) the registration is used, said."""
    from dexvantage.core.audit import plan as AP
    from dexvantage.core.datasets import open_dataset
    key = next(f['camera'] for te in camp['truth']['episodes'].values() for f in te['faults']
               if f['kind'] == 'mount_slip')

    def series(root, eps):
        with open_dataset(root) as ds:
            ep0 = ds.load(0)
            plan = AP.PlanContext(so100, ep0.joint_names)
            uid = AP.match_cameras(so100, list(ep0.cameras))[key][0]
            return AC.mount_series(ds, so100, plan, {key: uid}, eps, dict(AC.SETTINGS, mount_source='silhouette'),
                                   AE.requirements_of(None))[key]
    m = series(camp['blind'], [3])
    assert m['source'] == 'registration' and 'different shape' in m['notes'][0]
    root = str(tmp_path / 'sil')
    SE.generate(so100, root, {'episodes': 2, 'frames': 90, 'max_width': 320, 'seed': 3,
                              'faults': {'mount_slip': {'from_episode': 1, 'camera': key}}})
    truth = json.load(open(os.path.join(root, 'dexvantage_truth.json'), encoding='utf-8'))
    f = next(f for f in truth['episodes']['1']['faults'] if f['kind'] == 'mount_slip')
    m = series(root, [1])
    assert m['source'] == 'silhouette' and m['episodes'][0]['absolute']
    # the wiring, not render_calib's accuracy (its own checks measure that): the slip's sign and size, roughly
    # (measured here 1.15° against the injected 1.5° on 90 frames at 320 × 180)
    assert m['episodes'][0]['rotation_deg'][1] == pytest.approx(f['delta']['rotation_deg'][1], abs=0.6)


def test_verdicts_lists_and_json(camp):
    res, truth = camp['res'], camp['truth']
    s = res['summary']
    assert s['episodes'] == 6 and set(s['verdicts']) == {'pass', 'fix', 'recalibrate', 'drop'}
    for e in res['episodes']:
        faults = {f['kind'] for f in truth['episodes'][str(e['index'])]['faults']}
        if faults & {'drop', 'usb_reset', 'gripper_hidden'}:
            assert e['verdict'] == 'drop' and e['index'] in s['drop_or_rerecord']
        assert e['action']['lag_rows'] == pytest.approx(1.0, abs=0.1)     # the generator's action lead
    text = json.dumps(res, allow_nan=False)
    assert camp['base'] not in text and '/home/' not in text
    assert res['dataset']['name'] == 'blind'
    assert all(r['default'] for r in res['requirements'])
    assert s['advice'] and 'Traceback' not in s['advice']


def test_validation_checks_on_the_same_campaign(camp, so100):
    CA._CACHE['c'] = (camp['res'], camp['truth'], camp['blind'], so100)
    try:
        o = CA.fault_recall(None)  # the cached campaign: no context needed
        assert o.error == 0.0, o.detail
        assert set(o.detail['kinds']) == set(CA.KINDS)
        o = CA.desync_offset(None)
        assert o.error <= CA.DESYNC_TOL_MS
        o = CA.plan_vs_recording(None)
        assert o.ok and o.error <= CA.PLAN_TOL_DEG, o.detail      # the rotation vectors compared, not their sizes
    finally:
        CA._CACHE.clear()


# ---------------------------------------------------------------------- the check functions on made streams
def _stream(ts, fps=30.0, hw=None):
    return CameraStream('observation.images.cam', 64, 48, fps, np.asarray(ts, float), hw)


def test_timing_gaps_drops_jitter_and_device_clock():
    req = AE.requirements_of(None)
    t = np.arange(100) / 30.0
    keep = np.ones(100, bool)
    keep[[20]] = False                      # one dropped frame
    keep[50:58] = False                     # a 0.27 s gap
    ts = t[keep]
    tm = AE.timing(_stream(ts), 30.0, req)
    assert [g['frame'] for g in tm['drops']] == [20] and tm['dropped_frames'] == 1
    assert [g['missing'] for g in tm['usb_gaps']] == [8] and tm['gap_frames'] == 8
    assert tm['fps_measured'] == pytest.approx(30.0, rel=1e-9) and tm['jitter_ms'] < 1e-6
    rng = np.random.default_rng(0)
    jt = t + rng.normal(0, 0.002, 100)
    assert AE.timing(_stream(jt), 30.0, req)['jitter_ms'] == pytest.approx(2.0, rel=0.25)
    hw = t * (1 - 200e-6) + 5.0             # a device clock running 200 ppm slow
    tm = AE.timing(_stream(t, hw=hw), 30.0, req)
    assert tm['hw']['drift_ppm'] == pytest.approx(200.0, rel=1e-3)
    assert AE.timing(_stream(t[:2]), 30.0, req)['fps_measured'] is None


def test_exposure_steps_and_runs_and_recorded_column():
    req = AE.requirements_of(None)
    rng = np.random.default_rng(1)
    base = rng.uniform(40, 140, (48, 64))
    g = np.stack([np.clip(base * (1.4 if 10 <= k < 20 else 1.0) + rng.normal(0, 1, base.shape), 0, 255)
                  for k in range(30)]).astype(np.uint8)
    ex = AE.exposure(g, np.zeros(30, bool), req)
    assert [s['frame'] for s in ex['steps']] == [10, 20]
    assert ex['runs'] == [{'start': 10, 'end': 20, 'log_ratio': pytest.approx(np.log(1.4), abs=0.03)}]
    col = {'observation.cam.exposure_us': np.r_[np.full(10, 1000.0), np.full(20, 2000.0)]}
    ex = AE.exposure(g, np.zeros(30, bool), req, col, 'observation.images.cam')
    assert 'recorded column' in ex['source'] and [s['frame'] for s in ex['steps']] == [10]


def test_action_alignment_lag():
    t = np.arange(200) / 30.0
    q = np.stack([np.sin(t), np.cos(1.3 * t)], 1)
    ep = Episode('e', 0, t, q, ['a', 'b'], action=np.vstack([q[3:], np.repeat(q[-1:], 3, 0)]),
                 action_names=['a', 'b'])
    r = AE.action_alignment(ep)
    assert r['lag_rows'] == pytest.approx(3.0, abs=0.05) and r['joints'].startswith('2 joints')
    ep.action = None
    assert AE.action_alignment(ep)['lag_rows'] is None


def test_requirements_defaults_yours_and_refusals(so100):
    req = AE.requirements_of(so100)
    assert req['max_offset_ms'] == 25.0 and 'max_offset_ms' in req['defaults']
    p = so100.snapshot()
    p.data['requirements'] = {'max_offset_ms': 10, 'action_lag_rows': [0, 5]}
    req = AE.requirements_of(p)
    assert req['max_offset_ms'] == 10 and req['action_lag_rows'] == [0.0, 5.0]
    assert 'max_offset_ms' not in req['defaults']
    for bad, words in (({'colour': 1}, 'no audit requirement called colour'), ({'max_jitter_ms': 'x'}, 'a number'),
                       ({'action_lag_rows': [3, 1]}, 'low end above'), ({'max_gaps': -1}, 'zero or more')):
        with pytest.raises(jobs.StudyError) as e:
            AE.requirements_of(None, bad)
        assert words in str(e.value)


# ---------------------------------------------------------------------- the slip classifier (ported from Vantage)
def test_slip_matches_vantages_recorded_outputs():
    o = CA.slip_vs_vantage(None)
    assert o.ok and o.error <= 1e-9, o.detail
    assert o.detail['recorded']['vantage_commit']


def test_slip_model_selection_adapted_from_vantage():
    """Vantage tests/test_online_calib.py test_model_selection_finds_steps_and_slopes_and_no_ghosts_in_noise and
    test_a_step_that_returns_is_an_excursion_and_one_that_stays_a_knock, adapted to the campaign's time covariate."""
    rng = np.random.default_rng(11)
    n = 40
    t = np.arange(n) + 0.5
    s = np.full(n, 0.03)
    y = 0.02 + np.where(t > 23, 0.5, 0.0) + rng.normal(0, 0.03, n)
    c = SL.classify(t, y, s, min_step=0.1, min_amp=0.1)
    assert c['kind'] == 'step' and c['events'][0]['j'] == 23
    assert c['events'][0]['size'] == pytest.approx(0.5, abs=0.04)
    y = 0.1 + 0.02 * t + rng.normal(0, 0.03, n)
    c = SL.classify(t, y, s, min_step=0.1, min_amp=0.1)
    assert c['kind'] == 'gradual' and c['events'][0]['slope'] == pytest.approx(0.02, abs=0.003)
    ghosts = 0
    for seed in range(30):
        r = np.random.default_rng(100 + seed)
        c = SL.classify(t, r.normal(0, 0.03, n), s * r.uniform(0.7, 1.5), min_step=0.1, min_amp=0.1)
        ghosts += c['kind'] != 'steady'
    assert ghosts <= 1
    rng = np.random.default_rng(5)
    y = np.r_[np.zeros(8), np.full(6, 0.6), np.zeros(8)] + rng.normal(0, 0.03, 22)
    c = SL.classify(np.arange(22) + 0.5, y, np.full(22, 0.03), min_step=0.1, min_amp=0.1, min_seg=3)
    assert [e['kind'] for e in c['events']] == ['excursion'] and c['events'][0]['j'] == 8
    assert c['events'][0]['j_end'] == 14
    y = np.r_[np.zeros(8), np.full(6, 0.6), np.full(8, 0.3)] + rng.normal(0, 0.03, 22)
    c = SL.classify(np.arange(22) + 0.5, y, np.full(22, 0.03), min_step=0.1, min_amp=0.1, min_seg=3)
    assert sorted(e['kind'] for e in c['events']) == ['step', 'step']
    assert SL.classify([0, 1, 2], [0, 0, 1], [0.1] * 3)['kind'] == 'too short'
    tau, res = SL.refine_step(np.arange(0.125, 8.0, 0.25), np.where(np.arange(0.125, 8.0, 0.25) > 4.6, 0.8, 0.0),
                              4.5, 0.0, 0.8, 2.0, 7.0)
    assert tau == pytest.approx(4.625, abs=0.13) and res == pytest.approx(0.25)


# ---------------------------------------------------------------------- the project's data section
def test_data_section_round_trip_and_content_hash(so100, tmp_path):
    from dexvantage.core import provenance as PV
    from dexvantage.core.project import Project, data_from_dict
    p = so100.snapshot()
    before = PV.project_digest(p)['sha256']
    assert p.data == {'datasets': [], 'requirements': {}}
    p.data = data_from_dict({'datasets': [{'name': 'run 1', 'path': 'runs/run1', 'format': 'lerobot',
                                           'version': 'v3.0', 'episodes': 6, 'exclude': [3, 1, 'x'],
                                           'cameras': {'observation.images.wrist': 'abc'}}, 'junk'],
                             'requirements': {'max_offset_ms': 10}})
    assert len(p.data['datasets']) == 1 and p.data['datasets'][0]['exclude'] == [1, 3]
    after = PV.project_digest(p)['sha256']
    assert after != before
    path = str(tmp_path / 'p.dexv')
    p.save(path)
    q = Project.load(path)
    assert q.data == p.data and PV.project_digest(q)['sha256'] == after
    q.data = data_from_dict(None)
    assert PV.project_digest(q)['sha256'] == before          # empty: not in the content hash
    # the schema copies stay identical and name the section
    a = json.load(open(os.path.join(ROOT, 'docs', 'project.schema.json'), encoding='utf-8'))
    b = json.load(open(os.path.join(ROOT, 'dexvantage', 'resources', 'project.schema.json'), encoding='utf-8'))
    assert a == b and 'data' in a['properties']
    try:
        import jsonschema
    except ImportError:
        return
    jsonschema.validate(json.loads(json.dumps(p.to_dict())), a)


# ---------------------------------------------------------------------- reports and FiftyOne
def test_reports_and_fiftyone_export(camp, so100, tmp_path):
    res = camp['res']
    j = RP.write_json(res, so100, str(tmp_path / 'audit.json'))
    doc = json.load(open(j, encoding='utf-8'))
    assert 'provenance' in doc and doc['summary'] == json.loads(json.dumps(res['summary']))
    h = RP.write_html(res, so100, str(tmp_path / 'audit.html'))
    text = open(h, encoding='utf-8').read()
    assert '<svg' in text and 'Provenance' in text and 'Requirements' in text and camp['base'] not in text
    out = str(tmp_path / 'fo')
    r = RP.export_fiftyone(res, camp['blind'], out, so100)
    assert r['samples'] == 12 and r['frames'] == sum(c['timing']['frames'] for e in res['episodes']
                                                     for c in e['cameras'].values())
    meta = json.load(open(os.path.join(out, 'metadata.json'), encoding='utf-8'))
    assert 'version' not in meta and meta['media_type'] == 'video'          # the legacy layout FiftyOne reads
    samples = json.load(open(os.path.join(out, 'samples.json'), encoding='utf-8'))['samples']
    for sd in samples:
        assert sd['_media_type'] == 'video' and not os.path.isabs(sd['filepath'])
        assert os.path.isfile(os.path.join(out, sd['filepath']))
        fr = json.load(open(os.path.join(out, sd['frames']), encoding='utf-8'))['frames']
        assert min(map(int, fr)) == 1 and all(v['frame_number'] == int(k) for k, v in fr.items())
        assert any(t.startswith('audit:') for t in sd['tags'])
    s0 = next(sd for sd in samples if sd['episode_index'] == 0 and 'third_person' in sd['camera'])
    assert 'desync' in s0['tags'] and s0['desync_offset_ms'] > 25
    hid = next(sd for sd in samples if sd['episode_index'] == 1 and 'wrist' in sd['camera'])
    fr = json.load(open(os.path.join(out, hid['frames']), encoding='utf-8'))['frames']
    assert sorted(int(k) - 1 for k, v in fr.items() if v['gripper_hidden']) == \
        sorted(f['frames'] for f in camp['truth']['episodes']['1']['faults'] if f['kind'] == 'gripper_hidden')[0]
    with pytest.raises(jobs.StudyError, match='not empty'):
        RP.export_fiftyone(res, camp['blind'], out, so100)
    with pytest.raises(jobs.StudyError, match='LeRobot v2.1'):
        RP.export_fiftyone(res, camp['blind'], str(tmp_path / 'fo2'), so100, media='reference')
    try:
        import fiftyone as fo
    except ImportError:
        return
    ds = fo.Dataset.from_dir(out, dataset_type=fo.types.FiftyOneDataset)    # pragma: no cover - optional
    assert len(ds) == 12                                                     # pragma: no cover


# ---------------------------------------------------------------------- the study and batch mode
@pytest.fixture(scope='module')
def small(so100, tmp_path_factory):
    root = str(tmp_path_factory.mktemp('audit_small') / 'small')
    SE.generate(so100, root, dict(SMALL))
    return root


def test_study_refuses_without_a_dataset_and_runs_with_one(so100, small):
    from dexvantage.core.studies import audit as AU
    p = so100.snapshot()
    with pytest.raises(jobs.StudyError, match='Choose a dataset'):
        jobs.run_study('audit', p)
    AU.use_dataset(p, os.path.join(small, 'nothing'))
    with pytest.raises(jobs.StudyError, match='not there'):
        jobs.run_study('audit', p)
    AU.use_dataset(p, small)
    doc = jobs.run_study('audit', p)
    assert doc['schema'] == 'dexvantage.audit/1' and doc['summary']['episodes'] == 2
    assert all('drop' in e['flags'] or 'usb_reset' in e['flags'] for e in doc['episodes'])
    json.dumps(doc, allow_nan=False)
    assert jobs.STUDIES['audit'].batch == 'audit' and not jobs.STUDIES['audit'].quick


def test_batch_audit_with_dataset(small, tmp_path):
    out = str(tmp_path / 'a.json')
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    r = subprocess.run([sys.executable, '-m', 'dexvantage', '--sample', 'so100', '--audit', out, '--dataset', small],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    doc = json.load(open(out, encoding='utf-8'))
    assert doc['summary']['episodes'] == 2 and 'provenance' in doc and small not in json.dumps(doc)
    r = subprocess.run([sys.executable, '-m', 'dexvantage', '--audit', out, '--dataset', str(tmp_path / 'none')],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 2 and 'not there' in r.stderr and 'Traceback' not in r.stderr


# ---------------------------------------------------------------------- the page
def _for_project(res, project):
    """The campaign's result with its sensor uids replaced by those of another load of the same sample."""
    doc = json.loads(json.dumps(res))
    by_name = {s.name: s.uid for s in project.variant.sensors}
    old = {}
    for k, c in doc['cameras'].items():
        for e in doc['episodes']:
            name = e['cameras'][k].get('sensor')
            if name in by_name:
                old[c['sensor_uid']] = by_name[name]
    text = json.dumps(doc)
    for a, b in old.items():
        text = text.replace(a, b)
    return json.loads(text)


def test_page(app, win, camp, small, tmp_path):
    assert 'audit' in win.pages
    a = win.actions_by_id['audit']
    assert a.isEnabled() and (a.data() or {}).get('phase') is None
    assert win.open_sample('so100')
    page = win.pages['audit']
    win.open_page('audit')
    pump(app, 0.05)
    page.run()                                       # no dataset yet: said in the headline
    pump(app, 0.1)
    assert page.head.status.text.text() == 'no dataset'
    page.add_dataset(small)
    for _ in range(200):
        pump(app, 0.05)
        if win.store.project.data['datasets']:
            break
    d = win.store.project.data['datasets'][0]
    assert d['format'] == 'lerobot' and d['episodes'] == 2 and d['path'] == small
    assert page.ds_box.count() == 1
    doc = _for_project(camp['res'], win.store.project)
    page._finished('audit', doc)
    pump(app, 0.05)
    assert page.head.figure.text().endswith('/ 6') and page.reqs.topLevelItemCount() >= 8
    assert page.eps.topLevelItemCount() == 6 and page.cams.topLevelItemCount() == 2
    assert page.chart.series
    assert page.overlays(win.viewport)
    drop = page.suggestion()
    assert drop == doc['summary']['drop_or_rerecord']
    page.apply()
    assert win.store.project.data['datasets'][0]['exclude'] == drop
    win.store.undo.undo()
    assert 'exclude' not in win.store.project.data['datasets'][0]
    page.spins['max_offset_ms'].committed.emit(12.0)
    assert win.store.project.data['requirements']['max_offset_ms'] == 12.0
    win.store.undo.undo()
    assert 'max_offset_ms' not in win.store.project.data['requirements']
    j, h = page.export_report(str(tmp_path))
    assert os.path.isfile(j) and os.path.isfile(h)
    page.preview()
    page._replaced()
    pump(app, 0.1)
