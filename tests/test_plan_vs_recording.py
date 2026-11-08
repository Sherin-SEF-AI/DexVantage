"""Plan against recording (core/studies/plan_vs_recording.py): a synthetic recording of the SO-100 sample whose
third-person mount slips and whose wrist camera keeps its mount, compared with the plan from its data alone (a copy
without the ground-truth sidecar), then checked against the sidecar; the study, its batch flag and its page."""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip('pyarrow', reason='pyarrow (the lerobot extra) is not installed')
pytest.importorskip('av', reason='av (the lerobot extra) is not installed')

from conftest import pump  # noqa: E402
from dexvantage.core import jobs  # noqa: E402
from dexvantage.core.robot.samples import sample_project  # noqa: E402
from dexvantage.core.sim import episodes as SE  # noqa: E402
from dexvantage.core.studies import audit as AU  # noqa: E402
from dexvantage.core.studies import plan_vs_recording as PR  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SETTINGS = {'episodes': 3, 'frames': 120, 'max_width': 320, 'seed': 4,
            'faults': {'mount_slip': {'from_episode': 1}, 'desync': {'episodes': [0]}}}


@pytest.fixture(scope='module')
def so100():
    return sample_project('so100')


@pytest.fixture(scope='module')
def rec(so100, tmp_path_factory):
    from render_support import require
    require()                               # renders MuJoCo scenes
    base = tmp_path_factory.mktemp('pvr')
    root = str(base / 'rec')
    SE.generate(so100, root, dict(SETTINGS))
    blind = str(base / 'blind')
    shutil.copytree(root, blind)
    os.remove(os.path.join(blind, 'dexvantage_truth.json'))
    doc = PR.compare(so100, blind)
    truth = json.load(open(os.path.join(root, 'dexvantage_truth.json'), encoding='utf-8'))
    return {'doc': doc, 'truth': truth, 'blind': blind, 'base': str(base)}


def test_cameras_matched_and_resolution(rec, so100):
    doc, truth = rec['doc'], rec['truth']
    assert not doc['missing']
    for key, c in truth['cameras'].items():
        row = next(r for r in doc['cameras'] if r['key'] == key)
        assert row['present'] == 'yes' and row['sensor_uid'] == c['sensor_uid'] and row['matched'] == 'by name'
        r = row['resolution']
        assert r['recorded'] == [c['intrinsics']['width'], c['intrinsics']['height']]
        s = next(x for x in so100.variant.sensors if x.uid == c['sensor_uid'])
        assert r['verdict'] == ('same' if r['recorded'] == r['plan'] else 'scaled')
        assert r['plan'][0] / r['plan'][1] == pytest.approx(r['recorded'][0] / r['recorded'][1], rel=0.02)
        assert row['fps']['recorded'] == pytest.approx(30.0, rel=1e-3)          # the 1/90000 s time base
        assert s.name == row['sensor']


def test_mount_moved_on_the_slipped_camera_only(rec):
    doc, truth = rec['doc'], rec['truth']
    slip = next(f for te in truth['episodes'].values() for f in te['faults'] if f['kind'] == 'mount_slip')
    for row in doc['cameras']:
        m = row['mount']
        if row['key'] == slip['camera']:
            assert m['verdict'] == 'moved' and m['source'] == 'registration' and m['absolute'] is False
            assert m['change_norm_deg'] == pytest.approx(np.linalg.norm(slip['delta']['rotation_deg']), abs=0.2)
        else:
            assert m['verdict'] == 'steady'
    bad = {(c['what'], c['camera']) for c in doc['checks'] if not c['match']}
    assert ('mount', slip['camera']) in bad


def test_sync_and_visibility(rec):
    doc, truth = rec['doc'], rec['truth']
    desync = next(f for f in truth['episodes']['0']['faults'] if f['kind'] == 'desync')
    for row in doc['cameras']:
        sy = row['sync']
        assert sy['verdict'] in ('in sync', 'out of sync') and sy['range_ms'] is not None
        if row['key'] == desync['camera'] and not truth['cameras'][row['key']]['wrist']:
            assert sy['verdict'] == 'out of sync' and sy['range_ms'][1] > 25.0
        v = row['visibility']
        assert v['frames'] > 0 and v['both_seen'] + v['predicted_only'] + v['recorded_only'] + v['both_hidden'] == \
            v['frames']
    wrist = next(r for r in doc['cameras'] if truth['cameras'][r['key']]['wrist'])
    assert wrist['visibility']['agreement'] >= 0.8                 # the gripper is in view and nothing hides it
    s = doc['summary']
    assert 0 < s['match'] < s['checks'] and s['status'] in ('good', 'warning', 'serious', 'critical')
    assert 'mount' in s['advice'] and 'Traceback' not in s['advice']
    text = json.dumps(doc, allow_nan=False)
    assert rec['base'] not in text


def test_study_and_refusals(so100, rec):
    from dexvantage.core.project import blank_project
    p = so100.snapshot()
    with pytest.raises(jobs.StudyError, match='Choose a dataset'):
        jobs.run_study(PR.SID, p)
    AU.use_dataset(p, rec['blind'], PR.SID)
    doc = jobs.run_study(PR.SID, p)
    assert doc['schema'] == 'dexvantage.plan_vs_recording/1' and len(doc['cameras']) == 2
    b = blank_project('Empty')
    AU.use_dataset(b, rec['blind'], PR.SID)
    with pytest.raises(jobs.StudyError, match='needs a robot'):
        jobs.run_study(PR.SID, b)
    with pytest.raises(jobs.StudyError, match='must be a number'):
        PR.settings_of(p, {'fps_tol_pct': 'x'})


def test_batch_flag(rec, tmp_path):
    out = str(tmp_path / 'pvr.json')
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    r = subprocess.run([sys.executable, '-m', 'dexvantage', '--sample', 'so100', '--plan-vs-recording', out,
                        '--dataset', rec['blind']], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    doc = json.load(open(out, encoding='utf-8'))
    assert doc['summary']['checks'] > 0 and 'provenance' in doc
    r = subprocess.run([sys.executable, '-m', 'dexvantage', '--plan-vs-recording', out, '--dataset', rec['blind']],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 2 and 'needs a project' in r.stderr


def test_page(app, win, rec):
    assert 'plan_vs_recording' in win.pages
    assert win.actions_by_id['plan_vs_recording'].isEnabled()
    assert win.open_sample('so100')
    page = win.pages['plan_vs_recording']
    win.open_page('plan_vs_recording')
    pump(app, 0.05)
    page.run()
    pump(app, 0.1)
    assert page.head.status.text.text() == 'no dataset'
    p = win.store.project
    by_name = {s.name: s.uid for s in p.variant.sensors}
    doc = json.loads(json.dumps(rec['doc']))
    for row in doc['cameras']:
        row['sensor_uid'] = by_name[row['sensor']]
    page._finished(PR.SID, doc)
    pump(app, 0.05)
    assert page.cams.topLevelItemCount() == 2 and page.vis.topLevelItemCount() == 2
    assert page.head.figure.text() not in ('', '—')
    assert page.overlays(win.viewport)
    sg = page.suggestion()
    if sg:                                       # cameras recorded smaller than their plan stream
        page.apply()
        for uid, wh in sg:
            assert list(p.variant.sensor(uid).stream) == wh
        win.store.undo.undo()
    page.preview()
    page._replaced()
    pump(app, 0.1)
