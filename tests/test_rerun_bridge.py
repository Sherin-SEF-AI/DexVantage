"""The Rerun bridge (core/bridges/rerun.py): the scene's plain data, logging through a fake rerun module (no viewer,
no network), an .rrd written and read back by rerun's own reader when rerun-sdk is installed, the batch flag, the
message when it is missing, and Export ▸ Stream to Rerun… as a window job."""
import json
import sys
import time
import types

import numpy as np
import pytest

from dexvantage.core.bridges import rerun as RB
from dexvantage.core.robot.samples import sample_project

from conftest import pump

HAVE_RERUN = RB.available()
AUDIT = {'study': 'audit', 'episodes': [
    {'index': 0, 'id': 'ep0', 'verdict': 'pass', 'requirements': [{'id': 'max_dropped_frames', 'pass': True}]},
    {'index': 3, 'id': 'ep3', 'verdict': 'fix', 'requirements': [
        {'id': 'max_offset_ms', 'pass': False, 'camera': 'wrist'}]}]}


@pytest.fixture(scope='module')
def ur5e():
    return sample_project('ur5e')


def test_scene_data(ur5e):
    sc = RB.scene(ur5e, audit=AUDIT)
    rob = sc['robots'][0]
    assert rob['times'] is not None and rob['trajectory'] == 'Scripted pick (sample)'
    assert rob['T'].shape == (len(rob['times']), len(rob['links']), 4, 4)
    assert 'gripper_base' in rob['links'] and 'wrist_3_link' in rob['links']
    assert rob['times'][0] == 0.0 and rob['times'][-1] == pytest.approx(4.5)
    assert np.diff(rob['times']).max() <= 1 / RB.RATE_HZ + 1e-9
    names = [s['name'] for s in sc['sensors']]
    assert names == ['Wrist D405', 'Overhead D435']
    wrist = sc['sensors'][0]
    assert wrist['path'].startswith(rob['paths']['wrist_3_link'] + '/sensors/')
    assert sc['sensors'][1]['path'].startswith('world/sensors/')
    assert len(sc['blind_zones']) == 2 and all(z['min_z_m'] > 0 for z in sc['blind_zones'])
    assert sc['volumes'][0]['path'] == 'world/task_volumes/Tabletop_sample'
    assert [e['verdict'] for e in sc['audit']] == ['pass', 'fix']
    assert sc['audit'][1]['counts']['desync'] == 1 and sc['audit'][1]['failed'] == ['desync']
    # without a trajectory: one static pose at q
    p = sample_project('so100')
    sc2 = RB.scene(p, meshes=False)
    assert sc2['robots'][0]['times'] is None and sc2['robots'][0]['T'].shape[0] == 1
    with pytest.raises(RB.RerunError, match='sampling rate'):
        RB.scene(p, rate_hz=0)


class _Arch:
    def __init__(self, kind, *a, **k):
        self.kind, self.a, self.k = kind, a, k


def _fake_rerun(calls):
    m = types.ModuleType('rerun')
    for k in ('Mesh3D', 'LineStrips3D', 'Transform3D', 'Pinhole', 'Boxes3D', 'TextDocument', 'TextLog', 'Scalars'):
        setattr(m, k, (lambda kind: (lambda *a, **kw: _Arch(kind, *a, **kw)))(k))
    m.ViewCoordinates = types.SimpleNamespace(RIGHT_HAND_Z_UP='RHZU', RDF='RDF')

    class Rec:
        def __init__(self, app, **k):
            calls.append(('init', app))

        def save(self, path):
            calls.append(('save', path))

        def spawn(self):
            calls.append(('spawn',))

        def connect_grpc(self, url=None):
            calls.append(('connect', url))

        def log(self, path, arch, static=False):
            calls.append(('log', path, arch, static))

        def set_time(self, timeline, **k):
            calls.append(('time', timeline, k))

        def reset_time(self):
            calls.append(('reset',))

        def flush(self):
            pass

        def disconnect(self):
            calls.append(('disconnect',))
    m.RecordingStream = Rec
    return m


def test_logging_through_a_fake_module(ur5e, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, 'rerun', _fake_rerun(calls))
    monkeypatch.setattr(RB, 'available', lambda: True)
    res = RB.stream(ur5e, 'save', str(tmp_path / 's.rrd'), audit=AUDIT)
    assert res['text'].startswith('Sent 1 robot') and 'audited episodes' in res['text']
    assert calls[0] == ('init', 'dexvantage') and calls[1] == ('save', str(tmp_path / 's.rrd'))
    assert calls[-1] == ('disconnect',)
    logs = [c for c in calls if c[0] == 'log']
    first = {c[1]: c for c in logs}
    assert first['world'][2] == 'RHZU'
    logs = [c for c in logs if isinstance(c[2], _Arch)]
    pin = [c for c in logs if c[2].kind == 'Pinhole']
    assert len(pin) == 6 and all(c[2].k['camera_xyz'] == 'RDF' and c[3] for c in pin)
    K = pin[0][2].k['image_from_camera']
    assert K[0, 1] == 0 and K[2, 2] == 1
    moving = [c for c in logs if c[2].kind == 'Transform3D' and not c[3]]
    sc = RB.scene(ur5e)
    assert len(moving) == len(sc['robots'][0]['times']) * len(sc['robots'][0]['links'])
    assert any(c[0] == 'time' and c[1] == 'time' for c in calls)
    ep = [c for c in calls if c[0] == 'time' and c[1] == 'episode']
    assert [c[2]['sequence'] for c in ep] == [0, 3]
    assert {c[1] for c in logs if c[2].kind == 'TextLog'} == {'audit/verdict'}
    assert any(c[1] == 'audit/flags/desync' for c in logs)
    calls.clear()
    RB.stream(ur5e, 'connect', url='rerun+http://127.0.0.1:9876/proxy', meshes=False)
    assert ('connect', 'rerun+http://127.0.0.1:9876/proxy') in calls
    calls.clear()
    RB.stream(ur5e, 'spawn', meshes=False)
    assert ('spawn',) in calls


def test_missing_package_and_bad_inputs(ur5e, tmp_path, monkeypatch):
    monkeypatch.setattr(RB, 'available', lambda: False)
    with pytest.raises(RB.RerunError, match=r'pip install "dexvantage\[rerun\]"'):
        RB.stream(ur5e, 'save', str(tmp_path / 's.rrd'))
    monkeypatch.undo()
    with pytest.raises(RB.RerunError, match='mode'):
        RB.stream(ur5e, 'nope')
    with pytest.raises(RB.RerunError, match='does not exist'):
        RB.stream(ur5e, 'save', str(tmp_path / 'no' / 's.rrd'))
    with pytest.raises(RB.RerunError, match='.rrd'):
        RB.batch(ur5e, str(tmp_path / 's.txt'))
    bad = tmp_path / 'a.json'
    bad.write_text(json.dumps({'study': 'coverage'}))
    with pytest.raises(RB.RerunError, match='not an audit result'):
        RB.batch(ur5e, str(tmp_path / 's.rrd'), str(bad))


@pytest.mark.skipif(not HAVE_RERUN, reason='rerun-sdk is not installed (optional extra rerun)')
def test_rrd_read_back_by_rerun(ur5e, tmp_path):
    from dexvantage.validation import checks_rerun as CK
    out = CK.run_readback(ur5e, str(tmp_path))
    assert out.ok and out.error < 1e-4, out.detail
    assert out.detail['cameras'] == 6 and out.detail['audit_flags_on_timeline']
    rb = RB.read_back(str(tmp_path / 'scene.rrd'))
    vol = rb['/world/task_volumes/Tabletop_sample']['static']
    c = np.array(RB.first(vol['Boxes3D:centers']), float)
    assert np.allclose(c, ur5e.workspace.volumes[0].center, atol=1e-6)


@pytest.mark.skipif(not HAVE_RERUN, reason='rerun-sdk is not installed (optional extra rerun)')
def test_batch_flag(tmp_path):
    from test_batch import _run
    out = tmp_path / 'scene.rrd'
    r = _run('--sample', 'ur5e', '--rerun', str(out))
    assert r.returncode == 0, r.stderr
    assert 'Sent 1 robot' in r.stdout and out.stat().st_size > 1000
    assert '/world' in RB.read_back(str(out))
    r = _run('--sample', 'ur5e', '--rerun', str(tmp_path / 'x.txt'))
    assert r.returncode == 2 and '.rrd' in r.stderr and 'Traceback' not in r.stderr


def test_window_command(app, win, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    from dexvantage.ui import rerun_stream as RS
    a = win.actions_by_id['stream_rerun']
    assert a.isEnabled() and a in win.menus['Export'].actions()
    assert win.open_sample('ur5e')
    shown = []
    monkeypatch.setattr(QMessageBox, 'information', lambda *a, **k: shown.append(a[2]))
    monkeypatch.setattr(RB, 'available', lambda: False)
    assert not RS.stream(win, 'save', str(tmp_path / 'x.rrd'))
    assert shown and 'pip install' in shown[0]
    monkeypatch.undo()
    if not HAVE_RERUN:
        return
    path = str(tmp_path / 'w.rrd')
    assert RS.stream(win, 'save', path)
    t0 = time.time()
    while 'rerun' in win._jobs and time.time() - t0 < 60:
        pump(app, 0.05)
    pump(app, 0.1)
    assert win.last_rerun['path'] == path
    assert '/world' in RB.read_back(path)
