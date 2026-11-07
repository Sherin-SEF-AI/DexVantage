"""Time sync (core/studies/time_sync.py), project.rig['sync'] (core/project.py sync_from_dict and both schema copies),
the Time sync page (ui/pages/time_sync.py) and its validation checks (validation/checks_sync.py)."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from dexvantage.core import jobs
from dexvantage.core.project import Project, rig_from_dict, sync_from_dict
from dexvantage.core.studies import motion_blur as MB
from dexvantage.core.studies import time_sync as TS

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


@pytest.fixture(scope='module')
def ur5e():
    from dexvantage.core.robot.samples import sample_project
    return sample_project('ur5e')


@pytest.fixture(scope='module')
def doc(ur5e):
    return jobs.run_study('time_sync', ur5e)


def _hosted(p):
    """p with both sensors on one USB 3.2 Gen 1 controller, two 848 × 480 16-bit streams each."""
    q = p.snapshot()
    ids = ['depth|848x480|30|Z16', 'RGB|848x480|30|YUY2']
    q.rig = rig_from_dict({'hosts': [{'uid': 'h1', 'name': 'Host', 'library_id': 'usb-3.2-gen1-host',
                                      'controllers': [{'uid': 'c1', 'name': 'USB 3', 'kind': 'usb3'}]}],
                           'links': [{'sensor_uid': s.uid, 'stream_ids': ids, 'via': 'c1'} for s in q.variant.sensors]})
    return q


def _stream(d, name):
    return next(s for s in d['streams'] if s['name'] == name)


# ====================================================================== the rig's sync section
def test_sync_is_kept_normalised_and_validates(tmp_path):
    assert 'sync' not in rig_from_dict({})                          # an untouched rig stays as it was
    assert Project.from_dict({'format': 'dexvantage.project', 'version': 1}).rig == {'hosts': [], 'hubs': [],
                                                                                    'links': []}
    raw = {'clocks': [{'uid': 'k1', 'name': 'PTP grandmaster', 'kind': 'ptp',
                       'offset_s': {'value': '2e-6', 'source': 'ptp4l log'}, 'jitter_s': {'value': -1},
                       'drift_ppm': None},
                      {'kind': 'sundial'}, 7],
           'streams': {'abc': {'clock': 'k1', 'trigger': 'hardware'}, 'joint_states': {'trigger': 'magic'},
                       'bad': 3, '': {'clock': 'x'}}}
    s = sync_from_dict(raw)
    assert len(s['clocks']) == 2 and s['clocks'][0]['offset_s'] == {'value': 2e-6, 'source': 'ptp4l log'}
    assert s['clocks'][0]['jitter_s'] is None and s['clocks'][0]['drift_ppm'] is None   # negative jitter: dropped
    assert s['clocks'][1]['kind'] == 'host' and s['clocks'][1]['uid']
    assert s['streams'] == {'abc': {'clock': 'k1', 'trigger': 'hardware'},
                            'joint_states': {'clock': '', 'trigger': 'free'}}
    rig = rig_from_dict({'sync': raw})
    assert rig['sync'] == sync_from_dict(rig['sync'])
    from dexvantage.core.robot.samples import sample_project
    p = sample_project('so100')
    p.rig = rig
    path = str(tmp_path / 's.dexv')
    p.save(path)
    q = Project.load(path)
    assert q.rig['sync'] == rig['sync']
    jsonschema = pytest.importorskip('jsonschema')
    with open(os.path.join(ROOT, 'docs', 'project.schema.json')) as fh:
        jsonschema.validate(json.loads(json.dumps(q.to_dict())), json.load(fh))
    with open(os.path.join(ROOT, 'docs', 'project.schema.json')) as a, \
            open(os.path.join(ROOT, 'dexvantage', 'resources', 'project.schema.json')) as b:
        assert a.read() == b.read()


# ====================================================================== the model
def test_label_error_of_a_straight_line():
    t = np.linspace(0.0, 2.0, 2001)
    P = np.c_[0.25 * t, np.zeros_like(t), np.zeros_like(t)]        # 0.25 m/s along x
    for d in (0.004, -0.010, 0.0):
        e, _t = TS.label_error(t, P, d)
        assert e == pytest.approx(0.25 * abs(d), abs=1e-12)


def _cam(ur5e, name):
    return next(c for c in MB.imaging(ur5e) if c['name'] == name)


def test_stamp_budget_terms(ur5e):
    st = TS.settings_of(ur5e)
    c = _cam(ur5e, 'Overhead D435 · RGB')                          # rolling, no exposure set, readout not stated
    xfer = {'min_s': 0.001, 'max_s': 0.004, 'source': 'test'}
    b = TS.stamp_budget(c, {'clock': '', 'trigger': 'free'}, {}, xfer, st, False)
    period = 1 / 30
    assert b.lo == pytest.approx(0.001) and b.hi == pytest.approx(period / 2 + period / 2 + 0.004 + 125e-6)
    assert any('processing' in m for m in b.missing)
    # a hardware trigger the maker states for this imager: the stamp is the trigger edge
    ir = _cam(ur5e, 'Overhead D435 · IR left')                     # global, exposure 1 ms (stated) to 33 ms
    b = TS.stamp_budget(ir, {'clock': '', 'trigger': 'hardware'}, {}, xfer, st, True)
    assert b.lo == pytest.approx(-period / 2) and b.hi == pytest.approx(-0.0005)
    # one the maker says it does not take: it free-runs, said
    b = TS.stamp_budget(c, {'clock': '', 'trigger': 'hardware'}, {}, xfer, st, False)
    assert any('takes no hardware trigger' in m for m in b.missing) and b.lo == pytest.approx(0.001)
    # a device clock with its figures
    clk = {'uid': 'k', 'name': 'Cam clock', 'kind': 'camera_hw', 'offset_s': {'value': 0.001, 'source': 'm'},
           'jitter_s': {'value': 0.0002, 'source': 'm'}, 'drift_ppm': {'value': 10.0, 'source': 'm'}}
    b = TS.stamp_budget(ir, {'clock': 'k', 'trigger': 'free'}, {'k': clk}, xfer, st, True)
    drift = 10e-6 * st['episode_s']
    assert b.lo == pytest.approx(-period / 2 + 0.001 - 0.0002) and b.hi == pytest.approx(period / 2 + 0.001 + 0.0002
                                                                                         + drift)
    assert not b.missing
    clk2 = dict(clk, kind='ptp', offset_s=None)
    b = TS.stamp_budget(ir, {'clock': 'k', 'trigger': 'free'}, {'k': clk2}, xfer, st, True)
    assert any('offset not stated' in m for m in b.missing)


def test_joint_budget_pairing(ur5e):
    p = ur5e.snapshot()
    p.analysis.studies['time_sync'] = {'pairing': 'nearest'}
    b = TS.joint_budget({'clock': '', 'trigger': 'free'}, {}, TS.settings_of(p))
    assert any('rate' in m for m in b.missing) and any('latency' in m for m in b.missing)
    p.analysis.studies['time_sync'] = {'pairing': 'nearest', 'joint_rate_hz': {'value': 500, 'source': 'measured'},
                                       'latency_s': {'joint_states': {'value': 0.0015, 'source': 'measured'}}}
    b = TS.joint_budget({'clock': '', 'trigger': 'free'}, {}, TS.settings_of(p))
    assert not b.missing and b.lo == pytest.approx(0.0015 - 0.001) and b.hi == pytest.approx(0.0015 + 0.001)


# ====================================================================== the study
def test_study_on_the_ur5e_sample(doc):
    text = json.dumps(doc, allow_nan=False)
    assert doc['study'] == 'time_sync' and '/home/' not in text
    assert doc['skew_limit_s'] == 0.002 and doc['skew_limit_default']
    assert any('No clocks or triggers are set' in n for n in doc['notes'])
    assert doc['motions']['source'] == 'trajectories' and doc['motions']['trajectories'][0]['peak_tcp_speed_m_s'] > 0
    for s in doc['streams']:
        assert s['clock'] == 'host arrival (default)' and s['trigger'] == 'free' and not s['complete']
        assert s['skew_low_s'] > 0 and s['worst_skew_s'] == pytest.approx(max(abs(s['skew_low_s']),
                                                                             abs(s['skew_high_s'])))
        assert s['label_error_m'] > 0 and s['label_error_worst']['tcp_at_stamp']
        assert any('not assigned' in m for m in s['stamp']['not_stated'])       # no host on the Bandwidth page
    assert _stream(doc, 'Wrist D405 · IR left')['hardware_trigger'] is False
    assert _stream(doc, 'Overhead D435 · IR left')['hardware_trigger'] is True
    assert _stream(doc, 'Overhead D435 · RGB')['hardware_trigger'] is False     # only the depth sensor
    sm = doc['summary']
    assert sm['status'] == 'serious' and sm['word'] == 'out of sync' and not sm['complete']     # a lower bound
    sg = doc['suggestion']
    assert sg['kind'] == 'trigger' and sg['sensor'] == 'Overhead D435' and sg['key'].endswith('|IR left')
    assert sg['after']['worst_skew_s'] < sg['before']['worst_skew_s']
    assert '3.7.4' in sg['text']
    (pr,) = doc['pairs']
    assert not pr['locked'] and pr['phase_s'] == pytest.approx(1 / 60, abs=1e-9) and not pr['cable_possible']


def test_apply_the_trigger_and_measured_figures(ur5e, doc):
    p = ur5e.snapshot()
    TS.apply_suggestion(p, doc['suggestion'])
    uid = doc['suggestion']['sensor_uid']
    assert p.rig['sync']['streams'][uid] == {'clock': '', 'trigger': 'hardware'}
    d2 = jobs.run_study('time_sync', p)
    ir = _stream(d2, 'Overhead D435 · IR left')
    assert ir['trigger'] == 'hardware' and ir['worst_skew_s'] == pytest.approx(
        doc['suggestion']['after']['worst_skew_s'])
    assert ir['skew_high_s'] < 0                                     # the trigger edge comes before the exposure centre
    rgb = _stream(d2, 'Overhead D435 · RGB')
    assert any('takes no hardware trigger' in m for m in rgb['stamp']['not_stated'])
    assert d2['suggestion']['kind'] == 'advice' and 'D405' in d2['suggestion']['text']
    # with the topology, fixed exposures and measured latencies the budget is complete
    q = _hosted(ur5e)
    q.analysis.studies['motion_blur'] = {'exposure_s': {s.uid: {'value': 0.001, 'source': 'set'}
                                                        for s in q.variant.sensors}}
    q.analysis.studies['time_sync'] = {'latency_s': {
        **{s.uid: {'value': 0.0012, 'source': 'metadata'} for s in q.variant.sensors},
        'joint_states': {'value': 0.0002, 'source': 'measured'}}}
    d3 = jobs.run_study('time_sync', q)
    for s in d3['streams']:
        assert s['complete'] and s['skew_low_s'] == pytest.approx(0.001) and s['status'] == 'good'
        assert s['word'] == 'in sync'
    assert d3['summary']['complete'] and d3['summary']['status'] == 'good' and d3['suggestion'] is None


def test_transfer_from_the_topology(ur5e):
    q = _hosted(ur5e)
    cams = MB.imaging(q)
    x = TS.transfer_times(q, cams)
    bits = 848 * 480 * 16 * 1.05
    for c in cams:
        assert x[c['key']]['min_s'] == pytest.approx(bits / 3600e6)
        assert x[c['key']]['max_s'] == pytest.approx(4 * bits / 3600e6)
    unhosted = TS.transfer_times(ur5e, MB.imaging(ur5e))
    assert all(v['max_s'] is None and 'not assigned' in v['max_why'] for v in unhosted.values())


def test_refusals_and_batch(ur5e, tmp_path):
    p = ur5e.snapshot()
    p.variant.sensors = []
    with pytest.raises(jobs.StudyError, match='no camera'):
        jobs.run_study('time_sync', p)
    assert jobs.STUDIES['time_sync'].batch == 'time-sync'
    out = tmp_path / 'sync.json'
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    r = subprocess.run([sys.executable, '-m', 'dexvantage', '--sample', 'ur5e', '--time-sync', str(out)],
                       capture_output=True, text=True, env=env, cwd=ROOT, timeout=300)
    assert r.returncode == 0, r.stderr
    d = json.loads(out.read_text())
    assert d['study'] == 'time_sync' and 'provenance' in d


# ====================================================================== validation
def test_validation_checks_pass():
    from dexvantage.validation import checks_sync  # noqa: F401 — registers the checks
    from dexvantage.validation import suite as S
    ctx = S.Ctx(None, None, None, S.ValidationSettings(quick=True))
    ids = set()
    for spec in [s for s in S.CHECKS if s.id.startswith('sync.')]:
        out = spec.fn(ctx)
        ids.add(spec.id)
        assert out.ok and out.error <= (out.tolerance or spec.tolerance), (spec.id, out.error, out.detail)
    assert ids == {'sync.stamp_interval_sim', 'sync.action_label_sim'}


# ====================================================================== the page
def test_page_runs_applies_and_edits_the_rig(app, win):
    from conftest import pump
    from dexvantage.core.robot.samples import sample_project
    win.store.set_project(sample_project('ur5e'))
    win.ctx.analysis.wait(60)
    assert 'time_sync' in win.pages
    win.open_page('time_sync')
    page = win.pages['time_sync']
    page.run()
    win.ctx.analysis.wait(120)
    pump(app, 0.1)
    assert page.doc is not None and page.head.figure.text().startswith('≥ ')
    assert page.streams.topLevelItemCount() == 4 and page.budget.topLevelItemCount() > 0
    assert 'Not counted' in page.missing.text() and page.b_apply.isEnabled()
    assert page.overlays(win.viewport)
    # Apply: the hardware trigger, one undo step on the rig
    n = win.store.undo.count()
    assert page.apply()
    assert win.store.undo.count() == n + 1
    assert any(v['trigger'] == 'hardware' for v in win.store.project.rig['sync']['streams'].values())
    win.ctx.analysis.wait(120)
    pump(app, 0.1)
    win.store.undo.undo()
    assert 'sync' not in win.store.project.rig
    # a hardware trigger on the D405 is refused with Intel's words
    d405 = next(s for s in win.store.project.variant.sensors if s.name == 'Wrist D405')
    assert page.set_stream(d405.uid, '', 'hardware') is False
    assert page.set_stream(d405.uid, '', 'sync_cable')
    assert win.store.project.rig['sync']['streams'][d405.uid]['trigger'] == 'sync_cable'
    # clocks: add with figures, use, remove (each one undo step)
    n = win.store.undo.count()
    uid = page.add_clock('Robot controller', 'robot', {'offset_s': {'value': 0.0005, 'source': 'measured'}})
    assert win.store.undo.count() == n + 1 and win.store.project.rig['sync']['clocks'][0]['uid'] == uid
    assert page.set_stream('joint_states', uid, 'free')
    assert page.remove_clock(uid)
    assert win.store.project.rig['sync']['clocks'] == []
    assert win.store.project.rig['sync']['streams']['joint_states']['clock'] == ''
    assert page.set_joint_rate(500.0, 'controller manual') and page.settings()['joint_rate_hz']['value'] == 500.0
    assert page.set_joint_rate(-1.0, '') is False
    win.ctx.analysis.wait(120)
    pump(app, 0.1)
    page.run()
    win.ctx.analysis.wait(120)
    pump(app, 0.1)
    # the Camera view draws the worst label-error segment from a camera that sees it
    from dexvantage.core import camrender as CR
    from dexvantage.core import library as LB
    over = next(s for s in win.store.project.variant.sensors if s.name == 'Overhead D435')
    im = LB.sensor_imagers(over)[0]
    it = next(page.streams.topLevelItem(i) for i in range(page.streams.topLevelItemCount())
              if page.streams.topLevelItem(i).text(0).startswith('Overhead D435'))
    page.streams.setCurrentItem(it)
    ov = page.camview_overlays(im.camera, CR.optical_pose(win.store.project, over, im))
    assert ov and ov[0][0][0].shape == (2, 2) and 'action-label error' in ov[0][2]
    assert page.show_worst()
    assert 'Known limits' in page.explain_text()
