"""The bandwidth, hosts and power study (core/studies/bandwidth.py), the project's host topology (project.rig), its
page and its validation checks."""
import json
import os
import subprocess
import sys

import pytest

from conftest import pump
from dexvantage.core import jobs
from dexvantage.core import library as LB
from dexvantage.core.project import Project, Sensor, new_uid, rig_from_dict
from dexvantage.core.studies import bandwidth as BW

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def _project(*models, name='Rig'):
    p = Project(name)
    for k, (mid, kind) in enumerate(models):
        p.variant.sensors.append(Sensor(name=f'S{k}', kind=kind, model_id=mid, slot=k))
    return p


def _host(p, *kinds, library_id=''):
    cs = [{'uid': new_uid(), 'name': f'C{k}', 'kind': kind, 'practical_mbps': None, 'source': ''}
          for k, kind in enumerate(kinds)]
    p.rig['hosts'].append({'uid': new_uid(), 'name': 'Host', 'library_id': library_id, 'controllers': cs})
    return [c['uid'] for c in cs]


def _link(p, s, via, ids=()):
    p.rig['links'].append({'sensor_uid': s.uid, 'stream_ids': list(ids), 'via': via, 'cable_uid': ''})


def _hub(p, upstream, powered):
    u = new_uid()
    p.rig['hubs'].append({'uid': u, 'name': 'Hub 1', 'powered': powered, 'upstream': upstream})
    return u


def _sev(res, kind):
    return [f['severity'] for f in res['findings'] if f['kind'] == kind]


# ====================================================================== streams and rates
def test_stream_ids_round_trip_and_reject_junk():
    s = {'imager': 'depth', 'width': 848, 'height': 480, 'format': 'Z16'}
    sid = BW.stream_id(s, 30)
    assert sid == 'depth|848x480|30|Z16'
    assert BW.parse_stream_id(sid) == ('depth', 848, 480, 30.0, 'Z16')
    assert BW.parse_stream_id('RGB|640x480|15|') == ('RGB', 640, 480, 15.0, '')
    for bad in ('', 'depth|848x480|30', 'depth|848by480|30|Z16', 'depth|848x480|-1|Z16', '|848x480|30|Z16'):
        assert BW.parse_stream_id(bad) is None


def test_payload_is_width_height_fps_bits():
    assert BW.stream_payload_mbps(848, 480, 30, 16) == pytest.approx(848 * 480 * 30 * 16 / 1e6)
    assert BW.stream_payload_mbps(1920, 1080, 30, None) is None


def test_default_streams_depth_and_rgb_at_30_fps():
    p = _project(('intel-realsense-d435', 'depth'))
    s = p.variant.sensors[0]
    ids = BW.default_stream_ids(s, LB.get(s.model_id), 30.0)
    assert ids == ['depth|1280x720|30|Z16', 'RGB|1920x1080|30|YUY2']
    s.stream = [848, 480]                              # the stream chosen on the Sensors page
    ids = BW.default_stream_ids(s, LB.get(s.model_id), 30.0)
    assert ids == ['depth|848x480|30|Z16', 'RGB|848x480|30|YUY2']
    r = BW.sensor_streams(s, LB.get(s.model_id), [], BW.settings_of(p))
    assert r['default'] and r['payload_mbps'] == pytest.approx(848 * 480 * 30 * 32 / 1e6)


def test_default_streams_of_a_stereo_camera_without_a_depth_stream_send_both_images():
    p = _project(('stereolabs-zed-2i', 'depth'))
    s = p.variant.sensors[0]
    ids = BW.default_stream_ids(s, LB.get(s.model_id), 30.0)
    assert [i.split('|')[0] for i in ids] == ['Left', 'Right']
    assert all(i.split('|')[1:3] == ['2208x1242', '15'] for i in ids)       # the largest mode lists only 15 fps


def test_nearest_listed_rate_when_30_is_not_listed():
    p = _project(('leopard-li-ar0234cs-gmsl2-owl', 'camera'))
    s = p.variant.sensors[0]
    assert BW.default_stream_ids(s, LB.get(s.model_id), 30.0) == ['Colour|1920x1200|120|RAW10']


def test_not_stated_rates_stay_not_stated_until_measured():
    p = _project(('meta-digit', 'tactile'), ('gelsight-mini', 'tactile'), ('e-con-see3cam-24cug', 'camera'),
                 ('', 'camera'))
    st = BW.settings_of(p)
    digit, gel, cam, nodev = p.variant.sensors
    r = BW.sensor_streams(digit, LB.get('meta-digit'), [], st)
    assert r['payload_mbps'] is None and r['rate_source'] == 'not stated' and r['not_stated']
    assert BW.sensor_streams(gel, LB.get('gelsight-mini'), [], st)['payload_mbps'] is None
    mj = BW.sensor_streams(cam, LB.get('e-con-see3cam-24cug'), ['Colour|1920x1080|60|MJPEG'], st)
    assert mj['payload_mbps'] is None
    assert BW.sensor_streams(nodev, None, [], st)['rate_source'] == 'no device chosen'
    p.analysis.studies['bandwidth'] = {'measured': {digit.uid: {'mbps': 74.0, 'source': 'usbtop, 2026-11-06'}},
                                       'extra': {cam.uid: [{'name': 'IMU', 'mbps': 0.5, 'source': 'measured'}]}}
    res = BW.budget(p)
    rows = {r['uid']: r for r in res['sensors']}
    assert rows[digit.uid]['payload_mbps'] == 74.0 and rows[digit.uid]['rate_source'].startswith('measured')
    assert rows[cam.uid]['payload_mbps'] == pytest.approx(1920 * 1200 * 55 * 16 / 1e6 + 0.5)
    assert set(res['headline']['incomplete_sensors']) == {gel.uid, nodev.uid}
    assert any(f['kind'] == 'rate' and gel.uid in f['uids'] for f in res['findings'])


def test_stream_not_listed_is_reported_not_used():
    p = _project(('intel-realsense-d435', 'depth'))
    (c,) = _host(p, 'usb3')
    s = p.variant.sensors[0]
    _link(p, s, c, ['depth|1280x720|90|Z16', 'RGB|1920x1080|30|YUY2', 'nonsense'])
    r = BW.budget(p)['sensors'][0]
    assert len(r['streams']) == 1 and r['payload_mbps'] == pytest.approx(1920 * 1080 * 30 * 16 / 1e6)
    assert len(r['stream_problems']) == 2 and 'not 90' in r['stream_problems'][0]


@pytest.mark.parametrize('text,gen', [('USB 3.1 Gen 1', 'gen1'), ('USB 3.0 (USB Type-C)', 'gen1'),
                                      ('USB 3.2 Gen 1 (USB Type-C)', 'gen1'),
                                      ('USB 3.2 Gen 2x2 (USB Type-C)', 'gen2x2'),
                                      ('USB 3.2 Gen 2 / Gen 1, PCIe', 'gen2'), ('USB 2.0', 'usb2'),
                                      ('USB (Micro-B receptacle on the sensor)', None), ('GMSL2 (FAKRA)', None),
                                      ('USB 3.0 (SuperSpeed, nominal max. 5 Gbit/s); USB3 Vision', 'gen1')])
def test_usb_generation(text, gen):
    assert BW.usb_generation(text) == gen


def test_every_library_device_with_streams_resolves_its_default_streams():
    for e in LB.entries():
        if e.kind not in ('camera', 'depth', 'tactile'):
            continue
        s = Sensor(name='x', kind=e.kind, model_id=e.id)
        for sid in BW.default_stream_ids(s, e, 30.0):
            st, fps, why = BW.resolve_stream(e, sid)
            assert st is not None, (e.id, sid, why)


# ====================================================================== the budget
def test_ur5e_sample_has_no_host_then_the_suggestion_adds_one():
    from dexvantage.core.robot.samples import sample_project
    p = sample_project('ur5e')
    res = BW.budget(p)
    assert res['headline']['status_word'] == 'no hosts' and res['suggestion']['kind'] == 'add_host'
    BW.apply_suggestion(p, res['suggestion'])
    res = BW.budget(p)
    (c,) = res['controllers']
    assert c['generation'] == 'gen1' and c['capacity_mbps'] == 3600.0 and not c['capacity_stated']
    want = (1280 * 720 * 30 * 32 + 1280 * 720 * 30 * 16 + 1920 * 1080 * 30 * 16) / 1e6 * 1.05
    assert c['load_mbps'] == pytest.approx(want)
    assert res['headline']['status'] == 'good' and res['headline']['busiest_load_pct'] == pytest.approx(
        want / 3600 * 100, abs=0.01)
    d435 = next(r for r in res['sensors'] if r['model_id'] == 'intel-realsense-d435')
    assert d435['draw_ma'] == pytest.approx(3.40297 / 5 * 1000) and d435['allowance_ma'] == 900.0
    assert res['suggestion'] is None
    json.dumps(res, allow_nan=False)


def test_over_budget_controller_and_link_and_the_move():
    p = _project(*[('intel-realsense-d435', 'depth')] * 3)
    c2, c3 = _host(p, 'usb2', 'usb3')
    big = ['depth|1280x720|30|Z16', 'RGB|1920x1080|30|YUY2']
    for s in p.variant.sensors[:2]:
        _link(p, s, c2, big)
    res = BW.budget(p)
    cr = {c['uid']: c for c in res['controllers']}
    assert cr[c2]['status'] == 'critical' and cr[c2]['capacity_mbps'] == pytest.approx(425.984)
    assert all(r['link_status'] == 'critical' for r in res['sensors'][:2])
    assert 'critical' in _sev(res, 'controller') and 'critical' in _sev(res, 'link')
    assert res['headline']['status_word'] == 'over budget'
    sug = res['suggestion']                            # the third sensor is not assigned: assign it first
    assert sug['kind'] == 'assign' and sug['moves'][0]['to'] == c3
    BW.apply_suggestion(p, sug)
    sug = BW.budget(p)['suggestion']
    assert sug['kind'] == 'move' and sug['moves'][0]['to'] == c3
    BW.apply_suggestion(p, sug)
    after = {c['uid']: c for c in BW.budget(p)['controllers']}
    assert after[c2]['load'] < cr[c2]['load']


def test_tight_controller_reads_tight_with_its_margin():
    p = _project(('intel-realsense-d435', 'depth'), ('intel-realsense-d435', 'depth'))
    (c,) = _host(p, 'usb3')
    for s in p.variant.sensors:
        _link(p, s, c, ['depth|1280x720|30|Z16', 'RGB|1920x1080|30|YUY2', 'IR left|1280x720|30|Y8'])
    res = BW.budget(p)
    (cr,) = res['controllers']
    assert 0.8 < cr['load'] < 1.0 and cr['status'] == 'warning'
    assert cr['margin_mbps'] == pytest.approx(3600 - cr['load_mbps'])
    assert res['headline']['status_word'] == 'tight'


def test_measured_practical_figure_replaces_the_spec_bound():
    p = _project(('intel-realsense-d435', 'depth'))
    (c,) = _host(p, 'usb3')
    p.rig['hosts'][0]['controllers'][0].update(practical_mbps=1000.0, source='measured with a bulk test, 2026-11-06')
    _link(p, p.variant.sensors[0], c)
    res = BW.budget(p)
    assert res['controllers'][0]['capacity_mbps'] == 1000.0 and res['controllers'][0]['capacity_stated']
    assert res['sensors'][0]['link_capacity_mbps'] == 1000.0
    assert res['controllers'][0]['status'] == 'critical'


def test_host_entry_sets_the_controller_generation():
    p = _project(('intel-realsense-d435', 'depth'))
    (c,) = _host(p, 'usb3', library_id='usb-3.2-gen2-host')
    _link(p, p.variant.sensors[0], c)
    res = BW.budget(p)
    assert res['controllers'][0]['generation'] == 'gen2' and res['controllers'][0]['capacity_mbps'] == 8800.0
    assert res['sensors'][0]['link_capacity_mbps'] == 3600.0          # the D435 itself is Gen 1


def test_hub_on_shared_controller_and_uplink():
    p = _project(('intel-realsense-d405', 'depth'), ('intel-realsense-d435', 'depth'))
    (c,) = _host(p, 'usb3')
    h = _hub(p, c, powered=True)
    a, b = p.variant.sensors
    _link(p, a, h)
    _link(p, b, c)
    res = BW.budget(p)
    hr = res['hubs'][0]
    assert hr['shared_controller'] and hr['sensors'] == [a.uid]
    assert 'warning' in _sev(res, 'hub')
    assert hr['load_mbps'] == pytest.approx(res['sensors'][0]['wire_mbps'])
    p.rig['links'] = [k for k in p.rig['links'] if k['sensor_uid'] != b.uid]
    _link(p, b, h)
    assert not BW.budget(p)['hubs'][0]['shared_controller']


def test_topology_problems_are_findings_not_crashes():
    p = _project(('intel-realsense-d457', 'depth'), ('intel-realsense-d435', 'depth'),
                 ('intel-realsense-d405', 'depth'))
    cu, cg = _host(p, 'usb3', 'gmsl')
    d457, d435, d405 = p.variant.sensors
    _link(p, d457, cu)                                 # a GMSL camera on a USB controller
    _link(p, d435, 'gone')                             # plugged into nothing the rig has
    h1, h2 = new_uid(), new_uid()
    p.rig['hubs'] += [{'uid': h1, 'name': 'A', 'powered': False, 'upstream': h2},
                      {'uid': h2, 'name': 'B', 'powered': False, 'upstream': h1}]
    _link(p, d405, h1)                                 # a hub loop
    res = BW.budget(p)
    kinds = {(f['kind'], f['severity']) for f in res['findings']}
    assert ('mismatch', 'serious') in kinds and ('topology', 'serious') in kinds
    assert any('loop' in f['message'] for f in res['findings'])
    gm = next(c for c in res['controllers'] if c['uid'] == cg)
    assert gm['capacity_mbps'] is None and 'GMSL2' in gm['capacity_source']


def test_gmsl_camera_on_a_deserialiser_has_no_stated_capacity():
    p = _project(('intel-realsense-d457', 'depth'))
    (cg,) = _host(p, 'gmsl', library_id='leopard-li-jag-adp-gmsl2-8ch')
    _link(p, p.variant.sensors[0], cg)
    res = BW.budget(p)
    r = res['sensors'][0]
    assert r['payload_mbps'] > 0 and r['link_capacity_mbps'] is None and r['link_status'] == 'unknown'
    assert r['power_status'] == 'unknown' and 'not on a USB controller' in r['power_note']
    assert res['controllers'][0]['load_mbps'] == pytest.approx(r['wire_mbps'])


# ====================================================================== power
def test_bus_power_over_the_port_allowance():
    p = _project(('luxonis-oak-d', 'depth'), ('intel-realsense-d435', 'depth'))
    (c,) = _host(p, 'usb3')
    oak, d435 = p.variant.sensors
    _link(p, oak, c)
    _link(p, d435, c)
    res = BW.budget(p)
    rows = {r['uid']: r for r in res['sensors']}
    assert rows[oak.uid]['draw_ma'] == pytest.approx(1500.0) and rows[oak.uid]['power_status'] == 'critical'
    assert rows[d435.uid]['power_status'] == 'good'
    p.analysis.studies['bandwidth'] = {'external_power': [oak.uid]}
    assert {r['uid']: r for r in BW.budget(p)['sensors']}[oak.uid]['power_status'] == 'good'


def test_bus_powered_hub_shares_its_upstream_allowance():
    p = _project(('intel-realsense-d435', 'depth'), ('intel-realsense-d435', 'depth'))
    (c,) = _host(p, 'usb3')
    h = _hub(p, c, powered=False)
    for s in p.variant.sensors:
        _link(p, s, h)
    res = BW.budget(p)
    hr = res['hubs'][0]
    assert hr['draw_ma'] == pytest.approx(2 * 3.40297 / 5 * 1000) and hr['available_ma'] == 900.0
    assert hr['power_status'] == 'critical'
    assert all(r['power_status'] == 'critical' for r in res['sensors'])
    p.rig['hubs'][0]['powered'] = True
    res = BW.budget(p)
    assert all(r['power_status'] == 'good' and r['allowance_ma'] == 900.0 for r in res['sensors'])


def test_usb2_bus_powered_hub_gives_one_unit_load_per_port():
    p = _project(('e-con-see3cam-20cug', 'camera'))
    (c,) = _host(p, 'usb2')
    h = _hub(p, c, powered=False)
    _link(p, p.variant.sensors[0], h)
    r = BW.budget(p)['sensors'][0]
    assert r['allowance_ma'] == 100.0 and r['draw_ma'] == pytest.approx(0.69 / 5 * 1000)
    assert r['power_status'] == 'critical'


# ====================================================================== disk
def test_disk_budget_per_episode_and_campaign_and_write_check():
    p = _project(('intel-realsense-d435', 'depth'))
    (c,) = _host(p, 'usb3')
    _link(p, p.variant.sensors[0], c, ['depth|848x480|30|Z16'])
    pay = 848 * 480 * 30 * 16 / 1e6
    res = BW.budget(p)
    d = res['disk']
    mb_s = pay * 1.03 / 8
    assert d['recorded_mb_s'] == pytest.approx(mb_s)
    assert d['gb_per_episode'] == pytest.approx(mb_s * 60 / 1000)
    assert d['gb_per_campaign'] == pytest.approx(mb_s * 60 / 1000 * 100)
    assert 'note' in _sev(res, 'disk') and d['write_load'] is None
    p.analysis.studies['bandwidth'] = {'episode_s': 120, 'episodes': 500, 'record_ratio': 0.1,
                                       'recorder': {'write_mb_s': 2.0, 'capacity_gb': 10, 'source': 'test'}}
    res = BW.budget(p)
    d = res['disk']
    mb_s = pay * 0.1 * 1.03 / 8
    assert d['gb_per_episode'] == pytest.approx(mb_s * 120 / 1000)
    assert d['write_needed_mb_s'] == pytest.approx(mb_s * 1.3)
    assert 'critical' in _sev(res, 'disk')
    assert d['episodes_per_disk'] == int(10 // (mb_s * 120 / 1000)) and 'warning' in _sev(res, 'disk')


def test_settings_are_clamped_and_defaults_listed():
    p = Project()
    p.analysis.studies['bandwidth'] = {'margin_warn': 7, 'episodes': -3, 'episode_s': 'x', 'overhead': {'usb': 2.0}}
    st = BW.settings_of(p)
    assert st['margin_warn'] == 1.0 and st['episodes'] == 1 and st['episode_s'] == 60.0
    assert 'episode_s' in st['defaults'] and st['overhead']['usb']['value'] == 2.0


def test_empty_rig_is_a_state_not_an_error():
    doc = jobs.run_study('bandwidth', Project())     # live after every edit: must not fail on a blank project
    assert doc['headline']['status_word'] == 'no sensors' and 'no sensor' in doc['message']
    assert doc['suggestion'] is None and doc['sensors'] == []


# ====================================================================== the project's rig section
def test_rig_round_trips_through_the_file_and_validates(tmp_path):
    p = _project(('intel-realsense-d435', 'depth'))
    (c,) = _host(p, 'usb3')
    _link(p, p.variant.sensors[0], c, ['depth|848x480|30|Z16'])
    _hub(p, c, True)
    path = str(tmp_path / 'r.dexv')
    p.save(path)
    q = Project.load(path)
    assert q.rig == rig_from_dict(p.rig)
    assert Project.from_dict({'format': 'dexvantage.project', 'version': 1}).rig == {'hosts': [], 'hubs': [],
                                                                                    'links': []}
    junk = rig_from_dict({'hosts': [{'controllers': [{'kind': 'thunderbolt', 'practical_mbps': 'x'}, 3]}, 'x'],
                          'links': [{'via': 'a'}, {'sensor_uid': 's'}]})
    assert junk['hosts'][0]['controllers'][0]['kind'] == 'usb3'
    assert junk['hosts'][0]['controllers'][0]['practical_mbps'] is None and len(junk['links']) == 1
    jsonschema = pytest.importorskip('jsonschema', reason='jsonschema (the test extra) is not installed')
    with open(os.path.join(ROOT, 'docs', 'project.schema.json')) as fh:
        jsonschema.validate(json.loads(json.dumps(q.to_dict())), json.load(fh))


def test_rig_is_in_the_content_hash_only_when_set():
    from dexvantage.core import provenance as PV
    p = _project(('intel-realsense-d435', 'depth'))
    before = PV.project_digest(p)
    assert 'rig' not in before['sections']
    (c,) = _host(p, 'usb3')
    _link(p, p.variant.sensors[0], c)
    after = PV.project_digest(p)
    assert after['sha256'] != before['sha256'] and 'rig' in after['sections']


def test_registered_with_its_batch_flag_and_cached():
    jobs.load_studies()
    info = jobs.STUDIES['bandwidth']
    assert info.batch == 'bandwidth' and not info.needs_robot and info.quick
    p = _project(('intel-realsense-d435', 'depth'))
    cache = jobs.ResultCache()
    a = jobs.run_study('bandwidth', p, cache=cache)
    assert a['schema'] == 'dexvantage.bandwidth/1' and not a['cached']
    assert jobs.run_study('bandwidth', p, cache=cache)['cached']
    _host(p, 'usb3')
    assert not jobs.run_study('bandwidth', p, cache=cache)['cached']


def test_batch_flag_writes_the_result(tmp_path):
    out = tmp_path / 'bw.json'
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    r = subprocess.run([sys.executable, '-m', 'dexvantage', '--sample', 'ur5e', '--bandwidth', str(out)],
                       capture_output=True, text=True, env=env, cwd=ROOT, timeout=120)
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text())
    assert doc['study'] == 'bandwidth' and 'provenance' in doc and len(doc['sensors']) == 2
    assert doc['headline']['status_word'] == 'no hosts'


# ====================================================================== validation
def test_validation_checks_pass():
    from dexvantage.validation import suite
    import dexvantage.validation.checks_bandwidth  # noqa: F401
    r = suite.run(settings={'models': ['bandwidth'], 'quick': True})
    ids = {c.id: c for c in r.checks}
    assert set(ids) == {'bandwidth.stated_mode_rates', 'bandwidth.spec_figures', 'bandwidth.budget_recompute'}
    assert all(c.passed for c in r.checks), [(c.id, c.error, c.exception) for c in r.checks]


# ====================================================================== the page
def test_page_apply_is_one_undo_step_and_editor_plugs_sensors(win, app, tmp_path):
    from dexvantage.core.robot.samples import sample_project
    from dexvantage.ui.viewport3d import LineDraw
    win.store.set_project(sample_project('ur5e'))
    win.ctx.analysis.wait(10)
    page = win.pages['bandwidth']
    win.open_page('bandwidth')
    page.run()
    win.ctx.analysis.wait(30)
    pump(app, 0.05)
    assert page.doc is not None and page.head.status.text.text() == 'no hosts'
    assert page.b_apply.isEnabled()
    n = win.store.undo.count()
    page.apply()
    assert win.store.undo.count() == n + 1
    win.ctx.analysis.wait(30)
    pump(app, 0.1)
    page.run()
    win.ctx.analysis.wait(30)
    pump(app, 0.05)
    assert page.head.figure.text() not in ('—', '') and page.head.unit.text() == '%'
    assert page.chart.rows and page.sens.topLevelItemCount() == 2
    items = page.overlays(win.viewport)
    assert items and all(isinstance(i, LineDraw) for i in items)
    # the editor: add a second controller, plug one sensor into it, each one undo step
    host = win.store.project.rig['hosts'][0]
    page.topo.setCurrentItem(page.topo.topLevelItem(0))
    page.add_controller('usb3')
    assert len(win.store.project.rig['hosts'][0]['controllers']) == 2
    c2 = win.store.project.rig['hosts'][0]['controllers'][1]['uid']
    s = win.store.project.variant.sensors[0]
    n = win.store.undo.count()
    page.plug(s.uid, c2)
    assert win.store.undo.count() == n + 1
    assert next(k for k in win.store.project.rig['links'] if k['sensor_uid'] == s.uid)['via'] == c2
    win.store.undo.undo()
    assert next(k for k in win.store.project.rig['links'] if k['sensor_uid'] == s.uid)['via'] != c2
    page.add_hub(upstream=c2, powered=True)
    assert win.store.project.rig['hubs'][0]['powered']
    page.add_host('usb-3.2-gen2-host')
    assert win.store.project.rig['hosts'][-1]['library_id'] == 'usb-3.2-gen2-host'
    assert host['uid'] == win.store.project.rig['hosts'][0]['uid']
    page.preview()
    out = page.export(str(tmp_path / 'bw.json'))
    assert out and 'provenance' in json.loads(open(out).read())
    assert page.camview_overlays(None, None) == []
