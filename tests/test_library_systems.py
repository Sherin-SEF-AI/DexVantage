"""The library's systems entries (spec §6, §7.4, §1.1): hosts, policies, lights, cables and the teleop kind
(core/library, core/teleop_devices.py), the teleop page's library picker, and checks_library_systems.py.

The figures cannot be re-read offline; these tests pin each one to what its source states (the quote in the entry),
check the entries' provenance and shape, and check that a library teleop device reaches the teleop study as the
shared contract says (project.tasks.teleop_devices), with nothing invented on the way.
"""
from __future__ import annotations

import json
import os
import re

import pytest

from dexvantage.core import jobs
from dexvantage.core import library as L
from dexvantage.core import teleop_devices as TD
from dexvantage.core.studies import teleop as TS

LIB = L.BUILTIN_DIR
NEW = {
    'host': ('nvidia-jetson-agx-orin-32gb', 'nvidia-jetson-orin-nx-16gb', 'nvidia-jetson-orin-nx-8gb',
             'nvidia-jetson-orin-nano-4gb', 'nvidia-jetson-agx-thor-devkit', 'nvidia-jetson-t5000',
             'nvidia-jetson-t4000', 'basler-usb3-card-fl1100-4hc-x4', 'leopard-li-pciefg-gmsl2',
             'stereolabs-zed-link-quad'),
    'policy': ('pi05', 'gr00t-n1.5-3b', 'rdt-1b', 'openvla-oft-libero', 'openvla-oft-plus-aloha', 'smolvla-libero',
               'smolvla-metaworld', 'diffusion-policy-cnn-pusht', 'act-lerobot', 'hpt-base', 'octo-small'),
    'light': ('neewer-rl-18-55w', 'amaran-p60c', 'axis-t90d20-ir-led'),
    'cable': ('stereolabs-gmsl2-fakra-f-f', 'basler-cable-usb3-microb-sl-a-p', 'basler-cable-gige-cat6-rj45-drc-p'),
    'teleop': ('gello', 'so100-leader-arm', 'so101-leader-arm', 'koch-v1-1-leader-arm', 'aloha-widowx-250-leader',
               '3dconnexion-spacemouse-compact', 'meta-quest-3', 'meta-quest-3s', 'apple-vision-pro-m5',
               'manus-quantum-metagloves', 'rokoko-smartgloves-ii', 'haptx-gloves-g1',
               'ultraleap-leap-motion-controller-2'),
}
FILES = [(k, i) for k, ids in NEW.items() for i in ids]
FAST = {'voxel_mm': 40.0, 'search_voxels': 1, 'seeds': 3}


def _load(kind, i):
    with open(os.path.join(LIB, kind, i + '.json'), encoding='utf-8') as fh:
        return json.load(fh)


def _strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from _strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _strings(v)


@pytest.fixture(scope='module', autouse=True)
def _fresh_library():
    L.reload()
    yield


# ====================================================================== every new entry
@pytest.mark.parametrize('kind,i', FILES)
def test_entry_valid_named_sourced_and_clean(kind, i):
    d = _load(kind, i)
    assert d['id'] == i and d['kind'] == kind
    assert L.validate(d) == [] and L.validate_minimal(d) == []
    jsonschema = pytest.importorskip('jsonschema')
    jsonschema.Draft202012Validator(L.schema()).validate(d)
    assert d['sources']
    for s in d['sources']:
        assert s['url'].startswith('https://') and s['read'] == '2026-11-07'
        assert s.get('title', '').strip() and s.get('covers', '').strip()
    assert len(d['note']) > 80, 'the note says what is and is not stated'
    for s in _strings(d):
        assert '/' + 'home/' not in s and '/' + 'Users/' not in s and '/' + 'tmp/' not in s, s
        assert not re.search(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}', s), s
    e = L.get(i)
    assert e is not None and e.kind == kind and not e.is_custom


def test_library_loads_without_problems_and_has_the_teleop_kind():
    assert 'teleop' in L.KINDS and L.problems() == []
    assert {e.id for e in L.entries('teleop')} >= set(NEW['teleop'])
    assert L.schema()['properties']['kind']['enum'][-1] == 'teleop'
    # the schema's device kinds, the module's and the study's agree
    enum = L.schema()['properties']['teleop']['oneOf'][1]['properties']['device_kind']['enum']
    assert tuple(enum) == TD.KINDS == TS.KINDS


# ====================================================================== hosts
@pytest.mark.parametrize('i,ports,lanes,stated,practical,gmsl,power', [
    ('nvidia-jetson-agx-orin-32gb', 3, 16, 10000.0, None, None, 60.0),
    ('nvidia-jetson-orin-nx-16gb', 3, 8, 10000.0, None, None, 40.0),
    ('nvidia-jetson-orin-nx-8gb', 3, 8, 10000.0, None, None, 40.0),
    ('nvidia-jetson-orin-nano-4gb', 3, 8, 10000.0, None, None, 25.0),
    ('nvidia-jetson-agx-thor-devkit', 2, None, 10000.0, None, None, 130.0),
    ('nvidia-jetson-t5000', 3, 16, None, None, None, 130.0),
    ('nvidia-jetson-t4000', 3, 16, None, None, None, 70.0),
    ('basler-usb3-card-fl1100-4hc-x4', 4, 4, 5000.0, 3040.0, None, None),
    ('leopard-li-pciefg-gmsl2', 2, None, None, 51000.0, 8, None),
    ('stereolabs-zed-link-quad', 2, None, None, None, 8, 0.29),
])
def test_host_figures_as_stated(i, ports, lanes, stated, practical, gmsl, power):
    d = _load('host', i)
    h = d['host']
    assert (h['ports'], h['lanes'], h['stated_mbps'], h['practical_mbps'], h['gmsl_links'], d['power_w']) == \
        (ports, lanes, stated, practical, gmsl, power)
    assert h['ingest'] and all(g['what'] for g in h['ingest'])


def test_host_usb_generation_is_read_by_the_bandwidth_study():
    from dexvantage.core.studies import bandwidth as BW
    assert BW.host_generation(L.get('nvidia-jetson-agx-thor-devkit')) == 'gen2'
    assert BW.host_generation(L.get('basler-usb3-card-fl1100-4hc-x4')) == 'gen1'
    # the Thor module's USB 3.2 generation is not stated: no rate is entered for it
    assert L.get('nvidia-jetson-t5000').data['host']['stated_mbps'] is None


def test_no_desktop_gpu_is_entered_as_a_camera_host():
    """A GPU's stated figures (cores, memory, PCIe) are not camera-ingest figures; no host entry is a GPU."""
    for e in L.entries('host'):
        assert not re.search(r'\bRTX\b|GeForce', e.name), e.id


# ====================================================================== policies
@pytest.mark.parametrize('i,field,value', [
    ('pi05', 'input_resolution', [224, 224]), ('pi05', 'crop_rule', 'resize_pad'), ('pi05', 'action_chunk', 50),
    ('pi05', 'control_hz', 50), ('pi05', 'inference_ms', None),
    ('gr00t-n1.5-3b', 'crop_side_fraction', 0.95), ('gr00t-n1.5-3b', 'action_chunk', 16),
    ('gr00t-n1.5-3b', 'inference_ms', 47.88), ('gr00t-n1.5-3b', 'n_cameras', None),
    ('rdt-1b', 'input_resolution', [384, 384]), ('rdt-1b', 'action_chunk', 64), ('rdt-1b', 'inference_ms', 166.7),
    ('rdt-1b', 'control_hz', 25), ('rdt-1b', 'n_cameras', 3),
    ('openvla-oft-libero', 'action_chunk', 8), ('openvla-oft-libero', 'inference_ms', 112.0),
    ('openvla-oft-libero', 'n_cameras', 2), ('openvla-oft-plus-aloha', 'n_cameras', 3),
    ('openvla-oft-plus-aloha', 'action_chunk', 25), ('openvla-oft-plus-aloha', 'inference_ms', 321.0),
    ('openvla-oft-plus-aloha', 'control_hz', 25),
    ('smolvla-libero', 'n_cameras', 2), ('smolvla-metaworld', 'n_cameras', 1),
    ('smolvla-libero', 'crop_rule', 'resize_pad_top_left'),
    ('diffusion-policy-cnn-pusht', 'input_resolution', [84, 84]),
    ('diffusion-policy-cnn-pusht', 'pre_resize', [96, 96]),
    ('diffusion-policy-cnn-pusht', 'execute_steps', 8), ('diffusion-policy-cnn-pusht', 'inference_ms', None),
    ('act-lerobot', 'action_chunk', 100), ('act-lerobot', 'temporal_ensemble_m', False),
    ('act-lerobot', 'input_resolution', None),
    ('hpt-base', 'crop_rule', 'resize'), ('hpt-base', 'action_chunk', 8), ('hpt-base', 'inference_ms', 25.6),
    ('hpt-base', 'control_hz', 10),
    ('octo-small', 'inference_ms', 58.8), ('octo-small', 'tokens_per_image', 256),
])
def test_policy_figures_as_stated(i, field, value):
    assert L.get(i).data['policy'][field]['value'] == value


@pytest.mark.parametrize('i', NEW['policy'])
def test_policy_entries_are_consistent_and_usable(i):
    from dexvantage.core import policies as PO
    d = _load('policy', i)
    assert PO.entry_problems(d) == []
    p = d['policy']
    for k, f in p.items():
        if f['value'] is None:
            assert f['source'] is None, k
        else:
            assert isinstance(f['source'], int) and f.get('quote'), k
    t = p['inference_ms']
    if t['value'] is not None:
        assert t.get('hardware'), 'a stated inference time names its hardware'
    res = p['input_resolution']['value']
    if res is not None:
        g = PO.crop_geometry(p['crop_rule']['value'], (640, 480), res, p['crop_side_fraction']['value'],
                             p['pre_resize']['value'])
        assert 0 < g.used_fraction <= 1.0


# ====================================================================== lights and cables
def test_light_figures_and_nothing_derived():
    from dexvantage.core import lights as LT
    n = L.get('neewer-rl-18-55w').data['light']
    assert (n['type'], n['cri'], n['illuminance_lux'], n['illuminance_at_m'], n['luminous_flux_lm']) == \
        ('ring', 95, 5500.0, 0.5, None)
    a = L.get('amaran-p60c').data['light']
    assert (a['type'], a['beam_deg'], a['cri'], a['cct_k_range'], a['illuminance_lux']) == \
        ('spot', 45.0, 95, [2500.0, 7500.0], 6300.0)
    x = L.get('axis-t90d20-ir-led')
    assert (x.data['light']['wavelength_nm'], x.data['light']['beam_deg'], x.data['mass_g']) == (850.0, 10.0, 950.0)
    assert x.data['light']['ir_irradiance_w_m2'] is None            # not stated: not turned from the 144 m range
    for i in NEW['light']:
        lt = LT.from_library(i)
        assert lt['library_id'] == i and lt['luminous_flux_lm'] is None     # no flux is stated, none is made up
        assert lt['kind'] == L.get(i).data['light']['type']


def test_cable_figures_and_bend_kinds():
    from dexvantage.core import cables as CB
    s = L.get('stereolabs-gmsl2-fakra-f-f').data['cable']
    assert (s['type'], s['diameter_mm'], s['min_bend_static_mm'], s['min_bend_dynamic_mm'], s['flex_cycles'],
            s['stated_mbps']) == ('gmsl2', 3.3, 33.0, 66.0, 10000, 6000.0)
    b = L.get('basler-cable-usb3-microb-sl-a-p').data['cable']
    assert (b['min_bend_dynamic_mm'], b['flex_cycles'], b['robotics_rated'], b['drag_chain_rated']) == \
        (None, None, False, False)
    g = L.get('basler-cable-gige-cat6-rj45-drc-p').data['cable']
    assert (g['min_bend_dynamic_mm'], g['flex_cycles'], g['drag_chain_rated'], g['robotics_rated']) == \
        (60.0, 1000000, True, False)
    assert CB.from_library(L.get('basler-cable-usb3-microb-sl-a-p'))['min_bend_radius_m']['kind'] == 'static'
    assert CB.from_library(L.get('stereolabs-gmsl2-fakra-f-f'))['min_bend_radius_m']['kind'] == 'dynamic'


# ====================================================================== teleop entries
@pytest.mark.parametrize('i', NEW['teleop'])
def test_teleop_entry_consistent_and_says_what_is_missing(i):
    e = L.get(i)
    assert TD.entry_problems(e.data) == []
    t = e.data['teleop']
    # no maker states a workspace as a box or cylinder: it stays null, and the entry says so
    assert t['workspace']['value'] is None and t['workspace'].get('quote')
    for row in TD.figure_rows(e):
        if not row['stated']:
            assert row['source'].startswith(L.NOT_STATED) and row['text'] == '—'
        else:
            assert 'read 2026-11-07' in row['source']


@pytest.mark.parametrize('i,key,value', [
    ('gello', 'joint_resolution_deg', 0.088), ('gello', 'dof', None),
    ('so100-leader-arm', 'dof', 6), ('so101-leader-arm', 'dof', 6), ('koch-v1-1-leader-arm', 'dof', 6),
    ('aloha-widowx-250-leader', 'tracking_hz', 50), ('aloha-widowx-250-leader', 'reach_mm', 650),
    ('3dconnexion-spacemouse-compact', 'dof', 6), ('meta-quest-3', 'tracking_hz', None),
    ('apple-vision-pro-m5', 'latency_ms', 12), ('manus-quantum-metagloves', 'tracking_hz', 120),
    ('manus-quantum-metagloves', 'latency_ms', 7.5), ('rokoko-smartgloves-ii', 'tracking_hz', 250),
    ('rokoko-smartgloves-ii', 'accuracy_mm', 0.47), ('haptx-gloves-g1', 'dof', 36),
    ('haptx-gloves-g1', 'position_resolution_mm', 0.3), ('haptx-gloves-g1', 'latency_ms', None),
    ('ultraleap-leap-motion-controller-2', 'tracking_hz', 115),
])
def test_teleop_figures_as_stated(i, key, value):
    assert TD.figure(L.get(i), key)[0] == value


def test_teleop_masses_and_interfaces():
    m = {i: L.get(i).data['mass_g'] for i in NEW['teleop']}
    assert (m['3dconnexion-spacemouse-compact'], m['meta-quest-3'], m['meta-quest-3s'], m['apple-vision-pro-m5'],
            m['manus-quantum-metagloves'], m['haptx-gloves-g1'], m['ultraleap-leap-motion-controller-2']) == \
        (480.0, 515.0, 514.0, 800.0, 138.0, 454.0, 29.0)
    assert m['so100-leader-arm'] is None and m['gello'] is None           # not stated


def test_teleop_entries_are_not_sensors_or_hosts():
    from dexvantage.core.project import SENSOR_KINDS
    for i in NEW['teleop']:
        e = L.get(i)
        assert e.kind not in SENSOR_KINDS and e.imagers == [] and L.sensor_imagers(type('S', (), {
            'model_id': i})()) == []


def test_entry_problems_catch_bad_teleop_entries():
    bad = json.loads(json.dumps(L.get('so100-leader-arm').data))
    bad['teleop']['mapping'] = 'cartesian'
    bad['teleop']['dof'] = {'value': 6, 'source': 5}
    bad['teleop']['tracking_hz'] = {'value': None, 'source': 0}
    bad['teleop']['latency_ms'] = {'value': -3, 'source': 0, 'quote': 'x'}
    del bad['teleop']['feedback']
    msgs = ' '.join(TD.entry_problems(bad))
    for word in ('teleop.mapping', 'teleop.dof', 'teleop.tracking_hz', 'teleop.latency_ms', 'teleop.feedback'):
        assert word in msgs, word
    w = json.loads(json.dumps(L.get('meta-quest-3').data))
    w['teleop']['workspace'] = {'value': {'kind': 'box', 'size': [1, 1]}, 'source': 0, 'quote': 'q'}
    assert any('workspace' in m for m in TD.entry_problems(w))


# ====================================================================== library teleop devices in the study
def _so100():
    from dexvantage.core.robot.samples import sample_project
    p = sample_project('so100')
    p.analysis.studies['teleop'] = dict(FAST)
    return p


def test_leads_matches_the_robot_it_is_stated_to_lead():
    from dexvantage.core.robot.samples import sample_project
    so100, ur5e = _so100().robots[0], sample_project('ur5e').robots[0]
    assert TD.leads(L.get('so100-leader-arm'), so100) and not TD.leads(L.get('so100-leader-arm'), ur5e)
    assert not TD.leads(L.get('so101-leader-arm'), so100)
    assert TD.leads(L.get('gello'), ur5e) and TD.leads(L.get('gello'), so100)        # built for its target arm


def test_device_from_library_makes_the_contract_record():
    p = _so100()
    fu = p.robots[0].uid
    lead = TD.device_from_library(p, 'so100-leader-arm')
    assert lead == TS.clean_device(lead, p)                                          # already clean
    assert lead['kind'] == 'leader_arm' and lead['library_id'] == 'so100-leader-arm'
    assert lead['leader'] == {'robot_uid': fu} and lead['mapping']['kind'] == 'joint'
    assert 'joints' in lead['source']['leader'] and 'read 2026-11-07' in lead['source']['dof']
    koch = TD.device_from_library(p, 'koch-v1-1-leader-arm')
    assert koch['leader'] == {'library_id': 'koch-v1-1-leader-arm'}                  # not stated to lead an SO-100
    glove = TD.device_from_library(p, 'manus-quantum-metagloves')
    assert glove['kind'] == 'glove' and glove['mapping']['kind'] == 'cartesian' and glove['volume'] is None
    assert glove['source']['volume'].startswith('not stated')
    assert '120 Hz' in glove['source']['tracking_hz'] and 'manus-meta.com' in glove['source']['tracking_hz']
    with pytest.raises(L.LibraryError, match='not a teleop device'):
        TD.device_from_library(p, 'intel-realsense-d435')
    # an existing device record without library_id keeps the contract's keys only
    assert 'library_id' not in TS.new_device(p, 'vr')


def test_library_devices_in_the_study():
    p = _so100()
    p.tasks.teleop_devices = [TD.device_from_library(p, 'so100-leader-arm'),
                              TD.device_from_library(p, 'aloha-widowx-250-leader'),
                              TD.device_from_library(p, 'ultraleap-leap-motion-controller-2')]
    doc = jobs.run_study('teleop', p)
    by = {d['library_id']: d for d in doc['devices']}
    assert by['so100-leader-arm']['stated'] and by['so100-leader-arm']['teleoperable_pct'] is not None
    assert not by['aloha-widowx-250-leader']['stated'] and not by['ultraleap-leap-motion-controller-2']['stated']
    text = ' '.join(doc['notes'])
    assert 'states the leader\'s figures but no kinematics' in text
    assert 'tracking volume is not stated' in text and 'does not guess' in text
    assert doc['summary']['teleoperable_pct'] is not None


# ====================================================================== validation checks
def test_library_systems_checks_pass():
    from dexvantage.validation import checks_library_systems as C
    from dexvantage.validation import suite
    suite.load_checks()
    specs = {s.id: s for s in suite.CHECKS}
    for cid, fn in (('library.teleop_figures', C._teleop_figures), ('library.policy_token_grid', C._token_grid),
                    ('library.stated_totals', C._totals)):
        o = fn(None)
        assert o.ok and o.error <= specs[cid].tolerance, (cid, o.error, o.detail)
        assert specs[cid].model == 'library' and specs[cid].unit
    from dexvantage.validation.references import REFERENCES
    for cid in ('library.teleop_figures', 'library.policy_token_grid', 'library.stated_totals'):
        assert all(k in REFERENCES for k in specs[cid].refs), cid


def test_checks_catch_a_mistyped_figure(monkeypatch):
    from dexvantage.validation import checks_library_systems as C
    e = L.get('manus-quantum-metagloves')
    data = json.loads(json.dumps(e.data))
    data['teleop']['tracking_hz']['value'] = 100                    # the quote says 120Hz
    monkeypatch.setattr(e, 'data', data)
    problems, _n = C.teleop_figure_problems()
    assert any('tracking_hz' in p for p in problems)
    h = L.get('basler-usb3-card-fl1100-4hc-x4')
    hd = json.loads(json.dumps(h.data))
    hd['host']['ingest'][0]['practical_mbps'] = 3800.0              # MB/s typed as Mbit/s
    monkeypatch.setattr(h, 'data', hd)
    assert max(r['rel'] for r in C.totals_and_units()) > 0.1


# ====================================================================== the page
def test_page_offers_library_devices_and_adds_one_in_one_step(app, win):
    from conftest import pump
    win.store.set_project(_so100())
    win.ctx.analysis.wait(120)
    pump(app, 0.05)
    page = win.pages['teleop']
    win.open_page('teleop')
    ids = [page.lib_box.itemData(k) for k in range(page.lib_box.count())]
    assert set(NEW['teleop']) <= set(ids) and page.b_add_lib.isEnabled()
    n0, d0 = win.store.undo.count(), len(win.store.project.tasks.teleop_devices)
    page.lib_box.setCurrentIndex(ids.index('manus-quantum-metagloves'))
    page.b_add_lib.click()
    devs = win.store.project.tasks.teleop_devices
    assert len(devs) == d0 + 1 and win.store.undo.count() == n0 + 1
    assert devs[-1]['library_id'] == 'manus-quantum-metagloves' and devs[-1]['kind'] == 'glove'
    assert page.current()['uid'] == devs[-1]['uid']
    assert 'Not stated by the maker' in page.lib_figures.text() and 'workspace' in page.lib_figures.text()
    assert page.kind_box.currentData() == 'glove'
    win.store.undo.undo()
    assert len(win.store.project.tasks.teleop_devices) == d0
    # the sample's own device and the plain Add still work
    assert page.add_device('spacemouse') is not None
    assert 'library_id' not in win.store.project.tasks.teleop_devices[-1]
