"""Regression tests for the truthfulness review of Phases 3, 6 and 7: figures that were uncited or claims that went
beyond their sources, and a validation check that passed without saying what it disagreed with."""
from dexvantage.core import brackets as B
from dexvantage.core import library as LB
from dexvantage.core.project import Project, Sensor, new_uid
from dexvantage.core.studies import bandwidth as BW


# ====================================================================== bandwidth: hosts with mixed USB generations
def _rig_on(library_id):
    p = Project('Rig')
    s = Sensor(name='D435', kind='depth', model_id='intel-realsense-d435', slot=0)
    p.variant.sensors.append(s)
    cu = new_uid()
    p.rig['hosts'].append({'uid': new_uid(), 'name': 'Host', 'library_id': library_id,
                           'controllers': [{'uid': cu, 'name': 'C0', 'kind': 'usb3', 'practical_mbps': None,
                                            'source': ''}]})
    p.rig['links'].append({'sensor_uid': s.uid, 'stream_ids': [], 'via': cu, 'cable_uid': ''})
    return p


def test_host_generations_named_in_the_interface():
    assert BW.host_usb3_generations(LB.get('nvidia-jetson-agx-orin-devkit')) == ['gen1', 'gen2']
    assert BW.host_usb3_generations(LB.get('nvidia-jetson-agx-orin-64gb')) == ['gen2']      # "PCIe Gen4" is not USB
    assert BW.host_usb3_generations(LB.get('usb-3.2-gen2x2-host')) == ['gen2x2']
    assert BW.host_usb3_generations(LB.get('usb-3.2-gen1-host')) == ['gen1']
    assert BW.host_usb3_generations(None) == []


def test_mixed_generation_host_assumes_the_lowest_and_says_so():
    """The AGX Orin developer kit has Gen 2 and Gen 1 ports: a controller on it must not get Gen 2's capacity."""
    c = BW.budget(_rig_on('nvidia-jetson-agx-orin-devkit'))['controllers'][0]
    assert c['generation'] == 'gen1'
    assert c['capacity_mbps'] == BW.CAPACITY_MBPS['gen1']
    assert 'lowest' in c['generation_note'] and 'Gen 2' in c['generation_note']
    # a host that names one generation keeps it
    c2 = BW.budget(_rig_on('nvidia-jetson-agx-orin-64gb'))['controllers'][0]
    assert c2['generation'] == 'gen2' and c2['capacity_mbps'] == BW.CAPACITY_MBPS['gen2']


def test_no_unsupported_generalisation_in_the_sharing_text():
    assert all('usually' not in a for a in BW.ASSUMPTIONS)


# ====================================================================== brackets
def test_print_density_is_cited():
    m = B.MATERIAL
    assert m['rho_g_cm3'] == 1.27
    assert m['source'].startswith('https://') and 'prusament-petg' in m['source']
    assert 'ISO 1183' in m['note'] and 'typical' not in m['note']


def test_tslot_suggestion_makes_no_survey_claim():
    p = Project('Rig')
    s = Sensor(name='Overhead', kind='camera', model_id='intel-realsense-d435', slot=0)
    s.mount.parent_frame = 'world'
    p.variant.sensors.append(s)
    kind, why = B.suggest_kind(p, s)
    assert kind == 'tslot2020'
    assert 'most common' not in why and 'default' in why


def test_iso9409_and_tslot_figures_match_their_sources():
    """Typed again from the UR5e flange drawing (UR5e datasheet) and the Motedis pages, read 2026-11-07."""
    f = B.ISO9409['50-4-M6']
    assert (f['pcd_mm'], f['count'], f['thread_d_mm'], f['thread_depth_mm']) == (50.0, 4, 6.0, 6.0)
    assert (f['pilot_d_mm'], f['pilot_fit'], f['spigot_d_mm'], f['spigot_fit']) == (31.5, 'H7', 63.0, 'h8')
    assert (f['spigot_height_mm'], f['pin_d_mm'], f['pin_fit'], f['pin_depth_mm']) == (6.5, 6.0, 'H7', 6.2)
    assert (f['first_deg'], f['pitch_deg']) == (45.0, 90.0)
    assert f['pilot_depth_mm'] is None
    assert B.TSLOT['tslot2020'] | {} == {'profile_mm': 20.0, 'groove_width_mm': 6.0, 'groove_depth_mm': 5.5,
                                          'source': 'motedis_2020'}
    assert B.TSLOT['tslot4040'] | {} == {'profile_mm': 40.0, 'groove_width_mm': 10.0, 'groove_depth_mm': 12.0,
                                          'source': 'motedis_4040'}


# ====================================================================== validation: disagreement is reported
def test_naneyec_check_reports_the_default_overhead_disagreement():
    from dexvantage.validation import checks_bandwidth as CB
    from dexvantage.validation import suite

    class Ctx:
        quick = True

    out = CB._stated_rates(Ctx())
    assert out.error == 0.0 and out.ok
    rows = out.detail['modes']
    for r in rows:
        assert r['study_default_overhead'] == BW.DEFAULT_OVERHEAD['other']
        assert 0.15 < r['study_wire_below_stated'] < 0.2          # 75 Mbit/s stated, 62.4 on the wire at 58 fps
        assert 0.25 < r['implied_framing_overhead'] < 0.27
    assert 'disagrees' in out.note and '16.9 %' in out.note
    assert suite is not None
