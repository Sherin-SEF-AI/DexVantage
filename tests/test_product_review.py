"""The product review of research tools part 2 (a research team's notebook and window flows on ALOHA 2): what it
found broken and fixed. A variant made from Python (rig.add_variant); every research result made through the API
logged when the project's tracking is on (statistics, campaign, cells, a gathered Slurm job); bad inputs refused in
a sentence before any work (gap recordings, campaign specs, jobs counts, a run compared with itself); pxr's errors
without C++ names or full paths; a sim-to-real mount σ that is not observable is None, not NaN."""
import json
import os

import pytest

import dexvantage as dv
from dexvantage.core import tracking as TR
from dexvantage.core.project import blank_project

CAM = 'sensor:Third-person camera (sample placement)'


@pytest.fixture
def so100(tmp_path):
    rig = dv.sample('so100')
    path = str(tmp_path / 'so100.dexv')
    rig.save(path)
    rig = dv.open(path)
    rig.factors['setting:tracking.enabled'] = True
    return rig


def _studies(rig):
    return [r['study'] for r in TR.store_for(rig.project).runs()] if TR.store_for(rig.project).exists() else []


# ====================================================================== a variant from Python
def test_add_variant_copies_the_active_one_and_sets_its_factors_only():
    rig = dv.sample('so100')
    before = rig.factors[CAM + '.z']
    v = rig.add_variant('Raised camera', {CAM + '.dz': 0.05})
    assert [x.name for x in rig.variants][-1] == 'Raised camera' and len(rig.variants) == 2
    assert rig.variant.name != 'Raised camera'                        # the active variant is unchanged
    assert rig.factors[CAM + '.z'] == pytest.approx(before)
    assert rig.use_variant('Raised camera').factors[CAM + '.z'] == pytest.approx(before + 0.05)
    assert {s.uid for s in v.sensors} == {s.uid for s in rig.sensors}   # the sensors keep their uids
    for bad, words in ((('Raised camera',), 'already has a variant'), (('',), 'Give the new variant a name'),
                       (('C', {'setting:coverage.cube_n': 64}), 'not a sensor factor'),
                       (('D', {}, 'Nope'), 'has no variant "Nope"'),
                       (('E', {'sensor:No such camera.dz': 0.1}), 'No such camera')):
        with pytest.raises(dv.DexVantageError) as e:
            rig.add_variant(*bad)
        assert words in str(e.value), (bad, str(e.value))
    assert len(rig.variants) == 2                                     # a refusal adds nothing


# ====================================================================== logging through the API
def test_api_campaign_and_cells_are_logged_when_tracking_is_on(so100):
    dv.campaign.plan(so100, {'operators': ['A', 'B']})
    dv.cells.compare(so100, 'sample:so100', metrics=['rig.camera_count'])
    assert _studies(so100) == ['campaign', 'cells']


def test_api_statistics_on_a_rig_is_logged_and_rates_alone_are_not(so100):
    dv.stats.compare_rates([('ACT', 14, 20), ('Diffusion', 19, 20)])          # no rig: nothing to log into
    assert _studies(so100) == []
    dv.stats.run({'source': 'rates', 'rates': [{'condition': 'A', 'successes': 3, 'trials': 10},
                                               {'condition': 'B', 'successes': 9, 'trials': 10}]}, so100)
    assert _studies(so100) == ['stats']


def test_a_run_compared_with_itself_is_refused(so100):
    so100.run('rig')
    rid = TR.store_for(so100.project).runs()[-1]['run_id']
    with pytest.raises(dv.DexVantageError, match='on both sides'):
        dv.stats.compare_runs(so100, rid, rid)


def test_gathered_slurm_experiment_is_logged_in_the_rigs_store(so100, tmp_path):
    design = {'schema': 'dexvantage.experiment/1', 'factors': [{'path': CAM + '.dz', 'lo': -0.01, 'hi': 0.01}],
              'responses': ['rig.camera_count'], 'design': {'method': 'lhs', 'n': 2, 'seed': 1}}
    job = str(tmp_path / 'job')
    dv.cluster.slurm(so100, job, experiment=design, shards=1, activate='true')
    dv.cluster.run_shard(os.path.join(job, 'work.json'), 0, 1)
    rep = dv.cluster.gather(job, rig=so100)
    assert rep['complete'] and _studies(so100) == ['experiments']


@pytest.mark.parametrize('jobs', [0, -2, 1.5, '4', True])
def test_cluster_jobs_must_be_a_whole_number_of_at_least_one(so100, jobs):
    with pytest.raises(dv.DexVantageError, match='number of processes'):
        dv.cluster.experiment(so100, {'schema': 'dexvantage.experiment/1'}, jobs=jobs)


# ====================================================================== bad inputs, refused before the work
@pytest.mark.parametrize('gaps,words', [
    ({'brightness_dn': 'lots'}, 'brightness_dn must be a number'),
    ({'blur_px': -1}, 'blur_px must be a number of at least 0'),
    ({'gain': 0}, 'gain must be a number'),
    ({'noise_dn': float('nan')}, 'noise_dn must be a number'),
    ({'mount': {'position_m': [0, 0]}}, 'three numbers'),
    ({'mount': {'shift': 1}}, 'position_m'),
    ({'nope': 1}, 'no gap called nope'),
    ('bright', 'The gaps are a dict'),
])
def test_gap_recording_refuses_bad_gaps_in_a_sentence(tmp_path, gaps, words):
    from dexvantage.core.sim2real import inject as IJ
    with pytest.raises(dv.DexVantageError) as e:
        IJ.make_gap_dataset(blank_project('x'), str(tmp_path / 'g'), gaps)
    assert words in str(e.value), str(e.value)


def test_gap_recording_refuses_a_folder_that_is_not_empty_before_rendering(tmp_path):
    import time
    from dexvantage.core.sim2real import inject as IJ
    (tmp_path / 'g').mkdir()
    (tmp_path / 'g' / 'x.txt').write_text('x')
    t = time.time()
    with pytest.raises(dv.DexVantageError, match='not an empty folder'):
        IJ.make_gap_dataset(dv.sample('so100').project, str(tmp_path / 'g'), {'brightness_dn': 5})
    assert time.time() - t < 2.0                                       # no render first (it took 28 s on ALOHA 2)


def test_campaign_spec_refuses_an_unknown_setting_and_points_to_the_power_analysis(so100):
    with pytest.raises(dv.DexVantageError) as e:
        dv.campaign.plan(so100, {'power': {'p1': 0.6, 'p2': 0.8}})
    assert 'no setting "power"' in str(e.value) and '"mode": "power"' in str(e.value)
    with pytest.raises(dv.DexVantageError, match='no setting "operatorz"'):
        dv.campaign.plan(so100, operatorz=['A'])
    res = dv.campaign.plan(so100, {'demos': {'mode': 'power', 'p_base': 0.6, 'p_target': 0.8}})
    assert res['demos']['mode'] == 'power'
    from dexvantage.core import campaign as CP                      # a spec saved by the page reads back
    assert CP.read_spec(json.dumps({'schema': CP.SPEC_SCHEMA, **CP.DEFAULTS, 'plan': {}}))


# ====================================================================== plain sentences
def test_pxr_errors_are_read_without_cpp_names_or_full_paths():
    from dexvantage.core.robot.usd_import import _pxr_reason
    parse = ("\n\tError in 'pxrInternal_v0_26_8__pxrReserved__::Sdf_TextFileFormatParser::_RaiseErrorPEGTL' at line 48"
             " in file /opt/USD/pxr/usd/sdf/textParserHelpers.h : '/data/cells/bad.usda:3:1: Expected ) at '' in </>\n"
             "'\n\tError in 'pxrInternal_v0_26_8__pxrReserved__::UsdStage::Open' at line 1276 in file "
             "/opt/USD/pxr/usd/usd/stage.cpp : 'Failed to open layer @/data/cells/bad.usda@'")
    assert _pxr_reason(parse) == "line 3, column 1: Expected ) at '' in </>"
    only = ("\n\tError in 'pxrInternal_v0_26_8__pxrReserved__::UsdStage::Open' at line 1276 in file "
            "/opt/USD/pxr/usd/usd/stage.cpp : 'Failed to open layer @/data/cells/bad2.usd@'")
    assert _pxr_reason(only) == ''
    assert _pxr_reason('could not read /data/cells/arm.usda') == 'could not read arm.usda'


def test_sim2real_mount_sigma_is_none_where_it_is_not_finite(monkeypatch):
    """The mount σ of an axis the frames do not constrain was NaN (cam_low on ALOHA 2): the result could not be
    logged or written as JSON. It is None now (the gap module's mount step, the silhouette estimate stubbed)."""
    import math
    import types

    from dexvantage.core.calib import render_calib as RC
    from dexvantage.core.sim2real import gap as G
    rig = dv.sample('so100')
    s = next(x for x in rig.sensors if not x.mount.on_link)
    nan = float('nan')
    fake = {'observable': False, 'why': 'one view', 'delta': {'position_m': [0.0, 0.0, 0.0],
                                                             'rotation_deg': [0.0, 0.0, 0.0]},
            'sigma': {'position_mm': [nan, 1.5, nan], 'rotation_deg': [0.2, nan, 0.1]},
            'fit': {}, 'mount': {'pose': s.mount.pose.to_dict()}}
    monkeypatch.setattr(RC, 'estimate', lambda *a, **k: fake)
    cam = types.SimpleNamespace(sensor=s, stream=types.SimpleNamespace(key='cam'), R=None)
    st = {'mount': True, 'mount_settings': {}, 'device': 'auto'}
    out = G._mount(cam, rig.project, types.SimpleNamespace(cameras={}), None, st, None, None)
    assert out['sigma'] == {'position_mm': [None, 1.5, None], 'rotation_deg': [0.2, None, 0.1]}
    json.dumps(out['sigma'], allow_nan=False)
    assert not any(isinstance(x, float) and math.isnan(x) for v in out['sigma'].values() for x in v)


def test_lighting_cause_with_a_shared_offset_sets_render_responses_not_a_light_scale():
    """A +20 DN brightness on every camera fits as gain 0.99 and offset +20 DN: scaling the lights by 0.99 would not
    close it, and the text said "0.99x as bright" for a brighter recording. With an offset the cause names it and
    Apply sets each camera's render response, lights or not."""
    from dexvantage.core import lights as LT
    from dexvantage.core.sim2real import gap as G
    p = dv.sample('ur5e').project

    def cam(uid, gain, off, red):
        return {'sensor_uid': uid, 'photometric': {'gain': [gain] * 3, 'offset_dn': [off] * 3},
                'causes': [{'factor': 'exposure', 'camera': uid, 'sensor': uid, 'sensor_uid': uid,
                            'reduction_dn': red, 'share_pct': 80.0, 'title': 'Exposure',
                            'apply': {'kind': 'appearance', 'sensor_uid': uid, 'set': {'gain': [gain] * 3}}}]}
    p.workspace.lights = [LT.new_light('panel', [0.0, 0.0, 2.0])]
    p.workspace.lights[0]['dim'] = 0.5
    st = G.settings_of(p)
    rows = G._rank(p, [cam('a', 0.99, 20.0, 18.0), cam('b', 0.99, 19.6, 17.0)], st)
    assert rows[0]['factor'] == 'lighting' and rows[0]['apply']['kind'] == 'appearance_many'
    assert 'offset +20 DN' in rows[0]['text'] and 'not a constant offset' in rows[0]['text']
    assert 'as bright as the render' not in rows[0]['text']
    rows = G._rank(p, [cam('a', 1.30, 0.2, 20.0), cam('b', 1.25, -0.3, 15.0)], st)   # a gain alone: the lights
    assert rows[0]['apply'] == {'kind': 'lights', 'scale': pytest.approx(1.275)}


def test_a_logged_experiment_states_how_long_it_ran(so100, tmp_path):
    """Experiments state their time under execution.seconds (not a top-level 'seconds'): the run record had no
    duration and started equal to ended."""
    design = {'schema': 'dexvantage.experiment/1', 'factors': [{'path': CAM + '.dz', 'lo': -0.01, 'hi': 0.01}],
              'responses': ['rig.camera_count'], 'design': {'method': 'lhs', 'n': 2, 'seed': 1}}
    res = dv.cluster.experiment(so100, design, jobs=1)
    rec = TR.store_for(so100.project).runs()[-1]
    assert rec['study'] == 'experiments' and rec['seconds'] == pytest.approx(res['execution']['seconds'])
    assert rec['started_utc'] < rec['ended_utc'] or rec['seconds'] < 0.001
    assert TR.record.fmt(0.0915, 'p') == 'p = 0.0915'                 # a p-value in the Runs table
