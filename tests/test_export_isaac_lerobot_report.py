"""The Isaac, LeRobot and report exporters (core/exporters/isaac.py, lerobot.py, report.py, _pdf.py) and their
round-trip checks (validation/formats_isaac_lerobot.py, checks_formats.py formats.isaac_*, lerobot_export, report)."""
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from dexvantage.core import exporters as X
from dexvantage.core.jobs import StudyError
from dexvantage.core.optics import CameraModel
from dexvantage.core.project import Policy, Requirement
from dexvantage.core.robot.samples import sample_project
from dexvantage.validation import formats_isaac_lerobot as V

HAVE_PXR = __import__('importlib').util.find_spec('pxr') is not None


@pytest.fixture(scope='module')
def ur5e():
    return sample_project('ur5e')


def _files(d):
    return [os.path.join(dp, f) for dp, _, fs in os.walk(d) for f in fs]


# ====================================================================== registry and contract
def test_exporters_are_registered():
    X.load_exporters()
    for fid in ('isaac', 'lerobot', 'report'):
        assert fid in X.EXPORTERS


def test_unstamped_files_keep_their_bytes_and_their_hash(ur5e, tmp_path):
    man = X.export(ur5e, str(tmp_path), ['lerobot'])
    row = man['formats']['lerobot']
    assert row['status'] == 'ok' and row['unstamped']
    for rel in row['unstamped']:
        doc = json.load(open(tmp_path / rel))
        assert 'provenance' not in doc and rel in man['files']
    stamped = [r for r in row['files'] if r.endswith('.json') and r not in row['unstamped']]
    assert stamped and all('provenance' in json.load(open(tmp_path / r)) for r in stamped)


# ====================================================================== Isaac
def test_lens_mapping_per_model():
    from dexvantage.core.exporters import isaac as I
    pin = CameraModel(640, 480, 500.0, 500.0, 319.5, 239.5)
    assert I.lens_model(pin)['distortion'] is None
    ns = I.lens_model(CameraModel(640, 480, 500.0, 520.0, 319.5, 239.5))
    assert ns['distortion']['model'] == 'opencvPinhole' and ns['exact'] and not ns['distortion']['coefficients']
    br = I.lens_model(CameraModel(640, 480, 500.0, 500.0, 300.0, 250.0, 'brown', (0.1, -0.02, 0.001, 0.002, 0.01)))
    assert br['distortion']['coefficients'] == {'k1': 0.1, 'k2': -0.02, 'p1': 0.001, 'p2': 0.002, 'k3': 0.01}
    kb = I.lens_model(CameraModel(640, 480, 300.0, 300.0, 319.5, 239.5, 'kb', (0.01, -0.002, 0.0003, -0.0001)))
    assert kb['distortion']['model'] == 'opencvFisheye' and kb['exact']
    eq = I.lens_model(CameraModel(640, 480, 300.0, 300.0, 319.5, 239.5, 'equidistant'))
    assert eq['exact'] and set(eq['distortion']['coefficients'].values()) == {0.0}
    es = I.lens_model(CameraModel(640, 480, 250.0, 250.0, 319.5, 239.5, 'equisolid'))
    assert not es['exact'] and 'no equisolid model' in es['text']
    assert es['fit_error_px'] is not None and es['fit_error_px'] < 0.5     # a fit, stated, small on this lens


def test_usd_lens_offsets_reproduce_the_principal_point():
    """The GfCamera frustum written out: fx = W f / hAp, cx = (W − 1)/2 − W hOff / hAp, fy = H f / vAp,
    cy = (H − 1)/2 + H vOff / vAp (pxr checks the same in formats.isaac_usd)."""
    from dexvantage.core.exporters import isaac as I
    cam = CameraModel(1280, 720, 900.0, 870.0, 610.25, 380.5)
    u = I.usd_lens(cam)
    W, H = 1280, 720
    assert W * u['focalLength'] / u['horizontalAperture'] == pytest.approx(cam.fx, rel=1e-12)
    assert H * u['focalLength'] / u['verticalAperture'] == pytest.approx(cam.fy, rel=1e-12)
    assert (W - 1) / 2 - W * u['horizontalApertureOffset'] / u['horizontalAperture'] == pytest.approx(cam.cx)
    assert (H - 1) / 2 + H * u['verticalApertureOffset'] / u['verticalAperture'] == pytest.approx(cam.cy)


def test_isaac_export_files_and_notes(ur5e, tmp_path):
    man = X.export(ur5e, str(tmp_path), ['isaac'])
    row = man['formats']['isaac']
    assert row['status'] == 'ok', row.get('reason')
    for rel in ('isaac/cameras.usda', 'isaac/isaaclab_cameras.py', 'isaac/isaaclab_v2_cameras.py',
                'isaac/isaac_cameras.json'):
        assert rel in row['files'] and (tmp_path / rel).is_file()
    v3 = (tmp_path / 'isaac/isaaclab_cameras.py').read_text()
    assert 'from isaaclab.sensors import CameraCfg' in v3 and 'QUAT_ORDER = "xyzw"' in v3
    assert 'e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4' in v3
    v2 = (tmp_path / 'isaac/isaaclab_v2_cameras.py').read_text()
    assert 'TiledCameraCfg' in v2 and 'QUAT_ORDER = "wxyz"' in v2 and 'distortion=' not in v2
    doc = json.load(open(tmp_path / 'isaac/isaac_cameras.json'))
    assert doc['cameras'] and all(c['tolerance'] for c in doc['cameras'])
    assert any('labelled defaults' in n for n in row['notes'])
    depth = [c for c in doc['cameras'] if 'distance_to_image_plane' in c['data_types']]
    assert depth, 'a depth sensor\'s reference imager gets a depth output'


def test_link_prim_names_keep_their_case():
    from dexvantage.core.exporters import isaac as I
    p = sample_project('so100')
    recs, _ = I.camera_records(p)
    text = I.config_text(p, recs, {}, v3=True)
    assert "'/Robot/Fixed_Jaw/" in text, 'the URDF/MJCF link name as the importers keep it'
    assert I.prim_token('wrist 3-link') == 'wrist_3_link' and I.prim_token('3link') == '_3link'


def test_isaac_configs_execute_against_stubs_and_match(tmp_path):
    p = V.lens_variety(sample_project('ur5e'))
    d = V.isaac_config_errors(p, str(tmp_path), np.random.default_rng(0))
    assert not d['mismatches'], d['mismatches']
    assert d['pose_m'] < 1e-9 and d['pose_rad'] < 1e-9 and d['intrinsics_rel'] < 1e-9
    assert d['dr_m'] < 1e-9 and d['dr_rad'] < 1e-9 and d['draws_checked'] > 0


def test_stubs_refuse_a_field_the_source_does_not_have():
    ns_ok = V.sandbox_exec('import isaaclab.sim as s\nx = s.PinholeCameraCfg(focal_length=1.0)\n', V.isaac_stubs(True))
    assert ns_ok['x'].focal_length == 1.0
    with pytest.raises(TypeError):
        V.sandbox_exec('import isaaclab.sim as s\nx = s.PinholeCameraCfg(distortion=None)\n', V.isaac_stubs(False))
    with pytest.raises(ImportError):
        V.sandbox_exec('import os\n', V.isaac_stubs(True))


def test_usd_text_writer_reads_back_without_pxr():
    d = V.isaac_usd_text_errors(V.lens_variety(sample_project('dexhand')), '')
    assert not d['mismatches'] and d['proj_px'] < 1e-9 and d['pose_m'] < 1e-12 and d['pose_rad'] < 1e-6


@pytest.mark.skipif(not HAVE_PXR, reason='usd-core (the usd extra) is not installed')
def test_usd_reopened_with_pxr_matches(tmp_path):
    d = V.isaac_usd_errors(V.lens_variety(sample_project('ur5e')), str(tmp_path))
    assert not d['mismatches'], d['mismatches']
    assert d['proj_px'] < 1e-3 and d['pose_m'] < 1e-9 and d['lens_rel'] < 1e-6


def test_isaac_text_option_and_refusals(ur5e, tmp_path):
    man = X.export(ur5e, str(tmp_path), ['isaac'], {'isaac': {'usd': 'text'}})
    assert man['formats']['isaac']['status'] == 'ok'
    assert (tmp_path / 'isaac/cameras.usda').read_text().startswith('#usda 1.0')
    man = X.export(ur5e, str(tmp_path / 'b'), ['isaac'], {'isaac': {'usd': 'binary'}})
    assert man['formats']['isaac']['status'] == 'refused'
    from dexvantage.core.project import blank_project
    man = X.export(blank_project('Empty'), str(tmp_path / 'c'), ['isaac'])
    assert man['formats']['isaac']['status'] == 'refused' and 'camera' in man['formats']['isaac']['reason']


# ====================================================================== LeRobot
def _with_policy(lib, rec=None):
    p = sample_project('ur5e')
    p.policies = [Policy(name=f'{lib} test', library_id=lib, recommendations=rec or {})]
    return p


def test_act_train_config_carries_the_recommendations(tmp_path):
    rec = {'temporal_ensemble_m': {'value': 0.05, 'source': 'action-chunk study (test)'},
           'delay_offset_s': {'value': 0.04, 'source': 'action-chunk study (test)'}}
    p = _with_policy('act', rec)
    man = X.export(p, str(tmp_path), ['lerobot'])
    row = man['formats']['lerobot']
    assert row['status'] == 'ok', row.get('reason')
    tc_rel = next(r for r in row['files'] if r.endswith('train_config.json'))
    tc = json.load(open(tmp_path / tc_rel))
    pol = tc['policy']
    assert pol['type'] == 'act' and pol['temporal_ensemble_coeff'] == 0.05 and pol['n_action_steps'] == 1
    assert pol['push_to_hub'] is False and tc['dataset']['repo_id']
    cams = [k for k, f in pol['input_features'].items() if f['type'] == 'VISUAL']
    assert sorted(cams) == ['observation.images.overhead_d435', 'observation.images.wrist_d405']
    dt = json.load(open(tmp_path / tc_rel.replace('train_config.json', 'delta_timestamps.json')))
    fps = dt['fps']
    steps = round(0.04 * fps)
    assert dt['delay_steps'] == steps and dt['delta_timestamps']['action'][0] == pytest.approx(steps / fps)
    import yaml
    act = yaml.safe_load(open(tmp_path / 'act/task_config.yaml'))
    assert act['temporal_agg_k'] == 0.05 and act['delay_offset_steps'] == steps
    assert any('one image shape' in n for n in row['notes'])          # 1280 × 720 and 1920 × 1080 differ


def test_diffusion_policy_latency_steps_and_negative_offset(tmp_path):
    import yaml
    p = _with_policy('diffusion-policy-cnn', {'delay_offset_s': {'value': 0.2, 'source': 'test'}})
    man = X.export(p, str(tmp_path), ['lerobot'])
    row = man['formats']['lerobot']
    dp = yaml.safe_load(open(tmp_path / 'diffusion_policy/task.yaml'))
    assert dp['n_latency_steps'] == 2                                   # 0.2 s at the entry's 10 Hz control rate
    assert dp['shape_meta']['obs']['wrist_d405']['shape'] == [3, 240, 320]   # the entry's pre_resize
    tc = json.load(open(tmp_path / next(r for r in row['files'] if r.endswith('train_config.json'))))
    assert tc['policy']['type'] == 'diffusion' and tc['policy']['resize_shape'] == [240, 320]
    p = _with_policy('diffusion-policy-cnn', {'delay_offset_s': {'value': -0.1, 'source': 'test'}})
    man = X.export(p, str(tmp_path / 'neg'), ['lerobot'])
    assert yaml.safe_load(open(tmp_path / 'neg/diffusion_policy/task.yaml'))['n_latency_steps'] == 0
    assert any('negative' in n for n in man['formats']['lerobot']['notes'])


def test_policy_without_lerobot_type_and_no_policy(tmp_path):
    p = _with_policy('openvla-7b')
    man = X.export(p, str(tmp_path), ['lerobot'])
    row = man['formats']['lerobot']
    assert not any(r.endswith('train_config.json') for r in row['files'])
    assert any('no LeRobot policy type' in n for n in row['notes'])
    keys = json.load(open(tmp_path / 'openvla/oxe_dataset_config.json'))['entry']['image_obs_keys']
    assert keys == {'primary': 'overhead_d435', 'secondary': None, 'wrist': 'wrist_d405'}
    p = sample_project('dexhand')
    man = X.export(p, str(tmp_path / 'd'), ['lerobot'])
    assert any('no policy' in n for n in man['formats']['lerobot']['notes'])
    man = X.export(p, str(tmp_path / 'e'), ['lerobot'], {'lerobot': {'fps': 29.97}})
    assert man['formats']['lerobot']['status'] == 'refused' and 'whole number' in man['formats']['lerobot']['reason']


def test_lerobot_round_trip_through_the_reader(tmp_path):
    from dexvantage.core import jobs
    from dexvantage.core.studies import action_chunk_sync as AC
    p = sample_project('ur5e')
    AC.apply_recommendations(p, jobs.run_study('action_chunk_sync', p.snapshot()))
    d = V.lerobot_errors(p, str(tmp_path))
    assert not d['mismatches'], d['mismatches']
    assert d['detail']['train_configs'] == 1


# ====================================================================== report
def test_pdf_text_encoding():
    from dexvantage.core.exporters._pdf import to_winansi
    assert to_winansi('≥ 95 % · σ 0.08 px, 3 × 4 °') == '>= 95 % · sigma 0.08 px, 3 × 4 °'.encode('cp1252')
    assert to_winansi('π0 (sample)') == b'pi0 (sample)'


def test_report_html_and_pdf(tmp_path):
    p = sample_project('ur5e')
    from dexvantage.core import requirements as RQ
    p.requirements = [r for r in RQ.default_requirements() if r.study in V.REPORT_STUDIES]
    assert p.requirements
    d = V.report_errors(p, str(tmp_path))
    assert not d['mismatches'], d['mismatches']
    assert d['sections'] >= 4 and d['figures'] >= 1
    html = (tmp_path / 'report/report.html').read_text()
    assert 'Requirements scorecard' in html and 'Provenance' in html
    data = (tmp_path / 'report/report.pdf').read_bytes()
    assert data.startswith(b'%PDF-1.4') and data.rstrip().endswith(b'%%EOF')
    if shutil.which('pdftotext'):
        txt = subprocess.run(['pdftotext', str(tmp_path / 'report/report.pdf'), '-'], capture_output=True,
                             text=True).stdout
        assert 'Bandwidth, hosts and power' in txt


def test_report_uses_given_results_and_lists_refusals(ur5e, tmp_path):
    from dexvantage.core import jobs
    doc = jobs.run_study('bandwidth', ur5e.snapshot())
    man = X.export(ur5e, str(tmp_path), ['report'], {'report': {'results': {'bandwidth': doc},
                                                                 'studies': ['audit']}})
    row = man['formats']['report']
    assert row['status'] == 'ok'
    rep = json.load(open(tmp_path / 'report/report.json'))
    assert [s['id'] for s in rep['sections']] == ['bandwidth']
    assert rep['refused'] and rep['refused'][0]['id'] == 'audit' and 'dataset' in rep['refused'][0]['reason']
    assert 'provenance' in rep                                          # stamped by export()
    man = X.export(ur5e, str(tmp_path / 'none'), ['report'], {'report': {'studies': ['audit']}})
    assert man['formats']['report']['status'] == 'refused'
    assert 'no study result' in man['formats']['report']['reason']


def test_report_results_dir(ur5e, tmp_path):
    from dexvantage.core import jobs
    os.makedirs(tmp_path / 'res')
    with open(tmp_path / 'res/lighting.json', 'w') as fh:
        json.dump(jobs.run_study('lighting', ur5e.snapshot()), fh)
    man = X.export(ur5e, str(tmp_path / 'out'), ['report'], {'report': {'results_dir': str(tmp_path / 'res'),
                                                                         'pdf': False}})
    assert man['formats']['report']['status'] == 'ok'
    assert not (tmp_path / 'out/report/report.pdf').exists()
    rep = json.load(open(tmp_path / 'out/report/report.json'))
    assert [s['id'] for s in rep['sections']] == ['lighting'] and rep['sections'][0]['word'] == 'no lights'


def test_no_personal_paths_in_exports(ur5e, tmp_path):
    man = X.export(ur5e, str(tmp_path), ['isaac', 'lerobot'])
    assert all(r['status'] == 'ok' for r in man['formats'].values())
    home = os.path.expanduser('~')
    for f in _files(tmp_path):
        if f.endswith(('.json', '.py', '.yaml', '.usda', '.sh')):
            t = open(f, encoding='utf-8').read()
            assert home not in t and '/home/' not in t, f


def test_requirement_scorecard_in_report_document(ur5e):
    from dexvantage.core.exporters import report as XR
    p = ur5e.snapshot()
    p.requirements = [Requirement(name='Skew', metric='sync.skew_ms', op='<', threshold=2.0, unit='ms',
                                  study='time_sync')]
    doc = XR.build(p, {'studies': ['time_sync']})
    card = doc['scorecard']
    assert card is not None and len(card['items']) == 1 and card['items'][0]['value'] is not None
    assert card['studies']['time_sync']['status'] == 'ran'
    assert doc['provenance_block']['project']['sha256']
    with pytest.raises(StudyError):
        XR.build(p, {'studies': ['no_such_study']})


# ====================================================================== the validation checks
def test_the_validation_checks_pass():
    from dexvantage.validation import suite as S
    S.load_checks()
    by = {c.id: c for c in S.CHECKS}
    ctx = S.Ctx(None, None, None, S.ValidationSettings(quick=True))
    try:
        for cid in ('formats.isaac_usd', 'formats.isaac_config', 'formats.report'):
            r = S.run_one(by[cid], ctx)
            assert r.passed, (cid, r.error, r.exception, r.detail.get('mismatches'))
    finally:
        ctx.close()
