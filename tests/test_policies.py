"""Policies (spec §5, §1.1.6, §13): the policy library entries, the project's policies (file, schema, undo sections,
content hash), core/policies.py (resolve, overrides, crop geometry) and the Policies page."""
import glob
import json
import os
import types

import numpy as np
import pytest

from conftest import pump
from dexvantage.core import library as L
from dexvantage.core import policies as PO
from dexvantage.core import project as P

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
POLICY_DIR = os.path.join(L.BUILTIN_DIR, 'policy')
FILES = sorted(glob.glob(os.path.join(POLICY_DIR, '*.json')))
IDS = ('act', 'diffusion-policy-cnn', 'gr00t-n1-2b', 'octo-base', 'openvla-7b', 'pi0', 'pi0-fast', 'smolvla-base')


def _value(mid, k):
    return L.get(mid).data['policy'][k]['value']


# ====================================================================== the library entries
def test_every_policy_named_in_the_spec_has_an_entry():
    L.reload()
    # the spec's policies are there (more entries may follow: tests/test_library_systems.py pins the later ones)
    ids = {e.id for e in PO.library_entries()}
    assert set(IDS) <= ids
    assert sorted(os.path.splitext(os.path.basename(f))[0] for f in FILES) == sorted(ids)
    assert not [p for p in L.problems() if 'policy' in p]


@pytest.mark.parametrize('path', FILES, ids=os.path.basename)
def test_entry_is_valid_and_every_stated_figure_names_its_source(path):
    with open(path, encoding='utf-8') as fh:
        d = json.load(fh)
    assert d['kind'] == 'policy'
    assert L.validate(d) == [] and L.validate_minimal(d) == []
    assert PO.entry_problems(d) == []
    jsonschema = pytest.importorskip('jsonschema')
    jsonschema.Draft202012Validator(L.schema()).validate(d)
    for s in d['sources']:
        assert s['url'].startswith('https://') and s['read'] in ('2026-11-06', '2026-11-07') and s.get('title')
    # every figure is stated (with a source and what it says) or null with no source
    for k, f in d['policy'].items():
        if f['value'] is None:
            assert f['source'] is None, k
        else:
            assert isinstance(f['source'], int) and f.get('quote'), k
    text = open(path, encoding='utf-8').read()
    for bad in ('/' + 'home/', '/' + 'Users/', '/' + 'tmp/'):
        assert bad not in text


@pytest.mark.parametrize('mid, field, value', [
    # each pinned figure read from the source the entry cites (see the entry's quote)
    ('openvla-7b', 'input_resolution', [224, 224]), ('openvla-7b', 'crop_rule', 'resize'),
    ('openvla-7b', 'patch_px', 14), ('openvla-7b', 'n_cameras', 1), ('openvla-7b', 'inference_ms', 166.7),
    ('pi0', 'input_resolution', [224, 224]), ('pi0', 'crop_rule', 'resize_pad'), ('pi0', 'patch_px', 14),
    ('pi0', 'action_chunk', 50), ('pi0', 'execute_steps', 25), ('pi0', 'temporal_ensemble_m', False),
    ('pi0', 'inference_ms', 73), ('pi0', 'n_cameras', 3),
    ('pi0-fast', 'inference_ms', 750), ('pi0-fast', 'action_chunk', 32),
    ('smolvla-base', 'input_resolution', [512, 512]), ('smolvla-base', 'crop_rule', 'resize_pad_top_left'),
    ('smolvla-base', 'tokens_per_image', 64), ('smolvla-base', 'action_chunk', 50),
    ('act', 'input_resolution', [640, 480]), ('act', 'action_chunk', 100), ('act', 'temporal_ensemble_m', 0.01),
    ('act', 'inference_ms', 10), ('act', 'control_hz', 50), ('act', 'n_cameras', 4), ('act', 'tokens_per_image', 300),
    ('diffusion-policy-cnn', 'input_resolution', [288, 216]), ('diffusion-policy-cnn', 'pre_resize', [320, 240]),
    ('diffusion-policy-cnn', 'action_chunk', 16), ('diffusion-policy-cnn', 'execute_steps', 8),
    ('diffusion-policy-cnn', 'inference_ms', 100), ('diffusion-policy-cnn', 'control_hz', 10),
    ('octo-base', 'input_resolution', [256, 256]), ('octo-base', 'patch_px', 16), ('octo-base', 'action_chunk', 4),
    ('octo-base', 'wrist_input_resolution', [128, 128]),
    ('gr00t-n1-2b', 'input_resolution', [224, 224]), ('gr00t-n1-2b', 'crop_side_fraction', 0.95),
    ('gr00t-n1-2b', 'tokens_per_image', 64), ('gr00t-n1-2b', 'action_chunk', 16),
    ('gr00t-n1-2b', 'inference_ms', 63.9),
])
def test_figures_as_the_sources_state_them(mid, field, value):
    assert _value(mid, field) == value


@pytest.mark.parametrize('mid, field', [
    # figures the sources do not state stay null: never filled from memory
    ('openvla-7b', 'tokens_per_image'), ('openvla-7b', 'control_hz'), ('smolvla-base', 'inference_ms'),
    ('pi0-fast', 'control_hz'), ('act', 'patch_px'), ('gr00t-n1-2b', 'patch_px'), ('gr00t-n1-2b', 'n_cameras'),
    ('octo-base', 'control_hz'), ('diffusion-policy-cnn', 'temporal_ensemble_m'),
])
def test_figures_no_source_states_are_null(mid, field):
    assert _value(mid, field) is None
    p = P.blank_project()
    p.policies.append(P.Policy(library_id=mid))
    v, src = PO.resolve(p, p.policies[0].uid)[field]
    assert v is None and src.startswith(PO.NOT_STATED)


def test_pi0_fast_timing_names_no_hardware():
    assert L.get('pi0-fast').data['policy']['inference_ms']['hardware'] is None


@pytest.mark.parametrize('mid, scale', [('pi0', 1), ('pi0-fast', 1), ('octo-base', 1), ('smolvla-base', 4)])
def test_token_counts_agree_with_the_patch_grid(mid, scale):
    """Two independently cited figures (the patch from the code, the token count from a paper) agree with the patch
    grid (w / p) (h / p), divided by the pixel-shuffle factor squared where the model has one (SmolVLM2's
    scale_factor 4, from its config.json)."""
    w, h = _value(mid, 'input_resolution')
    p = _value(mid, 'patch_px')
    assert (w // p) * (h // p) // scale ** 2 == _value(mid, 'tokens_per_image')
    if mid == 'octo-base':
        ww, wh = _value(mid, 'wrist_input_resolution')
        assert (ww // p) * (wh // p) == _value(mid, 'wrist_tokens_per_image')


def test_entry_problems_catch_bad_entries():
    good = L.get('pi0').data
    assert PO.entry_problems(good) == []
    bad = json.loads(json.dumps(good))
    bad['policy']['crop_rule']['value'] = 'squash'
    bad['policy']['patch_px']['source'] = 9
    bad['policy']['tokens_per_image'] = {'value': None, 'source': 0}
    bad['policy']['lens'] = {'value': 1, 'source': 0}
    del bad['policy']['control_hz']
    msgs = ' '.join(PO.entry_problems(bad))
    for word in ('crop_rule', 'patch_px', 'tokens_per_image', 'lens', 'control_hz'):
        assert word in msgs, word


def test_policy_entries_do_not_appear_as_sensor_devices():
    """The Sensors page lists device kinds only; a policy entry has no imagers."""
    from dexvantage.ui import rigedit as RE
    for e in PO.library_entries():
        assert e.kind not in RE.SENSOR_KINDS and e.imagers == []


# ====================================================================== the project
def _project():
    p = P.blank_project('Policies test')
    a = P.Sensor(name='Wrist', kind='depth', mount=P.Mount('world'))
    b = P.Sensor(name='Overhead', kind='camera', slot=1)
    t = P.Sensor(name='Pad', kind='tactile', slot=2)
    p.variant.sensors += [a, b, t]
    p.policies.append(P.Policy(name='Pi', library_id='pi0', cameras=[b.uid],
                               overrides={'inference_ms': {'value': 95.0, 'source': 'measured by the user: Orin'}}))
    p.policies.append(P.Policy(name='Act', library_id='act'))
    return p


def test_file_round_trip_keeps_policies(tmp_path):
    p = _project()
    path = str(tmp_path / 'p.dexv')
    p.save(path)
    q = P.Project.load(path)
    assert [x.to_dict() for x in q.policies] == [x.to_dict() for x in p.policies]
    from dexvantage.core import provenance as PV
    assert PV.project_digest(p) == PV.project_digest(q)


def test_schema_copies_are_identical_and_validate_policies():
    a = open(os.path.join(ROOT, 'docs', 'project.schema.json'), encoding='utf-8').read()
    b = open(os.path.join(ROOT, 'dexvantage', 'resources', 'project.schema.json'), encoding='utf-8').read()
    assert a == b
    jsonschema = pytest.importorskip('jsonschema')
    schema = json.loads(a)
    doc = json.loads(json.dumps(_project().to_dict()))
    jsonschema.validate(doc, schema)
    doc['policies'][0]['overrides']['inference_ms'] = {'value': 1}            # no source: refused
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, schema)


def test_old_files_without_policies_read_and_hash_as_before():
    p = P.blank_project()
    d = p.to_dict()
    del d['policies']
    q = P.Project.from_dict(d)
    assert q.policies == []
    from dexvantage.core import provenance as PV
    assert 'policies' not in PV.project_digest(q)['sections']


def test_content_hash_covers_policies():
    from dexvantage.core import provenance as PV
    p = _project()
    a = PV.project_digest(p)
    p.policies[0].overrides['inference_ms']['value'] = 96.0
    b = PV.project_digest(p)
    assert a['sha256'] != b['sha256']
    assert [k for k in a['sections'] if a['sections'][k] != b['sections'].get(k)] == ['policies']


def test_section_and_restore():
    p = _project()
    assert 'policies' in P.SECTIONS
    s = p.section('policies')
    p.policies.clear()
    p.restore('policies', s)
    assert [x.name for x in p.policies] == ['Pi', 'Act']
    assert p.snapshot().policies[0].overrides == p.policies[0].overrides


def test_unknown_values_fall_back_safely():
    pol = P.Policy.from_dict({'name': 5, 'cameras': ['', 'abc', None], 'overrides': {'x': 3, 'inference_ms': {}}})
    assert pol.name == '5' and pol.cameras == ['abc'] and pol.overrides == {'inference_ms': {'value': None,
                                                                                            'source': ''}}


# ====================================================================== resolve
def test_resolve_applies_overrides_and_cites_the_library():
    p = _project()
    r = PO.resolve(p, p.policies[0].uid)
    assert set(r) == set(PO.FIELDS)
    assert r['inference_ms'] == (95.0, 'measured by the user: Orin')
    v, src = r['action_chunk']
    assert v == 50 and 'arXiv:2410.24164' in src and 'read 2026-11-06' in src and 'H=50' in src
    det = PO.resolve_detail(p, p.policies[0].uid)
    assert det['inference_ms']['origin'] == 'user' and det['inference_ms']['library_value'] == 73
    assert det['action_chunk']['origin'] == 'library' and det['tokens_per_image']['origin'] == 'library'


def test_resolve_refuses_an_unknown_policy_in_plain_words():
    from dexvantage.core.jobs import StudyError
    with pytest.raises(StudyError, match='not in the project'):
        PO.resolve(_project(), 'nope')


def test_missing_library_entry_is_said():
    p = P.blank_project()
    p.policies.append(P.Policy(library_id='no-such-policy'))
    v, src = PO.resolve(p, p.policies[0].uid)['input_resolution']
    assert v is None and 'no-such-policy' in src


def test_check_value_and_user_source():
    assert PO.check_value('input_resolution', '224 × 224') == [224, 224]
    assert PO.check_value('input_resolution', [640, 480.0]) == [640, 480]
    assert PO.check_value('action_chunk', '50') == 50
    assert PO.check_value('temporal_ensemble_m', 'not used') is False
    assert PO.check_value('temporal_ensemble_m', '0.01') == 0.01
    assert PO.check_value('crop_rule', 'resize_pad') == 'resize_pad'
    for k, bad in (('input_resolution', '224'), ('input_resolution', [224.5, 224]), ('action_chunk', 0),
                   ('action_chunk', 2.5), ('inference_ms', -3), ('inference_ms', 'fast'), ('crop_rule', 'squash'),
                   ('crop_side_fraction', 1.5), ('nope', 1), ('inference_ms', float('nan')), ('control_hz', True)):
        with pytest.raises(PO.PolicyError):
            PO.check_value(k, bad)
    assert PO.user_source('stopwatch over 100 runs') == 'measured by the user: stopwatch over 100 runs'
    assert PO.user_source('Measured by the user on a 4090') == 'Measured by the user on a 4090'
    with pytest.raises(PO.PolicyError):
        PO.user_source('  ')


def test_policy_cameras():
    p = _project()
    pi, act = p.policies
    assert [s.name for s in PO.policy_cameras(p, pi)] == ['Overhead']
    assert [s.name for s in PO.policy_cameras(p, act)] == ['Wrist', 'Overhead']      # all cameras, not tactile
    pi.cameras.append('gone-sensor')
    assert [s.name for s in PO.policy_cameras(p, pi)] == ['Overhead']


def test_summary_is_json_and_counts_gaps():
    p = _project()
    p.policies.append(P.Policy(name='Fast', library_id='pi0-fast'))
    doc = PO.summary(p)
    json.dumps(doc)
    by = {x['name']: x for x in doc['policies']}
    assert by['Pi']['not_stated'] == [] and by['Act']['not_stated'] == []
    assert by['Fast']['not_stated'] == ['control_hz']
    assert doc['not_stated'] == 1


# ====================================================================== crop geometry
def test_crop_geometry_known_values():
    """Values worked by hand from each source's code: openpi resize_with_pad, LeRobot's top-left padding,
    torchvision CenterCrop's int(round(...)), GR00T VideoCrop's int(H · 0.95)."""
    g = PO.crop_geometry('resize_pad', (640, 480), (224, 224))      # ratio 640/224: 224 × 168, divmod(56, 2) = 28
    assert g.dst_rect == (0.0, 28.0, 224.0, 196.0) and g.aspect_kept and g.padded_fraction == pytest.approx(0.25)
    g = PO.crop_geometry('resize_pad_top_left', (640, 480), (512, 512))   # 512 × 384, 128 rows padded on top
    assert g.dst_rect == (0.0, 128.0, 512.0, 512.0)
    g = PO.crop_geometry('center_crop', (1280, 720), (288, 216), pre_resize=(320, 240))
    assert g.src_rect == (64.0, 36.0, 1216.0, 684.0) and not g.aspect_kept
    g = PO.crop_geometry('center_crop_resize', (640, 480), (224, 224), 0.95)   # 608 × 456 at (16, 12)
    assert g.src_rect == (16.0, 12.0, 624.0, 468.0) and g.used_fraction == pytest.approx(608 * 456 / (640 * 480))
    g = PO.crop_geometry('none', (1280, 720), (640, 480))
    assert 'stretched' in g.note
    with pytest.raises(PO.PolicyError):
        PO.crop_geometry('center_crop', (200, 150), (288, 216))
    with pytest.raises(PO.PolicyError):
        PO.crop_geometry('squash', (640, 480), (224, 224))


def test_crop_geometry_maps_both_ways():
    rng = np.random.default_rng(7)
    for rule, frac, pre in (('resize', None, None), ('resize_pad', None, None), ('resize_pad_top_left', None, None),
                            ('center_crop', None, (320, 240)), ('center_crop_resize', 0.9, None)):
        g = PO.crop_geometry(rule, (1280, 720), (224, 224) if rule != 'center_crop' else (288, 216), frac, pre)
        x, y = rng.uniform(0, 1280, 50), rng.uniform(0, 720, 50)
        u, v = g.to_input(x, y)
        x2, y2 = g.to_source(u, v)
        assert np.allclose(x, x2) and np.allclose(y, y2)
        json.dumps(g.to_dict())


def test_input_geometry_uses_resolved_fields_and_says_what_is_missing():
    p = _project()
    g = PO.input_geometry(p, p.policies[0].uid, (640, 480))
    assert g.rule == 'resize_pad' and g.input_size == (224, 224)
    p.policies[0].overrides['input_resolution'] = {'value': [448, 448], 'source': 'measured by the user: fine-tune'}
    assert PO.input_geometry(p, p.policies[0].uid, (640, 480)).input_size == (448, 448)
    p.policies.append(P.Policy(library_id='no-such'))
    with pytest.raises(PO.PolicyError, match='input resolution'):
        PO.input_geometry(p, p.policies[-1].uid, (640, 480))


def test_validation_check_passes_and_would_catch_an_error(monkeypatch):
    from dexvantage.validation import checks_policies as C
    from dexvantage.validation import references as R
    from dexvantage.validation import suite as S
    S.load_checks()
    out = C._crop_geometry(None)
    assert out.ok and out.error <= 1e-6 and out.detail['pixels_compared'] > 1_000_000
    spec = next(c for c in S.CHECKS if c.id == 'policies.crop_geometry')
    assert spec.model == 'vla' and all(r in R.REFERENCES for r in spec.refs)
    # sensitivity: a geometry padded on the wrong side, or rounded the other way, fails
    real = PO.crop_geometry

    def wrong(rule, *a, **k):
        return real('resize_pad_top_left' if rule == 'resize_pad' else rule, *a, **k)
    monkeypatch.setattr(PO, 'crop_geometry', wrong)
    e, pads, _n = C.compare('resize_pad', (640, 480), (224, 224))
    assert pads > 0 and e > 1


# ====================================================================== the page
@pytest.fixture
def page(win, app):
    st = win.store

    def fn(p):
        p.variant.sensors += [P.Sensor(name='Wrist', kind='depth'), P.Sensor(name='Overhead', kind='camera', slot=1)]
    st.edit('variant', fn, 'Added two cameras')
    st.undo.setClean()
    pump(app, 0.02)
    pg = win.pages['policies']
    pg.refresh()
    return pg


def test_page_is_registered_and_starts_empty(page):
    from dexvantage.ui.pages import PAGES
    assert PAGES['policies'].cls is type(page)
    assert page.head.status.text.text() == 'no policy'
    assert page.add_box.count() == len(L.entries('policy')) >= len(IDS)


def test_add_override_clear_remove_are_one_undo_step_each(page, win):
    st = win.store
    n = st.undo.count()
    pol = page.add_policy('pi0-fast')
    assert pol is not None and st.undo.count() == n + 1 and st.project.policies[0].library_id == 'pi0-fast'
    assert page.head.figure.text() == '1' and 'control rate' in page.head.explain.text()
    # an override: the user's figure with its source
    assert page.set_override(pol.uid, 'control_hz', '15', 'robot controller rate, read from the driver')
    assert st.undo.count() == n + 2
    v, src = PO.resolve(st.project, pol.uid)['control_hz']
    assert v == 15.0 and src == 'measured by the user: robot controller rate, read from the driver'
    assert page.head.status.text.text() == 'too few'                 # 3 images, 2 cameras
    assert page.head.figure.text() == '2 of 3' and page.head.title.text() == 'Policy cameras assigned'
    # refused figures add no undo step and say why
    assert not page.set_override(pol.uid, 'inference_ms', 'fast', 'stopwatch')
    assert not page.set_override(pol.uid, 'inference_ms', '120', '')
    assert st.undo.count() == n + 2 and 'source' in page.ov_message.text()
    assert page.clear_override(pol.uid, 'control_hz')
    assert st.undo.count() == n + 3 and PO.resolve(st.project, pol.uid)['control_hz'][0] is None
    st.undo.undo()
    assert PO.resolve(st.project, pol.uid)['control_hz'][0] == 15.0
    page.rename(pol.uid, 'FAST on Orin')
    assert st.project.policies[0].name == 'FAST on Orin' and st.undo.count() == n + 3   # the redo step dropped
    assert page.remove(pol.uid) and st.project.policies == []
    st.undo.undo()
    assert st.project.policies[0].name == 'FAST on Orin'


def test_assign_cameras(page, win):
    st = win.store
    pol = page.add_policy('act')
    cams = [s.uid for s in st.project.variant.sensors]
    n = st.undo.count()
    assert page.set_cameras(pol.uid, [cams[1], 'not-a-sensor'])
    assert st.project.policies[0].cameras == [cams[1]] and st.undo.count() == n + 1
    assert not page.set_cameras(pol.uid, [cams[1]])                  # no change: no step
    assert page.set_cameras(pol.uid, [])
    assert st.project.policies[0].cameras == [] and st.undo.count() == n + 2
    st.undo.undo()
    assert st.project.policies[0].cameras == [cams[1]]


def test_headline_good_when_everything_needed_is_stated(page, win):
    pol = page.add_policy('pi0')
    st = win.store
    cams = [s.uid for s in st.project.variant.sensors]
    page.set_cameras(pol.uid, cams)
    page.set_override(pol.uid, 'n_cameras', 2, 'this rig has two cameras; the third slot masked')
    assert page.head.status.text.text() == 'all cited' and page.head.figure.text() == '0'


def test_export_writes_resolved_policies_with_provenance(page, tmp_path):
    page.add_policy('smolvla-base')
    path = page.export(str(tmp_path / 'policies.json'))
    doc = json.load(open(path, encoding='utf-8'))
    assert doc['policies'][0]['fields']['input_resolution']['value'] == [512, 512]
    from dexvantage.core import provenance as PV
    assert PV.KEY in doc and doc[PV.KEY]


def test_camera_view_overlay_draws_the_input_region(page, win):
    pol = page.add_policy('gr00t-n1-2b')
    cam = types.SimpleNamespace(width=640, height=480)
    out = page.camview_overlays(cam, None)
    assert len(out) == 1
    rect, _colour, text = out[0]
    assert np.allclose(rect[0], [15.5, 11.5]) and np.allclose(rect[2], [623.5, 467.5])
    assert '224 × 224' in text
    # a camera the policy does not take shows nothing
    other = win.store.project.variant.sensors[0].uid
    page.set_cameras(pol.uid, [win.store.project.variant.sensors[1].uid])
    win.camview.view.sensor_uid = other
    assert page.camview_overlays(cam, None) == []


def test_the_guide_on_the_ur5e_sample(win, app):
    """docs/guide/policies.md, step by step (after removing the sample's own policy, as the guide says)."""
    assert win.open_sample('ur5e')
    win.store.edit('policies', lambda p: setattr(p, 'policies', []), 'Removed the sample\'s policy')
    win.ctx.analysis.wait(5)
    pump(app, 0.02)
    page = win.pages['policies']
    page.refresh()
    assert page.head.status.text.text() == 'no policy'
    pol = page.add_policy('pi0')
    it = page.table.topLevelItem(0)
    assert [it.text(k) for k in range(5)] == ['π0 (pi0)', 'π0 (pi0)', 'all (2)', '224 × 224 px', '0']
    assert page.head.status.text.text() == 'too few' and 'Add a camera' in page.head.explain.text()
    overhead = next(s for s in win.store.project.variant.sensors if s.name == 'Overhead D435')
    win.camview.view.sensor_uid = overhead.uid
    (rect, _c, text), = page.camview_overlays(types.SimpleNamespace(width=1920, height=1080), None)
    assert text == 'π0 (pi0) input 224 × 224 · resized with padding · 44 % padding'
    g = PO.input_geometry(win.store.project, pol.uid, (1920, 1080))
    assert g.dst_rect == (0.0, 49.0, 224.0, 175.0) and round(g.sx, 3) == 0.117
    assert page.set_override(pol.uid, 'n_cameras', '2', 'this rig has two cameras; the third slot is masked')
    assert page.head.status.text.text() == 'all cited'
    fast = page.add_policy('pi0-fast')
    assert page.head.figure.text() == '1' and 'control rate is not stated' in page.head.explain.text()
    assert fast is not None
    win.store.undo.setClean()
