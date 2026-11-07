"""Lighting, glare and IR wash-out: the materials (core/materials.py), the lights (core/lights.py) and the 'light'
library entries, the study (core/studies/lighting.py), its batch flag, its validation checks and the page
(ui/pages/lighting.py)."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from dexvantage.core import jobs
from dexvantage.core import lights as LT
from dexvantage.core import materials as MT
from dexvantage.core.frames import Pose
from dexvantage.core.project import Mount, Project, Sensor, TaskObject
from dexvantage.core.studies import lighting as LS

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def _box(name, size, at, tags):
    return TaskObject(name=name, shape='box', size=tuple(size), pose=Pose(np.asarray(at, float),
                                                                           np.array([0, 0, 0, 1.0])),
                      materials=list(tags))


def _scene(tag='glossy', cam=(0.6, 0.0, 0.40), light=(-0.6, 0.0, 0.40), blocker=False, absolute=True,
           exposure=0.01, model_id=''):
    """A flat box under a point light, a camera at the mirror angle (about 56°, near Brewster's angle for n 1.49)."""
    from dexvantage.core.robot.samples import look_at
    p = Project('Glare')
    p.workspace.objects = [_box('Box', (0.2, 0.2, 0.02), (0, 0, 0.01), [tag])]
    if blocker:
        p.workspace.objects.append(_box('Wall', (0.02, 0.6, 0.6), (-0.3, 0, 0.3), ['matte']))
    p.variant.sensors = [Sensor(name='Cam', kind='depth' if model_id else 'camera', model_id=model_id,
                                mount=Mount('world', '', look_at(cam, [0, 0, 0.02])))]
    p.workspace.lights = [LT.clean_light({'kind': 'point', 'name': 'Bulb',
                                          'pose': {'position': list(light), 'quaternion': [1, 0, 0, 0]},
                                          'luminous_flux_lm': {'value': 1000.0, 'source': 'test'}})]
    u = p.variant.sensors[0].uid
    st = {'object_points': 2000}
    if absolute:
        st['sensors'] = {u: {'sensitivity_dn_per_lux_s': 2000.0, 'f_number': 2.0, 'exposure_s': exposure}}
    p.analysis.studies['lighting'] = st
    return p, u


# ====================================================================== materials
def test_material_indices_and_f0():
    pm = MT.MATERIALS['matte']
    assert pm.n('vis').real == pytest.approx(1.4926, abs=1e-4)       # Sultanova 2009 Sellmeier at 550 nm
    assert MT.f0_from_index(1.5) == pytest.approx(0.04)
    assert MT.MATERIALS['metallic'].f0('vis') == pytest.approx(0.9157, abs=1e-4)
    assert MT.MATERIALS['transparent'].transmissive and MT.MATERIALS['transparent'].depth_reads_through
    for m in MT.MATERIALS.values():
        assert set(m.sources) >= {'f0', 'alpha', 'albedo', 'ir_albedo'}
        assert 'refractiveindex.info' in m.sources['f0']


def test_material_for_rule_and_overrides():
    assert MT.material_for([]).tag == 'matte'
    assert MT.material_for(['matte', 'glossy']).tag == 'glossy'
    assert MT.material_for(['glossy', 'transparent']).tag == 'transparent'
    m = MT.material_for(['black', 'ir_absorbing'])
    assert m.tag == 'black' and m.ir_albedo == pytest.approx(min(0.04, MT.DEFAULT_IR_ABSORBING_ALBEDO))
    assert MT.material_for(['ir_absorbing']).ir_albedo == MT.DEFAULT_IR_ABSORBING_ALBEDO
    o = MT.material_for(['matte'], {'alpha': 0.07, 'albedo': 2.0, 'depth_reads_through': 1, 'junk': 3})
    assert o.alpha == 0.07 and o.albedo == 1.0 and o.depth_reads_through
    assert o.sources['alpha'] == 'your setting'


def test_brdf_closed_form_reciprocity_and_below_horizon():
    n = np.array([[0.0, 0.0, 1.0]])
    for m in MT.MATERIALS.values():
        f0 = m.f0()
        ref = f0 / (4 * math.pi * m.alpha ** 2) + 28 * m.albedo / (23 * math.pi) * (1 - f0) * (1 - 0.5 ** 5) ** 2
        assert MT.brdf(n, n, n, m)[0] == pytest.approx(ref, rel=1e-12)
    rng = np.random.default_rng(1)
    a = rng.normal(size=(500, 3))
    b = rng.normal(size=(500, 3))
    a[:, 2], b[:, 2] = np.abs(a[:, 2]), np.abs(b[:, 2])
    N = np.broadcast_to([0, 0, 1.0], a.shape)
    m = MT.MATERIALS['glossy']
    assert np.allclose(MT.brdf(N, a, b, m), MT.brdf(N, b, a, m), rtol=1e-12, atol=0)
    b[:, 2] *= -1                                                   # below the surface: nothing
    assert np.all(MT.brdf(N, a, b, m) == 0)


def test_polariser_at_brewster_and_schlick_endpoints():
    m = MT.MATERIALS['glossy']
    nb = m.n('vis').real
    cb = math.cos(math.atan(nb))                                    # Brewster's angle: R_p = 0
    kd, ks = MT.polariser_factors(np.array([cb, 1.0]), m)
    assert kd == 0.5 and ks[0] == pytest.approx(0.0, abs=1e-12) and ks[1] == pytest.approx(0.5)
    rs, rp = MT.fresnel_exact(np.array([1.0, 0.0]), m.n())
    assert rs[0] == pytest.approx(m.f0()) and rp[0] == pytest.approx(m.f0()) and rs[1] == pytest.approx(1.0)
    assert MT.schlick_f(1.0, 0.04) == pytest.approx(0.04) and MT.schlick_f(0.0, 0.04) == pytest.approx(1.0)


# ====================================================================== lights
def test_clean_light_is_tolerant_and_idempotent():
    lt = LT.clean_light({'kind': 'laser', 'pose': {'position': [1, 'x', 2]}, 'size_m': [0.2, -1],
                         'luminous_flux_lm': {'value': -5, 'source': 'x'}, 'luminance_cd_m2': {'value': 10},
                         'ir_irradiance_w_m2': {'value': 3, 'source': 'meter', 'at_m': 0.5}, 'dim': 7,
                         'sun': {'direction': [0, 0, 1]}})
    assert lt['kind'] == 'point' and lt['size_m'] is None and lt['luminous_flux_lm'] is None
    assert lt['luminance_cd_m2'] == {'value': 10.0, 'source': 'your entry'}
    assert lt['ir_irradiance_w_m2'] == {'value': 3.0, 'source': 'meter', 'at_m': 0.5}
    assert lt['dim'] == 1.0 and 'sun' not in lt and len(lt['uid']) == 10
    assert LT.clean_light(lt) == lt
    w = LT.clean_light({'kind': 'window', 'sun': {'direction': [0, 0, 2], 'illuminance_lux': {'value': 1e4,
                                                                                            'source': 's'}}})
    assert w['sun']['direction'] == [0.0, 0.0, 1.0] and w['sun']['ir_irradiance_w_m2'] is None


def test_photometry_rules():
    pt = LT.clean_light({'kind': 'point', 'luminous_flux_lm': {'value': 4 * math.pi * 100, 'source': 's'}})
    assert LT.visible_photometry(pt)['I_cd'] == pytest.approx(100.0)
    pan = LT.clean_light({'kind': 'panel', 'size_m': [0.5, 0.4], 'luminous_flux_lm': {'value': 1000.0,
                                                                                     'source': 's'}})
    assert LT.visible_photometry(pan)['L_cd_m2'] == pytest.approx(1000 / (math.pi * 0.2))
    spot = LT.clean_light({'kind': 'spot', 'luminous_flux_lm': {'value': 500.0, 'source': 's'}})
    assert LT.visible_photometry(spot)['missing'] and LT.emitters(spot) is None     # no beam angle: not guessed
    spot['beam_deg'] = {'value': 60.0, 'source': 's'}
    assert LT.visible_photometry(LT.clean_light(spot))['I_cd'] == pytest.approx(500 / (2 * math.pi *
                                                                                        (1 - math.cos(math.pi / 6))))
    assert LT.emitters(LT.new_light('panel', [0, 0, 1])) is None    # no figures stated
    ir = LT.clean_light({'kind': 'point', 'ir_irradiance_w_m2': {'value': 2.0, 'source': 'm'}})
    assert LT.ir_photometry(ir)['missing']                          # the distance is needed
    ir['ir_irradiance_w_m2']['at_m'] = 0.5
    assert LT.ir_photometry(LT.clean_light(ir))['J_w_sr'] == pytest.approx(0.5)
    win = LT.clean_light({'kind': 'window', 'size_m': [1, 1], 'ir_irradiance_w_m2': {'value': 6.0, 'source': 'm'}})
    assert LT.ir_photometry(win)['L_w_m2_sr'] == pytest.approx(6 / math.pi)
    dim = LT.clean_light(dict(pt, dim=0.25))
    assert LT.visible_photometry(dim)['I_cd'] == pytest.approx(25.0)


def test_inverse_square_and_lambert():
    pt = LT.clean_light({'kind': 'point', 'pose': {'position': [0, 0, 0]},
                         'luminous_flux_lm': {'value': 4 * math.pi * 50, 'source': 's'}})
    em = LT.emitters(pt)
    _wi, E = LT.irradiance(em, np.array([[0, 0, 1.0], [0, 0, 2.0], [3.0, 0, 0]]))
    assert E[0] == pytest.approx([50.0, 12.5, 50 / 9])
    pan = LT.clean_light({'kind': 'panel', 'pose': {'position': [0, 0, 0], 'quaternion': [0, 0, 0, 1]},
                          'size_m': [0.01, 0.01], 'luminance_cd_m2': {'value': 1e4, 'source': 's'}})
    em = LT.emitters(pan, n_area=1)
    _wi, E = LT.irradiance(em, np.array([[0, 0, 1.0], [1.0, 0, 0.0], [0, 0, -1.0]]))
    assert E[0, 0] == pytest.approx(1.0) and E[0, 1] == pytest.approx(0.0, abs=1e-12) and E[0, 2] == 0.0


def test_through_window():
    w = LT.clean_light({'kind': 'window', 'pose': {'position': [0, 1.0, 1.0], 'quaternion': [0.7071068, 0, 0,
                                                                                            0.7071068]},
                        'size_m': [1.0, 1.0]})            # +z of the window turned to −y: it faces into the room
    sun = np.array([0.0, 1.0, 1.0]) / math.sqrt(2)        # the sun up and behind the window
    ok = LT.through_window(w, np.array([[0, 0.5, 0.5], [0, 0.0, 0.0], [3.0, 0.5, 0.5]]), sun)
    assert ok.tolist() == [True, True, False]
    assert not LT.through_window(w, np.array([[0, 0.5, 0.5]]), -sun).any()


# ====================================================================== library
def test_light_library_entries_and_projector_power():
    from dexvantage.core import library as LB
    LB.reload()
    elgato = {'elgato-key-light', 'elgato-key-light-air-mk2', 'elgato-ring-light'}
    # the three Elgato lights (later entries without a stated flux are pinned in tests/test_library_systems.py)
    es = [e for e in LB.entries('light') if e.id in elgato]
    assert {e.id for e in es} == elgato
    for e in es:
        assert LB.validate(e.data) == [] and LB.validate_minimal(e.data) == []
        assert e.sources[0]['url'].startswith('https://www.elgato.com/') and e.sources[0]['read'] == '2026-11-07'
        assert e.data['light']['ir_irradiance_w_m2'] is None and e.data['light']['luminance_cd_m2'] is None
        lt = LT.from_library(e.id)
        assert lt['library_id'] == e.id and lt['luminous_flux_lm']['source'].startswith('library: Elgato')
        assert LT.visible_photometry(lt)['L_cd_m2'] > 0
    assert LT.from_library('elgato-key-light')['luminous_flux_lm']['value'] == 2800.0
    with pytest.raises(LT.LightError):
        LT.from_library('intel-realsense-d435')
    for mid in ('intel-realsense-d415', 'intel-realsense-d435', 'intel-realsense-d435i', 'intel-realsense-d455'):
        pj = LB.get(mid).data['projector']
        assert pj['optical_power_w'] == 0.36 and pj['wavelength_nm'] == 850 and 'Table 3-1' in pj['optical_power_what']
    assert 'optical_power_w' not in (LB.get('intel-realsense-d457').data.get('projector') or {})


# ====================================================================== the study
def test_no_lights_is_said_not_raised():
    p, _u = _scene()
    p.workspace.lights = []
    doc = jobs.run_study('lighting', p)
    assert doc['headline']['word'] == 'no lights' and 'add the lab' in doc['headline']['text']
    p.workspace.lights = [LT.new_light('panel', [0, 0, 1])]          # a light with nothing stated
    doc = jobs.run_study('lighting', p)
    assert doc['headline']['word'] == 'no lights' and 'stated' in doc['headline']['text']
    json.dumps(doc)


def test_no_camera_refuses_plainly():
    p, _u = _scene()
    p.variant.sensors = []
    with pytest.raises(jobs.StudyError, match='camera or depth sensor'):
        jobs.run_study('lighting', p)


def test_glare_saturates_with_exposure_and_the_polariser_helps_at_brewster():
    vals = []
    for e in (0.001, 0.01, 0.05):
        p, u = _scene(exposure=e)
        c = jobs.run_study('lighting', p)['cameras'][0]
        assert c['absolute'] and 'absolute' in c['mode']
        vals.append(c['sat_pct'])
    assert vals[0] == 0.0 and vals[0] < vals[1] < vals[2]
    p, u = _scene(exposure=0.01)
    doc = jobs.run_study('lighting', p)
    base = doc['cameras'][0]['sat_pct']
    pol = next(r for r in doc['recommendations'] if r['kind'] == 'polariser')
    assert pol['before'] == base and pol['after'] < base and pol['delta'] < 0
    p.analysis.studies['lighting']['sensors'][u]['polariser'] = True
    assert jobs.run_study('lighting', p)['cameras'][0]['sat_pct'] == pytest.approx(pol['after'])
    # every recommendation names its figure before and after
    for r in doc['recommendations']:
        assert {'kind', 'text', 'before', 'after', 'delta', 'apply'} <= set(r)
    assert doc['suggestion'] is not None and doc['suggestion']['delta'] < 0


def test_shadows_come_from_core_occlusion():
    p, _u = _scene(exposure=0.05, blocker=True)
    c = jobs.run_study('lighting', p)['cameras'][0]
    assert c['sat_pct'] == 0.0 and c['per_pose'][0]['peak_level'] == 0.0


def test_relative_model_is_labelled_and_scale_free():
    p, _u = _scene(tag='metallic', cam=(0.4, 0.0, 0.45), light=(-0.4, 0.0, 0.45), absolute=False)
    p.workspace.surfaces = [{'uid': 'table', 'name': 'Table', 'z': 0.0, 'size': [1.0, 1.0], 'material': 'matte'}]
    d1 = jobs.run_study('lighting', p)
    c = d1['cameras'][0]
    assert not c['absolute'] and c['mode'].startswith('relative') and 'ISO 12232' in c['mode']
    assert d1['headline']['relative'] and 'relative model' in d1['headline']['text']
    assert c['sat_pct'] > 0 and c['glare_map']['level_is'].endswith('(relative)')
    p.workspace.lights[0]['luminous_flux_lm']['value'] *= 8           # the auto-exposure makes up for it
    d2 = jobs.run_study('lighting', p)
    assert d2['cameras'][0]['sat_pct'] == pytest.approx(c['sat_pct'])
    gm = c['glare_map']
    assert len(gm['sat']) == gm['grid'][0] * gm['grid'][1] and max(gm['sat']) > 0
    assert c['sat_points_mm'] and len(c['sat_points_mm'][0]) == 3
    assert d1['settings']['sat_ratio'] == pytest.approx(1 / (0.18 / math.sqrt(2)))
    assert 'sat_ratio' in d1['defaults_used']


def test_ir_washout_dark_and_clear_surfaces():
    p, _u = _scene(tag='matte', absolute=False, model_id='intel-realsense-d435', cam=(0.0, 0.0, 0.8))
    win = LT.clean_light({'kind': 'window', 'name': 'Window', 'size_m': [1.0, 1.0],
                          'pose': {'position': [0.0, 0.0, 1.5], 'quaternion': [1, 0, 0, 0]},
                          'ir_irradiance_w_m2': {'value': 20.0, 'source': 'test'}})
    p.workspace.lights.append(win)
    doc = jobs.run_study('lighting', p)
    d = doc['depth'][0]
    assert d['projector_w'] == 0.36 and 'Table 3-17' in d['projector_note']
    assert d['washout_pct'] > 50 and d['loss_pct'] >= d['washout_pct'] and d['ambient_ir_w_m2_median'] > 0
    rem = next(r for r in doc['recommendations'] if r['kind'] == 'remove_ir')
    assert rem['target'] == win['uid'] and rem['after'] < rem['before'] and 'Blind or curtain' in rem['text']
    p.workspace.lights[1]['ir_irradiance_w_m2']['value'] = 1e-4
    d = jobs.run_study('lighting', p)['depth'][0]
    assert d['washout_pct'] == 0.0
    p.workspace.objects[0].materials = ['matte', 'ir_absorbing']
    d = jobs.run_study('lighting', p)['depth'][0]
    assert d['dark_pct'] == pytest.approx(100.0) and d['loss_pct'] == pytest.approx(100.0)
    p.workspace.objects[0].materials = ['transparent']
    d = jobs.run_study('lighting', p)['depth'][0]
    assert d['through_pct'] == pytest.approx(100.0)
    p.analysis.studies['lighting']['materials'] = {p.workspace.objects[0].uid: {'depth_reads_through': False}}
    d = jobs.run_study('lighting', p)['depth'][0]
    assert d['through_pct'] == 0.0


def test_passive_depth_has_no_washout_figure():
    p, _u = _scene(absolute=False, model_id='intel-realsense-d405', cam=(0.0, 0.0, 0.5))
    doc = jobs.run_study('lighting', p)
    assert doc['depth'] == [] and any('passive stereo' in n for n in doc['notes'])


@pytest.fixture(scope='module')
def ur5e_lit():
    from dexvantage.core.robot.samples import sample_project
    p = sample_project('ur5e')
    p.workspace.lights = [LT.from_library('elgato-key-light', {'position': [0.5, 0.3, 0.9],
                                                               'quaternion': [1, 0, 0, 0]})]
    p.workspace.objects.append(TaskObject(name='Can', shape='cylinder', size=(0.03, 0.1, 0),
                                          pose=Pose(np.array([0.45, 0.0, 0.05]), np.array([0, 0, 0, 1.0])),
                                          materials=['metallic']))
    return p


def test_ur5e_sample_over_the_grasp_poses(ur5e_lit):
    doc = jobs.run_study('lighting', ur5e_lit)
    json.dumps(doc)
    assert doc['poses']['source'] == 'grasps' and 1 <= doc['poses']['n'] <= LS.DEFAULTS['max_poses']
    assert {c['name'] for c in doc['cameras']} == {'Wrist D405', 'Overhead D435'}
    assert doc['points']['gripper'] == LS.DEFAULTS['gripper_points']
    assert any('default robot material' in n for n in doc['notes'])
    assert {d['name'] for d in doc['depth']} == {'Overhead D435'}
    assert doc['depth'][0]['washout_pct'] == 0.0                      # the Key Light states no IR
    h = doc['headline']
    assert h['status'] in LS.WORDS and h['word'].startswith(tuple(LS.WORDS.values()))
    if h.get('status_from') == 'depth':          # a large depth loss raises the status; the word names both
        assert h['word'].endswith(LS.DEPTH_WORDS[h['status']])
        assert LS.depth_status_for(h['depth_loss_pct'], LS.settings_of(ur5e_lit)) == h['status']
    else:
        assert h['word'] == LS.WORDS[h['status']]
    for c in doc['cameras']:
        assert len(c['per_pose']) == doc['poses']['n'] and not c['absolute']


def test_tilt_recommendation_mount_pose_turns_about_the_optical_centre(ur5e_lit):
    from dexvantage.core.camrender import optical_pose
    p = ur5e_lit.snapshot()
    st = LS.settings_of(p)
    ev = LS.Evaluator(p, st, [])
    cam = next(c for c in ev.cams if not c.on_link)
    mp = cam.mount_after_tilt(10.0)
    T0 = optical_pose(p, cam.sensor, cam.imager, {})
    cam.sensor.mount.pose = mp
    T1 = optical_pose(p, cam.sensor, cam.imager, {})
    assert np.allclose(T0.position, T1.position, atol=1e-9)
    R0, R1 = T0.matrix()[:3, :3], T1.matrix()[:3, :3]
    assert np.allclose(R0.T @ R1, LS._rx(10.0), atol=1e-9)


def test_batch_flag_writes_the_result(tmp_path):
    out = tmp_path / 'lighting.json'
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    r = subprocess.run([sys.executable, '-m', 'dexvantage', '--sample', 'so100', '--lighting', str(out)],
                       capture_output=True, text=True, env=env, cwd=ROOT, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    doc = json.loads(out.read_text())
    assert doc['study'] == 'lighting' and 'provenance' in doc and doc['headline']['word'] == 'no lights'
    assert str(tmp_path) not in out.read_text()


# ====================================================================== validation
def test_validation_checks_pass():
    from dexvantage.validation import ValidationSettings, run
    res = run(settings=ValidationSettings(models=('lighting',), quick=True))
    ids = {c.id for c in res.checks}
    assert ids == {'lighting.ggx_normalised', 'lighting.white_furnace', 'lighting.energy', 'lighting.reciprocity',
                   'lighting.peak_closed_form', 'lighting.schlick_vs_fresnel', 'lighting.inverse_square',
                   'lighting.area_vs_closed_form', 'lighting.visibility_vs_raycast'}
    for c in res.checks:
        assert c.passed, (c.id, c.error, c.exception, c.detail)
    from dexvantage.validation.references import REFERENCES
    from dexvantage.validation.suite import CHECKS
    for s in CHECKS:
        if s.model == 'lighting':
            assert all(k in REFERENCES for k in s.refs), s.id


# ====================================================================== the page
def _wait(win, app):
    from conftest import pump
    win.ctx.analysis.wait(120)
    pump(app, 0.05)


def test_page_lights_edit_run_apply_and_overlays(app, win, tmp_path):
    from dexvantage.core.robot.samples import sample_project
    win.store.set_project(sample_project('ur5e'))
    _wait(win, app)
    assert 'lighting' in win.pages
    page = win.pages['lighting']
    win.open_page('lighting')
    assert page.head.status.text.text() == 'not run yet'
    assert page.lights.topLevelItemCount() == 0 and not page.editor.isEnabled()
    # add from the library: one undo step, the figures sourced from the entry
    n0 = win.store.undo.count()
    uid = page.add_light(library_id='elgato-key-light')
    assert uid and win.store.undo.count() == n0 + 1 and page.lights.topLevelItemCount() == 1
    lt = win.store.project.workspace.lights[0]
    assert lt['luminous_flux_lm']['value'] == 2800.0 and 'library' in lt['luminous_flux_lm']['source']
    assert 'Luminance' in page.light_note.text()
    # move it (one undo step)
    page._set_pos(0, 600.0)
    assert win.store.project.workspace.lights[0]['pose']['position'][0] == pytest.approx(0.6)
    win.store.undo.undo()
    assert win.store.project.workspace.lights[0]['pose']['position'][0] != pytest.approx(0.6)
    win.store.undo.redo()
    # an IR figure with its source and distance
    page.f_fig['ir_irradiance_w_m2'][1].setText('my meter reading, 2026-11-07')
    page._set_figure('ir_irradiance_w_m2', 1.5)
    page._set_ir_at(1.0)
    ir = win.store.project.workspace.lights[0]['ir_irradiance_w_m2']
    assert ir == {'value': 1.5, 'source': 'my meter reading, 2026-11-07', 'at_m': 1.0}
    # the glyphs are drawn on every page (an overlay source)
    assert page.light_glyphs in win.ctx.overlay_sources
    gl = page.light_glyphs(win.viewport)
    assert gl and gl[0].segments and len(gl[0].pts) >= 7
    page.run()
    _wait(win, app)
    assert page.doc is not None and page.doc['study'] == 'lighting'
    assert page.head.status.text.text() == page.doc['headline']['word']
    assert page.head.status.text.text().split(';')[0] in LS.WORDS.values()      # a depth word may follow
    assert page.cams.topLevelItemCount() == 2 and page.deps.topLevelItemCount() == 1
    assert page.map.cam is not None
    # the Camera view overlay is the glare map's cells of the shown camera
    c = page.doc['cameras'][0]
    gm = dict(c['glare_map'], sat=[0.0] * len(c['glare_map']['sat']))
    gm['sat'][5] = 0.5
    polys = __import__('dexvantage.ui.pages.lighting', fromlist=['x']).glare_polylines(dict(c, glare_map=gm), 640,
                                                                                       480, 1e9)
    assert len(polys) == 1 and len(polys[0][0]) == 1 and polys[0][0][0].shape == (5, 2)
    # Apply the suggestion (if any) is one undo step
    if page.doc.get('suggestion'):
        n1 = win.store.undo.count()
        page.apply()
        assert win.store.undo.count() == n1 + 1
        _wait(win, app)
    # export with provenance
    path = page.export(str(tmp_path / 'lighting.json'))
    assert path and 'provenance' in json.loads(open(path).read())
    # remove: one undo step
    page.remove_selected()
    assert win.store.project.workspace.lights == []
    win.store.undo.undo()
    assert len(win.store.project.workspace.lights) == 1
