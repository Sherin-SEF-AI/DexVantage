"""Product review of Phases 3, 6 and 7 (a researcher's flows on the samples): the regressions it found, each fixed.

- The multi-finger study named the Robotiq pads "righ" and "lef" (character-level affixes), and on the SO-100 the
  fixed jaw (an ancestor of the moving jaw) had no links, so the study stopped with "Fixed has no contact patch".
- The hand-eye planner asked a 5-joint arm (the SO-100) for full 6-DoF poses: none of 375 candidates worked.
- Number fields showed "0.00 mm" for figures not stated: ScrubSpinBox ignored its special value text ("—").
- A cable run of zero length (Add point duplicates a point) cast NaN to int in the slack-loop interpolation.
- The cables headline said "Run 4's slack loop …" without naming the cable.
- The VLA headline said "resolved" in green when no task object was counted at all.
- The bracket STL export meshed for about a second on the UI thread, even before finding the folder did not exist.
- There was no way to try the hand-eye loop without a rig: "Simulate a capture" renders one with a known offset, and
  the solve recovers it.
- The multi-finger headline said the best single mount twice; --sample's help did not list the dexhand sample.
"""
import json
import os
import time
import warnings

import numpy as np
import pytest

from conftest import pump
from dexvantage.core import jobs
from dexvantage.core.robot.samples import sample_project


# ------------------------------------------------------------------ multi-finger: names and the fixed jaw
def test_finger_names_cut_at_word_boundaries():
    from dexvantage.core.dexhand import _short_names
    assert _short_names(['gripper_right_pad', 'gripper_left_pad']) == ['right', 'left']
    assert _short_names([f'hand_rh_{f}distal' for f in ('ff', 'mf', 'rf', 'lf', 'th')]) == \
        ['ff', 'mf', 'rf', 'lf', 'th']
    assert _short_names(['Fixed_Jaw', 'Moving_Jaw']) == ['Fixed', 'Moving']
    assert _short_names(['finger_a', 'finger_b']) == ['a', 'b']


@pytest.mark.parametrize('name,titles', [('ur5e', ['Right pad', 'Left pad']), ('so100', ['Fixed jaw', 'Moving jaw'])])
def test_gripper_fingers_have_links_and_plain_titles(name, titles):
    from dexvantage.core import dexhand as DH
    from dexvantage.core.robot.model import get_model
    p = sample_project(name)
    r = p.robots[0]
    h = DH.structure(p, r.uid, get_model(p, r.uid))
    assert [f.title for f in h.fingers] == titles
    assert all(f.links for f in h.fingers)          # the SO-100's fixed jaw owns its own link
    for f in h.fingers:
        assert h.link_group[f.tip] == f.name


def test_multi_finger_study_runs_on_the_so100_and_names_the_jaws():
    p = sample_project('so100')
    doc = jobs.run_study('dex_occlusion', p)
    assert [(f['title'], f['points']) for f in doc['fingers']] == [('Fixed jaw', 128), ('Moving jaw', 128)]
    json.dumps(doc)


# ------------------------------------------------------------------ hand-eye: a 5-joint arm, a simulated capture
def test_handeye_plan_works_on_the_5_joint_so100():
    p = sample_project('so100')
    p.analysis.studies['handeye_plan'] = {'n_poses': 12}
    doc = jobs.run_study('handeye_plan', p)
    assert doc['n_planned'] == 12 and doc['status'] in ('good', 'warning')
    assert any('position first' in n for n in doc['notes'])


def test_simulated_capture_is_solved_back_to_its_offset(tmp_path):
    from dexvantage.core.calib import capture as CAP
    from dexvantage.core.calib import simcapture as SIM
    p = sample_project('ur5e')
    p.analysis.studies['handeye_plan'] = {'n_poses': 12}
    plan = jobs.run_study('handeye_plan', p)
    sensor = p.variant.sensor(plan['camera']['sensor_uid'])
    sim = SIM.simulate(p, plan['poses'], str(tmp_path / 'cap'), sensor.uid)
    assert sim['n'] == 12 and os.path.isfile(sim['joints'])
    assert abs(sim['truth']['offset_norm_mm'] - float(np.linalg.norm(SIM.DEFAULT_OFFSET_MM))) < 1e-12
    sess = CAP.capture_folder(p, sensor, sim['images'], sim['joints'])
    assert sess['source']['detected'] == 12
    sess['source']['simulated'] = {k: sim['truth'][k] for k in ('offset_norm_mm', 'offset_angle_deg', 'X_true')}
    sess.pop('board', None)
    p.calibration['sessions'].append(sess)
    doc = jobs.run_study('handeye_solve', p)
    assert doc['ok']
    # tolerance: the solve's own 1σ here is about 0.01 mm and 0.003°; 0.2 mm and 0.05° leave room for the image noise
    assert doc['simulated']['error_mm'] < 0.2 and doc['simulated']['error_deg'] < 0.05
    assert abs(doc['apply']['change_mm'] - sim['truth']['offset_norm_mm']) < 0.2
    assert doc['notes'][0].startswith('This session is simulated') and 'not a measurement' in doc['advice']
    with pytest.raises(SIM.SimError, match='no planned poses'):
        SIM.simulate(p, [], str(tmp_path / 'none'))


def test_simulate_button_on_the_solve_page(win, app, tmp_path):
    assert win.open_sample('ur5e')
    win.ctx.analysis.wait(5)
    sv = win.pages['handeye_solve']
    assert sv.simulate_capture(str(tmp_path), background=False) is None
    assert 'run the Hand-eye pose plan first' in sv.import_note.text()
    win.ctx.store.project.analysis.studies['handeye_plan'] = {'n_poses': 8}
    win.pages['handeye_plan'].run()
    assert win.ctx.analysis.wait(120)
    pump(app, 0.05)
    n = win.store.undo.count()
    assert sv.simulate_capture(str(tmp_path), background=True) is None       # the button: a window job
    t0 = time.time()
    while 'handeye_simulate' in win._jobs and time.time() - t0 < 120:
        pump(app, 0.02)
    pump(app, 0.05)
    assert win.store.undo.count() == n + 1
    sess = win.ctx.store.project.calibration['sessions'][-1]
    assert sess['name'].endswith('(simulated)') and sess['source']['simulated']['offset_norm_mm'] > 0
    assert 'a test, not a measurement' in sv.import_note.text()
    assert str(tmp_path) not in json.dumps(sess)                             # no folders stored, only names


# ------------------------------------------------------------------ "not stated" fields
def test_number_field_shows_its_special_text_for_not_stated(app):
    from dexvantage.ui.widgets import ScrubSpinBox
    sb = ScrubSpinBox('', 'plain', 2, 'mm', step=0.1, width=110)
    sb.setRange(0.0, 100.0)
    sb.setSpecialValueText('—')
    sb.set_value(0.0)
    assert sb.lineEdit().text() == '—'
    sb.set_value(3.2)
    assert sb.lineEdit().text() == '3.20'
    got = []
    sb.committed.connect(got.append)
    sb.lineEdit().setText('—')
    sb._typed = True
    sb._on_finished()
    assert got == [0.0] and sb.lineEdit().text() == '—'
    plain = ScrubSpinBox('', 'plain', 1, 'mm')
    plain.setRange(0.0, 10.0)
    plain.set_value(0.0)
    assert plain.lineEdit().text() == '0.0'                  # no special text: a real zero


def test_cables_page_shows_a_dash_for_a_diameter_not_stated(win, app):
    assert win.open_sample('ur5e')
    pg = win.pages['cables']
    win.open_page('cables')
    pg.new_cable()
    pump(app, 0.05)
    assert pg.current()['diameter_m'] is None
    assert pg.dia.lineEdit().text() == '—' and pg.bend.lineEdit().text() == '—'
    assert pg.wrap.lineEdit().text() == 'default'


# ------------------------------------------------------------------ cables
def test_zero_length_run_does_not_make_nan():
    from dexvantage.core import cables as C
    P = np.zeros((2, 5, 3))
    P[1, :, 0] = np.linspace(0, 0.1, 5)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        out = C._interp(P, np.array([[0.0, 0.05], [0.0, 0.05]]), np.array([0.0, 0.025]))
    assert np.isfinite(out).all() and np.allclose(out[0], 0.0) and np.allclose(out[1, 1], [0.05, 0, 0])


def test_cables_headline_names_the_cable():
    from dexvantage.core.studies import cables as SC
    res = [{'name': 'Wrist cable', 'status': 'serious', 'flags': [],
            'segments': [{'flags': [{'kind': 'snag', 'severity': 'serious', 'message': "Run 2's slack loop …"}]}]}]
    s = SC.summarise(res, {'bend_factor': 1.5})
    assert s['advice'].startswith("Wrist cable: Run 2's")


# ------------------------------------------------------------------ VLA
def test_vla_headline_is_not_green_when_no_object_is_counted():
    p = sample_project('ur5e')
    doc = jobs.run_study('vla_tokens', p)
    s = doc['summary']
    assert s['fewest'] is None
    assert (s['status'], s['word']) == ('warning', 'no task object')
    assert 'fingertips are counted' in s['advice']


# ------------------------------------------------------------------ brackets: export off the UI thread
def test_bracket_export_checks_the_folder_first_and_runs_as_a_job(win, app, tmp_path):
    assert win.open_sample('ur5e')
    pg = win.pages['brackets']
    win.open_page('brackets')
    pg.run()
    assert win.ctx.analysis.wait(120)
    pump(app, 0.05)
    pg.apply()
    pump(app, 0.05)
    t0 = time.perf_counter()
    assert pg.export_stl(str(tmp_path / 'missing' / 'x.stl')) is None
    assert time.perf_counter() - t0 < 0.1                      # no meshing for a folder that is not there
    assert 'does not exist' in win.msg_label.text()
    t0 = time.perf_counter()
    assert pg.export_stl(str(tmp_path / 'b.stl'), background=True) is None
    assert time.perf_counter() - t0 < 0.1                      # the UI thread only starts the job
    while 'bracket_stl' in win._jobs and time.perf_counter() - t0 < 60:
        pump(app, 0.02)
    pump(app, 0.05)
    assert os.path.isfile(tmp_path / 'b.stl') and os.path.isfile(tmp_path / 'b.stl.json')
    assert 'Exported b.stl' in win.msg_label.text()


# ------------------------------------------------------------------ small things
def test_multi_finger_headline_names_the_best_mount_once(win, app):
    assert win.open_sample('so100')
    pg = win.pages['dexhand']
    pg.run()
    assert win.ctx.analysis.wait(120)
    pump(app, 0.05)
    assert pg.head.explain.text().count('best single mount') == 1


def test_sample_help_lists_every_sample_this_build_has(capsys):
    from dexvantage.app import main
    from dexvantage.core.robot.samples import SAMPLES
    with pytest.raises(SystemExit):
        main(['--help'])
    out = ' '.join(capsys.readouterr().out.split())
    for k, (_t, phase) in SAMPLES.items():
        if phase is None:
            assert k in out


# ------------------------------------------------------------------ large grasp sets: UI thread and the content hash
def test_project_digest_is_byte_for_byte_the_canonical_formula():
    """project_digest joins the per-section canonical JSON instead of canonicalising everything twice; the hashes
    must not change (results and --reproduce compare them)."""
    from dexvantage.core import grasps as GR
    from dexvantage.core import provenance as PV

    def reference(project):
        content = PV.plain(PV.project_content(project))
        content = PV._renumber(content, PV._uid_places(content))
        return {'sha256': PV.sha256(PV.canonical(content)),
                'sections': {k: PV.sha256(PV.canonical(v))[:16] for k, v in content.items()}}
    for name in ('so100', 'ur5e', 'dexhand'):
        p = sample_project(name)
        assert PV.project_digest(p) == reference(p), name
    p = sample_project('ur5e')
    GR.add_set(p, GR.sample(p, p.robots[0].uid, kinds=('top_down', 'side'), n=500, rng_seed=2))
    p.rig = {'hosts': [{'uid': 'h1', 'name': 'Host "A"', 'controllers': []}], 'hubs': [], 'links': []}
    assert PV.project_digest(p) == reference(p)


def test_add_set_trusted_skips_normalising_but_keeps_names_unique():
    from dexvantage.core import grasps as GR
    p = sample_project('ur5e')
    gs = GR.sample(p, p.robots[0].uid, kinds=('top_down',), n=20, rng_seed=0)
    n0 = len(GR.sets(p))
    uid = GR.add_set(p, gs, trusted=True)
    assert GR.get_set(p, uid) is gs                                 # added as it is
    uid2 = GR.add_set(p, GR.normalise_set(gs), trusted=True)
    assert uid2 != uid and len({g['name'] for g in GR.sets(p)}) == n0 + 2


def test_tasks_edit_undo_redo_with_the_single_copy(win, app):
    """store.edit no longer deep-copies a FRESH section twice: the undo states must still be independent."""
    from dexvantage.core import grasps as GR
    assert win.open_sample('ur5e')
    p = win.store.project
    n0 = len(GR.sets(p))
    t = win.pages['tasks']
    t.set_form(kinds=['top_down'], n=30, rng_seed=5)
    uid = t.apply()
    assert len(GR.sets(win.store.project)) == n0 + 1
    win.store.edit('tasks', lambda pr: GR.get_set(pr, uid).__setitem__('name', 'renamed'), 'rename')
    win.store.undo.undo()
    assert GR.get_set(win.store.project, uid)['name'] != 'renamed'
    win.store.undo.undo()
    assert GR.get_set(win.store.project, uid) is None
    win.store.undo.redo()
    win.store.undo.redo()
    assert GR.get_set(win.store.project, uid)['name'] == 'renamed'
    win.store.undo.undo()
    win.store.undo.redo()
    assert GR.get_set(win.store.project, uid)['name'] == 'renamed'


def test_tasks_buttons_sample_large_sets_on_a_worker(win, app):
    from dexvantage.ui.pages import tasks as TP
    assert win.open_sample('ur5e')
    t = win.pages['tasks']
    win.open_page('tasks')
    n = win.store.undo.count()
    t.set_form(kinds=['top_down', 'side'], n=TP.ASYNC_MIN * 2, rng_seed=1)
    t0 = time.perf_counter()
    t.b_apply.click()
    assert time.perf_counter() - t0 < 0.1                       # the click only starts the job
    while 'grasp_sample' in win._jobs and time.perf_counter() - t0 < 60:
        pump(app, 0.02)
    pump(app, 0.05)
    assert win.store.undo.count() == n + 1
    assert len(win.store.project.tasks.grasp_sets[-1]['poses']) == TP.ASYNC_MIN * 2
    t.set_form(kinds=['top_down'], n=10, rng_seed=1)               # small: sampled at once, as before
    t.b_preview.click()
    assert t.preview_set is not None and len(t.preview_set['poses']) == 10


# ------------------------------------------------------------------ policies: the messages and an impossible patch
def test_policy_messages_keep_acronyms():
    from dexvantage.core import policies as PO
    with pytest.raises(PO.PolicyError, match='The ViT patch must be a whole number'):
        PO.check_value('patch_px', 0)
    with pytest.raises(PO.PolicyError, match='The action chunk H must'):
        PO.check_value('action_chunk', -1)


def test_a_patch_larger_than_the_input_is_explained_not_counted_as_zero_tokens():
    from dexvantage.core import policies as PO
    p = sample_project('ur5e')
    pol = p.policies[0]
    pol.overrides = dict(pol.overrides, patch_px={'value': PO.check_value('patch_px', 100000),
                                                  'source': PO.user_source('a typing slip')})
    with pytest.raises(jobs.StudyError, match='larger than its input'):
        jobs.run_study('vla_tokens', p)


def test_simulated_views_are_not_flagged_as_moving(tmp_path):
    from dexvantage.core.calib import capture as CAP
    from dexvantage.core.calib import simcapture as SIM
    p = sample_project('ur5e')
    p.analysis.studies['handeye_plan'] = {'n_poses': 6}
    plan = jobs.run_study('handeye_plan', p)
    sensor = p.variant.sensor(plan['camera']['sensor_uid'])
    sim = SIM.simulate(p, plan['poses'], str(tmp_path), sensor.uid)
    sess = CAP.capture_folder(p, sensor, sim['images'], sim['joints'])
    assert not any('moving' in (ps.get('note') or '') for ps in sess['poses'])
