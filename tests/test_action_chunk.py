"""Action-chunk latency and temporal alignment (spec §7.13): core/studies/action_chunk_sync.py, its page, its batch flag
and its validation checks (validation/checks_chunk.py).

Independent references: ACT's temporal aggregation written out again here as the official code does it (an
all_time_actions table, the populated rows of a column, exp(−k·arange) weights normalised), the geometric series of
the exponential weights, and hand sums of the loop's terms."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from dexvantage.core import jobs
from dexvantage.core.jobs import StudyError
from dexvantage.core.project import Policy, Project, rig_from_dict
from dexvantage.core.robot import samples as S
from dexvantage.core.studies import action_chunk_sync as A

from conftest import pump

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def _ur5e(extra=('act', 'diffusion-policy-cnn', 'pi0-fast')):
    p = S.sample_project('ur5e')
    for lid in extra:
        p.policies.append(Policy(name=lid, library_id=lid))
    return p


def _rig_for(p):
    c = 'ctrl1'
    p.rig = rig_from_dict({'hosts': [{'uid': 'h1', 'name': 'Host', 'library_id': 'usb-3.2-gen1-host',
                                      'controllers': [{'uid': c, 'name': 'USB A', 'kind': 'usb3'}]}],
                           'links': [{'sensor_uid': s.uid, 'via': c} for s in p.variant.sensors]})
    return p


@pytest.fixture(scope='module')
def doc():
    return jobs.run_study(A.STUDY_ID, _rig_for(_ur5e()))


# ---------------------------------------------------------------------------------------------- ensembling
def test_weights_follow_acts_convention_oldest_first():
    w = A.ensemble_weights(0.01, 5)
    assert w[0] == 1.0 and np.all(np.diff(w) < 0)          # i = 0, the oldest prediction, weighs most for m > 0
    assert np.all(np.diff(A.ensemble_weights(-0.2, 5)) > 0)  # a negative m favours the newest


def test_ensembler_matches_the_official_code_written_out():
    rng = np.random.default_rng(3)
    H, T, k = 7, 25, 0.05
    chunks = rng.normal(size=(T, H, 2)) + 5.0               # never 0, so ACT's "populated" test sees every row
    all_time = np.zeros((T, T + H, 2))                      # ACT: all_time_actions[[t], t:t+num_queries] = all_actions
    ens = A.ActEnsembler(H, k)
    for t in range(T):
        all_time[t, t:t + H] = chunks[t]
        ens.add(t, chunks[t])
        col = all_time[:, t]
        col = col[np.all(col != 0, axis=1)]
        w = np.exp(-k * np.arange(len(col)))
        w = w / w.sum()
        want = (col * w[:, None]).sum(axis=0)
        assert np.allclose(ens.action(t), want, atol=1e-12)


def test_mean_age_closed_forms():
    assert A.ensemble_mean_age(0.0, 11) == pytest.approx(5.0)
    assert A.ensemble_mean_age(50.0, 11) == pytest.approx(10.0, abs=1e-12)   # all weight on the oldest
    assert A.ensemble_mean_age(-50.0, 11) == pytest.approx(0.0, abs=1e-12)   # all on the newest
    r = math.exp(-0.3)
    n = 9
    s0 = sum(r ** i for i in range(n))
    want = sum(r ** i * (n - 1 - i) for i in range(n)) / s0
    assert A.ensemble_mean_age(0.3, n) == pytest.approx(want, rel=1e-12)
    # chunks every 2 steps (H = 10): 1 or 0 steps since the newest chunk, each step covered by 5 chunks (the oldest
    # issued 8 or 9 steps ago); the step's place in the cycle is averaged
    a0 = np.mean([np.average([p + 2 * j for j in range(5)][::-1], weights=A.ensemble_weights(0.3, 5))
                  for p in (0, 1)])
    assert A.ensemble_mean_age(0.3, 10, 2) == pytest.approx(a0)
    assert A.ensemble_n_eff(0.0, 10) == pytest.approx(10.0)


# ---------------------------------------------------------------------------------------------- the loop
def _loop(stamp='host_receive', delta=0.0, skip=0, cmd_rec=None):
    dt = 0.02
    terms = [A._term('exposure_half', 0.004, 0.004, 'user', ''), A._term('readout', 0.01, 0.01, 'user', ''),
             A._term('transfer', 0.006, 0.006, 'study', ''), A._term('queue', 0.0, 1 / 30, 'default', ''),
             A._term('inference', 0.05, 0.05, 'policy', ''), A._term('tick', 0.0, dt, 'default', ''),
             A._term('command', 0.02, 0.02, 'default', '')]
    sch = {'dt': dt, 'hz': 50.0, 's': 1, 'H': 100, 'ensemble': False, 'm': None}
    st = {'command_rec_ms': cmd_rec, 'skip_steps': skip}
    return A.timing_error({'terms': terms}, sch, st, stamp, delta)


def test_timing_error_term_by_term():
    te = _loop()
    lo, mid, hi = te['e_s']
    # host stamps carry the image's age, so only queueing, inference, the tick and the step rounding remain
    assert lo == pytest.approx(0.05 - 0.01)
    assert mid == pytest.approx(1 / 60 + 0.05 + 0.01)
    assert hi == pytest.approx(1 / 30 + 0.05 + 0.02 + 0.01)
    assert _loop('mid_exposure')['e_s'][1] == pytest.approx(mid + 0.02)          # + exposure/2 + readout + transfer
    assert _loop(delta=0.04)['e_s'][1] == pytest.approx(mid - 0.04)
    assert _loop(skip=2)['e_s'][1] == pytest.approx(mid - 0.04)
    assert _loop(cmd_rec=10.0)['e_s'][1] == pytest.approx(mid + 0.01)            # a slower follower than recorded
    assert te['loop_s'][1] == pytest.approx(0.004 + 0.01 + 0.006 + 1 / 60 + 0.05 + 0.01 + 0.02)
    assert te['e_sd_s'] == pytest.approx(math.sqrt(((1 / 30) ** 2 + 0.02 ** 2 + 0.02 ** 2) / 12))


def test_camera_timing_reads_library_figures_and_says_not_stated():
    from dexvantage.core import library as LB
    e = LB.get('intel-realsense-d435')
    x, src = A.camera_timing(e, 'RGB', 'exposure')
    assert x is None and src == 'not stated by the maker'

    class _E:
        data = {'imagers': [{'name': 'Cam', 'exposure_ms': {'value': 8.0, 'source': 'datasheet p. 3'}}],
                'timing': {'readout_s': 0.012}}
        sources = [{'title': 'T', 'read': '2026-01-01'}]
        title = 'X'
    assert A.camera_timing(_E(), 'Cam', 'exposure') == (pytest.approx(0.008), 'datasheet p. 3')
    assert A.camera_timing(_E(), 'Cam', 'readout')[0] == pytest.approx(0.012)


# ---------------------------------------------------------------------------------------------- the study
def test_study_on_the_ur5e_sample(doc):
    json.dumps(doc, allow_nan=False)
    names = {r['name'] for r in doc['policies']}
    assert {'π0 (sample)', 'act', 'diffusion-policy-cnn'} <= names
    ref = {r['name']: r['reason'] for r in doc['refused']}
    assert 'pi0-fast' in ref and 'not stated' in ref['pi0-fast'] and 'Policies page' in ref['pi0-fast']
    act = next(r for r in doc['policies'] if r['name'] == 'act')
    assert act['scheme']['ensemble'] and act['scheme']['m'] == 0.01 and act['scheme']['H'] == 100
    dp = next(r for r in doc['policies'] if r['name'] == 'diffusion-policy-cnn')
    assert not dp['scheme']['ensemble'] and dp['scheme']['s'] == 8 and dp['recommendations']['temporal_ensemble_m'] \
        is None
    for r in doc['policies']:
        for c in r['cameras']:
            by = {t['key']: t for t in c['terms']}
            assert by['transfer']['origin'] == 'study' and by['transfer']['value_s'] > 0      # from the bandwidth study
            assert by['exposure_half']['origin'] == 'not_stated' and by['exposure_half']['value_s'] is None
            assert 'Half the exposure' in c['missing']
            assert by['inference']['value_s'] == pytest.approx(r['scheme']['tau_i'])
            assert c['error_s'][0] <= c['error_s'][1] <= c['error_s'][2]
    assert doc['approach']['origin'] == 'trajectory' and doc['approach']['along']
    assert doc['headline']['status'] in A.WORDS
    assert any('not stated' in n.lower() for n in doc['notes'])


def test_delay_offset_centres_the_close_and_apply_is_read_back():
    p = _ur5e(('act',))
    d = jobs.run_study(A.STUDY_ID, p)
    act = next(r for r in d['policies'] if r['name'] == 'act')
    rd = act['recommendations']['delay_offset_s']
    dt = act['scheme']['dt']
    assert abs(rd['value'] / dt - round(rd['value'] / dt)) < 1e-9                       # whole control steps
    assert abs(rd['error_s'][1]) <= dt / 2 + 1e-9
    names = A.apply_recommendations(p, d)
    assert 'act' in names
    pol = next(x for x in p.policies if x.name == 'act')
    assert pol.recommendations['delay_offset_s']['value'] == rd['value']
    assert pol.recommendations['temporal_ensemble_m']['convention'].startswith('w_i = exp(-m i)')
    d2 = jobs.run_study(A.STUDY_ID, p)
    act2 = next(r for r in d2['policies'] if r['name'] == 'act')
    assert act2['delay_offset_s'] == pytest.approx(rd['value']) and 'applied' in act2['delay_offset_source']
    assert act2['error_s'][1] == pytest.approx(rd['error_s'][1], abs=3e-6)       # both rounded to 1 µs
    assert act2['scheme']['m'] == pytest.approx(act['recommendations']['temporal_ensemble_m']['m'])
    # the record survives saving and reading the project
    q = Project.from_dict(json.loads(json.dumps(p.to_dict())))
    assert next(x for x in q.policies if x.name == 'act').recommendations == pol.recommendations


def test_constant_speed_without_a_trajectory_and_user_figures():
    p = _ur5e(('act',))
    p.tasks.trajectories = []
    s0 = p.variant.sensors[0]
    p.analysis.studies[A.STUDY_ID] = {'figures': {s0.uid: {'exposure_ms': {'value': 10.0, 'source': 'measured'},
                                                           'readout_ms': {'value': 8.0, 'source': 'measured'}}},
                                      'image_stamp': 'mid_exposure'}
    d = jobs.run_study(A.STUDY_ID, p)
    assert d['approach']['origin'] == 'default' and not d['approach']['along'] and 'default' in d['approach']['source']
    act = next(r for r in d['policies'] if r['name'] == 'act')
    c = next(c for c in act['cameras'] if c['uid'] == s0.uid)
    by = {t['key']: t for t in c['terms']}
    assert by['exposure_half']['value_s'] == pytest.approx(0.005) and by['exposure_half']['origin'] == 'user'
    assert by['readout']['value_s'] == pytest.approx(0.008)
    v = d['approach']['v']
    assert c['overshoot_m'] == pytest.approx(v * max(0.0, c['error_s'][2]), abs=1e-6)
    assert c['air_grasp_m'] == pytest.approx(v * max(0.0, -c['error_s'][0]), abs=1e-6)
    # the recommended m keeps the reaction within the tolerance with the most averaging on the grid
    mr = act['recommendations']['temporal_ensemble_m']
    tol = d['settings']['reaction_tolerance_mm'] / 1000
    if mr['met']:
        assert mr['reaction_m'] <= tol + 1e-9
        cv = mr['curve']
        ok = [n for n, r in zip(cv['n_eff'], cv['reaction_m']) if r <= tol + 1e-9]
        assert mr['n_eff'] >= max(ok) - 1e-3
    # a user speed overrides
    p.analysis.studies[A.STUDY_ID] = {'approach_speed_m_s': 0.3, 'approach_speed_source': 'my demos'}
    d = jobs.run_study(A.STUDY_ID, p)
    assert d['approach']['v'] == 0.3 and d['approach']['origin'] == 'user'


def test_map_cells_lie_in_the_task_volume(doc):
    r = doc['policies'][0]
    m = r['map']
    assert m['cells'] and m['points'] > 0
    lo, hi = (np.asarray(b) for b in m['bounds'])
    for c in m['cells']:
        assert np.all(np.asarray(c['center']) >= lo - 1e-9) and np.all(np.asarray(c['center']) <= hi + m['cell_m'])
        assert c['overshoot_m'] >= 0 and c['air_grasp_m'] >= 0


def test_refusals_are_plain_sentences():
    p = S.sample_project('ur5e')
    p.policies = []
    with pytest.raises(StudyError, match='Policies page'):
        jobs.run_study(A.STUDY_ID, p)
    p = _ur5e(('pi0-fast',))
    p.policies = [x for x in p.policies if x.library_id == 'pi0-fast']
    with pytest.raises(StudyError, match='not stated'):
        jobs.run_study(A.STUDY_ID, p)
    p = _ur5e(())
    p.analysis.studies[A.STUDY_ID] = {'tolerance_mm': -1}
    with pytest.raises(StudyError, match='tolerance_mm'):
        jobs.run_study(A.STUDY_ID, p)


def test_batch_flag_writes_the_result(tmp_path):
    out = tmp_path / 'chunk.json'
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    r = subprocess.run([sys.executable, '-m', 'dexvantage', '--sample', 'ur5e', '--action-chunk', str(out)],
                       capture_output=True, text=True, env=env, cwd=ROOT, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads(out.read_text())
    assert d['study'] == A.STUDY_ID and 'provenance' in d and d['policies']
    assert str(tmp_path) not in out.read_text()


def test_validation_checks_pass():
    from dexvantage.validation import suite
    suite.load_checks()
    ids = ('chunk.loop_vs_simulation', 'chunk.ensemble_closed_form')
    specs = [s for s in suite.CHECKS if s.id in ids]
    assert len(specs) == 2
    ctx = suite.Ctx(Project(), None, None, suite.ValidationSettings(quick=True))
    for s in specs:
        c = suite.run_one(s, ctx)
        assert c.passed, (s.id, c.error, c.detail)


# ---------------------------------------------------------------------------------------------- the page
def test_page_runs_shows_applies_and_undoes(app, win, tmp_path):
    from dexvantage.core import camrender as CR
    from dexvantage.core import library as LB
    from dexvantage.ui.pages import action_chunk as PG
    assert 'action_chunk' in win.pages
    assert win.open_sample('ur5e')
    win.store.edit('policies', lambda pr: pr.policies.append(Policy(name='ACT', library_id='act')), 'add ACT')
    page = win.pages['action_chunk']
    win.open_page('action_chunk')
    page.run()
    assert win.ctx.analysis.wait(120)
    pump(app, 0.1)
    d = page.doc
    assert d is not None and page.head.figure.text() not in ('', '—')
    assert page.head.status.text.text() in list(A.WORDS.values()) + ['closes late', 'closes early']
    i = page.pol_box.findText('ACT')
    page.pol_box.setCurrentIndex(i)
    assert page.bars.rows and page.terms.topLevelItemCount() > 0
    assert page.curve.isVisibleTo(page) and page.curve.markers
    assert 'delay offset' in page.rec.text()
    page.bars.grab()
    page.curve.grab()
    items = page.overlays(win.viewport)
    assert items and isinstance(items[0], PG.TriDraw)
    page.preview()
    assert page.overlays(win.viewport)
    s = win.store.project.variant.sensors[0]
    im = LB.sensor_imagers(s)[0]
    pose = CR.optical_pose(win.store.project, s, im, win.store.q)
    for polys, _c, _l in page.camview_overlays(im.camera, pose):
        assert all(len(x) == 2 for x in polys)
    n = win.store.undo.count()
    page.apply()
    assert win.store.undo.count() == n + 1
    pol = next(x for x in win.store.project.policies if x.name == 'ACT')
    assert 'delay_offset_s' in pol.recommendations
    assert win.ctx.analysis.wait(120)
    pump(app, 0.1)
    assert not page.b_apply.isEnabled()          # the re-run's recommendation is the one now stored
    win.store.undo.undo()
    assert next(x for x in win.store.project.policies if x.name == 'ACT').recommendations == {}
    out = page.export(str(tmp_path / 'chunk.json'))
    assert 'provenance' in json.loads(open(out, encoding='utf-8').read())


def test_page_explains_a_failed_run(app, win):
    assert win.open_sample('ur5e')
    page = win.pages['action_chunk']
    win.store.project.policies = []
    page.run()
    assert win.ctx.analysis.wait(60)
    pump(app, 0.05)
    assert page.head.status.text.text() == 'could not run'
    assert 'Traceback' not in page.head.explain.text() and 'Policies page' in page.head.explain.text()
