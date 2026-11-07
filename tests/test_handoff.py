"""The visual-to-tactile handoff (spec §7.15): part 2 of core/studies/dex_hand_tactile.py (the approaches, the tactile
pads' coverage at contact, the sensory blind gap), its chart on the Multi-finger occlusion page and its validation
check (validation/checks_chunk.py dexhand.blind_gap_bruteforce).

Independent references: hand-made seen/not-seen sequences with their gaps counted by hand, a pad rectangle and points
placed in its frame, and a brute-force scan of every distance along an approach."""
import json

import numpy as np
import pytest

from dexvantage.core import jobs
from dexvantage.core.frames import Pose
from dexvantage.core.jobs import StudyError
from dexvantage.core.project import Mount, Project, Sensor
from dexvantage.core.robot import samples as S
from dexvantage.core.studies import dex_hand_tactile as D
from dexvantage.validation.checks_chunk import blind_gap_scene

from conftest import pump


def _fast(p, **kw):
    st = dict(p.analysis.studies.get(D.STUDY_ID) or {})
    st.update({'synergy_ids': ['power'], 'steps': 1, 'suggest_deg': 0.0}, **kw)
    p.analysis.studies[D.STUDY_ID] = st
    return p


def _tactile_on_tips(p, model_id='meta-digit'):
    r = p.robots[0]
    for k, tip in enumerate(r.fingertip_frames):
        p.variant.sensors.append(Sensor(name=f'Tactile {k}', kind='tactile', model_id=model_id,
                                        mount=Mount(tip, r.uid, Pose())))
    return p


@pytest.fixture(scope='module')
def dex_doc():
    p = _fast(S.sample_project('dexhand'))
    return p, jobs.run_study(D.STUDY_ID, p)


# ---------------------------------------------------------------------------------------------- the gap
def test_gap_from_samples_by_hand():
    d = [0.0, 0.01, 0.02, 0.03, 0.04]
    assert D.gap_from_samples(d, [True] * 5) == (0.0, False, 0.0)
    gap, cens, hid = D.gap_from_samples(d, [False, False, True, True, True])
    assert gap == pytest.approx(0.015) and not cens and hid == pytest.approx(0.015)
    gap, cens, hid = D.gap_from_samples(d, [False] * 5)
    assert gap == pytest.approx(0.04) and cens and hid == pytest.approx(0.04)
    # seen at contact, hidden in between: no gap before contact, but a hidden stretch
    gap, cens, hid = D.gap_from_samples(d, [True, False, False, True, True])
    assert gap == 0.0 and hid == pytest.approx(0.02)
    # order does not matter
    assert D.gap_from_samples(d[::-1], [True, True, True, False, False])[0] == pytest.approx(0.015)


def test_pad_coverage_geometry():
    pad = {'pad_mm': [20.0, 10.0], 'center': np.zeros(3), 'normal': np.array([0, 0, 1.0]),
           'u': np.array([1.0, 0, 0]), 'v': np.array([0, 1.0, 0])}
    P = np.array([[0.0, 0.0, 0.0], [0.0099, 0.0049, 0.001], [0.011, 0.0, 0.0], [0.0, 0.006, 0.0],
                  [0.0, 0.0, 0.004], [-0.0099, -0.0049, -0.0029]])
    assert D.pad_coverage(P, pad, 0.003).tolist() == [True, True, False, False, False, True]
    assert not D.pad_coverage(P, dict(pad, pad_mm=None), 0.003).any()          # size not stated: nothing known


def test_tactile_pads_assumed_on_the_patch_and_placed():
    from dexvantage.core import dexhand as DH
    from dexvantage.core import grasps as GK
    p = _tactile_on_tips(S.sample_project('dexhand'))
    r = p.robots[0]
    hand = DH.structure(p, r.uid)
    pads = D.tactile_pads(p, r, hand, {})
    assert len(pads) == 5 and all(x['placement'].startswith('assumed') for x in pads)
    assert all(x['pad_mm'] == [19.0, 16.0] and 'library entry' in x['pad_source'] for x in pads)   # DIGIT, cited
    patches = GK.contact_patches(p, r.uid, None)
    ff = [v['points'] for v in patches.values() if hand.link_group.get(v['link']) == 'ff']
    cov = D.pad_coverage(np.concatenate(ff), next(x for x in pads if x['finger'] == 'ff'), 0.003).mean()
    assert cov > 0.3
    # a pad placed 5 cm off its link senses none of the patch
    s = p.variant.sensors[-5]
    s.mount.pose = Pose(np.array([0.05, 0.0, 0.0]), np.array([0, 0, 0, 1.0]))
    pads = D.tactile_pads(p, r, hand, {})
    placed = next(x for x in pads if x['uid'] == s.uid)
    assert placed['placement'].startswith('at its mount')
    assert not D.pad_coverage(np.concatenate(ff), placed, 0.003).any()


# ---------------------------------------------------------------------------------------------- the study
def test_handoff_in_the_study_result(dex_doc):
    _p, d = dex_doc
    h = d['handoff']
    json.dumps(h, allow_nan=False)
    assert set(D.HANDOFF_FIELDS) <= set(h)
    assert len(h['approach_visibility']) == 1                        # the power posture, fully closed
    a = h['approach_visibility'][0]
    assert a['distances_m'][0] == 0.0 and a['distances_m'][-1] == pytest.approx(0.05)
    assert all(len(v) == 5 for v in a['seen_pct'] if v is not None)
    assert h['status'] in D.HANDOFF_WORDS and h['status_word'] == D.HANDOFF_WORDS[h['status']]
    # the sample has no tactile pad: nothing is touched, and the notes say what to add
    assert not any(f['touched'] for f in a['fingers'])
    assert any('No tactile sensor' in n for n in h['notes'])
    for f in a['fingers']:
        assert f['sensed_at_contact'] == f['seen_at_contact']
        assert f['flagged'] == (f['blind_gap_m'] > h['threshold_m'] + 1e-12 or not f['sensed_at_contact'])
    assert h['blind_gap_m'] == max(f['blind_gap_m'] for f in a['fingers'])


def test_handoff_alone_matches_the_study(dex_doc):
    p, d = dex_doc
    h = D.tactile_handoff(p)
    assert h['blind_gap_m'] == d['handoff']['blind_gap_m']
    assert h['approach_visibility'][0]['seen_pct'] == d['handoff']['approach_visibility'][0]['seen_pct']


def test_wall_makes_a_blind_gap_and_tactile_pads_close_it():
    p = blind_gap_scene(quick=True)
    p = _fast(p)
    h = D.tactile_handoff(p)
    gaps = {f['name']: f for f in h['approach_visibility'][0]['fingers']}
    assert any(0 < f['blind_gap_m'] < 0.04 for f in gaps.values())   # some fingertip disappears behind the wall
    assert h['status'] == 'critical'                                  # ... and nothing touches it at contact
    assert any('neither seen nor touched' in f for f in h['flags'])
    _tactile_on_tips(p)
    h2 = D.tactile_handoff(p)
    f2 = {f['name']: f for f in h2['approach_visibility'][0]['fingers']}
    assert sum(f['touched'] for f in f2.values()) >= 3              # a DIGIT pad on each fingertip
    for t in h2['tactile_coverage']:                                  # a pad smaller than its patch says so
        c = t['cases'][0]
        assert c['touched'] or 'below' in c['why']
    assert all(f2[k]['blind_gap_m'] == gaps[k]['blind_gap_m'] for k in gaps)   # touch starts at contact only
    assert all(f['sensed_at_contact'] == f['touched'] or f['seen_at_contact'] for f in f2.values())
    cov = h2['tactile_coverage'][0]['cases'][0]
    assert cov['touched'] and cov['pads'][0]['placement'].startswith('assumed')


def test_adaptive_scan_matches_brute_force():
    out = D.blind_gap_check(blind_gap_scene(quick=True))
    assert out['transitions'] >= 1 and out['error_m'] <= out['refine_m'] + 1e-12
    assert out['evaluations_adaptive'] < out['evaluations_brute']


def test_handoff_off_and_bad_settings():
    p = _fast(S.sample_project('dexhand'), handoff=False)
    assert jobs.run_study(D.STUDY_ID, p)['handoff'] is None
    p = _fast(S.sample_project('dexhand'), handoff_refine_mm=20.0)
    with pytest.raises(StudyError, match='refine'):
        D.tactile_handoff(p)
    p = _fast(S.sample_project('dexhand'), blind_gap_max_mm=-1)
    with pytest.raises(StudyError, match='blind_gap_max_mm'):
        jobs.run_study(D.STUDY_ID, p)


def test_validation_check_passes_quick():
    from dexvantage.validation import suite
    suite.load_checks()
    spec = next(s for s in suite.CHECKS if s.id == 'dexhand.blind_gap_bruteforce')
    c = suite.run_one(spec, suite.Ctx(Project(), None, None, suite.ValidationSettings(quick=True)))
    assert c.passed, (c.error, c.detail, c.exception)


def test_page_draws_the_handoff_chart(app, win):
    assert win.open_sample('dexhand')
    _fast(win.store.project)
    page = win.pages['dexhand']
    win.open_page('dexhand')
    page.run()
    assert win.ctx.analysis.wait(300)
    pump(app, 0.1)
    h = page.doc['handoff']
    assert h and page.c_appr.count() == len(h['approach_visibility'])
    assert page.handoff.series and page.handoff.reverse_x and page.handoff.vlines
    assert D.HANDOFF_WORDS[h['status']] in page.handoff_head.text()
    assert 'tactile' in page.handoff_note.text()
    page.handoff.grab()
