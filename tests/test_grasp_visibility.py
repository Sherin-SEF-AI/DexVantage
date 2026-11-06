"""The grasp self-occlusion study (core/studies/grasp_visibility.py), its page (ui/pages/grasp.py), its batch flag
(--grasp) and its validation checks (validation/checks_grasp.py)."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from dexvantage.core import grasps as GR
from dexvantage.core import jobs
from dexvantage.core.frames import Pose
from dexvantage.core.jobs import StudyError
from dexvantage.core.robot.samples import sample_project
from dexvantage.core.studies import grasp_visibility as GV

from conftest import pump

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
FAST = {'sens_grasps': 20, 'gallery': 2, 'gallery_px': 160}


def _project(name='ur5e', n=40, seed=1, settings=None):
    p = sample_project(name)
    p.tasks.grasp_sets = []  # the sample's own grasp set and policy left out
    r = p.robots[0]
    GR.add_set(p, GR.sample(p, r.uid, kinds=('top_down', 'side', 'angled'), n=n, rng_seed=seed))
    p.analysis.studies[GV.SID] = dict(FAST, **(settings or {}))
    return p


@pytest.fixture(scope='module')
def ur5e_doc():
    p = _project()
    return p, jobs.run_study(GV.SID, p)


# ---------------------------------------------------------------------- the study on the UR5e sample
def test_result_is_json_and_consistent(ur5e_doc):
    p, doc = ur5e_doc
    json.dumps(doc, allow_nan=False)
    assert doc['schema'] == 'dexvantage.grasp_visibility/1' and not doc['partial']
    s = doc['summary']
    assert s['grasps'] == 40 and 0 < s['feasible'] <= 40
    assert s['meets'] <= s['evaluated'] <= s['feasible']
    assert sum(s['infeasible'].values()) == s['grasps'] - s['feasible']
    assert s['cvs'] == pytest.approx(s['meets'] / s['evaluated'])
    assert s['ci_low'] <= s['cvs'] <= s['ci_high'] and s['wilson_low'] <= s['cvs'] <= s['wilson_high']
    assert s['status'] in GV.WORDS and s['word'] == GV.WORDS[s['status']]
    assert 'every fingertip at least 50 % visible in at least one camera' == s['requirement']
    assert s['advice'].startswith(f'{100 * s["cvs"]:.0f} %')
    assert [c['name'] for c in doc['cameras']] == ['Wrist D405', 'Overhead D435']
    for c in doc['cameras']:
        assert set(c['fingertips']) == {'gripper_right_pad', 'gripper_left_pad'}
        pts = c['points']
        assert sum(pts.values()) == pytest.approx(1.0, abs=1e-3)
        assert len(c['precontact']) == 2
    assert doc['precontact_m'] == [0.025, 0.05]
    oc = doc['occlusion']
    assert oc['maps_built'] > 0 and oc['backend'] in ('gl', 'numpy') and oc['q_bucket_deg'] == 0.1
    assert doc['throughput']['grasps_per_s'] > 0
    assert set(doc['timings']) >= {'ik_seconds', 'occlusion_seconds', 'sensitivity_seconds', 'total_seconds'}
    assert 'q' not in doc['settings'] and 'min_visible' in doc['settings']['defaults']


def test_decode_matches_the_table(ur5e_doc):
    _p, doc = ur5e_doc
    d = GV.decode(doc)
    N, C = 40, 2
    assert d['tcp'].shape == (N, 3) and d['fingertips'].shape == (N, C, 2) and d['object'].shape == (N, C)
    assert d['precontact'].shape == (2, N, C, 2)
    feas = d['feasible']
    assert feas.sum() == doc['summary']['feasible']
    assert np.isfinite(d['q'][feas]).all() and np.isnan(d['q'][~feas]).all()
    assert np.isnan(d['fingertips'][~feas]).all() and np.isfinite(d['fingertips'][feas]).all()
    # the score is the weaker fingertip in the best camera, and meets is score >= 50 %
    sc = np.nanmax(d['fingertips'][feas].min(axis=2), axis=1)
    assert np.allclose(sc, d['score'][feas], atol=1e-4)
    assert (d['meets'][feas] == (d['score'][feas] >= 0.5)).all()
    assert d['meets'].sum() == doc['summary']['meets']
    # the grasp centres are the set's TCP targets
    p = _p
    _link, _off, T = GR.ik_targets(p, GR.sets(p)[0])
    assert np.allclose(d['tcp'], T[:, :3, 3], atol=1e-6)


def test_the_ik_solutions_reach_the_grasps(ur5e_doc):
    p, doc = ur5e_doc
    from dexvantage.core.robot import collision as CO
    from dexvantage.core.robot.model import get_model
    d = GV.decode(doc)
    r = p.robots[0]
    m = get_model(p, r.uid)
    link, off, T = GR.ik_targets(p, GR.sets(p)[0])
    for i in np.flatnonzero(d['feasible'])[:8]:
        q = d['q'][i].astype(float)
        X = m.link_pose(q, link) @ off
        assert np.linalg.norm(X[:3, 3] - T[i, :3, 3]) < 2e-4          # float32 storage of q
        assert not CO.self_collision(m, q).in_collision
        assert not CO.obstacle_distance(p, r.uid, q, 0.0).in_collision
    reasons = set(doc['grasps']['reason']) - {''}
    assert reasons <= set(['out of reach', 'joint limit', 'orientation unreachable', 'self-collision',
                           'no convergence', GV.OBSTACLE])
    if GV.OBSTACLE in reasons:
        assert doc['ik']['obstacle_pairs'] and all('×' in k for k in doc['ik']['obstacle_pairs'])


def test_sensitivity_and_gallery(ur5e_doc):
    p, doc = ur5e_doc
    sens = doc['sensitivity']
    assert len(sens) == 24 and [m['rank'] for m in sens] == list(range(1, 25))
    keys = [(-(m['delta_cvs'] or 0), -(m['delta_score'] or 0)) for m in sens]
    assert keys == sorted(keys)
    for m in sens:
        Pose.from_dict(m['mount_pose'])
        assert m['delta_low'] <= m['delta_cvs'] + 1e-9 <= m['delta_high'] + 2e-9
    move = next(m for m in sens if m['kind'] == 'move' and m['camera'] == 'Overhead D435' and m['axis'] == 'z'
                and m['sign'] == 1)
    s = p.variant.sensor(move['sensor_uid'])
    assert np.allclose(Pose.from_dict(move['mount_pose']).position - s.mount.pose.position, [0, 0, 0.02])
    turn = next(m for m in sens if m['kind'] == 'turn' and m['camera'] == 'Overhead D435')
    assert np.allclose(Pose.from_dict(turn['mount_pose']).position, s.mount.pose.position, atol=0.05)
    gal = doc['gallery']
    assert len(gal) == 2 and gal[0]['score'] <= gal[1]['score']
    v = GV.decode_view(gal[0]['views'][0])
    assert v['rgb'].shape == (90, 160, 3) and v['rgb'].dtype == np.uint8 and v['rgb'].std() > 1
    assert len(v['uv']) == len(v['state']) == 256 and set(np.unique(v['state'])) <= {0, 1, 2, 3}


def test_the_study_is_reproducible():
    a = jobs.run_study(GV.SID, _project(n=12, settings={'sensitivity': False, 'gallery': 0}))
    b = jobs.run_study(GV.SID, _project(n=12, settings={'sensitivity': False, 'gallery': 0}))
    assert a['summary'] == b['summary'] and a['grasps']['score'] == b['grasps']['score']


def test_so100_runs_position_first():
    doc = jobs.run_study(GV.SID, _project('so100', n=20, settings={'sensitivity': False, 'gallery': 0}))
    assert doc['ik']['settings']['mode'] == 'position'
    assert {f['name'] for f in doc['fingertips']} == {'Fixed_Jaw', 'Moving_Jaw'}
    assert any(f['moved_to_visual'] > 0 for f in doc['fingertips'])      # pad points snapped onto the visual mesh
    assert doc['summary']['feasible'] > 0


# ---------------------------------------------------------------------- the pieces
def test_score_rules():
    st = dict(GV.DEFAULTS)
    frac = np.array([[[0.9, 0.6], [0.2, 0.95]],          # cam 0 sees both >= 0.5
                     [[0.9, 0.1], [0.1, 0.9]]])          # each camera sees one fingertip only
    obj = np.array([[0.4, np.nan], [np.nan, np.nan]])
    assert np.allclose(GV.grasp_score(frac, obj, st), [0.6, 0.1])
    st2 = dict(st, same_camera=False)
    assert np.allclose(GV.grasp_score(frac, obj, st2), [0.9, 0.9])     # each fingertip in some camera
    st3 = dict(st, min_cameras=2)
    assert np.allclose(GV.grasp_score(frac, obj, st3), [0.2, 0.1])
    st4 = dict(st, requirement='any_fingertip')
    assert np.allclose(GV.grasp_score(frac, obj, st4), [0.95, 0.9])
    st5 = dict(st, requirement='object')
    s5 = GV.grasp_score(frac, obj, st5)
    assert s5[0] == pytest.approx(0.4) and np.isnan(s5[1])
    assert GV._applicable(obj, st5).tolist() == [True, False]
    assert GV._applicable(obj, st).tolist() == [True, True]
    st6 = dict(st, requirement='fingertips_and_object')
    assert GV.grasp_score(frac, obj, st6)[0] == pytest.approx(0.4)
    assert GV.grasp_score(frac, obj, dict(st, min_cameras=3)).tolist() == [0.0, 0.0]


def test_wilson_and_bootstrap():
    lo, hi = GV.wilson_interval(50, 100, 0.95)
    assert lo == pytest.approx(0.4038, abs=1e-4) and hi == pytest.approx(0.5962, abs=1e-4)
    lo, hi = GV.wilson_interval(0, 20, 0.95)
    assert lo == pytest.approx(0.0, abs=1e-12) and hi == pytest.approx(0.1611, abs=1e-4)
    assert GV.bootstrap_ci(np.ones(30), 500, 0.95, 1) == (1.0, 1.0)
    x = (np.random.default_rng(3).random(400) < 0.3).astype(float)
    a = GV.bootstrap_ci(x, 4000, 0.95, 7)
    w = GV.wilson_interval(int(x.sum()), 400, 0.95)
    assert abs(a[0] - w[0]) < 0.01 and abs(a[1] - w[1]) < 0.01
    assert GV.bootstrap_ci(x, 4000, 0.95, 7) == a                     # seeded
    y = np.random.default_rng(4).normal(0.0, 1.0, 200)                 # not 0/1: resampled
    lo, hi = GV.bootstrap_ci(y, 2000, 0.95, 1)
    se = y.std(ddof=1) / math.sqrt(200)
    assert lo < y.mean() < hi and (hi - lo) == pytest.approx(2 * 1.96 * se, rel=0.15)
    assert all(math.isnan(v) for v in GV.bootstrap_ci([], 100))


def test_cone_test_is_conservative():
    rng = np.random.default_rng(5)
    for _k in range(20):
        o = rng.normal(0, 0.3, 3)
        P = o + rng.normal(0, 0.05, (30, 3)) + rng.normal(0, 1, 3) * 0.4
        cone = GV._cone(o, P)
        if cone is None:
            continue
        C = o + rng.normal(0, 0.6, (400, 3))
        r = rng.uniform(0.001, 0.08, 400)
        keep = GV._in_cone(C, r, cone)
        # a sphere that holds a point of a segment from o to one of P must be kept
        t = rng.random((400, 40))
        j = rng.integers(0, len(P), (400, 40))
        S = o + t[..., None] * (P[j] - o)
        hits = (np.linalg.norm(S - C[:, None], axis=2) <= r[:, None]).any(axis=1)
        assert not (hits & ~keep).any()
        assert GV._points_in(P, cone)


def test_culling_changes_no_answer():
    p = _project(n=10)
    st = GV.settings_of(p)
    ev = GV.Evaluator(p, st)
    r = p.robots[0]
    m = ev.robots[r.uid].model
    pads = GV.snapped_pads(p, r.uid, st['snap_mm'] / 1000, [])
    link, off, T = GR.ik_targets(p, GR.sets(p)[0])
    b = GV.solve_grasps(p, r, m, link, T, off, st, m.home)
    n_cmp = 0
    for i in np.flatnonzero(b.ok)[:3]:
        qd = {r.uid: b.q[i]}
        fkd = ev.fk(qd)
        P, N, _s = GV.pad_world(pads, fkd[r.uid])
        Pl = P + ev.lift * N
        for j, c in enumerate(ev.cams):
            _R, o = c.optical(fkd)
            culled, cone = ev.build(fkd, qd, o, Pl, c, cull=True)
            full, none = ev.build(fkd, qd, o, Pl, c, cull=False)
            assert none is None and cone is not None
            assert len(ev.occluders(fkd, qd, cone, c).tris) < len(ev.occluders(fkd, qd, None, c).tris)
            assert (culled.visible(Pl, ev.eps) == full.visible(Pl, ev.eps)).all()
            n_cmp += 1
    assert n_cmp >= 2


def test_bucket_order_keeps_members_together():
    keys = [(1,), (2,), (1,), (3,), (2,), (1,)]
    order = GV._order(keys, np.random.default_rng(0))
    assert sorted(order) == list(range(6))
    runs = [keys[i] for i in order]
    for k in set(keys):
        idx = [n for n, x in enumerate(runs) if x == k]
        assert idx == list(range(idx[0], idx[0] + len(idx)))
    assert GV._bucket([0.0, math.radians(0.26)], math.radians(0.1), 5e-4, np.zeros(2, bool)) == (0, 3)


def test_settings_are_checked():
    p = _project(n=4)
    for bad, word in (({'min_visible': 1.5}, 'fraction'), ({'requirement': 'all'}, 'requirement'),
                      ({'rule': 'nearest_q0'}, 'IK rule'), ({'cube_n': 8}, 'cube-map'), ({'ci': 0.3}, 'interval'),
                      ({'ik_backend': 'trac'}, 'IK backend'), ({'max_view_angle_deg': 120}, 'max_view_angle_deg'),
                      ({'eps_mm': 'x'}, 'eps_mm'), ({'min_cameras': 0}, 'min_cameras')):
        p.analysis.studies[GV.SID] = bad
        with pytest.raises(StudyError, match=word):
            GV.settings_of(p)
    p.analysis.studies[GV.SID] = {}
    st = GV.settings_of(p)
    assert st['max_view_angle_deg'] is None and st['lift_mm'] == 2.0 and 'lift_mm' in st['defaults']


def test_plain_errors():
    p = sample_project('ur5e')
    p.tasks.grasp_sets = []  # the sample's own grasp set and policy left out
    with pytest.raises(StudyError, match='Tasks page'):
        jobs.run_study(GV.SID, p)
    p = _project(n=4)
    for s in p.variant.sensors:
        s.enabled = False
    with pytest.raises(StudyError, match='camera'):
        jobs.run_study(GV.SID, p)
    p = _project(n=4)
    p.analysis.studies[GV.SID] = {'grasp_set': 'nope'}
    with pytest.raises(StudyError, match='not in the project'):
        jobs.run_study(GV.SID, p)
    p = _project(n=4)
    p.tasks.grasp_sets[0]['robot_uid'] = 'gone'
    with pytest.raises(StudyError, match='robot that is not in the project'):
        jobs.run_study(GV.SID, p)


@pytest.mark.parametrize('backend', ['pinocchio', 'curobo'])
def test_optional_ik_backends_say_what_is_missing(backend, monkeypatch):
    import importlib.util
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, 'find_spec', lambda n, *a: None if n in ('pinocchio', 'curobo') else
                        real(n, *a))
    p = _project(n=4, settings={'ik_backend': backend})
    with pytest.raises(StudyError, match='not installed') as e:
        jobs.run_study(GV.SID, p)
    assert '"dls"' in str(e.value)


def test_partial_results_and_cancel(monkeypatch):
    monkeypatch.setattr(GV, 'BACKGROUND_GRASPS', 5)
    got = []
    p = _project(n=24, settings={'chunk': 10, 'sensitivity': False, 'gallery': 0})
    doc = jobs.run_study(GV.SID, p, jobs.Ctx(partial=got.append))
    assert got and got[0]['partial'] and got[0]['evaluated'] >= 10 and 'cvs_so_far' in got[0]
    assert not doc['partial']
    calls = {'n': 0}

    def cancel():
        calls['n'] += 1
        return calls['n'] > 3
    doc = jobs.run_study(GV.SID, _project(n=24, settings={'sensitivity': False, 'gallery': 0}),
                         jobs.Ctx(cancel=cancel))
    assert doc['cancelled']


def test_object_region_and_bucket_reuse():
    from dexvantage.core.project import TaskObject
    p = _project(n=1)
    v = p.workspace.volumes[0]
    p.workspace.objects.append(TaskObject(name='Box', shape='box', size=(0.05, 0.05, 0.05),
                                          pose=Pose(np.array([v.center[0], v.center[1], 0.025]),
                                                    np.array([0.0, 0.0, 0.0, 1.0]))))
    r = p.robots[0]
    gs = GR.sample(p, r.uid, kinds=('top_down',), n=6, rng_seed=2)
    gs['poses'] = gs['poses'] + [dict(x, id=x['id'] + '-again') for x in gs['poses']]     # duplicates: same bucket
    p.tasks.grasp_sets = []
    GR.add_set(p, gs)
    p.analysis.studies[GV.SID] = {'sensitivity': False, 'gallery': 0}
    doc = jobs.run_study(GV.SID, p)
    s = doc['summary']
    assert s['with_object'] == s['feasible'] and s['object_visible'] is not None
    oc = doc['occlusion']
    assert oc['evaluations_reused'] >= s['feasible'] and oc['cache_hit_rate'] > 0.3
    d = GV.decode(doc)
    half = len(d['score']) // 2
    assert np.allclose(np.nan_to_num(d['score'][:half], nan=-1), np.nan_to_num(d['score'][half:], nan=-1))


# ---------------------------------------------------------------------- the validation checks
@pytest.mark.parametrize('cid', ['grasp.visibility_vs_raycast', 'grasp.bootstrap_coverage', 'grasp.qbucket_vs_exact'])
def test_checks_pass_quick(cid):
    from dexvantage.validation import references as R
    from dexvantage.validation import suite as S
    import dexvantage.validation.checks_grasp  # noqa: F401
    spec = next(s for s in S.CHECKS if s.id == cid)
    assert spec.model == 'grasp' and all(k in R.REFERENCES for k in spec.refs)
    ctx = S.Ctx(None, None, None, S.ValidationSettings(quick=True))
    c = S.run_one(spec, ctx)
    assert c.passed, (c.error, c.tolerance, c.detail, getattr(c, 'exception', ''))


# ---------------------------------------------------------------------- the page
def test_page_runs_shows_and_applies(app, win, tmp_path):
    assert 'grasp' in win.pages and win.actions_by_id['grasp_study'].isEnabled()
    assert win.open_sample('ur5e')
    p = win.store.project
    win.store.edit('tasks', lambda pp: setattr(pp.tasks, 'grasp_sets', []), 'Left out the sample\'s grasp set')
    r = p.robots[0]
    gs = GR.sample(p, r.uid, kinds=('top_down', 'side'), n=16, rng_seed=4)
    win.store.edit('tasks', lambda pp: GR.add_set(pp, gs), 'Sampled grasps')
    win.store.edit('analysis', lambda pp: pp.analysis.studies.__setitem__(GV.SID, dict(FAST)), 'Fast settings')
    page = win.pages['grasp']
    win.open_page('grasp')
    pump(app, 0.05)
    assert page.head.status.text.text() == 'not run yet'
    assert page.set_box.count() == 1 and page.set_box.currentData() == GR.sets(p)[0]['uid']
    page.run()
    assert win.ctx.analysis.wait(180)
    pump(app, 0.1)
    doc = page.doc
    assert doc is not None and doc['summary']['grasps'] == 16
    assert page.head.figure.text() not in ('', '—') and page.head.status.text.text() in list(GV.WORDS.values())
    assert page.cams.topLevelItemCount() == 2 and page.moves.topLevelItemCount() == 12
    assert page.grasps.topLevelItemCount() == 16 and page.gallery_grid.count() >= 2
    # the gallery flows grasps across the width, each grasp's cameras side by side, and re-flows when it narrows
    ncell = page.gallery_grid.count()
    page.resize(1600, page.height())
    pump(app, 0.05)
    wide = page._gallery_per
    page.resize(500, page.height())
    pump(app, 0.05)
    assert wide > page._gallery_per == 1 and page.gallery_grid.count() == ncell
    rows = {}
    for gi, _j, cell in page._gallery_cells:
        rows.setdefault(gi, set()).add(page.gallery_grid.getItemPosition(page.gallery_grid.indexOf(cell))[0])
    assert all(len(rr) == 1 for rr in rows.values())                     # a grasp's views stay on one row
    assert 'q' not in (win.store.project.analysis.studies.get(GV.SID) or {})
    items = page.overlays(win.viewport)
    assert items and len(items[0].tri) == 8 * 16
    win.viewport.grab()
    cam = next(c for c in doc['cameras'] if c['name'] == 'Overhead D435')
    from dexvantage.core import camrender as CR
    from dexvantage.core import library as LB
    s = p.variant.sensor(cam['sensor_uid'])
    im = LB.sensor_imagers(s)[2]
    polys = page.camview_overlays(im.camera, CR.optical_pose(p, s, im, None))
    assert polys and all(len(x) == 3 for x in polys)
    # clicking a reachable grasp puts the robot in its pose
    i = int(np.flatnonzero(page.dec['feasible'])[0])
    page.show_grasp(i)
    assert np.allclose(win.store.q[r.uid], page.dec['q'][i], atol=1e-6)
    # settings are undoable and labelled
    sb, dl, _sc = page._spin['min_visible']
    assert dl.text() == 'default'
    sb.committed.emit(60.0)
    assert win.store.project.analysis.studies[GV.SID]['min_visible'] == pytest.approx(0.6)
    assert 'yours' in dl.text()
    win.store.undo.undo()
    assert 'min_visible' not in win.store.project.analysis.studies[GV.SID]
    # preview and apply the best move (one undo step)
    best = doc['best_move']
    assert best is not None and page.b_apply.isEnabled()
    page.preview()
    assert len(page.overlays(win.viewport)) == 3
    before = p.variant.sensor(best['sensor_uid']).mount.pose.to_dict()
    n0 = win.store.undo.index()
    assert page.apply() == best['sensor_uid']
    after = win.store.project.variant.sensor(best['sensor_uid']).mount.pose
    assert np.allclose(after.position, best['mount_pose']['position'])
    assert win.store.undo.index() == n0 + 1 and 'Run again' in page.stale_note.text()
    win.store.undo.undo()
    assert np.allclose(win.store.project.variant.sensor(best['sensor_uid']).mount.pose.position,
                       before['position'])
    # the camera chooser limits the study to one camera (one undo step)
    k = page.cam_box.findData(cam['sensor_uid'])
    assert page.cam_box.count() == 3 and k > 0
    page.cam_box.setCurrentIndex(k)
    page.cam_box.activated.emit(k)
    assert win.store.project.analysis.studies[GV.SID]['sensors'] == [cam['sensor_uid']]
    win.store.undo.undo()
    assert 'sensors' not in win.store.project.analysis.studies[GV.SID] and page.cam_box.currentIndex() == 0
    assert 'cube map' in page.explain_text()
    out = page.export(str(tmp_path / 'grasp.json'))
    exp = json.loads(open(out, encoding='utf-8').read())
    assert 'provenance' in exp and exp['summary'] == doc['summary']


def test_page_explains_a_failed_run(app, win):
    page = win.pages['grasp']
    page.run()
    assert win.ctx.analysis.wait(30)
    pump(app, 0.05)
    assert page.head.status.text.text() == 'could not run'
    text = page.head.explain.text()
    assert 'StudyError' not in text and 'Traceback' not in text and ('robot' in text or 'grasp set' in text)


# ---------------------------------------------------------------------- batch
def test_batch_flag_writes_the_study(tmp_path):
    p = _project(n=10, settings={'sens_grasps': 8, 'gallery': 1})
    path = str(tmp_path / 'cell.dexv')
    p.save(path)
    out = tmp_path / 'grasp.json'
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    r = subprocess.run([sys.executable, '-m', 'dexvantage', path, '--grasp', str(out)], capture_output=True,
                       text=True, env=env, cwd=ROOT, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    doc = json.loads(out.read_text())
    assert doc['study'] == GV.SID and doc['summary']['grasps'] == 10 and 'provenance' in doc
    assert 'Grasp self-occlusion' in r.stdout
    p.tasks.grasp_sets = []
    p.save(path)
    r = subprocess.run([sys.executable, '-m', 'dexvantage', path, '--grasp', str(tmp_path / 'x.json')],
                       capture_output=True, text=True, env=env, cwd=ROOT, timeout=300)
    assert r.returncode == 2 and 'Tasks page' in r.stderr and 'Traceback' not in r.stderr


def test_headline_uses_wilson_when_every_resample_agrees(win):
    """At 100 % (or 0 %) every bootstrap resample gives the same score; the headline shows Wilson's interval then."""
    win.open_page('grasp')
    page = win.pages['grasp']
    s = {'cvs': 1.0, 'cvs_pct': 100.0, 'ci_low': 1.0, 'ci_high': 1.0, 'ci_level': 0.95, 'ci_method': 'bootstrap',
         'wilson_low': 0.971928, 'wilson_high': 1.0, 'meets': 133, 'evaluated': 133, 'grasps': 200,
         'cvs_of_all': 0.665, 'requirement': 'r', 'advice': 'All seen.'}
    page._headline({'summary': s})
    assert page.head.figure.text() == '100.0 [97.2–100]' and "Wilson's 95 % score interval" in page.head.explain.text()
    page._headline({'summary': dict(s, cvs=0.9, ci_low=0.85, ci_high=0.94)})
    assert page.head.figure.text() == '90.0 [85–94]'
