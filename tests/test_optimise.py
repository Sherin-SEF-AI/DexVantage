"""The optimiser (core/optimise): NSGA-II against analytic fronts, constrained domination, seeds, cancel, and the rig
optimiser and design agent on cheap test metrics (locks never changed, requirements as constraints, explanations,
accept). Adapted from Vantage tests/test_rig_designer.py (determinism with a seed, cancel, settings tolerant of junk,
accept writes a variant) for the NSGA-II core and DexVantage's factors and metrics."""
import json
import math

import numpy as np
import pytest

from dexvantage.core import factors as FA
from dexvantage.core import jobs
from dexvantage.core import metrics as M
from dexvantage.core.frames import Pose
from dexvantage.core.optimise import designer as D
from dexvantage.core.optimise import nsga2 as N
from dexvantage.core.optimise import objectives as OB
from dexvantage.core.optimise import rig as R
from dexvantage.core.optimise import search as S
from dexvantage.core.project import Mount, Project, Requirement, Sensor, Volume
from dexvantage.core.scenegeom import sensor_world_pose

TARGET = np.array([0.5, 0.0, 0.3])


# ====================================================================== NSGA-II
@pytest.mark.parametrize('kind', ['zdt1', 'zdt2'])
def test_zdt_fronts_within_igd_tolerance(kind):
    r = N.run(N.benchmark_problem(kind), pop=100, budget=25000, seed=0)
    front = r.F[r.front()]
    assert N.igd(front, N.zdt_front(kind)) < 0.01
    assert r.evaluations == 25000 and r.generations == 249


def test_constrained_problem_front_feasible_and_close():
    r = N.run(N.benchmark_problem('constr'), pop=100, budget=10000, seed=0)
    fr = r.front()
    assert (r.CV[fr] == 0).all()
    assert N.igd(r.F[fr], N.constr_front()) < 0.05
    X = r.X[fr]                                         # independently: every constraint of CONSTR holds
    assert (X[:, 1] + 9 * X[:, 0] >= 6 - 1e-9).all() and (-X[:, 1] + 9 * X[:, 0] >= 1 - 1e-9).all()


def test_seed_reproducible_and_seeds_differ():
    p = N.benchmark_problem('zdt1', 10)
    a, b = N.run(p, pop=20, budget=400, seed=5), N.run(p, pop=20, budget=400, seed=5)
    c = N.run(p, pop=20, budget=400, seed=6)
    assert np.array_equal(a.X, b.X) and np.array_equal(a.F, b.F) and np.array_equal(a.archive_F, b.archive_F)
    assert not np.array_equal(a.X, c.X)


def test_sort_and_crowding_against_brute_force():
    rng = np.random.default_rng(1)
    F = rng.integers(0, 6, (60, 3)).astype(float)
    CV = np.where(rng.random(60) < 0.3, rng.random(60), 0.0)
    fronts = N.non_dominated_sort(F, CV)
    assert sorted(np.concatenate(fronts).tolist()) == list(range(60))

    def dom(i, j):                                      # Deb's constrained domination, written out again
        fi, fj = CV[i] <= 0, CV[j] <= 0
        if fi and not fj:
            return True
        if not fi and not fj:
            return CV[i] < CV[j]
        if not fi:
            return False
        return bool((F[i] <= F[j]).all() and (F[i] < F[j]).any())
    rank = {int(i): k for k, fr in enumerate(fronts) for i in fr}
    for i in range(60):
        for j in range(60):
            if dom(i, j):
                assert rank[i] < rank[j]
        if rank[i] > 0:                                 # dominated by some member of the front before
            assert any(dom(j, i) for j in fronts[rank[i] - 1])
    cd = N.crowding_distance(np.array([[0, 4], [1, 2], [2, 1], [4, 0]], float))
    assert math.isinf(cd[0]) and math.isinf(cd[3])
    assert cd[1] == pytest.approx((2 - 0) / 4 + (4 - 1) / 4) and cd[2] == pytest.approx((4 - 1) / 4 + (2 - 0) / 4)


def test_variation_stays_in_bounds_and_categorical_moves():
    rng = np.random.default_rng(0)
    lo, hi = np.array([0.0, -2.0]), np.array([1.0, 3.0])
    a = lo + rng.random((500, 2)) * (hi - lo)
    b = lo + rng.random((500, 2)) * (hi - lo)
    c1, c2 = N.sbx(a, b, lo, hi, 15.0, 1.0, rng)
    m = N.polynomial_mutation(c1, lo, hi, 20.0, 1.0, rng)
    for x in (c1, c2, m):
        assert (x >= lo).all() and (x <= hi).all()
    assert np.allclose(c1 + c2, a + b, atol=1e-9) or not np.allclose(c1, a)      # SBX keeps the pair's mean
    x = np.zeros((200, 2))
    y = N.categorical_mutation(x, [4, 2], 1.0, rng)
    assert (y != x).all() and (y[:, 0] < 4).all() and (y[:, 1] < 2).all()


def test_cancel_stops_part_way():
    calls = []

    def ev(x):
        calls.append(1)
        return [x[0], 1 - x[0]], []
    p = N.Problem([N.Var('x', 0, 1)], 2, ev)
    r = N.run(p, pop=8, budget=200, seed=0, cancel=lambda: len(calls) >= 13)
    assert r.cancelled and r.evaluations == 13 and len(r.archive_F) == 13


def test_mixed_genes_find_the_categorical_optimum():
    def ev(x):
        k, t = x
        return [abs(t - 0.3) + {'a': 1.0, 'b': 0.0, 'c': 2.0}[k], -t], []
    p = N.Problem([N.Var('k', choices=['a', 'b', 'c']), N.Var('t', 0.0, 1.0)], 2, ev)
    r = N.run(p, pop=16, budget=400, seed=2)
    best = r.archive_X[np.argmin(r.archive_F[:, 0])]
    assert p.decode(best)[0] == 'b' and abs(p.decode(best)[1] - 0.3) < 0.05


# ====================================================================== cheap metrics for the rig searches
@pytest.fixture(autouse=True, scope='module')
def fake_metrics():
    @jobs.study('_opt_geo', 'Test geometry', quick=False)
    def _geo(project, ctx):
        cams = [s for s in project.variant.sensors if s.enabled and s.kind in ('camera', 'depth')]
        d = [float(np.linalg.norm(sensor_world_pose(project, s).position - TARGET)) for s in cams]
        return {'dist': min(d) if d else None, 'cams': len(cams), 'height': max(
            [float(sensor_world_pose(project, s).position[2]) for s in cams] or [0.0])}
    OB.ensure_metrics()
    M.metric('_opt.dist', 'Distance to the target', 'm', 'lower', study='_opt_geo')(lambda d: d['dist'])
    M.metric('_opt.cams', 'Cameras', 'count', 'lower', study='_opt_geo')(lambda d: d['cams'])
    M.metric('_opt.height', 'Highest camera', 'm', 'higher', study='_opt_geo')(lambda d: d['height'])
    yield
    jobs.STUDIES.pop('_opt_geo', None)
    for k in ('_opt.dist', '_opt.cams', '_opt.height'):
        M.METRICS.pop(k, None)
    S.clear_cache()


def _proj():
    p = Project('Optimiser test')
    p.workspace.volumes.append(Volume(center=tuple(TARGET), size=(0.3, 0.3, 0.2)))
    p.variant.sensors = [
        Sensor(name='A', mount=Mount('cage', '', Pose(np.array([1.0, 0.0, 1.0])))),
        Sensor(name='B', mount=Mount('world', '', Pose(np.array([0.0, 1.0, 1.0])))),
        Sensor(name='L', locked=True, mount=Mount('cage', '', Pose(np.array([-1.0, -1.0, 2.0])))),
    ]
    p.analysis.studies['optimise'] = {'objectives': ['_opt.dist', '_opt.cams'], 'budget': 40, 'population': 8,
                                      'move_m': 0.3, 'turn_deg': 10, 'screen': {}, 'seed': 1}
    return p


def test_rig_optimiser_front_locks_and_json():
    p = _proj()
    before = json.dumps(p.to_dict(), sort_keys=True)
    doc = jobs.run_study('optimise', p)
    assert json.dumps(p.to_dict(), sort_keys=True) == before          # running it changed nothing
    json.dumps(doc, allow_nan=False)
    assert [o['id'] for o in doc['objectives']] == ['_opt.dist', '_opt.cams']
    assert doc['locked'] == ['L'] and not any('L' == f['title'].split()[0] for f in doc['factors'])
    assert doc['search']['evaluations'] == 40 and doc['current']['figures']['_opt.cams'] == 3
    opts = doc['options']
    assert opts and all(o['explanation'] and o['label'] for o in opts)
    F = np.array([[o['figures']['_opt.dist'], o['figures']['_opt.cams']] for o in opts])
    for i in range(len(F)):                                            # non-dominated among themselves
        assert not any((F[j] <= F[i]).all() and (F[j] < F[i]).any() for j in range(len(F)) if j != i)
    for o in opts:                                                     # the figures are the design's, re-measured
        q = R.option_project(p, doc, o['label'])
        assert next(s for s in q.variant.sensors if s.name == 'L').to_dict() == \
            next(s for s in p.variant.sensors if s.name == 'L').to_dict()
        g = jobs.run_study('_opt_geo', q)
        assert g['cams'] == o['figures']['_opt.cams'] and g['dist'] == pytest.approx(o['figures']['_opt.dist'])
    assert doc['headline']['status'] in ('good', 'warning') and doc['headline']['status_word']


def test_locks_never_violated_property():
    rng = np.random.default_rng(7)
    for case in range(12):
        p = Project('Locks')
        n = int(rng.integers(2, 6))
        for k in range(n):
            pos = rng.uniform(-1, 1, 3) + [0, 0, 1.0]
            p.variant.sensors.append(Sensor(name=f'S{k}', locked=bool(rng.random() < 0.5),
                                            mount=Mount(str(rng.choice(['cage', 'world', 'tripod'])), '',
                                                        Pose(pos))))
        if all(s.locked for s in p.variant.sensors):
            p.variant.sensors[0].locked = False
        locked = {s.uid: s.to_dict() for s in p.variant.sensors if s.locked}
        st = R.settings_of(p, {'objectives': ['_opt.dist', '_opt.cams', '_opt.height'], 'budget': 16,
                               'population': 4, 'move_m': 0.5, 'turn_deg': 45, 'swap_models': bool(case % 2),
                               'screen': {}, 'seed': case,
                               # a factor naming a locked sensor is refused, never applied
                               'factors': [] if case % 3 else [{'path': f'sensor:{u}.x', 'lo': -1, 'hi': 1}
                                                               for u in locked] + [
                                   {'path': f'sensor:{s.uid}.z', 'lo': 0, 'hi': 2} for s in p.variant.sensors
                                   if not s.locked]})
        seen = []
        real = FA.apply

        def spy(project, values, *a, **kw):
            out = real(project, values, *a, **kw)
            seen.append({s.uid: s.to_dict() for s in out.variant.sensors if s.locked})
            return out
        FA.apply = spy
        try:
            doc = R.optimise(p, st)
        finally:
            FA.apply = real
        assert seen and all(x == locked for x in seen), case
        for o in doc['options']:
            q = R.option_project(p, doc, o['label'])
            assert {s.uid: s.to_dict() for s in q.variant.sensors if s.locked} == locked
        if locked and not st['factors'] == []:
            assert any('locked' in n for n in doc['notes'])


def test_seeded_rig_search_reproducible():
    p = _proj()
    a, b = R.optimise(p, R.settings_of(p)), R.optimise(p, R.settings_of(p))
    assert [o['values'] for o in a['options']] == [o['values'] for o in b['options']]
    assert [o['figures'] for o in a['options']] == [o['figures'] for o in b['options']]


def test_requirements_constrain_and_unknown_ones_are_listed():
    p = _proj()
    p.requirements = [Requirement(name='Close enough', metric='_opt.dist', op='<=', threshold=0.9, unit='m',
                                  is_default=False),
                      Requirement(name='Ghost', metric='nobody.knows', op='>=', threshold=1)]
    doc = R.optimise(p, R.settings_of(p, dict(p.analysis.studies['optimise'], budget=48)))
    assert [c['name'] for c in doc['constraints']] == ['Close enough']
    assert doc['skipped_requirements'][0]['name'] == 'Ghost'
    feas = [o for o in doc['options'] if o['feasible']]
    assert feas, 'the search should find a rig within 0.9 m'
    for o in feas:
        assert o['figures']['_opt.dist'] <= 0.9 and 'meets every requirement' in o['explanation']
    for o in doc['options']:
        assert o['feasible'] == (o['figures']['_opt.dist'] <= 0.9)


def test_violation_rules():
    row = {'name': 'r', 'metric': 'x', 'op': '>=', 'threshold': 95.0, 'unit': '%'}
    assert OB.violation(row, 96) == 0 and OB.violation(row, 90) == pytest.approx(5 / 95)
    assert OB.violation(row, None) == 1.0
    assert OB.violation(dict(row, op='<', threshold=0.0, unit='count'), 0) > 0
    assert OB.violation(dict(row, op='<', threshold=2.0, unit='ms'), 1.0) == 0


def test_cost_left_out_without_prices_and_unknown_objectives():
    p = _proj()
    st = R.settings_of(p, dict(p.analysis.studies['optimise'], objectives=['_opt.dist', 'rig.cost', 'no.such'],
                               budget=8, population=4))
    doc = R.optimise(p, st)
    why = {d['id']: d['reason'] for d in doc['dropped']}
    assert 'rig.cost' in why and 'no price' in why['rig.cost'] and 'guess' in why['rig.cost']
    assert 'no.such' in why and [o['id'] for o in doc['objectives']] == ['_opt.dist']


def test_nothing_to_optimise_is_a_sentence():
    p = _proj()
    for s in p.variant.sensors:
        s.locked = True
    with pytest.raises(jobs.StudyError, match='nothing to optimise'):
        R.optimise(p, R.settings_of(p))


def test_settings_tolerant_of_junk():
    p = Project()
    st = R.settings_of(p, {'budget': 'lots', 'population': -3, 'seed': 3.7, 'objectives': 'x', 'move_m': 'far',
                           'screen': 'none', 'factors': [None, {'path': 'sensor:a.x'}], 'confirm': 1})
    assert st['budget'] == 96 and st['population'] == 4 and st['seed'] == 4
    assert st['objectives'] == list(OB.DEFAULT_OBJECTIVES) and st['move_m'] == 0.05
    assert st['screen'] == S.SCREEN and st['factors'] == [{'path': 'sensor:a.x'}] and st['confirm'] is True
    d = D.settings_of(p, {'max_cameras': 99, 'static_frames': ['moon', 'cage'], 'distance_m': [2, 1]})
    assert d['max_cameras'] == 8 and d['static_frames'] == ['cage'] and d['distance_m'] == [0.45, 1.2]


def test_accept_is_in_place_and_refuses_a_changed_rig():
    p = _proj()
    doc = R.optimise(p, R.settings_of(p))
    o = doc['options'][0]
    q = p.snapshot()
    changed = R.accept(q, doc, o['label'])
    assert changed or not o['changes']
    assert len(q.variants) == 1
    g = jobs.run_study('_opt_geo', q)
    assert g['dist'] == pytest.approx(o['figures']['_opt.dist'])
    assert q.analysis.studies['optimise']['accepted'][-1]['design'] == o['label']
    with pytest.raises(jobs.StudyError, match='changed since'):
        R.accept(q, doc, o['label'])                    # the rig is no longer the one searched
    with pytest.raises(jobs.StudyError):
        R.option(doc, 'Z')


# ====================================================================== the design agent
def _dproj():
    p = _proj()
    p.analysis.studies['design_agent'] = {'objectives': ['_opt.dist', '_opt.cams'], 'budget': 40, 'population': 8,
                                          'models': ['intel-realsense-d435', 'intel-realsense-d405', 'not-a-device'],
                                          'max_cameras': 2, 'screen': {}, 'seed': 3}
    return p


def test_design_agent_rigs_explained_and_locked_kept():
    p = _dproj()
    p.requirements = [Requirement(name='Close enough', metric='_opt.dist', op='<=', threshold=0.8, unit='m')]
    doc = jobs.run_study('design_agent', p)
    json.dumps(doc, allow_nan=False)
    assert doc['kept'] == ['L'] and doc['models'] == ['intel-realsense-d435', 'intel-realsense-d405']
    assert any('not-a-device' in n for n in doc['notes'])
    assert [s['key'] for s in doc['sites']] == ['static:cage']
    assert doc['options'] and all(o['explanation'] for o in doc['options'])
    for o in doc['options']:
        assert all(s['model_id'] in doc['models'] and s['mount']['parent_frame'] == 'cage' for s in o['sensors'])
        assert o['figures']['_opt.cams'] == len(o['sensors']) + 1           # + the locked one
        v = D.option_variant(p, doc, o['label'])
        q = p.snapshot()
        q.variants.append(v)
        q.active_uid = v.uid
        g = jobs.run_study('_opt_geo', q)
        assert g['dist'] == pytest.approx(o['figures']['_opt.dist']) and g['cams'] == o['figures']['_opt.cams']
        assert next(s for s in v.sensors if s.name == 'L').to_dict() == \
            next(s for s in p.variant.sensors if s.name == 'L').to_dict()
        # each camera looks at the task volume's centre
        for s in v.sensors:
            if s.name != 'L':
                ps = sensor_world_pose(q, s)
                x = ps.matrix()[:3, 0]
                to = TARGET - ps.position
                assert float(x @ to / np.linalg.norm(to)) > 0.999
    assert any(o['feasible'] for o in doc['options'])
    assert any('unlocked sensors (A, B)' in n for n in doc['notes'])


def test_design_agent_accept_adds_a_variant():
    p = _dproj()
    doc = D.design(p, D.settings_of(p))
    q = p.snapshot()
    n = len(q.variants)
    v = D.accept(q, doc, doc['options'][0]['label'], 'the closest view')
    assert len(q.variants) == n + 1 and q.active_uid == v.uid and v.name.startswith('Design A')
    assert q.analysis.studies['design_agent']['accepted'][-1]['why'] == 'the closest view'
    assert q.log[-1]['kind'] == 'decision'
    v2 = D.accept(q, doc, doc['options'][0]['label'])
    assert v2.name != v.name and v2.uid != v.uid


def test_design_agent_cancel_and_no_sites():
    p = _dproj()
    calls = []

    def cancel():
        calls.append(1)
        return len(calls) > 40
    doc = D.design(p, D.settings_of(p, dict(p.analysis.studies['design_agent'], budget=400)),
                   jobs.Ctx(cancel=cancel))
    assert doc['partial'] and doc['search']['stopped'] == 'cancelled' and doc['search']['evaluations'] < 400
    with pytest.raises(jobs.StudyError, match='nowhere'):
        D.design(p, D.settings_of(p, dict(p.analysis.studies['design_agent'], static_frames=[])))


def test_wrist_site_on_the_ur5e_sample_looks_past_the_fingertips():
    from dexvantage.core.robot.samples import sample_project
    p = sample_project('ur5e')
    st = D.settings_of(p, {'static_frames': []})
    sl, _notes = D.sites(p, st)
    assert [s.kind for s in sl] == ['link'] and sl[0].parent_frame == 'gripper_base'
    site = sl[0]
    for a, b, c in ((0.0, 0.0, 0.0), (0.3, 1.0, 1.0), (0.8, 0.5, 0.2)):
        pose = D.place(site, a, b, c, st)
        x = pose.matrix()[:3, 0]
        look = site.target + D.WRIST_LOOK_BEYOND_M * site.axis - pose.position
        assert float(x @ look / np.linalg.norm(look)) > 0.999
        off = pose.position - site.target
        along = float(off @ site.axis)
        radial = np.linalg.norm(off - along * site.axis)
        assert st['wrist_back_m'][0] - 1e-9 <= -along <= st['wrist_back_m'][1] + 1e-9
        assert st['wrist_radial_m'][0] - 1e-9 <= radial <= st['wrist_radial_m'][1] + 1e-9


def test_explanation_words_come_from_the_figures():
    objs = [{'id': 'a', 'title': 'Contact visibility', 'unit': '%', 'better': 'higher'},
            {'id': 'rig.camera_count', 'title': 'Cameras', 'unit': 'count', 'better': 'lower'}]
    A = {'label': 'A', 'figures': {'a': 97.0, 'rig.camera_count': 2.0}, 'violations': []}
    B = {'label': 'B', 'figures': {'a': 91.5, 'rig.camera_count': 1.0}, 'violations': []}
    cur = {'figures': {'a': 92.5, 'rig.camera_count': 2.0}}
    t = S.explain(A, [A, B], cur, objs, [])
    assert '4.5 points more contact visibility (97.0 %)' in t and 'the highest contact visibility' in t
    assert 'Against design B it trades one more camera for 5.5 points more contact visibility' in t
    t = S.explain(B, [A, B], cur, objs, [])
    assert 'one fewer camera' in t and '1.0 points less contact visibility (91.5 %)' in t


def test_designs_the_current_rig_dominates_are_not_options():
    entries = [{'f': [1.0, 1.0], 'cv': 0.0, 'n': 'worse'}, {'f': [-1.0, 3.0], 'cv': 0.0, 'n': 'trade'},
               {'f': [0.0, 0.0], 'cv': 0.0, 'n': 'same'}]
    got = [e['n'] for e in S.choose(entries, 2, 5, reference={'f': [0.0, 0.0], 'cv': 0.0})]
    assert got == ['trade', 'same'] or got == ['trade']
    assert 'worse' not in got
    # an infeasible current rig is dominated by any feasible design
    got = [e['n'] for e in S.choose(entries, 2, 5, reference={'f': [-5.0, -5.0], 'cv': 1.0})]
    assert 'worse' not in got and 'same' in got
    h = S.headline([], {'f': [0.0, 0.0], 'figures': {}}, [], False, 2)
    assert h['status_word'] == 'current rig is best'


def test_batch_optimise_with_settings_and_refusals(tmp_path):
    import os
    import subprocess
    import sys
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')

    def run(*args):
        return subprocess.run([sys.executable, '-m', 'dexvantage', *args], capture_output=True, text=True, env=env,
                              cwd=root, timeout=300)
    out = tmp_path / 'opt.json'
    st = tmp_path / 'st.json'
    st.write_text(json.dumps({'budget': 8, 'population': 4, 'seed': 2,
                              'objectives': ['grasp.contact_visibility', 'rig.camera_count']}))
    r = run('--sample', 'so100', '--optimise', str(out), '--optimise-settings', str(st))
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text())
    assert doc['study'] == 'optimise' and doc['search']['evaluations'] == 8 and doc['settings']['seed'] == 2
    assert doc['provenance'] and doc['headline']['status_word']
    r = run('--sample', 'so100', '--optimise-settings', '{"budget": 8}')
    assert r.returncode == 2 and 'goes with --optimise' in r.stderr and 'Traceback' not in r.stderr
    r = run('--sample', 'so100', '--optimise', str(out), '--optimise-settings', '{nope')
    assert r.returncode == 2 and 'not JSON' in r.stderr and 'Traceback' not in r.stderr
