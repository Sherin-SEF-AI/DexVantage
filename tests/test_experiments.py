"""Designed experiments (core/experiments.py, core/studies/experiments.py, Analysis ▸ Designed experiments).

Ported from Vantage tests/test_experiments.py (the estimator tests: Ishigami's Sobol indices, interval coverage,
degenerate responses, Latin hypercube stratification) and adapted: DexVantage's designs (full and fractional factorial,
Latin hypercube, Sobol sequence, Saltelli), factor paths of core/factors.py, responses that are named metrics of
core/metrics.py run through core/jobs (a toy study registered here), the process pool, cancel, batch and the page.

The toy study and its metrics are registered at import (module level), so a spawned worker process that imports this
module (core.experiments._modules_for) has them too; they are removed when the module's tests end.
"""
import copy
import csv
import io
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from dexvantage.core import experiments as EX
from dexvantage.core import factors as F
from dexvantage.core import jobs
from dexvantage.core import metrics as M
from dexvantage.core.project import Project, Sensor

A_ISH, B_ISH = 7.0, 0.1
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


# ============================================================================ a toy study and its metrics
def _toy_value(x, pitch, enabled, gain):
    return 100.0 * x - 0.5 * (pitch - 3.0) ** 2 + (5.0 if enabled else 0.0) + gain


CALLS = []


@jobs.study('_exp_toy', 'Experiments toy', quick=False)
def _toy(project, ctx):
    CALLS.append(1)
    s = next(s for s in project.variant.sensors if s.name == 'Wrist')
    x = float(s.mount.pose.position[0])
    pitch = F.get(project, f'sensor:{s.uid}.pitch')
    gain = float((project.analysis.studies.get('_exp_toy') or {}).get('gain', 0.0))
    return {'score': _toy_value(x, pitch, s.enabled, gain), 'x': x}


@jobs.study('_exp_refuse', 'Experiments refusing toy', quick=False)
def _refuse(project, ctx):
    s = next(s for s in project.variant.sensors if s.name == 'Wrist')
    if not s.enabled:
        raise jobs.StudyError('The wrist sensor is disabled.')
    return {'v': 1.0}


M.metric('_exp.score', 'Toy score', 'pt', 'higher', study='_exp_toy')(lambda d: d['score'])
M.metric('_exp.x', 'Toy x', 'm', 'lower', study='_exp_toy')(lambda d: d['x'])
M.metric('_exp.r', 'Toy refusing', '', 'higher', study='_exp_refuse')(lambda d: d['v'])


@pytest.fixture(scope='module', autouse=True)
def _unregister():
    yield
    for k in ('_exp_toy', '_exp_refuse'):
        jobs.STUDIES.pop(k, None)
    for k in ('_exp.score', '_exp.x', '_exp.r'):
        M.METRICS.pop(k, None)


def _proj():
    p = Project('Experiments cell')
    p.variant.sensors.append(Sensor(name='Wrist'))
    p.variant.sensors.append(Sensor(name='Top', locked=True))
    return p


def _ishigami(v):
    x1, x2, x3 = v['x1'], v['x2'], v['x3']
    return math.sin(x1) + A_ISH * math.sin(x2) ** 2 + B_ISH * x3 ** 4 * math.sin(x1)


def _ishigami_indices():
    """Ishigami and Homma (1990): the analytic first-order and total indices for x_i ~ U(−π, π)."""
    a, b, pi = A_ISH, B_ISH, math.pi
    V = a * a / 8 + b * pi ** 4 / 5 + b * b * pi ** 8 / 18 + 0.5
    V1 = 0.5 * (1 + b * pi ** 4 / 5) ** 2
    V2 = a * a / 8
    V13 = b * b * pi ** 8 * (1 / 18 - 1 / 50)
    return [V1 / V, V2 / V, 0.0], [(V1 + V13) / V, V2 / V, V13 / V]


def _ish_design(N, seed=0, bootstrap=400):
    return {'factors': [{'path': f'x{i}', 'lo': -math.pi, 'hi': math.pi} for i in (1, 2, 3)],
            'design': {'method': 'saltelli', 'n': N, 'seed': seed}, 'analysis': {'bootstrap': bootstrap}}


# ============================================================================ estimators against references
def test_ishigami_sobol_indices_within_their_intervals():
    """Jansen estimators on the Ishigami function: every analytic index inside its 95 % bootstrap interval, the
    estimates within 0.02 of it at N = 4096, S1 ≤ ST, and the variance the function's own."""
    s1, st_ = _ishigami_indices()
    assert s1[0] == pytest.approx(0.3139, abs=1e-4) and st_[2] == pytest.approx(0.2437, abs=1e-4)
    r = EX.run_function(_ishigami, _ish_design(4096, seed=11))
    assert len(r['rows']) == 4096 * 5
    s = r['analysis']['y']['sobol']
    assert s['N'] == 4096 and s['dropped'] == 0 and 'Jansen' in s['estimator']
    for i in range(3):
        assert s['S1'][i] == pytest.approx(s1[i], abs=0.02)
        assert s['ST'][i] == pytest.approx(st_[i], abs=0.02)
        assert s['S1_ci'][i][0] <= s1[i] <= s['S1_ci'][i][1]
        assert s['ST_ci'][i][0] <= st_[i] <= s['ST_ci'][i][1]
        assert s['S1'][i] <= s['ST'][i] + 0.02
    assert s['V'] == pytest.approx(A_ISH ** 2 / 8 + B_ISH * math.pi ** 4 / 5 + B_ISH ** 2 * math.pi ** 8 / 18 + 0.5,
                                   rel=0.03)


def test_sobol_intervals_cover_the_analytic_indices_over_seeds():
    """The bootstrap intervals are honest: over 12 seeds of a small design (N = 256) the analytic indices fall inside
    them at least 85 % of the time (nominal 95 %), and their width shrinks like 1/√N."""
    s1, st_ = _ishigami_indices()
    hit = tot = 0
    widths = {}
    for N, seeds in ((256, 12), (1024, 4)):
        for seed in range(seeds):
            s = EX.run_function(_ishigami, _ish_design(N, seed, 200))['analysis']['y']['sobol']
            widths.setdefault(N, []).append(np.mean([s['ST_ci'][i][1] - s['ST_ci'][i][0] for i in range(3)]))
            if N == 256:
                for i in range(3):
                    hit += s['S1_ci'][i][0] <= s1[i] <= s['S1_ci'][i][1]
                    hit += s['ST_ci'][i][0] <= st_[i] <= s['ST_ci'][i][1]
                    tot += 2
    assert hit / tot >= 0.85
    assert 1.5 < np.mean(widths[256]) / np.mean(widths[1024]) < 2.7          # √4 = 2


def test_sobol_degenerate_cases():
    d = {'factors': [{'path': 'a', 'lo': 0, 'hi': 1}, {'path': 'b', 'lo': 0, 'hi': 1}],
         'design': {'method': 'saltelli', 'n': 64}}
    s = EX.run_function(lambda v: 3.0, d)['analysis']['y']['sobol']
    assert s['S1'] == [None, None] and 'does not vary' in s['note']
    # missing values: the rows are left out, the estimate stays right for an additive function
    d['design']['n'] = 1024
    f = lambda v: float('nan') if v['a'] > 0.9 else 2 * v['a'] + v['b']     # noqa: E731
    a = EX.run_function(f, d)['analysis']['y']
    s = a['sobol']
    assert 0 < s['dropped'] < 1024 * 0.4 and s['N'] + s['dropped'] == 1024 and a['missing'] > 0
    # V(2a) = 4/12 over a's kept range ~ [0, 0.9]: a share of about 0.76 for a, 0.24 for b
    assert s['S1'][0] == pytest.approx(0.76, abs=0.07) and s['S1'][1] == pytest.approx(0.24, abs=0.07)


def test_latin_hypercube_stratifies_every_factor_and_is_seeded():
    for n in (7, 50):
        d = EX.Design.from_dict({'factors': [{'path': f'f{j}', 'lo': 0.0, 'hi': 1.0} for j in range(4)],
                                 'design': {'method': 'lhs', 'n': n, 'seed': 5}})
        U = EX.make_plan(d).U
        assert U.shape == (n, 4)
        for j in range(4):
            assert sorted((U[:, j] * n).astype(int)) == list(range(n))       # one point per stratum
    mk = lambda s: EX.make_plan(EX.Design.from_dict({'factors': [{'path': 'a', 'lo': 0, 'hi': 1}],      # noqa: E731
                                                     'design': {'method': 'lhs', 'n': 20, 'seed': s}})).U
    assert np.array_equal(mk(1), mk(1)) and not np.array_equal(mk(1), mk(2))


def test_sobol_sequence_matches_scipy_and_its_discrepancy():
    from scipy.stats import qmc
    d = EX.Design.from_dict({'factors': [{'path': f'f{j}', 'lo': 0, 'hi': 1} for j in range(3)],
                             'design': {'method': 'sobol', 'n': 128, 'seed': 4}})
    U = EX.make_plan(d).U
    assert np.allclose(U, EX._qmc_sobol(3, 4).random(128))
    assert EX.centred_l2_discrepancy(U) == pytest.approx(qmc.discrepancy(U, method='CD'), rel=1e-9)
    assert EX.centred_l2_discrepancy(U) < 0.2 * EX.iid_discrepancy(128, 3)
    # n not a power of two: allowed, said
    d2 = copy.deepcopy(d)
    d2.n = 100
    assert EX.make_plan(d2).info['power_of_two'] is False


# ============================================================================ factorial designs and main effects
def test_full_factorial_levels_kinds_and_order():
    d = EX.Design.from_dict({'factors': [{'path': 'a', 'lo': 0, 'hi': 1, 'n_levels': 3},
                                         {'path': 'c', 'levels': ['x', 'y']}, {'path': 'b', 'kind': 'bool'},
                                         {'path': 'n', 'lo': 1, 'hi': 4, 'integer': True}],
                             'design': {'method': 'full_factorial', 'levels': 2}})
    assert EX.point_count(d) == 3 * 2 * 2 * 2 and EX.design_text(d) == '24 runs = 3 × 2 × 2 × 2'
    rows = EX.plan_rows(EX.make_plan(d), d.factors)
    assert [r['a'] for r in rows[:3]] == [0.0, 0.5, 1.0]                      # the first factor fastest
    assert {r['c'] for r in rows} == {'x', 'y'} and {r['b'] for r in rows} == {False, True}
    assert {r['n'] for r in rows} == {1, 4} and all(isinstance(r['n'], int) for r in rows)
    assert len({json.dumps(r, sort_keys=True) for r in rows}) == 24           # every combination once


def test_fractional_designs_resolution_and_refusals():
    for k, res, n in ((3, 'III', 4), (7, 'III', 8), (4, 'IV', 8), (8, 'IV', 16), (5, 'IV', 16)):
        S, gens = EX.fractional_signs(k, res)
        assert S.shape == (n, k) and len(gens) == k - int(math.log2(n))
        assert EX.resolution_of(S) >= (3 if res == 'III' else 4)
        assert np.array_equal(S.T @ S, n * np.eye(k))                         # balanced and orthogonal
    S, gens = EX.fractional_signs(7, 'III')
    assert gens == ['D=AB', 'E=AC', 'F=BC', 'G=ABC'] and EX.resolution_of(S) == 3
    assert EX.resolution_of(EX.fractional_signs(3, 'IV')[0]) is None          # 8 runs for 3 factors: the full design
    d = EX.Design.from_dict({'factors': [{'path': 'a', 'lo': 0, 'hi': 1}, {'path': 'c', 'levels': [1, 2, 3]}],
                             'design': {'method': 'fractional'}})
    with pytest.raises(EX.ExperimentError, match='two levels'):
        EX.make_plan(d)


def test_main_effects_of_known_models():
    # full factorial, linear: the effect from the first level to the last is the coefficient times the range
    d = {'factors': [{'path': 'a', 'lo': 0.0, 'hi': 2.0}, {'path': 'b', 'lo': -1.0, 'hi': 1.0},
                     {'path': 'd', 'levels': ['A', 'B']}], 'design': {'method': 'full_factorial', 'levels': 3}}
    r = EX.run_function(lambda v: 1 + 3 * v['a'] - 1.5 * v['b'] + 4 * (v['d'] == 'B'), d)
    me = {m['factor']: m for m in r['analysis']['y']['main_effects']}
    assert me['a']['effect'] == pytest.approx(6.0) and me['b']['effect'] == pytest.approx(-3.0)
    assert me['d']['effect'] == pytest.approx(4.0) and [x['level'] for x in me['d']['levels']] == ['A', 'B']
    lo, hi = me['a']['effect_ci']
    assert lo < 6.0 < hi and [x['n'] for x in me['a']['levels']] == [6, 6, 6]
    # resolution IV keeps the main effects free of two-factor interactions; III does not
    coef = np.array([1.0, -2.0, 0.5, 3.0])

    def f(v):
        x = np.array([v[f'x{i}'] for i in range(4)])
        return float(coef @ x + 2.0 * x[0] * x[1])
    fac = [{'path': f'x{i}', 'lo': -1, 'hi': 1} for i in range(4)]
    eff = lambda res: [m['effect'] for m in EX.run_function(                  # noqa: E731
        f, {'factors': fac, 'design': {'method': 'fractional', 'resolution': res}})['analysis']['y']['main_effects']]
    assert np.allclose(eff('IV'), 2 * coef)
    e3 = eff('III')                                         # 4 factors in 8 runs: D = AB, aliased with x0·x1
    assert not np.allclose(e3, 2 * coef) and e3[3] == pytest.approx(2 * coef[3] + 2 * 2.0)


def test_binned_main_effects_and_welch():
    d = {'factors': [{'path': 'a', 'lo': 0, 'hi': 1}], 'design': {'method': 'lhs', 'n': 600, 'seed': 2}}
    me = EX.run_function(lambda v: 10 * v['a'], d)['analysis']['y']['main_effects'][0]
    assert me['binned'] and len(me['levels']) == 6
    assert me['effect'] == pytest.approx(10 * (11 / 12 - 1 / 12), abs=0.15)    # bin centres 1/12 and 11/12
    diff, ci = EX.welch([1, 2, 3], [4, 5, 6])
    from scipy.stats import ttest_ind
    t = ttest_ind([4, 5, 6], [1, 2, 3], equal_var=False)
    lo, hi = t.confidence_interval(0.95)
    assert diff == 3.0 and ci == pytest.approx([lo, hi])


# ============================================================================ design.json
def test_design_json_round_trip_and_plain_errors():
    d = EX.Design.from_dict({'schema': EX.SCHEMA, 'factors': [{'path': 'sensor:Wrist.pitch', 'lo': 10, 'hi': -10},
                                                              {'path': 'sensor:Wrist.enabled'},
                                                              {'path': 'sensor:Wrist.model',
                                                               'levels': ['a', 'b', 'a']}],
                             'responses': ['_exp.score', {'metric': '_exp.x'}],
                             'design': {'method': 'saltelli', 'n': 10 ** 9, 'seed': -4}, 'execution': {'workers': 3}})
    f0, f1, f2 = d.factors
    assert (f0.lo, f0.hi) == (-10, 10) and f1.kind == 'bool' and f2.levels == ['a', 'b']
    assert d.responses == ['_exp.score', '_exp.x'] and d.n == EX.LIMITS['n'][1] and d.seed == 0 and d.workers == 3
    back = EX.Design.from_dict(json.loads(json.dumps(d.to_dict())))
    assert back.to_dict() == d.to_dict() and back.to_dict()['schema'] == 'dexvantage.experiment/1'
    for bad, words in (({'schema': 'other/1'}, 'schema'), ({'factors': [{'path': 'a'}]}, 'needs a range'),
                       ({'factors': [{'lo': 1}]}, 'no "path"'),
                       ({'factors': [{'path': 'a', 'lo': 0, 'hi': 1}] * 2}, 'twice'),
                       ({'design': {'method': 'taguchi'}}, 'Unknown design method'),
                       ({'design': {'resolution': 'V'}}, 'III or IV'),
                       ({'factors': [{'path': 'a', 'kind': 'categorical'}]}, 'no "levels"')):
        with pytest.raises(EX.ExperimentError, match=words):
            EX.Design.from_dict(bad)
    with pytest.raises(EX.ExperimentError, match='not valid JSON'):
        EX.load_design('{nope')


# ============================================================================ on a project, through core.metrics
def _project_design(**kw):
    d = {'factors': [{'path': 'sensor:Wrist.x', 'lo': 0.0, 'hi': 0.2}, {'path': 'sensor:Wrist.pitch', 'lo': -10,
                                                                        'hi': 10, 'n_levels': 3},
                     {'path': 'sensor:Wrist.enabled'}, {'path': 'setting:_exp_toy.gain', 'levels': [0.0, 2.0]}],
         'responses': ['_exp.score', '_exp.x', '_exp.r'], 'design': {'method': 'full_factorial', 'levels': 2}}
    d.update(kw)
    return d


def test_runs_match_direct_evaluation_and_the_best_run():
    p = _proj()
    doc = EX.run(p, _project_design())
    assert doc['execution']['mode'] == 'serial (the default)' and len(doc['runs']) == 2 * 3 * 2 * 2
    for r in doc['runs']:
        v = r['values']
        want = _toy_value(v['sensor:Wrist.x'], v['sensor:Wrist.pitch'], v['sensor:Wrist.enabled'],
                          v['setting:_exp_toy.gain'])
        assert r['responses']['_exp.score'] == pytest.approx(want, abs=1e-9)
        if v['sensor:Wrist.enabled']:
            assert r['status']['_exp.r'] == 'ok'
        else:                                                # a refusal is a status and a sentence, not a crash
            assert r['status']['_exp.r'] == 'refused' and 'disabled' in r['reason']['_exp.r']
    best = doc['best']
    assert best['run'] and best['values'] == {'sensor:Wrist.x': 0.2, 'sensor:Wrist.pitch': 0.0,
                                              'sensor:Wrist.enabled': True, 'setting:_exp_toy.gain': 2.0}
    assert best['value'] == pytest.approx(_toy_value(0.2, 0.0, True, 2.0))
    assert best['baseline'] == pytest.approx(_toy_value(0.0, 0.0, True, 0.0)) and best['improves']
    assert doc['headline']['status'] == 'good' and doc['headline']['word'] == 'better than now'
    me = {m['factor']: m for m in doc['analysis']['_exp.score']['main_effects']}
    assert me['sensor:Wrist.x']['effect'] == pytest.approx(20.0)
    assert me['sensor:Wrist.enabled']['effect'] == pytest.approx(5.0)
    assert any('_exp.r: 12 of 24 runs have no value' in n for n in doc['notes'])
    # the project itself is never changed
    assert F.get(p, 'sensor:Wrist.x') == 0.0 and '_exp_toy' not in p.analysis.studies
    # the objective: the lower x
    doc2 = EX.run(p, _project_design(objective='_exp.x'))
    assert doc2['best']['value'] == 0.0 and doc2['headline']['word'] == 'same as now'
    json.dumps(doc, allow_nan=False)                         # JSON-able


def test_check_design_locked_unknown_and_cache():
    p = _proj()
    d = EX.Design.from_dict(_project_design(), p)
    assert EX.check_design(p, d) == []
    bad = EX.Design.from_dict(_project_design(factors=[{'path': 'sensor:Top.z', 'lo': 0, 'hi': 1},
                                                       {'path': 'sensor:Nobody.x', 'lo': 0, 'hi': 1}],
                                              responses=['_exp.score', 'nope.metric']), p)
    probs = EX.check_design(p, bad)
    assert any('locked' in s for s in probs) and any('Nobody' in s for s in probs)
    assert any('no metric called nope.metric' in s for s in probs)
    with pytest.raises(EX.ExperimentError, match='locked'):
        EX.run(p, bad)
    # the cache: a second run computes no study again
    cache = jobs.ResultCache(256)
    EX.run(p, _project_design(), cache=cache)
    h0 = cache.hits
    doc = EX.run(p, _project_design(), cache=cache)
    assert doc['execution']['cache_hits'] >= 24 and cache.hits - h0 >= 24


def test_cancel_keeps_the_runs_done_and_progress():
    p = _proj()
    seen = []

    def progress(f, text=''):
        seen.append((f, text))
    ctx = jobs.Ctx(progress=progress, cancel=lambda: sum(1 for f, _t in seen if f > 0) >= 5)
    doc = EX.run(p, _project_design(), ctx)
    assert doc['cancelled_at'] == 5 and doc['execution']['runs_done'] == 5
    assert sum(r['done'] for r in doc['runs']) == 5 and doc['runs'][6]['status']['_exp.score'] == 'not run'
    assert 'of 24 planned' in doc['headline']['text'] and seen[-1][1] == 'run 5 of 24'


def test_a_run_cancelled_while_it_ran_is_not_kept():
    CALLS.clear()
    ctx = jobs.Ctx(cancel=lambda: len(CALLS) >= 3)          # the baseline is call 1, the first run call 2
    doc = EX.run(_proj(), _project_design(), ctx)
    assert doc['execution']['runs_done'] == 1 and doc['cancelled_at'] == 1
    assert doc['runs'][1]['status']['_exp.score'] == 'not run' and doc['runs'][1]['responses']['_exp.score'] is None


def test_the_study_keeps_runs_between_experiments():
    p = _proj()
    p.analysis.studies['experiments'] = _project_design()
    jobs.run_study('experiments', p)
    CALLS.clear()
    doc = jobs.run_study('experiments', p)
    assert doc['execution']['runs_from_cache'] == 24 and CALLS == []     # the baseline is not counted
    p.analysis.studies['experiments']['design']['levels'] = 3             # an edited design keeps the shared runs
    doc = jobs.run_study('experiments', p)
    assert len(doc['runs']) == 36 and doc['execution']['runs_from_cache'] == 24 and len(CALLS) == 12
    p.analysis.studies['_exp_toy'] = {'other': 1}                          # a study setting changes: evaluated again
    CALLS.clear()
    doc = jobs.run_study('experiments', p)
    assert doc['execution']['runs_from_cache'] == 0 and len(CALLS) == 37


def test_process_pool_gives_the_serial_results():
    p = _proj()
    d = _project_design(design={'method': 'lhs', 'n': 6, 'seed': 3}, execution={'workers': 2})
    par = EX.run(p, d)
    ser = EX.run(p, dict(d, execution={'workers': 0}))
    assert par['execution']['mode'] == 'process pool of 2' and par['execution']['runs_done'] == 6
    assert [r['responses'] for r in par['runs']] == [r['responses'] for r in ser['runs']]
    assert [r['status'] for r in par['runs']] == [r['status'] for r in ser['runs']]


def test_study_batch_csv_and_apply(tmp_path):
    p = _proj()
    with pytest.raises(jobs.StudyError, match='no factor yet'):
        jobs.run_study('experiments', p)
    p.analysis.studies['experiments'] = _project_design()
    doc = jobs.run_study('experiments', p)
    assert doc['schema'] == 'dexvantage.experiments/1' and doc['best']['improves']
    assert jobs.STUDIES['experiments'].batch == 'experiment-saved' and not jobs.STUDIES['experiments'].quick
    # --experiment DESIGN.json, in process (the toy metrics are registered in this process only)
    design = tmp_path / 'design.json'
    design.write_text(json.dumps(dict(_project_design(), schema=EX.SCHEMA)))
    proj = tmp_path / 'cell.dexv'
    p.save(str(proj))
    from dexvantage.app import main
    assert main([str(proj), '--experiment', str(design)]) == 0
    out = json.loads((tmp_path / 'design.results.json').read_text())
    assert out['provenance']['interface'] == 'batch' and out['provenance']['output']['option'] == '--experiment'
    text = (tmp_path / 'design.results.csv').read_text()
    import hashlib
    assert out['files']['csv_sha256'] == hashlib.sha256(text.encode()).hexdigest()
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0][:3] == ['run', 'block', 'sensor:Wrist.x'] and len(rows) == 25 and '_exp.score (pt)' in rows[0]
    assert str(tmp_path) not in (tmp_path / 'design.results.json').read_text()
    assert main([str(proj), '--experiment', str(design), '--experiment-out', str(design)]) == 2   # never over it
    # Apply: the best run's factors set in place, in the sections they belong to
    q = p.snapshot()
    keys = EX.apply_keys(q, doc['best']['values'])
    assert keys == ['variant', 'analysis']
    EX.apply_values(q, doc['best']['values'])
    assert F.get(q, 'sensor:Wrist.x') == pytest.approx(0.2) and q.analysis.studies['_exp_toy']['gain'] == 2.0


def test_batch_errors_are_sentences(tmp_path):
    proj = tmp_path / 'cell.dexv'
    _proj().save(str(proj))
    design = tmp_path / 'd.json'
    design.write_text(json.dumps({'schema': EX.SCHEMA, 'factors': [{'path': 'sensor:Wrist.x', 'lo': 0, 'hi': 1}],
                                  'responses': ['no.such.metric']}))
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    r = subprocess.run([sys.executable, '-m', 'dexvantage', str(proj), '--experiment', str(design)],
                       capture_output=True, text=True, env=env, cwd=ROOT, timeout=120)
    assert r.returncode == 2 and 'no metric called no.such.metric' in r.stderr and 'Traceback' not in r.stderr


def test_candidates_from_the_rig():
    from dexvantage.core.robot.samples import sample_project
    p = sample_project('ur5e')
    p.analysis.studies['grasp_visibility'] = {'n_grasps': 200, 'flag': True, 'name': 'x'}
    cs = EX.candidates(p)
    d405 = next(s for s in p.variant.sensors if s.name == 'Wrist D405')
    pitch = next(c for c in cs if c['path'] == f'sensor:{d405.uid}.pitch')
    now = F.get(p, pitch['path'])
    assert pitch['unit'] == '°' and pitch['factor']['lo'] == pytest.approx(now - 5) and pitch['group'] == d405.name
    model = next(c for c in cs if c['path'] == f'sensor:{d405.uid}.model')
    assert model['factor']['levels'][0] == 'intel-realsense-d405' and len(model['factor']['levels']) >= 2
    n = next(c for c in cs if c['path'] == 'setting:grasp_visibility.n_grasps')
    assert n['factor'] == {'path': 'setting:grasp_visibility.n_grasps', 'kind': 'continuous', 'lo': 150, 'hi': 250,
                           'integer': True}
    assert any(c['path'] == 'setting:grasp_visibility.flag' and c['kind'] == 'bool' for c in cs)
    assert not any(c['path'].endswith('.name') for c in cs)
    for c in cs:                                          # every template is a valid factor of the project
        EX.Factor.from_dict(c['factor'], p)
        F.get(p, c['path'])


# ============================================================================ the page
def test_page_edits_runs_charts_applies_and_exports(app, win, tmp_path):
    from conftest import pump
    p = _proj()
    win.store.set_project(p)
    win.ctx.analysis.wait(30)
    assert 'experiments' in win.pages
    win.open_page('experiments')
    page = win.pages['experiments']
    pump(app, 0.05)
    assert not page.b_run.isEnabled() and 'no factor' in page.problems.text()
    wrist = next(s for s in win.store.project.variant.sensors if s.name == 'Wrist')
    n0 = win.store.undo.count()
    assert page.add_factor(f'sensor:{wrist.uid}.x')
    assert win.store.undo.count() == n0 + 1
    top = next(s for s in win.store.project.variant.sensors if s.name == 'Top')
    assert page.add_factor(f'sensor:{top.uid}.x') is False            # locked: refused, said
    assert page.set_factor(f'sensor:{wrist.uid}.x', lo=0.0, hi=0.1)
    assert page.add_factor(f'sensor:{wrist.uid}.pitch')
    assert page.set_responses(['_exp.score'])
    assert page._set_design(method='saltelli', text='design type') and page._set_design(n=64, text='n')
    pump(app, 0.05)
    assert page.b_run.isEnabled() and page.count_label.text() == '256 runs = 64 × (2 + 2)'
    page.run()
    win.ctx.analysis.wait(120)
    pump(app, 0.1)
    assert page.doc is not None and page.head.figure.text() != '—'
    assert page.doc['method'] == 'saltelli' and page.table.topLevelItemCount() == 256
    assert page.sobol.rows and page.effects.rows and page.b_apply.isEnabled()
    s = page.doc['analysis']['_exp.score']['sobol']
    assert s['ST'][1] > s['ST'][0]          # the pitch's −½(p − 3)² over ±5° outweighs 100·x over 0.1 m
    assert page.overlays(win.viewport)
    page.preview()
    assert page._preview == page.doc['best']['run']
    best = page.doc['best']['values']
    n = win.store.undo.count()
    assert page.apply()
    assert win.store.undo.count() == n + 1
    assert F.get(win.store.project, f'sensor:{wrist.uid}.x') == pytest.approx(best[f'sensor:{wrist.uid}.x'])
    win.store.undo.undo()
    assert F.get(win.store.project, f'sensor:{wrist.uid}.x') == 0.0
    out = page.export(str(tmp_path / 'exp.json'))
    assert out and (tmp_path / 'exp.csv').exists()
    assert 'provenance' in json.loads((tmp_path / 'exp.json').read_text())
    assert 'Jansen' in page.explain_text()
