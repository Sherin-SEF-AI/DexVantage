"""Statistics and power analysis (core/stats.py, core/statcompare.py, core/studies/stats.py): the maths against
SciPy and published examples, the comparison sources (success counts, variants over the tolerance Monte Carlo with a
cheap evaluator, experiment runs, tracker runs, samples), the study, batch --stats, dv.stats, the assistant tool,
the metrics, the validation checks and the self-test line."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from dexvantage.core import statcompare as SC
from dexvantage.core import stats as S
from dexvantage.core.studies import stats as SS

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
BH = (0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344, 0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590,
      1.000)


# ====================================================================== the maths
def test_intervals_agree_with_scipy():
    from scipy import stats as ss
    for k, n in ((0, 10), (3, 10), (10, 10), (17, 50)):
        for meth, fn in (('wilson', S.wilson), ('exact', S.clopper_pearson)):
            r = ss.binomtest(k, n).proportion_ci(0.95, meth)
            assert fn(k, n) == pytest.approx((r.low, r.high), abs=1e-12)


def test_bad_inputs_are_plain_sentences():
    with pytest.raises(S.StatsError, match='between 0 and'):
        S.wilson(12, 10)
    with pytest.raises(S.StatsError, match='alpha'):
        S.adjust([0.1], 'holm', alpha=0.7)
    with pytest.raises(S.StatsError, match='equal'):
        S.n_two_proportions(0.5, 0.5)
    with pytest.raises(S.StatsError, match='at least 2'):
        S.compare_samples([1.0], [2.0, 3.0])
    with pytest.raises(S.StatsError, match='not a finite number'):
        S.welch_t([1, 2, float('nan')], [1, 2, 3])
    with pytest.raises(S.StatsError, match='one value each per pair'):
        S.paired_t([1, 2, 3], [1, 2])


def test_benjamini_hochberg_1995_example():
    """Benjamini and Hochberg (1995), section 3.2, p. 295: Bonferroni rejects 3, the FDR procedure 4."""
    assert sum(S.adjust(BH, 'bonferroni')['reject']) == 3
    assert sum(S.adjust(BH, 'bh')['reject']) == 4
    assert sum(S.adjust(BH, 'holm')['reject']) == 3
    assert sum(S.adjust(BH, 'by')['reject']) == 3
    r = S.adjust([0.04, None, 0.01], 'holm')
    assert r['adjusted'][1] is None and r['m'] == 2 and r['adjusted'][2] == pytest.approx(0.02)


def test_tests_agree_with_scipy():
    from scipy import stats as ss
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=9), rng.normal(0.7, size=11)
    assert S.welch_t(x, y)['p'] == pytest.approx(ss.ttest_ind(y, x, equal_var=False).pvalue, abs=1e-12)
    assert S.mann_whitney(x, y)['p'] == pytest.approx(ss.mannwhitneyu(y, x).pvalue, abs=1e-12)
    d = rng.normal(0.4, 1, 12)
    assert S.wilcoxon_signed_rank(x[:1].repeat(12), x[:1].repeat(12) + d)['p'] == pytest.approx(
        ss.wilcoxon(x[:1].repeat(12) + d - x[:1].repeat(12)).pvalue, abs=1e-12)
    f = ss.fisher_exact([[14, 6], [19, 1]]).pvalue
    assert S.fisher_exact(14, 20, 19, 20)['p'] == pytest.approx(f, abs=1e-12)
    perm = S.permutation_test(x[:5], y[:6])
    assert perm['exact'] and perm['rearrangements'] == 462
    ref = ss.permutation_test((x[:5], y[:6]), lambda a, b, axis: b.mean(axis) - a.mean(axis), vectorized=True)
    assert perm['p'] == pytest.approx(ref.pvalue, abs=1e-12)


def test_bootstrap_is_seeded_and_brackets_the_estimate():
    rng = np.random.default_rng(1)
    x = rng.exponential(size=30)
    a = S.bootstrap_ci(x, seed=4)
    b = S.bootstrap_ci(x, seed=4)
    assert a == b and a['ci'][0] < a['value'] < a['ci'][1] and a['method'] == 'bca'
    c = S.bootstrap_ci([1.0] * 10)
    assert c['ci'] == [1.0, 1.0] and 'point' in c['note']


def test_effect_sizes():
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=20), rng.normal(0.8, size=20)
    e = S.cohens_d(x, y)
    assert e['ci'][0] < e['d'] < e['ci'][1] and abs(e['g']) < abs(e['d'])
    gt = sum(1 for a in x for b in y if b > a)
    lt = sum(1 for a in x for b in y if b < a)
    assert S.cliffs_delta(x, y)['value'] == pytest.approx((gt - lt) / 400)
    assert S.cohens_d([1, 1, 1], [1, 1, 1])['d'] is None
    rd = S.risk_difference(14, 20, 19, 20)
    assert rd['value'] == pytest.approx(0.25) and rd['ci'][0] < 0.25 < rd['ci'][1]
    o = S.odds_ratio(0, 10, 5, 10)
    assert o['corrected'] and 'zero' in o['method']


def test_power_against_gpower_and_cohen():
    """G*Power 3.1 manual p. 48 and 50; Cohen (1992) Table 2 (within one per group)."""
    assert S.power_t(0.421637, 50, paired=True) == pytest.approx(0.832114, abs=5e-6)
    assert S.power_t(0.2828427, 50, paired=True) == pytest.approx(0.500352, abs=5e-6)
    r = S.n_t(0.625, 0.05, 0.95, paired=True, alternative='greater')
    assert r['n_per_group'] == 30 and r['power_at_n'] == pytest.approx(0.955144, abs=5e-6)
    assert S.n_t(0.1, 0.01, 0.9, paired=True)['n_per_group'] == 1492
    assert abs(S.n_t(0.5)['n_per_group'] - 64) <= 1 and abs(S.n_t(0.2)['n_per_group'] - 393) <= 1


def test_episodes_needed_exact_and_normal():
    e = S.episodes_needed(0.5, 0.8)
    assert e['method'] == 'exact' and e['test'] == 'fisher' and e['power_at_n'] >= 0.8
    assert S.power_two_proportions(0.5, 0.8, e['n_per_group'], alpha=0.05, method='exact') >= 0.8
    assert e['curve']['n'] and len(e['curve']['n']) == len(e['curve']['power'])
    assert 'independent trial' in ' '.join(e['assumptions']) and 'Sequential' in e['out_of_scope']
    nn = S.n_two_proportions(0.5, 0.8, method='normal')
    assert nn['n_per_group'] >= S.n_two_proportions(0.5, 0.8, method='normal', continuity=False)['n_per_group']
    big = S.episodes_needed(0.50, 0.52)            # far above the exact limit: the normal approximation, said
    assert big['method'] == 'normal' and 'normal approximation' in big['note']


def test_demonstrations_needed_inflates_by_design_effect_and_discards():
    base = S.n_t(0.5)['n_per_group']
    r = S.demonstrations_needed('continuous', d=0.5, icc=0.1, per_operator=11, attrition=0.2)
    assert r['design_effect'] == pytest.approx(2.0)
    assert r['n_per_group'] == int(np.ceil(base * 2.0 / 0.8))
    assert 'not modelled' in r['not_modelled'] and 'Kish' in ' '.join(r['assumptions'])
    b = S.demonstrations_needed('binary', p1=0.6, p2=0.85)
    assert b['n_per_group'] == S.n_two_proportions(0.6, 0.85)['n_per_group']
    with pytest.raises(S.StatsError, match='discarded'):
        S.demonstrations_needed('continuous', d=0.5, attrition=1.0)


def test_compare_samples_contract_used_by_the_tracker():
    """core/tracking calls compare_samples(a, b, better) and reads 'text' and 'p'."""
    rng = np.random.default_rng(5)
    a, b = list(rng.normal(10, 1, 20)), list(rng.normal(11, 1, 20))
    r = S.compare_samples(a, b, 'higher')
    assert r['paired'] is False and 0 <= r['p'] <= 1 and r['text'] and r['test'] == 'permutation'
    from dexvantage.core import tracking as TR
    sig = TR.significance(a, b, 'higher')
    assert sig['available'] and sig['p'] == r['p']


def test_paired_comparison_says_why_and_what_it_detects():
    rng = np.random.default_rng(6)
    x = rng.normal(90, 2, 12)
    y = x + rng.normal(0.05, 0.5, 12)
    r = S.compare_samples(x, y, paired=True, pairing='paired by seed')
    assert 'sign' in r['why'] and r['exact'] and r['detectable']['difference'] > 0
    assert any('paired by seed' in a for a in r['assumptions'])
    v = S.verdict(r, 0.5, 'Coverage', '%')
    assert v['word'] == 'no clear difference' and 'does not show they are equal' in v['text']


# ====================================================================== sources
def test_rates_source_and_correction():
    res = SC.analyse_rates([{'condition': 'A', 'successes': 10, 'trials': 20},
                            {'condition': 'B', 'successes': 19, 'trials': 20},
                            {'condition': 'C', 'successes': 11, 'trials': 20}], 'pairs')
    assert [c['id'] for c in res['comparisons']] == ['A vs B', 'A vs C', 'B vs C']
    assert res['correction']['method'] == 'holm' and res['correction']['m'] == 3
    for c in res['comparisons']:
        assert c['p_adjusted'] >= c['p'] and c['verdict']['text']
    with pytest.raises(S.StatsError, match='called "A"'):
        SC.clean_rates([{'condition': 'A', 'successes': 1, 'trials': 2}, {'condition': 'A', 'successes': 1,
                                                                           'trials': 2}])


def _two_variant_project():
    from dexvantage.core.project import Variant
    from dexvantage.core.robot.samples import sample_project
    p = sample_project('so100')
    d = p.variant.to_dict()
    d['uid'], d['name'] = '', 'Rig B'
    for s in d['sensors']:
        s['uid'] = ''
    p.variants.append(Variant.from_dict(d))
    return p


@pytest.fixture
def test_metric():
    from dexvantage.core import metrics as M
    M.load_metrics()

    @M.metric('_test.height', 'Second sensor height', 'm', 'higher', study='summary')
    def _h(doc):
        return None
    yield '_test.height'
    M.METRICS.pop('_test.height', None)


def test_variants_paired_by_seed_with_a_cheap_evaluator(test_metric):
    p = _two_variant_project()
    vb = p.variants[1]
    vb.sensors[1].mount.pose.position[2] += 0.05          # B is 5 cm higher

    def ev(project, ids, ctx):
        z = float(project.variant.sensors[1].mount.pose.position[2])
        return {m: {'value': z, 'unit': 'm', 'better': 'higher', 'title': 'Second sensor height',
                    'status': 'ok', 'study': 'summary'} for m in ids}
    data = SC.variant_samples(p, None, 'Rig B', [test_metric], n=12, seed=3, evaluator=ev)
    assert data['common_random_numbers'] and 'paired by seed' in data['pairing']
    row = data['rows'][0]
    d = np.array(row['b']) - np.array(row['a'])
    assert np.allclose(d, 0.05, atol=1e-9)                # the same draws: only the 5 cm differs
    res = SC.analyse_samples(data['rows'], True, names=(data['a'], data['b']), pairing=data['pairing'])
    c = res['comparisons'][0]
    assert c['verdict']['word'] == 'clear difference' and c['difference']['value'] == pytest.approx(0.05)


def test_variants_need_two_variants():
    from dexvantage.core.robot.samples import sample_project
    with pytest.raises(S.StatsError, match='one variant'):
        SC.pick_variants(sample_project('so100'))


def _factorial_doc():
    runs, i = [], 0
    for z in (0.3, 0.4, 0.5):
        for cam in ('d405', 'd435'):
            i += 1
            y = 90 + 10 * z + (2.0 if cam == 'd435' else 0.0)
            runs.append({'run': i, 'block': '', 'values': {'sensor:Cam.z': z, 'sensor:Cam.model': cam}, 'done': True,
                         'responses': {'coverage.covered': y}, 'status': {'coverage.covered': 'ok'}})
    return {'method': 'full_factorial', 'method_label': 'Full factorial',
            'factors': [{'path': 'sensor:Cam.z', 'title': 'Cam z', 'kind': 'continuous', 'unit': 'm'},
                        {'path': 'sensor:Cam.model', 'title': 'Cam model', 'kind': 'categorical',
                         'levels': ['d405', 'd435']}],
            'responses': [{'id': 'coverage.covered', 'title': 'Task volume seen', 'unit': '%', 'better': 'higher'}],
            'runs': runs}


def test_experiment_factor_split_pairs_by_the_other_factors():
    data = SC.experiment_samples(_factorial_doc(), 'sensor:Cam.model')
    assert data['paired'] and 'other factors' in data['pairing'] and data['runs'] == [3, 3]
    row = data['rows'][0]
    assert np.allclose(np.array(row['b']) - np.array(row['a']), 2.0)
    cont = SC.experiment_samples(_factorial_doc(), 'sensor:Cam.z', levels=[0.3, 0.5])
    assert cont['paired'] and cont['a'].endswith('0.3')
    with pytest.raises(S.StatsError, match='no factor'):
        SC.experiment_samples(_factorial_doc(), 'nope')


class _Store:
    name = 'test.runs'

    def __init__(self, recs):
        self.recs = {r['run_id']: r for r in recs}

    def runs(self, study=None, tag=None, status=None, **_):
        return [r for r in self.recs.values() if tag is None or tag in r.get('tags', [])]

    def get(self, rid):
        return self.recs[rid]


def _rec(rid, seed, value, tags=(), samples=None):
    r = {'run_id': rid, 'name': rid, 'study': 'tolerance', 'seeds': {'tolerance': seed}, 'tags': list(tags),
         'metrics': {'coverage.covered': {'value': value, 'title': 'Task volume seen', 'unit': '%',
                                          'better': 'higher'}}}
    if samples is not None:
        r['samples'] = {'coverage.covered': samples}
    return r


def test_tracker_runs_paired_by_seed_and_by_draws():
    st = _Store([_rec(f'a{i}', i, 90 + i * 0.1, ('base',)) for i in range(5)] +
                [_rec(f'b{i}', 4 - i, 91 + (4 - i) * 0.1, ('new',)) for i in range(5)])
    data = SC.tracker_samples(st, {'tag': 'base'}, {'tag': 'new'})
    assert data['paired'] and np.allclose(np.array(data['rows'][0]['b']) - np.array(data['rows'][0]['a']), 1.0)
    rng = np.random.default_rng(0)
    s1 = list(rng.normal(90, 1, 16))
    st2 = _Store([_rec('x', 7, 90, samples=s1), _rec('y', 7, 91, samples=[v + 0.5 for v in s1])])
    d2 = SC.tracker_samples(st2, 'x', 'y')
    assert d2['paired'] and 'seed 7' in d2['pairing']


# ====================================================================== the study, batch, API, tool, metrics
def test_study_on_rates_with_power_and_suggestion():
    doc, snap = SS.run_spec({'source': 'rates', 'rates': [{'condition': 'ACT', 'successes': 14, 'trials': 20},
                                                          {'condition': 'Diffusion', 'successes': 19, 'trials': 20}],
                             'power': {'kind': 'rates', 'p1': 0.7, 'p2': 0.9, 'n': 40}})
    assert doc['study'] == 'stats' and doc['headline']['word'] == 'no clear difference'
    assert doc['headline']['unit'] == 'pp' and 'Fisher' in doc['headline']['text']
    sg = doc['suggestion']
    assert sg['kind'] == 'power' and sg['settings']['power']['n'] > 20
    assert doc['power']['planned']['power'] < 0.8 and doc['power']['n_per_group'] > 40
    json.dumps(doc, allow_nan=False)
    assert 'Sequential' in doc['out_of_scope'] and doc['references']


def test_study_refuses_in_a_sentence():
    with pytest.raises(S.StatsError, match='at least two conditions'):
        SS.run_spec({'source': 'rates'})
    with pytest.raises(S.StatsError, match='Give the samples'):
        SS.analyse(None, SS.settings_of({'source': 'samples', 'samples': {}}))
    with pytest.raises(S.StatsError, match='schema'):
        SS.run_spec({'schema': 'other/1', 'source': 'rates'})
    with pytest.raises(S.StatsError, match='correction'):
        SS.run_spec({'correction': 'magic', 'power': {'kind': 't2', 'd': 0.5}})
    with pytest.raises(S.StatsError, match='needs both rates'):
        SS.run_spec({'power': {'kind': 'rates'}})


def test_power_only_headline():
    doc, _ = SS.run_spec({'power': {'kind': 'paired_t', 'd': 0.5, 'n': 40}})
    assert doc['comparisons'] == [] and doc['headline']['word'] == 'enough planned'
    doc, _ = SS.run_spec({'power': {'kind': 'demonstrations', 'd': 0.5, 'icc': 0.05, 'per_operator': 20}})
    assert doc['headline']['word'] == 'needed' and doc['power']['design_effect'] == pytest.approx(1.95)


def test_metrics_read_the_study():
    from dexvantage.core import metrics as M
    doc, _ = SS.run_spec({'source': 'rates', 'rates': [{'condition': 'A', 'successes': 2, 'trials': 20},
                                                       {'condition': 'B', 'successes': 18, 'trials': 20}],
                          'power': {'kind': 'rates', 'p1': 0.1, 'p2': 0.9, 'n': 10}})
    M.load_metrics()
    assert M.read(M.METRICS['stats.p_adjusted'], doc)['value'] == pytest.approx(doc['headline']['p_adjusted'])
    assert M.read(M.METRICS['stats.n_needed'], doc)['value'] == doc['power']['n_per_group']
    assert M.read(M.METRICS['stats.planned_power'], doc)['value'] == pytest.approx(
        100 * doc['power']['planned']['power'])
    d2, _ = SS.run_spec({'power': {'kind': 't2', 'd': 0.5}})
    r = M.read(M.METRICS['stats.p_adjusted'], d2)
    assert r['value'] is None and 'no comparison' in r['reason']


def test_batch_stats_spec(tmp_path):
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({'schema': 'dexvantage.stats/1', 'source': 'samples',
                                'samples': {'a': [1, 2, 3, 4, 5, 6], 'b': [2, 3, 4, 5, 6, 8], 'paired': True,
                                            'title': 'Time', 'unit': 's', 'better': 'lower'}}))
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    r = subprocess.run([sys.executable, '-m', 'dexvantage', '--stats', str(spec)], capture_output=True, text=True,
                       env=env, cwd=ROOT, timeout=120)
    assert r.returncode == 0, r.stderr
    out = json.loads((tmp_path / 'spec.results.json').read_text())
    assert out['study'] == 'stats' and out['provenance']['inputs'][0]['name'] == 'spec.json'
    assert out['comparisons'][0]['exact'] and 'Time' in r.stdout
    bad = tmp_path / 'bad.json'
    bad.write_text('{"source": "rates", "rates": [{"condition": "A", "successes": 5, "trials": 3}]}')
    r = subprocess.run([sys.executable, '-m', 'dexvantage', '--stats', str(bad)], capture_output=True, text=True,
                       env=env, cwd=ROOT, timeout=120)
    assert r.returncode == 2 and 'Traceback' not in r.stderr and 'two conditions' in r.stderr


def test_api_dv_stats_without_qt():
    code = ('import sys, dexvantage as dv\n'
            'r = dv.stats.compare_rates([("ACT", 14, 20), ("Diffusion", 19, 20)])\n'
            'assert r.headline["word"] == "no clear difference"\n'
            'assert dv.stats.episodes_needed(0.5, 0.8)["n_per_group"] == 44\n'
            's = dv.stats.compare_samples([1, 2, 3, 4], [2, 3, 4, 6], paired=True)\n'
            'assert s["comparisons"][0]["paired"]\n'
            'assert "wilson" in dir(dv.stats)\n'
            'assert not any(m.startswith("PyQt6") for m in sys.modules)\n'
            'print("ok")\n')
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert r.returncode == 0 and 'ok' in r.stdout, r.stderr


def test_assistant_tool_get_statistics():
    from dexvantage.assistant import tools as T
    from dexvantage.assistant.session import FileSession
    from dexvantage.core.project import blank_project
    s = FileSession(project=blank_project('Stats'))
    out = T.call(s, 'get_statistics', {'rates': [{'condition': 'A', 'successes': 5, 'trials': 20},
                                                 {'condition': 'B', 'successes': 16, 'trials': 20}],
                                       'power': {'kind': 'rates', 'p1': 0.25, 'p2': 0.8}})
    assert 'error' not in out, out
    assert out['comparisons'][0]['test'] == 'Fisher\'s exact test' and out['headline']['word'] == 'clear difference'
    assert out['power']['n_per_group'] > 0 and out['assumptions']
    bad = T.call(s, 'get_statistics', {'rates': [{'condition': 'A', 'successes': 5, 'trials': 2},
                                                 {'condition': 'B', 'successes': 1, 'trials': 2}]})
    assert 'between 0 and' in bad['error']
    assert not T.validate_args('get_statistics', {'compare': 'pairs'})


def test_validation_checks_pass():
    from dexvantage.validation.suite import Ctx, ValidationSettings, run_one, specs
    st = ValidationSettings(models=('stats',), quick=True)
    ctx = Ctx(None, None, None, st)
    checks = [run_one(sp, ctx) for sp in specs(st)]
    assert len(checks) == 13
    bad = [(c.id, c.error, c.exception) for c in checks if not c.passed]
    assert not bad, bad


def test_self_test_line():
    from dexvantage import selftest as T
    status, text = T._check_stats(None)
    assert status == T.OK and 'rejects 4' in text and 'statistics' in T.NAMES


def test_study_compares_two_variants_over_real_tolerance_draws():
    p = _two_variant_project()
    p.variants[1].sensors[1].mount.pose.position[2] += 0.06
    doc, _ = SS.run_spec({'source': 'variants', 'variants': {'b': 'Rig B', 'n': 4, 'metrics': ['coverage.covered']}},
                         p)
    c = doc['comparisons'][0]
    assert doc['common_random_numbers'] and c['paired'] and c['exact'] and c['a']['n'] == 4
    assert 'paired by seed' in doc['pairing'] and doc['headline']['word'] in ('clear difference',
                                                                             'no clear difference')
    assert c['nominal'][0] is not None and c['nominal'][0] != c['nominal'][1]
