"""Checks of the designed experiments (core/experiments.py, spec §7.11) against independent references.

Ported from Vantage validation/checks_experiments.py (the Ishigami check); the others are DexVantage's.

- experiments.sobol_ishigami: the Sobol indices of the whole pipeline (a Saltelli design over [−π, π]³ from a
  scrambled Sobol sequence, the run, the Jansen estimators) on the Ishigami function y = sin x1 + a sin² x2 +
  b x3⁴ sin x1 (a = 7, b = 0.1) against its analytic indices, derived again here: with x ~ U(−π, π), E sin² = ½,
  E sin⁴ = 3/8, E x⁴ = π⁴/5, E x⁸ = π⁸/9, the partial variances are V1 = ½(1 + bπ⁴/5)², V2 = a²/8 (the variance of
  sin² is 3/8 − 1/4), V13 = ½ b²(π⁸/9 − π⁸/25) = 8b²π⁸/225, V3 = V12 = V23 = V123 = 0, so S1 = (V1, V2, 0)/V and
  ST = (V1 + V13, V2, V13)/V with V = V1 + V2 + V13. Tolerance by the base sample count N: 2/√N (0.031 at N = 4096,
  0.0625 at the quick run's N = 1024), about 0.6 of the bootstrap 95 % half-width measured at those N (0.045–0.053
  and 0.088–0.095); every analytic index must also lie in its 95 % bootstrap interval.
- experiments.lhs_strata: the Latin hypercube's defining property (McKay, Beckman and Conover 1979): for n points, every
  factor's range cut into n equal strata holds exactly one point in each, checked on the factor values the design
  gives (not its unit coordinates) for n = 7, 50 and 256 and five factors. Reported: the largest |count − 1| over every
  stratum; tolerance 0 (a count is exact).
- experiments.sobol_discrepancy: the centred L2 discrepancy of the Sobol-sequence design (n = 256, four factors) and
  of the Latin hypercube, by Hickernell's (1998) closed form written out in core.experiments, against
  scipy.stats.qmc.discrepancy (method 'CD'); and the design's points against scipy.stats.qmc.Sobol's own with the same
  seed. Tolerance 1e-9 relative (floating point; the two sum the same terms in a different order).
- experiments.sobol_vs_random: the same Sobol design's CD² against the expectation for n independent uniform points,
  ((5/4)^d − (13/12)^d)/n (the kernel's diagonal mean less its off-diagonal mean, over n), as a ratio. A
  low-discrepancy sequence must sit well below random; tolerance 0.5 (a loose bound: 0.03–0.04 measured at n = 256).
- experiments.main_effects_linear: the main effects of a full factorial (three continuous factors at three levels, a
  categorical at two) and of a resolution IV fraction (seven two-level factors, 16 runs) on known models: y = 2 +
  3a − 1.5b + 0.5c + 4[d = B] (the effect of a factor from low to high is its coefficient times its range), and in
  the fraction y = linear + 2·x1·x2 − x3·x5 (two-factor interactions, which a resolution IV design keeps out of the
  main effects). Tolerance 1e-9 (the estimates are exact; floating point).
- experiments.fractional_resolution: the two-level fractions for 3 to 12 factors at resolutions III and IV: the
  resolution found by brute force over the design's columns (the shortest product of columns that is constant, the
  definition in Box, Hunter and Hunter 2005) must reach the one asked, the run count must be the smallest power of
  two the resolution allows (k + 1 ≤ n for III, 2k ≤ n for IV), and the columns must be balanced and orthogonal
  (SᵀS = nI). Reported: the number of designs failing; tolerance 0.
"""
from __future__ import annotations

import math

import numpy as np

from .suite import Outcome, check, figure, series

ISHIGAMI_A, ISHIGAMI_B = 7.0, 0.1


def ishigami(v, a=ISHIGAMI_A, b=ISHIGAMI_B):
    x1, x2, x3 = v['x1'], v['x2'], v['x3']
    return math.sin(x1) + a * math.sin(x2) ** 2 + b * x3 ** 4 * math.sin(x1)


def ishigami_indices(a=ISHIGAMI_A, b=ISHIGAMI_B):
    """The analytic first-order and total indices (module docstring)."""
    pi = math.pi
    V1 = 0.5 * (1 + b * pi ** 4 / 5) ** 2
    V2 = a * a / 8
    V13 = 0.5 * b * b * (pi ** 8 / 9 - pi ** 8 / 25)
    V = V1 + V2 + V13
    return np.array([V1, V2, 0.0]) / V, np.array([V1 + V13, V2, V13]) / V


@check('experiments.sobol_ishigami', 'experiments', 'Sobol indices of the Ishigami function',
       'largest difference of the first-order and total Sobol indices the experiment gives (Saltelli design from a '
       'scrambled Sobol sequence, N = 4096 base samples, Jansen estimators) from the Ishigami function\'s analytic '
       'indices (a = 7, b = 0.1); every analytic index must lie in its 95 % bootstrap interval',
       'the analytic Sobol indices of the Ishigami function, derived again in the check', 2 / math.sqrt(4096), 'index',
       refs=('sobol1993', 'saltelli2002', 'saltelli2010', 'jansen1999', 'ishigami1990', 'efron1993'))
def _sobol(ctx):
    from ..core import experiments as EX
    N = 1024 if ctx.quick else 4096
    d = {'factors': [{'path': f'x{i}', 'lo': -math.pi, 'hi': math.pi} for i in (1, 2, 3)],
         'design': {'method': 'saltelli', 'n': N, 'seed': int(ctx.settings.seed) % 100000},
         'analysis': {'bootstrap': 300}}
    s = EX.run_function(ishigami, d)['analysis']['y']['sobol']
    S1, ST = ishigami_indices()
    e = float(max(np.abs(np.array(s['S1']) - S1).max(), np.abs(np.array(s['ST']) - ST).max()))
    inside = all(lo <= x <= hi for (lo, hi), x in zip(s['S1_ci'] + s['ST_ci'], list(S1) + list(ST)))
    fig = figure('Sobol indices of the Ishigami function', 'factor x1, x2, x3', 'index',
                 [series('analytic S1', [1, 2, 3], S1), series('estimated S1', [1, 2, 3], s['S1'], 'points'),
                  series('analytic ST', [1, 2, 3], ST), series('estimated ST', [1, 2, 3], s['ST'], 'points')])
    return Outcome(e, {'N': N, 'S1': s['S1'], 'ST': s['ST'], 'S1_ci': s['S1_ci'], 'ST_ci': s['ST_ci'],
                       'analytic_S1': S1.round(5).tolist(), 'analytic_ST': ST.round(5).tolist(),
                       'all_inside_their_interval': inside}, fig, ok=inside, tolerance=2 / math.sqrt(N),
                   note=f'tolerance 2/√N at N = {N}')


@check('experiments.lhs_strata', 'experiments', 'Latin hypercube: one point in every stratum',
       'largest |count − 1| over the n equal strata of every factor\'s range, for the factor values of Latin '
       'hypercube designs of n = 7, 50 and 256 points over five factors',
       'the defining property of a Latin hypercube (McKay, Beckman and Conover 1979)', 0, 'points',
       refs=('mckay1979',))
def _lhs(ctx):
    from ..core import experiments as EX
    worst = 0
    ranges = [(-1.0, 1.0), (0.0, 0.3), (10.0, 20.0), (-5.0, -4.0), (0.0, 1.0)]
    for n in (7, 50, 256):
        d = EX.Design.from_dict({'factors': [{'path': f'f{j}', 'lo': lo, 'hi': hi} for j, (lo, hi) in
                                             enumerate(ranges)],
                                 'design': {'method': 'lhs', 'n': n, 'seed': int(ctx.settings.seed) % 1000 + n}})
        rows = EX.plan_rows(EX.make_plan(d), d.factors)
        for j, (lo, hi) in enumerate(ranges):
            x = np.array([r[f'f{j}'] for r in rows])
            k = np.clip(np.floor((x - lo) / (hi - lo) * n).astype(int), 0, n - 1)
            counts = np.bincount(k, minlength=n)
            worst = max(worst, int(np.abs(counts - 1).max()))
    return Outcome(float(worst), {'sizes': [7, 50, 256], 'factors': len(ranges)})


def _sobol_design(n, k, seed):
    from ..core import experiments as EX
    d = EX.Design.from_dict({'factors': [{'path': f'f{j}', 'lo': 0.0, 'hi': 1.0} for j in range(k)],
                             'design': {'method': 'sobol', 'n': n, 'seed': seed}})
    return np.array([[r[f'f{j}'] for j in range(k)] for r in EX.plan_rows(EX.make_plan(d), d.factors)])


@check('experiments.sobol_discrepancy', 'experiments', 'Sobol design: discrepancy against scipy',
       'relative difference of the centred L2 discrepancy (CD²) of a 256-point Sobol design and a 256-point Latin '
       'hypercube over four factors, by the closed form in core.experiments, from scipy.stats.qmc.discrepancy; and '
       'the largest difference of the design\'s points from scipy.stats.qmc.Sobol\'s with the same seed',
       'scipy.stats.qmc.discrepancy (method \'CD\') and scipy.stats.qmc.Sobol', 1e-9, 'relative',
       refs=('hickernell1998', 'scipy_qmc'))
def _disc(ctx):
    from scipy.stats import qmc
    from ..core import experiments as EX
    seed = int(ctx.settings.seed) % 100000
    U = _sobol_design(256, 4, seed)
    ref = EX._qmc_sobol(4, seed).random(256)
    d_pts = float(np.abs(U - ref).max())
    L = EX.Design.from_dict({'factors': [{'path': f'f{j}', 'lo': 0.0, 'hi': 1.0} for j in range(4)],
                             'design': {'method': 'lhs', 'n': 256, 'seed': seed}})
    V = EX.make_plan(L).U
    errs = []
    for X in (U, V):
        ours, theirs = EX.centred_l2_discrepancy(X), float(qmc.discrepancy(X, method='CD'))
        errs.append(abs(ours - theirs) / theirs)
    return Outcome(float(max(max(errs), d_pts)), {'cd2_sobol': EX.centred_l2_discrepancy(U),
                                                  'cd2_lhs': EX.centred_l2_discrepancy(V),
                                                  'relative_differences': errs, 'points_vs_scipy': d_pts})


@check('experiments.sobol_vs_random', 'experiments', 'Sobol design: discrepancy against random points',
       'the CD² of a 256-point Sobol design over four factors as a ratio of the expected CD² of 256 independent '
       'uniform points, ((5/4)^d − (13/12)^d)/n',
       'the expected centred L2 discrepancy of independent uniform points, derived from Hickernell\'s kernel', 0.5,
       'ratio', refs=('hickernell1998', 'scipy_qmc'))
def _vs_random(ctx):
    from ..core import experiments as EX
    U = _sobol_design(256, 4, int(ctx.settings.seed) % 100000)
    cd, iid = EX.centred_l2_discrepancy(U), EX.iid_discrepancy(256, 4)
    # the expectation itself checked against random draws (the mean over 40 sets lies within 15 % of it)
    rng = ctx.rng('experiments.sobol_vs_random')
    mc = float(np.mean([EX.centred_l2_discrepancy(rng.random((256, 4))) for _ in range(40)]))
    return Outcome(cd / iid, {'cd2_sobol': cd, 'cd2_iid_expected': iid, 'cd2_iid_mean_of_40_draws': mc},
                   ok=abs(mc - iid) / iid < 0.15)


@check('experiments.main_effects_linear', 'experiments', 'Main effects of known models',
       'largest difference of the main effects (mean at the last level minus mean at the first) of a full factorial '
       '(3 × 3 × 3 × 2) on y = 2 + 3a − 1.5b + 0.5c + 4[d = B] and of a resolution IV fraction (7 factors, 16 runs) '
       'on a linear model plus two two-factor interactions, from each coefficient times its factor\'s range',
       'the main effects of a linear model, its coefficients times the ranges exactly (and their freedom from '
       'two-factor interactions at resolution IV)', 1e-9, 'response', refs=('box_hunter2005',))
def _effects(ctx):
    from ..core import experiments as EX
    d = {'factors': [{'path': 'a', 'lo': 0.0, 'hi': 2.0, 'n_levels': 3}, {'path': 'b', 'lo': -1.0, 'hi': 1.0},
                     {'path': 'c', 'lo': 10.0, 'hi': 14.0}, {'path': 'd', 'levels': ['A', 'B']}],
         'design': {'method': 'full_factorial', 'levels': 3}}
    r = EX.run_function(lambda v: 2 + 3 * v['a'] - 1.5 * v['b'] + 0.5 * v['c'] + 4 * (v['d'] == 'B'), d)
    want = {'a': 3 * 2.0, 'b': -1.5 * 2.0, 'c': 0.5 * 4.0, 'd': 4.0}
    got = {m['factor']: m['effect'] for m in r['analysis']['y']['main_effects']}
    e1 = max(abs(got[k] - w) for k, w in want.items())
    coef = np.array([1.0, -2.0, 0.5, 3.0, 0.0, -1.0, 2.5])
    d2 = {'factors': [{'path': f'x{i + 1}', 'lo': -1.0, 'hi': 1.0} for i in range(7)],
          'design': {'method': 'fractional', 'resolution': 'IV'}}

    def f2(v):
        x = np.array([v[f'x{i + 1}'] for i in range(7)])
        return float(coef @ x + 2 * x[0] * x[1] - x[2] * x[4])
    r2 = EX.run_function(f2, d2)
    got2 = [m['effect'] for m in r2['analysis']['y']['main_effects']]
    e2 = float(np.abs(np.array(got2) - 2 * coef).max())
    return Outcome(float(max(e1, e2)), {'factorial_runs': int(len(r['rows'])), 'fraction_runs': int(len(r2['rows'])),
                                        'factorial_effects': got, 'fraction_effects': got2,
                                        'fraction_generators': r2['plan']['generators']})


@check('experiments.fractional_resolution', 'experiments', 'Two-level fractions reach their resolution',
       'number of the fractions for 3 to 12 factors at resolutions III and IV whose brute-force resolution is below '
       'the one asked, whose run count is not the smallest the resolution allows, or whose columns are not balanced '
       'and orthogonal',
       'the definition of resolution (the shortest word of the defining relation, Box, Hunter and Hunter 2005), '
       'searched by brute force', 0, 'designs', refs=('box_hunter2005',))
def _resolution(ctx):
    from ..core import experiments as EX
    bad, rows = [], []
    for k in range(3, 13):
        for res, need, nmin in (('III', 3, k + 1), ('IV', 4, 2 * k)):
            S, gens = EX.fractional_signs(k, res)
            n = S.shape[0]
            want_n = min(2 ** k, 2 ** math.ceil(math.log2(nmin)))
            found = EX.resolution_of(S)
            ok = (found is None or found >= need) and n == want_n and np.array_equal(S.T @ S, n * np.eye(k))
            rows.append({'k': k, 'asked': res, 'runs': n, 'found': found, 'generators': gens})
            if not ok:
                bad.append(f'{k} factors at {res}')
    return Outcome(float(len(bad)), {'failing': bad, 'designs': rows})
