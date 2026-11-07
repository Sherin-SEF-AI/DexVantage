"""Checks of the statistics and power analysis (core/stats.py) against independent references: SciPy's own
implementations (scipy.stats binomtest, bootstrap, permutation_test, ttest_ind, ttest_rel, mannwhitneyu, wilcoxon,
fisher_exact, contingency.odds_ratio), published worked examples and tables (Benjamini and Hochberg 1995, Cohen 1992,
the G*Power 3.1 manual), brute-force counting written out here, and Monte Carlo simulation.

- stats.proportion_ci: Wilson and Clopper-Pearson intervals for every k of n in {1, 7, 20, 50, 137} at 90, 95 and
  99 % against scipy.stats.binomtest(k, n).proportion_ci; tolerance 1e-9 (closed forms and beta quantiles: only
  rounding differs).
- stats.t_tests: Welch's and the paired t (p, t, the interval) on 20 random data sets against scipy ttest_ind
  (equal_var=False) and ttest_rel; tolerance 1e-9 (the same closed forms).
- stats.rank_tests: Mann-Whitney U (exact and asymptotic, with and without ties) and the Wilcoxon signed-rank test
  (exact, the sign-flip permutation with ties or zeros, asymptotic) for the three alternatives against scipy
  mannwhitneyu and wilcoxon; tolerance 1e-9 (the same null distributions, computed by other code).
- stats.fisher_odds: Fisher's exact test (three alternatives) and the sample odds ratio's Woolf interval on 2 × 2
  tables without a zero cell against scipy fisher_exact and contingency.odds_ratio(kind='sample'); tolerance 1e-9.
- stats.permutation_exact: the exact permutation tests (every relabelling, every sign flip) against
  scipy.stats.permutation_test with as many resamples as rearrangements (also exact); tolerance 1e-12.
- stats.permutation_mc: Monte Carlo permutation p-values (9 999 random rearrangements) against SciPy's with its own
  random stream: error the largest |Δp| in standard errors of the difference √(2 p (1 − p) / B); tolerance 4.
- stats.bootstrap: percentile and BCa intervals (one sample, paired, two samples, 9 999 resamples) against
  scipy.stats.bootstrap with another random stream: error the largest endpoint difference in units of the bootstrap
  standard error; tolerance 0.25 (the Monte Carlo standard error of a 2.5 % quantile of 9 999 replicates is
  √(q (1 − q) / B) / φ(z_q) ≈ 0.027 σ; BCa moves the upper end further into a skewed tail: measured over 30 seeds on
  exponential data (2026-11-07), each endpoint's standard deviation was 0.022 to 0.045 bootstrap standard errors for
  either implementation and the mean difference under 0.01, so the difference of two independent runs has a standard
  deviation under 0.06: 0.25 is about 4 of them).
- stats.multiplicity_bh1995: the 15 p-values of Benjamini and Hochberg (1995, section 3.2, p. 295): Bonferroni at
  0.05 rejects 3, the FDR procedure 4 (published); Holm and Benjamini-Yekutieli 3 each, and every adjusted p-value
  against the step-down and step-up definitions written out again here; error the count mismatches plus the largest
  adjusted-p difference; tolerance 1e-12.
- stats.power_cohen1992: n per group for power 0.80 at alpha 0.01, 0.05, 0.10 against Cohen (1992, Table 2, p. 158):
  the two-sample t (noncentral t, d = 0.2, 0.5, 0.8) and the difference of proportions (arcsine h = 0.2, 0.5, 0.8);
  error the largest |n − n_table|; tolerance 1 per group (Cohen's tables round to the nearest unit and come from
  approximations: for d = 0.2 at alpha 0.05 the exact noncentral t needs 394, as G*Power also gives, the table 393).
- stats.power_gpower: the G*Power 3.1 manual's worked examples (section 19.3, p. 48 and 20.3, p. 50): matched-pairs
  power 0.832114 and 0.500352 at N = 50, the one-sample N = 30 (power 0.955144) and N = 1492 (power 0.900169);
  error the largest power difference; tolerance 5e-6 (the manual prints six decimals), the N equal as a further
  condition.
- stats.power_simulation: the exact power of Fisher's exact test (every outcome enumerated) against simulated
  evaluations each tested with scipy.stats.fisher_exact, and the t-test power (noncentral t) against simulated
  normal samples tested with scipy ttest_ind / ttest_rel: error the largest |Δ| in Monte Carlo standard errors;
  tolerance 4. The normal approximation with continuity correction (Fleiss) against the exact power of the
  Yates-corrected z test is reported in the detail (an approximation, not a check of code).
- stats.ci_coverage: the coverage of the noncentral-t interval of Cohen's d (exact for normal data) and of the
  Newcombe interval of a risk difference (approximate) over simulated data: error |coverage − 0.95| in Monte Carlo
  standard errors, with 0.01 allowed for Newcombe's own approximation; tolerance 4.
- stats.bruteforce: Cliff's delta against counting every pair, the matched-pairs rank-biserial against the signed
  ranks summed by a loop, and the smallest detectable difference fed back into the power function (power equals the
  target); tolerance 1e-9.
"""
from __future__ import annotations

import math

import numpy as np

from .suite import Outcome, check

BH1995_P = (0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344, 0.0459, 0.3240, 0.4262, 0.5719, 0.6528,
            0.7590, 1.000)                      # Benjamini and Hochberg 1995, section 3.2, p. 295
BH1995_REJECT = {'bonferroni': 3, 'bh': 4}      # published (p. 295); Holm and BY are computed by hand below
COHEN1992 = {                                   # Table 2, p. 158: N per group at power 0.80, alpha .01 / .05 / .10
    't2': {0.2: (586, 393, 310), 0.5: (95, 64, 50), 0.8: (38, 26, 20)},
    'h': {0.2: (584, 392, 309), 0.5: (93, 63, 49), 0.8: (36, 25, 19)},
}
COHEN_ALPHAS = (0.01, 0.05, 0.10)
GPOWER = (                                      # G*Power 3.1 manual, sections 19.3 (p. 48) and 20.3 (p. 50)
    {'what': 'matched pairs, d_z 0.421637, N 50, two-sided 0.05', 'd': 0.421637, 'n': 50, 'alpha': 0.05,
     'alt': 'two-sided', 'power': 0.832114},
    {'what': 'matched pairs, d_z 0.2828427, N 50, two-sided 0.05', 'd': 0.2828427, 'n': 50, 'alpha': 0.05,
     'alt': 'two-sided', 'power': 0.500352},
    {'what': 'one sample, d 0.625, one-sided 0.05, target 0.95', 'd': 0.625, 'alpha': 0.05, 'alt': 'greater',
     'target': 0.95, 'N': 30, 'power': 0.955144},
    {'what': 'one sample, d 0.1, two-sided 0.01, target 0.90', 'd': 0.1, 'alpha': 0.01, 'alt': 'two-sided',
     'target': 0.90, 'N': 1492, 'power': 0.900169},
)


def _rng(ctx, cid):
    return ctx.rng(cid)


@check('stats.proportion_ci', 'stats', 'Wilson and Clopper-Pearson intervals',
       'largest endpoint difference over every k of five n at three confidence levels',
       'scipy.stats.binomtest(k, n).proportion_ci (wilson, exact)', 1e-9, 'proportion',
       refs=('wilson1927', 'clopper_pearson1934', 'scipy_stats_tests'))
def _ci(ctx):
    from scipy import stats as ss

    from ..core import stats as S
    worst, n_cmp = 0.0, 0
    for n in (1, 7, 20, 50, 137):
        for k in range(n + 1):
            bt = ss.binomtest(k, n)
            for c in (0.90, 0.95, 0.99):
                for meth, fn in (('wilson', S.wilson), ('exact', S.clopper_pearson)):
                    r = bt.proportion_ci(c, meth)
                    a = fn(k, n, c)
                    worst = max(worst, abs(r.low - a[0]), abs(r.high - a[1]))
                    n_cmp += 1
    return Outcome(worst, {'intervals_compared': n_cmp})


@check('stats.t_tests', 'stats', 'Welch and paired t tests',
       'largest difference of p, t and the interval ends over 20 random data sets',
       'scipy.stats.ttest_ind(equal_var=False) and ttest_rel with confidence_interval()', 1e-9, 'p or value',
       refs=('scipy_stats_tests',))
def _t(ctx):
    from scipy import stats as ss

    from ..core import stats as S
    rng = _rng(ctx, 'stats.t_tests')
    worst = 0.0
    for _ in range(20):
        n1, n2 = rng.integers(3, 40, 2)
        x = rng.normal(0, rng.uniform(0.5, 3), n1)
        y = rng.normal(rng.normal(), rng.uniform(0.5, 3), n2)
        r, a = ss.ttest_ind(y, x, equal_var=False), S.welch_t(x, y)
        ci = r.confidence_interval()
        worst = max(worst, abs(r.pvalue - a['p']), abs(r.statistic - a['t']) / max(1, abs(r.statistic)),
                    abs(ci.low - a['ci'][0]), abs(ci.high - a['ci'][1]))
        y2 = x + rng.normal(rng.normal(0, 0.5), 1, n1)
        r, a = ss.ttest_rel(y2, x), S.paired_t(x, y2)
        ci = r.confidence_interval()
        worst = max(worst, abs(r.pvalue - a['p']), abs(ci.low - a['ci'][0]), abs(ci.high - a['ci'][1]))
    return Outcome(worst, {'data_sets': 20})


@check('stats.rank_tests', 'stats', 'Mann-Whitney U and Wilcoxon signed-rank tests',
       'largest p-value difference over exact, permutation and asymptotic cases and three alternatives',
       'scipy.stats.mannwhitneyu and scipy.stats.wilcoxon (method auto)', 1e-9, 'p',
       refs=('scipy_stats_tests',))
def _ranks(ctx):
    from scipy import stats as ss

    from ..core import stats as S
    rng = _rng(ctx, 'stats.rank_tests')
    worst, methods = 0.0, set()
    for n1, n2, tie in ((5, 7, False), (3, 40, False), (8, 8, False), (12, 15, False), (6, 9, True), (30, 40, True)):
        x, y = rng.normal(size=n1), rng.normal(0.5, size=n2)
        if tie:
            x, y = np.round(x), np.round(y)
        for alt in ('two-sided', 'less', 'greater'):
            r, a = ss.mannwhitneyu(y, x, alternative=alt), S.mann_whitney(x, y, alt)
            methods.add('mwu ' + a['method'])
            worst = max(worst, abs(r.pvalue - a['p']), abs(r.statistic - a['U']) / max(1, r.statistic))
    for n, tie, zero in ((10, False, False), (30, False, False), (60, False, False), (10, True, False),
                         (12, False, True), (20, True, True)):
        x, d = rng.normal(size=n), rng.normal(0.4, 1, n)
        if tie:
            d = np.round(d * 2) / 2
        if zero:
            d[:2] = 0
        for alt in ('two-sided', 'less', 'greater'):
            y = x + d                                  # SciPy is given y − x as computed (ties as rounded)
            r, a = ss.wilcoxon(y - x, alternative=alt), S.wilcoxon_signed_rank(x, y, alt)
            methods.add('wilcoxon ' + a['method'])
            worst = max(worst, abs(r.pvalue - a['p']))
    return Outcome(worst, {'methods_covered': sorted(methods)})


@check('stats.fisher_odds', 'stats', 'Fisher\'s exact test and the odds ratio interval',
       'largest difference of p and of the Woolf interval ends over eight 2 × 2 tables',
       'scipy.stats.fisher_exact and scipy.stats.contingency.odds_ratio(kind="sample")', 1e-9, 'p or ratio',
       refs=('scipy_stats_tests',))
def _fisher(ctx):
    from scipy import stats as ss

    from ..core import stats as S
    worst = 0.0
    tables = ((3, 10, 8, 10), (1, 12, 5, 11), (14, 20, 19, 20), (45, 100, 60, 100), (7, 9, 2, 13), (30, 31, 25, 40),
              (2, 50, 9, 48), (18, 25, 18, 25))
    for k1, n1, k2, n2 in tables:
        for alt in ('two-sided', 'less', 'greater'):
            # condition 2 in the first row: SciPy's alternative is then about B against A, as DexVantage's is
            r = ss.fisher_exact([[k2, n2 - k2], [k1, n1 - k1]], alternative=alt)
            worst = max(worst, abs(r.pvalue - S.fisher_exact(k1, n1, k2, n2, alt)['p']))
        o = ss.contingency.odds_ratio([[k2, n2 - k2], [k1, n1 - k1]], kind='sample')
        ci, a = o.confidence_interval(), S.odds_ratio(k1, n1, k2, n2)
        worst = max(worst, abs(o.statistic - a['value']) / o.statistic, abs(ci.low - a['ci'][0]) / ci.low,
                    abs(ci.high - a['ci'][1]) / ci.high)
    return Outcome(worst, {'tables': len(tables)})


def _mean_diff(a, b, axis=-1):
    return np.mean(b, axis=axis) - np.mean(a, axis=axis)


def _mean_paired(a, b, axis=-1):
    return np.mean(b - a, axis=axis)


@check('stats.permutation_exact', 'stats', 'Exact permutation tests',
       'largest p-value difference over unpaired (every relabelling) and paired (every sign flip) tests',
       'scipy.stats.permutation_test with as many resamples as rearrangements (its exact mode)', 1e-12, 'p',
       refs=('scipy_stats_tests',))
def _perm_exact(ctx):
    from scipy import stats as ss

    from ..core import stats as S
    rng = _rng(ctx, 'stats.permutation_exact')
    worst, exact = 0.0, True
    for n1, n2 in ((4, 5), (6, 7), (3, 9)):
        x, y = rng.normal(size=n1), rng.normal(0.8, size=n2)
        for alt in ('two-sided', 'less', 'greater'):
            r = ss.permutation_test((x, y), _mean_diff, n_resamples=9999, vectorized=True, alternative=alt)
            a = S.permutation_test(x, y, alternative=alt)
            exact = exact and a['exact']
            worst = max(worst, abs(r.pvalue - a['p']))
    for n in (6, 10, 13):
        x = rng.normal(size=n)
        y = x + rng.normal(0.5, 1, n)
        for alt in ('two-sided', 'less', 'greater'):
            r = ss.permutation_test((x, y), _mean_paired, permutation_type='samples', n_resamples=9999,
                                    vectorized=True, alternative=alt)
            a = S.permutation_test(x, y, paired=True, alternative=alt)
            exact = exact and a['exact']
            worst = max(worst, abs(r.pvalue - a['p']))
    return Outcome(worst, {'cases': 18, 'all_exact': exact}, ok=exact)


@check('stats.permutation_mc', 'stats', 'Monte Carlo permutation tests',
       'largest |Δp| between DexVantage and SciPy (9 999 random rearrangements each), in standard errors',
       'scipy.stats.permutation_test with its own random stream', 4.0, 'standard errors',
       refs=('scipy_stats_tests',))
def _perm_mc(ctx):
    from scipy import stats as ss

    from ..core import stats as S
    rng = _rng(ctx, 'stats.permutation_mc')
    B = 2999 if ctx.quick else 9999
    worst, rows = 0.0, []
    for k, (n1, n2, shift) in enumerate(((20, 25, 0.5), (30, 30, 0.3), (15, 40, 0.0))):
        x, y = rng.normal(size=n1), rng.normal(shift, size=n2)
        r = ss.permutation_test((x, y), _mean_diff, n_resamples=B, vectorized=True, rng=1000 + k)
        a = S.permutation_test(x, y, n_resamples=B, seed=k)
        p = max(min((r.pvalue + a['p']) / 2, 1 - 1 / B), 1 / B)
        se = math.sqrt(2 * p * (1 - p) / B) * 2               # two-sided: twice a tail
        z = abs(r.pvalue - a['p']) / se
        worst = max(worst, z)
        rows.append({'n': [n1, n2], 'scipy': r.pvalue, 'dexvantage': a['p'], 'z': z})
    x = rng.normal(size=25)
    y = x + rng.normal(0.25, 1, 25)
    r = ss.permutation_test((x, y), _mean_paired, permutation_type='samples', n_resamples=B, vectorized=True, rng=7)
    a = S.permutation_test(x, y, paired=True, n_resamples=B, seed=7)
    p = max(min((r.pvalue + a['p']) / 2, 1 - 1 / B), 1 / B)
    z = abs(r.pvalue - a['p']) / (2 * math.sqrt(2 * p * (1 - p) / B))
    rows.append({'n': [25], 'paired': True, 'scipy': r.pvalue, 'dexvantage': a['p'], 'z': z})
    return Outcome(max(worst, z), {'resamples': B, 'cases': rows})


@check('stats.bootstrap', 'stats', 'Bootstrap percentile and BCa intervals',
       'largest endpoint difference in bootstrap standard errors (one sample, paired, two samples)',
       'scipy.stats.bootstrap (percentile, BCa) with its own random stream', 0.25, 'standard errors',
       refs=('efron1993', 'scipy_stats_tests'))
def _boot(ctx):
    from scipy import stats as ss

    from ..core import stats as S
    rng = _rng(ctx, 'stats.bootstrap')
    B = 9999
    worst, rows = 0.0, []
    x, y = rng.exponential(size=25), rng.exponential(1.5, size=30)
    xp = rng.exponential(size=25)
    yp = xp * rng.uniform(0.8, 1.6, 25)
    cases = (('one sample', (x,), None, False, np.mean),
             ('paired', (xp, yp), yp, True, _mean_paired),
             ('two samples', (x, y), y, False, _mean_diff))
    for name, data, yy, paired, f in cases:
        for m in ('percentile', 'BCa'):
            r = ss.bootstrap(data, f, paired=paired, method=m, n_resamples=B, vectorized=True,
                             rng=len(rows) + 11)
            a = S.bootstrap_ci(data[0], yy, paired, 'mean', m.lower(), B, 0.95, seed=len(rows))
            e = max(abs(r.confidence_interval.low - a['ci'][0]), abs(r.confidence_interval.high - a['ci'][1])) / a['se']
            worst = max(worst, e)
            rows.append({'case': name, 'method': m, 'scipy': [r.confidence_interval.low, r.confidence_interval.high],
                         'dexvantage': a['ci'], 'error_se': e})
    return Outcome(worst, {'resamples': B, 'cases': rows})


@check('stats.multiplicity_bh1995', 'stats', 'Multiple-comparison corrections (Benjamini and Hochberg 1995 example)',
       'rejection-count mismatches plus the largest adjusted-p difference',
       'Benjamini and Hochberg (1995, p. 295) published rejections; Holm and BY written out again here', 1e-12,
       'count or p', refs=('benjamini_hochberg1995', 'holm1979', 'benjamini_yekutieli2001'))
def _bh(ctx):
    from ..core import stats as S
    p = list(BH1995_P)
    m = len(p)
    alpha = 0.05
    # written out again: sorted p (the published list is sorted)
    bonf = [min(1.0, m * x) for x in p]
    holm, run = [], 0.0
    for i, x in enumerate(p):
        run = max(run, min(1.0, (m - i) * x))
        holm.append(run)
    cm = sum(1.0 / i for i in range(1, m + 1))

    def stepup(c):
        out, best = [0.0] * m, 1.0
        for i in range(m - 1, -1, -1):
            best = min(best, c * m * p[i] / (i + 1))
            out[i] = min(1.0, best)
        return out
    bh, by = stepup(1.0), stepup(cm)
    hand = {'bonferroni': bonf, 'holm': holm, 'bh': bh, 'by': by}
    k_bh = max([i + 1 for i in range(m) if p[i] <= (i + 1) / m * alpha] or [0])        # the published procedure
    k_by = max([i + 1 for i in range(m) if p[i] <= (i + 1) / (m * cm) * alpha] or [0])
    k_holm = 0
    for i in range(m):
        if p[i] <= alpha / (m - i):
            k_holm += 1
        else:
            break
    want = {'bonferroni': BH1995_REJECT['bonferroni'], 'bh': BH1995_REJECT['bh'], 'holm': k_holm, 'by': k_by}
    miss, dpmax, rows = 0, 0.0, {}
    for meth in ('bonferroni', 'holm', 'bh', 'by'):
        r = S.adjust(p, meth, alpha)
        got = sum(r['reject'])
        dp = max(abs(a - b) for a, b in zip(r['adjusted'], hand[meth]))
        miss += abs(got - want[meth])
        dpmax = max(dpmax, dp)
        rows[meth] = {'rejected': got, 'expected': want[meth], 'max_adjusted_diff': dp}
    err = miss + dpmax
    ok = k_bh == BH1995_REJECT['bh'] and k_holm == 3
    return Outcome(err, {'methods': rows, 'published': 'Bonferroni 3, FDR (BH) 4 at 0.05', 'hand_bh': k_bh,
                         'hand_holm': k_holm, 'hand_by': k_by}, ok=ok)


@check('stats.power_cohen1992', 'stats', 'Sample size against Cohen\'s power tables',
       'largest |n − n_table| per group over d and h = 0.2, 0.5, 0.8 at alpha 0.01, 0.05, 0.10 (power 0.80)',
       'Cohen (1992), "A power primer", Table 2, p. 158 (rows 1 and 5)', 1.0, 'per group',
       refs=('cohen1992', 'cohen1988'))
def _cohen(ctx):
    from ..core import stats as S
    worst, rows = 0.0, []
    for eff, ns in COHEN1992['t2'].items():
        for a, want in zip(COHEN_ALPHAS, ns):
            got = S.n_t(eff, a, 0.80)['n_per_group']
            worst = max(worst, abs(got - want))
            rows.append({'test': 'mean difference', 'effect': eff, 'alpha': a, 'table': want, 'dexvantage': got})
    for h, ns in COHEN1992['h'].items():
        p1 = 0.5
        p2 = math.sin((2 * math.asin(math.sqrt(p1)) + h) / 2) ** 2
        for a, want in zip(COHEN_ALPHAS, ns):
            got = S.n_two_proportions(p1, p2, a, 0.80, 'arcsine')['n_per_group']
            worst = max(worst, abs(got - want))
            rows.append({'test': 'P difference (arcsine)', 'effect': h, 'alpha': a, 'table': want,
                         'dexvantage': got})
    return Outcome(float(worst), {'rows': rows})


@check('stats.power_gpower', 'stats', 'Power against the G*Power manual\'s examples',
       'largest power difference over four documented examples (N equal as a further condition)',
       'G*Power 3.1 manual, sections 19.3 (p. 48) and 20.3 (p. 50)', 5e-6, 'power', refs=('gpower31_manual',))
def _gpower(ctx):
    from ..core import stats as S
    worst, ok, rows = 0.0, True, []
    for ex in GPOWER:
        if 'n' in ex:
            got = S.power_t(ex['d'], ex['n'], ex['alpha'], paired=True, alternative=ex['alt'])
            rows.append({**ex, 'dexvantage': got})
        else:
            r = S.n_t(ex['d'], ex['alpha'], ex['target'], paired=True, alternative=ex['alt'])
            got = r['power_at_n']
            ok = ok and r['n_per_group'] == ex['N']
            rows.append({**ex, 'dexvantage_N': r['n_per_group'], 'dexvantage': got})
        worst = max(worst, abs(got - ex['power']))
    return Outcome(worst, {'examples': rows}, ok=ok)


@check('stats.power_simulation', 'stats', 'Exact and t-test power against simulation',
       'largest |power − simulated power| in Monte Carlo standard errors',
       'simulated evaluations tested with scipy.stats.fisher_exact, ttest_ind and ttest_rel', 4.0,
       'standard errors', refs=('scipy_stats_tests', 'fleiss2003'), cost='study')
def _power_sim(ctx):
    from scipy import stats as ss

    from ..core import stats as S
    rng = _rng(ctx, 'stats.power_simulation')
    R = 600 if ctx.quick else 2000
    worst, rows = 0.0, []
    for p1, p2, n in ((0.5, 0.8, 30), (0.7, 0.9, 50), (0.2, 0.25, 40)):
        exact = S.power_two_proportions(p1, p2, n, n, 0.05, 'exact', 'fisher')
        k1, k2 = rng.binomial(n, p1, R), rng.binomial(n, p2, R)
        sim = float(np.mean([ss.fisher_exact([[a, n - a], [b, n - b]]).pvalue <= 0.05 for a, b in zip(k1, k2)]))
        se = math.sqrt(max(exact * (1 - exact), 1e-4) / R)
        z = abs(sim - exact) / se
        worst = max(worst, z)
        yates = S.power_two_proportions(p1, p2, n, n, 0.05, 'exact', 'z', continuity=True)
        normal = S.power_two_proportions(p1, p2, n, n, 0.05, 'normal', continuity=True)
        rows.append({'p1': p1, 'p2': p2, 'n': n, 'fisher_exact_power': exact, 'simulated': sim, 'z': z,
                     'yates_z_exact_power': yates, 'normal_approximation': normal})
    for d, n, paired in ((0.5, 20, False), (0.8, 12, False), (0.5, 15, True)):
        pw = S.power_t(d, n, 0.05, paired=paired)
        if paired:
            dd = rng.normal(d, 1, (R, n))
            sim = float(np.mean(ss.ttest_1samp(dd, 0, axis=1).pvalue <= 0.05))
        else:
            x, y = rng.normal(0, 1, (R, n)), rng.normal(d, 1, (R, n))
            sim = float(np.mean(ss.ttest_ind(y, x, axis=1).pvalue <= 0.05))
        z = abs(sim - pw) / math.sqrt(pw * (1 - pw) / R)
        worst = max(worst, z)
        rows.append({'d': d, 'n': n, 'paired': paired, 'power': pw, 'simulated': sim, 'z': z})
    return Outcome(worst, {'replicates': R, 'cases': rows})


@check('stats.ci_coverage', 'stats', 'Coverage of the effect-size intervals',
       'largest |coverage − 0.95| in Monte Carlo standard errors (Newcombe: beyond 0.01 for its approximation)',
       'simulated normal samples (Cohen\'s d, noncentral t) and binomial counts (Newcombe risk difference)', 4.0,
       'standard errors', refs=('steiger_fouladi1997', 'newcombe1998'), cost='study')
def _coverage(ctx):
    from ..core import stats as S
    rng = _rng(ctx, 'stats.ci_coverage')
    R = 400 if ctx.quick else 1500
    rows, worst = [], 0.0
    for n1, n2, d in ((12, 15, 0.5), (8, 8, 1.0)):
        hit = 0
        for _ in range(R):
            x, y = rng.normal(0, 1, n1), rng.normal(d, 1, n2)
            ci = S.cohens_d(x, y)['ci']
            hit += ci[0] <= d <= ci[1]
        cov = hit / R
        z = abs(cov - 0.95) / math.sqrt(0.95 * 0.05 / R)
        worst = max(worst, z)
        rows.append({'interval': 'Cohen\'s d (noncentral t)', 'n': [n1, n2], 'd': d, 'coverage': cov, 'z': z})
    R2 = 4 * R
    for p1, n1, p2, n2 in ((0.3, 40, 0.5, 35), (0.85, 20, 0.6, 25)):
        k1, k2 = rng.binomial(n1, p1, R2), rng.binomial(n2, p2, R2)
        hit = 0
        for a, b in zip(k1, k2):
            ci = S.risk_difference(a, n1, b, n2)['ci']
            hit += ci[0] <= p2 - p1 <= ci[1]
        cov = hit / R2
        z = max(0.0, abs(cov - 0.95) - 0.01) / math.sqrt(0.95 * 0.05 / R2)
        worst = max(worst, z)
        rows.append({'interval': 'risk difference (Newcombe)', 'p': [p1, p2], 'n': [n1, n2], 'coverage': cov,
                     'z_beyond_0.01': z})
    return Outcome(worst, {'replicates': R, 'cases': rows})


@check('stats.bruteforce', 'stats', 'Rank effect sizes and the detectable difference',
       'largest difference against counting every pair, summing signed ranks in a loop, and power at the '
       'detectable difference against the target', 'brute force written out here; core/stats.power_t and the normal '
       'two-proportion power fed back', 1e-9, 'value', refs=('cliff1993',))
def _brute(ctx):
    from ..core import stats as S
    rng = _rng(ctx, 'stats.bruteforce')
    worst = 0.0
    for n1, n2 in ((7, 9), (20, 13)):
        x, y = np.round(rng.normal(size=n1), 1), np.round(rng.normal(0.4, size=n2), 1)
        gt = sum(1 for a in x for b in y if b > a)
        lt = sum(1 for a in x for b in y if b < a)
        worst = max(worst, abs(S.cliffs_delta(x, y)['value'] - (gt - lt) / (n1 * n2)))
    x = rng.normal(size=15)
    y = x + np.round(rng.normal(0.3, 1, 15), 1)
    d = [b - a for a, b in zip(x, y) if b != a]
    ad = sorted(abs(v) for v in d)
    tp = tm = 0.0
    for v in d:
        lo = ad.index(abs(v))
        cnt = ad.count(abs(v))
        rk = lo + (cnt + 1) / 2
        if v > 0:
            tp += rk
        else:
            tm += rk
    worst = max(worst, abs(S.rank_biserial_paired(x, y) - (tp - tm) / (tp + tm)))
    m = S.detectable_difference('t2', 25, 0.05, 0.8)
    worst = max(worst, abs(S.power_t(m['d'], 25, 0.05) - 0.8))
    r = S.detectable_difference('two_proportions', 40, 0.05, 0.8, p1=0.6)
    worst = max(worst, abs(S.power_two_proportions(0.6, r['down'], 40, 40, 0.05, 'normal') - 0.8),
                abs(S.power_two_proportions(0.6, r['up'], 40, 40, 0.05, 'normal') - 0.8))
    return Outcome(worst, {'t2_detectable_d_at_25': m['d'], 'rates_detectable_from_0.6_at_40': [r['down'], r['up']]})
