"""Statistics and power analysis for comparing rigs, runs and policies (research tools part 2): pure NumPy and SciPy,
no project, no Qt. core/statcompare.py turns the app's sources (variants over the tolerance Monte Carlo, experiment
runs, tracker runs, success counts) into the samples these functions compare; core/studies/stats.py is the study.

Every function returns plain numbers or a JSON-able dict that says which method ran and why. Inputs that cannot be
analysed raise StatsError (a core.jobs.StudyError) with a plain sentence.

Success rates (k successes in n trials)
    wilson(k, n, confidence)            Wilson (1927) score interval
    clopper_pearson(k, n, confidence)   Clopper and Pearson (1934) "exact" interval (beta quantiles): never below the
                                        nominal coverage, wider than Wilson
    risk_difference(k1, n1, k2, n2)     p2 − p1 with Newcombe's (1998, method 10) hybrid score interval
    odds_ratio(k1, n1, k2, n2)          the sample odds ratio with Woolf's logit interval (Haldane's +½ in every cell
                                        when a cell is zero, said)
    fisher_exact(k1, n1, k2, n2)        Fisher's exact test (two-sided: the tables no more likely than the observed)
    two_proportion_z(k1, n1, k2, n2)    the pooled two-proportion z test (optionally with Yates' correction)
    compare_rates(k1, n1, k2, n2, test) all of the above for two conditions, with the test's reason and assumptions

Continuous samples (metric values over draws, runs or episodes)
    bootstrap_ci(x, y, paired, statistic, method)   percentile or BCa (Efron 1987) intervals, seeded; one sample, a
                                        paired difference or a two-sample difference (each sample resampled alone)
    permutation_test(x, y, paired)      the difference of means: sign flips of the paired differences or relabellings
                                        of the pooled values; exact (every rearrangement) when there are at most
                                        n_resamples of them, else n_resamples random ones with p = (b + 1)/(B + 1)
                                        (Phipson and Smyth 2010)
    welch_t, paired_t                   Welch's (1947) unequal-variance t and the paired t, with the mean difference's
                                        t interval
    mann_whitney(x, y)                  the Mann-Whitney U test: the exact null distribution when one sample has at
                                        most 8 values and there are no ties, else the normal approximation with the
                                        tie correction and a continuity correction (the rule SciPy uses)
    wilcoxon_signed_rank(x, y)          the Wilcoxon signed-rank test on y − x, zero differences dropped (Wilcoxon's
                                        rule): exact without ties or zeros up to 50 pairs, all 2^n sign flips with
                                        ties or zeros up to 13 pairs, else the normal approximation with the tie
                                        correction (SciPy's rule)
    cohens_d(x, y, paired)              Cohen's d (pooled σ; paired: d_z = mean(d)/sd(d)) and Hedges' g (the exact
                                        small-sample factor J), with intervals from the noncentral t (Steiger and
                                        Fouladi 1997)
    cliffs_delta(x, y)                  P(Y > X) − P(Y < X), equal to the rank-biserial correlation 2U/(n₁n₂) − 1
                                        (Kerby 2014), with a seeded percentile bootstrap interval
    rank_biserial_paired(x, y)          (T⁺ − T⁻)/(T⁺ + T⁻) of the signed ranks (Kerby 2014)
    compare_samples(x, y, paired)       the comparison the app shows: the test chosen and why, the effect sizes with
                                        intervals, the reference tests, the smallest difference the data could detect

Several comparisons
    adjust(pvalues, method)             Bonferroni, Holm (1979), Benjamini-Hochberg (1995) or Benjamini-Yekutieli
                                        (2001) adjusted p-values and the rejections at alpha

Power and sample size
    power_two_proportions(p1, p2, n1, n2, alpha, method)   'normal' (Fleiss, Levin and Paik 2003: the pooled-variance
                                        normal approximation, with continuity correction by default), 'arcsine' (Cohen
                                        1988: h = 2 asin √p₂ − 2 asin √p₁), 'exact' (every outcome of both binomials
                                        enumerated and the chosen test applied to each: Fisher's exact or the z test)
                                        or 'simulation' (seeded draws of both binomials, the test applied to each)
    power_t(d, n, alpha, paired)        two-sample (n per group, df 2n − 2, λ = d √(n/2)) or paired (df n − 1,
                                        λ = d_z √n) t test through the noncentral t distribution
    n_two_proportions, n_t              the smallest n per group that reaches the target power
    power_curve(kind, ns, ...)          power at each n
    episodes_needed(p1, p2, ...)        evaluation episodes per condition to tell success rates p1 and p2 apart
    demonstrations_needed(...)          demonstrations per condition to tell a per-demonstration figure apart, with
                                        the operators' clustering (Kish's design effect) and the share the audit
                                        discards
    detectable_difference(...)          the smallest difference n per group detects with the target power

Defaults (DexVantage defaults, labelled so in the UI and the results, never standards): alpha 0.05, power 0.80,
confidence 0.95, 9 999 bootstrap resamples and permutations, seed 0, Holm's correction across figures.

Known limits
------------
Sequential testing and early stopping (looking at the data as episodes come in and stopping when a test turns
significant) are out of scope: every test here is a fixed-sample test, and looking repeatedly inflates the false-
positive rate. Group-sequential designs (O'Brien-Fleming, alpha spending) are not offered. Power for rates assumes
independent episodes with one success probability per condition (no drift, no learning, no correlation between
episodes of one scene or operator); for demonstrations, clustering is only the design effect 1 + (m − 1) ρ, which
assumes equal cluster sizes. The t-based power assumes normal data with equal variances; the rank tests' power is not
computed (their asymptotic relative efficiency against the t test is at least 0.864, Hodges and Lehmann 1956).
BCa intervals need more than a handful of values (with fewer than about 10 their coverage is poor); permutation tests
test exchangeability, so an unpaired test also reacts to a difference of spread, not only of the means. Equivalence
("the variants do not differ") is never concluded from a non-significant test: the smallest detectable difference is
shown instead (Hoenig and Heisey 2001 explain why observed power is not reported).
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from .jobs import StudyError

ALPHA = 0.05                # DexVantage defaults (labelled defaults wherever they show)
POWER = 0.80
CONFIDENCE = 0.95
N_BOOT = 9999
N_PERM = 9999
SEED = 0
CORRECTION = 'holm'
CORRECTIONS = ('holm', 'bonferroni', 'bh', 'by', 'none')
CORRECTION_LABEL = {'holm': 'Holm', 'bonferroni': 'Bonferroni', 'bh': 'Benjamini-Hochberg',
                    'by': 'Benjamini-Yekutieli', 'none': 'no correction'}
POWER_METHODS = ('normal', 'arcsine', 'exact', 'simulation')
RATE_TESTS = ('fisher', 'z')
EXACT_N_MAX = 2000          # 'exact' two-proportion power enumerates (n1 + 1)(n2 + 1) outcomes; above this, normal
N_MAX = 100_000             # sample sizes searched up to this per group
SIM_DEFAULT = 4000          # 'simulation' draws (a default)
FISHER_RTOL = 1 + 1e-7      # tables "as likely as the observed" (the tolerance R and SciPy use)

OUT_OF_SCOPE = ('Sequential testing and early stopping are out of scope: these are fixed-sample tests, so decide the '
                'number of episodes or draws first (the power calculator) and test once.')

REFERENCES = (
    'E. B. Wilson, "Probable inference, the law of succession, and statistical inference", JASA 22(158), 209-212, '
    '1927.',
    'C. J. Clopper and E. S. Pearson, "The use of confidence or fiducial limits illustrated in the case of the '
    'binomial", Biometrika 26(4), 404-413, 1934.',
    'R. G. Newcombe, "Interval estimation for the difference between independent proportions: comparison of eleven '
    'methods", Statistics in Medicine 17(8), 873-890, 1998 (method 10).',
    'B. Efron, "Better bootstrap confidence intervals", JASA 82(397), 171-185, 1987 (BCa).',
    'B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap, Chapman & Hall, 1993.',
    'B. Phipson and G. K. Smyth, "Permutation p-values should never be zero", Statistical Applications in Genetics '
    'and Molecular Biology 9(1), article 39, 2010.',
    'B. L. Welch, "The generalization of Student\'s problem when several different population variances are '
    'involved", Biometrika 34(1-2), 28-35, 1947.',
    'H. B. Mann and D. R. Whitney, "On a test of whether one of two random variables is stochastically larger than '
    'the other", Annals of Mathematical Statistics 18(1), 50-60, 1947.',
    'F. Wilcoxon, "Individual comparisons by ranking methods", Biometrics Bulletin 1(6), 80-83, 1945.',
    'J. Cohen, Statistical Power Analysis for the Behavioral Sciences, 2nd ed., Lawrence Erlbaum, 1988.',
    'L. V. Hedges and I. Olkin, Statistical Methods for Meta-Analysis, Academic Press, 1985 (Hedges\' g, J).',
    'J. H. Steiger and R. T. Fouladi, "Noncentrality interval estimation and the evaluation of statistical models", '
    'in What If There Were No Significance Tests?, Erlbaum, 1997.',
    'N. Cliff, "Dominance statistics: ordinal analyses to answer ordinal questions", Psychological Bulletin 114(3), '
    '494-509, 1993.',
    'D. S. Kerby, "The simple difference formula: an approach to teaching nonparametric correlation", '
    'Comprehensive Psychology 3, 11.IT.3.1, 2014.',
    'S. Holm, "A simple sequentially rejective multiple test procedure", Scandinavian Journal of Statistics 6(2), '
    '65-70, 1979.',
    'Y. Benjamini and Y. Hochberg, "Controlling the false discovery rate: a practical and powerful approach to '
    'multiple testing", JRSS B 57(1), 289-300, 1995.',
    'Y. Benjamini and D. Yekutieli, "The control of the false discovery rate in multiple testing under dependency", '
    'Annals of Statistics 29(4), 1165-1188, 2001.',
    'J. L. Fleiss, B. Levin and M. C. Paik, Statistical Methods for Rates and Proportions, 3rd ed., Wiley, 2003 '
    '(sample size for two proportions with the continuity correction).',
    'L. Kish, Survey Sampling, Wiley, 1965 (the design effect 1 + (m - 1) rho).',
    'J. M. Hoenig and D. M. Heisey, "The abuse of power: the pervasive fallacy of power calculations for data '
    'analysis", The American Statistician 55(1), 19-24, 2001.',
)


class StatsError(StudyError):
    """Data a statistic cannot be computed on; the message says why in plain words."""


# =========================================================================== helpers
def _sp():
    from scipy import special, stats
    return stats, special


def _conf(confidence) -> float:
    c = float(confidence)
    if not (0.5 <= c < 1.0):
        raise StatsError(f'The confidence level must be between 0.5 and 1 (0.95 is 95 %); it is {confidence!r}.')
    return c


def _alpha(alpha) -> float:
    a = float(alpha)
    if not (0.0 < a < 0.5):
        raise StatsError(f'The significance level alpha must be between 0 and 0.5 (0.05 is usual); it is {alpha!r}.')
    return a


def _target(power) -> float:
    p = float(power)
    if not (0.0 < p < 1.0):
        raise StatsError(f'The target power must be between 0 and 1 (0.8 is usual); it is {power!r}.')
    return p


def _counts(k, n, what='') -> tuple[int, int]:
    try:
        kf, nf = float(k), float(n)
    except (TypeError, ValueError):
        raise StatsError(f'{what or "A success count"} must be whole numbers; it is {k!r} of {n!r}.') from None
    if not (math.isfinite(kf) and math.isfinite(nf)) or kf != int(kf) or nf != int(nf):
        raise StatsError(f'{what or "A success count"} must be whole numbers; it is {k!r} of {n!r}.')
    k, n = int(kf), int(nf)
    if n < 1:
        raise StatsError(f'{what or "A condition"} needs at least one trial; it has {n}.')
    if not 0 <= k <= n:
        raise StatsError(f'{what or "A condition"} has {k} successes in {n} trials: successes must be between 0 and '
                         'the number of trials.')
    return k, n


def _vec(x, what='The values') -> np.ndarray:
    try:
        a = np.asarray(x, float).ravel()
    except (TypeError, ValueError):
        raise StatsError(f'{what} must be numbers.') from None
    if not np.isfinite(a).all():
        raise StatsError(f'{what} hold a value that is not a finite number (a figure not stated?): leave such runs '
                         'out first.')
    return a


def _pair(x, y) -> tuple[np.ndarray, np.ndarray]:
    x, y = _vec(x, 'The first sample'), _vec(y, 'The second sample')
    if len(x) != len(y):
        raise StatsError(f'Paired samples must have one value each per pair; they have {len(x)} and {len(y)}.')
    if len(x) < 2:
        raise StatsError('A paired comparison needs at least 2 pairs.')
    return x, y


def _two(x, y) -> tuple[np.ndarray, np.ndarray]:
    x, y = _vec(x, 'The first sample'), _vec(y, 'The second sample')
    if len(x) < 2 or len(y) < 2:
        raise StatsError(f'Each sample needs at least 2 values; they have {len(x)} and {len(y)}.')
    return x, y


def _f(v):
    """A JSON-able float (None for NaN or infinite)."""
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def _norm_ppf(q: float) -> float:
    return float(_sp()[1].ndtri(q))


def _norm_cdf(z) -> np.ndarray:
    return _sp()[1].ndtr(z)


def ranks(x) -> np.ndarray:
    """1-based ranks, ties given their average rank."""
    x = np.asarray(x, float)
    _u, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    avg = np.cumsum(cnt) - (cnt - 1) / 2.0
    return avg[inv.reshape(-1)]


def _tie_counts(x) -> np.ndarray:
    return np.unique(np.asarray(x, float), return_counts=True)[1].astype(float)


def _p_two(z: float) -> float:
    return float(min(1.0, 2.0 * _sp()[1].ndtr(-abs(z))))


def _pv(z, alternative):
    if alternative == 'less':
        return float(_sp()[1].ndtr(z))
    if alternative == 'greater':
        return float(_sp()[1].ndtr(-z))
    return _p_two(z)


ALTERNATIVES = ('two-sided', 'less', 'greater')


def _alt(a) -> str:
    if a not in ALTERNATIVES:
        raise StatsError(f'The alternative must be one of {", ".join(ALTERNATIVES)}; it is {a!r}.')
    return a


# =========================================================================== success rates
def wilson(k, n, confidence: float = CONFIDENCE) -> tuple[float, float]:
    """The Wilson score interval of k successes in n trials."""
    k, n = _counts(k, n)
    z = _norm_ppf(0.5 + _conf(confidence) / 2)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    lo, hi = max(0.0, mid - half), min(1.0, mid + half)
    if k == 0:
        lo = 0.0
    if k == n:
        hi = 1.0
    return lo, hi


def clopper_pearson(k, n, confidence: float = CONFIDENCE) -> tuple[float, float]:
    """The Clopper-Pearson interval: the beta quantiles B(α/2; k, n − k + 1) and B(1 − α/2; k + 1, n − k)."""
    k, n = _counts(k, n)
    a = 1 - _conf(confidence)
    st, _ = _sp()
    lo = 0.0 if k == 0 else float(st.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(st.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def proportion(k, n, confidence: float = CONFIDENCE, name: str = '') -> dict:
    """A success rate with both intervals."""
    k, n = _counts(k, n, f'"{name}"' if name else '')
    return {'name': name, 'successes': k, 'trials': n, 'rate': k / n, 'confidence': confidence,
            'wilson': list(wilson(k, n, confidence)), 'clopper_pearson': list(clopper_pearson(k, n, confidence))}


def risk_difference(k1, n1, k2, n2, confidence: float = CONFIDENCE) -> dict:
    """p2 − p1 with Newcombe's hybrid score interval (method 10): from the two Wilson intervals (l_i, u_i),
    [d − √((p2 − l2)² + (u1 − p1)²), d + √((u2 − p2)² + (p1 − l1)²)]."""
    k1, n1 = _counts(k1, n1)
    k2, n2 = _counts(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, confidence)
    l2, u2 = wilson(k2, n2, confidence)
    d = p2 - p1
    lo = d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)
    hi = d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
    return {'value': d, 'ci': [max(-1.0, lo), min(1.0, hi)], 'confidence': confidence,
            'method': 'Newcombe hybrid score interval (method 10, from the Wilson intervals)'}


def odds_ratio(k1, n1, k2, n2, confidence: float = CONFIDENCE) -> dict:
    """The sample odds ratio of condition 2 against condition 1, (k2 (n1 − k1)) / (k1 (n2 − k2)), with Woolf's
    interval exp(ln OR ± z √(Σ 1/cell)); with a zero cell ½ is added to every cell (Haldane-Anscombe), said."""
    k1, n1 = _counts(k1, n1)
    k2, n2 = _counts(k2, n2)
    a, b, c, d = float(k2), float(n2 - k2), float(k1), float(n1 - k1)
    corrected = min(a, b, c, d) == 0
    if corrected:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    orr = (a * d) / (b * c)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z = _norm_ppf(0.5 + _conf(confidence) / 2)
    return {'value': orr, 'ci': [math.exp(math.log(orr) - z * se), math.exp(math.log(orr) + z * se)],
            'confidence': confidence, 'corrected': corrected,
            'method': 'sample odds ratio, Woolf logit interval' + (' (½ added to every cell: a cell is zero)'
                                                                   if corrected else '')}


def _hyper_logpmf(x, n1, n2, m):
    """log P(X = x) of the hypergeometric: x successes of m in group 1 (n1), the rest in group 2 (n2)."""
    from scipy.special import gammaln

    def lc(a, b):
        return gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)
    x = np.asarray(x, float)
    return lc(n1, x) + lc(n2, m - x) - lc(n1 + n2, m)


def fisher_exact(k1, n1, k2, n2, alternative: str = 'two-sided') -> dict:
    """Fisher's exact test of the 2 × 2 table (k1, n1 − k1; k2, n2 − k2), conditional on the total successes.
    two-sided: the probability of the tables no more likely than the observed (relative tolerance 1e-7).
    alternative, as everywhere in this module, is about condition 2 against condition 1 (B against A, p2 − p1, the
    direction of two_proportion_z, risk_difference and odds_ratio): 'less' p2 < p1, 'greater' p2 > p1. (SciPy's
    fisher_exact on the table [[k1, n1 − k1], [k2, n2 − k2]] names the other direction: its 'less' is p1 < p2.)"""
    k1, n1 = _counts(k1, n1)
    k2, n2 = _counts(k2, n2)
    alternative = _alt(alternative)
    m = k1 + k2
    lo, hi = max(0, m - n2), min(n1, m)
    xs = np.arange(lo, hi + 1)
    pm = np.exp(_hyper_logpmf(xs, n1, n2, m))
    pm = pm / pm.sum()
    i = k1 - lo
    if alternative == 'less':           # p2 < p1: condition 1 has at least as many successes as observed
        p = pm[i:].sum()
    elif alternative == 'greater':      # p2 > p1: condition 1 has at most as many successes as observed
        p = pm[:i + 1].sum()
    else:
        p = pm[pm <= pm[i] * FISHER_RTOL].sum()
    return {'test': 'fisher', 'p': float(min(1.0, p)), 'alternative': alternative}


def two_proportion_z(k1, n1, k2, n2, continuity: bool = False, alternative: str = 'two-sided') -> dict:
    """The pooled two-proportion z test of p2 − p1 (with Yates' correction ½(1/n1 + 1/n2) when continuity)."""
    k1, n1 = _counts(k1, n1)
    k2, n2 = _counts(k2, n2)
    alternative = _alt(alternative)
    p1, p2 = k1 / n1, k2 / n2
    pp = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    d = p2 - p1
    if se == 0:
        return {'test': 'z', 'z': None, 'p': 1.0, 'alternative': alternative, 'continuity': continuity}
    num = d
    if continuity:
        cc = 0.5 * (1 / n1 + 1 / n2)
        num = math.copysign(max(abs(d) - cc, 0.0), d) if alternative == 'two-sided' else \
            (d - cc if alternative == 'greater' else d + cc)
    z = num / se
    return {'test': 'z', 'z': z, 'p': _pv(z, alternative), 'alternative': alternative, 'continuity': continuity}


RATE_TEST_WHY = {
    'fisher': ('Fisher\'s exact test: exact at any number of episodes (it conditions on the total successes), so it '
               'holds for the small evaluations robot policies get; it is conservative (its true false-positive rate '
               'is at most alpha).'),
    'z': ('The two-proportion z test (pooled): a normal approximation, fair when every expected count (successes '
          'and failures of each condition under no difference) is at least 5.'),
}
RATE_ASSUMPTIONS = ('each episode is an independent trial (no learning between episodes, no shared scene that makes '
                    'two episodes fail together)',
                    'one success probability per condition over its episodes (no drift)',
                    'the two conditions\' episodes are independent of each other (different episodes, not the same '
                    'episode scored twice)')


def compare_rates(k1, n1, k2, n2, test: str = 'fisher', confidence: float = CONFIDENCE, names=('A', 'B'),
                  alternative: str = 'two-sided') -> dict:
    """Two success rates: each rate with its intervals, the test (with why), the risk difference (Newcombe) and the
    odds ratio (Woolf); the z test's expected counts are checked and a small one is said."""
    if test not in RATE_TESTS:
        raise StatsError(f'The test for success rates must be fisher or z; it is {test!r}.')
    a = proportion(k1, n1, confidence, names[0])
    b = proportion(k2, n2, confidence, names[1])
    res = fisher_exact(k1, n1, k2, n2, alternative) if test == 'fisher' else \
        two_proportion_z(k1, n1, k2, n2, False, alternative)
    pp = (a['successes'] + b['successes']) / (a['trials'] + b['trials'])
    expected = min(a['trials'] * pp, a['trials'] * (1 - pp), b['trials'] * pp, b['trials'] * (1 - pp))
    notes = []
    if test == 'z' and expected < 5:
        notes.append(f'The smallest expected count is {expected:.1f} (under 5): the z test\'s normal approximation is '
                     'poor here; Fisher\'s exact test is the safer choice.')
    rd = risk_difference(k1, n1, k2, n2, confidence)
    if (rd['ci'][0] > 0 or rd['ci'][1] < 0) and res['p'] > 1 - confidence:
        notes.append(f'The difference\'s interval excludes zero while the test does not reject at '
                     f'{1 - confidence:.2g}: the evidence is borderline (Fisher\'s exact test is conservative; the '
                     'interval and the test are different methods).')
    return {'kind': 'rates', 'a': a, 'b': b, 'test': test, 'names': [a['name'] or 'A', b['name'] or 'B'],
            'confidence': confidence,
            'test_label': 'Fisher\'s exact test' if test == 'fisher' else 'two-proportion z test (pooled)',
            'why': RATE_TEST_WHY[test], 'p': res['p'], 'z': res.get('z'), 'alternative': alternative,
            'difference': rd,
            'odds_ratio': odds_ratio(k1, n1, k2, n2, confidence), 'min_expected': expected,
            'assumptions': list(RATE_ASSUMPTIONS), 'notes': notes}


# =========================================================================== bootstrap
STATISTICS = ('mean', 'median')


def _stat_fn(statistic):
    if callable(statistic):
        return statistic
    if statistic == 'mean':
        return lambda a, axis=-1: np.mean(a, axis=axis)
    if statistic == 'median':
        return lambda a, axis=-1: np.median(a, axis=axis)
    raise StatsError(f'The statistic must be one of {", ".join(STATISTICS)}; it is {statistic!r}.')


def _boot_theta(f, data, B, rng, paired, batch=1000):
    """The B bootstrap replicates of the statistic. data: (x,) one sample, (d,) paired differences, (x, y) two
    samples resampled independently (theta = f(y) − f(x))."""
    out = np.empty(B)
    done = 0
    while done < B:
        k = min(batch, B - done)
        if len(data) == 1:
            a = data[0]
            idx = rng.integers(0, len(a), (k, len(a)))
            out[done:done + k] = f(a[idx], axis=-1)
        else:
            x, y = data
            ix = rng.integers(0, len(x), (k, len(x)))
            iy = rng.integers(0, len(y), (k, len(y)))
            out[done:done + k] = f(y[iy], axis=-1) - f(x[ix], axis=-1)
        done += k
    return out


def _theta(f, data):
    if len(data) == 1:
        return float(f(data[0], axis=-1))
    return float(f(data[1], axis=-1) - f(data[0], axis=-1))


def _jackknife_acc(f, data) -> float:
    """BCa's acceleration from the jackknife, each sample left out in turn (Efron and Tibshirani 1993, eq. 14.15; for
    two samples the sums run over both samples' jackknife values, as SciPy does)."""
    nums, dens = 0.0, 0.0
    for j in range(len(data)):
        s = data[j]
        n = len(s)
        if n < 2:
            continue
        keep = ~np.eye(n, dtype=bool)
        loo = s[np.newaxis, :].repeat(n, 0)[keep].reshape(n, n - 1)
        if len(data) == 1:
            th = f(loo, axis=-1)
        elif j == 0:
            th = f(data[1], axis=-1) - f(loo, axis=-1)
        else:
            th = f(loo, axis=-1) - f(data[0], axis=-1)
        u = (n - 1) * (th.mean() - th)
        nums += float(np.sum(u ** 3)) / n ** 3
        dens += float(np.sum(u ** 2)) / n ** 2
    if dens <= 0:
        return 0.0
    return nums / (6.0 * dens ** 1.5)


def bootstrap_ci(x, y=None, paired: bool = False, statistic='mean', method: str = 'bca',
                 n_resamples: int = N_BOOT, confidence: float = CONFIDENCE, seed: int = SEED) -> dict:
    """A bootstrap interval of the statistic of x (y None), of the statistic of the paired differences y − x
    (paired; for 'mean' that is the mean difference), or of statistic(y) − statistic(x) (each sample resampled on
    its own). method 'percentile' or 'bca'. Returns {'value', 'ci', 'method', 'n_resamples', 'seed', 'se'}."""
    if method not in ('percentile', 'bca'):
        raise StatsError(f'The bootstrap method must be percentile or bca; it is {method!r}.')
    conf = _conf(confidence)
    B = int(n_resamples)
    if B < 100:
        raise StatsError(f'The bootstrap needs at least 100 resamples; it is set to {B}.')
    f = _stat_fn(statistic)
    if y is None:
        data = (_vec(x),)
        if len(data[0]) < 2:
            raise StatsError('The bootstrap needs at least 2 values.')
        what = 'one sample'
    elif paired:
        xa, ya = _pair(x, y)
        data = (ya - xa,)
        what = 'paired differences (pairs resampled together)'
    else:
        data = _two(x, y)
        what = 'two samples (each resampled on its own)'
    theta = _theta(f, data)
    rng = np.random.default_rng(int(seed))
    tb = _boot_theta(f, data, B, rng, paired)
    a = 1 - conf
    lo_q, hi_q = a / 2, 1 - a / 2
    note = ''
    if np.ptp(tb) == 0:
        return {'value': theta, 'ci': [theta, theta], 'method': method, 'n_resamples': B, 'seed': int(seed),
                'se': 0.0, 'confidence': conf, 'resampled': what,
                'note': 'Every resample gives the same value (the data do not vary): the interval is a point.'}
    if method == 'bca':
        pct = (np.count_nonzero(tb < theta) + np.count_nonzero(tb <= theta)) / (2.0 * B)
        if pct <= 0 or pct >= 1:
            note = 'BCa is undefined here (the estimate lies outside every resample): the percentile interval is used.'
            method = 'percentile'
        else:
            z0 = _norm_ppf(pct)
            acc = _jackknife_acc(f, data)
            za, zb = _norm_ppf(lo_q), _norm_ppf(hi_q)
            lo_q = float(_norm_cdf(z0 + (z0 + za) / (1 - acc * (z0 + za))))
            hi_q = float(_norm_cdf(z0 + (z0 + zb) / (1 - acc * (z0 + zb))))
    lo, hi = np.quantile(tb, [lo_q, hi_q])
    out = {'value': theta, 'ci': [float(lo), float(hi)], 'method': method, 'n_resamples': B, 'seed': int(seed),
           'se': float(tb.std(ddof=1)), 'confidence': conf, 'resampled': what,
           'statistic': statistic if isinstance(statistic, str) else getattr(statistic, '__name__', 'custom')}
    if note:
        out['note'] = note
    return out


# =========================================================================== permutation tests
def _perm_p(null, obs, alternative, exact, B):
    """SciPy's convention: counts within a relative 100 ε of the observed count as ties; Monte Carlo adds one to
    numerator and denominator (Phipson and Smyth 2010); two-sided is twice the smaller tail."""
    gamma = abs(np.finfo(float).eps * 100 * obs)
    adj = 0 if exact else 1
    n = len(null)

    def less():
        return (np.count_nonzero(null <= obs + gamma) + adj) / (n + adj)

    def greater():
        return (np.count_nonzero(null >= obs - gamma) + adj) / (n + adj)
    if alternative == 'less':
        p = less()
    elif alternative == 'greater':
        p = greater()
    else:
        p = 2 * min(less(), greater())
    return float(min(1.0, max(0.0, p)))


def permutation_test(x, y, paired: bool = False, statistic='mean', alternative: str = 'two-sided',
                     n_resamples: int = N_PERM, seed: int = SEED) -> dict:
    """The permutation test of statistic(y) − statistic(x) (paired: statistic of y − x, the null flips the sign of
    each difference; unpaired: the null relabels the pooled values). Exact when there are at most n_resamples
    rearrangements (2^n sign flips, C(n1 + n2, n1) relabellings), else n_resamples random ones."""
    alternative = _alt(alternative)
    f = _stat_fn(statistic)
    B = int(n_resamples)
    if B < 99:
        raise StatsError(f'The permutation test needs at least 99 rearrangements; it is set to {B}.')
    rng = np.random.default_rng(int(seed))
    if paired:
        xa, ya = _pair(x, y)
        d = ya - xa
        n = len(d)
        obs = float(f(d, axis=-1))
        total = 2 ** n if n < 63 else math.inf
        exact = total <= B
        if exact:
            signs = 1 - 2 * ((np.arange(int(total))[:, None] >> np.arange(n)[None, :]) & 1)
            null = f(signs * d[None, :], axis=-1)
        else:
            null = np.empty(B)
            for s in range(0, B, 2000):
                k = min(2000, B - s)
                signs = rng.choice((-1.0, 1.0), (k, n))
                null[s:s + k] = f(signs * d[None, :], axis=-1)
        kind = 'sign flips of the paired differences'
        count = total
    else:
        xa, ya = _two(x, y)
        n1, n2 = len(xa), len(ya)
        z = np.r_[xa, ya]
        obs = float(f(ya, axis=-1) - f(xa, axis=-1))
        total = math.comb(n1 + n2, n1)
        exact = total <= B
        if exact:
            null = np.empty(total)
            allidx = np.arange(n1 + n2)
            for i, comb in enumerate(itertools.combinations(range(n1 + n2), n1)):
                mask = np.zeros(n1 + n2, bool)
                mask[list(comb)] = True
                null[i] = f(z[allidx[~mask]], axis=-1) - f(z[mask], axis=-1)
        else:
            null = np.empty(B)
            for s in range(0, B, 1000):
                k = min(1000, B - s)
                perm = rng.permuted(np.tile(z, (k, 1)), axis=1)
                null[s:s + k] = f(perm[:, n1:], axis=-1) - f(perm[:, :n1], axis=-1)
        kind = 'relabellings of the pooled values'
        count = total
    p = _perm_p(np.asarray(null, float), obs, alternative, exact, B)
    return {'test': 'permutation', 'paired': paired, 'statistic': obs, 'p': p, 'alternative': alternative,
            'exact': bool(exact), 'rearrangements': int(count) if math.isfinite(count) else None,
            'n_resamples': int(len(null)), 'seed': None if exact else int(seed),
            'method': (f'exact: all {int(count):,} {kind}' if exact else
                       f'Monte Carlo: {B:,} random {kind}, p = (b + 1)/(B + 1)')}


# =========================================================================== t tests
def welch_t(x, y, alternative: str = 'two-sided', confidence: float = CONFIDENCE) -> dict:
    """Welch's t test of mean(y) − mean(x) with the Welch-Satterthwaite degrees of freedom and the difference's
    t interval."""
    alternative = _alt(alternative)
    x, y = _two(x, y)
    st, _ = _sp()
    n1, n2 = len(x), len(y)
    v1, v2 = x.var(ddof=1) / n1, y.var(ddof=1) / n2
    diff = float(y.mean() - x.mean())
    se = math.sqrt(v1 + v2)
    if se == 0:
        return {'test': 'welch_t', 't': None, 'df': None, 'p': 1.0 if diff == 0 else 0.0, 'difference': diff,
                'ci': [diff, diff], 'alternative': alternative, 'note': 'Both samples are constant.'}
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    t = diff / se
    p = _t_p(t, df, alternative)
    q = float(st.t.ppf(0.5 + _conf(confidence) / 2, df))
    return {'test': 'welch_t', 't': t, 'df': float(df), 'p': p, 'difference': diff, 'se': se,
            'ci': [diff - q * se, diff + q * se], 'alternative': alternative, 'confidence': confidence}


def _t_p(t, df, alternative):
    st, _ = _sp()
    if alternative == 'less':
        return float(st.t.cdf(t, df))
    if alternative == 'greater':
        return float(st.t.sf(t, df))
    return float(min(1.0, 2 * st.t.sf(abs(t), df)))


def paired_t(x, y, alternative: str = 'two-sided', confidence: float = CONFIDENCE) -> dict:
    """The paired t test of the mean of y − x, with its t interval."""
    alternative = _alt(alternative)
    x, y = _pair(x, y)
    st, _ = _sp()
    d = y - x
    n = len(d)
    diff = float(d.mean())
    se = float(d.std(ddof=1)) / math.sqrt(n)
    if se == 0:
        return {'test': 'paired_t', 't': None, 'df': n - 1, 'p': 1.0 if diff == 0 else 0.0, 'difference': diff,
                'ci': [diff, diff], 'alternative': alternative, 'note': 'Every pair differs by the same amount.'}
    t = diff / se
    q = float(st.t.ppf(0.5 + _conf(confidence) / 2, n - 1))
    return {'test': 'paired_t', 't': t, 'df': n - 1, 'p': _t_p(t, n - 1, alternative), 'difference': diff, 'se': se,
            'ci': [diff - q * se, diff + q * se], 'alternative': alternative, 'confidence': confidence}


# =========================================================================== rank tests
def _mwu_counts(n1: int, n2: int) -> np.ndarray:
    """The number of arrangements giving each U = 0 … n1·n2 (the Mann-Whitney recurrence
    f(i, j, u) = f(i − 1, j, u − j) + f(i, j − 1, u)), as floats."""
    if n1 > n2:
        n1, n2 = n2, n1
    prev = [np.ones(1)] * (n2 + 1)          # i = 0: one arrangement, U = 0, for every j
    for i in range(1, n1 + 1):
        cur = [np.ones(1)]                  # j = 0
        for j in range(1, n2 + 1):
            a = np.zeros(i * j + 1)
            p = prev[j]                     # f(i − 1, j, ·), shifted by j
            a[j:j + len(p)] += p
            c = cur[j - 1]                  # f(i, j − 1, ·)
            a[:len(c)] += c
            cur.append(a)
        prev = cur
    return prev[n2]


def mann_whitney(x, y, alternative: str = 'two-sided', method: str = 'auto') -> dict:
    """The Mann-Whitney U test (U of y, U_y = R_y − n_y(n_y + 1)/2, large when y tends to be larger). method 'auto':
    exact when one sample has at most 8 values and there are no ties, else the normal approximation with the tie
    correction and a continuity correction of ½."""
    alternative = _alt(alternative)
    x, y = _two(x, y)
    n1, n2 = len(x), len(y)
    r = ranks(np.r_[x, y])
    u_y = float(r[n1:].sum() - n2 * (n2 + 1) / 2)
    u_x = n1 * n2 - u_y
    ties = len(np.unique(np.r_[x, y])) < n1 + n2
    if method == 'auto':
        method = 'exact' if (min(n1, n2) <= 8 and not ties) else 'asymptotic'
    if method not in ('exact', 'asymptotic'):
        raise StatsError(f'The Mann-Whitney method must be auto, exact or asymptotic; it is {method!r}.')
    out = {'test': 'mann_whitney', 'U': u_y, 'U_x': u_x, 'method': method, 'alternative': alternative}
    if method == 'exact':
        c = _mwu_counts(n1, n2)
        pmf = c / c.sum()
        sf = lambda u: float(pmf[int(math.ceil(u - 1e-9)):].sum())        # noqa: E731 — P(U ≥ u)
        if alternative == 'greater':
            p = sf(u_y)
        elif alternative == 'less':
            p = sf(u_x)
        else:
            p = 2 * sf(max(u_y, u_x))
    else:
        t = _tie_counts(np.r_[x, y])
        n = n1 + n2
        s = math.sqrt(n1 * n2 / 12 * ((n + 1) - float(np.sum(t ** 3 - t)) / (n * (n - 1))))
        num = u_y - n1 * n2 / 2
        if s == 0:
            return {**out, 'p': 1.0, 'z': None}
        if alternative == 'two-sided':
            num -= 0.5 * np.sign(num)
        elif alternative == 'greater':
            num -= 0.5
        else:
            num += 0.5
        z = num / s
        out['z'] = float(z)
        p = _pv(z, alternative)
    out['p'] = float(min(1.0, max(0.0, p)))
    return out


def _wilcoxon_counts(n: int) -> np.ndarray:
    """The number of sign assignments giving each T⁺ = 0 … n(n + 1)/2 for ranks 1 … n (no ties)."""
    c = np.zeros(n * (n + 1) // 2 + 1)
    c[0] = 1.0
    top = 0
    for k in range(1, n + 1):
        c[k:top + k + 1] += c[:top + 1].copy()
        top += k
    return c


def wilcoxon_signed_rank(x, y, alternative: str = 'two-sided', method: str = 'auto') -> dict:
    """The Wilcoxon signed-rank test of d = y − x (zero differences dropped: Wilcoxon's rule). T⁺ is the sum of the
    ranks of |d| over the positive d. method 'auto': exact without ties or zeros up to 50 pairs; with ties or zeros
    up to 13 pairs, all 2^n sign flips of the observed differences; else the normal approximation with the tie
    correction (no continuity correction)."""
    alternative = _alt(alternative)
    x, y = _pair(x, y)
    d = y - x
    n_all = len(d)
    zeros = int(np.count_nonzero(d == 0))
    dz = d[d != 0]
    n = len(dz)
    out = {'test': 'wilcoxon', 'n_pairs': n_all, 'zeros': zeros, 'n_used': n, 'alternative': alternative}
    if n == 0:
        return {**out, 'T_plus': 0.0, 'T_minus': 0.0, 'p': 1.0, 'method': 'none',
                'note': 'Every pair has a zero difference.'}
    r = ranks(np.abs(dz))
    tp, tm = float(r[dz > 0].sum()), float(r[dz < 0].sum())
    out.update(T_plus=tp, T_minus=tm)
    ties = len(np.unique(np.abs(dz))) < n
    if method == 'auto':
        if n_all > 50:
            method = 'asymptotic'
        elif not ties and zeros == 0:
            method = 'exact'
        elif n_all <= 13:
            method = 'permutation'
        else:
            method = 'asymptotic'
    out['method'] = method
    if method == 'exact':
        c = _wilcoxon_counts(n)
        pmf = c / c.sum()
        cdf = lambda t: float(pmf[:int(math.ceil(t)) + 1].sum())           # noqa: E731
        sf = lambda t: float(pmf[int(math.floor(t)):].sum())               # noqa: E731
        p = cdf(tp) if alternative == 'less' else sf(tp) if alternative == 'greater' else 2 * min(sf(tp), cdf(tp))
    elif method == 'permutation':
        # every sign flip of all n_all differences (zeros stay zero and are left out of T⁺)
        rr = np.zeros(n_all)
        rr[d != 0] = r
        signs = 1 - 2 * ((np.arange(2 ** n_all)[:, None] >> np.arange(n_all)[None, :]) & 1)
        sd = signs * d[None, :]
        null = ((sd > 0) * rr[None, :]).sum(axis=1)
        p = _perm_p(null.astype(float), tp, alternative, True, len(null))
    elif method == 'asymptotic':
        mn = n * (n + 1) / 4
        t = _tie_counts(np.abs(dz))
        se = math.sqrt((n * (n + 1) * (2 * n + 1) - float(np.sum(t ** 3 - t)) / 2) / 24)
        z = (tp - mn) / se if se > 0 else 0.0
        out['z'] = float(z)
        p = _pv(z, alternative)
    else:
        raise StatsError(f'The Wilcoxon method must be auto, exact, permutation or asymptotic; it is {method!r}.')
    out['p'] = float(min(1.0, max(0.0, p)))
    return out


# =========================================================================== effect sizes
def _hedges_j(df: float) -> float:
    """Hedges' exact small-sample factor J(df) = Γ(df/2) / (√(df/2) Γ((df − 1)/2))."""
    from scipy.special import gammaln
    if df <= 1:
        return float('nan')
    return math.exp(gammaln(df / 2) - 0.5 * math.log(df / 2) - gammaln((df - 1) / 2))


def _nct_interval(t_obs: float, df: float, conf: float) -> tuple[float, float] | None:
    """The noncentrality parameters λ_L, λ_U with P(T ≤ t_obs | λ_L) = 1 − α/2 and P(T ≤ t_obs | λ_U) = α/2."""
    from scipy.optimize import brentq
    st, _ = _sp()
    a = 1 - conf

    def solve(target):
        def fn(lam):                                    # falls with λ; SciPy's nct gives NaN far in a tail
            v = float(st.nct.cdf(t_obs, df, lam))
            if not math.isfinite(v):
                v = 1.0 if lam < t_obs else 0.0
            return v - target
        lo, hi = t_obs - 10.0, t_obs + 10.0
        k = 0
        while fn(lo) < 0 and k < 40:
            lo -= 10.0 * (k + 1)
            k += 1
        k = 0
        while fn(hi) > 0 and k < 40:
            hi += 10.0 * (k + 1)
            k += 1
        if fn(lo) < 0 or fn(hi) > 0:
            return None
        return brentq(fn, lo, hi, xtol=1e-10)
    lo, hi = solve(1 - a / 2), solve(a / 2)
    if lo is None or hi is None:
        return None
    return lo, hi


def cohens_d(x, y, paired: bool = False, confidence: float = CONFIDENCE) -> dict:
    """Cohen's d of y against x and Hedges' g = J·d, with intervals from the noncentral t.
    unpaired: d = (mean y − mean x) / s_pooled, t = d / √(1/n1 + 1/n2), df = n1 + n2 − 2;
    paired:   d_z = mean(y − x) / sd(y − x), t = d_z √n, df = n − 1 (d_z is the paired design's own scale: it is not
              comparable with an unpaired d)."""
    conf = _conf(confidence)
    if paired:
        x, y = _pair(x, y)
        dd = y - x
        n = len(dd)
        sd = float(dd.std(ddof=1))
        if sd == 0:
            return {'kind': 'd_z', 'd': None, 'ci': None, 'g': None, 'g_ci': None,
                    'note': 'Every pair differs by the same amount, so the standardised effect is undefined.'}
        d = float(dd.mean()) / sd
        df = n - 1
        scale = 1 / math.sqrt(n)
        kind = 'd_z'
    else:
        x, y = _two(x, y)
        n1, n2 = len(x), len(y)
        df = n1 + n2 - 2
        sp = math.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / df)
        if sp == 0:
            return {'kind': 'd', 'd': None, 'ci': None, 'g': None, 'g_ci': None,
                    'note': 'Both samples are constant, so the standardised effect is undefined.'}
        d = float(y.mean() - x.mean()) / sp
        scale = math.sqrt(1 / n1 + 1 / n2)
        kind = 'd'
    J = _hedges_j(df)
    iv = _nct_interval(d / scale, df, conf)
    ci = None if iv is None else [iv[0] * scale, iv[1] * scale]
    return {'kind': kind, 'd': d, 'ci': ci, 'g': J * d, 'g_ci': None if ci is None else [J * ci[0], J * ci[1]],
            'df': df, 'J': J, 'confidence': conf,
            'method': 'noncentral t interval (Steiger and Fouladi 1997); Hedges\' g = J·d with the exact J'}


def _cliff(x, y):
    """Cliff's delta P(Y > X) − P(Y < X) of rows of x (k, n1) and y (k, n2) through the ranks of the pooled values."""
    n1, n2 = x.shape[-1], y.shape[-1]
    z = np.concatenate([x, y], axis=-1)
    from scipy.stats import rankdata
    r = rankdata(z, axis=-1)
    u_y = r[..., n1:].sum(axis=-1) - n2 * (n2 + 1) / 2
    return 2 * u_y / (n1 * n2) - 1


def cliffs_delta(x, y, confidence: float = CONFIDENCE, n_resamples: int = 2000, seed: int = SEED) -> dict:
    """Cliff's delta of y against x (= the rank-biserial correlation 2U/(n1 n2) − 1, Kerby 2014): from −1 (every y
    below every x) to +1, with a seeded percentile bootstrap interval (each sample resampled on its own)."""
    x, y = _two(x, y)
    conf = _conf(confidence)
    d = float(_cliff(x[None, :], y[None, :])[0])
    rng = np.random.default_rng(int(seed))
    B = max(200, int(n_resamples))
    tb = np.empty(B)
    for s in range(0, B, 500):
        k = min(500, B - s)
        tb[s:s + k] = _cliff(x[rng.integers(0, len(x), (k, len(x)))], y[rng.integers(0, len(y), (k, len(y)))])
    a = 1 - conf
    lo, hi = np.quantile(tb, [a / 2, 1 - a / 2])
    return {'value': d, 'ci': [float(lo), float(hi)], 'confidence': conf, 'n_resamples': B, 'seed': int(seed),
            'method': 'Cliff\'s delta (the rank-biserial correlation), percentile bootstrap interval'}


def rank_biserial_paired(x, y) -> float | None:
    """The matched-pairs rank-biserial correlation (T⁺ − T⁻)/(T⁺ + T⁻) of y − x (zeros dropped); None without a
    non-zero difference."""
    x, y = _pair(x, y)
    d = y - x
    d = d[d != 0]
    if not len(d):
        return None
    r = ranks(np.abs(d))
    tp, tm = r[d > 0].sum(), r[d < 0].sum()
    return float((tp - tm) / (tp + tm))


# =========================================================================== multiple comparisons
def adjust(pvalues, method: str = CORRECTION, alpha: float = ALPHA) -> dict:
    """Adjusted p-values (each the smallest family level at which that hypothesis is rejected) and the rejections at
    alpha. bonferroni: min(1, m p); holm: step-down max_{j ≤ i} min(1, (m − j + 1) p_(j)); bh: step-up
    min_{j ≥ i} min(1, m p_(j) / j); by: bh times c(m) = Σ 1/i (any dependence); none: as given. None entries (no
    p-value) are left out of m and come back None."""
    if method not in CORRECTIONS:
        raise StatsError(f'The correction must be one of {", ".join(CORRECTIONS)}; it is {method!r}.')
    alpha = _alpha(alpha)
    vals = list(pvalues)
    idx = [i for i, p in enumerate(vals) if p is not None and math.isfinite(float(p))]
    p = np.array([float(vals[i]) for i in idx])
    if ((p < 0) | (p > 1)).any():
        raise StatsError('A p-value is outside 0 to 1.')
    m = len(p)
    adj = np.zeros(m)
    if m:
        o = np.argsort(p, kind='mergesort')
        ps = p[o]
        if method == 'none':
            a = ps.copy()
        elif method == 'bonferroni':
            a = np.minimum(1.0, m * ps)
        elif method == 'holm':
            a = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * ps))
        else:
            c = float(np.sum(1.0 / np.arange(1, m + 1))) if method == 'by' else 1.0
            a = np.minimum(1.0, np.minimum.accumulate((c * m * ps / np.arange(1, m + 1))[::-1])[::-1])
        adj[o] = a
    out = [None] * len(vals)
    rej = [None] * len(vals)
    for k, i in enumerate(idx):
        out[i] = float(adj[k])
        rej[i] = bool(adj[k] <= alpha)
    return {'method': method, 'label': CORRECTION_LABEL[method], 'alpha': alpha, 'm': m, 'adjusted': out,
            'reject': rej, 'controls': ('nothing (each test at alpha alone)' if method == 'none' else
                                        'the family-wise error rate' if method in ('holm', 'bonferroni') else
                                        'the false discovery rate' + (' under any dependence' if method == 'by' else
                                                                      ' (independent or positively dependent '
                                                                      'tests)'))}


# =========================================================================== power: two proportions
def _binom_pmf(n: int, p: float) -> np.ndarray:
    st, _ = _sp()
    return st.binom.pmf(np.arange(n + 1), n, p)


def _fisher_reject_masks(n1: int, n2: int, alpha: float) -> np.ndarray:
    """(n1 + 1, n2 + 1) booleans: Fisher's two-sided exact test rejects the table (x1, x2) at alpha."""
    R = np.zeros((n1 + 1, n2 + 1), bool)
    for m in range(n1 + n2 + 1):
        lo, hi = max(0, m - n2), min(n1, m)
        xs = np.arange(lo, hi + 1)
        pm = np.exp(_hyper_logpmf(xs, n1, n2, m))
        pm /= pm.sum()
        srt = np.sort(pm)
        cs = np.cumsum(srt)
        k = np.searchsorted(srt, pm * FISHER_RTOL, side='right')
        pv = cs[k - 1]
        rej = pv <= alpha
        R[xs[rej], m - xs[rej]] = True
    return R


def _z_reject_masks(n1: int, n2: int, alpha: float, continuity: bool) -> np.ndarray:
    x1 = np.arange(n1 + 1)[:, None] / n1
    x2 = np.arange(n2 + 1)[None, :] / n2
    pp = (x1 * n1 + x2 * n2) / (n1 + n2)
    se = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    d = np.abs(x2 - x1)
    if continuity:
        d = np.maximum(d - 0.5 * (1 / n1 + 1 / n2), 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.where(se > 0, d / se, 0.0)
    return z >= _norm_ppf(1 - alpha / 2)


def _rate_inputs(p1, p2):
    try:
        p1, p2 = float(p1), float(p2)
    except (TypeError, ValueError):
        raise StatsError('The success rates must be numbers between 0 and 1.') from None
    for p in (p1, p2):
        if not (0.0 <= p <= 1.0) or not math.isfinite(p):
            raise StatsError(f'A success rate must be between 0 and 1 (0.8 is 80 %); it is {p}.')
    return p1, p2


def power_two_proportions(p1, p2, n1: int, n2: int | None = None, alpha: float = ALPHA, method: str = 'normal',
                          test: str = 'fisher', continuity: bool = True, n_sim: int = SIM_DEFAULT,
                          seed: int = SEED) -> float:
    """The power of a two-sided test of p1 = p2 with n1 and n2 trials (n2 = n1 when None) when the true rates are p1
    and p2. normal: Φ((|Δ| √n_e − z_{α/2} √(2 p̄ q̄)) / √(p1 q1 + p2 q2)) with n_e the harmonic-mean-equivalent n
    and, with continuity, Fleiss's correction inverted (n_e from n through n = (n' − 1/|Δ|)²/n' with equal groups);
    arcsine: Φ(|h| √(n1 n2 / (n1 + n2)) − z_{α/2}); exact: every (x1, x2) enumerated, the test (Fisher's exact, or the
    z test with Yates' correction when continuity) applied; simulation: n_sim seeded draws of (x1, x2), that test
    applied to each."""
    p1, p2 = _rate_inputs(p1, p2)
    alpha = _alpha(alpha)
    n1 = int(n1)
    n2 = n1 if n2 is None else int(n2)
    if n1 < 1 or n2 < 1:
        raise StatsError('Each condition needs at least one trial.')
    if method not in POWER_METHODS:
        raise StatsError(f'The power method must be one of {", ".join(POWER_METHODS)}; it is {method!r}.')
    if test not in RATE_TESTS:
        raise StatsError(f'The test must be fisher or z; it is {test!r}.')
    za = _norm_ppf(1 - alpha / 2)
    delta = abs(p2 - p1)
    if method == 'arcsine':
        h = abs(2 * math.asin(math.sqrt(p2)) - 2 * math.asin(math.sqrt(p1)))
        z = h * math.sqrt(n1 * n2 / (n1 + n2)) - za
        return float(_norm_cdf(z) + _norm_cdf(-z - 2 * za))
    if method == 'normal':
        if delta == 0:
            return alpha
        ne = 2 * n1 * n2 / (n1 + n2)                       # equal-group equivalent n per group
        if continuity:
            ne = (ne - 1 / delta) ** 2 / ne if ne > 1 / delta else 0.0
        if ne <= 0:
            return alpha
        pb = (p1 + p2) / 2
        s1 = math.sqrt(2 * pb * (1 - pb))
        s2 = math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        if s2 == 0:
            return 1.0
        z = (delta * math.sqrt(ne) - za * s1) / s2
        z2 = (-delta * math.sqrt(ne) - za * s1) / s2
        return float(min(1.0, _norm_cdf(z) + _norm_cdf(z2)))
    if method == 'exact':
        if max(n1, n2) > EXACT_N_MAX:
            raise StatsError(f'Exact power enumerates every outcome; above {EXACT_N_MAX} trials per condition use '
                             'the normal approximation.')
        R = _fisher_reject_masks(n1, n2, alpha) if test == 'fisher' else _z_reject_masks(n1, n2, alpha, continuity)
        return float(_binom_pmf(n1, p1) @ R.astype(float) @ _binom_pmf(n2, p2))
    rng = np.random.default_rng(int(seed))
    ns = max(100, int(n_sim))
    x1 = rng.binomial(n1, p1, ns)
    x2 = rng.binomial(n2, p2, ns)
    R = _fisher_reject_masks(n1, n2, alpha) if test == 'fisher' else _z_reject_masks(n1, n2, alpha, continuity)
    return float(R[x1, x2].mean())


def _search_n(power_at, target: float, n0: int, n_max: int = N_MAX, monotone: bool = True) -> tuple[int, float]:
    """The smallest n ≥ 2 with power_at(n) ≥ target: bracketing from n0 by doubling, bisection, then (monotone False:
    exact tests, whose power saw-tooths with n) a scan of the 20 n below the bisection's answer."""
    n0 = max(2, min(int(n0), n_max))
    lo, hi = 1, None
    n = n0
    while True:
        if power_at(n) >= target:
            hi = n
            break
        lo = n
        if n >= n_max:
            raise StatsError(f'Even {n_max:,} per group do not reach the target power: the difference is too small '
                             'to tell apart.')
        n = min(n_max, n * 2)
    k = hi // 2
    while k > lo and power_at(k) >= target:      # bracket from below when n0 overshot
        hi, k = k, k // 2
    lo = max(lo, k)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if power_at(mid) >= target:
            hi = mid
        else:
            lo = mid
    if not monotone:
        for n in range(max(2, hi - 20), hi):
            if power_at(n) >= target:
                hi = n
                break
    return hi, power_at(hi)


TWO_PROPORTION_ASSUMPTIONS = (
    'every episode is an independent trial with the condition\'s success probability (no drift over the evaluation, '
    'no learning, no two episodes failing together because they share a scene or a reset)',
    'the same number of episodes per condition',
    'a two-sided test at alpha',
    'the rates p1 and p2 are the true rates to tell apart (a guess or a pilot\'s estimate: a pilot of a few episodes '
    'is itself uncertain, so plan for the smallest difference that matters)')


def n_two_proportions(p1, p2, alpha: float = ALPHA, power: float = POWER, method: str = 'normal',
                      test: str = 'fisher', continuity: bool = True, n_sim: int = SIM_DEFAULT,
                      seed: int = SEED) -> dict:
    """The smallest number of trials per condition (equal groups) whose power reaches the target."""
    p1, p2 = _rate_inputs(p1, p2)
    alpha, target = _alpha(alpha), _target(power)
    if p1 == p2:
        raise StatsError('The two success rates are equal: no number of episodes tells them apart. Enter the rates '
                         'you need to distinguish.')
    za, zb = _norm_ppf(1 - alpha / 2), _norm_ppf(target)
    pb = (p1 + p2) / 2
    delta = abs(p2 - p1)
    n_unc = (za * math.sqrt(2 * pb * (1 - pb)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / delta ** 2
    n_cc = n_unc / 4 * (1 + math.sqrt(1 + 4 / (n_unc * delta))) ** 2
    if method == 'normal':
        n0 = n_cc if continuity else n_unc
        n = int(math.ceil(n0 - 1e-9))
        pw = power_two_proportions(p1, p2, n, n, alpha, 'normal', test, continuity)
        formula = ('n = [z_{α/2} √(2 p̄ q̄) + z_β √(p1 q1 + p2 q2)]² / Δ²' +
                   (', then n\' = n/4 (1 + √(1 + 4/(n Δ)))² (Fleiss, Levin and Paik 2003)' if continuity else ''))
    elif method == 'arcsine':
        h = abs(2 * math.asin(math.sqrt(p2)) - 2 * math.asin(math.sqrt(p1)))
        n = int(math.ceil(2 * ((za + zb) / h) ** 2 - 1e-9))
        pw = power_two_proportions(p1, p2, n, n, alpha, 'arcsine')
        formula = 'n = 2 ((z_{α/2} + z_β) / h)², h = 2 asin √p2 − 2 asin √p1 (Cohen 1988)'
    else:
        def at(k):
            return power_two_proportions(p1, p2, k, k, alpha, method, test, continuity, n_sim, seed)
        if n_cc > EXACT_N_MAX and method == 'exact':
            raise StatsError(f'About {n_cc:,.0f} episodes per condition would be needed, above the {EXACT_N_MAX} the '
                             'exact enumeration handles: use the normal approximation.')
        n, pw = _search_n(at, target, int(math.ceil(n_cc)), EXACT_N_MAX if method == 'exact' else N_MAX,
                          monotone=False)
        formula = ('every outcome of both binomials enumerated and ' if method == 'exact' else
                   f'{max(100, int(n_sim)):,} seeded simulated evaluations, each tested with ') + \
            ('Fisher\'s exact test' if test == 'fisher' else 'the z test' + (' with Yates\' correction'
                                                                               if continuity else ''))
    out = {'kind': 'two_proportions', 'p1': p1, 'p2': p2, 'alpha': alpha, 'power': target, 'method': method,
           'test': test if method in ('exact', 'simulation') else ('z' if method == 'normal' else 'arcsine z'),
           'continuity': continuity if method != 'arcsine' else False, 'n_per_group': int(n),
           'total': 2 * int(n), 'power_at_n': float(pw), 'formula': formula,
           'assumptions': list(TWO_PROPORTION_ASSUMPTIONS)}
    if method in ('exact', 'simulation'):
        out['note'] = ('The power of an exact test does not rise smoothly with n (the discreteness of the counts), '
                       'so a slightly larger n can have slightly lower power; this is the smallest n that reaches '
                       'the target.')
    return out


# =========================================================================== power: t tests
def power_t(d, n: int, alpha: float = ALPHA, paired: bool = False, alternative: str = 'two-sided',
            n2: int | None = None) -> float:
    """The power of the t test for the standardised effect d: two-sample with n (and n2) per group, df n + n2 − 2,
    λ = d √(n n2 / (n + n2)); paired (or one sample) with n pairs, df n − 1, λ = d_z √n."""
    alpha = _alpha(alpha) if alternative == 'two-sided' else float(alpha)
    alternative = _alt(alternative)
    st, _ = _sp()
    d = float(d)
    n = int(n)
    if paired:
        if n < 2:
            return float('nan')
        df, lam = n - 1, d * math.sqrt(n)
    else:
        m = n if n2 is None else int(n2)
        if n < 2 or m < 2:
            return float('nan')
        df, lam = n + m - 2, d * math.sqrt(n * m / (n + m))
    two = alternative == 'two-sided'
    tc = float(st.t.ppf(1 - alpha / 2 if two else 1 - alpha, df))
    with np.errstate(all='ignore'):
        if two:
            pw = float(st.nct.sf(tc, df, lam) + st.nct.cdf(-tc, df, lam))
        elif alternative == 'greater':
            pw = float(st.nct.sf(tc, df, lam))
        else:
            pw = float(st.nct.cdf(-tc, df, lam))
    if not math.isfinite(pw):                 # SciPy's nct gives NaN far in a tail, where the power is 0 or 1
        pw = float(_norm_cdf(lam - tc) + (_norm_cdf(-lam - tc) if two else 0.0)) if alternative != 'less' else \
            float(_norm_cdf(-lam - tc))
    return min(1.0, max(0.0, pw))


T_ASSUMPTIONS = ('the per-unit values are independent and close to normal with one standard deviation in both '
                 'conditions (paired: the differences are)',
                 'the standardised effect d (difference / σ) is the effect to detect (a guess or a pilot\'s '
                 'estimate)')


def n_t(d, alpha: float = ALPHA, power: float = POWER, paired: bool = False,
        alternative: str = 'two-sided') -> dict:
    """The smallest n per group (paired: pairs) whose t-test power reaches the target."""
    target = _target(power)
    d = abs(float(d))
    if d == 0 or not math.isfinite(d):
        raise StatsError('The effect size is zero: no sample size detects it. Enter the smallest effect that matters.')
    za = _norm_ppf(1 - float(alpha) / (2 if alternative == 'two-sided' else 1))
    n0 = ((za + _norm_ppf(target)) / d) ** 2 * (1 if paired else 2)
    sign = d if alternative != 'less' else -d
    n, pw = _search_n(lambda k: power_t(sign, k, alpha, paired, alternative), target, int(math.ceil(n0)) + 1)
    return {'kind': 'paired_t' if paired else 't2', 'd': d, 'alpha': float(alpha), 'power': target,
            'alternative': alternative, 'n_per_group': int(n), 'total': int(n) if paired else 2 * int(n),
            'power_at_n': float(pw),
            'formula': ('noncentral t: df n − 1, λ = d_z √n' if paired else
                        'noncentral t: df 2n − 2, λ = d √(n/2)'),
            'assumptions': list(T_ASSUMPTIONS)}


def power_curve(kind: str, ns, alpha: float = ALPHA, **kw) -> dict:
    """Power at each n of ns. kind 'two_proportions' (p1, p2, method, test, continuity), 't2' or 'paired_t' (d)."""
    ns = [int(n) for n in ns if int(n) >= 2]
    out = []
    for n in ns:
        if kind == 'two_proportions':
            m = kw.get('method', 'normal')
            if m == 'exact' and n > EXACT_N_MAX:
                m = 'normal'
            out.append(power_two_proportions(kw['p1'], kw['p2'], n, n, alpha, m, kw.get('test', 'fisher'),
                                             kw.get('continuity', True), kw.get('n_sim', SIM_DEFAULT),
                                             kw.get('seed', SEED)))
        elif kind in ('t2', 'paired_t'):
            out.append(power_t(kw['d'], n, alpha, paired=kind == 'paired_t'))
        else:
            raise StatsError(f'Unknown power kind {kind!r}: two_proportions, t2 or paired_t.')
    return {'kind': kind, 'n': ns, 'power': [_f(p) for p in out]}


def curve_ns(n_needed: int, points: int = 40) -> list:
    """The n of a power curve: from 2 to about 2.5 times n_needed, evenly spaced integers."""
    top = max(10, int(2.5 * n_needed))
    return sorted(set(int(round(v)) for v in np.linspace(2, top, points)))


def episodes_needed(p1, p2, alpha: float = ALPHA, power: float = POWER, method: str = 'exact',
                    test: str = 'fisher', continuity: bool = True, n_sim: int = SIM_DEFAULT,
                    seed: int = SEED) -> dict:
    """Evaluation episodes per condition to tell success rates p1 and p2 apart with a two-sided test at alpha with
    the target power, with the assumptions spelled out and the curve of power against episodes. method 'exact'
    (default: Fisher's exact test, the test the Statistics page uses on success counts) falls back to the normal
    approximation when the enumeration would be too large (said)."""
    note = ''
    if method == 'exact':
        approx = n_two_proportions(p1, p2, alpha, power, 'normal', test, True)
        if approx['n_per_group'] > EXACT_N_MAX:
            method = 'normal'
            note = (f'About {approx["n_per_group"]:,} episodes per condition: above {EXACT_N_MAX} the exact '
                    'enumeration is replaced by the normal approximation with continuity correction.')
    r = n_two_proportions(p1, p2, alpha, power, method, test, continuity, n_sim, seed)
    ns = curve_ns(r['n_per_group'])
    r['curve'] = power_curve('two_proportions', ns, alpha, p1=r['p1'], p2=r['p2'], method=method, test=test,
                             continuity=continuity, n_sim=n_sim, seed=seed)
    r['question'] = (f'How many evaluation episodes per condition tell a success rate of {100 * r["p1"]:.0f} % from '
                     f'{100 * r["p2"]:.0f} % apart?')
    r['answer'] = (f'{r["n_per_group"]} episodes per condition ({r["total"]} in all) give '
                   f'{100 * r["power_at_n"]:.1f} % power at alpha {r["alpha"]:g} (two-sided).')
    if note:
        r['note'] = (note + ' ' + r.get('note', '')).strip()
    r['out_of_scope'] = OUT_OF_SCOPE
    return r


def design_effect(per_operator: float, icc: float) -> float:
    """Kish's design effect 1 + (m − 1) ρ of clusters of m units with intra-class correlation ρ."""
    m, rho = float(per_operator), float(icc)
    if m < 1 or not (0.0 <= rho <= 1.0):
        raise StatsError('The demonstrations per operator must be at least 1 and the intra-class correlation between '
                         '0 and 1.')
    return 1.0 + (m - 1.0) * rho


def demonstrations_needed(kind: str = 'continuous', d=None, difference=None, sd=None, p1=None, p2=None,
                          alpha: float = ALPHA, power: float = POWER, paired: bool = False, icc: float = 0.0,
                          per_operator: float = 1.0, attrition: float = 0.0, method: str = 'normal') -> dict:
    """Demonstrations to record per condition (two rig variants, two operators' setups, ...) to tell a figure
    measured on each demonstration apart.

    kind 'continuous': a per-demonstration value (completion time, a smoothness or quality score, a grasp error):
        the effect is d, or difference / sd; the t test's n (paired when every demonstration is recorded under both
        conditions, as a scene recorded twice).
    kind 'binary': a per-demonstration pass or fail (the audit passes it, the grasp succeeds): rates p1 and p2, the
        two-proportion n (method 'normal' or 'exact').
    Then the operators' clustering inflates n by Kish's design effect 1 + (m − 1) ρ (m demonstrations per operator,
    ρ the intra-class correlation of the figure between one operator's demonstrations; ρ = 0, the default, means
    independent demonstrations), and the share the audit discards (attrition, default 0) by 1 / (1 − attrition).

    This is not the number of demonstrations a policy needs to learn the task: that depends on the policy and the
    task and is not modelled here."""
    alpha, target = _alpha(alpha), _target(power)
    a = float(attrition)
    if not (0.0 <= a < 1.0):
        raise StatsError(f'The share of demonstrations discarded must be from 0 to below 1; it is {attrition!r}.')
    de = design_effect(per_operator, icc)
    if kind == 'continuous':
        if d is None:
            if difference is None or sd is None or float(sd) <= 0:
                raise StatsError('Give the effect as d, or as the difference to detect and the standard deviation '
                                 'of the figure between demonstrations.')
            d = float(difference) / float(sd)
        base = n_t(d, alpha, target, paired)
        what = ('pairs of demonstrations (one under each condition)' if paired else 'demonstrations per condition')
    elif kind == 'binary':
        base = n_two_proportions(p1, p2, alpha, target, method if method in ('normal', 'exact') else 'normal')
        what = 'demonstrations per condition'
    else:
        raise StatsError(f'kind must be continuous or binary; it is {kind!r}.')
    n0 = base['n_per_group']
    n = int(math.ceil(n0 * de / (1 - a) - 1e-9))
    assumptions = list(base['assumptions'])
    assumptions.append('operators: ' + ('demonstrations independent of who recorded them (ρ = 0, the default)'
                                        if icc == 0 or per_operator <= 1 else
                                        f'{per_operator:g} demonstrations per operator with intra-class correlation '
                                        f'ρ = {icc:g}: design effect {de:.2f} (Kish 1965, equal cluster sizes)'))
    assumptions.append(f'{100 * a:.0f} % of the demonstrations discarded by the audit' +
                       (' (the default)' if a == 0 else ''))
    return {'kind': f'demonstrations_{kind}', 'base': base, 'design_effect': de, 'attrition': a, 'icc': float(icc),
            'per_operator': float(per_operator), 'n_per_group': n, 'paired': bool(paired) and kind == 'continuous',
            'what': what, 'alpha': alpha, 'power': target, 'assumptions': assumptions,
            'answer': f'{n} {what} (from {n0} before the clustering and the discards).',
            'not_modelled': 'How many demonstrations a policy needs to learn the task is not modelled here.',
            'out_of_scope': OUT_OF_SCOPE}


def detectable_difference(kind: str, n: int, alpha: float = ALPHA, power: float = POWER, sd=None, p1=None,
                          paired: bool = False) -> dict | None:
    """The smallest difference n per group (paired: pairs) detects with the target power: for 't2' / 'paired_t' a
    standardised d (and, with sd, in the figure's unit); for 'two_proportions' the rate p2 above (or below) p1 the
    normal approximation detects. None when n is too small."""
    target = _target(power)
    from scipy.optimize import brentq
    n = int(n)
    if kind in ('t2', 'paired_t'):
        if n < 2 or (kind == 'paired_t' and n < 2):
            return None
        fn = lambda dd: power_t(dd, n, alpha, kind == 'paired_t') - target       # noqa: E731
        if fn(20.0) < 0:
            return None
        d = brentq(fn, 1e-9, 20.0, xtol=1e-9)
        out = {'kind': kind, 'n': n, 'd': d, 'power': target, 'alpha': alpha}
        if sd is not None and sd > 0:
            out['difference'] = d * float(sd)
        return out
    if kind == 'two_proportions':
        p1 = float(p1)
        res = {'kind': kind, 'n': n, 'p1': p1, 'power': target, 'alpha': alpha}
        for side, lim in (('up', 1.0), ('down', 0.0)):
            fn = lambda p2: power_two_proportions(p1, p2, n, n, alpha, 'normal') - target   # noqa: E731
            if abs(lim - p1) < 1e-9 or fn(lim) < 0:
                res[side] = None
                continue
            a, b = (p1 + 1e-9, lim) if side == 'up' else (lim, p1 - 1e-9)
            res[side] = brentq(fn, a, b, xtol=1e-9)
        return res
    return None


# =========================================================================== the comparison the app shows
def describe(x) -> dict:
    a = _vec(x)
    if not len(a):
        return {'n': 0, 'mean': None, 'sd': None, 'median': None, 'min': None, 'max': None}
    return {'n': int(len(a)), 'mean': float(a.mean()), 'sd': float(a.std(ddof=1)) if len(a) > 1 else None,
            'median': float(np.median(a)), 'min': float(a.min()), 'max': float(a.max())}


PAIRED_WHY = ('The values are paired, so the test compares each pair\'s difference: the permutation test flips the '
              'sign of each difference at random, which needs no assumption about the shape of the distribution (the '
              'figures over tolerance draws or runs are often skewed or pile up at a bound), only that, if the '
              'variants did not differ, each difference would be as likely positive as negative.')
UNPAIRED_WHY = ('The values are not paired, so the test relabels the pooled values at random: the permutation test '
                'of the difference of means needs no normality, only that, if the conditions did not differ, every '
                'value would be as likely to have come from either (it also reacts to a difference of spread).')


def compare_samples(x, y, better: str = 'higher', paired: bool = False, alpha: float = ALPHA,
                    confidence: float = CONFIDENCE, n_boot: int = N_BOOT, n_perm: int = N_PERM, seed: int = SEED,
                    names=('A', 'B'), pairing: str = '', power: float = POWER) -> dict:
    """The comparison of y (B) against x (A): the primary test (the permutation test of the mean difference, exact
    when small) with why, the mean difference with its BCa bootstrap interval, the standardised effect (paired d_z
    or Hedges' g) with its interval, Cliff's delta (unpaired) or the matched-pairs rank-biserial (paired), the
    reference tests (paired t and Wilcoxon, or Welch t and Mann-Whitney), the smallest difference the data detect
    with the target power, and 'text': a one-sentence verdict on the unadjusted p (adjust() across figures).
    better: 'higher' or 'lower' (which direction of the figure is better; core/tracking calls
    compare_samples(a, b, better))."""
    if better not in ('higher', 'lower'):
        better = 'higher'
    if paired:
        x, y = _pair(x, y)
    else:
        x, y = _two(x, y)
    alpha, conf = _alpha(alpha), _conf(confidence)
    perm = permutation_test(x, y, paired, 'mean', 'two-sided', n_perm, seed)
    boot = bootstrap_ci(x, y, paired, 'mean', 'bca', n_boot, conf, seed)
    eff = cohens_d(x, y, paired, conf)
    out = {'kind': 'samples', 'paired': bool(paired), 'pairing': pairing, 'names': list(names),
           'a': describe(x), 'b': describe(y), 'difference': boot, 'standardised': eff,
           'test': 'permutation', 'test_label': ('paired permutation test (sign flips)' if paired else
                                                 'permutation test (relabelling)'),
           'method': perm['method'], 'why': PAIRED_WHY if paired else UNPAIRED_WHY, 'p': perm['p'],
           'exact': perm['exact'], 'alpha': alpha, 'confidence': conf, 'better': better}
    if paired:
        dd = y - x
        out['differences'] = describe(dd)
        out['rank_biserial'] = rank_biserial_paired(x, y)
        out['reference_tests'] = [paired_t(x, y, confidence=conf), wilcoxon_signed_rank(x, y)]
        sd = float(dd.std(ddof=1))
        mde = detectable_difference('paired_t', len(dd), alpha, power, sd=sd if sd > 0 else None)
        out['assumptions'] = ['the values are paired' + (f' ({pairing})' if pairing else ''),
                              'the pairs are independent of each other',
                              'under no difference, each pair\'s difference is symmetric about zero (the sign flip)']
        if np.all(dd == 0):
            out['notes'] = ['Every pair has the same value under both: there is no difference to test.']
    else:
        out['cliffs_delta'] = cliffs_delta(x, y, conf, 2000, seed)
        out['reference_tests'] = [welch_t(x, y, confidence=conf), mann_whitney(x, y)]
        n = min(len(x), len(y))
        sp = math.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
        mde = detectable_difference('t2', n, alpha, power, sd=sp if sp > 0 else None)
        out['assumptions'] = ['the values are independent (no pairing between the conditions)',
                              'under no difference, the values of both conditions are exchangeable (the same '
                              'distribution)']
    out['detectable'] = mde
    out['text'] = verdict(out, out['p'], 'The figure', '', alpha)['text'].replace('adjusted p', 'p')
    return out


def verdict(cmp: dict, p_adj, metric_title: str = 'the figure', unit: str = '', alpha=None) -> dict:
    """A plain-English verdict of one comparison with its adjusted p-value: (status, word, text)."""
    alpha = cmp.get('alpha', ALPHA) if alpha is None else alpha
    names = cmp.get('names') or ['A', 'B']
    if cmp['kind'] == 'rates':
        d = cmp['difference']
        val, ci = 100 * d['value'], [100 * c for c in d['ci']]
        u = ' pp'
        better_b = d['value'] > 0
    else:
        d = cmp['difference']
        val, ci = d['value'], d['ci']
        u = f' {unit}' if unit else ''
        hi = cmp.get('better', 'higher') == 'higher'
        better_b = (val > 0) == hi
    ptxt = '—' if p_adj is None else (f'= {p_adj:.3g}' if p_adj >= 0.001 else '< 0.001')
    def sg(x, spec):                                  # a value that rounds to zero is written 0, never +0 or -0
        t = format(float(x), '+' + spec)
        return '0' if float(t) == 0 else t
    diff_txt = (f'{sg(val, ".1f")}{u} [{sg(ci[0], ".1f")}, {sg(ci[1], ".1f")}]' if cmp['kind'] == 'rates' else
                f'{sg(val, ".3g")}{u} [{sg(ci[0], ".3g")}, {sg(ci[1], ".3g")}]')
    if p_adj is None:
        return {'status': 'critical', 'word': 'no test', 'text': f'{metric_title}: no p-value could be computed.'}
    if p_adj <= alpha and val != 0:
        who = names[1] if better_b else names[0]
        return {'status': 'good', 'word': 'clear difference',
                'text': (f'{metric_title}: {who} is better ({names[1]} − {names[0]} = {diff_txt}, '
                         f'{int(round(100 * cmp.get("confidence", CONFIDENCE)))} % interval; adjusted p {ptxt}). '
                         'A difference this large is unlikely to be chance.')}
    det = cmp.get('detectable') or {}
    pw = f'{100 * det.get("power", POWER):.0f} %'
    if cmp['kind'] == 'rates':
        det_txt = ''
        if det.get('up') is not None or det.get('down') is not None:
            parts = [f'{w} {100 * v:.0f} %' for w, v in (('below', det.get('down')), ('above', det.get('up')))
                     if v is not None]
            det_txt = (f' With these episodes, only a true rate {" or ".join(parts)} (against '
                       f'{100 * cmp["a"]["rate"]:.0f} %) would be detected {pw} of the time.')
    else:
        dd = det.get('difference')
        det_txt = (f' These data detect a difference of about {dd:.3g}{u} ({pw} of the time); smaller ones may be '
                   'missed.' if dd is not None else '')
    return {'status': 'warning', 'word': 'no clear difference',
            'text': (f'{metric_title}: no clear difference ({names[1]} − {names[0]} = {diff_txt}; adjusted p '
                     f'{ptxt}). This does not show they are equal.' + det_txt)}
