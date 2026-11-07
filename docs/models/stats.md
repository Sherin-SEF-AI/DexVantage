# Statistics and power analysis

Research tools part 2. Code: `dexvantage/core/stats.py` (the maths: NumPy and SciPy, no project, no Qt),
`dexvantage/core/statcompare.py` (the comparison sources), `dexvantage/core/studies/stats.py` (the study `stats`,
batch flag `--stats-saved OUT.json`; a spec file with `--stats SPEC.json`, `dexvantage/batch_stats.py`),
`dexvantage/ui/pages/statistics.py` (Analysis ▸ Statistics), `dv.stats` (`dexvantage/api.py`), the assistant tool
`get_statistics` (`dexvantage/assistant/tools.py`). Checks: `dexvantage/validation/checks_stats.py` (model `stats`).
Metrics: `stats.p_adjusted` (headline), `stats.n_needed`, `stats.planned_power`.

The question it answers: is a difference between two rigs, two sets of runs or two policies' success rates larger
than chance, how large is it, and how many evaluation episodes, tolerance draws or demonstrations does it take to
tell two conditions apart? Every result names the test that ran and why, and spells out its assumptions.

**Out of scope: sequential testing and early stopping.** Every test here is a fixed-sample test. Looking at the
results as episodes come in and stopping as soon as a test turns significant inflates the false-positive rate well
above alpha; group-sequential designs (O'Brien-Fleming, alpha spending) are not offered. Decide the number of
episodes first (the power calculator), then test once.

## Sources

| Source | The two sides | Paired? |
|---|---|---|
| Success counts (the default) | rows of (condition, successes, trials): each against the first, or every pair | no: different episodes |
| Variants over the tolerance Monte Carlo | `core/tolerance.monte_carlo` on variant A and on variant B with **one seed**, the figures the tolerance study would carry (the requirements', else its defaults) | yes, **by seed**: draw *i* of both variants uses the same random numbers |
| Experiment runs, by a factor | the last designed experiment's runs at a factor's two levels (default the first and last level; a continuous factor with more than two values splits at its median) | when every run at one level has exactly one run at the other with the same other factors (a full factorial), else no |
| Experiment runs, two variants | the saved design run on variants A and B (`core/experiments.run`, `design.variant`) | yes, by run (the same factor values) |
| Tracker runs | two runs that kept their draws (a tolerance Monte Carlo logged by `core/tracking`): draw by draw; or two groups of runs (ids or a tag), one value per run | draws: when both runs state the same seed and number of draws; groups: when every run states a seed and the seeds match one to one |
| Samples (batch, API) | numbers given | as given |

Pairing by seed: with the same seed, `monte_carlo` draws the same uniform (or normal) numbers for the same sensors in
the same order, so draw *i* of A and draw *i* of B are the same relative mount errors (common random numbers). The
comparison is then of the designs, not of the luck of the draws, and a paired test is much sharper. When the
variants' sensors differ (another count or a calibrated sensor on one side), draw *i* is paired by index only; the
paired test stays valid (the pairs are independent of each other) but gains less.

## Tests and why

### Success rates

**Fisher's exact test** (default): conditional on the total successes *m*, the count of condition 1 is
hypergeometric, P(X = x) = C(n₁, x) C(n₂, m − x) / C(n₁ + n₂, m). Two-sided p: the probability of the tables no more
likely than the observed (relative tolerance 10⁻⁷, as R and SciPy); one-sided, as every test here, about condition 2 against condition 1 ('greater': p₂ > p₁, the z test's direction; SciPy's `fisher_exact` names that direction for the table with condition 2 in the first row). Why: exact at any number of episodes, which is
what robot-policy evaluations have (10 to 50 episodes per condition); conservative (its false-positive rate is at
most alpha). The **pooled two-proportion z test** is the option: z = (p̂₂ − p̂₁) / √(p̄ q̄ (1/n₁ + 1/n₂)); a note warns
when an expected count is under 5.

Intervals of each rate: **Wilson** (1927) score interval and **Clopper-Pearson** (1934): lower B(α/2; k, n − k + 1),
upper B(1 − α/2; k + 1, n − k) (beta quantiles; never below the nominal coverage). Effect sizes: the **risk
difference** p₂ − p₁ with **Newcombe's hybrid score interval** (1998, method 10, from the two Wilson intervals
(lᵢ, uᵢ)): [d − √((p₂ − l₂)² + (u₁ − p₁)²), d + √((u₂ − p₂)² + (p₁ − l₁)²)]; the **odds ratio** (k₂(n₁ − k₁)) /
(k₁(n₂ − k₂)) with **Woolf's** interval exp(ln OR ± z √Σ 1/cell), with ½ added to every cell when one is zero
(Haldane-Anscombe, said). When the Newcombe interval excludes zero but Fisher's test does not reject, a note says the
evidence is borderline (the two are different methods and Fisher's is conservative).

### Figures over draws, runs or episodes

The primary test is a **permutation test of the mean difference**:

- paired: T = mean(dᵢ), dᵢ = bᵢ − aᵢ; under H₀ each dᵢ is as likely positive as negative, so the null distribution is
  T over sign flips of the dᵢ;
- unpaired: T = mean(b) − mean(a); under H₀ every value is as likely to have come from either side, so the null
  distribution is T over relabellings of the pooled values.

It is **exact** (every rearrangement: 2ⁿ sign flips, C(n₁ + n₂, n₁) relabellings) when there are at most
`n_perm` (default 9 999) of them, else `n_perm` random rearrangements with p = (b + 1) / (B + 1) (Phipson and Smyth
2010). Two-sided p is twice the smaller tail, values within 100 ε of the observed counting as ties (SciPy's
convention). Why it is the default: the figures over tolerance draws or runs are often skewed or pile up at a bound
(a coverage share near 100 %), so a t test's normality is doubtful at the small n a study allows; the permutation
test needs only exchangeability under H₀. It also reacts to a difference of spread in the unpaired case (said).

Shown for reference with every comparison: the **paired t** test and the **Wilcoxon signed-rank** test (paired), or
**Welch's t** (Welch-Satterthwaite df = (v₁ + v₂)² / (v₁² / (n₁ − 1) + v₂² / (n₂ − 1)), vᵢ = sᵢ² / nᵢ) and the
**Mann-Whitney U** (unpaired). Their assumptions: the t tests, normal values (paired: normal differences); the rank
tests, continuous values (ties are handled by average ranks and the tie corrections). Method choice, as SciPy's:
Mann-Whitney exact (the U distribution by the recurrence f(i, j, u) = f(i − 1, j, u − j) + f(i, j − 1, u)) when a
sample has at most 8 values and there are no ties, else normal with the tie correction and a ½ continuity
correction; Wilcoxon (zero differences dropped) exact without ties or zeros up to 50 pairs (the distribution of T⁺
by subset sums), all 2ⁿ sign flips with ties or zeros up to 13 pairs, else normal with the tie correction.

Effect sizes: the **mean difference with a BCa bootstrap interval** (Efron 1987; bias correction
z₀ = Φ⁻¹(#{θ*ᵦ < θ̂} + #{θ*ᵦ ≤ θ̂}) / 2B), acceleration from the jackknife, a = Σ U³ / (6 (Σ U²)^{3/2}), the two
samples' jackknife values summed as SciPy does; seeded; percentile when BCa is undefined); **Cohen's d** (pooled σ)
and **Hedges' g** = J d with the exact J(df) = Γ(df/2) / (√(df/2) Γ((df − 1)/2)), or, paired, **d_z** = mean(d) /
sd(d); their intervals from the noncentral t (Steiger and Fouladi 1997: the λ with P(T ≤ t_obs | λ) = 1 − α/2 and
α/2, scaled back); **Cliff's delta** P(B > A) − P(B < A) = 2U/(n₁n₂) − 1 (the rank-biserial correlation, Kerby
2014) with a seeded percentile bootstrap interval; the **matched-pairs rank-biserial** (T⁺ − T⁻)/(T⁺ + T⁻).

A comparison that is not clear never reads as "equal". Instead of the observed power (which only restates the
p-value, Hoenig and Heisey 2001), it gives the **smallest difference the data detect** with the target power: the d
with power_t(d, n) = 0.8 times the sd (paired: of the differences), or, for rates, the rates below and above the
reference that the normal approximation detects at that n.

### Several figures or comparisons

p-values are corrected across the figures (or the rate comparisons) of one analysis: **Holm** (default; step-down,
p̃₍ᵢ₎ = max_{j ≤ i} min(1, (m − j + 1) p₍ⱼ₎), the family-wise error rate), **Bonferroni** (min(1, m p)),
**Benjamini-Hochberg** (step-up, p̃₍ᵢ₎ = min_{j ≥ i} min(1, m p₍ⱼ₎ / j), the false discovery rate under independence
or positive dependence), **Benjamini-Yekutieli** (BH times c(m) = Σ 1/i, any dependence) or none. A figure with no
p-value (too few values) is left out of m.

## Power and sample size

**Two proportions**, two-sided at alpha, equal n per condition:

- `normal` (Fleiss, Levin and Paik 2003): n = [z_{α/2} √(2 p̄ q̄) + z_β √(p₁q₁ + p₂q₂)]² / Δ², with the continuity
  correction n′ = n/4 (1 + √(1 + 4/(nΔ)))² (default on). Its power at a given n inverts the correction
  (n = (n′ − 1/Δ)² / n′) and uses Φ((Δ√n − z_{α/2}√(2p̄q̄)) / √(p₁q₁ + p₂q₂)) plus the other tail;
- `arcsine` (Cohen 1988): h = 2 asin √p₂ − 2 asin √p₁, n = 2 ((z_{α/2} + z_β) / h)²;
- `exact` (the default of the episodes calculator): every outcome (x₁, x₂) of the two binomials enumerated, each
  tested (Fisher's exact test, the rejection region found per total m from the hypergeometric probabilities; or the
  z test with Yates' correction), power = Σ Bin(x₁; n, p₁) Bin(x₂; n, p₂) 1[reject] (up to 2 000 per condition; the
  normal approximation above, said);
- `simulation`: seeded draws of both binomials, the same test applied.

The power of an exact test saw-tooths with n (the counts are discrete): the smallest n that reaches the target is
reported, found by bisection and a scan of the 20 n below. For 70 % against 95 % at alpha 0.05 and power 0.80 the
three methods give 39 (exact, Fisher), 43 (normal with the continuity correction) and 32 (arcsine) episodes per
condition: the arcsine formula is the least conservative at rates near 1.

**t tests** through the noncentral t: two-sample df = 2n − 2, λ = d √(n/2); paired df = n − 1, λ = d_z √n;
power = P(T > t_c | λ) + P(T < −t_c | λ). Where SciPy's noncentral t gives NaN far in a tail the normal approximation
is used (the power there is 0 or 1 to many digits).

**Evaluation episodes** (`episodes_needed(p1, p2)`): the two-proportion n with the assumptions spelled out: every
episode an independent trial with its condition's success probability (no drift, no learning, no two episodes failing
together because they share a scene or a reset), the same number per condition, a two-sided test, and p₁, p₂ the
rates to tell apart (a pilot's estimate is itself uncertain: plan for the smallest difference that matters).

**Demonstrations per condition** (`demonstrations_needed`): to tell a figure measured on each demonstration apart
between two conditions (two rigs, two setups): a per-demonstration value (completion time, a quality score: the t
test's n, paired when a scene is recorded under both) or pass/fail (the two-proportion n); then inflated by Kish's
(1965) design effect 1 + (m − 1) ρ for m demonstrations per operator with intra-class correlation ρ (default 0:
independent demonstrations), and by 1 / (1 − attrition) for the share the audit discards (default 0). This is **not**
the number of demonstrations a policy needs to learn the task (that depends on the policy and the task and is not
modelled).

Defaults (DexVantage defaults, labelled so in the UI and the results): alpha 0.05, power 0.80, confidence 0.95,
9 999 bootstrap resamples and permutations, seed 0, Holm's correction, 32 tolerance draws per variant.

## The result, the headline and the suggestion

The study's result: the comparisons (each: the test and why, both sides, the difference with its interval, the
effect sizes with intervals, the reference tests, p and the corrected p, the verdict), the correction, the pairing,
the assumptions, the power calculation with its curve, references and the out-of-scope note. The headline is the
comparison with the smallest corrected p, as its difference B − A (rates in percentage points), status **clear
difference** (good: corrected p ≤ alpha), **no clear difference** (warning) or **cannot test** (critical: too few
values); with only the power calculator, n per condition with **enough planned** / **underpowered** / **needed**.

The suggestion (the page's Apply, one undo step): when the headline difference is not clear, the episodes per
condition (rates), draws per variant (variants) or runs (experiments, tracker) that would detect the observed
difference with the target power, written into the settings. It plans from the observed difference, itself uncertain.

## Validation (measured 2026-11-07, `--validate --models stats`)

| Check | Reference | Measured | Tolerance |
|---|---|---|---|
| `stats.proportion_ci` | `scipy.stats.binomtest(k, n).proportion_ci` (wilson, exact), every k of n ∈ {1, 7, 20, 50, 137} at 90/95/99 % | 3.2 × 10⁻¹⁵ | 10⁻⁹ |
| `stats.t_tests` | SciPy `ttest_ind(equal_var=False)`, `ttest_rel`, 20 data sets, p, t and the interval | 8.9 × 10⁻¹⁶ | 10⁻⁹ |
| `stats.rank_tests` | SciPy `mannwhitneyu`, `wilcoxon` (exact, permutation, asymptotic; three alternatives) | 1.1 × 10⁻¹⁶ | 10⁻⁹ |
| `stats.fisher_odds` | SciPy `fisher_exact`, `contingency.odds_ratio(kind='sample')` | 1.1 × 10⁻¹⁵ | 10⁻⁹ |
| `stats.permutation_exact` | SciPy `permutation_test` in its exact mode, 18 cases | 0 | 10⁻¹² |
| `stats.permutation_mc` | SciPy `permutation_test`, its own random stream, 9 999 rearrangements | 1.6 SE | 4 SE |
| `stats.bootstrap` | SciPy `bootstrap` (percentile, BCa; one sample, paired, two samples) | 0.12 bootstrap SE | 0.25 |
| `stats.multiplicity_bh1995` | Benjamini and Hochberg (1995) p. 295: Bonferroni 3, FDR 4 rejections; Holm 3, BY 3 and every adjusted p written out again | 5.6 × 10⁻¹⁷ | 10⁻¹² |
| `stats.power_cohen1992` | Cohen (1992) Table 2: n per group for d and h = 0.2, 0.5, 0.8 at alpha 0.01, 0.05, 0.10 | 1 per group | 1 |
| `stats.power_gpower` | G*Power 3.1 manual §19.3 (p. 48), §20.3 (p. 50): 0.832114, 0.500352, N 30 (0.955144), N 1492 (0.900169) | 4.6 × 10⁻⁷ | 5 × 10⁻⁶ |
| `stats.power_simulation` | exact Fisher power against 2 000 simulated evaluations tested with SciPy's `fisher_exact`; t power against simulated normal samples | 1.2 SE | 4 SE |
| `stats.ci_coverage` | coverage of the d interval (noncentral t) and of Newcombe's interval over simulated data | 0.24 SE | 4 SE (Newcombe: beyond 0.01) |
| `stats.bruteforce` | Cliff's delta by counting every pair, the paired rank-biserial by a loop, power at the detectable difference | 7.4 × 10⁻¹⁴ | 10⁻⁹ |

Tolerances and why: closed forms and exact distributions computed by other code agree to rounding (10⁻⁹); Monte
Carlo comparisons are judged in their standard errors (4: a false alarm about once in 15 000 per comparison); the
bootstrap's 0.25 comes from measurement (over 30 seeds each endpoint's standard deviation was 0.022 to 0.045
bootstrap standard errors in either implementation, the mean difference under 0.01, so two independent runs differ
with a standard deviation under 0.06); Cohen's table rounds to the nearest unit and comes from approximations (for
d = 0.2 at alpha 0.05 the exact noncentral t needs 394, as G*Power also gives, the table 393), so one per group.
The G*Power examples are one-sample and matched-pairs (the same noncentral t with df n − 1); the manual has no
worked two-sample example, which Cohen's table covers. Newcombe's interval is an approximation: at the two settings
simulated its coverage was 94.8 % and 95.6 %.

## Known limits

Fixed-sample tests only (no sequential testing or early stopping). Episodes and runs are taken as independent; the
pairing is what the source states (by seed, by run, by the other factors). Power is computed for success rates and
for normal data with equal variances; the permutation and rank tests' own power is not (their asymptotic relative
efficiency against the t test is at least 0.864, Hodges and Lehmann 1956), so the t-based plan is an approximation
for them. The operators' clustering is only Kish's design effect with equal cluster sizes. BCa intervals are poor
with fewer than about 10 values. A difference that is not clear is not shown to be absent: read the smallest
detectable difference. Corrections apply within one analysis, not across analyses run separately. The variants'
Monte Carlo inherits the tolerance model's limits (`docs/models/tolerance.md`), and its default 32 draws per variant
re-run every study the figures need.

## References

Wilson 1927; Clopper and Pearson 1934; Newcombe 1998; Efron 1987; Efron and Tibshirani 1993; Phipson and Smyth 2010;
Welch 1947; Mann and Whitney 1947; Wilcoxon 1945; Cohen 1988 and 1992; Hedges and Olkin 1985; Steiger and Fouladi
1997; Cliff 1993; Kerby 2014; Holm 1979; Benjamini and Hochberg 1995; Benjamini and Yekutieli 2001; Fleiss, Levin
and Paik 2003; Kish 1965; Hoenig and Heisey 2001; the G*Power 3.1 manual (2023); SciPy 1.18 `scipy.stats`. Full
entries in `core/stats.py` REFERENCES and `validation/references.py`.
