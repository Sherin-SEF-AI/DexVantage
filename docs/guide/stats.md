# Statistics: a worked example

Three questions a research team asks, answered on the SO-100 sample and on success counts. The model is in
`docs/models/stats.md`. Figures below are from runs on 2026-11-07 at the default settings (alpha 0.05, power 0.80,
Holm's correction, seed 0).

## 1. Is Diffusion Policy better than ACT on our task?

Twenty evaluation episodes each: ACT succeeded 14 times, Diffusion Policy 19 times. Open Analysis ▸ Statistics, keep
**Success counts**, and enter the two rows (condition, successes, trials). The page runs as you type:

- the headline: **+25.0 pp**, status **no clear difference**:
  > Diffusion against ACT: no clear difference (Diffusion − ACT = +25.0 pp [+0.8, +47.3]; adjusted p = 0.0915).
  > This does not show they are equal. With these episodes, only a true rate below 22 % (against 70 %) would be
  > detected 80 % of the time.
- the table: 14/20 (70 %) against 19/20 (95 %), the odds ratio 8.14 [0.878, 75.5] (Woolf), Fisher's exact test,
  p = 0.091; Wilson intervals 48–85 % and 76–99 % in the tooltip;
- the note that the difference's interval (Newcombe) excludes zero while Fisher's test does not reject: the evidence
  is borderline, which twenty episodes cannot settle.

The assumptions are spelled out under the table: each episode an independent trial, one success probability per
condition, the conditions' episodes independent of each other.

## 2. How many episodes would settle it?

The suggestion under the headline says: about 43 episodes per condition would tell 70 % from 95 % apart 80 % of the
time (normal approximation). **Apply** writes this plan into the power calculator (one undo step); **Preview** shows
it there without storing it. The calculator, method **Exact (Fisher, every outcome)**, then answers **39 episodes per
condition (78 in all) give 80.7 % power**, and draws power against n with the target line, the n needed and the n
planned. The other methods give 43 (normal approximation with the continuity correction) and 32 (arcsine): near a
rate of 1 the arcsine formula is optimistic, which is why the exact enumeration is the default here.

From the command line, with a spec file:

    {"schema": "dexvantage.stats/1", "source": "rates",
     "rates": [{"condition": "ACT", "successes": 14, "trials": 20},
               {"condition": "Diffusion", "successes": 19, "trials": 20}],
     "power": {"kind": "rates", "p1": 0.7, "p2": 0.95}}

    dexvantage --stats spec.json            # → spec.results.json with provenance, exit 0

or in a notebook:

    import dexvantage as dv
    dv.stats.compare_rates([('ACT', 14, 20), ('Diffusion', 19, 20)]).headline['word']   # 'no clear difference'
    dv.stats.episodes_needed(0.7, 0.95)['n_per_group']                                    # 39

Demonstrations instead of evaluation episodes: choose **Demonstrations**, a pass rate of 70 % against 90 %, 25
demonstrations per operator with an intra-class correlation of 0.05 and 10 % discarded by the audit: 176
demonstrations per condition (72 before the clustering and the discards). The calculator says what it does not
model: how many demonstrations a policy needs to learn the task.

## 3. Does raising the third-person camera by 6 cm change coverage?

On the SO-100 sample, add a variant (Variants ▸ Add variant, a copy) and raise its third-person camera by 6 cm. On the
Statistics page choose **Variants**, A the default (the active variant), B the copy, and figures
`coverage.covered, coverage.seen_twice`; set 16 draws and press **Run** (about 4 s). Each variant's mounts are drawn
within their tolerances with the same seed, so the draws are paired by seed:

| Figure | A | B | B − A [95 % BCa] | p (permutation, 9 999 sign flips) | Holm |
|---|---|---|---|---|---|
| Task volume seen | 94.56 % | 94.38 % | −0.179 [−0.187, −0.172] | 0.0002 | 0.0004 |
| Seen by two or more cameras | 47.21 % | 46.32 % | −0.890 [−0.896, −0.883] | 0.0002 | 0.0004 |

Both differences are **clear** (A is better), because the paired draws cancel the mount noise: the raised camera
loses 0.18 points of the task volume and 0.89 points of the two-camera share. Whether a 0.2-point loss matters is the
team's call; the statistics say only that it is not chance. The same comparison with `--stats` and
`"source": "variants"` in the spec, or `dv.stats.compare_variants(rig, b='Rig B', metrics=[...], n=16)`.

## Experiment and tracker runs

**Experiment runs** compares the last designed experiment's runs at a factor's two levels: a full factorial pairs the
runs that share the other factors (with three pairs the smallest two-sided p is 2/8 = 0.25, so even a constant
difference is not clear: the page says so through the p-value). **Tracker runs** compares two logged tolerance runs
draw by draw (paired when they used the same seed) or two groups of runs (by ids or `tag:NAME`), one value per run.

## What it will not do

It does not stop early: decide the episodes first, test once (sequential designs are out of scope). It does not call
two conditions equal when the difference is not clear; it gives the smallest difference the data could have detected.
