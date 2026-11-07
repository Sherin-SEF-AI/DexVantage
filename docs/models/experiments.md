# Designed experiments {#experiments}

Module: `dexvantage/core/experiments.py` (the designs, the runs, the estimators, design.json, the batch run). Study:
`dexvantage/core/studies/experiments.py` (study id `experiments`; batch flag `--experiment-saved OUT.json` for the
experiment saved in the project; `--experiment DESIGN.json [--experiment-out OUT.json]` runs a design file). Page:
`dexvantage/ui/pages/experiments.py` (Analysis ▸ Designed experiments). Checks: `dexvantage/validation/checks_experiments.py`
(model `experiments`). Tests: `tests/test_experiments.py`. Spec §7.11. Ported from Vantage `core/experiments.py`: the
designs, the bootstrap of the Sobol indices, the main effects and the result layout. The automotive factors (cameras
on a vehicle body, load cases, weather, the ROI) are replaced by DexVantage's factor paths (`core/factors.py`). The
responses are the named metrics of `core/metrics.py`.

## What is computed

An experiment evaluates the rig at many settings of a few of its quantities (the *factors*). It reads one or more
figures at each setting (the *responses*). It reports:
- each run's values;
- what drives each figure: main effects and, for a Saltelli design, Sobol indices;
- the best run, against the rig as it is.

**Factors** are `core/factors` paths, in three kinds:

| Kind | design.json | Values |
|---|---|---|
| continuous | `"lo"`, `"hi"`; optional `"n_levels"` (full factorial), `"integer": true` | `lo + u·(hi − lo)` (rounded when integer) |
| categorical | `"levels": [...]` (any JSON values, in order) | level `⌊u·L⌋` of `L` |
| bool | `"kind": "bool"` (or the path's own kind, e.g. `.enabled`) | `false`, `true` |

`u ∈ [0, 1]` is the run's unit coordinate. The kind comes from `"kind"`, else from `"levels"`, else from
`core.factors.describe`. Typical paths are:
- `sensor:<name or uid>.x|y|z|roll|pitch|yaw`: absolute values in the mount's parent frame, metres and degrees;
- `.dx|dy|dz` and `.droll|dpitch|dyaw`: a shift of the mount, or a small turn about the sensor body's own axes,
  relative to the design;
- `.model`, `.enabled`;
- `setting:<study>.<key>`, a study setting;
- `json:/pointer`, for anything else.

Use the relative paths to explore around a placement. An absolute `pitch` range of ±5° turns a camera that looks
60° down to the horizontal. Locked sensors are not varied: a design with one is refused, with the sentence
`core.factors` gives.

**Responses** are metric ids of `core/metrics.py` (`core/metric_defs.py` defines those of the studies, such as
`coverage.covered` or `grasp.contact_visibility`). Each run sets the factors on a copy of the project
(`core.factors.apply`, a new project; the project itself is never changed). It then reads the responses with
`core.metrics.evaluate`, which runs each study once per run through `core.jobs.run_study` and the study
`ResultCache`. A study that refuses gives the run no value for its responses. The run records the status
(`refused`, `not stated`, `error`) and the study's sentence. The analyses leave that run out of that response and
count it. The rig as it is (the *baseline*) is evaluated first.

## Designs

| Method | Runs | How |
|---|---|---|
| `full_factorial` | Π Lᵢ | every combination of the levels. A continuous factor takes `n_levels` evenly spaced levels (the design's `levels`, default 2); the first factor varies fastest |
| `fractional` | 2^m | a two-level fraction at resolution III or IV (below) |
| `lhs` | n | Latin hypercube (McKay, Beckman and Conover 1979). Column j is `(πⱼ(i) + Uᵢⱼ)/n`, with a random permutation `πⱼ` and `U ~ U(0, 1)`, seeded |
| `sobol` | n | the first n points of a scrambled Sobol sequence (`scipy.stats.qmc.Sobol`, Owen scrambling, seeded). Its balance holds for n a power of two; another n is allowed and said |
| `saltelli` | N(k + 2) | the A, B and AB_i matrices of Saltelli (2002). A and B are the two halves of a 2k-dimensional scrambled Sobol sequence of N points; AB_i is A with column i taken from B |

**Fractional factorials** (Box, Hunter and Hunter 2005). Every factor must have two levels: a continuous factor's
`lo` and `hi`, a bool, or a categorical with two levels. A factor with more levels is refused. The m base factors form
a full 2^m factorial in standard order. Each added factor is the elementwise product of some base columns (its
*generator*, e.g. `E=ABC`):
- **Resolution III**: m is the smallest with 2^m ≥ k + 1. The generators are the two-factor interactions of the base
  factors, then the three-factor ones, and so on (`D=AB, E=AC, F=BC, G=ABC` for 2^(7−4)). Main effects are free of
  each other, but aliased with two-factor interactions.
- **Resolution IV**: m is the smallest with 2^(m−1) ≥ k. Every generator is a product of an odd number (≥ 3) of base
  factors, largest first. Each word of the defining relation is then a generator times its base factors (even
  length), or a product of such words (still even). So the shortest word has length ≥ 4. Main effects are free of
  two-factor interactions.

When m = k there is no fraction to take, and the design is the full 2^k factorial (said). `resolution_of` finds the
resolution reached again by brute force: the length of the shortest set of columns whose product is constant. The
result reports it with the generators.

## Analysis

**Main effects.** For each factor, the mean response at each level, with a t interval and its count. The *effect* is
the mean at the factor's last level minus the mean at its first. For a two-level design this is the classical main
effect; for a continuous factor it runs from `lo` to `hi`. Its interval is Welch's t interval:

    d = ȳ_last − ȳ_first,   se² = s²_last/n_last + s²_first/n_first,
    ν = se⁴ / ((s²_last/n_last)²/(n_last − 1) + (s²_first/n_first)²/(n_first − 1)),   d ± t_{ν, 1−α/2} · se

It treats the other factors' variation as noise. The sampled designs (the Latin hypercube, the Sobol sequence, and
the A and B rows of a Saltelli design) are handled differently. There, a continuous factor's values are cut into
quantile bins: `bins`, six by default, with at most one bin per two runs. The effect is then the last bin's mean minus
the first's. Categorical and bool factors always use their levels.

**Sobol indices** (Sobol 1993). These come from a Saltelli design. They use Jansen's (1999) estimators as Saltelli et
al. (2010, Table 2, (f)) give them, with `f(·)` the response at each row:

    V    = Var(f(A) ∪ f(B))            (the 2N values together, ddof 1)
    S_i  = (V − ½ · mean_j (f(B)_j − f(AB_i)_j)²) / V          first order
    S_Ti = ½ · mean_j (f(A)_j − f(AB_i)_j)² / V                 total

The intervals are percentile bootstrap intervals (Efron and Tibshirani 1993): the N rows are resampled with
replacement (`bootstrap`, 500 by default; 95 % by default). Rows with a missing value anywhere are left out, and
`dropped` counts them. A response that does not vary gets no index, and the result says so.

**The best run.** This is the run with the best value of the objective response (`objective`, default the first
response) in its metric's better direction. It is compared with the baseline. The headline's status word is:
- **better than now** (good);
- **no run beats the rig** or **same as now** (warning);
- **not evaluated**, when no run gave a value (critical).

## design.json (schema `dexvantage.experiment/1`)

```json
{
  "schema": "dexvantage.experiment/1",
  "name": "Overhead and wrist camera tilt",
  "variant": "",
  "factors": [
    {"path": "sensor:Overhead D435.dpitch", "lo": -5, "hi": 5},
    {"path": "sensor:Overhead D435.dz", "lo": -0.05, "hi": 0.05},
    {"path": "sensor:Wrist D405.model", "levels": ["intel-realsense-d405", "intel-realsense-d435"]},
    {"path": "sensor:Overhead D435.enabled", "kind": "bool"},
    {"path": "setting:grasp_visibility.n_grasps", "lo": 100, "hi": 400, "integer": true}
  ],
  "responses": ["coverage.covered", "depth.valid"],
  "objective": "coverage.covered",
  "design": {"method": "lhs", "n": 32, "levels": 2, "resolution": "IV", "seed": 0},
  "analysis": {"bootstrap": 500, "confidence": 0.95, "bins": 6},
  "execution": {"workers": 0}
}
```

| Key | Default | Meaning |
|---|---|---|
| `schema` | — | `dexvantage.experiment/1`; another value is refused |
| `name` | `""` | a name for the experiment |
| `variant` | `""` (active) | the uid of the variant the runs change and evaluate |
| `factors` | — | up to 32, each `{"path", "kind"?, "lo"/"hi" or "levels", "n_levels"?, "integer"?, "name"?}`; a path at most once |
| `responses` | — | metric ids, in order (or `{"metric": id}`) |
| `objective` | first response | the response that picks the best run |
| `design.method` | `lhs` | `full_factorial`, `fractional`, `lhs`, `sobol`, `saltelli` |
| `design.n` | 32 | points (`lhs`, `sobol`); base samples N (`saltelli`); 2–100 000 |
| `design.levels` | 2 | full factorial: levels of a continuous factor without `n_levels` (2–11) |
| `design.resolution` | `IV` | fractional: `III` or `IV` |
| `design.seed` | 0 | the sampled designs and the bootstrap |
| `analysis.bootstrap` | 500 | bootstrap resamples (0–5000) |
| `analysis.confidence` | 0.95 | interval level (0.5–0.999) |
| `analysis.bins` | 6 | quantile bins of the sampled designs' main effects (2–12) |
| `execution.workers` | 0 | 0 or 1: serial (the default); n > 1: a pool of n processes |

These are DexVantage defaults, not standards. A flat layout (`"method": "lhs"` at the top level) is read too.
Numbers out of range are clamped. A missing setting takes its default. The same document is what the page saves in
`project.analysis.studies["experiments"]`.

The result (`dexvantage.experiments/1`) holds:
- `design`, as read;
- `plan`: the generators and resolution of a fraction;
- `factors`: title, unit, range or levels, current value;
- `responses`: title, unit, better, study;
- `baseline`;
- `runs`: each run's `values`, `responses`, `status` and `reason`, and its Saltelli `block` (`A`, `B`, `AB<i>`);
- `analysis` per response: count, missing, mean, sd, min, max, `main_effects` and `sobol`;
- `best`, `headline`, `execution`, `notes`, `known_limits`, `references` and the provenance block.

`--experiment` also writes the runs as CSV beside the JSON (same name, `.csv`). It has columns `run`, `block`, one
per factor path, one per response (with its unit), and one status per response. The JSON records the CSV's name and
SHA-256 under `files`.

`--reproduce RESULT.json` re-runs the `experiments` study, which reads the experiment saved in the project. So a
result of `--experiment-saved` (or of the page) reproduces. A result of `--experiment DESIGN.json` from a project
without that design saved is refused with a sentence ("The experiment has no factor yet …"). Run the design again
instead: the result carries it under `design`.

## Execution

Runs are serial by default. With `workers` > 1 they are evaluated in a `concurrent.futures.ProcessPoolExecutor` of
that many *spawned* processes. Each process:
1. receives the project once;
2. imports the modules that register the responses' metrics and studies;
3. keeps its own `ResultCache`.

The results are identical to a serial run (tested); only the order of evaluation differs. A cancel (the page's
**Cancel run**, or the study's cancel) stops between runs; pooled runs already queued, at most two per process,
finish first. A run during which the cancel came is discarded, because a study stopped part-way gives no value to
keep. The runs done are analysed.

Two caches serve the runs:
- The study result cache (`core.jobs.ResultCache`) holds 8 whole results. A coverage result of the UR5e sample is
  about 1.7 MB of JSON, so it keeps few.
- The experiment study also keeps the response values of up to 20 000 runs (`RowCache`). They are keyed by the
  ResultCache keys of each run's project: its content hash, the study settings, the library entries and a dataset's
  fingerprint.

Runs are evaluated on a copy of the project without the experiment's own settings. So **Run** after a cancel, or
after an edit to the design that keeps some of its runs, evaluates only the runs not done (tested). A process pool
does not use the run cache.

Timings were measured on the development machine (2026-11-07):
- the overhead is about 12 ms per run on the UR5e sample (copying the project and hashing it for the cache);
- a coverage and depth-validity run of that sample takes about 1 s;
- 16 Latin hypercube runs took 16.2 s serially;
- 320 Saltelli runs took 194.9 s in a pool of three processes, about 1.6 times quicker than serial at that rate.

## Validation

`dexvantage --validate out/ --models experiments` (measured 2026-11-07, seed 20261027):

| Check | Reference | Tolerance (why) | Measured |
|---|---|---|---|
| `experiments.sobol_ishigami` | the Ishigami function's analytic indices (a = 7, b = 0.1), derived again in the check: V1 = ½(1 + bπ⁴/5)², V2 = a²/8, V13 = ½b²(π⁸/9 − π⁸/25) | 2/√N: 0.031 at N = 4096, 0.0625 at the quick run's 1024. About 0.6 of the bootstrap 95 % half-width at those N (0.045–0.053; 0.088–0.095). Every analytic index must also lie in its interval | 0.0008 (N = 4096), 0.012 (N = 1024); all inside |
| `experiments.lhs_strata` | the Latin hypercube's definition: one point per stratum | 0 (counts are exact) | 0 (n = 7, 50, 256; five factors, on the factor values) |
| `experiments.sobol_discrepancy` | `scipy.stats.qmc.discrepancy` (CD) on the Sobol design and the Latin hypercube; `scipy.stats.qmc.Sobol` points with the same seed | 1e-9 relative (floating point) | 3.4e-11 |
| `experiments.sobol_vs_random` | E[CD²] of n independent uniform points, ((5/4)^d − (13/12)^d)/n, itself checked against 40 random sets (within 15 %) | 0.5 ratio (a low-discrepancy set sits far below random; a loose bound) | 0.027 (n = 256, d = 4); the 40-set mean is 3.6 % from the expectation |
| `experiments.main_effects_linear` | the coefficients × ranges of y = 2 + 3a − 1.5b + 0.5c + 4[d = B] (3 × 3 × 3 × 2 factorial), and a linear model plus 2·x1·x2 − x3·x5 on a resolution IV fraction (7 factors, 16 runs) | 1e-9 (exact) | 0 |
| `experiments.fractional_resolution` | the definition of resolution, by brute force; the smallest run count; SᵀS = nI | 0 designs failing | 0 of 20 (3–12 factors, III and IV) |

The tests add the following:
- interval coverage: over 12 seeds at N = 256, the analytic Ishigami indices lie inside their 95 % intervals at least
  85 % of the time;
- interval width shrinks by 1.5–2.7× from N = 256 to 1024 (√4 = 2 expected);
- Welch's interval against `scipy.stats.ttest_ind(equal_var=False)`;
- a resolution III fraction biases an aliased main effect by exactly the interaction (and resolution IV does not);
- a project experiment's every run against a direct evaluation of the toy study.

Self-test line `experiments`: Ishigami at N = 1024 within 2/√N, and a 16-run resolution IV fraction's exact main
effects: 31 ms in a full self-test, about 0.4 s alone (importing scipy.stats).

## Known limits

- Every response is the study's own figure, with that study's assumptions and limits.
- Sobol indices assume independent factors and settle slowly. Read them with their intervals: they should be a few
  hundredths wide before small differences mean anything. Percentile bootstrap intervals over quasi-random rows are
  conservative.
- Main-effect intervals treat the other factors as noise. They say nothing about interactions. A resolution III
  fraction aliases main effects with two-factor interactions.
- The binned main effects of a sampled design are coarse with few runs. Eight runs give four bins of two.
- A run a study refuses has no value. It is left out of that response's analysis and counted in the notes.
- Locked sensors are not varied.
- The best run is the best *sampled* point, not an optimum. The optimiser (§7.11) searches.
- A process pool pays DexVantage's import and each robot model's build once per process. It pays only for studies
  slower than about a second per run. It needs multiprocessing's freeze support in a frozen bundle, which has not
  been tested.
- The Camera view has no overlay for this page: a run is a mount pose and settings, not an image feature.

## References

- I. M. Sobol', "Sensitivity estimates for nonlinear mathematical models", Mathematical Modelling and Computational
  Experiments 1(4), 407–414, 1993.
- A. Saltelli, "Making best use of model evaluations to compute sensitivity indices", Computer Physics Communications
  145(2), 280–297, 2002.
- A. Saltelli, P. Annoni, I. Azzini, F. Campolongo, M. Ratto and S. Tarantola, "Variance based sensitivity analysis
  of model output. Design and estimator for the total sensitivity index", Computer Physics Communications 181(2),
  259–270, 2010.
- M. J. W. Jansen, "Analysis of variance designs for model output", Computer Physics Communications 117(1–2), 35–43,
  1999.
- T. Ishigami and T. Homma, "An importance quantification technique in uncertainty analysis for computer models",
  ISUMA '90, 398–403, 1990.
- M. D. McKay, R. J. Beckman and W. J. Conover, "A comparison of three methods for selecting values of input
  variables in the analysis of output from a computer code", Technometrics 21(2), 239–245, 1979.
- F. J. Hickernell, "A generalized discrepancy and quadrature error bound", Mathematics of Computation 67(221),
  299–322, 1998.
- G. E. P. Box, J. S. Hunter and W. G. Hunter, Statistics for Experimenters, 2nd ed., Wiley, 2005.
- B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap, Chapman & Hall, 1993.
- SciPy `scipy.stats.qmc`, https://docs.scipy.org/doc/scipy/reference/stats.qmc.html.
