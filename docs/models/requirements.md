# Requirements, metrics and the readiness scorecard {#requirements}

Modules: `dexvantage/core/metric_defs.py` (the named figures of every study), `dexvantage/core/requirements.py`
(requirements, the scorecard, the variants compare), `dexvantage/core/studies/requirements.py` (studies
`requirements`, batch `--scorecard OUT.json`, and `variants_compare`, batch `--variants-compare OUT.json`),
`dexvantage/core/studies/rig_inventory.py` (study `rig`, batch `--rig-inventory OUT.json`),
`dexvantage/core/reproduce.py` (`--reproduce`), `dexvantage/core/run_plan.py` (`--run-plan`),
`dexvantage/batch_gate.py` (the batch options). Page: `dexvantage/ui/pages/requirements.py` (Analysis ▸ Requirements
and variants; Review workspace). Checks: `dexvantage/validation/checks_requirements.py` (model `requirements`).
Tests: `tests/test_metric_defs.py`, `tests/test_requirements.py`, `tests/test_requirements_batch.py`,
`tests/test_requirements_page.py`. Spec §7.11, §11 and §13 ("Requirements and Variants compare").

Ported from Vantage `core/requirements.py`: the signed margin, the verdict over the items, the scorecard document and
the CI gate. Stripped: Vantage's automotive metric registry (sectors, object classes, lidar, GNSS and power figures),
the must/should priorities, the brief parser and the design-review PDF.

## Metrics

A metric is one number a study's result already holds, with its unit and the direction that is better
(`core/metrics.py`, the shared contract). `core/metric_defs.py` registers one for every study's headline figure (tag
`headline`) and the obvious secondary figures. Nothing is computed there beyond unit changes (fractions to %, metres
to mm, seconds to ms) and sums or extremes over rows the result lists.

A figure the result does not state reads `None` with status `not stated` and, where the study says why, its sentence
(`reason`). A figure the study states only as a bound carries `bound`:

- `at_least`: the time-sync skew and action-label error when some clock or exposure figure is not stated (the study's
  "≥"), the controllers over budget and the total payload when a stream has no stated rate, the cable pulls and snags
  when a cable's length or bend radius is not stated, the blind gap when a finger is blind from the start of the scan;
- `at_most`: the motion smear and rolling-shutter skew over the maximum exposure or readout the stream allows.

The contract change (additive): an extractor may return `{'value', 'bound', 'reason', 'note'}` instead of a number;
`metrics.read(metric, doc)` turns any extractor's output into a row, and a result shaped otherwise than the extractor
expects reads `not stated` instead of raising.

| Id | Figure | Unit | Better | Study |
|---|---|---|---|---|
| `grasp.contact_visibility` | Contact visibility | % | higher | `grasp_visibility` |
| `grasp.contact_visibility_low` | Contact visibility, lower bound of its interval | % | higher | `grasp_visibility` |
| `grasp.both_fingertips` | Both fingertips seen | % | higher | `grasp_visibility` |
| `grasp.reachable` | Grasps reachable | % | higher | `grasp_visibility` |
| `coverage.covered` | Task volume seen | % | higher | `coverage` |
| `coverage.seen_twice` | Task volume seen by two or more cameras | % | higher | `coverage` |
| `coverage.blind` | Task volume blind | % | lower | `coverage` |
| `coverage.ppmm_p5` | Resolution, 5th percentile | px/mm | higher | `coverage` |
| `depth.valid` | Depth-valid task volume | % | higher | `depth_validity` |
| `depth.worst_sensor_valid` | Depth-valid share of the weakest depth sensor | % | higher | `depth_validity` |
| `depth.sigma_z_p95_mm` | Depth noise, 95th percentile (worst sensor) | mm | lower | `depth_validity` |
| `bandwidth.margin` | Smallest controller margin | Mbit/s | higher | `bandwidth` |
| `bandwidth.busiest_load` | Busiest link or controller load | % | lower | `bandwidth` |
| `bandwidth.over_budget` | Controllers over budget | count | lower | `bandwidth` |
| `bandwidth.payload_mbps` | Total payload | Mbit/s | lower | `bandwidth` |
| `sync.skew_ms` | Worst camera to joint-state skew | ms | lower | `time_sync` |
| `sync.label_error_mm` | Worst action-label error from the skew | mm | lower | `time_sync` |
| `cables.worst` | Cables not clear | count | lower | `cables` |
| `cables.pull_snag` | Cable pulls and snags | count | lower | `cables` |
| `cables.bend` | Cable bends under the allowed radius | count | lower | `cables` |
| `vla.min_tokens` | Fewest ViT tokens on a critical object | tokens | higher | `vla_tokens` |
| `vla.fingertip_tokens` | Fewest ViT tokens on a fingertip | tokens | higher | `vla_tokens` |
| `vla.warnings` | Token warnings | count | lower | `vla_tokens` |
| `action_chunk.air_grasp_mm` | Worst air grasp | mm | lower | `action_chunk_sync` |
| `action_chunk.worst_mm` | Worst closing error | mm | lower | `action_chunk_sync` |
| `action_chunk.overshoot_mm` | Worst overshoot | mm | lower | `action_chunk_sync` |
| `action_chunk.reaction_mm` | Reaction distance | mm | lower | `action_chunk_sync` |
| `teleop.overlap` | Teleoperable task volume | % | higher | `teleop` |
| `lighting.glare` | Specular saturation (worst camera) | % | lower | `lighting` |
| `lighting.depth_loss` | Depth lost to lighting (worst sensor) | % | lower | `lighting` |
| `motion_blur.worst_px` | Worst motion smear | px | lower | `motion_blur` |
| `motion_blur.skew_px` | Worst rolling-shutter skew | px | lower | `motion_blur` |
| `handeye.sigma_mm` | Predicted camera position σ (worst axis) | mm | lower | `handeye_plan` |
| `handeye.sigma_deg` | Predicted camera rotation σ (worst axis) | ° | lower | `handeye_plan` |
| `handeye.ratio` | Predicted σ over the requirement (worst) | ratio | lower | `handeye_plan` |
| `groundtruth.agreement` | Rendered against predicted agreement | % | higher | `groundtruth` |
| `groundtruth.depth_agreement` | Rendered depth agreement | % | higher | `groundtruth` |
| `dexhand.fingertips_seen` | Fingertips seen | % | higher | `dex_occlusion` |
| `dexhand.all_fingertips` | Grasps with every fingertip seen | % | higher | `dex_occlusion` |
| `dexhand.blind_gap_mm` | Sensory blind gap before contact (worst) | mm | lower | `dex_occlusion` |
| `dexhand.unsensed` | Fingertips neither seen nor touched at contact | count | lower | `dex_occlusion` |
| `brackets.not_ready` | Sensors without a ready bracket | count | lower | `brackets` |
| `audit.pass_rate` | Episodes that pass the audit | % | higher | `audit` |
| `plan.match` | Plan against recording: comparisons that match | % | higher | `plan_vs_recording` |
| `silhouette.offset_mm` | Mount offset found by the silhouettes | mm | lower | `silhouette_calib` |
| `silhouette.offset_deg` | Mount rotation found by the silhouettes | ° | lower | `silhouette_calib` |
| `rig.camera_count` | Cameras | count | lower | `rig` |
| `rig.sensor_count` | Sensors | count | lower | `rig` |
| `rig.cost` | Rig cost (where the library states a price) | USD | lower | `rig` |
| `rig.mass_g` | Sensor mass | g | lower | `rig` |
| `rig.power_w` | Sensor power | W | lower | `rig` |

`rig.cost` reads an optional `price` object of a library entry (`{"value", "currency", "source", "read"}`). **No
built-in entry states a price** (none was invented), so on every sample the cost reads "not stated" with the devices
named; prices in different currencies are not added.

## Requirements

A requirement is a metric, a comparison (`>=`, `<=`, `<`, `>`, `==`) and a threshold in the metric's unit, with the
study that measures it (`core.project.Requirement`: `uid`, `name`, `metric`, `op`, `threshold`, `unit`, `study`,
`is_default`, `note`; the project file's `requirements` section, unchanged). A requirements file (YAML or JSON) is

```yaml
format: dexvantage.requirements
version: 1
requirements:
- {name: Contact visibility, metric: grasp.contact_visibility, comparison: '>=', threshold: 95, unit: '%',
   study: grasp_visibility, default: true}
```

`comparison` also reads `op` and words (`at least`, `less than`, `≥` …). A unit or study that is not the metric's is
an error, not a conversion: a threshold in the wrong unit would be silently wrong.

**Defaults** (labelled defaults, `is_default` true, note "a documented starting point, not a standard";
`docs/examples/requirements.yaml`; `--check defaults`): contact visibility ≥ 95 %; camera to joint-state skew < 2 ms;
controllers over budget ≤ 0; cable pulls and snags ≤ 0; sensory blind gap before contact ≤ 0 mm. Editing a default's
threshold or comparison on the page makes it the user's own figure (no longer labelled a default).

## The scorecard

For each requirement, with value v, threshold t and the metric's figure as the study states it:

    margin = v − t      for ≥ and >          pass when margin ≥ −ε (≥) or margin > ε (>)
    margin = t − v      for ≤ and <          pass when margin ≥ −ε (≤) or margin > ε (<)
    margin = −|v − t|   for ==               pass when margin ≥ −ε,       ε = 1e-9 · max(1, |t|)

Positive margins are good. With a bound the comparison on v decides only in the bound's direction: `at_least` passes
`≥`/`>` and fails `≤`/`<` (and `==` when v > t); `at_most` the reverse; anything else is **not evaluated**, with the
sentence "The study states only that the figure is at least v, which does not settle less than t: give the figures it
lacks." A study that refuses gives "not evaluated" with its own sentence; a study that crashes gives an error.

Verdict: `pass` ("ready") when every item passes; `fail` ("not ready") when one fails; else `incomplete` ("not
evaluated"); `empty` ("no requirements"). Each study runs at most once per scorecard (a shared result cache of 24
results, keyed by the project's content, the study's settings, the library entries, **the active variant** and the
dataset fingerprint).

The CI gate `dexvantage PROJECT --check requirements.yaml [OUT.json]` prints the table, writes the scorecard JSON with
its provenance (default `scorecard.json`) and exits **0** when every requirement passes, **1** when one fails or is not
evaluated (a gate cannot certify what it did not measure) and **2** on an error (a file that cannot be read, invalid
YAML, an unknown metric, a threshold that is not a number, a wrong unit, no project, a study that crashed).

## Variants compare

Every variant of the project against every requirement (the scorecard per variant, each study once per variant): a
cell per variant and requirement (value, bound, status, margin, reason), the **best variant per requirement** (by the
metric's better direction; for `==` the smallest distance to the threshold; ties are all named; a cell without a value
is never best), and a ranking by requirements passed, then fewest failures. The page's Apply makes the top-ranked
variant active when it passes more (one undo step); Preview draws its sensors next to the active ones; New variant
from this one copies the active variant (its sensors keep their uids, so host links, cables and policy cameras
apply to the copy) and makes it active (one undo step).

## Reproduce and plans

`--reproduce RESULT.json` re-runs what wrote a result (the study in its provenance; a scorecard with the requirements
recorded in it; a run-plan summary step by step with each step's settings) on the project given, else the sample its
command names (`--sample NAME`), else the project file it names next to the result or in the current folder, and
compares every number and text (`core/provenance.py` `compare`), exactly or within `--tolerance`. Computation-time
blocks that are not under a `seconds` key (`timings`, `throughput`, `timing`, `timing_s`) may differ and are counted as
such. It prints the verdict, the reasons and each mismatch (path, recorded, reproduced) and with `--reproduce-out
REPORT.json` writes the full report. Exit codes as `--check`: 0 reproduces, 1 does not, 2 cannot be tried.

`--run-plan plan.json [OUT_DIR]` runs a list of steps in order (`study` with `settings` and `out`; `check` with a
requirements file, `defaults` or inline items; `export` with `formats` and `options`; `reproduce` with `tolerance`),
each on a snapshot with that step's settings only, every result stamped with the plan and the step in its provenance;
one summary (`plan-summary.json`, schema `dexvantage.run_plan/1`) lists each step's status, output, result SHA-256,
the headline metrics of a study and the verdict of a check. Exit 2 when a step could not run, else 1 when a check
failed or a reproduction did not match, else 0; a step that cannot run is recorded with its sentence and the plan
carries on.

## Validation (measured 2026-11-07, `--validate --models requirements`, 7.2 s with start-up)

| Check | Reference | Measured | Tolerance |
|---|---|---|---|
| `requirements.gate_exit_codes` | the exit code of 8 synthetic cases written out from spec §11 (pass, fail, not evaluated, unknown metric, bad threshold, wrong unit, broken YAML, missing file) | 0 cases wrong, no traceback | 0 cases (an exit code is right or wrong) |
| `requirements.bound_decisions` | a Monte Carlo over the true figures each bound allows (64 per case, with the bound and the threshold), compared by the definition written out again; 4 000 seeded cases (1 000 quick) | 0 disagreements | 0 (the rule is exact) |
| `requirements.metrics_vs_results` | 14 figures read off the bandwidth, time-sync, cable, coverage, depth, motion-blur, bracket and rig results of the UR5e sample (with its suggested host) by hand | 0 difference | 0 (the same number) |
| `requirements.reproduce_round_trip` | the bandwidth and time-sync results of `--sample ur5e`, re-run from their provenance alone | 0 numbers or texts differ | 0 (deterministic studies) |
| `requirements.reproduce_detects_perturbation` | one number of each result changed by 1e-6 relative and the hash rewritten | both found, exactly that path | 0 results missed |

Self-test line "requirements" (the first two checks, quick): 0.8 s.

## Known limits

- The scorecard evaluates the active variant with each study's stored settings; the studies' own Known limits apply
  unchanged (contact visibility is the grasp study's "every fingertip at least 50 % visible in at least one camera";
  the bandwidth margin is the specification's capacity bound minus the computed load, and so on).
- A bound decides only in its direction; the multi-finger blind gap is scanned at the study's refine step (2 mm by
  default), so a gap shorter than a step reads 0.
- There are no priorities: every requirement is decisive. Thresholds compare with the study's rounded figure.
- The variants compare runs every study each variant needs (the grasp study takes its time per variant).
- `rig.cost`, `rig.mass_g` and `rig.power_w` count the enabled sensors only (no hosts, cables, lights, brackets); the
  cost is not stated until a library entry states a price.
- `--reproduce` uses the project's settings as they are now (a change shows as "the project differs"); dataset
  studies need `--dataset` as when they ran; GPU renders may differ in their last bits between drivers.
