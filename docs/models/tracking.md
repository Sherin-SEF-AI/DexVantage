# Experiment tracker

`dexvantage/core/tracking/` keeps a record of every study run, designed experiment, optimiser run, tolerance Monte
Carlo and readiness scorecard (`--check`) as a **run**, in a **run store** beside the project, and exports the runs to
the tools research teams already use (CSV, Parquet, MLflow, Weights & Biases). The Runs page (Analysis ▸ Runs), batch
mode (`--track`, `--runs`, `--runs-export`), the Python API (`dv.runs`, `dv.track`) and the assistant (`get_runs`)
read the same store.

The tracker computes no figure of its own: every number in a run is a number its study's result states, read
through `core/metrics` (`read()`: `None` with status "not stated" when the result does not state it, `bound` kept
when the study states only a bound).

## Opt-in

Nothing is logged unless the project turns tracking on, `project.analysis.studies['tracking']`:

| Setting | Default (a labelled default) | Meaning |
|---|---|---|
| `enabled` | `false` | log runs of this project |
| `folder` | `''` | the run store; empty: `<project file name>.runs` beside the project file; relative paths from the project's folder |
| `quick` | `false` | also log the quick studies the window re-runs after each edit (off: they would log a run per edit) |
| `tags` | `[]` | tags added to every run logged from this project |

or batch mode is given `--track DIR`. The setting is left out of the project's content hash
(`core/provenance.py`, `_UI_STUDIES`, as the live toggle is): turning tracking on does not make the project "differ"
from the runs logged before it, and a project that never had the setting hashes exactly as before.

In the window, a result is logged when the analysis controller finishes it (a worker thread writes it), unless it is
a quick study and `quick` is off, the tracker's own summary, a result already logged in the session (the same store,
study, project hash, variant and result hash, so a result from the cache is not logged twice), or a result made
before the latest edit (its project is no longer the one in the window). A study that refuses is logged as a
**failed** run with its plain sentence. In batch mode and the API every requested run is logged: study flags,
`--experiment`, `--check`, `--stats`, `--campaign`, `--cells`, the study and check steps of a `--run-plan`, the
cluster runs (`--jobs N`, and a complete `--gather` of an experiment or a Monte Carlo) and `dv.cluster`'s results
(integration of research tools part 2).

## A run (schema `dexvantage.run/1`, `core/tracking/record.py`)

| Field | What |
|---|---|
| `run_id` | uuid4 hex (32 characters): unique without coordination, and a valid MLflow run id |
| `kind` | `study`, `experiment` (study `experiments`), `optimise` (`optimise`, `design_agent`), `tolerance`, `check` (`requirements`, `--check`) |
| `study`, `title`, `name`, `interface` | the core.jobs study id, its title, a display name, `app` / `batch` / `api` |
| `started_utc`, `ended_utc`, `seconds` | UTC, `YYYY-MM-DDTHH:MM:SS.mmmZ` (sorts as time); `seconds` is the study's own computation time; `started_utc` = end − seconds unless given |
| `status`, `reason` | `finished`, `cancelled`, `failed` (with the study's sentence) |
| `project`, `variant` | name, file name (never its folder), content hash (core/provenance); variant name, index, uid, own hash |
| `settings`, `seeds` | the study's settings with defaults filled in and every seed, as core/provenance records them |
| `metrics` | `{key: {metric, stat, title, unit, better, value, status, bound?, reason?, check?}}` (below) |
| `headline` | `{text, status, word}`: the study's headline metric and its status word |
| `result` | `{file: result.json, schema, sha256 (the result hash), output (where batch wrote it)}` |
| `provenance` | DexVantage version, code hash, git commit and dirty flag (from a checkout), library versions, platform, interface, command, inputs, plan |
| `tags`, `note` | free; tags are trimmed, without repeats, commas turned into spaces |
| `rerun_of`, `reproduce` | for a re-run: the run it re-ran and `{verdict, verdict_text, message, reasons}` (core/reproduce) |
| `samples` | (run.json only) per-draw values of a tolerance Monte Carlo: the replicates a significance test can use, in draw order with `null` for a draw that states no value (so two runs with one seed pair draw by draw; a pair with a missing side is dropped) |

**Metric keys.** Every core/metrics figure registered for the study, read off its result; plus, per kind:

- tolerance: each metric's nominal value (key = the metric id) and `id@p5`, `@p50`, `@p95`, `@mean`, `@std` over the
  draws; the draws themselves as `samples[id]`;
- experiment: each response at the baseline (the id) and `objective@best` (the best run's value);
- optimiser and design agent: each objective of the current rig (the id), `id@best` (the best of the designs
  offered, by the metric's direction) and `optimise.options`;
- scorecard: each requirement's figure (the id; a second requirement on the same metric `id@<uid>`) with `check`
  (`pass`, `fail`, `not_evaluated`), its bound and requirement text; `requirements.pass`, `.fail`, `.not_evaluated`.

## The run store (`core/tracking/store.py`)

    <store>/store.json            {schema dexvantage.runstore/1, created_utc}
    <store>/runs.jsonl            the index: one JSON object per line (JSON Lines)
                                  {op: add, run: {record without samples}} | {op: update, run_id, tags, note, at}
                                  | {op: delete, run_id, at}
    <store>/runs/<run_id>/run.json, result.json
    <store>/deleted/<run_id>/     a deleted run (moved, not erased)
    <store>/.lock                 the lock appends and updates take

**Adding a run.** The run's folder is written under a temporary name inside `runs/` (each file written to a
temporary file, fsync'd and renamed) and renamed to `runs/<run_id>` in one step; then its index line is appended
under an exclusive advisory lock (`fcntl.flock`) as one `write()` followed by `fsync`.
POSIX `rename()` keeps the new name "visible … and refer[ring] either to the file referred to by new or old before the
operation began": a reader sees a run folder whole or not at all. Updates (tags, note) rewrite `run.json` by
write-then-rename and append their index line, both under the lock. Nothing is edited in place.

**Reading.** The index is folded in order (add, then updates, deletes); a line that does not parse is counted as
damaged and skipped (a final line without its newline may be mid-write and is not counted); a run folder the index
does not name is read from its `run.json` ("recovered"); an indexed run whose folder is gone is marked
`folder_missing`. So:

| A writer that dies … | leaves | readers see |
|---|---|---|
| before its rename | a `runs/.tmp-…` folder | nothing (not a run id) |
| between rename and index line | a folder without an index line | the run, recovered from its folder |
| inside its append | a line without its newline | the next writer starts a new line; the cut line is one damaged line; the run is recovered from its folder |

## Comparing runs (`compare()`)

For runs r₁ … rₙ (r₁ the baseline, the oldest selected) and each metric key: the values vᵢ, the deltas
dᵢ = vᵢ − v₁, the relative changes dᵢ / |v₁| (none when v₁ is 0 or not stated), the best run by the metric's
direction (none when all stated values are equal), and the settings, seeds, project content hash, variant and
DexVantage version that differ. For two runs that kept replicates (a tolerance Monte Carlo's draws) the comparison of
the base metric adds a **significance note** from `core/stats.compare_samples(a, b, better, paired)` (a permutation
test with a bootstrap interval, written by the statistics module; paired when both runs drew with the same seed, so
draw k of each saw the same perturbation). Without replicates, or in a build without `core/stats`, the note says why
there is no test; a p-value is never made up.

## Re-running a run (`rerun()`)

`core/reproduce.rerun` runs the run's study again on the project given (the window's project as it is now) with the
settings it now holds, `core/provenance.report` compares every number and text with the stored result, and the new
result is logged as a run with `rerun_of` and the verdict (`reproduces`, `reproduces within the tolerances`, `does not
reproduce` with the reasons: a different project content, code, library, …).

## Exports (`export()`)

- **CSV** (`runs.csv` wide: one row per run, one column per metric key; `runs_long.csv`: one row per run and metric
  with unit, status, bound, check; `units.json`), RFC 4180 through Python's `csv` module. Numbers are written with
  `repr()`, Python's shortest text that reads back as the same float64, so the CSV and the Parquet files hold
  identical values; not stated is an empty cell.
- **Parquet** (`runs.parquet`, `runs_long.parquet`) through pyarrow: float64 numbers, UTF-8 strings, each metric's
  unit in its field metadata, null where not stated.
- **MLflow** (`mlruns/`): the layout of MLflow 3.17.2's `FileStore` (`mlflow/store/tracking/file_store.py`, read at
  the v3.17.2 tag on 2026-11-07), written directly: `<experiment>/meta.yaml`, `<experiment>/<run_id>/meta.yaml`
  (the persisted RunInfo: status as the `RunStatus` integer, FINISHED 3, FAILED 4, KILLED 5 for a cancelled run;
  `source_type` 4, LOCAL), `metrics/<name>` lines `"<timestamp ms> <value> <step>"`, `params/<name>` (the study, kind,
  flattened settings, seeds; at most 6000 characters), `tags/<name>` (`mlflow.runName`, `mlflow.note.content` = the
  note, `mlflow.source.git.commit`, `dexvantage.*`, `tag.<user tag>`, `dexvantage.unit.<metric>`,
  `dexvantage.not_stated`; at most 8000 characters), `artifacts/` (result.json, run.json). One experiment per project
  name ("DexVantage · <project>", its id the first 12 hex digits of the name's SHA-256 as an integer), plus the
  "Default" experiment 0. Names keep only what MLflow's `validate_param_and_metric_name` accepts; `@` and `/` become
  `_` (`coverage.covered@p5` → `coverage.covered_p5`). MLflow 3.13 and later put the file store "in maintenance mode"
  (3.7 to 3.12 only warn): open it with `MLFLOW_ALLOW_FILE_STORE=true` (or migrate it with `mlflow migrate-filestore`).
- **Weights & Biases** (`wandb/wandb_runs.json`, `wandb/replay_wandb.py`): W&B's offline run format is internal to
  its client and is not written. The standalone script replays each run through the official API of wandb 0.30.0
  (`wandb/sdk/wandb_init.py`, `wandb_run.py`, read 2026-11-07): `wandb.init(project, entity, id=<run id>, name,
  notes=<note>, tags, config=<study, kind, project, variant, settings, seeds, provenance>, group=<study>,
  job_type=<kind>, dir, mode="offline", reinit="finish_previous", resume="allow")`, `run.log(figures)`,
  `run.summary.update(figures)`, `run.finish()`; offline by default (`--online` to log directly), so `wandb sync`
  uploads later.
- `export_runs_manifest.json`: every file with its size and SHA-256, and the provenance block.

`mlflow-skinny 3.17.2` and `wandb 0.30.0` are the optional extra `tracking` (Apache-2.0; MIT): writing the exports
never needs them; the MLflow read-back check uses MLflow when it is installed, and the replay script needs wandb.

## Validation (`dexvantage/validation/checks_tracking.py`)

Measured 2026-11-07 (full run; tmpfs `/tmp`):

| Check | Reference | Tolerance | Measured |
|---|---|---|---|
| `tracking.store_round_trip` | four records built in the check (a rig-inventory result, a tolerance result with draws, a value not stated and samples, a scorecard with a bound and the three check states, a failed run; Unicode tags, a two-line note, rerun_of, reproduce), through a JSON round trip | 0 fields | 0 of 443 fields differ (run.json, index line, result.json, after an update, read by a new RunStore) |
| `tracking.concurrent_writers` | what each of N writer processes reports it wrote (run ids; figures writer × 1000 + i) | 0 runs | 8 writers × 100 runs (and an update of every third): 800 stored, 0 lost, 0 foreign, 0 corrupt, 0 damaged lines, 0 temporary files; all 8 writing at once at the busiest; 0.16 s on tmpfs, 1.8 s on ext4 (16 × 100 also 0 lost) |
| `tracking.crash_recovery` | the meaning of each crash, written out in the check | 0 cases | 7 of 7 cases hold (1 damaged line, 2 recovered folders, the temporary folder ignored, the later run intact) |
| `tracking.csv_parquet_identical` | the records' own values; read back with Python's `csv` module and `pyarrow.parquet` | 0 cells | 0 of 8 240 cells differ; numbers compared bit for bit over magnitudes 10⁻³⁰⁰ to 10³⁰⁰, both signs, −0.0 and integers |
| `tracking.mlflow_reader` | MLflow 3.17.2's `MlflowClient` (mlflow-skinny installed); without it, a reader of the layout written out from file_store.py | 0 fields | 0 of 80 fields differ (ids, names, statuses, start and end ms, metrics bit for bit, params, note, run name and user tags) |

Why tolerance 0: storage and file formats are exact or wrong; `repr()` of a float64 reads back to the same bits, and
MLflow parses metric values with `float()`.

The self-test line `tracking` runs the round trip, the crash recovery and the table check (0.9 s in a fresh process,
most of it imports and the built-in sample).

## Known limits

- A run records the project's content hash, not the project: keep the project file (or its Git history, the folder
  form of `core/projfolder.py`) to re-run an old run on the project it was made from; a re-run on a changed project
  says that the project differs.
- The lock is advisory, and a store on a network file system is as safe as that file system's `flock` and `rename`
  (NFS v4 honours both; some SMB shares do not).
- The index is read whole on each listing (about 2 kB per run).
- Significance needs replicates: single figures are compared by their difference only; more than two runs get deltas
  against the first and no significance note.
- MLflow: artifact URIs are absolute `file:` URIs of the folder as written (as MLflow writes them), so a moved
  `mlruns/` keeps its metrics, params and tags but its artifact links point to the old place; MLflow datasets, inputs,
  traces and logged models are not written; MLflow ≥ 3.13 needs `MLFLOW_ALLOW_FILE_STORE=true` to read a file store.
- W&B: units go in the run's config (W&B has no unit field); figures not stated are left out of history and summary
  (listed in config); running the replay twice resumes the same runs and logs the figures again as a new step.
- In the window, logging hangs off the Runs page, which the window builds when it starts; the page listens to the
  analysis controller itself, not to its own held signals (`ui/pages/_gate.py`), so runs are logged while it is
  hidden. If pages are ever built only when first opened, the hook must move into the window.

## References

- MLflow 3.17.2, `mlflow/store/tracking/file_store.py`, `mlflow/entities/run_info.py`, `run_status.py`,
  `source_type.py`, `mlflow/utils/validation.py`, `mlflow/utils/yaml_utils.py`,
  https://github.com/mlflow/mlflow/tree/v3.17.2 (read 2026-11-07); `mlflow/protos/service.proto` (RunStatus).
- wandb 0.30.0, `wandb/sdk/wandb_init.py`, `wandb/sdk/wandb_run.py`, https://github.com/wandb/wandb/tree/v0.30.0
  (read 2026-11-07).
- POSIX.1-2017 `rename()`, https://pubs.opengroup.org/onlinepubs/9699919799/functions/rename.html; Linux `flock(2)`,
  https://man7.org/linux/man-pages/man2/flock.2.html (read 2026-11-07).
- JSON Lines, https://jsonlines.org/ (read 2026-11-07); RFC 4180 (CSV); Apache Parquet file format.
