# The research tools from Python: diff, merge, BibTeX, figures and run export

`dexvantage/api_research.py` (re-exported as `dv.diff`, `dv.merge`, `dv.bibtex`, `dv.figures`, `dv.figure`,
`dv.export_runs`), added by the completeness review of research tools part 2. Batch mode already had every one of
these (`--diff`, `--merge`, `--cite`, `--figure`, `--runs-export`); a notebook had to import core modules for them.

## Model

There is none of its own: each function hands its inputs to the module that batch mode uses and wraps the answer.

| Function | Calls | Returns |
|---|---|---|
| `dv.diff(a, b, metrics=None)` | `core/projdiff.diff`, and `evaluate_deltas` with the API's result cache for `metrics` | `Diff`: the `dexvantage.diff/1` document (a dict) with `.identical`, `.text`, `.html`, `.to_frame()`, `.save(path)` (with provenance), an HTML view |
| `dv.merge(base, ours, theirs)` | `core/projdiff.merge3` | `Merge`: `.rig` (unsaved), `.from_ours`, `.from_theirs`, `.notes`; conflicts raise `dv.MergeConflict` (a `DexVantageError`, `.conflicts` in words) |
| `dv.bibtex(rig, path=None, references='checks')` | `core/cite.bibtex` | the BibTeX text (written to `path` too) |
| `dv.figures(result)` | `core/publication.figure_specs` (the paper kit's) | `[{'id', 'study', 'spec'}]` |
| `dv.figure(source, path, width='single', palette='tol', index=0)` | `core/figures.write`; bars sized and labels cut as the paper kit does (`10 + 5 mm` per row at least, `publication.fit_labels`) | `{'path', 'format', 'width_mm', 'height_mm', 'series'}` |
| `dv.export_runs(where, out_dir, formats=…, **filters)` | `core/tracking.export` | the export's manifest: CSV, Parquet, MLflow `mlruns/`, W&B offline replay |

Projects are a Rig, a core Project or a path (a `.dexv` file or an unpacked `.dexv.d` folder). Errors are
`DexVantageError` with one sentence. No Qt is imported on this path except to write a PDF figure (Qt's PDF writer).

The same review made three small changes the API relies on:

- `Result.headline` for a study without a headline metric (the tolerance Monte Carlo, the scorecard, experiments,
  the optimiser) shows the study's own figure with its unit and quantile ("Monte Carlo tolerances: P5 94.4 % (no
  requirement)") instead of the raw float ("94.344085"); a p-value reads "p = 0.0915", not "0.0915 p".
- `core/provenance.environment` reads the PyQt6 and Qt versions from the installed wheels' metadata when Qt has not
  been imported, so `dv.track`, `Result.save` and every other provenance stamp no longer import Qt headless (they did:
  `PyQt6.QtCore` was imported to read two version strings). With Qt imported it reads them from Qt as before; the
  two agree (6.12.0 and 6.12.0 here).
- `core/cluster/local.check_main_guard`: a process pool (`dv.cluster`, `--jobs`, the experiments' pool) started from
  a script's top-level code outside `if __name__ == '__main__':` is refused in one sentence naming the line, before
  any worker starts. Before, each spawned worker re-ran the script, multiprocessing raised its 16-line bootstrapping
  error in every worker and the parent died with `BrokenPipeError`. The check mirrors `multiprocessing.spawn`: a main
  module run by name (`python -m`, so `python -m dexvantage` and `python -m pytest`) or without a file (a notebook,
  the interactive prompt) is not re-run, so it is not checked; a script's line is looked up in its own syntax tree.

And one to the plugin SDK: `PluginAPI.check(...)` registers a plugin's validation check under its own model
(`plugin.<name>`, `validation/suite.py` `PLUGIN_MODELS`, `all_models()`, `model_ids()`); its references join
`validation/references.REFERENCES` while the plugin is loaded. Unloading, or a plugin that fails, removes the checks,
the model and the references again. `--validate --models` accepts the plugin models (`available_models()`).

## Validation

Nothing here computes a figure, so there is no validation check of its own; the tests compare each function with the
module it calls on the SO-100 sample (`tests/test_completeness_api.py`, 16 tests):

- `dv.diff` returns exactly `projdiff.diff`'s document and `to_text`'s text; its metric delta equals the two rigs'
  `coverage.covered` read through `rig.metrics` (94.52 % → 94.32 % for the third-person camera raised 100 mm).
- `dv.merge` keeps both sides' edits of two different sensors and raises `MergeConflict` with the place in words when
  both sides move the same sensor differently.
- `dv.bibtex(…, references='none')` equals `cite.bibtex`; `dv.figures(result)` equals the paper kit's specs; an SVG
  written from a result, from a spec item and from a spec file has the stated width (88.9 and 182 mm).
- `dv.export_runs` writes every format the store export writes; filters pick the runs.
- In a fresh process, every new function (and `dv.track`) leaves `sys.modules` without PyQt6, PyQt5 or PySide6.
- The guard: an unguarded script gets the sentence with its line number and no traceback; a guarded one runs; the
  syntax-tree lookup finds lines inside `if __name__ == '__main__':` and `if '__main__' == __name__:` blocks and not in
  a function body or above the block.

The plugin checks are tested in `tests/test_completeness_plugins.py` (5 tests): a toy plugin's check runs in the suite
under its model, passes, appears in the JSON and HTML report with its reference and in the checks' BibTeX; unloading
and a failing plugin remove it; a taken id, an unknown reference key and a key with other text are skipped with a
sentence each.

## Known limits

The functions inherit their modules' limits (lists such as grasp poses are diffed and merged whole; only the curve and
bar charts have figure specs; `dv.figure` of a result draws its first per-item list of numbers unless `index` picks
another). `dv.diff(…, metrics=…)` runs those metrics' studies on both projects in the calling thread. The main-guard
check cannot see a pool started from code that `exec`s a script, and a script that starts a pool in a guarded block
still re-runs its unguarded top-level code in every worker (Python's spawn rule: keep all top-level work under the
guard, as `examples/03_research_workflow.py` explains). Plugin checks join a run only when the plugin is loaded in that
process (the app, batch mode and the API load plugins; `python -m dexvantage.validation` does not), and the paper
kit's per-study validation summary names the built-in models only.
