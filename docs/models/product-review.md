# Product review of research tools part 2: a research team on ALOHA 2

This review used the research tools the way a lab would, on the ALOHA 2 sample (two arms, four D405 cameras), once
from a notebook (`import dexvantage as dv`) and once in the window (offscreen). It covered these steps:
- a two-variant comparison;
- a Sobol experiment on the local pool and through a Slurm array (mocked: the shards run by hand) with gather;
- tracking every run, and comparing runs with statistics;
- a data-collection campaign sized by a power analysis;
- the sim-to-real gap on a synthetic recording with injected gaps;
- three cells and their pooling mapping;
- the paper kit, written and opened;
- a robot imported from USD.

Each step was also given bad inputs. The review measured UI-thread time and looked at screenshots. Measured on
2026-11-07 (RTX 5080 through EGL, 24 CPUs, each run under a 4–5 GB memory cap). The fixes are tested in
`tests/test_product_review.py` (24 tests).

## What was broken, and what changed

| Found | Fix |
|---|---|
| A notebook could not make a second variant: there was no API for it. A two-variant comparison needed `core.project.Variant` and `core.factors` by hand. | `rig.add_variant(name, {sensor factors}, from_variant=None)` adds a copy of the active variant to the rig (the window's "New variant from this one"). The sensors keep their uids, the factors are set on the copy only, and the active variant does not change. A duplicate or empty name, a setting path, an unknown variant and an unknown sensor are each refused in one sentence. |
| The API logged `rig.run`, `rig.check` and `dv.cluster.*` runs only. The `dv.stats.*` comparisons on a rig, `dv.campaign.plan`, `dv.cells.compare` and `dv.cluster.gather` were not logged, while their batch forms were. | Each is logged when the project's tracking is on. Success counts with no rig have no project to log into, so they stay unlogged. `dv.cluster.gather(folder, rig=rig)` logs into the rig's store; without `rig` it logs into the job folder's base project, as `--gather` does. |
| The sim-to-real result on ALOHA 2 held NaN: the mount σ of `cam_low` on axes the frames do not constrain. The tracker refused the run ("Out of range float values are not JSON compliant: nan"), and the result was not valid JSON. | That σ is `None` (not stated); `core/sim2real/gap.py` `_mount`. |
| A +20 DN brightness gap on every camera was read as gain 0.99 and offset +20 DN. The lighting cause said "Every camera sees the scene 0.99× as bright as the render" about a recording 20 DN brighter. With lights in the project, Apply would scale them by 0.99, which cannot close an offset. | The cause names the offset ("gain 0.99, offset +20 DN"). The lights are scaled only when the cameras share no offset of `min_reduction_dn` (1 DN, the existing default) or more. Otherwise Apply sets each camera's render response, and the text says why: a scale of the lights gives a gain, not an offset. |
| `rig.gap_dataset(folder, {'brightness_dn': 'lots'})` raised a raw `ValueError`. A folder that was not empty was refused only after every frame was rendered (28 s on ALOHA 2). | The gaps are checked first: each is a finite number in its range (gain > 0; blur and noise ≥ 0), and mount is three positions and three angles. A folder that is not empty is refused before any render. |
| A campaign spec with an unknown key was silently ignored. `{"power": {"p1": 0.6, "p2": 0.8}}` gave 50 demonstrations per condition, the entered default, instead of a power analysis. | `core.campaign.read_spec` refuses an unknown setting. The refusal lists the settings, and for power-like keys it shows the `demos` form. `dv.campaign.plan(rig, **settings)` checks its settings the same way. |
| `dv.cluster.*(jobs=0)` (and −2, 1.5, "4", True) ran serially without a word, while `--jobs 0` is refused. | Refused in one sentence; `jobs=None` is the default count. |
| `dv.stats.compare_runs(rig, id, id)` compared a run with itself and reported a result. | Refused: "Run … is on both sides". |
| Opening a broken USD file showed pxr's C++ names, its source path and a cut-off temporary path: "(Error in 'pxrInternal_v0_26_8__pxrReserved__::Sdf_TextFileFormatParser::_RaiseErrorPEGTL' at line 48 in file /opt/USD/… : '/tmp/tm)". | The reader's own words are shown: "bad.usda could not be opened as a USD file (line 3, column 1: Expected ) at '' in </>)". A file that is not a USD layer reads "it is not a USD layer DexVantage can read". No full path is shown. |
| Statistics page, Variants: the controls were spread over the panel's height, and the figures field had no label. | The spare height goes below the controls on the Variants, Experiment runs and Tracker runs panels, as on Success counts. The figures field is labelled and takes the row's width. |
| Statistics: a zero difference read "+0 % [+0, +0]" (headline, table and verdict text). | Written as 0, as elsewhere ("a value that rounds to zero is written 0"): `core/stats` verdicts, the study's headline, the page. |
| The Runs table and the paper kit's runs read a p-value as "1 p". | "p = 1", as `dv.Result.headline` does. |
| Campaign summary: "1 operator, 1 sessions". The HTML schedule had the same plural fault. | Plurals agree with their counts. |
| Cells page: a file that is not there was added as a cell, and then the whole comparison refused to run. | It is refused when added ("There is no file at missing.dexv: it was not added as a cell"). The other cells stay. |
| A logged designed experiment had no duration: `seconds` was empty, and its start equalled its end. | The tracker reads an experiment's `execution.seconds`. |

## Measured (notebook, ALOHA 2)

- `rig.add_variant('Raised high cam', {'sensor:cam_high (overhead).dz': 0.05})`, then
  `dv.stats.compare_variants(rig, b='Raised high cam', metrics=['coverage.covered'], n=8)`: 21.0 s. Result: "no clear
  difference", p = 1. Task volume seen is 100 % in both variants at every draw, so the difference is exactly 0. It is
  logged.
- Saltelli design, N = 4 with two factors: 16 runs plus the rig as it is. `dv.cluster.experiment(rig, design,
  jobs=2)` took 11.2 s. Through `dv.cluster.slurm(..., shards=3)`, `run_shard` × 3 took 5.1–6.4 s each, and
  `gather` 0.1 s. The gathered runs equal the pool's. Both runs are logged, with their duration.
- `dv.campaign.plan(rig, {'demos': {'mode': 'power', 'p_base': 0.6, 'p_target': 0.8}})`: 91 episodes per condition
  (Fleiss, Tytun and Ury's continuity correction through `core/stats`), 73.3 GB, 1 day, 1 session.
- `rig.gap_dataset(folder, {'brightness_dn': 20, 'blur_px': 1.0}, episodes=1, frames=20)` took 2.5 s, and
  `rig.sim2real(folder)` 15.5 s: 20.4 DN, "very large gap". The lighting cause reads gain 0.99 and offset +20 DN in
  four cameras, and leaves 1.0 DN after the corrections.
- `dv.cells.compare(rig, 'sample:so100', 'sample:ur5e')`: 4.9 s, largest gap 0.373 ("large gap"), with the pooling
  mapping.
- Paper kit from the window (methods, tables, figures): 90 of 90 methods numbers and 81 table numbers traced,
  "ready".
- The bundled UR5e USD: imported headless and through the wizard (6 revolute joints and the fixed base joint, one robot added).

## Measured (window, offscreen, ALOHA 2, tracking on)

No step went over 100 ms on the UI thread. Every page action was timed directly, and the event loop was watched with
a 5 ms timer while each study ran and after each action. The software 3D view's paint was left out, as in
`scripts/measure_ui.py`. Offscreen, that paint takes 0.9–1.1 s per frame on ALOHA 2: the stall the first pass of
this review measured was that paint (sampled stacks: `viewport3d._paint_scene`), not the pages. With the GL view it
does not run. Pages checked:
- Requirements, "New variant from this one";
- Statistics: Variants, and bad success counts (refused, row by row);
- Designed experiments (Saltelli), Preview and Apply (one undo step);
- Runs, with a comparison;
- Campaign: power mode, and five bad field values (each refused);
- Sim-to-real: a missing dataset (a plain sentence), then the gap recording, and Apply;
- Cells: two samples and a missing file;
- Paper kit: preview and export;
- the USD import wizard: the bundled UR5e imported, a broken file and a missing file refused.

The visible labels were scanned for tracebacks, raw exceptions, "nan"/"None" and full paths after each step; none
remained after the fixes.

## Known limits

The offscreen measurements leave out the software 3D view's paint (about 1 s per frame on ALOHA 2 with soft
rendering). A machine without GL that falls back to it will feel that. The "mocked" Slurm array runs the shards by
hand with `run_shard`; `sbatch` itself was not run (no Slurm here). The campaign spec check refuses keys that a
newer DexVantage might add: a newer spec read by this version names the key it does not know. The sim-to-real
offset rule reuses `min_reduction_dn` (1 DN) as the smallest offset a light scale cannot absorb. That is a
labelled default, not a calibrated figure.
