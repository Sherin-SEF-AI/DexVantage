# The Python and notebook API

`import dexvantage as dv` gives DexVantage without the window: open or make a rig, run any study, read the named
metrics, sweep factors, check requirements, export, and read recorded datasets. It runs the same code as the window and
batch mode (`core.jobs`, `core.metrics`, `core.factors`, `core.exporters`, `core.requirements`, `core.datasets`), so
every figure is the study's own: the API computes nothing itself. No Qt is imported on this path (a test checks it), so
it runs on a headless server, in CI and in Jupyter.

The module is `dexvantage/api.py`; its docstring examples are run as a doctest by `tests/test_api.py`. Two worked
notebooks are in `examples/` (below).

## Install

    pip install -e ".[research]"        # pandas 2.3.3 for DataFrames; without it tables are lists of dicts

`Result.to_frame()` returns a pandas DataFrame when pandas is installed, else `dv.Records` (a list of row dicts) whose
`.note` says how to get a DataFrame. In a DataFrame a figure that is not stated is `NaN` (pandas' missing number); in
the dicts it is `None`.

## A worked example on the UR5e sample

    >>> import dexvantage as dv
    >>> rig = dv.sample('ur5e')
    >>> rig
    <Rig 'UR5e + Robotiq 2F-85 sample': 1 robot, 2 sensors, 1 variant>
    >>> cov = rig.run('coverage')                       # 0.9 s on the RTX 5080 machine the figures below are from
    >>> cov.headline['text']
    'Task volume seen: 91.9 % (mostly single or blind)'
    >>> cov.to_frame('cameras', columns=['name', 'seen', 'hidden', 'outside_image', 'ppmm_median'])
                name      seen    hidden  outside_image  ppmm_median
    0     Wrist D405  0.464096  0.140706       0.395198       2.1325
    1  Overhead D435  0.719290  0.280710       0.000000       1.6760

`run` takes a study id (`dv.studies()` lists them: id, title, batch flag, whether it needs a robot, and the plugin that
added it), a dash form of it (`depth-validity`) or its batch flag (`rig-inventory`). Keyword arguments are settings laid
over the stored ones **for this run only**: `rig.run('coverage', cube_n=256)`. `progress=fn(fraction, text)` reports
progress. The result is a `dv.Result`: the study's own JSON (a `dict`, so `json.dumps(cov)` works) plus

- `.headline`: the study's headline metric with its unit, the status word the result gives and a sentence
  (`'Worst camera to joint-state skew: ≥ 54.4 ms (out of sync)'` on this sample: a bound is shown as one);
- `.metrics()`: every metric of the study read off this result, without running anything again;
- `.to_frame(key=None, columns=None)`: the metrics, or the list under `key` (one row per item, nested fields as
  dotted columns such as `fov_deg.h`); `.tables()` lists the keys that hold such lists;
- `.save(path)`: the JSON with the provenance block, as batch mode writes it.

### A factor sweep

Factors are the settable quantities of `core/factors.py` (`rig.factors.paths()` lists them with their values):

    >>> cam = 'Overhead D435'
    >>> rows = []
    >>> for dz in (-0.10, 0.0, 0.10):
    ...     moved = rig.with_factors({f'sensor:{cam}.dz': dz})       # a new Rig: rig is unchanged
    ...     rows.append({'dz_m': dz, 'z_m': round(moved.factors[f'sensor:{cam}.z'], 3),
    ...                  **moved.metrics(['coverage.covered', 'depth.valid']).figures()})

gives (pandas)

       dz_m  z_m  coverage.covered  depth.valid
    0  -0.1  0.9           90.5456    79.508750
    1   0.0  1.0           91.9015    84.607292
    2   0.1  1.1           92.0444    87.004375

so raising the overhead D435 by 10 cm buys 0.1 points of task volume seen and 2.4 points of depth-valid volume on this
sample (the coverage and depth-validity models' own Known limits apply: `docs/models/coverage.md`,
`docs/models/depth.md`).

### What changes the rig and what makes a new one

| Call | Effect |
|---|---|
| `rig.factors[path] = value`, `rig.factors.update({...})` | edits `rig` itself: its project is replaced by the edited copy `core.factors.apply` returns (Robot and Sensor objects read before are then stale) |
| `rig.with_factors({...})`, `rig.factors.apply({...})`, `rig.use_variant(name)`, `rig.copy()` | a new Rig; `rig` is unchanged |
| `rig.run(study, **settings)` | runs on a copy with the settings for this run; stored settings unchanged (store one with `rig.factors['setting:coverage.cube_n'] = 256`) |
| `rig.robots`, `rig.sensors`, `rig.variants`, `rig.project` | the project's own objects: changing one changes the rig |
| `rig.add_variant(name, {sensor factors})` | edits `rig`: adds a copy of the active variant (sensor uids kept) with those factors set on the copy; the active variant stays (`dv.stats.compare_variants(rig, b=name)` then compares them) |

A locked sensor is the user's: setting one of its factors raises `FactorError` unless `allow_locked=True` is given.

### Metrics, requirements, export, save

    >>> rig.metrics(['coverage.covered', 'sync.skew_ms', 'bandwidth.margin']).to_frame()[['id', 'value', 'status', 'bound']]
                     id    value      status     bound
    0  coverage.covered  91.9015          ok      None
    1      sync.skew_ms  54.4258          ok  at_least
    2  bandwidth.margin      NaN  not stated      None
    >>> rig.check([{'name': 'Task volume seen', 'metric': 'coverage.covered', 'op': '>=', 'threshold': 90}]).headline['text']
    'Readiness scorecard: 1 / 1 (ready). All 1 requirement pass.'
    >>> manifest = rig.export('out/ur5e', ['opencv_yaml', 'mjcf'])   # files, notes, SHA-256, provenance
    >>> rig.save('ur5e.dexv')

`bandwidth.margin` is "not stated" because the sample has no host controller; the 90 % threshold is this example's own
figure, not a standard. `rig.save(path)` writes a `.dexv` file, or the unpacked, Git-friendly form when the path is a
folder or ends in `.dexv.d`. Every refusal, from `dv.open` and `dv.sample` to a factor value that is not a finite
number or a device the library does not have, is a `dv.DexVantageError` with a plain sentence. `rig.check` takes a
requirements file (YAML or JSON, `docs/guide/requirements.md`), a list of
requirement dicts, `'defaults'` (DexVantage's labelled defaults: they include the grasp study, about 30 s on a sample)
or nothing (the rig's own requirements). Each study runs once per call and results are cached by content for the
process.

### Datasets

    >>> ds = dv.datasets.open('out/so100_synthetic')   # LeRobot v2.1 / v3.0, RLDS, ROS 2 bags, plugin formats
    >>> ds
    <Recording LeRobot v3.0 'so100_synthetic': 2 episodes>
    >>> ds.to_frame()                 # one row per episode
    >>> ds.to_frame(episode=0)        # that episode's joint-state rows (t and one column per joint)
    >>> rig.run('audit', dataset='out/so100_synthetic').headline['text']
    'Episodes that pass the audit: 0 % (not usable)'

(`dv.sample('so100').synth_episodes('out/so100_synthetic', episodes=2, frames=24, max_width=160)` makes that dataset:
every fault on with its defaults, so no episode passes, which is the point of it.) `dv.datasets.formats()` lists the
readers, plugins' included.

### Jupyter

`Rig`, `Result`, `Recording` and the tables have HTML views: a rig shows its robots, variants (the active one in bold),
task volumes and a sensor table; a result shows its headline figure in a large mono font with its status word in the
status colour, its metrics, its run time and its tables. Every value is HTML-escaped; a figure that is not stated is
"—" with the tooltip "not stated".

## The example notebooks

`examples/01_rig_studies_factors.py` (the UR5e: sensors, coverage, the sweep above, a requirement, an OpenCV YAML
export, saving the moved rig; 4.9 s, 0.8 GB) and `examples/02_datasets_audit.py` (the SO-100: two synthetic episodes
rendered in MuJoCo, read back, audited; 2.7 s on the RTX 5080) are jupytext percent-format scripts: run them with
`python examples/01_rig_studies_factors.py`, or open them as notebooks with `jupytext --to notebook <file>` (jupytext
is not a DexVantage dependency). `DEXVANTAGE_EXAMPLE_OUT` chooses where they write (default: a new temporary folder).
`tests/test_examples.py` runs both. `examples/03_research_workflow.py` (the SO-100: tracked runs and their MLflow,
W&B, CSV and Parquet export, statistics and power, the campaign, a tolerance sweep on a process pool, two cells, the
project diff, the paper kit, BibTeX and a vector figure; about 4 s) is walked through in `docs/guide/research-workflow.md`;
`tests/test_completeness_api.py` runs it. The research tools of part 1 are in the API too: `dv.diff`, `dv.merge`,
`dv.bibtex`, `dv.figures`, `dv.figure` and `dv.export_runs` (`docs/models/research-api.md`).

## Plugins

The API loads the installed plugins on first use (`dexvantage/plugin_sdk.py`, `docs/guide/plugins.md`), so their
studies, metrics, exporters and dataset readers work here as built-in ones do. `dv.plugins()` lists what was found;
`DEXVANTAGE_NO_PLUGINS=1` turns them off.

## Known limits

The API is a thin layer over the studies, so their Known limits apply unchanged. Studies run in the calling thread (a
long study blocks the cell; pass `progress=`). The result cache holds 32 results per process, keyed by the project's
content, the study settings, the library entries the project names and the dataset fingerprint. UI pages are not
reachable from Python. HTML views show at most 200 rows of a table (`to_frame()` has them all). `rig.synth_episodes`
needs MuJoCo's offscreen renderer (EGL).
