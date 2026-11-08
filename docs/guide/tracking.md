# Tracking runs

The experiment tracker keeps every study run of a project, with its settings, figures and provenance, so you can
see what you tried, compare it and run it again. The model and its validation are in `docs/models/tracking.md`.

## In the window

1. Open the project and save it (the runs go in a folder beside the project file: `cell.dexv` → `cell.runs/`).
2. **Analysis ▸ Runs.** The headline says **tracking off**. Press **Apply** ("Turn tracking on", one undo step), or
   tick *Log every run of this project*. *Choose folder…* keeps the runs elsewhere; *Tags* are added to every run.
3. Run studies as usual. Every long study, designed experiment, optimiser run, tolerance Monte Carlo and scorecard
   that finishes is logged. The quick studies that re-run after each edit are logged only with *Also log the quick
   studies* ticked. A refusal ("Grasp self-occlusion needs a robot…") is logged as a **failed** run with its sentence.
4. The table lists the runs, newest first. Filter by study, tag and date (`since 2026-11-01`, `until 2026-11-30`: a
   date alone means the whole day).
5. Select two or more runs and press **Compare**: each figure side by side, the change from the first selected
   (Δ), which run is best by the figure's direction, and what differs (settings, project content, version). Two
   tolerance Monte Carlo runs also get a significance note from the statistics module (their draws are replicates).
6. **Open result** opens the run's result file; **Re-run** runs its study again on the project as it is now and says
   whether every number reproduces (logged as a new run, tagged `rerun`); **Tags…** and **Note…** annotate runs.
7. **Export runs…** writes the runs shown as CSV and Parquet tables, an MLflow `mlruns/` folder and a W&B replay
   script, with a manifest of SHA-256 digests.

## Worked example: the SO-100 sample, a camera raised 14 cm

    import dexvantage as dv
    from dexvantage.core import tracking as TR

    rig = dv.sample('so100'); rig.save('so100.dexv'); rig = dv.open('so100.dexv')
    rig.factors['setting:tracking.enabled'] = True          # the project's tracking on
    rig.run('coverage')                                      # Task volume seen: 94.5 % (gaps)   -> logged
    z = 'sensor:Third-person camera (sample placement).z'    # 0.36 m
    rig.factors[z] = 0.5
    rig.run('coverage')                                      # Task volume seen: 94.3 % (gaps)   -> logged
    dv.runs(rig)[['study', 'coverage.covered', 'coverage.seen_twice', 'coverage.blind']]

         study  coverage.covered  coverage.seen_twice  coverage.blind
    0  coverage           94.5208              47.3021          5.4792
    1  coverage           94.3403              45.2691          5.6597

    st = TR.store_for(rig.project)
    cmp = TR.compare([st.get(r['run_id']) for r in st.runs()])

| Figure | Run 1 (z 0.36 m) | Run 2 (z 0.50 m) | Δ | Best |
|---|---|---|---|---|
| Task volume seen | 94.52 % | 94.34 % | −0.18 pp | run 1 |
| Seen by two or more cameras | 47.30 % | 45.27 % | −2.03 pp | run 1 |
| Task volume blind | 5.48 % | 5.66 % | +0.18 pp | run 1 |
| Resolution, 5th percentile | 1.044 px/mm | 0.931 px/mm | −0.113 px/mm | run 1 |

What differs: `project.sha256` (the camera's height). Raising the third-person camera loses a little coverage and
resolution here: keep 0.36 m. (Measured 2026-11-07 on the bundled sample; one coverage run takes about 0.5 s.)

`TR.export(st, 'exp')` then writes `exp/runs.csv`, `runs.parquet`, `runs_long.*`, `units.json`, `mlruns/`, `wandb/`
and `export_runs_manifest.json`.

## In batch mode (CI)

    dexvantage cell.dexv --coverage cov.json --check requirements.yaml card.json --track runs/ --track-tag nightly
    dexvantage cell.dexv --runs summary.json --track runs/          # the store summarised
    dexvantage --runs-export exp/ --track runs/ --runs-formats csv,mlflow

A project with tracking turned on logs into its own store without `--track`. Several CI jobs may log into the same
store at once (it is safe for concurrent writers).

## In MLflow and W&B

    MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri exp/mlruns   # MLflow 3.13 and later need the variable
    python exp/wandb/replay_wandb.py --project my-rig                        # offline by default
    wandb sync wandb_offline/wandb/offline-run-*                             # upload later

The replay needs wandb (`pip install "dexvantage[tracking]"` installs it with mlflow-skinny, which reads the store
from Python; `mlflow ui` needs the full mlflow package); DexVantage writes the files without either.

## In Python

    dv.runs('cell.dexv', study='coverage', tag='nightly', since='2026-11-01')   # a DataFrame (units in .attrs)
    dv.track(rig.run('bandwidth'), rig, tags=['by hand'], note='before the hub')   # log one result explicitly
