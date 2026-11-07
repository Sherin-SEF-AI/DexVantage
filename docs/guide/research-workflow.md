# A research workflow from Python

`examples/03_research_workflow.py` goes once through the research tools on the SO-100 sample: the experiment tracker,
statistics and power, the campaign planner, a tolerance sweep on a process pool, the multi-cell comparison, the project
diff and the publication kit. Run it as a script (about 4 s and 0.8 GB here, the result cache warm) or open it as a
notebook with `jupytext --to notebook examples/03_research_workflow.py`; `DEXVANTAGE_EXAMPLE_OUT` chooses where it
writes (default: a new temporary folder). `tests/test_completeness_api.py` runs it.

## 1. Two rigs

    import dexvantage as dv
    rig = dv.sample('so100')
    cam = next(s.name for s in rig.sensors if 'Third-person' in s.name)
    raised = rig.with_factors({f'sensor:{cam}.dz': 0.10})     # the third-person camera 100 mm higher

## 2. Track and export the runs

    store = dv.run_store('out/runs')
    dv.track(rig.run('coverage'), rig, store=store, tags=['baseline'])
    dv.track(raised.run('coverage'), raised, store=store, tags=['camera raised'])
    dv.runs(store)                                     # one row per run, one column per metric
    dv.export_runs(store, 'out/runs_export')           # runs.csv, runs.parquet, mlruns/, wandb/

The two headlines read "Task volume seen: 94.5 % (gaps)" and "Task volume seen: 94.3 % (gaps)". The export says
`{'csv': 'ok', 'parquet': 'ok', 'mlflow': 'ok', 'wandb': 'ok'}`; `mlflow ui --backend-store-uri out/runs_export/mlruns`
shows the runs (MLflow is an optional extra), and `wandb/replay_wandb.py` replays them into W&B offline.

## 3. Statistics and power

    dv.stats.compare_rates([('Policy A', 14, 20), ('Policy B', 19, 20)]).headline['text']
    # 'Smallest corrected p-value of the comparisons: p = 0.0915 (no clear difference)'
    dv.stats.episodes_needed(0.70, 0.95)['n_per_group']     # 39

The success counts are example inputs: use your evaluation's. α 0.05 and power 0.8 are defaults.

## 4. The campaign

    dv.campaign.plan(rig).headline['text']       # 'Campaign episodes recorded: 50 (no budget set)'

## 5. A sweep on a process pool

    mc = dv.cluster.tolerance(rig, jobs=2, n=16, metrics=['coverage.covered'], cache_dir='out/cluster-cache')

"Monte Carlo tolerances: P5 94.4 % (no requirement). Task volume seen: 94.56 % ± 0.116 % (σ over 16 draws) …". The
result equals a single process's. Spawned workers import the main script again, so a script that starts workers keeps
its top-level code under `if __name__ == '__main__':`; DexVantage refuses a pool started outside it with one sentence
("A pool of 2 workers starts worker processes, and each spawned worker imports your script again, so the script must
make this call under "if __name__ == '__main__':" (line 3 of sweep.py is outside it). …"). A notebook needs no guard;
run as a script, the example uses one worker.

## 6. Two cells, and the change reviewed

    dv.cells.compare(rig, raised).headline['text']   # 'Largest geometric gap to the reference cell: 0.0209 index (small gap)'
    d = dv.diff(rig, raised, metrics=['coverage.covered'])
    print(d.text)
    # sensors  changed  Sensor Third-person camera (sample placement): mount pose moved 100.0 mm (x +0.0, y +0.0, z +100.0 mm) ...
    #   Task volume seen  94.52 % → 94.32 %  -0.2014 % (worse)
    d.save('out/diff.json'); open('out/diff.html', 'w').write(d.html)
    m = dv.merge(base, ours, theirs)                  # m.rig, or dv.MergeConflict listing each conflict
    raised.save('out/so100_raised.dexv.d')            # the Git-friendly folder form

## 7. Publish

    dv.publication.export(rig, 'out/paper', {'coverage': cov})   # main.tex, methods.tex, tables/, figures/, refs.bib
    dv.bibtex(rig, 'out/refs.bib')                                # every source the rig depends on
    dv.figure(cov, 'out/coverage.svg')                            # 88.9 mm wide (one IEEE column); .pdf works too

## Plugins with validation checks

A plugin that adds a study can now add its validation check too (`api.check`, `docs/guide/plugins.md`): it runs in
`--validate` under the plugin's own model and cites its references in the report.

## Known limits

See `docs/models/research-api.md`: the functions are thin layers over the modules batch mode uses and keep their
limits; the main-guard check sees only pools started from a script's own top-level code.
