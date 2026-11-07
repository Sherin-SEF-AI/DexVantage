# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # A research workflow from Python: track, compare, plan, sweep, review and publish
#
# One pass through the research tools on the SO-100 sample, from a notebook or a script: log runs in an experiment
# tracker and export them for MLflow, W&B, CSV and Parquet; test a difference with statistics and size the next
# evaluation with a power analysis; plan the data-collection campaign; run a tolerance sweep on a process pool with
# a shared result cache; compare two cells; review the change as a project diff; and write the publication kit
# (methods paragraph, LaTeX tables, vector figures, BibTeX). Open it in Jupyter with
# `jupytext --to notebook 03_research_workflow.py`, or run it as a script:
# `python examples/03_research_workflow.py`. It needs no window and no network.
#
# The camera placements in the sample are sample placements, not published figures; the device figures come from
# the makers' datasheets (the library entries cite them). The policy success counts in the statistics cell are
# example inputs: put your own evaluation's counts there.

# %%
import os
import tempfile

import dexvantage as dv

OUT = os.environ.get('DEXVANTAGE_EXAMPLE_OUT') or tempfile.mkdtemp(prefix='dexvantage-example-')
os.makedirs(OUT, exist_ok=True)
rig = dv.sample('so100')
rig.save(os.path.join(OUT, 'so100.dexv'))
cam = next(s.name for s in rig.sensors if 'Third-person' in s.name)
raised = rig.with_factors({f'sensor:{cam}.dz': 0.10})          # the third-person camera 100 mm higher
raised.save(os.path.join(OUT, 'so100_raised.dexv'))
rig

# %% [markdown]
# ## Track runs
# `dv.track` logs a result in a run store (here a folder of its own; a saved project keeps one beside it when its
# tracking is on). `dv.runs` reads the runs back as a table, one column per metric.

# %%
store = dv.run_store(os.path.join(OUT, 'runs'))
cov = rig.run('coverage')
cov_raised = raised.run('coverage')
dv.track(cov, rig, store=store, tags=['baseline'])
dv.track(cov_raised, raised, store=store, tags=['camera raised'])
print(cov.headline['text'])
print(cov_raised.headline['text'])
dv.runs(store)

# %% [markdown]
# The same runs for other tools: CSV and Parquet tables, an MLflow `mlruns/` folder (open it with `mlflow ui
# --backend-store-uri`), and a W&B offline replay script.

# %%
exported = dv.export_runs(store, os.path.join(OUT, 'runs_export'))
print({f: v['status'] for f, v in exported['formats'].items()})

# %% [markdown]
# ## Statistics and power
# Two policies' success counts (example inputs) compared with Fisher's exact test: the difference in percentage
# points with its interval, the corrected p-value and a verdict. Then the episodes per policy a power analysis asks
# for to detect that difference (α 0.05, power 0.8, both defaults).

# %%
rates = dv.stats.compare_rates([('Policy A', 14, 20), ('Policy B', 19, 20)])
print(rates.headline['text'])
need = dv.stats.episodes_needed(0.70, 0.95)
print(f'Episodes per policy for 70 % against 95 %: {need["n_per_group"]}')

# %% [markdown]
# ## The data-collection campaign
# Conditions, demonstrations per condition, days and storage, from the project's tasks and rig (every default is
# labelled in the result).

# %%
plan = dv.campaign.plan(rig)
print(plan.headline['text'])

# %% [markdown]
# ## A tolerance sweep on a process pool
# Mount tolerances drawn 16 times, the draws spread over worker processes that share a result cache folder (a Slurm
# array uses the same folder through `dv.cluster.slurm`); the result equals a single process's. Each spawned worker
# imports the main script again, so a script that starts workers keeps all its top-level code under
# `if __name__ == '__main__':` (DexVantage refuses, in one sentence, a pool started outside it). A notebook has no
# script to re-import: there this cell uses two workers, and run as this script it uses one.

# %%
in_script = '__file__' in globals()
mc = dv.cluster.tolerance(rig, jobs=1 if in_script else 2, n=16, metrics=['coverage.covered'],
                          cache_dir=os.path.join(OUT, 'cluster-cache'))
print(mc.headline['text'])

# %% [markdown]
# ## Compare two cells
# The rig as it is against the raised camera, treated as two cells: the figures of merit side by side and the
# geometric gap between them.

# %%
cells = dv.cells.compare(rig, raised)
print(cells.headline['text'])

# %% [markdown]
# ## Review the change
# What changed between the two projects, in words, with the metric deltas; the same as
# `dexvantage so100.dexv --diff so100_raised.dexv`. Saved in the unpacked folder form, a project is a set of small
# JSON files that Git diffs and merges line by line.

# %%
d = dv.diff(rig, raised, metrics=['coverage.covered'])
print(d.text)
raised.save(os.path.join(OUT, 'so100_raised.dexv.d'))
print(sorted(os.listdir(os.path.join(OUT, 'so100_raised.dexv.d')))[:4])

# %% [markdown]
# ## Publish
# The paper kit: a methods paragraph and LaTeX tables in which every number is traced to a result, vector figures
# (SVG and PDF) and the BibTeX of every source the rig depends on. `dv.figure` writes one figure on its own.

# %%
kit = dv.publication.export(rig, os.path.join(OUT, 'paper'), {'coverage': cov})
print(kit['formats']['paper_kit']['status'], len(kit['files']), 'files')
bib = dv.bibtex(rig, os.path.join(OUT, 'refs.bib'))
print(bib.count('\n@'), 'BibTeX entries')
fig = dv.figure(cov, os.path.join(OUT, 'coverage.svg'))
print(os.path.basename(fig['path']), f'{fig["width_mm"]} × {fig["height_mm"]:.1f} mm')
