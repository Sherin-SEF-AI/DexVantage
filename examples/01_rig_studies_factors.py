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
# # A rig from Python: studies, metrics, factors, requirements and export
#
# The UR5e + Robotiq 2F-85 sample (a wrist D405 and an overhead D435), studied from Python with the same code the
# window and batch mode run. Open it in Jupyter with `jupytext --to notebook 01_rig_studies_factors.py`, or run it as a
# script: `python examples/01_rig_studies_factors.py`. It needs no window and no network; with pandas installed
# (`pip install "dexvantage[research]"`) the tables are DataFrames, otherwise lists of dicts.
#
# Camera placements in the sample are sample placements, not published figures; the device figures come from the
# makers' datasheets (the library entries cite them).

# %%
import os
import tempfile

import dexvantage as dv

OUT = os.environ.get('DEXVANTAGE_EXAMPLE_OUT') or tempfile.mkdtemp(prefix='dexvantage-example-')
rig = dv.sample('ur5e')
rig

# %% [markdown]
# ## The rig's sensors
# One row per sensor of the active variant: its device (a library id), the frame it is mounted on and its mount pose
# in that frame (metres and degrees).

# %%
rig.to_frame()

# %% [markdown]
# ## One study
# `run` takes a study id (`dv.studies()` lists them) and settings laid over the stored ones for this run only. The
# result is the study's own JSON (a dict) with a headline figure and its status word.

# %%
cov = rig.run('coverage')
print(cov.headline['text'])
cov.to_frame('cameras', columns=['name', 'seen', 'hidden', 'outside_image', 'ppmm_median'])

# %% [markdown]
# ## A factor sweep
# Factors are the settable quantities of the rig (`rig.factors.paths()` lists them). `with_factors` returns a new rig
# and leaves `rig` as it was, so a sweep is a loop. Here the overhead camera moves up and down by 10 cm and the
# coverage and depth-validity metrics are read for each height.

# %%
cam = next(s.name for s in rig.sensors if 'Overhead' in s.name)
rows = []
for dz in (-0.10, 0.0, 0.10):
    moved = rig.with_factors({f'sensor:{cam}.dz': dz})
    figures = moved.metrics(['coverage.covered', 'depth.valid']).figures()
    rows.append({'dz_m': dz, 'z_m': round(moved.factors[f'sensor:{cam}.z'], 3), **figures})
try:
    import pandas as pd
    sweep = pd.DataFrame(rows)
except ImportError:
    sweep = rows
sweep

# %% [markdown]
# ## Requirements
# A requirement is a metric, a comparison and a threshold. The 90 % here is this example's own figure, not a standard.

# %%
card = rig.check([{'name': 'Task volume seen', 'metric': 'coverage.covered', 'op': '>=', 'threshold': 90}])
print(card['verdict_text'], '·', card['items'][0]['text'])

# %% [markdown]
# ## Export
# The cameras as OpenCV YAML and Kalibr camchain files, with a manifest that lists every file's SHA-256 and the
# provenance block.

# %%
manifest = rig.export(os.path.join(OUT, 'export'), ['opencv_yaml'])
print(manifest['formats']['opencv_yaml']['status'], sorted(manifest['files'])[:3])

# %% [markdown]
# ## Save the rig with the higher overhead camera
# `rig.factors[path] = value` edits the rig itself; `save` writes the .dexv file.

# %%
rig.factors[f'sensor:{cam}.dz'] = 0.10
path = rig.save(os.path.join(OUT, 'ur5e_higher_overhead.dexv'))
print(os.path.basename(path), round(dv.open(path).factors[f'sensor:{cam}.z'], 3))
