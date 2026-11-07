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
# # Datasets from Python: synthetic episodes, reading them back, and the audit
#
# On the SO-100 sample: render a small synthetic LeRobot v3.0 dataset with known faults (MuJoCo's offscreen renderer),
# open it with `dv.datasets.open`, look at an episode, and audit it against the plan. The same `dv.datasets.open`
# reads LeRobot v2.1 and v3.0 folders, RLDS folders, ROS 2 bags (MCAP or SQLite) and any format a plugin adds. Run it
# with `python examples/02_datasets_audit.py` or open it in Jupyter through jupytext. The episode settings here are
# small on purpose (two episodes of 24 rows, images 160 pixels wide) so it runs in seconds.

# %%
import os
import tempfile

import dexvantage as dv

OUT = os.environ.get('DEXVANTAGE_EXAMPLE_OUT') or tempfile.mkdtemp(prefix='dexvantage-example-')
rig = dv.sample('so100')
folder = os.path.join(OUT, 'so100_synthetic')

# %% [markdown]
# ## Render the episodes
# Every fault is on with DexVantage's defaults (dropped and duplicated frames, a desync, drift, jitter, an exposure
# change, blur, the gripper hidden, a mount slip, a USB reset); their ground truth is written beside the dataset.

# %%
try:
    made = rig.synth_episodes(folder, episodes=2, frames=24, max_width=160, version='v3.0')
    print(made['episodes'], 'episodes,', made['camera_frames'], 'camera frames,', made['codec'])
except dv.DexVantageError as e:          # no offscreen renderer on this machine: say so and stop here
    made = None
    print('Could not render the episodes:', e)

# %% [markdown]
# ## Read it back
# A `Recording`: its format and version, one row per episode, and an episode's joint-state rows.

# %%
if made:
    ds = dv.datasets.open(folder)
    print(ds)
    episodes = ds.to_frame()
    first = ds.to_frame(episode=0)
    print(len(first), 'rows in episode 0')
    episodes

# %% [markdown]
# ## Audit it against the plan
# The audit study reads the dataset named by its `dataset` setting (for this run only here). Its headline is the share
# of episodes that pass every check; the per-episode table says which checks failed.

# %%
if made:
    audit = rig.run('audit', dataset=folder)
    print(audit.headline['text'])
    table = audit.to_frame('episodes')
    ds.close()
    table
