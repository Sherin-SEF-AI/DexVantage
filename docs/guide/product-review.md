# A research team's notebook on ALOHA 2

This is one session on the ALOHA 2 sample, end to end, from Python. It goes through:
- two variants and their comparison;
- a Sobol experiment, run in a local pool and as a Slurm array;
- the runs tracked and compared;
- a campaign sized by a power analysis;
- the sim-to-real gap;
- three cells and their pooling;
- the paper kit, and a USD robot.

The figures are from the development machine (2026-11-07). The review that made these steps work is in
[../models/product-review.md](../models/product-review.md).

Put the code under `if __name__ == '__main__':` in a script; a notebook needs no guard. The local pool starts
processes, and a script's top-level code would run again in each of them. DexVantage refuses that in one sentence.

```python
import dexvantage as dv

rig = dv.sample('aloha')
rig.save('aloha.dexv')
rig = dv.open('aloha.dexv')                        # a saved file: its run store sits beside it (aloha.runs/)
rig.factors['setting:tracking.enabled'] = True     # log every run from here on

# 1. two variants: B raises the overhead camera 5 cm
rig.add_variant('Raised high cam', {'sensor:cam_high (overhead).dz': 0.05})
cmp = dv.stats.compare_variants(rig, b='Raised high cam', metrics=['coverage.covered'], n=8)
cmp.headline['text']     # 'Smallest corrected p-value of the comparisons: p = 1 (no clear difference)'
```

ALOHA 2's four cameras see the whole task volume in both variants (100 % at every draw), so the difference is exactly
zero. The page and the result say "no clear difference", never "equal".

```python
# 2. what drives coverage: Sobol indices over two factors, N = 4 (16 runs)
design = {'schema': 'dexvantage.experiment/1', 'name': 'aloha sobol',
          'factors': [{'path': 'sensor:cam_high (overhead).dpitch', 'lo': -5, 'hi': 5},
                      {'path': 'sensor:cam_low (worms-eye).dz', 'lo': -0.03, 'hi': 0.03}],
          'responses': ['coverage.covered'], 'design': {'method': 'saltelli', 'n': 4, 'seed': 1}}
local = dv.cluster.experiment(rig, design, jobs=2)          # 11 s
job = dv.cluster.slurm(rig, 'sobol-job', experiment=design, shards=3, activate='source <your venv>/bin/activate')
# on the cluster: sbatch run.sbatch; here, by hand:
for k in range(3):
    dv.cluster.run_shard('sobol-job/work.json', k, 3)
rep = dv.cluster.gather('sobol-job', rig=rig)                # logged in aloha.runs like the local run
rep['result']['runs'] == local['runs']                       # True
```

```python
# 3. every run so far, and two runs compared with statistics
runs = dv.runs(rig)                                         # stats, experiments, experiments (with their seconds)
a, b = rig.run('tolerance', n=8, seed=3), rig.with_factors({'sensor:cam_high (overhead).dz': 0.05}).run(
    'tolerance', n=8, seed=3)
ids = list(dv.runs(rig, study='tolerance')['run_id'])
dv.stats.compare_runs(rig, ids[0], ids[-1]).headline['text']   # paired by seed, draw by draw
```

The two tolerance runs share the seed, so their draws are paired. Comparing a run with itself is refused.

```python
# 4. how many demonstrations: a power analysis for 60 % against 80 % success
plan = dv.campaign.plan(rig, {'demos': {'mode': 'power', 'p_base': 0.6, 'p_target': 0.8}})
plan.headline['text']        # 'Campaign episodes recorded: 91 (no budget set)': 73.3 GB over 1 day
dv.campaign.export(plan, 'out/aloha-plan')                  # .csv, .ics, .html
```

A misspelt setting, for example `{"power": {...}}`, is refused, with the `demos` form shown. Before this review it
was ignored, and the plan fell back to the entered 50 demonstrations.

```python
# 5. the sim-to-real gap of a recording with known gaps (+20 DN and a 1 px blur on every camera)
rig.gap_dataset('out/aloha-gaps', {'brightness_dn': 20, 'blur_px': 1.0}, episodes=1, frames=20, size=[320, 180])
gap = rig.sim2real('out/aloha-gaps')
gap.headline['text']         # 'Sim-to-real image gap (worst camera): 20.4 DN (very large gap)'
gap['causes'][0]['text']     # 'Every camera differs from its render in the same way (gain 0.99, offset +20 DN): ...'
```

The project has no lights, so Apply sets each camera's render response. With lights, a shared offset still does:
a scale of the lights gives a gain, not an offset.

```python
# 6. three cells and the pooling mapping
cells = dv.cells.compare(rig, 'sample:so100', 'sample:ur5e')
cells.headline['text']       # 'Largest geometric gap to the reference cell: 0.373 index (large gap)'
cells.pooling_mapping()['text']
cells.save_files('out/aloha-cells')

# 7. the paper kit, from the results in hand
kit = dv.publication.export(rig, 'out/aloha-kit', {'rig': rig.run('rig'), 'coverage': rig.run('coverage'),
                                                  'stats': cmp, 'cells': cells})
# open out/aloha-kit/paper_kit/main.tex; methods.tex and tables/*.tex are \input by it

# 8. a robot from USD into a new cell
cell = dv.new('USD import')
arm = dv.robot_io.add_robot(cell, 'ur5e.usda')
```

In the window the same steps are:
- **Requirements and variants ▸ New variant from this one**, then the **Statistics** page's **Variants** source;
- **Designed experiments**, then **Run on…**;
- **Runs**;
- **Campaign**, with *Demonstrations from: Power analysis*;
- **Sim-to-real gap ▸ Dataset…**;
- **Cells ▸ Add a sample**;
- **Publication ▸ Export…**;
- **Robot ▸ Import robot…** for the USD file.
