# Cluster sweeps: a worked example on the SO-100 sample

This guide shows two ways to make a long study faster. A tolerance Monte Carlo or a designed experiment runs the same
studies many times, so you can spread those runs:
- over this machine's cores (**Run on… ▸ This machine**, or `--jobs N`);
- over a Slurm cluster (**Run on… ▸ Write a Slurm job folder**, or `--slurm FOLDER`).

The result is the one a single process gives. The model and its validation are in
[../models/cluster.md](../models/cluster.md). The figures below were measured on the development machine (24 CPUs)
on 2026-11-07.

## 1. On this machine

1. Save the sample: **File ▸ Open sample ▸ SO-100**, then **File ▸ Save as…** `so100.dexv`. A saved file keeps its
   sensor ids. `--sample` makes new ids each time, so two `--sample` runs label their sensors differently.
2. Open **Analysis ▸ Tolerances** and press **Run on…**.
3. Leave **This machine** selected. The dialog proposes a number of processes: the CPUs, within the memory available
   at about 2 GiB per worker. That is a planning default; the run reports each worker's measured peak.
4. Press **Run**. The job bar shows progress and **Cancel** stops the workers at once. The result arrives on the page
   as a normal run, and **Export…** writes it with provenance.

The same run in batch mode:

    dexvantage so100.dexv --tolerance-mc tol.json --jobs 4
    Monte Carlo tolerances (64 draws): the nominal rig and 64 draws on 4 processes in 7.9 s, peak 560 MB per worker.

The same run with `--jobs 1` took 26.9 s, and the result was the same, number for number, as the run without
`--jobs`. A designed experiment works the same way:
- with a design file: `--experiment DESIGN.json --jobs N`;
- with the experiment saved in the project: `--experiment-saved OUT.json --jobs N`.

The optimiser scores each generation's designs in the pool (`--optimise OUT.json --jobs N`). Its search and its
options are the serial ones.

Any list of studies on any projects can run the same way:

    [{"project": "cell_a.dexv", "study": "coverage"},
     {"project": "cell_b.dexv", "study": "coverage", "settings": {"cube_n": 256}}]

    dexvantage --cluster-studies jobs.json --jobs 4          → jobs.results.json

## 2. On a Slurm cluster

1. Choose a folder on the cluster's shared filesystem, for example `tol-job`.
2. Write the job folder:

       dexvantage so100.dexv --tolerance-mc tol.json --slurm tol-job --shards 16 \
           --slurm-settings '{"mem": "4G", "time": "00:30:00", "partition": "cpu",
                              "activate": "source <your venv>/bin/activate"}'

   The **Run on…** dialog writes the same folder: pick **Write a Slurm job folder** and fill in the settings. CPUs (1),
   memory (4G), time (2 hours) and no GPU are DexVantage defaults, not your site's. Set them for your cluster;
   `job.json` records which ones you left at their default. The activation line has no default: write the line that
   makes `python -m dexvantage` work on a compute node.

3. On the cluster, from the folder:

       cd tol-job
       sbatch run.sbatch                     # 16 array tasks, logs in logs/shard-<job>_<task>.out
       python -m dexvantage --gather .      # → result.json, or which shards are still to do

4. If some tasks hit their time limit or a node failed, gather says which shards are left. It writes
   `resubmit.sbatch` for just those (here a 4-run experiment in 2 shards with shard 1 not yet run):

       3 of 5 runs done; of 2 shards, 1 missing (1). Submit resubmit.sbatch to run just those.
       sbatch resubmit.sbatch
       python -m dexvantage --gather .

   A resubmitted shard skips the draws it finished before.

5. Back on your machine, **Run on… ▸ Gather a job folder…** reads the folder into the page.

`--submit` also calls `sbatch` for you (on a login node). Each task stores its study results in the folder's `cache/`
(safe for many writers at once, without file locks), so a resubmitted or repeated job reuses them.

## 3. The optimiser on Slurm

The generations of NSGA-II follow each other, so the optimiser runs on Slurm in rounds, one array job per generation:

    dexvantage so100.dexv --optimise opt.json --slurm opt-job --shards 2 --submit \
        --optimise-settings '{"budget": 12, "population": 4, "objectives": ["coverage.covered"]}'
    Round 1: 3 designs of the next generation in 2 shards → round-001/. Submitted (array …, then the continue job …).

The continue job waits for the round to end (`afterany`). It then gathers the round and submits the next one, until
the budget is spent and `opt-job/result.json` is written. Without `--submit`, run the round yourself, then:

    dexvantage --optimise-continue opt-job

With these settings the search took 3 rounds on the SO-100 sample and gave the serial options (the first round's
three designs are the initial population of four less the current rig, which the driver scores itself).

## 4. From Python

    import dexvantage as dv
    rig = dv.open('so100.dexv')
    res = dv.cluster.tolerance(rig, jobs=4)                         # a dv.Result
    job = dv.cluster.slurm(rig, 'tol-job', tolerance=True, shards=16, mem='4G', activate='module load python')
    rep = dv.cluster.gather('tol-job')                              # rep['complete'], rep['result'], rep['text']
    rep = dv.cluster.gather('tol-job', rig=rig)                     # also logs a complete result in rig's run store
    dv.cluster.settings()                                           # the Slurm settings and their defaults

## What to watch

- **The job folder** must be on a filesystem every node mounts.
- **Machine-dependent figures** are identical only between like machines (GPU and CPU occlusion can differ at a
  silhouette).
- **Long-lived processes:** a window or notebook that has run other studies can differ from fresh shards in the last
  digits of a coverage figure (the model document's known limits say why).
- **Plugins** must be installed where the shards run.
- **PBS and LSF** are not supported. Their jobs can still run the `--run-shard` lines by hand.
