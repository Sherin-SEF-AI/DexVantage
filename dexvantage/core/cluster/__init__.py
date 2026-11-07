"""Cluster sweeps: an experiment's runs, a tolerance Monte Carlo's draws, an optimiser's generations or any list of
(project, study, settings) jobs on this machine's cores or on a Slurm cluster, with the same result as a run in one
process.

    experiment(project, design, jobs=N)          core/experiments' result, the runs in a pool of N processes
    tolerance(project, settings, jobs=N)         the tolerance study's result, the draws in the pool
    optimise(project, settings, jobs=N)          the optimiser's result, each generation's designs in the pool
    studies([(project, study, settings)], jobs)  every job's study result
    work_for(kind, project, ...)                 the Work (core/cluster/work.py) of any of these
    slurm_folder(work, folder, settings, shards) a Slurm job folder (core/cluster/slurm.py)
    run_shard(work.json, i, n, out)              one shard (python -m dexvantage --run-shard ...)
    gather(folder)                               merge the shards (python -m dexvantage --gather FOLDER)
    DiskResultCache(folder)                      the shared on-disk result cache (core/cluster/cache.py)

The model: a work is a list of independent items; shard i of N takes items i, i + N, i + 2N, … (every item exactly
once for any N); each item is computed by the function the in-process run uses on it, on the same project (checked
by its content hash), and the results are put back together by the in-process code's own analysis, so the gathered
result equals the serial one number for number (the validation check cluster.gather_equals_serial and the tests
compare them as canonical JSON). Only what says how it ran differs: an experiment's 'execution' block (mode, workers,
seconds), a study's 'seconds' and 'cached', the optimiser's search counts.

Known limits: PBS, LSF and other schedulers are out of scope (Slurm job arrays only; any scheduler can still run
`--run-shard` lines by hand); the job folder must be on a filesystem every node mounts; a study whose figures depend
on the machine (GPU against CPU occlusion at a silhouette) is identical only between like machines; plugins and the
modules a work names must be installed where the shards run; an optimiser round waits for the round before it (one
queue wait per generation); the design agent is not distributed. Each part's own limits are in its module.
"""
from __future__ import annotations

from .cache import DiskResultCache, atomic_write, prune  # noqa: F401
from .work import (KINDS, ClusterError, State, Work, assemble, execute, experiment_work, partition,  # noqa: F401
                   shard_indices, studies_work, tolerance_work)


def work_for(kind: str, project=None, design=None, settings=None, job_list=None) -> Work:
    """The Work of a kind: 'experiment' (project, design), 'tolerance' (project, settings), 'studies' (job_list)."""
    if kind == 'experiment':
        return experiment_work(project, design)
    if kind == 'tolerance':
        return tolerance_work(project, settings)
    if kind == 'studies':
        return studies_work(job_list)
    raise ClusterError(f'There is no cluster work of kind "{kind}"; one of experiment, tolerance, studies (the '
                       'optimiser runs through optimise()).')


def experiment(project, design, jobs=None, ctx=None, out_dir=None, cache_dir=None) -> dict:
    """The designed experiment's result (core/experiments.run's), its runs in a pool of `jobs` processes."""
    from .local import run_local
    return run_local(experiment_work(project, design), jobs, ctx, out_dir, cache_dir)


def tolerance(project, settings=None, jobs=None, ctx=None, out_dir=None, cache_dir=None) -> dict:
    """The tolerance study's result (core/studies/tolerance's), its draws in a pool of `jobs` processes."""
    from .local import run_local
    return run_local(tolerance_work(project, settings), jobs, ctx, out_dir, cache_dir)


def studies(job_list, jobs=None, ctx=None, out_dir=None, cache_dir=None) -> dict:
    """Every job's study result ({'jobs': [{'study', 'project', 'settings', 'doc' or 'refused'}]})."""
    from .local import run_local
    return run_local(studies_work(job_list), jobs, ctx, out_dir, cache_dir)


def optimise(project, settings=None, jobs=None, ctx=None, cache_dir=None) -> dict:
    """The optimiser's result (core/optimise/rig.optimise's), each generation's designs in a pool."""
    from .optimiser import optimise_local
    return optimise_local(project, settings, jobs, ctx, cache_dir)


def slurm_folder(work: Work, folder: str, settings: dict | None = None, shards: int | None = None) -> dict:
    from .slurm import write_job_folder
    return write_job_folder(work, folder, settings, shards)


def optimise_slurm(folder, project=None, settings=None, slurm_settings=None, shards=None, submit=False, ctx=None):
    from .optimiser import optimise_slurm as fn
    return fn(folder, project, settings, slurm_settings, shards, submit, ctx)


def run_shard(work_path, shard: int, of: int, out_dir: str | None = None, **kw) -> dict:
    from .shards import run_shard as fn
    return fn(work_path, shard, of, out_dir, **kw)


def gather(folder: str, out: str | None = None, **kw) -> dict:
    from .shards import gather as fn
    return fn(folder, out, **kw)


def status(folder: str, **kw) -> dict:
    from .shards import status as fn
    st = fn(folder, **kw)
    st.pop('results', None)
    w = st.pop('work')
    st['title'] = w.describe()
    return st


def slurm_settings() -> list:
    """[{'key', 'default', 'meaning'}] of the Slurm settings (each a labelled DexVantage default)."""
    from .slurm import DEFAULTS, SETTINGS
    return [{'key': k, 'default': DEFAULTS[k], 'meaning': SETTINGS[k]} for k in DEFAULTS]
