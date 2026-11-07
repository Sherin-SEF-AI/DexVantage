"""dv.cluster: the cluster sweeps of core/cluster from Python (no Qt on this path).

    import dexvantage as dv
    rig = dv.sample('ur5e')
    res = dv.cluster.experiment(rig, 'design.json', jobs=4)        # a Result: core/experiments' result, runs in a pool
    res = dv.cluster.tolerance(rig, jobs=4, n=256)                  # the tolerance study's result, draws in the pool
    res = dv.cluster.optimise(rig, jobs=4, budget=200)              # the optimiser's, each generation in the pool
    out = dv.cluster.studies([(rig, 'coverage', {}), (other, 'depth', {})], jobs=2)    # [Result or the refusal]
    job = dv.cluster.slurm(rig, 'job/', experiment='design.json', shards=16, mem='8G', time='04:00:00',
                           activate='source <your venv>/bin/activate')                 # a Slurm job folder
    rep = dv.cluster.gather('job/')                                 # the report; rep['result'] when complete
    dv.cluster.status('job/'), dv.cluster.partition(10, 3), dv.cluster.settings()

jobs=None picks min(CPUs, available memory / 2 GiB per worker: a labelled default); jobs=1 runs in this process. The
results are the in-process ones (core/cluster's module docstring says what may differ: how it ran). Each result is
logged in the experiment tracker (core/tracking) when the project's tracking is on, as rig.run's are.

Known limits: those of core/cluster (Slurm only; a shared filesystem; like machines for machine-dependent figures).
"""
from __future__ import annotations

import os


def _project(rig):
    from .api import DexVantageError, Rig
    if isinstance(rig, Rig):
        return rig.project
    from .core.project import Project
    if isinstance(rig, Project):
        return rig
    raise DexVantageError('Pass a dv.Rig (dv.open, dv.sample, dv.new) or a DexVantage Project.')


def _jobs(n):
    """jobs=None (the default count) or a whole number of processes, at least 1; anything else in a sentence."""
    from .api import DexVantageError
    if n is None:
        return None
    import numbers
    if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 1:
        raise DexVantageError(f'jobs is the number of processes, a whole number of at least 1 (jobs=1 runs in this '
                              f'process; leave it out for the default count); it is {n!r}.')
    return int(n)


def _work_project(folder: str):
    from .batch_cluster import _work_project as W
    return W(folder)


def _result(doc, sid, project):
    from .api import Result, _autolog
    _autolog(sid, doc, project)                 # the experiment tracker, when the project's tracking is on
    return Result(doc, sid, project=project)


class _Cluster:
    """dv.cluster: experiment, tolerance, optimise, studies (this machine); slurm, gather, status, run_shard (a
    cluster); partition, settings, DiskResultCache."""

    def experiment(self, rig, design, jobs=None, progress=None, cache_dir=None):
        from .core import cluster as C
        from .core import jobs as J
        p = _project(rig)
        doc = C.experiment(p, os.fspath(design) if isinstance(design, os.PathLike) else design, _jobs(jobs),
                           J.Ctx(progress=progress), cache_dir=cache_dir)
        return _result(doc, 'experiments', p)

    def tolerance(self, rig, jobs=None, progress=None, cache_dir=None, **settings):
        from .core import cluster as C
        from .core import jobs as J
        p = _project(rig)
        return _result(C.tolerance(p, settings or None, _jobs(jobs), J.Ctx(progress=progress), cache_dir=cache_dir),
                       'tolerance', p)

    def optimise(self, rig, jobs=None, progress=None, cache_dir=None, **settings):
        from .core import cluster as C
        from .core import jobs as J
        p = _project(rig)
        doc = C.optimise(p, settings or None, _jobs(jobs), J.Ctx(progress=progress), cache_dir)
        return _result({'schema': 'dexvantage.optimise/1', 'study': 'optimise', **doc}, 'optimise', p)

    def studies(self, job_list, jobs=None, progress=None, cache_dir=None) -> list:
        """[Result, or {'refused': sentence}] per (rig, study, settings) job, in order."""
        from .core import cluster as C
        from .core import jobs as J
        rows = [(_project(j[0]),) + tuple(j[1:]) for j in job_list]
        doc = C.studies(rows, _jobs(jobs), J.Ctx(progress=progress), cache_dir=cache_dir)
        return [(_result(r['doc'], r['study'], rows[k][0]) if 'doc' in r else {'refused': r['refused']})
                for k, r in enumerate(doc['jobs'])]

    def slurm(self, rig, folder, experiment=None, tolerance=None, optimise=None, shards=None, submit=False,
              **slurm_settings) -> dict:
        """A Slurm job folder for one of: experiment=design (a dict, JSON text or a path), tolerance={settings} (or
        True), optimise={settings} (or True: the optimiser's first round). Settings: core/cluster/slurm.DEFAULTS."""
        from .core import cluster as C
        from .core.cluster import slurm as SL
        p = _project(rig)
        given = [k for k, v in (('experiment', experiment), ('tolerance', tolerance), ('optimise', optimise))
                 if v is not None and v is not False]
        if len(given) != 1:
            raise C.ClusterError('Name one of experiment=, tolerance= or optimise= for the job folder.')
        folder = os.fspath(folder)
        if given[0] == 'optimise':
            return C.optimise_slurm(folder, p, optimise if isinstance(optimise, dict) else None, slurm_settings,
                                    shards, submit)
        w = C.experiment_work(p, experiment) if given[0] == 'experiment' else \
            C.tolerance_work(p, tolerance if isinstance(tolerance, dict) else None)
        out = SL.write_job_folder(w, folder, slurm_settings, shards)
        if submit:
            out['job_id'] = SL.submit(folder)
        out = {k: v for k, v in out.items() if k != 'settings'}
        return out

    def gather(self, folder, out=None, write=True, rig=None) -> dict:
        """The gather report; a complete experiment or Monte Carlo is logged in the experiment tracker as batch
        --gather logs it: in rig's project's store when rig is given, else the job folder's base project's."""
        from .core import cluster as C
        rep = C.gather(os.fspath(folder), out, write=write)
        if rep.get('result') is not None and rep.get('kind') in ('experiment', 'tolerance'):
            p = _project(rig) if rig is not None else _work_project(os.fspath(folder))
            if p is not None:
                from .api import _autolog
                _autolog('experiments' if rep['kind'] == 'experiment' else 'tolerance', rep['result'], p)
        return rep

    def status(self, folder) -> dict:
        from .core import cluster as C
        return C.status(os.fspath(folder))

    def run_shard(self, work, shard, of, out=None) -> dict:
        from .core import cluster as C
        return C.run_shard(os.fspath(work), shard, of, out)

    def optimise_continue(self, folder, submit=False) -> dict:
        from .core import cluster as C
        return C.optimise_slurm(os.fspath(folder), submit=submit)

    def partition(self, m: int, n: int) -> list:
        from .core import cluster as C
        return C.partition(m, n)

    def settings(self):
        from .api import Table
        from .core import cluster as C
        return Table(C.slurm_settings())

    @property
    def DiskResultCache(self):
        from .core.cluster import DiskResultCache
        return DiskResultCache

    def __repr__(self) -> str:
        return ('<dv.cluster: experiment, tolerance, optimise, studies (this machine); slurm, gather, status, '
                'run_shard, optimise_continue (Slurm); partition, settings>')


cluster = _Cluster()
