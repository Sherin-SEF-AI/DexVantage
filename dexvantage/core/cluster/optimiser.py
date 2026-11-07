"""The optimiser's generations on several processes or a Slurm cluster (core/optimise/rig.optimise, NSGA-II).

NSGA-II is sequential across generations (each generation's genomes come from the scores of the one before) but the
designs of one generation are independent. rig.optimise calls a prefetch hook with each generation's designs before it
scores them one by one; here that hook scores the designs it has not seen in parallel and puts each score in the
evaluator's memo under the key the evaluator itself uses, so the search then finds every design scored and runs exactly
as it does serially: the same genomes, the same front, the same options (only the search block's counts of studies
scored and reused, and its seconds, differ).

    optimise_local(project, settings, jobs, ctx)          a pool of `jobs` workers for the run's generations
    optimise_slurm(folder, project=None, settings, ...)   one round: replay the search from the designs gathered so far;
                                                          the first generation with designs not yet scored becomes a
                                                          job folder (round-NNN/) for those designs, or, when none is
                                                          left, the result is written (result.json)

On Slurm the search is replayed from the start at every round. That costs nothing but bookkeeping: NSGA-II is seeded,
every design scored before comes from the gathered rounds, and the studies the driver itself runs (the current rig, the
confirm step) come from the shared result cache (cache/). Each round is one array job; with submit, the round's job
and then a continue job (afterany on the array) are submitted, so the rounds follow each other without anyone
watching.

Known limits: a round can only start when the one before is gathered, so a cluster's queue wait is paid once per
generation (population × generations designs, budget / population rounds); the design agent (designer.py) is not
distributed; a design a shard could not score fails the round (its sentence is reported) rather than being scored
as "not stated".
"""
from __future__ import annotations

import json
import os

from .. import jobs
from .work import ClusterError, Work, design_key, designs_work

STATE_SCHEMA = 'dexvantage.cluster-optimise/1'


class NeedDesigns(Exception):
    """The search reached designs no round has scored yet."""

    def __init__(self, base, paths, rows, ev):
        super().__init__(f'{len(rows)} designs to score')
        self.base, self.paths, self.rows, self.ev = base, paths, rows, ev


def _missing(base, factors, rows, ev, known=None):
    """The rows (first of each memo key) the evaluator has not scored; those known are put in its memo."""
    from .. import factors as FA
    from ..optimise import rig
    paths = [f.path for f in factors]
    out, seen = [], set()
    for values in rows:
        k = rig._key(values)
        if k in ev.memo or k in seen:
            continue
        try:
            FA.apply(base, dict(zip(paths, values)))
        except FA.FactorError:                  # not a design (rig.optimise says so itself)
            continue
        seen.add(k)
        rec = (known or {}).get(design_key(values))
        if rec is not None:
            ev.memo[k] = ev.make(rec['figures'], rec['status'], rec['reasons'], rec['bounds'])
            continue
        out.append(list(values))
    return paths, out


def _settings(project, settings):
    from ..optimise import rig
    p = project.snapshot()
    if settings:
        p.analysis.studies[rig.SID] = {**(p.analysis.studies.get(rig.SID) or {}), **dict(settings)}
    return p, rig.settings_of(p)


def optimise_local(project, settings: dict | None = None, jobs_n: int | None = None, ctx: jobs.Ctx | None = None,
                   cache_dir: str | None = None) -> dict:
    """rig.optimise with every generation's new designs scored in a pool of jobs_n workers (default: as many as
    memory allows). The result is the serial one; its 'search' block gains 'cluster' (workers, peak memory)."""
    from ..optimise import rig
    from . import local as L
    ctx = ctx or jobs.Ctx()
    p, st = _settings(project, settings)
    n = L.recommended_workers() if jobs_n is None else max(1, int(jobs_n))
    holder: dict = {}
    sent = [0]

    def prefetch(base, factors, rows, ev):
        paths, todo = _missing(base, factors, rows, ev)
        if not todo or n <= 1:
            return
        if 'pool' not in holder:
            w = designs_work(base, paths, [], ev.objectives, ev.constraints, ev.screen)
            holder['pool'] = L.ItemPool(w, n, cache_dir)
        failed = []

        def on_result(i, status, res, _sec):
            if status == 'ok':
                ev.memo[rig._key(todo[i])] = ev.make(res['figures'], res['status'], res['reasons'], res['bounds'])
            elif status == 'failed':
                failed.append(res)
        tasks = [(i, {'values': v, 'key': design_key(v)}) for i, v in enumerate(todo)]
        sent[0] += len(tasks)
        holder['pool'].map(tasks, on_result, ctx.cancelled)     # a cancel: the search stops at its next design
    try:
        doc = rig.optimise(p, st, ctx, prefetch=prefetch)
    finally:
        pool = holder.get('pool')
        if pool is not None:
            holder['peaks'] = dict(pool.peaks)
            pool.terminate()
    peaks = holder.get('peaks') or {}
    doc['search']['cluster'] = {'workers': n, 'designs_in_pool': sent[0],
                                'peak_rss_mb_per_worker': max(peaks.values()) if peaks else None,
                                'mem_per_worker_mb_planned': float(L.MEM_PER_WORKER_MB)}
    return doc


# ====================================================================== Slurm rounds
def _state(folder):
    try:
        with open(os.path.join(folder, 'optimise.json'), encoding='utf-8') as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) and d.get('schema') == STATE_SCHEMA else None


def _write_state(folder, d):
    from .cache import atomic_write
    atomic_write(os.path.join(folder, 'optimise.json'), (json.dumps(d, indent=1) + '\n').encode('utf-8'))


def _continue_script(st: dict, python: str) -> str:
    from . import slurm as SL
    s = SL.read_settings({k: v for k, v in st.items() if k in SL.DEFAULTS and k not in st.get('defaults', [])})
    lines = ['#!/bin/bash', f'#SBATCH --job-name={s["job_name"]}-next', f'#SBATCH --cpus-per-task={s["cpus_per_task"]}',
             f'#SBATCH --mem={s["mem"]}', f'#SBATCH --time={s["time"]}', '#SBATCH --output=logs/continue-%j.out']
    if s['partition']:
        lines.append(f'#SBATCH --partition={s["partition"]}')
    if s['account']:
        lines.append(f'#SBATCH --account={s["account"]}')
    lines += [f'#SBATCH {x}' for x in s['extra']]
    lines += ['#', '# DexVantage optimiser: gather the round just run and start the next one (or write the result).',
              '', 'set -euo pipefail', 'JOB_DIR="${DEXVANTAGE_JOB_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"', 'cd "$JOB_DIR"']
    lines += s['setup'] + ([s['activate']] if s['activate'] else [])
    lines += [f'exec {s["python"] or python} -m dexvantage --optimise-continue . --submit', '']
    return '\n'.join(lines)


def optimise_slurm(folder: str, project=None, settings: dict | None = None, slurm_settings: dict | None = None,
                   shards: int | None = None, submit: bool = False, ctx: jobs.Ctx | None = None) -> dict:
    """One round of the optimiser on Slurm (module docstring). The first call names the project (and the optimiser
    and Slurm settings) and creates the folder; later calls (--optimise-continue FOLDER) take everything from it.
    Returns {'status': 'round' | 'done', 'round', 'designs', 'folder', 'result_path', 'text', 'job_ids'}."""
    from ..optimise import rig
    from ..project import Project, ProjectFormatError
    from . import slurm as SL
    from .cache import DiskResultCache
    from .shards import gather, write_result
    ctx = ctx or jobs.Ctx()
    state = _state(folder)
    if state is None:
        if project is None:
            raise ClusterError(f'{os.path.basename(os.path.normpath(folder))} is not an optimiser job folder: start '
                               'one with --optimise OUT.json --slurm FOLDER.')
        if os.path.exists(folder) and os.listdir(folder):
            raise ClusterError(f'The folder {os.path.basename(os.path.normpath(folder))} is not empty: choose a new or '
                               'empty folder for the optimiser job.')
        p, st = _settings(project, settings)
        os.makedirs(os.path.join(folder, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(folder, 'cache'), exist_ok=True)
        p.snapshot().save(os.path.join(folder, 'project.dexv'))
        sl = SL.read_settings(slurm_settings)
        state = {'schema': STATE_SCHEMA, 'settings': json.loads(json.dumps(st)), 'slurm': {
            k: v for k, v in sl.items() if k != 'defaults'}, 'slurm_defaults': sl['defaults'], 'shards': shards,
            'rounds': []}
        _write_state(folder, state)
    try:
        p = Project.load(os.path.join(folder, 'project.dexv'))
    except ProjectFormatError as e:
        raise ClusterError(f'The optimiser folder\'s project could not be read: {e}') from None
    known: dict = {}
    for r in state['rounds']:
        rep = gather(os.path.join(folder, r), write=False)
        if not rep['complete']:
            rd = os.path.join(folder, r)
            rp = SL.write_resubmit(rd, rep['redo'], rep['shards'])
            raise ClusterError(f'Round {r} is not complete: {rep["text"]}' +
                               (f' Resubmit {r}/{os.path.basename(rp)}, then continue.' if rp else ''))
        known.update(rep['result']['designs'])

    def prefetch(base, factors, rows, ev):
        paths, todo = _missing(base, factors, rows, ev, known)
        if todo:
            raise NeedDesigns(base, paths, todo, ev)
    cache = DiskResultCache(os.path.join(folder, 'cache'), size=64)
    sl_given = {k: v for k, v in state['slurm'].items() if k not in state.get('slurm_defaults', [])}
    try:
        doc = rig.optimise(p, state['settings'], ctx, prefetch=prefetch, cache=cache)
    except NeedDesigns as nd:
        k = len(state['rounds']) + 1
        name = f'round-{k:03d}'
        w = designs_work(nd.base, nd.paths, nd.rows, nd.ev.objectives, nd.ev.constraints, nd.ev.screen)
        job = SL.write_job_folder(w, os.path.join(folder, name), sl_given, state.get('shards') or None)
        state['rounds'].append(name)
        _write_state(folder, state)
        cont = os.path.join(folder, 'continue.sbatch')
        with open(cont, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write(_continue_script({**state['slurm'], 'defaults': state.get('slurm_defaults', [])}, 'python'))
        os.chmod(cont, 0o755)
        ids = []
        if submit:
            a = SL.submit(os.path.join(folder, name), 'run.sbatch')
            ids = [a, SL.submit(folder, 'continue.sbatch', dependency=a)]
        text = (f'Round {k}: {len(nd.rows)} designs of the next generation in {job["shards"]} shards → {name}/. ' +
                (f'Submitted (array {ids[0]}, then the continue job {ids[1]}).' if ids else
                 f'Submit {name}/run.sbatch from its folder, then run continue.sbatch (or --optimise-continue) after '
                 'it.'))
        return {'status': 'round', 'round': k, 'designs': len(nd.rows), 'folder': os.path.join(folder, name),
                'result_path': None, 'text': text, 'job_ids': ids}
    doc['search']['cluster'] = {'rounds': len(state['rounds']), 'designs_on_cluster': len(known),
                                'cache': cache.stats()}
    doc = {'schema': f'dexvantage.{rig.SID}/1', 'study': rig.SID, **doc}
    out = os.path.join(folder, 'result.json')
    write_result(doc, p, out, 'optimise', option='optimise_continue')
    head = doc.get('headline') or {}
    return {'status': 'done', 'round': len(state['rounds']), 'designs': len(known), 'folder': folder,
            'result_path': out, 'result': doc, 'job_ids': [],
            'text': (f'The optimiser finished after {len(state["rounds"])} rounds → result.json. '
                     + head.get('text', ''))}


__all__ = ['optimise_local', 'optimise_slurm', 'NeedDesigns', 'Work']
