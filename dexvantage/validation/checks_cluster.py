"""Checks of the cluster sweeps (core/cluster) against independent references.

- cluster.gather_equals_serial: a designed experiment (a 6-run Latin hypercube over the SO-100 sample's third-person
  camera pitch and height, the coverage figure) and a tolerance Monte Carlo (6 draws of the sample's mount tolerances,
  the coverage figure), each split into 3 shards that run separately (in the full run as `python -m dexvantage
  --run-shard` processes; in the quick run in this process with a fresh state and result cache per shard, as a new
  process starts) and gathered, against the same design and seed run serially in process (core/experiments.run,
  core/jobs.run_study('tolerance')). Reported: the number of numbers and strings that differ between the two results'
  canonical JSON, leaving out only what says how it ran (an experiment's execution block, a study's seconds and
  cached flag). Tolerance 0: the shards compute each item with the serial code on the same project and seed, so any
  difference is a bug.
- cluster.partition_exact: shard i of N takes items i, i + N, …; for every item count m from 0 to 300 and every N from 1
  to 40, the shards' indices together must be range(m) with each index once. Reported: the number of (m, N) pairs that
  fail; tolerance 0 (a count).
- cluster.cache_concurrent_writers: three processes write the same 24 keys of the shared disk cache (each payload about
  200 kB) at the same time while each also reads every key back. Reported: reads that returned a payload other than
  the one written (a torn or mixed file) plus files left torn at the end; tolerance 0. A reader may see no file yet (a
  miss), never a part of one.
- cluster.sbatch_directives: the array script's #SBATCH lines, read as Slurm reads them (up to the first command),
  against the settings given (array range and throttle, CPUs, memory, time, partition, account, output pattern), and
  bash -n on the script when bash is installed. Reported: mismatches; tolerance 0.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

from .suite import Outcome, check

CAM = 'sensor:Third-person camera (sample placement)'


def leaf_diffs(a, b, path='') -> list:
    """The paths at which two JSON values differ (leaves compared by their canonical JSON)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f'{path}/{k}')
            else:
                out += leaf_diffs(a[k], b[k], f'{path}/{k}')
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f'{path} (length)']
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += leaf_diffs(x, y, f'{path}/{i}')
        return out
    same = json.dumps(a, sort_keys=True, allow_nan=True) == json.dumps(b, sort_keys=True, allow_nan=True)
    return [] if same else [path or '/']


HOW_IT_RAN = ('execution', 'seconds', 'cached')


def neutral(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k not in HOW_IT_RAN}


def run_shards(folder: str, of: int, separate: bool) -> None:
    from ..core import cluster as C
    for i in range(of):
        if separate:
            env = dict(os.environ)
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            env['PYTHONPATH'] = root + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
            r = subprocess.run([sys.executable, '-m', 'dexvantage', '--run-shard', os.path.join(folder, 'work.json'),
                                '--shard', str(i), '--of', str(of)], capture_output=True, text=True, timeout=600,
                               env=env)
            if r.returncode != 0:
                raise RuntimeError(f'shard {i} exited {r.returncode}: {r.stderr.strip()[-300:]}')
        else:                                 # in this process, from a cold occlusion cache as a new process
            from ..core import occlusion as OC
            OC.CACHE.clear()
            C.run_shard(os.path.join(folder, 'work.json'), i, of)


def experiment_design() -> dict:
    return {'factors': [{'path': CAM + '.pitch', 'lo': -6, 'hi': 6}, {'path': CAM + '.z', 'lo': 0.3, 'hi': 0.45}],
            'responses': ['coverage.covered'], 'design': {'method': 'lhs', 'n': 6, 'seed': 5},
            'analysis': {'bootstrap': 50}}


@check('cluster.gather_equals_serial', 'cluster', 'Gathered shards equal the serial run',
       'numbers and strings that differ between the gathered result of 3 shards and the serial in-process result '
       '(a 6-run Latin hypercube and a 6-draw tolerance Monte Carlo on the SO-100 sample), leaving out only how it '
       'ran (execution, seconds, cached)',
       'the same design and seed run serially in one process (core/experiments.run, the tolerance study)', 0,
       'values', refs=('slurm_job_array',), cost='study')
def _gather_equals_serial(ctx):
    from ..core import cluster as C
    from ..core import experiments as EX
    from ..core import jobs
    from ..core.robot.samples import sample_project
    p = sample_project('so100')
    tmp = ctx.tmp()
    separate = not ctx.quick
    diffs = {}
    d = experiment_design()
    ser = EX.run(p, d)
    w = C.experiment_work(p, d)
    C.slurm_folder(w, os.path.join(tmp, 'cluster-exp'), shards=3)
    run_shards(os.path.join(tmp, 'cluster-exp'), 3, separate)
    got = C.gather(os.path.join(tmp, 'cluster-exp'))['result']
    diffs['experiment'] = leaf_diffs(neutral({k: v for k, v in got.items() if k not in ('schema', 'study')}),
                                     neutral(ser))
    q = p.snapshot()
    q.analysis.studies['tolerance'] = {'n': 6, 'seed': 3, 'metrics': ['coverage.covered']}
    ser_t = jobs.run_study('tolerance', q.snapshot())
    w = C.tolerance_work(q)
    C.slurm_folder(w, os.path.join(tmp, 'cluster-tol'), shards=3)
    run_shards(os.path.join(tmp, 'cluster-tol'), 3, separate)
    got_t = C.gather(os.path.join(tmp, 'cluster-tol'))['result']
    diffs['tolerance'] = leaf_diffs(neutral(got_t), neutral(ser_t))
    n = sum(len(v) for v in diffs.values())
    return Outcome(float(n), {'differing_paths': {k: v[:10] for k, v in diffs.items()},
                              'shards_in_separate_processes': separate, 'runs': len(ser['runs']),
                              'draws': ser_t['done'],
                              'best_run': (ser.get('best') or {}).get('run'),
                              'tolerance_p5': ser_t['metrics']['coverage.covered']['p5']})


@check('cluster.partition_exact', 'cluster', 'Shards cover every item exactly once',
       '(item count, shard count) pairs, m = 0 … 300 and N = 1 … 40, whose shards do not hold every item exactly once',
       'range(m): every index once (by construction of the definition, counted here)', 0, 'pairs')
def _partition(ctx):
    from ..core.cluster import partition
    bad = []
    for m in range(0, 301):
        for n in range(1, 41):
            flat = [i for s in partition(m, n) for i in s]
            if sorted(flat) != list(range(m)) or len(flat) != m:
                bad.append((m, n))
    return Outcome(float(len(bad)), {'pairs_checked': 301 * 40, 'first_failures': bad[:5]})


def _writer(folder, keys, rounds, seed, q):
    """One concurrent writer: put every key `rounds` times, reading every key back each round."""
    import random

    from ..core.cluster.cache import DiskResultCache, read_entry
    rnd = random.Random(seed)
    bad = 0
    for _r in range(rounds):
        order = list(keys)
        rnd.shuffle(order)
        c = DiskResultCache(folder, size=1)
        for k in order:
            c.put(k, payload_of(k))
            for k2 in keys:
                got = read_entry(c.path(k2), k2)
                if got is not None and got != payload_of(k2):
                    bad += 1
    q.put(bad)


def payload_of(key) -> dict:
    """The deterministic payload of a key (about 200 kB of JSON)."""
    n = int(key[1])
    return {'key': list(key), 'values': [n * 1e-3 + j for j in range(12000)], 'text': f'entry {n}' * 200}


def concurrent_writers(folder: str, writers: int = 3, n_keys: int = 24, rounds: int = 2) -> dict:
    import multiprocessing as mp

    from ..core.cluster.cache import TMP_SUFFIX, read_entry
    keys = [('cluster_check', str(k), 'p', 's', 'l', 0, '') for k in range(n_keys)]
    ctxm = mp.get_context('spawn')
    q = ctxm.Queue()
    ps = [ctxm.Process(target=_writer, args=(folder, keys, rounds, s, q)) for s in range(writers)]
    for p in ps:
        p.start()
    bad_reads = sum(q.get(timeout=300) for _p in ps)
    for p in ps:
        p.join(timeout=60)
    from ..core.cluster.cache import key_digest
    torn = 0
    for k in keys:
        d = key_digest(k)
        if read_entry(os.path.join(folder, d[:2], d + '.json'), k) != payload_of(k):
            torn += 1
    left = sum(1 for _r, _d, fs in os.walk(folder) for f in fs if f.endswith(TMP_SUFFIX))
    return {'bad_reads': bad_reads, 'torn_files': torn, 'temporary_files_left': left, 'writers': writers,
            'keys': n_keys, 'exit_codes': [p.exitcode for p in ps]}


@check('cluster.cache_concurrent_writers', 'cluster', 'Shared cache: no torn files under concurrent writers',
       'reads that returned a payload other than the one written, plus torn files at the end, with three processes '
       'writing and reading the same 24 keys (about 200 kB each) at once',
       'the payload each key was written with (deterministic), compared in full', 0, 'reads',
       refs=('rename2', 'nfs5', 'flock2'), cost='study')
def _cache(ctx):
    folder = os.path.join(ctx.tmp(), 'cluster-cache')
    r = concurrent_writers(folder, rounds=1 if ctx.quick else 2)
    return Outcome(float(r['bad_reads'] + r['torn_files']), r, ok=r['temporary_files_left'] == 0 and
                   all(c == 0 for c in r['exit_codes']))


@check('cluster.sbatch_directives', 'cluster', 'Slurm array script: its #SBATCH lines and bash syntax',
       'mismatches between the #SBATCH options Slurm would read from the script and the settings given, plus a bash '
       '-n failure', 'the settings given, and bash\'s own parser', 0, 'mismatches', refs=('slurm_sbatch',
                                                                                        'slurm_job_array'))
def _sbatch(ctx):
    from ..core.cluster import slurm as SL
    st = SL.read_settings({'cpus_per_task': 4, 'mem': '6G', 'time': '1-02:00:00', 'partition': 'cpu', 'account': 'lab',
                           'throttle': 8, 'activate': 'source venv/bin/activate', 'extra': ['--qos=short']})
    text = SL.sbatch_script(st, 40)
    got = dict(SL.directives(text))
    want = {'--array': '0-39%8', '--cpus-per-task': '4', '--mem': '6G', '--time': '1-02:00:00', '--partition': 'cpu',
            '--account': 'lab', '--output': 'logs/shard-%A_%a.out', '--job-name': 'dexvantage', '--qos': 'short'}
    bad = [k for k in want if got.get(k) != want[k]] + [k for k in got if k not in want]
    bash = shutil.which('bash')
    syntax = None
    if bash:
        path = os.path.join(ctx.tmp(), 'cluster-check.sbatch')
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(text)
        syntax = subprocess.run([bash, '-n', path], capture_output=True, text=True).returncode
        if syntax != 0:
            bad.append('bash -n')
    return Outcome(float(len(bad)), {'directives': got, 'mismatches': bad, 'bash_n': syntax})
