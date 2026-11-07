"""This machine: a work's items in a pool of N worker processes (spawned, so each process imports DexVantage afresh
and no thread state or GPU context is inherited), with progress, cancel and the memory each worker used.

    recommended_workers(mem_per_worker_mb)   min(CPUs, available memory / memory per worker), at least 1
    available_memory_mb()                    MemAvailable from /proc/meminfo (None where it cannot be read)
    run_items(work, jobs, ctx, ...)          {index: result}, {index: sentence}, what ran (workers, peak memory)
    run_local(work, jobs, ctx, ...)          the assembled in-process result (core/cluster/work.assemble)
    ItemPool(work, jobs)                     a pool kept open across batches (the optimiser's generations)
    check_main_guard(where)                  ClusterError in one sentence before a pool starts from a script's
                                             top-level code outside `if __name__ == '__main__':` (spawned workers
                                             import the script again)

Memory per worker. Every worker holds its own copy of the project, the robot models the studies load, the studies'
result caches and, for the GPU paths, an EGL context. The planning figure MEM_PER_WORKER_MB (2 048 MiB, a labelled
DexVantage default) is above the peaks measured on the samples (docs/models/cluster.md: a spawned worker's peak
resident memory running coverage and depth on the UR5e sample and the tolerance draws of the G1); a heavier project
(the grasp study on 10 000 grasps, a large dataset) needs more, so the run reports each worker's measured peak
(VmHWM on Linux, else ru_maxrss: shards.peak_rss_mb) and the page shows it. The default number of workers keeps
workers × MEM_PER_WORKER_MB within the memory available when the run starts.

Cancel terminates the workers at once: an item in progress is discarded (as core/experiments discards a run a cancel
interrupts), the items done are kept (and, with out_dir, written, so the next run resumes after them).

Known limits: workers do not share a memory cache (give a shared DiskResultCache folder to share results); spawning a
worker costs the import of DexVantage (about a second) and the robot models, so a pool pays only for items slower than
that; a frozen bundle needs multiprocessing's freeze support (Phase 11).
"""
from __future__ import annotations

import os
import time
import uuid

from .. import jobs
from .work import ITEM_WORD, State, Work, assemble, execute

MEM_PER_WORKER_MB = 2048          # a labelled default: above the measured worker peaks on the samples (module doc)
MAX_WORKERS = 64

_W: dict = {}


def available_memory_mb() -> float | None:
    try:
        with open('/proc/meminfo', encoding='ascii') as fh:
            for line in fh:
                if line.startswith('MemAvailable:'):
                    return float(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


def recommended_workers(mem_per_worker_mb: float = MEM_PER_WORKER_MB) -> int:
    cpus = os.cpu_count() or 1
    avail = available_memory_mb()
    by_mem = cpus if avail is None else int(avail // max(1.0, float(mem_per_worker_mb)))
    return max(1, min(cpus, by_mem, MAX_WORKERS))


def memory_statement(n: int, mem_per_worker_mb: float = MEM_PER_WORKER_MB) -> str:
    avail = available_memory_mb()
    return (f'{n} worker{"" if n == 1 else "s"} × about {mem_per_worker_mb / 1024:.1f} GiB each (a planning default; '
            'the run reports each worker\'s measured peak)' +
            (f'; {avail / 1024:.1f} GiB available now' if avail is not None else '') + '.')


NO_GUARD = ('{where} starts worker processes, and each spawned worker imports your script again, so the script '
            'must make this call under "if __name__ == \'__main__\':" (line {line} of {file} is outside it). A '
            'notebook or "python -m" needs no guard; jobs=1 runs without workers.')


def _guard_line(path: str, line: int) -> bool | None:
    """Is line inside an `if __name__ == '__main__':` block of the script at path? None when it cannot be read."""
    import ast
    try:
        with open(path, encoding='utf-8') as fh:
            tree = ast.parse(fh.read(), path)
    except (OSError, SyntaxError, ValueError, UnicodeDecodeError):
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        t = node.test
        names = []
        if isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq):
            names = [t.left, t.comparators[0]]
        if any(isinstance(x, ast.Name) and x.id == '__name__' for x in names) and \
                any(isinstance(x, ast.Constant) and x.value == '__main__' for x in names):
            if node.body and node.body[0].lineno <= line <= max(getattr(n, 'end_lineno', n.lineno) for n in node.body):
                return True
    return False


def check_main_guard(where: str = 'This run'):
    """Raise ClusterError, in one sentence, before a spawned pool starts from a script's top-level code outside
    `if __name__ == '__main__':` (each worker would run the script again and the pool would fail with Python's own
    traceback). Mirrors multiprocessing.spawn: a main module run by name ('python -m', '__main__' modules) or without a
    file (a notebook, the interactive prompt) is not re-run, so it is not checked."""
    import multiprocessing as mp
    import sys
    from .work import ClusterError
    if mp.parent_process() is not None and getattr(mp.current_process(), '_inheriting', False):
        raise ClusterError(NO_GUARD.format(where=where, line='?', file='the script'))   # a worker re-running it
    main = sys.modules.get('__main__')
    spec = getattr(main, '__spec__', None)
    path = getattr(main, '__file__', None)
    if main is None or not path or getattr(sys, 'frozen', False) or \
            (spec is not None and getattr(spec, 'name', '').endswith('__main__')):
        return                          # not re-run by spawn (a frozen bundle's workers start through freeze_support)
    f = sys._getframe(1)
    while f is not None and not (f.f_globals is main.__dict__ and f.f_code.co_name == '<module>'):
        f = f.f_back
    if f is None:
        return
    if _guard_line(path, f.f_lineno) is False:
        raise ClusterError(NO_GUARD.format(where=where, line=f.f_lineno, file=os.path.basename(path)))


def _init(work, cache_dir):
    from ..memlimit import limit_arenas
    limit_arenas()
    cache = None
    if cache_dir:
        from .cache import DiskResultCache
        cache = DiskResultCache(cache_dir)
    _W.clear()
    _W.update(work=work, state=State(work, cache))


def _task(arg):
    from .shards import peak_rss_mb
    i, item = arg
    status, res, sec = execute(_W['work'], i, _W['state'], item=item)
    return i, status, res, sec, os.getpid(), peak_rss_mb()


class ItemPool:
    """A spawned pool of `jobs` workers for one work, kept open across calls of map() (close() when done)."""

    def __init__(self, work: Work, jobs_n: int, cache_dir: str | None = None):
        import multiprocessing as mp
        self.work = work
        self.n = max(1, int(jobs_n))
        self.peaks: dict = {}
        light = Work(work.kind, [], work.params, work.projects, work.modules, work.title, work.digests)
        self.pool = None
        check_main_guard(f'A pool of {self.n} workers')
        self.pool = mp.get_context('spawn').Pool(self.n, initializer=_init, initargs=(light, cache_dir))

    def map(self, tasks, on_result, cancel=None) -> bool:
        """tasks: [(index, item)] (the workers hold the work without its items); on_result(index, status, result,
        seconds) as each finishes. False when cancelled
        (the pool is then terminated)."""
        import multiprocessing as mp
        it = self.pool.imap_unordered(_task, tasks, chunksize=1)
        left = len(tasks)
        while left:
            if cancel is not None and cancel():
                self.terminate()
                return False
            try:
                i, status, res, sec, pid, peak = it.next(timeout=0.2)
            except mp.TimeoutError:
                continue
            left -= 1
            self.peaks[pid] = max(self.peaks.get(pid, 0.0), peak)
            on_result(i, status, res, sec)
        return True

    def terminate(self):
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        self.terminate() if self.pool is not None else None


def run_items(work: Work, jobs_n: int | None = None, ctx: jobs.Ctx | None = None, out_dir: str | None = None,
              cache_dir: str | None = None, mem_per_worker_mb: float = MEM_PER_WORKER_MB) -> dict:
    """Every item of work on this machine: serially in this process when jobs ≤ 1, else in a pool of that many
    spawned workers (default: recommended_workers). out_dir: write each result there as a shard output (shard 0 of
    1) and skip the items it already holds. Returns {'results', 'failed', 'execution', 'cancelled'}."""
    from .shards import append_line, peak_rss_mb, read_outputs
    ctx = ctx or jobs.Ctx()
    m = len(work.items)
    n = recommended_workers(mem_per_worker_mb) if jobs_n is None else max(1, int(jobs_n))
    n = min(n, max(1, m))
    results, failed = {}, {}
    path = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        results.update(read_outputs(out_dir, work)['results'])
        path = os.path.join(out_dir, f'shard-00000-of-00001.{uuid.uuid4().hex[:12]}.jsonl')
        append_line(path, {'type': 'header', 'schema': 'dexvantage.cluster-shard/1', 'work_id': work.work_id,
                           'kind': work.kind, 'shard': 0, 'of': 1, 'items': m, 'todo': m - len(results)})
    todo = [i for i in range(m) if i not in results]
    word = ITEM_WORD[work.kind]
    t0 = time.perf_counter()
    done = [len(results)]

    def on_result(i, status, res, sec):
        if status == 'ok':
            results[i] = res
        elif status == 'failed':
            failed[i] = res
        if path and status in ('ok', 'failed'):
            append_line(path, {'type': 'item', 'index': i, 'status': status, 'seconds': sec,
                               ('result' if status == 'ok' else 'error'): res})
        done[0] += 1
        ctx.report(done[0] / max(1, m), f'{word} {done[0]} of {m}' + (f' ({n} processes)' if n > 1 else ''))
    peaks: dict = {}
    cancelled = False
    if n <= 1 or len(todo) < 2:
        cache = None
        if cache_dir:
            from .cache import DiskResultCache
            cache = DiskResultCache(cache_dir)
        state = State(work, cache)
        for i in todo:
            if ctx.cancelled():
                cancelled = True
                break
            status, res, sec = execute(work, i, state, ctx.cancelled)
            if status == 'cancelled':
                cancelled = True
                break
            on_result(i, status, res, sec)
        peaks[os.getpid()] = peak_rss_mb()
        n = 1
    else:
        pool = ItemPool(work, n, cache_dir)
        try:
            cancelled = not pool.map([(i, work.items[i]) for i in todo], on_result, ctx.cancelled)
        finally:
            peaks = dict(pool.peaks)
            pool.terminate() if cancelled else pool.close()
    secs = round(time.perf_counter() - t0, 3)
    if path and not cancelled:
        append_line(path, {'type': 'end', 'done': len(results), 'failed': len(failed),
                           'peak_rss_mb': max(peaks.values()) if peaks else None, 'seconds': secs})
    ex = {'workers': n, 'mode': ('serial (core/cluster)' if n == 1 else f'process pool of {n} (core/cluster)'),
          'seconds': secs, 'peak_rss_mb_per_worker': max(peaks.values()) if peaks else None,
          'mem_per_worker_mb_planned': float(mem_per_worker_mb)}
    return {'results': results, 'failed': failed, 'execution': ex, 'cancelled': cancelled}


def run_local(work: Work, jobs_n: int | None = None, ctx: jobs.Ctx | None = None, out_dir: str | None = None,
              cache_dir: str | None = None, mem_per_worker_mb: float = MEM_PER_WORKER_MB) -> dict:
    """run_items, then the assembled in-process result. ClusterError when an item failed (its sentence) or the run
    was cancelled before every item had a result (with out_dir the results so far are kept for the next run)."""
    from .work import ClusterError
    r = run_items(work, jobs_n, ctx, out_dir, cache_dir, mem_per_worker_mb)
    if r['failed']:
        k = min(r['failed'])
        raise ClusterError(f'{len(r["failed"])} of {len(work.items)} {ITEM_WORD[work.kind]}s failed; the first: '
                           f'{r["failed"][k]}')
    if r['cancelled']:
        raise ClusterError(f'Cancelled after {len(r["results"])} of {len(work.items)} {ITEM_WORD[work.kind]}s' +
                           (': run again to resume from the results kept.' if out_dir else '.'))
    return assemble(work, r['results'], r['execution'])
