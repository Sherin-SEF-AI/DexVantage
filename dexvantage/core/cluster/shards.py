"""Shards: one Slurm array task (or any process) runs the items of one shard and appends their results to its own
output file; gather reads every output, says which shards are missing, incomplete or failed, writes a resubmit script
for just those, and, when every item has a result, assembles the in-process result.

    run_shard(work_path, shard, of, out_dir)     python -m dexvantage --run-shard work.json --shard i --of N --out DIR
    read_outputs(out_dir, work)                  {index: result}, {index: sentence}, per-shard counts
    gather(folder, out=None)                     the report (and the result when complete)

An output file is `shard-<i>-of-<N>.<attempt id>.jsonl`: one JSON object per line, a header (the work id, the shard, its
item count), then one line per item ({index, status, result or error, seconds}), then an end line. Each attempt (a
resubmitted or duplicated task) writes its own file, so two attempts of one shard never write to the same file; each
line is written with one write, flushed and fsynced, and the file is closed after every line (NFS's close-to-open
consistency then shows it to a reader on another node). A line cut short by a task that was killed is not valid JSON
and is skipped; the item is simply not done. An attempt first reads the shard's earlier attempts and skips the items
they finished, so a resubmitted shard does only what is left.

Known limits: gather trusts that items are deterministic (two attempts' results for one item are taken to be the same;
the first read is used); the attempt files are kept (delete the out folder to start again); the output of a 'studies'
work holds each study's whole result, which can be megabytes per job.
"""
from __future__ import annotations

import glob
import json
import os
import re
import time
import uuid
from collections import Counter

from .work import ITEM_WORD, ClusterError, State, Work, assemble, execute, shard_indices

SHARD_SCHEMA = 'dexvantage.cluster-shard/1'
REPORT_SCHEMA = 'dexvantage.cluster-gather/1'
NAME = re.compile(r'^shard-(\d+)-of-(\d+)\.([0-9a-f]+)\.jsonl$')


def peak_rss_mb() -> float:
    """This process's peak resident memory in MB: the high-water mark of its own address space (VmHWM in
    /proc/self/status). ru_maxrss keeps, across the exec that starts a spawned worker, the peak of the process it was
    forked from, so a worker spawned by a large parent (a test run, the window) reported the parent's size (measured
    2918 MB for a worker whose own peak is about 500 MB); it is read (KiB on Linux) only when /proc does not say.
    NaN when neither says."""
    try:
        with open('/proc/self/status', encoding='ascii') as fh:
            for line in fh:
                if line.startswith('VmHWM:'):
                    return round(int(line.split()[1]) / 1024, 1)
    except (OSError, ValueError, IndexError):
        pass
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except (ImportError, OSError):
        return float('nan')


def append_line(path: str, rec: dict) -> None:
    line = (json.dumps(rec, ensure_ascii=False, allow_nan=True, separators=(',', ':')) + '\n').encode('utf-8')
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        n = os.write(fd, line)
        while n < len(line):        # a short write (a full disk, a signal, an NFS server): finish the line, so the
            n += os.write(fd, line[n:])     # next one does not start inside it (both would be lost)
        os.fsync(fd)
    finally:
        os.close(fd)


def read_lines(path: str):
    """The complete JSON objects of an output file (a line cut short is skipped)."""
    try:
        with open(path, 'rb') as fh:
            data = fh.read()
    except OSError:
        return
    for raw in data.split(b'\n'):
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw)
        except ValueError:
            continue
        if isinstance(rec, dict):
            yield rec


def shard_files(out_dir: str) -> list:
    return sorted(glob.glob(os.path.join(out_dir, 'shard-*-of-*.*.jsonl')))


def read_outputs(out_dir: str, work: Work) -> dict:
    """What the output folder holds for this work: {'results': {index: result}, 'failed': {index: sentence},
    'shards': {(i, n): {'attempts', 'ended', 'peak_rss_mb', 'seconds'}}, 'foreign': files of another work}."""
    wid = work.work_id
    results, failed, shards, foreign = {}, {}, {}, []
    for f in shard_files(out_dir):
        m = NAME.match(os.path.basename(f))
        if not m:
            continue
        i, n = int(m.group(1)), int(m.group(2))
        recs = list(read_lines(f))
        head = next((r for r in recs if r.get('type') == 'header'), None)
        if head is None or head.get('work_id') != wid:
            if head is not None:
                foreign.append(os.path.basename(f))
            continue
        s = shards.setdefault((i, n), {'attempts': 0, 'ended': False, 'peak_rss_mb': None, 'seconds': 0.0})
        s['attempts'] += 1
        for r in recs:
            if r.get('type') == 'item':
                k = r.get('index')
                if not isinstance(k, int) or not 0 <= k < len(work.items):
                    continue
                if r.get('status') == 'ok' and k not in results:
                    results[k] = r.get('result')
                    failed.pop(k, None)
                elif r.get('status') == 'failed' and k not in results:
                    failed[k] = str(r.get('error') or 'failed')
            elif r.get('type') == 'end':
                s['ended'] = True
                if r.get('peak_rss_mb') is not None:
                    s['peak_rss_mb'] = max(s['peak_rss_mb'] or 0.0, float(r['peak_rss_mb']))
                s['seconds'] += float(r.get('seconds') or 0.0)
    return {'results': results, 'failed': failed, 'shards': shards, 'foreign': foreign}


def run_shard(work_path, shard: int, of: int, out_dir: str | None = None, cache_dir: str | None = None,
              progress=None, cancel=None) -> dict:
    """Run the items of shard `shard` of `of` and append their results to a new attempt file in out_dir (default:
    the work folder's out/). Items earlier attempts finished are skipped. The summary: done, failed, skipped, the
    output file, peak memory. cache_dir: a DiskResultCache folder (default: the work's 'cache_dir' parameter, else
    the job folder's cache/ when it exists)."""
    work = work_path if isinstance(work_path, Work) else Work.load(work_path)
    idx = shard_indices(len(work.items), shard, of)
    out_dir = out_dir or os.path.join(work.folder or '.', 'out')
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        raise ClusterError(f'Could not create the output folder {out_dir}: {e.strerror or e}.') from None
    have = read_outputs(out_dir, work)['results']
    todo = [i for i in idx if i not in have]
    if cache_dir is None and work.folder and os.path.isdir(os.path.join(work.folder, 'cache')):
        cache_dir = os.path.join(work.folder, 'cache')
    cache = None
    if cache_dir:
        from .cache import DiskResultCache
        cache = DiskResultCache(cache_dir)
    t0 = time.perf_counter()
    path = os.path.join(out_dir, f'shard-{shard:05d}-of-{of:05d}.{uuid.uuid4().hex[:12]}.jsonl')
    from ... import __version__
    append_line(path, {'type': 'header', 'schema': SHARD_SCHEMA, 'work_id': work.work_id, 'kind': work.kind,
                       'shard': shard, 'of': of, 'items': len(idx), 'todo': len(todo), 'dexvantage': __version__,
                       'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    state = State(work, cache)
    done = failed = 0
    stopped = False
    for k, i in enumerate(todo):
        if cancel is not None and cancel():
            stopped = True
            break
        status, res, sec = execute(work, i, state, cancel)
        if status == 'cancelled':
            stopped = True
            break
        rec = {'type': 'item', 'index': i, 'status': status, 'seconds': sec}
        rec['result' if status == 'ok' else 'error'] = res
        append_line(path, rec)
        done += status == 'ok'
        failed += status == 'failed'
        if progress:
            progress((k + 1) / max(1, len(todo)), f'{ITEM_WORD[work.kind]} {k + 1} of {len(todo)}')
    out = {'shard': shard, 'of': of, 'items': len(idx), 'skipped': len(idx) - len(todo), 'done': done,
           'failed': failed, 'cancelled': stopped, 'file': os.path.basename(path), 'peak_rss_mb': peak_rss_mb(),
           'seconds': round(time.perf_counter() - t0, 3),
           'cache': cache.stats() if cache is not None else None}
    if not stopped:
        append_line(path, {'type': 'end', 'done': done, 'failed': failed, 'peak_rss_mb': out['peak_rss_mb'],
                           'seconds': out['seconds']})
    return out


def _ranges(xs) -> str:
    """'0-3,7,9-10' for [0, 1, 2, 3, 7, 9, 10] (Slurm's --array syntax)."""
    xs = sorted(set(int(x) for x in xs))
    out, k = [], 0
    while k < len(xs):
        j = k
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(str(xs[k]) if j == k else f'{xs[k]}-{xs[j]}')
        k = j + 1
    return ','.join(out)


def status(folder: str, out_dir: str | None = None, shards: int | None = None) -> dict:
    """Which shards of a job folder are complete, incomplete, failed or missing (no output at all)."""
    work = Work.load(folder)
    jf = _job_file(work.folder)
    out_dir = out_dir or os.path.join(work.folder, 'out')
    got = read_outputs(out_dir, work)
    n = shards or jf.get('shards')
    if not n:
        seen = Counter(nn for (_i, nn) in got['shards'])
        n = seen.most_common(1)[0][0] if seen else None
    if not n:
        raise ClusterError('There is no shard output in the folder yet, and the folder does not say how many shards '
                           'the work was split into: pass the number of shards.')
    m = len(work.items)
    rows, missing, incomplete, failed = [], [], [], []
    for i in range(int(n)):
        idx = shard_indices(m, i, int(n))
        ok = sum(1 for k in idx if k in got['results'])
        bad = [k for k in idx if k in got['failed']]
        att = got['shards'].get((i, int(n)), {}).get('attempts', 0)
        if ok == len(idx):
            st = 'complete'
        elif bad:
            st = 'failed'
            failed.append(i)
        elif att == 0:
            st = 'missing'
            missing.append(i)
        else:
            st = 'incomplete'
            incomplete.append(i)
        rows.append({'shard': i, 'status': st, 'items': len(idx), 'done': ok, 'failed': len(bad), 'attempts': att,
                     **({'error': got['failed'][bad[0]]} if bad else {})})
    peaks = [s['peak_rss_mb'] for s in got['shards'].values() if s.get('peak_rss_mb') is not None]
    return {'schema': REPORT_SCHEMA, 'work': work, 'out_dir': out_dir, 'shards': int(n), 'items': m,
            'items_done': len(got['results']), 'rows': rows, 'missing': missing, 'incomplete': incomplete,
            'failed': failed, 'redo': sorted(missing + incomplete + failed), 'foreign': got['foreign'],
            'results': got['results'], 'peak_rss_mb': max(peaks) if peaks else None,
            'cpu_seconds': round(sum(s['seconds'] for s in got['shards'].values()), 3)}


def _job_file(folder: str) -> dict:
    try:
        with open(os.path.join(folder, 'job.json'), encoding='utf-8') as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def gather(folder: str, out: str | None = None, out_dir: str | None = None, shards: int | None = None,
           write: bool = True, project=None) -> dict:
    """Gather a job folder (or the folder of a work.json): the report (JSON-able: which shards are complete,
    incomplete, failed or missing, the text to show), a resubmit script for the shards still to do (Slurm job
    folders), and, when every item has a result, the assembled result (the in-process one) written to out (default
    <folder>/result.json; an experiment also writes its runs CSV beside it). project: the project to name in the
    result's provenance (default: the work's own)."""
    st = status(folder, out_dir, shards)
    work = st.pop('work')
    results = st.pop('results')
    rep = {k: v for k, v in st.items()}
    rep.update(kind=work.kind, title=work.describe(), work_id=work.work_id, complete=not st['redo'] and
               len(results) == len(work.items), result_path=None, resubmit_path=None, result=None)
    w = ITEM_WORD[work.kind]
    if not rep['complete']:
        if write and st['redo']:
            from . import slurm
            rp = slurm.write_resubmit(work.folder, st['redo'], st['shards'])
            rep['resubmit_path'] = rp
        parts = []
        for key, word in (('missing', 'missing'), ('incomplete', 'incomplete'), ('failed', 'failed')):
            if st[key]:
                parts.append(f'{len(st[key])} {word} ({_ranges(st[key])})')
        rep['text'] = (f'{st["items_done"]} of {st["items"]} {w}s done; of {st["shards"]} shards, ' + ', '.join(parts)
                       + '.' + (f' Submit {os.path.basename(rep["resubmit_path"])} to run just those.'
                                if rep['resubmit_path'] else ''))
        first = next((r for r in st['rows'] if r.get('error')), None)
        if first:
            rep['text'] += f' First failure: {first["error"]}'
        return rep
    execution = {'workers': st['shards'], 'mode': f'Slurm job array of {st["shards"]} shards (core/cluster)',
                 'shards': st['shards'], 'peak_rss_mb': st['peak_rss_mb'], 'seconds': st['cpu_seconds']}
    doc = assemble(work, results, execution)
    rep['result'] = doc
    if write:
        out = out or os.path.join(work.folder, 'result.json')
        proj = project if project is not None else work.projects.get('base') or next(iter(work.projects.values()))
        rep['result_path'] = write_result(doc, proj, out, work.kind)
    rep['text'] = (f'Gathered {work.describe()} from {st["shards"]} shard{"" if st["shards"] == 1 else "s"}' +
                   (f' → {os.path.basename(rep["result_path"])}' if rep['result_path'] else '') + '.')
    head = doc.get('headline') or {}
    if head.get('text'):
        rep['text'] += ' ' + head['text']
    return rep


def write_result(doc: dict, project, out: str, kind: str, option: str = 'gather') -> str:
    """The result as JSON with provenance (an experiment's runs as CSV beside it, as --experiment writes them)."""
    from .. import provenance as PV
    from .cache import atomic_write
    if kind == 'experiment':
        from .. import experiments as EX
        jp, _cp = EX.write_outputs(doc, project, out, option=option)
        return jp
    d = {k: v for k, v in doc.items() if k != 'cached'}
    d = PV.stamp(d, project, output=PV.batch_output(d, out, option=option))
    atomic_write(out, (json.dumps(d, indent=1, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))
    return out


def report_json(rep: dict) -> dict:
    """The gather report without the (possibly large) result."""
    return {k: v for k, v in rep.items() if k != 'result'}
