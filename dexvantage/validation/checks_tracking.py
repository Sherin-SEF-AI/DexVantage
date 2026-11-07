"""Checks of the experiment tracker (core/tracking): the run store, its tables and its MLflow layout, each against a
reference built independently here.

- tracking.store_round_trip: four runs built here (a real rig-inventory result; a tolerance Monte Carlo result with
  draws, statistics, a value not stated and samples; a scorecard with a bound, pass, fail and not evaluated; a failed
  run with its reason), each with Unicode tags and a two-line note, a rerun_of and a reproduction verdict, stored and
  read back through a new RunStore object: every leaf field of every record (run.json and the index line, which
  leaves out only the samples) and every result file against the records as built (the reference: the in-memory
  dicts through a JSON round trip, plus the two fields add() sets), then a tag and note update read back the same
  way. Error: the number of fields that differ. Tolerance 0: storage is exact or it is wrong.
- tracking.concurrent_writers: N writer processes (8; 4 in a quick run) started together on one store, each adding M
  synthetic runs (100; 25 quick) and updating the tags of every third one at once. The reference is what the writers
  report they wrote (their run ids) and the figures written into each (writer × 1000 + i). Error: runs lost + runs
  not written by a writer + index lines that do not parse + runs whose figures, result or tags differ + temporary
  files left. Tolerance 0. The check also requires that at least two writers were writing at the same moment (from
  the start and end times they report), so that the writes really were concurrent.
- tracking.crash_recovery: three crashes written by hand: an index line cut mid-write (no newline), a run folder
  renamed into place whose writer died before its index line, and a temporary folder whose writer died before the
  rename; then a normal writer adds a run. The reference is what each crash means: the cut line is one damaged
  line (its run's folder is read instead), the unindexed folder is recovered, the temporary folder is not a run, the
  later run is intact. Error: the number of those that do not hold. Tolerance 0.
- tracking.csv_parquet_identical: 40 runs with 12 metrics of random figures (magnitudes from 10⁻³⁰⁰ to 10³⁰⁰, both
  signs, negative zero, whole numbers, about one in five not stated) written as CSV and as Parquet, then read with
  Python's csv module and with pyarrow.parquet (independent readers): every cell against the value in the record,
  numbers compared bit for bit (their IEEE 754 binary64 bytes), strings exactly, not stated as an empty cell or null.
  Error: the number of cells that differ. Tolerance 0: repr() writes the shortest text that reads back as the same
  float64, so nothing may change.
- tracking.mlflow_reader: the runs of tracking.store_round_trip written as an mlruns/ folder and read back with
  MLflow's own MlflowClient (mlflow-skinny 3.17.2, when installed: the optional extra "tracking") or, without it, with
  a reader of the layout written out here from MLflow's file_store.py (meta.yaml with PyYAML; "timestamp value step"
  metric lines; params and tags as text). Every run's id, name, status, start and end time (ms), every stated metric
  (exactly), every param and the note, run-name and user-tag tags against the records. Error: the number that
  differ. Tolerance 0.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import struct
import subprocess
import sys
import time

from .suite import Outcome, check

_REFS_STORE = ('posix_rename_flock', 'json_lines')


# ====================================================================== records built here
def _records(ctx) -> list:
    """[(record, stamped result or None)]: four runs that fill every field of the record."""
    from ..core import jobs
    from ..core.project import blank_project
    from ..core.tracking import record as RC
    if 'records' in ctx.cache:
        return ctx.cache['records']
    p = blank_project('Validation · tracker')
    rng = ctx.rng('tracking.records')
    draws = [float(x) for x in rng.normal(91.0, 2.0, 7)] + [None]
    good = [x for x in draws if x is not None]
    tol = {'schema': 'dexvantage.tolerance/1', 'study': 'tolerance', 'n': 8, 'done': 8, 'seed': 3, 'seconds': 1.25,
           'cancelled': False,
           'metrics': {'coverage.covered': {'title': 'Task volume seen', 'unit': '%', 'better': 'higher',
                                            'nominal': 91.5, 'nominal_status': 'ok', 'values': draws,
                                            'p5': min(good), 'p50': sorted(good)[3], 'p95': max(good),
                                            'mean': sum(good) / len(good), 'std': None},
                       'sync.skew_ms': {'title': 'Worst camera to joint-state skew', 'unit': 'ms', 'better': 'lower',
                                        'nominal': None, 'nominal_status': 'refused', 'values': [None] * 8,
                                        'reason': 'The rig has no camera.'}},
           'headline': {'value': 91.5, 'unit': '%', 'status': 'warning', 'status_word': 'wide spread',
                        'text': 'Coverage P5 over 8 draws.'}}
    card = {'schema': 'dexvantage.requirements/1', 'study': 'requirements', 'verdict': 'fail', 'level': 'serious',
            'seconds': 0.5, 'cancelled': False,
            'counts': {'pass': 1, 'fail': 1, 'not_evaluated': 1, 'total': 3},
            'items': [{'uid': 'a1', 'metric': 'sync.skew_ms', 'title': 'Skew', 'op': '<', 'op_text': '<',
                       'threshold': 2.0, 'unit': 'ms', 'value': 1.0, 'bound': 'at_least', 'status': 'pass'},
                      {'uid': 'a2', 'metric': 'coverage.covered', 'title': 'Coverage', 'op': '>=', 'op_text': '≥',
                       'threshold': 95.0, 'unit': '%', 'value': 91.5, 'bound': None, 'status': 'fail'},
                      {'uid': 'a3', 'metric': 'bandwidth.margin', 'title': 'Margin', 'op': '>=', 'op_text': '≥',
                       'threshold': 0.0, 'unit': 'Mbit/s', 'value': None, 'bound': None, 'status': 'not_evaluated',
                       'reason': 'No host.'}],
            'headline': {'status': 'serious', 'word': 'not ready', 'text': '1 / 3 pass.'}}
    rig = jobs.run_study('rig', p.snapshot())
    common = dict(tags=['α-sweep', 'café, wrist', 'thesis ch3'], note='line one\nline two: ±0.5 mm, 100 %',
                  rerun_of='0' * 32, reproduce={'verdict': 'reproduces', 'verdict_text': 'reproduces exactly',
                                                'message': 'Reproduces exactly.', 'reasons': []},
                  started='2026-11-07T10:00:00.125Z', ended='2026-11-07T10:00:02.5Z')
    out = [RC.build('rig', rig, p, interface='batch', **common),
           RC.build('tolerance', tol, p, interface='app', **common),
           RC.build('requirements', card, p, kind='check', interface='api', **common),
           RC.build('coverage', None, p, reason='Coverage needs a camera: add one.', **common)]
    ctx.cache['records'] = out
    return out


def _leaves(o, path='', out=None) -> dict:
    out = {} if out is None else out
    if isinstance(o, dict):
        if not o:
            out[path + '{}'] = None
        for k, v in o.items():
            _leaves(v, f'{path}.{k}' if path else str(k), out)
    elif isinstance(o, list):
        if not o:
            out[path + '[]'] = None
        for i, v in enumerate(o):
            _leaves(v, f'{path}[{i}]', out)
    else:
        out[path] = o
    return out


def _diff(a, b) -> list:
    la, lb = _leaves(a), _leaves(b)
    return [k for k in sorted(set(la) | set(lb)) if la.get(k, KeyError) != lb.get(k, KeyError) or
            type(la.get(k)) is not type(lb.get(k))]


@check('tracking.store_round_trip', 'tracking', 'Run store round trip (every field)',
       'fields of four stored runs (run.json, index line, result.json, after an update) that differ from the records '
       'as built', 'the records as built here, through a JSON round trip', 0, 'fields', refs=_REFS_STORE)
def store_round_trip(ctx):
    from ..core.tracking import RunStore
    from ..core.tracking.store import INDEX_OMIT
    folder = os.path.join(ctx.tmp(), 'round-trip')
    built = _records(ctx)
    st = RunStore(folder)
    bad, fields = [], 0
    stored = [st.add(rec, res) for rec, res in built]
    back = RunStore(folder)                                  # a new reader: nothing kept in memory
    idx = back.scan()['runs']
    for (rec, res), s in zip(built, stored):
        want = json.loads(json.dumps(rec, ensure_ascii=False))
        want['stored_utc'] = s['stored_utc']
        if res is not None:
            want['result'] = dict(want['result'], file='result.json')
        got = back.get(rec['run_id'])
        d = _diff(want, got)
        fields += len(_leaves(want))
        bad += [f'{rec["study"]}: run.json {k}' for k in d]
        wi = {k: v for k, v in want.items() if k not in INDEX_OMIT}
        bad += [f'{rec["study"]}: index {k}' for k in _diff(wi, idx.get(rec['run_id']) or {})]
        if res is not None:
            bad += [f'{rec["study"]}: result.json {k}' for k in _diff(json.loads(json.dumps(res)),
                                                                     back.result(rec['run_id']))]
        elif back.result(rec['run_id']) is not None:
            bad.append(f'{rec["study"]}: a result.json the failed run should not have')
    rid = built[1][0]['run_id']
    st.update(rid, tags=['après', 'β'], note='changed\nnote')
    again = RunStore(folder)
    for where, r in (('run.json', again.get(rid)), ('index', again.scan()['runs'][rid])):
        if r.get('tags') != ['après', 'β'] or r.get('note') != 'changed\nnote':
            bad.append(f'update not read back from {where}')
    others = [x for x in again.runs() if x['run_id'] != rid]
    if any(x.get('tags') != built[0][0]['tags'] for x in others):
        bad.append('an update changed another run')
    return Outcome(float(len(bad)), {'runs': len(built), 'fields_compared': fields, 'differences': bad[:20]})


# ====================================================================== several writers
def _env() -> dict:
    from ..core import provenance as PV
    env = dict(os.environ)
    env['PYTHONPATH'] = PV.CHECKOUT_DIR + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    return env


def concurrent(folder: str, writers: int, per: int, timeout: float = 180.0) -> dict:
    """Run `writers` writer processes (core/tracking/_stress.py) at once on folder; what they report and what the
    store holds, compared. Used by the check and by tests/test_tracking.py."""
    from ..core.tracking import RunStore
    from ..core.tracking.store import RUNS
    os.makedirs(folder, exist_ok=True)
    go = os.path.join(os.path.dirname(folder), os.path.basename(folder) + '.go')
    procs = [subprocess.Popen([sys.executable, '-m', 'dexvantage.core.tracking._stress', folder, str(w), str(per), go],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=_env(), cwd=os.path.dirname(folder))
             for w in range(writers)]
    time.sleep(0.3)
    with open(go, 'w'):
        pass
    reported, errors, spans = {}, [], []
    t0 = time.monotonic()
    for w, pr in enumerate(procs):
        try:
            out, err = pr.communicate(timeout=max(1.0, timeout - (time.monotonic() - t0)))
        except subprocess.TimeoutExpired:
            pr.kill()
            out, err = pr.communicate()
            errors.append(f'writer {w} timed out')
        if pr.returncode != 0:
            last = (err.decode(errors='replace').strip().splitlines() or [''])[-1]
            errors.append(f'writer {w} exited {pr.returncode}: {last}')
            continue
        try:
            rep = json.loads(out.decode('utf-8').strip().splitlines()[-1])
            reported[w] = rep['ids']
            spans.append((rep['start'], rep['end']))
        except (ValueError, IndexError, KeyError):
            errors.append(f'writer {w} reported nothing readable')
    os.remove(go)
    st = RunStore(folder)
    scan = st.scan()
    runs = scan['runs']
    want = {rid: (w, i) for w, ids in reported.items() for i, rid in enumerate(ids)}
    lost = [r for r in want if r not in runs]
    extra = [r for r in runs if r not in want]
    corrupt = []
    for rid, (w, i) in want.items():
        if rid not in runs:
            continue
        try:
            full, res = st.get(rid), st.result(rid)
        except Exception as e:                           # noqa: BLE001 — counted as corrupt
            corrupt.append(f'{rid[:8]}: {e}')
            continue
        v = float(w * 1000 + i)
        tags = {f'w{w}'} | ({f'updated-{w}'} if i % 3 == 0 else set())
        if (full['metrics']['synthetic.value']['value'] != v or runs[rid]['metrics']['synthetic.value']['value'] != v
                or not res or res.get('writer') != w or res.get('i') != i or len(res.get('payload', '')) != 2048
                or set(full.get('tags') or []) != tags or set(runs[rid].get('tags') or []) != tags):
            corrupt.append(rid[:8])
    left = [n for n in os.listdir(os.path.join(folder, RUNS)) if n.startswith('.')]
    left += [n for n in os.listdir(folder) if n.endswith('.part')]
    # how many writers were writing at the busiest moment (the writes really were concurrent)
    events = sorted([(t0, 1) for t0, _ in spans] + [(t1, -1) for _, t1 in spans], key=lambda e: (e[0], e[1]))
    busy = peak = 0
    for _t, d in events:
        busy += d
        peak = max(peak, busy)
    return {'writers': writers, 'per_writer': per, 'reported': len(want), 'stored': len(runs), 'lost': len(lost),
            'writers_at_once': peak,
            'extra': len(extra), 'corrupt': len(corrupt), 'damaged_lines': scan['damaged_lines'],
            'recovered': scan['recovered'], 'temporary_left': len(left), 'errors': errors,
            'first_corrupt': corrupt[:5], 'seconds': round(time.monotonic() - t0, 2)}


@check('tracking.concurrent_writers', 'tracking', 'Several writer processes on one run store',
       'runs lost, foreign, corrupted (figures, result, tags) or left half-written when N processes add and update '
       'runs at once, plus index lines that do not parse', 'what each writer reports it wrote (its run ids and the '
       'figures it put in them)', 0, 'runs', refs=_REFS_STORE, cost='study')
def concurrent_writers(ctx):
    n, m = (4, 25) if ctx.quick else (8, 100)
    r = concurrent(os.path.join(ctx.tmp(), 'concurrent'), n, m)
    err = r['lost'] + r['extra'] + r['corrupt'] + r['damaged_lines'] + r['temporary_left'] + len(r['errors']) + \
        (r['reported'] != n * m) * abs(n * m - r['reported'])
    return Outcome(float(err), r, ok=r['writers_at_once'] >= 2,
                   note=f'{r["writers_at_once"]} of {n} writers were writing at the same moment at the busiest')


@check('tracking.crash_recovery', 'tracking', 'Run store after writers that died mid-write',
       'what readers make of an index line cut mid-write, a run folder without its index line and a temporary folder '
       'left before its rename, after another writer adds a run', 'the meaning of each crash, written out here', 0,
       'cases', refs=_REFS_STORE)
def crash_recovery(ctx):
    from ..core.tracking import RunStore
    from ..core.tracking._stress import synthetic
    from ..core.tracking.store import RUN_FILE, dumps
    folder = os.path.join(ctx.tmp(), 'crash')
    st = RunStore(folder)
    a, ra = synthetic(1, 0)
    st.add(a, ra)
    b, rb = synthetic(1, 1)                                  # its writer dies inside the append: half a line
    st.add(b, rb)
    with open(st.index_path, 'rb') as fh:
        lines = fh.read().split(b'\n')
    whole = b'\n'.join(lines[:-2]) + b'\n' + lines[-2][: len(lines[-2]) // 2]
    with open(st.index_path, 'wb') as fh:
        fh.write(whole)
    c, _rc = synthetic(1, 2)                                 # renamed into place, died before its index line
    os.makedirs(st.run_dir(c['run_id']))
    with open(os.path.join(st.run_dir(c['run_id']), RUN_FILE), 'wb') as fh:
        fh.write(dumps(c, indent=1))
    tmp = os.path.join(st.runs_dir, f'.tmp-{"f" * 32}-1-abc')   # died before the rename
    os.makedirs(tmp)
    with open(os.path.join(tmp, RUN_FILE), 'w') as fh:
        fh.write('{"run_id": "' + 'f' * 32 + '"')
    d, rd = synthetic(1, 3)                                  # a later writer
    st.add(d, rd)
    scan = RunStore(folder).scan()
    runs = scan['runs']
    cases = {'the run before the crash is intact': a['run_id'] in runs and
             runs[a['run_id']]['metrics']['synthetic.value']['value'] == 1000.0,
             'the cut line counts as one damaged line': scan['damaged_lines'] == 1,
             'the run of the cut line is read from its folder': b['run_id'] in runs,
             'the folder without an index line is recovered': c['run_id'] in runs and scan['recovered'] == 2,
             'the temporary folder is not a run': ('f' * 32) not in runs,
             'the later run is intact': d['run_id'] in runs and runs[d['run_id']]['metrics']['synthetic.value'][
                 'value'] == 1003.0 and RunStore(folder).result(d['run_id'])['i'] == 3,
             'nothing else appears': len(runs) == 4}
    failed = [k for k, ok in cases.items() if not ok]
    return Outcome(float(len(failed)), {'cases': list(cases), 'failed': failed, 'damaged_lines': scan['damaged_lines'],
                                        'recovered': scan['recovered'], 'runs': len(runs)})


# ====================================================================== tables
def table_records(rng, n_runs: int = 40, n_metrics: int = 12) -> list:
    """Runs with random figures for the table checks (also used by tests/test_tracking.py)."""
    from ..core.tracking.store import new_run_id
    out = []
    for k in range(n_runs):
        ms = {}
        for j in range(n_metrics):
            u = rng.random()
            if u < 0.2:
                v = None
            elif u < 0.25:
                v = -0.0
            elif u < 0.35:
                v = float(int(rng.integers(-10 ** 6, 10 ** 6)))
            else:
                v = float(rng.choice([-1.0, 1.0]) * rng.random() * 10.0 ** float(rng.uniform(-300, 300)))
            ms[f'm{j:02d}.figure' + ('@p5' if j % 4 == 3 else '')] = {
                'metric': f'm{j:02d}.figure', 'stat': 'p5' if j % 4 == 3 else None, 'title': f'Figure {j}',
                'unit': ['%', 'mm', 'ms', 'px'][j % 4], 'better': 'higher', 'value': v,
                'status': 'ok' if v is not None else 'not stated'}
        out.append({'run_id': new_run_id(), 'started_utc': f'2026-11-07T10:{k // 60:02d}:{k % 60:02d}.000Z',
                    'ended_utc': f'2026-11-07T11:{k // 60:02d}:{k % 60:02d}.000Z', 'seconds': float(rng.random()),
                    'kind': 'study', 'study': 'synthetic', 'title': 'Synthetic, "quoted"', 'status': 'finished',
                    'project': {'name': 'Cell, A', 'sha256': 'ab' * 32}, 'variant': {'name': 'v1\nnew line'},
                    'headline': {'text': 'a, b; "c"'}, 'tags': ['x', 'y z'], 'note': 'é ü\n2nd line',
                    'metrics': ms})
    return out


def _bits(v) -> bytes | None:
    return None if v is None else struct.pack('<d', float(v))


@check('tracking.csv_parquet_identical', 'tracking', 'CSV and Parquet tables hold identical values',
       'cells of the runs × metrics tables (wide and long) whose value read back from CSV (Python\'s csv module) or '
       'Parquet (pyarrow) differs from the record, numbers bit for bit', 'the records\' own values', 0, 'cells',
       refs=('rfc4180', 'parquet_format'))
def csv_parquet_identical(ctx):
    import csv

    from ..core.tracking import tables as TB
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return Outcome(float('nan'), {'skipped': 'pyarrow is not installed'}, ok=False,
                       note='pyarrow is not installed: Parquet cannot be checked.')
    rng = ctx.rng('tracking.csv_parquet_identical')
    runs = table_records(rng)
    folder = os.path.join(ctx.tmp(), 'tables')
    os.makedirs(folder, exist_ok=True)
    bad, cells = [], 0
    for form in ('wide', 'long'):
        cols, want, _u = (TB.wide(runs) if form == 'wide' else (*TB.long(runs), None))
        fl = TB.float_columns(cols, form)
        cp, pp = os.path.join(folder, f'{form}.csv'), os.path.join(folder, f'{form}.parquet')
        TB.write_csv(runs, cp, form)
        TB.write_parquet(runs, pp, form)
        with open(cp, encoding='utf-8', newline='') as fh:
            rd = list(csv.reader(fh))
        head, body = rd[0], rd[1:]
        pt = pq.read_table(pp).to_pylist()
        if head != cols or list(pq.read_schema(pp).names) != cols or len(body) != len(want) or len(pt) != len(want):
            bad.append(f'{form}: columns or row count differ')
            continue
        for r, (w, c, p) in enumerate(zip(want, body, pt)):
            for j, col in enumerate(cols):
                cells += 1
                v = w.get(col)
                if col in fl:
                    cv = None if c[j] == '' else float(c[j])
                    if _bits(cv) != _bits(v) or _bits(p[col]) != _bits(v):
                        bad.append(f'{form} row {r} {col}: {v!r} csv {c[j]!r} parquet {p[col]!r}')
                else:
                    sv = None if v in (None, '') else str(v)
                    if (c[j] or None) != sv or (p[col] or None) != sv:
                        bad.append(f'{form} row {r} {col}: {v!r} csv {c[j]!r} parquet {p[col]!r}')
    finite = sum(1 for r in runs for m in r['metrics'].values() if m['value'] is not None)
    return Outcome(float(len(bad)), {'runs': len(runs), 'cells': cells, 'figures_stated': finite,
                                     'differences': bad[:10]})


# ====================================================================== MLflow
@contextlib.contextmanager
def mlflow_env():
    """The environment MLflow 3.17.2 needs to open a file store (MLFLOW_ALLOW_FILE_STORE) with its telemetry and
    agent hint off; restored afterwards."""
    keys = {'MLFLOW_ALLOW_FILE_STORE': 'true', 'MLFLOW_DISABLE_TELEMETRY': 'true', 'DO_NOT_TRACK': 'true',
            'MLFLOW_DISABLE_AGENT_HINT': '1', 'MLFLOW_TRACKING_URI': ''}
    old = {k: os.environ.get(k) for k in keys}
    os.environ.update({k: v for k, v in keys.items() if v})
    os.environ.pop('MLFLOW_TRACKING_URI', None)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _read_independent(root: str) -> dict:
    """The mlruns layout read as MLflow 3.17.2's file_store.py reads it, written out again here: {run_id: {'name',
    'status' (int), 'start', 'end', 'experiment', 'metrics' {name: last value}, 'params', 'tags'}}."""
    import yaml
    out = {}
    for eid in os.listdir(root):
        d = os.path.join(root, eid)
        if eid == '.trash' or not os.path.isdir(d):
            continue
        with open(os.path.join(d, 'meta.yaml'), encoding='utf-8') as fh:
            exp = yaml.safe_load(fh)
        for rid in os.listdir(d):
            rd = os.path.join(d, rid)
            if not os.path.isfile(os.path.join(rd, 'meta.yaml')):
                continue
            with open(os.path.join(rd, 'meta.yaml'), encoding='utf-8') as fh:
                meta = yaml.safe_load(fh)
            r = {'name': meta.get('run_name'), 'status': meta.get('status'), 'start': meta.get('start_time'),
                 'end': meta.get('end_time'), 'experiment': exp.get('name'), 'id': meta.get('run_id'),
                 'metrics': {}, 'params': {}, 'tags': {}}
            for sub in ('metrics', 'params', 'tags'):
                sd = os.path.join(rd, sub)
                for n in os.listdir(sd) if os.path.isdir(sd) else []:
                    with open(os.path.join(sd, n), encoding='utf-8') as fh:
                        t = fh.read()
                    if sub == 'metrics':
                        parts = [ln.strip().split(' ') for ln in t.strip().splitlines()]
                        r[sub][n] = float(max(parts, key=lambda q: (int(q[2]), int(q[0])))[1])
                    else:
                        r[sub][n] = t
            out[rid] = r
    return out


def _read_mlflow(root: str) -> dict:
    from mlflow.tracking import MlflowClient
    c = MlflowClient(tracking_uri='file:' + os.path.abspath(root))
    out = {}
    for e in c.search_experiments():
        for r in c.search_runs([e.experiment_id]):
            st = {'RUNNING': 1, 'SCHEDULED': 2, 'FINISHED': 3, 'FAILED': 4, 'KILLED': 5}[r.info.status]
            out[r.info.run_id] = {'name': r.info.run_name, 'status': st, 'start': r.info.start_time,
                                  'end': r.info.end_time, 'experiment': e.name, 'id': r.info.run_id,
                                  'metrics': dict(r.data.metrics), 'params': dict(r.data.params),
                                  'tags': dict(r.data.tags)}
    return out


def mlflow_compare(root: str, records: list, reader: str = 'auto') -> dict:
    """Read root (an mlruns folder) with MLflow (reader 'mlflow', or 'auto' when it is installed) or the independent
    layout reader, and compare with the records. Used by the check and by tests/test_tracking.py."""
    from ..core.tracking import mlruns as ML
    from ..core.tracking.store import epoch_ms
    use = reader
    if reader == 'auto':
        try:
            with mlflow_env():
                import mlflow  # noqa: F401
            use = 'mlflow'
        except ImportError:
            use = 'layout'
    with mlflow_env():
        got = _read_mlflow(root) if use == 'mlflow' else _read_independent(root)
    bad, compared = [], 0
    for r in records:
        rid = r['run_id']
        g = got.get(rid)
        if g is None:
            bad.append(f'{rid[:8]}: missing')
            continue
        names = ML.metric_names(r)
        want = {'name': r.get('name'), 'status': ML.STATUS[r.get('status')], 'start': epoch_ms(r['started_utc']),
                'end': epoch_ms(r['ended_utc']), 'experiment': ML.experiment_name(r), 'id': rid}
        for k, v in want.items():
            compared += 1
            if g.get(k) != v:
                bad.append(f'{rid[:8]} {k}: {v!r} written, {g.get(k)!r} read')
        stated = {names[k]: m['value'] for k, m in (r.get('metrics') or {}).items() if m.get('value') is not None}
        compared += len(stated) + 1
        if set(stated) != set(g['metrics']):
            bad.append(f'{rid[:8]} metric names: {sorted(set(stated) ^ set(g["metrics"]))}')
        for n, v in stated.items():
            if n in g['metrics'] and _bits(g['metrics'][n]) != _bits(v):
                bad.append(f'{rid[:8]} {n}: {v!r} written, {g["metrics"][n]!r} read')
        for n, v in ML.run_params(r).items():
            compared += 1
            if g['params'].get(n) != v:
                bad.append(f'{rid[:8]} param {n}')
        tags = {'mlflow.runName': r.get('name'), 'mlflow.note.content': r.get('note') or None,
                **{ML.mlflow_name(f'tag.{t}'): 'true' for t in r.get('tags') or []}}
        for n, v in tags.items():
            compared += 1
            if v is not None and g['tags'].get(n) != v:
                bad.append(f'{rid[:8]} tag {n}: {v!r} written, {g["tags"].get(n)!r} read')
    return {'reader': 'MLflow 3.17.2 MlflowClient' if use == 'mlflow' else 'layout reader written here',
            'runs': len(records), 'read': len(got), 'compared': compared, 'differences': bad}


@check('tracking.mlflow_reader', 'tracking', 'MLflow file store read back by MLflow',
       'run ids, names, statuses, times, metrics, params and tags of the mlruns/ export that MLflow reads back '
       'otherwise than written', 'MLflow 3.17.2\'s MlflowClient (when installed), else a reader of the layout written '
       'out here from its file_store.py', 0, 'fields', refs=('mlflow_filestore_3172', 'mlflow_client_3172'),
       cost='study')
def mlflow_reader(ctx):
    from ..core.tracking import RunStore
    from ..core.tracking import mlruns as ML
    folder = os.path.join(ctx.tmp(), 'mlflow-store')
    st = RunStore(folder)
    recs = [st.add(rec, res) for rec, res in _records(ctx)] if not st.exists() else st.runs()
    root = os.path.join(ctx.tmp(), 'mlruns')
    ML.write(recs, root, st)
    r = mlflow_compare(root, recs)
    err = float(len(r['differences']) + abs(r['read'] - len(recs)))
    return Outcome(err if math.isfinite(err) else float('nan'), dict(r, differences=r['differences'][:20]),
                   note=f'read back with the {r["reader"]}')
