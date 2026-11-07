"""Runs as tables: one row per run and one column per metric (wide), and one row per run and metric (long), written
as CSV and as Parquet with the same values.

    wide(runs) -> (columns, rows, units)       rows are dicts; metric columns hold floats or None
    long(runs) -> (columns, rows)
    write_csv(runs, path, form='wide'), write_parquet(runs, path, form='wide'), read_csv(path) (for checks)

CSV: UTF-8, a header row, RFC 4180 quoting (Python's csv module); numbers written with repr(), the shortest text that
reads back as the same float64 (Python's float repr, round-trip exact), so the CSV and the Parquet file hold
identical values; a figure not stated is an empty cell (CSV) or null (Parquet). Tags are comma-separated in one
cell. Metric columns are named by their key (core/tracking/record.py: a metric id, or id@stat) and their units are
in the long table, in the Parquet field metadata ('unit') and in units.json beside the wide CSV when written by
export().

Known limits: the wide table keeps a metric's value only (its status, bound and requirement check are in the long
table); a store with many different metrics makes a wide table with many mostly empty columns.
"""
from __future__ import annotations

import csv
import json
import math

RUN_COLUMNS = ('run_id', 'started_utc', 'ended_utc', 'seconds', 'kind', 'study', 'title', 'status', 'reason',
               'interface', 'project', 'project_file', 'project_sha256', 'variant', 'variant_sha256', 'headline',
               'status_level', 'status_word', 'tags', 'note', 'rerun_of', 'reproduce_verdict', 'dexvantage_version',
               'git_commit', 'git_dirty', 'result_sha256')
FLOAT_COLUMNS = ('seconds',)
LONG_COLUMNS = ('run_id', 'started_utc', 'study', 'key', 'metric', 'stat', 'title', 'value', 'unit', 'better',
                'status', 'bound', 'check', 'reason')


def _git(r) -> dict:
    return (((r.get('provenance') or {}).get('dexvantage') or {}).get('git') or {})


def run_row(r: dict) -> dict:
    """The run's own columns (RUN_COLUMNS)."""
    p, v, h = r.get('project') or {}, r.get('variant') or {}, r.get('headline') or {}
    dv = (r.get('provenance') or {}).get('dexvantage') or {}
    g = _git(r)
    secs = r.get('seconds')
    return {'run_id': r.get('run_id'), 'started_utc': r.get('started_utc'), 'ended_utc': r.get('ended_utc'),
            'seconds': float(secs) if isinstance(secs, (int, float)) and not isinstance(secs, bool) else None,
            'kind': r.get('kind'), 'study': r.get('study'), 'title': r.get('title'), 'status': r.get('status'),
            'reason': r.get('reason') or '', 'interface': r.get('interface'), 'project': p.get('name'),
            'project_file': p.get('file'), 'project_sha256': p.get('sha256'), 'variant': v.get('name'),
            'variant_sha256': v.get('sha256'), 'headline': h.get('text', ''), 'status_level': h.get('status', ''),
            'status_word': h.get('word', ''), 'tags': ','.join(r.get('tags') or []), 'note': r.get('note') or '',
            'rerun_of': r.get('rerun_of'), 'reproduce_verdict': (r.get('reproduce') or {}).get('verdict'),
            'dexvantage_version': dv.get('version'), 'git_commit': g.get('commit'),
            'git_dirty': None if g.get('dirty') is None else str(bool(g.get('dirty'))).lower(),
            'result_sha256': (r.get('result') or {}).get('sha256')}


def metric_keys(runs) -> list:
    keys: list = []
    seen = set()
    for r in runs:
        for k in (r.get('metrics') or {}):
            if k not in seen:
                seen.add(k)
                keys.append(k)
    return sorted(keys)


def wide(runs) -> tuple[list, list, dict]:
    """(columns, rows, units): the run columns then one column per metric key."""
    keys = metric_keys(runs)
    units = {}
    for r in runs:
        for k, m in (r.get('metrics') or {}).items():
            units.setdefault(k, m.get('unit') or '')
    rows = []
    for r in runs:
        row = run_row(r)
        ms = r.get('metrics') or {}
        for k in keys:
            v = (ms.get(k) or {}).get('value')
            row[k] = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) \
                else None
        rows.append(row)
    return list(RUN_COLUMNS) + keys, rows, units


def long(runs) -> tuple[list, list]:
    rows = []
    for r in runs:
        for k, m in sorted((r.get('metrics') or {}).items()):
            v = m.get('value')
            rows.append({'run_id': r.get('run_id'), 'started_utc': r.get('started_utc'), 'study': r.get('study'),
                         'key': k, 'metric': m.get('metric'), 'stat': m.get('stat'), 'title': m.get('title'),
                         'value': float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None,
                         'unit': m.get('unit') or '', 'better': m.get('better') or '', 'status': m.get('status'),
                         'bound': m.get('bound'), 'check': m.get('check'), 'reason': m.get('reason') or ''})
    return list(LONG_COLUMNS), rows


def _cell(v) -> str:
    if v is None:
        return ''
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _table(runs, form):
    if form == 'wide':
        cols, rows, units = wide(runs)
        return cols, rows, units
    if form == 'long':
        cols, rows = long(runs)
        return cols, rows, {}
    raise ValueError(f'form must be "wide" or "long", not {form!r}')


def write_csv(runs, path: str, form: str = 'wide') -> dict:
    """Write the table as CSV; {'rows', 'columns', 'units'}."""
    from .store import write_atomic
    import io
    cols, rows, units = _table(runs, form)
    buf = io.StringIO(newline='')
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(cols)
    for r in rows:
        w.writerow([_cell(r.get(c)) for c in cols])
    write_atomic(path, buf.getvalue().encode('utf-8'))
    return {'rows': len(rows), 'columns': len(cols), 'units': units}


def _pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError('Parquet needs pyarrow, which is not installed: pip install "dexvantage[lerobot]" (or '
                          'pyarrow) and export again; the CSV tables hold the same values.') from None
    return pa, pq


def float_columns(cols, form: str) -> set:
    if form == 'wide':
        return set(FLOAT_COLUMNS) | set(cols[len(RUN_COLUMNS):])
    return {'value'}


def write_parquet(runs, path: str, form: str = 'wide') -> dict:
    """Write the table as Parquet (float64 for numbers, UTF-8 strings otherwise, units in the field metadata)."""
    import os
    pa, pq = _pyarrow()
    cols, rows, units = _table(runs, form)
    fl = float_columns(cols, form)
    fields, arrays = [], []
    for c in cols:
        meta = {b'unit': units[c].encode('utf-8')} if c in units else None
        if c in fl:
            t = pa.float64()
            data = [r.get(c) for r in rows]
        else:
            t = pa.string()
            data = [None if r.get(c) is None else str(r.get(c)) for r in rows]
        fields.append(pa.field(c, t, metadata=meta))
        arrays.append(pa.array(data, type=t))
    schema = pa.schema(fields, metadata={b'dexvantage': json.dumps({'schema': 'dexvantage.runs_table/1',
                                                                    'form': form}).encode('utf-8')})
    table = pa.Table.from_arrays(arrays, schema=schema)
    tmp = path + f'.{os.getpid()}.part'
    pq.write_table(table, tmp)
    os.replace(tmp, path)
    return {'rows': len(rows), 'columns': len(cols), 'units': units}


def read_csv(path: str, form: str = 'wide') -> tuple[list, list]:
    """(columns, rows) of a CSV this module wrote, numbers as floats and empty cells as None."""
    with open(path, encoding='utf-8', newline='') as fh:
        rd = csv.reader(fh)
        cols = next(rd)
        fl = float_columns(cols, form)
        rows = []
        for vals in rd:
            row = {}
            for c, v in zip(cols, vals):
                row[c] = (float(v) if v != '' else None) if c in fl else (v if v != '' else None)
            rows.append(row)
    return cols, rows


def read_parquet(path: str) -> tuple[list, list]:
    pa, pq = _pyarrow()
    t = pq.read_table(path)
    return list(t.column_names), t.to_pylist()
