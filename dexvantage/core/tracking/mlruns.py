"""Runs as an MLflow file store: the mlruns/ folder layout MLflow's FileStore reads, written directly (MLflow itself
is not needed to write it).

The layout is MLflow 3.17.2's mlflow/store/tracking/file_store.py (FileStore), read at
https://github.com/mlflow/mlflow/blob/v3.17.2/mlflow/store/tracking/file_store.py on 2026-11-07, with the entities
it persists (mlflow/entities/run_info.py, run_status.py, experiment.py, source_type.py), the name rules of
mlflow/utils/validation.py and the YAML writer of mlflow/utils/yaml_utils.py at the same tag:

    mlruns/
      .trash/                                  FileStore.TRASH_FOLDER_NAME (created by FileStore when missing)
      0/meta.yaml                              the "Default" experiment (FileStore.DEFAULT_EXPERIMENT_ID)
      <experiment_id>/meta.yaml                artifact_location, creation_time, effective_trace_archival_retention
                                               (null), experiment_id, last_update_time, lifecycle_stage, name,
                                               trace_location (null), workspace ('default',
                                               utils/workspace_utils.DEFAULT_WORKSPACE_NAME): dict(Experiment)
                                               without its tags, as FileStore._create_experiment_with_id writes it
                                               (the same keys a file store made by mlflow-skinny 3.17.2 holds,
                                               compared by tests/test_tracking_mlruns_meta.py). Older readers
                                               keep only the keys their Experiment knows
                                               (_MlflowObject.from_dictionary filters them, read at v1.30.0, v2.0.1
                                               and v3.17.2), so the three newer keys do not upset them.
      <experiment_id>/<run_id>/meta.yaml       _make_persisted_run_info_dict(): artifact_uri, end_time,
                                               entry_point_name '', experiment_id, lifecycle_stage, run_id,
                                               run_name, source_name '', source_type 4 (LOCAL), source_version '',
                                               start_time, status (an integer: RUNNING 1, FINISHED 3, FAILED 4,
                                               KILLED 5), tags [], user_id, deleted_time
      <experiment_id>/<run_id>/metrics/<key>   one line per value: "<timestamp ms> <value> <step>" (_log_run_metric)
      <experiment_id>/<run_id>/params/<key>    the value, as text (_log_run_param)
      <experiment_id>/<run_id>/tags/<key>      the value, as text, no trailing newline (_set_run_tag)
      <experiment_id>/<run_id>/artifacts/      result.json and run.json

Metric, param and tag names may hold only letters, digits, '_', '-', '.', ' ', ':' and '/' and must be normalised
POSIX paths (validate_param_and_metric_name, path_not_unique); here every other character becomes '_' and '/' becomes
'_' too (a '/' would make a sub-folder, which could clash with a metric of the same name), so 'coverage.covered@p5'
is the metric 'coverage.covered_p5'. Values: params at most 6000 characters, tags 8000 (MAX_PARAM_VAL_LENGTH,
MAX_TAG_VAL_LENGTH; longer ones are cut, as MLflow's own client cuts them). A figure the run does not state is not
written as a metric (MLflow has no "not stated"); the tag dexvantage.not_stated lists those keys and
dexvantage.unit.<key> gives each metric's unit.

Experiments: one per DexVantage project name ("DexVantage · <project>"), its id a number derived from the name
(the first 12 hexadecimal digits of its SHA-256, as an integer), so exporting again adds to the same experiment.
A run keeps its DexVantage run id as its MLflow run id (uuid4 hex, which MLflow's _RUN_ID_REGEX accepts).

Since MLflow 3.13 the file store is "in maintenance mode": MLflow's own FileStore refuses to open one unless the
environment variable MLFLOW_ALLOW_FILE_STORE=true is set (file_store.py, FileStore.__init__; 3.7 to 3.12 only warn
that it "will be deprecated", read at the v3.7.0, v3.12.0 and v3.13.0 tags on 2026-11-07); set it to read this
folder with mlflow ui or MlflowClient, or migrate it with mlflow migrate-filestore.

Known limits: artifact URIs are absolute file: URIs of the folder as written (as MLflow writes them), so a moved
folder keeps its metrics, params and tags but its artifact links point to the old place; an export over an existing
mlruns/ folder replaces runs with the same id and leaves other runs alone; MLflow datasets, inputs, traces and
logged models are not written.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil

from .store import epoch_ms, write_atomic

MLFLOW_VERSION_READ = '3.17.2'
DEFAULT_WORKSPACE = 'default'                   # mlflow/utils/workspace_utils.py DEFAULT_WORKSPACE_NAME at v3.17.2
DEFAULT_EXPERIMENT_ID = '0'
STATUS = {'running': 1, 'scheduled': 2, 'finished': 3, 'failed': 4, 'cancelled': 5}   # RunStatus (service.proto)
SOURCE_TYPE_LOCAL = 4                    # SourceType: NOTEBOOK, JOB, PROJECT, LOCAL, UNKNOWN = range(1, 6)
MAX_PARAM = 6000
MAX_TAG = 8000
MAX_KEY = 250
_BAD = re.compile(r'[^\w.\- :]')


def mlflow_name(key: str) -> str:
    """A name MLflow accepts for a metric, param or tag (letters, digits, _ - . space : only; no '/'), stable for a
    given key."""
    s = _BAD.sub('_', str(key).replace('/', '_')).strip() or '_'
    s = s.replace(':', '_') if os.name == 'nt' else s
    s = re.sub(r'\.{2,}', '.', s).strip('.') or '_'      # normpath would fold '..', a leading or trailing '.'
    return s[:MAX_KEY]


def experiment_id(name: str) -> str:
    return str(int(hashlib.sha256(name.encode('utf-8')).hexdigest()[:12], 16))


def _yaml(path: str, data: dict):
    import yaml
    text = yaml.safe_dump(data, default_flow_style=False, allow_unicode=True, sort_keys=True)
    write_atomic(path, text.encode('utf-8'))


def _text(path: str, value: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_atomic(path, value.encode('utf-8'))


def _file_uri(path: str) -> str:
    from pathlib import Path
    return Path(os.path.abspath(path)).as_uri()


def _flatten(o, prefix='') -> dict:
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(_flatten(v, f'{prefix}.{k}' if prefix else str(k)))
    elif isinstance(o, list) and o and all(isinstance(x, dict) for x in o) and len(o) <= 20:
        for i, v in enumerate(o):
            out.update(_flatten(v, f'{prefix}.{i}'))
    else:
        out[prefix or 'value'] = o
    return out


def _value_text(v) -> str:
    if v is None:
        return ''
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, (int, float, str)):
        return str(v)
    return json.dumps(v, ensure_ascii=False, separators=(',', ':'), default=str)


def _experiment(root: str, eid: str, name: str, when_ms: int):
    d = os.path.join(root, eid)
    os.makedirs(d, exist_ok=True)
    meta = os.path.join(d, 'meta.yaml')
    if os.path.isfile(meta):
        return
    _yaml(meta, {'artifact_location': _file_uri(d), 'creation_time': when_ms,
                 'effective_trace_archival_retention': None, 'experiment_id': eid, 'last_update_time': when_ms,
                 'lifecycle_stage': 'active', 'name': name, 'trace_location': None,
                 'workspace': DEFAULT_WORKSPACE})


def experiment_name(r: dict) -> str:
    return f'DexVantage · {(r.get("project") or {}).get("name") or "no project"}'


def run_params(r: dict) -> dict:
    """{param name: text}: the study, kind and settings (flattened) and seeds."""
    p = {'study': r.get('study'), 'kind': r.get('kind'), 'interface': r.get('interface')}
    for k, v in _flatten(r.get('settings') or {}).items():
        p[f'settings.{k}'] = v
    for k, v in (r.get('seeds') or {}).items():
        p[f'seed.{k}'] = v
    out = {}
    for k, v in p.items():
        n = mlflow_name(k)
        out.setdefault(n, _value_text(v)[:MAX_PARAM])
    return out


def run_tags(r: dict, names: dict) -> dict:
    proj, var = r.get('project') or {}, r.get('variant') or {}
    dv = (r.get('provenance') or {}).get('dexvantage') or {}
    git = dv.get('git') or {}
    t = {'mlflow.runName': r.get('name') or r.get('run_id'), 'mlflow.source.type': 'LOCAL',
         'mlflow.source.name': 'dexvantage', 'mlflow.note.content': r.get('note') or '',
         'dexvantage.run_id': r.get('run_id'), 'dexvantage.kind': r.get('kind'), 'dexvantage.study': r.get('study'),
         'dexvantage.title': r.get('title'), 'dexvantage.interface': r.get('interface'),
         'dexvantage.status': r.get('status'), 'dexvantage.project': proj.get('name'),
         'dexvantage.project_file': proj.get('file'), 'dexvantage.project_sha256': proj.get('sha256'),
         'dexvantage.variant': var.get('name'), 'dexvantage.variant_sha256': var.get('sha256'),
         'dexvantage.version': dv.get('version'), 'dexvantage.code_sha256': dv.get('code_sha256'),
         'dexvantage.result_sha256': (r.get('result') or {}).get('sha256'),
         'dexvantage.headline': (r.get('headline') or {}).get('text'),
         'dexvantage.tags': ','.join(r.get('tags') or []), 'dexvantage.rerun_of': r.get('rerun_of'),
         'dexvantage.reproduce': (r.get('reproduce') or {}).get('verdict'), 'dexvantage.reason': r.get('reason')}
    if git.get('commit'):
        t['mlflow.source.git.commit'] = git['commit']
        t['dexvantage.git_dirty'] = git.get('dirty')
    for tag in r.get('tags') or []:
        t[f'tag.{tag}'] = 'true'
    ns = [k for k, m in (r.get('metrics') or {}).items() if m.get('value') is None]
    if ns:
        t['dexvantage.not_stated'] = ','.join(ns)
    for k, m in (r.get('metrics') or {}).items():
        if m.get('unit'):
            t[f'dexvantage.unit.{names[k]}'] = m['unit']
    out = {}
    for k, v in t.items():
        if v is None or v == '':
            continue
        out.setdefault(mlflow_name(k) if not k.startswith('mlflow.') else k, _value_text(v)[:MAX_TAG])
    return out


def metric_names(r: dict) -> dict:
    """{metric key: MLflow metric name}, unique within the run (a clash gets _2, _3 …)."""
    out, used = {}, set()
    for k in sorted(r.get('metrics') or {}):
        n = base = mlflow_name(k)
        i = 2
        while n in used:
            n = f'{base}_{i}'[:MAX_KEY]
            i += 1
        used.add(n)
        out[k] = n
    return out


def write_run(root: str, r: dict, artifacts: dict | None = None) -> str:
    """Write one run (a record of core/tracking/record.py) into the mlruns folder root; returns its folder."""
    name = experiment_name(r)
    eid = experiment_id(name)
    start = epoch_ms(r.get('started_utc')) or 0
    end = epoch_ms(r.get('ended_utc'))
    _experiment(root, eid, name, start)
    rid = r['run_id']
    run_dir = os.path.join(root, eid, rid)
    tmp = run_dir + '.part'
    shutil.rmtree(tmp, ignore_errors=True)
    for sub in ('metrics', 'params', 'tags', 'artifacts'):
        os.makedirs(os.path.join(tmp, sub), exist_ok=True)
    names = metric_names(r)
    ts = end if end is not None else start
    for k, m in (r.get('metrics') or {}).items():
        v = m.get('value')
        if v is None:
            continue
        _text(os.path.join(tmp, 'metrics', names[k]), f'{ts} {float(v)!r} 0\n')
    for k, v in run_params(r).items():
        _text(os.path.join(tmp, 'params', k), v)
    for k, v in run_tags(r, names).items():
        _text(os.path.join(tmp, 'tags', k), v)
    for fname, data in (artifacts or {}).items():
        write_atomic(os.path.join(tmp, 'artifacts', fname), data)
    _yaml(os.path.join(tmp, 'meta.yaml'), {
        'artifact_uri': _file_uri(os.path.join(run_dir, 'artifacts')), 'end_time': end, 'entry_point_name': '',
        'experiment_id': eid, 'lifecycle_stage': 'active', 'run_id': rid, 'run_name': r.get('name') or rid,
        'source_name': '', 'source_type': SOURCE_TYPE_LOCAL, 'source_version': '',
        'start_time': start, 'status': STATUS.get(r.get('status') or 'finished', 3), 'tags': [], 'user_id': '',
        'deleted_time': None})
    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    os.rename(tmp, run_dir)
    return run_dir


def write(runs, out_dir: str, store=None) -> dict:
    """Write runs into out_dir/ as an MLflow file store (out_dir is the mlruns folder itself). store: the RunStore
    the runs come from (their result.json and run.json go in as artifacts). {'folder', 'experiments', 'runs'}."""
    root = os.path.abspath(out_dir)
    os.makedirs(os.path.join(root, '.trash'), exist_ok=True)
    first = min((epoch_ms(r.get('started_utc')) or 0 for r in runs), default=0)
    _experiment(root, DEFAULT_EXPERIMENT_ID, 'Default', first)
    exps = set()
    for r in runs:
        arts = {}
        if store is not None:
            for f in ('result.json', 'run.json'):
                p = os.path.join(store.run_dir(r['run_id']), f)
                if os.path.isfile(p):
                    with open(p, 'rb') as fh:
                        arts[f] = fh.read()
        write_run(root, r, arts)
        exps.add(experiment_id(experiment_name(r)))
    return {'folder': os.path.basename(root), 'experiments': len(exps), 'runs': len(runs),
            'mlflow_layout': f'MLflow {MLFLOW_VERSION_READ} FileStore'}


def read_layout(root: str) -> dict:
    """Read an mlruns folder back by its layout alone (for the checks when MLflow is not installed): {run_id:
    {'experiment_id', 'meta', 'metrics' {name: [(ts, value, step)]}, 'params', 'tags'}} and the experiments."""
    import yaml
    exps, runs = {}, {}
    for eid in sorted(os.listdir(root)):
        d = os.path.join(root, eid)
        if eid.startswith('.') or not os.path.isdir(d):
            continue
        with open(os.path.join(d, 'meta.yaml'), encoding='utf-8') as fh:
            exps[eid] = yaml.safe_load(fh)
        for rid in sorted(os.listdir(d)):
            rd = os.path.join(d, rid)
            if not os.path.isdir(rd) or rid in ('tags', 'datasets', 'traces', 'models') or rid.endswith('.part'):
                continue                       # a '.part' folder: a write_run stopped before its rename
            with open(os.path.join(rd, 'meta.yaml'), encoding='utf-8') as fh:
                meta = yaml.safe_load(fh)
            run = {'experiment_id': eid, 'meta': meta, 'metrics': {}, 'params': {}, 'tags': {}}
            for sub in ('metrics', 'params', 'tags'):
                sd = os.path.join(rd, sub)
                for n in sorted(os.listdir(sd)) if os.path.isdir(sd) else []:
                    with open(os.path.join(sd, n), encoding='utf-8') as fh:
                        text = fh.read()
                    if sub == 'metrics':
                        run[sub][n] = [(int(a), float(b), int(c)) for a, b, c in
                                       (ln.split(' ') for ln in text.strip().splitlines())]
                    else:
                        run[sub][n] = text
            runs[rid] = run
    return {'experiments': exps, 'runs': runs}
