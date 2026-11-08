"""The experiment tracker: every study run, designed experiment, optimiser run, tolerance Monte Carlo and readiness
scorecard (--check) can be logged as a run, kept in a run store beside the project, compared, exported (CSV,
Parquet, an MLflow file store, a Weights & Biases replay) and re-run.

    from dexvantage.core import tracking as TR
    TR.settings_of(project)             {'enabled', 'folder', 'quick', 'tags'} (project.analysis.studies['tracking'])
    TR.store_for(project)               the project's RunStore (default folder: <project file name>.runs beside it)
    TR.log(study, doc, project, store=…, interface='batch', tags=…, note=…) -> the run record
    TR.log_failure(study, reason, project, …)
    store.runs(study=…, tag=…, since=…, until=…)   index records, oldest first (core/tracking/store.py)
    TR.compare([full records])          metrics side by side with deltas against the first, settings that differ,
                                        and a significance note (core/stats when present)
    TR.rerun(store, run_id, project)    re-run a run's result (core/reproduce) and log the re-run as a new run
    TR.export(store, out_dir, formats=('csv', 'parquet', 'mlflow', 'wandb'))

What a run records: core/tracking/record.py (run id, start and end, the project's content hash, name and variant,
study and settings, the metrics with units and status, the result file and its provenance: DexVantage version, git
commit when from a checkout, library versions, seeds; tags and a note). The store: core/tracking/store.py (JSON lines
plus one folder per run, atomic renames and a file lock: safe for several writers at once).

Opt-in: nothing is logged unless the project turns tracking on (the Runs page, or
project.analysis.studies['tracking'] = {'enabled': true}) or batch mode is given --track DIR. The tracking setting is
not part of the project's content hash (core/provenance.py leaves it out, as it does the live toggle): turning it on
does not make the project "differ" from runs logged before. In the window, the quick studies that re-run after each
edit are logged only when the setting 'quick' is on (they would log a run per edit otherwise), a result already
logged in the session (the same study, project hash and result hash) is not logged twice, and a result made before
the latest edit is not logged (its project is no longer the one in the window).

Known limits: a run records the project's content hash, not the project itself (keep the project file, or its Git
history, to re-run an old run; core/reproduce compares a re-run on the project as it is now and says when it
differs); a failed run records its reason but no result; significance needs replicates (a tolerance Monte Carlo's
draws): single figures are compared by their difference only; the MLflow and W&B exports are written for MLflow
3.17.2's file store layout and wandb 0.30.0's API (see core/tracking/mlruns.py and wandb_replay.py for theirs).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from typing import TYPE_CHECKING

from .record import KINDS, NOT_LOGGED, SCHEMA, build, kind_of, metrics_of  # noqa: F401
from .store import RunStore, StoreError, clean_tags, filter_runs, utc_now  # noqa: F401

if TYPE_CHECKING:                                    # names for the type hints only
    import datetime

    from ..project import Project

SETTINGS_KEY = 'tracking'
DEFAULTS = {'enabled': False, 'folder': '', 'quick': False, 'tags': []}
STORE_SUFFIX = '.runs'
FORMATS = ('csv', 'parquet', 'mlflow', 'wandb')
EXPORT_SCHEMA = 'dexvantage.runs_export/1'
TrackingError = StoreError


# ====================================================================== settings and the store
def settings_of(project: Project | None) -> dict:
    """The project's tracking settings with the defaults filled in (DEFAULTS: off, the default folder, quick studies
    not logged, no tags)."""
    raw = project.analysis.studies.get(SETTINGS_KEY) if project is not None else None
    raw = raw if isinstance(raw, dict) else {}
    out = dict(DEFAULTS)
    out['enabled'] = bool(raw.get('enabled', False))
    out['folder'] = str(raw.get('folder') or '')
    out['quick'] = bool(raw.get('quick', False))
    out['tags'] = clean_tags(raw.get('tags') or [])
    return out


def enabled(project: Project | None) -> bool:
    return project is not None and settings_of(project)['enabled']


def default_folder_name(project: Project | None) -> str:
    base = os.path.splitext(os.path.basename(getattr(project, 'path', '') or ''))[0]
    return (base or 'project') + STORE_SUFFIX


def store_folder(project: Project, folder: str | None = None) -> str:
    """The run store folder: folder if given, else the project's setting (relative paths from the project file's
    folder), else <project file name>.runs beside the project file. A project never saved has no folder beside it:
    TrackingError says so."""
    f = folder if folder else settings_of(project)['folder'] if project is not None else ''
    if f and os.path.isabs(os.path.expanduser(f)):
        return os.path.abspath(os.path.expanduser(f))
    path = getattr(project, 'path', '') or ''
    if not path:
        raise TrackingError('The project is not saved yet, so it has no run store beside it: save it (File ▸ Save) '
                            'or choose a run store folder.')
    here = os.path.dirname(os.path.abspath(path))
    return os.path.abspath(os.path.join(here, f or default_folder_name(project)))


def store_for(project: Project, folder: str | None = None) -> RunStore:
    return RunStore(store_folder(project, folder))


def should_log(study: str, project: Project | None, requested: bool = True) -> bool:
    """Whether a run of study is logged from this project: tracking on, not the tracker's own summary, and (for a
    quick study the window re-runs after each edit, requested=False) only when the setting 'quick' is on."""
    from .. import jobs
    if study in NOT_LOGGED or not enabled(project):
        return False
    info = jobs.STUDIES.get(study)
    if not requested and info is not None and info.quick and not settings_of(project)['quick']:
        return False
    return True


# ====================================================================== logging
def log(study: str, doc: dict, project: Project | None, *, store: RunStore | None = None, folder: str | None = None,
        kind: str | None = None, interface: str = 'app', tags: tuple | list = (), note: str = '',
        started: str | datetime.datetime | None = None, ended: str | datetime.datetime | None = None,
        rerun_of: str | None = None, reproduce: dict | None = None) -> dict:
    """Log one run of study (its result doc) made on project into store (default: the project's). The project's
    own tracking tags are added to tags. Returns the record as stored."""
    store = store or store_for(project, folder)
    t = clean_tags(list(settings_of(project)['tags'] if project is not None else []) + clean_tags(tags))
    rec, stamped = build(study, doc, project, kind=kind, interface=interface, tags=t, note=note, started=started,
                         ended=ended, rerun_of=rerun_of, reproduce=reproduce)
    return store.add(rec, stamped)


def log_failure(study: str, reason: str, project: Project | None, *, store: RunStore | None = None,
                folder: str | None = None, kind: str | None = None, interface: str = 'app', tags: tuple | list = (),
                note: str = '', started: str | datetime.datetime | None = None) -> dict:
    """Log a run that could not run (a study's refusal, in its plain sentence)."""
    store = store or store_for(project, folder)
    t = clean_tags(list(settings_of(project)['tags'] if project is not None else []) + clean_tags(tags))
    rec, _ = build(study, None, project, kind=kind, interface=interface, status='failed', reason=reason, tags=t,
                   note=note, started=started)
    return store.add(rec, None)


class SessionLog:
    """The window's logger: one per window; remembers what it logged so the same result is not logged twice (a
    result from the cache, or a page that re-shows a result). Thread-safe; log() runs on a worker thread."""

    def __init__(self):
        self._seen: set = set()
        self._lock = threading.Lock()

    def key(self, store: RunStore, study: str, doc: dict, project: Project) -> tuple:
        from .. import provenance as PV
        doc = {k: v for k, v in doc.items() if k not in ('cached', 'revision', PV.KEY)}
        return (store.folder, study, PV.project_digest(project)['sha256'],
                project.variant.uid if project.variants else '', PV.result_digest(doc))

    def log(self, study: str, doc: dict, project: Project | None, **kw: object) -> dict | None:
        """Log unless already logged this session; None when it was."""
        store = kw.pop('store', None) or store_for(project)
        k = self.key(store, study, doc, project)
        with self._lock:
            if k in self._seen:
                return None
            self._seen.add(k)
        try:
            return log(study, doc, project, store=store, **kw)
        except Exception:
            with self._lock:
                self._seen.discard(k)
            raise


# ====================================================================== comparing
def _flat(o: object, prefix: str = '') -> dict:
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(_flat(v, f'{prefix}.{k}' if prefix else str(k)))
    else:
        out[prefix] = o
    return out


def significance(a: list, b: list, better: str, paired: bool = False) -> dict:
    """A note on whether two runs' figures differ beyond chance: core/stats.compare_samples(a, b, better, paired)
    when this build has core/stats (it returns {'text', 'p', 'test', ...}; paired when both runs drew with the same
    seed, so draw k of each saw the same perturbation); else, or with fewer than two replicates a side, a note saying
    why there is no test (never a made-up p-value). A draw without a value (None) drops its pair when paired (the
    lists are by draw), the value alone when not."""
    paired = bool(paired and len(a) == len(b))
    if paired:
        keep = [(x, y) for x, y in zip(a, b) if finite(x) and finite(y)]
        a, b = [x for x, _ in keep], [y for _, y in keep]
    else:
        a, b = [x for x in a if finite(x)], [y for y in b if finite(y)]
    if len(a) < 2 or len(b) < 2:
        return {'available': False, 'text': 'one value per run: a significance test needs replicates (a tolerance '
                                            'Monte Carlo\'s draws, repeated recordings)'}
    try:
        from .. import stats as ST
    except ImportError:
        return {'available': False, 'text': f'{len(a)} and {len(b)} replicates; no test: core/stats is not in this '
                                            'build'}
    fn = getattr(ST, 'compare_samples', None)
    if fn is None:
        return {'available': False, 'text': 'core/stats has no compare_samples(a, b, better): no test'}
    try:
        res = fn(list(a), list(b), better if better in ('higher', 'lower') else 'higher',
                 paired=bool(paired and len(a) == len(b)))
    except Exception as e:                      # noqa: BLE001 — a note, never a crash of the comparison
        return {'available': False, 'text': f'core/stats could not compare them ({type(e).__name__}: {e})'}
    res = dict(res) if isinstance(res, dict) else {'text': str(res)}
    res.setdefault('available', True)
    return res


def compare(records: list, keys: list | None = None) -> dict:
    """The runs side by side: for each metric key (default: every key any run states), each run's value, its delta
    and relative change against the first run (the baseline), which run is best by the metric's direction, and a
    significance note; plus the settings and inputs that differ between the runs. records: full run records
    (RunStore.get, with samples)."""
    if len(records) < 2:
        raise TrackingError('Choose two or more runs to compare.')
    ids = [r['run_id'] for r in records]
    allkeys = []
    for r in records:
        allkeys += [k for k in (r.get('metrics') or {}) if k not in allkeys]
    keys = list(keys) if keys else sorted(allkeys)
    seeds = [r.get('seeds') or {} for r in records]
    same_seed = len(records) == 2 and bool(seeds[0]) and seeds[0] == seeds[1]
    rows = []
    for k in keys:
        ms = [(r.get('metrics') or {}).get(k) or {} for r in records]
        vals = [m.get('value') for m in ms]
        meta = next((m for m in ms if m), {})
        base = vals[0]
        deltas = [None if (v is None or base is None) else v - base for v in vals]
        rel = [None if (d is None or not base) else d / abs(base) for d in deltas]
        better = meta.get('better') or ''
        known = [(v, i) for i, v in enumerate(vals) if v is not None]
        best = None
        if known and better in ('higher', 'lower') and len({v for v, _ in known}) > 1:
            best = (max if better == 'higher' else min)(known, key=lambda t: t[0])[1]
        sig = None
        if len(records) == 2 and vals[0] is not None and vals[1] is not None and not meta.get('stat'):
            sa = (records[0].get('samples') or {}).get(k) or []
            sb = (records[1].get('samples') or {}).get(k) or []
            sig = significance(sa, sb, better, paired=same_seed)
        rows.append({'key': k, 'title': meta.get('title', k), 'unit': meta.get('unit', ''), 'better': better,
                     'values': vals, 'status': [m.get('status') for m in ms], 'deltas': deltas, 'relative': rel,
                     'best': best, 'significance': sig, 'same': len({json.dumps(v) for v in vals}) == 1})
    diffs = []
    flats = [_flat({'settings': r.get('settings') or {}, 'seeds': r.get('seeds') or {},
                    'project.sha256': (r.get('project') or {}).get('sha256'),
                    'variant.name': (r.get('variant') or {}).get('name'),
                    'dexvantage.version': ((r.get('provenance') or {}).get('dexvantage') or {}).get('version'),
                    'dexvantage.code_sha256': ((r.get('provenance') or {}).get('dexvantage') or {})
                    .get('code_sha256')}) for r in records]
    for p in sorted(set().union(*flats)):
        vs = [f.get(p) for f in flats]
        if len({json.dumps(v, sort_keys=True, default=str) for v in vs}) > 1:
            diffs.append({'path': p, 'values': vs})
    notes = []
    if len({r.get('study') for r in records}) > 1:
        notes.append('The runs are of different studies: only the metrics they share line up.')
    if len({(r.get('project') or {}).get('sha256') for r in records}) > 1:
        notes.append('The runs were made on different project contents (their content hashes differ).')
    return {'runs': ids, 'baseline': ids[0], 'names': [r.get('name') for r in records],
            'started': [r.get('started_utc') for r in records], 'rows': rows, 'differences': diffs, 'notes': notes}


def delta_text(v: float | None, unit: str = '') -> str:
    """A change against the baseline run with its unit: a change of a percentage is in percentage points ('pp')."""
    if v is None:
        return '—'
    from ..metrics import delta_unit
    s = f'{v:+.4g}'
    return f'{s} {delta_unit(unit)}'.strip()


# ====================================================================== re-running
def rerun(store: RunStore, run_id: str, project: Project, *, interface: str = 'app', tags: tuple | list = (),
          dataset: str | None = None, log_it: bool = True) -> tuple[dict | None, dict]:
    """Re-run what a run's result was made from on project (core/reproduce: the same study and settings as the
    project now holds) and compare every number. (the new run's record, logged with rerun_of and the verdict, or None
    when log_it is False; the reproduction report)."""
    from .. import provenance as PV
    from .. import reproduce as RP
    rec = store.get(run_id)
    recorded = store.result(run_id)
    if recorded is None:
        raise TrackingError(f'Run {run_id[:8]} kept no result (it {rec.get("status") or "did not finish"}), so '
                            'there is nothing to re-run.')
    now = RP.rerun(recorded, project, dataset)          # StudyError (plain sentence) when it cannot be re-run
    stamped = PV.stamp(now, project)
    tol = RP.tolerances_with_defaults(())
    rep = PV.report('result.json', recorded, stamped, tol, now_provenance=stamped[PV.KEY],
                    project_now=PV.project_digest(project))
    summary = {'verdict': rep.get('verdict'), 'verdict_text': rep.get('verdict_text'), 'message': rep.get('message'),
               'reasons': list(rep.get('reasons') or [])[:10]}
    new = None
    if log_it:
        new = log(rec.get('study') or recorded.get('study'), stamped, project, store=store, kind=rec.get('kind'),
                  interface=interface, tags=clean_tags(list(tags) + ['rerun']), rerun_of=run_id, reproduce=summary)
    return new, rep


# ====================================================================== export
def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def export(store: RunStore, out_dir: str, formats: str | list | tuple = FORMATS, runs: list | None = None,
           **filters: object) -> dict:
    """Write the runs (default: every run of the store, or those the filters of RunStore.runs pick) into out_dir:
    runs.csv, runs_long.csv, units.json (csv); runs.parquet, runs_long.parquet (parquet); mlruns/ (mlflow);
    wandb/wandb_runs.json and wandb/replay_wandb.py (wandb); and export_runs_manifest.json (every file's SHA-256 and
    the provenance block). {'files', 'formats': {fmt: {'status', 'files', 'notes'|'reason'}}, 'runs', 'manifest'}."""
    from .. import provenance as PV
    from . import mlruns as ML
    from . import tables as TB
    from . import wandb_replay as WB
    fmts = [f.strip().lower() for f in (formats.split(',') if isinstance(formats, str) else formats) if f.strip()]
    bad = [f for f in fmts if f not in FORMATS]
    if bad or not fmts:
        raise TrackingError(f'Unknown runs export format{"s" if len(bad) > 1 else ""} {", ".join(bad) or "(none)"}: '
                            f'the formats are {", ".join(FORMATS)}.')
    rows = runs if runs is not None else store.runs(**filters)
    if not rows:
        raise TrackingError(f'The run store {store.name} has no run to export' +
                            (' with those filters.' if filters else '.'))
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        raise TrackingError(f'Could not create {os.path.basename(out_dir)}: {e.strerror or e}.') from None
    out: dict = {}
    files: list = []
    for f in fmts:
        try:
            if f == 'csv':
                w = TB.write_csv(rows, os.path.join(out_dir, 'runs.csv'), 'wide')
                TB.write_csv(rows, os.path.join(out_dir, 'runs_long.csv'), 'long')
                from .store import dumps, write_atomic
                write_atomic(os.path.join(out_dir, 'units.json'), dumps(w['units'], indent=1))
                fs = ['runs.csv', 'runs_long.csv', 'units.json']
                out[f] = {'status': 'ok', 'files': fs, 'notes': [f'{w["rows"]} runs × {w["columns"]} columns']}
            elif f == 'parquet':
                w = TB.write_parquet(rows, os.path.join(out_dir, 'runs.parquet'), 'wide')
                TB.write_parquet(rows, os.path.join(out_dir, 'runs_long.parquet'), 'long')
                fs = ['runs.parquet', 'runs_long.parquet']
                out[f] = {'status': 'ok', 'files': fs, 'notes': [f'{w["rows"]} runs × {w["columns"]} columns']}
            elif f == 'mlflow':
                w = ML.write(rows, os.path.join(out_dir, 'mlruns'), store)
                fs = []
                for root, _d, names in os.walk(os.path.join(out_dir, 'mlruns')):
                    fs += [os.path.relpath(os.path.join(root, n), out_dir).replace(os.sep, '/') for n in names]
                out[f] = {'status': 'ok', 'files': sorted(fs),
                          'notes': [f'{w["runs"]} runs in {w["experiments"]} experiment(s), {w["mlflow_layout"]} '
                                    'layout; MLflow 3.13 and later read it with MLFLOW_ALLOW_FILE_STORE=true']}
            elif f == 'wandb':
                w = WB.export(rows, os.path.join(out_dir, 'wandb'))
                fs = [f'wandb/{n}' for n in w['files']]
                out[f] = {'status': 'ok', 'files': fs, 'notes': [f'{w["runs"]} runs; run python wandb/replay_wandb.py '
                                                                 '(needs wandb) to replay them offline']}
            files += out[f]['files']
        except ImportError as e:
            out[f] = {'status': 'refused', 'files': [], 'reason': str(e)}
        except OSError as e:
            out[f] = {'status': 'refused', 'files': [], 'reason': f'could not write: {e.strerror or e}'}
    entries = []
    for rel in files:
        p = os.path.join(out_dir, rel)
        if os.path.isfile(p):
            entries.append({'path': rel, 'bytes': os.path.getsize(p), 'sha256': _sha(p)})
    man = {'schema': EXPORT_SCHEMA, 'store': store.name, 'runs': len(rows),
           'run_ids': [r['run_id'] for r in rows], 'formats': out, 'files': entries}
    man = PV.stamp(man, None)
    from .store import dumps, write_atomic
    write_atomic(os.path.join(out_dir, 'export_runs_manifest.json'), dumps(man, indent=1))
    return {'files': files, 'formats': out, 'runs': len(rows), 'manifest': 'export_runs_manifest.json'}


def rows_text(r: dict) -> str:
    """One line for a run: when, what, the headline."""
    h = (r.get('headline') or {}).get('text') or (r.get('reason') or '')
    return f'{(r.get("started_utc") or "")[:19].replace("T", " ")} · {r.get("title") or r.get("study")} · ' \
           f'{r.get("status")}' + (f' · {h}' if h else '')


def finite(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
