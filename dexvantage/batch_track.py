"""Batch mode's experiment tracker options (core/tracking), added to batch.py's parser:

    dexvantage PROJECT.dexv --coverage out.json --track RUNS_DIR [--track-tag T] [--track-note TEXT]
        log every study run, --experiment, --check and --scorecard of this command as a run in RUNS_DIR
        (a project with tracking turned on logs into its own store without --track)
    dexvantage PROJECT.dexv --runs summary.json [--track RUNS_DIR]
        the run store summarised (the 'tracking' study)
    dexvantage [PROJECT.dexv] --runs-export OUT_DIR [--track RUNS_DIR] [--runs-formats csv,parquet,mlflow,wandb]
        the runs as CSV and Parquet tables, an MLflow mlruns/ folder and a W&B replay script, with a manifest

Exit codes as batch.py: 0 done, 2 an error (one sentence on stderr). A run that cannot be logged (a store that
cannot be written) is said in a sentence and makes the command exit 2 after its other outputs are written.
"""
from __future__ import annotations

import os
import sys

EXIT_OK, EXIT_FAIL, EXIT_ERROR = 0, 1, 2
_FAILED: list = []                     # sentences of runs that could not be logged during this command


def add_arguments(g):
    g.add_argument('--track', metavar='RUNS_DIR', help='log every study run, experiment and check of this command '
                   'as a run in the run store RUNS_DIR (core/tracking; created if missing)')
    g.add_argument('--track-tag', action='append', metavar='TAG', help='with --track (or a project with tracking '
                   'on): a tag for the runs logged (repeat for more)')
    g.add_argument('--track-note', metavar='TEXT', help='with --track: a note for the runs logged')
    g.add_argument('--runs-export', metavar='OUT_DIR', help='export the run store (--track RUNS_DIR, else the '
                   'project\'s) as CSV and Parquet tables, an MLflow mlruns/ folder and a W&B replay script')
    g.add_argument('--runs-formats', metavar='A,B', help='with --runs-export: csv, parquet, mlflow, wandb (default: '
                   'all four)')


def wanted(args) -> bool:
    return bool(getattr(args, 'track', None) or getattr(args, 'track_tag', None) or
                getattr(args, 'track_note', None) or getattr(args, 'runs_export', None) or
                getattr(args, 'runs_formats', None))


def _err(text) -> int:
    print(f'dexvantage: {text}', file=sys.stderr)
    return EXIT_ERROR


def _args():
    from .core import provenance as PV
    b = PV.current_batch()
    return b.get('args') if b else None


def _stream(args):
    """Where the tracker's lines go: standard error when some output of the command is standard output ('-')."""
    vals = [] if args is None else [x for v in vars(args).values() for x in (v if isinstance(v, (list, tuple))
                                                                             else [v])]
    return sys.stderr if '-' in vals else sys.stdout


def _store(args, project):
    """The store this command logs into: --track DIR, else the project's when its tracking is on; else None."""
    from .core import tracking as TR
    folder = getattr(args, 'track', None) if args is not None else None
    if folder:
        return TR.RunStore(folder)
    if project is not None and TR.enabled(project):
        return TR.store_for(project)
    return None


def record(study: str, doc: dict | None, project, dest=None, kind: str | None = None, reason: str = '') -> dict | None:
    """Log one result of this batch command (doc stamped already) when --track or the project's tracking asks for
    it; a failure (doc None) with its reason. Never raises: a store that cannot be written is said once at the end."""
    from .core import tracking as TR
    args = _args()
    if study in TR.NOT_LOGGED:
        return None
    try:
        store = _store(args, project)
        if store is None:
            return None
        tags = list(getattr(args, 'track_tag', None) or []) if args is not None else []
        note = (getattr(args, 'track_note', None) or '') if args is not None else ''
        if doc is None:
            rec = TR.log_failure(study, reason, project, store=store, kind=kind, interface='batch', tags=tags,
                                 note=note)
        else:
            rec = TR.log(study, doc, project, store=store, kind=kind, interface='batch', tags=tags, note=note)
    except TR.TrackingError as e:
        _FAILED.append(str(e))
        return None
    except OSError as e:
        _FAILED.append(f'The run could not be logged: {e.strerror or e}.')
        return None
    print(f'  logged run {rec["run_id"][:8]} in {store.name}', file=_stream(args))   # '-': stdout holds JSON
    return rec


def run(args, project) -> int:
    """--runs-export, the option checks, and the sentence for runs that could not be logged."""
    from .core import tracking as TR
    code = EXIT_OK
    if (getattr(args, 'track_tag', None) or getattr(args, 'track_note', None)) and not getattr(args, 'track', None) \
            and not (project is not None and TR.enabled(project)):
        return _err('--track-tag and --track-note go with --track RUNS_DIR (or a project with tracking turned on)')
    if getattr(args, 'runs_formats', None) and not getattr(args, 'runs_export', None):
        return _err('--runs-formats goes with --runs-export OUT_DIR')
    if getattr(args, 'runs_export', None):
        try:
            store = _store(args, project) if getattr(args, 'track', None) else (
                TR.store_for(project) if project is not None else None)
            if store is None:
                return _err('--runs-export needs the run store: give --track RUNS_DIR or a project file')
            if not store.exists():
                return _err(f'There is no run store at {store.name}: log runs with --track first.')
            out = TR.export(store, args.runs_export, getattr(args, 'runs_formats', None) or TR.FORMATS)
        except TR.TrackingError as e:
            return _err(str(e))
        for f, row in out['formats'].items():
            if row['status'] == 'ok':
                print(f'{f}: {len(row["files"])} file{"" if len(row["files"]) == 1 else "s"}' +
                      ''.join(f'\n  · {n}' for n in row.get('notes') or []))
            else:
                print(f'{f}: refused: {row.get("reason", "")}', file=sys.stderr)
                code = EXIT_FAIL
        print(f'{out["runs"]} runs exported; manifest → {os.path.join(args.runs_export, out["manifest"])}')
    if getattr(args, 'track', None) and not _FAILED:
        try:
            n = len(TR.RunStore(args.track).runs())
        except TR.TrackingError:
            n = None
        if n is not None:
            print(f'Run store {os.path.basename(os.path.abspath(args.track))}: {n} run{"" if n == 1 else "s"}',
                  file=_stream(args))
    if _FAILED:
        for s in dict.fromkeys(_FAILED):
            _err(s)
        _FAILED.clear()
        code = EXIT_ERROR
    return code
