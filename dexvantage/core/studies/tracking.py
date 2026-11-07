"""The experiment tracker's summary as a study: the project's run store read back (core/tracking), so the Runs page,
batch mode (--runs OUT.json) and the assistant read it like every other result.

    @jobs.study('tracking', 'Runs', quick=False, batch='runs')   -> --runs OUT.json

Settings: project.analysis.studies['tracking'] (core/tracking DEFAULTS, each a labelled default: enabled false,
folder '' = <project file name>.runs beside the project, quick false = the quick studies the window re-runs after
each edit are not logged, tags [] added to every run). In batch mode --track DIR names the store instead.

The result: whether tracking is on, the store's folder name (never its path), every run (newest first, at most
MAX_ROWS) with its study, status, headline, tags, note and figures, the counts by study, kind and status, what the
store's health check found (index lines that do not parse, run folders the index does not name, indexed runs whose
folder is gone), and a headline: the number of runs with a status word ('tracking on', 'tracking off', 'no runs yet',
'not saved', 'damaged lines'). The suggestion when tracking is off is to turn it on (the page's Apply).

Known limits: the summary is of the store as it was read (runs logged later need a new read: the page reads again
after each run it logs); at most MAX_ROWS runs are listed (the counts cover all); see core/tracking for the
tracker's own limits.
"""
from __future__ import annotations

import os

from .. import jobs
from .. import tracking as TR

SID = 'tracking'
MAX_ROWS = 2000
KNOWN_LIMITS = ('The summary is of the store as read; at most 2000 runs are listed (the counts cover all); a run '
                'records the project\'s content hash, not the project itself.')


def _batch_folder() -> str | None:
    from .. import provenance as PV
    b = PV.current_batch()
    args = b.get('args') if b else None
    folder = getattr(args, 'track', None) if args is not None else None
    return os.path.abspath(folder) if folder else None          # --track DIR: from the current folder


def row(r: dict) -> dict:
    """A run as the page's table lists it."""
    return {'run_id': r.get('run_id'), 'started_utc': r.get('started_utc'), 'ended_utc': r.get('ended_utc'),
            'seconds': r.get('seconds'), 'kind': r.get('kind'), 'study': r.get('study'), 'title': r.get('title'),
            'status': r.get('status'), 'reason': r.get('reason') or '', 'interface': r.get('interface'),
            'project': (r.get('project') or {}).get('name'), 'project_sha256': (r.get('project') or {}).get('sha256'),
            'variant': (r.get('variant') or {}).get('name'), 'headline': r.get('headline') or {},
            'tags': list(r.get('tags') or []), 'note': r.get('note') or '', 'rerun_of': r.get('rerun_of'),
            'reproduce': (r.get('reproduce') or {}).get('verdict'), 'folder_missing': bool(r.get('folder_missing')),
            'metrics': {k: {'value': m.get('value'), 'unit': m.get('unit', ''), 'status': m.get('status'),
                            'title': m.get('title', k), 'better': m.get('better', ''),
                            **({'bound': m['bound']} if m.get('bound') else {})}
                        for k, m in (r.get('metrics') or {}).items()}}


def _count(rows, key) -> dict:
    out: dict = {}
    for r in rows:
        k = r.get(key) or ''
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


@jobs.study(SID, 'Runs', quick=False, batch='runs', needs_robot=False)
def run_tracking(project, ctx: jobs.Ctx) -> dict:
    st = TR.settings_of(project)
    folder = _batch_folder()
    base = {'enabled': st['enabled'] or bool(folder), 'settings': st, 'defaults': dict(TR.DEFAULTS),
            'known_limits': KNOWN_LIMITS}
    try:
        store = TR.store_for(project, folder)
    except TR.TrackingError as e:
        word = 'not saved' if st['enabled'] else 'tracking off'
        return dict(base, store=None, store_exists=False, runs=[], counts={'runs': 0},
                    headline={'value': 0, 'unit': 'runs', 'status': 'serious' if st['enabled'] else 'warning',
                              'word': word, 'text': str(e)},
                    suggestion=None if st['enabled'] else _suggest())
    ctx.report(0.1, 'reading the run store')
    scan = store.scan() if store.exists() else {'runs': {}, 'damaged_lines': 0, 'recovered': 0, 'missing': 0}
    rows = TR.filter_runs(list(scan['runs'].values()))
    rows.reverse()                                       # newest first
    counts = {'runs': len(rows), 'by_study': _count(rows, 'study'), 'by_kind': _count(rows, 'kind'),
              'by_status': _count(rows, 'status'), 'damaged_lines': scan['damaged_lines'],
              'recovered': scan['recovered'], 'missing': scan['missing']}
    n = len(rows)
    if not base['enabled']:
        sev, word = 'warning', 'tracking off'
        text = (f'Tracking is off: {n} run{"" if n == 1 else "s"} in {store.name} from before. Turn it on to log every '
                'study run, experiment, optimiser run, tolerance Monte Carlo and scorecard.') if n else \
            ('Tracking is off: no run is logged. Turn it on (Apply) to keep every study run with its settings, '
             f'figures and provenance in {store.name} beside the project.')
    elif counts['damaged_lines']:
        sev, word = 'serious', 'damaged lines'
        text = (f'{n} runs; {counts["damaged_lines"]} index line{"" if counts["damaged_lines"] == 1 else "s"} of '
                f'{store.name} could not be read (a writer stopped mid-line or the file was edited): the runs\' own '
                'folders are intact and are read instead.')
    elif not n:
        sev, word = 'warning', 'no runs yet'
        text = f'Tracking is on: the next study run is logged in {store.name}.'
    else:
        sev, word = 'good', 'tracking on'
        last = rows[0]
        text = (f'{n} run{"" if n == 1 else "s"} in {store.name}; the latest: {last.get("title")} '
                f'({last.get("status")}) at {(last.get("started_utc") or "")[:16].replace("T", " ")} UTC.')
    ctx.report(1.0, '')
    return dict(base, store=store.name, store_exists=store.exists(), fingerprint=store.fingerprint(),
                runs=[row(r) for r in rows[:MAX_ROWS]], truncated=max(0, n - MAX_ROWS), counts=counts,
                headline={'value': n, 'unit': 'runs', 'status': sev, 'word': word, 'text': text},
                suggestion=None if base['enabled'] else _suggest())


def _suggest() -> dict:
    return {'action': 'enable', 'text': 'Turn tracking on: every study run, experiment, optimiser run, tolerance '
                                        'Monte Carlo and scorecard is then logged with its settings, figures and '
                                        'provenance (one undo step; the setting is not part of the project\'s '
                                        'content hash).'}
