"""A run record (schema dexvantage.run/1) built from a result DexVantage wrote: what ran, on what, with which
settings, the figures it gave and where its result and provenance are.

    build(study, doc, project, kind=None, interface='app', status='finished', ...) -> (record, stamped result)

The record's fields:

- run_id (uuid4 hex), schema, kind ('study', 'experiment', 'optimise', 'tolerance', 'check'), study (the core.jobs
  study id), title, name (a display name), interface ('app', 'batch', 'api').
- started_utc, ended_utc (UTC, millisecond text), seconds (the study's own computation time when it states one),
  status ('finished', 'cancelled', 'failed'), reason (a failed run's plain sentence).
- project {name, file (its file name, never its folder), sha256 (core/provenance content hash)}, variant {name,
  index, uid, sha256}.
- settings (the study's settings with defaults filled in, as core/provenance records them), seeds.
- metrics {key: {metric, stat, title, unit, better, value, status, bound?, reason?, check?}}: key is a core/metrics
  id ('coverage.covered') or, for a figure of a distribution or of a search, the id with a statistic
  ('coverage.covered@p5', 'grasp.contact_visibility@best'). Values are the numbers the result states (core.metrics
  read(): None with status 'not stated' when it does not); nothing is computed here.
- headline {text, status, word} as the result states it.
- result {file 'result.json', schema, sha256 (the result hash of core/provenance), output (where batch wrote it)}.
- provenance {dexvantage (version, code hash, git commit and dirty flag when from a checkout), runtime (Python and
  library versions, platform), interface, command, inputs, plan, created_utc}: core/provenance's block, without a
  path, user name or host name.
- tags, note, rerun_of (the run id this run re-ran) and reproduce {verdict, verdict_text, message} for a re-run.
- samples {key: [numbers]} (run.json only): the per-draw values of a tolerance Monte Carlo, the replicates a
  significance test can use, in draw order (null for a draw that states no value, so the draws of two runs with one
  seed pair by index).
"""
from __future__ import annotations

import datetime as _dt
import math

from .store import clean_tags, new_run_id, parse_utc, utc_now, utc_text

SCHEMA = 'dexvantage.run/1'
KINDS = ('study', 'experiment', 'optimise', 'tolerance', 'check')
KIND_OF_STUDY = {'experiments': 'experiment', 'optimise': 'optimise', 'design_agent': 'optimise',
                 'tolerance': 'tolerance', 'requirements': 'check'}
STATUSES = ('finished', 'cancelled', 'failed')
TOLERANCE_STATS = ('p5', 'p50', 'p95', 'mean', 'std')
STATUS_LEVELS = ('good', 'warning', 'serious', 'critical')
NOT_LOGGED = ('tracking', 'summary')        # studies that are not runs (the tracker's own summary, the project's)


def kind_of(study: str) -> str:
    return KIND_OF_STUDY.get(study or '', 'study')


def _num(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _row(mid, stat, title, unit, better, value, status=None, **extra) -> dict:
    v = _num(value)
    r = {'metric': mid, 'stat': stat, 'title': title, 'unit': unit or '', 'better': better or '',
         'value': v, 'status': status if status and v is not None else ('ok' if v is not None else 'not stated')}
    if v is None and status in ('refused',):
        r['status'] = status
    r.update({k: x for k, x in extra.items() if x not in (None, '')})
    if v is None:
        r.pop('bound', None)
    return r


def _meta(mid: str) -> tuple:
    from .. import metrics as M
    M.load_metrics()
    m = M.METRICS.get(mid)
    return (m.title, m.unit, m.better) if m else (mid, '', '')


def metrics_of(study: str, doc: dict) -> tuple[dict, dict]:
    """(metrics, samples) of a study's result: every core/metrics figure registered for the study, read off the
    result; plus, for a tolerance Monte Carlo, each metric's nominal value and P5 / P50 / P95 / mean / standard
    deviation over the draws (and the draws as samples); for an experiment, each response at the baseline and the
    best run's objective; for the optimiser, each objective of the current rig and the best option's; for a
    scorecard, each requirement's figure with its pass / fail and the counts."""
    from .. import metrics as M
    M.load_metrics()
    out: dict = {}
    samples: dict = {}
    if not isinstance(doc, dict):
        return out, samples
    for m in M.METRICS.values():
        if m.study != study:
            continue
        r = M.read(m, doc)
        out[m.id] = _row(m.id, None, m.title, m.unit, m.better, r.get('value'), r.get('status'),
                         bound=r.get('bound'), reason=r.get('reason'), note=r.get('note'))
    try:
        if study == 'tolerance':
            _tolerance(doc, out, samples)
        elif study == 'experiments':
            _experiment(doc, out)
        elif study in ('optimise', 'design_agent'):
            _optimise(doc, out)
        elif study == 'requirements':
            _scorecard(doc, out)
    except (KeyError, TypeError, AttributeError, IndexError, ValueError):
        pass                                     # a result shaped otherwise: its generic figures only
    return out, samples


def _tolerance(doc, out, samples):
    done = doc.get('done')
    for mid, row in (doc.get('metrics') or {}).items():
        title, unit, better = row.get('title') or mid, row.get('unit', ''), row.get('better', '')
        st = row.get('nominal_status') or None
        out[mid] = _row(mid, None, title, unit, better, row.get('nominal'), st if st in ('ok', 'refused') else None,
                        reason=row.get('reason'))
        for s in TOLERANCE_STATS:
            out[f'{mid}@{s}'] = _row(mid, s, f'{title}, {s.upper() if s.startswith("p") else s} over {done} draws',
                                     unit, better, row.get(s))
        # kept by draw (None where a draw states no value), so draw k of two runs with one seed still line up when
        # each run lost a different draw: a paired comparison drops those pairs, not a shifted list
        vals = [float(x) if _num(x) is not None else None for x in row.get('values') or []]
        if any(v is not None for v in vals):
            samples[mid] = vals


def _experiment(doc, out):
    resp = {r.get('id'): r for r in doc.get('responses') or [] if isinstance(r, dict)}
    for mid, b in (doc.get('baseline') or {}).items():
        title, unit, better = (resp.get(mid, {}).get(k) for k in ('title', 'unit', 'better'))
        t, u, bt = _meta(mid)
        out[mid] = _row(mid, None, title or t, unit if unit is not None else u, better or bt, b.get('value'),
                        b.get('status'), reason=b.get('reason'))
    best = doc.get('best')
    if isinstance(best, dict) and best.get('response'):
        mid = best['response']
        t, u, bt = _meta(mid)
        out[f'{mid}@best'] = _row(mid, 'best', f'{t}, best run (run {best.get("run")})', best.get('unit', u),
                                  best.get('better', bt), best.get('value'),
                                  bound=best.get('bound') if best.get('bound') in ('at_least', 'at_most') else None,
                                  settled=best.get('settled') if isinstance(best.get('settled'), bool) else None)


def _optimise(doc, out):
    cur = (doc.get('current') or {}).get('figures') or {}
    opts = doc.get('options') or []
    for o in doc.get('objectives') or []:
        mid = o.get('id')
        if not mid:
            continue
        out[mid] = _row(mid, None, o.get('title', mid), o.get('unit', ''), o.get('better', ''), cur.get(mid))
        vals = [(_num((x.get('figures') or {}).get(mid)), x.get('label')) for x in opts]
        vals = [(v, lab) for v, lab in vals if v is not None]
        if vals:
            v, lab = (max if o.get('better') == 'higher' else min)(vals, key=lambda t: t[0])
            out[f'{mid}@best'] = _row(mid, 'best', f'{o.get("title", mid)}, best option ({lab})', o.get('unit', ''),
                                      o.get('better', ''), v)
        else:
            out[f'{mid}@best'] = _row(mid, 'best', f'{o.get("title", mid)}, best option', o.get('unit', ''),
                                      o.get('better', ''), None)
    out['optimise.options'] = _row('optimise.options', None, 'Designs offered', 'designs', 'higher',
                                   len(opts) if isinstance(opts, list) else None)


def _scorecard(doc, out):
    for it in doc.get('items') or []:
        mid = it.get('metric')
        if not mid:
            continue
        key = mid if mid not in out or out[mid].get('check') is None else f'{mid}@{it.get("uid", "")}'
        t, u, bt = _meta(mid)
        out[key] = _row(mid, None if key == mid else 'requirement', it.get('title') or t, it.get('unit') or u, bt,
                        it.get('value'), bound=it.get('bound'), reason=it.get('reason'), check=it.get('status'),
                        requirement=f'{it.get("op_text", it.get("op", ""))} {it.get("threshold")}'.strip())
    counts = doc.get('counts') or {}
    for s in ('pass', 'fail', 'not_evaluated'):
        if s in counts:
            out[f'requirements.{s}'] = _row(f'requirements.{s}', None, f'Requirements {s.replace("_", " ")}',
                                            'count', 'higher' if s == 'pass' else 'lower', counts[s])


def fmt(v, unit: str = '') -> str:
    """A figure as the tables show it: four significant digits and its unit, '—' when not stated."""
    v = _num(v)
    if v is None:
        return '—'
    s = f'{v:.4g}' if abs(v) < 1e6 else f'{v:.4e}'
    if unit == 'p':                                      # a p-value: "p = 0.0915", not "0.0915 p" (as dv.Result)
        return f'p = {s}'
    return f'{s} {unit}'.strip() if unit not in ('', 'count') else s


def headline_of(doc, study: str = '', metrics: dict | None = None) -> dict:
    """{text, status, word} the result states for itself (its headline or summary); the text names the study's
    headline metric with its value when it has one (as dv.Result.headline does), else the result's own text."""
    out = {'text': '', 'status': '', 'word': ''}
    if not isinstance(doc, dict):
        return out
    for part in (doc.get('headline'), doc.get('summary'), doc):
        if not isinstance(part, dict):
            continue
        for k in ('text', 'explain', 'message'):
            if not out['text'] and isinstance(part.get(k), str):
                out['text'] = part[k]
        for k in ('status_word', 'word', 'verdict_text'):
            if not out['word'] and isinstance(part.get(k), str):
                out['word'] = part[k]
        for k in ('status', 'level'):
            if not out['status'] and part.get(k) in STATUS_LEVELS:
                out['status'] = part[k]
    from .. import metrics as M
    M.load_metrics()
    m = next((m for m in M.METRICS.values() if m.study == study and 'headline' in m.tags), None)
    row = (metrics or {}).get(m.id) if m is not None else None
    if row is not None:
        shown = fmt(row.get('value'), row.get('unit', ''))
        if row.get('bound') and row.get('value') is not None:
            shown = ('≥ ' if row['bound'] == 'at_least' else '≤ ') + shown
        out['text'] = f'{m.title}: {shown}' + (f' ({out["word"]})' if out['word'] else '')
    elif not out['text'] and out['word']:
        out['text'] = out['word']
    return out


def _clean(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k not in ('cached', 'revision')}


def build(study: str, doc: dict | None, project, *, kind: str | None = None, interface: str = 'app',
          status: str | None = None, reason: str = '', started=None, ended=None, tags=(), note: str = '',
          rerun_of: str | None = None, reproduce: dict | None = None, run_id: str | None = None,
          title: str = '') -> tuple[dict, dict | None]:
    """(record, stamped result) for one run. doc: the study's result (stamped already, as batch mode writes it, or
    not: it is stamped here with core/provenance against project); None for a run that failed (reason says why).
    started / ended: datetimes or ISO text (ended defaults to now, started to ended minus the result's seconds)."""
    from .. import jobs
    from .. import provenance as PV
    end = parse_utc(ended) if ended is not None else _dt.datetime.now(_dt.timezone.utc)
    stamped = None
    if doc is not None:
        doc = _clean(doc)
        stamped = doc if isinstance(doc.get(PV.KEY), dict) else PV.stamp(doc, project)
        prov = stamped[PV.KEY]
    else:
        prov = PV.block(project, None, study=study)
    secs = _num((doc or {}).get('seconds'))
    if secs is None and isinstance((doc or {}).get('execution'), dict):     # an experiment's: how it ran
        secs = _num(doc['execution'].get('seconds'))
    if started is not None:
        start = parse_utc(started)
    else:
        start = end - _dt.timedelta(seconds=secs or 0.0)
    if status is None:
        status = 'failed' if doc is None else ('cancelled' if (doc or {}).get('cancelled') else 'finished')
    metrics, samples = metrics_of(study, doc) if doc is not None else ({}, {})
    info = jobs.STUDIES.get(study)
    title = title or (info.title if info else study)
    proj = prov.get('project') or {}
    var = prov.get('variant') or {}
    name = ' · '.join(x for x in (title, proj.get('name') or '', var.get('name') or '') if x)
    rec = {
        'schema': SCHEMA,
        'run_id': run_id or new_run_id(),
        'kind': kind or kind_of(study),
        'study': study,
        'title': title,
        'name': name,
        'interface': interface,
        'started_utc': utc_text(start),
        'ended_utc': utc_text(end),
        'seconds': secs,
        'status': status,
        'reason': (str(reason).splitlines() or [''])[0][:500] if reason else '',
        'project': {k: proj.get(k) for k in ('name', 'file', 'sha256', 'input_sha256') if proj.get(k) is not None},
        'variant': {k: var.get(k) for k in ('name', 'index', 'uid', 'sha256')} if var else None,
        'settings': prov.get('settings') or {},
        'seeds': prov.get('seeds') or {},
        'metrics': metrics,
        'headline': headline_of(doc, study, metrics),
        'result': ({'file': 'result.json', 'schema': stamped.get('schema'), 'sha256': prov.get('result_sha256'),
                    'output': prov.get('output')} if stamped is not None else None),
        'provenance': {'schema': prov.get('schema'), 'dexvantage': prov.get('dexvantage'),
                       'runtime': prov.get('runtime'), 'interface': prov.get('interface'),
                       'command': prov.get('command'), 'inputs': prov.get('inputs') or [], 'plan': prov.get('plan'),
                       'created_utc': prov.get('created_utc')},
        'tags': clean_tags(tags),
        'note': str(note or ''),
        'rerun_of': rerun_of,
        'reproduce': reproduce,
        'logged_utc': utc_now(),
    }
    if samples:
        rec['samples'] = samples
    return rec, stamped
