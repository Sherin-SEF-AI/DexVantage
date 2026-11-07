"""Reports (spec §9.2): an HTML report and a PDF report of every study result present, with figures, the requirements
scorecard when the project has requirements, and a provenance block.

    @exporter('report', 'Report (HTML and PDF)', ...)
    options: results ({study id: result doc} or [doc]: the results present, e.g. the app's computed results),
             results_dir (a folder of batch result JSON files, each with its 'study'), studies ('all' or [study id]:
             studies to run now for those not given; default 'all' when neither results nor results_dir is given,
             else none; 'all' leaves out META_STUDIES, the ones that run other studies many times: name them, or
             give their results), pdf (default True), title (default '<project> report')

Files (under out_dir/report/):

    report.html     a page that needs nothing else (inline SVG figures, light or dark), sections in study order
    report.pdf      the same content on A4 pages (core/exporters/_pdf.py: pure Python, no optional package)
    report.json     what the report shows, as data (dexvantage.report/1; stamped like every export)

What each study section holds, read off the study's own result (nothing is computed here): the headline (the result's
'headline' or 'summary' block, or its top-level 'status' and 'word'), shown as the STATUS word with its colour and
the advice the study gives; the study's named figures (core/metric_defs.py, read with core.metrics.read: a figure the
result does not state reads "not stated"); one figure: a bar chart of the first per-camera (or per-policy, per-cable,
per-device) list in the result that carries a number with a unit suffix (_pct, _mm, _m, _px, _ms, _s, _deg, _mbps),
labelled by name; the result's notes; and when the result came with its own provenance block (a batch file), its
result hash. Studies that refused are listed with their sentence. The scorecard is core/requirements.check on the
same results (a shared result cache holds them, so no study runs twice). The provenance block is
core.provenance.block of the report document: DexVantage version and code hash, runtime, the project's content hash,
the variant and the time.

Known limits: the figure picks the first matching list and number by a naming rule, so it shows one figure a study
has, not necessarily its most important one (the page of each study has the full set); results given with the export
are reported as given (a result computed on another state of the project says so only through its own provenance
block, when it has one); the PDF uses the standard fonts (characters outside cp1252 are spelled out, e.g. σ as
sigma) and approximate line breaking.
"""
from __future__ import annotations

import html as _html
import json
import math
import os

from ..jobs import StudyError
from . import _common as C
from . import exporter

FOLDER = 'report'
SCHEMA = 'dexvantage.report/1'
HTML_FILE, PDF_FILE, JSON_FILE = f'{FOLDER}/report.html', f'{FOLDER}/report.pdf', f'{FOLDER}/report.json'
LEVELS = ('good', 'warning', 'serious', 'critical')
LEVEL_ALIASES = {'ok': 'good', 'pass': 'good', 'fine': 'good', 'info': 'warning', 'fail': 'critical',
                 'error': 'critical', 'bad': 'critical'}
STATUS_HEX = {'good': '#0ca30c', 'warning': '#fab219', 'serious': '#ec835a', 'critical': '#d03b3b'}
UNIT_SUFFIX = (('_px_s', 'px/s'), ('_mm_s', 'mm/s'), ('_m_s', 'm/s'), ('_deg_s', '°/s'), ('_pct', '%'),
               ('_mm', 'mm'), ('_px', 'px'), ('_ms', 'ms'), ('_mbps', 'Mbit/s'), ('_deg', '°'), ('_m', 'm'),
               ('_s', 's'))
SINGULAR = {'cameras': 'camera', 'sensors': 'sensor', 'policies': 'policy', 'devices': 'device', 'cables': 'cable',
            'streams': 'stream', 'controllers': 'controller', 'fingers': 'finger', 'hosts': 'host'}
LIST_KEYS = ('cameras', 'sensors', 'policies', 'devices', 'cables', 'streams', 'controllers', 'fingers', 'hosts')
NAME_KEYS = ('name', 'sensor', 'title', 'camera', 'key')
MAX_BARS = 24
# studies that run other studies many times (Monte Carlo, designed experiments, the optimiser, the design agent, the
# variant comparison, the scorecard itself): reported when their results are given, never run by default ('all')
META_STUDIES = ('tolerance', 'experiments', 'optimise', 'design_agent', 'variants_compare', 'requirements')


# ====================================================================== the results
def _load_dir(path: str) -> dict:
    out = {}
    try:
        names = sorted(os.listdir(path))
    except OSError as e:
        raise StudyError(f'The results folder {os.path.basename(path) or path} cannot be read: '
                         f'{e.strerror or e}.') from None
    for n in names:
        if not n.lower().endswith('.json'):
            continue
        try:
            with open(os.path.join(path, n), encoding='utf-8') as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and isinstance(doc.get('study'), str):
            out[doc['study']] = doc
    return out


def collect(project, options: dict, ctx=None, cache=None) -> tuple[dict, dict]:
    """({study id: result doc} present, {study id: refusal sentence}) in study order."""
    from .. import jobs
    jobs.load_studies()
    given = options.get('results')
    res: dict = {}
    if isinstance(given, dict):
        res.update({str(k): v for k, v in given.items() if isinstance(v, dict)})
    elif isinstance(given, (list, tuple)):
        res.update({d['study']: d for d in given if isinstance(d, dict) and isinstance(d.get('study'), str)})
    elif given is not None:
        raise StudyError('The report\'s results must map each study to its result.')
    if options.get('results_dir'):
        for k, v in _load_dir(str(options['results_dir'])).items():
            res.setdefault(k, v)
    want = options.get('studies', 'all' if given is None and not options.get('results_dir') else ())
    if want == 'all':
        want = [s for s in jobs.STUDIES if s not in META_STUDIES]
    elif isinstance(want, str):
        want = [w.strip() for w in want.split(',') if w.strip()]
    unknown = [w for w in want if w not in jobs.STUDIES]
    if unknown:
        raise StudyError(f'DexVantage has no study {", ".join(unknown)}.')
    refused = {}
    todo = [w for w in want if w not in res]
    for k, sid in enumerate(todo):
        if ctx is not None:
            if ctx.cancelled():
                break
            ctx.report(k / max(1, len(todo)), jobs.STUDIES[sid].title)
        try:
            res[sid] = jobs.run_study(sid, project.snapshot(), cache=cache)
        except StudyError as e:
            refused[sid] = str(e)
    order = list(jobs.STUDIES)
    res = dict(sorted(res.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else len(order)))
    return res, refused


def _level(v) -> str | None:
    if not isinstance(v, str):
        return None
    v = v.strip().lower()
    return v if v in LEVELS else LEVEL_ALIASES.get(v)


def headline(doc: dict) -> dict:
    """{'level', 'word', 'text', 'figures': [(label, value text)]} read off a result (module docstring)."""
    out = {'level': None, 'word': None, 'text': None, 'figures': []}
    blocks = [doc.get('headline'), doc.get('summary'), doc]
    for b in blocks:
        if not isinstance(b, dict):
            continue
        out['level'] = out['level'] or _level(b.get('status')) or _level(b.get('level'))
        for k in ('word', 'status_word'):
            if out['word'] is None and isinstance(b.get(k), str):
                out['word'] = b[k]
        for k in ('advice', 'text', 'message'):
            if out['text'] is None and isinstance(b.get(k), str) and b[k].strip():
                out['text'] = b[k]
        if b is not doc and not out['figures']:
            for k, v in b.items():
                if isinstance(v, bool) or not isinstance(v, (int, float)) or k in ('status',):
                    continue
                if isinstance(v, float) and not math.isfinite(v):
                    continue
                out['figures'].append((k.replace('_', ' '), _fmt(v, _unit_of(k))))
                if len(out['figures']) >= 8:
                    break
    if out['word'] is None and out['level']:
        out['word'] = out['level']
    return out


def _unit_of(key: str) -> str:
    for suf, unit in UNIT_SUFFIX:
        if key.endswith(suf):
            return unit
    return ''


def _fmt(v, unit: str = '') -> str:
    if v is None:
        return '—'
    if isinstance(v, bool):
        return 'yes' if v else 'no'
    if isinstance(v, int):
        s = f'{v:,}'
    elif isinstance(v, float):
        s = f'{v:.4g}' if abs(v) < 1e5 else f'{v:,.0f}'
    else:
        s = str(v)
    return f'{s} {unit}'.strip() if unit != '%' else f'{s} %'


def metric_rows(sid: str, doc: dict) -> list:
    """[(title, value text, better, status)] of the study's named figures (core/metric_defs.py)."""
    from .. import metrics as M
    M.load_metrics()
    rows = []
    for m in M.METRICS.values():
        if m.study != sid:
            continue
        r = M.read(m, doc)
        val = _fmt(r['value'], m.unit)
        if r.get('bound') == 'at_least':
            val = '≥ ' + val
        elif r.get('bound') == 'at_most':
            val = '≤ ' + val
        rows.append({'id': m.id, 'title': m.title, 'value': r['value'], 'text': val if r['value'] is not None else
                     'not stated', 'unit': m.unit, 'better': m.better, 'reason': r.get('reason') or ''})
    return rows


def figure(doc: dict) -> dict | None:
    """{'title', 'labels', 'values', 'unit', 'key', 'list'} of the first per-item list with a number (module
    docstring), or None."""
    for lk in LIST_KEYS:
        items = doc.get(lk)
        if not isinstance(items, list) or not items or not all(isinstance(x, dict) for x in items):
            continue
        keys = [k for k, v in items[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)
                and _unit_of(k)]
        for k in keys:
            vals = [x.get(k) for x in items]
            if not any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
                continue
            labels = []
            for x in items:
                lab = next((str(x[n]) for n in NAME_KEYS if isinstance(x.get(n), str) and x[n]), '?')
                if isinstance(x.get('imager'), str) and x['imager'] and x['imager'] not in lab:
                    lab = f'{lab} · {x["imager"]}'
                labels.append(lab)
            vals = [float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None
                    for v in vals]
            name = k
            for suf, _u in UNIT_SUFFIX:
                if name.endswith(suf):
                    name = name[:-len(suf)]
                    break
            return {'title': f'{name.replace("_", " ")} per {SINGULAR[lk]}', 'labels': labels[:MAX_BARS],
                    'values': vals[:MAX_BARS], 'unit': _unit_of(k), 'key': k, 'list': lk}
    return None


def section(sid: str, doc: dict) -> dict:
    from .. import jobs
    info = jobs.STUDIES.get(sid)
    h = headline(doc)
    notes = [str(n) for n in (doc.get('notes') or []) if isinstance(n, str)][:6] \
        if isinstance(doc.get('notes'), list) else []
    prov = doc.get('provenance') if isinstance(doc.get('provenance'), dict) else None
    return {'id': sid, 'title': info.title if info else sid, 'level': h['level'], 'word': h['word'],
            'text': h['text'], 'figures': h['figures'], 'metrics': metric_rows(sid, doc), 'figure': figure(doc),
            'notes': notes, 'seconds': doc.get('seconds'), 'cancelled': bool(doc.get('cancelled')),
            'result_sha256': prov.get('result_sha256') if prov else None,
            'result_created': prov.get('created_utc') if prov else None}


def scorecard(project, results: dict, cache) -> dict | None:
    """core/requirements.check on the same results, or None when the project has no requirements."""
    if not project.requirements:
        return None
    from .. import requirements as RQ
    try:
        return RQ.check(project, cache=cache)
    except StudyError as e:
        return {'verdict': 'error', 'headline': {'text': str(e), 'word': 'not evaluated', 'level': 'warning',
                                                 'value': '—'}, 'items': []}


def build(project, options: dict | None = None, ctx=None) -> dict:
    """The report document (JSON-able; what report.html and report.pdf show)."""
    from .. import jobs
    from .. import provenance as PV
    o = dict(options or {})
    cache = jobs.ResultCache(64)
    results, refused = collect(project, o, ctx, cache)
    for sid, doc in results.items():                         # share them with the scorecard
        if sid in jobs.STUDIES and not doc.get('cancelled'):
            try:
                cache.put(cache.key(sid, project), {k: v for k, v in doc.items() if k != 'provenance'})
            except Exception:                                 # noqa: BLE001 — the scorecard runs the study instead
                pass
    card = scorecard(project, results, cache)
    v = project.variant
    title = str(o.get('title') or f'{project.name} report')
    doc = {'schema': SCHEMA, 'title': title, 'project': project.name, 'variant': v.name,
           'counts': {'studies': len(results), 'refused': len(refused),
                      'sensors': sum(1 for s in v.sensors if s.enabled), 'robots': len(project.robots),
                      'requirements': len(project.requirements)},
           'sections': [section(sid, d) for sid, d in results.items()],
           'refused': [{'id': k, 'title': jobs.STUDIES[k].title, 'reason': r} for k, r in refused.items()],
           'scorecard': card}
    doc['provenance_block'] = PV.block(project, doc, study='report',
                                       output={'option': 'export', 'name': HTML_FILE, 'in_folder': True})
    return doc


# ====================================================================== HTML
def _e(s) -> str:
    return _html.escape(str(s if s is not None else '—'))


def _chip(level, word) -> str:
    lv = level if level in LEVELS else None
    color = STATUS_HEX.get(lv, '#898781')
    return (f'<span class="chip" style="color:{color};border-color:{color}" data-level="{_e(lv or "none")}">'
            f'<i style="background:{color}"></i>{_e(word or "no status")}</span>')


def _figure_svg(fig: dict) -> str:
    from ...validation.report import svg_figure
    vals = fig['values']
    spec = {'kind': 'bars', 'title': fig['title'], 'categories': fig['labels'],
            'series': [{'label': fig['key'], 'y': [v if v is not None else 0.0 for v in vals]}],
            'ylabel': fig['unit'], 'xlabel': ''}
    if any(v is not None and v < 0 for v in vals):
        spec['series'][0]['y'] = [abs(v) if v is not None else 0.0 for v in vals]
        spec['title'] += ' (magnitude)'
    return svg_figure(spec)


def _prov_rows(pb: dict) -> list:
    dv = pb.get('dexvantage') or {}
    pj = pb.get('project') or {}
    var = pb.get('variant') or {}
    rt = pb.get('runtime') or {}
    return [('DexVantage', f'{dv.get("version") or "—"}' + (f' · commit {str(dv["commit"])[:12]}' if dv.get('commit')
                                                              else '') + (' (modified)' if dv.get('dirty') else '')),
            ('Code SHA-256', dv.get('code_sha256') or 'not available (a bundle ships no sources)'),
            ('Project', f'{pj.get("name") or "—"}' + (f' · file {pj["file"]}' if pj.get('file') else '')),
            ('Project content SHA-256', pj.get('sha256') or '—'),
            ('Variant', f'{var.get("name") or "—"} (#{var.get("index") or "—"}) · SHA-256 {var.get("sha256") or "—"}'),
            ('Report SHA-256', pb.get('result_sha256') or '—'),
            ('Created (UTC)', pb.get('created_utc') or '—'),
            ('Runtime', ', '.join(f'{k} {v}' for k, v in rt.items() if isinstance(v, (str, int, float)) and v)[:400]),
            ('Interface', pb.get('interface') or '—')]


def to_html(doc: dict) -> str:
    from ...validation.report import PAGE_CSS
    css = PAGE_CSS + ('.chip{display:inline-flex;align-items:center;gap:6px;font-weight:600;font-size:12.5px;'
                      'border:1px solid;border-radius:999px;padding:2px 10px}.chip i{width:8px;height:8px;'
                      'border-radius:50%;display:inline-block}.num{font-family:"IBM Plex Mono",ui-monospace,monospace}'
                      '.toc a{margin-right:12px}')
    P = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
         f'content="width=device-width,initial-scale=1"><title>{_e(doc["title"])}</title><style>{css}</style>'
         f'</head><body><main>', f'<h1>{_e(doc["title"])}</h1>',
         f'<p class="meta">Project {_e(doc["project"])} · rig {_e(doc["variant"])} · {doc["counts"]["studies"]} '
         f'stud{"y" if doc["counts"]["studies"] == 1 else "ies"} reported · {doc["counts"]["sensors"]} sensors in use '
         f'· {doc["counts"]["robots"]} robot{"s" if doc["counts"]["robots"] != 1 else ""}</p>']
    toc = ''.join(f'<a href="#study-{_e(s["id"])}">{_e(s["title"])}</a>' for s in doc['sections'])
    P.append(f'<nav class="toc meta">{toc}</nav>')
    tiles = ''.join(f'<div class="tile"><b>{_e(s["word"] or "—")}</b><span>{_e(s["title"])}</span></div>'
                    for s in doc['sections'] if s['word'])
    if tiles:
        P.append(f'<div class="tiles">{tiles}</div>')
    card = doc.get('scorecard')
    if card:
        h = card.get('headline') or {}
        P.append(f'<h2 id="scorecard">Requirements scorecard</h2><div class="card">'
                 f'{_chip(h.get("level"), h.get("word"))} <b class="num">{_e(h.get("value"))}</b>'
                 f'<p>{_e(h.get("text"))}</p><div class="wrap"><table><tr><th>Requirement</th><th>Value</th>'
                 f'<th>Threshold</th><th>Status</th><th>Why</th></tr>')
        for it in card.get('items') or []:
            thr = f'{it.get("op_text", "")} {_fmt(it.get("threshold"), it.get("unit") or "")}' + \
                (' (default)' if it.get('is_default') else '')
            val = _fmt(it.get('value'), it.get('unit') or '')
            P.append(f'<tr><td>{_e(it.get("name"))}</td><td class="num">{_e(val)}</td>'
                     f'<td class="num">{_e(thr)}</td><td>{_chip(it.get("level"), it.get("status_text"))}</td>'
                     f'<td>{_e(it.get("reason") or "")}</td></tr>')
        P.append('</table></div></div>')
    for s in doc['sections']:
        P.append(f'<h2 id="study-{_e(s["id"])}" data-study="{_e(s["id"])}">{_e(s["title"])}</h2><div class="card">')
        P.append(_chip(s['level'], s['word']))
        if s['cancelled']:
            P.append(' <span class="meta">(stopped before it finished: a partial result)</span>')
        if s['text']:
            P.append(f'<p>{_e(s["text"])}</p>')
        if s['metrics']:
            P.append('<div class="wrap"><table><tr><th>Figure</th><th>Value</th><th>Better</th></tr>')
            for m in s['metrics']:
                tip = f' title="{_e(m["reason"])}"' if m['reason'] else ''
                P.append(f'<tr><td>{_e(m["title"])}</td><td class="num"{tip}>{_e(m["text"])}</td>'
                         f'<td>{_e(m["better"])}</td></tr>')
            P.append('</table></div>')
        elif s['figures']:
            P.append('<dl class="kv">' + ''.join(f'<dt>{_e(k)}</dt><dd class="num">{_e(v)}</dd>'
                                                 for k, v in s['figures']) + '</dl>')
        if s['figure']:
            P.append(_figure_svg(s['figure']))
        if s['notes']:
            P.append('<details><summary>Notes</summary><ul>' + ''.join(f'<li>{_e(n)}</li>' for n in s['notes'])
                     + '</ul></details>')
        meta = []
        if s['seconds'] is not None:
            meta.append(f'computed in {float(s["seconds"]):.3g} s')
        if s['result_sha256']:
            meta.append(f'result SHA-256 {s["result_sha256"][:16]}… ({_e(s["result_created"])})')
        if meta:
            P.append(f'<p class="meta">{" · ".join(meta)}</p>')
        P.append('</div>')
    if doc['refused']:
        P.append('<h2 id="not-run">Not run</h2><div class="card"><table><tr><th>Study</th><th>Why</th></tr>' +
                 ''.join(f'<tr><td>{_e(r["title"])}</td><td>{_e(r["reason"])}</td></tr>' for r in doc['refused'])
                 + '</table></div>')
    pb = doc['provenance_block']
    P.append('<h2 id="provenance">Provenance</h2><div class="card"><dl class="kv">' +
             ''.join(f'<dt>{_e(k)}</dt><dd><code>{_e(v)}</code></dd>' for k, v in _prov_rows(pb)) +
             f'</dl><p class="meta">{_e(pb.get("result_hash"))}. {_e(pb.get("content_hash"))}.</p></div>')
    P.append('</main></body></html>')
    return '\n'.join(P) + '\n'


# ====================================================================== PDF
def to_pdf(doc: dict) -> bytes:
    from ._pdf import PdfDoc
    from ...validation.report import SERIES
    pdf = PdfDoc(doc['title'])
    pdf.heading(doc['title'], 18)
    c = doc['counts']
    pdf.para(f'Project {doc["project"]} · rig {doc["variant"]} · {c["studies"]} studies reported · {c["sensors"]} '
             f'sensors in use · {c["robots"]} robots')
    card = doc.get('scorecard')
    if card:
        h = card.get('headline') or {}
        pdf.heading('Requirements scorecard', 13)
        pdf.status(h.get('word') or '—', h.get('level') or '', str(h.get('value') or ''))
        if h.get('text'):
            pdf.para(h['text'])
        rows = [['Requirement', 'Value', 'Threshold', 'Status']]
        for it in card.get('items') or []:
            rows.append([it.get('name'), _fmt(it.get('value'), it.get('unit') or ''),
                         f'{it.get("op_text", "")} {_fmt(it.get("threshold"), it.get("unit") or "")}'
                         + (' (default)' if it.get('is_default') else ''), it.get('status_text')])
        pdf.table(rows, [0.4, 0.2, 0.25, 0.15])
    for s in doc['sections']:
        pdf.heading(s['title'], 13)
        pdf.status(s['word'] or 'no status', s['level'] or '')
        if s['text']:
            pdf.para(s['text'])
        if s['metrics']:
            pdf.table([['Figure', 'Value', 'Better']] + [[m['title'], m['text'], m['better']] for m in s['metrics']],
                      [0.55, 0.3, 0.15])
        elif s['figures']:
            pdf.table([['Figure', 'Value']] + [list(x) for x in s['figures']], [0.6, 0.4])
        if s['figure']:
            f = s['figure']
            pdf.bars(f'{f["title"]} ({f["unit"]})', f['labels'], f['values'], f['unit'], colors=[SERIES[0][0]])
        for n in s['notes'][:3]:
            pdf.para('Note: ' + n, 8.5, '#898781')
        if s['result_sha256']:
            pdf.para(f'Result SHA-256 {s["result_sha256"]}', 7.5, '#898781', 'F3')
    if doc['refused']:
        pdf.heading('Not run', 13)
        pdf.table([['Study', 'Why']] + [[r['title'], r['reason']] for r in doc['refused']], [0.3, 0.7])
    pdf.heading('Provenance', 13)
    for k, v in _prov_rows(doc['provenance_block']):
        pdf.para(f'{k}: {v}', 8, '#52514e', 'F3' if 'SHA' in k else 'F1')
    return pdf.tobytes()


# ====================================================================== the exporter
@exporter('report', 'Report (HTML and PDF)', needs_robot=False,
          description='Every study result present with its status, figures and a chart, the requirements scorecard, '
                      'and a provenance block, as a standalone HTML page and a PDF.')
def export_report(project, out_dir: str, options: dict, ctx) -> dict:
    o = dict(options or {})
    doc = build(project, o, ctx)
    if not doc['sections'] and not doc['scorecard']:
        why = '; '.join(f'{r["title"]}: {r["reason"]}' for r in doc['refused'][:3])
        raise StudyError('There is no study result to report' + (f' ({why})' if why else '') +
                         ': run a study first, or pass its results.')
    files = [C.write_text(out_dir, HTML_FILE, to_html(doc))]
    notes = [f'{len(doc["sections"])} study result(s) reported' + (f', {len(doc["refused"])} not run (listed)'
                                                                   if doc['refused'] else '') + '.']
    if o.get('pdf', True):
        files.append(C.write_bytes(out_dir, PDF_FILE, to_pdf(doc)))
        notes.append('The PDF uses the standard PDF fonts: characters they lack are spelled out (σ as sigma).')
    files.append(C.write_json(out_dir, JSON_FILE, {k: v for k, v in doc.items() if k != 'provenance_block'}))
    return {'files': files, 'notes': notes}
