"""Checks of the assistant's tool layer and the MCP server (assistant/tools.py, assistant/figures.py, mcp/) against
independent references.

- assistant.figures_from_studies: every figure the reading tools return against the study's own result document,
  read here by an extraction written out again (its own keys and unit conversions, not assistant/figures.py), on
  the built-in sample. The tools copy figures, they never compute one, so the tolerance is a rounding: a relative
  difference of 5e-4 (the briefs keep 4 significant digits below 1 and 4 decimals above).
- assistant.mcp_conformance: the server's replies to the request messages the MCP 2025-11-25 specification prints
  (lifecycle: the initialize request; tools: tools/list, tools/call and the unknown-tool protocol error -32602;
  JSON-RPC 2.0's parse error -32700 and method-not-found -32601) and every tool's input schema checked against the
  JSON Schema 2020-12 meta-schema by the jsonschema package (an independent implementation; when it is not
  installed, a smaller structural check). The figure is the number of nonconformities; tolerance 0.
- assistant.edit_undo_identity: edit_project then undo_last_change on the sample must give back the project's
  content hash (core/provenance.py: the SHA-256 of its canonical JSON) bit for bit, and a change to a locked sensor
  must leave it unchanged; the figure is the number of mismatches, tolerance 0.
"""
from __future__ import annotations

import json
import math

from .suite import Outcome, check

_RTOL = 5e-4


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=_RTOL, abs_tol=1e-12)


def _session(ctx):
    from ..assistant.session import FileSession
    return FileSession(project=ctx.project.snapshot())


@check('assistant.figures_from_studies', 'assistant', 'Tool figures against the study results they come from',
       'largest relative difference between a figure a reading tool returns and the same figure read off the '
       'study\'s result document here', 'the study result documents (core/jobs.run_study), read by an extraction '
       'written out again', _RTOL, 'relative', refs=('mcp_2025_11_25',), cost='study')
def _figures(ctx):
    from ..assistant import tools as T
    from ..core import jobs
    s = _session(ctx)
    rows, worst, bad = [], 0.0, []

    def cmp(tool, what, got, ref):
        nonlocal worst
        ok = _close(got, ref)
        d = 0.0 if (got is None and ref is None) else (
            math.inf if (got is None or ref is None) else abs(float(got) - float(ref)) / max(abs(float(ref)), 1e-12))
        worst = max(worst, d if math.isfinite(d) else 1.0)
        rows.append({'tool': tool, 'figure': what, 'tool_value': got, 'study_value': ref, 'ok': ok})
        if not ok:
            bad.append(f'{tool} {what}')

    p = ctx.project
    cov = T.call(s, 'get_coverage', {})
    doc = jobs.run_study('coverage', p.snapshot())
    sm = doc['summary']
    cmp('get_coverage', 'seen by 2+ (%)', cov['figure']['value'], sm['seen_ge2'] * 100)
    cmp('get_coverage', 'blind (%)', cov['figures']['blind_pct'], sm['blind'] * 100)
    cmp('get_coverage', 'best px/mm median', cov['figures']['best_px_per_mm_median'], sm['best_ppmm_median'])
    for c in doc['cameras']:
        row = next(r for r in cov['rows'] if r['sensor_uid'] == c['sensor_uid'])
        cmp('get_coverage', f'{c["name"]} seen (%)', row['seen_pct'], c['seen'] * 100)
        cmp('get_coverage', f'{c["name"]} px/mm median', row['px_per_mm_median'], c['ppmm_median'])
    ts = T.call(s, 'get_sync', {})
    doc = jobs.run_study('time_sync', p.snapshot())
    cmp('get_sync', 'worst skew (s)', ts['figure']['value'], doc['summary']['worst_skew_s'])
    cmp('get_sync', 'label error (m)', ts['figures']['label_error_m'], doc['summary']['label_error_m'])
    bw = T.call(s, 'get_bandwidth', {})
    doc = jobs.run_study('bandwidth', p.snapshot())
    cmp('get_bandwidth', 'busiest load (%)', bw['figure']['value'], doc['headline'].get('busiest_load_pct'))
    cmp('get_bandwidth', 'payload (Mbit/s)', bw['figures']['total_payload_mbps'],
        doc['headline'].get('total_payload_mbps'))
    for x in doc['sensors']:
        row = next(r for r in bw['rows'] if r['sensor_uid'] == x['uid'])
        cmp('get_bandwidth', f'{x["name"]} payload (Mbit/s)', row['payload_mbps'], x.get('payload_mbps'))
    cab = T.call(s, 'get_cable_flex', {})
    if 'error' not in cab:
        doc = jobs.run_study('cables', p.snapshot())
        for c in doc['cables']:
            row = next((r for r in cab['rows'] if r['cable_uid'] == c['uid']), None) or \
                next(r for r in cab['rows'] if r['cable'] == c['name'])     # a suggested route has a new uid per run
            cmp('get_cable_flex', f'{c["name"]} required (m)', row['required_m'], c.get('required_m'))
    return Outcome(worst if not bad else max(worst, 1.0), {'compared': len(rows), 'mismatches': bad, 'rows': rows},
                   note=f'{len(rows)} figures compared on the sample; the tools return the studies\' own figures, '
                        'rounded for reading.')


@check('assistant.mcp_conformance', 'assistant', 'MCP server replies against the 2025-11-25 specification',
       'number of replies or tool definitions that do not conform (the spec\'s own example requests; every input '
       'schema against the JSON Schema 2020-12 meta-schema)', 'MCP 2025-11-25 lifecycle, tools and transports '
       'pages; JSON-RPC 2.0; the jsonschema package', 0.0, 'count', refs=('mcp_2025_11_25', 'jsonrpc2', 'json_schema'))
def _conformance(ctx):
    import re

    from ..mcp.server import PROTOCOL, Router, Server
    problems = []
    srv = Server(Router(mode='file'))
    # the initialize request as the lifecycle page prints it
    init = srv.handle({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {
        'protocolVersion': '2025-11-25', 'capabilities': {'roots': {'listChanged': True}, 'sampling': {},
                                                          'elicitation': {'form': {}, 'url': {}}},
        'clientInfo': {'name': 'ExampleClient', 'title': 'Example Client Display Name', 'version': '1.0.0'}}})
    r = init.get('result') or {}
    if init.get('id') != 1 or init.get('jsonrpc') != '2.0':
        problems.append('initialize: the reply does not echo the id')
    if r.get('protocolVersion') != PROTOCOL:
        problems.append('initialize: a supported version must be answered with the same version')
    if 'tools' not in (r.get('capabilities') or {}):
        problems.append('initialize: a server with tools must declare the tools capability')
    if not isinstance((r.get('serverInfo') or {}).get('name'), str) or not (r.get('serverInfo') or {}).get('version'):
        problems.append('initialize: serverInfo needs name and version')
    if srv.handle({'jsonrpc': '2.0', 'method': 'notifications/initialized'}) is not None:
        problems.append('notifications/initialized was answered (a notification gets no reply)')
    lst = srv.handle({'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list', 'params': {}})
    tools = (lst.get('result') or {}).get('tools') or []
    if not tools:
        problems.append('tools/list: no tools')
    try:
        import jsonschema
        meta = jsonschema.Draft202012Validator
    except ImportError:
        meta = None
    names = set()
    for t in tools:
        n = t.get('name', '')
        if not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', n) or n in names:
            problems.append(f'tool name {n!r}: 1-128 of A-Z a-z 0-9 _ - ., unique')
        names.add(n)
        sch = t.get('inputSchema')
        if not isinstance(sch, dict) or sch.get('type') != 'object':
            problems.append(f'{n}: inputSchema must be a JSON Schema object of type object')
            continue
        if meta is not None:
            try:
                meta.check_schema(sch)
            except jsonschema.SchemaError as e:
                problems.append(f'{n}: inputSchema is not valid JSON Schema 2020-12 ({e.message})')
        if not isinstance(t.get('description'), str) or not t['description']:
            problems.append(f'{n}: no description')
    # tools/call of an unknown tool: a protocol error, -32602 (tools page, Error Handling)
    unk = srv.handle({'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call',
                      'params': {'name': 'invalid_tool_name', 'arguments': {}}})
    if (unk.get('error') or {}).get('code') != -32602:
        problems.append('tools/call of an unknown tool must be error -32602')
    # a tool execution error: a result with isError and text content (the input does not match the tool)
    bad = srv.handle({'jsonrpc': '2.0', 'id': 4, 'method': 'tools/call',
                      'params': {'name': 'get_sensor', 'arguments': {}}})
    res = bad.get('result') or {}
    if res.get('isError') is not True or (res.get('content') or [{}])[0].get('type') != 'text':
        problems.append('a tool execution error must be a result with isError true and text content')
    else:
        try:
            json.loads(res['content'][0]['text'])
        except ValueError:
            problems.append('the text content is not the JSON it says')
    if (srv.handle({'jsonrpc': '2.0', 'id': 5, 'method': 'no/such'}).get('error') or {}).get('code') != -32601:
        problems.append('an unknown method must be error -32601')
    out_lines = []

    class _Out:
        def write(self, s):
            out_lines.append(s)

        def flush(self):
            pass
    Server(Router(mode='file'), _Out()).serve(['{not json\n', json.dumps({'jsonrpc': '2.0', 'id': 6,
                                                                           'method': 'ping'}) + '\n'])
    replies = [json.loads(x) for x in ''.join(out_lines).splitlines()]
    if not replies or (replies[0].get('error') or {}).get('code') != -32700:
        problems.append('a line that is not JSON must be answered with error -32700')
    if any('\n' in x.rstrip('\n') for x in out_lines):
        problems.append('a stdio message must not contain an embedded newline')
    if len(replies) < 2 or replies[1].get('result') != {}:
        problems.append('ping must be answered with an empty result')
    return Outcome(float(len(problems)), {'problems': problems, 'tools': len(tools),
                                          'schema_check': 'jsonschema Draft202012Validator' if meta else 'structural'},
                   note=f'{len(tools)} tool definitions and 8 request kinds checked against the specification.')


@check('assistant.edit_undo_identity', 'assistant', 'An edit and its undo give back the same project',
       'mismatches of the project content hash after edit_project then undo_last_change, and after a refused change '
       'to a locked sensor', 'the project content hash (SHA-256 of its canonical JSON, core/provenance.py)', 0.0,
       'count', refs=('mcp_2025_11_25',))
def _undo_identity(ctx):
    from ..assistant import tools as T
    from ..assistant.session import FileSession
    from ..core import provenance as PV
    p = ctx.project.snapshot()
    s = FileSession(project=p)
    rows, bad = [], 0
    for x in p.variant.sensors[:3]:
        h0 = PV.project_digest(s.project())['sha256']
        e = T.call(s, 'edit_project', {'changes': {f'sensor:{x.uid}.dx': 0.013, f'sensor:{x.uid}.dyaw': 4.0}})
        h1 = PV.project_digest(s.project())['sha256']
        T.call(s, 'undo_last_change', {})
        h2 = PV.project_digest(s.project())['sha256']
        ok = 'error' not in e and h1 != h0 and h2 == h0
        bad += not ok
        rows.append({'sensor': x.name, 'changed': h1 != h0, 'restored': h2 == h0})
    lp = ctx.project.snapshot()
    if lp.variant.sensors:
        lp.variant.sensors[0].locked = True
        ls = FileSession(project=lp)
        h0 = PV.project_digest(ls.project())['sha256']
        r = T.call(ls, 'edit_project', {'changes': {f'sensor:{lp.variant.sensors[0].uid}.x': 0.3}})
        ok = 'locked' in r.get('error', '') and PV.project_digest(ls.project())['sha256'] == h0
        bad += not ok
        rows.append({'sensor': lp.variant.sensors[0].name + ' (locked)', 'refused': 'error' in r, 'unchanged': ok})
    return Outcome(float(bad), {'rows': rows}, note='Each edit moves and turns one sensor of the sample; its undo must '
                                                    'give back the content hash bit for bit.')
