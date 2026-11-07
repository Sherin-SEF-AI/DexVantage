"""Claude over MCP (dexvantage.mcp): the stdio protocol (MCP 2025-11-25), a scripted client over pipes on a temp copy
of a sample project, every tool's input schema, undoable edits, and the open window through its bridge."""
import json
import os
import re
import subprocess
import sys
import threading
import time

import pytest

from dexvantage.core.robot.samples import sample_project
from dexvantage.mcp.server import PROTOCOL, Router, Server

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WRIST = 'Wrist camera (sample placement)'


@pytest.fixture(scope='module')
def so100():
    return sample_project('so100')


def _rpc(srv, method, params=None, mid=1):
    msg = {'jsonrpc': '2.0', 'id': mid, 'method': method}
    if params is not None:
        msg['params'] = params
    return srv.handle(msg)


def _tool(srv, name, **args):
    r = _rpc(srv, 'tools/call', {'name': name, 'arguments': args})['result']
    doc = json.loads(r['content'][0]['text'])
    assert r['structuredContent'] == json.loads(json.dumps(doc))
    return doc, r['isError']


# ======================================================================= protocol
def test_initialize_negotiates_the_version_and_gives_instructions():
    srv = Server(Router(mode='file'))
    r = _rpc(srv, 'initialize', {'protocolVersion': '2025-06-18', 'capabilities': {},
                                 'clientInfo': {'name': 't', 'version': '0'}})['result']
    assert r['protocolVersion'] == '2025-06-18' and r['capabilities'] == {'tools': {'listChanged': False}}
    assert r['serverInfo']['name'] == 'dexvantage' and 'REP-103' in r['instructions']
    assert _rpc(srv, 'initialize', {'protocolVersion': '1999-01-01'})['result']['protocolVersion'] == PROTOCOL
    assert PROTOCOL == '2025-11-25'
    assert srv.handle({'jsonrpc': '2.0', 'method': 'notifications/initialized'}) is None
    assert _rpc(srv, 'ping')['result'] == {}
    assert _rpc(srv, 'nope')['error']['code'] == -32601
    assert _rpc(srv, 'tools/call', {'name': 'nope'})['error']['code'] == -32602          # unknown tool: protocol error
    assert srv.handle({'id': 3, 'method': 'ping'})['error']['code'] == -32600              # not JSON-RPC 2.0
    assert _rpc(srv, 'resources/list')['result'] == {'resources': []}


def test_tools_list_names_every_tool_with_a_valid_input_schema():
    tools = _rpc(Server(Router(mode='file')), 'tools/list')['result']['tools']
    names = [t['name'] for t in tools]
    assert len(names) == len(set(names))
    from dexvantage.assistant import tools as T
    assert set(T.TOOLS) <= set(names) and {'dexvantage_status', 'open_project', 'run_batch'} <= set(names)
    jsonschema = pytest.importorskip('jsonschema')
    for t in tools:
        assert re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', t['name'])               # the 2025-11-25 tool-name rule
        assert t['description'] and t['title'] and isinstance(t['annotations']['readOnlyHint'], bool)
        sch = t['inputSchema']
        jsonschema.Draft202012Validator.check_schema(sch)                       # the default dialect (2020-12)
        assert sch['type'] == 'object'
    by = {t['name']: t for t in tools}
    assert by['edit_project']['annotations']['readOnlyHint'] is False
    assert by['get_coverage']['annotations']['readOnlyHint'] is True


def test_schema_examples_validate_with_jsonschema_too():
    """The server's own check (assistant/tools._check) and jsonschema agree on good and bad inputs."""
    jsonschema = pytest.importorskip('jsonschema')
    from dexvantage.assistant import tools as T
    cases = [('edit_project', {'changes': {'sensor:a.x': 1.0}}, True),
             ('edit_project', {'ops': [{'op': 'add_variant', 'name': 'B'}]}, True),
             ('edit_project', {'ops': [{'op': 'fly'}]}, False),
             ('edit_project', {'bogus': 1}, False),
             ('get_sensor', {}, False), ('get_sensor', {'sensor': 'x'}, True),
             ('run_study', {'study': 'coverage', 'full': True}, True), ('run_study', {'full': 'yes'}, False),
             ('propose_plan', {'title': 't', 'steps': []}, False),
             ('propose_plan', {'title': 't', 'steps': [{'kind': 'study', 'study': 'coverage'}]}, True),
             ('show_in_app', {'action': 'dance'}, False), ('undo_last_change', {}, True)]
    for name, args, good in cases:
        ours = not T.validate_args(name, args)
        theirs = jsonschema.Draft202012Validator(T.TOOLS[name].input_schema).is_valid(args)
        assert ours == theirs == good, (name, args)


# ======================================================================= the file session in process
def test_tools_on_a_file_through_the_server(tmp_path, so100):
    path = str(tmp_path / 'so100.dexv')
    so100.snapshot().save(path)
    srv = Server(Router(project=path, mode='file'))
    st, err = _tool(srv, 'dexvantage_status')
    assert not err and st['mode'] == 'file' and st['project'] == so100.name
    doc, err = _tool(srv, 'edit_project', changes={f'sensor:{WRIST}.dz': 0.01})
    assert not err and doc['undo_step'].startswith('Claude ·')
    doc, err = _tool(srv, 'get_history')
    assert doc['steps'][-1]['text'].startswith('Claude ·')
    doc, err = _tool(srv, 'undo_last_change')
    assert doc['result'].startswith('Undid')
    doc, err = _tool(srv, 'redo_change')
    assert doc['result'].startswith('Redid')
    doc, err = _tool(srv, 'get_sensor', sensor='nobody')
    assert err and 'no sensor' in doc['error']
    doc, err = _tool(srv, 'open_project', sample='ur5e')
    assert not err and 'opened' in doc
    assert _tool(srv, 'dexvantage_status')[0]['project'] != so100.name
    doc, err = _tool(srv, 'save_project', path=str(tmp_path / 'ur5e-copy'))
    assert doc['result'] == 'Saved ur5e-copy.dexv.' and os.path.isfile(tmp_path / 'ur5e-copy.dexv')


def test_run_batch_checks_its_options():
    from dexvantage.mcp import batchrun as B
    assert 'not available' in B.check(['--sample', 'so100'])
    assert 'not a batch option' in B.check(['--fly', 'x'])
    assert B.check(['--coverage', 'out.json']) == ''
    assert '--coverage' in B.help_doc()['options']


def test_run_batch_runs_on_a_copy(tmp_path, so100):
    path = str(tmp_path / 'so100.dexv')
    so100.snapshot().save(path)
    before = os.path.getmtime(path)
    srv = Server(Router(project=path, mode='file'))
    doc, err = _tool(srv, 'run_batch', args=['--bandwidth', 'bw.json'], timeout_s=300)
    assert not err and doc['exit_code'] == 0, doc
    assert doc['json']['bw.json']['study'] == 'bandwidth' and os.path.getmtime(path) == before
    assert os.path.dirname(doc['folder']) == str(tmp_path)


# ======================================================================= a scripted client over pipes
class Client:
    """A minimal MCP client: the server as a subprocess, one JSON-RPC message per line each way."""

    def __init__(self, args, env):
        self.p = subprocess.Popen([sys.executable, '-m', 'dexvantage.mcp', *args], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8',
                                  cwd=ROOT, env=env)
        self.n = 0
        self.lines = []

    def send(self, method, params=None, notify=False):
        msg = {'jsonrpc': '2.0', 'method': method}
        if params is not None:
            msg['params'] = params
        if not notify:
            self.n += 1
            msg['id'] = self.n
        self.p.stdin.write(json.dumps(msg) + '\n')
        self.p.stdin.flush()
        if notify:
            return None
        line = self.p.stdout.readline()
        assert line, self.p.stderr.read()
        self.lines.append(line)
        doc = json.loads(line)
        assert doc['id'] == self.n and doc['jsonrpc'] == '2.0'
        return doc

    def tool(self, name, **args):
        r = self.send('tools/call', {'name': name, 'arguments': args})['result']
        return json.loads(r['content'][0]['text']), r['isError']

    def close(self):
        self.p.stdin.close()
        rest = self.p.stdout.read()
        self.p.wait(60)
        return rest


def test_scripted_client_over_pipes_on_a_temp_copy(tmp_path, so100):
    path = str(tmp_path / 'so100.dexv')
    so100.snapshot().save(path)
    env = dict(os.environ, XDG_CONFIG_HOME=str(tmp_path / 'cfg'), XDG_DATA_HOME=str(tmp_path / 'data'),
               XDG_CACHE_HOME=str(tmp_path / 'cache'), QT_QPA_PLATFORM='offscreen')
    c = Client(['--project', path], env)
    try:
        init = c.send('initialize', {'protocolVersion': PROTOCOL, 'capabilities': {},
                                     'clientInfo': {'name': 'test', 'version': '1'}})
        assert init['result']['protocolVersion'] == PROTOCOL
        c.send('notifications/initialized', notify=True)
        tools = c.send('tools/list')['result']['tools']
        assert {'get_overview', 'edit_project', 'undo_last_change'} <= {t['name'] for t in tools}
        ov, err = c.tool('get_overview')
        assert not err and ov['project'] == so100.name and ov['mode'] == 'file'
        cov, err = c.tool('get_coverage')
        assert not err and cov['study'] == 'coverage' and cov['figure']['unit'] == '%'
        z0 = next(s for s in ov['sensors'] if s['name'] == WRIST)['position_m'][2]
        ed, err = c.tool('edit_project', changes={f'sensor:{WRIST}.z': z0 + 0.02})
        assert not err and ed['undo_step'].startswith('Claude ·'), ed
        from dexvantage.core.project import Project
        z = next(s for s in Project.load(path).variant.sensors if s.name == WRIST).mount.pose.position[2]
        assert z == pytest.approx(z0 + 0.02, abs=1e-6)
        zb = next(s for s in Project.load(path + '.bak').variant.sensors if s.name == WRIST).mount.pose.position[2]
        assert zb == pytest.approx(z0, abs=1e-6)                                 # the file as it was first opened
        un, err = c.tool('undo_last_change')
        assert not err and un['result'].startswith('Undid')
        z = next(s for s in Project.load(path).variant.sensors if s.name == WRIST).mount.pose.position[2]
        assert z == pytest.approx(z0, abs=1e-6)
        bad, err = c.tool('edit_project', changes={'sensor:nobody.x': 1})
        assert err and 'no sensor' in bad['error']
        unknown = c.send('tools/call', {'name': 'nope', 'arguments': {}})
        assert unknown['error']['code'] == -32602
    finally:
        rest = c.close()
    assert rest == ''                                                            # stdout held the protocol only
    assert all(json.loads(x)['jsonrpc'] == '2.0' for x in c.lines)


# ======================================================================= the window, through its bridge
def test_the_window_through_its_bridge(win, so100, app, tmp_path, monkeypatch):
    from tests.conftest import pump

    from dexvantage.mcp import session as MS
    from dexvantage.ui.mcp_bridge import Bridge
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'cfg'))
    win.store.set_project(so100.snapshot())
    b = Bridge(win.ctx)
    assert b.start()
    try:
        ws = MS.windows()
        assert len(ws) == 1 and ws[0]['pid'] == os.getpid() and ws[0]['project'] == so100.name
        assert oct(os.stat(b.file).st_mode & 0o777) == '0o600'
        router = Router(mode='auto')
        out = {}

        def client():
            try:
                out['status'] = router.call('dexvantage_status', {})
                out['edit'] = router.call('edit_project', {'changes': {f'sensor:{WRIST}.dz': 0.01}})
                out['undo'] = router.call('undo_last_change', {})
                bad = MS.LiveLink({**ws[0], 'token': 'wrong'})
                try:
                    bad.request('call', name='get_overview', args={})
                except MS.LinkError as e:
                    out['refused'] = str(e)
            except Exception as e:                       # noqa: BLE001 — reported by the assertion below
                out['crash'] = repr(e)
        n = win.store.undo.count()
        th = threading.Thread(target=client)
        th.start()
        t0 = time.time()
        while th.is_alive() and time.time() - t0 < 120:
            pump(app, 0.02)
        th.join(1)
        assert 'crash' not in out, out
        st = json.loads(out['status']['content'][0]['text'])
        assert st['mode'] == 'live' and st['project'] == so100.name
        ed = json.loads(out['edit']['content'][0]['text'])
        assert ed['undo_step'].startswith('Claude ·'), ed
        assert win.store.undo.count() == n + 1 and win.store.undo.index() == n      # made, then undone
        assert 'wrong token' in out['refused']
    finally:
        b.stop()
    assert MS.windows() == []
