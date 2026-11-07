"""The DexVantage MCP server: Claude (Claude Code, Claude Desktop or any MCP client) reads and changes DexVantage
projects through the assistant's tools.

    python -m dexvantage.mcp [--project FILE.dexv | --sample so100] [--mode auto|live|file]

It speaks the Model Context Protocol over stdio with the Python standard library only: JSON-RPC 2.0, one message
per line, UTF-8, nothing but protocol messages on stdout (logs go to stderr). The protocol version implemented is
2025-11-25 (https://modelcontextprotocol.io/specification/2025-11-25, read 2026-11-07: basic/lifecycle,
basic/transports, server/tools); a client asking for 2025-06-18, 2025-03-26 or 2024-11-05 gets that version (the
messages used here are the same in each), any other the latest. Methods: initialize (protocolVersion, the tools
capability, serverInfo, instructions), ping, tools/list, tools/call, and empty resources/list and prompts/list;
notifications (notifications/initialized, notifications/cancelled) need no answer. An unknown tool is a protocol
error (-32602), as the 2025-11-25 tools page says; a tool that cannot do what was asked answers with isError and a
plain sentence (a tool execution error).

Each tool call goes to the DexVantage window that is open and lets Claude in (with --project: the window that has
that file open; live: the change shows there as an undo step "Claude · …"), or, with no such window, to the project
file (file mode: analysed here, saved after each
change, the file as first opened kept as .bak). --mode file never attaches to a window; --mode live only does.

Register it with Claude Code once:

    claude mcp add dexvantage -- python -m dexvantage.mcp --project path/to/project.dexv

Ported from Vantage mcp/server.py (the router between the window and a file, the stdio loop).

Known limits: one request at a time (a long study blocks the next call until it ends); no list-changed
notifications, no progress notifications, no task-augmented calls; JSON-RPC batches (removed from MCP in
2025-06-18) are still answered, one reply each.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

from .. import __version__
from ..assistant.session import FileSession, SessionError
from . import service as SV
from .session import LinkError, LiveLink, same_file, windows

PROTOCOL = '2025-11-25'
VERSIONS = (PROTOCOL, '2025-06-18', '2025-03-26', '2024-11-05')


class Router:
    """Which session a call goes to: with --project, the window that has that file open, else the file; without,
    the newest open window, else the file opened with open_project."""

    def __init__(self, project: str | None = None, mode: str = 'auto', sample: str | None = None):
        self.mode = mode
        self.project = project
        self.sample = sample
        self.file = FileSession()
        self.link = None
        self.sticky_file = mode == 'file'
        self._tried = False

    def session(self):
        if not self.sticky_file:
            ws = windows()
            if self.project:                             # --project: the window that has it open, else the file
                ws = [w for w in ws if same_file(w.get('path', ''), self.project)]
            if ws:
                pick = next((w for w in ws if self.link is not None and int(w['pid']) == self.link.pid), None) or ws[0]
                if self.link is None or self.link.pid != int(pick['pid']):
                    if self.link is not None:
                        self.link.close()
                    self.link = LiveLink(pick)
                else:
                    self.link.info = pick
                return self.link
            if self.mode == 'live':
                raise LinkError('No DexVantage window is open with Claude allowed in.')
        if self.file.p is None and not self._tried and (self.project or self.sample):
            self._tried = True
            r = self.file.open(path=self.project) if self.project else self.file.open(sample=self.sample)
            if 'error' in r:
                print(f'dexvantage mcp: {r["error"]}', file=sys.stderr)
        return self.file

    def open(self, args: dict) -> dict:
        if not args.get('path') and not args.get('sample'):
            self.sticky_file = self.mode == 'file'
            return {'result': 'Working in the DexVantage window again.' if windows() and not self.sticky_file else
                    'No DexVantage window is open: give path or sample to work on a project here.'}
        r = self.file.open(path=args.get('path'), sample=args.get('sample'))
        if 'error' not in r:
            self.sticky_file = True
            r['note'] = ('Claude works on this project now, even with a DexVantage window open; open_project without '
                         'arguments goes back to the window.')
        return r

    def call(self, name: str, args) -> dict:
        if name == 'open_project':
            return SV.result(self.open(args if isinstance(args, dict) else {}))
        try:
            s = self.session()
        except LinkError as e:
            return SV.result({'error': str(e)})
        if isinstance(s, LiveLink):
            try:
                return s.request('call', name=name, args=args or {})
            except LinkError as e:
                self.link = None
                return SV.result({'error': f'{e} Call the tool again to use whatever is open now.'})
        try:
            return SV.call(s, name, args)
        except SessionError as e:
            return SV.result({'error': str(e)})


class Server:
    def __init__(self, router: Router, out=None):
        self.router = router
        self.out = out

    def send(self, doc):
        self.out.write(json.dumps(doc, ensure_ascii=False, default=str) + '\n')
        self.out.flush()

    @staticmethod
    def _error(mid, code, message):
        return {'jsonrpc': '2.0', 'id': mid, 'error': {'code': code, 'message': message}}

    def handle(self, msg):
        if not isinstance(msg, dict) or msg.get('jsonrpc') != '2.0' or not isinstance(msg.get('method', ''), str):
            if isinstance(msg, dict) and 'method' not in msg and ('result' in msg or 'error' in msg):
                return None                              # a response to a request we never sent: nothing to say
            return self._error(msg.get('id') if isinstance(msg, dict) else None, -32600, 'Invalid request')
        mid, method, params = msg.get('id'), msg.get('method'), msg.get('params') or {}
        if 'id' not in msg:                              # a notification: nothing to answer
            return None
        if not isinstance(params, dict):
            return self._error(mid, -32602, 'params must be an object')
        try:
            if method == 'initialize':
                want = params.get('protocolVersion')
                result = {'protocolVersion': want if want in VERSIONS else PROTOCOL,
                          'capabilities': {'tools': {'listChanged': False}},
                          'serverInfo': {'name': 'dexvantage', 'title': 'DexVantage', 'version': __version__},
                          'instructions': SV.INSTRUCTIONS}
            elif method == 'ping':
                result = {}
            elif method == 'tools/list':
                result = {'tools': SV.listing()}
            elif method == 'tools/call':
                name = params.get('name')
                if not isinstance(name, str) or name not in SV.BY_NAME:
                    return self._error(mid, -32602, f'Unknown tool: {name}')
                args = params.get('arguments')
                if args is not None and not isinstance(args, dict):
                    return self._error(mid, -32602, 'arguments must be an object')
                result = self.router.call(name, args or {})
            elif method in ('resources/list', 'prompts/list'):
                result = {method.split('/')[0]: []}
            elif method == 'resources/templates/list':
                result = {'resourceTemplates': []}
            else:
                return self._error(mid, -32601, f'Method not found: {method}')
        except Exception as e:                           # noqa: BLE001 — the server keeps running
            traceback.print_exc(file=sys.stderr)
            return self._error(mid, -32603, f'Internal error ({type(e).__name__})')
        return {'jsonrpc': '2.0', 'id': mid, 'result': result}

    def serve(self, inp):
        for raw in inp:
            line = raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                self.send(self._error(None, -32700, 'Parse error'))
                continue
            if isinstance(msg, list):
                replies = [r for r in (self.handle(m) for m in msg) if r is not None]
                if replies:
                    self.send(replies)
                continue
            reply = self.handle(msg)
            if reply is not None:
                self.send(reply)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='python -m dexvantage.mcp', description='DexVantage MCP server (stdio)')
    ap.add_argument('--project', help='a .dexv project to work on when no DexVantage window is open')
    from ..core.robot.samples import SAMPLES
    ap.add_argument('--sample', choices=tuple(SAMPLES), help='work on a bundled sample (unsaved)')
    ap.add_argument('--mode', choices=('auto', 'live', 'file'), default='auto',
                    help='auto: the open window, else the file; live: the window only; file: never the window')
    args = ap.parse_args(argv)
    out = sys.stdout
    sys.stdout = sys.stderr                              # only protocol messages may reach stdout
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    try:
        out.reconfigure(encoding='utf-8')
        sys.stdin.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    project = os.path.abspath(os.path.expanduser(args.project)) if args.project else None
    Server(Router(project, args.mode, args.sample), out).serve(sys.stdin)
    return 0
