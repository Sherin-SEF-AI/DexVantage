"""The window's side of Claude over MCP: the DexVantage MCP server (python -m dexvantage.mcp, started by Claude Code
or Claude Desktop) sends each tool call here, and it runs on this window's project.

The bridge listens on 127.0.0.1 only, on a free port, and writes <config>/DexVantage/mcp/<pid>.json (the port and a
random token, readable by the user alone) for the server to find (mcp/session.windows). Every request carries the
token, compared in constant time. A tool runs on a worker thread so the window stays responsive; whatever touches the
project or the window is handed to the UI thread (assistant/session.StoreSession). A change is one undo step
"Claude · …" in Edit ▸ Undo, like the user's own. The setting mcp/bridge (on by default) turns it off.

Ported from Vantage ui/mcp_bridge.py (the local server, the token file, worker threads, replies on the UI thread).

Known limits: one window's bridge serves any number of MCP servers, one request at a time per connection; tools that
change the project run one at a time (mcp/service.py's lock).
"""
from __future__ import annotations

import hmac
import json
import os
import secrets
import threading
import time

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtNetwork import QHostAddress, QTcpServer

SETTING = 'mcp/bridge'


def enabled(settings) -> bool:
    try:
        return settings.value(SETTING, True, type=bool)
    except (TypeError, ValueError):
        return True


class Bridge(QObject):
    """The local server the MCP process talks to."""
    _reply = pyqtSignal(object, object)              # (socket, reply dict): written on the UI thread
    calls = pyqtSignal(str)                          # a tool Claude called (for the status bar)

    def __init__(self, ctx):
        super().__init__()
        from ..assistant.session import StoreSession
        self.ctx = ctx
        self.server = None
        self.token = ''
        self.file = ''
        self.session = StoreSession(ctx, allow_long=True)     # the MCP client asks the user before each call
        self._buf: dict = {}
        self._reply.connect(self._write)

    @property
    def running(self) -> bool:
        return self.server is not None and self.server.isListening()

    def start(self) -> bool:
        if self.running:
            return True
        self.server = QTcpServer(self)
        if not self.server.listen(QHostAddress(QHostAddress.SpecialAddress.LocalHost), 0):
            self.server = None
            return False
        self.server.newConnection.connect(self._accept)
        self.token = secrets.token_hex(24)
        self._publish()
        self.ctx.store.project_replaced.connect(self._publish)
        return True

    def stop(self):
        if self.server is not None:
            self.server.close()
            self.server.deleteLater()
            self.server = None
        try:
            self.ctx.store.project_replaced.disconnect(self._publish)
        except (TypeError, RuntimeError):
            pass
        if self.file:
            try:
                os.remove(self.file)
            except OSError:
                pass
            self.file = ''

    def _publish(self, *_):
        from ..mcp.session import bridge_dir
        if not self.running:
            return
        d = bridge_dir()
        try:
            os.makedirs(d, mode=0o700, exist_ok=True)
            self.file = os.path.join(d, f'{os.getpid()}.json')
            p = self.ctx.store.project
            doc = {'pid': os.getpid(), 'port': int(self.server.serverPort()), 'token': self.token,
                   'project': p.name, 'path': p.path, 'started': time.time()}
            tmp = self.file + '.tmp'
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(doc, f)
            os.replace(tmp, self.file)
        except OSError:
            import traceback
            traceback.print_exc()

    # ---------------------------------------------------------------- connections
    def _accept(self):
        while self.server is not None and self.server.hasPendingConnections():
            sock = self.server.nextPendingConnection()
            self._buf[id(sock)] = b''
            sock.readyRead.connect(lambda s=sock: self._read(s))
            sock.disconnected.connect(lambda s=sock: (self._buf.pop(id(s), None), s.deleteLater()))

    def _read(self, sock):
        buf = self._buf.get(id(sock), b'') + bytes(sock.readAll())
        if len(buf) > 32 << 20:
            sock.abort()
            return
        while b'\n' in buf:
            line, buf = buf.split(b'\n', 1)
            self._handle(sock, line)
        self._buf[id(sock)] = buf

    def _handle(self, sock, line):
        from ..mcp import service as SV
        try:
            req = json.loads(line.decode('utf-8'))
        except (UnicodeDecodeError, ValueError):
            return self._write(sock, {'id': None, 'error': 'not JSON'})
        rid = req.get('id') if isinstance(req, dict) else None
        if not isinstance(req, dict) or not hmac.compare_digest(str(req.get('token', '')), self.token):
            self._write(sock, {'id': rid, 'error': 'Refused: wrong token.'})
            sock.disconnectFromHost()
            return
        if req.get('method') == 'ping':
            p = self.ctx.store.project
            return self._write(sock, {'id': rid, 'result': {'project': p.name}})
        if req.get('method') != 'call':
            return self._write(sock, {'id': rid, 'error': f'No method {req.get("method")!r}.'})
        name, args = str(req.get('name', '')), req.get('args') or {}
        t = SV.BY_NAME.get(name)
        self.calls.emit(name)
        w = self.ctx.window
        if w is not None and t is not None:
            w.msg(f'Claude: {t.title}…', 4000)
        threading.Thread(target=self._work, args=(sock, rid, name, args), daemon=True).start()

    def _work(self, sock, rid, name, args):
        from ..mcp import service as SV
        out = SV.call(self.session, name, args)
        try:
            self._reply.emit(sock, {'id': rid, 'result': out})
        except RuntimeError:                         # the window closed meanwhile
            pass

    def _write(self, sock, doc):
        try:
            if sock.state() == sock.SocketState.ConnectedState:
                sock.write((json.dumps(doc, ensure_ascii=False, default=str) + '\n').encode('utf-8'))
        except RuntimeError:                         # the socket is gone
            pass


def start_bridge(window):
    """Start the bridge for a window when the setting allows it (app.py calls this); None when off or it cannot
    listen."""
    ctx = window.ctx
    if not enabled(ctx.settings):
        return None
    b = Bridge(ctx)
    if not b.start():
        return None
    window.mcp_bridge = b
    return b
