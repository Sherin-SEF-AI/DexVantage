"""Finding an open DexVantage window and talking to it (the MCP process's side of the window bridge).

A window that lets Claude in (ui/mcp_bridge.py; Preferences: the setting mcp/bridge, on by default) listens on
127.0.0.1 only, on a free port, and writes <config>/DexVantage/mcp/<pid>.json (readable by the user alone) with its
port, a random token, the project's name and file. Every request carries the token; one JSON line each way:

    -> {"token": ..., "id": n, "method": "call", "name": "get_coverage", "args": {...}}
    <- {"id": n, "result": {...the tool's JSON...}}      or {"id": n, "error": "a plain sentence"}

Without a window, the server works on a project file (assistant/session.FileSession).

Ported from Vantage mcp/session.py (windows(), LiveLink); the file session moved to assistant/session.py.

Known limits: a window that crashed leaves its file until the next windows() call finds its process gone.
"""
from __future__ import annotations

import json
import os
import socket


def config_dir() -> str:
    """DexVantage's configuration folder: $XDG_CONFIG_HOME/DexVantage, XDG_CONFIG_HOME defaulting to ~/.config (the
    XDG Base Directory specification; DexVantage supports Linux only)."""
    root = os.environ.get('XDG_CONFIG_HOME') or os.path.join(os.path.expanduser('~'), '.config')
    return os.path.join(root, 'DexVantage')


def bridge_dir() -> str:
    return os.path.join(config_dir(), 'mcp')


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def windows() -> list:
    """The open DexVantage windows that let Claude in, newest first (files of windows that are gone are removed)."""
    out = []
    try:
        names = os.listdir(bridge_dir())
    except OSError:
        return out
    for n in names:
        if not n.endswith('.json'):
            continue
        f = os.path.join(bridge_dir(), n)
        try:
            with open(f, encoding='utf-8') as fh:
                d = json.load(fh)
            if not _alive(int(d.get('pid', 0))):
                os.remove(f)
                continue
            out.append(d)
        except (OSError, ValueError, TypeError):
            continue
    return sorted(out, key=lambda d: -float(d.get('started', 0)))


def same_file(a: str, b: str) -> bool:
    try:
        return bool(a) and bool(b) and os.path.samefile(a, b)
    except OSError:
        return bool(a) and bool(b) and os.path.abspath(a) == os.path.abspath(b)


class LinkError(Exception):
    """The window could not be reached or refused the request; a plain sentence."""


class LiveLink:
    """Requests to one window's bridge."""
    mode = 'live'

    def __init__(self, info: dict, timeout: float = 900.0):
        self.info = info
        self.timeout = timeout
        self._sock = None
        self._buf = b''
        self._n = 0

    @property
    def pid(self) -> int:
        return int(self.info.get('pid', 0))

    def _connect(self):
        if self._sock is None:
            try:
                self._sock = socket.create_connection(('127.0.0.1', int(self.info['port'])), timeout=10)
            except (OSError, KeyError, ValueError) as e:
                raise LinkError(f'The DexVantage window (process {self.pid}) does not answer ({e}).') from None
            self._sock.settimeout(self.timeout)

    def request(self, method: str, **params) -> dict:
        self._connect()
        self._n += 1
        msg = {'token': self.info.get('token', ''), 'id': self._n, 'method': method, **params}
        try:
            self._sock.sendall((json.dumps(msg, default=str) + '\n').encode('utf-8'))
            while b'\n' not in self._buf:
                chunk = self._sock.recv(1 << 20)
                if not chunk:
                    raise OSError('the window closed the connection')
                self._buf += chunk
        except OSError as e:
            self.close()
            raise LinkError(f'Lost the DexVantage window (process {self.pid}): {e}.') from None
        line, self._buf = self._buf.split(b'\n', 1)
        doc = json.loads(line.decode('utf-8'))
        if 'error' in doc:
            raise LinkError(str(doc['error']))
        return doc.get('result') or {}

    def close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock, self._buf = None, b''
