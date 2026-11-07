"""A small client for an OpenAI-compatible chat API (Groq): the model list, streamed chat completions with tool calls,
and errors a user can act on. Standard library only (plain HTTPS), so the Groq provider needs no extra package.

ApiError is the one error type the assistant's loop handles; every provider maps its failures onto its kinds, each
with a plain sentence the dock shows (no key, offline, rate limit, ...).

Ported from Vantage assistant/client.py: the same transport and error mapping; the model table moved to providers.py,
the base URL is no longer read from the environment (a test passes url=)."""
from __future__ import annotations

import json
import ssl
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field

GROQ_URL = 'https://api.groq.com/openai/v1'
_NOT_CHAT = ('whisper', 'orpheus', 'guard', 'tts', 'playai', 'distil-whisper', 'embed')


class ApiError(Exception):
    """kind: no_key | auth | permission | rate | overloaded | model | bad_request | tool_use_failed | server | network |
    cancelled | refusal | truncated | rounds | unavailable | billing | invalid_json | internal"""

    def __init__(self, kind: str, message: str, status: int = 0, retry_after: float | None = None, detail: str = ''):
        super().__init__(message)
        self.kind, self.message, self.status, self.retry_after, self.detail = kind, message, status, retry_after, detail


def no_key(provider_label: str) -> ApiError:
    return ApiError('no_key', f'There is no {provider_label} API key: add one in Assistant settings (the key is kept '
                              'in the system keyring).')


def _error_from(status: int, body: bytes, headers) -> ApiError:
    msg, code, detail = '', '', ''
    try:
        doc = json.loads(body.decode('utf-8', 'replace'))
        err = doc.get('error', doc) if isinstance(doc, dict) else {}
        msg = str(err.get('message') or '')
        code = str(err.get('code') or err.get('type') or '')
        detail = str(err.get('failed_generation') or '')
    except Exception:                       # noqa: BLE001
        msg = body.decode('utf-8', 'replace')[:300]
    retry = None
    try:
        ra = headers.get('retry-after') if headers is not None else None
        retry = float(ra) if ra else None
    except (TypeError, ValueError):
        retry = None
    if status == 401:
        return ApiError('auth', 'Groq did not accept the API key. Check it in Assistant settings.', status)
    if status == 403:
        return ApiError('permission', msg or 'This key may not use that model.', status)
    if status == 429:
        return ApiError('rate', 'Groq is rate-limiting this key' + (f'; try again in {retry:.0f} s.' if retry else
                                                                     '; try again in a minute.'), status, retry)
    if status == 404 or code == 'model_not_found':
        return ApiError('model', msg or 'That model is not available to this key.', status)
    if code == 'tool_use_failed':
        return ApiError('tool_use_failed', msg or 'The model wrote an invalid tool call.', status, detail=detail)
    if 400 <= status < 500:
        return ApiError('bad_request', msg or f'The request was refused ({status}).', status, detail=code)
    return ApiError('server', f'Groq returned an error ({status}). Try again in a moment.', status)


def _ssl_context():
    ctx = ssl.create_default_context()
    try:                                    # bundles do not always find the system certificates; certifi does
        import certifi
        ctx.load_verify_locations(certifi.where())
    except Exception:                       # noqa: BLE001
        pass
    return ctx


def is_chat_model(model_id: str) -> bool:
    return not any(k in model_id.lower() for k in _NOT_CHAT)


@dataclass
class Delta:
    """One piece of a streamed answer."""
    content: str = ''
    reasoning: str = ''
    tool_calls: list = field(default_factory=list)     # raw OpenAI-style tool_call deltas (index, id, function...)
    finish: str | None = None
    usage: dict | None = None


class Client:
    def __init__(self, api_key: str, url: str | None = None, timeout: float = 90.0):
        from .. import __version__
        self.key = (api_key or '').strip()
        self.url = (url or GROQ_URL).rstrip('/')
        self.timeout = timeout
        self.agent = f'DexVantage/{__version__}'
        self._ssl = _ssl_context() if self.url.startswith('https') else None

    def _request(self, path, payload=None, stream=False):
        data = json.dumps(payload).encode('utf-8') if payload is not None else None
        req = urllib.request.Request(self.url + path, data=data, method='POST' if data is not None else 'GET')
        req.add_header('Authorization', f'Bearer {self.key}')
        req.add_header('User-Agent', self.agent)
        req.add_header('Accept', 'text/event-stream' if stream else 'application/json')
        if data is not None:
            req.add_header('Content-Type', 'application/json')
        try:
            return urllib.request.urlopen(req, timeout=self.timeout, context=self._ssl)
        except urllib.error.HTTPError as e:
            raise _error_from(e.code, e.read() or b'', e.headers) from None
        except urllib.error.URLError:
            raise ApiError('network', f'Could not reach {self.url.split("/")[2]}: you may be offline. Check the '
                                      'connection and try again.') from None
        except (TimeoutError, OSError):
            raise ApiError('network', 'The connection to Groq failed or timed out: check the connection and try '
                                      'again.') from None

    def models(self) -> list[str]:
        """Chat models this key may use."""
        with self._request('/models') as r:
            doc = json.loads(r.read().decode('utf-8'))
        ids = [m.get('id', '') for m in doc.get('data', []) if m.get('id') and m.get('active', True) is not False]
        return sorted(i for i in ids if is_chat_model(i))

    def stream(self, payload: dict, cancel: threading.Event | None = None):
        """Yield Delta objects for a streamed chat completion (payload without 'stream')."""
        resp = self._request('/chat/completions', dict(payload, stream=True), stream=True)
        try:
            for raw in resp:
                if cancel is not None and cancel.is_set():
                    raise ApiError('cancelled', 'Stopped.')
                line = raw.decode('utf-8', 'replace').strip()
                if not line or not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if 'error' in chunk:
                    err = chunk['error'] if isinstance(chunk['error'], dict) else {'message': str(chunk['error'])}
                    raise _error_from(int(err.get('status') or 400), json.dumps({'error': err}).encode(), None)
                usage = chunk.get('usage') or (chunk.get('x_groq') or {}).get('usage')
                for ch in chunk.get('choices') or [{}]:
                    d = ch.get('delta') or {}
                    yield Delta(content=d.get('content') or '', reasoning=d.get('reasoning') or '',
                                tool_calls=d.get('tool_calls') or [], finish=ch.get('finish_reason'), usage=usage)
                    usage = None
                if not chunk.get('choices') and usage:
                    yield Delta(usage=usage)
        except (TimeoutError, OSError):
            raise ApiError('network', 'The connection to Groq dropped: check the connection and ask again.') from None
        finally:
            resp.close()
