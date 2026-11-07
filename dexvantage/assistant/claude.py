"""Claude as the assistant's model, through Anthropic's official Python SDK (the optional `anthropic` package, extra
'assistant'; without it the settings say how to install it and Groq still works).

How the requests are made (from Anthropic's documentation for these models, as the Claude API reference states it,
read 2026-11-07):
- Models: claude-opus-5-5 (the default), claude-sonnet-5-5 and claude-haiku-5-5, exact ids. Thinking is adaptive
  (thinking: {type: adaptive}; it cannot be turned off on Opus 5.5); output_config.effort is set explicitly, high on a
  Copilot (planning) turn and medium for a plain question. budget_tokens, temperature, top_p and top_k are never sent.
- Streaming with client.beta.messages.stream() and get_final_message(); every tool is declared with
  eager_input_streaming, so the loop validates each parsed input against its schema before running it, and checks the
  stop_reason first: max_tokens with a tool call runs nothing, a refusal runs nothing and says so plainly.
- tool_choice is auto (forced tool choice is refused by these models); none for the wrap-up request.
- Prompt caching: one frozen system prompt (the turn's context goes in its user message), the tools in a fixed order,
  a cache_control breakpoint on the system prompt and automatic caching of the conversation's tail.
- Append-only history: the assistant's response.content goes back unchanged, thinking blocks included; long
  conversations use the server's compaction (beta compact-2026-01-12). A failed turn is removed from the end only.
- Refusals: server-side fallback (fallbacks: "default", beta server-side-fallback-2026-07-01) on the models that have
  it (Opus 5.5 and Sonnet 5.5; Haiku 5.5 has none); a response that fell back mid-output is echoed without the
  thinking and tool_use blocks before its last fallback block.
- The memory notes go in as a mid-conversation system message after the user's message, only when they changed.
- Errors: the SDK's typed exceptions, most specific first, each as a plain sentence (no key, offline, rate limit,
  overloaded, ...). The key is passed explicitly; the SDK is never left to look for one of its own.

Ported from Vantage assistant/claude.py, without its strict-schema selection (every input is validated before it
runs either way) and its picture blocks."""
from __future__ import annotations

import json

from . import providers as PR
from . import toolbox as TB
from .agent import WRAP_UP, BaseSession, Call, Reply, Event
from .client import ApiError, no_key

try:
    import anthropic
except ImportError:                     # optional: Groq still works
    anthropic = None

BETA_COMPACT = 'compact-2026-01-12'
BETA_FALLBACK = 'server-side-fallback-2026-07-01'
COMPACT_EDIT = {'type': 'compact_20260112'}
MAX_TOKENS = 32000                      # thinking counts towards it
EFFORT_PLANNING, EFFORT_QUESTION = 'high', 'medium'
JSON_RETRIES = 2
EMPTY_MEMORY = 'MEMORY\nThe memory notes are now empty: earlier notes no longer apply.'


class ClaudeUnavailable(ApiError):
    def __init__(self):
        super().__init__('unavailable', PR.INSTALL[PR.CLAUDE])


def _plain(o):
    if hasattr(o, 'to_dict'):
        return o.to_dict()
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    return o


def echo_content(content) -> list:
    """The assistant content to pass back: unchanged, except after a mid-output fallback, where the thinking and
    tool_use blocks before the last fallback block are left out (Anthropic's echo rule)."""
    blocks = list(content or [])
    idx = [i for i, b in enumerate(blocks) if getattr(b, 'type', None) == 'fallback']
    if not idx:
        return blocks
    last = idx[-1]
    return [b for i, b in enumerate(blocks) if i >= last or getattr(b, 'type', None) not in
            ('thinking', 'redacted_thinking', 'tool_use', 'server_tool_use')]


def _message(e) -> str:
    body = getattr(e, 'body', None)
    err = body.get('error') if isinstance(body, dict) else None
    text = err.get('message') if isinstance(err, dict) else None
    return str(text or getattr(e, 'message', '') or '')


def map_error(e) -> ApiError:
    """The SDK's exception as an error the dock shows, most specific class first."""
    A = anthropic
    msg = _message(e)
    if isinstance(e, A.RateLimitError):
        retry = None
        try:
            ra = e.response.headers.get('retry-after')
            retry = float(ra) if ra else None
        except (TypeError, ValueError, AttributeError):
            retry = None
        return ApiError('rate', 'Anthropic is rate-limiting this key' + (f'; try again in {retry:.0f} s.' if retry
                                                                          else '; try again in a minute.'), 429, retry)
    if isinstance(e, A.AuthenticationError):
        return ApiError('auth', 'Anthropic did not accept the API key. Check it in Assistant settings.', 401)
    if isinstance(e, A.PermissionDeniedError):
        return ApiError('permission', msg or 'This key may not do that.', 403)
    if isinstance(e, A.NotFoundError):
        return ApiError('model', msg or 'That model is not available to this key: pick another in Assistant '
                                        'settings.', 404)
    if isinstance(e, A.BadRequestError):
        return ApiError('bad_request', msg or 'The request was refused (400).', 400)
    if isinstance(e, A.OverloadedError):
        return ApiError('overloaded', 'Anthropic\'s API is overloaded at the moment; try again shortly.', 529)
    if isinstance(e, A.APIStatusError):
        st = int(getattr(e, 'status_code', 0) or 0)
        if st == 402:
            return ApiError('billing', 'Anthropic reports a billing problem with this key\'s account.', 402)
        if st >= 500:
            return ApiError('server', f'Anthropic returned an error ({st}). Try again in a moment.', st)
        return ApiError('bad_request', msg or f'The request was refused ({st}).', st)
    if isinstance(e, A.APIConnectionError):
        return ApiError('network', 'Could not reach Anthropic\'s API: you may be offline. Check the connection and '
                                   'try again.')
    return ApiError('server', f'The request to Anthropic failed ({type(e).__name__}).')


class ClaudeSession(BaseSession):
    """A conversation with Claude. self.messages is append-only (see the module doc)."""
    provider = PR.CLAUDE
    memory_in_note = False

    def __init__(self, key: str, settings, url: str | None = None):
        if anthropic is None:
            raise ClaudeUnavailable()
        if not (key or '').strip():             # never let the SDK look for a key of its own (environment, profile)
            raise no_key('Claude')
        super().__init__(settings)
        self.client = anthropic.Anthropic(api_key=key.strip(), base_url=url or None,
                                          timeout=anthropic.Timeout(300.0, connect=15.0))
        self.system_blocks = [{'type': 'text', 'text': self.system, 'cache_control': {'type': 'ephemeral'}}]
        self.tools = TB.claude_defs()
        self.messages: list = []

    def reset(self):
        self.messages, self.sources = [], []
        self.usage = PR.Usage(provider=self.provider, model=self.settings.model)

    def _mark(self):
        return len(self.messages)

    def _rollback(self, mark):
        del self.messages[mark:]

    def _begin(self, text, note, tc):
        self.messages.append({'role': 'user', 'content': [{'type': 'text', 'text': f'{text}\n\n{note}'}]})
        mem = tc.memory or ''
        last = self.memory_sent()
        if (mem or last) and (mem or EMPTY_MEMORY) != last:
            self.messages.append({'role': 'system', 'content': mem or EMPTY_MEMORY})

    def memory_sent(self) -> str:
        return next((m['content'] for m in reversed(self.messages) if m.get('role') == 'system'), '')

    def request_kwargs(self, wrap=False) -> dict:
        st = self.settings
        info = PR.model_info(PR.CLAUDE, st.model)
        fallback = bool(st.fallbacks and info.fallbacks)
        kw = {'model': st.model, 'max_tokens': MAX_TOKENS, 'system': self.system_blocks, 'tools': self.tools,
              'messages': self.messages, 'tool_choice': {'type': 'none' if wrap else 'auto'},
              'thinking': {'type': 'adaptive', 'display': 'summarized'} if st.show_reasoning else {'type': 'adaptive'},
              'output_config': {'effort': EFFORT_PLANNING if st.planning else EFFORT_QUESTION},
              'cache_control': {'type': 'ephemeral'}, 'context_management': {'edits': [dict(COMPACT_EDIT)]},
              'betas': [BETA_COMPACT] + ([BETA_FALLBACK] if fallback else [])}
        if fallback:
            kw['fallbacks'] = 'default'
        return kw

    def _request(self, wrap, emit, cancel) -> Reply:
        if wrap and self.messages and self.messages[-1]['role'] == 'user' and \
                not any(isinstance(b, dict) and b.get('text') == WRAP_UP for b in self.messages[-1]['content']):
            self.messages[-1]['content'].append({'type': 'text', 'text': WRAP_UP})
        kw = self.request_kwargs(wrap)
        self.last_payload = _plain(dict(kw))
        try:
            with self.client.beta.messages.stream(**kw) as stream:
                for ev in stream:
                    if cancel.is_set():
                        raise ApiError('cancelled', 'Stopped.')
                    t = getattr(ev, 'type', '')
                    if t == 'thinking' and getattr(ev, 'thinking', ''):
                        emit(Event('reasoning', {'text': ev.thinking}))
                    elif t == 'text' and getattr(ev, 'text', ''):
                        emit(Event('content', {'text': ev.text}))
                msg = stream.get_final_message()
        except ApiError:
            raise
        except ValueError as e:             # tool input JSON the SDK could not parse at all
            raise ApiError('invalid_json', f'The model wrote a tool input that is not JSON ({e}).') from None
        except anthropic.AnthropicError as e:
            raise map_error(e) from None
        u = msg.usage
        self.usage.add(getattr(msg, 'model', None) or self.settings.model, u.input_tokens, u.output_tokens,
                       getattr(u, 'cache_creation_input_tokens', 0) or 0, getattr(u, 'cache_read_input_tokens', 0) or 0)
        emit(Event('usage', {'usage': self.usage}))
        content = echo_content(msg.content)
        text = ''.join(b.text for b in content if getattr(b, 'type', None) == 'text')
        calls = []
        for b in content:
            if getattr(b, 'type', None) == 'tool_use':
                inp = b.input
                calls.append(Call(b.id, b.name, inp if isinstance(inp, dict) else None,
                                  json.dumps(inp, default=str) if not isinstance(inp, str) else inp))
        refusal = None
        if msg.stop_reason == 'refusal':
            sd = getattr(msg, 'stop_details', None)
            refusal = {'category': getattr(sd, 'category', None), 'explanation': getattr(sd, 'explanation', None)}
        return Reply(text, calls, msg.stop_reason or '', refusal, content)

    def _append_assistant(self, reply, wrap):
        self.messages.append({'role': 'assistant', 'content': list(reply.native or [])})
        if wrap:                            # told not to call tools: any call it made still needs its result
            pending = [b for b in reply.native or [] if getattr(b, 'type', None) == 'tool_use']
            if pending:
                self.messages.append({'role': 'user', 'content': [
                    {'type': 'tool_result', 'tool_use_id': b.id, 'is_error': True,
                     'content': 'Not run: the budget for this request is used up.'} for b in pending]})

    def _append_results(self, results):
        blocks = []
        for call, result, ok in results:
            b = {'type': 'tool_result', 'tool_use_id': call.id, 'content': result}
            if not ok:
                b['is_error'] = True
            blocks.append(b)
        self.messages.append({'role': 'user', 'content': blocks})

    def _retry(self, e, emit, cancel, retries) -> bool:
        if e.kind == 'invalid_json' and retries.get('json', 0) < JSON_RETRIES:
            retries['json'] = retries.get('json', 0) + 1
            emit(Event('status', {'text': 'The model wrote a tool input that was not JSON; asking again…'}))
            return True
        return super()._retry(e, emit, cancel, retries)
