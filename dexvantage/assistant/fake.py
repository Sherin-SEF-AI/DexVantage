"""The fake provider: scripted replies and tool calls, so the tests (and the self-test) drive the real loop, the real
tools and the real dock without a network or a key. Every assistant test uses it; nothing here talks to an API.

A script is a list of turns; each turn is a list of replies the model gives in order, one per request of that turn:

    FakeReply(text='', calls=[('get_page_context', {})])          # call tools (their results come back next)
    FakeReply(text='Coverage is 87.5 % ...')                       # the answer: no calls, the turn ends
    FakeReply(stop='refusal', refusal={'category': None})          # a refusal
    FakeReply(error=ApiError('rate', 'Rate limited ...'))          # the request fails

A reply's text may be a function fn(results) -> str of the tool results seen so far in the turn (a list of
(name, result_dict)), so a scripted answer can quote what a tool returned. Requests are recorded (requests: the
system prompt, the tools offered, the messages) for the tests to inspect."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field

from . import providers as PR
from . import toolbox as TB
from .agent import BaseSession, Call, Reply
from .client import ApiError


@dataclass
class FakeReply:
    text: object = ''
    calls: list = field(default_factory=list)       # [(name, args)]
    stop: str = ''
    refusal: dict | None = None
    error: ApiError | None = None
    usage: tuple = (120, 40)                         # (input, output) tokens, so the usage line has something


class FakeSession(BaseSession):
    provider = PR.FAKE

    def __init__(self, settings, script=None):
        super().__init__(settings)
        self.script = [list(t) for t in (script or [])]
        self.messages: list = []
        self.requests: list = []
        self._turn: list = []
        self._results: list = []
        self._n = 0

    def reset(self):
        self.messages, self.sources = [], []
        self.usage = PR.Usage(provider=self.provider, model=self.settings.model)

    def add_turn(self, replies):
        self.script.append(list(replies))

    def _mark(self):
        return len(self.messages)

    def _rollback(self, mark):
        del self.messages[mark:]

    def _begin(self, text, note, tc):
        self._turn = self.script.pop(0) if self.script else [FakeReply('(the fake provider has no scripted reply)')]
        self._results = []
        self.messages.append({'role': 'user', 'content': f'{text}\n\n{note}'})

    def _request(self, wrap, emit, cancel) -> Reply:
        from .agent import Event
        self.requests.append({'system': self.system, 'tools': [d['name'] for d in TB.claude_defs()],
                              'messages': copy.deepcopy(self.messages), 'wrap': wrap})
        r = self._turn.pop(0) if self._turn else FakeReply('(no more scripted replies in this turn)')
        if r.error is not None:
            raise r.error
        text = r.text(list(self._results)) if callable(r.text) else str(r.text or '')
        if text:
            emit(Event('content', {'text': text}))
        self.usage.add(self.settings.model, *r.usage)
        emit(Event('usage', {'usage': self.usage}))
        calls = []
        for name, args in r.calls:
            self._n += 1
            calls.append(Call(f'fake_{self._n}', name, copy.deepcopy(args) if isinstance(args, dict) else None,
                              json.dumps(args, default=str)))
        stop = r.stop or ('tool_use' if calls else 'end_turn')
        return Reply(text, calls, stop, r.refusal, {'text': text, 'calls': [(c.name, c.args) for c in calls]})

    def _append_assistant(self, reply, wrap):
        self.messages.append({'role': 'assistant', 'content': reply.text, 'calls': reply.native.get('calls', [])})

    def _append_results(self, results):
        for call, result, ok in results:
            try:
                doc = json.loads(result)
            except ValueError:
                doc = result
            self._results.append((call.name, doc))
            self.messages.append({'role': 'tool', 'name': call.name, 'content': result, 'ok': ok})
