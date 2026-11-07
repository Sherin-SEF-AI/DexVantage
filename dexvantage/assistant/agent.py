"""One conversation with the model: streams the answer, runs the tools it calls, and reports everything as events
(called on the worker thread that runs the turn; the dock forwards them to the Qt thread).

The loop (rounds, the budget, cancelling, the events, a failed turn leaving nothing behind, the number post-check
after the answer) is the same for every provider; how a request is made and how the history is kept is the provider's
own: GroqSession here (an OpenAI-style message list), claude.ClaudeSession (Anthropic's content blocks, append-only)
and fake.FakeSession (scripted replies, for the tests).

Three rules the loop enforces whatever the model writes:
- No invented numbers: the system prompt says so, and after every answer numcheck lists the figures not found in
  what the tools returned, the page on screen, the user's words or the memory notes (Event 'answer' carries them; the
  dock shows them under the answer).
- No direct edits: a changing tool runs on a copy and comes back as a proposal (toolbox.py, edits.py); the user
  accepts it as one undo step, or not.
- No long study without an approved plan: run_study of a study that is not quick is refused with the reason; the
  model proposes a plan instead (copilot.py).

Ported from Vantage assistant/agent.py: the same loop, events, rollback and retry; DexVantage's system prompt, context
note and the number post-check."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field

from . import numcheck as NC
from . import providers as PR
from . import toolbox as TB
from .client import ApiError, Client

MAX_ROUNDS = 12                 # tool rounds per question
MAX_HISTORY_CHARS = 240_000     # Groq: older tool results are shortened past this

SYSTEM = """You are the assistant built into DexVantage, a desktop app for planning the cameras, depth sensors and \
teleop of robot-learning data collection: where the cameras go on the cell and on the robot's links, what they see at \
the grasp, whether the depth is valid, the bandwidth, the time sync, and whether a recorded dataset matches the plan. \
You help a robotics researcher understand and improve the project that is open in the app.

How to work:
- Get every number from the tools. Never guess, estimate from memory or invent a value: not a resolution, field of \
view, min-Z, data rate, joint limit, inference time or score. If a figure is not stated by the maker the app shows \
it as "not stated"; say exactly that, and do not offer a typical value. If a study has not run, say so and say where \
to run it.
- The user's question comes with the page they are on. Read it with get_page_context and cite the figures shown there \
(the headline figure, its status word) before anything else.
- Call the tools you need before answering; a few targeted calls are better than many.
- You never change the project yourself. To suggest a change, offer it with propose_changes (core factor paths such \
as sensor:<id>.z, sensor:<id>.pitch, setting:<study>.<key>); the user sees the diff and accepts it (one undo step) or \
not. A changing tool does the same: its change is shown as a proposal. Locked sensors are the user's: leave them.
- Long studies run only from a plan the user approves. To run one, call copilot_catalogue, then propose_plan with \
the steps (kind study or export, the study id, the values you set with where they came from); then say in two or \
three sentences what the plan does and its warnings. Never say a run started or finished unless a tool result says \
so. A follow-up ("only the wrist camera") revises the plan on screen: propose_plan with revise=true.
- When the user states a decision or a constraint worth keeping, offer it with remember (they accept it or not). \
Respect the memory notes you are given: never propose again what was rejected.
- Units: SI (metres, radians shown as degrees, seconds). World frame: the primary robot's base, x forward, y left, z \
up; camera optical frame z forward, x right, y down.

How to answer:
- Lead with the answer in one or two sentences, then the evidence. Be specific and brief.
- Use Markdown: short paragraphs, bullet lists, small tables. Quote figures as the tools give them, with their units \
and their intervals ("value [low–high]") where they have one.
- Say plainly when an answer depends on something DexVantage does not have (a real unit's calibration, real \
recordings, a figure the maker does not state)."""

WRAP_UP = ('[DexVantage: the budget for this request is used up. Do not call tools any more; answer now with what you '
           'have found.]')


@dataclass
class Settings:
    model: str
    effort: str = 'medium'
    show_reasoning: bool = True
    max_tokens: int = 8192
    planning: bool = False          # a Copilot turn (Claude: effort high; plain questions medium)
    fallbacks: bool = True          # Claude: server-side fallback on a refusal, where the model has one


@dataclass
class Event:
    kind: str                   # turn | reasoning | content | tool_start | tool_done | proposal | plan | memory |
                                # status | usage | answer | error | done
    data: dict = field(default_factory=dict)


@dataclass
class Call:
    """A tool call as the loop runs it: id, name, the parsed input (None when not valid JSON) and the raw text."""
    id: str
    name: str
    args: dict | None
    raw: str = ''


@dataclass
class Reply:
    text: str = ''
    calls: list = field(default_factory=list)
    stop: str = ''                  # end_turn | tool_use | max_tokens | refusal | stop | tool_calls
    refusal: dict | None = None
    native: object = None           # the provider's own message


# ------------------------------------------------------------------ privacy
_HOME = os.path.expanduser('~')
_ABS = re.compile(r'(?<![\w:/.\\~-])(?:/(?:[^\s"\'<>|*?\\/,;]+/)+)([^\s"\'<>|*?\\/,;]*)')
_KEYS = re.compile(r'\b(sk-ant-[A-Za-z0-9_-]{8,}|gsk_[A-Za-z0-9]{8,}|sk-[A-Za-z0-9]{20,})')


def mask(text: str) -> str:
    """What leaves the machine or is saved never names this machine's folders or a key: the home folder becomes ~, any
    other absolute path keeps only its last part, anything shaped like an API key is removed."""
    if not isinstance(text, str) or not text:
        return text
    if _HOME and _HOME not in ('/', ''):
        text = text.replace(_HOME, '~')
    text = _ABS.sub(lambda m: '…/' + m.group(1), text)
    return _KEYS.sub('[key removed]', text)


HANDOVER_HEAD = '[Earlier in this conversation'


def strip_handover(text: str) -> str:
    if isinstance(text, str) and text.startswith(HANDOVER_HEAD) and ']\n\n' in text:
        return text.split(']\n\n', 1)[1]
    return text


def context_note(tc, memory: bool = True) -> str:
    """The context sent with each question: the project, the page the user is on (its headline), the selection, the
    plan on screen and the memory notes."""
    bits = [f'project "{tc.project_name}"']
    if tc.variant_name:
        bits.append(f'variant "{tc.variant_name}"')
    pc = tc.page_context or {}
    if pc:
        bits.append(f'open page: {pc.get("label") or pc.get("page")} ({pc.get("headline") or pc.get("status", "")})')
    if tc.selection:
        bits.append('selected: ' + ', '.join(tc.selection))
    out = '[DexVantage context: ' + '; '.join(bits) + ']'
    if tc.plan is not None:
        pl = tc.plan
        out += (f'\n[Copilot: the plan on screen is {pl.id} v{pl.version} "{pl.title}" ({pl.status}, {len(pl.steps)} '
                f'step{"s" if len(pl.steps) != 1 else ""}); a follow-up revises it with propose_plan(revise=true).]')
    if memory and tc.memory:
        out += '\n' + tc.memory
    return mask(out)


class BaseSession:
    """The loop every provider shares. A subclass makes the requests and keeps the history."""
    provider = PR.GROQ
    memory_in_note = True               # the memory notes travel in the user message (Claude: a system message)

    def __init__(self, settings: Settings):
        self.settings = settings
        self.usage = PR.Usage(provider=self.provider, model=settings.model)
        self.lock = threading.Lock()
        self.last_payload = None
        self.handover = ''
        self.sources: list = []             # every tool result and context of this conversation (the number check)

    @property
    def system(self) -> str:
        return SYSTEM

    def _take_handover(self, text: str) -> str:
        h, self.handover = self.handover, ''
        return f'{h}\n\n{text}' if h else text

    # ---------------------------------------------------------------- one question
    def run_turn(self, text: str, tc, emit, cancel: threading.Event | None = None):
        cancel = cancel or threading.Event()
        with self.lock:
            self.usage.model = self.settings.model
            mark, n_sources, handover = self._mark(), len(self.sources), self.handover
            tc.request = text
            note = context_note(tc, self.memory_in_note)
            self.sources += [text, note, tc.memory or '', json.dumps(tc.page_context or {}, default=str)]
            self._begin(mask(self._take_handover(text)), note, tc)
            tc.emit = emit
            ok, answer = False, ''
            try:
                ok, answer = self._loop(tc, emit, cancel)
            except ApiError as e:
                emit(Event('error', {'kind': e.kind, 'message': e.message, 'detail': e.detail}))
            except Exception as e:                  # noqa: BLE001 — never leave the dock waiting
                import traceback
                traceback.print_exc()
                emit(Event('error', {'kind': 'internal',
                                     'message': f'The assistant stopped with an unexpected error ({type(e).__name__}). '
                                                'The details are in the log.'}))
            if not ok:
                self._rollback(mark)                # a failed or stopped turn leaves no half answer behind
                del self.sources[n_sources:]
                self.handover = handover
            else:
                try:
                    res = NC.unsupported(answer, self.sources)
                except Exception:                   # noqa: BLE001 — the check must never lose the answer
                    import traceback
                    traceback.print_exc()
                    res = {'checked': 0, 'unsupported': [], 'ok': True, 'failed': True}
                warn = NC.warning_text(res) or ('The figures in this answer could not be checked against the tool '
                                                'results.' if res.get('failed') else '')
                emit(Event('answer', {'text': answer, 'check': res, 'warning': warn}))
            emit(Event('done', {'ok': ok, 'usage': self.usage}))

    def _loop(self, tc, emit, cancel) -> tuple[bool, str]:
        rounds, retries, wrapped = 0, {}, False
        while True:
            if cancel.is_set():
                raise ApiError('cancelled', 'Stopped.')
            wrap = rounds >= MAX_ROUNDS - 1
            if wrap and wrapped:
                raise ApiError('rounds', f'Stopped after {rounds} rounds of tool calls. Ask a narrower question.')
            emit(Event('turn', {'round': rounds, 'max': MAX_ROUNDS}))
            try:
                reply = self._request(wrap, emit, cancel)
            except ApiError as e:
                if self._retry(e, emit, cancel, retries):
                    continue
                raise
            if wrap:
                wrapped = True
                if reply.calls and not reply.text.strip():
                    raise ApiError('rounds', f'Stopped after {rounds} rounds of tool calls. Ask a narrower question.')
                reply.calls = []
            if reply.stop == 'refusal':
                rf = reply.refusal or {}
                cat = rf.get('category')
                raise ApiError('refusal', 'The model declined this request' + (f' ({cat})' if cat else '') +
                               (f': {rf["explanation"]}' if rf.get('explanation') else '.') +
                               ' Nothing was run. Rephrase it, or ask about the project itself.',
                               detail=json.dumps(rf, default=str))
            if reply.stop == 'max_tokens' and reply.calls:
                raise ApiError('truncated', 'The answer was cut off at its length limit while writing a tool call, so '
                                            'the call was not run. Ask again, more narrowly.')
            self._append_assistant(reply, wrap)
            if not reply.calls:
                return True, reply.text
            results = []
            for call in reply.calls:
                if cancel.is_set():
                    raise ApiError('cancelled', 'Stopped.')
                info = TB.describe(call.name)
                emit(Event('tool_start', {'id': call.id, 'name': call.name, 'label': info['label'],
                                          'changes': info['changes'], 'args': call.args or {}}))
                t1 = time.perf_counter()
                if call.args is None:
                    result, ok = json.dumps({'error': 'the arguments were not valid JSON',
                                             'INVALID_JSON': call.raw[:2000]}), False
                else:
                    bad = TB.validate_args(call.name, call.args)
                    if bad:
                        result, ok = json.dumps({'error': 'the input does not match the tool\'s schema: ' +
                                                 '; '.join(bad)}), False
                    else:
                        result, ok = TB.run(tc, call.name, call.args)
                result = mask(result)
                self.sources.append(result)
                emit(Event('tool_done', {'id': call.id, 'name': call.name, 'result': result, 'ok': ok,
                                         'ms': (time.perf_counter() - t1) * 1000}))
                results.append((call, result, ok))
            self._append_results(results)
            rounds += 1

    # ---------------------------------------------------------------- the provider's part
    def _mark(self):
        raise NotImplementedError

    def _rollback(self, mark):
        raise NotImplementedError

    def _begin(self, text, note, tc):
        raise NotImplementedError

    def _request(self, wrap, emit, cancel) -> Reply:
        raise NotImplementedError

    def _append_assistant(self, reply: Reply, wrap: bool):
        raise NotImplementedError

    def _append_results(self, results):
        raise NotImplementedError

    def _retry(self, e: ApiError, emit, cancel, retries: dict) -> bool:
        if e.kind in ('rate', 'overloaded') and not retries.get(e.kind) and (e.retry_after or 0) <= 20:
            retries[e.kind] = True
            wait = e.retry_after or 3.0
            who = PR.LABEL.get(self.provider, self.provider)
            emit(Event('status', {'text': (f'Rate limited by {who}' if e.kind == 'rate' else f'{who} is overloaded')
                                  + f'; trying again in {wait:.0f} s…'}))
            if cancel.wait(wait):
                raise ApiError('cancelled', 'Stopped.')
            return True
        return False

    def reset(self):
        raise NotImplementedError


class GroqSession(BaseSession):
    """A conversation on Groq (OpenAI-compatible chat completions)."""
    provider = PR.GROQ

    def __init__(self, key: str, settings: Settings, url: str | None = None):
        if not (key or '').strip():
            from .client import no_key
            raise no_key('Groq')
        super().__init__(settings)
        self.client = Client(key, url)
        self.messages: list = []

    def reset(self):
        self.messages = []
        self.sources = []
        self.usage = PR.Usage(provider=self.provider, model=self.settings.model)

    def _mark(self):
        return len(self.messages)

    def _rollback(self, mark):
        del self.messages[mark:]

    def _begin(self, text, note, tc):
        self.messages.append({'role': 'user', 'content': f'{text}\n\n{note}'})

    def _payload(self, wrap_up=False):
        msgs = [{'role': 'system', 'content': self.system}] + self._history()
        if wrap_up:
            msgs.append({'role': 'user', 'content': WRAP_UP})
        return {'model': self.settings.model, 'messages': msgs, 'tools': TB.openai_schemas(),
                'tool_choice': 'none' if wrap_up else 'auto', 'max_completion_tokens': int(self.settings.max_tokens)}

    def _retry(self, e, emit, cancel, retries) -> bool:
        if e.kind == 'tool_use_failed' and not retries.get('tool'):
            retries['tool'] = True
            self.messages.append({'role': 'user', 'content': 'The API refused your last tool call: '
                                  + (e.detail or e.message)[:400] + ' Call the tool again with arguments that match '
                                  'its schema.'})
            emit(Event('status', {'text': 'The model wrote an invalid tool call; asking again…'}))
            return True
        return super()._retry(e, emit, cancel, retries)

    def _request(self, wrap, emit, cancel) -> Reply:
        payload = self._payload(wrap_up=wrap)
        self.last_payload = payload
        content, parts = [], {}
        for d in self.client.stream(payload, cancel):
            if d.reasoning:
                emit(Event('reasoning', {'text': d.reasoning}))
            if d.content:
                content.append(d.content)
                emit(Event('content', {'text': d.content}))
            for tcall in d.tool_calls:
                i = int(tcall.get('index', len(parts)))
                cur = parts.setdefault(i, {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                if tcall.get('id'):
                    cur['id'] = tcall['id']
                fn = tcall.get('function') or {}
                if fn.get('name') and not cur['function']['name']:
                    cur['function']['name'] = fn['name']
                if fn.get('arguments'):
                    cur['function']['arguments'] += fn['arguments']
            if d.usage:
                self.usage.add_openai(d.usage)
                emit(Event('usage', {'usage': self.usage}))
        native, calls = [], []
        for i in sorted(parts):
            c = parts[i]
            if not c['function']['name']:
                continue
            c['id'] = c['id'] or f'call_{len(self.messages)}_{i}'
            native.append(c)
            try:
                args = json.loads(c['function'].get('arguments') or '{}')
                args = {k: v for k, v in args.items() if v is not None} if isinstance(args, dict) else None
            except ValueError:
                args = None
            calls.append(Call(c['id'], c['function']['name'], args, c['function'].get('arguments') or ''))
        return Reply(''.join(content), calls, 'tool_calls' if calls else 'stop', native=native)

    def _append_assistant(self, reply, wrap):
        msg = {'role': 'assistant', 'content': reply.text or None}
        if reply.calls:
            msg['tool_calls'] = reply.native
        self.messages.append(msg)

    def _append_results(self, results):
        for call, result, ok in results:
            self.messages.append({'role': 'tool', 'tool_call_id': call.id, 'name': call.name, 'content': result})

    def _history(self):
        total = sum(len(str(m.get('content') or '')) for m in self.messages)
        if total <= MAX_HISTORY_CHARS:
            return self.messages
        last_user = max(i for i, m in enumerate(self.messages) if m['role'] == 'user')
        return [dict(m, content=m['content'][:400] + '…(shortened: ask again for the full figures)')
                if m['role'] == 'tool' and i < last_user and len(m['content']) > 400 else m
                for i, m in enumerate(self.messages)]


def new_session(provider: str, key: str | None, settings: Settings, url: str | None = None, script=None) -> BaseSession:
    """A conversation with the chosen provider: ApiError('no_key' | 'unavailable') in plain words when it cannot."""
    if provider == PR.CLAUDE:
        from .claude import ClaudeSession
        return ClaudeSession(key or '', settings, url)
    if provider == PR.FAKE:
        from .fake import FakeSession
        return FakeSession(settings, script)
    return GroqSession(key or '', settings, url)
