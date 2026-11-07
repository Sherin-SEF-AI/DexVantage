"""The tools the dock's assistant offers the model: assistant/tools.py's TOOLS (the same tools the MCP server serves:
the reading tools of every study, get_page_context, edit_project, simulate_changes, run_study, ...) plus the dock's
own (propose_changes, propose_plan, copilot_catalogue, remember), all run against a session that never changes the
project directly.

ProposingSession (a mixin over session.StoreSession for the window, session.FileSession headless) is what the tools
see as their session:
- edit(keys, fn, text) runs fn on a copy and records the difference as an edit proposal (edits.py) instead of
  changing the project; the tool's result says so, and the dock shows the proposal with Accept and Discard;
- undo() and redo() refuse in plain words (the user undoes with Edit ▸ Undo);
- allow_long is False, and run() refuses run_study for a long study before the tool sees it: such a study runs only
  from a Copilot plan the user approved;
- propose(plan) and the dock's propose_plan show the plan as a plan card;
- page_context() returns the page the user was on when they asked (read on the UI thread before the turn).

No Qt here (StoreSession's own UI-thread hand-over is used where a window is present)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import copilot as CP
from . import edits as ED
from . import memory as MM

RESULT_MAX = 24_000             # characters of one tool result sent to the model


@dataclass
class LocalTool:
    name: str
    description: str
    input_schema: dict
    fn: object                  # fn(tc, **args) -> dict
    changes: bool = False
    label: str = ''


@dataclass
class ToolContext:
    """What a turn's tools work with (built on the UI thread when the question is asked)."""
    session: object = None              # the tools' session (a ProposingSession)
    project_name: str = ''
    variant_name: str = ''
    page_context: dict | None = None
    selection: list = field(default_factory=list)
    memory: str = ''                    # memory.prompt_block
    plan: object = None                 # the Copilot plan on screen (copilot.Plan) or None
    provider: str = ''
    model: str = ''
    request: str = ''
    last_seconds: dict = field(default_factory=dict)    # study id -> seconds of its last run in this session
    emit: object = None                 # the loop's emit(Event)
    proposals: list = field(default_factory=list)       # edits.Proposal made this turn
    plans: list = field(default_factory=list)           # copilot.Plan proposed this turn
    notes: list = field(default_factory=list)           # memory entries offered this turn

    def project(self):
        return self.session.project()

    def _event(self, kind, data):
        if self.emit is not None:
            from .agent import Event
            self.emit(Event(kind, data))

    def add_proposal(self, prop):
        self.proposals.append(prop)
        self._event('proposal', {'proposal': prop})

    def add_plan(self, plan):
        self.plans.append(plan)
        self.plan = plan
        self._event('plan', {'plan': plan})

    def add_note(self, entry):
        self.notes.append(entry)
        self._event('memory', {'entry': entry})


# =========================================================================== the session the tools see
class Proposing:
    """Mixin: edits become proposals, no undo, no long studies (module docstring)."""
    allow_long = False

    def bind(self, tc: ToolContext):
        self.tc = tc
        return self

    def page_context(self):
        tc = getattr(self, 'tc', None)
        return tc.page_context if tc is not None else None

    def edit(self, keys, fn, text: str):
        before = self.project()
        after = before.snapshot()
        fn(after)
        title = ' '.join(str(text or 'Proposed change').split()).removeprefix('Claude ·').strip()
        prop = ED.from_projects(before, after, title, 'tool')
        if prop is None:
            return 'Nothing would change: the project already has those values.'
        self.tc.add_proposal(prop)
        return f'Proposed, not applied: {title} (proposal {prop.id}; the user accepts or discards it).'

    def undo(self) -> str:
        return ('The assistant does not change the project itself, so it has nothing to undo: the user undoes an '
                'accepted proposal with Edit ▸ Undo.')

    def redo(self) -> str:
        return 'The assistant has nothing to redo: the user redoes with Edit ▸ Redo.'

    def propose(self, plan) -> dict:
        p = plan if isinstance(plan, CP.Plan) else CP.Plan.from_dict(plan) if isinstance(plan, dict) and \
            plan.get('schema') == CP.PLAN_SCHEMA else None
        if p is None:
            return {'shown': False, 'note': 'Use propose_plan to put a plan on screen.'}
        self.tc.add_plan(p)
        return {'shown': True, 'note': 'The plan is on screen for the user to approve; nothing runs before.'}


def dock_session(ctx, tc: ToolContext):
    """The window's session for one turn (reads through the window, edits as proposals)."""
    from .session import StoreSession

    class DockSession(Proposing, StoreSession):
        pass
    return DockSession(ctx, allow_long=False).bind(tc)


def file_session(project, tc: ToolContext):
    """A session on a project without the window (tests, the self-test, batch)."""
    from .session import FileSession

    class HeadlessSession(Proposing, FileSession):
        pass
    s = HeadlessSession(project=project, allow_long=False)
    s.allow_long = False
    return s.bind(tc)


# =========================================================================== the dock's own tools
def _propose_changes(tc, title='', changes=None, reason='', ops=None):
    project = tc.project()
    prop = ED.from_factors(project, title, changes or [], reason, ops)
    rejected = [ln[2:] for ln in (tc.memory or '').splitlines() if ln.startswith('- rejected')
                and any(str(r.get('title', '')) in ln for r in prop.rows)]
    tc.add_proposal(prop)
    out = prop.for_model()
    if rejected:
        out['memory'] = 'A memory note rejected a change to the same quantity: ' + '; '.join(rejected[:3])
    return out


def _propose_plan(tc, title='', rationale='', steps=None, revise=False):
    project = tc.project()
    prev = tc.plan if isinstance(tc.plan, CP.Plan) else None
    plan = CP.build(project, steps or [], title, rationale, prev=prev, request=tc.request, provider=tc.provider,
                    model=tc.model, revise=bool(revise) and prev is not None, last_seconds=tc.last_seconds)
    tc.add_plan(plan)
    return CP.for_model(plan)


def _catalogue(tc):
    return CP.catalogue(tc.project())


def _remember(tc, kind='note', text=''):
    e = MM.new_entry(kind, text, 'assistant')
    if e is None:
        return {'error': 'A memory note needs text.'}
    tc.add_note(e)
    return {'offered': e, 'note': 'Shown to the user with Keep and Discard; it is kept only if they accept it.'}


def _page_context(tc):
    pc = tc.page_context
    if not pc:
        return {'page': None, 'note': 'No analysis page is shown in the window (or there is no window).'}
    return pc


_STEP = {'type': 'object', 'properties': {
    'kind': {'type': 'string', 'enum': list(CP.KINDS), 'description': 'study, export or edit'},
    'label': {'type': 'string', 'description': 'A short label for the step'},
    'study': {'type': 'string', 'description': 'kind study: the study id (copilot_catalogue lists them)'},
    'settings': {'type': 'object', 'description': 'kind study: the settings for this run, {key: value}; the rest keep '
                                                  'the page\'s values'},
    'values': {'type': 'array', 'description': 'kind study, instead of settings: each value with where it came from',
               'items': {'type': 'object', 'properties': {
                   'key': {'type': 'string'}, 'value': {'type': 'string', 'description': 'JSON text: "0.5", "true"'},
                   'source': {'type': 'string', 'enum': ['request', 'copilot']},
                   'reason': {'type': 'string'}}, 'required': ['key', 'value']}},
    'formats': {'type': 'array', 'items': {'type': 'string'}, 'description': 'kind export: the format ids'},
    'changes': {'type': 'object', 'description': 'kind edit: {factor path: value}, as edit_project takes it'},
    'ops': {'type': 'array', 'items': {'type': 'object'}, 'description': 'kind edit: structural operations'}},
    'required': ['kind']}

LOCAL = {t.name: t for t in (
    LocalTool('propose_changes', 'Propose a change to the project as named factors and their new values (sensor:<id>.x '
              '| y | z in metres, .roll | pitch | yaw in degrees, .dx | dy | dz shifts, .model, .enabled; '
              'setting:<study>.<key>). Nothing changes until the user accepts it; locked sensors are refused.',
              {'type': 'object', 'properties': {
                  'title': {'type': 'string', 'description': 'What the change does, in a few words'},
                  'reason': {'type': 'string', 'description': 'Why, with the figures that support it'},
                  'changes': {'type': 'array', 'items': {'type': 'object', 'properties': {
                      'factor': {'type': 'string'}, 'value': {}}, 'required': ['factor', 'value']}},
                  'ops': {'type': 'array', 'items': {'type': 'object'}, 'description': 'structural operations '
                          '(add_sensor, remove_sensor, add_variant, ...), as edit_project takes them'}},
               'required': ['title']}, _propose_changes, True, 'Propose a change'),
    LocalTool('propose_plan', 'Put a Copilot plan on screen: the long studies and exports to run, in order. Nothing '
              'runs until the user approves it on the plan card.',
              {'type': 'object', 'properties': {
                  'title': {'type': 'string'}, 'rationale': {'type': 'string'},
                  'steps': {'type': 'array', 'items': _STEP},
                  'revise': {'type': 'boolean', 'description': 'A follow-up: revise the plan on screen'}},
               'required': ['title', 'steps']}, _propose_plan, False, 'Propose a plan'),
    LocalTool('copilot_catalogue', 'What a Copilot plan can contain: every study (whether it is long, what it needs, '
              'its stored settings) and every export format.', {'type': 'object', 'properties': {}}, _catalogue,
              False, 'Read the plan catalogue'),
    LocalTool('remember', 'Offer a memory note for this project (a decision, a constraint, a rejected idea or a note); '
              'it is kept only if the user accepts it.',
              {'type': 'object', 'properties': {
                  'kind': {'type': 'string', 'enum': list(MM.KINDS)}, 'text': {'type': 'string'}},
               'required': ['kind', 'text']}, _remember, False, 'Offer a memory note'),
)}
FALLBACK = {'get_page_context': LocalTool(
    'get_page_context', 'The page the user is on: its headline figure, status word and the study\'s figures.',
    {'type': 'object', 'properties': {}}, _page_context, False, 'Read the page on screen')}
OVERRIDE = ('propose_plan', 'propose_changes')      # the dock's versions win over tools.py's of the same name


def _their() -> dict:
    try:
        from . import tools as T
    except ImportError:
        return {}
    return dict(getattr(T, 'TOOLS', {}) or {})


def all_tools() -> dict:
    out = dict(FALLBACK)
    out.update({k: v for k, v in _their().items() if k not in OVERRIDE})
    out.update(LOCAL)
    return out


def describe(name) -> dict:
    t = all_tools().get(name)
    return {'label': getattr(t, 'label', '') or getattr(t, 'title', '') or name.replace('_', ' ').capitalize(),
            'changes': bool(getattr(t, 'changes', False))}


def schema(t) -> dict:
    s = getattr(t, 'input_schema', None) or {'type': 'object', 'properties': {}}
    return s if isinstance(s, dict) else {'type': 'object', 'properties': {}}


_TYPES = {'string': str, 'boolean': bool, 'array': list, 'object': dict}


def validate_args(name, args) -> list:
    """Problems with a tool input (an unknown tool, a missing required key, a top-level value of the wrong type)."""
    t = all_tools().get(name)
    if t is None:
        return [f'there is no tool {name}']
    if not isinstance(t, LocalTool):
        from . import tools as T
        return list(T.validate_args(name, args))
    s = schema(t)
    out = [f'{k} is required' for k in s.get('required', ()) if k not in args]
    for k, v in args.items():
        spec = (s.get('properties') or {}).get(k)
        if not isinstance(spec, dict) or 'type' not in spec:
            continue
        want = spec['type']
        if want in ('number', 'integer'):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                out.append(f'{k} must be a number')
        elif want in _TYPES and not isinstance(v, _TYPES[want]):
            out.append(f'{k} must be a{"n" if want[0] in "aeiou" else ""} {want}')
        if 'enum' in spec and v not in spec['enum']:
            out.append(f'{k} must be one of {", ".join(map(str, spec["enum"]))}')
    return out


def claude_defs() -> list:
    """The tools as Claude takes them, in a fixed order (sorted by name: the cached prefix stays the same)."""
    return [{'name': n, 'description': getattr(t, 'description', ''), 'input_schema': schema(t),
             'eager_input_streaming': True} for n, t in sorted(all_tools().items())]


def openai_schemas() -> list:
    return [{'type': 'function', 'function': {'name': n, 'description': getattr(t, 'description', ''),
                                              'parameters': schema(t)}} for n, t in sorted(all_tools().items())]


def _dump(o) -> str:
    try:
        from .figures import compact
        o = compact(o, max_depth=6, items=40, text=2000)
    except Exception:                       # noqa: BLE001 — compact is a nicety
        pass
    s = json.dumps(o, default=str, ensure_ascii=False)
    return s if len(s) <= RESULT_MAX else s[:RESULT_MAX] + '…(cut)'


STUDY_ARGS = ('study', 'study_id', 'sid', 'id')
DOCK_REFUSES = {'export_files': 'In the app, exports run from a plan the user approves: propose_plan with an export '
                                'step (the files go into a new folder of the working folder).'}


def long_study_refusal(name, args) -> str:
    """'' when the call may run; else why not (run_study of a long study outside an approved plan)."""
    if name not in ('run_study', 'run_batch'):
        return ''
    sid = next((str(args[k]) for k in STUDY_ARGS if isinstance(args.get(k), str)), '')
    if sid and CP.is_long(sid):
        return (f'{CP.studies()[sid].title} is a long study: it runs only from a plan the user approves. Call '
                'copilot_catalogue and propose_plan with it as a step; the user approves it on the plan card.')
    return ''


def run(tc: ToolContext, name: str, args: dict) -> tuple[str, bool]:
    """(the result as JSON text, ok). Errors are plain sentences in {'error': ...}, never a traceback."""
    from ..core.jobs import StudyError
    t = all_tools().get(name)
    if t is None:
        return _dump({'error': f'There is no tool {name}.'}), False
    why = long_study_refusal(name, args)
    if why:
        return _dump({'error': why, 'plan_required': True}), False
    if name in DOCK_REFUSES:
        return _dump({'error': DOCK_REFUSES[name]}), False
    n_props = len(tc.proposals)
    predicted = None
    try:
        if isinstance(t, LocalTool):
            res = t.fn(tc, **args)
        else:
            from . import tools as T
            if name == 'edit_project' and 'simulate_changes' in T.TOOLS:
                predicted = T.call(tc.session, 'simulate_changes', args)     # what it would do (it is not made)
                if isinstance(predicted, dict) and predicted.get('error'):
                    return _dump(predicted), False
            res = T.call(tc.session, name, args)
    except StudyError as e:                 # FactorError, SessionError-like plain sentences
        return _dump({'error': str(e)}), False
    except TypeError as e:
        return _dump({'error': f'The input does not fit the tool: {e}.'}), False
    except Exception as e:                  # noqa: BLE001 — the model gets a sentence; the log the traceback
        if type(e).__name__ in ('SessionError', 'NeedsWindow'):
            return _dump({'error': str(e)}), False
        import traceback
        traceback.print_exc()
        return _dump({'error': f'{name} stopped with an unexpected error ({type(e).__name__}).'}), False
    made = tc.proposals[n_props:]
    if not isinstance(t, LocalTool) and (made or predicted is not None) and not (isinstance(res, dict)
                                                                                  and res.get('error')):
        res = {'applied': False, 'proposals': [p.for_model() for p in made],
               'note': ('The change was not made: it is shown to the user as a proposal to accept or discard.'
                        if made else 'Nothing was proposed (nothing would change).'),
               **({'predicted': predicted} if predicted is not None else {'tool_result': res})}
    ok = not (isinstance(res, dict) and res.get('error'))
    return _dump(res), ok
