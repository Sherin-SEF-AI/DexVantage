"""The tools Claude gets over MCP, and running one on a session (the open DexVantage window or a project file).

The same code runs in both places: in the window (ui/mcp_bridge.py, on a worker thread, through
assistant/session.StoreSession) and in the MCP process on a file (assistant/session.FileSession). The assistant's
tools (assistant/tools.py) come along as they are; a few belong to MCP only: dexvantage_status, redo_change,
save_project, get_history, run_batch and open_project (handled by the server's router: it chooses the session).

    listing() -> [MCP tool definitions]         name, title, description, inputSchema, annotations
    call(session, name, args) -> MCP result     {'content': [{'type': 'text', 'text': JSON}], 'structuredContent':
                                                 {...}, 'isError': bool}
    INSTRUCTIONS                                the server's instructions text (initialize)

Ported from Vantage mcp/service.py (the listing with annotations, whole-JSON results, one edit at a time); the tools
are DexVantage's.

Known limits: results are whole JSON text (MCP clients cut long results themselves); images are not returned.
"""
from __future__ import annotations

import json
import os
import threading

from ..assistant import tools as T
from ..core.robot.samples import SAMPLES as _SAMPLES

INSTRUCTIONS = (
    'DexVantage plans the cameras, depth sensors, tactile sensors and teleop devices of a robot-learning data '
    'collection rig, and checks recorded datasets against the plan. Units: metres, degrees (joint values in radians), '
    'seconds, percent; world = the primary robot\'s base, REP-103 (x forward, y left, z up); a sensor mount is a pose '
    'in its parent frame (world, cage, tripod, table or a robot link), rpy in degrees. Start with get_overview, then '
    'read before you change anything (get_robot, get_sensor, the get_* studies). Every figure comes from a study or a '
    'cited datasheet: quote them, never estimate one; None means "not stated". edit_project changes the project as '
    'one undo step and returns the requirement metrics before and after; simulate_changes tries a change without '
    'making it; undo_last_change takes back your own last change. Locked sensors and robots are the user\'s: leave '
    'them. Long studies (grasp visibility, teleop, hand-eye, ground truth, dataset audits) take tens of seconds. '
    'In the DexVantage window every change shows at once and the user can undo it; on a project file each change '
    'is saved, with a backup of the file as it was first opened.')

_LOCK = threading.Lock()


class McpTool:
    def __init__(self, name, title, description, schema, fn, read_only=True, destructive=False):
        self.name, self.title, self.description, self.schema, self.fn = name, title, description, schema, fn
        self.read_only, self.destructive = read_only, destructive

    def listing(self) -> dict:
        return {'name': self.name, 'title': self.title, 'description': self.description, 'inputSchema': self.schema,
                'annotations': {'title': self.title, 'readOnlyHint': self.read_only,
                                'destructiveHint': self.destructive, 'idempotentHint': self.read_only,
                                'openWorldHint': False}}


def _obj(props: dict, required=()) -> dict:
    s = {'type': 'object', 'properties': props, 'additionalProperties': False}
    if required:
        s['required'] = list(required)
    return s


# =========================================================================== MCP-only tools
def _status(s, **_):
    out = {'mode': s.mode, 'where': s.where(), 'unsaved_changes': s.unsaved(), 'long_studies_allowed': s.allow_long,
           'instructions': INSTRUCTIONS}
    try:
        p = s.project()
        out.update(project=p.name, variant=p.variant.name, robots=[r.name for r in p.robots],
                   sensors=[x.name for x in p.variant.sensors])
    except Exception as e:                               # noqa: BLE001 — no project yet: say so
        out['project'] = None
        out['note'] = str(e)
    return out


def _redo(s, **_):
    return {'result': s.redo()}


def _save(s, path=None, **_):
    return {'result': s.save(path)}


def _history(s, limit=20, **_):
    h = s.history(max(1, min(int(limit or 20), 100))) if hasattr(s, 'history') else []
    return {'where': s.where(), 'steps': h, 'unsaved_changes': s.unsaved()}


def _batch(s, args=None, help=False, timeout_s=600, **_):
    from .. import workdir
    from . import batchrun as B
    if help or not args:
        return B.help_doc()
    why = B.check(args)
    if why:
        return {'error': why}
    p = s.project()
    here = os.path.dirname(p.path) if p.path else ''
    base = here if here and os.access(here, os.W_OK) else workdir.project_dir(p, None, 'runs')
    os.makedirs(base, exist_ok=True)
    return B.run(p, args, base, timeout_s)


def _open_doc(s, **_):                                  # the router handles open_project; this is its listing
    return {'error': 'open_project is handled by the server.'}


MCP_TOOLS = [
    McpTool('dexvantage_status', 'DexVantage status',
            'Where Claude is working (the open DexVantage window or a project file), the project, its robots and '
            'sensors, unsaved changes, and the conventions. Call it first.', _obj({}), _status),
    McpTool('redo_change', 'Redo', 'Redo the change undo_last_change took back (your own only).', _obj({}), _redo,
            read_only=False),
    McpTool('save_project', 'Save the project',
            'Save the project. On a file, changes are saved as they are made; path saves it to a new .dexv file and '
            'works on that from then on. In the window: File ▸ Save.',
            _obj({'path': {'type': 'string', 'description': 'a .dexv path (optional)'}}), _save, read_only=False,
            destructive=True),
    McpTool('get_history', 'What changed lately',
            'The latest undo steps (in the window: who made each, the user or Claude), and unsaved changes.',
            _obj({'limit': {'type': 'integer', 'description': 'steps (default 20)'}}), _history),
    McpTool('run_batch', 'Run a batch job',
            'Run DexVantage batch jobs on the project as it is now (a copy; the project is not changed): any study '
            'flag (--coverage, --grasp, --teleop, --bandwidth …), --analyse, --synth-episodes, --audit with '
            '--dataset, --validate, --reproduce. In a run folder of its own; returns the exit code and its meaning, '
            'the output\'s end, the files written and the small JSON ones. help: true lists the options.',
            _obj({'args': {'type': 'array', 'items': {'type': 'string'},
                           'description': 'the command-line words after the project'},
                  'help': {'type': 'boolean'},
                  'timeout_s': {'type': 'integer', 'description': 'default 600, at most 3600'}}),
            _batch, read_only=False),
    McpTool('open_project', 'Open a project',
            'Work on another project. With a DexVantage window open, Claude works on the project in it; path opens a '
            '.dexv file and sample a bundled sample (so100, ur5e, dexhand, aloha, panda-bimanual, g1) to work on here, '
            'even with a window open; with no arguments Claude goes back to the window.',
            _obj({'path': {'type': 'string'}, 'sample': {'type': 'string', 'enum': list(_SAMPLES)}}),
            _open_doc, read_only=False),
]


def _assistant_tools() -> list:
    out = []
    for t in T.TOOLS.values():
        desc = t.description
        if t.name in ('show_in_app', 'get_page_context'):
            desc += ' (Needs the DexVantage window open.)'
        out.append(McpTool(t.name, t.title, desc, t.input_schema, None, read_only=not t.changes,
                           destructive=t.name in ('edit_project',)))
    return out


TOOLS = MCP_TOOLS + _assistant_tools()
BY_NAME = {t.name: t for t in TOOLS}


def listing() -> list:
    return [t.listing() for t in TOOLS]


# =========================================================================== results
def result(doc, error=None) -> dict:
    """An MCP tools/call result: the whole JSON as text, and as structuredContent."""
    if not isinstance(doc, dict):
        doc = {'result': doc}
    err = ('error' in doc) if error is None else bool(error)
    return {'content': [{'type': 'text', 'text': json.dumps(doc, ensure_ascii=False, default=str,
                                                            separators=(',', ':'))}],
            'structuredContent': json.loads(json.dumps(doc, default=str)), 'isError': err}


def call(session, name: str, args) -> dict:
    """Run tool name on the session: an MCP result; errors as {'error': 'a plain sentence'} with isError."""
    t = BY_NAME.get(name)
    if t is None:
        return result({'error': f'There is no tool {name!r}.'})
    args = {k: v for k, v in (args or {}).items() if v is not None} if isinstance(args, dict) else args
    if t.fn is None:                                     # an assistant tool: its own check, errors and lock
        with _LOCK if not t.read_only else _NoLock():
            return result(T.call(session, name, args))
    problems: list = []
    T._check(t.schema, args, '', problems)               # noqa: SLF001
    if problems:
        return result({'error': 'The input does not match the tool: ' + '; '.join(problems[:6]) + '.'})
    try:
        with _LOCK if not t.read_only else _NoLock():
            return result(t.fn(session, **args))
    except Exception as e:                               # noqa: BLE001 — a tool never takes the server down
        import sys
        import traceback
        traceback.print_exc(file=sys.stderr)
        from ..assistant.session import SessionError
        if isinstance(e, SessionError):
            return result({'error': str(e)})
        return result({'error': f'{t.title} stopped with an unexpected error ({type(e).__name__}). The details are '
                                'in the server\'s log (stderr).'})


class _NoLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
