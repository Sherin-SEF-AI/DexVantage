"""Where the assistant's and Claude's tool calls work: a project file (FileSession, headless) or the open window
(StoreSession, through the window's ProjectStore). The tools (assistant/tools.py) see one interface:

    mode                      'file' | 'live'
    allow_long                the long studies may run from a tool call (False: they need a plan the user approves)
    project()                 a snapshot of the project as it is now (the tools never change it in place)
    cache                     the study result cache (core.jobs.ResultCache) the figures come through
    last_result(sid)          the result the app already holds for the design as it is now, else None
    edit(keys, fn, text)      fn(project) as ONE undo step over the named sections; text is the step's label
    undo() / redo()           the assistant's own last step only; a plain sentence either way
    where() / unsaved()       what is being worked on; whether there are changes not saved
    page_context()            the page the user is on (assistant/page_guide.py), or None without a window
    show(req)                 something to show in the window (StoreSession only; FileSession raises NeedsWindow)
    propose(plan)             a Copilot plan for the user to approve (kept; the window shows it)
    save(path=None)           'saved …' or a reason

Ported from Vantage mcp/session.py (FileSession: a backup of the file as it was first opened, the file saved after
each change, a change on disk since it was opened refused) and ui/mcp_bridge.py (LiveSession: work handed to the UI
thread, "Claude · …" undo steps, undo only of the assistant's own step). Undo steps hold the sections they touch
before and after, as ui/store.py's EditCommand does.

Known limits: a FileSession's undo history lives as long as the session (the saved file keeps only the result, and
the backup the file as first opened); in the window, the assistant undoes only its own last step and never a step
the user made after it; studies run on the project as stored (the window's view state, such as the joint sliders'
pose, is not part of it)."""
from __future__ import annotations

import copy
import os
import shutil
import threading

from ..core import jobs

CLAUDE = 'Claude ·'                 # the label every assistant edit starts with (Edit ▸ Undo shows it)
EXTENSION = '.dexv'


class NeedsWindow(Exception):
    """A tool that shows its result in the window, called without one."""


class SessionError(Exception):
    """A plain sentence: why the session cannot do what was asked (nothing was changed)."""


def _label(text: str) -> str:
    text = ' '.join(str(text or 'change').split())
    return text if text.startswith(CLAUDE) else f'{CLAUDE} {text}'


class Session:
    """The interface (module docstring); FileSession and StoreSession implement it."""
    mode = 'file'
    allow_long = True

    def __init__(self):
        self.plans: list = []
        self.cache = jobs.ResultCache()

    def last_result(self, sid: str):
        return None

    def page_context(self):
        return None

    def show(self, req: dict) -> dict:
        raise NeedsWindow('This shows in the DexVantage window, and none is open.')

    def propose(self, plan: dict) -> dict:
        self.plans.append(plan)
        del self.plans[:-20]
        return {'shown': False, 'note': 'No DexVantage window is open: the plan is kept with this session and runs '
                                        'only after the user approves it in the window (or runs it with --run-plan).'}


# =========================================================================== a file
class FileSession(Session):
    """A .dexv project worked on without the window: edits are undoable steps, and the file is saved after each one
    (the first save copies the file as it was opened to <name>.dexv.bak)."""
    mode = 'file'

    def __init__(self, project=None, path: str | None = None, allow_long: bool = True):
        super().__init__()
        self.allow_long = allow_long
        self.p = None
        self.undo_stack: list = []
        self.redo_stack: list = []
        self.revision = 0
        self._mtime = 0.0
        self._backed_up: set = set()
        self._dirty = False
        self._lock = threading.RLock()
        if path:
            r = self.open(path=path)
            if 'error' in r:
                raise SessionError(r['error'])
        elif project is not None:
            self.p = project

    # ---------------------------------------------------------------- opening
    def open(self, path: str | None = None, sample: str | None = None) -> dict:
        from ..core.project import Project, ProjectFormatError
        if path:
            path = os.path.abspath(os.path.expanduser(str(path)))
            folder = os.path.isdir(path) and path.rstrip(os.sep).lower().endswith(EXTENSION + '.d')   # unpacked form
            if not os.path.exists(path):
                return {'error': f'There is no file at {os.path.basename(path) or path}.'}
            if not folder and (not path.lower().endswith(EXTENSION) or not os.path.isfile(path)):
                return {'error': f'{os.path.basename(path) or path} is not a DexVantage project file ({EXTENSION}, '
                                 f'or a {EXTENSION}.d folder).'}
            try:
                p = Project.load(path)
            except ProjectFormatError as e:
                return {'error': str(e)}
            except OSError as e:
                return {'error': f'Could not read {os.path.basename(path)}: {e.strerror or e}.'}
            self._mtime = os.path.getmtime(path)
        elif sample:
            from ..core.robot.samples import sample_project
            try:
                p = sample_project(str(sample))
            except ProjectFormatError as e:
                return {'error': str(e)}
            self._mtime = 0.0
        else:
            from ..core.robot.samples import SAMPLES
            return {'error': f'Give path (a {EXTENSION} file) or sample ({", ".join(SAMPLES)}).'}
        with self._lock:
            self.p = p
            self.undo_stack, self.redo_stack, self._dirty = [], [], False
            self.revision += 1
        return {'opened': os.path.basename(p.path) if p.path else f'{p.name} (not saved: save_project with a path '
                                                                  'keeps it)', 'project': p.name}

    def _need(self):
        if self.p is None:
            raise SessionError('No project is open: start the server with --project FILE.dexv, open the DexVantage '
                               'window, or call open_project with a path or a sample.')

    # ---------------------------------------------------------------- reading
    def project(self):
        self._need()
        with self._lock:
            return self.p.snapshot()

    def where(self) -> str:
        if self.p is None:
            return 'no project'
        return f'the project file {os.path.basename(self.p.path)}' if self.p.path else 'a project in memory (not saved)'

    def unsaved(self) -> bool:
        return self._dirty

    # ---------------------------------------------------------------- changing
    def edit(self, keys, fn, text: str):
        self._need()
        with self._lock:
            self._check_disk()
            p = self.p
            keys = [k if k != 'variant' else f'variant:{p.active_uid}' for k in keys]
            before = {k: copy.deepcopy(p.section(k)) for k in keys}
            try:
                fn(p)
            except Exception:
                for k, st in before.items():
                    p.restore(k, st)
                raise
            after = {k: copy.deepcopy(p.section(k)) for k in keys}
            label = _label(text)
            self.undo_stack.append((label, keys, before, after))
            self.redo_stack.clear()
            p.add_log('edit', label)
            self._changed()
            return label

    def _changed(self):
        self.revision += 1
        self._dirty = True
        if self.p.path:
            self._write(self.p.path)

    def _check_disk(self):
        path = self.p.path
        if path and os.path.exists(path) and self._mtime and abs(os.path.getmtime(path) - self._mtime) > 1e-3:
            raise SessionError(f'{os.path.basename(path)} changed on disk since it was opened here: open it again '
                               'before changing it (nothing was changed).')

    def _write(self, path):
        if os.path.exists(path) and path not in self._backed_up:
            shutil.copy2(path, path + '.bak')
            self._backed_up.add(path)
        self.p.save(path)
        self._mtime = os.path.getmtime(path)
        self._dirty = False

    def _step(self, redo: bool) -> str:
        self._need()
        with self._lock:
            src, dst = (self.redo_stack, self.undo_stack) if redo else (self.undo_stack, self.redo_stack)
            if not src:
                return 'Nothing to ' + ('redo.' if redo else 'undo.')
            self._check_disk()
            label, keys, before, after = src.pop()
            for k in keys:
                self.p.restore(k, copy.deepcopy(after[k] if redo else before[k]))
            dst.append((label, keys, before, after))
            self._changed()
            return f'{"Redid" if redo else "Undid"}: {label}'

    def undo(self) -> str:
        return self._step(False)

    def redo(self) -> str:
        return self._step(True)

    def save(self, path: str | None = None) -> str:
        self._need()
        with self._lock:
            if path:
                path = os.path.abspath(os.path.expanduser(str(path)))
                if not path.lower().endswith(EXTENSION):
                    path += EXTENSION
                self.p.path = path
                self._mtime = os.path.getmtime(path) if os.path.exists(path) else 0.0
            if not self.p.path:
                return 'Not saved: the project has no file yet; give path.'
            try:
                self._write(self.p.path)
            except OSError as e:
                return f'Not saved: could not write {os.path.basename(self.p.path)} ({e.strerror or e}).'
            return f'Saved {os.path.basename(self.p.path)}.'

    def history(self, limit: int = 20) -> list:
        return [{'text': t, 'state': 'done'} for t, *_ in self.undo_stack[-limit:]]


# =========================================================================== the window
class _Invoker:
    """Runs a function on the UI thread for a worker thread and hands back its result (or re-raises its error)."""

    def __init__(self):
        from PyQt6.QtCore import QObject, pyqtSignal

        class _Q(QObject):
            job = pyqtSignal(object)
        self._q = _Q()
        self._q.job.connect(self._run)

    @staticmethod
    def _run(item):
        fn, box, done = item
        try:
            box['value'] = fn()
        except BaseException as e:               # noqa: BLE001 — re-raised in the worker
            box['error'] = e
        finally:
            done.set()

    def __call__(self, fn, timeout: float = 120.0):
        if threading.current_thread() is threading.main_thread():
            return fn()
        box, done = {}, threading.Event()
        self._q.job.emit((fn, box, done))
        if not done.wait(timeout):
            raise SessionError('The DexVantage window did not answer in time.')
        if 'error' in box:
            raise box['error']
        return box.get('value')


class StoreSession(Session):
    """The window's project (ui/context.AppContext: its store, analysis controller and window). Safe to call from a
    worker thread: whatever touches the store or the window runs on the UI thread. Each edit is one undo step
    "Claude · …" in Edit ▸ Undo, like the user's own."""
    mode = 'live'

    def __init__(self, ctx, allow_long: bool = False):
        super().__init__()
        self.ctx = ctx
        self.allow_long = allow_long
        self.ui = _Invoker()
        self.cache = ctx.analysis.cache                      # the window's own cache: what its pages computed

    def project(self):
        return self.ui(lambda: self.ctx.store.project.snapshot())

    def last_result(self, sid: str):
        def get():
            doc = self.ctx.analysis.results.get(sid)
            if doc is None or doc.get('revision', -1) != self.ctx.store.revision:
                return None
            return doc
        return self.ui(get)

    def where(self) -> str:
        def get():
            p = self.ctx.store.project
            return (f'the DexVantage window (process {os.getpid()}): '
                    f'{os.path.basename(p.path) if p.path else "a project not saved yet"}')
        return self.ui(get)

    def unsaved(self) -> bool:
        return self.ui(lambda: not self.ctx.store.undo.isClean())

    def edit(self, keys, fn, text: str):
        label = _label(text)

        def run():
            st = self.ctx.store
            st.edit(list(keys), fn, label)
            w = self.ctx.window
            if w is not None:
                w.msg(label, 5000)
            return label
        return self.ui(run)

    def _step(self, redo: bool) -> str:
        u = self.ctx.store.undo
        i = u.index()
        j = i if redo else i - 1
        if not 0 <= j < u.count():
            return 'Nothing to ' + ('redo.' if redo else 'undo.')
        text = u.command(j).text()
        if not text.startswith(CLAUDE):
            return f'The {"next" if redo else "last"} step is the user\'s ({text}): it was left as it is.'
        (u.redo if redo else u.undo)()
        if self.ctx.window is not None:
            self.ctx.window.msg(f'{"Redid" if redo else "Undid"}: {text}', 5000)
        return f'{"Redid" if redo else "Undid"}: {text}'

    def undo(self) -> str:
        return self.ui(lambda: self._step(False))

    def redo(self) -> str:
        return self.ui(lambda: self._step(True))

    def save(self, path: str | None = None) -> str:
        def run():
            w, p = self.ctx.window, self.ctx.store.project
            if w is None:
                return 'Not saved: there is no window.'
            if path:
                q = os.path.abspath(os.path.expanduser(str(path)))
                if not q.lower().endswith(EXTENSION):
                    q += EXTENSION
                ok = w._write(q) if hasattr(w, '_write') else False
                return f'Saved {os.path.basename(q)}.' if ok else 'Not saved (the window reported a problem).'
            if not p.path:
                return 'Not saved: this project has no file yet; give path (or save it in the window).'
            return f'Saved {os.path.basename(p.path)}.' if w.save() else 'Not saved (the window reported a problem).'
        return self.ui(run)

    def history(self, limit: int = 20) -> list:
        def run():
            u = self.ctx.store.undo
            i = u.index()
            out = []
            for j in range(max(0, u.count() - limit), u.count()):
                t = u.command(j).text()
                out.append({'text': t, 'by': 'claude' if t.startswith(CLAUDE) else 'user',
                            'state': 'done' if j < i else 'undone'})
            return out
        return self.ui(run)

    def page_context(self):
        from . import page_guide as PG
        return self.ui(lambda: PG.current_page_state(self.ctx.window))

    def show(self, req: dict) -> dict:
        w = self.ctx.window
        if w is None:
            raise NeedsWindow('This shows in the DexVantage window, and none is open.')
        return self.ui(lambda: _show(w, req))

    def propose(self, plan: dict) -> dict:
        self.plans.append(plan)
        del self.plans[:-20]

        def run():
            hook = getattr(self.ctx, 'on_copilot_plan', None)       # the Assistant dock shows it (approve / reject)
            w = self.ctx.window
            if callable(hook):
                hook(plan)
                return {'shown': True, 'note': 'The plan is on screen for the user to approve; nothing runs before.'}
            if w is not None:
                w.msg(f'Claude proposed a plan: {plan.get("title", "")} (it runs only after you approve it)', 8000)
            return {'shown': False, 'note': 'The plan is kept for the user to approve; nothing runs before.'}
        return self.ui(run)


def _show(w, req: dict) -> dict:
    """Do what show_in_app asked, on the UI thread (the request is already checked by the tool)."""
    act = req['action']
    st = w.store
    if act == 'open_page':
        if req['page'] not in getattr(w, 'pages', {}):
            return {'error': f'There is no page {req["page"]}.'}
        w.open_page(req['page'])
    elif act == 'select':
        st.select(tuple(req['what']))
    elif act == 'frame_all':
        w.frame_all()
    elif act == 'message':
        w.msg(str(req['text'])[:300], 10000)
    return {'done': act}

