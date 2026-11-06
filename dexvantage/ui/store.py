"""Project store (undoable edits, selection, signals) and the background analysis controller.

Ported from Vantage ui/store.py. The undo pattern is the same: an edit names the project sections it touches, the store
keeps a deep copy of each before and after, and one QUndoCommand restores them; drags merge into one step per gesture.
The controller runs registered studies (core/jobs.py) on a worker thread with progress, cancel and partial results,
debounced after edits and cached by content digest.
"""
from __future__ import annotations

import copy
import threading
import time
import traceback

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtGui import QUndoCommand, QUndoStack

from ..core import jobs
from ..core.project import Project, Sensor, Variant  # noqa: F401  (re-exported for the UI)

_MERGE_ID = 4242


class EditCommand(QUndoCommand):
    def __init__(self, store, keys, before, after, text, merge_key=None):
        super().__init__(text)
        self.store = store
        self.keys = keys
        self.before = before
        self.after = after
        self.merge_key = merge_key
        self.first = True
        self.t = time.time()
        self.gesture = store.gesture

    def id(self):
        return _MERGE_ID if self.merge_key else -1

    def mergeWith(self, other):
        if other.merge_key is None or other.merge_key != self.merge_key or other.keys != self.keys:
            return False
        if other.gesture != self.gesture:      # a new drag is a new undo step, however soon it follows
            return False
        if other.t - self.t > 2.5:
            return False
        self.after = other.after
        self.t = other.t
        self.setText(other.text())
        return True

    def redo(self):
        if self.first:
            self.first = False
            return
        self.store._restore(self.keys, self.after)

    def undo(self):
        self.store._restore(self.keys, self.before)


class ProjectStore(QObject):
    changed = pyqtSignal(set)              # sections changed
    project_replaced = pyqtSignal()
    selection_changed = pyqtSignal()
    status = pyqtSignal(str, int)          # transient message, ms
    dirty_changed = pyqtSignal(bool)

    def __init__(self, project: Project):
        super().__init__()
        self.project = project
        self.undo = QUndoStack(self)
        self.undo.cleanChanged.connect(self._clean_changed)   # a bound slot: dropped with the store
        self.sel: tuple = ()               # ('robot', uid) ('sensor', uid) ('volume', uid) ('object', uid) ('project',)
        self.gesture = 0                   # bumped when a drag starts
        self.revision = 0                  # bumped by every change to the project (edits, undo, redo, replace)
        self._dragging = False

    def _clean_changed(self, clean):
        self.dirty_changed.emit(not clean)

    @property
    def dragging(self):
        return self._dragging

    @dragging.setter
    def dragging(self, on):
        if on and not self._dragging:
            self.gesture += 1
        self._dragging = bool(on)

    # ------------------------------------------------------------------ project
    def set_project(self, project: Project):
        self.revision += 1
        self.project = project
        self.undo.clear()
        self.sel = ()
        self.project_replaced.emit()
        self.selection_changed.emit()

    @property
    def variant(self) -> Variant:
        return self.project.variant

    def select(self, obj: tuple):
        self.sel = tuple(obj)
        self.selection_changed.emit()

    def selected(self):
        """The selected object (a Robot, Sensor, Volume or TaskObject), or None."""
        if len(self.sel) < 2:
            return None
        kind, uid = self.sel[0], self.sel[1]
        p = self.project
        pool = {'robot': p.robots, 'sensor': p.variant.sensors, 'volume': p.workspace.volumes,
                'object': p.workspace.objects}.get(kind, [])
        return next((o for o in pool if o.uid == uid), None)

    # ------------------------------------------------------------------ edits
    def edit(self, keys, fn, text: str, merge_key=None, sections=None):
        """Run fn(project) as one undoable step covering the given section keys ('variant' = the active variant)."""
        if isinstance(keys, str):
            keys = [keys]
        keys = [k if k != 'variant' else f'variant:{self.project.active_uid}' for k in keys]
        before = {k: copy.deepcopy(self.project.section(k)) for k in keys}
        result = fn(self.project)
        self.revision += 1
        after = {k: copy.deepcopy(self.project.section(k)) for k in keys}
        self.undo.push(EditCommand(self, keys, before, after, text, merge_key))
        self.project.add_log('edit', text)
        self.changed.emit(set(sections or self._sections(keys)))
        return result

    def _sections(self, keys):
        return {'variant' if k.startswith('variant:') else k for k in keys}

    def _restore(self, keys, state):
        for k in keys:
            self.project.restore(k, copy.deepcopy(state[k]))
        self.revision += 1
        if self.sel and self.selected() is None and self.sel[0] != 'project':
            self.sel = ()
        self.changed.emit(self._sections(keys))
        self.selection_changed.emit()

    def cancel_gesture(self) -> bool:
        """Undo the edits of the drag in progress and drop them from the history (Esc during a drag)."""
        i = self.undo.index()
        cmd = self.undo.command(i - 1) if i > 0 else None
        if not self._dragging or cmd is None or getattr(cmd, 'gesture', None) != self.gesture or i != self.undo.count():
            return False
        cmd.undo()
        cmd.setObsolete(True)
        self.undo.undo()
        return True


def _closed(e) -> bool:
    """A signal emitted from a worker after its Qt object was deleted (the window closed meanwhile)."""
    return isinstance(e, RuntimeError) and 'has been deleted' in str(e)


class AnalysisController(QObject):
    """Runs studies on worker threads: the quick ones debounced after each edit (live), the long ones on request with
    progress, cancel and partial results. One run per study at a time; a request during a run queues one more."""
    started = pyqtSignal(str)              # study id
    progress = pyqtSignal(str, float, str)
    partial = pyqtSignal(str, object)
    finished = pyqtSignal(str, object)     # study id, result doc
    failed = pyqtSignal(str, str)
    stale_changed = pyqtSignal(bool)
    _rerun = pyqtSignal(str)

    def __init__(self, store: ProjectStore):
        super().__init__()
        self.store = store
        self.cache = jobs.ResultCache()
        self.results: dict = {}
        self.stale = False
        self._busy: dict = {}              # study id -> threading.Event (cancel)
        self._pending: set = set()
        self._lock = threading.Lock()
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._kick_live)
        self.finished.connect(self._on_finished)
        self._rerun.connect(self.run)
        store.changed.connect(lambda _s: self.request())
        store.project_replaced.connect(lambda: self.request(force=True))

    def request(self, force=False):
        if not force and not self.store.project.analysis.live:
            self.set_stale(True)
            return
        if self.store.dragging and not force:
            self.set_stale(True)
            return
        self.timer.start(30)

    def set_stale(self, s):
        if s != self.stale:
            self.stale = s
            self.stale_changed.emit(s)

    def _kick_live(self):
        for sid, info in jobs.STUDIES.items():
            if info.quick:
                self.run(sid)

    def busy(self, sid=None) -> bool:
        with self._lock:
            return bool(self._busy) if sid is None else sid in self._busy

    def run(self, sid: str) -> bool:
        """Start study sid on a snapshot of the project; False (and one more run queued) when it is running."""
        with self._lock:
            if sid in self._busy:
                self._pending.add(sid)
                return False
            cancel = threading.Event()
            self._busy[sid] = cancel
        self.started.emit(sid)
        project = self.store.project.snapshot()
        threading.Thread(target=self._work, args=(sid, project, cancel, self.store.revision), daemon=True).start()
        return True

    def cancel(self, sid=None):
        with self._lock:
            for k, ev in self._busy.items():
                if sid is None or k == sid:
                    ev.set()
            if sid is None:
                self._pending.clear()
            else:
                self._pending.discard(sid)

    def _work(self, sid, project, cancel, revision):
        try:
            ctx = jobs.Ctx(progress=lambda f, t='': self.progress.emit(sid, f, t),
                           partial=lambda d: self.partial.emit(sid, d), cancel=cancel.is_set)
            doc = jobs.run_study(sid, project, ctx, self.cache)
            doc['revision'] = revision
            self.finished.emit(sid, doc)
        except Exception as e:             # noqa: BLE001 — explained in the UI, the traceback logged
            if _closed(e):
                return
            try:
                self.failed.emit(sid, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
            except RuntimeError:
                pass
        finally:
            with self._lock:
                self._busy.pop(sid, None)
                again = sid in self._pending
                self._pending.discard(sid)
            if again:
                try:
                    self._rerun.emit(sid)
                except RuntimeError:
                    pass

    def _on_finished(self, sid, doc):
        self.results[sid] = doc
        self.set_stale(doc.get('revision', self.store.revision) != self.store.revision)

    def wait(self, timeout=10.0) -> bool:
        """Block (processing events) until no study runs; for tests and the self-test."""
        from PyQt6.QtCore import QCoreApplication
        t0 = time.time()
        while time.time() - t0 < timeout:
            QCoreApplication.processEvents()
            if not self.busy() and not self.timer.isActive():
                QCoreApplication.processEvents()
                return True
            time.sleep(0.005)
        return False
