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
    pose_changed = pyqtSignal(str)         # robot uid whose joint state q changed (view state, not undoable)

    def __init__(self, project: Project):
        super().__init__()
        self.project = project
        self.undo = QUndoStack(self)
        self.undo.cleanChanged.connect(self._clean_changed)   # a bound slot: dropped with the store
        self.sel: tuple = ()               # ('robot', uid) ('sensor', uid) ('volume', uid) ('object', uid) ('project',)
        self.gesture = 0                   # bumped when a drag starts
        self.revision = 0                  # bumped by every change to the project (edits, undo, redo, replace)
        self._dragging = False
        # view state, not project content: each robot's current joint values (radians / metres), in its model's
        # joint_names order; reset to the robot's home when the project is replaced or the robot is added
        self.q: dict = {}
        self._sync_q()

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
        self.q = {}
        self._sync_q()
        self.project_replaced.emit()
        self.selection_changed.emit()

    @property
    def variant(self) -> Variant:
        return self.project.variant

    def select(self, obj: tuple):
        self.sel = tuple(obj)
        self.selection_changed.emit()

    def selected(self):
        """The selected object (a Robot, Sensor, Volume or TaskObject, or a trajectory's dict), or None."""
        if len(self.sel) < 2:
            return None
        kind, uid = self.sel[0], self.sel[1]
        p = self.project
        if kind == 'trajectory':
            return next((t for t in p.tasks.trajectories if isinstance(t, dict) and t.get('uid') == uid), None)
        pool = {'robot': p.robots, 'sensor': p.variant.sensors, 'volume': p.workspace.volumes,
                'object': p.workspace.objects}.get(kind, [])
        return next((o for o in pool if o.uid == uid), None)

    # ------------------------------------------------------------------ edits
    def edit(self, keys, fn, text: str, merge_key=None, sections=None):
        """Run fn(project) as one undoable step covering the given section keys ('variant' = the active variant)."""
        if isinstance(keys, str):
            keys = [keys]
        keys = [k if k != 'variant' else f'variant:{self.project.active_uid}' for k in keys]
        from ..core.project import FRESH_SECTIONS

        def state(k):              # section() of a FRESH section is already an independent deep copy
            sec = self.project.section(k)
            return sec if k in FRESH_SECTIONS else copy.deepcopy(sec)
        before = {k: state(k) for k in keys}
        result = fn(self.project)
        self.revision += 1
        after = {k: state(k) for k in keys}
        self.undo.push(EditCommand(self, keys, before, after, text, merge_key))
        self.project.add_log('edit', text)
        self._sync_q()
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
        self._sync_q()
        self.changed.emit(self._sections(keys))
        self.selection_changed.emit()

    # ------------------------------------------------------------------ joint state (view state)
    def _sync_q(self):
        """Give every robot a joint vector (its home) and drop those of removed robots; keep the rest."""
        uids = {r.uid for r in self.project.robots}
        for k in [k for k in self.q if k not in uids]:
            del self.q[k]
        if not self.project.robots:
            return
        from ..core.robot.model import RobotModelError, get_model
        for r in self.project.robots:
            try:
                m = get_model(self.project, r.uid)
            except RobotModelError:
                self.q.pop(r.uid, None)
                continue
            cur = self.q.get(r.uid)
            self.q[r.uid] = m.home.copy() if cur is None or len(cur) != m.nq else m.clip(cur)

    def set_q(self, uid: str, q):
        """Set a robot's joint values, clipped to its joint limits; emits pose_changed(uid) when they change.
        Returns the values kept."""
        import numpy as np

        from ..core.robot.model import get_model
        m = get_model(self.project, uid)
        q = m.clip(q)
        cur = self.q.get(uid)
        if cur is not None and len(cur) == len(q) and np.array_equal(cur, q):
            return cur
        self.q[uid] = q
        self.pose_changed.emit(uid)
        return q

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


# errors whose message is written for the user (a plain sentence that says why and what to do)
EXPLAINED_ERRORS = ('StudyError', 'DepthSetupError', 'LibraryError', 'OcclusionError', 'ProjectFormatError',
                    'RobotModelError', 'KinematicsError', 'TrajectoryError', 'SampleError')


def failure_text(message: str, title: str = 'The study') -> tuple[str, bool]:
    """(the sentence to show, explained) for AnalysisController.failed's message ('Type: text' + traceback): an
    explained error's own sentence, else a plain sentence pointing at the log (the caller logs the traceback)."""
    first = (message or '').splitlines()[0] if message else ''
    kind, _, text = first.partition(': ')
    if kind in EXPLAINED_ERRORS and text.strip():
        return text.strip(), True
    return f'{title} stopped with an unexpected error ({kind or "unknown"}). The details are in the log.', False


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
        self._run_settings: dict = {}      # study id -> settings for its next run (AnalysisController.run)
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

    def run(self, sid: str, settings: dict | None = None) -> bool:
        """Start study sid on a snapshot of the project; False (and one more run queued) when it is running.
        settings: keys merged into the snapshot's project.analysis.studies[sid] for the next run only (view state
        such as the current joint state q, which is not project content); the project itself is not changed."""
        with self._lock:
            if settings is not None:
                self._run_settings[sid] = dict(settings)
            if sid in self._busy:
                self._pending.add(sid)
                return False
            cancel = threading.Event()
            self._busy[sid] = cancel
            extra = self._run_settings.pop(sid, None)
        self.started.emit(sid)
        project = self.store.project.snapshot()
        if extra:
            project.analysis.studies[sid] = {**(project.analysis.studies.get(sid) or {}), **extra}
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
                # every StudyError subclass (DatasetError, PolicyError, SynthError, CalibError, ...) is explained
                kind = type(e).__name__ if type(e).__name__ in EXPLAINED_ERRORS or not isinstance(
                    e, jobs.StudyError) else 'StudyError'
                self.failed.emit(sid, f'{kind}: {e}\n{traceback.format_exc()}')
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
