"""Hidden pages wait: a page that is not shown keeps the store's and the analysis controller's signals until it is.

Each edit used to cost 55-60 ms on the UI thread on the two-arm samples because every analysis page rebuilt its
tables and editors on every edit, shown or not (Sensors 16 ms, Policies 11 ms, Canonical view 9 ms). The window now
builds each page through build_page(cls, ctx): the page gets a context whose store and analysis controller are
views of the real ones, identical except for the six signals pages listen to (store.changed, pose_changed,
selection_changed, project_replaced; analysis.finished, failed). A gate relays those at once while the page is
shown (or while the window itself is not shown: scripts and the self-test read pages of an unshown window) and holds
them while it is hidden. When the page is shown (its Show event) the held signals are replayed in their order,
coalesced: the sections of consecutive edits are joined into one changed(...) call, and only the last result (or
failure) of each study, the last pose of each robot and the last selection are kept. The page is then exactly as it
would have been had it followed every signal, without doing the work for states nobody saw.

Anything that reads a hidden page through its hooks sees it up to date: page_context() (the assistant), apply(),
preview(), export() and explain() replay what is held first (StudyPage wraps them). StudyPage.flush_pending() does
the same for any other reader. A page class with DEFER_HIDDEN = False is never held.

Scripts and tests that take a page by name, window.pages[pid] (PageDict), get one that follows every signal from then
on: they hold the page to read it as they go. The window's own code paths use pages.get(pid), `in` and values(),
which leave the page held while hidden.

Known limits: a page built directly (SomePage(ctx), as some tests and plugins do) is not gated; it follows every
signal as before. Signals other than the six (progress, partial, the theme) reach hidden pages at once (they are
cheap). The replay runs on the UI thread when the page is shown, so the first show after many edits costs one
refresh of that page (not one per edit).
"""
from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject, pyqtSignal

STORE_SIGNALS = ('changed', 'pose_changed', 'selection_changed', 'project_replaced')
ANALYSIS_SIGNALS = ('finished', 'failed')


class _View:
    """A stand-in for an object: every attribute is the object's own, except the named replacements."""
    __slots__ = ('_dv_target', '_dv_over')

    def __init__(self, target, over: dict):
        object.__setattr__(self, '_dv_target', target)
        object.__setattr__(self, '_dv_over', over)

    def __getattr__(self, k):
        over = object.__getattribute__(self, '_dv_over')
        if k in over:
            return over[k]
        return getattr(object.__getattribute__(self, '_dv_target'), k)

    def __setattr__(self, k, v):
        setattr(object.__getattribute__(self, '_dv_target'), k, v)

    def __eq__(self, other):
        t = object.__getattribute__(self, '_dv_target')
        return t is other or (isinstance(other, _View) and object.__getattribute__(other, '_dv_target') is t)

    def __hash__(self):
        return hash(id(object.__getattribute__(self, '_dv_target')))

    def __repr__(self):
        return f'<view of {object.__getattribute__(self, "_dv_target")!r}>'


def real(obj):
    """The object behind a view (the object itself when it is not one)."""
    return object.__getattribute__(obj, '_dv_target') if isinstance(obj, _View) else obj


class PageGate(QObject):
    """Relays the store's and the controller's signals to one page, holding them while it is hidden."""
    changed = pyqtSignal(set)
    pose_changed = pyqtSignal(str)
    selection_changed = pyqtSignal()
    project_replaced = pyqtSignal()
    finished = pyqtSignal(str, object)
    failed = pyqtSignal(str, str)

    def __init__(self, ctx):
        super().__init__()
        self.page = None
        self.queue: list = []              # [(key, signal name, args)] in arrival order
        self.held = 0                      # signals held since the gate was made (for tests and measurements)
        self.followed = False              # taken with window.pages[pid]: follows every signal (module docstring)
        self.replayed = 0                  # signals replayed after coalescing
        self._flushing = False
        st, an = ctx.store, ctx.analysis
        st.changed.connect(self._changed)
        st.pose_changed.connect(self._pose)
        st.selection_changed.connect(self._selection)
        st.project_replaced.connect(self._replaced)
        an.finished.connect(self._finished)
        an.failed.connect(self._failed)
        self.store = _View(st, {k: getattr(self, k) for k in STORE_SIGNALS})
        self.analysis = _View(an, {k: getattr(self, k) for k in ANALYSIS_SIGNALS})
        self.ctx = _View(ctx, {'store': self.store, 'analysis': self.analysis})

    # ---------------------------------------------------------------- the page
    def bind(self, page):
        self.page = page
        self.setParent(page)               # the gate (and its connections to the store) goes with the page
        page.installEventFilter(self)
        page._gate = self

    def is_open(self) -> bool:
        p = self.page
        if p is None or self._flushing or self.followed or not getattr(p, 'DEFER_HIDDEN', True):
            return True
        try:
            if p.isVisible():
                return True
            top = p.window()
            return top is p or not top.isVisible()          # an unshown window: follow every signal, as before
        except RuntimeError:                                # the page was deleted
            return True

    def eventFilter(self, obj, ev):
        if obj is self.page and ev.type() == QEvent.Type.Show and self.queue:
            self.flush()
        return False

    # ---------------------------------------------------------------- holding
    def _hold(self, key, name, args):
        if self.is_open():
            getattr(self, name).emit(*args)
            return
        self.held += 1
        q = self.queue
        if name == 'changed':
            if q and q[-1][1] == 'changed':
                q[-1] = (None, 'changed', (set(q[-1][2][0]) | set(args[0]),))
                return
            q.append((None, 'changed', (set(args[0]),)))
            return
        self.queue = q = [e for e in q if e[0] != key]
        q.append((key, name, args))

    def _changed(self, sections):
        self._hold(None, 'changed', (sections,))

    def _pose(self, uid):
        self._hold(('pose', uid), 'pose_changed', (uid,))

    def _selection(self):
        self._hold(('selection',), 'selection_changed', ())

    def _replaced(self):
        self._hold(('replaced',), 'project_replaced', ())

    def _finished(self, sid, doc):
        self._hold(('study', sid), 'finished', (sid, doc))

    def _failed(self, sid, message):
        self._hold(('study', sid), 'failed', (sid, message))

    def follow(self):
        """From now on relay every signal at once, shown or not (and replay what is held)."""
        self.followed = True
        self.flush()

    # ---------------------------------------------------------------- replay
    def flush(self):
        """Replay what is held, in order (consecutive changed(...) joined), now."""
        if not self.queue or self._flushing:
            return
        out: list = []
        for e in self.queue:
            if out and e[1] == 'changed' and out[-1][1] == 'changed':
                out[-1] = (None, 'changed', (out[-1][2][0] | e[2][0],))
            else:
                out.append(e)
        self.queue = []
        self._flushing = True
        try:
            for _key, name, args in out:
                self.replayed += 1
                getattr(self, name).emit(*args)
        finally:
            self._flushing = False


class PageDict(dict):
    """window.pages: pid -> page. Taking a page with pages[pid] makes it follow every signal (a script or a test holds
    it to read it); get(), `in`, values() and items() do not (the window's own paths)."""

    def __getitem__(self, pid):
        page = super().__getitem__(pid)
        gate = getattr(page, '_gate', None)
        if gate is not None:
            gate.follow()
        return page


def build_page(cls, ctx):
    """cls(view of ctx), gated (module docstring); what the window builds its analysis pages with."""
    if not all(hasattr(getattr(ctx, 'store', None), s) for s in STORE_SIGNALS) or not all(
            hasattr(getattr(ctx, 'analysis', None), s) for s in ANALYSIS_SIGNALS):
        return cls(ctx)
    gate = PageGate(ctx)
    try:
        page = cls(gate.ctx)
    except Exception:
        gate.deleteLater()
        raise
    gate.bind(page)
    return page
