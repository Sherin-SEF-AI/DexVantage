"""The page pattern every study page follows (spec §4): what it shows, the number that matters (a large mono figure
with a status word), and the action to improve it, with Apply (undoable), Preview and Explain, plus Run and Export.

Subclass StudyPage, set STUDY and TITLE, build the body in build_body(layout) and fill it in show_result(doc). The
page re-renders whenever the controller finishes its study. Overlays for the 3D view come from overlays(vp) and are
drawn while the page is the Analysis dock's current tab (return [] to draw nothing). page_context() is what the
assistant reads of the page (the headline as shown and what the page is about); a page may add to it.

A page the window builds (pages/_gate.build_page) is held while hidden: its store and study signals wait until it is
shown, then replay once, coalesced, so an edit does not rebuild every hidden page. page_context, apply, preview,
export and explain replay what is held first, in subclasses too; flush_pending() does it for any other reader. Set
DEFER_HIDDEN = False on a page that must follow every signal while hidden.
"""
from __future__ import annotations

import functools
import json
import os

from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

from ..panels import Headline
from ..widgets import button, hline, label


class _Headline(Headline):
    """The page's headline, remembering what it shows (the figure, its unit, the status and its word, the
    explanation) for page_context(): the assistant reads the page as the user sees it."""

    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self.state = {'figure': '—', 'unit': '', 'status': '', 'word': '', 'explain': ''}

    def set(self, figure: str, unit: str, sev: str, word: str, explain: str, tip: str = ''):
        self.state = {'figure': str(figure), 'unit': str(unit), 'status': str(sev), 'word': str(word),
                      'explain': str(explain)}
        super().set(figure, unit, sev, word, explain, tip)


_FLUSH_FIRST = ('page_context', 'apply', 'preview', 'export', 'explain')


def _flushing(fn):
    @functools.wraps(fn)
    def wrapper(self, *a, **kw):
        self.flush_pending()
        return fn(self, *a, **kw)
    wrapper._dv_flushes = True
    return wrapper


class StudyPage(QWidget):
    STUDY: str | None = None
    TITLE = ''
    HEADLINE = ''                      # the headline's title line
    DEFER_HIDDEN = True                # held while hidden (pages/_gate.py) when the window built the page

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        for name in _FLUSH_FIRST:          # a reader of a hidden page sees it up to date
            fn = cls.__dict__.get(name)
            if callable(fn) and not getattr(fn, '_dv_flushes', False):
                setattr(cls, name, _flushing(fn))

    def flush_pending(self):
        """Replay the store and study signals held while the page was hidden (pages/_gate.py), now."""
        gate = getattr(self, '_gate', None)
        if gate is not None:
            gate.flush()

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.doc = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.head = _Headline(self.HEADLINE or self.TITLE)
        self.b_run = button('Run', icon_name='play', tip='Run the study now (long studies run in the background)')
        self.b_apply = button('Apply', role='primary', tip='Make the suggested change (one undo step)')
        self.b_preview = button('Preview', role='ghost', tip='Show the suggested change without making it')
        self.b_explain = button('Explain', role='ghost', icon_name='help', tip='How this figure is computed')
        self.b_export = button('Export…', role='ghost', icon_name='export', tip='The result as JSON with provenance')
        for b in (self.b_run, self.b_apply, self.b_preview, self.b_explain, self.b_export):
            self.head.actions.addWidget(b)
        self.head.actions.addStretch(1)
        self.b_apply.setEnabled(False)
        self.b_preview.setEnabled(False)
        self.b_run.clicked.connect(self.run)
        self.b_apply.clicked.connect(self.apply)
        self.b_preview.clicked.connect(self.preview)
        self.b_explain.clicked.connect(self.explain)
        self.b_export.clicked.connect(lambda: self.export())
        self.note = label('', muted=True, wrap=True)
        outer.addWidget(self.head)
        outer.addWidget(hline())
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(12, 8, 12, 8)
        self.body_layout.setSpacing(6)
        self.build_body(self.body_layout)
        self.body_layout.addWidget(self.note)
        self.body_layout.addStretch(1)
        scroll.setWidget(self.body)
        outer.addWidget(scroll, 1)
        ctx.analysis.finished.connect(self._finished)
        ctx.analysis.failed.connect(self._failed)
        ctx.analysis.progress.connect(self._progress)
        if self.STUDY and self.STUDY in ctx.analysis.results:
            self._finished(self.STUDY, ctx.analysis.results[self.STUDY])
        else:
            self.empty()

    # ---------------------------------------------------------------- for subclasses
    def build_body(self, layout: QVBoxLayout):
        """Add the chart, tables and controls."""

    def show_result(self, doc: dict):
        """Fill the page from the study's result."""

    def empty(self):
        self.head.set('—', '', 'warning', 'not run yet', f'Run {self.TITLE.lower() or "the study"} to see its figures.')

    def apply(self):
        """Make the suggested change through ctx.store.edit (one undo step)."""

    def preview(self):
        """Show the suggested change in the 3D view without making it."""

    def explain(self):
        win = self.ctx.window
        doc = (self.__doc__ or '').strip().split('\n\n')[0]
        if win is not None and doc:
            win.msg(' '.join(doc.split()), 15000)

    def overlays(self, vp) -> list:
        """LineDraw / TriDraw items for the 3D view while this page is shown."""
        return []

    def camview_overlays(self, cam_model, pose) -> list:
        """Camera view overlays while this page is shown: [(polylines, colour, label)], each polyline a (K, 2) array
        in cam_model's pixel coordinates (OpenCV: pixel centres at integers), colour a QColor or '#rrggbb', label a
        short text for the view's legend ('' for none). cam_model is the imager's optics.CameraModel at its stream
        size and pose its optical frame in the world (frames.Pose); core/camrender.project_world and
        visible_polylines turn world points into such polylines. Called once per Camera view image."""
        return []

    def page_context(self) -> dict:
        """What the assistant reads of this page (assistant/page_guide.py, get_page_context): the headline as shown,
        what the page is about (the first paragraph of the class docstring), the note line, whether a suggestion is
        ready to Apply. Pages may extend it with figures they show beyond the headline; keep it small and JSON-able,
        and never put paths, keys or computed-here numbers in it (only what the page shows)."""
        about = ' '.join(((self.__doc__ or '').strip().split('\n\n') or [''])[0].split())
        return {'title': self.TITLE, 'study': self.STUDY, 'headline': dict(self.head.state), 'about': about,
                'note': self.note.text(), 'suggestion_ready': self.b_apply.isEnabled(),
                'has_result': self.doc is not None}

    def suggestion_ready(self, on: bool):
        self.b_apply.setEnabled(on)
        self.b_preview.setEnabled(on)

    # ---------------------------------------------------------------- plumbing
    def run(self):
        if self.STUDY:
            self.ctx.analysis.run(self.STUDY)

    def _progress(self, sid, frac, text):
        if sid == self.STUDY:
            self.note.setText(f'Running · {frac * 100:.0f} %' + (f' · {text}' if text else ''))

    def _finished(self, sid, doc):
        if sid != self.STUDY:
            return
        self.doc = doc
        self.note.setText(f'{self.TITLE} · {doc.get("seconds", 0):.2f} s' + (' (from the cache)' if doc.get('cached')
                                                                             else ''))
        self.show_result(doc)
        if self.ctx.window is not None:
            self.ctx.window.viewport.update()

    def _failed(self, sid, message):
        if sid == self.STUDY:
            from ..store import failure_text
            text, _explained = failure_text(message, self.TITLE or 'The study')
            self.head.set('—', '', 'critical', 'could not run', text)

    def export(self, path=None):
        from ...core import provenance as PV
        if self.doc is None:
            if path is None and self.ctx.window is not None:     # the button: say why nothing happens
                self.ctx.window.msg(f'There is no {self.TITLE.lower() or "result"} result to export yet: Run it '
                                    'first.', 5000)
            return None
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, f'Export {self.TITLE}', f'{self.STUDY}.json', 'JSON (*.json)')
            if not path:
                return None
        doc = {k: v for k, v in self.doc.items() if k not in ('cached', 'revision')}
        doc = PV.stamp(doc, self.ctx.store.project, output={'option': None, 'name': os.path.basename(path),
                                                            'in_folder': False})
        try:
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(doc, fh, indent=1, ensure_ascii=False)
                fh.write('\n')
        except OSError as e:                                     # a folder without write access, a full disk
            if self.ctx.window is not None:
                self.ctx.window.msg(f'Could not write {os.path.basename(path)}: {e.strerror or e}.', 8000)
            return None
        if self.ctx.window is not None:
            self.ctx.window.msg(f'Exported {self.TITLE.lower() or "the result"} to {os.path.basename(path)}', 5000)
        return path


for _name in _FLUSH_FIRST:                  # the base's own hooks too (a page that does not override them)
    setattr(StudyPage, _name, _flushing(getattr(StudyPage, _name)))


def row(*widgets) -> QWidget:
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(6)
    for x in widgets:
        h.addWidget(x)
    h.addStretch(1)
    return w
