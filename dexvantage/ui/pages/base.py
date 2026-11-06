"""The page pattern every study page follows (spec §4): what it shows, the number that matters (a large mono figure
with a status word), and the action to improve it, with Apply (undoable), Preview and Explain, plus Run and Export.

Subclass StudyPage, set STUDY and TITLE, build the body in build_body(layout) and fill it in show_result(doc). The
page re-renders whenever the controller finishes its study. Overlays for the 3D view come from overlays(vp) and are
drawn while the page is the Analysis dock's current tab (return [] to draw nothing).
"""
from __future__ import annotations

import json
import os

from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

from ..panels import Headline
from ..widgets import button, hline, label


class StudyPage(QWidget):
    STUDY: str | None = None
    TITLE = ''
    HEADLINE = ''                      # the headline's title line

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.doc = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.head = Headline(self.HEADLINE or self.TITLE)
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
            self.head.set('—', '', 'critical', 'could not run', message.splitlines()[0])

    def export(self, path=None):
        from ...core import provenance as PV
        if self.doc is None:
            return None
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, f'Export {self.TITLE}', f'{self.STUDY}.json', 'JSON (*.json)')
            if not path:
                return None
        doc = {k: v for k, v in self.doc.items() if k not in ('cached', 'revision')}
        doc = PV.stamp(doc, self.ctx.store.project, output={'option': None, 'name': os.path.basename(path),
                                                            'in_folder': False})
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(doc, fh, indent=1, ensure_ascii=False)
            fh.write('\n')
        return path


def row(*widgets) -> QWidget:
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(6)
    for x in widgets:
        h.addWidget(x)
    h.addStretch(1)
    return w
