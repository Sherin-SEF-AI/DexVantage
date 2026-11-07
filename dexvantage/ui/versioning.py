"""File ▸ Compare with…, Save as folder (for Git)… and Open folder…: the project diff (core/projdiff.py) in a window,
and the unpacked project form (core/projfolder.py).

    compare(window, path=None) -> CompareWindow      the open project against another project file or folder
    save_folder(window, path=None) -> bool           Save as folder: the unpacked, Git-friendly form
    open_folder(window, path=None) -> bool           Open an unpacked project folder

The Compare window reads the other project and computes the diff as a window job (off the UI thread). It shows the
headline (how many changes, a status word), the changes in a table with a section filter, and, with Compare figures,
the figures of merit of the studies run on the open project, computed for both projects in the background
(core/metrics; the open project's results come from the analysis cache). Export writes the diff as JSON (with
provenance) or as an HTML page.
"""
from __future__ import annotations

import json
import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QMessageBox, QTableWidget,
                             QTableWidgetItem, QVBoxLayout)

from .dialogs.base import VDialog
from .panels import Headline
from .widgets import button, label

COLUMNS = ('Section', 'Change', 'Item', 'Field', 'Before', 'After')


class CompareWindow(VDialog):
    """The open project compared with another: what changed, section by section, and the figures of merit."""

    def __init__(self, window):
        super().__init__(window, 'Compare projects', '', size=(980, 640))
        self.win = window
        self.other_path = ''
        self.other = None
        self.other_first = True          # the other project is "before", the open one "after"
        self.diff = None
        v = QVBoxLayout(self.body)
        v.setContentsMargins(0, 0, 0, 8)
        v.setSpacing(6)
        self.head = Headline('Changes')
        self.b_choose = button('Choose project…', role='primary', icon_name='folder',
                               tip='A .dexv file or an unpacked .dexv.d folder')
        self.b_swap = button('Swap', role='ghost', tip='Compare in the other direction')
        self.b_figs = button('Compare figures', role='ghost', tip='Run the metrics of the studies run on the open '
                                                                  'project on both projects and show their change')
        self.b_json = button('Export JSON…', role='ghost', icon_name='export', tip='The diff as JSON with provenance')
        self.b_html = button('Export HTML…', role='ghost', icon_name='export', tip='The diff as a standalone page')
        for b in (self.b_choose, self.b_swap, self.b_figs, self.b_json, self.b_html):
            self.head.actions.addWidget(b)
        self.head.actions.addStretch(1)
        v.addWidget(self.head)
        row = QHBoxLayout()
        row.setContentsMargins(12, 0, 12, 0)
        row.addWidget(label('Section', muted=True))
        self.section = QComboBox()
        self.section.addItem('All sections', '')
        self.section.currentIndexChanged.connect(self._fill)
        row.addWidget(self.section)
        row.addStretch(1)
        v.addLayout(row)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setStretchLastSection(True)
        v.addWidget(self.table, 3)
        self.metrics = QTableWidget(0, 4)
        self.metrics.setHorizontalHeaderLabels(('Figure', 'Before', 'After', 'Change'))
        self.metrics.verticalHeader().setVisible(False)
        self.metrics.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.metrics.horizontalHeader().setStretchLastSection(True)
        self.metrics.hide()
        v.addWidget(self.metrics, 1)
        close = button('Close')
        close.clicked.connect(self.reject)
        self.buttons.addWidget(close)
        self.b_choose.clicked.connect(lambda: self.choose())
        self.b_swap.clicked.connect(self.swap)
        self.b_figs.clicked.connect(self.compare_figures)
        self.b_json.clicked.connect(lambda: self.export_json())
        self.b_html.clicked.connect(lambda: self.export_html())
        self._enable(False)
        self.head.set('—', '', 'warning', 'nothing compared', 'Choose a project file or folder to compare the open '
                                                              'project with.')

    def _enable(self, on: bool):
        for b in (self.b_swap, self.b_figs, self.b_json, self.b_html):
            b.setEnabled(on)

    # ---------------------------------------------------------------- the diff
    def choose(self, path=None) -> bool:
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, 'Compare with', '', 'DexVantage project (*.dexv);;'
                                                                            'Project folder (project.json)')
            if not path:
                return False
            if os.path.basename(path) == 'project.json':
                path = os.path.dirname(path)
        self.other_path = path
        return self.recompute()

    def swap(self):
        self.other_first = not self.other_first
        self.recompute()

    def recompute(self) -> bool:
        from ..core import projdiff as D
        from ..core.project import Project
        if not self.other_path:
            return False
        here = self.win.store.project.snapshot()
        path, first = self.other_path, self.other_first

        def work(progress, cancel):
            other = Project.load(path)
            progress(0.5, 'comparing')
            return other, (D.diff(other, here) if first else D.diff(here, other))

        def done(out):
            self.other, self.diff = out
            self.metrics.hide()
            self.show_diff()

        def failed(m):                                  # a ProjectFormatError's own sentence; the traceback is logged
            self.head.set('—', '', 'critical', 'could not compare', m.splitlines()[0])
        self.head.set('…', '', 'warning', 'comparing', f'Reading {os.path.basename(path.rstrip("/"))}…')
        return self.win.run_job('compare', 'Comparing projects', work, done, failed)

    def show_diff(self):
        d = self.diff
        n = len(d['changes'])
        a, b = d['a'], d['b']
        self.sub_lbl.setText(f'{a["file"] or a["name"]}  →  {b["file"] or b["name"] + " (open)"}')
        if d['identical']:
            self.head.set('0', 'changes', 'good', 'same', 'The two projects hold the same content (the design log and '
                                                          'save times are not compared).')
        else:
            parts = ', '.join(f'{k} {v}' for k, v in d['counts'].items())
            self.head.set(str(n), 'change' if n == 1 else 'changes', 'warning', 'differs',
                          f'{parts}. Before is the left project, after the right one; Swap reverses them.')
        cur = self.section.currentData()
        self.section.blockSignals(True)
        self.section.clear()
        self.section.addItem('All sections', '')
        for k, v in d['counts'].items():
            self.section.addItem(f'{k} ({v})', k)
        i = self.section.findData(cur)
        self.section.setCurrentIndex(max(0, i))
        self.section.blockSignals(False)
        self._fill()
        self._enable(True)

    def _fill(self, *_):
        from ..core.projdiff import brief_record
        if self.diff is None:
            return
        sec = self.section.currentData() or ''
        rows = [c for c in self.diff['changes'] if not sec or c['section'] == sec]
        self.table.setRowCount(len(rows))
        for i, c in enumerate(rows):
            vals = (c['section'], c['op'], f'{c["kind"]} {c["item"]}', c['field'], brief_record(c['before']),
                    brief_record(c['after']))
            for j, t in enumerate(vals):
                it = QTableWidgetItem(str(t))
                it.setToolTip(c['text'])
                self.table.setItem(i, j, it)

    # ---------------------------------------------------------------- figures of merit
    def metric_ids(self) -> list:
        from ..core import metrics as MX
        MX.load_metrics()
        done = set(self.win.ctx.analysis.results)
        return sorted(m for m, x in MX.METRICS.items() if x.study in done)

    def compare_figures(self) -> bool:
        from ..core import projdiff as D
        ids = self.metric_ids()
        if self.diff is None:
            return False
        if not ids:
            self.win.msg('Run a study on the open project first: the figures compared are those of the studies '
                         'already run.', 8000)
            return False
        here = self.win.store.project.snapshot()
        other = self.other.snapshot()
        cache = self.win.ctx.analysis.cache
        first = self.other_first

        def work(progress, cancel):
            a, b = (other, here) if first else (here, other)
            return D.evaluate_deltas(a, b, ids, cache=cache)

        def done(rows):
            self.diff['metrics'] = rows
            self.show_metrics()
        return self.win.run_job('compare_figures', 'Comparing figures of merit', work, done)

    def show_metrics(self):
        from ..core.projdiff import _fmt, _metric_text
        rows = self.diff.get('metrics') or []
        self.metrics.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, t in enumerate((r['title'], _fmt(r['a'], r['unit']), _fmt(r['b'], r['unit']), _metric_text(r))):
                it = QTableWidgetItem(t)
                if j:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                tip = r.get('reason_a') or r.get('reason_b') or ''
                it.setToolTip(tip)
                self.metrics.setItem(i, j, it)
        self.metrics.setVisible(bool(rows))

    # ---------------------------------------------------------------- export
    def export_json(self, path=None):
        from ..core import provenance as PV
        if self.diff is None:
            return None
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, 'Export the diff', 'diff.json', 'JSON (*.json)')
            if not path:
                return None
        doc = PV.stamp(dict(self.diff), self.win.store.project, output={'option': None,
                                                                        'name': os.path.basename(path),
                                                                        'in_folder': False})
        return self._write(path, json.dumps(doc, indent=1, ensure_ascii=False) + '\n')

    def export_html(self, path=None):
        from ..core import projdiff as D
        if self.diff is None:
            return None
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, 'Export the diff', 'diff.html', 'HTML (*.html)')
            if not path:
                return None
        return self._write(path, D.to_html(self.diff))

    def _write(self, path, text):
        try:
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(text)
        except OSError as e:
            QMessageBox.warning(self, 'Could not export', f'{os.path.basename(path)} could not be written: '
                                                          f'{e.strerror or e}.')
            return None
        self.win.msg(f'Exported the diff to {os.path.basename(path)}', 5000)
        return path


def compare(window, path=None) -> CompareWindow:
    """File ▸ Compare with…: the window (kept on the main window), and the diff when a path is given."""
    w = getattr(window, '_compare_window', None)
    if w is None:
        w = CompareWindow(window)
        window._compare_window = w
    w.fit_screen()
    w.show()
    w.raise_()
    if path is not None:
        w.choose(path)
    elif not w.other_path:
        w.choose()
    return w


def save_folder(window, path=None) -> bool:
    """File ▸ Save as folder (for Git)…: the unpacked form (one JSON file per section; core/projfolder.py)."""
    from ..core.project import ProjectFormatError
    p = window.store.project
    if path is None:
        start = p.path if p.path.endswith('.dexv.d') else f'{p.name}.dexv.d'
        path, _ = QFileDialog.getSaveFileName(window, 'Save as folder (for Git)', start,
                                              'Unpacked DexVantage project (*.dexv.d)')
        if not path:
            return False
    path = path.rstrip('/\\')
    if not path.endswith('.dexv.d'):
        path += '.dexv.d'
    if os.path.isfile(path):
        QMessageBox.warning(window, 'Could not save', f'{os.path.basename(path)} is a file: choose a folder name.')
        return False
    try:
        return window._write(path)
    except ProjectFormatError as e:
        QMessageBox.warning(window, 'Could not save', str(e))
        return False


def open_folder(window, path=None) -> bool:
    """File ▸ Open folder…: an unpacked project (a .dexv.d folder)."""
    if path is None:
        path = QFileDialog.getExistingDirectory(window, 'Open project folder')
        if not path:
            return False
    if not os.path.isfile(os.path.join(path, 'project.json')):
        QMessageBox.warning(window, 'Could not open the project',
                            f'{os.path.basename(path)} has no project.json, so it is not an unpacked DexVantage '
                            'project.')
        return False
    return window.open_project(path)
