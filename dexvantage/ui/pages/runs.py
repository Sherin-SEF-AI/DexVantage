"""The Runs page (Analysis ▸ Runs): the experiment tracker (core/tracking). With tracking on, every study run,
designed experiment, optimiser run, tolerance Monte Carlo and scorecard the window finishes is logged as a run in the
project's run store (its settings, figures with units and status, result and provenance); this page lists the runs
(filtered by study, tag and date), compares two or more side by side with their differences, opens a run's result,
re-runs it (core/reproduce: the same study on the project as it is now, logged as a new run with the verdict) and
exports the runs as CSV and Parquet tables, an MLflow mlruns/ folder and a W&B replay script.

The headline counts the runs in the store, with a status word: tracking on, tracking off, no runs yet, not saved (a
project never saved has no folder beside it), damaged lines. Apply turns tracking on (one undo step; the setting is not
part of the project's content hash), Preview says where the runs would go and what is logged, Explain says how.

Logging happens off the UI thread: the window's quick studies that re-run after each edit are logged only with "Also
log the quick studies" ticked; a result already logged in this session is not logged twice; a result made before the
latest edit is not logged (its project is no longer the one in the window). Logging listens to the window's analysis
controller itself, not the page's held signals (ui/pages/_gate.py), so runs are logged while the page is hidden.

Known limits: those of core/tracking; a comparison of more than two runs gives deltas against the first selected and
no significance note; the table shows the newest 500 runs the filters keep (of at most 2000 read).
"""
from __future__ import annotations

import os
import threading
import traceback

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QHeaderView, QInputDialog,
                             QLineEdit, QTreeWidget, QTreeWidgetItem)

from ...core import tracking as TR
from ...core.jobs import StudyError
from ...core.tracking.record import fmt
from ..theme import STATUS, mono
from ..widgets import Section, button, label
from . import register
from ._gate import real
from .base import StudyPage, row

ROLE = Qt.ItemDataRole.UserRole
COLUMNS = ['Started (UTC)', 'Study', 'Kind', 'Status', 'Headline', 'Project · variant', 'Tags', 'Note']
STATUS_LEVEL = {'finished': 'good', 'cancelled': 'warning', 'failed': 'critical'}


class RunsPage(StudyPage):
    """Runs: every study run, experiment, optimiser run, tolerance Monte Carlo and scorecard logged with its
    settings, figures and provenance while tracking is on, in a run store beside the project (JSON lines and one
    folder per run, safe for several writers). Select two or more runs to compare their figures side by side."""
    STUDY = 'tracking'
    TITLE = 'Runs'
    HEADLINE = 'Runs logged'
    MAX_SHOWN = 500                                 # rows in the table (the newest); the filters reach the rest
    _worker_done = pyqtSignal(str, object)          # (what, result) from a worker thread

    def __init__(self, ctx):
        self._log = TR.SessionLog()
        self._threads: list = []
        self._rows: list = []
        self._building = False
        self.compared = None
        self.last_message = ''
        super().__init__(ctx)
        self.b_apply.setText('Turn tracking on')
        self.b_apply.setToolTip('Log every study run of this project from now on (one undo step)')
        self.b_preview.setToolTip('Say where the runs would go and what is logged')
        self._worker_done.connect(self._on_worker)
        # logging follows the real controller: a hidden page's signals are held until it is shown (pages/_gate.py),
        # and a run must be logged when it finishes, against the design it was made for
        live = real(ctx.analysis)
        live.finished.connect(self._study_finished)
        live.failed.connect(self._study_failed)
        ctx.store.changed.connect(self._changed)
        ctx.store.project_replaced.connect(self._replaced)
        self._sync_settings()

    # ---------------------------------------------------------------- layout
    def build_body(self, layout):
        sec = Section('Tracking')
        self.c_on = QCheckBox('Log every run of this project')
        self.c_on.setToolTip('Each study run, experiment, optimiser run, tolerance Monte Carlo and scorecard is '
                             'logged as a run (one undo step to change; not part of the project\'s content hash)')
        self.c_on.toggled.connect(lambda on: self.set_setting('enabled', bool(on)))
        self.c_quick = QCheckBox('Also log the quick studies (they re-run after each edit)')
        self.c_quick.setToolTip('Default off: the live studies would log a run per edit')
        self.c_quick.toggled.connect(lambda on: self.set_setting('quick', bool(on)))
        sec.add(row(self.c_on, self.c_quick))
        self.l_folder = label('', muted=True, mono_=True)
        self.b_folder = button('Choose folder…', role='ghost', icon_name='folder', tip='Keep the runs in another '
                               'folder (default: <project file name>.runs beside the project)')
        self.b_folder.clicked.connect(lambda: self.choose_folder())
        self.b_default_folder = button('Beside the project', role='ghost', tip='Use the default folder again')
        self.b_default_folder.clicked.connect(lambda: self.set_setting('folder', ''))
        sec.row('Run store', self.l_folder, self.b_folder, self.b_default_folder, stretch_last=False)
        self.e_tags = QLineEdit()
        self.e_tags.setPlaceholderText('tags added to every run, comma-separated (e.g. thesis-ch3, wrist-v2)')
        self.e_tags.editingFinished.connect(lambda: self.set_setting('tags', TR.clean_tags(self.e_tags.text())))
        sec.row('Tags', self.e_tags)
        layout.addWidget(sec)

        rs = Section('Runs')
        self.f_study = QComboBox()
        self.f_study.addItem('Every study', '')
        self.f_study.currentIndexChanged.connect(lambda _i: self.fill_table())
        self.f_tag = QLineEdit()
        self.f_tag.setPlaceholderText('tag')
        self.f_tag.textChanged.connect(lambda _t: self.fill_table())
        self.f_since = QLineEdit()
        self.f_since.setPlaceholderText('since 2026-11-01')
        self.f_since.setFont(mono(10))
        self.f_since.textChanged.connect(lambda _t: self.fill_table())
        self.f_until = QLineEdit()
        self.f_until.setPlaceholderText('until 2026-11-30')
        self.f_until.setFont(mono(10))
        self.f_until.textChanged.connect(lambda _t: self.fill_table())
        rs.add(row(label('Filter', ink2=True), self.f_study, self.f_tag, self.f_since, self.f_until))
        self.table = QTreeWidget()
        self.table.setColumnCount(len(COLUMNS))
        self.table.setHeaderLabels(COLUMNS)
        self.table.setRootIsDecorated(False)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        hdr = self.table.header()
        for k in range(len(COLUMNS)):
            hdr.setSectionResizeMode(k, QHeaderView.ResizeMode.Stretch if k == 4 else
                                     QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._picked)
        self.table.setMinimumHeight(160)
        rs.add(self.table)
        self.b_compare = button('Compare', icon_name='compare', tip='The selected runs\' figures side by side, with '
                                'their differences from the first selected')
        self.b_compare.clicked.connect(lambda: self.compare_selected())
        self.b_open = button('Open result', role='ghost', icon_name='external', tip='Open the selected run\'s result '
                             'file (JSON with provenance)')
        self.b_open.clicked.connect(lambda: self.open_result())
        self.b_rerun = button('Re-run', role='ghost', icon_name='refresh', tip='Run the selected run\'s study again on '
                              'the project as it is now and compare every number (core/reproduce); logged as a new '
                              'run')
        self.b_rerun.clicked.connect(lambda: self.rerun_selected())
        self.b_tags = button('Tags…', role='ghost', tip='Set the selected runs\' tags')
        self.b_tags.clicked.connect(lambda: self.edit_tags())
        self.b_note = button('Note…', role='ghost', tip='Set the selected run\'s note')
        self.b_note.clicked.connect(lambda: self.edit_note())
        self.b_export_runs = button('Export runs…', role='ghost', icon_name='export', tip='The runs shown as CSV and '
                                    'Parquet tables, an MLflow mlruns/ folder and a W&B replay script, with a '
                                    'manifest')
        self.b_export_runs.clicked.connect(lambda: self.export_runs())
        rs.add(row(self.b_compare, self.b_open, self.b_rerun, self.b_tags, self.b_note, self.b_export_runs))
        self.l_sel = label('', muted=True, wrap=True)
        rs.add(self.l_sel)
        layout.addWidget(rs)

        cs = Section('Compare')
        self.cmp = QTreeWidget()
        self.cmp.setRootIsDecorated(False)
        self.cmp.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.cmp.setMinimumHeight(140)
        cs.add(self.cmp)
        self.l_diff = label('', muted=True, wrap=True)
        cs.add(self.l_diff)
        layout.addWidget(cs)
        self._enable_actions()

    # ---------------------------------------------------------------- settings
    def _store(self):
        try:
            return TR.store_for(self.ctx.store.project)
        except TR.TrackingError:
            return None

    def _sync_settings(self):
        st = TR.settings_of(self.ctx.store.project)
        self._building = True
        self.c_on.setChecked(st['enabled'])
        self.c_quick.setChecked(st['quick'])
        self.e_tags.setText(', '.join(st['tags']))
        self._building = False
        try:
            folder = TR.store_folder(self.ctx.store.project)
            # one line: the folder's name and where it is (the full path in the tooltip; a long path wrapped and
            # was cut off in the row)
            ppath = getattr(self.ctx.store.project, 'path', '') or ''
            if not st['folder'] and ppath:
                text = f'{os.path.basename(folder)} (beside {os.path.basename(ppath)})'
            else:
                parent = os.path.basename(os.path.dirname(folder))
                text = os.path.join('…', parent, os.path.basename(folder)) if parent else folder
            self.l_folder.setText(text)
            self.l_folder.setToolTip(folder)
        except TR.TrackingError as e:
            self.l_folder.setText('— (save the project first, or choose a folder)')
            self.l_folder.setToolTip(str(e))
        self.suggestion_ready(not st['enabled'])

    def set_setting(self, key: str, value) -> bool:
        """Change one tracking setting (one undo step); False when it is already so."""
        if self._building:
            return False
        if TR.settings_of(self.ctx.store.project).get(key) == value:
            return False
        words = {'enabled': 'turn tracking ' + ('on' if value else 'off'),
                 'quick': ('log' if value else 'stop logging') + ' the quick studies',
                 'folder': 'run store folder', 'tags': 'run tags'}

        def fn(pp):
            d = dict(pp.analysis.studies.get(TR.SETTINGS_KEY) or {})
            d[key] = value
            pp.analysis.studies[TR.SETTINGS_KEY] = d
        self.ctx.store.edit('analysis', fn, f'Runs · {words.get(key, key)}')
        return True

    def choose_folder(self, folder: str | None = None) -> bool:
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Run store folder')
            if not folder:
                return False
        return self.set_setting('folder', os.path.abspath(folder))

    def apply(self):
        """Turn tracking on (one undo step)."""
        if self.set_setting('enabled', True):
            self._msg('Tracking is on: the next study run is logged.')

    def preview(self):
        try:
            where = TR.store_folder(self.ctx.store.project)
        except TR.TrackingError as e:
            self._msg(str(e))
            return
        self._msg('With tracking on, each study run, experiment, optimiser run, tolerance Monte Carlo and scorecard '
                  f'is logged in {os.path.basename(where)} (its settings, figures, result and provenance).')

    def _changed(self, sections):
        if 'analysis' in sections:
            self._sync_settings()
            self.refresh()

    def _replaced(self):
        self.doc = None
        self._rows = []
        self.compared = None
        self._sync_settings()
        self.fill_table()
        self.empty()

    # ---------------------------------------------------------------- reading the store
    def run(self):
        self.refresh()

    def refresh(self):
        """Read the store again (the 'tracking' study, keyed by the store's fingerprint so the cache never serves an
        old listing)."""
        st = self._store()
        fp = st.fingerprint() if st is not None else 'none'
        self.ctx.analysis.run(self.STUDY, settings={'read': fp})

    def show_result(self, doc):
        h = doc.get('headline') or {}
        self.head.set(str(h.get('value', 0)), 'runs', h.get('status') or 'warning', h.get('word') or '',
                      h.get('text') or '')
        self._rows = list(doc.get('runs') or [])
        studies = sorted({r.get('study') or '' for r in self._rows} - {''})
        cur = self.f_study.currentData()
        self.f_study.blockSignals(True)
        self.f_study.clear()
        self.f_study.addItem('Every study', '')
        for s in studies:
            self.f_study.addItem(s, s)
        k = self.f_study.findData(cur)
        self.f_study.setCurrentIndex(max(0, k))
        self.f_study.blockSignals(False)
        self.fill_table()
        self.suggestion_ready(not doc.get('enabled'))

    def filtered(self) -> list:
        """The runs the filters keep (newest first)."""
        self._filter_error = ''
        try:
            rows = TR.filter_runs(self._rows, study=self.f_study.currentData() or None,
                                  tag=self.f_tag.text().strip() or None, since=self.f_since.text().strip() or None,
                                  until=self.f_until.text().strip() or None)
        except TR.TrackingError as e:
            self._filter_error = str(e)
            return []
        rows.reverse()
        return rows

    def fill_table(self):
        sel = set(self.selected_ids())
        rows = self.filtered()
        self._hidden = max(0, len(rows) - self.MAX_SHOWN)
        self._building = True
        self.table.setUpdatesEnabled(False)
        self.table.clear()
        items = []
        for r in rows[:self.MAX_SHOWN]:
            h = r.get('headline') or {}
            pv = ' · '.join(x for x in (r.get('project'), r.get('variant')) if x)
            it = QTreeWidgetItem([(r.get('started_utc') or '')[:19].replace('T', ' '), r.get('title') or '',
                                  r.get('kind') or '', r.get('status') or '', h.get('text') or r.get('reason') or '',
                                  pv, ', '.join(r.get('tags') or []), r.get('note') or ''])
            it.setData(0, ROLE, r.get('run_id'))
            it.setFont(0, mono(10))
            it.setForeground(3, QColor(STATUS.get(STATUS_LEVEL.get(r.get('status'), ''), '#8a8a8a')))
            it.setToolTip(4, h.get('text') or r.get('reason') or '')
            it.setToolTip(0, f'run {r.get("run_id")}' + (f'; re-run of {r["rerun_of"][:8]} ({r.get("reproduce")})'
                                                         if r.get('rerun_of') else ''))
            items.append((it, r.get('run_id') in sel))
        self.table.addTopLevelItems([it for it, _ in items])
        for it, on in items:
            if on:
                it.setSelected(True)
        self.table.setUpdatesEnabled(True)
        self._building = False
        self._picked()

    # ---------------------------------------------------------------- selection
    def selected_ids(self) -> list:
        """The selected runs' ids in table order (newest first)."""
        out = []
        for k in range(self.table.topLevelItemCount()):
            it = self.table.topLevelItem(k)
            if it.isSelected():
                out.append(it.data(0, ROLE))
        return out

    def select_ids(self, ids):
        want = set(ids)
        self._building = True
        for k in range(self.table.topLevelItemCount()):
            it = self.table.topLevelItem(k)
            it.setSelected(it.data(0, ROLE) in want)
        self._building = False
        self._picked()

    def _picked(self):
        if self._building:
            return
        n = len(self.selected_ids())
        shown = self.table.topLevelItemCount()
        more = getattr(self, '_hidden', 0)
        if getattr(self, '_filter_error', ''):
            self.l_sel.setText(self._filter_error)
            self._enable_actions()
            return
        self.l_sel.setText(f'{shown} run{"" if shown == 1 else "s"} shown' +
                           (f' (the newest; {more} older: narrow the filters to see them)' if more else '') +
                           f', {n} selected' + ('; select two or more to compare' if n == 1 else ''))
        self._enable_actions()

    def _enable_actions(self):
        n = len(self.selected_ids()) if hasattr(self, 'table') else 0
        self.b_compare.setEnabled(n >= 2)
        self.b_open.setEnabled(n == 1)
        self.b_rerun.setEnabled(n == 1)
        self.b_tags.setEnabled(n >= 1)
        self.b_note.setEnabled(n == 1)

    # ---------------------------------------------------------------- work off the UI thread
    def _start(self, what: str, fn):
        def work():
            try:
                res = fn()
            except StudyError as e:                             # a plain sentence (store, re-run, compare)
                res = e
            except Exception as e:                              # noqa: BLE001 — explained, the traceback logged
                traceback.print_exc()
                res = RuntimeError(f'{type(e).__name__}: {(str(e).splitlines() or [""])[0][:200]}')
            try:
                self._worker_done.emit(what, res)
            except RuntimeError:
                pass                                           # the page is gone
        t = threading.Thread(target=work, daemon=True)
        self._threads = [x for x in self._threads if x.is_alive()] + [t]
        t.start()
        return t

    def wait(self, timeout: float = 60.0) -> bool:
        """Join the page's worker threads (tests); the results arrive as queued signals."""
        import time
        t0 = time.time()
        for t in list(self._threads):
            t.join(max(0.0, timeout - (time.time() - t0)))
        return not any(t.is_alive() for t in self._threads)

    def _msg(self, text: str, ms: int = 8000):
        self.last_message = text
        if self.ctx.window is not None:
            self.ctx.window.msg(text, ms)

    def _on_worker(self, what: str, res):
        if isinstance(res, Exception):
            text = str(res) if isinstance(res, StudyError) else f'Could not {what.replace("_", " ")}: {res}.'
            self._msg(text)
            if what == 'compare':
                self.l_diff.setText(text)
            return
        if what == 'logged':
            if res is not None:
                self.refresh()
        elif what == 'compare':
            self.show_comparison(res)
        elif what == 'rerun':
            new, rep = res
            self._msg(f'Re-run: {rep.get("message") or rep.get("verdict_text", "")}', 15000)
            self.refresh()
        elif what == 'export':
            ok = [f for f, r in res['formats'].items() if r['status'] == 'ok']
            bad = [f'{f} ({r.get("reason", "")})' for f, r in res['formats'].items() if r['status'] != 'ok']
            self._msg(f'Exported {res["runs"]} runs as {", ".join(ok)}' + (f'; refused: {"; ".join(bad)}' if bad
                                                                          else '') + '.')
        elif what == 'update':
            self.refresh()

    # ---------------------------------------------------------------- logging what the window runs
    def _study_finished(self, sid, doc):
        p = self.ctx.store.project
        if not TR.should_log(sid, p, requested=False) or not isinstance(doc, dict):
            return
        if doc.get('revision', self.ctx.store.revision) != self.ctx.store.revision:
            return                                              # made before the latest edit
        snap = p.snapshot()
        self._start('logged', lambda: self._log.log(sid, doc, snap, interface='app'))

    def _study_failed(self, sid, message):
        p = self.ctx.store.project
        if not TR.should_log(sid, p, requested=False):
            return
        first = (str(message).splitlines() or [''])[0]
        reason = first.split(': ', 1)[1] if ': ' in first else first
        snap = p.snapshot()
        self._start('logged', lambda: TR.log_failure(sid, reason, snap, interface='app'))

    # ---------------------------------------------------------------- actions
    def compare_selected(self, ids=None):
        """Compare the selected runs (or ids) in the background; the comparison fills the Compare table."""
        ids = list(ids) if ids is not None else list(reversed(self.selected_ids()))   # oldest first: the baseline
        st = self._store()
        if st is None or len(ids) < 2:
            self._msg('Select two or more runs to compare.')
            return None
        return self._start('compare', lambda: TR.compare([st.get(i) for i in ids]))

    def show_comparison(self, cmp: dict):
        self.compared = cmp
        n = len(cmp['runs'])
        heads = ['Figure', 'Unit'] + [f'Run {k + 1} ({(cmp["started"][k] or "")[:16].replace("T", " ")})'
                                      for k in range(n)] + [f'Δ run {k + 1}' for k in range(1, n)] + ['Best', 'Note']
        self.cmp.clear()
        self.cmp.setColumnCount(len(heads))
        self.cmp.setHeaderLabels(heads)
        for r in cmp['rows']:
            sig = (r.get('significance') or {}).get('text', '') if r.get('significance') else ''
            best = '' if r['best'] is None else f'run {r["best"] + 1}'
            vals = [fmt(v, '') for v in r['values']]
            deltas = [TR.delta_text(d) for d in r['deltas'][1:]]
            it = QTreeWidgetItem([r['title'], r['unit']] + vals + deltas + [best, 'same' if r['same'] else sig])
            for c in range(2, 2 + 2 * n - 1):
                it.setFont(c, mono(10))
            it.setToolTip(0, r['key'])
            self.cmp.addTopLevelItem(it)
        for c in range(len(heads)):
            self.cmp.resizeColumnToContents(c)
        parts = list(cmp['notes'])
        if cmp['differences']:
            parts.append('What differs: ' + '; '.join(f'{d["path"]}' for d in cmp['differences'][:12]) +
                         (f' and {len(cmp["differences"]) - 12} more' if len(cmp['differences']) > 12 else ''))
        else:
            parts.append('Same settings, project content and DexVantage version.')
        self.l_diff.setText(' '.join(parts))

    def _one(self):
        ids = self.selected_ids()
        st = self._store()
        if st is None or len(ids) != 1:
            self._msg('Select one run.')
            return None, None
        return st, ids[0]

    def open_result(self) -> str | None:
        st, rid = self._one()
        if st is None:
            return None
        path = st.result_path(rid)
        if not os.path.isfile(path):
            self._msg('This run kept no result (it failed, or its folder is gone).')
            return None
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        return path

    def rerun_selected(self):
        st, rid = self._one()
        if st is None:
            return None
        snap = self.ctx.store.project.snapshot()
        self._msg('Re-running the run\'s study in the background…', 4000)
        return self._start('rerun', lambda: TR.rerun(st, rid, snap, interface='app'))

    def edit_tags(self, tags=None):
        ids = self.selected_ids()
        st = self._store()
        if st is None or not ids:
            return None
        if tags is None:
            cur = next((r.get('tags') for r in self._rows if r.get('run_id') == ids[0]), []) or []
            text, ok = QInputDialog.getText(self, 'Tags', 'Tags (comma-separated):', text=', '.join(cur))
            if not ok:
                return None
            tags = text
        tags = TR.clean_tags(tags)
        return self._start('update', lambda: [st.update(i, tags=tags) for i in ids])

    def edit_note(self, note=None):
        st, rid = self._one()
        if st is None:
            return None
        if note is None:
            cur = next((r.get('note') for r in self._rows if r.get('run_id') == rid), '') or ''
            note, ok = QInputDialog.getMultiLineText(self, 'Note', 'Note:', cur)
            if not ok:
                return None
        return self._start('update', lambda: st.update(rid, note=str(note)))

    def export_runs(self, folder=None):
        st = self._store()
        if st is None or not st.exists():
            self._msg('There are no runs to export yet.')
            return None
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export runs to')
            if not folder:
                return None
        ids = {r['run_id'] for r in self.filtered()}
        return self._start('export', lambda: TR.export(st, folder, runs=[r for r in st.runs()
                                                                         if r['run_id'] in ids]))


register('runs', 'Runs', RunsPage, menu='Analysis', icon='history', action_id=None, workspace=None, order=96)
