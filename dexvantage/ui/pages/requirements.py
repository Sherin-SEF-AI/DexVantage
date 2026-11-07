"""The Requirements page (spec §13 "Requirements and Variants compare", §7.11, §11): the project's measurable
requirements, the readiness scorecard and every variant against every requirement.

The list is editable: add a requirement (a metric of core/metric_defs.py, a comparison and a threshold in the metric's
unit), the labelled defaults, or a YAML file (the format batch --check reads); edit or remove one. Every edit is one
undo step (the project's 'requirements' section); editing a default's threshold makes it your own figure (no longer
labelled a default). Run evaluates the scorecard in the background (core/studies/requirements.py): each item's value,
margin and status word (pass, fail, not evaluated with the reason). New variant from this one copies the active
variant (one undo step) to change and compare. Compare variants evaluates every variant and marks
the best variant per requirement. Apply makes the variant that passes most requirements the active one (one undo
step); Preview shows its sensors in the 3D view next to the active variant's; Show in 3D draws any other variant's
cameras (dashed frusta, ui/variant_ghost.py) beside the active variant's. Export writes the scorecard as JSON with
provenance; Save as YAML writes the list as a requirements file for the CI gate.

3D view: while the page is shown, each sensor of the active variant is marked in the scorecard's status colour; with
Preview, the best variant's sensors are drawn too, with a line from each active sensor to its counterpart.

Known limits: those of core/requirements.py; the variants compare re-runs every study each variant needs (long
studies take their time per variant); the page has no Camera view overlay (a scorecard has nothing to draw in an
image).
"""
from __future__ import annotations

import traceback

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QHeaderView, QLineEdit,
                             QTreeWidget, QTreeWidgetItem)

from ...core import metrics as M
from ...core import requirements as RQ
from ...core.project import Requirement
from ..theme import SEQ, STATUS, mono
from ..viewport3d import LineDraw
from ..widgets import Section, button, label
from . import register
from .base import StudyPage, row

ROLE = Qt.ItemDataRole.UserRole
VARIANTS = 'variants_compare'


def _col(level: str) -> QColor:
    return QColor(STATUS.get(level, '#8a8a8a'))


class RequirementsPage(StudyPage):
    """Requirements: the rig's measurable targets, each a study's figure against a threshold. The headline counts
    the requirements that pass on the active variant; Run evaluates them, Compare variants every variant. The
    defaults (contact visibility at least 95 %, skew under 2 ms, no controller over budget, no cable pull or snag, no
    blind gap before contact) are DexVantage's documented starting points, not standards: set your own."""
    STUDY = 'requirements'
    TITLE = 'Requirements'
    HEADLINE = 'Requirements that pass'

    def __init__(self, ctx):
        self._uid = ''
        self._building = False
        self.vdoc = None
        self._preview_uid = ''
        self._stale = False
        self._vstale = False            # the variants compare is from before the last edit
        super().__init__(ctx)
        self.b_apply.setText('Use the best variant')
        self.b_apply.setToolTip('Make the variant that passes most requirements the active one (one undo step)')
        self.b_preview.setToolTip('Show the best variant\'s sensors in the 3D view next to the active ones')
        st = ctx.store
        st.changed.connect(self._changed)
        st.project_replaced.connect(self._replaced)
        if VARIANTS in ctx.analysis.results:
            self._finished(VARIANTS, ctx.analysis.results[VARIANTS])
        self.refresh()

    # ---------------------------------------------------------------- layout
    def build_body(self, layout):
        M.load_metrics()
        sec = Section('Requirements')
        self.metric_box = QComboBox()
        for m in RQ.catalogue():
            self.metric_box.addItem(f'{m["title"]} ({m["unit"]}) · {m["study_title"]}', m['id'])
            self.metric_box.setItemData(self.metric_box.count() - 1, f'{m["id"]}: {m["note"] or m["title"]}',
                                        Qt.ItemDataRole.ToolTipRole)
        self.b_add = button('Add', icon_name='plus', tip='Add a requirement on the chosen metric (one undo step)')
        self.b_add.clicked.connect(lambda: self.add(self.metric_box.currentData()))
        self.b_defaults = button('Add the defaults', role='ghost', tip='Add the labelled defaults of the readiness '
                                 'scorecard that are not in the list yet (one undo step)')
        self.b_defaults.clicked.connect(self.add_defaults)
        self.b_del = button('Remove', role='ghost', icon_name='trash', tip='Remove the selected requirement')
        self.b_del.clicked.connect(lambda: self.remove())
        self.b_import = button('Import…', role='ghost', icon_name='import', tip='Add the requirements of a YAML file '
                               '(the format dexvantage --check reads)')
        self.b_import.clicked.connect(lambda: self.import_file())
        self.b_save = button('Save as YAML…', role='ghost', icon_name='export',
                             tip='Write the list as a requirements file for dexvantage --check')
        self.b_save.clicked.connect(lambda: self.save_yaml())
        sec.add(row(self.metric_box, self.b_add, self.b_defaults, self.b_del, self.b_import, self.b_save))
        self.table = QTreeWidget()
        self.table.setColumnCount(6)
        self.table.setHeaderLabels(['Requirement', 'Needs', 'Value', 'Margin', 'Status', 'Study'])
        self.table.setRootIsDecorated(False)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        hdr = self.table.header()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for k in range(1, 6):
            hdr.setSectionResizeMode(k, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._picked)
        sec.add(self.table)
        # the editor of the selected requirement
        self.e_name = QLineEdit()
        self.e_name.editingFinished.connect(lambda: self.set_field('name', self.e_name.text()))
        self.e_op = QComboBox()
        for op in RQ.OPS:
            self.e_op.addItem(f'{RQ.OP_TEXT[op]}  {RQ.OP_WORDS[op]}', op)
        self.e_op.activated.connect(lambda _i: self.set_field('op', self.e_op.currentData()))
        self.e_thr = QLineEdit()
        self.e_thr.setFont(mono(11))
        self.e_thr.editingFinished.connect(lambda: self.set_field('threshold', self.e_thr.text()))
        self.e_unit = label('', muted=True, mono_=True)
        self.e_default = QCheckBox('DexVantage default')
        self.e_default.setToolTip('A documented starting point, not a standard: unticked once you set your own')
        self.e_default.setEnabled(False)
        self.e_note = QLineEdit()
        self.e_note.setPlaceholderText('where the figure comes from (your programme\'s requirement, a paper …)')
        self.e_note.editingFinished.connect(lambda: self.set_field('note', self.e_note.text()))
        self.e_metric = label('', muted=True, wrap=True)
        sec.row('Name', self.e_name)
        sec.row('Needs', self.e_op, self.e_thr, self.e_unit)
        sec.row('', self.e_default)
        sec.row('Note', self.e_note)
        sec.add(self.e_metric)
        layout.addWidget(sec)
        # variants compare
        vs = Section('Variants compare')
        self.b_compare = button('Compare variants', icon_name='compare', tip='Evaluate every requirement on every '
                                'variant (each study once per variant, in the background)')
        self.b_compare.clicked.connect(self.compare)
        self.b_copy = button('New variant from this one', role='ghost', icon_name='plus', tip='A copy of the active '
                             'variant to change and compare (its sensors keep their wiring; one undo step)')
        self.b_copy.clicked.connect(lambda: self.copy_variant())
        self.v_note = label('', muted=True, wrap=True)
        vs.add(row(self.b_compare, self.b_copy, self.v_note))
        self._show_uid = ''                 # a variant drawn beside the active one in the 3D view (view state)
        self.show_box = QComboBox()
        self.show_box.setToolTip('Draw another variant\'s cameras (dashed frusta) beside the active variant\'s in the '
                                 '3D view, with a line from each sensor to its counterpart of the same name')
        self.show_box.currentIndexChanged.connect(lambda _i: self._show_changed())
        vs.add(row(label('Show in 3D', ink2=True), self.show_box))
        self.matrix = QTreeWidget()
        self.matrix.setRootIsDecorated(False)
        self.matrix.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        vs.add(self.matrix)
        self.v_rank = label('', wrap=True)
        vs.add(self.v_rank)
        layout.addWidget(vs)

    # ---------------------------------------------------------------- state
    def current(self) -> Requirement | None:
        rs = self.ctx.store.project.requirements
        r = next((x for x in rs if x.uid == self._uid), None) or (rs[0] if rs else None)
        self._uid = r.uid if r is not None else ''
        return r

    def _fill_show(self):
        p = self.ctx.store.project
        self.show_box.blockSignals(True)
        self.show_box.clear()
        self.show_box.addItem('Only the active variant', '')
        for v in p.variants:
            if v.uid != p.active_uid:
                self.show_box.addItem(f'{v.name} beside it', v.uid)
        k = self.show_box.findData(self._show_uid)
        if k < 0:
            self._show_uid = ''
        self.show_box.setCurrentIndex(max(0, k))
        self.show_box.blockSignals(False)

    def _show_changed(self):
        self._show_uid = self.show_box.currentData() or ''
        if self.ctx.window is not None:
            self.ctx.window.viewport.update()

    def _replaced(self):
        self.doc = None
        self.vdoc = None
        self._preview_uid = ''
        self._show_uid = ''
        self._uid = ''
        self.refresh()

    def _changed(self, sections):
        if sections & {'requirements', 'variants', 'variant', 'analysis', 'robots', 'workspace', 'tasks', 'rig',
                       'cables', 'policies'}:
            self._stale = self.doc is not None
            self._vstale = self.vdoc is not None
            self.refresh()

    def _picked(self):
        if self._building:
            return
        its = self.table.selectedItems()
        if its and its[0].data(0, ROLE) and its[0].data(0, ROLE) != self._uid:
            self._uid = its[0].data(0, ROLE)
            self._fill_editor()

    def _finished(self, sid, doc):
        if sid == VARIANTS:
            self.vdoc = doc
            self._vstale = False
            self._fill_matrix()
            self._suggest()
            return
        if sid == self.STUDY:
            self._stale = False
        super()._finished(sid, doc)

    def _failed(self, sid, message):
        if sid == VARIANTS:
            from ..store import failure_text
            text, _ = failure_text(message, 'The variants compare')
            self.v_note.setText(text)
            return
        super()._failed(sid, message)

    # ---------------------------------------------------------------- edits (one undo step each)
    def add(self, mid):
        if not mid:
            return None
        m = M.get(mid)
        r = Requirement(name=m.title, metric=mid, op='>=' if m.better == 'higher' else '<=', threshold=0.0,
                        unit=m.unit, study=m.study, is_default=False, note='')
        self.ctx.store.edit('requirements', lambda p: p.requirements.append(r), f'Add requirement {m.title}')
        self._uid = r.uid
        self.refresh()
        return r

    def add_defaults(self):
        have = {(r.metric, r.op) for r in self.ctx.store.project.requirements}
        new = [r for r in RQ.default_requirements() if (r.metric, r.op) not in have]
        if not new:
            self._msg('Every default is in the list already.')
            return 0
        self.ctx.store.edit('requirements', lambda p: p.requirements.extend(new),
                            f'Add {len(new)} default requirement{"s" if len(new) != 1 else ""}')
        self._uid = new[0].uid
        self.refresh()
        return len(new)

    def remove(self, uid=None):
        r = self.current() if uid is None else next((x for x in self.ctx.store.project.requirements
                                                     if x.uid == uid), None)
        if r is None:
            return
        self.ctx.store.edit('requirements', lambda p: p.requirements.remove(
            next(x for x in p.requirements if x.uid == r.uid)), f'Remove requirement {r.name}')
        self._uid = ''
        self.refresh()

    def set_field(self, key, value, uid=None):
        """Change one field of a requirement (one undo step); a threshold that is not a number is refused in words."""
        if self._building:
            return False
        r = self.current() if uid is None else next((x for x in self.ctx.store.project.requirements
                                                     if x.uid == uid), None)
        if r is None:
            return False
        if key == 'threshold':
            v = RQ._num(str(value).replace(',', '.').replace('−', '-'))
            if v is None:
                unit = r.unit or "the metric's unit"
                self._msg(f'"{value}" is not a number: give the threshold in {unit}.')
                self._fill_editor()
                return False
            value = v
        if key == 'op' and value not in RQ.OPS:
            return False
        if getattr(r, key) == value:
            return False
        uid0 = r.uid

        def fn(p):
            x = next(x for x in p.requirements if x.uid == uid0)
            setattr(x, key, value)
            if key in ('threshold', 'op'):
                x.is_default = False                      # the user's own figure now
        words = {'name': 'Rename', 'op': 'Change the comparison of', 'threshold': 'Change the threshold of',
                 'note': 'Change the note of'}[key]
        self.ctx.store.edit('requirements', fn, f'{words} requirement {r.name}')
        self.refresh()
        return True

    def import_file(self, path=None):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, 'Import requirements', '', 'Requirements (*.yaml *.yml '
                                                  '*.json)')
            if not path:
                return 0
        try:
            new = RQ.load(path)
        except RQ.RequirementsError as e:
            self._msg(str(e))
            return 0
        self.ctx.store.edit('requirements', lambda p: p.requirements.extend(new),
                            f'Import {len(new)} requirement{"s" if len(new) != 1 else ""}')
        self.refresh()
        return len(new)

    def save_yaml(self, path=None):
        rs = self.ctx.store.project.requirements
        if not rs:
            self._msg('There are no requirements to save.')
            return None
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, 'Save requirements', 'requirements.yaml', 'YAML (*.yaml)')
            if not path:
                return None
        try:
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(RQ.dumps(rs))
        except OSError as e:
            self._msg(f'Could not write the file: {e.strerror or e}.')
            return None
        self._msg(f'Saved {len(rs)} requirements: check them with dexvantage PROJECT --check FILE.')
        return path

    def copy_variant(self):
        """A copy of the active variant, made active (one undo step). Its sensors keep their uids, so the host links,
        cables and policy cameras that name them apply to the copy too."""
        from ...core.project import Variant, new_uid
        src = self.ctx.store.project.variant
        v = Variant.from_dict(src.to_dict())
        v.uid = new_uid()
        names = {x.name for x in self.ctx.store.project.variants}
        k = 2
        while f'{src.name} ({k})' in names:
            k += 1
        v.name = f'{src.name} ({k})'

        def fn(p):
            p.variants.append(v)
            p.active_uid = v.uid
        self.ctx.store.edit('variants', fn, f'New variant {v.name}')
        return v

    def compare(self):
        self.v_note.setText('Comparing the variants…')
        self.ctx.analysis.run(VARIANTS)

    def apply(self):
        """Make the best variant the active one (one undo step)."""
        best = self._best_variant()
        if best is None:
            return False
        name = next((v['name'] for v in self.vdoc['variants'] if v['uid'] == best), best)

        def fn(p):
            p.active_uid = best
        self.ctx.store.edit('variants', fn, f'Use variant {name} (passes most requirements)')
        self._preview_uid = ''
        self._suggest()
        return True

    def preview(self):
        best = self._best_variant()
        self._preview_uid = '' if self._preview_uid or best is None else best
        if self.ctx.window is not None:
            self.ctx.window.viewport.update()

    def _best_variant(self):
        d = self.vdoc
        if not d or not d.get('ranking'):
            return None
        best = d['ranking'][0]
        cur = self.ctx.store.project.active_uid
        if best == cur or self.ctx.store.project.variant_by_uid(best) is None:
            return None
        counts = {v['uid']: v['counts'] for v in d['variants']}
        a, b = counts.get(best, {}), counts.get(cur, {})
        if (a.get('pass', 0), -a.get('fail', 0)) <= (b.get('pass', 0), -b.get('fail', 0)):
            return None
        return best

    def _suggest(self):
        self.suggestion_ready(self._best_variant() is not None)

    def _msg(self, text):
        if self.ctx.window is not None:
            self.ctx.window.msg(text, 8000)
        self.note.setText(text)

    # ---------------------------------------------------------------- filling
    def empty(self):
        """refresh() fills the headline."""

    def show_result(self, doc):
        self.refresh()

    def refresh(self):
        self._building = True
        try:
            self._fill_table()
            self._fill_editor()
            self._headline()
            self._fill_matrix()
            self._fill_show()
            self._suggest()
        except Exception:                                 # noqa: BLE001 — explained, the traceback logged
            traceback.print_exc()
            self.head.set('—', '', 'critical', 'could not show', 'The requirements could not be shown: details are '
                          'in the log.')
        finally:
            self._building = False

    def _items(self) -> dict:
        return {it['uid']: it for it in (self.doc or {}).get('items') or []}

    def _fill_table(self):
        p = self.ctx.store.project
        res = self._items()
        cur = self.current()
        self.table.clear()
        for r in p.requirements:
            it_res = res.get(r.uid)
            need = f'{RQ.OP_TEXT.get(r.op, r.op)} {RQ._fmt(r.threshold, r.unit)}' + (' (default)' if r.is_default
                                                                                    else '')
            fresh = it_res is not None and (it_res['op'], it_res['threshold'], it_res['metric']) == \
                (r.op, r.threshold, r.metric)
            if fresh:
                val = RQ._fmt(it_res['value'], r.unit)
                if it_res['bound'] and it_res['value'] is not None:
                    val = ('≥ ' if it_res['bound'] == 'at_least' else '≤ ') + val
                mg = '—' if it_res['margin'] is None else ('+' if it_res['margin'] >= 0 else '−') + \
                    RQ._fmt(abs(it_res['margin']), r.unit)
                status, level = it_res['status_text'], it_res['level']
            else:
                val, mg, status, level = '—', '—', 'not run', 'warning'
            study = (it_res or {}).get('study_title') or r.study
            row_ = QTreeWidgetItem([r.name or r.metric, need, val, mg, status, study])
            row_.setData(0, ROLE, r.uid)
            for k in (1, 2, 3):
                row_.setFont(k, mono(11))
                row_.setTextAlignment(k, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row_.setForeground(4, _col(level))
            tip = (it_res or {}).get('text') if fresh else 'Run the scorecard to evaluate this requirement.'
            for k in range(6):
                row_.setToolTip(k, tip or '')
            if fresh and (it_res.get('reason') or it_res.get('figure_note')):
                row_.setToolTip(4, ' '.join(x for x in (it_res.get('reason'), it_res.get('figure_note')) if x))
            self.table.addTopLevelItem(row_)
            if cur is not None and r.uid == cur.uid:
                row_.setSelected(True)
        n = max(1, min(self.table.topLevelItemCount(), 10))
        self.table.setFixedHeight(self.table.header().sizeHint().height() + n * 22 + 6)
        self.b_del.setEnabled(cur is not None)
        self.b_save.setEnabled(bool(p.requirements))

    def _fill_editor(self):
        r = self.current()
        was = self._building
        self._building = True
        try:
            on = r is not None
            for w in (self.e_name, self.e_op, self.e_thr, self.e_note):
                w.setEnabled(on)
            if not on:
                for w in (self.e_name, self.e_thr, self.e_note):
                    w.setText('')
                self.e_unit.setText('')
                self.e_default.setChecked(False)
                self.e_metric.setText('Add a requirement, or the defaults, to edit it here.')
                return
            m = M.METRICS.get(r.metric)
            self.e_name.setText(r.name)
            self.e_op.setCurrentIndex(max(0, list(RQ.OPS).index(r.op) if r.op in RQ.OPS else 0))
            self.e_thr.setText(f'{r.threshold:g}')
            self.e_unit.setText('' if (r.unit or '') == 'count' else r.unit)
            self.e_default.setChecked(bool(r.is_default))
            self.e_note.setText(r.note)
            if m is None:
                self.e_metric.setText(f'{r.metric}: this build has no such metric, so the requirement cannot be '
                                      'evaluated.')
            else:
                self.e_metric.setText(f'{m.id}: {m.title}, {"higher" if m.better == "higher" else "lower"} is '
                                      f'better, measured by the {m.study} study.' + (f' {m.note[0].upper()}'
                                                                                      f'{m.note[1:]}.' if m.note
                                                                                      else ''))
        finally:
            self._building = was

    def _headline(self):
        p = self.ctx.store.project
        if not p.requirements:
            self.head.set('—', '', 'warning', 'no requirements', 'Add the defaults or your own: each a study\'s '
                          'figure against a threshold, for the readiness scorecard and dexvantage --check.')
            return
        if self.doc is None:
            self.head.set('—', '', 'warning', 'not run yet', f'{len(p.requirements)} requirement'
                          f'{"s" if len(p.requirements) != 1 else ""}: Run evaluates them on the active variant.')
            return
        h = self.doc['headline']
        text = h['text'] + (' The list or the design changed since: Run again.' if self._stale else '')
        self.head.set(h['value'], 'pass', h['level'], h['word'], text,
                      f'Scorecard of {self.doc.get("variant")}: {self.doc["counts"]["pass"]} pass, '
                      f'{self.doc["counts"]["fail"]} fail, {self.doc["counts"]["not_evaluated"]} not evaluated.')

    def _fill_matrix(self):
        d = self.vdoc
        self.matrix.clear()
        if not d:
            self.matrix.setColumnCount(1)
            self.matrix.setHeaderLabels(['Requirement'])
            self.v_rank.setText('')
            if not self.v_note.text():
                self.v_note.setText('Every variant against every requirement, the best per requirement marked.')
            self.matrix.setFixedHeight(self.matrix.header().sizeHint().height() + 6)
            return
        vs = d['variants']
        self.matrix.setColumnCount(1 + len(vs))
        self.matrix.setHeaderLabels(['Requirement'] + [v['name'] for v in vs])
        hdr = self.matrix.header()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for k in range(1, len(vs) + 1):
            hdr.setSectionResizeMode(k, QHeaderView.ResizeMode.ResizeToContents)
        for j, rq in enumerate(d['requirements']):
            cells = []
            for i in range(len(vs)):
                c = d['cells'][i][j]
                val = RQ._fmt(c['value'], rq['unit'])
                if c['bound'] and c['value'] is not None:
                    val = ('≥ ' if c['bound'] == 'at_least' else '≤ ') + val
                best = vs[i]['uid'] in (d['best'].get(rq['uid']) or [])
                cells.append(f'{val}  {RQ.STATUS_TEXT[c["status"]]}' + ('  · best' if best else ''))
            it = QTreeWidgetItem([f'{rq["name"]} {rq["op_text"]} {RQ._fmt(rq["threshold"], rq["unit"])}'] + cells)
            for i in range(len(vs)):
                c = d['cells'][i][j]
                it.setFont(i + 1, mono(11))
                it.setForeground(i + 1, _col(RQ.STATUS_LEVEL[c['status']]))
                it.setToolTip(i + 1, c.get('reason') or RQ.STATUS_TEXT[c['status']])
            self.matrix.addTopLevelItem(it)
        n = max(1, min(len(d['requirements']), 10))
        self.matrix.setFixedHeight(self.matrix.header().sizeHint().height() + n * 22 + 6)
        names = {v['uid']: v for v in vs}
        rank = [names[u] for u in d['ranking'] if u in names]
        self.v_rank.setText('Ranking by requirements passed: ' + ', '.join(
            f'{v["name"]} {v["counts"]["pass"]} of {v["counts"]["total"]} ({v["verdict_text"]})' for v in rank)
            + ('.' if rank else ''))
        self.v_note.setText(f'{len(vs)} variant{"s" if len(vs) != 1 else ""} × {len(d["requirements"])} '
                            f'requirement{"s" if len(d["requirements"]) != 1 else ""}' +
                            (' · from before the last edit: compare again.' if self._vstale else ''))

    # ---------------------------------------------------------------- 3D view
    def overlays(self, vp) -> list:
        from ...core import scenegeom as SG
        from ..variant_ghost import frustum_segments
        p = self.ctx.store.project
        if self.doc is None and not self._preview_uid and not self._show_uid:
            return []
        q = self.ctx.store.q
        size = 0.03

        def cross(c):
            e = np.eye(3) * size
            return np.stack([np.stack([c - e[k], c + e[k]]) for k in range(3)])

        def positions(v):
            out = {}
            for s in v.sensors:
                try:
                    out[s.name] = np.asarray(SG.sensor_world_pose(p, s, q).position, float)
                except Exception:                         # noqa: BLE001 — a sensor that cannot be posed is not drawn
                    continue
            return out
        items = []
        here = positions(p.variant)
        level = (self.doc or {}).get('level', 'warning')
        for c in here.values():
            items.append(LineDraw(cross(c), _col(level), 2.0, segments=True, depth=False))
        other = self._preview_uid or self._show_uid
        v2 = p.variant_by_uid(other) if other and other != p.active_uid else None
        if v2 is not None:
            there = positions(v2)
            col = QColor(SEQ[8])                            # the sequential ramp's blue: the preview
            key = (self.ctx.store.revision, v2.uid, tuple((k, np.asarray(v).tobytes()) for k, v in sorted(q.items())))
            if getattr(self, '_ghost', (None,))[0] != key:          # once per edit and pose, not per frame
                self._ghost = (key, frustum_segments(p, v2, q))
            for segs in self._ghost[1].values():
                items.append(LineDraw(segs, col, 1.2, dash=True, segments=True, dash_len=0.01))
            for name, c in there.items():
                items.append(LineDraw(cross(c), col, 2.0, segments=True, depth=False))
                if name in here and np.linalg.norm(here[name] - c) > 1e-6:
                    items.append(LineDraw(np.stack([here[name], c]), col, 1.0, dash=True, depth=False))
        return items


register('requirements', 'Requirements and variants', RequirementsPage, menu='Analysis', icon='checklist',
         workspace='review', order=95)
