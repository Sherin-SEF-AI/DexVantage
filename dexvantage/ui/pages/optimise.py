"""The Optimise page (spec §7.11): the optimiser ("Improve this rig": NSGA-II over the rig's mount shifts, turns,
cameras kept and devices) and the design agent ("Design new rigs": complete rigs from the library for the project's
requirements), each a Pareto front of designs.

What it shows: how many designs the front offers, with a status word (better rig found, trade-offs only, misses the
requirements); a scatter of every design scored for any two objectives (grey: scored; hollow: misses a requirement;
labelled squares: the designs shown; the circle: the current rig); a table of the designs with their figures, the
requirements they meet and what they change; the selected design's plain-English explanation of its trade-offs,
computed from its figures; what was left out of the search and why. Accept makes the selected design the rig (one
undo step: the optimiser sets its factors on the active variant, the design agent adds it as a new variant and makes
it active); Preview draws it in the 3D view (its cameras' frustums, dashed) without changing anything; Export writes
the result as JSON with provenance. Run starts the search in the background (with progress); Stop cancels it and
keeps the designs scored so far.

3D view: the selected design's cameras (frustums) while previewing, and a cross at every camera of the other designs
shown. Camera view: the selected design's cameras as labelled marks where they fall in the image.
"""
from __future__ import annotations

import math
import traceback

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QComboBox, QHeaderView, QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem

from ..charts import ChartBase
from ..theme import THEME, mono, sans
from ..widgets import ScrubSpinBox, SegmentedControl, StatusLine, ToggleSwitch, button, label
from . import register
from .base import StudyPage, row

STUDIES = ('optimise', 'design_agent')
MODE_TITLES = {'optimise': 'Improve this rig', 'design_agent': 'Design new rigs'}
PREVIEW_RGBA = '#2f7de1'


def _fmt(v, unit):
    from ...core.optimise.search import fmt
    return fmt(v, unit) if v is not None else '—'


# ====================================================================== the scatter
class ParetoScatter(ChartBase):
    """Every design scored, for two objectives: grey dots (meets the requirements) and hollow ones (misses one); the
    designs shown labelled; the current rig circled. Click a labelled design to select it."""
    picked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pts = np.zeros((0, 2))
        self.ok = np.zeros(0, bool)
        self.labels = []                 # [(x, y, label)]
        self.current = None
        self.selected = None
        self.xlabel = self.ylabel = ''
        self.xunit = self.yunit = ''
        self.setMinimumHeight(220)

    def set_data(self, pts, ok, labels, current, xlabel, ylabel, xunit, yunit, selected=None):
        self.pts = np.asarray(pts, float).reshape(-1, 2)
        self.ok = np.asarray(ok, bool).reshape(-1)
        self.labels = list(labels)
        self.current = current
        self.xlabel, self.ylabel, self.xunit, self.yunit = xlabel, ylabel, xunit, yunit
        self.selected = selected
        self.update()

    def _plot(self):
        return QRectF(56, 30, max(10.0, self.width() - 72), max(10.0, self.height() - 62))

    def _range(self):
        allp = [p for p in self.pts if np.isfinite(p).all()] + [(x, y) for x, y, _ in self.labels]
        if self.current is not None and all(v is not None for v in self.current):
            allp.append(tuple(self.current))
        if not allp:
            return 0.0, 1.0, 0.0, 1.0
        a = np.asarray(allp, float)
        out = []
        for k, unit in enumerate((self.xunit, self.yunit)):
            lo, hi = float(a[:, k].min()), float(a[:, k].max())
            pad = (hi - lo) * 0.08 if hi > lo else max(abs(hi) * 0.1, 0.5)
            lo, hi = lo - pad, hi + pad
            if unit == '%':                      # a share is never below 0 (the top keeps its pad for the labels)
                lo = max(lo, 0.0)
            out += [lo, hi]
        return tuple(out)

    def _map(self, x, y):
        r = self._plot()
        x0, x1, y0, y1 = self._range()
        return QPointF(r.left() + (x - x0) / (x1 - x0) * r.width(), r.bottom() - (y - y0) / (y1 - y0) * r.height())

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.paint_title(p)
        r = self._plot()
        x0, x1, y0, y1 = self._range()
        p.setFont(mono(10))
        from ...core.figures import nice_ticks, tick_text
        pct = (self.xunit == '%', self.yunit == '%')
        for t in nice_ticks(x0, x1, 5):                 # round values, as the publication figures
            if pct[0] and t > 100:
                continue
            gx = r.left() + (t - x0) / (x1 - x0) * r.width()
            p.setPen(QPen(THEME.color('hair'), 1))
            p.drawLine(QPointF(gx, r.top()), QPointF(gx, r.bottom()))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(gx - 34, r.bottom() + 2, 68, 12), Qt.AlignmentFlag.AlignCenter, tick_text(t))
        for t in nice_ticks(y0, y1, 4):
            if pct[1] and t > 100:
                continue
            gy = r.bottom() - (t - y0) / (y1 - y0) * r.height()
            p.setPen(QPen(THEME.color('hair'), 1))
            p.drawLine(QPointF(r.left(), gy), QPointF(r.right(), gy))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, gy - 6, r.left() - 6, 12), Qt.AlignmentFlag.AlignRight, tick_text(t))
        p.setFont(sans(10))
        p.setPen(THEME.color('muted'))
        p.drawText(QRectF(r.left(), r.top() - 16, r.width(), 14), Qt.AlignmentFlag.AlignLeft,
                   f'{self.ylabel} ({self.yunit})' if self.yunit else self.ylabel)
        p.drawText(QRectF(r.left(), r.bottom() + 15, r.width(), 14), Qt.AlignmentFlag.AlignRight,
                   f'{self.xlabel} ({self.xunit})' if self.xunit else self.xlabel)
        p.setPen(QPen(THEME.color('strong'), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(r)
        for (x, y), ok in zip(self.pts, self.ok):
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            q = self._map(x, y)
            if ok:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(THEME.color('muted'))
            else:
                p.setPen(QPen(THEME.color('muted'), 1))
                p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(q, 2.2, 2.2)
        for i, (x, y, lab) in enumerate(self.labels):
            q = self._map(x, y)
            sel = self.selected == i
            p.setPen(QPen(THEME.color('ink'), 1))
            p.setBrush(THEME.color('ink') if sel else THEME.color('panel'))
            p.drawRect(QRectF(q.x() - 3.5, q.y() - 3.5, 7, 7))
            p.setFont(mono(10))
            p.drawText(QPointF(q.x() + 6, q.y() - 5), lab)
        if self.current is not None and all(v is not None for v in self.current):
            q = self._map(*self.current)
            p.setPen(QPen(THEME.color('ink'), 1.2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(q, 6, 6)
            p.setFont(mono(10))
            p.drawText(QPointF(q.x() + 8, q.y() + 14), 'current')
        self.paint_tip(p)
        p.end()

    def _hit(self, pos):
        best = None
        for i, (x, y, _lab) in enumerate(self.labels):
            q = self._map(x, y)
            d = math.hypot(q.x() - pos.x(), q.y() - pos.y())
            if d < 12 and (best is None or d < best[1]):
                best = (i, d)
        return None if best is None else best[0]

    def mousePressEvent(self, e):
        i = self._hit(e.position())
        if i is not None:
            self.picked.emit(i)

    def mouseMoveEvent(self, e):
        i = self._hit(e.position())
        old = self.tip_text()
        self.tip = None
        if i is not None:
            x, y, lab = self.labels[i]
            self.set_tip(self._map(x, y), f'{_fmt(x, self.xunit)} · {_fmt(y, self.yunit)}', f'design {lab}',
                         f'{self.xlabel} · {self.ylabel}', 'click to select')
        if self.tip_text() != old:
            self.update()


def _tree(headers) -> QTreeWidget:
    t = QTreeWidget()
    t.setRootIsDecorated(False)
    t.setHeaderLabels(headers)
    t.setUniformRowHeights(True)
    t.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    t.setMinimumHeight(120)
    return t


# ====================================================================== the page
class OptimisePage(StudyPage):
    """Optimise: an evolutionary multi-objective search (NSGA-II) for rigs that do better on contact visibility,
    depth-valid coverage, camera count and cost, bandwidth margin, bracket printability and VLA token coverage, within
    the project's requirements, locked sensors never changed. "Improve this rig" moves, turns, keeps or swaps the
    current cameras; "Design new rigs" builds complete rigs from the library on static frames and the robot's
    gripper. Each design on the Pareto front is explained from its figures; nothing changes until you accept one."""
    STUDY = 'optimise'
    TITLE = 'Optimise'
    HEADLINE = 'Designs on the Pareto front'

    def build_body(self, layout):
        self.docs: dict = {}
        self.sel_label = ''
        self._preview = None              # (study, label, [(name, segments, centre)])
        self._busy = False
        self.b_apply.setText('Accept')
        self.b_apply.setToolTip('Make the selected design the rig (one undo step)')
        self.b_preview.setToolTip('Draw the selected design in the 3D view without changing anything')
        self.b_stop = button('Stop', role='ghost', icon_name='stop', tip='Cancel the search, keeping the designs '
                                                                          'scored so far')
        self.b_stop.clicked.connect(lambda: self.ctx.analysis.cancel(self.STUDY))
        self.b_stop.setEnabled(False)
        self.head.actions.insertWidget(1, self.b_stop)
        self.mode = SegmentedControl([(k, MODE_TITLES[k]) for k in STUDIES], 'optimise')
        self.mode.changed.connect(self._mode_changed)
        layout.addWidget(row(self.mode))
        # settings (each change is one undo step on the analysis settings)
        self.s_budget = self._spin(4, 20000, 8, '', 'Designs to score in all (default 96 for the optimiser, 120 for '
                                                    'the design agent)', 0)
        self.s_pop = self._spin(4, 400, 2, '', 'Designs per generation (default 12 and 16)', 0)
        self.s_seed = self._spin(0, 1_000_000, 1, '', 'The seed: the same seed and settings give the same designs', 0)
        layout.addWidget(row(label('Evaluations', ink2=True), self.s_budget, label('Population', ink2=True),
                             self.s_pop, label('Seed', ink2=True), self.s_seed))
        self.t_remove = ToggleSwitch()
        self.t_remove.setToolTip('Let the optimiser switch cameras off, so the camera count can fall (default on)')
        self.t_swap = ToggleSwitch()
        self.t_swap.setToolTip('Let the optimiser change a camera\'s device among the library\'s of its kind '
                               '(default off)')
        self.s_move = self._spin(0, 100, 1, ' cm', 'How far the optimiser may shift each mount (default 5 cm)', 1)
        self.s_turn = self._spin(0, 180, 1, '°', 'How far it may tilt and pan each camera (default 15°)', 0)
        self.s_maxcam = self._spin(1, 8, 1, '', 'The most new cameras a designed rig may have (default 3)', 0)
        self.t_confirm = ToggleSwitch()
        self.t_confirm.setToolTip('Score the designs shown again at the project\'s own study settings (slower; '
                                  'default off: the search scores at screening settings)')
        self.l_remove, self.l_swap = label('Remove cameras', ink2=True), label('Swap devices', ink2=True)
        self.l_move, self.l_turn = label('Shift', ink2=True), label('Turn', ink2=True)
        self.l_maxcam = label('Cameras at most', ink2=True)
        layout.addWidget(row(self.l_remove, self.t_remove, self.l_swap, self.t_swap, self.l_move, self.s_move,
                             self.l_turn, self.s_turn, self.l_maxcam, self.s_maxcam, label('Confirm', ink2=True),
                             self.t_confirm))
        layout.addWidget(label('Objectives', bold=True))
        self.objlist = QListWidget()
        self.objlist.setMaximumHeight(110)
        self.objlist.setToolTip('The figures the search improves (each from its study); one left out of a run says '
                                'why below')
        layout.addWidget(self.objlist)
        self.c_x, self.c_y = QComboBox(), QComboBox()
        for c in (self.c_x, self.c_y):
            c.setMinimumContentsLength(24)
            c.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.c_x.setToolTip('The objective along the scatter\'s x axis')
        self.c_y.setToolTip('The objective along the scatter\'s y axis')
        layout.addWidget(row(label('Scatter', ink2=True), self.c_x, label('against', ink2=True), self.c_y))
        self.chart = ParetoScatter()
        self.chart.picked.connect(lambda i: self._select_index(i))
        layout.addWidget(self.chart)
        self.table = _tree(['Design'])
        self.table.currentItemChanged.connect(self._row_clicked)
        layout.addWidget(self.table)
        self.explain_l = StatusLine('', '', wrap=True)
        layout.addWidget(self.explain_l)
        self.left_out = label('', muted=True, wrap=True, small=True)
        layout.addWidget(self.left_out)
        self.c_x.activated.connect(lambda _i: self._fill_chart())
        self.c_y.activated.connect(lambda _i: self._fill_chart())
        self.objlist.itemChanged.connect(self._objectives_changed)
        self.s_budget.editingFinished.connect(lambda: self._set('budget', int(self.s_budget.value())))
        self.s_pop.editingFinished.connect(lambda: self._set('population', int(self.s_pop.value())))
        self.s_seed.editingFinished.connect(lambda: self._set('seed', int(self.s_seed.value())))
        self.s_move.editingFinished.connect(lambda: self._set('move_m', round(self.s_move.value() / 100.0, 4)))
        self.s_turn.editingFinished.connect(lambda: self._set('turn_deg', float(self.s_turn.value())))
        self.s_maxcam.editingFinished.connect(lambda: self._set('max_cameras', int(self.s_maxcam.value())))
        self.t_remove.toggled.connect(lambda on: self._set('allow_remove', bool(on)))
        self.t_swap.toggled.connect(lambda on: self._set('swap_models', bool(on)))
        self.t_confirm.toggled.connect(lambda on: self._set('confirm', bool(on)))
        self.ctx.analysis.started.connect(self._started)
        self.ctx.analysis.finished.connect(self._ended)
        self.ctx.analysis.failed.connect(lambda sid, _m: self._ended(sid, None))
        st = self.ctx.store
        st.changed.connect(lambda _s: self._fill_controls())
        st.project_replaced.connect(self._replaced)
        self._fill_controls()

    def _spin(self, lo, hi, step, suffix, tip, dp):
        s = ScrubSpinBox('', 'plain', dp, suffix.strip(), step=step, width=84)
        s.setRange(lo, hi)
        s.setToolTip(tip)
        s.setKeyboardTracking(False)
        return s

    # ---------------------------------------------------------------- settings
    def _settings(self) -> dict:
        from ...core.optimise import designer as DS
        from ...core.optimise import rig as R
        p = self.ctx.store.project
        try:
            return R.settings_of(p) if self.STUDY == 'optimise' else DS.settings_of(p)
        except Exception:                                # noqa: BLE001 — the controls must not take the page down
            traceback.print_exc()
            return {}

    def _fill_controls(self):
        from ...core.optimise import objectives as OB
        st = self._settings()
        if not st:
            return
        opt = self.STUDY == 'optimise'
        for w, v in ((self.s_budget, st['budget']), (self.s_pop, st['population']), (self.s_seed, st['seed']),
                     (self.s_maxcam, st.get('max_cameras', 3)), (self.s_move, 100 * st.get('move_m', 0.05)),
                     (self.s_turn, st.get('turn_deg', 15.0))):
            w.blockSignals(True)
            w.setValue(float(v))
            w.blockSignals(False)
        for t, v in ((self.t_remove, st.get('allow_remove', True)), (self.t_swap, st.get('swap_models', False)),
                     (self.t_confirm, st['confirm'])):
            t.blockSignals(True)
            t.setChecked(bool(v))
            t.blockSignals(False)
        for w in (self.l_remove, self.t_remove, self.l_swap, self.t_swap, self.l_move, self.s_move, self.l_turn,
                  self.s_turn):
            w.setVisible(opt)
        self.l_maxcam.setVisible(not opt)
        self.s_maxcam.setVisible(not opt)
        chosen = set(st['objectives'])
        self.objlist.blockSignals(True)
        self.objlist.clear()
        try:
            avail = OB.available()
        except Exception:                                # noqa: BLE001
            traceback.print_exc()
            avail = []
        ids = [a['id'] for a in avail]
        for mid in st['objectives']:                     # named in the settings but not registered: still shown
            if mid not in ids:
                avail.append({'id': mid, 'title': f'{mid} (no such metric)', 'unit': '', 'better': ''})
        for a in avail:
            if a['id'] not in OB.DEFAULT_OBJECTIVES and a['id'] not in chosen:
                continue
            it = QListWidgetItem(f'{a["title"]} ({a["unit"]}, {a["better"]} is better)' if a['better'] else
                                 a['title'])
            it.setData(Qt.ItemDataRole.UserRole, a['id'])
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if a['id'] in chosen else Qt.CheckState.Unchecked)
            self.objlist.addItem(it)
        self.objlist.blockSignals(False)

    def _set(self, key, value):
        sid = self.STUDY
        cur = dict(self.ctx.store.project.analysis.studies.get(sid) or {})
        if cur.get(key) == value:
            return

        def fn(p):
            p.analysis.studies[sid] = {**(p.analysis.studies.get(sid) or {}), key: value}
        self.ctx.store.edit('analysis', fn, f'{MODE_TITLES[sid]}: {key.replace("_", " ")} {value}')

    def _objectives_changed(self, _item):
        ids = [self.objlist.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.objlist.count())
               if self.objlist.item(i).checkState() == Qt.CheckState.Checked]
        if not ids:
            if self.ctx.window is not None:
                self.ctx.window.msg('Keep at least one objective.', 4000)
            self._fill_controls()
            return
        self._set('objectives', ids)

    # ---------------------------------------------------------------- mode and runs
    def _mode_changed(self, sid):
        if sid == self.STUDY:
            return
        self.STUDY = sid
        self._preview = None
        self.sel_label = ''
        self._fill_controls()
        doc = self.docs.get(sid) or self.ctx.analysis.results.get(sid)
        self.doc = doc
        self.b_stop.setEnabled(self.ctx.analysis.busy(sid))
        if doc is not None:
            self.show_result(doc)
        else:
            self.empty()
        self._redraw()

    def empty(self):
        self.head.set('—', '', 'warning', 'not run yet', f'Run to search for designs ({MODE_TITLES[self.STUDY].lower()}'
                      '). The search runs in the background; Stop keeps what it has.')
        self.table.clear()
        self.chart.set_data([], [], [], None, '', '', '', '')
        self.explain_l.set('', '')
        self.left_out.setText('')
        self.suggestion_ready(False)

    def _started(self, sid):
        if sid == self.STUDY:
            self.b_stop.setEnabled(True)

    def _ended(self, sid, _doc):
        if sid == self.STUDY:
            self.b_stop.setEnabled(False)

    def _finished(self, sid, doc):
        if sid not in STUDIES:
            return
        self.docs[sid] = doc
        if sid == self.STUDY:
            super()._finished(sid, doc)

    def _replaced(self):
        self.docs.clear()
        self.doc = None
        self._preview = None
        self.sel_label = ''
        self._fill_controls()
        self.empty()

    # ---------------------------------------------------------------- the result
    def show_result(self, doc: dict):
        self._preview = None
        h = doc.get('headline') or {}
        self.head.set(str(h.get('value', '—')), h.get('unit', ''), h.get('status', 'warning'),
                      h.get('status_word', ''), h.get('text', ''),
                      tip=f'{doc["search"]["evaluations"]} designs scored in {doc["search"]["seconds"]:.1f} s, '
                          f'seed {doc["search"]["seed"]} ({doc["search"]["reused"]} reused)')
        objs = doc.get('objectives') or []
        for c in (self.c_x, self.c_y):
            keep = c.currentData()
            c.blockSignals(True)
            c.clear()
            for o in objs:
                c.addItem(o['title'], o['id'])
            c.blockSignals(False)
            k = c.findData(keep)
            if k >= 0:
                c.setCurrentIndex(k)
        if self.c_x.currentData() == self.c_y.currentData() and len(objs) > 1:
            self.c_y.setCurrentIndex(1 if self.c_x.currentIndex() != 1 else 0)
        cons = doc.get('constraints') or []
        heads = ['Design'] + [o['title'] for o in objs] + (['Requirements'] if cons else []) + \
            ['Changes' if doc.get('kind') == 'optimise' else 'Rig']
        self.table.blockSignals(True)
        self.table.clear()
        self.table.setHeaderLabels(heads)
        rows = [('current', doc['current'])] + [(o['label'], o) for o in doc.get('options') or []]
        for lab, o in rows:
            met = sum(v['met'] for v in o.get('violations') or [])
            vals = [_fmt(o['figures'].get(x['id']), x['unit']) for x in objs]
            what = ('as it is' if lab == 'current' else
                    '; '.join(o.get('changes') or []) or o.get('summary', '')) if doc.get('kind') == 'optimise' else \
                (o.get('summary') or '')
            it = QTreeWidgetItem([lab] + vals + ([f'{met} of {len(cons)}'] if cons else []) + [what])
            it.setData(0, Qt.ItemDataRole.UserRole, lab)
            for k in range(1, len(vals) + 1):
                it.setFont(k, mono(11))
            it.setToolTip(len(heads) - 1, what)
            self.table.addTopLevelItem(it)
        self.table.blockSignals(False)
        notes = [f'{d["title"]} left out: {d["reason"]}.' for d in doc.get('dropped') or []]
        notes += [f'Requirement {s["name"]} not used: {s["why"]}.' for s in doc.get('skipped_requirements') or []]
        notes += list(doc.get('notes') or [])
        if doc.get('locked'):
            notes.append(f'Locked, never changed: {", ".join(doc["locked"])}.')
        self.left_out.setText('\n'.join(notes))
        opts = doc.get('options') or []
        if opts:
            self._select(self.sel_label if any(o['label'] == self.sel_label for o in opts) else opts[0]['label'])
        else:
            self.sel_label = ''
            self.explain_l.set('warning', 'No design to show.')
            self._fill_chart()
        self.suggestion_ready(bool(opts))

    def _option(self):
        doc = self.doc or {}
        return next((o for o in doc.get('options') or [] if o['label'] == self.sel_label), None)

    def _select(self, lab):
        self.sel_label = lab
        o = self._option()
        if o is not None:
            word = 'good' if o['feasible'] and o.get('dominates_current') else 'warning' if o['feasible'] else \
                'critical'
            self.explain_l.set(word, f'Design {lab}: {o["explanation"]}')
        for i in range(self.table.topLevelItemCount()):
            it = self.table.topLevelItem(i)
            if it.data(0, Qt.ItemDataRole.UserRole) == lab:
                self.table.blockSignals(True)
                self.table.setCurrentItem(it)
                self.table.blockSignals(False)
        if self._preview is not None:
            self._preview = None
            self.preview()
        self._fill_chart()

    def _select_index(self, i):
        opts = (self.doc or {}).get('options') or []
        if 0 <= i < len(opts):
            self._select(opts[i]['label'])

    def _row_clicked(self, cur, _prev):
        if cur is None:
            return
        lab = cur.data(0, Qt.ItemDataRole.UserRole)
        if lab and lab != 'current':
            self._select(lab)

    def _fill_chart(self):
        doc = self.doc or {}
        objs = doc.get('objectives') or []
        ids = [o['id'] for o in objs]
        xi, yi = self.c_x.currentData(), self.c_y.currentData()
        if xi not in ids or yi not in ids:
            self.chart.set_data([], [], [], None, '', '', '', '')
            return
        a, b = ids.index(xi), ids.index(yi)
        ev = doc.get('evaluated') or []
        pts = [[np.nan if e['figures'][a] is None else e['figures'][a],
                np.nan if e['figures'][b] is None else e['figures'][b]] for e in ev]
        opts = doc.get('options') or []
        labels = [(o['figures'].get(xi), o['figures'].get(yi), o['label']) for o in opts
                  if o['figures'].get(xi) is not None and o['figures'].get(yi) is not None]
        cur = (doc['current']['figures'].get(xi), doc['current']['figures'].get(yi))
        sel = next((i for i, (_x, _y, lab) in enumerate(labels) if lab == self.sel_label), None)
        self.chart.set_data(pts, [e['feasible'] for e in ev], labels, cur, objs[a]['title'], objs[b]['title'],
                            objs[a]['unit'], objs[b]['unit'], sel)

    # ---------------------------------------------------------------- Accept / Preview
    def _design_project(self):
        """A copy of the project with the selected design in it (its variant active), or None."""
        from ...core.optimise import designer as DS
        from ...core.optimise import rig as R
        doc, o = self.doc, self._option()
        if doc is None or o is None:
            return None
        p = self.ctx.store.project
        if doc.get('kind') == 'optimise':
            if doc.get('variant_uid') != p.variant.uid or doc.get('rig_fingerprint') != R.rig_fingerprint(p):
                from ...core.jobs import StudyError
                raise StudyError('The rig changed since the search ran, so its designs no longer start from it: run '
                                 'the optimiser again.')
            return R.option_project(p, doc, o['label'])
        q = p.snapshot()
        v = DS.option_variant(q, doc, o['label'])
        q.variants.append(v)
        q.active_uid = v.uid
        return q

    def apply(self):
        from ...core import jobs
        from ...core.optimise import designer as DS
        from ...core.optimise import rig as R
        doc, o = self.doc, self._option()
        win = self.ctx.window
        if doc is None or o is None:
            return
        lab = o['label']
        try:
            if doc.get('kind') == 'optimise':
                R.accept(self.ctx.store.project.snapshot(), doc, lab)      # refuses a changed rig, here first
                self.ctx.store.edit(['variant', 'analysis'], lambda p: R.accept(p, doc, lab),
                                    f'Optimiser: design {lab}')
                text = f'Design {lab} is now the rig (Undo puts the rig back).'
            else:
                v = self.ctx.store.edit(['variants', 'analysis'], lambda p: DS.accept(p, doc, lab),
                                        f'Design agent: design {lab} as a new variant')
                text = f'Design {lab} added as the variant "{v.name}" and made active (Undo removes it).'
        except jobs.StudyError as e:
            if win is not None:
                win.msg(str(e), 8000)
            return
        self._preview = None
        self._redraw()
        if win is not None:
            win.msg(text, 6000)

    def preview(self):
        if self._preview is not None:
            self._preview = None
            self._redraw()
            if self.ctx.window is not None:
                self.ctx.window.msg('Preview off', 3000)
            return
        from ...core.jobs import StudyError
        try:
            q = self._design_project()
        except StudyError as e:
            if self.ctx.window is not None:
                self.ctx.window.msg(str(e), 8000)
            return
        except Exception:                                # noqa: BLE001 — explained, logged
            traceback.print_exc()
            if self.ctx.window is not None:
                self.ctx.window.msg('Could not preview the design: an unexpected error (details in the log).', 8000)
            return
        if q is None:
            return
        self._preview = (self.STUDY, self.sel_label, self._posed(q, q.variant.sensors))
        self._redraw()
        if self.ctx.window is not None:
            self.ctx.window.msg(f'Previewing design {self.sel_label} (dashed): nothing is changed until you Accept.',
                                8000)

    def _posed(self, project, sensors) -> list:
        """[(name, frustum segments (S, 2, 3) world, centre (3,))] of the enabled camera and depth sensors."""
        from ...core import camrender as CR
        from ...core import library as LB
        from ...core.frames import quat_to_matrix
        from ...core.scenegeom import sensor_world_pose
        q = self.ctx.store.q
        out = []
        for s in sensors:
            if not s.enabled or s.kind not in ('camera', 'depth'):
                continue
            try:
                pw = sensor_world_pose(project, s, q)
                R, t = quat_to_matrix(pw.quaternion), np.asarray(pw.position, float)
                ims = LB.sensor_imagers(s, tuple(s.stream) if len(s.stream) == 2 else None)
                apex, border, spokes = CR.frustum_body(ims[0].camera, ims[0].pose_in_body, 0.18)
            except Exception:                            # noqa: BLE001 — a sensor that cannot be posed is not drawn
                continue
            B = border @ R.T + t
            ok = np.isfinite(B).all(axis=1)
            segs = [[a @ R.T + t, b @ R.T + t] for a, b in spokes]
            segs += [[B[i], B[i + 1]] for i in range(len(B) - 1) if ok[i] and ok[i + 1]]
            out.append((s.name, np.array(segs, float), apex @ R.T + t))
        return out

    def _redraw(self):
        if self.ctx.window is not None:
            self.ctx.window.viewport.update()

    def overlays(self, vp) -> list:
        from ..viewport3d import LineDraw
        if self._preview is None or self._preview[0] != self.STUDY:
            return []
        items = []
        col = QColor(PREVIEW_RGBA)
        for _name, segs, c in self._preview[2]:
            items.append(LineDraw(segs.astype(np.float32), col, 1.6, dash=True, segments=True))
            d = 0.02
            cross = np.array([[c - [d, 0, 0], c + [d, 0, 0]], [c - [0, d, 0], c + [0, d, 0]],
                              [c - [0, 0, d], c + [0, 0, d]]], np.float32)
            items.append(LineDraw(cross, col, 2.0, segments=True, depth=False))
        return items

    def camview_overlays(self, cam_model, pose) -> list:
        """The previewed design's cameras as small squares where they fall in the image."""
        from ...core import camrender as CR
        if self._preview is None or self._preview[0] != self.STUDY:
            return []
        out = []
        for name, _segs, c in self._preview[2]:
            uv, ok, _d = CR.project_world(cam_model, pose, np.asarray(c, float)[None, :])
            if not ok[0]:
                continue
            u, v = uv[0]
            s = max(6.0, cam_model.width / 120.0)
            sq = np.array([[u - s, v - s], [u + s, v - s], [u + s, v + s], [u - s, v + s], [u - s, v - s]])
            out.append(([sq], QColor(PREVIEW_RGBA), f'design {self._preview[1]} · {name}'))
        return out

    def page_context(self) -> dict:
        d = super().page_context()
        o = self._option()
        d.update(mode=MODE_TITLES[self.STUDY], selected=self.sel_label or None,
                 selected_explanation=o['explanation'] if o is not None else None)
        return d


register('optimise', 'Optimise', OptimisePage, menu='Analysis', icon='target', action_id=None, workspace=None,
         order=90)
