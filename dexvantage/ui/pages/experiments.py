"""The Designed experiments page (spec §7.11): a design over quantities of the rig, its runs, what drives each figure,
and the best run (core/experiments.py, the study core/studies/experiments.py).

What it shows: the best run's value of the objective response against the rig as it is (the headline, with a status
word: "better than now", "no run beats the rig", "not evaluated"); the design editor (factors picked from the rig
through core.factors.describe, with their ranges or levels; the responses, named metrics of core.metrics; the design
type, n, levels, resolution, seed and the number of processes, serial by default), the runs table with the best run
first, the main effects as bars with their Welch intervals and, for a Saltelli design, the first-order and total Sobol
indices with their bootstrap intervals. Apply sets the best run's factors (one undo step); Preview draws them in the
3D view without changing the project; Export writes the result JSON with provenance and the runs as CSV. The 3D view
draws every run's position of each sensor the design moves, shaded by the objective (darker is better), the best run
ringed and joined to where the sensor is now. There is no Camera view overlay: a run is a mount pose and a set of
settings, not an image feature.

Known limits: those of core/experiments.py.
"""
from __future__ import annotations

import copy
import json
import math
import os
import traceback

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (QComboBox, QFileDialog, QGridLayout, QHeaderView, QLineEdit, QMessageBox, QSpinBox,
                             QTreeWidget, QTreeWidgetItem, QWidget)

from ...core import experiments as EX
from ...core import factors as F
from ...core import metrics as M
from ..charts import ChartBase
from ..theme import SEQ, STATUS, THEME, mono, sans
from ..widgets import ScrubSpinBox, button, label
from . import register
from .base import StudyPage, row

MAX_TABLE_ROWS = 500              # rows of the runs table (the export has every run)
MAX_DRAWN_RUNS = 400              # runs drawn in the 3D view
POSE_FIELDS = F.POSITION + F.ANGLES + F.SHIFTS + F.TURNS


def _g(v, nd=4) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return '—'
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, (int, float)):
        return f'{v:.{nd}g}'
    return str(v)


def _ci(c, nd=3) -> str:
    if not c or c[0] is None:
        return '—'
    return f'{c[0]:.{nd}g} – {c[1]:.{nd}g}'


def _ticks(lo, hi, n=4):
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return [lo]
    raw = (hi - lo) / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag + 1e-12)
    t = math.ceil(lo / step - 1e-9) * step
    out = []
    while t <= hi + 1e-9 * step:
        out.append(round(t, 10))
        t += step
    return out


class IntervalBars(ChartBase):
    """Horizontal bars with interval whiskers, one row per factor, on a shared axis through zero. Mode 'sobol' draws
    the first-order index filled and the total outlined, each with its interval; mode 'effect' one filled bar.
    Ported from Vantage ui/workspaces/experiments_ws.py (IndexChart)."""
    LEFT = 150

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode, self.rows, self.unit, self.message = 'effect', [], '', 'Run the experiment'
        self._hits = []
        self.setMinimumHeight(180)

    def set_data(self, mode, rows, unit='', title='', message=''):
        """rows: effect [(label, value, ci)], sobol [(label, S1, S1ci, ST, STci)]."""
        self.mode, self.rows, self.unit, self.title = mode, rows, unit, title
        self.message = message or 'Run the experiment'
        self.tip = None
        self.setMinimumHeight(max(180, 50 + 26 * len(rows)))
        self.update()

    def _vals(self):
        out = []
        for r in self.rows:
            for v in r[1:]:
                if isinstance(v, (list, tuple)):
                    out += [x for x in v if x is not None]
                elif v is not None:
                    out.append(v)
        return out

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.paint_title(p, 'first order ■ · total □ · bootstrap interval' if self.mode == 'sobol'
                         else (f'{self.unit} · Welch interval' if self.unit else 'Welch interval'))
        if not self.rows:
            p.setFont(sans(12))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, 0, self.width(), self.height()), Qt.AlignmentFlag.AlignCenter, self.message)
            p.end()
            return
        vals = self._vals() or [0.0]
        lo, hi = min(min(vals), 0.0), max(max(vals), 0.0)
        if self.mode == 'sobol':
            lo, hi = min(lo, 0.0), max(hi, 1.0)
        if hi - lo < 1e-12:
            lo, hi = lo - 1.0, hi + 1.0
        pad = (hi - lo) * 0.05
        lo, hi = lo - pad, hi + pad
        r = QRectF(self.LEFT, 26, max(self.width() - self.LEFT - 18, 10), max(self.height() - 26 - 30, 10))
        X = lambda v: r.left() + (v - lo) / (hi - lo) * r.width()          # noqa: E731
        p.setFont(mono(10))
        for t in _ticks(lo, hi):
            x = X(t)
            p.setPen(QPen(THEME.color('hair'), 1))
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(x - 32, r.bottom() + 3, 64, 13), Qt.AlignmentFlag.AlignCenter, f'{t:.3g}')
        p.setPen(QPen(THEME.color('strong'), 1))
        p.drawLine(QPointF(X(0.0), r.top()), QPointF(X(0.0), r.bottom()))
        p.setFont(sans(10))
        p.setPen(THEME.color('muted'))
        p.drawText(QRectF(r.left(), r.bottom() + 15, r.width(), 13), Qt.AlignmentFlag.AlignRight,
                   'share of the variance' if self.mode == 'sobol' else 'mean at the last level − mean at the first')
        acc = THEME.cam(0)
        n = len(self.rows)
        rowh = r.height() / max(n, 1)
        self._hits = []
        for i, rw in enumerate(self.rows):
            y0 = r.top() + i * rowh
            if i == self.hover:
                p.fillRect(QRectF(0, y0, self.width(), rowh), THEME.color('raised'))
            p.setFont(sans(11))
            p.setPen(THEME.color('ink'))
            p.drawText(QRectF(4, y0, self.LEFT - 10, rowh), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                       p.fontMetrics().elidedText(rw[0], Qt.TextElideMode.ElideMiddle, self.LEFT - 12))
            self._hits.append((QRectF(0, y0, self.width(), rowh), i))
            bars = [(rw[1], rw[2], True, rowh / 2 - min(rowh * 0.3, 9.0) - 1), (rw[3], rw[4], False, rowh / 2 + 1)] \
                if self.mode == 'sobol' else [(rw[1], rw[2], True, rowh / 2 - min(rowh * 0.25, 7.0))]
            bh = min(rowh * 0.3, 9.0) if self.mode == 'sobol' else min(rowh * 0.5, 14.0)
            for val, ci, fill, off in bars:
                if val is None:
                    continue
                x0, x1 = sorted((X(0.0), X(val)))
                rr = QRectF(x0, y0 + off, max(x1 - x0, 1.0), bh)
                if fill:
                    p.fillRect(rr, acc)
                else:
                    p.setPen(QPen(acc, 1.2))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawRect(rr)
                if ci and ci[0] is not None:
                    ym = y0 + off + bh / 2
                    p.setPen(QPen(THEME.color('ink'), 1))
                    p.drawLine(QPointF(X(ci[0]), ym), QPointF(X(ci[1]), ym))
                    for c in ci:
                        p.drawLine(QPointF(X(c), ym - 3), QPointF(X(c), ym + 3))
        self.paint_tip(p)
        p.end()

    def mouseMoveEvent(self, e):
        hit = next((i for rr, i in self._hits if rr.contains(e.position())), None)
        self.hover = hit
        self.tip = None
        if hit is not None:
            rw = self.rows[hit]
            if self.mode == 'sobol':
                self.set_tip(e.position(), f'S1 {_g(rw[1], 3)} · ST {_g(rw[3], 3)}', rw[0],
                             f'first order interval {_ci(rw[2])}', f'total interval {_ci(rw[4])}')
            else:
                self.set_tip(e.position(), f'{_g(rw[1])} {self.unit}'.strip(), rw[0], f'interval {_ci(rw[2])}')
        self.update()


class ExperimentsPage(StudyPage):
    """Designed experiments: the rig evaluated at every run of a design (full or fractional factorial, Latin
    hypercube, Sobol sequence, or Saltelli sampling for Sobol indices), each run's figures from the studies through
    the named metrics. Main effects are the mean at a factor's last level minus the mean at its first, with a Welch
    interval; Sobol indices use the Jansen estimators with bootstrap intervals. The headline is the best run's
    objective against the rig as it is."""
    STUDY = EX.STUDY_ID
    TITLE = 'Designed experiments'
    HEADLINE = 'Best run of the experiment'

    # ---------------------------------------------------------------- body
    def build_body(self, layout):
        from ..cluster_run import add_run_on
        add_run_on(self, 'experiment')                # Run on…: N processes or a Slurm job folder (core/cluster)
        self._syncing = False
        self._overlay3d = None
        self._preview = None
        self._cands = []
        layout.addWidget(label('Factors: quantities of the rig the experiment varies', bold=True))
        self.pick = QComboBox()
        self.pick.setMinimumWidth(320)
        self.pick.setToolTip('A quantity of the rig (core/factors paths); its starting range is a DexVantage default '
                             '(±20 mm, ±5°, ±25 % of a setting) for you to edit')
        self.b_add = button('Add factor', icon_name='plus', tip='Add the picked quantity as a factor')
        self.b_add.clicked.connect(lambda: self.add_factor())
        self.b_remove = button('Remove', role='ghost', tip='Remove the selected factor')
        self.b_remove.clicked.connect(lambda: self.remove_factor())
        layout.addWidget(row(self.pick, self.b_add, self.b_remove))
        self.ftable = QTreeWidget()
        self.ftable.setRootIsDecorated(False)
        self.ftable.setHeaderLabels(['Factor', 'Kind', 'Low', 'High', 'Levels', 'Now'])
        _columns(self.ftable)
        self.ftable.currentItemChanged.connect(lambda *_a: self._sync_editor())
        layout.addWidget(self.ftable)
        ed = QWidget()
        g = QGridLayout(ed)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(8)
        self.lo_spin = ScrubSpinBox('', 'plain', 4, '', step=0.001, width=110)
        self.hi_spin = ScrubSpinBox('', 'plain', 4, '', step=0.001, width=110)
        for s in (self.lo_spin, self.hi_spin):
            s.setRange(-1e9, 1e9)
        self.lo_spin.committed.connect(lambda v: self.set_factor(lo=float(v)))
        self.hi_spin.committed.connect(lambda v: self.set_factor(hi=float(v)))
        self.nlev_spin = QSpinBox()
        self.nlev_spin.setRange(0, EX.LIMITS['levels'][1])
        self.nlev_spin.setSpecialValueText('design default')
        self.nlev_spin.setToolTip('Levels of this factor in a full factorial (0: the design\'s default)')
        self.nlev_spin.editingFinished.connect(lambda: self.set_factor(n_levels=int(self.nlev_spin.value())))
        self.levels_edit = QLineEdit()
        self.levels_edit.setPlaceholderText('levels, comma-separated (categorical)')
        self.levels_edit.editingFinished.connect(self._levels_edited)
        g.addWidget(label('Low', muted=True), 0, 0)
        g.addWidget(self.lo_spin, 0, 1)
        g.addWidget(label('High', muted=True), 0, 2)
        g.addWidget(self.hi_spin, 0, 3)
        g.addWidget(label('Factorial levels', muted=True), 0, 4)
        g.addWidget(self.nlev_spin, 0, 5)
        g.addWidget(label('Levels', muted=True), 1, 0)
        g.addWidget(self.levels_edit, 1, 1, 1, 5)
        layout.addWidget(ed)

        layout.addWidget(label('Responses: the figures each run measures', bold=True))
        self.rlist = QTreeWidget()
        self.rlist.setRootIsDecorated(False)
        self.rlist.setHeaderLabels(['Metric', 'Unit', 'Better', 'Study'])
        _columns(self.rlist)
        self.rlist.itemChanged.connect(self._response_ticked)
        layout.addWidget(self.rlist)
        self.obj_box = QComboBox()
        self.obj_box.setToolTip('The response that picks the best run (default: the first ticked)')
        self.obj_box.activated.connect(lambda _i: self._set_design(objective=self.obj_box.currentData() or '',
                                                                   text='objective'))
        layout.addWidget(row(label('Best run by', muted=True), self.obj_box))

        layout.addWidget(label('Design', bold=True))
        dg = QWidget()
        g2 = QGridLayout(dg)
        g2.setContentsMargins(0, 0, 0, 0)
        g2.setHorizontalSpacing(8)
        self.method_box = QComboBox()
        for m in EX.METHODS:
            self.method_box.addItem(EX.METHOD_LABEL[m] + (' (default)' if m == EX.DEFAULTS['method'] else ''), m)
        self.method_box.activated.connect(lambda _i: self._set_design(method=self.method_box.currentData(),
                                                                      text='design type'))
        self.n_spin = QSpinBox()
        self.n_spin.setRange(*EX.LIMITS['n'])
        self.n_spin.setToolTip(f'Points of a Latin hypercube or Sobol sequence; base samples N of a Saltelli design '
                               f'(N(k + 2) runs). Default {EX.DEFAULTS["n"]}')
        self.n_spin.editingFinished.connect(lambda: self._set_design(n=int(self.n_spin.value()), text='n'))
        self.levels_spin = QSpinBox()
        self.levels_spin.setRange(*EX.LIMITS['levels'])
        self.levels_spin.setToolTip(f'Levels of a continuous factor in a full factorial (default '
                                    f'{EX.DEFAULTS["levels"]})')
        self.levels_spin.editingFinished.connect(lambda: self._set_design(levels=int(self.levels_spin.value()),
                                                                          text='levels'))
        self.res_box = QComboBox()
        for r in EX.RESOLUTIONS:
            self.res_box.addItem(f'Resolution {r}' + (' (default)' if r == EX.DEFAULTS['resolution'] else ''), r)
        self.res_box.setToolTip('III: main effects free of each other, aliased with two-factor interactions. IV: '
                                'main effects free of two-factor interactions')
        self.res_box.activated.connect(lambda _i: self._set_design(resolution=self.res_box.currentData(),
                                                                   text='resolution'))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(*EX.LIMITS['seed'])
        self.seed_spin.setToolTip('The seed of the sampled designs and of the bootstrap (default 0)')
        self.seed_spin.editingFinished.connect(lambda: self._set_design(seed=int(self.seed_spin.value()), text='seed'))
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(*EX.LIMITS['workers'])
        self.workers_spin.setSpecialValueText('serial (default)')
        self.workers_spin.setToolTip('Processes that evaluate runs at once. 0: serial, the default. More pays only '
                                     'when a run takes over about a second (each process loads DexVantage)')
        self.workers_spin.editingFinished.connect(lambda: self._set_design(workers=int(self.workers_spin.value()),
                                                                           text='processes'))
        for i, (t, w) in enumerate((('Type', self.method_box), ('n', self.n_spin), ('Levels', self.levels_spin),
                                    ('Fraction', self.res_box), ('Seed', self.seed_spin),
                                    ('Processes', self.workers_spin))):
            g2.addWidget(label(t, muted=True), i // 3, 2 * (i % 3))
            g2.addWidget(w, i // 3, 2 * (i % 3) + 1)
        layout.addWidget(dg)
        self.count_label = label('', mono_=True)
        self.b_cancel = button('Cancel run', role='ghost', tip='Stop the experiment after the run in progress; the '
                                                               'runs done are kept and analysed')
        self.b_cancel.clicked.connect(lambda: self.ctx.analysis.cancel(self.STUDY))
        layout.addWidget(row(self.count_label, self.b_cancel))
        self.problems = label('', wrap=True)
        layout.addWidget(self.problems)

        layout.addWidget(label('Runs (best first)', bold=True))
        self.table = QTreeWidget()
        self.table.setRootIsDecorated(False)
        self.table.setMinimumHeight(160)
        layout.addWidget(self.table)
        self.chart_box = QComboBox()
        self.chart_box.setToolTip('The response the charts show')
        self.chart_box.activated.connect(lambda _i: self._fill_charts())
        layout.addWidget(row(label('Charts for', muted=True), self.chart_box))
        self.effects = IntervalBars()
        self.sobol = IntervalBars()
        layout.addWidget(self.effects)
        layout.addWidget(self.sobol)
        self.notes_label = label('', muted=True, wrap=True)
        layout.addWidget(self.notes_label)
        self.ctx.store.changed.connect(self._changed)            # bound slots: disconnected when the page goes
        self.ctx.store.project_replaced.connect(self._replaced)
        self._sync()

    # ---------------------------------------------------------------- the design in the project
    def design_dict(self) -> dict:
        d = EX.settings_of(self.ctx.store.project)
        d.setdefault('schema', EX.SCHEMA)
        d.setdefault('factors', [])
        d.setdefault('responses', [])
        d.setdefault('design', {})
        return d

    def _store(self, d: dict, text: str, merge=None):
        d = copy.deepcopy(d)

        def fn(p):
            p.analysis.studies[EX.STUDY_ID] = d
        self.ctx.store.edit('analysis', fn, f'Experiments · {text}', merge_key=(EX.STUDY_ID, merge) if merge else None)

    def _set_design(self, text='design', **kw):
        if self._syncing:
            return False
        d = self.design_dict()
        if 'objective' in kw:
            d['objective'] = kw.pop('objective')
        des = dict(d.get('design') or {})
        exe = dict(d.get('execution') or {})
        if 'workers' in kw:
            exe['workers'] = kw.pop('workers')
        des.update(kw)
        if des == (d.get('design') or {}) and exe == (d.get('execution') or {}) and not text == 'objective':
            return False
        d['design'], d['execution'] = des, exe
        self._store(d, text, merge=text)
        return True

    def add_factor(self, path: str | None = None) -> bool:
        """Add the picked quantity (or the template for path) as a factor."""
        if path is None:
            path = self.pick.currentData()
        c = next((c for c in self._cands if c['path'] == path), None)
        if c is None:
            return False
        d = self.design_dict()
        if any(f.get('path') == path for f in d['factors']):
            self._say(f'{c["title"]} is already a factor.')
            return False
        if c['locked']:
            self._say(f'{c["title"]}: the sensor is locked; unlock it to vary it.')
            return False
        d['factors'].append(copy.deepcopy(c['factor']))
        self._store(d, f'add factor {c["title"]}')
        self._select(path)
        return True

    def remove_factor(self, path: str | None = None) -> bool:
        path = path or self._current_path()
        d = self.design_dict()
        keep = [f for f in d['factors'] if f.get('path') != path]
        if len(keep) == len(d['factors']):
            return False
        d['factors'] = keep
        self._store(d, 'remove a factor')
        return True

    def set_factor(self, path: str | None = None, **kw) -> bool:
        if self._syncing:
            return False
        path = path or self._current_path()
        d = self.design_dict()
        f = next((f for f in d['factors'] if f.get('path') == path), None)
        if f is None:
            return False
        before = dict(f)
        f.update(kw)
        if f.get('kind') == 'continuous' and f.get('lo') is not None and f.get('hi') is not None and f['lo'] > f['hi']:
            f['lo'], f['hi'] = f['hi'], f['lo']
        if f.get('n_levels') == 0:
            f.pop('n_levels', None)
        if f == before:
            return False
        self._store(d, f'factor {path.split(".")[-1]}', merge=('factor', path, tuple(sorted(kw))))
        return True

    def _levels_edited(self):
        text = self.levels_edit.text().strip()
        if not text:
            return
        vals = []
        for part in text.split(','):
            part = part.strip()
            try:
                vals.append(json.loads(part))
            except ValueError:
                vals.append(part)
        self.set_factor(levels=vals)

    def set_responses(self, ids) -> bool:
        d = self.design_dict()
        ids = [str(i) for i in ids]
        if ids == d['responses']:
            return False
        d['responses'] = ids
        if d.get('objective') and d['objective'] not in ids:
            d['objective'] = ''
        self._store(d, 'responses')
        return True

    def _response_ticked(self, _item, _col=0):
        if self._syncing:
            return
        ids = []
        for i in range(self.rlist.topLevelItemCount()):
            it = self.rlist.topLevelItem(i)
            if it.checkState(0) == Qt.CheckState.Checked:
                ids.append(it.data(0, Qt.ItemDataRole.UserRole))
        self.set_responses(ids)

    # ---------------------------------------------------------------- the editor
    def _current_path(self):
        it = self.ftable.currentItem()
        return it.data(0, Qt.ItemDataRole.UserRole) if it is not None else None

    def _select(self, path):
        for i in range(self.ftable.topLevelItemCount()):
            it = self.ftable.topLevelItem(i)
            if it.data(0, Qt.ItemDataRole.UserRole) == path:
                self.ftable.setCurrentItem(it)
                return

    def _changed(self, _sections=None):
        self._sync()

    def _replaced(self):
        self.doc = None
        self._preview = None
        self._overlay3d = None
        self._sync()
        self.empty()

    def _sync(self):
        """The editor from the project's saved design (cheap: no study runs here)."""
        if self._syncing:
            return
        self._syncing = True
        try:
            p = self.ctx.store.project
            d = self.design_dict()
            try:
                self._cands = EX.candidates(p)
            except Exception:                              # noqa: BLE001 — an empty picker, the traceback logged
                traceback.print_exc()
                self._cands = []
            cur_pick = self.pick.currentData()
            self.pick.clear()
            for c in self._cands:
                self.pick.addItem(f'{c["group"]} · {c["title"]}' + (' (locked)' if c['locked'] else ''), c['path'])
            if cur_pick is not None:
                i = self.pick.findData(cur_pick)
                if i >= 0:
                    self.pick.setCurrentIndex(i)
            cur = self._current_path()
            self.ftable.clear()
            for f in d['factors']:
                path = str(f.get('path', ''))
                try:
                    desc = F.describe(path, p)
                    title, unit = desc['title'], desc['unit']
                    now = F.get(p, path)
                except Exception:                          # noqa: BLE001 — said in the row
                    title, unit, now = path, '', None
                kind = f.get('kind') or ('categorical' if 'levels' in f else 'continuous')
                levels = ', '.join(_g(x) for x in f.get('levels') or []) if kind == 'categorical' else \
                    'false, true' if kind == 'bool' else str(f.get('n_levels') or 'design default')
                it = QTreeWidgetItem([f.get('name') or title, kind,
                                      _g(f.get('lo')) + (f' {unit}' if unit and kind == 'continuous' else ''),
                                      _g(f.get('hi')) + (f' {unit}' if unit and kind == 'continuous' else ''),
                                      levels, _g(now)])
                it.setData(0, Qt.ItemDataRole.UserRole, path)
                it.setToolTip(0, path)
                for k in (2, 3, 5):
                    it.setFont(k, mono(11))
                    it.setTextAlignment(k, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.ftable.addTopLevelItem(it)
            _fit(self.ftable)
            if cur:
                self._select(cur)
            if self.ftable.currentItem() is None and self.ftable.topLevelItemCount():
                self.ftable.setCurrentItem(self.ftable.topLevelItem(0))
            self.rlist.clear()
            M.load_metrics()
            chosen = list(d['responses'])
            for mid in chosen + sorted(k for k in M.METRICS if k not in chosen):
                m = M.METRICS.get(mid)
                if m is not None and m.study == EX.STUDY_ID:
                    continue
                it = QTreeWidgetItem([m.title if m else mid, m.unit if m else '', m.better if m else '',
                                      m.study if m else 'no such metric'])
                it.setData(0, Qt.ItemDataRole.UserRole, mid)
                it.setToolTip(0, mid + (f' · {m.note}' if m and m.note else ''))
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(0, Qt.CheckState.Checked if mid in chosen else Qt.CheckState.Unchecked)
                self.rlist.addTopLevelItem(it)
            if not M.METRICS:
                it = QTreeWidgetItem(['No metric is registered yet (core/metric_defs.py names the studies\' figures)',
                                      '', '', ''])
                self.rlist.addTopLevelItem(it)
            _fit(self.rlist)
            self.obj_box.clear()
            self.obj_box.addItem('The first response (default)', '')
            for mid in chosen:
                m = M.METRICS.get(mid)
                self.obj_box.addItem(m.title if m else mid, mid)
            self.obj_box.setCurrentIndex(max(0, self.obj_box.findData(d.get('objective') or '')))
            des = d.get('design') or {}
            self.method_box.setCurrentIndex(max(0, self.method_box.findData(des.get('method',
                                                                                    EX.DEFAULTS['method']))))
            self.n_spin.setValue(int(des.get('n', EX.DEFAULTS['n'])))
            self.levels_spin.setValue(int(des.get('levels', EX.DEFAULTS['levels'])))
            self.res_box.setCurrentIndex(max(0, self.res_box.findData(des.get('resolution',
                                                                              EX.DEFAULTS['resolution']))))
            self.seed_spin.setValue(int(des.get('seed', 0)))
            self.workers_spin.setValue(int((d.get('execution') or {}).get('workers', 0)))
            try:
                design = EX.Design.from_dict(d, p)
                self.count_label.setText(EX.design_text(design))
                probs = EX.check_design(p, design)
            except EX.ExperimentError as e:
                self.count_label.setText('—')
                probs = [str(e)]
            self.problems.setText(' '.join(probs))
            self.problems.setVisible(bool(probs))
            self.b_run.setEnabled(not probs)
            self._sync_editor()
        finally:
            self._syncing = False

    def _sync_editor(self):
        path = self._current_path()
        f = next((f for f in self.design_dict()['factors'] if f.get('path') == path), None)
        kind = (f or {}).get('kind') or ('categorical' if f and 'levels' in f else 'continuous')
        cont = f is not None and kind == 'continuous'
        was = self._syncing
        self._syncing = True
        try:
            for w in (self.lo_spin, self.hi_spin, self.nlev_spin):
                w.setEnabled(cont)
            self.levels_edit.setEnabled(f is not None and kind == 'categorical')
            if cont:
                unit = F.describe(path)['unit'] if ':' in path else ''
                self.lo_spin.set_unit(unit)
                self.hi_spin.set_unit(unit)
                self.lo_spin.set_value(float(f.get('lo', 0.0)))
                self.hi_spin.set_value(float(f.get('hi', 0.0)))
                self.nlev_spin.setValue(int(f.get('n_levels') or 0))
            self.levels_edit.setText(', '.join(json.dumps(x) if not isinstance(x, str) else x
                                               for x in (f or {}).get('levels') or []) if kind == 'categorical'
                                     else '')
        finally:
            self._syncing = was

    def _say(self, text, ms=6000):
        if self.ctx.window is not None:
            self.ctx.window.msg(text, ms)

    # ---------------------------------------------------------------- results
    def show_result(self, doc):
        self._overlay3d = None
        self._preview = None
        h = doc.get('headline') or {}
        v = h.get('value')
        self.head.set(_g(v), h.get('unit', ''), h.get('status', 'warning'), h.get('word', ''), h.get('text', ''),
                      tip=f'{doc.get("method_label", "")}: {doc.get("design_text", "")}')
        best = doc.get('best') or {}
        self.suggestion_ready(bool(best and best.get('improves') and best.get('change')))
        fs, rs = doc.get('factors') or [], doc.get('responses') or []
        self.table.clear()
        self.table.setColumnCount(2 + len(fs) + len(rs))
        self.table.setHeaderLabels(['Run', 'Block'] + [f['title'] + (f' ({f["unit"]})' if f.get('unit') else '')
                                                       for f in fs]
                                   + [r['title'] + (f' ({r["unit"]})' if r['unit'] else '') for r in rs])
        obj = doc.get('objective')
        better = next((r['better'] for r in rs if r['id'] == obj), 'higher')
        runs = [r for r in doc.get('runs') or [] if r.get('done')]
        runs.sort(key=lambda r: (r['responses'].get(obj) is None,
                                 -(r['responses'].get(obj) or 0.0) if better == 'higher'
                                 else (r['responses'].get(obj) or 0.0)))
        for r in runs[:MAX_TABLE_ROWS]:
            cells = [str(r['run']), r.get('block') or ''] + [_g(r['values'].get(f['path'])) for f in fs] + \
                [{'at_least': '≥ ', 'at_most': '≤ '}.get((r.get('bound') or {}).get(x['id'], ''), '')   # a bound
                 + _g(r['responses'].get(x['id'])) if r['status'].get(x['id']) == 'ok' else r['status'].get(x['id'], '')
                 for x in rs]
            it = QTreeWidgetItem(cells)
            for k in range(len(cells)):
                it.setFont(k, mono(11))
                if k != 1:
                    it.setTextAlignment(k, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            for j, x in enumerate(rs):
                why = r.get('reason', {}).get(x['id'])
                if why:
                    it.setToolTip(2 + len(fs) + j, why)
            if best and r['run'] == best.get('run'):
                it.setForeground(0, QColor(STATUS['good']))
                it.setToolTip(0, 'The best run')
            self.table.addTopLevelItem(it)
        hh = self.table.header()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        cur = self.chart_box.currentData()
        self.chart_box.clear()
        for r in rs:
            self.chart_box.addItem(r['title'], r['id'])
        self.chart_box.setCurrentIndex(max(0, self.chart_box.findData(cur or obj)))
        self._fill_charts()
        ex = doc.get('execution') or {}
        notes = list(doc.get('notes') or [])
        notes.append(f'{ex.get("runs_done", 0)} of {ex.get("runs_planned", 0)} runs, {ex.get("mode", "")}, '
                     f'{ex.get("seconds", 0):.1f} s' + (f'; {ex["cache_hits"]} study results from the cache'
                                                       if ex.get('cache_hits') else '') + '.')
        if doc.get('cancelled_at') is not None:
            notes.append('Cancelled: the runs done are analysed; Run again resumes from the cache.')
        notes.append('3D: each run\'s position of the sensors the design moves, darker = better objective; the best '
                     'run ringed in green and joined to where the sensor is now.')
        notes.append('Known limits: ' + doc.get('known_limits', ''))
        self.notes_label.setText('\n'.join(notes))
        self._redraw()

    def _fill_charts(self):
        doc = self.doc or {}
        mid = self.chart_box.currentData()
        a = (doc.get('analysis') or {}).get(mid) or {}
        titles = {f['path']: f['title'] for f in doc.get('factors') or []}
        unit = next((r['unit'] for r in doc.get('responses') or [] if r['id'] == mid), '')
        me = a.get('main_effects') or []
        binned = any(m.get('binned') for m in me)
        self.effects.set_data('effect', [(titles.get(m['factor'], m['factor']), m.get('effect'), m.get('effect_ci'))
                                         for m in me], unit,
                              'Main effects' + (' (binned: last quantile bin minus first)' if binned else ''),
                              'No main effects for this response' if doc else '')
        s = a.get('sobol')
        if s and s.get('S1') and s['S1'][0] is not None:
            self.sobol.set_data('sobol', [(titles.get(p, p), s['S1'][i], s['S1_ci'][i], s['ST'][i], s['ST_ci'][i])
                                          for i, p in enumerate(s.get('factors') or [])], '',
                                f'Sobol indices (N = {s["N"]}, {s.get("dropped", 0)} rows without a value)')
            self.sobol.show()
        else:
            self.sobol.set_data('sobol', [], '', 'Sobol indices',
                                (s or {}).get('note') or 'Sobol indices need the Saltelli design')
            self.sobol.setVisible(doc.get('method') == 'saltelli')

    # ---------------------------------------------------------------- 3D
    def _sensor_tracks(self):
        """{sensor uid: (name, current world position, [(run, world position, objective)])} for the sensors whose
        pose the design varies."""
        doc = self.doc or {}
        p = self.ctx.store.project
        from ...core import scenegeom as SG
        from ...core.frames import Pose
        from ...core.project import Project, Sensor
        variant = (doc.get('design') or {}).get('variant') or None
        by_sensor: dict = {}
        for f in doc.get('factors') or []:
            path = f['path']
            if not path.startswith('sensor:'):
                continue
            ident, _, fld = path[len('sensor:'):].rpartition('.')
            if fld in POSE_FIELDS:
                by_sensor.setdefault(ident, []).append((path, fld))
        obj = doc.get('objective')
        out = {}
        q = getattr(self.ctx.store, 'q', None)
        runs = [r for r in doc.get('runs') or [] if r.get('done')][:MAX_DRAWN_RUNS]
        best = (doc.get('best') or {}).get('run')
        if best and not any(r['run'] == best for r in runs):
            runs += [r for r in doc['runs'] if r['run'] == best]
        for ident, fl in by_sensor.items():
            try:
                s = F.find_sensor(p, ident, variant)
                T_world = SG.sensor_world_pose(p, s, q).matrix()
                T_link = T_world @ np.linalg.inv(s.mount.pose.matrix())
            except Exception:                              # noqa: BLE001 — that sensor is not drawn
                traceback.print_exc()
                continue
            mini = Project()
            c = Sensor.from_dict(s.to_dict())
            c.locked = False
            mini.variant.sensors = [c]
            pts = []
            for r in runs:
                vals = {f'sensor:{c.uid}.{fld}': r['values'][path] for path, fld in fl if path in r['values']}
                try:
                    m2 = F.apply(mini, vals)
                except Exception:                          # noqa: BLE001
                    continue
                P = (T_link @ m2.variant.sensors[0].mount.pose.matrix())[:3, 3]
                pts.append((r['run'], P, r['responses'].get(obj)))
            out[s.uid] = (s.name, Pose.from_matrix(T_world).position, pts)
        return out

    def overlays(self, vp) -> list:
        if self.doc is None:
            return []
        if self._overlay3d is not None:
            return self._overlay3d
        from ..viewport3d import LineDraw
        items = []
        try:
            tracks = self._sensor_tracks()
        except Exception:                                  # noqa: BLE001 — no overlay, the traceback logged
            traceback.print_exc()
            tracks = {}
        rs = {r['id']: r for r in self.doc.get('responses') or []}
        better = rs.get(self.doc.get('objective'), {}).get('better', 'higher')
        best = (self.doc.get('best') or {}).get('run')
        show = self._preview if self._preview is not None else best
        for _uid, (_name, now, pts) in tracks.items():
            vals = [v for _r, _P, v in pts if v is not None]
            lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
            bins: dict = {}
            for run, P, v in pts:
                if v is None:
                    k = -1
                else:
                    t = (v - lo) / (hi - lo) if hi > lo else 1.0
                    t = t if better == 'higher' else 1.0 - t
                    k = int(round(t * (len(SEQ) - 1)))
                bins.setdefault(k, []).append(P)
                if run == show:
                    items += _ring(P, QColor(STATUS['good']), 0.012)
                    items.append(LineDraw(np.array([now, P], np.float32), QColor(STATUS['good']), 1.5, dash=True,
                                          depth=False))
            for k, P in bins.items():
                col = QColor(SEQ[k]) if k >= 0 else THEME.color('muted')
                items.append(LineDraw(_crosses(np.array(P), 0.004), col, 1.5, depth=False, segments=True))
        self._overlay3d = items
        return items

    def _redraw(self):
        self._overlay3d = None
        win = self.ctx.window
        if win is not None and hasattr(win, 'viewport'):
            win.viewport.update()

    # ---------------------------------------------------------------- apply / preview / export / explain
    def _best_values(self):
        best = (self.doc or {}).get('best') or {}
        return best.get('values'), (self.doc.get('design') or {}).get('variant', '') if self.doc else ''

    def apply(self):
        """Set the best run's factors (one undo step), then run again."""
        best = (self.doc or {}).get('best')
        if not best or not best.get('values'):
            return False
        values, variant = self._best_values()
        p = self.ctx.store.project
        if variant and p.variant_by_uid(variant) is None:
            self._say('The experiment\'s variant is not in the project any more.')
            return False
        try:
            F.apply(p, values, variant=variant or None)           # refuses a locked sensor before the edit
        except F.FactorError as e:
            self._say(str(e), 8000)
            return False
        keys = EX.apply_keys(p, values, variant)
        self.ctx.store.edit(keys, lambda pp: EX.apply_values(pp, values, variant),
                            f'Experiments · set run {best["run"]}\'s {len(values)} factors')
        self._preview = None
        self.suggestion_ready(False)
        self._redraw()
        return True

    def preview(self):
        best = (self.doc or {}).get('best')
        if not best:
            return None
        self._preview = best['run']
        titles = {f['path']: (f['title'], f.get('unit') or '') for f in self.doc.get('factors') or []}
        parts = [f'{titles.get(k, (k, ""))[0]} {_g(v)}{(" " + titles[k][1]) if k in titles and titles[k][1] else ""}'
                 for k, v in best['values'].items()]
        self._say(f'Preview of run {best["run"]}: ' + '; '.join(parts) + '. Apply makes the change (one undo step).',
                  15000)
        self._redraw()
        return best

    def export(self, path=None):
        if self.doc is None:
            if path is None:
                self._say('There is no experiment result to export yet: Run it first.', 5000)
            return None
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, 'Export the experiment', 'experiment.json', 'JSON (*.json)')
            if not path:
                return None
        doc = {k: v for k, v in self.doc.items() if k not in ('cached', 'revision')}
        try:
            jp, cp = EX.write_outputs(doc, self.ctx.store.project, path, option=None)
        except (OSError, ValueError) as e:
            self._say(f'Could not write {os.path.basename(path)}: {getattr(e, "strerror", None) or e}.', 8000)
            return None
        self._say(f'Exported the experiment to {os.path.basename(jp)} and {os.path.basename(cp)}', 5000)
        return jp

    def explain_text(self) -> str:
        return ('Each run sets the factors on a copy of the project (core/factors.py) and reads the responses from '
                'the studies through the named metrics (core/metrics.py), with the study result cache. Designs: '
                'full factorial; two-level fractions of resolution III or IV (generators from the base factors\' '
                'interactions, the resolution found again by brute force); Latin hypercube (one point per stratum of '
                'every factor); a scrambled Sobol sequence; Saltelli sampling (A, B and AB_i, N(k + 2) runs).\n\n'
                'Main effect: the mean response at a factor\'s last level minus the mean at its first, with a Welch '
                't interval that treats the other factors as noise. Sobol indices (Jansen 1999): first order '
                'S_i = (V − ½·mean((f(B) − f(AB_i))²))/V, total ST_i = ½·mean((f(A) − f(AB_i))²)/V, with bootstrap '
                'percentile intervals.\n\n'
                'Defaults (DexVantage defaults, not standards): a factor picked from the rig starts at ±20 mm or ±5° '
                'around its value; Latin hypercube, 32 runs; 2 levels; resolution IV; 500 bootstrap resamples; 95 % '
                'intervals; serial runs.\n\n'
                'Known limits: ' + EX.KNOWN_LIMITS)

    def explain(self):
        box = QMessageBox(self)
        box.setWindowTitle('How the experiment is computed')
        box.setText(self.explain_text())
        box.setModal(False)
        box.show()


def _crosses(P: np.ndarray, h: float) -> np.ndarray:
    """(S, 2, 3) segments: a small three-axis cross at each point."""
    out = []
    for p in np.asarray(P, float).reshape(-1, 3):
        for ax in np.eye(3):
            out.append([p - h * ax, p + h * ax])
    return np.asarray(out, np.float32).reshape(-1, 2, 3)


def _ring(P, col, r):
    from ..viewport3d import LineDraw
    t = np.linspace(0, 2 * math.pi, 25)
    out = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        pts = np.tile(np.asarray(P, float), (len(t), 1))
        pts[:, a] += r * np.cos(t)
        pts[:, b] += r * np.sin(t)
        out.append(LineDraw(pts.astype(np.float32), col, 2.0, depth=False))
    return out


def _columns(tree: QTreeWidget):
    h = tree.header()
    h.setStretchLastSection(False)
    h.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    for k in range(1, tree.columnCount()):
        h.setSectionResizeMode(k, QHeaderView.ResizeMode.ResizeToContents)


def _fit(tree: QTreeWidget):
    n = tree.topLevelItemCount()
    rh = tree.sizeHintForRow(0) if n else 20
    tree.setFixedHeight(tree.header().sizeHint().height() + max(1, min(n, 12)) * max(rh, 18) + 6)


register('experiments', 'Designed experiments', ExperimentsPage, menu='Analysis', icon='snap-grid', order=90)
