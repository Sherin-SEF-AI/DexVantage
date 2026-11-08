"""The Statistics page (Analysis ▸ Statistics, research tools part 2): is a difference real, and how many episodes or
draws does it take to tell? (core/studies/stats.py, core/stats.py, core/statcompare.py).

What it shows: a comparison source (success counts entered in a small table, two variants over the tolerance Monte
Carlo, a designed experiment's runs, or the tracker's runs); the headline, the comparison with the smallest corrected
p-value as its difference B − A with a status word (clear difference, no clear difference, cannot test); a table of
every figure or comparison (both sides, the difference with its interval, the effect size with its interval, the
test, p and the corrected p, the verdict); which test ran and why, with the assumptions in plain words ("the draws
are paired by seed"); and a power calculator (success rates, a two-group or paired t test, or demonstrations with
the operators' clustering and the audit's discards) with its curve of power against n.

Apply writes the suggested plan (one undo step): when a difference is not clear, the episodes, draws or runs that
would detect the observed difference with the target power (the variants' draws, or the power calculator's plan).
Preview shows that plan in the power calculator without storing it. Export writes the result with provenance (the
power calculator as shown included).

Known limits: see the study's. The power curve is computed on a worker thread; an exact curve above a few hundred
episodes per condition takes a moment. Sequential testing and early stopping are out of scope.
"""
from __future__ import annotations

import math
import threading
import traceback

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QComboBox, QGridLayout, QHeaderView, QLineEdit, QMessageBox, QSizePolicy,
                             QStackedWidget, QTableWidget, QTableWidgetItem, QTreeWidget, QTreeWidgetItem,
                             QVBoxLayout, QWidget)

from ...core import statcompare as SC
from ...core import stats as ST
from ...core.metrics import delta_unit
from ...core.studies import stats as SS
from ..theme import STATUS, THEME, mono
from ..widgets import ScrubSpinBox, SegmentedControl, button, label
from . import register
from ._curves import CurveChart
from .base import StudyPage, row


def _sg(v: float, spec: str) -> str:
    """A signed figure ('+3.2'); a value that rounds to zero is written 0, never +0 or -0."""
    t = format(float(v), '+' + spec)
    return '0' if float(t) == 0 else t


SOURCES = (('rates', 'Success counts'), ('variants', 'Variants'), ('experiments', 'Experiment runs'),
           ('tracker', 'Tracker runs'))
POWER_KINDS = (('rates', 'Success rates'), ('t2', 'Two groups (t)'), ('paired_t', 'Paired (t)'),
               ('demonstrations', 'Demonstrations'))
POWER_DEFAULT = {'kind': 'rates', 'p1': 0.5, 'p2': 0.8, 'd': 0.5, 'method': 'exact'}   # example inputs, labelled
CORRECTION_ITEMS = [(k, ST.CORRECTION_LABEL[k]) for k in ST.CORRECTIONS]


def _g(v, nd=3) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return '—'
    return f'{v:.{nd}g}'


def _p(v) -> str:
    if v is None:
        return '—'
    return '< 0.001' if v < 0.001 else f'{v:.3f}'


class _Sig(QObject):
    power = pyqtSignal(object, int)


class StatisticsPage(StudyPage):
    """Statistics: whether a difference between two rigs, two sets of runs or two policies' success rates is larger
    than chance, with the test and why, the effect size with its interval, p-values corrected across the figures, and
    a power calculator for how many evaluation episodes, draws or demonstrations a comparison needs."""
    STUDY = SS.SID
    TITLE = 'Statistics'
    HEADLINE = 'Clearest difference (B − A)'

    # ---------------------------------------------------------------- building
    def build_body(self, layout):
        self._power = None
        self._power_gen = 0
        self._preview = None
        self._filling = False
        self._sig = _Sig()
        self._sig.power.connect(self._power_done)
        self.source = SegmentedControl(list(SOURCES), 'rates')
        self.source.setToolTip('Where the two sides come from')
        self.source.changed.connect(lambda k: self._set({'source': k}, f'source: {dict(SOURCES)[k].lower()}'))
        layout.addWidget(row(label('Compare'), self.source))
        self.stack = QStackedWidget()
        self.stack.addWidget(self._rates_panel())
        self.stack.addWidget(self._variants_panel())
        self.stack.addWidget(self._experiment_panel())
        self.stack.addWidget(self._tracker_panel())
        self.stack.currentChanged.connect(self._fit_stack)       # as tall as the panel shown, not the tallest
        self._fit_stack(self.stack.currentIndex())
        layout.addWidget(self.stack)
        self.alpha = ScrubSpinBox('', 'plain', 3, '', step=0.005, width=70)
        self.alpha.setRange(0.001, 0.2)
        self.alpha.setToolTip('The significance level alpha (default 0.05, a DexVantage default)')
        self.alpha.committed.connect(lambda v: self._set({'alpha': round(float(v), 4)}, f'alpha {v:g}'))
        self.correction = QComboBox()
        for k, t in CORRECTION_ITEMS:
            self.correction.addItem(t, k)
        self.correction.setToolTip('The correction across the figures or comparisons (default Holm: the family-wise '
                                   'error rate; Benjamini-Hochberg: the false discovery rate)')
        self.correction.activated.connect(lambda _i: self._set({'correction': self.correction.currentData()},
                                                               f'{self.correction.currentText()} correction'))
        layout.addWidget(row(label('Alpha'), self.alpha, label('Correction across figures'), self.correction))
        layout.addWidget(label('Comparisons', bold=True))
        self.table = QTreeWidget()
        self.table.setRootIsDecorated(False)
        self.table.setHeaderLabels(['Figure', 'A', 'B', 'B − A [interval]', 'Effect size [interval]', 'Test', 'p',
                                    'p corrected', 'Verdict'])
        h = self.table.header()
        h.setStretchLastSection(True)
        for k in range(self.table.columnCount() - 1):
            h.setSectionResizeMode(k, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setMinimumHeight(110)
        self.table.setMaximumHeight(260)
        layout.addWidget(self.table)
        self.why = label('', wrap=True)
        layout.addWidget(self.why)
        self.assumptions = label('', muted=True, small=True, wrap=True)
        layout.addWidget(self.assumptions)
        layout.addWidget(self._power_panel())
        self.ctx.store.changed.connect(self._changed)
        self.ctx.store.project_replaced.connect(self._replaced)
        self._fill()
        self._compute_power()

    def _fit_stack(self, index: int):
        for k in range(self.stack.count()):
            w = self.stack.widget(k)
            pol = QSizePolicy.Policy.Preferred if k == index else QSizePolicy.Policy.Ignored
            w.setSizePolicy(QSizePolicy.Policy.Preferred, pol)
        self.stack.adjustSize()

    def _rates_panel(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        self.rates = QTableWidget(0, 3)
        self.rates.setHorizontalHeaderLabels(['Condition', 'Successes', 'Trials'])
        self.rates.horizontalHeader().setStretchLastSection(True)
        self.rates.verticalHeader().setVisible(False)
        self.rates.setMinimumHeight(110)
        self.rates.setMaximumHeight(170)
        self.rates.setToolTip('One row per condition (a policy, a rig): its successful episodes and all its episodes. '
                              'The first row is the reference.')
        self.rates.cellChanged.connect(lambda *_: self._rates_edited())
        self.b_add = button('Add condition', role='ghost', tip='A new row')
        self.b_add.clicked.connect(self.add_rate_row)
        self.b_del = button('Remove', role='ghost', tip='Remove the selected condition')
        self.b_del.clicked.connect(self.remove_rate_row)
        self.compare = SegmentedControl([('reference', 'Each against the first'), ('pairs', 'Every pair')],
                                        'reference')
        self.compare.changed.connect(lambda k: self._set({'compare': k}, 'compare ' + k))
        self.rate_test = SegmentedControl([('fisher', 'Fisher exact'), ('z', 'z test')], 'fisher')
        self.rate_test.setToolTip('Fisher\'s exact test (default: exact at any count, conservative) or the pooled '
                                  'two-proportion z test (fair when every expected count is at least 5)')
        self.rate_test.changed.connect(lambda k: self._set({'rate_test': k}, 'test ' + k))
        v.addWidget(self.rates)
        v.addWidget(row(self.b_add, self.b_del, label('Compare'), self.compare, label('Test'), self.rate_test))
        self.rates_note = label('', muted=True, small=True, wrap=True)
        v.addWidget(self.rates_note)
        v.addStretch(1)                                  # spare height below the controls, not between them
        return w

    def _variants_panel(self) -> QWidget:
        self.var_a, self.var_b = QComboBox(), QComboBox()
        for c, side in ((self.var_a, 'a'), (self.var_b, 'b')):
            c.setToolTip('A' if side == 'a' else 'B')
            c.activated.connect(lambda _i, c=c, side=side: self._set_variants({side: c.currentData()},
                                                                              f'variant {side.upper()}'))
        self.draws = ScrubSpinBox('', 'plain', 0, '', step=8.0, width=70)
        self.draws.setRange(2, SS.MAX_DRAWS)
        self.draws.setToolTip('Draws of the tolerance Monte Carlo per variant (default 32); both variants use the same '
                              'seed, so the draws are paired')
        self.draws.committed.connect(lambda v: self._set_variants({'n': int(round(v))}, f'{int(round(v))} draws'))
        self.vseed = ScrubSpinBox('', 'plain', 0, '', step=1.0, width=70)
        self.vseed.setRange(0, 2 ** 31 - 1)
        self.vseed.committed.connect(lambda v: self._set_variants({'seed': int(round(v))}, f'seed {int(round(v))}'))
        self.var_metrics = QLineEdit()
        self.var_metrics.setPlaceholderText('figures: default the requirements\' (metric ids, comma separated)')
        self.var_metrics.editingFinished.connect(self._metrics_edited)
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(row(label('A'), self.var_a, label('B'), self.var_b, label('Draws'), self.draws, label('Seed'),
                        self.vseed))
        fig_row = row(label('Figures'), self.var_metrics)
        fig_row.layout().setStretch(1, 100)            # the field takes the width, not the row's end stretch
        v.addWidget(fig_row)
        v.addWidget(label('Each variant\'s mounts are drawn within their tolerances with the same seed, and every '
                          'figure is recomputed per draw: the draws are paired by seed. Run starts it (long work).',
                          muted=True, small=True, wrap=True))
        v.addStretch(1)                                  # spare height below the controls, not between them
        return w

    def _experiment_panel(self) -> QWidget:
        self.ex_factor = QComboBox()
        self.ex_factor.setToolTip('The factor whose two levels to compare (from the last experiment run)')
        self.ex_factor.activated.connect(lambda _i: self._set_experiment({'mode': 'factor',
                                                                          'factor': self.ex_factor.currentData()},
                                                                         'experiment factor'))
        self.ex_levels = QLineEdit()
        self.ex_levels.setPlaceholderText('levels A, B (default: first and last; a continuous factor splits at '
                                          'its median)')
        self.ex_levels.editingFinished.connect(self._levels_edited)
        self.ex_note = label('', muted=True, small=True, wrap=True)
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(row(label('Factor'), self.ex_factor, self.ex_levels))
        v.addWidget(self.ex_note)
        v.addStretch(1)
        return w

    def _tracker_panel(self) -> QWidget:
        self.tr_a, self.tr_b = QLineEdit(), QLineEdit()
        for e, side in ((self.tr_a, 'a'), (self.tr_b, 'b')):
            e.setPlaceholderText(f'side {side.upper()}: run ids (comma separated) or tag:NAME')
            e.editingFinished.connect(lambda e=e, side=side: self._tracker_edited(side, e.text()))
        self.tr_runs = QTreeWidget()
        self.tr_runs.setRootIsDecorated(False)
        self.tr_runs.setHeaderLabels(['Run', 'Study', 'Seed', 'Started', 'Id'])
        self.tr_runs.setMinimumHeight(90)
        self.tr_runs.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self.b_use_a = button('Use as A', role='ghost', tip='The selected runs are side A')
        self.b_use_b = button('Use as B', role='ghost', tip='The selected runs are side B')
        self.b_use_a.clicked.connect(lambda: self._use_runs('a'))
        self.b_use_b.clicked.connect(lambda: self._use_runs('b'))
        self.tr_note = label('', muted=True, small=True, wrap=True)
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self.tr_runs)
        v.addWidget(row(self.b_use_a, self.b_use_b))
        v.addWidget(self.tr_a)
        v.addWidget(self.tr_b)
        v.addWidget(self.tr_note)
        v.addStretch(1)
        return w

    def _power_panel(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 8, 0, 0)
        v.addWidget(label('Power calculator: how many episodes, draws or demonstrations?', bold=True))
        self.pw_kind = SegmentedControl(list(POWER_KINDS), 'rates')
        self.pw_kind.changed.connect(lambda k: self._set_power({'kind': k}, dict(POWER_KINDS)[k].lower()))
        v.addWidget(self.pw_kind)
        g = QGridLayout()
        g.setHorizontalSpacing(8)

        def box(dec, unit, step, lo, hi, tip, key, scale=1.0):
            b = ScrubSpinBox('', 'plain', dec, unit, step=step, width=86)
            b.setRange(lo, hi)
            b.setToolTip(tip)
            b.committed.connect(lambda val, k=key, s=scale: self._set_power({k: round(float(val) / s, 6)}, k))
            return b
        self.pw_p1 = box(1, '%', 1.0, 0, 100, 'Success rate of condition A (an estimate or the rate you expect)',
                         'p1', 100.0)
        self.pw_p2 = box(1, '%', 1.0, 0, 100, 'Success rate of condition B: the difference you need to tell apart',
                         'p2', 100.0)
        self.pw_d = box(3, '', 0.05, 0.001, 10, 'The standardised effect d = difference / σ (paired: of the '
                        'differences)', 'd')
        self.pw_target = box(2, '', 0.01, 0.05, 0.99, 'The target power (default 0.80, a DexVantage default)', 'power')
        self.pw_n = box(0, '', 1.0, 0, 100000, 'Episodes (units) planned per condition: their power is shown; 0 for '
                        'none', 'n')
        self.pw_icc = box(3, '', 0.01, 0, 1, 'Intra-class correlation of the figure between one operator\'s '
                          'demonstrations (default 0: independent)', 'icc')
        self.pw_per_op = box(0, '', 1.0, 1, 10000, 'Demonstrations each operator records (default 1)',
                             'per_operator')
        self.pw_attr = box(1, '%', 1.0, 0, 95, 'Share of demonstrations the audit discards (default 0 %)', 'attrition',
                           100.0)
        self.pw_method = QComboBox()
        for k, t in (('exact', 'Exact (Fisher, every outcome)'), ('normal', 'Normal approximation (Fleiss)'),
                     ('arcsine', 'Arcsine (Cohen h)'), ('simulation', 'Simulation (seeded)')):
            self.pw_method.addItem(t, k)
        self.pw_method.activated.connect(lambda _i: self._set_power({'method': self.pw_method.currentData()},
                                                                    'method'))
        cells = ((label('p1'), self.pw_p1), (label('p2'), self.pw_p2), (label('d'), self.pw_d),
                 (label('Power'), self.pw_target), (label('Planned n'), self.pw_n), (label('Method'), self.pw_method),
                 (label('Per operator'), self.pw_per_op), (label('ICC ρ'), self.pw_icc),
                 (label('Discarded'), self.pw_attr))
        for i, (a, b) in enumerate(cells):
            g.addWidget(a, i // 3, 2 * (i % 3))
            g.addWidget(b, i // 3, 2 * (i % 3) + 1)
        v.addLayout(g)
        self.pw_answer = label('', wrap=True)
        self.pw_answer.setFont(mono(12))
        v.addWidget(self.pw_answer)
        self.pw_curve = CurveChart(height=200)
        self.pw_curve.title = 'Power against n per condition'
        self.pw_curve.xname, self.pw_curve.xunit, self.pw_curve.yunit = 'n', '', ''
        v.addWidget(self.pw_curve)
        self.pw_assume = label('', muted=True, small=True, wrap=True)
        v.addWidget(self.pw_assume)
        return w

    # ---------------------------------------------------------------- settings (one undo step each)
    def _settings(self) -> dict:
        return dict(self.ctx.store.project.analysis.studies.get(SS.SID) or {})

    def _set(self, upd: dict, what: str, merge=None) -> bool:
        """Lay upd over the stored settings as one undo step (successive edits of the same field merge, as a
        dragged number does)."""
        if self._filling:
            return False
        cur = self._settings()
        new = {**cur, **upd}
        if new == cur:
            return False

        def fn(p):
            p.analysis.studies[SS.SID] = {**(p.analysis.studies.get(SS.SID) or {}), **upd}
        self.ctx.store.edit('analysis', fn, f'Statistics · {what}', merge_key=(SS.SID, merge or tuple(sorted(upd))))
        return True

    def _set_sub(self, key: str, upd: dict, what: str) -> bool:
        cur = self._settings().get(key) or {}
        return self._set({key: {**cur, **upd}}, what, (key,) + tuple(sorted(upd)))

    def _set_variants(self, upd, what):
        if self._set_sub('variants', upd, what):
            self.suggestion_ready(False)

    def _set_experiment(self, upd, what):
        self._set_sub('experiment', upd, what)

    def _set_power(self, upd, what):
        self._preview = None
        if 'n' in upd and not upd['n']:
            upd = {**upd, 'n': None}
        st = self._settings().get('power') or dict(POWER_DEFAULT)
        if self._set({'power': {**POWER_DEFAULT, **st, **upd}}, f'power {what}', ('power',) + tuple(sorted(upd))):
            self._compute_power()

    def _metrics_edited(self):
        ids = [x.strip() for x in self.var_metrics.text().split(',') if x.strip()]
        self._set_variants({'metrics': ids}, 'figures')

    def _levels_edited(self):
        text = self.ex_levels.text().strip()
        if not text:
            self._set_experiment({'levels': None}, 'levels')
            return
        parts = [x.strip() for x in text.split(',')]
        if len(parts) != 2:
            self.ex_note.setText('Give two levels separated by a comma, or leave it empty.')
            return
        vals = []
        for x in parts:
            try:
                vals.append(float(x))
            except ValueError:
                vals.append({'true': True, 'false': False}.get(x.lower(), x))
        self._set_experiment({'levels': vals}, 'levels')

    def _tracker_edited(self, side, text):
        text = text.strip()
        sel = {'tag': text[4:].strip()} if text.lower().startswith('tag:') else \
            [x.strip() for x in text.split(',') if x.strip()]
        self._set_sub('tracker', {side: sel}, f'tracker side {side.upper()}')

    def _use_runs(self, side):
        ids = [it.text(4) for it in self.tr_runs.selectedItems()]
        if not ids:
            self._msg('Select one or more runs first.')
            return
        (self.tr_a if side == 'a' else self.tr_b).setText(', '.join(ids))
        self._tracker_edited(side, ', '.join(ids))

    # ---------------------------------------------------------------- the success-count table
    def add_rate_row(self, name: str = '', k=None, n=None):
        self._filling = True
        r = self.rates.rowCount()
        self.rates.insertRow(r)
        for c, v in enumerate((name or f'Condition {r + 1}', '' if k is None else str(k),
                               '' if n is None else str(n))):
            self.rates.setItem(r, c, QTableWidgetItem(v))
        self._filling = False
        self._rates_edited()

    def remove_rate_row(self):
        r = self.rates.currentRow()
        if r < 0:
            r = self.rates.rowCount() - 1
        if r >= 0:
            self.rates.removeRow(r)
            self._rates_edited()

    def table_rows(self) -> tuple[list, str]:
        """The rows the table holds as success counts, and a plain sentence for any that cannot be read ('' when all
        can)."""
        out = []
        for r in range(self.rates.rowCount()):
            cell = [(self.rates.item(r, c).text().strip() if self.rates.item(r, c) else '') for c in range(3)]
            if not cell[1] and not cell[2]:
                continue
            try:
                k, n = int(float(cell[1])), int(float(cell[2]))
            except ValueError:
                return out, f'Row {r + 1}: successes and trials must be whole numbers.'
            if n < 1 or not 0 <= k <= n:
                return out, f'Row {r + 1}: successes must be between 0 and the trials (at least 1).'
            out.append({'condition': cell[0] or f'Condition {r + 1}', 'successes': k, 'trials': n})
        return out, ''

    def _rates_edited(self):
        if self._filling:
            return
        rows, bad = self.table_rows()
        self.rates_note.setText(bad)
        if bad:
            return
        if self._set({'rates': rows, 'source': 'rates'}, 'success counts') and len(rows) >= 2:
            self.run()

    def set_rates(self, rows) -> bool:
        """Fill the table with [(condition, successes, trials)] and store it (one undo step)."""
        self._filling = True
        self.rates.setRowCount(0)
        for name, k, n in rows:
            r = self.rates.rowCount()
            self.rates.insertRow(r)
            for c, v in enumerate((name, str(k), str(n))):
                self.rates.setItem(r, c, QTableWidgetItem(v))
        self._filling = False
        self._rates_edited()
        return not self.rates_note.text()

    # ---------------------------------------------------------------- filling from the project
    def _fill(self):
        self._filling = True
        try:
            st = self._settings()
            src = st.get('source') or 'rates'
            self.source.set_value(src)
            self.stack.setCurrentIndex([k for k, _ in SOURCES].index(src) if src in dict(SOURCES) else 0)
            self.alpha.set_value(float(st.get('alpha', ST.ALPHA)))
            i = self.correction.findData(st.get('correction') or ST.CORRECTION)
            self.correction.setCurrentIndex(max(0, i))
            self.compare.set_value(st.get('compare') or 'reference')
            self.rate_test.set_value(st.get('rate_test') or 'fisher')
            rows = st.get('rates') or []
            if [self._row_tuple(r) for r in rows] != [self._row_tuple(r) for r in self.table_rows()[0]]:
                self.rates.setRowCount(0)
                for r in rows:
                    i = self.rates.rowCount()
                    self.rates.insertRow(i)
                    for c, v in enumerate((r.get('condition', ''), str(r.get('successes', '')),
                                           str(r.get('trials', '')))):
                        self.rates.setItem(i, c, QTableWidgetItem(str(v)))
            p = self.ctx.store.project
            v = st.get('variants') or {}
            for c, side in ((self.var_a, 'a'), (self.var_b, 'b')):
                c.clear()
                c.addItem('default' + (' (the active variant)' if side == 'a' else ' (the first other)'), '')
                for var in p.variants:
                    c.addItem(var.name, var.uid)
                c.setCurrentIndex(max(0, c.findData(v.get(side) or '')))
            self.draws.set_value(float(v.get('n', SS.VARIANT_DEFAULTS['n'])))
            self.vseed.set_value(float(v.get('seed', SS.VARIANT_DEFAULTS['seed'])))
            self.var_metrics.setText(', '.join(v.get('metrics') or []))
            self._fill_experiment()
            self._fill_tracker()
            pw = {**POWER_DEFAULT, **(self._preview or st.get('power') or {})}     # a preview shows its plan
            self.pw_kind.set_value(pw.get('kind', 'rates'))
            self.pw_p1.set_value(100 * float(pw.get('p1') or 0))
            self.pw_p2.set_value(100 * float(pw.get('p2') or 0))
            self.pw_d.set_value(float(pw.get('d') or 0.5))
            self.pw_target.set_value(float(pw.get('power') or st.get('power_target') or ST.POWER))
            self.pw_n.set_value(float(pw.get('n') or 0))
            self.pw_icc.set_value(float(pw.get('icc') or 0))
            self.pw_per_op.set_value(float(pw.get('per_operator') or 1))
            self.pw_attr.set_value(100 * float(pw.get('attrition') or 0))
            self.pw_method.setCurrentIndex(max(0, self.pw_method.findData(pw.get('method') or 'exact')))
            kind = pw.get('kind', 'rates')
            rates_like = kind == 'rates' or (kind == 'demonstrations' and pw.get('demo_kind') == 'binary')
            for b in (self.pw_p1, self.pw_p2):
                b.setEnabled(rates_like)
            self.pw_d.setEnabled(not rates_like)
            self.pw_method.setEnabled(kind == 'rates')
            for b in (self.pw_icc, self.pw_per_op, self.pw_attr):
                b.setEnabled(kind == 'demonstrations')
        finally:
            self._filling = False

    @staticmethod
    def _row_tuple(r):
        return (str(r.get('condition', '')), r.get('successes'), r.get('trials'))

    def _fill_experiment(self):
        doc = self.ctx.analysis.results.get('experiments')
        ex = self._settings().get('experiment') or {}
        self.ex_factor.clear()
        if not isinstance(doc, dict) or not doc.get('factors'):
            self.ex_note.setText('No experiment result yet: run a designed experiment (Analysis ▸ Designed '
                                 'experiments) and come back; its runs are split by a factor\'s two levels.')
            return
        for f in doc['factors']:
            self.ex_factor.addItem(f'{f.get("title", f["path"])} ({f.get("kind")})', f['path'])
        i = self.ex_factor.findData(ex.get('factor') or '')
        self.ex_factor.setCurrentIndex(max(0, i))
        lv = ex.get('levels')
        self.ex_levels.setText(', '.join(str(x) for x in lv) if lv else '')
        self.ex_note.setText(f'The last experiment: {doc.get("method_label", "")}, {len(doc.get("runs") or [])} runs. '
                             'Runs with the same other factors pair up (a full factorial); otherwise they are compared '
                             'unpaired.')

    def _fill_tracker(self):
        self.tr_runs.clear()
        tr = self._settings().get('tracker') or {}
        for e, side in ((self.tr_a, 'a'), (self.tr_b, 'b')):
            sel = tr.get(side)
            e.setText(f'tag:{sel["tag"]}' if isinstance(sel, dict) and sel.get('tag') else
                      ', '.join(sel) if isinstance(sel, list) else (sel or ''))
        try:
            store = SC.tracker_store(self.ctx.store.project, tr.get('folder'))
            runs = store.runs(status='finished')
        except Exception as e:                   # noqa: BLE001 — no store: said in words
            self.tr_note.setText(str(e) if isinstance(e, ST.StatsError) else 'The run store could not be read.')
            return
        for r in runs[-200:]:
            it = QTreeWidgetItem([r.get('name') or r.get('title') or '', r.get('study') or '',
                                  _g(SC._run_seed(r), 6), (r.get('started_utc') or '')[:19],      # noqa: SLF001
                                  r.get('run_id') or ''])
            self.tr_runs.addTopLevelItem(it)
        self.tr_note.setText(f'{len(runs)} finished runs in {store.name}. Two runs that kept their draws (a tolerance '
                             'Monte Carlo) are compared draw by draw; groups of runs one value per run, paired by seed '
                             'when the seeds match.')

    def _changed(self, sections):
        if 'analysis' in sections or 'variants' in sections:
            self._fill()

    def _replaced(self):
        self.doc = None
        self._preview = None
        self.table.clear()
        self.suggestion_ready(False)
        self._fill()
        self._compute_power()
        self.empty()

    def empty(self):
        self.head.set('—', '', 'warning', 'not run yet', 'Enter success counts (two or more conditions), or pick two '
                      'variants, an experiment\'s factor or tracker runs, then Run.')

    # ---------------------------------------------------------------- running
    def run(self):
        st = self._settings()
        extra = None
        if (st.get('source') or 'rates') == 'experiments':
            doc = self.ctx.analysis.results.get('experiments')
            ex = dict(st.get('experiment') or {})
            if ex.get('mode', 'factor') == 'factor' and isinstance(doc, dict) and not ex.get('result'):
                ex['result'] = {k: v for k, v in doc.items() if k not in ('cached', 'revision')}
                extra = {'experiment': ex}
        self.ctx.analysis.run(SS.SID, extra)

    # ---------------------------------------------------------------- the result
    def show_result(self, doc):
        try:
            h = doc.get('headline') or {}
            v = h.get('value')
            fig = '—' if v is None else (_sg(v, '.1f') if h.get('unit') == 'pp' and h.get('kind') != 'samples' else
                                         f'{v:.0f}' if h.get('unit') == 'per condition' else _sg(v, '.3g'))
            tip = ('The comparison with the smallest corrected p-value. Status: clear difference when its corrected p '
                   'is at most alpha, else no clear difference (which does not show the two are equal).')
            self.head.set(fig, h.get('unit', ''), h.get('status', 'warning'), h.get('word', ''),
                          h.get('text', '') + (f' {doc["advice"]}' if doc.get('advice') else ''), tip)
            self.table.clear()
            for c in doc.get('comparisons') or []:
                self.table.addTopLevelItem(self._item(c))
            why = []
            for c in (doc.get('comparisons') or [])[:1]:
                if c.get('why'):
                    lab = c.get('test_label', c.get('test', ''))
                    why.append(c['why'] if lab.lower() in c['why'].lower() else f'Test: {lab}. {c["why"]}')
            corr = doc.get('correction') or {}
            if corr:
                why.append(f'p corrected with {corr.get("label")} across {corr.get("m")} '
                           f'comparison{"" if corr.get("m") == 1 else "s"} (it controls {corr.get("controls")}).')
            self.why.setText(' '.join(why))
            assume = doc.get('assumptions') or []
            self.assumptions.setText(('Assumptions: ' + '; '.join(assume) + '. ' if assume else '') +
                                     ' '.join(doc.get('notes') or []) + ' ' + (doc.get('out_of_scope') or ''))
            if doc.get('power'):
                self._show_power(doc['power'])
        except (KeyError, TypeError, ValueError):
            traceback.print_exc()
        self.suggestion_ready(bool(doc.get('suggestion')))

    def _item(self, c) -> QTreeWidgetItem:
        unit = c.get('unit', '')
        if c.get('error'):
            it = QTreeWidgetItem([c.get('title', ''), '—', '—', '—', '—', '—', '—', '—', c['verdict']['text']])
            it.setForeground(8, QColor(STATUS['critical']))
            return it
        if c.get('kind') == 'rates':
            a = f'{c["a"]["successes"]}/{c["a"]["trials"]} ({100 * c["a"]["rate"]:.0f} %)'
            b = f'{c["b"]["successes"]}/{c["b"]["trials"]} ({100 * c["b"]["rate"]:.0f} %)'
            d = c['difference']
            diff = f'{_sg(100 * d["value"], ".1f")} pp [{_sg(100 * d["ci"][0], ".1f")}, {_sg(100 * d["ci"][1], ".1f")}]'
            o = c['odds_ratio']
            eff = f'OR {o["value"]:.3g} [{o["ci"][0]:.3g}, {o["ci"][1]:.3g}]'
        else:
            a = f'{_g(c["a"]["mean"])} ± {_g(c["a"]["sd"], 2)} {unit} (n {c["a"]["n"]})'.replace('  ', ' ')
            b = f'{_g(c["b"]["mean"])} ± {_g(c["b"]["sd"], 2)} {unit} (n {c["b"]["n"]})'.replace('  ', ' ')
            d = c['difference']
            diff = (f'{_sg(d["value"], ".3g")} [{_sg(d["ci"][0], ".3g")}, {_sg(d["ci"][1], ".3g")}] '
                    f'{delta_unit(unit)}').strip()
            s = c.get('standardised') or {}
            if s.get('d') is None:
                eff = '—'
            elif c.get('paired'):
                eff = f'd_z {s["d"]:.2f}' + (f' [{s["ci"][0]:.2f}, {s["ci"][1]:.2f}]' if s.get('ci') else '')
            else:
                eff = f'g {s["g"]:.2f}' + (f' [{s["g_ci"][0]:.2f}, {s["g_ci"][1]:.2f}]' if s.get('g_ci') else '')
        v = c.get('verdict') or {}
        it = QTreeWidgetItem([c.get('title', ''), a, b, diff, eff, c.get('test_label', c.get('test', '')),
                              _p(c.get('p')), _p(c.get('p_adjusted')), v.get('word', '')])
        for col in (1, 2, 3, 4, 6, 7):
            it.setFont(col, mono(11))
        it.setForeground(8, QColor(STATUS.get(v.get('status'), STATUS['warning'])))
        it.setToolTip(8, v.get('text', ''))
        it.setToolTip(5, (c.get('method') or '') + ('. ' if c.get('method') else '') + (c.get('why') or ''))
        refs = c.get('reference_tests') or []
        if refs:
            it.setToolTip(6, 'For reference: ' + '; '.join(f'{r["test"]} p = {_p(r.get("p"))}' for r in refs))
        if c.get('kind') == 'rates':
            it.setToolTip(3, 'p2 − p1 with Newcombe\'s hybrid score interval; Wilson intervals of each rate: '
                             f'A [{100 * c["a"]["wilson"][0]:.0f}–{100 * c["a"]["wilson"][1]:.0f} %], '
                             f'B [{100 * c["b"]["wilson"][0]:.0f}–{100 * c["b"]["wilson"][1]:.0f} %]')
        else:
            it.setToolTip(3, f'{c["difference"].get("method", "")} bootstrap interval of the mean difference '
                             f'({c["difference"].get("n_resamples")} resamples, seed {c["difference"].get("seed")})')
        return it

    # ---------------------------------------------------------------- the power calculator
    def _power_settings(self) -> dict:
        st = self._settings()
        pw = {**POWER_DEFAULT, **(self._preview or st.get('power') or {})}
        return {**st, 'power': pw}

    def _compute_power(self, block: bool = False):
        """The power calculation for the settings as they are now, on a worker thread (block: in this thread)."""
        self._power_gen += 1
        gen = self._power_gen
        try:
            st = SS.settings_of(self._power_settings())
        except ST.StatsError as e:
            self._power_failed(str(e))
            return

        def work():
            try:
                res = SS.power_of(st)
            except ST.StatsError as e:
                res = {'error': str(e)}
            except Exception as e:                      # noqa: BLE001 — explained, the traceback logged
                traceback.print_exc()
                res = {'error': f'The power calculation stopped ({type(e).__name__}).'}
            if block:
                self._power_done(res, gen)
            else:
                try:
                    self._sig.power.emit(res, gen)
                except RuntimeError:
                    pass
        if block:
            work()
        else:
            threading.Thread(target=work, daemon=True).start()

    def _power_failed(self, text):
        self._power = None
        self.pw_answer.setText(text)
        self.pw_curve.set_data([], (0, 1), (0, 1))

    def _power_done(self, res, gen):
        if gen != self._power_gen:
            return
        if not res or res.get('error'):
            self._power_failed((res or {}).get('error', ''))
            return
        self._power = res
        self._show_power(res)

    def _show_power(self, pw: dict):
        cv = pw.get('curve') or {}
        ns, ps = cv.get('n') or [], cv.get('power') or []
        target = pw.get('power', ST.POWER)
        n = pw.get('n_per_group')
        markers = []
        if n is not None and pw.get('power_at_n') is not None:
            markers.append((n, pw['power_at_n'], STATUS['good'], f'{n} needed'))
        plan = pw.get('planned')
        if plan:
            ok = plan['power'] >= target
            markers.append((plan['n'], plan['power'], STATUS['good' if ok else 'serious'], f'{plan["n"]} planned'))
        if ns:
            self.pw_curve.set_data([('power', ns, ps, THEME.color('ink').name(), 'line')],
                                   (min(ns), max(max(ns), plan['n'] if plan else 0)), (0.0, 1.0),
                                   hlines=[(target, f'target {target:g}' + (' (default)' if target == ST.POWER
                                                                             else ''))],
                                   markers=markers, xlabel='n per condition', ylabel='power')
        txt = pw.get('answer', '')
        if plan:
            txt += f' The plan of {plan["n"]} gives {100 * plan["power"]:.1f} % power.'
        if self._preview:
            txt = 'Preview: ' + txt
        self.pw_answer.setText(txt)
        assume = pw.get('assumptions') or []
        self.pw_assume.setText(('Assumptions: ' + '; '.join(assume) + '. ' if assume else '') +
                               (pw.get('note', '') + ' ' if pw.get('note') else '') +
                               (pw.get('not_modelled', '') + ' ' if pw.get('not_modelled') else '') +
                               (pw.get('out_of_scope') or ST.OUT_OF_SCOPE))

    # ---------------------------------------------------------------- Apply, Preview, Explain, Export
    def apply(self):
        sg = (self.doc or {}).get('suggestion')
        if not sg:
            return None
        upd = sg['settings']

        def fn(p):
            cur = dict(p.analysis.studies.get(SS.SID) or {})
            for k, v in upd.items():
                cur[k] = {**(cur.get(k) or {}), **v} if isinstance(v, dict) else v
            p.analysis.studies[SS.SID] = cur
        self.ctx.store.edit('analysis', fn, f'Statistics · plan {sg["n"]} ' +
                            ('draws' if sg['kind'] == 'draws' else 'per condition'))
        self._preview = None
        self.suggestion_ready(False)
        self._compute_power()
        self._msg(sg['text'], 8000)
        return sg

    def preview(self):
        sg = (self.doc or {}).get('suggestion')
        if not sg:
            return
        if self._preview is not None:
            self._preview = None
            self._msg('Preview off.')
        elif sg['kind'] == 'power':
            self._preview = {**POWER_DEFAULT, **sg['settings']['power']}
            self._msg('Preview: the suggested plan in the power calculator (not stored). ' + sg['text'], 10000)
        else:
            self._msg(sg['text'], 10000)
            return
        self._fill()                                         # the calculator's fields show what it computes
        self._compute_power()

    def explain(self):
        box = QMessageBox(self)
        box.setWindowTitle('How the statistics are computed')
        box.setText('Success counts are compared with Fisher\'s exact test (exact at any number of episodes), the '
                    'difference of rates with Newcombe\'s interval and the odds ratio with Woolf\'s. Figures over '
                    'tolerance draws, experiment runs or tracker runs are compared with a permutation test of the mean '
                    'difference (sign flips when paired, relabelling when not; exact when there are at most 9 999 '
                    'rearrangements), the difference with a BCa bootstrap interval and the effect size (d_z or '
                    'Hedges\' g) with a noncentral-t interval; the t and rank tests are shown for reference. The '
                    'p-values are corrected across the figures (Holm by default). A difference that is not clear '
                    'does not show that the two are equal: the page gives the smallest difference the data detect. '
                    'Power: Fisher\'s exact test by enumerating every outcome, or the normal approximation with '
                    'continuity correction; t tests through the noncentral t. Sequential testing and early stopping '
                    'are out of scope. docs/models/stats.md has the equations and the validation.')
        box.setModal(False)
        box.show()

    def export(self, path=None):
        if self._power is not None:
            if self.doc is None:
                self.doc = {'schema': f'dexvantage.{SS.SID}/1', 'study': SS.SID, 'source': None, 'comparisons': [],
                            'power': self._power, 'headline': SS.headline({'power': self._power}),
                            'out_of_scope': ST.OUT_OF_SCOPE}
            else:
                self.doc = {**self.doc, 'power': self._power}
        return super().export(path)

    def page_context(self) -> dict:
        out = super().page_context()
        out['power'] = (self._power or {}).get('answer')
        return out

    def _msg(self, text, ms=5000):
        if self.ctx.window is not None:
            self.ctx.window.msg(text, ms)


register('statistics', 'Statistics', StatisticsPage, menu='Analysis', icon='compare', action_id=None,
         workspace=None, order=97)
