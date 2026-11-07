"""The Tolerances page (spec §7.11, §8.5): every sensor's mount tolerance (or calibration covariance) drawn many times
and carried through the requirement metrics (core/studies/tolerance.py, core/tolerance.py).

What it shows: the headline, the requirement met by the fewest draws as its pessimistic 5 % figure (P5, or P95 for a
lower-is-better metric) with a status word from the share of draws meeting it (holds from 95 %, marginal from 80 %,
often fails from 50 %: DexVantage defaults); per figure a distribution chart (every draw, the P5–P95 band, the median,
the nominal design and the requirement's threshold) and a table of figure ± spread (mean ± σ, [P5–P95], the share
meeting the requirement with its Wilson 95 % interval, the mount that contributes most); the contribution of each
mount to the headline figure (rank correlation); and a tolerance editor per sensor (± mm, ± °, one undo step each).
Quick-study headlines on the other pages stay as they are (the design as it is); this page is where their spread
under tolerance shows.

Apply halves the position or rotation tolerance of the unlocked mount that contributes most to the headline figure
(one undo step); Preview draws that mount's perturbed frusta, as they are and as they would be, in the 3D view.
While the page is shown, the 3D view draws every sensor's perturbed frusta (to scale, out to the nearest task volume)
and the Camera view the task volumes' centres as each draw of that camera would see them. Export writes the result
with provenance; Export perturbations writes n perturbed extrinsics as JSON and .npz (core/exporters/perturbations.py)
for Isaac Lab domain randomisation and the 3DGS renderer.

Known limits: see the study's. The frusta and Camera view marks show at most 24 draws per sensor (96 in Preview).
"""
from __future__ import annotations

import math
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (QFileDialog, QGridLayout, QHeaderView, QMessageBox, QTreeWidget, QTreeWidgetItem,
                             QWidget)

from ...core import tolerance as TL
from ...core.studies import tolerance as TS
from ..charts import BarList, ChartBase
from ..theme import STATUS, THEME, mono, sans
from ..widgets import SegmentedControl, ScrubSpinBox, button, label
from . import register
from .base import StudyPage, row

OVERLAY_DRAWS = 24
PREVIEW_DRAWS = 96
DEFAULT_DEPTH_M = 0.5            # the frusta's reach when the project has no task volume (drawing only)


def fnum(v, unit='') -> str:
    if v is None:
        return '—'
    a = abs(v)
    s = f'{v:.4g}' if a >= 100 or a == 0 else f'{v:.3f}' if a < 1 else f'{v:.2f}'
    return f'{s}{unit}' if unit == '°' else f'{s} {unit}'.strip()


def spread_text(m: dict) -> str:
    """'mean ± σ unit [P5–P95]' (the interval named)."""
    if m.get('mean') is None:
        return '—'
    u = m.get('unit', '')
    return f'{fnum(m["mean"])} ± {fnum(m["std"])} {u} [P5–P95 {fnum(m["p5"])}–{fnum(m["p95"])}]'.replace('  ', ' ')


class DistributionChart(ChartBase):
    """One row per figure: every draw as a dot, the P5–P95 band, the median tick, the nominal design (diamond) and
    the requirement's threshold (a line in its status colour); each row on its own scale."""
    ROW_H = 38

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = 'Figures over the draws'
        self.rows = []                   # (title, metric dict, [(threshold, sev)])
        self.setMinimumHeight(60)

    def set_rows(self, rows):
        self.rows = rows
        self.setMinimumHeight(28 + self.ROW_H * max(1, len(rows)))
        self.tip = None
        self.update()

    def _plot(self, i):
        y = 24 + i * self.ROW_H
        return QRectF(150, y + 14, max(self.width() - 160, 20), self.ROW_H - 20)

    @staticmethod
    def _range(m, thr):
        vals = [v for v in m.get('values') or [] if v is not None]
        xs = vals + [t for t, _ in thr] + ([m['nominal']] if m.get('nominal') is not None else [])
        if not xs:
            return None
        lo, hi = min(xs), max(xs)
        pad = (hi - lo) * 0.06 or max(abs(hi) * 0.01, 1e-9)
        return lo - pad, hi + pad

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.paint_title(p, 'dots: draws · band: P5–P95 · ◆ nominal')
        for i, (title, m, thr) in enumerate(self.rows):
            r = self._plot(i)
            p.setFont(sans(11))
            p.setPen(THEME.color('ink'))
            p.drawText(QRectF(0, r.top() - 14, 146, 14), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       p.fontMetrics().elidedText(title, Qt.TextElideMode.ElideRight, 146))   # full title: tooltip
            p.setFont(mono(10))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(r.left(), r.top() - 14, r.width(), 13), Qt.AlignmentFlag.AlignRight |
                       Qt.AlignmentFlag.AlignVCenter, spread_text(m))
            rng = self._range(m, thr)
            if rng is None:
                p.drawText(QRectF(0, r.top(), 146, r.height()), Qt.AlignmentFlag.AlignVCenter, 'no value')
                continue
            lo, hi = rng

            def fx(v, r=r, lo=lo, hi=hi):
                return r.left() + (v - lo) / (hi - lo) * r.width()
            p.setPen(QPen(THEME.color('strong'), 1))
            p.drawLine(QPointF(r.left(), r.bottom()), QPointF(r.right(), r.bottom()))
            if m.get('p5') is not None:
                band = QColor(THEME.tok['ink2'])
                band.setAlphaF(0.28)
                p.fillRect(QRectF(fx(m['p5']), r.top() + 3, max(fx(m['p95']) - fx(m['p5']), 1), r.height() - 6), band)
                p.setPen(QPen(THEME.color('ink'), 1.5))
                p.drawLine(QPointF(fx(m['p50']), r.top() + 1), QPointF(fx(m['p50']), r.bottom() - 1))
            dot = QColor(THEME.tok['ink2'])
            dot.setAlphaF(0.55)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(dot)
            vals = [v for v in m.get('values') or [] if v is not None]
            for k, v in enumerate(vals):
                jitter = ((k * 37) % 11 - 5) / 5.0 * (r.height() / 2 - 4)
                p.drawEllipse(QPointF(fx(v), r.center().y() + jitter), 1.6, 1.6)
            for t, sev in thr:
                p.setPen(QPen(QColor(STATUS.get(sev, STATUS['warning'])), 1.5))
                p.drawLine(QPointF(fx(t), r.top() - 2), QPointF(fx(t), r.bottom() + 2))
            if m.get('nominal') is not None:
                x, y = fx(m['nominal']), r.center().y()
                p.setPen(QPen(THEME.color('raised'), 1))
                p.setBrush(THEME.color('ink'))
                p.drawPolygon(QPolygonF([QPointF(x, y - 5), QPointF(x + 5, y), QPointF(x, y + 5), QPointF(x - 5, y)]))
        self.paint_tip(p)
        p.end()

    def mouseMoveEvent(self, e):
        i = int((e.position().y() - 24) // self.ROW_H)
        self.hover = i if 0 <= i < len(self.rows) and e.position().y() >= 24 else None
        if self.hover is None:
            self.tip = None
        else:
            title, m, thr = self.rows[i]
            u = m.get('unit', '')
            lines = [title, f'nominal {fnum(m.get("nominal"), u)} · P50 {fnum(m.get("p50"), u)}',
                     f'{m.get("valid", 0)} draws with a value']
            for rq in m.get('requirements') or []:
                if rq.get('share') is not None:
                    lines.append(f'{rq["op"]} {fnum(rq["threshold"], u)}: {100 * rq["share"]:.0f} % meet it')
            self.set_tip(QPointF(e.position().x(), self._plot(i).top()), spread_text(m), *lines)
        self.update()


class TolerancePage(StudyPage):
    """Monte Carlo tolerances: every sensor's mount is moved within its tolerance (a ± bound per axis) or drawn from
    its calibration covariance, many times, and each requirement's figure is recomputed by its own study for every
    draw. The headline is the requirement met by the fewest draws, shown as its pessimistic 5 % figure."""
    STUDY = TS.SID
    TITLE = 'Tolerances'
    HEADLINE = 'Worst requirement under mount tolerances'

    def build_body(self, layout):
        self._preview = False
        self._cache = None
        self._sig = _Sig()
        self._sig.done.connect(self._export_done)
        self.n_box = ScrubSpinBox('', 'plain', 0, '', step=8.0, width=70)
        self.n_box.setRange(2, TS.MAX_N)
        self.n_box.setToolTip('Draws of every mount (default 64; each draw re-runs the studies of the figures)')
        self.n_box.committed.connect(lambda v: self._set_setting('n', int(round(v)), f'{int(round(v))} draws'))
        self.seed_box = ScrubSpinBox('', 'plain', 0, '', step=1.0, width=70)
        self.seed_box.setRange(0, 2 ** 31 - 1)
        self.seed_box.setToolTip('The draws\' seed (default 0): the same seed gives the same draws')
        self.seed_box.committed.connect(lambda v: self._set_setting('seed', int(round(v)), f'seed {int(round(v))}'))
        self.dist = SegmentedControl([('uniform', 'Uniform'), ('normal', 'Normal')], 'uniform')
        self.dist.setToolTip('Inside a ± tolerance: uniform (default) or normal with the bound at 2 σ, truncated '
                             'there')
        self.dist.changed.connect(lambda k: self._set_setting('distribution', k, f'{k} within the tolerances'))
        self.b_perturb = button('Export perturbations…', role='ghost', icon_name='export',
                                tip='n perturbed extrinsics as JSON and .npz for Isaac Lab and the 3DGS renderer')
        self.b_perturb.clicked.connect(lambda: self.export_perturbations())
        layout.addWidget(row(label('Draws'), self.n_box, label('Seed'), self.seed_box, label('Within ±'), self.dist,
                             self.b_perturb))
        self.chart = DistributionChart()
        layout.addWidget(self.chart)
        layout.addWidget(label('Figure ± spread', bold=True))
        self.table = QTreeWidget()
        self.table.setRootIsDecorated(False)
        self.table.setHeaderLabels(['Figure', 'Nominal', 'Mean ± σ', 'P5–P95', 'Meets', 'Largest contribution'])
        h = self.table.header()
        h.setStretchLastSection(False)
        for k in range(self.table.columnCount()):
            h.setSectionResizeMode(k, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table)
        self.contrib = BarList()
        self.contrib.title = 'Contribution to the headline figure'
        self.contrib.unit = '%'
        self.contrib.what = 'of the spread (rank correlation)'
        self.contrib.vmax = 100.0
        self.contrib.picked.connect(lambda uid: self.ctx.store.select(('sensor', uid)))
        layout.addWidget(self.contrib)
        self.contrib_note = label('', muted=True, small=True, wrap=True)
        layout.addWidget(self.contrib_note)
        layout.addWidget(label('Mount tolerances (± per axis)', bold=True))
        self.editor = QWidget()
        self.grid = QGridLayout(self.editor)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(8)
        self._rows = {}
        self._uids = None
        layout.addWidget(self.editor)
        self.notes = label('', muted=True, small=True, wrap=True)
        layout.addWidget(self.notes)
        self.ctx.store.changed.connect(self._changed)
        self.ctx.store.project_replaced.connect(self._replaced)
        self._fill_settings()
        self._fill_editor()

    # ---------------------------------------------------------------- settings and the editor
    def _settings(self) -> dict:
        return dict(self.ctx.store.project.analysis.studies.get(TS.SID) or {})

    def _fill_settings(self):
        st = self._settings()
        self.n_box.set_value(float(st.get('n', TS.DEFAULTS['n'])))
        self.seed_box.set_value(float(st.get('seed', TS.DEFAULTS['seed'])))
        self.dist.set_value(st.get('distribution', TS.DEFAULTS['distribution']))

    def _set_setting(self, key, value, what):
        if self._settings().get(key) == value:
            return

        def fn(p):
            d = dict(p.analysis.studies.get(TS.SID) or {})
            d[key] = value
            p.analysis.studies[TS.SID] = d
        self.ctx.store.edit('analysis', fn, f'Tolerances · {what}', merge_key=(TS.SID, key))

    def _fill_editor(self):
        p = self.ctx.store.project
        sensors = [s for s in p.variant.sensors if s.enabled]
        uids = [s.uid for s in sensors]
        if uids != self._uids:
            while self.grid.count():
                w = self.grid.takeAt(0).widget()
                if w is not None:
                    w.setParent(None)
            self._rows = {}
            for k, head in enumerate(('Sensor', 'Drawn from', '± position', '± rotation', 'Contribution')):
                self.grid.addWidget(label(head, muted=True, small=True), 0, k)
            for i, s in enumerate(sensors, start=1):
                name = label(s.name + (' (locked)' if s.locked else ''))
                src = label('', muted=True, small=True)
                pos = ScrubSpinBox('', 'plain', 2, 'mm', step=0.1, width=86)
                pos.setRange(0, 1000)
                pos.setToolTip(f'{s.name}: each shift along its mount frame\'s axes lies within ± this')
                rot = ScrubSpinBox('', 'plain', 2, '°', step=0.05, width=86)
                rot.setRange(0, 90)
                rot.setToolTip(f'{s.name}: each turn about its own axes lies within ± this')
                share = label('—', mono_=True)
                pos.committed.connect(lambda v, u=s.uid: self.set_tolerance(u, 'position_mm', v))
                rot.committed.connect(lambda v, u=s.uid: self.set_tolerance(u, 'rotation_deg', v))
                for k, w in enumerate((name, src, pos, rot, share)):
                    self.grid.addWidget(w, i, k)
                self._rows[s.uid] = (name, src, pos, rot, share)
            self._uids = uids
        try:
            srcs = {x.uid: x for x in TL.sources(p)}
        except Exception:                               # noqa: BLE001 — the editor still shows the tolerances
            traceback.print_exc()
            srcs = {}
        from ...core.project import Tolerance
        default = Tolerance()
        for s in sensors:
            name, src, pos, rot, share = self._rows[s.uid]
            pos.set_value(float(s.mount.tolerance.position_mm))
            rot.set_value(float(s.mount.tolerance.rotation_deg))
            x = srcs.get(s.uid)
            if x is not None and x.kind == TL.CALIB:
                d = x.to_dict()
                src.setText(f'calibration "{x.session}" (σ up to {max(d["sigma_mm"]):.2f} mm, '
                            f'{max(d["sigma_deg"]):.3f}°)')
                src.setToolTip('The hand-eye covariance replaces the mount tolerance while the sensor stays where the '
                               'calibration put it.')
            else:
                tol = s.mount.tolerance
                is_def = (tol.position_mm, tol.rotation_deg) == (default.position_mm, default.rotation_deg)
                src.setText('mount tolerance' + (' (default)' if is_def else ''))
                src.setToolTip((x.note + '. ') if x is not None and x.note else
                               f'The project\'s default is ± {default.position_mm:g} mm, '
                               f'± {default.rotation_deg:g}° (a labelled default).')

    def set_tolerance(self, uid: str, field: str, value: float) -> bool:
        """Set one sensor's mount tolerance field (position_mm or rotation_deg): one undo step."""
        p = self.ctx.store.project
        s = p.variant.sensor(uid)
        if s is None or not math.isfinite(float(value)) or value < 0:
            return False
        if float(getattr(s.mount.tolerance, field)) == float(value):
            return False

        def fn(pr):
            setattr(pr.variant.sensor(uid).mount.tolerance, field, float(value))
        unit = ' mm' if field == 'position_mm' else '°'
        self.ctx.store.edit('variant', fn, f'Tolerances · {s.name} ± {float(value):g}{unit}')
        return True

    def _changed(self, sections):
        if {'variant', 'variants', 'calibration'} & set(sections):
            self._fill_editor()
            self._cache = None
        if 'analysis' in sections:
            self._fill_settings()

    def _replaced(self):
        self.doc = None
        self._preview = False
        self._cache = None
        self._uids = None
        self.table.clear()
        self.chart.set_rows([])
        self.contrib.set_rows([])
        self.suggestion_ready(False)
        self._fill_settings()
        self._fill_editor()
        self.empty()

    # ---------------------------------------------------------------- the result
    def show_result(self, doc):
        self._cache = None
        self._preview = False
        try:
            h = doc.get('headline') or {}
            word = h.get('word', '')
            fig = fnum(h.get('value')) if h.get('value') is not None else '—'
            q = h.get('quantile', '')
            tip = (f'{q} of the figure over {doc.get("done", 0)} draws (seed {doc.get("seed")}). Status from the share '
                   'of draws meeting the requirement: holds from 95 %, marginal from 80 %, often fails from 50 % '
                   '(DexVantage defaults).')
            self.head.set(fig, h.get('unit', ''), h.get('status', 'warning'), f'{q} · {word}' if q else word,
                          h.get('text', '') + (f' {doc["advice"]}' if doc.get('advice') else ''), tip)
            rows, self.table_rows = [], []
            self.table.clear()
            for mid, m in (doc.get('metrics') or {}).items():
                thr = []
                meets = []
                for rq in m.get('requirements') or []:
                    sev, _w = TS.status_of(rq.get('share'))
                    thr.append((rq['threshold'], sev))
                    if rq.get('share') is not None:
                        lo, hi = rq['share_ci95']
                        meets.append(f'{100 * rq["share"]:.0f} % [{100 * lo:.0f}–{100 * hi:.0f} %]')
                rows.append((m.get('title', mid), m, thr))
                top = next(iter(m.get('contributions') or []), None)
                u = m.get('unit', '')
                it = QTreeWidgetItem([m.get('title', mid), fnum(m.get('nominal'), u),
                                      '—' if m.get('mean') is None else f'{fnum(m["mean"])} ± {fnum(m["std"])} {u}',
                                      '—' if m.get('p5') is None else f'[{fnum(m["p5"])}–{fnum(m["p95"])}] {u}',
                                      '; '.join(meets) or 'no requirement',
                                      f'{top["name"]} {100 * top["share"]:.0f} %' + ('' if top['significant'] else
                                                                                    ' (noise)') if top else '—'])
                for col in (1, 2, 3):
                    it.setFont(col, mono(11))
                if thr:
                    it.setForeground(4, QColor(STATUS[min((TS.status_of(r.get('share')) for r in
                                                           m['requirements']), key=lambda x: ['critical', 'serious',
                                                           'warning', 'good'].index(x[0]))[0]]))
                it.setToolTip(4, 'The share of draws meeting the requirement, with its Wilson 95 % interval.')
                if m.get('no_value'):
                    it.setToolTip(0, f'{m["no_value"]} draws gave no value' + (f': {m["reason"]}' if m.get('reason')
                                                                              else '.'))
                self.table.addTopLevelItem(it)
            self.chart.set_rows(rows)
            self._fill_contrib(doc)
            self.notes.setText(' '.join(doc.get('notes') or []))
            self._fill_editor()
        except (KeyError, TypeError, ValueError):
            traceback.print_exc()
        self.suggestion_ready(bool(doc.get('suggestion')))
        self._redraw()

    def _fill_contrib(self, doc):
        mid = (doc.get('headline') or {}).get('metric')
        m = (doc.get('metrics') or {}).get(mid) or {}
        p = self.ctx.store.project
        rows = []
        for c in m.get('contributions') or []:
            s = p.variant.sensor(c['sensor_uid'])
            rows.append((c['sensor_uid'], s.slot if s is not None else 0, c['name'][:14], 100 * c['share']))
        self.contrib.set_rows(rows)
        for uid, (_n, _s, _p, _r, share) in self._rows.items():
            c = next((c for c in m.get('contributions') or [] if c['sensor_uid'] == uid), None)
            share.setText('—' if c is None else f'{100 * c["share"]:.0f} %' + ('' if c['significant'] else ' ~'))
            share.setToolTip('' if c is None else
                             f'max |ρ| {c["max_rho"]:.2f} against the noise level {doc.get("noise_rho", 0):.2f}'
                             + ('' if c['significant'] else ': not distinguishable from noise'))
        self.contrib_note.setText(
            (f'{m.get("title", "")}: the share of its rank variance each mount carries (Spearman ρ² of the figure with '
             f'each shift and turn and its size, summed). Below |ρ| = {doc.get("noise_rho", 0):.2f} '
             '(2 / √(n − 1)) a contribution is noise (~).') if m else '')

    # ---------------------------------------------------------------- the suggestion
    def apply(self):
        sg = (self.doc or {}).get('suggestion')
        if not sg:
            return None
        p = self.ctx.store.project
        s = p.variant.sensor(sg['sensor_uid'])
        if s is None:
            self._msg('That sensor is not in this rig any more.')
            return None

        def fn(pr):
            setattr(pr.variant.sensor(sg['sensor_uid']).mount.tolerance, sg['field'], float(sg['to']))
        unit = ' mm' if sg['field'] == 'position_mm' else '°'
        self.ctx.store.edit('variant', fn, f'Tolerances · tighten {s.name} to ± {sg["to"]:g}{unit}')
        self._preview = False
        self.suggestion_ready(False)
        self._msg(f'{s.name}: ± {sg["to"]:g}{unit}. Run the study again to see the effect.', 8000)
        return sg

    def preview(self):
        sg = (self.doc or {}).get('suggestion')
        self._preview = bool(sg) and not self._preview
        self._cache = None
        if sg:
            unit = ' mm' if sg['field'] == 'position_mm' else '°'
            self._msg((f'{sg["name"]}: its perturbed frusta as they are (its colour) and at ± {sg["to"]:g}{unit} '
                       f'(green), to scale. ' if self._preview else 'Preview off. ') + sg['text'], 10000)
        self._redraw()

    def explain(self):
        box = QMessageBox(self)
        box.setWindowTitle('How the tolerances are carried')
        box.setText('Each draw moves every sensor\'s mount: within its tolerance, a ± bound per axis (each shift '
                    'along the mount frame\'s axes and each turn about the sensor\'s own axes, uniform within the '
                    'bound by default, or normal with the bound at 2 σ), or from its hand-eye calibration\'s '
                    'covariance when the sensor is calibrated and has not moved since. Every requirement\'s figure is '
                    'then recomputed by its own study. The page shows each figure\'s mean ± σ and P5–P95 over the '
                    'draws and the share of draws meeting the requirement (with its Wilson 95 % interval). A mount\'s '
                    'contribution is the rank correlation (Spearman ρ²) of the figure with its shifts and turns and '
                    'their sizes; below 2 / √(n − 1) it is noise. Apply halves the tolerance of the mount that '
                    'contributes most.')
        box.setModal(False)
        box.show()

    def _msg(self, text, ms=5000):
        if self.ctx.window is not None:
            self.ctx.window.msg(text, ms)

    # ---------------------------------------------------------------- the perturbation export
    def export_perturbations(self, folder=None, block: bool = False):
        """Write n perturbed extrinsics (core/exporters/perturbations.py) to folder on a worker thread (block: in this
        thread, for tests and scripts); the manifest when blocking, else True once started."""
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export perturbed extrinsics to a folder')
            if not folder:
                return None
        st = self._settings()
        opts = {'n': int(st.get('n_export', 100)), 'seed': int(st.get('seed', 0)),
                'distribution': st.get('distribution', 'uniform'),
                'q': {k: np.asarray(v, float).copy() for k, v in self.ctx.store.q.items()}}
        project = self.ctx.store.project.snapshot()
        if block:
            return self._export_done(_write_perturbations(project, folder, opts))
        self.b_perturb.setEnabled(False)
        threading.Thread(target=lambda: self._sig.done.emit(_write_perturbations(project, folder, opts)),
                         daemon=True).start()
        return True

    def _export_done(self, res):
        self.b_perturb.setEnabled(True)
        man, why, opts = res
        if man is None:
            self._msg(why, 8000)
            return None
        self._msg(f'Exported {opts["n"]} perturbed extrinsics (seed {opts["seed"]}) to perturbations/ (JSON and .npz '
                  'with provenance).', 6000)
        return man

    # ---------------------------------------------------------------- overlays
    def _reach(self, project, centre) -> float:
        vols = project.workspace.volumes
        if not vols:
            return DEFAULT_DEPTH_M
        d = min(float(np.linalg.norm(np.asarray(v.center, float) - centre)) for v in vols)
        return max(0.1, min(d, 2.0))

    def _clouds(self):
        """[(sensor, slot, [(world segments (S, 2, 3), tightened)])] for the 3D view, cached per result, edit and
        joint state."""
        doc = self.doc or {}
        draws = doc.get('draws') or {}
        sg = doc.get('suggestion') if self._preview else None
        st = self.ctx.store
        key = (id(self.doc), st.revision, self._preview,
               tuple((k, np.asarray(v).tobytes()) for k, v in sorted(st.q.items())))
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        from ...core import camrender as CR
        from ...core import scenegeom as SG
        from ...core.calib import records as RC
        p = st.project
        out = []
        for uid, D in draws.items():
            if sg and uid != sg['sensor_uid']:
                continue
            s = p.variant.sensor(uid)
            if s is None or not s.enabled:
                continue
            try:
                im = RC.sensor_imager(s)
                W = SG.sensor_world_pose(p, s, st.q).matrix()
                Tw = W @ np.linalg.inv(s.mount.pose.matrix())
                depth = self._reach(p, W[:3, 3])
                _apex, border, _sp = CR.frustum_body(im.camera, im.pose_in_body, depth)
            except Exception:                           # noqa: BLE001 — a sensor that cannot be drawn is left out
                continue
            D = np.asarray(D, float)[: (PREVIEW_DRAWS if sg else OVERLAY_DRAWS)]
            sets = [(D, False)]
            if sg:
                Dt = D.copy()
                cols = slice(0, 3) if sg['field'] == 'position_mm' else slice(3, 6)
                Dt[:, cols] *= sg['to'] / max(sg['from'], 1e-12)       # exact for a ± bound scaled
                sets.append((Dt, True))
            ok = np.isfinite(border).all(axis=1)
            pairs = [(i, i + 1) for i in range(len(border) - 1) if ok[i] and ok[i + 1]]
            if not pairs:
                continue
            ia, ib = np.array(pairs).T
            res = []
            for DD, tight in sets:
                segs = []
                for d in DD:
                    T = Tw @ s.mount.pose.matrix() @ TL.delta_body(s.mount.pose, d)
                    B = border @ T[:3, :3].T + T[:3, 3]
                    segs.append(np.stack([B[ia], B[ib]], axis=1))
                res.append((np.concatenate(segs), tight))
            out.append((s, s.slot, res))
        self._cache = (key, out)
        return out

    def overlays(self, vp) -> list:
        from ..viewport3d import LineDraw
        items = []
        try:
            for _s, slot, res in self._clouds():
                for segs, tight in res:
                    col = QColor(STATUS['good']) if tight else QColor(THEME.cam(slot))
                    col.setAlphaF(0.85 if tight else (0.6 if self._preview else 0.35))
                    items.append(LineDraw(segs, col, width=1.0, depth=True, segments=True))
        except Exception:                               # noqa: BLE001 — an overlay never breaks the view
            traceback.print_exc()
        return items

    def camview_overlays(self, cam_model, pose) -> list:
        """The task volumes' centres as each draw of this camera would see them (crosses), against the nominal."""
        win = self.ctx.window
        cv = getattr(win, 'camview', None) if win is not None else None
        uid = getattr(cv, 'sensor_uid', None) or getattr(getattr(cv, 'view', None), 'sensor_uid', None)
        D = ((self.doc or {}).get('draws') or {}).get(uid)
        p = self.ctx.store.project
        s = p.variant.sensor(uid) if uid else None
        if D is None or s is None or not p.workspace.volumes:
            return []
        try:
            from ...core import camrender as CR
            from ...core.frames import Pose
            from ...core.calib import records as RC
            T_im = RC.sensor_imager(s).pose_in_body.matrix()
            O = pose.matrix()                                    # the imager's optical frame in the world
            Wb = O @ np.linalg.inv(T_im)                          # the body
            pts = np.array([np.asarray(v.center, float) for v in p.workspace.volumes])
            polys = []
            r = max(3.0, cam_model.width / 160)
            for d in np.asarray(D, float)[:OVERLAY_DRAWS]:
                Tp = Wb @ TL.delta_body(s.mount.pose, d) @ T_im
                uv, ok, _ = CR.project_world(cam_model, Pose.from_matrix(Tp), pts)
                for (u, v), good in zip(uv, ok):
                    if good:
                        polys.append(np.array([[u - r, v], [u + r, v]]))
                        polys.append(np.array([[u, v - r], [u, v + r]]))
            return [(polys, QColor(THEME.cam(s.slot)), f'task volume centres under {min(len(D), OVERLAY_DRAWS)} '
                                                      'draws of the mount')] if polys else []
        except Exception:                               # noqa: BLE001
            traceback.print_exc()
            return []

    def _redraw(self):
        win = self.ctx.window
        if win is not None and hasattr(win, 'viewport'):
            win.viewport.update()
        cv = getattr(win, 'camview', None) if win is not None else None
        if cv is not None and hasattr(cv, 'refresh_overlays'):
            cv.refresh_overlays()
        elif cv is not None and hasattr(cv, 'view'):
            cv.view.refresh_overlays()


def _write_perturbations(project, folder, opts):
    """(manifest or None, the reason in a plain sentence, options): the export, on any thread."""
    from ...core import exporters as X
    from ...core.jobs import StudyError
    try:
        man = X.export(project, folder, ['perturbations'], options={'perturbations': opts})
    except StudyError as e:
        return None, str(e), opts
    except Exception:                                   # noqa: BLE001 — explained, the traceback logged
        traceback.print_exc()
        return None, 'The perturbation export stopped with an unexpected error. The details are in the log.', opts
    row_ = man['formats']['perturbations']
    if row_['status'] != 'ok':
        return None, row_.get('reason', 'The perturbations could not be written.'), opts
    return man, '', opts


class _Sig(QObject):
    done = pyqtSignal(object)


register('tolerance', 'Tolerances', TolerancePage, menu='Analysis', icon='tolerance', action_id=None,
         workspace=None, order=90)
