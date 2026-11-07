"""The Action-chunk latency page (spec §7.13): how late or early each action-chunked policy closes the gripper on the
contact point, given the rig's sensor-to-action loop and the arm's approach (core/studies/action_chunk_sync.py).

What it shows: the worst policy's error at contact in millimetres (the headline, its status against the 5 mm
tolerance, a default), and for the chosen policy: the loop per camera as a bar of its terms (exposure, readout,
transfer, queueing, inference, the control tick, command latency; a term not stated is listed as missing, never
filled in), the terms with where each comes from, the close-timing interval and what it means in space (overshoot,
air-grasp, the reaction distance), and for an ensembling policy the reaction distance against the ensemble weight m
(ACT's convention, w_i = exp(−m·i), i = 0 the oldest prediction) with the current and the recommended m marked.

Apply stores the recommended proprioceptive delay offset and ensemble weight on every policy of the result
(Policy.recommendations, one undo step; the page's Export carries them; the LeRobot training-config exporter is
part of §9.2, Phase 10, and is not built yet). Preview draws the
overshoot map as it would be with the recommended offset. The 3D view draws the overshoot map over the task volume
(one box per cell, its status colour from its larger error); the Camera view marks the cells that exceed the
tolerance. Export writes the result with provenance.
"""
from __future__ import annotations

import math
import traceback

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QComboBox, QHeaderView, QTreeWidget, QTreeWidgetItem

from ...core.studies import action_chunk_sync as A
from ..charts import ChartBase
from ..theme import STATUS, THEME, mono, sans
from ..viewport3d import TriDraw
from ..widgets import button, label
from . import register
from ._curves import CurveChart
from .base import StudyPage, row

TERM_COLORS = {'exposure_half': '#2a78d6', 'readout': '#199e70', 'transfer': '#c98500', 'queue': '#9085e9',
               'inference': '#d95926', 'tick': '#8a8980', 'command': '#d55181'}
ORIGIN_WORDS = {'library': 'library entry', 'user': 'your figure', 'default': 'default', 'study': 'bandwidth study',
                'policy': 'policy entry', 'not_stated': 'not stated'}


def _ms(x, nd=1) -> str:
    return '—' if x is None else f'{x * 1000:.{nd}f} ms'


def _mm(x, nd=1) -> str:
    return '—' if x is None else f'{x * 1000:.{nd}f} mm'


def cell_status(c: dict, tol_m: float) -> str:
    e = max(c.get('overshoot_m') or 0.0, c.get('air_grasp_m') or 0.0)
    return 'good' if e <= tol_m + 1e-12 else ('warning' if e <= 2 * tol_m else
                                              ('serious' if e <= 4 * tol_m else 'critical'))


def box_triangles(center, half) -> np.ndarray:
    """(12, 3, 3) triangles of an axis-aligned box."""
    c = np.asarray(center, float)
    s = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
                 float) * half + c
    f = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6),
         (3, 0, 4), (3, 4, 7)]
    return s[np.array(f)]


class LoopBars(ChartBase):
    """One stacked bar per camera: the loop's terms at their nominal values (ms), a missing term named after the
    bar; hover gives the term, its value and where it comes from."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows: list = []           # [(camera name, [term dicts], [missing titles])]
        self.vmax = 1.0
        self.rects: list = []
        self.setMinimumHeight(70)

    def set_rows(self, rows):
        self.rows = rows
        tot = [sum(t['value_s'] or 0.0 for t in ts) for _n, ts, _m in rows]
        self.vmax = max(tot + [1e-3]) * 1.1
        self.setMinimumHeight(30 + 30 * max(len(rows), 1) + 24)
        self.tip = None
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.paint_title(p)
        left, top, w = 150.0, 26.0, max(self.width() - 170.0, 40.0)
        self.rects = []
        for i, (name, terms, missing) in enumerate(self.rows):
            y = top + 30 * i
            p.setFont(sans(11))
            p.setPen(THEME.color('ink2'))
            p.drawText(QRectF(0, y, left - 8, 22), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, name)
            x = left
            for t in terms:
                v = t['value_s']
                if not v:
                    continue
                r = QRectF(x, y + 3, max(w * v / self.vmax - 1.5, 1.0), 16)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(TERM_COLORS.get(t['key'], '#888888')))
                p.drawRect(r)
                self.rects.append((r, t))
                x += w * v / self.vmax
            tot = sum(t['value_s'] or 0.0 for t in terms)
            p.setFont(mono(10))
            p.setPen(THEME.color('ink'))
            txt = (f'{tot * 1000:.0f} ms' if not missing else f'at least {tot * 1000:.0f} ms')
            p.drawText(QRectF(x + 6, y, 160, 22), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, txt)
        y = top + 30 * len(self.rows) + 2
        x = left
        p.setFont(sans(9))
        for k, col in TERM_COLORS.items():
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(col))
            p.drawRect(QRectF(x, y + 4, 8, 8))
            p.setPen(THEME.color('ink2'))
            t = A.TERM_TITLES[k]
            p.drawText(QRectF(x + 11, y, 140, 16), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, t)
            x += 15 + 6.2 * len(t)
            if x > self.width() - 60:
                break
        self.paint_tip(p)
        p.end()

    def mouseMoveEvent(self, e):
        pos = e.position()
        hit = next(((r, t) for r, t in self.rects if r.contains(pos)), None)
        if hit:
            r, t = hit
            rng = '' if t['lo_s'] == t['hi_s'] else f' [{t["lo_s"] * 1000:.1f}–{t["hi_s"] * 1000:.1f}]'
            self.set_tip(QPointF(r.center().x(), r.top()), f'{t["value_s"] * 1000:.2f} ms{rng}', t['title'],
                         ORIGIN_WORDS.get(t['origin'], t['origin']), t['source'][:90])
        else:
            self.tip = None
        self.update()


def _tree(headers) -> QTreeWidget:
    t = QTreeWidget()
    t.setRootIsDecorated(False)
    t.setHeaderLabels(headers)
    t.setUniformRowHeights(True)
    t.setWordWrap(False)
    for k in range(len(headers) - 1):
        t.header().setSectionResizeMode(k, QHeaderView.ResizeMode.ResizeToContents)
    t.setMinimumHeight(80)
    return t


class ActionChunkPage(StudyPage):
    """Action-chunk latency: an action-chunked policy closes the gripper late (the hand pushes past the contact point,
    overshoot) or early (it closes in the air, air-grasp) when its sensor-to-action loop differs from what its
    training data's timing built in. The headline is the worst policy's error at contact, at the approach the
    trajectories show (or the default approach speed)."""
    STUDY = A.STUDY_ID
    TITLE = 'Action-chunk latency'
    HEADLINE = 'Error at the contact point'

    def build_body(self, layout):
        self._preview = False
        self._ov_key = None
        self._ov_items: list = []
        self.sel = ''
        self.pol_box = QComboBox()
        self.pol_box.setToolTip('The policy whose loop, timing and recommendations are shown (and drawn)')
        self.pol_box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.pol_box.setMinimumContentsLength(18)
        self.pol_box.currentIndexChanged.connect(self._policy_changed)
        layout.addWidget(row(label('Policy', ink2=True), self.pol_box))
        self.scheme = label('', wrap=True)
        layout.addWidget(self.scheme)
        self.bars = LoopBars()
        self.bars.title = 'Sensor-to-action loop per camera'
        self.bars.subtitle = 'ms, nominal'
        layout.addWidget(self.bars)
        self.terms = _tree(['Camera', 'Term', 'Value', 'From', 'Source'])
        layout.addWidget(self.terms)
        layout.addWidget(label('At the contact point', bold=True))
        self.pred = label('', wrap=True, mono_=False)
        layout.addWidget(self.pred)
        self.curve = CurveChart(height=200)
        self.curve.title = 'Reaction distance against the ensemble weight m'
        self.curve.subtitle = 'i = 0 the oldest prediction (ACT)'
        self.curve.xname, self.curve.xunit, self.curve.yunit = 'm', '', 'mm'
        layout.addWidget(self.curve)
        self.curve_note = label('', muted=True, wrap=True, small=True)
        layout.addWidget(self.curve_note)
        layout.addWidget(label('What to write into the training config', bold=True))
        self.rec = label('', wrap=True)
        layout.addWidget(self.rec)
        self.b_show_map = button('Show the overshoot map', role='ghost', icon_name='coverage',
                                 tip='Draw the overshoot map in the 3D view (on while this page is shown)')
        self.b_show_map.setCheckable(True)
        self.b_show_map.setChecked(True)
        self.b_show_map.toggled.connect(lambda _on: self._redraw())
        layout.addWidget(row(self.b_show_map))
        self.assume = label('', muted=True, wrap=True, small=True)
        layout.addWidget(self.assume)
        st = self.ctx.store
        st.project_replaced.connect(self._replaced)

    # ---------------------------------------------------------------- result
    def _replaced(self):
        self.doc = None
        self.empty()

    def empty(self):
        self.head.set('—', '', 'warning', 'not run yet', 'Run the action-chunk study to see how late or early each '
                                                         'policy closes on the contact point.')
        self.suggestion_ready(False)
        if hasattr(self, 'bars'):
            self.bars.set_rows([])
            self.terms.clear()

    def _row(self):
        rows = (self.doc or {}).get('policies') or []
        return next((r for r in rows if r['uid'] == self.sel), rows[0] if rows else None)

    def show_result(self, doc: dict):
        self._preview = False
        h = doc.get('headline') or {}
        st = doc.get('settings') or {}
        tip = (f'Tolerance {st.get("tolerance_mm", 5):g} mm (a default): good up to it, warning up to 2×, serious up '
               f'to 4×, critical beyond. Approach: {(doc.get("approach") or {}).get("source", "")}')
        self.head.set(f'{h.get("worst_mm", 0):.1f}', 'mm', h.get('status', 'warning'), h.get('status_word', ''),
                      f'{h.get("policy", "")}: overshoot {h.get("overshoot_mm", 0):.1f} mm, air-grasp '
                      f'{h.get("air_grasp_mm", 0):.1f} mm. {doc.get("advice", "")}', tip)
        self.suggestion_ready(bool(doc.get('policies')))
        if not self._differs_from_stored(doc):        # already stored: Apply would add nothing (Preview still shows)
            self.b_apply.setEnabled(False)
        rows = doc.get('policies') or []
        self.pol_box.blockSignals(True)
        self.pol_box.clear()
        for r in rows:
            self.pol_box.addItem(r['name'], r['uid'])
        if self.sel not in [r['uid'] for r in rows]:
            self.sel = h.get('policy_uid') or (rows[0]['uid'] if rows else '')
        self.pol_box.setCurrentIndex(max(self.pol_box.findData(self.sel), 0))
        self.pol_box.blockSignals(False)
        notes = [f'{r["name"]} is left out: {r["reason"]}' for r in doc.get('refused') or []]
        notes += list(doc.get('notes') or [])
        notes.append('The policy is modelled as an ideal imitator of its demonstrations: timing errors only (see '
                     'Explain and docs/models/action_chunk.md).')
        self.assume.setText('\n'.join(notes))
        self._fill()

    def _differs_from_stored(self, doc: dict) -> bool:
        """True when Apply would change a policy: some recommended value is not the one already stored on it."""
        pols = {p.uid: p for p in self.ctx.store.project.policies}
        for r in doc.get('policies') or []:
            pol = pols.get(r['uid'])
            if pol is None:
                continue
            stored = pol.recommendations or {}
            for k, v in A.recommendation_records(r, '').items():
                if (stored.get(k) or {}).get('value') != v['value']:
                    return True
        return False

    def _policy_changed(self, i):
        self.sel = self.pol_box.itemData(i) or ''
        self._fill()
        self._redraw()

    def _fill(self):  # noqa: C901 — one policy's panels
        r = self._row()
        if r is None:
            return
        sc = r['scheme']
        if sc['ensemble']:
            how = (f'temporal ensembling, m = {sc["m"]:g} ({sc["m_source"][:80]}); the executed action averages '
                   f'{sc["n_eff"]:.1f} predictions in effect, {sc["mean_age_steps"]:.1f} steps old on average')
        else:
            how = (f'chunks executed open loop, a new one every {sc["s"]} step{"s" if sc["s"] != 1 else ""} '
                   f'(executed per chunk {sc["E"]})')
        hw = f' on {sc["hardware"]}' if sc.get('hardware') else ''
        self.scheme.setText(f'{r["entry"]}: horizon H = {sc["H"]}, control {sc["hz"]:g} Hz, inference '
                            f'{sc["tau_i"] * 1000:.0f} ms{hw}; {how}.')
        self.bars.set_rows([(c['name'].replace(' (sample placement)', ''), c['terms'], c['missing'])
                            for c in r['cameras']])
        self.terms.clear()
        for c in r['cameras']:
            for t in c['terms']:
                val = '—' if t['lo_s'] is None else (_ms(t['value_s'], 2) if t['lo_s'] == t['hi_s'] else
                                                       f'{_ms(t["value_s"], 2)} [{t["lo_s"] * 1000:.1f}–'
                                                       f'{t["hi_s"] * 1000:.1f}]')
                it = QTreeWidgetItem([c['name'].replace(' (sample placement)', ''), t['title'], val,
                                      ORIGIN_WORDS.get(t['origin'], t['origin']), t['source']])
                it.setFont(2, mono(11))
                it.setTextAlignment(2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                it.setToolTip(4, t['source'])
                if t['lo_s'] is None:
                    it.setToolTip(2, 'not stated: nothing is filled in; enter a measured figure')
                self.terms.addTopLevelItem(it)
        self.terms.setFixedHeight(min(26 + 20 * self.terms.topLevelItemCount() + 8, 360))
        e = r['error_s']
        stamp = r['cameras'][0]['stamp'] if r['cameras'] else ''
        lines = [f'The close reaches the gripper {e[1] * 1000:+.0f} ms from contact [{e[0] * 1000:+.0f} to '
                 f'{e[2] * 1000:+.0f} ms] (training offset {r["delay_offset_s"] * 1000:.0f} ms, '
                 f'{r["delay_offset_source"]}; images stamped {stamp.replace("_", " ")}).',
                 f'Overshoot {_mm(r["overshoot_m"])} · air-grasp {_mm(r["air_grasp_m"])} · reaction distance '
                 f'{_mm(r["reaction_m"][0])} [up to {_mm(r["reaction_m"][1])}]: how far the hand travels before an '
                 'action that has seen a change at the contact lands.']
        if r['missing']:
            lines.append('Not stated, so left out (the loop is "at least"): ' + ', '.join(r['missing']) + '.')
        self.pred.setText('\n'.join(lines))
        rd = r['recommendations']['delay_offset_s']
        mr = r['recommendations']['temporal_ensemble_m']
        txt = [f'Proprioceptive delay offset: {rd["value"] * 1000:.0f} ms ({rd["steps"]} control steps; now '
               f'{rd["from"] * 1000:.0f} ms). Worst error at contact {_mm(rd["from_worst_m"])} → '
               f'{_mm(rd["worst_m"])}; timing [{rd["error_s"][0] * 1000:+.0f} to {rd["error_s"][2] * 1000:+.0f} ms].']
        if mr:
            txt.append(f'Ensemble weight m: {mr["m"]:g} (now {mr["from_m"]:g}). Reaction distance '
                       f'{_mm(mr["from_reaction_m"])} → {_mm(mr["reaction_m"])}; predictions averaged '
                       f'{mr["from_n_eff"]:.1f} → {mr["n_eff"]:.1f}' + ('.' if mr['met'] else
                                                                     ': the reaction tolerance cannot be met by '
                                                                     'ensembling alone.'))
            cv = mr['curve']
            ys = [None if y is None else y * 1000 for y in cv['reaction_m']]
            top = max([y for y in ys if y is not None] + [1.0]) * 1.1
            tol = ((self.doc or {}).get('settings') or {}).get('reaction_tolerance_mm', 10.0)
            self.curve.set_data([('reaction distance', cv['m'], ys, THEME.color('ink').name(), 'line')],
                                (min(cv['m']), max(cv['m'])), (0.0, max(top, tol * 1.2)),
                                hlines=[(tol, f'tolerance {tol:g} mm (default)')],
                                markers=[(mr['from_m'], mr['from_reaction_m'] * 1000, STATUS['warning'], 'now'),
                                         (mr['m'], mr['reaction_m'] * 1000, STATUS['good'], 'recommended')],
                                xlabel='ensemble weight m', ylabel='reaction distance, mm')
            self.curve.show()
            self.curve_note.setText('A positive m weights older predictions more (ACT and LeRobot), a negative m the '
                                    'newer ones. The recommended m averages the most predictions (smoothing) while '
                                    'the reaction distance stays within the tolerance.')
        else:
            self.curve.hide()
            self.curve_note.setText('This policy executes its chunks open loop (its source uses no temporal '
                                    'ensembling, or does not say): there is no ensemble weight to recommend; fewer '
                                    'actions executed per chunk shorten its reaction.')
        self.rec.setText('\n'.join(txt))
        self._redraw()

    # ---------------------------------------------------------------- Apply / Preview
    def apply(self):
        doc = self.doc
        if not doc or not doc.get('policies'):
            return
        names = [r['name'] for r in doc['policies']]
        uids = {p.uid for p in self.ctx.store.project.policies}
        if not any(r['uid'] in uids for r in doc['policies']):
            if self.ctx.window is not None:
                self.ctx.window.msg('Those policies are no longer in the project; run the study again.', 6000)
            return
        self.ctx.store.edit('policies', lambda pr: A.apply_recommendations(pr, doc),
                            f'Stored the action-chunk recommendations on {", ".join(names)}')
        self._preview = False
        self.suggestion_ready(False)
        if self.ctx.window is not None:
            self.ctx.window.msg('Stored the delay offset and ensemble weight on the policies. Enter them in your '
                                'training config (Export writes them with the result).', 6000)
        self.run()

    def preview(self):
        self._preview = not self._preview
        if self.ctx.window is not None:
            self.ctx.window.msg('Preview: the overshoot map with the recommended delay offset' if self._preview
                                else 'Preview off', 6000)
        self._redraw()

    # ---------------------------------------------------------------- overlays
    def _redraw(self):
        self._ov_key = None
        win = self.ctx.window
        if win is not None and hasattr(win, 'viewport'):
            win.viewport.update()

    def _cells(self):
        r = self._row()
        if r is None:
            return [], 0.0, 0.005
        m = r['map_after' if self._preview else 'map']
        tol = ((self.doc or {}).get('settings') or {}).get('tolerance_mm', 5.0) / 1000.0
        return m.get('cells') or [], tol, m.get('cell_m') or 0.03

    def overlays(self, vp) -> list:
        if not self.doc or not self.b_show_map.isChecked():
            return []
        key = (id(self.doc), self.sel, self._preview)
        if key == self._ov_key:
            return self._ov_items
        items = []
        try:
            cells, tol, cell = self._cells()
            if cells:
                tris, cols = [], []
                for c in cells:
                    t = box_triangles(c['center'], 0.45 * cell)
                    col = QColor(STATUS[cell_status(c, tol)])
                    tris.append(t)
                    cols += [(col.redF(), col.greenF(), col.blueF(), 0.28)] * len(t)
                items.append(TriDraw(np.concatenate(tris).astype(np.float32), np.asarray(cols, np.float32)))
        except Exception:                                    # noqa: BLE001 — an overlay must not take the view down
            traceback.print_exc()
            items = []
        self._ov_key, self._ov_items = key, items
        return items

    def camview_overlays(self, cam_model, pose) -> list:
        """The overshoot map's cells beyond the tolerance as crosses in the Camera view."""
        from ...core import camrender as CR
        if not self.doc:
            return []
        cells, tol, _cell = self._cells()
        bad = [c for c in cells if cell_status(c, tol) != 'good']
        if not bad:
            return []
        try:
            uv, ok, _d = CR.project_world(cam_model, pose, np.array([c['center'] for c in bad], float))
        except Exception:                                    # noqa: BLE001
            return []
        polys = []
        r = 4.0
        for (u, v), k in zip(np.asarray(uv), np.asarray(ok, bool)):
            if k and math.isfinite(u) and math.isfinite(v):
                polys.append(np.array([[u - r, v - r], [u + r, v + r]]))
                polys.append(np.array([[u - r, v + r], [u + r, v - r]]))
        return [(polys, QColor(STATUS['serious']), 'contact error above tolerance')] if polys else []


register('action_chunk', 'Action-chunk latency', ActionChunkPage, menu='Analysis', icon='target', action_id=None,
         workspace='grasp', order=35)
