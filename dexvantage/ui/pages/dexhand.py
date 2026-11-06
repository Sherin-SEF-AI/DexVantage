"""The Multi-finger occlusion page (spec §7.14): which fingers hide which fingertip's contact patch from each camera of
the rig, over the hand's grasp postures or a grasp set, and which candidate mount sees the fingertips best.

What it shows: the share of fingertip-cases the rig sees (a fingertip counts as seen in a case from 50 % of its
patch, a default; thresholds 90, 60 and 30 % for the status, defaults), the finger × camera matrix (the share of each
patch not seen, with what hides most: a finger, the palm, the wrist and arm, or "outside the image") as a heatmap,
the split by owner for one camera ("who hides whom"), the mount comparison table, and the sweep (each case's
not-seen share per finger for the chosen camera). Choose a case and Show it: the 3D view takes that posture and
draws the patch points, seen by the chosen camera in its colour and not seen in grey; the Camera view of a camera in
the study draws them too.

Run passes the arm's joint state on screen (or home) and the cases' source to the study; nothing is stored. Apply
turns the suggested camera about its optical centre (one undo step); Preview draws its new optical axis dashed.
"""
from __future__ import annotations

import copy
import math
import traceback

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QComboBox, QHeaderView, QTreeWidget, QTreeWidgetItem

from ...core.studies import dex_hand_tactile as D
from ..charts import ChartBase
from ..theme import THEME, mono, sans
from ..viewport3d import LineDraw
from ..widgets import SegmentedControl, button, label
from . import register
from .base import StudyPage, row

# owner categories, always shown with their word (not status colours)
FINGER_COLORS = ['#2a78d6', '#d95926', '#199e70', '#c98500', '#d55181', '#9085e9', '#008300', '#e66767']
GROUP_COLORS = {'palm': '#8a63d2', 'arm': '#6f6e69', 'objects': '#2fa3a8', 'scene': '#a3a29b',
                D.OUT: '#c3c2b7'}
CROSS_M = 0.0012                 # half-size of a patch point's cross in the 3D view (m)
AXIS_M = 0.25                    # length of a camera axis drawn by Preview (m)
Q_ATOL = 1e-5                    # rad: the posture on screen is "this case" within this
_RAMPS: dict = {}


def _pct(x, nd=1) -> str:
    return '—' if x is None else f'{x:.{nd}f} %'


def owner_color(key: str, fingers: list) -> QColor:
    names = [f['name'] for f in fingers]
    if key in names:
        return QColor(FINGER_COLORS[names.index(key) % len(FINGER_COLORS)])
    return QColor(GROUP_COLORS.get(key, '#888888'))


class Heatmap(ChartBase):
    """A matrix of percentages (rows × columns) on a sequential blue scale (more = darker against the surface), each
    cell with its value and an optional note; hover gives the cell's lines; a click on a column emits its index."""
    column_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows: list = []
        self.cols: list = []
        self.values: list = []            # [[float | None]] percent
        self.notes: list = []             # [[str]]
        self.tips: list = []              # [[[str]]]
        self.col_colors: list = []
        self.selected_col = -1
        self.left = 132
        self.setMinimumHeight(60)

    def set_data(self, rows, cols, values, notes=None, tips=None, col_colors=None, selected_col=-1):
        from PyQt6.QtGui import QFontMetricsF
        self.rows, self.cols, self.values = list(rows), list(cols), [list(r) for r in values]
        fm = QFontMetricsF(sans(11))
        self.left = min(max([fm.horizontalAdvance(r) for r in self.rows] + [80.0]) + 16, 320.0)
        self.notes = notes or [[''] * len(cols) for _ in rows]
        self.tips = tips or [[[] for _ in cols] for _ in rows]
        self.col_colors = col_colors or []
        self.selected_col = selected_col
        self.hover = None
        self.tip = None
        self.setMinimumHeight(self._top() + max(1, len(self.rows)) * self._rh() + 6)
        self.updateGeometry()
        self.update()

    def _top(self):
        return 22 + 30

    def _rh(self):
        return 34

    def _cw(self):
        return max((self.width() - self.left - 4) / max(len(self.cols), 1), 30.0)

    def cell_rect(self, i, j) -> QRectF:
        return QRectF(self.left + j * self._cw() + 1, self._top() + i * self._rh() + 1, self._cw() - 2, self._rh() - 2)

    @staticmethod
    def color_of(v):
        from .. import colormaps as CM
        if v is None or not math.isfinite(v):
            return QColor(CM.BELOW_MIN[CM._theme(None)])
        key = CM._theme(None)
        if key not in _RAMPS:
            _RAMPS[key] = CM.ramp(CM.DENSITY[key], 101)             # (101, 3) uint8
        c = _RAMPS[key][int(round(min(max(v, 0.0), 100.0)))]
        return QColor(int(c[0]), int(c[1]), int(c[2]))

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.paint_title(p)
        top, rh, cw = self._top(), self._rh(), self._cw()
        p.setFont(sans(10))
        for j, c in enumerate(self.cols):
            r = QRectF(self.left + j * cw + 2, 22, cw - 4, 28)
            if j == self.selected_col:
                p.fillRect(r, THEME.color('raised'))
            if j < len(self.col_colors) and self.col_colors[j] is not None:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(self.col_colors[j])
                p.drawEllipse(QPointF(r.left() + 5, r.center().y()), 3.5, 3.5)
                r = r.adjusted(11, 0, 0, 0)
            p.setPen(THEME.color('ink' if j == self.selected_col else 'ink2'))
            p.drawText(r, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap, c)
        for i, name in enumerate(self.rows):
            y = top + i * rh
            p.setFont(sans(11))
            p.setPen(THEME.color('ink2'))
            p.drawText(QRectF(4, y, self.left - 10, rh), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       name)
            for j in range(len(self.cols)):
                v = self.values[i][j] if j < len(self.values[i]) else None
                r = self.cell_rect(i, j)
                col = self.color_of(v)
                hot = self.hover == (i, j)
                p.setPen(QPen(THEME.color('ink'), 1) if hot else Qt.PenStyle.NoPen)
                p.setBrush(col)
                p.drawRoundedRect(r, 3, 3)
                lum = 0.2126 * col.redF() + 0.7152 * col.greenF() + 0.0722 * col.blueF()
                p.setPen(QColor('#ffffff') if lum < 0.45 else QColor('#1a1a19'))
                note = self.notes[i][j] if i < len(self.notes) and j < len(self.notes[i]) else ''
                p.setFont(mono(11, QFont.Weight.DemiBold))
                p.drawText(QRectF(r.left(), r.top() + 2, r.width(), (r.height() - 4) / (2 if note else 1)),
                           Qt.AlignmentFlag.AlignCenter, '—' if v is None else f'{v:.0f} %')
                if note:
                    p.setFont(sans(9))
                    p.drawText(QRectF(r.left() + 2, r.center().y(), r.width() - 4, r.height() / 2 - 2),
                               Qt.AlignmentFlag.AlignCenter, note)
        self.paint_tip(p)
        p.end()

    def _cell_at(self, pos):
        top, rh, cw = self._top(), self._rh(), self._cw()
        x, y = pos.x(), pos.y()
        if x < self.left:
            return None
        j = int((x - self.left) // cw)
        i = int((y - top) // rh) if y >= top else -1
        if 0 <= j < len(self.cols):
            return i, j
        return None

    def mouseMoveEvent(self, e):
        c = self._cell_at(e.position())
        if c is not None and 0 <= c[0] < len(self.rows):
            self.hover = c
            i, j = c
            v = self.values[i][j]
            lines = self.tips[i][j] if i < len(self.tips) and j < len(self.tips[i]) else []
            r = self.cell_rect(i, j)
            self.set_tip(QPointF(r.center().x(), r.top()), '—' if v is None else f'{v:.1f} %',
                         f'{self.rows[i]} · {self.cols[j]}', *lines)
        else:
            self.hover = None
            self.tip = None
        self.update()

    def mousePressEvent(self, e):
        c = self._cell_at(e.position())
        if c is not None:
            self.column_clicked.emit(c[1])


def _tree(headers) -> QTreeWidget:
    t = QTreeWidget()
    t.setRootIsDecorated(False)
    t.setHeaderLabels(headers)
    t.setUniformRowHeights(True)
    for k in range(1, len(headers)):
        t.header().setSectionResizeMode(k, QHeaderView.ResizeMode.ResizeToContents)
    t.setMinimumHeight(60)
    return t


def _fit(t: QTreeWidget):
    n = t.topLevelItemCount()
    rh = t.sizeHintForRow(0) if n else 20
    t.setFixedHeight(t.header().sizeHint().height() + max(n, 1) * max(rh, 18) + 2 * t.frameWidth() + 18)


class DexHandPage(StudyPage):
    """Multi-finger occlusion: for each fingertip of a dexterous hand and each camera, the share of the fingertip's
    contact patch the camera does not see over the hand's grasp postures (or a grasp set), and what hides it: the
    owner of the first hit along the line of sight (a finger, the palm, the wrist and arm), or the patch is outside
    the image. A fingertip counts as seen in a case from 50 % of its patch (a default); the headline is the share of
    fingertip-cases the rig's cameras together see."""
    STUDY = D.STUDY_ID
    TITLE = 'Multi-finger occlusion'
    HEADLINE = 'Fingertip-cases whose contact patch the rig sees'

    # ---------------------------------------------------------------- body
    def build_body(self, layout):
        self._preview = False
        self._overlay_key = None
        self._overlay_items: list = []
        self.sel_cam = ''
        self.source = SegmentedControl([('synergy', 'Grasp postures'), ('grasp_set', 'Grasp set')], 'synergy')
        self.source.setToolTip('Sweep the hand\'s power and precision postures, or pose the hand at every grasp of a '
                               'grasp set (IK) with the power posture closed')
        self.c_set = QComboBox()
        self.c_set.setToolTip('The grasp set (Tasks page) for the grasp-set cases')
        self.pose = SegmentedControl([('current', 'Arm as on screen'), ('home', 'Arm at home')], 'current')
        self.pose.setToolTip('The arm\'s joints during the posture sweep: as in the 3D view, or at home')
        layout.addWidget(row(label('Cases', ink2=True), self.source, self.c_set, self.pose))
        self.case_note = label('', muted=True, wrap=True)
        layout.addWidget(self.case_note)
        self.matrix = Heatmap()
        self.matrix.title = 'Finger × camera: share of each contact patch not seen'
        self.matrix.subtitle = '% of the patch, mean over the cases'
        layout.addWidget(self.matrix)
        self.matrix_note = label('', muted=True, wrap=True, small=True)
        layout.addWidget(self.matrix_note)
        self.c_cam = QComboBox()
        self.c_cam.setToolTip('The camera the split by owner, the sweep and the patch points are for')
        layout.addWidget(row(label('Who hides whom, for', bold=True), self.c_cam))
        self.owners = Heatmap()
        self.owners.title = ''
        layout.addWidget(self.owners)
        layout.addWidget(label('Candidate mounts', bold=True))
        self.table = _tree(['#', 'Camera', 'Mount', 'Fingertips seen', 'Every fingertip', 'Patch seen',
                            'Outside image', 'Hidden', 'Worst finger', 'Hides most'])
        layout.addWidget(self.table)
        self.c_case = QComboBox()
        self.c_case.setToolTip('A case of the sweep (a posture at a closure, or a grasp)')
        self.b_show = button('Show this case', role='ghost', icon_name='hand',
                             tip='Pose the hand as in this case in the 3D view and draw its patch points')
        layout.addWidget(row(label('Sweep', bold=True), self.c_case, self.b_show))
        self.sweep = Heatmap()
        self.sweep.title = ''
        layout.addWidget(self.sweep)
        self.assume = label('', muted=True, wrap=True, small=True)
        layout.addWidget(self.assume)
        for c in (self.c_set, self.c_cam, self.c_case):
            c.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
            c.setMinimumContentsLength(18)
        self.c_cam.currentIndexChanged.connect(self._cam_changed)
        self.matrix.column_clicked.connect(self._matrix_col)
        self.b_show.clicked.connect(lambda: self.show_case(self.c_case.currentIndex()))
        self.c_case.currentIndexChanged.connect(lambda _i: self._redraw())
        self.source.changed.connect(lambda _v: self._fill_sets())
        st = self.ctx.store
        st.changed.connect(lambda _s: self._fill_sets())
        st.project_replaced.connect(self._replaced)
        st.pose_changed.connect(lambda _uid: self._redraw())
        self._fill_sets()

    # ---------------------------------------------------------------- controls
    def _replaced(self):
        self.doc = None
        self._fill_sets()
        self.empty()

    def _fill_sets(self):
        from ...core import grasps as GR
        cur = self.c_set.currentData()
        self.c_set.blockSignals(True)
        self.c_set.clear()
        try:
            sets = GR.sets(self.ctx.store.project)
        except Exception:                                # noqa: BLE001 — the controls must not take the page down
            traceback.print_exc()
            sets = []
        for gs in sets:
            self.c_set.addItem(f'{gs.get("name", "Grasp set")} ({len(gs.get("poses") or [])})', gs.get('uid'))
        i = self.c_set.findData(cur)
        if i >= 0:
            self.c_set.setCurrentIndex(i)
        self.c_set.blockSignals(False)
        gs_mode = self.source.value() == 'grasp_set'
        self.c_set.setEnabled(gs_mode and bool(sets))
        if gs_mode and not sets:
            self.case_note.setText('There is no grasp set in the project: make one on the Tasks page.')
        elif gs_mode:
            self.case_note.setText('Each grasp is solved by IK with the hand closed in its power posture; '
                                   'infeasible grasps are counted and left out.')
        else:
            self.case_note.setText('The open hand, then each grasp posture at 25, 50, 75 and 100 % closed (steps, '
                                   'a default). The postures are sample sweeps between two joint vectors.')

    def live_q(self) -> dict:
        return {uid: [float(x) for x in np.asarray(q, float).reshape(-1)] for uid, q in self.ctx.store.q.items()}

    def run(self):
        """Run with the cases' source and the arm's joint state (passed with the run, not stored)."""
        settings = {'source': self.source.value(), 'q': {} if self.pose.value() == 'home' else self.live_q()}
        if self.source.value() == 'grasp_set':
            uid = self.c_set.currentData()
            if not uid:
                if self.ctx.window is not None:
                    self.ctx.window.msg('Make a grasp set on the Tasks page first, or sweep the grasp postures.',
                                        6000)
                return
            settings['grasp_set_uid'] = uid
        self.ctx.analysis.run(self.STUDY, settings)

    # ---------------------------------------------------------------- result
    def empty(self):
        self.head.set('—', '', 'warning', 'not run yet', 'Run multi-finger occlusion to see which fingers hide which '
                                                         'fingertip from each camera.')
        self.matrix.set_data([], [], [])
        self.owners.set_data([], [], [])
        self.sweep.set_data([], [], [])
        self.table.clear()
        self.suggestion_ready(False)

    def _cams(self, doc) -> list:
        return doc.get('cameras') or []

    def _cam_index(self, doc) -> int:
        cams = self._cams(doc)
        uids = [c['uid'] for c in cams]
        if self.sel_cam not in uids:
            self.sel_cam = doc.get('ranking', uids)[0] if uids else ''
        return uids.index(self.sel_cam) if self.sel_cam in uids else -1

    def _slot(self, uid) -> int:
        s = self.ctx.store.project.variant.sensor(uid)
        return s.slot if s is not None else 0

    def show_result(self, doc: dict):
        self._preview = False
        rig = doc.get('rig') or {}
        st = doc.get('settings') or {}
        fingers = doc.get('fingers') or []
        cams = self._cams(doc)
        if not cams or not fingers:
            self.head.set('—', '', 'warning', 'nothing to show', 'No camera or fingertip was analysed.')
            return
        best = next((c for c in cams if c['uid'] == (doc.get('ranking') or [''])[0]), cams[0])
        tip = (f'A fingertip counts as seen in a case from {st.get("min_visible_pct", 50):g} % of its patch '
               f'(min_visible_pct); status good from {st.get("good_pct", 90):g} %, warning from '
               f'{st.get("warn_pct", 60):g} %, serious from {st.get("serious_pct", 30):g} % (defaults). '
               f'{len(doc.get("cases") or [])} cases, {len(cams)} cameras.')
        advice = doc.get('advice', '')
        mount = '' if 'best single mount' in advice else \
            f'; best single mount: {best["name"]} ({best["fingertips_seen_pct"]:.1f} %)'    # said once, not twice
        self.head.set(f'{rig.get("fingertips_seen_pct", 0):.1f}', '%', rig.get('status', 'warning'),
                      rig.get('status_word', ''),
                      f'Every fingertip seen in {rig.get("all_fingertips_rate_pct", 0):.1f} % of the cases{mount}. '
                      f'{advice}', tip)
        self.suggestion_ready(bool(doc.get('suggestion')))
        # the camera combo
        self.c_cam.blockSignals(True)
        self.c_cam.clear()
        for c in cams:
            self.c_cam.addItem(c['name'], c['uid'])
        j = self._cam_index(doc)
        self.c_cam.setCurrentIndex(max(j, 0))
        self.c_cam.blockSignals(False)
        # the case combo
        self.c_case.blockSignals(True)
        cur = self.c_case.currentData()
        self.c_case.clear()
        for c in doc.get('per_case') or []:
            self.c_case.addItem(c['label'], c['id'])
        k = self.c_case.findData(cur)
        self.c_case.setCurrentIndex(k if k >= 0 else 0)
        self.c_case.blockSignals(False)
        self._fill_matrix(doc)
        self._fill_detail(doc)
        self._fill_table(doc)
        notes = list(doc.get('notes') or [])
        g = doc.get('grasps')
        if g:
            bad = ', '.join(f'{v} {k}' for k, v in (g.get('counts') or {}).items() if k != 'ok')
            notes.insert(0, f'Grasp set "{g.get("set_name", "")}": {g["feasible"]} of {g["taken"]} grasps reached'
                            + (f' ({bad})' if bad else '') + ('; self-collision not checked'
                                                              if not g.get('collision_checked') else '') + '.')
        oc = doc.get('occlusion') or {}
        tm = doc.get('timing') or {}
        notes.append(f'Occlusion by {oc.get("renderer") or oc.get("backend")}; {tm.get("cases", 0)} cases × '
                     f'{tm.get("cameras", 0)} cameras × {tm.get("patch_points", 0)} patch points in '
                     f'{tm.get("seconds", 0):.1f} s.')
        notes += [f'{c["name"]}: {c["imager_why"]}' for c in cams]
        self.assume.setText('\n'.join(notes))
        self._redraw()

    def _fill_matrix(self, doc):
        fingers = doc['fingers']
        cams = self._cams(doc)
        mx = doc['matrix']
        titles = {g['key']: g['title'] for g in doc.get('groups') or []}
        cats = mx['categories']
        notes, tips = [], []
        for i, f in enumerate(fingers):
            nr, tr = [], []
            for j, c in enumerate(cams):
                dom = mx['dominant'][i][j]
                nr.append(('by ' + titles.get(dom, dom).lower()) if dom and dom != D.OUT else
                          ('outside the image' if dom == D.OUT else ''))
                split = mx['by_owner_pct'][i][j]
                parts = [f'{titles.get(cats[g], cats[g])} {v:.1f} %' for g, v in sorted(
                    enumerate(split), key=lambda t: -(t[1] or 0)) if v and v >= 0.05]
                tr.append(['Not seen: ' + (', '.join(parts) if parts else 'nothing')])
            notes.append(nr)
            tips.append(tr)
        j = self._cam_index(doc)
        self.matrix.set_data([f['title'] for f in fingers], [c['name'].replace(' (sample placement)', '')
                                                            for c in cams],
                             mx['not_seen_pct'], notes, tips, [THEME.cam(self._slot(c['uid'])) for c in cams], j)
        from .. import colormaps as CM
        more = 'Lighter' if CM._theme(None) == 'dark' else 'Darker'      # the ramp runs away from the background
        self.matrix_note.setText(f'{more} is more hidden. Each cell: the share of that fingertip\'s contact patch the '
                                 'camera does not see, averaged over the cases, and what hides most (the owner of the '
                                 'first hit along the line of sight, or outside the image). Click a column to choose '
                                 'its camera.')

    def _fill_detail(self, doc):
        fingers = doc['fingers']
        j = self._cam_index(doc)
        if j < 0:
            return
        mx = doc['matrix']
        titles = {g['key']: g['title'] for g in doc.get('groups') or []}
        cats = mx['categories']
        vals = [[mx['by_owner_pct'][i][j][g] for g in range(len(cats))] for i in range(len(fingers))]
        keep = [g for g in range(len(cats)) if any((vals[i][g] or 0) > 0 for i in range(len(fingers)))] or [0]
        self.owners.set_data([f['title'] for f in fingers], [titles.get(cats[g], cats[g]) for g in keep],
                             [[vals[i][g] for g in keep] for i in range(len(fingers))],
                             tips=[[[f'share of {f["title"]}\'s patch hidden by {titles.get(cats[g], cats[g]).lower()}'
                                     if cats[g] != D.OUT else 'share outside the image'] for g in keep]
                                   for f in fingers],
                             col_colors=[owner_color(cats[g], fingers) for g in keep])
        self.owners.subtitle = '% of the patch, rows: whose patch, columns: what hides it'
        pc = doc.get('per_case') or []
        self.sweep.set_data([c['label'] for c in pc], [f['title'] for f in fingers],
                            [[None if c['seen_pct'][j][i] is None else 100 - c['seen_pct'][j][i]
                              for i in range(len(fingers))] for c in pc],
                            notes=[[('by ' + titles.get(c['dominant'][j][i], '').lower()
                                     if c['dominant'][j][i] not in ('', D.OUT) else
                                     ('outside' if c['dominant'][j][i] == D.OUT else ''))
                                    for i in range(len(fingers))] for c in pc])
        self.sweep.subtitle = f'% of the patch not seen by {self._cams(doc)[j]["name"]}'

    def _fill_table(self, doc):
        self.table.clear()
        titles = {g['key']: g['title'] for g in doc.get('groups') or []}
        ftitle = {f['name']: f['title'] for f in doc.get('fingers') or []}
        n_f = len(doc.get('fingers') or [])
        by = {c['uid']: c for c in self._cams(doc)}
        for uid in doc.get('ranking') or []:
            c = by[uid]
            vals = [str(c['rank']), c['name'], c['mount'], _pct(c['fingertips_seen_pct']),
                    _pct(c['all_fingertips_rate_pct']), _pct(c['seen_pct']), _pct(c['outside_image_pct']),
                    _pct(c['hidden_pct']), f'{ftitle.get(c["worst_finger"], c["worst_finger"])} '
                                           f'({c["worst_finger_seen_pct"]:.0f} % seen)',
                    titles.get(c['main_occluder'], c['main_occluder'] or '—')]
            it = QTreeWidgetItem(vals)
            for k in (0, 3, 4, 5, 6, 7):
                it.setFont(k, mono(11))
                it.setTextAlignment(k, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            it.setForeground(1, THEME.cam(self._slot(uid)))
            it.setToolTip(3, 'Share of fingertip-cases in which this camera alone sees the patch (from the threshold)')
            it.setToolTip(4, f'Share of cases in which this camera sees all {n_f} fingertips')
            it.setToolTip(5, 'Mean share of the patch points this camera sees')
            it.setToolTip(2, f'Mounted on {c["parent_frame"]}' + (' (locked)' if c['locked'] else ''))
            self.table.addTopLevelItem(it)
        _fit(self.table)

    def _cam_changed(self, i):
        self.sel_cam = self.c_cam.itemData(i) or ''
        if self.doc is not None:
            self._fill_matrix(self.doc)
            self._fill_detail(self.doc)
        self._redraw()

    def _matrix_col(self, j):
        if 0 <= j < self.c_cam.count():
            self.c_cam.setCurrentIndex(j)

    # ---------------------------------------------------------------- cases in the 3D view
    def _case(self):
        pc = (self.doc or {}).get('per_case') or []
        k = self.c_case.currentIndex()
        return pc[k] if 0 <= k < len(pc) else None

    def show_case(self, k: int) -> bool:
        """Pose the hand's robot as in case k (view state, not an edit)."""
        pc = (self.doc or {}).get('per_case') or []
        if not (0 <= k < len(pc)):
            return False
        uid = (self.doc.get('robot') or {}).get('uid')
        try:
            self.ctx.store.set_q(uid, np.asarray(pc[k]['q'], float))
        except Exception:                                # noqa: BLE001 — a robot that cannot be posed: say so
            traceback.print_exc()
            if self.ctx.window is not None:
                self.ctx.window.msg('That case cannot be shown: the robot has changed since the study ran. Run it '
                                    'again.', 6000)
            return False
        if self.c_case.currentIndex() != k:
            self.c_case.setCurrentIndex(k)
        self._redraw()
        return True

    def _case_on_screen(self):
        """(case, robot uid) when the posture on screen is the chosen case, else (None, uid)."""
        doc = self.doc
        if not doc:
            return None, ''
        uid = (doc.get('robot') or {}).get('uid', '')
        c = self._case()
        q = self.ctx.store.q.get(uid)
        if c is None or q is None or len(q) != len(c['q']):
            return None, uid
        if not np.allclose(np.asarray(q, float), np.asarray(c['q'], float), atol=Q_ATOL):
            return None, uid
        return c, uid

    def _patch_state(self):
        """[(finger, points (K, 3) world, seen (K,) bool)] of the case on screen for the chosen camera, or []."""
        from ...core import dexhand as DH
        from ...core import grasps as GR
        c, uid = self._case_on_screen()
        doc = self.doc
        if c is None:
            return []
        j = self._cam_index(doc)
        if j < 0:
            return []
        p = self.ctx.store.project
        try:
            hand = DH.structure(p, uid)
            patches = GR.contact_patches(p, uid, np.asarray(c['q'], float))
        except Exception:                                # noqa: BLE001 — the robot changed since the run
            return []
        n_total = sum(f['points'] for f in doc['fingers'])
        bits = D.unpack_bits(c['seen_bits'][j], n_total)
        out, at = [], 0
        for f in doc['fingers']:
            pts = [v['points'] for v in patches.values() if hand.link_group.get(v['link']) == f['name']]
            P = np.concatenate(pts) if pts else np.zeros((0, 3))
            if len(P) != f['points']:
                return []
            out.append((f, P, bits[at:at + len(P)]))
            at += len(P)
        return out

    def _redraw(self):
        self._overlay_key = None
        if self.ctx.window is not None:
            self.ctx.window.viewport.update()

    def overlays(self, vp) -> list:
        doc = self.doc
        if not doc:
            return []
        c, _uid = self._case_on_screen()
        key = (id(doc), self.sel_cam, None if c is None else c['id'], self._preview, self.ctx.store.revision)
        if key == self._overlay_key:
            return self._overlay_items
        items = []
        j = self._cam_index(doc)
        cam_col = THEME.cam(self._slot(self.sel_cam)) if j >= 0 else THEME.color('ink')
        for _f, P, seen in self._patch_state():
            for mask, col in ((seen, cam_col), (~seen, THEME.color('muted'))):
                Q = P[mask]
                if not len(Q):
                    continue
                segs = []
                for ax in range(3):
                    d = np.zeros(3)
                    d[ax] = CROSS_M
                    segs.append(np.stack([Q - d, Q + d], 1))
                items.append(LineDraw(np.concatenate(segs), col, 1.2, segments=True))
        if self._preview:
            items += self._axis_preview()
        self._overlay_key, self._overlay_items = key, items
        return items

    def _axis_preview(self) -> list:
        from ...core import library as LB
        from ...core import scenegeom as SG
        from ...core.frames import Pose
        sug = (self.doc or {}).get('suggestion')
        if not sug:
            return []
        p = self.ctx.store.project
        s = p.variant.sensor(sug['sensor_uid'])
        if s is None:
            return []
        try:
            im = LB.sensor_imagers(s)[0]
            T_now = SG.sensor_world_pose(p, s, self.ctx.store.q).matrix()
            s2 = copy.deepcopy(s)
            s2.mount.pose = Pose.from_dict(sug['pose'])
            T_new = SG.sensor_world_pose(p, s2, self.ctx.store.q).matrix()
        except Exception:                                # noqa: BLE001 — a sensor that cannot be posed is not drawn
            return []
        out = []
        for T, dash in ((T_now, False), (T_new, True)):
            M = T @ im.pose_in_body.matrix()
            o, z = M[:3, 3], M[:3, 2]
            out.append(LineDraw(np.stack([o, o + AXIS_M * z]), THEME.cam(s.slot), 1.6, dash=dash, dash_len=0.01))
        return out

    # ---------------------------------------------------------------- Camera view
    def camview_overlays(self, cam_model, pose) -> list:
        """The chosen case's patch points in the Camera view of a camera of the study: seen ones as crosses in the
        finger's colour, the others in grey (they are behind something, or outside this image)."""
        from ...core import camrender as CR
        doc = self.doc
        if not doc:
            return []
        cv = getattr(self.ctx.window, 'camview', None) if self.ctx.window is not None else None
        shown = cv.view.sensor() if cv is not None and hasattr(cv, 'view') else None
        if shown is None or shown.uid not in [c['uid'] for c in self._cams(doc)]:
            return []
        keep = self.sel_cam
        self.sel_cam = shown.uid
        try:
            state = self._patch_state()
        finally:
            self.sel_cam = keep
        out = []
        r = 3.0
        for i, (f, P, seen) in enumerate(state):
            uv, ok, _d = CR.project_world(cam_model, pose, P)
            col = QColor(FINGER_COLORS[i % len(FINGER_COLORS)])
            for mask, colour, word in ((seen & ok, col, 'seen'), (~seen & ok, THEME.color('muted'), 'not seen')):
                U = np.asarray(uv)[mask]
                if not len(U):
                    continue
                polys = []
                for u, v in U:
                    polys.append(np.array([[u - r, v], [u + r, v]]))
                    polys.append(np.array([[u, v - r], [u, v + r]]))
                pct = 100.0 * float(seen.mean()) if len(seen) else 0.0
                lab = f'{f["title"]} · {pct:.0f} % seen' if word == 'seen' else ''
                out.append((polys, colour, lab))
        return out

    # ---------------------------------------------------------------- Apply / Preview
    def apply(self):
        from ...core.frames import Pose
        sug = (self.doc or {}).get('suggestion')
        if not sug:
            return
        p = self.ctx.store.project
        s = p.variant.sensor(sug['sensor_uid'])
        if s is None or s.locked:
            if self.ctx.window is not None:
                self.ctx.window.msg('That camera is no longer in the rig or is locked; run the study again.', 6000)
            return
        if s.mount.parent_frame != sug['parent_frame']:
            if self.ctx.window is not None:
                self.ctx.window.msg(f'{s.name} has moved to another mount since the study ran; run it again.', 6000)
            return
        uid = s.uid

        def fn(pr):
            x = pr.variant.sensor(uid)
            x.mount.pose = Pose.from_dict(sug['pose'])
        self.ctx.store.edit('variant', fn, f'Turn {s.name} {abs(sug["deg"]):g}° {sug["direction"]}')
        self._preview = False
        self.suggestion_ready(False)
        self.run()

    def preview(self):
        self._preview = not self._preview and bool((self.doc or {}).get('suggestion'))
        if self.ctx.window is not None:
            self.ctx.window.msg('Preview: the dashed line is the suggested optical axis, the solid one the current'
                                if self._preview else 'Preview off', 6000)
        self._redraw()


register('dexhand', 'Multi-finger occlusion', DexHandPage, menu='Analysis', icon='hand', action_id=None,
         workspace='grasp', order=33)
