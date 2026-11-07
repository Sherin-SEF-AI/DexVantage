"""The hand-eye calibration pages (spec §8.1, §13): plan the poses, record and solve, and the coach.

Hand-eye pose plan (Calibrate ▸ Plan hand-eye poses…): the camera and the board, how many poses (default 20, yours to
set), the corner noise and the requirement (defaults, labelled); the board's definition and where it lies or is held
(saved to the project, one undo step; printable at true scale). Run plans the poses (core/calib/planner.py): the
headline is the predicted 1σ of the camera's loosest axis with a status word against the requirement; tables give σ of
X and Z in mm and degrees, the planned poses with what each adds, and what one more pose would gain. Apply saves the
plan as a calibration session (one undo step); Preview moves the robot to the next planned pose; the 3D view draws the
planned camera and gripper poses, the path between them and the board; the Camera view draws the board's outline.

Hand-eye solve (Calibrate ▸ Solve hand-eye…): import a folder of board images and a joint_states file (CSV or JSON,
matched by time within a tolerance, default 50 ms; core/calib/capture.py) as a session (one undo step), or, with "From
a LeRobot dataset or ROS 2 bag…", the frames of a recording with the board in view and the joint states at their times
(capture_dataset, on a worker thread; errors said in words). Run solves the session (core/calib/handeye.py): the
headline is the calibrated camera's loosest 1σ, the residual table each view's RMS and whether it was used, the
covariance table σ of X and Z. Apply writes the sensor's mount pose (and its lens calibration when the intrinsics
were estimated) and marks the session applied, one undo step. The coach (core/calib/coach.py) says which pose to take
next and why; Preview moves the robot there.
"""
from __future__ import annotations

import math
import os
import traceback

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QGridLayout, QHeaderView, QMessageBox,
                             QTreeWidget, QTreeWidgetItem, QWidget)

from ..theme import THEME, mono
from ..widgets import ScrubSpinBox, SegmentedControl, button, label
from . import register
from .base import StudyPage, row

PLAN = 'handeye_plan'
SOLVE = 'handeye_solve'
RIGHT = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
SIG_COLS = ('x_mm', 'y_mm', 'z_mm', 'rx_deg', 'ry_deg', 'rz_deg')


# =========================================================================== small helpers
def _tree(headers, stretch: int | None = None) -> QTreeWidget:
    """A table; column `stretch` takes the spare width (default the last, a text column)."""
    t = QTreeWidget()
    t.setRootIsDecorated(False)
    t.setHeaderLabels(list(headers))
    h = t.header()
    h.setStretchLastSection(False)
    for k in range(len(headers)):
        h.setSectionResizeMode(k, QHeaderView.ResizeMode.ResizeToContents)
    h.setSectionResizeMode(len(headers) - 1 if stretch is None else stretch, QHeaderView.ResizeMode.Stretch)
    return t


def _fit(tree: QTreeWidget, cap: int = 12):
    n = tree.topLevelItemCount()
    rh = tree.sizeHintForRow(0) if n else 20
    tree.setFixedHeight(tree.header().sizeHint().height() + max(1, min(n, cap)) * max(rh, 18) + 6)


def _item(texts, mono_from: int = 1) -> QTreeWidgetItem:
    it = QTreeWidgetItem([str(t) for t in texts])
    for k in range(mono_from, len(texts)):
        it.setFont(k, mono(11))
        it.setTextAlignment(k, RIGHT)
    return it


def _fmt(key: str, v) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return 'not determined'
    return f'{v:.4f} mm' if key.endswith('_mm') else f'{v:.5f}°'


def _sigma_rows(tree: QTreeWidget, rows):
    tree.clear()
    for name, frame, sig in rows:
        it = _item([name] + [_fmt(k, (sig or {}).get(k)) for k in SIG_COLS])
        it.setToolTip(0, frame)
        for k, key in enumerate(SIG_COLS, start=1):
            it.setToolTip(k, '1σ ' + ('along the parent frame\'s ' + key[0] if key.endswith('_mm') else
                                      'about the frame\'s own ' + key[1] + ' axis') +
                          ('; "not determined": the views do not pin it' if (sig or {}).get(key) is None else ''))
        tree.addTopLevelItem(it)
    _fit(tree)


def _headline_figure(doc) -> tuple[str, str]:
    w = doc.get('worst') or {}
    key, v = w.get('key') or '', w.get('value')
    if v is None or not key:
        return '—', ''
    unit = 'mm' if key.endswith('_mm') else '°'
    return (f'{v:.4f}' if unit == 'mm' else f'{v:.5f}'), f'{unit} ({w.get("axis", "")}, 1σ)'


def _board_ring(board, T) -> np.ndarray:
    """The board's printed area as a closed world polyline (5, 3)."""
    P = np.asarray(board.points(), float)
    s = float(getattr(board, 'square_m', getattr(board, 'tag_m', 0.02)))
    lo, hi = P[:, :2].min(0) - s, P[:, :2].max(0) + s
    ring = np.array([[lo[0], lo[1], 0], [hi[0], lo[1], 0], [hi[0], hi[1], 0], [lo[0], hi[1], 0], [lo[0], lo[1], 0]])
    return ring @ T[:3, :3].T + T[:3, 3]


def _triad(T, size=0.025) -> list:
    from ..viewport3d import LineDraw
    o = T[:3, 3]
    cols = (QColor('#e5484d'), QColor('#30a46c'), QColor('#3e63dd'))
    return [LineDraw(np.array([[o, o + size * T[:3, k]]]), cols[k], 1.5, segments=True) for k in range(3)]


def _pose_matrix(d) -> np.ndarray:
    from ...core.frames import Pose
    return Pose.from_dict(d).matrix()


class _HandEyeBase(StudyPage):
    """Shared plumbing of the two pages."""

    def _set(self, sid, key, value, text):
        cur = (self.ctx.store.project.analysis.studies.get(sid) or {}).get(key)
        if cur == value:
            return

        def fn(p):
            d = dict(p.analysis.studies.get(sid) or {})
            if value in (None, ''):
                d.pop(key, None)
            else:
                d[key] = value
            p.analysis.studies[sid] = d
        self.ctx.store.edit('analysis', fn, text, merge_key=(sid, key))

    def live_q(self) -> dict:
        return {uid: [float(x) for x in np.asarray(q, float).reshape(-1)] for uid, q in self.ctx.store.q.items()}

    def inputs_key(self) -> str:
        """What the study reads from the project, as text (the robot's live pose left out)."""
        import json
        p = self.ctx.store.project
        st = {k: v for k, v in (p.analysis.studies.get(self.STUDY) or {}).items() if k != 'q'}
        # a plan reads the boards, not the sessions
        cal = p.calibration if self.STUDY == SOLVE else {'boards': p.calibration.get('boards')}
        return json.dumps([p.variant.to_dict(), [r.to_dict() for r in p.robots], p.workspace.to_dict(), cal, st],
                          sort_keys=True, default=str)

    def _check_stale(self):
        if self.doc is None or getattr(self, '_inputs', None) is None:
            self.stale_note.setText('')
            return
        self.stale_note.setText('The project changed since this run: Run again to update it.'
                                if self.inputs_key() != self._inputs else '')

    def _msg(self, text, ms=6000):
        if self.ctx.window is not None:
            self.ctx.window.msg(text, ms)

    def _redraw(self):
        win = self.ctx.window
        if win is not None and hasattr(win, 'viewport'):
            win.viewport.update()
        cv = getattr(win, 'camview', None) if win is not None else None
        if cv is not None and hasattr(cv, 'view') and hasattr(cv.view, 'refresh_overlays'):
            try:
                cv.view.refresh_overlays()
            except Exception:                       # noqa: BLE001 — an overlay must not take the page down
                traceback.print_exc()

    def _failed(self, sid, message):
        if sid == self.STUDY:
            from ..store import failure_text
            text, _explained = failure_text(message, self.TITLE)
            self.head.set('—', '', 'critical', 'could not run', text)

    def _board_now(self):
        """(board, world pose 4 × 4) of the board at the robot's current joint state, from the last result."""
        doc = self.doc or {}
        b = doc.get('board') or {}
        try:
            from ...core.calib.boards import board_from_dict
            if 'definition' in b:
                board = board_from_dict(dict(b['definition']))
                Z = _pose_matrix(b['Z_nominal'])
            else:
                from ...core.calib import records as RC
                rec = RC.find_board(self.ctx.store.project, b.get('uid') or None)
                if rec is None:
                    return None
                board, Z = RC.board_of(rec), RC.board_mount(rec)[2]
        except (ValueError, KeyError, TypeError):
            return None
        mode = doc.get('mode') or (doc.get('session') or {}).get('mode')
        if mode == 'eye_to_hand':
            from ...core.robot.model import RobotModelError, get_model
            uid = doc.get('robot_uid') or ''
            link = doc.get('link') or ''
            if not uid:
                return None
            try:
                m = get_model(self.ctx.store.project, uid)
                q = self.ctx.store.q.get(uid, m.home)
                return board, m.link_pose(q, link) @ Z
            except (RobotModelError, KeyError, ValueError):
                return None
        return board, Z

    def camview_overlays(self, cam_model, pose) -> list:
        """The board's outline as the camera sees it at the current joint state."""
        from ...core import camrender as CR
        got = self._board_now()
        if got is None:
            return []
        board, T = got
        polys = CR.visible_polylines(None, cam_model, pose, _board_ring(board, T), step=0.005)
        return [(polys, THEME.color('ink'), 'calibration board')] if polys else []


# =========================================================================== the plan page
class HandEyePlanPage(_HandEyeBase):
    """Hand-eye pose plan: robot poses the arm reaches without touching itself or the scene, with the board in the
    camera's view, chosen greedily to maximise the Fisher information of the camera's pose X and the board's Z (the
    D-optimal criterion), and the covariance of X and Z they predict in mm and degrees."""
    STUDY = PLAN
    TITLE = 'Hand-eye pose plan'
    HEADLINE = 'Predicted 1σ of the camera\'s loosest axis'

    def build_body(self, layout):
        from ...core.calib import planner as PL
        self._overlay = []
        self._preview_k = -1
        layout.addWidget(label('Setup (your settings; defaults labelled)', bold=True))
        self.sensor_box = QComboBox()
        self.sensor_box.setToolTip('The camera to calibrate: on a robot link (eye-in-hand) or fixed (eye-to-hand)')
        self.board_box = QComboBox()
        self.board_box.setToolTip('The board: one saved in the project, or DexVantage\'s default placement')
        layout.addWidget(row(label('Camera'), self.sensor_box, label('Board'), self.board_box))
        g = QWidget()
        grid = QGridLayout(g)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        self._spin = {}
        specs = (('n_poses', 'Poses', 0, 'poses', 1.0, PL.LIMITS['n_poses'],
                  'How many poses to plan (default 20; the specification asks for 15 to 25)'),
                 ('pixel_sigma', 'Corner noise', 2, 'px', 0.05, PL.LIMITS['pixel_sigma'],
                  'The 1σ noise of a detected corner on each image axis (default 0.3 px, a default: the Hand-eye solve '
                  'page measures yours)'),
                 ('need_mm', 'Requirement', 3, 'mm', 0.05, PL.LIMITS['need_mm'],
                  'The 1σ the camera\'s position should reach (default 0.5 mm, a DexVantage default, not a standard)'),
                 ('need_deg', 'and', 3, '°', 0.01, PL.LIMITS['need_deg'],
                  'The 1σ the camera\'s rotation should reach (default 0.05°, a DexVantage default)'))
        for i, (key, text, dec, unit, step, (lo, hi), tip) in enumerate(specs):
            sb = ScrubSpinBox('', 'plain', dec, unit, step=step, width=100)
            sb.setRange(lo, hi)
            sb.setToolTip(tip)
            dl = label('', muted=True, small=True)
            grid.addWidget(label(text), i // 2, (i % 2) * 3)
            grid.addWidget(sb, i // 2, (i % 2) * 3 + 1)
            grid.addWidget(dl, i // 2, (i % 2) * 3 + 2)
            self._spin[key] = (sb, dl)
            sb.committed.connect(lambda v, k=key, t=text: self._set_number(k, v, t))
        grid.setColumnStretch(6, 1)
        layout.addWidget(g)

        layout.addWidget(label('Board and where it is', bold=True))
        self.kind = SegmentedControl([('charuco', 'ChArUco'), ('aprilgrid', 'AprilGrid')], 'charuco')
        self.b_cols = ScrubSpinBox('', 'plain', 0, 'cols', step=1, width=80)
        self.b_rows = ScrubSpinBox('', 'plain', 0, 'rows', step=1, width=80)
        self.b_size = ScrubSpinBox('', 'plain', 2, 'mm', step=0.5, width=90)
        self.b_marker = ScrubSpinBox('', 'plain', 2, 'mm', step=0.5, width=90)
        for sb, lo, hi in ((self.b_cols, 2, 40), (self.b_rows, 2, 40), (self.b_size, 2, 500), (self.b_marker, 1, 500)):
            sb.setRange(lo, hi)
        self.b_cols.setToolTip('Squares (ChArUco) or tags (AprilGrid) across')
        self.b_rows.setToolTip('Squares or tags down')
        self.b_size.setToolTip('A square\'s side (ChArUco) or a tag\'s outer edge (AprilGrid) as printed: measure it')
        self.b_marker.setToolTip('The ArUco marker\'s side (ChArUco only)')
        layout.addWidget(row(self.kind, self.b_cols, self.b_rows, label('size'), self.b_size, label('marker'),
                             self.b_marker))
        self._place = {}
        cells = []
        for key, unit, tip in (('x', 'mm', 'x'), ('y', 'mm', 'y'), ('z', 'mm', 'z'), ('roll', '°', 'roll about x'),
                               ('pitch', '°', 'pitch about y'), ('yaw', '°', 'yaw about z')):
            sb = ScrubSpinBox(key, 'plain', 1, unit, step=1.0, width=96 if unit == 'mm' else 116)  # axis named
            sb.setRange(-10000, 10000)
            sb.setToolTip(f'The board frame\'s {tip} in its parent frame (fixed axes: roll, then pitch, then yaw)')
            self._place[key] = sb
            cells.append(sb)
        self.parent_label = label('', muted=True, small=True)
        layout.addWidget(row(*cells, self.parent_label))
        self.b_save = button('Save board', tip='Save this board and its place in the project (one undo step)')
        self.b_print = button('Print board…', role='ghost', tip='Write the board as a PDF (vector, true size) or a PNG')
        self.board_note = label('', muted=True, wrap=True)
        layout.addWidget(row(self.b_save, self.b_print))
        layout.addWidget(self.board_note)
        self.b_save.clicked.connect(lambda: self.save_board())
        self.b_print.clicked.connect(lambda: self.print_board())

        self.stale_note = label('', muted=True, wrap=True)
        self._inputs = None
        layout.addWidget(self.stale_note)
        layout.addWidget(label('Predicted covariance', bold=True))
        self.sig_tree = _tree(['Transform', 'σ x', 'σ y', 'σ z', 'σ rx', 'σ ry', 'σ rz'], stretch=0)
        layout.addWidget(self.sig_tree)
        layout.addWidget(label('Planned poses (in the order to take them)', bold=True))
        self.pose_tree = _tree(['#', 'Camera', 'Corners', 'Hidden', 'Loosest axis without it'], stretch=1)
        layout.addWidget(self.pose_tree)
        self.next_label = label('', wrap=True)
        self.cand_label = label('', muted=True, wrap=True)
        self.notes_label = label('', muted=True, wrap=True)
        self.show3d = QCheckBox('Show in the 3D view')
        self.show3d.setChecked(True)
        self.show3d.toggled.connect(lambda _on: self._rebuild_overlay())
        for w in (self.next_label, self.cand_label, self.notes_label, self.show3d):
            layout.addWidget(w)
        self.b_apply.setText('Save as session')
        self.b_apply.setToolTip('Save the planned poses as a calibration session to record (one undo step)')
        self.b_preview.setText('Next pose')
        self.b_preview.setToolTip('Move the robot in the 3D view to the next planned pose (view only)')
        self.sensor_box.activated.connect(lambda _i: self._set(PLAN, 'sensor', self.sensor_box.currentData() or '',
                                                               'Hand-eye plan · camera'))
        self.board_box.activated.connect(lambda _i: self._set(PLAN, 'board', self.board_box.currentData() or '',
                                                              'Hand-eye plan · board'))
        self.pose_tree.itemSelectionChanged.connect(self._rebuild_overlay)
        st = self.ctx.store
        st.changed.connect(lambda _s: (self._sync(), self._check_stale()))
        st.project_replaced.connect(self._replaced)
        self._sync()

    # ---------------------------------------------------------------- settings
    def settings(self) -> dict:
        return dict(self.ctx.store.project.analysis.studies.get(PLAN) or {})

    def plan_settings(self):
        from ...core.calib import planner as PL
        return PL.PlanSettings.from_dict(self.settings())

    def _set_number(self, key, value, text):
        v = int(round(value)) if key == 'n_poses' else float(value)
        self._set(PLAN, key, v, f'Hand-eye plan · {text.lower()} {v:g}')

    def _sync(self):
        from ...core.calib import planner as PL
        from ...core.calib import records as RC
        p = self.ctx.store.project
        st = self.settings()
        ps = PL.PlanSettings.from_dict(st)
        for key, (sb, dl) in self._spin.items():
            sb.set_value(float(getattr(ps, key)))
            d = getattr(PL.PlanSettings(), key)
            dl.setText('default' if key in ps.to_dict()['defaults'] else f'yours (default {d:g})')
        self.sensor_box.blockSignals(True)
        self.sensor_box.clear()
        for s in p.variant.sensors:
            if s.kind in ('camera', 'depth'):
                where = 'on ' + s.mount.parent_frame if s.mount.on_link else 'fixed'
                self.sensor_box.addItem(f'{s.name} ({where})', s.uid)
        k = self.sensor_box.findData(st.get('sensor') or '')
        self.sensor_box.setCurrentIndex(k if k >= 0 else 0)
        self.sensor_box.blockSignals(False)
        self.board_box.blockSignals(True)
        self.board_box.clear()
        for b in RC.calibration_of(p).get('boards') or []:
            self.board_box.addItem(str(b.get('name') or 'Board'), b.get('uid'))
        if not self.board_box.count():
            self.board_box.addItem('Default board (not saved)', '')
        k = self.board_box.findData(st.get('board') or '')
        self.board_box.setCurrentIndex(k if k >= 0 else 0)
        self.board_box.blockSignals(False)
        self._load_board()

    def _replaced(self):
        self.doc, self._overlay, self._preview_k, self._inputs = None, [], -1, None
        self.stale_note.setText('')
        for t in (self.sig_tree, self.pose_tree):
            t.clear()
        for lb in (self.next_label, self.cand_label, self.notes_label):
            lb.setText('')
        self.suggestion_ready(False)
        self._sync()
        self.empty()

    # ---------------------------------------------------------------- the board editor
    def board_record(self) -> dict | None:
        """The board record the plan uses: the chosen one, else the default for the chosen camera's mode."""
        from ...core.calib import records as RC
        p = self.ctx.store.project
        rec = RC.find_board(p, self.board_box.currentData() or None)
        if rec is not None:
            return rec
        s = p.variant.sensor(self.sensor_box.currentData() or '') if self.sensor_box.count() else None
        if s is None or not p.robots:
            return None
        mode = 'eye_in_hand' if s.mount.on_link else 'eye_to_hand'
        try:
            return RC.default_board_record(p, mode, s.mount.robot_uid if s.mount.on_link else '')
        except Exception:                               # noqa: BLE001 — no robot to hold it: nothing to edit
            traceback.print_exc()
            return None

    def _load_board(self):
        from scipy.spatial.transform import Rotation

        from ...core.frames import Pose
        rec = self.board_record()
        on = rec is not None
        for w in (self.kind, self.b_cols, self.b_rows, self.b_size, self.b_marker, self.b_save, self.b_print,
                  *self._place.values()):
            w.setEnabled(on)
        if not on:
            self.parent_label.setText('')
            self.board_note.setText('Open a project with a robot and a camera to plan a hand-eye calibration.')
            return
        b = rec.get('board') or {}
        kind = b.get('kind', 'charuco')
        self.kind.set_value(kind)
        if kind == 'aprilgrid':
            self.b_cols.set_value(float(b.get('tag_cols', 6)))
            self.b_rows.set_value(float(b.get('tag_rows', 6)))
            self.b_size.set_value(1000 * float(b.get('tag_m', 0.04)))
            self.b_marker.set_value(0.0)
        else:
            self.b_cols.set_value(float(b.get('squares_x', 9)))
            self.b_rows.set_value(float(b.get('squares_y', 7)))
            self.b_size.set_value(1000 * float(b.get('square_m', 0.03)))
            self.b_marker.set_value(1000 * float(b.get('marker_m', 0.022)))
        m = rec.get('mount') or {}
        pose = Pose.from_dict(m.get('pose'))
        for k, v in zip('xyz', pose.position):
            self._place[k].set_value(1000 * float(v))
        e = Rotation.from_quat(pose.quaternion).as_euler('xyz', degrees=True)
        for k, v in zip(('roll', 'pitch', 'yaw'), e):
            self._place[k].set_value(float(v))
        parent = m.get('parent_frame') or 'world'
        self.parent_label.setText('the board frame\'s origin (a corner of the pattern) in '
                                  + ('the world' if parent == 'world' else f'link {parent}'))
        if rec.get('default'):
            self.board_note.setText('Default board and place (DexVantage defaults, not yours): ' +
                                    str(rec.get('where') or '') + '. Enter the board you print and where you put it, '
                                    'then Save board.')
        else:
            self.board_note.setText('')

    def _board_from_fields(self) -> dict:
        n_c, n_r = int(round(self.b_cols.value())), int(round(self.b_rows.value()))
        size = self.b_size.value() / 1000.0
        if self.kind.value() == 'aprilgrid':
            return {'kind': 'aprilgrid', 'tag_cols': n_c, 'tag_rows': n_r, 'tag_m': size, 'spacing': 0.3,
                    'first_id': 0, 'corner_squares': True}
        return {'kind': 'charuco', 'squares_x': n_c, 'squares_y': n_r, 'square_m': size,
                'marker_m': self.b_marker.value() / 1000.0, 'dictionary': 'DICT_5X5_100', 'legacy': False}

    def save_board(self) -> str | None:
        """Save the board and its place from the fields (one undo step); the board's uid, or None when invalid."""
        from scipy.spatial.transform import Rotation

        from ...core.calib.boards import BoardError, board_from_dict
        from ...core.frames import Pose
        from ...core.project import new_uid
        rec = self.board_record()
        if rec is None:
            return None
        bd = self._board_from_fields()
        try:
            board_from_dict(dict(bd))
        except (BoardError, ValueError) as e:
            self.board_note.setText(f'This board cannot be built: {e}.')
            return None
        pos = [self._place[k].value() / 1000.0 for k in 'xyz']
        quat = Rotation.from_euler('xyz', [self._place[k].value() for k in ('roll', 'pitch', 'yaw')],
                                   degrees=True).as_quat()
        new = {k: v for k, v in rec.items() if k not in ('where', 'default')}
        new.update(board=bd, default=False, mount=dict(rec.get('mount') or {}, pose=Pose(pos, quat).to_dict()))
        exists = any(b.get('uid') == rec.get('uid') for b in self.ctx.store.project.calibration.get('boards') or [])
        if not exists:
            new['uid'] = new_uid()
        uid = new['uid']

        def fn(p):
            bs = [b for b in p.calibration.get('boards') or [] if b.get('uid') != uid]
            idx = next((i for i, b in enumerate(p.calibration.get('boards') or []) if b.get('uid') == uid), len(bs))
            bs.insert(idx, new)
            p.calibration['boards'] = bs
            d = dict(p.analysis.studies.get(PLAN) or {})
            d['board'] = uid
            p.analysis.studies[PLAN] = d
        self.ctx.store.edit(['calibration', 'analysis'], fn, f'Hand-eye · save board {new.get("name", "")}')
        return uid

    def print_board(self, path: str | None = None) -> str | None:
        from ...core.calib.boards import BoardError, board_from_dict, save_board
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, 'Print board', 'calibration-board.pdf',
                                                  'PDF, true size (*.pdf);;PNG (*.png)')
            if not path:
                return None
        try:
            out = save_board(board_from_dict(self._board_from_fields()), path)
        except (BoardError, ValueError, OSError) as e:
            self._msg(f'The board could not be written: {e}.', 8000)
            return None
        self._msg(f'Wrote {os.path.basename(out)}: print it at 100 % (actual size) and measure a square.', 8000)
        return out

    # ---------------------------------------------------------------- run and result
    def run(self):
        self.ctx.analysis.run(self.STUDY, {'q': self.live_q()})

    def show_result(self, doc):
        fig, unit = _headline_figure(doc)
        req = doc.get('requirement') or {}
        tip = (f'Requirement {req.get("need_mm", 0):g} mm and {req.get("need_deg", 0):g}° (1σ; '
               + ('the defaults' if req.get('default') else 'yours') + '). ' + doc.get('assumptions', ''))
        self.head.set(fig, unit, doc.get('status', 'warning'), doc.get('word', ''), doc.get('advice', ''), tip)
        sg = doc.get('sigma') or {}
        _sigma_rows(self.sig_tree, [(f'X: {doc["camera"]["name"]}', doc['camera']['frame'], sg.get('X')),
                                    (f'Z: {doc["board"]["name"]}', doc['board']['frame'], sg.get('Z'))])
        self.pose_tree.clear()
        for ps in doc.get('poses') or []:
            inc = ps.get('worst_increase_pct')
            it = _item([ps['index'], ps['what'], ps['corners'], ps['hidden'],
                        '—' if inc is None else f'+{inc:.1f} %'], mono_from=2)
            it.setToolTip(4, 'How much the camera\'s loosest σ grows when this pose is left out')
            self.pose_tree.addTopLevelItem(it)
        _fit(self.pose_tree)
        nx = doc.get('next')
        if nx and nx.get('worst_gain_pct') is not None:
            k = nx.get('worst_axis_before') or ''
            a = _fmt(k, (nx.get('sigma_X_before') or {}).get(k))
            b = _fmt(k, (nx.get('sigma_X_after') or {}).get(k))
            self.next_label.setText(f'One more pose (the camera {nx["what"]}) would bring the loosest axis from {a} to '
                                    f'{b} ({nx["worst_gain_pct"]:.1f} % less).')
        else:
            self.next_label.setText('No further candidate pose is feasible.' if not nx else '')
        c = doc.get('candidates') or {}
        rej = ', '.join(f'{k} {v}' for k, v in (c.get('rejected') or {}).items())
        self.cand_label.setText(f'{c.get("feasible", 0)} of {c.get("total", 0)} candidate poses are feasible'
                                + (f' (left out: {rej})' if rej else '') + '.')
        self.notes_label.setText('\n'.join(doc.get('notes') or []))
        self._preview_k = -1
        self.suggestion_ready(bool(doc.get('poses')))
        self._inputs = self.inputs_key()
        self._check_stale()
        self._rebuild_overlay()

    # ---------------------------------------------------------------- apply and preview
    def apply(self) -> str | None:
        """Save the plan as a calibration session (one undo step); the session's uid."""
        from ...core.calib import records as RC
        from ...core.project import new_uid
        doc = self.doc
        if not doc or not doc.get('poses'):
            return None
        p = self.ctx.store.project
        sensor = p.variant.sensor(doc['camera'].get('sensor_uid') or '')
        if sensor is None:
            self._msg('The planned camera is not in this rig any more: run the plan again.')
            return None
        b = doc['board']
        board_uid = b.get('uid') or ''
        new_board = None
        if not RC.find_board(p, board_uid):
            board_uid = new_uid()
            parent = 'world' if doc['mode'] == 'eye_in_hand' else doc['link']
            new_board = {'uid': board_uid, 'name': b.get('name') or 'Calibration board', 'board': b['definition'],
                         'mount': {'parent_frame': parent, 'robot_uid': '' if doc['mode'] == 'eye_in_hand' else
                                   doc['robot_uid'], 'pose': b['Z_nominal']}, 'default': bool(b.get('default'))}
        try:
            im = RC.sensor_imager(sensor)
        except ValueError as e:
            self._msg(f'{sensor.name} cannot be calibrated: {e}.')
            return None
        poses = [{'name': f'plan pose {ps["index"]:02d}', 'q': ps['q'], 'joint_names': ps['joint_names'],
                  'note': ps['what']} for ps in doc['poses']]
        sess = RC.session_record(sensor, im, doc['mode'], board_uid, doc['robot_uid'], doc['link'], poses,
                                 {'kind': 'plan', 'predicted_sigma': doc['sigma'], 'n_poses': len(poses)},
                                 name=f'{sensor.name} · plan of {len(poses)} poses')

        def fn(pr):
            if new_board is not None:
                pr.calibration.setdefault('boards', []).append(new_board)
            pr.calibration.setdefault('sessions', []).append(sess)
        self.ctx.store.edit('calibration', fn, f'Hand-eye · save the plan ({len(poses)} poses)')
        self._msg(f'Saved the plan as session "{sess["name"]}": record the poses, then import the images on the '
                  'Hand-eye solve page.', 8000)
        return sess['uid']

    def preview(self):
        """Move the robot (the view, not the project) to the next planned pose, in order."""
        doc = self.doc
        if not doc or not doc.get('poses'):
            return
        sel = self.pose_tree.selectedItems()
        if sel:
            k = self.pose_tree.indexOfTopLevelItem(sel[0])
            self.pose_tree.clearSelection()
        else:
            k = (self._preview_k + 1) % len(doc['poses'])
        self._preview_k = k
        ps = doc['poses'][k]
        try:
            self.ctx.store.set_q(doc['robot_uid'], np.asarray(ps['q'], float))
        except Exception:                               # noqa: BLE001 — a robot that changed meanwhile
            traceback.print_exc()
            self._msg('The planned pose does not fit the robot any more: run the plan again.')
            return
        self._msg(f'Pose {ps["index"]} of {len(doc["poses"])}: the camera {ps["what"]}.', 5000)
        self._rebuild_overlay()

    # ---------------------------------------------------------------- overlays
    def _rebuild_overlay(self):
        self._overlay = []
        doc = self.doc
        if doc and doc.get('poses') and self.show3d.isChecked():
            try:
                self._overlay = self._build_overlay(doc)
            except Exception:                           # noqa: BLE001 — the overlay must not take the page down
                traceback.print_exc()
                self._overlay = []
        self._redraw()

    def _build_overlay(self, doc) -> list:
        from ...core.calib.boards import board_from_dict
        from ..viewport3d import LineDraw
        out = []
        s = self.ctx.store.project.variant.sensor(doc['camera'].get('sensor_uid') or '')
        accent = THEME.cam(s.slot if s is not None else 0)
        muted = THEME.color('ink', 0.35)
        sel = self.pose_tree.selectedItems()
        cur = self.pose_tree.indexOfTopLevelItem(sel[0]) if sel else self._preview_k
        board = board_from_dict(dict(doc['board']['definition']))
        path = []
        for k, ps in enumerate(doc['poses']):
            C = _pose_matrix(ps['camera_world'])
            L = _pose_matrix(ps['link_world'])
            B = _pose_matrix(ps['board_world'])
            path.append(L[:3, 3])
            out += _triad(L, 0.03 if k == cur else 0.018)
            bc = _board_ring(board, B)[:4].mean(0)
            out.append(LineDraw(np.array([[C[:3, 3], bc]]), accent if k == cur else muted, 1.5 if k == cur else 1.0,
                                dash=k != cur, segments=True))
            if doc['mode'] == 'eye_to_hand' and k == cur:
                out.append(LineDraw(_board_ring(board, B), accent, 2.0))
        out.append(LineDraw(np.array(path), THEME.color('ink', 0.5), 1.0, dash=True))
        if doc['mode'] == 'eye_in_hand':
            out.append(LineDraw(_board_ring(board, _pose_matrix(doc['board']['Z_nominal'])), accent, 2.0))
        return out

    def overlays(self, vp) -> list:
        return list(self._overlay) if self.show3d.isChecked() else []

    # ---------------------------------------------------------------- explain
    def explain_text(self) -> str:
        from ...core.calib import planner as PL
        return (
            'Candidate poses put the camera on a hemisphere around the board, on its printed side: at three '
            'distances (the board filling about 80, 55 and 35 % of the image width), tilted 0, 20, 35 and 50° from '
            'the board\'s normal in eight directions, and rolled −90 to 90° about the optical axis (defaults). Each '
            'is solved by inverse kinematics; it is kept when the arm reaches it without touching itself, keeps '
            f'{1000 * PL.OBSTACLE_MARGIN_M:g} mm from the tables and objects (a default), and sees enough of the '
            'board: corners inside the image, squares at least 8 px wide, seen less than 60° off the board\'s normal '
            '(defaults), and not hidden by the robot.\n\n'
            'Every counted corner\'s two pixel coordinates carry independent noise σ (your corner noise); their '
            'derivatives with respect to the camera\'s pose X and the board\'s Z give each pose\'s Fisher '
            'information. The plan takes, one at a time, the pose that most increases log det of the information '
            '(the D-optimal criterion: the volume of the uncertainty ellipsoid shrinks the most); log det does not '
            'depend on units, so millimetres and degrees need no weighting. The predicted covariance is the inverse '
            'of the information, the covariance the bundle adjustment reports when its corners are that noisy.\n\n'
            'The headline is the camera\'s loosest 1σ against your requirement: good when within it, close up to '
            'twice it, too loose beyond, not determined when the poses do not pin an axis.\n\n'
            'Known limits: joint readings and the camera\'s intrinsics are taken as exact; corners as independent '
            '(detected corners are somewhat correlated, so the real σ is larger: treat the prediction as a lower '
            'bound); the camera body is not collision-checked; the motion between poses is not checked.')

    def explain(self):
        box = QMessageBox(self)
        box.setWindowTitle('How the hand-eye plan is made')
        box.setText(self.explain_text())
        box.setModal(False)
        box.show()


# =========================================================================== the solve page
class HandEyeSolvePage(_HandEyeBase):
    """Hand-eye solve: the bundle adjustment of a recorded session (board images matched to joint readings): the
    camera's pose in its mount with its covariance, each view's residual, and the coach's next pose."""
    STUDY = SOLVE
    TITLE = 'Hand-eye solve'
    HEADLINE = 'Calibrated camera: 1σ of its loosest axis'

    def build_body(self, layout):
        from ...core.calib import capture as CAP
        self._overlay = []
        self.folder = ''
        self.joints = ''
        layout.addWidget(label('Recording', bold=True))
        self.source = QComboBox()                       # kept for scripts: the Import button reads a folder
        for k in CAP.SOURCES:
            self.source.addItem(CAP.SOURCE_LABELS[k], k)
        self.source.setVisible(False)
        self.source_note = label('A folder of board images with a joint_states file, or a recording: a LeRobot '
                                 'dataset or a ROS 2 bag, whose frames with the board in view are taken with the '
                                 'joint states at their times.', muted=True, wrap=True)
        layout.addWidget(self.source_note)
        self.b_folder = button('Images folder…', role='ghost', tip='The folder of board images (one per pose)')
        self.b_joints = button('Joint states…', role='ghost', tip='CSV or JSON: a time column t (s) and one column per '
                                                                   'joint (rad)')
        self.folder_label = label('no folder', muted=True)
        self.joints_label = label('no file', muted=True)
        layout.addWidget(row(self.b_folder, self.folder_label, self.b_joints, self.joints_label))
        self.tol = ScrubSpinBox('', 'plain', 0, 'ms', step=5.0, width=90)
        self.tol.setRange(1, 5000)
        self.tol.set_value(1000 * CAP.TOLERANCE_S)
        self.tol.setToolTip('An image takes the joint sample nearest in time when it is this close (default 50 ms)')
        self.tol_label = label('default', muted=True, small=True)
        self.tol.committed.connect(lambda v: self.tol_label.setText('default' if abs(v - 1000 * CAP.TOLERANCE_S) < 1e-9
                                                                    else f'yours (default {1000 * CAP.TOLERANCE_S:g})'))
        self.unit = SegmentedControl([('rad', 'rad'), ('deg', 'deg')], 'rad')
        self.unit.setToolTip('The unit of the joint values in the file')
        self.b_import = button('Import', tip='Match the images to the joint states, detect the board and save the '
                                             'session (one undo step)')
        from ...core.calib import simcapture as SIM
        off_mm, off_deg = float(np.linalg.norm(SIM.DEFAULT_OFFSET_MM)), float(np.linalg.norm(SIM.DEFAULT_OFFSET_DEG))
        self.b_simulate = button('Simulate a capture…', role='ghost',
                                 tip='No rig yet? Render board images at the pose plan\'s poses with the camera moved '
                                     f'a known {off_mm:.1f} mm and {off_deg:.1f}° from its mount (defaults), save them '
                                     'with a joint_states '
                                     'file in a folder you choose and import them. A test of the pipeline, not a '
                                     'measurement: the solve should find that offset.')
        self.b_simulate.clicked.connect(lambda: self.simulate_capture())
        layout.addWidget(row(label('Match within'), self.tol, self.tol_label, label('Joints in'), self.unit,
                             self.b_import, self.b_simulate))
        self.b_dataset = button('From a LeRobot dataset or ROS 2 bag…', role='ghost', icon_name='import',
                                tip='Take the frames of the camera\'s stream with the board in view, with the joint '
                                    'states at their times (interpolated), from a LeRobot dataset (v2.1 or v3.0), a '
                                    'ROS 2 bag (MCAP or SQLite) or RLDS, as a session (one undo step)')
        self.b_dataset.clicked.connect(lambda: self.import_dataset(background=True))
        layout.addWidget(row(self.b_dataset))
        self.import_note = label('', muted=True, wrap=True)
        layout.addWidget(self.import_note)
        self.b_folder.clicked.connect(self._choose_folder)
        self.b_joints.clicked.connect(self._choose_joints)
        self.b_import.clicked.connect(lambda: self.import_capture(background=True))

        layout.addWidget(label('Solve', bold=True))
        self.session_box = QComboBox()
        self.session_box.setToolTip('The capture session to solve')
        self.session_box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.session_box.setMinimumWidth(220)
        self.offsets = QCheckBox('Estimate joint zero offsets')
        self.offsets.setToolTip('Also estimate the encoder zero offsets of the joints that move the mount (prior 1°, a '
                                'default); offsets the views cannot tell from X or Z are marked not determined')
        self.intr = QCheckBox('Refine the intrinsics')
        self.intr.setToolTip('Also refine fx, fy, cx, cy and the distortion terms (give 20 or more views across the '
                             'image)')
        self.together = QCheckBox('Every session together')
        self.together.setToolTip('Solve every session with detections in one bundle adjustment: a camera or board '
                                 'shared by sessions is estimated once, so a fixed camera that sees the same board as '
                                 'a wrist camera is calibrated through it; Apply writes every camera\'s mount')
        self.together.toggled.connect(lambda on: self.session_box.setEnabled(not on))
        layout.addWidget(row(label('Session'), self.session_box, self.together))
        layout.addWidget(row(self.offsets, self.intr))
        self.stale_note = label('', muted=True, wrap=True)
        self._inputs = None
        layout.addWidget(self.stale_note)
        layout.addWidget(label('Covariance', bold=True))
        self.sig_tree = _tree(['Transform', 'σ x', 'σ y', 'σ z', 'σ rx', 'σ ry', 'σ rz'], stretch=0)
        layout.addWidget(self.sig_tree)
        layout.addWidget(label('Residuals per view', bold=True))
        self.view_tree = _tree(['View', 'Image', 'Corners', 'RMS', 'Used', 'Note'])
        layout.addWidget(self.view_tree)
        layout.addWidget(label('Coach: the next pose', bold=True))
        self.coach_label = label('Run the solve: the coach then says which pose to take next and why.', wrap=True)
        layout.addWidget(self.coach_label)
        self.notes_label = label('', muted=True, wrap=True)
        layout.addWidget(self.notes_label)
        self.b_apply.setToolTip('Write the calibrated mount pose (and lens calibration, when refined) to the sensor '
                                '(one undo step)')
        self.b_preview.setText('Show next pose')
        self.b_preview.setToolTip('Move the robot in the 3D view to the pose the coach advises (view only)')
        st = self.ctx.store
        st.changed.connect(lambda _s: (self._sync(), self._check_stale()))
        st.project_replaced.connect(self._replaced)
        self._sync()

    # ---------------------------------------------------------------- sessions
    def _sync(self):
        from ...core.calib import records as RC
        cur = self.session_box.currentData()
        self.session_box.blockSignals(True)
        self.session_box.clear()
        for s in RC.calibration_of(self.ctx.store.project).get('sessions') or []:
            n = sum(1 for p in s.get('poses') or [] if p.get('ids'))
            tag = 'plan, no images' if s.get('source', {}).get('kind') == 'plan' else f'{n} views'
            if (s.get('result') or {}).get('applied'):
                tag += ', applied'
            self.session_box.addItem(f'{s.get("name", "session")} ({tag})', s.get('uid'))
        k = self.session_box.findData(cur)
        self.session_box.setCurrentIndex(k if k >= 0 else self.session_box.count() - 1)
        self.session_box.blockSignals(False)

    def _replaced(self):
        self.doc, self._overlay, self._inputs = None, [], None
        self.stale_note.setText('')
        self.sig_tree.clear()
        self.view_tree.clear()
        self.coach_label.setText('Run the solve: the coach then says which pose to take next and why.')
        self.notes_label.setText('')
        self.suggestion_ready(False)
        self._sync()
        self.empty()

    def _choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, 'Images folder')
        if d:
            self.folder = d
            self.folder_label.setText(os.path.basename(os.path.normpath(d)))

    def _choose_joints(self):
        f, _ = QFileDialog.getOpenFileName(self, 'Joint states', '', 'Joint states (*.csv *.json);;All files (*)')
        if f:
            self.joints = f
            self.joints_label.setText(os.path.basename(f))

    def import_capture(self, folder: str | None = None, joints: str | None = None, sensor_uid: str | None = None,
                       board_uid: str | None = None, background: bool = False) -> str | None:
        """Capture a folder and a joint_states file as a session (one undo step); its uid, or None (said why).
        background: detect the board on a worker thread (the Import button: detection takes seconds) and add the
        session when it is done; the call then returns None at once."""
        from ...core.calib import capture as CAP
        from ...core.calib import records as RC
        folder = folder or self.folder
        joints = joints or self.joints
        try:
            kind = CAP.check_source(self.source.currentData() or 'folder')
        except CAP.CaptureError as e:
            self.import_note.setText(str(e))
            return None
        if kind == 'dataset':
            self.import_note.setText('A recording is read with "From a LeRobot dataset or ROS 2 bag…"; Import reads a '
                                     'folder of board images and a joint states file.')
            return None
        if not folder or not joints:
            self.import_note.setText('Choose the images folder and the joint states file first.')
            return None
        p = self.ctx.store.project
        plan_st = p.analysis.studies.get(PLAN) or {}
        sensor = p.variant.sensor(sensor_uid or plan_st.get('sensor') or '')
        if sensor is None:
            sensor = next((s for s in p.variant.sensors if s.kind in ('camera', 'depth') and s.mount.on_link), None) \
                or next((s for s in p.variant.sensors if s.kind in ('camera', 'depth')), None)
        if sensor is None:
            self.import_note.setText('The rig has no camera to calibrate: add one (Sensors ▸ Add camera).')
            return None
        rec = RC.find_board(p, board_uid or plan_st.get('board') or None)
        tol, unit = self.tol.value() / 1000.0, self.unit.value() or 'rad'
        win = self.ctx.window
        if background and win is not None and hasattr(win, 'run_job'):
            snap = p.snapshot()
            snap_sensor = snap.variant.sensor(sensor.uid) or sensor

            def work(_progress, _cancel):
                try:
                    return CAP.capture_folder(snap, snap_sensor, folder, joints, rec, tol, unit)
                except (CAP.CaptureError, ValueError) as e:      # said in the page, as the synchronous path does
                    return {'error': str(e)}

            def done(sess):
                if 'error' in sess:
                    self.import_note.setText(sess['error'])
                else:
                    self._add_session(sess)

            def failed(_message):                       # the window printed the traceback
                self.import_note.setText('The capture could not be read (an unexpected error; the details are in the '
                                         'log).')
            if win.run_job('handeye_import', 'Detecting the board in the images', work, done, failed):
                self.import_note.setText('Detecting the board in the images…')
            return None
        try:
            sess = CAP.capture_folder(p, sensor, folder, joints, rec, tol, unit)
        except (CAP.CaptureError, ValueError) as e:
            self.import_note.setText(str(e))
            return None
        except Exception:                               # noqa: BLE001 — explained, the traceback logged
            traceback.print_exc()
            self.import_note.setText('The capture could not be read (an unexpected error; the details are in the log).')
            return None
        return self._add_session(sess)

    def _dataset_sensor(self, sensor_uid: str | None):
        p = self.ctx.store.project
        plan_st = p.analysis.studies.get(PLAN) or {}
        sensor = p.variant.sensor(sensor_uid or plan_st.get('sensor') or '')
        if sensor is None and not sensor_uid:
            sensor = next((s for s in p.variant.sensors if s.kind in ('camera', 'depth') and s.mount.on_link),
                          None) or next((s for s in p.variant.sensors if s.kind in ('camera', 'depth')), None)
        return sensor

    def import_dataset(self, path: str | None = None, sensor_uid: str | None = None, camera: str | None = None,
                       episodes=None, every_n: int = 1, unit: str | None = None, offset_s: float = 0.0,
                       still_only: bool = True, board_uid: str | None = None,
                       background: bool = False) -> str | None:
        """Capture a session from a recorded dataset (core/calib/capture.capture_dataset: a LeRobot dataset, a ROS 2
        bag or RLDS; one undo step); its uid, or None (said why). Without a path the dialog asks. background: read on
        a worker thread (the button) and add the session when done; the call then returns None at once."""
        from ...core.calib import capture as CAP
        from ...core.calib import records as RC
        p = self.ctx.store.project
        if path is None:
            from ..dialogs.dataset_source import DatasetSourceDialog
            cams = [(s.name, s.uid) for s in p.variant.sensors if s.kind in ('camera', 'depth')]
            if not cams:
                self.import_note.setText('The rig has no camera to calibrate: add one (Sensors ▸ Add camera).')
                return None
            cur = self._dataset_sensor(None)
            d = DatasetSourceDialog(self, 'handeye', cams, cur.uid if cur is not None else '')
            if not d.exec():
                return None
            try:
                v = d.values()
            except ValueError as e:
                self.import_note.setText(str(e))
                return None
            path, sensor_uid, camera, episodes = v['path'], v['sensor_uid'], v['camera'], v['episodes']
            every_n, unit, offset_s, still_only = v['every'], v['unit'], v['offset_s'], v['still_only']
        sensor = self._dataset_sensor(sensor_uid)
        if sensor is None:
            self.import_note.setText('The rig has no such camera to calibrate: choose one of its cameras (or add one: '
                                     'Sensors ▸ Add camera).')
            return None
        plan_st = p.analysis.studies.get(PLAN) or {}
        unit = unit or self.unit.value() or 'rad'
        name = os.path.basename(os.path.normpath(path))
        snap = p.snapshot() if background else p
        rec = RC.find_board(snap, board_uid or plan_st.get('board') or None)
        suid = sensor.uid

        def work(progress=None, cancel=None):
            try:
                return CAP.capture_dataset(snap, path, suid, episodes, every_n, camera=camera, board_rec=rec,
                                           unit=unit, offset_s=offset_s, still_only=still_only, progress=progress,
                                           cancel=cancel)
            except (CAP.CaptureError, ValueError) as e:
                return {'error': str(e)}

        def done(sess):
            if 'error' in sess:
                self.import_note.setText(f'No capture from {name}: {sess["error"]}')
                return None
            uid = self._add_session(sess)
            src = sess['source']
            self.import_note.setText(
                f'{src["images"]} frames of {src["camera"]} looked at in {len(src["episodes"])} episode(s) of {name}, '
                f'the board found in {src["board_frames"]}: {src["detected"]} views. ' + ' '.join(src['notes']))
            return uid

        def failed(_message):                           # the window logged the traceback
            self.import_note.setText(f'{name} could not be read (an unexpected error; the details are in the log).')
        win = self.ctx.window
        if background and win is not None and hasattr(win, 'run_job'):
            if win.run_job('handeye_dataset', f'Looking for the board in {name}', work, done, failed):
                self.import_note.setText(f'Looking for the board in the frames of {name}…')
            else:
                self.import_note.setText('A dataset is already being read: wait for it to finish.')
            return None
        try:
            res = work()
        except Exception:                               # noqa: BLE001 — explained, the traceback logged
            traceback.print_exc()
            failed('')
            return None
        return done(res)

    def simulate_capture(self, folder: str | None = None, background: bool | None = None) -> str | None:
        """Render a simulated capture at the pose plan's poses (core/calib/simcapture.py) into a new folder inside
        folder, import it as a session named "(simulated)" (one undo step); its uid, or None (said why). The button
        asks for the folder and works on a worker thread; a script passing folder gets the uid back."""
        import time as _time

        from ...core.calib import capture as CAP
        from ...core.calib import records as RC
        from ...core.calib import simcapture as SIM
        p = self.ctx.store.project
        plan = self.ctx.analysis.results.get(PLAN) or {}
        poses = plan.get('poses') or []
        sensor_uid = (plan.get('camera') or {}).get('sensor_uid') or ''
        board_uid = (plan.get('board') or {}).get('uid') or None
        if not poses:                                   # a saved plan session
            sess = next((x for x in reversed(RC.calibration_of(p).get('sessions') or [])
                         if (x.get('source') or {}).get('kind') == 'plan' and x.get('poses')), None)
            if sess is not None:
                poses, sensor_uid = sess['poses'], sess.get('sensor_uid') or ''
                board_uid = sess.get('board_uid') or None
        if board_uid and not RC.find_board(p, board_uid):
            board_uid = None                            # the plan's default board placement, not a project record
        if not poses:
            self.import_note.setText('There is no pose plan to simulate: run the Hand-eye pose plan first (Calibrate ▸ '
                                     'Hand-eye pose plan).')
            return None
        sensor = p.variant.sensor(sensor_uid) if sensor_uid else None
        if sensor is None:
            self.import_note.setText('The planned camera is not in this rig any more: run the Hand-eye pose plan '
                                     'again.')
            return None
        if background is None:
            background = folder is None
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Folder for the simulated capture')
            if not folder:
                return None
        if not os.path.isdir(folder) or not os.access(folder, os.W_OK):
            self.import_note.setText('Choose a folder that exists and that you can write to.')
            return None
        out = os.path.join(folder, 'simulated-capture-' + _time.strftime('%Y%m%d-%H%M%S'))
        k = 1
        while os.path.exists(out + (f'-{k}' if k > 1 else '')):
            k += 1
        out += f'-{k}' if k > 1 else ''
        snap = p.snapshot()
        snap_sensor = snap.variant.sensor(sensor.uid) or sensor
        rec = RC.find_board(snap, board_uid)
        tol = self.tol.value() / 1000.0

        def work(progress=None, cancel=None):
            try:
                sim = SIM.simulate(snap, poses, out, sensor.uid, board_uid, progress=(
                    (lambda f, t='': progress(0.6 * f, t)) if progress else None), cancel=cancel)
                sess = CAP.capture_folder(snap, snap_sensor, sim['images'], sim['joints'], rec, tol, 'rad',
                                          progress=(lambda f, t='': progress(0.6 + 0.4 * f, t)) if progress else None,
                                          cancel=cancel)
            except (SIM.SimError, CAP.CaptureError, ValueError) as e:
                return {'error': str(e)}
            tr = sim['truth']
            sess['source']['simulated'] = {k_: tr[k_] for k_ in ('note', 'offset_mm', 'offset_deg', 'offset_norm_mm',
                                                                 'offset_angle_deg', 'frame', 'noise_grey', 'X_true')}
            sess['name'] = f'{sensor.name} · {sim["n"]} poses (simulated)'
            return {'session': sess, 'truth': tr, 'folder': os.path.basename(out)}

        def done(res):
            if 'error' in res:
                self.import_note.setText(res['error'])
                return
            uid = self._add_session(res['session'])
            tr = res['truth']
            self.import_note.setText(
                f'{self.import_note.text()} Simulated in {res["folder"]}: the camera was moved '
                f'{tr["offset_norm_mm"]:.2f} mm and {tr["offset_angle_deg"]:.2f}° from its mount (defaults; a test, '
                'not a measurement). Run the '
                'solve: it should find that offset.')
            return uid

        def failed(_message):                           # the window logged the traceback
            self.import_note.setText('The simulated capture could not be made (an unexpected error; the details are '
                                     'in the log).')
        win = self.ctx.window
        if background and win is not None and hasattr(win, 'run_job'):
            if win.run_job('handeye_simulate', 'Rendering a simulated capture', work, done, failed):
                self.import_note.setText('Rendering a simulated capture…')
            return None
        try:
            res = work()
        except Exception:                               # noqa: BLE001 — explained, the traceback logged
            traceback.print_exc()
            failed('')
            return None
        return done(res)

    def _add_session(self, sess: dict) -> str:
        """Add a captured session (and the board it used, when new) as one undo step; its uid."""
        from ...core.calib import records as RC
        board = sess.pop('board', None)

        def fn(pr):
            if board is not None and not RC.find_board(pr, board.get('uid')):
                pr.calibration.setdefault('boards', []).append({k: v for k, v in board.items() if k != 'where'})
            pr.calibration.setdefault('sessions', []).append(sess)
        self.ctx.store.edit('calibration', fn, f'Hand-eye · import {sess["source"]["folder"]}')
        src = sess['source']
        self.import_note.setText(f'{src["images"]} images, {src["matched"]} matched to a joint reading, the board '
                                 f'found in {src["detected"]}. ' + ' '.join(src.get('notes') or []))
        k = self.session_box.findData(sess['uid'])
        if k >= 0:
            self.session_box.setCurrentIndex(k)
        return sess['uid']

    # ---------------------------------------------------------------- run and result
    def run(self):
        st = {'joint_offsets': self.offsets.isChecked(), 'intrinsics': self.intr.isChecked()}
        if self.together.isChecked():
            st['sessions'] = 'all'
        else:
            st['session'] = self.session_box.currentData() or ''
        self.ctx.analysis.run(self.STUDY, st)

    def show_result(self, doc):
        if not doc.get('ok'):
            self.head.set('—', '', 'critical', doc.get('word', 'not solved'),
                          doc.get('advice') or doc.get('message', ''))
            self.suggestion_ready(False)
            return
        fig, unit = _headline_figure(doc)
        req = doc.get('requirement') or {}
        tip = (f'Requirement {req.get("need_mm", 0):g} mm and {req.get("need_deg", 0):g}° (1σ; '
               + ('the defaults' if req.get('default') else 'yours') + '; set on the Hand-eye pose plan page). '
               + doc.get('message', ''))
        self.head.set(fig, unit, doc.get('status', 'warning'), doc.get('word', ''), doc.get('advice', ''), tip)
        if doc.get('cameras'):
            _sigma_rows(self.sig_tree, [(f'X: {c["name"]}', c['frame'], c['sigma']) for c in doc['cameras']]
                        + [(f'Z: {b["name"]}', b['frame'], b['sigma']) for b in doc.get('boards') or []])
        else:
            _sigma_rows(self.sig_tree, [(f'X: {doc["camera"]["name"]}', doc['camera']['frame'],
                                         doc['camera']['sigma']),
                                        (f'Z: {doc["board"]["name"]}', doc['board']['frame'], doc['board']['sigma'])])
        self.view_tree.clear()
        for v in doc.get('views') or []:
            rms = v.get('rms_px')
            it = _item([v.get('name') or '', v.get('image') or '', v.get('corners', 0),
                        '—' if rms is None else f'{rms:.3f} px', 'yes' if v.get('used') else 'no', v.get('note') or ''],
                       mono_from=2)
            it.setTextAlignment(5, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            if v.get('dt_ms') is not None:
                it.setToolTip(1, f'Joint sample {v["dt_ms"]:+.1f} ms from the image')
            self.view_tree.addTopLevelItem(it)
        _fit(self.view_tree)
        co = doc.get('coach') or {}
        self.coach_label.setText(co.get('advice') or 'The coach did not run.')
        self.notes_label.setText('\n'.join(doc.get('notes') or []))
        self.b_apply.setEnabled(bool(doc.get('apply')) and bool((doc.get('session') or {}).get('stored')))
        self.b_preview.setEnabled(bool(co.get('next')))
        self._inputs = self.inputs_key()
        self._check_stale()
        self._rebuild_overlay()

    def apply(self) -> bool:
        """Write the calibrated mount (and lens calibration) to the sensor and mark the session applied, one undo
        step."""
        from ...core.calib import records as RC
        from ...core.frames import Pose
        doc = self.doc
        ap = (doc or {}).get('apply')
        if not ap or not (doc.get('session') or {}).get('stored'):
            return False
        aps = list(doc.get('applies') or [ap])
        p = self.ctx.store.project
        sens = [p.variant.sensor(a['sensor_uid']) for a in aps]
        if any(s is None for s in sens):
            self._msg('A calibrated camera is not in this rig any more.')
            return False
        locked = [s.name for s in sens if s.locked]
        if locked:
            self._msg(f'{", ".join(locked)} {"is" if len(locked) == 1 else "are"} locked: unlock on the Sensors and '
                      'mounts page to apply the calibration.')
            return False
        suids = set(doc['session'].get('uids') or [doc['session']['uid']])
        boards = [b for b in doc.get('boards') or [] if b.get('stored')] if doc.get('boards') else \
            ([{'uid': ap['board_uid'], 'board_pose': ap['board_pose']}] if ap.get('board_uid') else [])

        def fn(pr):
            for a in aps:
                sen = pr.variant.sensor(a['sensor_uid'])
                sen.mount.pose = Pose.from_dict(a['pose'])
                sen.mount.parent_frame = a['parent_frame']
                sen.mount.robot_uid = a['robot_uid']
                if a.get('calibration'):
                    cal = dict(sen.calibration or {})
                    per = dict(cal.get('imagers') or {})
                    per[a['calibration']['imager']] = a['calibration']['record']
                    cal['imagers'] = per
                    sen.calibration = cal
            for ss in pr.calibration.get('sessions') or []:
                if ss.get('uid') in suids:
                    ss['result'] = dict(doc['result'], applied=True, applied_at=RC.now_utc())
            for bd in boards:
                for b in pr.calibration.get('boards') or []:
                    if b.get('uid') == bd['uid']:
                        b['mount'] = dict(b.get('mount') or {}, pose=bd['board_pose']['pose'])
                        b['default'] = False
        names = ', '.join(s.name for s in sens)
        self.ctx.store.edit(['variant', 'calibration'], fn, f'Hand-eye · apply the calibration of {names}')
        if len(aps) == 1:
            self._msg(f'Applied: {sens[0].name} moved {ap["change_mm"]:.2f} mm and {ap["change_deg"]:.3f}° to its '
                      'calibrated mount.', 8000)
        else:
            self._msg(f'Applied the calibrated mounts of {names}.', 8000)
        return True

    def preview(self):
        co = (self.doc or {}).get('coach') or {}
        nx = co.get('next')
        if not nx:
            return
        uid = None
        sess = (self.doc or {}).get('session') or {}
        from ...core.calib import records as RC
        ss = RC.find_session(self.ctx.store.project, sess.get('uid'))
        uid = co.get('robot_uid') or (ss or {}).get('robot_uid')
        if not uid:
            return
        try:
            self.ctx.store.set_q(uid, np.asarray(nx['q'], float))
        except Exception:                               # noqa: BLE001
            traceback.print_exc()
            return
        self._msg(f'The next pose: the camera {nx.get("what", "")}.', 5000)
        self._rebuild_overlay()

    # ---------------------------------------------------------------- overlays
    def _board_now(self):
        doc = self.doc or {}
        if not doc.get('ok'):
            return None
        from ...core.calib import records as RC
        p = self.ctx.store.project
        b = doc.get('board') or {}
        rec = RC.find_board(p, b.get('uid') or None)
        if rec is None:
            return None
        board = RC.board_of(rec)
        Z = _pose_matrix(doc['apply']['board_pose']['pose'])
        ss = RC.find_session(p, (doc.get('session') or {}).get('uid'))
        if (doc.get('session') or {}).get('mode') == 'eye_to_hand' and ss:
            from ...core.robot.model import get_model
            m = get_model(p, ss['robot_uid'])
            return board, m.link_pose(self.ctx.store.q.get(ss['robot_uid'], m.home), ss['link']) @ Z
        return board, Z

    def _rebuild_overlay(self):
        from ..viewport3d import LineDraw
        self._overlay = []
        try:
            got = self._board_now()
            if got is not None:
                board, T = got
                self._overlay.append(LineDraw(_board_ring(board, T), THEME.color('ink', 0.8), 2.0))
            nx = ((self.doc or {}).get('coach') or {}).get('next')
            if nx:
                self._overlay += _triad(_pose_matrix(nx['link_world']), 0.03)
                C = _pose_matrix(nx['camera_world'])
                if got is not None:
                    bc = _board_ring(*got)[:4].mean(0)
                    self._overlay.append(LineDraw(np.array([[C[:3, 3], bc]]), THEME.color('ink', 0.5), 1.0, dash=True,
                                                  segments=True))
        except Exception:                               # noqa: BLE001 — the overlay must not take the page down
            traceback.print_exc()
            self._overlay = []
        self._redraw()

    def overlays(self, vp) -> list:
        return list(self._overlay)

    def explain_text(self) -> str:
        return (
            'Each image is paired with the joint sample nearest in time (within the tolerance), the board is detected '
            '(ChArUco or AprilGrid corners, sub-pixel), and one bundle adjustment over every corner of every view '
            'estimates the camera\'s pose X in its mount (the wrist link eye-in-hand, the world eye-to-hand) and the '
            'board\'s Z, optionally the joint zero offsets and the intrinsics. It starts from the best of the '
            'closed-form solvers (Tsai-Lenz, Park-Martin, Horaud-Dornaika, Andreff, Daniilidis, Shah, Li). Views whose '
            'RMS is beyond 3 × the median (at least 1 px; defaults) are dropped and the problem solved again.\n\n'
            'σ is the Gauss-Newton covariance scaled by the residuals\' own σ̂: x, y, z in mm along the parent frame, '
            'rx, ry, rz in degrees about the frame\'s own axes. Corners detected in real images are somewhat '
            'correlated, so treat σ as a lower bound and look at the per-view RMS.\n\n'
            'The coach scores every feasible candidate pose (as the plan does) by what it would add to the views '
            'taken and names the one that most reduces the camera\'s loosest σ, and why.\n\n'
            'Apply writes the sensor\'s mount pose (its body frame, from the calibrated optical frame) and, when the '
            'intrinsics were refined, its lens calibration; one undo step.')

    def explain(self):
        box = QMessageBox(self)
        box.setWindowTitle('How the hand-eye solve works')
        box.setText(self.explain_text())
        box.setModal(False)
        box.show()


register('handeye_plan', 'Hand-eye pose plan', HandEyePlanPage, menu='Calibrate', icon='checker',
         action_id='handeye_plan', workspace='calibration', order=60)
register('handeye_solve', 'Hand-eye solve', HandEyeSolvePage, menu='Calibrate', icon='target',
         action_id='handeye_solve', workspace='calibration', order=61)
