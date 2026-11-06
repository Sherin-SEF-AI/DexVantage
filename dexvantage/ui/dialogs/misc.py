"""Smaller dialogs: confirm, ask for text, keyboard shortcuts, coordinate frames, about.

confirm, ask_text and ShortcutsDialog are ported from Vantage ui/dialogs/misc.py; the frames diagram is DexVantage's.
"""
from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QKeySequence, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (QAbstractItemView, QHBoxLayout, QHeaderView, QLineEdit, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

from .. import icons
from ..navigation import AXIS_COLORS
from ..theme import THEME, mono, sans
from ..widgets import ROW_H, button, label
from .base import VDialog


def confirm(parent, title, text, ok_text='OK', destructive=True) -> bool:
    d = VDialog(parent, title, '', (460, 190), resizable=False)
    lay = QVBoxLayout(d.body)
    lay.setContentsMargins(14, 12, 14, 12)
    lay.addWidget(label(text, wrap=True))
    lay.addStretch(1)
    c = button('Cancel', 'ghost')
    c.clicked.connect(d.reject)
    o = button(ok_text, 'destructive' if destructive else 'primary', 'critical' if destructive else None)
    o.clicked.connect(d.accept)
    d.buttons.addWidget(c)
    d.buttons.addWidget(o)
    return bool(d.exec())


def ask_text(parent, title, lab, default='') -> str | None:
    d = VDialog(parent, title, '', (420, 160), resizable=False)
    lay = QHBoxLayout(d.body)
    lay.setContentsMargins(14, 12, 14, 12)
    lay.addWidget(label(lab, ink2=True))
    e = QLineEdit(default)
    e.selectAll()
    lay.addWidget(e, 1)
    c = button('Cancel', 'ghost')
    c.clicked.connect(d.reject)
    o = button('OK', 'primary')
    o.clicked.connect(d.accept)
    e.returnPressed.connect(d.accept)
    d.buttons.addWidget(c)
    d.buttons.addWidget(o)
    return e.text().strip() if d.exec() else None


VIEW_SHORTCUTS = [
    ('Orbit (about the point under the cursor)', 'Viewport',
     'Middle-drag or left-drag on empty space · touchpad: two fingers · ← → ↑ ↓'),
    ('Pan', 'Viewport', 'Shift + middle-drag, right-drag · touchpad: Shift + two fingers · Shift + arrows'),
    ('Zoom (towards the cursor)', 'Viewport', 'Wheel · pinch or Ctrl + two fingers · + / −'),
    ('Navigation control', 'Viewport', 'Top right: drag the ball, click an axis, drag the hand or magnifier'),
    ('Top / front / back / side view', 'Viewport', '7 / 1 / Ctrl+1 / 3 (number row or numpad)'),
    ('Toggle orthographic', 'Viewport', '5'),
    ('Scrub a number', 'Inspector', 'Drag the axis letter · Shift fine · Ctrl coarse'),
    ('Step a number', 'Inspector', '↑ / ↓'),
]


class ShortcutsDialog(VDialog):
    def __init__(self, window, parent=None):
        super().__init__(parent or window, 'Keyboard shortcuts', 'searchable', (720, 600))
        lay = QVBoxLayout(self.body)
        lay.setContentsMargins(14, 10, 14, 10)
        self.q = QLineEdit()
        self.q.setPlaceholderText('Filter')
        self.q.addAction(icons.icon('search', 12), QLineEdit.ActionPosition.LeadingPosition)
        lay.addWidget(self.q)
        self.t = QTableWidget(0, 3)
        self.t.setHorizontalHeaderLabels(['Command', 'Menu', 'Shortcut'])
        self.t.verticalHeader().setVisible(False)
        self.t.verticalHeader().setDefaultSectionSize(ROW_H)
        self.t.setShowGrid(False)
        self.t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hh = self.t.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        lay.addWidget(self.t)
        rows = []
        for a in window.actions_by_id.values():
            sc = a.shortcut().toString(QKeySequence.SequenceFormat.NativeText)
            if sc:
                rows.append((a.text().replace('&', ''), (a.data() or {}).get('category', ''), sc))
        self.rows = sorted(rows + VIEW_SHORTCUTS, key=lambda r: (r[1], r[0]))
        self.q.textChanged.connect(self.fill)
        self.fill()
        o = button('Close', 'primary')
        o.clicked.connect(self.accept)
        self.buttons.addWidget(o)

    def fill(self):
        q = self.q.text().lower()
        rows = [r for r in self.rows if not q or q in ' '.join(r).lower()]
        self.t.setRowCount(len(rows))
        for i, (a, b, c) in enumerate(rows):
            self.t.setItem(i, 0, QTableWidgetItem(a))
            it = QTableWidgetItem(b)
            it.setForeground(THEME.color('muted'))
            self.t.setItem(i, 1, it)
            s = QTableWidgetItem(c)
            s.setFont(mono(12))
            self.t.setItem(i, 2, s)


class FramesDiagram(QWidget):
    """Two axis triads in an oblique projection: the world frame (REP-103 body: x forward, y left, z up) and a
    camera's optical frame (z forward, x right, y down), with the camera body frame it hangs from."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(560, 260)

    @staticmethod
    def _proj(v):
        # oblique view: world x towards the lower left, y to the right, z up
        x, y, z = v
        return QPointF(y - 0.55 * x, -z + 0.32 * x)

    def _triad(self, p, origin, axes, names, title, sub, L=70):
        o = QPointF(*origin)
        for k, (v, name) in enumerate(zip(axes, names)):
            d = self._proj(v)
            end = o + d * L
            col = QColor(AXIS_COLORS[k])
            p.setPen(QPen(col, 2.0, cap=Qt.PenCapStyle.RoundCap))
            p.drawLine(o, end)
            n = math.hypot(d.x(), d.y()) or 1.0
            ux, uy = d.x() / n, d.y() / n
            tip = QPolygonF([end + QPointF(ux * 8, uy * 8), end + QPointF(-uy * 4, ux * 4),
                             end + QPointF(uy * 4, -ux * 4)])
            p.setBrush(col)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPolygon(tip)
            p.setPen(THEME.color('ink'))
            p.setFont(mono(11))
            p.drawText(end + QPointF(ux * 18 - 14, uy * 18 + 4), name)
        p.setFont(sans(12, 600))
        p.setPen(THEME.color('ink'))
        p.drawText(QRectF(origin[0] - 120, origin[1] + 70, 240, 18), Qt.AlignmentFlag.AlignHCenter, title)
        p.setFont(sans(11))
        p.setPen(THEME.color('ink2'))
        p.drawText(QRectF(origin[0] - 130, origin[1] + 88, 260, 34),
                   Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap, sub)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), THEME.color('raised'))
        w = self.width()
        self._triad(p, (w * 0.25, 120), [(1, 0, 0), (0, 1, 0), (0, 0, 1)], ['x fwd', 'y left', 'z up'],
                    'World and links (REP-103)', 'The primary robot\'s base frame. Joint angles in radians, shown '
                                                 'in degrees.')
        # optical: x right = body −y, y down = body −z, z forward = body x
        self._triad(p, (w * 0.75, 120), [(0, -1, 0), (0, 0, -1), (1, 0, 0)], ['x right', 'y down', 'z fwd'],
                    'Camera optical frame', '_optical_frame per REP-103: z along the optical axis, x right, y down.')
        p.end()


class FramesDialog(VDialog):
    def __init__(self, parent=None):
        super().__init__(parent, 'Coordinate frames', 'REP-103', (640, 470), resizable=False)
        lay = QVBoxLayout(self.body)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)
        lay.addWidget(FramesDiagram())
        lay.addWidget(label('Units are SI: metres, seconds, kilograms. Poses are stored as a parent frame (a link or a '
                            'static frame such as world) plus a position and a quaternion (x, y, z, w). URDF rpy is '
                            'roll about fixed x, then pitch about fixed y, then yaw about fixed z. MuJoCo writes '
                            'quaternions as (w, x, y, z); DexVantage converts at the boundary.', wrap=True))
        lay.addStretch(1)
        o = button('Close', 'primary')
        o.clicked.connect(self.accept)
        self.buttons.addWidget(o)


class AboutDialog(VDialog):
    def __init__(self, parent=None):
        super().__init__(parent, 'About DexVantage', '', (520, 300), resizable=False)
        lay = QVBoxLayout(self.body)
        lay.setContentsMargins(14, 12, 14, 12)
        from ... import __version__
        lay.addWidget(label(f'DexVantage {__version__} · camera, depth and teleop planning for robot-learning data '
                            'collection', bold=True, wrap=True))
        lay.addWidget(label('Plans where the wrist, head, chest, overhead and third-person cameras go so the '
                            'fingertips, the object and the contact are visible at the grasp; checks depth validity, '
                            'teleop reach, cables, bandwidth and sync; audits recorded datasets against the plan.',
                            wrap=True))
        lay.addWidget(label('Free software under the GNU General Public License, version 3 or (at your option) any '
                            'later version. It comes with no warranty. Source: github.com/Sherin-SEF-AI/DexVantage',
                            muted=True, small=True, wrap=True))
        lay.addWidget(label('Typeface: IBM Plex Sans and Plex Mono, SIL Open Font License 1.1.', muted=True,
                            small=True, wrap=True))
        lay.addWidget(label('Built with PyQt6, NumPy, SciPy, OpenCV and MuJoCo.', muted=True, small=True))
        lay.addStretch(1)
        o = button('Close', 'primary')
        o.clicked.connect(self.accept)
        self.buttons.addWidget(o)
