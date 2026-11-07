"""A small x-y chart for the analysis pages: lines and points, shaded x bands, reference lines and labelled markers,
with a hover tooltip that names the nearest point (value first, then what it is). The chart conventions of ui/charts.py
(hairline grid, mono tick labels, round ticks). A module whose name starts with "_" is not a page (pages.load_pages).
"""
from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF

from ..charts import ChartBase, nice_ticks
from ..theme import THEME, mono, sans


def _ticks(lo, hi, n=5):
    span = hi - lo
    if not (span > 0) or not math.isfinite(span):
        return [lo]
    t = nice_ticks(span, n)
    step = t[1] - t[0] if len(t) > 1 else span
    k0 = math.ceil(lo / step - 1e-9)
    out = []
    v = k0 * step
    while v <= hi + 1e-9 * step:
        out.append(round(v, 10))
        v += step
    return out


def _fmt(v):
    return f'{round(v, 6):g}'


class CurveChart(ChartBase):
    """series: [(label, xs, ys, colour, style 'line' | 'points' | 'dash')], None in ys breaks a line; bands: [(x0, x1,
    colour, label)]; hlines / vlines: [(value, label)]; markers: [(x, y, colour, label)]. reverse_x draws x from right
    to left (distance to contact falling towards the right)."""

    def __init__(self, parent=None, height=220):
        super().__init__(parent)
        self.series, self.bands, self.hlines, self.vlines, self.markers = [], [], [], [], []
        self.xlabel = self.ylabel = ''
        self.xname, self.xunit, self.yunit = 'at', '', ''         # the hover text: '<y> <yunit>', '<xname> <x> <xunit>'
        self.xr = (0.0, 1.0)
        self.yr = (0.0, 1.0)
        self.reverse_x = False
        self.setMinimumHeight(height)

    def set_data(self, series, xr, yr, *, bands=(), hlines=(), vlines=(), markers=(), xlabel='', ylabel='',
                 reverse_x=False):
        self.series, self.bands, self.hlines = list(series), list(bands), list(hlines)
        self.vlines, self.markers = list(vlines), list(markers)
        self.xr = (float(xr[0]), float(xr[1]) if xr[1] > xr[0] else float(xr[0]) + 1.0)
        self.yr = (float(yr[0]), float(yr[1]) if yr[1] > yr[0] else float(yr[0]) + 1.0)
        self.xlabel, self.ylabel, self.reverse_x = xlabel, ylabel, reverse_x
        self.tip = None
        self.update()

    def _plot(self) -> QRectF:
        top = 40 if (self.title or self.subtitle) and self.ylabel else 24    # the y label goes below the title row
        return QRectF(52, top, max(self.width() - 66, 40), max(self.height() - 38 - top, 40))

    def _m(self, x, y) -> QPointF:
        R = self._plot()
        fx = (x - self.xr[0]) / (self.xr[1] - self.xr[0])
        if self.reverse_x:
            fx = 1.0 - fx
        fy = (y - self.yr[0]) / (self.yr[1] - self.yr[0])
        return QPointF(R.left() + fx * R.width(), R.bottom() - fy * R.height())

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.paint_title(p)
        R = self._plot()
        for x0, x1, col, lab in self.bands:
            a, b = self._m(x0, self.yr[1]), self._m(x1, self.yr[0])
            r = QRectF(a, b).normalized().intersected(R)
            c = QColor(col)
            c.setAlpha(46)
            p.fillRect(r, c)
            if lab:
                p.setFont(sans(9))
                p.setPen(THEME.color('ink2'))
                p.drawText(r.adjusted(3, 2, -2, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, lab)
        p.setFont(mono(10))
        for v in _ticks(*self.yr):
            y = self._m(self.xr[0], v).y()
            p.setPen(QPen(THEME.color('hair'), 1))
            p.drawLine(QPointF(R.left(), y), QPointF(R.right(), y))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, y - 7, R.left() - 6, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       _fmt(v))
        for v in _ticks(*self.xr):
            x = self._m(v, self.yr[0]).x()
            p.setPen(QPen(THEME.color('hair'), 1))
            p.drawLine(QPointF(x, R.top()), QPointF(x, R.bottom()))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(x - 24, R.bottom() + 3, 48, 13), Qt.AlignmentFlag.AlignCenter, _fmt(v))
        p.setPen(QPen(THEME.color('strong'), 1))
        p.drawLine(R.bottomLeft(), R.bottomRight())
        p.setFont(sans(10))
        p.setPen(THEME.color('muted'))
        p.drawText(QRectF(R.left(), R.bottom() + 17, R.width(), 14), Qt.AlignmentFlag.AlignRight, self.xlabel)
        p.drawText(QRectF(0, R.top() - 16, 260, 14), Qt.AlignmentFlag.AlignLeft, self.ylabel)
        p.setClipRect(R.adjusted(-3, -3, 3, 3))
        for v, lab in self.hlines:
            pen = QPen(THEME.color('ink2'), 1, Qt.PenStyle.CustomDashLine)
            pen.setDashPattern([3, 3])
            p.setPen(pen)
            y = self._m(self.xr[0], v).y()
            p.drawLine(QPointF(R.left(), y), QPointF(R.right(), y))
            if lab:
                p.setFont(sans(9))
                p.drawText(QRectF(R.left() + 4, y - 14, R.width() - 8, 12), Qt.AlignmentFlag.AlignRight, lab)
        for v, lab in self.vlines:
            p.setPen(QPen(THEME.color('ink2'), 1))
            x = self._m(v, self.yr[0]).x()
            p.drawLine(QPointF(x, R.top()), QPointF(x, R.bottom()))
            if lab:
                p.setFont(sans(9))
                p.drawText(QRectF(x + 3, R.top() + 2, 140, 12), Qt.AlignmentFlag.AlignLeft, lab)
        for lab, xs, ys, col, style in self.series:
            c = QColor(col)
            if style == 'points':
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(c)
                for x, y in zip(xs, ys):
                    if y is not None:
                        p.drawEllipse(self._m(x, y), 2.6, 2.6)
                continue
            pen = QPen(c, 1.6)
            if style == 'dash':
                pen.setStyle(Qt.PenStyle.CustomDashLine)
                pen.setDashPattern([4, 3])
            p.setPen(pen)
            run = []
            for x, y in list(zip(xs, ys)) + [(None, None)]:
                if y is None:
                    if len(run) > 1:
                        p.drawPolyline(QPolygonF(run))
                    elif len(run) == 1:
                        p.drawEllipse(run[0], 1.8, 1.8)
                    run = []
                else:
                    run.append(self._m(x, y))
        for x, y, col, lab in self.markers:
            q = self._m(x, y)
            p.setPen(QPen(THEME.color('panel'), 1.5))
            p.setBrush(QColor(col))
            p.drawEllipse(q, 4.5, 4.5)
            if lab:
                p.setFont(sans(9))
                p.setPen(THEME.color('ink'))
                p.drawText(QRectF(q.x() + 6, q.y() - 15, 180, 12), Qt.AlignmentFlag.AlignLeft, lab)
        p.setClipping(False)
        self.paint_tip(p)
        p.end()

    def mouseMoveEvent(self, e):
        best = None
        pos = e.position()
        for lab, xs, ys, _c, _s in self.series:
            for x, y in zip(xs, ys):
                if y is None:
                    continue
                q = self._m(x, y)
                d = (q.x() - pos.x()) ** 2 + (q.y() - pos.y()) ** 2
                if best is None or d < best[0]:
                    best = (d, q, lab, x, y)
        if best is not None and best[0] < 18 ** 2:
            _d, q, lab, x, y = best
            self.set_tip(q, f'{_fmt(y)} {self.yunit}'.strip(), lab, f'{self.xname} {_fmt(x)} {self.xunit}'.strip())
        else:
            self.tip = None
        self.update()
