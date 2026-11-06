"""The Coverage and resolution page (spec §7.1): how much of the task volume the rig's cameras see, and how sharply.

What it shows: the share of the task volume seen by two or more cameras (the headline, with a status word from the
user's thresholds; defaults 90 %, 60 % and 30 %, labelled as defaults), a table per camera (the imager used, the share
it sees, the share outside its image and hidden by the scene, pixels per millimetre, the viewing angle, the defocus
when it can be computed), a table per task volume, and a slice through the voxels drawn in the 3D view, coloured by
the number of cameras that see each voxel or by pixels per millimetre (the best camera's or one camera's), with a
legend. Run passes the 3D view's current joint state to the study, so the coverage is for the pose on screen.
"""
from __future__ import annotations

import json
import traceback

import numpy as np
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QHeaderView, QMessageBox, QSlider,
                             QTreeWidget, QTreeWidgetItem, QWidget)

from ...core.studies import coverage as CV
from .. import colormaps as CM
from ..theme import STATUS, THEME, mono
from ..widgets import ScrubSpinBox, SegmentedControl, label
from . import register
from .base import StudyPage, row

NOT_STATED = 'not stated by the maker'
SLICE_ALPHA = 0.85
UNSEEN_ALPHA = 0.30
THRESHOLDS = (('good_pct', 'Good from', '%', 'The share seen by two or more cameras that reads "good" (default 90 %).'),
              ('warn_pct', 'Warning from', '%', 'Below "good" and from this share it reads "warning" (default 60 %).'),
              ('serious_pct', 'Serious from', '%', 'Below "warning" and from this share it reads "serious"; below it '
                                                   '"critical" (default 30 %).'),
              ('min_px_per_mm', 'Resolved from', 'px/mm', 'A camera resolves a voxel when it sees it at this pixel '
                                                         'density or more (default 1 px/mm). Run again to apply.'))


def _pct(x, nd=0) -> str:
    return '—' if x is None else f'{100 * x:.{nd}f} %'


def _f(x, fmt='{:.2f}') -> str:
    return '—' if x is None else fmt.format(x)


class Legend(QWidget):
    """A colour bar (continuous, log ticks) or a row of swatches (classes), in the chart conventions."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(44)
        self.setMinimumWidth(240)
        self.mode = None
        self.classes: list = []
        self.scale = None
        self.title = ''
        self.extra: list = []

    def set_classes(self, title: str, classes: list):
        self.mode, self.title, self.classes, self.scale = 'classes', title, classes, None
        self.update()

    def set_scale(self, title: str, scale, extra=()):
        self.mode, self.title, self.scale, self.extra = 'scale', title, scale, list(extra)
        self.update()

    def paintEvent(self, _ev):
        if self.mode is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        ink = THEME.color('ink2')
        p.setFont(mono(10))
        p.setPen(ink)
        p.drawText(QRectF(0, 0, self.width(), 14), Qt.AlignmentFlag.AlignLeft, self.title)
        y = 16
        if self.mode == 'classes':
            x = 0
            for text, col in self.classes:
                p.fillRect(QRectF(x, y + 2, 12, 10), col)
                p.setPen(ink)
                w = p.fontMetrics().horizontalAdvance(text)
                p.drawText(QRectF(x + 16, y, w + 4, 14), Qt.AlignmentFlag.AlignLeft, text)
                x += 16 + w + 14
            p.end()
            return
        sc = self.scale
        x0 = 0
        for text, col in self.extra:
            p.fillRect(QRectF(x0, y + 2, 12, 10), col)
            p.setPen(ink)
            w = p.fontMetrics().horizontalAdvance(text)
            p.drawText(QRectF(x0 + 16, y, w + 4, 14), Qt.AlignmentFlag.AlignLeft, text)
            x0 += 16 + w + 14
        bw = max(80.0, self.width() - x0 - 24)
        n = len(sc.lut)
        for i in range(int(bw)):
            c = sc.lut[min(n - 1, int(i / bw * n))]
            p.fillRect(QRectF(x0 + i, y + 2, 1.5, 10), QColor(int(c[0]), int(c[1]), int(c[2])))
        p.setPen(ink)
        for v in sc.ticks():
            t = float(np.nan_to_num(sc.t(v)))
            xx = x0 + t * bw
            txt = f'{v:g}'
            w = p.fontMetrics().horizontalAdvance(txt)
            p.drawText(QRectF(min(max(xx - w / 2, x0), x0 + bw - w), y + 12, w + 2, 12), Qt.AlignmentFlag.AlignLeft,
                       txt)
        p.end()


class CoveragePage(StudyPage):
    """Coverage and resolution: the share of the task volume (voxels weighted by their volume) that two or more
    cameras see at the current pose, where seeing means inside the image and not hidden by the robot, the objects or
    the tables (a cube map of distances around each camera)."""
    STUDY = 'coverage'
    TITLE = 'Coverage and resolution'
    HEADLINE = 'Task volume seen by two or more cameras'

    # ---------------------------------------------------------------- body
    def build_body(self, layout):
        self.dec = None
        self._inputs = None
        self._overlay = []
        self._spin = {}
        layout.addWidget(label('Cameras', bold=True))
        self.cams = QTreeWidget()
        self.cams.setRootIsDecorated(False)
        self.cams.setHeaderLabels(['Camera', 'Imager', 'Sees', 'Outside image', 'Hidden', 'Only camera',
                                   'px/mm median', 'px/mm 5 %', 'Angle median', 'Blur median'])
        _columns(self.cams)
        layout.addWidget(self.cams)
        layout.addWidget(label('Task volumes', bold=True))
        self.vols = QTreeWidget()
        self.vols.setRootIsDecorated(False)
        self.vols.setHeaderLabels(['Task volume', 'Voxels', 'Seen by ≥ 1', 'Seen by ≥ 2', 'Seen by none'])
        _columns(self.vols)
        layout.addWidget(self.vols)
        self.notes_label = label('', muted=True, wrap=True)     # the study's notes: cameras left out and why
        self.notes_label.setVisible(False)
        layout.addWidget(self.notes_label)

        # slice through the voxels, drawn in the 3D view
        layout.addWidget(label('Slice in the 3D view', bold=True))
        self.layer = QComboBox()
        self.layer.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.layer.setToolTip('What the slice is coloured by')
        self.axis = SegmentedControl([('x', 'x'), ('y', 'y'), ('z', 'z')], 'z')
        self.axis.setToolTip('The world axis the slice is across')
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(500)
        self.slider.setMinimumWidth(160)
        self.slider.setToolTip('Where the slice is along the axis')
        self.pos_label = label('', mono_=True)
        self.show3d = QCheckBox('Show')
        self.show3d.setChecked(True)
        self.show3d.setToolTip('Draw the slice in the 3D view while this page is shown')
        layout.addWidget(row(self.layer, self.axis, self.slider, self.pos_label, self.show3d))
        self.legend = Legend()
        layout.addWidget(self.legend)
        self.layer.currentIndexChanged.connect(self._slice_changed)
        self.axis.changed.connect(lambda _k: self._slice_changed())
        self.slider.valueChanged.connect(lambda _v: self._slice_changed())
        self.show3d.toggled.connect(lambda _on: self._redraw())

        # thresholds: user settings with documented defaults
        layout.addWidget(label('Thresholds (your settings; defaults labelled)', bold=True))
        g = QWidget()
        grid = QGridLayout(g)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        for i, (key, text, unit, tip) in enumerate(THRESHOLDS):
            # ScrubSpinBox, not a plain QDoubleSpinBox: Qt 6.12 draws a style-sheeted one empty (BUILD_STATE)
            sb = ScrubSpinBox('', 'plain', 1 if unit == '%' else 2, unit, step=5.0 if unit == '%' else 0.1, width=110)
            sb.setRange(0.0, 100.0 if unit == '%' else 1000.0)
            sb.setToolTip(tip)
            dl = label('', muted=True, small=True)
            grid.addWidget(label(text), i // 2, (i % 2) * 3)
            grid.addWidget(sb, i // 2, (i % 2) * 3 + 1)
            grid.addWidget(dl, i // 2, (i % 2) * 3 + 2)
            self._spin[key] = (sb, dl)
            sb.committed.connect(lambda v, k=key: self._set_threshold(k, v))
        layout.addWidget(g)
        self.stale_note = label('', muted=True, wrap=True)
        layout.addWidget(self.stale_note)
        self._sync_thresholds()
        st = self.ctx.store
        st.pose_changed.connect(lambda _uid: self._check_stale())
        st.changed.connect(lambda _s: (self._sync_thresholds(), self._check_stale()))
        st.project_replaced.connect(self._replaced)

    # ---------------------------------------------------------------- settings
    def settings(self) -> dict:
        try:
            return CV.settings_of(self.ctx.store.project)
        except Exception:                               # noqa: BLE001 — a bad stored value: show the defaults
            from ...core.optics import DEFAULT_COC_PX
            return dict(CV.DEFAULTS, coc_px=DEFAULT_COC_PX, defaults=list(CV.DEFAULTS))

    def _sync_thresholds(self):
        st = self.settings()
        for key, (sb, dl) in self._spin.items():
            sb.set_value(float(st[key]))
            d = CV.DEFAULTS[key]
            dl.setText('default' if key in st.get('defaults', ()) or abs(float(st[key]) - d) < 1e-12 else
                       f'yours (default {d:g})')

    def _set_threshold(self, key, value):
        if abs(float(self.settings()[key]) - float(value)) < 1e-12:
            return

        def fn(p):
            d = dict(p.analysis.studies.get('coverage') or {})
            d[key] = float(value)
            p.analysis.studies['coverage'] = d
        text = next(t for k, t, _u, _tip in THRESHOLDS if k == key)
        self.ctx.store.edit('analysis', fn, f'Coverage · {text.lower()} {value:g}', merge_key=('coverage', key))
        if self.doc is not None:
            self._headline(self.doc)

    # ---------------------------------------------------------------- run
    def live_q(self) -> dict:
        return {uid: [float(x) for x in np.asarray(q, float).reshape(-1)] for uid, q in self.ctx.store.q.items()}

    def run(self):
        """Run the study for the joint state on screen (passed in its settings, not stored in the project)."""
        self.stale_note.setText('')
        self.ctx.analysis.run(self.STUDY, {'q': self.live_q()})

    def _failed(self, sid, message):
        if sid == self.STUDY:
            from ..store import failure_text
            first, _explained = failure_text(message, 'The coverage study')   # the window logs the traceback
            self.head.set('—', '', 'critical', 'could not run', first)

    def _replaced(self):
        self.doc, self.dec, self._overlay, self._inputs = None, None, [], None
        self.cams.clear()
        self.vols.clear()
        self.notes_label.setText('')
        self.notes_label.setVisible(False)
        self.stale_note.setText('')
        self._sync_thresholds()
        self.empty()

    def inputs_key(self) -> str:
        """What the study reads from the project, as text (the headline thresholds left out: changing them only
        re-reads the status, it does not need a new run)."""
        p = self.ctx.store.project
        cov = {k: v for k, v in (p.analysis.studies.get('coverage') or {}).items()
               if k not in ('good_pct', 'warn_pct', 'serious_pct', 'q')}
        return json.dumps([p.workspace.to_dict(), p.variant.to_dict(), [r.to_dict() for r in p.robots],
                           p.analysis.voxel_mm, p.analysis.refine_mm, cov], sort_keys=True, default=str)

    def _check_stale(self):
        if self.doc is None:
            return
        if self._inputs is not None and self.inputs_key() != self._inputs:
            self.stale_note.setText('The project changed since this run: Run again to update the coverage.')
            return
        q = self.doc.get('q') or {}
        now = self.live_q()
        moved = any(uid in now and (len(now[uid]) != len(v) or not np.allclose(now[uid], v, atol=1e-9))
                    for uid, v in q.items())
        self.stale_note.setText('The robot has moved since this run: Run again for the coverage at this pose.'
                                if moved else '')

    # ---------------------------------------------------------------- result
    def _headline(self, doc):
        s = doc.get('summary') or {}
        pct2 = s.get('pct_ge2')
        if pct2 is None:
            self.head.set('—', '', 'warning', 'no voxels', 'The task volumes have no voxels to cover.')
            return
        st = self.settings()
        sev, word = CV.status_for(pct2, st)
        tip = (f'Good from {st["good_pct"]:g} %, warning from {st["warn_pct"]:g} %, serious from '
               f'{st["serious_pct"]:g} % (' + ('the defaults' if all(k in st.get('defaults', ()) for k in
                                                                    ('good_pct', 'warn_pct', 'serious_pct'))
                                               else 'your thresholds') + '). '
               f'Seen by at least one camera: {_pct(s.get("seen_ge1"), 1)}; resolved (≥ '
               f'{doc["settings"].get("min_px_per_mm", 1):g} px/mm) by two or more: {_pct(s.get("resolved_ge2"), 1)}.')
        self.head.set(f'{pct2:.1f}', '% of the task volume', sev, word,
                      s.get('advice', '') + f' Seen by at least one camera: {_pct(s.get("seen_ge1"))}.', tip)

    def show_result(self, doc):
        if doc.get('partial') and not doc.get('voxels'):
            return
        self._headline(doc)
        self.cams.clear()
        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        for c in doc.get('cameras', []):
            df = c.get('defocus') or {}
            blur = _f(df.get('blur_median_px'), '{:.1f} px') if df.get('computed') else 'not computed'
            it = QTreeWidgetItem([c['name'], c['imager'] + (' (default)' if c.get('is_default') else ''),
                                  _pct(c['seen']), _pct(c['outside_image']), _pct(c['hidden']), _pct(c['only_camera']),
                                  _f(c['ppmm_median']), _f(c['ppmm_p5']), _f(c['angle_median_deg'], '{:.0f}°'), blur])
            it.setIcon(0, _swatch(c.get('slot', 0)))
            it.setToolTip(0, f'{c["name"]}: {c["mode"][0]} × {c["mode"][1]} px, FOV {c["fov_deg"]["h"]:.1f}° × '
                             f'{c["fov_deg"]["v"]:.1f}°' + (' (on a robot link: it moves with q)' if c.get('on_link')
                                                            else ''))
            it.setToolTip(1, f'{c.get("imager_label", "")}. Used: {c.get("imager_why", "")}.')
            it.setToolTip(2, 'Share of the task volume this camera sees (in its image and not hidden)')
            it.setToolTip(3, 'Share of the task volume outside its image: aim it at the volume or widen the lens')
            it.setToolTip(4, 'Share in its image but hidden by the robot, the objects or the tables at this pose')
            it.setToolTip(5, 'Share that only this camera sees')
            it.setToolTip(6, 'Pixels per millimetre on a patch facing the camera, median over the voxels it sees '
                             '(volume-weighted)')
            it.setToolTip(7, '5 % of the voxels it sees are below this pixel density')
            it.setToolTip(8, 'Angle between the optical axis and the line of sight, median over the voxels it sees')
            it.setToolTip(9, df.get('why', '') + ('' if not df.get('computed') else
                                                  f' Sharp (blur ≤ {df.get("coc_px", 2):g} px, a default): '
                                                  f'{_pct(df.get("sharp"))}.'))
            for k in range(2, 10):
                it.setFont(k, mono(11))
                it.setTextAlignment(k, right)
            self.cams.addTopLevelItem(it)
        _fit(self.cams)
        self.vols.clear()
        for v in doc.get('volumes', []):
            it = QTreeWidgetItem([v['name'], f'{v["voxels"]:,}'.replace(',', ' '), _pct(v['seen_ge1']),
                                  _pct(v['seen_ge2']), _pct(v['blind'])])
            for k in range(1, 5):
                it.setFont(k, mono(11))
                it.setTextAlignment(k, right)
            if v.get('inside_objects'):
                it.setToolTip(1, f'{v["inside_objects"]:,} voxels lie inside task objects and are left out')
            self.vols.addTopLevelItem(it)
        _fit(self.vols)
        notes = [str(n) for n in doc.get('notes') or []]
        self.notes_label.setText('\n'.join(notes))
        self.notes_label.setVisible(bool(notes))
        cur = self.layer.currentData()
        self.layer.blockSignals(True)
        self.layer.clear()
        self.layer.addItem('Cameras that see it', 'redundancy')
        self.layer.addItem('Best pixels per mm', 'best')
        for j, c in enumerate(doc.get('cameras', [])):
            self.layer.addItem(f'Pixels per mm · {c["name"]}', f'cam:{j}')
        k = self.layer.findData(cur)
        self.layer.setCurrentIndex(k if k >= 0 else 0)
        self.layer.blockSignals(False)
        try:
            self.dec = CV.decode(doc)
            self._inputs = self.inputs_key()
        except (KeyError, ValueError):
            traceback.print_exc()
            self.dec = None
        self._check_stale()
        self._slice_changed()

    # ---------------------------------------------------------------- slice and 3D overlay
    def slice_position(self) -> tuple[str, float]:
        """(axis, coordinate in m) of the slice."""
        ax = self.axis.value() or 'z'
        if self.dec is None or not len(self.dec['centers']):
            return ax, 0.0
        a = 'xyz'.index(ax)
        c, s = self.dec['centers'][:, a], self.dec['size']
        lo, hi = float((c - s / 2).min()), float((c + s / 2).max())
        # the far end of the slider is the last layer of voxels, not the empty plane just past it (the mask is
        # half-open: c - s/2 <= x < c + s/2)
        return ax, min(lo + (hi - lo) * self.slider.value() / 1000.0, hi - 1e-6 * max(hi - lo, 1e-3))

    def _colors(self, sel):
        """(K, 4) RGBA 0..1 for the selected voxels, and set the legend."""
        d = self.dec
        layer = self.layer.currentData() or 'redundancy'
        min_ppmm = float((self.doc or {}).get('settings', {}).get('min_px_per_mm', CV.DEFAULTS['min_px_per_mm']))
        if layer == 'redundancy':
            r = d['redundancy'][sel].astype(int)
            pal = [QColor(STATUS['critical'])] + [QColor(h) for h in CM.redundancy()]
            cols = np.array([pal[min(k, 3)].getRgbF() for k in range(4)])
            out = cols[np.minimum(r, 3)]
            out[:, 3] = np.where(r == 0, UNSEEN_ALPHA * 2, SLICE_ALPHA)
            self.legend.set_classes('Cameras that see each voxel', [('none', pal[0]), ('1', pal[1]), ('2', pal[2]),
                                                                    ('3 or more', pal[3])])
            return out
        if layer == 'best':
            v = d['best_ppmm'][sel]
            seen = d['redundancy'][sel] > 0
            title = 'Best pixels per mm'
        else:
            j = int(layer.split(':')[1])
            cam = d['cameras'][j]
            v = cam['ppmm'][sel]
            seen = cam['state'][sel] == CV.STATE_SEEN
            title = f'Pixels per mm · {self.doc["cameras"][j]["name"]}'
        sc = CM.density_scale(min_ppmm, min_ppmm * 2)
        rgb = sc.rgb(np.where(seen, v, 0.0)) / 255.0
        alpha = np.where(seen, SLICE_ALPHA, UNSEEN_ALPHA)
        grey = QColor(sc.below_hex)
        self.legend.set_scale(f'{title} (log scale; grey: not seen or below {min_ppmm:g} px/mm)', sc,
                              extra=[('below / not seen', grey)])
        return np.c_[rgb, alpha]

    def _slice_changed(self):
        self._overlay = []
        ax, x = self.slice_position()
        self.pos_label.setText(f'{ax} = {x:.3f} m' if self.dec is not None else '')
        if self.dec is not None and len(self.dec['centers']):
            try:
                self._overlay = self._build_overlay(ax, x)
            except Exception:                           # noqa: BLE001 — the overlay must not take the page down
                traceback.print_exc()
                self._overlay = []
        self._redraw()
        cv = getattr(self.ctx.window, 'camview', None) if self.ctx.window is not None else None
        if cv is not None and hasattr(cv, 'view'):
            cv.view.refresh_overlays()                  # the Camera view draws the slice's outline too

    def slice_mask(self, ax: str, x: float) -> np.ndarray:
        a = 'xyz'.index(ax)
        c, s = self.dec['centers'][:, a], self.dec['size']
        return (c - s / 2 <= x) & (x < c + s / 2) & ~self.dec['inside']

    def _build_overlay(self, ax, x) -> list:
        from ..viewport3d import TriDraw
        sel = self.slice_mask(ax, x)
        if not sel.any():
            return []
        a = 'xyz'.index(ax)
        b, c = [k for k in range(3) if k != a]
        C = self.dec['centers'][sel]
        h = self.dec['size'][sel][:, None] / 2
        corners = []
        for sb, sc in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            q = C.copy()
            q[:, a] = x
            q[:, b] += sb * h[:, 0]
            q[:, c] += sc * h[:, 0]
            corners.append(q)
        p0, p1, p2, p3 = corners
        tri = np.concatenate([np.stack([p0, p1, p2], 1), np.stack([p0, p2, p3], 1)]).astype(np.float32)
        col = self._colors(sel)
        cols = np.concatenate([col, col]).astype(np.float32)
        return [TriDraw(tri, cols, depth=True)]

    def _redraw(self):
        win = self.ctx.window
        if win is not None and hasattr(win, 'viewport'):
            win.viewport.update()

    def overlays(self, vp) -> list:
        return list(self._overlay) if self.show3d.isChecked() else []

    def camview_overlays(self, cam_model, pose) -> list:
        """The slice drawn in the 3D view, seen through the Camera view: its outline across each task volume."""
        from ...core import camrender as CR
        if self.dec is None or not self.show3d.isChecked():
            return []
        ax, x = self.slice_position()
        a = 'xyz'.index(ax)
        b, c = [k for k in range(3) if k != a]
        polys = []
        for v in self.ctx.store.project.workspace.volumes:
            lo, hi = (np.asarray(t, float) for t in v.bounds())
            if not lo[a] <= x <= hi[a]:
                continue
            ring = []
            for sb, sc in ((0, 0), (1, 0), (1, 1), (0, 1), (0, 0)):
                p = np.empty(3)
                p[a], p[b], p[c] = x, (lo[b], hi[b])[sb], (lo[c], hi[c])[sc]
                ring.append(p)
            polys += CR.visible_polylines(None, cam_model, pose, np.array(ring), step=0.005)
        return [(polys, THEME.color('ink'), f'slice {ax} = {x:.3f} m')] if polys else []

    # ---------------------------------------------------------------- explain
    def explain_text(self) -> str:
        st = self.settings()
        return (
            'Each task volume is cut into voxels (5 mm by default, finer near the objects and in refinement '
            'regions). For every voxel and every camera the study asks: is its centre inside the image (the lens '
            'model of the library entry, or the labelled default when no device is chosen), and is anything of the '
            'scene nearer along the line of sight (the robot posed at the joint state on screen, the objects and the '
            'tables, from a cube map of distances around the camera)? A voxel inside the image with nothing in '
            'front is seen. Depth sensors are judged by their RGB imager when they have one (their reference imager '
            'otherwise); the table says which.\n\n'
            'The headline is the share of the task volume seen by two or more cameras, each voxel weighted by its '
            f'volume. It reads good from {st["good_pct"]:g} %, warning from {st["warn_pct"]:g} % and serious from '
            f'{st["serious_pct"]:g} % (your thresholds; the defaults are 90, 60 and 30 %, DexVantage defaults, not '
            'standards).\n\n'
            'Pixels per millimetre is the local pixel density of a small patch facing the camera: a 1 mm feature '
            f'spans that many pixels. A camera resolves a voxel at {st["min_px_per_mm"]:g} px/mm or more (default '
            '1 px/mm). The viewing angle is measured from the optical axis. The defocus blur needs the f-number, the '
            'focus distance and the pixel pitch; makers rarely state the focus distance of a fixed-focus lens, so it '
            'is shown only when every figure is stated or entered by you.\n\n'
            'Known limits: visibility is judged at the voxel centre and per cube-map texel; voxels inside the robot '
            'count as unseen; no sensor housings occlude; lighting, motion blur and depth validity are other '
            'studies.')

    def explain(self):
        box = QMessageBox(self)
        box.setWindowTitle('How coverage is computed')
        box.setText(self.explain_text())
        box.setModal(False)
        box.show()


def _swatch(slot: int):
    from PyQt6.QtGui import QIcon, QPixmap
    pm = QPixmap(10, 10)
    pm.fill(THEME.cam(slot))
    return QIcon(pm)


def _columns(tree: QTreeWidget):
    h = tree.header()
    h.setStretchLastSection(False)
    h.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    for k in range(1, tree.columnCount()):
        h.setSectionResizeMode(k, QHeaderView.ResizeMode.ResizeToContents)


def _fit(tree: QTreeWidget):
    n = tree.topLevelItemCount()
    rh = tree.sizeHintForRow(0) if n else 20
    tree.setFixedHeight(tree.header().sizeHint().height() + max(1, n) * max(rh, 18) + 6)


register('coverage', 'Coverage and resolution', CoveragePage, menu='Analysis', icon='coverage', action_id='coverage',
         workspace='placement', order=20)
