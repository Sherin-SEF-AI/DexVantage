"""The Camera view: the selected sensor's simulated image through its imager's lens model, with the task volumes and
the current analysis page's overlays drawn on it.

Ported from Vantage ui/camview.py (CamImageView: render on change, fit the image to the panel keeping the camera's
aspect, overlays as projected polylines). The image itself comes from core/camrender.py (the posed robots, the task
objects and the surfaces, through core/optics' CameraModel from library.sensor_imagers).

Frame budget: an image is rendered on a worker thread, never on the UI thread (the UI thread only takes a snapshot of
the project and the joint state, well under a millisecond). While the scene is changing quickly (a slider or the pose
scrubber being dragged, a mouse button held, or changes arriving within INTERACTIVE_MS of each other) the image is
rendered at DRAG_SCALE of its size; once the changes stop it is rendered again at full size. Requests that arrive
while a render runs are merged into one.

Overlays: the task volumes (their edges where the image sees them, i.e. not behind the robot or the table) and, while
an analysis page is shown, what the page's camview_overlays(cam_model, pose) returns: [(polylines, colour, label)]
with polylines in cam_model's pixel coordinates (cam_model is the imager at its stream size, pose its optical frame
in the world).
"""
from __future__ import annotations

import threading
import time
import traceback

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFontMetricsF, QGuiApplication, QImage, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QVBoxLayout, QWidget

from ..core import camrender as CR
from .theme import THEME, mono, sans
from .widgets import hline, label

DRAG_SCALE = 0.5                  # image size while the scene changes quickly
INTERACTIVE_MS = 150              # changes closer together than this count as a drag
SETTLE_MS = 180                   # after the last change: render at full size
MAX_PIXELS = 640 * 400            # the full-size image is at most this many pixels (the view scales it up)
DEBOUNCE_MS = 30


def _rgb(c: QColor) -> tuple:
    return (c.redF(), c.greenF(), c.blueF())


def sensor_mode(sensor):
    """The stream mode chosen for a sensor ((w, h)) or None (each imager's largest listed stream)."""
    m = getattr(sensor, 'stream', None)
    return tuple(int(v) for v in m) if m and len(m) == 2 else None


def pick_imager(imagers: list, want: str = ''):
    """The imager named want, else the first RGB one, else the first."""
    if not imagers:
        return None
    for im in imagers:
        if im.name == want:
            return im
    return next((im for im in imagers if im.role == 'rgb'), imagers[0])


def render_job(job: dict) -> dict:
    """Render one Camera view image from a job made by CameraView._job (pure: runs on any thread)."""
    from ..core import library
    t0 = time.perf_counter()
    p = job['project']
    s = p.variant.sensor(job['sensor'])
    if s is None:
        raise LookupError('The sensor is no longer in the rig.')
    ims = library.sensor_imagers(s, sensor_mode(s))
    im = pick_imager(ims, job['imager'])
    if im is None:
        return {'job': job, 'imager': None, 'imagers': ims, 'message': f'{s.name}\'s device has no imager to show.'}
    q = job['q']
    pose = CR.optical_pose(p, s, im, q)
    geom = CR.gather(p, q, colors=job['colors'])
    origins = [CR.optical_pose(p, s, i, q).position for i in ims]
    geom = CR.without_seat(geom, p, s, origins)
    cam = im.camera
    W, H = job['size']
    aspect = cam.width / cam.height
    w = min(W, int(round(H * aspect)))
    h = int(round(w / aspect))
    k = min(1.0, (MAX_PIXELS / max(w * h, 1)) ** 0.5, cam.width / max(w, 1)) * job['scale']
    w, h = max(16, int(w * k)), max(12, int(h * k))
    r = CR.render(geom, cam, pose, w, h, colors=job['colors'])
    rgba = np.concatenate([r.rgb, np.full(r.rgb.shape[:2] + (1,), 255, np.uint8)], axis=2)
    img = QImage(rgba.tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
    vols = []
    for v in p.workspace.volumes:
        lo, hi = v.bounds()
        for e in CR.box_polylines(lo, hi):
            vols += CR.visible_polylines(r, cam, pose, e, step=0.01)
    return {'job': job, 'imager': im, 'imagers': ims, 'pose': pose, 'render': r, 'image': img, 'volumes': vols,
            'camera': cam, 'sensor_name': s.name, 'slot': s.slot, 'seconds': time.perf_counter() - t0,
            'notes': geom.notes}


class _Image(QWidget):
    """Paints the last image fitted to the widget, the volumes and the page overlays."""

    def __init__(self, view):
        super().__init__()
        self.view = view
        self.setMinimumSize(80, 60)

    def image_rect(self) -> QRectF | None:
        res = self.view.result
        if not res or res.get('image') is None:
            return None
        cam = res['camera']
        W, H = self.width(), self.height()
        s = min(W / cam.width, H / cam.height)
        w, h = cam.width * s, cam.height * s
        return QRectF((W - w) / 2, (H - h) / 2, w, h)

    def to_px(self, uv, r: QRectF):
        cam = self.view.result['camera']
        sx, sy = r.width() / cam.width, r.height() / cam.height
        return [QPointF(r.left() + (u + 0.5) * sx, r.top() + (v + 0.5) * sy) for u, v in np.asarray(uv)]

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), THEME.color('viewport'))
        res = self.view.result
        r = self.image_rect()
        if r is None:
            p.setPen(THEME.color('muted'))
            p.setFont(sans(11))
            msg = (res or {}).get('message') or self.view.message or 'Rendering…'
            p.drawText(self.rect().adjusted(12, 12, -12, -12), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                       msg)
            p.end()
            return
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.drawImage(r, res['image'])
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setClipRect(r)
        pen = QPen(THEME.color('ink', 0.85), 1.2, Qt.PenStyle.CustomDashLine)
        pen.setDashPattern([4, 3])
        p.setPen(pen)
        for pl in res.get('volumes') or []:
            p.drawPolyline(QPolygonF(self.to_px(pl, r)))
        labels = []
        for polys, colour, text in self.view.page_overlays():
            c = colour if isinstance(colour, QColor) else QColor(str(colour))
            p.setPen(QPen(c, 1.6))
            for pl in polys:
                pts = self.to_px(pl, r)
                if len(pts) > 1:
                    p.drawPolyline(QPolygonF(pts))
            if text:
                labels.append((c, text))
        p.setClipping(False)
        f = sans(10)
        p.setFont(f)
        rows = [(THEME.color('ink'), 'task volumes (dashed)')] * bool(res.get('volumes')) + labels
        if rows:
            # a translucent plate behind the legend, so it reads over any image (a floor grid, a bright table)
            fm = QFontMetricsF(f)
            w = max(fm.horizontalAdvance(t) for _c, t in rows) + 12
            plate = QRectF(r.left() + 4, r.top() + 4, min(w, r.width() - 8), 15 * len(rows) + 6)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(THEME.color('viewport', 0.78))
            p.drawRoundedRect(plate, 3, 3)
            p.setBrush(Qt.BrushStyle.NoBrush)
        y = r.top() + 6
        for c, text in rows:
            p.setPen(QPen(c, 1))
            p.drawText(QPointF(r.left() + 10, y + 10), text)
            y += 15
        p.end()


class CameraView(QWidget):
    """The Camera view: a header (sensor, imager), the image and a line saying what it is."""
    rendered = pyqtSignal()
    _done = pyqtSignal(int, object)

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: dict | None = None
        self.message = ''
        self.sensor_uid = ''
        self.imager_name = ''
        self._job_id = 0
        self._busy = False
        self._pending = False
        self._last_change = 0.0
        self._reduced = False
        self._overlays = None
        self._lock = threading.Lock()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        head = QWidget()
        h = QHBoxLayout(head)
        h.setContentsMargins(6, 3, 6, 3)
        h.setSpacing(4)
        self.sensor_box = QComboBox()
        self.sensor_box.setToolTip('The sensor shown: follows the selection')
        self.sensor_box.activated.connect(self._sensor_picked)
        self.imager_box = QComboBox()
        self.imager_box.setToolTip('The imager of the device: RGB, IR left, IR right…')
        self.imager_box.activated.connect(self._imager_picked)
        h.addWidget(self.sensor_box, 3)
        h.addWidget(self.imager_box, 2)
        v.addWidget(head)
        v.addWidget(hline())
        self.image = _Image(self)
        v.addWidget(self.image, 1)
        self.info = label('', muted=True, small=True, wrap=True)
        self.info.setFont(mono(10))
        self.info.setContentsMargins(8, 2, 8, 4)
        v.addWidget(self.info)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(DEBOUNCE_MS)
        self.timer.timeout.connect(self._kick)
        self.settle = QTimer(self)
        self.settle.setSingleShot(True)
        self.settle.setInterval(SETTLE_MS)
        self.settle.timeout.connect(self._settled)
        self._done.connect(self._finished)
        st = ctx.store
        # bound methods, so Qt drops the connections with this widget
        st.changed.connect(self._on_change)
        st.pose_changed.connect(self._on_change)
        st.selection_changed.connect(self._on_selection)
        st.project_replaced.connect(self._on_replaced)
        THEME.changed.connect(self._on_change)
        ctx.analysis.finished.connect(self.refresh_overlays)
        self._refill_sensors()
        QTimer.singleShot(0, self._hook_window)

    def _hook_window(self):
        """Page overlays follow the Analysis dock's current tab (the dock exists once the window is built)."""
        bottom = getattr(self.ctx.window, 'bottom', None)
        if bottom is not None:
            bottom.tabs.currentChanged.connect(self.refresh_overlays)

    # ---------------------------------------------------------------- what is shown
    def sensor(self):
        return self.ctx.store.project.variant.sensor(self.sensor_uid) if self.sensor_uid else None

    def _follow(self):
        """The sensor to show: the selected one, else the one shown, else the first enabled sensor of the rig."""
        st = self.ctx.store
        v = st.project.variant
        if st.sel[:1] == ('sensor',) and len(st.sel) > 1 and v.sensor(st.sel[1]) is not None:
            self.sensor_uid = st.sel[1]
        elif v.sensor(self.sensor_uid) is None:
            s = next((s for s in v.sensors if s.enabled), v.sensors[0] if v.sensors else None)
            self.sensor_uid = s.uid if s is not None else ''

    def _refill_sensors(self):
        self._follow()
        sensors = self.ctx.store.project.variant.sensors
        self.sensor_box.blockSignals(True)
        self.sensor_box.clear()
        for s in sensors:
            self.sensor_box.addItem(s.name + ('' if s.enabled else ' (off)'), s.uid)
        i = self.sensor_box.findData(self.sensor_uid)
        self.sensor_box.setCurrentIndex(max(i, 0))
        self.sensor_box.blockSignals(False)
        self._refill_imagers()

    def _refill_imagers(self):
        from ..core import library
        s = self.sensor()
        self.imager_box.blockSignals(True)
        self.imager_box.clear()
        ims = []
        if s is not None:
            try:
                ims = library.sensor_imagers(s, sensor_mode(s))
            except Exception:                            # noqa: BLE001 — a bad calibration: said in the view
                traceback.print_exc()
        for im in ims:
            self.imager_box.addItem(im.name, im.name)
            self.imager_box.setItemData(self.imager_box.count() - 1, im.label, Qt.ItemDataRole.ToolTipRole)
        cur = pick_imager(ims, self.imager_name)
        if cur is not None:
            self.imager_box.setCurrentIndex(self.imager_box.findData(cur.name))
        self.imager_box.setEnabled(len(ims) > 1)
        self.imager_box.blockSignals(False)

    def _sensor_picked(self, i):
        uid = self.sensor_box.itemData(i)
        if uid:
            self.sensor_uid = uid
            self.ctx.store.select(('sensor', uid))
            self._refill_imagers()
            self.schedule()

    def _imager_picked(self, i):
        self.imager_name = self.imager_box.itemData(i) or ''
        self.schedule()

    def _on_selection(self):
        old = self.sensor_uid
        self._follow()
        if self.sensor_uid != old:
            self._refill_sensors()
            self.schedule()

    def _on_replaced(self):
        self.result = None
        self.sensor_uid = ''
        self._refill_sensors()
        self.schedule()

    def _on_change(self, *_):
        now = time.perf_counter()
        self._interactive = (now - self._last_change) * 1000 < INTERACTIVE_MS
        self._last_change = now
        if _ and isinstance(_[0], set):                  # an edit: the sensors or their devices may have changed
            self._refill_sensors()
        self.schedule()

    # ---------------------------------------------------------------- rendering
    def interactive(self) -> bool:
        """True while the scene is being dragged (render small)."""
        st = self.ctx.store
        ik = getattr(self.ctx, 'ik_drag', None)
        held = QGuiApplication.mouseButtons() != Qt.MouseButton.NoButton
        recent = (time.perf_counter() - self._last_change) * 1000 < INTERACTIVE_MS
        return bool(st.dragging or (ik is not None and getattr(ik, 'dragging', False)) or held
                    or (recent and getattr(self, '_interactive', False)))

    def schedule(self):
        """Render soon: a throttle, not a debounce (a running timer is left to fire), so a continuous drag still
        updates the image every DEBOUNCE_MS plus a render."""
        if self.isVisible() and not self.timer.isActive():
            self.timer.start()

    def showEvent(self, e):
        super().showEvent(e)
        self.schedule()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.schedule()

    def _colors(self) -> dict:
        return {'sky': _rgb(THEME.color('viewport')), 'object': _rgb(THEME.color('body')),
                'surface': _rgb(THEME.color('ground')), 'floor': _rgb(THEME.color('grid2')),
                'grid': _rgb(THEME.color('ink2'))}

    def _job(self, scale: float | None = None) -> dict | None:
        self._follow()
        s = self.sensor()
        if s is None:
            return None
        st = self.ctx.store
        dpr = self.devicePixelRatioF() or 1.0
        size = (max(32, int(self.image.width() * dpr)), max(24, int(self.image.height() * dpr)))
        if scale is None:
            scale = DRAG_SCALE if self.interactive() else 1.0
        self._job_id += 1
        return {'id': self._job_id, 'project': st.project.snapshot(), 'q': {k: np.array(v) for k, v in st.q.items()},
                'sensor': s.uid, 'imager': self.imager_name, 'size': size, 'scale': float(scale),
                'colors': self._colors(), 'revision': st.revision}

    def _kick(self):
        if self._busy:
            self._pending = True
            return
        job = self._job()
        if job is None:
            self.result = None
            self.message = 'No sensor in this rig.'
            self.image.update()
            return
        self._busy = True
        self._reduced = job['scale'] < 1.0
        threading.Thread(target=self._work, args=(job,), daemon=True).start()

    def _work(self, job):
        try:
            out = render_job(job)
        except Exception as e:                           # noqa: BLE001 — explained in the view, logged
            traceback.print_exc()
            out = {'job': job, 'error': str(e) or type(e).__name__}
        try:
            self._done.emit(job['id'], out)
        except RuntimeError:                             # the view was closed meanwhile
            pass

    def _finished(self, jid, out):
        self._busy = False
        self._apply(out)
        if self._pending:
            self._pending = False
            self.timer.start(0)
        elif self._reduced:
            self.settle.start()

    def _settled(self):
        if self.interactive():
            self.settle.start()
        elif self._reduced and not self._busy:
            self.timer.start(0)

    def _apply(self, out):
        if 'error' in out:
            self.result = None
            self.message = f'The image could not be made: {out["error"]}'
            self.info.setText('')
        else:
            self.result = out
            self.message = out.get('message', '')
            self._overlays = None
            self._describe(out)
        self.image.update()
        self.rendered.emit()

    def _describe(self, out):
        im = out.get('imager')
        if im is None:
            self.info.setText('')
            return
        r, cam = out['render'], out['camera']
        f = cam.fov_deg()
        shown = r.rgb.shape
        txt = (f'{im.name} · {cam.width} × {cam.height} px · {f["h"]:.0f}° × {f["v"]:.0f}° · '
               f'shown {shown[1]} × {shown[0]} · {out["seconds"] * 1000:.0f} ms')
        if im.is_default:
            if 'no device chosen' not in im.name:
                txt = 'Default camera (no device chosen) · ' + txt
        elif not im.pose_stated:
            txt += ' · offset assumed'
        self.info.setText(txt)
        cov = r.coverage
        self.info.setToolTip(f'{im.label}\n\nPixels showing the robot {cov["robot"] * 100:.0f} %, objects '
                             f'{cov["object"] * 100:.0f} %, surfaces {cov["surface"] * 100:.0f} %, floor '
                             f'{cov["floor"] * 100:.0f} %. Rendered by core/camrender.py ({r.path}, '
                             f'{r.triangles:,} triangles).'.replace(',', ' ')
                             + ('\n' + '\n'.join(out['notes']) if out.get('notes') else ''))

    def page_overlays(self) -> list:
        """[(polylines, colour, label)] from the current analysis page's camview_overlays (cached per image)."""
        if self._overlays is not None:
            return self._overlays
        out = []
        res = self.result
        win = self.ctx.window
        if res and res.get('imager') is not None and win is not None and hasattr(win, 'current_page'):
            page = win.current_page()
            fn = getattr(page, 'camview_overlays', None)
            if fn is not None and win.docks['bottom'].isVisible():
                try:
                    for item in fn(res['camera'], res['pose']) or []:
                        polys, colour, text = item
                        if isinstance(polys, np.ndarray) and polys.ndim == 2:
                            polys = [polys]
                        out.append(([np.asarray(pl, float).reshape(-1, 2) for pl in polys], colour, text or ''))
                except Exception:                        # noqa: BLE001 — an overlay must not take the view down
                    traceback.print_exc()
        self._overlays = out
        return out

    def refresh_overlays(self, *_):
        self._overlays = None
        self.image.update()

    # ---------------------------------------------------------------- for tests and reports
    def render_sync(self, scale: float = 1.0) -> dict | None:
        """Render now on this thread (tests, reports); returns the result (None without a sensor)."""
        self._refill_imagers()
        job = self._job(scale)
        if job is None:
            return None
        try:
            out = render_job(job)
        except Exception as e:                           # noqa: BLE001
            traceback.print_exc()
            out = {'job': job, 'error': str(e) or type(e).__name__}
        self._apply(out)
        return self.result

    def wait(self, timeout: float = 10.0) -> bool:
        """Process events until no render is running or waiting."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            QGuiApplication.processEvents()
            if not self._busy and not self._pending and not self.timer.isActive() and not self.settle.isActive():
                return True
            time.sleep(0.005)
        return False
