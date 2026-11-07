"""The Silhouette calibration page (spec §8.4): a static camera's mount, and its drift over a recorded dataset's
episodes, from the robot's own silhouette in the frames, with no board (core/calib/render_calib.py,
core/studies/silhouette_calib.py).

What it shows: the recovered offset of the camera from its mount with its 1σ (the headline, with a status word from
the tolerance, defaults 2 mm and 0.2°), the frame of the headline episode with the rendered silhouette before (the
project's mount, dashed) and after (the recovered mount), a row per episode (offset, σ, the foreground IoU and edge
RMS, observable or why not), the steps between episodes, and the render's GPU. The 3D view shows the camera at its
mount and at the recovered pose; the Camera view, when it shows the calibrated camera, outlines the robot rendered
through the recovered mount at the joint state on screen.

Apply writes the recovered mount (the sensor body pose in its parent frame) to the sensor, one undo step; Preview
shows where the camera would go in the 3D view.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QComboBox, QFileDialog, QHeaderView, QLabel, QMessageBox, QTreeWidget, QTreeWidgetItem

from ...core.studies import silhouette_calib as SC
from ..theme import STATUS, THEME
from ..widgets import ScrubSpinBox, button, label
from . import register
from .base import StudyPage, row

THUMB_W = 320
BEFORE_COLOR = '#8a8a85'
AFTER_COLOR = STATUS['good']
CAMVIEW_COLOR = '#e0883a'


class _Signals(QObject):
    seg_ready = pyqtSignal(object)


def _num(x, fmt='{:.2f}') -> str:
    return '—' if x is None or (isinstance(x, float) and not np.isfinite(x)) else fmt.format(x)


class SilhouettePage(StudyPage):
    """Silhouette calibration: the robot is rendered at each recorded joint state through MuJoCo's segmentation and
    its outline aligned with the recorded frame's edges and foreground (truncated directional chamfer distance and
    IoU), optimising the camera's mount correction. The figure is the offset of the camera from its mount, with its
    1σ from the alignment's Hessian, a cluster-robust covariance and a bootstrap over frames."""
    STUDY = SC.SID
    TITLE = 'Silhouette calibration'
    HEADLINE = 'Camera offset from its mount, from the robot\'s silhouette'

    # ---------------------------------------------------------------- body
    def build_body(self, layout):
        self._overlay = []
        self._preview = False
        self._seg = {}
        self._seg_busy = False
        self._seg_pending = None
        self._seg_renderer = None
        self._seg_lock = threading.Lock()
        self._sig = _Signals()
        self._sig.seg_ready.connect(self._seg_done)
        self.b_dataset = button('Dataset…', role='ghost', icon_name='dataset',
                                tip='Choose the recorded dataset (a LeRobot, RLDS or ROS 2 bag folder)')
        self.b_dataset.clicked.connect(lambda: self.choose_dataset())
        self.dataset_label = label('', muted=True)
        self.sensor_box = QComboBox()
        self.sensor_box.setToolTip('The static camera to calibrate (cameras on a robot link use the hand-eye solve)')
        self.sensor_box.activated.connect(self._sensor_chosen)
        layout.addWidget(row(label('Dataset'), self.b_dataset, self.dataset_label))
        self.frames = ScrubSpinBox('', 'plain', 0, 'frames', step=1.0, width=90)
        self.frames.setRange(1, 200)
        self.frames.setToolTip('Frames per episode, chosen where the joints differ most (default 12)')
        self.frames.committed.connect(lambda v: self._set('max_frames', int(round(v)), f'{int(round(v))} frames'))
        self.episodes = ScrubSpinBox('', 'plain', 0, 'episodes', step=1.0, width=90)
        self.episodes.setRange(1, 1000)
        self.episodes.setToolTip('Episodes estimated, from the first (default 8)')
        self.episodes.committed.connect(lambda v: self._set('max_episodes', int(round(v)),
                                                            f'{int(round(v))} episodes'))
        layout.addWidget(row(label('Camera'), self.sensor_box, label('Frames'), self.frames, label('Episodes'),
                             self.episodes))
        self.defaults_label = label('', muted=True, small=True, wrap=True)
        layout.addWidget(self.defaults_label)
        layout.addWidget(label('The headline episode: the recorded frame and the rendered silhouette', bold=True))
        self.thumb = QLabel()
        self.thumb.setMinimumHeight(60)
        layout.addWidget(self.thumb)
        self.thumb_legend = label('', muted=True, small=True, wrap=True)
        layout.addWidget(self.thumb_legend)
        layout.addWidget(label('Episodes', bold=True))
        self.table = QTreeWidget()
        self.table.setRootIsDecorated(False)
        self.table.setHeaderLabels(['Episode', 'Offset', 'Rotation', '1σ', 'IoU before → after', 'Edge RMS',
                                    'Frames', 'Observable'])
        h = self.table.header()
        h.setStretchLastSection(True)
        for k in range(self.table.columnCount() - 1):
            h.setSectionResizeMode(k, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setMinimumHeight(120)
        layout.addWidget(self.table)
        self.steps_label = label('', wrap=True)
        layout.addWidget(self.steps_label)
        self.render_label = label('', muted=True, small=True, wrap=True)
        layout.addWidget(self.render_label)
        self.notes_label = label('', muted=True, wrap=True)
        self.notes_label.setVisible(False)
        layout.addWidget(self.notes_label)
        st = self.ctx.store
        st.changed.connect(lambda _s: self._sync())
        st.project_replaced.connect(self._replaced)
        self._sync()

    # ---------------------------------------------------------------- settings
    def settings(self) -> dict:
        return SC.settings(self.ctx.store.project)

    def _set(self, key, value, what):
        cur = self.settings().get(key)
        if cur == value:
            return

        def fn(p):
            d = dict(p.analysis.studies.get(SC.SID) or {})
            if value is None:
                d.pop(key, None)
            else:
                d[key] = value
            p.analysis.studies[SC.SID] = d
        self.ctx.store.edit('analysis', fn, f'Silhouette calibration · {what}', merge_key=(SC.SID, key))

    def choose_dataset(self, path: str | None = None):
        """Set the dataset (a folder; the dialog when path is None)."""
        if path is None:
            path = QFileDialog.getExistingDirectory(self, 'Choose the recorded dataset')
            if not path:
                return None
        self._set('dataset', str(path), f'dataset {os.path.basename(os.path.normpath(str(path)))}')
        return path

    def _sensor_chosen(self, i):
        uid = self.sensor_box.itemData(i)
        self._set('sensor', uid or None, f'camera {self.sensor_box.itemText(i)}')

    def _sync(self):
        from ...core.calib import render_calib as RC
        p = self.ctx.store.project
        st = self.settings()
        cams = RC.static_cameras(p)
        self.sensor_box.blockSignals(True)
        self.sensor_box.clear()
        for s in cams:
            self.sensor_box.addItem(s.name, s.uid)
        want = st.get('sensor') or (cams[0].uid if cams else None)
        k = self.sensor_box.findData(want)
        if k >= 0:
            self.sensor_box.setCurrentIndex(k)
        self.sensor_box.blockSignals(False)
        self.sensor_box.setEnabled(bool(cams))
        ds = st.get('dataset')
        self.dataset_label.setText(os.path.basename(os.path.normpath(str(ds))) if ds else 'none chosen')
        try:
            cs = RC.settings_of(p)
        except RC.CalibError:
            cs = dict(RC.DEFAULTS, defaults=list(RC.DEFAULTS))
        for w, key in ((self.frames, 'max_frames'), (self.episodes, 'max_episodes')):
            w.blockSignals(True)
            w.setValue(float(cs[key]))
            w.blockSignals(False)
        shown = [k for k in ('max_frames', 'max_episodes', 'tolerance_mm', 'tolerance_deg', 'need_mm', 'need_deg',
                             'truncate_px', 'bootstrap') if k in cs.get('defaults', [])]
        self.defaults_label.setText('Defaults in use: ' + ', '.join(f'{k} {RC.DEFAULTS[k]}' for k in shown) +
                                    ' (DexVantage defaults, not standards).' if shown else '')
        self._apply_state()

    def _apply_state(self):
        """After Apply (and after its undo) the headline says whether the camera has the recovered mount now."""
        from ...core.frames import Pose
        ap = (self.doc or {}).get('apply')
        if not ap:
            return
        s = self.ctx.store.project.variant.sensor(ap['sensor_uid'])
        if s is None:
            return
        try:
            applied = bool(np.allclose(Pose.from_dict(ap['pose']).matrix(), s.mount.pose.matrix(), atol=1e-9))
        except (KeyError, TypeError, ValueError):
            return
        self.suggestion_ready(not applied and not s.locked)
        self.head.explain.setText(
            f'Applied: {s.name} has the mount recovered from episode {ap.get("episode")}. Run again to check it '
            'against the recording.' if applied else self.doc.get('advice', ''))

    def _replaced(self):
        self.doc, self._overlay, self._preview = None, [], False
        self.table.clear()
        self.thumb.clear()
        self.thumb_legend.setText('')
        self.steps_label.setText('')
        self.render_label.setText('')
        self.notes_label.setVisible(False)
        self._close_seg_renderer()
        self.suggestion_ready(False)
        self._sync()
        self.empty()

    # ---------------------------------------------------------------- result
    def show_result(self, doc):
        try:
            self._headline(doc)
            self._fill_table(doc)
            self._fill_thumb(doc)
            steps = [s for s in (doc.get('drift') or {}).get('steps') or [] if s.get('step')]
            if steps:
                self.steps_label.setText('Mount steps: ' + '; '.join(
                    f'between episodes {s["from"]} and {s["to"]}, {s["norm_mm"]:.1f} mm and {s["angle_deg"]:.2f}° '
                    f'({s["z"]:.1f} σ)' for s in steps) + '.')
            else:
                n = len([d for d in doc.get('episodes') or [] if 'delta' in d])
                self.steps_label.setText(f'No step between the {n} episodes estimated '
                                         f'({(doc.get("drift") or {}).get("step_rule", "")}).' if n > 1 else '')
            r = doc.get('render') or {}
            self.render_label.setText(f'Rendered by MuJoCo {r.get("mujoco", "")} through EGL on '
                                      f'{r.get("renderer") or "?"} ({r.get("device", "")}): {r.get("renders", 0)} '
                                      'renders for the headline episode.')
            notes = [str(n) for n in doc.get('notes') or [] if not str(n).startswith('The mount moved between')]
            self.notes_label.setText('\n'.join(notes))
            self.notes_label.setVisible(bool(notes))
        except (KeyError, ValueError, TypeError):
            traceback.print_exc()
        self.suggestion_ready(bool(doc.get('apply')))
        self._apply_state()
        self._build_overlay()
        self._refresh_camview()

    def _headline(self, doc):
        h = doc.get('headline') or {}
        if not h:
            self.head.set('—', '', 'warning', 'nothing estimated', doc.get('advice', ''))
            return
        fig = f'{h["offset_mm"]:.1f} mm · {h["offset_deg"]:.2f}°'
        unit = f'± {h["sigma_mm"]:.1f} mm · {h["sigma_deg"]:.2f}° (1σ), episode {h["episode"]}'
        rq = doc.get('requirement') or {}
        tip = (f'Status: the mount holds within {rq.get("tolerance_mm", 0):g} mm and {rq.get("tolerance_deg", 0):g}°; '
               f'moved when beyond them by more than 3σ (DexVantage defaults, not standards). Observability: '
               f'{h.get("why", "")}.')
        self.head.set(fig, unit, doc.get('status', 'warning'), doc.get('word', ''), doc.get('advice', ''), tip)

    def _fill_table(self, doc):
        self.table.clear()
        for d in doc.get('episodes') or []:
            if 'delta' not in d:
                it = QTreeWidgetItem([str(d.get('episode')), '—', '—', '—', '—', '—', '—', d.get('error', '')])
                self.table.addTopLevelItem(it)
                continue
            dl, sg, fit = d['delta'], d['sigma'], d['fit']
            smm = float(np.linalg.norm(np.nan_to_num(sg['position_mm'])))
            sdeg = float(np.linalg.norm(np.nan_to_num(sg['rotation_deg'])))
            it = QTreeWidgetItem([
                str(d['episode']), f'{dl["norm_mm"]:.2f} mm', f'{dl["angle_deg"]:.3f}°',
                f'{smm:.2f} mm · {sdeg:.3f}°',
                f'{_num(fit.get("iou_before"))} → {_num(fit.get("iou_after"))}', f'{_num(fit.get("rms_px"))} px',
                str(len(d.get('frames_used') or [])), 'yes' if d.get('observable') else d.get('why', 'no')])
            it.setToolTip(1, 'ΔT in the sensor body: ' + ', '.join(f'{a} {v * 1000:+.2f} mm' for a, v in
                                                                    zip('xyz', dl['position_m'])))
            it.setToolTip(2, 'rotation vector: ' + ', '.join(f'{a} {v:+.3f}°' for a, v in
                                                            zip('xyz', dl['rotation_deg'])))
            it.setToolTip(7, d.get('why', ''))
            self.table.addTopLevelItem(it)

    def _fill_thumb(self, doc):
        ov = (doc.get('estimate') or {}).get('overlay') or {}
        if not ov.get('png_base64'):
            self.thumb.clear()
            self.thumb_legend.setText('')
            return
        self.thumb.setPixmap(overlay_pixmap(ov, THUMB_W))
        self.thumb_legend.setText(f'Frame {ov.get("frame")}: grey dashed, the robot rendered through the project\'s '
                                  'mount; green, through the recovered mount.')

    # ---------------------------------------------------------------- Apply / Preview
    def apply(self) -> bool:
        """Write the recovered mount to the sensor (one undo step)."""
        from ...core.frames import Pose
        ap = (self.doc or {}).get('apply')
        if not ap:
            return False
        p = self.ctx.store.project
        s = p.variant.sensor(ap['sensor_uid'])
        win = self.ctx.window
        if s is None:
            if win is not None:
                win.msg('The calibrated camera is not in this rig any more.', 6000)
            return False
        if s.locked:
            if win is not None:
                win.msg(f'{s.name} is locked: unlock it on the Sensors and mounts page to apply the calibration.',
                        6000)
            return False

        def fn(pr):
            sen = pr.variant.sensor(ap['sensor_uid'])
            sen.mount.pose = Pose.from_dict(ap['pose'])
        self.ctx.store.edit('variant', fn, f'Silhouette calibration · apply the recovered mount of {s.name}')
        self._preview = False
        self.suggestion_ready(False)
        if win is not None:
            win.msg(f'Applied: {s.name} moved {ap["change_mm"]:.2f} mm and {ap["change_deg"]:.3f}° to the mount '
                    f'recovered from episode {ap.get("episode")}.', 8000)
        self._build_overlay()
        return True

    def preview(self):
        if not (self.doc or {}).get('apply'):
            return
        self._preview = True
        self._build_overlay()
        if self.ctx.window is not None:
            ap = self.doc['apply']
            self.ctx.window.msg(f'Preview: the camera moves {ap["change_mm"]:.2f} mm and {ap["change_deg"]:.3f}° '
                                '(grey: its mount now; coloured: recovered). Apply makes the change.', 8000)

    # ---------------------------------------------------------------- 3D overlay
    def _build_overlay(self):
        from ...core.frames import Pose
        from ..viewport3d import LineDraw
        self._overlay = []
        est = (self.doc or {}).get('estimate') or {}
        try:
            if est.get('pose') and est.get('pose_before'):
                A = Pose.from_dict(est['pose_before']).matrix()
                B = Pose.from_dict(est['pose']).matrix()
                size = 0.05 if self._preview else 0.03
                self._overlay += _triad(A, size, grey=True) + _triad(B, size)
                self._overlay.append(LineDraw(np.array([[A[:3, 3], B[:3, 3]]]), THEME.color('ink', 0.6), 1.0,
                                              dash=True, segments=True, depth=False))
        except (KeyError, ValueError, TypeError):
            traceback.print_exc()
            self._overlay = []
        win = self.ctx.window
        if win is not None and hasattr(win, 'viewport'):
            win.viewport.update()

    def overlays(self, vp) -> list:
        return list(self._overlay)

    # ---------------------------------------------------------------- explain
    def explain(self):
        st = self.settings()
        box = QMessageBox(self)
        box.setWindowTitle('How the silhouette calibration works')
        box.setText(
            'For each episode, up to ' + str(st.get('max_frames', 12)) + ' frames where the joints differ most are '
            'chosen. The robot is rendered in MuJoCo at each frame\'s joint state through the camera\'s lens and the '
            'mount being estimated; its outline is compared with the frame\'s Canny edges (each outline point to the '
            'nearest edge of similar direction, cut off at a truncation distance, so clutter does not pull) and with '
            'the foreground: the pixels that differ from the median of the frames.\n\n'
            'A coarse search (each move rendered) on chamfer distance and IoU is followed by Gauss-Newton on the '
            'outline points, coarse to fine, re-rendering the outline at each step. The correction ΔT is in the '
            'sensor\'s body frame.\n\n'
            '1σ is the largest of three estimates per axis: the Gauss-Newton covariance from the Hessian, a '
            'cluster-robust covariance (residuals of one link at one pose share their errors, plus a 0.25 px '
            'systematic edge error, a default) and a bootstrap over frames. The estimate is observable when the '
            'robot covers enough of the image, the frames hold at least three distinct joint states, and every '
            'direction of the Hessian is within the requirement (5 mm and 0.5°, defaults).\n\n'
            'Known limits: the robot must look like its model; the intrinsics and the joint timestamps are taken as '
            'exact (a camera–joint offset shows as a mount error); a small, distant robot leaves sideways shift and '
            'yaw hard to tell apart, and σ is a lower bound along that direction. Apply writes the recovered mount, '
            'one undo step.')
        box.setModal(False)
        box.show()

    # ---------------------------------------------------------------- Camera view: the silhouette at the new mount
    def _refresh_camview(self):
        cv = getattr(self.ctx.window, 'camview', None) if self.ctx.window is not None else None
        if cv is not None and hasattr(cv, 'view'):
            cv.view.refresh_overlays()

    def _seg_key(self, cam_model) -> str:
        q = {u: [round(float(x), 4) for x in np.asarray(v).reshape(-1)] for u, v in self.ctx.store.q.items()}
        est = (self.doc or {}).get('estimate') or {}
        return json.dumps([cam_model.to_dict(), est.get('pose'), q, self._inputs_key()], sort_keys=True, default=str)

    def _inputs_key(self) -> str:
        p = self.ctx.store.project
        return json.dumps([p.workspace.to_dict(), [r.to_dict() for r in p.robots]], sort_keys=True, default=str)

    def camview_overlays(self, cam_model, pose) -> list:
        """The robot rendered through the recovered mount, outlined, when the Camera view shows the calibrated
        camera (rendered in the background)."""
        est = (self.doc or {}).get('estimate') or {}
        cv = getattr(self.ctx.window, 'camview', None) if self.ctx.window is not None else None
        uid = getattr(getattr(cv, 'view', None), 'sensor_uid', None)
        if not est.get('pose') or uid != est.get('sensor_uid'):
            return []
        key = self._seg_key(cam_model)
        hit = self._seg.get(key)
        if hit is None:
            self._start_seg(key, cam_model)
            return []
        return hit

    def _start_seg(self, key, cam_model):
        if not ((self.doc or {}).get('estimate') or {}).get('pose'):      # replaced or cleared meanwhile
            return
        if self._seg_busy:
            self._seg_pending = (key, cam_model)
            return
        self._seg_busy = True
        self._seg_pending = None
        from ...core.frames import Pose
        snap = self.ctx.store.project.snapshot()
        q = {u: np.asarray(v, float).copy() for u, v in self.ctx.store.q.items()}
        ikey = self._inputs_key()
        T = Pose.from_dict(self.doc['estimate']['pose'])

        def work():
            out = []
            try:
                import cv2
                with self._seg_lock:
                    r = self._seg_renderer
                    if r is None or r[0] != ikey:
                        if r is not None:
                            r[1].close()
                        from ...core.sim.render import Renderer
                        r = self._seg_renderer = (ikey, Renderer(snap))
                    s = min(1.0, 640.0 / cam_model.width)
                    cm = cam_model.scaled(max(1, int(round(cam_model.width * s))),
                                          max(1, int(round(cam_model.height * s))))
                    img = r[1].render_pose(cm, T, q, ('seg',))
                rob = [k for k, o in img['seg_owner'].items() if o[0] == 'robot']
                S = np.isin(img['seg'], rob).astype(np.uint8)
                cs, _h = cv2.findContours(S, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                sc = cam_model.width / cm.width
                polys = []
                for c in cs[:200]:
                    if len(c) >= 3:
                        pts = (c.reshape(-1, 2).astype(float) + 0.5) * sc - 0.5
                        polys.append(np.vstack([pts, pts[:1]]))
                if polys:
                    out = [(polys, QColor(CAMVIEW_COLOR), 'The robot through the recovered mount')]
            except Exception:                           # noqa: BLE001 — no overlay rather than a broken view
                traceback.print_exc()
            self._sig.seg_ready.emit((key, out))
        threading.Thread(target=work, daemon=True).start()

    def _seg_done(self, res):
        # a slot: an exception here would end the process (PyQt's default), so it is logged instead
        try:
            key, out = res
            self._seg_busy = False
            if self.doc is None:                         # the project was replaced while it rendered
                self._seg_pending = None
                return
            if len(self._seg) > 16:
                self._seg.clear()
            self._seg[key] = out
            pend, self._seg_pending = self._seg_pending, None
            if pend is not None and pend[0] not in self._seg:
                self._start_seg(*pend)
            self._refresh_camview()
        except Exception:                               # noqa: BLE001
            traceback.print_exc()

    def _close_seg_renderer(self):
        def close():
            with self._seg_lock:
                if self._seg_renderer is not None:
                    self._seg_renderer[1].close()
                    self._seg_renderer = None
        threading.Thread(target=close, daemon=True).start()
        self._seg = {}
        self._seg_pending = None


def _triad(T, size, grey=False) -> list:
    from ..viewport3d import LineDraw
    o = T[:3, 3]
    cols = (QColor(BEFORE_COLOR),) * 3 if grey else (QColor('#e5484d'), QColor('#30a46c'), QColor('#3e63dd'))
    return [LineDraw(np.array([[o, o + size * T[:3, k]]]), cols[k], 1.5, segments=True, depth=False)
            for k in range(3)]


def overlay_pixmap(ov: dict, width: int) -> QPixmap:
    """The study's overlay (a PNG thumbnail with outlines before and after) drawn as a pixmap."""
    img = QImage.fromData(base64.b64decode(ov['png_base64']), 'PNG')
    if img.isNull():
        return QPixmap()
    img = img.convertToFormat(QImage.Format.Format_RGB32)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    for key, col, dash in (('before', BEFORE_COLOR, True), ('after', AFTER_COLOR, False)):
        pen = QPen(QColor(col))
        pen.setWidthF(1.2)
        if dash:
            pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
        for poly in ov.get(key) or []:
            pts = [QPointF(float(u) + 0.5, float(v) + 0.5) for u, v in poly]
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a, b)
    p.end()
    pm = QPixmap.fromImage(img)
    return pm.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation) if img.width() != width else pm


register('silhouette', 'Silhouette calibration', SilhouettePage, menu='Calibrate', icon='target',
         action_id=None, workspace='calibration', order=62)
