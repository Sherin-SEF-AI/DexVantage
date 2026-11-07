"""The Sim-to-real gap page (research tools part 2): one recorded episode rendered again through the project's rig and
compared with the recording, camera by camera (core/sim2real/gap.py, core/studies/sim2real.py).

What it shows: the largest image gap over the cameras (the mean absolute difference between the recorded and the
rendered frames, grey levels, before any correction), with a status word against DexVantage's default thresholds
(4, 8 and 16 DN); the recorded frame beside the render (as planned, or with every correction found) and their
difference image (or the difference laid over the recorded frame); the ranked list of what explains the gap, each with
its share, its figure and the page that applies it; and the camera's figures (photometric, geometric, depth, timing,
MMD). Apply makes the selected suggestion (the top one by default) in one undo step: the measured latency in the time
sync settings, the recovered mount, the render's camera response or the lights; Preview says what it would change;
Open page goes to the page that owns the setting.

Known limits: see the study's.
"""
from __future__ import annotations

import base64
import os
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QMessageBox,
                             QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ...core.datasets.episode import format_title
from ...core.sim2real import gap as G
from ..theme import STATUS
from ..widgets import button, label
from . import register
from .base import StudyPage, row

THUMB_W = 240
DIFF_GAIN = 4.0                    # the difference image is shown ×4 (a display choice)


class _Sig(QObject):
    fp = pyqtSignal(object)


def _f(x, nd=1, unit=''):
    if x is None:
        return '—'
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    t = f'{x:.{nd}f}' if np.isfinite(x) else '—'
    if t.startswith('-') and float(t) == 0:              # never '-0.0'
        t = t[1:]
    return t + (f' {unit}' if unit else '')


def _decode(b64: str) -> np.ndarray | None:
    import cv2
    if not b64:
        return None
    a = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    return None if a is None else cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def _pixmap(rgb: np.ndarray, width: int = THUMB_W) -> QPixmap:
    a = np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))
    h, w = a.shape[:2]
    img = QImage(a.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    pm = QPixmap.fromImage(img)
    return pm.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation) if w != width else pm


class Sim2RealPage(StudyPage):
    """Sim-to-real gap: the recorded episode is rendered again in MuJoCo through each camera's lens and mount with the
    robot at the recorded joint states, and the frames are compared: brightness, colour, sharpness and noise, the
    robot's silhouette, depth, timing and a kernel MMD. The figure is the largest image difference over the cameras;
    the list ranks what explains it (each correction's Shapley share) with the change that closes it."""
    STUDY = G.SID
    TITLE = 'Sim-to-real gap'
    HEADLINE = 'Image gap between the recording and its render (worst camera)'

    def build_body(self, layout):
        self._sig = _Sig()
        self._sig.fp.connect(lambda fp: self.ctx.analysis.run(G.SID, {'fingerprint': fp}))
        self._chosen = None
        self.b_dataset = button('Dataset…', role='ghost', icon_name='dataset',
                                tip='Choose the recorded dataset (a LeRobot, RLDS or ROS 2 bag folder)')
        self.b_dataset.clicked.connect(lambda: self.choose_dataset())
        self.dataset_label = label('', muted=True, wrap=True)
        layout.addWidget(row(label('Dataset'), self.b_dataset, self.dataset_label))
        self.cam_box = QComboBox()
        self.cam_box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)   # full camera names
        self.cam_box.setToolTip('The camera whose frames and figures are shown')
        self.cam_box.activated.connect(lambda _i: self._show_camera())
        self.frame_box = QComboBox()
        self.frame_box.setToolTip('The frame shown (the compared frames kept as thumbnails)')
        self.frame_box.activated.connect(lambda _i: self._show_frames())
        self.mode_box = QComboBox()
        self.mode_box.addItems(['Rendered as planned', 'Rendered with every correction'])
        self.mode_box.setToolTip('As planned: the project\'s mount, the recorded stamps and the stored camera '
                                 'response. With every correction: the latency, mount, blur and exposure found '
                                 'applied.')
        self.mode_box.activated.connect(lambda _i: self._show_frames())
        self.overlay = QCheckBox('Difference over the recording')
        self.overlay.setToolTip(f'Lay the difference image (×{DIFF_GAIN:g}, red) over the recorded frame')
        self.overlay.toggled.connect(lambda _v: self._show_frames())
        layout.addWidget(row(label('Camera'), self.cam_box, label('Frame'), self.frame_box, self.mode_box,
                             self.overlay))
        pics = QWidget()
        h = QHBoxLayout(pics)
        h.setContentsMargins(0, 0, 0, 0)
        self.pics = {}
        for k, title in (('real', 'Recorded'), ('sim', 'Rendered'), ('diff', f'Difference (×{DIFF_GAIN:g})')):
            box = QWidget()
            v = QVBoxLayout(box)
            v.setContentsMargins(0, 0, 0, 0)
            v.addWidget(label(title, small=True, muted=True))
            pic = QLabel()
            pic.setMinimumSize(80, 45)
            v.addWidget(pic)
            self.pics[k] = pic
            h.addWidget(box)
        h.addStretch(1)
        layout.addWidget(pics)
        self.frame_note = label('', muted=True, small=True, wrap=True)
        layout.addWidget(self.frame_note)
        layout.addWidget(label('What explains the gap (click one to choose it for Apply)', bold=True))
        self.causes = QTreeWidget()
        self.causes.setRootIsDecorated(False)
        self.causes.setHeaderLabels(['#', 'Cause', 'Camera', 'Removes', 'Share', 'Figure', 'Fixed on'])
        hd = self.causes.header()
        hd.setStretchLastSection(False)
        for k in range(self.causes.columnCount()):
            hd.setSectionResizeMode(k, QHeaderView.ResizeMode.ResizeToContents)
        self.causes.setMinimumHeight(110)
        self.causes.itemClicked.connect(lambda it, _c: self._choose(it.data(0, Qt.ItemDataRole.UserRole)))
        layout.addWidget(self.causes)
        self.b_page = button('Open page', role='ghost', tip='Open the page that owns the chosen change')
        self.b_page.clicked.connect(self.open_linked_page)
        self.b_page.setEnabled(False)
        self.cause_text = label('', wrap=True)
        layout.addWidget(row(self.b_page, self.cause_text))
        layout.addWidget(label('Figures of the camera', bold=True))
        self.figs = QTreeWidget()
        self.figs.setRootIsDecorated(True)
        self.figs.setHeaderLabels(['Figure', 'Recorded', 'Rendered', 'Gap'])
        self.figs.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.figs.setMinimumHeight(160)
        layout.addWidget(self.figs)
        self.notes_label = label('', muted=True, small=True, wrap=True)
        layout.addWidget(self.notes_label)
        st = self.ctx.store
        st.changed.connect(lambda _s: self._sync())
        st.project_replaced.connect(self._replaced)
        self._sync()

    # ---------------------------------------------------------------- dataset
    def choose_dataset(self, path: str | None = None):
        if path is None:
            path = QFileDialog.getExistingDirectory(self, 'Choose the recorded dataset')
            if not path:
                return None
        path = str(path)

        def fn(p):
            d = dict(p.analysis.studies.get(G.SID) or {})
            d['dataset'] = path
            p.analysis.studies[G.SID] = d
        self.ctx.store.edit('analysis', fn, f'Sim-to-real gap · dataset {os.path.basename(os.path.normpath(path))}')
        return path

    def _dataset_path(self):
        from ...core.studies import audit as AU
        return AU.dataset_path_for(self.ctx.store.project, G.SID)

    def _sync(self):
        try:
            p = self._dataset_path()
            self.dataset_label.setText(os.path.basename(os.path.normpath(p)))
        except Exception:                                # noqa: BLE001 — none chosen: say so
            self.dataset_label.setText('none chosen (Dataset…, or the audit\'s dataset)')
        self._apply_state()

    def run(self):
        try:
            path = self._dataset_path()
        except Exception as e:                          # noqa: BLE001 — no dataset: say so in the headline
            self.head.set('—', '', 'warning', 'no dataset', str(e))
            return
        from ...core.studies import audit as AU
        threading.Thread(target=lambda: self._sig.fp.emit(AU.fingerprint(path)), daemon=True).start()

    def _replaced(self):
        self.doc = None
        self._chosen = None
        for w in (self.causes, self.figs):
            w.clear()
        for pic in self.pics.values():
            pic.clear()
        self.cam_box.clear()
        self.frame_box.clear()
        self.suggestion_ready(False)
        self.b_page.setEnabled(False)
        self._sync()
        self.empty()

    # ---------------------------------------------------------------- result
    def show_result(self, doc):
        try:
            s = doc.get('summary') or {}
            tip = ('The mean absolute difference between the recorded and the rendered frames over pixels and '
                   'channels, before any correction, for the camera where it is largest. Status: close to real up to '
                   f'{_f(doc["settings"].get("gap_good_dn"), 0)} DN, a small gap up to '
                   f'{_f(doc["settings"].get("gap_warn_dn"), 0)}, a large gap up to '
                   f'{_f(doc["settings"].get("gap_serious_dn"), 0)} (DexVantage defaults, not standards). '
                   f'{_f(s.get("explained_pct"), 0)} % of it is explained by the corrections found.')
            unit = f'DN · {s.get("sensor", "")}' + (f' · {_f(s.get("remaining_dn"))} DN left after the corrections'
                                                    if s.get('remaining_dn') is not None else '')
            self.head.set(_f(s.get('gap_dn')), unit, s.get('status', 'warning'), s.get('word', ''),
                          s.get('advice', ''), tip)
            ds = doc.get('dataset') or {}
            fmt = format_title(ds.get('format', ''), ds.get('version', ''))
            self.dataset_label.setText(f'{ds.get("name", "")}: {fmt}'
                                       f', episode {doc.get("episode")} of {ds.get("episodes", 0)}.')
            self.cam_box.clear()
            for c in doc.get('cameras') or []:
                self.cam_box.addItem(c['sensor'], c['sensor_uid'])
            self._fill_causes(doc)
            self._show_camera()
            notes = list(doc.get('notes') or []) + [(doc.get('fid') or {}).get('why', '')]
            self.notes_label.setText('\n'.join(n for n in notes if n))
        except (KeyError, TypeError, ValueError):
            traceback.print_exc()
        self._chosen = (doc or {}).get('apply')
        self._apply_state()

    def _fill_causes(self, doc):
        self.causes.clear()
        rank = 0
        for i, c in enumerate(doc.get('causes') or []):
            ranked = c.get('reduction_dn') is not None
            rank += ranked
            it = QTreeWidgetItem([str(rank) if ranked else '', c['title'],
                                  str(c.get('camera') or '').replace('observation.images.', ''),
                                  _f(c.get('reduction_dn'), 1, 'DN') if ranked else 'not ranked',
                                  _f(c.get('share_pct'), 0, '%') if ranked else '—', c.get('figure') or '',
                                  c.get('page_title') or ''])
            it.setToolTip(1, c.get('text') or '')
            it.setToolTip(3, 'The Shapley share: the image difference this correction removes, averaged over every '
                             'order of applying the corrections found.')
            if not c.get('apply'):
                it.setForeground(1, QColor(STATUS['warning']))
            it.setData(0, Qt.ItemDataRole.UserRole, i)
            self.causes.addTopLevelItem(it)

    def _choose(self, i):
        rows = (self.doc or {}).get('causes') or []
        if i is None or not 0 <= int(i) < len(rows):
            return
        c = rows[int(i)]
        self.cause_text.setText(c.get('text') or '')
        self._chosen = c.get('apply')
        self._link = c.get('page')
        self.b_page.setEnabled(bool(self._link))
        self._apply_state()

    def _camera_doc(self):
        uid = self.cam_box.currentData()
        return next((c for c in (self.doc or {}).get('cameras') or [] if c['sensor_uid'] == uid), None)

    def _show_camera(self):
        c = self._camera_doc()
        self.frame_box.clear()
        self.figs.clear()
        if c is None:
            return
        for t in c.get('thumbs') or []:
            self.frame_box.addItem(f'frame {t["frame"]} · {t["time_s"]:.2f} s')
        self._show_frames()
        self._fill_figures(c)
        top = next((x for x in (self.doc or {}).get('causes') or [] if x.get('sensor_uid') == c['sensor_uid']
                    or c['sensor_uid'] in (x.get('cameras') or [])), None)
        if top is not None:
            self.cause_text.setText(top.get('text') or '')
            self._link = top.get('page')
            self.b_page.setEnabled(bool(self._link))

    def _show_frames(self):
        c = self._camera_doc()
        th = (c or {}).get('thumbs') or []
        k = self.frame_box.currentIndex()
        if not th or not 0 <= k < len(th):
            return
        t = th[k]
        real = _decode(t['real'])
        sim = _decode(t['corrected' if self.mode_box.currentIndex() == 1 else 'planned'])
        if real is None or sim is None:
            return
        diff = np.abs(real.astype(np.float32) - sim.astype(np.float32)).mean(-1)
        heat = np.clip(diff * DIFF_GAIN, 0, 255)
        if self.overlay.isChecked():
            a = (heat / 255.0)[..., None]
            shown = real.astype(np.float32) * (1 - a) + np.array([229, 72, 77], np.float32) * a
        else:
            shown = real
        self.pics['real'].setPixmap(_pixmap(shown))
        self.pics['sim'].setPixmap(_pixmap(sim))
        self.pics['diff'].setPixmap(_pixmap(np.repeat(heat[..., None], 3, -1)))
        fixed = self.mode_box.currentIndex() == 1
        key = 'mad_corrected_dn' if fixed else 'mad_planned_dn'
        self.frame_note.setText(f'Frame {t["frame"]} at {t["time_s"]:.2f} s: mean absolute difference '
                                f'{_f(t.get(key))} DN ({"with every correction" if fixed else "as planned"}).')

    def _fill_figures(self, c):
        ph = c.get('photometric') or {}
        rec, ren = ph.get('recorded') or {}, ph.get('rendered') or {}

        def item(parent, name, a='', b='', g='', tip=''):
            it = QTreeWidgetItem([name, a, b, g])
            if tip:
                it.setToolTip(0, tip)
            (parent.addChild(it) if parent is not None else self.figs.addTopLevelItem(it))
            return it
        p = item(None, 'Photometric')
        item(p, 'Mean (grey)', _f(rec.get('mean')), _f(ren.get('mean')), _f(ph.get('mean_gap_dn'), 1, 'DN'))
        item(p, 'Mean R, G, B', ', '.join(_f(x, 0) for x in rec.get('mean_rgb') or []),
             ', '.join(_f(x, 0) for x in ren.get('mean_rgb') or []), '')
        item(p, 'RMS contrast', _f(rec.get('contrast')), _f(ren.get('contrast')),
             f'× {_f(ph.get("contrast_ratio"), 2)}')
        item(p, 'Histogram EMD R, G, B', '', '', ', '.join(_f(x, 1) for x in ph.get('emd_dn') or []) + ' DN',
             'Earth mover\'s distance (Wasserstein-1) of each channel\'s histogram, grey levels')
        sh = ph.get('sharpness') or {}
        item(p, 'Sharpness (Laplacian variance)', _f(sh.get('recorded'), 0), _f(sh.get('rendered'), 0),
             f'× {_f(sh.get("ratio"), 2)}; blur σ {_f(ph.get("blur_px_full"), 2)} px')
        nz = ph.get('noise_dn') or {}
        item(p, 'Noise σ (Immerkær)', _f(nz.get('recorded'), 2), _f(nz.get('rendered'), 2), _f(nz.get('gap'), 2, 'DN'))
        item(p, 'Exposure fit (gain, offset)', '', '',
             ', '.join(_f(x, 2) for x in ph.get('gain') or []) + ' · ' +
             ', '.join(_f(x, 0) for x in ph.get('offset_dn') or []) + f' DN ({_f(ph.get("mid_grey_change_dn"), 0)} DN, '
             f'{_f(ph.get("exposure_stops"), 2)} stops at mid grey)')
        g = item(None, 'Geometric')
        si = c.get('silhouette') or {}
        item(g, 'Silhouette IoU before → after', '', '',
             f'{_f(si.get("iou_before"), 2)} → {_f(si.get("iou_after"), 2)}' if si.get('method') else
             si.get('why', '—'), si.get('method') or '')
        m = c.get('mount') or {}
        item(g, 'Mount offset', '', '', f'{_f(m["delta"]["norm_mm"], 1)} mm, {_f(m["delta"]["angle_deg"], 2)}°'
             if m.get('observable') else (m.get('why') or '—'))
        if c.get('depth'):
            d = c['depth']
            dd = item(None, 'Depth')
            item(dd, 'Valid pixels', _f(d.get('measured_valid_pct'), 1, '%'), _f(d.get('predicted_valid_pct'), 1, '%'),
                 _f(d.get('valid_gap_pct'), 1, '%') if d.get('measured_valid_pct') is not None else d.get('why', ''),
                 d.get('range_note', ''))
            e = d.get('robot_error_mm')
            if e:
                item(dd, 'Depth error on the robot', '', '', f'median {_f(e["median"])} mm, RMS {_f(e["rms"])} mm')
        tm = c.get('timing') or {}
        t = item(None, 'Timing')
        lat = c.get('latency') or {}
        lat_txt = (_f(tm.get('latency_ms'), 1, 'ms') + f' ± {_f(tm.get("latency_sigma_ms"), 1)}'
                   if lat.get('observable') else (lat.get('why') or '—'))
        pred = (tm.get('plan') or {}).get('predicted_offset_ms')
        item(t, 'Latency (content − stamp)', lat_txt, f'{_f(pred[0], 1)} to {_f(pred[1], 1)} ms' if pred else '—',
             (tm.get('plan') or {}).get('verdict', ''), 'Rendered column: the time-sync study\'s predicted offset')
        item(t, 'Stamp jitter', _f(tm.get('jitter_ms'), 2, 'ms'), _f(tm.get('predicted_spread_ms'), 2, 'ms'), '',
             'Recorded: robust spread of the stamps about a regular grid; rendered column: the time-sync study\'s '
             'predicted skew spread')
        mm = c.get('mmd') or {}
        d = item(None, 'Distribution (kernel MMD, not FID)')
        for k, name in (('planned', 'as planned'), ('corrected', 'with every correction')):
            r = mm.get(k) or {}
            item(d, f'MMD² {name}', '', '', f'{_f(r.get("mmd2"), 4)} (p {_f(r.get("p_value"), 3)})',
                 f'{mm.get("features", "")}; {r.get("kernel", "")}; {r.get("permutations", 0)} permutations')
        self.figs.expandAll()

    # ---------------------------------------------------------------- Apply / Preview
    def _apply_state(self):
        ch = self._chosen
        ok = bool(ch) and self.doc is not None
        if ok and ch.get('kind') == 'mount':
            s = self.ctx.store.project.variant.sensor(ch['sensor_uid'])
            ok = s is not None and not s.locked
        self.suggestion_ready(ok)

    def apply(self):
        ch = self._chosen
        if not ch:
            return None
        p = self.ctx.store.project
        win = self.ctx.window
        try:
            G.apply_change(p.snapshot(), ch)              # refused on a copy first: nothing half done
        except G.GapError as e:
            if win is not None:
                win.msg(str(e), 8000)
            return None
        text = G.describe_change(p, ch).replace('Apply would ', '').rstrip('.')
        self.ctx.store.edit(G.sections_of(ch), lambda pr: G.apply_change(pr, ch), f'Sim-to-real gap · {text}')
        self.suggestion_ready(False)
        if win is not None:
            win.msg(f'Applied: {text}. Run the gap again to see what is left.', 8000)
        return ch

    def preview(self):
        if self.ctx.window is not None:
            self.ctx.window.msg(G.describe_change(self.ctx.store.project, self._chosen) if self._chosen else
                                'Nothing to change.', 8000)

    def open_linked_page(self):
        pid = getattr(self, '_link', None)
        win = self.ctx.window
        if pid and win is not None and pid in getattr(win, 'pages', {}):
            win.open_page(pid)

    def explain(self):
        box = QMessageBox(self)
        box.setWindowTitle('How the sim-to-real gap is measured')
        box.setText(
            'The episode is rendered again in MuJoCo through each camera\'s lens and mount with the robot at the '
            'recorded joint states. Latency: renders at the frame\'s stamp plus τ on a grid, the τ whose gradient '
            'image correlates best with the recorded frame. Mount: the silhouette calibration (static cameras). '
            'Blur: the Gaussian that brings the render\'s Laplacian variance to the recording\'s. Exposure: a gain '
            'and offset per channel that line up the two histograms.\n\n'
            'The gap is the mean absolute difference of the frames (grey levels). Each correction\'s share is its '
            'Shapley value: what it removes, averaged over every order of applying the corrections, so the shares '
            'add up to what all of them remove. What is left is the scene itself (textures, shadows, objects the '
            'project does not model).\n\n'
            'Also shown: each channel\'s histogram EMD, the noise σ (Immerkær 1996), the robot silhouette IoU '
            '(background difference: the dataset stores no masks), depth where a reader decodes it, the stamps\' '
            'jitter against the time-sync study, and a kernel MMD between the frame sets (an RBF kernel on 32 × 24 '
            'thumbnails, not FID). Every threshold is a DexVantage default.')
        box.setModal(False)
        box.show()


register('sim2real', 'Sim-to-real gap', Sim2RealPage, menu='Data', icon='compare', action_id=None, workspace='audit',
         order=83)
