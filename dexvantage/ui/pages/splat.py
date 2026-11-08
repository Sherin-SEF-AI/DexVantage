"""The Splat views page (spec §9.3, Phase 12): a robot-aligned 3D Gaussian Splatting rig fitted to a recorded episode,
or to a MuJoCo render of the project for demonstration (core/sim/splat_views.py, core/studies/splat_views.py).

What it shows: the rig's PSNR on held-out frames (views the fit did not use) with a status word against DexVantage's
default thresholds (25, 20, 15 dB), SSIM and LPIPS beside it, the same on held-out joint states, and the before/after
view of the target camera: its current mount against the proposed pose (an offset and turn in the sensor's own frame,
entered here), with the recorded frame beside them when the episode records that camera. Every splat image carries a
"synthetic" badge. Write replay… renders the episode at its logged q(t) through the proposed camera to a video and
per-frame images; Write augmented… writes perturbed copies of the episode (§8.5's mount tolerances) as a LeRobot
dataset; both mark every render synthetic. Get LPIPS weights… downloads AlexNet's weights after asking. Apply moves
the camera's mount to the proposed pose (one undo step); Preview says what it would change.

The page needs the splat extra (PyTorch); without it the page says so and its actions are off. The fit runs on a
CUDA GPU; without one the study refuses unless the device is set to cpu.

Known limits: see the study's.
"""
from __future__ import annotations

import base64
import os
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QTreeWidget,
                             QTreeWidgetItem, QVBoxLayout, QWidget)

from ...core.sim import splat_views as SV
from ..widgets import ScrubSpinBox, button, label
from . import register
from .base import StudyPage, row

THUMB_W = 240
SOURCES = (('auto', 'Auto (the episode when a dataset is chosen)'), ('episode', 'Recorded episode'),
           ('demo', 'MuJoCo render of the project (demonstration)'))
DEVICES = (('auto', 'CUDA GPU (auto)'), ('cpu', 'Processor (slow)'))


_CUDA: dict = {}                    # {'ok': bool} once asked, {'asking': True} meanwhile


def _probe_cuda(sig):
    """Ask whether PyTorch sees a CUDA device, in a child process (core/cuda_probe.py says why); the answer arrives
    as sig.cuda."""
    import subprocess
    import sys
    _CUDA['asking'] = True

    def work():
        try:                                             # core/cuda_probe.py; a bundle's launcher runs it too
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            r = subprocess.run([sys.executable, '-m', 'dexvantage.core.cuda_probe'], cwd=root, capture_output=True,
                               timeout=120)
            ok = r.returncode == 0
        except Exception:                                # noqa: BLE001 — a broken install or a timeout: no CUDA
            ok = False
        _CUDA['ok'] = ok
        try:
            sig.cuda.emit()
        except RuntimeError:                             # the page has gone
            pass
    threading.Thread(target=work, daemon=True).start()


class _Sig(QObject):
    cuda = pyqtSignal()
    done = pyqtSignal(str, object)          # kind, result (dict) or the error sentence (str)
    progress = pyqtSignal(str)


class _Job:
    """A replay or augmentation being written: cancel() stops it at the next frame; join/is_alive as for a thread."""

    def __init__(self):
        self.stop = threading.Event()
        self.ended = threading.Event()

    def cancel(self):
        self.stop.set()

    def is_alive(self) -> bool:
        return not self.ended.is_set()

    def join(self, timeout: float | None = None):
        self.ended.wait(timeout)


def _f(x, nd=1, unit=''):
    if x is None:
        return '—'
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    return (f'{x:.{nd}f}' if np.isfinite(x) else '—') + (f' {unit}' if unit else '')


def _decode(b64: str) -> np.ndarray | None:
    import cv2
    if not b64:
        return None
    a = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    return None if a is None else cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def badge_pixmap(rgb: np.ndarray, synthetic: bool, width: int = THUMB_W) -> QPixmap:
    """The image as a pixmap of the given width; a splat render gets a 'synthetic' badge in its top-left corner."""
    a = np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))
    h, w = a.shape[:2]
    img = QImage(a.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    pm = QPixmap.fromImage(img)
    if w != width:
        pm = pm.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)
    if synthetic:
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        f = QFont()
        f.setPixelSize(11)
        f.setBold(True)
        p.setFont(f)
        r = QRectF(4, 4, p.fontMetrics().horizontalAdvance('synthetic') + 12, 17)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(20, 20, 24, 200))
        p.drawRoundedRect(r, 4, 4)
        p.setPen(QColor('#fab219'))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, 'synthetic')
        p.end()
    return pm


def _spin(axis, lo, hi, step, unit, tip):
    s = ScrubSpinBox(axis, 'plain', 1, unit, step=step, width=88)
    s.setRange(lo, hi)
    s.setToolTip(tip)
    return s


class SplatPage(StudyPage):
    """Splat views: a 3D Gaussian Splatting scene of the cell, fitted to a recorded episode (or to a MuJoCo render
    of the project, a synthetic demonstration), whose robot Gaussians follow the joint state. The figure is its PSNR
    on held-out frames, the views the fit did not use; the images show what a proposed camera pose would see. Every
    splat image is a render, marked synthetic."""
    STUDY = SV.SID
    TITLE = 'Splat views'
    HEADLINE = 'Reconstruction on held-out frames (PSNR)'

    def build_body(self, layout):
        self._sig = _Sig()
        self._sig.done.connect(self._done)
        self._sig.progress.connect(lambda t: self.note.setText(t))
        self._sig.cuda.connect(self._sync)
        self._busy = False                  # a replay, an augmentation or the LPIPS download is running
        self._fitting = False               # the study (the fit) is running
        self._job = None
        self._src_stored = object()         # the stored (source, device) the Fit from and Device boxes last showed
        self.b_cancel = button('Cancel', role='ghost', icon_name='close',
                               tip='Stop the fit, replay or augmentation now; nothing half-done is kept (Ctrl+.)')
        self.b_cancel.clicked.connect(lambda: self.cancel())
        self.b_cancel.hide()
        self.head.actions.insertWidget(1, self.b_cancel)
        an = self.ctx.analysis
        an.started.connect(lambda sid: self._study_state(sid, True))
        an.finished.connect(lambda sid, _d: self._study_state(sid, False))
        an.failed.connect(lambda sid, _m: self._study_state(sid, False))
        self.avail = label('', wrap=True)
        layout.addWidget(self.avail)
        self.src_box = QComboBox()
        for k, t in SOURCES:
            self.src_box.addItem(t, k)
        self.src_box.setToolTip('Where the rig is fitted from: a recorded episode with enough view diversity (a '
                                'camera that moves), or a MuJoCo render of the project for demonstration '
                                '(labelled synthetic)')
        self.b_dataset = button('Dataset…', role='ghost', icon_name='dataset',
                                tip='Choose the recorded dataset (a LeRobot folder)')
        self.b_dataset.clicked.connect(lambda: self.choose_dataset())
        self.dataset_label = label('', muted=True, wrap=True)
        self.dev_box = QComboBox()
        for k, t in DEVICES:
            self.dev_box.addItem(t, k)
        self.dev_box.setToolTip('Where the fit runs: a CUDA GPU, or the processor, which is far slower (the fit stops '
                                'at its time budget, 120 s by default, with what it has fitted)')
        layout.addWidget(row(label('Fit from'), self.src_box, label('Device'), self.dev_box, self.b_dataset,
                             self.dataset_label))
        self.cam_box = QComboBox()
        self.cam_box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.cam_box.setToolTip('The camera whose pose you propose to change')
        self.dx = [_spin(a, -500, 500, 1.0, 'mm', f'Proposed move along the sensor\'s own {a} axis (mm; drag the '
                                                  'letter to scrub)') for a in 'xyz']
        self.dr = [_spin(a, -180, 180, 0.5, '°', f'Proposed turn about the sensor\'s own {a} axis ({n}, degrees)')
                   for a, n in zip('xyz', ('roll', 'pitch', 'yaw'))]
        layout.addWidget(row(label('Camera'), self.cam_box, label('Move'), *self.dx, label('Turn'), *self.dr))
        for w in self.dx + self.dr:
            w.valueChanged.connect(lambda _v: self._apply_state())
            w.committed.connect(lambda _v: self._apply_state())
        self.cam_box.activated.connect(lambda _i: self._apply_state())
        pics = QWidget()
        h = QHBoxLayout(pics)
        h.setContentsMargins(0, 0, 0, 0)
        self.pics, self.pic_boxes = {}, {}
        for k, title in (('recorded', 'Recorded'), ('current', 'Current mount (splat render)'),
                         ('proposed', 'Proposed pose (splat render)')):
            box = QWidget()
            v = QVBoxLayout(box)
            v.setContentsMargins(0, 0, 0, 0)
            v.addWidget(label(title, small=True, muted=True))
            pic = QLabel()
            pic.setMinimumSize(80, 45)
            pic.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            v.addWidget(pic)
            v.addStretch(1)
            self.pics[k] = pic
            self.pic_boxes[k] = box
            h.addWidget(box)
        h.addStretch(1)
        self.frame_box = QComboBox()
        self.frame_box.setToolTip('The frame of the preview')
        self.frame_box.activated.connect(lambda _i: self._show_frame())
        layout.addWidget(row(label('Before and after', bold=True), self.frame_box))
        layout.addWidget(pics)
        self.frame_note = label('', muted=True, small=True, wrap=True)
        layout.addWidget(self.frame_note)
        self.stale_label = label('The images are for the camera and pose of the last run: Run to render the ones '
                                 'entered (the fitted rig is reused, so it takes seconds).', small=True, wrap=True)
        self.stale_label.hide()
        layout.addWidget(self.stale_label)
        layout.addWidget(label('Reconstruction quality on frames the fit did not use', bold=True))
        self.qual = QTreeWidget()
        self.qual.setRootIsDecorated(False)
        self.qual.setHeaderLabels(['Frames', 'PSNR', 'SSIM', 'LPIPS', 'Robot PSNR', 'Views'])
        self.qual.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.qual.setMinimumHeight(110)
        layout.addWidget(self.qual)
        self.b_replay = button('Write replay…', role='ghost', icon_name='export',
                               tip='Render the episode at its logged joint states through the proposed camera: a '
                                   'video and per-frame images, every one marked synthetic')
        self.b_augment = button('Write augmented…', role='ghost', icon_name='export',
                                tip='Perturbed copies of the episode through the mounts\' tolerances (§8.5), as a '
                                    'LeRobot dataset with every render marked synthetic')
        self.b_lpips = button('Get LPIPS weights…', role='ghost',
                              tip='Download AlexNet\'s ImageNet weights for LPIPS (asks first)')
        self.b_replay.clicked.connect(lambda: self.write('replay'))
        self.b_augment.clicked.connect(lambda: self.write('augment'))
        self.b_lpips.clicked.connect(lambda: self.get_lpips())
        layout.addWidget(row(self.b_replay, self.b_augment, self.b_lpips))
        self.notes_label = label('', muted=True, small=True, wrap=True)
        layout.addWidget(self.notes_label)
        st = self.ctx.store
        st.changed.connect(lambda _s: self._sync())
        st.project_replaced.connect(self._replaced)
        self._sync()

    def showEvent(self, e):
        super().showEvent(e)
        if 'ok' not in _CUDA and 'asking' not in _CUDA and SV.requirement() is None:   # once, when first shown
            _probe_cuda(self._sig)

    # ---------------------------------------------------------------- state
    def _avail(self) -> str | None:
        return SV.requirement()

    def _sync(self):
        why = self._avail()
        if why:
            self.avail.setText('')          # the headline says it (empty()), once
        elif _CUDA.get('ok') is False:
            self.avail.setText('There is no CUDA device here: set Device to Processor to fit on the processor (slow: '
                               'the fit stops at its time budget, 120 s by default, with what it has fitted).')
        else:
            self.avail.setText('')
        self.avail.setVisible(bool(self.avail.text()))
        for b in (self.b_run, self.b_replay, self.b_augment, self.b_lpips):
            b.setEnabled(why is None and not self._busy and not self._fitting)
        if why is None and SV.lpips_state() != 'no weights':
            self.b_lpips.setEnabled(False)
        self.b_cancel.setVisible(self._fitting or self._job is not None)
        p = self.ctx.store.project
        st = p.analysis.studies.get(SV.SID) or {}
        cur = self.cam_box.currentData()
        self.cam_box.clear()
        for s in p.variant.sensors:
            if s.enabled and s.kind in ('camera', 'depth'):
                self.cam_box.addItem(s.name + (' (on a link)' if s.mount.on_link else ''), s.uid)
        want = cur or st.get('target_sensor')
        if want:
            i = self.cam_box.findData(want)
            if i >= 0:
                self.cam_box.setCurrentIndex(i)
        stored = (st.get('source') or 'auto', 'cpu' if st.get('device') == 'cpu' else 'auto')
        if stored != self._src_stored:      # only when the project's own settings change: the choice made in the
            self._src_stored = stored       # boxes stays through other edits (choosing the dataset among them)
            self.src_box.setCurrentIndex(max(0, self.src_box.findData(stored[0])))
            self.dev_box.setCurrentIndex(max(0, self.dev_box.findData(stored[1])))
        try:
            path = SV.dataset_path(p, SV.settings_of(p))
            self.dataset_label.setText(os.path.basename(os.path.normpath(path)))
        except Exception:                                # noqa: BLE001 — none chosen: say so
            self.dataset_label.setText('none chosen: Auto fits a MuJoCo render of the project (demonstration)')

    def _replaced(self):
        self._src_stored = object()         # a new project: show its stored source
        self._clear_result()
        self._sync()
        self.empty()

    def _clear_result(self):
        self.doc = None
        self.qual.clear()
        for pic in self.pics.values():
            pic.clear()
        self.frame_box.clear()
        self.frame_note.setText('')
        self.notes_label.setText('')
        self._apply_state()

    def empty(self):
        why = self._avail() if hasattr(self, 'avail') else SV.requirement()
        if why:
            self.head.set('—', '', 'warning', 'splat extra not installed', why)
        else:
            super().empty()

    def form(self) -> dict:
        """The settings the page's controls give, for the next run (the stored ones underneath)."""
        d = {'source': self.src_box.currentData() or 'auto', 'device': self.dev_box.currentData() or 'auto',
             'target_offset_m': [round(s.value() / 1000.0, 6) for s in self.dx],
             'target_rpy_deg': [round(s.value(), 4) for s in self.dr]}
        if self.cam_box.currentData():
            d['target_sensor'] = self.cam_box.currentData()
        return d

    def set_proposal(self, offset_m=(0.0, 0.0, 0.0), rpy_deg=(0.0, 0.0, 0.0), sensor: str | None = None):
        """Enter a proposed pose change (scripts and tests; the spin boxes otherwise)."""
        for s, v in zip(self.dx, offset_m):
            s.set_value(float(v) * 1000.0)
        for s, v in zip(self.dr, rpy_deg):
            s.set_value(float(v))
        if sensor:
            i = self.cam_box.findData(sensor)
            if i < 0:
                i = next((k for k in range(self.cam_box.count()) if sensor in self.cam_box.itemText(k)), -1)
            if i >= 0:
                self.cam_box.setCurrentIndex(i)
        self._apply_state()

    def choose_dataset(self, path: str | None = None):
        if path is None:
            path = QFileDialog.getExistingDirectory(self, 'Choose the recorded dataset')
            if not path:
                return None
        path = str(path)

        def fn(p):
            d = dict(p.analysis.studies.get(SV.SID) or {})
            d['dataset'] = path
            p.analysis.studies[SV.SID] = d
        self.ctx.store.edit('analysis', fn, f'Splat views · dataset {os.path.basename(os.path.normpath(path))}')
        return path

    def run(self):
        why = self._avail()
        if why:
            self.head.set('—', '', 'warning', 'splat extra not installed', why)
            return
        if self._fitting or self._busy:     # one at a time: a second click would queue a second fit
            return
        self.ctx.analysis.run(SV.SID, self.form())

    def _study_state(self, sid, running: bool):
        if sid != self.STUDY:
            return
        self._fitting = running
        self._sync()

    def cancel(self):
        """Stop the running fit, replay or augmentation (nothing half-done is kept)."""
        if self._fitting:
            self.ctx.analysis.cancel(self.STUDY)
        if self._job is not None:
            self._job.cancel()
        self.note.setText('Stopping…')

    def _failed(self, sid, message):
        if sid != self.STUDY:
            return
        from ..store import failure_text
        text, _explained = failure_text(message, self.TITLE)
        if 'cancelled' in text and self.doc is not None:          # the result shown before stays, and says so
            self.show_result(self.doc)
            self.note.setText(text + ' The result shown is the previous run\'s.')
        elif 'cancelled' in text:
            self.head.set('—', '', 'warning', 'cancelled', text + ' Run starts the fit again.')
            self.note.setText('')
        else:
            self._clear_result()                                  # never an older result under "could not run"
            self.head.set('—', '', 'critical', 'could not run', text)
            self.note.setText('')

    # ---------------------------------------------------------------- result
    def show_result(self, doc):
        try:
            h = doc.get('headline') or {}
            q = doc.get('quality') or {}
            st = doc.get('settings') or {}
            tip = ('PSNR of the splat rig\'s renders against the frames the fit did not use (every '
                   f'{st.get("holdout_every", 8)}th view), mean over those views; SSIM (Wang et al. 2004) and LPIPS '
                   '(Zhang et al. 2018) beside it. Status: close up to '
                   f'{_f(st.get("psnr_good_db"), 0)} dB, visible differences down to {_f(st.get("psnr_warn_db"), 0)}, '
                   f'large differences down to {_f(st.get("psnr_serious_db"), 0)} dB (DexVantage defaults, not '
                   'standards). Every splat image is synthetic.')
            unit = 'dB · SSIM ' + _f(h.get('ssim'), 3) + ' · LPIPS ' + (_f(h.get('lpips'), 3) if h.get('lpips')
                                                                         is not None else 'not computed')
            self.head.set(_f(h.get('value')), unit, h.get('status', 'warning'), h.get('word', ''),
                          h.get('advice', ''), tip)
            src = doc.get('source') or {}
            self.dataset_label.setText(src.get('label', ''))
            self.qual.clear()
            for key, name in (('held_out_views', 'Held-out views'), ('held_out_q', 'Held-out joint states'),
                              ('train_views', 'Training views (for comparison)'),
                              ('before_fit', 'Held-out views before the fit')):
                r = q.get(key)
                if not r:
                    continue
                it = QTreeWidgetItem([name, _f(r.get('psnr_db'), 1, 'dB'), _f(r.get('ssim'), 3),
                                      _f(r.get('lpips'), 3) if r.get('lpips') is not None else '—',
                                      _f(r.get('robot_psnr_db'), 1, 'dB'), str(r.get('views', ''))])
                if r.get('how'):
                    it.setToolTip(0, r['how'])
                self.qual.addTopLevelItem(it)
            pv = doc.get('preview') or {}
            self.frame_box.clear()
            for k, fr in enumerate(pv.get('frames') or []):
                self.frame_box.addItem(f'frame {k + 1}' + (f' · {fr["time_s"]:.2f} s' if fr.get('time_s') is not None
                                                           else ''))
            self._show_frame()
            self._apply_state()
            notes = list(doc.get('notes') or []) + [q.get('lpips_note') or '', q.get('how') or '']
            self.notes_label.setText('\n'.join(dict.fromkeys(n for n in notes if n)))     # each note once
        except (KeyError, TypeError, ValueError):
            traceback.print_exc()

    def _show_frame(self):
        pv = (self.doc or {}).get('preview') or {}
        frs = pv.get('frames') or []
        k = self.frame_box.currentIndex()
        for pic in self.pics.values():
            pic.clear()
        if not frs or not 0 <= k < len(frs):
            self.frame_note.setText(pv.get('why', ''))
            return
        fr = frs[k]
        for key in ('recorded', 'current', 'proposed'):
            img = _decode(fr.get(key, ''))
            if img is not None:
                self.pics[key].setPixmap(badge_pixmap(img, key != 'recorded'))
            else:
                self.pics[key].setText('not recorded' if key == 'recorded' else '—')
        recorded = any(f.get('recorded') for f in frs)
        self.pic_boxes['recorded'].setVisible(recorded)       # a demonstration fit has no recording to show
        self.frame_note.setText(f'{pv.get("sensor", "")}: {pv.get("proposal", "")}.' + (
            '' if recorded else ' No recorded frame of this camera to show beside the renders.'))

    def page_context(self) -> dict:
        out = super().page_context()
        d = self.doc or {}
        q = d.get('quality') or {}
        out['quality'] = {k: {x: (q.get(k) or {}).get(x) for x in ('psnr_db', 'ssim', 'lpips', 'views')}
                          for k in ('held_out_views', 'held_out_q') if q.get(k)}
        out['synthetic'] = True
        return out

    # ---------------------------------------------------------------- Apply / Preview: the proposed mount
    def _proposed(self):
        """(sensor, new mount pose) of the pose change entered, or (None, why)."""
        from ...core.frames import Pose
        uid = self.cam_box.currentData()
        s = self.ctx.store.project.variant.sensor(uid) if uid else None
        f = self.form()
        if s is None:
            return None, 'Choose a camera first.'
        if not any(f['target_offset_m']) and not any(f['target_rpy_deg']):
            return None, 'Enter a move or a turn first: the proposed pose is the current one.'
        if s.locked:
            return None, f'{s.name} is locked: unlock it to move its mount.'
        D = SV._delta(SV.settings_of(self.ctx.store.project, f))
        return s, Pose.from_matrix(s.mount.pose.matrix() @ D)

    def _apply_state(self):
        s, _ = self._proposed()
        self.suggestion_ready(s is not None and self.doc is not None)
        self.stale_label.setVisible(self._proposal_stale())

    def _proposal_stale(self) -> bool:
        """True when the camera, move or turn entered differ from those the images shown were rendered for."""
        st = (self.doc or {}).get('settings')
        if not st:
            return False
        f = self.form()
        try:
            same = (np.allclose(st.get('target_offset_m') or [0, 0, 0], f['target_offset_m'], atol=1e-9)
                    and np.allclose(st.get('target_rpy_deg') or [0, 0, 0], f['target_rpy_deg'], atol=1e-9))
        except (TypeError, ValueError):
            return False
        want = st.get('target_sensor')
        return not same or bool(want and f.get('target_sensor') and want != f['target_sensor'])

    def apply(self):
        """Move the camera's mount to the proposed pose (one undo step) and clear the entered change."""
        s, pose = self._proposed()
        if s is None:
            if self.ctx.window is not None:
                self.ctx.window.msg(pose, 6000)
            return None
        uid = s.uid

        def fn(pr):
            pr.variant.sensor(uid).mount.pose = pose
        self.ctx.store.edit('variant', fn, f'Move {s.name} as proposed on the Splat views page')
        self.set_proposal()
        self._apply_state()
        return pose

    def preview(self):
        s, pose = self._proposed()
        if self.ctx.window is not None:
            self.ctx.window.msg(pose if s is None else
                                f'Apply would move {s.name}\'s mount ({s.mount.parent_frame}) to position '
                                f'{", ".join(f"{1000 * x:.1f}" for x in pose.position)} mm: '
                                f'{SV._proposal_words(SV.settings_of(self.ctx.store.project, self.form()))}.', 8000)

    # ---------------------------------------------------------------- files
    def write(self, kind: str, folder: str | None = None):
        """Write replay or augment into folder (a dialog when None), in the background."""
        why = self._avail()
        if why:
            self.note.setText(why)
            return None
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Choose the folder to write into')
            if not folder:
                return None
        snap = self.ctx.store.project.snapshot()
        snap.analysis.studies[SV.SID] = {**(snap.analysis.studies.get(SV.SID) or {}), **self.form()}
        fn = SV.replay if kind == 'replay' else SV.augment
        title = 'Writing the splat replay' if kind == 'replay' else 'Writing the augmented dataset'
        job = _Job()

        def work(win_progress=None, win_cancel=None):
            def prog(f, text=''):
                self._sig.progress.emit(f'{"Replay" if kind == "replay" else "Augmentation"} · {f * 100:.0f} % · '
                                        f'{text}')
                if win_progress is not None:
                    win_progress(f, text)
            try:
                res = fn(snap, folder, progress=prog,
                         cancel=lambda: job.stop.is_set() or bool(win_cancel is not None and win_cancel()))
            except SV.StudyError as e:
                res = str(e)
            except Exception as e:                        # noqa: BLE001 — explained, never a traceback in the UI
                traceback.print_exc()
                res = f'It stopped with an unexpected error ({type(e).__name__}); the details are in the log.'
            finally:
                job.ended.set()
            return res

        win = self.ctx.window
        if win is not None and hasattr(win, 'run_job'):   # the status bar shows it, with Cancel and Ctrl+.
            if not win.run_job(f'splat_{kind}', title, work, lambda res: self._done(kind, res),
                               lambda m: self._done(kind, m.splitlines()[0] if m else 'It stopped.')):
                self.note.setText(f'{title} is running already.')
                return None
        else:
            threading.Thread(target=lambda: self._sig.done.emit(kind, work()), daemon=True).start()
        self._job = job
        self._busy = True
        self._sync()
        self.note.setText(f'{title}…')
        return job

    def get_lpips(self, ask: bool = True):
        if ask:
            box = QMessageBox(self)
            box.setWindowTitle('Get the LPIPS weights')
            box.setText(SV.LPIPS_ABOUT + ' They are downloaded from ' + SV.LPIPS_URL + ' into DexVantage\'s cache '
                        'folder and used only to compute LPIPS. Download them now?')
            box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if box.exec() != QMessageBox.StandardButton.Yes:
                return None
        self._busy = True
        self._sync()
        self.note.setText(f'Downloading the AlexNet weights ({SV.LPIPS_MB} MB)…')

        def work():
            try:
                res = {'path': SV.fetch_lpips_weights()}
            except SV.StudyError as e:
                res = str(e)
            self._sig.done.emit('lpips', res)
        t = threading.Thread(target=work, daemon=True)
        t.start()
        self._thread = t
        return t

    def _done(self, kind, res):
        self._busy = False
        if kind != 'lpips':
            self._job = None
        self._sync()
        win = self.ctx.window
        if isinstance(res, str):
            self.note.setText(res)
            if win is not None:
                win.msg(res, 10000)
            return
        if kind == 'lpips':
            text = 'The LPIPS weights are in the cache: run again to see LPIPS.'
        elif kind == 'replay':
            text = (f'Replay written ({res["frames"]} frames, {res["video"]["codec"]}, synthetic): '
                    f'{os.path.basename(res["root"])}/{res["video"]["file"]}')
        else:
            text = (f'Augmented dataset written ({res["episodes"]} perturbed copies, synthetic): '
                    f'{os.path.basename(res["root"])}')
        self.last_written = res
        self.note.setText(text)
        if win is not None:
            win.msg(text, 10000)

    def explain(self):
        box = QMessageBox(self)
        box.setWindowTitle('How splat views are made and measured')
        box.setText(
            'A 3D Gaussian Splatting scene of the cell is fitted to posed images: the frames of a recorded episode '
            '(each camera\'s pose is its mount at the frame\'s logged joint state, its depth and masks MuJoCo renders '
            'of the project\'s model) or, for demonstration, MuJoCo renders of the project. The robot\'s Gaussians '
            'are segmented per link and stored in the link\'s frame, so the scene re-poses with the joint state '
            '(the approach of SplatSim and Robo-GS; their code is not used).\n\n'
            'Quality is measured on frames the fit did not use: every 8th view, and joint states away from the '
            'capture. PSNR = 10 log10(1 / MSE) on images in [0, 1]; SSIM (Wang et al. 2004, 11 × 11 Gaussian window, '
            'σ 1.5); LPIPS (Zhang et al. 2018, AlexNet features) when its weights are in the cache.\n\n'
            'Quality falls for joint states and camera poses far from the capture, on shiny surfaces and in regions '
            'no view saw. Every splat image is a render, never a photograph, and is marked synthetic.')
        box.setModal(False)
        box.show()


register('splat', 'Splat views', SplatPage, menu='Simulate', icon='frustum', action_id=None, workspace=None, order=84)
