"""The Motion blur and rolling shutter page (spec §7.9): how far each camera's image smears during one exposure and
skews across a rolling-shutter readout along the project's trajectories (core/studies/motion_blur.py).

What it shows: the worst smear in pixels (the headline, with a status word: "sharp" within the smear limit, 1 px by
default; "some blur" up to twice it; "blurred" beyond; "exposure not set" when no exposure and no frame rate are
known), a table per camera imager (shutter, exposure and readout with where each comes from, peak image speed,
smear, skew, the longest exposure that keeps the limit), a table of the selected trajectory's segments, the worst
frames, and the settings (the limit, your exposure for a camera with its source). Apply sets the suggested exposures
as yours (one undo step); Preview says what it would change; Show frame poses the robot at a worst frame and shows its
camera, where the Camera view draws the smear vectors of every point it sees (lengthened when short, the legend says
by how much); the 3D view draws the TCP's path coloured by each segment's status.

Known limits: the overlays show the frame the study kept (its q), not an arbitrary pose; see the study's own limits.
"""
from __future__ import annotations

import traceback

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QHeaderView, QLineEdit, QMessageBox, QTreeWidget,
                             QTreeWidgetItem, QWidget)

from ...core.studies import motion_blur as MB
from ..theme import STATUS, THEME, mono
from ..widgets import ScrubSpinBox, button, label
from . import register
from .base import StudyPage, row

SEV_COLOUR = {'good': 'good', 'warning': 'warning', 'serious': 'serious', 'unknown': 'warning'}
MIN_DRAWN_PX = 30.0                # the longest smear vector of a frame is drawn at least this long


def _px(x, d=2) -> str:
    return '—' if x is None else f'{x:.{d}f}'


def _ms(x) -> str:
    return '—' if x is None else f'{x * 1000:.3g}'


def _kind(d: dict) -> str:
    k = d.get('kind')
    return {'yours': 'yours', 'stated': 'stated', 'upper bound': 'at most', 'none': 'global',
            'unknown': 'not known'}.get(k, k or '')


class MotionBlurPage(StudyPage):
    """Motion blur and rolling shutter: smear = image speed × exposure, from the camera's twist (J(q) q̇ of its link)
    and the motion of the gripper seen from fixed cameras; skew = image speed × readout for a rolling shutter. The
    headline is the worst smear over the cameras and the trajectories, against the smear limit (1 px by default)."""
    STUDY = 'motion_blur'
    TITLE = 'Motion blur and rolling shutter'
    HEADLINE = 'Worst smear in one exposure'

    # ---------------------------------------------------------------- body
    def build_body(self, layout):
        self._frame = None                 # (camera row, frame) the Camera view draws
        self._overlay3d = None
        self.traj_box = QComboBox()
        self.traj_box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)   # units not cut off
        self.traj_box.setToolTip('The trajectory whose segments are listed')
        self.traj_box.setMinimumWidth(200)
        self.traj_box.currentIndexChanged.connect(lambda _i: self._fill_segments())
        layout.addWidget(label('Cameras', bold=True))
        self.cams = QTreeWidget()
        self.cams.setRootIsDecorated(False)
        self.cams.setHeaderLabels(['Camera', 'Shutter', 'Exposure (ms)', 'Readout (ms)', 'Image speed (px/s)',
                                   'Smear (px)', 'Skew (px)', 'Longest exposure (ms)', 'Status'])
        _columns(self.cams)
        self.cams.currentItemChanged.connect(lambda *_a: self._camera_selected())
        layout.addWidget(self.cams)
        layout.addWidget(row(label('Trajectory'), self.traj_box))
        self.segs = QTreeWidget()
        self.segs.setRootIsDecorated(False)
        self.segs.setHeaderLabels(['Segment', 'Time (s)', 'Worst camera', 'Smear (px)', 'Skew (px)', 'Where',
                                   'Status'])
        _columns(self.segs)
        layout.addWidget(self.segs)
        layout.addWidget(label('Worst frames of the selected camera', bold=True))
        self.frame_box = QComboBox()
        self.frame_box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)   # units not cut off
        self.frame_box.setMinimumWidth(260)
        self.frame_box.setToolTip('A frame the study kept: the times of the largest image speed')
        self.b_frame = button('Show frame', role='ghost', icon_name='eye',
                              tip='Pose the robot at this frame and show the camera: the Camera view draws the smear '
                                  'vectors')
        self.b_frame.clicked.connect(lambda: self.show_frame())
        layout.addWidget(row(self.frame_box, self.b_frame))
        self.sugg_label = label('', muted=True, wrap=True)
        layout.addWidget(self.sugg_label)
        self.notes_label = label('', muted=True, wrap=True)
        layout.addWidget(self.notes_label)
        layout.addWidget(label('Settings (yours; defaults labelled)', bold=True))
        g = QWidget()
        grid = QGridLayout(g)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        self.limit_spin = ScrubSpinBox('', 'plain', 2, 'px', step=0.1, width=110)
        self.limit_spin.setRange(0.01, 1000.0)
        self.limit_spin.setToolTip(f'The most smear accepted (default {MB.DEFAULTS["smear_limit_px"]:g} px, a '
                                   'DexVantage default, not a standard)')
        self.limit_def = label('', muted=True, small=True)
        self.limit_spin.committed.connect(lambda v: self._set_setting('smear_limit_px', float(v),
                                                                      f'Motion blur · smear limit {float(v):g} px'))
        grid.addWidget(label('Smear limit'), 0, 0)
        grid.addWidget(self.limit_spin, 0, 1)
        grid.addWidget(self.limit_def, 0, 2)
        self.exp_cam = QComboBox()
        self.exp_cam.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)   # units not cut off
        self.exp_cam.setToolTip('The camera whose exposure you set (all its imagers)')
        self.exp_spin = ScrubSpinBox('', 'plain', 3, 'ms', step=0.1, width=110)
        self.exp_spin.setRange(0.001, 10000.0)
        self.exp_src = QLineEdit()
        self.exp_src.setPlaceholderText('where it comes from (e.g. the driver\'s setting)')
        self.b_exp = button('Set exposure', role='ghost', tip='Use this exposure for the camera (one undo step)')
        self.b_exp_clear = button('Clear', role='ghost', tip='Go back to the frame period bound (one undo step)')
        self.b_exp.clicked.connect(lambda: self.set_exposure(self.exp_cam.currentData(), self.exp_spin.value() / 1000,
                                                             self.exp_src.text()))
        self.b_exp_clear.clicked.connect(lambda: self.set_exposure(self.exp_cam.currentData(), None, ''))
        grid.addWidget(label('Your exposure'), 1, 0)
        grid.addWidget(row(self.exp_cam, self.exp_spin, self.exp_src, self.b_exp, self.b_exp_clear), 1, 1, 1, 2)
        layout.addWidget(g)
        self.show3d = QCheckBox('Show the TCP path in 3D, coloured by smear')
        self.show3d.setChecked(True)
        self.show3d.toggled.connect(lambda _on: self._redraw())
        layout.addWidget(self.show3d)
        st = self.ctx.store
        st.changed.connect(lambda _s: self._sync_settings())
        st.project_replaced.connect(self._replaced)
        st.pose_changed.connect(lambda _uid: self._redraw_camview())
        self._sync_settings()

    # ---------------------------------------------------------------- settings
    def settings(self) -> dict:
        try:
            return MB.settings_of(self.ctx.store.project)
        except Exception:                               # noqa: BLE001 — a bad stored value: the defaults
            traceback.print_exc()
            return dict(MB.DEFAULTS, defaults=list(MB.DEFAULTS), exposure_s={}, readout_s={}, trajectories=[])

    def _sync_settings(self):
        st = self.settings()
        self.limit_spin.set_value(float(st['smear_limit_px']))
        self.limit_def.setText('default' if 'smear_limit_px' in st['defaults'] else
                               f'yours (default {MB.DEFAULTS["smear_limit_px"]:g} px)')
        cur = self.exp_cam.currentData()
        self.exp_cam.blockSignals(True)
        self.exp_cam.clear()
        for s in self.ctx.store.project.variant.sensors:
            if s.kind in ('camera', 'depth'):
                mine = st['exposure_s'].get(s.uid)
                self.exp_cam.addItem(s.name + (f' · {mine["value"] * 1000:.3g} ms' if mine else ''), s.uid)
        k = self.exp_cam.findData(cur)
        self.exp_cam.setCurrentIndex(k if k >= 0 else 0)
        self.exp_cam.blockSignals(False)

    def _set_setting(self, key, value, text):
        st = self.settings()
        if st.get(key) == value:
            return

        def fn(p):
            d = dict(p.analysis.studies.get(MB.STUDY_ID) or {})
            d[key] = value
            p.analysis.studies[MB.STUDY_ID] = d
        self.ctx.store.edit('analysis', fn, text, merge_key=(MB.STUDY_ID, key))

    def set_exposure(self, sensor_uid, seconds, source: str) -> bool:
        """Set (or clear, seconds None) your exposure for a camera: one undo step."""
        p = self.ctx.store.project
        s = p.variant.sensor(sensor_uid) if sensor_uid else None
        if s is None:
            return False

        def fn(proj):
            d = dict(proj.analysis.studies.get(MB.STUDY_ID) or {})
            ex = dict(d.get('exposure_s') or {})
            if seconds is None:
                ex.pop(sensor_uid, None)
            else:
                ex[sensor_uid] = {'value': float(seconds), 'source': str(source or '').strip()}
            d['exposure_s'] = ex
            proj.analysis.studies[MB.STUDY_ID] = d
        text = (f'Motion blur · {s.name} exposure {seconds * 1000:.3g} ms' if seconds is not None else
                f'Motion blur · {s.name} exposure cleared')
        self.ctx.store.edit('analysis', fn, text)
        return True

    def _replaced(self):
        self.doc = None
        self._frame = None
        self._overlay3d = None
        for w in (self.cams, self.segs):
            w.clear()
        self.traj_box.clear()
        self.frame_box.clear()
        self.sugg_label.setText('')
        self.notes_label.setText('')
        self._sync_settings()
        self.suggestion_ready(False)
        self.empty()

    # ---------------------------------------------------------------- result
    def _headline(self, doc):
        s = doc.get('summary') or {}
        sev, word = s.get('status', 'warning'), s.get('word', '')
        lim = doc.get('smear_limit_px', 1.0)
        tip = (f'Smear limit {lim:g} px (' + ('the default' if doc.get('smear_limit_default') else 'your setting') +
               '): within it "sharp", up to twice "some blur", beyond "blurred"; "may blur" and "may be blurred" '
               'when no exposure is set and the smear is an upper bound.')
        key = s.get('worst')
        cam = next((c for c in doc.get('cameras', []) if c['key'] == key), None)
        if cam is None:
            self.head.set('—', '', sev, word, s.get('advice', ''), tip)
            return
        w = cam.get('worst') or {}
        where = (f'at t = {w.get("t", 0):.2f} s of {w.get("trajectory", "")}, {w.get("point", "")}'
                 if w else '')
        ex = cam['exposure']
        if cam['smear_px'] is None:
            self.head.set('—', 'px', 'warning', word,
                          f'{cam["name"]}: the image moves at up to {cam["peak_speed_px_s"]:.0f} px/s {where}; '
                          f'{ex["source"]}. An exposure of at most {_ms(cam["max_exposure_s"])} ms keeps the smear '
                          f'within {lim:g} px.', tip)
            return
        bound = 'up to ' if cam['smear_bound'] else ''
        text = (f'{cam["name"]} {where}: {cam["peak_speed_px_s"]:.0f} px/s × {_ms(ex["value"])} ms exposure '
                f'({_kind(ex)}). ')
        if sev != 'good':
            text += f'Keep the exposure at most {_ms(cam["max_exposure_s"])} ms for {lim:g} px.'
        else:
            text += f'Within {lim:g} px.'
        if s.get('skew_px'):
            text += (f' Rolling-shutter skew {"up to " if s.get("skew_bound") else ""}{s["skew_px"]:.2f} px '
                     f'({next((c["name"] for c in doc["cameras"] if c["key"] == s.get("worst_skew")), "")}).')
        self.head.set(f'{bound}{cam["smear_px"]:.2f}', 'px', sev, word, text, tip)

    def show_result(self, doc):
        self._headline(doc)
        cur = self._selected_key()
        self.cams.clear()
        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        for c in doc.get('cameras', []):
            ex, ro = c['exposure'], c['readout']
            it = QTreeWidgetItem([c['name'], c['shutter'] or '—', f'{_ms(ex["value"])} ({_kind(ex)})',
                                  f'{_ms(ro["value"])} ({_kind(ro)})' if ro['kind'] != 'none' else 'global',
                                  _px(c['peak_speed_px_s'], 0), _px(c['smear_px']), _px(c['skew_px']),
                                  _ms(c['max_exposure_s']), c['word'] if c['seen'] else 'not in view'])
            it.setData(0, Qt.ItemDataRole.UserRole, c['key'])
            it.setToolTip(2, ex['source'] + (f'\nStated range: {_ms(ex.get("stated_min"))} – '
                                             f'{_ms(ex.get("stated_max"))} ms ({ex.get("stated_what")})'
                                             if ex.get('stated_what') else ''))
            it.setToolTip(3, ro['source'])
            it.setToolTip(7, 'The longest exposure that keeps the smear within the limit: limit / peak image speed')
            it.setToolTip(0, c['lens'])
            for k in range(2, 8):
                it.setFont(k, mono(11))
                it.setTextAlignment(k, right)
            it.setForeground(8, QColor(STATUS[SEV_COLOUR.get(c['status'], 'warning')]))
            self.cams.addTopLevelItem(it)
            if c['key'] == cur:
                self.cams.setCurrentItem(it)
        if self.cams.currentItem() is None and self.cams.topLevelItemCount():
            key = (doc.get('summary') or {}).get('worst')
            items = [self.cams.topLevelItem(i) for i in range(self.cams.topLevelItemCount())]
            self.cams.setCurrentItem(next((i for i in items if i.data(0, Qt.ItemDataRole.UserRole) == key),
                                          items[0]))
        _fit(self.cams)
        cur_t = self.traj_box.currentData()
        self.traj_box.blockSignals(True)
        self.traj_box.clear()
        for t in doc.get('trajectories', []):
            self.traj_box.addItem(f'{t["name"]} · {t["duration_s"]:.2f} s', t['uid'])
        k = self.traj_box.findData(cur_t)
        self.traj_box.setCurrentIndex(k if k >= 0 else 0)
        self.traj_box.blockSignals(False)
        self._fill_segments()
        self._camera_selected()
        sg = doc.get('suggestion')
        self.sugg_label.setText(('Suggestion: ' + sg['text']) if sg else 'Every camera is within the smear limit.')
        self.suggestion_ready(bool(sg and sg.get('changes')))
        notes = list(doc.get('notes') or [])
        src = {'trajectories': 'the project\'s trajectories', 'grasp approaches': 'grasp approaches (no trajectory)',
               'none': 'no motion'}.get(doc.get('source'), '')
        self.notes_label.setText('\n'.join([f'Motion: {src}.'] + notes))
        self._overlay3d = None
        self._redraw()

    def _selected_key(self):
        it = self.cams.currentItem()
        return it.data(0, Qt.ItemDataRole.UserRole) if it is not None else None

    def _camera(self, key=None):
        key = key or self._selected_key()
        return next((c for c in (self.doc or {}).get('cameras', []) if c['key'] == key), None)

    def _fill_segments(self):
        self.segs.clear()
        uid = self.traj_box.currentData()
        tr = next((t for t in (self.doc or {}).get('trajectories', []) if t['uid'] == uid), None)
        if tr is None:
            _fit(self.segs)
            return
        names = {c['key']: c['name'] for c in self.doc.get('cameras', [])}
        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        for sg in tr['segments']:
            w = sg['cameras'][0] if sg['cameras'] else None
            it = QTreeWidgetItem([str(sg['index'] + 1), f'{sg["t0"]:.2f} – {sg["t1"]:.2f}',
                                  names.get(w['key'], '—') if w else '—', _px(w['smear_px']) if w else '—',
                                  _px(w['skew_px']) if w else '—', (w or {}).get('point', ''),
                                  sg.get('word') or MB.WORDS.get(sg['status'], sg['status'])])
            for k in (1, 3, 4):
                it.setFont(k, mono(11))
                it.setTextAlignment(k, right)
            it.setForeground(6, QColor(STATUS[SEV_COLOUR.get(sg['status'], 'warning')]))
            self.segs.addTopLevelItem(it)
        _fit(self.segs)

    def _camera_selected(self):
        c = self._camera()
        self.frame_box.clear()
        for k, f in enumerate((c or {}).get('frames', [])):
            self.frame_box.addItem(f'{f["trajectory"]} · t = {f["t"]:.2f} s · {f["speed_px_s"]:.0f} px/s'
                                   + (f' · smear {f["smear_px"]:.2f} px' if f['smear_px'] is not None else ''), k)
        self.b_frame.setEnabled(bool(c and c.get('frames')))
        if c is not None:
            k = self.exp_cam.findData(c['sensor_uid'])
            if k >= 0:
                self.exp_cam.setCurrentIndex(k)

    # ---------------------------------------------------------------- frames
    def show_frame(self, index: int | None = None) -> bool:
        """Pose the robot at a kept frame of the selected camera and show that camera in the Camera view."""
        c = self._camera()
        if c is None or not c.get('frames'):
            return False
        k = self.frame_box.currentData() if index is None else index
        f = c['frames'][int(k or 0)]
        st = self.ctx.store
        if st.project.robot(f['robot_uid']) is None:
            return False
        try:
            st.set_q(f['robot_uid'], np.asarray(f['q'], float))
        except Exception:                               # noqa: BLE001 — the robot cannot be posed: said
            traceback.print_exc()
            if self.ctx.window is not None:
                self.ctx.window.msg('The robot cannot be posed at that frame.', 6000)
            return False
        self._frame = (c, f)
        cv = getattr(self.ctx.window, 'camview', None) if self.ctx.window is not None else None
        if cv is not None and hasattr(cv, 'view'):
            cv.view.imager_name = c['imager']
        st.select(('sensor', c['sensor_uid']))
        self._redraw()
        return True

    def frame_vectors(self, cam_model) -> list:
        """[(K, 2) polylines]: the kept frame's smear vectors in cam_model's pixels, and the drawing scale."""
        if self._frame is None:
            return [], 1.0
        c, f = self._frame
        q = self.ctx.store.q.get(f['robot_uid'])
        if q is None or len(q) != len(f['q']) or not np.allclose(q, f['q'], atol=1e-6):
            return [], 1.0                               # the robot has moved on: the vectors are for another pose
        sx = cam_model.width / float(c['stream'][0])
        sy = cam_model.height / float(c['stream'][1])
        V = np.asarray([v[:4] for v in f['vectors']], float).reshape(-1, 4)
        if not len(V):
            return [], 1.0
        d = V[:, 2:4] if f['smear_px'] is not None else V[:, 2:4] * 0.0
        if not np.any(d):                                # no exposure known: the speed over 10 ms, said in the legend
            d = np.asarray([v[4:6] for v in f['vectors']], float) * 0.01
        longest = float(np.linalg.norm(d, axis=1).max())
        scale = max(1.0, MIN_DRAWN_PX / longest) if longest > 0 else 1.0
        out = []
        for (u, v, _du, _dv), (du, dv) in zip(V, d * scale):
            out.append(np.array([[(u - du / 2) * sx, (v - dv / 2) * sy], [(u + du / 2) * sx, (v + dv / 2) * sy]]))
        return out, scale

    # ---------------------------------------------------------------- overlays
    def overlays(self, vp) -> list:
        if not self.show3d.isChecked() or self.doc is None:
            return []
        if self._overlay3d is not None:
            return self._overlay3d
        from ..viewport3d import LineDraw
        items = []
        for tr in self.doc.get('trajectories', []):
            P = np.asarray(tr.get('tcp_path') or [], float)
            ts = np.asarray(tr.get('tcp_path_t') or [], float)
            if len(P) < 2 or len(ts) != len(P):
                continue
            for sg in tr['segments']:
                sel = (ts >= sg['t0'] - 1e-9) & (ts <= sg['t1'] + 1e-9)
                idx = np.flatnonzero(sel)
                if len(idx) == 0:
                    continue
                lo, hi = max(int(idx[0]) - 1, 0), min(int(idx[-1]) + 1, len(P) - 1)
                seg = P[lo:hi + 1]
                if len(seg) >= 2:
                    sev = SEV_COLOUR.get(sg['status'], 'warning')
                    items.append(LineDraw(seg.astype(np.float32), QColor(STATUS[sev]), 3.0, depth=False))
        self._overlay3d = items
        return items

    def camview_overlays(self, cam_model, pose) -> list:
        if self._frame is None:
            return []
        c, f = self._frame
        try:
            view = self.ctx.window.camview.view
            if view.sensor_uid != c['sensor_uid'] or (view.imager_name and view.imager_name != c['imager']):
                return []
        except AttributeError:
            return []
        polys, scale = self.frame_vectors(cam_model)
        if not polys:
            return []
        what = 'smear vectors' if f['smear_px'] is not None else 'image motion in 10 ms (no exposure known)'
        lab = f'{what} at t = {f["t"]:.2f} s' + (f' (drawn × {scale:.0f})' if scale > 1.0 else '')
        sev = SEV_COLOUR.get(c['status'], 'warning')
        return [(polys, QColor(STATUS[sev]) if sev != 'good' else THEME.color('ink'), lab)]

    def _redraw(self):
        self._overlay3d = None
        win = self.ctx.window
        if win is not None and hasattr(win, 'viewport'):
            win.viewport.update()
        self._redraw_camview()

    def _redraw_camview(self):
        win = self.ctx.window
        cv = getattr(win, 'camview', None) if win is not None else None
        if cv is not None and hasattr(cv, 'view'):
            cv.view.refresh_overlays()

    # ---------------------------------------------------------------- apply / preview / explain
    def apply(self):
        """Set the suggested exposures as yours (one undo step), then run again."""
        sg = (self.doc or {}).get('suggestion')
        if not sg or not sg.get('changes'):
            return False
        names = ', '.join(f'{ch["sensor"]} {ch["exposure_s"] * 1000:.3g} ms' for ch in sg['changes'])
        self.ctx.store.edit('analysis', lambda p: MB.apply_suggestion(p, sg), f'Motion blur · exposure {names}')
        self.suggestion_ready(False)
        self.run()
        return True

    def preview(self):
        sg = (self.doc or {}).get('suggestion')
        if sg and self.ctx.window is not None:
            self.ctx.window.msg('Preview: ' + sg['text'] + (' Apply makes the change (one undo step).'
                                                            if sg.get('changes') else ''), 15000)
        return sg

    def explain_text(self) -> str:
        st = self.settings()
        return ('Each camera\'s image of the task volume, the task objects, the TCP and the fingertip pads is followed '
                'along every trajectory (or the grasp approaches when there is none). A camera on a link moves with '
                'the link\'s twist J(q)·q̇; a moving point with its link\'s. The image velocity is the projection\'s '
                'derivative applied to the point\'s velocity in the camera frame; smear = image speed × exposure, '
                'rolling-shutter skew = image speed × readout.\n\n'
                'Exposure and readout come from your figures, else the library entry (cited), else the frame period '
                '1/fps as an upper bound (said "at most"). The smear limit is '
                f'{st["smear_limit_px"]:g} px (default 1 px, a DexVantage default, not a standard); the longest '
                'exposure that keeps it is limit / peak image speed.\n\n'
                'Known limits: first order in the exposure; occlusion not tested; a shorter exposure needs more light; '
                'other joints and robots stay at home; the carried object is not moved with the gripper.')

    def explain(self):
        box = QMessageBox(self)
        box.setWindowTitle('How motion blur is computed')
        box.setText(self.explain_text())
        box.setModal(False)
        box.show()


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


register('motion_blur', 'Motion blur and rolling shutter', MotionBlurPage, menu='Analysis', icon='move',
         workspace='placement', order=64)
