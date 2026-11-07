"""The Canonical view page (spec §7.16): robot B's wrist camera warped to robot A's canonical viewpoint through the
fingertip contact plane (core/studies/canonical_view.py, model in core/exporters/canonical_warp.py).

What it shows: the residual parallax off the contact plane at the task volume's depth extremes (the headline: the
P95 at the worse extreme, in canonical pixels, with a status word: aligned ≤ 1 px, mild parallax ≤ 4 px, parallax
≤ 16 px, large parallax above; DexVantage defaults), a row per extreme (distance, max, P95, median), the parallax map
over the canonical image, the homography and the share of the canonical image the warp fills. The source camera,
the canonical camera (a camera of the project or a canonical view file written by an earlier export), the plane
offset, the closing-axis sign and, in place of the task volume's bound, an own depth range (depth_range_m) are
chosen here (each one undo step). Apply mounts the source camera at the
canonical camera's pose relative to its own fingertips (one undo step: the parallax is then zero); Preview draws
that pose. Export writes the result as JSON; Export warp maps… writes the OpenCV remap maps (.npy, .yml), the
grid_sample .npy and H as JSON with provenance (the exporter 'canonical_warp', a window job). The 3D view draws the
contact plane, the source camera's frustum and the canonical camera placed at the source robot's fingertips; the
Camera view draws the warped grid: on the canonical camera, where the source image's grid lands; on the source
camera, the canonical image's grid (which of its pixels the warp reads).

Known limits: see the study's.
"""
from __future__ import annotations

import json
import os
import traceback

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtWidgets import QCheckBox, QComboBox, QFileDialog, QHeaderView, QLabel, QMessageBox, QTreeWidget, \
    QTreeWidgetItem

from ...core.exporters import canonical_warp as CW
from ..theme import SEQ, STATUS
from ..widgets import ScrubSpinBox, button, label
from . import register
from .base import StudyPage, row

LOADED = '__loaded__'
MAP_W = 320


def _f(x, nd=2):
    return '—' if x is None else f'{x:.{nd}f}'


def warp_from_doc(doc: dict):
    """The Warp the study's result describes (rebuilt from its views: no project needed)."""
    from ...core.optics import CameraModel

    def view(d):
        return CW.View(d['name'], CameraModel.from_dict(d['camera']), np.asarray(d['T_contact_cam'], float),
                       d.get('robot_uid') or '', d.get('robot') or '', d.get('sensor_uid') or '',
                       d.get('imager') or '', bool(d.get('virtual')), bool(d.get('rigid', True)), '',
                       None if d.get('T_world_contact') is None else np.asarray(d['T_world_contact'], float))
    return CW.make_warp(view(doc['source']), view(doc['canonical']), float(doc['plane']['offset_m']))


def headline_figure(h: dict) -> tuple[str, str]:
    """(figure, unit) of the headline: the P95 in px, or 'larger than the image' with the bound when the residual
    parallax exceeds the canonical image's width (a 11 878 px figure in a 1280 px image is right but unreadable)."""
    v = h.get('value')
    width = h.get('image_width_px')
    if v is not None and (h.get('beyond_image') or (width and v > float(width))):
        return 'larger than the image', f'(bound {v:,.0f} px)'.replace(',', ' ')
    return _f(v, 1 if v is not None and v >= 10 else 2), 'px'


def _range_of(v):
    """(lo, hi) metres of a depth_range_m setting, or None when it is not two finite numbers."""
    try:
        lo, hi = (float(x) for x in v)
    except (TypeError, ValueError):
        return None
    return (lo, hi) if np.isfinite(lo) and np.isfinite(hi) else None


class CanonicalViewPage(StudyPage):
    """Canonical view: robot B's wrist image warped to robot A's canonical viewpoint by the homography of the
    fingertip contact plane (exact there), with the residual parallax of points off the plane over the task volume's
    depth range. Lower is better; zero when the two cameras sit at the same place relative to their fingertips."""
    STUDY = CW.SID
    TITLE = 'Canonical view'
    HEADLINE = 'Residual parallax off the contact plane'

    def build_body(self, layout):
        self._loaded = None                     # a canonical view document loaded from a file
        self._preview = False
        self._warp = None
        self._filling = False
        self.last_export = None
        self.src_box = QComboBox()
        self.src_box.setToolTip('Robot B\'s camera: the images to warp')
        self.can_box = QComboBox()
        self.can_box.setToolTip('Robot A\'s camera: the canonical viewpoint (a camera of this project or a canonical '
                                'view file)')
        for b in (self.src_box, self.can_box):
            b.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
            b.setMinimumContentsLength(16)
        self.src_box.currentIndexChanged.connect(lambda _i: self._pick('source', self.src_box))
        self.can_box.currentIndexChanged.connect(lambda _i: self._pick('canonical', self.can_box))
        self.b_load = button('Load canonical view…', role='ghost', icon_name='folder',
                             tip='A <stem>_canonical_view.json written by an export of robot A\'s project')
        self.b_load.clicked.connect(lambda: self.load_view())
        layout.addWidget(row(label('Source (robot B)', ink2=True), self.src_box))
        layout.addWidget(row(label('Canonical (robot A)', ink2=True), self.can_box, self.b_load))
        self.s_offset = ScrubSpinBox('', 'length', 1, 'mm', step=1.0, width=96)
        self.s_offset.setToolTip('The contact plane\'s distance from the fingertip centre along the approach axis '
                                 '(default 0 mm: through the TCP)')
        self.s_offset.committed.connect(lambda v: self._set('plane_offset_m', float(v) / 1000.0, 'plane offset'))
        self.c_flip = QCheckBox('Flip the source\'s closing axis')
        self.c_flip.setToolTip('Which pad comes first is arbitrary: flipping turns the source\'s contact frame half a '
                               'turn about the approach axis')
        self.c_flip.toggled.connect(lambda on: self._set('closing_sign_source', -1 if on else 1, 'closing axis'))
        layout.addWidget(row(label('Plane offset', ink2=True), self.s_offset, self.c_flip))
        self.c_range = QCheckBox('Own depth range')
        self.c_range.setToolTip('Off: the depth range is a bound from the task volume and the pads\' reach (the '
                                'default). On: the two distances from the contact plane given here')
        self.s_lo = ScrubSpinBox('', 'length', 0, 'mm', step=1.0, width=80)
        self.s_lo.setToolTip('How far beyond the contact plane (away from the source camera) points lie: a negative '
                             'distance')
        self.s_hi = ScrubSpinBox('', 'length', 0, 'mm', step=1.0, width=80)
        self.s_hi.setToolTip('How far towards the source camera points lie: a positive distance (held short of the '
                             'camera\'s near limit)')
        self.c_range.toggled.connect(self._range_toggled)
        self.s_lo.committed.connect(lambda _v: self._range_changed())
        self.s_hi.committed.connect(lambda _v: self._range_changed())
        layout.addWidget(row(label('Depth range', ink2=True), self.c_range, label('from', muted=True), self.s_lo,
                             label('to', muted=True), self.s_hi))
        self.views = label('', wrap=True)
        layout.addWidget(self.views)
        self.rng = label('', muted=True, wrap=True, small=True)
        layout.addWidget(self.rng)
        self.tab = QTreeWidget()
        self.tab.setRootIsDecorated(False)
        self.tab.setHeaderLabels(['Depth extreme', 'Distance from the plane', 'Max', 'P95', 'Median'])
        self.tab.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.tab.setMaximumHeight(90)
        layout.addWidget(self.tab)
        layout.addWidget(label('Parallax over the canonical image (the worse extreme)', bold=True))
        self.map = QLabel()
        self.map.setMinimumHeight(60)
        self.map.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.map)
        self.map_note = label('', muted=True, small=True, wrap=True)
        layout.addWidget(self.map_note)
        layout.addWidget(label('Homography (source to canonical pixels, undistorted images)', bold=True))
        self.h_label = label('', mono_=True, small=True)
        self.h_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.h_label)
        self.notes = label('', muted=True, wrap=True, small=True)
        layout.addWidget(self.notes)
        self.b_maps = button('Export warp maps…', role='ghost', icon_name='export',
                             tip='OpenCV remap maps (.npy and .yml), a grid_sample .npy and H as JSON, with '
                                 'provenance, into a folder')
        self.b_maps.clicked.connect(lambda: self.export_maps())
        layout.addWidget(row(self.b_maps))
        self.ctx.store.changed.connect(self._changed)
        self.ctx.store.project_replaced.connect(self._replaced)
        self._fill_boxes()

    # ---------------------------------------------------------------- choices
    def _settings(self) -> dict:
        return dict(self.ctx.store.project.analysis.studies.get(CW.SID) or {})

    def _fill_boxes(self):
        p = self.ctx.store.project
        st = self._settings()
        self._filling = True
        try:
            try:
                cams = CW.candidate_cameras(p)
            except Exception:                           # noqa: BLE001 — a library entry that cannot load
                traceback.print_exc()
                cams = []
            try:
                dsrc, dcan = CW.default_pair(p) if cams and p.robots else ('', None)
            except CW.WarpError:
                dsrc, dcan = '', None
            for box, key, dflt in ((self.src_box, 'source', dsrc), (self.can_box, 'canonical', dcan)):
                box.clear()
                for s in cams:
                    robot = p.robot(s.mount.robot_uid) if s.mount.on_link else None
                    box.addItem(f'{s.name}' + (f' · {robot.name}' if robot else ' · static'), s.uid)
                want = st.get(key) or dflt
                if key == 'canonical' and isinstance(want, dict):
                    self._loaded = want
                    box.addItem(f'{want.get("name", "Canonical view")} (file)', LOADED)
                    want = LOADED
                elif key == 'canonical' and self._loaded is not None:
                    box.addItem(f'{self._loaded.get("name", "Canonical view")} (file)', LOADED)
                i = box.findData(want)
                box.setCurrentIndex(i if i >= 0 else 0)
            self.s_offset.setValue(float(st.get('plane_offset_m', 0.0)) * 1000.0)
            self.c_flip.setChecked(int(st.get('closing_sign_source', 1)) < 0)
            given = _range_of(st.get('depth_range_m'))
            self.c_range.setChecked(given is not None)
            lo, hi = given if given is not None else self._shown_range()
            self.s_lo.setValue(lo * 1000.0)
            self.s_hi.setValue(hi * 1000.0)
            self.s_lo.setEnabled(given is not None)
            self.s_hi.setEnabled(given is not None)
        finally:
            self._filling = False

    def _shown_range(self) -> tuple:
        """The depth range of the result on screen (metres), else ±50 mm: where an own range starts."""
        r = (self.doc or {}).get('depth_range') or {}
        try:
            return float(r['h_lo_m']), float(r['h_hi_m'])
        except (KeyError, TypeError, ValueError):
            return -0.05, 0.05

    def _range_toggled(self, on):
        if self._filling:
            return
        self.s_lo.setEnabled(on)
        self.s_hi.setEnabled(on)
        if on:
            self._range_changed()
        else:
            self._set('depth_range_m', None, 'depth range (the default)')

    def _range_changed(self):
        if self._filling or not self.c_range.isChecked():
            return
        lo, hi = float(self.s_lo.value()) / 1000.0, float(self.s_hi.value()) / 1000.0
        if lo > hi:
            self._say('The depth range runs from beyond the plane (the lower, negative end) to towards the camera '
                      '(the upper end): the lower end must not exceed the upper.')
            self._fill_boxes()
            return
        self._set('depth_range_m', [lo, hi], 'depth range')

    def _pick(self, key, box):
        if self._filling:
            return
        uid = box.currentData()
        if uid is None:
            return
        value = self._loaded if uid == LOADED else uid
        self._set(key, value, f'{key} camera')

    def _set(self, key, value, what):
        if self._filling:
            return
        if self._settings().get(key) == value:
            return

        if value is None and key not in self._settings():
            return

        def fn(p):
            d = p.analysis.studies.setdefault(CW.SID, {})
            if value is None:
                d.pop(key, None)                        # back to the default
            else:
                d[key] = value
        self.ctx.store.edit('analysis', fn, f'Canonical view · {what}', merge_key=(CW.SID, key))
        self.run()

    def load_view(self, path=None):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, 'Load a canonical view', '', 'Canonical view (*.json)')
            if not path:
                return None
        try:
            with open(path, encoding='utf-8') as fh:
                doc = json.load(fh)
            if isinstance(doc, dict) and isinstance(doc.get('canonical'), dict) and 'camera' not in doc:
                doc = dict(doc['canonical'], schema=CW.VIEW_SCHEMA)         # an export's homography JSON
            v = CW.view_from_dict(doc)
        except (OSError, ValueError) as e:
            self._say(f'Could not read {os.path.basename(path)}: {getattr(e, "strerror", None) or e}.')
            return None
        except CW.WarpError as e:
            self._say(str(e))
            return None
        doc = CW.view_document(v)
        doc.pop('schema', None)
        self._loaded = doc
        self._set('canonical', doc, f'canonical view from {os.path.basename(path)}')
        self._fill_boxes()
        return doc

    def _say(self, text, ms=8000):
        if self.ctx.window is not None:
            self.ctx.window.msg(text, ms)

    def _changed(self, sections):
        if 'variant' in sections or 'variants' in sections or 'robots' in sections:
            self._fill_boxes()
        elif 'analysis' in sections:
            self._fill_boxes()

    def _replaced(self):
        self.doc = None
        self._warp = None
        self._loaded = None
        self._preview = False
        self.suggestion_ready(False)
        self._fill_boxes()
        self.empty()

    def activate(self):
        self._fill_boxes()

    # ---------------------------------------------------------------- result
    def show_result(self, doc):
        try:
            h = doc.get('headline') or {}
            th = h.get('thresholds_px') or CW.DEFAULTS['thresholds_px']
            fig, unit = headline_figure(h)
            self.head.set(fig, unit, h.get('status', 'warning'), h.get('word', ''), h.get('advice', ''),
                          f'P95 of the residual parallax at the worse depth extreme, in {doc["canonical"]["name"]}\'s '
                          f'pixels (max {_f(h.get("max_px"))} px). Status: aligned ≤ {th["good"]:g} px, mild '
                          f'parallax ≤ {th["warning"]:g} px, parallax ≤ {th["serious"]:g} px'
                          + (' (DexVantage defaults).' if h.get('thresholds_default') else ' (your thresholds).'))
            s, c, rel = doc['source'], doc['canonical'], doc['relative']
            self.views.setText(f'{s["name"]} ({s.get("robot") or "—"}) to {c["name"]} '
                               f'({"a canonical view file" if c.get("virtual") else c.get("robot") or "—"}): '
                               f'{rel["baseline_m"] * 1000:.1f} mm apart relative to the fingertips; the plane is '
                               f'{doc["plane"]["d_B_m"] * 1000:.1f} mm from the source and '
                               f'{abs(doc["plane"]["d_A_m"]) * 1000:.1f} mm from the canonical camera. '
                               f'{doc["coverage"]["text"]}.')
            r = doc['depth_range']
            self.rng.setText(f'Depth range {r["h_lo_m"] * 1000:.0f} to +{r["h_hi_m"] * 1000:.0f} mm from the plane '
                             f'(negative: beyond it; positive: towards the source camera): {r["rule"]}'
                             + (', its upper end held short of the source camera\'s near limit.' if r.get('clamped')
                                else '.'))
            if not self.c_range.isChecked():                # the bound in use, where an own range would start
                self._filling = True
                try:
                    self.s_lo.setValue(float(r['h_lo_m']) * 1000.0)
                    self.s_hi.setValue(float(r['h_hi_m']) * 1000.0)
                finally:
                    self._filling = False
            self.tab.clear()
            par = doc['parallax']
            for name, key, hh in (('Beyond the plane', 'at_lo', r['h_lo_m']), ('Towards the camera', 'at_hi',
                                                                               r['h_hi_m'])):
                st = par[key]
                it = QTreeWidgetItem([name, f'{hh * 1000:+.0f} mm', f'{_f(st["max_px"])} px', f'{_f(st["p95_px"])} px',
                                      f'{_f(st["median_px"])} px'])
                sev, _w = CW.status_of(st['p95_px'], th)
                it.setForeground(3, QColor(STATUS[sev]))
                self.tab.addTopLevelItem(it)
            self._draw_map(par.get('map') or {}, th)
            H = doc['homography']
            self.h_label.setText('\n'.join('  '.join(f'{x:+.6e}' for x in rr) for rr in H))
            self.notes.setText(' '.join(doc.get('notes') or []))
            self._warp = warp_from_doc(doc)
        except (KeyError, TypeError, ValueError, CW.WarpError):
            traceback.print_exc()
            self._warp = None
        sg = (self.doc or {}).get('suggestion')
        self.suggestion_ready(bool(sg) and not sg.get('locked'))
        self._redraw()

    def _draw_map(self, m: dict, th: dict):
        vals = m.get('values') or []
        if not vals:
            self.map.clear()
            self.map_note.setText('')
            return
        a = np.array([[np.nan if x is None else float(x) for x in r] for r in vals], float)
        hi = max(float(th.get('serious', 16.0)), float(np.nanmax(a)) if np.isfinite(a).any() else 1.0)
        stops = np.array([QColor(c).getRgb()[:3] for c in SEQ], float)
        t = np.clip(np.nan_to_num(a, nan=0.0) / hi, 0, 1) * (len(stops) - 1)
        i0 = np.floor(t).astype(int).clip(0, len(stops) - 2)
        f = (t - i0)[..., None]
        rgb = stops[i0] * (1 - f) + stops[i0 + 1] * f
        rgba = np.concatenate([rgb, np.full(a.shape + (1,), 255.0)], -1)
        rgba[~np.isfinite(a)] = (56, 56, 53, 255)
        img = np.ascontiguousarray(rgba.astype(np.uint8))
        h, w = img.shape[:2]
        q = QImage(img.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
        scale = max(1, MAP_W // max(1, w))
        self.map.setPixmap(QPixmap.fromImage(q).scaled(w * scale, h * scale, Qt.AspectRatioMode.KeepAspectRatio,
                                                       Qt.TransformationMode.FastTransformation))
        self.map_note.setText(f'Light to dark blue: 0 to {hi:.1f} px; dark grey: no source pixel there. '
                              f'Cells of {m.get("stride", 1)} px.')

    # ---------------------------------------------------------------- the suggestion
    def apply(self):
        sg = (self.doc or {}).get('suggestion')
        if not sg:
            return None
        if sg.get('locked'):
            self._say(f'{sg["sensor"]} is locked: unlock it to move it.')
            return None

        def fn(p):
            from ...core.frames import Pose
            s = p.variant.sensor(sg['sensor_uid'])
            if s is not None and not s.locked:
                s.mount.pose = Pose.from_dict(sg['pose'])
        self.ctx.store.edit('variant', fn, f'Canonical view · {sg["sensor"]} at the canonical pose '
                                           f'({sg["move_m"] * 1000:.0f} mm)')
        self._preview = False
        self.suggestion_ready(False)
        self.run()
        return sg

    def preview(self):
        sg = (self.doc or {}).get('suggestion')
        self._preview = not self._preview and bool(sg)
        if sg:
            self._say(sg['text'] + (' The dashed frustum in the 3D view shows the pose.' if self._preview else ''),
                      10000)
        self._redraw()

    def explain(self):
        box = QMessageBox(self)
        box.setWindowTitle('How the canonical view is computed')
        box.setText('Both robots\' contact frames (origin between the fingertips, z the approach axis, x the closing '
                    'axis, measured from the contact pads) are taken as the same frame. That gives the source '
                    'camera\'s pose relative to the canonical camera (R, t) and the fingertip contact plane in each '
                    'camera. Every plane point maps between the undistorted images by H = K_A (R − t nᵀ/d) K_B⁻¹; '
                    'the warp maps go further and include both lenses. A point off the plane by h along the source '
                    'ray lands in the canonical image off by the parallax Δ = h / (d Z_A) ((K_A t)_xy − p (K_A t)_z) '
                    '(pinhole form): zero on the plane and when t = 0. The figure is the P95 of |Δ| at the task '
                    'volume\'s depth extremes. Thresholds and the depth range are DexVantage defaults.')
        box.setModal(False)
        box.show()

    # ---------------------------------------------------------------- export of the maps (a window job)
    def export_maps(self, folder=None, wait=False):
        win = self.ctx.window
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the canonical warp into a folder')
            if not folder:
                return None
        snap = self.ctx.store.project.snapshot()

        def work(progress, cancel):
            from ...core import exporters as X
            from ...core.jobs import Ctx
            return X.export(snap, folder, ['canonical_warp'], ctx=Ctx(progress=progress, cancel=cancel))

        def done(man):
            row_ = man['formats']['canonical_warp']
            if row_['status'] == 'ok':
                self._say(f'Exported the canonical warp ({len(row_["files"])} files) with export_manifest.json to '
                          f'{os.path.basename(folder) or folder}', 8000)
            else:
                self._say(row_.get('reason', 'The canonical warp could not be exported.'), 10000)
            self.last_export = man

        def failed(message):
            self._say('The canonical warp export stopped: ' + message.splitlines()[0], 10000)
        if win is None or wait:
            man = work(lambda *_: None, lambda: False)
            done(man)
            return man
        if not win.run_job('canonical_warp_export', 'Exporting the canonical warp', work, done, failed):
            self._say('The canonical warp is being exported already', 4000)
        return folder

    # ---------------------------------------------------------------- overlays
    def overlays(self, vp) -> list:
        from ..viewport3d import LineDraw
        w = self._warp
        if w is None:
            return []
        try:
            g = CW.world_geometry(w)
        except Exception:                               # noqa: BLE001 — an overlay must not take the view down
            traceback.print_exc()
            return []
        if not g:
            return []
        out = [LineDraw(np.asarray(g['plane']), QColor(STATUS['good']), width=2.0, depth=False)]
        for k, col, dash in (('source', '#3987e5', False), ('canonical', '#d95926', True)):
            for pl in g[k]:
                out.append(LineDraw(np.asarray(pl, float), QColor(col), width=2.0 if self._preview and dash else 1.5,
                                    dash=dash, depth=False))
        out.append(LineDraw(np.array([g['source_centre'], g['canonical_centre']]), QColor('#d95926'), width=1.0,
                            dash=True, depth=False))
        return out

    def camview_overlays(self, cam_model, pose) -> list:
        w = self._warp
        cv = getattr(self.ctx.window, 'camview', None) if self.ctx.window is not None else None
        uid = getattr(cv, 'sensor_uid', None) or getattr(getattr(cv, 'view', None), 'sensor_uid', None)
        if w is None or not uid:
            return []
        try:
            if uid == w.canonical.sensor_uid and (cam_model.width, cam_model.height) == (w.canonical.camera.width,
                                                                                         w.canonical.camera.height):
                return [(CW.warped_grid(w, 'canonical'), QColor('#3987e5'), f'{w.source.name}\'s grid warped here '
                                                                            '(exact on the contact plane)')]
            if uid == w.source.sensor_uid and (cam_model.width, cam_model.height) == (w.source.camera.width,
                                                                                      w.source.camera.height):
                return [(CW.warped_grid(w, 'source'), QColor('#d95926'), f'the pixels {w.canonical.name}\'s warp '
                                                                         'reads')]
        except Exception:                               # noqa: BLE001
            traceback.print_exc()
        return []

    def _redraw(self):
        win = self.ctx.window
        if win is not None and hasattr(win, 'viewport'):
            win.viewport.update()
        cv = getattr(win, 'camview', None) if win is not None else None
        if cv is not None and hasattr(cv, 'view') and hasattr(cv.view, 'refresh_overlays'):
            cv.view.refresh_overlays()


register('canonical_view', 'Canonical view', CanonicalViewPage, menu='Export', icon='frustum', action_id=None,
         workspace=None, order=95)
