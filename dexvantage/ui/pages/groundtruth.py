"""The "Rendered against predicted" page (spec §9.1, MuJoCo ground truth): how well the grasp study's predicted
fingertip visibility and the depth-validity study's predicted valid region agree with MuJoCo renders of the same
scene.

What it shows: the agreement of the contact points in view (the headline, with a status word from thresholds that
are defaults: 90 % and 75 %), the confusion matrix (predicted visible or hidden against rendered visible or hidden)
with the IoU of the visible-point sets, why points disagree (at a segmentation edge, a difference of rule, or of
occlusion), the grasp study's requirement on the predicted and on the rendered fractions, a row per grasp (and home)
and per camera, the depth validity per depth sensor (on the study's own voxels and on the rendered surfaces), side by
side thumbnails of the worst disagreements (the prediction's Camera view render on the left, MuJoCo's on the right,
each contact point coloured by its own side's verdict) and the render's GPU and frame rate. The 3D view shows the
contact points of the chosen pose coloured by agreement; the Camera view outlines MuJoCo's segmentation of what it
shows (rendered in the background at the view's own pose and joint state).

Simulate ▸ Build the MuJoCo scene… (and the page's button) writes the scene as a self-contained MJCF folder. Apply
makes the suggested change: doubling the grasp study's cube-map size when occlusion disagreements are more than a
labelled default share of the points (one undo step); Preview shows the worst disagreement's pose.
"""
from __future__ import annotations

import json
import os
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (QFileDialog, QGridLayout, QHeaderView, QMessageBox, QTreeWidget, QTreeWidgetItem,
                             QWidget)

from ...core.studies import groundtruth as GT
from ..theme import STATUS, THEME
from ..widgets import ScrubSpinBox, button, label
from . import register
from .base import StudyPage, row

THUMB_W = 260
GLYPH_M = 0.0015                 # half-size of a contact point glyph in the 3D view (m)
OCCLUSION_SHARE = 0.01           # default: suggest finer cube maps when occlusion disagreements exceed 1 % of points
STATE_COLORS = {GT.ST_TP: STATUS['good'], GT.ST_FP: STATUS['serious'], GT.ST_FN: '#3987e5',
                GT.ST_TN: '#8a8a85'}
SEG_PALETTE = ('#e0883a', '#3987e5', '#2fa3a8', '#d6457a', '#8a63d2', '#c9a227', '#0ca30c', '#8f8e86')
MAX_CONTOURS = 400


def _pct(x, nd=1) -> str:
    return '—' if x is None else f'{100 * x:.{nd}f} %'


class _Signals(QObject):
    seg_ready = pyqtSignal(object)


class GroundTruthPage(StudyPage):
    """Rendered against predicted: the scene the predictions use is built in MuJoCo (the robots at each grasp's joint
    state, the objects and tables, one camera per imager with the same intrinsics, lens distortion applied by
    remapping) and rendered. A contact point counts as visible in the render when its pixel shows the finger it
    belongs to at the point's depth; the grasp study counts it visible when nothing hides the contact location in its
    cube map. The figure is the share of contact points in view on which the two agree."""
    STUDY = GT.SID
    TITLE = 'Rendered against predicted'
    HEADLINE = 'Rendered against predicted visibility'

    # ---------------------------------------------------------------- body
    def build_body(self, layout):
        self.dec = None
        self._case = 0
        self._overlay = []
        self._seg = {}                  # Camera view key -> contours
        self._seg_busy = False
        self._seg_pending = None
        self._seg_renderer = None
        self._seg_lock = threading.Lock()
        self._sig = _Signals()
        self._sig.seg_ready.connect(self._seg_done)
        self.grasps = ScrubSpinBox('', 'plain', 0, '', step=10.0, width=80)
        self.grasps.setRange(1, 10000)
        self.grasps.setToolTip('How many grasps of the set are rendered (default 50; drawn with a fixed seed)')
        self.grasps.committed.connect(lambda v: self._set_setting('grasps', int(round(v)), f'{int(round(v))} grasps'))
        self.tol = ScrubSpinBox('', 'plain', 1, 'mm', step=1.0, width=80)
        self.tol.setRange(0.1, 100)
        self.tol.setToolTip('A contact point is visible in the render when its pixel shows its finger within this '
                            'depth of the point (default 5 mm)')
        self.tol.committed.connect(lambda v: self._set_setting('depth_tol_mm', float(v), f'depth tolerance {v:g} mm'))
        self.b_scene = button('Save the MuJoCo scene…', role='ghost', icon_name='cube',
                              tip='Write the scene (robots, objects, tables, lights, one camera per imager) as a '
                                  'self-contained MJCF folder')
        self.b_scene.clicked.connect(self.save_scene_dialog)
        layout.addWidget(row(label('Grasps'), self.grasps, label('Depth tolerance'), self.tol, self.b_scene))
        self.defaults_label = label('', muted=True, small=True, wrap=True)
        layout.addWidget(self.defaults_label)

        layout.addWidget(label('Contact points in view: predicted against rendered', bold=True))
        self.matrix = QTreeWidget()
        self.matrix.setRootIsDecorated(False)
        self.matrix.setHeaderLabels(['', 'Rendered visible', 'Rendered hidden', 'Total'])
        _columns(self.matrix)
        layout.addWidget(self.matrix)
        self.matrix_note = label('', muted=True, wrap=True)
        layout.addWidget(self.matrix_note)
        self.why = QTreeWidget()
        self.why.setRootIsDecorated(False)
        self.why.setHeaderLabels(['Why they disagree', 'Points', 'Meaning'])
        _columns(self.why, stretch=2)
        layout.addWidget(self.why)

        layout.addWidget(label('Worst disagreements (left: the prediction, right: MuJoCo)', bold=True))
        self.thumbs = QWidget()
        self.thumbs_grid = QGridLayout(self.thumbs)
        self.thumbs_grid.setContentsMargins(0, 0, 0, 0)
        self.thumbs_grid.setSpacing(6)
        self.thumbs_grid.setColumnStretch(3, 1)           # the pairs and their captions packed to the left
        layout.addWidget(self.thumbs)
        self.thumbs_legend = label('', muted=True, small=True, wrap=True)
        layout.addWidget(self.thumbs_legend)

        layout.addWidget(label('Cameras', bold=True))
        self.cams = QTreeWidget()
        self.cams.setRootIsDecorated(False)
        self.cams.setHeaderLabels(['Camera', 'Imager', 'Agreement', 'IoU', 'Both visible', 'Predicted only',
                                   'Rendered only', 'Both hidden'])
        _columns(self.cams)
        layout.addWidget(self.cams)

        layout.addWidget(label('Poses (click one to put the robot there)', bold=True))
        self.cases = QTreeWidget()
        self.cases.setRootIsDecorated(False)
        self.cases.setUniformRowHeights(True)
        self.cases.setMinimumHeight(180)
        self.cases.setHeaderLabels(['Pose', 'Agreement', 'IoU', 'Predicted only', 'Rendered only', 'Edge',
                                    'Rule', 'Occlusion', 'Requirement (predicted / rendered)'])
        _columns(self.cases, stretch=8)
        self.cases.itemClicked.connect(lambda it, _c: self.show_case(it.data(0, Qt.ItemDataRole.UserRole)))
        layout.addWidget(self.cases)

        layout.addWidget(label('Depth validity: the depth study against rendered depth with its noise model',
                               bold=True))
        self.depth = QTreeWidget()
        self.depth.setRootIsDecorated(False)
        self.depth.setHeaderLabels(['Sensor', 'Voxels agree', 'IoU', 'Surfaces agree', 'Surface IoU',
                                    'Disagreements (predicted → rendered)', 'Noise'])
        _columns(self.depth, stretch=5)
        layout.addWidget(self.depth)
        self.render_label = label('', muted=True, small=True, wrap=True)
        layout.addWidget(self.render_label)
        self.legend = label('', muted=True, small=True, wrap=True)
        layout.addWidget(self.legend)
        self.notes_label = label('', muted=True, wrap=True)
        self.notes_label.setVisible(False)
        layout.addWidget(self.notes_label)
        st = self.ctx.store
        st.changed.connect(lambda _s: self._sync())
        st.project_replaced.connect(self._replaced)
        self._take_scene_command()
        self._sync()

    def _take_scene_command(self):
        """The disabled placeholder Simulate ▸ Build the MuJoCo scene… opens the save dialog from now on."""
        win = self.ctx.window
        a = getattr(win, 'actions_by_id', {}).get('mujoco_scene') if win is not None else None
        if a is None:
            return
        a.setEnabled(True)
        d = dict(a.data() or {})
        d['phase'] = None
        a.setData(d)
        a.setToolTip('Write the project as a self-contained MuJoCo scene (MJCF and its meshes)')
        a.triggered.connect(lambda _=False: self.save_scene_dialog())

    # ---------------------------------------------------------------- settings
    def settings(self) -> dict:
        try:
            return GT.settings_of(self.ctx.store.project)
        except Exception:                               # noqa: BLE001 — a bad stored value: show the defaults
            return dict(GT.DEFAULTS, defaults=list(GT.DEFAULTS))

    def _sync(self):
        st = self.settings()
        for w, k in ((self.grasps, 'grasps'), (self.tol, 'depth_tol_mm')):
            w.blockSignals(True)
            w.setValue(float(st[k]))
            w.blockSignals(False)
        d = st.get('defaults', [])
        self.defaults_label.setText('Defaults in use: ' + ', '.join(
            f'{k} {GT.DEFAULTS[k]}' for k in ('grasps', 'depth_tol_mm', 'depth_lr_tol_mm', 'good_pct', 'warn_pct')
            if k in d) + ' (DexVantage defaults, not standards).' if d else '')

    def _set_setting(self, key, value, what):
        cur = (self.ctx.store.project.analysis.studies.get(GT.SID) or {}).get(key)
        if cur == value:
            return

        def fn(p):
            d = dict(p.analysis.studies.get(GT.SID) or {})
            d[key] = value
            p.analysis.studies[GT.SID] = d
        self.ctx.store.edit('analysis', fn, f'Rendered against predicted · {what}', merge_key=(GT.SID, key))

    def _replaced(self):
        self.doc, self.dec, self._overlay = None, None, []
        for t in (self.matrix, self.why, self.cams, self.cases, self.depth):
            t.clear()
        self._clear_thumbs()
        self._close_seg_renderer()
        self.render_label.setText('')
        self.notes_label.setVisible(False)
        self.suggestion_ready(False)
        self._sync()
        self.empty()

    # ---------------------------------------------------------------- result
    def _headline(self, doc):
        s = doc.get('summary') or {}
        st = self.settings()
        pct = s.get('agreement_pct')
        if pct is None:
            self.head.set('—', '', 'warning', 'nothing compared', s.get('advice') or 'Nothing could be compared.')
            return
        sev, word = GT.status_for(pct, st)
        c = doc.get('contacts') or {}
        unit = '% of contact points in view agree' if s.get('headline') == 'contacts' else '% of depth voxels agree'
        iou = s.get('contact_iou') if s.get('headline') == 'contacts' else s.get('depth_iou')
        tip = (f'IoU of the visible sets {_pct(iou)}. Status: agrees from {st["good_pct"]:g} %, mostly agrees from '
               f'{st["warn_pct"]:g} % (DexVantage defaults, not standards).')
        if c.get('totals'):
            t = c['totals']
            tip += (f' {t["n"]:,} (camera, contact point) pairs in view over {c["cases"]} poses; '
                    f'{c.get("outside_image", 0):,} outside the image are not counted.')
        self.head.set(f'{pct:.1f}', unit, sev, word, s.get('advice', ''), tip)

    def show_result(self, doc):
        try:
            self._headline(doc)
            self._fill_matrix(doc)
            self._fill_cams(doc)
            self._fill_cases(doc)
            self._fill_depth(doc)
            self._fill_thumbs(doc)
            self.dec = GT.decode(doc)
            r = doc.get('render') or {}
            fps = r.get('frames_per_s')
            self.render_label.setText(
                f'Rendered by MuJoCo {r.get("mujoco", "")} through EGL on {r.get("renderer") or "?"} '
                f'({r.get("device", "")}): {r.get("frames", 0)} frames, '
                + (f'{fps:.0f} frames/s in the renderer' if fps else 'no frame rate') + '. The grasp study\'s cube '
                f'maps: {(doc.get("contacts") or {}).get("occlusion", {}).get("backend", "—")}.')
            self._case = 0
            self._build_overlay()
        except (KeyError, ValueError, TypeError):
            traceback.print_exc()
        notes = [str(n) for n in (doc.get('notes') or []) + ((doc.get('scene') or {}).get('notes') or [])]
        c = doc.get('contacts') or {}
        if c.get('skipped'):
            notes.insert(0, c['skipped'])
        self.notes_label.setText('\n'.join(notes))
        self.notes_label.setVisible(bool(notes))
        self.suggestion_ready(self.suggestion() is not None)
        self._refresh_camview()

    def _fill_matrix(self, doc):
        self.matrix.clear()
        self.why.clear()
        c = doc.get('contacts') or {}
        t = c.get('totals')
        if not t:
            self.matrix_note.setText(c.get('skipped') or 'No contact comparison in this run.')
            return
        for name, a, b in (('Predicted visible', t['tp'], t['fp']), ('Predicted hidden', t['fn'], t['tn'])):
            it = QTreeWidgetItem([name, f'{a:,}', f'{b:,}', f'{a + b:,}'])
            self.matrix.addTopLevelItem(it)
        self.matrix.addTopLevelItem(QTreeWidgetItem(['Total', f'{t["tp"] + t["fn"]:,}', f'{t["fp"] + t["tn"]:,}',
                                                     f'{t["n"]:,}']))
        _fit(self.matrix)
        g = c.get('grasp_level') or {}
        off = c.get('depth_offset_mm') or {}
        r = c.get('rules') or {}
        text = (f'Agreement {_pct(t["agreement"])}, IoU of the visible sets {_pct(t["iou"])}. The grasp study\'s '
                f'requirement ({r.get("requirement", "")}) gives the same verdict on '
                f'{g.get("tp", 0) + g.get("tn", 0)} of {g.get("n", 0)} grasps. The prediction tests each point '
                f'{r.get("lift_mm", 2):g} mm in front of the pad (cube maps of {r.get("cube_n", 512)} texels); the '
                f'render asks for the finger at the point\'s depth within {r.get("depth_tol_mm", 5):g} mm.')
        if off.get('n'):
            text += (f' Rendered depth minus the point\'s, where the finger shows: median {off["median"]:.2f} mm, '
                     f'5–95 % {off["p5"]:.1f} to {off["p95"]:.1f} mm.')
        self.matrix_note.setText(text)
        why, text = c.get('why') or {}, c.get('why_text') or GT.WHY_TEXT
        for k in GT.WHY:
            self.why.addTopLevelItem(QTreeWidgetItem([{'edge': 'At an edge', 'definition': 'Rule',
                                                       'occlusion': 'Occlusion'}[k], f'{why.get(k, 0):,}',
                                                      text.get(k, '')]))
        _fit(self.why)

    def _fill_cams(self, doc):
        self.cams.clear()
        for cm in (doc.get('contacts') or {}).get('cameras') or []:
            self.cams.addTopLevelItem(QTreeWidgetItem([cm['name'], f'{cm["imager"]} {cm["mode"][0]} × '
                                                       f'{cm["mode"][1]}', _pct(cm.get('agreement')),
                                                       _pct(cm.get('iou')), f'{cm["tp"]:,}', f'{cm["fp"]:,}',
                                                       f'{cm["fn"]:,}', f'{cm["tn"]:,}']))
        _fit(self.cams)

    def _fill_cases(self, doc):
        self.cases.clear()
        rows = (doc.get('contacts') or {}).get('table') or []
        order = sorted(range(len(rows)), key=lambda i: (rows[i].get('agreement') or 0) if i else -1)
        for i in order:
            r = rows[i]
            w = r.get('why') or {}
            req = ('—' if r['kind'] != 'grasp' else
                   f'{"meets" if r["meets_predicted"] else "fails"} / {"meets" if r["meets_rendered"] else "fails"}')
            it = QTreeWidgetItem([r['case'], _pct(r.get('agreement')), _pct(r.get('iou')), f'{r["fp"]:,}',
                                  f'{r["fn"]:,}', f'{w.get("edge", 0):,}', f'{w.get("definition", 0):,}',
                                  f'{w.get("occlusion", 0):,}', req])
            it.setData(0, Qt.ItemDataRole.UserRole, i)
            if r['kind'] == 'grasp' and r['meets_predicted'] != r['meets_rendered']:
                it.setForeground(8, QColor(STATUS['critical']))
            self.cases.addTopLevelItem(it)

    def _fill_depth(self, doc):
        self.depth.clear()
        d = doc.get('depth') or {}
        for s in d.get('sensors') or []:
            v, sf = s.get('voxels') or {}, s.get('surfaces') or {}
            dis = v.get('disagreements') or sf.get('disagreements') or {}
            it = QTreeWidgetItem([s['name'], _pct(v.get('agreement'), 2), _pct(v.get('iou'), 2),
                                  _pct(sf.get('agreement'), 2), _pct(sf.get('iou'), 2),
                                  ', '.join(f'{k} {n:,}' for k, n in list(dis.items())[:3]) or 'none',
                                  s.get('noise', '')])
            it.setToolTip(3, 'On the surfaces the reference imager sees (every '
                          f'{sf.get("stride", 2)}nd pixel): the study\'s cube maps test points exactly on a surface, '
                          'where a texel\'s quantisation can call the surface hidden from itself; inside the task '
                          f'volumes {_pct((sf.get("inside_volumes") or {}).get("agreement"), 2)}.')
            it.setToolTip(6, s.get('noise', ''))
            self.depth.addTopLevelItem(it)
        for s in d.get('skipped') or []:
            self.depth.addTopLevelItem(QTreeWidgetItem([s['name'], '—', '—', '—', '—', s['why'], '']))
        if not d.get('sensors') and d.get('message'):
            self.depth.addTopLevelItem(QTreeWidgetItem([d['message'], '', '', '', '', '', '']))
        _fit(self.depth)

    def _clear_thumbs(self):
        while self.thumbs_grid.count():
            w = self.thumbs_grid.takeAt(0).widget()
            if w is not None:
                w.deleteLater()

    def _fill_thumbs(self, doc):
        self._clear_thumbs()
        th = (doc.get('contacts') or {}).get('thumbs') or []
        for k, t in enumerate(th):
            try:
                d = GT.decode_thumb(t)
            except (KeyError, ValueError):
                continue
            for col, (img, which) in enumerate(((d['predicted_rgb'], d['pred']), (d['rendered_rgb'], d['rend']))):
                lab = label('')
                lab.setPixmap(thumb_pixmap(img, d['uv'], which, which != (d['pred'] if col else d['rend']), THUMB_W))
                lab.setToolTip(f'{t["case"]} · {t["camera"]}: {t["disagree"]} points disagree. '
                               + ('The prediction (core/camrender, the Camera view\'s renderer)' if col == 0 else
                                  'MuJoCo\'s render') + ': green visible, orange hidden on this side; ringed where '
                               'the other side differs.')
                self.thumbs_grid.addWidget(lab, k, col)
            self.thumbs_grid.addWidget(label(f'{t["case"]} · {t["camera"]} · {t["disagree"]} disagree', muted=True,
                                             small=True), k, 2)
        self.thumbs_legend.setText('Each contact point: green where this side calls it visible, orange hidden; a '
                                   'ring where the two sides differ.' if th else '')

    # ---------------------------------------------------------------- the pose of a case
    def show_case(self, i):
        rows = ((self.doc or {}).get('contacts') or {}).get('table') or []
        if i is None or not 0 <= int(i) < len(rows):
            return
        self._case = int(i)
        try:
            self.ctx.store.set_q(self.doc['contacts']['robot']['uid'], np.asarray(rows[self._case]['q'], float))
        except Exception:                               # noqa: BLE001 — the robot may have gone
            traceback.print_exc()
        self._build_overlay()

    def _build_overlay(self):
        from ..viewport3d import TriDraw
        from .grasp import octahedra
        self._overlay = []
        if not self.dec:
            self.legend.setText('')
            return
        try:
            S = self.dec['states'][self._case]                  # (C, K)
            P = self.dec['points'][self._case]
            worst = np.full(S.shape[1], GT.ST_OUT, np.uint8)
            for code in (GT.ST_TN, GT.ST_TP, GT.ST_FN, GT.ST_FP):  # later wins: a disagreement shows
                worst = np.where((S == code).any(axis=0), code, worst)
            keep = worst != GT.ST_OUT
            cols = np.array([QColor(STATE_COLORS.get(int(s), '#8a8a85')).getRgbF() for s in worst[keep]], np.float32)
            if keep.any():
                self._overlay = [TriDraw(octahedra(P[keep], GLYPH_M), np.repeat(cols, 8, axis=0), depth=True)]
            case = self.doc['contacts']['table'][self._case]['case']
            self.legend.setText(f'3D view: the contact points at {case}, by camera agreement: green visible in both, '
                                'orange predicted visible but hidden in the render, blue the other way, grey hidden in '
                                'both (a disagreement in any camera shows).')
        except Exception:                               # noqa: BLE001 — the overlay must not take the page down
            traceback.print_exc()
        self._redraw()

    def overlays(self, vp) -> list:
        return list(self._overlay)

    def _redraw(self):
        win = self.ctx.window
        if win is not None and hasattr(win, 'viewport'):
            win.viewport.update()

    # ---------------------------------------------------------------- suggestion
    def suggestion(self):
        """(text, new cube_n) when occlusion disagreements exceed OCCLUSION_SHARE of the points in view, else None."""
        c = (self.doc or {}).get('contacts') or {}
        t, w = c.get('totals') or {}, c.get('why') or {}
        if not t.get('n'):
            return None
        n = int((c.get('rules') or {}).get('cube_n', 512))
        if w.get('occlusion', 0) / t['n'] > OCCLUSION_SHARE and n < 2048:
            return (f'Draw the grasp study\'s cube maps at {2 * n} texels (now {n}): {w["occlusion"]:,} contact points '
                    'disagree on occlusion away from an edge', 2 * n)
        return None

    def apply(self):
        sg = self.suggestion()
        if sg is None:
            return None

        def fn(p):
            d = dict(p.analysis.studies.get('grasp_visibility') or {})
            d['cube_n'] = int(sg[1])
            p.analysis.studies['grasp_visibility'] = d
        self.ctx.store.edit('analysis', fn, f'Grasp study · cube maps of {sg[1]} texels')
        self.suggestion_ready(False)
        if self.ctx.window is not None:
            self.ctx.window.msg('Changed the grasp study\'s cube-map size: Run again to compare.', 6000)
        return sg[1]

    def preview(self):
        """Put the robot at the pose with the most disagreements and show its points."""
        rows = ((self.doc or {}).get('contacts') or {}).get('table') or []
        if not rows:
            return
        i = max(range(len(rows)), key=lambda k: rows[k]['fp'] + rows[k]['fn'])
        self.show_case(i)
        if self.ctx.window is not None:
            sg = self.suggestion()
            self.ctx.window.msg(f'Preview: {rows[i]["case"]}, the pose with the most disagreements.'
                                + (f' Apply: {sg[0]}.' if sg else ''), 8000)

    def explain(self):
        st = self.settings()
        box = QMessageBox(self)
        box.setWindowTitle('How predicted and rendered visibility are compared')
        box.setText(
            'The project is built as a MuJoCo scene: the robots from their own files, the task objects and tables, '
            'and one MuJoCo camera per imager with the imager\'s focal lengths and principal point. A lens with '
            'distortion is rendered as a wider pinhole (or cube faces) and remapped through the lens model, pixel by '
            'pixel.\n\n'
            f'Contacts: {st["grasps"]} grasps of the set (a default sample, seed {st["rng_seed"]}) and the home pose '
            'are solved and evaluated by the grasp study itself. A contact point is predicted visible when nothing '
            'hides the contact location in front of the pad in the camera\'s cube map; it is rendered visible when '
            'the pixel it falls on shows its own finger at its depth (within '
            f'{st["depth_tol_mm"]:g} mm). Disagreements are split into those at a segmentation edge, those where '
            'the rules differ (a pad seen edge-on) and those where the occlusion differs.\n\n'
            'Depth: the depth study\'s own test of every voxel of the task volumes is compared with the rendered '
            'depth of every imager the depth needs, the reference depth with the sensor\'s noise model added.\n\n'
            f'The status reads agrees from {st["good_pct"]:g} % and mostly agrees from {st["warn_pct"]:g} % '
            '(defaults, not standards). Known limits: the prediction leaves the camera\'s seat out and the render '
            'does not; one pixel per contact point; the noise model is Gaussian and first order; the projector '
            'pattern and IR materials are not rendered.')
        box.setModal(False)
        box.show()

    # ---------------------------------------------------------------- the MuJoCo scene folder
    def save_scene_dialog(self, folder: str | None = None):
        """Ask for a folder and write the scene there in the background (a window job)."""
        win = self.ctx.window
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Save the MuJoCo scene into a folder')
            if not folder:
                return None
        p = self.ctx.store.project
        if not p.robots:
            if win is not None:
                win.msg('There is no robot to put in a MuJoCo scene: import one or open a sample.', 6000)
            return None
        snap = p.snapshot()

        def job(progress, cancel):
            from ...core.sim import save_scene
            progress(0.1, 'Building the scene')
            try:
                return save_scene(snap, folder)
            except OSError as e:                         # no write access, a full disk: a sentence, not errno
                raise OSError(f'DexVantage cannot write into the folder {os.path.basename(os.path.normpath(folder))} '
                              f'({(e.strerror or str(e)).lower()}): choose another folder') from None

        def done(path):
            if win is not None:
                win.msg(f'Saved the MuJoCo scene: {os.path.basename(os.path.dirname(path))}/scene.xml with its '
                        'assets and scene_notes.txt', 8000)

        def failed(msg):
            if win is not None:
                win.msg(f'The MuJoCo scene could not be saved: {msg.splitlines()[0]}', 10000)
        if win is not None and hasattr(win, 'run_job'):
            win.run_job('mujoco_scene', 'Saving the MuJoCo scene', job, done, failed)
            return folder
        return job(lambda *_: None, lambda: False)

    # ---------------------------------------------------------------- Camera view: rendered segmentation
    def _refresh_camview(self):
        cv = getattr(self.ctx.window, 'camview', None) if self.ctx.window is not None else None
        if cv is not None and hasattr(cv, 'view'):
            cv.view.refresh_overlays()

    def _seg_key(self, cam_model, pose) -> str:
        q = {u: [round(float(x), 4) for x in np.asarray(v).reshape(-1)] for u, v in self.ctx.store.q.items()}
        return json.dumps([cam_model.to_dict(), np.round(pose.matrix(), 5).tolist(), q, self._inputs_key()],
                          sort_keys=True, default=str)

    def _inputs_key(self) -> str:
        p = self.ctx.store.project
        return json.dumps([p.workspace.to_dict(), [r.to_dict() for r in p.robots]], sort_keys=True, default=str)

    def camview_overlays(self, cam_model, pose) -> list:
        """MuJoCo's segmentation of what the Camera view shows, as outlines (rendered in the background)."""
        if self.doc is None:
            return []
        key = self._seg_key(cam_model, pose)
        hit = self._seg.get(key)
        if hit is None:
            self._start_seg(key, cam_model, pose)
            return []
        return hit

    def _start_seg(self, key, cam_model, pose):
        if self._seg_busy:
            self._seg_pending = (key, cam_model, pose)          # the newest request waits for the running one
            return
        self._seg_busy = True
        self._seg_pending = None
        snap = self.ctx.store.project.snapshot()
        q = {u: np.asarray(v, float).copy() for u, v in self.ctx.store.q.items()}
        ikey = self._inputs_key()

        def work():
            out = []
            try:
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
                    img = r[1].render_pose(cm, pose, q, ('seg',))
                out = seg_outlines(img['seg'], img['seg_owner'], cam_model.width / cm.width)
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
            pend, self._seg_pending = getattr(self, '_seg_pending', None), None
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


def seg_outlines(seg: np.ndarray, owners: dict, scale: float) -> list:
    """[(polylines, colour, label)]: the outline of each robot link, object and surface in a segmentation image,
    in pixels of an image scale times larger (OpenCV pixel centres)."""
    import cv2
    out, kinds = [], {}
    n = 0
    for k in [int(x) for x in np.unique(seg) if x >= 0]:
        o = owners.get(k)
        if o is None:
            continue
        cs, _h = cv2.findContours((seg == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys = []
        for c in cs:
            if len(c) < 3 or n >= MAX_CONTOURS:
                continue
            pts = c.reshape(-1, 2).astype(float)
            pts = (pts + 0.5) * scale - 0.5
            polys.append(np.vstack([pts, pts[:1]]))
            n += 1
        if polys:
            kinds.setdefault(o[0], []).extend(polys)
    for i, (kind, polys) in enumerate(sorted(kinds.items())):
        out.append((polys, QColor(SEG_PALETTE[i % len(SEG_PALETTE)]), f'MuJoCo segmentation: {kind}'))
    return out


def thumb_pixmap(rgb, uv, vis, differs, width: int) -> QPixmap:
    """A thumbnail with each contact point drawn by this side's verdict (a ring where the other side differs)."""
    rgb = np.ascontiguousarray(rgb, np.uint8)
    h, w = rgb.shape[:2]
    img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    for (u, v), a, d in zip(uv, vis, differs):
        if not (-1 <= u <= w and -1 <= v <= h):
            continue
        col = QColor(STATUS['good'] if a else STATUS['serious'])
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(col)
        p.drawEllipse(QPointF(float(u), float(v)), 1.4, 1.4)
        if d:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(THEME.color('ink'))
            p.drawEllipse(QPointF(float(u), float(v)), 3.0, 3.0)
    p.end()
    pm = QPixmap.fromImage(img)
    return pm.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation) if w != width else pm


def _columns(tree: QTreeWidget, stretch: int = 0):
    h = tree.header()
    h.setStretchLastSection(False)
    for k in range(tree.columnCount()):
        h.setSectionResizeMode(k, QHeaderView.ResizeMode.Stretch if k == stretch else
                               QHeaderView.ResizeMode.ResizeToContents)


def _fit(tree: QTreeWidget):
    n = tree.topLevelItemCount()
    rh = tree.sizeHintForRow(0) if n else 20
    tree.setFixedHeight(tree.header().sizeHint().height() + max(1, n) * max(rh, 18) + 6)


register('groundtruth', 'Rendered against predicted', GroundTruthPage, menu='Simulate', icon='compare',
         action_id='mujoco_validate', workspace='grasp', order=60)
