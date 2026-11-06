"""The Brackets page (spec §7.7): a printable bracket per sensor, its checks and the clearance to the robot.

What it shows: how many of the rig's brackets are ready to print (closed solid, holes where the cited sources put them,
clear of every link over the sampled joint range and of the cables), with a status word; per sensor its bracket type,
size, mass, the smallest clearance and what to fix; the selected bracket's holes, interface figures with their
sources, flags and print notes. The bracket type per sensor is Mount.bracket (one undo step when changed); its
options (flange pattern, side, standoff, tilt, wall) are the study's per-sensor settings (one undo step each).
Apply sets the suggested bracket types for the sensors that have none (one undo step); Preview draws the suggestion's
note; Export writes the result as JSON with provenance; Export STL writes the selected bracket's STL with a
provenance sidecar (STL + .json).

3D view: every generated bracket (thinned for drawing) and its camera body box at the joint state on screen, red when
it collides. Camera view: the brackets' outlines.
"""
from __future__ import annotations

import copy
import os
import traceback

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QComboBox, QFileDialog, QHeaderView, QTreeWidget, QTreeWidgetItem)

from ...core import brackets as B
from ...core.studies import brackets as SB
from ..theme import mono
from ..viewport3d import LineDraw, TriDraw
from ..widgets import ScrubSpinBox, StatusLine, button, label
from . import register
from .base import StudyPage, row

STATUS_COLOR = {'good': '#0ca30c', 'warning': '#fab219', 'critical': '#d03b3b', 'unset': '#8f8e86',
                'skipped': '#8f8e86'}
STATUS_WORD = {'good': 'ready', 'warning': 'check', 'critical': 'collides', 'unset': 'not chosen',
               'skipped': 'none'}
BRACKET_RGBA = (0.78, 0.78, 0.80, 0.85)
CLASH_RGBA = (0.82, 0.23, 0.23, 0.85)
BOX_EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def _tree(headers) -> QTreeWidget:
    t = QTreeWidget()
    t.setRootIsDecorated(False)
    t.setHeaderLabels(headers)
    t.setUniformRowHeights(True)
    for k in range(1, len(headers)):
        t.header().setSectionResizeMode(k, QHeaderView.ResizeMode.ResizeToContents)
    t.setMinimumHeight(80)
    return t


def _mm(v, nd=1):
    return '—' if v is None else f'{v:.{nd}f} mm'


def box_corners(c, A, half) -> np.ndarray:
    s = np.array([[a, b, d] for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)], float)
    return np.asarray(c, float) + (s * np.asarray(half, float)) @ np.asarray(A, float).T


class BracketsPage(StudyPage):
    """Brackets: a watertight, printable bracket per sensor that holds it at its planned pose (a wrist-flange adapter
    on an ISO 9409-1 flange, a clamp for 2020 or 4040 T-slot profile, a pan-tilt neck plate, or a palm or dorsal
    sleeve for a micro camera), meshed into a closed STL, with the bracket and camera body checked against every
    robot link over sampled joint states and against the cable paths."""
    STUDY = SB.STUDY_ID
    TITLE = 'Brackets'
    HEADLINE = 'Brackets ready to print and clear of the robot'

    def build_body(self, layout):
        self.sel_uid = ''
        self._overlay_key = None
        self._overlay_items: list = []
        self._preview = None
        self.c_sensor = QComboBox()
        self.c_sensor.setToolTip('The sensor whose bracket the controls, the details and Export STL are for')
        self.c_kind = QComboBox()
        for k, t in B.TYPES.items():
            self.c_kind.addItem(t, k)
        self.c_kind.insertItem(0, 'Not chosen', '')
        self.c_kind.setToolTip('The bracket type (stored on the sensor\'s mount; one undo step)')
        layout.addWidget(row(label('Sensor', ink2=True), self.c_sensor, label('Bracket', ink2=True), self.c_kind))
        self.c_pattern = QComboBox()
        for k, v in B.ISO9409.items():
            self.c_pattern.addItem(v['designation'], k)
        self.c_pattern.setToolTip('The robot flange\'s ISO 9409-1 pattern; figures from the cited drawings')
        self.c_side = QComboBox()
        for k in B.SIDES:
            self.c_side.addItem({'auto': 'Auto side'}.get(k, f'{k.capitalize()} side'), k)
        self.c_side.setToolTip('Where the base sits on the camera: behind it, under it or above it')
        self.s_standoff = self._spin(0, 200, 1.0, ' mm', 'Distance from the camera face to the base (default 10 mm)')
        self.s_tilt = self._spin(-90, 90, 5.0, '°', 'The base tilted about its long axis')
        self.s_wall = self._spin(0.8, 12, 0.5, ' mm', 'Wall and plate thickness (default 3 mm)')
        self.l_standoff, self.l_tilt, self.l_wall = (label(t, ink2=True) for t in ('Standoff', 'Tilt', 'Wall'))
        layout.addWidget(row(self.c_pattern, self.c_side, self.l_standoff, self.s_standoff, self.l_tilt, self.s_tilt,
                             self.l_wall, self.s_wall))
        self.b_stl = button('Export STL…', role='ghost', icon_name='export',
                            tip='The selected bracket as a binary STL (mm) with a JSON sidecar carrying provenance')
        self.b_stl.clicked.connect(lambda: self.export_stl())
        layout.addWidget(row(self.b_stl))
        self.table = _tree(['Sensor', 'Bracket', 'Status', 'Size', 'Mass', 'Clearance', 'Image'])
        layout.addWidget(self.table)
        self.detail = StatusLine('', '', wrap=True)
        layout.addWidget(self.detail)
        layout.addWidget(label('Holes and interface', bold=True))
        self.holes = _tree(['Hole', 'Diameter', 'Centre (bracket frame)', 'Source'])
        layout.addWidget(self.holes)
        self.flags = label('', wrap=True)
        layout.addWidget(self.flags)
        self.notes_l = label('', muted=True, wrap=True, small=True)
        layout.addWidget(self.notes_l)
        self.c_sensor.currentIndexChanged.connect(self._sensor_changed)
        self.c_kind.activated.connect(self._kind_chosen)
        self.c_pattern.activated.connect(lambda _i: self._opt('pattern', self.c_pattern.currentData()))
        self.c_side.activated.connect(lambda _i: self._opt('side', self.c_side.currentData()))
        self.s_standoff.editingFinished.connect(lambda: self._opt('standoff_mm', self.s_standoff.value()))
        self.s_tilt.editingFinished.connect(lambda: self._opt('tilt_deg', self.s_tilt.value()))
        self.s_wall.editingFinished.connect(lambda: self._opt('wall_mm', self.s_wall.value()))
        self.table.currentItemChanged.connect(self._row_clicked)
        st = self.ctx.store
        st.changed.connect(lambda _s: self._fill_controls())
        st.project_replaced.connect(self._replaced)
        self._fill_controls()

    def _spin(self, lo, hi, step, suffix, tip):
        # ScrubSpinBox, not a plain QDoubleSpinBox: Qt 6.12 draws a style-sheeted spin box without buttons with a
        # 1 px line edit (docs/BUILD_STATE.md, decision 3), which hid these fields
        s = ScrubSpinBox('', 'plain', 1, suffix.strip(), step=step, width=96)
        s.setRange(lo, hi)
        s.setToolTip(tip)
        s.setKeyboardTracking(False)
        return s

    # ---------------------------------------------------------------- controls
    def _sensors(self):
        try:
            return [s for s in self.ctx.store.project.variant.sensors if s.enabled]
        except Exception:                               # noqa: BLE001 — the controls must not take the page down
            traceback.print_exc()
            return []

    def _sensor(self):
        return next((s for s in self._sensors() if s.uid == self.sel_uid), None)

    def _replaced(self):
        self.doc = None
        self._fill_controls()
        self.empty()

    def _fill_controls(self):
        ss = self._sensors()
        self.c_sensor.blockSignals(True)
        self.c_sensor.clear()
        for s in ss:
            self.c_sensor.addItem(s.name, s.uid)
        uids = [s.uid for s in ss]
        if self.sel_uid not in uids:
            self.sel_uid = uids[0] if uids else ''
        if self.sel_uid:
            self.c_sensor.setCurrentIndex(uids.index(self.sel_uid))
        self.c_sensor.blockSignals(False)
        self._fill_options()

    def _fill_options(self):
        s = self._sensor()
        on = s is not None
        for w in (self.c_kind, self.c_pattern, self.c_side, self.s_standoff, self.s_tilt, self.s_wall):
            w.setEnabled(on)
        if s is None:
            return
        spec = B.spec_for(self.ctx.store.project, s)
        i = self.c_kind.findData(s.mount.bracket or '')
        self.c_kind.setCurrentIndex(max(i, 0))
        k = spec.kind
        self.c_pattern.setVisible(k == 'flange')
        i = self.c_pattern.findData(spec.pattern or '50-4-M6')
        if i >= 0:
            self.c_pattern.setCurrentIndex(i)
        side_on = k in ('tslot2020', 'tslot4040', 'pantilt', 'palm', 'dorsal')
        for w in (self.c_side, self.s_standoff, self.s_tilt, self.l_standoff, self.l_tilt):
            w.setVisible(side_on)
        self.c_side.setCurrentIndex(max(self.c_side.findData(spec.side), 0))
        for w, v in ((self.s_standoff, spec.standoff_mm), (self.s_tilt, spec.tilt_deg), (self.s_wall, spec.wall_mm)):
            w.blockSignals(True)
            w.setValue(float(v))
            w.blockSignals(False)
        self.s_wall.setVisible(k in B.BUILT)
        self.l_wall.setVisible(k in B.BUILT)

    def _sensor_changed(self, i):
        self.sel_uid = self.c_sensor.itemData(i) or ''
        self._fill_options()
        if self.doc is not None:
            self._show_detail(self.doc)
        self._redraw()

    def set_kind(self, uid: str, kind: str):
        """Set a sensor's bracket type (Mount.bracket): one undo step."""
        s = next((x for x in self._sensors() if x.uid == uid), None)
        if s is None or (s.mount.bracket or '') == kind:
            return

        def fn(p):
            p.variant.sensor(uid).mount.bracket = kind
        self.ctx.store.edit('variant', fn, f'Bracket of {s.name}: {B.lc(B.TYPES.get(kind, "not chosen"))}')

    def _kind_chosen(self, _i):
        if self.sel_uid:
            self.set_kind(self.sel_uid, self.c_kind.currentData() or '')

    def _opt(self, key, value):
        """Set one per-sensor bracket option in the study settings: one undo step."""
        uid = self.sel_uid
        s = self._sensor()
        if s is None:
            return
        cur = ((self.ctx.store.project.analysis.studies.get(SB.STUDY_ID) or {}).get('sensors') or {}).get(uid) or {}
        if cur.get(key) == value:
            return

        def fn(p):
            st = copy.deepcopy(p.analysis.studies.get(SB.STUDY_ID) or {})
            st.setdefault('sensors', {}).setdefault(uid, {})[key] = value
            p.analysis.studies[SB.STUDY_ID] = st
        self.ctx.store.edit('analysis', fn, f'Bracket of {s.name}: {key.replace("_", " ")} {value}')

    # ---------------------------------------------------------------- run
    def run(self):
        q = {uid: [float(x) for x in np.asarray(v, float).reshape(-1)] for uid, v in self.ctx.store.q.items()}
        self.ctx.analysis.run(self.STUDY, {'q': q})

    def empty(self):
        self.head.set('—', '', 'warning', 'not run yet', 'Run Brackets to generate each sensor\'s bracket and check '
                                                         'its clearance to the robot.')
        self.table.clear()
        self.holes.clear()

    def show_result(self, doc: dict):
        self._preview = None
        rig = doc.get('rig') or {}
        n, ok = rig.get('brackets', 0), rig.get('ready', 0)
        tip = (f'Status: ready when every bracket is a closed solid and clear of the robot by the margin '
               f'({(doc.get("settings") or {}).get("margin_mm", 2):g} mm, a default) over '
               f'{(doc.get("settings") or {}).get("samples", 48)} sampled joint states. '
               f'Pose: {(doc.get("pose") or {}).get("note", "")}.')
        sug = doc.get('suggestion')
        act = rig.get('text', '')
        if sug:
            act += ' Suggested: ' + '; '.join(f'{x["sensor"]}: {B.lc(x["label"])}' for x in sug) + '.'
        self.head.set(f'{ok}/{n}' if n else '—', '', rig.get('status', 'warning'), rig.get('status_word', ''), act,
                      tip)
        self.suggestion_ready(bool(sug))
        self.table.clear()
        for r in doc.get('sensors') or []:
            cl = (r.get('clearance') or {}).get('links') or []
            dims = r.get('dims_mm')
            vals = [r['name'], r.get('label', ''), STATUS_WORD.get(r['status'], r['status']),
                    ' × '.join(f'{v:.0f}' for v in dims) + ' mm' if dims else '—',
                    f'{r["mesh"]["mass_g"]:.0f} g' if r.get('mesh') else '—',
                    _mm(cl[0]['min_mm']) + f' ({cl[0]["link"]})' if cl else '—',
                    f'{(r.get("image_blocked") or {}).get("blocked_pct", 0):.1f} %' if r.get('image_blocked') else '—']
            it = QTreeWidgetItem(vals)
            it.setData(0, Qt.ItemDataRole.UserRole, r['uid'])
            for k in (3, 4, 5, 6):
                it.setFont(k, mono(11))
            it.setForeground(2, QColor(STATUS_COLOR.get(r['status'], '#8f8e86')))
            it.setToolTip(2, r.get('message', ''))
            it.setToolTip(4, 'Solid print in PETG (1.27 g/cm³, Prusament PETG\'s stated density); infill lowers it')
            it.setToolTip(5, (r.get('clearance') or {}).get('method', ''))
            it.setToolTip(6, 'Share of the camera\'s image the bracket covers')
            self.table.addTopLevelItem(it)
        self._show_detail(doc)
        self._redraw()

    def _row(self, doc):
        rs = doc.get('sensors') or []
        return next((r for r in rs if r['uid'] == self.sel_uid), rs[0] if rs else None)

    def _row_clicked(self, cur, _prev):
        if cur is None:
            return
        uid = cur.data(0, Qt.ItemDataRole.UserRole)
        if uid and uid != self.sel_uid:
            i = self.c_sensor.findData(uid)
            if i >= 0:
                self.c_sensor.setCurrentIndex(i)

    def _show_detail(self, doc):
        r = self._row(doc)
        self.holes.clear()
        if r is None:
            self.detail.set('', '')
            self.flags.setText('')
            self.notes_l.setText('')
            return
        sev = {'unset': 'warning', 'skipped': ''}.get(r['status'], r['status'])
        self.detail.set(sev, f'{r["name"]} · {r.get("label", "")}: {r.get("message", "")}')
        src = ''
        if r.get('pattern'):
            src = '; '.join(s['url'] for s in r['pattern'].get('sources') or [])
        elif r.get('profile'):
            src = r['profile']['source']['url']
        for h in ((r.get('design') or {}).get('holes') or []):
            c = h['centre']
            it = QTreeWidgetItem([h['label'], f'{h["d_mm"]:.2f} mm', f'{c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f} mm',
                                  'cited' if h['role'] in ('flange_bolt', 'flange_pin', 'tslot_bolt') else
                                  ('your figures' if h['role'] == 'pad' and r['kind'] == 'pantilt' else
                                   ('drawing' if h.get('stated') else 'not dimensioned'))])
            for k in (1, 2):
                it.setFont(k, mono(11))
            it.setToolTip(3, src if h['role'] in ('flange_bolt', 'flange_pin', 'tslot_bolt') else
                          (r.get('camera') or {}).get('holes_note', ''))
            self.holes.addTopLevelItem(it)
        self.flags.setText('\n'.join(f'Check · {f}' for f in r.get('flags') or []))
        notes = list(r.get('notes') or [])
        if r.get('pattern'):
            notes.append(r['pattern'].get('note', ''))
        cam = r.get('camera') or {}
        if cam.get('box_assumption'):
            notes.append(cam['box_assumption'] + f' Size: {cam.get("size_note", "")}.')
        cl = r.get('clearance') or {}
        notes += cl.get('notes') or []
        if cl.get('cables'):
            notes.append(cl['cables'].get('note', ''))
        self.notes_l.setText('\n'.join(x for x in notes if x))

    # ---------------------------------------------------------------- Apply / Preview / Export STL
    def apply(self):
        sug = (self.doc or {}).get('suggestion') or []
        if not sug:
            return
        names = ', '.join(x['sensor'] for x in sug)

        def fn(p):
            for x in sug:
                s = p.variant.sensor(x['sensor_uid'])
                if s is not None and not (s.mount.bracket or '').strip():
                    s.mount.bracket = x['kind']
        self.ctx.store.edit('variant', fn, f'Bracket types for {names}')
        self._fill_controls()
        self.run()

    def preview(self):
        sug = (self.doc or {}).get('suggestion') or []
        self._preview = None if self._preview else sug
        if self.ctx.window is not None:
            text = ('Suggested: ' + '; '.join(f'{x["sensor"]}: {x["label"]} ({x["why"]})' for x in sug)
                    if self._preview else 'Preview off')
            self.ctx.window.msg(text, 12000)

    def export_stl(self, path=None, background: bool | None = None):
        """Generate the selected sensor's bracket at full resolution and write its STL and sidecar; the paths, or None
        (said why). The button (no path given) builds and writes on a worker thread, since a full-resolution mesh takes
        about a second; background=False, or a path given by a script or test, does it here and returns the paths."""
        win = self.ctx.window
        s = self._sensor()
        p = self.ctx.store.project
        if s is None or not (s.mount.bracket or '').strip() or s.mount.bracket == 'none':
            if win is not None:
                win.msg('Choose a bracket type for the sensor first (Bracket).', 6000)
            return None
        if background is None:
            background = path is None
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, 'Export bracket STL', f'{B.safe_name(s.name)}.stl',
                                                  'STL (*.stl)')
            if not path:
                return None
        folder = os.path.dirname(os.path.abspath(path))
        if not os.path.isdir(folder) or not os.access(folder, os.W_OK):     # say so before a second of meshing
            if win is not None:
                win.msg(f'Could not export the bracket: the folder {os.path.basename(folder) or folder} does not exist '
                        'or cannot be written to. Choose another.', 8000)
            return None
        snap = p.snapshot()
        snap_sensor = snap.variant.sensor(s.uid) or s

        def work(_progress=None, _cancel=None):
            try:
                spec = B.spec_for(snap, snap_sensor)
                extra = ((snap.analysis.studies.get(SB.STUDY_ID) or {}).get('sensors') or {}).get(s.uid) or {}
                d = dict(extra, kind=spec.kind, pattern=extra.get('pattern') or spec.pattern)
                row_ = B.build(snap, snap_sensor, B.BracketSpec.from_dict(d), clear=False)
                if row_.get('_mesh') is None:
                    return {'error': f'No bracket to export: {row_.get("message", "")}'}
                return {'paths': B.export_stl(row_, snap, path)}
            except (B.BracketError, OSError) as e:
                return {'error': f'Could not export the bracket: {getattr(e, "strerror", None) or e}'}

        def done(res):
            if win is None:
                return
            if 'error' in res:
                win.msg(res['error'], 8000)
            else:
                paths = res['paths']
                win.msg(f'Exported {os.path.basename(paths[0])} and its sidecar {os.path.basename(paths[1])}', 6000)

        def failed(_message):                           # the window logged the traceback
            if win is not None:
                win.msg('Could not export the bracket: an unexpected error (details in the log).', 8000)
        if background and win is not None and hasattr(win, 'run_job'):
            if win.run_job('bracket_stl', f'Exporting the bracket of {s.name}', work, done, failed):
                win.msg(f'Building the bracket of {s.name} at full resolution…', 4000)
            else:
                win.msg('A bracket is already being exported: wait for it to finish.', 5000)
            return None
        try:
            res = work()
        except Exception:                               # noqa: BLE001 — explained, logged, never a traceback
            traceback.print_exc()
            failed('')
            return None
        done(res)
        return res.get('paths')

    # ---------------------------------------------------------------- 3D and Camera view
    def _redraw(self):
        self._overlay_key = None
        if self.ctx.window is not None:
            self.ctx.window.viewport.update()

    def _parent_world(self, s):
        try:
            return B.sensor_static_matrix(self.ctx.store.project, s, self.ctx.store.q)
        except Exception:                               # noqa: BLE001 — a sensor that cannot be posed is not drawn
            return None

    def _placed(self):
        """[(row, T_world_bracket (m))] for the generated brackets at the pose on screen."""
        out = []
        doc = self.doc or {}
        ss = {s.uid: s for s in self._sensors()}
        for r in doc.get('sensors') or []:
            s = ss.get(r['uid'])
            pl = r.get('placement')
            if s is None or not pl or not r.get('display'):
                continue
            Tp = self._parent_world(s)
            if Tp is None:
                continue
            out.append((r, Tp @ np.asarray(pl['T_parent_bracket'], float)))
        return out

    def overlays(self, vp) -> list:
        if not self.doc:
            return []
        key = (id(self.doc), tuple(sorted((k, tuple(np.round(np.asarray(v, float).reshape(-1), 5)))
                                          for k, v in self.ctx.store.q.items())))
        if key == self._overlay_key:
            return self._overlay_items
        items = []
        for r, T in self._placed():
            tri = np.asarray(r['display']['tris_mm'], float).reshape(-1, 3, 3) / 1000.0
            W = tri @ T[:3, :3].T + T[:3, 3]
            rgba = CLASH_RGBA if r['status'] == 'critical' else BRACKET_RGBA
            items.append(TriDraw(W.astype(np.float32), np.tile(np.array(rgba, np.float32), (len(W), 1))))
            Tb = (r.get('design') or {}).get('T_bracket_body_mm')
            size = (r.get('camera') or {}).get('size_mm')
            if Tb is not None and size:
                c0, half = B.box_of(size)
                Tb = np.asarray(Tb, float)
                P = box_corners(c0, np.eye(3), half) @ Tb[:3, :3].T + Tb[:3, 3]
                P = (P / 1000.0) @ T[:3, :3].T + T[:3, 3]
                segs = np.array([[P[a], P[b]] for a, b in BOX_EDGES])
                items.append(LineDraw(segs, QColor(STATUS_COLOR.get(r['status'], '#8f8e86')), 1.4, segments=True))
        self._overlay_key, self._overlay_items = key, items
        return items

    def camview_overlays(self, cam_model, pose) -> list:
        """Each generated bracket's outline (its bounding box in the bracket frame) in the Camera view."""
        from ...core import camrender as CR
        out = []
        for r, T in self._placed():
            tri = np.asarray(r['display']['tris_mm'], float).reshape(-1, 3)
            lo, hi = tri.min(axis=0), tri.max(axis=0)
            P = box_corners((lo + hi) / 2, np.eye(3), (hi - lo) / 2) / 1000.0
            P = P @ T[:3, :3].T + T[:3, 3]
            polys = []
            for a, b in BOX_EDGES:
                polys += CR.visible_polylines(None, cam_model, pose, np.array([P[a], P[b]]), step=0.003)
            if polys:
                out.append((polys, QColor(STATUS_COLOR.get(r['status'], '#8f8e86')), f'bracket · {r["name"]}'))
        return out


register('brackets', 'Brackets', BracketsPage, menu='Analysis', icon='bracket', action_id=None,
         workspace=None, order=62)
