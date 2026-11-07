"""The Workspace page (spec §13 core pages): the task volumes, task objects and support surfaces every study works in,
listed and edited in one place, with a check that every object lies inside a task volume.

What it shows: the number of task volumes (the headline) with a status word: "ready" when there is a volume and
every task object's bounding box lies inside one, "objects outside" when some do not (the coverage, depth and grasp
studies measure inside the task volumes, so an object outside them is not measured), "no task volume" when there is
none. The action: Apply grows the task volume nearest to the objects left outside until it holds them with a margin
(MARGIN_M, 20 mm, a DexVantage default), or, with no volume at all, adds one around the objects (or the default
tabletop volume in front of the robot when there are no objects); one undo step. Preview draws that box dashed.

Below: the list (task volumes, objects, surfaces, and the lights, which the Lighting page edits) with Add volume,
Add box object, Add surface and Delete; and an editor for the selected item: a volume's name, kind, centre, size and
refinement flag; an object's name, shape, size, position, roll, pitch and yaw and its material tags (the lighting and
depth studies read them); a surface's name, height, size, centre, thickness and material. Every change is one undo
step of the project's 'workspace' section. Export writes the workspace and the check as JSON with provenance.

3D view: the selected item's outline (a volume's box, an object's oriented bounding box, a surface's rectangle) and,
with Preview, the suggested volume dashed. Camera view: the same outline projected into the selected camera's image
(not tested against occlusion: an outline behind the robot is still drawn).

The check and its suggestion are core/workspace_check.py's (no Qt).

Known limits: those of core/workspace_check.py (an object is inside when its axis-aligned bounding box is, within
1 mm; volumes are axis-aligned boxes); lights are listed, not edited, here (Lighting page); a mesh object's size
comes from its mesh and cannot be edited here. A volume, object or surface selected here is selected in the store
(the Outliner, the Inspector) and one selected in the Outliner is selected here.
"""
from __future__ import annotations

import math
import traceback

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QHeaderView, QLineEdit, QTreeWidget, QTreeWidgetItem,
                             QWidget)

from ...core import project as P
from ...core.frames import Pose, quat_from_rpy, rpy_from_quat
from ...core.workspace_check import (INSIDE_TOL_M, apply_suggestion, box_edges, check, object_outline,
                                     surface_outline)
from ..theme import STATUS, THEME
from ..widgets import ScrubSpinBox, button, label
from . import register
from .base import StudyPage, row

ROLE = Qt.ItemDataRole.UserRole
SHAPES = ('box', 'cylinder', 'sphere')
SIZE_HINT = {'box': ('x', 'y', 'z'), 'cylinder': ('radius', 'height', ''), 'sphere': ('radius', '', ''),
             'mesh': ('', '', '')}
KIND_TITLES = {'tabletop': 'Tabletop', 'shelf': 'Shelf', 'bin': 'Bin', 'handover': 'Handover zone',
               'custom': 'Custom'}
MATERIAL_TITLES = {'matte': 'Matte', 'glossy': 'Glossy', 'metallic': 'Metallic', 'transparent': 'Transparent',
                   'black': 'Black', 'ir_absorbing': 'IR-absorbing'}


# ====================================================================== the page
class WorkspacePage(StudyPage):
    """Workspace: the task volumes the studies measure in, the task objects and their material tags, and the support
    surfaces, edited here (each change one undo step). Every task object should lie inside a task volume: one
    outside is not measured by the coverage, depth, grasp or VLA studies."""
    STUDY = None
    TITLE = 'Workspace'
    HEADLINE = 'Task volumes'

    def build_body(self, layout):
        self._sel = None                     # ('volume' | 'object' | 'surface' | 'light', uid)
        self._preview = False
        self._filling = False
        self.b_run.setVisible(False)
        self.b_apply.setToolTip('Grow or add the task volume suggested (one undo step)')
        self.b_preview.setToolTip('Draw the suggested task volume in the 3D view without making it')
        self.b_export.setToolTip('The workspace and its check as JSON with provenance')
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['Item', 'Size', 'Notes'])
        self.tree.setRootIsDecorated(True)
        self.tree.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setMinimumHeight(140)
        self.tree.itemSelectionChanged.connect(self._picked)
        layout.addWidget(self.tree)
        self.b_vol = button('Add volume', role='ghost', icon_name='task-volume', tip='A new task volume (one undo '
                                                                                     'step)')
        self.b_obj = button('Add box object', role='ghost', icon_name='cube', tip='A 60 mm box on the table, a '
                                                                                  'default (one undo step)')
        self.b_surf = button('Add surface', role='ghost', icon_name='plus', tip='A 1 × 1 m table top at z = 0, a '
                                                                                'default (one undo step)')
        self.b_del = button('Delete', role='ghost', tip='Delete the selected volume, object or surface (one undo step)')
        self.b_light = button('Lighting page', role='ghost', icon_name='sun', tip='Lights are edited on the Lighting '
                                                                                  'page')
        self.b_vol.clicked.connect(self.add_volume)
        self.b_obj.clicked.connect(self.add_object)
        self.b_surf.clicked.connect(self.add_surface)
        self.b_del.clicked.connect(self.delete)
        self.b_light.clicked.connect(self._open_lighting)
        layout.addWidget(row(self.b_vol, self.b_obj, self.b_surf, self.b_del, self.b_light))
        self.ed_title = label('', bold=True)
        layout.addWidget(self.ed_title)
        self.editor = QWidget()
        g = QGridLayout(self.editor)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(8)
        self.e_name = QLineEdit()
        self.e_name.setToolTip('Its name in the project')
        self.e_name.editingFinished.connect(lambda: self._set('name', self.e_name.text().strip()))
        self.e_kind = QComboBox()
        self.e_kind.currentIndexChanged.connect(lambda _i: self._kind_changed())
        self.l_kind = label('Kind', ink2=True)

        def trio(kind, dec, unit, tip):
            out = []
            for _ in range(3):
                b = ScrubSpinBox('', kind, dec, unit, step=1.0, width=82)
                b.setToolTip(tip)
                out.append(b)
            return out
        self.e_pos = trio('length', 1, 'mm', 'Centre (volume, surface) or position (object) in the world, mm')
        self.e_size = trio('length', 1, 'mm', 'Edges in mm (an object: box edges, or radius and height)')
        self.e_rot = trio('angle', 1, '°', 'Roll, pitch and yaw in degrees (about fixed x, then y, then z)')
        for b in self.e_pos:
            b.committed.connect(lambda _v: self._set_vec('pos'))
        for b in self.e_size:
            b.committed.connect(lambda _v: self._set_vec('size'))
        for b in self.e_rot:
            b.committed.connect(lambda _v: self._set_vec('rot'))
        self.l_pos, self.l_size, self.l_rot = label('Centre', ink2=True), label('Size', ink2=True), \
            label('Roll, pitch, yaw', ink2=True)
        self.e_refine = QCheckBox('Refinement region (the finer voxel size applies inside)')
        self.e_refine.toggled.connect(lambda on: self._set('refine', bool(on)))
        self.e_thick = ScrubSpinBox('', 'length', 1, 'mm', step=1.0, width=82)
        self.e_thick.setToolTip('0: a rectangle at its height; more: a slab that thick below it')
        self.e_thick.committed.connect(lambda v: self._set('thickness', max(0.0, float(v)) / 1000.0))
        self.l_thick = label('Thickness', ink2=True)
        self.mat_w = QWidget()
        ml = QGridLayout(self.mat_w)
        ml.setContentsMargins(0, 0, 0, 0)
        self.e_mat = {}
        for i, m in enumerate(P.MATERIAL_TAGS):
            cb = QCheckBox(MATERIAL_TITLES.get(m, m))
            cb.setToolTip('A material tag the lighting and depth studies read (glare, IR wash-out, depth holes)')
            cb.toggled.connect(lambda _on: self._set_materials())
            self.e_mat[m] = cb
            ml.addWidget(cb, i // 3, i % 3)
        self.l_mat = label('Materials', ink2=True)
        self.e_smat = QComboBox()
        for m in P.MATERIAL_TAGS:
            self.e_smat.addItem(MATERIAL_TITLES.get(m, m), m)
        self.e_smat.setToolTip('The surface\'s material tag (the lighting study reads it)')
        self.e_smat.currentIndexChanged.connect(lambda _i: self._set('material', self.e_smat.currentData()))
        self.l_smat = label('Material', ink2=True)
        rows = [(label('Name', ink2=True), [self.e_name]), (self.l_kind, [self.e_kind]),
                (self.l_pos, self.e_pos), (self.l_size, self.e_size), (self.l_rot, self.e_rot),
                (self.l_thick, [self.e_thick]), (self.l_smat, [self.e_smat]), (self.l_mat, [self.mat_w])]
        for r, (lab, ws) in enumerate(rows):
            g.addWidget(lab, r, 0)
            for k, w in enumerate(ws):
                g.addWidget(w, r, 1 + k, 1, 3 if len(ws) == 1 else 1)
        g.addWidget(self.e_refine, len(rows), 0, 1, 4)
        g.setColumnStretch(4, 1)
        layout.addWidget(self.editor)
        self.ed_note = label('', muted=True, small=True, wrap=True)
        layout.addWidget(self.ed_note)
        self.ctx.store.changed.connect(self._changed)
        self.ctx.store.project_replaced.connect(self._replaced)
        self.ctx.store.selection_changed.connect(self._store_selection)
        self.refresh()

    # ---------------------------------------------------------------- the check and the list
    def empty(self):
        """No study: the headline is the workspace check (refresh())."""

    def refresh(self):
        p = self.ctx.store.project
        try:
            self.doc = check(p)
        except Exception:                               # noqa: BLE001 — the page stays usable
            traceback.print_exc()
            self.doc = None
            self.head.set('—', '', 'critical', 'could not check', 'The workspace check failed; the details are in '
                                                                  'the log.')
            return
        d = self.doc
        n = d['volumes']
        self.head.set(str(n), 'volume' if n == 1 else 'volumes', d['status'], d['word'],
                      d['advice'] + (f' Suggested: {d["suggestion"]["text"]}.' if d['suggestion'] else ''),
                      f'{d["objects"]} objects, {d["surfaces"]} surfaces, {d["lights"]} lights. An object counts as '
                      f'inside when its bounding box lies inside a volume within {INSIDE_TOL_M * 1000:g} mm.')
        self.suggestion_ready(d['suggestion'] is not None)
        if d['suggestion'] is None:
            self._preview = False
        self._fill_tree()
        self._fill_editor()

    def _fill_tree(self):
        p = self.ctx.store.project
        out = {o['uid'] for o in (self.doc or {}).get('outside', [])}
        self.tree.blockSignals(True)
        self.tree.clear()

        def group(title, n):
            it = QTreeWidgetItem([f'{title} ({n})', '', ''])
            it.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.tree.addTopLevelItem(it)
            return it

        def mm(v):
            return ' × '.join(f'{x * 1000:.0f}' for x in v) + ' mm'
        gv = group('Task volumes', len(p.workspace.volumes))
        for v in p.workspace.volumes:
            it = QTreeWidgetItem([v.name, mm(v.size), KIND_TITLES.get(v.kind, v.kind) + (' · refinement' if v.refine
                                                                                         else '')])
            it.setData(0, ROLE, ('volume', v.uid))
            gv.addChild(it)
        go = group('Objects', len(p.workspace.objects))
        for o in p.workspace.objects:
            size = 'mesh' if o.shape == 'mesh' else mm([x for x, h in zip(o.size, SIZE_HINT.get(o.shape, ('',) * 3))
                                                        if h])
            note = ', '.join(MATERIAL_TITLES.get(m, m).lower() for m in o.materials)
            it = QTreeWidgetItem([o.name, f'{o.shape} {size}', note + (' · outside every volume' if o.uid in out
                                                                       else '')])
            if o.uid in out:
                it.setForeground(2, QColor(STATUS['warning']))
            it.setData(0, ROLE, ('object', o.uid))
            go.addChild(it)
        gs = group('Surfaces', len(p.workspace.surfaces))
        for k, s in enumerate(p.workspace.surfaces):
            if not isinstance(s, dict):
                continue
            it = QTreeWidgetItem([str(s.get('name') or f'Surface {k + 1}'), mm(s.get('size') or (0, 0)),
                                  f'z {float(s.get("z", 0.0)) * 1000:.0f} mm · '
                                  f'{MATERIAL_TITLES.get(s.get("material"), "matte (not set)").lower()}'])
            it.setData(0, ROLE, ('surface', str(s.get('uid') or k)))
            gs.addChild(it)
        gl = group('Lights', len(p.workspace.lights))
        for k, li in enumerate(p.workspace.lights):
            if not isinstance(li, dict):
                continue
            it = QTreeWidgetItem([str(li.get('name') or f'Light {k + 1}'), str(li.get('kind') or ''),
                                  'edited on the Lighting page'])
            it.setData(0, ROLE, ('light', str(li.get('uid') or k)))
            gl.addChild(it)
        self.tree.expandAll()
        sel = None
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            for j in range(top.childCount()):
                if tuple(top.child(j).data(0, ROLE) or ()) == tuple(self._sel or ()):
                    sel = top.child(j)
        if sel is not None:
            sel.setSelected(True)
        elif self._sel is not None and self._find(*self._sel) is None:
            self._sel = None
        self.tree.blockSignals(False)

    def _find(self, kind, uid):
        w = self.ctx.store.project.workspace
        if kind == 'volume':
            return next((v for v in w.volumes if v.uid == uid), None)
        if kind == 'object':
            return next((o for o in w.objects if o.uid == uid), None)
        pool = w.surfaces if kind == 'surface' else w.lights if kind == 'light' else []
        return next((s for k, s in enumerate(pool) if isinstance(s, dict) and str(s.get('uid') or k) == uid), None)

    def selected(self):
        return self._find(*self._sel) if self._sel else None

    def select(self, kind, uid):
        """Select an item of the list (tests, the Outliner through the store's selection)."""
        self._sel = (kind, uid)
        if kind in ('volume', 'object', 'surface') and tuple(self.ctx.store.sel[:2]) != (kind, uid):
            self.ctx.store.select((kind, uid))
        self._fill_tree()
        self._fill_editor()
        self._redraw()

    def _picked(self):
        its = self.tree.selectedItems()
        d = its[0].data(0, ROLE) if its else None
        if d:
            self.select(*d)

    def _store_selection(self):
        sel = tuple(self.ctx.store.sel[:2])
        if len(sel) == 2 and sel[0] in ('volume', 'object', 'surface') and sel != tuple(self._sel or ()):
            self._sel = sel
            self._fill_tree()
            self._fill_editor()

    # ---------------------------------------------------------------- the editor
    def _fill_editor(self):
        o = self.selected()
        kind = self._sel[0] if self._sel else None
        self._filling = True
        try:
            self.editor.setVisible(o is not None and kind != 'light')
            if o is None:
                self.ed_title.setText('Select a volume, object or surface to edit it.')
                self.ed_note.setText('')
                return
            if kind == 'light':
                self.ed_title.setText(f'{o.get("name") or "Light"}: a light')
                self.ed_note.setText('Lights are placed and set on the Lighting page (Analysis ▸ Lighting and glare).')
                return
            is_v, is_o, is_s = kind == 'volume', kind == 'object', kind == 'surface'
            self.ed_title.setText({'volume': 'Task volume', 'object': 'Task object', 'surface': 'Surface'}[kind])
            self.e_name.setText(o.name if not is_s else str(o.get('name') or ''))
            self.e_kind.blockSignals(True)
            self.e_kind.clear()
            if is_v:
                for k in P.VOLUME_KINDS:
                    self.e_kind.addItem(KIND_TITLES.get(k, k), k)
                self.e_kind.setCurrentIndex(max(0, self.e_kind.findData(o.kind)))
                self.e_kind.setToolTip('What the volume is (a handover zone is where bimanual handoff grasps happen)')
            elif is_o:
                for k in SHAPES + (('mesh',) if o.shape == 'mesh' else ()):
                    self.e_kind.addItem(k, k)
                self.e_kind.setCurrentIndex(max(0, self.e_kind.findData(o.shape)))
                self.e_kind.setEnabled(o.shape != 'mesh')
                self.e_kind.setToolTip('The object\'s shape (a mesh object keeps its mesh)')
            self.e_kind.blockSignals(False)
            self.e_kind.setEnabled(not (is_o and o.shape == 'mesh'))
            for w in (self.l_kind, self.e_kind):
                w.setVisible(not is_s)
            self.l_pos.setText('Position' if is_o else 'Centre')
            if is_v:
                pos, size = o.center, o.size
            elif is_o:
                pos, size = o.pose.position, o.size
            else:
                c = list(o.get('center') or (0.0, 0.0))[:2]
                pos = (c + [0.0, 0.0])[:2] + [float(o.get('z', 0.0))]
                size = (list(o.get('size') or (0.0, 0.0))[:2] + [0.0])[:3]
            for b, x in zip(self.e_pos, pos):
                b.setValue(float(x) * 1000.0)
            hint = SIZE_HINT.get(o.shape, ('x', 'y', 'z')) if is_o else (('x', 'y', '') if is_s else ('x', 'y', 'z'))
            for b, x, h in zip(self.e_size, size, hint):
                b.setValue(float(x) * 1000.0)
                b.setVisible(bool(h))
                b.setEnabled(not (is_o and o.shape == 'mesh'))
            self.e_pos[2].setToolTip('Height of the table top (z), mm' if is_s else
                                     'Centre (volume) or position (object) in the world, mm')
            self.l_size.setText('Size (' + ', '.join(h for h in hint if h) + ')' if any(hint) else 'Size')
            for w in (self.l_rot, *self.e_rot):
                w.setVisible(is_o)
            if is_o:
                for b, a in zip(self.e_rot, rpy_from_quat(o.pose.quaternion)):
                    b.setValue(math.degrees(a))
            self.e_refine.setVisible(is_v)
            if is_v:
                self.e_refine.setChecked(bool(o.refine))
            for w in (self.l_thick, self.e_thick, self.l_smat, self.e_smat):
                w.setVisible(is_s)
            if is_s:
                self.e_thick.setValue(float(o.get('thickness') or 0.0) * 1000.0)
                self.e_smat.setCurrentIndex(max(0, self.e_smat.findData(o.get('material') or 'matte')))
            for w in (self.l_mat, self.mat_w):
                w.setVisible(is_o)
            if is_o:
                for m, cb in self.e_mat.items():
                    cb.setChecked(m in o.materials)
            out = {x['uid'] for x in (self.doc or {}).get('outside', [])}
            self.ed_note.setText('This object lies outside every task volume: the studies do not measure it.'
                                 if is_o and o.uid in out else
                                 ('A mesh object: its size comes from its mesh.' if is_o and o.shape == 'mesh' else ''))
        finally:
            self._filling = False

    def _edit(self, fn, text, key):
        """One undo step of the 'workspace' section on the selected item; consecutive edits of one field merge."""
        kind, uid = self._sel

        def wrap(p):
            o = None
            w = p.workspace
            if kind == 'volume':
                o = next(v for v in w.volumes if v.uid == uid)
            elif kind == 'object':
                o = next(v for v in w.objects if v.uid == uid)
            else:
                o = next(s for k, s in enumerate(w.surfaces) if isinstance(s, dict) and str(s.get('uid') or k) == uid)
            fn(o)
        self.ctx.store.edit('workspace', wrap, text, merge_key=('workspace', kind, uid, key))

    def _label(self):
        o = self.selected()
        return (o.get('name') if isinstance(o, dict) else getattr(o, 'name', '')) or 'the item'

    def _set(self, key, value):
        if self._filling or not self._sel or self._sel[0] == 'light' or self.selected() is None:
            return
        kind = self._sel[0]
        o = self.selected()
        if key == 'name':
            if not value:
                self._fill_editor()
                return
            cur = o.get('name') if isinstance(o, dict) else o.name
            if cur == value:
                return
        if kind == 'surface':
            if o.get(key) == value:
                return

            def fn(s):
                s[key] = value
        else:
            if getattr(o, key, None) == value:
                return

            def fn(x):
                setattr(x, key, value)
        self._edit(fn, f'Workspace · {self._label()} {key}', key)

    def _kind_changed(self):
        if self._filling or not self._sel:
            return
        v = self.e_kind.currentData()
        if self._sel[0] == 'volume':
            self._set('kind', v)
        elif self._sel[0] == 'object' and v in SHAPES:
            o = self.selected()
            if o is None or o.shape == v:
                return
            r = max(float(x) for x in o.size) / 2
            size = {'box': (2 * r,) * 3, 'cylinder': (r, 2 * r, 0.0), 'sphere': (r, 0.0, 0.0)}[v]

            def fn(x):
                x.shape, x.size = v, size
            self._edit(fn, f'Workspace · {self._label()} shape', 'shape')

    def _set_vec(self, which):
        if self._filling or not self._sel or self.selected() is None:
            return
        kind = self._sel[0]
        box = {'pos': self.e_pos, 'size': self.e_size, 'rot': self.e_rot}[which]
        vals = [float(b.value()) for b in box]
        if which == 'rot':
            q = quat_from_rpy(*(math.radians(a) for a in vals))

            def fn(x):
                x.pose = Pose(np.asarray(x.pose.position, float), q)
        elif which == 'pos':
            m = [v / 1000.0 for v in vals]
            if kind == 'volume':
                def fn(x):
                    x.center = tuple(m)
            elif kind == 'object':
                def fn(x):
                    x.pose = Pose(np.asarray(m), np.asarray(x.pose.quaternion, float))
            else:
                def fn(x):
                    x['center'], x['z'] = [m[0], m[1]], m[2]
        else:
            m = [max(0.001, v / 1000.0) for v in vals]
            if kind == 'surface':
                def fn(x):
                    x['size'] = m[:2]
            elif kind == 'object':
                o = self.selected()
                if o.shape == 'mesh':
                    return
                hint = SIZE_HINT.get(o.shape, ('x', 'y', 'z'))
                m = [v if h else float(old) for v, h, old in zip(m, hint, o.size)]

                def fn(x):
                    x.size = tuple(m)
            else:
                def fn(x):
                    x.size = tuple(m)
        self._edit(fn, f'Workspace · {self._label()} {"rotation" if which == "rot" else which}', which)

    def _set_materials(self):
        if self._filling or not self._sel or self._sel[0] != 'object':
            return
        tags = [m for m, cb in self.e_mat.items() if cb.isChecked()]
        if not tags:                                     # an object always has a tag (core/project.py: matte)
            self._filling = True
            self.e_mat['matte'].setChecked(True)
            self._filling = False
            tags = ['matte']
        o = self.selected()
        if o is None or list(o.materials) == tags:
            return

        def fn(x):
            x.materials = tags
        self._edit(fn, f'Workspace · {self._label()} materials', 'materials')

    # ---------------------------------------------------------------- add, delete, apply
    def add_volume(self):
        def fn(p):
            v = P.Volume(name=f'Task volume {len(p.workspace.volumes) + 1}', kind='custom')
            p.workspace.volumes.append(v)
            return v
        v = self.ctx.store.edit('workspace', fn, 'Added a task volume')
        self.select('volume', v.uid)
        return v

    def add_object(self):
        def fn(p):
            c = p.workspace.volumes[0].center if p.workspace.volumes else (0.45, 0.0, 0.15)
            lo = p.workspace.volumes[0].bounds()[0][2] if p.workspace.volumes else 0.0
            o = P.TaskObject(name=f'Box {len(p.workspace.objects) + 1}', size=(0.06, 0.06, 0.06),
                             pose=Pose([float(c[0]), float(c[1]), float(lo) + 0.03]))
            p.workspace.objects.append(o)
            return o
        o = self.ctx.store.edit('workspace', fn, 'Added a box object')
        self.select('object', o.uid)
        return o

    def add_surface(self):
        def fn(p):
            s = {'uid': P.new_uid(), 'name': f'Surface {len(p.workspace.surfaces) + 1}', 'z': 0.0, 'size': [1.0, 1.0],
                 'center': [0.45, 0.0], 'material': 'matte'}
            p.workspace.surfaces.append(s)
            return s
        s = self.ctx.store.edit('workspace', fn, 'Added a surface')
        self.select('surface', s['uid'])
        return s

    def delete(self):
        if not self._sel or self._sel[0] == 'light' or self.selected() is None:
            return
        kind, uid = self._sel
        name = self._label()

        def fn(p):
            w = p.workspace
            if kind == 'volume':
                w.volumes = [v for v in w.volumes if v.uid != uid]
            elif kind == 'object':
                w.objects = [o for o in w.objects if o.uid != uid]
            else:
                w.surfaces = [s for k, s in enumerate(w.surfaces)
                              if not (isinstance(s, dict) and str(s.get('uid') or k) == uid)]
        self.ctx.store.edit('workspace', fn, f'Deleted {name}')
        self._sel = None
        if tuple(self.ctx.store.sel[:2]) == (kind, uid):
            self.ctx.store.select(())
        self.refresh()

    def apply(self):
        sug = (self.doc or {}).get('suggestion')
        if not sug:
            return None
        v = self.ctx.store.edit('workspace', lambda p: apply_suggestion(p, sug),
                                f'Workspace · {sug["text"][:1].upper()}{sug["text"][1:]}')
        self._preview = False
        self.select('volume', v.uid)
        return v

    def preview(self):
        self._preview = (self.doc or {}).get('suggestion') is not None and not self._preview
        self._redraw()

    def _open_lighting(self):
        win = self.ctx.window
        if win is not None and 'lighting' in getattr(win, 'pages', {}):
            win.open_page('lighting')

    # ---------------------------------------------------------------- store
    def _changed(self, sections):
        if 'workspace' in sections or 'meshes' in sections:
            self.refresh()

    def _replaced(self):
        self._sel = None
        self._preview = False
        self.refresh()

    def activate(self):
        self.refresh()

    def _redraw(self):
        if self.ctx.window is not None:
            self.ctx.window.viewport.update()

    # ---------------------------------------------------------------- drawing
    def _outline(self):
        o = self.selected()
        if o is None or not self._sel:
            return None
        kind = self._sel[0]
        try:
            if kind == 'volume':
                return box_edges(*o.bounds())
            if kind == 'object':
                return object_outline(o, self.ctx.store.project)
            if kind == 'surface':
                return surface_outline(o)
        except Exception:                               # noqa: BLE001 — an outline never breaks the view
            traceback.print_exc()
        return None

    def _suggested_box(self):
        sug = (self.doc or {}).get('suggestion')
        if not (self._preview and sug):
            return None
        c, s = np.asarray(sug['center'], float), np.asarray(sug['size'], float)
        return box_edges(c - s / 2, c + s / 2)

    def overlays(self, vp) -> list:
        from ..viewport3d import LineDraw
        out = []
        seg = self._outline()
        if seg is not None:
            out.append(LineDraw(seg, THEME.color('ink', 0.95), 2.0, segments=True, depth=False))
        seg = self._suggested_box()
        if seg is not None:
            out.append(LineDraw(seg, QColor(STATUS['good']), 1.6, dash=True, segments=True, dash_len=0.02))
        return out

    def camview_overlays(self, cam_model, pose) -> list:
        from ...core import camrender as CR
        out = []
        for seg, col, text in ((self._outline(), THEME.color('ink', 0.95), 'selected'),
                               (self._suggested_box(), QColor(STATUS['good']), 'suggested volume')):
            if seg is None:
                continue
            lines = []
            for a, b in seg:
                lines += CR.visible_polylines(None, cam_model, pose, np.array([a, b]))
            lines = [x for x in lines if len(x) >= 2]
            if lines:
                out.append((lines, col, text))
        return out

    def page_context(self) -> dict:
        ctx = super().page_context()
        d = self.doc or {}
        ctx['workspace'] = {k: d.get(k) for k in ('volumes', 'objects', 'surfaces', 'lights')}
        ctx['outside'] = [o['name'] for o in d.get('outside', [])]
        return ctx


register('workspace', 'Workspace', WorkspacePage, menu='Workspace', icon='task-volume', order=12)
