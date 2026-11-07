"""The Lighting page (spec §7.8): the lights of the workspace, how much of the object and gripper each camera sees in
specular saturation, the predicted loss of valid depth from IR wash-out and dark or clear surfaces, glare maps, and
what to change.

What it shows: the headline (the worst camera's saturated share of the visible object and gripper area over the
working poses, with a status word; thresholds 1 %, 5 % and 15 % are defaults, and the word "relative" when the
camera's sensitivity is not stated), the glare map of the chosen camera (the image in cells: the largest saturated
share in each over the poses), the cameras and depth sensors, and the recommendations each with the figure before and
after. Apply makes the best one (one undo step): a polariser setting, a light moved or dimmed, or the camera tilted.

The lights are edited here: Add (a kind, or a light from the library with its stated figures), Remove, the position,
which way it faces, its size, its luminous flux, luminance and IR irradiance (each with its source: type where the
figure is from), whether it is on and the share it is run at; each edit is one undo step. Every light is drawn in the
3D view on every page once this page has been opened; the saturated points of the chosen camera while it is shown.
The Camera view outlines the glare map's cells over the image of the camera it shows.
"""
from __future__ import annotations

import copy
import traceback

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QHeaderView, QLineEdit, QMenu, QTreeWidget,
                             QTreeWidgetItem, QWidget)

from ...core import lights as LT
from ...core.studies import lighting as LS
from ..charts import ChartBase
from ..theme import STATUS, THEME, mono, sans
from ..viewport3d import LineDraw
from ..widgets import ScrubSpinBox, SegmentedControl, button, label
from . import register
from .base import StudyPage, row

NOT_STATED = 'not stated'
LIGHT_COLOR = '#e8b931'           # the lights' glyphs (a category colour, always with the light's name in the table)
FACINGS = {'down': [1.0, 0.0, 0.0, 0.0], 'up': [0.0, 0.0, 0.0, 1.0], '+x': [0.0, 0.7071067811865476, 0.0,
                                                                            0.7071067811865476],
           '−x': [0.0, -0.7071067811865476, 0.0, 0.7071067811865476],
           '+y': [-0.7071067811865476, 0.0, 0.0, 0.7071067811865476],
           '−y': [0.7071067811865476, 0.0, 0.0, 0.7071067811865476]}
FIGURE_FIELDS = (('luminous_flux_lm', 'Luminous flux', 'lm'), ('luminance_cd_m2', 'Luminance', 'cd/m²'),
                 ('ir_irradiance_w_m2', 'IR irradiance', 'W/m²'))


def _pct(x, nd=1) -> str:
    return '—' if x is None else f'{x:.{nd}f} %'


def _tree(headers) -> QTreeWidget:
    t = QTreeWidget()
    t.setRootIsDecorated(False)
    t.setHeaderLabels(headers)
    t.setUniformRowHeights(True)
    for k in range(1, len(headers)):
        t.header().setSectionResizeMode(k, QHeaderView.ResizeMode.ResizeToContents)
    t.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    t.header().setStretchLastSection(False)
    t.setMinimumHeight(60)
    return t


def _fit(t: QTreeWidget):
    n = t.topLevelItemCount()
    rh = t.sizeHintForRow(0) if n else 20
    t.setFixedHeight(t.header().sizeHint().height() + max(n, 1) * max(rh, 18) + 2 * t.frameWidth() + 18)


def _cells(it, cols):
    for k in cols:
        it.setFont(k, mono(11))
        it.setTextAlignment(k, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)


def facing_of(q) -> str:
    """The facing whose quaternion the light's is (else 'custom')."""
    q = np.asarray(q, float)
    for k, v in FACINGS.items():
        if abs(abs(float(np.dot(q, v))) - 1.0) < 1e-6:
            return k
    return 'custom'


class GlareMap(ChartBase):
    """The glare map of one camera: the image in grid cells, each filled by the largest saturated share of the visible
    area in it over the working poses (the critical colour, stronger for more), cells near saturation outlined."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = 'Glare map'
        self.subtitle = 'saturated share per image cell, worst pose'
        self.cam = None
        self.near = 0.5
        self.setMinimumHeight(220)

    def set_camera(self, cam: dict | None, near_level: float):
        self.cam, self.near = cam, near_level
        self.tip = None
        self.update()

    def _rect(self):
        if not self.cam:
            return QRectF(0, 22, self.width(), self.height() - 26)
        W, H = self.cam['glare_map']['image']
        avail = QRectF(0, 22, self.width(), self.height() - 26)
        s = min(avail.width() / W, avail.height() / H)
        return QRectF(0, 22, W * s, H * s)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.paint_title(p)
        r = self._rect()
        p.fillRect(r, THEME.color('raised'))
        p.setPen(QPen(THEME.color('strong'), 1))
        p.drawRect(r)
        if not self.cam or not self.cam.get('glare_map'):
            p.setFont(sans(11))
            p.setPen(THEME.color('muted'))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, 'Run the study to see the glare map')
            p.end()
            return
        gm = self.cam['glare_map']
        gw, gh = gm['grid']
        cw, ch = r.width() / gw, r.height() / gh
        crit = QColor(STATUS['critical'])
        warn = QColor(STATUS['warning'])
        for j in range(gh):
            for i in range(gw):
                k = j * gw + i
                s, lv = gm['sat'][k], gm['level'][k]
                cell = QRectF(r.left() + i * cw, r.top() + j * ch, cw, ch)
                if s > 0:
                    c = QColor(crit)
                    c.setAlphaF(0.25 + 0.75 * min(1.0, s))
                    p.fillRect(cell, c)
                elif lv >= self.near:
                    p.setPen(QPen(warn, 1))
                    p.drawRect(cell.adjusted(1, 1, -1, -1))
        p.setFont(sans(10))
        p.setPen(THEME.color('muted'))
        p.drawText(QRectF(r.left(), r.bottom() - 14, r.width() - 4, 14),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   'filled: saturated · outlined: near saturation')
        self.paint_tip(p)
        p.end()

    def mouseMoveEvent(self, e):
        if not self.cam or not self.cam.get('glare_map'):
            return
        gm = self.cam['glare_map']
        r = self._rect()
        x, y = e.position().x(), e.position().y()
        if not r.contains(QPointF(x, y)):
            self.tip = None
            self.update()
            return
        gw, gh = gm['grid']
        i = min(int((x - r.left()) / r.width() * gw), gw - 1)
        j = min(int((y - r.top()) / r.height() * gh), gh - 1)
        k = j * gw + i
        self.set_tip(QPointF(x, y), f'{gm["sat"][k] * 100:.0f} % saturated',
                     f'peak {gm["level_is"]}: {gm["level"][k]:.3g}', f'cell {i + 1}, {j + 1} of {gw} × {gh}')
        self.update()


class LightingPage(StudyPage):
    """Lighting and glare: the share of the object and gripper area each camera sees in specular saturation over the
    working poses (the BRDF of each material under the stated lights; relative to an 18 % grey auto-exposure when the
    camera's sensitivity is not stated), the IR wash-out and dark or clear surfaces that cost an active depth sensor
    valid depth, and the change that helps most."""
    STUDY = LS.SID
    TITLE = 'Lighting and glare'
    HEADLINE = 'Object and gripper area in specular saturation (worst camera)'

    # ---------------------------------------------------------------- body
    def build_body(self, layout):
        self.sel_cam = ''
        self.sel_light = ''
        self._preview = None
        self.c_cam = QComboBox()
        self.c_cam.setToolTip('The camera whose glare map, saturated points and Camera view outline are shown')
        self.c_cam.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.c_cam.setMinimumContentsLength(16)
        self.c_cam.setPlaceholderText('Run with lights to choose')
        self.pose = SegmentedControl([('auto', 'Working poses'), ('current', 'Pose on screen')], 'auto')
        self.pose.setToolTip('Working poses: the grasp set\'s reachable grasps (home without a set). Pose on screen: '
                             'the joint state of the 3D view only.')
        layout.addWidget(row(label('Camera', ink2=True), self.c_cam, self.pose))
        self.map = GlareMap()
        layout.addWidget(self.map)
        self.map_note = label('', muted=True, wrap=True, small=True)
        layout.addWidget(self.map_note)
        layout.addWidget(label('Cameras', bold=True))
        self.cams = _tree(['Camera', 'Imager', 'Model', 'Saturated', 'Worst pose', 'Surfaces', 'Polariser'])
        layout.addWidget(self.cams)
        layout.addWidget(label('Active depth sensors: IR wash-out and surfaces', bold=True))
        self.deps = _tree(['Sensor', 'Projector', 'Ambient IR', 'Pattern', 'Washed out', 'Dark', 'Clear',
                           'Depth loss'])
        layout.addWidget(self.deps)
        layout.addWidget(label('What to change', bold=True))
        self.recs = _tree(['Change', 'Figure', 'Before', 'After'])
        layout.addWidget(self.recs)
        # ---- the lights
        layout.addWidget(label('Lights (each edit is one undo step)', bold=True))
        self.lights = _tree(['Light', 'Kind', 'Position', 'Faces', 'Visible light', 'IR', 'On'])
        layout.addWidget(self.lights)
        self.b_add = button('Add light', icon_name='plus', tip='Add a light of a kind (enter its figures below), or '
                            'one from the library with its stated figures')
        self.add_menu = QMenu(self.b_add)
        self.b_add.setMenu(self.add_menu)
        self.add_menu.aboutToShow.connect(self._fill_add_menu)
        self.b_remove = button('Remove', role='ghost', icon_name='trash', tip='Remove the selected light (one undo '
                               'step)')
        self.b_remove.clicked.connect(self.remove_selected)
        layout.addWidget(row(self.b_add, self.b_remove))
        ed = QWidget()
        g = QGridLayout(ed)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(8)
        self.f_pos = []
        for i, ax in enumerate('xyz'):
            sb = ScrubSpinBox(ax, 'plain', 0, 'mm', step=10.0, width=100)
            sb.setRange(-20000.0, 20000.0)
            sb.setToolTip(f'The light\'s centre, world {ax} (mm)')
            sb.committed.connect(lambda v, k=i: self._set_pos(k, v))
            self.f_pos.append(sb)
            g.addWidget(sb, 0, i)
        self.c_face = QComboBox()
        for k in FACINGS:
            self.c_face.addItem(f'faces {k}', k)
        self.c_face.setToolTip('Which way the light\'s face (its local +z) points, in the world')
        self.c_face.activated.connect(self._set_facing)
        g.addWidget(self.c_face, 0, 3)
        self.f_size = []
        for i, ax in enumerate(('w', 'h')):
            sb = ScrubSpinBox(ax, 'plain', 0, 'mm', step=10.0, width=100)
            sb.setRange(0.0, 20000.0)
            sb.setToolTip('The emitting face: width along the light\'s x, height along its y (mm); a ring: its outer '
                          'diameter')
            sb.committed.connect(lambda v, k=i: self._set_size(k, v))
            self.f_size.append(sb)
            g.addWidget(sb, 1, i)
        self.f_dim = ScrubSpinBox('', 'plain', 0, '%', step=5.0, width=90)
        self.f_dim.setRange(0.0, 100.0)
        self.f_dim.setToolTip('The share of its stated output the light is run at')
        self.f_dim.committed.connect(self._set_dim)
        g.addWidget(self.f_dim, 1, 2)
        self.on = QCheckBox('On')
        self.on.toggled.connect(self._set_on)
        g.addWidget(self.on, 1, 3)
        self.f_fig = {}
        for i, (key, title, unit) in enumerate(FIGURE_FIELDS):
            sb = ScrubSpinBox('', 'plain', 1, unit, step=10.0, width=130)
            sb.setRange(0.0, 1e9)
            sb.setToolTip(f'{title} ({unit}) as stated by the maker or measured by you; empty: not stated')
            src = QLineEdit()
            src.setPlaceholderText('where the figure comes from (datasheet URL and date, or "my meter reading")')
            sb.committed.connect(lambda v, k=key: self._set_figure(k, v))
            src.editingFinished.connect(lambda k=key: self._set_source(k))
            clear = button('Not stated', role='ghost', tip=f'Clear the {title.lower()} (not stated)')
            clear.clicked.connect(lambda _c=False, k=key: self._clear_figure(k))
            g.addWidget(label(title), 2 + i, 0)
            g.addWidget(sb, 2 + i, 1)
            g.addWidget(src, 2 + i, 2, 1, 2)
            g.addWidget(clear, 2 + i, 4)
            self.f_fig[key] = (sb, src)
        self.f_at = ScrubSpinBox('d', 'plain', 2, 'm', step=0.1, width=100)
        self.f_at.setRange(0.0, 100.0)
        self.f_at.setToolTip('The distance on the light\'s axis its IR irradiance was measured or stated at (empty for '
                             'a window: then it is the irradiance at the glass)')
        self.f_at.committed.connect(self._set_ir_at)
        g.addWidget(self.f_at, 4, 5)
        self.editor = ed
        layout.addWidget(ed)
        self.light_note = label('', muted=True, wrap=True, small=True)
        layout.addWidget(self.light_note)
        # ---- the task objects' material tags (the BRDF of core/materials.py per tag)
        layout.addWidget(label('Task object materials (each edit is one undo step)', bold=True))
        self.obj_box = QComboBox()
        self.obj_box.setToolTip('The task object whose material tags are shown; the study\'s BRDF, IR albedo and '
                                'transmission come from them (core/materials.py: the roughness and albedos are '
                                'defaults; a measured value goes in the study\'s settings)')
        self.obj_box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.obj_box.setPlaceholderText('No task object (Workspace ▸ Add object)')
        self.obj_box.currentIndexChanged.connect(lambda _i: self._fill_material_checks())
        self.mat_checks = {}
        from ...core.materials import MATERIALS
        from ...core.project import MATERIAL_TAGS
        for tag in MATERIAL_TAGS:
            cb = QCheckBox(tag.replace('_', '-') if tag == 'ir_absorbing' else tag)
            cb.setToolTip(MATERIALS[tag].title if tag in MATERIALS else tag)
            cb.toggled.connect(lambda on, t=tag: self.set_material(t, on))
            self.mat_checks[tag] = cb
        layout.addWidget(row(self.obj_box))
        layout.addWidget(row(*self.mat_checks.values()))
        self.mat_note = label('', muted=True, wrap=True, small=True)
        layout.addWidget(self.mat_note)
        self.assume = label('', muted=True, wrap=True, small=True)
        layout.addWidget(self.assume)
        self.c_cam.currentIndexChanged.connect(self._cam_changed)
        self.lights.currentItemChanged.connect(lambda cur, _prev: self._light_chosen(cur))
        st = self.ctx.store
        st.changed.connect(lambda _s: (self._fill_lights(), self._fill_objects()))
        st.project_replaced.connect(self._replaced)
        self.ctx.overlay_sources.append(self.light_glyphs)
        self._fill_lights()
        self._fill_objects()

    # ---------------------------------------------------------------- the task objects' materials
    def _fill_objects(self):
        objs = list(self.ctx.store.project.workspace.objects)
        cur = self.obj_box.currentData()
        self.obj_box.blockSignals(True)
        self.obj_box.clear()
        for o in objs:
            self.obj_box.addItem(o.name, o.uid)
        i = self.obj_box.findData(cur)
        self.obj_box.setCurrentIndex(i if i >= 0 else (0 if objs else -1))
        self.obj_box.blockSignals(False)
        self._fill_material_checks()

    def _object(self):
        uid = self.obj_box.currentData()
        return next((o for o in self.ctx.store.project.workspace.objects if o.uid == uid), None)

    def _fill_material_checks(self):
        o = self._object()
        tags = set(o.materials) if o is not None else set()
        for t, cb in self.mat_checks.items():
            cb.blockSignals(True)
            cb.setChecked(t in tags)
            cb.setEnabled(o is not None)
            cb.blockSignals(False)
        self.obj_box.setEnabled(o is not None)
        self.mat_note.setText('No task objects: add one (Workspace ▸ Add box object) to give it a material.'
                              if o is None else
                              f'{o.name}: {", ".join(o.materials)}. Run again to see the change.')

    def set_material(self, tag: str, on: bool, uid: str | None = None) -> bool:
        """Add or remove a material tag on a task object (one undo step); an object keeps at least one tag (matte
        when the last one is removed)."""
        from ...core.project import MATERIAL_TAGS
        uid = uid or self.obj_box.currentData()
        o = next((x for x in self.ctx.store.project.workspace.objects if x.uid == uid), None)
        if o is None or tag not in MATERIAL_TAGS:
            return False
        tags = [t for t in o.materials if t != tag] + ([tag] if on else [])
        tags = [t for t in MATERIAL_TAGS if t in tags] or ['matte']
        if tags == list(o.materials):
            self._fill_material_checks()
            return False

        def fn(p):
            for x in p.workspace.objects:
                if x.uid == uid:
                    x.materials = list(tags)
        self.ctx.store.edit('workspace', fn, f'{o.name}: materials {", ".join(tags)}')
        return True

    # ---------------------------------------------------------------- lights
    def _lights(self) -> list:
        try:
            return LT.lights_of(self.ctx.store.project)
        except Exception:                               # noqa: BLE001 — the table must not take the page down
            traceback.print_exc()
            return []

    def _replaced(self):
        self.doc = None
        self.sel_light = ''
        self._fill_lights()
        self._fill_objects()
        self.empty()

    def _fill_lights(self):
        ls = self._lights()
        self.lights.blockSignals(True)
        self.lights.clear()
        cur = None
        for lt in ls:
            d = LT.describe(lt)
            pos = ', '.join(f'{x * 1000:.0f}' for x in lt['pose']['position']) + ' mm'
            it = QTreeWidgetItem([lt['name'], LT.KIND_TITLE[lt['kind']], pos, facing_of(lt['pose']['quaternion']),
                                  'ok' if d['vis']['ok'] else NOT_STATED, 'ok' if d['ir']['ok'] else NOT_STATED,
                                  ('on' if lt['enabled'] else 'off') + (f' · {lt["dim"] * 100:.0f} %'
                                                                        if lt['dim'] < 1 else '')])
            it.setData(0, Qt.ItemDataRole.UserRole, lt['uid'])
            it.setToolTip(4, d['vis']['text'])
            it.setToolTip(5, d['ir']['text'])
            it.setFont(2, mono(11))
            self.lights.addTopLevelItem(it)
            if lt['uid'] == self.sel_light:
                cur = it
        if cur is None and self.lights.topLevelItemCount():
            cur = self.lights.topLevelItem(0)
        if cur is not None:
            self.lights.setCurrentItem(cur)
        self.lights.blockSignals(False)
        _fit(self.lights)
        self._light_chosen(cur)
        if self.ctx.window is not None:
            self.ctx.window.viewport.update()

    def _light(self) -> dict | None:
        return next((x for x in self._lights() if x['uid'] == self.sel_light), None)

    def _light_chosen(self, it):
        self.sel_light = it.data(0, Qt.ItemDataRole.UserRole) if it is not None else ''
        lt = self._light()
        self.editor.setEnabled(lt is not None)
        self.b_remove.setEnabled(lt is not None)
        if lt is None:
            self.light_note.setText('No lights yet: Add the lab\'s lights (from the library, or a kind with the '
                                    'figures you measured). The study needs at least one with a stated figure.')
            return
        for i, sb in enumerate(self.f_pos):
            sb.set_value(lt['pose']['position'][i] * 1000)
        f = facing_of(lt['pose']['quaternion'])
        i = self.c_face.findData(f)
        self.c_face.blockSignals(True)
        self.c_face.setCurrentIndex(i if i >= 0 else 0)
        self.c_face.blockSignals(False)
        s = lt.get('size_m') or [None, None]
        for i, sb in enumerate(self.f_size):
            sb.set_value(None if s[i] is None else s[i] * 1000, mixed=s[i] is None)
            sb.setEnabled(lt['kind'] != 'point')
        self.f_dim.set_value(lt['dim'] * 100)
        self.on.blockSignals(True)
        self.on.setChecked(lt['enabled'])
        self.on.blockSignals(False)
        for key, (sb, src) in self.f_fig.items():
            v = lt.get(key)
            sb.set_value(None if v is None else v['value'], mixed=v is None)
            src.setText('' if v is None else v['source'])
        ir = lt.get('ir_irradiance_w_m2') or {}
        self.f_at.set_value(ir.get('at_m'), mixed=not ir.get('at_m'))
        d = LT.describe(lt)
        txt = [d['vis']['text'], d['ir']['text']] + [d[k]['text'] for k in ('sun', 'dim') if k in d]
        if lt.get('library_id'):
            txt.append(f'From the library entry {lt["library_id"]}; figures you change say your source instead.')
        self.light_note.setText(' '.join(txt))

    def _fill_add_menu(self):
        from ...core import library as LB
        m = self.add_menu
        m.clear()
        for k in LT.KINDS:
            m.addAction(f'{LT.KIND_TITLE[k]} (enter its figures)', lambda kk=k: self.add_light(kind=kk))
        es = LB.entries('light')
        if es:
            m.addSeparator()
            for e in es:
                m.addAction(f'{e.title} (library)', lambda i=e.id: self.add_light(library_id=i))

    def _edit_lights(self, fn, text, merge_key=None):
        def wrap(p):
            ls = [LT.clean_light(x) for x in (p.workspace.lights or []) if isinstance(x, dict)]
            fn(ls)
            p.workspace.lights = ls
        self.ctx.store.edit('workspace', wrap, text, merge_key=merge_key)

    def _edit_light(self, fn, text, merge_key=None):
        uid = self.sel_light

        def f(ls):
            for i, x in enumerate(ls):
                if x['uid'] == uid:
                    fn(x)
                    ls[i] = LT.clean_light(x)
        if uid:
            self._edit_lights(f, text, merge_key)

    def add_light(self, kind: str = 'panel', library_id: str | None = None) -> str:
        """Add a light (one undo step) above the task volume's centre (or 1.5 m above the origin), facing down."""
        p = self.ctx.store.project
        c = np.array([0.4, 0.0, 0.0])
        if p.workspace.volumes:
            c = np.asarray(p.workspace.volumes[0].center, float)
        pos = [float(c[0]), float(c[1]), float(c[2]) + 1.2]
        try:
            lt = (LT.from_library(library_id, {'position': pos, 'quaternion': FACINGS['down']}) if library_id
                  else LT.new_light(kind, pos))
        except LT.LightError as e:
            if self.ctx.window is not None:
                self.ctx.window.msg(str(e), 6000)
            return ''
        self._edit_lights(lambda ls: ls.append(lt), f'Add light {lt["name"]}')
        self.sel_light = lt['uid']
        self._fill_lights()
        return lt['uid']

    def remove_selected(self):
        lt = self._light()
        if lt is None:
            return
        uid = lt['uid']
        self._edit_lights(lambda ls: ls.__setitem__(slice(None), [x for x in ls if x['uid'] != uid]),
                          f'Remove light {lt["name"]}')
        self.sel_light = ''
        self._fill_lights()

    def _set_pos(self, k, v_mm):
        def fn(x):
            x['pose']['position'][k] = float(v_mm) / 1000.0
        self._edit_light(fn, 'Move light', merge_key=('light_pos', self.sel_light, k))

    def _set_facing(self, i):
        f = self.c_face.itemData(i)

        def fn(x):
            x['pose']['quaternion'] = list(FACINGS[f])
        self._edit_light(fn, f'Light faces {f}')

    def _set_size(self, k, v_mm):
        def fn(x):
            s = list(x.get('size_m') or LT.DEFAULT_SIZE_M.get(x['kind']) or [0.3, 0.3])
            s[k] = max(float(v_mm), 1.0) / 1000.0
            x['size_m'] = s
        self._edit_light(fn, 'Light size')

    def _set_dim(self, v):
        self._edit_light(lambda x: x.__setitem__('dim', float(v) / 100.0), f'Light run at {v:.0f} %')

    def _set_on(self, on):
        self._edit_light(lambda x: x.__setitem__('enabled', bool(on)), 'Light on' if on else 'Light off')

    def _source_text(self, key) -> str:
        t = self.f_fig[key][1].text().strip()
        return t or 'your entry (no source given)'

    def _set_figure(self, key, v):
        src = self._source_text(key)

        def fn(x):
            x[key] = {**(x.get(key) or {}), 'value': float(v), 'source': src}
        self._edit_light(fn, 'Light figure')

    def _set_source(self, key):
        lt = self._light()
        if lt is None or lt.get(key) is None:
            return
        src = self._source_text(key)
        if src == lt[key]['source']:
            return
        self._edit_light(lambda x: x[key].__setitem__('source', src), 'Light figure source')

    def _clear_figure(self, key):
        self._edit_light(lambda x: x.__setitem__(key, None), 'Light figure not stated')

    def _set_ir_at(self, v):
        lt = self._light()
        if lt is None or lt.get('ir_irradiance_w_m2') is None:
            if self.ctx.window is not None:
                self.ctx.window.msg('Enter the IR irradiance first, then the distance it is for.', 5000)
            return

        def fn(x):
            if v > 0:
                x['ir_irradiance_w_m2']['at_m'] = float(v)
            else:
                x['ir_irradiance_w_m2'].pop('at_m', None)
        self._edit_light(fn, 'IR irradiance distance')

    # ---------------------------------------------------------------- run
    def live_q(self) -> dict:
        return {uid: [float(x) for x in np.asarray(q, float).reshape(-1)] for uid, q in self.ctx.store.q.items()}

    def run(self):
        if self.pose.value() == 'current':
            self.ctx.analysis.run(self.STUDY, {'poses': 'home', 'q': self.live_q()})
        else:
            self.ctx.analysis.run(self.STUDY, {})

    # ---------------------------------------------------------------- result
    def empty(self):
        self.head.set('—', '', 'warning', 'not run yet', 'Add the lab\'s lights below, then Run to see the glare and '
                                                         'the IR wash-out the cameras will get.')
        self.map.set_camera(None, 0.5)

    def _cam(self, doc=None):
        doc = doc or self.doc or {}
        cs = doc.get('cameras') or []
        return next((c for c in cs if c['uid'] == self.sel_cam), cs[0] if cs else None)

    def _near_level(self, c) -> float:
        st = (self.doc or {}).get('settings') or {}
        return 0.5 if c and c.get('absolute') else 0.5 * float(st.get('sat_ratio', LS.DEFAULTS['sat_ratio']))

    def show_result(self, doc: dict):
        self._preview = None
        h = doc.get('headline') or {}
        v = h.get('value')
        tip = ('Status: low glare under 1 %, some glare under 5 %, glare under 15 % (defaults; settings good_pct, '
               'warn_pct, serious_pct); a depth sensor losing 5, 15 or 30 % of what it sees (defaults '
               'depth_good_pct, depth_warn_pct, depth_serious_pct) raises it. ' + (doc.get('sat_ratio_source') or ''))
        text = h.get('text', '') + (f' {h["action"]}' if h.get('action') else '')
        self.head.set('—' if v is None else f'{v:.1f}', '' if v is None else '%' + (' rel.' if h.get('relative')
                                                                                    else ''),
                      h.get('status', 'warning'), h.get('word', ''), text, tip)
        self.suggestion_ready(bool(doc.get('suggestion')))
        cs = doc.get('cameras') or []
        self.c_cam.blockSignals(True)
        self.c_cam.clear()
        for c in cs:
            self.c_cam.addItem(c['name'], c['uid'])
        uids = [c['uid'] for c in cs]
        if self.sel_cam not in uids:
            self.sel_cam = h.get('camera') or (uids[0] if uids else '')
        if self.sel_cam in uids:
            self.c_cam.setCurrentIndex(uids.index(self.sel_cam))
        self.c_cam.blockSignals(False)
        self._show_map()
        self.cams.clear()
        for c in cs:
            it = QTreeWidgetItem([c['name'], c['imager'], 'absolute' if c['absolute'] else 'relative',
                                  _pct(c['sat_pct']), f'{_pct(c["sat_pct_worst"])} ({c["worst_pose"] or "—"})',
                                  _pct(c['surface_sat_pct']), 'fitted' if c['polariser'] else 'none'])
            _cells(it, (3, 4, 5))
            it.setToolTip(2, c['mode'])
            it.setToolTip(1, c.get('imager_why', ''))
            it.setForeground(3, QColor(STATUS.get(c['status'], STATUS['warning'])))
            it.setToolTip(3, f'{c["word"]}: the saturated share of the object and gripper area it sees, mean over '
                             'the working poses')
            self.cams.addTopLevelItem(it)
        _fit(self.cams)
        self.deps.clear()
        for d in doc.get('depth') or []:
            pj = '—' if d['projector_w'] is None else f'{d["projector_w"] * 1000:g} mW'
            amb = d.get('ambient_ir_w_m2_median')
            pat = d.get('projector_w_m2_median')
            it = QTreeWidgetItem([d['name'], pj, '—' if amb is None else f'{amb:.3g} W/m²',
                                  '—' if pat is None else f'{pat:.3g} W/m²', _pct(d['washout_pct']),
                                  _pct(d['dark_pct']), _pct(d['through_pct']), _pct(d['loss_pct'])])
            _cells(it, range(1, 8))
            it.setToolTip(1, d['projector_note'] if d['projector_w'] is not None else
                          'not stated by the maker: ' + d['projector_note'])
            it.setToolTip(2, 'Median in-band IR irradiance from the lights and the sun on the surfaces it sees')
            it.setToolTip(3, 'Median irradiance of the projector\'s pattern on those surfaces')
            self.deps.addTopLevelItem(it)
        _fit(self.deps)
        self.recs.clear()
        for r in doc.get('recommendations') or []:
            it = QTreeWidgetItem([r['text'], r['figure'], _pct(r['before']), _pct(r['after'])])
            _cells(it, (2, 3))
            it.setToolTip(0, r['text'] + (f' Source: {r["cite"]}.' if r.get('cite') else ''))
            self.recs.addTopLevelItem(it)
        _fit(self.recs)
        notes = list(doc.get('notes') or [])
        notes += [f'{s["name"]}: {s["why"]}' for s in doc.get('skipped') or []]
        mats = doc.get('materials') or []
        if mats:
            notes.append('Materials: ' + '; '.join(
                f'{m["title"]} α {m["alpha"]:g}, F0 {m["f0_vis"]:.3f}, albedo {m["albedo"]:g}, IR albedo '
                f'{m["ir_albedo"]:g}' for m in mats) + ' (roughness and albedos are DexVantage defaults; F0 from '
                'measured refractive indices).')
        rd = doc.get('renderer') or {}
        if rd:
            notes.append(f'Occlusion cube maps on {rd.get("renderer") or "numpy"}' + (f' ({rd["how"]})'
                                                                                     if rd.get('how') else '') + '.')
        self.assume.setText('\n'.join(notes))
        self._redraw()

    def _show_map(self):
        c = self._cam()
        self.map.set_camera(c, self._near_level(c))
        if c is None:
            self.map_note.setText('')
            return
        self.map_note.setText(f'{c["name"]} · {c["imager"]}: {c["mode"]}. Filled cells: the largest saturated '
                              'share of the visible area in the cell over the working poses; outlined: near '
                              'saturation (half the threshold).')

    def _cam_changed(self, i):
        self.sel_cam = self.c_cam.itemData(i) or ''
        self._show_map()
        self._redraw()

    # ---------------------------------------------------------------- Apply / Preview
    def apply(self):
        sug = (self.doc or {}).get('suggestion')
        if sug:
            self.apply_recommendation(sug)
            self.run()

    def apply_recommendation(self, r: dict):
        """Make one recommendation (one undo step)."""
        a = r.get('apply') or {}
        st = self.ctx.store
        if a.get('section') == 'analysis':
            def fn(p):
                s = copy.deepcopy(p.analysis.studies.get(LS.SID) or {})
                s.setdefault('sensors', {}).setdefault(a['sensor_uid'], {})['polariser'] = bool(a['polariser'])
                p.analysis.studies[LS.SID] = s
            st.edit('analysis', fn, f'Polariser on {r.get("target_name", "the camera")}')
        elif a.get('section') == 'workspace':
            uid = a['light_uid']

            def fl(ls):
                for i, x in enumerate(ls):
                    if x['uid'] == uid:
                        if 'position' in a:
                            x['pose']['position'] = [float(v) for v in a['position']]
                        if 'dim' in a:
                            x['dim'] = float(a['dim'])
                        ls[i] = LT.clean_light(x)
            self._edit_lights(fl, r['text'].split(':')[0])
        elif a.get('section') == 'variant':
            from ...core.frames import Pose

            def fv(p):
                s = p.variant.sensor(a['sensor_uid'])
                if s is not None and not s.locked:
                    s.mount.pose = Pose.from_dict(a['mount_pose'])
            st.edit('variant', fv, r['text'].split(':')[0])

    def preview(self):
        sug = (self.doc or {}).get('suggestion')
        self._preview = sug if sug and self._preview is None else None
        if self.ctx.window is not None:
            msg = 'Preview off'
            if self._preview:
                msg = ('Preview: the dashed glyph is the light at its suggested place' if
                       self._preview['kind'] == 'move_light' else f'Preview: {self._preview["text"]}')
            self.ctx.window.msg(msg, 8000)
            self.ctx.window.viewport.update()

    # ---------------------------------------------------------------- 3D
    def _redraw(self):
        if self.ctx.window is not None:
            self.ctx.window.viewport.update()

    def light_glyphs(self, vp) -> list:
        """Every light of the workspace as a glyph (an overlay source: on every page once this one is built)."""
        out = []
        for lt in self._lights():
            segs = glyph_segments(lt)
            if len(segs):
                col = QColor(LIGHT_COLOR) if lt['enabled'] else THEME.color('muted')
                out.append(LineDraw(segs, col, 1.6 if lt['uid'] == self.sel_light else 1.1, segments=True,
                                    dash=not lt['enabled'], dash_len=0.02))
        pv = self._preview
        if pv and pv.get('kind') == 'move_light':
            lt = next((x for x in self._lights() if x['uid'] == pv['target']), None)
            if lt is not None:
                lt2 = copy.deepcopy(lt)
                lt2['pose']['position'] = list(pv['apply']['position'])
                out.append(LineDraw(glyph_segments(lt2), THEME.color('ink', 0.9), 1.4, dash=True, segments=True,
                                    dash_len=0.02))
        return out

    def overlays(self, vp) -> list:
        c = self._cam()
        if not c or not c.get('sat_points_mm'):
            return []
        P = np.asarray(c['sat_points_mm'], float).reshape(-1, 3) / 1000.0
        hl = 0.004
        segs = [np.stack([P - d, P + d], 1) for d in np.eye(3) * hl]
        return [LineDraw(np.concatenate(segs), QColor(STATUS['critical']), 1.2, segments=True)]

    # ---------------------------------------------------------------- Camera view
    def camview_overlays(self, cam_model, pose) -> list:
        """The glare map of the camera the view shows, as cell outlines over its image: saturated cells in the
        critical colour, cells near saturation in the warning colour."""
        doc = self.doc
        if not doc:
            return []
        cv = getattr(self.ctx.window, 'camview', None) if self.ctx.window is not None else None
        shown = cv.view.sensor() if cv is not None and hasattr(cv, 'view') else None
        if shown is None:
            return []
        c = next((x for x in doc.get('cameras') or [] if x['uid'] == shown.uid), None)
        if c is None or not c.get('glare_map'):
            return []
        return glare_polylines(c, cam_model.width, cam_model.height, self._near_level(c))


def glare_polylines(cam: dict, width: int, height: int, near_level: float) -> list:
    """[(polylines, colour, legend)] of a camera result's glare map at an image of width × height pixels."""
    gm = cam['glare_map']
    gw, gh = gm['grid']
    sx, sy = width / gw, height / gh
    sat, near = [], []
    for j in range(gh):
        for i in range(gw):
            k = j * gw + i
            x0, y0 = i * sx - 0.5, j * sy - 0.5
            box = np.array([[x0, y0], [x0 + sx, y0], [x0 + sx, y0 + sy], [x0, y0 + sy], [x0, y0]])
            if gm['sat'][k] > 0:
                sat.append(box)
            elif gm['level'][k] >= near_level:
                near.append(box)
    out = []
    if sat:
        out.append((sat, STATUS['critical'], 'saturated (glare map)'))
    if near:
        out.append((near, STATUS['warning'], 'near saturation'))
    return out


def glyph_segments(lt: dict) -> np.ndarray:
    """(S, 2, 3) world segments of a light's glyph: an area light's outline, a cross and its facing; a ring's two
    circles; a point's star; a spot's cone."""
    from ...core.frames import quat_to_matrix
    R = quat_to_matrix(np.asarray(lt['pose']['quaternion'], float))
    t = np.asarray(lt['pose']['position'], float)
    segs = []

    def poly(P):
        Q = np.asarray(P, float)
        for a, b in zip(Q, np.roll(Q, -1, 0)):
            segs.append([a, b])
    k = lt['kind']
    if k in ('panel', 'window') and lt.get('size_m'):
        w, h = lt['size_m']
        c = np.array([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]])
        poly(c)
        segs += [[c[0], c[2]], [c[1], c[3]]]
        segs.append([[0, 0, 0], [0, 0, 0.15]])
    elif k == 'ring':
        s = lt.get('size_m') or [0.3, 0.3]
        Ro = 0.5 * min(s)
        ri = 0.5 * float(lt.get('inner_m') or 0.0)
        a = np.linspace(0, 2 * np.pi, 33)[:-1]
        for r in (Ro, ri) if ri > 0 else (Ro,):
            poly(np.stack([r * np.cos(a), r * np.sin(a), np.zeros_like(a)], 1))
        segs.append([[0, 0, 0], [0, 0, 0.15]])
    elif k == 'spot':
        half = np.radians((lt.get('beam_deg') or {}).get('value', 40.0)) / 2
        L = 0.25
        r = L * np.tan(half)
        a = np.linspace(0, 2 * np.pi, 25)[:-1]
        rim = np.stack([r * np.cos(a), r * np.sin(a), np.full_like(a, L)], 1)
        poly(rim)
        for i in range(0, 24, 6):
            segs.append([[0, 0, 0], rim[i]])
    else:
        for d in np.eye(3) * 0.05:
            segs.append([-d, d])
        e = np.array([1, 1, 1]) / np.sqrt(3) * 0.035
        segs += [[-e, e], [[-e[0], e[1], -e[2]], [e[0], -e[1], e[2]]]]
    S = np.asarray(segs, float).reshape(-1, 2, 3)
    return S @ R.T + t


register('lighting', 'Lighting and glare', LightingPage, menu='Analysis', icon='sun', action_id=None,
         workspace='placement', order=45)
