"""The Sensors and mounts page (spec §13): the active rig's sensors with their library device, mount, pose, stream
mode and on/off state, and the commands to add, duplicate and remove them.

Each sensor is a library device (core/library: figures from the maker's datasheet, "—" where it states none) plus a
mount: a static frame (world, cage, tripod, table) or any link of a robot in the project (RobotModel.link_names). The
pose is the sensor body frame (REP-103: x along the optical axis, y left, z up) in its parent frame, in millimetres
and degrees (roll, pitch, yaw about fixed x, y, z, as URDF). Mounting a sensor on another frame keeps it where it is
in the world by default (its pose is recomputed in the new parent at the current joint state); untick "Keep its place"
to keep the numbers instead.

The stream mode decides the imagers' image size (library.sensor_imagers): "largest listed" is the default; a mode
applies to every imager of the device that streams it, the others keep their own largest stream.

Every change is one undo step (the active variant's section).

Known limits: the mount tolerance is shown, not edited here (its default is ±1 mm, ±0.5°); brackets arrive in Phase
6; a sensor's unit calibration is cleared when its device changes (undo brings it back).
"""
from __future__ import annotations

import math
import traceback

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QHeaderView, QLineEdit, QTreeWidget,
                             QTreeWidgetItem, QVBoxLayout, QWidget)

from ...core.frames import Pose, quat_from_rpy, rpy_from_quat
from ...core.mathutil import fmt_num
from .. import rigedit as RE
from ..theme import THEME, mono
from ..widgets import ScrubSpinBox, Section, button, label
from . import register
from .base import StudyPage

KIND_LABEL = {'camera': 'Camera', 'depth': 'Depth', 'tactile': 'Tactile'}
STATIC_LABEL = {'world': 'World', 'cage': 'Cage', 'tripod': 'Tripod', 'table': 'Table'}
ROLE = Qt.ItemDataRole.UserRole


class SensorsPage(StudyPage):
    """Sensors and mounts: each sensor's library device, its mount (a static frame or a robot link), its pose in that
    frame, its stream mode and whether it is in use. Every change is one undo step."""
    STUDY = None
    TITLE = 'Sensors and mounts'
    MODEL = ('Model: none is run; the count is of the active variant\'s sensors and their datasheet devices '
             '(core/library).')
    HEADLINE = 'Sensors in the rig'

    def __init__(self, ctx):
        self._uid = ''
        self._building = False
        super().__init__(ctx)
        for b in (self.b_run, self.b_apply, self.b_preview, self.b_export):
            b.hide()
        self.b_add = button('Add camera', icon_name='plus', tip='A camera on a tripod looking at the task volume; '
                                                                  'pick its device below')
        self.b_dup = button('Duplicate', role='ghost', tip='A copy of the selected sensor, 50 mm to its left')
        self.b_del = button('Remove', role='ghost', icon_name='trash', tip='Remove the selected sensor from the rig')
        self.b_new = button('New custom device…', role='ghost', icon_name='sensor',
                            tip='A device of your own, every figure marked as given or not given')
        for i, b in enumerate((self.b_add, self.b_dup, self.b_del, self.b_new)):
            self.head.actions.insertWidget(i, b)
        self.b_add.clicked.connect(lambda: self.add_sensor())
        self.b_dup.clicked.connect(lambda: self.duplicate())
        self.b_del.clicked.connect(lambda: self.remove())
        self.b_new.clicked.connect(self._new_device)
        st = ctx.store
        st.changed.connect(self._changed)
        st.project_replaced.connect(self.refresh)
        st.selection_changed.connect(self._selection)
        self.refresh()

    # ---------------------------------------------------------------- layout
    def build_body(self, layout):
        self.table = QTreeWidget()
        self.table.setColumnCount(6)
        self.table.setHeaderLabels(['Sensor', 'Device', 'Mounted on', 'Position (mm)', 'Stream', 'In use'])
        self.table.setRootIsDecorated(False)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        hdr = self.table.header()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for k in range(2, 6):
            hdr.setSectionResizeMode(k, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._picked)
        layout.addWidget(self.table)
        self.editor_host = QWidget()
        self.editor_lay = QVBoxLayout(self.editor_host)
        self.editor_lay.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.editor_host)

    def empty(self):
        """No study: the headline is the rig's own (refresh fills it)."""

    # ---------------------------------------------------------------- state
    def current(self):
        v = self.ctx.store.project.variant
        s = v.sensor(self._uid)
        if s is None:
            sel = self.ctx.store.sel
            s = v.sensor(sel[1]) if sel[:1] == ('sensor',) and len(sel) > 1 else None
        if s is None and v.sensors:
            s = v.sensors[0]
        self._uid = s.uid if s is not None else ''
        return s

    def _changed(self, sections):
        if 'variant' in sections or 'variants' in sections or 'robots' in sections:
            self.refresh()

    def _selection(self):
        sel = self.ctx.store.sel
        if sel[:1] == ('sensor',) and len(sel) > 1 and sel[1] != self._uid:
            self._uid = sel[1]
            self.refresh()

    def _picked(self):
        if self._building:
            return
        its = self.table.selectedItems()
        if its:
            uid = its[0].data(0, ROLE)
            if uid and uid != self._uid:
                self._uid = uid
                self.ctx.store.select(('sensor', uid))
                self.refresh()

    # ---------------------------------------------------------------- filling
    def refresh(self):
        from ...core import library
        self._building = True
        try:
            p = self.ctx.store.project
            cur = self.current()
            self.table.clear()
            for s in p.variant.sensors:
                e = library.get(s.model_id) if s.model_id else None
                dev = (RE.device_title(e) if e is not None else (f'{s.model_id} (not in the library)' if s.model_id
                                                      else 'no device (default camera)'))
                pos = s.mount.pose.position * 1000
                stream = f'{s.stream[0]} × {s.stream[1]}' if len(s.stream) == 2 else 'largest'
                it = QTreeWidgetItem([s.name, dev, self._mount_text(s), ', '.join(fmt_num(v, 0) for v in pos), stream,
                                      'yes' if s.enabled else 'off'])
                it.setData(0, ROLE, s.uid)
                for k in (3, 4, 5):
                    it.setFont(k, mono(11))
                it.setForeground(0, THEME.cam(s.slot))
                if e is None:
                    it.setToolTip(1, library.DEFAULT_LABEL)
                else:
                    it.setToolTip(1, RE.entry_tooltip(e))
                self.table.addTopLevelItem(it)
                if cur is not None and s.uid == cur.uid:
                    it.setSelected(True)
            rows = max(1, min(self.table.topLevelItemCount(), 8))
            self.table.setFixedHeight(self.table.header().sizeHint().height() + rows * 22 + 6)
            self._headline()
            self._build_editor(cur)
            self.b_dup.setEnabled(cur is not None)
            self.b_del.setEnabled(cur is not None and not cur.locked)
        finally:
            self._building = False

    def _mount_text(self, s) -> str:
        m = s.mount
        if not m.on_link:
            return STATIC_LABEL.get(m.parent_frame, m.parent_frame) + ' (static)'
        r = self.ctx.store.project.robot(m.robot_uid)
        return f'{r.name if r else "missing robot"} ▸ {m.parent_frame}'

    def _headline(self):
        v = self.ctx.store.project.variant
        on = [s for s in v.sensors if s.enabled]
        bare = [s for s in v.sensors if not s.model_id]
        if not v.sensors:
            self.head.set('0', 'sensors', 'warning', 'none yet',
                          'Add a camera, or double-click a device in the Library, to start the rig.')
        elif bare:
            self.head.set(str(len(on)), f'of {len(v.sensors)} in use', 'warning', 'default camera',
                          f'{bare[0].name}{" and others have" if len(bare) > 1 else " has"} no device: its image and '
                          'figures use the default camera (90° × 60°, 1280 × 720 px, a default). Pick its device '
                          'below.')
        else:
            self.head.set(str(len(on)), f'of {len(v.sensors)} in use', 'good', 'devices chosen',
                          f'{v.name}: every sensor has a library device. Select one to change its mount, pose or '
                          'stream.')

    def _build_editor(self, s):
        while self.editor_lay.count():
            w = self.editor_lay.takeAt(0).widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        if s is None:
            return
        try:
            self.editor_lay.addWidget(self._editor(s))
        except Exception:                                 # noqa: BLE001 — explained, the traceback logged
            traceback.print_exc()
            self.editor_lay.addWidget(label(f'{s.name} cannot be edited here: details are in the log.', wrap=True))

    def _editor(self, s) -> QWidget:
        from ...core import library
        p = self.ctx.store.project
        sec = Section(s.name)
        self.name_edit = QLineEdit(s.name)
        self.name_edit.editingFinished.connect(lambda: self.rename(s.uid, self.name_edit.text()))
        sec.row('Name', self.name_edit)
        # the device
        self.device_box = QComboBox()
        self.device_box.addItem('No device (default camera)', '')
        self.device_box.setItemData(0, library.DEFAULT_LABEL, Qt.ItemDataRole.ToolTipRole)
        for e in library.entries():
            if e.kind not in RE.SENSOR_KINDS:
                continue
            self.device_box.addItem(f'{KIND_LABEL[e.kind]} · {RE.device_title(e)}'
                                    + (' (custom)' if e.is_custom else ''), e.id)
            self.device_box.setItemData(self.device_box.count() - 1, RE.entry_tooltip(e),
                                        Qt.ItemDataRole.ToolTipRole)
        if s.model_id and self.device_box.findData(s.model_id) < 0:
            self.device_box.addItem(f'{s.model_id} (not in the library)', s.model_id)
        self.device_box.setCurrentIndex(max(0, self.device_box.findData(s.model_id)))
        self.device_box.activated.connect(lambda i: self.set_device(s.uid, self.device_box.itemData(i)))
        sec.row('Device', self.device_box)
        # the mount
        self.mount_box = QComboBox()
        for f in RE.STATIC_FRAMES:
            self.mount_box.addItem(f'{STATIC_LABEL[f]} (static)', ('', f))
        for r in p.robots:
            try:
                from ...core.robot.model import get_model
                links = get_model(p, r.uid).link_names
            except Exception:                             # noqa: BLE001 — a robot that does not compile
                links = [lk.get('name') for lk in r.links if lk.get('name')]
            for lk in links:
                self.mount_box.addItem(f'{r.name} ▸ {lk}', (r.uid, lk))
        want = (s.mount.robot_uid if s.mount.on_link else '', s.mount.parent_frame)
        idx = next((i for i in range(self.mount_box.count()) if tuple(self.mount_box.itemData(i)) == want), -1)
        if idx < 0:
            self.mount_box.addItem(f'{s.mount.parent_frame} (missing)', want)
            idx = self.mount_box.count() - 1
        self.mount_box.setCurrentIndex(idx)
        self.mount_box.setToolTip('A static frame, or a robot link: a link-mounted sensor moves with the joints')
        self.keep_place = QCheckBox('Keep its place')
        self.keep_place.setChecked(True)
        self.keep_place.setToolTip('Mounting it elsewhere keeps the sensor where it is in the world (its pose is '
                                   'recomputed in the new frame at the current joint state)')
        self.mount_box.activated.connect(lambda i: self.set_mount(s.uid, self.mount_box.itemData(i)[1],
                                                                  self.mount_box.itemData(i)[0],
                                                                  self.keep_place.isChecked()))
        sec.row('Mounted on', self.mount_box)
        sec.row('', self.keep_place)
        # the pose
        self.pos_boxes = []
        for k, ax in enumerate('XYZ'):
            b = ScrubSpinBox(ax, 'length', 1, unit='mm')
            b.set_display_unit('mm')
            b.set_value(float(s.mount.pose.position[k]) * 1000)
            b.committed.connect(lambda _v: self.set_position(s.uid, [x.value() / 1000 for x in self.pos_boxes]))
            self.pos_boxes.append(b)
        sec.row('Position', *self.pos_boxes, tip='The sensor body origin in its parent frame (mm)')
        rpy = [math.degrees(a) for a in rpy_from_quat(s.mount.pose.quaternion)]
        self.rpy_boxes = []
        for k, ax in enumerate(('R', 'P', 'Y')):
            b = ScrubSpinBox(ax, 'angle', 1, unit='°')
            b.set_value(rpy[k])
            b.committed.connect(lambda _v: self.set_rpy(s.uid, [x.value() for x in self.rpy_boxes]))
            self.rpy_boxes.append(b)
        sec.row('Roll, pitch, yaw', *self.rpy_boxes,
                tip='Degrees, about fixed x, then y, then z (URDF); the body x axis is the optical axis')
        # the stream
        e = library.get(s.model_id) if s.model_id else None
        self.stream_box = QComboBox()
        self.stream_box.addItem('Largest listed (default)', None)
        for w, h, text in RE.stream_modes(e):
            self.stream_box.addItem(text, (w, h))
        cur = tuple(s.stream) if len(s.stream) == 2 else None
        i = next((i for i in range(self.stream_box.count()) if self.stream_box.itemData(i) == cur), 0)
        self.stream_box.setCurrentIndex(i)
        self.stream_box.setEnabled(e is not None and self.stream_box.count() > 1)
        self.stream_box.setToolTip('The stream mode sets each imager\'s image size (the datasheet\'s stream tables)')
        self.stream_box.activated.connect(lambda i: self.set_stream(s.uid, self.stream_box.itemData(i)))
        sec.row('Stream', self.stream_box)
        self.enabled_box = QCheckBox('In use (drawn, rendered and studied)')
        self.enabled_box.setChecked(s.enabled)
        self.enabled_box.toggled.connect(lambda on: self.set_enabled(s.uid, on))
        sec.row('', self.enabled_box)
        tol = s.mount.tolerance
        is_def = tol.to_dict() == type(tol)().to_dict()            # labelled "default" only while it is one
        sec.row('Tolerance', f'± {tol.position_mm:g} mm · ± {tol.rotation_deg:g}°' + (' (default)' if is_def else ''),
                tip='Mount tolerance (set it on Analysis ▸ Tolerances); the Monte Carlo studies carry it into every '
                    'figure')
        try:
            ims = library.sensor_imagers(s, tuple(s.stream) if len(s.stream) == 2 else None)
        except Exception as ex:                           # noqa: BLE001 — a bad calibration: said, not raised
            ims = []
            sec.add(label(str(ex), wrap=True, small=True))
        for im in ims:
            t = label(im.label, muted=True, wrap=True, small=True)
            t.setToolTip(im.camera.summary())
            sec.add(t)
        if e is not None:
            t = label(e.note, muted=True, wrap=True, small=True)
            t.setMaximumHeight(120)
            t.setToolTip(e.note)
            sec.add(t)
        return sec

    # ---------------------------------------------------------------- edits (one undo step each)
    def _edit(self, uid, fn, text):
        st = self.ctx.store
        s = st.project.variant.sensor(uid)
        if s is None:
            return False
        if s.locked:
            if self.ctx.window is not None:
                self.ctx.window.msg(f'{s.name} is locked: unlock it to change it.', 5000)
            return False

        def wrap(p):
            fn(p.variant.sensor(uid))
        st.edit('variant', wrap, text.replace('{name}', s.name))
        return True

    def add_sensor(self, model_id: str = ''):
        s = RE.add_sensor(self.ctx.store, model_id)
        self._uid = s.uid
        self.refresh()
        return s

    def duplicate(self, uid: str | None = None):
        s = RE.duplicate_sensor(self.ctx.store, uid or self._uid)
        if s is not None:
            self._uid = s.uid
            self.refresh()
        return s

    def remove(self, uid: str | None = None) -> bool:
        uid = uid or self._uid
        s = self.ctx.store.project.variant.sensor(uid)
        if s is None or s.locked:
            return False
        ok = RE.remove_sensor(self.ctx.store, uid)
        self._uid = ''
        self.refresh()
        return ok

    def rename(self, uid, text):
        t = (text or '').strip()
        s = self.ctx.store.project.variant.sensor(uid)
        if t and s is not None and t != s.name:
            self._edit(uid, lambda o: setattr(o, 'name', t), f'Renamed {{name}} to {t}')

    def set_device(self, uid, model_id):
        from ...core import library
        s = self.ctx.store.project.variant.sensor(uid)
        if s is None or (model_id or '') == s.model_id:
            return
        e = library.get(model_id) if model_id else None
        modes = {(w, h) for w, h, _t in RE.stream_modes(e)}
        had_cal = bool(s.calibration)

        def fn(o):
            o.model_id = e.id if e is not None else ''
            if e is not None:
                o.kind = RE.SENSOR_KINDS.get(e.kind, o.kind)
            if tuple(o.stream) not in modes:
                o.stream = []
            o.calibration = {}
        what = RE.device_title(e) if e is not None else 'no device (the default camera)'
        self._edit(uid, fn, f'Set {{name}}\'s device to {what}' + (' (its unit calibration was cleared)'
                                                                     if had_cal else ''))

    def set_mount(self, uid, frame, robot_uid='', keep_place=True):
        s = self.ctx.store.project.variant.sensor(uid)
        if s is None or s.locked:
            return False
        if (s.mount.parent_frame, s.mount.robot_uid if s.mount.on_link else '') == (frame, robot_uid or ''):
            return False
        return RE.remount(self.ctx.store, uid, frame, robot_uid or '', keep_place)

    def set_position(self, uid, xyz_m):
        xyz = np.asarray(xyz_m, float)
        self._edit(uid, lambda o: setattr(o.mount, 'pose', Pose(xyz, o.mount.pose.quaternion)), 'Moved {name}')

    def set_rpy(self, uid, rpy_deg):
        q = quat_from_rpy(*(math.radians(float(a)) for a in rpy_deg))
        self._edit(uid, lambda o: setattr(o.mount, 'pose', Pose(o.mount.pose.position, q)), 'Turned {name}')

    def set_stream(self, uid, mode):
        m = [int(mode[0]), int(mode[1])] if mode else []
        self._edit(uid, lambda o: setattr(o, 'stream', m),
                   'Set {name}\'s stream to ' + (f'{m[0]} × {m[1]}' if m else 'the largest listed'))

    def set_enabled(self, uid, on):
        self._edit(uid, lambda o: setattr(o, 'enabled', bool(on)), ('Switched on ' if on else 'Switched off ')
                   + '{name}')

    def _new_device(self):
        from ..dialogs.custom_device import CustomDeviceDialog
        CustomDeviceDialog.run(self.ctx, self.window())


register('sensors', 'Sensors and mounts', SensorsPage, menu='Sensors', icon='sensor', order=20)
