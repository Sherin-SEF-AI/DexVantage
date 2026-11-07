"""The Teleop overlap page (spec §7.4): how much of the task volume a teleoperator can drive the follower through
without a dead zone, the dead zones with the poses that cause them, and the teleop devices themselves.

What it shows: the headline share of the task volume teleoperable through at least one device (status word "clear"
at 90 % or more, "gaps" from 50 %, "poor" below; labelled bands), the device editor (add, edit and delete devices,
each change one undo step; a device figure is entered with its source or left "not stated"; Add from library makes a
device from a library teleop entry, core/teleop_devices.py, with the entry's stated figures and sources), a table per
device (comfortable for the leader, teleoperable, dead zones), the dead zones by cause with their size and the
follower's pose at their worst voxel (a click poses the robot there), the follower's reach and dexterity, the bimanual
overlap and handover figures when two followers have devices, and the dead-zone voxels in the 3D view coloured by
cause (the zones' boxes in the Camera view). Apply makes the suggested follower base move or mapping-offset move
(one undo step); Preview draws it; Explain says how every figure is computed; Export writes the result with
provenance.
"""
from __future__ import annotations

import traceback

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QHeaderView, QLineEdit, QMessageBox, QTreeWidget,
                             QTreeWidgetItem, QWidget)

from ...core import teleop_devices as TD
from ...core.studies import teleop as TS
from ..theme import STATUS, THEME, mono
from ..widgets import ScrubSpinBox, button, label
from . import register
from .base import StudyPage, row

CAUSE_SLOT = {'unreachable': 7, 'self-collision': 1, 'singularity': 6, 'joint limit': 3}     # THEME.cam slots
CUBE_SHRINK = 0.8                       # voxels drawn at this share of their edge, so neighbours stay apart
SELF = '__self__'


def cause_hex(cause: str) -> str:
    return THEME.cam_hex(CAUSE_SLOT.get(cause, 0))


def cubes(centers, half) -> np.ndarray:
    """(12 K, 3, 3) triangles of axis-aligned cubes of half-edge `half` round each centre (K, 3)."""
    C = np.asarray(centers, float).reshape(-1, 3)
    s = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6),
             (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    V = C[:, None, :] + half * s[None]
    return np.concatenate([V[:, list(f)] for f in faces], axis=0).astype(np.float32) if len(C) else \
        np.zeros((0, 3, 3), np.float32)


def _pct(v) -> str:
    return '—' if v is None else f'{v:.1f} %'


class TeleopPage(StudyPage):
    """Teleop overlap: each device's comfortable workspace (a leader arm's joints within 0.8 of their ranges, a VR
    tracking volume or space-mouse bounds, scaled and offset) is mapped into the follower's frame; over the task voxels
    the follower's IK, manipulability, condition number, joint-limit distance and self-collision margin flag the dead
    zones where the leader is comfortable but the follower is not."""
    STUDY = 'teleop'
    TITLE = 'Teleop overlap'
    HEADLINE = 'Task volume teleoperable without a dead zone'

    # ---------------------------------------------------------------- body
    def build_body(self, layout):
        self._overlay = None
        self._preview = []
        self._busy = False
        self._pending_vol = {}
        # ---- devices
        layout.addWidget(label('Teleop devices', bold=True))
        self.dev_box = QComboBox()
        self.dev_box.setMinimumWidth(200)
        self.dev_box.setToolTip('The device to edit')
        self.kind_add = QComboBox()
        for k in TS.KINDS:
            self.kind_add.addItem(TS.KIND_TITLES[k], k)
        self.kind_add.setToolTip('The kind of device Add creates')
        self.b_add = button('Add', role='ghost', tip='Add a device of this kind (one undo step)')
        self.b_del = button('Delete', role='ghost', tip='Delete this device (one undo step)')
        layout.addWidget(row(self.dev_box, self.kind_add, self.b_add, self.b_del))
        # library teleop devices (core/teleop_devices.py): their stated figures come with their sources
        self.lib_box = QComboBox()
        self.lib_box.setMinimumWidth(260)
        self.lib_box.setToolTip('A teleop device from the library; its stated figures come with their sources')
        self.b_add_lib = button('Add from library', role='ghost',
                                tip='Add this library device, driving the chosen follower (one undo step)')
        layout.addWidget(row(self.lib_box, self.b_add_lib))
        self._fill_library()
        self.dev_hint = label('', muted=True, wrap=True)
        layout.addWidget(self.dev_hint)
        g = QWidget()
        grid = QGridLayout(g)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        self.name_edit = QLineEdit()
        self.name_edit.setMinimumWidth(260)
        self.kind_box = QComboBox()
        for k in TS.KINDS:
            self.kind_box.addItem(TS.KIND_TITLES[k], k)
        self.follower_box = QComboBox()
        self.follower_box.setToolTip('The robot the device drives')
        self.leader_box = QComboBox()
        self.leader_box.setToolTip('The leader arm\'s kinematics: the follower\'s own model (an SO-100 leader, a GELLO '
                                   'of the same joints) or another robot of the project')
        self.map_box = QComboBox()
        self.map_box.addItem('Joint to joint', 'joint')
        self.map_box.addItem('Cartesian (tool position)', 'cartesian')
        self.scale = ScrubSpinBox('', 'plain', 3, '×', step=0.01, width=110)
        self.scale.setRange(-100.0, 100.0)
        self.scale.setToolTip('Joint: follower joint = scale × leader joint. Cartesian: device positions are scaled '
                              'by this before the offset')
        self.off = [ScrubSpinBox('', 'plain', 1, 'mm', step=1.0, width=96) for _ in range(3)]
        for sb, a in zip(self.off, 'xyz'):
            sb.setRange(-10000.0, 10000.0)
            sb.setToolTip(f'Mapping offset {a}: where the device frame\'s origin sits in the follower\'s base frame')
        self.yaw = ScrubSpinBox('', 'plain', 1, '°', step=1.0, width=86)
        self.yaw.setRange(-180.0, 180.0)
        self.yaw.setToolTip('Mapping offset: turn of the device frame about the follower\'s z axis')
        self.vol_kind = QComboBox()
        self.vol_kind.addItem('Not stated', '')
        self.vol_kind.addItem('Box', 'box')
        self.vol_kind.addItem('Cylinder', 'cylinder')
        self.vol_kind.setToolTip('The tracking volume (VR) or bounds (space mouse) in the device frame; not stated '
                                 'until you enter a measured or published figure')
        self.vol_c = [ScrubSpinBox('', 'plain', 3, 'm', step=0.01, width=96) for _ in range(3)]
        self.vol_s = [ScrubSpinBox('', 'plain', 3, 'm', step=0.01, width=96) for _ in range(3)]
        for sb in self.vol_c:
            sb.setRange(-100.0, 100.0)
        for sb in self.vol_s:
            sb.setRange(0.0, 100.0)
            sb.setSpecialValueText('—')
        self.source_edit = QLineEdit()
        self.source_edit.setMinimumWidth(420)
        self.source_edit.setPlaceholderText('Where the device figures come from: a URL and the date read, or '
                                            '"measured"')
        self.source_more = label('', muted=True, small=True, wrap=True)
        rows = (('Name', [self.name_edit]), ('Kind', [self.kind_box]), ('Follower', [self.follower_box]),
                ('Leader', [self.leader_box]), ('Mapping', [self.map_box, self.scale]),
                ('Offset', self.off + [self.yaw]), ('Volume', [self.vol_kind]), ('Centre', self.vol_c),
                ('Size', self.vol_s), ('Source', [self.source_edit]))
        self._field_rows = {}
        for i, (text, ws) in enumerate(rows):
            lab = label(text)
            grid.addWidget(lab, i, 0)
            grid.addWidget(row(*ws), i, 1)
            self._field_rows[text] = lab
        grid.addWidget(self.source_more, len(rows), 1)
        self.lib_figures = label('', muted=True, small=True, wrap=True)
        grid.addWidget(self.lib_figures, len(rows) + 1, 1)
        grid.setColumnStretch(1, 1)
        self.editor = g
        layout.addWidget(g)
        # ---- results
        layout.addWidget(label('Devices', bold=True))
        self.table = QTreeWidget()
        self.table.setRootIsDecorated(False)
        self.table.setHeaderLabels(['Device', 'Follower', 'Comfortable', 'Teleoperable', 'Dead zones', 'Status'])
        h = self.table.header()
        h.setStretchLastSection(False)
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for k in range(1, 6):
            h.setSectionResizeMode(k, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table)
        layout.addWidget(label('Dead zones (click one to pose the follower at its worst voxel)', bold=True))
        self.zone_list = QTreeWidget()
        self.zone_list.setRootIsDecorated(False)
        self.zone_list.setHeaderLabels(['Device', 'Cause', 'Volume', 'Where', 'Why'])
        h = self.zone_list.header()
        h.setStretchLastSection(True)
        for k in range(4):
            h.setSectionResizeMode(k, QHeaderView.ResizeMode.ResizeToContents)
        self.zone_list.itemClicked.connect(lambda it, _c: self.show_zone(it.data(0, Qt.ItemDataRole.UserRole)))
        layout.addWidget(self.zone_list)
        self.robots_label = label('', wrap=True)
        layout.addWidget(self.robots_label)
        self.bim_label = label('', wrap=True)
        layout.addWidget(self.bim_label)
        self.b_handover = button('Pose the worst handover', role='ghost',
                                 tip='Put both arms at the handover voxel with the smallest arm-to-arm margin')
        self.b_handover.clicked.connect(self.show_handover)
        self.b_handover.setVisible(False)
        layout.addWidget(self.b_handover)
        self.show3d = QCheckBox('Show dead zones in the 3D view')
        self.show3d.setChecked(True)
        self.show3d.toggled.connect(lambda _on: self._invalidate())
        layout.addWidget(self.show3d)
        self.legend = label('', muted=True, small=True, wrap=True)
        layout.addWidget(self.legend)
        self.notes_label = label('', muted=True, wrap=True)
        layout.addWidget(self.notes_label)
        # ---- settings
        layout.addWidget(label('Settings (your settings; defaults labelled)', bold=True))
        sg = QWidget()
        sgrid = QGridLayout(sg)
        sgrid.setContentsMargins(0, 0, 0, 0)
        self._spin = {}
        for i, (key, (text, unit, tip)) in enumerate(TS.SETTING_TEXT.items()):
            dec = 0 if key in ('search_voxels',) else (1 if unit == 'mm' else 3)
            sb = ScrubSpinBox('', 'plain', dec, unit, step=1.0 if dec <= 1 else 0.01, width=110)
            lo, hi = TS.LIMITS[key]
            sb.setRange(lo, hi)
            sb.setToolTip(f'{text}: {tip}')
            dl = label('', muted=True, small=True)
            sgrid.addWidget(label(text), i // 2, (i % 2) * 3)
            sgrid.addWidget(sb, i // 2, (i % 2) * 3 + 1)
            sgrid.addWidget(dl, i // 2, (i % 2) * 3 + 2)
            self._spin[key] = (sb, dl)
            sb.committed.connect(lambda v, k=key: self._set_setting(k, v))
        layout.addWidget(sg)
        self.stale_note = label('', muted=True, wrap=True)
        layout.addWidget(self.stale_note)
        _fit(self.table, 6)
        _fit(self.zone_list, 10)
        # ---- wiring
        self.dev_box.currentIndexChanged.connect(lambda _i: self._fill_device())
        self.b_add.clicked.connect(lambda: self.add_device(self.kind_add.currentData()))
        self.b_del.clicked.connect(self.delete_device)
        self.b_add_lib.clicked.connect(lambda: self.add_from_library(self.lib_box.currentData()))
        self.name_edit.editingFinished.connect(
            lambda: self._set(lambda d: d.update(name=self.name_edit.text().strip() or d['name']), 'renamed'))
        self.kind_box.activated.connect(lambda i: self._set_kind(self.kind_box.itemData(i)))
        self.follower_box.activated.connect(lambda i: self._set_follower(self.follower_box.itemData(i)))
        self.leader_box.activated.connect(lambda i: self._set_leader(self.leader_box.itemData(i)))
        self.map_box.activated.connect(
            lambda i: self._set(lambda d: d['mapping'].update(kind=self.map_box.itemData(i)), 'mapping'))
        self.scale.committed.connect(lambda v: self._set(lambda d: d['mapping'].update(scale=v if v else 1.0),
                                                         f'scale {v:g}', merge='scale'))
        for sb in self.off + [self.yaw]:
            sb.committed.connect(lambda _v: self._set_offset())
        self.vol_kind.activated.connect(lambda _i: self._set_volume())
        for sb in self.vol_c + self.vol_s:
            sb.committed.connect(lambda _v: self._set_volume())
        self.source_edit.editingFinished.connect(self._set_source)
        st = self.ctx.store
        st.changed.connect(self._changed)
        st.project_replaced.connect(self._replaced)
        self._refresh_all()
        self._sync_settings()

    # ---------------------------------------------------------------- project access
    @property
    def project(self):
        return self.ctx.store.project

    def devices(self) -> list:
        return [TS.clean_device(d, self.project) for d in self.project.tasks.teleop_devices if isinstance(d, dict)]

    def current(self) -> dict | None:
        uid = self.dev_box.currentData()
        return next((d for d in self.devices() if d['uid'] == uid), None)

    def _edit(self, fn, text, merge_key=None):
        """fn(list of device records) changes them in place; one undo step on the 'tasks' section."""
        def apply(p):
            ds = [TS.clean_device(d, p) for d in p.tasks.teleop_devices if isinstance(d, dict)]
            fn(ds)
            p.tasks.teleop_devices = [TS.clean_device(d, p) for d in ds]
        self.ctx.store.edit('tasks', apply, text, merge_key=merge_key)

    def _set(self, fn, what, merge=None):
        cur = self.current()
        if cur is None or self._busy:
            return
        uid = cur['uid']

        def f(ds):
            for d in ds:
                if d['uid'] == uid:
                    fn(d)
        self._edit(f, f'{cur["name"]} · {what}', merge_key=('teleop', uid, merge) if merge else None)

    def add_device(self, kind: str = 'leader_arm') -> dict | None:
        if not self.project.robots:
            self._say('Add a robot first: a teleop device drives a follower robot.')
            return None
        dev = TS.new_device(self.project, kind or 'leader_arm')

        def f(ds):
            ds.append(dev)
        self._edit(f, f'Added the teleop device {dev["name"]}')
        k = self.dev_box.findData(dev['uid'])
        if k >= 0:
            self.dev_box.setCurrentIndex(k)
        return dev

    def _fill_library(self):
        """The library teleop devices in the picker, each with its headline figures and sources in the tooltip."""
        self.lib_box.clear()
        try:
            es = TD.entries()
        except Exception:                               # noqa: BLE001 — a broken library must not take the page
            traceback.print_exc()
            es = []
        for e in es:
            kind = (e.data.get('teleop') or {}).get('device_kind', '')
            self.lib_box.addItem(f'{e.name} · {TS.KIND_TITLES.get(kind, kind)}', e.id)
            srcs = '; '.join(f'{s.get("title") or s.get("url")} (read {s.get("read")})' for s in e.sources)
            self.lib_box.setItemData(self.lib_box.count() - 1, f'{e.title}: {TD.summary(e)}.\nSources: {srcs}',
                                     Qt.ItemDataRole.ToolTipRole)
        if not es:
            self.lib_box.addItem('No teleop devices in the library', '')
        self.b_add_lib.setEnabled(bool(es))

    def add_from_library(self, library_id: str, follower_uid: str | None = None) -> dict | None:
        """Add a device made from a library teleop entry (one undo step), driving follower_uid, else the follower of
        the device being edited, else the primary robot."""
        if not self.project.robots:
            self._say('Add a robot first: a teleop device drives a follower robot.')
            return None
        if not library_id:
            return None
        cur = self.current()
        fu = follower_uid or (cur['follower_uid'] if cur is not None else None)
        try:
            dev = TD.device_from_library(self.project, library_id, fu)
        except Exception as e:                          # noqa: BLE001 — said in a sentence, logged in full
            traceback.print_exc()
            self._say(f'The library device could not be added: {e}')
            return None

        def f(ds):
            ds.append(dev)
        self._edit(f, f'Added the teleop device {dev["name"]} from the library')
        k = self.dev_box.findData(dev['uid'])
        if k >= 0:
            self.dev_box.setCurrentIndex(k)
        return dev

    def _library_text(self, d) -> str:
        """For a device made from a library entry: which entry, and the figures its sources do not state."""
        lid = d.get('library_id')
        if not lid:
            return ''
        e = TD._entry(lid)
        if e is None:
            return f'Library device "{lid}": no longer in the library (its figures above are kept as entered).'
        missing = [r['label'].lower() for r in TD.figure_rows(e) if not r['stated']]
        return (f'From the library: {e.maker}, {e.name}. Not stated by the maker: {", ".join(missing)}.' if missing
                else f'From the library: {e.maker}, {e.name}.')

    def delete_device(self):
        cur = self.current()
        if cur is None:
            return
        self._edit(lambda ds: ds.__setitem__(slice(None), [d for d in ds if d['uid'] != cur['uid']]),
                   f'Deleted the teleop device {cur["name"]}')

    def _set_kind(self, kind):
        def fn(d):
            d['kind'] = kind
            d['mapping']['kind'] = 'joint' if kind in ('leader_arm', 'gello') else 'cartesian'
            if kind in ('leader_arm', 'gello') and not d.get('leader'):
                d['leader'] = {'robot_uid': d['follower_uid']}
        self._set(fn, f'kind {TS.KIND_TITLES[kind]}')

    def _set_follower(self, uid):
        def fn(d):
            if d.get('leader') and d['leader'].get('robot_uid') == d['follower_uid']:
                d['leader'] = {'robot_uid': uid}
            d['follower_uid'] = uid
        self._set(fn, 'follower')

    def _set_leader(self, data):
        def fn(d):
            if data == SELF:
                d['leader'] = {'robot_uid': d['follower_uid']}
            elif data:
                d['leader'] = {'robot_uid': data}
            else:
                d['leader'] = None
        self._set(fn, 'leader')

    def _set_offset(self):
        from ...core.frames import Pose, quat_from_axis_angle
        pos = [sb.value() / 1000.0 for sb in self.off]
        q = quat_from_axis_angle([0.0, 0.0, 1.0], np.radians(self.yaw.value()))
        self._set(lambda d: d['mapping'].update(offset=Pose(pos, q).to_dict()), 'mapping offset', merge='offset')

    def _set_volume(self):
        cur = self.current()
        if cur is None or self._busy:
            return
        kind = self.vol_kind.currentData()
        if not kind:
            self._pending_vol.pop(cur['uid'], None)
            self._set(lambda d: d.update(volume=None), 'volume not stated')
            return
        n = 3 if kind == 'box' else 2
        size = [sb.value() for sb in self.vol_s[:n]]
        if not all(s > 0 for s in size):
            self._pending_vol[cur['uid']] = kind
            self._fill_device()
            self._say('Enter every size of the volume (measured, or a published figure with its source) to state it.')
            return
        self._pending_vol.pop(cur['uid'], None)
        vol = {'kind': kind, 'center': [sb.value() for sb in self.vol_c], 'size': size}
        self._set(lambda d: d.update(volume=vol), 'volume', merge='volume')

    def _set_source(self):
        text = self.source_edit.text().strip()
        cur = self.current()
        if cur is None or text == cur['source'].get('device', ''):
            return

        def fn(d):
            src = dict(d.get('source') or {})
            if text:
                src['device'] = text
            else:
                src.pop('device', None)
            d['source'] = src
        self._set(fn, 'source')

    # ---------------------------------------------------------------- refresh
    def _changed(self, sections):
        if not sections or {'tasks', 'robots'} & set(sections):
            self._refresh_all()
        if 'analysis' in (sections or ()):
            self._sync_settings()
        self._check_stale()
        self._invalidate()

    def _replaced(self):
        self.doc = None
        self._pending_vol = {}
        self.table.clear()
        self.zone_list.clear()
        _fit(self.table, 6)
        _fit(self.zone_list, 10)
        for w in (self.robots_label, self.bim_label, self.notes_label, self.stale_note, self.legend):
            w.setText('')
        self.b_handover.setVisible(False)
        self._preview = []
        self._refresh_all()
        self._sync_settings()
        self.empty()
        self.suggestion_ready(False)
        self._invalidate()

    def _refresh_all(self):
        self._busy = True
        try:
            cur = self.dev_box.currentData()
            self.dev_box.blockSignals(True)
            self.dev_box.clear()
            for d in self.devices():
                self.dev_box.addItem(f'{d["name"]} · {TS.KIND_TITLES[d["kind"]]}', d['uid'])
            k = self.dev_box.findData(cur)
            self.dev_box.setCurrentIndex(k if k >= 0 else (0 if self.dev_box.count() else -1))
            self.dev_box.blockSignals(False)
            for box in (self.follower_box,):
                box.clear()
                for r in self.project.robots:
                    box.addItem(r.name, r.uid)
        finally:
            self._busy = False
        self._fill_device()

    def _fill_device(self):
        from ...core.frames import Pose, rpy_from_quat
        d = self.current()
        self.b_del.setEnabled(d is not None)
        self.editor.setVisible(d is not None)
        if d is None:
            self.dev_hint.setText('No teleop device yet: Add one, or add one from the library (its stated figures come '
                                  'with their sources). A leader arm or GELLO leads with the follower\'s own '
                                  'model (the SO-100 kit\'s leader is an SO-100); a VR or space-mouse device needs '
                                  'its tracking volume or bounds, measured or published, entered with its source.'
                                  if self.project.robots else 'Add a robot first: a teleop device drives a follower.')
            return
        self.dev_hint.setText('')
        self._busy = True
        try:
            self.name_edit.setText(d['name'])
            self.kind_box.setCurrentIndex(max(self.kind_box.findData(d['kind']), 0))
            self.follower_box.setCurrentIndex(self.follower_box.findData(d['follower_uid']))
            self.leader_box.clear()
            self.leader_box.addItem('The follower\'s own model', SELF)
            for r in self.project.robots:
                if r.uid != d['follower_uid']:
                    self.leader_box.addItem(r.name, r.uid)
            self.leader_box.addItem('None', '')
            lead = (d.get('leader') or {}).get('robot_uid', '')
            k = self.leader_box.findData(SELF if lead == d['follower_uid'] else lead)
            self.leader_box.setCurrentIndex(k if k >= 0 else self.leader_box.count() - 1)
            self.map_box.setCurrentIndex(max(self.map_box.findData(d['mapping']['kind']), 0))
            self.scale.set_value(d['mapping']['scale'])
            off = Pose.from_dict(d['mapping']['offset'])
            for sb, v in zip(self.off, off.position):
                sb.set_value(float(v) * 1000.0)
            self.yaw.set_value(float(np.degrees(rpy_from_quat(off.quaternion)[2])))
            vol = d.get('volume')
            vk = vol['kind'] if vol else self._pending_vol.get(d['uid'], '')
            self.vol_kind.setCurrentIndex(max(self.vol_kind.findData(vk), 0))
            for i, sb in enumerate(self.vol_c):
                sb.set_value(float(vol['center'][i]) if vol else sb.value())
            for i, sb in enumerate(self.vol_s):
                sb.set_value(float(vol['size'][i]) if vol and i < len(vol['size']) else (sb.value() if not vol
                                                                                         else 0.0))
            self.vol_s[2].setVisible(vk != 'cylinder')
            self.vol_s[0].setToolTip('Radius' if vk == 'cylinder' else 'Edge along x')
            self.vol_s[1].setToolTip('Height (along z)' if vk == 'cylinder' else 'Edge along y')
            self.vol_s[2].setToolTip('Edge along z')
            arm = d['kind'] in ('leader_arm', 'gello')
            cart = d['mapping']['kind'] == 'cartesian'
            for text, ws in (('Leader', [self.leader_box]), ('Volume', [self.vol_kind]), ('Centre', self.vol_c),
                             ('Size', self.vol_s), ('Offset', self.off + [self.yaw])):
                show = arm if text == 'Leader' else (cart if text == 'Offset' else
                                                     (not arm and (text == 'Volume' or bool(vk))))
                self._field_rows[text].setVisible(show)
                for w in ws:
                    w.parentWidget().setVisible(show)
            self.source_edit.setText(d['source'].get('device', ''))
            more = [f'{k}: {v}' for k, v in d['source'].items() if k != 'device']
            if not arm and not vol:
                more.append('volume: not stated (the study does not guess one)')
            self.source_more.setText('\n'.join(more))
            self.lib_figures.setText(self._library_text(d))
            self.lib_figures.setVisible(bool(d.get('library_id')))
        finally:
            self._busy = False

    def _say(self, text):
        if self.ctx.window is not None:
            self.ctx.window.msg(text, 6000)
        self.notes_label.setText(text)

    # ---------------------------------------------------------------- settings
    def settings(self) -> dict:
        try:
            return TS.settings_of(self.project)
        except Exception:                               # noqa: BLE001 — a bad stored setting: show the defaults
            return dict(TS.DEFAULTS, defaults=list(TS.DEFAULTS))

    def _sync_settings(self):
        st = self.settings()
        for key, (sb, dl) in self._spin.items():
            sb.set_value(float(st[key]))
            dl.setText('default' if key in st['defaults'] else '')

    def _set_setting(self, key, value):
        if abs(float(self.settings()[key]) - float(value)) < 1e-12:
            return

        def fn(p):
            d = dict(p.analysis.studies.get(TS.STUDY_ID) or {})
            d[key] = float(value)
            p.analysis.studies[TS.STUDY_ID] = d
        text = TS.SETTING_TEXT[key][0]
        self.ctx.store.edit('analysis', fn, f'Teleop · {text.lower()} {value:g}', merge_key=('teleop', key))

    # ---------------------------------------------------------------- result
    def inputs_key(self) -> str:
        import json
        p = self.project
        return json.dumps([p.tasks.teleop_devices, [r.to_dict() for r in p.robots], p.workspace.to_dict(),
                           p.analysis.studies.get(TS.STUDY_ID) or {}], sort_keys=True, default=str)

    def _check_stale(self):
        if self.doc is not None and getattr(self, '_inputs', None) is not None and self.inputs_key() != self._inputs:
            self.stale_note.setText('The devices, robots or task volumes changed since this run: Run again to update.')
        else:
            self.stale_note.setText('')

    def show_result(self, doc):
        self._preview = []
        s = doc.get('summary') or {}
        sev = s.get('status', 'warning')
        pct = s.get('teleoperable_pct')
        tip = ('Teleoperable: inside a device\'s comfortable workspace mapped into the follower\'s frame, and the '
               'follower reaches it without a singularity, a joint limit or a self-collision (thresholds are labelled '
               'defaults). The share is of the task voxels, through at least one device.')
        if pct is None:
            self.head.set('—', '', 'warning', s.get('status_word', 'not stated'), s.get('advice', ''), tip)
        else:
            self.head.set(f'{pct:.1f}', '%', sev, s.get('status_word', TS.WORDS[sev]), s.get('advice', ''), tip)
        self.table.clear()
        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        for r in doc.get('devices') or []:
            dz = '—' if r['dead_zone_pct'] is None else f'{r["dead_zone_pct"]:.1f} % · {r["zones_total"]}'
            it = QTreeWidgetItem([f'{r["name"]} · {r["kind_title"]}', r['follower'], _pct(r['comfortable_pct']),
                                  _pct(r['teleoperable_pct']), dz,
                                  TS.WORDS[r['status']] if r['stated'] else 'not stated'])
            for k in (2, 3, 4):
                it.setFont(k, mono(11))
                it.setTextAlignment(k, right)
            it.setForeground(5, QColor(STATUS[r['status']]))
            it.setToolTip(0, r['how'] or '')
            it.setToolTip(4, 'Share of the task volume in a dead zone, and the number of separate zones')
            self.table.addTopLevelItem(it)
        self.zone_list.clear()
        for di, r in enumerate(doc.get('devices') or []):
            for zi, z in enumerate(r['zones']):
                it = QTreeWidgetItem([r['name'], z['cause'], f'{z["volume_m3"] * 1e6:.0f} cm³',
                                      ', '.join(f'{v:.3f}' for v in z['center']) + ' m', z['text']])
                it.setData(0, Qt.ItemDataRole.UserRole, (di, zi))
                it.setForeground(1, QColor(cause_hex(z['cause'])))
                it.setFont(2, mono(11))
                it.setTextAlignment(2, right)
                it.setToolTip(3, f'Box {z["lo"]} to {z["hi"]} m, {z["voxels"]} voxels, in {z["volume"]}')
                self.zone_list.addTopLevelItem(it)
        if not self.zone_list.topLevelItemCount():       # say it, rather than leave an empty table
            none = ('no dead zone: every comfortable voxel is reached with good dexterity' if doc.get('devices')
                    else 'no teleop device: add one above')
            it = QTreeWidgetItem(['—', none, '', '', ''])
            it.setForeground(1, THEME.color('muted'))
            self.zone_list.addTopLevelItem(it)
        lines = []
        for r in doc.get('robots') or []:
            lines.append(f'{r["name"]}: reaches {r["reachable_pct"]:.1f} % of the task volume; 1/κ median '
                         f'{_fmt(r["inverse_condition_median"])} (lowest {_fmt(r["inverse_condition_min"])}); near a '
                         f'singularity {r["near_singularity_pct"]:.1f} %, near a limit {r["near_limit_pct"]:.1f} %, '
                         f'close to itself {r["self_close_pct"]:.1f} %.')
        self.robots_label.setText('\n'.join(lines))
        bim = doc.get('bimanual')
        self.b_handover.setVisible(False)
        if bim:
            bl = []
            for pr in bim['pairs']:
                ho = pr['handover']
                t = (f'{pr["a_name"]} and {pr["b_name"]}: both reach {pr["overlap_pct"]:.1f} % of the task volume, '
                     f'both teleoperable {pr["both_teleoperable_pct"]:.1f} %.')
                if ho.get('quality_pct') is not None:
                    t += (f' Handover ({ho["where"]}, {ho["voxels"]} voxels, {ho["gap_m"] * 1000:.0f} mm apart): '
                          f'quality {ho["quality_pct"]:.1f} %, arm-to-arm margin '
                          f'{_mm(ho["arm_margin_min_m"])} smallest, {_mm(ho["arm_margin_median_m"])} median.')
                    self.b_handover.setVisible(bool(ho.get('worst')))
                bl.append(t)
            self.bim_label.setText('\n'.join(bl) + '\n' + bim['note'])
        else:
            self.bim_label.setText('')
        self.notes_label.setText('\n'.join(doc.get('notes') or []))
        _fit(self.table, 6)
        _fit(self.zone_list, 10)
        sug = doc.get('suggestion')
        self.suggestion_ready(bool(sug) and not (sug or {}).get('blocked'))
        if sug and sug.get('blocked'):
            self.notes_label.setText((self.notes_label.text() + '\n' if self.notes_label.text() else '')
                                     + sug['blocked'])
        self.legend.setText('Dead-zone voxels: ' + ', '.join(
            f'<span style="color:{cause_hex(c)}">■</span> {c}' for c in TS.CAUSES) + '.')
        self.legend.setTextFormat(Qt.TextFormat.RichText)
        self._inputs = self.inputs_key()
        self._check_stale()
        self._invalidate()

    def show_zone(self, key):
        """Pose the follower at the zone's worst voxel (view state, not an edit)."""
        if self.doc is None or key is None:
            return
        di, zi = key
        try:
            r = self.doc['devices'][di]
            z = r['zones'][zi]
        except (IndexError, KeyError, TypeError):
            return
        if not z.get('q'):
            self._say('The follower has no pose here (it cannot reach this zone).')
            return
        try:
            self.ctx.store.set_q(r['follower_uid'], np.asarray(z['q'], float))
        except Exception:                               # noqa: BLE001 — the robot may have gone
            traceback.print_exc()

    def show_handover(self):
        bim = (self.doc or {}).get('bimanual') or {}
        for pr in bim.get('pairs') or []:
            w = pr['handover'].get('worst')
            if w:
                try:
                    self.ctx.store.set_q(pr['a'], np.asarray(w['q_a'], float))
                    self.ctx.store.set_q(pr['b'], np.asarray(w['q_b'], float))
                except Exception:                       # noqa: BLE001
                    traceback.print_exc()
                return

    # ---------------------------------------------------------------- Apply, Preview, Explain
    def apply(self):
        sug = (self.doc or {}).get('suggestion')
        if not sug or sug.get('blocked'):
            return None
        keys = TS.suggestion_sections(self.project, sug)
        self.ctx.store.edit(keys, lambda p: TS.apply_suggestion(p, sug), f'Teleop · {sug["text"].split(":")[0]}')
        self._preview = []
        self.suggestion_ready(False)
        self._say(f'Applied: {sug["text"].split(":")[0]}. Run again to check the prediction.')
        return keys

    def preview(self):
        from ..viewport3d import LineDraw
        sug = (self.doc or {}).get('suggestion')
        if not sug:
            return
        if self._preview:
            self._preview = []
            self._invalidate()
            return
        r = self.project.robot(sug['robot_uid'])
        if r is None:
            return
        a = np.asarray(r.base.position, float)
        b = a + np.asarray(sug['offset_m'], float)
        col = QColor(STATUS['good'])
        self._preview = [LineDraw(np.array([[a, b]], np.float32), col, 2.5, dash=True, depth=False, segments=True)]
        self._invalidate()
        self._say(f'Preview: {sug["text"]} Apply makes the move.')

    def explain_text(self) -> str:
        st = self.settings()
        return (
            'Each device\'s comfortable workspace is mapped into the follower\'s frame. A leader arm or GELLO moves '
            'the follower joint to joint (follower joint = scale × leader joint), comfortable inside '
            f'{st["comfort_fraction"]:g} of each leader joint\'s range (a default): a voxel is comfortable when the '
            'follower\'s IK reaches its centre with the joints inside that box. A VR tracking volume or space-mouse '
            'bounds are scaled and offset into the follower\'s base frame; a device whose volume is not stated has '
            'no workspace (nothing is guessed).\n\n'
            'Over the task voxels the follower solves IK for its tool point (position, orientation free). At the '
            'solution: Yoshikawa\'s manipulability (the product of the Jacobian\'s singular values, √det(J Jᵀ)), the '
            'condition number κ, the distance to the nearer joint stop (as a share of the half-range) and the '
            'self-collision distance between links the arm\'s joints move against each other.\n\n'
            'A dead zone is a voxel the leader reaches comfortably where the follower is unreachable, closer to itself '
            f'than {st["self_margin_mm"]:g} mm, singular (1/κ below {st["min_inverse_condition"]:g}) or near a limit '
            f'(below {st["min_limit_distance"]:g}); these thresholds are defaults you can change. Connected voxels '
            'of one cause form a zone; a click on it poses the follower at its worst voxel.\n\n'
            'The suggestion moves the follower\'s base (or a cartesian device\'s offset) by whole voxels and scores '
            'each move from the grid already computed. Known limits: position-only reach; a local IK; the comfortable '
            'fraction is not a human-factors model; the prediction holds to the voxel size.')

    def explain(self):
        box = QMessageBox(self)
        box.setWindowTitle('How teleop overlap is computed')
        box.setText(self.explain_text())
        box.setModal(False)
        box.show()

    # ---------------------------------------------------------------- overlays
    def _invalidate(self):
        self._overlay = None
        win = self.ctx.window
        if win is not None and hasattr(win, 'viewport'):
            win.viewport.update()
        cv = getattr(win, 'camview', None) if win is not None else None
        if cv is not None and hasattr(cv, 'view'):
            try:
                cv.view.refresh_overlays()
            except Exception:                           # noqa: BLE001
                traceback.print_exc()

    def _build_overlay(self) -> list:
        from ..viewport3d import LineDraw, TriDraw
        doc = self.doc or {}
        gridd = doc.get('grid') or {}
        C = np.asarray(gridd.get('centers') or [], float).reshape(-1, 3)
        h = float(gridd.get('voxel_m') or 0.0)
        out = []
        if not len(C) or h <= 0:
            return out
        state = np.full(len(C), -1)
        for r in doc.get('devices') or []:          # the first device that flags a voxel colours it
            s = np.asarray(r['state'], int)
            if len(s) == len(C):
                take = (state < 0) & (s >= 2) & (s < 2 + len(TS.CAUSES))
                state[take] = s[take]
        for k, cause in enumerate(TS.CAUSES):
            sel = state == 2 + k
            if not sel.any():
                continue
            tri = cubes(C[sel], h * CUBE_SHRINK / 2)
            c = QColor(cause_hex(cause))
            col = np.tile(np.array([c.redF(), c.greenF(), c.blueF(), 0.55], np.float32), (len(tri), 1))
            out.append(TriDraw(tri, col, depth=True))
        p = self.project
        for r in doc.get('devices') or []:
            if r.get('stated') and r.get('volume') and r.get('mapping', {}).get('kind') == 'cartesian':
                robot = p.robot(r['follower_uid'])
                if robot is None:
                    continue
                dev = TS.clean_device({k: r[k] for k in ('uid', 'name', 'kind', 'follower_uid', 'leader', 'mapping',
                                                         'volume', 'source')}, p)
                for poly in TS.volume_outline(dev['volume']):
                    W = TS.from_device(dev, robot.base.matrix(), poly)
                    out.append(LineDraw(W.astype(np.float32), QColor(STATUS['good']), 1.5, dash=True))
        return out

    def overlays(self, vp) -> list:
        if not self.show3d.isChecked():
            return list(self._preview)
        if self._overlay is None:
            try:
                self._overlay = self._build_overlay()
            except Exception:                           # noqa: BLE001 — an overlay must not take the page down
                traceback.print_exc()
                self._overlay = []
        return self._overlay + list(self._preview)

    def camview_overlays(self, cam_model, pose) -> list:
        """The dead zones' boxes seen through the Camera view, one entry per cause."""
        from ...core import camrender as CR
        if not self.show3d.isChecked() or self.doc is None:
            return []
        by = {}
        for r in self.doc.get('devices') or []:
            for z in r['zones'][:20]:
                for poly in CR.box_polylines(np.asarray(z['lo'], float), np.asarray(z['hi'], float)):
                    by.setdefault(z['cause'], []).extend(CR.visible_polylines(None, cam_model, pose, poly,
                                                                              step=0.005))
        return [(polys, cause_hex(c), f'dead zone · {c}') for c, polys in by.items() if polys]


def _fit(tree: QTreeWidget, rows: int):
    """Height for the rows shown (at most `rows`), so short tables leave no empty band."""
    n = max(1, min(tree.topLevelItemCount(), rows))
    rh = tree.sizeHintForRow(0) if tree.topLevelItemCount() else 22
    tree.setFixedHeight(tree.header().sizeHint().height() + n * max(rh, 18) + 2 * tree.frameWidth() + 4)


def _fmt(v) -> str:
    return '—' if v is None else f'{v:.3f}'


def _mm(v) -> str:
    return '—' if v is None else f'{v * 1000:.0f} mm'


register('teleop', 'Teleop overlap', TeleopPage, menu='Analysis', icon='teleop', action_id='teleop_overlap',
         workspace='teleop', order=40)
