"""The dock panels: Outliner, Library, Inspector, Camera view, Analysis, Assistant, and the pose scrubber under the
3D view.

Each follows Vantage's page pattern: what it shows, the number that matters (a large mono figure with a status word)
and what to do about it. Panels whose engine arrives in a later phase say so plainly instead of showing invented
content.
"""
from __future__ import annotations

import html
import traceback

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPlainTextEdit,
                             QScrollArea, QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from . import icons
from ..core.jobs import MOVING_JOINTS
from .theme import THEME, STATUS, mono, sans
from .scrubber import PoseScrubber  # noqa: F401  (the joint-state timeline, re-exported for the window)
from .widgets import ScrubSpinBox, Section, StatusLine, button, hline, label

ROLE = Qt.ItemDataRole.UserRole


def empty_state(title: str, text: str) -> QWidget:
    w = QWidget()
    v = QVBoxLayout(w)
    v.setContentsMargins(16, 16, 16, 16)
    v.setSpacing(6)
    t = label(title, bold=True)
    b = label(text, muted=True, wrap=True)
    v.addWidget(t)
    v.addWidget(b)
    v.addStretch(1)
    return w


class Headline(QWidget):
    """The page pattern's top: a large mono figure with its unit and a status word, what it means, and the actions."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 10, 12, 8)
        v.setSpacing(4)
        self.title = label(title, ink2=True)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.figure = QLabel('—')
        # the application style sheet sets every widget's font size, so the large figure says its own
        self.figure.setStyleSheet('font-family: "IBM Plex Mono"; font-size: 26px; font-weight: 600;')
        self.unit = label('', muted=True)
        self.unit.setFont(mono(12))
        self.status = StatusLine('good', '', wrap=False)
        row.addWidget(self.figure)
        row.addWidget(self.unit, 0, Qt.AlignmentFlag.AlignBottom)
        row.addSpacing(10)
        row.addWidget(self.status, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addStretch(1)
        self.explain = label('', wrap=True)
        self.actions = QHBoxLayout()
        self.actions.setSpacing(6)
        v.addWidget(self.title)
        v.addLayout(row)
        v.addWidget(self.explain)
        v.addLayout(self.actions)

    def set(self, figure: str, unit: str, sev: str, word: str, explain: str, tip: str = ''):
        self.figure.setText(figure)
        self.unit.setText(unit)
        self.status.set(sev, word)
        self.explain.setText(explain)
        self.figure.setToolTip(tip)


# =============================================================================== outliner
class Outliner(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText('Filter')
        self.filter.textChanged.connect(self.refill)
        f = QWidget()
        fl = QHBoxLayout(f)
        fl.setContentsMargins(6, 4, 6, 4)
        fl.addWidget(self.filter)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setColumnCount(2)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.itemSelectionChanged.connect(self._picked)
        v.addWidget(f)
        v.addWidget(hline())
        v.addWidget(self.tree, 1)
        st = ctx.store
        st.changed.connect(lambda _s: self.refill())
        st.project_replaced.connect(self.refill)
        st.selection_changed.connect(self._sync_sel)
        THEME.changed.connect(self._theme_changed)      # icons are tinted per theme; bound: dropped with the panel
        self.refill()

    def _theme_changed(self, *_):
        self.refill()

    def _group(self, text, icon, count):
        it = QTreeWidgetItem([text, str(count)])
        it.setIcon(0, icons.icon(icon, 12))
        it.setFont(1, mono(11))
        it.setForeground(1, THEME.color('muted'))
        it.setFlags(Qt.ItemFlag.ItemIsEnabled)
        self.tree.addTopLevelItem(it)
        return it

    def _item(self, parent, text, detail, sel, icon=None):
        q = self.filter.text().strip().lower()
        if q and q not in text.lower():
            return None
        it = QTreeWidgetItem([text, detail])
        it.setData(0, ROLE, sel)
        it.setFont(1, mono(11))
        it.setForeground(1, THEME.color('muted'))
        if icon:
            it.setIcon(0, icons.icon(icon, 12))
        parent.addChild(it)
        return it

    def refill(self):
        p = self.ctx.store.project
        self.tree.blockSignals(True)
        self.tree.clear()
        top = self._group(p.name, 'file', '')
        top.setData(0, ROLE, ('project',))
        top.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        g = self._group('Robots', 'robot-arm', len(p.robots))
        for r in p.robots:
            nj = sum(j.get('type') in MOVING_JOINTS for j in r.joints)
            self._item(g, r.name + (' · primary' if r.primary else ''), f'{nj} joints', ('robot', r.uid), 'robot-arm')
        g = self._group(f'Sensors · {p.variant.name}', 'camera-wide', len(p.variant.sensors))
        for s in p.variant.sensors:
            where = s.mount.parent_frame
            self._item(g, s.name, f'{s.kind} · {where}', ('sensor', s.uid),
                       {'camera': 'camera-wide', 'depth': 'depth', 'tactile': 'tactile'}.get(s.kind, 'sensor'))
        g = self._group('Workspace', 'task-volume', len(p.workspace.volumes) + len(p.workspace.objects))
        for v in p.workspace.volumes:
            self._item(g, v.name, f'{v.size[0] * 1000:.0f} × {v.size[1] * 1000:.0f} × {v.size[2] * 1000:.0f} mm',
                       ('volume', v.uid), 'task-volume')
        for o in p.workspace.objects:
            self._item(g, o.name, ', '.join(o.materials), ('object', o.uid), 'cube')
        t = p.tasks
        g = self._group('Tasks', 'grasp', len(t.grasp_sets) + len(t.trajectories) + len(t.teleop_devices))
        for tr in t.trajectories:
            ts = tr.get('t') or [0.0]
            it = self._item(g, str(tr.get('name') or 'Trajectory'), f'{len(ts)} · {ts[-1] - ts[0]:.2f} s',
                            ('trajectory', tr.get('uid')), 'timeline')
            if it is not None:
                r = p.robot(tr.get('robot_uid'))
                it.setToolTip(0, f'Joint trajectory for {r.name if r else "a robot no longer in the project"}: '
                                 f'{len(ts)} samples over {ts[-1] - ts[0]:.2f} s. Select it to play it in the pose '
                                 'scrubber; Delete removes it.')
        self.tree.expandAll()
        self.tree.blockSignals(False)
        self._sync_sel()

    def _picked(self):
        its = self.tree.selectedItems()
        sel = its[0].data(0, ROLE) if its else None
        if sel and tuple(sel) != self.ctx.store.sel:
            self.ctx.store.select(tuple(sel))

    def _sync_sel(self):
        want = self.ctx.store.sel
        self.tree.blockSignals(True)
        self.tree.clearSelection()
        stack = [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
        while stack:
            it = stack.pop()
            d = it.data(0, ROLE)
            if d and tuple(d) == want:
                it.setSelected(True)
                self.tree.scrollToItem(it)
            stack += [it.child(i) for i in range(it.childCount())]
        self.tree.blockSignals(False)


# =============================================================================== library
# (title, icon, phase it arrives in or None when this build has it, what it holds, [(entry, action id)])
LIBRARY_ENTRIES = {
    'Robots': [('Import a URDF or MJCF file…', 'import_robot'),
               ('Import from MuJoCo Menagerie…', 'import_menagerie')],
    'Grippers': [('Import an arm whose model file includes its gripper…', 'import_robot')],
}
# groups filled from the hardware library (core/library) name its kind as the fifth item
LIBRARY_GROUPS = (
    ('Robots', 'robot-arm', None, 'Robots are read from their official model files (URDF or MJCF). Joint limits '
                                 'come from the file; nothing is retyped.'),
    ('Cameras', 'camera-wide', None, 'Global-shutter USB and GMSL cameras, each figure from the maker\'s datasheet '
                                     'with its source and the date it was read.', 'camera'),
    ('Depth sensors', 'depth', None, 'RealSense, ZED, OAK-D and Orbbec units: baseline, depth FOV, min-Z per '
                                     'resolution and stream modes, from datasheets only.', 'depth'),
    ('Tactile sensors', 'tactile', None, 'GelSight Mini and DIGIT: pad size, internal camera and interface. Contact '
                                         'sensors: they go on a fingertip, not on a tripod.', 'tactile'),
    ('Hosts', 'sensor', None, 'USB host controllers, GMSL deserialiser boards and computers, for the bandwidth and '
                              'power budget. They are not placed in the scene.', 'host'),
    ('Grippers', 'gripper', None, 'From their official model files, as part of the arm\'s file (an MJCF attach or '
                                  'a URDF fixed joint); the UR5e sample has the Robotiq 2F-85 on its flange.'),
    ('Mounts', 'bracket', 6, 'ISO 9409-1 flange adapters and 2020/4040 T-slot clamps, generated as printable STL.'),
)
DEVICE = Qt.ItemDataRole.UserRole + 1


class _LibraryTree(QTreeWidget):
    """The library tree; a device drags out as its id (rigedit.MIME), for the 3D view to drop."""

    def mimeData(self, items):
        from PyQt6.QtCore import QMimeData
        from .rigedit import MIME
        md = QMimeData()
        ids = [it.data(0, DEVICE) for it in items if it.data(0, DEVICE)]
        if ids:
            md.setData(MIME, ids[0].encode('utf-8'))
            md.setText(ids[0])
        return md

    def mimeTypes(self):
        from .rigedit import MIME
        return [MIME, 'text/plain']


class LibraryPanel(QWidget):
    """The Library: robots (import), the hardware library's devices by kind (core/library: makers' datasheets, plus
    your custom devices), grippers and mounts. Double-click a camera, depth or tactile device, or drag it into the 3D
    view, to add a sensor of that model to the rig; hover it for every figure and its source."""

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.search = QLineEdit()
        self.search.setPlaceholderText('Search the library')
        self.search.textChanged.connect(lambda _t: self.refill())
        s = QWidget()
        sl = QHBoxLayout(s)
        sl.setContentsMargins(6, 4, 6, 4)
        sl.addWidget(self.search)
        self.tree = _LibraryTree()
        self.tree.setTextElideMode(Qt.TextElideMode.ElideMiddle)     # keep the model number: 'Intel Real…D405'
        self.tree.setHeaderHidden(True)
        self.tree.setColumnCount(2)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setDragEnabled(True)
        self.tree.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._menu)
        v.addWidget(s)
        v.addWidget(hline())
        v.addWidget(self.tree, 1)
        self.tree.itemActivated.connect(self._activated)
        THEME.changed.connect(self._theme_changed)      # bound: dropped with the panel
        self.refill()

    def _theme_changed(self, *_):
        self.refill()

    def _entries(self, kind):
        from ..core import library
        try:
            return library.entries(kind)
        except Exception:                                # noqa: BLE001 — a broken library must not take the panel
            traceback.print_exc()
            return []

    def refill(self):
        from .rigedit import device_title, entry_summary, entry_tooltip
        q = self.search.text().strip().lower()
        open_ = {self.tree.topLevelItem(i).text(0) for i in range(self.tree.topLevelItemCount())
                 if self.tree.topLevelItem(i).isExpanded()}
        self.tree.clear()
        for title, icon, phase, text, *kind in LIBRARY_GROUPS:
            it = QTreeWidgetItem([title, '' if phase is None else '0'])
            it.setIcon(0, icons.icon(icon, 12))
            it.setFont(1, mono(11))
            it.setForeground(1, THEME.color('muted'))
            it.setToolTip(0, text if phase is None else f'{text}\nArrives in Phase {phase}.')
            it.setFlags(Qt.ItemFlag.ItemIsEnabled)
            if kind:
                es = [e for e in self._entries(kind[0])
                      if not q or q in f'{e.maker} {e.name} {e.data.get("family") or ""} {e.id}'.lower()]
                it.setText(1, str(len(es)))
                for e in es:
                    c = QTreeWidgetItem([device_title(e) + (' (custom)' if e.is_custom else ''), entry_summary(e)])
                    c.setData(0, DEVICE, e.id)
                    c.setFont(1, mono(10))
                    c.setForeground(1, THEME.color('muted'))
                    tip = entry_tooltip(e) + ('\nDouble-click or drag into the 3D view to add it to the rig.'
                                              if e.kind != 'host' else '')
                    c.setToolTip(0, tip)
                    c.setToolTip(1, tip)
                    flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                    if e.kind != 'host':
                        flags |= Qt.ItemFlag.ItemIsDragEnabled
                    c.setFlags(flags)
                    it.addChild(c)
                if q and not es:
                    continue
            elif phase is None:
                for entry, aid in LIBRARY_ENTRIES.get(title, []):
                    if q and q not in entry.lower() and q not in title.lower():
                        continue
                    e = QTreeWidgetItem([entry, ''])
                    e.setData(0, ROLE, aid)
                    e.setToolTip(0, f'{text}\nDouble-click to open.')
                    it.addChild(e)
            else:
                note = QTreeWidgetItem([f'Arrives in Phase {phase}', ''])
                note.setForeground(0, THEME.color('muted'))
                note.setToolTip(0, text)
                note.setFlags(Qt.ItemFlag.ItemIsEnabled)
                it.addChild(note)
            if q and it.childCount() == 0 and q not in title.lower():
                continue
            self.tree.addTopLevelItem(it)
            it.setExpanded(bool(q) or title in open_)
        new = QTreeWidgetItem(['New custom device…', ''])
        new.setIcon(0, icons.icon('plus', 12))
        new.setData(0, ROLE, 'new_device')
        new.setToolTip(0, 'A device of your own: every figure marked as given by the maker or not given.')
        self.tree.addTopLevelItem(new)

    def item_for(self, model_id):
        """The tree item of a device (tests, the palette)."""
        for i in range(self.tree.topLevelItemCount()):
            g = self.tree.topLevelItem(i)
            for k in range(g.childCount()):
                if g.child(k).data(0, DEVICE) == model_id:
                    return g.child(k)
        return None

    def _activated(self, item, _col=0):
        mid = item.data(0, DEVICE)
        if mid:
            self.add_device(mid)
            return
        win = self.ctx.window
        a = getattr(win, 'actions_by_id', {}).get(item.data(0, ROLE) or '') if win is not None else None
        if a is not None and a.isEnabled():
            a.trigger()

    def add_device(self, model_id, at=None):
        """Add a sensor of library device model_id to the rig (one undo step); a host says why it is not placed."""
        from ..core import library
        from .rigedit import add_sensor
        e = library.get(model_id)
        win = self.ctx.window
        if e is None:
            return None
        if e.kind == 'host':
            if win is not None:
                win.msg(f'{e.name} is a host, not a sensor: it is not placed in the scene.', 5000)
            return None
        s = add_sensor(self.ctx.store, e.id, at)
        if win is not None:
            win.msg(f'Added {s.name} to {self.ctx.store.project.variant.name}', 4000)
        return s

    def _menu(self, pos):
        from PyQt6.QtWidgets import QMenu
        it = self.tree.itemAt(pos)
        mid = it.data(0, DEVICE) if it is not None else None
        m = QMenu(self)
        if mid:
            from ..core import library
            e = library.get(mid)
            if e is not None and e.kind != 'host':
                m.addAction('Add to the rig', lambda: self.add_device(mid))
            if e is not None:
                m.addAction('Start a custom device from this one…', lambda: self._custom(mid))
            if e is not None and e.is_custom:
                m.addAction('Remove this custom device', lambda: self._remove(mid))
        m.addAction('New custom device…', lambda: self._custom(None))
        m.exec(self.tree.viewport().mapToGlobal(pos))

    def _custom(self, mid):
        from .dialogs.custom_device import CustomDeviceDialog
        CustomDeviceDialog.run(self.ctx, self.window(), start_from=mid)

    def _remove(self, mid):
        from ..core import library
        try:
            library.remove_custom(mid)
        except library.LibraryError as e:
            if self.ctx.window is not None:
                self.ctx.window.msg(str(e), 6000)
        self.refill()


# =============================================================================== inspector
class Inspector(QWidget):
    titleChanged = pyqtSignal(str)

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(0)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.lay.addWidget(self.scroll)
        st = ctx.store
        st.selection_changed.connect(self.rebuild)
        st.changed.connect(lambda _s: self.rebuild() if not st.dragging else None)
        st.project_replaced.connect(self.rebuild)
        THEME.changed.connect(self._theme_changed)         # a bound slot: dropped with the panel
        self.rebuild()

    def _theme_changed(self, *_):
        self.rebuild()                                     # the source links carry the theme's ink

    def rebuild(self):
        st = self.ctx.store
        obj = st.selected()
        body = QWidget()
        v = QVBoxLayout(body)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        kind = st.sel[0] if st.sel else ''
        if kind == 'volume' and obj is not None:
            self.titleChanged.emit('Task volume')
            v.addWidget(self._volume(obj))
        elif kind == 'sensor' and obj is not None:
            self.titleChanged.emit('Sensor')
            v.addWidget(self._sensor(obj))
        elif kind == 'robot' and obj is not None:
            self.titleChanged.emit('Robot')
            v.addWidget(self._robot(obj))
        elif kind == 'trajectory' and obj is not None:
            self.titleChanged.emit('Trajectory')
            v.addWidget(self._trajectory(obj))
        elif kind == 'object' and obj is not None:
            self.titleChanged.emit('Task object')
            sec = Section(obj.name)
            sec.row('Shape', obj.shape)
            sec.row('Size', ' × '.join(f'{s * 1000:.0f}' for s in obj.size) + ' mm')
            sec.row('Materials', ', '.join(obj.materials))
            v.addWidget(sec)
        elif kind == 'project' or not st.sel:
            self.titleChanged.emit('Project')
            v.addWidget(self._project())
        else:
            self.titleChanged.emit('')
            v.addWidget(empty_state('Nothing selected', 'Pick an item in the Outliner.'))
        v.addStretch(1)
        self.scroll.setWidget(body)

    def _project(self):
        p = self.ctx.store.project
        sec = Section('Project')
        name = QLineEdit(p.name)

        def rename():
            t = name.text().strip()
            if t and t != p.name:
                self.ctx.store.edit('meta', lambda q: setattr(q, 'name', t), f'Renamed the project to {t}')
        name.editingFinished.connect(rename)
        sec.row('Name', name)
        sec.row('File', p.path.rsplit('/', 1)[-1] if p.path else 'not saved yet')
        sec.row('Robots', str(len(p.robots)))
        sec.row('Variant', p.variant.name)
        sec.row('Voxel', f'{p.analysis.voxel_mm:g} mm', tip='Coverage voxel size (default 5 mm; 1–2 mm only inside a '
                                                             'refinement region)')
        return sec

    def _robot(self, r):
        """A robot: its name (editable), role, source file and licence, links and joints, and its tags."""
        p = self.ctx.store.project
        sec = Section(r.name)
        name = QLineEdit(r.name)

        def rename():
            t = name.text().strip()
            if t and t != r.name:
                uid = r.uid
                self.ctx.store.edit('robots', lambda q: setattr(q.robot(uid), 'name', t), f'Renamed the robot to {t}')
        name.editingFinished.connect(rename)
        sec.row('Name', name)

        def kv(k, v, tip=''):
            w = label(v, muted=True, mono_=True, small=True)
            w.setToolTip(tip or v)
            sec.row(k, w)
        kv('Role', 'primary' if r.primary else 'further robot',
           'The primary robot\'s base is the world frame' if r.primary else
           'Placed by its base pose in the world frame')
        src = r.source
        kv('File', f'{(src.file or "").rsplit("/", 1)[-1] or "—"} ({(src.kind or "?").upper()})',
           f'sha256 {src.sha256}' if src.sha256 else '')
        lic = getattr(src, 'license', '') or ''
        kv('Licence', lic or 'not recorded', lic or 'Imported from a file on disk: its licence is the one that came '
                                                    'with that file')
        nj = sum(j.get('type') in MOVING_JOINTS for j in r.joints)
        kv('Links', str(len(r.links)))
        kv('Moving joints', str(nj))
        grip = [j['name'] for j in r.joints if j.get('gripper')]
        kv('Gripper joints', ', '.join(grip) or 'none tagged')
        kv('Gripper frame', r.gripper_frame or 'none: IK drag uses the last link')
        kv('Fingertips', ', '.join(r.fingertip_frames) or 'none tagged')
        kv('Contact pads', str(len(r.contact_pads)))
        trajs = [t for t in p.tasks.trajectories if t.get('robot_uid') == r.uid]
        kv('Trajectories', str(len(trajs)))
        win = self.ctx.window
        if win is not None and hasattr(win, 'open_page'):
            b = button('Joint sliders', icon_name='joint', tip='Move its joints and read the gripper frame\'s pose, '
                                                               'manipulability and self-collision')
            b.clicked.connect(lambda: win.open_page('robot'))
            sec.add(b)
        if src.note:
            sec.add(label(src.note, muted=True, wrap=True, small=True))
        return sec

    def _trajectory(self, tr):
        """A joint trajectory q(t): its robot, samples, duration and joints (values as the CSV gave them)."""
        p = self.ctx.store.project
        ts = tr.get('t') or [0.0]
        r = p.robot(tr.get('robot_uid'))
        sec = Section(str(tr.get('name') or 'Trajectory'))
        joints = list(tr.get('joints') or [])
        for k, v in (('Robot', r.name if r else 'not in the project'), ('Samples', str(len(ts))),
                     ('Duration', f'{ts[-1] - ts[0]:.3f} s'), ('Joints', f'{len(joints)}: {", ".join(joints)}')):
            w = label(v, muted=True, mono_=True, small=True)
            w.setToolTip(v)
            sec.row(k, w)
        sec.add(label('Play or scrub it under the 3D view. Joints the file does not name keep their current value.',
                      muted=True, wrap=True, small=True))
        return sec

    def _vec(self, sec, title, values, apply, minimum=None):
        """Three length fields in metres (3 decimals: millimetre steps), as Vantage's inspector; typing '450 mm'
        works too."""
        boxes = []
        for k, ax in enumerate('XYZ'):
            b = ScrubSpinBox(ax, 'length', 3, unit='m')
            if minimum is not None:
                b.setMinimum(minimum)
            b.set_value(float(values[k]))
            boxes.append(b)

        def commit(_=None):
            apply([b.value() for b in boxes])
        for b in boxes:
            b.committed.connect(commit)
        sec.row(title, *boxes)
        return boxes

    def _volume(self, vol):
        st = self.ctx.store
        sec = Section(vol.name)
        sec.row('Kind', vol.kind)

        def set_attr(attr, vals):
            def fn(p):
                o = next(x for x in p.workspace.volumes if x.uid == vol.uid)
                setattr(o, attr, tuple(vals))
            st.edit('workspace', fn, f'Changed {vol.name} {attr}')
        self._vec(sec, 'Centre', vol.center, lambda v: set_attr('center', v))
        self._vec(sec, 'Size', vol.size, lambda v: set_attr('size', [max(0.001, x) for x in v]), minimum=0.001)
        from ..core.jobs import voxel_count, DENSE_LIMIT
        a = st.project.analysis
        n = voxel_count(vol.size, a.voxel_mm / 1000)
        n1 = voxel_count(vol.size, 0.001)
        sec.row('Voxels', f'{n:,}'.replace(',', ' ') + f' at {a.voxel_mm:g} mm',
                tip='Dense grid voxel count: the product of ceil(edge / voxel) per axis')
        line = StatusLine('good' if n1 <= DENSE_LIMIT else 'warning',
                          (f'A dense 1 mm grid would need {n1:,} voxels'.replace(',', ' ') +
                           ('.' if n1 <= DENSE_LIMIT else ': not feasible. Use 1–2 mm only inside a refinement '
                                                          'region around the objects.')))
        sec.add(line)
        return sec

    def _sensor(self, s):
        st = self.ctx.store
        sec = Section(s.name)
        sec.row('Kind', s.kind)
        self._device(sec, s)
        sec.row('Mounted on', s.mount.parent_frame + (' (link)' if s.mount.on_link else ' (static)'))

        def set_pos(vals):
            def fn(p):
                o = p.variant.sensor(s.uid)
                o.mount.pose.position = np.asarray(vals, float)
            st.edit('variant', fn, f'Moved {s.name}')
        self._vec(sec, 'Position', s.mount.pose.position, set_pos)
        from ..core.frames import rpy_from_quat
        r, pch, y = (np.degrees(a) for a in rpy_from_quat(s.mount.pose.quaternion))
        sec.row('Roll, pitch, yaw', f'{r:.1f}°  {pch:.1f}°  {y:.1f}°',
                tip='URDF convention: about fixed x, then y, then z; stored as a quaternion (x, y, z, w)')
        sec.row('Tolerance', f'± {s.mount.tolerance.position_mm:g} mm · ± {s.mount.tolerance.rotation_deg:g}° '
                             '(default)', tip='Mount tolerance; the Monte Carlo studies carry it into every figure')
        win = self.ctx.window
        if win is not None and hasattr(win, 'open_page') and 'sensors' in getattr(win, 'pages', {}):
            b = button('Sensors and mounts', icon_name='sensor', tip='Change its device, mount, pose and stream')
            b.clicked.connect(lambda: win.open_page('sensors'))
            sec.add(b)
        return sec

    def _device(self, sec, s):
        """The sensor's library device and its figures: "—" with the tooltip "not stated by the maker" where the
        datasheet gives none, and the sources as links."""
        from ..core import library
        from .rigedit import DASH, device_title, entry_tooltip, num, res_text

        def fig(k, text, tip=''):
            w = label(text, muted=True, mono_=True, small=True)
            t = library.NOT_STATED if text.startswith(DASH) else (tip or text)
            if DASH in text and not text.startswith(DASH):          # '87° × 58° × —': say what the dash means
                t += f'\n"{DASH}": {library.NOT_STATED}'
            w.setToolTip(t)
            sec.row(k, w)
        e = library.get(s.model_id) if s.model_id else None
        if e is None:
            fig('Model', 'no device chosen' if not s.model_id else f'{s.model_id} (not in the library)',
                library.DEFAULT_LABEL)
            return
        fig('Model', device_title(e) + (' (custom)' if e.is_custom else ''), entry_tooltip(e))
        try:
            ims = library.sensor_imagers(s, tuple(s.stream) if len(s.stream) == 2 else None)
        except Exception:                                # noqa: BLE001 — a bad calibration: shown, not raised
            traceback.print_exc()
            ims = []
        ref = e.reference_imager
        im = next((i for i in ims if ref is not None and i.name == ref.get('name')), ims[0] if ims else None)
        if im is not None:
            d = e.imager(im.name) or {}
            f = im.camera.fov_deg()
            fig('Imager', f'{im.name} · {res_text(im.mode)} px', im.label)
            fig('FOV stated', ' × '.join(num(d.get(k), '°') for k in ('hfov_deg', 'vfov_deg', 'dfov_deg')),
                'Horizontal × vertical × diagonal as the datasheet states them (native image)')
            fig('FOV modelled', f'{f["h"]:.1f}° × {f["v"]:.1f}° × {f["d"]:.1f}°', im.label)
            fig('Shutter', d.get('shutter') or DASH)
            fig('Pixel size', num(d.get('pixel_um'), 'µm', 2))
        dp = library.depth_spec(s)
        if dp:
            fig('Baseline', num(None if dp.get('baseline_m') is None else dp['baseline_m'] * 1000, 'mm', 1))
            if im is not None:
                mz = library.min_z(s, *im.mode)
                fig('Min-Z', num(None if mz is None else mz * 1000, 'mm', 0) +
                    (f' at {res_text(im.mode)}' if mz is not None else ''),
                    'The datasheet\'s minimum depth for this stream size')
            fig('Range', num(dp.get('max_range_m'), 'm', 2), dp.get('range_text') or '')
        fig('Interface', e.data.get('interface') or DASH)
        fig('Power', num(e.data.get('power_w'), 'W', 2))
        fig('Mass', num(e.data.get('mass_g'), 'g', 0))
        for src in e.sources[:3]:
            url = str(src.get('url') or '')
            text = html.escape(str(src.get('title') or url)[:60])
            read = html.escape(str(src.get('read') or DASH))
            if url.lower().startswith(('http://', 'https://')):
                # the theme's ink, not Qt's default bright blue (unreadable on the dark panel)
                w = QLabel(f'<a href="{html.escape(url, quote=True)}" style="color: {THEME.tok["ink2"]};">{text}</a>'
                           f' · read {read}')
                w.setOpenExternalLinks(True)
            else:                                        # a document or "measured by me": not a link
                w = QLabel(f'{text} · read {read}')
            w.setProperty('muted', True)
            w.setTextFormat(Qt.TextFormat.RichText)
            w.setWordWrap(True)
            w.setToolTip(f'{url}\nCovers: {src.get("covers") or DASH}')
            sec.add(w)


# =============================================================================== camera view
class CameraViewPanel(QWidget):
    """The selected sensor's simulated image through its lens model (ui/camview.py), or an empty state while the rig
    has no sensor."""
    titleChanged = pyqtSignal(str)

    def __init__(self, ctx):
        super().__init__()
        from PyQt6.QtWidgets import QStackedLayout

        from .camview import CameraView
        self.ctx = ctx
        self.stack = QStackedLayout(self)
        self.stack.setContentsMargins(0, 0, 0, 0)
        self.empty = empty_state('No sensor to show',
                                 'Add a camera (Sensors ▸ Add camera on a tripod) or double-click a device in the '
                                 'Library: its simulated image through its own lens model appears here, posed with '
                                 'the robot.')
        self.view = CameraView(ctx)
        self.stack.addWidget(self.empty)
        self.stack.addWidget(self.view)
        ctx.store.changed.connect(self._sync)
        ctx.store.project_replaced.connect(self._sync)
        self._sync()

    def _sync(self, *_):
        self.stack.setCurrentWidget(self.view if self.ctx.store.project.variant.sensors else self.empty)

    def render_sync(self, scale: float = 1.0):
        """Render the shown sensor's image now (tests, reports): the result dict, or None without a sensor."""
        return self.view.render_sync(scale)

    def wait(self, timeout: float = 10.0) -> bool:
        return self.view.wait(timeout)


# =============================================================================== analysis
class SummaryPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.head = Headline('Dense voxels over the task volumes')
        self.b_explain = button('Explain', role='ghost', icon_name='help',
                                tip='How the count is made and why it matters')
        self.b_explain.clicked.connect(self._explain)
        self.b_run = button('Run again', icon_name='refresh', tip='Re-run the summary now')
        self.b_run.clicked.connect(lambda: ctx.analysis.run('summary'))
        self.head.actions.addWidget(self.b_run)
        self.head.actions.addWidget(self.b_explain)
        self.head.actions.addStretch(1)
        self.table = QTreeWidget()
        self.table.setColumnCount(5)
        self.table.setHeaderLabels(['Task volume', 'Size', 'Voxels at chosen size', 'Voxels at 1 mm', 'Dense 1 mm'])
        self.table.setRootIsDecorated(False)
        for k in range(1, 5):
            self.table.header().setSectionResizeMode(k, QHeaderView.ResizeMode.ResizeToContents)
        v.addWidget(self.head)
        v.addWidget(hline())
        v.addWidget(self.table, 1)
        ctx.analysis.finished.connect(self._done)

    def _explain(self):
        win = self.ctx.window
        if win is not None:
            win.msg('A dense grid has ceil(edge / voxel) cells along each axis. At 1 mm a 1.5 × 1.5 × 1.2 m volume '
                    'needs 2.7 × 10⁹ voxels, so studies use 5 mm by default and refine only near objects and '
                    'contacts.', 12000)

    def _done(self, sid, doc):
        if sid != 'summary':
            return
        vols = doc.get('volumes', [])
        total = sum(v['voxels_dense'] for v in vols)
        bad = [v for v in vols if not v['dense_feasible']]
        mm = doc.get('voxel_mm', 5)
        if not vols:
            self.head.set('0', 'voxels', 'warning', 'no task volume',
                          'Add a task volume (Workspace ▸ Add task volume) so the studies know where the work is.')
        elif bad:
            self.head.set(f'{total:,}'.replace(',', ' '), f'voxels at {mm:g} mm', 'critical', 'too many',
                          f'{bad[0]["name"]} is too large for a dense grid at {mm:g} mm: shrink it or use a coarser '
                          'voxel with a refinement region.')
        else:
            self.head.set(f'{total:,}'.replace(',', ' '), f'voxels at {mm:g} mm', 'good', 'feasible',
                          f'{len(vols)} task volume{"s" if len(vols) != 1 else ""} at {mm:g} mm. A dense 1 mm grid is '
                          'only used inside refinement regions.',
                          tip=f'Summary study · {doc.get("seconds", 0) * 1000:.1f} ms'
                              + (' (from the cache)' if doc.get('cached') else ''))
        self.table.clear()
        for vd in vols:
            sz = ' × '.join(f'{s * 1000:.0f}' for s in vd['size_m']) + ' mm'
            it = QTreeWidgetItem([vd['name'], sz, f'{vd["voxels_dense"]:,}'.replace(',', ' '),
                                  f'{vd["voxels_dense_1mm"]:,}'.replace(',', ' '),
                                  'feasible' if vd['dense_1mm_feasible'] else 'not feasible'])
            for k in (1, 2, 3):
                it.setFont(k, mono(11))
                it.setTextAlignment(k, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            sev = 'good' if vd['dense_1mm_feasible'] else 'warning'
            it.setIcon(4, icons.status_icon(sev, 12))
            self.table.addTopLevelItem(it)


class LogPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        v.addWidget(self.text)
        ctx.store.changed.connect(lambda _s: self.refill())
        ctx.store.project_replaced.connect(self.refill)
        self.refill()

    def refill(self):
        log = self.ctx.store.project.log[-400:]
        self.text.setPlainText('\n'.join(f'{e["t"]}  {e["kind"]:<10} {e["text"]}' for e in reversed(log)))


class AnalysisPanel(QWidget):
    def __init__(self, ctx):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.summary = SummaryPage(ctx)
        self.log = LogPage(ctx)
        self.tabs.addTab(self.summary, 'Summary')
        self.tabs.addTab(self.log, 'Design log')
        v.addWidget(self.tabs)

        self.extra = {}

    def add_page(self, pid, title, widget):
        self.extra[pid] = widget
        self.tabs.insertTab(self.tabs.count() - 1, widget, title)      # the design log stays last

    def show_tab(self, name):
        w = {'summary': self.summary, 'log': self.log, **self.extra}.get(name, self.summary)
        self.tabs.setCurrentWidget(w)


# =============================================================================== assistant
class AssistantPanel(QWidget):
    def __init__(self, ctx):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        head = QWidget()
        head.setFixedHeight(22)
        h = QHBoxLayout(head)
        h.setContentsMargins(8, 0, 4, 0)
        h.addWidget(label('Assistant'))
        h.addStretch(1)
        v.addWidget(head)
        v.addWidget(hline())
        v.addWidget(empty_state('Not connected yet',
                                'The assistant reads the page you are on and cites the figures shown; it never invents '
                                'numbers. It arrives with the MCP server in Phase 10 (Groq or Claude, with keys kept '
                                'in the system keyring).'), 1)


__all__ = ['Outliner', 'LibraryPanel', 'Inspector', 'CameraViewPanel', 'AnalysisPanel', 'AssistantPanel',
           'PoseScrubber', 'Headline', 'empty_state', 'STATUS', 'sans']
