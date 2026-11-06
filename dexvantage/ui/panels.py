"""The dock panels: Outliner, Library, Inspector, Camera view, Analysis, Assistant, and the pose scrubber under the
3D view.

Each follows Vantage's page pattern: what it shows, the number that matters (a large mono figure with a status word)
and what to do about it. Panels whose engine arrives in a later phase say so plainly instead of showing invented
content.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPlainTextEdit,
                             QScrollArea, QSlider, QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from . import icons
from .theme import THEME, STATUS, mono, sans
from .widgets import ScrubSpinBox, Section, StatusLine, button, hline, label, tool_button

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
        THEME.changed.connect(lambda *_: self.refill())      # icons are tinted per theme
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
            self._item(g, r.name + (' · primary' if r.primary else ''), f'{len(r.joints)} joints', ('robot', r.uid),
                       'robot-arm')
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
        self._group('Tasks', 'grasp', len(t.grasp_sets) + len(t.trajectories) + len(t.teleop_devices))
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
LIBRARY_GROUPS = (
    ('Robots', 'robot-arm', 1, 'Robots are read from their official model files (URDF or MJCF). Joint limits come '
                              'from the file; nothing is retyped.'),
    ('Cameras', 'camera-wide', 2, 'Global-shutter USB and GMSL cameras, each figure from the maker\'s datasheet with '
                                  'its source and the date it was read.'),
    ('Depth sensors', 'depth', 2, 'RealSense, ZED, OAK-D and Orbbec units: baseline, depth FOV, min-Z per resolution '
                                  'and stream modes, from datasheets only.'),
    ('Tactile sensors', 'tactile', 2, 'GelSight Mini and DIGIT: pad size, internal camera and interface.'),
    ('Grippers', 'gripper', 1, 'From their official model files.'),
    ('Mounts', 'bracket', 6, 'ISO 9409-1 flange adapters and 2020/4040 T-slot clamps, generated as printable STL.'),
)


class LibraryPanel(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.search = QLineEdit()
        self.search.setPlaceholderText('Search the library')
        s = QWidget()
        sl = QHBoxLayout(s)
        sl.setContentsMargins(6, 4, 6, 4)
        sl.addWidget(self.search)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setColumnCount(2)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        v.addWidget(s)
        v.addWidget(hline())
        v.addWidget(self.tree, 1)
        THEME.changed.connect(lambda *_: self.refill())
        self.refill()

    def refill(self):
        self.tree.clear()
        for title, icon, phase, text in LIBRARY_GROUPS:
            it = QTreeWidgetItem([title, '0'])
            it.setIcon(0, icons.icon(icon, 12))
            it.setFont(1, mono(11))
            it.setForeground(1, THEME.color('muted'))
            it.setToolTip(0, f'{text}\nArrives in Phase {phase}.')
            note = QTreeWidgetItem([f'Arrives in Phase {phase}', ''])
            note.setForeground(0, THEME.color('muted'))
            note.setToolTip(0, text)
            note.setFlags(Qt.ItemFlag.ItemIsEnabled)
            it.addChild(note)
            self.tree.addTopLevelItem(it)


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
        self.rebuild()

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

    def _vec(self, sec, title, values, apply, minimum=None):
        """Three length fields in metres (3 decimals: millimetre steps), as Vantage's inspector; typing '450 mm'
        works too."""
        boxes = []
        for k, ax in enumerate('XYZ'):
            b = ScrubSpinBox(ax, 'length', 3)
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
        sec.row('Model', s.model_id or 'none yet (library in Phase 2)')
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
        return sec


# =============================================================================== camera view
class CameraViewPanel(QWidget):
    titleChanged = pyqtSignal(str)

    def __init__(self, ctx):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(empty_state('No camera view yet',
                                'The selected camera\'s simulated image, its frustum overlays, the depth-valid mask '
                                'and the finger shadows appear here once the lens models arrive (Phase 2).'))


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

    def show_tab(self, name):
        self.tabs.setCurrentWidget({'summary': self.summary, 'log': self.log}.get(name, self.summary))


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


# =============================================================================== pose scrubber
class PoseScrubber(QWidget):
    """The joint-state timeline along the 3D view's bottom edge: scrubs a trajectory or a grasp set."""

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setFixedHeight(30)
        self.setObjectName('panel')
        h = QHBoxLayout(self)
        h.setContentsMargins(6, 0, 8, 0)
        h.setSpacing(4)
        self.b_back = tool_button('step-back', 'Previous pose', size=22, icon_size=12)
        self.b_play = tool_button('play', 'Play', size=22, icon_size=12)
        self.b_step = tool_button('step', 'Next pose', size=22, icon_size=12)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.where = label('', muted=True, small=True)
        self.where.setFont(mono(11))
        for w in (self.b_back, self.b_play, self.b_step):
            h.addWidget(w)
        h.addWidget(self.slider, 1)
        h.addWidget(self.where)
        ctx.store.changed.connect(lambda _s: self.refresh())
        ctx.store.project_replaced.connect(self.refresh)
        self.refresh()

    def refresh(self):
        p = self.ctx.store.project
        n = sum(len(t.get('q') or []) for t in p.tasks.trajectories)
        on = bool(p.robots) and n > 0
        for w in (self.b_back, self.b_play, self.b_step, self.slider):
            w.setEnabled(on)
        self.where.setText('pose 0 / 0' if not on else f'pose 1 / {n}')
        self.setToolTip('' if on else 'No trajectory or grasp set yet: the scrubber moves through joint states once a '
                                      'robot is imported (Phase 1).')

    def paintEvent(self, e):
        from PyQt6.QtGui import QPainter
        p = QPainter(self)
        p.fillRect(self.rect(), THEME.color('panel'))
        p.setPen(THEME.color('hair'))
        p.drawLine(0, 0, self.width(), 0)
        p.end()


__all__ = ['Outliner', 'LibraryPanel', 'Inspector', 'CameraViewPanel', 'AnalysisPanel', 'AssistantPanel',
           'PoseScrubber', 'Headline', 'empty_state', 'STATUS', 'sans']
