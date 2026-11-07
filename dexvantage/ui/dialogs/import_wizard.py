"""Robot ▸ Import robot…: the import wizard (spec §13), on the step pattern of Vantage ui/dialogs/import_wizard.py.

Steps:
1. File: a URDF, MuJoCo MJCF or USD file (dropped or browsed); its name in the project. A USD file (a UsdPhysics
   articulation; needs the optional usd-core package, and the step says so when it is missing) is converted to MJCF
   by core/robot/usd_import.py and imported from that; the conversion notes are in the robot's source note.
2. Meshes: every mesh, included file and texture the model names, found or missing, with the importer's plain
   message for a missing one (core/robot/filecheck.py).
3. Scale and units: the robot's bounding box at its home pose in millimetres and every mesh's size. A mesh over
   MESH_MAX_EXTENT (10 m) is taken to be in millimetres and stops the import, as the importer does; a robot outside
   SIZE_RANGE (20 mm to 5 m across, a plausibility default, not a rule) gets a warning.
4. Joints: every joint with its type and the limits, velocity and effort the file states (degrees or millimetres;
   read-only: nothing is typed in again). Tick the joints that open and close the gripper: they are stored as
   'gripper': True on the joint's record in Robot.joints.
5. Gripper and pads: the gripper frame (the frame IK drags and the grasp studies use), the fingertip frames, and the
   contact pads: named boxes on links, in the link's frame (Robot.contact_pads: [{'link', 'name', 'box': {'center',
   'size'}}], metres). A new pad starts as the bounding box of its link's visual meshes.
6. Summary: what will be added. Import adds the robot and its files as one undo step.

A preview viewport (ui/viewport3d.make_viewport with a small SceneProvider over a throw-away project holding only this
robot) shows the robot at home, the gripper frame and fingertip frames as axes, the pads as boxes and the selected
joint's axis.

The model is read on a worker thread (the window stays responsive for large meshes); load(path, background=False)
reads it on the calling thread (tests, scripts).

Known limits: frames are links (MJCF sites are not offered yet); pads are axis-aligned boxes in the link frame; xacro
and Menagerie import are not part of this wizard yet; a USD robot's mesh step lists the stage's layers, not each
mesh.
"""
from __future__ import annotations

import importlib.util
import os
import re
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPen
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
                             QLineEdit, QListWidget, QListWidgetItem, QPushButton, QStackedWidget,
                             QStyledItemDelegate, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ...core import project as P
from ...core.robot.importer import MESH_MAX_EXTENT
from .. import icons
from .. import jointfmt as JF
from ..scene import SceneProvider, box_edges
from ..theme import STATUS, THEME, mono, sans
from ..viewport3d import LineDraw
from ..widgets import ROW_H, ScrubSpinBox, StatusLine, button, hline, label
from .base import VDialog

STEPS = ['File', 'Meshes', 'Scale and units', 'Joints', 'Gripper and pads', 'Summary']
HINTS = ['Drop a URDF, MJCF or USD file anywhere on this dialog. The meshes it names are read from the same folder.',
         'Every file the model names; a missing mesh stops the import, a missing texture does not.',
         'Robot files are in metres. A mesh of hundreds of metres was exported in millimetres.',
         'Limits are the file\'s own figures and are not edited here. Tick the joints of the gripper.',
         'The gripper frame is what IK drags and what the grasp studies aim; pads are where the fingers touch.',
         'Import adds the robot and its files as one undo step.']
SIZE_RANGE = (0.02, 5.0)          # m: a robot outside this is flagged for a second look (a plausibility default)
USD_EXTS = ('.usd', '.usda', '.usdc', '.usdz')
PAD_PLACEHOLDER = 0.01            # m: a pad's edge when its link has no visual mesh to take the size from
GRIPPER_WORDS = ('finger', 'gripper', 'jaw', 'claw', 'grip')
FRAME_TOKENS = ('tcp', 'tool0', 'tool', 'ee', 'eef', 'end_effector', 'endeffector', 'gripper', 'grasp', 'hand',
                'flange')
TIP_TOKENS = ('tip', 'fingertip', 'distal')


def usd_message() -> str:
    """Why a USD file cannot be read here, or '' when it can (usd-core installed)."""
    if importlib.util.find_spec('pxr') is None:
        return ('USD files need the optional usd-core package (optional: install usd-core, for example pip install '
                '"dexvantage[usd]"), then import the file again; or export the robot as URDF or MJCF and import that.')
    return ''


def _n(k: int, word: str) -> str:
    """'1 link', '3 links'."""
    return f'{k} {word}{"" if k == 1 else "s"}'


def _tokens(name: str) -> list:
    return [t for t in re.split(r'[^a-z0-9]+', name.lower()) if t]


def guess_frame(links: list, exclude=()) -> str:
    """The link most likely to be the gripper frame by its name (tcp, tool0, ee, …), else the last link; links in
    `exclude` (the fingers: links a gripper joint moves) are never guessed."""
    links = [n for n in links if n not in set(exclude)] or list(links)
    for want in FRAME_TOKENS:
        for n in reversed(links):
            if want in _tokens(n) or (len(want) > 3 and want in n.lower().replace('_', '')):
                return n
    return links[-1] if links else ''


def finger_links(model, gripper_joints) -> set:
    """The links that a gripper joint moves (a finger, a jaw, a linkage bar)."""
    from ...core.robot.manipulability import chain_mask
    g = np.isin(model.joint_names, list(gripper_joints))
    if not g.any():
        return set()
    return {n for n in model.link_names if (chain_mask(model, n) & g).any()}


# ====================================================================== loading
class _LoadSig(QObject):
    loaded = pyqtSignal(object)


def load_robot(path: str) -> dict:
    """{'path', 'check', 'robot', 'sources', 'error'}: the reference check and the import (units unchecked: the
    wizard's own step shows them). Never raises; 'error' holds the plain message."""
    from ...core.robot import import_robot
    from ...core.robot.filecheck import check_references
    from ...core.robot.model import RobotModelError, get_model
    out = {'path': path, 'check': None, 'robot': None, 'sources': None, 'error': '', 'project': None}
    if os.path.splitext(path)[1].lower() in USD_EXTS and usd_message():
        out['error'] = usd_message()
        return out
    try:
        out['check'] = check_references(path)
    except P.ProjectFormatError as e:
        out['error'] = str(e)
        return out
    try:
        robot, sources = import_robot(path, check_units=False)
    except P.ProjectFormatError as e:
        out['error'] = str(e)
        return out
    except Exception as e:                             # noqa: BLE001 — explained, the traceback logged
        traceback.print_exc()
        out['error'] = (f'{os.path.basename(path)} could not be read ({type(e).__name__}). Details are in the log.')
        return out
    robot.primary = True
    proj = P.Project('Import preview')
    proj.robots = [robot]
    proj.sources = dict(sources)
    try:
        get_model(proj, robot.uid)                     # compiled here, shared with the preview through the cache
    except RobotModelError as e:
        out['error'] = str(e)
        return out
    out.update(robot=robot, sources=sources, project=proj)
    return out


# ====================================================================== the preview
class _PreviewCtx:
    """The little a SceneProvider needs: a store over the preview project, the view toggles, overlay sources."""

    def __init__(self, project):
        from ..store import ProjectStore
        self.store = ProjectStore(project)
        self.overlays = {'grid': True, 'frames': True, 'volumes': False, 'shadows': True}
        self.overlay_sources = []
        self.window = None

    def overlay(self, key) -> bool:
        return bool(self.overlays.get(key, False))


class PreviewScene(SceneProvider):
    """The robot alone at its home pose, with the wizard's tags drawn on it."""

    def __init__(self, wiz, project):
        super().__init__(_PreviewCtx(project))
        self.wiz = wiz
        self.ctx.overlay_sources.append(self._tag_lines)

    def hud_label(self, vp):
        rm = self.robot_models()
        if not rm:
            return 'Preview'
        r, m, _q = rm[0]
        return f'Preview · {r.name} · {len(m.link_names)} links · {m.nq} moving joints · home pose'

    def _tag_lines(self, vp) -> list:
        rm = self.robot_models()
        if not rm:
            return []
        _r, m, q = rm[0]
        T = m.fk(q)
        out = []

        def triad(Tl, L, w):
            for k, col in enumerate(('#e0533f', '#3fae49', '#3d86e6')):
                out.append(LineDraw(np.array([Tl[:3, 3], Tl[:3, 3] + Tl[:3, k] * L]), QColor(col), w, depth=False))
        w = self.wiz
        if w.gripper_frame in T:
            triad(T[w.gripper_frame], 0.06, 2.4)
        for f in w.fingertips:
            if f in T:
                triad(T[f], 0.025, 1.4)
        for i, pad in enumerate(w.pads):
            Tl = T.get(pad['link'])
            if Tl is None:
                continue
            c, s = np.asarray(pad['box']['center'], float), np.asarray(pad['box']['size'], float)
            e = box_edges(c - s / 2, c + s / 2) @ Tl[:3, :3].T + Tl[:3, 3]
            on = i == w.pad_row()
            out.append(LineDraw(e, QColor(STATUS['warning']), 2.0 if on else 1.2, segments=True, depth=False))
        j = w.selected_joint()
        if j is not None and j.get('child') in T and j.get('type') != 'fixed':
            Tl = T[j['child']]
            a = Tl[:3, :3] @ np.asarray(j.get('axis') or [1, 0, 0], float)
            o = Tl[:3, :3] @ np.asarray(j.get('anchor') or [0, 0, 0], float) + Tl[:3, 3]
            n = np.linalg.norm(a)
            if n > 1e-9:
                a = a / n * 0.08
                out.append(LineDraw(np.array([o - a, o + a]), THEME.color('ink'), 2.0, depth=False))
        return out


# ====================================================================== the step list
class StepDelegate(QStyledItemDelegate):
    """Ported from Vantage's wizard: number or tick, title, and a one-line summary of the step."""

    def __init__(self, wiz):
        super().__init__()
        self.wiz = wiz

    def sizeHint(self, opt, idx):
        from PyQt6.QtCore import QSize
        return QSize(opt.rect.width(), 42)

    def paint(self, p, opt, idx):
        i = idx.row()
        r = opt.rect
        cur = i == self.wiz.step
        if cur:
            p.fillRect(r, THEME.color('sel'))
            p.fillRect(QRectF(r.left(), r.top(), 2, r.height()), THEME.color('ink'))
        done = i < self.wiz.step
        x = r.left() + 14
        cy = r.top() + 16
        if done:
            p.drawPixmap(int(x), int(cy - 6), icons.pixmap('good', STATUS['good'], 12))
        else:
            p.setPen(QPen(THEME.color('ink' if cur else 'strong'), 1))
            p.setBrush(THEME.color('ink') if cur else Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(x - 1, cy - 7, 14, 14))
            p.setFont(mono(10))
            p.setPen(THEME.color('panel') if cur else THEME.color('ink2'))
            p.drawText(QRectF(x - 1, cy - 7, 14, 14), Qt.AlignmentFlag.AlignCenter, str(i + 1))
        p.setFont(sans(12))
        p.setPen(THEME.color('ink' if cur or done else 'ink2'))
        p.drawText(QPointF(x + 22, cy + 4), STEPS[i])
        sub = self.wiz.step_summary(i) if i <= self.wiz.step else ''
        if sub:
            p.setFont(mono(10))
            p.setPen(THEME.color('muted'))
            p.drawText(QPointF(x + 22, cy + 19), sub)


def _table(cols) -> QTableWidget:
    t = QTableWidget(0, len(cols))
    t.setHorizontalHeaderLabels(cols)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H + 2)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    hh = t.horizontalHeader()
    hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    for k in range(1, len(cols)):
        hh.setSectionResizeMode(k, QHeaderView.ResizeMode.ResizeToContents)
    return t


def _cell(text, num=False, tip='', muted=False):
    it = QTableWidgetItem(text)
    it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)      # read-only: the file's own figures
    if num:
        it.setFont(mono(11))
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if tip:
        it.setToolTip(tip)
    if muted:
        it.setForeground(THEME.color('muted'))
    return it


# ====================================================================== the wizard
class ImportWizard(VDialog):
    def __init__(self, ctx, parent=None, path=None, soft_3d=None):
        super().__init__(parent, 'Import robot', 'Step 1 of 6', (1180, 720))
        self.ctx = ctx
        self.soft_3d = bool(getattr(parent, 'soft_3d', False)) if soft_3d is None else soft_3d
        self.path = ''
        self.res = None                # load_robot's result
        self.loading = False
        self.step = 0
        self.imported = None           # the Robot added to the project on Import
        self.gripper_joints: list = []
        self.gripper_frame = ''
        self.fingertips: list = []
        self.pads: list = []
        self.scene = None
        self.vp = None
        self.sig = _LoadSig()
        self.sig.loaded.connect(self._loaded)
        self.setAcceptDrops(True)
        lay = QHBoxLayout(self.body)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.steps = QListWidget()
        self.steps.setObjectName('steps')
        self.steps.setFixedWidth(200)
        self.steps.setItemDelegate(StepDelegate(self))
        for s in STEPS:
            self.steps.addItem(s)
        self.steps.itemClicked.connect(lambda it: self.go(self.steps.row(it)))
        lay.addWidget(self.steps)
        self.stack = QStackedWidget()
        lay.addWidget(self.stack, 1)
        self.preview = QWidget()
        self.preview.setMinimumWidth(380)
        self.pv_lay = QVBoxLayout(self.preview)
        self.pv_lay.setContentsMargins(0, 0, 0, 0)
        self.pv_lay.setSpacing(0)
        self.pv_empty = label('The robot appears here once its file is read.', muted=True, wrap=True)
        self.pv_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pv_lay.addWidget(self.pv_empty, 1)
        lay.addWidget(self.preview, 1)
        self._page_file()
        self._page_meshes()
        self._page_scale()
        self._page_joints()
        self._page_tags()
        self._page_summary()
        self.b_cancel = button('Cancel', 'ghost')
        self.b_cancel.clicked.connect(self.reject)
        self.b_back = button('Back')
        self.b_back.clicked.connect(lambda: self.go(self.step - 1))
        self.b_next = button('Next', 'primary')
        self.b_next.clicked.connect(self.next)
        for b in (self.b_cancel, self.b_back, self.b_next):
            self.buttons.addWidget(b)
        self.go(0)
        if path:
            QTimer.singleShot(0, lambda: self.load(path))

    @classmethod
    def run(cls, win, path=None) -> bool:
        """Open the wizard over the window; True when a robot was imported."""
        w = cls(win.ctx, win, path)
        try:
            w.exec()
            return w.imported is not None
        finally:
            w.release()
            w.deleteLater()

    # ------------------------------------------------------------------ state
    @property
    def robot(self):
        return self.res['robot'] if self.res else None

    @property
    def model(self):
        from ...core.robot.model import RobotModelError, get_model
        if self.robot is None:
            return None
        try:
            return get_model(self.res['project'], self.robot.uid)
        except RobotModelError:
            return None

    def units_problem(self) -> list:
        """[(mesh name, key, extent m)] of the meshes over MESH_MAX_EXTENT (the importer's units rule)."""
        m = self.model
        return [x for x in m.mesh_extents() if x[2] > MESH_MAX_EXTENT] if m is not None else []

    def can_import(self) -> bool:
        return self.robot is not None and self.model is not None and not self.units_problem()

    def _max_step(self):
        if self.robot is not None:
            return len(STEPS) - 1
        return 1 if self.res and self.res.get('check') else 0

    # ------------------------------------------------------------------ pages
    def _page(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(10)
        self.stack.addWidget(w)
        return lay

    def _page_file(self):
        lay = self._page()
        self.drop = QPushButton()
        self.drop.setObjectName('drop')
        self.drop.setProperty('role', 'ghost')
        self.drop.setStyleSheet(f'QPushButton#drop {{ border: 1px dashed {THEME.tok["strong"]}; border-radius: 2px; '
                                'min-height: 110px; max-height: 110px; }')
        dl = QHBoxLayout(self.drop)
        dl.setContentsMargins(20, 0, 20, 0)
        col = QVBoxLayout()
        self.f_name = label('Drop a robot file here')
        self.f_name.setFont(mono(13))
        self.f_meta = label('or press Browse…', muted=True, mono_=True, small=True)
        col.addStretch(1)
        col.addWidget(self.f_name)
        col.addWidget(self.f_meta)
        col.addStretch(1)
        br = button('Browse…')
        br.clicked.connect(self.browse)
        dl.addLayout(col, 1)
        dl.addWidget(br)
        self.drop.clicked.connect(self.browse)
        lay.addWidget(self.drop)
        g = QGridLayout()
        g.setHorizontalSpacing(10)
        g.addWidget(label('Supported', ink2=True), 0, 0)
        chips = QHBoxLayout()
        for e, tip in (('URDF', '.urdf (expand a .xacro to plain URDF first)'), ('MJCF', 'MuJoCo .xml'),
                       ('USD', 'a UsdPhysics articulation, converted to MJCF (needs the optional usd-core)')):
            c = label(e, mono_=True)
            c.setToolTip(tip)
            c.setStyleSheet(f'border: 1px solid {THEME.tok["strong"]}; border-radius: 2px; padding: 1px 6px;')
            chips.addWidget(c)
        chips.addStretch(1)
        g.addLayout(chips, 0, 1)
        g.addWidget(label('Name', ink2=True), 1, 0)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText('The robot\'s name in the project')
        g.addWidget(self.name_edit, 1, 1)
        g.setColumnStretch(1, 1)
        lay.addLayout(g)
        self.f_status = StatusLine('', '')
        lay.addWidget(self.f_status)
        lay.addStretch(1)

    def _page_meshes(self):
        lay = self._page()
        self.m_status = StatusLine('', '')
        lay.addWidget(self.m_status)
        self.m_table = _table(['File', 'Kind', 'Status', 'Read from'])
        self.m_table.itemSelectionChanged.connect(self._mesh_picked)
        lay.addWidget(self.m_table, 1)
        self.m_detail = label('', muted=True, wrap=True)
        lay.addWidget(self.m_detail)

    def _page_scale(self):
        lay = self._page()
        self.s_box = label('', mono_=True)
        self.s_status = StatusLine('', '')
        lay.addWidget(label('Bounding box at the home pose', ink2=True))
        lay.addWidget(self.s_box)
        lay.addWidget(self.s_status)
        self.s_table = _table(['Mesh', 'Largest edge', 'Status'])
        lay.addWidget(self.s_table, 1)

    def _page_joints(self):
        lay = self._page()
        self.j_note = label('', muted=True, wrap=True)
        lay.addWidget(self.j_note)
        self.j_table = _table(['Joint', 'Type', 'Lower', 'Upper', 'Velocity', 'Effort', 'Gripper'])
        self.j_table.itemChanged.connect(self._joint_ticked)
        self.j_table.itemSelectionChanged.connect(self._repaint)
        lay.addWidget(self.j_table, 1)

    def _page_tags(self):
        lay = self._page()
        g = QGridLayout()
        g.setHorizontalSpacing(10)
        g.addWidget(label('Gripper frame', ink2=True), 0, 0)
        self.t_frame = QComboBox()
        self.t_frame.setToolTip('The frame IK drags and the grasp studies aim (a link of the robot)')
        self.t_frame.activated.connect(lambda i: self.set_gripper_frame(self.t_frame.itemData(i)))
        g.addWidget(self.t_frame, 0, 1)
        g.setColumnStretch(1, 1)
        lay.addLayout(g)
        lay.addWidget(label('Fingertip frames', ink2=True))
        self.t_tips = QListWidget()
        self.t_tips.setMaximumHeight(150)
        self.t_tips.itemChanged.connect(self._tip_ticked)
        lay.addWidget(self.t_tips)
        lay.addWidget(hline())
        head = QHBoxLayout()
        head.addWidget(label('Contact pads', ink2=True))
        head.addStretch(1)
        self.t_pad_link = QComboBox()
        self.t_pad_link.setToolTip('The link a new pad goes on')
        b_add = button('Add pad', icon_name='plus', tip='A pad on the chosen link, sized to its visual mesh')
        b_add.clicked.connect(lambda: self.add_pad(self.t_pad_link.currentData()))
        b_del = button('Remove', 'ghost', icon_name='trash')
        b_del.clicked.connect(lambda: self.remove_pad(self.pad_row()))
        for w in (self.t_pad_link, b_add, b_del):
            head.addWidget(w)
        lay.addLayout(head)
        self.t_pads = _table(['Pad', 'Link', 'Centre (mm)', 'Size (mm)'])
        self.t_pads.itemSelectionChanged.connect(self._pad_picked)
        lay.addWidget(self.t_pads, 1)
        ed = QGridLayout()
        ed.setHorizontalSpacing(4)
        self.p_name = QLineEdit()
        self.p_name.editingFinished.connect(lambda: self.update_pad(self.pad_row(), name=self.p_name.text()))
        ed.addWidget(label('Name', ink2=True), 0, 0)
        ed.addWidget(self.p_name, 0, 1, 1, 3)
        self.p_c, self.p_s = [], []
        for row, (title, store) in enumerate((('Centre', self.p_c), ('Size', self.p_s)), start=1):
            ed.addWidget(label(title, ink2=True), row, 0)
            for k, ax in enumerate('XYZ'):
                b = ScrubSpinBox(ax, 'length', 1, 'mm')
                b.set_display_unit('mm')
                if title == 'Size':
                    b.setMinimum(0.1)
                b.committed.connect(lambda _v, t=title: self._pad_edited(t))
                ed.addWidget(b, row, 1 + k)
                store.append(b)
        self.pad_editor = QWidget()
        self.pad_editor.setLayout(ed)
        self.pad_editor.setEnabled(False)
        lay.addWidget(self.pad_editor)

    def _page_summary(self):
        lay = self._page()
        self.sum_grid = QGridLayout()
        self.sum_grid.setHorizontalSpacing(12)
        self.sum_grid.setVerticalSpacing(4)
        lay.addLayout(self.sum_grid)
        self.sum_status = StatusLine('', '')
        lay.addWidget(self.sum_status)
        self.sum_notes = label('', muted=True, wrap=True)
        lay.addWidget(self.sum_notes)
        lay.addStretch(1)

    # ------------------------------------------------------------------ navigation
    def go(self, i):
        i = max(0, min(int(i), self._max_step()))
        self.step = i
        self.stack.setCurrentIndex(i)
        self.set_subtitle(f'Step {i + 1} of {len(STEPS)}')
        self.steps.viewport().update()
        self.b_back.setEnabled(i > 0)
        last = i == len(STEPS) - 1
        self.b_next.setText('Import' if last else 'Next')
        self.b_next.setEnabled(self.can_import() if last else i < self._max_step())
        self.hint.setText(HINTS[i])
        [self._fill_file, self._fill_meshes, self._fill_scale, self._fill_joints, self._fill_tags,
         self._fill_summary][i]()
        return self.step

    def next(self):
        if self.step == len(STEPS) - 1:
            return self.finish()
        self.go(self.step + 1)
        return True

    def step_summary(self, i):
        r = self.robot
        if self.res is None:
            return ''
        if i == 0:
            return os.path.basename(self.path)[:28]
        if i == 1 and self.res.get('check'):
            items = self.res['check']['items']
            miss = sum(1 for x in items if not x['found'] and x['required'])
            return f'{len(items) - miss} of {len(items)} found'
        if r is None:
            return ''
        m = self.model
        if i == 2 and m is not None:
            lo, hi = m.bounds()
            return f'{(hi - lo).max() * 1000:.0f} mm across'
        if i == 3 and m is not None:
            return f'{m.nq} moving · {len(self.gripper_joints)} gripper'
        if i == 4:
            return f'{self.gripper_frame or "no frame"} · {len(self.pads)} pads'[:28]
        return ''

    # ------------------------------------------------------------------ loading
    def browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Robot model', self.ctx.settings.value('dir/robots', '') if hasattr(self.ctx, 'settings') else '',
            'Robot model (*.urdf *.xml *.usd *.usda *.usdc *.usdz);;URDF (*.urdf);;MuJoCo MJCF (*.xml);;'
            'USD (*.usd *.usda *.usdc *.usdz)')
        if path:
            if hasattr(self.ctx, 'settings'):
                self.ctx.settings.setValue('dir/robots', os.path.dirname(path))
            self.load(path)

    def load(self, path: str, background: bool = True):
        """Read the file (on a worker thread unless background is False)."""
        self.path = path
        self.loading = True
        self.f_name.setText(os.path.basename(path))
        self.f_meta.setText('Reading the model and its meshes…')
        self.f_status.set('', '')
        self.b_next.setEnabled(False)
        if not background:
            self._loaded(load_robot(path))
            return self.res

        def work():
            res = load_robot(path)
            try:
                self.sig.loaded.emit(res)
            except RuntimeError:
                pass                                   # the dialog closed meanwhile
        threading.Thread(target=work, daemon=True).start()
        return None

    def wait(self, timeout=20.0) -> bool:
        """Process events until a background load finishes (tests, scripts)."""
        import time

        from PyQt6.QtCore import QCoreApplication
        t0 = time.time()
        while self.loading and time.time() - t0 < timeout:
            QCoreApplication.processEvents()
            time.sleep(0.005)
        return not self.loading

    def _loaded(self, res):
        if res.get('path') != self.path:
            return                                     # an older file's result
        self.loading = False
        self.release()
        self.res = res
        r = res['robot']
        if r is not None:
            m = self.model
            self.name_edit.setText(r.name)
            moving = set(m.joint_names) if m is not None else set()
            self.gripper_joints = [j['name'] for j in r.joints if j.get('name') in moving and
                                   any(w in j['name'].lower() for w in GRIPPER_WORDS)]
            links = m.link_names if m is not None else []
            self.gripper_frame = guess_frame(links, finger_links(m, self.gripper_joints) if m is not None else ())
            self.fingertips = [n for n in links if any(t in _tokens(n) for t in TIP_TOKENS)]
            self.pads = []
            self.scene = PreviewScene(self, res['project'])
            if self.soft_3d:
                from ..viewport3d import SoftViewport
                self.vp = SoftViewport(self.scene)
            else:
                from ..viewport3d import make_viewport
                self.vp = make_viewport(self.scene)
            self.pv_empty.hide()
            self.pv_lay.addWidget(self.vp, 1)
            QTimer.singleShot(0, self._frame)
        self.go(0)

    def _frame(self):
        if self.vp is not None and self.scene is not None:
            lo, hi, mg = self.scene.home_box(self.vp)
            if self.model is not None:
                lo, hi = self.model.bounds()
                mg = 1.4
            self.vp.frame_box(lo, hi, mg)

    def release(self):
        """Drop the preview (its viewport and the throw-away project)."""
        if self.vp is not None:
            self.pv_lay.removeWidget(self.vp)
            self.vp.hide()
            self.vp.deleteLater()
            self.vp = None
        if self.scene is not None:
            self.scene.ctx.store.project = P.Project('Import preview')
            self.scene._mesh_draw = None
            self.scene = None
        self.pv_empty.show()

    def _repaint(self):
        if self.vp is not None:
            self.vp.update()
        self.steps.viewport().update()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if urls:
            self.load(urls[0])

    # ------------------------------------------------------------------ fill the steps
    def _fill_file(self):
        if self.loading:
            return
        if self.res is None:
            self.f_status.set('', 'URDF, MuJoCo MJCF or USD, with the meshes it names in the same folder (or its '
                                  'ROS package).')
            return
        r = self.robot
        if r is None:
            self.f_meta.setText('Could not be read')
            self.f_status.set('critical', self.res['error'])
            return
        m = self.model
        size = sum(len(b) for b in self.res['sources'].values())
        kind = ('USD, converted to MJCF' if os.path.splitext(self.path or '')[1].lower() in USD_EXTS
                else r.source.kind.upper())
        self.f_meta.setText(f'{kind}\n{len(self.res["sources"])} files · {size / 1e6:.2f} MB · '
                            f'sha256 {r.source.sha256[:12]}')          # two short lines: clear of Browse…
        self.f_status.set('good', f'Read: {_n(len(m.link_names), "link")}, {_n(m.nq, "moving joint")}'
                                  + (f'. {r.source.note}' if r.source.note else '.'))

    def _fill_meshes(self):
        chk = (self.res or {}).get('check')
        self.m_table.setRowCount(0)
        if not chk:
            self.m_status.set('', 'Read a file first.')
            return
        items = chk['items']
        for x in items:
            row = self.m_table.rowCount()
            self.m_table.insertRow(row)
            sev = 'good' if x['found'] else ('critical' if x['required'] else 'warning')
            word = 'found' if x['found'] else ('not read' if x['what'] == 'model' else
                                               'missing' if x['required'] else 'missing (optional)')
            self.m_table.setItem(row, 0, _cell(x['ref'] or '(file)', tip=x['message'] or x['ref']))
            self.m_table.setItem(row, 1, _cell(x['what']))
            st = _cell(word)
            st.setIcon(icons.status_icon(sev, 12))
            self.m_table.setItem(row, 2, st)
            self.m_table.setItem(row, 3, _cell(x['file'] or '—', muted=not x['found']))
        unread = [x for x in items if x['what'] == 'model']
        items = [x for x in items if x['what'] != 'model']
        miss = [x for x in items if not x['found'] and x['required']]
        opt = [x for x in items if not x['found'] and not x['required']]
        if unread:
            self.m_status.set('critical', f'{unread[0]["message"]} The files it names cannot be listed until it '
                                          'reads.')
        elif miss:
            self.m_status.set('critical', f'{len(miss)} of {_n(len(items), "file")} '
                                          f'{"is" if len(miss) == 1 else "are"} missing. {miss[0]["message"]}')
        elif not items:
            self.m_status.set('good', 'The model names no mesh files: it is drawn from its primitive shapes.')
        else:
            self.m_status.set('warning' if opt else 'good',
                              (f'All {len(items) - len(opt)} required files found' if len(items) - len(opt) != 1
                               else 'The required file was found')
                              + ((f'; {len(opt)} layer{"s" if len(opt) != 1 else ""} missing: what '
                                  f'{"they hold" if len(opt) != 1 else "it holds"} (often meshes) is left out.'
                                  if chk.get('kind') == 'usd' else
                                  f'; {len(opt)} texture{"s" if len(opt) != 1 else ""} missing: the robot is drawn '
                                  'without them.') if opt else '.'))
        if self.robot is None and self.res.get('error') and not miss and not unread:
            self.m_status.set('critical', self.res['error'])
        self.m_detail.setText('')

    def _mesh_picked(self):
        rows = {i.row() for i in self.m_table.selectedItems()}
        chk = (self.res or {}).get('check') or {}
        items = chk.get('items', [])
        if rows and min(rows) < len(items):
            x = items[min(rows)]
            self.m_detail.setText(x['message'] or f'{x["ref"]} is read from {x["file"]}.')

    def _fill_scale(self):
        m = self.model
        self.s_table.setRowCount(0)
        if m is None:
            return
        lo, hi = m.bounds()
        ext = hi - lo
        a, b = np.round(lo * 1000) + 0.0, np.round(hi * 1000) + 0.0      # mm; + 0.0 turns −0 into 0
        def mm(v, sign=''):
            return f'{v:{sign},.0f}'.replace(',', ' ').replace('-', '−')
        self.s_box.setText(f'{mm(ext[0] * 1000)} × {mm(ext[1] * 1000)} × {mm(ext[2] * 1000)} mm   ('
                           + ', '.join(f'{ax} {mm(a[k], "+")} … {mm(b[k], "+")}' for k, ax in enumerate('xyz')) + ')')
        bad = self.units_problem()
        for name, key, e in m.mesh_extents():
            row = self.s_table.rowCount()
            self.s_table.insertRow(row)
            self.s_table.setItem(row, 0, _cell(os.path.basename(key) if key else name, tip=key or name))
            self.s_table.setItem(row, 1, _cell(f'{e * 1000:,.1f} mm'.replace(',', ' '), num=True))
            over = e > MESH_MAX_EXTENT
            st = _cell('too large: millimetres?' if over else 'plausible')
            st.setIcon(icons.status_icon('critical' if over else 'good', 12))
            self.s_table.setItem(row, 2, st)
        big = float(ext.max())
        if bad:
            name, key, e = bad[0]
            where = 'on its <mesh> in the URDF' if self.robot.source.kind == 'urdf' else 'on its <mesh> asset'
            self.s_status.set('critical', f'The mesh {os.path.basename(key) if key else name} is {e:,.0f} m across, '
                                          'which looks like millimetres rather than metres: set scale="0.001 0.001 '
                                          f'0.001" {where}, then import again. Import is off until then.')
        elif big < SIZE_RANGE[0]:
            self.s_status.set('warning', f'The robot is only {big * 1000:.1f} mm across. If it should be larger, its '
                                         'meshes may be scaled twice; check the scale attributes in the file.')
        elif big > SIZE_RANGE[1]:
            self.s_status.set('warning', f'The robot is {big:.1f} m across, more than most robots. Check that its '
                                         'meshes are in metres.')
        else:
            self.s_status.set('good', f'Looks like metres: the robot is {big * 1000:.0f} mm across at home '
                                      f'(a check against {SIZE_RANGE[0] * 1000:.0f} mm to {SIZE_RANGE[1]:.0f} m, a '
                                      'plausibility default).')

    def _fill_joints(self):
        r, m = self.robot, self.model
        t = self.j_table
        t.blockSignals(True)
        t.setRowCount(0)
        if r is None or m is None:
            t.blockSignals(False)
            return
        moving = set(m.joint_names)
        for j in r.joints:
            row = t.rowCount()
            t.insertRow(row)
            jt = j.get('type', '')
            t.setItem(row, 0, _cell(j.get('name', ''), tip=f'{j.get("parent", "")} → {j.get("child", "")}'))
            t.setItem(row, 1, _cell(jt, muted=jt == 'fixed'))
            for k, key in ((2, 'lower'), (3, 'upper')):
                v = j.get(key)
                txt = JF.fmt(v, jt) if v is not None else ('none' if jt == 'continuous' else '—')
                t.setItem(row, k, _cell(txt, num=True, tip=JF.NOT_STATED if v is None and jt not in
                                        ('fixed', 'continuous') else ''))
            t.setItem(row, 4, _cell(JF.fmt_rate(j.get('velocity'), jt), num=True,
                                    tip=JF.NOT_STATED if j.get('velocity') is None else ''))
            t.setItem(row, 5, _cell(JF.fmt_effort(j.get('effort'), jt), num=True,
                                    tip=JF.NOT_STATED if j.get('effort') is None else ''))
            g = QTableWidgetItem('')
            if j.get('name') in moving:
                g.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable)
                g.setCheckState(Qt.CheckState.Checked if j['name'] in self.gripper_joints else Qt.CheckState.Unchecked)
                g.setToolTip('Tick if this joint opens and closes the gripper')
            else:
                g.setFlags(Qt.ItemFlag.ItemIsEnabled)
            t.setItem(row, 6, g)
        t.blockSignals(False)
        self.j_note.setText(f'{len(r.joints)} joints, {m.nq} of them moving. Angles in degrees, lengths in mm, as '
                            'the file states them (— where it states none). Ticked gripper joints: '
                            f'{", ".join(self.gripper_joints) or "none"}.')

    def _joint_ticked(self, it):
        if it.column() != 6:
            return
        name = self.j_table.item(it.row(), 0).text()
        on = it.checkState() == Qt.CheckState.Checked
        names = [n for n in self.gripper_joints if n != name] + ([name] if on else [])
        self.set_gripper_joints(names)

    def selected_joint(self):
        if self.robot is None or self.step != 3:
            return None
        rows = {i.row() for i in self.j_table.selectedItems()}
        if not rows:
            return None
        k = min(rows)
        return self.robot.joints[k] if k < len(self.robot.joints) else None

    def _fill_tags(self):
        m = self.model
        if m is None:
            return
        links = m.link_names
        self.t_frame.clear()
        self.t_frame.addItem('(none)', '')
        for n in links:
            self.t_frame.addItem(n, n)
        self.t_frame.setCurrentIndex(max(0, self.t_frame.findData(self.gripper_frame)))
        self.t_pad_link.clear()
        for n in links:
            self.t_pad_link.addItem(n, n)
        self.t_tips.blockSignals(True)
        self.t_tips.clear()
        for n in links:
            it = QListWidgetItem(n)
            it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if n in self.fingertips else Qt.CheckState.Unchecked)
            self.t_tips.addItem(it)
        self.t_tips.blockSignals(False)
        self._fill_pads()

    def _fill_pads(self, select=None):
        t = self.t_pads
        cur = self.pad_row() if select is None else select
        t.blockSignals(True)
        t.setRowCount(0)
        for p in self.pads:
            row = t.rowCount()
            t.insertRow(row)
            c = [v * 1000 for v in p['box']['center']]
            s = [v * 1000 for v in p['box']['size']]
            t.setItem(row, 0, _cell(p['name']))
            t.setItem(row, 1, _cell(p['link']))
            t.setItem(row, 2, _cell(' '.join(f'{v:+.1f}' for v in c).replace('-', '−'), num=True))
            t.setItem(row, 3, _cell(' × '.join(f'{v:.1f}' for v in s), num=True))
        t.blockSignals(False)
        if 0 <= cur < len(self.pads):
            t.selectRow(cur)
        self._pad_picked()

    def pad_row(self) -> int:
        rows = {i.row() for i in self.t_pads.selectedItems()}
        return min(rows) if rows else -1

    def _pad_picked(self):
        i = self.pad_row()
        on = 0 <= i < len(self.pads)
        self.pad_editor.setEnabled(on)
        if on:
            p = self.pads[i]
            self.p_name.setText(p['name'])
            for b, v in zip(self.p_c, p['box']['center']):
                b.set_value(v * 1000)
            for b, v in zip(self.p_s, p['box']['size']):
                b.set_value(v * 1000)
        self._repaint()

    def _pad_edited(self, which):
        i = self.pad_row()
        if not 0 <= i < len(self.pads):
            return
        if which == 'Centre':
            self.update_pad(i, center=[b.value() / 1000 for b in self.p_c])
        else:
            self.update_pad(i, size=[max(1e-4, b.value() / 1000) for b in self.p_s])

    def _tip_ticked(self, it):
        names = [self.t_tips.item(k).text() for k in range(self.t_tips.count())
                 if self.t_tips.item(k).checkState() == Qt.CheckState.Checked]
        self.fingertips = names
        self._repaint()

    def _fill_summary(self):
        while self.sum_grid.count():
            w = self.sum_grid.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        r, m = self.robot, self.model
        if r is None or m is None:
            self.sum_status.set('critical', (self.res or {}).get('error') or 'Read a robot file first.')
            return
        primary = not self.ctx.store.project.robots
        rows = [('Name', self.robot_name()), ('File', f'{os.path.basename(self.path)} ({r.source.kind.upper()})'),
                ('Links', str(len(m.link_names))), ('Moving joints', str(m.nq)),
                ('Gripper joints', ', '.join(self.gripper_joints) or 'none'),
                ('Gripper frame', self.gripper_frame or 'none'),
                ('Fingertip frames', ', '.join(self.fingertips) or 'none'),
                ('Contact pads', ', '.join(f'{p["name"]} on {p["link"]}' for p in self.pads) or 'none'),
                ('Role', 'primary robot: its base is the world frame' if primary else 'a further robot at the world '
                                                                                      'origin (move its base later)')]
        for k, (a, b) in enumerate(rows):
            self.sum_grid.addWidget(label(a, ink2=True), k, 0)
            v = label(b, mono_=a in ('Links', 'Moving joints'), wrap=True)
            self.sum_grid.addWidget(v, k, 1)
        self.sum_grid.setColumnStretch(1, 1)
        if self.units_problem():
            self.sum_status.set('critical', 'A mesh looks like millimetres (step 3): fix its scale in the file, then '
                                            'import again.')
        elif not self.gripper_frame:
            self.sum_status.set('warning', 'No gripper frame: IK drag uses the last link and the grasp studies need '
                                           'one. Pick it in step 5, or import now and tag it later.')
        else:
            self.sum_status.set('good', 'Ready to import.')
        notes = list(m.notes) + ([r.source.note] if r.source.note else [])
        self.sum_notes.setText(' '.join(notes))

    # ------------------------------------------------------------------ tagging (programmatic too)
    def robot_name(self) -> str:
        t = self.name_edit.text().strip()
        return t or (self.robot.name if self.robot is not None else '')

    def set_gripper_joints(self, names):
        m = self.model
        moving = set(m.joint_names) if m is not None else set()
        bad = [n for n in names if n not in moving]
        if bad:
            raise KeyError(f'{bad[0]} is not a moving joint of this robot.')
        self.gripper_joints = list(dict.fromkeys(names))
        if self.step == 3:
            self._fill_joints()
        self._repaint()

    def set_gripper_frame(self, link):
        m = self.model
        if link and (m is None or link not in m.link_names):
            raise KeyError(f'{link} is not a link of this robot.')
        self.gripper_frame = link or ''
        i = self.t_frame.findData(self.gripper_frame)
        if i >= 0:
            self.t_frame.setCurrentIndex(i)
        self._repaint()

    def set_fingertips(self, links):
        m = self.model
        bad = [n for n in links if m is None or n not in m.link_names]
        if bad:
            raise KeyError(f'{bad[0]} is not a link of this robot.')
        self.fingertips = list(dict.fromkeys(links))
        if self.step == 4:
            self._fill_tags()
        self._repaint()

    def link_box(self, link):
        """(center, size) in metres of the link's visual meshes in the link frame; a placeholder cube without one."""
        m = self.model
        V = [v for ln, v, _f, _c in (m.visual_meshes() if m is not None else []) if ln == link]
        if not V:
            return [0.0, 0.0, 0.0], [PAD_PLACEHOLDER] * 3
        V = np.concatenate(V)
        lo, hi = V.min(axis=0).astype(float), V.max(axis=0).astype(float)
        return [float(x) for x in (lo + hi) / 2], [float(max(x, 1e-4)) for x in hi - lo]

    def add_pad(self, link, name=None, center=None, size=None) -> int:
        m = self.model
        if m is None or link not in m.link_names:
            raise KeyError(f'{link} is not a link of this robot.')
        c0, s0 = self.link_box(link)
        names = {p['name'] for p in self.pads}
        k = len(self.pads) + 1
        while name is None or name in names:
            name = f'Pad {k}'
            k += 1
        self.pads.append({'link': link, 'name': name,
                          'box': {'center': [float(v) for v in (center if center is not None else c0)],
                                  'size': [float(v) for v in (size if size is not None else s0)]}})
        if self.step == 4:
            self._fill_pads(select=len(self.pads) - 1)
        self._repaint()
        return len(self.pads) - 1

    def update_pad(self, i, name=None, center=None, size=None):
        if not 0 <= i < len(self.pads):
            return
        p = self.pads[i]
        if name is not None and name.strip():
            p['name'] = name.strip()
        if center is not None:
            p['box']['center'] = [float(v) for v in center]
        if size is not None:
            p['box']['size'] = [float(max(v, 1e-4)) for v in size]
        if self.step == 4:
            self._fill_pads(select=i)
        self._repaint()

    def remove_pad(self, i):
        if 0 <= i < len(self.pads):
            del self.pads[i]
            if self.step == 4:
                self._fill_pads(select=min(i, len(self.pads) - 1))
            self._repaint()

    # ------------------------------------------------------------------ import
    def tagged_robot(self):
        """A copy of the read robot with the wizard's name and tags, ready to add to the project."""
        r = P.Robot.from_dict(self.robot.to_dict())
        r.name = self.robot_name()
        r.primary = False
        for j in r.joints:
            j.pop('gripper', None)
            if j.get('name') in self.gripper_joints:
                j['gripper'] = True
        r.gripper_frame = self.gripper_frame
        r.fingertip_frames = list(self.fingertips)
        r.contact_pads = [{'link': p['link'], 'name': p['name'],
                           'box': {'center': list(p['box']['center']), 'size': list(p['box']['size'])}}
                          for p in self.pads]
        return r

    def finish(self) -> bool:
        """Add the robot and its files to the project as one undo step; closes the wizard."""
        if not self.can_import():
            return False
        robot = self.tagged_robot()
        sources = dict(self.res['sources'])
        st = self.ctx.store

        def fn(p):
            robot.primary = not p.robots          # the first robot is the primary one: its base is the world frame
            p.robots.append(robot)
            p.sources.update(sources)
        st.edit(['robots', 'sources'], fn, f'Imported robot {robot.name}')
        st.select(('robot', robot.uid))
        self.imported = robot
        win = getattr(self.ctx, 'window', None)
        if win is not None:
            win.frame_all()
            nj = len(st.q.get(robot.uid, ()))
            win.msg(f'Imported {robot.name}: {len(robot.links)} links, {nj} moving joints'
                    + (f', gripper frame {robot.gripper_frame}' if robot.gripper_frame else ''))
        self.accept()
        return True

    def done(self, r):
        self.release()
        super().done(r)
