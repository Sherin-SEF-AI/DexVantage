"""The Tasks page (spec §13, §7.3): the project's grasp sets, made by sampling the task volume, imported from a file,
taken from a trajectory or from a recorded dataset, listed with their counts, deletable, and drawn in the 3D view as
small gripper glyphs.

What it shows: the number of grasps in the project (the headline, "ready" when a set exists for a robot in the project),
a form for a sampled set (robot, kinds, count, seed, task volume, handover zone), Import file…, From a dataset… (the
grasps where the gripper closes in the episodes of a LeRobot dataset, ROS 2 bag or RLDS dataset, read on a worker
thread; errors said in words), a trajectory picker taking the grasps where the gripper closes or every N samples, and
the table of sets (name, source, robot, poses, kinds). Every change is one undo step. The selected set (or the preview
of the form's set) is drawn as glyphs: a stem along −approach, a palm across the closing axis and two fingers ending at
the grasp centre, coloured by kind; at most MAX_GLYPHS are drawn (evenly picked), and the Camera view shows the first
MAX_CAM_GLYPHS of them. Grasp sets are written only through core/grasps.py. "Swept volume" draws the robot see-through
along the chosen trajectory (core/robot/swept.py; the first time per robot its meshes are simplified as a background
job).
"""
from __future__ import annotations

import json
import os
import traceback

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QGridLayout, QHeaderView, QMessageBox, QTreeWidget,
                             QTreeWidgetItem, QWidget)

from ...core import grasps as GR
from ..theme import THEME, mono
from ..widgets import ScrubSpinBox, SegmentedControl, button, label
from . import register
from .base import StudyPage, row

MAX_GLYPHS = 1500
MAX_CAM_GLYPHS = 200
STEM_M = 0.03
FINGER_M = 0.02
KIND_SLOT = {'top_down': 0, 'side': 1, 'angled': 2, 'handoff': 3, 'imported': 4, 'trajectory': 5}
DEFAULT_N = 200
DEFAULT_SEED = 0
ASYNC_MIN = 1000                 # the Preview and Add set buttons sample sets this large on a worker thread


def glyph_segments(gs: dict, geom: dict | None, limit: int = MAX_GLYPHS) -> dict:
    """kind -> (S, 2, 3) world segments of the gripper glyphs of a grasp set's poses (at most `limit`, evenly
    picked). geom is the robot's gripper geometry (core.grasps.gripper_geometry) or None (TCP at the frame origin,
    closing along its x axis)."""
    poses = gs.get('poses') or []
    if not poses:
        return {}
    idx = np.unique(np.linspace(0, len(poses) - 1, min(limit, len(poses))).round().astype(int))
    tcp_l = np.asarray(geom['tcp'], float) if geom else np.zeros(3)
    c_l = np.asarray(geom['closing'], float) if geom else np.array([1.0, 0.0, 0.0])
    w0 = (geom or {}).get('width_m') or 0.04
    out: dict = {}
    for i in idx:
        g = poses[i]
        T = GR.grasp_matrix(g)
        R = T[:3, :3]
        tcp = R @ tcp_l + T[:3, 3]
        a = np.asarray(g.get('approach'), float)
        c = R @ c_l
        w = float(np.clip(g.get('width_m') or w0, 0.01, 0.12)) / 2
        palm = tcp - FINGER_M * a
        segs = [(palm + w * c, palm - w * c), (palm + w * c, tcp + w * c), (palm - w * c, tcp - w * c),
                (palm, palm - STEM_M * a)]
        out.setdefault(g.get('kind', 'imported'), []).extend(segs)
    return {k: np.asarray(v, np.float32) for k, v in out.items()}


class TasksPage(StudyPage):
    """Tasks: the grasp sets the grasp studies use. A sampled set places the gripper's grasp centre (between its
    contact pads) on points of the task objects, or on the support surface of a task volume, with the approach of
    the kind chosen (top-down pinch, side, angled, bimanual handoff in a handover zone); the same seed gives the same
    set. Sets can also be imported from CSV or JSON or taken from a trajectory."""
    STUDY = None
    TITLE = 'Tasks'
    HEADLINE = 'Grasps in the project'

    # ---------------------------------------------------------------- body
    def build_body(self, layout):
        self.preview_set = None
        self._overlay = []
        self._geom_cache = {}
        self.b_run.setVisible(False)
        self.b_apply.setText('Add set')
        self.b_apply.setToolTip('Add the sampled set below to the project (one undo step)')
        self.b_preview.setToolTip('Draw the sampled set below in the 3D view without adding it')
        self.b_export.setToolTip('The selected grasp set as JSON with provenance')
        for b, fn in ((self.b_apply, lambda: self.apply(background=True)),
                      (self.b_preview, lambda: self.preview(background=True))):
            b.clicked.disconnect()                      # the buttons sample large sets off the UI thread
            b.clicked.connect(fn)
        self._glyphs = (None, None)

        layout.addWidget(label('Sample a grasp set', bold=True))
        g = QWidget()
        grid = QGridLayout(g)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        self.robot_combo = QComboBox()
        self.robot_combo.setToolTip('The robot whose gripper frame the poses are for')
        self.volume_combo = QComboBox()
        self.volume_combo.setToolTip('The task volume the grasp points are drawn in (on its objects when it holds any, '
                                     'else on its support surface)')
        self.handover_combo = QComboBox()
        self.handover_combo.setToolTip('The handover zone of bimanual handoff grasps (a task volume of kind '
                                       '"handover")')
        self.count = ScrubSpinBox('', 'plain', 0, '', step=10.0, width=90)
        self.count.setRange(1, GR.MAX_GRASPS)
        self.count.set_value(DEFAULT_N)
        self.count.setToolTip(f'How many grasps (1 to {GR.MAX_GRASPS:,}), shared round-robin over the kinds'
                              .replace(',', ' '))
        self.seed = ScrubSpinBox('', 'plain', 0, '', step=1.0, width=90)
        self.seed.setRange(0, 2 ** 31 - 1)
        self.seed.set_value(DEFAULT_SEED)
        self.seed.setToolTip('The random seed: the same seed and settings give the same set')
        self.kind_box = {}
        kinds_w = QWidget()
        kl = QGridLayout(kinds_w)
        kl.setContentsMargins(0, 0, 0, 0)
        for i, k in enumerate(GR.KINDS):
            cb = QCheckBox(GR.KIND_TITLES[k])
            cb.setChecked(k == 'top_down')
            cb.toggled.connect(lambda _on: self._form_changed())
            self.kind_box[k] = cb
            kl.addWidget(cb, i // 2, i % 2)
        grid.addWidget(label('Robot'), 0, 0)
        grid.addWidget(self.robot_combo, 0, 1)
        grid.addWidget(label('Task volume'), 0, 2)
        grid.addWidget(self.volume_combo, 0, 3)
        grid.addWidget(label('Kinds'), 1, 0, Qt.AlignmentFlag.AlignTop)
        grid.addWidget(kinds_w, 1, 1)
        grid.addWidget(label('Handover zone'), 1, 2)
        grid.addWidget(self.handover_combo, 1, 3)
        grid.addWidget(label('Grasps'), 2, 0)
        grid.addWidget(self.count, 2, 1)
        grid.addWidget(label('Seed'), 2, 2)
        grid.addWidget(self.seed, 2, 3)
        layout.addWidget(g)
        self.form_note = label('', muted=True, wrap=True)
        layout.addWidget(self.form_note)
        for w in (self.robot_combo, self.volume_combo, self.handover_combo):
            w.activated.connect(lambda _i: self._form_changed())
        self.count.committed.connect(lambda _v: self._form_changed())
        self.seed.committed.connect(lambda _v: self._form_changed())

        layout.addWidget(label('Import or take from a trajectory', bold=True))
        self.b_import = button('Import file…', icon_name='import', tip='A CSV or JSON file of gripper poses (columns '
                               'id, x, y, z, qx, qy, qz, qw, optionally ax, ay, az, kind, object)')
        self.b_import.clicked.connect(lambda: self.import_path())
        self.b_dataset = button('From a dataset…', icon_name='import',
                                tip='The grasps in a recorded dataset (LeRobot v2.1 or v3.0, a ROS 2 bag, RLDS): one '
                                    'each time the gripper closes in an episode\'s joint states, for the form\'s '
                                    'robot (one undo step)')
        self.b_dataset.clicked.connect(lambda: self.add_from_dataset(background=True))
        self.traj_combo = QComboBox()
        self.traj_combo.setToolTip('A joint trajectory of the project: the grasps in it, or the gripper frame at its '
                                   'samples')
        self.traj_mode = SegmentedControl([('closures', 'Where it closes'), ('every', 'Every N samples')], 'closures')
        self.traj_mode.setToolTip('Where it closes: one grasp each time the gripper closes by 5 mm or more (a '
                                  'default), at the moment it starts closing, approaching along the path it came in '
                                  'on. Every N samples: the gripper frame at every Nth sample.')
        self.every = ScrubSpinBox('', 'plain', 0, '', step=1.0, width=70)
        self.every.setRange(1, 100000)
        self.every.set_value(10)
        self.every.setToolTip('Take every Nth sample of the trajectory')
        self.every.setEnabled(False)
        self.traj_mode.changed.connect(lambda k: self.every.setEnabled(k == 'every'))
        self.b_traj = button('Add from trajectory', tip='Add a set of the grasps in the trajectory, or of the gripper '
                             'poses along it (one undo step)')
        self.b_traj.clicked.connect(lambda: self.add_from_trajectory())
        self.swept = QCheckBox('Swept volume')
        self.swept.setToolTip('Draw the robot see-through at 8 evenly spaced times of the trajectory (a default), '
                              'fainter early and stronger late, while this page is shown')
        self.swept.toggled.connect(lambda _on: self.show_swept())
        self.traj_combo.currentIndexChanged.connect(lambda _i: self.show_swept() if self.swept.isChecked() else None)
        self._ghost = []
        layout.addWidget(row(self.b_import, self.b_dataset))
        layout.addWidget(row(label('Trajectory'), self.traj_combo, self.traj_mode, self.every, self.b_traj))
        layout.addWidget(row(self.swept))

        layout.addWidget(label('Grasp sets', bold=True))
        self.tree = QTreeWidget()
        self.tree.setRootIsDecorated(False)
        self.tree.setHeaderLabels(['Grasp set', 'Source', 'Robot', 'Poses', 'Kinds'])
        h = self.tree.header()
        h.setStretchLastSection(True)
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for k in range(1, 4):
            h.setSectionResizeMode(k, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.itemSelectionChanged.connect(self._selection_changed)
        layout.addWidget(self.tree)
        self.b_delete = button('Delete set', role='ghost', icon_name='trash', tip='Delete the selected grasp set (one '
                               'undo step)')
        self.b_delete.clicked.connect(lambda: self.delete_set())
        self.show3d = QCheckBox('Show in 3D')
        self.show3d.setChecked(True)
        self.show3d.setToolTip('Draw the selected set (or the preview) as gripper glyphs while this page is shown')
        self.show3d.toggled.connect(lambda _on: self._rebuild_overlay())
        layout.addWidget(row(self.b_delete, self.show3d))
        self.set_note = label('', muted=True, wrap=True)
        layout.addWidget(self.set_note)
        self.msg = label('', wrap=True)
        layout.addWidget(self.msg)
        st = self.ctx.store
        st.changed.connect(lambda s: self.refresh() if set(s) & {'tasks', 'robots', 'workspace', 'sources'} else None)
        st.project_replaced.connect(self._replaced)
        self.refresh()

    # ---------------------------------------------------------------- state
    def _replaced(self):
        self.preview_set = None
        self._geom_cache = {}
        self.refresh()

    def _say(self, text: str, sev: str = ''):
        self.msg.setText(text)
        self.msg.setStyleSheet(f'color: {THEME.color(sev).name()};' if sev else '')
        if self.ctx.window is not None and text:
            self.ctx.window.msg(text, 6000)

    def _fill(self, combo, items, keep=True):
        cur = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for text, data in items:
            combo.addItem(text, data)
        k = combo.findData(cur) if keep else -1
        combo.setCurrentIndex(k if k >= 0 else (0 if items else -1))
        combo.blockSignals(False)

    def refresh(self):
        p = self.ctx.store.project
        self._fill(self.robot_combo, [(r.name, r.uid) for r in p.robots])
        vols = p.workspace.volumes
        self._fill(self.volume_combo, [(v.name, v.uid) for v in vols if v.kind != 'handover'] +
                   [(f'{v.name} (handover)', v.uid) for v in vols if v.kind == 'handover'])
        self._fill(self.handover_combo, [(v.name, v.uid) for v in vols if v.kind == 'handover'] or
                   [('none: add a handover volume', '')])
        self._fill(self.traj_combo, [(t.get('name', 'Trajectory'), t.get('uid')) for t in p.tasks.trajectories
                                     if isinstance(t, dict)] or [('none in the project', '')])
        self.b_traj.setEnabled(bool(self.traj_combo.currentData()))
        self.swept.setEnabled(bool(self.traj_combo.currentData()))
        if self.swept.isChecked() or self._ghost:
            self.show_swept()                        # the trajectory or robot may have changed (posing is cheap)
        sel = self.selected_uid()
        self.tree.blockSignals(True)
        self.tree.clear()
        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        names = {r.uid: r.name for r in p.robots}
        for gs in GR.sets(p):
            c = GR.counts(gs)
            kinds = ', '.join(f'{GR.KIND_TITLES.get(k, k).lower()} {v}' for k, v in c.items())
            it = QTreeWidgetItem([gs['name'], gs.get('source', ''), names.get(gs.get('robot_uid'), '—'),
                                  f'{len(gs.get("poses") or []):,}'.replace(',', ' '), kinds])
            it.setData(0, Qt.ItemDataRole.UserRole, gs['uid'])
            it.setFont(3, mono(11))
            it.setTextAlignment(3, right)
            if gs.get('robot_uid') and gs['robot_uid'] not in names:
                it.setToolTip(2, 'The robot this set was made for is no longer in the project')
            it.setToolTip(0, f'Frame {gs.get("frame") or "not set"}')
            self.tree.addTopLevelItem(it)
            if gs['uid'] == sel:
                it.setSelected(True)
                self.tree.setCurrentItem(it)
        self.tree.blockSignals(False)
        n = self.tree.topLevelItemCount()
        rh = self.tree.sizeHintForRow(0) if n else 20
        self.tree.setFixedHeight(self.tree.header().sizeHint().height() + max(2, min(n, 10)) * max(rh, 18) + 6)
        self.b_delete.setEnabled(self.selected_uid() is not None)
        self._headline()
        self._form_changed(clear_preview=False)
        self._selection_changed()

    def _headline(self):
        p = self.ctx.store.project
        ss = GR.sets(p)
        total = sum(len(g.get('poses') or []) for g in ss)
        uids = {r.uid for r in p.robots}
        usable = [g for g in ss if g.get('robot_uid') in uids]
        if usable:
            self.head.set(f'{total:,}'.replace(',', ' '), f'grasps in {len(ss)} set{"s" if len(ss) != 1 else ""}',
                          'good', 'ready', 'The grasp studies use these sets. Add a sampled set, import one or take '
                          'one from a trajectory to cover more of the task.')
        elif ss:
            self.head.set(f'{total:,}'.replace(',', ' '), f'grasps in {len(ss)} set{"s" if len(ss) != 1 else ""}',
                          'warning', 'no robot', 'No set names a robot of the project: the grasp studies need one. '
                          'Sample a set for a robot, or import the file again with a robot chosen.')
        else:
            self.head.set('0', 'grasps', 'warning', 'none yet', 'The grasp studies need a grasp set: choose the kinds '
                          'and the count below and press Add set (Preview draws it first).')

    def empty(self):
        """No study result: the headline counts the project's grasps instead (refresh() sets it)."""

    def selected_uid(self) -> str | None:
        it = self.tree.currentItem() if hasattr(self, 'tree') else None
        if it is None or not it.isSelected():
            return None
        return it.data(0, Qt.ItemDataRole.UserRole)

    def select_set(self, uid: str):
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            if it.data(0, Qt.ItemDataRole.UserRole) == uid:
                self.tree.setCurrentItem(it)
                it.setSelected(True)
                return

    # ---------------------------------------------------------------- the sampled set
    def form(self) -> dict:
        """The sampling form's settings (core.grasps.sample keyword arguments)."""
        return {'robot_uid': self.robot_combo.currentData() or '',
                'kinds': [k for k, cb in self.kind_box.items() if cb.isChecked()],
                'n': int(round(self.count.value())), 'rng_seed': int(round(self.seed.value())),
                'volume_uid': self.volume_combo.currentData() or None,
                'handover_uid': self.handover_combo.currentData() or None}

    def set_form(self, robot_uid=None, kinds=None, n=None, rng_seed=None, volume_uid=None, handover_uid=None):
        """Fill the form (scripts and tests)."""
        if robot_uid is not None:
            self.robot_combo.setCurrentIndex(max(0, self.robot_combo.findData(robot_uid)))
        if kinds is not None:
            for k, cb in self.kind_box.items():
                cb.blockSignals(True)
                cb.setChecked(k in kinds)
                cb.blockSignals(False)
        if n is not None:
            self.count.set_value(float(n))
        if rng_seed is not None:
            self.seed.set_value(float(rng_seed))
        if volume_uid is not None:
            self.volume_combo.setCurrentIndex(max(0, self.volume_combo.findData(volume_uid)))
        if handover_uid is not None:
            self.handover_combo.setCurrentIndex(max(0, self.handover_combo.findData(handover_uid)))
        self._form_changed()

    def _form_changed(self, clear_preview=True):
        if clear_preview and self.preview_set is not None:
            self.preview_set = None
            self._rebuild_overlay()
        f = self.form()
        p = self.ctx.store.project
        if not p.robots:
            why = 'Import a robot first (Robot ▸ Import robot…): grasps are posed for a robot\'s gripper.'
        elif not f['kinds']:
            why = 'Tick at least one kind of grasp.'
        elif 'handoff' in f['kinds'] and not f['handover_uid']:
            why = 'Bimanual handoff needs a handover zone: add a task volume of kind "handover".'
        elif any(k != 'handoff' for k in f['kinds']) and not f['volume_uid']:
            why = 'Add a task volume to sample grasps in (Workspace ▸ Add task volume).'
        else:
            why = ''
            self._warm(f['robot_uid'])
        self.form_note.setText(why or 'Defaults (labelled, not standards): 30 mm above the support surface, angled '
                               'grasps 45° from vertical, side and angled headings within ±60° of the line from the '
                               'robot\'s base. Explain says more.')
        self.suggestion_ready(not why)

    def _warm(self, robot_uid: str):
        """Build the robot's kinematic model and its gripper geometry (pads by ray casting) on a worker thread once
        Apply is offered, so Apply's sampling finds them in the shared caches: cold, they took about 200 ms on the UI
        thread on ALOHA 2 (a MuJoCo compile and the pad ray casts), the sampling itself under 2 ms."""
        import threading
        p = self.ctx.store.project
        r = p.robot(robot_uid) if robot_uid else None
        if r is None:
            return
        key = (robot_uid, id(p), len(p.robots))           # cheap: once per robot and project (an edited robot is
        #                                                    built by the next Apply, as before)
        if key == getattr(self, '_warmed', None):
            return
        self._warmed = key
        snap = p.snapshot()

        def work():
            try:
                GR.gripper_geometry(snap, robot_uid)
            except Exception:                            # noqa: BLE001 — a warm-up only; Apply says what fails
                pass
        threading.Thread(target=work, name='tasks-warm', daemon=True).start()

    def make_sampled(self) -> dict | None:
        f = self.form()
        try:
            return GR.sample(self.ctx.store.project, **f)
        except GR.GraspError as e:
            self._say(str(e), 'warning')
        except Exception as e:                           # noqa: BLE001 — explained, the traceback logged
            traceback.print_exc()
            self._say(f'The grasps could not be sampled ({type(e).__name__}). Details are in the log.', 'critical')
        return None

    def _sample_in_background(self, then) -> bool:
        """Sample the form's set on a worker thread (a window job) and call then(gs) here when it is done; False when
        the set is small enough to sample here or there is no window (scripts and tests sample synchronously)."""
        f = self.form()
        win = self.ctx.window
        if f['n'] < ASYNC_MIN or win is None or not hasattr(win, 'run_job'):
            return False
        snap = self.ctx.store.project.snapshot()

        def work(_progress=None, _cancel=None):
            try:
                return {'gs': GR.sample(snap, **f)}
            except GR.GraspError as e:
                return {'error': str(e)}

        def done(res):
            if 'error' in res:
                self._say(res['error'], 'warning')
            elif self.form() != f:                      # the form changed while it sampled: that set is not wanted
                self._say('The form changed while the grasps were sampled: press the button again.', 'warning')
            else:
                then(res['gs'])

        def failed(_message):                           # the window logged the traceback
            self._say('The grasps could not be sampled (an unexpected error). Details are in the log.', 'critical')
        if win.run_job('grasp_sample', f'Sampling {f["n"]:,} grasps'.replace(',', ' '), work, done, failed):
            self._say(f'Sampling {f["n"]:,} grasps…'.replace(',', ' '))
        else:
            self._say('Grasps are already being sampled: wait for them to finish.', 'warning')
        return True

    def preview(self, background: bool = False):
        """Draw the form's sampled set in the 3D view without adding it (background: the Preview button)."""
        if background and self._sample_in_background(self._show_preview):
            return
        gs = self.make_sampled()
        if gs is not None:
            self._show_preview(gs)

    def _show_preview(self, gs):
        self.preview_set = gs
        notes = gs['settings'].get('notes') or []
        self._say(f'Preview: {len(gs["poses"])} grasps, not added yet. ' + ' '.join(notes))
        self._rebuild_overlay()

    def apply(self, background: bool = False):
        """Add the form's sampled set to the project (one undo step); its uid (None when sampled in the background:
        background is the Add set button)."""
        if self.preview_set is None and background and self._sample_in_background(self._add_sampled):
            return None
        gs = self.preview_set if self.preview_set is not None else self.make_sampled()
        if gs is None:
            return None
        return self._add_sampled(gs)

    def _add_sampled(self, gs) -> str:
        self.preview_set = None
        uid = self._add(gs, f'Add grasp set {gs["name"]}')
        self._say(f'Added {gs["name"]}. ' + ' '.join(gs['settings'].get('notes') or []))
        return uid

    def _add(self, gs, text) -> str:
        # every set the page adds is normalise_set's own output (sample, import_file, from_trajectory): trusted
        uid = self.ctx.store.edit('tasks', lambda p: GR.add_set(p, gs, trusted=True), text)
        self.refresh()
        self.select_set(uid)
        return uid

    # ---------------------------------------------------------------- import, trajectory, delete
    def import_path(self, path: str | None = None) -> str | None:
        """Import a CSV or JSON grasp file for the form's robot (one undo step); returns the new set's uid."""
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, 'Import grasps', '', 'Grasp files (*.csv *.json *.txt);;All '
                                                                            'files (*)')
            if not path:
                return None
        robot_uid = self.robot_combo.currentData() or ''
        try:
            gs = GR.import_file(path, self.ctx.store.project if robot_uid else None, robot_uid)
        except GR.GraspError as e:
            self._say(f'Could not import {os.path.basename(path)}: {e}', 'warning')
            return None
        except Exception as e:                           # noqa: BLE001 — explained, the traceback logged
            traceback.print_exc()
            self._say(f'{os.path.basename(path)} could not be read ({type(e).__name__}). Details are in the log.',
                      'critical')
            return None
        uid = self._add(gs, f'Import grasp set {gs["name"]}')
        self._say(f'Imported {len(gs["poses"])} grasps from {os.path.basename(path)}. '
                  + ' '.join(gs['settings'].get('notes') or []))
        return uid

    def add_from_trajectory(self, traj_uid: str | None = None, every: int | None = None,
                            mode: str | None = None) -> str | None:
        """Add a set from a trajectory (one undo step): mode 'closures' (the grasps where the gripper closes) or
        'every' (every Nth sample); a call that gives `every` and no mode takes every Nth sample."""
        traj_uid = traj_uid or self.traj_combo.currentData()
        mode = mode or ('every' if every is not None else self.traj_mode.value() or 'closures')
        every = int(every if every is not None else round(self.every.value()))
        if not traj_uid:
            self._say('There is no trajectory in the project: import one first (Robot ▸ Import trajectory…).',
                      'warning')
            return None
        try:
            gs = GR.from_trajectory(self.ctx.store.project, traj_uid, every, mode=mode)
        except GR.GraspError as e:
            self._say(str(e), 'warning')
            return None
        uid = self._add(gs, f'Add grasp set {gs["name"]}')
        if mode == 'closures':
            self._say(f'Added {len(gs["poses"])} grasps: the gripper closes {len(gs["poses"])} times in the '
                      'trajectory. ' + ' '.join(gs['settings'].get('notes') or []))
        else:
            self._say(f'Added {len(gs["poses"])} gripper poses from the trajectory.')
        return uid

    def add_from_dataset(self, path: str | None = None, robot_uid: str | None = None, mode: str | None = None,
                         episodes=None, every: int | None = None, unit: str = 'rad',
                         background: bool = False) -> str | None:
        """Add a set from a recorded dataset (core/grasps.from_dataset; one undo step): mode 'closures' (default)
        or 'every'. Without a path the dialog asks (dataset, episodes, mode, unit). background: read on a worker
        thread (the button) and add the set when done; the call then returns None at once."""
        if path is None:
            from ..dialogs.dataset_source import DatasetSourceDialog
            d = DatasetSourceDialog(self, 'grasps')
            if not d.exec():
                return None
            try:
                v = d.values()
            except ValueError as e:
                self._say(str(e), 'warning')
                return None
            path, episodes, unit, mode = v['path'], v['episodes'], v['unit'], v['mode']
            every = v['every'] if mode == 'every' else 1
        robot_uid = robot_uid or self.robot_combo.currentData() or ''
        if not robot_uid:
            self._say('There is no robot to take the grasps for: import one first (Robot ▸ Import robot…).',
                      'warning')
            return None
        mode = mode or 'closures'
        every = int(every or 1)
        snap = self.ctx.store.project.snapshot() if background else self.ctx.store.project
        name = os.path.basename(os.path.normpath(path))

        def work(progress=None, cancel=None):
            try:
                return {'gs': GR.from_dataset(snap, path, robot_uid, mode, episodes, every, unit=unit,
                                              progress=progress, cancel=cancel)}
            except GR.GraspError as e:
                return {'error': str(e)}

        def done(res):
            if 'error' in res:
                self._say(f'No grasps from {name}: {res["error"]}', 'warning')
                return None
            gs = res['gs']
            uid = self._add(gs, f'Add grasp set {gs["name"]}')
            n_ep = len(gs['settings'].get('episodes') or [])
            what = (f'the gripper closes {len(gs["poses"])} times' if mode == 'closures' else
                    f'every {every} rows')
            self._say(f'Added {len(gs["poses"])} grasps from {n_ep} episode(s) of {name}: {what}. '
                      + ' '.join(gs['settings'].get('notes') or []))
            return uid

        def failed(_message):                           # the window logged the traceback
            self._say(f'{name} could not be read (an unexpected error). Details are in the log.', 'critical')
        win = self.ctx.window
        if background and win is not None and hasattr(win, 'run_job'):
            if win.run_job('grasp_dataset', f'Reading grasps from {name}', work, done, failed):
                self._say(f'Reading the episodes of {name}…')
            else:
                self._say('A dataset is already being read: wait for it to finish.', 'warning')
            return None
        try:
            res = work()
        except Exception:                               # noqa: BLE001 — explained, the traceback logged
            traceback.print_exc()
            failed('')
            return None
        return done(res)

    def delete_set(self, uid: str | None = None) -> bool:
        uid = uid or self.selected_uid()
        gs = GR.get_set(self.ctx.store.project, uid) if uid else None
        if gs is None:
            self._say('Select a grasp set to delete.', 'warning')
            return False
        self.ctx.store.edit('tasks', lambda p: GR.remove_set(p, uid), f'Delete grasp set {gs["name"]}')
        self.refresh()
        self._say(f'Deleted {gs["name"]} (Edit ▸ Undo brings it back).')
        return True

    # ---------------------------------------------------------------- 3D and Camera view
    def _geom(self, robot_uid):
        p = self.ctx.store.project
        r = p.robot(robot_uid) if robot_uid else None
        if r is None:
            return None
        key = (robot_uid, json.dumps(r.to_dict(), sort_keys=True, default=str))
        if key not in self._geom_cache:
            try:
                self._geom_cache[key] = GR.gripper_geometry(p, robot_uid)
            except Exception:                            # noqa: BLE001 — glyphs then use the frame origin
                traceback.print_exc()
                self._geom_cache[key] = None
        return self._geom_cache[key]

    def shown_set(self) -> dict | None:
        if self.preview_set is not None:
            return self.preview_set
        uid = self.selected_uid()
        return GR.get_set(self.ctx.store.project, uid) if uid else None

    def _selection_changed(self):
        uid = self.selected_uid()
        self.b_delete.setEnabled(uid is not None)
        gs = GR.get_set(self.ctx.store.project, uid) if uid else None
        if gs is not None:
            s = gs.get('settings') or {}
            bits = [f'{len(gs.get("poses") or [])} poses for {gs.get("frame") or "no frame"}']
            if gs.get('source') == 'sampled':
                bits.append(f'seed {s.get("rng_seed")}')
                if s.get('defaults'):
                    bits.append('defaults: ' + ', '.join(s['defaults']))
            if s.get('file'):
                bits.append(f'from {s["file"]}')
            g = s.get('gripper') or {}
            if g.get('rule'):
                bits.append('TCP and axes ' + g['rule'])
            self.set_note.setText('; '.join(bits) + '. ' + ' '.join(s.get('notes') or []))
        else:
            self.set_note.setText('')
        self._rebuild_overlay()

    def _rebuild_overlay(self):
        from ..viewport3d import LineDraw
        self._overlay = []
        gs = self.shown_set() if self.show3d.isChecked() else None
        if gs is not None:
            try:
                geom = self._geom(gs.get('robot_uid'))
                key = (id(gs), gs.get('uid'), len(gs.get('poses') or []), id(geom))
                if self._glyphs[0] != key:              # a selection change rebuilds three times: draw the glyphs once
                    self._glyphs = (key, glyph_segments(gs, geom))
                segs = self._glyphs[1]
                for kind, S in segs.items():
                    col = THEME.cam(KIND_SLOT.get(kind, 6))
                    self._overlay.append(LineDraw(S, col, 1.4, dash=self.preview_set is not None, segments=True))
            except Exception:                            # noqa: BLE001 — the overlay must not take the page down
                traceback.print_exc()
                self._overlay = []
        win = self.ctx.window
        if win is not None and hasattr(win, 'viewport'):
            win.viewport.update()
        cv = getattr(win, 'camview', None) if win is not None else None
        if cv is not None and hasattr(cv, 'view'):
            cv.view.refresh_overlays()

    def overlays(self, vp) -> list:
        return list(self._overlay) + list(self._ghost)

    # ---------------------------------------------------------------- swept volume
    def show_swept(self) -> bool:
        """Draw (or clear) the ghosted swept volume of the chosen trajectory (core/robot/swept.py). The first time
        for a robot its meshes are decimated as a background job; True when the ghosts are drawn now."""
        from ...core.robot import swept as SW
        from ...core.robot.model import RobotModelError, get_model
        self._ghost = []
        p = self.ctx.store.project
        uid = self.traj_combo.currentData()
        tr = next((t for t in p.tasks.trajectories if isinstance(t, dict) and t.get('uid') == uid), None)
        if not self.swept.isChecked() or tr is None or p.robot(tr.get('robot_uid', '')) is None:
            self._redraw()
            return False
        try:
            model = get_model(p, tr['robot_uid'])
        except RobotModelError as e:
            self._say(f'The swept volume cannot be drawn: {e}', 'warning')
            return False
        win = self.ctx.window
        if SW.is_cached(model) or win is None or not hasattr(win, 'run_job'):
            self._set_ghost(SW.swept_ghosts(model, tr))
            return True

        def work(_progress, _cancel):
            SW.ghost_mesh(model)
            return True

        def done(_r):
            if self.swept.isChecked() and self.traj_combo.currentData() == uid:
                self._set_ghost(SW.swept_ghosts(model, tr))

        def failed(_message):                           # the window printed the traceback
            self._say('The swept volume could not be drawn (an unexpected error; the details are in the log).',
                      'warning')
        win.run_job('swept_volume', 'Drawing the swept volume', work, done, failed)
        return False

    def _set_ghost(self, g: dict):
        from ..viewport3d import TriDraw
        n = max(len(g['t']), 1)
        c = THEME.cam(KIND_SLOT['trajectory'])
        alpha = 0.10 + 0.25 * g['ghost'].astype(np.float32) / max(n - 1, 1)
        cols = np.empty((len(g['tri']), 4), np.float32)
        cols[:, 0], cols[:, 1], cols[:, 2], cols[:, 3] = c.redF(), c.greenF(), c.blueF(), alpha
        self._ghost = [TriDraw(g['tri'], cols, depth=True)] if len(g['tri']) else []
        self._redraw()

    def _redraw(self):
        win = self.ctx.window
        if win is not None and hasattr(win, 'viewport'):
            win.viewport.update()

    def camview_overlays(self, cam_model, pose) -> list:
        """The shown set's glyphs (the first MAX_CAM_GLYPHS) projected into the camera's image, by kind."""
        from ...core import camrender as CR
        gs = self.shown_set() if self.show3d.isChecked() else None
        if gs is None:
            return []
        out = []
        for kind, S in glyph_segments(gs, self._geom(gs.get('robot_uid')), MAX_CAM_GLYPHS).items():
            polys = []
            for seg in S:
                polys += CR.visible_polylines(None, cam_model, pose, np.asarray(seg, float), step=0.005)
            if polys:
                out.append((polys, QColor(THEME.cam(KIND_SLOT.get(kind, 6))), GR.KIND_TITLES.get(kind, kind)))
        return out

    # ---------------------------------------------------------------- export and explain
    def export(self, path=None):
        """The selected grasp set as JSON with provenance."""
        from ...core import provenance as PV
        uid = self.selected_uid()
        gs = GR.get_set(self.ctx.store.project, uid) if uid else None
        if gs is None:
            if path is None:
                self._say('Select a grasp set to export.', 'warning')
            return None
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, 'Export grasp set', f'{gs["name"]}.json', 'JSON (*.json)')
            if not path:
                return None
        doc = PV.stamp({'grasp_set': gs}, self.ctx.store.project,
                       output={'option': None, 'name': os.path.basename(path), 'in_folder': False})
        try:
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(doc, fh, indent=1, ensure_ascii=False)
                fh.write('\n')
        except OSError as e:
            self._say(f'Could not write {os.path.basename(path)}: {e.strerror or e}.', 'warning')
            return None
        self._say(f'Exported {gs["name"]} to {os.path.basename(path)}.')
        return path

    def explain_text(self) -> str:
        return (
            'A grasp is the pose of the robot\'s gripper frame at contact and the direction it moves in (the '
            'approach). Sampled sets place the grasp centre (the mean of the fingertip contact pads, measured from the '
            'robot\'s model at home) on a point and turn the gripper to the kind\'s approach.\n\n'
            'Points: on the task objects inside the task volume (their bounding box: the top for a top-down pinch, the '
            'centre otherwise), or, with no objects, uniform over the volume\'s footprint, 30 mm above the highest '
            'table under the point (a default). Top-down pinch approaches straight down; side grasps horizontally, '
            'heading away from the robot\'s base within ±60° (a default); angled grasps 45° from vertical (a '
            'default); bimanual handoff grasps anywhere in a handover zone, towards the other robot within ±20° (a '
            'default). The seed fixes every draw.\n\n'
            'Imported files: CSV with id, x, y, z, qx, qy, qz, qw (metres, quaternion x, y, z, w) and optional ax, '
            'ay, az, kind, object, width_m, t; or a JSON list of objects with id, position and quaternion.\n\n'
            'From a trajectory, "Where it closes" finds the grasps: each time the opening between the contact pads '
            'falls by 5 mm or more (a default), one grasp at the moment it starts to close, its approach the '
            'direction the tool centre moved over its last 20 mm (a default). "Every N samples" takes the gripper '
            'frame at every Nth sample.\n\n'
            'From a dataset (LeRobot v2.1 or v3.0, a ROS 2 bag, RLDS): each episode\'s recorded joint states are '
            'read as a trajectory, matched to the robot\'s joints by name, and its grasps found the same way. Joints '
            'recorded in degrees need "deg" in the dialog.\n\n'
            'Known limits: sampling does not check reach or collisions (the grasp study\'s IK does and says why a '
            'grasp is infeasible); object grasps use the bounding box, not the shape.')

    def explain(self):
        box = QMessageBox(self)
        box.setWindowTitle('How grasp sets are made')
        box.setText(self.explain_text())
        box.setModal(False)
        box.show()


register('tasks', 'Tasks', TasksPage, menu='Tasks', icon='grasp', order=15)
