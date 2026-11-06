"""Main window: menus, toolbar, docks, status bar, workspaces and background jobs.

Rebuilt for robots on Vantage's ui/mainwindow.py patterns: the dock shell (Outliner and Library left, Inspector and
Camera view right, the Assistant in its own full-height column, Analysis along the bottom), actions registered by id
(so the command palette and the shortcuts sheet list them), saved and resettable workspaces, and background jobs with
progress and cancel in the status bar.

Commands whose engine arrives in a later phase are in the menus, disabled, with the phase in their tooltip, so the
shape of the product is visible without anything pretending to work.
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback

from PyQt6.QtCore import QBuffer, QByteArray, QIODevice, QObject, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup, QGuiApplication, QKeySequence
from PyQt6.QtWidgets import (QComboBox, QDockWidget, QFileDialog, QLabel, QMainWindow, QMessageBox, QProgressBar,
                             QStatusBar, QToolBar, QVBoxLayout, QWidget)

from ..core import project as P
from ..core.frames import Pose, quat_from_rpy
from . import icons
from .context import OVERLAYS, AppContext
from .panels import AnalysisPanel, AssistantPanel, CameraViewPanel, Inspector, LibraryPanel, Outliner, PoseScrubber
from .scene import SceneProvider
from .store import ProjectStore
from .theme import THEME, mono
from .viewport3d import make_viewport
from .widgets import DockTitle, tool_button


class JobSignals(QObject):
    progress = pyqtSignal(str, float, str)
    done = pyqtSignal(str, object)
    failed = pyqtSignal(str, str)


JOB_LABEL_W = 420
MAX_RECENT = 10

# workspace -> (docks shown, analysis tab, a line for the status bar)
WORKSPACES = {
    'placement': (('outliner', 'library', 'inspector', 'camview', 'bottom'), 'summary', 'Place sensors and mounts'),
    'grasp': (('outliner', 'inspector', 'camview', 'bottom'), 'summary', 'Grasp self-occlusion (Phase 3)'),
    'teleop': (('outliner', 'inspector', 'bottom'), 'summary', 'Teleop overlap and dead zones (Phase 5)'),
    'calibration': (('outliner', 'inspector', 'camview', 'bottom'), 'summary', 'Hand-eye calibration (Phase 7)'),
    'audit': (('outliner', 'inspector', 'bottom'), 'log', 'Dataset and episode audit (Phase 8)'),
    'review': (('outliner', 'inspector', 'bottom'), 'log', 'Variants, requirements and the design log'),
}
WORKSPACE_TITLES = {'placement': 'Placement', 'grasp': 'Grasp study', 'teleop': 'Teleop', 'calibration': 'Calibration',
                    'audit': 'Data audit', 'review': 'Review'}


class MainWindow(QMainWindow):
    def __init__(self, project: P.Project, soft_3d=False, settings=None):
        super().__init__()
        self.soft_3d = soft_3d
        self.store = ProjectStore(project)
        self.ctx = AppContext(self.store, settings)
        self.ctx.window = self
        self.actions_by_id: dict[str, QAction] = {}
        self._jobs = {}
        self.jobs = JobSignals()
        self.jobs.progress.connect(self._job_progress)
        self.jobs.done.connect(self._job_done)
        self.jobs.failed.connect(self._job_failed)
        self.current_workspace = 'placement'
        self.setWindowTitle('DexVantage')
        self.setAcceptDrops(True)
        self.resize(1680, 1000)
        self.setDockNestingEnabled(True)
        self.setCorner(Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.BottomDockWidgetArea)
        self.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.BottomDockWidgetArea)
        self._build_central()
        self._build_docks()
        self._build_actions()
        self._build_menus()
        self._build_pages()
        self._build_toolbar()
        self._build_statusbar()
        self._wire()
        self._default_state = self.saveState()
        self.set_project_meta()
        self.ctx.analysis.request(force=True)

    # ------------------------------------------------------------------ layout
    def _build_central(self):
        self.scene = SceneProvider(self.ctx)
        if self.soft_3d:
            from .viewport3d import SoftViewport
            self.viewport = SoftViewport(self.scene)
        else:
            self.viewport = make_viewport(self.scene)
        self.scrubber = PoseScrubber(self.ctx)
        c = QWidget()
        c.setObjectName('base')
        v = QVBoxLayout(c)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self.viewport, 1)
        v.addWidget(self.scrubber)
        c.setMinimumSize(360, 260)
        self.setCentralWidget(c)
        QTimer.singleShot(0, lambda: self.viewport.frame_box(*self.scene.home_box(self.viewport)))

    def _dock(self, key, title, widget, area, right=''):
        d = QDockWidget(title, self)
        d.setObjectName(f'dock_{key}')
        body = QWidget()
        body.setObjectName('dockBody')
        lay = QVBoxLayout(body)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(widget)
        d.setWidget(body)
        tb = DockTitle(d, title, right)
        d.setTitleBarWidget(tb)
        d.title_widget = tb
        d.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable | QDockWidget.DockWidgetFeature.DockWidgetMovable
                      | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.addDockWidget(area, d)
        self.docks[key] = d
        return d

    def _build_docks(self):
        self.docks = {}
        self.outliner = Outliner(self.ctx)
        self.library = LibraryPanel(self.ctx)
        self.inspector = Inspector(self.ctx)
        self.camview = CameraViewPanel(self.ctx)
        self.bottom = AnalysisPanel(self.ctx)
        self.assistant = AssistantPanel(self.ctx)
        L, R, B = (Qt.DockWidgetArea.LeftDockWidgetArea, Qt.DockWidgetArea.RightDockWidgetArea,
                   Qt.DockWidgetArea.BottomDockWidgetArea)
        self._dock('outliner', 'Outliner', self.outliner, L)
        self._dock('library', 'Library', self.library, L)
        self._dock('inspector', 'Inspector', self.inspector, R, 'Project')
        self._dock('camview', 'Camera view', self.camview, R)
        bd = self._dock('bottom', 'Analysis', self.bottom, B)
        bd.setTitleBarWidget(QWidget())
        ad = self._dock('assistant', 'Assistant', self.assistant, R)
        self.addDockWidget(R, ad, Qt.Orientation.Horizontal)      # its own full-height column
        ad.setTitleBarWidget(QWidget())
        ad.setMinimumWidth(280)
        ad.hide()
        self.splitDockWidget(self.docks['outliner'], self.docks['library'], Qt.Orientation.Vertical)
        self.splitDockWidget(self.docks['inspector'], self.docks['camview'], Qt.Orientation.Vertical)
        self.resizeDocks([self.docks['outliner'], self.docks['inspector']], [260, 320], Qt.Orientation.Horizontal)
        self.resizeDocks([self.docks['outliner'], self.docks['library']], [520, 260], Qt.Orientation.Vertical)
        self.resizeDocks([self.docks['inspector'], self.docks['camview']], [520, 240], Qt.Orientation.Vertical)
        self.resizeDocks([bd], [220], Qt.Orientation.Vertical)
        self.inspector.titleChanged.connect(lambda t: self.docks['inspector'].title_widget.right.setText(t))

    # ------------------------------------------------------------------ actions
    def act(self, aid, text, slot=None, shortcut=None, icon=None, checkable=False, category='', tip='', checked=False,
            phase=None):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
            a.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        if icon:
            a.setIcon(icons.icon(icon, 16))
        a.setCheckable(checkable)
        if checkable:
            a.setChecked(checked)
        if slot is not None:
            if checkable:
                a.toggled.connect(slot)
            else:
                a.triggered.connect(lambda _=False, s=slot: s())
        if phase is not None:
            a.setEnabled(False)
            tip = (tip + ' ' if tip else '') + f'Arrives in Phase {phase}.'
        a.setData({'id': aid, 'category': category, 'icon': icon, 'tip': tip, 'phase': phase})
        sc = f'  {QKeySequence(shortcut).toString(QKeySequence.SequenceFormat.NativeText)}' if shortcut else ''
        a.setToolTip(f'{text.replace("…", "").replace("&", "")}{sc}' + (f'\n{tip}' if tip else ''))
        a.setStatusTip(tip)
        self.addAction(a)
        self.actions_by_id[aid] = a
        return a

    def _build_actions(self):
        A = self.act
        st = self.store
        # file
        A('new', 'New project', self.new_project, 'Ctrl+N', 'plus', category='File')
        A('open', 'Open…', self.open_project, 'Ctrl+O', 'folder', category='File')
        A('save', 'Save', self.save, 'Ctrl+S', 'file', category='File')
        A('save_as', 'Save as…', self.save_as, 'Ctrl+Shift+S', category='File')
        A('render', 'Render image…', self.render_image, 'F12', 'export', category='File',
          tip='A high-resolution render of the 3D view, for reports')
        A('quit', 'Quit', self.close, 'Ctrl+Q', category='File')
        # edit
        u = st.undo.createUndoAction(self, 'Undo')
        u.setShortcut(QKeySequence.StandardKey.Undo)
        u.setIcon(icons.icon('undo', 16))
        u.setData({'id': 'undo', 'category': 'Edit', 'icon': 'undo'})
        self.addAction(u)
        self.actions_by_id['undo'] = u
        r = st.undo.createRedoAction(self, 'Redo')
        r.setShortcuts([QKeySequence('Ctrl+Shift+Z'), QKeySequence('Ctrl+Y')])
        r.setIcon(icons.icon('redo', 16))
        r.setData({'id': 'redo', 'category': 'Edit', 'icon': 'redo'})
        self.addAction(r)
        self.actions_by_id['redo'] = r
        A('palette', 'Command palette…', self.command_palette, 'Ctrl+K', 'search', category='Edit')
        A('find', 'Find object…', lambda: self.command_palette('obj '), 'Ctrl+F', 'search', category='Edit')
        A('delete', 'Delete selection', self.delete_selection, 'Delete', 'trash', category='Edit')
        # view
        for k, (name, sc) in {'persp': ('Perspective', '0'), 'top': ('Top', '7'), 'front': ('Front', '1'),
                              'side': ('Side', '3')}.items():
            A(f'view_{k}', f'{name} view', lambda k=k: self.viewport.set_view(k, animate=True), category='View',
              icon='cube' if k == 'persp' else 'axes', tip=f'Viewport key {sc}')
        A('frame_all', 'Frame everything', self.frame_all, 'Home', 'frame', category='View')
        for k, (text, on) in OVERLAYS.items():
            A(f'ov_{k}', text, lambda v, k=k: self.ctx.set_overlay(k, v), checkable=True, checked=on, category='View')
        g = QActionGroup(self)
        for k, text in (('dark', 'Dark'), ('light', 'Light'), ('system', 'Follow the system')):
            a = A(f'theme_{k}', f'{text} theme', lambda k=k: self.set_theme(k), category='View', checkable=False)
            a.setCheckable(True)
            g.addAction(a)
        self.actions_by_id[f'theme_{self.ctx.settings.value("ui/theme", "dark")}'].setChecked(True)
        A('frames_help', 'Coordinate frames', self.frames_help, category='Help', icon='axes')
        # robot
        A('import_robot', 'Import robot…', lambda: self.import_robot(), category='Robot', icon='robot-arm',
          tip='A URDF or MuJoCo MJCF file, with the meshes it names. The first robot becomes the primary one.')
        A('import_menagerie', 'Import from MuJoCo Menagerie…',
          lambda: __import__('dexvantage.ui.dialogs.menagerie', fromlist=['open_dialog']).open_dialog(self),
          category='Robot', icon='robot-arm', tip='Download a Menagerie model, read its own licence, accept, import.')
        A('joint_sliders', 'Joint sliders', category='Robot', icon='joint', phase=1)
        A('ik_drag', 'Drag the end effector (IK)', category='Robot', icon='gripper', phase=1)
        # sensors
        A('add_camera', 'Add camera on a tripod', self.add_static_camera, category='Sensors', icon='camera-wide',
          tip='A camera on a static frame, looking at the task volume. Pick its model from the library in Phase 2.')
        A('new_device', 'New custom device…', category='Sensors', icon='sensor', phase=2,
          tip='Every field marked as given by the maker or not given.')
        # workspace
        A('add_volume', 'Add task volume', self.add_volume, category='Workspace', icon='task-volume')
        A('add_object', 'Add box object', self.add_object, category='Workspace', icon='cube')
        # analysis
        A('run_summary', 'Run the project summary', lambda: self.ctx.analysis.run('summary'), 'F5', 'refresh',
          category='Analysis')
        A('live', 'Live analysis', self.set_live, checkable=True, checked=self.store.project.analysis.live,
          category='Analysis', tip='Re-run quick studies after every edit')
        A('cancel_jobs', 'Cancel running jobs', self.cancel_jobs, 'Ctrl+.', 'stop', category='Analysis')
        A('coverage', 'Coverage and resolution', category='Analysis', icon='coverage', phase=2)
        A('depth_validity', 'Depth validity', category='Analysis', icon='depth', phase=2)
        A('grasp_study', 'Grasp self-occlusion', category='Analysis', icon='grasp', phase=3)
        A('teleop_overlap', 'Teleop overlap', category='Analysis', icon='teleop', phase=5)
        # calibrate, simulate, data
        A('handeye_plan', 'Plan hand-eye poses…', category='Calibrate', icon='checker', phase=7)
        A('handeye_solve', 'Solve hand-eye…', category='Calibrate', icon='target', phase=7)
        A('mujoco_scene', 'Build the MuJoCo scene…', category='Simulate', icon='cube', phase=4)
        A('mujoco_validate', 'Rendered against predicted visibility…', category='Simulate', icon='compare', phase=4)
        A('audit', 'Audit a dataset…', category='Data', icon='dataset', phase=8,
          tip='LeRobot, RLDS / Open X-Embodiment or a ROS 2 bag.')
        A('plan_vs_recording', 'Plan against recording…', category='Data', icon='compare', phase=8)
        # export
        A('export_summary', 'Export the project summary…', self.export_summary, category='Export', icon='export',
          tip='JSON with its provenance block')
        A('export_formats', 'Export to MuJoCo, Isaac, ROS 2, LeRobot…', category='Export', icon='export', phase=10)
        # window
        for k, title in WORKSPACE_TITLES.items():
            A(f'ws_{k}', f'{title} workspace', lambda k=k: self.workspace(k), category='Window', icon='layers')
        A('ws_save', 'Save workspace layout', self.save_workspace, category='Window',
          tip='Keep the current dock layout for this workspace')
        A('ws_reset', 'Reset workspace layout', self.reset_workspace, category='Window')
        A('assistant', 'Assistant', self.toggle_assistant, 'Ctrl+J', 'chat', category='Window')
        # help
        A('shortcuts', 'Keyboard shortcuts', self.shortcuts, '?', 'keyboard', category='Help')
        A('validate', 'Validate the models…', self.validate_models, category='Help', icon='checklist',
          tip='Run the validation suite and open its report')
        A('about', 'About DexVantage', self.about, category='Help', icon='help')

    def _build_menus(self):
        mb = self.menuBar()

        self.menus = {}

        def menu(title, ids):
            m = mb.addMenu(title)
            self.menus[title.replace('&', '')] = m
            for i in ids:
                if i == '-':
                    m.addSeparator()
                elif isinstance(i, tuple):
                    sub = m.addMenu(i[0])
                    for j in i[1]:
                        sub.addSeparator() if j == '-' else sub.addAction(self.actions_by_id[j])
                else:
                    m.addAction(self.actions_by_id[i])
            return m
        self.m_file = menu('&File', ['new', 'open', '-', 'save', 'save_as', '-', 'render', '-', 'quit'])
        self.m_recent = self.m_file.addMenu('Open recent')
        self.m_file.insertMenu(self.actions_by_id['save'], self.m_recent)
        self.m_recent.aboutToShow.connect(self._fill_recent)
        self.m_samples = self.m_file.addMenu('Open sample')
        self.m_file.insertMenu(self.actions_by_id['save'], self.m_samples)
        from ..core.robot.samples import list_samples
        for smp in list_samples():
            a = self.m_samples.addAction(smp['title'], lambda n=smp['name']: self.open_sample(n))
            a.setEnabled(smp['available'])
            a.setToolTip(f'dexvantage --sample {smp["name"]}' if smp['available'] else
                         f'Arrives in Phase {smp["phase"]}.')
        menu('&Edit', ['undo', 'redo', '-', 'palette', 'find', '-', 'delete'])
        menu('&View', ['view_persp', 'view_top', 'view_front', 'view_side', 'frame_all', '-',
                       ('Overlays', [f'ov_{k}' for k in OVERLAYS]), ('Theme', ['theme_dark', 'theme_light',
                                                                             'theme_system'])])
        menu('&Robot', ['import_robot', 'import_menagerie', '-', 'joint_sliders', 'ik_drag'])
        menu('&Sensors', ['add_camera', '-', 'new_device'])
        menu('W&orkspace', ['add_volume', 'add_object'])
        menu('&Analysis', ['run_summary', 'live', 'cancel_jobs', '-', 'coverage', 'depth_validity', 'grasp_study',
                           'teleop_overlap'])
        menu('&Calibrate', ['handeye_plan', 'handeye_solve'])
        menu('Si&mulate', ['mujoco_scene', 'mujoco_validate'])
        menu('&Data', ['audit', 'plan_vs_recording'])
        menu('E&xport', ['export_summary', 'export_formats'])
        menu('&Window', [f'ws_{k}' for k in WORKSPACE_TITLES] + ['-', 'ws_save', 'ws_reset', '-', 'assistant'])
        menu('&Help', ['frames_help', 'shortcuts', '-', 'validate', '-', 'about'])

    def _build_pages(self):
        """Every registered analysis page (ui/pages): a tab in the Analysis dock and a command that opens it."""
        from .pages import load_pages
        self.pages = {}
        for info in load_pages():
            try:
                w = info.cls(self.ctx)
            except Exception:                           # noqa: BLE001 — one broken page must not stop the window
                traceback.print_exc()
                continue
            self.pages[info.id] = w
            self.bottom.add_page(info.id, info.title, w)
            a = self.actions_by_id.get(info.action_id or '')
            if a is not None:
                a.setEnabled(True)
                d = dict(a.data() or {})
                d['phase'] = None
                a.setData(d)
                a.setToolTip(info.title)
                a.triggered.connect(lambda _=False, pid=info.id: self.open_page(pid))
            else:
                a = self.act(f'page_{info.id}', info.title, lambda pid=info.id: self.open_page(pid), info.shortcut,
                             info.icon, category=info.menu)
                m = self.menus.get(info.menu) or self.menus['Analysis']
                m.addAction(a)

    def open_page(self, pid):
        self.docks['bottom'].show()
        self.bottom.show_tab(pid)
        w = self.pages.get(pid)
        if w is not None and hasattr(w, 'activate'):
            w.activate()
        self.viewport.update()

    def current_page(self):
        w = self.bottom.tabs.currentWidget()
        return w if w in self.pages.values() else None

    def page_overlays(self, vp) -> list:
        """The current analysis page's 3D overlays (LineDraw / TriDraw items)."""
        w = self.current_page()
        if w is None or not self.docks['bottom'].isVisible():
            return []
        try:
            return list(w.overlays(vp) or [])
        except Exception:                               # noqa: BLE001 — an overlay must not take the view down
            traceback.print_exc()
            return []

    def _build_toolbar(self):
        tb = QToolBar('Main')
        tb.setObjectName('toolbar_main')
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))
        self.addToolBar(tb)
        for i in ('new', 'open', 'save', '|', 'undo', 'redo', '|', 'view_persp', 'view_top', 'frame_all', '|',
                  'add_camera', 'add_volume', '|', 'run_summary'):
            if i == '|':
                tb.addSeparator()
            else:
                tb.addAction(self.actions_by_id[i])
        spacer = QWidget()
        spacer.setSizePolicy(spacer.sizePolicy().horizontalPolicy().Expanding,
                             spacer.sizePolicy().verticalPolicy().Preferred)
        tb.addWidget(spacer)
        self.ws_combo = QComboBox()
        for k, t in WORKSPACE_TITLES.items():
            self.ws_combo.addItem(t, k)
        self.ws_combo.setToolTip('Workspace: a saved arrangement of the panels for one kind of work')
        self.ws_combo.activated.connect(lambda i: self.workspace(self.ws_combo.itemData(i)))
        tb.addWidget(self.ws_combo)
        b = tool_button('search', 'Command palette  Ctrl+K', size=26, icon_size=16)
        b.clicked.connect(self.command_palette)
        tb.addWidget(b)
        b2 = tool_button('chat', 'Assistant  Ctrl+J', size=26, icon_size=16)
        b2.clicked.connect(self.toggle_assistant)
        tb.addWidget(b2)

    def _build_statusbar(self):
        sb = QStatusBar()
        sb.setSizeGripEnabled(False)
        self.setStatusBar(sb)
        self.msg_label = QLabel('')
        self.coord_label = QLabel('')
        self.coord_label.setFont(mono(11))
        self.coord_label.setMinimumWidth(260)
        self.sel_label = QLabel('')
        self.ws_label = QLabel('')
        for w in (self.msg_label, self.coord_label, self.sel_label, self.ws_label):
            sb.addWidget(w)
        self.msg_label.hide()
        self.job_label = QLabel('')
        self.job_label.setMaximumWidth(JOB_LABEL_W + 8)
        self.job_bar = QProgressBar()
        self.job_bar.setProperty('inline', True)
        self.job_bar.setFixedWidth(120)
        self.job_bar.setTextVisible(False)
        self.job_bar.setRange(0, 1000)
        self.job_pct = QLabel('')
        self.job_pct.setFont(mono(11))
        self.job_cancel = tool_button('close', 'Cancel', size=16, icon_size=12)
        self.job_cancel.clicked.connect(self.cancel_jobs)
        for w in (self.job_label, self.job_bar, self.job_pct, self.job_cancel):
            sb.addPermanentWidget(w)
            w.hide()
        self.state_label = QLabel('')
        self.state_label.setFont(mono(11))
        sb.addPermanentWidget(self.state_label)
        self._msg_timer = QTimer(self)
        self._msg_timer.setSingleShot(True)
        self._msg_timer.timeout.connect(self.msg_label.hide)

    def _wire(self):
        st = self.store
        st.dirty_changed.connect(lambda _d: self.set_project_meta())
        st.project_replaced.connect(self.set_project_meta)
        st.project_replaced.connect(self._sync_actions)
        st.selection_changed.connect(self._on_selection)
        an = self.ctx.analysis
        an.started.connect(lambda sid: self._update_state())
        an.finished.connect(lambda sid, doc: self._update_state())
        an.failed.connect(self._study_failed)
        an.stale_changed.connect(lambda _s: self._update_state())
        self.ctx.overlays_changed.connect(self.viewport.update)
        self.bottom.tabs.currentChanged.connect(lambda _i: self.viewport.update())
        THEME.changed.connect(lambda *_: self.viewport.update())

    # ------------------------------------------------------------------ status
    def msg(self, text, ms=3000):
        self.msg_label.setText(text)
        self.msg_label.show()
        self._msg_timer.start(ms)

    def set_coords(self, text):
        self.coord_label.setText(text)

    def _on_selection(self):
        o = self.store.selected()
        name = o.get('name') if isinstance(o, dict) else getattr(o, 'name', None)
        self.sel_label.setText(f'Selected: {name}' if o is not None else '')

    def _update_state(self):
        an = self.ctx.analysis
        if an.busy():
            self.state_label.setText('analysing…')
        elif an.stale:
            self.state_label.setText('out of date · F5 to run')
        else:
            doc = an.results.get('summary')
            self.state_label.setText(f'up to date · {doc["seconds"] * 1000:.1f} ms' if doc else '')

    def _study_failed(self, sid, m):
        print(m)                                         # the traceback goes to the log, never to the user
        self.msg(f'The {sid} study stopped with an error: {m.splitlines()[0]}', 8000)

    def set_project_meta(self):
        p = self.store.project
        dirty = not self.store.undo.isClean()
        name = os.path.basename(p.path) if p.path else p.name
        self.setWindowTitle(f'{name}{" •" if dirty else ""} · DexVantage')
        self.ws_label.setText(WORKSPACES[self.current_workspace][2])

    def _sync_actions(self):
        self.actions_by_id['live'].setChecked(self.store.project.analysis.live)

    # ------------------------------------------------------------------ workspaces
    def workspace(self, name):
        if name not in WORKSPACES:
            return
        self.current_workspace = name
        saved = self.ctx.settings.value(f'workspaces/{name}')
        docks, tab, _ = WORKSPACES[name]
        from .pages import PAGES
        tab = next((pid for pid in getattr(self, 'pages', {}) if PAGES[pid].workspace == name), tab)
        if saved is not None and self.restoreState(saved):
            pass
        else:
            self.restoreState(self._default_state)
            for k, d in self.docks.items():
                if k != 'assistant':
                    d.setVisible(k in docks)
        self.bottom.show_tab(tab)
        i = self.ws_combo.findData(name)
        if i >= 0:
            self.ws_combo.setCurrentIndex(i)
        self.set_project_meta()
        self.msg(f'{WORKSPACE_TITLES[name]} workspace')

    def save_workspace(self):
        self.ctx.settings.setValue(f'workspaces/{self.current_workspace}', self.saveState())
        self.msg(f'Saved the {WORKSPACE_TITLES[self.current_workspace]} layout')

    def reset_workspace(self):
        self.ctx.settings.remove(f'workspaces/{self.current_workspace}')
        self.workspace(self.current_workspace)
        self.msg(f'Reset the {WORKSPACE_TITLES[self.current_workspace]} layout')

    def toggle_assistant(self):
        d = self.docks['assistant']
        d.setVisible(not d.isVisible())

    # ------------------------------------------------------------------ view
    def frame_all(self):
        self.viewport.frame_box(*self.scene.home_box(self.viewport), animate=True)

    def set_theme(self, k):
        app = QGuiApplication.instance()
        THEME.apply(app, k)
        self.ctx.settings.setValue('ui/theme', k)
        icons.clear_cache()
        for a in self.actions_by_id.values():
            ic = (a.data() or {}).get('icon')
            if ic:
                a.setIcon(icons.icon(ic, 16))

    def set_live(self, on):
        self.store.edit('analysis', lambda p: setattr(p.analysis, 'live', bool(on)),
                        'Live analysis ' + ('on' if on else 'off'))

    def render_image(self, path=None, scale=2.0):
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, 'Render image', 'render.png', 'PNG image (*.png)')
            if not path:
                return None
        img = self.viewport.render_image(scale=scale)
        img.save(path)
        self.msg(f'Rendered {img.width()} × {img.height()} px to {os.path.basename(path)}')
        return path

    def thumbnail_png(self) -> bytes:
        try:
            img = self.viewport.render_image(scale=0.5)
        except Exception:                               # noqa: BLE001 — a project saves without its thumbnail
            return b''
        buf = QByteArray()
        b = QBuffer(buf)
        b.open(QIODevice.OpenModeFlag.WriteOnly)
        img.save(b, 'PNG')
        return bytes(buf)

    # ------------------------------------------------------------------ edits
    def import_robot(self, path=None) -> bool:
        """Robot ▸ Import robot…: read a URDF or MJCF and its meshes into the project, one undo step. Without a path
        the import wizard opens (ui/dialogs/import_wizard.py: mesh check, units, joint and gripper taggers)."""
        from ..core.robot import import_robot
        if not path:
            from .dialogs.import_wizard import ImportWizard
            return ImportWizard.run(self)
        try:
            robot, sources = import_robot(path)
        except P.ProjectFormatError as e:
            QMessageBox.warning(self, 'Could not import the robot', str(e))
            return False
        except Exception as e:                          # noqa: BLE001 — explained, the traceback logged
            traceback.print_exc()
            QMessageBox.warning(self, 'Could not import the robot',
                                f'{os.path.basename(path)} could not be read ({type(e).__name__}). Details are in '
                                'the log.')
            return False

        def fn(p):
            robot.primary = not p.robots          # the first robot is the primary one: its base is the world frame
            p.robots.append(robot)
            p.sources.update(sources)
        self.store.edit(['robots', 'sources'], fn, f'Imported robot {robot.name}')
        self.store.select(('robot', robot.uid))
        self.frame_all()
        nj = len(self.store.q.get(robot.uid, ()))
        self.msg(f'Imported {robot.name}: {len(robot.links)} links, {nj} moving joints')
        return True

    def add_volume(self):
        def fn(p):
            v = P.Volume(name=f'Task volume {len(p.workspace.volumes) + 1}', kind='custom')
            p.workspace.volumes.append(v)
            return v
        v = self.store.edit('workspace', fn, 'Added a task volume')
        self.store.select(('volume', v.uid))

    def add_object(self):
        def fn(p):
            o = P.TaskObject(name=f'Box {len(p.workspace.objects) + 1}', size=(0.06, 0.06, 0.06),
                             pose=Pose([0.45, 0.0, 0.03]))
            p.workspace.objects.append(o)
            return o
        o = self.store.edit('workspace', fn, 'Added a box object')
        self.store.select(('object', o.uid))

    def add_static_camera(self):
        import math

        def fn(p):
            n = len(p.variant.sensors)
            s = P.Sensor(name=f'Camera {n + 1}', kind='camera', slot=n % 8,
                         mount=P.Mount('tripod', '', Pose([0.45 - 0.5 * math.cos(0.6 * n), 0.55 * math.sin(0.6 * n + 1),
                                                           0.75], quat_from_rpy(0, math.radians(40), 0))))
            # look at the default task volume's centre
            c = p.workspace.volumes[0].center if p.workspace.volumes else (0.45, 0.0, 0.15)
            d = [c[i] - s.mount.pose.position[i] for i in range(3)]
            yaw = math.atan2(d[1], d[0])
            pitch = -math.atan2(d[2], math.hypot(d[0], d[1]))
            s.mount.pose = Pose(s.mount.pose.position, quat_from_rpy(0, pitch, yaw))
            p.variant.sensors.append(s)
            return s
        s = self.store.edit('variant', fn, 'Added a camera on a tripod')
        self.store.select(('sensor', s.uid))

    def delete_selection(self):
        st = self.store
        if len(st.sel) < 2:
            return
        kind, uid = st.sel[:2]
        o = st.selected()
        if o is None:
            return
        if kind == 'sensor':
            st.edit('variant', lambda p: p.variant.sensors.remove(p.variant.sensor(uid)), f'Deleted {o.name}')
        elif kind == 'volume':
            st.edit('workspace', lambda p: setattr(p.workspace, 'volumes', [v for v in p.workspace.volumes
                                                                              if v.uid != uid]), f'Deleted {o.name}')
        elif kind == 'object':
            st.edit('workspace', lambda p: setattr(p.workspace, 'objects', [v for v in p.workspace.objects
                                                                              if v.uid != uid]), f'Deleted {o.name}')
        elif kind == 'robot':
            if not self._delete_robot(o):
                return
        elif kind == 'trajectory':
            st.edit('tasks', lambda p: setattr(p.tasks, 'trajectories', [t for t in p.tasks.trajectories
                                                                          if t.get('uid') != uid]),
                    f'Deleted trajectory {o.get("name")}')
        else:
            return
        st.select(())

    def _delete_robot(self, robot) -> bool:
        """Delete a robot with its stored files and trajectories (one undo step); refused, with the reason, while it is
        locked or a sensor in any variant is mounted on it."""
        st = self.store
        uid = robot.uid
        if robot.locked:
            self.msg(f'{robot.name} is locked: unlock it to delete it.', 6000)
            return False
        mounted = [s.name for v in st.project.variants for s in v.sensors if s.mount.robot_uid == uid]
        if mounted:
            one = len(mounted) == 1
            QMessageBox.information(self, 'Robot not deleted',
                                    f'{", ".join(mounted[:3])}{" …" if len(mounted) > 3 else ""} '
                                    f'{"is" if one else "are"} mounted on {robot.name}: delete '
                                    f'{"it" if one else "them"} or mount {"it" if one else "them"} elsewhere first, '
                                    'then delete the robot.')
            return False

        def fn(p):
            was_primary = robot.primary
            p.robots = [r for r in p.robots if r.uid != uid]
            p.sources = {k: v for k, v in p.sources.items() if not k.startswith(uid + '/')}
            p.tasks.trajectories = [t for t in p.tasks.trajectories if t.get('robot_uid') != uid]
            if was_primary and p.robots:
                p.robots[0].primary = True
        st.edit(['robots', 'sources', 'tasks'], fn, f'Deleted {robot.name}')
        return True

    # ------------------------------------------------------------------ files
    def recent_files(self):
        return [f for f in (self.ctx.settings.value('recent', [], type=list) or []) if os.path.exists(f)]

    def add_recent(self, path):
        r = [path] + [f for f in self.recent_files() if f != path]
        self.ctx.settings.setValue('recent', r[:MAX_RECENT])

    def _fill_recent(self):
        self.m_recent.clear()
        files = self.recent_files()
        for f in files:
            self.m_recent.addAction(os.path.basename(f), lambda f=f: self.open_project(f))
        if not files:
            a = self.m_recent.addAction('No recent projects')
            a.setEnabled(False)

    def _maybe_save(self) -> bool:
        if self.store.undo.isClean():
            return True
        r = QMessageBox.question(self, 'Unsaved changes', f'Save the changes to {self.store.project.name}?',
                                 QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
                                 | QMessageBox.StandardButton.Cancel)
        if r == QMessageBox.StandardButton.Save:
            return self.save()
        return r == QMessageBox.StandardButton.Discard

    def open_sample(self, name) -> bool:
        """File ▸ Open sample: a bundled sample project (core/robot/samples.py), unsaved."""
        from ..core.robot.samples import sample_project
        if not self._maybe_save():
            return False
        try:
            proj = sample_project(name)
        except P.ProjectFormatError as e:
            QMessageBox.warning(self, 'Could not open the sample', str(e))
            return False
        except Exception as e:                          # noqa: BLE001 — explained, the traceback logged
            traceback.print_exc()
            QMessageBox.warning(self, 'Could not open the sample',
                                f'The sample could not be built ({type(e).__name__}). Details are in the log.')
            return False
        self.store.set_project(proj)
        self.frame_all()
        self.msg(f'Opened the {proj.name}: save it under a name of your own to keep changes.', 6000)
        return True

    def new_project(self):
        if not self._maybe_save():
            return
        self.store.set_project(P.blank_project())
        self.frame_all()

    def open_project(self, path=None):
        if not self._maybe_save():
            return False
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, 'Open project', '', f'DexVantage project (*{P.EXTENSION})')
            if not path:
                return False
        try:
            proj = P.Project.load(path)
        except P.ProjectFormatError as e:
            QMessageBox.warning(self, 'Could not open the project', str(e))
            return False
        except Exception as e:                          # noqa: BLE001 — explained, the traceback logged
            traceback.print_exc()
            QMessageBox.warning(self, 'Could not open the project',
                                f'{os.path.basename(path)} could not be read ({type(e).__name__}). Details are in '
                                'the log.')
            return False
        self.store.set_project(proj)
        self.add_recent(path)
        self.frame_all()
        self.msg(f'Opened {os.path.basename(path)}')
        return True

    def save(self) -> bool:
        p = self.store.project
        if not p.path:
            return self.save_as()
        return self._write(p.path)

    def save_as(self) -> bool:
        p = self.store.project
        path, _ = QFileDialog.getSaveFileName(self, 'Save project as', p.path or f'{p.name}{P.EXTENSION}',
                                              f'DexVantage project (*{P.EXTENSION})')
        if not path:
            return False
        if not path.endswith(P.EXTENSION):
            path += P.EXTENSION
        return self._write(path)

    def _write(self, path) -> bool:
        try:
            self.store.project.save(path, thumb=self.thumbnail_png())
        except OSError as e:
            QMessageBox.warning(self, 'Could not save', f'{os.path.basename(path)} could not be written: '
                                                        f'{e.strerror or e}.')
            return False
        self.store.undo.setClean()
        self.add_recent(path)
        self.set_project_meta()
        self.msg(f'Saved {os.path.basename(path)}')
        return True

    def export_summary(self, path=None):
        from ..core import jobs, provenance as PV
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, 'Export the project summary', 'summary.json', 'JSON (*.json)')
            if not path:
                return None
        doc = jobs.run_study('summary', self.store.project.snapshot(), cache=self.ctx.analysis.cache)
        doc = {k: v for k, v in doc.items() if k not in ('cached', 'revision')}
        doc = PV.stamp(doc, self.store.project, output={'option': None, 'name': os.path.basename(path),
                                                        'in_folder': False})
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(doc, fh, indent=1, ensure_ascii=False)
            fh.write('\n')
        self.msg(f'Exported the summary with its provenance to {os.path.basename(path)}')
        return path

    # ------------------------------------------------------------------ dialogs
    def command_palette(self, prefix=''):
        from .dialogs.palette import CommandPalette
        CommandPalette(self.ctx, self, prefix).exec()

    def palette_objects(self):
        st = self.store
        p = st.project
        out = []
        for r in p.robots:
            out.append({'kind': 'obj', 'text': r.name, 'cat': 'Robots', 'icon': 'robot-arm',
                        'run': lambda r=r: st.select(('robot', r.uid))})
        for s in p.variant.sensors:
            pos = s.mount.pose.position
            out.append({'kind': 'obj', 'text': s.name, 'cat': f'Sensors · {s.mount.parent_frame}',
                        'chip': (s.slot, s.name[:3]),
                        'detail': f'{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}',
                        'run': lambda s=s: st.select(('sensor', s.uid))})
        for v in p.workspace.volumes:
            out.append({'kind': 'obj', 'text': v.name, 'cat': 'Task volumes', 'icon': 'task-volume',
                        'run': lambda v=v: st.select(('volume', v.uid))})
        for o in p.workspace.objects:
            out.append({'kind': 'obj', 'text': o.name, 'cat': 'Task objects', 'icon': 'cube',
                        'run': lambda o=o: st.select(('object', o.uid))})
        return out

    def shortcuts(self):
        from .dialogs.misc import ShortcutsDialog
        ShortcutsDialog(self).exec()

    def frames_help(self):
        from .dialogs.misc import FramesDialog
        FramesDialog(self).exec()

    def about(self):
        from .dialogs.misc import AboutDialog
        AboutDialog(self).exec()

    def validate_models(self, folder=None, open_report=True, quick=False):
        """Help ▸ Validate the models: the suite in the background, the report into the working folder."""
        from .. import validation as VA, workdir
        from ..core import provenance as PV
        project = self.store.project.snapshot()
        if folder is None:
            folder = workdir.project_dir(project, workdir.get(self.ctx.settings), 'validation',
                                         time.strftime('%Y%m%d-%H%M%S'))
        st = VA.ValidationSettings.from_dict({'quick': bool(quick)})

        def work(progress, cancel):
            res = VA.run(project, settings=st, progress=lambda f: progress(f, f'{f * 100:.0f} % of the checks'),
                         cancel=cancel)
            os.makedirs(folder, exist_ok=True)
            paths = PV.stamp_paths(VA.export(res, project, folder), project, folder)
            return res, paths

        def done(out):
            res, _paths = out
            self.validation_report = os.path.join(folder, 'validation.html')
            text = f'{res.passed} of {len(res.checks)} checks passed' + (' (cancelled)' if res.cancelled else '')
            self.store.project.add_log('validation', text)
            self.msg(f'{text} · report in {folder}', 8000)
            if open_report and not res.cancelled and QGuiApplication.platformName() != 'offscreen':
                from PyQt6.QtCore import QUrl
                from PyQt6.QtGui import QDesktopServices
                QDesktopServices.openUrl(QUrl.fromLocalFile(self.validation_report))
        started = self.run_job('validation', 'Validating the models', work, done)
        if not started:
            self.msg('The models are being validated already', 4000)
        return started

    # ------------------------------------------------------------------ background jobs
    def run_job(self, kind, title, fn, on_done, on_failed=None):
        """fn(progress, cancel) -> result on a thread, then on_done(result) here (on_failed(message) if it raised)."""
        if kind in self._jobs:
            return False
        cancel = threading.Event()
        self._jobs[kind] = {'cancel': cancel, 'title': title, 'on_done': on_done, 'on_failed': on_failed}
        self._show_job(title, 0.0)

        def run():
            try:
                res = fn(lambda f, text='': self.jobs.progress.emit(kind, f, text), cancel.is_set)
                self.jobs.done.emit(kind, res)
            except Exception as e:                      # noqa: BLE001
                self.jobs.failed.emit(kind, f'{e}\n{traceback.format_exc()}')
        threading.Thread(target=run, daemon=True).start()
        return True

    def _show_job(self, name, frac, text=''):
        full = name + (f' · {text}' if text else '')
        self.job_label.setText(self.job_label.fontMetrics().elidedText(full, Qt.TextElideMode.ElideMiddle,
                                                                       JOB_LABEL_W))
        self.job_label.setToolTip(full)
        self.job_bar.setValue(int(frac * 1000))
        self.job_pct.setText(f'{frac * 100:.0f} %')
        for w in (self.job_label, self.job_bar, self.job_pct, self.job_cancel):
            w.show()

    def _job_progress(self, kind, frac, text):
        j = self._jobs.get(kind)
        if j:
            self._show_job(j['title'], frac, text)

    def _job_done(self, kind, result):
        job = self._jobs.pop(kind, None)
        self._hide_job()
        if job and job.get('on_done'):
            job['on_done'](result)

    def _job_failed(self, kind, m):
        job = self._jobs.pop(kind, None)
        self._hide_job()
        title = job['title'] if job else kind
        print(m)
        if job and job.get('on_failed'):
            job['on_failed'](m)
        elif job and job['cancel'].is_set():
            self.msg(f'{title} cancelled')
        else:
            QMessageBox.warning(self, f'{title} stopped', m.splitlines()[0])

    def _hide_job(self):
        if not self._jobs:
            for w in (self.job_label, self.job_bar, self.job_pct, self.job_cancel):
                w.hide()

    def cancel_jobs(self):
        for j in self._jobs.values():
            j['cancel'].set()
        self.ctx.analysis.cancel()

    # ------------------------------------------------------------------ window
    def closeEvent(self, e):
        if not self._maybe_save():
            e.ignore()
            return
        self.ctx.settings.setValue('window/geometry', self.saveGeometry())
        self.cancel_jobs()
        super().closeEvent(e)
