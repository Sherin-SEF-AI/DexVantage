"""The pose scrubber along the 3D view's bottom edge: a joint-state timeline.

When the project holds trajectories (project.tasks.trajectories, core/robot/trajectory.py) the slider scrubs q(t) of
the chosen one, interpolated linearly between its samples, and Play animates it at real time (wall-clock seconds,
whatever the frame rate). The pose goes to the robot's joint state (store.set_q: view state, not an undo step); the
robot's joints the trajectory does not name keep their values. The step buttons jump to the previous or next sample.

Without a trajectory the scrubber is disabled and says how to get one (Robot ▸ Import trajectory (CSV)…).
"""
from __future__ import annotations

import time

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QSlider, QWidget

from . import icons
from .theme import THEME, mono
from .widgets import label, tool_button

TICK_S = 0.001                    # one slider step: 1 ms of trajectory time
FRAME_MS = 16                     # the play timer's interval (the pose follows the wall clock, not the tick count)


class PoseScrubber(QWidget):
    """The joint-state timeline (see the module docstring)."""

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setFixedHeight(30)
        self.setObjectName('panel')
        h = QHBoxLayout(self)
        h.setContentsMargins(6, 0, 8, 0)
        h.setSpacing(4)
        self.b_back = tool_button('step-back', 'Previous sample', size=22, icon_size=12)
        self.b_play = tool_button('play', 'Play at real time', size=22, icon_size=12)
        self.b_step = tool_button('step', 'Next sample', size=22, icon_size=12)
        self.combo = QComboBox()
        self.combo.setFixedWidth(170)
        self.combo.setToolTip('The trajectory the scrubber plays')
        self.combo.hide()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.where = label('', muted=True, small=True)
        self.where.setFont(mono(11))
        for w in (self.b_back, self.b_play, self.b_step, self.combo):
            h.addWidget(w)
        h.addWidget(self.slider, 1)
        h.addWidget(self.where)
        self.traj_uid = None
        self.t = 0.0
        self._wall0 = 0.0
        self._t_start = 0.0
        self.timer = QTimer(self)
        self.timer.setInterval(FRAME_MS)
        self.timer.timeout.connect(self._tick)
        self.b_play.clicked.connect(self.toggle_play)
        self.b_back.clicked.connect(lambda: self.step(-1))
        self.b_step.clicked.connect(lambda: self.step(1))
        self.slider.valueChanged.connect(self._slid)
        self.combo.activated.connect(lambda i: self.select(self.combo.itemData(i)))
        ctx.store.changed.connect(lambda _s: self.refresh())
        ctx.store.project_replaced.connect(self.refresh)
        ctx.store.selection_changed.connect(self._picked)
        self.refresh()

    def _picked(self):
        """A trajectory selected in the Outliner becomes the one the scrubber plays."""
        sel = self.ctx.store.sel
        if sel[:1] == ('trajectory',) and len(sel) > 1:
            self.select(sel[1])

    # ---------------------------------------------------------------- what plays
    def trajectories(self) -> list:
        """The project's trajectories whose robot is in the project."""
        p = self.ctx.store.project
        return [t for t in p.tasks.trajectories if isinstance(t, dict) and p.robot(t.get('robot_uid')) is not None
                and t.get('t')]

    def current(self) -> dict | None:
        ts = self.trajectories()
        return next((t for t in ts if t.get('uid') == self.traj_uid), ts[0] if ts else None)

    @property
    def playing(self) -> bool:
        return self.timer.isActive()

    def refresh(self):
        from ..core.robot.trajectory import duration
        ts = self.trajectories()
        cur = self.current()
        self.traj_uid = cur.get('uid') if cur else None
        self.combo.blockSignals(True)
        self.combo.clear()
        for t in ts:
            self.combo.addItem(str(t.get('name') or 'Trajectory'), t.get('uid'))
        if cur is not None:
            self.combo.setCurrentIndex(self.combo.findData(self.traj_uid))
        self.combo.blockSignals(False)
        self.combo.setVisible(bool(ts))
        on = cur is not None
        for w in (self.b_back, self.b_play, self.b_step, self.slider):
            w.setEnabled(on)
        if not on:
            self.stop()
            self.slider.blockSignals(True)
            self.slider.setRange(0, 0)
            self.slider.blockSignals(False)
            self.where.setText('no trajectory')
            self.setToolTip('No trajectory yet: Robot ▸ Import trajectory (CSV)… adds one, and the scrubber then '
                            'plays it on the robot.' if self.ctx.store.project.robots else
                            'No robot yet: import one (Robot ▸ Import robot…), then a trajectory (Robot ▸ Import '
                            'trajectory (CSV)…).')
            return
        t0, t1 = duration(cur)
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, int(round((t1 - t0) / TICK_S))))
        self.slider.blockSignals(False)
        self.t = min(max(self.t, t0), t1)
        self._show()
        self.setToolTip(f'{cur.get("name")}: {len(cur["t"])} samples over {t1 - t0:.3f} s · {len(cur["joints"])} '
                        'joints. Drag to scrub; Play runs it at real time.')

    def select(self, uid):
        if uid != self.traj_uid:
            self.stop()
            self.traj_uid = uid
            cur = self.current()
            self.t = cur['t'][0] if cur else 0.0
            self.refresh()
            if cur is not None:
                self.scrub_to(self.t)

    # ---------------------------------------------------------------- scrubbing
    def _show(self):
        from ..core.robot.trajectory import duration
        cur = self.current()
        if cur is None:
            return
        t0, t1 = duration(cur)
        self.slider.blockSignals(True)
        self.slider.setValue(int(round((self.t - t0) / TICK_S)))
        self.slider.blockSignals(False)
        self.where.setText(f't {self.t - t0:7.3f} / {t1 - t0:.3f} s')

    def _slid(self, v):
        from ..core.robot.trajectory import duration
        cur = self.current()
        if cur is not None:
            self.scrub_to(duration(cur)[0] + v * TICK_S)

    def scrub_to(self, t: float):
        """Pose the trajectory's robot at time t (seconds on the trajectory's own clock, clamped to its range)."""
        from ..core.robot.model import RobotModelError, get_model
        from ..core.robot.trajectory import duration, on_model
        cur = self.current()
        if cur is None:
            return None
        t0, t1 = duration(cur)
        self.t = min(max(float(t), t0), t1)
        self._show()
        st = self.ctx.store
        uid = cur['robot_uid']
        try:
            m = get_model(st.project, uid)
        except RobotModelError:
            return None
        base = st.q.get(uid)
        base = m.home if base is None or len(base) != m.nq else base
        return st.set_q(uid, on_model(cur, m.joint_names, base, self.t))

    def step(self, d: int):
        cur = self.current()
        if cur is None:
            return
        self.stop()
        ts = cur['t']
        eps = 1e-9
        if d > 0:
            nxt = next((x for x in ts if x > self.t + eps), ts[-1])
        else:
            nxt = next((x for x in reversed(ts) if x < self.t - eps), ts[0])
        self.scrub_to(nxt)

    # ---------------------------------------------------------------- playing
    def toggle_play(self):
        self.stop() if self.playing else self.play()

    def play(self):
        from ..core.robot.trajectory import duration
        cur = self.current()
        if cur is None:
            return
        t0, t1 = duration(cur)
        if self.t >= t1 - 1e-9:
            self.t = t0
        self._t_start = self.t
        self._wall0 = time.monotonic()
        self.b_play.setIcon(icons.icon('pause', 12))
        self.b_play.setToolTip('Pause')
        self.timer.start()
        self.scrub_to(self.t)

    def stop(self):
        if self.timer.isActive():
            self.timer.stop()
        self.b_play.setIcon(icons.icon('play', 12))
        self.b_play.setToolTip('Play at real time')

    def _tick(self):
        from ..core.robot.trajectory import duration
        cur = self.current()
        if cur is None:
            self.stop()
            return
        t = self._t_start + (time.monotonic() - self._wall0)
        t1 = duration(cur)[1]
        self.scrub_to(min(t, t1))
        if t >= t1:
            self.stop()

    def paintEvent(self, e):
        from PyQt6.QtGui import QPainter
        p = QPainter(self)
        p.fillRect(self.rect(), THEME.color('panel'))
        p.setPen(THEME.color('hair'))
        p.drawLine(0, 0, self.width(), 0)
        p.end()
