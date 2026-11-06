"""The Robot page (spec §13): joint sliders for every robot, the end-effector pose, manipulability, the condition number
and self-collision at the current pose.

Each moving joint has a slider and a number field, in degrees (revolute, continuous) or millimetres (prismatic), over
the limits the model file states (a continuous joint's slider spans one turn, ±180°; its field takes any value). The
sliders drive the robot's joint state live (store.set_q: view state, not an undo step), so the 3D view, the
link-mounted sensors and every read-out follow. Home puts the robot back at its home pose (the file's "home"
keyframe, else zero clipped to the limits).

Read-outs, for the gripper frame (or the last link while none is set):
- position in millimetres and roll, pitch, yaw in degrees (URDF convention), in the world frame (REP-103);
- Yoshikawa manipulability w = sqrt(det(J Jᵀ)) of the 6 × n world Jacobian, from core/robot/manipulability.py, and
  the Jacobian's condition number (σ_max / σ_min): large near a singularity;
- self-collision from core/robot/collision.py, with a status word, and the distance to the project's obstacles
  (workspace objects and surfaces, collision.obstacle_distance) when the project has any.
Each read-out says so plainly when its model is not part of this build yet.

A joint within NEAR_LIMIT of its range from a limit is flagged "near a limit" (a default: 2 % of the joint's range).

The page also installs the Robot menu's tools on the window: IK drag (ui/ikdrag.py) and Import trajectory (CSV)…,
which reads q(t) into project.tasks.trajectories as one undo step (core/robot/trajectory.py) for the pose scrubber.

Known limits: the read-outs are for one frame per robot; the Jacobian mixes metres and radians, so w and the condition
number compare poses of one robot, not robots with each other.
"""
from __future__ import annotations

import importlib
import importlib.util
import math
import os
import traceback

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QFileDialog, QGridLayout, QMessageBox, QSlider, QVBoxLayout, QWidget

from ...core.frames import quat_from_matrix, rpy_from_quat
from ...core.project import ProjectFormatError
from .. import jointfmt as JF
from ..panels import empty_state
from ..theme import mono
from ..widgets import ScrubSpinBox, Section, StatusLine, ValueLabel, button, label
from . import register
from .base import StudyPage

SLIDER_STEPS = 2000               # positions along a slider
CONTINUOUS_SPAN = math.pi         # rad: a continuous joint's slider spans ±180°
FREE_SLIDE_SPAN = 0.5             # m: an unlimited prismatic joint's slider spans ±500 mm
NEAR_LIMIT = 0.02                 # default: "near a limit" within 2 % of the joint's range
READOUT_LABEL_W = 124           # px: the read-out labels ('Condition number')
SINGULAR_COND = 1e12              # σ_max / σ_min above this: rank-deficient to double precision (a numerical bound)
READOUT_MS = 40                   # read-outs follow the sliders at most every 40 ms


def singular(cond: float) -> bool:
    """True when the Jacobian's condition number says the frame has lost a direction of motion."""
    return not math.isfinite(cond) or cond > SINGULAR_COND


# ====================================================================== the models the read-outs use (lazy)
def _module(name):
    """dexvantage.core.robot.<name>, or None when this build does not have it yet."""
    full = f'dexvantage.core.robot.{name}'
    try:
        if importlib.util.find_spec(full) is None:
            return None
    except (ImportError, ValueError):
        return None
    return importlib.import_module(full)


def manipulability(model, q, link):
    """(w, condition number, part) at q, or None when core/robot/manipulability.py is not available. part is 'full'
    (the 6 × n Jacobian) when six or more joints move the frame, else 'position' (its 3 × n linear rows: an arm with
    fewer joints cannot set all six pose coordinates, so the full measure would be 0 at every pose)."""
    M = _module('manipulability')
    if M is None:
        return None
    if hasattr(M, 'measures'):
        mask = M.chain_mask(model, link) if hasattr(M, 'chain_mask') else np.ones(model.nq, bool)
        part = 'full' if int(np.count_nonzero(mask)) >= 6 else 'position'
        d = M.measures(model, q, link, part=part)
        return float(d['yoshikawa']), float(d['condition']), part
    J = model.jacobian(q, link)
    w = float(M.yoshikawa(J))
    fn = getattr(M, 'condition', None)
    k = float(fn(J)) if fn is not None else float(np.linalg.cond(J))
    return w, k, 'full'


def self_collision(model, q):
    """(sev, word, text) of the self-collision check, or None when core/robot/collision.py is not available."""
    C = _module('collision')
    if C is None or not hasattr(C, 'self_collision'):
        return None
    res = C.self_collision(model, q)
    pairs = getattr(res, 'pairs', None)
    if hasattr(res, 'in_collision'):                   # collision.CollisionResult (a NamedTuple: test it first)
        hit = bool(res.in_collision)
    elif hasattr(res, 'colliding'):
        hit = bool(res.colliding)
    elif isinstance(res, (bool, np.bool_)):
        hit = bool(res)
    elif isinstance(res, (list, tuple, set)):
        pairs = list(res)
        hit = bool(pairs)
    else:
        hit = bool(pairs)
    dist = None
    for a in ('min_distance', 'distance', 'clearance'):
        v = getattr(res, a, None)
        if isinstance(v, (int, float)) and math.isfinite(v):
            dist = float(v)
            break
    if hit:
        names = []
        for p in (pairs or [])[:3]:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                names.append(f'{p[0]} and {p[1]}')
        what = ('; '.join(names)) if names else 'two links'
        return 'critical', 'in collision', f'{what} touch at this pose: move the joints apart.'
    cap = getattr(C, 'DIST_MAX', None)
    extra = ''
    if dist is not None and cap is not None and dist >= cap:
        extra = f' No checked pair is within {cap * 1000:.0f} mm.'
    elif dist is not None:
        p = (pairs or [None])[0]
        who = f' ({p[0]} and {p[1]})' if isinstance(p, (list, tuple)) and len(p) >= 2 else ''
        extra = f' Closest pair{who} {dist * 1000:.1f} mm apart.'
    return 'good', 'clear', 'No two links touch at this pose.' + extra


def obstacle_collision(project, uid, q):
    """(sev, word, text) of the robot's distance to the project's obstacles (workspace objects and surfaces, and the
    static geoms of its own file), or None when there are none or collision.py has no obstacle_distance."""
    C = _module('collision')
    if C is None or not hasattr(C, 'obstacle_distance'):
        return None
    if not project.workspace.objects and not project.workspace.surfaces:
        return None
    res = C.obstacle_distance(project, uid, q)
    p = (res.pairs or [None])[0]
    if res.in_collision:
        names = [f'{a} and {b}' for a, b, _d in res.pairs[:3]]
        return ('critical', 'in collision', f'{"; ".join(names)} touch at this pose: move the joints or the '
                                            'obstacle.')
    if p is None:
        return 'good', 'clear', f'No obstacle within {C.DIST_MAX * 1000:.0f} mm.'
    return 'good', 'clear', f'Nearest obstacle {p[1]}, {p[2] * 1000:.1f} mm from {p[0]}.'


# ====================================================================== one robot's sliders
class RobotPanel(Section):
    def __init__(self, page, robot, model):
        super().__init__(robot.name + (' · primary' if robot.primary else ''))
        self.page = page
        self.uid = robot.uid
        self.model = model
        self.sliders, self.spins, self.flags = [], [], []
        self._spans = []
        grid_w = QWidget()
        g = QGridLayout(grid_w)
        g.setContentsMargins(0, 2, 0, 2)
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(2)
        for i, (name, jt) in enumerate(zip(model.joint_names, model.joint_types)):
            lo, hi = float(model.lower[i]), float(model.upper[i])
            if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
                span = (lo, hi)
            else:
                w = FREE_SLIDE_SPAN if JF.is_linear(jt) else CONTINUOUS_SPAN
                span = (lo if math.isfinite(lo) else -w, hi if math.isfinite(hi) else w)
            self._spans.append(span)
            n = label(name)
            n.setToolTip(f'{name}: {jt} joint')
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(0, SLIDER_STEPS)
            s.setMinimumWidth(160)
            s.setToolTip(f'{name}: drag to move the joint ({JF.unit(jt)})')
            sp = ScrubSpinBox('', 'length' if JF.is_linear(jt) else 'angle', 2, JF.unit(jt), width=96)
            if JF.is_linear(jt):
                sp.set_display_unit('mm')
            lim = label(f'{JF.fmt(lo, jt)} … {JF.fmt(hi, jt)}' if math.isfinite(lo) or math.isfinite(hi)
                        else 'no limit (continuous)', muted=True, small=True)
            lim.setFont(mono(10))
            lim.setToolTip('The limits the model file states' if math.isfinite(lo) or math.isfinite(hi) else
                           'The file sets no limit: the slider spans one turn, the field takes any value')
            flag = StatusLine('', '', wrap=False)
            g.addWidget(n, i, 0)
            g.addWidget(s, i, 1)
            g.addWidget(sp, i, 2)
            g.addWidget(lim, i, 3)
            g.addWidget(flag, i, 4)
            s.valueChanged.connect(lambda v, i=i: self._slid(i, v))
            s.sliderReleased.connect(self.page.refresh_readouts)
            sp.committed.connect(lambda v, i=i: self.set_joint(i, JF.from_display(v, self.model.joint_types[i])))
            self.sliders.append(s)
            self.spins.append(sp)
            self.flags.append(flag)
        g.setColumnStretch(1, 1)
        if model.nq:
            self.add(grid_w)
        else:
            self.add(label('This robot has no moving joints.', muted=True))
        self.b_home = button('Home', icon_name='refresh', tip='Back to the home pose (the file\'s "home" keyframe, '
                                                              'else zero clipped to the limits)')
        self.b_home.clicked.connect(self.home)
        self.frame = label('', ink2=True)
        self.pos = ValueLabel()
        self.rot = ValueLabel()
        self.manip = ValueLabel()
        self.cond = ValueLabel()
        self.coll = StatusLine('', '', wrap=True)
        lw = READOUT_LABEL_W
        self.row('Frame', self.frame, self.b_home, stretch_last=False, label_w=lw)
        self.row('Position', self.pos, label_w=lw,
                 tip='The frame\'s origin in the world frame (REP-103: x forward, y left, z up)')
        self.row('Roll, pitch, yaw', self.rot, label_w=lw,
                 tip='URDF convention: about fixed x, then y, then z; world frame')
        self.row('Manipulability', self.manip, label_w=lw,
                 tip='Yoshikawa w = sqrt(det(J Jᵀ)) of the world Jacobian at the frame (its position rows, in m³, '
                     'when fewer than 6 joints move it): 0 at a singularity. Compare poses of one robot.')
        self.row('Condition number', self.cond, label_w=lw,
                 tip='σ_max / σ_min of the same Jacobian: large near a singularity (the 6 × n Jacobian mixes metres '
                     'and radians)')
        self.add(self.coll)
        self.sync()

    # ---------------------------------------------------------------- values
    def q(self) -> np.ndarray:
        q = self.page.ctx.store.q.get(self.uid)
        return self.model.home.copy() if q is None or len(q) != self.model.nq else np.asarray(q, float).copy()

    def _slid(self, i, v):
        lo, hi = self._spans[i]
        self.set_joint(i, lo + (hi - lo) * v / SLIDER_STEPS)

    def set_joint(self, i, value):
        q = self.q()
        q[i] = float(value)
        self.page.ctx.store.set_q(self.uid, q)

    def home(self):
        self.page.ctx.store.set_q(self.uid, self.model.home)

    def sync(self):
        """Put the sliders, fields and limit flags at the store's q (signals blocked: no feedback)."""
        q = self.q()
        for i, (s, sp, f) in enumerate(zip(self.sliders, self.spins, self.flags)):
            lo, hi = self._spans[i]
            jt = self.model.joint_types[i]
            s.blockSignals(True)
            s.setValue(int(round((min(max(q[i], lo), hi) - lo) / (hi - lo) * SLIDER_STEPS)) if hi > lo else 0)
            s.blockSignals(False)
            sp.set_value(JF.to_display(q[i], jt))
            L, U = float(self.model.lower[i]), float(self.model.upper[i])
            near = None
            if math.isfinite(L) and math.isfinite(U) and U > L:
                m = NEAR_LIMIT * (U - L)
                near = 'lower' if q[i] - L <= m else ('upper' if U - q[i] <= m else None)
            f.set('warning' if near else '', f'near its {near} limit' if near else '')
            f.setToolTip(f'Within {NEAR_LIMIT * 100:.0f} % of its range (a default) of the {near} limit' if near
                         else '')

    def near_limits(self) -> list:
        """The joints that move the read-out frame and sit near a limit (a gripper's open fingers at their stop, or
        the other arm, are not the frame's problem: their own sliders still flag them)."""
        q = self.q()
        out = []
        M = _module('manipulability')
        moves = M.chain_mask(self.model, self.link()[0]) if hasattr(M, 'chain_mask') else np.ones(self.model.nq, bool)
        for i, n in enumerate(self.model.joint_names):
            if not moves[i]:
                continue
            L, U = float(self.model.lower[i]), float(self.model.upper[i])
            if math.isfinite(L) and math.isfinite(U) and U > L:
                m = NEAR_LIMIT * (U - L)
                if q[i] - L <= m or U - q[i] <= m:
                    out.append(n)
        return out

    def link(self):
        r = self.page.ctx.store.project.robot(self.uid)
        gf = r.gripper_frame if r is not None else ''
        return (gf, True) if gf in self.model.link_names else (self.model.link_names[-1], False)

    def readouts(self) -> dict:
        """Compute and show the read-outs; returns them (for the headline and the tests)."""
        q = self.q()
        link, tagged = self.link()
        T = self.model.link_pose(q, link)
        p = T[:3, 3] * 1000
        rpy = np.degrees(rpy_from_quat(quat_from_matrix(T[:3, :3])))
        pr, rr = np.round(p, 1) + 0.0, np.round(rpy, 1) + 0.0          # + 0.0: no "−0.0" in the read-outs
        self.frame.setText(link + ('' if tagged else ' (last link: no gripper frame set)'))
        self.pos.set(f'{pr[0]:+.1f}  {pr[1]:+.1f}  {pr[2]:+.1f}'.replace('-', '−'), 'mm')
        self.rot.set(f'{rr[0]:+.1f}  {rr[1]:+.1f}  {rr[2]:+.1f}'.replace('-', '−'), '°')
        out = {'link': link, 'position_mm': p, 'rpy_deg': rpy, 'w': None, 'cond': None, 'collision': None}
        try:
            mk = manipulability(self.model, q, link) if self.model.nq else None
        except Exception:                               # noqa: BLE001 — explained, the traceback logged
            traceback.print_exc()
            mk = 'error'
        if mk is None:
            self.manip.set('—', '', note='arrives with core/robot/manipulability.py')
            self.cond.set('—', '')
        elif mk == 'error':
            self.manip.set('—', '', 'warning', 'could not be computed (see the log)')
            self.cond.set('—', '')
        else:
            out['w'], out['cond'], out['part'] = mk
            pos = mk[2] == 'position'
            self.manip.set(f'{mk[0]:.4g}', 'm³' if pos else '',
                           note=('position rows (3 × n): fewer than 6 joints move the frame' if pos else
                                 '6 × n Jacobian'))
            sing = singular(mk[1])
            self.cond.set('∞' if sing else (f'{mk[1]:.1f}' if mk[1] < 1e4 else f'{mk[1]:.3g}'), '',
                          'warning' if sing else None,
                          'singular: the frame cannot move in some direction' if sing else '')
        try:
            sc = self_collision(self.model, q)
        except Exception:                               # noqa: BLE001
            traceback.print_exc()
            sc = ('warning', 'unknown', 'The self-collision check stopped with an error (see the log).')
        try:
            oc = obstacle_collision(self.page.ctx.store.project, self.uid, q)
        except Exception:                               # noqa: BLE001
            traceback.print_exc()
            oc = ('warning', 'unknown', 'The obstacle check stopped with an error (see the log).')
        if sc is None:
            self.coll.set('', 'Self-collision: arrives with core/robot/collision.py.')
        elif oc is None:
            self.coll.set(sc[0], f'Self-collision · {sc[1]}: {sc[2]}')
        else:
            rank = {'critical': 2, 'warning': 1}
            worst = max((sc, oc), key=lambda x: rank.get(x[0], 0))
            self.coll.set(worst[0], f'Collision · {worst[1]}: {sc[2]} Obstacles: {oc[2]}')
            sc = (worst[0], worst[1], sc[2] if worst is sc else oc[2])
        out['collision'] = sc
        out['obstacles'] = oc
        out['near'] = self.near_limits()
        return out


# ====================================================================== the page
class RobotPage(StudyPage):
    """Joint sliders: move each joint of each robot in degrees or millimetres over the limits its model file states,
    and read the gripper frame's pose, manipulability, condition number and self-collision at that pose."""
    STUDY = None
    TITLE = 'Robot'
    HEADLINE = 'Robot pose'

    def __init__(self, ctx):
        self._panels: dict = {}
        self._key = None
        self._dirty = True
        super().__init__(ctx)
        for b in (self.b_run, self.b_apply, self.b_preview, self.b_export):
            b.hide()
        self.b_home_all = button('Home all', icon_name='refresh', tip='Every robot back to its home pose')
        self.b_home_all.clicked.connect(self.home_all)
        self.head.actions.insertWidget(0, self.b_home_all)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(READOUT_MS)
        self.timer.timeout.connect(self.refresh_readouts)
        st = ctx.store
        st.changed.connect(lambda _s: self._mark())
        st.project_replaced.connect(self._mark)
        st.pose_changed.connect(self._pose)
        if ctx.window is not None:
            install_tools(ctx.window)

    def build_body(self, layout):
        self.holder = QWidget()
        self.hl = QVBoxLayout(self.holder)
        self.hl.setContentsMargins(0, 0, 0, 0)
        self.hl.setSpacing(6)
        layout.addWidget(self.holder)

    def empty(self):
        self.head.set('—', '', 'warning', 'no robot', 'Import a robot (Robot ▸ Import robot…) to move its joints here.')

    # ---------------------------------------------------------------- building (lazy: only while shown)
    def _models(self):
        from ...core.robot.model import RobotModelError, get_model
        out = []
        for r in self.ctx.store.project.robots:
            try:
                out.append((r, get_model(self.ctx.store.project, r.uid)))
            except RobotModelError:
                continue
        return out

    def _mark(self):
        self._dirty = True
        if self.isVisible():
            self.rebuild()

    def showEvent(self, e):
        super().showEvent(e)
        if self._dirty:
            self.rebuild()

    def activate(self):
        self.rebuild()

    def rebuild(self):
        """(Re)build the per-robot panels when the robots or their joints changed; otherwise just re-sync them."""
        self._dirty = False
        rm = self._models()
        key = tuple((r.uid, r.name, r.primary, m) for r, m in rm)
        if key != self._key:
            self._key = key
            while self.hl.count():
                w = self.hl.takeAt(0).widget()
                if w is not None:
                    w.deleteLater()
            self._panels = {}
            if not rm:
                self.hl.addWidget(empty_state('No robot yet', 'Import a URDF or MJCF file (Robot ▸ Import robot…); '
                                                              'its joints appear here with the limits the file '
                                                              'states.'))
            for r, m in rm:
                p = RobotPanel(self, r, m)
                self._panels[r.uid] = p
                self.hl.addWidget(p)
        for p in self._panels.values():
            p.sync()
        self.refresh_readouts()

    def panel(self, uid) -> RobotPanel | None:
        if self._dirty or uid not in self._panels:
            self.rebuild()
        return self._panels.get(uid)

    # ---------------------------------------------------------------- live updates
    def _pose(self, uid):
        p = self._panels.get(uid)
        if p is None:
            return
        p.sync()
        if not self.timer.isActive():
            self.timer.start()

    def home_all(self):
        for r, m in self._models():
            self.ctx.store.set_q(r.uid, m.home)

    def focus_uid(self):
        st = self.ctx.store
        if st.sel[:1] == ('robot',) and len(st.sel) > 1 and st.sel[1] in self._panels:
            return st.sel[1]
        r = st.project.primary_robot
        return r.uid if r is not None and r.uid in self._panels else next(iter(self._panels), None)

    def refresh_readouts(self) -> dict:
        """Every panel's read-outs and the headline (for the robot in focus); returns {uid: read-outs}."""
        out = {uid: p.readouts() for uid, p in self._panels.items()}
        uid = self.focus_uid()
        if uid is None:
            self.empty()
            return out
        r = self.ctx.store.project.robot(uid)
        d = out[uid]
        coll, near = d['collision'], d.get('near') or []
        fig = '—' if d['w'] is None else f'{d["w"]:.4g}'
        unit = 'm³ · manipulability w' if d.get('part') == 'position' else 'manipulability w'
        tip = ('Yoshikawa manipulability of the gripper frame (0 at a singularity)' if d['w'] is not None else
               'Manipulability arrives with core/robot/manipulability.py')
        if coll is not None and coll[0] == 'critical':
            self.head.set(fig, unit, 'critical', 'in collision', f'{r.name}: {coll[2]}', tip)
        elif near:
            self.head.set(fig, unit, 'warning', 'near a limit',
                          f'{r.name}: {", ".join(near[:3])} {"is" if len(near) == 1 else "are"} within '
                          f'{NEAR_LIMIT * 100:.0f} % (default) of a limit: move away from it, or the robot cannot '
                          'follow a motion in that direction.', tip)
        elif d['cond'] is not None and singular(d['cond']):
            self.head.set(fig, unit, 'warning', 'singular',
                          f'{r.name} at {d["link"]}: the frame cannot move in some direction at this pose (an arm '
                          'stretched straight or two wrist axes lined up). Bend a joint away from it.', tip)
        else:
            word = 'clear' if coll is not None else 'within limits'
            text = (f'{r.name} at {d["link"]}: every joint inside its limits'
                    + ((' and no link touches another' + (' or an obstacle.' if d.get('obstacles') else '.'))
                       if coll is not None else '. Self-collision arrives with core/robot/collision.py.'))
            self.head.set(fig, unit, 'good', word, text, tip)
        return out


# ====================================================================== Robot ▸ Import trajectory (CSV)…
def import_trajectory(ctx, path: str | None = None, parent=None) -> bool:
    """Read a q(t) CSV (t, then joint names; radians, metres) into project.tasks.trajectories as one undo step."""
    from ...core.robot.model import RobotModelError, get_model
    from ...core.robot.trajectory import read_csv
    st = ctx.store
    win = ctx.window
    parent = parent or win
    if not path:
        path, _ = QFileDialog.getOpenFileName(parent, 'Import trajectory', '', 'Trajectory CSV (*.csv *.txt);;All '
                                                                                'files (*)')
        if not path:
            return False
    robots = []
    for r in st.project.robots:
        try:
            m = get_model(st.project, r.uid)
        except RobotModelError:
            continue
        robots.append((r.uid, m.joint_names, m.lower, m.upper, m.joint_types))
    prefer = st.sel[1] if st.sel[:1] == ('robot',) and len(st.sel) > 1 else (
        st.project.primary_robot.uid if st.project.primary_robot else None)
    try:
        rec, warnings = read_csv(path, robots, prefer_uid=prefer)
    except ProjectFormatError as e:
        QMessageBox.warning(parent, 'Could not import the trajectory', str(e))
        return False
    except Exception as e:                              # noqa: BLE001 — explained, the traceback logged
        traceback.print_exc()
        QMessageBox.warning(parent, 'Could not import the trajectory',
                            f'{os.path.basename(path)} could not be read ({type(e).__name__}). Details are in the '
                            'log.')
        return False
    st.edit('tasks', lambda p: p.tasks.trajectories.append(rec), f'Imported trajectory {rec["name"]}')
    robot = st.project.robot(rec['robot_uid'])
    if win is not None:
        sc = getattr(win, 'scrubber', None)
        if sc is not None and hasattr(sc, 'select'):
            sc.select(rec['uid'])
        dt = rec['t'][-1] - rec['t'][0]
        n = len(rec['t'])
        text = (f'Imported {rec["name"]}: {n} sample{"" if n == 1 else "s"} over {dt:.2f} s for {robot.name}. Scrub '
                'or play it under the 3D view.')
        win.msg(text + (' ' + ' '.join(warnings) if warnings else ''), 12000 if warnings else 6000)
    return True


def install_tools(win):
    """Put the Robot menu's tools on the window once: IK drag and Import trajectory (CSV)…."""
    if getattr(win, '_robot_tools', False) or not hasattr(win, 'actions_by_id'):
        return
    win._robot_tools = True
    from .. import ikdrag
    ikdrag.install(win)
    if 'import_trajectory' not in win.actions_by_id:
        a = win.act('import_trajectory', 'Import trajectory (CSV)…', lambda: import_trajectory(win.ctx),
                    category='Robot', icon='timeline',
                    tip='q(t) for the pose scrubber: a header row with t, then joint names; seconds and radians '
                        '(metres for sliding joints)')
        m = getattr(win, 'menus', {}).get('Robot')
        if m is not None:
            m.addAction(a)


register('robot', 'Robot', RobotPage, menu='Robot', icon='joint', action_id='joint_sliders', workspace='placement',
         order=10)
