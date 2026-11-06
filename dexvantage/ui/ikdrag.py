"""IK drag (Robot ▸ Drag the end effector): move a robot's gripper frame in the 3D view and let inverse kinematics
follow.

While the mode is on, the gripper frame of the robot being dragged (the selected robot, else the primary one) shows a
small handle. A left drag that starts on the handle moves the target on the plane through the handle that faces the
camera (view-aligned), and every move calls core.robot.ik.solve(model, link, target, q0=current q, rule='nearest_q0':
of the solutions, the one nearest the current pose, so the arm does not jump between branches while dragging). A
feasible solution becomes the robot's joint state (store.set_q: view state, not an undo step); the handle is then
green with the word "reachable". An infeasible one leaves the pose at the last reachable one and turns the handle red
with the solver's reason in the status bar. Esc during a drag puts the pose back where the drag started.

The target is the gripper frame's position in the world frame (orientation free), which every arm with three or more
joints can follow; when the solver only takes a full pose, the drag passes a 4 × 4 target holding the orientation the
frame had when the drag started.

The scene (ui/scene.py) hands its mouse, key and overlay events here first (SceneProvider.mouse_press, mouse_move,
mouse_release, key_press, overlay); the handle is drawn through ctx.overlay_sources.

Known limits: one robot and one frame at a time; the gripper frame must be a link (MJCF sites are not offered yet);
the drag follows the solver's own tolerance and iteration settings.
"""
from __future__ import annotations

import traceback

import numpy as np
from PyQt6.QtCore import QObject, QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPen

from .theme import STATUS, THEME, mono
from .viewport3d import LineDraw, TriDraw

HANDLE_PX = 8.0                   # the handle's radius on screen
PICK_PX = 14.0                    # how near the handle a press must be to start a drag
RULE = 'nearest_q0'               # of the solver's solutions, the one nearest the current pose: no jumps while dragging


def solver():
    """core.robot.ik.solve; raises ImportError when the IK module is not installed in this build."""
    from ..core.robot import ik
    return ik.solve


def _octahedron(c, r) -> np.ndarray:
    c = np.asarray(c, float)
    a = [c + [r, 0, 0], c + [-r, 0, 0], c + [0, r, 0], c + [0, -r, 0], c + [0, 0, r], c + [0, 0, -r]]
    tri = []
    for z in (4, 5):
        for x, y in ((0, 2), (2, 1), (1, 3), (3, 0)):
            tri.append([a[x], a[y], a[z]])
    return np.array(tri)


class IKDrag(QObject):
    """The IK drag mode of one window (see the module docstring)."""

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.on = False
        self._drag = None              # {'uid', 'link', 'q0', 'n', 'T0'} while dragging
        self.target = None             # world position the drag asks for (None: not dragging)
        self.last = None               # (ok, word, detail) of the last solve
        self.calls = 0

    # ---------------------------------------------------------------- state
    def set_enabled(self, on: bool):
        self.on = bool(on)
        if not self.on:
            self._drag, self.target = None, None
        win = self.ctx.window
        if win is not None:
            s = self.subject()
            if self.on and s is None:
                win.msg('IK drag is on, but there is no robot to drag yet: import one (Robot ▸ Import robot…).', 6000)
            elif self.on:
                r, _m, _q, link = s
                extra = '' if r.gripper_frame == link else f' (no gripper frame set: dragging {link}, its last link)'
                win.msg(f'IK drag: left-drag the handle on {r.name} · {link}{extra}', 6000)
            win.viewport.update()

    @property
    def dragging(self) -> bool:
        return self._drag is not None

    def subject(self):
        """(robot, model, q, link) of the robot to drag, or None."""
        from ..core.robot.model import RobotModelError, get_model
        st = self.ctx.store
        p = st.project
        r = None
        if st.sel[:1] == ('robot',) and len(st.sel) > 1:
            r = p.robot(st.sel[1])
        r = r or p.primary_robot
        if r is None:
            return None
        try:
            m = get_model(p, r.uid)
        except RobotModelError:
            return None
        if not m.nq or not m.link_names:
            return None
        q = st.q.get(r.uid)
        q = m.home if q is None or len(q) != m.nq else q
        link = r.gripper_frame if r.gripper_frame in m.link_names else m.link_names[-1]
        return r, m, q, link

    def handle(self):
        """The handle's world position (the drag target while dragging), or None."""
        if not self.on:
            return None
        if self.target is not None:
            return self.target
        s = self.subject()
        if s is None:
            return None
        _r, m, q, link = s
        return m.link_pose(q, link)[:3, 3].copy()

    # ---------------------------------------------------------------- solving
    def solve_to(self, target) -> object | None:
        """Run IK towards a world position for the robot being dragged; update store.q when it is feasible.
        Returns the solver's result (None when there is nothing to solve or no solver)."""
        s = self.subject()
        if s is None:
            return None
        r, m, q, link = s
        target = np.asarray(target, float).reshape(3)
        self.target = target
        try:
            solve = solver()
        except ImportError:
            self.last = (False, 'not available', 'The IK solver (core/robot/ik.py) is not part of this build yet.')
            self._say()
            return None
        q0 = self._drag['q_now'] if self._drag else q
        try:
            res = self._call(solve, m, link, target, q0)
        except Exception as e:                         # noqa: BLE001 — explained, the traceback logged
            traceback.print_exc()
            self.last = (False, 'solver error', f'The IK solver stopped with an error ({type(e).__name__}); details '
                                                'are in the log.')
            self._say()
            return None
        self.calls += 1
        ok = bool(getattr(res, 'ok', False))
        qn = getattr(res, 'q', None)
        err = getattr(res, 'pos_err', None)
        if ok and qn is not None:
            kept = self.ctx.store.set_q(r.uid, qn)
            if self._drag:
                self._drag['q_now'] = np.asarray(kept, float).copy()
            e = '' if err is None else f' · position error {float(err) * 1000:.2f} mm'
            self.last = (True, 'reachable', f'{link} follows the target{e}')
        else:
            reason = str(getattr(res, 'reason', '') or 'the solver found no pose that reaches the target')
            e = '' if err is None else f' (closest {float(err) * 1000:.1f} mm away)'
            self.last = (False, 'not reachable', f'{reason}{e}. The pose stays at the last reachable one.')
        self._say()
        return res

    def _call(self, solve, m, link, target, q0):
        try:
            try:
                return solve(m, link, target, q0=q0, rule=RULE)
            except TypeError:                  # a solver without the rule option
                return solve(m, link, target, q0=q0)
        except (TypeError, ValueError, IndexError):
            if self._drag is None or self._drag.get('T0') is None:
                raise
            T = self._drag['T0'].copy()          # the solver wants a full pose: hold the start orientation
            T[:3, 3] = target
            return solve(m, link, T, q0=q0)

    def _say(self):
        win = self.ctx.window
        if win is not None and self.last is not None:
            ok, word, detail = self.last
            win.msg(f'IK drag · {word}: {detail}', 6000 if ok else 10000)

    # ---------------------------------------------------------------- events (from SceneProvider)
    def press(self, vp, e) -> bool:
        if not self.on or e is None or e.button() != Qt.MouseButton.LeftButton:
            return False
        s = self.subject()
        if s is None:
            return False
        r, m, q, link = s
        T0 = m.link_pose(q, link)
        h = T0[:3, 3]
        sc, _d, ok = vp.project([h])
        pos = e.position()
        if not ok[0] or np.hypot(sc[0][0] - pos.x(), sc[0][1] - pos.y()) > PICK_PX:
            return False
        n = np.asarray(vp.cam.view()[2, :3], float)       # the camera's back axis: the plane faces the camera
        self._drag = {'uid': r.uid, 'link': link, 'q0': np.asarray(q, float).copy(),
                      'q_now': np.asarray(q, float).copy(), 'n': n / max(np.linalg.norm(n), 1e-12), 'p0': h.copy(),
                      'T0': T0.copy()}
        self.target = h.copy()
        vp.update()
        return True

    def plane_point(self, vp, pos):
        """Where the cursor's ray meets the drag plane (None when it runs parallel to it)."""
        o, d = vp.ray(pos.x(), pos.y())
        n, p0 = self._drag['n'], self._drag['p0']
        den = float(d @ n)
        if abs(den) < 1e-9:
            return None
        return o + d * (float((p0 - o) @ n) / den)

    def move(self, vp, e):
        if self._drag is None or e is None:
            return
        p = self.plane_point(vp, e.position())
        if p is not None:
            self.solve_to(p)
            vp.update()

    def release(self, vp, e):
        if self._drag is None:
            return
        self._drag, self.target = None, None
        vp.update()

    def key(self, vp, e) -> bool:
        if self._drag is not None and e is not None and e.key() == Qt.Key.Key_Escape:
            self.ctx.store.set_q(self._drag['uid'], self._drag['q0'])
            self._drag, self.target = None, None
            if hasattr(vp, 'cancel_provider_drag'):
                vp.cancel_provider_drag()
            win = self.ctx.window
            if win is not None:
                win.msg('IK drag cancelled: the pose is back where the drag started.')
            vp.update()
            return True
        return False

    # ---------------------------------------------------------------- drawing
    def _sev(self):
        if self.last is None or self.target is None:
            return 'good'
        return 'good' if self.last[0] else 'critical'

    def draws(self, vp) -> list:
        """The handle (a small octahedron, constant size on screen) and, while a target is out of reach, a dashed
        line from the frame to it."""
        h = self.handle()
        if h is None:
            return []
        r = HANDLE_PX / max(vp.px_per_m(h), 1e-6)
        col = QColor(STATUS[self._sev()])
        tri = _octahedron(h, r)
        rgba = np.tile(np.array(col.getRgbF()), (len(tri), 1))
        out = [TriDraw(tri, rgba, depth=False)]
        if self.target is not None and self.last is not None and not self.last[0]:
            s = self.subject()
            if s is not None:
                _r, m, q, link = s
                a = m.link_pose(q, link)[:3, 3]
                out.append(LineDraw(np.array([a, self.target]), col, 1.4, dash=True, depth=False, dash_len=0.01))
        return out

    def paint(self, p, vp):
        """The status word next to the handle while dragging (the colour never stands alone)."""
        if self.target is None or self.last is None:
            return
        sc, _d, ok = vp.project([self.target])
        if not ok[0]:
            return
        x, y = sc[0]
        sev = self._sev()
        txt = self.last[1]
        p.setFont(mono(11))
        fw = p.fontMetrics().horizontalAdvance(txt) + 12
        rect = QRectF(x + 14, y - 10, fw, 20)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(THEME.color('panel', 0.92))
        p.drawRect(rect)
        p.setPen(QPen(QColor(STATUS[sev]), 1))
        p.drawText(QPointF(rect.left() + 6, rect.top() + 14), txt)


def install(win) -> IKDrag:
    """Give the window its IK drag mode once: ctx.ik_drag, the handle overlay and the 'ik_drag' command (checkable)."""
    ik = getattr(win.ctx, 'ik_drag', None)
    if ik is not None:
        return ik
    ik = IKDrag(win.ctx)
    win.ctx.ik_drag = ik
    win.ctx.overlay_sources.append(ik.draws)
    a = win.actions_by_id.get('ik_drag')
    if a is not None:
        a.setEnabled(True)
        a.setCheckable(True)
        d = dict(a.data() or {})
        d['phase'] = None
        d['tip'] = 'Left-drag the handle on the gripper frame; inverse kinematics follows, green when reachable'
        a.setData(d)
        a.setToolTip('Drag the end effector (IK)\n' + d['tip'])
        a.setStatusTip(d['tip'])
        a.toggled.connect(ik.set_enabled)
    return ik
