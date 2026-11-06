"""What the 3D view draws: the SceneProvider the viewport asks for meshes, lines and overlays (ui/viewport3d.py).

Phase 0 draws the world frame (REP-103 axes at the primary robot's base), the floor grid, the task volumes, the task
objects and the sensors mounted on static frames (as a small body and a frustum outline). Robots, link-mounted sensors
(posed at the current joint state) and the study overlays arrive with the phases that build them.

The provider interface follows Vantage ui/scene.py, so the ported viewport, renderer and navigation work unchanged.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QColor

from ..core.frames import quat_to_matrix
from .theme import THEME
from .viewport3d import LineDraw, TriDraw

FRUSTUM_DEPTH = 0.12              # m: how far the sensor glyph's outline reaches (a symbol, not the field of view)
GLYPH_FOV = (60.0, 45.0)          # degrees: the glyph's shape until a lens model is assigned (Phase 2)


def box_edges(lo, hi) -> np.ndarray:
    """(12, 2, 3) segments of an axis-aligned box."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    idx = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    return np.stack([[c[a], c[b]] for a, b in idx])


def box_tris(lo, hi, R=None, t=None) -> np.ndarray:
    """(12, 3, 3) triangles of a box from lo to hi (in its own frame), optionally posed by R, t."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tri = np.array([[c[a], c[b], c[d]] for f in faces for a, b, d in ((f[0], f[1], f[2]), (f[0], f[2], f[3]))])
    if R is not None:
        tri = tri @ np.asarray(R).T + np.asarray(t)
    return tri


class SceneProvider(QObject):
    changed = pyqtSignal()

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        ctx.store.changed.connect(lambda _s: self.changed.emit())
        ctx.store.project_replaced.connect(self.changed.emit)
        ctx.store.selection_changed.connect(self.changed.emit)
        THEME.changed.connect(lambda *_: self.changed.emit())

    @property
    def project(self):
        return self.ctx.store.project

    # ---------------------------------------------------------------- geometry
    def mesh(self):
        return None                         # the posed robot meshes (Phase 1)

    def ground(self):
        return None

    def static_sensors(self):
        return [s for s in self.project.variant.sensors if s.enabled and not s.mount.on_link]

    def lines(self, vp):
        out = []
        if getattr(vp, 'renderer', None) is None and self.ctx.overlay('grid'):
            out += self._grid_lines()
        if self.ctx.overlay('frames'):
            for k, col in enumerate(('#e0533f', '#3fae49', '#3d86e6')):
                d = np.zeros(3)
                d[k] = 0.15
                out.append(LineDraw(np.array([[0, 0, 0.0005], d + [0, 0, 0.0005]]), QColor(col), 2.0, depth=False))
        sel = self.ctx.store.sel
        if self.ctx.overlay('volumes'):
            for v in self.project.workspace.volumes:
                lo, hi = v.bounds()
                on = sel[:2] == ('volume', v.uid)
                out.append(LineDraw(box_edges(lo, hi), THEME.color('ink' if on else 'ink2', 0.95 if on else 0.6),
                                    1.6 if on else 1.0, dash=not on and v.refine, segments=True, dash_len=0.02))
        for s in self.static_sensors():
            col = THEME.cam(s.slot)
            R, t = self._sensor_pose(s)
            hx = np.tan(np.radians(GLYPH_FOV[0] / 2)) * FRUSTUM_DEPTH
            hy = np.tan(np.radians(GLYPH_FOV[1] / 2)) * FRUSTUM_DEPTH
            # body frame: x forward (the optical axis), y left, z up
            far = np.array([[FRUSTUM_DEPTH, hx, hy], [FRUSTUM_DEPTH, -hx, hy], [FRUSTUM_DEPTH, -hx, -hy],
                            [FRUSTUM_DEPTH, hx, -hy]]) @ R.T + t
            segs = [[t, f] for f in far] + [[far[i], far[(i + 1) % 4]] for i in range(4)]
            on = sel[:2] == ('sensor', s.uid)
            out.append(LineDraw(np.array(segs), col, 1.8 if on else 1.2, segments=True))
        return out

    def _grid_lines(self):
        """The floor grid as lines, for the software renderer (the shader draws its own)."""
        ext, step = 1.0, 0.1
        k = np.arange(-ext, ext + 1e-9, step)
        segs = [[[x, -ext, 0], [x, ext, 0]] for x in k] + [[[-ext, y, 0], [ext, y, 0]] for y in k]
        return [LineDraw(np.array(segs), THEME.color('ink', 0.08), 1.0, segments=True)]

    def _sensor_pose(self, s):
        return quat_to_matrix(s.mount.pose.quaternion), np.asarray(s.mount.pose.position, float)

    def tris_extra(self, vp):
        out = []
        if self.ctx.overlay('volumes'):
            for v in self.project.workspace.volumes:
                lo, hi = v.bounds()
                tri = box_tris(lo, hi)
                c = np.array(THEME.color('ink2').getRgbF())
                c[3] = 0.05
                out.append(TriDraw(tri, np.tile(c, (len(tri), 1)), depth=True))
        for o in self.project.workspace.objects:
            if o.shape != 'box':
                continue                    # cylinders, spheres and meshes are drawn from Phase 2 on
            h = np.asarray(o.size, float) / 2
            R, t = quat_to_matrix(o.pose.quaternion), o.pose.position
            tri = box_tris(-h, h, R, t)
            base = np.array(THEME.color('body').getRgbF())
            on = self.ctx.store.sel[:2] == ('object', o.uid)
            out.append(TriDraw(tri, np.tile(base, (len(tri), 1)),
                               outline=tuple(THEME.color('ink').getRgbF()) if on else None))
        for s in self.static_sensors():
            R, t = self._sensor_pose(s)
            tri = box_tris((-0.02, -0.045, -0.015), (0.0, 0.045, 0.015), R, t)
            col = np.array(THEME.cam(s.slot).getRgbF())
            out.append(TriDraw(tri, np.tile(col, (len(tri), 1))))
        return out

    def render_style(self, vp):
        from .render_gl import Style
        dark = THEME.name != 'light'
        bg = np.array(THEME.color('viewport').getRgbF()[:3])
        ink = np.array(THEME.color('ink').getRgbF()[:3])
        lit = bool(self.ctx.overlay('shadows'))
        grid = bool(self.ctx.overlay('grid')) and getattr(vp, 'renderer', None) is not None
        return Style(shadows=lit, ao=lit, ground_plane=True, ground_rgb=tuple(bg * 0.9 + ink * 0.1),
                     sky_rgb=(0.62, 0.66, 0.72) if dark else (0.86, 0.88, 0.91),
                     bounce_rgb=(0.24, 0.23, 0.22) if dark else (0.50, 0.48, 0.45),
                     grid=grid, axes=grid, grid_minor=(*ink, 0.07 if dark else 0.10),
                     grid_major=(*ink, 0.15 if dark else 0.20))

    def background_color(self):
        return THEME.color('viewport')

    def background_image(self, vp):
        return None

    def hud_label(self, vp):
        p = self.project
        n = len(p.variant.sensors)
        if not p.robots:
            return f'{p.variant.name} · no robot yet: Robot ▸ Import arrives in Phase 1 · {n} sensors'
        return f'{p.variant.name} · {len(p.robots)} robots · {n} sensors'

    def overlay(self, p, vp):
        return None

    def home_box(self, vp):
        lo, hi = np.array([-0.3, -0.6, 0.0]), np.array([0.9, 0.6, 0.8])
        for v in self.project.workspace.volumes:
            a, b = v.bounds()
            lo, hi = np.minimum(lo, a), np.maximum(hi, b)
        return lo, hi, 1.05

    def pick_point(self, vp, pos):
        return None

    # ---------------------------------------------------------------- interaction
    def mouse_press(self, vp, e):
        return False

    def mouse_move(self, vp, e):
        pass

    def mouse_release(self, vp, e):
        pass

    def hover(self, vp, pos):
        win = self.ctx.window
        if win is None or pos is None:
            if win is not None:
                win.set_coords('')
            return
        g = vp.ground_at(pos.x(), pos.y(), 0.0)
        win.set_coords('' if g is None else f'x {g[0]:+.3f}  y {g[1]:+.3f} m · world (REP-103)')

    def click(self, vp, e):
        pass

    def double_click(self, vp, e):
        pass

    def context_menu(self, vp, pos, gpos):
        pass

    def key_press(self, vp, e):
        return False

    def drag_enter(self, vp, e):
        return False

    def drag_move(self, vp, e):
        pass

    def drop(self, vp, e):
        pass
