"""What the 3D view draws: the SceneProvider the viewport asks for meshes, lines and overlays (ui/viewport3d.py).

It draws the world frame (REP-103 axes at the primary robot's base), the floor grid, the task volumes, the task
objects, every robot posed at the store's joint state (store.q) as the lit and shadowed mesh(), and the sensors (as a
small body and a frustum outline): those on static frames where they are, those on a link at the link's pose for the
current q composed with the mount pose.

Frusta come from each imager's real lens model (library.sensor_imagers, core/optics CameraModel): the image border
unprojected through the model (core/camrender.frustum_body), so a fisheye draws a curved cone. The first imager
(the depth reference) is drawn whole, the device's other imagers by their outline. The outline lies FRUSTUM_DEPTH in
front of the imager (SELECTED_DEPTH for the selected sensor): a symbol of the field of view, not a range. A library
device dropped on the view (from the Library panel) is added to the rig above the floor point it was dropped on.

The provider interface follows Vantage ui/scene.py, so the ported viewport, renderer and navigation work unchanged.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QColor

from ..core.frames import quat_to_matrix
from ..core.robot.model import RobotModelError, get_model
from .theme import THEME
from .viewport3d import LineDraw, MeshDraw, TriDraw

FRUSTUM_DEPTH = 0.12              # m: how far the sensor glyph's outline reaches (a symbol, not the field of view)
SELECTED_DEPTH = 0.30             # m: the selected sensor's outline reaches further
GLYPH_FOV = (60.0, 45.0)          # degrees: the glyph's shape when a sensor's imagers cannot be built


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
        ctx.store.pose_changed.connect(lambda _uid: self.changed.emit())
        THEME.changed.connect(lambda *_: self.changed.emit())
        self._mesh_key = None
        self._mesh_draw = None
        self._mesh_version = 0
        self._reported: set = set()
        self._frustum_cache: dict = {}
        self._vnormal_cache: dict = {}

    @property
    def project(self):
        return self.ctx.store.project

    # ---------------------------------------------------------------- geometry
    def robot_models(self) -> list:
        """[(robot, RobotModel, q)] for every robot that compiles, q the store's joint state (else its home)."""
        out = []
        for r in self.project.robots:
            try:
                m = get_model(self.project, r.uid)
            except RobotModelError as e:
                if (r.uid, str(e)) not in self._reported:          # said once, not on every frame
                    self._reported.add((r.uid, str(e)))
                    print('Robot not drawn:', e)
                continue
            q = self.ctx.store.q.get(r.uid)
            out.append((r, m, m.home if q is None or len(q) != m.nq else q))
        return out

    def mesh(self):
        """Every robot's visual meshes posed at store.q, one MeshDraw (cached until q or the project changes)."""
        rm = self.robot_models()
        # the shared models stay the same objects until a robot's files, records or base change (get_model's cache)
        key = tuple((m, q.tobytes()) for _r, m, q in rm)
        if key == self._mesh_key:
            return self._mesh_draw
        tris, nrms, cols, vns = [], [], [], []
        for _r, m, q in rm:
            if not m.n_triangles:
                continue
            t, n, _li = m.posed_triangles(q)
            tris.append(t)
            nrms.append(n)
            cols.append(m.triangle_rgba())
            vns.append(m.pose_vectors(q, self._local_vnormals(m)))
        self._mesh_key = key
        if not tris:
            self._mesh_draw = None
            return None
        self._mesh_version += 1
        rgba = np.concatenate(cols)
        self._mesh_draw = MeshDraw(np.concatenate(tris).astype(np.float32), np.concatenate(nrms).astype(np.float32),
                                   rgba[:, :3].astype(float), rgba[:, 3].astype(float), version=self._mesh_version,
                                   vnormals=np.concatenate(vns))
        return self._mesh_draw

    def _local_vnormals(self, m):
        """The shader's smooth per-corner normals of a robot model's meshes, in each link's own frame: computed once
        per model (merging 580 000 corners of the UR5e takes about 0.25 s) and only rotated at each q."""
        hit = self._vnormal_cache.get(id(m))
        if hit is not None and hit[0] is m:
            return hit[1]
        from .render_gl import vertex_normals
        tl = m.local_triangles()
        fn = np.cross(tl[:, 1] - tl[:, 0], tl[:, 2] - tl[:, 0])
        vn = vertex_normals(tl, fn, group=m.triangle_links())
        if len(self._vnormal_cache) > 8:
            self._vnormal_cache.clear()
        self._vnormal_cache[id(m)] = (m, vn)
        return vn

    def ground(self):
        return None

    def static_sensors(self):
        return [s for s in self.project.variant.sensors if s.enabled and not s.mount.on_link]

    def posed_sensors(self) -> list:
        """[(sensor, R, t)] of every enabled sensor that can be placed: static ones at their mount pose, link-mounted
        ones at the link's pose for the current q composed with the mount pose (skipped while their robot or link is
        missing)."""
        out = []
        for s in self.project.variant.sensors:
            if not s.enabled:
                continue
            p = self._sensor_pose(s)
            if p is not None:
                out.append((s, *p))
        return out

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
        for fr in self.frusta():
            on = sel[:2] == ('sensor', fr['sensor'].uid)
            col = THEME.cam(fr['sensor'].slot, None if fr['index'] == 0 else 0.7)
            w = (1.8 if on else 1.2) if fr['index'] == 0 else 0.8
            out.append(LineDraw(fr['segments'], col, w, segments=True))
        out += [d for d in self._extra(vp) if isinstance(d, LineDraw)]
        return out

    def _extra(self, vp) -> list:
        """Draw items from the registered sources (ctx.overlay_sources: fn(vp) -> [LineDraw | TriDraw]) and the
        current analysis page's overlays."""
        out = []
        for fn in list(getattr(self.ctx, 'overlay_sources', [])):
            try:
                out += list(fn(vp) or [])
            except Exception as ex:                      # noqa: BLE001
                print('3D overlay failed:', ex)
        win = self.ctx.window
        if win is not None and hasattr(win, 'page_overlays'):
            out += win.page_overlays(vp)
        return out

    def _grid_lines(self):
        """The floor grid as lines, for the software renderer (the shader draws its own)."""
        ext, step = 1.0, 0.1
        k = np.arange(-ext, ext + 1e-9, step)
        segs = [[[x, -ext, 0], [x, ext, 0]] for x in k] + [[[-ext, y, 0], [ext, y, 0]] for y in k]
        return [LineDraw(np.array(segs), THEME.color('ink', 0.08), 1.0, segments=True)]

    def _frustum_body(self, s, depth) -> list:
        """[(imager name, apex, border (K, 3), spokes)] in the sensor body frame (cached per device and the library's
        generation (a custom device can change under its id), calibration, stream and depth)."""
        import json

        from ..core import camrender, library
        key = (s.model_id, library.generation(), json.dumps(s.calibration, sort_keys=True, default=str),
               tuple(getattr(s, 'stream', ())), depth)
        hit = self._frustum_cache.get(key)
        if hit is not None:
            return hit
        try:
            mode = tuple(s.stream) if len(getattr(s, 'stream', ())) == 2 else None
            out = [(im.name, *camrender.frustum_body(im.camera, im.pose_in_body, depth))
                   for im in library.sensor_imagers(s, mode)]
        except Exception as e:                           # noqa: BLE001 — a bad calibration: the glyph, said once
            if (s.uid, str(e)) not in self._reported:
                self._reported.add((s.uid, str(e)))
                print(f'{s.name}: its frustum is a glyph:', e)
            hx = np.tan(np.radians(GLYPH_FOV[0] / 2)) * depth
            hy = np.tan(np.radians(GLYPH_FOV[1] / 2)) * depth
            far = np.array([[depth, hx, hy], [depth, -hx, hy], [depth, -hx, -hy], [depth, hx, -hy], [depth, hx, hy]])
            out = [('glyph', np.zeros(3), far, [np.array([np.zeros(3), f]) for f in far[:4]])]
        if len(self._frustum_cache) > 256:
            self._frustum_cache.clear()
        self._frustum_cache[key] = out
        return out

    def frusta(self) -> list:
        """[{'sensor', 'imager', 'index', 'segments' (S, 2, 3) world}] of every enabled, placeable sensor's imagers,
        from their lens models, at the current joint state."""
        out = []
        sel = self.ctx.store.sel
        for s, R, t in self.posed_sensors():
            depth = SELECTED_DEPTH if sel[:2] == ('sensor', s.uid) else FRUSTUM_DEPTH
            for k, (name, apex, border, spokes) in enumerate(self._frustum_body(s, depth)):
                B = border @ R.T + t
                ok = np.isfinite(B).all(axis=1)
                segs = [[a @ R.T + t, b @ R.T + t] for a, b in spokes] if k == 0 else []   # from the imager out
                segs += [[B[i], B[i + 1]] for i in range(len(B) - 1) if ok[i] and ok[i + 1]]
                if segs:
                    out.append({'sensor': s, 'imager': name, 'index': k, 'segments': np.array(segs, float)})
        return out

    def _sensor_pose(self, s):
        """(R, t) of the sensor body frame in the world, or None when its link cannot be posed."""
        R, t = quat_to_matrix(s.mount.pose.quaternion), np.asarray(s.mount.pose.position, float)
        if not s.mount.on_link:
            return R, t
        try:
            m = get_model(self.project, s.mount.robot_uid)
            q = self.ctx.store.q.get(s.mount.robot_uid)
            T = m.link_pose(m.home if q is None or len(q) != m.nq else q, s.mount.parent_frame)
        except (RobotModelError, KeyError):
            return None
        return T[:3, :3] @ R, T[:3, :3] @ t + T[:3, 3]

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
        for s, R, t in self.posed_sensors():
            tri = box_tris((-0.02, -0.045, -0.015), (0.0, 0.045, 0.015), R, t)
            col = np.array(THEME.cam(s.slot).getRgbF())
            out.append(TriDraw(tri, np.tile(col, (len(tri), 1))))
        out += [d for d in self._extra(vp) if isinstance(d, TriDraw)]
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
            return f'{p.variant.name} · no robot yet: Robot ▸ Import robot… · {n} sensors'
        nj = sum(m.nq for _r, m, _q in self.robot_models())
        nr = len(p.robots)
        return (f'{p.variant.name} · {nr} robot{"s" if nr != 1 else ""} · {nj} joint{"s" if nj != 1 else ""} · '
                f'{n} sensors')

    def overlay(self, p, vp):
        ik = getattr(self.ctx, 'ik_drag', None)
        if ik is not None:
            ik.paint(p, vp)

    def home_box(self, vp):
        """What the frame button shows: the robots at q, the task volumes and the sensors; the default 1.2 m cell only
        while there are no robots or volumes (an SO-100 is 0.4 m tall and would be lost in it)."""
        boxes = [v.bounds() for v in self.project.workspace.volumes]
        boxes += [m.bounds(q) for _r, m, q in self.robot_models()]
        if not boxes:
            return np.array([-0.3, -0.6, 0.0]), np.array([0.9, 0.6, 0.8]), 1.05
        boxes += [(t, t) for _s, _R, t in self.posed_sensors()]
        lo = np.min([np.asarray(a, float) for a, _ in boxes], axis=0)
        hi = np.max([np.asarray(b, float) for _, b in boxes], axis=0)
        lo[2] = min(lo[2], 0.0)                                   # keep the floor in view
        return lo, hi, 1.05

    def pick_point(self, vp, pos):
        return None

    # ---------------------------------------------------------------- interaction
    # IK drag (ui/ikdrag.py) sees the mouse first while its mode is on
    def mouse_press(self, vp, e):
        ik = getattr(self.ctx, 'ik_drag', None)
        return bool(ik is not None and ik.press(vp, e))

    def mouse_move(self, vp, e):
        ik = getattr(self.ctx, 'ik_drag', None)
        if ik is not None:
            ik.move(vp, e)

    def mouse_release(self, vp, e):
        ik = getattr(self.ctx, 'ik_drag', None)
        if ik is not None:
            ik.release(vp, e)

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
        ik = getattr(self.ctx, 'ik_drag', None)
        return bool(ik is not None and ik.key(vp, e))

    def drag_enter(self, vp, e):
        from .rigedit import MIME
        return bool(e.mimeData().hasFormat(MIME))

    def drag_move(self, vp, e):
        pass

    def drop(self, vp, e):
        """A library device dropped on the view: a sensor of that model above the floor point under the mouse."""
        from .rigedit import MIME
        win = self.ctx.window
        md = e.mimeData()
        if win is None or not md.hasFormat(MIME) or not hasattr(win, 'library'):
            return
        mid = bytes(md.data(MIME)).decode('utf-8', 'ignore')
        pos = e.position()
        g = vp.ground_at(pos.x(), pos.y(), 0.0)
        win.library.add_device(mid, at=None if g is None else (float(g[0]), float(g[1])))
        e.acceptProposedAction()
