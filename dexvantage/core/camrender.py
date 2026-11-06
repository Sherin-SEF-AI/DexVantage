"""A sensor's simulated image: the posed robots, the task objects and the surfaces seen through the imager's own lens
model (core/optics CameraModel), for the Camera view (ui/camview.py) and report images.

    gather(project, q, sensor=None, colors=None) -> Geometry      world triangles with a colour, normal and owner kind
    optical_pose(project, sensor, imager, q) -> frames.Pose        the imager's optical frame in the world
    render(geometry, camera, pose, width, height, ...) -> Render   colour, distance and owner per pixel
    raster(tris_cam, fx, fy, cx, cy, width, height, near) -> (z, tri)   the pinhole z-buffer underneath
    project_world(camera, pose, points) -> (uv, ok, dist)          world points to the imager's pixels

Ported from Vantage ui/camview.py: the idea of shading each pixel from its own ray (unproject every pixel through the
real lens model, look the ray up in a distance map, shade it). What changed: Vantage looked rays up in the cube map of
a static car body; here the scene is re-posed for every joint vector, so each image is rasterised again.

How an image is made (the lens decides the path):
- pinhole: the triangles are rasterised straight into the image (OpenCV's integer pixel centres; the triangles are
  tested exactly per pixel centre ray, so a triangle crossing the near plane is drawn correctly);
- a distorted or wide lens whose rays stay within BASE_MAX_DEG of the axis (Brown-Conrady, a mild Kannala-Brandt):
  a pinhole base image covering every pixel's ray is rasterised, and each pixel of the real image looks its own
  ray up in it (nearest base pixel; the base has at least the real image's density at its centre);
- a fisheye beyond that: the world-axis cube faces the rays need are rasterised (core/occlusion.py FACES and
  texel_index, ported from Vantage), and each pixel looks its ray up there.
In every path the pixel's distance is recomputed exactly along its own ray to the plane of the triangle found, and the
shading is |n · ray| (a head light at the camera), so the look-up only decides which triangle a pixel sees.

The floor (z = 0, the primary robot's base plane, as the 3D view draws it) is shown beyond the geometry with a grid of
GRID_M (a view aid, not geometry: it occludes nothing in the studies).

The rule for a sensor's own surroundings is scenegeom.sensor_occluders': the triangles of the link a sensor is mounted
on stay (a wrist camera sees its gripper), except those with every corner within OWN_RADIUS_M of the imager (its
seat). Sensor housings are not drawn (Phase 2, as in scenegeom).

Known limits: one triangle per pixel (no anti-aliasing: edges are as sharp as the pixel grid); the base or cube image
is sampled at its nearest pixel, so edges move by up to half a base pixel; no shadows, textures or exposure (this is a
geometric view, not a photometric one: Phase 5 adds lighting); materials are drawn in their model colours; the floor
grid is drawn in a pixel's colour, not as lines of constant width.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .frames import Pose, quat_to_matrix
from .scenegeom import OWN_RADIUS_M, object_triangles, sensor_world_pose, surface_triangles

NEAR_M = 0.005                    # default near clip (m), as core/occlusion.py
BASE_MAX_DEG = 60.0               # rays within this of the axis: one pinhole base image; beyond it: cube faces
BASE_MAX_SCALE = 2.0              # the base image is at most this many times the real image along each axis
CUBE_N = (64, 1024)               # cube face size bounds (texels)
CHUNK = 1_500_000                 # candidate pixels tested at once
GRID_M = 0.1                      # floor grid spacing (m), a view aid
FLOOR_FAR_M = 6.0                 # the floor is drawn out to this distance (m)

KIND_NONE, KIND_ROBOT, KIND_OBJECT, KIND_SURFACE, KIND_FLOOR = 0, 1, 2, 3, 4
KIND_NAMES = {KIND_NONE: 'nothing', KIND_ROBOT: 'robot', KIND_OBJECT: 'object', KIND_SURFACE: 'surface',
              KIND_FLOOR: 'floor'}

DEFAULT_COLORS = {'object': (0.78, 0.74, 0.66), 'surface': (0.55, 0.53, 0.50), 'floor': (0.33, 0.33, 0.34),
                  'grid': (0.45, 0.45, 0.46), 'sky': (0.16, 0.17, 0.19)}


# ====================================================================== the scene
@dataclass
class Geometry:
    """World triangles with what they are: rgb (M, 3) 0..1, normal (M, 3) unit, kind (M,) KIND_*, owner (M,) index
    into owners ('robot:<uid>:<link>', 'object:<uid>', 'surface:<uid>')."""
    tris: np.ndarray
    rgb: np.ndarray
    normal: np.ndarray
    kind: np.ndarray
    owner: np.ndarray
    owners: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def __len__(self):
        return len(self.tris)

    def select(self, keep) -> 'Geometry':
        k = np.asarray(keep, bool)
        return Geometry(self.tris[k], self.rgb[k], self.normal[k], self.kind[k], self.owner[k], list(self.owners),
                        list(self.notes))


def _normals(t: np.ndarray) -> np.ndarray:
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return np.where(ln > 1e-20, n / np.maximum(ln, 1e-30), np.array([0.0, 0.0, 1.0])).astype(np.float32)


def _q_of(model, q, uid):
    v = (q or {}).get(uid)
    if v is None or len(v) != model.nq:
        return model.home
    return np.asarray(v, float)


def gather(project, q: dict | None = None, *, robots: bool = True, objects: bool = True, surfaces: bool = True,
           colors: dict | None = None) -> Geometry:
    """Every visible triangle of the project at joint state q (robot uid -> joint vector; others at home): the robots'
    visual meshes in their model colours, the task objects and the surfaces in colors['object'] / colors['surface']
    (RGB 0..1; defaults DEFAULT_COLORS)."""
    from .robot.model import RobotModelError, get_model
    col = dict(DEFAULT_COLORS, **(colors or {}))
    T, C, N, K, O, owners, notes = [], [], [], [], [], [], []

    def add(owner, tri, rgb, kind, nrm=None):
        tri = np.asarray(tri, np.float32).reshape(-1, 3, 3)
        if not len(tri):
            return
        owners.append(owner)
        T.append(tri)
        C.append(np.broadcast_to(np.asarray(rgb, np.float32).reshape(-1, 3), (len(tri), 3)))
        N.append(_normals(tri) if nrm is None else np.asarray(nrm, np.float32))
        K.append(np.full(len(tri), kind, np.int8))
        O.append(np.full(len(tri), len(owners) - 1, np.int32))

    if robots:
        for r in project.robots:
            try:
                m = get_model(project, r.uid)
            except RobotModelError as e:
                notes.append(f'{r.name} is not drawn: {e}')
                continue
            if not m.n_triangles:
                continue
            tri, nrm, li = m.posed_triangles(_q_of(m, q, r.uid))
            rgba = m.triangle_rgba()
            # one owner per link, as scenegeom names them; triangles keep the model's order
            base = len(owners)
            owners.extend(f'robot:{r.uid}:{n}' for n in m.link_names)
            T.append(tri)
            C.append(rgba[:, :3].astype(np.float32))
            N.append(nrm)
            K.append(np.full(len(tri), KIND_ROBOT, np.int8))
            O.append((li + base).astype(np.int32))
    if objects:
        for o in project.workspace.objects:
            t = object_triangles(o, project)
            if not len(t):
                notes.append(f'{o.name} is not drawn: its {o.shape} has no size or mesh.')
            add(f'object:{o.uid}', t, col['object'], KIND_OBJECT)
    if surfaces:
        for k, s in enumerate(project.workspace.surfaces):
            if isinstance(s, dict):
                add(f'surface:{s.get("uid") or k}', surface_triangles(s), col['surface'], KIND_SURFACE)
    if not T:
        z = np.zeros((0, 3), np.float32)
        return Geometry(np.zeros((0, 3, 3), np.float32), z, z, np.zeros(0, np.int8), np.zeros(0, np.int32), owners,
                        notes)
    return Geometry(np.concatenate(T), np.concatenate(C), np.concatenate(N), np.concatenate(K), np.concatenate(O),
                    owners, notes)


def without_seat(geom: Geometry, project, sensor, origins, own_radius: float = OWN_RADIUS_M) -> Geometry:
    """geom less the triangles of the sensor's mount link that lie wholly within own_radius of any of origins (the
    sensor's seat, scenegeom.sensor_occluders' rule)."""
    if not sensor.mount.on_link or own_radius <= 0 or not len(geom):
        return geom
    own = f'robot:{sensor.mount.robot_uid}:{sensor.mount.parent_frame}'
    ids = [i for i, o in enumerate(geom.owners) if o == own]
    if not ids:
        return geom
    idx = np.nonzero(np.isin(geom.owner, ids))[0]
    if not len(idx):
        return geom
    t = geom.tris[idx].astype(float)
    near = np.zeros(len(idx), bool)
    for o in np.asarray(origins, float).reshape(-1, 3):
        near |= (np.linalg.norm(t - o, axis=2) <= own_radius).all(axis=1)
    if not near.any():
        return geom
    keep = np.ones(len(geom), bool)
    keep[idx[near]] = False
    return geom.select(keep)


def optical_pose(project, sensor, imager, q: dict | None = None) -> Pose:
    """The imager's optical frame in the world: the sensor body pose (scenegeom.sensor_world_pose) composed with the
    imager's pose in the body (library.sensor_imagers)."""
    return sensor_world_pose(project, sensor, q) @ imager.pose_in_body


# ====================================================================== the rasteriser
def raster(tris_cam, fx: float, fy: float, cx: float, cy: float, width: int, height: int, near: float = NEAR_M):
    """Pinhole z-buffer of triangles (M, 3, 3) given in a camera frame (z forward, x right, y down): (z (H, W)
    float64 depth along the axis, inf = nothing; tri (H, W) int64 index of the triangle seen, -1 = nothing).

    Each pixel centre (u, v) (integers, OpenCV) has the ray d = ((u − cx)/fx, (v − cy)/fy, 1); it hits the triangle
    (a0, a1, a2) iff the triple products d · (aᵢ × aᵢ₊₁) all have the sign of a0 · (a1 × a2), at depth
    z = a0 · (a1 × a2) / (n · d) with n = (a1 − a0) × (a2 − a0) (the test of core/occlusion.py raster_numpy, ported
    to an arbitrary pinhole). The candidate pixels of a triangle are those in the bounding box of its corners in front
    of the near plane and of its edges' crossings of that plane."""
    W, H = int(width), int(height)
    zb = np.full(H * W, np.inf)
    ib = np.full(H * W, -1, np.int64)
    rel = np.asarray(tris_cam, np.float64).reshape(-1, 3, 3)
    if not len(rel):
        return zb.reshape(H, W), ib.reshape(H, W)
    z = rel[:, :, 2]
    front = z >= near
    keep = front.any(axis=1)
    if not keep.any():
        return zb.reshape(H, W), ib.reshape(H, W)
    idx0 = np.nonzero(keep)[0]
    rel, z, front = rel[idx0], z[idx0], front[idx0]
    x, y = rel[:, :, 0], rel[:, :, 1]
    with np.errstate(divide='ignore', invalid='ignore'):
        pu = [np.where(front[:, k], fx * x[:, k] / z[:, k] + cx, np.nan) for k in range(3)]
        pv = [np.where(front[:, k], fy * y[:, k] / z[:, k] + cy, np.nan) for k in range(3)]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            cr = front[:, a] != front[:, b]
            s = (near - z[:, a]) / (z[:, b] - z[:, a])
            pu.append(np.where(cr, fx * (x[:, a] + s * (x[:, b] - x[:, a])) / near + cx, np.nan))
            pv.append(np.where(cr, fy * (y[:, a] + s * (y[:, b] - y[:, a])) / near + cy, np.nan))
    PU, PV = np.stack(pu, axis=1), np.stack(pv, axis=1)
    big = 4.0 * (W + H) + 8
    with np.errstate(invalid='ignore'):
        c0 = np.clip(np.ceil(np.clip(np.nanmin(PU, axis=1), -big, big)), 0, W).astype(np.int64)
        c1 = np.clip(np.floor(np.clip(np.nanmax(PU, axis=1), -big, big)), -1, W - 1).astype(np.int64)
        r0 = np.clip(np.ceil(np.clip(np.nanmin(PV, axis=1), -big, big)), 0, H).astype(np.int64)
        r1 = np.clip(np.floor(np.clip(np.nanmax(PV, axis=1), -big, big)), -1, H - 1).astype(np.int64)
    w, h = c1 - c0 + 1, r1 - r0 + 1
    ok = (w > 0) & (h > 0)
    if not ok.any():
        return zb.reshape(H, W), ib.reshape(H, W)
    sel = np.nonzero(ok)[0]
    rel, c0, r0, w, h = rel[sel], c0[sel], r0[sel], w[sel], h[sel]
    tri_id = idx0[sel]
    a0, a1, a2 = rel[:, 0], rel[:, 1], rel[:, 2]
    Cx = np.stack([np.cross(a0, a1), np.cross(a1, a2), np.cross(a2, a0), np.cross(a1 - a0, a2 - a0)], axis=1)
    D = np.einsum('ij,ij->i', a0, Cx[:, 1])
    area = np.linalg.norm(Cx[:, 3], axis=1)
    scale = np.linalg.norm(rel, axis=2).max(axis=1)
    good = (area > 1e-18) & (np.abs(D) > 1e-12 * area * np.maximum(scale, 1e-9))     # not edge-on or degenerate
    if not good.all():
        g = np.nonzero(good)[0]
        Cx, D, c0, r0, w, h, tri_id = Cx[g], D[g], c0[g], r0[g], w[g], h[g], tri_id[g]
    if not len(D):
        return zb.reshape(H, W), ib.reshape(H, W)
    Cx[:, :3] *= np.sign(D)[:, None, None]
    # d = (a, b, 1) with a = (u − cx)/fx, b = (v − cy)/fy: each product is C·d = Cz + a Cx + b Cy
    cnt = w * h
    cum = np.cumsum(cnt)
    s = 0
    while s < len(D):
        base = int(cum[s - 1]) if s else 0
        e = max(s + 1, int(np.searchsorted(cum, base + CHUNK, 'right')))
        cc = cnt[s:e]
        k = int(cc.sum())
        ti = np.repeat(np.arange(s, e), cc)
        local = np.arange(base, base + k, dtype=np.int64) - np.repeat(cum[s:e] - cc, cc)
        col = c0[ti] + local % w[ti]
        row = r0[ti] + local // w[ti]
        a = (col - cx) / fx
        b = (row - cy) / fy
        g = Cx[ti]
        val = g[:, :, 2] + a[:, None] * g[:, :, 0] + b[:, None] * g[:, :, 1]
        inside = (val[:, 0] >= 0) & (val[:, 1] >= 0) & (val[:, 2] >= 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = D[ti] / val[:, 3]
        hit = inside & (t >= near) & np.isfinite(t)
        if hit.any():
            pix = row[hit] * W + col[hit]
            th = t[hit]
            np.minimum.at(zb, pix, th)
            win = th <= zb[pix]
            ib[pix[win]] = tri_id[ti[hit][win]]
        s = e
    return zb.reshape(H, W), ib.reshape(H, W)


# ====================================================================== the image
@dataclass
class Render:
    """One simulated image: rgb (H, W, 3) uint8, dist (H, W) float32 metres along each pixel's ray (inf: nothing,
    the floor included), kind (H, W) int8 (KIND_*), tri (H, W) int64 triangle index (-1: none), valid (H, W) bool
    (the pixel has a ray in the lens model), the camera model at this size, how it was made and the time taken."""
    rgb: np.ndarray
    dist: np.ndarray
    kind: np.ndarray
    tri: np.ndarray
    valid: np.ndarray
    camera: object
    path: str
    seconds: float
    triangles: int

    @property
    def coverage(self) -> dict:
        """The fraction of the valid pixels showing each kind of thing."""
        n = max(int(self.valid.sum()), 1)
        return {KIND_NAMES[k]: float(((self.kind == k) & self.valid).sum()) / n for k in KIND_NAMES}


def _path_for(cam, rays_ok) -> str:
    if cam.model == 'pinhole':
        return 'pinhole'
    if not rays_ok.size:
        return 'pinhole'
    return 'base' if rays_ok.max() <= math.radians(BASE_MAX_DEG) else 'cube'


def render(geom: Geometry, camera, pose: Pose, width: int | None = None, height: int | None = None, *,
           colors: dict | None = None, floor: bool = True, near: float = NEAR_M) -> Render:
    """The image of geom seen by camera (an optics.CameraModel) at pose (its optical frame in the world), at
    width × height (default the camera's own size; the lens is scaled with CameraModel.scaled)."""
    import time
    t0 = time.perf_counter()
    col = dict(DEFAULT_COLORS, **(colors or {}))
    W = int(width or camera.width)
    H = int(height or camera.height)
    cam = camera if (W, H) == (camera.width, camera.height) else camera.scaled(W, H)
    R = quat_to_matrix(pose.quaternion)                   # optical -> world
    o = np.asarray(pose.position, float)
    rel_w = geom.tris.astype(np.float64) - o              # world-aligned, relative to the centre
    if cam.model == 'pinhole':
        tc = rel_w @ R                                    # (R^T x) per vertex: world -> optical
        zb, tri = raster(tc, cam.fx, cam.fy, cam.cx, cam.cy, W, H, near)
        us, vs = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
        d = np.stack([(us - cam.cx) / cam.fx, (vs - cam.cy) / cam.fy, np.ones_like(us)], axis=-1)
        rays = d / np.linalg.norm(d, axis=-1, keepdims=True)
        valid = np.ones((H, W), bool)
        path = 'pinhole'
    else:
        rays, valid = cam.unproject(np.stack(np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float)),
                                             axis=-1), with_ok=True)
        rays = np.where(valid[..., None], rays, np.array([0.0, 0.0, 1.0]))
        theta = np.arccos(np.clip(rays[..., 2][valid], -1, 1))
        path = _path_for(cam, theta)
        if path == 'base':
            tc = rel_w @ R
            rv = rays[valid]
            ax = float(np.max(np.abs(rv[:, 0] / rv[:, 2]))) + 1e-9
            ay = float(np.max(np.abs(rv[:, 1] / rv[:, 2]))) + 1e-9
            f = min(max(cam.fx, cam.fy), BASE_MAX_SCALE * W / (2 * ax), BASE_MAX_SCALE * H / (2 * ay))
            bw, bh = int(math.ceil(2 * f * ax)) + 3, int(math.ceil(2 * f * ay)) + 3
            bcx, bcy = (bw - 1) / 2, (bh - 1) / 2
            _zb, btri = raster(tc, f, f, bcx, bcy, bw, bh, near)
            with np.errstate(divide='ignore', invalid='ignore'):
                bu = np.clip(np.rint(f * rays[..., 0] / rays[..., 2] + bcx), 0, bw - 1).astype(np.int64)
                bv = np.clip(np.rint(f * rays[..., 1] / rays[..., 2] + bcy), 0, bh - 1).astype(np.int64)
            tri = np.where(valid, btri[bv, bu], -1)
        else:
            from .occlusion import FACES, texel_index
            rw = rays.reshape(-1, 3) @ R.T                 # world directions
            fx_n = float(max(cam.fx, cam.fy))
            n = int(np.clip(math.ceil(2 * fx_n), *CUBE_N))
            fidx, rr, cc = texel_index(rw, n)
            tri = np.full(H * W, -1, np.int64)
            vflat = valid.reshape(-1)
            for fi in np.unique(fidx[vflat]):
                fw, rt, up = FACES[int(fi)]
                B = np.stack([rt, -up, fw])               # world -> face camera (x right, y down, z forward)
                _zb, ft = raster(rel_w @ B.T, n / 2, n / 2, n / 2 - 0.5, n / 2 - 0.5, n, n, near)
                m = vflat & (fidx == fi)
                tri[m] = ft[rr[m], cc[m]]
            tri = tri.reshape(H, W)
    # exact distance along each pixel's own ray to the plane of the triangle it sees
    rw = rays @ R.T                                       # (H, W, 3) world directions
    hit = tri >= 0
    dist = np.full((H, W), np.inf)
    kind = np.zeros((H, W), np.int8)
    shade = np.zeros((H, W))
    rgb = np.empty((H, W, 3), np.float32)
    rgb[:] = np.asarray(col['sky'], np.float32)
    if hit.any():
        ti = tri[hit]
        nrm = geom.normal[ti].astype(np.float64)
        a0 = rel_w[ti, 0]
        dn = np.einsum('ij,ij->i', nrm, rw[hit])
        with np.errstate(divide='ignore', invalid='ignore'):
            dd = np.einsum('ij,ij->i', nrm, a0) / dn
        dd = np.where(np.isfinite(dd) & (dd > 0), dd, np.linalg.norm(a0, axis=1))
        dist[hit] = dd
        kind[hit] = geom.kind[ti]
        shade[hit] = np.abs(dn)
        rgb[hit] = geom.rgb[ti] * (0.30 + 0.70 * shade[hit])[:, None]
    if floor:
        dz = rw[..., 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            tf = np.where(dz < -1e-6, -o[2] / dz, np.inf)
        fl = valid & (tf > 0) & (tf < FLOOR_FAR_M) & (tf < dist) & (o[2] > 0)
        if fl.any():
            P = o + rw[fl] * tf[fl][:, None]
            pix = 1.0 / max(cam.fx, cam.fy, 1e-9)
            half = 0.7 * tf[fl] * pix / np.maximum(np.abs(dz[fl]), 0.15)
            gx = np.abs((P[:, 0] / GRID_M + 0.5) % 1.0 - 0.5) * GRID_M
            gy = np.abs((P[:, 1] / GRID_M + 0.5) % 1.0 - 0.5) * GRID_M
            # lines closer together than about 5 px are left out (they would read as a flat tone)
            apart = GRID_M * np.maximum(np.abs(dz[fl]), 0.15) / (tf[fl] * pix) > 5.0
            on = ((gx < half) | (gy < half)) & apart
            fade = np.clip(1.0 - tf[fl] / FLOOR_FAR_M, 0.0, 1.0)[:, None]
            base = np.where(on[:, None], np.asarray(col['grid'], np.float32), np.asarray(col['floor'], np.float32))
            rgb[fl] = base * fade + np.asarray(col['sky'], np.float32) * (1 - fade)
            kind[fl] = KIND_FLOOR
    rgb[~valid] = 0.0
    out = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Render(out, dist.astype(np.float32), kind, tri, valid, cam, path, time.perf_counter() - t0, len(geom))


# ====================================================================== projecting world points
def project_world(camera, pose: Pose, points):
    """World points (N, 3) -> (uv (N, 2) in camera's pixels, ok (N,) in the image, distance (N,) m from the centre)."""
    P = np.asarray(points, float).reshape(-1, 3)
    R = quat_to_matrix(pose.quaternion)
    rel = P - np.asarray(pose.position, float)
    uv, ok = camera.project(rel @ R)
    return uv, ok, np.linalg.norm(rel, axis=1)


def densify(polyline, step: float) -> np.ndarray:
    """A world polyline (K, 3) with points added so no segment is longer than step (m)."""
    P = np.asarray(polyline, float).reshape(-1, 3)
    if len(P) < 2:
        return P
    out = [P[:1]]
    for a, b in zip(P[:-1], P[1:]):
        k = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        out.append(a + (b - a) * (np.arange(1, k + 1) / k)[:, None])
    return np.concatenate(out)


def visible_polylines(render_: Render, camera, pose: Pose, polyline, step: float = 0.01, tol: float = 0.01) -> list:
    """The parts of a world polyline the render sees (in front of what it shows, within tol (m) plus 1 %), as a list
    of (K, 2) arrays in camera's pixel coordinates (camera at any size: the depth look-up scales)."""
    P = densify(polyline, step)
    uv, ok, dist = project_world(camera, pose, P)
    if render_ is not None:
        sx = render_.dist.shape[1] / camera.width
        sy = render_.dist.shape[0] / camera.height
        with np.errstate(invalid='ignore'):
            iu = np.clip(np.rint((uv[:, 0] + 0.5) * sx - 0.5), 0, render_.dist.shape[1] - 1)
            iv = np.clip(np.rint((uv[:, 1] + 0.5) * sy - 0.5), 0, render_.dist.shape[0] - 1)
        good = ok & np.isfinite(iu) & np.isfinite(iv)
        dz = np.full(len(P), np.inf)
        dz[good] = render_.dist[iv[good].astype(int), iu[good].astype(int)]
        ok = good & (dist <= dz * 1.01 + tol)
    out, cur = [], []
    for k in range(len(P)):
        if ok[k]:
            cur.append(uv[k])
        elif cur:
            if len(cur) > 1:
                out.append(np.array(cur))
            cur = []
    if len(cur) > 1:
        out.append(np.array(cur))
    return out


def box_polylines(lo, hi) -> list:
    """The 12 edges of an axis-aligned box as world polylines."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    idx = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    return [c[[a, b]] for a, b in idx]


# ====================================================================== frusta
def image_border(camera, per_edge: int = 16):
    """(K, 3) unit rays (optical frame) along the image's outer edge (u = −0.5 … W − 0.5, the edges the FOV is measured
    to), clockwise from the top-left corner, NaN rows where the edge has no ray (outside a circular fisheye's image
    circle)."""
    W, H = camera.width - 0.5, camera.height - 0.5
    t = np.linspace(0, 1, per_edge, endpoint=False)
    a, b = -0.5 + t * (W + 0.5), -0.5 + t * (H + 0.5)
    uv = np.concatenate([np.c_[a, np.full_like(t, -0.5)], np.c_[np.full_like(t, W), b],
                         np.c_[W - 0.5 - a, np.full_like(t, H)], np.c_[np.full_like(t, -0.5), H - 0.5 - b],
                         [[-0.5, -0.5]]])
    return camera.unproject(uv)


def frustum_body(camera, pose_in_body: Pose, depth: float, per_edge: int = 16) -> tuple:
    """A frustum outline in the sensor body frame: (apex (3,), border (K, 3) with NaN gaps, corner rays [(2, 3)]).
    The border lies on the plane depth (m) in front of the imager while every ray is within 75° of the axis, else on
    the sphere of that radius (a fisheye's curved cone)."""
    rays = image_border(camera, per_edge)
    ok = np.isfinite(rays).all(axis=1)
    wide = ok.any() and float(np.max(np.arccos(np.clip(rays[ok, 2], -1, 1)))) > math.radians(75)
    with np.errstate(divide='ignore', invalid='ignore'):
        pts = rays * (np.full(len(rays), float(depth)) if wide else depth / np.where(ok, rays[:, 2], np.nan))[:, None]
    R = quat_to_matrix(pose_in_body.quaternion)
    apex = np.asarray(pose_in_body.position, float)
    border = pts @ R.T + apex
    k = per_edge
    corners = [0, k, 2 * k, 3 * k]
    spokes = [np.array([apex, border[i]]) for i in corners if ok[i]]
    if len(spokes) < 4:                                  # a circular fisheye: the edge midpoints instead
        spokes += [np.array([apex, border[i + k // 2]]) for i in corners if ok[i + k // 2]]
    return apex, border, spokes
