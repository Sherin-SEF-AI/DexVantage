"""The scene's geometry as world-frame triangles, for occlusion (core/occlusion.py, core/raycast.py) and the studies.

    scene_occluders(project, q, *, robots=True, objects=True, surfaces=True, exclude_links=(), decimate=None)
        -> Occluders(tris (M,3,3) float32 world, owner (M,) int32, owners [str])
    sensor_world_pose(project, sensor, q) -> frames.Pose of the sensor body frame in the world
    sensor_occluders(project, sensor, q, ...) -> the occluders a sensor sees past (its own surroundings left out)
    decimate(occluders, cell_m) -> Occluders simplified by vertex clustering (for occlusion only)
    object_triangles(obj, project), surface_triangles(surface) -> (M,3,3) world triangles

What is in the scene:
- every robot's visual meshes posed at q[robot.uid] (RobotModel.posed_triangles; a robot missing from q is at its
  home pose), owner 'robot:<uid>:<link>' per link. A robot whose files do not compile is left out and named in
  Occluders.notes.
- the task objects (project.workspace.objects), owner 'object:<uid>': 'box' with size = its edges (m); 'cylinder'
  with size = (radius, height, -), the height the full length along the object's z axis, centred on its pose;
  'sphere' with size = (radius, -, -); 'mesh' with mesh_key naming project.meshes entries: an (M, 3, 3) triangle
  array, or (V, 3) vertices with '<key>/faces' (F, 3) indices, or a (3K, 3) triangle soup; in the object's frame,
  metres, its size ignored. Cylinders and spheres are tessellated as core/robot/model.py tessellates MuJoCo's
  primitives (24 segments around, 12 rings pole to pole).
- the surfaces (project.workspace.surfaces: {'uid', 'name', 'z', 'size': [x, y], 'center': [x, y] (optional,
  default the world origin), 'thickness' (optional, m)}), owner 'surface:<uid>': a rectangle at height z (the
  table's top), or a slab from z − thickness to z when a thickness is given.

The rule for a sensor's own surroundings (sensor_occluders): the sensor's own housing is not modelled (no sensor
housings are occluders in Phase 2); the triangles of the link a sensor is mounted on (owner 'robot:<uid>:<link>' of
its mount) are kept, so a wrist camera sees its own gripper and flange, except those with every corner within
own_radius (default OWN_RADIUS_M = 25 mm, a default) of the sensor's origin (the bracket face or the link surface the
sensor sits on); exclude_mount_link=True leaves the whole mount link out. The cube map's near clip (core/occlusion.py,
5 mm along each face axis) also ignores anything closer than that to an imager.

Known limits: the robots' visual meshes are the occluders (not the collision geometry), with MuJoCo's equality
constraints not applied (the 2F-85 linkage, docs/BUILD_STATE.md); the tessellation of cylinders and spheres is
polygonal (a 24-gon is 1.1 % inside the circle at its edge midpoints); vertex clustering moves vertices by up to
half a cell diagonal and can close thin gaps.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from .frames import Pose, quat_to_matrix

OWN_RADIUS_M = 0.025          # default: triangles of the mount link wholly this close to the sensor are its seat


@dataclass
class Occluders:
    tris: np.ndarray                                    # (M, 3, 3) float32, world frame, metres
    owner: np.ndarray                                   # (M,) int32 index into owners
    owners: list = field(default_factory=list)          # 'robot:<uid>:<link>' | 'object:<uid>' | 'surface:<uid>'
    notes: list = field(default_factory=list)           # what was left out and why
    info: dict = field(default_factory=dict)            # e.g. decimation figures
    _digest: str = ''

    def __post_init__(self):
        self.tris = np.ascontiguousarray(self.tris, np.float32).reshape(-1, 3, 3)
        self.owner = np.ascontiguousarray(self.owner, np.int32).reshape(-1)
        if len(self.owner) != len(self.tris):
            raise ValueError('each occluder triangle needs one owner')

    def __len__(self):
        return len(self.tris)

    def digest(self) -> str:
        """A content digest of the triangles (owners do not change what occludes)."""
        if not self._digest:
            self._digest = hashlib.blake2b(self.tris.tobytes(), digest_size=16).hexdigest()
        return self._digest

    def select(self, mask) -> 'Occluders':
        """The triangles where mask is True; owners left without a triangle are dropped from owners."""
        m = np.asarray(mask, bool)
        own = self.owner[m]
        used = np.unique(own)
        remap = np.full(max(len(self.owners), 1), -1, np.int32)
        remap[used] = np.arange(len(used), dtype=np.int32)
        return Occluders(self.tris[m], remap[own], [self.owners[i] for i in used], list(self.notes),
                         dict(self.info))

    def owner_mask(self, owners) -> np.ndarray:
        """(M,) True for the triangles whose owner is one of owners (owner strings)."""
        want = {i for i, o in enumerate(self.owners) if o in set(owners)}
        return np.isin(self.owner, list(want)) if want else np.zeros(len(self.owner), bool)

    def without(self, owners) -> 'Occluders':
        return self.select(~self.owner_mask(owners))

    def bounds(self):
        if not len(self.tris):
            return np.zeros(3), np.zeros(3)
        v = self.tris.reshape(-1, 3)
        return v.min(axis=0).astype(float), v.max(axis=0).astype(float)

    @staticmethod
    def concat(parts: list) -> 'Occluders':
        """[(owner string, (M,3,3) triangles)] -> one Occluders."""
        owners, tris, own = [], [], []
        for name, t in parts:
            t = np.asarray(t, np.float32).reshape(-1, 3, 3)
            if not len(t):
                continue
            if name not in owners:
                owners.append(name)
            tris.append(t)
            own.append(np.full(len(t), owners.index(name), np.int32))
        if not tris:
            return Occluders(np.zeros((0, 3, 3), np.float32), np.zeros(0, np.int32), owners)
        return Occluders(np.concatenate(tris), np.concatenate(own), owners)


# ====================================================================== primitives
def _box_tris(half) -> np.ndarray:
    from .robot.model import _box, _outward
    V, F = _box(np.asarray(half, float))
    return V[_outward(V, F)]


def _cylinder_tris(r: float, half_h: float) -> np.ndarray:
    from .robot.model import _lathe, _outward
    V, F = _lathe([(0, -half_h), (r, -half_h), (r, half_h), (0, half_h)])
    return V[_outward(V, F)]


def _sphere_tris(r: float) -> np.ndarray:
    from .robot.model import _lathe, _outward, _sphere_profile
    V, F = _lathe(_sphere_profile(r))
    return V[_outward(V, F)]


def _mesh_tris(project, key: str) -> np.ndarray | None:
    meshes = getattr(project, 'meshes', {}) or {}
    a = meshes.get(key)
    if a is None:
        return None
    a = np.asarray(a, float)
    faces = meshes.get(key + '/faces')
    if a.ndim == 3 and a.shape[1:] == (3, 3):
        return a
    if a.ndim == 2 and a.shape[1] == 3:
        if faces is not None:
            return a[np.asarray(faces, np.int64).reshape(-1, 3)]
        if len(a) % 3 == 0:
            return a.reshape(-1, 3, 3)
    return None


def object_triangles(obj, project=None) -> np.ndarray:
    """(M, 3, 3) world triangles of a TaskObject (see the module docstring for the shapes); empty when its shape or
    mesh cannot be read."""
    s = np.asarray(obj.size, float).reshape(-1)
    s = np.r_[s, np.zeros(3)][:3]
    needs = {'box': 3, 'cylinder': 2, 'sphere': 1}.get(obj.shape, 0)
    if needs and not np.all(np.abs(s[:needs]) > 0):
        return np.zeros((0, 3, 3))
    if obj.shape == 'box':
        t = _box_tris(np.abs(s) / 2)
    elif obj.shape == 'cylinder':
        t = _cylinder_tris(abs(s[0]), abs(s[1]) / 2)
    elif obj.shape == 'sphere':
        t = _sphere_tris(abs(s[0]))
    elif obj.shape == 'mesh':
        t = _mesh_tris(project, obj.mesh_key) if project is not None else None
        if t is None:
            return np.zeros((0, 3, 3))
    else:
        return np.zeros((0, 3, 3))
    R, p = quat_to_matrix(obj.pose.quaternion), np.asarray(obj.pose.position, float)
    return t @ R.T + p


def surface_triangles(surface: dict) -> np.ndarray:
    """(M, 3, 3) world triangles of a surface record (a rectangle at its z, or a slab with a thickness)."""
    sz = np.asarray(surface.get('size') or (0.0, 0.0), float).reshape(-1)[:2]
    c = np.asarray(surface.get('center') or (0.0, 0.0), float).reshape(-1)[:2]
    z = float(surface.get('z', 0.0))
    if not (sz > 0).all():
        return np.zeros((0, 3, 3))
    th = float(surface.get('thickness') or 0.0)
    if th > 0:
        lo = np.r_[c - sz / 2, z - th]
        hi = np.r_[c + sz / 2, z]
        return _box_tris((hi - lo) / 2) + (hi + lo) / 2
    x0, y0 = c - sz / 2
    x1, y1 = c + sz / 2
    a, b, cc, d = [x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]
    return np.array([[a, b, cc], [a, cc, d]], float)


# ====================================================================== the scene
def _q_for(model, q: dict | None, uid: str) -> np.ndarray:
    v = (q or {}).get(uid)
    if v is None:
        return model.home
    v = np.asarray(v, float).reshape(-1)
    if len(v) != model.nq:
        raise ValueError(f'{model.name} has {model.nq} moving joints; q gives {len(v)} values.')
    return v


def _exclusion_set(exclude_links) -> tuple[set, set]:
    """(owner strings, bare link names) from exclude_links: 'robot:<uid>:<link>' strings, (uid, link) pairs or link
    names (which match that link on every robot)."""
    owners, names = set(), set()
    for e in exclude_links or ():
        if isinstance(e, (tuple, list)) and len(e) == 2:
            owners.add(f'robot:{e[0]}:{e[1]}')
        elif isinstance(e, str) and e.startswith('robot:'):
            owners.add(e)
        elif isinstance(e, str) and e:
            names.add(e)
    return owners, names


def scene_occluders(project, q: dict | None = None, *, robots: bool = True, objects: bool = True,
                    surfaces: bool = True, exclude_links=(), decimate: float | None = None) -> Occluders:
    """Every occluder of the project at joint state q (robot uid -> joint vector; robots not in q at home).
    exclude_links: owner strings 'robot:<uid>:<link>', (uid, link) pairs or link names left out. decimate: a
    vertex-clustering cell (m) applied to the result (occlusion only; Occluders.info says how many triangles)."""
    from .robot.model import RobotModelError, get_model
    parts, notes = [], []
    ex_owner, ex_name = _exclusion_set(exclude_links)
    if robots:
        for r in project.robots:
            try:
                m = get_model(project, r.uid)
            except RobotModelError as e:
                notes.append(f'{r.name} is not an occluder: {e}')
                continue
            if not m.n_triangles:
                continue
            tri, _n, li = m.posed_triangles(_q_for(m, q, r.uid))
            order = np.argsort(li, kind='stable')
            li_s = li[order]
            cuts = np.flatnonzero(np.diff(li_s)) + 1
            for grp in np.split(order, cuts):
                link = m.link_names[int(li[grp[0]])]
                own = f'robot:{r.uid}:{link}'
                if own in ex_owner or link in ex_name:
                    continue
                parts.append((own, tri[grp]))
    if objects:
        for o in project.workspace.objects:
            t = object_triangles(o, project)
            if not len(t):
                notes.append(f'Object {o.name} is not an occluder: its {o.shape} has no size or mesh to draw.')
                continue
            parts.append((f'object:{o.uid}', t))
    if surfaces:
        for k, s in enumerate(project.workspace.surfaces):
            if not isinstance(s, dict):
                continue
            t = surface_triangles(s)
            if len(t):
                parts.append((f'surface:{s.get("uid") or k}', t))
    occ = Occluders.concat(parts)
    occ.notes = notes
    if decimate:
        occ = globals()['decimate'](occ, float(decimate))
    return occ


def sensor_world_pose(project, sensor, q: dict | None = None) -> Pose:
    """The sensor body frame in the world: the mount pose for static mounts, the link pose at q @ the mount pose for
    link mounts. Raises KeyError or RobotModelError when the mount's robot or link cannot be posed."""
    from .robot.model import RobotModelError, get_model
    mp = sensor.mount.pose
    if not sensor.mount.on_link:
        return Pose(mp.position.copy(), mp.quaternion.copy())
    rob = project.robot(sensor.mount.robot_uid)
    if rob is None:
        raise RobotModelError(f'{sensor.name} is mounted on a robot that is not in the project.')
    m = get_model(project, rob.uid)
    T = m.link_pose(_q_for(m, q, rob.uid), sensor.mount.parent_frame)
    return Pose.from_matrix(T @ mp.matrix())


def sensor_occluders(project, sensor, q: dict | None = None, *, own_radius: float = OWN_RADIUS_M,
                     exclude_mount_link: bool = False, origins=None, base: Occluders | None = None,
                     **kw) -> Occluders:
    """The occluders a sensor sees past, by the rule in the module docstring. origins: the world points to measure
    own_radius from (default the sensor body origin; pass every imager's centre for a multi-imager sensor). base:
    scene occluders already built for this q (else built here with **kw)."""
    occ = base if base is not None else scene_occluders(project, q, **kw)
    if not sensor.mount.on_link:
        return occ
    own = f'robot:{sensor.mount.robot_uid}:{sensor.mount.parent_frame}'
    if own not in occ.owners:
        return occ
    if exclude_mount_link:
        return occ.without([own])
    if own_radius <= 0:
        return occ
    if origins is None:
        origins = sensor_world_pose(project, sensor, q).position[None]
    O = np.asarray(origins, float).reshape(-1, 3)
    on = occ.owner_mask([own])
    idx = np.nonzero(on)[0]
    if not len(idx):
        return occ
    t = occ.tris[idx].astype(float)
    near = np.zeros(len(idx), bool)
    for o in O:
        near |= (np.linalg.norm(t - o, axis=2) <= own_radius).all(axis=1)
    if not near.any():
        return occ
    keep = np.ones(len(occ), bool)
    keep[idx[near]] = False
    out = occ.select(keep)
    out.info = dict(occ.info, own_dropped=int(near.sum()))
    return out


def decimate(occ: Occluders, cell_m: float) -> Occluders:
    """Vertex clustering [Rossignac and Borrel 1993]: every vertex moves to the mean of the vertices in its cubic cell
    of edge cell_m; triangles that collapse (two corners in one cell) and duplicates are dropped. Owners are kept.
    For occlusion only: decimation_report in core/occlusion.py measures the visibility it changes."""
    if not cell_m or cell_m <= 0 or not len(occ):
        return occ
    V = occ.tris.reshape(-1, 3).astype(np.float64)
    key = np.floor(V / cell_m).astype(np.int64)
    _u, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    k = int(inv.max()) + 1
    mean = np.stack([np.bincount(inv, V[:, j], minlength=k) for j in range(3)], axis=1)
    mean /= np.bincount(inv, minlength=k)[:, None]
    ids = inv.reshape(-1, 3)
    good = (ids[:, 0] != ids[:, 1]) & (ids[:, 1] != ids[:, 2]) & (ids[:, 0] != ids[:, 2])
    srt = np.sort(ids, axis=1)
    _s, first = np.unique(srt[good], axis=0, return_index=True)
    sel = np.nonzero(good)[0][np.sort(first)]
    out = Occluders(mean[ids[sel]].astype(np.float32), occ.owner[sel], list(occ.owners), list(occ.notes),
                    dict(occ.info, decimate_cell_m=float(cell_m), triangles_before=int(len(occ)),
                         triangles_after=int(len(sel))))
    return out
