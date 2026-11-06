"""Self-collision and environment distances of a posed robot, computed by MuJoCo.

    self_collision(model, q, margin=0.0)          -> CollisionResult(in_collision, min_distance m, pairs)
    self_collision_batch(model, Q, margin=0.0)    -> (in_collision (N,), min_distance (N,), [pairs per q])
    environment_distance(model, q, margin=0.0)    -> CollisionResult against the static geoms of the robot's own file
    environment_distance_batch(model, Q, ...)     -> as self_collision_batch
    obstacle_distance(project, robot_uid, q=None, margin=0.0) -> CollisionResult against the project's obstacles
                                                     (workspace objects and surfaces) and the file's static geoms
    env_distance(project, robot_uid, q=None)      -> (min_distance m, pair (link, obstacle, distance m) | None)
    obstacles(project)                            -> [Obstacle] the project's obstacles as MuJoCo primitives
    checked_pairs(model, kind='self')             -> [(link a, link b)] the link pairs that are checked

The distance of a geom pair is MuJoCo's signed distance mj_geomDistance (negative when the geoms overlap: the
penetration depth), evaluated for every pair whose bounding spheres (MuJoCo's geom_rbound) come within `distmax`.
min_distance is the smallest signed distance over the checked pairs, capped at distmax (default 0.2 m): a value of
distmax means "at least distmax". A pose is in collision when min_distance < margin. `pairs` lists the link pairs
closer than margin, nearest first, as (link a, link b, distance m); when none is, it holds the single closest pair.

Which geom pairs are checked (self): the colliding geoms of the robot (the model's collision geoms: those with a
non-zero contype or conaffinity when the file marks visual-only geoms that way, else every geom), on bodies other than
MuJoCo's world body, and of those pairs, not

- two geoms of one rigid group of bodies (bodies joined without a moving joint, e.g. URDF fixed joints);
- adjacent links: two rigid groups joined by one joint (they touch at the joint by design);
- pairs the model excludes: MuJoCo's contype/conaffinity rule ((contype₁ & conaffinity₂) | (contype₂ & conaffinity₁)
  = 0) and <contact><exclude body1 body2/>;
- planes and height fields;

while every <contact><pair geom1 geom2/> the file names is checked whatever the rules above say. Environment pairs
are each colliding robot geom with each colliding geom of the world body (floors, tables written into the file),
filtered by the contype/conaffinity rule.

Project obstacles (obstacle_distance, env_distance): the workspace objects (project.workspace.objects) and surfaces
(project.workspace.surfaces) become static MuJoCo geoms in a second, compiled scene: a box object is a box of its edges,
a cylinder a cylinder of its radius and full height along its z axis (centred on its pose, as core/scenegeom.py draws
it), a sphere a sphere, a mesh object a mesh geom of its vertices (MuJoCo collides it as their convex hull); a surface
is a box under its top at z over its size and centre, as thick as its 'thickness', else SURFACE_THICKNESS_M (50 mm, a
default: a point less than 50 mm under a table top without a stated thickness counts as inside the table, one deeper as
under it). Each colliding robot geom that moves with q is copied into that scene on its own mocap body, posed at q from
the robot's model (Robot.base applied), and every robot geom × obstacle pair whose bounding spheres come within distmax
is measured with mj_geomDistance, as above. The robot's links that no moving joint carries (its base, bolted to the
table it stands on) are not paired with obstacles: their distance does not change with q and would read as a collision
for every robot standing on a surface. obstacle_distance also includes the static geoms of the robot's own file
(environment_distance); `pairs` name the obstacle by its name in the project.

Known limits: MuJoCo collides meshes as their convex hulls, so a concave link (a U-shaped bracket, a hand's palm) or a
concave mesh obstacle (a bowl, a shelf) is fuller than its mesh and distances to it are short of the true ones (never
long: the hull contains the mesh); other robots in the project are not obstacles of a robot (only its own file's static
geoms, the workspace objects and the surfaces are); a robot's base is not checked against obstacles (above); a URDF link
named "world" is merged into MuJoCo's world body, so its geoms count as environment; a mesh obstacle whose vertices are
flat (fewer than four not in one plane) cannot be hulled and is left out, named in obstacles()'s notes; distances beyond
distmax are not computed.
"""
from __future__ import annotations

import threading
import weakref
from typing import NamedTuple

import numpy as np

DIST_MAX = 0.2                    # m: distances beyond this are not computed (a default, see the module docstring)
SURFACE_THICKNESS_M = 0.05        # m: a surface without a stated thickness is a slab this thick under its top (default)

_CACHE: 'weakref.WeakKeyDictionary' = weakref.WeakKeyDictionary()
_LOCK = threading.Lock()


class CollisionResult(NamedTuple):
    in_collision: bool
    min_distance: float           # m, signed (negative = penetration), capped at distmax
    pairs: list                   # [(link a, link b, distance m)]


class _Pairs:
    """The geom pairs of one model that are checked, and their link names."""

    def __init__(self, model):
        import mujoco
        m = model.m
        G = mujoco.mjtGeom
        self.m = m
        ng = m.ngeom
        body = np.asarray(m.geom_bodyid, int)
        ctype = np.asarray(m.geom_contype, np.int64)
        caff = np.asarray(m.geom_conaffinity, np.int64)
        nonc = (ctype == 0) & (caff == 0)
        colliding = ~nonc if nonc.any() else np.ones(ng, bool)
        gtype = np.asarray(m.geom_type, int)
        flat = np.isin(gtype, [int(G.mjGEOM_PLANE), int(G.mjGEOM_HFIELD)])
        # rigid groups: a body without joints belongs to its parent's group (unless the parent is the world)
        parent = np.asarray(m.body_parentid, int)
        jnum = np.asarray(m.body_jntnum, int)
        group = np.arange(m.nbody)
        for b in range(1, m.nbody):                    # MuJoCo orders bodies parents first
            if jnum[b] == 0 and parent[b] != 0:
                group[b] = group[parent[b]]
        pgroup = np.array([group[parent[g]] if g > 0 and parent[g] != 0 else -1 for g in range(m.nbody)])
        excl = set()
        for s in np.asarray(m.exclude_signature, np.int64).reshape(-1):
            b1, b2 = int(s) >> 16, int(s) & 0xFFFF
            excl.add((b1, b2))
            excl.add((b2, b1))
        robot = np.flatnonzero(colliding & (body != 0) & ~flat)
        world = np.flatnonzero(colliding & (body == 0))

        def masks_ok(g1, g2):
            return bool((ctype[g1] & caff[g2]) | (ctype[g2] & caff[g1]))

        selfp = []
        for i, g1 in enumerate(robot):
            for g2 in robot[i + 1:]:
                b1, b2 = body[g1], body[g2]
                a, c = group[b1], group[b2]
                if a == c or pgroup[a] == c or pgroup[c] == a:
                    continue
                if (b1, b2) in excl or not masks_ok(g1, g2):
                    continue
                selfp.append((int(g1), int(g2)))
        have = {frozenset(p) for p in selfp}
        for k in range(m.npair):
            g1, g2 = int(m.pair_geom1[k]), int(m.pair_geom2[k])
            if body[g1] != 0 and body[g2] != 0 and frozenset((g1, g2)) not in have:
                selfp.append((g1, g2))
                have.add(frozenset((g1, g2)))
        envp = [(int(g1), int(g2)) for g1 in robot for g2 in world if masks_ok(g1, g2)]
        self.self_pairs = np.array(selfp, int).reshape(-1, 2)
        self.env_pairs = np.array(envp, int).reshape(-1, 2)
        self.rbound = np.asarray(m.geom_rbound, float).copy()
        self.unbounded = flat | (self.rbound <= 0)
        own = {int(model._lb[i]): n for i, n in reversed(list(enumerate(model.link_names)))
               if np.allclose(model._loff[i], np.eye(4))}
        self.link_of_geom = [own.get(int(body[g]), model.body_link[int(body[g])]) for g in range(ng)]


def _pairs(model) -> _Pairs:
    with _LOCK:
        p = _CACHE.get(model)
        if p is None:
            p = _CACHE[model] = _Pairs(model)
        return p


def checked_pairs(model, kind: str = 'self') -> list:
    """[(link a, link b)] the distinct link pairs checked ('self' or 'environment')."""
    P = _pairs(model)
    arr = P.self_pairs if kind == 'self' else P.env_pairs
    out = []
    for g1, g2 in arr:
        t = (P.link_of_geom[g1], P.link_of_geom[g2])
        if t not in out:
            out.append(t)
    return out


def _evaluate(model, P: _Pairs, pairs: np.ndarray, q, margin: float, distmax: float) -> CollisionResult:
    import mujoco
    d = model._kin(model._q(q))
    if not len(pairs):
        return CollisionResult(False, float(distmax), [])
    xp = np.asarray(d.geom_xpos)
    g1, g2 = pairs[:, 0], pairs[:, 1]
    lb = np.linalg.norm(xp[g1] - xp[g2], axis=1) - P.rbound[g1] - P.rbound[g2]
    near = (lb < distmax) | P.unbounded[g1] | P.unbounded[g2]
    idx = np.flatnonzero(near)
    if not len(idx):
        return CollisionResult(False, float(distmax), [])
    dist = np.empty(len(idx))
    m = model.m
    for k, i in enumerate(idx):
        dist[k] = mujoco.mj_geomDistance(m, d, int(g1[i]), int(g2[i]), float(distmax), None)
    order = np.argsort(dist, kind='stable')
    dmin = float(min(dist[order[0]], distmax))
    hit = dist < margin
    names = P.link_of_geom
    if hit.any():
        out, seen = [], set()
        for k in order:
            if not hit[k]:
                break
            a, b = names[g1[idx[k]]], names[g2[idx[k]]]
            if (a, b) not in seen:
                seen.add((a, b))
                out.append((a, b, float(dist[k])))
    else:
        k = order[0]
        out = [(names[g1[idx[k]]], names[g2[idx[k]]], float(min(dist[k], distmax)))] if dist[k] < distmax else []
    return CollisionResult(bool(dmin < margin), dmin, out)


def self_collision(model, q, margin: float = 0.0, distmax: float = DIST_MAX) -> CollisionResult:
    """Whether the robot at q collides with itself (closest pair nearer than margin, m), the smallest signed distance
    between its checked geom pairs (capped at distmax) and the link pairs involved; see the module docstring."""
    P = _pairs(model)
    return _evaluate(model, P, P.self_pairs, q, float(margin), max(float(distmax), float(margin) + 1e-9))


def environment_distance(model, q, margin: float = 0.0, distmax: float = DIST_MAX) -> CollisionResult:
    """As self_collision, between the robot's geoms and the static (world-body) geoms of its own file."""
    P = _pairs(model)
    return _evaluate(model, P, P.env_pairs, q, float(margin), max(float(distmax), float(margin) + 1e-9))


def _batch(fn, model, Q, margin, distmax):
    Q = model.batch_q(Q)
    hit = np.zeros(len(Q), bool)
    dmin = np.empty(len(Q))
    pairs = []
    for k, q in enumerate(Q):
        r = fn(model, q, margin, distmax)
        hit[k], dmin[k] = r.in_collision, r.min_distance
        pairs.append(r.pairs)
    return hit, dmin, pairs


def self_collision_batch(model, Q, margin: float = 0.0, distmax: float = DIST_MAX):
    """(in_collision (N,) bool, min_distance (N,) m, [pairs] per row) for joint vectors Q (N, nq)."""
    return _batch(self_collision, model, Q, margin, distmax)


def environment_distance_batch(model, Q, margin: float = 0.0, distmax: float = DIST_MAX):
    """As self_collision_batch, against the static geoms of the robot's file."""
    return _batch(environment_distance, model, Q, margin, distmax)


# ====================================================================== project obstacles
class Obstacle(NamedTuple):
    """One project obstacle as a MuJoCo primitive, in the world frame (REP-103, metres)."""
    owner: str                    # 'object:<uid>' | 'surface:<uid>' (as core/scenegeom.py names occluders)
    name: str                     # its name in the project
    gtype: str                    # 'box' | 'sphere' | 'cylinder' | 'mesh'
    size: tuple                   # MuJoCo size: box half-edges; sphere (r,); cylinder (r, half-height); mesh ()
    position: np.ndarray          # (3,) world
    quat_wxyz: np.ndarray         # (4,) MuJoCo order
    vertices: np.ndarray | None   # (V, 3) world vertices of a mesh obstacle (its geom at the identity)


def obstacles(project) -> tuple[list, list]:
    """([Obstacle], [notes]) of the project's workspace objects and surfaces (see the module docstring); objects that
    cannot be made into a geom are left out and named in the notes."""
    from ..frames import quat_to_mujoco
    out, notes = [], []
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    for o in project.workspace.objects:
        s = np.r_[np.asarray(o.size, float).reshape(-1), np.zeros(3)][:3]
        pos = np.asarray(o.pose.position, float).copy()
        qw = np.asarray(quat_to_mujoco(o.pose.quaternion), float)
        own = f'object:{o.uid}'
        if o.shape == 'box' and np.all(np.abs(s) > 0):
            out.append(Obstacle(own, o.name, 'box', tuple(np.abs(s) / 2), pos, qw, None))
        elif o.shape == 'sphere' and abs(s[0]) > 0:
            out.append(Obstacle(own, o.name, 'sphere', (abs(s[0]),), pos, qw, None))
        elif o.shape == 'cylinder' and abs(s[0]) > 0 and abs(s[1]) > 0:
            out.append(Obstacle(own, o.name, 'cylinder', (abs(s[0]), abs(s[1]) / 2), pos, qw, None))
        elif o.shape == 'mesh':
            from ..scenegeom import object_triangles
            t = object_triangles(o, project)
            V = np.unique(np.asarray(t, float).reshape(-1, 3), axis=0) if len(t) else np.zeros((0, 3))
            if len(V) < 4 or np.linalg.matrix_rank(V - V.mean(axis=0), tol=1e-9) < 3:
                notes.append(f'Object {o.name} is not an obstacle: its mesh is missing or flat, so it has no hull.')
                continue
            out.append(Obstacle(own, o.name, 'mesh', (), np.zeros(3), ident, V))
        else:
            notes.append(f'Object {o.name} is not an obstacle: its {o.shape} has no size.')
    for k, sf in enumerate(project.workspace.surfaces):
        if not isinstance(sf, dict):
            continue
        sz = np.asarray(sf.get('size') or (0.0, 0.0), float).reshape(-1)[:2]
        if len(sz) < 2 or not (sz > 0).all():
            continue
        c = np.asarray(sf.get('center') or (0.0, 0.0), float).reshape(-1)[:2]
        th = float(sf.get('thickness') or 0.0)
        th = th if th > 0 else SURFACE_THICKNESS_M
        z = float(sf.get('z', 0.0))
        name = str(sf.get('name') or sf.get('uid') or f'surface {k + 1}')
        out.append(Obstacle(f'surface:{sf.get("uid") or k}', name, 'box', (sz[0] / 2, sz[1] / 2, th / 2),
                            np.array([c[0], c[1], z - th / 2]), ident, None))
    return out, notes


def _obstacle_key(obs: list) -> str:
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    for o in obs:
        h.update(f'{o.owner}|{o.gtype}|{o.size}'.encode())
        h.update(np.asarray(o.position, float).tobytes())
        h.update(np.asarray(o.quat_wxyz, float).tobytes())
        if o.vertices is not None:
            h.update(np.ascontiguousarray(o.vertices, float).tobytes())
    return h.hexdigest()


def _moving_bodies(model) -> np.ndarray:
    """(nbody,) True for the bodies an actuated joint (one of q) carries."""
    m = model.m
    moved = np.zeros(m.nbody, bool)
    moved[np.asarray(m.jnt_bodyid, int)[model._jid]] = True
    parent = np.asarray(m.body_parentid, int)
    for b in range(1, m.nbody):                        # parents first
        moved[b] |= moved[parent[b]]
    return moved


class _ObstacleScene:
    """The obstacles as static geoms, and copies of the robot's moving colliding geoms on mocap bodies."""

    def __init__(self, model, obs: list):
        import mujoco
        G = mujoco.mjtGeom
        m = model.m
        P = _pairs(model)
        moved = _moving_bodies(model)
        body = np.asarray(m.geom_bodyid, int)
        gtype = np.asarray(m.geom_type, int)
        ctype = np.asarray(m.geom_contype, np.int64)
        caff = np.asarray(m.geom_conaffinity, np.int64)
        nonc = (ctype == 0) & (caff == 0)
        colliding = ~nonc if nonc.any() else np.ones(m.ngeom, bool)
        flat = np.isin(gtype, [int(G.mjGEOM_PLANE), int(G.mjGEOM_HFIELD)])
        self.robot_geoms = np.flatnonzero(colliding & (body != 0) & moved[body] & ~flat)
        spec = mujoco.MjSpec()
        spec.compiler.boundmass = 1e-6                 # mocap copies: their mass is not used
        spec.compiler.boundinertia = 1e-9
        wb = spec.worldbody
        kinds = {'box': G.mjGEOM_BOX, 'sphere': G.mjGEOM_SPHERE, 'cylinder': G.mjGEOM_CYLINDER,
                 'capsule': G.mjGEOM_CAPSULE, 'ellipsoid': G.mjGEOM_ELLIPSOID, 'mesh': G.mjGEOM_MESH}
        for k, g in enumerate(self.robot_geoms):
            b = wb.add_body(name=f'robot_{k}', mocap=True)
            t = int(gtype[g])
            if t == int(G.mjGEOM_MESH):
                mid = int(m.geom_dataid[g])
                va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
                spec.add_mesh(name=f'robot_mesh_{k}', uservert=m.mesh_vert[va:va + vn].astype(float).ravel())
                b.add_geom(name=f'robot_{k}', type=G.mjGEOM_MESH, meshname=f'robot_mesh_{k}')
            else:
                b.add_geom(name=f'robot_{k}', type=G(t), size=np.asarray(m.geom_size[g], float))
        for k, o in enumerate(obs):
            if o.gtype == 'mesh':
                spec.add_mesh(name=f'obstacle_mesh_{k}', uservert=np.asarray(o.vertices, float).ravel())
                wb.add_geom(name=f'obstacle_{k}', type=G.mjGEOM_MESH, meshname=f'obstacle_mesh_{k}')
            else:
                sz = np.r_[np.asarray(o.size, float), np.zeros(3)][:3]
                wb.add_geom(name=f'obstacle_{k}', type=kinds[o.gtype], size=sz, pos=np.asarray(o.position, float),
                            quat=np.asarray(o.quat_wxyz, float))
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        n = len(self.robot_geoms)
        self.obs = list(obs)
        # MuJoCo orders geoms by body (the world's first), so the ids are looked up by name
        self.rob_geoms = np.array([self.m.geom(f'robot_{k}').id for k in range(n)], int)
        self.obs_geoms = np.array([self.m.geom(f'obstacle_{k}').id for k in range(len(obs))], int)
        self.mocap = np.array([int(self.m.body_mocapid[self.m.body(f'robot_{k}').id]) for k in range(n)], int)
        self.rbound = np.asarray(self.m.geom_rbound, float).copy()
        self.link = [P.link_of_geom[int(g)] for g in self.robot_geoms]
        self.lock = threading.Lock()


_OBS_CACHE: 'weakref.WeakKeyDictionary' = weakref.WeakKeyDictionary()


def _obstacle_scene(model, obs: list) -> _ObstacleScene:
    key = _obstacle_key(obs)
    with _LOCK:
        per = _OBS_CACHE.setdefault(model, {})
        sc = per.get(key)
    if sc is None:
        sc = _ObstacleScene(model, obs)
        with _LOCK:
            if len(per) > 8:
                per.clear()
            per[key] = sc
    return sc


def obstacle_distance(project, robot_uid: str, q=None, margin: float = 0.0, distmax: float = DIST_MAX,
                      model=None) -> CollisionResult:
    """Whether robot robot_uid at q (its joint vector; None: home) is nearer than margin (m) to a project obstacle (a
    workspace object or surface) or a static geom of its own file, the smallest signed distance (capped at distmax)
    and the pairs as (link, obstacle name, distance m), nearest first; see the module docstring. Raises
    RobotModelError when the robot cannot be compiled."""
    import mujoco
    from ..frames import quat_from_matrix, quat_to_mujoco
    from .model import get_model
    model = model if model is not None else get_model(project, robot_uid)
    q = model.home if q is None else model._q(q)
    distmax = max(float(distmax), float(margin) + 1e-9)
    margin = float(margin)
    own = environment_distance(model, q, margin, distmax)
    obs, _notes = obstacles(project)
    found = [(a, b, dd) for a, b, dd in own.pairs]
    best = own.min_distance
    if obs:
        sc = _obstacle_scene(model, obs)
        if len(sc.robot_geoms):
            with sc.lock:
                d0 = model._kin(q)
                R0, t0 = model.base[:3, :3], model.base[:3, 3]
                pos = np.asarray(d0.geom_xpos)[sc.robot_geoms] @ R0.T + t0
                mats = R0[None] @ np.asarray(d0.geom_xmat)[sc.robot_geoms].reshape(-1, 3, 3)
                d = sc.d
                d.mocap_pos[sc.mocap] = pos
                d.mocap_quat[sc.mocap] = np.array([quat_to_mujoco(quat_from_matrix(M)) for M in mats])
                mujoco.mj_kinematics(sc.m, d)
                xp = np.asarray(d.geom_xpos)
                ia = np.repeat(np.arange(len(sc.rob_geoms)), len(sc.obs_geoms))
                ib = np.tile(np.arange(len(sc.obs_geoms)), len(sc.rob_geoms))
                A, B = sc.rob_geoms[ia], sc.obs_geoms[ib]
                lb = np.linalg.norm(xp[A] - xp[B], axis=1) - sc.rbound[A] - sc.rbound[B]
                for i in np.flatnonzero(lb < distmax):
                    dist = float(mujoco.mj_geomDistance(sc.m, d, int(A[i]), int(B[i]), distmax, None))
                    if dist < distmax:
                        found.append((sc.link[ia[i]], sc.obs[ib[i]].name, dist))
                        best = min(best, dist)
    found.sort(key=lambda t: t[2])
    best = float(min(best, distmax))
    if best < margin:
        out, seen = [], set()
        for a, b, dd in found:
            if dd >= margin:
                break
            if (a, b) not in seen:
                seen.add((a, b))
                out.append((a, b, float(dd)))
    else:
        out = [(found[0][0], found[0][1], float(min(found[0][2], distmax)))] if found else []
    return CollisionResult(bool(best < margin), best, out)


def env_distance(project, robot_uid: str, q=None, distmax: float = DIST_MAX) -> tuple[float, tuple | None]:
    """(min_distance m, pair) of robot robot_uid at q against the project's obstacles and its file's static geoms:
    the smallest signed distance (negative = penetration, capped at distmax) and the closest (link, obstacle name,
    distance m), or None when nothing is within distmax."""
    r = obstacle_distance(project, robot_uid, q, 0.0, distmax)
    return r.min_distance, (r.pairs[0] if r.pairs else None)
