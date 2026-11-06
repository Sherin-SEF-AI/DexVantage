"""Self-collision and environment distances of a posed robot, computed by MuJoCo.

    self_collision(model, q, margin=0.0)          -> CollisionResult(in_collision, min_distance m, pairs)
    self_collision_batch(model, Q, margin=0.0)    -> (in_collision (N,), min_distance (N,), [pairs per q])
    environment_distance(model, q, margin=0.0)    -> CollisionResult against the static geoms of the robot's own file
    environment_distance_batch(model, Q, ...)     -> as self_collision_batch
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

Known limits: MuJoCo collides meshes as their convex hulls, so a concave link (a U-shaped bracket, a hand's palm) is
fuller than its mesh and distances to it are short of the true ones (never long: the hull contains the mesh); the
environment is only what the robot's own file puts in its world body (project obstacles are not included yet); a URDF
link named "world" is merged into MuJoCo's world body, so its geoms count as environment; distances beyond distmax are
not computed.
"""
from __future__ import annotations

import threading
import weakref
from typing import NamedTuple

import numpy as np

DIST_MAX = 0.2                    # m: distances beyond this are not computed (a default, see the module docstring)

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
