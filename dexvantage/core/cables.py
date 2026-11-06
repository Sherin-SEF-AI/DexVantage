"""Cables across robot joints: the cable record kept in project.cables, the route between two attachment points, its
length at a pose, the slack loop and its bend radius, and the snag test against the robot's links (spec §7.5).

The study (core/studies/cables.py) sweeps the joints and reports; the geometry lives here so the page, the 3D
overlay and the validation checks use the same functions at one pose.

The record. project.cables is a list of plain dicts (clean_cable normalises one):

    {'uid', 'name', 'sensor_uid', 'type' (CABLE_TYPES), 'library_id' ('' when none),
     'min_bend_radius_m': {'value': m, 'source': text, 'kind': 'dynamic' | 'static' | 'unspecified'} | None,
     'diameter_m': m | None, 'length_m': the cable's provided length | None,
     'wrap_radius_m': m | None (None: the default, below),
     'attachments': [{'robot_uid', 'link', 'point': [x, y, z] in the link frame (m),
                      'length_to_next_m': the cable laid between this point and the next | None}],
     'connectors': [text, ...], 'strain_relief': text, 'note': text}

A figure the maker does not state stays None ("not stated"); a cable made from a library entry takes its diameter,
connectors and minimum bend radius from the entry (the moving / dynamic figure when the maker states one, since a
cable across a joint flexes; else the fixed / static figure, labelled so).

The route between consecutive attachments A (on link a) and B (on link b). The joints it crosses are the moving
joints on the kinematic tree's path from a to b (crossed_joints). Three cases:

- no joint: a and b move together; the route is the straight chord, its length never changes;
- one revolute joint (axis through O along u): the routed arc model. In cylindrical coordinates about the axis A is
  at radius d_a, height z_a and B at d_b, z_b, the angle from A to B about u being θ. The cable is guided around the
  joint on a cylinder of radius ρ (the wrap radius: the cable's wrap_radius_m, by default min(d_a, d_b), a labelled
  default: the cable lies around the joint at the nearer attachment's distance from the axis). Seen along the axis it
  runs on the tangent from A to the circle, along the circle and on the tangent to B (a belt over a pulley):

      α_a = arccos(ρ / d_a),  α_b = arccos(ρ / d_b),  t_a = √(d_a² − ρ²),  t_b = √(d_b² − ρ²)
      L_plane = t_a + t_b + ρ (|θ| − α_a − α_b)            when |θ| > α_a + α_b (it wraps)
      L_plane = √(d_a² + d_b² − 2 d_a d_b cos θ)              otherwise (the straight line clears the cylinder)
      L = √(L_plane² + (z_b − z_a)²)                          (the tangent planes and the cylinder unroll flat, so the
                                                               shortest path over them is a straight line there)

  θ is not reduced to ±π: it is the angle at home (the cable laid the short way round at home, an assumption)
  plus the joint's travel from home, θ = θ_home ± (q − q_home), so a joint turning more than half a turn winds the
  cable round it and the length keeps growing. The bend radius on the wrapped part is that of the helix,
  ρ / cos²φ with tan φ = (z_b − z_a) / L_plane;
- a prismatic joint, or several joints: the straight chord (route_kind 'chord'); the study says to add an attachment
  on each link so each run crosses one joint.

The slack loop. A run is laid with a length s (its length_to_next_m, else the suggested one: the largest length over
the sweeps and trajectories plus a service loop, loop_m). At a pose where the route is shorter than s the extra
length bows outwards, away from the joint axis (radially; for a run without a revolute joint, perpendicular to the
chord towards world +z): a circular arc laid over the route in its unrolled length and that outward offset (bulge),
its length found by bisection so the polyline is s long; on a straight route this is exactly the circular arc of
length s over the chord. The bend radius estimate (bend_estimate) is the smaller of the route's own bend on the
joint cylinder (ρ / cos²φ where it wraps) and the slack loop's radius taken as a circular arc of length s over the
route's length L: sin β / β = L / s, R = s / (2β). The drawn polyline is for the 3D view and the snag test; on a
straight run its three-point radius equals that arc's (checked). s shorter than the route is a pull.

The snag test (snag_hits): the polyline's segments outside a seat of seat_m around each attachment (the clamp, where
the cable touches its link by design) are cast as rays against each link's colliding triangles in the link's own
frame (core/raycast, Möller–Trumbore through a uniform grid built once per link), and the polyline's middle point is
tested for lying inside a link by the parity of crossings along a ray. A hit is a snag.

Known limits: the cable is a curve without stiffness or weight (no sag, no torsion, no friction on the links); the
loop's shape is a circular bow over the route, not an elastica, so the bend radius is an estimate; the wrap radius is
a single cylinder about the axis (no joint housing shape); a run across several joints is a chord, which
under-states the length when the arm folds; meshes MuJoCo would treat as convex hulls are tested as their triangles
here; the snag test samples poses (between samples the swept volume is not tested) and only the cable's own
robot's links count; the parity test needs closed link meshes.
"""
from __future__ import annotations

import math
import threading
import weakref

import numpy as np

from .project import new_uid

CABLE_TYPES = ('usb3', 'usb2', 'gmsl2', 'gmsl', 'coax', 'ethernet', 'power', 'other')
TYPE_LABEL = {'usb3': 'USB 3', 'usb2': 'USB 2.0', 'gmsl2': 'GMSL2 coax', 'gmsl': 'GMSL coax', 'coax': 'Coax',
              'ethernet': 'Ethernet', 'power': 'Power', 'other': 'Cable'}
BEND_KINDS = ('dynamic', 'static', 'unspecified')
NOT_STATED = 'not stated'
CLEARANCE_M = 0.005              # default: suggested points sit this far off a link's surface (plus half the diameter)
SUGGEST_DIAMETER_M = 0.005       # default: the diameter assumed by suggest_attachments when the cable states none
PARITY_DIR = np.array([0.5773, 0.5774, 0.5775]) / np.linalg.norm([0.5773, 0.5774, 0.5775])


# ====================================================================== records
def _num(v, positive=True):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or (positive and x <= 0):
        return None
    return x


def clean_bend(v):
    """{'value', 'source', 'kind'} or None. A bare number counts as entered by the user, kind unspecified."""
    if isinstance(v, dict):
        x = _num(v.get('value'))
        if x is None:
            return None
        kind = str(v.get('kind') or 'unspecified')
        return {'value': x, 'source': str(v.get('source') or 'entered by you'),
                'kind': kind if kind in BEND_KINDS else 'unspecified'}
    x = _num(v)
    return None if x is None else {'value': x, 'source': 'entered by you', 'kind': 'unspecified'}


def clean_attachment(d) -> dict:
    d = d if isinstance(d, dict) else {}
    p = d.get('point')
    try:
        pt = [float(x) for x in p][:3] if isinstance(p, (list, tuple)) else [0.0, 0.0, 0.0]
    except (TypeError, ValueError):
        pt = [0.0, 0.0, 0.0]
    pt = (pt + [0.0, 0.0, 0.0])[:3]
    pt = [x if math.isfinite(x) else 0.0 for x in pt]
    return {'robot_uid': str(d.get('robot_uid') or ''), 'link': str(d.get('link') or ''), 'point': pt,
            'length_to_next_m': _num(d.get('length_to_next_m'))}


def clean_cable(d) -> dict:
    """A cable record with every key, tolerant of junk (unknown keys dropped, bad numbers None)."""
    d = d if isinstance(d, dict) else {}
    t = str(d.get('type') or 'other')
    con = d.get('connectors')
    con = [str(c) for c in con] if isinstance(con, (list, tuple)) else ([str(con)] if con else [])
    wrap = _num(d.get('wrap_radius_m'), positive=False)
    wrap = wrap if wrap is not None and wrap >= 0 else None      # a negative radius is junk: the default
    return {'uid': str(d.get('uid') or new_uid()), 'name': str(d.get('name') or 'Cable'),
            'sensor_uid': str(d.get('sensor_uid') or ''), 'type': t if t in CABLE_TYPES else 'other',
            'library_id': str(d.get('library_id') or ''), 'min_bend_radius_m': clean_bend(d.get('min_bend_radius_m')),
            'diameter_m': _num(d.get('diameter_m')), 'length_m': _num(d.get('length_m')),
            'wrap_radius_m': wrap,
            'attachments': [clean_attachment(a) for a in d.get('attachments') or [] if isinstance(a, dict)],
            'connectors': con, 'strain_relief': str(d.get('strain_relief') or ''), 'note': str(d.get('note') or '')}


def new_cable(name: str = 'Cable', sensor_uid: str = '', entry=None) -> dict:
    """A new cable record (from a library entry when given)."""
    c = clean_cable({'name': name, 'sensor_uid': sensor_uid})
    if entry is not None:
        c.update(from_library(entry))
    return c


def from_library(entry) -> dict:
    """The record fields a library cable entry gives: type, library_id, diameter, connectors and the minimum bend
    radius (moving / dynamic when stated, else fixed / static), each with its source."""
    data = entry.data if hasattr(entry, 'data') else dict(entry)
    cb = data.get('cable') or {}
    src = (data.get('sources') or [{}])[0]
    where = f'{data.get("maker", "")} {data.get("name", "")}'.strip()
    cite = f'{src.get("title") or where}, {src.get("url", "")}, read {src.get("read", "?")}'
    dyn, sta = _num(cb.get('min_bend_dynamic_mm')), _num(cb.get('min_bend_static_mm'))
    if dyn is not None:
        bend = {'value': dyn / 1000.0, 'source': f'{cite}: {dyn:g} mm, moving (dynamic)', 'kind': 'dynamic'}
    elif sta is not None:
        bend = {'value': sta / 1000.0, 'source': f'{cite}: {sta:g} mm, fixed (static); the maker states no moving '
                                                 'figure', 'kind': 'static'}
    else:
        bend = None
    dia = _num(cb.get('diameter_mm'))
    t = str(cb.get('type') or 'other')
    return {'type': t if t in CABLE_TYPES else 'other', 'library_id': str(data.get('id') or ''),
            'diameter_m': None if dia is None else dia / 1000.0, 'min_bend_radius_m': bend,
            'connectors': [str(c) for c in cb.get('connectors') or []]}


# ====================================================================== kinematics
def _anc(m, b) -> list:
    out = [int(b)]
    while out[-1] != 0:
        out.append(int(m.body_parentid[out[-1]]))
    return out


def link_body(model, link: str) -> int:
    return int(model._lb[model.link_index(link)])


def crossed_joints(model, link_a: str, link_b: str) -> list:
    """[(joint index into model.joint_names, side)] of the moving joints on the tree path from link_a to link_b, in
    order from a to b. side = +1 when b is on the joint's child side (turning the joint by +δ turns b by +δ about the
    axis relative to a), −1 when a is."""
    m = model.m
    ba, bb = link_body(model, link_a), link_body(model, link_b)
    A, B = _anc(m, ba), _anc(m, bb)
    sb = set(B)
    lca = next(b for b in A if b in sb)
    up_a, up_b = A[:A.index(lca)], B[:B.index(lca)]
    jb = {}
    for i, j in enumerate(model._jid):
        jb.setdefault(int(m.jnt_bodyid[j]), []).append(i)
    return [(i, -1) for b in up_a for i in jb.get(b, [])] + \
           [(i, +1) for b in reversed(up_b) for i in jb.get(b, [])]


def posed(model, Q) -> tuple:
    """(T (K, n_links, 4, 4) world link transforms, anchors (K, nq, 3), axes (K, nq, 3)) of the moving joints in the
    world (Robot.base applied) at the joint vectors Q (K, nq)."""
    Q = model.batch_q(Q)
    K = len(Q)
    T = np.empty((K, len(model.link_names), 4, 4))
    anc = np.empty((K, model.nq, 3))
    ax = np.empty((K, model.nq, 3))
    R, t = model.base[:3, :3], model.base[:3, 3]
    for k, q in enumerate(Q):
        d = model._kin(q)               # RobotModel's MjData at q (mj_kinematics): anchors and axes come with it
        T[k] = model._links_T(d)
        anc[k] = d.xanchor[model._jid] @ R.T + t
        ax[k] = d.xaxis[model._jid] @ R.T
    return T, anc, ax


def world_points(T, li: int, point) -> np.ndarray:
    """(K, 3) world positions of a point fixed in link li over the posed transforms T."""
    p = np.asarray(point, float)
    return T[:, li, :3, :3] @ p + T[:, li, :3, 3]


def route_kind(model, crossed) -> str:
    if not crossed:
        return 'rigid'
    if len(crossed) == 1 and model.joint_types[crossed[0][0]] in ('revolute', 'continuous'):
        return 'wrap'
    return 'chord'


# ====================================================================== the routed arc model
def cylindrical(A, B, O, u) -> dict:
    """Cylindrical coordinates of A and B (K, 3) about the axes (O, u) (K, 3): radii, heights, the unit radial e1 of
    A, e2 = u × e1, and the signed angle from A to B about u in (−π, π]."""
    A, B, O, u = (np.atleast_2d(np.asarray(x, float)) for x in (A, B, O, u))
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    a, b = A - O, B - O
    za, zb = np.einsum('ij,ij->i', a, u), np.einsum('ij,ij->i', b, u)
    ra, rb = a - za[:, None] * u, b - zb[:, None] * u
    da, db = np.linalg.norm(ra, axis=1), np.linalg.norm(rb, axis=1)
    fallback = np.cross(u, np.where(np.abs(u[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]]))
    e1 = np.where(da[:, None] > 1e-12, ra / np.maximum(da, 1e-300)[:, None], fallback)
    e1 = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(u, e1)
    ang = np.arctan2(np.einsum('ij,ij->i', np.cross(ra, rb), u), np.einsum('ij,ij->i', ra, rb))
    return {'da': da, 'db': db, 'za': za, 'zb': zb, 'e1': e1, 'e2': e2, 'u': u, 'O': O, 'angle': ang}


def default_wrap_radius(da, db) -> np.ndarray:
    """The labelled default: the nearer attachment's distance from the axis."""
    return np.minimum(da, db)


def wrap_length(da, db, dz, theta, rho) -> dict:
    """The routed arc model's length (see the module docstring), vectorised: L, L_plane, wraps (bool), the tangent
    lengths and angles, and the helix bend radius on the wrapped part (inf where it does not wrap)."""
    da, db, dz, theta, rho = np.broadcast_arrays(*(np.asarray(x, float) for x in (da, db, dz, theta, rho)))
    rho = np.clip(rho, 0.0, np.minimum(da, db))
    t = np.abs(theta)
    ta, tb = np.sqrt(np.maximum(da ** 2 - rho ** 2, 0)), np.sqrt(np.maximum(db ** 2 - rho ** 2, 0))
    # α = arccos(ρ / d), taken as atan2(t, ρ) from the same tangent length: arccos is ill-conditioned near 1 and
    # would disagree with t by about 1e-8 rad when an attachment lies on the wrap cylinder
    aa = np.where(da > 1e-12, np.arctan2(ta, rho), np.pi / 2)
    ab = np.where(db > 1e-12, np.arctan2(tb, rho), np.pi / 2)
    wraps = t > aa + ab
    chord = np.sqrt(np.maximum(da ** 2 + db ** 2 - 2 * da * db * np.cos(t), 0.0))
    lp = np.where(wraps, ta + tb + rho * (t - aa - ab), chord)
    L = np.hypot(lp, dz)
    with np.errstate(invalid='ignore', divide='ignore'):
        cos2 = np.where(L > 1e-15, (lp / np.where(L > 1e-15, L, 1.0)) ** 2, 1.0)
        rbend = np.where(wraps, np.where(cos2 > 1e-15, rho / np.maximum(cos2, 1e-15), np.inf), np.inf)
    return {'L': L, 'L_plane': lp, 'wraps': wraps, 'ta': ta, 'tb': tb, 'aa': aa, 'ab': ab, 'rho': rho,
            'bend_m': rbend}


def wrap_polyline(cyl: dict, theta, rho, n: int) -> np.ndarray:
    """(K, n, 3) world points along the routed arc model's route (tangent, helix, tangent; or the straight chord when
    it does not wrap), evenly spaced in unrolled length."""
    da, db, za, zb = cyl['da'], cyl['db'], cyl['za'], cyl['zb']
    theta = np.broadcast_to(np.asarray(theta, float), da.shape)
    w = wrap_length(da, db, zb - za, theta, rho)
    lp, ta, tb, aa, ab, r = w['L_plane'], w['ta'], w['tb'], w['aa'], w['ab'], w['rho']
    K = len(da)
    s = np.where(theta < 0, -1.0, 1.0)
    f = np.linspace(0.0, 1.0, n)
    ell = f[None, :] * lp[:, None]                                      # (K, n) unrolled length from A
    ax, ay = da[:, None] * np.ones((1, n)), np.zeros((K, n))
    bx, by = (db * np.cos(theta))[:, None], (db * np.sin(theta))[:, None]
    # straight in-plane line (not wrapping)
    x, y = ax + f[None, :] * (bx - ax), ay + f[None, :] * (by - ay)
    # wrapping: tangent A→Ta, arc, tangent Tb→B
    wr = w['wraps'][:, None] & np.ones((1, n), bool)
    if wr.any():
        tax, tay = r * np.cos(s * aa), r * np.sin(s * aa)
        tbx, tby = r * np.cos(theta - s * ab), r * np.sin(theta - s * ab)
        arc = r * (np.abs(theta) - aa - ab)
        g1 = ell <= ta[:, None]
        g3 = ell >= (ta + arc)[:, None]
        with np.errstate(invalid='ignore', divide='ignore'):
            fa = np.where(ta[:, None] > 0, ell / np.maximum(ta, 1e-300)[:, None], 1.0)
            phi = (s * aa)[:, None] + s[:, None] * (ell - ta[:, None]) / np.maximum(r, 1e-300)[:, None]
            fb = np.where(tb[:, None] > 0, (ell - (ta + arc)[:, None]) / np.maximum(tb, 1e-300)[:, None], 1.0)
        x1, y1 = da[:, None] + fa * (tax - da)[:, None], fa * tay[:, None]
        x2, y2 = r[:, None] * np.cos(phi), r[:, None] * np.sin(phi)
        x3, y3 = tbx[:, None] + fb * (bx - tbx[:, None]), tby[:, None] + fb * (by - tby[:, None])
        xw = np.where(g1, x1, np.where(g3, x3, x2))
        yw = np.where(g1, y1, np.where(g3, y3, y2))
        x, y = np.where(wr, xw, x), np.where(wr, yw, y)
    with np.errstate(invalid='ignore', divide='ignore'):
        fz = np.where(lp[:, None] > 1e-12, ell / np.maximum(lp, 1e-300)[:, None], f[None, :])
    z = za[:, None] + fz * (zb - za)[:, None]
    P = (cyl['O'][:, None, :] + z[..., None] * cyl['u'][:, None, :] + x[..., None] * cyl['e1'][:, None, :]
         + y[..., None] * cyl['e2'][:, None, :])
    return P


def chord_polyline(A, B, n: int) -> np.ndarray:
    A, B = np.atleast_2d(A), np.atleast_2d(B)
    f = np.linspace(0.0, 1.0, n)[None, :, None]
    return A[:, None, :] + f * (B - A)[:, None, :]


# ====================================================================== the slack loop and bend radius
def poly_length(P) -> np.ndarray:
    return np.linalg.norm(np.diff(P, axis=-2), axis=-1).sum(axis=-1)


def outward(P, O=None, u=None) -> np.ndarray:
    """(K, n, 3) unit directions the loop bows along: away from the axis (O, u) (K, 3) when given, perpendicular to the
    route as seen along the axis (the radial direction less its component along the route's tangent projected on the
    plane across the axis: constant along a straight part, the radius on the helix, square to the route and to the
    axis), else perpendicular to the chord towards world +z."""
    K, n, _ = P.shape
    c = P[:, -1] - P[:, 0]
    cl = np.linalg.norm(c, axis=1, keepdims=True)
    c = np.where(cl > 1e-12, c / np.maximum(cl, 1e-300), [[1.0, 0, 0]])
    z = np.array([0.0, 0.0, 1.0])
    perp = z - (c @ z)[:, None] * c
    alt = np.cross(c, [[0.0, 1.0, 0.0]])
    perp = np.where(np.linalg.norm(perp, axis=1, keepdims=True) > 1e-6, perp, alt)
    perp = perp / np.maximum(np.linalg.norm(perp, axis=1, keepdims=True), 1e-300)
    out = np.broadcast_to(perp[:, None, :], (K, n, 3)).copy()
    if O is not None:
        u = np.asarray(u, float) / np.linalg.norm(u, axis=-1, keepdims=True)
        r = P - O[:, None, :]
        r = r - np.einsum('kni,ki->kn', r, u)[..., None] * u[:, None, :]
        tg = np.gradient(P, axis=1) if n > 1 else np.zeros_like(P)
        tg = tg - np.einsum('kni,ki->kn', tg, u)[..., None] * u[:, None, :]      # the tangent seen along the axis
        tl = np.linalg.norm(tg, axis=-1, keepdims=True)
        tg = np.where(tl > 1e-12, tg / np.maximum(tl, 1e-300), 0.0)
        r = r - np.einsum('kni,kni->kn', r, tg)[..., None] * tg      # the route's normal in that view, off the axis
        rl = np.linalg.norm(r, axis=-1, keepdims=True)
        out = np.where(rl > 1e-9, r / np.maximum(rl, 1e-300), out)
    return out


def _arc_beta(ratio) -> np.ndarray:
    """β in [0, π] with sin β / β = ratio (chord over arc length, 0..1): the half angle of a circular arc."""
    ratio = np.clip(np.asarray(ratio, float), 0.0, 1.0)
    lo, hi = np.zeros(ratio.shape), np.full(ratio.shape, np.pi)
    for _ in range(60):
        mid = (lo + hi) / 2
        f = np.sin(mid) / np.maximum(mid, 1e-300)
        big = f > ratio
        lo, hi = np.where(big, mid, lo), np.where(big, hi, mid)
    return (lo + hi) / 2


def _interp(P, ell, step):
    """Points of the evenly spaced polylines P (K, n, 3) at unrolled lengths ell (K, m), extended along the end
    segments beyond either end."""
    K, n, _ = P.shape
    step = np.asarray(step, float)
    ok = (step > 1e-12)[:, None]
    # a route of zero length (two attachments at one point, as Add point leaves them): every point is that point
    x = np.where(ok, np.asarray(ell, float) / np.where(ok, step[:, None], 1.0), 0.0)
    x = np.nan_to_num(x, nan=0.0, posinf=float(n - 1), neginf=0.0)
    i = np.clip(np.floor(x).astype(int), 0, n - 2)
    f = (x - i)[..., None]
    rows = np.arange(K)[:, None]
    return P[rows, i] + f * (P[rows, i + 1] - P[rows, i])


def bulge(P, s, n_dir, iters: int = 40) -> np.ndarray:
    """The slack loop: a circular bow of length σ laid over each route P (K, n, 3) (points evenly spaced along it),
    its height along n_dir (K, n, 3): in the route's unrolled length ℓ and the offset y, the arc
    ℓ = L/2 + R sin ψ, y = R (cos ψ − cos β), ψ from −β to β, sin β / β = L / σ, R = σ / (2β) (beyond a half circle
    the bow overhangs the ends, continued along the end segments). σ is found by bisection so the polyline is s (K,)
    long. On a straight route it is the circular arc of length s over the chord; where s is not longer than the
    route, the route itself (taut)."""
    P = np.asarray(P, float)
    K, n, _ = P.shape
    s = np.broadcast_to(np.asarray(s, float), (K,)).astype(float)
    L = poly_length(P)
    need = s > L + 1e-12
    if not need.any():
        return P.copy()
    step = L / max(n - 1, 1)
    u = np.linspace(0.0, 1.0, n)

    def shape(sig):
        beta = np.where(L > 1e-12, _arc_beta(np.where(sig > 0, L / np.maximum(sig, 1e-300), 1.0)), np.pi)
        R = sig / np.maximum(2 * beta, 1e-300)
        psi = (-1 + 2 * u)[None, :] * beta[:, None]
        ell = L[:, None] / 2 + R[:, None] * np.sin(psi)
        y = R[:, None] * (np.cos(psi) - np.cos(beta)[:, None])
        base = _interp(P, ell, step)
        nn = _interp(n_dir, np.clip(ell, 0.0, L[:, None]), step)
        nn = nn / np.maximum(np.linalg.norm(nn, axis=-1, keepdims=True), 1e-300)
        return base + y[..., None] * nn

    lo, hi = L.copy(), np.maximum(s, L) * 1.0
    for _ in range(30):                                  # grow hi until the polyline passes s
        short = need & (poly_length(shape(hi)) < s)
        if not short.any():
            break
        hi = np.where(short, hi * 1.5, hi)
    for _ in range(iters):
        mid = (lo + hi) / 2
        longer = poly_length(shape(mid)) > s
        hi = np.where(longer, mid, hi)
        lo = np.where(longer, lo, mid)
    out = shape((lo + hi) / 2)
    return np.where(need[:, None, None], out, P)


def bend_radius(P) -> np.ndarray:
    """(K,) the smallest radius of the circle through three consecutive points of each polyline (K, n, 3); inf for a
    straight one."""
    P = np.asarray(P, float)
    if P.shape[1] < 3:
        return np.full(len(P), np.inf)
    a = np.linalg.norm(P[:, 1:-1] - P[:, :-2], axis=2)
    b = np.linalg.norm(P[:, 2:] - P[:, 1:-1], axis=2)
    c = np.linalg.norm(P[:, 2:] - P[:, :-2], axis=2)
    cr = np.linalg.norm(np.cross(P[:, 1:-1] - P[:, :-2], P[:, 2:] - P[:, :-2]), axis=2)
    with np.errstate(divide='ignore', invalid='ignore'):
        R = np.where(cr > 1e-18 * np.maximum(a * b, 1e-300), a * b * c / (2 * cr), np.inf)
    return R.min(axis=1)


def bend_estimate(route_bend, L, s) -> np.ndarray:
    """(K,) the bend radius estimate of a run of route length L (K,) laid with s: the smaller of the route's own bend
    on the joint cylinder (route_bend, inf where it does not wrap) and the slack loop's radius as a circular arc of
    length s over L (arc_radius; none when s is not longer than L: the run is taut)."""
    L = np.asarray(L, float)
    s = np.broadcast_to(np.asarray(s, float), L.shape)
    loop = np.where(s > L + 1e-12, arc_radius(s, L), np.inf)
    return np.minimum(np.asarray(route_bend, float), loop)


def arc_radius(s, c) -> np.ndarray:
    """The radius of a circular arc of length s over a chord c (s ≥ c): sin β / β = c / s, R = s / (2β), β by
    bisection (the slack loop's radius in bend_estimate)."""
    s, c = np.broadcast_arrays(np.asarray(s, float), np.asarray(c, float))
    target = np.clip(c / np.maximum(s, 1e-300), 0.0, 1.0)
    lo, hi = np.zeros(s.shape), np.full(s.shape, np.pi)
    for _ in range(80):
        mid = (lo + hi) / 2
        f = np.where(mid > 0, np.sin(mid) / np.maximum(mid, 1e-300), 1.0)
        big = f > target
        lo, hi = np.where(big, mid, lo), np.where(big, hi, mid)
    beta = (lo + hi) / 2
    return np.where(target >= 1.0 - 1e-15, np.inf, s / (2 * np.maximum(beta, 1e-300)))


# ====================================================================== snag test
_GRIDS: 'weakref.WeakKeyDictionary' = weakref.WeakKeyDictionary()
_GLOCK = threading.Lock()


def link_grids(model) -> dict:
    """{link index: raycast.Grid of its colliding triangles in its own frame}, built once per model."""
    from . import raycast as RC
    with _GLOCK:
        g = _GRIDS.get(model)
        if g is not None:
            return g
    by = {}
    for ln, V, F, _c in model.collision_geoms():
        if len(F):
            by.setdefault(model.link_index(ln), []).append(np.asarray(V, float)[np.asarray(F)])
    g = {li: RC.Grid.build(np.concatenate(ts)) for li, ts in by.items()}
    with _GLOCK:
        _GRIDS[model] = g
    return g


def _inside(grid, X) -> np.ndarray:
    """(N,) True where the points X (N, 3) lie inside the grid's triangles by the parity of crossings along one
    fixed direction."""
    X = np.asarray(X, float).reshape(-1, 3)
    cnt = np.zeros(len(X), int)
    o = X.copy()
    act = np.ones(len(X), bool)
    for _ in range(64):
        if not act.any():
            break
        t = grid.first_hit(o[act], np.broadcast_to(PARITY_DIR, (int(act.sum()), 3)))
        hit = np.isfinite(t)
        idx = np.flatnonzero(act)
        cnt[idx[hit]] += 1
        o[idx[hit]] = o[idx[hit]] + (t[hit] + 1e-7)[:, None] * PARITY_DIR
        act[idx[~hit]] = False
    return cnt % 2 == 1


def snag_hits(model, T, P, A, B, seat_m: float, grids: dict | None = None) -> list:
    """Per pose k, the links (names) the polyline P[k] (K, n, 3) passes through, outside a seat of seat_m around its
    ends A[k] and B[k] (K, 3), at the posed transforms T (K, n_links, 4, 4): [[(link, point (3,)) ...] per pose]."""
    grids = link_grids(model) if grids is None else grids
    P = np.asarray(P, float)
    K, n, _ = P.shape
    P0, P1 = P[:, :-1], P[:, 1:]

    def far(X):
        return (np.linalg.norm(X - A[:, None], axis=2) > seat_m) & (np.linalg.norm(X - B[:, None], axis=2) > seat_m)
    keep = far(P0) & far(P1)
    mid = P[:, n // 2]
    mid_ok = (np.linalg.norm(mid - A, axis=1) > seat_m) & (np.linalg.norm(mid - B, axis=1) > seat_m)
    out = [[] for _ in range(K)]
    kk, jj = np.nonzero(keep)
    for li, grid in grids.items():
        Ti = np.linalg.inv(T[:, li])
        R, t = Ti[:, :3, :3], Ti[:, :3, 3]
        hit_k = set()
        if len(kk):
            o = np.einsum('kij,kj->ki', R[kk], P0[kk, jj]) + t[kk]
            e = np.einsum('kij,kj->ki', R[kk], P1[kk, jj]) + t[kk]
            d = e - o
            ln = np.linalg.norm(d, axis=1)
            th = grid.first_hit(o, d, tmax=ln)
            h = np.isfinite(th) & (th <= ln)
            for k_, j_, tt in zip(kk[h], jj[h], th[h]):
                if k_ not in hit_k:
                    hit_k.add(int(k_))
                    dd = P1[k_, j_] - P0[k_, j_]
                    out[k_].append((model.link_names[li], P0[k_, j_] + dd / max(np.linalg.norm(dd), 1e-300) * tt))
        km = np.flatnonzero(mid_ok)
        if len(km):
            xm = np.einsum('kij,kj->ki', R[km], mid[km]) + t[km]
            ins = _inside(grid, xm)
            for k_ in km[ins]:
                if int(k_) not in hit_k:
                    out[k_].append((model.link_names[li], mid[k_].copy()))
    return out


# ====================================================================== suggested attachments
def _link_vertices(model, li) -> np.ndarray:
    V = [np.asarray(v, float) for ln, v, _f, _c in model.collision_geoms() if model.link_index(ln) == li]
    if not V:
        V = [np.asarray(v, float) for ln, v, _f, _c in model.visual_meshes() if model.link_index(ln) == li]
    return np.concatenate(V) if V else np.zeros((0, 3))


def _push_out(model, li, M, n, diameter, tube=0.03) -> np.ndarray:
    """M (link frame) moved along the unit n to just outside the link: past the farthest of its colliding vertices
    (visual when it has none) that lie within `tube` of the line M + t·n, plus half the diameter and CLEARANCE_M."""
    V = _link_vertices(model, li)
    off = (diameter or SUGGEST_DIAMETER_M) / 2 + CLEARANCE_M
    if not len(V):
        return M + n * off
    r = V - M
    t = r @ n
    lat = np.linalg.norm(r - t[:, None] * n, axis=1)
    near = lat <= tube
    tmax = float(t[near].max()) if near.any() else float(t.max())
    return M + n * (max(tmax, 0.0) + off)


def _off_axes(ua, ub, seg) -> np.ndarray:
    """A unit direction (world) to push a suggested point along so it moves away from both joint axes ua and ub: their
    common perpendicular when they are not parallel, else the perpendicular to ua and the link (seg), else any
    perpendicular to ua; turned towards world +z (or, when level, towards world −x: behind the robot)."""
    ua = ua / np.linalg.norm(ua)
    cands = []
    if ub is not None:
        cands.append(np.cross(ua, ub))
    sl = np.linalg.norm(seg)
    if sl > 1e-6:
        cands.append(np.cross(ua, seg / sl))
    cands += [np.cross(ua, [0.0, 0.0, 1.0]), np.cross(ua, [1.0, 0.0, 0.0])]
    d = next(c for c in cands if np.linalg.norm(c) > 0.2)
    d = d / np.linalg.norm(d)
    if abs(d[2]) > 1e-3:
        return d if d[2] > 0 else -d
    return d if d[0] <= 0 else -d


def suggest_attachments(model, sensor, diameter=None) -> tuple[list, list]:
    """(attachments, notes): a route from a link-mounted sensor down the kinematic chain to the first link no moving
    joint carries (the base), one attachment per link that a moving joint separates, so each run crosses one joint.
    The sensor's own point is its mount position; the others sit at the middle of their link (between its two joint
    anchors at home; the base's at its child joint's anchor), pushed out past the link's colliding surface along a
    direction away from both joint axes (_off_axes). A suggestion to check against the real robot."""
    notes = []
    link = sensor.mount.parent_frame
    li0 = model.link_index(link)
    m = model.m
    home = model.home
    T, anc, axs = posed(model, home[None])
    T, anc, axs = T[0], anc[0], axs[0]
    jb = {}
    for i, j in enumerate(model._jid):
        jb.setdefault(int(m.jnt_bodyid[j]), []).append(i)
    body_link = {int(model._lb[i]): i for i in range(len(model.link_names)) if np.allclose(model._loff[i], np.eye(4))}
    out = [{'robot_uid': model.robot_uid, 'link': link, 'point': [float(v) for v in sensor.mount.pose.position],
            'length_to_next_m': None}]
    chain = _anc(m, int(model._lb[li0]))[:-1]                     # the sensor's body up to the robot's root body
    # stop points: each body whose parent joint moves, starting from the sensor's
    below = None                                                  # the joint just crossed (index), on the child side
    for b in chain:
        js = jb.get(b, [])
        if below is not None:
            li = body_link.get(b)
            if li is None:
                continue
            # this body is between joint `below` (its child side) and its own joints (its parent side)
            pa = anc[below]
            pb = anc[js[0]] if js else T[li, :3, 3]
            Mw = (pa + pb) / 2
            dirw = _off_axes(axs[below], axs[js[0]] if js else None, pb - pa)
            if not js:                                             # the base: behind its child joint, off its axis
                Mw = pa
            Ti = np.linalg.inv(T[li])
            M = Ti[:3, :3] @ Mw + Ti[:3, 3]
            nl = Ti[:3, :3] @ dirw
            P = _push_out(model, li, M, nl, diameter)
            out.append({'robot_uid': model.robot_uid, 'link': model.link_names[li],
                        'point': [round(float(v), 4) + 0.0 for v in P], 'length_to_next_m': None})
            if not js:
                break
        if js:
            below = js[0]
            if len(js) > 1:
                notes.append(f'Link {model.body_link[b]} has {len(js)} joints on one body: its run crosses them '
                             'together (a chord).')
    if len(out) < 2:
        notes.append(f'{sensor.name} sits on {link}, which no moving joint separates from the base: there is no joint '
                     'for a cable to cross.')
    return out, notes
