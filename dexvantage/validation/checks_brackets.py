"""Checks of the bracket generator (core/brackets.py, core/meshing.py) against independent references.

- brackets.watertight: every kind of bracket the generator makes (a flange adapter, both T-slot clamps, a pan-tilt
  plate, palm and dorsal micro-camera sleeves), meshed as for printing, examined by code written out here (not
  meshing.check): every undirected edge in exactly two triangles (closed, manifold edges), every directed edge once
  (its two triangles wind it opposite ways: consistent orientation), the Euler characteristic V - E + F even and at
  most 2 per shell (a closed orientable surface: 2 - 2g), no triangle of zero area, and a positive signed volume
  (divergence theorem: outward normals). The error figure is the count of defects; the tolerance is 0.
- brackets.volume: the flange adapter's disc alone (ISO 9409-1-50-4-M6, no fillets) against its volume in closed form,
  written out here from the cylinders it is made of, and its Euler characteristic against 2 - 2g with g the number of
  through holes counted here (4 bolts, the dowel, the centre hole: g = 6, V - E + F = -10).
- brackets.iso_holes: the hole centres and diameters measured on the generated adapter's mesh (circles fitted to the
  hole walls' vertices by least squares, written out here) against the ISO 9409-1-50-4-M6 positions typed again in
  this check from the cited drawing: bolt circle 50 mm, 4 holes at 45 degrees and 4 x 90 degrees from the dowel, the
  dowel on the bolt circle on +x (Universal Robots, UR5e datasheet, drawing "Tool flange for UR5e"; ISO 9409-1:2004
  public preview, clause 4.1 and Figure 1; read 2026-11-06). This is a second reading of the same drawing, not a
  second source.
"""
from __future__ import annotations

import math

import numpy as np

from .suite import Outcome, check, figure, series

UR5E_URL = 'https://www.universal-robots.com/manuals/latest/en/datasheets/ur5e/'
ISO_URL = 'https://cdn.standards.iteh.ai/samples/36578/348a837664f444bc83e2902c7a5acf4c/ISO-9409-1-2004.pdf'

# Typed again for this check from the UR5e drawing "Tool flange for UR5e" (read 2026-11-06), NOT read from
# core/brackets.py: Ø50 ±0.1 bolt circle, 4 x M6 at 45° and 4 x 90°, Ø6 H7 dowel on the circle on +Xm (ISO 9409-1
# clause 4.1), Ø31.5 H7 centring recess, Ø63 h8 spigot.
PCD = 50.0
BOLT_ANGLES = (45.0, 135.0, 225.0, 315.0)
PIN_ANGLE = 0.0
M6 = 6.0
PIN = 6.0
PILOT = 31.5
SPIGOT = 63.0


def edge_report(V, F) -> dict:
    """Closedness, orientation, Euler characteristic and signed volume, written out here."""
    V, F = np.asarray(V, float), np.asarray(F, np.int64)
    directed = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    _ud, n_dir = np.unique(directed, axis=0, return_counts=True)
    und = np.sort(directed, axis=1)
    _uu, n_und = np.unique(und, axis=0, return_counts=True)
    used = np.unique(F.reshape(-1))
    chi = len(used) - len(n_und) + len(F)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area2 = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    vol = float(np.sum(np.einsum('ij,ij->i', a, np.cross(b, c))) / 6.0)
    # shells: union-find over the triangles' vertices
    parent = np.arange(len(V))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for t in F:
        r0 = find(t[0])
        for v in t[1:]:
            rv = find(v)
            if rv != r0:
                parent[rv] = r0
    shells = len({find(v) for v in used})
    return {'triangles': int(len(F)), 'edges_not_two': int((n_und != 2).sum()),
            'directed_twice': int((n_dir > 1).sum()), 'zero_area': int((area2 <= 1e-12).sum()), 'euler': int(chi),
            'shells': int(shells),
            'euler_ok': bool(chi % 2 == 0 and chi <= 2 * shells), 'volume_mm3': vol}


def _defects(r) -> int:
    return r['edges_not_two'] + r['directed_twice'] + r['zero_area'] + (0 if r['euler_ok'] else 1) + \
        (0 if r['volume_mm3'] > 0 else 1)


def sample_designs(quick: bool) -> list:
    """[(label, Design)]: one of each kind, on library cameras (a D435 and a micro camera) at scripted poses."""
    from ..core import brackets as B
    from ..core.project import Sensor
    d435 = B.camera_geom(Sensor(model_id='intel-realsense-d435'))
    micro = B.camera_geom(Sensor(model_id='omnivision-ovm6948'))
    T = np.eye(4)                                   # camera looking along the flange axis, 60 mm out, 30 mm up
    T[:3, :3] = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float)
    T[:3, 3] = [70.0, 0.0, 30.0]
    out = [('flange adapter, D435', B.design(B.BracketSpec(kind='flange', keepout_mm=70.0), d435, T))]
    out.append(('2020 clamp, D435', B.design(B.BracketSpec(kind='tslot2020'), d435)))
    out.append(('4040 clamp, D435, tilted 20°', B.design(B.BracketSpec(kind='tslot4040', tilt_deg=20.0), d435)))
    if not quick:
        out.append(('pan-tilt plate, D435', B.design(B.BracketSpec(kind='pantilt', pt_pcd_mm=16.0, pt_hole_mm=2.4),
                                                       d435)))
        out.append(('palm sleeve, OVM6948', B.design(B.BracketSpec(kind='palm', wall_mm=1.2), micro)))
        out.append(('dorsal sleeve, OVM6948', B.design(B.BracketSpec(kind='dorsal', wall_mm=1.2), micro)))
    return out


@check('brackets.watertight', 'brackets', 'Bracket meshes are closed, manifold and outward',
       'defects over one bracket of each kind: edges not in exactly two triangles, directed edges used twice '
       '(inconsistent orientation), zero-area triangles, an Euler characteristic V - E + F that is odd or above 2 '
       'per shell, a signed volume that is not positive',
       'the definitions of a closed orientable 2-manifold and of the divergence-theorem volume, written out in the '
       'check (independent of meshing.check)', 0.0, 'defects', refs=('treece1999', 'euler_poincare'), cost='study')
def _watertight(ctx):
    from ..core import brackets as B
    rows, total = [], 0
    for label, D in sample_designs(ctx.quick):
        V, F, _chk = B.mesh_design(D)
        r = edge_report(V, F)
        n = _defects(r)
        total += n
        rows.append(dict(r, bracket=label, defects=n, res_mm=D.res_mm))
    return Outcome(float(total), {'brackets': rows},
                   note='A printable STL must be a closed, consistently wound surface; any defect is a fault (a '
                        'slicer may then fill or drop regions), so the tolerance is zero.')


@check('brackets.volume', 'brackets', 'Flange adapter disc volume and topology',
       'relative difference between the mesh volume of the ISO 9409-1-50-4-M6 adapter disc (meshed at 0.25 mm, no '
       'fillets) and its closed-form volume; its Euler characteristic must equal 2 - 2g with g = 6 through holes',
       'the volume of the disc written out from its cylinders: π t (R² - rc² - 4 rb² - rp²) + π h (rs² - rc²) - '
       'π h (rr² - rc²), and the Euler-Poincaré formula', 0.01, 'relative', refs=('euler_poincare', 'ur5e_flange'),
       cost='study')
def _volume(ctx):
    from ..core import brackets as B
    from ..core import meshing as M
    spec = B.BracketSpec(kind='flange', fillet_mm=0.0)
    solid, _fi = B.flange_plate(spec, '50-4-M6')
    h_res = 0.35 if ctx.quick else 0.25
    V, F = M.mesh(solid, h_res)
    r = edge_report(V, F)
    # closed form, from the figures typed above and the spec's own defaults (the generator's design choices)
    R = SPIGOT / 2
    t = spec.disc_mm
    rb = (M6 + spec.clearance_mm) / 2
    rp = (PIN + spec.fit_mm) / 2
    rs = PILOT / 2 - spec.fit_mm / 2
    rr = PILOT / 2 + spec.fit_mm / 2
    rc = min(spec.centre_hole_mm / 2, rs - 1.5)
    h = spec.spigot_mm
    exact = math.pi * t * (R ** 2 - rc ** 2 - 4 * rb ** 2 - rp ** 2) + math.pi * h * (rs ** 2 - rc ** 2) - \
        math.pi * h * (rr ** 2 - rc ** 2)
    rel = abs(r['volume_mm3'] - exact) / exact
    genus = 4 + 1 + 1
    ok = r['euler'] == 2 - 2 * genus and r['shells'] == 1 and _defects(r) == 0
    return Outcome(rel, {'mesh_volume_mm3': r['volume_mm3'], 'closed_form_mm3': exact, 'res_mm': h_res,
                         'euler': r['euler'], 'euler_expected': 2 - 2 * genus, 'shells': r['shells'],
                         'triangles': r['triangles']}, ok=ok,
                   note='The mesher interpolates the distance linearly between grid points, so every sharp edge is '
                        'cut by a fraction of a grid step; summed over the disc\'s edges that removes about 0.1 to '
                        '0.2 % of the volume (measured at 0.35 and 0.25 mm). 1 % allows for it while '
                        'a wrong hole size (0.4 mm on a bolt hole is 1.5 % of its area) or a missing feature fails.')


def fit_circle(x, y) -> tuple[float, float, float]:
    """(cx, cy, r) of the least-squares circle x² + y² = 2 cx x + 2 cy y + c (Kåsa's fit), written out here."""
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    cx, cy = sol[0], sol[1]
    return float(cx), float(cy), float(math.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0)))


@check('brackets.iso_holes', 'brackets', 'Flange adapter holes on the ISO 9409-1 pattern',
       'largest distance between a hole centre measured on the generated adapter\'s mesh (a circle fitted to the '
       'hole wall at mid-thickness) and its position in the ISO 9409-1-50-4-M6 pattern; diameters must match the '
       'nominal plus the clearance within 0.1 mm',
       'the pattern typed again in the check from Universal Robots\' UR5e flange drawing and ISO 9409-1 Figure 1: '
       'bolt circle 50 mm, bolts at 45° + k 90°, the dowel on +x on the circle', 0.05, 'mm',
       refs=('ur5e_flange', 'iso9409_1'), cost='study')
def _iso_holes(ctx):
    from ..core import brackets as B
    label, D = sample_designs(True)[0]
    V, F, _chk = B.mesh_design(D, 0.25)
    spec = B.BracketSpec(kind='flange')
    want = [(f'bolt at {a:g}°', PCD / 2 * math.cos(math.radians(a)), PCD / 2 * math.sin(math.radians(a)),
             M6 + spec.clearance_mm) for a in BOLT_ANGLES]
    want.append(('dowel', PCD / 2 * math.cos(math.radians(PIN_ANGLE)), PCD / 2 * math.sin(math.radians(PIN_ANGLE)),
                 PIN + spec.fit_mm))
    t = D.interface['disc_mm']
    rows, worst, dworst = [], 0.0, 0.0
    for name, x0, y0, d in want:
        rad = np.hypot(V[:, 0] - x0, V[:, 1] - y0)
        wall = (np.abs(rad - d / 2) < 0.5) & (V[:, 2] > 1.0) & (V[:, 2] < t - 1.0)
        if wall.sum() < 12:
            rows.append({'hole': name, 'found': False})
            worst = math.inf
            continue
        cx, cy, r = fit_circle(V[wall, 0], V[wall, 1])
        e = math.hypot(cx - x0, cy - y0)
        worst = max(worst, e)
        dworst = max(dworst, abs(2 * r - d))
        rows.append({'hole': name, 'found': True, 'measured_mm': [cx, cy], 'expected_mm': [x0, y0], 'error_mm': e,
                     'diameter_mm': 2 * r, 'expected_d_mm': d, 'wall_vertices': int(wall.sum())})
    ok = dworst <= 0.1
    xs = [r_['expected_mm'][0] for r_ in rows if r_.get('found')]
    ys = [r_['measured_mm'][0] for r_ in rows if r_.get('found')]
    return Outcome(worst, {'holes': rows, 'largest_diameter_error_mm': dworst, 'bracket': label, 'res_mm': 0.25,
                           'sources': [UR5E_URL, ISO_URL]}, ok=ok,
                   figure=figure('Hole centres, x', 'pattern (mm)', 'measured on the mesh (mm)',
                                 [series('holes', xs, ys, 'points')], diag=True),
                   note='UR\'s drawing holds the bolt circle to ±0.1 mm on its diameter (±0.05 mm on a hole\'s '
                        'radius position); a generated hole must sit well inside that, so 0.05 mm. A circle fitted '
                        'to some hundred wall vertices locates its centre far better than the 0.25 mm grid.')
