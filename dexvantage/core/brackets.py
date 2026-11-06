"""Brackets and mounts (spec §7.7): watertight, 3D-printable STL brackets that hold a sensor at its planned pose, with
their dimensions and print notes, and the clearance of bracket and camera body to the robot and its cables.

Ported from Vantage core/brackets.py (the design as signed distance solids meshed by core/meshing.py into a closed,
consistently wound mesh checked edge by edge; the camera plate on a housing face with clearance holes and screw
access bored through the support; the convex-hull support between the plate and the base; print orientation; STL
out). Vantage's automotive parts (vehicle body surfaces, roof bars, a base fitted to a body panel, VHB tape sized for
road shocks, the vehicle frame) are left out. DexVantage's bracket types:

    flange     wrist-flange adapter: a disc on an ISO 9409-1 robot flange (bolt circle, threads, centring and dowel
               from ISO9409, each figure cited), sandwiched between the flange and the gripper, with an arm to the
               camera
    tslot2020  camera clamp on a 20 x 20 T-slot profile (slot figures from the maker's page, TSLOT)
    tslot4040  camera clamp on a 40 x 40 T-slot profile
    pantilt    pan-tilt neck mount: a plate for the pan-tilt unit's own hole pattern (the user's figures)
    palm       palm micro-camera mount: a sleeve around a micro camera on a pad behind it
    dorsal     dorsal-hand micro-camera mount: the sleeve on a pad under it (on the back of the hand)
    none       no printed bracket

Frames and units: the bracket frame is in millimetres with its origin at the centre of the contact face (the face that
sits on the robot flange, the profile or the hand) and z out of that surface into the bracket, so the STL prints with
z up. For a flange adapter it is the flange frame: z along the flange normal, x towards the ISO dowel (ISO 9409-1
clause 4.1: the location pin lies on +Xm). T_parent_bracket maps the bracket frame (metres) into the sensor's mount
frame (the link or the static frame it is mounted on).

The camera body is a box of the library entry's size_mm (x along the optical axis, y across, z up) with the optical
centre of the reference imager (the body origin) at the middle of its front face (an ASSUMPTION, labelled in the
result: makers state the size, rarely where the optics sit in it). Its mounting holes come from the entry's
`mounting` block when the maker's drawing states them (thread, count, face, spacing, insertion depth, position where
dimensioned); a hole whose position is not dimensioned is placed at the face's middle and flagged; a camera with no
stated holes gets a plate without holes, flagged.

Hole sizes in printed parts: a clearance hole is the screw's nominal diameter plus clearance_mm (0.4 mm, a default:
printed holes come out small); centring spigots and dowel holes use fit_mm (0.1 mm, a default). The screwdriver bore
behind a camera screw is twice the nominal diameter plus 1 mm (a default). Every default is a BracketSpec field, named
in the result.

Clearance check (clearance()): the bracket and the camera body against every robot link but the one it is mounted on,
over joint states sampled uniformly within the joint limits (a seeded sample, plus the home pose and the pose on
screen), and against the cable paths when the project has any. The distance is the bracket solid's signed distance at
the links' mesh vertices and triangle centres: its sign (inside or out) is exact at those points and its value is a
lower bound of the true clearance (core/meshing.py).

Known limits: the bracket follows the planned pose (it does not move the camera); the camera body is a box placed by
the assumption above; a flange adapter assumes the mount link's fixed child frame (the gripper mount) is the flange
face (moved to where the gripper's coupling meets the flange in the model), or the link's own frame when it has none,
and that its x axis is ISO's +Xm: check the pattern's angle on the robot; a flange adapter moves the gripper out by
its thickness, which the robot model does not include: the clearance check moves every link below the flange out by
it, and the gripper's contact face on the adapter is its seat; joints are
sampled independently within their limits, so coupled joints (a gripper's linkage) are tried in poses the mechanism
cannot reach (conservative: such a pose can only add collisions), while states in which the robot runs into itself
are dropped (core/robot/collision.py) and more are drawn; points are mesh vertices and triangle centres, so a
long thin triangle edge passing through the bracket between its vertices is missed; a T-slot clamp's position on the
profile, a pan-tilt unit's pattern and a hand's surface are not in the project: the interface is placed from the camera
(side, standoff and tilt settings); printed parts are not load-tested; no figure here is a strength figure.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field

import numpy as np

from . import meshing as M

SCHEMA = 'dexvantage.brackets/1'
READ = '2026-11-06'

# ====================================================================== cited sources
SOURCES = {
    'ur5e_datasheet': {
        'url': 'https://www.universal-robots.com/manuals/latest/en/datasheets/ur5e/',
        'title': 'Universal Robots, UR5e datasheet (the PDF served at that address, "Updated October 2026"): "Tool '
                 'flange EN ISO-9409-1-50-4-M6" and the drawing "Tool flange for UR5e"', 'read': READ},
    'ur_flange_article': {
        'url': 'https://www.universal-robots.com/articles/ur/application-installation/'
               'end-effector-mounting-on-tool-output-flange/',
        'title': 'Universal Robots, "End effector mounting on tool output flange" (UR5e: M6, property class 8.8, '
                 'ISO 4762, 8.0 Nm; "Do not use bolts that extend beyond 8mm to mount the tool.")', 'read': READ},
    'iso9409_preview': {
        'url': 'https://cdn.standards.iteh.ai/samples/36578/348a837664f444bc83e2902c7a5acf4c/ISO-9409-1-2004.pdf',
        'title': 'ISO 9409-1:2004, public preview (iTeh): clause 4.1 (the location pin hole on +Xm), clause 5 '
                 '(designation ISO 9409-1-d1-n-thread), Figure 1 (4 holes at 45 degrees and 4 x 90 degrees, 6 holes '
                 'at 30 degrees and 6 x 60 degrees from the pin, on the d1 circle). Table 1 (the dimensions) is not in '
                 'the preview: the standard is sold.', 'read': READ},
    'motedis_2020': {
        'url': 'https://www.motedis.es/en/Profile-20x20-B-type-slot-6',
        'title': 'Motedis, "Profile 20x20 B-Type Slot 6": "Dimensions: 20x20", "Groove Width: 6mm", '
                 '"Groove Depth: 5.5mm"', 'read': READ},
    'motedis_4040': {
        'url': 'https://www.motedis.es/en/Profile-40x40-B-type-slot-10',
        'title': 'Motedis, "Profile 40x40 B-type slot 10": "Dimensions: 40x40", "Groove width: 10 mm", '
                 '"Groove depth: 12 mm"', 'read': READ},
}

# ISO 9409-1 patterns. Table 1 of the standard is not public; each figure here is read from a public drawing that
# reproduces it (cited per pattern), else None (not stated). The layout (angles, the pin on +X on the pitch circle) is
# from the standard's public Figure 1 and clause 4.1.
ISO9409 = {
    '50-4-M6': {
        'designation': 'ISO 9409-1-50-4-M6', 'pcd_mm': 50.0, 'count': 4, 'thread': 'M6', 'thread_d_mm': 6.0,
        'thread_depth_mm': 6.0, 'first_deg': 45.0, 'pitch_deg': 90.0,
        'pilot_d_mm': 31.5, 'pilot_fit': 'H7', 'pilot_depth_mm': None,
        'spigot_d_mm': 63.0, 'spigot_fit': 'h8', 'spigot_height_mm': 6.5,
        'pin_d_mm': 6.0, 'pin_fit': 'H7', 'pin_depth_mm': 6.2,
        'sources': ('ur5e_datasheet', 'iso9409_preview', 'ur_flange_article'),
        'note': 'Read from Universal Robots\' drawing "Tool flange for UR5e" (UR5e datasheet): "Ø 50 ±0.1" bolt '
                'circle, "M6 - 6H ↓ 6" threads (4 x Ø5 ↓ 6.2 tap drill), "45°" and "4x 90°", "Ø 6 H7 ↓ 6.2±0.20" '
                'dowel hole drawn on the bolt circle, "Ø31.5 H7" centring recess (d3), "Ø63 h8" raised spigot (d2), '
                'standing "6.5 ±0.2" (section A-A). The recess depth is not dimensioned legibly (None). The angles '
                'and the pin on +Xm agree with ISO 9409-1 Figure 1 and clause 4.1 (public preview).',
    },
}
# robots whose flange pattern a cited source states: (regular expression on the robot's name, ISO9409 key, source)
ROBOT_FLANGES = ((r'\bur5e\b', '50-4-M6', 'ur5e_datasheet'),)

TSLOT = {
    'tslot2020': {'profile_mm': 20.0, 'groove_width_mm': 6.0, 'groove_depth_mm': 5.5, 'source': 'motedis_2020'},
    'tslot4040': {'profile_mm': 40.0, 'groove_width_mm': 10.0, 'groove_depth_mm': 12.0, 'source': 'motedis_4040'},
}

TYPES = {
    'flange': 'Wrist-flange adapter (ISO 9409-1)',
    'tslot2020': 'Camera clamp, 2020 T-slot',
    'tslot4040': 'Camera clamp, 4040 T-slot',
    'pantilt': 'Pan-tilt neck mount',
    'palm': 'Palm micro-camera mount',
    'dorsal': 'Dorsal-hand micro-camera mount',
    'none': 'No printed bracket',
}
BUILT = ('flange', 'tslot2020', 'tslot4040', 'pantilt', 'palm', 'dorsal')
SIDES = ('auto', 'back', 'bottom', 'top')
FACES = ('auto', 'back', 'bottom', 'top', 'left', 'right')
MICRO_MM = 5.0                     # a camera no larger than this in every direction gets a sleeve, not a plate
MATERIAL = {'name': 'PETG', 'rho_g_cm3': 1.27,
            'source': 'https://storage.googleapis.com/prusa3d-content-prod-14e8-wordpress-prusament-prod/2023/10/'
                      'c34f5f47-tds_prusament-petg_n_de.pdf',
            'note': 'Density 1.27 g/cm³ as Prusa Polymers states it for Prusament PETG ("Dichte [g/cm3] 1.27", ISO '
                    '1183; technical data sheet updated 16.02.2022, read 2026-11-07); other PETG filaments differ. '
                    'Used only for the mass estimate of a solid print (infill lowers it); not a strength figure.'}

LIMITS = {'wall_mm': (0.8, 12.0), 'margin_mm': (0.0, 20.0), 'clearance_mm': (0.0, 1.5), 'fit_mm': (0.0, 0.5),
          'disc_mm': (3.0, 30.0), 'spigot_mm': (0.0, 10.0), 'centre_hole_mm': (0.0, 60.0),
          'standoff_mm': (0.0, 200.0), 'tilt_deg': (-90.0, 90.0), 'key_mm': (0.0, 10.0),
          'pad_length_mm': (5.0, 300.0), 'pt_pcd_mm': (1.0, 200.0), 'pt_count': (1, 12), 'pt_hole_mm': (0.5, 20.0),
          'pt_first_deg': (-360.0, 360.0), 'pad_hole_mm': (0.0, 10.0), 'pocket_mm': (0.0, 1.0), 'res_mm': (0.05, 2.0),
          'fillet_mm': (0.0, 5.0), 'keepout_mm': (0.0, 400.0)}


@dataclass
class BracketSpec:
    """One sensor's bracket settings. Every figure is a design default (labelled so) or the user's own entry."""
    kind: str = 'none'                 # TYPES
    pattern: str = ''                  # flange: an ISO9409 key or a designation 'd1-n-thread'; '' = the robot's
    side: str = 'auto'                 # where the interface sits on the camera: back | bottom | top (not flange)
    camera_face: str = 'auto'          # the camera face the plate holds
    wall_mm: float = 3.0               # default
    margin_mm: float = 2.0             # plate beyond the camera face (default)
    clearance_mm: float = 0.4          # screw clearance over nominal (default)
    fit_mm: float = 0.1                # centring spigot and dowel fit (default)
    fillet_mm: float = 1.0             # default
    disc_mm: float = 8.0               # flange adapter thickness (default)
    spigot_mm: float = 3.0             # flange adapter centring spigot height (default)
    centre_hole_mm: float = 20.0       # flange adapter cable hole (default; ISO 9409-1 4.4: at most d3)
    standoff_mm: float = 10.0          # interface face to camera face (default)
    tilt_deg: float = 0.0              # interface tilted about the bracket's x axis
    key_mm: float = 2.0                # T-slot key height into the groove (default)
    pad_length_mm: float | None = None   # T-slot pad length (None: twice the profile width, a default)
    pt_pcd_mm: float | None = None     # pan-tilt pattern: the user's figures (None: not given)
    pt_count: int = 4
    pt_hole_mm: float | None = None
    pt_first_deg: float = 0.0
    pad_hole_mm: float = 2.4           # palm/dorsal pad screw holes (default: M2 + 0.4)
    pocket_mm: float = 0.1             # micro-camera pocket clearance (default)
    res_mm: float = 0.5                # mesher grid (default)
    size_mm: list | None = None        # the camera's size when the library does not state it (the user's figure)
    keepout_mm: float | None = None    # flange: diameter kept free above the adapter for the gripper (None: measured)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d) -> 'BracketSpec':
        s = BracketSpec()
        d = d if isinstance(d, dict) else {}
        if d.get('kind') in TYPES:
            s.kind = d['kind']
        if isinstance(d.get('pattern'), str):
            s.pattern = d['pattern'].strip()
        if d.get('side') in SIDES:
            s.side = d['side']
        if d.get('camera_face') in FACES:
            s.camera_face = d['camera_face']
        for k, (lo, hi) in LIMITS.items():
            if k not in d or d[k] is None:
                continue
            try:
                v = float(d[k])
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                v = min(max(v, lo), hi)
                setattr(s, k, int(round(v)) if k == 'pt_count' else v)
        sz = d.get('size_mm')
        if isinstance(sz, (list, tuple)) and len(sz) == 3:
            try:
                v = [float(x) for x in sz]
                if all(math.isfinite(x) and x > 0 for x in v):
                    s.size_mm = v
            except (TypeError, ValueError):
                pass
        return s


class BracketError(ValueError):
    """A bracket that cannot be designed; the message says why in plain words."""


# ====================================================================== small helpers
def lc(text: str) -> str:
    """A title in running text: its first letter lower case ('Wrist-flange adapter (ISO 9409-1)' -> 'wrist-flange
    adapter (ISO 9409-1)')."""
    return text[:1].lower() + text[1:]


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def thread_diameter_mm(thread: str) -> float | None:
    """Nominal diameter of a thread as written: 'M6' -> 6.0 (ISO 261: the number is the nominal diameter in mm),
    '1/4-20' or '1/4-20 UNC' -> 6.35 (a quarter inch, 25.4 mm to the inch); None when it cannot be read."""
    t = str(thread or '').strip().upper()
    m = re.match(r'^M(\d+(?:\.\d+)?)', t)
    if m:
        return float(m.group(1))
    m = re.match(r'^(\d+)\s*/\s*(\d+)\s*-\s*\d+', t)
    if m and int(m.group(2)):
        return 25.4 * int(m.group(1)) / int(m.group(2))
    return None


def iso_pattern(key: str) -> dict:
    """An ISO 9409-1 pattern: the cited table entry, or one read from a designation 'd1-n-Mx' (ISO 9409-1 clause 5)
    with the angles of Figure 1 for 4 or 6 holes; figures no public source gives are None and the note says so."""
    k = str(key or '').strip()
    k = re.sub(r'^(ISO\s*9409-1[-–\s]*)', '', k, flags=re.I).replace('–', '-').replace(' ', '')
    k = re.sub(r'^A', '', k)
    if k in ISO9409:
        return dict(ISO9409[k], key=k, stated=True)
    m = re.match(r'^(\d+(?:\.\d+)?)-(\d+)-(M\d+(?:\.\d+)?)$', k, flags=re.I)
    if not m:
        raise BracketError(f'"{key}" is not an ISO 9409-1 designation: write it as pitch circle-holes-thread, e.g. '
                           '50-4-M6.')
    pcd, n, thr = float(m.group(1)), int(m.group(2)), m.group(3).upper()
    first, pitch = {4: (45.0, 90.0), 6: (30.0, 60.0)}.get(n, (None, None))
    note = (f'ISO 9409-1-{m.group(1)}-{n}-{thr}: the pitch circle, the number of holes and the thread are read from '
            'the designation (ISO 9409-1 clause 5). The centring diameters, the dowel diameter and the depths are in '
            'the standard\'s Table 1, which no public source we found reproduces for this size: they are not '
            'stated, so the adapter has no centring spigot and no dowel hole.')
    if first is None:
        note += f' Figure 1 shows the angles for 4 and 6 holes only; {n} holes are spaced evenly, a half pitch from ' \
                'the pin (an ASSUMPTION).'
        pitch = 360.0 / max(n, 1)
        first = pitch / 2
    return {'designation': f'ISO 9409-1-{m.group(1)}-{n}-{thr}', 'pcd_mm': pcd, 'count': n, 'thread': thr,
            'thread_d_mm': thread_diameter_mm(thr), 'thread_depth_mm': None, 'first_deg': first, 'pitch_deg': pitch,
            'pilot_d_mm': None, 'pilot_fit': None, 'pilot_depth_mm': None, 'spigot_d_mm': None, 'spigot_fit': None,
            'spigot_height_mm': None, 'pin_d_mm': None, 'pin_fit': None, 'pin_depth_mm': None,
            'sources': ('iso9409_preview',), 'note': note, 'key': k, 'stated': False}


def iso_hole_positions(pat: dict) -> tuple[np.ndarray, np.ndarray]:
    """(bolt hole centres (n, 2), pin centre (2,)) in the flange plane, mm: the pin on +x on the pitch circle, the
    holes from first_deg in steps of pitch_deg (ISO 9409-1 Figure 1)."""
    r = pat['pcd_mm'] / 2
    a = np.radians(pat['first_deg'] + pat['pitch_deg'] * np.arange(pat['count']))
    return np.column_stack([r * np.cos(a), r * np.sin(a)]), np.array([r, 0.0])


def robot_flange(robot) -> tuple[str, str] | None:
    """(ISO9409 key, source key) of a robot whose flange pattern a cited source states, by its name."""
    name = str(getattr(robot, 'name', '') or '').lower()
    for rx, key, src in ROBOT_FLANGES:
        if re.search(rx, name):
            return key, src
    return None


def source(key) -> dict:
    return dict(SOURCES[key], key=key)


# ====================================================================== the camera
@dataclass
class CameraGeom:
    size_mm: tuple | None            # (x along the optical axis, y across, z up)
    size_note: str
    holes: list                      # the entry's mounting groups (dicts)
    holes_note: str
    model_id: str = ''
    title: str = ''

    @property
    def micro(self) -> bool:
        return self.size_mm is not None and max(self.size_mm) <= MICRO_MM


def camera_geom(sensor, spec: BracketSpec | None = None) -> CameraGeom:
    """The camera body's size and stated mounting holes from the library entry (or the user's size in spec)."""
    from . import library as L
    mid = getattr(sensor, 'model_id', '') or ''
    entry = L.get(mid) if mid else None
    title = entry.title if entry is not None else ''
    size, note = None, ''
    if spec is not None and spec.size_mm:
        size, note = tuple(float(v) for v in spec.size_mm), 'the size entered in the bracket settings (your figure)'
    elif entry is not None and entry.data.get('size_mm'):
        size, note = tuple(float(v) for v in entry.data['size_mm']), f'size_mm of the library entry {mid}'
    elif entry is None:
        note = 'no library device is chosen for this sensor, so its size is not known: choose its device'
    else:
        note = f'the library entry {mid} does not state its size (not stated by the maker): enter it'
    holes, hnote = [], ''
    if entry is not None:
        holes = [dict(h) for h in ((entry.data.get('mounting') or {}).get('holes') or [])]
        hnote = (f'mounting holes as {entry.title}\'s drawing states them' if holes else
                 f'{entry.title}: no mounting holes stated in the library entry')
    else:
        hnote = 'no library device: no mounting holes known'
    return CameraGeom(size, note, holes, hnote, mid, title)


FACE_GEOM = {     # outward normal (body), in-face axes u (along which holes are spaced), v
    'back': ((-1, 0, 0), (0, 1, 0), (0, 0, 1)),
    'front': ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    'bottom': ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
    'top': ((0, 0, 1), (0, 1, 0), (1, 0, 0)),
    'left': ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
    'right': ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
}


def box_of(size) -> tuple[np.ndarray, np.ndarray]:
    """(centre (3,), half sizes (3,)) of the camera box in the body frame, mm: front face at x = 0, centred in y, z
    (the ASSUMPTION in the module docstring)."""
    sx, sy, sz = size
    return np.array([-sx / 2, 0.0, 0.0]), np.array([sx / 2, sy / 2, sz / 2])


def face_frame(size, face) -> tuple:
    """(outward normal n, centre c, u, v, half extent along u, along v) of a box face in the body frame, mm."""
    c0, h = box_of(size)
    n, u, v = (np.array(x, float) for x in FACE_GEOM[face])
    c = c0 + n * (np.abs(n) @ h)
    return n, c, u, v, float(np.abs(u) @ h), float(np.abs(v) @ h)


def stated_holes(cam: CameraGeom, face: str) -> tuple[list, list]:
    """The face's holes in the body frame: ([{'thread', 'd_mm', 'centre' (3,), 'axis' (3,) outward, 'max_insert_mm',
    'position_stated'}], [flag texts])."""
    if cam.size_mm is None:
        return [], []
    n, c, u, v, hu, hv = face_frame(cam.size_mm, face)
    out, flags = [], []
    for g in cam.holes:
        if g.get('face') != face:
            continue
        d = thread_diameter_mm(g.get('thread'))
        if d is None:
            flags.append(f'The {g.get("thread")} thread on the {face} face cannot be read as a size: left out.')
            continue
        centre = c.copy()
        stated = True
        for k, ax in (('x_mm', 0), ('y_mm', 1), ('z_mm', 2)):
            if abs(n[ax]) > 0.5:
                continue                          # the face's own coordinate
            val = g.get(k)
            if val is None:
                stated = False
            else:
                centre[ax] = float(val)
        cnt = int(g.get('count') or 1)
        sp = g.get('spacing_mm')
        if cnt > 1 and sp is None:
            flags.append(f'{cnt} × {g.get("thread")} on the {face} face: their spacing is not stated: left out.')
            continue
        offs = [0.0] if cnt == 1 else list((np.arange(cnt) - (cnt - 1) / 2) * float(sp))
        for o in offs:
            out.append({'thread': str(g.get('thread')), 'd_mm': d, 'centre': centre + u * o, 'axis': n.copy(),
                        'max_insert_mm': g.get('max_insert_mm'), 'position_stated': stated})
        if not stated:
            flags.append(f'The {cnt} × {g.get("thread")} on the {face} face: the maker\'s drawing does not dimension '
                         'where they sit on the face, so they are drawn at its middle: check against the camera '
                         'before printing.')
    return out, flags


def choose_face(cam: CameraGeom, toward_iface_body: np.ndarray, want: str = 'auto') -> tuple[str, str]:
    """(face, why): the camera face the plate holds. auto: among the faces with stated holes, the one that faces the
    interface best (the plate then lies between the camera and the bracket's base); without stated holes, the face
    that faces the interface."""
    if want in FACE_GEOM:
        return want, 'chosen in the settings'
    t = _unit(toward_iface_body)
    score = {f: float(np.array(FACE_GEOM[f][0], float) @ t) for f in ('back', 'bottom', 'top', 'left', 'right')}
    with_holes = [f for f in score if any(g.get('face') == f for g in cam.holes)]
    good = [f for f in with_holes if score[f] > -0.2]
    if good:
        f = max(good, key=lambda k: (score[k], sum(int(g.get('count') or 1) for g in cam.holes if g.get('face') == k)))
        return f, 'the face with stated mounting holes that faces the bracket\'s base'
    f = max(score, key=score.get)
    return f, 'the face towards the bracket\'s base (no stated holes on a face that faces it)'


# ====================================================================== the design
@dataclass
class Design:
    kind: str
    solid: object = None                       # meshing.Solid (bracket frame, mm)
    T_b_body: np.ndarray = field(default_factory=lambda: np.eye(4))    # camera body in the bracket frame (mm)
    camera_box: tuple | None = None            # (centre, axes (columns), half) in the bracket frame
    holes: list = field(default_factory=list)  # [{'role', 'label', 'centre', 'axis', 'd_mm', 'through'}]
    interface: dict = field(default_factory=dict)
    plate: dict = field(default_factory=dict)
    keepout: dict | None = None                # flange: the coupling cylinder {'r_mm', 'z0_mm'}
    flags: list = field(default_factory=list)  # plain warnings
    notes: list = field(default_factory=list)  # print and assembly notes
    pattern: dict | None = None
    res_mm: float = 0.5


def _rod_z(x, y, z0, z1, r):
    return M.rod([x, y, z0], [x, y, z1], r)


def _frame_from_side(cam: CameraGeom, side: str, standoff: float, tilt_deg: float, extra: float = 0.0) -> np.ndarray:
    """T_body_bracket (mm) for an interface on the camera's side face at standoff (+ extra) from it: z towards the
    camera, x along the camera's y for a back interface (a profile across the camera), along its x otherwise."""
    n, c, _u, _v, _hu, _hv = face_frame(cam.size_mm, side)
    z = -n
    x = np.array([0.0, 1.0, 0.0]) if side in ('back', 'front') else np.array([1.0, 0.0, 0.0])
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    if tilt_deg:
        a = math.radians(tilt_deg)
        Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
        R = R @ Rx
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = c + n * (standoff + extra)
    return T


def auto_side(cam: CameraGeom, kind: str) -> str:
    """The camera side an interface sits on when the setting is auto: a T-slot clamp or a palm mount behind the camera
    (back), a pan-tilt neck or a dorsal mount under it (bottom)."""
    return 'back' if kind in ('tslot2020', 'tslot4040', 'palm') else 'bottom'


def _flange_core(spec: BracketSpec, pat: dict) -> tuple:
    """The flange adapter's disc: (parts [Solid], cuts [Solid], holes, flags, figures {'R', 'td', 'rb', 'rp', 'rc',
    'rs', 'rr', 'h'}) in the flange frame (mm, z out of the robot)."""
    t = spec.wall_mm
    R = float((pat['spigot_d_mm'] / 2) if pat['spigot_d_mm'] else (pat['pcd_mm'] / 2 + pat['thread_d_mm'] + t))
    td = spec.disc_mm
    parts, cuts, holes, flags = [_rod_z(0, 0, 0.0, td, R)], [], [], []
    bolts, pin = iso_hole_positions(pat)
    rb = (pat['thread_d_mm'] + spec.clearance_mm) / 2
    z_lo = -(spec.spigot_mm + 1.0)
    for k, (x, y) in enumerate(bolts):
        cuts.append(_rod_z(x, y, z_lo, td + 1.0, rb))
        holes.append({'role': 'flange_bolt', 'label': f'{pat["thread"]} clearance (bolt {k + 1})',
                      'centre': [float(x), float(y), 0.0], 'axis': [0.0, 0.0, 1.0],
                      'd_mm': 2 * rb, 'through': True, 'stated': True})
    rp = rs = rr = None
    if pat['pin_d_mm']:
        rp = (pat['pin_d_mm'] + spec.fit_mm) / 2
        cuts.append(_rod_z(pin[0], pin[1], z_lo, td + 1.0, rp))
        holes.append({'role': 'flange_pin', 'label': f'Ø{pat["pin_d_mm"]:g} dowel',
                      'centre': [float(pin[0]), float(pin[1]), 0.0], 'axis': [0.0, 0.0, 1.0],
                      'd_mm': 2 * rp, 'through': True, 'stated': True})
    else:
        flags.append('The dowel diameter of this flange pattern is not stated by a public source: no dowel '
                     'hole; the adapter is located by its bolts only.')
    rc = spec.centre_hole_mm / 2
    h = spec.spigot_mm
    if pat['pilot_d_mm']:
        rs = pat['pilot_d_mm'] / 2 - spec.fit_mm / 2
        rr = pat['pilot_d_mm'] / 2 + spec.fit_mm / 2
        rc = min(rc, rs - 1.5)
        if h > 0:
            parts.append(_rod_z(0, 0, -h, min(1.0, td / 2), rs))       # overlaps the disc: one solid
            # the same recess on top for the gripper's own centring spigot
            cuts.append(_rod_z(0, 0, td - h, td + 1.0, rr))
            if pat.get('pilot_depth_mm') is None:
                flags.append(f'The depth of the flange\'s Ø{pat["pilot_d_mm"]:g} centring recess is not stated: '
                             f'the adapter\'s spigot is {h:g} mm high (a default): check that it does not bottom '
                             'out.')
    else:
        flags.append('The centring diameter of this flange pattern is not stated by a public source: no '
                     'centring spigot.')
    if rc > 0:
        cuts.append(_rod_z(0, 0, z_lo - 1, td + 1.0, rc))
    return parts, cuts, holes, flags, {'R': R, 'td': td, 'rb': rb, 'rp': rp, 'rc': max(rc, 0.0), 'rs': rs, 'rr': rr,
                                       'h': h}


def flange_plate(spec: BracketSpec | None = None, pattern: str = '50-4-M6'):
    """The flange adapter's disc alone, without the arm: (Solid, figures) — for validation (its volume has a closed
    form)."""
    spec = spec or BracketSpec(kind='flange', fillet_mm=0.0)
    pat = iso_pattern(pattern)
    parts, cuts, holes, _flags, fi = _flange_core(spec, pat)
    fi = dict(fi, holes=holes, pattern=pat)
    return M.subtract(M.union(*parts), *cuts), fi


def design(spec: BracketSpec, cam: CameraGeom, T_b_body: np.ndarray | None = None) -> Design:
    """The bracket's solid and features. T_b_body (4 x 4, mm): the camera body in the bracket frame; required for a
    flange adapter (from the flange frame), computed from side/standoff/tilt for the other kinds when None."""
    kind = spec.kind
    if kind not in BUILT:
        raise BracketError(f'{TYPES.get(kind, kind)}: nothing to print.')
    if cam.size_mm is None:
        raise BracketError(f'The camera\'s size is needed to size the bracket and {cam.size_note}.')
    D = Design(kind)
    t, mrg, fil = spec.wall_mm, spec.margin_mm, spec.fillet_mm
    micro = kind in ('palm', 'dorsal')
    if micro and not cam.micro:
        D.flags.append(f'The {lc(TYPES[kind])} is meant for a micro camera (at most {MICRO_MM:g} mm); this one is '
                       f'{" × ".join(f"{v:g}" for v in cam.size_mm)} mm.')
    side = None
    if kind != 'flange':
        side = spec.side if spec.side != 'auto' else auto_side(cam, kind)
        extra = (t + spec.pocket_mm) if micro else 0.0
        T_body_b = _frame_from_side(cam, side, spec.standoff_mm, spec.tilt_deg, extra)
        T_b_body = np.linalg.inv(T_body_b)
    elif T_b_body is None:
        raise BracketError('A flange adapter needs the camera\'s pose relative to the flange.')
    D.T_b_body = np.asarray(T_b_body, float)
    Rb, tb = D.T_b_body[:3, :3], D.T_b_body[:3, 3]

    def to_b(p):
        return np.asarray(p, float) @ Rb.T + tb
    c0, half = box_of(cam.size_mm)
    D.camera_box = (to_b(c0), Rb.copy(), half.copy())
    cam_box = M.box(*D.camera_box)
    toward_iface = Rb.T @ (-(to_b(c0)))           # body-frame direction from the camera towards the bracket origin
    # ---------------------------------------------------------------- the camera holder
    access_cuts, hole_cuts = [], []
    if micro:
        p = spec.pocket_mm
        sleeve_half = half + np.array([t / 2, p + t, p + t])
        sleeve_c = c0 + np.array([-t / 2, 0.0, 0.0])      # flush with the camera front, a back wall behind it
        holder = M.box(to_b(sleeve_c), Rb, sleeve_half, r=min(fil, 0.3 * t))
        tunnel = M.box(to_b(c0 + np.array([0.0, 0, 0])), Rb, np.array([half[0] + 2 * t + 5, half[1] + p, half[2] + p]))
        hole_cuts.append(tunnel)
        sc = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * sleeve_half
        holder_pts = to_b(sleeve_c + sc)
        D.plate = {'kind': 'sleeve', 'pocket_mm': [float(v) for v in 2 * (half + p)], 'wall_mm': t,
                   'outer_mm': [float(v) for v in 2 * sleeve_half]}
        D.notes.append(f'The camera slides into the {2 * (half[1] + p):.2f} × {2 * (half[2] + p):.2f} mm tunnel from '
                       'the back and is glued; its cable leaves through the back. Features this small need a resin '
                       'printer, or drill the tunnel out.')
    else:
        face, why = choose_face(cam, toward_iface, spec.camera_face)
        n, fc, u, v, hu, hv = face_frame(cam.size_mm, face)
        holes, hflags = stated_holes(cam, face)
        D.flags += hflags
        pax = np.stack([Rb @ u, Rb @ v, Rb @ n], axis=1)
        need_u = max([abs((h['centre'] - fc) @ u) + h['d_mm'] + 1.5 for h in holes] + [0.0])
        need_v = max([abs((h['centre'] - fc) @ v) + h['d_mm'] + 1.5 for h in holes] + [0.0])
        pu, pv = max(hu + mrg, need_u), max(hv + mrg, need_v)
        phalf = np.array([pu, pv, t / 2])
        pc = to_b(fc + n * t / 2)
        holder = M.box(pc, pax, phalf, r=min(fil, t / 2 - 0.1) if fil > 0 else 0.0)
        holder_pts = np.array([pc + pax @ (phalf * np.array([a, b, 1.0])) for a in (-1, 1) for b in (-1, 1)])
        for h in holes:
            cb = to_b(h['centre'])
            ax = Rb @ h['axis']
            dh = h['d_mm'] + spec.clearance_mm
            hole_cuts.append(M.rod(cb - ax * 1.0, cb + ax * (t + 0.5), dh / 2))
            access_cuts.append(M.rod(cb + ax * t, cb + ax * 400.0, (2 * h['d_mm'] + 1.0) / 2))
            D.holes.append({'role': 'camera', 'label': f'{h["thread"]} clearance', 'centre': cb.tolist(),
                            'axis': ax.tolist(), 'd_mm': dh, 'through': True, 'stated': h['position_stated']})
        if not holes:
            D.flags.append(f'No mounting holes are stated for the camera\'s {face} face ({cam.holes_note}): the plate '
                           'has none: drill them to your camera, or glue it.')
        mi = [h['max_insert_mm'] for h in holes if h.get('max_insert_mm') is not None]
        D.plate = {'kind': 'plate', 'face': face, 'why': why, 'size_mm': [2 * pu, 2 * pv], 'thickness_mm': t,
                   'centre': pc.tolist(), 'normal': (Rb @ n).tolist(), 'u': (Rb @ u).tolist(),
                   'v': (Rb @ v).tolist(), 'holes': len(holes)}
        if holes:
            sc = sorted({h['thread'] for h in holes})
            txt = f'Screw the camera to the plate with {len(holes)} × {"/".join(sc)}'
            if mi:
                txt += (f' screws reaching at most {min(mi):g} mm into the camera (the maker\'s maximum insertion '
                        f'depth): screw length = {t:g} mm plate + up to {min(mi):g} mm.')
            else:
                txt += ' screws (the maker states no insertion depth: do not force them).'
            D.notes.append(txt + ' The heads are reached through the bores in the support.')
    # ---------------------------------------------------------------- the interface
    inter_cuts = []
    up = np.array([0.0, 0.0, 1.0])
    keep_out = None
    if kind == 'flange':
        pat = iso_pattern(spec.pattern or '50-4-M6')
        D.pattern = pat
        parts, inter_cuts, fholes, fflags, fi = _flange_core(spec, pat)
        D.holes += fholes
        D.flags += fflags
        R, td, rb, rc = fi['R'], fi['td'], fi['rb'], fi['rc']
        # the arm: a tab on the rim towards the camera
        cxy = D.camera_box[0][:2]
        phi = math.atan2(cxy[1], cxy[0]) if np.hypot(*cxy) > 1e-6 else 0.0
        er = np.array([math.cos(phi), math.sin(phi), 0.0])
        et = np.array([-math.sin(phi), math.cos(phi), 0.0])
        Rk = max(R, (spec.keepout_mm or 0.0) / 2)
        tab_len = max(2 * t, 6.0) + (Rk - R)
        tab_w = max(3 * t, 12.0)
        tab_c = er * (R + tab_len / 2 - 2.0) + up * td / 2
        tab = M.box(tab_c, np.stack([er, et, up], axis=1), np.array([tab_len / 2 + 2.0, tab_w / 2, td / 2]))
        parts.append(tab)
        anchor = np.array([tab_c + er * a * (tab_len / 2 + 2.0) + et * b * tab_w / 2 + up * (c * td / 2)
                           for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
        anchor = anchor[np.linalg.norm(anchor[:, :2], axis=1) >= Rk - 0.5]
        iface = M.union(*parts)
        keep_out = {'r_mm': Rk, 'z0_mm': td, 'disc_r_mm': R}
        D.keepout = keep_out
        D.interface = {'kind': 'flange', 'designation': pat['designation'], 'disc_d_mm': 2 * R, 'disc_mm': td,
                       'pcd_mm': pat['pcd_mm'], 'bolt_holes': pat['count'], 'bolt_hole_d_mm': 2 * rb,
                       'spigot_d_mm': (pat['pilot_d_mm'] - spec.fit_mm) if pat['pilot_d_mm'] else None,
                       'spigot_mm': spec.spigot_mm if pat['pilot_d_mm'] else 0.0, 'centre_hole_mm': 2 * rc,
                       'pin_hole_d_mm': (pat['pin_d_mm'] + spec.fit_mm) if pat['pin_d_mm'] else None,
                       'arm_angle_deg': math.degrees(phi), 'keepout_d_mm': 2 * Rk}
        D.notes.append(f'Fit the adapter between the robot flange and the gripper: {pat["count"]} × {pat["thread"]} '
                       f'bolts through the adapter and the gripper\'s coupling, {td:g} mm longer than without it. '
                       'Universal Robots: "Do not use bolts that extend beyond 8mm to mount the tool" (M6, property '
                       'class 8.8, ISO 4762, 8.0 Nm). The gripper sits '
                       f'{td:g} mm further out: add it to the tool centre point.')
        if pat['pin_d_mm']:
            D.notes.append(f'Press a Ø{pat["pin_d_mm"]:g} mm dowel through the adapter into the flange\'s dowel hole: '
                           'it sets the camera\'s angle about the flange axis.')
    elif kind in ('tslot2020', 'tslot4040'):
        ts = TSLOT[kind]
        w = ts['profile_mm']
        dh = ts['groove_width_mm']
        # the bolts go beyond the camera holder's ends, so a screwdriver reaches them past the camera
        xb = float(np.abs(holder_pts[:, 0]).max()) + (2 * dh + 1.0) / 2 + 0.5
        L = spec.pad_length_mm or max(2 * w, 2 * (xb + dh / 2 + t))
        pad = M.box([0, 0, t / 2], np.eye(3), [L / 2, w / 2, t / 2], r=0.0)
        kw = ts['groove_width_mm'] - spec.clearance_mm
        kh = spec.key_mm
        parts = [pad]
        if kh > 0:
            parts.append(M.box([0, 0, (min(1.0, t / 2) - kh) / 2], np.eye(3),
                               [L / 2, kw / 2, (kh + min(1.0, t / 2)) / 2]))      # overlaps the pad: one solid
            if kh >= ts['groove_depth_mm']:
                D.flags.append(f'The key ({kh:g} mm) is as deep as the groove ({ts["groove_depth_mm"]:g} mm): '
                               'lower it.')
        xe = min(xb, L / 2 - dh / 2 - t)
        xs = [-xe, xe] if xe > dh else [0.0]
        for k, x in enumerate(xs):
            inter_cuts.append(_rod_z(x, 0, -kh - 1, t + 1, dh / 2))
            D.holes.append({'role': 'tslot_bolt', 'label': f'Ø{dh:g} for a T-nut bolt', 'centre': [x, 0.0, 0.0],
                            'axis': [0.0, 0.0, 1.0], 'd_mm': dh, 'through': True, 'stated': True})
        iface = M.union(*parts)
        anchor = np.array([[a * L / 2, b * w / 2, t] for a in (-1, 1) for b in (-1, 1)])
        D.interface = {'kind': kind, 'profile_mm': w, 'groove_width_mm': ts['groove_width_mm'],
                       'groove_depth_mm': ts['groove_depth_mm'], 'pad_mm': [L, w, t], 'key_mm': [L, kw, kh],
                       'bolt_hole_d_mm': dh, 'bolts': len(xs), 'side': side, 'source': ts['source']}
        D.notes.append(f'Clamp to the {int(w)} × {int(w)} profile with {len(xs)} drop-in T-nuts and bolts: the holes '
                       f'are Ø{dh:g} mm, the slot\'s width, so any bolt that fits the slot passes. The {kw:g} mm key '
                       'runs in the slot and keeps the camera square to the profile.')
    else:                         # pantilt, palm, dorsal: a pad
        holes = []
        if kind == 'pantilt':
            if spec.pt_pcd_mm and spec.pt_hole_mm:
                n_ = max(int(spec.pt_count), 1)
                a = np.radians(spec.pt_first_deg + 360.0 / n_ * np.arange(n_))
                r = spec.pt_pcd_mm / 2
                holes = [(r * math.cos(x), r * math.sin(x), spec.pt_hole_mm) for x in a]
            else:
                D.flags.append('The pan-tilt unit\'s hole pattern is not entered (pitch circle, count, hole '
                               'diameter): the pad has no holes. Enter your unit\'s figures from its drawing.')
        else:
            if spec.pad_hole_mm > 0:
                holes = 'pad'
        hb = np.array([np.abs(holder_pts[:, 0]).max(), np.abs(holder_pts[:, 1]).max()])
        if holes == 'pad':
            dh = spec.pad_hole_mm
            hx = hb[0] + dh / 2 + t
            holes = [(-hx, 0.0, dh), (hx, 0.0, dh)]
        ext = np.array([max([abs(h[0]) + h[2] / 2 + t for h in holes] + [hb[0] + mrg]),
                        max([abs(h[1]) + h[2] / 2 + t for h in holes] + [hb[1] + mrg])])
        pad = M.box([0, 0, t / 2], np.eye(3), [ext[0], ext[1], t / 2], r=0.0)
        for k, (x, y, dh) in enumerate(holes):
            inter_cuts.append(_rod_z(x, y, -1.0, t + 1.0, dh / 2))
            D.holes.append({'role': 'pad', 'label': f'Ø{dh:g} pad hole', 'centre': [x, y, 0.0],
                            'axis': [0.0, 0.0, 1.0], 'd_mm': dh, 'through': True, 'stated': kind != 'pantilt'})
        iface = pad
        anchor = np.array([[a * ext[0], b * ext[1], t] for a in (-1, 1) for b in (-1, 1)])
        D.interface = {'kind': kind, 'pad_mm': [2 * ext[0], 2 * ext[1], t], 'holes': len(holes), 'side': side,
                       'pattern': ({'pcd_mm': spec.pt_pcd_mm, 'count': spec.pt_count, 'hole_mm': spec.pt_hole_mm,
                                    'first_deg': spec.pt_first_deg, 'source': 'your figures'}
                                   if kind == 'pantilt' else None)}
        if kind == 'pantilt':
            D.notes.append('Bolt the pad to the pan-tilt unit\'s output with the unit\'s own screws.')
        else:
            where = 'palm' if kind == 'palm' else 'back of the hand'
            D.notes.append(f'Fix the pad to the {where} with {len(holes)} screws through its Ø{spec.pad_hole_mm:g} '
                           'mm holes (or bond it).')
    # ---------------------------------------------------------------- screw access above the base's holes
    above = M.union(holder, cam_box)
    blocked = []
    for h in D.holes:
        if h['role'] not in ('tslot_bolt', 'pad', 'flange_bolt'):
            continue
        x, y, _z = h['centre']
        top = float(max(holder.hi[2], D.camera_box[0][2] + np.abs(D.camera_box[1][2]) @ D.camera_box[2])) + 5.0
        rad = (2 * h['d_mm'] + 1.0) / 2
        if h['role'] != 'flange_bolt':
            access_cuts.append(_rod_z(x, y, 0.5, top + 400.0, rad))
        zs = np.arange(0.5, top, 0.5)
        line = np.column_stack([np.full(len(zs), x), np.full(len(zs), y), zs])
        h['access'] = not (len(zs) and float(above(line).min()) < rad - 0.5)
        if not h['access']:
            blocked.append(h['label'])
    if blocked:
        what = blocked[0] if len(set(blocked)) == 1 else 'base hole'
        D.flags.append(f'{len(blocked)} × {what} under the camera or its plate: a screwdriver cannot reach '
                       f'{"it" if len(blocked) == 1 else "them"} from above; fit those screws before the camera, or '
                       'move the base (standoff, side).')
    # ---------------------------------------------------------------- the support and the solid
    support = None
    pts = np.vstack([holder_pts, anchor])
    if np.ptp(pts, axis=0).min() > 0.5:
        try:
            support = M.hull(pts)
        except Exception:                               # noqa: BLE001 — a flat point set: no support, said below
            support = None
    body = M.round_union(iface, holder, fil) if fil > 0 else M.union(iface, holder)
    if support is not None:
        sup = M.subtract(support, *access_cuts) if access_cuts else support
        cut_sup = [M.halfspace([0, 0, 0.0], [0, 0, 1.0])]           # remove z < 0 (behind the mount surface)
        if keep_out is not None:
            cut_sup.append(_rod_z(0, 0, keep_out['z0_mm'], keep_out['z0_mm'] + 2000.0, keep_out['r_mm']))
        sup = _clip(sup, cut_sup)
        body = M.round_union(body, sup, fil) if fil > 0 else M.union(body, sup)
    grown = M.box(D.camera_box[0], D.camera_box[1], D.camera_box[2] + 0.2)
    D.solid = M.subtract(body, *(inter_cuts + hole_cuts + [grown]))
    D.res_mm = _resolution(spec, cam, micro)
    if not micro:
        D.flags += _camera_into_interface(D, cam_box, kind)
    return D


def _clip(s, cuts):
    """s with the regions of cuts removed (cuts may be unbounded half-spaces: evaluated everywhere)."""
    def f(P):
        out = s.f(P)
        for c in cuts:
            out = np.maximum(out, -c.f(P))
        return out
    return M.Solid(f, s.lo, s.hi)


def _resolution(spec, cam, micro) -> float:
    """The mesher grid: the setting, or finer for a micro camera (a quarter of its smallest side, at least 0.05 mm)."""
    r = spec.res_mm
    if micro and cam.size_mm:
        r = min(r, max(0.05, min(cam.size_mm) / 4, spec.pocket_mm))
        r = min(r, max(0.05, spec.wall_mm / 3))
    return float(r)


def _camera_into_interface(D: Design, cam_box, kind) -> list:
    """A flag when the camera body overlaps the interface region (the flange disc, the profile, the pad) or the
    mount surface below it."""
    c, A, half = D.camera_box
    corners = np.array([c + A @ (half * np.array([a, b, d])) for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)])
    zmin = float(corners[:, 2].min())
    top = D.interface.get('disc_mm') or (D.interface.get('pad_mm') or [0, 0, 0])[2]
    out = []
    if kind == 'flange':
        r = np.linalg.norm(corners[:, :2], axis=1)
        inside = (r < D.keepout['r_mm']) & (corners[:, 2] < 2000)
        if zmin < top and inside.any():
            out.append('The camera body overlaps the flange adapter or the gripper\'s coupling space above it: move '
                       'the camera out from the flange axis.')
        elif zmin < -0.05:
            out.append(f'The camera body reaches {-zmin:.1f} mm behind the flange face, into the robot: move it out.')
    elif zmin < top - 0.01:
        out.append(f'The camera body reaches {top - zmin:.1f} mm into the base: raise the standoff.')
    return out


# ====================================================================== mesh, checks, print
def mesh_design(D: Design, res_mm: float | None = None, progress=None, cancel=None):
    """(V mm, F) of the design's solid, and its check (core/meshing.check). (None, None, None) when cancelled."""
    h = float(res_mm or D.res_mm)
    V, F = M.mesh(D.solid, h, progress=progress, cancel=cancel)
    if V is None:
        return None, None, None
    chk = M.check(V, F)
    chk['mass_g'] = chk['volume_mm3'] / 1000.0 * MATERIAL['rho_g_cm3']
    chk['res_mm'] = h
    return V, F, chk


def print_orientation(V, F) -> list:
    """For each piece of the mesh: the up direction (bracket frame) with the least overhang (faces turned more than
    45 degrees down that are not on the bed) and that area (mm²). Ported from Vantage core/brackets.py."""
    V, F = np.asarray(V, float), np.asarray(F, np.int64)
    tri = V[F]
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ar = np.linalg.norm(cr, axis=1) / 2
    n = cr / np.maximum(2 * ar, 1e-12)[:, None]
    cands = {'with the base on the bed': np.array([0, 0, 1.0]), 'upside down': np.array([0, 0, -1.0]),
             'on its side (+x down)': np.array([-1.0, 0, 0]), 'on its side (−x down)': np.array([1.0, 0, 0]),
             'on its side (+y down)': np.array([0, -1.0, 0]), 'on its side (−y down)': np.array([0, 1.0, 0])}
    out = []
    for faces in M.shells(V, F):
        best = None
        for name, up in cands.items():
            hmin = (tri[faces].reshape(-1, 3) @ up).min()
            fh = (tri[faces] @ up).max(axis=1)
            nu = n[faces] @ up
            on_bed = (nu < -0.99) & (fh < hmin + 0.3)
            over = (nu < -0.7071) & ~on_bed
            a = float(ar[faces][over].sum())
            key = (round(a, 0), 0 if name == 'with the base on the bed' else 1)
            if best is None or key < best[0]:
                best = (key, name, up, a)
        out.append({'faces': int(len(faces)), 'orientation': best[1], 'up': best[2].tolist(),
                    'overhang_mm2': best[3]})
    out.sort(key=lambda d: -d['faces'])
    return out


def stl_bytes(V, F, label: str) -> bytes:
    return M.stl_bytes(V, F, f'DexVantage bracket {label}: mm, bracket frame, z up')


def safe_name(text) -> str:
    t = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(text)).strip('._')
    return t or 'bracket'


# ====================================================================== placing it on the rig
def flange_frame(project, sensor, pcd_mm: float = 50.0) -> tuple[np.ndarray, str]:
    """(T_link_flange (4 x 4, m), what it is): the flange face of the sensor's mount link. When the link has a fixed
    child (the gripper's coupling in a robot file), the face is where that coupling meets the link: the child's frame
    moved along its z to the lowest point of the meshes below the link within the pattern's pitch circle (the
    coupling's contact face; a robot file may put the child's frame a few millimetres above it). Without a fixed child,
    the link's own frame (an ASSUMPTION, said in the note)."""
    rob = project.robot(sensor.mount.robot_uid)
    link = sensor.mount.parent_frame
    from .frames import Pose
    for j in (rob.joints if rob is not None else []):
        if j.get('parent') == link and j.get('type') == 'fixed':
            T = Pose.from_dict(j.get('origin') or {}).matrix()
            note = f'the frame of {j.get("child")}, the fixed child of {link}'
            off = _coupling_face(project, sensor, T, pcd_mm)
            if off is not None and abs(off) > 1e-4:
                T = T @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, off], [0, 0, 0, 1.0]])
                note += (f', moved {off * 1000:+.1f} mm along its axis to the face where the gripper\'s coupling '
                         'meets the flange in the robot model')
            return T, note + ' (taken as the flange face)'
    return np.eye(4), f'{link}\'s own frame (it has no fixed child frame: taken as the flange face, an ASSUMPTION)'


def _coupling_face(project, sensor, T_link_child, pcd_mm) -> float | None:
    """The z (m, in the child frame) of the lowest mesh point below the mount link within the pitch circle, at home;
    None when there is none or it lies more than 30 mm from the child frame."""
    from .robot.model import RobotModelError, get_model
    rob = project.robot(sensor.mount.robot_uid)
    try:
        m = get_model(project, rob.uid)
    except (RobotModelError, AttributeError):
        return None
    below = descendants(rob, sensor.mount.parent_frame)
    if not below or not m.n_triangles:
        return None
    tri, _n, li = m.posed_triangles(m.home)
    sel = np.array([m.link_names[int(k)] in below for k in li], bool)
    if not sel.any():
        return None
    T = m.link_pose(m.home, sensor.mount.parent_frame) @ T_link_child
    P = (np.asarray(tri[sel], float).reshape(-1, 3) - T[:3, 3]) @ T[:3, :3]
    P = P[np.hypot(P[:, 0], P[:, 1]) <= pcd_mm / 2000.0]
    if not len(P):
        return None
    z = float(P[:, 2].min())
    return z if abs(z) <= 0.03 else None


def placement(project, sensor, spec: BracketSpec, cam: CameraGeom) -> dict:
    """{'T_parent_bracket' (m), 'T_b_body' (mm) or None, 'note'}: where the bracket sits in the sensor's mount frame."""
    Tm = sensor.mount.pose.matrix()
    if spec.kind == 'flange':
        if not sensor.mount.on_link:
            raise BracketError(f'{sensor.name} is not mounted on a robot link: a flange adapter fits a robot flange.')
        try:
            pcd = iso_pattern(spec.pattern or '50-4-M6')['pcd_mm']
        except BracketError:
            pcd = 50.0
        Tlf, note = flange_frame(project, sensor, pcd)
        Tfb = np.linalg.inv(Tlf) @ Tm
        Tfb_mm = Tfb.copy()
        Tfb_mm[:3, 3] *= 1000.0
        return {'T_parent_bracket': Tlf, 'T_b_body': Tfb_mm, 'note': note}
    side = spec.side if spec.side != 'auto' else auto_side(cam, spec.kind)
    micro = spec.kind in ('palm', 'dorsal')
    extra = (spec.wall_mm + spec.pocket_mm) if micro else 0.0
    T_body_b = _frame_from_side(cam, side, spec.standoff_mm, spec.tilt_deg, extra)
    T = T_body_b.copy()
    T[:3, 3] /= 1000.0
    return {'T_parent_bracket': Tm @ T, 'T_b_body': None, 'note': f'placed on the camera\'s {side} side'}


# ====================================================================== clearance
def body_solid(D: Design):
    """The bracket and the camera body as one solid (bracket frame, mm)."""
    return M.union(D.solid, M.box(*D.camera_box))


def _bound(D: Design) -> tuple[np.ndarray, float]:
    lo, hi = D.solid.lo, D.solid.hi
    c, A, h = D.camera_box
    e = np.abs(A) @ h
    lo, hi = np.minimum(lo, c - e), np.maximum(hi, c + e)
    return (lo + hi) / 2, float(np.linalg.norm(hi - lo) / 2)


def sample_q(model, n: int, seed: int) -> np.ndarray:
    """n joint vectors uniform within the limits (a joint without limits over ±π), seeded."""
    lo = np.asarray(model.lower, float).copy()
    hi = np.asarray(model.upper, float).copy()
    bad = ~np.isfinite(lo) | ~np.isfinite(hi) | (hi <= lo)
    lo[bad], hi[bad] = -math.pi, math.pi
    rng = np.random.default_rng(seed)
    return lo + rng.random((n, len(lo))) * (hi - lo)


def free_samples(model, n: int, seed: int, notes: list, name: str = '') -> tuple[np.ndarray, int, int]:
    """(up to n joint states free of self-collision, how many were dropped, how many drawn): batches of sample_q
    screened with core/robot/collision.self_collision_batch, at most 8 n drawn."""
    try:
        from .robot.collision import self_collision_batch
    except Exception as e:                              # noqa: BLE001
        notes.append(f'{name}: sampled joint states were not screened for self-collision ({e}).')
        return sample_q(model, n, seed), 0, n
    keep, dropped, drawn, k = [], 0, 0, 0
    while len(keep) < n and drawn < 8 * n:
        Q = sample_q(model, n, seed + 7919 * k)
        k += 1
        drawn += len(Q)
        try:
            hit = np.asarray(self_collision_batch(model, Q)[0], bool)
        except Exception as e:                          # noqa: BLE001 — no collision model: keep every sample
            notes.append(f'{name}: sampled joint states were not screened for self-collision ({e}).')
            return Q, 0, len(Q)
        dropped += int(hit.sum())
        keep += list(Q[~hit])
    return np.asarray(keep[:n], float).reshape(-1, model.nq), dropped, drawn


def _points(tri: np.ndarray) -> np.ndarray:
    """(M, 4, 3): each triangle's three vertices and its centre."""
    return np.concatenate([tri, tri.mean(axis=1, keepdims=True)], axis=1)


def clearance(project, sensor, D: Design, T_parent_bracket: np.ndarray, *, samples: int = 48, seed: int = 0,
              q_now: dict | None = None, margin_mm: float = 2.0, cancel=None) -> dict:
    """The bracket and camera body against every robot link but the mount link, over sampled joint states (and the
    home pose and q_now), and against the project's cable paths. See the module docstring."""
    from .robot.model import RobotModelError, get_model
    from .scenegeom import _q_for
    solid = body_solid(D)
    centre, rad = _bound(D)
    links, notes = [], []
    worst = {'min_mm': math.inf}
    mounted_robot = sensor.mount.robot_uid if sensor.mount.on_link else ''
    for rob in project.robots:
        try:
            m = get_model(project, rob.uid)
        except RobotModelError as e:
            notes.append(f'{rob.name} is not checked: {e}')
            continue
        if not m.n_triangles:
            continue
        Q = [m.home]
        try:
            Q.append(_q_for(m, q_now, rob.uid))
        except ValueError:
            pass
        Qs, dropped, drawn = free_samples(m, samples, seed, notes, rob.name)
        if dropped:
            notes.append(f'{rob.name}: {dropped} of {drawn} sampled joint states put the robot into itself and were '
                         f'left out; {len(Qs)} collision-free ones were kept.')
        Q += list(Qs)
        Q = np.asarray(Q, float)
        names = m.link_names
        desc = np.zeros(len(names), bool)
        if rob.uid == mounted_robot and D.kind == 'flange':
            below = descendants(rob, sensor.mount.parent_frame)
            desc = np.array([n in below for n in names], bool)
        best = {}
        for k, q in enumerate(Q):
            if cancel is not None and cancel():
                return {'cancelled': True}
            tri, _n, li = m.posed_triangles(q)
            if rob.uid == mounted_robot:
                T = m.link_pose(q, sensor.mount.parent_frame) @ T_parent_bracket
            else:
                T = (sensor_static_matrix(project, sensor, q_now)) @ T_parent_bracket
            # into the bracket frame, mm
            Rt, tt = T[:3, :3], T[:3, 3]
            P = _points(np.asarray(tri, float))                      # (M, 4, 3) world
            Pb = ((P - tt) @ Rt) * 1000.0
            dc = np.linalg.norm(Pb - centre, axis=2).min(axis=1)
            near = dc < rad + 30.0
            if rob.uid == mounted_robot:
                near &= np.asarray(names, object)[li] != sensor.mount.parent_frame
            if not near.any():
                continue
            idx = np.nonzero(near)[0]
            pts = Pb[idx].copy()
            keep = np.ones(pts.shape[:2], bool)
            if desc.any():
                # the gripper and everything beyond the flange sits on the adapter: out by its thickness; its
                # contact face on the adapter (within 1 mm of it, inside the disc) is the seat, not a collision
                dd = desc[li[idx]]
                pts[dd, :, 2] += D.interface['disc_mm']
                r = np.hypot(pts[..., 0], pts[..., 1])
                keep &= ~(dd[:, None] & (r < D.keepout['r_mm']) &
                          (np.abs(pts[..., 2] - D.interface['disc_mm']) < 1.0))
            pts, keep = pts.reshape(-1, 3), keep.reshape(-1)
            d = np.full(len(pts), np.inf)
            if keep.any():
                d[keep] = solid(pts[keep])
            d = d.reshape(-1, 4).min(axis=1)
            for lk in np.unique(li[idx]):
                sel = li[idx] == lk
                v = float(d[sel].min())
                name = names[int(lk)]
                b = best.get(name)
                if b is None:
                    best[name] = {'robot': rob.name, 'link': name, 'min_mm': v, 'at': k, 'max_mm': v, 'hits': 0}
                    b = best[name]
                elif v < b['min_mm']:
                    b['min_mm'], b['at'] = v, k
                b['max_mm'] = max(b['max_mm'], v)
                b['hits'] += int(v < 0)
        for name, b in best.items():
            if not math.isfinite(b['min_mm']):
                continue
            b['moves'] = bool(b['max_mm'] - b['min_mm'] > 0.05)
            b['pose'] = 'home' if b['at'] == 0 else ('on screen' if b['at'] == 1 and len(Q) > samples + 1 else
                                                    'sampled')
            b['q'] = [float(x) for x in Q[b['at']]]
            b['share_pct'] = 100.0 * b['hits'] / len(Q)
            b['samples'] = int(len(Q))
            del b['at'], b['max_mm']
            links.append(b)
            if b['min_mm'] < worst['min_mm']:
                worst = dict(b)
    links.sort(key=lambda b: b['min_mm'])
    cab = cable_clearance(project, sensor, solid, T_parent_bracket, q_now)
    collide = [b for b in links if b['min_mm'] < 0]
    close = [b for b in links if 0 <= b['min_mm'] < margin_mm]
    if collide:
        b = collide[0]
        fixed = not b['moves']
        status = 'critical' if (fixed or b['pose'] in ('home', 'on screen')) else 'warning'
        text = (f'{b["link"]} runs {-b["min_mm"]:.1f} mm into the bracket or camera ' +
                ('at every pose (it is fixed to the mount)' if fixed else
                 f'in {b["share_pct"]:.0f} % of the sampled joint states'))
    elif cab.get('min_mm') is not None and cab['min_mm'] < 0:
        status, text = 'warning', f'A cable ({cab["cable"]}) passes through the bracket or camera.'
    elif close:
        b = close[0]
        status, text = 'warning', (f'{b["link"]} comes within {b["min_mm"]:.1f} mm (under the {margin_mm:g} mm '
                                   'margin, a default).')
    elif links:
        status, text = 'good', (f'Clear of every link by at least {links[0]["min_mm"]:.1f} mm '
                                f'({links[0]["link"]}) over {links[0]["samples"]} joint states.')
    else:
        status, text = 'good', 'No robot link comes near the bracket.'
    return {'status': status, 'text': text, 'links': links[:12], 'worst': worst if links else None,
            'margin_mm': margin_mm, 'margin_default': True, 'samples': samples, 'seed': seed, 'cables': cab,
            'notes': notes, 'method': 'signed distance of the bracket and camera solid at link mesh vertices and '
                                      'triangle centres (a lower bound of the clearance; the sign is exact)'}


def descendants(robot, link: str) -> set:
    """The links below link in the robot's joint tree (the gripper and fingers below a flange)."""
    kids: dict = {}
    for j in robot.joints:
        kids.setdefault(j.get('parent'), []).append(j.get('child'))
    out, todo = set(), [link]
    while todo:
        for c in kids.get(todo.pop(), []):
            if c not in out:
                out.add(c)
                todo.append(c)
    return out


def gripper_keepout_mm(project, sensor, T_parent_bracket, disc_mm: float, height_mm: float = 20.0) -> float | None:
    """Diameter (mm) of the gripper's footprint on the adapter: the links below the flange, at home, moved out by
    the adapter, their mesh vertices within height_mm above it; None when there is none."""
    from .robot.model import RobotModelError, get_model
    rob = project.robot(sensor.mount.robot_uid)
    if rob is None:
        return None
    try:
        m = get_model(project, rob.uid)
    except RobotModelError:
        return None
    below = descendants(rob, sensor.mount.parent_frame)
    if not below or not m.n_triangles:
        return None
    tri, _n, li = m.posed_triangles(m.home)
    sel = np.array([m.link_names[int(k)] in below for k in li], bool)
    if not sel.any():
        return None
    T = m.link_pose(m.home, sensor.mount.parent_frame) @ T_parent_bracket
    P = ((np.asarray(tri[sel], float).reshape(-1, 3) - T[:3, 3]) @ T[:3, :3]) * 1000.0
    P[:, 2] += disc_mm
    band = (P[:, 2] >= disc_mm - 0.5) & (P[:, 2] <= disc_mm + height_mm)
    if not band.any():
        return None
    return float(2 * np.hypot(P[band, 0], P[band, 1]).max() + 2.0)


def sensor_static_matrix(project, sensor, q=None) -> np.ndarray:
    from .scenegeom import sensor_world_pose
    if sensor.mount.on_link:
        T = sensor_world_pose(project, sensor, q).matrix()
        return T @ np.linalg.inv(sensor.mount.pose.matrix())
    return np.eye(4)


def cable_polylines(project, q=None) -> list:
    """[(name, (K, 3) world points m, radius m, sensor uid or '')]: the project's cable paths at q when it has any:
    each cable laid along its attachments (core/studies/cables.py cable_polylines, the slack loops the Cables page
    draws). Falls back to records that carry world points ('points' or 'path') and tolerates unknown shapes."""
    out = []
    try:
        from .studies import cables as CF
        fn = getattr(CF, 'cable_polylines', None)
        if fn is not None:
            for c in fn(project, q) or []:
                if isinstance(c, dict):
                    pts = c.get('points')
                    pts = np.asarray([] if pts is None else pts, float).reshape(-1, 3)
                    out.append((str(c.get('name', 'cable')), pts, float(c.get('radius_m') or 0.0),
                                str(c.get('sensor_uid') or '')))
                elif isinstance(c, (tuple, list)) and len(c) >= 2:
                    out.append((str(c[0]), np.asarray(c[1], float).reshape(-1, 3),
                                float(c[2]) if len(c) > 2 else 0.0, str(c[3]) if len(c) > 3 else ''))
            if out:
                return out
    except Exception:                                   # noqa: BLE001 — no cable model (yet): none checked
        pass
    for c in getattr(project, 'cables', None) or []:
        d = c if isinstance(c, dict) else getattr(c, '__dict__', {})
        pts = d.get('points') or d.get('path') or []
        try:
            P = np.asarray(pts, float).reshape(-1, 3)
        except (TypeError, ValueError):
            continue
        if len(P) >= 2:
            dia = d.get('diameter_mm') or 0.0
            out.append((str(d.get('name', 'cable')), P, float(dia) / 2000.0, str(d.get('sensor_uid') or '')))
    return out


def cable_clearance(project, sensor, solid, T_parent_bracket, q=None) -> dict:
    """The cable paths' clearance to the bracket and camera (mm), the sensor's own cable excluded within 15 mm of the
    camera (it leaves the camera there by design)."""
    try:
        cables = cable_polylines(project, q)
    except Exception:                                   # noqa: BLE001
        cables = []
    if not cables:
        return {'checked': 0, 'min_mm': None, 'cable': None,
                'note': 'The project has no cable paths: the cable check did not run.'}
    T = sensor_static_matrix(project, sensor, q) @ T_parent_bracket
    best, name = math.inf, None
    for nm, P, r, suid in cables:
        if len(P) < 2:
            continue
        seg = np.concatenate([np.linspace(P[i], P[i + 1], 20) for i in range(len(P) - 1)])
        Pb = ((seg - T[:3, 3]) @ T[:3, :3]) * 1000.0
        d = solid(Pb) - r * 1000.0
        if suid and suid == sensor.uid:
            d = d[d > 15.0 - r * 1000.0] if (d > 15.0).any() else np.array([math.inf])
        v = float(d.min()) if len(d) else math.inf
        if v < best:
            best, name = v, nm
    return {'checked': len(cables), 'min_mm': None if not math.isfinite(best) else best, 'cable': name,
            'note': 'cable centre lines sampled 20 points per segment, minus the cable radius'}


# ====================================================================== image blocked
def image_blocked(sensor, D: Design, stride: int = 64) -> dict | None:
    """The share of each imager's pixels whose ray meets the bracket (sphere tracing against its solid), on a grid of
    about `stride` rays across the image."""
    from . import library as L
    try:
        ims = L.sensor_imagers(sensor)
    except Exception:                                   # noqa: BLE001
        return None
    out = []
    for im in ims:
        cam = im.camera
        d = cam.pixel_rays(max(1, int(cam.width) // max(int(stride), 8))).reshape(-1, 3)
        ok = np.all(np.isfinite(d), axis=1)
        if not ok.any():
            continue
        Tbi = D.T_b_body @ _mm(im.pose_in_body.matrix())
        Db = d[ok] @ Tbi[:3, :3].T
        O = np.repeat(Tbi[:3, 3][None], len(Db), axis=0)
        lo, hi = D.solid.lo, D.solid.hi
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        tmax = float(np.linalg.norm(corners - Tbi[:3, 3], axis=1).max()) + 2.0
        hit, _t = M.raytrace(D.solid, O, Db, tmax, eps=0.02)
        out.append({'imager': im.name, 'blocked_pct': 100.0 * float(hit.mean()), 'rays': int(len(Db))})
    if not out:
        return None
    return {'imagers': out, 'blocked_pct': max(x['blocked_pct'] for x in out)}


def _mm(T):
    T = np.array(T, float)
    T[:3, 3] *= 1000.0
    return T


# ====================================================================== one sensor
def suggest_kind(project, sensor) -> tuple[str, str]:
    """(kind, why) a sensor's bracket would most likely be; kind '' when no type fits (why says so)."""
    cam = camera_geom(sensor)
    if sensor.mount.on_link:
        if cam.micro:
            return 'palm', 'a micro camera on a link of the hand'
        rob = project.robot(sensor.mount.robot_uid)
        fl = robot_flange(rob) if rob is not None else None
        if fl is not None and _is_flange_link(project, sensor):
            return 'flange', f'{rob.name}\'s flange is {ISO9409[fl[0]]["designation"]} ({SOURCES[fl[1]]["url"]})'
        link = sensor.mount.parent_frame
        if re.search(r'head|neck|pan|tilt', link, flags=re.I):
            return 'pantilt', f'a camera on {link}, a pan-tilt link'
        return '', (f'no bracket type fits a camera on {link} without a cited ISO 9409-1 flange: choose one (the '
                    'pan-tilt neck mount is a plate for your own hole pattern)')
    if sensor.mount.parent_frame == 'tripod':
        return 'none', 'a tripod holds the camera by its own thread'
    return 'tslot2020', ('a camera on a static frame (the 2020 profile is a DexVantage default guess, not a survey of '
                         'rigs; change it to your profile)')


def _is_flange_link(project, sensor) -> bool:
    rob = project.robot(sensor.mount.robot_uid)
    if rob is None:
        return False
    link = sensor.mount.parent_frame
    if any(j.get('parent') == link and j.get('type') == 'fixed' for j in rob.joints):
        moving_children = [j for j in rob.joints if j.get('parent') == link and j.get('type') != 'fixed']
        return not moving_children
    return False


def build(project, sensor, spec: BracketSpec, *, mesh_it: bool = True, clear: bool = True, samples: int = 48,
          seed: int = 0, q_now: dict | None = None, margin_mm: float = 2.0, progress=None, cancel=None) -> dict:
    """One sensor's bracket: the design, its mesh and checks, the clearance check: a JSON-able dict (V and F are
    returned under '_mesh' as arrays, which callers drop before JSON)."""
    out = {'uid': sensor.uid, 'name': sensor.name, 'kind': spec.kind, 'label': TYPES.get(spec.kind, spec.kind),
           'spec': spec.to_dict(), 'flags': [], 'notes': [], 'status': 'skipped', 'message': ''}
    if spec.kind == 'none':
        out['message'] = 'No printed bracket for this sensor.'
        return out
    cam = camera_geom(sensor, spec)
    out['camera'] = {'model_id': cam.model_id, 'title': cam.title, 'size_mm': list(cam.size_mm) if cam.size_mm else
                     None, 'size_note': cam.size_note, 'holes_note': cam.holes_note, 'holes': cam.holes,
                     'micro': cam.micro,
                     'box_assumption': 'ASSUMPTION: the camera is a box of its stated size with the reference '
                                       'imager\'s optical centre at the middle of its front face.'}
    try:
        pl = placement(project, sensor, spec, cam)
        if spec.kind == 'flange' and spec.keepout_mm is None:
            k = gripper_keepout_mm(project, sensor, pl['T_parent_bracket'], spec.disc_mm)
            if k is not None:
                spec = copy.deepcopy(spec)
                spec.keepout_mm = k
                out['notes'].append(f'The arm keeps a Ø{k:.0f} mm cylinder above the adapter free for the gripper '
                                    '(its footprint in the robot model, 20 mm up from the adapter, plus 1 mm).')
        D = design(spec, cam, pl['T_b_body'])
    except BracketError as e:
        out['status'], out['message'] = 'critical', str(e)
        return out
    out['placement'] = {'parent_frame': sensor.mount.parent_frame, 'robot_uid': sensor.mount.robot_uid,
                        'T_parent_bracket': [[float(x) for x in r] for r in pl['T_parent_bracket']],
                        'note': pl['note'], 'frame': 'bracket frame: mm, origin at the centre of the contact face, z '
                                                     'out of the mount surface (the STL prints z up)'}
    out['design'] = {'interface': clean(D.interface), 'plate': clean(D.plate), 'holes': clean(D.holes),
                     'res_mm': D.res_mm, 'T_bracket_body_mm': clean(D.T_b_body)}
    if D.pattern is not None:
        out['pattern'] = clean({k: v for k, v in D.pattern.items() if k != 'sources'})
        out['pattern']['sources'] = [source(k) for k in D.pattern['sources']]
    if spec.kind in TSLOT:
        out['profile'] = dict(TSLOT[spec.kind], source=source(TSLOT[spec.kind]['source']))
    out['flags'] += D.flags
    out['notes'] += D.notes
    out['defaults'] = {k: getattr(spec, k) for k in ('wall_mm', 'margin_mm', 'clearance_mm', 'fit_mm', 'fillet_mm',
                                                       'res_mm')}
    if mesh_it:
        V, F, chk = mesh_design(D, progress=progress, cancel=cancel)
        if V is None:
            out['status'], out['message'] = 'skipped', 'Cancelled.'
            return out
        out['mesh'] = clean(chk)
        out['_mesh'] = (V, F)
        lo, hi = V.min(axis=0), V.max(axis=0)
        out['dims_mm'] = [float(v) for v in hi - lo]
        po = print_orientation(V, F)
        out['print'] = po
        if po:
            out['notes'].append(f'Print {po[0]["orientation"]}: {po[0]["overhang_mm2"] / 100:.1f} cm² of overhang '
                                'needs support.')
        out['notes'].append(f'{MATERIAL["name"]} solid: about {chk["mass_g"]:.0f} g ({MATERIAL["note"]})')
        if not chk['ok']:
            out['flags'].append('The mesh is not a closed, consistently wound solid: do not print it; report the '
                                'sensor and its settings.')
    ib = image_blocked(sensor, D)
    if ib is not None:
        out['image_blocked'] = ib
        if ib['blocked_pct'] > 0.05:
            out['flags'].append(f'The bracket shows in {ib["blocked_pct"]:.1f} % of the camera\'s image: move it back '
                                'from the lens or choose another face.')
    if clear:
        cl = clearance(project, sensor, D, pl['T_parent_bracket'], samples=samples, seed=seed, q_now=q_now,
                       margin_mm=margin_mm, cancel=cancel)
        out['clearance'] = clean(cl)
    sev = 'good'
    if out.get('mesh') and not out['mesh']['ok']:
        sev = 'critical'
    cs = (out.get('clearance') or {}).get('status')
    if cs == 'critical':
        sev = 'critical'
    elif (cs == 'warning' or out['flags']) and sev == 'good':
        sev = 'warning'
    out['status'] = sev
    if sev == 'critical' and cs == 'critical':
        out['message'] = out['clearance']['text']
    elif out['flags']:
        out['message'] = out['flags'][0]
    elif cs:
        out['message'] = out['clearance']['text']
    else:
        out['message'] = 'Generated: every check passes.'
    out['_design'] = D
    return out


# ====================================================================== output
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return round(v, 6) if math.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def public(row: dict) -> dict:
    """A build() row without its arrays and objects (JSON-able)."""
    return {k: v for k, v in row.items() if not k.startswith('_')}


def sidecar(row: dict, stl_name: str, data: bytes) -> dict:
    """The JSON written beside an STL: what it is, its dimensions, holes, checks and print notes."""
    import hashlib
    return clean({'schema': SCHEMA, 'stl': stl_name, 'stl_sha256': hashlib.sha256(data).hexdigest(),
                   'units': 'millimetres', 'frame': (row.get('placement') or {}).get('frame'),
                   **{k: row.get(k) for k in ('name', 'uid', 'kind', 'label', 'status', 'message', 'dims_mm', 'mesh',
                                              'design', 'pattern', 'profile', 'camera', 'placement', 'flags', 'notes',
                                              'defaults', 'clearance', 'image_blocked', 'spec')}})


def export_stl(row: dict, project, path: str) -> list[str]:
    """Write the row's STL at path and its provenance-stamped sidecar at path + '.json'; returns both paths."""
    from . import provenance as PV
    if row.get('_mesh') is None:
        raise BracketError(f'{row.get("name", "This sensor")} has no generated bracket to export.')
    V, F = row['_mesh']
    data = stl_bytes(V, F, f'{row.get("name", "")} {row.get("kind", "")}')
    with open(path, 'wb') as fh:
        fh.write(data)
    doc = sidecar(row, os.path.basename(path), data)
    doc = PV.stamp(doc, project, output={'option': None, 'name': os.path.basename(path) + '.json',
                                         'in_folder': False})
    side = path + '.json'
    with open(side, 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
        fh.write('\n')
    return [path, side]


def spec_for(project, sensor) -> BracketSpec:
    """The sensor's bracket spec: kind from Mount.bracket, options from the study settings."""
    st = (project.analysis.studies.get('brackets') or {}).get('sensors') or {}
    d = copy.deepcopy(st.get(sensor.uid) or {})
    kind = str(sensor.mount.bracket or '').strip()
    d['kind'] = kind if kind in TYPES else 'none'
    if d['kind'] == 'flange' and not d.get('pattern'):
        rob = project.robot(sensor.mount.robot_uid)
        fl = robot_flange(rob) if rob is not None else None
        if fl is not None:
            d['pattern'] = fl[0]
    return BracketSpec.from_dict(d)
