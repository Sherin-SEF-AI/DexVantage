"""Grasp sets (spec §7.3): sampled over a task volume, imported from a file, taken from a trajectory, and the contact
patches of the posed fingertips. Grasp sets are written and read only through this module.

A grasp set lives in project.tasks.grasp_sets as a plain record:

    {'uid', 'name', 'source': 'sampled' | 'imported' | 'trajectory' | 'dataset', 'robot_uid',
     'frame': the robot link the poses are for (default Robot.gripper_frame),
     'poses': [{'id', 'pose': {'position': [x, y, z], 'quaternion': [x, y, z, w]}, 'approach': [x, y, z],
                'kind': 'top_down' | 'side' | 'angled' | 'handoff' | 'imported' | 'trajectory',
                'object_uid': str or '', 'width_m': float or None, 't': float or None}],
     'settings': {what made it: the sampler's settings, the file, the trajectory; the gripper geometry used}}

`pose` is the gripper frame in the world at contact; `approach` is a unit vector in the world, the direction the
gripper travels into the grasp. Positions in metres, quaternions (x, y, z, w) (core/frames.py).

    sample(project, robot_uid, kinds, n, rng_seed, volume_uid=None, handover_uid=None, **options) -> grasp set
    import_file(path, project=None, robot_uid='') -> grasp set   (CSV or JSON; GraspError names the row and column)
    from_trajectory(project, traj_uid, every=1, mode='every' | 'closures') -> grasp set
    pad_opening(project, robot_uid, Q) -> (N,), closure_events(opening) -> [(onset, end, drop)]   grasps in a trajectory
    pre_contact_poses(grasp, distances_m) -> [4 × 4]             the gripper backed off along −approach
    contact_points(project, robot_uid, q) -> {fingertip: (K, 3)} world points on the fingertip contact pads
    contact_patches(project, robot_uid, q) -> {fingertip: {'points', 'normals', 'link', 'rule'}}
    object_points(project, object_uid, n, rng_seed=0) -> (n, 3)  points on a task object's surface (area-weighted)
    gripper_geometry(project, robot_uid) -> the tool centre, approach and closing axes in the gripper frame
    ik_targets(project, gs) -> (link, tcp 4 × 4, targets)      for ik.solve_batch(model, link, targets, tcp=tcp)
    sets(project), get_set(project, uid), add_set(project, gs), remove_set(project, uid), counts(gs)

The gripper geometry (gripper_geometry). Every sampler places the grasp centre, not the gripper frame, on its target,
so the module needs to know where the grasp centre is in the gripper frame and which way the gripper approaches and
closes. They are measured from the contact pads at the robot's home pose (nothing typed in):
- the tool centre point (TCP) is the mean of the pads' centroids (each the area-weighted centroid of its contact faces);
- the closing axis is the direction the two pads face each other along (the difference of their mean inward
  normals; the line between their centroids when the normals do not oppose within 60°); with more than two pads (a
  dexterous hand) the principal axis of their centroids, the direction of largest spread;
- the approach axis is the direction from the gripper frame's origin to the TCP, with its closing component taken
  out. When the TCP lies within 1 mm of that line (a frame placed at the pinch point), the frame's z axis is used.
A robot with no fingertips and no pads gets the gripper frame's origin as TCP, its z axis as the approach and its x
axis as the closing axis; the set's settings say "assumed" and the page shows it. Each can be overridden with the
sampler's options tcp, approach_axis and closing_axis (gripper-frame coordinates).

The contact pads (contact_patches). The rule, in order:
1. Robot.contact_pads, when the robot has any: each pad is a box {'center', 'size'} in its link's frame (the import
   wizard's pad tagger), or 'faces', a list of triangles [[x, y, z] × 3] in the link frame. Of a box, the face whose
   outward normal points most nearly at the other pads' centroid (on other links) is the contact face; a pad with no
   pad on another link to face uses its whole surface. Points: a regular PAD_GRID × PAD_GRID (12 × 12) grid of cell
   centres.
2. Otherwise the robot's fingertip_frames: each link's collision geoms (its visual meshes when it has none). Of
   their triangles, those (a) within PAD_LENGTH_M (default 40 mm) of the finger's distal end along the approach
   axis, (b) whose outward normal is within PAD_FACING_DEG (default 45°) of the direction to the other fingertips'
   centroid, and (c) not covered along that direction by the same finger's other triangles (a ray cast,
   core/raycast.py) are its inner faces. When the rule keeps nothing it drops (c), then (b) and (a), and says so.
   PAD_POINTS (default 128) points are drawn on them, area-weighted, from a generator seeded per fingertip (the same
   points every time).
The patch is chosen at the home pose, held in its link's frame and carried with the link at any q. On the UR5e
sample this keeps the inner face of each Robotiq 2F-85 pad box (22 × 37.5 mm, two collision boxes per pad); on the
SO-100 the inner faces of the fixed and the moving jaw's distal boxes.

The samplers (sample). n grasps go round-robin over the kinds asked for; one numpy generator seeded with rng_seed
draws everything, so a seed gives the same set. The grasp centre is placed, then the gripper frame is posed so its
TCP lands there with its approach and closing axes as below; world z is up (REP-103).
- Target points. With task objects whose bounding-box centre is inside the task volume, each grasp picks one object
  (uniformly); its world bounding box gives the centre c and the extents e.
  Without objects: x, y uniform over the volume's footprint, the support height the highest table surface
  (workspace.surfaces) under that point inside the volume's height range, else the volume's floor, and the TCP
  height_m (default 30 mm) above it, capped at the volume's top.
- top_down (top-down pinch): approach straight down (0, 0, −1). On an object the TCP is depth_m below its top (default
  half the pads' length along the approach, at most half the object's height) above its centre, the fingers close
  across the narrower horizontal side of its box, turned by up to ±yaw_jitter_deg (default 15°); without an object
  the closing direction's heading is uniform over 180° (a two-finger grasp is symmetric).
- side: approach horizontal, heading away from the robot's base (the line from the base to the target) turned by up
  to ±spread_deg (default 60°); fingers close horizontally. On an object the TCP is its centre.
- angled: as side, but the approach tilted down: approach = (cos φ sin θ, sin φ sin θ, −cos θ) with θ =
  angled_tilt_deg from the vertical (default 45°) and φ the heading as for side; fingers close horizontally, across
  the approach.
- handoff (bimanual handoff): targets uniform in the handover volume (volume kind 'handover', or handover_uid), in
  mid-air (no support surface); approach horizontal towards the partner robot's base (the other robot nearest the
  volume; without one, away from this robot's base through the volume's centre), turned by up to
  ±handoff_spread_deg (default 20°); fingers close horizontally. The partner's uid is in the settings.
Every option is a labelled default, not a standard; the set's settings record the values used and which are
defaults. width_m is the object's bounding-box extent along the closing axis (an upper bound of its width there), and
None for grasps on volume points.

Imported files (import_file). CSV: a header row naming at least id, x, y, z, qx, qy, qz, qw (metres, the quaternion
(x, y, z, w) of the gripper frame in the world), optionally ax, ay, az (approach), kind, object (or object_uid),
width_m and t; columns in any order, case ignored, separated by commas, semicolons or tabs (a semicolon or tab file
may write decimal commas); blank lines and lines starting with # skipped. JSON: a list of objects with id and either
'pose': {'position', 'quaternion'} or 'position' and 'quaternion' or the CSV's keys, the same optional keys; or a
grasp set record. A row without an approach gets the gripper's approach axis turned by its orientation (the frame's
z axis when no robot is given). Each refused row is named with its line and column and what to write instead.

From a trajectory (from_trajectory). mode 'every': the gripper frame at every Nth sample. mode 'closures' (the
grasps in the trajectory, the Tasks page's default): the gripper's opening at each sample (pad_opening: twice the
mean distance of the pads' centroids, carried with their links, from their common centroid; for two pads the
distance between them) is split into closing strokes (closure_events: a run that starts with a fall of more than
FLAT_MM, 0.1 mm, between two samples, may pause, ends at its last such fall before the opening rises again by more
than FLAT_MM) and a stroke counts when the opening falls by CLOSE_MM (5 mm) or more. The grasp is the gripper frame at
the stroke's onset (the arm in place, the fingers still open, as the grasp study poses them; the fingers stay at their
home opening there) and its approach the direction the TCP travelled over its last APPROACH_MM (20 mm) before the
onset, or the gripper's approach axis when it did not travel that far. All three are labelled defaults; the set's
settings list the strokes (onset and end times, how far the opening fell).

Known limits: the samplers do not check reachability or collisions (the grasp study's IK does, and says why a grasp
is infeasible); object grasps use the object's axis-aligned world bounding box, not its shape, so a grasp on a
rotated or concave object can miss it; the closing axis is fixed by the kind (a side grasp never closes vertically);
the TCP and axes are measured at the home pose, so a gripper whose home is closed and asymmetric gets a biased TCP;
'dataset' sets arrive with the dataset readers (Phase 8); an imported row's approach is not checked against its
orientation; grasps found in a trajectory are its closing strokes, so a trajectory that closes the gripper without
grasping (a test close, a regrasp) gives a grasp there too, and fingers that close by less than FLAT_MM between two
samples (slow fingers, dense samples) are not seen to close.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import threading
from collections import OrderedDict

import numpy as np

from .frames import Pose, quat_normalize, quat_to_matrix
from .project import new_uid

SOURCES = ('sampled', 'imported', 'trajectory', 'dataset')
KINDS = ('top_down', 'side', 'angled', 'handoff')                 # what sample() draws
POSE_KINDS = KINDS + ('imported', 'trajectory')
KIND_TITLES = {'top_down': 'Top-down pinch', 'side': 'Side grasp', 'angled': 'Angled', 'handoff': 'Bimanual handoff',
               'imported': 'Imported', 'trajectory': 'From a trajectory'}
MAX_GRASPS = 100_000

# sampler options: DexVantage defaults (labelled as such in the settings and the page), not standards
DEFAULTS = {
    'height_m': 0.03,             # TCP above the support surface when there are no objects
    'depth_m': None,              # TCP below an object's top for top-down grasps; None = half the pads' length
    'angled_tilt_deg': 45.0,      # angled approach, from the vertical
    'spread_deg': 60.0,           # side and angled heading spread about the line from the robot's base
    'handoff_spread_deg': 20.0,   # handoff heading spread about the line to the partner
    'yaw_jitter_deg': 15.0,       # object top-down grasps: turn of the closing axis
}
OPTION_RANGES = {'height_m': (0.0, 2.0), 'depth_m': (0.0, 1.0), 'angled_tilt_deg': (0.0, 90.0),
                 'spread_deg': (0.0, 180.0), 'handoff_spread_deg': (0.0, 180.0), 'yaw_jitter_deg': (0.0, 90.0)}

# contact pads (defaults, documented in the module docstring and docs/models/grasps.md)
PAD_LENGTH_M = 0.04
PAD_FACING_DEG = 45.0
PAD_POINTS = 128
PAD_GRID = 12
AXIS_DEGENERATE_M = 1e-3

CSV_REQUIRED = ('id', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw')
CSV_ALIASES = {'object_uid': 'object', 'width': 'width_m', 'time': 't', 'name': 'id'}
CSV_OPTIONAL = ('ax', 'ay', 'az', 'kind', 'object', 'width_m', 't')
MM_HINT_M = 5.0               # a position farther than this from the origin looks like millimetres

# grasps found in a trajectory (from_trajectory mode 'closures'): defaults, not standards
TRAJECTORY_MODES = ('every', 'closures')
CLOSE_MM = 5.0                # a closing stroke counts when the opening falls by this much
FLAT_MM = 0.1                 # a change of opening below this between two samples is a pause, not a move
APPROACH_MM = 20.0            # the approach is the tool centre's travel over this distance before the onset


class GraspError(ValueError):
    """Inputs a grasp-set function cannot work with; the message says why and what to do, in plain words."""


# ====================================================================== the records
def sets(project) -> list:
    """The project's grasp sets (the records themselves: change them only through add_set / remove_set)."""
    return [g for g in project.tasks.grasp_sets if isinstance(g, dict)]


def get_set(project, uid: str) -> dict | None:
    return next((g for g in sets(project) if g.get('uid') == uid), None)


def add_set(project, gs: dict, trusted: bool = False) -> str:
    """Add a grasp set (normalised, given a uid when it has none, its name made unique); returns its uid. trusted:
    gs is normalise_set's own output (sample, import_file, from_trajectory return one), so it is added as it is,
    without normalising its poses again (0.15 s for 10 000 grasps on the UI thread)."""
    gs = gs if trusted else normalise_set(gs)
    if any(g.get('uid') == gs['uid'] for g in sets(project)):
        gs['uid'] = new_uid()
    names = {g.get('name') for g in sets(project)}
    base, k = gs['name'], 2
    while gs['name'] in names:
        gs['name'] = f'{base} ({k})'
        k += 1
    project.tasks.grasp_sets.append(gs)
    return gs['uid']


def remove_set(project, uid: str) -> bool:
    before = len(project.tasks.grasp_sets)
    project.tasks.grasp_sets = [g for g in project.tasks.grasp_sets if not (isinstance(g, dict) and
                                                                           g.get('uid') == uid)]
    return len(project.tasks.grasp_sets) != before


def counts(gs: dict) -> dict:
    """kind -> number of poses."""
    out: dict = {}
    for p in gs.get('poses') or []:
        k = p.get('kind', 'imported')
        out[k] = out.get(k, 0) + 1
    return out


def _f(v) -> float | None:
    return None if v is None else float(v)


def _pose_rec(pid, T, approach, kind, object_uid='', width_m=None, t=None) -> dict:
    P = Pose.from_matrix(T)
    return {'id': str(pid), 'pose': P.to_dict(), 'approach': [float(v) for v in approach], 'kind': kind,
            'object_uid': str(object_uid or ''), 'width_m': _f(width_m), 't': _f(t)}


def normalise_set(gs: dict) -> dict:
    """A clean, JSON-able copy of a grasp set record; raises GraspError when it is not one."""
    if not isinstance(gs, dict):
        raise GraspError('A grasp set is a record with a name and poses.')
    src = str(gs.get('source') or 'imported')
    if src not in SOURCES:
        raise GraspError(f'A grasp set\'s source is one of {", ".join(SOURCES)}, not "{src}".')
    poses = []
    for i, p in enumerate(gs.get('poses') or []):
        if not isinstance(p, dict) or not isinstance(p.get('pose'), dict):
            raise GraspError(f'Pose {i + 1} of grasp set "{gs.get("name", "")}" has no gripper pose.')
        P = Pose.from_dict(p['pose'])
        a = np.asarray(p.get('approach') or quat_to_matrix(P.quaternion)[:, 2], float).reshape(3)
        na = float(np.linalg.norm(a))
        if not math.isfinite(na) or na < 1e-12 or not np.isfinite(P.position).all():
            raise GraspError(f'Pose {i + 1} of grasp set "{gs.get("name", "")}" has a position or approach that is '
                             'not a finite vector.')
        kind = str(p.get('kind') or 'imported')
        poses.append({'id': str(p.get('id', i + 1)), 'pose': P.to_dict(), 'approach': [float(v) for v in a / na],
                      'kind': kind if kind in POSE_KINDS else 'imported', 'object_uid': str(p.get('object_uid') or ''),
                      'width_m': _f(p.get('width_m')), 't': _f(p.get('t'))})
    return {'uid': str(gs.get('uid') or new_uid()), 'name': str(gs.get('name') or 'Grasp set'), 'source': src,
            'robot_uid': str(gs.get('robot_uid') or ''), 'frame': str(gs.get('frame') or ''), 'poses': poses,
            'settings': json.loads(json.dumps(gs.get('settings') or {}, default=_json_default))}


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return str(o)


def grasp_matrix(grasp: dict) -> np.ndarray:
    """The 4 × 4 world transform of a grasp's gripper frame at contact."""
    return Pose.from_dict(grasp.get('pose')).matrix()


def pre_contact_poses(grasp: dict, distances_m) -> list:
    """The gripper frame backed off along −approach by each distance (m): [4 × 4 world transforms], orientation as at
    contact (a straight-line approach)."""
    T = grasp_matrix(grasp)
    a = np.asarray(grasp.get('approach') or T[:3, 2], float)
    a = a / max(float(np.linalg.norm(a)), 1e-300)
    out = []
    for d in np.atleast_1d(np.asarray(distances_m, float)):
        Tk = T.copy()
        Tk[:3, 3] = T[:3, 3] - float(d) * a
        out.append(Tk)
    return out


# ====================================================================== the robot's gripper and pads
def _robot_and_model(project, robot_uid):
    from .robot.model import RobotModelError, get_model
    r = project.robot(robot_uid)
    if r is None:
        raise GraspError('There is no such robot in the project: choose one of its robots.')
    try:
        m = get_model(project, robot_uid)
    except RobotModelError as e:
        raise GraspError(str(e)) from None
    return r, m


def gripper_frame(robot, model) -> str:
    """The link grasps are posed for: Robot.gripper_frame, else the model's last link."""
    if robot.gripper_frame and robot.gripper_frame in model.link_names:
        return robot.gripper_frame
    if robot.gripper_frame:
        raise GraspError(f'{robot.name}\'s gripper frame {robot.gripper_frame} is not a link of its model: set the '
                         'gripper frame again on the Robot page.')
    return model.link_names[-1]


_PAD_CACHE: OrderedDict = OrderedDict()
_PAD_LOCK = threading.Lock()


def _box_faces(center, size):
    """[(outward normal (3,), face centre (3,), u half-axis (3,), v half-axis (3,))] of a box in its own frame."""
    c, h = np.asarray(center, float), np.abs(np.asarray(size, float)) / 2
    out = []
    for ax in range(3):
        u, v = [k for k in range(3) if k != ax]
        for s in (-1.0, 1.0):
            n = np.zeros(3)
            n[ax] = s
            fu, fv = np.zeros(3), np.zeros(3)
            fu[u], fv[v] = h[u], h[v]
            out.append((n, c + n * h[ax], fu, fv))
    return out


def _grid_on_face(fc, fu, fv, g=PAD_GRID):
    s = (np.arange(g) + 0.5) / g * 2 - 1
    a, b = np.meshgrid(s, s, indexing='ij')
    return fc + a.reshape(-1, 1) * fu + b.reshape(-1, 1) * fv


def _tri_area_normal(tri):
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    a = np.linalg.norm(n, axis=1)
    return a / 2, n / np.maximum(a, 1e-300)[:, None]


def sample_triangles(tri: np.ndarray, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """n points drawn uniformly by area on triangles (M, 3, 3), with each point's triangle index."""
    area, _n = _tri_area_normal(np.asarray(tri, float))
    tot = float(area.sum())
    if not len(tri) or tot <= 0:
        raise GraspError('The surface has no area to put points on.')
    k = rng.choice(len(tri), size=int(n), p=area / tot)
    r1, r2 = rng.random(int(n)), rng.random(int(n))
    s = np.sqrt(r1)
    t = tri[k]
    P = (1 - s)[:, None] * t[:, 0] + (s * (1 - r2))[:, None] * t[:, 1] + (s * r2)[:, None] * t[:, 2]
    return P, k


def _pads_from_boxes(robot, model, T) -> list:
    pads = []
    for i, pad in enumerate(robot.contact_pads):
        if not isinstance(pad, dict):
            continue
        link = str(pad.get('link') or '')
        if link not in model.link_names:
            continue
        name = str(pad.get('name') or f'{link} pad {i + 1}')
        if isinstance(pad.get('box'), dict):
            pads.append({'name': name, 'link': link, 'box': pad['box']})
        elif pad.get('faces'):
            tri = np.asarray(pad['faces'], float).reshape(-1, 3, 3)
            pads.append({'name': name, 'link': link, 'tri': tri})
    out = []
    centres = {}
    for p in pads:
        Tl = T[p['link']]
        c = np.asarray(p['box']['center'], float) if 'box' in p else p['tri'].reshape(-1, 3).mean(axis=0)
        centres[p['name']] = Tl[:3, :3] @ c + Tl[:3, 3]
    for i, p in enumerate(pads):
        Tl = T[p['link']]
        others = [centres[q['name']] for q in pads if q['link'] != p['link']]
        d = None
        if others:
            d = np.mean(others, axis=0) - centres[p['name']]
            d = d / max(np.linalg.norm(d), 1e-300)
        if 'box' in p:
            faces = _box_faces(p['box'].get('center', (0, 0, 0)), p['box'].get('size', (0, 0, 0)))
            if d is not None:
                k = int(np.argmax([float((Tl[:3, :3] @ f[0]) @ d) for f in faces]))
                faces = [faces[k]]
                rule = 'contact pad box: the face towards the other pads'
            else:
                rule = 'contact pad box: its whole surface (no pad on another link to face)'
            P = np.concatenate([_grid_on_face(fc, fu, fv) for _n, fc, fu, fv in faces])
            N = np.concatenate([np.tile(nn, (PAD_GRID * PAD_GRID, 1)) for nn, _c, _u, _v in faces])
            centroid = np.mean([fc for _n, fc, _u, _v in faces], axis=0)
            normal = np.mean([nn for nn, _c, _u, _v in faces], axis=0)
        else:
            rng = np.random.default_rng(1000 + i)
            P, k = sample_triangles(p['tri'], PAD_POINTS, rng)
            a, Nt = _tri_area_normal(p['tri'])
            N = Nt[k]
            centroid = (p['tri'].mean(axis=1) * a[:, None]).sum(axis=0) / max(float(a.sum()), 1e-300)
            normal = (Nt * a[:, None]).sum(axis=0)
            rule = 'contact pad faces as tagged'
        out.append({'name': p['name'], 'link': p['link'], 'points': P, 'normals': N, 'centroid': centroid,
                    'normal': normal / max(float(np.linalg.norm(normal)), 1e-300), 'rule': rule})
    return out


def _pads_from_fingertips(robot, model, T, frame) -> list:  # noqa: C901 — one rule, its fallbacks spelled out
    from . import raycast
    links = [ln for ln in robot.fingertip_frames if ln in model.link_names]
    if not links:
        return []
    geoms = {ln: [] for ln in links}
    for ln, V, F, _c in model.collision_geoms():
        if ln in geoms and len(F):
            geoms[ln].append((V.astype(float), F))
    used_visual = []
    for ln in links:
        if not geoms[ln]:
            geoms[ln] = [(V.astype(float), F) for l2, V, F, _c in model.visual_meshes() if l2 == ln and len(F)]
            if geoms[ln]:
                used_visual.append(ln)
    world, normals, cents = {}, {}, {}
    for ln in links:
        if not geoms[ln]:
            continue
        Tl = T[ln]
        tris, ns = [], []
        for V, F in geoms[ln]:
            tri = V[F]
            _a, n = _tri_area_normal(tri)
            # outward: away from the geom's own centroid (exact for the convex pads and jaw boxes)
            flip = np.einsum('ij,ij->i', n, tri.mean(axis=1) - V.mean(axis=0)) < 0
            n[flip] *= -1
            tris.append(tri @ Tl[:3, :3].T + Tl[:3, 3])
            ns.append(n @ Tl[:3, :3].T)
        world[ln] = np.concatenate(tris)
        normals[ln] = np.concatenate(ns)
        a, _n = _tri_area_normal(world[ln])
        cents[ln] = (world[ln].mean(axis=1) * a[:, None]).sum(axis=0) / max(float(a.sum()), 1e-300)
    if not world:
        return []
    g = T[frame][:3, 3]
    ax = np.mean(list(cents.values()), axis=0) - g
    if np.linalg.norm(ax) < AXIS_DEGENERATE_M:
        ax = T[frame][:3, 2]
    ax = ax / np.linalg.norm(ax)
    cos_face = math.cos(math.radians(PAD_FACING_DEG))
    out = []
    for i, ln in enumerate(links):
        if ln not in world:
            continue
        tri, n = world[ln], normals[ln]
        tc = tri.mean(axis=1)
        others = [cents[o] for o in world if o != ln]
        d = None
        if others:
            d = np.mean(others, axis=0) - cents[ln]
            d = d - ax * float(d @ ax)
            d = d / max(np.linalg.norm(d), 1e-300)
        s = tri.reshape(-1, 3) @ ax
        distal = (tc @ ax) >= float(s.max()) - PAD_LENGTH_M
        facing = (n @ d >= cos_face) if d is not None else np.ones(len(tri), bool)
        keep = distal & facing
        steps = []
        if d is not None and keep.any():
            idx = np.flatnonzero(keep)
            hit = raycast.first_hit(tc[idx] + d * 1e-5, np.broadcast_to(d, (len(idx), 3)), tri)
            exposed = np.zeros(len(tri), bool)
            exposed[idx] = ~np.isfinite(hit)
            if (keep & exposed).any():
                keep = keep & exposed
            else:
                steps.append('none of them is uncovered towards the other fingertips, so covered faces are kept')
        if not keep.any():
            keep = facing
            steps.append(f'no face within {PAD_LENGTH_M * 1000:.0f} mm of the tip faces the other fingertips, so the '
                         'distal band is dropped')
        if not keep.any():
            keep = distal
            steps.append('no face faces the other fingertips, so the distal band\'s faces are used')
        if not keep.any():
            keep = np.ones(len(tri), bool)
        rng = np.random.default_rng(2000 + i)
        P, k = sample_triangles(tri[keep], PAD_POINTS, rng)
        N = n[keep][k]
        Ti = np.linalg.inv(T[ln])
        a, _n = _tri_area_normal(tri[keep])
        centroid = (tri[keep].mean(axis=1) * a[:, None]).sum(axis=0) / max(float(a.sum()), 1e-300)
        normal = (n[keep] * a[:, None]).sum(axis=0)
        normal = normal / max(float(np.linalg.norm(normal)), 1e-300)
        rule = (f'inner faces of {ln}\'s {"visual meshes" if ln in used_visual else "collision geoms"}: within '
                f'{PAD_LENGTH_M * 1000:.0f} mm of the tip, facing the other fingertips (within {PAD_FACING_DEG:g}°)'
                + (', uncovered' if d is not None else ' (no other fingertip: not filtered by direction)'))
        if steps:
            rule += '; ' + '; '.join(steps)
        out.append({'name': ln, 'link': ln, 'points': P @ Ti[:3, :3].T + Ti[:3, 3], 'normals': N @ Ti[:3, :3].T,
                    'centroid': Ti[:3, :3] @ centroid + Ti[:3, 3], 'normal': Ti[:3, :3] @ normal, 'rule': rule})
    return out


def _pads_local(project, robot_uid) -> tuple[list, str, object]:
    """The contact pads in their links' frames, chosen at the home pose (cached per compiled model)."""
    r, m = _robot_and_model(project, robot_uid)
    key = (id(m), json.dumps(r.contact_pads, sort_keys=True, default=str), tuple(r.fingertip_frames),
           r.gripper_frame)
    with _PAD_LOCK:
        hit = _PAD_CACHE.get(key)
        if hit is not None and hit[0] is m:
            return hit[1], hit[2], m
    frame = gripper_frame(r, m)
    T = m.fk(m.home)
    pads = _pads_from_boxes(r, m, T) if r.contact_pads else []
    if not pads:
        pads = _pads_from_fingertips(r, m, T, frame)
    with _PAD_LOCK:
        _PAD_CACHE[key] = (m, pads, frame)
        while len(_PAD_CACHE) > 12:                    # as robot.model.MAX_MODELS: it holds the models
            _PAD_CACHE.popitem(last=False)
    return pads, frame, m


def _q_of(model, robot_uid, q):
    if q is None:
        return model.home
    if isinstance(q, dict):
        q = q.get(robot_uid)
        return model.home if q is None else model.clip(q)
    return model.clip(q)


def contact_patches(project, robot_uid: str, q=None) -> dict:
    """fingertip (or pad) name -> {'points' (K, 3) world, 'normals' (K, 3) world outward unit vectors, 'link',
    'rule'} at joint values q (the model's joint order, or {robot uid: q}; None = home). {} when the robot has no
    fingertips or pads tagged."""
    pads, _frame, m = _pads_local(project, robot_uid)
    if not pads:
        return {}
    T = m.fk(_q_of(m, robot_uid, q))
    out = {}
    for p in pads:
        Tl = T[p['link']]
        out[p['name']] = {'points': p['points'] @ Tl[:3, :3].T + Tl[:3, 3], 'normals': p['normals'] @ Tl[:3, :3].T,
                          'link': p['link'], 'rule': p['rule']}
    return out


def contact_points(project, robot_uid: str, q=None) -> dict:
    """fingertip (or pad) name -> (K, 3) world points on its contact pad at q (see contact_patches)."""
    return {k: v['points'] for k, v in contact_patches(project, robot_uid, q).items()}


def _unit(v):
    v = np.asarray(v, float).reshape(3)
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n < 1e-12:
        raise GraspError('A direction must be a vector of non-zero length.')
    return v / n


def gripper_geometry(project, robot_uid: str, tcp=None, approach_axis=None, closing_axis=None) -> dict:
    """{'frame', 'tcp' (3,), 'approach' (3,), 'closing' (3,) in the gripper frame, 'width_m' between the pads at
    home (or None), 'pad_length_m' (the pads' extent along the approach, or None), 'assumed': bool, 'rule': text}.
    See the module docstring for the rule; tcp / approach_axis / closing_axis override it."""
    pads, frame, m = _pads_local(project, robot_uid)
    T = m.fk(m.home)
    Ti = np.linalg.inv(T[frame])
    assumed = not pads
    width = pad_len = None
    if pads:
        def in_frame(X, link):
            M = Ti @ T[link]
            return np.asarray(X, float).reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
        C = np.array([in_frame(p['centroid'], p['link'])[0] for p in pads])
        Nn = np.array([(Ti @ T[p['link']])[:3, :3] @ p['normal'] for p in pads])
        t = C.mean(axis=0)
        if len(C) == 2:
            c = C[1] - C[0]
            c = c / max(np.linalg.norm(c), 1e-300)
            nd = Nn[0] - Nn[1]
            if float(Nn[0] @ Nn[1]) <= -0.5:                 # the pads face each other within 60°
                c = nd / np.linalg.norm(nd)
            width = float(abs((C[1] - C[0]) @ c))
        elif len(C) > 2:
            _u, _s, Vt = np.linalg.svd(C - t)
            c = Vt[0]
            span = (C - t) @ c
            width = float(span.max() - span.min())
        else:
            c = np.array([1.0, 0.0, 0.0])
        a = t - c * float(t @ c)
        rule = 'measured from the contact pads at home'
        if np.linalg.norm(a) < AXIS_DEGENERATE_M:
            a = np.array([0.0, 0.0, 1.0]) - c * c[2]
            rule += ' (TCP on the frame\'s closing line: the frame\'s z axis is the approach)'
        a = a / np.linalg.norm(a)
        allp = np.concatenate([in_frame(p['points'], p['link']) for p in pads])
        pad_len = float((allp @ a).max() - (allp @ a).min())
    else:
        t, a, c = np.zeros(3), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
        rule = ('assumed: no fingertips or contact pads are tagged, so the TCP is the frame origin, the approach its '
                'z axis and the closing axis its x axis (tag them on the Robot page)')
    if tcp is not None:
        t = np.asarray(tcp, float).reshape(3)
        rule += '; TCP given'
    if approach_axis is not None:
        a = _unit(approach_axis)
        rule += '; approach axis given'
    if closing_axis is not None:
        c = _unit(closing_axis)
        rule += '; closing axis given'
    c = c - a * float(c @ a)
    if np.linalg.norm(c) < 1e-9:
        c = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
    c = c / np.linalg.norm(c)
    return {'frame': frame, 'tcp': t, 'approach': a, 'closing': c, 'width_m': width, 'pad_length_m': pad_len,
            'assumed': assumed, 'rule': rule}


def _geom_json(g: dict) -> dict:
    return {k: ([float(x) for x in v] if isinstance(v, np.ndarray) else v) for k, v in g.items()}


def poses_from_tcp(geom: dict, tcp_w, approach_w, closing_w) -> np.ndarray:
    """(N, 4, 4) gripper-frame world transforms whose TCP is at tcp_w (N, 3), approach axis along approach_w and
    closing axis along closing_w (N, 3, unit vectors, orthogonal)."""
    a_l, c_l = np.asarray(geom['approach'], float), np.asarray(geom['closing'], float)
    Bl = np.stack([c_l, a_l, np.cross(c_l, a_l)], axis=1)                       # columns
    A = np.asarray(approach_w, float).reshape(-1, 3)
    C = np.asarray(closing_w, float).reshape(-1, 3)
    Bw = np.stack([C, A, np.cross(C, A)], axis=2)                               # (N, 3, 3) columns
    R = Bw @ Bl.T
    T = np.zeros((len(A), 4, 4))
    T[:, :3, :3] = R
    T[:, :3, 3] = np.asarray(tcp_w, float).reshape(-1, 3) - R @ np.asarray(geom['tcp'], float)
    T[:, 3, 3] = 1.0
    return T


# ====================================================================== sampling
def _volume(project, uid, kind_wanted=None):
    vols = list(project.workspace.volumes)
    if uid:
        v = next((v for v in vols if v.uid == uid), None)
        if v is None:
            raise GraspError('The task volume chosen is no longer in the project: choose another.')
        return v
    if kind_wanted == 'handover':
        v = next((v for v in vols if v.kind == 'handover'), None)
        if v is None:
            raise GraspError('Bimanual handoff grasps need a handover zone: add a task volume of kind "handover" '
                             '(Workspace ▸ Add task volume) between the two robots, or leave the handoff kind out.')
        return v
    v = next((v for v in vols if v.kind != 'handover'), None) or (vols[0] if vols else None)
    if v is None:
        raise GraspError('The project has no task volume to sample grasps in: add one (Workspace ▸ Add task '
                         'volume).')
    return v


def _objects_in(project, vol) -> list:
    """[(object, bbox lo (3,), bbox hi (3,))] of the task objects whose bounding-box centre is inside vol."""
    from .scenegeom import object_triangles
    lo, hi = (np.asarray(x, float) for x in vol.bounds())
    out = []
    for o in project.workspace.objects:
        try:
            tri = object_triangles(o, project)
        except Exception:                                # noqa: BLE001 — an object without geometry is left out
            continue
        if tri is None or not len(tri):
            continue
        P = np.asarray(tri, float).reshape(-1, 3)
        a, b = P.min(axis=0), P.max(axis=0)
        c = (a + b) / 2
        if ((c >= lo - 1e-9) & (c <= hi + 1e-9)).all():
            out.append((o, a, b))
    return out


def _support_z(project, vol, xy) -> np.ndarray:
    lo, hi = (np.asarray(x, float) for x in vol.bounds())
    z = np.full(len(xy), lo[2])
    for s in project.workspace.surfaces:
        if not isinstance(s, dict):
            continue
        try:
            sz = float(s.get('z', 0.0))
            sx, sy = (float(v) for v in (s.get('size') or (0, 0))[:2])
            cx, cy = (float(v) for v in (s.get('center') or (0, 0))[:2])
        except (TypeError, ValueError):
            continue
        if not lo[2] - 1e-9 <= sz <= hi[2]:
            continue
        inside = (np.abs(xy[:, 0] - cx) <= sx / 2) & (np.abs(xy[:, 1] - cy) <= sy / 2)
        z = np.where(inside, np.maximum(z, sz), z)
    return z


def _heading(rng, n, base_xy, pts, spread_deg, toward=None):
    if toward is not None:
        d = np.broadcast_to(np.asarray(toward, float)[:2], (n, 2))
    else:
        d = pts[:, :2] - base_xy[:2]
    az0 = np.arctan2(d[:, 1], d[:, 0])
    small = np.linalg.norm(d, axis=1) < 1e-9
    az0 = np.where(small, rng.uniform(-np.pi, np.pi, n), az0)
    return az0 + np.radians(spread_deg) * rng.uniform(-1.0, 1.0, n)


def _options(options: dict) -> tuple[dict, list]:
    st = dict(DEFAULTS)
    given = []
    for k, v in (options or {}).items():
        if k not in DEFAULTS:
            raise GraspError(f'"{k}" is not a sampler setting; the settings are {", ".join(DEFAULTS)}.')
        if v is None:
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            raise GraspError(f'The sampler setting {k} must be a number, not "{v}".') from None
        a, b = OPTION_RANGES[k]
        if not (math.isfinite(x) and a <= x <= b):
            raise GraspError(f'The sampler setting {k} must be between {a:g} and {b:g}; it is {x:g}.')
        st[k] = x
        given.append(k)
    return st, [k for k in DEFAULTS if k not in given]


def sample(project, robot_uid: str, kinds=('top_down',), n: int = 100, rng_seed: int = 0, volume_uid=None,
           handover_uid=None, name: str | None = None, tcp=None, approach_axis=None, closing_axis=None,
           **options) -> dict:
    """A sampled grasp set (not added to the project: add_set does that). See the module docstring for the
    samplers; options are DEFAULTS' keys."""
    robot, model = _robot_and_model(project, robot_uid)
    kinds = [k for k in dict.fromkeys(kinds or ())]
    bad = [k for k in kinds if k not in KINDS]
    if bad or not kinds:
        raise GraspError(f'Choose the kinds of grasp to sample from {", ".join(KINDS)}'
                         + (f' ("{bad[0]}" is not one).' if bad else '.'))
    try:
        n = int(n)
    except (TypeError, ValueError):
        raise GraspError('The number of grasps must be a whole number.') from None
    if not 1 <= n <= MAX_GRASPS:
        raise GraspError(f'Sample between 1 and {MAX_GRASPS:,} grasps; {n:,} was asked for.'.replace(',', ' '))
    try:
        seed = int(rng_seed)
    except (TypeError, ValueError):
        raise GraspError('The seed must be a whole number.') from None
    if seed < 0:
        raise GraspError('The seed must be zero or a positive whole number.')
    st, defaults = _options(options)
    geom = gripper_geometry(project, robot_uid, tcp, approach_axis, closing_axis)
    rng = np.random.default_rng(seed)
    base = np.asarray(robot.base.position, float)
    notes = []
    seq = np.array([kinds[i % len(kinds)] for i in range(n)])
    table_kinds = [k for k in kinds if k != 'handoff']
    vol = _volume(project, volume_uid) if table_kinds else None
    if vol is not None and vol.kind == 'handover' and table_kinds:
        notes.append(f'{vol.name} is a handover zone: table grasps were sampled in it, on its floor or the tables.')
    hvol = _volume(project, handover_uid, 'handover') if 'handoff' in kinds else None
    if hvol is not None and hvol.kind != 'handover':
        notes.append(f'{hvol.name} is not of kind "handover"; handoff grasps were sampled in it as asked.')
    objs = _objects_in(project, vol) if vol is not None else []
    if vol is not None:
        notes.append(f'Grasp points on {len(objs)} task object{"s" if len(objs) != 1 else ""} in {vol.name}.' if objs
                     else f'No task object is in {vol.name}: grasp points are on its support surface, '
                          f'{st["height_m"] * 1000:.0f} mm up.')
    partner = None
    if hvol is not None:
        hc = np.asarray(hvol.center, float)
        others = [r for r in project.robots if r.uid != robot.uid]
        if others:
            partner = min(others, key=lambda r: float(np.linalg.norm(np.asarray(r.base.position) - hc)))
        else:
            notes.append('There is no second robot: handoff grasps approach away from this robot\'s base (a handover '
                         'to a person or a fixture).')
    depth = st['depth_m'] if st['depth_m'] is not None else (geom['pad_length_m'] / 2 if geom['pad_length_m']
                                                              else 0.0)
    up = np.array([0.0, 0.0, 1.0])
    tcp_w = np.zeros((n, 3))
    app = np.zeros((n, 3))
    clo = np.zeros((n, 3))
    obj_uid = np.full(n, '', dtype=object)
    width = np.full(n, np.nan)
    for kind in kinds:
        idx = np.flatnonzero(seq == kind)
        m = len(idx)
        if not m:
            continue
        if kind == 'handoff':
            lo, hi = (np.asarray(x, float) for x in hvol.bounds())
            P = lo + (hi - lo) * rng.random((m, 3))
            if partner is not None:
                tw = np.asarray(partner.base.position, float) - base
                az = _heading(rng, m, base, P, st['handoff_spread_deg'], toward=tw if np.linalg.norm(tw[:2]) > 1e-9
                              else None)
            else:
                az = _heading(rng, m, base, np.broadcast_to(np.asarray(hvol.center, float), (m, 3)),
                              st['handoff_spread_deg'])
            A = np.c_[np.cos(az), np.sin(az), np.zeros(m)]
            C = np.c_[-np.sin(az), np.cos(az), np.zeros(m)]
            tcp_w[idx], app[idx], clo[idx] = P, A, C
            continue
        lo, hi = (np.asarray(x, float) for x in vol.bounds())
        if objs:
            pick = rng.integers(0, len(objs), m)
            a_lo = np.array([objs[k][1] for k in pick])
            a_hi = np.array([objs[k][2] for k in pick])
            c = (a_lo + a_hi) / 2
            e = a_hi - a_lo
            obj_uid[idx] = [objs[k][0].uid for k in pick]
            P = c.copy()
        else:
            xy = lo[:2] + (hi[:2] - lo[:2]) * rng.random((m, 2))
            z = np.minimum(_support_z(project, vol, xy) + st['height_m'], hi[2])
            P = np.c_[xy, z]
            e = None
        if kind == 'top_down':
            A = np.tile(-up, (m, 1))
            if objs:
                narrow_x = e[:, 0] <= e[:, 1]
                yaw = np.where(narrow_x, 0.0, np.pi / 2) + np.radians(st['yaw_jitter_deg']) * rng.uniform(-1, 1, m)
                P[:, 2] = c[:, 2] + e[:, 2] / 2 - np.minimum(depth, e[:, 2] / 2)
            else:
                yaw = rng.uniform(0.0, np.pi, m)
            C = np.c_[np.cos(yaw), np.sin(yaw), np.zeros(m)]
        else:
            az = _heading(rng, m, base, P, st['spread_deg'])
            if kind == 'side':
                A = np.c_[np.cos(az), np.sin(az), np.zeros(m)]
            else:
                th = math.radians(st['angled_tilt_deg'])
                A = np.c_[np.cos(az) * math.sin(th), np.sin(az) * math.sin(th), np.full(m, -math.cos(th))]
            C = np.c_[-np.sin(az), np.cos(az), np.zeros(m)]
        if objs:
            width[idx] = (np.abs(C) * e).sum(axis=1)
        tcp_w[idx], app[idx], clo[idx] = P, A, C
    T = poses_from_tcp(geom, tcp_w, app, clo)
    poses = [_pose_rec(f'g{i + 1:05d}', T[i], app[i], str(seq[i]), obj_uid[i],
                       None if not math.isfinite(width[i]) else float(width[i])) for i in range(n)]
    if geom['assumed']:
        notes.append(geom['rule'][0].upper() + geom['rule'][1:] + '.')
    label = ', '.join(KIND_TITLES[k].lower() for k in kinds)
    settings = {'kinds': list(kinds), 'n': n, 'rng_seed': seed, 'volume_uid': vol.uid if vol is not None else '',
                'handover_uid': hvol.uid if hvol is not None else '',
                'partner_robot_uid': partner.uid if partner is not None else '',
                'options': {k: st[k] for k in DEFAULTS}, 'defaults': defaults, 'depth_used_m': float(depth),
                'gripper': _geom_json(geom), 'notes': notes}
    return normalise_set({'uid': new_uid(), 'name': name or f'Sampled: {label} ({n})', 'source': 'sampled',
                          'robot_uid': robot.uid, 'frame': geom['frame'], 'poses': poses, 'settings': settings})


# ====================================================================== imported files
def _csv_rows(text: str):
    lines = [(k + 1, ln) for k, ln in enumerate(text.splitlines()) if ln.strip() and not ln.lstrip().startswith('#')]
    if not lines:
        raise GraspError('The file is empty: a grasp CSV needs a header row (id, x, y, z, qx, qy, qz, qw) and at least '
                         'one row of numbers.')
    head = lines[0][1]
    delim = ';' if head.count(';') > head.count(',') else ('\t' if head.count('\t') > head.count(',') else ',')
    rows = []
    for (ln, _t), r in zip(lines, csv.reader(io.StringIO('\n'.join(t for _k, t in lines)), delimiter=delim)):
        r = [c.strip() for c in r]
        if delim != ',' and rows:
            r = [c.replace(',', '.') if c.count(',') == 1 and '.' not in c else c for c in r]
        rows.append((ln, r))
    return rows


def _num(cell, where, col):
    try:
        v = float(str(cell).replace('−', '-'))
    except (TypeError, ValueError):
        raise GraspError(f'{where}, column {col}: "{cell}" is not a number. Write positions in metres and the '
                         'quaternion as four numbers (x, y, z, w).') from None
    if not math.isfinite(v):
        raise GraspError(f'{where}, column {col} is not a finite number.')
    return v


def _row_to_pose(rec: dict, where: str, approach_local, warn: list) -> dict:
    pos = np.array([_num(rec[k], where, k) for k in ('x', 'y', 'z')])
    q = np.array([_num(rec[k], where, k) for k in ('qx', 'qy', 'qz', 'qw')])
    nq = float(np.linalg.norm(q))
    if nq < 1e-9:
        raise GraspError(f'{where}: the quaternion (qx, qy, qz, qw) is all zeros; a rotation needs a quaternion of '
                         'length 1, for example 0, 0, 0, 1 for no rotation.')
    if abs(nq - 1) > 1e-3:
        warn.append('norm')
    q = quat_normalize(q)
    if float(np.linalg.norm(pos)) > MM_HINT_M:
        warn.append('mm')
    av = [rec.get(k) for k in ('ax', 'ay', 'az')]
    given = [v not in (None, '') for v in av]
    if any(given) and not all(given):
        raise GraspError(f'{where}: give all three approach components (ax, ay, az) or none of them.')
    R = quat_to_matrix(q)
    if all(given):
        a = np.array([_num(v, where, k) for v, k in zip(av, ('ax', 'ay', 'az'))])
        if float(np.linalg.norm(a)) < 1e-9:
            raise GraspError(f'{where}: the approach (ax, ay, az) has zero length; write the direction the gripper '
                             'moves into the grasp, for example 0, 0, -1 for straight down.')
        a = a / np.linalg.norm(a)
    else:
        a = R @ np.asarray(approach_local, float)
    kind = str(rec.get('kind') or 'imported').strip().lower().replace('-', '_').replace(' ', '_') or 'imported'
    if kind not in POSE_KINDS:
        raise GraspError(f'{where}: the kind "{rec.get("kind")}" is not one DexVantage knows; write one of '
                         f'{", ".join(POSE_KINDS)}, or leave it blank.')
    w = rec.get('width_m')
    t = rec.get('t')
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    pid = str(rec.get('id') if rec.get('id') not in (None, '') else '').strip()
    if not pid:
        raise GraspError(f'{where}: the id is blank; give every grasp an id (a number or a name).')
    return _pose_rec(pid, T, a, kind, str(rec.get('object') or ''),
                     None if w in (None, '') else _num(w, where, 'width_m'),
                     None if t in (None, '') else _num(t, where, 't'))


def _parse_csv(text, approach_local, warn) -> list:
    rows = _csv_rows(text)
    ln0, head = rows[0]
    cols = [CSV_ALIASES.get(h.strip().lower(), h.strip().lower()) for h in head]
    missing = [c for c in CSV_REQUIRED if c not in cols]
    if missing:
        raise GraspError(f'The header (line {ln0}) has no column {", ".join(missing)}: a grasp CSV needs the columns '
                         'id, x, y, z, qx, qy, qz, qw (metres and a quaternion x, y, z, w), and may add ax, ay, az, '
                         'kind, object, width_m and t.')
    dup = sorted({c for c in cols if cols.count(c) > 1 and c})
    if dup:
        raise GraspError(f'The header (line {ln0}) names column {dup[0]} twice: keep one.')
    out = []
    for ln, r in rows[1:]:
        if len(r) != len(cols):
            raise GraspError(f'Line {ln} has {len(r)} values; the header has {len(cols)} columns. Every row needs one '
                             'value per column (leave optional ones empty, not out).')
        out.append(_row_to_pose(dict(zip(cols, r)), f'Line {ln}', approach_local, warn))
    if not out:
        raise GraspError('The file has a header but no rows of grasps.')
    return out


def _parse_json(doc, approach_local, warn) -> tuple[list, str]:
    name = ''
    if isinstance(doc, dict):
        name = str(doc.get('name') or '')
        doc = doc.get('poses', doc.get('grasps'))
    if not isinstance(doc, list):
        raise GraspError('The JSON file holds no list of grasps: write a list of objects, each with an id, a '
                         'position [x, y, z] and a quaternion [x, y, z, w] (or a grasp set with "poses").')
    if not doc:
        raise GraspError('The JSON file\'s list of grasps is empty.')
    out = []
    for i, item in enumerate(doc):
        where = f'Item {i + 1}'
        if not isinstance(item, dict):
            raise GraspError(f'{where} is not an object: each grasp is {{"id": …, "position": [x, y, z], '
                             '"quaternion": [x, y, z, w]}.')
        rec = {k.lower(): v for k, v in item.items()}
        rec = {CSV_ALIASES.get(k, k): v for k, v in rec.items()}
        pose = rec.get('pose') if isinstance(rec.get('pose'), dict) else rec
        pos, quat = pose.get('position'), pose.get('quaternion')
        if pos is not None or quat is not None:
            if not (isinstance(pos, (list, tuple)) and len(pos) == 3):
                raise GraspError(f'{where}: "position" must be a list of three numbers [x, y, z] in metres.')
            if not (isinstance(quat, (list, tuple)) and len(quat) == 4):
                raise GraspError(f'{where}: "quaternion" must be a list of four numbers [x, y, z, w].')
            rec.update(zip(('x', 'y', 'z'), pos))
            rec.update(zip(('qx', 'qy', 'qz', 'qw'), quat))
        missing = [k for k in CSV_REQUIRED if k != 'id' and rec.get(k) is None]
        if missing:
            raise GraspError(f'{where} has no {", ".join(missing)}: give a position and a quaternion.')
        a = rec.get('approach')
        if a is not None:
            if not (isinstance(a, (list, tuple)) and len(a) == 3):
                raise GraspError(f'{where}: "approach" must be a list of three numbers [x, y, z].')
            rec.update(zip(('ax', 'ay', 'az'), a))
        if rec.get('id') in (None, ''):
            rec['id'] = str(i + 1)
        out.append(_row_to_pose(rec, where, approach_local, warn))
    return out, name


def import_file(path: str, project=None, robot_uid: str = '', name: str | None = None) -> dict:
    """A grasp set read from a CSV or JSON file (see the module docstring); not added to the project. With a project
    and a robot, rows without an approach get the gripper's approach axis and the set names the robot's gripper
    frame. Raises GraspError naming the line (CSV) or item (JSON) it cannot read."""
    base = os.path.splitext(os.path.basename(path))[0] or 'Imported'
    try:
        with open(path, encoding='utf-8-sig') as fh:
            text = fh.read()
    except FileNotFoundError:
        raise GraspError(f'There is no file at {os.path.basename(path)}.') from None
    except UnicodeDecodeError:
        raise GraspError(f'{os.path.basename(path)} is not a text file: save the grasps as CSV or JSON.') from None
    except OSError as e:
        raise GraspError(f'Could not read {os.path.basename(path)}: {e.strerror or e}.') from None
    frame, approach_local, rule = '', np.array([0.0, 0.0, 1.0]), 'the gripper frame\'s z axis (no robot given)'
    if project is not None and robot_uid:
        g = gripper_geometry(project, robot_uid)
        frame, approach_local, rule = g['frame'], g['approach'], 'the gripper\'s approach axis (' + g['rule'] + ')'
    warn: list = []
    is_json = path.lower().endswith('.json') or text.lstrip()[:1] in ('[', '{')
    title = ''
    if is_json:
        try:
            doc = json.loads(text)
        except ValueError as e:
            raise GraspError(f'{os.path.basename(path)} is not valid JSON ({e}).') from None
        poses, title = _parse_json(doc, approach_local, warn)
    else:
        poses = _parse_csv(text, approach_local, warn)
    if len(poses) > MAX_GRASPS:
        raise GraspError(f'The file holds {len(poses)} grasps; a set holds at most {MAX_GRASPS}.')
    ids = [p['id'] for p in poses]
    dup = next((i for i in ids if ids.count(i) > 1), None)
    if dup is not None:
        raise GraspError(f'The id {dup} is used by more than one grasp: give every grasp its own id.')
    notes = []
    if 'norm' in warn:
        notes.append(f'{warn.count("norm")} quaternion{"s were" if warn.count("norm") != 1 else " was"} not of unit '
                     'length and were normalised.')
    if 'mm' in warn:
        notes.append(f'{warn.count("mm")} position{"s are" if warn.count("mm") != 1 else " is"} more than '
                     f'{MM_HINT_M:g} m from the origin: the file must be in metres, not millimetres.')
    settings = {'file': os.path.basename(path), 'format': 'json' if is_json else 'csv',
                'approach_default': rule, 'notes': notes}
    return normalise_set({'uid': new_uid(), 'name': name or title or base, 'source': 'imported',
                          'robot_uid': robot_uid or '', 'frame': frame, 'poses': poses, 'settings': settings})


# ====================================================================== from a trajectory
def pad_opening(project, robot_uid: str, Q) -> np.ndarray:
    """(N,) the gripper's opening at the joint vectors Q (N, nq): twice the mean distance of the contact pads'
    centroids (carried with their links) from their common centroid, which for two pads is the distance between
    them. Raises GraspError when the robot has fewer than two pads or fingertips."""
    pads, _frame, m = _pads_local(project, robot_uid)
    if len(pads) < 2:
        raise GraspError('Grasps are found in a trajectory where the gripper closes, which needs two or more '
                         'fingertips or contact pads tagged (Robot page ▸ Fingertips): take every Nth sample '
                         'instead.')
    T = m.fk_batch(np.asarray(Q, float).reshape(-1, m.nq))
    li = [m.link_index(p['link']) for p in pads]
    C = np.stack([T[:, k, :3, :3] @ np.asarray(p['centroid'], float) + T[:, k, :3, 3]
                  for k, p in zip(li, pads)], axis=1)                                        # (N, P, 3)
    return 2.0 * np.linalg.norm(C - C.mean(axis=1, keepdims=True), axis=2).mean(axis=1)


def closure_events(opening, close_m: float = CLOSE_MM / 1000.0, flat_m: float = FLAT_MM / 1000.0) -> list:
    """[(onset, end, drop_m)]: sample indices where a closing stroke starts and where it ends, and how far the
    opening fell. A stroke is a run of samples over which the opening never rises by more than flat_m between two
    samples (pauses are part of it), that starts with a fall of more than flat_m and ends at its last such fall; it
    counts when the opening falls by close_m or more over it."""
    d = np.asarray(opening, float).reshape(-1)
    dd = np.diff(d)
    falls, rises = dd < -flat_m, dd > flat_m
    out, k = [], 0
    while k < len(dd):
        if not falls[k]:
            k += 1
            continue
        start, last, j = k, k, k
        while j < len(dd) and not rises[j]:
            if falls[j]:
                last = j
            j += 1
        drop = float(d[start] - d[last + 1])
        if drop >= close_m:
            out.append((start, last + 1, drop))
        k = j
    return out


def from_trajectory(project, traj_uid: str, every: int = 1, name: str | None = None, mode: str = 'every',
                    close_mm: float = CLOSE_MM, approach_mm: float = APPROACH_MM) -> dict:
    """A grasp set from a joint trajectory (not added to the project); t is recorded on each pose.

    mode 'every': the gripper frame at every `every`-th sample, the approach the gripper's approach axis there.
    mode 'closures': the grasps in the trajectory, one where the gripper closes (closure_events on pad_opening at
    the trajectory's samples, a fall of close_mm or more, default 5 mm, a default): the gripper frame at the stroke's
    onset (the arm in place, the fingers still open, as the grasp study poses them) and the approach the direction
    the tool centre travelled over its last approach_mm (default 20 mm, a default) before it; a grasp the gripper
    reached without moving that far (the trajectory starts there) takes the gripper's approach axis, said in the
    notes."""
    from .robot import trajectory as TR
    if mode not in TRAJECTORY_MODES:
        raise GraspError(f'The trajectory mode must be one of {", ".join(TRAJECTORY_MODES)}.')
    tr = next((t for t in project.tasks.trajectories if isinstance(t, dict) and t.get('uid') == traj_uid), None)
    if tr is None:
        raise GraspError('There is no such trajectory in the project: import one first (Robot ▸ Import '
                         'trajectory…).')
    if mode == 'closures':
        return _closures_from_trajectory(project, tr, name, close_mm, approach_mm)
    try:
        every = int(every)
    except (TypeError, ValueError):
        raise GraspError('"Every" must be a whole number of samples.') from None
    if every < 1:
        raise GraspError('"Every" must be 1 or more (1 takes every sample).')
    robot, model = _robot_and_model(project, tr.get('robot_uid', ''))
    geom = gripper_geometry(project, robot.uid)
    i = model.link_index(geom['frame'])
    ts = list(tr.get('t') or [])
    if not ts:
        raise GraspError(f'The trajectory {tr.get("name", "")} has no samples.')
    ks = list(range(0, len(ts), every))
    if len(ks) > MAX_GRASPS:
        raise GraspError(f'That takes {len(ks)} poses; a set holds at most {MAX_GRASPS}: take every '
                         f'{math.ceil(len(ts) / MAX_GRASPS)}th sample or more.')
    Q = np.array([model.clip(TR.on_model(tr, model.joint_names, model.home, float(ts[k]))) for k in ks])
    T = model.fk_batch(Q)[:, i]
    poses = [_pose_rec(f't{k}', T[j], T[j, :3, :3] @ geom['approach'], 'trajectory', t=float(ts[k]))
             for j, k in enumerate(ks)]
    settings = {'trajectory_uid': traj_uid, 'trajectory': str(tr.get('name', '')), 'every': every,
                'gripper': _geom_json(geom), 'notes': [] if not geom['assumed'] else [geom['rule']]}
    return normalise_set({'uid': new_uid(), 'name': name or f'{tr.get("name", "Trajectory")} (every {every})',
                          'source': 'trajectory', 'robot_uid': robot.uid, 'frame': geom['frame'], 'poses': poses,
                          'settings': settings})


def _closures_from_trajectory(project, tr: dict, name, close_mm, approach_mm) -> dict:
    from .robot import trajectory as TR
    try:
        close_mm, approach_mm = float(close_mm), float(approach_mm)
    except (TypeError, ValueError):
        raise GraspError('The closing and approach distances must be numbers of millimetres.') from None
    if not (0.1 <= close_mm <= 500.0) or not (0.0 <= approach_mm <= 2000.0):
        raise GraspError('The closing distance must be 0.1 to 500 mm and the approach distance 0 to 2000 mm.')
    robot, model = _robot_and_model(project, tr.get('robot_uid', ''))
    geom = gripper_geometry(project, robot.uid)
    i = model.link_index(geom['frame'])
    ts = [float(t) for t in tr.get('t') or []]
    if len(ts) < 2:
        raise GraspError(f'The trajectory {tr.get("name", "")} has fewer than two samples: no closing can be seen.')
    Q = np.array([model.clip(TR.on_model(tr, model.joint_names, model.home, t)) for t in ts])
    opening = pad_opening(project, robot.uid, Q)
    events = closure_events(opening, close_mm / 1000.0)
    if not events:
        raise GraspError(f'The gripper never closes by {close_mm:g} mm or more in {tr.get("name", "the trajectory")} '
                         f'(its opening runs from {1000 * opening.min():.1f} to {1000 * opening.max():.1f} mm): '
                         'lower the closing distance, or take every Nth sample instead.')
    if len(events) > MAX_GRASPS:
        raise GraspError(f'The trajectory closes the gripper {len(events)} times; a set holds at most {MAX_GRASPS}.')
    T = model.fk_batch(Q)[:, i]
    tcp = T[:, :3, :3] @ np.asarray(geom['tcp'], float) + T[:, :3, 3]
    poses, notes, axis_used, events_doc = [], [], 0, []
    for n, (k0, k1, drop) in enumerate(events):
        a = T[k0, :3, :3] @ geom['approach']
        back = np.linalg.norm(tcp[:k0 + 1] - tcp[k0], axis=1)
        far = np.nonzero(back >= approach_mm / 1000.0)[0]
        if approach_mm > 0 and len(far):
            v = tcp[k0] - tcp[far[-1]]
            a = v / np.linalg.norm(v)
        else:
            axis_used += 1
        poses.append(_pose_rec(f'c{n + 1}', T[k0], a, 'trajectory', t=ts[k0]))
        events_doc.append({'id': f'c{n + 1}', 't': ts[k0], 't_end': ts[k1], 'drop_m': drop,
                           'opening_m': float(opening[k0])})
    if axis_used:
        notes.append(f'{axis_used} of {len(events)} grasps were reached without the tool centre moving '
                     f'{approach_mm:g} mm first: their approach is the gripper\'s approach axis.')
    if geom['assumed']:
        notes.append(geom['rule'])
    settings = {'trajectory_uid': tr.get('uid'), 'trajectory': str(tr.get('name', '')), 'mode': 'closures',
                'close_mm': close_mm, 'approach_mm': approach_mm,
                'defaults': [k for k, v, d in (('close_mm', close_mm, CLOSE_MM), ('approach_mm', approach_mm,
                                                                                     APPROACH_MM)) if v == d],
                'closures': len(events), 'events': events_doc, 'gripper': _geom_json(geom), 'notes': notes}
    return normalise_set({'uid': new_uid(), 'name': name or f'{tr.get("name", "Trajectory")} (grasps)',
                          'source': 'trajectory', 'robot_uid': robot.uid, 'frame': geom['frame'], 'poses': poses,
                          'settings': settings})


# ====================================================================== objects
def object_points(project, object_uid: str, n: int, rng_seed: int = 0) -> np.ndarray:
    """(n, 3) world points drawn uniformly by area on a task object's surface (core/scenegeom.object_triangles)."""
    from .scenegeom import object_triangles
    o = next((o for o in project.workspace.objects if o.uid == object_uid), None)
    if o is None:
        raise GraspError('There is no such task object in the project.')
    tri = object_triangles(o, project)
    if tri is None or not len(tri):
        raise GraspError(f'{o.name} has no surface to sample (a mesh object whose mesh is not stored?).')
    P, _k = sample_triangles(np.asarray(tri, float), int(n), np.random.default_rng(int(rng_seed)))
    return P


def ik_targets(project, gs: dict) -> tuple[str, np.ndarray, np.ndarray]:
    """(link, tcp (4 × 4 offset in the link frame), targets (N, 4, 4) world poses of that TCP frame) for
    core.robot.ik.solve_batch(model, link, targets, tcp=tcp): solving for the TCP rather than the frame's origin
    makes a position-first solve (a 5-joint arm) put the grasp centre, not the gripper frame, on the target. The TCP
    is the one the set was made with (settings['gripper'], which records a sampler's tcp override) when it names the
    same frame, else the one measured now."""
    geom = gripper_geometry(project, gs.get('robot_uid', ''))
    frame = gs.get('frame') or geom['frame']
    off = np.eye(4)
    used = (gs.get('settings') or {}).get('gripper') or {}
    tcp = geom['tcp']
    if used.get('frame') == frame and used.get('tcp') is not None:
        t = np.asarray(used['tcp'], float).reshape(-1)
        if t.size == 3 and np.isfinite(t).all():
            tcp = t
    off[:3, 3] = tcp if frame == geom['frame'] else 0.0
    T = np.array([grasp_matrix(g) for g in gs.get('poses') or []]).reshape(-1, 4, 4)
    return frame, off, T @ off
