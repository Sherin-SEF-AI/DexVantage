"""Dexterous multi-finger self-occlusion (spec §7.14, part 1) and the visual-to-tactile handoff (spec §7.15, part 2).

    @jobs.study('dex_occlusion', 'Multi-finger occlusion', quick=False, batch='dex-occlusion')  → --dex-occlusion OUT
    run(project, ctx) -> dict                 the study (also through core.jobs.run_study('dex_occlusion', project))
    visibility_owner(points, origin, occluders, group_of, n_groups, ...) -> (visible, owner)   the kernel
    visibility_owner_raycast(...)             the same answer by exact ray casting (backend 'raycast', the default)
    patch_counts(in_view, visible, owner, n_groups)    a patch's points: hidden by each group, outside, seen
    status_for(pct, settings) -> (status, word)
    tactile_handoff(project, ctx=None) -> dict   part 2 alone (the document run() puts under 'handoff')
    tactile_pads(project, robot, hand, q), pad_coverage(points, pad, gap_m), gap_from_samples(dist, seen)
    blind_gap_check(project)                  part 2's adaptive scan against a brute-force scan (validation)

What it computes. The hand is the robot whose fingertips are tagged (settings robot_uid, else the first robot with
two or more fingertip frames); its fingers, palm and the rest of the robot come from core/dexhand.py structure()
(each fingertip's finger is the chain of links from the tip up to the palm). Each fingertip's contact patch is the
point set of core/grasps.py contact_patches (Robot.contact_pads when tagged, else the fingertips' inner faces;
the 'dexhand' sample tags the palmar side of each distal link, core/dexhand.py pad_faces). The study poses the robot
at a series of cases:

- 'synergy' (the default): the open hand, then each grasp posture (core/dexhand.py Synergy, a straight line in joint
  space from an open to a closed posture) at closures 1/steps, 2/steps, ..., 1 (steps = 4, a default); the other
  joints at the joint state in the settings (q, the page passes the one on screen) or home.
- 'grasp_set': the grasps of a grasp set (core/grasps.py, project.tasks.grasp_sets, at most max_grasps, default 200,
  evenly spaced through the set), each solved by IK (core/robot/ik.py solve_batch for grasps.ik_targets(), seeds 4,
  the rule nearest to home) with the hand held at grasp_synergy (default 'power') at grasp_closure (default 1, fully
  closed); infeasible grasps are counted by reason and left out. Self-collision is not checked by default
  (ik_collision False): in a closed grasp posture neighbouring fingers touch, and the convex hulls core/robot/
  collision.py uses would refuse every solution.

For each case, each camera c (every enabled camera and depth sensor, or settings 'sensors'; a depth sensor's RGB
imager, as the coverage study chooses, core/studies/coverage.py pick_imager) and each patch point P:

- in view: P projects inside the imager's image (its lens model, core/optics: CameraModel.project) from the
  imager's optical centre o at its pose at the case's joint state;
- the owner of the first hit: the occluders (the posed robots' visual meshes and the tables, core/scenegeom.py; task
  objects only with objects_occlude, default False, because a fingertip touching an object is hidden by it by
  definition, and this study measures what the hand itself hides; the camera's own seat on its mount link is left
  out by scenegeom.sensor_occluders' rule) are split into owner groups: one per finger, the palm, the wrist and arm
  (every other link of the hand's robot), task objects, and the rest of the scene. d_g(P) is the distance from o to
  the first triangle of group g along the ray towards P, and d(P) = min_g d_g(P) the first hit of the whole scene;
  the owner is the group that attains it. P is seen when |P − o| ≤ d(P) + eps (eps = 1 mm, core/occlusion.py's
  default). backend 'raycast' (the default) finds d_g exactly (core/raycast.py first_hit, one grid per group, the
  ray started at the 5 mm near distance). backends 'auto', 'gl' and 'numpy' read d_g from one cube map of distances
  per group around o (core/occlusion.py); since a z-buffer of a union keeps the nearest of its parts, min_g d_g is
  exactly the whole scene's cube-map distance (validation dexhand.cube_owner_union), and a group whose bounding box
  is farther from o than every point is skipped. The cube map answers per texel, which matters here: a patch point
  lies on its own surface, and where that surface curves or is seen at a grazing angle the texel-centre ray (one
  texel is 2.7 mm wide at 0.7 m for n = 512) meets it a few millimetres in front of or behind the point. On the
  dexhand sample the thumb's not-seen share from the 0.7 m head camera differed from exact ray casting by up to 6
  points (percent of its patch), in both directions (docs/models/dexhand.md), so exact casting is the default; the
  cube maps are there for very large grasp sets on a GPU.
- a patch point is seen by camera c when it is in view and seen; else it is "outside the image" or "hidden by" the
  owner of the first hit. The pad's own finger is a possible owner: a pad facing away from a camera is hidden by the
  back of its own finger.

The finger × camera occlusion matrix is, for every finger f and camera c, the share of f's patch not seen by c,
averaged over the cases, split by why (each owner group and "outside the image"); its dominant entry names which
finger (or the palm, or the arm) hides the patch most. A fingertip counts as seen in a case when at least
min_visible_pct (default 50 %, a default) of its patch is seen, by one camera (the camera's own figures) or by the
rig (any camera seeing each point, the union). The headline is the share of fingertip-cases the rig sees; its
status is 'good' from good_pct (default 90 %), 'warning' from warn_pct (60 %), 'serious' from serious_pct (30 %) and
'critical' below (user settings with documented defaults, not standards). The mount comparison ranks the cameras
by the share of fingertip-cases each sees alone, with the share of cases in which it sees every fingertip, its
worst finger and what hides the most.

What to move. For each unlocked camera, the study turns it by ±suggest_deg (default 10°, a default) in pitch and in
yaw about its own optical centre and recounts the rig's headline. Turning about the optical centre changes only which
points are in view: the cube maps (world-axis aligned around the centre) are reused, so the search costs no
rendering. The best turn that raises the headline (or, when it is equal, the rig's mean share of patch seen by 0.5
points or more) is the suggestion; Apply sets that mount pose (one undo step).

Part 2, the visual-to-tactile handoff (spec §7.15; settings handoff, default on). The approaches end at the contact
cases: each grasp of a grasp set (reached by IK), or each grasp posture fully closed (at most handoff_max_cases,
default 6, evenly spaced). Along each, the gripper frame is backed off along −approach (the grasp's approach, else
the gripper's approach axis, core/grasps.py gripper_geometry, turned with the frame) by a distance d, and the arm is
solved by IK from the solution at the nearer distance ('nearest_q0', one seed, self-collision not checked: the hand
keeps the case's closed posture), as the grasp study poses its pre-contact steps. At each d, part 1's kernel
(visibility_at) gives every fingertip's share of its contact patch that some camera sees (the rig's union), with the
task objects occluding (handoff_objects_occlude, default True: the object is what hides the contact as the fingers
close on it). A fingertip is seen at d from min_visible_pct (50 %) of its patch. The distances: 0 to handoff_max_mm
(default 50 mm, the grasp study's pre-contact default) in handoff_step_mm (10 mm), then every handoff_refine_mm
(2 mm) inside each coarse interval where a fingertip's seen state changes (all three defaults).

At contact (d = 0), the tactile pads: every tactile sensor in use mounted on a link of a finger covers that finger's
patch points within pad_gap_mm (default 3 mm) of its sensing plane and inside its pad (the library entry's pad_mm,
cited; not stated: coverage not known). A pad left at its link's origin (the mount pose not set) is taken to sit
centred on the patch, facing along its mean normal (an assumption, labelled in the result); a placed pad senses along
its mount's body x axis. A fingertip is touched from min_touch_pct (50 %) of its patch on a pad.

The SENSORY BLIND GAP of a fingertip along an approach is the distance before contact over which it is neither seen
nor touched: from contact to the midpoint between the last distance not seen and the first seen (0 when seen at
contact; the whole scanned distance, "or more", when never seen). Touch begins only at contact, so a fingertip
neither seen nor touched at contact is "unsensed at contact" whatever its gap. Flags: a gap above blind_gap_max_mm
(default 10 mm) or a contact unsensed. Status: good with no flag, warning for flagged gaps up to 2 × the threshold,
serious above, critical when a contact is unsensed (defaults). The result's 'handoff' holds HANDOFF_FIELDS
(approach_visibility: per approach the distances, the seen share per fingertip, the gaps; tactile_coverage; blind_gap_m,
the worst) and the flags, status and notes, or {'error'} when part 2 could not run.

Settings (project.analysis.studies['dex_occlusion'], every key optional): robot_uid; source ('synergy' |
'grasp_set'); synergies (records, core/dexhand.py Synergy.to_dict; the 'dexhand' sample stores the Shadow Hand's);
synergy_ids (a subset); steps; q {robot uid: joint vector}; grasp_set_uid; grasp_synergy; grasp_closure;
max_grasps; ik_seeds; ik_collision; min_visible_pct; good_pct, warn_pct, serious_pct; cube_n (512); backend ('raycast',
default, or 'auto', 'gl', 'numpy'); sensors [uid]; imagers, modes, depth_imager (as coverage); objects_occlude;
suggest_deg (0 switches the search off).

The result is JSON-able: the fingers, the owner groups, the cameras with their figures, the cases, the matrix
(not-seen % per finger and camera and its split by owner), per case the seen share per camera and finger and the
dominant owner and the seen patch points (seen_bits: per camera, every finger's points in order, pack_bits), the
rig's headline with its status, the comparison, the suggestion, the timings and the notes.

Validation (validation/checks_dexhand.py, and validation/checks_chunk.py for part 2): dexhand.matrix_vs_analytic builds
a scripted two-finger scene whose occlusion is known in closed form (a box finger's shadow, by central projection, on
the other's pad, and a palm plate's edge seen from a low side camera) and compares this module's matrix entries with it;
dexhand.owner_vs_bruteforce compares the per-point owners with an all-pairs Möller–Trumbore ray cast that records the
owner of every hit; dexhand.cube_owner_union checks d = min_g d_g against one cube map of the whole scene on the sample;
dexhand.blind_gap_bruteforce compares part 2's adaptive scan of an approach with a scan of every distance at the refine
spacing, on the sample with a wall that hides some fingertips near contact.

Known limits: with a cube-map backend visibility is judged per texel (a point within about a texel of a silhouette,
or on its own surface seen at a grazing angle, may go either way); the patch is a fixed point set carried with the
fingertip link (its contact area does not change with the object); the postures are straight lines between two
joint vectors, not measured synergies (core/dexhand.py), and tendon-coupled joints move independently; the arm's
joints are fixed during the synergy sweep; grasp-set IK holds the hand at one posture for every grasp and does not
check self-collision by default; no camera housing occludes; the re-aim search tries single turns of one camera,
not combinations or moves; lighting and blur are other studies. Part 2: the approach is a straight line in the
gripper frame's position with the hand already in its contact posture (the fingers do not close as it comes in); a
seen or hidden stretch shorter than the coarse step inside one interval can be missed (the validation check measures
it on one scene); IK that fails at a distance leaves it out; touch is judged only at contact, by geometry (a pad's
footprint on the patch), not by contact force or the sensor's sensitivity; a pad's assumed placement is a guess the
result labels, and only its mount pose says where it really is.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .. import jobs
from ..jobs import StudyError

STUDY_ID = 'dex_occlusion'
DEFAULTS = {
    'source': 'synergy',
    'steps': 4,                  # default: closures 1/4, 1/2, 3/4, 1 of each posture (and the open hand once)
    'grasp_synergy': 'power',    # default: the hand's posture for grasp-set cases
    'grasp_closure': 1.0,        # default: fully closed
    'max_grasps': 200,           # default: grasps of a set taken (evenly spaced)
    'ik_seeds': 4,
    'ik_collision': False,
    'min_visible_pct': 50.0,     # default: a fingertip counts as seen from this share of its patch
    'good_pct': 90.0,            # default: rig share of fingertip-cases seen that reads 'good'
    'warn_pct': 60.0,
    'serious_pct': 30.0,
    'cube_n': 512,
    'backend': 'raycast',          # default: exact ray casting; 'auto', 'gl', 'numpy' use cube maps
    'objects_occlude': False,
    'suggest_deg': 10.0,         # default: the re-aim search step
    'depth_imager': 'rgb',
    # part 2, the visual-to-tactile handoff (§7.15); every figure a default
    'handoff': True,             # default: run the handoff with the occlusion matrix
    'handoff_max_mm': 50.0,      # default: the last 5 cm of the approach (the grasp study's pre-contact default)
    'handoff_step_mm': 10.0,     # default: coarse distances along the approach
    'handoff_refine_mm': 2.0,    # default: the spacing inside an interval where a fingertip's seen state changes
    'handoff_max_cases': 6,      # default: approaches taken (evenly spaced over the contact cases)
    'handoff_objects_occlude': True,   # default: along the approach the task objects hide the contact
    'min_touch_pct': 50.0,       # default: a fingertip counts as touched from this share of its patch on a pad
    'pad_gap_mm': 3.0,           # default: a patch point this close to a pad's plane is on the pad
    'blind_gap_max_mm': 10.0,    # default: a blind gap longer than this is flagged
}
LIMITS = {'steps': (1, 20), 'max_grasps': (1, 5000), 'ik_seeds': (1, 32), 'min_visible_pct': (0.0, 100.0),
          'good_pct': (0.0, 100.0), 'warn_pct': (0.0, 100.0), 'serious_pct': (0.0, 100.0), 'cube_n': (16, 4096),
          'grasp_closure': (0.0, 1.0), 'suggest_deg': (0.0, 45.0), 'handoff_max_mm': (0.5, 1000.0),
          'handoff_step_mm': (0.1, 500.0), 'handoff_refine_mm': (0.05, 100.0), 'handoff_max_cases': (1, 500),
          'min_touch_pct': (0.0, 100.0), 'pad_gap_mm': (0.0, 50.0), 'blind_gap_max_mm': (0.0, 1000.0)}
INT_SETTINGS = ('steps', 'max_grasps', 'ik_seeds', 'cube_n', 'handoff_max_cases')
BACKENDS = ('raycast', 'auto', 'gl', 'numpy')
WORDS = {'good': 'fingertips seen', 'warning': 'partly hidden', 'serious': 'mostly hidden',
         'critical': 'fingertips blind'}
OUT = 'out_of_view'
GROUP_TITLES = {'palm': 'Palm', 'arm': 'Wrist and arm', 'objects': 'Task objects', 'scene': 'Rest of the scene',
                OUT: 'Outside the image'}
HANDOFF_FIELDS = ('approach_visibility', 'tactile_coverage', 'blind_gap_m')     # part 2: the result's handoff keys
HANDOFF_WORDS = {'good': 'no blind gap', 'warning': 'short blind gap', 'serious': 'blind gap',
                 'critical': 'contact unsensed'}
OCC_NEAR = 0.005                 # m, the ray casting's start along each ray: core/occlusion.py's NEAR_M default


# ====================================================================== settings
def settings_of(project) -> dict:
    """The study's settings with the defaults filled in; 'defaults' lists the keys left at their default."""
    given = dict(project.analysis.studies.get(STUDY_ID) or {})
    out = dict(DEFAULTS)
    used = [k for k in DEFAULTS if k not in given]
    for k, v in given.items():
        out[k] = v
    for k, (lo, hi) in LIMITS.items():
        try:
            v = float(out[k])
        except (TypeError, ValueError):
            raise StudyError(f'The multi-finger occlusion setting {k} must be a number; got {out[k]!r}.') from None
        if not math.isfinite(v) or v < lo or v > hi:
            raise StudyError(f'The multi-finger occlusion setting {k} must be between {lo:g} and {hi:g}; got {v:g}.')
        out[k] = int(v) if k in INT_SETTINGS else v
    if out['backend'] not in BACKENDS:
        raise StudyError(f'The occlusion backend must be one of {", ".join(BACKENDS)}; got "{out["backend"]}".')
    if out['source'] not in ('synergy', 'grasp_set'):
        raise StudyError('The cases come from "synergy" (the grasp postures) or "grasp_set"; got '
                         f'"{out["source"]}".')
    if out['handoff_refine_mm'] > out['handoff_step_mm']:
        raise StudyError('The handoff\'s refine spacing must not be larger than its coarse step.')
    out['handoff'] = bool(out.get('handoff', True))
    if not (out['good_pct'] >= out['warn_pct'] >= out['serious_pct']):
        raise StudyError('The status thresholds must satisfy good_pct ≥ warn_pct ≥ serious_pct.')
    out['sensors'] = list(out.get('sensors') or [])
    out['defaults'] = used
    return out


def status_for(pct: float, st: dict) -> tuple[str, str]:
    if pct >= st['good_pct']:
        s = 'good'
    elif pct >= st['warn_pct']:
        s = 'warning'
    elif pct >= st['serious_pct']:
        s = 'serious'
    else:
        s = 'critical'
    return s, WORDS[s]


def hand_robot(project, st: dict):
    """The robot whose hand is studied: settings robot_uid, else the first robot with two or more fingertips."""
    uid = st.get('robot_uid')
    if uid:
        r = project.robot(uid)
        if r is None:
            raise StudyError('The robot named in the multi-finger occlusion settings is not in the project: choose '
                             'the hand\'s robot again.')
        return r
    r = next((x for x in project.robots if len(x.fingertip_frames) >= 2), None)
    if r is None:
        raise StudyError('No robot has two or more fingertip frames tagged: tag the hand\'s fingertip links on the '
                         'Robot page (or open the dexhand sample).')
    return r


# ====================================================================== the kernel
def _aabb_dist(o, lo, hi) -> float:
    d = np.maximum(np.maximum(lo - o, o - hi), 0.0)
    return float(np.linalg.norm(d))


def visibility_owner(points, origin, occluders, group_of, n_groups: int, n: int = 512, backend: str = 'auto',
                     eps: float | None = None, near: float | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """(visible (N,) bool, owner (N,) int: the group of the first hit, −1 where seen, info) for points seen from
    origin past occluders (Occluders or (M, 3, 3) world triangles) whose triangle k belongs to group group_of[k]
    (0 .. n_groups − 1). One cube map per group; d = min over the groups (module docstring)."""
    from .. import occlusion as OC
    eps = OC.EPS_M if eps is None else float(eps)
    near = OC.NEAR_M if near is None else float(near)
    tris = np.asarray(getattr(occluders, 'tris', occluders), np.float32).reshape(-1, 3, 3)
    group_of = np.asarray(group_of, int).reshape(-1)
    P = np.asarray(points, float).reshape(-1, 3)
    o = np.asarray(origin, float).reshape(3)
    v = P - o
    dist = np.linalg.norm(v, axis=1)
    best = np.full(len(P), np.inf)
    owner = np.full(len(P), -1, int)
    info = {'maps': 0, 'skipped_groups': 0, 'backend': '', 'renderer': '', 'seconds': 0.0}
    if not len(P):
        return np.zeros(0, bool), owner, info
    reach = float(dist.max()) + eps
    t0 = time.perf_counter()
    for g in range(int(n_groups)):
        sel = group_of == g
        if not sel.any():
            continue
        tg = tris[sel]
        V = tg.reshape(-1, 3).astype(float)
        if _aabb_dist(o, V.min(axis=0), V.max(axis=0)) > reach:
            info['skipped_groups'] += 1
            continue
        cube = OC.CubeMap.build(tg, o, n, backend, near)
        info['maps'] += 1
        info['backend'], info['renderer'] = cube.backend, cube.renderer
        d = cube.depth(v)
        closer = d < best
        best[closer] = d[closer]
        owner[closer] = g
    vis = dist <= best + eps
    owner[vis] = -1
    info['seconds'] = time.perf_counter() - t0
    return vis, owner, info


def visibility_owner_raycast(points, origin, occluders, group_of, n_groups: int, eps: float | None = None,
                             near: float | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """The same as visibility_owner by exact ray casting (core/raycast.py first_hit per group); the ray starts at
    the near distance along each ray (a sphere, not the cube map's per-face plane: the two differ only for occluders
    within a few millimetres of the centre)."""
    from .. import occlusion as OC
    from .. import raycast as RC
    eps = OC.EPS_M if eps is None else float(eps)
    near = OC.NEAR_M if near is None else float(near)
    tris = np.asarray(getattr(occluders, 'tris', occluders), np.float64).reshape(-1, 3, 3)
    group_of = np.asarray(group_of, int).reshape(-1)
    P = np.asarray(points, float).reshape(-1, 3)
    o = np.asarray(origin, float).reshape(3)
    v = P - o
    dist = np.linalg.norm(v, axis=1)
    d = v / np.maximum(dist, 1e-300)[:, None]
    best = np.full(len(P), np.inf)
    owner = np.full(len(P), -1, int)
    t0 = time.perf_counter()
    for g in range(int(n_groups)):
        sel = group_of == g
        if not sel.any() or not len(P):
            continue
        t = RC.first_hit(o + d * near, d, tris[sel]) + near
        closer = t < best
        best[closer] = t[closer]
        owner[closer] = g
    vis = dist <= best + eps
    owner[vis] = -1
    return vis, owner, {'maps': 0, 'backend': 'raycast', 'renderer': 'exact ray casting',
                        'seconds': time.perf_counter() - t0, 'skipped_groups': 0}


def patch_counts(in_view, visible, owner, n_groups: int) -> np.ndarray:
    """(n_groups + 2,) point counts of one patch from one camera: hidden by each group (in view, not seen, by the
    owner of the first hit), outside the image, seen (in view and nothing in front)."""
    iv = np.asarray(in_view, bool)
    vis = np.asarray(visible, bool)
    own = np.asarray(owner, int)
    out = np.zeros(int(n_groups) + 2, np.int64)
    hid = iv & ~vis
    if hid.any():
        out[:n_groups] = np.bincount(own[hid], minlength=n_groups)[:n_groups]
    out[n_groups] = int((~iv).sum())
    out[n_groups + 1] = int((iv & vis).sum())
    return out


def owner_groups(occ, hand, robot_uid: str) -> tuple[list, np.ndarray]:
    """([(key, title)], group index per occluder triangle): one group per finger, then palm, arm, objects, scene."""
    keys = [(f.name, f.title) for f in hand.fingers] + [(k, GROUP_TITLES[k]) for k in ('palm', 'arm', 'objects',
                                                                                         'scene')]
    index = {k: i for i, (k, _t) in enumerate(keys)}
    per_owner = np.empty(len(occ.owners), int)
    pre = f'robot:{robot_uid}:'
    for i, own in enumerate(occ.owners):
        if own.startswith(pre):
            per_owner[i] = index[hand.link_group.get(own[len(pre):], 'arm')]
        elif own.startswith('object:'):
            per_owner[i] = index['objects']
        else:
            per_owner[i] = index['scene']
    return keys, per_owner[occ.owner] if len(occ.owner) else np.zeros(0, int)


# ====================================================================== the cameras and the cases
def _cameras(project, st: dict, notes: list) -> list:
    from .. import library as LB
    from .coverage import pick_imager, sensor_mode
    sens = [s for s in project.variant.sensors if s.enabled and s.kind in ('camera', 'depth')
            and (not st['sensors'] or s.uid in st['sensors'])]
    if not sens:
        raise StudyError('Multi-finger occlusion needs at least one camera or depth sensor in use: add the '
                         'candidate mounts (Sensors ▸ Add camera), or tick "In use" on the Sensors and mounts page.')
    out = []
    for s in sens:
        try:
            ims = LB.sensor_imagers(s, sensor_mode(s, st))
        except LB.LibraryError as e:
            notes.append(f'{s.name} is left out: {e}')
            continue
        if not ims:
            notes.append(f'{s.name} has no imager and is left out.')
            continue
        im, why = pick_imager(s, ims, st)
        out.append({'sensor': s, 'imager': im, 'why': why})
    if not out:
        raise StudyError('None of the cameras could be modelled: ' + ' '.join(notes[-3:]))
    return out


def _mount_kind(project, s, hand) -> str:
    if not s.mount.on_link:
        return 'static'
    if s.mount.robot_uid != hand.robot_uid:
        return 'on another robot'
    g = hand.link_group.get(s.mount.parent_frame, 'arm')
    return {'palm': 'on the palm', 'arm': 'on the wrist or arm'}.get(g, 'on a finger')


def _joint_state(project, st: dict, notes: list) -> dict:
    from .coverage import joint_state
    qd, _src = joint_state(project, {'q': st.get('q') or {}}, notes)
    return qd


def _synergy_cases(model, syns: list, st: dict, base_q) -> list:
    from .. import dexhand as DH
    cases = [{'id': 'open', 'label': 'Open hand', 'synergy': '', 'closure': 0.0, 'grasp': '',
              'q': DH.posture(model, syns[0], 0.0, base_q) if syns else model.clip(base_q)}]
    n = st['steps']
    for s in syns:
        for k in range(1, n + 1):
            c = k / n
            cases.append({'id': f'{s.id}:{c:.3g}', 'label': f'{s.title} · {c * 100:.0f} % closed', 'synergy': s.id,
                          'closure': c, 'grasp': '', 'q': DH.posture(model, s, c, base_q)})
    return cases


def _grasp_cases(project, robot, model, syns: list, st: dict, base_q, notes: list, ctx) -> tuple[list, dict]:
    from .. import dexhand as DH
    from .. import grasps as GK
    from ..robot import ik
    gs = GK.get_set(project, st.get('grasp_set_uid') or '')
    if gs is None:
        raise StudyError('Choose a grasp set (Tasks page) for the grasp-set cases, or switch the cases to the grasp '
                         'postures.')
    if gs.get('robot_uid') and gs['robot_uid'] != robot.uid:
        raise StudyError(f'The grasp set "{gs.get("name", "")}" is for another robot; choose one made for '
                         f'{robot.name}.')
    poses = gs.get('poses') or []
    if not poses:
        raise StudyError(f'The grasp set "{gs.get("name", "")}" has no grasps.')
    syn = next((s for s in syns if s.id == st['grasp_synergy']), None)
    if syn is None:
        raise StudyError(f'There is no grasp posture "{st["grasp_synergy"]}"; the postures are '
                         f'{", ".join(s.id for s in syns)}.')
    try:
        link, tcp, T = GK.ik_targets(project, dict(gs, robot_uid=robot.uid))
    except GK.GraspError as e:
        raise StudyError(str(e)) from None
    idx = np.unique(np.linspace(0, len(poses) - 1, min(len(poses), st['max_grasps'])).round().astype(int))
    if len(idx) < len(poses):
        notes.append(f'{len(idx)} of the set\'s {len(poses)} grasps are taken, evenly spaced (max_grasps).')
    q_hand = DH.posture(model, syn, st['grasp_closure'], base_q)
    ctx.report(0.02, f'Solving IK for {len(idx)} grasps')
    res = ik.solve_batch(model, link, T[idx], q0=q_hand, seeds=st['ik_seeds'], tcp=tcp,
                         collision=bool(st['ik_collision']), cancel=ctx.cancelled)
    cases = []
    for k, i in enumerate(idx):
        if res.ok[k]:
            g = poses[int(i)]
            cases.append({'id': f'grasp:{g.get("id", i)}', 'label': f'Grasp {g.get("id", i)} ({g.get("kind", "")})',
                          'synergy': syn.id, 'closure': st['grasp_closure'], 'grasp': str(g.get('id', i)),
                          'q': res.q[k], 'approach': g.get('approach')})
    report = {'set_uid': gs.get('uid', ''), 'set_name': gs.get('name', ''), 'taken': int(len(idx)),
              'feasible': int(res.ok.sum()), 'counts': res.counts(), 'synergy': syn.id,
              'closure': st['grasp_closure'], 'collision_checked': bool(st['ik_collision'])}
    if not cases:
        raise StudyError(f'None of the {len(idx)} grasps could be reached by {robot.name} with the hand '
                         f'{syn.title.lower()}: ' + ', '.join(f'{v} {k}' for k, v in res.counts().items()
                                                              if k != 'ok') + '.')
    return cases, report


# ====================================================================== the study
def visibility_at(project, q: dict, settings: dict | None = None, *, hand=None, cams=None, notes=None) -> dict:
    """Part 1's kernel at one joint state q {robot uid: joint vector}: {'points': {finger: (K, 3)}, 'groups': keys,
    'cameras': [{'uid', 'R', 't', 'in_view' {finger: (K,)}, 'visible' {finger: (K,)}, 'owner' {finger: (K,)}}],
    'info'}. Part 2 (spec §7.15) calls it along each approach."""
    from .. import dexhand as DH
    from .. import grasps as GK
    from .. import scenegeom as SG
    from ..robot.model import RobotModelError
    st = dict(DEFAULTS, **(settings or {}))
    notes = notes if notes is not None else []
    robot = hand_robot(project, st)
    if hand is None:
        try:
            hand = DH.structure(project, robot.uid)
        except DH.HandError as e:
            raise StudyError(str(e)) from None
    cams = cams if cams is not None else _cameras(project, st, notes)
    occ = SG.scene_occluders(project, q, objects=bool(st['objects_occlude']))
    patches = GK.contact_patches(project, robot.uid, q.get(robot.uid))
    by_finger = {}
    for f in hand.fingers:
        pts = [v['points'] for v in patches.values() if hand.link_group.get(v['link']) == f.name]
        by_finger[f.name] = np.concatenate(pts) if pts else np.zeros((0, 3))
    names = [f.name for f in hand.fingers]
    P = np.concatenate([by_finger[n] for n in names]) if names else np.zeros((0, 3))
    cuts = np.cumsum([len(by_finger[n]) for n in names])[:-1]
    out = []
    info = {'maps': 0, 'backend': '', 'renderer': '', 'seconds': 0.0}
    for c in cams:
        s, im = c['sensor'], c['imager']
        try:
            T_wo = SG.sensor_world_pose(project, s, q).matrix() @ im.pose_in_body.matrix()
        except (RobotModelError, KeyError) as e:
            raise StudyError(f'{s.name} could not be posed: {e}') from None
        R, t = T_wo[:3, :3], T_wo[:3, 3]
        occ_c = SG.sensor_occluders(project, s, q, base=occ, origins=t[None])
        keys, gidx = owner_groups(occ_c, hand, robot.uid)
        if st['backend'] == 'raycast':
            vis, own, inf = visibility_owner_raycast(P, t, occ_c, gidx, len(keys), near=OCC_NEAR)
        else:
            from .. import occlusion as OC
            try:
                vis, own, inf = visibility_owner(P, t, occ_c, gidx, len(keys), st['cube_n'], st['backend'])
            except OC.OcclusionError as e:
                raise StudyError(f'The occlusion maps for {s.name} could not be built: {e}. Set the backend to '
                                 '"numpy" to use the software rasteriser.') from None
        _uv, inview = im.camera.project((P - t) @ R)
        info['maps'] += inf['maps']
        info['seconds'] += inf['seconds']
        info['backend'], info['renderer'] = inf['backend'] or info['backend'], inf['renderer'] or info['renderer']
        out.append({'uid': s.uid, 'R': R, 't': t,
                    'in_view': dict(zip(names, np.split(np.asarray(inview, bool), cuts))),
                    'visible': dict(zip(names, np.split(vis, cuts))),
                    'owner': dict(zip(names, np.split(own, cuts)))})
    keys = [(f.name, f.title) for f in hand.fingers] + [(k, GROUP_TITLES[k]) for k in ('palm', 'arm', 'objects',
                                                                                         'scene')]
    return {'points': by_finger, 'groups': keys, 'cameras': out, 'info': info, 'notes': list(occ.notes)}


def pack_bits(mask) -> str:
    """A boolean mask as base64 of numpy.packbits (big-endian bit order); unpack_bits(text, n) reverses it."""
    import base64
    return base64.b64encode(np.packbits(np.asarray(mask, bool)).tobytes()).decode('ascii')


def unpack_bits(text: str, n: int) -> np.ndarray:
    import base64
    raw = np.frombuffer(base64.b64decode(text or ''), np.uint8)
    return np.unpackbits(raw)[:int(n)].astype(bool)


def _r(x, nd=4):
    return None if x is None or not math.isfinite(float(x)) else round(float(x), nd)


def _rot(axis: str, deg: float) -> np.ndarray:
    """A turn of the optical frame: pitch about its x (right) axis, yaw about its y (down) axis."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == 'pitch':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


@jobs.study(STUDY_ID, 'Multi-finger occlusion', quick=False, batch='dex-occlusion', needs_robot=True)
def run(project, ctx: jobs.Ctx) -> dict:  # noqa: C901 — one pass over the cases, its figures spelled out
    """The finger × camera occlusion matrix and the mount comparison (module docstring)."""
    from .. import dexhand as DH
    from ..frames import Pose
    from ..robot.model import RobotModelError, get_model
    t_all = time.perf_counter()
    st = settings_of(project)
    notes: list = []
    robot = hand_robot(project, st)
    try:
        model = get_model(project, robot.uid)
    except RobotModelError as e:
        raise StudyError(str(e)) from None
    try:
        hand = DH.structure(project, robot.uid, model)
        syns = DH.synergies(project, robot.uid, st, model)
    except DH.HandError as e:
        raise StudyError(str(e)) from None
    if st.get('synergy_ids'):
        want = set(st['synergy_ids'])
        syns = [s for s in syns if s.id in want]
        if not syns:
            raise StudyError('None of the chosen grasp postures exists; choose at least one.')
    for s in syns:
        bad = DH.unknown_joints(model, s)
        if bad:
            notes.append(f'The posture "{s.title}" names joints {robot.name} does not have ({", ".join(bad[:4])}'
                         f'{"…" if len(bad) > 4 else ""}); they are ignored.')
    if any(s.sample for s in syns):
        notes.append('The grasp postures are sample postures (straight lines between two joint vectors), not '
                     'measured synergies: ' + '; '.join(f'{s.title}: {s.source}' for s in syns if s.sample) + '.')
    qd = _joint_state(project, st, notes)
    base_q = qd.get(robot.uid, model.home)
    grasp_report = None
    if st['source'] == 'grasp_set':
        cases, grasp_report = _grasp_cases(project, robot, model, syns, st, base_q, notes, ctx)
    else:
        if not syns:
            raise StudyError(f'{robot.name} has no grasp postures to sweep.')
        cases = _synergy_cases(model, syns, st, base_q)
    cams = _cameras(project, st, notes)
    fingers = hand.fingers
    F, C, K = len(fingers), len(cams), len(cases)
    groups = [(f.name, f.title) for f in fingers] + [(k, GROUP_TITLES[k]) for k in ('palm', 'arm', 'objects',
                                                                                     'scene')]
    G = len(groups)
    OUTI = G                                              # category index of "outside the image" (patch_counts)
    # per case: counts[k, c, f, category] (categories: groups, outside the image, seen)
    counts = np.zeros((K, C, F, G + 2), np.int64)
    npts = np.zeros(F, int)
    store = []                                            # per case: points, visible-by-occlusion, R, t (for re-aim)
    info_all = {'maps': 0, 'backend': '', 'renderer': '', 'seconds': 0.0}
    union_frac = np.zeros((K, F))
    cancelled = False
    occ_notes: list = []
    for k, case in enumerate(cases):
        if ctx.cancelled():
            cancelled = True
            break
        ctx.report(0.05 + 0.9 * k / max(K, 1), f'{case["label"]} ({k + 1} of {K})')
        q = dict(qd)
        q[robot.uid] = case['q']
        res = visibility_at(project, q, st, hand=hand, cams=cams)
        for n_ in res['notes']:
            if n_ not in occ_notes:
                occ_notes.append(n_)
        inf = res['info']
        info_all['maps'] += inf['maps']
        info_all['seconds'] += inf['seconds']
        info_all['backend'], info_all['renderer'] = inf['backend'] or info_all['backend'], \
            inf['renderer'] or info_all['renderer']
        rec = {'points': res['points'], 'cams': [], 'seen_bits': []}
        any_seen = {f.name: np.zeros(len(res['points'][f.name]), bool) for f in fingers}
        for j, cr in enumerate(res['cameras']):
            rec['cams'].append({'R': cr['R'], 't': cr['t'], 'visible': cr['visible']})
            rec['seen_bits'].append(pack_bits(np.concatenate([cr['in_view'][f.name] & cr['visible'][f.name]
                                                              for f in fingers])))
            for i, f in enumerate(fingers):
                iv, vis, own = cr['in_view'][f.name], cr['visible'][f.name], cr['owner'][f.name]
                npts[i] = len(iv)
                any_seen[f.name] |= iv & vis
                counts[k, j, i] = patch_counts(iv, vis, own, G)
        for i, f in enumerate(fingers):
            union_frac[k, i] = any_seen[f.name].mean() if len(any_seen[f.name]) else 0.0
        store.append(rec)
        if ctx.partial:
            ctx.partial({'study': STUDY_ID, 'partial': True, 'cases_done': k + 1, 'cases': K})
    Kd = len(store)
    if Kd == 0:
        raise StudyError('The study was stopped before the first case.') if cancelled else \
            StudyError('There were no cases to study.')
    counts = counts[:Kd]
    union_frac = union_frac[:Kd]
    if (npts == 0).any():
        miss = [f.title for f, n_ in zip(fingers, npts) if n_ == 0]
        raise StudyError(f'{", ".join(miss)} {"has" if len(miss) == 1 else "have"} no contact patch: tag contact pads '
                         'or check the fingertip links on the Robot page.')
    frac = counts / npts[None, None, :, None]            # (K, C, F, G + 2)
    seen = frac[..., G + 1]                               # (K, C, F)
    thr = st['min_visible_pct'] / 100.0
    rig_ok = union_frac >= thr - 1e-12                    # (K, F)
    rig_pct = float(rig_ok.mean() * 100)
    status, word = status_for(rig_pct, st)

    # ---------------------------------------------------------------- the matrix
    mean_frac = frac.mean(axis=0)                         # (C, F, G + 2)
    not_seen = (1 - mean_frac[..., G + 1]) * 100          # (C, F)
    by_owner = mean_frac[..., :G + 1] * 100               # (C, F, G + 1) incl. outside the image
    cat_keys = [g[0] for g in groups] + [OUT]
    dominant = [[cat_keys[int(np.argmax(by_owner[j, i]))] if not_seen[j, i] > 0.05 else ''
                 for j in range(C)] for i in range(F)]
    # ---------------------------------------------------------------- per camera
    cam_rows = []
    for j, c in enumerate(cams):
        s, im = c['sensor'], c['imager']
        ok_cf = seen[:, j, :] >= thr - 1e-12               # (K, F)
        hidden_by = by_owner[j, :, :G].sum(axis=0)        # (G,) summed over fingers
        main = int(np.argmax(hidden_by)) if hidden_by.max() > 0 else -1
        worst = int(np.argmin(seen[:, j, :].mean(axis=0)))
        cam_rows.append({
            'uid': s.uid, 'name': s.name, 'device': s.model_id, 'imager': im.name, 'imager_why': im.label or c['why'],
            'mount': _mount_kind(project, s, hand), 'parent_frame': s.mount.parent_frame, 'locked': bool(s.locked),
            'seen_pct': _r(seen[:, j, :].mean() * 100, 2),
            'fingertips_seen_pct': _r(ok_cf.mean() * 100, 2),
            'all_fingertips_rate_pct': _r(ok_cf.all(axis=1).mean() * 100, 2),
            'outside_image_pct': _r(mean_frac[j, :, OUTI].mean() * 100, 2),
            'hidden_pct': _r(mean_frac[j, :, :G].sum(axis=1).mean() * 100, 2),
            'main_occluder': groups[main][0] if main >= 0 else '',
            'main_occluder_pct': _r(hidden_by[main] / F, 2) if main >= 0 else 0.0,
            'worst_finger': fingers[worst].name, 'worst_finger_seen_pct': _r(seen[:, j, worst].mean() * 100, 2),
            'per_finger_seen_pct': [_r(x * 100, 2) for x in seen[:, j, :].mean(axis=0)],
        })
    ranking = sorted(range(C), key=lambda j: (-(cam_rows[j]['fingertips_seen_pct'] or 0),
                                              -(cam_rows[j]['seen_pct'] or 0)))
    for r_, j in enumerate(ranking):
        cam_rows[j]['rank'] = r_ + 1

    # ---------------------------------------------------------------- what to move (re-aim about the optical centre)
    suggestion = None
    tried = 0
    if st['suggest_deg'] > 0 and not cancelled:
        ctx.report(0.96, 'Trying a turn of each camera')
        names = [f.name for f in fingers]
        base_union = union_frac.mean()

        seen_pts = [[{n_: np.asarray(cams[jj]['imager'].camera.project((rec['points'][n_] - cr['t']) @ cr['R'])[1],
                                     bool) & cr['visible'][n_] for n_ in names}
                     for jj, cr in enumerate(rec['cams'])] for rec in store]

        def rig_with(j, Rd):
            un = np.zeros((Kd, F))
            for k, rec in enumerate(store):
                cr = rec['cams'][j]
                for i, n_ in enumerate(names):
                    P = rec['points'][n_]
                    iv = np.asarray(cams[j]['imager'].camera.project((P - cr['t']) @ (cr['R'] @ Rd))[1], bool)
                    anyv = iv & cr['visible'][n_]
                    for jj in range(C):
                        if jj != j:
                            anyv = anyv | seen_pts[k][jj][n_]
                    un[k, i] = anyv.mean()
            return float((un >= thr - 1e-12).mean() * 100), float(un.mean())

        best = None
        for j, c in enumerate(cams):
            if c['sensor'].locked:
                continue
            for axis in ('pitch', 'yaw'):
                for sgn in (1.0, -1.0):
                    deg = sgn * st['suggest_deg']
                    pct, un = rig_with(j, _rot(axis, deg))
                    tried += 1
                    gain = (pct - rig_pct, un - base_union)
                    good = gain[0] > 1e-9 or (abs(gain[0]) <= 1e-9 and gain[1] >= 0.005)
                    if good and (best is None or gain > best[0]):
                        best = (gain, j, axis, deg, pct, un)
        if best is not None:
            gain, j, axis, deg, pct, un = best
            s, im = cams[j]['sensor'], cams[j]['imager']
            Tb = s.mount.pose.matrix()
            To = im.pose_in_body.matrix()
            D = np.eye(4)
            D[:3, :3] = _rot(axis, deg)
            newp = Pose.from_matrix(Tb @ To @ D @ np.linalg.inv(To))
            word_dir = {('pitch', 1): 'down', ('pitch', -1): 'up', ('yaw', 1): 'right', ('yaw', -1): 'left'}
            dir_ = word_dir[(axis, 1 if deg > 0 else -1)]
            suggestion = {
                'sensor_uid': s.uid, 'name': s.name, 'axis': axis, 'deg': deg, 'direction': dir_,
                'from_pct': _r(rig_pct, 2), 'to_pct': _r(pct, 2),
                'from_union_seen_pct': _r(base_union * 100, 2), 'to_union_seen_pct': _r(un * 100, 2),
                'pose': newp.to_dict(), 'parent_frame': s.mount.parent_frame,
                'text': (f'Turn {s.name} {abs(deg):g}° {dir_} about its optical centre: '
                         + (f'the rig sees {pct:.1f} % of fingertip-cases instead of {rig_pct:.1f} % (patch seen by '
                            f'any camera {un * 100:.1f} % instead of {base_union * 100:.1f} %).'
                            if pct > rig_pct + 1e-9 else
                            f'the share of patch points some camera sees rises from {base_union * 100:.1f} % to '
                            f'{un * 100:.1f} % (fingertip-cases seen stay at {rig_pct:.1f} %).')),
            }

    # ---------------------------------------------------------------- part 2: the visual-to-tactile handoff
    handoff = None
    if st['handoff'] and not cancelled:
        ctx.report(0.97, 'The visual-to-tactile handoff')
        try:
            handoff = _handoff(project, robot, model, hand, cams, st, qd, cases[:Kd], ctx)
        except StudyError as e:
            handoff = {'error': str(e)}
            notes.append(f'The visual-to-tactile handoff could not run: {e}')

    # ---------------------------------------------------------------- the result
    best_cam = cam_rows[ranking[0]]
    worst_i = int(np.argmin(union_frac.mean(axis=0)))
    if suggestion:
        advice = suggestion['text']
    elif rig_pct >= st['good_pct']:
        advice = f'The rig sees the fingertips in {rig_pct:.1f} % of the cases; nothing to move within ±' \
                 f'{st["suggest_deg"]:g}°.'
    else:
        top = best_cam['main_occluder']
        top_t = next((t for k_, t in groups if k_ == top), top)
        if top_t and top in {f.name for f in fingers}:
            named = any(w in top_t.lower() for w in ('finger', 'thumb', 'jaw', 'pad', 'tip'))
            top_t = f'the {top_t.lower()}' if named else f'the {top_t} finger'
        advice = (f'{fingers[worst_i].title} is seen least ({union_frac[:, worst_i].mean() * 100:.1f} % of its patch '
                  f'by any camera). The best single mount is {best_cam["name"]} ({best_cam["fingertips_seen_pct"]:.1f} '
                  f'% of fingertip-cases); what hides most from it is {top_t.lower() if top_t else "nothing"}: add a '
                  'camera that looks into the palm from the side the pads face.')
    per_case = []
    for k in range(Kd):
        c_ = cases[k]
        per_case.append({
            'id': c_['id'], 'label': c_['label'], 'synergy': c_['synergy'], 'closure': _r(c_['closure'], 4),
            'grasp': c_['grasp'],
            'seen_pct': [[_r(seen[k, j, i] * 100, 2) for i in range(F)] for j in range(C)],
            'dominant': [[(cat_keys[int(np.argmax(frac[k, j, i, :G + 1]))] if seen[k, j, i] < 1 - 1e-12 else '')
                          for i in range(F)] for j in range(C)],
            'rig_seen_pct': [_r(union_frac[k, i] * 100, 2) for i in range(F)],
            'seen_bits': store[k]['seen_bits'],
            'q': [_r(x, 6) for x in np.asarray(c_['q'], float)],
        })
    notes += occ_notes
    if not st['objects_occlude'] and project.workspace.objects:
        notes.append('Task objects are not occluders here (objects_occlude off, the default): a fingertip that touches '
                     'an object is hidden by it, and this study measures what the hand hides.')
    doc = {
        'robot': {'uid': robot.uid, 'name': robot.name},
        'hand': hand.to_dict(),
        'fingers': [{'name': f.name, 'title': f.title, 'tip': f.tip, 'points': int(npts[i])}
                    for i, f in enumerate(fingers)],
        'groups': [{'key': k_, 'title': t} for k_, t in groups] + [{'key': OUT, 'title': GROUP_TITLES[OUT]}],
        'synergies': [s.to_dict() for s in syns],
        'source': st['source'],
        'grasps': grasp_report,
        'cases': [{'id': c_['id'], 'label': c_['label'], 'synergy': c_['synergy'], 'closure': _r(c_['closure'], 4),
                   'grasp': c_['grasp']} for c_ in cases[:Kd]],
        'cameras': cam_rows,
        'ranking': [cam_rows[j]['uid'] for j in ranking],
        'matrix': {
            'rows': [f.name for f in fingers], 'cols': [c['sensor'].uid for c in cams],
            'not_seen_pct': [[_r(not_seen[j, i], 2) for j in range(C)] for i in range(F)],
            'by_owner_pct': [[[_r(by_owner[j, i, g], 2) for g in range(G + 1)] for j in range(C)] for i in range(F)],
            'dominant': dominant,
            'categories': cat_keys,
        },
        'per_case': per_case,
        'rig': {'fingertips_seen_pct': _r(rig_pct, 2), 'status': status, 'status_word': word,
                'all_fingertips_rate_pct': _r(rig_ok.all(axis=1).mean() * 100, 2),
                'patch_seen_pct': _r(union_frac.mean() * 100, 2),
                'per_finger_seen_pct': [_r(x * 100, 2) for x in union_frac.mean(axis=0)],
                'worst_finger': fingers[worst_i].name},
        'advice': advice,
        'suggestion': suggestion,
        'handoff': handoff,                               # part 2 (spec §7.15): HANDOFF_FIELDS, or None
        'settings': {k: v for k, v in st.items() if k not in ('synergies', 'q')},
        'q': {uid: [_r(x, 6) for x in np.asarray(v, float)] for uid, v in qd.items()},
        'notes': notes,
        'occlusion': {'backend': info_all['backend'], 'renderer': info_all['renderer'], 'maps': info_all['maps'],
                      'cube_n': st['cube_n'], 'seconds': _r(info_all['seconds'], 3)},
        'timing': {'cases': Kd, 'cameras': C, 'patch_points': int(npts.sum()), 'turns_tried': tried,
                   'seconds': _r(time.perf_counter() - t_all, 3),
                   'cases_per_s': _r(Kd / max(time.perf_counter() - t_all, 1e-9), 3)},
    }
    ctx.report(1.0, 'Done')
    return doc


# ====================================================================== part 2: the visual-to-tactile handoff
def _identity(pose) -> bool:
    q = np.asarray(pose.quaternion, float)
    return float(np.linalg.norm(pose.position)) < 1e-9 and abs(abs(q[3]) - 1.0) < 1e-9


def tactile_pads(project, robot, hand, q, notes: list | None = None) -> list:
    """The tactile sensors in use on the hand's links at joint state q {robot uid: q}: [{'uid', 'name', 'model_id',
    'finger', 'link', 'pad_mm' (from the library entry, or None: not stated), 'placement', 'center', 'normal', 'u',
    'v'}] (world; u along the pad's first side, v along its second). A pad at its link's origin (the mount pose left
    at identity, as Library ▸ add puts it) is taken to sit centred on that finger's contact patch, facing along the
    patch's mean normal, its longer side along the patch's longer extent (an assumption, labelled); a placed pad
    senses on its mount's body x axis (REP-103 forward), its first side along body y, its second along body z."""
    from .. import grasps as GK
    from .. import library as LB
    from .. import scenegeom as SG
    notes = notes if notes is not None else []
    out = []
    patches = None
    for s in project.variant.sensors:
        if s.kind != 'tactile' or not s.enabled or s.mount.robot_uid != robot.uid:
            continue
        link = s.mount.parent_frame
        finger = hand.link_group.get(link)
        e = LB.get(s.model_id) if s.model_id else None
        pad = ((e.data.get('tactile') or {}).get('pad_mm') if e is not None else None)
        rec = {'uid': s.uid, 'name': s.name, 'model_id': s.model_id, 'finger': finger or '', 'link': link,
               'pad_mm': [float(x) for x in pad] if pad else None,
               'pad_source': (f'{e.title} library entry' if pad else
                              ('not stated by the maker' if e is not None else 'no device chosen'))}
        if finger is None or finger in ('palm', 'arm'):
            rec['placement'] = f'on {link}, not a fingertip\'s finger: it covers no fingertip patch here'
            out.append(rec)
            continue
        if _identity(s.mount.pose):
            if patches is None:
                patches = GK.contact_patches(project, robot.uid, q.get(robot.uid))
            pts = [v for v in patches.values() if hand.link_group.get(v['link']) == finger]
            if not pts:
                rec['placement'] = 'its finger has no contact patch'
                out.append(rec)
                continue
            P = np.concatenate([v['points'] for v in pts])
            N = np.concatenate([v['normals'] for v in pts])
            c = P.mean(axis=0)
            n = N.mean(axis=0)
            n = n / max(float(np.linalg.norm(n)), 1e-300)
            Q = (P - c) - np.outer((P - c) @ n, n)
            _w, V = np.linalg.eigh(Q.T @ Q)
            u = V[:, 2] - (V[:, 2] @ n) * n
            u = u / max(float(np.linalg.norm(u)), 1e-300)
            v = np.cross(n, u)
            if rec['pad_mm'] and rec['pad_mm'][1] > rec['pad_mm'][0]:
                u, v = v, u                              # the pad's longer side along the patch's longer extent
            rec.update(placement='assumed: centred on the finger\'s contact patch (the mount pose is not set)',
                       center=c, normal=n, u=u, v=v)
        else:
            T = SG.sensor_world_pose(project, s, q).matrix()
            rec.update(placement='at its mount pose (senses along body x)', center=T[:3, 3], normal=T[:3, 0],
                       u=T[:3, 1], v=T[:3, 2])
        out.append(rec)
    return out


def pad_coverage(points, pad: dict, gap_m: float) -> np.ndarray:
    """(K,) True for the patch points on the pad: within gap_m of its plane and inside its w × h rectangle."""
    P = np.asarray(points, float).reshape(-1, 3)
    if pad.get('pad_mm') is None or 'center' not in pad:
        return np.zeros(len(P), bool)
    d = P - pad['center']
    w, h = (x / 1000.0 for x in pad['pad_mm'])
    return (np.abs(d @ pad['normal']) <= gap_m + 1e-12) & (np.abs(d @ pad['u']) <= w / 2 + 1e-12) & \
        (np.abs(d @ pad['v']) <= h / 2 + 1e-12)


def gap_from_samples(dist, seen) -> tuple[float, bool, float]:
    """(blind gap m, censored, total hidden m) of one fingertip along an approach from (distance, seen) samples:
    the gap runs from contact to the midpoint between the last distance not seen and the first seen (0 when seen at
    contact; the whole scanned distance, censored, when never seen); the hidden total counts every stretch not seen,
    a change of state at the midpoint between two samples."""
    d = np.asarray(dist, float)
    s = np.asarray(seen, bool)
    order = np.argsort(d)
    d, s = d[order], s[order]
    if not len(d):
        return 0.0, True, 0.0
    if s[0] and d[0] <= 1e-12:
        gap, cens = 0.0, False
    elif s.any():
        i = int(np.argmax(s))
        gap, cens = 0.5 * (d[i - 1] + d[i]) if i > 0 else d[0], False
    else:
        gap, cens = float(d[-1]), True
    hid = 0.0
    for k in range(len(d) - 1):
        a, b = (not s[k]), (not s[k + 1])
        hid += (d[k + 1] - d[k]) * (1.0 if a and b else (0.5 if a or b else 0.0))
    return float(gap), cens, float(hid)


def _handoff_cases(cases: list, st: dict) -> list:
    """The contact cases the approaches end at: grasp cases, or each posture fully closed (its last case)."""
    if st['source'] == 'grasp_set':
        pick = [c for c in cases if c.get('grasp')]
    else:
        last = {}
        for c in cases:
            if c['synergy']:
                if c['synergy'] not in last or c['closure'] >= last[c['synergy']]['closure']:
                    last[c['synergy']] = c
        pick = list(last.values())
    if len(pick) > st['handoff_max_cases']:
        idx = np.unique(np.linspace(0, len(pick) - 1, st['handoff_max_cases']).round().astype(int))
        pick = [pick[i] for i in idx]
    return pick


class _Approach:
    """Visibility along the straight approaches into a set of contact cases: the gripper frame backed off along
    −approach by a distance, the arm solved by IK from the nearer pose ('nearest_q0', so it keeps its branch; the hand
    keeps the case's posture), and the rig's union of what each camera sees of every fingertip's patch (part 1's
    kernel, visibility_at)."""

    def __init__(self, project, robot, model, hand, cams, st, qd, cases):
        from .. import grasps as GK
        self.p, self.robot, self.model, self.hand, self.cams, self.st, self.qd = (project, robot, model, hand, cams,
                                                                                  st, qd)
        geom = GK.gripper_geometry(project, robot.uid)
        self.link = geom['frame']
        self.cases = cases
        self.T0, self.a = [], []
        for c in cases:
            T = model.link_pose(c['q'], self.link)
            a = np.asarray(c.get('approach') if c.get('approach') is not None else T[:3, :3] @ geom['approach'],
                           float)
            self.T0.append(T)
            self.a.append(a / max(float(np.linalg.norm(a)), 1e-300))
        self.evals = 0
        self.ik_failed = 0
        self.names = [f.name for f in hand.fingers]
        self.vst = dict(st, objects_occlude=bool(st.get('handoff_objects_occlude', True)))

    def solve(self, idx, dists, q0) -> tuple[np.ndarray, np.ndarray]:
        from ..robot import ik
        T = np.array([self.T0[i].copy() for i in idx])
        for k, (i, d) in enumerate(zip(idx, dists)):
            T[k, :3, 3] = self.T0[i][:3, 3] - float(d) * self.a[i]
        res = ik.solve_batch(self.model, self.link, T, q0=np.asarray(q0, float), seeds=1, rule='nearest_q0',
                             collision=False, retry_seeds=0)
        return np.asarray(res.q, float), np.asarray(res.ok, bool)

    def seen(self, q) -> np.ndarray:
        """(F,) share of each fingertip's patch some camera sees at the robot's joint vector q."""
        qq = dict(self.qd)
        qq[self.robot.uid] = q
        res = visibility_at(self.p, qq, self.vst, hand=self.hand, cams=self.cams)
        self.evals += 1
        out = np.zeros(len(self.names))
        for i, n in enumerate(self.names):
            m = np.zeros(len(res['points'][n]), bool)
            for c in res['cameras']:
                m |= c['in_view'][n] & c['visible'][n]
            out[i] = m.mean() if len(m) else 0.0
        return out

    def scan(self, dists_per_case, cancel=None) -> list:
        """For each case: {distance (m): (F,) seen share or None where IK failed}, distances solved in order from
        contact, each from the solution at the previous distance."""
        out = []
        for i, c in enumerate(self.cases):
            rec, q_prev = {}, np.asarray(c['q'], float)
            for d in sorted(dists_per_case[i]):
                if cancel is not None and cancel():
                    break
                if d <= 1e-12:
                    rec[0.0] = self.seen(q_prev)
                    continue
                q, ok = self.solve([i], [d], q_prev[None])
                if not ok[0]:
                    self.ik_failed += 1
                    rec[float(d)] = None
                    continue
                q_prev = q[0]
                rec[float(d)] = self.seen(q_prev)
            out.append(rec)
        return out

    def adaptive(self, cancel=None) -> list:
        """Coarse distances, then the refine spacing inside every coarse interval where a fingertip's seen state
        changes (module docstring of part 2)."""
        st = self.st
        dmax, step, ref = (st[k] / 1000.0 for k in ('handoff_max_mm', 'handoff_step_mm', 'handoff_refine_mm'))
        coarse = sorted(set(np.round(np.r_[np.arange(0.0, dmax + 1e-12, step), dmax], 9)))
        thr = st['min_visible_pct'] / 100.0
        recs = self.scan([coarse] * len(self.cases), cancel)
        for i, rec in enumerate(recs):
            ds = sorted(d for d, v in rec.items() if v is not None)
            extra = []
            for a, b in zip(ds[:-1], ds[1:]):
                if np.any((rec[a] >= thr - 1e-12) != (rec[b] >= thr - 1e-12)):
                    n = int(round((b - a) / ref))
                    extra += [round(a + k * (b - a) / n, 9) for k in range(1, n)]
            if extra:
                more = self.scan_one(i, rec, extra)
                rec.update(more)
        return recs

    def scan_one(self, i, rec, extra) -> dict:
        """Distances inside known intervals, each solved from the nearest solved distance below it."""
        out = {}
        solved = {}
        for d in sorted(extra):
            below = max([x for x in list(rec) + list(out) if x < d and (rec.get(x) if x in rec else out.get(x))
                         is not None], default=None)
            q0 = solved.get(below)
            if q0 is None:
                q0 = self._q_at(i, below)
            q, ok = self.solve([i], [d], q0[None])
            if not ok[0]:
                self.ik_failed += 1
                out[d] = None
                continue
            solved[d] = q[0]
            out[d] = self.seen(q[0])
        return out

    def _q_at(self, i, d):
        q = np.asarray(self.cases[i]['q'], float)
        if d is None or d <= 1e-12:
            return q
        sol, ok = self.solve([i], [d], q[None])
        return sol[0] if ok[0] else q


def _handoff(project, robot, model, hand, cams, st, qd, cases, ctx=None) -> dict:  # noqa: C901 — the gap, spelled out
    t0 = time.perf_counter()
    hc = _handoff_cases(cases, st)
    if not hc:
        raise StudyError('There is no contact case to approach (a fully closed posture or a reached grasp).')
    ap = _Approach(project, robot, model, hand, cams, st, qd, hc)
    recs = ap.adaptive(ctx.cancelled if ctx is not None else None)
    thr = st['min_visible_pct'] / 100.0
    touch_thr = st['min_touch_pct'] / 100.0
    gap_m = st['pad_gap_mm'] / 1000.0
    notes = []
    approaches, worst = [], None
    cov_rows: dict = {}
    for i, (c, rec) in enumerate(zip(hc, recs)):
        q = dict(qd)
        q[robot.uid] = c['q']
        pads = tactile_pads(project, robot, hand, q, notes)
        from .. import grasps as GK
        patches = GK.contact_patches(project, robot.uid, c['q'])
        ds = sorted(rec)
        ok_ds = [d for d in ds if rec[d] is not None]
        fingers = []
        for j, f in enumerate(hand.fingers):
            P = [v['points'] for v in patches.values() if hand.link_group.get(v['link']) == f.name]
            P = np.concatenate(P) if P else np.zeros((0, 3))
            on = np.zeros(len(P), bool)
            mine = [p for p in pads if p.get('finger') == f.name]
            per_pad = []
            for p in mine:
                m = pad_coverage(P, p, gap_m)
                on |= m
                per_pad.append({'uid': p['uid'], 'name': p['name'], 'model_id': p['model_id'], 'pad_mm': p['pad_mm'],
                                'pad_source': p['pad_source'], 'placement': p.get('placement', ''),
                                'coverage_pct': _r(m.mean() * 100 if len(m) else 0.0, 2)})
            cov = float(on.mean()) if len(on) else 0.0
            touched = bool(mine) and cov >= touch_thr - 1e-12
            if not mine:
                why = 'no tactile pad on this finger'
            elif all(p['pad_mm'] is None for p in mine):
                why = 'the pad size is not stated, so its coverage is not known'
            else:
                why = '' if touched else (f'the pads cover {cov * 100:.0f} % of the patch (below '
                                          f'{st["min_touch_pct"]:g} %)')
            cov_rows.setdefault(f.name, []).append({'case': c['id'], 'coverage_pct': _r(cov * 100, 2),
                                                    'touched': touched, 'why': why, 'pads': per_pad})
            seen = [bool(rec[d][j] >= thr - 1e-12) for d in ok_ds]
            gap, cens, hid = gap_from_samples(ok_ds, seen)
            seen0 = bool(ok_ds and ok_ds[0] <= 1e-12 and seen[0])
            sensed = seen0 or touched
            flagged = (gap > st['blind_gap_max_mm'] / 1000.0 + 1e-12) or not sensed
            row = {'name': f.name, 'title': f.title, 'blind_gap_m': _r(gap, 5), 'censored': cens,
                   'hidden_m': _r(hid, 5), 'seen_at_contact': seen0, 'touched': touched,
                   'coverage_pct': _r(cov * 100, 2), 'sensed_at_contact': sensed, 'flagged': flagged}
            fingers.append(row)
            key = (not sensed, gap)
            if worst is None or key > worst[0]:
                worst = (key, c, row)
        approaches.append({'case': c['id'], 'label': c['label'], 'distances_m': [_r(d, 6) for d in ds],
                           'ik_ok': [rec[d] is not None for d in ds],
                           'seen_pct': [None if rec[d] is None else [_r(x * 100, 2) for x in rec[d]] for d in ds],
                           'approach': [_r(x, 6) for x in ap.a[i]], 'fingers': fingers})
    gaps = [r['blind_gap_m'] for a in approaches for r in a['fingers']]
    worst_gap = max(gaps) if gaps else 0.0
    thr_m = st['blind_gap_max_mm'] / 1000.0
    unsensed = [(a, r) for a in approaches for r in a['fingers'] if not r['sensed_at_contact']]
    flagged = [(a, r) for a in approaches for r in a['fingers'] if r['flagged']]
    if unsensed:
        status = 'critical'
    elif worst_gap > 2 * thr_m + 1e-12:
        status = 'serious'
    elif flagged:
        status = 'warning'
    else:
        status = 'good'
    flags = []
    for a, r in flagged:
        what = (f'{r["title"]} in "{a["label"]}": ' + (f'not seen for the last {r["blind_gap_m"] * 1000:.0f} mm'
                                                       + (' or more' if r['censored'] else '') + ' of the approach'
                                                       if r['blind_gap_m'] > 0 else 'seen up to contact'))
        if not r['sensed_at_contact']:
            what += ', and at contact neither seen nor touched'
        flags.append(what + '.')
    if not any(p for rows in cov_rows.values() for x in rows for p in x['pads']):
        notes.append('No tactile sensor is on the hand\'s fingers: add one from the Library (Tactile sensors) and '
                     'mount it on a fingertip link; until then nothing is touched at contact.')
    (_k, wc, wr) = worst
    text = (f'Worst: {wr["title"]} in "{wc["label"]}", '
            + (f'blind for {wr["blind_gap_m"] * 1000:.0f} mm' + (' or more' if wr['censored'] else '')
               + ' before contact' if wr['blind_gap_m'] > 0 else 'seen up to contact')
            + ('; at contact it is neither seen nor touched.' if not wr['sensed_at_contact'] else '.'))
    if ap.ik_failed:
        notes.append(f'{ap.ik_failed} approach distances could not be reached by IK and are left out.')
    return {
        'approach_visibility': approaches,
        'tactile_coverage': [{'name': f.name, 'title': f.title, 'cases': cov_rows.get(f.name, [])}
                             for f in hand.fingers],
        'blind_gap_m': _r(worst_gap, 5),
        'worst': {'case': wc['id'], 'finger': wr['name']},
        'flags': flags, 'status': status, 'status_word': HANDOFF_WORDS[status], 'text': text,
        'threshold_m': _r(thr_m, 6), 'notes': notes,
        'settings': {k: st[k] for k in ('handoff_max_mm', 'handoff_step_mm', 'handoff_refine_mm', 'handoff_max_cases',
                                        'handoff_objects_occlude', 'min_visible_pct', 'min_touch_pct',
                                        'pad_gap_mm', 'blind_gap_max_mm')},
        'defaults': [k for k in st.get('defaults', []) if k.startswith('handoff') or k in (
            'min_touch_pct', 'pad_gap_mm', 'blind_gap_max_mm')],
        'evaluations': ap.evals, 'seconds': _r(time.perf_counter() - t0, 3),
    }


def _setup(project, st, notes):
    """(robot, model, hand, synergies, cases, cameras, joint state) as run() builds them."""
    from .. import dexhand as DH
    from ..robot.model import RobotModelError, get_model
    robot = hand_robot(project, st)
    try:
        model = get_model(project, robot.uid)
        hand = DH.structure(project, robot.uid, model)
        syns = DH.synergies(project, robot.uid, st, model)
    except (RobotModelError, DH.HandError) as e:
        raise StudyError(str(e)) from None
    if st.get('synergy_ids'):
        syns = [s for s in syns if s.id in set(st['synergy_ids'])] or syns
    qd = _joint_state(project, st, notes)
    base_q = qd.get(robot.uid, model.home)
    if st['source'] == 'grasp_set':
        cases, _rep = _grasp_cases(project, robot, model, syns, st, base_q, notes, jobs.Ctx())
    else:
        if not syns:
            raise StudyError(f'{robot.name} has no grasp postures to approach with.')
        cases = _synergy_cases(model, syns, st, base_q)
    return robot, model, hand, cases, _cameras(project, st, notes), qd


def tactile_handoff(project, ctx: jobs.Ctx | None = None) -> dict:
    """Spec §7.15: the handoff part of the study on its own (the same document run() puts under 'handoff'): along
    each approach (the gripper frame backed off along −approach, solved by IK) the share of each fingertip's contact
    patch the rig sees against the distance to contact, the tactile pads' coverage of the patch at contact, and the
    sensory blind gap (module docstring)."""
    st = settings_of(project)
    notes: list = []
    robot, model, hand, cases, cams, qd = _setup(project, st, notes)
    out = _handoff(project, robot, model, hand, cams, st, qd, cases, ctx)
    out['notes'] = notes + out['notes']
    return out


def blind_gap_check(project) -> dict:
    """For validation (checks_chunk.py dexhand.blind_gap_bruteforce): the first approach's gaps from the adaptive
    scan against a scan of every distance at the refine spacing."""
    st = settings_of(project)
    notes: list = []
    robot, model, hand, cases, cams, qd = _setup(project, st, notes)
    hc = _handoff_cases(cases, st)[:1]
    thr = st['min_visible_pct'] / 100.0
    a = _Approach(project, robot, model, hand, cams, st, qd, hc)
    rec_a = a.adaptive()[0]
    b = _Approach(project, robot, model, hand, cams, st, qd, hc)
    ref = st['handoff_refine_mm'] / 1000.0
    dense = sorted(set(np.round(np.r_[np.arange(0.0, st['handoff_max_mm'] / 1000.0 + 1e-12, ref),
                                      st['handoff_max_mm'] / 1000.0], 9)))
    rec_b = b.scan([dense])[0]
    out_a, out_b = [], []
    for j, _f in enumerate(hand.fingers):
        for rec, out in ((rec_a, out_a), (rec_b, out_b)):
            ds = [d for d in sorted(rec) if rec[d] is not None]
            out.append(gap_from_samples(ds, [rec[d][j] >= thr - 1e-12 for d in ds]))
    err = max(max(abs(x[0] - y[0]), abs(x[2] - y[2])) for x, y in zip(out_a, out_b))
    return {'case': hc[0]['id'], 'fingers': [f.name for f in hand.fingers], 'adaptive_m': [x[0] for x in out_a],
            'brute_m': [x[0] for x in out_b], 'adaptive_hidden_m': [x[2] for x in out_a],
            'brute_hidden_m': [x[2] for x in out_b], 'error_m': float(err), 'refine_m': ref,
            'evaluations_adaptive': a.evals, 'evaluations_brute': b.evals,
            'transitions': int(sum(1 for j in range(len(hand.fingers)) for d1, d2 in zip(
                [d for d in sorted(rec_b) if rec_b[d] is not None][:-1],
                [d for d in sorted(rec_b) if rec_b[d] is not None][1:])
                if (rec_b[d1][j] >= thr - 1e-12) != (rec_b[d2][j] >= thr - 1e-12)))}
