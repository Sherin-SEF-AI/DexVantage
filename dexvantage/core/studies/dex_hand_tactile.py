"""Dexterous multi-finger self-occlusion (spec §7.14, part 1) and the extension point of the visual-to-tactile
handoff (spec §7.15, part 2, Phase 5).

    @jobs.study('dex_occlusion', 'Multi-finger occlusion', quick=False, batch='dex-occlusion')  → --dex-occlusion OUT
    run(project, ctx) -> dict                 the study (also through core.jobs.run_study('dex_occlusion', project))
    visibility_owner(points, origin, occluders, group_of, n_groups, ...) -> (visible, owner)   the kernel
    visibility_owner_raycast(...)             the same answer by exact ray casting (backend 'raycast', the default)
    patch_counts(in_view, visible, owner, n_groups)    a patch's points: hidden by each group, outside, seen
    status_for(pct, settings) -> (status, word)

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

Part 2 (spec §7.15, Phase 5) belongs in this module and is not built here: tactile_handoff() is its entry point and
says so. visibility_at(project, q, settings) is the part 1 kernel for one joint state, for part 2 to call along each
approach (core/grasps.py pre_contact_poses), and HANDOFF_FIELDS lists what part 2 adds to the result.

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

Validation (validation/checks_dexhand.py): dexhand.matrix_vs_analytic builds a scripted two-finger scene whose
occlusion is known in closed form (a box finger's shadow, by central projection, on the other's pad, and a palm
plate's edge seen from a low side camera) and compares this module's matrix entries with it;
dexhand.owner_vs_bruteforce compares the per-point owners with an all-pairs Möller–Trumbore ray cast that records
the owner of every hit; dexhand.cube_owner_union checks d = min_g d_g against one cube map of the whole scene on the
sample.

Known limits: with a cube-map backend visibility is judged per texel (a point within about a texel of a silhouette,
or on its own surface seen at a grazing angle, may go either way); the patch is a fixed point set carried with the
fingertip link (its contact area does not change with the object); the postures are straight lines between two
joint vectors, not measured synergies (core/dexhand.py), and tendon-coupled joints move independently; the arm's
joints are fixed during the synergy sweep; grasp-set IK holds the hand at one posture for every grasp and does not
check self-collision by default; no camera housing occludes; the re-aim search tries single turns of one camera,
not combinations or moves; lighting, blur and the tactile side are other studies (part 2 arrives in Phase 5).
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
}
LIMITS = {'steps': (1, 20), 'max_grasps': (1, 5000), 'ik_seeds': (1, 32), 'min_visible_pct': (0.0, 100.0),
          'good_pct': (0.0, 100.0), 'warn_pct': (0.0, 100.0), 'serious_pct': (0.0, 100.0), 'cube_n': (16, 4096),
          'grasp_closure': (0.0, 1.0), 'suggest_deg': (0.0, 45.0)}
BACKENDS = ('raycast', 'auto', 'gl', 'numpy')
WORDS = {'good': 'fingertips seen', 'warning': 'partly hidden', 'serious': 'mostly hidden',
         'critical': 'fingertips blind'}
OUT = 'out_of_view'
GROUP_TITLES = {'palm': 'Palm', 'arm': 'Wrist and arm', 'objects': 'Task objects', 'scene': 'Rest of the scene',
                OUT: 'Outside the image'}
HANDOFF_FIELDS = ('approach_visibility', 'tactile_coverage', 'blind_gap_m')     # part 2 (Phase 5)
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
        out[k] = int(v) if isinstance(DEFAULTS.get(k), int) or k in ('steps', 'max_grasps', 'ik_seeds', 'cube_n') \
            else v
    if out['backend'] not in BACKENDS:
        raise StudyError(f'The occlusion backend must be one of {", ".join(BACKENDS)}; got "{out["backend"]}".')
    if out['source'] not in ('synergy', 'grasp_set'):
        raise StudyError('The cases come from "synergy" (the grasp postures) or "grasp_set"; got '
                         f'"{out["source"]}".')
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
                          'q': res.q[k]})
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
        'handoff': None,                                  # part 2 (spec §7.15, Phase 5): HANDOFF_FIELDS
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
def tactile_handoff(project, ctx: jobs.Ctx | None = None) -> dict:
    """Spec §7.15 (Phase 5), the extension point: along each approach (core/grasps.py pre_contact_poses), the
    visibility of the contact region from visibility_at() against the distance to contact, the tactile pads'
    coverage once in contact (the tactile model of §6), and the sensory blind gap (the approach distance over which
    the contact is neither seen nor touched; a user threshold with a labelled default). Its result goes into the
    part 1 document under 'handoff' with the keys HANDOFF_FIELDS. Not built in this phase."""
    raise StudyError('The visual-to-tactile handoff (blind gap) arrives in Phase 5.')
