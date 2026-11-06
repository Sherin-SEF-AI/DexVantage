"""Grasp self-occlusion (spec §7.3, the hero study): can the rig's cameras see the fingertips and the object at the
moment of grasp, and on the way in?

    @jobs.study('grasp_visibility', 'Grasp self-occlusion', batch='grasp', needs_robot=True)   → --grasp OUT.json
    run(project, ctx) -> dict            the study (also core.jobs.run_study('grasp_visibility', project))
    settings_of(project) -> dict         the settings with the labelled defaults filled in
    decode(doc) -> dict                  the per-grasp table and the gallery images as numpy arrays
    status_for(pct, settings)            the headline's status and word for a Contact Visibility Score (%)
    bootstrap_ci(x, B, level, rng)       percentile bootstrap interval of a mean
    wilson_interval(k, n, level)         Wilson's score interval of a binomial proportion (the check's reference)
    Evaluator(project, st)               the visibility machinery, for the validation checks and the tests

What it computes. The grasp set (core/grasps.py, project.tasks.grasp_sets; settings 'grasp_set', default the first
set) names a robot and gripper-frame poses at contact. For every grasp:

1. Inverse kinematics, batched (core/robot/ik.solve_batch over every grasp at once: damped least squares with
   joint-limit avoidance and several seeds; the target is the grasp centre, grasps.ik_targets). A solution must be
   free of self-collision (ik's own test) and of the project's obstacles (core/robot/collision.obstacle_distance:
   the task objects, the tables and the static geoms of the robot's file) by at least collision_margin_m (default
   0 m). A grasp whose chosen solution hits an obstacle is solved again twice, each time keeping the solution
   nearest a joint vector drawn uniformly within the limits (so another IK branch can be found), and keeps one that
   is clear, else it is infeasible with the reason 'obstacle collision' (the link and obstacle of each first contact
   are counted). Infeasible grasps are counted by reason (ik.REASONS and 'obstacle collision'). Among several
   solutions the rule picks one: 'nearest_home' (default) or 'most_manipulable'. Joints that do not move the
   gripper (its fingers, another arm) stay at the joint state of the settings 'q' (else home): the fingers are at
   their home opening, not closed on the object. Optional backends
   (settings ik_backend): 'pinocchio' (pip install pin) and 'curobo' (NVIDIA cuRobo, CUDA), imported only when asked
   for and only for a robot read from a URDF; every solution they return is checked again with DexVantage's own
   forward kinematics (the target within 10 × the tolerance) and self-collision test, and a solution that fails
   that is infeasible ('no convergence'). Without the package the study says how to install it or to use 'dls'.
2. The contact points. Each fingertip's contact patch (grasps.contact_patches: the robot's tagged pads, else the inner
   faces of its fingertip links' collision geoms) is carried with its link to the solution q. A patch is taken from the
   collision geometry while the cameras see the visual meshes, so each point is first moved onto the visual surface
   in front of it: a ray from snap_mm (default 5 mm) outside the point along its outward normal, back towards it,
   against the robot's visual meshes at home; a point with visual surface in front of it is moved to that surface
   (UR5e sample: no point moves; SO-100 sample: 2 % of the fixed jaw's and 11 % of the moving jaw's points). The
   object's grasp region, for a grasp on a task object, is the part of object_points (default 400 points drawn on
   its surface by area, seeded) within region_radius_m (default 30 mm) of the grasp centre (the 8 nearest when fewer
   lie within it).
3. Visibility from every camera (the rig's enabled camera and depth sensors; a depth sensor by its RGB imager, as
   coverage chooses, core/studies/coverage.pick_imager). What is tested is the contact location: each patch point
   lifted lift_mm (default 2 mm, a default) along its outward normal, just in front of the pad's face, where the
   object's surface meets it. It is visible from a camera when (a) the point projects inside the image through the
   imager's lens model at its stream (CameraModel.project) and (b) nothing is nearer along the line of sight to the
   lifted point than it less eps (default 1 mm): a cube map of distances around the optical centre
   (core/occlusion.CubeMap, the GPU through EGL when there is one, face size cube_n = 512 by default), of the scene
   posed at q (the robots' visual meshes, the task objects, the tables; the camera's seat on its mount link left
   out by scenegeom.sensor_occluders' rule). A camera behind a finger has that finger in the way; a camera looking at
   the pinch from the side (in profile) sees both contact locations, as an image from there would show both
   fingertips on the object. Why the lift: the two inner faces of a parallel gripper face each other, so no camera
   outside the gap sees both faces head-on; testing the faces themselves would call every profile view blind.
   (c) Optionally, max_view_angle_deg (off by default) also asks the surface to face the camera within that angle.
   The object's grasp region is tested the same way (its points lifted along the object's normals). Only the
   triangles that can lie between the centre and the points are drawn: those whose bounding sphere meets the cone
   from the centre around the points (widened by 2° and 2 cm); a triangle outside it cannot hide any of the
   points, so the answer is the full scene's (the validation check compares with exact ray casting against the
   full scene). A camera with no point in its image builds no map. The diagnostics split the points that are not
   seen into outside the image, behind their own finger (their face turned away, n · (o − P) ≤ 0) and hidden by the
   scene.
4. The q-bucket cache. Maps are cached by camera and q-bucket: the grasp robot's joint vector divided by
   q_bucket_deg (default 0.1°, prismatic joints by q_bucket_mm, default 0.5 mm) and rounded: the joint state is
   quantised. Every grasp (or pre-contact pose) in a bucket is evaluated at the bucket's first joint vector: its
   maps, its camera poses and its contact points (world-fixed points, the object region, keep their own positions),
   and a map is reused when the points lie inside the cone it was drawn for (else it is drawn again). Grasps are
   evaluated in a seeded random order with the members of a bucket together. The result reports the maps reused
   (the cache hit rate) and the evaluations that took a bucket's joint vector; the validation check measures what
   this changes against an evaluation at each grasp's own q (UR5e sample: a patch's visible fraction changes by
   0.2 to 0.6 % on average; where an occluder's edge crosses a patch a row of its points can flip, up to 7 %).
5. Pre-contact: the gripper backed off along −approach by precontact_cm (default 5 cm) in precontact_steps equal
   steps (default 2: 2.5 and 5 cm), each solved by IK from the contact solution ('nearest_q0', so the arm does not
   change branch; self-collision checked, obstacles not), and the same visibility measured there.

Measures per camera and for the rig. Per grasp and camera: each fingertip's visible fraction (the share of its
patch's points visible; the points are spread evenly by area, so this is the visible share of the patch), the
object grasp region's visible fraction, and the same at each pre-contact step. The requirement (settings
requirement, default 'all_fingertips' with min_visible 50 % in at least min_cameras = 1 camera, same_camera true:
one camera sees every fingertip at 50 % or more; all three are labelled defaults) gives each grasp a score: per
camera the weaker fingertip's fraction (any_fingertip: the stronger; object: the object region's;
fingertips_and_object: the smaller of the two), and over cameras the min_cameras-th best (same_camera false: per
fingertip the min_cameras-th best camera, then the weakest fingertip). The requirement holds when the score is at
least min_visible. The Contact Visibility Score (CVS) is the share of the feasible grasps (with an object, for the
object requirements) where it holds, with a percentile bootstrap interval (default 95 %, 2000 resamples, seeded
by rng_seed): the resampled mean of a 0/1 indicator is Binomial(n, p̂)/n, which is drawn directly (the same
distribution as resampling the grasps). Wilson's score interval is given beside it. Also reported: the share of all
grasps (infeasible ones count as failing), the both-fingertips-visible rate per camera and for the rig, the object's
visibility, the pre-contact visibility per step and averaged, and per camera why contact points are not seen
(outside the image, behind their own finger, hidden by the scene).

Outputs. A per-grasp table (decode); the grasp centres with their scores (the page's heatmap over the task volume);
a worst-case gallery: Camera view renders (core/camrender) of the gallery (default 6) lowest-scoring feasible grasps
from every camera (up to 4), with each contact point's state (outside the image, behind its own finger, hidden by
the scene, visible); and "what to move": the CVS on a seeded subset of
sens_grasps feasible grasps (default 250) with one camera at a time moved by ±sens_step_m (default 2 cm) along each
axis of its mount's parent frame (the world for a fixed mount, the link for one on a robot) or turned by
±sens_step_deg (default 5°) about its optical x (tilt), y (pan) or z (roll) axis, ranked by the change in CVS
(paired, with a paired bootstrap interval), each with the mount pose it means (the page's Apply). Turning a camera
about its optical centre reuses the maps (they are aligned with the world, not the camera); moving it draws new
ones.

Scale and timing. 1,000 to 10,000 grasps; above BACKGROUND_GRASPS (2,000) the study hands partial results
(ctx.partial) every chunk. The result reports the throughput (grasps per second, overall and of the occlusion
step), the maps drawn, the cache hit rate and the renderer; docs/models/grasp.md records the measured budget.

Settings (project.analysis.studies['grasp_visibility'], every key optional): grasp_set; q; rule; ik_backend; ik_seeds;
collision_margin_m; requirement; min_visible; min_cameras; same_camera; precontact_cm; precontact_steps;
region_radius_m; object_points; lift_mm; max_view_angle_deg; snap_mm; eps_mm; cube_n; backend; q_bucket_deg;
q_bucket_mm; bootstrap; ci; rng_seed; good_pct, warn_pct, serious_pct (headline: 'good' from 90 %, 'warning' from 70 %,
'serious' from 40 %, else 'critical'; defaults, not standards); sensitivity; sens_grasps; sens_step_m;
sens_step_deg; gallery; gallery_px; sensors [uid]; modes {uid: [w, h]}; imagers {uid: name}; limit (evaluate at most
this many grasps, evenly picked).

Validation (dexvantage/validation/checks_grasp.py): per-point visibility against exact ray casting of the full scene
(core/raycast.visible_exact) at IK solutions of sampled grasps on the UR5e sample; the bootstrap interval's coverage
over many seeded synthetic binomial samples against Wilson's score interval's; maps reused within a q-bucket against
maps drawn at each q.

Known limits: visibility is per cube-map texel (a point within about a texel of a silhouette may be judged either
way; at 512 texels a texel spans 3.9 mrad at a face centre, 2 mm at 0.5 m); the fingers stay at their home opening
(they are not closed to the object's width) and the 2F-85's linkage joints are independent entries of q; the
contact patch comes from the collision geometry, moved onto the visual mesh only within snap_mm; the object's grasp
region is a ball around the grasp centre on its surface, not the true contact footprint; IK is a local method (a
reachable grasp can be reported infeasible); obstacles are checked at contact, not along the approach; the
pre-contact path is a straight line along the approach; no sensor housings occlude (scenegeom); the sensitivity is
one camera and one axis at a time on a subset (interactions between moves are not explored), and the pinocchio and
cuRobo backends were not available where this was built, so they are untested here.
"""
from __future__ import annotations

import hashlib
import math
import time
from collections import OrderedDict

import numpy as np

from .. import jobs
from ..jobs import StudyError

SID = 'grasp_visibility'
DEFAULTS = {
    'rule': 'nearest_home',
    'ik_backend': 'dls',
    'ik_seeds': 8,
    'collision_margin_m': 0.0,
    'requirement': 'all_fingertips',   # default requirement: one camera sees every fingertip at min_visible or more
    'min_visible': 0.5,                # default: 50 % of a patch
    'min_cameras': 1,                  # default: in at least one camera
    'same_camera': True,
    'precontact_cm': 5.0,              # default: the last 5 cm of the approach
    'precontact_steps': 2,             # default: measured at 2.5 and 5 cm
    'region_radius_m': 0.03,           # default object grasp region around the grasp centre
    'object_points': 400,
    'lift_mm': 2.0,                    # default: the contact location tested this far in front of each pad point
    'max_view_angle_deg': None,        # optional obliquity rule (off by default): a surface seen more obliquely
                                       # than this does not count as seen
    'snap_mm': 5.0,                    # default search distance to move pad points onto the visual surface
    'eps_mm': 1.0,                     # default visibility tolerance (core/occlusion EPS_M)
    'cube_n': 512,
    'backend': 'auto',
    'q_bucket_deg': 0.1,               # default q-bucket size, revolute joints
    'q_bucket_mm': 0.5,                # default q-bucket size, prismatic joints
    'bootstrap': 2000,
    'ci': 0.95,
    'rng_seed': 0,
    'good_pct': 90.0,
    'warn_pct': 70.0,
    'serious_pct': 40.0,
    'sensitivity': True,
    'sens_grasps': 250,
    'sens_step_m': 0.02,               # default "what to move" step: 2 cm
    'sens_step_deg': 5.0,              # default "what to move" turn: 5°
    'gallery': 6,
    'gallery_px': 320,
    'chunk': 250,
    'limit': None,
}
REQUIREMENTS = ('all_fingertips', 'any_fingertip', 'object', 'fingertips_and_object')
REQUIREMENT_TITLES = {'all_fingertips': 'every fingertip', 'any_fingertip': 'at least one fingertip',
                      'object': 'the object\'s grasp region', 'fingertips_and_object': 'every fingertip and the object'}
IK_BACKENDS = ('dls', 'pinocchio', 'curobo')
IK_PACKAGES = {'pinocchio': ('pinocchio', 'pip install pin'), 'curobo': ('curobo', 'NVIDIA cuRobo (needs CUDA; see '
                                                                         'its install guide)')}
OBSTACLE = 'obstacle collision'
WORDS = {'good': 'well seen', 'warning': 'often hidden', 'serious': 'mostly hidden', 'critical': 'blind at contact'}
BACKGROUND_GRASPS = 2000
CONE_MARGIN_DEG = 2.0
CONE_MARGIN_M = 0.02
CONE_MAX_DEG = 75.0               # a wider cone culls nothing (every triangle is drawn)
MIN_REGION_POINTS = 8
CACHE_BYTES = 256 * 2 ** 20
MAX_GALLERY_CAMERAS = 4
ST_OUT, ST_AWAY, ST_HIDDEN, ST_SEEN = 0, 1, 2, 3          # contact point states (gallery)
STATE_NAMES = {ST_OUT: 'outside the image', ST_AWAY: 'behind its own finger', ST_HIDDEN: 'hidden by the scene',
               ST_SEEN: 'visible'}
AXES = ('x', 'y', 'z')
TURNS = {'x': 'tilt', 'y': 'pan', 'z': 'roll'}


# ====================================================================== settings
def settings_of(project) -> dict:
    """The study's settings with the defaults filled in; 'defaults' lists the keys left at their default."""
    given = dict(project.analysis.studies.get(SID) or {})
    out = dict(DEFAULTS)
    used = []
    for k in DEFAULTS:
        if given.get(k) is not None:
            out[k] = given[k]
        else:
            used.append(k)
    floats = ('collision_margin_m', 'min_visible', 'precontact_cm', 'region_radius_m', 'lift_mm', 'snap_mm',
              'eps_mm', 'q_bucket_deg', 'q_bucket_mm', 'ci', 'good_pct', 'warn_pct', 'serious_pct', 'sens_step_m',
              'sens_step_deg')
    ints = ('ik_seeds', 'min_cameras', 'precontact_steps', 'object_points', 'cube_n', 'bootstrap', 'rng_seed',
            'sens_grasps', 'gallery', 'gallery_px', 'chunk')
    for k in floats:
        try:
            out[k] = float(out[k])
        except (TypeError, ValueError):
            raise StudyError(f'The grasp study setting {k} must be a number; it is {out[k]!r}.') from None
        if not math.isfinite(out[k]) or out[k] < 0:
            raise StudyError(f'The grasp study setting {k} must be zero or a positive number; it is {out[k]:g}.')
    for k in ints:
        try:
            out[k] = int(out[k])
        except (TypeError, ValueError):
            raise StudyError(f'The grasp study setting {k} must be a whole number; it is {out[k]!r}.') from None
        if out[k] < 0:
            raise StudyError(f'The grasp study setting {k} must not be negative; it is {out[k]}.')
    if out['limit'] is not None:
        try:
            out['limit'] = max(1, int(out['limit']))
        except (TypeError, ValueError):
            raise StudyError('The grasp study setting limit must be a whole number of grasps.') from None
    if not 0.0 <= out['min_visible'] <= 1.0:
        raise StudyError('min_visible is a fraction from 0 to 1 (0.5 = half of a patch); it is '
                         f'{out["min_visible"]:g}.')
    if not 0.5 <= out['ci'] < 1.0:
        raise StudyError(f'The interval level ci is between 0.5 and 1 (0.95 for 95 %); it is {out["ci"]:g}.')
    if out['min_cameras'] < 1:
        raise StudyError('min_cameras must be at least 1.')
    if not 16 <= out['cube_n'] <= 4096:
        raise StudyError(f'The cube-map face size must be 16 to 4096 texels; it is {out["cube_n"]}.')
    if out['max_view_angle_deg'] is not None:
        try:
            out['max_view_angle_deg'] = float(out['max_view_angle_deg'])
        except (TypeError, ValueError):
            raise StudyError('max_view_angle_deg must be a number of degrees, or empty to switch the rule off.') \
                from None
        if not 0 < out['max_view_angle_deg'] <= 90:
            raise StudyError('max_view_angle_deg is more than 0° and at most 90° (or empty: the rule is off).')
    if out['requirement'] not in REQUIREMENTS:
        raise StudyError(f'The requirement is one of {", ".join(REQUIREMENTS)}; it is {out["requirement"]!r}.')
    if out['rule'] not in ('nearest_home', 'most_manipulable'):
        raise StudyError(f'The IK rule is nearest_home or most_manipulable; it is {out["rule"]!r}.')
    if out['ik_backend'] not in IK_BACKENDS:
        raise StudyError(f'The IK backend is one of {", ".join(IK_BACKENDS)}; it is {out["ik_backend"]!r}.')
    if out['backend'] not in ('auto', 'gl', 'numpy'):
        raise StudyError(f'The occlusion backend is auto, gl or numpy; it is {out["backend"]!r}.')
    out['ik_seeds'] = max(1, out['ik_seeds'])
    out['bootstrap'] = max(100, out['bootstrap'])
    out['chunk'] = max(10, out['chunk'])
    out['same_camera'] = bool(out['same_camera'])
    out['sensitivity'] = bool(out['sensitivity'])
    out['gallery_px'] = min(max(out['gallery_px'], 64), 1280)
    out['grasp_set'] = given.get('grasp_set') or ''
    for k in ('q', 'imagers', 'modes'):
        v = given.get(k)
        out[k] = dict(v) if isinstance(v, dict) else {}
    out['sensors'] = [str(u) for u in given.get('sensors') or []]
    out['depth_imager'] = given.get('depth_imager') or 'rgb'
    out['defaults'] = used
    return out


def status_for(pct: float, st: dict) -> tuple[str, str]:
    """(severity, word) of a Contact Visibility Score in percent, from the user's thresholds."""
    if pct >= st['good_pct']:
        sev = 'good'
    elif pct >= st['warn_pct']:
        sev = 'warning'
    elif pct >= st['serious_pct']:
        sev = 'serious'
    else:
        sev = 'critical'
    return sev, WORDS[sev]


# ====================================================================== statistics
def bootstrap_ci(x, B: int = 2000, level: float = 0.95, rng=None) -> tuple[float, float]:
    """Percentile bootstrap interval [Efron 1979] of the mean of x. For a 0/1 indicator the mean of a resample of n
    with replacement is Binomial(n, p̂)/n exactly, so that is drawn directly; other data are resampled."""
    x = np.asarray(x, float).reshape(-1)
    n = len(x)
    if n == 0:
        return float('nan'), float('nan')
    g = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    a = (1.0 - level) / 2.0
    if np.isin(x, (0.0, 1.0)).all():
        m = g.binomial(n, float(x.mean()), int(B)) / n
    else:
        m = np.empty(int(B))
        for s in range(0, int(B), 256):
            k = min(256, int(B) - s)
            m[s:s + k] = x[g.integers(0, n, (k, n))].mean(axis=1)
    lo, hi = np.quantile(m, [a, 1.0 - a])
    return float(lo), float(hi)


def wilson_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson's score interval [Wilson 1927] of a proportion k/n: (p̂ + z²/2n ± z √(p̂(1−p̂)/n + z²/4n²)) / (1 + z²/n)."""
    from statistics import NormalDist
    if n <= 0:
        return float('nan'), float('nan')
    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


# ====================================================================== helpers
def _num(x, nd=6):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def _nums(a, nd=4) -> list:
    return [_num(v, nd) for v in np.asarray(a, float).reshape(-1)]


def _rot(axis: str, ang: float) -> np.ndarray:
    c, s = math.cos(ang), math.sin(ang)
    k = AXES.index(axis)
    R = np.eye(3)
    i, j = [(1, 2), (2, 0), (0, 1)][k]
    R[i, i] = R[j, j] = c
    R[i, j], R[j, i] = -s, s
    return R


class _Occ:
    """Triangles for occlusion.CubeMap.build with a content digest given by the caller (see Evaluator.occluders)."""
    __slots__ = ('tris', '_d')

    def __init__(self, tris, digest: str):
        self.tris = tris
        self._d = digest

    def digest(self) -> str:
        return self._d


def _cone(o, P):
    """(axis, half-angle rad, farthest distance, apex o) of the cone from o around points P, widened by
    CONE_MARGIN_DEG and CONE_MARGIN_M, or None when it is too wide to cull anything (the points surround o)."""
    v = np.asarray(P, float) - o
    d = np.linalg.norm(v, axis=1)
    if not len(d) or d.min() < 1e-9:
        return None
    u = v / d[:, None]
    m = u.mean(axis=0)
    nm = float(np.linalg.norm(m))
    if nm < 0.5:
        return None
    a = m / nm
    ang = float(np.arccos(np.clip((u @ a).min(), -1.0, 1.0)))
    ang += math.radians(CONE_MARGIN_DEG)
    if ang >= math.radians(CONE_MAX_DEG):
        return None
    return a, ang, float(d.max()) + CONE_MARGIN_M, np.asarray(o, float).reshape(3)


def _in_cone(C, r, cone) -> np.ndarray:
    """(M,) True where a sphere (centre C (M, 3), radius r (M,)) may meet the cone; conservative (it may keep a
    sphere that does not meet it, never drops one that does). With w = C − apex, the axis a and the half-angle θ,
    the distance from the centre to the cone's surface is |w| sin(φ − θ) = |w − (w·a) a| cos θ − (w·a) sin θ when
    the angle φ between w and a is within θ + 90° (negative inside the cone); beyond that the nearest point is the
    apex, |w| ≥ |w| sin(φ − θ). So the sphere is kept when that value is at most r and |w| − r is at most the cone's
    length."""
    a, ang, dmax, o = cone
    w = C - o
    along = w @ a
    ww = np.einsum('ij,ij->i', w, w)
    perp = np.sqrt(np.maximum(ww - along * along, 0.0))
    return (perp * math.cos(ang) - along * math.sin(ang) <= r) & (np.sqrt(ww) - r <= dmax)


# ====================================================================== the scene and the cameras
class _RobotGeom:
    """One robot's visual triangles grouped by link, in link frames, with a bounding sphere per link."""

    def __init__(self, robot, model):
        self.uid = robot.uid
        self.name = robot.name
        self.model = model
        tl = np.asarray(model.local_triangles(), np.float64) if model.n_triangles else np.zeros((0, 3, 3))
        li = np.asarray(model.triangle_links(), np.int64) if model.n_triangles else np.zeros(0, np.int64)
        L = len(model.link_names)
        self.local = [tl[li == k].astype(np.float32) for k in range(L)]
        self.tri_centre = [t.astype(float).mean(axis=1) for t in self.local]       # per triangle, link frame
        self.tri_radius = [np.linalg.norm(t.astype(float) - c[:, None], axis=2).max(axis=1) if len(t) else np.zeros(0)
                           for t, c in zip(self.local, self.tri_centre)]
        self.centre = np.zeros((L, 3))
        self.radius = np.zeros(L)
        for k, t in enumerate(self.local):
            if len(t):
                v = t.reshape(-1, 3).astype(float)
                c = (v.min(axis=0) + v.max(axis=0)) / 2
                self.centre[k] = c
                self.radius[k] = float(np.linalg.norm(v - c, axis=1).max())
        self.links = [k for k in range(L) if len(self.local[k])]
        self.digest = hashlib.blake2b(tl.astype(np.float32).tobytes(), digest_size=16).hexdigest()


class _Cam:
    """One camera of the rig: its imager, lens model and where it is (fixed, or on a link)."""

    def __init__(self, sensor, imager, why, project):
        self.sensor = sensor
        self.uid = sensor.uid
        self.name = sensor.name
        self.imager = imager
        self.cam = imager.camera
        self.why = why
        self.on_link = bool(sensor.mount.on_link)
        self.robot_uid = sensor.mount.robot_uid if self.on_link else ''
        self.link = sensor.mount.parent_frame if self.on_link else ''
        self.link_i = -1
        T_po = sensor.mount.pose.matrix() @ imager.pose_in_body.matrix()     # optical frame in the mount's parent
        self.T_po = T_po
        self.seat = None

    def optical(self, fkd: dict, T_po=None) -> tuple[np.ndarray, np.ndarray]:
        """(R (3, 3), t (3,)) of the optical frame in the world at the forward kinematics fkd."""
        T = self.T_po if T_po is None else T_po
        if self.on_link:
            T = fkd[self.robot_uid][self.link_i] @ T
        return T[:3, :3], T[:3, 3]


class Evaluator:
    """The visibility machinery of the study (module docstring, steps 2 to 4), without the IK."""

    def __init__(self, project, st: dict, notes: list | None = None, cameras=None):
        from .. import scenegeom as SG
        from ..robot.model import RobotModelError, get_model
        self.project = project
        self.st = st
        self.notes = notes if notes is not None else []
        self.robots: dict = {}
        for r in project.robots:
            try:
                m = get_model(project, r.uid)
            except RobotModelError as e:
                self.notes.append(f'{r.name} is not an occluder: {e}')
                continue
            self.robots[r.uid] = _RobotGeom(r, m)
        static = SG.scene_occluders(project, None, robots=False)
        self.notes += list(static.notes)
        self.static = np.ascontiguousarray(static.tris, np.float32)
        if len(self.static):
            v = self.static.astype(float)
            self.s_centre = v.mean(axis=1)
            self.s_radius = np.linalg.norm(v - self.s_centre[:, None], axis=2).max(axis=1)
        else:
            self.s_centre, self.s_radius = np.zeros((0, 3)), np.zeros(0)
        self.static_digest = static.digest()
        self.cams = list(cameras) if cameras is not None else self._cameras()
        for c in self.cams:
            self._place(c)
        mva = st.get('max_view_angle_deg')
        self.cos_view = None if mva is None else math.cos(math.radians(mva))
        self.lift = st['lift_mm'] / 1000.0
        self.eps = st['eps_mm'] / 1000.0
        self._posed_for, self._posed_links = None, {}
        self.maps: OrderedDict = OrderedDict()
        self.hits = self.misses = self.builds = self.reuse_refused = 0
        self.build_s = 0.0
        self.backend = ''
        self.renderer = ''

    # ---------------------------------------------------------------- cameras
    def _cameras(self) -> list:
        from .. import library as LB
        from . import coverage as CV
        st = self.st
        sensors = [s for s in self.project.variant.sensors if s.enabled and s.kind in ('camera', 'depth')
                   and (not st['sensors'] or s.uid in st['sensors'])]
        out = []
        for s in sensors:
            try:
                ims = LB.sensor_imagers(s, CV.sensor_mode(s, st))
            except LB.LibraryError as e:
                self.notes.append(f'{s.name} is left out: {e}')
                continue
            if not ims:
                self.notes.append(f'{s.name} has no imager in its library entry and is left out.')
                continue
            if s.model_id and ims[0].is_default:
                self.notes.append(f'{s.name}: its device "{s.model_id}" is not in the library, so the default camera '
                                  'is used (a default, not a device figure).')
            im, why = CV.pick_imager(s, ims, st)
            out.append(_Cam(s, im, why, self.project))
        return out

    def _place(self, c: _Cam):
        if not c.on_link:
            return
        g = self.robots.get(c.robot_uid)
        if g is None:
            raise StudyError(f'{c.name} is mounted on a robot that cannot be posed.')
        try:
            c.link_i = g.model.link_index(c.link)
        except KeyError:
            raise StudyError(f'{c.name} is mounted on {c.link}, which its robot does not have.') from None
        c.seat = self.seat_mask(c, c.T_po)

    def seat_mask(self, c: _Cam, T_po) -> np.ndarray | None:
        """(K,) True for the mount link's triangles wholly within scenegeom.OWN_RADIUS_M of the optical centre (the
        sensor's seat, scenegeom.sensor_occluders' rule), in the link frame."""
        from ..scenegeom import OWN_RADIUS_M
        if not c.on_link:
            return None
        t = self.robots[c.robot_uid].local[c.link_i]
        if not len(t):
            return None
        return (np.linalg.norm(t.astype(float) - T_po[:3, 3], axis=2) <= OWN_RADIUS_M).all(axis=1)

    # ---------------------------------------------------------------- geometry
    def fk(self, qd: dict) -> dict:
        return {uid: g.model.fk_array(qd[uid] if uid in qd else g.model.home) for uid, g in self.robots.items()}

    def _centroids(self, fkd: dict, uid: str, g: _RobotGeom, k: int) -> np.ndarray:
        """(K, 3) world centroids of link k's triangles at fkd, kept while the same fkd is asked for (the cameras
        of one joint state share them)."""
        if self._posed_for is not fkd:
            self._posed_for, self._posed_links = fkd, {}
        c = self._posed_links.get((uid, k))
        if c is None:
            T = fkd[uid][k]
            c = self._posed_links[(uid, k)] = g.tri_centre[k] @ T[:3, :3].T + T[:3, 3]
        return c

    def occluders(self, fkd: dict, qd: dict, cone, c: _Cam | None = None, seat=None) -> _Occ:
        """The scene's triangles at fkd that can lie in cone (all when cone is None), the seat of camera c left out."""
        parts = []
        key = [self.static_digest, 'all' if cone is None else
               repr((np.round(cone[0], 9).tolist(), round(cone[1], 9), round(cone[2], 9),
                     np.round(cone[3], 9).tolist()))]
        seat = c.seat if (c is not None and seat is None) else seat
        if len(self.static):
            keep = np.ones(len(self.static), bool) if cone is None else \
                _in_cone(self.s_centre, self.s_radius, cone)
            parts.append(self.static[keep])
        for uid, g in self.robots.items():
            T = fkd[uid]
            # the posed link transforms, not q alone: they carry the robot's base pose too, and the GL backend keeps
            # uploaded geometry by this digest (a moved second arm at the same q must not reuse its old triangles)
            key += [uid, g.digest, hashlib.blake2b(np.ascontiguousarray(T, np.float64).tobytes(),
                                                   digest_size=16).hexdigest()]
            if cone is not None:
                C = np.einsum('lij,lj->li', T[:, :3, :3], g.centre) + T[:, :3, 3]
                ok = _in_cone(C, g.radius, cone)
            else:
                ok = np.ones(len(T), bool)
            for k in g.links:
                if not ok[k]:
                    continue
                t = g.local[k]
                keep = None
                if c is not None and c.on_link and uid == c.robot_uid and k == c.link_i and seat is not None:
                    keep = ~seat
                    key.append(f'seat:{c.uid}:' + hashlib.blake2b(np.packbits(seat).tobytes(),
                                                                  digest_size=8).hexdigest())
                if cone is not None and len(t) > 64:
                    m = _in_cone(self._centroids(fkd, uid, g, k), g.tri_radius[k], cone)
                    keep = m if keep is None else keep & m
                if keep is not None:
                    t = t[keep]
                if not len(t):
                    continue
                R32, t32 = T[k, :3, :3].T.astype(np.float32), T[k, :3, 3].astype(np.float32)
                parts.append((t.reshape(-1, 3) @ R32 + t32).reshape(-1, 3, 3))
        tris = np.ascontiguousarray(np.concatenate(parts) if parts else np.zeros((0, 3, 3)), np.float32)
        dg = hashlib.blake2b('|'.join(map(str, key)).encode(), digest_size=16).hexdigest()
        return _Occ(tris, dg)

    def build(self, fkd, qd, o, P, c: _Cam | None, seat=None, cull: bool = True):
        """A cube map around o of the scene at fkd, drawing only what can hide points P (cull)."""
        from .. import occlusion as OC
        cone = _cone(o, P) if cull else None
        occ = self.occluders(fkd, qd, cone, c, seat)
        t0 = time.perf_counter()
        try:
            cube = OC.CubeMap.build(occ, o, self.st['cube_n'], self.st['backend'])
        except OC.OcclusionError as e:
            raise StudyError(f'The occlusion map could not be built: {e}. Set the grasp study\'s backend to "numpy" '
                             'to use the software rasteriser.') from None
        self.build_s += time.perf_counter() - t0
        self.builds += 1
        self.backend, self.renderer = cube.backend, cube.renderer
        return cube, cone

    def cube(self, j: int, bucket, fkd, qd, o, P, use_cache: bool = True, T_po=None):
        """The map of camera j for points P from o, through the q-bucket cache when bucket is not None."""
        c = self.cams[j]
        seat = None if T_po is None else self.seat_mask(c, T_po)
        if not use_cache or bucket is None:
            return self.build(fkd, qd, o, P, c, seat)[0]
        key = (j, bucket)
        hit = self.maps.get(key)
        if hit is not None:
            cube, cone = hit
            if cone is None or _points_in(P, cone):
                self.maps.move_to_end(key)
                self.hits += 1
                return cube
            self.reuse_refused += 1
        self.misses += 1
        cube, cone = self.build(fkd, qd, o, P, c, seat)
        self.maps[key] = (cube, cone)
        total = sum(x[0].nbytes for x in self.maps.values())
        while total > CACHE_BYTES and len(self.maps) > 1:
            _k, old = self.maps.popitem(last=False)
            total -= old[0].nbytes
        return cube

    # ---------------------------------------------------------------- visibility
    def states(self, j: int, fkd, qd, P, N, bucket=None, use_cache=True, full=False, T_po=None, R_turn=None):
        """Per point (K,) of camera j: (in_image, facing, unoccluded, front) bool arrays. A point is visible when all
        of the first three hold. Occlusion is tested at the point lifted lift_mm along its outward normal N (the
        contact location in front of the surface), for the points in the image (full=True: for every point, for the
        turned-camera sensitivity). facing is the optional obliquity rule (max_view_angle_deg; all True when it is
        off); front says which points' surfaces face the camera's side (n · (o − P) > 0), for the diagnostics."""
        c = self.cams[j]
        R, o = c.optical(fkd, T_po)
        if R_turn is not None:
            R = R @ R_turn
        P = np.asarray(P, float).reshape(-1, 3)
        K = len(P)
        if N is None:
            front = facing = np.ones(K, bool)
            Pt = P
        else:
            N = np.asarray(N, float).reshape(-1, 3)
            v = o - P
            d = np.linalg.norm(v, axis=1)
            with np.errstate(invalid='ignore', divide='ignore'):
                cosv = np.einsum('ij,ij->i', N, v) / np.maximum(d, 1e-300)
            front = cosv > 0
            facing = np.ones(K, bool) if self.cos_view is None else cosv >= self.cos_view
            Pt = P + self.lift * N
        _uv, inimg = c.cam.project((P - o) @ R)
        inimg = np.asarray(inimg, bool)
        cand = facing & (inimg | full)
        unocc = np.zeros(K, bool)
        if cand.any():
            cube = self.cube(j, bucket, fkd, qd, o, Pt[cand], use_cache, T_po)
            unocc[cand] = cube.visible(Pt[cand], self.eps)
        return inimg, facing, unocc, front

    @property
    def hit_rate(self) -> float | None:
        n = self.hits + self.misses
        return self.hits / n if n else None


def _points_in(P, cone) -> bool:
    """True when every point lies inside the cone a map was drawn for (so the map may be reused for them)."""
    a, ang, dmax, o = cone
    v = np.asarray(P, float) - o
    d = np.linalg.norm(v, axis=1)
    if not len(d):
        return True
    if (d > dmax).any() or (d < 1e-9).any():
        return False
    return bool(((v @ a) / d >= math.cos(ang)).all())


# ====================================================================== contact points
def snapped_pads(project, robot_uid: str, snap_m: float, notes: list) -> list:
    """[{'name', 'link_i', 'P' (K, 3), 'N' (K, 3) link frame, 'moved': int}] of the robot's contact patches, each point
    moved onto the visual surface in front of it within snap_m (module docstring, step 2)."""
    from .. import grasps as GR
    from .. import raycast as RC
    from ..robot.model import get_model
    try:
        pads = GR.contact_patches(project, robot_uid, None)
    except GR.GraspError as e:
        raise StudyError(str(e)) from None
    if not pads:
        raise StudyError('The grasp study needs the gripper\'s fingertips or contact pads: tag them on the Robot page '
                         '(the import wizard\'s fingertip step).')
    m = get_model(project, robot_uid)
    T = m.fk_array(m.home)
    grid = None
    if snap_m > 0 and m.n_triangles:
        grid = RC.Grid.build(m.posed_triangles(m.home)[0])
    out = []
    for name, p in pads.items():
        P, N = np.asarray(p['points'], float), np.asarray(p['normals'], float)
        moved = 0
        if grid is not None and len(P):
            o = P + N * snap_m
            t = RC.first_hit(o, -N, grid, tmax=snap_m)
            fix = np.isfinite(t) & (t < snap_m - 1e-6)
            P = np.where(fix[:, None], o - N * t[:, None], P)
            moved = int(fix.sum())
        li = m.link_index(p['link'])
        R, t0 = T[li, :3, :3], T[li, :3, 3]
        out.append({'name': name, 'link': p['link'], 'link_i': li, 'P': (P - t0) @ R, 'N': N @ R, 'moved': moved,
                    'points': len(P), 'rule': p.get('rule', '')})
    return out


def pad_world(pads: list, T: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(P (K, 3), N (K, 3), segment (K,)) of the pads at link transforms T (n_links, 4, 4)."""
    Ps, Ns, seg = [], [], []
    for f, p in enumerate(pads):
        A = T[p['link_i']]
        Ps.append(p['P'] @ A[:3, :3].T + A[:3, 3])
        Ns.append(p['N'] @ A[:3, :3].T)
        seg.append(np.full(len(p['P']), f, np.int32))
    return np.concatenate(Ps), np.concatenate(Ns), np.concatenate(seg)


def _object_surface(project, uid: str, n: int, seed: int):
    """(P (n, 3), N (n, 3)) points on a task object (grasps.object_points' draw) and their outward normals."""
    from .. import grasps as GR
    from ..scenegeom import object_triangles
    o = next((x for x in project.workspace.objects if x.uid == uid), None)
    if o is None:
        return None
    tri = np.asarray(object_triangles(o, project), float)
    if not len(tri):
        return None
    P, k = GR.sample_triangles(tri, int(n), np.random.default_rng(int(seed)))
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-300)
    N = nrm[np.asarray(k, int)]
    # the tessellations wind outward (scenegeom); turn any normal that points at the object's centre
    cen = tri.reshape(-1, 3).mean(axis=0)
    flip = np.einsum('ij,ij->i', N, P - cen) < 0
    N[flip] *= -1
    return P, N


def object_region(surface, tcp_w, radius: float):
    """The object's grasp region: the surface points within radius of the grasp centre (the MIN_REGION_POINTS nearest
    when fewer lie within it)."""
    if surface is None:
        return np.zeros((0, 3)), np.zeros((0, 3))
    P, N = surface
    d = np.linalg.norm(P - tcp_w, axis=1)
    sel = d <= radius
    if sel.sum() < MIN_REGION_POINTS:
        sel = np.zeros(len(P), bool)
        sel[np.argsort(d)[:MIN_REGION_POINTS]] = True
    return P[sel], N[sel]


# ====================================================================== scores
def grasp_score(frac: np.ndarray, obj: np.ndarray, st: dict) -> np.ndarray:
    """(N,) score of each grasp (module docstring) from frac (N, C, F) fingertip fractions and obj (N, C) object
    fractions (NaN where there is none). The requirement holds where score >= min_visible."""
    req, k = st['requirement'], int(st['min_cameras'])
    frac = np.asarray(frac, float)
    obj = np.asarray(obj, float)
    N, C = frac.shape[:2]
    if k > C:
        return np.zeros(N)

    def kth_best(a, axis):           # the k-th largest along axis (NaN counts as lowest)
        s = -np.sort(-np.nan_to_num(a, nan=-1.0), axis=axis)
        return np.take(s, k - 1, axis=axis)
    if req == 'object':
        s = kth_best(obj, 1)
    elif not st['same_camera'] and req in ('all_fingertips', 'fingertips_and_object'):
        s = kth_best(frac, 1).min(axis=1)
        if req == 'fingertips_and_object':
            s = np.minimum(s, kth_best(obj, 1))
    else:
        per = frac.min(axis=2) if req != 'any_fingertip' else frac.max(axis=2)
        if req == 'fingertips_and_object':
            per = np.minimum(per, np.nan_to_num(obj, nan=-1.0))
        s = kth_best(per, 1)
    return np.where(s < 0, np.nan, s)


def _applicable(obj: np.ndarray, st: dict) -> np.ndarray:
    """(N,) True for grasps the requirement applies to (an object region where the requirement needs one)."""
    if st['requirement'] in ('object', 'fingertips_and_object'):
        return np.isfinite(np.asarray(obj, float)).any(axis=1)
    return np.ones(len(obj), bool)


# ====================================================================== IK
class _Clip2D:
    """The robot model with a clip() that also takes (N, nq) stacks: ik.solve_batch clips per-target starts q0 (N, nq)
    with model.clip, which accepts one joint vector only (an open issue of core/robot/ik.py, worked around here)."""

    def __init__(self, model):
        self._model = model

    def __getattr__(self, k):
        return getattr(self._model, k)

    def clip(self, q):
        q = np.asarray(q, float)
        if q.ndim == 2:
            return np.clip(q, self._model.lower, self._model.upper)
        return self._model.clip(q)


_PROXIES: dict = {}


def _clip2d(model):
    p = _PROXIES.get(id(model))
    if p is None or p._model is not model:
        _PROXIES.clear()
        p = _PROXIES[id(model)] = _Clip2D(model)
    return p


def _ik_dls(model, link, targets, tcp, st, hold, rng, progress=None, cancel=None, rule=None, seeds=None, q0=None,
            retry=None):
    from ..robot import ik as IK
    if q0 is not None and np.asarray(q0).ndim == 2:
        model = _clip2d(model)
    return IK.solve_batch(model, link, targets, q0=hold if q0 is None else q0, seeds=seeds or st['ik_seeds'],
                          rng=rng, rule=rule or st['rule'], collision=True, margin=st['collision_margin_m'], tcp=tcp,
                          retry_seeds=retry, progress=progress, cancel=cancel)


def _optional_backend(name: str):
    import importlib.util
    mod, how = IK_PACKAGES[name]
    if importlib.util.find_spec(mod) is None:
        raise StudyError(f'The IK backend "{name}" needs {mod}, which is not installed here ({how}). Choose the '
                         'default backend "dls" in the grasp study\'s settings.')
    return importlib.import_module(mod)


def _urdf_text(project, robot) -> str:
    if robot.source.kind != 'urdf':
        raise StudyError(f'The optional IK backends read a URDF; {robot.name} was read from '
                         f'{robot.source.kind.upper() or "another format"}. Use the default backend "dls".')
    raw = project.sources.get(robot.source.file) if hasattr(project, 'sources') else None
    if raw is None:
        raise StudyError(f'The URDF of {robot.name} is not in the project, so the optional IK backend cannot read it.')
    return raw.decode('utf-8', 'replace') if isinstance(raw, (bytes, bytearray)) else str(raw)


def _verified(model, link, targets, tcp, Q, ok, st):
    """IKBatch of solutions from an optional backend, each checked with DexVantage's FK and self-collision test."""
    from ..robot import collision as CO
    from ..robot import ik as IK
    N = len(targets)
    pe, re = np.full(N, np.inf), np.full(N, np.inf)
    good = np.zeros(N, bool)
    reason = ['no convergence'] * N
    dmin = np.full(N, np.nan)
    tc = np.eye(4) if tcp is None else np.asarray(tcp, float)
    for i in range(N):
        if not ok[i]:
            continue
        q = model.clip(Q[i])
        T = model.link_pose(q, link) @ tc
        pe[i] = float(np.linalg.norm(T[:3, 3] - targets[i, :3, 3]))
        Rd = T[:3, :3].T @ targets[i, :3, :3]
        re[i] = float(np.arccos(np.clip((np.trace(Rd) - 1) / 2, -1, 1)))
        if pe[i] > 10 * IK.POS_TOL or re[i] > 10 * IK.ROT_TOL:
            continue
        r = CO.self_collision(model, q, st['collision_margin_m'])
        dmin[i] = r.min_distance
        if r.in_collision:
            reason[i] = 'self-collision'
            continue
        good[i], reason[i], Q[i] = True, '', q
    return IK.IKBatch(np.asarray(Q, float), good, pe, re, np.zeros(N, int), reason, good.copy(), np.full(N, np.nan),
                      dmin, good.astype(int), 'pose', {'backend': 'verified'})


def _ik_pinocchio(project, robot, model, link, targets, tcp, st, hold):     # pragma: no cover - needs pinocchio
    pin = _optional_backend('pinocchio')
    pm = pin.buildModelFromXML(_urdf_text(project, robot))
    pd = pm.createData()
    names = list(pm.names)[1:]
    if pm.nq != pm.nv or any(n not in names for n in model.joint_names):
        raise StudyError('The pinocchio backend handles robots whose moving joints all have one coordinate and whose '
                         'URDF names every joint; use the default backend "dls".')
    idx = [pm.getJointId(n) - 1 for n in model.joint_names]
    fid = pm.getFrameId(link)
    if fid >= pm.nframes:
        raise StudyError(f'pinocchio does not find the link {link} in the URDF; use the default backend "dls".')
    Binv = np.linalg.inv(model.base)
    tc = np.eye(4) if tcp is None else np.asarray(tcp, float)
    N = len(targets)
    Q = np.tile(hold, (N, 1))
    ok = np.zeros(N, bool)
    for i in range(N):
        goal = pin.SE3(Binv @ targets[i] @ np.linalg.inv(tc))
        q = np.zeros(pm.nq)
        q[idx] = hold
        for _it in range(200):
            pin.forwardKinematics(pm, pd, q)
            pin.updateFramePlacements(pm, pd)
            err = pin.log6(pd.oMf[fid].actInv(goal)).vector
            if np.linalg.norm(err[:3]) < 1e-4 and np.linalg.norm(err[3:]) < 1e-3:
                ok[i] = True
                break
            J = pin.computeFrameJacobian(pm, pd, q, fid, pin.LOCAL)
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            q = pin.integrate(pm, q, dq)
            q = np.clip(q, pm.lowerPositionLimit, pm.upperPositionLimit)
        Q[i] = q[idx]
    return _verified(model, link, targets, tcp, Q, ok, st)


def _ik_curobo(project, robot, model, link, targets, tcp, st, hold):        # pragma: no cover - needs cuRobo
    import os
    import tempfile
    _optional_backend('curobo')
    import torch
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose as CPose
    from curobo.types.robot import RobotConfig
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from ..frames import quat_from_matrix
    text = _urdf_text(project, robot)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'robot.urdf')
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(text)
        ta = TensorDeviceType()
        base = model.link_names[0]
        cfg = RobotConfig.from_basic(path, base, link, ta)
        solver = IKSolver(IKSolverConfig.load_from_robot_config(cfg, None, num_seeds=st['ik_seeds'],
                                                                self_collision_check=False, tensor_args=ta))
        tc = np.eye(4) if tcp is None else np.asarray(tcp, float)
        Tl = np.linalg.inv(model.base) @ targets @ np.linalg.inv(tc)
        pos = torch.tensor(Tl[:, :3, 3], **ta.as_torch_dict())
        qx = np.array([quat_from_matrix(T[:3, :3]) for T in Tl])
        quat = torch.tensor(np.c_[qx[:, 3], qx[:, :3]], **ta.as_torch_dict())     # cuRobo: (w, x, y, z)
        res = solver.solve_batch(CPose(pos, quat))
        sol = res.solution.view(len(Tl), -1).cpu().numpy()
        ok = res.success.view(-1).cpu().numpy().astype(bool)
        names = list(solver.kinematics.joint_names)
    N = len(targets)
    Q = np.tile(hold, (N, 1))
    for k, n in enumerate(model.joint_names):
        if n in names:
            Q[:, k] = sol[:, names.index(n)]
    return _verified(model, link, targets, tcp, Q, ok, st)


def solve_grasps(project, robot, model, link, targets, tcp, st, hold, ctx=None, notes=None):
    """IKBatch for the grasp targets (N, 4, 4) with the obstacles checked and one retry (module docstring, step 1);
    returns (batch, reasons list)."""
    from ..robot import collision as CO
    notes = notes if notes is not None else []
    rng = np.random.default_rng([int(st['rng_seed']), 1])
    prog = (lambda f: ctx.report(0.02 + 0.28 * f, 'Inverse kinematics')) if ctx is not None else None
    cancel = ctx.cancelled if ctx is not None else None
    be = st['ik_backend']
    if be == 'dls':
        b = _ik_dls(model, link, targets, tcp, st, hold, rng, prog, cancel)
    else:
        try:
            fn = _ik_pinocchio if be == 'pinocchio' else _ik_curobo
            b = fn(project, robot, model, link, targets, tcp, st, hold)
        except StudyError:
            raise
        except Exception as e:                       # noqa: BLE001 — an optional library's own failure, explained
            raise StudyError(f'The {be} IK backend failed ({type(e).__name__}: {str(e)[:200]}). Use the default '
                             'backend "dls".') from None
        notes.append(f'IK by the optional {be} backend; every solution checked with DexVantage\'s own kinematics.')
    reasons = list(b.reason)
    _obs, onotes = CO.obstacles(project)
    notes += onotes
    # the robot file's static geoms count even when the project has no obstacles
    pairs: dict = {}

    def collides(q, record=False):
        r = CO.obstacle_distance(project, robot.uid, q, st['collision_margin_m'], model=model)
        if r.in_collision and record and r.pairs:
            k = f'{r.pairs[0][0]} × {r.pairs[0][1]}'
            pairs[k] = pairs.get(k, 0) + 1
        return r.in_collision
    hit = [int(i) for i in np.flatnonzero(b.ok) if collides(b.q[i], True)]
    if hit and be == 'dls':
        # the rule would pick the same branch again: start each retry near a random joint vector instead (the joints
        # that do not move the gripper stay held) and keep the solution nearest it
        from ..robot.manipulability import chain_mask
        chain = np.asarray(chain_mask(model, link), bool)
        g = np.random.default_rng([int(st['rng_seed']), 2])
        lo, hi = np.where(np.isfinite(model.lower), model.lower, -np.pi), np.where(np.isfinite(model.upper),
                                                                                  model.upper, np.pi)
        left = list(hit)
        for _attempt in range(2):
            if not left or (cancel and cancel()):
                break
            idx = np.array(left)
            Q0 = np.tile(hold, (len(idx), 1))
            Q0[:, chain] = (lo + g.random((len(idx), model.nq)) * (hi - lo))[:, chain]
            b2 = _ik_dls(model, link, targets[idx], tcp, st, hold, g, None, cancel, rule='nearest_q0', q0=Q0,
                         retry=0)
            still = []
            for a, i in enumerate(left):
                if b2.ok[a] and not collides(b2.q[a]):
                    b.q[i], b.manipulability[i], b.pos_err[i], b.rot_err[i] = b2.q[a], b2.manipulability[a], \
                        b2.pos_err[a], b2.rot_err[a]
                else:
                    still.append(i)
            left = still
        hit = left
    for i in hit:
        b.ok[i] = False
        reasons[i] = OBSTACLE
    b.reason = reasons
    b.obstacle_pairs = dict(sorted(pairs.items(), key=lambda t: -t[1]))   # first contact of the first solution
    return b


# ====================================================================== the study
def _order(buckets: list, rng) -> list:
    """A seeded random order of the grasps with the members of each q-bucket together."""
    first: dict = {}
    for i, k in enumerate(buckets):
        first.setdefault(k, []).append(i)
    groups = list(first.values())
    perm = rng.permutation(len(groups))
    return [i for g in perm for i in groups[g]]


def _bucket(q, step_rad, step_m, prismatic) -> tuple:
    q = np.asarray(q, float)
    s = np.where(prismatic, step_m, step_rad)
    return tuple(np.rint(q / np.maximum(s, 1e-12)).astype(np.int64).tolist())


def _prismatic(model) -> np.ndarray:
    import mujoco
    return np.array([int(model.m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_SLIDE) for j in model._jid], bool)


@jobs.study(SID, 'Grasp self-occlusion', quick=False, batch='grasp', needs_robot=True)
def run(project, ctx: jobs.Ctx) -> dict:   # noqa: C901 — the study's steps in order, each said in the docstring
    """Grasp self-occlusion: the Contact Visibility Score of a grasp set from the rig's cameras (module docstring)."""
    from .. import grasps as GR
    from . import coverage as CV
    t_all = time.perf_counter()
    st = settings_of(project)
    notes: list = []
    timings: dict = {}
    ctx.report(0.0, 'Reading the grasp set')
    sets = GR.sets(project)
    if not sets:
        raise StudyError('There is no grasp set to study: make one on the Tasks page (sample the task volume, import '
                         'a file or take a trajectory).')
    gs = GR.get_set(project, st['grasp_set']) if st['grasp_set'] else sets[0]
    if gs is None:
        raise StudyError('The grasp set chosen in the settings is not in the project any more: choose another.')
    robot = project.robot(gs.get('robot_uid', ''))
    if robot is None:
        raise StudyError(f'The grasp set "{gs.get("name")}" is for a robot that is not in the project: make a set for '
                         'one of its robots.')
    poses = list(gs.get('poses') or [])
    if not poses:
        raise StudyError(f'The grasp set "{gs.get("name")}" has no grasps.')
    sel = np.arange(len(poses))
    if st['limit'] and len(poses) > st['limit']:
        sel = np.unique(np.linspace(0, len(poses) - 1, st['limit']).round().astype(int))
        notes.append(f'{len(sel):,} of the set\'s {len(poses):,} grasps are evaluated (the limit setting), evenly '
                     'picked.')
    gsel = dict(gs, poses=[poses[i] for i in sel])
    N = len(gsel['poses'])

    ev = Evaluator(project, st, notes)
    if robot.uid not in ev.robots:
        raise StudyError(f'{robot.name} cannot be posed, so its grasps cannot be studied: ' + ' '.join(notes[-1:]))
    if not ev.cams:
        raise StudyError('The grasp study needs at least one camera or depth sensor in use: add one (Sensors ▸ Add '
                         'camera) or tick "In use" on the Sensors and mounts page.')
    model = ev.robots[robot.uid].model
    qd0, qsrc = CV.joint_state(project, st, notes)
    hold = np.asarray(qd0.get(robot.uid, model.home), float)
    t0 = time.perf_counter()
    pads = snapped_pads(project, robot.uid, st['snap_mm'] / 1000.0, notes)
    try:
        geom = GR.gripper_geometry(project, robot.uid)
        link, tcp_off, targets = GR.ik_targets(project, gsel)
    except GR.GraspError as e:
        raise StudyError(str(e)) from None
    if geom.get('assumed'):
        notes.append('The grasp centre is assumed (no fingertips or pads tagged): ' + geom.get('rule', ''))
    approach = np.array([np.asarray(p.get('approach'), float) for p in gsel['poses']]).reshape(-1, 3)
    approach /= np.maximum(np.linalg.norm(approach, axis=1, keepdims=True), 1e-300)
    tcp_w = targets[:, :3, 3].copy()
    timings['setup_seconds'] = round(time.perf_counter() - t0, 4)

    # ---------------------------------------------------------------- 1. IK at contact
    ctx.report(0.02, f'Inverse kinematics of {N:,} grasps')
    t0 = time.perf_counter()
    b = solve_grasps(project, robot, model, link, targets, tcp_off, st, hold, ctx, notes)
    timings['ik_seconds'] = round(time.perf_counter() - t0, 4)
    feas = np.asarray(b.ok, bool).copy()
    Q = np.asarray(b.q, float)
    reasons = list(b.reason)
    if ctx.cancelled():
        return _cancelled(project, gs, N)

    # ---------------------------------------------------------------- 5. pre-contact IK
    steps = st['precontact_steps'] if st['precontact_cm'] > 0 else 0
    dists = [st['precontact_cm'] / 100.0 * (k + 1) / steps for k in range(steps)] if steps else []
    Qpre = np.full((len(dists), N, model.nq), np.nan)
    pre_ok = np.zeros((len(dists), N), bool)
    t0 = time.perf_counter()
    fi = np.flatnonzero(feas)
    for k, dd in enumerate(dists):
        if not len(fi) or ctx.cancelled():
            break
        ctx.report(0.30 + 0.05 * k / max(len(dists), 1), f'Pre-contact poses at {dd * 100:.1f} cm')
        Tk = targets[fi].copy()
        Tk[:, :3, 3] -= dd * approach[fi]
        start = Q[fi] if k == 0 else np.where(pre_ok[k - 1][fi][:, None], Qpre[k - 1][fi], Q[fi])
        bk = _ik_dls(model, link, Tk, tcp_off, st, hold, np.random.default_rng([int(st['rng_seed']), 3, k]),
                     rule='nearest_q0', seeds=1, q0=start, retry=0)
        Qpre[k][fi] = bk.q
        pre_ok[k][fi] = bk.ok
    timings['precontact_ik_seconds'] = round(time.perf_counter() - t0, 4)

    # ---------------------------------------------------------------- 2.-4. visibility
    C, F = len(ev.cams), len(pads)
    S1 = 1 + len(dists)
    frac = np.full((S1, N, C, F), np.nan)
    objf = np.full((S1, N, C), np.nan)
    counts = np.zeros((C, 4), np.int64)             # contact points per state, per camera
    surf_cache: dict = {}
    has_obj = np.zeros(N, bool)
    regions = []
    for i, p in enumerate(gsel['poses']):
        ou = p.get('object_uid') or ''
        if ou and ou not in surf_cache:
            surf_cache[ou] = _object_surface(project, ou, st['object_points'], st['rng_seed'] + 7)
        reg = object_region(surf_cache.get(ou), tcp_w[i], st['region_radius_m']) if ou else (np.zeros((0, 3)),) * 2
        regions.append(reg)
        has_obj[i] = len(reg[0]) > 0
    if any(ou and surf_cache.get(ou) is None for ou in surf_cache):
        notes.append('A grasp names a task object without a surface to sample; its object visibility is not computed.')
    sens_on = st['sensitivity'] and len(fi) > 0
    rng = np.random.default_rng([int(st['rng_seed']), 4])
    sens_idx = np.sort(rng.choice(fi, min(len(fi), st['sens_grasps']), replace=False)) if sens_on else np.zeros(0, int)
    sens_set = set(int(i) for i in sens_idx)
    sens_store: dict = {}
    prism = _prismatic(model)
    step_rad, step_m = math.radians(st['q_bucket_deg']), st['q_bucket_mm'] / 1000.0
    use_cache = st['q_bucket_deg'] > 0
    others = {u: v for u, v in qd0.items() if u != robot.uid}

    def qdict(q):
        d = dict(others)
        d[robot.uid] = q
        return d

    jobs_list = [(i, 0, Q[i]) for i in fi] + [(i, k + 1, Qpre[k][i]) for k in range(len(dists)) for i in fi
                                              if pre_ok[k][i]]
    buckets = [_bucket(q, step_rad, step_m, prism) if use_cache else (s, i) for i, s, q in jobs_list]
    order = _order(buckets, np.random.default_rng([int(st['rng_seed']), 5]))
    t0 = time.perf_counter()
    reps: dict = {}
    n_reused = 0
    done_contact = 0
    last_partial = 0.0
    chunk = st['chunk']
    cancelled = False
    for n_done, jx in enumerate(order):
        if n_done % 16 == 0 and ctx.cancelled():
            cancelled = True
            break
        i, s, q = jobs_list[jx]
        if use_cache:
            # the q-bucket: every grasp in a bucket is evaluated at the bucket's first joint vector (its maps, its
            # camera poses and its contact points); world-fixed points (the object region) keep their own positions
            rep = reps.get(buckets[jx])
            if rep is None:
                reps[buckets[jx]] = q
            else:
                q = rep
                n_reused += 1
        qd = qdict(q)
        fkd = ev.fk(qd)
        Pp, Np, seg = pad_world(pads, fkd[robot.uid])
        Po, No = regions[i]
        P = np.concatenate([Pp, Po]) if len(Po) else Pp
        Nn = np.concatenate([Np, No]) if len(Po) else Np
        sg = np.concatenate([seg, np.full(len(Po), F, np.int32)]) if len(Po) else seg
        full = s == 0 and i in sens_set
        for j in range(C):
            inimg, facing, unocc, front = ev.states(j, fkd, qd, P, Nn, buckets[jx] if use_cache else None,
                                                    use_cache, full=full)
            vis = inimg & facing & unocc
            cnt = np.bincount(sg, minlength=F + 1).astype(float)
            fr = np.bincount(sg, weights=vis.astype(float), minlength=F + 1) / np.maximum(cnt, 1)
            frac[s, i, j] = fr[:F]
            if len(Po):
                objf[s, i, j] = fr[F]
            if s == 0:
                pm = sg < F
                away = inimg & ~vis & ~front                  # behind its own finger: its face turned away
                counts[j] += [int((~inimg[pm]).sum()), int(away[pm].sum()),
                              int((inimg & ~vis & front)[pm].sum()), int(vis[pm].sum())]
                if full:
                    sens_store[(i, j)] = (P, Nn, sg, facing & unocc, q)
        if s == 0:
            done_contact += 1
        if N > BACKGROUND_GRASPS and ctx.partial and (done_contact % chunk == 0 and s == 0) and \
                time.perf_counter() - last_partial > 0.5:
            last_partial = time.perf_counter()
            sc = grasp_score(frac[0, fi], objf[0, fi], st)
            ok_ = np.isfinite(sc)
            ctx.partial({'study': SID, 'partial': True, 'grasps': N, 'evaluated': done_contact, 'feasible': len(fi),
                         'cvs_so_far': _num(float((sc[ok_] >= st['min_visible']).mean()) if ok_.any() else None),
                         'grasps_per_s': _num(done_contact / max(time.perf_counter() - t0, 1e-9), 2)})
        if n_done % 8 == 0:
            ctx.report(0.36 + 0.44 * (n_done + 1) / max(len(order), 1),
                       f'Visibility: {done_contact:,} of {len(fi):,} feasible grasps')
    occ_s = time.perf_counter() - t0
    timings['occlusion_seconds'] = round(occ_s, 4)

    # ---------------------------------------------------------------- measures
    score = grasp_score(frac[0], objf[0], st)
    appl = _applicable(objf[0], st) & feas
    evaluated = appl & np.isfinite(score)
    meets = evaluated & (score >= st['min_visible'])
    n_eval = int(evaluated.sum())
    k_meet = int(meets.sum())
    cvs = k_meet / n_eval if n_eval else None
    lo, hi = bootstrap_ci(meets[evaluated].astype(float), st['bootstrap'], st['ci'],
                          np.random.default_rng([int(st['rng_seed']), 6])) if n_eval else (None, None)
    wl, wh = wilson_interval(k_meet, n_eval, st['ci']) if n_eval else (None, None)
    mv = st['min_visible']
    both_c = (frac[0] >= mv).all(axis=2)                      # (N, C): every fingertip at min_visible in camera c
    rig_both = both_c.any(axis=1)
    cam_docs = []
    for j, c in enumerate(ev.cams):
        f_j = frac[0, feas, j]
        tot = max(int(counts[j].sum()), 1)
        pre = [_num(np.nanmean(frac[k + 1, feas, j].min(axis=1)) if np.isfinite(frac[k + 1, feas, j]).any()
                    else None, 4) for k in range(len(dists))]
        R, t = c.optical(ev.fk(qd0))
        fov = c.cam.fov_deg()
        cam_docs.append({
            'sensor_uid': c.uid, 'name': c.name, 'kind': c.sensor.kind, 'slot': c.sensor.slot,
            'model_id': c.sensor.model_id, 'imager': c.imager.name, 'imager_why': c.why,
            'imager_label': c.imager.label, 'is_default': c.imager.is_default, 'mode': [c.cam.width, c.cam.height],
            'fov_deg': {k: _num(v, 3) for k, v in fov.items()}, 'on_link': c.on_link,
            'parent_frame': c.sensor.mount.parent_frame,
            'position_m': _nums(t, 6), 'axis': _nums(R[:, 2], 6),
            'fingertips': {p['name']: _num(np.nanmean(f_j[:, f]) if len(f_j) else None, 4) for f, p in enumerate(pads)},
            'both_fingertips_rate': _num(both_c[feas, j].mean() if feas.any() else None, 4),
            'object_visible': _num(np.nanmean(objf[0, feas & has_obj, j]) if (feas & has_obj).any() else None, 4),
            'precontact': pre,
            'precontact_mean': _num(np.nanmean([x for x in pre if x is not None]) if any(x is not None for x in pre)
                                    else None, 4),
            'points': {'outside_image': _num(counts[j][0] / tot, 4), 'behind_own_finger': _num(counts[j][1] / tot, 4),
                       'hidden_by_scene': _num(counts[j][2] / tot, 4), 'visible': _num(counts[j][3] / tot, 4)},
        })
    pre_rig = []
    for k in range(len(dists)):
        m_k = feas & pre_ok[k]
        v = frac[k + 1, m_k].min(axis=2).max(axis=1) if m_k.any() else np.zeros(0)
        pre_rig.append({'distance_m': _num(dists[k]), 'feasible': int(m_k.sum()),
                        'best_camera_weaker_fingertip': _num(np.nanmean(v) if len(v) else None, 4),
                        'both_fingertips_rate': _num(((frac[k + 1, m_k] >= mv).all(axis=2).any(axis=1)).mean()
                                                     if m_k.any() else None, 4)})
    reason_counts = {}
    for i in np.flatnonzero(~feas):
        reason_counts[reasons[i] or 'no convergence'] = reason_counts.get(reasons[i] or 'no convergence', 0) + 1
    obj_best = np.nanmax(objf[0], axis=1) if C and has_obj.any() else np.full(N, np.nan)
    pct = None if cvs is None else 100.0 * cvs
    sev, word = status_for(pct, st) if pct is not None else ('critical', 'nothing to score')
    summary = {
        'cvs': _num(cvs), 'cvs_pct': _num(pct, 3), 'ci_low': _num(lo), 'ci_high': _num(hi),
        'ci_level': st['ci'], 'ci_method': f'percentile bootstrap, {st["bootstrap"]} resamples, seed {st["rng_seed"]}',
        'wilson_low': _num(wl), 'wilson_high': _num(wh),
        'meets': k_meet, 'evaluated': n_eval, 'grasps': N, 'feasible': int(feas.sum()),
        'cvs_of_all': _num(k_meet / N if N else None),
        'infeasible': reason_counts,
        'both_fingertips_rate': _num(rig_both[feas].mean() if feas.any() else None, 4),
        'object_visible': _num(np.nanmean(obj_best[feas & has_obj]) if (feas & has_obj).any() else None, 4),
        'with_object': int((feas & has_obj).sum()),
        'precontact': pre_rig,
        'status': sev, 'word': word,
        'requirement': requirement_text(st),
    }

    # ---------------------------------------------------------------- what to move
    t0 = time.perf_counter()
    sens = []
    if sens_on and not cancelled and C:
        ctx.report(0.80, 'What to move: one camera at a time')
        sens = _sensitivity(ev, st, pads, Q, sens_idx, sens_store, frac, objf, qdict, ctx)
    timings['sensitivity_seconds'] = round(time.perf_counter() - t0, 4)
    best = next((m for m in sens if (m['delta_cvs'] or 0) > 0), None)
    if best is None:
        best = next((m for m in sens if (m['delta_score'] or 0) > 1e-9), None)
    summary['advice'] = advice(summary, cam_docs, best, st)

    # ---------------------------------------------------------------- gallery
    t0 = time.perf_counter()
    gallery = []
    if st['gallery'] and not cancelled and evaluated.any():
        ctx.report(0.94, 'Worst-case gallery')
        gallery = _gallery(project, ev, pads, Q, score, evaluated, qdict, st, gsel, notes, robot.uid)
    timings['gallery_seconds'] = round(time.perf_counter() - t0, 4)
    total = time.perf_counter() - t_all
    timings['total_seconds'] = round(total, 4)

    # ---------------------------------------------------------------- the table
    from .coverage import encode_array
    table = {
        'id': [str(p.get('id', '')) for p in gsel['poses']], 'kind': [p.get('kind', '') for p in gsel['poses']],
        'object_uid': [p.get('object_uid', '') for p in gsel['poses']],
        'tcp': encode_array(tcp_w.astype(np.float32)), 'feasible': [bool(x) for x in feas],
        'reason': [r if not ok else '' for r, ok in zip(reasons, feas)],
        'score': _nums(score, 4), 'meets': [bool(x) for x in meets],
        'manipulability': _nums(np.where(feas, b.manipulability, np.nan), 6),
        'q': encode_array(np.where(feas[:, None], Q, np.nan).astype(np.float32)),
        'fingertips': encode_array(np.round(frac[0], 4).astype(np.float32)),
        'object': encode_array(np.round(objf[0], 4).astype(np.float32)),
        'precontact': encode_array(np.round(frac[1:], 4).astype(np.float32)) if len(dists) else None,
    }
    st_out = {k: v for k, v in st.items() if k not in ('q',)}
    from .. import occlusion as OC
    gl = OC.gl_status() if ev.backend == 'gl' else {}
    return {
        'project': project.name, 'variant': project.variant.name,
        'grasp_set': {'uid': gs.get('uid'), 'name': gs.get('name'), 'source': gs.get('source'),
                      'poses': len(poses), 'evaluated': N, 'kinds': GR.counts(gsel)},
        'robot': {'uid': robot.uid, 'name': robot.name, 'link': link, 'joints': list(model.joint_names),
                  'tcp_m': _nums(tcp_off[:3, 3], 6), 'gripper_rule': geom.get('rule', '')},
        'q_others': {u: [float(x) for x in v] for u, v in others.items()}, 'q_source': qsrc,
        'fingertips': [{'name': p['name'], 'link': p['link'], 'points': p['points'], 'moved_to_visual': p['moved'],
                        'rule': p['rule']} for p in pads],
        'precontact_m': [_num(d) for d in dists],
        'settings': st_out, 'summary': summary, 'cameras': cam_docs, 'sensitivity': sens, 'best_move': best,
        'grasps': table, 'gallery': gallery, 'notes': notes, 'partial': bool(cancelled),
        'ik': {'backend': st['ik_backend'], 'rule': st['rule'], 'settings': {k: v for k, v in b.settings.items()},
               'counts': {'ok': int(feas.sum()), **reason_counts},
               'obstacle_pairs': getattr(b, 'obstacle_pairs', {})},
        'occlusion': {'backend': ev.backend, 'renderer': ev.renderer, 'cube_n': st['cube_n'],
                      'gl_how': gl.get('how', ''), 'gl_mode': gl.get('mode', ''),
                      'maps_built': ev.builds, 'map_seconds': round(ev.build_s, 4), 'cache_hits': ev.hits,
                      'cache_misses': ev.misses, 'cache_hit_rate': _num(ev.hit_rate, 4),
                      'cache_reuse_refused': ev.reuse_refused, 'q_bucket_deg': st['q_bucket_deg'],
                      'evaluations': len(jobs_list), 'evaluations_reused': n_reused,
                      'evaluation_reuse_rate': _num(n_reused / len(jobs_list) if jobs_list else None, 4),
                      'q_bucket_mm': st['q_bucket_mm']},
        'throughput': {'grasps_per_s': _num(N / total if total > 0 else None, 3),
                       'occlusion_grasps_per_s': _num(len(fi) / occ_s if occ_s > 0 else None, 3),
                       'maps_per_s': _num(ev.builds / ev.build_s if ev.build_s > 0 else None, 2)},
        'timings': timings,
    }


def _cancelled(project, gs, N) -> dict:
    return {'project': project.name, 'grasp_set': {'uid': gs.get('uid'), 'name': gs.get('name')}, 'partial': True,
            'summary': {'grasps': N}, 'notes': ['Cancelled before the visibility was measured.'], 'cameras': [],
            'sensitivity': [], 'gallery': []}


def requirement_text(st: dict) -> str:
    what = REQUIREMENT_TITLES[st['requirement']]
    cams = 'one camera' if st['min_cameras'] == 1 else f'{st["min_cameras"]} cameras'
    same = '' if st['same_camera'] or st['requirement'] in ('any_fingertip', 'object') else \
        ' (not necessarily the same)'
    return f'{what} at least {100 * st["min_visible"]:.0f} % visible in at least {cams}{same}'


# ====================================================================== what to move
def _sensitivity(ev: Evaluator, st, pads, Q, sidx, store, frac, objf, qdict, ctx) -> list:
    """One-at-a-time moves of each camera (module docstring): [{'camera', 'move', 'delta_cvs', ...}] ranked."""
    from ..frames import Pose
    C, F = len(ev.cams), len(pads)
    base_f, base_o = frac[0, sidx].copy(), objf[0, sidx].copy()
    base_s = grasp_score(base_f, base_o, st)
    appl = _applicable(base_o, st) & np.isfinite(base_s)
    if not appl.any():
        return []
    base_m = (base_s >= st['min_visible']) & appl
    dm, dr = st['sens_step_m'], math.radians(st['sens_step_deg'])
    out = []
    total = C * 12
    done = 0
    g = np.random.default_rng([int(st['rng_seed']), 8])
    for j, c in enumerate(ev.cams):
        moves = []
        for ax in AXES:
            for sgn in (1, -1):
                moves.append(('move', ax, sgn))
        for ax in AXES:
            for sgn in (1, -1):
                moves.append(('turn', ax, sgn))
        for kind, ax, sgn in moves:
            if ctx.cancelled():
                return _rank(out)
            f2, o2 = base_f.copy(), base_o.copy()
            if kind == 'turn':
                Rt = _rot(ax, sgn * dr)
                T_po = c.T_po.copy()
                T_po[:3, :3] = c.T_po[:3, :3] @ Rt
                for a, i in enumerate(sidx):
                    P, Nn, sg, ok, qe = store[(int(i), j)]
                    fkd = ev.fk(qdict(qe))
                    R, o = c.optical(fkd)
                    _uv, inimg = c.cam.project((P - o) @ (R @ Rt))
                    vis = np.asarray(inimg, bool) & ok
                    f2[a, j], o2[a, j] = _fractions(vis, sg, F, o2[a, j])
            else:
                T_po = c.T_po.copy()
                T_po[:3, 3] = c.T_po[:3, 3] + sgn * dm * np.eye(3)[AXES.index(ax)]
                for a, i in enumerate(sidx):
                    P, Nn, sg, _ok, qe = store[(int(i), j)]
                    qd = qdict(qe)
                    fkd = ev.fk(qd)
                    inimg, facing, unocc, _fr = ev.states(j, fkd, qd, P, Nn, None, False, T_po=T_po)
                    f2[a, j], o2[a, j] = _fractions(inimg & facing & unocc, sg, F, o2[a, j])
            s2 = grasp_score(f2, o2, st)
            m2 = (s2 >= st['min_visible']) & appl
            d = m2.astype(float)[appl] - base_m.astype(float)[appl]
            lo, hi = bootstrap_ci(d, 1000, st['ci'], g) if len(d) else (None, None)
            new_pose = Pose.from_matrix(T_po @ np.linalg.inv(c.imager.pose_in_body.matrix()))
            frame = c.sensor.mount.parent_frame if c.on_link else 'world'
            if kind == 'move':
                text = f'move {c.name} {sgn * dm * 100:+.0f} cm along {ax} of {frame}'
            else:
                text = f'{TURNS[ax]} {c.name} {sgn * st["sens_step_deg"]:+.0f}° (about its optical {ax} axis)'
            out.append({'sensor_uid': c.uid, 'camera': c.name, 'kind': kind, 'axis': ax, 'sign': sgn,
                        'step': dm if kind == 'move' else st['sens_step_deg'],
                        'unit': 'm' if kind == 'move' else 'deg', 'frame': frame, 'text': text,
                        'cvs': _num(m2[appl].mean()), 'base_cvs': _num(base_m[appl].mean()),
                        'delta_cvs': _num(d.mean()), 'delta_low': _num(lo), 'delta_high': _num(hi),
                        'delta_score': _num(np.nanmean(s2[appl] - base_s[appl]), 5), 'grasps': int(appl.sum()),
                        'mount_pose': new_pose.to_dict()})
            done += 1
            ctx.report(0.80 + 0.14 * done / total, f'What to move: {c.name}')
    return _rank(out)


def _fractions(vis, sg, F, obj_old):
    cnt = np.bincount(sg, minlength=F + 1).astype(float)
    fr = np.bincount(sg, weights=vis.astype(float), minlength=F + 1) / np.maximum(cnt, 1)
    return fr[:F], (fr[F] if cnt[F] else obj_old)


def _rank(out: list) -> list:
    out.sort(key=lambda m: (-(m['delta_cvs'] or 0.0), -(m['delta_score'] or 0.0)))
    for k, m in enumerate(out):
        m['rank'] = k + 1
    return out


# ====================================================================== gallery
def _gallery(project, ev: Evaluator, pads, Q, score, evaluated, qdict, st, gsel, notes, robot_uid) -> list:
    """Camera view renders of the lowest-scoring grasps (module docstring)."""
    from .. import camrender as CR
    from ..frames import Pose
    from .coverage import encode_array
    idx = np.flatnonzero(evaluated)
    fr_mean = np.nan_to_num(score[idx], nan=-1.0)
    worst = idx[np.argsort(fr_mean, kind='stable')][:st['gallery']]
    out = []
    W = st['gallery_px']
    for i in worst:
        qd = qdict(Q[i])
        fkd = ev.fk(qd)
        try:
            geom = CR.gather(project, qd)
        except Exception as e:                       # noqa: BLE001 — a render must not stop the study
            notes.append(f'The gallery could not draw the scene: {e}')
            return out
        Pp, Np, seg = pad_world(pads, fkd[robot_uid])
        views = []
        for j, c in enumerate(ev.cams[:MAX_GALLERY_CAMERAS]):
            R, o = c.optical(fkd)
            pose = Pose.from_matrix(np.r_[np.c_[R, o], [[0, 0, 0, 1]]])
            H = max(16, int(round(W * c.cam.height / c.cam.width)))
            g = CR.without_seat(geom, project, c.sensor, o[None]) if c.on_link else geom
            try:
                rnd = CR.render(g, c.cam, pose, W, H)
            except Exception as e:                   # noqa: BLE001
                notes.append(f'The gallery could not render {c.name}: {e}')
                continue
            inimg, facing, unocc, front = ev.states(j, fkd, qd, Pp, Np, None, False)
            seen = inimg & facing & unocc
            state = np.where(~inimg, ST_OUT, np.where(seen, ST_SEEN, np.where(front, ST_HIDDEN, ST_AWAY)))
            uv, _ok, _d = CR.project_world(rnd.camera, pose, Pp)
            views.append({'sensor_uid': c.uid, 'camera': c.name, 'width': W, 'height': H,
                          'rgb': encode_array(rnd.rgb.astype(np.uint8)),
                          'uv': encode_array(np.nan_to_num(np.round(uv, 1), nan=-1e4).astype(np.float32)),
                          'state': encode_array(state.astype(np.uint8)), 'segment': encode_array(seg.astype(np.uint8)),
                          'fingertips': _nums(np.bincount(seg, weights=(state == ST_SEEN).astype(float),
                                                          minlength=len(pads)) /
                                              np.maximum(np.bincount(seg, minlength=len(pads)), 1), 4)})
        p = gsel['poses'][i]
        out.append({'index': int(i), 'id': str(p.get('id', '')), 'kind': p.get('kind', ''), 'score': _num(score[i], 4),
                    'q': [float(x) for x in Q[i]], 'views': views})
    return out


# ====================================================================== decode and advice
def decode(doc: dict) -> dict:
    """The per-grasp arrays of a result: {'tcp' (N, 3), 'q' (N, nq) (NaN when infeasible), 'fingertips' (N, C, F),
    'object' (N, C), 'precontact' (S, N, C, F) or None, 'score' (N,), 'feasible' (N,), 'meets' (N,)}."""
    from .coverage import decode_array
    t = doc['grasps']
    return {'tcp': decode_array(t['tcp']).astype(float), 'q': decode_array(t['q']).astype(float),
            'fingertips': decode_array(t['fingertips']).astype(float),
            'object': decode_array(t['object']).astype(float),
            'precontact': decode_array(t['precontact']).astype(float) if t.get('precontact') else None,
            'score': np.array([np.nan if v is None else v for v in t['score']], float),
            'feasible': np.asarray(t['feasible'], bool), 'meets': np.asarray(t['meets'], bool)}


def decode_view(view: dict) -> dict:
    """A gallery view's image (H, W, 3) uint8, the contact points' pixels (K, 2), states (K,) and fingertips (K,)."""
    from .coverage import decode_array
    return {'rgb': decode_array(view['rgb']), 'uv': decode_array(view['uv']).astype(float),
            'state': decode_array(view['state']), 'segment': decode_array(view['segment'])}


def _pct(x) -> str:
    return '—' if x is None else f'{100 * x:.0f} %'


def advice(summary: dict, cams: list, best: dict | None, st: dict) -> str:
    """One or two plain sentences: what the score means and the change that would help most."""
    if not summary.get('evaluated'):
        if not summary.get('feasible'):
            return ('No grasp of the set is reachable without a collision, so nothing could be scored: check the '
                    'reasons, the grasp set or the robot\'s placement.')
        return 'No feasible grasp has what the requirement needs (an object region): choose another requirement.'
    text = (f'{_pct(summary["cvs"])} of the {summary["evaluated"]:,} reachable grasps have '
            f'{summary["requirement"]}.')
    if cams:
        worst = min(cams, key=lambda c: (c['points']['visible'] or 0.0))
        pts = worst['points']
        why = max((('outside its image', pts['outside_image'] or 0.0),
                   ('on the far side of their own finger', pts['behind_own_finger'] or 0.0),
                   ('hidden by the robot, the objects or the tables', pts['hidden_by_scene'] or 0.0)),
                  key=lambda t: t[1])
        text += f' {worst["name"]} sees {_pct(pts["visible"])} of the contact points; most of the rest are {why[0]}.'
    if summary.get('cvs') == 1.0:
        text += ' Every reachable grasp meets the requirement.'
    elif best is not None:
        if (best['delta_cvs'] or 0) > 0:
            text += f' Best single change: {best["text"]} ({100 * best["delta_cvs"]:+.0f} points on the score'
            if best.get('delta_low') is not None and best['delta_low'] <= 0:
                text += f'; its {100 * st["ci"]:.0f} % interval reaches zero, so it may be noise'
            text += ').'
        else:
            text += f' No single small move raises the score; {best["text"]} raises the visible share most.'
    return text
