"""Motion blur and rolling shutter (spec §7.9): how far the image of the scene and of the gripper moves while a camera
exposes one frame, and how far it moves between the first and the last row of a rolling shutter, along each
trajectory of the project.

    @jobs.study('motion_blur', 'Motion blur and rolling shutter', batch='motion-blur') → --motion-blur OUT.json
    run(project, ctx) -> dict              the study (also core.jobs.run_study('motion_blur', project))
    settings_of(project) -> dict           the settings with their defaults (DEFAULTS), and which are defaults
    imaging(project, st) -> [dict]         per camera imager: its lens at the stream size, frame rate, shutter,
                                           exposure and readout and where each comes from (time_sync.py reads it)
    motions(project, st) -> (list, str, notes)   the trajectories analysed (or grasp approaches when there are none)
    image_velocity(cam, R, o, v_c, w_c, P, v_P)  pixel positions and velocities of world points
    projection_jacobian(cam, Pc)           d(u, v)/d(x, y, z) in the optical frame
    rolling_capture(uv, uv_dot, height, readout_s)   where a rolling shutter records a moving image point
    suggest(result)                        the change the page's Apply makes (or None)

**Image-plane velocity.** A camera's optical frame (z forward, x right, y down) sits in the world at position o with
rotation R (its columns the optical axes in world coordinates). A camera on a robot link moves with the link's twist
at the trajectory's joint velocity: v_c = J_v(q) q̇ and ω_c = J_ω(q) q̇, where J(q) is MuJoCo's world-frame Jacobian of
the camera's optical centre fixed in the link (core/robot/model.py RobotModel.jacobian); a camera on a tripod or the
cage has v_c = ω_c = 0. A world point P moving with velocity v_P (0 for the scene; J_v q̇ of its link for the gripper
and the fingertip pads) is at P_c = Rᵀ (P − o) in the optical frame, and

    Ṗ_c = Rᵀ (v_P − v_c − ω_c × (P − o))                         (the derivative of Rᵀ(P − o) with Ṙ = [ω_c]× R)
    (u̇, v̇) = ∂π/∂P_c · Ṗ_c                                      π the imager's lens model (core/optics)

with ∂π/∂P_c written out for a pinhole (fx/z, −fx x/z², ...) and taken by central differences of π in the optical
frame for the other lens models. q̇ is the slope of the trajectory between its samples (it is linear between them,
core/robot/trajectory.py), so q̇ is constant over each interval.

**Smear** = |(u̇, v̇)| × exposure (pixels): the length of the streak a point draws during one exposure, to first order
in the exposure (the image velocity is taken as constant over it). **Rolling-shutter skew** = |(u̇, v̇)| × readout:
how far a point's image moves between the first and the last row being read (Liang et al. 2008). A global shutter has
no skew. Rows are read at t(r) = readout · r/(H − 1) after the first; a point at (u₀, v₀) when the first row is read,
moving at (u̇, v̇), is recorded where its row is being read: t* = v₀ / ((H − 1)/readout − v̇), at (u₀, v₀) + (u̇, v̇) t*
(rolling_capture).

**Exposure and readout come from the library entry** (core/library, every figure cited), the user's own figures
(settings 'exposure_s' and 'readout_s' {sensor uid: {'value', 'source'}}, e.g. what the driver is set to) taking
precedence. When no exposure is set, the longest the stream allows is used as an upper bound: the frame period
1 / fps, or the maker's stated maximum exposure when that is shorter; the result says "upper bound". A rolling
shutter whose readout the maker does not state is bounded the same way (the rows of one frame are read within one
frame period at a steady frame rate); the skew is then "at most". Neither bound is a device figure. A sensor without a
library device has no frame rate: its smear needs your exposure; its image speed and the maximum exposure are still
given.

**What is seen.** Fixed points of the scene (a grid of volume_grid³ points over each task volume and object_points
points on each task object's surface) and moving points of the trajectory's robot (its TCP and its fingertip pads'
points, core/grasps.py), each counted for a camera while it projects inside that camera's image in front of it. A
wrist camera sees the scene stream past; a fixed camera sees the gripper move. Occlusion is not tested.

**Report.** Per trajectory and per segment (the intervals between the trajectory's samples, merged into at most
max_segments equal spans), per camera imager: the worst smear and skew and where they happen; per camera its worst
frames (the times of the largest image speed) with every seen point's smear vector, which the Camera view draws.
The recommendation: the longest exposure that keeps the smear within the limit (smear_limit_px, 1 px by default, a
DexVantage default, not a standard), max_exposure = limit / peak image speed, with the predicted smear; when that is
shorter than the shortest exposure the maker states, a slower motion (by the factor needed); a global-shutter camera
when the rolling-shutter skew is over the limit (skew 0). Apply records the suggested exposure as your exposure for
that sensor (settings 'exposure_s', one undo step): set the same on the camera.

**Without a trajectory** the study uses the approaches of the first grasp set: for up to max_grasps grasps, the arm
moves along a joint-linear path from the pre-grasp pose (pre_grasp_m, 0.10 m back along the approach, a default) to
contact at approach_speed_m_s (0.10 m/s along the approach, a default), both poses solved by IK (core/robot/ik.py).

Settings (project.analysis.studies['motion_blur'], every key optional; DEFAULTS): smear_limit_px, step_s, max_segments,
volume_grid, object_points, pre_grasp_m, approach_speed_m_s, max_grasps, frames_kept, vectors_kept, exposure_s,
readout_s, trajectories (uids; empty or missing = every trajectory).

Validation (dexvantage/validation/checks_blur.py, model 'timing'): the image-plane velocity against central finite
differences in time of the points projected at q(t ± h) (forward kinematics, no Jacobian); the smear of a point moving
parallel to the image plane against f·v·E/Z; the rolling-shutter position against a brute-force simulation that reads
the rows one at a time and finds the row each moving point is in when it is read.

Known limits: smear to first order in the exposure (the image velocity is constant over one exposure; at the joint
velocity jumps between trajectory samples it changes at once); occlusion is not tested, so a point counted may be
hidden behind the gripper; the exposure bound is the frame period when no exposure is set, which overstates the smear
of a camera whose auto-exposure runs shorter; a shorter exposure needs more light or gain (the lighting study treats
light); the robot's other joints and other robots stay at home; the object being carried is not moved with the
gripper (its points are counted as scene points); rolling-shutter rows are read top to bottom at a constant rate;
lens distortion of non-pinhole models enters through a numerical derivative; vibration of the mount is not modelled.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from .. import jobs

STUDY_ID = 'motion_blur'
DEFAULTS = {
    'smear_limit_px': 1.0,          # the most smear accepted (a DexVantage default, not a standard)
    'step_s': 0.01,                 # sampling step along a trajectory
    'max_segments': 24,             # trajectory intervals are merged into at most this many spans
    'volume_grid': 4,               # scene points per edge of each task volume's grid
    'object_points': 40,            # scene points on each task object's surface
    'pre_grasp_m': 0.10,            # without a trajectory: the pre-grasp distance along the approach
    'approach_speed_m_s': 0.10,     # without a trajectory: the approach speed
    'max_grasps': 8,                # without a trajectory: grasps of the first set used
    'frames_kept': 3,               # worst frames kept per camera imager
    'vectors_kept': 150,            # smear vectors kept per frame
}
LIMITS = {'smear_limit_px': (0.01, 1000.0), 'step_s': (0.0005, 1.0), 'max_segments': (1, 500), 'volume_grid': (2, 12),
          'object_points': (0, 500), 'pre_grasp_m': (0.0, 2.0), 'approach_speed_m_s': (0.001, 10.0),
          'max_grasps': (1, 200), 'frames_kept': (1, 20), 'vectors_kept': (1, 2000)}
INTS = ('max_segments', 'volume_grid', 'object_points', 'max_grasps', 'frames_kept', 'vectors_kept')
MAX_SAMPLES = 20000                 # time samples per trajectory at most (the step grows to fit)
WORDS = {'good': 'sharp', 'warning': 'some blur', 'serious': 'blurred', 'unknown': 'exposure not set'}
# when the exposure is only bounded (the frame period, or the maker's stated maximum) a smear over the limit is the
# worst case, not a finding: the word says so (the severity stays that of the bound)
BOUND_WORDS = {'warning': 'may blur', 'serious': 'may be blurred'}


def _word(status: str, bound: bool) -> str:
    return BOUND_WORDS.get(status, WORDS[status]) if bound else WORDS[status]
REFERENCES = [
    'C.-K. Liang, L.-W. Chang and H. H. Chen, "Analysis and compensation of rolling shutter effect", IEEE Trans. '
    'Image Processing 17(8), 1323-1330, 2008.',
    'E. Todorov, T. Erez and Y. Tassa, "MuJoCo: A physics engine for model-based control", IEEE/RSJ IROS 2012 '
    '(mj_jac, the world-frame Jacobian).',
    'K. M. Lynch and F. C. Park, Modern Robotics, Cambridge University Press, 2017 (twists, chapter 3).',
]
ASSUMPTIONS = (
    'The image velocity is constant over one exposure (first order in the exposure).',
    'No exposure set: the frame period 1/fps (or the maker\'s stated maximum, when shorter) is used as an upper bound.',
    'A rolling shutter whose readout is not stated is bounded by the frame period (rows of one frame are read within '
    'one frame period).',
    'Occlusion is not tested; other joints and other robots stay at home.',
)


# ====================================================================== settings
def _sourced(d) -> dict:
    """{uid: {'value': positive seconds, 'source': text}} from a stored map (bad entries dropped)."""
    out = {}
    for k, v in (d or {}).items() if isinstance(d, dict) else ():
        if isinstance(v, dict):
            try:
                x = float(v.get('value'))
            except (TypeError, ValueError):
                continue
            if math.isfinite(x) and x > 0:
                out[str(k)] = {'value': x, 'source': str(v.get('source') or '')}
    return out


def settings_of(project) -> dict:
    """The study's settings: the stored ones (clamped) over DEFAULTS; 'defaults' lists the keys left at default."""
    raw = dict((project.analysis.studies.get(STUDY_ID) or {}) if project is not None else {})
    st, defaults = {}, []
    for k, dv in DEFAULTS.items():
        v = raw.get(k)
        try:
            x = float(v)
            ok = math.isfinite(x)
        except (TypeError, ValueError):
            ok = False
        if v is None or isinstance(v, bool) or not ok:
            st[k] = dv
            defaults.append(k)
            continue
        lo, hi = LIMITS[k]
        x = min(max(x, lo), hi)
        st[k] = int(round(x)) if k in INTS else x
    st['exposure_s'] = _sourced(raw.get('exposure_s'))
    st['readout_s'] = _sourced(raw.get('readout_s'))
    tr = raw.get('trajectories')
    st['trajectories'] = [str(u) for u in tr if u] if isinstance(tr, (list, tuple)) else []
    st['defaults'] = defaults
    return st


# ====================================================================== geometry
def projection_jacobian(cam, Pc) -> np.ndarray:
    """(N, 2, 3): ∂(u, v)/∂(x, y, z) of optical-frame points Pc (N, 3) through the camera model cam (core/optics
    CameraModel). Pinhole: written out; other models: central differences of cam.project (step 1e-6 of the distance)."""
    P = np.asarray(Pc, float).reshape(-1, 3)
    if cam.model == 'pinhole':
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            iz = 1.0 / z
            J = np.zeros((len(P), 2, 3))
            J[:, 0, 0] = cam.fx * iz
            J[:, 0, 2] = -cam.fx * x * iz * iz
            J[:, 1, 1] = cam.fy * iz
            J[:, 1, 2] = -cam.fy * y * iz * iz
        return J
    h = 1e-6 * np.maximum(np.linalg.norm(P, axis=1), 1e-3)
    J = np.zeros((len(P), 2, 3))
    for k in range(3):
        d = np.zeros_like(P)
        d[:, k] = h
        up, _ = cam.project(P + d)
        um, _ = cam.project(P - d)
        J[:, :, k] = (up - um) / (2.0 * h[:, None])
    return J


def image_velocity(cam, R, o, v_c, w_c, P, v_P):
    """World points P (N, 3) moving at v_P (N, 3), seen by a camera whose optical frame has rotation R (3 × 3, columns
    the optical axes in the world) at o (3,), moving at v_c (3,) and turning at w_c (3,) rad/s, all in the world:
    (uv (N, 2) px, uv_dot (N, 2) px/s, ok (N,) in the image and in front, depth (N,) m along the axis)."""
    R = np.asarray(R, float)
    rel = np.asarray(P, float).reshape(-1, 3) - np.asarray(o, float)
    Pc = rel @ R
    vrel = np.asarray(v_P, float).reshape(-1, 3) - np.asarray(v_c, float) - np.cross(np.asarray(w_c, float), rel)
    Pc_dot = vrel @ R
    uv, ok = cam.project(Pc)
    J = projection_jacobian(cam, Pc)
    uvd = np.einsum('nij,nj->ni', J, Pc_dot)
    ok = ok & np.isfinite(uvd).all(axis=1) & (Pc[:, 2] > 0)
    return uv, uvd, ok, Pc[:, 2]


def rolling_capture(uv, uv_dot, height: int, readout_s: float):
    """Where a rolling shutter (rows read top to bottom, row r at readout · r/(H − 1) after the first) records image
    points that are at uv (N, 2) when the first row is read and move at uv_dot (N, 2) px/s: (captured uv (N, 2), the
    time each is read t* (N,) s). A point moving with its row (v̇ ≥ (H − 1)/readout) is never caught: NaN."""
    uv = np.asarray(uv, float).reshape(-1, 2)
    ud = np.asarray(uv_dot, float).reshape(-1, 2)
    if readout_s <= 0 or height < 2:
        return uv.copy(), np.zeros(len(uv))
    rate = (height - 1) / float(readout_s)                 # rows per second
    with np.errstate(divide='ignore', invalid='ignore'):
        t = uv[:, 1] / (rate - ud[:, 1])
    t = np.where((rate - ud[:, 1]) > 0, t, np.nan)
    return uv + ud * t[:, None], t


# ====================================================================== imagers
def _entry(sensor):
    from .. import library as LB
    return LB.get(sensor.model_id) if sensor.model_id else None


def _stream_choice(project, sensor, entry) -> list:
    """[(imager name, width, height, fps)] the sensor streams: its link's streams on the bandwidth page, else the
    default selection of the bandwidth study (core/studies/bandwidth.py)."""
    from . import bandwidth as BW
    if entry is None:
        return []
    rig = project.rig if isinstance(getattr(project, 'rig', None), dict) else {}
    link = next((k for k in rig.get('links') or [] if k.get('sensor_uid') == sensor.uid), None)
    ss = BW.sensor_streams(sensor, entry, (link or {}).get('stream_ids') or [], BW.settings_of(project))
    out, seen = [], set()
    for s in ss['streams']:
        if s.get('extra') or not s.get('width'):
            continue
        im = entry.imager(s['imager'])
        if im is None or im['name'] in seen:
            continue
        seen.add(im['name'])
        out.append((im['name'], int(s['width']), int(s['height']), float(s['fps'])))
    return out


def imaging(project, st: dict | None = None) -> list:
    """Per camera imager of the active variant (cameras and depth sensors switched on): {'key' ('<uid>|<imager>'),
    'sensor' (project Sensor), 'imager' (library Imager: lens, pose in the body), 'name', 'stream' [w, h], 'fps' (None:
    not known), 'shutter' ('global' | 'rolling' | None), 'exposure' {'value', 'kind' ('yours' | 'upper bound' |
    'unknown'), 'source', 'stated_min', 'stated_max', 'stated_what'}, 'readout' {'value', 'kind' ('yours' | 'stated' |
    'upper bound' | 'none' (global shutter) | 'unknown'), 'source'} (the rolling-shutter skew), 'frame_readout' (the
    same for the time the frame takes to be read out, global shutters included), 'label'}."""
    from .. import library as LB
    st = st or settings_of(project)
    out = []
    for s in project.variant.sensors:
        if not s.enabled or s.kind not in ('camera', 'depth'):
            continue
        entry = _entry(s)
        choice = _stream_choice(project, s, entry)
        mode = {n: (w, h) for n, w, h, _f in choice} or None
        try:
            ims = LB.sensor_imagers(s, mode)
        except LB.LibraryError:
            continue
        if choice:
            fps_of = {n: f for n, _w, _h, f in choice}
            ims = [im for im in ims if im.name in fps_of]
        else:
            fps_of = {}
            ims = ims[:1]
        for im in ims:
            d = (entry.imager(im.name) if entry is not None else None) or {}
            fps = fps_of.get(im.name)
            shutter = d.get('shutter') if entry is not None else None
            ex = d.get('exposure_s') if isinstance(d.get('exposure_s'), dict) else {}
            smin, smax = ex.get('min'), ex.get('max')
            mine = st['exposure_s'].get(s.uid)
            if mine is not None:
                exp = {'value': mine['value'], 'kind': 'yours', 'source': mine['source'] or 'your figure, no source '
                                                                                              'given'}
            elif fps:
                period = 1.0 / fps
                if smax is not None and smax < period:
                    exp = {'value': float(smax), 'kind': 'upper bound',
                           'source': f'no exposure set: the longest the maker states ({ex.get("what") or "library"})'}
                else:
                    exp = {'value': period, 'kind': 'upper bound',
                           'source': f'no exposure set: the frame period at {fps:g} fps, the longest the stream allows '
                                     '(an upper bound, not a device figure)'}
            else:
                exp = {'value': None, 'kind': 'unknown',
                       'source': 'no exposure set and no frame rate known (no library device): enter the exposure'}
            exp.update({'stated_min': smin, 'stated_max': smax, 'stated_what': ex.get('what') or ''})
            ro_mine = st['readout_s'].get(s.uid)
            ro_lib = d.get('readout_s') if isinstance(d.get('readout_s'), dict) else None
            if ro_mine is not None:
                frame_ro = {'value': ro_mine['value'], 'kind': 'yours', 'source': ro_mine['source'] or 'your figure'}
            elif ro_lib is not None and ro_lib.get('value'):
                frame_ro = {'value': float(ro_lib['value']), 'kind': 'stated',
                            'source': ro_lib.get('what') or 'library'}
            elif shutter in ('rolling', 'global') and fps:
                frame_ro = {'value': 1.0 / fps, 'kind': 'upper bound',
                            'source': f'{shutter} shutter, readout not stated by the maker: at most the frame '
                                      f'period at {fps:g} fps'}
            else:
                why = (f'{shutter} shutter, readout not stated by the maker and no frame rate known' if shutter else
                       'the shutter type is not stated by the maker' if entry is not None else
                       'no library device: shutter and readout not known')
                frame_ro = {'value': None, 'kind': 'unknown', 'source': why}
            # the skew: a global shutter exposes every row at once, whatever its readout takes afterwards
            ro = ({'value': 0.0, 'kind': 'none', 'source': 'global shutter: every row is exposed at once'}
                  if shutter == 'global' else frame_ro)
            out.append({'key': f'{s.uid}|{im.name}', 'sensor': s, 'imager': im, 'name': f'{s.name} · {im.name}',
                        'stream': [im.camera.width, im.camera.height], 'fps': fps, 'shutter': shutter,
                        'exposure': exp, 'readout': ro, 'frame_readout': frame_ro, 'label': im.label,
                        'device': entry.title if entry is not None else 'no device chosen'})
    return out


# ====================================================================== motions
def _qd_rows(traj: dict, names) -> tuple:
    """(T (K,), Qm (K, nq) on the model's joints, slopes (K−1, nq)) of a trajectory; joints it does not name: 0."""
    T = np.asarray(traj.get('t') or [], float)
    Q = np.asarray(traj.get('q') or [], float).reshape(len(T), -1)
    idx = {n: i for i, n in enumerate(names)}
    cols = [(c, idx[j]) for c, j in enumerate(traj.get('joints') or []) if j in idx]
    Qm = np.zeros((len(T), len(names)))
    for c, i in cols:
        Qm[:, i] = Q[:, c]
    dT = np.diff(T)
    S = np.diff(Qm, axis=0) / np.where(dT > 0, dT, 1.0)[:, None] if len(T) > 1 else np.zeros((0, len(names)))
    mask = np.zeros(len(names), bool)
    for _c, i in cols:
        mask[i] = True
    return T, Qm, S, mask


def _grasp_approaches(project, st, notes) -> list:
    from .. import grasps as GR
    from ..robot import ik as IK
    from ..robot.model import RobotModelError, get_model
    sets = GR.sets(project)
    if not sets:
        return []
    gs = sets[0]
    poses = list(gs.get('poses') or [])[:st['max_grasps']]
    if not poses:
        return []
    ruid = gs.get('robot_uid') or (project.primary_robot.uid if project.primary_robot else '')
    try:
        m = get_model(project, ruid)
    except RobotModelError as e:
        notes.append(f'The grasp approaches could not be used: {e}')
        return []
    sub = dict(gs, poses=poses)
    link, tcp, T = GR.ik_targets(project, sub)
    A = []
    for g, Tk in zip(poses, T):
        a = np.asarray(g.get('approach') or Tk[:3, 2], float)
        A.append(a / max(np.linalg.norm(a), 1e-300))
    A = np.array(A)
    Tpre = T.copy()
    Tpre[:, :3, 3] -= st['pre_grasp_m'] * A
    rc = IK.solve_batch(m, link, T, seeds=4, rng=0, tcp=tcp, collision=False)
    rp = IK.solve_batch(m, link, Tpre, q0=rc.q, seeds=4, rng=0, tcp=tcp, collision=False, rule='nearest_q0')
    dur = max(st['pre_grasp_m'] / st['approach_speed_m_s'], 1e-3)
    out = []
    for k, g in enumerate(poses):
        if not (rc.ok[k] and rp.ok[k]):
            continue
        out.append({'uid': f'approach-{g.get("id", k)}', 'name': f'Approach to grasp {g.get("id", k)}',
                    'robot_uid': ruid, 'joints': list(m.joint_names), 't': [0.0, dur],
                    'q': [rp.q[k].tolist(), rc.q[k].tolist()], 'synthetic': True})
    if len(out) < len(poses):
        notes.append(f'{len(poses) - len(out)} of {len(poses)} grasp approaches have no IK solution and are left '
                     'out.')
    return out


def motions(project, st: dict | None = None) -> tuple:
    """(trajectories, source, notes): the project's trajectories (those named in settings 'trajectories', else all),
    or when it has none the approaches of its first grasp set (module docstring); source 'trajectories', 'grasp
    approaches' or 'none'."""
    st = st or settings_of(project)
    notes = []
    trajs = [t for t in project.tasks.trajectories if isinstance(t, dict) and project.robot(t.get('robot_uid', ''))
             and len(t.get('t') or []) >= 2]
    if st['trajectories']:
        chosen = [t for t in trajs if t.get('uid') in st['trajectories']]
        if chosen:
            trajs = chosen
        else:
            notes.append('None of the chosen trajectories is in the project: every trajectory is used.')
    if trajs:
        return trajs, 'trajectories', notes
    ap = _grasp_approaches(project, st, notes)
    if ap:
        notes.append(f'The project has no trajectory: the study uses {len(ap)} grasp approaches of the first grasp '
                     f'set ({st["pre_grasp_m"] * 1000:g} mm back along the approach, at '
                     f'{st["approach_speed_m_s"] * 1000:g} mm/s'
                     + (', defaults' if {'pre_grasp_m', 'approach_speed_m_s'} <= set(st['defaults']) else '') + ').')
        return ap, 'grasp approaches', notes
    return [], 'none', notes


def _segments(T, max_segments: int) -> list:
    """[(t0, t1)]: the trajectory's intervals, merged into at most max_segments equal time spans."""
    if len(T) - 1 <= max_segments:
        return [(float(T[k]), float(T[k + 1])) for k in range(len(T) - 1)]
    edges = np.linspace(T[0], T[-1], max_segments + 1)
    return [(float(edges[k]), float(edges[k + 1])) for k in range(max_segments)]


# ====================================================================== points
def scene_points(project, st) -> tuple:
    """(P (N, 3) world, labels [N]) of the fixed scene: a grid over each task volume and points on each object."""
    from .. import grasps as GR
    P, labels = [], []
    n = st['volume_grid']
    for v in project.workspace.volumes:
        lo, hi = v.bounds()
        g = [np.linspace(lo[i], hi[i], n) for i in range(3)]
        G = np.stack(np.meshgrid(*g, indexing='ij'), axis=-1).reshape(-1, 3)
        P.append(G)
        labels += [f'task volume {v.name}'] * len(G)
    for o in project.workspace.objects:
        if st['object_points'] <= 0:
            break
        try:
            X = GR.object_points(project, o.uid, st['object_points'], rng_seed=0)
        except Exception:                               # noqa: BLE001 — an object without a surface: no points
            continue
        if len(X):
            P.append(np.asarray(X, float).reshape(-1, 3))
            labels += [o.name] * len(X)
    return (np.concatenate(P) if P else np.zeros((0, 3))), labels


def robot_points(project, robot_uid: str, model) -> list:
    """[(link, points (K, 3) in the link frame, label)]: the robot's TCP and its fingertip pads (core/grasps.py)."""
    from .. import grasps as GR
    out = []
    try:
        geom = GR.gripper_geometry(project, robot_uid)
        out.append((geom['frame'], np.asarray(geom['tcp'], float).reshape(1, 3), 'TCP'))
    except Exception:                                   # noqa: BLE001 — no gripper frame: no TCP point
        geom = None
    try:
        pads = GR.contact_patches(project, robot_uid, model.home)
    except Exception:                                   # noqa: BLE001
        pads = {}
    Th = model.fk(model.home)
    for name, pd in pads.items():
        link = pd['link']
        Ti = np.linalg.inv(Th[link])
        X = np.asarray(pd['points'], float).reshape(-1, 3)
        if len(X) > 9:                                   # a few points per pad: its corners and centre
            c = X.mean(axis=0)
            far = X[np.argsort(-np.linalg.norm(X - c, axis=1))[:8]]
            X = np.vstack([c, far])
        out.append((link, X @ Ti[:3, :3].T + Ti[:3, 3], name))
    return out


# ====================================================================== the study
def _r(x, nd=6):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def _status(smear, limit) -> str:
    if smear is None:
        return 'unknown'
    if smear <= limit + 1e-12:
        return 'good'
    return 'warning' if smear <= 2.0 * limit else 'serious'


def _cam_pose(project, model, robot_uid, q, sensor, im):
    """(R (3 × 3), o (3,), link or None, offset in the link (4 × 4) or None): the imager's optical frame in the world
    at q of robot robot_uid (other robots at home)."""
    from ..robot.model import get_model
    B = sensor.mount.pose.matrix() @ im.pose_in_body.matrix()
    if not sensor.mount.on_link:
        return B[:3, :3], B[:3, 3], None, None
    mr = sensor.mount.robot_uid
    if mr == robot_uid:
        Tl = model.link_pose(q, sensor.mount.parent_frame)
        T = Tl @ B
        return T[:3, :3], T[:3, 3], sensor.mount.parent_frame, B
    m2 = get_model(project, mr)
    T = m2.link_pose(m2.home, sensor.mount.parent_frame) @ B
    return T[:3, :3], T[:3, 3], None, None


def camera_twist(project, model, robot_uid, q, qd, sensor, im) -> tuple:
    """(R, o, v_c, w_c): the imager's optical frame in the world at q of robot robot_uid and its linear and angular
    velocity at joint velocity qd: J(q) q̇ of its optical centre when it rides on that robot, else zero."""
    R, o, link, B = _cam_pose(project, model, robot_uid, q, sensor, im)
    if link is None:
        return R, o, np.zeros(3), np.zeros(3)
    J = model.jacobian(q, link, point=B[:3, 3])
    return R, o, J[:3] @ qd, J[3:] @ qd


def moving_points(model, q, qd, moving) -> tuple:
    """([world points (K, 3)], [their velocities (K, 3)]) of robot_points' groups at q, q̇: one Jacobian per link,
    v = v_o + ω × (P − o) for a point fixed in the link."""
    Pw, Vw = [], []
    if not moving:
        return Pw, Vw
    Tall = model.fk(q)
    for link, X, _lab in moving:
        Tl = Tall[link]
        J = model.jacobian(q, link)
        vo, w = J[:3] @ qd, J[3:] @ qd
        W = X @ Tl[:3, :3].T + Tl[:3, 3]
        Pw.append(W)
        Vw.append(vo + np.cross(w, W - Tl[:3, 3]))
    return Pw, Vw


def joint_state(traj: dict, model, t: float) -> tuple:
    """(q, q̇) of a trajectory at time t on the model's joints: linear between samples, the slope of the interval t
    is in (the left one at a sample); joints the trajectory does not name at home and still."""
    T, Qm, S, mask = _qd_rows(traj, model.joint_names)
    base = np.array(model.home, float)
    k = int(np.clip(np.searchsorted(T, t, side='right') - 1, 0, len(T) - 2))
    a = (t - T[k]) / (T[k + 1] - T[k]) if T[k + 1] > T[k] else 0.0
    a = min(max(a, 0.0), 1.0)
    return np.where(mask, Qm[k] * (1 - a) + Qm[k + 1] * a, base), np.where(mask, S[k], 0.0)


def analyse(project, ctx: jobs.Ctx | None = None) -> dict:
    """The study's result without the job wrapper (module docstring)."""
    from ..robot.model import RobotModelError, get_model
    ctx = ctx or jobs.Ctx()
    st = settings_of(project)
    limit = st['smear_limit_px']
    cams = imaging(project, st)
    trajs, source, notes = motions(project, st)
    if not cams:
        raise jobs.StudyError('There is no camera in the rig to analyse: add a camera or depth sensor (Sensors ▸ Add '
                              'camera) and switch it on.')
    P_scene, L_scene = scene_points(project, st)
    acc = {c['key']: {'peak': 0.0, 'worst': None, 'frames': []} for c in cams}
    traj_out = []
    for ti, tr in enumerate(trajs):
        if ctx.cancelled():
            break
        ruid = tr['robot_uid']
        try:
            m = get_model(project, ruid)
        except RobotModelError as e:
            notes.append(f'{tr.get("name", "A trajectory")} is left out: {e}')
            continue
        T, Qm, S, mask = _qd_rows(tr, m.joint_names)
        base = np.array(m.home, float)
        dur = float(T[-1] - T[0])
        step = max(st['step_s'], dur / MAX_SAMPLES)
        n = max(1, int(math.ceil(dur / step)))
        times = T[0] + (np.arange(n) + 0.5) * (dur / n)
        segs = _segments(T, st['max_segments'])
        seg_of = np.clip(np.searchsorted([s[1] for s in segs], times, side='left'), 0, len(segs) - 1)
        moving = robot_points(project, ruid, m)
        L_mov = [lab for _l, X, lab in moving for _ in range(len(X))]
        labels = L_scene + L_mov
        seg_rec = [{'index': k, 't0': _r(a, 4), 't1': _r(b, 4), 'cameras': {}} for k, (a, b) in enumerate(segs)]
        tcp_path = []
        for i, t in enumerate(times):
            if ctx.cancelled():
                break
            k = int(np.clip(np.searchsorted(T, t, side='right') - 1, 0, len(T) - 2))
            a = (t - T[k]) / (T[k + 1] - T[k]) if T[k + 1] > T[k] else 0.0
            q = np.where(mask, Qm[k] * (1 - a) + Qm[k + 1] * a, base)
            qd = np.where(mask, S[k], 0.0)
            Pw, Vw = moving_points(m, q, qd, moving)
            if moving:
                tcp_path.append(Pw[0][0])
            P = np.concatenate([P_scene] + Pw) if Pw else P_scene
            V = np.concatenate([np.zeros_like(P_scene)] + Vw) if Vw else np.zeros_like(P_scene)
            for c in cams:
                R, o, v_c, w_c = camera_twist(project, m, ruid, q, qd, c['sensor'], c['imager'])
                uv, uvd, ok, _z = image_velocity(c['imager'].camera, R, o, v_c, w_c, P, V)
                if not ok.any():
                    continue
                sp = np.where(ok, np.linalg.norm(uvd, axis=1), -1.0)
                j = int(np.argmax(sp))
                peak = float(sp[j])
                sr = seg_rec[int(seg_of[i])]['cameras'].setdefault(c['key'], {'speed': -1.0})
                if peak > sr['speed']:
                    sr.update({'speed': peak, 't': float(t), 'point': labels[j]})
                A = acc[c['key']]
                if peak > A['peak']:
                    A['peak'] = peak
                    A['worst'] = {'trajectory_uid': tr['uid'], 'trajectory': tr.get('name', ''),
                                  'segment': int(seg_of[i]), 't': _r(t, 4), 'point': labels[j]}
                A['frames'].append((peak, ti, float(t), q, np.flatnonzero(ok), uv, uvd, ruid))
                if len(A['frames']) > 4 * st['frames_kept'] + 32:
                    A['frames'] = _thin(A['frames'], st['frames_kept'], step)
            ctx.report((ti + (i + 1) / len(times)) / max(1, len(trajs)), tr.get('name', ''))
        # per segment figures
        by_key = {c['key']: c for c in cams}
        for sr in seg_rec:
            out = []
            for key, v in sr['cameras'].items():
                c = by_key[key]
                e, ro = c['exposure']['value'], c['readout']['value']
                out.append({'key': key, 'speed_px_s': _r(v['speed'], 3), 't': _r(v['t'], 4), 'point': v['point'],
                            'smear_px': _r(v['speed'] * e, 4) if e is not None else None,
                            'skew_px': _r(v['speed'] * ro, 4) if ro is not None else None,
                            'status': _status(v['speed'] * e if e is not None else None, limit),
                            'bound': c['exposure']['kind'] == 'upper bound'})
                out[-1]['word'] = _word(out[-1]['status'], out[-1]['bound'])
            sr['cameras'] = sorted(out, key=lambda x: -(x['speed_px_s'] or 0))
            ws = [x['status'] for x in out]
            sr['status'] = next((w for w in ('serious', 'warning', 'unknown', 'good') if w in ws), 'good')
            # a segment is said for certain only when a camera with a stated exposure reaches its status
            sr['word'] = _word(sr['status'], bool(out) and all(x['bound'] for x in out if x['status'] == sr['status']))
        path = np.asarray(tcp_path, float)
        if len(path) > 200:
            path = path[np.linspace(0, len(path) - 1, 200).round().astype(int)]
        traj_out.append({'uid': tr['uid'], 'name': tr.get('name', ''), 'robot_uid': ruid,
                         'synthetic': bool(tr.get('synthetic')), 'duration_s': _r(dur, 4), 'samples': len(times),
                         'step_s': _r(dur / len(times), 6), 'segments': seg_rec,
                         'tcp_path': [[_r(x, 5) for x in p] for p in path],
                         'tcp_path_t': [_r(x, 4) for x in (times if len(times) <= 200 else
                                                           times[np.linspace(0, len(times) - 1, 200).round()
                                                                 .astype(int)])]})
    cam_out = [_camera_result(c, acc[c['key']], st, trajs) for c in cams]
    res = {
        'settings': {k: (copy.deepcopy(v)) for k, v in st.items()},
        'smear_limit_px': limit, 'smear_limit_default': 'smear_limit_px' in st['defaults'],
        'source': source, 'trajectories': traj_out, 'cameras': cam_out, 'notes': notes,
        'assumptions': list(ASSUMPTIONS), 'references': list(REFERENCES),
        'units': 'pixels of each imager\'s stream; seconds; px/s',
    }
    res['summary'] = _summary(res)
    res['suggestion'] = suggest(res)
    return res


def _thin(frames, k: int, step: float) -> list:
    """The k frames of largest peak speed, at least 5 steps apart in time (or in another trajectory)."""
    keep = []
    for f in sorted(frames, key=lambda f: -f[0]):
        if all(f[1] != g[1] or abs(f[2] - g[2]) >= 5 * step - 1e-12 for g in keep):
            keep.append(f)
        if len(keep) >= k:
            break
    return keep


def _camera_result(c, A, st, trajs) -> dict:
    limit = st['smear_limit_px']
    e, ro = c['exposure']['value'], c['readout']['value']
    peak = A['peak'] if A['worst'] is not None else None
    smear = peak * e if (peak is not None and e is not None) else None
    skew = peak * ro if (peak is not None and ro is not None) else None
    max_e = limit / peak if peak and peak > 0 else None
    frames = []
    for sp, ti, t, q, idx, uv, uvd, ruid in _thin(A['frames'], st['frames_kept'], st['step_s']):
        order = idx[np.argsort(-np.linalg.norm(uvd[idx], axis=1))][:st['vectors_kept']]
        vec = [[_r(uv[j, 0], 2), _r(uv[j, 1], 2), _r(uvd[j, 0] * (e or 0.0), 4), _r(uvd[j, 1] * (e or 0.0), 4),
                _r(uvd[j, 0], 3), _r(uvd[j, 1], 3)] for j in order]
        frames.append({'trajectory_uid': trajs[ti]['uid'], 'trajectory': trajs[ti].get('name', ''), 't': _r(t, 4),
                       'robot_uid': ruid, 'q': [_r(x, 7) for x in q], 'speed_px_s': _r(sp, 3),
                       'smear_px': _r(sp * e, 4) if e is not None else None, 'vectors': vec})
    status = _status(smear, limit) if peak is not None else 'good'
    s = c['sensor']
    return {'key': c['key'], 'sensor_uid': s.uid, 'sensor': s.name, 'imager': c['imager'].name,
            'role': c['imager'].role, 'name': c['name'], 'slot': s.slot, 'on_link': s.mount.on_link,
            'device': c['device'], 'stream': c['stream'], 'fps': c['fps'], 'shutter': c['shutter'],
            'exposure': {k: (_r(v, 9) if isinstance(v, float) else v) for k, v in c['exposure'].items()},
            'readout': {k: (_r(v, 9) if isinstance(v, float) else v) for k, v in c['readout'].items()},
            'seen': peak is not None, 'peak_speed_px_s': _r(peak, 3), 'smear_px': _r(smear, 4),
            'skew_px': _r(skew, 4), 'skew_bound': c['readout']['kind'] == 'upper bound',
            'smear_bound': c['exposure']['kind'] == 'upper bound', 'max_exposure_s': _r(max_e, 9),
            'status': status, 'word': _word(status, c['exposure']['kind'] == 'upper bound'), 'worst': A['worst'],
            'frames': frames, 'lens': c['label']}


def _summary(res) -> dict:
    limit = res['smear_limit_px']
    seen = [c for c in res['cameras'] if c['seen']]
    if not res['trajectories']:
        return {'status': 'warning', 'word': 'no motion', 'worst': None,
                'advice': 'The project has no trajectory and no grasp set to take approaches from: import a trajectory '
                          '(Robot ▸ Import trajectory…) or add a grasp set on the Tasks page.'}
    if not seen:
        return {'status': 'warning', 'word': 'nothing seen', 'worst': None,
                'advice': 'No camera sees the task volumes, the objects or the gripper along the motion.'}
    known = [c for c in seen if c['smear_px'] is not None]
    worst = max(known, key=lambda c: c['smear_px']) if known else max(seen, key=lambda c: c['peak_speed_px_s'])
    st = worst['status'] if known else 'unknown'
    skews = [c for c in seen if c['skew_px']]
    wskew = max(skews, key=lambda c: c['skew_px']) if skews else None
    return {'status': st, 'word': _word(st, bool(known) and worst['smear_bound']), 'worst': worst['key'],
            'smear_px': worst['smear_px'],
            'smear_bound': worst['smear_bound'], 'peak_speed_px_s': worst['peak_speed_px_s'],
            'max_exposure_s': worst['max_exposure_s'], 'worst_skew': wskew['key'] if wskew else None,
            'skew_px': wskew['skew_px'] if wskew else None, 'skew_bound': wskew['skew_bound'] if wskew else False,
            'over_limit': [c['key'] for c in known if c['smear_px'] > limit + 1e-12],
            'unknown_exposure': [c['key'] for c in seen if c['smear_px'] is None]}


def _floor2(x: float) -> float:
    """x rounded down to two significant figures (an exposure that keeps the smear within the limit)."""
    if x <= 0:
        return x
    e = math.floor(math.log10(x)) - 1
    return math.floor(x / 10 ** e) * 10 ** e


def _global_alternatives(kind: str, exclude: str) -> list:
    from .. import library as LB
    out = []
    for e in LB.entries(kind):
        ims = e.imagers
        if e.id != exclude and ims and all(im.get('shutter') == 'global' for im in ims):
            out.append(e.title)
    return out[:3]


def suggest(res: dict, project=None) -> dict | None:
    """The change that keeps every seen camera within the smear limit: {'kind': 'exposure', 'changes': [{'sensor_uid',
    'sensor', 'exposure_s', 'smear_before_px', 'smear_after_px'}], 'text', 'notes'} (Apply sets those exposures), or
    {'kind': 'advice', 'text'} when no exposure the maker allows is short enough or the skew needs a global shutter;
    None when everything is within the limit."""
    limit = res['smear_limit_px']
    by_sensor: dict = {}
    advice = []
    for c in res['cameras']:
        if not c['seen'] or not c['peak_speed_px_s']:
            continue
        over = c['smear_px'] is None or c['smear_px'] > limit + 1e-12
        if over:
            e = _floor2(limit / c['peak_speed_px_s'])
            smin = c['exposure'].get('stated_min')
            if smin is not None and e < smin:
                k = c['peak_speed_px_s'] * smin / limit
                advice.append(f'{c["name"]}: even the shortest exposure the maker states ({smin * 1000:g} ms) smears '
                              f'{c["peak_speed_px_s"] * smin:.2f} px; slow the motion to 1/{k:.2g} of its speed or '
                              'use a camera that allows a shorter exposure.')
                continue
            b = by_sensor.setdefault(c['sensor_uid'], {'sensor_uid': c['sensor_uid'], 'sensor': c['sensor'],
                                                       'exposure_s': e, 'imagers': []})
            b['exposure_s'] = min(b['exposure_s'], e)
            b['imagers'].append(c)
        if c['skew_px'] and c['skew_px'] > limit + 1e-12 and c['shutter'] == 'rolling':
            kind = 'depth' if c['role'] in ('ir_left', 'ir_right') or 'depth' in c['device'].lower() else 'camera'
            alts = _global_alternatives(kind, '')
            advice.append(f'{c["name"]}: its rolling shutter skews the image by {"up to " if c["skew_bound"] else ""}'
                          f'{c["skew_px"]:.2f} px; a global-shutter camera has none'
                          + (f' (in the library: {", ".join(alts)})' if alts else '') + '.')
    changes = []
    for b in by_sensor.values():
        e = b['exposure_s']
        worst_before = max((c['smear_px'] for c in b['imagers'] if c['smear_px'] is not None), default=None)
        after = max(c['peak_speed_px_s'] * e for c in b['imagers'])
        changes.append({'sensor_uid': b['sensor_uid'], 'sensor': b['sensor'], 'exposure_s': _r(e, 9),
                        'smear_before_px': _r(worst_before, 4), 'smear_after_px': _r(after, 4),
                        'before_bound': any(c['smear_bound'] for c in b['imagers'])})
    if changes:
        parts = []
        for ch in changes:
            before = ('not known' if ch['smear_before_px'] is None else
                      f'{"up to " if ch["before_bound"] else ""}{ch["smear_before_px"]:.2f} px')
            parts.append(f'{ch["sensor"]}: exposure at most {ch["exposure_s"] * 1000:.3g} ms (smear {before} → '
                         f'{ch["smear_after_px"]:.2f} px)')
        text = ('Set ' + '; '.join(parts) + f' to keep the smear within {limit:g} px'
                + (' (the default limit)' if res.get('smear_limit_default') else '') + '. Set the same exposure on '
                'the camera; a shorter exposure needs more light or gain.')
        if advice:
            text += ' ' + ' '.join(advice)
        return {'kind': 'exposure', 'changes': changes, 'text': text, 'advice': advice}
    if advice:
        return {'kind': 'advice', 'changes': [], 'text': ' '.join(advice), 'advice': advice}
    return None


def apply_suggestion(project, sug: dict) -> None:
    """Record the suggested exposures as the user's (settings 'exposure_s'); the caller makes it one undo step."""
    d = dict(project.analysis.studies.get(STUDY_ID) or {})
    ex = dict(d.get('exposure_s') or {})
    lim = settings_of(project)['smear_limit_px']
    for ch in sug.get('changes') or []:
        ex[ch['sensor_uid']] = {'value': float(ch['exposure_s']),
                                'source': f'the motion blur study\'s suggestion for a {lim:g} px smear limit'}
    d['exposure_s'] = ex
    project.analysis.studies[STUDY_ID] = d


@jobs.study('motion_blur', 'Motion blur and rolling shutter', quick=False, batch='motion-blur', needs_robot=True)
def run(project, ctx: jobs.Ctx) -> dict:
    """Pixel smear and rolling-shutter skew per camera along the project's trajectories (module docstring)."""
    return analyse(project, ctx)
