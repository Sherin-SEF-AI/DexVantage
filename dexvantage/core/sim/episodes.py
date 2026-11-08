"""Synthetic LeRobot episodes with ground truth (spec §9.1, last bullet): the project's robot moved along its
trajectories or along grasp approaches, rendered through every camera's own lens model (core/sim/render.py), written
as a LeRobot dataset of codebase version v2.1 or v3.0 (core/datasets/lerobot.py) with known, seeded faults injected,
and the ground truth written beside it as dexvantage_truth.json. It is how the dataset auditor (§8.2) and training
pipelines are tested against data whose faults are known exactly.

    settings_of(project, given=None) -> dict      the settings with the labelled defaults filled in
    plan(project, settings=None) -> Plan          what would be generated (motions, cameras, faults, timelines);
                                                  no rendering, no files
    generate(project, folder, settings=None, *, progress=None, cancel=None, keep_frames=False, overwrite=False)
                                                  -> dict (the summary; 'truth' the sidecar; with keep_frames the
                                                  generated arrays under 'arrays', which are not JSON)

Motion. settings motion 'auto' (the project's trajectories when it has any, else grasp approaches), 'trajectories' or
'grasps'. Episode e plays trajectory e mod n (core/robot/trajectory.py: linear in the joints between its samples; the
robot's joints it does not name at home), time-scaled to the episode when frames is set, at its own speed otherwise.
A grasp approach is home → pre-grasp (pre_grasp_m back along the approach, default 80 mm) → grasp → a hold → back to
the pre-grasp, each waypoint solved by IK (core/robot/ik.solve_batch, from home, self-collision off as in the motion
blur study's approaches, nearest to home), with minimum-jerk timing between waypoints (s = 10τ³ − 15τ⁴ + 6τ⁵; a
segment lasts its largest joint move over max_joint_speed, default 0.8 rad/s, at least 0.3 s, each scaled by a seeded
factor in [0.85, 1.15]); grasps are drawn without replacement from the first grasp set with the seed. Each episode adds
a seeded constant joint offset (N(0, q_offset_rad), default 0.02 rad, gripper joints left out), clipped to the limits,
so no two episodes are the same. Rows: t_i = i / fps; observation.state = q(t_i); action = q(t_i + action_lead_rows /
fps) (the commanded next state, default one row ahead), the last rows held.

Cameras. One camera per enabled camera or depth sensor: its primary imager (the RGB imager of a depth camera, as the
scene builds it), at its stream size, scaled down to max_width (default 640 px, a default, to keep datasets small)
with even sides (yuv420p), or to size [w, h] when given. Its frame rate is the stream's (the bandwidth study's stream
choice, motion_blur.imaging), else default_fps (30 fps, a default) when no device states one; the dataset's fps is the
largest, as a whole number. Camera key: 'observation.images.<sensor name>' (lower case, the parenthesised part left
out). RGB is MuJoCo's shading; every frame gets seeded Gaussian pixel noise (noise_dn, default 1 grey level, a default:
not a sensor noise model), so two real frames are never bit-identical while a duplicated frame is.

Timelines. Camera c has slots k = 0 … M−1 at t_k = k / fps_c over the rows' time span. A slot's stored timestamp is
s_k = t_k (+ jitter); its image shows the robot at the content time c_k = t_k (+ offset, + drift); the truth records
both per stored frame (episodes[e]['timing'][camera]: slot, stamp_s, content_time_s).

Faults (each switchable in settings faults {kind: {'enabled', 'episodes' (None: every episode), 'camera' (a camera key
or sensor uid; None: a seeded choice per episode among the cameras frame faults have used least), ...}}, each drawn
from its own generator seeded with seed, the episode and the fault, so switching one fault off does not move the
others; every figure below is a default chosen so the fault is visible in a short episode, not a measurement of real
hardware):
- drop (count 2): slots not stored (an isolated missing frame: a timestamp gap of two periods).
- duplicate (count 2): the stored frame repeats the previous stored frame's pixels exactly (a stale buffer), at the
  next slot's timestamp; placed where the robot moves fastest, so the scene did change.
- desync (offset_s 0.05 s): c_k = t_k + offset_s for one camera (its images show the robot offset_s later than their
  timestamps say).
- drift (drift_ppm 5000): c_k = t_k + drift_ppm · 10⁻⁶ · t_k (a clock running fast; exaggerated so that it shows within
  one episode).
- jitter (jitter_s 3 ms): s_k = t_k + n_k, n_k ~ N(0, jitter_s) clipped to ±0.2 of the period (so stamps stay in order
  and a gap stays distinguishable from jitter); the content stays at t_k.
- exposure_change (gain 1.4 over frames_frac 0.25 of the frames): an automatic-exposure step: the linear pixel values
  times gain, clipped, over a contiguous run of frames.
- blur (count 2, subframes 9, exposure_s = the frame period): temporal supersampling: the mean of `subframes` renders
  at content times spread evenly over [c_k − E/2, c_k + E/2]; placed where the robot moves fastest. Other frames are
  rendered at one instant.
- gripper_hidden (count 3): in the wrist camera (the first camera on a link of the moving robot), a contiguous run of
  frames where an occluder (a flat dark rectangle over the gripper's pixels, grown by 5 % of the width; painted in the
  image, not geometry) hides the gripper.
- mount_slip (from_episode = episodes // 2, position_m [3 mm, 0, 0], rotation_deg [0, 1.5°, 0], the mount-tolerance
  defaults of §8.5): from that episode on, one camera (default the first static one) renders through its mount
  perturbed by delta: T_body_sensor · ΔT, ΔT in the sensor's body frame, its rotation a rotation vector (axis × angle)
  in degrees.
- usb_reset (gap_s 0.25 s): a contiguous run of slots not stored (at most a fifth of the episode).
Faults that need frames are placed on interior slots that no other fault uses (a duplicate's previous frame
included); when an episode is too short for all of them, fewer are placed and the notes say so. The truth lists
exactly what was placed: 'frames' are indices into the camera's stored frames (for drop and usb_reset: the stored
frame right after each gap, and 'slots' the missing slots).

Ground truth (dexvantage_truth.json, schema dexvantage.truth/1): project_digest; generator (the settings, the seed,
the renderer, the codec, the motion of each episode); cameras {key: sensor_uid, imager, fps, intrinsics {model,
width, height, K, dist}, extrinsics {frame (the mount's link, or 'world'), pose (the optical frame in it, position
and quaternion x, y, z, w), the true mount}}; episodes {str(e): faults, extrinsics {key: the mount actually used},
visibility {key: gripper_visible (per stored frame: at least gripper_min_frac, default 0.2 %, of the pixels are the
gripper's links in the rendered segmentation, after any occluder), gripper_px}, timing, motion}; and a provenance
block. Dataset.truth() returns it.

Known limits: images are MuJoCo's OpenGL shading (no auto-exposure dynamics, rolling shutter, sensor noise model,
compression artefacts beyond the codec's); blur averages the 8-bit images (not linear radiance); the gripper occluder
is painted, not a body in the scene, so it casts no shadow and has no depth; the arm's motion is kinematic (no
dynamics, no tracking error besides the action lead), IK approaches are not checked against obstacles; one robot moves
and its joints are the state vector (other robots stay at home); the gripper is not closed in a grasp approach; one
imager per sensor is recorded; drift is a constant rate and jitter Gaussian; the faults are placed independently, and
several on one camera combine (desync, drift and jitter add).
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
import time
from dataclasses import dataclass, field

import numpy as np

from ..jobs import StudyError

SID = 'synth_episodes'
TRUTH_SCHEMA = 'dexvantage.truth/1'
FAULT_KINDS = ('drop', 'duplicate', 'desync', 'drift', 'jitter', 'exposure_change', 'blur', 'gripper_hidden',
               'mount_slip', 'usb_reset')
FAULT_TITLES = {'drop': 'Dropped frames', 'duplicate': 'Duplicated frames', 'desync': 'Camera-joint offset',
                'drift': 'Clock drift', 'jitter': 'Timestamp jitter', 'exposure_change': 'Exposure change',
                'blur': 'Motion-blurred frames', 'gripper_hidden': 'Gripper hidden in the wrist camera',
                'mount_slip': 'Mount slip', 'usb_reset': 'USB reset gap'}
FAULT_WORDS = {k: v[0].lower() + v[1:] for k, v in FAULT_TITLES.items()}
FAULT_WORDS['usb_reset'] = 'USB reset gaps'
FAULT_DEFAULTS = {                 # defaults of DexVantage, chosen so each fault shows in a short episode
    'drop': {'count': 2},
    'duplicate': {'count': 2},
    'desync': {'offset_s': 0.05},
    'drift': {'drift_ppm': 5000.0},
    'jitter': {'jitter_s': 0.003},
    'exposure_change': {'gain': 1.4, 'frames_frac': 0.25},
    'blur': {'count': 2, 'subframes': 9, 'exposure_s': None},
    'gripper_hidden': {'count': 3},
    'mount_slip': {'from_episode': None, 'position_m': [0.003, 0.0, 0.0], 'rotation_deg': [0.0, 1.5, 0.0]},
    'usb_reset': {'gap_s': 0.25},
}
DEFAULTS = {
    'episodes': 4,
    'frames': None,                # rows per episode; None: the motion's own duration
    'version': 'v3.0',
    'seed': 0,
    'motion': 'auto',              # 'auto' | 'trajectories' | 'grasps'
    'size': None,                  # [w, h] for every camera; None: the stream size scaled to max_width
    'max_width': 640,              # default: images at most this wide
    'default_fps': 30.0,           # default frame rate of a camera whose device states none
    'noise_dn': 1.0,               # default pixel noise (grey levels, 1 σ)
    'action_lead_rows': 1,         # default: the action is the state this many rows later
    'q_offset_rad': 0.02,          # default per-episode joint offset (1 σ)
    'pre_grasp_m': 0.08,           # default pre-grasp distance
    'max_joint_speed': 0.8,        # default joint speed of a grasp approach (rad/s)
    'gripper_min_frac': 0.002,     # default share of the image the gripper needs to count as visible
    'codec': 'auto',
    'device': 'auto',
    'chunk': 48,                   # renders per worker call (at most CHUNK_PIXELS pixels in all)
    'faults': None,                # None: every fault on with its defaults
}
CHOICES = {'version': ('v2.1', 'v3.0'), 'motion': ('auto', 'trajectories', 'grasps')}
OCCLUDER_RGB = (38, 36, 34)        # the painted occluder's colour (a drawing default)
OCCLUDER_GROW = 0.05               # the occluder covers the gripper's pixels grown by this share of the width
JITTER_CLIP = 0.2                  # jitter is clipped to this share of the frame period
APPROACH_MIN_S = 0.3               # shortest approach segment and the hold (s), defaults
EDGE = 2                           # frame faults stay this many slots from either end
CHUNK_PIXELS = 8_000_000           # pixels per render call at most (memory: RGB and segmentation of each)


class SynthError(StudyError):
    """Synthetic episodes cannot be made for this project or with these settings; the message says why."""


# ====================================================================== settings
def _fault_settings(given) -> dict:
    out = {k: dict(FAULT_DEFAULTS[k], enabled=True, episodes=None, camera=None) for k in FAULT_KINDS}
    if given is None:
        return out
    if isinstance(given, (list, tuple, set)):
        bad = [k for k in given if k not in FAULT_KINDS]
        if bad:
            raise SynthError(f'There is no fault called {bad[0]}; the faults are {", ".join(FAULT_KINDS)}.')
        for k in FAULT_KINDS:
            out[k]['enabled'] = k in given
        return out
    if not isinstance(given, dict):
        raise SynthError('The faults setting must be a list of fault names or a dictionary of fault settings.')
    for k, v in given.items():
        if k not in FAULT_KINDS:
            raise SynthError(f'There is no fault called {k}; the faults are {", ".join(FAULT_KINDS)}.')
        if isinstance(v, bool):
            out[k]['enabled'] = v
        elif isinstance(v, dict):
            unknown = [x for x in v if x not in out[k]]
            if unknown:
                raise SynthError(f'The {k} fault has no setting {unknown[0]}; it has {", ".join(sorted(out[k]))}.')
            out[k].update(v)
            out[k]['enabled'] = bool(v.get('enabled', True))
        else:
            raise SynthError(f'The {k} fault setting must be true, false or a dictionary.')
    return out


def _num(st, k, lo=None, integer=False, allow_none=False):
    v = st[k]
    if v is None and allow_none:
        return None
    try:
        v = int(v) if integer else float(v)
    except (TypeError, ValueError):
        raise SynthError(f'The setting {k} must be a {"whole " if integer else ""}number; it is {st[k]!r}.') from None
    if not math.isfinite(v) or (lo is not None and v < lo):
        raise SynthError(f'The setting {k} must be at least {lo:g}; it is {v:g}.')
    return v


def settings_of(project, given: dict | None = None) -> dict:
    """The settings (project.analysis.studies['synth_episodes'], then given over them) with the labelled defaults
    filled in; 'defaults' lists the keys left at their default."""
    stored = dict((getattr(getattr(project, 'analysis', None), 'studies', {}) or {}).get(SID) or {})
    stored.update(given or {})
    unknown = [k for k in stored if k not in DEFAULTS and k != 'defaults']
    if unknown:
        raise SynthError(f'There is no setting {unknown[0]} for synthetic episodes; the settings are '
                         f'{", ".join(DEFAULTS)}.')
    st = dict(DEFAULTS)
    used = [k for k in DEFAULTS if stored.get(k) is None]
    st.update({k: v for k, v in stored.items() if v is not None and k != 'defaults'})
    for k, ch in CHOICES.items():
        if st[k] not in ch:
            raise SynthError(f'The setting {k} must be one of {", ".join(ch)}; it is {st[k]!r}.')
    st['episodes'] = _num(st, 'episodes', 1, True)
    st['frames'] = _num(st, 'frames', 2, True, allow_none=True)
    st['seed'] = _num(st, 'seed', 0, True)
    st['max_width'] = _num(st, 'max_width', 16, True)
    st['chunk'] = max(1, _num(st, 'chunk', 1, True))
    st['action_lead_rows'] = _num(st, 'action_lead_rows', 0, True)
    for k in ('default_fps', 'noise_dn', 'q_offset_rad', 'pre_grasp_m', 'max_joint_speed', 'gripper_min_frac'):
        st[k] = _num(st, k, 0.0)
    if st['default_fps'] <= 0 or st['max_joint_speed'] <= 0:
        raise SynthError('The default frame rate and the joint speed must be above zero.')
    if st['size'] is not None:
        try:
            w, h = (int(x) for x in st['size'])
        except (TypeError, ValueError):
            raise SynthError('The setting size must be [width, height] in pixels.') from None
        if w < 2 or h < 2 or w % 2 or h % 2:
            raise SynthError(f'The image size must have an even width and height of 2 px or more (the videos are '
                             f'yuv420p); it is {w} × {h}.')
        st['size'] = [w, h]
    st['faults'] = _fault_settings(stored.get('faults'))
    f = st['faults']
    for k, key in (('drop', 'count'), ('duplicate', 'count'), ('blur', 'count'), ('gripper_hidden', 'count')):
        f[k][key] = max(0, int(f[k][key]))
    f['blur']['subframes'] = max(3, int(f['blur']['subframes']) | 1)          # odd: a centre subframe
    for k, key in (('desync', 'offset_s'), ('drift', 'drift_ppm'), ('jitter', 'jitter_s'), ('usb_reset', 'gap_s'),
                   ('exposure_change', 'gain'), ('exposure_change', 'frames_frac')):
        try:
            f[k][key] = float(f[k][key])
        except (TypeError, ValueError):
            raise SynthError(f'The {k} fault\'s {key} must be a number; it is {f[k][key]!r}.') from None
    if f['jitter']['jitter_s'] < 0 or f['usb_reset']['gap_s'] < 0 or f['exposure_change']['gain'] <= 0:
        raise SynthError('Jitter and the USB gap cannot be negative, and the exposure gain must be above zero.')
    st['defaults'] = used
    return st


# ====================================================================== motions
def _minjerk(tau):
    tau = np.clip(tau, 0.0, 1.0)
    return tau ** 3 * (10.0 - 15.0 * tau + 6.0 * tau ** 2)


@dataclass
class Motion:
    """q(t) of one robot over [0, duration]: knots (K,) s and Q (K, nq) on the model's joints, linear or minimum-jerk
    between knots, plus a constant offset; held outside the span."""
    robot_uid: str
    model: object
    name: str
    source: str                    # 'trajectory' | 'grasp approach'
    T: np.ndarray
    Q: np.ndarray
    smooth: bool = False
    offset: np.ndarray | None = None
    info: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return float(self.T[-1] - self.T[0])

    def scaled(self, duration: float) -> 'Motion':
        m = copy.copy(self)
        d = max(self.duration, 1e-9)
        m.T = (self.T - self.T[0]) * (float(duration) / d)
        return m

    def q(self, t) -> np.ndarray:
        """(nq,) for a scalar t, (n, nq) for an array of times."""
        ts = np.atleast_1d(np.asarray(t, float))
        T, Q = self.T, self.Q
        k = np.clip(np.searchsorted(T, ts, side='right') - 1, 0, max(len(T) - 2, 0))
        if len(T) < 2:
            out = np.repeat(Q[:1], len(ts), axis=0)
        else:
            span = np.where(T[k + 1] > T[k], T[k + 1] - T[k], 1.0)
            a = np.clip((ts - T[k]) / span, 0.0, 1.0)
            if self.smooth:
                a = _minjerk(a)
            out = Q[k] + (Q[k + 1] - Q[k]) * a[:, None]
        if self.offset is not None:
            out = out + self.offset
        out = np.clip(out, self.model.lower, self.model.upper)
        return out[0] if np.ndim(t) == 0 else out


def _gripper_joint_mask(robot, model) -> np.ndarray:
    tagged = {j['name'] for j in robot.joints if j.get('gripper')}
    return np.array([n in tagged for n in model.joint_names], bool)


def _trajectory_motions(project) -> list:
    from ..robot.model import RobotModelError, get_model
    out = []
    for tr in project.tasks.trajectories:
        if not isinstance(tr, dict) or len(tr.get('t') or []) < 2 or project.robot(tr.get('robot_uid', '')) is None:
            continue
        try:
            m = get_model(project, tr['robot_uid'])
        except RobotModelError:
            continue
        T = np.asarray(tr['t'], float)
        Q = np.asarray(tr['q'], float).reshape(len(T), -1)
        idx = {n: i for i, n in enumerate(m.joint_names)}
        Qm = np.tile(np.asarray(m.home, float), (len(T), 1))
        for c, j in enumerate(tr.get('joints') or []):
            if j in idx and c < Q.shape[1]:
                Qm[:, idx[j]] = Q[:, c]
        out.append(Motion(tr['robot_uid'], m, str(tr.get('name') or 'Trajectory'), 'trajectory', T - T[0], Qm,
                          info={'trajectory_uid': tr.get('uid'), 'trajectory': tr.get('name')}))
    return out


def _approach_motions(project, st, n: int, notes: list) -> list:
    from .. import grasps as GR
    from ..robot import ik as IK
    from ..robot.model import RobotModelError, get_model
    sets = GR.sets(project)
    if not sets:
        return []
    gs = sets[0]
    poses = list(gs.get('poses') or [])
    ruid = gs.get('robot_uid') or ''
    robot = project.robot(ruid)
    if not poses or robot is None:
        return []
    try:
        m = get_model(project, ruid)
    except RobotModelError as e:
        notes.append(f'The grasp approaches could not be used: {e}')
        return []
    rng = np.random.default_rng([st['seed'], 31])
    order = rng.permutation(len(poses))[:min(len(poses), max(4 * n, 16))]
    sub = dict(gs, poses=[poses[i] for i in order])
    try:
        link, tcp, T = GR.ik_targets(project, sub)
    except GR.GraspError as e:
        raise SynthError(str(e)) from None
    A = []
    for g, Tk in zip(sub['poses'], T):
        a = np.asarray(g.get('approach') or Tk[:3, 2], float)
        A.append(a / max(float(np.linalg.norm(a)), 1e-300))
    Tpre = T.copy()
    Tpre[:, :3, 3] -= st['pre_grasp_m'] * np.asarray(A)
    home = np.asarray(m.home, float)
    rc = IK.solve_batch(m, link, T, q0=home, seeds=4, rng=int(st['seed']), tcp=tcp, collision=False)
    rp = IK.solve_batch(m, link, Tpre, q0=rc.q, seeds=4, rng=int(st['seed']), tcp=tcp, collision=False,
                        rule='nearest_q0')
    ok = [k for k in range(len(order)) if rc.ok[k] and rp.ok[k]]
    if not ok:
        return []
    out = []
    for e in range(n):
        k = ok[e % len(ok)]
        g = sub['poses'][k]
        knots = [home, np.asarray(rp.q[k], float), np.asarray(rc.q[k], float), np.asarray(rc.q[k], float),
                 np.asarray(rp.q[k], float)]
        er = np.random.default_rng([st['seed'], 37, e])
        T = [0.0]
        for a, b in zip(knots[:-1], knots[1:]):
            d = float(np.max(np.abs(b - a))) / st['max_joint_speed'] if len(a) else 0.0
            T.append(T[-1] + max(APPROACH_MIN_S, d) * float(er.uniform(0.85, 1.15)))
        out.append(Motion(ruid, m, f'Approach to grasp {g.get("id", int(order[k]))}', 'grasp approach',
                          np.asarray(T), np.asarray(knots), smooth=True,
                          info={'grasp_set': gs.get('uid'), 'grasp_id': g.get('id', int(order[k])),
                                'pre_grasp_m': st['pre_grasp_m']}))
    if len(ok) < n:
        notes.append(f'Only {len(ok)} of the grasps tried have an IK solution for the grasp and the pre-grasp, so '
                     'episodes repeat their approaches.')
    return out


def motions(project, st: dict, notes: list) -> list:
    """One Motion per episode (module docstring)."""
    n = st['episodes']
    trs = _trajectory_motions(project) if st['motion'] in ('auto', 'trajectories') else []
    if st['motion'] == 'trajectories' and not trs:
        raise SynthError('The project has no trajectory to play: import one on the Tasks page, or choose grasp '
                         'approaches.')
    if trs:
        trs = [m for m in trs if m.robot_uid == trs[0].robot_uid]     # one robot moves (Known limits)
        return [copy.copy(trs[e % len(trs)]) for e in range(n)]
    ap = _approach_motions(project, st, n, notes)
    if not ap:
        raise SynthError('There is no motion to record: the project has no trajectory, and no grasp of its first '
                         'grasp set has an IK solution (make a grasp set on the Tasks page or import a trajectory).')
    if st['motion'] == 'auto':
        notes.append('The project has no trajectory: the episodes are grasp approaches of its first grasp set.')
    return ap


# ====================================================================== cameras
def _key_of(name: str, used: set) -> str:
    base = re.sub(r'\([^)]*\)', '', str(name or '')).strip().lower()
    base = re.sub(r'[^a-z0-9]+', '_', base).strip('_') or 'camera'
    k, i = f'observation.images.{base}', 2
    while k in used:
        k, i = f'observation.images.{base}_{i}', i + 1
    used.add(k)
    return k


@dataclass
class CamPlan:
    key: str
    sensor_uid: str
    sensor: str
    imager: str
    scene_camera: object
    model: object                  # CameraModel at the stored size
    fps: float
    fps_source: str
    on_link: bool
    robot_uid: str
    parent: str                    # the mount's link, or 'world'
    T_parent_sensor: np.ndarray    # mount pose (in the link or the world)


def _even(x: float) -> int:
    return max(2, int(round(x / 2.0)) * 2)


def cameras(project, R, st: dict, notes: list) -> list:
    """[CamPlan] for every enabled camera or depth sensor that has a camera in the scene."""
    from ..studies import motion_blur as MB
    try:
        imaging = MB.imaging(project, MB.settings_of(project))
    except Exception:                                  # noqa: BLE001 — the frame rate then falls back to the default
        imaging = []
    out, used = [], set()
    for s in project.variant.sensors:
        if not s.enabled or s.kind not in ('camera', 'depth'):
            continue
        c = R.build.camera(s.uid)
        if c is None:
            notes.append(f'{s.name} has no camera in the MuJoCo scene, so it is not recorded.')
            continue
        fps, src = None, ''
        mine = [x for x in imaging if x['sensor'].uid == s.uid and x.get('fps')]
        hit = next((x for x in mine if x['imager'].name == c.imager), mine[0] if mine else None)
        if hit is not None:
            fps, src = float(hit['fps']), f'the {hit["imager"].name} stream ({hit["stream"][0]} × ' \
                                          f'{hit["stream"][1]} at {float(hit["fps"]):g} fps)'
        else:
            fps, src = float(st['default_fps']), f'the default {st["default_fps"]:g} fps (no device states one)'
        cam = c.model
        if st['size'] is not None:
            w, h = st['size']
        elif cam.width > st['max_width']:
            s_ = st['max_width'] / cam.width
            w, h = _even(cam.width * s_), _even(cam.height * s_)
        else:
            w, h = _even(cam.width - 1 if cam.width % 2 else cam.width), \
                _even(cam.height - 1 if cam.height % 2 else cam.height)
        if (w, h) != (cam.width, cam.height):
            cam = cam.scaled(w, h)
        out.append(CamPlan(_key_of(s.name, used), s.uid, s.name, c.imager, c, cam, fps, src, bool(s.mount.on_link),
                           s.mount.robot_uid if s.mount.on_link else '',
                           s.mount.parent_frame if s.mount.on_link else 'world', s.mount.pose.matrix()))
    if not out:
        raise SynthError('There is no camera to record: add a camera or depth sensor and switch it on.')
    return out


# ====================================================================== faults and timelines
def _delta(pos, rot_deg) -> np.ndarray:
    from ..frames import quat_from_axis_angle, quat_to_matrix
    r = np.radians(np.asarray(rot_deg, float).reshape(3))
    ang = float(np.linalg.norm(r))
    D = np.eye(4)
    if ang > 0:
        D[:3, :3] = quat_to_matrix(quat_from_axis_angle(r / ang, ang))
    D[:3, 3] = np.asarray(pos, float).reshape(3)
    return D


def _pose_dict(T) -> dict:
    from ..frames import Pose
    return Pose.from_matrix(T).to_dict()


def extrinsics(cp: CamPlan, delta=None) -> dict:
    """{'frame', 'pose'}: the camera's optical frame in its mount's link (or the world), the mount perturbed by
    delta (4 × 4 in the sensor body) when given."""
    T = cp.T_parent_sensor @ (np.eye(4) if delta is None else delta) @ cp.scene_camera.T_sensor_optical
    return {'frame': cp.parent, 'pose': _pose_dict(T)}


def intrinsics(cam) -> dict:
    """{'model', 'width', 'height', 'K', 'dist'} of a CameraModel (its own model name and terms)."""
    K = [[cam.fx, 0.0, cam.cx], [0.0, cam.fy, cam.cy], [0.0, 0.0, 1.0]]
    return {'model': cam.model, 'width': int(cam.width), 'height': int(cam.height), 'K': K,
            'dist': [float(x) for x in cam.dist]}


@dataclass
class Timeline:
    """One camera in one episode: per slot its nominal time; the stored slots with their stamps, content times (one
    per subframe), gains and flags."""
    fps: float
    t_slot: np.ndarray
    removed: set = field(default_factory=set)
    stamp: np.ndarray | None = None
    content: np.ndarray | None = None
    gain: np.ndarray | None = None
    dup: np.ndarray | None = None
    blur: np.ndarray | None = None
    paint: np.ndarray | None = None
    exposure_s: float = 0.0
    subframes: int = 1
    delta: np.ndarray | None = None

    @property
    def stored(self) -> np.ndarray:
        return np.array([k for k in range(len(self.t_slot)) if k not in self.removed], int)


@dataclass
class EpisodePlan:
    index: int
    motion: Motion
    t: np.ndarray
    q: np.ndarray
    action: np.ndarray
    timelines: dict                # camera key -> Timeline
    faults: list                   # truth entries
    notes: list = field(default_factory=list)


@dataclass
class Plan:
    settings: dict
    fps: int
    cameras: list                  # [CamPlan]
    episodes: list                 # [EpisodePlan]
    robot: object
    gripper_links: list
    wrist: str | None              # camera key of the wrist camera
    notes: list = field(default_factory=list)


def _pick_camera(cfg, cams, rng, load=None):
    """The camera a fault goes to: the one its settings name, else a seeded choice among the cameras with the
    fewest slots taken by frame faults so far (load: camera key -> slots used; None: any camera)."""
    keys = [c.key for c in cams]
    if load is not None:
        low = min(len(load[k]) for k in keys)
        keys = [k for k in keys if len(load[k]) == low]
    want = cfg.get('camera')
    if want:
        hit = next((c.key for c in cams if want in (c.key, c.sensor_uid)), None)
        if hit is None:
            raise SynthError(f'A fault names camera {want}, which is not one of the recorded cameras '
                             f'({", ".join(c.key for c in cams)}).')
        return hit
    return keys[int(rng.integers(len(keys)))] if keys else None


def _applies(cfg, e: int) -> bool:
    eps = cfg.get('episodes')
    return bool(cfg.get('enabled')) and (eps is None or e in [int(x) for x in eps])


def _speed(motion: Motion, t: np.ndarray) -> np.ndarray:
    h = 1e-3
    return np.linalg.norm(motion.q(t + h) - motion.q(t - h), axis=1) / (2 * h)


def _episode_plan(e: int, st: dict, plan: Plan, motion: Motion, n_rows: int) -> EpisodePlan:
    fps = plan.fps
    rng = np.random.default_rng([st['seed'], 101, e])
    mask = _gripper_joint_mask(plan.robot, motion.model)
    off = rng.normal(0.0, st['q_offset_rad'], motion.model.nq) * ~mask if st['q_offset_rad'] > 0 else None

    def frng(kind):
        """Each fault draws from its own seeded generator: switching one off does not move the others."""
        return np.random.default_rng([st['seed'], 103, e, FAULT_KINDS.index(kind)])
    motion = copy.copy(motion)
    motion.offset = off
    t = np.arange(n_rows) / fps
    q = motion.q(t)
    lead = st['action_lead_rows'] / fps
    action = motion.q(np.minimum(t + lead, t[-1]))
    D = float(t[-1])
    tls = {}
    for c in plan.cameras:
        M = int(math.floor(D * c.fps + 1e-9)) + 1
        ts = np.arange(M) / c.fps
        tls[c.key] = Timeline(c.fps, ts, gain=np.ones(M), dup=np.zeros(M, bool), blur=np.zeros(M, bool),
                              paint=np.zeros(M, bool), content=ts.copy(), stamp=ts.copy())
    faults, notes = [], []
    F = st['faults']
    used = {c.key: set() for c in plan.cameras}

    def free(key, lo=EDGE, hi=None):
        M = len(tls[key].t_slot)
        hi = M - EDGE if hi is None else hi
        return [k for k in range(lo, hi) if k not in used[key]]

    def short(kind, key, got, want):
        if got < want:
            notes.append(f'Episode {e}: {got} of {want} {FAULT_WORDS[kind]} fit in {key}.')

    # the gripper occluder first (it can only go to the wrist camera), then the gap (it needs the most room)
    cfg = F['gripper_hidden']
    rng = frng('gripper_hidden')
    if _applies(cfg, e) and cfg['count']:
        if plan.wrist is None:
            if e == 0:
                plan.notes.append('No camera rides on the moving robot, so no gripper-hidden frames are injected.')
        else:
            key = plan.wrist if not cfg.get('camera') else _pick_camera(cfg, plan.cameras, rng)
            n = int(cfg['count'])
            cand = [k for k in free(key, EDGE, len(tls[key].t_slot) - EDGE - n + 1)
                    if all(x not in used[key] for x in range(k, k + n))]
            if cand:
                k0 = int(cand[int(rng.integers(len(cand)))])
                slots = list(range(k0, k0 + n))
                tls[key].paint[slots] = True
                used[key].update(slots)
                faults.append({'kind': 'gripper_hidden', 'camera': key, 'slots': slots, 'frames': None,
                               'occluder': 'a flat dark rectangle painted over the gripper\'s pixels'})
            else:
                short('gripper_hidden', key, 0, n)
    cfg = F['usb_reset']
    rng = frng('usb_reset')
    if _applies(cfg, e):
        key = _pick_camera(cfg, plan.cameras, rng, used)
        tl = tls[key]
        M = len(tl.t_slot)
        n_gap = int(min(max(2, round(cfg['gap_s'] * tl.fps)), max(0, M // 5)))
        starts = [k for k in free(key, EDGE + 1, M - EDGE - n_gap) if all(x not in used[key]
                                                                           for x in range(k - 1, k + n_gap + 1))]
        if n_gap >= 2 and starts:
            k0 = int(starts[int(rng.integers(len(starts)))])
            slots = list(range(k0, k0 + n_gap))
            tl.removed.update(slots)
            used[key].update(range(k0 - 1, k0 + n_gap + 1))
            faults.append({'kind': 'usb_reset', 'camera': key, 'slots': slots, 'frames': None,
                           'gap_s': float((n_gap + 1) / tl.fps)})
        else:
            short('usb_reset', key, 0, 1)
    cfg = F['drop']
    rng = frng('drop')
    if _applies(cfg, e) and cfg['count']:
        key = _pick_camera(cfg, plan.cameras, rng, used)
        tl = tls[key]
        slots = []
        for _ in range(cfg['count']):
            cand = [k for k in free(key) if k - 1 not in used[key] and k + 1 not in used[key]]
            if not cand:
                break
            k = int(cand[int(rng.integers(len(cand)))])
            slots.append(k)
            used[key].update((k - 1, k, k + 1))
        tl.removed.update(slots)
        short('drop', key, len(slots), cfg['count'])
        if slots:
            faults.append({'kind': 'drop', 'camera': key, 'slots': sorted(slots), 'frames': None})

    def fast_slots(key, n, pairs=False):
        """n slots where the robot moves fastest (from the faster half, seeded)."""
        tl = tls[key]
        cand = [k for k in free(key) if not pairs or (k - 1 not in used[key] and k - 1 not in tl.removed)]
        if not cand:
            return []
        sp = _speed(motion, tl.t_slot[cand])
        order = [cand[i] for i in np.argsort(-sp, kind='stable')]
        pool = order[:max(n, (len(order) + 1) // 2)]
        out = []
        for k in rng.permutation(pool):
            k = int(k)
            if k in used[key] or (pairs and (k - 1 in used[key] or k - 1 in tl.removed)):
                continue
            out.append(k)
            used[key].update((k - 1, k) if pairs else (k,))
            if len(out) == n:
                break
        return sorted(out)

    cfg = F['duplicate']
    rng = frng('duplicate')
    if _applies(cfg, e) and cfg['count']:
        key = _pick_camera(cfg, plan.cameras, rng, used)
        slots = fast_slots(key, cfg['count'], pairs=True)
        tls[key].dup[slots] = True
        short('duplicate', key, len(slots), cfg['count'])
        if slots:
            faults.append({'kind': 'duplicate', 'camera': key, 'slots': slots, 'frames': None})
    cfg = F['blur']
    rng = frng('blur')
    if _applies(cfg, e) and cfg['count']:
        key = _pick_camera(cfg, plan.cameras, rng, used)
        tl = tls[key]
        slots = fast_slots(key, cfg['count'])
        tl.blur[slots] = True
        tl.exposure_s = float(cfg['exposure_s']) if cfg['exposure_s'] is not None else 1.0 / tl.fps
        tl.subframes = int(cfg['subframes'])
        short('blur', key, len(slots), cfg['count'])
        if slots:
            faults.append({'kind': 'blur', 'camera': key, 'slots': slots, 'frames': None,
                           'exposure_s': tl.exposure_s, 'subframes': tl.subframes})
    cfg = F['exposure_change']
    rng = frng('exposure_change')
    if _applies(cfg, e):
        key = _pick_camera(cfg, plan.cameras, rng, used)
        tl = tls[key]
        M = len(tl.t_slot)
        n = max(1, int(round(cfg['frames_frac'] * M)))
        if M - 2 * EDGE - n > 0:
            k0 = int(rng.integers(EDGE, M - EDGE - n + 1))
            slots = [k for k in range(k0, k0 + n) if k not in tl.removed]
            tl.gain[slots] = cfg['gain']
            faults.append({'kind': 'exposure_change', 'camera': key, 'slots': slots, 'frames': None,
                           'gain': float(cfg['gain'])})
        else:
            short('exposure_change', key, 0, 1)
    for kind, par in (('desync', 'offset_s'), ('drift', 'drift_ppm')):
        cfg = F[kind]
        rng = frng(kind)
        if _applies(cfg, e):
            key = _pick_camera(cfg, plan.cameras, rng)
            tl = tls[key]
            tl.content = tl.content + (cfg['offset_s'] if kind == 'desync' else
                                       cfg['drift_ppm'] * 1e-6 * tl.t_slot)
            faults.append({'kind': kind, 'camera': key, 'frames': None, par: float(cfg[par])})
    cfg = F['jitter']
    rng = frng('jitter')
    if _applies(cfg, e) and cfg['jitter_s'] > 0:
        key = _pick_camera(cfg, plan.cameras, rng)
        tl = tls[key]
        lim = JITTER_CLIP / tl.fps
        j = np.clip(rng.normal(0.0, cfg['jitter_s'], len(tl.t_slot)), -lim, lim)
        j[0] = abs(j[0])
        tl.stamp = tl.t_slot + j
        faults.append({'kind': 'jitter', 'camera': key, 'frames': None, 'jitter_s': float(cfg['jitter_s']),
                       'clip_s': float(lim)})
    cfg = F['mount_slip']
    rng = frng('mount_slip')
    k_from = cfg['from_episode'] if cfg['from_episode'] is not None else st['episodes'] // 2
    if cfg.get('enabled') and e >= int(k_from) and (cfg.get('episodes') is None or e in cfg['episodes']):
        if cfg.get('camera'):
            key = _pick_camera(cfg, plan.cameras, rng)
        else:
            key = next((c.key for c in plan.cameras if not c.on_link), plan.cameras[0].key)
        tls[key].delta = _delta(cfg['position_m'], cfg['rotation_deg'])
        faults.append({'kind': 'mount_slip', 'camera': key, 'frames': None, 'from_episode': int(k_from),
                       'delta': {'position': [float(x) for x in cfg['position_m']],
                                 'rotation_deg': [float(x) for x in cfg['rotation_deg']]},
                       'delta_frame': 'the sensor body (T_body_sensor · ΔT); rotation_deg is a rotation vector '
                                      '(axis × angle) in degrees'})
    # stored indices
    for f in faults:
        tl = tls.get(f.get('camera'))
        if tl is None or 'slots' not in f:
            continue
        pos = {int(k): i for i, k in enumerate(tl.stored)}
        if f['kind'] == 'exposure_change':
            # the frames whose pixels carry the gain: a duplicate repeats its previous frame's pixels
            eff, last = [], 1.0
            for k in tl.stored:
                last = last if tl.dup[k] else float(tl.gain[k])
                eff.append(last)
            f['frames'] = [i for i, g_ in enumerate(eff) if g_ != 1.0]
            f['slots'] = [int(tl.stored[i]) for i in f['frames']]
            continue
        if f['kind'] in ('drop', 'usb_reset'):
            after = sorted({min(k for k in pos if k > s) for s in f['slots'] if any(k > s for k in pos)})
            if f['kind'] == 'drop':
                f['frames'] = [pos[k] for k in after]
            else:
                f['frames'] = [pos[after[0]]] if after else []
            f['times_s'] = [float(tl.t_slot[s]) for s in f['slots']]
        else:
            f['frames'] = [pos[k] for k in f['slots'] if k in pos]
    return EpisodePlan(e, motion, t, q, action, tls, faults, notes)


def gripper_links(robot, model) -> list:
    """The gripper's links: the gripper frame's body and every body below it in the model tree."""
    from .. import grasps as GR
    try:
        frame = GR.gripper_frame(robot, model)
    except GR.GraspError:
        frame = model.link_names[-1]
    m = model.m
    b0 = int(model._lb[model.link_index(frame)])

    def under(b):
        while b > 0:
            if b == b0:
                return True
            b = int(m.body_parentid[b])
        return False
    return [n for i, n in enumerate(model.link_names) if under(int(model._lb[i]))]


def plan(project, settings: dict | None = None, *, renderer=None) -> Plan:
    """What generate() would make (module docstring), without rendering or writing: motions, cameras, timelines and
    faults. renderer: an open Renderer to take the scene's cameras from (one is built from the scene otherwise)."""
    from .scene import build_scene
    st = settings if (settings is not None and 'defaults' in settings) else settings_of(project, settings)
    if not project.robots:
        raise SynthError('Synthetic episodes need a robot: import one first (Robot ▸ Import robot…) or open a sample.')
    notes = []
    mots = motions(project, st, notes)

    class _B:                                          # the scene's cameras without a render worker
        build = build_scene(project, objects=True, lights=True) if renderer is None else renderer.build
    cams = cameras(project, renderer or _B, st, notes)
    fps_all = [c.fps for c in cams]
    fps = int(round(max(fps_all)))
    if abs(max(fps_all) - fps) > 1e-9:
        notes.append(f'LeRobot stores a whole-number frame rate: the joint states are at {fps} fps, the cameras at '
                     'their own rates.')
    robot = project.robot(mots[0].robot_uid)
    glinks = gripper_links(robot, mots[0].model)
    wrist = next((c.key for c in cams if c.on_link and c.robot_uid == robot.uid), None)
    p = Plan(st, fps, cams, [], robot, glinks, wrist, notes)
    for e, m in enumerate(mots):
        if st['frames'] is not None:
            n_rows = int(st['frames'])
            m = m.scaled((n_rows - 1) / fps)
        else:
            n_rows = max(2, int(round(m.duration * fps)) + 1)
        p.episodes.append(_episode_plan(e, st, p, m, n_rows))
    for ep in p.episodes:
        p.notes += ep.notes
    return p


# ====================================================================== rendering and writing
class _Frames:
    """One camera of one episode: renders its stored frames in order, as the LeRobot writer encodes them."""

    def __init__(self, gen, ep: EpisodePlan, cp: CamPlan, ci: int):
        self.gen, self.ep, self.cp, self.ci = gen, ep, cp, ci
        self.tl = ep.timelines[cp.key]
        self.kept = None
        self.visible = None
        self.px = None

    def _views(self, k: int) -> list:
        tl, cp = self.tl, self.cp
        c = float(tl.content[k])
        times = [c] if not tl.blur[k] else list(c + tl.exposure_s * np.linspace(-0.5, 0.5, tl.subframes))
        ruid = self.ep.motion.robot_uid
        return [{'sensor_uid': cp.sensor_uid, 'imager': cp.imager, 'q': {ruid: self.ep.motion.q(x)},
                 'outputs': ('rgb', 'seg'), 'camera_model': cp.model, 'extrinsic_delta': tl.delta} for x in times]

    def __call__(self, indices=None):
        if self.kept is not None:
            for i in (range(len(self.kept)) if indices is None else indices):
                yield i, self.kept[i]
            return
        want = None if indices is None else set(int(i) for i in indices)
        for i, img in self._generate():
            if want is None or i in want:
                yield i, img

    def _generate(self):
        g, tl, cp = self.gen, self.tl, self.cp
        stored = tl.stored
        W, H = cp.model.width, cp.model.height
        min_px = max(1, int(math.ceil(g.st['gripper_min_frac'] * W * H)))
        own = g.gripper_owners
        chunk = max(1, min(int(g.st['chunk']), CHUNK_PIXELS // max(1, W * H)))
        keep = [] if g.keep else None
        vis, pxs = [], []
        count = self.visible is None                     # a second pass (the writer encoding again) is not counted
        prev = prev_px = None
        j = 0
        while j < len(stored):
            if g.cancel is not None and g.cancel():
                raise SynthError('The generation was cancelled.')
            # the next frames up to `chunk` renders
            batch, views = [], []
            while j < len(stored) and (not views or len(views) < chunk):
                k = int(stored[j])
                v = [] if tl.dup[k] else self._views(k)
                batch.append((j, k, len(views), len(v)))
                views += v
                j += 1
            imgs = g.render(views) if views else []
            for i, k, a, n in batch:
                if n == 0:                               # a duplicate: the previous stored frame's pixels
                    img, px = prev.copy(), prev_px
                else:
                    sub = imgs[a:a + n]
                    rgb = sub[0]['rgb'].astype(np.float32)
                    for s_ in sub[1:]:                   # temporal supersampling: the mean over the exposure
                        rgb += s_['rgb']
                    if n > 1:
                        rgb /= n
                    seg = sub[n // 2]['seg'].copy()
                    gm = np.isin(seg, own)
                    if tl.paint[k] and gm.any():
                        vv, uu = np.nonzero(gm)
                        grow = int(math.ceil(OCCLUDER_GROW * W))
                        u0, u1 = max(0, uu.min() - grow), min(W, uu.max() + grow + 1)
                        v0, v1 = max(0, vv.min() - grow), min(H, vv.max() + grow + 1)
                        rgb[v0:v1, u0:u1] = OCCLUDER_RGB
                        seg[v0:v1, u0:u1] = -2
                        gm = np.isin(seg, own)
                    px = int(gm.sum())
                    if tl.gain[k] != 1.0:
                        rgb *= np.float32(tl.gain[k])
                    if g.st['noise_dn'] > 0:
                        nr = np.random.default_rng([g.st['seed'], 211, self.ep.index, self.ci, k])
                        noise = nr.standard_normal(rgb.shape, dtype=np.float32)
                        noise *= np.float32(g.st['noise_dn'])
                        rgb += noise
                    np.rint(rgb, out=rgb)
                    np.clip(rgb, 0, 255, out=rgb)
                    img = rgb.astype(np.uint8)
                prev, prev_px = img, px
                vis.append(bool(px >= min_px))
                pxs.append(int(px))
                if keep is not None:
                    keep.append(img)
                if count:
                    g.frames_done += 1
                    g.tick()
                yield i, img
        self.visible, self.px = vis, pxs
        if keep is not None:
            self.kept = keep


class _Generator:
    def __init__(self, st, R, plan_, progress, cancel, keep):
        self.st, self.R, self.plan = st, R, plan_
        self.progress, self.cancel, self.keep = progress, cancel, keep
        self.render_s = 0.0
        self.renders = 0
        self.frames_done = 0
        self.total = sum(len(tl.stored) for ep in plan_.episodes for tl in ep.timelines.values())
        own = [R.owner_index(('robot', plan_.robot.uid, ln)) for ln in plan_.gripper_links]
        self.gripper_owners = np.array([o for o in own if o >= 0], np.int32)
        self._last = 0.0

    def render(self, views):
        t0 = time.perf_counter()
        out = self.R.render(views)
        self.render_s += time.perf_counter() - t0
        self.renders += len(views)
        return out

    def tick(self):
        now = time.perf_counter()
        if self.progress is not None and (now - self._last > 0.2 or self.frames_done == self.total):
            self._last = now
            self.progress(0.05 + 0.9 * self.frames_done / max(1, self.total),
                          f'{self.frames_done:,} of {self.total:,} camera frames')


def generate(project, folder: str, settings: dict | None = None, *, progress=None, cancel=None,
             keep_frames: bool = False, overwrite: bool = False) -> dict:
    """Render and write the synthetic dataset to folder (a new or empty folder; an existing LeRobot dataset only with
    overwrite) with its ground truth (module docstring). Returns the summary: root, version, codec, episodes, rows,
    camera_frames, renders, seconds, render_seconds, throughput figures, the renderer, the faults placed per kind,
    notes, truth (the sidecar document) and, with keep_frames, arrays: [{'t', 'q', 'action', 'cameras': {key:
    {'timestamps', 'frames' [HxWx3 uint8]}}}] per episode."""
    from ... import __version__
    from ...validation.mjrender import RenderUnavailable
    from .. import provenance as PV
    from ..datasets import lerobot as L
    from ..datasets.episode import CameraStream, DatasetError, Episode
    from .render import Renderer
    st = settings_of(project, settings)
    t_all = time.perf_counter()
    report = (lambda f, text='': progress(f, text)) if progress is not None else (lambda f, text='': None)
    report(0.0, 'Building the MuJoCo scene')
    try:
        R = Renderer(project, device=st['device'])
    except RuntimeError as e:
        raise SynthError(f'The MuJoCo scene could not be built: {e}') from None
    try:
        report(0.02, 'Planning the motions and the faults')
        p = plan(project, st, renderer=R)
        if not len(R.build.owners) or not p.gripper_links:
            raise SynthError('The robot has no visual geometry in the MuJoCo scene, so it cannot be rendered.')
        gen = _Generator(st, R, p, progress, cancel, keep_frames)
        if not len(gen.gripper_owners):
            p.notes.append('The gripper has no visual geometry in the scene: every frame counts it as hidden.')
        eps, streams = [], {}
        for ep in p.episodes:
            cams = {}
            for ci, cp in enumerate(p.cameras):
                tl = ep.timelines[cp.key]
                fr = _Frames(gen, ep, cp, ci)
                streams[(ep.index, cp.key)] = fr
                ex = extrinsics(cp, tl.delta)
                cams[cp.key] = CameraStream(cp.key, cp.model.width, cp.model.height, cp.fps,
                                            np.asarray(tl.stamp[tl.stored], float), None, intrinsics(cp.model), ex,
                                            {'synthetic': True, 'sensor_uid': cp.sensor_uid}, fr)
            m = ep.motion
            eps.append(Episode(f'episode_{ep.index:06d}', ep.index, ep.t, ep.q, list(m.model.joint_names),
                               ep.action, list(m.model.joint_names), float(p.fps), p.robot.name,
                               f'{m.name} (synthetic)', cams, {'synthetic': True}))
        t_write = time.perf_counter()
        was_empty = os.path.isdir(folder) and not os.listdir(folder)
        try:
            summary = L.write_lerobot(folder, eps, st['version'], p.fps, p.robot.name, codec=st['codec'],
                                      overwrite=overwrite)
        except BaseException as e:
            if was_empty:                              # the writer removes only a folder it made: empty this one
                for x in os.listdir(folder):
                    q = os.path.join(folder, x)
                    shutil.rmtree(q, ignore_errors=True) if os.path.isdir(q) else os.remove(q)
            if isinstance(e, (RenderUnavailable, DatasetError)):
                raise SynthError(str(e)) from None
            raise
        write_s = time.perf_counter() - t_write
        info = R.info
        stats = dict(R.stats)
    finally:
        R.close()
    # the ground truth
    root = summary['root']
    cams_t = {}
    for cp in p.cameras:
        cams_t[cp.key] = {'sensor_uid': cp.sensor_uid, 'sensor': cp.sensor, 'imager': cp.imager, 'fps': cp.fps,
                          'fps_source': cp.fps_source, 'wrist': cp.key == p.wrist,
                          'intrinsics': intrinsics(cp.model), 'extrinsics': extrinsics(cp)}
    ep_t = {}
    counts = {k: {'episodes': 0, 'frames': 0} for k in FAULT_KINDS}
    for ep in p.episodes:
        vis, timing, extr = {}, {}, {}
        for cp in p.cameras:
            tl = ep.timelines[cp.key]
            fr = streams[(ep.index, cp.key)]
            if fr.visible is None:
                raise SynthError(f'Camera {cp.key} of episode {ep.index} was not rendered.')
            s = tl.stored
            vis[cp.key] = {'gripper_visible': list(fr.visible), 'gripper_px': list(fr.px),
                           'min_px': max(1, int(math.ceil(st['gripper_min_frac'] * cp.model.width *
                                                          cp.model.height)))}
            timing[cp.key] = {'slot': [int(k) for k in s], 'stamp_s': [float(x) for x in tl.stamp[s]],
                              'content_time_s': [float(x) for x in tl.content[s]], 'fps': cp.fps}
            extr[cp.key] = extrinsics(cp, tl.delta)
        for f in ep.faults:
            c_ = counts[f['kind']]
            c_['episodes'] += 1
            c_['frames'] += len(f.get('slots') or []) if f['kind'] in ('drop', 'usb_reset') else \
                len(f.get('frames') or [])
        ep_t[str(ep.index)] = {'faults': copy.deepcopy(ep.faults), 'extrinsics': extr, 'visibility': vis,
                               'timing': timing, 'rows': int(len(ep.t)),
                               'motion': {'name': ep.motion.name, 'source': ep.motion.source,
                                          'robot_uid': ep.motion.robot_uid, 'duration_s': float(ep.t[-1]),
                                          'joint_offset_rad': [float(x) for x in (ep.motion.offset if
                                                                                    ep.motion.offset is not None
                                                                                    else np.zeros(
                                                                                        ep.motion.model.nq))],
                                          **{k: v for k, v in ep.motion.info.items()}}}
    gen_settings = {k: v for k, v in st.items() if k != 'defaults'}
    truth = {'schema': TRUTH_SCHEMA, 'project_digest': PV.project_digest(project)['sha256'],
             'generator': {'name': 'dexvantage.core.sim.episodes', 'dexvantage': __version__, 'seed': st['seed'],
                           'settings': gen_settings, 'defaults': list(st['defaults']), 'version': st['version'],
                           'codec': summary.get('codec'), 'fps': p.fps, 'renderer': info,
                           'robot': {'uid': p.robot.uid, 'name': p.robot.name,
                                     'joint_names': list(p.episodes[0].motion.model.joint_names),
                                     'gripper_links': list(p.gripper_links)},
                           'conventions': {
                               'frames': 'indices into the camera\'s stored frames (CameraStream.frames); for drop '
                                         'and usb_reset the stored frame right after each gap, slots the missing '
                                         'slots',
                               'timing': 'slot k at t_k = k / fps; stamp_s is the stored timestamp, content_time_s '
                                         'the time the image shows the robot at',
                               'action': f'action row i = the state at t_i + {st["action_lead_rows"]} / fps',
                               'extrinsics': 'the optical frame (z forward, x right, y down) in the mount\'s link, '
                                             'or the world; quaternion (x, y, z, w)'},
                           'notes': list(p.notes)},
             'cameras': cams_t, 'episodes': ep_t}
    truth = PV.stamp(truth, project, study=SID)
    path = os.path.join(root, L.TRUTH_FILE)
    tmp = path + '.part'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(truth, fh, indent=1, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, path)
    total = time.perf_counter() - t_all
    cf = gen.frames_done
    out = {'root': root, 'version': st['version'], 'codec': summary.get('codec'), 'fps': p.fps,
           'episodes': len(p.episodes), 'rows': int(summary.get('frames') or 0), 'camera_frames': cf,
           'cameras': [cp.key for cp in p.cameras], 'renders': gen.renders, 'seconds': round(total, 3),
           'render_seconds': round(gen.render_s, 3), 'write_seconds': round(write_s, 3),
           'gpu_render_seconds': round(float(stats.get('render_seconds') or 0.0), 3),
           'camera_frames_per_s': round(cf / total, 2) if total > 0 else None,
           'renders_per_s': round(gen.renders / gen.render_s, 1) if gen.render_s > 0 else None,
           'renderer': info, 'faults': counts, 'notes': list(p.notes), 'files': summary.get('files'),
           'truth_file': L.TRUTH_FILE, 'truth': truth}
    if keep_frames:
        arr = []
        for ep in p.episodes:
            arr.append({'t': ep.t, 'q': ep.q, 'action': ep.action,
                        'cameras': {cp.key: {'timestamps': np.asarray(ep.timelines[cp.key].stamp[
                            ep.timelines[cp.key].stored], float), 'frames': streams[(ep.index, cp.key)].kept}
                            for cp in p.cameras}})
        out['arrays'] = arr
    report(1.0, f'{len(p.episodes)} episodes written')
    return out


def summary_text(s: dict) -> str:
    """One plain sentence about a generate() summary."""
    def n(x, word):
        return f'{x} {word}' + ('' if x == 1 else 's')
    placed = '; '.join(f'{FAULT_WORDS[k]} (' + (f'{n(c["frames"], "frame")} in ' if c['frames'] else '')
                       + n(c['episodes'], 'episode') + ')' for k, c in s['faults'].items() if c['episodes'])
    return (f'{s["episodes"]} synthetic episode{"s" if s["episodes"] != 1 else ""} ({s["rows"]:,} rows, '
            f'{s["camera_frames"]:,} camera frames from {len(s["cameras"])} camera'
            f'{"s" if len(s["cameras"]) != 1 else ""}) written as LeRobot {s["version"]} in {s["seconds"]:.1f} s '
            f'({s["camera_frames_per_s"] or 0:.0f} frames/s)' + (f'. Faults injected: {placed}' if placed else
                                                                 '; no faults injected') + '.')


def batch_generate(project, folder: str, settings_json: str | None = None) -> str:
    """--synth-episodes OUT_DIR [--synth-settings JSON]: generate into folder; returns the summary sentence. The
    settings are JSON text or the path of a .json file, over the project's own; raises SynthError in words."""
    given = None
    if settings_json:
        text = settings_json
        if not text.lstrip().startswith('{'):
            try:
                with open(text, encoding='utf-8') as fh:
                    text = fh.read()
            except OSError as e:
                raise SynthError(f'The settings file {os.path.basename(settings_json)} could not be read: '
                                 f'{e.strerror or e}.') from None
        try:
            given = json.loads(text)
        except ValueError as e:
            raise SynthError(f'The synthetic-episode settings are not valid JSON: {e}.') from None
        if not isinstance(given, dict):
            raise SynthError('The synthetic-episode settings must be a JSON object, e.g. {"episodes": 2}.')
    settings_of(project, given)                         # a bad setting is said before the folder is looked at
    if os.path.exists(folder) and (not os.path.isdir(folder) or os.listdir(folder)):
        raise SynthError(f'{os.path.basename(os.path.abspath(folder))} is not an empty folder: choose a new or empty '
                         'folder for the synthetic dataset.')
    s = generate(project, folder, given)
    return summary_text(s) + f' Ground truth: {s["truth_file"]}; rendered on {s["renderer"].get("renderer") or "?"}.'
