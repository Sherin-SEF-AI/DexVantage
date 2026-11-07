"""Teleop overlap and follower dexterity (spec §7.4): where a teleoperator can drive the follower robot over the task
volume, and where the follower answers badly (a dead zone).

    @jobs.study('teleop', 'Teleop overlap and dead zones', quick=False, batch='teleop')   -> --teleop OUT.json

Devices. project.tasks.teleop_devices holds one record per device (clean_device normalises it):

    {'uid', 'name', 'kind': 'leader_arm' | 'gello' | 'vr' | 'spacemouse', 'follower_uid': a robot of the project,
     'leader': {'robot_uid'} | {'library_id'} | None,
     'mapping': {'kind': 'joint' | 'cartesian', 'scale': float, 'offset': pose, 'joint_map': {leader j: follower j}},
     'volume': {'kind': 'box' | 'cylinder', 'center', 'size'} | None, 'source': {figure: where it comes from}}

The leader's comfortable workspace, mapped into the follower's frame through the retargeting:

- Leader arm and GELLO (mapping 'joint'): the leader's joints move the follower's, q_f[j_f] = scale · q_l[j_l] for each
  entry of joint_map (default: the joints of the same name; a follower joint the map leaves out is held at home). The
  leader is comfortable inside the central `comfort_fraction` of each joint's range (default 0.8, a labelled
  DexVantage default, not a measured human-comfort figure; an unlimited revolute joint: home ± 0.8 π). Its
  comfortable workspace in the follower's frame is therefore FK_f(scale · C), C the comfortable box: a voxel belongs
  to it when the follower's IK (core/robot/ik.py solve_batch, position target, orientation free) converges with the
  joint bounds replaced by scale · C. The bounds are not cut to the follower's own limits: where the mapped range runs
  past a follower stop, the follower sits at that stop while the leader is comfortable, which is a joint-limit dead
  zone. A leader whose robot_uid is the follower's own uid is a leader of the same model (the SO-100 kit: LeRobot's
  SO-100 guide builds the leader arm from the same parts through the wrist and calibrates both "to ensure that the
  leader and follower arms have the same position values when they are in the same physical position"; GELLO "has
  the same kinematic structure as the target arm", Wu et al. 2023). Mapping 'cartesian' for an arm: the leader's
  tool point in its base frame, p_d, is mapped as below and the follower solves its own IK.
- VR and space mouse (mapping 'cartesian'): the device's volume (the tracking volume or the space-mouse bounds, a box
  of edges `size` or a cylinder of [radius, height] along z, in the device frame) is mapped as
  p_world = B_f · O · (scale · p_d), B_f the follower's base and O the mapping offset (a pose in the follower's base
  frame). A device whose volume is not stated has no comfortable workspace: the study says so and does not guess one.
  Data gloves and hand trackers (kinds 'glove', 'hand_tracker') are cartesian devices the same way, their volume
  the wrist tracker's or the tracker's own. A figure comes from the device's own `source` (cited or measured), or it
  is null; a device made from a library teleop entry (core/teleop_devices.py device_from_library) keeps its
  `library_id` and carries the entry's stated figures, each with its source, in `source`.

The follower over the task volume: the task volumes are voxelised with core/voxels.py SparseGrid at `voxel_mm`
(default 25 mm, no refinement: each voxel costs an IK solve), plus `search_voxels` (default 2) layers around each
volume for the suggestion below. At each voxel centre:

    reach     IK of the follower's tool point (the gripper frame and TCP of core/grasps.gripper_geometry), position
              only, `seeds` seeds (default 4; a target none solves gets 4 × seeds fresh ones), the solution nearest
              home (ik_rule, default 'nearest_home': what a retargeting IK started from home finds), self-collision
              avoided (margin 0)
    w         Π σᵢ of the tool point's Jacobian J (MuJoCo mj_jac; the chain's joints only): √det(J Jᵀ) [Yoshikawa
              1985] when J has no more rows than columns, √det(Jᵀ J) otherwise (a planar arm within its plane)
    κ         σ_max / σ_min of the same J (1/κ is reported: 0 singular, 1 isotropic)
    joint lim the normalised distance to the nearer stop, 2·min(q − lo, hi − q)/(hi − lo) (manipulability.py), the
              smallest over the chain's joints that move the tool point (a column of J above min_column_mm per rad,
              default 1 mm: a wrist roll about the tool axis cannot block a position)
    self      the self-collision distance (core/robot/collision.py), over link pairs whose relative pose the chain's
              joints change (a gripper's own linkage, fixed relative to itself, is left out), capped at 100 mm

J is the position part ('jacobian' = 'position', the default, matching the position-only reach) or the full 6 × n
Jacobian ('full': metres and radians mixed, so κ depends on the unit; it adds the wrist singularities of a 6-joint
arm).

Dead zones (labelled default thresholds, the user's settings): a voxel the leader reaches comfortably where the
follower is 'unreachable' (no IK solution), in 'self-collision' (closer than self_margin_mm, default 10 mm), near a
'singularity' (1/κ below min_inverse_condition, default 0.1, κ > 10) or near a 'joint limit' (normalised distance
below min_limit_distance, default 0.05: within 2.5 % of the range from a stop). Causes are listed in that order; a
voxel's first cause names it. Connected dead-zone voxels of one cause (6-neighbours, scipy.ndimage.label) form a dead
zone, listed with its volume, its box and the pose of its worst voxel (the follower's joint values there).
Teleoperable = comfortable for the leader and no cause; the headline is the share of the task volume teleoperable
through at least one device.

Bimanual (two or more follower robots among the devices): per pair, the overlap of their reach (both IK solutions
exist), the share teleoperable by both, and at the handover zone (the voxels of 'handover' task volumes, else up to
handover_samples overlap voxels, seed 0) each arm reaching a point handover_gap_mm (default 100 mm) apart along the
line between their bases: the handover quality (the share where both reach without a dead-zone cause and the arms
keep arm_margin_mm, default 20 mm, apart) and the arm-to-arm margin (the smallest distance between their colliding
geometry's vertices).

The suggestion (Apply): moving a follower's base by whole voxels (up to ±search_voxels per axis), or a cartesian
device's mapping offset, shifts the follower's dexterity map (or the mapped comfortable region) rigidly over the task
voxels, so every candidate is scored from the grid already computed, without new IK: the move that most raises the
teleoperable share is suggested with the predicted share and dead-zone volume. For the primary robot (whose base is
the world frame) the base move is applied as the opposite move of everything else in the world: task volumes,
objects, surfaces, lights, fixed sensors and the other robots (apply_suggestion).

References: T. Yoshikawa 1985; P. Wu et al., GELLO, 2023; Hugging Face LeRobot SO-100 guide (SOURCES).

Known limits: reach is position-only with a free orientation (a task that needs a given gripper orientation can be
harder); IK is local, so a voxel every seed misses reads as unreachable or outside the comfortable workspace; the
comfortable fraction of the joint ranges is a default, not a human-factors model; the comfortable workspace is the
set of tool positions, not the operator's arm posture or speed; dexterity is first order at one pose; the suggestion
searches translations on the voxel lattice only (no rotations, no finer steps) and its prediction holds only to the
voxel size; obstacles and other robots are not avoided by the reach (only self-collision is), and bimanual arm-to-arm
distances are measured between mesh vertices, an upper bound of the true distance by up to the vertices' spacing;
hand-eye board placements are not moved with a primary-robot base move.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from .. import jobs
from ..frames import Pose

STUDY_ID = 'teleop'
KINDS = ('leader_arm', 'gello', 'vr', 'spacemouse', 'glove', 'hand_tracker')
KIND_TITLES = {'leader_arm': 'Leader arm', 'gello': 'GELLO', 'vr': 'VR tracking', 'spacemouse': 'Space mouse',
               'glove': 'Data glove', 'hand_tracker': 'Hand tracker'}
TRACKED_KINDS = ('vr', 'glove', 'hand_tracker')      # their region is a tracking volume (a space mouse: bounds)
MAPPINGS = ('joint', 'cartesian')
VOLUME_KINDS = ('box', 'cylinder')
CAUSES = ('unreachable', 'self-collision', 'singularity', 'joint limit')
# per-voxel state codes in the result: 0 teleoperable, 1 outside the leader's comfortable workspace, 2 + k dead zone of
# CAUSES[k], 9 the device's region is not stated
TELEOPERABLE, OUTSIDE, NOT_STATED = 0, 1, 9
STATE_WORDS = {0: 'teleoperable', 1: 'outside the leader\'s comfortable workspace', 9: 'device region not stated',
               **{2 + k: c for k, c in enumerate(CAUSES)}}
WORDS = {'good': 'clear', 'warning': 'gaps', 'critical': 'poor'}
SELF_CAP_M = 0.1                       # self-collision distances are reported up to this (m)
GOOD_PCT, WARN_PCT = 90.0, 50.0        # headline status bands (labelled defaults)
MAX_POINTS = 60_000                    # evaluation points (task voxels and the search layers) a run may solve
ARM_POINTS = 4000                      # colliding-geometry vertices kept per arm for arm-to-arm distances
RETRY = 4                              # targets no seed solved get RETRY × seeds fresh seeds (a default)

DEFAULTS = {'voxel_mm': 25.0, 'comfort_fraction': 0.8, 'min_inverse_condition': 0.1, 'min_limit_distance': 0.05,
            'self_margin_mm': 10.0, 'jacobian': 'position', 'ik_rule': 'nearest_home', 'seeds': 4,
            'search_voxels': 2, 'handover_gap_mm': 100.0, 'arm_margin_mm': 20.0, 'handover_samples': 60,
            'min_column_mm': 1.0}
LIMITS = {'voxel_mm': (5.0, 200.0), 'comfort_fraction': (0.05, 1.0), 'min_inverse_condition': (0.0, 1.0),
          'min_limit_distance': (0.0, 1.0), 'self_margin_mm': (0.0, 100.0), 'seeds': (1, 32),
          'search_voxels': (0, 4), 'handover_gap_mm': (0.0, 500.0), 'arm_margin_mm': (0.0, 200.0),
          'handover_samples': (1, 500), 'min_column_mm': (0.0, 100.0)}
SETTING_TEXT = {
    'voxel_mm': ('Voxel size', 'mm', 'default 25 mm: each voxel costs an IK solve'),
    'comfort_fraction': ('Comfortable share of each leader joint range', '', 'default 0.8, a DexVantage default'),
    'min_inverse_condition': ('Singular below 1/κ', '', 'default 0.1 (κ above 10)'),
    'min_limit_distance': ('Near a limit below', '', 'default 0.05 of the half-range'),
    'self_margin_mm': ('Self-collision margin', 'mm', 'default 10 mm'),
    'search_voxels': ('Suggestion search', 'voxels', 'default 2 voxels each way'),
}

SOURCES = {
    'lerobot_so100': {'url': 'https://huggingface.co/docs/lerobot/so100', 'title': 'Hugging Face LeRobot, SO-100 '
                      'guide', 'read': '2026-11-07',
                      'quote': 'For the leader configuration, perform Steps 1–23 [the follower\'s assembly through '
                               'the wrist] … calibrate your robot to ensure that the leader and follower arms have '
                               'the same position values when they are in the same physical position.',
                      'covers': 'the SO-100 leader arm shares the follower\'s joints through the wrist (a handle and '
                                'trigger replace the gripper); joint positions map one to one'},
    'gello2023': {'url': 'https://arxiv.org/abs/2309.13037', 'title': 'P. Wu, Y. Shentu, Z. Yi, X. Lin, P. Abbeel, '
                  '"GELLO: A General, Low-Cost, and Intuitive Teleoperation Framework for Robot Manipulators", 2023',
                  'read': '2026-11-07',
                  'quote': 'Given a target robot arm, we build a GELLO controller device that has the same kinematic '
                           'structure as the target arm',
                  'covers': 'a GELLO maps joint to joint; its own joint ranges and link lengths are not stated in the '
                            'abstract'},
}


# ====================================================================== helpers
def _r(x, d: int = 4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, d) if math.isfinite(x) else None


def _num(v, default=None):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def settings_of(project) -> dict:
    """The study's settings: project.analysis.studies['teleop'] over DEFAULTS, checked; 'defaults' names the keys left
    at their default."""
    given = dict(project.analysis.studies.get(STUDY_ID) or {})
    out, used = dict(DEFAULTS), []
    for k, v in DEFAULTS.items():
        g = given.get(k)
        if g is None:
            used.append(k)
            continue
        if isinstance(v, str):
            opts = {'jacobian': ('position', 'full'), 'ik_rule': ('nearest_home', 'most_manipulable')}[k]
            if str(g) not in opts:
                raise jobs.StudyError(f'The teleop setting {k} must be one of {", ".join(opts)}; it is {g!r}.')
            out[k] = str(g)
            continue
        x = _num(g)
        if x is None:
            raise jobs.StudyError(f'The teleop setting {k} must be a number; it is {g!r}.')
        lo, hi = LIMITS[k]
        if not lo <= x <= hi:
            raise jobs.StudyError(f'The teleop setting {k} must be between {lo:g} and {hi:g}; it is {x:g}.')
        out[k] = int(round(x)) if isinstance(v, int) else x
    out['defaults'] = used
    return out


def _pose_dict(d) -> dict:
    try:
        return Pose.from_dict(d if isinstance(d, dict) else {}).to_dict()
    except (TypeError, ValueError):
        return Pose().to_dict()


def clean_device(d, project=None) -> dict:
    """A teleop device record in the shared contract's shape, with defaults for what is missing (figures stay None)."""
    from ..project import new_uid
    d = d if isinstance(d, dict) else {}
    kind = str(d.get('kind') or 'leader_arm')
    kind = kind if kind in KINDS else 'leader_arm'
    follower = str(d.get('follower_uid') or '')
    if not follower and project is not None and project.robots:
        follower = project.primary_robot.uid
    lead = d.get('leader')
    if isinstance(lead, dict) and (lead.get('robot_uid') or lead.get('library_id')):
        lead = {k: str(lead[k]) for k in ('robot_uid', 'library_id') if lead.get(k)}
    else:
        lead = None
    m = d.get('mapping') if isinstance(d.get('mapping'), dict) else {}
    mk = str(m.get('kind') or ('joint' if kind in ('leader_arm', 'gello') else 'cartesian'))
    mk = mk if mk in MAPPINGS else 'cartesian'
    scale = _num(m.get('scale'), 1.0)
    jm = m.get('joint_map') if isinstance(m.get('joint_map'), dict) else {}
    mapping = {'kind': mk, 'scale': scale if scale != 0 else 1.0, 'offset': _pose_dict(m.get('offset')),
               'joint_map': {str(a): str(b) for a, b in jm.items() if a and b}}
    vol = d.get('volume')
    if isinstance(vol, dict) and vol.get('size') is not None:
        vk = str(vol.get('kind') or 'box')
        vk = vk if vk in VOLUME_KINDS else 'box'
        c = [float(_num(x, 0.0)) for x in list(vol.get('center') or [0, 0, 0])[:3]]
        c += [0.0] * (3 - len(c))
        s = [abs(float(_num(x, 0.0))) for x in list(vol.get('size') or [])[:3]]
        need = 3 if vk == 'box' else 2
        vol = {'kind': vk, 'center': c, 'size': (s + [0.0] * 3)[:need]} if len(s) >= need and all(
            x > 0 for x in s[:need]) else None
    else:
        vol = None
    src = d.get('source')
    if isinstance(src, str):
        src = {'device': src} if src.strip() else {}
    src = {str(k): str(v) for k, v in (src or {}).items() if v} if isinstance(src, dict) else {}
    out = {'uid': str(d.get('uid') or new_uid()), 'name': str(d.get('name') or KIND_TITLES[kind]), 'kind': kind,
           'follower_uid': follower, 'leader': lead, 'mapping': mapping, 'volume': vol, 'source': src}
    if d.get('library_id'):                     # the library teleop entry it was made from (core/teleop_devices)
        out['library_id'] = str(d['library_id'])
    return out


def new_device(project, kind: str = 'leader_arm', follower_uid: str | None = None) -> dict:
    """A new device of a kind with nothing invented: a leader arm or GELLO leads with the follower's own model (same
    joints, mapped one to one); a VR or space-mouse device has no volume until the user enters a cited or measured
    one."""
    fu = follower_uid or (project.primary_robot.uid if project.robots else '')
    n = sum(1 for d in project.tasks.teleop_devices if isinstance(d, dict) and d.get('kind') == kind) + 1
    d = {'kind': kind, 'name': f'{KIND_TITLES.get(kind, "Device")} {n}', 'follower_uid': fu}
    if kind in ('leader_arm', 'gello'):
        d['leader'] = {'robot_uid': fu}
        d['source'] = {'leader': 'a leader of the follower\'s own model (same joints), as the SO-100 kit and GELLO '
                                 'are built; enter another robot as leader if yours differs'}
    else:
        d['source'] = {'volume': 'not stated: enter the tracking volume or bounds you measured, or a published figure '
                                 'with its source'}
    return clean_device(d, project)


# ====================================================================== geometry of a device's region
def device_transform(device: dict, follower_base: np.ndarray) -> tuple[np.ndarray, float]:
    """(B_f · O as 4 × 4, scale): p_world = M[:3,:3] (scale p_d) + M[:3,3]."""
    O = Pose.from_dict(device['mapping']['offset']).matrix()
    return np.asarray(follower_base, float) @ O, float(device['mapping']['scale'])


def to_device(device: dict, follower_base, P) -> np.ndarray:
    """World points P (N, 3) in the device's frame: p_d = (M⁻¹ p) / scale."""
    M, s = device_transform(device, follower_base)
    P = np.asarray(P, float).reshape(-1, 3)
    return ((P - M[:3, 3]) @ M[:3, :3]) / s


def from_device(device: dict, follower_base, P) -> np.ndarray:
    """Device-frame points P (N, 3) in the world: p = M (scale p_d)."""
    M, s = device_transform(device, follower_base)
    return (s * np.asarray(P, float).reshape(-1, 3)) @ M[:3, :3].T + M[:3, 3]


def inside_volume(volume: dict, Pd) -> np.ndarray:
    """(N,) bool: device-frame points inside the device's box (edges `size`) or cylinder ([radius, height], axis z)."""
    Pd = np.asarray(Pd, float).reshape(-1, 3)
    c = np.asarray(volume['center'], float)
    s = np.asarray(volume['size'], float)
    d = Pd - c
    if volume['kind'] == 'cylinder':
        return (np.hypot(d[:, 0], d[:, 1]) <= s[0] + 1e-12) & (np.abs(d[:, 2]) <= s[1] / 2 + 1e-12)
    return np.all(np.abs(d) <= s / 2 + 1e-12, axis=1)


def volume_outline(volume: dict, n: int = 48) -> list:
    """Device-frame polylines of the volume's edges (box: 12 edges as 2-point lines; cylinder: two rims, 4 sides)."""
    c = np.asarray(volume['center'], float)
    s = np.asarray(volume['size'], float)
    if volume['kind'] == 'cylinder':
        t = np.linspace(0, 2 * np.pi, n + 1)
        rim = np.c_[s[0] * np.cos(t), s[0] * np.sin(t), np.zeros_like(t)]
        out = [rim + c + [0, 0, -s[1] / 2], rim + c + [0, 0, s[1] / 2]]
        for a in (0, n // 4, n // 2, 3 * n // 4):
            out.append(np.array([rim[a] + c + [0, 0, -s[1] / 2], rim[a] + c + [0, 0, s[1] / 2]]))
        return out
    lo, hi = c - s / 2, c + s / 2
    C = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    out = []
    for i in range(8):
        for j in range(i + 1, 8):
            if int(np.sum(C[i] != C[j])) == 1:
                out.append(np.array([C[i], C[j]]))
    return out


# ====================================================================== joints and the comfortable box
def comfort_box(model, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) of the central `fraction` of each joint's range (unlimited revolute: home ± fraction · π; unlimited
    slide: held at home)."""
    import mujoco
    lo, hi = model.lower.copy(), model.upper.copy()
    fin = np.isfinite(lo) & np.isfinite(hi)
    mid = np.where(fin, 0.5 * (np.where(fin, lo, 0) + np.where(fin, hi, 0)), model.home)
    half = np.where(fin, 0.5 * (np.where(fin, hi, 0) - np.where(fin, lo, 0)) * fraction, 0.0)
    hinge = np.array([int(model.m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE) for j in model._jid], bool)
    half = np.where(~fin & hinge, fraction * np.pi, half)
    return mid - half, mid + half


def joint_map_of(device: dict, leader, follower, notes: list) -> dict:
    """{follower joint index: leader joint index}: the device's joint_map (leader name → follower name), else the
    joints of the same name, else the chain joints in order when the two chains are equally long."""
    jm = device['mapping']['joint_map']
    fi = {n: i for i, n in enumerate(follower.joint_names)}
    li = {n: i for i, n in enumerate(leader.joint_names)}
    out = {}
    if jm:
        for a, b in jm.items():
            if a in li and b in fi:
                out[fi[b]] = li[a]
            else:
                notes.append(f'The joint map entry {a} → {b} names a joint the leader or follower does not have; it '
                             'is left out.')
        return out
    same = {fi[n]: li[n] for n in fi if n in li}
    if same:
        return same
    if leader.nq == follower.nq:
        notes.append('The leader and follower share no joint names: their joints are mapped in order.')
        return {k: k for k in range(follower.nq)}
    notes.append('The leader and follower share no joint names and have different joint counts: set the joint map.')
    return {}


def proxy(model, lo, hi):
    """A shallow copy of a RobotModel with other joint bounds (for IK inside a box; FK is the model's own)."""
    pm = copy.copy(model)
    pm.lower = np.asarray(lo, float).copy()
    pm.upper = np.asarray(hi, float).copy()
    pm.home = np.clip(model.home, pm.lower, pm.upper)
    return pm


# ====================================================================== the follower's measures
class Arm:
    """A robot's tool point and the per-point measures the study needs (see the module docstring)."""

    def __init__(self, project, robot_uid: str, st: dict):
        from .. import grasps as GR
        from ..robot import manipulability as M
        from ..robot.model import get_model
        self.robot = project.robot(robot_uid)
        self.model = m = get_model(project, robot_uid)
        try:
            g = GR.gripper_geometry(project, robot_uid)
            self.link, self.tcp = g['frame'], np.asarray(g['tcp'], float)
            self.tcp_rule = g['rule']
        except Exception:                               # noqa: BLE001 — a robot without a gripper frame
            self.link = self.robot.gripper_frame if self.robot.gripper_frame in m.link_names else m.link_names[-1]
            self.tcp, self.tcp_rule = np.zeros(3), 'the frame origin (no gripper geometry)'
        self.T_tcp = np.eye(4)
        self.T_tcp[:3, 3] = self.tcp
        self.chain = M.chain_mask(m, self.link)
        self.st = st
        self._relevant = {}

    @property
    def base(self) -> np.ndarray:
        return self.model.base

    def solve(self, P, model=None, q0=None, cancel=None, progress=None):
        from ..robot import ik
        mdl = model or self.model
        P = np.asarray(P, float).reshape(-1, 3)
        if not len(P):
            return None
        return ik.solve_batch(mdl, self.link, P, q0=q0, seeds=self.st['seeds'], tcp=self.T_tcp,
                              rule=self.st['ik_rule'], collision=True, margin=0.0, retry_seeds=RETRY * self.st['seeds'],
                              cancel=cancel, progress=progress)

    def jacobians(self, Q) -> np.ndarray:
        """(N, rows, n_chain) Jacobians of the tool point over the chain's joints."""
        Q = self.model.batch_q(Q)
        J = np.empty((len(Q), 6, self.model.nq))
        for k, q in enumerate(Q):
            J[k] = self.model.jacobian(q, self.link, self.tcp)
        J = J[:, :3] if self.st['jacobian'] == 'position' else J
        return J[:, :, self.chain]

    def dexterity(self, Q) -> dict:
        """{'w', 'inverse_condition', 'condition', 'limit_distance', 'limit_joint'} at joint vectors Q (N, nq)."""
        from ..robot import manipulability as M
        Q = self.model.batch_q(Q)
        J = self.jacobians(Q)
        s = M.singular_values(J)
        if s.shape[-1] == 0:
            z = np.zeros(len(Q))
            return {'w': z, 'inverse_condition': z, 'condition': np.full(len(Q), np.inf),
                    'limit_distance': np.ones(len(Q)), 'limit_joint': np.full(len(Q), -1)}
        w = np.prod(s, axis=-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            inv = np.where(s[:, 0] > 0, s[:, -1] / np.where(s[:, 0] > 0, s[:, 0], 1), 0.0)
            kap = np.where(s[:, -1] > 0, s[:, 0] / np.where(s[:, -1] > 0, s[:, -1], 1), np.inf)
        Jp = J[:, :3]
        col = np.zeros((len(Q), self.model.nq))
        col[:, self.chain] = np.linalg.norm(Jp, axis=1)
        moves = col > self.st['min_column_mm'] / 1000.0
        ld = M.limit_distance(Q, self.model.lower, self.model.upper)
        ldm = np.where(moves & self.chain, ld, np.inf)
        jmin = np.argmin(ldm, axis=1)
        dmin = ldm[np.arange(len(Q)), jmin]
        none = ~np.isfinite(dmin)
        return {'w': w, 'inverse_condition': inv, 'condition': kap, 'limit_distance': np.where(none, 1.0, dmin),
                'limit_joint': np.where(none, -1, jmin)}

    def relevant(self, a: str, b: str) -> bool:
        """Whether the chain's joints change the relative pose of links a and b."""
        from ..robot.manipulability import chain_mask
        k = (a, b)
        if k not in self._relevant:
            try:
                ma, mb = chain_mask(self.model, a), chain_mask(self.model, b)
                self._relevant[k] = bool(((ma ^ mb) & self.chain).any())
            except KeyError:
                self._relevant[k] = True
        return self._relevant[k]

    def self_distance(self, q) -> tuple[float, tuple | None]:
        """(the smallest self-collision distance over the relevant link pairs, capped at SELF_CAP_M; the pair)."""
        from ..robot import collision as C
        r = C.self_collision(self.model, q, margin=SELF_CAP_M)
        for a, b, d in r.pairs:
            if d < SELF_CAP_M and self.relevant(a, b):
                return float(d), (a, b)
        return SELF_CAP_M, None

    def self_distances(self, Q, cancel=None) -> tuple[np.ndarray, list]:
        Q = self.model.batch_q(Q)
        d = np.full(len(Q), SELF_CAP_M)
        pairs = [None] * len(Q)
        for k, q in enumerate(Q):
            if cancel is not None and k % 256 == 0 and cancel():
                break
            if np.isfinite(q).all():
                d[k], pairs[k] = self.self_distance(q)
        return d, pairs

    def vertices(self) -> list:
        """[(link index, (V, 3) link-frame vertices)] of the colliding geometry, thinned to about ARM_POINTS."""
        v = getattr(self, '_verts', None)
        if v is None:
            geoms = self.model.collision_geoms() or self.model.visual_meshes()
            total = sum(len(V) for _l, V, _F, _c in geoms) or 1
            stride = max(1, int(math.ceil(total / ARM_POINTS)))
            v = [(self.model.link_index(ln), np.asarray(V, float)[::stride]) for ln, V, _F, _c in geoms if len(V)]
            self._verts = v
        return v

    def posed_vertices(self, q) -> tuple[np.ndarray, np.ndarray]:
        T = self.model.fk_array(q)
        P, L = [], []
        for li, V in self.vertices():
            P.append(V @ T[li, :3, :3].T + T[li, :3, 3])
            L.append(np.full(len(V), li))
        if not P:
            return np.zeros((0, 3)), np.zeros(0, int)
        return np.concatenate(P), np.concatenate(L)


def arm_distance(a: Arm, qa, b: Arm, qb) -> tuple[float, tuple | None]:
    """The smallest distance between the two arms' colliding-geometry vertices at qa and qb, and the link pair."""
    from scipy.spatial import cKDTree
    Pa, La = a.posed_vertices(qa)
    Pb, Lb = b.posed_vertices(qb)
    if not len(Pa) or not len(Pb):
        return float('inf'), None
    d, i = cKDTree(Pb).query(Pa, k=1)
    k = int(np.argmin(d))
    return float(d[k]), (a.model.link_names[int(La[k])], b.model.link_names[int(Lb[int(i[k])])])


# ====================================================================== the grid
class Grid:
    """The task voxels (core/voxels.py SparseGrid at voxel_mm, no refinement) and the evaluation lattice: each volume's
    voxels plus `g` layers round it, indexed so a whole-voxel shift is an index shift."""

    def __init__(self, project, voxel_mm: float, g: int):
        from ..voxels import SparseGrid, dims
        sg = SparseGrid.from_project(project, voxel_mm=voxel_mm, refine_mm=voxel_mm)
        self.h = h = float(sg.voxel_m)
        self.task = sg.centers.astype(float)
        self.task_volume = sg.volume_index.astype(int)
        self.volumes = list(sg.volumes)
        self.g = g
        self.lo, self.dims, self.off = [], [], []
        E, tidx, n = [], np.zeros(len(self.task), int), 0
        for i, v in enumerate(self.volumes):
            d = dims(v.size, h)
            lo = np.asarray(v.center, float) - d * h / 2
            D = d + 2 * g
            zz, yy, xx = np.meshgrid(np.arange(D[2]), np.arange(D[1]), np.arange(D[0]), indexing='ij')
            I = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], 1) - g
            E.append(lo + (I + 0.5) * h)
            sel = np.flatnonzero(self.task_volume == i)
            ti = np.rint((self.task[sel] - lo) / h - 0.5).astype(int)
            tidx[sel] = n + self._flat(ti + g, D)
            self.lo.append(lo)
            self.dims.append(d)
            self.off.append(n)
            n += int(np.prod(D))
        self.eval = np.concatenate(E) if E else np.zeros((0, 3))
        self.task_eval = tidx
        self.task_ijk = [np.rint((self.task[self.task_volume == i] - self.lo[i]) / h - 0.5).astype(int)
                         for i in range(len(self.volumes))]

    @staticmethod
    def _flat(ijk, D) -> np.ndarray:
        return ijk[:, 0] + D[0] * (ijk[:, 1] + D[1] * ijk[:, 2])

    def shifted(self, s) -> np.ndarray:
        """(N_task,) evaluation indices of each task voxel's centre minus s · h (s an integer 3-vector, |s| ≤ g)."""
        out = np.empty(len(self.task), int)
        s = np.asarray(s, int)
        for i in range(len(self.volumes)):
            sel = np.flatnonzero(self.task_volume == i)
            D = self.dims[i] + 2 * self.g
            out[sel] = self.off[i] + self._flat(self.task_ijk[i] - s + self.g, D)
        return out


# ====================================================================== per device
def _causes(st, comf, reach, dex, sd) -> np.ndarray:
    """(N, 4) bool cause flags (CAUSES order), only where the leader is comfortable."""
    c = np.zeros((len(comf), 4), bool)
    c[:, 0] = comf & ~reach
    ok = comf & reach
    c[:, 1] = ok & (sd < st['self_margin_mm'] / 1000.0)
    c[:, 2] = ok & (dex['inverse_condition'] < st['min_inverse_condition'])
    c[:, 3] = ok & (dex['limit_distance'] < st['min_limit_distance'])
    return c


def _state(comf, cause, stated: bool) -> np.ndarray:
    if not stated:
        return np.full(len(comf), NOT_STATED, int)
    s = np.where(comf, TELEOPERABLE, OUTSIDE)
    anyc = cause.any(axis=1)
    first = np.argmax(cause, axis=1)
    return np.where(anyc, 2 + first, s).astype(int)


def evaluate_device(project, device: dict, arm: Arm, follower_eval: dict, grid: Grid, st: dict, notes: list,
                    cancel=None, progress=None) -> dict:
    """The device's comfortable region and the follower's measures where it would be driven, at every evaluation
    point: {'stated', 'comf', 'reach', 'q', 'dex', 'sd', 'pairs', 'cause', 'how', 'leader'}."""
    from ..robot.model import get_model
    E = grid.eval
    N = len(E)
    mk = device['mapping']['kind']
    out = {'stated': True, 'how': '', 'leader': None}
    lead = device.get('leader') or {}
    if device['kind'] in ('leader_arm', 'gello'):
        if lead.get('library_id') and not lead.get('robot_uid'):
            out['stated'] = False
            from .. import library as _lib
            le = _lib.get(lead['library_id'])
            if le is not None and le.kind == 'teleop':
                notes.append(f'{device["name"]}: the library entry "{le.name}" states the leader\'s figures but no '
                             'kinematics, and it is not stated to lead this follower; add the leader arm as a robot '
                             '(Robot ▸ Import) and choose it as the leader.')
            else:
                notes.append(f'{device["name"]}: the leader "{lead["library_id"]}" is not in the library, so it has '
                             'no kinematics here; add the leader arm as a robot (Robot ▸ Import) and choose it as '
                             'the leader.')
        elif not lead.get('robot_uid') or project.robot(lead['robot_uid']) is None:
            out['stated'] = False
            notes.append(f'{device["name"]}: no leader robot is chosen, so its workspace is not known; choose the '
                         'leader (the follower itself for a leader of the same model).')
    elif device.get('volume') is None:
        out['stated'] = False
        tracked = device['kind'] in TRACKED_KINDS
        what = 'tracking volume' if tracked else 'bounds'
        notes.append(f'{device["name"]}: its {what} {"is" if tracked else "are"} not stated, so its '
                     'comfortable workspace is not known. Enter a measured or published figure with its source; '
                     'DexVantage does not guess one.')
    if not out['stated']:
        z = np.zeros(N, bool)
        out.update(comf=z, reach=follower_eval['reach'], q=follower_eval['q'], dex=follower_eval['dex'],
                   sd=follower_eval['sd'], pairs=follower_eval['pairs'], cause=np.zeros((N, 4), bool))
        return out
    fm = arm.model
    if device['kind'] in ('leader_arm', 'gello') and mk == 'joint':
        lm = fm if lead['robot_uid'] == arm.robot.uid else get_model(project, lead['robot_uid'])
        out['leader'] = {'robot_uid': lead['robot_uid'], 'same_model': lm is fm,
                         'name': project.robot(lead['robot_uid']).name}
        llo, lhi = comfort_box(lm, st['comfort_fraction'])
        jmap = joint_map_of(device, lm, fm, notes)
        s = float(device['mapping']['scale'])
        lo, hi = fm.home.copy(), fm.home.copy()                  # follower joints the map leaves out are held
        for jf, jl in jmap.items():
            a, b = sorted((s * llo[jl], s * lhi[jl]))
            lo[jf], hi[jf] = a, b
        unmapped = [fm.joint_names[k] for k in np.flatnonzero(arm.chain) if k not in jmap]
        if unmapped:
            notes.append(f'{device["name"]}: the joint map leaves {", ".join(unmapped)} out; '
                         f'{"it is" if len(unmapped) == 1 else "they are"} held at home.')
        pm = proxy(fm, lo, hi)
        q0 = np.clip(np.where(np.isfinite(follower_eval['q']), follower_eval['q'], fm.home), lo, hi)
        b = arm.solve(E, pm, q0=q0, cancel=cancel, progress=progress)
        comf = b.ok | np.array([r == 'self-collision' for r in b.reason], bool)
        q = np.where(comf[:, None], b.q, follower_eval['q'])
        reach = comf.copy() | follower_eval['reach']
        dex = {k: v.copy() for k, v in follower_eval['dex'].items()}
        sd, pairs = follower_eval['sd'].copy(), list(follower_eval['pairs'])
        idx = np.flatnonzero(comf)
        if len(idx):
            d2 = arm.dexterity(q[idx])
            for k in dex:
                dex[k][idx] = d2[k]
            s2, p2 = arm.self_distances(q[idx], cancel)
            sd[idx] = s2
            for a, k in enumerate(idx):
                pairs[k] = p2[a]
        out['how'] = (f'joint to joint (scale {s:g}) from '
                      + ('a leader of the follower\'s own model' if lm is fm else out['leader']['name'])
                      + f', inside {st["comfort_fraction"]:g} of each leader joint\'s range')
        out['comfort_box'] = {'joints': [fm.joint_names[k] for k in range(fm.nq)], 'lower': [_r(x) for x in lo],
                              'upper': [_r(x) for x in hi]}
    else:
        if device['kind'] in ('leader_arm', 'gello'):
            ru = lead['robot_uid']
            la = Arm(project, ru, st) if ru != arm.robot.uid else arm
            lm = la.model
            out['leader'] = {'robot_uid': ru, 'same_model': la is arm, 'name': project.robot(ru).name}
            llo, lhi = comfort_box(lm, st['comfort_fraction'])
            Pd = to_device(device, arm.base, E)
            Pl = Pd @ lm.base[:3, :3].T + lm.base[:3, 3]
            b = la.solve(Pl, proxy(lm, llo, lhi), cancel=cancel, progress=progress)
            comf = b.ok | np.array([r == 'self-collision' for r in b.reason], bool)
            out['how'] = (f'the leader\'s tool position inside {st["comfort_fraction"]:g} of each joint\'s range, '
                          f'scaled {device["mapping"]["scale"]:g} and offset into the follower\'s base frame')
        else:
            comf = inside_volume(device['volume'], to_device(device, arm.base, E))
            out['how'] = (f'the device\'s {device["volume"]["kind"]} scaled {device["mapping"]["scale"]:g} and offset '
                          'into the follower\'s base frame')
        q, reach = follower_eval['q'], follower_eval['reach']
        dex, sd, pairs = follower_eval['dex'], follower_eval['sd'], follower_eval['pairs']
    out.update(comf=comf, reach=reach, q=q, dex=dex, sd=sd, pairs=pairs,
               cause=_causes(st, comf, reach, dex, sd))
    return out


def evaluate_follower(arm: Arm, grid: Grid, cancel=None, progress=None) -> dict:
    """The follower's own IK solution and measures at every evaluation point."""
    b = arm.solve(grid.eval, cancel=cancel, progress=progress)
    n = len(grid.eval)
    if b is None:
        z = np.zeros(n)
        return {'reach': np.zeros(n, bool), 'q': np.zeros((n, arm.model.nq)), 'dex': {}, 'sd': z, 'pairs': [],
                'reason': []}
    reach = b.ok | np.array([r == 'self-collision' for r in b.reason], bool)
    q = np.where(reach[:, None], b.q, np.nan)
    dex = {'w': np.zeros(n), 'inverse_condition': np.zeros(n), 'condition': np.full(n, np.inf),
           'limit_distance': np.ones(n), 'limit_joint': np.full(n, -1)}
    idx = np.flatnonzero(reach)
    sd, pairs = np.full(n, SELF_CAP_M), [None] * n
    if len(idx):
        d = arm.dexterity(q[idx])
        for k in dex:
            dex[k][idx] = d[k]
        s2, p2 = arm.self_distances(q[idx], cancel)
        sd[idx] = s2
        for a, k in enumerate(idx):
            pairs[k] = p2[a]
    return {'reach': reach, 'q': q, 'dex': dex, 'sd': sd, 'pairs': pairs, 'reason': list(b.reason)}


# ====================================================================== dead zones
def _cause_text(cause: str, arm: Arm, ev: dict, k: int, st: dict) -> str:
    m = arm.model
    if cause == 'unreachable':
        return 'The follower has no IK solution here while the leader is comfortable.'
    if cause == 'self-collision':
        pr = ev['pairs'][k]
        who = f'{pr[0]} and {pr[1]}' if pr else 'two links'
        return (f'{who} come {ev["sd"][k] * 1000:.0f} mm apart (the margin is {st["self_margin_mm"]:g} mm, '
                'a default).')
    if cause == 'singularity':
        kap = ev['dex']['condition'][k]
        return (f'The follower is near a singularity: κ = {kap:.0f} (1/κ {ev["dex"]["inverse_condition"][k]:.3f}, '
                f'below {st["min_inverse_condition"]:g}, a default).' if np.isfinite(kap) else
                'The follower is singular here (it cannot move the tool in every direction).')
    j = int(ev['dex']['limit_joint'][k])
    name = m.joint_names[j] if j >= 0 else 'a joint'
    share = ev['dex']['limit_distance'][k] * 50
    where = 'is at its stop' if share < 0.05 else f'is {share:.1f} % of its range from a stop'
    return f'{name} {where} (near a limit below {st["min_limit_distance"] * 50:g} %, a default).'


def zones(arm: Arm, ev: dict, grid: Grid, st: dict, state_task: np.ndarray) -> list:
    """Connected dead-zone voxels of one cause, worst first."""
    from scipy import ndimage
    h = grid.h
    out = []
    ntot = max(len(grid.task), 1)
    for vi, v in enumerate(grid.volumes):
        sel = np.flatnonzero(grid.task_volume == vi)
        ijk = grid.task_ijk[vi]
        d = grid.dims[vi]
        for ci, cause in enumerate(CAUSES):
            mask = state_task[sel] == 2 + ci
            if not mask.any():
                continue
            vol = np.zeros(tuple(d), bool)
            vol[tuple(ijk[mask].T)] = True
            lab, n = ndimage.label(vol)
            lab_of = lab[tuple(ijk.T)]
            for z in range(1, n + 1):
                mem = sel[lab_of == z]
                ek = grid.task_eval[mem]
                if cause == 'self-collision':
                    worst = int(np.argmin(ev['sd'][ek]))
                elif cause == 'singularity':
                    worst = int(np.argmin(ev['dex']['inverse_condition'][ek]))
                elif cause == 'joint limit':
                    worst = int(np.argmin(ev['dex']['limit_distance'][ek]))
                else:
                    c = grid.task[mem].mean(axis=0)
                    worst = int(np.argmin(np.linalg.norm(grid.task[mem] - c, axis=1)))
                k = int(ek[worst])
                C = grid.task[mem]
                q = ev['q'][k]
                out.append({'cause': cause, 'volume': v.name, 'voxels': int(len(mem)),
                            'volume_m3': _r(len(mem) * h ** 3, 7), 'pct': _r(100.0 * len(mem) / ntot, 2),
                            'lo': [_r(x) for x in C.min(axis=0) - h / 2], 'hi': [_r(x) for x in C.max(axis=0) + h / 2],
                            'center': [_r(x) for x in C.mean(axis=0)], 'worst_point': [_r(x) for x in grid.task[
                                mem[worst]]],
                            'q': [_r(x, 6) for x in q] if np.isfinite(q).all() else None,
                            'text': _cause_text(cause, arm, ev, k, st)})
    out.sort(key=lambda z: (-z['voxels'], CAUSES.index(z['cause'])))
    return out


# ====================================================================== the suggestion
def _good(ev) -> np.ndarray:
    return ev['comf'] & ~ev['cause'].any(axis=1) if ev['stated'] else np.zeros(len(ev['comf']), bool)


def _dead(ev) -> np.ndarray:
    return ev['cause'].any(axis=1) if ev['stated'] else np.zeros(len(ev['comf']), bool)


def suggest(project, devices: list, evs: list, arms: dict, grid: Grid, st: dict) -> dict | None:
    """The whole-voxel base or mapping-offset move that most raises the share teleoperable through some device."""
    g = grid.g
    if g <= 0 or not devices:
        return None
    t0 = grid.task_eval
    good0 = [_good(e)[t0] for e in evs]
    dead0 = [_dead(e)[t0] for e in evs]
    base = np.any(good0, axis=0) if good0 else np.zeros(len(t0), bool)
    n0, d0 = int(base.sum()), int(np.any(dead0, axis=0).sum()) if dead0 else 0
    rng = range(-g, g + 1)
    shifts = [np.array([x, y, z]) for x in rng for y in rng for z in rng if (x, y, z) != (0, 0, 0)]
    best = None
    for s in shifts:
        idx = grid.shifted(s)
        for fu in dict.fromkeys(d['follower_uid'] for d in devices):
            if fu not in arms:
                continue
            gd, dd = list(good0), list(dead0)
            for k, d in enumerate(devices):
                if d['follower_uid'] == fu:
                    gd[k], dd[k] = _good(evs[k])[idx], _dead(evs[k])[idx]
            cand = ('base_offset', fu, None, s, int(np.any(gd, axis=0).sum()), int(np.any(dd, axis=0).sum()))
            best = _better(best, cand)
        for k, d in enumerate(devices):
            ev = evs[k]
            if d['mapping']['kind'] != 'cartesian' or not ev['stated']:
                continue
            comf = ev['comf'][idx]
            ok_here = ev['reach'][t0] & ~_causes(st, np.ones(len(t0), bool), ev['reach'][t0],
                                                 {kk: vv[t0] for kk, vv in ev['dex'].items()},
                                                 ev['sd'][t0]).any(axis=1)
            gd, dd = list(good0), list(dead0)
            gd[k] = comf & ok_here
            dd[k] = comf & ~ok_here
            best = _better(best, ('mapping_offset', d['follower_uid'], d['uid'], s, int(np.any(gd, axis=0).sum()),
                                  int(np.any(dd, axis=0).sum())))
    if best is None or best[4] <= n0:
        return None
    kind, fu, du, s, n1, d1 = best
    dvec = s * grid.h
    robot = project.robot(fu)
    ntot = max(len(t0), 1)
    h3 = grid.h ** 3
    what = ', '.join(f'{v * 1000:+.0f} mm in {a}' for a, v in zip('xyz', dvec) if abs(v) > 1e-12)
    if kind == 'base_offset':
        text = f'Move {robot.name}\'s base {what}'
    else:
        dn = next(d['name'] for d in devices if d['uid'] == du)
        text = f'Move {dn}\'s mapping offset {what}'
    blocked = ''
    if kind == 'base_offset' and robot.primary:
        locked = [s_.name for s_ in project.variant.sensors if s_.locked and not s_.mount.on_link]
        if locked:
            blocked = (f'{robot.name} is the primary robot, so its base move is applied by moving the rest of the '
                       f'world the other way, and {", ".join(locked)} {"is" if len(locked) == 1 else "are"} locked: '
                       'unlock to apply, or move the arm on the real table.')
    return {'kind': kind, 'robot_uid': fu, 'device_uid': du, 'offset_m': [_r(x, 6) for x in dvec],
            'voxels': [int(x) for x in s], 'primary': bool(robot.primary) if kind == 'base_offset' else False,
            'before_pct': _r(100.0 * n0 / ntot, 2), 'after_pct': _r(100.0 * n1 / ntot, 2),
            'dead_before_m3': _r(d0 * h3, 7), 'dead_after_m3': _r(d1 * h3, 7),
            'text': f'{text}: teleoperable {100.0 * n0 / ntot:.1f} % → {100.0 * n1 / ntot:.1f} % of the task volume, '
                    f'dead zones {d0 * h3 * 1e6:.0f} → {d1 * h3 * 1e6:.0f} cm³ (predicted from the grid, to the '
                    f'{grid.h * 1000:.0f} mm voxel).', 'blocked': blocked}


def _better(best, cand):
    if best is None:
        return cand
    key = (lambda c: (c[4], -c[5], -int(np.abs(c[3]).sum()), c[0] == 'mapping_offset'))
    return cand if key(cand) > key(best) else best


def _shift_pose(d: dict, dv) -> dict:
    p = Pose.from_dict(d if isinstance(d, dict) else {})
    p.position = p.position + dv
    return p.to_dict()


def suggestion_sections(project, sug: dict) -> list:
    """The project sections apply_suggestion changes (the undo step's keys)."""
    if sug['kind'] == 'mapping_offset':
        return ['tasks']
    robot = project.robot(sug['robot_uid'])
    return ['robots', 'workspace', 'variants'] if robot is not None and robot.primary else ['robots']


def apply_suggestion(project, sug: dict) -> list:
    """Make the suggested move on project (in place); returns the sections changed (for one undo step). A primary
    robot's base move is the opposite move of everything else in the world (its base stays the world frame)."""
    dv = np.asarray(sug['offset_m'], float)
    if sug['kind'] == 'mapping_offset':
        robot = project.robot(sug['robot_uid'])
        Rb = robot.base.matrix()[:3, :3] if robot is not None else np.eye(3)
        for d in project.tasks.teleop_devices:
            if isinstance(d, dict) and d.get('uid') == sug['device_uid']:
                m = d.setdefault('mapping', {})
                m['offset'] = _shift_pose(m.get('offset'), Rb.T @ dv)      # the offset is in the follower's base frame
        return ['tasks']
    robot = project.robot(sug['robot_uid'])
    if robot is None:
        return []
    if not robot.primary:
        robot.base = Pose(robot.base.position + dv, robot.base.quaternion)
        return ['robots']
    w = project.workspace
    for v in w.volumes:
        v.center = tuple(float(x) for x in np.asarray(v.center, float) - dv)
    for o in w.objects:
        o.pose = Pose(o.pose.position - dv, o.pose.quaternion)
    for sf in w.surfaces:
        c = np.asarray(sf.get('center') or (0.0, 0.0), float).reshape(-1)[:2]
        sf['center'] = [float(c[0] - dv[0]), float(c[1] - dv[1])]
        sf['z'] = float(sf.get('z', 0.0)) - float(dv[2])
    for li in w.lights:
        if isinstance(li, dict) and isinstance(li.get('pose'), dict):
            li['pose'] = _shift_pose(li['pose'], -dv)
    for var in project.variants:
        for s in var.sensors:
            if not s.mount.on_link:
                s.mount.pose = Pose(s.mount.pose.position - dv, s.mount.pose.quaternion)
    for r in project.robots:
        if r.uid != robot.uid:
            r.base = Pose(r.base.position - dv, r.base.quaternion)
    return ['robots', 'workspace', 'variants']


# ====================================================================== bimanual
def bimanual(project, arms: dict, fevs: dict, devices: list, evs: list, grid: Grid, st: dict,
             cancel=None) -> dict | None:
    """Reach overlap, handover quality and arm-to-arm margin for each pair of follower robots."""
    fol = [u for u in dict.fromkeys(d['follower_uid'] for d in devices) if u in arms]
    if len(fol) < 2:
        return None
    t0 = grid.task_eval
    ntot = max(len(t0), 1)
    rng = np.random.default_rng(0)
    pairs = []
    for i in range(len(fol)):
        for j in range(i + 1, len(fol)):
            ua, ub = fol[i], fol[j]
            A, B = arms[ua], arms[ub]
            ra, rb = fevs[ua]['reach'][t0], fevs[ub]['reach'][t0]
            both = ra & rb
            ga = np.any([_good(e)[t0] for d, e in zip(devices, evs) if d['follower_uid'] == ua], axis=0)
            gb = np.any([_good(e)[t0] for d, e in zip(devices, evs) if d['follower_uid'] == ub], axis=0)
            hv = [k for k, v in enumerate(grid.volumes) if v.kind == 'handover']
            cand = np.flatnonzero(np.isin(grid.task_volume, hv)) if hv else np.flatnonzero(both)
            where = 'the handover volumes' if hv else 'the voxels both arms reach'
            if len(cand) > st['handover_samples']:
                cand = np.sort(rng.choice(cand, int(st['handover_samples']), replace=False))
            res = {'a': ua, 'b': ub, 'a_name': A.robot.name, 'b_name': B.robot.name,
                   'overlap_pct': _r(100.0 * both.sum() / ntot, 2), 'both_teleoperable_pct':
                   _r(100.0 * (ga & gb).sum() / ntot, 2), 'handover': {'where': where, 'voxels': int(len(cand))}}
            if len(cand) and not (cancel and cancel()):
                u = B.base[:3, 3] - A.base[:3, 3]
                u[2] = 0.0
                u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-9 else np.array([1.0, 0.0, 0.0])
                gap = st['handover_gap_mm'] / 1000.0
                C = grid.task[cand]
                ba = A.solve(C - u * gap / 2, cancel=cancel)
                bb = B.solve(C + u * gap / 2, cancel=cancel)
                da, db = A.dexterity(np.where(ba.ok[:, None], ba.q, A.model.home)), \
                    B.dexterity(np.where(bb.ok[:, None], bb.q, B.model.home))
                ok = ba.ok & bb.ok
                margin = np.full(len(cand), np.nan)
                links = [None] * len(cand)
                for k in np.flatnonzero(ok):
                    margin[k], links[k] = arm_distance(A, ba.q[k], B, bb.q[k])
                sa, _p = A.self_distances(np.where(ba.ok[:, None], ba.q, A.model.home))
                sb, _p = B.self_distances(np.where(bb.ok[:, None], bb.q, B.model.home))
                clean = ok & (da['inverse_condition'] >= st['min_inverse_condition']) & \
                    (db['inverse_condition'] >= st['min_inverse_condition']) & \
                    (da['limit_distance'] >= st['min_limit_distance']) & \
                    (db['limit_distance'] >= st['min_limit_distance']) & \
                    (sa >= st['self_margin_mm'] / 1000) & (sb >= st['self_margin_mm'] / 1000)
                good = clean & (np.nan_to_num(margin, nan=-1) >= st['arm_margin_mm'] / 1000)
                fm = margin[np.isfinite(margin)]
                worst = int(np.nanargmin(np.where(ok, margin, np.inf))) if ok.any() else None
                res['handover'].update({
                    'quality_pct': _r(100.0 * good.sum() / len(cand), 2),
                    'reached_pct': _r(100.0 * ok.sum() / len(cand), 2),
                    'median_inverse_condition': _r(np.median(np.minimum(da['inverse_condition'],
                                                                        db['inverse_condition'])[ok]), 4)
                    if ok.any() else None,
                    'arm_margin_min_m': _r(fm.min(), 4) if len(fm) else None,
                    'arm_margin_median_m': _r(np.median(fm), 4) if len(fm) else None,
                    'gap_m': _r(gap, 4),
                    'worst': None if worst is None else {
                        'point': [_r(x) for x in C[worst]], 'margin_m': _r(margin[worst], 4),
                        'links': list(links[worst]) if links[worst] else None,
                        'q_a': [_r(x, 6) for x in ba.q[worst]], 'q_b': [_r(x, 6) for x in bb.q[worst]]}})
            pairs.append(res)
    return {'pairs': pairs, 'note': 'Arm-to-arm distances are measured between the colliding geometry\'s vertices '
                                    f'(about {ARM_POINTS} per arm): an upper bound of the true distance.'}


# ====================================================================== the study
def _status(pct) -> str:
    if pct is None:
        return 'warning'
    return 'good' if pct >= GOOD_PCT else ('warning' if pct >= WARN_PCT else 'critical')


@jobs.study(STUDY_ID, 'Teleop overlap and dead zones', quick=False, batch='teleop', needs_robot=True)
def run(project, ctx: jobs.Ctx) -> dict:
    st = settings_of(project)
    if not project.workspace.volumes:
        raise jobs.StudyError('There is no task volume to check teleop over: add one (Workspace ▸ Add task volume).')
    notes = []
    devices = [clean_device(d, project) for d in project.tasks.teleop_devices if isinstance(d, dict)]
    usable = []
    for d in devices:
        if project.robot(d['follower_uid']) is None:
            notes.append(f'{d["name"]}: its follower robot is not in the project; the device is left out.')
        else:
            usable.append(d)
    grid = Grid(project, st['voxel_mm'], st['search_voxels'])
    if not len(grid.task):
        raise jobs.StudyError('The task volumes have no voxels at this voxel size.')
    n_eval = len(grid.eval) * max(1, len({d['follower_uid'] for d in usable}))
    if n_eval > MAX_POINTS:
        raise jobs.StudyError(f'At {st["voxel_mm"]:g} mm the teleop study would solve IK at {n_eval:,} points, over '
                              f'its limit of {MAX_POINTS:,}: use a larger voxel size or fewer search layers.')
    followers = list(dict.fromkeys(d['follower_uid'] for d in usable)) or [project.primary_robot.uid]
    arms = {u: Arm(project, u, st) for u in followers}
    nsteps = len(followers) + len(usable) + 1
    step = [0]

    def prog(text):
        def f(x):
            ctx.report((step[0] + x) / nsteps, text)
        return f
    fevs = {}
    for u in followers:
        fevs[u] = evaluate_follower(arms[u], grid, ctx.cancelled, prog(f'{arms[u].robot.name}: reach and dexterity'))
        step[0] += 1
    evs = []
    for d in usable:
        if ctx.cancelled():
            break
        evs.append(evaluate_device(project, d, arms[d['follower_uid']], fevs[d['follower_uid']], grid, st, notes,
                                   ctx.cancelled, prog(f'{d["name"]}: the leader\'s comfortable workspace')))
        step[0] += 1
    usable = usable[:len(evs)]
    t0 = grid.task_eval
    ntot = len(t0)
    h3 = grid.h ** 3
    # ------------------------------------------------------------ the followers
    robots = []
    for u in followers:
        fe, a = fevs[u], arms[u]
        r = fe['reach'][t0]
        dx = {k: v[t0] for k, v in fe['dex'].items()}
        sdt = fe['sd'][t0]
        rr = np.flatnonzero(r)
        robots.append({
            'uid': u, 'name': a.robot.name, 'primary': a.robot.primary, 'link': a.link,
            'tcp': [_r(x, 5) for x in a.tcp], 'tcp_rule': a.tcp_rule,
            'chain': [a.model.joint_names[k] for k in np.flatnonzero(a.chain)],
            'reachable_pct': _r(100.0 * r.sum() / ntot, 2),
            'w_median': _r(np.median(dx['w'][rr]), 6) if len(rr) else None,
            'w_min': _r(dx['w'][rr].min(), 6) if len(rr) else None,
            'w_unit': ({1: 'm', 2: 'm²'}.get(int(a.chain.sum()), 'm³') + ' (per radian of each joint)'
                       if st['jacobian'] == 'position' else 'mixed (metres and radians)'),
            'inverse_condition_median': _r(np.median(dx['inverse_condition'][rr]), 4) if len(rr) else None,
            'inverse_condition_min': _r(dx['inverse_condition'][rr].min(), 4) if len(rr) else None,
            'near_singularity_pct': _r(100.0 * (r & (dx['inverse_condition'] < st['min_inverse_condition'])).sum()
                                       / ntot, 2),
            'near_limit_pct': _r(100.0 * (r & (dx['limit_distance'] < st['min_limit_distance'])).sum() / ntot, 2),
            'self_close_pct': _r(100.0 * (r & (sdt < st['self_margin_mm'] / 1000)).sum() / ntot, 2),
            'unreachable_reasons': {str(k): int(v) for k, v in zip(*np.unique(
                [fe['reason'][int(i)] for i in t0[~r]] or ['none'], return_counts=True)) if k != 'none'},
            'state': [int(x) for x in np.where(r, 0, 1)]})
    # ------------------------------------------------------------ the devices
    rows, states = [], []
    for d, ev in zip(usable, evs):
        arm = arms[d['follower_uid']]
        comf, cause = ev['comf'][t0], ev['cause'][t0]
        s = _state(comf, cause, ev['stated'])
        states.append(s)
        zs = zones(arm, ev, grid, st, s) if ev['stated'] else []
        dead = cause.any(axis=1)
        good = comf & ~dead
        row = {'uid': d['uid'], 'name': d['name'], 'kind': d['kind'], 'kind_title': KIND_TITLES[d['kind']],
               'follower_uid': d['follower_uid'], 'follower': arm.robot.name, 'mapping': d['mapping'],
               'leader': ev.get('leader'), 'volume': d['volume'], 'source': d['source'], 'stated': ev['stated'],
               'library_id': d.get('library_id'), 'how': ev['how'], 'comfort_box': ev.get('comfort_box'),
               'comfortable_pct': _r(100.0 * comf.sum() / ntot, 2) if ev['stated'] else None,
               'teleoperable_pct': _r(100.0 * good.sum() / ntot, 2) if ev['stated'] else None,
               'dead_zone_pct': _r(100.0 * dead.sum() / ntot, 2) if ev['stated'] else None,
               'dead_zone_m3': _r(dead.sum() * h3, 7) if ev['stated'] else None,
               'causes_pct': {c: _r(100.0 * (s == 2 + k).sum() / ntot, 2) for k, c in enumerate(CAUSES)},
               'zones': zs[:50], 'zones_total': len(zs), 'state': [int(x) for x in s]}
        if ev['stated'] and d['mapping']['kind'] == 'cartesian':
            M, sc = device_transform(d, arm.base)
            row['transform'] = {'matrix': [[_r(x, 6) for x in r_] for r_ in M], 'scale': sc}
        row['status'] = _status(row['teleoperable_pct'])
        rows.append(row)
    ctx.report((nsteps - 1) / nsteps, 'bimanual and the suggestion')
    union = np.any([s == TELEOPERABLE for s in states], axis=0) if states else np.zeros(ntot, bool)
    stated = [r for r in rows if r['stated']]
    pct = _r(100.0 * union.sum() / ntot, 2) if stated else None
    bim = bimanual(project, arms, fevs, usable, evs, grid, st, ctx.cancelled) if not ctx.cancelled() else None
    sug = suggest(project, usable, evs, arms, grid, st) if stated and not ctx.cancelled() else None
    # ------------------------------------------------------------ the headline
    sev = _status(pct)
    if not devices:
        advice = ('There is no teleop device yet: add one on the Teleop overlap page (a leader arm leads with the '
                  'follower\'s own model). The follower\'s reach and dexterity over the task volume are shown.')
        sev = 'warning'
    elif not stated:
        advice = 'No device has a stated workspace: ' + (notes[0] if notes else 'complete the devices below.')
        sev = 'warning'
    else:
        worst = max((z for r in rows for z in r['zones']), key=lambda z: z['voxels'], default=None)
        advice = (f'{pct:.1f} % of the task volume can be teleoperated without a dead zone. ')
        if worst:
            advice += (f'Largest dead zone: {worst["cause"]}, {worst["volume_m3"] * 1e6:.0f} cm³ in {worst["volume"]}'
                       f': {worst["text"]} ')
        outside = 100.0 - (stated[0]['comfortable_pct'] or 0.0) if len(stated) == 1 else 0.0
        if outside > 0.5:
            advice += f'{outside:.1f} % is outside the leader\'s comfortable workspace. '
        if sug:
            advice += sug['text'] + (' Apply makes the move (one undo step).' if not sug['blocked'] else '')
    ctx.report(1.0, '')
    return {
        'summary': {'teleoperable_pct': pct, 'status': sev, 'status_word': WORDS[sev] if devices else 'no device',
                    'advice': advice.strip(), 'devices': len(devices), 'stated_devices': len(stated),
                    'task_voxels': int(ntot), 'task_m3': _r(ntot * h3, 6),
                    'eval_points': int(len(grid.eval)), 'voxel_m': _r(grid.h, 5)},
        'devices': rows,
        'robots': robots,
        'grid': {'voxel_m': _r(grid.h, 5), 'centers': [[_r(x, 4) for x in c] for c in grid.task],
                 'volume_index': [int(x) for x in grid.task_volume],
                 'volumes': [{'uid': v.uid, 'name': v.name, 'kind': v.kind} for v in grid.volumes],
                 'states': STATE_WORDS, 'search_voxels': grid.g},
        'bimanual': bim,
        'suggestion': sug,
        'settings': {k: st[k] for k in DEFAULTS},
        'defaults': st['defaults'],
        'thresholds_note': ('The dead-zone thresholds, the comfortable fraction and the voxel size are DexVantage '
                            'defaults (your settings), not figures of any robot or device.'),
        'sources': list(SOURCES.values()),
        'notes': list(dict.fromkeys(notes)),
        'frame': 'world: the primary robot\'s base, REP-103, m',
    }
