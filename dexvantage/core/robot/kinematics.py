"""An independent forward-kinematics implementation: product of exponentials, without MuJoCo.

MuJoCo is DexVantage's primary kinematics engine (core/robot/model.py). This module computes the same link poses by a
different route so the validation suite can compare the two (spec §3, §10): it reads the robot's stored URDF or MJCF
with xml.etree (not MuJoCo's parser), builds the kinematic tree itself and poses the links by the space-form product
of exponentials (Lynch and Park, Modern Robotics, 2017, chapter 4, written out again here; no rotation code is shared
with core/frames.py or MuJoCo):

    T_link(q) = e^[S_1]θ_1 · e^[S_2]θ_2 · … · e^[S_k]θ_k · M_link

where S_i = (ω_i, v_i) are the screw axes of the joints on the path from the root to the link, expressed in the world
frame at the zero configuration, θ_i the joint displacements and M_link the link's world pose at the zero
configuration. A revolute joint through point p with unit axis ω has S = (ω, −ω × p); a prismatic joint along unit
axis u has S = (0, u). The exponential of a twist (Rodrigues' formula for the rotation; Modern Robotics chapter 3 for
the translation):

    e^[ω]θ = I + sin θ [ω] + (1 − cos θ) [ω]²
    e^[S]θ = [[e^[ω]θ, (I θ + (1 − cos θ)[ω] + (θ − sin θ)[ω]²) v], [0, 1]]       (|ω| = 1)
    e^[S]θ = [[I, v θ], [0, 1]]                                                  (ω = 0)

The space Jacobian (Modern Robotics chapter 5) is

    J_s = [S_1, Ad(e^[S_1]θ_1) S_2, …, Ad(e^[S_1]θ_1 ⋯ e^[S_{k−1}]θ_{k−1}) S_k]

with Ad(R, p) = [[R, 0], [[p]R, R]] acting on (ω, v); the velocity of a point x fixed in the link is v_s + ω_s × x.

PoEChain(robot, sources) gives:

    .joint_names, .nq             the one-degree-of-freedom joints (URDF revolute, continuous, prismatic; MJCF hinge,
                                  slide) in tree order (MJCF: MuJoCo's order); q is in radians or metres
    .joint_types                  'revolute' | 'prismatic' per joint
    .link_names                   every link (URDF link; MJCF body, unnamed ones "body_<MuJoCo index>")
    .fk(q) / .fk_array(q)         link name -> 4×4 world transform (Robot.base applied), or (n_links, 4, 4)
    .jacobian(q, link, point)     (6, nq) world-frame [linear; angular] Jacobian of a point fixed in the link
    .space_jacobian(q, link)      (6, nq) space-frame twists [ω; v] of the joints (zero for joints not on the path)
    .screws                       (nq, 6) [ω; v] in the world frame at the zero configuration

q is a vector in joint_names order, or a {joint name: value} dict (a joint not named sits at its zero: 0 for URDF,
the joint's ref for MJCF).

What is read. URDF: <link>, <joint> (type, <origin xyz rpy> with rpy about fixed x, y, z axes, <axis>, default
(1, 0, 0), normalised as MuJoCo does). MJCF: <include> (from the stored files, through the import manifest), <compiler
angle eulerseq>, <default> classes with childclass and class inheritance (joint type, pos, axis, ref), <body pos
quat|axisangle|euler|xyaxes|zaxis>, <frame>, <joint type pos axis ref>, <freejoint>, the "home" keyframe.

MJCF semantics written out again: quat is (w, x, y, z); angles are degrees unless <compiler angle="radian">; euler
uses eulerseq (default "xyz"), a lowercase letter rotating about the moving axis (post-multiply), an uppercase one
about the fixed axis (pre-multiply); xyaxes orthogonalises y against x and takes z = x × y; zaxis is the shortest
rotation taking (0, 0, 1) to the vector; a hinge turns by q − ref about its axis through its pos, a slide moves by
q − ref along its axis; several joints in one body apply in file order, each in the frame the previous ones left.

Known limits: free and ball joints are held at the "home" keyframe (or their zero) and are not part of q, as in
RobotModel; URDF mimic joints move independently; URDF planar joints, MJCF <replicate>, <attach> and <composite> are
not read (a plain error says so); MJCF <default> attributes other than a joint's type, pos, axis and ref do not change
kinematics and are ignored.
"""
from __future__ import annotations

import json
import math
import posixpath
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from .importer import MANIFEST
from .model import RobotModelError

_EPS = 1e-12


class KinematicsError(RobotModelError):
    """The stored files cannot be read by the independent kinematics; the message says why in plain words."""


# ====================================================================== rotations and twists (written out again)
def hat(w) -> np.ndarray:
    """The skew-symmetric matrix [w] with [w] x = w × x."""
    x, y, z = np.asarray(w, float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def exp_rotation(w, theta: float) -> np.ndarray:
    """Rodrigues: e^[w]θ for a unit axis w."""
    W = hat(w)
    return np.eye(3) + math.sin(theta) * W + (1.0 - math.cos(theta)) * (W @ W)


def exp_twist(S, theta: float) -> np.ndarray:
    """e^[S]θ, the 4×4 rigid motion of the screw S = (ω, v) (|ω| = 1, or ω = 0 and |v| = 1) moved by θ."""
    S = np.asarray(S, float).reshape(6)
    w, v = S[:3], S[3:]
    T = np.eye(4)
    if np.linalg.norm(w) < _EPS:
        T[:3, 3] = v * theta
        return T
    W = hat(w)
    W2 = W @ W
    s, c = math.sin(theta), math.cos(theta)
    T[:3, :3] = np.eye(3) + s * W + (1.0 - c) * W2
    T[:3, 3] = (np.eye(3) * theta + (1.0 - c) * W + (theta - s) * W2) @ v
    return T


def adjoint(T) -> np.ndarray:
    """Ad_T acting on twists ordered (ω, v): [[R, 0], [[p]R, R]]."""
    T = np.asarray(T, float)
    R, p = T[:3, :3], T[:3, 3]
    A = np.zeros((6, 6))
    A[:3, :3] = R
    A[3:, 3:] = R
    A[3:, :3] = hat(p) @ R
    return A


def _rot_axis(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, float)
    n = float(np.linalg.norm(a))
    if n < _EPS:
        raise ValueError('zero axis')
    return exp_rotation(a / n, angle)


_E = {'x': np.array([1.0, 0.0, 0.0]), 'y': np.array([0.0, 1.0, 0.0]), 'z': np.array([0.0, 0.0, 1.0])}


def rot_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF rpy: roll about fixed x, then pitch about fixed y, then yaw about fixed z: Rz(yaw) Ry(pitch) Rx(roll)."""
    return exp_rotation(_E['z'], yaw) @ exp_rotation(_E['y'], pitch) @ exp_rotation(_E['x'], roll)


def rot_quat_wxyz(q) -> np.ndarray:
    """Rotation matrix of a quaternion written (w, x, y, z), as MJCF writes it (normalised first)."""
    q = np.asarray(q, float).reshape(4)
    n = float(np.linalg.norm(q))
    if n < _EPS:
        raise ValueError('zero quaternion')
    w, x, y, z = q / n
    return np.array([[w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z]])


def rot_euler(angles, seq: str) -> np.ndarray:
    """MJCF euler: for each letter of seq, a rotation about that axis; lowercase letters rotate about the moving
    axes (post-multiplied), uppercase about the fixed axes (pre-multiplied)."""
    R = np.eye(3)
    for a, ch in zip(angles, seq):
        Ri = exp_rotation(_E[ch.lower()], float(a))
        R = R @ Ri if ch.islower() else Ri @ R
    return R


def rot_xyaxes(v) -> np.ndarray:
    v = np.asarray(v, float).reshape(6)
    x = v[:3] / np.linalg.norm(v[:3])
    y = v[3:] - x * float(x @ v[3:])
    y = y / np.linalg.norm(y)
    return np.c_[x, y, np.cross(x, y)]


def rot_zaxis(z) -> np.ndarray:
    """The shortest rotation taking (0, 0, 1) to z (a half turn about x when z points down)."""
    z = np.asarray(z, float).reshape(3)
    z = z / np.linalg.norm(z)
    ax = np.cross(_E['z'], z)
    s, c = float(np.linalg.norm(ax)), float(z[2])
    if s < _EPS:
        return np.eye(3) if c > 0 else exp_rotation(_E['x'], math.pi)
    return exp_rotation(ax / s, math.atan2(s, c))


def _T(R=None, p=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


# ====================================================================== the parsed tree
@dataclass
class _Joint:
    name: str
    kind: str                 # 'hinge' | 'slide' | 'ball' | 'free' | 'fixed'
    pos: np.ndarray           # anchor in the body frame (after any <frame>)
    axis: np.ndarray          # unit axis in the body frame
    ref: float = 0.0          # the zero of q (MJCF ref); the displacement is q − ref
    held: np.ndarray | None = None   # ball: (w, x, y, z); free: (x, y, z, w, x, y, z) as MJCF writes qpos


@dataclass
class _Body:
    name: str
    parent: int               # index into the body list, −1 = the world
    T: np.ndarray             # pose in the parent body's frame at the zero configuration
    joints: list = field(default_factory=list)


def _floats(text, n, what):
    try:
        v = [float(x) for x in str(text).split()]
    except ValueError:
        raise KinematicsError(f'{what} is not a list of numbers ("{text}").') from None
    if len(v) != n:
        raise KinematicsError(f'{what} should have {n} numbers; it has {len(v)} ("{text}").')
    return np.array(v)


# ---------------------------------------------------------------------- URDF
def _parse_urdf(root, name) -> list:
    if root.tag != 'robot':
        raise KinematicsError(f'The stored URDF of {name} does not start with <robot>.')
    if any(str(el.tag).startswith('{') and 'xacro' in str(el.tag) for el in root.iter()):
        raise KinematicsError(f'The stored URDF of {name} still has xacro macros: expand it to plain URDF first.')
    links = [ln.get('name') for ln in root.findall('link') if ln.get('name')]
    joints = root.findall('joint')
    by_child, children = {}, {n: [] for n in links}
    for j in joints:
        pe, ce = j.find('parent'), j.find('child')
        if pe is None or ce is None:
            raise KinematicsError(f'Joint {j.get("name")} of {name} has no parent or child link.')
        p, c = pe.get('link'), ce.get('link')
        if p not in children or c not in children:
            raise KinematicsError(f'Joint {j.get("name")} of {name} names a link that is not in the file.')
        if c in by_child:
            raise KinematicsError(f'Link {c} of {name} has two parent joints; a URDF must be a tree.')
        by_child[c] = j
        children[p].append(c)
    roots = [n for n in links if n not in by_child]
    if not roots and links:
        raise KinematicsError(f'The links of {name} form a loop; a URDF must be a tree.')
    bodies = []

    def add(link, parent):
        j = by_child.get(link)
        T = np.eye(4)
        jl = []
        if j is not None:
            o = j.find('origin')
            xyz = _floats(o.get('xyz', '0 0 0'), 3, f'The origin of joint {j.get("name")}') if o is not None \
                else np.zeros(3)
            rpy = _floats(o.get('rpy', '0 0 0'), 3, f'The rpy of joint {j.get("name")}') if o is not None \
                else np.zeros(3)
            T = _T(rot_rpy(*rpy), xyz)
            t = j.get('type', '')
            ax_el = j.find('axis')
            ax = _floats(ax_el.get('xyz', '1 0 0'), 3, f'The axis of joint {j.get("name")}') if ax_el is not None \
                else np.array([1.0, 0.0, 0.0])
            kind = {'revolute': 'hinge', 'continuous': 'hinge', 'prismatic': 'slide', 'fixed': 'fixed',
                    'floating': 'free'}.get(t)
            if kind is None:
                raise KinematicsError(f'Joint {j.get("name")} of {name} is a {t or "untyped"} joint, which the '
                                      'independent kinematics does not read.')
            if kind in ('hinge', 'slide'):
                n = float(np.linalg.norm(ax))
                if n < _EPS:
                    raise KinematicsError(f'Joint {j.get("name")} of {name} has a zero axis.')
                ax = ax / n
            jl.append(_Joint(str(j.get('name') or ''), kind, np.zeros(3), ax))
        k = len(bodies)
        bodies.append(_Body(link, parent, T, jl))
        for c in children[link]:
            add(c, k)

    for r in roots:
        add(r, -1)
    return bodies


# ---------------------------------------------------------------------- MJCF
def _expand_includes(el, load, depth=0):
    """Replace every <include file> under el with the children of the included file's root, in place."""
    if depth > 16:
        raise KinematicsError('The MJCF includes nest more than 16 deep (a file probably includes itself).')
    for i, ch in enumerate(list(el)):
        if ch.tag == 'include':
            fn = ch.get('file') or ''
            sub = load(fn)
            _expand_includes(sub, load, depth + 1)
            pos = list(el).index(ch)
            el.remove(ch)
            for k, s in enumerate(list(sub)):
                el.insert(pos + k, s)
        else:
            _expand_includes(ch, load, depth)


class _Mjcf:
    JOINT_KEYS = ('type', 'pos', 'axis', 'ref')

    def __init__(self, root, name):
        if root.tag != 'mujoco':
            raise KinematicsError(f'The stored MJCF of {name} does not start with <mujoco>.')
        self.name = name
        self.radian = False
        self.seq = 'xyz'
        for c in root.iter('compiler'):
            if c.get('angle'):
                self.radian = c.get('angle').strip().lower() == 'radian'
            if c.get('eulerseq'):
                self.seq = c.get('eulerseq').strip()
            if c.get('coordinate', 'local').strip().lower() != 'local':
                raise KinematicsError(f'{name} uses global coordinates, which MuJoCo 3 no longer reads.')
        if len(self.seq) != 3 or any(ch.lower() not in 'xyz' for ch in self.seq):
            raise KinematicsError(f'The eulerseq "{self.seq}" of {name} is not three of x, y, z.')
        self.scale = 1.0 if self.radian else math.pi / 180.0
        self.cls = {'main': (None, {})}               # class -> (parent class, {attr: value} of <joint>)
        for d in root.findall('default'):
            self._defaults(d, None)
        for bad in ('replicate', 'attach', 'composite', 'flexcomp'):
            if any(True for wb in root.iter('worldbody') for _ in wb.iter(bad)):
                raise KinematicsError(f'{name} uses <{bad}>, which the independent kinematics does not read.')
        self.bodies, self.joints_all = [], []
        for wb in root.findall('worldbody'):
            self._children(wb, -1, np.eye(4), 'main')
        self._keyframe(root)

    def _defaults(self, d, parent):
        c = d.get('class') or ('main' if parent is None else None)
        if c is None:
            raise KinematicsError(f'A nested <default> in {self.name} has no class name.')
        own = dict(self.cls.get(c, (parent, {}))[1]) if c == 'main' else {}
        j = d.find('joint')
        if j is not None:
            own.update({k: j.get(k) for k in self.JOINT_KEYS if j.get(k) is not None})
        self.cls[c] = (parent if c != 'main' else None, own)
        for sub in d.findall('default'):
            self._defaults(sub, c)

    def _joint_attr(self, el, cls, key):
        if el.get(key) is not None:
            return el.get(key)
        c = el.get('class') or cls
        seen = 0
        while c is not None and seen < 64:
            if c not in self.cls:
                raise KinematicsError(f'{self.name} uses the default class "{c}", which it does not define.')
            par, own = self.cls[c]
            if key in own:
                return own[key]
            c = par if par is not None else (None if c == 'main' else 'main')
            seen += 1
        return None

    def _orient(self, el, what) -> np.ndarray:
        if el.get('quat') is not None:
            return rot_quat_wxyz(_floats(el.get('quat'), 4, f'The quat of {what}'))
        if el.get('axisangle') is not None:
            v = _floats(el.get('axisangle'), 4, f'The axisangle of {what}')
            return _rot_axis(v[:3], v[3] * self.scale)
        if el.get('euler') is not None:
            return rot_euler(_floats(el.get('euler'), 3, f'The euler of {what}') * self.scale, self.seq)
        if el.get('xyaxes') is not None:
            return rot_xyaxes(_floats(el.get('xyaxes'), 6, f'The xyaxes of {what}'))
        if el.get('zaxis') is not None:
            return rot_zaxis(_floats(el.get('zaxis'), 3, f'The zaxis of {what}'))
        return np.eye(3)

    def _frame_T(self, el, what) -> np.ndarray:
        p = _floats(el.get('pos'), 3, f'The pos of {what}') if el.get('pos') is not None else np.zeros(3)
        return _T(self._orient(el, what), p)

    def _children(self, el, body, F, cls, jl=None):
        """Walk el's children: bodies (posed by the frame F within body), joints (into jl) and nested frames."""
        for ch in el:
            if ch.tag == 'body':
                k = len(self.bodies) + 1                 # MuJoCo's body index (the world is 0)
                nm = ch.get('name') or f'body_{k}'
                b = _Body(nm, body, F @ self._frame_T(ch, f'body {nm}'))
                self.bodies.append(b)
                self._children(ch, len(self.bodies) - 1, np.eye(4), ch.get('childclass') or cls, b.joints)
            elif ch.tag == 'frame':
                self._children(ch, body, F @ self._frame_T(ch, 'a frame'), ch.get('childclass') or cls, jl)
            elif ch.tag in ('joint', 'freejoint'):
                if jl is None:
                    raise KinematicsError(f'{self.name} has a joint directly in the world body, which MuJoCo does '
                                          'not allow.')
                self._joint(ch, F, cls, jl)

    def _joint(self, el, F, cls, jl):
        k = len(self.joints_all)
        nm = el.get('name') or f'joint_{k}'
        if el.tag == 'freejoint':
            kind = 'free'
        else:
            kind = (self._joint_attr(el, cls, 'type') or 'hinge').strip().lower()
        if kind not in ('hinge', 'slide', 'ball', 'free'):
            raise KinematicsError(f'Joint {nm} of {self.name} has the unknown type "{kind}".')
        pos = self._joint_attr(el, cls, 'pos') if kind != 'free' else None
        axis = self._joint_attr(el, cls, 'axis') if kind in ('hinge', 'slide') else None
        ref = self._joint_attr(el, cls, 'ref') if kind in ('hinge', 'slide') else None
        p = _floats(pos, 3, f'The pos of joint {nm}') if pos is not None else np.zeros(3)
        a = _floats(axis, 3, f'The axis of joint {nm}') if axis is not None else np.array([0.0, 0.0, 1.0])
        if kind in ('hinge', 'slide'):
            n = float(np.linalg.norm(a))
            if n < _EPS:
                raise KinematicsError(f'Joint {nm} of {self.name} has a zero axis.')
            a = a / n
        r = float(_floats(ref, 1, f'The ref of joint {nm}')[0]) if ref is not None else 0.0
        if kind == 'hinge':
            r *= self.scale
        j = _Joint(nm, kind, F[:3, :3] @ p + F[:3, 3], F[:3, :3] @ a, r)
        jl.append(j)
        self.joints_all.append(j)

    def _keyframe(self, root):
        sizes = {'free': 7, 'ball': 4, 'hinge': 1, 'slide': 1}
        nq = sum(sizes[j.kind] for j in self.joints_all)
        self.notes = []
        for key in root.iter('key'):
            if key.get('name') != 'home' or key.get('qpos') is None:
                continue
            try:
                qp = np.array([float(x) for x in key.get('qpos').split()])
            except ValueError:
                self.notes.append('The "home" keyframe\'s qpos is not a list of numbers; free and ball joints are '
                                  'held at their zero instead.')
                return
            if len(qp) != nq:
                self.notes.append(f'The "home" keyframe has {len(qp)} qpos values for {nq}; free and ball joints '
                                  'are held at their zero instead.')
                return
            a = 0
            for j in self.joints_all:
                n = sizes[j.kind]
                if j.kind in ('ball', 'free'):
                    j.held = qp[a:a + n].copy()
                a += n
            return


# ====================================================================== the chain
class PoEChain:
    """Forward kinematics and Jacobians by the product of exponentials (see the module docstring)."""

    def __init__(self, robot, sources: dict):
        self.name = robot.name
        self.kind = robot.source.kind or 'urdf'
        self.notes: list = []
        key = robot.source.file
        if key not in sources:
            raise KinematicsError(f'The model file of {robot.name} is not stored in the project: import the robot '
                                  'again.')
        if self.kind not in ('urdf', 'mjcf'):
            raise KinematicsError(f'{robot.name} comes from a {self.kind} file, which the independent kinematics '
                                  'does not read.')
        root = self._xml(sources[key], key, robot.name)
        if self.kind == 'urdf':
            bodies = _parse_urdf(root, robot.name)
        else:
            _expand_includes(root, self._loader(robot, sources))
            mj = _Mjcf(root, robot.name)
            bodies = mj.bodies
            self.notes += mj.notes
        self._build(bodies, self._base(robot))

    # ---------------------------------------------------------------- reading
    @staticmethod
    def _xml(raw: bytes, key: str, name: str):
        try:
            return ET.fromstring(raw)
        except ET.ParseError as e:
            raise KinematicsError(f'The stored file {posixpath.basename(key)} of {name} is not valid XML ({e}).') \
                from None

    def _loader(self, robot, sources):
        pre = robot.uid + '/'
        refs = {}
        raw = sources.get(pre + MANIFEST)
        if raw is not None:
            try:
                man = json.loads(raw.decode('utf-8'))
                refs = man.get('refs', {}) if isinstance(man.get('refs'), dict) else {}
            except (UnicodeDecodeError, ValueError, AttributeError):
                refs = {}
        here = posixpath.dirname(robot.source.file)

        def load(fn):
            rel = refs.get('include:' + fn)
            cands = ([pre + rel] if rel else []) + [posixpath.normpath(posixpath.join(here, fn))]
            for c in cands:
                if c in sources:
                    return self._xml(sources[c], c, robot.name)
            raise KinematicsError(f'The MJCF of {robot.name} includes {posixpath.basename(fn)}, which is not stored '
                                  'in the project: import the robot again from its complete folder.')
        return load

    @staticmethod
    def _base(robot) -> np.ndarray:
        b = getattr(robot, 'base', None)
        if b is None:
            return np.eye(4)
        x, y, z, w = np.asarray(b.quaternion, float)
        return _T(rot_quat_wxyz([w, x, y, z]), np.asarray(b.position, float))

    # ---------------------------------------------------------------- the zero configuration
    def _build(self, bodies, base):
        """World poses M of every body at the zero configuration and the world screw of every moving joint."""
        n = len(bodies)
        self.link_names = [b.name for b in bodies]
        self._parent = np.array([b.parent for b in bodies], int)
        self._M = np.zeros((n, 4, 4))
        self._body_joints = [[] for _ in range(n)]         # per body: indices into the moving joints
        screws, names, types, refs, owner = [], [], [], [], []
        for k, b in enumerate(bodies):
            Tw = (base if b.parent < 0 else self._M[b.parent]) @ b.T
            for j in b.joints:
                if j.kind in ('hinge', 'slide'):
                    w = Tw[:3, :3] @ j.axis
                    if j.kind == 'hinge':
                        p = Tw[:3, :3] @ j.pos + Tw[:3, 3]
                        screws.append(np.r_[w, -np.cross(w, p)])
                    else:
                        screws.append(np.r_[0.0, 0.0, 0.0, w])
                    self._body_joints[k].append(len(names))
                    names.append(j.name)
                    types.append('revolute' if j.kind == 'hinge' else 'prismatic')
                    refs.append(j.ref)
                    owner.append(k)
                elif j.kind == 'ball' and j.held is not None:
                    A = _T(None, j.pos)
                    Tw = Tw @ A @ _T(rot_quat_wxyz(j.held)) @ np.linalg.inv(A)
                elif j.kind == 'free' and j.held is not None:
                    Tw = base @ _T(rot_quat_wxyz(j.held[3:7]), j.held[:3])
            self._M[k] = Tw
        self.joint_names = names
        self.joint_types = types
        self.nq = len(names)
        self.ref = np.array(refs, float)
        self.screws = np.array(screws, float).reshape(-1, 6)
        self._owner = np.array(owner, int)
        self._link_index = {nm: i for i, nm in enumerate(self.link_names)}
        # the moving joints on the path from the root to each body, in order
        self._path = []
        for k in range(n):
            up = self._path[self._parent[k]] if self._parent[k] >= 0 else []
            self._path.append(up + self._body_joints[k])

    # ---------------------------------------------------------------- kinematics
    def _q(self, q) -> np.ndarray:
        if isinstance(q, dict):
            out = self.ref.copy()
            for i, nm in enumerate(self.joint_names):
                if nm in q:
                    out[i] = float(q[nm])
            return out
        q = np.asarray(q, float).reshape(-1)
        if len(q) != self.nq:
            raise ValueError(f'{self.name} has {self.nq} moving joints; got {len(q)} joint values.')
        return q

    def _exps(self, q):
        """Per body: the product of exponentials along its path, E_k; and per joint the product before it."""
        th = self._q(q) - self.ref
        n = len(self.link_names)
        E = np.zeros((n, 4, 4))
        before = np.zeros((self.nq, 4, 4))
        for k in range(n):
            P = E[self._parent[k]].copy() if self._parent[k] >= 0 else np.eye(4)
            for i in self._body_joints[k]:
                before[i] = P
                P = P @ exp_twist(self.screws[i], th[i])
            E[k] = P
        return E, before

    def fk_array(self, q) -> np.ndarray:
        """(n_links, 4, 4) world transforms in link_names order."""
        E, _ = self._exps(q)
        return E @ self._M

    def fk(self, q) -> dict:
        """link name -> 4×4 world transform (Robot.base applied)."""
        T = self.fk_array(q)
        return {nm: T[i] for i, nm in enumerate(self.link_names)}

    def link_index(self, link: str) -> int:
        i = self._link_index.get(link)
        if i is None:
            raise KeyError(f'{self.name} has no link named {link}.')
        return i

    def space_jacobian(self, q, link: str) -> np.ndarray:
        """(6, nq) space-frame twists [ω; v] of the joints at q; zero columns for joints not on the link's path."""
        k = self.link_index(link)
        _E, before = self._exps(q)
        J = np.zeros((6, self.nq))
        for i in self._path[k]:
            J[:, i] = adjoint(before[i]) @ self.screws[i]
        return J

    def jacobian(self, q, link: str, point=None) -> np.ndarray:
        """(6, nq) world-frame [linear; angular] Jacobian of a point fixed in the link (link coordinates, default its
        origin), the same convention as RobotModel.jacobian."""
        k = self.link_index(link)
        E, before = self._exps(q)
        T = E[k] @ self._M[k]
        x = T[:3, 3] if point is None else T[:3, :3] @ np.asarray(point, float).reshape(3) + T[:3, 3]
        J = np.zeros((6, self.nq))
        for i in self._path[k]:
            tw = adjoint(before[i]) @ self.screws[i]
            J[:3, i] = tw[3:] + np.cross(tw[:3], x)
            J[3:, i] = tw[:3]
        return J
