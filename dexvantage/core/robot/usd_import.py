"""USD robot import (spec §13): a UsdPhysics articulation converted to MJCF, which DexVantage then imports like any
MuJoCo file (core/robot/importer.py), so the robot is stored, drawn and analysed from the converted files.

    have_pxr() -> bool                                  usd-core (the 'usd' extra) is installed
    convert(path) -> Converted                          the MJCF text, its STL meshes, notes and a record
    convert_to_folder(path, folder) -> Converted        the same, written as robot.xml + meshes/*.stl
    check_references(path) -> {'kind': 'usd', 'items'}  the layers and assets the stage names, found or missing
                                                        (the import wizard's mesh step)

What is read (pxr, usd-core 26.8; the OpenUSD UsdPhysics schemas, https://openusd.org/release/api/usd_physics_page_front
.html, read 2026-11-07). The stage is opened with every payload loaded and walked with instance proxies (Isaac's
instanceable meshes). Bodies are the prims with PhysicsRigidBodyAPI under the stage's PhysicsArticulationRootAPI prim
(the first one when there are several, said in the notes; every rigid body when there is none); joints are the
UsdPhysicsJoint prims that link two of those bodies, or one of them and the world. A joint's body0 or body1 that is
not a rigid body (empty, or an Xform such as the asset's root) is the world frame of that prim.

Frames. UsdPhysics states a joint by its frame in each body: localPos0 / localRot0 in body0's local space, localPos1 /
localRot1 in body1's, with the constraint X_body0 · F0 · M(q) = X_body1 · F1, M(q) the joint's motion along or about
its axis token (X, Y or Z of the joint frame; revolute angles in degrees, prismatic distances in stage units). Each
body's world transform is pxr's (UsdGeom.XformCache, the composed xformOps), scaled to metres (metersPerUnit) and
turned to Z up when the stage is Y up (Rx(+90°)); a body frame is its rotation and translation (scale taken out by a
polar decomposition, kept for the points, localPos and primitives it scales). The MJCF body of a child c of parent p
is placed at B = J0 · J1⁻¹, J0 = X_p⁻¹ · F0 and J1 = X_c⁻¹ · F1 (the joint frames in each body), so the joints rule
the kinematics, not the authored poses (a disagreement between the two at the authored state is stated); a hinge or
slide joint sits at J1's origin with axis R(J1) · e_axis, which reproduces X_c(q) = X_p · J0 · M(q) · J1⁻¹ exactly
(MuJoCo's mj_kinematics: X_p · B · Trans(pos) · Rot(axis, q) · Trans(−pos)). A joint whose body0 is the child in the
tree (written the other way round) is turned: frames swapped and the axis negated, the same motion for the same q.
Revolute limits are converted from degrees to radians, prismatic limits to metres; a limit that is not finite (or
beyond ±10⁶ degrees or stage units, the float sentinels some writers use) leaves that joint unlimited.

The tree is a breadth-first walk from the body jointed to the world (the articulation root when none is): each
other body once, by the first joint that reaches it. A joint that would close a loop (a four-bar, a parallel
gripper's second link) is left out and named in the notes, as is a disabled joint, one excluded from the
articulation, or a generic (D6) or distance joint (imported as fixed). Spherical joints become ball joints (held at
home, as DexVantage holds every ball joint). A floating base (no joint to the world) is fixed at its authored pose.

Geometry. Every UsdGeom.Mesh, Cube, Sphere, Capsule and Cylinder below a body (and not below another body) is a geom
of that body, in its frame: a prim with PhysicsCollisionAPI is a collision geom (MuJoCo's default contype and
conaffinity), any other visible one a visual geom (contype = conaffinity = 0, group 1: what core/robot/model.py reads
as visual-only). Meshes are triangulated (fans; leftHanded orientation flipped), moved into the body frame with their
full transform (scale included) and written as binary STL in metres (core/meshing.stl_bytes). A Cube is a box of half
size size/2 times its scale per axis; a Sphere a sphere (an ellipsoid when its scale is not uniform); a Capsule or
Cylinder a MuJoCo capsule or cylinder of radius × the scale across its axis and half length height/2 × the scale
along it, turned so its axis token is the geom's z. Cones, planes and other prims are left out (said).

Mass. A body whose PhysicsMassAPI states both a mass and a diagonal inertia gets an MJCF <inertial>: mass ×
kilogramsPerUnit, pos the centerOfMass (body local space, in metres), quat the principalAxes, diaginertia ×
kilogramsPerUnit × metersPerUnit². A body without both gets none (the file then states no mass for it; the compiler
setting inertiafromgeom="false" keeps MuJoCo from inventing one from the geoms' volume), and core/robot/model.py
compiles such a model with its stated boundmass, as for a kinematics-only URDF; the notes list those bodies.

Validation (validation/checks_robot_io.py): the SO-100, UR5e and Panda samples written to USD by
core/robot/usd_writer.py and imported back match DexVantage's kinematics of every link over 50 random joint states
(10⁻⁶ m, 10⁻⁶ rad) with the joint limits equal to the float32 the schema stores; and a third party's UR5e
(newton-assets, written by the mujoco-usd-converter from the Menagerie MJCF) imported here poses like DexVantage's
UR5e sample.

Known limits: one articulation per file; loops, mimic and tendon couplings, drives, joint friction, damping,
armature, effort and velocity limits are not imported (DexVantage's joint records state None for them); materials
become one display colour per geom; textures, cones, planes, height fields, point instancers and subdivision are not
read; a body's non-uniform scale is applied to its points and primitives but not to its joint frames' rotations or
its inertia (said in the notes when present); a mesh whose referenced layer is missing is left out (the stage
composes without it; check_references lists it); time samples are read at the earliest time.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

USD_EXTS = ('.usd', '.usda', '.usdc', '.usdz')
MAIN = 'robot.xml'
MESH_DIR = 'meshes'
RECORD = '.dexvantage/usd_source.json'
RECORD_FORMAT = 'dexvantage.usd-source/1'
LIMIT_SENTINEL = 1e6          # |limit| at or beyond this (degrees or stage units) is "no limit": float-max sentinels
POSE_TOL = 1e-5               # m and rad: an authored body pose further than this from the joints' is stated
UP_Y = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])    # Rx(+90°): a Y-up stage's +Y to +Z


class UsdImportError(ValueError):
    """A USD file DexVantage cannot convert; the message says why in plain words (importer re-raises it as a
    RobotImportError)."""


def have_pxr() -> bool:
    try:
        import pxr.Usd  # noqa: F401
        import pxr.UsdPhysics  # noqa: F401
    except ImportError:
        return False
    return True


def missing_pxr_message(base: str) -> str:
    return (f'{base} is a USD file, which needs the optional usd-core package (optional: install usd-core, for '
            'example pip install "dexvantage[usd]" or pip install usd-core==26.8), then import it again.')


@dataclass
class Converted:
    name: str
    mjcf: str
    meshes: dict                       # 'meshes/<file>.stl' -> bytes
    notes: list = field(default_factory=list)
    record: dict = field(default_factory=dict)


# ====================================================================== small maths
def _rigid(M) -> tuple[np.ndarray, np.ndarray]:
    """(R, S): the rotation and the stretch of a 3 × 3 linear map, M = R S (polar decomposition by SVD)."""
    U, s, Vt = np.linalg.svd(np.asarray(M, float))
    R = U @ Vt
    if np.linalg.det(R) < 0:                           # a mirror: keep a rotation, the stretch takes the sign
        U[:, -1] *= -1
        R = U @ Vt
    return R, R.T @ np.asarray(M, float)


def _T(R=None, t=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def _inv(T) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return _T(R.T, -R.T @ t)


def _quat_wxyz(R) -> np.ndarray:
    from ..frames import quat_from_matrix, quat_to_mujoco
    return quat_to_mujoco(quat_from_matrix(R))


def _gf_quat_matrix(q) -> np.ndarray:
    """The rotation of a Gf.Quatf / Quatd (real, imaginary), normalised, in float64."""
    if q is None:
        return np.eye(3)
    w = float(q.GetReal())
    x, y, z = (float(v) for v in q.GetImaginary())
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


AXES = {'X': np.array([1.0, 0.0, 0.0]), 'Y': np.array([0.0, 1.0, 0.0]), 'Z': np.array([0.0, 0.0, 1.0])}
# a rotation that takes the geom's z (MuJoCo's capsule and cylinder axis) to the USD axis token
Z_TO = {'Z': np.eye(3), 'X': np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
        'Y': np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])}


def _f(x) -> str:
    x = float(x)
    return '0' if x == 0 else repr(x)


def _vec(v) -> str:
    return ' '.join(_f(x) for x in v)


def _name(text: str) -> str:
    s = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(text or '')).strip('_')
    return s or 'x'


class _Names:
    def __init__(self):
        self.used = set()

    def __call__(self, text: str) -> str:
        base = _name(text)
        n, i = base, 2
        while n in self.used:
            n, i = f'{base}_{i}', i + 1
        self.used.add(n)
        return n


# ====================================================================== reading the stage
def _get(attr, default=None):
    """An attribute's value at the default time, else at its earliest time sample; default when not authored."""
    if attr is None or not attr.IsValid():
        return default
    from pxr import Usd
    v = attr.Get()
    if v is None and attr.GetNumTimeSamples():
        v = attr.Get(Usd.TimeCode.EarliestTime())
    return default if v is None else v


def _pxr_reason(text: str) -> str:
    """The reader's own words from a pxr error ("Error in '<C++ function>' at line N in file <source> : '<words>'"),
    without the C++ names, source files and full paths: 'line 3, column 1: Expected ) ...', or '' when it says only
    that the layer did not open."""
    import re
    msgs = re.findall(r"Error in '[^']*' at line \d+ in file [^:]* : '(.*?)'\s*(?=\n\s*Error in|\Z)", text, re.S)
    for m in msgs or [text]:
        m = ' '.join(m.split())
        if not m or m.startswith('Failed to open layer'):
            continue
        loc = re.match(r'(?:[^:\s]*[/\\])?[^/\\:\s]+:(\d+):(\d+):\s*(.*)', m)
        if loc:
            m = f'line {loc.group(1)}, column {loc.group(2)}: {loc.group(3)}'
        elif msgs:                                       # a message without the file's place: the next one may have it
            continue
        m = re.sub(r'@[^@]*?([^/\\@]+)@', r'\1', m)      # @/full/path/x.usda@ -> x.usda
        m = re.sub(r'(?:[A-Za-z]:)?(?:[/\\][^/\\\s:\'"]+)+[/\\]([^/\\\s:\'"]+)', r'\1', m)   # any other full path
        return m[:160].rstrip()
    return ''


def _open(path: str):
    """(stage, prims, missing): the stage with every payload loaded, its prims (instance proxies included) and the
    basenames of the assets it names that could not be opened. pxr's composition warnings are collected by a
    UsdUtils.CoalescingDiagnosticDelegate instead of printed (they would carry full paths and C++ source lines)."""
    from pxr import Usd, UsdUtils
    base = os.path.basename(path)
    delegate = UsdUtils.CoalescingDiagnosticDelegate()
    try:
        try:
            stage = Usd.Stage.Open(path, Usd.Stage.LoadAll)
        except Exception as e:                         # noqa: BLE001 — explained
            why = _pxr_reason(str(e))
            raise UsdImportError(f'{base} could not be opened as a USD file' + (f' ({why})' if why else
                                 ': it is not a USD layer DexVantage can read') + '.') from None
        if stage is None:
            raise UsdImportError(f'{base} could not be opened as a USD file.')
        prims = list(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
    finally:
        diags = delegate.TakeUncoalescedDiagnostics()
        del delegate
    missing = sorted({os.path.basename(m) for d in diags for m in re.findall(r'Could not open asset @([^@]+)@',
                                                                              str(d.commentary))})
    return stage, prims, missing


class _Stage:
    """The stage's rigid bodies, joints and frames in metres, Z up."""

    def __init__(self, path: str):
        from pxr import Usd, UsdGeom, UsdPhysics
        self.path = path
        self.base = os.path.basename(path)
        self.stage, prims, self.missing = _open(path)
        self.notes = []
        if self.missing:
            self.notes.append(f'{self.base} names {", ".join(self.missing)}, which could not be opened beside it: the '
                              'stage is read without '
                              f'{"it" if len(self.missing) == 1 else "them"} (meshes or materials it holds are left '
                              'out).')
        self.mpu = float(UsdGeom.GetStageMetersPerUnit(self.stage) or 1.0)
        self.kgpu = float(UsdPhysics.GetStageKilogramsPerUnit(self.stage) or 1.0)
        self.up = str(UsdGeom.GetStageUpAxis(self.stage))
        self.C = np.diag([self.mpu] * 3 + [1.0])
        if self.up == 'Y':
            self.C = _T(UP_Y) @ self.C
            self.notes.append(f'{self.base} is Y up: it is turned by +90° about x so that its up axis is z, as '
                              'DexVantage\'s world is.')
        self.xc = UsdGeom.XformCache(Usd.TimeCode.EarliestTime())
        roots = [p for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        bodies = [p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if roots:
            root = roots[0]
            if len(roots) > 1:
                self.notes.append(f'{self.base} has {len(roots)} articulations; the first, {root.GetName()}, is '
                                  'imported.')
            rp = root.GetPath()
            under = [b for b in bodies if b.GetPath().HasPrefix(rp)]
            if not under:                              # the API on a body's parent Xform elsewhere: every body
                under = bodies
            bodies = under
            self.root_prim = root
        else:
            self.root_prim = None
        if not bodies:
            raise UsdImportError(f'{self.base} is a USD file with no UsdPhysics rigid bodies (no prim has '
                                 'PhysicsRigidBodyAPI), so there is no robot to import: export the robot from its '
                                 'simulator with its physics (an articulation), or import its URDF or MJCF.')
        self.bodies = bodies
        self.body_paths = {b.GetPath() for b in bodies}
        self.prims = prims
        self.joints = [p for p in prims if p.IsA(UsdPhysics.Joint)]
        self._world = {}

    # ---------------------------------------------------------------- transforms
    def world(self, prim) -> np.ndarray:
        """The prim's local-to-world map in metres, Z up (4 × 4, column vectors, scale included)."""
        key = prim.GetPath()
        M = self._world.get(key)
        if M is None:
            M = self.C @ np.asarray(self.xc.GetLocalToWorldTransform(prim), float).T
            self._world[key] = M
        return M

    def frame(self, prim) -> np.ndarray:
        """The prim's frame: the rotation and translation of world(prim), its scale taken out."""
        M = self.world(prim)
        R, _S = _rigid(M[:3, :3])
        return _T(R, M[:3, 3])

    def stretch(self, prim) -> np.ndarray:
        """The stretch of the prim's world map relative to metres (identity for an unscaled prim)."""
        M = self.world(prim)
        _R, S = _rigid(M[:3, :3])
        return S

    def body_of(self, prim):
        """The rigid body a prim belongs to: itself or its nearest rigid-body ancestor in the set; None if none."""
        p = prim
        while p and p.IsValid() and not p.IsPseudoRoot():
            if p.GetPath() in self.body_paths:
                return p
            p = p.GetParent()
        return None


# ====================================================================== joints
@dataclass
class _Joint:
    prim: object
    name: str
    kind: str                     # hinge | slide | fixed | ball
    b0: object                    # a body prim, or None for the world
    b1: object
    F0: np.ndarray                # the joint frame on body0's side, in the world (m)
    F1: np.ndarray
    axis: str
    lower: float | None
    upper: float | None


def _limit(v, scale: float) -> float | None:
    if v is None:
        return None
    v = float(v)
    if not math.isfinite(v) or abs(v) >= LIMIT_SENTINEL:
        return None
    return v * scale


def _joints(S: _Stage) -> list:
    from pxr import UsdPhysics
    out, skipped = [], []
    for prim in S.joints:
        j = UsdPhysics.Joint(prim)
        nm = prim.GetName()
        if not _get(j.GetJointEnabledAttr(), True):
            skipped.append(f'{nm} (disabled)')
            continue
        if _get(j.GetExcludeFromArticulationAttr(), False):
            skipped.append(f'{nm} (excluded from the articulation)')
            continue
        ends = []
        for rel in (j.GetBody0Rel(), j.GetBody1Rel()):
            t = rel.GetTargets() if rel else []
            p = S.stage.GetPrimAtPath(t[0]) if t else None
            ends.append(p if p is not None and p.IsValid() else None)
        b0, b1 = (e if e is not None and e.GetPath() in S.body_paths else None for e in ends)
        if b0 is None and b1 is None:
            continue                                   # a joint of something else in the stage
        frames = []
        for prim_e, pos_a, rot_a in ((ends[0], j.GetLocalPos0Attr(), j.GetLocalRot0Attr()),
                                     (ends[1], j.GetLocalPos1Attr(), j.GetLocalRot1Attr())):
            pos = np.asarray(_get(pos_a, (0.0, 0.0, 0.0)), float)
            R = _gf_quat_matrix(_get(rot_a))
            if prim_e is None:                         # the world: localPos in stage units
                M = S.C
            else:
                M = S.world(prim_e)
            Rb, _St = _rigid(M[:3, :3])
            frames.append(_T(Rb @ R, (M @ np.r_[pos, 1.0])[:3]))
        if prim.IsA(UsdPhysics.RevoluteJoint):
            rj = UsdPhysics.RevoluteJoint(prim)
            kind, axis = 'hinge', str(_get(rj.GetAxisAttr(), 'X'))
            lo, hi = (_limit(_get(rj.GetLowerLimitAttr()), math.pi / 180),
                      _limit(_get(rj.GetUpperLimitAttr()), math.pi / 180))
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            pj = UsdPhysics.PrismaticJoint(prim)
            kind, axis = 'slide', str(_get(pj.GetAxisAttr(), 'X'))
            lo, hi = _limit(_get(pj.GetLowerLimitAttr()), S.mpu), _limit(_get(pj.GetUpperLimitAttr()), S.mpu)
        elif prim.IsA(UsdPhysics.FixedJoint):
            kind, axis, lo, hi = 'fixed', 'X', None, None
        elif prim.IsA(UsdPhysics.SphericalJoint):
            kind, axis, lo, hi = 'ball', 'X', None, None
        else:
            kind, axis, lo, hi = 'fixed', 'X', None, None
            S.notes.append(f'{nm} is a {prim.GetTypeName() or "generic"} joint, which is imported as fixed (only '
                           'revolute, prismatic, fixed and spherical joints are converted).')
        if axis not in AXES:
            axis = 'X'
        if lo is not None and hi is not None and lo > hi:
            lo = hi = None                             # UsdPhysics: lower above upper is no limit
        out.append(_Joint(prim, nm, kind, b0, b1, frames[0], frames[1], axis, lo, hi))
    if skipped:
        S.notes.append('Not imported: ' + ', '.join(skipped) + '.')
    return out


# ====================================================================== the tree
@dataclass
class _Node:
    prim: object
    name: str = ''
    parent: object = None         # _Node or None (the world)
    joint: _Joint | None = None
    reversed: bool = False
    children: list = field(default_factory=list)


def _tree(S: _Stage, joints: list) -> list:
    """[_Node] roots under the world; every body once."""
    nodes = {b.GetPath(): _Node(b) for b in S.bodies}
    adj: dict = {}
    world_j = []
    for j in joints:
        if j.b0 is None or j.b1 is None:
            world_j.append(j)
        else:
            if j.b0.GetPath() == j.b1.GetPath():
                continue
            adj.setdefault(j.b0.GetPath(), []).append(j)
            adj.setdefault(j.b1.GetPath(), []).append(j)
    roots, seen, used, loops = [], set(), set(), []

    def walk(start: _Node):
        seen.add(start.prim.GetPath())
        queue = [start]
        while queue:
            n = queue.pop(0)
            for j in adj.get(n.prim.GetPath(), []):
                if id(j) in used:
                    continue
                used.add(id(j))
                other = j.b1 if j.b0.GetPath() == n.prim.GetPath() else j.b0
                if other.GetPath() in seen:
                    loops.append(j.name)
                    continue
                c = nodes[other.GetPath()]
                c.parent, c.joint, c.reversed = n, j, j.b0.GetPath() != n.prim.GetPath()
                n.children.append(c)
                seen.add(other.GetPath())
                queue.append(c)

    for j in world_j:
        b = j.b1 if j.b1 is not None else j.b0
        if b.GetPath() in seen:
            loops.append(j.name)
            continue
        n = nodes[b.GetPath()]
        n.joint, n.reversed = j, j.b1 is None
        roots.append(n)
        walk(n)
    # floating bodies: the articulation root first, then bodies no joint names as its body1 (a tree's top)
    child = {j.b1.GetPath() for j in joints if j.b0 is not None and j.b1 is not None}
    top = S.root_prim.GetPath() if S.root_prim is not None else None
    order = sorted(S.bodies, key=lambda b: (b.GetPath() != top, b.GetPath() in child, str(b.GetPath())))
    floating = []
    for b in order:
        if b.GetPath() in seen:
            continue
        n = nodes[b.GetPath()]
        roots.append(n)
        floating.append(b.GetName())
        walk(n)
    if floating:
        S.notes.append(f'{", ".join(floating)} {"has" if len(floating) == 1 else "have"} no joint to the world (a '
                       'floating base), so it is fixed at its authored pose; DexVantage places the robot by its base '
                       'pose.')
    if loops:
        S.notes.append(f'Left out to keep a tree (the joint would close a kinematic loop): {", ".join(loops)}.')
    return roots


# ====================================================================== geometry
def _triangles(mesh) -> tuple[np.ndarray, np.ndarray] | None:
    from pxr import UsdGeom
    pts = _get(mesh.GetPointsAttr())
    counts = _get(mesh.GetFaceVertexCountsAttr())
    idx = _get(mesh.GetFaceVertexIndicesAttr())
    if pts is None or counts is None or idx is None or len(pts) < 3:
        return None
    V = np.asarray(pts, float).reshape(-1, 3)
    counts = np.asarray(counts, np.int64)
    idx = np.asarray(idx, np.int64)
    if counts.sum() != len(idx) or (idx < 0).any() or (idx >= len(V)).any():
        return None
    F = []
    start = 0
    for c in counts:
        if c >= 3:
            a = idx[start]
            for k in range(1, c - 1):
                F.append((a, idx[start + k], idx[start + k + 1]))
        start += c
    if not F:
        return None
    F = np.asarray(F, np.int64)
    if str(_get(mesh.GetOrientationAttr(), UsdGeom.Tokens.rightHanded)) == 'leftHanded':
        F = F[:, [0, 2, 1]]
    return V, F


def _rgba(gprim) -> tuple:
    col = _get(gprim.GetDisplayColorAttr())
    op = _get(gprim.GetDisplayOpacityAttr())
    c = tuple(float(x) for x in col[0]) if col is not None and len(col) else (0.6, 0.6, 0.6)
    a = float(op[0]) if op is not None and len(op) else 1.0
    return (*c, a)


def _geoms(S: _Stage, body, frame: np.ndarray, mesh_name, names: _Names, meshes: dict, skipped: dict,
           counts: dict) -> list:
    """MJCF <geom> (and <mesh> asset) text of every geometry prim of body, in the body's frame."""
    from pxr import Usd, UsdGeom, UsdPhysics
    from ..meshing import stl_bytes
    out, assets = [], []
    Xinv = _inv(frame)
    for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
        if prim == body or not prim.IsA(UsdGeom.Gprim):
            continue
        if S.body_of(prim) != body:
            continue
        g = UsdGeom.Gprim(prim)
        coll = prim.HasAPI(UsdPhysics.CollisionAPI)
        if not coll:
            if str(_get(g.GetVisibilityAttr(), 'inherited')) == 'invisible' or \
                    str(_get(g.GetPurposeAttr(), 'default')) in ('guide', 'proxy'):
                continue
        G = Xinv @ S.world(prim)                       # the prim's points in the body frame (m), scale included
        R, St = _rigid(G[:3, :3])
        sc = np.diag(St)
        name = names(prim.GetName())
        kind = prim.GetTypeName()
        attrs = (' contype="0" conaffinity="0" group="1"' if not coll else ' group="3"')
        rgba = _rgba(g)
        common = f' rgba="{_vec(rgba)}"{attrs}'
        if prim.IsA(UsdGeom.Mesh):
            tri = _triangles(UsdGeom.Mesh(prim))
            if tri is None:
                skipped.setdefault('meshes without points or faces (their layer may be missing)', []).append(
                    prim.GetName())
                continue
            V, F = tri
            Vb = V @ G[:3, :3].T + G[:3, 3]
            ext = Vb.max(axis=0) - Vb.min(axis=0)
            if (ext < 1e-9).sum() >= 2 or len(F) < 1:
                skipped.setdefault('degenerate meshes', []).append(prim.GetName())
                continue
            mn = mesh_name(prim.GetName())
            rel = f'{MESH_DIR}/{mn}.stl'
            meshes[rel] = stl_bytes(Vb, F, f'{body.GetName()} {prim.GetName()}: m, body frame')
            assets.append(f'    <mesh name="{mn}" file="{rel}" inertia="shell"/>')
            out.append(f'<geom name="{name}" type="mesh" mesh="{mn}"{common}/>')
            counts['mesh'] = counts.get('mesh', 0) + 1
            continue
        pos = _vec(G[:3, 3])
        if prim.IsA(UsdGeom.Cube):
            size = float(_get(UsdGeom.Cube(prim).GetSizeAttr(), 2.0))
            h = np.abs(sc) * size / 2
            out.append(f'<geom name="{name}" type="box" size="{_vec(h)}" pos="{pos}" quat="{_vec(_quat_wxyz(R))}"'
                       f'{common}/>')
        elif prim.IsA(UsdGeom.Sphere):
            r = float(_get(UsdGeom.Sphere(prim).GetRadiusAttr(), 1.0))
            s = np.abs(sc)
            if np.ptp(s) <= 1e-9 * max(1.0, s.max()):
                out.append(f'<geom name="{name}" type="sphere" size="{_f(r * s[0])}" pos="{pos}" '
                           f'quat="{_vec(_quat_wxyz(R))}"{common}/>')
            else:
                out.append(f'<geom name="{name}" type="ellipsoid" size="{_vec(r * s)}" pos="{pos}" '
                           f'quat="{_vec(_quat_wxyz(R))}"{common}/>')
        elif prim.IsA(UsdGeom.Capsule) or prim.IsA(UsdGeom.Cylinder):
            cap = prim.IsA(UsdGeom.Capsule)
            sch = UsdGeom.Capsule(prim) if cap else UsdGeom.Cylinder(prim)
            r = float(_get(sch.GetRadiusAttr(), 0.5 if cap else 1.0))
            hgt = float(_get(sch.GetHeightAttr(), 1.0 if cap else 2.0))
            ax = str(_get(sch.GetAxisAttr(), 'Z'))
            ax = ax if ax in AXES else 'Z'
            k = 'XYZ'.index(ax)
            s = np.abs(sc)
            across = [s[i] for i in range(3) if i != k]
            out.append(f'<geom name="{name}" type="{"capsule" if cap else "cylinder"}" '
                       f'size="{_f(r * float(np.mean(across)))} {_f(hgt / 2 * s[k])}" pos="{pos}" '
                       f'quat="{_vec(_quat_wxyz(R @ Z_TO[ax]))}"{common}/>')
            if np.ptp(across) > 1e-9 * max(1.0, max(across)):
                skipped.setdefault('kept with the mean radius (scaled unevenly across the axis)', []).append(
                    prim.GetName())
        else:
            skipped.setdefault(f'{kind} prims (no MuJoCo primitive)', []).append(prim.GetName())
            continue
        counts[kind] = counts.get(kind, 0) + 1
    return out, assets


def _inertial(S: _Stage, body, frame: np.ndarray) -> str | None:
    from pxr import UsdPhysics
    if not body.HasAPI(UsdPhysics.MassAPI):
        return None
    mapi = UsdPhysics.MassAPI(body)
    mass = _get(mapi.GetMassAttr())
    I = _get(mapi.GetDiagonalInertiaAttr())
    if mass is None or float(mass) <= 0 or I is None or not (np.asarray(I, float) > 0).all():
        return None
    com = _get(mapi.GetCenterOfMassAttr())
    com = np.zeros(3) if com is None or not np.isfinite(np.asarray(com, float)).all() or \
        np.asarray(com, float).min() <= -1e30 else np.asarray(com, float)
    M = S.world(body)
    c = (_inv(frame) @ (M @ np.r_[com, 1.0]))[:3]
    Rb, _ = _rigid(M[:3, :3])
    Rp = frame[:3, :3].T @ Rb @ _gf_quat_matrix(_get(mapi.GetPrincipalAxesAttr()))
    d = np.asarray(I, float) * S.kgpu * S.mpu ** 2
    return (f'<inertial pos="{_vec(c)}" quat="{_vec(_quat_wxyz(Rp))}" mass="{_f(float(mass) * S.kgpu)}" '
            f'diaginertia="{_vec(d)}"/>')


# ====================================================================== the conversion
def convert(path: str) -> Converted:
    """Convert the USD file at path (see the module docstring). Raises UsdImportError."""
    if not have_pxr():
        raise UsdImportError(missing_pxr_message(os.path.basename(path)))
    S = _Stage(path)
    joints = _joints(S)
    roots = _tree(S, joints)
    bnames, jnames, gnames, mnames = _Names(), _Names(), _Names(), _Names()
    meshes, assets, skipped, counts = {}, [], {}, {}
    no_mass, scaled, posed = [], [], []
    frames = {}
    for b in S.bodies:
        frames[b.GetPath()] = S.frame(b)
        st = S.stretch(b) / S.mpu
        if not np.allclose(st, np.eye(3), atol=1e-6):
            scaled.append(b.GetName())
    n_joint = {'hinge': 0, 'slide': 0, 'ball': 0, 'fixed': 0}

    def emit(n: _Node, ind: str) -> list:
        Xc = frames[n.prim.GetPath()]
        Xp = frames[n.parent.prim.GetPath()] if n.parent is not None else np.eye(4)
        j = n.joint
        jl = []
        if j is None:
            B = _inv(Xp) @ Xc                          # a floating base or body: its authored pose
        else:
            Fp, Fc = (j.F1, j.F0) if n.reversed else (j.F0, j.F1)
            J0, J1 = _inv(Xp) @ Fp, _inv(Xc) @ Fc
            B = J0 @ _inv(J1)
            A = _inv(Xp) @ Xc
            dp = float(np.linalg.norm(A[:3, 3] - B[:3, 3]))
            c = (np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2
            dr = math.acos(max(-1.0, min(1.0, c)))
            if dp > POSE_TOL or (j.kind == 'fixed' and dr > POSE_TOL):
                posed.append(f'{n.prim.GetName()} ({dp * 1000:.3g} mm, {math.degrees(dr):.3g}°)')
            n_joint[j.kind] += 1
            if j.kind in ('hinge', 'slide'):
                a = J1[:3, :3] @ AXES[j.axis] * (-1.0 if n.reversed else 1.0)
                rng = ''
                if j.lower is not None and j.upper is not None:
                    rng = f' limited="true" range="{_f(j.lower)} {_f(j.upper)}"'
                elif j.lower is not None or j.upper is not None:
                    S.notes.append(f'{j.name} states only one limit, so it is imported without limits.')
                jl.append(f'{ind}  <joint name="{jnames(j.name)}" type="{j.kind}" pos="{_vec(J1[:3, 3])}" '
                          f'axis="{_vec(a)}"{rng}/>')
            elif j.kind == 'ball':
                jl.append(f'{ind}  <joint name="{jnames(j.name)}" type="ball" pos="{_vec(J1[:3, 3])}"/>')
        n.name = bnames(n.prim.GetName())
        lines = [f'{ind}<body name="{n.name}" pos="{_vec(B[:3, 3])}" quat="{_vec(_quat_wxyz(B[:3, :3]))}">']
        ine = _inertial(S, n.prim, Xc)
        if ine is None:
            no_mass.append(n.prim.GetName())
        else:
            lines.append(f'{ind}  {ine}')
        lines += jl
        gl, al = _geoms(S, n.prim, Xc, mnames, gnames, meshes, skipped, counts)
        assets.extend(al)
        lines += [f'{ind}  {g}' for g in gl]
        for c in n.children:
            lines += emit(c, ind + '  ')
        lines.append(f'{ind}</body>')
        return lines

    body_lines = []
    for r in roots:
        body_lines += emit(r, '    ')
    # the robot's name: its articulation prim, unless that is a link (the API often sits on the base body, which
    # named the newton-assets UR5e "base"); then the stage's default prim, else the file name
    from pxr import UsdPhysics
    rp = S.root_prim
    use_root = rp is not None and not rp.HasAPI(UsdPhysics.RigidBodyAPI)
    model = _name(rp.GetName() if use_root else
                  (S.stage.GetDefaultPrim().GetName() if S.stage.GetDefaultPrim() else
                   os.path.splitext(S.base)[0]))
    head = (f'<!-- Converted by DexVantage from the USD file {S.base} (UsdPhysics articulation '
            f'{S.root_prim.GetPath() if S.root_prim is not None else "none: every rigid body"}): metres, radians, '
            'Z up. Mass and inertia only where the file states them (inertiafromgeom false). -->')
    mj = ['<mujoco model="' + model + '">', '  ' + head,
          '  <compiler angle="radian" inertiafromgeom="false" autolimits="true"/>']
    if assets:
        mj += ['  <asset>'] + assets + ['  </asset>']
    mj += ['  <worldbody>'] + body_lines + ['  </worldbody>', '</mujoco>']
    text = '\n'.join(mj) + '\n'
    ET.fromstring(text)                                # well-formed (names are escaped by _name)
    notes = list(S.notes)
    if posed:
        notes.append('The authored poses of ' + ', '.join(posed) + ' differ from where the joints place them; the '
                     'joints are used (the file may be authored at another joint position).')
    if no_mass:
        notes.append(f'The file states no mass and inertia for {", ".join(no_mass)}; the converted file gives '
                     f'{"it" if len(no_mass) == 1 else "them"} none (DexVantage\'s kinematics does not use mass).')
    if scaled:
        notes.append(f'{", ".join(scaled)} {"is" if len(scaled) == 1 else "are"} scaled in the file: the scale is '
                     'applied to points and primitives, not to joint-frame rotations or inertia.')
    for why, names in sorted(skipped.items()):
        lead = 'Geoms' if why.startswith('kept') else 'Left out'
        shown = ', '.join(names[:8]) + (f' and {len(names) - 8} more' if len(names) > 8 else '')
        notes.append(f'{lead} {why}: {len(names)} ({shown}).' if lead == 'Left out' else f'{lead} {why}: {shown}.')
    notes.append('Joint drives, friction, damping, armature, effort and velocity limits are not imported (not '
                 'stated in DexVantage\'s joint records).')
    layers = []
    for lay in S.stage.GetUsedLayers():
        rp = lay.realPath or ''
        if rp and os.path.isfile(rp):
            with open(rp, 'rb') as fh:
                layers.append({'layer': os.path.basename(rp), 'sha256': hashlib.sha256(fh.read()).hexdigest()})
    record = {'format': RECORD_FORMAT, 'source': S.base, 'articulation': str(S.root_prim.GetPath())
              if S.root_prim is not None else None, 'meters_per_unit': S.mpu, 'kilograms_per_unit': S.kgpu,
              'up_axis': S.up, 'bodies': len(S.bodies), 'joints': dict(n_joint), 'geoms': counts,
              'layers': sorted(layers, key=lambda d: d['layer']), 'notes': notes,
              'converter': 'DexVantage core/robot/usd_import.py (pxr ' + _pxr_version() + ')'}
    return Converted(model, text, meshes, notes, record)


def _pxr_version() -> str:
    try:
        from pxr import Usd
        return '.'.join(str(v) for v in Usd.GetVersion())
    except Exception:                                  # noqa: BLE001
        return '?'


def convert_to_folder(path: str, folder: str) -> Converted:
    """convert(path), written as folder/robot.xml, folder/meshes/*.stl and folder/.dexvantage/usd_source.json."""
    c = convert(path)
    os.makedirs(os.path.join(folder, MESH_DIR), exist_ok=True)
    with open(os.path.join(folder, MAIN), 'w', encoding='utf-8') as fh:
        fh.write(c.mjcf)
    for rel, data in c.meshes.items():
        with open(os.path.join(folder, *rel.split('/')), 'wb') as fh:
            fh.write(data)
    return c


def record_bytes(c: Converted) -> bytes:
    return json.dumps(c.record, indent=1, sort_keys=True).encode('utf-8')


# ====================================================================== the wizard's file check
def check_references(path: str) -> dict:
    """{'kind': 'usd', 'items': [...]} as core/robot/filecheck.check_references gives for URDF and MJCF: each layer
    the stage uses (found) and each asset path it names that does not resolve (missing; not required: the stage
    composes without it, and a mesh it held is left out). Raises UsdImportError when the file has no robot."""
    if not have_pxr():
        raise UsdImportError(missing_pxr_message(os.path.basename(path)))
    from pxr import UsdUtils
    S = _Stage(path)
    items, unresolved = [], []
    for lay in S.stage.GetUsedLayers():
        if lay.anonymous:
            continue
        rp = lay.realPath or lay.identifier
        items.append({'what': 'layer', 'ref': os.path.basename(rp), 'found': True, 'file': os.path.basename(rp),
                      'required': True, 'message': ''})
    delegate = UsdUtils.CoalescingDiagnosticDelegate()    # its warnings would print full paths
    try:
        _layers, _assets, unresolved = UsdUtils.ComputeAllDependencies(path)
    except Exception:                                  # noqa: BLE001 — the list is a courtesy
        unresolved = []
    finally:
        delegate.TakeUncoalescedDiagnostics()
        del delegate
    for u in unresolved:
        items.append({'what': 'layer', 'ref': str(u), 'found': False, 'file': '', 'required': False,
                      'message': f'Not found: {os.path.basename(str(u))}; what it holds (often meshes) is left out.'})
    return {'kind': 'usd', 'items': items}


__all__ = ['USD_EXTS', 'Converted', 'UsdImportError', 'convert', 'convert_to_folder', 'check_references', 'have_pxr',
           'missing_pxr_message', 'record_bytes']
