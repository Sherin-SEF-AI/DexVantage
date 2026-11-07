"""The robot's kinematics model, compiled by MuJoCo from the project's stored files.

RobotModel(robot, sources) compiles the robot's URDF or MJCF from memory (mujoco.MjSpec with an in-memory asset
dictionary; nothing is read from disk) and gives:

    .joint_names, .nq, .lower, .upper, .home     the actuated one-degree-of-freedom joints (hinge, slide) in MuJoCo's
                                                 tree order; radians or metres; ±inf where the file sets no limit
    .link_names                                  every link (MuJoCo body, plus URDF links MuJoCo merged away)
    .fk(q) / .link_pose(q, link) / .fk_batch(Q)  4×4 world transforms (world = Robot.base applied to MuJoCo's frame)
    .jacobian(q, link, point=None)               (6, nq) world-frame [linear; angular] Jacobian of a point fixed in
                                                 the link, from mujoco.mj_jac
    .visual_meshes() / .collision_geoms()        [(link, vertices (V,3) in the link frame, faces (F,3), rgba)]
    .posed_triangles(q)                          (tri (M,3,3) world, normals (M,3), link index (M,)) of the visual
                                                 meshes, for drawing and occlusion; .triangle_rgba() their colours

get_model(project, robot_uid) shares compiled models between the UI and the studies (thread-safe), keyed by the
robot's stored files (SHA-256), its records and its base pose.

URDF goes through MuJoCo's URDF compiler with discardvisual="false" (visual meshes kept, in group 1, not colliding)
and fusestatic="false" (bodies joined by fixed joints kept, so their names survive; an MJCF file's own
fusestatic="true" is overridden the same way, so its jointless bodies stay frames). A URDF link MuJoCo still merges
away (a link named "world", which becomes MuJoCo's world body) is mapped to the body it was merged into plus the fixed
offset from the URDF's joint records. Visual and collision geoms are told apart by MuJoCo's collision masks: a geom
with contype = conaffinity = 0 is visual-only; a model with no such geoms uses every geom for both.

When MuJoCo refuses a model only because a moving link has no mass or inertia (common in kinematics-only URDFs), the
model is compiled again with boundmass = 1e-6 kg and boundinertia = 1e-9 kg·m²; .notes says so. Kinematics does not
depend on mass; nothing here uses those figures.

Known limits: free and ball joints are held at the file's home keyframe (or MuJoCo's qpos0) and are not part of q;
URDF mimic joints move independently; planes and height fields are not drawn; primitives are tessellated with 24
segments around (a drawing approximation, at most 0.9 % of the radius short of the true surface).
"""
from __future__ import annotations

import json
import os
import posixpath
import re
import threading
import xml.etree.ElementTree as ET
from collections import OrderedDict

import numpy as np

from ..frames import Pose, quat_from_mujoco, quat_to_matrix
from ..project import ProjectFormatError
from .importer import MANIFEST, mjcf_spec_files

SEGMENTS = 24                     # around a primitive's axis
RINGS = 12                        # pole to pole on a sphere
BOUND_MASS = 1e-6                 # kg, only when MuJoCo refuses a massless moving link (see the module docstring)
BOUND_INERTIA = 1e-9              # kg·m²
_DEFAULT_RGBA = (0.5, 0.5, 0.5, 1.0)


class RobotModelError(ProjectFormatError):
    """The robot's stored files cannot be compiled; the message says why in plain words."""


# ====================================================================== compiling from memory
class _Compiled:
    def __init__(self, m, kind, mesh_keys, notes):
        self.m = m
        self.kind = kind
        self.mesh_keys = mesh_keys        # per MuJoCo mesh id: the stored source key it came from ('' if none)
        self.notes = notes


def _manifest(robot, sources) -> dict:
    raw = sources.get(f'{robot.uid}/{MANIFEST}')
    if raw is None:
        return {}
    try:
        return json.loads(raw.decode('utf-8'))
    except (UnicodeDecodeError, ValueError):
        return {}


def _flat_namer():
    used = set()

    def name(rel: str) -> str:
        stem, ext = os.path.splitext(posixpath.basename(rel))
        stem = re.sub(r'[^A-Za-z0-9_.-]', '_', stem) or 'mesh'
        cand, i = stem, 1
        while cand.lower() in used:
            cand, i = f'{stem}_{i}', i + 1
        used.add(cand.lower())
        return cand + ext
    return name


class _Resolver:
    """Which stored file a reference means: the import manifest first, then the reference resolved against the main
    file's stored folder (for robots stored without a manifest)."""

    def __init__(self, robot, sources):
        self.uid = robot.uid
        self.sources = sources
        self.prefix = robot.uid + '/'
        man = _manifest(robot, sources)
        self.refs = man.get('refs', {}) if isinstance(man.get('refs'), dict) else {}
        main_key = robot.source.file
        self.main_rel = main_key[len(self.prefix):] if main_key.startswith(self.prefix) else main_key
        self.rels = [k[len(self.prefix):] for k in sources if k.startswith(self.prefix)]

    def key(self, cat: str, ref: str, folder: str = '') -> str | None:
        rel = self.refs.get(f'{cat}:{ref}')
        if rel is not None and self.prefix + rel in self.sources:
            return self.prefix + rel
        here = posixpath.dirname(self.main_rel)
        r = ref
        for p in ('package://', 'file://'):
            if r.startswith(p):
                r = r[len(p):]
        cands = [posixpath.normpath(posixpath.join(here, folder, r)), posixpath.normpath(posixpath.join(here, r))]
        for c in cands:
            if c in self.rels:
                return self.prefix + c
        tail = r.lstrip('/')
        hits = sorted((x for x in self.rels if x == tail or x.endswith('/' + tail)), key=len)
        if hits:
            return self.prefix + hits[0]
        hits = sorted((x for x in self.rels if posixpath.basename(x) == posixpath.basename(r)), key=len)
        return self.prefix + hits[0] if hits else None


def _first_line(e) -> str:
    s = str(e).strip()
    return s.splitlines()[0] if s else type(e).__name__


def _compile_urdf(robot, sources, res: _Resolver, bound: bool):
    import mujoco
    try:
        root = ET.fromstring(sources[robot.source.file])
    except ET.ParseError as e:
        raise RobotModelError(f'The stored URDF of {robot.name} is not valid XML ({e}).') from None
    name = _flat_namer()
    assets, flat_key, by_rel = {}, {}, {}
    for m in root.iter('mesh'):
        fn = m.get('filename')
        if not fn:
            continue
        key = res.key('mesh', fn)
        if key is None:
            raise RobotModelError(f'The URDF of {robot.name} names mesh {posixpath.basename(fn)}, which is not '
                                  'stored in the project: import the robot again from its package folder.')
        if key not in by_rel:
            by_rel[key] = name(key)
            assets[by_rel[key]] = sources[key]
            flat_key[by_rel[key]] = key
        m.set('filename', by_rel[key])
    mj = root.find('mujoco')
    if mj is None:
        mj = ET.Element('mujoco')
        root.insert(0, mj)
    comp = mj.find('compiler')
    if comp is None:
        comp = ET.SubElement(mj, 'compiler')
    comp.set('discardvisual', 'false')
    comp.set('fusestatic', 'false')
    comp.set('strippath', 'false')
    for k in ('meshdir', 'assetdir', 'texturedir'):
        comp.attrib.pop(k, None)
    if bound:
        comp.set('boundmass', repr(BOUND_MASS))
        comp.set('boundinertia', repr(BOUND_INERTIA))
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding='unicode'))
    spec.assets = assets
    m = spec.compile()
    return m, flat_key, []


def _compile_mjcf(robot, sources, res: _Resolver, bound: bool):
    import mujoco
    incs = {}
    for k, rel in res.refs.items():
        if k.startswith('include:') and res.prefix + rel in sources:
            incs[k[len('include:'):]] = sources[res.prefix + rel]
    try:
        spec = mujoco.MjSpec.from_string(sources[robot.source.file].decode('utf-8'), include=incs)
    except Exception as e:                             # noqa: BLE001 — explained
        raise RobotModelError(f'MuJoCo could not read the stored MJCF of {robot.name}: {_first_line(e)}.') from None
    name = _flat_namer()
    assets, flat_key, by_key, drop, notes = {}, {}, {}, [], []
    for cat, el, attr, fn in mjcf_spec_files(spec):
        folder = spec.texturedir if cat == 'texture' else spec.meshdir
        key = res.key(cat, fn, folder)
        if key is None:
            if cat == 'texture':
                if el not in drop:
                    drop.append(el)
                    notes.append(f'texture {posixpath.basename(fn)} not stored: drawn without it')
                continue
            raise RobotModelError(f'The MJCF of {robot.name} names the {cat} {posixpath.basename(fn)}, which is not '
                                  'stored in the project: import the robot again from its complete folder.')
        if key not in by_key:
            by_key[key] = name(key)
            assets[by_key[key]] = sources[key]
            flat_key[by_key[key]] = key
        if attr == 'file':
            el.file = by_key[key]
        else:
            cf = list(el.cubefiles)
            cf[attr] = by_key[key]
            el.cubefiles = cf
    for t in drop:
        for mat in spec.materials:
            mat.textures = ['' if x == t.name else x for x in mat.textures]
        spec.delete(t)
    spec.meshdir = ''
    spec.texturedir = ''
    spec.compiler.fusestatic = False                   # as for URDF: jointless bodies keep their names (link frames)
    if bound:
        spec.compiler.boundmass = BOUND_MASS
        spec.compiler.boundinertia = BOUND_INERTIA
    spec.assets = assets
    m = spec.compile()
    return m, flat_key, notes


def compile_robot(robot, sources) -> _Compiled:
    """Compile the robot's stored files with MuJoCo (no caching; get_model caches)."""
    if robot.source.file not in sources:
        raise RobotModelError(f'The model file of {robot.name} is not stored in the project, so it cannot be drawn '
                              'or analysed: import the robot again.')
    kind = robot.source.kind or 'urdf'
    if kind not in ('urdf', 'mjcf'):
        raise RobotModelError(f'{robot.name} comes from a {kind} file, which this build cannot compile yet.')
    res = _Resolver(robot, sources)
    fn = _compile_urdf if kind == 'urdf' else _compile_mjcf
    notes = []
    try:
        m, flat_key, n2 = fn(robot, sources, res, False)
    except RobotModelError:
        raise
    except Exception as e:                             # noqa: BLE001 — retried or explained
        msg = _first_line(e)
        dec = re.search(r"no decoder found for mesh file '([^']+)'", msg)
        if dec:
            f = posixpath.basename(dec.group(1))
            ext = posixpath.splitext(f)[1].lower()
            what = {'.dae': 'a COLLADA (.dae) mesh', '.ply': 'a PLY mesh', '.glb': 'a glTF mesh',
                    '.gltf': 'a glTF mesh'}.get(ext, f'a {ext or "mesh"} file')
            raise RobotModelError(f'{robot.name} names {f}, {what}, which MuJoCo cannot read. Convert the mesh to '
                                  'STL or OBJ (Blender or MeshLab can), point the model file at the converted file and '
                                  'import again.') from None
        if 'mass' not in msg.lower() and 'inertia' not in msg.lower():
            raise RobotModelError(f'MuJoCo could not compile {robot.name}: {msg}.') from None
        try:
            m, flat_key, n2 = fn(robot, sources, res, True)
        except Exception as e2:                        # noqa: BLE001 — explained
            raise RobotModelError(f'MuJoCo could not compile {robot.name}: {_first_line(e2)}.') from None
        notes.append(f'MuJoCo refused a moving link without mass or inertia ({msg}); compiled with boundmass '
                     f'{BOUND_MASS:g} kg and boundinertia {BOUND_INERTIA:g} kg·m², which kinematics does not use.')
    notes += n2
    mesh_keys = []
    paths = bytes(m.paths)
    for i in range(m.nmesh):
        a = int(m.mesh_pathadr[i])
        p = paths[a:paths.index(b'\0', a)].decode('utf-8', 'replace') if a >= 0 else ''
        mesh_keys.append(flat_key.get(posixpath.basename(p), ''))
    return _Compiled(m, kind, mesh_keys, notes)


# ====================================================================== primitives
def _lathe(profile, n=SEGMENTS):
    """Vertices and faces of a surface of revolution about z: profile [(rho, z)] from bottom to top."""
    prof = np.asarray(profile, float)
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    V = np.stack([np.outer(prof[:, 0], np.cos(a)), np.outer(prof[:, 0], np.sin(a)),
                  np.repeat(prof[:, 1:2], n, axis=1)], axis=-1).reshape(-1, 3)
    K = len(prof)
    F = []
    for k in range(K - 1):
        for j in range(n):
            v00, v01 = k * n + j, k * n + (j + 1) % n
            v10, v11 = (k + 1) * n + j, (k + 1) * n + (j + 1) % n
            F += [(v00, v01, v11), (v00, v11, v10)]
    F = np.asarray(F, np.int32)
    t = V[F]
    area = np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    return V, F[area > 1e-14]


def _sphere_profile(r, z0=0.0, a0=-np.pi / 2, a1=np.pi / 2, rings=RINGS):
    a = np.linspace(a0, a1, rings + 1)
    return np.c_[r * np.cos(a), z0 + r * np.sin(a)]


def _box(h):
    hx, hy, hz = h
    V = np.array([[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)], float)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    F = np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))], np.int32)
    return V, F


def _outward(V, F):
    """Faces of a convex shape centred on the origin, wound so their normals point outward."""
    t = V[F]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    flip = np.einsum('ij,ij->i', n, t.mean(axis=1)) < 0
    F = F.copy()
    F[flip] = F[flip][:, [0, 2, 1]]
    return F


def primitive_mesh(gtype: int, size) -> tuple[np.ndarray, np.ndarray] | None:
    """(V, F) of a MuJoCo primitive geom in its own frame, or None for types not drawn (plane, height field)."""
    import mujoco
    G = mujoco.mjtGeom
    gtype = int(gtype)
    s = np.asarray(size, float)
    if gtype == int(G.mjGEOM_BOX):
        V, F = _box(s)
    elif gtype == int(G.mjGEOM_SPHERE):
        V, F = _lathe(_sphere_profile(s[0]))
    elif gtype == int(G.mjGEOM_ELLIPSOID):
        V, F = _lathe(_sphere_profile(1.0))
        V = V * s[:3]
    elif gtype == int(G.mjGEOM_CYLINDER):
        r, h = s[0], s[1]
        V, F = _lathe([(0, -h), (r, -h), (r, h), (0, h)])
    elif gtype == int(G.mjGEOM_CAPSULE):
        r, h = s[0], s[1]
        prof = np.r_[_sphere_profile(r, -h, -np.pi / 2, 0, RINGS // 2), _sphere_profile(r, h, 0, np.pi / 2,
                                                                                         RINGS // 2)]
        V, F = _lathe(prof)
    else:
        return None
    return V, _outward(V, F)


# ====================================================================== the model
def _T(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


class RobotModel:
    """A robot compiled by MuJoCo: kinematics, Jacobians and meshes (see the module docstring)."""

    def __init__(self, robot, sources: dict, compiled: _Compiled | None = None):
        import mujoco
        c = compiled or compile_robot(robot, sources)
        self.m = c.m
        self.kind = c.kind
        self.robot_uid = robot.uid
        self.name = robot.name
        self.notes = list(c.notes)
        # the compile may be shared with a robot of another uid read from the same files: its keys name that uid
        self.mesh_keys = [f'{robot.uid}/{k.split("/", 1)[1]}' if '/' in k else k for k in c.mesh_keys]
        self.base = robot.base.matrix()
        self._tls = threading.local()
        m = self.m
        J = mujoco.mjtJoint
        # ---------------------------------------------------------------- links
        self.body_link = ['world'] + [m.body(b).name or f'body_{b}' for b in range(1, m.nbody)]
        lb = {n: (b, np.eye(4)) for b, n in enumerate(self.body_link) if b > 0}
        recs = {j.get('child'): j for j in robot.joints if isinstance(j, dict) and j.get('type') == 'fixed'}
        wanted = [ln.get('name') for ln in robot.links if isinstance(ln, dict) and ln.get('name')]

        def place(name, depth=0):
            if name in lb:
                return lb[name]
            if name == 'world':
                return 0, np.eye(4)
            j = recs.get(name)
            if j is None or depth > 64:
                return None
            up = place(j.get('parent', ''), depth + 1)
            if up is None:
                return None
            return up[0], up[1] @ Pose.from_dict(j.get('origin')).matrix()

        for n in wanted:
            if n not in lb:
                p = place(n)
                if p is not None:
                    lb[n] = p
        self.link_names = [n for n in self.body_link[1:]] + [n for n in wanted if n in lb and n not in
                                                              self.body_link[1:]]
        self._lb = np.array([lb[n][0] for n in self.link_names], int)
        self._loff = np.array([lb[n][1] for n in self.link_names]).reshape(-1, 4, 4)
        self._link_index = {n: i for i, n in enumerate(self.link_names)}
        self._body_own = {}                             # body -> the link whose frame it is (offset identity)
        for i, n in enumerate(self.link_names):
            b = int(self._lb[i])
            if b not in self._body_own and np.allclose(self._loff[i], np.eye(4)):
                self._body_own[b] = i
        # ---------------------------------------------------------------- joints
        act = [j for j in range(m.njnt) if int(m.jnt_type[j]) in (int(J.mjJNT_HINGE), int(J.mjJNT_SLIDE))]
        self._jid = np.array(act, int)
        self.joint_names = [self.joint_name(j) for j in act]
        self.nq = len(act)
        self._qadr = np.array([m.jnt_qposadr[j] for j in act], int)
        self._dadr = np.array([m.jnt_dofadr[j] for j in act], int)
        self.joint_types = ['prismatic' if int(m.jnt_type[j]) == int(J.mjJNT_SLIDE) else
                            ('revolute' if m.jnt_limited[j] else 'continuous') for j in act]
        byname = {j.get('name'): j for j in robot.joints if isinstance(j, dict)}
        lo, hi = np.full(self.nq, -np.inf), np.full(self.nq, np.inf)
        for i, j in enumerate(act):
            r = byname.get(self.joint_names[i])
            if r is not None and (r.get('lower') is not None or r.get('upper') is not None):
                lo[i] = -np.inf if r.get('lower') is None else float(r['lower'])
                hi[i] = np.inf if r.get('upper') is None else float(r['upper'])
            elif r is None and m.jnt_limited[j]:
                lo[i], hi[i] = m.jnt_range[j]
        self.lower, self.upper = lo, hi
        kh = self._key('home')
        self._q0 = (m.key_qpos[kh] if kh is not None else m.qpos0).astype(float).copy()
        fh = self.file_home()
        rh = robot.home if isinstance(robot.home, dict) else {}
        self.home = np.clip(np.array([float(rh.get(n, fh[i])) for i, n in enumerate(self.joint_names)]), lo, hi) \
            if self.nq else np.zeros(0)
        # ---------------------------------------------------------------- geoms
        self._visual, self._collision = self._geoms()
        if self._visual:
            tri = np.concatenate([V[F] for _l, V, F, _c, _g in self._visual]).astype(np.float64)
            li = np.concatenate([np.full(len(F), self._link_index[ln], np.int32)
                                 for ln, _V, F, _c, _g in self._visual])
            rgba = np.concatenate([np.tile(np.asarray(c, np.float32), (len(F), 1))
                                   for _l, _V, F, c, _g in self._visual])
        else:
            tri, li, rgba = np.zeros((0, 3, 3)), np.zeros(0, np.int32), np.zeros((0, 4), np.float32)
        self._tri_local, self._tri_link, self._tri_rgba = tri, li, rgba
        self._posed_cache = None
        self._posed_lock = threading.Lock()
        self._bounds_cache = None          # (q, lo, hi): the frame button asks for the same pose many times

    # ---------------------------------------------------------------- names
    def joint_name(self, j: int) -> str:
        return self.m.joint(j).name or f'joint_{j}'

    def _key(self, name):
        for k in range(self.m.nkey):
            if self.m.key(k).name == name:
                return k
        return None

    def file_home(self) -> np.ndarray:
        """The file's keyframe "home" at the actuated joints, else zeros; clipped to the limits."""
        k = self._key('home')
        q = self.m.key_qpos[k][self._qadr].astype(float) if k is not None else np.zeros(self.nq)
        return np.clip(q, self.lower, self.upper)

    def clip(self, q) -> np.ndarray:
        return np.clip(self._q(q), self.lower, self.upper)

    # ---------------------------------------------------------------- kinematics
    def _q(self, q) -> np.ndarray:
        q = np.asarray(q, float).reshape(-1)
        if len(q) != self.nq:
            raise ValueError(f'{self.name} has {self.nq} moving joints; got {len(q)} joint values.')
        return q

    def _data(self):
        d = getattr(self._tls, 'd', None)
        if d is None:
            import mujoco
            d = self._tls.d = mujoco.MjData(self.m)
        return d

    def _kin(self, q):
        import mujoco
        d = self._data()
        d.qpos[:] = self._q0
        if self.nq:
            d.qpos[self._qadr] = q
        mujoco.mj_kinematics(self.m, d)
        return d

    def _bodies_T(self, d) -> np.ndarray:
        nb = self.m.nbody
        T = np.zeros((nb, 4, 4))
        T[:, :3, :3] = d.xmat.reshape(nb, 3, 3)
        T[:, :3, 3] = d.xpos
        T[:, 3, 3] = 1.0
        return T

    def _links_T(self, d, world=True) -> np.ndarray:
        T = self._bodies_T(d)[self._lb] @ self._loff
        return self.base @ T if world else T

    def fk_array(self, q) -> np.ndarray:
        """(n_links, 4, 4) world transforms in link_names order."""
        return self._links_T(self._kin(self._q(q)))

    def fk(self, q) -> dict:
        """link name -> 4×4 world transform (Robot.base applied)."""
        T = self.fk_array(q)
        return {n: T[i] for i, n in enumerate(self.link_names)}

    def link_index(self, link: str) -> int:
        i = self._link_index.get(link)
        if i is None:
            raise KeyError(f'{self.name} has no link named {link}.')
        return i

    def link_pose(self, q, link: str) -> np.ndarray:
        i = self.link_index(link)
        return self.fk_array(q)[i]

    def batch_q(self, Q) -> np.ndarray:
        """Q as an (N, nq) array: one joint vector (nq,) or a stack (N, nq). A stack whose rows are not nq long is
        refused (reshaping it would silently split and join joint vectors)."""
        Q = np.asarray(Q, float)
        if Q.ndim == 1 and len(Q) == self.nq:
            return Q.reshape(1, self.nq)
        if Q.ndim == 2 and Q.shape[1] == self.nq:
            return Q
        raise ValueError(f'{self.name} has {self.nq} moving joints; expected joint vectors of shape (N, {self.nq}), '
                         f'got an array of shape {Q.shape}.')

    def fk_batch(self, Q) -> np.ndarray:
        """(N, n_links, 4, 4) world transforms for joint vectors Q (N, nq); one MjData reused."""
        Q = self.batch_q(Q)
        out = np.empty((len(Q), len(self.link_names), 4, 4))
        for k, q in enumerate(Q):
            out[k] = self._links_T(self._kin(q))
        return out

    def jacobian(self, q, link: str, point=None) -> np.ndarray:
        """(6, nq) world-frame Jacobian [linear; angular] of a point fixed in the link (link coordinates, default its
        origin): d(point, rotation)/dq for the actuated joints."""
        import mujoco
        i = self.link_index(link)
        d = self._kin(self._q(q))
        mujoco.mj_comPos(self.m, d)
        Tl = self._links_T(d, world=False)[i]
        p = np.zeros(3) if point is None else np.asarray(point, float).reshape(3)
        pm = Tl[:3, :3] @ p + Tl[:3, 3]
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        mujoco.mj_jac(self.m, d, jp, jr, pm, int(self._lb[i]))
        R = self.base[:3, :3]
        return np.r_[R @ jp[:, self._dadr], R @ jr[:, self._dadr]]

    # ---------------------------------------------------------------- geometry
    def _geoms(self):
        import mujoco
        m = self.m
        G = mujoco.mjtGeom
        nonc = (m.geom_contype == 0) & (m.geom_conaffinity == 0)
        vis_mask = nonc if nonc.any() else np.ones(m.ngeom, bool)
        col_mask = ~nonc if nonc.any() else np.ones(m.ngeom, bool)
        vis, col, skipped = [], [], set()
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            li = self._body_own.get(b)
            if li is None:
                continue                                 # world-body geoms (floors) are not the robot's
            t = int(m.geom_type[g])
            if t == int(G.mjGEOM_MESH):
                k = int(m.geom_dataid[g])
                va, vn = m.mesh_vertadr[k], m.mesh_vertnum[k]
                fa, fn = m.mesh_faceadr[k], m.mesh_facenum[k]
                V = m.mesh_vert[va:va + vn].astype(float)
                F = m.mesh_face[fa:fa + fn].astype(np.int32)
            else:
                pm = primitive_mesh(t, m.geom_size[g])
                if pm is None:
                    skipped.add(G(t).name[len('mjGEOM_'):].lower())
                    continue
                V, F = pm
            R = quat_to_matrix(quat_from_mujoco(m.geom_quat[g]))
            V = (V @ R.T + m.geom_pos[g]).astype(np.float32)
            rgba = np.asarray(m.geom_rgba[g], np.float32)
            mat = int(m.geom_matid[g])
            if mat >= 0 and np.allclose(rgba, _DEFAULT_RGBA):
                rgba = np.asarray(m.mat_rgba[mat], np.float32)
            rec = (self.link_names[li], V, F, rgba.copy(), g)
            if vis_mask[g] and rgba[3] > 0:
                vis.append(rec)
            if col_mask[g]:
                col.append(rec)
        if skipped:
            self.notes.append('Not drawn: ' + ', '.join(sorted(skipped)) + ' geoms.')
        return vis, col

    def visual_meshes(self) -> list:
        """[(link name, vertices (V,3) float32 in the link frame, faces (F,3) int32, rgba (4,) float32)]."""
        return [(ln, V.copy(), F.copy(), c.copy()) for ln, V, F, c, _g in self._visual]

    def collision_geoms(self) -> list:
        """Like visual_meshes, for the colliding geoms."""
        return [(ln, V.copy(), F.copy(), c.copy()) for ln, V, F, c, _g in self._collision]

    def body_mesh_keys(self, body: int, visual: bool = True) -> list:
        """The stored source keys of the mesh files drawn (visual) or colliding on a MuJoCo body."""
        import mujoco
        out = []
        for _ln, _V, _F, _c, g in (self._visual if visual else self._collision):
            if int(self.m.geom_bodyid[g]) == body and int(self.m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH):
                k = self.mesh_keys[int(self.m.geom_dataid[g])]
                if k and k not in out:
                    out.append(k)
        return out

    def mesh_extents(self) -> list:
        """[(mesh name, stored key, largest bounding-box edge in metres, after the file's scale)]."""
        out = []
        for i in range(self.m.nmesh):
            va, vn = self.m.mesh_vertadr[i], self.m.mesh_vertnum[i]
            V = self.m.mesh_vert[va:va + vn]
            ext = float((V.max(axis=0) - V.min(axis=0)).max()) if vn else 0.0
            out.append((self.m.mesh(i).name, self.mesh_keys[i] if i < len(self.mesh_keys) else '', ext))
        return out

    @property
    def n_triangles(self) -> int:
        return len(self._tri_local)

    def triangle_rgba(self) -> np.ndarray:
        """(M, 4) float32 colours of posed_triangles' triangles (fixed: they do not depend on q)."""
        return self._tri_rgba

    def triangle_links(self) -> np.ndarray:
        """(M,) int32 link index (into link_names) of each of posed_triangles' triangles."""
        return self._tri_link

    def local_triangles(self) -> np.ndarray:
        """(M, 3, 3) float64 visual triangles in their own link's frame (read-only: do not change in place)."""
        return self._tri_local

    def _link_runs(self) -> list:
        """[(start, stop, link index)]: the runs of consecutive triangles that belong to one link (the meshes are
        stored link by link, so a robot has about one run per link)."""
        runs = getattr(self, '_runs', None)
        if runs is None:
            li = self._tri_link
            cut = np.flatnonzero(np.diff(li)) + 1 if len(li) else np.zeros(0, int)
            starts = np.r_[0, cut].astype(int) if len(li) else np.zeros(0, int)
            stops = np.r_[cut, len(li)].astype(int) if len(li) else np.zeros(0, int)
            runs = [(int(a), int(b), int(li[a])) for a, b in zip(starts, stops)]
            self._runs = runs
        return runs

    def pose_vectors(self, q, V) -> np.ndarray:
        """Directions given per triangle in their link's frame (V: (M, ..., 3), e.g. per-corner normals (M, 3, 3))
        rotated into the world at q: (M, ..., 3) float32."""
        V = np.asarray(V)
        R = self.fk_array(self._q(q))[:, :3, :3]
        out = np.empty(V.shape, np.float32)
        for a, b, li in self._link_runs():
            out[a:b] = (V[a:b].reshape(-1, 3) @ R[li].T).reshape(V[a:b].shape)
        return out

    def posed_triangles(self, q):
        """(tri (M,3,3) float32 world, normals (M,3) float32, link index (M,) int32) of the visual meshes at q."""
        q = self._q(q)
        with self._posed_lock:
            c = self._posed_cache
            if c is not None and c[0].shape == q.shape and np.array_equal(c[0], q):
                return c[1]
        T = self.fk_array(q)
        # one matrix product per run of a link's triangles (BLAS), not a 3 × 3 product per triangle
        tri = np.empty(self._tri_local.shape, np.float32)
        for a, b, li in self._link_runs():
            tri[a:b] = (self._tri_local[a:b].reshape(-1, 3) @ T[li, :3, :3].T + T[li, :3, 3]).reshape(-1, 3, 3)
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        n = np.where(ln > 1e-20, n / np.maximum(ln, 1e-30), np.array([0.0, 0.0, 1.0], np.float32)).astype(np.float32)
        out = (tri, n, self._tri_link)
        with self._posed_lock:
            self._posed_cache = (q.copy(), out)
        return out

    def bounds(self, q=None) -> tuple[np.ndarray, np.ndarray]:
        """World-frame axis-aligned bounds of the visual meshes (or the link origins when there are none) at q."""
        q = self._q(self.home if q is None else q)
        with self._posed_lock:
            c = self._bounds_cache
            if c is not None and c[0].shape == q.shape and np.array_equal(c[0], q):
                return c[1].copy(), c[2].copy()
        lo, hi = self._bounds(q)
        with self._posed_lock:
            self._bounds_cache = (q.copy(), lo, hi)
        return lo.copy(), hi.copy()

    def _bounds(self, q):
        if self.n_triangles:
            tri = self.posed_triangles(q)[0].reshape(-1, 3)
        else:
            tri = self.fk_array(q)[:, :3, 3]
        if not len(tri):
            tri = self.base[None, :3, 3]
        return tri.min(axis=0).astype(float), tri.max(axis=0).astype(float)


# ====================================================================== the shared cache
_LOCK = threading.RLock()
_COMPILED: OrderedDict = OrderedDict()
_MODELS: OrderedDict = OrderedDict()
_DIGESTS: dict = {}               # id(bytes) -> (the bytes, their SHA-256): bytes never change in place
MAX_COMPILED = 8
MAX_MODELS = 12                   # a RobotModel of the UR5e sample holds about 35 MB


DIGEST_BYTES = 256 * 2 ** 20      # the digest cache keeps the bytes it hashed alive: at most this many in all
_DIGEST_HELD = [0]


def _digest(b: bytes) -> str:
    import hashlib
    e = _DIGESTS.get(id(b))
    if e is not None and e[0] is b:
        return e[1]
    if len(_DIGESTS) > 4096 or _DIGEST_HELD[0] + len(b) > DIGEST_BYTES:
        _DIGESTS.clear()
        _DIGEST_HELD[0] = 0
    h = hashlib.sha256(b).hexdigest()
    _DIGESTS[id(b)] = (b, h)
    _DIGEST_HELD[0] += len(b)
    return h


def _files_key(robot, sources) -> tuple:
    """The robot's files by content: two robots read from the same files (two openings of a sample) share one
    compiled model, whatever their uids."""
    pre = robot.uid + '/'
    f = robot.source.file
    return (robot.source.kind, f[len(pre):] if f.startswith(pre) else f,
            tuple((k[len(pre):], _digest(v)) for k, v in sorted(sources.items()) if k.startswith(pre)))


def get_model(project, robot_uid: str) -> RobotModel:
    """The project's robot robot_uid compiled, shared between threads; raises RobotModelError when it cannot be."""
    robot = project.robot(robot_uid)
    if robot is None:
        raise RobotModelError(f'There is no robot {robot_uid} in the project.')
    with _LOCK:
        fkey = _files_key(robot, project.sources)
        mkey = (fkey, json.dumps(robot.to_dict(), sort_keys=True, default=str))
        mod = _MODELS.get(mkey)
        if mod is not None:
            _MODELS.move_to_end(mkey)
            return mod
        comp = _COMPILED.get(fkey)
        if comp is None:
            comp = compile_robot(robot, project.sources)
            _COMPILED[fkey] = comp
            while len(_COMPILED) > MAX_COMPILED:
                _COMPILED.popitem(last=False)
        else:
            _COMPILED.move_to_end(fkey)
        mod = RobotModel(robot, project.sources, comp)
        _MODELS[mkey] = mod
        while len(_MODELS) > MAX_MODELS:
            _MODELS.popitem(last=False)
        return mod


def clear_cache():
    with _LOCK:
        _COMPILED.clear()
        _MODELS.clear()
        _DIGESTS.clear()
        _DIGEST_HELD[0] = 0
