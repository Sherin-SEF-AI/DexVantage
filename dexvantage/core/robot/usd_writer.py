"""A small UsdPhysics writer: a project robot (its compiled MuJoCo model, core/robot/model.py) as a USD articulation,
for the USD import's round trip (validation/checks_robot_io.py) and for handing a robot to a USD tool.

    write_robot_usd(project, robot_uid, path, meshes=True) -> dict     the stage written with pxr (usd-core)
    usd_name(text) -> str                                              the prim name a body or joint gets

MuJoCo 3.17's own USD exporter (mujoco.usd.exporter.USDExporter) writes the scene for external renderers from
mujoco.Renderer frames: meshes and their animated transforms, no UsdPhysics bodies, joints or limits (its module
docstring and source, read 2026-11-07), and it needs an OpenGL context; so it cannot stand in for an articulation.

The stage (metersPerUnit 1, kilogramsPerUnit 1, upAxis Z, defaultPrim the robot):

    /<robot>                      Xform, PhysicsArticulationRootAPI
    /<robot>/<body>               one Xform per MuJoCo body (flat, as Isaac Lab's URDF importer lays links out),
                                  PhysicsRigidBodyAPI and PhysicsMassAPI (mass, centerOfMass = body_ipos,
                                  diagonalInertia = body_inertia, principalAxes = body_iquat), its world pose at the
                                  reference state (every hinge and slide at qpos0, free and ball joints at home) as one
                                  xformOp:transform
    /<robot>/<body>/visuals/<g>   visual geoms; collisions/<g> colliding geoms with PhysicsCollisionAPI: meshes as
                                  UsdGeom.Mesh (MuJoCo's compiled vertices and faces), box as Cube (size 2, scaled to
                                  the half sizes), sphere, ellipsoid (a scaled Sphere), capsule and cylinder (axis Z,
                                  height twice MuJoCo's half length); planes and height fields are left out
    /<robot>/joints/<joint>       one joint per body: PhysicsRevoluteJoint or PhysicsPrismaticJoint with axis X and the
                                  joint frames below, PhysicsFixedJoint for a body without hinge or slide (and one
                                  from the world to each root body)

Joint frames. MuJoCo moves a body with a hinge or slide j (anchor p, axis a in the body, reference qpos0) as X_p · T0 ·
Trans(p) · Rot(a, q − qpos0) · Trans(−p), or X_p · T0 · Trans(a (q − qpos0)) for a slide; UsdPhysics requires
X_body0 · F0 · M(q) = X_body1 · F1 with M about or along the joint frame's X. With R_a a rotation taking X to a, F1 =
Trans(p) · R_a and F0 = T0 · Trans(p) · Rot(a, −qpos0) · R_a (a slide: F0 = T0 · Trans(−a qpos0) · Trans(p) · R_a)
give M(q) = Rot(X, q): the USD joint's position is DexVantage's q. A body with several hinges and slides gets one
massless intermediate body '<body>__<joint>' per extra joint. Limits: radians to degrees for revolute joints, metres
for prismatic; a joint without finite limits keeps the schema's −inf / inf. Positions and rotations are float32
in the schema (point3f, quatf): about 10⁻⁷ relative, which sets the round trip's tolerance.

Known limits: one robot per stage; equality constraints, actuators, tendons and materials are not written (each geom
gets its display colour); a sensor or camera is not written; free and ball joints are fixed at home.
"""
from __future__ import annotations

import math
import re

import numpy as np

from ..frames import quat_from_matrix, quat_from_mujoco, quat_to_matrix


def usd_name(text: str) -> str:
    """A USD prim name (an identifier: letters, digits, underscores, not starting with a digit)."""
    s = re.sub(r'[^A-Za-z0-9_]+', '_', str(text or '')).strip('_') or 'x'
    return s if not s[0].isdigit() else '_' + s


def _T(R=None, t=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def _axis_angle(a, ang) -> np.ndarray:
    a = np.asarray(a, float)
    a = a / (np.linalg.norm(a) or 1.0)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def x_to(a) -> np.ndarray:
    """A rotation taking +X to the unit vector a (the shortest turn; a half turn about z for −X)."""
    a = np.asarray(a, float)
    a = a / (np.linalg.norm(a) or 1.0)
    x = np.array([1.0, 0.0, 0.0])
    c = float(x @ a)
    if c > 1 - 1e-15:
        return np.eye(3)
    if c < -1 + 1e-15:
        return np.diag([-1.0, -1.0, 1.0])
    v = np.cross(x, a)
    return _axis_angle(v, math.atan2(np.linalg.norm(v), c))


def _quatf(R):
    from pxr import Gf
    x, y, z, w = quat_from_matrix(R)
    return Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))


def _mat(T):
    from pxr import Gf
    return Gf.Matrix4d(np.asarray(T, float).T.tolist())


def _geom(stage, path: str, m, g: int, collide: bool, meshes: bool):
    """Define one geom prim; False when its type is not written."""
    import mujoco
    from pxr import Gf, UsdGeom, UsdPhysics, Vt
    G = mujoco.mjtGeom
    t = int(m.geom_type[g])
    s = np.asarray(m.geom_size[g], float)
    T = _T(quat_to_matrix(quat_from_mujoco(m.geom_quat[g])), m.geom_pos[g])
    scale = None
    if t == int(G.mjGEOM_MESH):
        if not meshes:
            return False
        k = int(m.geom_dataid[g])
        va, vn = m.mesh_vertadr[k], m.mesh_vertnum[k]
        fa, fn = m.mesh_faceadr[k], m.mesh_facenum[k]
        prim = UsdGeom.Mesh.Define(stage, path)
        prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(m.mesh_vert[va:va + vn], np.float32)))
        prim.CreateFaceVertexCountsAttr(Vt.IntArray([3] * int(fn)))
        prim.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(
            np.ascontiguousarray(m.mesh_face[fa:fa + fn].reshape(-1), np.int32)))
        prim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    elif t == int(G.mjGEOM_BOX):
        prim = UsdGeom.Cube.Define(stage, path)
        prim.CreateSizeAttr(2.0)
        scale = s[:3]
    elif t == int(G.mjGEOM_SPHERE):
        prim = UsdGeom.Sphere.Define(stage, path)
        prim.CreateRadiusAttr(float(s[0]))
    elif t == int(G.mjGEOM_ELLIPSOID):
        prim = UsdGeom.Sphere.Define(stage, path)
        prim.CreateRadiusAttr(1.0)
        scale = s[:3]
    elif t in (int(G.mjGEOM_CAPSULE), int(G.mjGEOM_CYLINDER)):
        cap = t == int(G.mjGEOM_CAPSULE)
        prim = (UsdGeom.Capsule if cap else UsdGeom.Cylinder).Define(stage, path)
        prim.CreateRadiusAttr(float(s[0]))
        prim.CreateHeightAttr(float(2 * s[1]))
        prim.CreateAxisAttr(UsdGeom.Tokens.z)
    else:
        return False
    prim.AddTransformOp().Set(_mat(T))
    if scale is not None:
        prim.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in scale]))
    rgba = np.asarray(m.geom_rgba[g], float)
    prim.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*[float(v) for v in rgba[:3]])]))
    prim.CreateDisplayOpacityAttr(Vt.FloatArray([float(rgba[3])]))
    if collide:
        UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    return True


def write_robot_usd(project, robot_uid: str, path: str, meshes: bool = True) -> dict:
    """Write the robot as a UsdPhysics articulation at path (see the module docstring): {'root', 'bodies' {MuJoCo
    body name: prim name}, 'joints' {joint name: prim name}, 'geoms', 'skipped'}."""
    import mujoco
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from .model import get_model
    rm = get_model(project, robot_uid)
    m = rm.m
    J = mujoco.mjtJoint
    d = mujoco.MjData(m)
    d.qpos[:] = rm._q0
    for j in rm._jid:
        d.qpos[m.jnt_qposadr[j]] = m.qpos0[m.jnt_qposadr[j]]
    mujoco.mj_kinematics(m, d)

    def X(b):
        return _T(d.xmat[b].reshape(3, 3), d.xpos[b])

    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = f'/{usd_name(rm.name)}'
    rp = UsdGeom.Xform.Define(stage, root).GetPrim()
    stage.SetDefaultPrim(rp)
    UsdPhysics.ArticulationRootAPI.Apply(rp)
    UsdGeom.Scope.Define(stage, f'{root}/joints')
    used, bodies, joints = set(), {}, {}

    def unique(n):
        k, i = n, 2
        while k in used:
            k, i = f'{n}_{i}', i + 1
        used.add(k)
        return k

    def body_prim(name, T):
        p = UsdGeom.Xform.Define(stage, f'{root}/{name}')
        p.AddTransformOp().Set(_mat(T))
        UsdPhysics.RigidBodyAPI.Apply(p.GetPrim())
        return p.GetPrim()

    def joint(kind, name, b0, b1, F0, F1, lo=None, hi=None):
        jp = f'{root}/joints/{unique(name)}'
        if kind == 'revolute':
            jj = UsdPhysics.RevoluteJoint.Define(stage, jp)
        elif kind == 'prismatic':
            jj = UsdPhysics.PrismaticJoint.Define(stage, jp)
        else:
            jj = UsdPhysics.FixedJoint.Define(stage, jp)
        if kind != 'fixed':
            jj.CreateAxisAttr(UsdPhysics.Tokens.x)
            if lo is not None and hi is not None:
                f = 180.0 / math.pi if kind == 'revolute' else 1.0
                jj.CreateLowerLimitAttr(float(lo * f))
                jj.CreateUpperLimitAttr(float(hi * f))
        if b0 is not None:
            jj.CreateBody0Rel().SetTargets([f'{root}/{b0}'])
        jj.CreateBody1Rel().SetTargets([f'{root}/{b1}'])
        jj.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in F0[:3, 3]]))
        jj.CreateLocalRot0Attr(_quatf(F0[:3, :3]))
        jj.CreateLocalPos1Attr(Gf.Vec3f(*[float(v) for v in F1[:3, 3]]))
        jj.CreateLocalRot1Attr(_quatf(F1[:3, :3]))
        return jp.rsplit('/', 1)[1]

    nonc = (m.geom_contype == 0) & (m.geom_conaffinity == 0)
    any_visual = bool(nonc.any())
    n_geom, skipped = 0, set()
    for b in range(1, m.nbody):
        bn = unique(usd_name(rm.body_link[b]))
        bodies[rm.body_link[b]] = bn
        bp = body_prim(bn, X(b))
        mapi = UsdPhysics.MassAPI.Apply(bp)
        mapi.CreateMassAttr(float(m.body_mass[b]))
        mapi.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in m.body_ipos[b]]))
        mapi.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in m.body_inertia[b]]))
        mapi.CreatePrincipalAxesAttr(_quatf(quat_to_matrix(quat_from_mujoco(m.body_iquat[b]))))
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) != b:
                continue
            collide = (not nonc[g]) if any_visual else True
            folder = 'collisions' if collide else 'visuals'
            if not _geom(stage, f'{root}/{bn}/{folder}/{usd_name(m.geom(g).name or f"geom_{g}")}_{g}', m, g,
                         collide, meshes):
                skipped.add(mujoco.mjtGeom(int(m.geom_type[g])).name[len('mjGEOM_'):].lower())
            else:
                n_geom += 1
        pb = int(m.body_parentid[b])
        parent = bodies[rm.body_link[pb]] if pb > 0 else None
        T0 = _T(quat_to_matrix(quat_from_mujoco(m.body_quat[b])), m.body_pos[b])
        jids = [int(m.body_jntadr[b]) + k for k in range(int(m.body_jntnum[b]))] if m.body_jntnum[b] > 0 else []
        moving = [j for j in jids if int(m.jnt_type[j]) in (int(J.mjJNT_HINGE), int(J.mjJNT_SLIDE))]
        if len(moving) != len(jids) or not moving:
            # a fixed body, or free / ball joints held at home: welded at the reference pose
            T0 = np.linalg.inv(X(pb)) @ X(b) if pb > 0 else X(b)
            joint('fixed', f'{bn}_fixed', parent, bn, T0, np.eye(4))
            continue
        pend, cur_parent = T0, parent
        for k, j in enumerate(moving):
            p = np.asarray(m.jnt_pos[j], float)
            a = np.asarray(m.jnt_axis[j], float)
            a = a / (np.linalg.norm(a) or 1.0)
            q0 = float(m.qpos0[m.jnt_qposadr[j]])
            Ra = x_to(a)
            hinge = int(m.jnt_type[j]) == int(J.mjJNT_HINGE)
            F1 = _T(Ra, p)
            F0 = pend @ (_T(_axis_angle(a, -q0), p) if hinge else _T(None, p - a * q0)) @ _T(Ra)
            i = list(rm._jid).index(j)
            lo, hi = rm.lower[i], rm.upper[i]
            lim = (lo, hi) if np.isfinite(lo) and np.isfinite(hi) else (None, None)
            last = k == len(moving) - 1
            child = bn
            if not last:                               # an intermediate body carries this joint
                child = unique(f'{bn}__{usd_name(rm.joint_name(j))}')
                body_prim(child, X(b))
            joints[rm.joint_name(j)] = joint('revolute' if hinge else 'prismatic', usd_name(rm.joint_name(j)),
                                             cur_parent, child, F0, F1, *lim)
            # MuJoCo composes a body's joints in order, X_b = X_p T0 M1(q1) M2(q2) ..., each M about the body's own
            # frame: the intermediate body sits at X_p T0 M1(q1), so the next joint's T0 is the identity
            pend, cur_parent = np.eye(4), child
    stage.GetRootLayer().Save()
    return {'root': root, 'bodies': bodies, 'joints': joints, 'geoms': n_geom, 'skipped': sorted(skipped)}


__all__ = ['write_robot_usd', 'usd_name', 'x_to']
