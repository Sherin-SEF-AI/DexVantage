"""Check of the project-obstacle distances (core/robot/collision.py obstacle_distance and env_distance) against closed
forms written out here.

A scripted robot (a static base and, three slide joints further, a sphere of radius 0.05 m) stands at a base pose
rotated and moved away from the world origin (so Robot.base must be applied). The project holds one obstacle of each
kind collision.py makes: a rotated box object, a sphere object, a tilted cylinder object, a mesh object (the eight
corners of a box, which MuJoCo collides as their convex hull: the box itself) and a table surface with a thickness.
At random slide positions (some overlapping an obstacle) the smallest signed distance from the robot's sphere to the
obstacles is computed in closed form: the signed distance of the sphere's centre to each solid, minus its radius
(exact for a sphere against a convex solid, inside and out), and compared with env_distance; the closest obstacle's
name must match whenever the nearest two differ by more than 1 mm.

Closed forms (a point p in the solid's own frame): box of half-edges h: d = |p| − h, sd = ‖max(d, 0)‖ + min(max_i d_i,
0); cylinder of radius a and half-height h along z: dr = ‖p_xy‖ − a, dz = |p_z| − h, sd = ‖(max(dr, 0), max(dz, 0))‖ +
min(max(dr, dz), 0); sphere of radius R: ‖p‖ − R.
"""
from __future__ import annotations

import os

import numpy as np

from .suite import Outcome, check, figure, series

R_ROBOT = 0.05
C_OFF = np.array([0.05, -0.02, 0.10])     # the sphere's centre in the last slide body, m


def sd_box(p, half):
    d = np.abs(p) - half
    return np.linalg.norm(np.maximum(d, 0.0), axis=-1) + np.minimum(d.max(axis=-1), 0.0)


def sd_cylinder(p, a, h):
    dr = np.linalg.norm(p[..., :2], axis=-1) - a
    dz = np.abs(p[..., 2]) - h
    return np.hypot(np.maximum(dr, 0.0), np.maximum(dz, 0.0)) + np.minimum(np.maximum(dr, dz), 0.0)


def _rot(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def obstacle_project(folder: str):
    """(project, robot uid, closed-form fn(x world (N,3)) -> (N, k) signed distances of the centre, names)."""
    from ..core.frames import Pose, quat_from_matrix
    from ..core.project import Project, TaskObject
    from .checks_ik import model_from_mjcf, xyz_pair_mjcf
    a = '<geom type="sphere" size="0.02" pos="0 0 0"/>'
    b = f'<geom type="sphere" size="{R_ROBOT}" pos="{C_OFF[0]} {C_OFF[1]} {C_OFF[2]}"/>'
    _m, robot, sources = model_from_mjcf(xyz_pair_mjcf(a, b), folder)
    Rb = _rot([0.2, 0.1, 1.0], 0.6)
    robot.base = Pose(np.array([0.3, -0.2, 0.15]), quat_from_matrix(Rb))
    p = Project('Obstacle distance check')
    p.robots, p.sources = [robot], dict(sources)
    R1, R2 = _rot([1, 2, 0.5], 0.7), _rot([0.3, 1, 0], 0.9)
    box = TaskObject(name='box', shape='box', size=(0.20, 0.10, 0.14),
                     pose=Pose(np.array([0.55, -0.15, 0.30]), quat_from_matrix(R1)))
    sph = TaskObject(name='ball', shape='sphere', size=(0.06, 0, 0), pose=Pose(np.array([0.25, 0.10, 0.40])))
    cyl = TaskObject(name='can', shape='cylinder', size=(0.04, 0.16, 0),
                     pose=Pose(np.array([0.35, -0.45, 0.30]), quat_from_matrix(R2)))
    mh = np.array([0.05, 0.08, 0.06])
    mc = np.array([0.10, -0.25, 0.45])
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * mh
    p.meshes['crate'] = corners
    p.meshes['crate/faces'] = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                                        [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], np.int32)
    mesh = TaskObject(name='crate', shape='mesh', size=(0, 0, 0), pose=Pose(mc), mesh_key='crate')
    p.workspace.objects = [box, sph, cyl, mesh]
    p.workspace.surfaces = [{'uid': 'table', 'name': 'table', 'z': 0.12, 'size': [1.0, 0.8], 'center': [0.3, -0.1],
                             'thickness': 0.04}]
    names = ['box', 'ball', 'can', 'crate', 'table']

    def ref(X):
        X = np.asarray(X, float)
        return np.stack([
            sd_box((X - box.pose.position) @ R1, np.array(box.size) / 2),
            np.linalg.norm(X - sph.pose.position, axis=1) - 0.06,
            sd_cylinder((X - cyl.pose.position) @ R2, 0.04, 0.08),
            sd_box(X - mc, mh),
            sd_box(X - np.array([0.3, -0.1, 0.10]), np.array([0.5, 0.4, 0.02])),
        ], axis=1)
    return p, robot.uid, ref, names


@check('kinematics.obstacle_distance', 'kinematics', 'Robot-to-obstacle distance (project objects and surfaces)',
       'largest difference between env_distance (a sphere on a robot with a moved and rotated base, against a box, '
       'sphere, cylinder, convex mesh and table of the project) and the closed-form signed distance, over random '
       'joint positions, overlapping ones included; the nearest obstacle\'s name must match',
       'the signed distance of a point to a box, sphere and cylinder, written out in the check, minus the sphere\'s '
       'radius', 1e-6, 'm', refs=('mujoco_geomdistance', 'ericson2005'))
def _obstacles(ctx):
    from ..core.robot import collision as C
    p, uid, ref, names = obstacle_project(os.path.join(ctx.tmp(), 'kinematics.obstacle_distance'))
    rng = ctx.rng('kinematics.obstacle_distance')
    n = 100 if ctx.quick else 600
    Q = rng.uniform(-0.35, 0.35, (n, 3))
    B = p.robots[0].base.matrix()
    X = (C_OFF + Q) @ B[:3, :3].T + B[:3, 3]
    S = ref(X) - R_ROBOT
    want = np.minimum(S.min(axis=1), 1.0)
    got, wrong_name, flag = np.empty(n), 0, 0
    for k, q in enumerate(Q):
        d, pair = C.env_distance(p, uid, q, distmax=1.0)
        got[k] = d
        srt = np.sort(S[k])
        if pair is not None and srt[1] - srt[0] > 1e-3 and pair[1] != names[int(np.argmin(S[k]))]:
            wrong_name += 1
        if C.obstacle_distance(p, uid, q, distmax=1.0).in_collision != (want[k] < 0):
            flag += 1
    err = np.abs(got - want)
    return Outcome(float(err.max()), {'placements': n, 'overlapping': int((want < 0).sum()),
                                      'wrong_nearest_obstacle': wrong_name, 'flag_mismatches': flag,
                                      'median_m': float(np.median(err)), 'obstacles': names},
                   ok=wrong_name == 0 and flag == 0,
                   figure=figure('Distance to the nearest obstacle', 'closed form (m)', 'DexVantage (m)',
                                 [series('placements', want, got, 'points')], diag=True),
                   note='MuJoCo measures sphere–box and sphere–sphere distances in closed form and sphere–cylinder '
                        'and sphere–mesh through its convex solver (GJK, with EPA for penetration), whose '
                        'convergence tolerance is far below a micrometre at these sizes; 1e-6 m is the solver\'s '
                        'margin, while a wrong frame (the base pose, a quaternion order, a half-size taken for a '
                        'full size) shows as centimetres.')
