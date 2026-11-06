"""Checks of the grasp sets (core/grasps.py): the contact patches against the pad boxes MuJoCo itself holds, the
measured tool centre against the model file's own pinch site, the samplers against their closed forms, and the object
surface points against a box's closed-form signed distance.

- grasp.pads_vs_mujoco_boxes: on the UR5e sample's Robotiq 2F-85, at random arm and finger joint values, every contact
  point must lie on the inner face of one of its pad's box geoms. The faces are built here from MuJoCo's compiled
  geom data (geom_xpos, geom_xmat and geom_size after mj_kinematics on a fresh MjData), not from the tessellated
  meshes, link transforms or face-selection rule the module uses: the error is a point's distance to the nearest
  inner-face rectangle (the plane distance and the distance outside the rectangle).
- grasp.tcp_vs_pinch_site: the TCP and approach axis measured from the 2F-85's pads against the "pinch" site the
  Menagerie file places on the gripper's axis (gripper_base frame, 145 mm out).
- grasp.sampler_closed_form: sampled grasps on the UR5e sample (no objects, a table at z = 0): the TCP read back
  through SciPy's Rotation (an independent quaternion implementation) is height_m above the table and inside the
  volume's footprint, the approach is the kind's closed form (straight down; horizontal; 45° from vertical), side and
  angled headings are within ±spread_deg of the line from the base, and the closing axis is horizontal and
  perpendicular to the approach.
- grasp.object_points_box: points on a rotated box object against its closed-form signed distance, and the share of
  points on each face against the face's share of the area (binomial, 4σ).
- grasp.trajectory_closures: grasps found in a scripted pick trajectory on the UR5e sample (from_trajectory mode
  'closures'): the gripper's opening against the distance between the 2F-85 pads' inner faces built from MuJoCo's
  compiled geoms, and the closing strokes found against the sample ranges the script closes the fingers over.
"""
from __future__ import annotations

import math

import numpy as np

from .suite import Outcome, check, figure, series


def _ur5e(ctx):
    p = ctx.sample()
    r = next(r for r in p.robots if 'gripper_right_pad' in r.fingertip_frames)
    return p, r


def _inner_faces_mujoco(model, q, pad_bodies, pick=None):
    """({body name: [(centre, normal, u, v, hu, hv)]}, pick): the box geoms of each pad body, from MuJoCo's own data.
    Without pick, the face of each box pointing at the other body's geoms (pick records which: geom axis and sign);
    with pick (from the home pose), those same faces carried to q."""
    import mujoco
    m = model.m
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    for name, val in zip(model.joint_names, q):
        j = m.joint(name).id
        d.qpos[m.jnt_qposadr[j]] = val
    mujoco.mj_kinematics(m, d)
    B = model.base
    boxes = {}
    for b in pad_bodies:
        bid = m.body(b).id
        boxes[b] = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != bid or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
                continue
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                continue
            R = B[:3, :3] @ d.geom_xmat[g].reshape(3, 3)
            c = B[:3, :3] @ d.geom_xpos[g] + B[:3, 3]
            boxes[b].append((c, R, np.asarray(m.geom_size[g], float)))
    out, chosen = {}, {}
    for b, lst in boxes.items():
        other = np.mean([c for o, l2 in boxes.items() if o != b for c, _R, _s in l2], axis=0)
        faces = []
        for i, (c, R, h) in enumerate(lst):
            if pick is None:
                dirn = other - c
                k = int(np.argmax(np.abs(R.T @ dirn)))
                s = math.copysign(1.0, float(R[:, k] @ dirn))
            else:
                k, s = pick[(b, i)]
            chosen[(b, i)] = (k, s)
            u, v = [j for j in range(3) if j != k]
            faces.append((c + s * h[k] * R[:, k], s * R[:, k], R[:, u], R[:, v], h[u], h[v]))
        out[b] = faces
    return out, chosen


def _face_distance(P, face):
    fc, n, u, v, hu, hv = face
    rel = P - fc
    du = np.maximum(np.abs(rel @ u) - hu, 0.0)
    dv = np.maximum(np.abs(rel @ v) - hv, 0.0)
    return np.sqrt((rel @ n) ** 2 + du ** 2 + dv ** 2)


@check('grasp.pads_vs_mujoco_boxes', 'grasp', 'Contact pads of the Robotiq 2F-85 (UR5e sample)',
       'largest distance of a contact point (contact_points at random arm and finger joint values) from the inner '
       'face of its pad\'s box geoms (the faces facing each other at home, carried with the pads)', 'the box geoms\' '
       'inner faces built from MuJoCo\'s compiled geom poses and sizes (mj_kinematics on a fresh MjData)', 1e-6, 'm',
       refs=('mujoco_todorov2012', 'mujoco_xml'))
def _pads(ctx):
    from ..core import grasps as GR
    from ..core.robot.model import get_model
    p, r = _ur5e(ctx)
    m = get_model(p, r.uid)
    rng = ctx.rng('grasp.pads_vs_mujoco_boxes')
    n = 10 if ctx.quick else 40
    lo = np.where(np.isfinite(m.lower), m.lower, -np.pi)
    hi = np.where(np.isfinite(m.upper), m.upper, np.pi)
    worst, per, facing = 0.0, [], 0
    _f, pick = _inner_faces_mujoco(m, m.home, r.fingertip_frames)       # the faces facing each other at home
    for _k in range(n):
        q = lo + (hi - lo) * rng.random(m.nq)
        patches = GR.contact_patches(p, r.uid, q)
        faces, _p = _inner_faces_mujoco(m, q, r.fingertip_frames, pick)
        for name, pt in patches.items():
            D = np.min([_face_distance(pt['points'], f) for f in faces[pt['link']]], axis=0)
            worst = max(worst, float(D.max()))
            per.append(float(D.max()))
            n_ref = faces[pt['link']][0][1]
            facing += int(np.all(pt['normals'] @ n_ref > 0.999))
    total = n * len(r.fingertip_frames)
    return Outcome(worst, {'joint_vectors': n, 'pads': list(r.fingertip_frames), 'points_per_pad': GR.PAD_POINTS,
                           'normals_matching_the_inner_face': f'{facing} of {total}'},
                   ok=facing == total,
                   figure=figure('Contact points off the pad\'s inner face', 'pad at a joint vector', 'worst (m)',
                                 [series('worst per pad', np.arange(len(per)), per, 'points')], hline=1e-6),
                   note='The meshes are stored as float32 (about 3e-8 m at half a metre) and the transforms are '
                        'double; 1e-6 m leaves room for that, while picking the wrong face (the outer one, 8 mm '
                        'away) or the wrong link frame shows as millimetres.')


@check('grasp.tcp_vs_pinch_site', 'grasp', 'Tool centre of the Robotiq 2F-85 against the model\'s pinch site',
       'angle between the approach axis measured from the pads and the line from the gripper frame to the file\'s '
       '"pinch" site; the TCP must lie within 1 mm of that line',
       'the gripper_pinch site of MuJoCo Menagerie\'s robotiq_2f85 (on the gripper\'s axis of symmetry)', 0.5, 'deg',
       refs=('mujoco_xml',))
def _tcp(ctx):
    from ..core import grasps as GR
    from ..core.robot.model import get_model
    p, r = _ur5e(ctx)
    m = get_model(p, r.uid)
    g = GR.gripper_geometry(p, r.uid)
    sid = m.m.site('gripper_pinch').id
    body = m.body_link[int(m.m.site_bodyid[sid])]
    T = m.fk(m.home)
    pin_w = T[body][:3, :3] @ m.m.site_pos[sid] + T[body][:3, 3]
    Ti = np.linalg.inv(T[g['frame']])
    pin = Ti[:3, :3] @ pin_w + Ti[:3, 3]
    axis = pin / np.linalg.norm(pin)
    ang = math.degrees(math.acos(min(1.0, float(abs(axis @ g['approach'])))))
    lateral = float(np.linalg.norm(g['tcp'] - axis * float(g['tcp'] @ axis)))
    return Outcome(ang, {'frame': g['frame'], 'tcp_m': [float(v) for v in g['tcp']],
                         'pinch_site_m': [float(v) for v in pin], 'tcp_off_axis_m': lateral,
                         'tcp_to_pinch_along_axis_m': float((pin - g['tcp']) @ axis)},
                   ok=lateral <= 1e-3,
                   note='The pads are mirror images about the axis on which the file puts the pinch site, so the '
                        'measured axis should match it to rounding; 0.5° is 1.3 mm over the 145 mm to the site. The '
                        'TCP is the pads\' centre, 14 mm short of the pinch site (which sits at the pads\' tips).')


@check('grasp.sampler_closed_form', 'grasp', 'Grasp samplers against their closed forms',
       'largest deviation of a sampled grasp from its kind\'s closed form (TCP height above the table and footprint, '
       'approach direction, heading spread, closing axis), the pose read back with SciPy\'s Rotation',
       'the samplers\' definitions written out again; SciPy\'s quaternion-to-matrix conversion', 1e-9,
       'm or unit vector', refs=('scipy_rotation', 'rep103'))
def _sampler(ctx):
    from scipy.spatial.transform import Rotation

    from ..core import grasps as GR
    p, r = _ur5e(ctx)
    p.workspace.objects = []
    vol = next(v for v in p.workspace.volumes if v.kind != 'handover')
    n = 300 if ctx.quick else 1500
    gs = GR.sample(p, r.uid, ['top_down', 'side', 'angled'], n, int(ctx.rng('grasp.sampler_closed_form')
                                                                    .integers(0, 2 ** 31 - 1)), volume_uid=vol.uid)
    g = GR.gripper_geometry(p, r.uid)
    lo, hi = (np.asarray(x, float) for x in vol.bounds())
    table = max([s['z'] for s in p.workspace.surfaces] + [lo[2]])
    st = GR.DEFAULTS
    errs = {'tcp_height_m': 0.0, 'outside_footprint_m': 0.0, 'approach': 0.0, 'heading_beyond_spread_rad': 0.0,
            'closing': 0.0, 'stored_approach': 0.0}
    base = np.asarray(r.base.position, float)
    for x in gs['poses']:
        R = Rotation.from_quat(x['pose']['quaternion']).as_matrix()      # SciPy: (x, y, z, w), scalar last
        t = R @ g['tcp'] + np.asarray(x['pose']['position'])
        a = R @ g['approach']
        c = R @ g['closing']
        errs['tcp_height_m'] = max(errs['tcp_height_m'], abs(t[2] - (table + st['height_m'])))
        errs['outside_footprint_m'] = max(errs['outside_footprint_m'],
                                          float(np.maximum(np.maximum(lo[:2] - t[:2], t[:2] - hi[:2]), 0).max()))
        errs['stored_approach'] = max(errs['stored_approach'], float(np.abs(a - x['approach']).max()))
        if x['kind'] == 'top_down':
            want = np.array([0.0, 0.0, -1.0])
        else:
            az = math.atan2(a[1], a[0])
            az0 = math.atan2(t[1] - base[1], t[0] - base[0])
            dev = abs((az - az0 + math.pi) % (2 * math.pi) - math.pi)
            errs['heading_beyond_spread_rad'] = max(errs['heading_beyond_spread_rad'],
                                                    max(0.0, dev - math.radians(st['spread_deg'])))
            th = 0.5 * math.pi if x['kind'] == 'side' else math.radians(st['angled_tilt_deg'])
            want = np.array([math.cos(az) * math.sin(th), math.sin(az) * math.sin(th), -math.cos(th)])
        errs['approach'] = max(errs['approach'], float(np.abs(a - want).max()))
        errs['closing'] = max(errs['closing'], abs(float(c[2])), abs(float(c @ a)), abs(float(np.linalg.norm(c)) - 1))
    return Outcome(max(errs.values()), dict(errs, grasps=n, kinds=GR.counts(gs)),
                   note='Every quantity is a closed form of the sampler\'s definition; the only rounding is double '
                        'precision through a quaternion (about 1e-15), so 1e-9 catches any frame or sign mistake.')


@check('grasp.object_points_box', 'grasp', 'Points on a task object\'s surface',
       'largest |signed distance| of object_points on a rotated box from the box\'s surface; each face\'s share of '
       'the points must be within 4σ (binomial) of its share of the area',
       'the closed-form signed distance of a box, written out in the check', 1e-9, 'm')
def _object_points(ctx):
    from ..core import grasps as GR
    from ..core.frames import Pose, quat_from_matrix
    from ..core.project import Project, TaskObject
    from .checks_obstacles import _rot, sd_box
    R = _rot([0.3, -0.5, 1.0], 0.8)
    size = np.array([0.08, 0.05, 0.12])
    o = TaskObject(name='box', shape='box', size=tuple(size), pose=Pose(np.array([0.4, -0.1, 0.06]),
                                                                         quat_from_matrix(R)))
    p = Project('object points check')
    p.workspace.objects = [o]
    n = 4000 if ctx.quick else 40000
    P = GR.object_points(p, o.uid, n, int(ctx.rng('grasp.object_points_box').integers(0, 2 ** 31 - 1)))
    L = (P - o.pose.position) @ R
    sd = sd_box(L, size / 2)
    face = np.argmax(np.abs(L) / (size / 2), axis=1)
    areas = np.array([size[1] * size[2], size[0] * size[2], size[0] * size[1]])
    share = areas / areas.sum()
    got = np.bincount(face, minlength=3) / n
    z = np.abs(got - share) / np.sqrt(share * (1 - share) / n)
    return Outcome(float(np.abs(sd).max()), {'points': n, 'face_share': got.tolist(), 'area_share': share.tolist(),
                                            'worst_z': float(z.max())}, ok=bool(z.max() <= 4.0),
                   note='Points are drawn on the box\'s own triangles, so their distance to the surface is rounding '
                        '(1e-17 m); a wrong pose or size shows as centimetres. 4σ keeps the area test from failing '
                        'by chance (about 1 in 16 000 per face) while a sampler ignoring area is off by tens of σ.')


def scripted_pick(m) -> tuple[dict, list]:
    """(trajectory record, [(onset, end)]): a pick on the UR5e sample at 20 Hz. The arm lowers (samples 0-20), holds,
    the 2F-85 closes (25-35: both spring-link joints 0 -> 0.5 rad), the arm lifts holding (35-45), the gripper opens
    (45-50), the arm moves (50-70), closes again (70-75: 0 -> 0.3 rad), opens (80-84), and a 0.01 rad twitch at 86-88
    that is not a grasp (its fall is below 5 mm). The script's own closing ranges are the reference."""
    n = 90
    t = np.arange(n) * 0.05
    J = {nm: i for i, nm in enumerate(m.joint_names)}
    q0 = m.home.copy()
    a = np.zeros(n)
    a[:21] = np.linspace(0.0, 1.0, 21)
    a[21:] = 1.0
    a[35:46] = np.linspace(1.0, 0.6, 11)
    a[46:] = 0.6
    a[50:71] = np.linspace(0.6, 1.1, 21)
    a[71:] = 1.1
    g = np.zeros(n)
    g[25:36] = np.linspace(0.0, 0.5, 11)
    g[36:45] = 0.5
    g[45:51] = np.linspace(0.5, 0.0, 6)
    g[70:76] = np.linspace(0.0, 0.3, 6)
    g[76:80] = 0.3
    g[80:85] = np.linspace(0.3, 0.0, 5)
    g[86:89] = np.array([0.0, 0.005, 0.01])
    g[89:] = 0.01
    Q = np.tile(q0, (n, 1))
    Q[:, J['shoulder_lift_joint']] += 0.25 * a
    Q[:, J['elbow_joint']] -= 0.35 * a
    Q[:, J['wrist_1_joint']] += 0.10 * a
    for side in ('right', 'left'):
        Q[:, J[f'gripper_{side}_spring_link_joint']] = g
    tr = {'uid': 'scripted-pick', 'name': 'Scripted pick', 'joints': list(m.joint_names), 't': t.tolist(),
          'q': Q.tolist()}
    return tr, [(25, 35), (70, 75)]


@check('grasp.trajectory_closures', 'grasp', 'Grasps found where the gripper closes in a trajectory',
       'largest |opening − the distance between the pads\' inner-face centres| over the trajectory\'s samples (m); '
       'the strokes found must be the script\'s closing ranges, sample for sample',
       'the 2F-85 pads\' inner faces built from MuJoCo\'s compiled box geoms (mj_kinematics on a fresh MjData); the '
       'sample ranges the scripted trajectory closes the fingers over', 1e-6, 'm',
       refs=('mujoco_todorov2012', 'mujoco_xml'))
def _closures(ctx):
    from ..core import grasps as GR
    from ..core.robot.model import get_model
    p, r = _ur5e(ctx)
    m = get_model(p, r.uid)
    tr, want = scripted_pick(m)
    tr['robot_uid'] = r.uid
    p.tasks.trajectories = [tr]
    Q = np.asarray(tr['q'], float)
    opening = GR.pad_opening(p, r.uid, Q)
    _f, pick = _inner_faces_mujoco(m, m.home, r.fingertip_frames)
    ref = []
    for q in Q:
        faces, _p = _inner_faces_mujoco(m, q, r.fingertip_frames, pick)
        cs = [np.average([f[0] for f in fl], axis=0, weights=[f[4] * f[5] for f in fl]) for fl in faces.values()]
        ref.append(float(np.linalg.norm(cs[0] - cs[1])))
    err = np.abs(opening - np.asarray(ref))
    gs = GR.from_trajectory(p, tr['uid'], mode='closures')
    got = [(e['t'], e['t_end']) for e in gs['settings']['events']]
    t = tr['t']
    expect = [(t[a], t[b]) for a, b in want]
    same = len(got) == len(expect) and all(abs(x - y) < 1e-12 for g, w in zip(got, expect) for x, y in zip(g, w))
    return Outcome(float(err.max()), {'samples': len(Q), 'strokes_found_s': got, 'strokes_scripted_s': expect,
                                      'opening_range_mm': [1000 * float(opening.min()), 1000 * float(opening.max())]},
                   ok=same,
                   figure=figure('Gripper opening along the scripted pick', 't (s)', 'opening (m)',
                                 [series('pad_opening', t, opening), series('MuJoCo inner faces', t, ref, 'points')]),
                   note='The pads are rigid boxes, so the opening is exact up to float32 meshes (about 3e-8 m); '
                        'picking the wrong face or link shows as millimetres. The strokes are exact: the script holds '
                        'the fingers still between its ramps.')
