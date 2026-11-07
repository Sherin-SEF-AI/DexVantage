"""The work of the export round-trip checks (registered in checks_formats.py): each exporter's files read back by the
target's own parser (MuJoCo's MJCF and URDF compilers, cv2.FileStorage, PyYAML, Python's ast and ElementTree) and
compared with DexVantage's own models, which wrote them.

    mjcf_snippets(ctx) -> (error m, ok, detail)
    urdf_roundtrip(ctx) -> (error m, ok, detail)
    ros2_tf(ctx) -> (error m, ok, detail)
    opencv_yaml(ctx) -> (error, ok, detail)
    kalibr_camchain(ctx) -> (error, ok, detail)
    masks_vs_occlusion(ctx) -> (error, ok, detail)

The references are written again here where a check needs a formula (urdfdom's RPY quaternion, Rodrigues' rotation,
the depth-validity conditions per pixel), and none calls the exporter's own conversion code to check itself.
"""
from __future__ import annotations

import ast
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

SAMPLES = ('ur5e', 'so100', 'dexhand')
MJ_INTRINSICS_PX = 1e-3     # MuJoCo keeps cam_intrinsic in float32: about 6 × 10⁻⁵ px at a 1 000 px focal length


# ---------------------------------------------------------------------------------------------------------------
# shared
def random_q(project, rng) -> dict:
    """A random joint state of every robot, 20–80 % of each joint's range (±π about home when unlimited)."""
    from ..core.robot.model import get_model
    q = {}
    for r in project.robots:
        rm = get_model(project, r.uid)
        home = np.asarray(rm.home, float)
        lo = np.where(np.isfinite(rm.lower), rm.lower, home - np.pi)
        hi = np.where(np.isfinite(rm.upper), rm.upper, home + np.pi)
        q[r.uid] = lo + (hi - lo) * rng.uniform(0.2, 0.8, len(home))
    return q


def rot_angle(Ra, Rb) -> float:
    """The angle (rad) of Ra^T Rb, by 2·atan2(|v|, |w|) of its quaternion (full precision near zero)."""
    R = np.asarray(Ra, float).T @ np.asarray(Rb, float)
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 4
    if w < 1e-6:                                      # near 180°: the angle from the trace
        return math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    return 2 * math.atan2(float(np.linalg.norm(v)) / w, w)


def pose_errors(T_ref, T_got) -> tuple[float, float]:
    T_ref, T_got = np.asarray(T_ref, float), np.asarray(T_got, float)
    return float(np.linalg.norm(T_ref[:3, 3] - T_got[:3, 3])), rot_angle(T_ref[:3, :3], T_got[:3, :3])


def _export(project, folder, fmt, options=None):
    from ..core import exporters as X
    man = X.export(project, folder, [fmt], {fmt: options or {}})
    row = man['formats'][fmt]
    if row['status'] != 'ok':
        raise RuntimeError(f'the {fmt} export was refused: {row.get("reason")}')
    return row


# ---------------------------------------------------------------------------------------------------------------
# MJCF snippets
def mjcf_snippets(ctx):
    """Each sample's camera snippets pasted into the robot's own MJCF (the stored file, as core/sim/scene.robot_spec
    reads it, no prefix), loaded by MuJoCo's XML parser, posed at a random joint state by joint name: the cameras'
    optical frames against camrender.optical_pose and their intrinsics against the lens models."""
    import mujoco

    from ..core import camrender as CR
    from ..core.exporters import _common as C
    from ..core.robot.model import get_model
    from ..core.robot.samples import sample_project
    from ..core.sim.scene import robot_spec
    from .mjrender import camera_model
    rng = ctx.rng('formats.mjcf_snippets')
    flip = np.diag([1.0, -1.0, -1.0])
    worst_p = worst_r = worst_k = 0.0
    detail = {}
    for name in SAMPLES:
        p = sample_project(name)
        folder = os.path.join(ctx.tmp(), f'mjcf_{name}')
        _export(p, folder, 'mjcf', {'mode': 'snippets'})
        snip = ET.parse(os.path.join(folder, 'mjcf', 'cameras_snippet.xml')).getroot()
        text = open(os.path.join(folder, 'mjcf', 'cameras_snippet.xml'), encoding='utf-8').read()
        # the block comments name the body each run of elements goes into
        blocks, cur = [], None
        for line in text.splitlines():
            s = line.strip()
            if s.startswith('<!-- inside <body name="'):
                cur = s.split('"')[1]
                blocks.append([cur, []])
            elif s.startswith('<!-- inside <worldbody>'):
                cur = '__world__'
                blocks.append([cur, []])
            elif cur and (s.startswith('<camera') or s.startswith('<site')):
                blocks[-1][1].append(ET.fromstring(s))
        n_el = sum(len(b[1]) for b in blocks)
        if n_el != len(list(snip)):
            raise RuntimeError(f'{name}: {len(list(snip))} elements in the snippet, {n_el} under its comments')
        q = random_q(p, rng)
        cams = C.rig_cameras(p, q)
        rows = []
        for r in p.robots:
            # (1) pasted as text into the robot's file (MuJoCo's own writer, 6 digits) and loaded from XML: every
            # camera and site is found under its body
            spec, assets, _n = robot_spec(p, r, '')
            spec.assets = assets
            root = ET.fromstring(spec.to_xml())
            bodies = {b.get('name'): b for b in root.iter('body')}
            world = root.find('worldbody')
            for body, els in blocks:
                tgt = world if body == '__world__' else bodies.get(body)
                if tgt is None:
                    continue
                for e in els:
                    tgt.append(e)
            mx = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'), assets)
            # (2) for the figures, at full precision: each block parsed by MuJoCo's XML parser on its own and
            # attached to its body of the robot's in-memory model (the file's numbers are not rewritten)
            spec, assets, _n = robot_spec(p, r, '')
            for k, (body, els) in enumerate(blocks):
                tgt = spec.worldbody if body == '__world__' else spec.body(body)
                if tgt is None:
                    continue
                xml = ('<mujoco><worldbody><body name="snip">' + ''.join(ET.tostring(e, encoding='unicode')
                                                                         for e in els) + '</body></worldbody></mujoco>')
                tgt.add_frame().attach_body(mujoco.MjSpec.from_string(xml).body('snip'), f'b{k}_', '')
            spec.assets = assets
            m = spec.compile()
            prefixes = {e.get('name'): f'b{k}_' for k, (_b, els) in enumerate(blocks) for e in els}
            d = mujoco.MjData(m)
            rm = get_model(p, r.uid)
            for jn, v in zip(rm.joint_names, q[r.uid]):
                d.qpos[m.joint(jn).qposadr[0]] = v
            mujoco.mj_forward(m, d)
            Binv = np.linalg.inv(r.base.matrix())
            for c in cams:
                if (c.robot_uid and c.robot_uid != r.uid) or (not c.robot_uid and not r.primary):
                    continue
                try:
                    mx.camera(c.key)
                except KeyError:
                    raise RuntimeError(f'{name}: camera {c.key} is not in the pasted file') from None
                cid = m.camera(prefixes[c.key] + c.key).id
                T = np.eye(4)
                T[:3, :3] = d.cam_xmat[cid].reshape(3, 3) @ flip
                T[:3, 3] = d.cam_xpos[cid]
                ref = Binv @ CR.optical_pose(p, c.sensor, c.imager, q).matrix()
                e_p, e_r = pose_errors(ref, T)
                cm = camera_model(m, prefixes[c.key] + c.key, c.camera.width, c.camera.height)
                e_k = max(abs(cm.fx - c.camera.fx), abs(cm.fy - c.camera.fy), abs(cm.cx - c.camera.cx),
                          abs(cm.cy - c.camera.cy))
                worst_p, worst_r, worst_k = max(worst_p, e_p), max(worst_r, e_r), max(worst_k, e_k)
                rows.append({'camera': c.key, 'position_m': e_p, 'rotation_rad': e_r, 'intrinsics_px': e_k})
        if len(rows) != len(cams):
            raise RuntimeError(f'{name}: {len(cams)} cameras exported, {len(rows)} found in the loaded models')
        detail[name] = rows
    return worst_p, worst_r <= 1e-9 and worst_k <= MJ_INTRINSICS_PX, dict(detail, rotation_rad_max=worst_r,
                                                              intrinsics_px_max=worst_k)


# ---------------------------------------------------------------------------------------------------------------
# URDF
def urdf_quat_rpy(r, p, y) -> np.ndarray:
    """urdfdom's Rotation::setFromRPY written again (urdf_model/pose.h): the quaternion (x, y, z, w)."""
    phi, the, psi = r / 2.0, p / 2.0, y / 2.0
    x = math.sin(phi) * math.cos(the) * math.cos(psi) - math.cos(phi) * math.sin(the) * math.sin(psi)
    yy = math.cos(phi) * math.sin(the) * math.cos(psi) + math.sin(phi) * math.cos(the) * math.sin(psi)
    z = math.cos(phi) * math.cos(the) * math.sin(psi) - math.sin(phi) * math.sin(the) * math.cos(psi)
    w = math.cos(phi) * math.cos(the) * math.cos(psi) + math.sin(phi) * math.sin(the) * math.sin(psi)
    q = np.array([x, yy, z, w])
    return q / np.linalg.norm(q)


def _qmat(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _rodrigues(axis, ang) -> np.ndarray:
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) * math.cos(ang) + math.sin(ang) * K + (1 - math.cos(ang)) * np.outer(a, a)


def urdf_fk(text: str, qnamed: dict) -> tuple[dict, list]:
    """{link: 4 × 4 in the root's frame} by walking the URDF with ElementTree and urdfdom's rules (origin xyz and
    rpy, then the joint's motion about its axis through the child frame); and the structural problems found (no
    single root, a link reached twice, a revolute or prismatic joint without lower, upper, effort and velocity)."""
    root = ET.fromstring(text)
    problems = []
    links = [ln.get('name') for ln in root.findall('link')]
    if len(set(links)) != len(links):
        problems.append('duplicate link names')
    kids = {}
    children = set()
    for j in root.findall('joint'):
        par, ch = j.find('parent').get('link'), j.find('child').get('link')
        if ch in children:
            problems.append(f'{ch} has two parents')
        children.add(ch)
        kids.setdefault(par, []).append(j)
        kind = j.get('type')
        if kind in ('revolute', 'prismatic'):
            lim = j.find('limit')
            if lim is None or any(lim.get(k) is None for k in ('lower', 'upper', 'effort', 'velocity')):
                problems.append(f'joint {j.get("name")} has an incomplete <limit>')
        elif kind == 'continuous':
            lim = j.find('limit')
            if lim is not None and any(lim.get(k) is None for k in ('effort', 'velocity')):
                problems.append(f'joint {j.get("name")} has a <limit> without effort or velocity')
        elif kind != 'fixed':
            problems.append(f'joint {j.get("name")} has type {kind}')
    roots = [ln for ln in links if ln not in children]
    if len(roots) != 1:
        problems.append(f'{len(roots)} root links')
    T = {roots[0]: np.eye(4)} if roots else {}
    stack = list(roots[:1])
    while stack:
        par = stack.pop()
        for j in kids.get(par, []):
            ch = j.find('child').get('link')
            o = j.find('origin')
            xyz = [float(v) for v in (o.get('xyz') or '0 0 0').split()]
            rpy = [float(v) for v in (o.get('rpy') or '0 0 0').split()]
            O = np.eye(4)
            O[:3, :3] = _qmat(urdf_quat_rpy(*rpy))
            O[:3, 3] = xyz
            M = np.eye(4)
            kind = j.get('type')
            if kind != 'fixed':
                ax = [float(v) for v in (j.find('axis').get('xyz')).split()]
                v = float(qnamed.get(j.get('name'), 0.0))
                if kind == 'prismatic':
                    M[:3, 3] = np.asarray(ax) / np.linalg.norm(ax) * v
                else:
                    M[:3, :3] = _rodrigues(ax, v)
            T[ch] = T[par] @ O @ M
            stack.append(ch)
    if len(T) != len(links):
        problems.append(f'{len(links) - len(T)} links not reached from the root')
    return T, problems


def urdf_roundtrip(ctx):
    """Each sample's rig.urdf: (1) parsed with ElementTree and posed by urdf_fk; (2) compiled by MuJoCo's URDF compiler
    (fusestatic off, every link a body) and posed by mj_kinematics; every sensor body frame and optical frame against
    DexVantage (scenegeom.sensor_world_pose, camrender.optical_pose) at a random joint state."""
    import mujoco

    from ..core import camrender as CR
    from ..core.exporters import _common as C
    from ..core.robot.model import get_model
    from ..core.robot.samples import sample_project
    from ..core.scenegeom import sensor_world_pose
    rng = ctx.rng('formats.urdf_roundtrip')
    worst_p = worst_r = 0.0
    worst_mj = 0.0
    problems, detail = [], {}
    for name in SAMPLES:
        p = sample_project(name)
        folder = os.path.join(ctx.tmp(), f'urdf_{name}')
        _export(p, folder, 'urdf')
        text = open(os.path.join(folder, 'ros2', 'rig.urdf'), encoding='utf-8').read()
        q = random_q(p, rng)
        qn = {}
        for r in p.robots:
            rm = get_model(p, r.uid)
            qn.update(dict(zip(rm.joint_names, q[r.uid])))
        Tx, prob = urdf_fk(text, qn)
        problems += [f'{name}: {x}' for x in prob]
        root = ET.fromstring(text)
        for old in root.findall('mujoco'):            # the file's own compiler element, replaced
            root.remove(old)
        mj = ET.SubElement(root, 'mujoco')
        ET.SubElement(mj, 'compiler', fusestatic='false', discardvisual='true', boundmass='1e-6',
                      boundinertia='1e-9')
        md = os.path.join(folder, 'ros2', 'meshes')
        assets = {f: open(os.path.join(md, f), 'rb').read() for f in os.listdir(md)} if os.path.isdir(md) else {}
        m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'), assets)
        d = mujoco.MjData(m)
        for jn, v in qn.items():
            d.qpos[m.joint(jn).qposadr[0]] = v
        mujoco.mj_kinematics(m, d)
        rows = []
        for c in C.rig_cameras(p, q):
            ref_o = CR.optical_pose(p, c.sensor, c.imager, q).matrix()
            ref_s = sensor_world_pose(p, c.sensor, q).matrix()
            for frame, ref in ((c.optical_frame, ref_o), (c.sensor_frame, ref_s)):
                e1 = pose_errors(ref, Tx[frame])
                b = m.body(frame).id
                Tm = np.eye(4)
                Tm[:3, :3] = d.xmat[b].reshape(3, 3)
                Tm[:3, 3] = d.xpos[b]
                e2 = pose_errors(ref, Tm)
                worst_p = max(worst_p, e1[0])
                worst_r = max(worst_r, e1[1])
                worst_mj = max(worst_mj, e2[0], e2[1])
                rows.append({'frame': frame, 'xml_position_m': e1[0], 'xml_rotation_rad': e1[1],
                             'mujoco_position_m': e2[0], 'mujoco_rotation_rad': e2[1]})
        detail[name] = {'links': len(root.findall('link')), 'joints': len(root.findall('joint')), 'frames': rows}
    detail['problems'] = problems
    detail['xml_rotation_rad_max'] = worst_r
    detail['mujoco_max'] = worst_mj
    return max(worst_p, worst_mj), not problems and worst_r <= 1e-9, detail


def _launch_nodes(text: str) -> list:
    """[(parent, child, 4 × 4)] of every static_transform_publisher Node in a ROS 2 Python launch file, read with
    Python's ast (no ROS needed): the Node(...) calls' arguments=[...] lists of tf2_ros's flags."""
    out = []
    for node in ast.walk(ast.parse(text)):
        if not (isinstance(node, ast.Call) and getattr(node.func, 'id', '') == 'Node'):
            continue
        kw = {k.arg: k.value for k in node.keywords}
        if ast.literal_eval(kw['executable']) != 'static_transform_publisher' or \
                ast.literal_eval(kw['package']) != 'tf2_ros':
            continue
        args = ast.literal_eval(kw['arguments'])
        a = {args[i]: args[i + 1] for i in range(0, len(args), 2)}
        qv = np.array([float(a.get(k, 0.0)) for k in ('--qx', '--qy', '--qz', '--qw')])
        T = np.eye(4)
        T[:3, :3] = _qmat(qv / np.linalg.norm(qv))
        T[:3, 3] = [float(a.get(k, 0.0)) for k in ('--x', '--y', '--z')]
        out.append((a['--frame-id'], a['--child-frame-id'], T))
    return out


def ros2_tf(ctx):
    """Each sample's static_transforms.launch.py (Python's ast) and tf_tree.yaml (PyYAML): every optical frame's world
    pose, its parent link posed by DexVantage's kinematics composed with the file's static transforms, against
    camrender.optical_pose at a random joint state; and the tree's world poses at the export's pose."""
    import yaml

    from ..core import camrender as CR
    from ..core.exporters import _common as C
    from ..core.robot.model import get_model
    from ..core.robot.samples import sample_project
    rng = ctx.rng('formats.ros2_tf')
    worst, worst_r, detail, problems = 0.0, 0.0, {}, []
    for name in SAMPLES:
        p = sample_project(name)
        folder = os.path.join(ctx.tmp(), f'tf_{name}')
        q = random_q(p, rng)
        _export(p, folder, 'urdf', {'q': {k: list(map(float, v)) for k, v in q.items()}})
        nodes = _launch_nodes(open(os.path.join(folder, 'ros2', 'static_transforms.launch.py'),
                                   encoding='utf-8').read())
        tree = yaml.safe_load(open(os.path.join(folder, 'ros2', 'tf_tree.yaml'), encoding='utf-8'))
        par = {c: (pa, T) for pa, c, T in nodes}
        yfr = {f['child']: f for f in tree['frames']}
        if set(yfr) != set(par):
            problems.append(f'{name}: the launch file and the tf2 tree name different frames')
        rows = []
        cams = C.rig_cameras(p, q)
        for c in cams:
            T = np.eye(4)
            f = c.optical_frame
            while f in par:
                pa, Tf = par[f]
                T = Tf @ T
                f = pa
            if c.robot_uid:
                base = get_model(p, c.robot_uid).link_pose(q[c.robot_uid], f)
            else:
                base = np.eye(4) if f == C.WORLD else None
            if base is None:
                problems.append(f'{name}: {c.optical_frame} does not reach the world or a robot link')
                continue
            ref = CR.optical_pose(p, c.sensor, c.imager, q).matrix()
            e = pose_errors(ref, base @ T)
            y = yfr.get(c.optical_frame) or {}
            Ty = np.eye(4)
            qy = np.asarray(y.get('world_rotation_xyzw_at_export', [0, 0, 0, 1]), float)
            Ty[:3, :3] = _qmat(qy / np.linalg.norm(qy))
            Ty[:3, 3] = y.get('world_translation_at_export', [np.nan] * 3)
            ey = pose_errors(ref, Ty)
            worst = max(worst, e[0], ey[0])
            worst_r = max(worst_r, e[1], ey[1])
            rows.append({'frame': c.optical_frame, 'launch_position_m': e[0], 'launch_rotation_rad': e[1],
                         'tree_position_m': ey[0], 'tree_rotation_rad': ey[1]})
        detail[name] = {'nodes': len(nodes), 'frames': rows}
    detail['problems'] = problems
    detail['rotation_rad_max'] = worst_r
    return worst, not problems and worst_r <= 1e-9, detail


# ---------------------------------------------------------------------------------------------------------------
# OpenCV YAML, Kalibr
REF_LENSES = (
    ('pinhole', ()), ('brown', (-0.21, 0.07, 1.2e-3, -8e-4, -0.012)),
    ('brown', (0.12, -0.03, 4e-4, 2e-4, 0.002, 0.45, -0.02, 0.01)), ('kb', (0.08, -0.02, 0.004, -0.0006)),
    ('equidistant', ()), ('equisolid', ()), ('stereographic', ()),
)


def reference_rig(ctx):
    """The UR5e sample (a D405 on the wrist, a D435 on the cage: six imagers) with every imager calibrated to each
    lens model of REF_LENSES in turn (Sensor.calibration, as a calibrated unit), so the lens mapping of every model is
    written and read back."""
    from ..core import library as LB
    from ..core.robot.samples import sample_project
    out = []
    for model, dist in REF_LENSES:
        p = sample_project('ur5e')
        # Brown and the narrow fisheyes inside 90° of the axis (exact mappings); the equisolid and stereographic
        # lenses reach past 90° in the corners (no exact equivalent, and some rays OpenCV cannot take)
        f = {'pinhole': 520.0, 'brown': 520.0, 'kb': 600.0, 'equidistant': 600.0}.get(model, 500.0)
        K = [[f, 0.0, 641.3], [0.0, f * 1.003, 357.9], [0.0, 0.0, 1.0]]
        for s in p.variant.sensors:
            s.calibration = {'imagers': {im.name: {'K': K, 'D': list(dist), 'width': 1280, 'height': 720,
                                                   'model': model} for im in LB.sensor_imagers(s)}}
        out.append((f'{model}{len(dist) or ""}', p))
    return out


def opencv_yaml(ctx):
    """Every imager's OpenCV YAML read back by cv2.FileStorage against its source (size, K, D, the frames' 4 × 4
    matrices), and projected through cv2.projectPoints or cv2.fisheye.projectPoints against DexVantage's lens model
    (exact mappings within 10⁻⁶ px; approximations within the error they state); cameras.json read back too."""
    import json

    import cv2

    from ..core.exporters import _common as C
    from ..core.exporters.calibration import CAMERAS_JSON, OPENCV_DIR
    worst, worst_px, problems, rows = 0.0, 0.0, [], []
    rng = ctx.rng('formats.opencv_yaml')
    for tag, p in reference_rig(ctx):
        folder = os.path.join(ctx.tmp(), f'cv_{tag}')
        _export(p, folder, 'opencv_yaml')
        cams = C.rig_cameras(p, None)
        js = json.load(open(os.path.join(folder, CAMERAS_JSON), encoding='utf-8'))
        jcams = {c['name']: c for c in js['cameras']}
        for c in cams:
            fs = cv2.FileStorage(os.path.join(folder, OPENCV_DIR, c.key + '.yaml'), cv2.FILE_STORAGE_READ)
            try:
                W = int(fs.getNode('image_width').real())
                H = int(fs.getNode('image_height').real())
                K = fs.getNode('camera_matrix').mat()
                D = fs.getNode('distortion_coefficients').mat().ravel()
                model = fs.getNode('distortion_model').string()
                exact = int(fs.getNode('exact').real())
                err_stated = float(fs.getNode('approximation_error_px').real())
                Tp = fs.getNode('T_parent_camera').mat()
                Tw = fs.getNode('T_world_camera').mat()
                parent = fs.getNode('parent_frame').string()
            finally:
                fs.release()
            cm = c.camera
            K0 = np.array([[cm.fx, 0, cm.cx], [0, cm.fy, cm.cy], [0, 0, 1.0]])
            diff = max(float(np.max(np.abs(K - K0))), float(np.max(np.abs(Tp - c.T_parent_optical))),
                       float(np.max(np.abs(Tw - c.T_world_optical))), abs(W - cm.width) + abs(H - cm.height))
            if parent != c.parent:
                problems.append(f'{tag} {c.key}: parent {parent}')
            j = jcams.get(c.key) or {}
            ji = j.get('intrinsics') or {}
            diff = max(diff, max(abs(float(ji.get(k, np.nan)) - getattr(cm, k)) for k in ('fx', 'fy', 'cx', 'cy')))
            if [float(v) for v in ji.get('dist', [])] != list(cm.dist) or ji.get('model') != cm.model:
                problems.append(f'{tag} {c.key}: cameras.json lens')
            if model in ('plumb_bob', 'rational_polynomial') and cm.model in ('pinhole', 'brown') and \
                    len(D) != (5 if model == 'plumb_bob' else 8):
                problems.append(f'{tag} {c.key}: {len(D)} coefficients for {model}')
            worst = max(worst, diff)
            # projection through OpenCV against the lens, at random rays inside the image
            uv0 = np.stack([rng.uniform(0, W - 1, 400), rng.uniform(0, H - 1, 400)], 1)
            rays, ok = cm.unproject(uv0, with_ok=True)
            rays, uv0 = rays[ok], uv0[ok]
            front = rays[:, 2] > 0.05
            rays, uv0 = rays[front], uv0[front]
            P = rays.reshape(-1, 1, 3).astype(np.float64)
            if model == 'fisheye':
                uv, _ = cv2.fisheye.projectPoints(P, np.zeros(3), np.zeros(3), K, D.reshape(4, 1))
            else:
                uv, _ = cv2.projectPoints(P, np.zeros(3), np.zeros(3), K, D)
            e_px = float(np.max(np.linalg.norm(uv.reshape(-1, 2) - uv0, axis=1)))
            limit = 1e-6 if exact else (err_stated + 1e-6 if err_stated >= 0 else np.inf)
            if e_px > limit:
                problems.append(f'{tag} {c.key}: OpenCV projects {e_px:.3g} px from the lens (allowed {limit:.3g})')
            worst_px = max(worst_px, e_px if exact else 0.0)
            rows.append({'lens': tag, 'camera': c.key, 'model': model, 'exact': bool(exact),
                         'stated_error_px': err_stated, 'opencv_vs_lens_px': e_px, 'readback_max_abs': diff})
    return worst, not problems and worst_px <= 1e-6, {'rows': rows, 'problems': problems,
                                                      'exact_projection_px_max': worst_px}


KALIBR_MODELS = {'none': 0, 'radtan': 4, 'equidistant': 4, 'fov': 1}     # ConfigReader / Supported models


def kalibr_camchain(ctx):
    """camchain.yaml parsed back with PyYAML: Kalibr's keys, intrinsics [fu fv pu pv], coefficient counts per
    distortion model, resolution, and T_cn_cnm1 against the chain from DexVantage's optical poses
    (inv(T_world_cn) · T_world_cnm1) on every reference lens and the UR5e sample."""
    import yaml

    from ..core.exporters import _common as C
    from ..core.exporters.calibration import KALIBR
    from ..core.robot.samples import sample_project
    worst, problems, rows = 0.0, [], []
    cases = reference_rig(ctx) + [('ur5e', sample_project('ur5e'))]
    for tag, p in cases:
        folder = os.path.join(ctx.tmp(), f'kalibr_{tag}')
        _export(p, folder, 'opencv_yaml')
        doc = yaml.safe_load(open(os.path.join(folder, KALIBR), encoding='utf-8'))
        cams = C.rig_cameras(p, None)
        if sorted(doc) != [f'cam{i}' for i in range(len(cams))]:
            problems.append(f'{tag}: keys {sorted(doc)}')
            continue
        for n, c in enumerate(cams):
            e = doc[f'cam{n}']
            cm = c.camera
            if e.get('camera_model') != 'pinhole' or len(e.get('intrinsics', [])) != 4:
                problems.append(f'{tag} cam{n}: camera model or intrinsics')
                continue
            dm = e.get('distortion_model')
            if dm not in KALIBR_MODELS or len(e.get('distortion_coeffs', [])) != KALIBR_MODELS[dm]:
                problems.append(f'{tag} cam{n}: distortion {dm} with {len(e.get("distortion_coeffs", []))} terms')
            if not isinstance(e.get('rostopic'), str):
                problems.append(f'{tag} cam{n}: rostopic')
            d = max(abs(a - b) for a, b in zip(e['intrinsics'], (cm.fx, cm.fy, cm.cx, cm.cy)))
            if list(e.get('resolution', [])) != [cm.width, cm.height]:
                problems.append(f'{tag} cam{n}: resolution')
            if n:
                ref = np.linalg.inv(c.T_world_optical) @ cams[n - 1].T_world_optical
                d = max(d, float(np.max(np.abs(np.asarray(e['T_cn_cnm1'], float) - ref))))
            elif 'T_cn_cnm1' in e:
                problems.append(f'{tag} cam0 has T_cn_cnm1')
            if cm.model == 'brown' and dm == 'radtan' and len(cm.dist) == 5 and cm.dist[4] == 0.0:
                d = max(d, max(abs(a - b) for a, b in zip(e['distortion_coeffs'], cm.dist[:4])))
            if cm.model == 'kb' and dm == 'equidistant':
                d = max(d, max(abs(a - b) for a, b in zip(e['distortion_coeffs'], cm.dist)))
            worst = max(worst, d)
            rows.append({'case': tag, 'cam': n, 'name': c.key, 'distortion_model': dm, 'max_abs': d})
    return worst, not problems, {'rows': rows, 'problems': problems}


# ---------------------------------------------------------------------------------------------------------------
# masks
def _pixel_rays(cam, stride: int):
    vs, us = np.mgrid[0:cam.height:stride, 0:cam.width:stride]
    uv = np.stack([us.ravel(), vs.ravel()], 1).astype(float)
    rays, ok = cam.unproject(uv, with_ok=True)
    return us.ravel(), vs.ravel(), rays, ok


def masks_vs_occlusion(ctx):
    """The UR5e sample's masks read back from the PNG files (cv2.imread) against, per sampled pixel:
    (a) exact ray casting (core/raycast.py, Möller-Trumbore on a uniform grid) against the occlusion study's occluders
    (scenegeom.sensor_occluders: the robots, objects and tables, the camera's seat left out): the pixel is the
    robot's when the nearest robot hit is nearer than the nearest other hit; (b) the occlusion study's cube map
    (core/occlusion.CubeMap of the same occluders, n = 2048): the first non-robot surface on the pixel's ray (or a
    point 3 m out) is hidden; and the depth-validity codes against the conditions evaluated again per pixel from
    exact ray casting (first surface, z against min-Z and range, line of sight from every imager the depth needs,
    the projector cone)."""
    import json

    import cv2

    from ..core import occlusion as OC
    from ..core import raycast as RC
    from ..core.exporters import _common as C
    from ..core.frames import Pose
    from ..core.robot.samples import sample_project
    from ..core.scenegeom import sensor_occluders, sensor_world_pose
    from ..core.studies import depth_validity as DV
    p = sample_project('ur5e')
    stride = 8 if ctx.quick else 4
    folder = os.path.join(ctx.tmp(), 'masks_ur5e')
    _export(p, folder, 'masks', {'atlas': False})
    idx = json.load(open(os.path.join(folder, 'masks', 'masks.json'), encoding='utf-8'))
    q = {uid: np.asarray(v, float) for uid, v in idx['q'].items()}
    detail, worst_rc, worst_cube = {}, 0.0, 0.0
    cams = {c.key: c for c in C.rig_cameras(p, q, 'primary')}
    for rec in idx['cameras']:
        c = cams[rec['name']]
        mask = cv2.imread(os.path.join(folder, rec['mask']), cv2.IMREAD_UNCHANGED) > 0
        pose = Pose.from_matrix(c.T_world_optical)
        o = np.asarray(pose.position, float)
        occ = sensor_occluders(p, c.sensor, q, origins=o[None])
        rob = np.array([w.startswith('robot:') for w in occ.owners])[occ.owner]
        us, vs, rays, ok = _pixel_rays(c.camera, stride)
        dw = rays @ c.T_world_optical[:3, :3].T
        dr = RC.first_hit(o, dw[ok], occ.tris[rob])
        dn = RC.first_hit(o, dw[ok], occ.tris[~rob])
        ref = np.zeros(len(us), bool)
        ref[ok] = np.isfinite(dr) & (dr < dn)
        got = mask[vs, us]
        n_rc = int((ref != got).sum())
        # the occlusion study's cube map
        cube = OC.CubeMap.build(occ, o, 2048, 'auto')
        far = np.where(np.isfinite(dn), dn, 3.0)
        P = o + dw[ok] * far[:, None]
        hid = np.zeros(len(us), bool)
        hid[ok] = ~cube.visible(P)
        n_cube = int((hid != got).sum())
        worst_rc = max(worst_rc, n_rc / len(us))
        worst_cube = max(worst_cube, n_cube / len(us))
        detail[rec['name']] = {'pixels': int(len(us)), 'robot_pixels': int(got.sum()), 'vs_raycast': n_rc,
                               'vs_cube_map': n_cube, 'cube_backend': cube.backend if hasattr(cube, 'backend') else '',
                               'stride': stride}
    # depth validity
    worst_depth = 0.0
    for rec in idx['depth']:
        s = next(x for x in p.variant.sensors if x.uid == rec['sensor_uid'])
        ds = DV.setup(p, s)
        body = sensor_world_pose(p, s, q)
        code = cv2.imread(os.path.join(folder, rec['code']), cv2.IMREAD_UNCHANGED)
        cam = ds.ref.camera
        refT = DV.imager_world(body, ds.ref).matrix()
        o = refT[:3, 3]
        origins = np.array([DV.imager_world(body, im).position for im in ds.needed])
        occ = sensor_occluders(p, s, q, origins=origins)
        grid = RC.Grid.build(occ.tris)
        us, vs, rays, ok = _pixel_rays(cam, stride)
        dw = rays @ refT[:3, :3].T
        t = np.full(len(us), np.inf)
        t[ok] = grid.first_hit(o, dw[ok])
        hit = np.isfinite(t)
        Pw = o + dw * np.where(hit, t, 0.0)[:, None]
        exp = np.full(len(us), 255, np.int64)
        z = (Pw - o) @ refT[:3, 2]
        conds = []
        in_fov = np.ones(len(us), bool)
        for im in ds.needed:
            T = DV.imager_world(body, im).matrix()
            Q = (Pw - T[:3, 3]) @ T[:3, :3]
            _uv, okp = im.camera.project(Q)
            in_fov &= okp
        conds.append(in_fov)
        conds.append(np.ones(len(us), bool) if ds.min_z_m is None else z >= ds.min_z_m)
        conds.append(np.ones(len(us), bool) if ds.max_range_m is None else z <= ds.max_range_m)
        vis = np.ones(len(us), bool)
        for im in ds.needed:
            vis[hit] &= RC.visible_exact(Pw[hit], DV.imager_world(body, im).position, grid, eps=OC.EPS_M)
        conds.append(vis)
        pj = np.ones(len(us), bool)
        if ds.need_projector and ds.projector is not None and ds.technology != 'structured_light':
            Tp = (body @ ds.projector['pose']).matrix()
            Qp = (Pw - Tp[:3, 3]) @ Tp[:3, :3]
            th = math.tan(math.radians(float(ds.projector['hfov_deg'])) / 2)
            tv = math.tan(math.radians(float(ds.projector['vfov_deg'])) / 2)
            pj = (Qp[:, 2] > 0) & (np.abs(Qp[:, 0]) <= th * Qp[:, 2]) & (np.abs(Qp[:, 1]) <= tv * Qp[:, 2])
        conds.append(pj)
        conds.append(np.ones(len(us), bool))        # no material-tagged object in the UR5e sample's scene
        e = np.zeros(len(us), np.int64)
        for k in range(len(conds) - 1, -1, -1):
            e[~conds[k]] = k + 1
        exp[hit] = e[hit]
        got = code[vs, us].astype(np.int64)
        n_bad = int((got != exp).sum())
        worst_depth = max(worst_depth, n_bad / len(us))
        detail['depth_' + rec['name']] = {'pixels': int(len(us)), 'mismatched': n_bad,
                                          'by_code': {int(k): int(((exp == k) & (got != exp)).sum())
                                                      for k in np.unique(exp)}}
    detail['raycast_fraction_max'] = worst_rc
    detail['cube_map_fraction_max'] = worst_cube
    detail['depth_fraction_max'] = worst_depth
    return worst_rc, worst_cube <= 0.01 and worst_depth <= 0.01, detail


def selftest_export(tmp: str) -> tuple[bool, str]:
    """The self-test's line: the UR5e sample exported as URDF and OpenCV YAML into tmp, the URDF compiled by MuJoCo
    and every optical frame posed at a random joint state against DexVantage, one YAML read back by
    cv2.FileStorage. (ok, text)."""
    import time

    import cv2
    import mujoco

    from ..core import camrender as CR
    from ..core.exporters import _common as C
    from ..core.robot.model import get_model
    from ..core.robot.samples import sample_project
    t0 = time.perf_counter()
    p = sample_project('ur5e')
    from ..core import exporters as X
    man = X.export(p, tmp, ['urdf', 'opencv_yaml'])
    bad = [f for f, r in man['formats'].items() if r['status'] != 'ok']
    if bad:
        return False, f'{", ".join(bad)} refused: {man["formats"][bad[0]].get("reason")}'
    root = ET.fromstring(open(os.path.join(tmp, 'ros2', 'rig.urdf'), encoding='utf-8').read())
    for old in root.findall('mujoco'):            # the file's own compiler element, replaced
        root.remove(old)
    mj = ET.SubElement(root, 'mujoco')
    ET.SubElement(mj, 'compiler', fusestatic='false', discardvisual='true', boundmass='1e-6', boundinertia='1e-9')
    md = os.path.join(tmp, 'ros2', 'meshes')
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'),
                                       {f: open(os.path.join(md, f), 'rb').read() for f in os.listdir(md)})
    d = mujoco.MjData(m)
    q = random_q(p, np.random.default_rng(0))
    for r in p.robots:
        for jn, v in zip(get_model(p, r.uid).joint_names, q[r.uid]):
            d.qpos[m.joint(jn).qposadr[0]] = v
    mujoco.mj_kinematics(m, d)
    worst = 0.0
    cams = C.rig_cameras(p, q)
    for c in cams:
        b = m.body(c.optical_frame).id
        worst = max(worst, float(np.linalg.norm(d.xpos[b] - CR.optical_pose(p, c.sensor, c.imager, q).position)))
    fs = cv2.FileStorage(os.path.join(tmp, 'calibration', 'opencv', cams[0].key + '.yaml'), cv2.FILE_STORAGE_READ)
    K = fs.getNode('camera_matrix').mat()
    fs.release()
    k_err = abs(float(K[0, 0]) - cams[0].camera.fx)
    ms = (time.perf_counter() - t0) * 1000
    if worst > 1e-9 or k_err > 0:
        return False, f'URDF frames {worst:.2e} m from DexVantage, OpenCV fx read back {k_err:.2e} px off'
    return True, (f'UR5e URDF in MuJoCo: {len(cams)} camera frames within {worst:.0e} m; OpenCV YAML read back '
                  f'exactly ({ms:.0f} ms)')


def bracket_step(ctx):
    """The UR5e sample's wrist bracket (its suggested type) exported with STEP at a 2 mm grid; the STEP read back by
    gmsh's OpenCASCADE kernel: one closed volume whose volume and bounding box equal the faceted mesh's (the mesh
    rebuilt at that grid, its volume by the divergence theorem)."""
    import gmsh

    from ..core import brackets as B
    from ..core.robot.samples import sample_project
    p = sample_project('ur5e')
    s = p.variant.sensors[0]
    s.mount.bracket = B.suggest_kind(p, s)[0]
    folder = os.path.join(ctx.tmp(), 'step_ur5e')
    row = _export(p, folder, 'brackets', {'step': True, 'step_res_mm': 2.0})
    step = next(f for f in row['files'] if f.endswith('.step'))
    spec = B.spec_for(p, s)
    srow = B.build(p, s, B.BracketSpec.from_dict(dict(spec.to_dict(), res_mm=2.0)), clear=False)
    V, F = srow['_mesh']
    tri = np.asarray(V, float)[np.asarray(F)]
    vol = float(np.einsum('ij,ij->i', tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)
    lo, hi = np.asarray(V).min(axis=0), np.asarray(V).max(axis=0)
    started = gmsh.isInitialized() if hasattr(gmsh, 'isInitialized') else False
    if not started:
        gmsh.initialize([], readConfigFiles=False)
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.model.add('readback')
        ents = gmsh.model.occ.importShapes(os.path.join(folder, step))
        gmsh.model.occ.synchronize()
        vols = [t for d, t in ents if d == 3]
        v_step = sum(gmsh.model.occ.getMass(3, t) for t in vols)
        bb = np.array(gmsh.model.getBoundingBox(-1, -1), float)
    finally:
        gmsh.model.remove()
        if not started:
            gmsh.finalize()
    rel = abs(v_step - vol) / abs(vol)
    bb_err = float(np.max(np.abs(np.r_[bb[:3] - lo, bb[3:] - hi])))
    return rel, len(vols) == 1 and bb_err <= 1e-3, {'triangles': int(len(F)), 'volume_mesh_mm3': vol,
                                                     'volume_step_mm3': v_step, 'volumes': len(vols),
                                                     'bounding_box_mm_error': bb_err, 'file': step}
