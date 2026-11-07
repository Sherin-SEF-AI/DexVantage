"""Mask export (spec §9.2): per-camera self-occlusion masks at a joint state, a mask atlas over a grasp set, and
depth-validity masks, as PNG images with a JSON index.

    @exporter('masks', ...)   options: q, imagers ('primary' | 'all'), scale (1.0), atlas (True), grasp_set (uid),
                              atlas_grasps (8, a default), atlas_scale (0.25, a default), depth (True),
                              cube_n (2048, a default)
    self_masks(project, cam, q, scale=1.0) -> (mask (H, W) bool, owner (H, W) int32, owners [str], Render)
    depth_mask(project, sensor, q) -> (code (H, W) uint8, info)

What a self-occlusion mask is. For each camera (each sensor's primary imager by default: the coverage study's
choice, the RGB imager of a depth camera), the image is the Camera view's renderer (core/camrender.py) at the
imager's stream size through its own lens model, of the robots posed at q, the task objects and the tables; the
camera's seat on its mount link is left out by the occlusion study's rule (core/scenegeom.sensor_occluders: the
mount link's triangles wholly within OWN_RADIUS_M of the imager, a default). A pixel is in the mask (255) when the
first thing its ray meets is a robot link, of any robot: the part of the image the robot itself hides. The owner
image (16-bit PNG) gives every pixel what it sees: 0 nothing (or a pixel with no ray in the lens), k ≥ 1 the
(k − 1)-th entry of masks.json's 'owners' list ('robot:<uid>:<link>', 'object:<uid>', 'surface:<uid>'), so a policy
can mask one link or the whole arm. The floor is not geometry in DexVantage and is not in the masks.

The atlas. Over a grasp set (options grasp_set, default the first set), the robot is posed at each grasp by the
grasp study's inverse kinematics (core/studies/grasp_visibility.solve_grasps with its settings; infeasible grasps are
skipped and counted), atlas_grasps of them (8, a default: each mask is a render; up to four times as many are
solved, evenly picked from the set, and the reachable ones are evenly kept), each camera rendered at atlas_scale of
its stream size (0.25, a default; the lens scaled by CameraModel.scaled): one tiled PNG per camera (row-major, tile
k is the k-th grasp in masks.json's atlas list) and a frequency image (the share of those grasps in which the robot
covers each pixel, 0–255).

Depth-validity masks. For each depth sensor the depth validity study can set up (core/studies/depth_validity.setup:
its resolution, imagers, min-Z, range and projector rules), every pixel of the depth (reference) imager at the
depth resolution looks along its ray to the first surface (the same renderer), and that point is tested by the
study's own evaluate() (its cube maps at DEPTH_CUBE_N = 2048 texels per face, a default: at the study's 512 a
texel spans more depth than the 1 mm visibility tolerance on far grazing surfaces, and 5 % of the overhead D435's
table pixels read occluded from the imager that sees them; the reference imager sees each pixel's point by
construction, since it is the first surface on that pixel's ray, so only the other imagers' lines of sight are
tested): the first condition it fails is the pixel's code (0 depth-valid, 1 fov, 2 min_z, 3 max_range,
4 occluded, 5 projector, 6 material, as depth_validity.REASONS) and 255 marks a pixel whose ray meets nothing the
project models (no depth there; the floor is not geometry in DexVantage, so a camera looking past the tables at
the floor reads 255 there). A second PNG holds 255 where the depth is valid.

Known limits: masks are geometric (the visual meshes, no cables, no sensor housings, no motion blur); edges are as
sharp as the pixel grid (no anti-aliasing) and, for a distorted lens, move by up to half a pixel of the renderer's
pinhole base image; the atlas poses only the grasp's robot (others at q) with the fingers at their home opening (as
the grasp study); the depth masks test the visible surface at the pixel's centre ray, so a pixel straddling an edge
takes the nearer surface's code.
"""
from __future__ import annotations

import math

import numpy as np

from ..jobs import StudyError
from . import _common as C
from . import exporter

DEFAULT_ATLAS_GRASPS = 8          # a default: grasps in the atlas (each is a render per camera)
DEFAULT_ATLAS_SCALE = 0.25        # a default: the atlas tiles' size as a share of the stream size
ATLAS_TRIES = 4                   # grasps solved per atlas grasp wanted (unreachable ones are skipped)
DEPTH_CUBE_N = 2048              # a default: cube-map texels per face for the depth masks' lines of sight
NO_SURFACE = 255
DIR = 'masks'


def _render(project, cam_model, pose, sensor, q, scale: float = 1.0):
    """(Render, Geometry) of the scene from an imager (seat left out by the occlusion study's rule)."""
    from .. import camrender as CR
    geom = CR.gather(project, q)
    geom = CR.without_seat(geom, project, sensor, np.asarray(pose.position, float)[None])
    W = max(1, int(round(cam_model.width * scale)))
    H = max(1, int(round(cam_model.height * scale)))
    return CR.render(geom, cam_model, pose, W, H, floor=False), geom


def self_masks(project, cam, q, scale: float = 1.0):
    """(mask (H, W) bool, owner (H, W) int32 0 = nothing else 1 + index into owners, owners, Render) of a
    RigCamera (module docstring)."""
    from .. import camrender as CR
    from ..frames import Pose
    pose = Pose.from_matrix(cam.T_world_optical)
    r, geom = _render(project, cam.camera, pose, cam.sensor, q, scale)
    hit = (r.tri >= 0) & r.valid
    owner = np.zeros(r.tri.shape, np.int32)
    owner[hit] = geom.owner[r.tri[hit]] + 1
    return hit & (r.kind == CR.KIND_ROBOT), owner, list(geom.owners), r


def depth_mask(project, sensor, q, cube_n: int = DEPTH_CUBE_N):
    """(code (H, W) uint8, info dict) of a depth sensor's reference imager (module docstring); raises
    depth_validity.DepthSetupError when the study cannot set the sensor up."""
    from ..scenegeom import sensor_occluders, sensor_world_pose
    from ..studies import depth_validity as DV
    ds = DV.setup(project, sensor)
    body = sensor_world_pose(project, sensor, q)
    ref_pose = DV.imager_world(body, ds.ref)
    origins = np.array([DV.imager_world(body, im).position for im in ds.needed])
    from .. import camrender as CR
    geom = CR.without_seat(CR.gather(project, q), project, sensor, origins)
    cam = ds.ref.camera
    r = CR.render(geom, cam, ref_pose, floor=False)
    hit = (r.tri >= 0) & r.valid & np.isfinite(r.dist)
    code = np.full(r.tri.shape, NO_SURFACE, np.uint8)
    if hit.any():
        rays, _ok = cam.unproject(np.stack(np.nonzero(hit)[::-1], 1).astype(float), with_ok=True)
        Rw = ref_pose.matrix()[:3, :3]
        P = np.asarray(ref_pose.position, float) + (rays @ Rw.T) * r.dist[hit].astype(float)[:, None]
        occ = sensor_occluders(project, sensor, q, origins=origins)
        ev = DV.evaluate(ds, body, P, occluders=occ, mats=DV.material_objects(project, ds.technology),
                         project=project, n=cube_n)
        vis = np.asarray(ev['visible'], bool).copy()
        for k, im in enumerate(ds.needed):
            if im is ds.ref or im.name == ds.ref.name:
                vis[:, k] = True                   # the pixel's own first surface: seen by construction
        conds = [ev['in_fov'], ev['min_z_ok'], ev['max_ok'], vis.all(axis=1), ev['proj_ok'], ev['material'] == 0]
        c = np.zeros(len(P), np.uint8)
        for k in range(len(conds) - 1, -1, -1):
            c[~np.asarray(conds[k], bool)] = k + 1
        code[hit] = c
    info = {'imager': ds.ref.name, 'resolution': [cam.width, cam.height], 'technology': ds.technology,
            'min_z_m': ds.min_z_m, 'max_range_m': ds.max_range_m, 'needed_imagers': [im.name for im in ds.needed],
            'camera': cam.to_dict(), 'T_world_optical': C.pose_record(ref_pose.matrix()),
            'resolution_note': ds.resolution_note, 'cube_n': int(cube_n)}
    return code, info


def _png(out_dir: str, rel: str, img: np.ndarray) -> str:
    import cv2
    ok, buf = cv2.imencode('.png', img)
    if not ok:
        raise StudyError(f'Could not encode {rel} as PNG.')
    return C.write_bytes(out_dir, rel, buf.tobytes())


def _atlas_q(project, options, notes):
    """(grasp set, [(grasp id, q dict)], counts) for the atlas, or (None, [], {}) when there is no grasp set."""
    from .. import grasps as GR
    from ..robot.model import RobotModelError, get_model
    from ..studies import grasp_visibility as GV
    sets = GR.sets(project)
    if not sets:
        return None, [], {}
    want = options.get('grasp_set') or ''
    gs = GR.get_set(project, want) if want else sets[0]
    if gs is None:
        raise StudyError('The grasp set chosen for the mask atlas is not in the project any more: choose another.')
    robot = project.robot(gs.get('robot_uid', ''))
    poses = list(gs.get('poses') or [])
    if robot is None or not poses:
        notes.append(f'The grasp set "{gs.get("name")}" has no grasps for a robot of the project: no atlas.')
        return gs, [], {}
    try:
        model = get_model(project, robot.uid)
    except RobotModelError as e:
        raise StudyError(f'{robot.name} cannot be posed for the atlas: {e}') from None
    n = int(options.get('atlas_grasps') or DEFAULT_ATLAS_GRASPS)
    if n < 1:
        raise StudyError('The number of grasps in the mask atlas must be at least 1.')
    # IK is cheap next to a render: solve up to ATLAS_TRIES × n grasps evenly picked, keep n of the reachable ones
    m_try = min(len(poses), ATLAS_TRIES * n)
    sel = np.unique(np.linspace(0, len(poses) - 1, m_try).round().astype(int))
    st = GV.settings_of(project)
    q0, _qt = C.joint_state(project, options)
    hold = np.asarray(q0.get(robot.uid, model.home), float)
    sub = dict(gs, poses=[poses[i] for i in sel])
    try:
        link, tcp, targets = GR.ik_targets(project, sub)
    except GR.GraspError as e:
        raise StudyError(str(e)) from None
    b = GV.solve_grasps(project, robot, model, link, targets, tcp, st, hold, None, notes)
    ok = np.flatnonzero(np.asarray(b.ok, bool))
    keep = ok[np.unique(np.linspace(0, len(ok) - 1, min(n, len(ok))).round().astype(int))] if len(ok) else ok
    out = []
    for i in keep:
        qd = dict(q0)
        qd[robot.uid] = np.asarray(b.q[i], float)
        out.append((str(sub['poses'][i].get('id', i)), qd))
    return gs, out, {'tried': int(len(sel)), 'reachable': int(len(ok)), 'used': len(out), 'set_size': len(poses),
                     'robot': robot.name}


@exporter('masks', 'Self-occlusion and depth-validity masks', needs_robot=True,
          description='Per camera, the pixels the robot hides at the pose (PNG, with a per-pixel owner image), a mask '
                      'atlas over a grasp set, and per depth sensor the depth-validity code of every pixel.')
def export_masks(project, out_dir, options, ctx):
    imagers = str(options.get('imagers') or 'primary')
    if imagers not in ('all', 'primary'):
        raise StudyError('The imagers option must be "all" or "primary".')
    scale = float(options.get('scale') or 1.0)
    a_scale = float(options.get('atlas_scale') or DEFAULT_ATLAS_SCALE)
    if not (0 < scale <= 1 and 0 < a_scale <= 1):
        raise StudyError('The mask scales must be above 0 and at most 1 (the stream size).')
    q, q_text = C.joint_state(project, options)
    cams = C.rig_cameras(project, q, imagers)
    C.need_cameras(cams, 'The mask export')
    files, notes = [], []
    owners_all: list = []
    index = {'schema': 'dexvantage.masks/1', 'joint_state': q_text,
             'q': {k: [float(x) for x in v] for k, v in q.items()},
             'convention': 'self: 255 where the first surface a pixel sees is a robot link, else 0. owner: 16-bit, 0 '
                           'nothing, k the (k - 1)-th entry of owners. depth: 0 depth-valid, 1 fov, 2 min_z, 3 '
                           'max_range, 4 occluded, 5 projector, 6 material, 255 no surface. Pixel centres at integers '
                           '(OpenCV); row 0 at the top.',
             'cameras': [], 'depth': [], 'atlas': None}
    want_atlas = options.get('atlas', True)
    want_depth = options.get('depth', True)
    steps = len(cams) + (len(cams) if want_atlas else 0) + 1
    k = 0
    for c in cams:
        ctx.report(k / steps, f'Mask of {c.title}')
        k += 1
        if ctx.cancelled():
            raise StudyError('The mask export was cancelled.')
        mask, owner, owners, r = self_masks(project, c, q, scale)
        remap = np.zeros(len(owners) + 1, np.int32)
        for i, o in enumerate(owners):
            if o not in owners_all:
                owners_all.append(o)
            remap[i + 1] = owners_all.index(o) + 1
        owner = remap[owner]
        if owner.max(initial=0) > 65535:
            raise StudyError('The scene has more than 65 535 owners, which a 16-bit owner image cannot hold.')
        f1 = _png(out_dir, f'{DIR}/self/{c.key}.png', (mask * 255).astype(np.uint8))
        f2 = _png(out_dir, f'{DIR}/self/{c.key}_owner.png', owner.astype(np.uint16))
        files += [f1, f2]
        index['cameras'].append({'name': c.key, 'sensor': c.sensor.name, 'sensor_uid': c.sensor.uid,
                                 'imager': c.imager.name, 'mask': f1, 'owner': f2,
                                 'size': [int(mask.shape[1]), int(mask.shape[0])], 'scale': scale,
                                 'camera': r.camera.to_dict(), 'T_world_optical': C.pose_record(c.T_world_optical),
                                 'robot_fraction': float(mask.sum() / max(1, int(r.valid.sum()))),
                                 'renderer_path': r.path})
        if c.camera.model != 'pinhole':
            notes.append(f'{c.title}: a {c.camera.model} lens is drawn through a pinhole base image, so mask edges can '
                         'move by up to half a base pixel.')
    index['owners'] = owners_all
    # ---------------------------------------------------------------- the atlas
    if want_atlas:
        gs, poses, counts = _atlas_q(project, options, notes)
        if gs is None:
            notes.append('The project has no grasp set, so no mask atlas is written (make one on the Tasks page).')
        elif poses:
            at = {'grasp_set': {'uid': gs.get('uid'), 'name': gs.get('name')}, **counts, 'scale': a_scale,
                  'grasps': [g for g, _q in poses], 'cameras': [], 'default_grasps': DEFAULT_ATLAS_GRASPS}
            n = len(poses)
            cols = int(math.ceil(math.sqrt(n)))
            rows = int(math.ceil(n / cols))
            for c in cams:
                ctx.report(k / steps, f'Atlas of {c.title}')
                k += 1
                tiles = []
                for _gid, qd in poses:
                    if ctx.cancelled():
                        raise StudyError('The mask export was cancelled.')
                    cc = [x for x in C.rig_cameras(project, qd, imagers) if x.key == c.key]
                    if not cc:
                        continue
                    m, _o, _ow, _r = self_masks(project, cc[0], qd, a_scale)
                    tiles.append(m)
                if not tiles:
                    continue
                h, w = tiles[0].shape
                img = np.zeros((rows * h, cols * w), np.uint8)
                for t, m in enumerate(tiles):
                    rr, cc_ = divmod(t, cols)
                    img[rr * h:(rr + 1) * h, cc_ * w:(cc_ + 1) * w] = m * 255
                freq = np.round(255.0 * np.mean(np.stack(tiles), axis=0)).astype(np.uint8)
                fa = _png(out_dir, f'{DIR}/atlas/{c.key}_atlas.png', img)
                ff = _png(out_dir, f'{DIR}/atlas/{c.key}_frequency.png', freq)
                files += [fa, ff]
                at['cameras'].append({'name': c.key, 'atlas': fa, 'frequency': ff, 'tile': [w, h],
                                      'grid': [cols, rows], 'tiles': len(tiles)})
            index['atlas'] = at
            notes.append(f'Mask atlas: {counts["used"]} grasps of "{gs.get("name")}" ({counts["reachable"]} of the '
                         f'{counts["tried"]} tried of its {counts["set_size"]} are reachable by inverse kinematics); '
                         f'tiles at {a_scale:g} of the stream size (by default {DEFAULT_ATLAS_GRASPS} grasps at '
                         f'{DEFAULT_ATLAS_SCALE:g}).')
        else:
            notes.append(f'No grasp of "{gs.get("name")}" could be reached by inverse kinematics, so the atlas is '
                         'empty.')
    # ---------------------------------------------------------------- depth validity
    if want_depth:
        from ..studies import depth_validity as DV
        ctx.report(k / steps, 'Depth-validity masks')
        for s in DV.depth_sensors(project):
            try:
                code, info = depth_mask(project, s, q, int(options.get('cube_n') or DEPTH_CUBE_N))
            except DV.DepthSetupError as e:
                notes.append(f'No depth-validity mask for {s.name}: {e}')
                continue
            key = f'{C.frame_name(s.name)}__{C.frame_name(info["imager"])}'
            f1 = _png(out_dir, f'{DIR}/depth/{key}.png', code)
            f2 = _png(out_dir, f'{DIR}/depth/{key}_valid.png', ((code == 0) * 255).astype(np.uint8))
            files += [f1, f2]
            n = max(1, code.size)
            info.update(name=key, sensor=s.name, sensor_uid=s.uid, code=f1, valid=f2,
                        fractions={**{'valid': float((code == 0).sum() / n)},
                                   **{rs: float((code == i + 1).sum() / n) for i, rs in enumerate(DV.REASONS)},
                                   'no_surface': float((code == NO_SURFACE).sum() / n)})
            index['depth'].append(info)
    files.append(C.write_json(out_dir, f'{DIR}/masks.json', index))
    return {'files': files, 'notes': list(dict.fromkeys(notes))}
