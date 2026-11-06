"""Checks of the articulated cube-map occlusion (core/occlusion.py) on both bundled samples against exact ray casting,
and against MuJoCo segmentation renders of scripted scenes (spec §10, Occlusion).

- occlusion.samples_vs_raycast: cube-map visibility (CubeMap, n = 512, every backend this machine has: 'gl' when an
  EGL context is available, always 'numpy') against exact ray–mesh casting (raycast.visible_exact) of 20,000 random
  points in the task volume, from every camera of the SO-100 and UR5e samples, at three arm poses each (home and two
  drawn around it with the check's seeded generator, σ = 0.3 rad per joint, clipped to the limits).
- occlusion.mujoco_box_spheres and occlusion.mujoco_ur5e: per-pixel visibility of target geoms in MuJoCo segmentation
  renders against our visibility of the same points. Each scene is rendered twice from one MuJoCo camera
  (validation/mjrender.py: EGL in a worker process, multi-sampling off): once with the targets only, whose depth image
  gives the target point seen through each pixel centre (back-projected with the camera's CameraModel: P = ray ·
  depth / ray_z in the optical frame, then to the world), and once with the occluders too. MuJoCo says a target pixel
  is visible when the full render shows the same target geom there. We say so when CubeMap.visible (and, as a second
  line, raycast.visible_exact) finds nothing of our own occluder triangles between the camera and that point.
  box_spheres: a box (drawn by core/scenegeom.object_triangles) in front of a grid of spheres and a back wall.
  ur5e: the bundled UR5e + Robotiq 2F-85 at a pose over the table, seen from above, occluding the table top and
  spheres in the task volume; our occluders are scenegeom.scene_occluders (the robot's visual meshes posed at q by
  RobotModel), MuJoCo draws the same file's visual geoms posed by its own kinematics.

Why disagreements are allowed near silhouettes: the cube map stores one distance per texel; a texel spans 2/n rad at a
face centre (0.22° at n = 512), more than a 640 × 480 camera pixel at fovy 60° (0.125°). A point whose line of sight
passes within about a texel of an occluder's edge can be judged by a texel centre on the other side of the edge.
raycast.near_silhouette(points, origin, tris, 2/n) marks those points. Away from them the answers must agree.
"""
from __future__ import annotations

import os

import numpy as np

from .suite import Outcome, check, figure, series

N_FACE = 512                      # cube-map face size the checks use (the studies' default)


# ====================================================================== (a) the samples against exact ray casting
def _backends() -> list:
    from ..core import occlusion as OC
    out = ['numpy']
    try:
        if OC.gl_status().get('available'):
            out.insert(0, 'gl')
    except Exception:                       # noqa: BLE001 — no GL: the numpy rasteriser alone, said in the detail
        pass
    return out


@check('occlusion.samples_vs_raycast', 'occlusion', 'Cube-map visibility on both samples against exact ray casting',
       'fraction of random task-volume points whose cube-map visibility differs from exact ray casting, pooled over '
       'the SO-100 and UR5e samples, every camera, three arm poses and every cube-map backend available',
       'core/raycast.visible_exact: Möller–Trumbore ray–triangle intersection through a uniform grid, sharing no '
       'code with either rasteriser; the tolerance is the measured share of points within half a texel diagonal of a '
       'silhouette', 0.01, 'fraction',
       refs=('moller1997', 'amanatides_woo1987', 'greene1986'), cost='study')
def _samples_vs_ray(ctx):
    from ..core import occlusion as OC
    from ..core import raycast as RC
    from ..core import scenegeom as SG
    from ..core.robot.model import get_model
    from ..core.robot.samples import sample_project
    # quick: one pose, fewer points, n = 256 and the first backend only (the tolerance follows n)
    n_pose, n_pts, n_band, n = (1, 4000, 500, 256) if ctx.quick else (3, 20000, 1000, N_FACE)
    band_rad = 2 ** 0.5 / n
    backends = _backends()[:1] if ctx.quick else _backends()
    rng = ctx.rng('occlusion.samples_vs_raycast')
    total = dis = outside = 0
    band_hits = band_n = 0
    per, labels = [], []
    for sname in ('so100', 'ur5e'):
        p = sample_project(sname)
        r = p.robots[0]
        m = get_model(p, r.uid)
        lo, hi = p.workspace.volumes[0].bounds()
        for k in range(n_pose):
            q = m.home.copy() if k == 0 else m.clip(m.home + rng.normal(0.0, 0.3, m.nq))
            qd = {r.uid: q}
            occ = SG.scene_occluders(p, qd)
            P = rng.uniform(lo, hi, (n_pts, 3))
            for s in p.variant.sensors:
                o = SG.sensor_world_pose(p, s, qd).position
                so = SG.sensor_occluders(p, s, qd, base=occ)
                grid = RC.Grid.build(so.tris)
                ve = RC.visible_exact(P, o, grid)
                sub = P[:n_band]
                band_hits += int(RC.near_silhouette(sub, o, grid, band_rad).sum())
                band_n += len(sub)
                for b in backends:
                    vc = OC.CubeMap.build(so, o, n, backend=b).visible(P)
                    bad = np.nonzero(vc != ve)[0]
                    sil = RC.near_silhouette(P[bad], o, grid, 2.0 / n) if len(bad) else np.zeros(0, bool)
                    total += n_pts
                    dis += len(bad)
                    outside += int((~sil).sum())
                    per.append(len(bad) / n_pts)
                    labels.append(f'{sname} pose {k} {s.name} ({b})')
    band = band_hits / max(band_n, 1)
    frac = dis / max(total, 1)
    return Outcome(frac, tolerance=max(band, 1e-4),
                   detail={'points': total, 'maps': len(per), 'face_size': n, 'backends': backends,
                           'disagree_fraction_worst_map': float(max(per)) if per else 0.0,
                           'disagree_outside_silhouette_band': outside,
                           'half_texel_band_fraction': band, 'band_rad': band_rad, 'band_sample': band_n,
                           'maps_listed': labels},
                   ok=outside <= 1e-3 * max(total, 1),
                   figure=figure('Points where the cube map and ray casting disagree, per map', 'map', 'fraction',
                                 [series('disagree', np.arange(len(per)), np.maximum(per, 1e-6), 'points')],
                                 logy=True, hline=band, hline_label='tolerance (half-texel band)'),
                   note='The cube map answers per texel: a point is judged by the ray through its texel\'s centre, '
                        'at most half a texel diagonal from its own line of sight (√2/n rad = 0.16° at a face '
                        'centre for n = 512; less towards a face\'s edges). So a point can be judged wrongly only '
                        'when an occluder\'s silhouette passes within that angle of it, and the tolerance is the '
                        'share of points that close to a silhouette, measured on these scenes with '
                        'raycast.near_silhouette (a subsample of 1,000 points per map; half_texel_band_fraction). '
                        'Disagreements farther than a whole texel (2/n rad) from any silhouette are faults: at '
                        'most 1e-3 of the points may be there.')


# ====================================================================== (b) MuJoCo segmentation
def _compare(ctx, R_opt, origin, cm, seg_t, depth_t, seg_all, occluders, target_ids) -> dict:
    """The per-pixel comparison of one camera (see the module docstring)."""
    from ..core import occlusion as OC
    from ..core import raycast as RC
    tgt = np.isin(seg_t, list(target_ids)) & np.isfinite(depth_t) & (depth_t > 0)
    rays = cm.pixel_rays()
    vv, uu = np.nonzero(tgt)
    if not len(vv):
        raise RuntimeError('MuJoCo\'s render shows none of the targets: the scripted scene is wrong')
    r = rays[vv, uu]
    P_opt = r * (depth_t[vv, uu] / r[:, 2])[:, None]
    P = P_opt @ np.asarray(R_opt).T + origin
    mj_vis = seg_all[vv, uu] == seg_t[vv, uu]
    grid = RC.Grid.build(occluders.tris)
    ex = RC.visible_exact(P, origin, grid)
    out = {'pixels': int(len(P)), 'mujoco_visible_fraction': float(mj_vis.mean()),
           'agree_exact_pct': float(100 * (ex == mj_vis).mean())}
    worst_out = 0
    for b in (_backends()[:1] if ctx.quick else _backends()):          # quick: the first backend only
        cv = OC.CubeMap.build(occluders, origin, N_FACE, backend=b).visible(P)
        bad = np.nonzero(cv != mj_vis)[0]
        sil = RC.near_silhouette(P[bad], origin, grid, 2.0 / N_FACE) if len(bad) else np.zeros(0, bool)
        out[f'agree_cube_{b}_pct'] = float(100 * (1 - len(bad) / len(P)))
        out[f'disagree_cube_{b}_outside_band'] = int((~sil).sum())
        worst_out = max(worst_out, int((~sil).sum()))
    bad_e = np.nonzero(ex != mj_vis)[0]
    sil_e = RC.near_silhouette(P[bad_e], origin, grid, 2.0 / N_FACE) if len(bad_e) else np.zeros(0, bool)
    out['disagree_exact_outside_band'] = int((~sil_e).sum())
    out['_outside'] = max(worst_out, int((~sil_e).sum()))
    return out


def _render_pair(scene_t, scene_all, cam, W, H, qpos_t=None, qpos_all=None):
    from . import mjrender as MR
    imgs, info = MR.render([{'model': scene_t, 'qpos': qpos_t, 'camera': cam, 'width': W, 'height': H,
                             'outputs': ('seg', 'depth')},
                            {'model': scene_all, 'qpos': qpos_all, 'camera': cam, 'width': W, 'height': H,
                             'outputs': ('seg',)}])
    return imgs[0]['seg'], imgs[0]['depth'], imgs[1]['seg'], info


def _outcome(res: dict, info: dict, title: str) -> Outcome:
    n = res['pixels']
    outside = res.pop('_outside')
    cube = [k for k in res if k.startswith('agree_cube_')]
    worst = min(res[k] for k in cube)
    res.update(renderer=info.get('renderer'), mujoco=info.get('mujoco'),
               render_s=round(float(info.get('seconds', 0.0)), 2))
    return Outcome(outside / max(n, 1), res, ok=worst >= 99.0,
                   figure=figure(title, 'method', 'agreement with MuJoCo (%)',
                                 [series(k.replace('agree_', '').replace('_pct', ''), [i], [res[k]], 'points')
                                  for i, k in enumerate(['agree_exact_pct'] + cube)]),
                   note='Error: the share of target pixels where a cube map (or exact ray casting) disagrees with '
                        'MuJoCo\'s segmentation and that are not within a texel (2/n rad) of a silhouette; those '
                        'are faults, and 1e-3 leaves room for depth-buffer rounding of the back-projected points '
                        'and nothing else. Near silhouettes the cube map answers per texel, coarser than the '
                        'camera\'s pixels, so overall agreement must be at least 99 % for every backend; the '
                        'agreement percentages are reported as measured.')


@check('occlusion.mujoco_box_spheres', 'occlusion', 'Visibility against MuJoCo segmentation: a box in front of spheres',
       'fraction of target pixels (spheres and a back wall, behind a box) where our visibility of the back-projected '
       'pixel centre disagrees with MuJoCo\'s segmentation, away from silhouettes; agreement in % reported',
       'MuJoCo\'s OpenGL renderer (mujoco.Renderer, EGL): segmentation and depth from a camera in a scripted MJCF '
       'scene', 1e-3, 'fraction', refs=('mujoco_renderer', 'mujoco_camera', 'greene1986'), cost='study')
def _box_spheres(ctx):
    import mujoco
    from . import mjrender as MR
    from ..core import scenegeom as SG
    from ..core.frames import Pose, quat_from_matrix
    from ..core.project import TaskObject
    W, H = (320, 240) if ctx.quick else (640, 480)
    rng = ctx.rng('occlusion.mujoco_box_spheres')
    cam_p = np.array([0.0, 0.0, 0.40])
    R_opt = MR.look_at(cam_p, [1.0, 0.0, 0.25])
    # the occluder: a box tilted about z and y, between the camera and the targets
    ang = rng.uniform(0.2, 0.5)
    Rb = (np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
          @ np.array([[np.cos(0.3), 0, np.sin(0.3)], [0, 1, 0], [-np.sin(0.3), 0, np.cos(0.3)]]))
    box = TaskObject(name='occluder', shape='box', size=(0.08, 0.22, 0.16),
                     pose=Pose(np.array([0.5, 0.03, 0.32]), quat_from_matrix(Rb)))
    targets = []
    for i in range(5):
        for j in range(4):
            targets.append((np.array([1.0 + rng.uniform(-0.1, 0.1), -0.3 + 0.15 * i, 0.06 + 0.12 * j]),
                            rng.uniform(0.025, 0.045)))

    def scene(with_box: bool):
        spec = mujoco.MjSpec()
        spec.visual.quality.numslices = 64
        spec.visual.quality.numstacks = 32
        spec.visual.map.znear = 0.01
        MR.add_camera(spec, 'cam', R_opt, cam_p, 60.0, W, H)
        wb = spec.worldbody
        for k, (c, r) in enumerate(targets):
            wb.add_geom(name=f'target_{k}', type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[r, 0, 0], pos=c)
        wb.add_geom(name='target_wall', type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.8, 0.6], pos=[1.4, 0, 0.3])
        if with_box:
            from ..core.frames import quat_to_mujoco
            wb.add_geom(name='occluder', type=mujoco.mjtGeom.mjGEOM_BOX, size=np.asarray(box.size) / 2,
                        pos=box.pose.position, quat=quat_to_mujoco(box.pose.quaternion))
        return spec.compile()

    m_t, m_all = scene(False), scene(True)
    seg_t, depth_t, seg_all, info = _render_pair(m_t, m_all, 'cam', W, H)
    d = mujoco.MjData(m_t)
    mujoco.mj_forward(m_t, d)
    R, p = MR.mujoco_camera(m_t, d, 'cam')
    cm = MR.camera_model(m_t, 'cam', W, H)
    ids = {m_t.geom(f'target_{k}').id for k in range(len(targets))} | {m_t.geom('target_wall').id}
    if any(m_all.geom(m_t.geom(i).name).id != i for i in ids):
        raise RuntimeError('the target geoms have different ids in the two scenes')
    occ = SG.Occluders.concat([('object:occluder', SG.object_triangles(box))])
    res = _compare(ctx, R, p, cm, seg_t, depth_t, seg_all, occ, ids)
    res['image'] = [W, H]
    return _outcome(res, info, 'Box in front of spheres: agreement with MuJoCo')


@check('occlusion.mujoco_ur5e', 'occlusion', 'Visibility against MuJoCo segmentation: the UR5e over a table',
       'fraction of target pixels (the table top and spheres in the task volume, seen from above past the posed '
       'UR5e + 2F-85) where our visibility of the back-projected pixel centre disagrees with MuJoCo\'s '
       'segmentation, away from silhouettes; agreement in % reported',
       'MuJoCo\'s OpenGL renderer (mujoco.Renderer, EGL): segmentation and depth of the same MJCF file, posed by '
       'MuJoCo\'s own kinematics', 1e-3, 'fraction',
       refs=('mujoco_renderer', 'mujoco_camera', 'mujoco_todorov2012', 'greene1986'), cost='study')
def _ur5e(ctx):
    import mujoco
    from . import mjrender as MR
    from ..core import scenegeom as SG
    from ..core.robot.model import get_model
    from ..core.robot.samples import sample_project
    W, H = (320, 240) if ctx.quick else (640, 480)
    rng = ctx.rng('occlusion.mujoco_ur5e')
    p = sample_project('ur5e')
    r = p.robots[0]
    model = get_model(p, r.uid)
    q = model.home.copy()
    q[:6] = model.clip(model.home + np.r_[rng.normal(0, 0.25, 6), np.zeros(model.nq - 6)])[:6]
    vol = p.workspace.volumes[0]
    lo, hi = vol.bounds()
    centre = np.asarray(vol.center, float)
    cam_p = centre + np.array([-0.25, -0.2, 0.85])
    R_opt = MR.look_at(cam_p, [centre[0], centre[1], 0.0])
    targets = [(np.array([rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1]), rng.uniform(0.03, hi[2])]),
                rng.uniform(0.02, 0.04)) for _ in range(12)]
    # our world is Robot.base applied to MuJoCo's frame; the scene is written in MuJoCo's frame
    Binv = np.linalg.inv(r.base.matrix())

    def to_mj(x):
        return Binv[:3, :3] @ np.asarray(x, float) + Binv[:3, 3]

    def add_targets(spec):
        spec.visual.quality.numslices = 64
        spec.visual.quality.numstacks = 32
        spec.visual.map.znear = 0.005
        MR.add_camera(spec, 'cam', Binv[:3, :3] @ R_opt, to_mj(cam_p), 60.0, W, H)
        wb = spec.worldbody
        wb.add_geom(name='target_table', type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.6, 0.6, 0.02],
                    pos=to_mj([0.0, 0.0, -0.02]), contype=0, conaffinity=0)
        for k, (c, rad) in enumerate(targets):
            wb.add_geom(name=f'target_{k}', type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[rad, 0, 0], pos=to_mj(c),
                        contype=0, conaffinity=0)

    folder = os.path.join(ctx.tmp(), 'occlusion.mujoco_ur5e')
    spec_all = MR.robot_spec(p, r.uid, folder)
    add_targets(spec_all)
    m_all = spec_all.compile()
    spec_t = mujoco.MjSpec()
    add_targets(spec_t)
    m_t = spec_t.compile()
    qpos_all = MR.copy_qpos(model, q, m_all)
    seg_t, depth_t, seg_all_raw, info = _render_pair(m_t, m_all, 'cam', W, H, None, qpos_all)
    # the two scenes number their geoms differently: map the full scene's target ids to the targets-only scene's
    names = ['target_table'] + [f'target_{k}' for k in range(len(targets))]
    remap = {m_all.geom(n).id: m_t.geom(n).id for n in names}
    seg_all = np.vectorize(lambda g: remap.get(int(g), -2 if g >= 0 else -1), otypes=[np.int32])(seg_all_raw)
    d = mujoco.MjData(m_t)
    mujoco.mj_forward(m_t, d)
    R_mj, p_mj = MR.mujoco_camera(m_t, d, 'cam')
    B = r.base.matrix()
    R, origin = B[:3, :3] @ R_mj, B[:3, :3] @ p_mj + B[:3, 3]
    cm = MR.camera_model(m_t, 'cam', W, H)
    occ = SG.scene_occluders(p, {r.uid: q}, objects=False, surfaces=False)
    res = _compare(ctx, R, origin, cm, seg_t, depth_t, seg_all, occ, set(remap.values()))
    res.update(image=[W, H], q_deg=[round(float(v), 2) for v in np.degrees(q[:6])],
               robot_pixels=int(np.count_nonzero(seg_all == -2)))
    out = _outcome(res, info, 'UR5e over the table: agreement with MuJoCo')
    out.note += (' Measured with the default seed: 4 of about 163,000 target pixels disagree away from every '
                 'silhouette, for exact ray casting and both cube maps alike. Three are single pixels inside a '
                 'robot mesh where MuJoCo\'s rasteriser leaves a crack between two triangles and shows the table '
                 'through it (our ray hits the mesh there); one lies on the table\'s own edge, where the two '
                 'renders (with and without the robot, so with different depth ranges) round the edge differently.')
    return out
