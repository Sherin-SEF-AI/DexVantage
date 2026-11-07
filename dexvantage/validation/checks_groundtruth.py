"""Checks of the MuJoCo ground truth (core/sim/scene.py, core/sim/render.py) and of the study that compares it with
the predictions (core/studies/groundtruth.py); spec §9.1 and §10.

- groundtruth.lens_remap_fiducials: the distortion remap. For each of the six lens models (pinhole, Brown-Conrady,
  Kannala-Brandt, equidistant, equisolid, stereographic) a scripted MJCF scene holds 35 small spheres (fiducials)
  spread over the lens's image, 0.4 to 1.2 m away, each about 6 px in radius. render.render_lens renders MuJoCo
  segmentation through the lens (a wide pinhole, or cube faces, remapped through CameraModel). Each sphere's rendered
  mask is compared with the mask CameraModel predicts (the pixel centres whose ray lies within the sphere's angular
  radius of its centre): the distance between the two centroids. The centroid's distance from CameraModel.project of
  the sphere's centre is reported too (it carries the sphere image's own bias: the image of a sphere is not centred
  on its centre's projection).
- groundtruth.scene_intrinsics: every scene camera of the three samples, read back from the compiled MuJoCo model
  (validation/mjrender.camera_model: fx, fy from cam_intrinsic and cam_sensorsize, the principal offset with the sign
  measured on renders), against the imager's CameraModel.
- groundtruth.scene_vs_camrender: MuJoCo's segmentation of every sample camera against core/camrender's (the Camera
  view's) rasterisation of the same scene: the share of pixels whose owner (robot link, object, surface, nothing)
  differs. Both sample pixel centres, so they differ only at edges where the two rasterisers' edge rules decide a
  pixel differently.
- groundtruth.ur5e_contacts and groundtruth.ur5e_depth: the study on the UR5e sample (50 grasps, 10 in a quick run):
  the agreement of the contact points in view and of the depth study's voxels with the renders, against stated floors.
"""
from __future__ import annotations

import math

import numpy as np

from .suite import Outcome, check, figure, series

R_PX = 6.0                         # fiducial radius in pixels at its place in the image
LENSES = (                         # (name, CameraModel arguments at 640 × 480)
    ('pinhole', (500.0, 500.0, 321.3, 238.2, 'pinhole', ())),
    ('brown', (480.0, 482.0, 318.0, 241.5, 'brown', (-0.28, 0.09, 0.0008, -0.0006, -0.012))),
    ('kb', (260.0, 260.0, 319.5, 239.5, 'kb', (0.03, -0.006, 0.001, -0.0002))),
    ('equidistant', (200.0, 200.0, 319.5, 239.5, 'equidistant', ())),
    ('equisolid', (210.0, 210.0, 320.5, 238.5, 'equisolid', ())),
    ('stereographic', (170.0, 170.0, 319.5, 239.5, 'stereographic', ())),
)
W, H = 640, 480
REMAP_TOL_PX = 0.4
CONTACT_FLOOR = 0.85               # stated floor: contact agreement on the UR5e sample (measured 0.913, 50 grasps)
OCCLUSION_MAX = 0.02               # and at most this share of the points in view disagreeing on occlusion
DEPTH_FLOOR = 0.99                 # stated floor: depth voxel agreement on the UR5e sample (measured 0.998)


def _fiducials(cam, rng, n_x=7, n_y=5):
    """(points (N, 3) optical, radii (N,)): spheres spread over the image, each R_PX px in radius at its place."""
    pts, rad = [], []
    for i in range(n_x):
        for j in range(n_y):
            u = (i + 0.5) / n_x * cam.width - 0.5 + rng.uniform(-0.2, 0.2) * cam.width / n_x
            v = (j + 0.5) / n_y * cam.height - 0.5 + rng.uniform(-0.2, 0.2) * cam.height / n_y
            ray = cam.unproject(np.array([[u, v]]))[0]
            if not np.isfinite(ray).all():
                continue
            d = rng.uniform(0.4, 1.2)
            th = 1e-4                                  # local px per radian, by a small turn about the image radius
            perp = np.cross(ray, [0.0, 0.0, 1.0]) if abs(ray[2]) < 0.999 else np.array([1.0, 0.0, 0.0])
            perp /= np.linalg.norm(perp)
            r2 = ray * math.cos(th) + np.cross(perp, ray) * math.sin(th)
            dens = np.linalg.norm(cam.project(r2[None])[0][0] - cam.project(ray[None])[0][0]) / th
            pts.append(ray * d)
            rad.append(d * math.sin(R_PX / dens))
    return np.array(pts), np.array(rad)


def _remap(ctx) -> dict:
    key = ('groundtruth.remap', ctx.quick)
    if key in ctx.cache:
        return ctx.cache[key]
    import mujoco

    from ..core.optics import CameraModel
    from ..core.sim.render import lens_plan, render_lens
    from . import mjrender as MR
    rng = ctx.rng('groundtruth.lens_remap_fiducials')
    T = np.eye(4)
    T[:3, :3] = MR.look_at([0, 0, 2], [0, 0, 0])
    T[:3, 3] = [0, 0, 2]
    rows, info = [], {}
    with MR.Session() as sess:
        for name, (fx, fy, cx, cy, model, dist) in LENSES:
            cam = CameraModel(W, H, fx, fy, cx, cy, model, dist)
            P, rad = _fiducials(cam, rng, *((5, 4) if ctx.quick else (7, 5)))
            Xw = P @ T[:3, :3].T + T[:3, 3]
            xml = ('<mujoco><visual><quality numslices="128" numstacks="64"/><map znear="0.001"/></visual>'
                   '<worldbody><camera name="cam" pos="0 0 2"/>')
            for k, (c, r) in enumerate(zip(Xw, rad)):
                xml += f'<geom name="f{k}" type="sphere" size="{r:.9g}" pos="{c[0]:.9g} {c[1]:.9g} {c[2]:.9g}"/>'
            m = mujoco.MjModel.from_xml_string(xml + '</worldbody></mujoco>')
            out = render_lens(sess, m, 'cam', T, cam, ('seg',))
            info = out['info']
            seg = out['seg']
            rays = np.nan_to_num(cam.pixel_rays())
            plan = lens_plan(cam)
            for k in range(len(P)):
                gid = m.geom(f'f{k}').id
                vm, um = np.nonzero(seg == gid)
                dist_k = float(np.linalg.norm(P[k]))
                al = math.asin(rad[k] / dist_k)
                pred = (rays @ (P[k] / dist_k)) > math.cos(al)
                vp, up = np.nonzero(pred)
                if not len(um) or not len(up):
                    rows.append((name, np.nan, np.nan, 0.0, out['path']))
                    continue
                c_r = np.array([um.mean(), vm.mean()])
                c_p = np.array([up.mean(), vp.mean()])
                uvc = cam.project(P[k][None])[0][0]
                mk = seg == gid
                iou = float((pred & mk).sum() / max((pred | mk).sum(), 1))
                rows.append((name, float(np.linalg.norm(c_r - c_p)), float(np.linalg.norm(c_r - uvc)), iou,
                             f'{out["path"]} ({len(plan.subs)} × {plan.subs[0].K.width} × {plan.subs[0].K.height})'))
    res = {'rows': rows, 'info': info}
    ctx.cache[key] = res
    return res


@check('groundtruth.lens_remap_fiducials', 'groundtruth', 'Distortion remap against the lens model, six lens models',
       'largest distance between the centroid of a sphere\'s rendered mask (MuJoCo segmentation remapped through '
       'CameraModel by core/sim/render.py) and the centroid of the mask CameraModel predicts (pixel centres whose ray '
       'lies within the sphere\'s angular radius); 35 spheres per lens model, six models',
       'MuJoCo\'s OpenGL renderer through EGL on a scripted scene of spheres, the pinhole renders resampled with '
       'cv2.remap', REMAP_TOL_PX, 'px', refs=('mujoco_camera_317', 'mujoco_renderer', 'opencv_remap',
                                               'kannala_brandt2006'), cost='study')
def _lens_remap(ctx):
    r = _remap(ctx)
    rows = r['rows']
    err = np.array([x[1] for x in rows], float)
    iou = np.array([x[3] for x in rows], float)
    missing = int(np.count_nonzero(~np.isfinite(err)))
    by = {}
    for name, *_ in LENSES:
        e = np.array([x[1] for x in rows if x[0] == name], float)
        c = np.array([x[2] for x in rows if x[0] == name], float)
        path = next((x[4] for x in rows if x[0] == name and isinstance(x[4], str)), '')
        by[name] = {'worst_px': float(np.nanmax(e)) if np.isfinite(e).any() else None,
                    'median_px': float(np.nanmedian(e)) if np.isfinite(e).any() else None,
                    'centre_vs_project_max_px': float(np.nanmax(c)) if np.isfinite(c).any() else None,
                    'path': path}
    return Outcome(float(np.nanmax(err)) if missing < len(err) else float('nan'),
                   {'fiducials': len(rows), 'missing_masks': missing, 'by_lens': by,
                    'min_mask_iou': float(np.nanmin(iou)), 'renderer': r['info'].get('renderer'),
                    'mujoco': r['info'].get('mujoco'), 'device': r['info'].get('device')},
                   ok=missing == 0 and float(np.nanmin(iou)) >= 0.85,
                   figure=figure('Rendered against predicted fiducial masks, by lens model', 'lens model',
                                 'largest centroid distance (px)',
                                 [series('worst |Δ centroid|', np.arange(len(LENSES)),
                                         [by[n]['worst_px'] or np.nan for n, *_ in LENSES], 'bars')],
                                 kind='bars', categories=[n for n, *_ in LENSES], hline=REMAP_TOL_PX,
                                 hline_label='tolerance'),
                   note='Both masks sample the same pixel centres through the same rays, so they differ only where a '
                        'boundary pixel goes the other way: MuJoCo draws a sphere as a 128 × 64 polygon mesh (0.10 px '
                        'measured on the pinhole, as in optics.mujoco_fiducial_masks), and the remap samples the base '
                        'image at its nearest pixel, which moves a boundary by up to half a base pixel; the base is at '
                        'least twice as dense as the lens image (render.OVERSAMPLE = 2, a default), so that is at most '
                        'a quarter of a lens pixel at a boundary and less on a centroid of about 110 pixels (measured '
                        'up to 0.33 px with OVERSAMPLE = 1). 0.4 px allows both; a '
                        'principal point off by a pixel, a focal length off by 0.3 %, a wrong remap direction or a '
                        'wrong cube face would each show as one pixel or more. The centroid against '
                        'CameraModel.project of the centre (centre_vs_project_max_px) adds the sphere image\'s own '
                        'offset from its centre\'s projection. Every mask must be found and overlap its prediction by '
                        'at least 85 % (IoU; a 6 px sphere has about 110 pixels, 38 of them on its boundary).')


def _sample_scenes(ctx) -> dict:
    key = 'groundtruth.samples'
    if key not in ctx.cache:
        from ..core.robot.samples import sample_project
        from ..core.sim.scene import build_scene
        out = {}
        for name in ('ur5e', 'so100', 'dexhand'):
            p = sample_project(name)
            out[name] = (p, build_scene(p))
        ctx.cache[key] = out
    return ctx.cache[key]


@check('groundtruth.scene_intrinsics', 'groundtruth', 'Scene cameras against the sensors\' lens models',
       'largest difference in fx, fy, cx or cy (px) between each MuJoCo scene camera, read back from the compiled '
       'model (cam_intrinsic, cam_sensorsize; the principal offset\'s measured sign), and its imager\'s CameraModel; '
       'every imager of the UR5e, SO-100 and dexhand samples',
       'the compiled MuJoCo model (mujoco.MjSpec.compile of the camera\'s sensorsize, focalpixel, principalpixel and '
       'resolution)', 1e-3, 'px', refs=('mujoco_camera_317',))
def _intrinsics(ctx):
    from . import mjrender as MR
    worst, rows = 0.0, []
    for name, (_p, b) in _sample_scenes(ctx).items():
        for c in b.cameras.values():
            cm = MR.camera_model(b.model, c.name, c.model.width, c.model.height)
            d = max(abs(cm.fx - c.model.fx), abs(cm.fy - c.model.fy), abs(cm.cx - c.model.cx),
                    abs(cm.cy - c.model.cy))
            worst = max(worst, d)
            rows.append({'sample': name, 'camera': c.name, 'size': [c.model.width, c.model.height],
                         'error_px': float(d)})
    return Outcome(worst, {'cameras': len(rows), 'rows': rows},
                   note='MuJoCo stores the intrinsics as float32 lengths (cam_intrinsic = focalpixel · sensorsize / '
                        'resolution): a relative rounding of 6 × 10⁻⁸ on a focal length of 1 400 px is 10⁻⁴ px. 10⁻³ '
                        'px allows that; a principal point taken with the wrong sign is off by twice its offset.')


@check('groundtruth.scene_vs_camrender', 'groundtruth', 'MuJoCo segmentation against the Camera view\'s rasteriser',
       'share of pixels whose owner (robot link, object, surface or nothing) differs between MuJoCo\'s segmentation '
       'of each sample camera (core/sim/render.py) and core/camrender\'s rasterisation of the same scene at the home '
       'pose; every camera of the UR5e, SO-100 and dexhand samples at half their stream size',
       'MuJoCo\'s OpenGL renderer through EGL (segmentation, no multi-sampling) on the scene built by '
       'core/sim/scene.py', 1e-3, 'fraction of pixels', refs=('mujoco_renderer', 'mujoco_camera_317'), cost='study')
def _vs_camrender(ctx):
    from ..core import camrender as CR
    from ..core import library as LB
    from ..core.sim.render import Renderer
    rows, diff, total, rng_err, far_n, agree_n = [], 0, 0, 0.0, 0, 0
    scale = 0.25 if ctx.quick else 0.5
    for name, (p, _b) in _sample_scenes(ctx).items():
        with Renderer(p) as r:
            sensors = [s for s in p.variant.sensors if s.enabled and s.kind in ('camera', 'depth')]
            imgs = r.render([{'sensor_uid': s.uid, 'outputs': ('seg', 'depth'), 'scale': scale} for s in sensors])
            info = r.info
        g = CR.gather(p, None)
        for s, im in zip(sensors, imgs):
            c = r.build.camera(s.uid)
            cam = im['model']
            imager = next(i for i in LB.sensor_imagers(s, (c.model.width, c.model.height)) if i.name == c.imager)
            pose = CR.optical_pose(p, s, imager, None)
            rr = CR.render(g, cam, pose, cam.width, cam.height, floor=False)
            own = np.where(rr.tri >= 0, g.owner[np.clip(rr.tri, 0, None)], -1)
            names = np.array(list(g.owners) + [''])
            pred = names[own]
            mj_names = np.array([':'.join(o) for _k, o in sorted(im['seg_owner'].items())] + [''])
            mj = mj_names[np.where(im['seg'] >= 0, im['seg'], len(mj_names) - 1)]
            d = int((pred != mj).sum())
            hit = (rr.tri >= 0) & (im['seg'] >= 0) & (pred == mj)
            dr = np.abs(rr.dist[hit] - im['range'][hit])
            e = float(dr.max()) if hit.any() else 0.0
            rng_err = max(rng_err, e)
            far_n += int((dr > 1e-3).sum())
            agree_n += int(hit.sum())
            diff += d
            total += pred.size
            rows.append({'sample': name, 'sensor': s.name, 'size': [cam.width, cam.height], 'differ': d,
                         'share': d / pred.size, 'range_max_m': e})
    return Outcome(diff / max(total, 1), {'pixels': total, 'differ': diff, 'rows': rows,
                                          'range_max_m_where_owners_agree': rng_err,
                                          'range_over_1mm_share': far_n / max(agree_n, 1),
                                          'renderer': info.get('renderer'), 'mujoco': info.get('mujoco')},
                   ok=far_n <= 1e-3 * max(agree_n, 1),
                   note='Both sample each pixel at its centre (MuJoCo with multi-sampling off), so a pixel differs '
                        'only where a triangle edge passes within rounding of a pixel centre and the two edge rules '
                        '(OpenGL\'s top-left rule, camrender\'s inclusive test) decide it differently: a few pixels '
                        'per image (measured 0 to 1.8 × 10⁻⁴ of the pixels). 10⁻³ allows that; a camera off by a '
                        'pixel, or a link at the wrong pose, would differ along every edge (about 10⁻² here). Where '
                        'the owners agree, at most 10⁻³ of the pixels may differ in range along the pixel ray by more '
                        'than 1 mm (they agree to float32 rounding except where a pixel sees one link\'s silhouette '
                        'through two of its triangles, up to 9 mm on the SO-100\'s third-person camera).')


def _study(ctx, sample: str = 'ur5e') -> dict:
    key = ('groundtruth.study', ctx.quick, sample)
    if key not in ctx.cache:
        from ..core import jobs
        from ..core.robot.samples import sample_project
        p = sample_project(sample)
        p.analysis.studies['groundtruth'] = {'grasps': 10 if ctx.quick else 50, 'thumbs': 0}
        ctx.cache[key] = jobs.run_study('groundtruth', p)
    return ctx.cache[key]


@check('groundtruth.ur5e_contacts', 'groundtruth', 'Predicted against rendered contact visibility, UR5e sample',
       '1 − the agreement of the grasp study\'s predicted contact-point visibility with the rendered one (contact '
       'points in view, both cameras, home, a 50-grasp sample of the set (10 in a quick run) and 8 frames of the '
       'scripted pick trajectory); also at most '
       f'{OCCLUSION_MAX:.0%} of them may disagree on occlusion away from an edge, on the UR5e and on the SO-100 '
       'sample (whose agreement is reported beside it, not held to the floor)',
       'MuJoCo segmentation and depth of the same scene and joint states (core/studies/groundtruth.py)',
       1 - CONTACT_FLOOR, 'fraction of points', refs=('mujoco_renderer', 'mujoco_camera_317'), cost='study')
def _contacts(ctx):
    doc = _study(ctx)
    c = doc['contacts']
    t, w = c['totals'], c['why']
    occ = w['occlusion'] / max(t['n'], 1)
    so = _study(ctx, 'so100')['contacts']
    so_t = so['totals']
    so_occ = so['why']['occlusion'] / max(so_t['n'], 1)
    return Outcome(1.0 - float(t['agreement']),
                   {'agreement': t['agreement'], 'iou': t['iou'], 'confusion': {k: t[k] for k in ('tp', 'fp', 'fn',
                                                                                                  'tn', 'n')},
                    'why': w, 'occlusion_share': occ, 'grasp_level': c['grasp_level'], 'cases': c['cases'],
                    'by_kind': c.get('by_kind'),
                    'renderer': doc['render'].get('renderer'), 'frames_per_s': doc['render'].get('frames_per_s'),
                    'so100': {'agreement': so_t['agreement'], 'iou': so_t['iou'], 'why': so['why'],
                              'occlusion_share': so_occ, 'grasp_level': so['grasp_level']}},
                   ok=occ <= OCCLUSION_MAX and so_occ <= OCCLUSION_MAX,
                   note=f'The floor is {CONTACT_FLOOR:.0%} agreement (measured 92.8 % over home, 50 grasps and 8 '
                        'trajectory frames: the grasps 91.1 %, the trajectory frames 98.9 %): most '
                        'disagreements are points within a pixel of a segmentation edge (the overhead camera sees the '
                        'pads nearly edge-on) or a difference of rule (the prediction tests the contact location 2 mm '
                        'in front of the pad, the render the pad face); those are reported, not hidden. Occlusion '
                        f'disagreements away from an edge must stay under {OCCLUSION_MAX:.0%} of the points (measured '
                        '0.4 %): more would mean the prediction\'s scene or cube maps differ from the render. The '
                        f'SO-100 sample reads {100 * so_t["agreement"]:.1f} % here, not held to this floor (its jaws\' '
                        'contact patches come from collision geometry off the visual mesh, so its rule disagreements '
                        f'are larger); it is held to the same occlusion limit ({100 * so_occ:.1f} % measured) and its '
                        'agreement is in the detail, not hidden.')


@check('groundtruth.ur5e_depth', 'groundtruth', 'Predicted depth validity against rendered depth, UR5e sample',
       '1 − the agreement of the depth-validity study\'s voxels (valid or not) with the rendered depth of every '
       'imager the depth needs, the reference depth with the sensor\'s noise model added; both depth sensors of the '
       'UR5e sample at home',
       'MuJoCo depth renders of the same scene (core/studies/groundtruth.py)', 1 - DEPTH_FLOOR, 'fraction of voxels',
       refs=('mujoco_renderer', 'intel_d400'), cost='study')
def _depth(ctx):
    doc = _study(ctx)
    d = doc['depth']
    t = d['totals']
    per = {s['name']: {'voxel_agreement': (s.get('voxels') or {}).get('agreement'),
                       'surface_agreement': (s.get('surfaces') or {}).get('agreement'),
                       'disagreements': (s.get('voxels') or {}).get('disagreements')} for s in d['sensors']}
    return Outcome(1.0 - float(t['agreement']), {'agreement': t['agreement'], 'iou': t['iou'], 'sensors': per},
                   ok=len(d['sensors']) == 2,
                   note=f'The floor is {DEPTH_FLOOR:.0%} of the voxels (measured 99.8 %). Field of view, min-Z, range, '
                        'projector and materials are the same functions on both sides; what is compared is the '
                        'study\'s cube-map occlusion from both imagers against the rendered depth, and the noise, '
                        'which moves voxels within a few σ_z of a surface. On the surfaces themselves (reported, not '
                        'checked) the agreement is lower (92 % for the overhead D435): a point exactly on a surface is '
                        'where a cube-map texel\'s 1 mm tolerance is weakest.')
