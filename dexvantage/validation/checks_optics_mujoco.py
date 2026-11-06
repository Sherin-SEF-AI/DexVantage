"""Checks of the pinhole camera model (core/optics/lens.py) against MuJoCo renders of fiducials (spec §10, Optics).

A scripted MJCF scene holds a grid of small spheres (fiducials) spread over the image of a MuJoCo camera, at depths of
0.4 to 1.5 m, each about 14 px across. MuJoCo renders a segmentation image through EGL (validation/mjrender.py: a
worker process, multi-sampling off, spheres drawn with 128 slices and 64 stacks so their outline is round to a small
fraction of a pixel). The camera model is optics.CameraModel built from the MuJoCo camera's attributes
(mjrender.camera_model): CameraModel.from_fov for a fovy camera; fx, fy, cx, cy from focal/sensorsize and
principalpixel for the others.

Frames, written out: MuJoCo's camera looks along its −z axis with +x right and +y up (XML reference, body/camera). Our
optical frame looks along +z with +x right and +y down. So a point X in the world is, in the optical frame,
P = R_optᵀ (X − p) with R_opt = R_cam · diag(1, −1, −1), R_cam = data.cam_xmat (its columns are the camera axes in
the world) and p = data.cam_xpos. Row 0 of a mujoco.Renderer image is the top of the image, column 0 the left, and a
pixel's centre is at integer (u, v) = (column, row), as OpenCV and CameraModel count them.

- optics.mujoco_fiducial_masks: each sphere's MuJoCo mask against the mask CameraModel predicts (the pixels whose
  centre ray, CameraModel.pixel_rays, passes within the sphere's angular radius asin(r/d) of its centre): the
  centroids of the two masks. Both sample the same pixel centres, so the comparison has no quantisation bias.
- optics.mujoco_fiducial_centres: each MuJoCo mask's centroid against the centre of the sphere's image that
  CameraModel predicts. A sphere's perspective image is an ellipse whose centre is not the projection of the sphere's
  centre: it lies midway between the projections of the two tangent rays at θ − α and θ + α from the optical axis
  (θ the centre's angle from the axis, α = asin(r/d)), in the plane of the axis and the centre. The raw offset of the
  centroid from the projected centre (the perspective bias, up to a quarter pixel here) is reported too.

Three cameras: fovy 60° at 640 × 480; focal in length units with a sensorsize (fx ≠ fy) at 640 × 480; focalpixel with
a principalpixel offset at 800 × 600. The direction of MuJoCo's principal offset is not stated in its reference and was
read off a render (mjrender's module docstring); the third camera checks that the offset is a pure shift of the image
and combines correctly with the focal lengths, not the sign itself.
"""
from __future__ import annotations

import math

import numpy as np

from .suite import Outcome, check, figure, series

R_PX = 7.0                        # fiducial radius in pixels at its depth (about 14 px across)
CAMERAS = (                       # (name, MJCF camera attributes, width, height)
    ('fovy', 'fovy="60"', 640, 480),
    ('focal', 'focal="0.0040 0.0042" sensorsize="0.0064 0.0048" resolution="640 480"', 640, 480),
    ('principal', 'focalpixel="560 560" principalpixel="24 -18" sensorsize="0.0080 0.0060" resolution="800 600"',
     800, 600),
)
SLICES = 128                      # MuJoCo's tessellation of a drawn sphere (visual/quality numslices; stacks half)


def _fiducial_scene(rng, quick: bool):
    """(scene list, MuJoCo model, data): one MJCF per camera, the fiducials placed in its own image."""
    import mujoco
    from . import mjrender as MR
    nx, ny = (5, 4) if quick else (9, 7)
    out = []
    for name, attrs, W, H in CAMERAS:
        # the camera sits at (0, 0, 2) with MuJoCo's identity orientation (it looks down the world −z axis): a
        # fiducial at optical P is at world (P_x, −P_y, 2 − P_z)
        xml = (f'<mujoco model="fiducials_{name}"><visual><quality numslices="{SLICES}" numstacks="{SLICES // 2}"/>'
               f'<map znear="0.01"/><global offwidth="{W}" offheight="{H}"/></visual><worldbody>'
               f'<camera name="{name}" pos="0 0 2" xyaxes="1 0 0 0 1 0" {attrs}/>')
        m0 = mujoco.MjModel.from_xml_string(xml + '</worldbody></mujoco>')
        cm = MR.camera_model(m0, name, W, H)
        pts, rad = [], []
        for i in range(nx):
            for j in range(ny):
                u = (i + 0.5) / nx * W - 0.5 + rng.uniform(-0.2, 0.2) * W / nx
                v = (j + 0.5) / ny * H - 0.5 + rng.uniform(-0.2, 0.2) * H / ny
                z = rng.uniform(0.4, 1.5)
                ray = cm.unproject(np.array([[u, v]]))[0]
                pts.append(ray / ray[2] * z)
                rad.append(z * R_PX / cm.fx)
        pts, rad = np.array(pts), np.array(rad)
        world = pts * np.array([1.0, -1.0, -1.0]) + np.array([0.0, 0.0, 2.0])
        for k, (c, r) in enumerate(zip(world, rad)):
            xml += f'<geom name="f{k}" type="sphere" size="{r:.9g}" pos="{c[0]:.9g} {c[1]:.9g} {c[2]:.9g}"/>'
        m = mujoco.MjModel.from_xml_string(xml + '</worldbody></mujoco>')
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        out.append({'name': name, 'W': W, 'H': H, 'm': m, 'd': d, 'world': world, 'radius': rad})
    return out


def _measure(ctx, cid: str) -> dict:
    """Render the fiducials and measure both checks' figures (cached on ctx: the two checks share one render)."""
    key = ('optics.mujoco_fiducials', ctx.quick)
    if key in ctx.cache:
        return ctx.cache[key]
    from . import mjrender as MR
    scenes = _fiducial_scene(ctx.rng('optics.mujoco_fiducials'), ctx.quick)
    imgs, info = MR.render([{'model': s['m'], 'camera': s['name'], 'width': s['W'], 'height': s['H'],
                             'outputs': ('seg',)} for s in scenes])
    rows = []
    for s, im in zip(scenes, imgs):
        seg = im['seg']
        R, p = MR.mujoco_camera(s['m'], s['d'], s['name'])
        P = MR.world_to_optical(R, p, s['world'])                 # P = R_optᵀ (X − p)
        cm = MR.camera_model(s['m'], s['name'], s['W'], s['H'])
        uv, ok = cm.project(P)
        dist = np.linalg.norm(P, axis=1)
        alpha = np.arcsin(s['radius'] / dist)
        theta = np.arccos(np.clip(P[:, 2] / dist, -1, 1))
        az = np.arctan2(P[:, 1], P[:, 0])

        def ray(t):
            return np.c_[np.sin(t) * np.cos(az), np.sin(t) * np.sin(az), np.cos(t)]
        centre = (cm.project(ray(theta - alpha))[0] + cm.project(ray(theta + alpha))[0]) / 2
        rays = cm.pixel_rays()
        for k in range(len(P)):
            gid = s['m'].geom(f'f{k}').id
            vm, um = np.nonzero(seg == gid)
            pred = (rays @ (P[k] / dist[k])) > math.cos(alpha[k])
            vp, up = np.nonzero(pred)
            if not len(um) or not len(up):
                rows.append((s['name'], len(um), len(up), np.nan, np.nan, np.nan, np.nan, 0.0, bool(ok[k])))
                continue
            c_mj = np.array([um.mean(), vm.mean()])
            c_pr = np.array([up.mean(), vp.mean()])
            mk = seg == gid
            iou = float((pred & mk).sum() / max((pred | mk).sum(), 1))
            rows.append((s['name'], len(um), len(up), float(np.linalg.norm(c_mj - c_pr)),
                         float(np.linalg.norm(c_mj - centre[k])), float(np.linalg.norm(c_mj - uv[k])),
                         float(np.linalg.norm(c_pr - centre[k])), iou, bool(ok[k])))
    res = {'rows': rows, 'info': info, 'n': len(rows)}
    ctx.cache[key] = res
    return res


def _per_camera(rows, col) -> dict:
    out = {}
    for name, *_ in CAMERAS:
        v = np.array([r[col] for r in rows if r[0] == name], float)
        out[name] = float(np.nanmax(v)) if len(v) else float('nan')
    return out


@check('optics.mujoco_fiducial_masks', 'optics', 'Pinhole camera against MuJoCo-rendered fiducial masks',
       'largest distance between the centroid of a sphere\'s MuJoCo segmentation mask and the centroid of the mask '
       'CameraModel predicts (pixel centres whose ray passes within the sphere\'s angular radius); three MuJoCo '
       'cameras (fovy; focal + sensorsize; focalpixel + principalpixel), 63 spheres each',
       'MuJoCo\'s OpenGL renderer (mujoco.Renderer, EGL, segmentation, no multi-sampling) on a scripted MJCF scene',
       0.1, 'px', refs=('mujoco_camera', 'mujoco_renderer', 'hartley2004'), cost='study')
def _masks(ctx):
    r = _measure(ctx, 'optics.mujoco_fiducial_masks')
    rows = r['rows']
    err = np.array([x[3] for x in rows], float)
    iou = np.array([x[7] for x in rows], float)
    missing = int(np.count_nonzero(~np.isfinite(err)))
    by = _per_camera(rows, 3)
    return Outcome(float(np.nanmax(err)) if missing < len(err) else float('nan'),
                   {'fiducials': len(rows), 'missing_masks': missing, 'worst_px_by_camera': by,
                    'median_px': float(np.nanmedian(err)), 'min_mask_iou': float(np.nanmin(iou)),
                    'mask_px_range': [int(min(x[1] for x in rows)), int(max(x[1] for x in rows))],
                    'renderer': r['info'].get('renderer'), 'mujoco': r['info'].get('mujoco'),
                    'render_s': round(float(r['info'].get('seconds', 0.0)), 2)},
                   ok=missing == 0 and float(np.nanmin(iou)) >= 0.95,
                   figure=figure('Mask centroid differences, MuJoCo against CameraModel', 'fiducial', 'px',
                                 [series('|Δ centroid|', np.arange(len(err)), np.maximum(err, 1e-4), 'points')],
                                 logy=True, hline=0.1, hline_label='tolerance'),
                   note='Both masks sample the same pixel centres, so they differ only where MuJoCo\'s drawn sphere '
                        '(a 128 × 64 polygon mesh, inscribed in the sphere) or its rasteriser\'s edge rule decides '
                        'a boundary pixel the other way. One such pixel moves the centroid of a 150–200 px mask by '
                        'about (its distance from the centroid) / (mask area) ≈ 7 / 180 ≈ 0.04 px; 0.1 px allows two. '
                        'A focal length 0.1 % off, or a principal point 0.1 px off, would show as more at the image '
                        'edge (0.3 px for 0.1 % at 320 px from the centre). Every mask must be found and overlap '
                        'its prediction by at least 95 % (intersection over union).')


@check('optics.mujoco_fiducial_centres', 'optics', 'Pinhole projection of sphere centres against MuJoCo renders',
       'largest distance between the centroid of a sphere\'s MuJoCo segmentation mask and the centre of its image '
       'predicted by CameraModel.project (midway between the projections of the two tangent rays in the plane of '
       'the optical axis, which removes the perspective bias); same scene as optics.mujoco_fiducial_masks',
       'MuJoCo\'s OpenGL renderer (mujoco.Renderer, EGL, segmentation, no multi-sampling) on a scripted MJCF scene',
       0.3, 'px', refs=('mujoco_camera', 'mujoco_renderer', 'hartley2004'), cost='study')
def _centres(ctx):
    r = _measure(ctx, 'optics.mujoco_fiducial_centres')
    rows = r['rows']
    err = np.array([x[4] for x in rows], float)
    bias = np.array([x[5] for x in rows], float)
    quant = np.array([x[6] for x in rows], float)
    missing = int(np.count_nonzero(~np.isfinite(err)))
    return Outcome(float(np.nanmax(err)) if missing < len(err) else float('nan'),
                   {'fiducials': len(rows), 'missing_masks': missing, 'worst_px_by_camera': _per_camera(rows, 4),
                    'median_px': float(np.nanmedian(err)),
                    'perspective_bias_max_px': float(np.nanmax(bias)),
                    'sampling_error_max_px': float(np.nanmax(quant)),
                    'renderer': r['info'].get('renderer'), 'mujoco': r['info'].get('mujoco')},
                   ok=missing == 0,
                   figure=figure('Centroid against the predicted image centre', 'fiducial', 'px',
                                 [series('MuJoCo centroid − predicted centre', np.arange(len(err)),
                                         np.maximum(err, 1e-4), 'points'),
                                  series('MuJoCo centroid − projected sphere centre (uncorrected)',
                                         np.arange(len(bias)), np.maximum(bias, 1e-4), 'points')],
                                 logy=True, hline=0.3, hline_label='tolerance'),
                   note='The centroid of a mask sampled at pixel centres is not the centre of the ellipse it '
                        'samples: for these 14 px fiducials the sampling error alone, measured on CameraModel\'s own '
                        'predicted masks against the predicted centres, is reported as sampling_error_max_px (about '
                        '0.2 px). 0.3 px is that measured sampling error with the 0.1 px of the mask check added. '
                        'Comparing with the projected sphere centre without the ellipse correction would add the '
                        'perspective bias (perspective_bias_max_px), which grows with the angle from the axis and '
                        'the sphere\'s size.')
