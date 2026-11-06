"""Checks of the coverage study (core/studies/coverage.py) against independent paths.

- occlusion.coverage_vs_raycast: the shares of a scripted scene's task volume seen by at least one and at least two
  cameras, and by each camera, against a path that shares nothing with the study but the scene's triangles: pinhole
  projection written out here from the cameras' stated field of view (fx = (W/2) / tan(hfov/2), the REP-103 body to
  optical rotation typed in again), an inside-the-object test written out again, and exact ray casting
  (core/raycast.visible_exact, Möller–Trumbore through a uniform grid) in place of the cube maps.
- optics.coverage_density_fd: the study's pixels per millimetre per voxel against finite differences of OpenCV's
  cv2.projectPoints across a small patch facing the camera (√|det J| of the 2 × 2 Jacobian of the pixel coordinates
  with respect to two orthonormal directions across the line of sight).

The scripted scene (built in the check, no robot): a 0.30 × 0.30 × 0.20 m task volume on a 1 × 1 m table, a box and a
cylinder standing in it, and three cameras on static mounts around it, two with the library's labelled default imager
(no device) and one with the Intel RealSense D435 entry (its RGB imager, pinhole fitted to the stated FOV).
"""
from __future__ import annotations

import math

import numpy as np

from .suite import Outcome, check, figure, series

# REP-103 (typed in again here, not imported): optical = (−y_body, −z_body, x_body)
_B2O = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


def _look_at(eye, target, up=(0.0, 0.0, 1.0)):
    from ..core.frames import Pose, quat_from_matrix
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    x = target - eye
    x /= np.linalg.norm(x)
    y = np.cross(np.asarray(up, float), x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    return Pose(eye, quat_from_matrix(np.c_[x, y, z]))


def scripted_project(quick: bool):
    """The check's scene (see the module docstring)."""
    from ..core.frames import Pose
    from ..core.project import Mount, Project, Sensor, TaskObject, Volume
    p = Project('Coverage check scene')
    c = np.array([0.5, 0.0, 0.10])
    p.workspace.volumes.append(Volume(name='Check volume', center=tuple(c), size=(0.30, 0.30, 0.20)))
    p.workspace.surfaces.append({'uid': 'table', 'name': 'Table', 'z': 0.0, 'size': [1.0, 1.0],
                                 'center': [0.5, 0.0]})
    p.workspace.objects += [
        TaskObject(name='Box', shape='box', size=(0.06, 0.08, 0.12), pose=Pose(np.array([0.46, 0.05, 0.06]))),
        TaskObject(name='Can', shape='cylinder', size=(0.035, 0.10, 0.0), pose=Pose(np.array([0.56, -0.06, 0.05]))),
    ]
    cams = [((0.95, 0.30, 0.45), ''), ((0.10, -0.35, 0.40), ''), ((0.50, 0.05, 0.85), 'intel-realsense-d435')]
    for k, (eye, mid) in enumerate(cams):
        up = (1.0, 0.0, 0.0) if eye[2] > 0.8 else (0.0, 0.0, 1.0)
        p.variant.sensors.append(Sensor(name=f'Camera {k + 1}', kind='depth' if mid else 'camera', slot=k,
                                        model_id=mid, mount=Mount('cage', '', _look_at(eye, c, up))))
    vm, rm = (10.0, 5.0) if quick else (5.0, 2.5)
    p.analysis.voxel_mm, p.analysis.refine_mm = vm, rm
    p.analysis.studies['coverage'] = {'backend': 'auto'}
    return p


def _result(ctx):
    key = 'coverage.scripted'
    if key not in ctx.cache:
        from ..core import jobs
        from ..core.studies import coverage as CV
        p = scripted_project(ctx.quick)
        doc = jobs.run_study('coverage', p)
        ctx.cache[key] = (p, doc, CV.decode(doc))
    return ctx.cache[key]


def _inside_written_out(p, P):
    """Inside the box (axis-aligned here) or the upright cylinder, written out for the scene above."""
    box, can = p.workspace.objects
    bc, bh = np.asarray(box.pose.position), np.asarray(box.size) / 2
    ins = np.all(np.abs(P - bc) <= bh, axis=1)
    cc, r, hh = np.asarray(can.pose.position), can.size[0], can.size[1] / 2
    ins |= (np.hypot(P[:, 0] - cc[0], P[:, 1] - cc[1]) <= r) & (np.abs(P[:, 2] - cc[2]) <= hh)
    return ins


@check('occlusion.coverage_vs_raycast', 'occlusion', 'Coverage shares against projection and exact ray casting',
       'largest absolute difference of the task-volume shares (seen by ≥ 1 camera, by ≥ 2, and by each camera) '
       'between the coverage study and an independent path; scripted scene with a box, a cylinder, a table and three '
       'static cameras',
       'pinhole projection written out from the stated FOV and core/raycast.visible_exact (Möller–Trumbore through a '
       'uniform grid), with the voxels inside the objects found by a test written out again',
       5e-3, 'fraction of the volume', refs=('moller1997', 'amanatides_woo1987', 'greene1986'), cost='study')
def _coverage_vs_raycast(ctx):
    from ..core import library as LB
    from ..core import raycast as RC
    from ..core import scenegeom as SG
    from ..core.frames import quat_to_matrix
    p, doc, dec = _result(ctx)
    P = dec['centers']
    w = dec['size'] ** 3
    ins = _inside_written_out(p, P)
    w_task = np.where(ins, 0.0, w)
    tris = SG.scene_occluders(p).tris
    grid = RC.Grid.build(tris)
    seen, inimgs, sil_far = [], [], 0
    texel = 2.0 / doc['occluders']['cube_n']
    for j, (s, cd) in enumerate(zip(p.variant.sensors, doc['cameras'])):
        if s.model_id:
            e = LB.get(s.model_id)
            im = e.imager('rgb')
            W, H = cd['mode']
            hf, vf = im['hfov_deg'], im['vfov_deg']
            if tuple(im['resolution']) != (W, H):
                raise RuntimeError('the check expects the D435 RGB stream at its native resolution')
        else:
            W, H = LB.DEFAULT_RESOLUTION
            hf, vf = LB.DEFAULT_HFOV_DEG, LB.DEFAULT_VFOV_DEG
        fx = (W / 2) / math.tan(math.radians(hf) / 2)
        fy = (H / 2) / math.tan(math.radians(vf) / 2)
        cx, cy = (W - 1) / 2, (H - 1) / 2
        Rb = quat_to_matrix(s.mount.pose.quaternion)
        tb = np.asarray(s.mount.pose.position, float)
        if s.model_id:                                 # the imager's optical frame in the body, from the entry
            R_bo = quat_to_matrix(im['pose']['quaternion'])
            t = tb + Rb @ np.asarray(im['pose']['position'], float)
            Q = (P - t) @ (Rb @ R_bo)
        else:                                          # the default imager sits at the body origin
            t = tb
            Q = ((P - t) @ Rb) @ _B2O.T                # body, then optical coordinates
        z = Q[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            u = fx * Q[:, 0] / z + cx
            v = fy * Q[:, 1] / z + cy
        inimg = (z > 0) & (u >= -0.5) & (u <= W - 0.5) & (v >= -0.5) & (v <= H - 0.5)
        vis = np.zeros(len(P), bool)
        idx = np.nonzero(inimg)[0]
        vis[idx] = RC.visible_exact(P[idx], t, grid)
        seen.append(vis)
        inimgs.append(inimg)
        # disagreements in the occlusion answer that are not within two texels of a silhouette
        bad = np.nonzero(inimg & ((dec['cameras'][j]['state'] == 2) != vis) & ~ins)[0]
        if len(bad):
            sil_far += int((~RC.near_silhouette(P[bad], t, grid, 2 * texel)).sum())
    seen = np.array(seen)
    red = seen.sum(axis=0)
    tot = w_task.sum()
    ref = {'seen_ge1': float(w_task[red >= 1].sum() / tot), 'seen_ge2': float(w_task[red >= 2].sum() / tot)}
    for k in range(len(seen)):
        ref[f'camera_{k + 1}'] = float(w_task[seen[k]].sum() / tot)
    got = {'seen_ge1': doc['summary']['seen_ge1'], 'seen_ge2': doc['summary']['seen_ge2']}
    for k, cd in enumerate(doc['cameras']):
        got[f'camera_{k + 1}'] = cd['seen']
    diffs = {k: abs(got[k] - ref[k]) for k in ref}
    st = np.array([c['state'] == 2 for c in dec['cameras']])
    task = ~ins
    dis = (st != seen) & task[None]
    frac_dis = float(dis.sum() / max(task.sum() * len(seen), 1))
    inside_agree = bool(np.array_equal(ins, dec['inside']))
    img = np.array([c['state'] != 0 for c in dec['cameras']])
    img_dis = int(((img != np.array(inimgs)) & task[None]).sum())
    n_cells = max(int(task.sum()) * len(seen), 1)
    keys = list(ref)
    return Outcome(max(diffs.values()),
                   {'voxels': int(len(P)), 'triangles': int(len(tris)), 'reference': ref, 'study': got,
                    'abs_diff': diffs, 'voxel_disagreement_fraction': frac_dis, 'inside_test_agrees': inside_agree,
                    'in_image_disagreements': img_dis, 'occlusion_disagreements_not_near_silhouette': sil_far,
                    'backend': doc['occluders']['backend'], 'cube_n': doc['occluders']['cube_n'],
                    'voxel_mm': doc['grid']['voxel_mm'], 'refine_mm': doc['grid']['refine_mm']},
                   ok=frac_dis < 0.02 and inside_agree and sil_far / n_cells <= 1e-3 and img_dis / n_cells <= 1e-4,
                   figure=figure('Coverage share: study against the independent path', 'share', 'fraction',
                                 [series('study', np.arange(len(keys)), [got[k] for k in keys], 'points'),
                                  series('reference', np.arange(len(keys)), [ref[k] for k in keys], 'points')]),
                   note='The study judges occlusion per cube-map texel (2/n rad at a face centre) and the reference '
                        'per exact ray, so voxels within about a texel of a silhouette may differ either way; the '
                        'image edges are the same closed interval in both. Half a percent of the volume leaves room '
                        'for those voxels and nothing else. Per voxel, the disagreement must stay under 2 %, the '
                        'occlusion answers farther than two texels from a silhouette must agree but for 1e-3 '
                        '(rounding at the near clip), and the in-image answers but for 1e-4 (a voxel centre exactly '
                        'on an image edge).')


@check('optics.coverage_density_fd', 'optics', 'Coverage pixels per millimetre against OpenCV finite differences',
       'largest relative difference between the study\'s pixels per millimetre and √|det J| of cv2.projectPoints '
       'differentiated across a patch facing the camera, over seen voxels of the scripted scene',
       'cv2.projectPoints (OpenCV), central differences with a 0.1 mm step across the line of sight',
       1e-2, 'relative', refs=('opencv_calib3d',), cost='study')
def _coverage_density(ctx):
    import cv2

    from ..core import library as LB
    p, doc, dec = _result(ctx)
    rng = ctx.rng('optics.coverage_density_fd')
    worst, n_tot, per = 0.0, 0, []
    for j, (s, cd) in enumerate(zip(p.variant.sensors, doc['cameras'])):
        cam_doc = dec['cameras'][j]
        idx = np.nonzero((cam_doc['state'] == 2) & (cam_doc['ppmm'] >= 0.5))[0]
        if not len(idx):
            continue
        idx = rng.choice(idx, min(len(idx), 300 if ctx.quick else 2000), replace=False)
        ims = LB.sensor_imagers(s)
        im = next(i for i in ims if i.name == cd['imager'])
        K, D, _m = im.camera.to_opencv()
        T = np.asarray(cd['position_m'], float)
        from ..core.scenegeom import sensor_world_pose
        Two = sensor_world_pose(p, s).matrix() @ im.pose_in_body.matrix()
        Po = (dec['centers'][idx] - T) @ Two[:3, :3]
        los = Po / np.linalg.norm(Po, axis=1, keepdims=True)
        a = np.where(np.abs(los[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
        e1 = np.cross(los, a)
        e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(los, e1)
        h = 1e-4

        def proj(X):
            uv, _ = cv2.projectPoints(X.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, np.asarray(D, float))
            return uv.reshape(-1, 2)
        j1 = (proj(Po + h * e1) - proj(Po - h * e1)) / (2 * h)
        j2 = (proj(Po + h * e2) - proj(Po - h * e2)) / (2 * h)
        det = j1[:, 0] * j2[:, 1] - j1[:, 1] * j2[:, 0]
        ref = np.sqrt(np.abs(det)) / 1000.0
        rel = np.abs(cam_doc['ppmm'][idx] - ref) / ref
        worst = max(worst, float(rel.max()))
        per.append(float(rel.max()))
        n_tot += len(idx)
    return Outcome(worst, {'points': n_tot, 'per_camera_max_rel': per, 'quantisation_px_per_mm': 0.01},
                   figure=figure('Largest relative difference per camera', 'camera', 'relative',
                                 [series('max', np.arange(len(per)), np.maximum(per, 1e-9), 'points')], logy=True,
                                 hline=1e-2, hline_label='tolerance'),
                   note='The study stores pixels per millimetre in steps of 0.01 px/mm, so a value of 0.5 px/mm or '
                        'more is within 1 % of its exact figure; the finite differences are accurate to about 1e-8 '
                        'at a 0.1 mm step. 1 % is the storage step and nothing else.')
