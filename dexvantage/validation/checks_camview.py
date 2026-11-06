"""Checks of core/camrender.py (the Camera view's simulated image) against exact ray casting (core/raycast.py:
Möller–Trumbore through a uniform grid, which shares no code with the rasteriser).

- occlusion.camview_vs_raycast: for pixels sampled at random in images of the UR5e sample's cameras (every imager of
  both D400 units through its own lens model, the first imager of each in a quick run, plus a Brown-Conrady lens and
  a 190° equidistant fisheye at the wrist, so the three rendering paths are all exercised) at random arm poses, the
  distance the image gives against the
  exact first hit along the same pixel ray (the ray unprojected through the same CameraModel). A pixel counts as
  wrong when only one of them finds a surface or their distances differ by more than 1 mm, unless the pixel's hit is
  within two pixels' angle of a silhouette (raycast.near_silhouette), where the rasteriser's pixel-centre rule and the
  base / cube look-up may pick the other side of the edge.
"""
from __future__ import annotations

import math

import numpy as np

from .suite import Outcome, check, figure, series


def _cameras(p, s, q, quick):
    from ..core import camrender as CR
    from ..core import library
    from ..core.optics import CameraModel
    out = []
    for im in library.sensor_imagers(s)[:1 if quick else None]:
        out.append((f'{s.name} {im.name}', im.camera, CR.optical_pose(p, s, im, q), im))
    if s.mount.on_link:
        im = library.sensor_imagers(s)[0]
        pose = CR.optical_pose(p, s, im, q)
        brown = CameraModel(320, 240, 200.0, 200.0, 159.5, 119.5, 'brown', (-0.25, 0.06, 0.001, -0.001, 0.0))
        fish = CameraModel.from_fov(320, 320, 190.0, model='equidistant', max_theta=math.radians(95))
        out += [(f'{s.name} Brown lens', brown, pose, im), (f'{s.name} 190° fisheye', fish, pose, im)]
    return out


@check('occlusion.camview_vs_raycast', 'occlusion', 'Camera view pixel distances against exact ray casting',
       'fraction of sampled pixels (away from silhouettes) whose distance in the Camera view image differs from the '
       'exact first hit along the same lens ray by more than 1 mm, or where only one finds a surface; UR5e sample, '
       'every imager, a Brown and a fisheye lens, random arm poses',
       'core/raycast.py first_hit (Möller–Trumbore through a uniform grid) along CameraModel.unproject rays',
       1e-3, 'fraction', refs=('moller1997', 'amanatides_woo1987'))
def _camview(ctx):
    from ..core import camrender as CR
    from ..core import raycast as RC
    from ..core.frames import quat_to_matrix
    from ..core.robot.model import get_model
    from ..core.robot.samples import sample_project
    n_q, n_px, size = (1, 400, (96, 72)) if ctx.quick else (3, 3000, (192, 144))
    p = sample_project('ur5e')
    r = p.robots[0]
    m = get_model(p, r.uid)
    rng = ctx.rng('occlusion.camview_vs_raycast')
    wrong, elsewhere, total, per, paths, names, sil_total = 0, 0, 0, [], set(), [], 0
    for _ in range(n_q):
        q = m.home.copy()
        q[:6] += rng.normal(0, 0.15, 6)
        qd = {r.uid: m.clip(q)}
        base = CR.gather(p, qd)
        for s in p.variant.sensors:
            from ..core.scenegeom import sensor_world_pose
            g = CR.without_seat(base, p, s, [sensor_world_pose(p, s, qd).position])
            grid = RC.Grid.build(g.tris)
            for name, cam, pose, _im in _cameras(p, s, qd, ctx.quick):
                aspect = cam.width / cam.height
                w = size[0]
                h = max(8, int(round(w / aspect)))
                img = CR.render(g, cam, pose, w, h, floor=False)
                paths.add(img.path)
                c = img.camera
                vv, uu = rng.integers(0, h, n_px), rng.integers(0, w, n_px)
                keep = img.valid[vv, uu]
                vv, uu = vv[keep], uu[keep]
                rays = c.unproject(np.c_[uu, vv].astype(float))
                R = quat_to_matrix(pose.quaternion)
                d = rays @ R.T
                # the near clip: on the optical axis for a pinhole image (pinhole and base paths), on the cube face
                # axis for the cube path; rays starting where the clip is for this ray agree in both cases except
                # for hits closer than 1 cm, which are left out
                near = np.full(len(d), CR.NEAR_M) / np.maximum(np.abs(rays[:, 2]), 0.2)
                t = RC.first_hit(pose.position + d * near[:, None], d, grid) + near
                got = img.dist[vv, uu].astype(float)
                close = (np.minimum(np.where(np.isfinite(t), t, 1e9), got) < 0.01)
                fa, fb = np.isfinite(got), np.isfinite(t)
                gap = np.zeros(len(t))
                both = fa & fb
                gap[both] = np.abs(got[both] - t[both])
                bad = ((fa != fb) | (gap > 1e-3)) & ~close
                hit = np.where(fb, t, np.where(fa, got, 1.0))
                pts = pose.position + d * hit[:, None]
                sil = RC.near_silhouette(pts, pose.position, grid, 2.0 / max(c.fx, c.fy)) if bad.any() else \
                    np.zeros(len(t), bool)
                wrong += int(bad.sum())
                elsewhere += int((bad & ~sil).sum())
                sil_total += int((bad & sil).sum())
                total += int((~close).sum())
                per.append(float((bad & ~sil).sum()) / max(int((~close).sum()), 1))
                names.append(name)
    return Outcome(elsewhere / max(total, 1),
                   {'pixels': total, 'images': len(per), 'paths': sorted(paths), 'image_width_px': size[0],
                    'disagree_fraction': wrong / max(total, 1), 'disagree_near_silhouette': sil_total,
                    'disagree_elsewhere': elsewhere, 'cameras': sorted(set(names))},
                   ok=wrong / max(total, 1) < 0.03 and {'pinhole', 'base', 'cube'} <= paths,
                   figure=figure('Pixels off by more than 1 mm away from a silhouette, per image', 'image',
                                 'fraction', [series('wrong', np.arange(len(per)), np.maximum(per, 1e-9), 'points')],
                                 logy=True, hline=1e-3, hline_label='tolerance'),
                   note='Both measure the distance to the same triangle\'s plane along the same lens ray, so away '
                        'from silhouettes they agree to rounding. Near a silhouette the image answers for its pixel '
                        'centre (pinhole) or for the nearest pixel of its base or cube image, so a ray grazing an '
                        'edge may land on the other side: those are counted but not held against it (they must stay '
                        'under 3 % of the pixels). Hits closer than 1 cm, where the two near-clip rules differ, are '
                        'left out.')
