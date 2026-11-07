"""Checks of the motion blur and rolling-shutter study (core/studies/motion_blur.py) against independent references.

- blur.image_velocity_fd: the image-plane velocity the study computes from Jacobians (the camera's twist J(q) q̇ and
  each moving point's J_v q̇, through the lens model's derivative) against central finite differences in time of
  the same points projected at q(t − h) and q(t + h): forward kinematics and the lens model only, no Jacobian. On
  the UR5e sample's scripted pick, both cameras (one on the wrist, one fixed), at random times, scene and gripper
  points.
- blur.smear_analytic: the smear of a point moving parallel to the image plane, f · v · E / Z, and of a camera
  turning about its own y axis, f · ω · E at the image centre, written out again here; a pinhole (the study's
  derivative is written out) and a Kannala-Brandt lens (the study's derivative is numerical; at the centre the
  fisheye's focal length is the pinhole's).
- blur.rolling_shutter_bruteforce: where a rolling shutter records moving points (the study's closed form for linear
  image motion) against a brute-force simulation that reads the H rows one at a time, at readout · r/(H − 1) after
  the first, projects each point at the robot's true pose at that instant (forward kinematics along the trajectory)
  and finds the row being read when the point is on it.
"""
from __future__ import annotations

import numpy as np

from .suite import Outcome, check


def _ur5e(ctx):
    """The UR5e sample (with its scripted pick), built once per run."""
    p = ctx.cache.get('blur.ur5e')
    if p is None:
        from ..core.robot.samples import sample_project
        p = ctx.cache['blur.ur5e'] = sample_project('ur5e')
    return p


def _setup(ctx):
    from ..core.robot.model import get_model
    from ..core.studies import motion_blur as MB
    p = _ur5e(ctx)
    tr = p.tasks.trajectories[0]
    m = get_model(p, tr['robot_uid'])
    st = MB.settings_of(p)
    cams = MB.imaging(p, st)
    moving = MB.robot_points(p, tr['robot_uid'], m)
    P_scene, _l = MB.scene_points(p, st)
    return p, tr, m, cams, moving, P_scene


def _world_points_fk(m, q, moving, P_scene):
    """Every point at q from forward kinematics alone (the scene is fixed)."""
    T = m.fk(q)
    parts = [P_scene] + [X @ T[link][:3, :3].T + T[link][:3, 3] for link, X, _n in moving]
    return np.concatenate(parts)


def _optical_fk(p, m, ruid, q, c):
    """The imager's optical frame (R, o) at q from forward kinematics alone."""
    s, im = c['sensor'], c['imager']
    B = s.mount.pose.matrix() @ im.pose_in_body.matrix()
    T = m.link_pose(q, s.mount.parent_frame) @ B if (s.mount.on_link and s.mount.robot_uid == ruid) else B
    return T[:3, :3], T[:3, 3]


@check('blur.image_velocity_fd', 'timing', 'Image-plane velocity against finite differences of projected points',
       'largest |(u̇, v̇)_study − (u̇, v̇)_fd| over the points in view, as a share of that frame\'s largest image speed',
       'central differences (h = 10 µs) of the points projected through the lens at q(t ± h): forward kinematics of '
       'the camera\'s link and of the gripper\'s links, no Jacobian', 1e-5, 'fraction',
       refs=('mujoco_todorov2012', 'lynch_park2017'))
def _velocity_fd(ctx):
    from ..core.robot import trajectory as TR
    from ..core.studies import motion_blur as MB
    p, tr, m, cams, moving, P_scene = _setup(ctx)
    rng = ctx.rng('blur.image_velocity_fd')
    T = np.asarray(tr['t'], float)
    n = 8 if ctx.quick else 30
    h = 1e-5
    worst, rows, frames = 0.0, [], 0
    for _ in range(n):
        k = int(rng.integers(0, len(T) - 1))
        if T[k + 1] - T[k] < 10 * h:
            continue
        t = float(rng.uniform(T[k] + 2 * h, T[k + 1] - 2 * h))      # inside one interval: q̇ is constant there
        q, qd = MB.joint_state(tr, m, t)
        if not np.any(qd):
            continue
        Pw, Vw = MB.moving_points(m, q, qd, moving)
        P = np.concatenate([P_scene] + Pw)
        V = np.concatenate([np.zeros_like(P_scene)] + Vw)
        qm = TR.on_model(tr, m.joint_names, m.home, t - h)
        qp = TR.on_model(tr, m.joint_names, m.home, t + h)
        Pm, Pp = _world_points_fk(m, qm, moving, P_scene), _world_points_fk(m, qp, moving, P_scene)
        for c in cams:
            R, o, v_c, w_c = MB.camera_twist(p, m, tr['robot_uid'], q, qd, c['sensor'], c['imager'])
            uv, uvd, ok, _z = MB.image_velocity(c['imager'].camera, R, o, v_c, w_c, P, V)
            Rm, om = _optical_fk(p, m, tr['robot_uid'], qm, c)
            Rp, op = _optical_fk(p, m, tr['robot_uid'], qp, c)
            um, okm = c['imager'].camera.project((Pm - om) @ Rm)
            up, okp = c['imager'].camera.project((Pp - op) @ Rp)
            sel = ok & okm & okp
            if not sel.any():
                continue
            fd = (up - um) / (2 * h)
            peak = max(float(np.linalg.norm(fd[sel], axis=1).max()), 1.0)
            err = float(np.linalg.norm(uvd[sel] - fd[sel], axis=1).max()) / peak
            worst = max(worst, err)
            frames += 1
            if len(rows) < 12:
                rows.append({'camera': c['name'], 't': round(t, 4), 'points': int(sel.sum()),
                             'peak_px_s': round(peak, 3), 'error_share': err})
    return Outcome(worst, {'frames': frames, 'step_s': h, 'trajectory': tr['name'], 'rows': rows},
                   ok=frames > 0,
                   note='Central differences are exact to O(h²): at h = 10 µs the truncation is far below 10⁻⁸ of the '
                        'speed and the rounding of uv (about 10⁻¹³ px) over 2h is about 10⁻⁸ px/s, so 10⁻⁵ of the '
                        'peak speed leaves room for both while a wrong frame, sign or lever arm shows as a share near '
                        '1.')


@check('blur.smear_analytic', 'timing', 'Smear of a point moving parallel to the image plane against f·v·E/Z',
       'largest relative difference of the smear (px) from the closed forms f·v·E/Z (translation) and f·ω·E (a camera '
       'turning about its own y axis, a point on the axis)',
       'the pinhole relation u = fx·X/Z + cx differentiated by hand; the Kannala-Brandt lens has r = f·θ near the axis',
       1e-6, 'fraction', refs=('hartley2004', 'kannala_brandt2006'))
def _smear_analytic(ctx):
    from ..core.optics import CameraModel
    from ..core.studies import motion_blur as MB
    rng = ctx.rng('blur.smear_analytic')
    cams = {'pinhole': CameraModel.from_fov(1280, 720, 90.0, None),
            'kb': CameraModel(1280, 720, 500.0, 500.0, 639.5, 359.5, 'kb', (0.0, 0.0, 0.0, 0.0))}
    E = 0.004
    worst, rows = 0.0, []
    R, o = np.eye(3), np.zeros(3)
    for name, cam in cams.items():
        f = cam.fx
        for _ in range(4 if ctx.quick else 20):
            Z = float(rng.uniform(0.2, 2.0))
            v = rng.uniform(-0.5, 0.5, 2)
            if name == 'pinhole':
                X, Y = rng.uniform(-0.5, 0.5, 2) * Z
            else:
                X = Y = 0.0                                   # the fisheye's r = f·θ is a pinhole only at the axis
            P = np.array([[X, Y, Z]])
            _uv, uvd, ok, _z = MB.image_velocity(cam, R, o, np.zeros(3), np.zeros(3), P, np.array([[v[0], v[1], 0]]))
            smear = float(np.linalg.norm(uvd[0])) * E
            ref = f * float(np.linalg.norm(v)) * E / Z
            err = abs(smear - ref) / ref
            worst = max(worst, err)
            # the camera turning about its y axis at ω: a point on the axis moves at f·ω (pixels per second)
            w = float(rng.uniform(0.1, 3.0))
            _uv, uvd, ok2, _z = MB.image_velocity(cam, R, o, np.zeros(3), np.array([0.0, w, 0.0]),
                                                  np.array([[0.0, 0.0, Z]]), np.zeros((1, 3)))
            refw = f * w * E
            errw = abs(float(np.linalg.norm(uvd[0])) * E - refw) / refw
            worst = max(worst, errw)
            if len(rows) < 8:
                rows.append({'lens': name, 'Z_m': round(Z, 3), 'v_m_s': [round(x, 3) for x in v],
                             'smear_px': smear, 'closed_form_px': ref, 'turn_smear_px': refw, 'error': max(err, errw)})
    return Outcome(worst, {'exposure_s': E, 'rows': rows},
                   note='The pinhole derivative is exact, so the difference is rounding (10⁻¹⁵); the fisheye\'s '
                        'derivative is a central difference with a step of 10⁻⁶ of the distance, accurate to about '
                        '10⁻¹⁰; 10⁻⁶ leaves room for both.')


@check('blur.rolling_shutter_bruteforce', 'timing', 'Rolling-shutter capture against a row-by-row simulation',
       'largest distance between where the study puts a moving point in a rolling-shutter image and where a '
       'row-by-row simulation of the readout finds it, as a share of how far the point moved during the readout',
       'a brute-force readout: row r is read at readout · r/(H − 1); each point is projected at the robot\'s true pose '
       'at every row time (forward kinematics along the trajectory) and caught where its image crosses the row being '
       'read', 0.05, 'fraction', refs=('liang2008',))
def _rolling(ctx):
    from ..core.robot import trajectory as TR
    from ..core.studies import motion_blur as MB
    p, tr, m, cams, moving, P_scene = _setup(ctx)
    T = np.asarray(tr['t'], float)
    readouts = (0.010, 0.030)
    worst, worst_px, rows, n_pts = 0.0, 0.0, [], 0
    # the start of each readout: inside an interval with motion, so q̇ is constant over the readout
    starts = []
    for k in range(len(T) - 1):
        q, qd = MB.joint_state(tr, m, 0.5 * (T[k] + T[k + 1]))
        if np.any(qd) and T[k + 1] - T[k] > 0.1:
            starts.append(float(T[k] + 0.02))
    if ctx.quick:
        starts = starts[:1]
    for t0 in starts:
        q, qd = MB.joint_state(tr, m, t0)
        Pw, Vw = MB.moving_points(m, q, qd, moving)
        P = np.concatenate([P_scene] + Pw)
        V = np.concatenate([np.zeros_like(P_scene)] + Vw)
        for c in cams:
            cam = c['imager'].camera
            H = cam.height
            R, o, v_c, w_c = MB.camera_twist(p, m, tr['robot_uid'], q, qd, c['sensor'], c['imager'])
            uv, uvd, ok, _z = MB.image_velocity(cam, R, o, v_c, w_c, P, V)
            for Tr in readouts:
                pred, ts = MB.rolling_capture(uv, uvd, H, Tr)
                # brute force: every row's read time, the scene re-posed there
                r = np.arange(H)
                tr_rows = Tr * r / (H - 1)
                vs = np.empty((H, len(P)))
                us = np.empty((H, len(P)))
                okr = np.ones(len(P), bool)
                for i, tau in enumerate(tr_rows):
                    qq = TR.on_model(tr, m.joint_names, m.home, t0 + float(tau))
                    Pt = _world_points_fk(m, qq, moving, P_scene)
                    Rt, ot = _optical_fk(p, m, tr['robot_uid'], qq, c)
                    u, okk = cam.project((Pt - ot) @ Rt)
                    us[i], vs[i] = u[:, 0], u[:, 1]
                    okr &= okk
                d = vs - r[:, None]                              # the point's row minus the row being read
                sel = ok & okr & np.isfinite(pred).all(axis=1)
                for j in np.flatnonzero(sel):
                    s = np.flatnonzero(np.sign(d[:-1, j]) != np.sign(d[1:, j]))
                    if not len(s):
                        continue                                  # not crossed within the readout (at an edge)
                    i = int(s[0])
                    a = d[i, j] / (d[i, j] - d[i + 1, j])
                    cu = us[i, j] + a * (us[i + 1, j] - us[i, j])
                    cv = vs[i, j] + a * (vs[i + 1, j] - vs[i, j])
                    e = float(np.hypot(cu - pred[j, 0], cv - pred[j, 1]))
                    moved = max(float(np.hypot(cu - uv[j, 0], cv - uv[j, 1])), 1.0)
                    worst = max(worst, e / moved)
                    worst_px = max(worst_px, e)
                    n_pts += 1
                if len(rows) < 8:
                    rows.append({'camera': c['name'], 't0': round(t0, 3), 'readout_s': Tr,
                                 'peak_speed_px_s': round(float(np.linalg.norm(uvd[sel], axis=1).max()), 2)
                                 if sel.any() else None, 'worst_share': worst, 'worst_px': worst_px})
    return Outcome(worst, {'points': n_pts, 'worst_px': worst_px, 'rows': rows, 'trajectory': tr['name']},
                   ok=n_pts > 0,
                   note='The study takes the image motion as linear over the readout; the simulation follows the true '
                        'motion (the camera turning, the perspective changing), so they differ at second order: '
                        '½·ü·t² against u̇·t, a share ü·t/(2·u̇) of how far the point moved (at least 1 px is used), '
                        'a few per cent for a wrist camera over a 30 ms readout. 5 % allows that; a wrong row time, '
                        'direction or sign moves the point by a share near 1.')
