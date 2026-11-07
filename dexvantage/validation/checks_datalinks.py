"""Checks of the data links (Phase 8): grasp sets taken from a recorded dataset (core/grasps.from_dataset) and hand-eye
sessions captured from one (core/calib/capture.capture_dataset), each on a LeRobot dataset written for the check
whose truth is known by construction.

- grasp.dataset_closures: the UR5e sample's scripted pick (checks_grasps.scripted_pick: the script's own closing
  ranges are the reference) written as a LeRobot v3.0 dataset of two episodes (the second played 1.5 times slower);
  the strokes found must be the script's, row for row, and the gripper frame at each onset must be the forward
  kinematics of the script's joints at that time (the state column is float32: 10⁻⁶ m).
- handeye.dataset_capture: the UR5e sample's wrist D405 calibration recorded as a LeRobot v3.0 episode: planned
  poses held 1 s and joined by 0.8 s minimum-jerk moves, the board rendered (calib/synthetic.render_view) through the
  camera moved by simcapture's default offset (2.06 mm, 0.62°) at half the stream size and AV1-coded; captured from
  the data alone (frames with the board, still, joints interpolated at the frame times, corners scaled to the model's
  pixels) and solved; the calibrated mount against the mount the frames were rendered through.
"""
from __future__ import annotations

import math

import numpy as np

from .suite import Outcome, check

FPS = 20


@check('grasp.dataset_closures', 'grasp', 'Grasps found where the gripper closes in a recorded dataset',
       'largest |gripper frame at a stroke\'s onset − the forward kinematics of the script at that time| (m); the '
       'strokes found must be the script\'s closing ranges, row for row, in both episodes',
       'the closing ranges the scripted pick is built with, and MuJoCo forward kinematics of its joint vectors', 1e-6,
       'm', refs=('lerobot_dataset_format', 'mujoco_todorov2012'))
def _dataset_closures(ctx):
    from ..core import grasps as GR
    from ..core.datasets import lerobot as L
    from ..core.datasets.episode import Episode
    from ..core.frames import Pose
    from ..core.robot.model import get_model
    from ..core.robot.samples import sample_project
    from .checks_grasps import scripted_pick
    p = sample_project('ur5e')
    r = p.robots[0]
    m = get_model(p, r.uid)
    tr, want = scripted_pick(m)
    Q = np.asarray(tr['q'], float)
    t = np.arange(len(Q)) / FPS
    slow = np.repeat(Q, 3, axis=0)[::2]                    # every row 1.5 times: 1.5 times slower
    t_slow = np.arange(len(slow)) / FPS
    eps = [Episode('pick', 0, t, Q, list(m.joint_names), fps=float(FPS)),
           Episode('pick_slow', 1, t_slow, slow, list(m.joint_names), fps=float(FPS))]
    root = f'{ctx.tmp()}/pick'
    L.write_lerobot(root, eps, 'v3.0', FPS, 'ur5e')
    gs = GR.from_dataset(p, root, r.uid)
    got = [(e['episode'], round(e['t'] * FPS), round(e['t_end'] * FPS)) for e in gs['settings']['events']]

    # the slow episode's row j shows script sample floor(2j / 3): a stroke over samples a..b starts at the last row
    # before sample a + 1 shows and ends at the first row showing sample b
    expect = [(0, a, b) for a, b in want] + [(1, math.ceil(3 * (a + 1) / 2) - 1, math.ceil(3 * b / 2))
                                             for a, b in want]
    err = 0.0
    for g in gs['poses']:
        e = int(g['id'][1:g['id'].index('c')])
        Qe, te = (Q, t) if e == 0 else (slow, t_slow)
        k = int(np.argmin(np.abs(te - g['t'])))
        T = m.link_pose(m.clip(Qe[k]), gs['frame'])
        P = Pose.from_dict(g['pose']).matrix()
        err = max(err, float(np.linalg.norm(P[:3, 3] - T[:3, 3])))
    same = sorted(got) == sorted(expect)
    return Outcome(err, {'strokes_found_rows': got, 'strokes_scripted_rows': expect, 'grasps': len(gs['poses'])},
                   ok=same, note='The dataset stores the joint state as float32, so the frame at the onset matches '
                                 'the script to about 10⁻⁷ m; reading the wrong row or episode shows as millimetres. '
                                 'The script holds the fingers still between its ramps, so the strokes are exact.')


@check('handeye.dataset_capture', 'handeye', 'Hand-eye calibration from a recorded LeRobot episode',
       'distance of the calibrated camera from the mount the frames were rendered through (mm); its rotation must '
       'also be within 0.05°', 'the true mount: the project\'s mount moved by simcapture\'s default offset (2.06 mm, '
       '0.62°), through which the board images were rendered', 0.2, 'mm',
       refs=('lerobot_dataset_format', 'opencv_charuco', 'park_martin1994'), cost='study')
def _dataset_capture(ctx):
    from ..core.calib import capture as CAP
    from ..core.calib import handeye as HE
    from ..core.calib import planner as PL
    from ..core.calib import records as RC
    from ..core.calib import simcapture as SIM
    from ..core.calib import synthetic as S
    from ..core.calib.axxb import transform_error
    from ..core.datasets import lerobot as L
    from ..core.datasets.episode import CameraStream, Episode
    from ..core.robot.samples import sample_project
    p = sample_project('ur5e')
    p.calibration['boards'].append(RC.default_board_record(p, 'eye_in_hand', p.robots[0].uid))
    rig = PL.rig_from_project(p)
    n_poses = 8 if ctx.quick else 12
    qs = [np.asarray(x['q']) for x in PL.plan(rig, PL.PlanSettings(n_poses=n_poses)).doc['poses']]
    X0 = rig.X.copy()
    Xt = X0 @ SIM.offset_matrix()
    rig.X = Xt
    cam = rig.camera.scaled(640, 360)
    fps, dwell, move = 5, 1.0, 0.8
    span = dwell + move

    def q_at(x):
        k = int(x // span)
        r_ = x - k * span
        if k >= len(qs) - 1 or r_ <= dwell:
            return qs[min(k, len(qs) - 1)]
        s = (r_ - dwell) / move
        s = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
        return qs[k] * (1 - s) + qs[k + 1] * s
    t = np.arange(0.0, len(qs) * span - move, 1.0 / fps)
    Q = np.array([q_at(x) for x in t])
    rng = np.random.default_rng(ctx.settings.seed)
    cache, frames = {}, []
    for q in Q:
        key = q.tobytes()
        if key not in cache:
            ss = 2 if any(np.array_equal(q, z) for z in qs) else 1
            cache[key] = S.render_view(rig.board, cam, rig.T_cam_board(q), supersample=ss)
        g = np.clip(cache[key] + rng.normal(0.0, 1.0, cache[key].shape), 0, 255).astype(np.uint8)
        frames.append(np.repeat(g[:, :, None], 3, axis=2))

    def dec(idx):
        for i in (range(len(frames)) if idx is None else idx):
            yield i, frames[i]
    key = 'observation.images.wrist_d405'
    cs = CameraStream(key, 640, 360, float(fps), t.copy(), _decode=dec)
    root = f'{ctx.tmp()}/calib'
    L.write_lerobot(root, [Episode('calib', 0, t, Q, list(rig.model.joint_names), fps=float(fps),
                                   cameras={key: cs})], 'v3.0', fps, 'ur5e')
    sess = CAP.capture_dataset(p, root, rig.sensor.uid)
    sess.pop('board', None)
    prob, info = CAP.session_problem(p, sess)
    res = HE.solve(prob, HE.Settings())
    X = res.X[info['camera']]
    dt, dr = transform_error(X, Xt)
    d0, r0 = transform_error(X0, Xt)
    sig = PL.sigma_dict(res.cov_X[info['camera']])
    deg = float(np.degrees(dr))
    return Outcome(1000 * float(dt), {'error_deg': deg, 'start_mm': 1000 * float(d0),
                                      'start_deg': float(np.degrees(r0)), 'views': len(sess['poses']),
                                      'board_frames': sess['source']['board_frames'],
                                      'frames': sess['source']['images'], 'rms_px': float(res.rms_px),
                                      'sigma': {k: float(v) for k, v in sig.items()}},
                   ok=bool(res.ok and deg <= 0.05 and len(sess['poses']) == len(qs)),
                   note='0.2 mm and 0.05°: the error is 3-4 times the solver\'s own σ even on uncoded frames at '
                        'full size (0.020 mm against σ 0.006 mm, measured: the corner errors of a rendered view are '
                        'not independent, as σ assumes); half the stream size doubles the corner error in the '
                        'sensor\'s pixels and AV1 coding adds under 0.002 mm. The camera starts 2.7 mm and 0.62° '
                        'away; joints read at the wrong frames or corners left in the stream\'s pixels miss by '
                        'millimetres.')
