"""The targetless link-silhouette calibration (core/calib/render_calib.py, spec §8.4 and §10 "silhouette calibration
against known perturbations").

- handeye.silhouette_recovery: synthetic LeRobot episodes (core/sim/episodes.py: MuJoCo renders through the lens
  model, 640 × 360, AV1 at LeRobot's settings, 1 grey level of pixel noise; only the mount_slip fault on) of the
  SO-100 sample (its third-person camera) and the UR5e sample (its overhead D435), two episodes each, the camera's
  mount slipping by SLIP_MM = (4, −3, 0) mm and SLIP_DEG = (0.6, 0, 0.8)° (5 mm and 1°, defaults of this check)
  from episode 1. The silhouette calibration estimates each episode from the frames and joint states alone; only then
  is the generator's sidecar read and the recovered optical frame compared with the true one (episodes[e].extrinsics).
  Error: the largest error in pixels of the working image (the position error over one pixel's width at the robot's
  median depth, the rotation error over one pixel's angle, whichever is larger); every estimate must be observable
  and the drift must show a step between episodes 0 and 1.
- handeye.silhouette_vs_board: on the same synthetic SO-100 rig, the board-based hand-eye solve (core/calib: a
  15-pose plan, a ChArUco capture simulated with core/calib/simcapture.py with the same mount slip, the bundle
  adjustment) against the silhouette estimate of the slipped episode. Error: their difference in pixels of the
  working image (as above); the board solve must be within 0.5 mm and 0.05° of the truth (so the board is the
  sharper reference).

The tolerance: PX_TOL = 2 pixels of the working image (640 px wide). One pixel is 1.9 mm sideways at the SO-100's
contour depth through its third-person camera (fx 320 px) and 0.18°; 1.8 mm at the UR5e's through the overhead D435
(fx 466 px) and 0.12°. A pixel-scaled tolerance holds every rig to the same image-space standard: the edges are
located to a fraction of a pixel, but the codec (crf 30) and the noise blur them, and a small, distant robot leaves
sideways shift and yaw nearly interchangeable, so an image-space error of one to two pixels along that direction is
what the silhouette can resolve. Measured 2026-11-07 on the RTX 5080: 1.7 px on the SO-100 (2.4 mm, 0.30°, the same in
both episodes: a bias of the scene, while the slip between the episodes is recovered to about 1 mm), 0.7 px on the
UR5e (1.3 mm, 0.03°).
"""
from __future__ import annotations

import math
import os

import numpy as np

from .suite import Outcome, check

SLIP_MM = (4.0, -3.0, 0.0)          # a default of this check: the mount slip (sensor body frame)
SLIP_DEG = (0.6, 0.0, 0.8)          # a default of this check: its rotation vector (degrees)
PX_TOL = 2.0
BOARD_TOL_MM = 0.5
BOARD_TOL_DEG = 0.05
WIDTH = 640
QUICK_SETTINGS = {'bootstrap': 10}    # a quick run: fewer bootstrap resamples (σ coarser), the tolerances unchanged
QUICK_FRAMES = 90                      # a quick run: rows per synthetic episode (the full run plays each motion whole)
REFS = ('barrow1977', 'borgefors1988', 'canny1986', 'liu2010_fdcm', 'opencv_distance_transform', 'efron1979',
        'liang_zeger1986', 'mujoco_camera_317')


def _faults() -> dict:
    from ..core.sim import episodes as SE
    f = {k: False for k in SE.FAULT_KINDS}
    f['mount_slip'] = {'from_episode': 1, 'position_m': [v / 1000 for v in SLIP_MM], 'rotation_deg': list(SLIP_DEG)}
    return f


def static_camera(project):
    from ..core.calib import render_calib as RC
    return RC.static_cameras(project)[0]


def make_dataset(ctx, sample: str):
    """(project, folder, generator summary) of the synthetic dataset of a sample, made once per suite run."""
    key = ('silhouette_ds', sample)
    if key in ctx.cache:
        return ctx.cache[key]
    from ..core.robot.samples import sample_project
    from ..core.sim import episodes as SE
    p = sample_project(sample)
    root = os.path.join(ctx.tmp(), f'silhouette_{sample}')
    gen = {'episodes': 2, 'max_width': WIDTH, 'seed': 2, 'faults': _faults()}
    if ctx.quick:
        gen['frames'] = QUICK_FRAMES
    summ = SE.generate(p, root, gen)
    ctx.cache[key] = (p, summ['root'], summ)
    return ctx.cache[key]


def pose_error(doc_pose: dict, truth_pose: dict) -> tuple[float, float]:
    """(mm, °) between two {'position', 'quaternion'} poses."""
    from ..core.calib.axxb import transform_error
    from ..core.frames import Pose
    t, r = transform_error(Pose.from_dict(doc_pose).matrix(), Pose.from_dict(truth_pose).matrix())
    return 1000 * t, math.degrees(r)


def recover(ctx, sample: str) -> dict:
    """The drift estimate of a sample's synthetic dataset and its errors against the sidecar."""
    key = ('silhouette_est', sample)
    if key in ctx.cache:
        return ctx.cache[key]
    from ..core.calib import render_calib as RC
    from ..core.datasets import open_dataset
    p, root, summ = make_dataset(ctx, sample)
    cam = static_camera(p)
    with open_dataset(root) as ds:
        dr = RC.drift(ds, p, cam.uid, settings=QUICK_SETTINGS if ctx.quick else None)
        truth = ds.truth()                       # read only now, after the estimate
    rows = []
    for d in dr['episodes']:
        if 'delta' not in d:
            rows.append({'episode': d['episode'], 'error': d.get('error')})
            continue
        tp = truth['episodes'][str(d['episode'])]['extrinsics'][d['camera_key']]['pose']
        emm, edeg = pose_error(d['pose'], tp)
        sc = d['scale']
        rows.append({'episode': d['episode'], 'error_mm': emm, 'error_deg': edeg, 'observable': d['observable'],
                     'error_px': max(emm / sc['mm_per_px'], edeg / sc['deg_per_px']), 'scale': sc,
                     'delta_mm': [1000 * v for v in d['delta']['position_m']],
                     'delta_deg': d['delta']['rotation_deg'], 'sigma_mm': d['sigma']['position_mm'],
                     'sigma_deg': d['sigma']['rotation_deg'], 'iou_after': d['fit']['iou_after'],
                     'rms_px': d['fit']['rms_px'], 'seconds': d['seconds'], 'pose': d['pose'],
                     'camera_key': d['camera_key']})
    out = {'sample': sample, 'camera': cam.name, 'rows': rows, 'steps': dr['steps'],
           'renderer': (dr['episodes'][0].get('render') or {}).get('renderer', ''),
           'generator_seconds': summ.get('seconds')}
    ctx.cache[key] = out
    return out


@check('handeye.silhouette_recovery', 'handeye', 'Silhouette calibration recovers a known mount slip',
       'largest error, in pixels of the 640 px working image (position over a pixel\'s width at the robot\'s depth, '
       'rotation over a pixel\'s angle), of a static camera\'s optical frame recovered from the robot\'s silhouette '
       'in synthetic LeRobot episodes (MuJoCo renders, 640 × 360, AV1), before and after a slip of 5 mm and 1° (a '
       'default of this check), on the SO-100 third-person and UR5e overhead cameras; every estimate must be '
       'observable and the slip found as a step between the episodes',
       'the true mount in the generator\'s ground-truth sidecar, read only after the estimate', PX_TOL, 'px',
       refs=REFS, cost='study')
def _recovery(ctx):
    samples = ('ur5e',) if ctx.quick else ('so100', 'ur5e')
    per, worst, worst_mm, worst_deg, ok, why = [], 0.0, 0.0, 0.0, True, []
    for s in samples:
        r = recover(ctx, s)
        per.append(r)
        for row in r['rows']:
            if 'error_mm' not in row:
                ok = False
                why.append(f'{s} episode {row["episode"]}: {row.get("error")}')
                continue
            worst = max(worst, row['error_px'])
            worst_mm = max(worst_mm, row['error_mm'])
            worst_deg = max(worst_deg, row['error_deg'])
            if not row['observable']:
                ok = False
                why.append(f'{s} episode {row["episode"]} not observable')
        if not any(st['step'] and st['from'] == 0 and st['to'] == 1 for st in r['steps']):
            ok = False
            why.append(f'{s}: the slip was not found as a step')
    detail = {'worst_position_mm': worst_mm, 'worst_rotation_deg': worst_deg,
              'slip': {'position_mm': list(SLIP_MM), 'rotation_deg': list(SLIP_DEG)}, 'samples': per,
              'problems': why}
    note = ('; '.join(f'{r["sample"]} ({r["camera"]}): ' + ', '.join(
        f'episode {x["episode"]} {x["error_px"]:.2f} px ({x["error_mm"]:.2f} mm, {x["error_deg"]:.3f}°)'
        for x in r['rows'] if 'error_mm' in x)
        for r in per) + (f'; rendered on {per[0]["renderer"]}' if per and per[0]['renderer'] else ''))
    return Outcome(worst, detail, ok=ok, note=note)


@check('handeye.silhouette_vs_board', 'handeye', 'Silhouette calibration against the board-based hand-eye solve',
       'difference, in pixels of the 640 px working image (as handeye.silhouette_recovery), between the static '
       'camera\'s optical frame from the silhouette calibration (the slipped episode of the synthetic SO-100 '
       'dataset) and from the board-based bundle adjustment of a simulated ChArUco capture (15 planned poses) of the '
       f'same rig with the same slip; the board solve must be within {BOARD_TOL_MM} mm and {BOARD_TOL_DEG}° of the '
       'truth', 'the board-based hand-eye bundle adjustment (core/calib/handeye.py) on the same synthetic rig', PX_TOL,
       'px', refs=REFS + ('triggs2000', 'garrido2014'), cost='study')
def _vs_board(ctx):
    from ..core.calib import capture as CAP
    from ..core.calib import handeye as HE
    from ..core.calib import planner as PL
    from ..core.calib import simcapture as SIM
    from ..core.calib.axxb import transform_error
    from ..core.frames import Pose
    from ..core.sim.episodes import _delta
    r = recover(ctx, 'so100')
    p, _root, _summ = make_dataset(ctx, 'so100')
    cam = static_camera(p)
    row = next((x for x in r['rows'] if x.get('episode') == 1 and 'pose' in x), None)
    if row is None:
        return Outcome(math.inf, {'problem': 'the slipped episode could not be estimated'}, ok=False)
    rig = PL.rig_from_project(p, cam.uid)
    pl = PL.plan(rig, PL.PlanSettings.from_dict({'n_poses': 15}), start=rig.q_hold)
    # the same slip, expressed as simcapture's offset in the optical frame: T_off = T_so⁻¹ · ΔT · T_so
    from ..core.calib import records as RCR
    T_so = RCR.sensor_imager(cam).pose_in_body.matrix()
    T_off = np.linalg.inv(T_so) @ _delta(np.asarray(SLIP_MM) / 1000, SLIP_DEG) @ T_so
    import cv2
    off_deg = np.degrees(cv2.Rodrigues(T_off[:3, :3])[0].ravel())
    folder = os.path.join(ctx.tmp(), 'silhouette_board_capture')
    sim = SIM.simulate(p, pl.doc['poses'], folder, cam.uid, None, offset_mm=T_off[:3, 3] * 1000, offset_deg=off_deg)
    sess = CAP.capture_folder(p, cam, sim['images'], sim['joints'], None, 0.05, 'rad')
    prob, info = CAP.session_problem(p, sess)
    res = HE.solve(prob, HE.Settings())
    if not res.ok:
        return Outcome(math.inf, {'problem': res.message}, ok=False)
    X = res.X[info['camera']]
    bt, br = transform_error(X, np.asarray(sim['truth']['X_true'], float))
    st, sr = transform_error(Pose.from_dict(row['pose']).matrix(), X)
    truth_gap = transform_error(np.asarray(sim['truth']['X_true'], float), Pose.from_dict(
        _truth_pose(ctx, 'so100', row['camera_key'])).matrix())
    ok = 1000 * bt <= BOARD_TOL_MM and math.degrees(br) <= BOARD_TOL_DEG
    sc = row['scale']
    px = max(1000 * st / sc['mm_per_px'], math.degrees(sr) / sc['deg_per_px'])
    detail = {'silhouette_vs_board_px': px, 'silhouette_vs_board_mm': 1000 * st,
              'silhouette_vs_board_deg': math.degrees(sr), 'scale': sc,
              'board_vs_truth_mm': 1000 * bt, 'board_vs_truth_deg': math.degrees(br),
              'silhouette_vs_truth_mm': row['error_mm'], 'silhouette_vs_truth_deg': row['error_deg'],
              'board_rms_px': float(res.rms_px), 'board_views': len(pl.doc['poses']),
              'truths_agree_mm': 1000 * truth_gap[0], 'truths_agree_deg': math.degrees(truth_gap[1])}
    note = (f'silhouette {px:.2f} px ({1000 * st:.2f} mm, {math.degrees(sr):.3f}°) from the board solve; the board '
            f'solve {1000 * bt:.3f} mm and {math.degrees(br):.4f}° from the truth')
    return Outcome(px, detail, ok=ok, note=note)


def _truth_pose(ctx, sample, key) -> dict:
    from ..core.datasets import open_dataset
    _p, root, _s = make_dataset(ctx, sample)
    with open_dataset(root) as ds:
        return ds.truth()['episodes']['1']['extrinsics'][key]['pose']
