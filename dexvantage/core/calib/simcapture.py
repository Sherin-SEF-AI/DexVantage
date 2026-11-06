"""A simulated hand-eye capture: board images rendered at planned joint vectors with the camera moved from where the
project says by a known offset, written as an image folder and a joint_states CSV that the Hand-eye solve page imports
exactly like a recording. It lets a researcher try the whole loop (plan, record, import, solve, Apply) before the rig
exists, and check that the solve finds the offset it was given. It is a test of the pipeline, never a measurement: the
session it makes is named "(simulated)" and its source records the offset.

    simulate(project, poses, folder, ...) -> {'images', 'joints', 'truth', 'n'}

The truth. The rendering rig is planner.rig_from_project's (the sensor, its imager's lens model at the stream size,
the board record or DexVantage's default board placement). The camera's true mount is the project's mount X followed
by the offset T(offset_deg, offset_mm) in the camera's optical frame: X_true = X · T. A solve of the capture should
then move the camera by about |offset_mm| and |offset_deg| (the rotation angle of the offset), within its own σ and the
image noise. Defaults (labelled, not figures of any rig): 2.0, −1.5, 1.0 mm and 0.3°, −0.2°, 0.5° about x, y, z;
image noise σ 1 grey level; 2 × 2 supersampling.

Files. <folder>/images/pose_NN_<t>.png (the time in the name, as capture.image_times reads it), one per pose with
t = 100 + 5k s, and <folder>/joint_states.csv (t, then one column per joint of the robot model, radians) with each
pose read 4 ms after its image and again 0.5 s later (the arm holds still, so capture's speed check sees 0°/s);
<folder>/truth.json holds the offset and X_true.

Known limits: the images come from synthetic.render_view (the board pattern through the lens model): no lighting,
shading, motion blur, rolling shutter, defocus or lens defects beyond the model's distortion, and the joint readings
are exact (no encoder error or backlash); the board is flat and printed exactly at its nominal size. A solve on such a
capture is therefore tighter than one on a real recording.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

DEFAULT_OFFSET_MM = (2.0, -1.5, 1.0)        # a default: the camera's true mount differs from the project's by this
DEFAULT_OFFSET_DEG = (0.3, -0.2, 0.5)       # a default (rotation vector components, degrees, optical frame)
DEFAULT_NOISE = 1.0                          # a default: image noise σ in grey levels
SUPERSAMPLE = 2
T0_S, DT_S, LAG_S = 100.0, 5.0, 0.004        # image times and the joint reading's lag (file names and CSV)
HOLD_S = 0.5                                 # a second reading of the same pose: the arm is still (no "moving" flag)


class SimError(ValueError):
    """A simulated capture that cannot be made; the message says why in plain words."""


def offset_matrix(offset_mm=DEFAULT_OFFSET_MM, offset_deg=DEFAULT_OFFSET_DEG) -> np.ndarray:
    """4 × 4: a rotation by the rotation vector offset_deg (degrees) and a translation offset_mm (mm)."""
    import cv2
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.radians(np.asarray(offset_deg, float)).reshape(3, 1))[0]
    T[:3, 3] = np.asarray(offset_mm, float) / 1000.0
    return T


def simulate(project, poses: list, folder: str, sensor_uid: str | None = None, board_uid: str | None = None,
             offset_mm=DEFAULT_OFFSET_MM, offset_deg=DEFAULT_OFFSET_DEG, noise: float = DEFAULT_NOISE, seed: int = 0,
             progress=None, cancel=None) -> dict:
    """Render the capture (module docstring) into folder; {'images': folder, 'joints': csv path, 'truth', 'n'}.
    poses: [{'q': joint vector}] (a plan's poses). Raises SimError (a plain sentence)."""
    import cv2

    from ..jobs import StudyError
    from . import planner as PL
    from . import synthetic as S
    qs = [np.asarray(p.get('q') if isinstance(p, dict) else p, float) for p in poses or []]
    qs = [q for q in qs if q.size]
    if not qs:
        raise SimError('There are no planned poses to render: run the Hand-eye pose plan first.')
    try:
        rig = PL.rig_from_project(project, sensor_uid, board_uid)
    except StudyError as e:
        raise SimError(str(e)) from None
    if any(len(q) != rig.model.nq for q in qs):
        raise SimError('The planned poses are for another robot model: run the Hand-eye pose plan again.')
    for v, what in ((offset_mm, 'offset (mm)'), (offset_deg, 'offset (°)')):
        if len(v) != 3 or not all(math.isfinite(float(x)) for x in v):
            raise SimError(f'The {what} needs three finite numbers.')
    T_off = offset_matrix(offset_mm, offset_deg)
    X_nominal = rig.X.copy()
    rig.X = X_nominal @ T_off
    img_dir = os.path.join(folder, 'images')
    try:
        os.makedirs(img_dir, exist_ok=True)
    except OSError as e:
        raise SimError(f'Could not make the folder {os.path.basename(img_dir)}: {e.strerror or e}.') from None
    names = list(rig.model.joint_names)
    rows = ['t,' + ','.join(names)]
    written = 0
    for k, q in enumerate(qs):
        if cancel is not None and cancel():
            break
        if progress:
            progress(k / len(qs), f'Rendering pose {k + 1} of {len(qs)}')
        img = S.render_view(rig.board, rig.camera, rig.T_cam_board(q), supersample=SUPERSAMPLE, noise=float(noise),
                            seed=int(seed) + k)
        t = T0_S + DT_S * k
        if not cv2.imwrite(os.path.join(img_dir, f'pose_{k:02d}_{t:.3f}.png'), img):
            raise SimError(f'Could not write the images into {os.path.basename(folder)}: choose a folder you can '
                           'write to.')
        row = ','.join(f'{v:.10f}' for v in q)
        rows += [f'{t + LAG_S:.3f},{row}', f'{t + LAG_S + HOLD_S:.3f},{row}']     # the arm holds still at the pose
        written += 1
    joints = os.path.join(folder, 'joint_states.csv')
    truth = {'kind': 'simulated', 'note': 'a simulated capture: a test of the pipeline, not a measurement',
             'offset_mm': [float(v) for v in offset_mm], 'offset_deg': [float(v) for v in offset_deg],
             'offset_norm_mm': float(np.linalg.norm(offset_mm)),
             'offset_angle_deg': float(np.linalg.norm(offset_deg)),
             'frame': "the camera's optical frame: X_true = X_project · T(offset)",
             'noise_grey': float(noise), 'supersample': SUPERSAMPLE, 'sensor': rig.camera_name,
             'X_true': rig.X.tolist(), 'X_project': X_nominal.tolist()}
    try:
        with open(joints, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(rows) + '\n')
        with open(os.path.join(folder, 'truth.json'), 'w', encoding='utf-8') as fh:
            json.dump(truth, fh, indent=1)
    except OSError as e:
        raise SimError(f'Could not write the joint states: {e.strerror or e}.') from None
    return {'images': img_dir, 'joints': joints, 'truth': truth, 'n': written, 'sensor_uid': getattr(rig.sensor, 'uid',
                                                                                                       sensor_uid)}
