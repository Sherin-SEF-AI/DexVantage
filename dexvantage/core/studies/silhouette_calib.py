"""Targetless link-silhouette calibration of a static camera over a recorded dataset (spec §8.4): the mount of the
camera (and its drift from episode to episode) from the robot's own silhouette in the recorded frames, with no board
(core/calib/render_calib.py does the work; its docstring has the method).

silhouette_calib (batch --silhouette-calib OUT.json): settings in project.analysis.studies['silhouette_calib']:
'dataset' (the recorded dataset's folder or file, any format core/datasets reads; required), 'sensor' (a static
camera's uid; default the first enabled camera or depth sensor not on a robot link), 'episodes' (indices; default the
first max_episodes = 8, a default), 'camera' (the dataset's camera key; default matched to the sensor by name) and
every render_calib setting (max_frames, levels, truncate_px, need_mm, need_deg, tolerance_mm, tolerance_deg, ...).
In batch, --dataset PATH sets 'dataset' for the run (batch.py DATASET_STUDIES).

Output: per episode the estimate (ΔT of the mount in the sensor body frame with its σ, the recovered pose and mount,
the frames used, observable and why, the fit before and after), the steps between consecutive episodes (a change over
step_z = 3 σ and beyond the tolerance, defaults), the trend per episode, and the headline: the last episode whose
estimate is observable (the mount as it is now), its status against the tolerance (defaults 2 mm and 0.2°: 'mount
holds', 'mount moved', 'uncertain', or 'not observable'), and 'apply': the recovered mount pose that the page's Apply
writes (one undo step). The thumbnail overlay (the frame and the rendered silhouette before and after) is kept for
the headline episode only.

Known limits: those of core/calib/render_calib.py (the robot must look like its model; intrinsics and joint
timestamps taken as exact, so a camera–joint desync or clock drift biases the mount; a robot that hardly moves, or
that is small in the image, leaves directions unobservable; σ a lower bound along the weakest direction, about
lateral shift against yaw for a small, distant robot: measured up to 7 σ on the SO-100 sample's third-person camera);
one static camera per run; episodes are estimated independently (no smoothing across them).
"""
from __future__ import annotations

import math
import os

import numpy as np

from .. import jobs
from ..jobs import StudyError

SID = 'silhouette_calib'
STUDY_KEYS = ('dataset', 'sensor')         # the study's own settings, besides render_calib's


def settings(project) -> dict:
    return dict(project.analysis.studies.get(SID) or {})


def calib_settings(project) -> dict:
    """render_calib's settings: the stored ones without the study's own keys."""
    return {k: v for k, v in settings(project).items() if k not in STUDY_KEYS}


def pick_sensor(project, uid: str | None):
    from ..calib import render_calib as RC
    if uid:
        return RC.static_sensor(project, uid)
    cams = RC.static_cameras(project)
    if not cams:
        raise StudyError('The silhouette calibration needs a static camera (one not on a robot link): add one or '
                         'move a camera off the robot.')
    return cams[0]


@jobs.study(SID, 'Silhouette calibration', quick=False, batch='silhouette-calib', needs_robot=True)
def silhouette_study(project, ctx: jobs.Ctx) -> dict:
    """The silhouette calibration of a static camera over the dataset's episodes (module docstring)."""
    from ..calib import render_calib as RC
    from ..datasets.episode import DatasetError, open_dataset
    st = settings(project)
    path = st.get('dataset')
    if not path:
        raise StudyError('Choose a recorded dataset first: the silhouette calibration aligns the robot\'s rendered '
                         'silhouette with its recorded frames (Calibrate ▸ Silhouette calibration ▸ Dataset…).')
    from ..datasets.hub import is_hub_ref
    if not os.path.exists(str(path)) and not is_hub_ref(str(path)):
        raise StudyError(f'The dataset {os.path.basename(str(path)) or path} is not there (nothing at that path): '
                         'choose it again, or check the path.')
    sensor = pick_sensor(project, st.get('sensor') or None)
    cs = RC.settings_of(project, calib_settings(project))
    ctx.report(0.0, 'Opening the dataset')
    try:
        ds = open_dataset(str(path))
    except DatasetError as e:
        raise StudyError(str(e)) from None
    with ds:
        refs = ds.episodes()
        dr = RC.drift(ds, project, sensor.uid, None, calib_settings(project), progress=ctx.report,
                      cancel=ctx.cancelled)
        dinfo = {'name': os.path.basename(os.path.normpath(str(path))), 'format': ds.format, 'version': ds.version,
                 'episodes': len(refs)}
    eps = dr['episodes']
    good = [d for d in eps if 'delta' in d]
    obs = [d for d in good if d.get('observable')]
    head = obs[-1] if obs else (good[-1] if good else None)
    for d in good:
        if d is not head:
            d.pop('overlay', None)
    notes = []
    if not good:
        bad = [d for d in eps if 'error' in d]
        if bad and len({d['error'] for d in bad}) == 1:          # one reason for every episode: say it once
            errs = f'{"the episode" if len(bad) == 1 else f"all {len(bad)} episodes"}: {bad[0]["error"]}'
        else:
            errs = '; '.join(f'episode {d["episode"]}: {d["error"]}' for d in bad)
        raise StudyError(f'No episode could be used: {errs}' if errs else 'No episode could be used.')
    sev, word, advice = RC.status_of(head, cs)
    apply = None
    if head.get('observable'):
        from ..frames import Pose
        cur = sensor.mount.pose.matrix()
        new = Pose.from_dict(head['mount']['pose']).matrix()
        d = np.linalg.inv(cur) @ new
        ang = math.degrees(float(np.linalg.norm(RC.log_se3(d)[3:])))
        apply = {'sensor_uid': sensor.uid, 'parent_frame': sensor.mount.parent_frame, 'robot_uid': '',
                 'pose': head['mount']['pose'], 'change_mm': float(np.linalg.norm(d[:3, 3]) * 1000),
                 'change_deg': ang, 'episode': head['episode']}
    else:
        notes.append('Apply is off: the headline estimate is not observable.')
    steps = [s for s in dr['steps'] if s['step']]
    if steps:
        s = steps[-1]
        notes.append(f'The mount moved between episodes {s["from"]} and {s["to"]}: {s["norm_mm"]:.1f} mm and '
                     f'{s["angle_deg"]:.2f}° ({s["z"]:.1f} σ). {dr["step_rule"].capitalize()}.')
    sig = head['sigma']
    smm = float(np.linalg.norm(np.nan_to_num(sig['position_mm'])))
    sdeg = float(np.linalg.norm(np.nan_to_num(sig['rotation_deg'])))
    doc = {'dataset': dinfo, 'sensor_uid': sensor.uid, 'sensor': sensor.name, 'camera_key': head['camera_key'],
           'episodes': eps, 'drift': {k: dr[k] for k in ('steps', 'largest_step', 'trend', 'step_rule')},
           'headline': {'episode': head['episode'], 'offset_mm': head['delta']['norm_mm'],
                        'offset_deg': head['delta']['angle_deg'], 'sigma_mm': smm, 'sigma_deg': sdeg,
                        'observable': bool(head.get('observable')), 'why': head.get('why', '')},
           'estimate': head, 'status': sev, 'word': word, 'advice': advice, 'apply': apply,
           'requirement': {'tolerance_mm': cs['tolerance_mm'], 'tolerance_deg': cs['tolerance_deg'],
                           'need_mm': cs['need_mm'], 'need_deg': cs['need_deg'],
                           'defaults': [k for k in ('tolerance_mm', 'tolerance_deg', 'need_mm', 'need_deg')
                                        if k in cs['defaults']]},
           'render': head.get('render', {}), 'notes': notes + list(head.get('notes') or []),
           'defaults': list(cs['defaults'])}
    ctx.report(1.0, advice)
    return doc
