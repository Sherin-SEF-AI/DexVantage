"""Check of the Rerun bridge (core/bridges/rerun.py, spec §9.4), registered only when rerun-sdk is installed (the
optional extra 'rerun'; spec §10: "Rerun .rrd opens (if installed)").

formats.rerun_rrd_readback — the UR5e sample (its scripted pick as the trajectory) and a synthetic two-episode audit
result written to an .rrd file and read back with rerun's own reader (rerun.chunk.RrdReader): every entity path the
bridge says it logged must be in the file; each camera's Pinhole must hold the imager's K (column-major, as Rerun
stores a 3 × 3) and resolution, and its imager transform the pose in the body; the gripper link's Transform3D at
the trajectory sample nearest 1.5 s must equal the link's pose from forward kinematics (core/robot/model.py, MuJoCo)
at the trajectory's joint values for that time; the audit flags must be on the 'episode' timeline with the values
logged. The error is the largest difference in metres or pixels; tolerance 1e-4: Rerun stores these components as
float32, whose relative rounding (2^-24, 6e-8) at intrinsics up to about 1400 px gives at most 8.3e-5 px, and at
coordinates up to 1 m 6e-8 m. The detail reports the pixel and metre figures separately.
"""
from __future__ import annotations

import os

import numpy as np

from ..core.bridges import rerun as RB
from .suite import Outcome, check


def _audit_doc():
    return {'episodes': [
        {'index': 0, 'id': 'ep0', 'verdict': 'pass', 'requirements': [{'id': 'max_dropped_frames', 'pass': True}]},
        {'index': 1, 'id': 'ep1', 'verdict': 'drop', 'requirements': [
            {'id': 'max_dropped_frames', 'pass': False, 'camera': 'wrist'},
            {'id': 'max_blur_frames', 'pass': False, 'camera': 'wrist'},
            {'id': 'max_blur_frames', 'pass': False, 'camera': 'top'}]}]}


def run_readback(project, folder: str, t_probe: float = 1.5) -> Outcome:
    """The check's work on a project (also used by tests/test_rerun_bridge.py)."""
    from ..core import library as LB
    from ..core.robot import trajectory as TJ
    from ..core.robot.model import get_model
    path = os.path.join(folder, 'scene.rrd')
    res = RB.stream(project, 'save', path, audit=_audit_doc())
    rb = RB.read_back(path)
    missing = [p for p in res['paths'] if '/' + p.lstrip('/') not in rb]
    e_px, e_m, cams = 0.0, 0.0, 0
    sc = RB.scene(project, meshes=False)
    for s in sc['sensors']:
        sensor = next(x for x in project.variant.sensors if x.uid == s['uid'])
        ims = {im.name: im for im in LB.sensor_imagers(sensor, tuple(sensor.stream) if len(sensor.stream) == 2
                                                       else None)}
        for im in s['imagers']:
            ent = rb.get('/' + im['path'] + '/image')
            if ent is None:
                missing.append(im['path'] + '/image')
                continue
            K = np.array(RB.first(ent['static']['Pinhole:image_from_camera']), float).reshape(3, 3).T
            c = ims[im['name']].camera
            K0 = np.array([[c.fx, 0, c.cx], [0, c.fy, c.cy], [0, 0, 1.0]])
            res_ = RB.first(ent['static']['Pinhole:resolution'])
            e_px = max(e_px, float(np.abs(K - K0).max()), abs(res_[0] - c.width), abs(res_[1] - c.height))
            tr = rb['/' + im['path']]['static']
            M = np.array(RB.first(tr['Transform3D:mat3x3']), float).reshape(3, 3).T
            t = np.array(RB.first(tr['Transform3D:translation']), float)
            T0 = ims[im['name']].pose_in_body.matrix()
            e_m = max(e_m, float(np.abs(M - T0[:3, :3]).max()), float(np.abs(t - T0[:3, 3]).max()))
            cams += 1
    # one link transform over the trajectory
    rob = project.robots[0]
    m = get_model(project, rob.uid)
    tr = next(t for t in project.tasks.trajectories if t.get('robot_uid') == rob.uid)
    link = rob.gripper_frame
    ent = rb.get('/' + next(r for r in sc['robots'] if r['uid'] == rob.uid)['paths'][link], {'rows': []})
    rows = ent['rows']
    if not rows:
        return Outcome(float('nan'), {'missing': missing, 'why': f'no transform of {link} over time'}, ok=False)
    t0 = TJ.duration(tr)[0]
    row = min(rows, key=lambda r: abs(r['time'] * 1e-9 - t_probe))
    t_row = row['time'] * 1e-9
    T = m.link_pose(TJ.on_model(tr, m.joint_names, m.home, t0 + t_row), link)
    M = np.array(RB.first(row['Transform3D:mat3x3']), float).reshape(3, 3).T
    t = np.array(RB.first(row['Transform3D:translation']), float)
    e_link = max(float(np.abs(M - T[:3, :3]).max()), float(np.abs(t - T[:3, 3]).max()))
    # the audit flags on their timeline
    blur = rb.get('/audit/flags/blur', {'rows': []})['rows']
    vals = {r['episode']: RB.first(r['Scalars:scalars']) for r in blur}
    flat = {k: (v[0] if isinstance(v, list) else v) for k, v in vals.items()}
    audit_ok = flat == {0: 0.0, 1: 2.0} and len(rb.get('/audit/verdict', {'rows': []})['rows']) == 2
    err = max(e_px, e_m, e_link)
    return Outcome(err, {'entities_logged': len(res['paths']), 'entities_missing': missing[:10], 'cameras': cams,
                         'intrinsics_px': e_px, 'imager_pose_m': e_m, 'link': link, 'link_time_s': t_row,
                         'link_pose_m': e_link, 'audit_flags_on_timeline': audit_ok,
                         'rerun_sdk': _version()},
                   ok=not missing and cams > 0 and audit_ok)


def _version() -> str:
    try:
        import rerun
        return str(getattr(rerun, '__version__', '?'))
    except Exception:                                    # noqa: BLE001
        return 'not installed'


if RB.available():
    @check('formats.rerun_rrd_readback', 'formats', 'Rerun .rrd file read back by Rerun\'s own reader',
           'largest difference between what the bridge logged and what rerun.chunk.RrdReader reads back: every '
           'camera\'s Pinhole intrinsics and resolution (px), its imager transform (m) and the gripper link\'s '
           'transform at one time of the UR5e sample\'s trajectory against forward kinematics (m); every entity path '
           'present and the audit flags on the episode timeline', 'rerun-sdk\'s own .rrd reader; the library\'s '
           'lens figures; MuJoCo forward kinematics', 1e-4, 'px or m', refs=('rerun_sdk', 'mujoco_todorov2012'),
           cost='study')
    def _rrd(ctx):
        from ..core.robot.samples import sample_project
        return run_readback(sample_project('ur5e'), ctx.tmp())
