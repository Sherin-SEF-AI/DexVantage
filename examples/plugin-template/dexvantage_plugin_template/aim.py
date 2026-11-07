"""Aim at the task volume: the angle between each enabled camera's viewing direction and the direction from the camera
to the centre of the project's first task volume, at the robot's home pose. An example study for a DexVantage plugin.

    angle = arccos( x_body · (c − o) / |c − o| )

with o the sensor body origin in the world, x_body its body x axis in the world (REP-103: the body frame looks along
x, which is the optical axis of the device's reference imager) and c the task volume's centre. The headline is the
worst angle; a camera is "aimed" when its angle is at most max_angle_deg (default 20°, a template default, not a
standard). Apply turns the worst static camera to look at the centre (its position is kept).

Known limits: the angle says nothing about whether the volume is inside the image (no field of view is used: the
coverage study answers that); the home pose only (a camera on a link is evaluated where the link is at home, and is
not re-aimed by Apply); the reference imager's own pose inside the device is ignored (the body x axis is taken as the
optical axis); only the first task volume is used, by its centre.
"""
from __future__ import annotations

import math

import numpy as np

STUDY_ID = 'aim_check'
TITLE = 'Aim at the task volume'
DEFAULTS = {'max_angle_deg': 20.0}       # template default: change it in the study settings
WORDS = {'good': 'aimed', 'warning': 'off target', 'critical': 'aimed away'}


def settings_of(project) -> dict:
    st = dict(DEFAULTS)
    st.update({k: v for k, v in (project.analysis.studies.get(STUDY_ID) or {}).items() if k in DEFAULTS})
    return st


def angle_deg(origin, axis, target) -> float:
    """The angle (degrees) between the axis at origin and the direction from origin to target."""
    d = np.asarray(target, float) - np.asarray(origin, float)
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return 0.0
    a = np.asarray(axis, float)
    c = float(np.dot(a / np.linalg.norm(a), d / n))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def aim(project, ctx) -> dict:
    """The study: fn(project, ctx) -> JSON-able dict (registered by the plugin's register(api))."""
    from dexvantage.core.frames import quat_to_matrix
    from dexvantage.core.jobs import StudyError
    from dexvantage.core.scenegeom import sensor_world_pose
    st = settings_of(project)
    vols = project.workspace.volumes
    if not vols:
        raise StudyError('Aim at the task volume needs a task volume: add one (Workspace ▸ Add task volume).')
    sensors = [s for s in project.variant.sensors if s.enabled and s.kind in ('camera', 'depth')]
    if not sensors:
        raise StudyError('Aim at the task volume needs an enabled camera: add one (Sensors ▸ Add camera on a tripod).')
    c = np.asarray(vols[0].center, float)
    rows, notes = [], []
    for k, s in enumerate(sensors):
        if ctx.cancelled():
            break
        try:
            pose = sensor_world_pose(project, s)
        except Exception as e:                  # noqa: BLE001 — a sensor that cannot be posed is said, not fatal
            notes.append(f'{s.name} could not be placed in the world ({e}); it is left out.')
            continue
        o = np.asarray(pose.position, float)
        x = quat_to_matrix(pose.quaternion)[:, 0]
        a = angle_deg(o, x, c)
        rows.append({'sensor_uid': s.uid, 'name': s.name, 'angle_deg': round(a, 4),
                     'distance_m': round(float(np.linalg.norm(c - o)), 4), 'on_link': s.mount.on_link,
                     'aimed': a <= st['max_angle_deg'], 'origin': [float(v) for v in o]})
        ctx.report((k + 1) / len(sensors), s.name)
    if not rows:
        raise StudyError('No camera of the rig could be placed in the world, so none can be checked.')
    worst = max(rows, key=lambda r: r['angle_deg'])
    lim = st['max_angle_deg']
    status = 'good' if worst['angle_deg'] <= lim else 'warning' if worst['angle_deg'] <= 2 * lim else 'critical'
    n_ok = sum(r['aimed'] for r in rows)
    suggestion = None
    if status != 'good':
        if worst['on_link']:
            advice = (f'{worst["name"]} looks {worst["angle_deg"]:.0f}° away from the centre of {vols[0].name}; it '
                      'is on a link, so turn it on the Sensors and mounts page.')
        else:
            advice = (f'{worst["name"]} looks {worst["angle_deg"]:.0f}° away from the centre of {vols[0].name}: Apply '
                      'turns it to look at the centre.')
            suggestion = {'sensor_uid': worst['sensor_uid'], 'target': [float(v) for v in c],
                          'text': f'Aim {worst["name"]} at the centre of {vols[0].name}'}
    else:
        advice = f'Every camera looks within {lim:.0f}° of the centre of {vols[0].name}.'
    return {'settings': st, 'volume': {'uid': vols[0].uid, 'name': vols[0].name, 'center': [float(v) for v in c]},
            'cameras': rows, 'notes': notes, 'suggestion': suggestion,
            'summary': {'worst_deg': worst['angle_deg'], 'worst_sensor': worst['name'], 'aimed': n_ok,
                        'total': len(rows), 'share_aimed_pct': 100.0 * n_ok / len(rows), 'status': status,
                        'word': WORDS[status], 'advice': advice}}


def apply_suggestion(project, suggestion: dict):
    """Turn the suggested static camera to look at the target, keeping its position (edits project in place)."""
    from dexvantage.core.robot.samples import look_at
    s = project.variant.sensor(suggestion['sensor_uid'])
    if s is None or s.mount.on_link:
        return False
    s.mount.pose = look_at(s.mount.pose.position, suggestion['target'])
    return True
