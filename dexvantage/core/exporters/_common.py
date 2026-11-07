"""What the exporters share: the rig's cameras with their frames, frame names, the joint state of an export, lens
approximations and their measured error, and small writers.

    rig_cameras(project, q=None, imagers='all') -> [RigCamera]   every imager of the active variant's cameras in use
    joint_state(project, options) -> (q dict, text)              the export's pose: options['q'] or every robot at home
    frame_name(text) -> str                                      a ROS-style frame / file name (a-z, 0-9, _)
    approx_error_px(cam, approx) -> float                        largest pixel gap between two lens models
    write_text(out_dir, rel, text) -> rel                        a text file under out_dir (folders made)

Frames (docs/models/exporters.md): a sensor's body frame is '<sensor>_link' (REP-103: x forward, y left, z up); each
imager's optical frame is '<sensor>_<imager>_optical_frame' (REP-103's _optical_frame: z forward, x right, y down). A
sensor on a robot link hangs from that link's frame (DexVantage's link frame: the MuJoCo body of the robot's file, or
the URDF link); a sensor on a static frame (cage, tripod, table) hangs from 'world', the primary robot's base, since
static mount frames have no transforms of their own (core/scenegeom.sensor_world_pose).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

from ..frames import Pose
from ..jobs import StudyError

WORLD = 'world'


@dataclass
class RigCamera:
    sensor: object
    imager: object               # library.Imager
    primary: bool
    why: str
    parent: str                  # the parent frame name: a robot link or WORLD
    robot_uid: str               # '' for WORLD
    T_parent_sensor: np.ndarray  # 4 × 4, the mount
    T_sensor_optical: np.ndarray
    T_world_optical: np.ndarray  # at the export's joint state
    sensor_frame: str = ''
    optical_frame: str = ''
    key: str = ''                # a unique file stem '<sensor>__<imager>'
    notes: list = field(default_factory=list)

    @property
    def camera(self):
        return self.imager.camera

    @property
    def T_parent_optical(self) -> np.ndarray:
        return self.T_parent_sensor @ self.T_sensor_optical

    @property
    def title(self) -> str:
        return f'{self.sensor.name} · {self.imager.name}'


def frame_name(text) -> str:
    s = re.sub(r'[^a-z0-9]+', '_', str(text or '').lower()).strip('_')
    return s or 'x'


def imager_name(im) -> str:
    """An imager's part of a frame name: its name, or its role for the default imager of a sensor without a
    device ('Default camera (no device chosen)' would make an unreadable frame name)."""
    return frame_name(im.role or 'camera') if getattr(im, 'is_default', False) else frame_name(im.name)


def joint_state(project, options: dict) -> tuple[dict, str]:
    """({robot uid: q}, what it is) from options['q'] ({robot uid: [joint values]}, RobotModel.joint_names order);
    robots it does not name, or names with the wrong number of values, are at home (said in the text)."""
    from ..robot.model import RobotModelError, get_model
    given = options.get('q') or {}
    if not isinstance(given, dict):
        raise StudyError('The joint state of an export must map each robot to its joint values.')
    q, used, bad = {}, [], []
    for r in project.robots:
        try:
            m = get_model(project, r.uid)
        except RobotModelError:
            continue
        v = given.get(r.uid)
        if v is None:
            q[r.uid] = np.asarray(m.home, float).copy()
            continue
        a = np.asarray(v, float).reshape(-1)
        if len(a) != m.nq or not np.isfinite(a).all():
            bad.append(r.name)
            q[r.uid] = np.asarray(m.home, float).copy()
            continue
        q[r.uid] = a
        used.append(r.name)
    if used:
        text = f'the joint state given with the export ({", ".join(used)})'
    else:
        text = 'the home pose of every robot'
    if bad:
        text += f'; {", ".join(bad)} at home (the given joint values do not fit the robot)'
    if used and len(used) < len(q):
        text += '; the other robots at home'
    return q, text


def rig_cameras(project, q: dict | None = None, imagers: str = 'all') -> list:
    """[RigCamera] of every imager (imagers='all') or each sensor's primary imager ('primary': the coverage study's
    choice, the RGB imager of a depth camera) of the active variant's enabled camera and depth sensors, posed at q."""
    from .. import library as LB
    from ..camrender import optical_pose
    from ..robot.model import RobotModelError, get_model
    from ..studies import coverage as CV
    out, used = [], set()
    for s in project.variant.sensors:
        if not s.enabled or s.kind not in ('camera', 'depth'):
            continue
        try:
            ims = LB.sensor_imagers(s, CV.sensor_mode(s, {}))
        except LB.LibraryError:
            continue
        if not ims:
            continue
        prim, why = CV.pick_imager(s, ims, {})
        if s.mount.on_link:
            r = project.robot(s.mount.robot_uid)
            if r is None:
                continue
            try:
                get_model(project, r.uid).link_index(s.mount.parent_frame)
            except (RobotModelError, KeyError):
                continue
            parent, ruid = s.mount.parent_frame, r.uid
        else:
            parent, ruid = WORLD, ''
        sf = frame_name(s.name) + '_link'
        for im in ims:
            if imagers == 'primary' and im is not prim:
                continue
            iname = imager_name(im)
            key = f'{frame_name(s.name)}__{iname}'
            k = 2
            while key in used:
                key = f'{frame_name(s.name)}__{iname}_{k}'
                k += 1
            used.add(key)
            try:
                Tw = optical_pose(project, s, im, q).matrix()
            except (RobotModelError, KeyError, ValueError):
                continue
            out.append(RigCamera(s, im, im is prim, why if im is prim else '', parent, ruid, s.mount.pose.matrix(),
                                 im.pose_in_body.matrix(), Tw, sf,
                                 f'{frame_name(s.name)}_{key.split("__", 1)[1]}_optical_frame', key))
    return out


def pose_record(T) -> dict:
    """{'position': [x, y, z] m, 'quaternion_xyzw': [...], 'matrix': 4 × 4} of a transform."""
    p = Pose.from_matrix(np.asarray(T, float))
    return {'position': [float(v) for v in p.position], 'quaternion_xyzw': [float(v) for v in p.quaternion],
            'matrix': [[float(v) for v in row] for row in np.asarray(T, float)]}


def approx_error_px(cam, approx, n: int = 41) -> tuple[float, float]:
    """(largest distance (px) between where cam and approx put the same ray, share of cam's rays approx cannot
    map) over an n × n grid of cam's pixels that have a ray (the image corners and edges included)."""
    u, v = np.meshgrid(np.linspace(-0.5, cam.width - 0.5, n), np.linspace(-0.5, cam.height - 0.5, n))
    uv = np.stack([u.ravel(), v.ravel()], 1)
    rays, ok = cam.unproject(uv, with_ok=True)
    rays = rays[ok]
    if not len(rays):
        return float('nan'), 1.0
    uv2, ok2 = approx.project(rays)
    if not ok2.any():
        return float('nan'), 1.0
    d = np.linalg.norm(uv2[ok2] - uv[ok][ok2], axis=1)
    return float(d.max()), float((~ok2).mean())


def write_text(out_dir: str, rel: str, text: str) -> str:
    path = os.path.join(out_dir, rel)
    try:
        os.makedirs(os.path.dirname(path) or out_dir, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(text)
    except OSError as e:
        raise StudyError(f'Could not write {rel}: {e.strerror or e}.') from None
    return rel


def write_bytes(out_dir: str, rel: str, data: bytes) -> str:
    path = os.path.join(out_dir, rel)
    try:
        os.makedirs(os.path.dirname(path) or out_dir, exist_ok=True)
        with open(path, 'wb') as fh:
            fh.write(data)
    except OSError as e:
        raise StudyError(f'Could not write {rel}: {e.strerror or e}.') from None
    return rel


def write_json(out_dir: str, rel: str, doc) -> str:
    import json
    return write_text(out_dir, rel, json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False) + '\n')


def need_cameras(cams: list, what: str):
    if not cams:
        raise StudyError(f'{what} needs at least one camera or depth sensor in use: add one (Sensors ▸ Add camera) or '
                         'tick "In use" on the Sensors and mounts page.')


def fmt(x: float) -> str:
    """A number for XML, YAML and Python text: the shortest form that reads back to the same float64."""
    x = float(x)
    if x == 0.0:
        return '0'
    return repr(x)
