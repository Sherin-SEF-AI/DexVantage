"""Another variant's cameras in the 3D view beside the active variant's (spec §13 "side-by-side variants"): each
enabled camera and depth sensor of that variant as a dashed frustum outline from its lens model, at the current joint
state, the same outline the scene draws for the active variant's sensors (core/camrender.frustum_body), and a cross
at each sensor's place. Used by the Requirements and variants page (Show in 3D).

Known limits: the frusta are drawn at FRUSTUM_DEPTH (the scene's default depth) and only for the first imager of
each sensor; a sensor whose link cannot be posed or whose device cannot be loaded is left out.
"""
from __future__ import annotations

import json
import traceback

import numpy as np

FRUSTUM_DEPTH = 0.12     # m, as ui/scene.py draws an unselected sensor's frustum (a symbol, not the FOV)
_CACHE: dict = {}


def _body(s, depth):
    from ..core import camrender, library
    key = (s.model_id, library.generation(), json.dumps(s.calibration, sort_keys=True, default=str),
           tuple(getattr(s, 'stream', ())), depth)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    mode = tuple(s.stream) if len(getattr(s, 'stream', ())) == 2 else None
    ims = library.sensor_imagers(s, mode)
    out = camrender.frustum_body(ims[0].camera, ims[0].pose_in_body, depth) if ims else None
    if len(_CACHE) > 128:
        _CACHE.clear()
    _CACHE[key] = out
    return out


def _pose(project, s, q, models: dict):
    """(R, t) of a sensor body frame in the world (core/scenegeom.sensor_world_pose, with one model per robot per
    call: get_model hashes the robot's records on every call)."""
    from ..core.robot.model import get_model
    M = s.mount.pose.matrix()
    if s.mount.on_link:
        uid = s.mount.robot_uid
        if uid not in models:
            models[uid] = get_model(project, uid) if project.robot(uid) is not None else None
        m = models[uid]
        if m is None:
            return None
        qq = (q or {}).get(uid)
        qq = m.home if qq is None or len(qq) != m.nq else np.asarray(qq, float)
        M = m.link_pose(qq, s.mount.parent_frame) @ M
    return M[:3, :3], M[:3, 3]


def frustum_segments(project, variant, q, depth: float = FRUSTUM_DEPTH) -> dict:
    """{sensor name: (S, 2, 3) world segments} of variant's enabled camera and depth sensors (first imager)."""
    out, models = {}, {}
    for s in variant.sensors:
        if not s.enabled or s.kind not in ('camera', 'depth'):
            continue
        try:
            body = _body(s, depth)
            if body is None:
                continue
            pose = _pose(project, s, q, models)
            if pose is None:
                continue
            R, t = pose
            _apex, border, spokes = body
            B = np.asarray(border, float) @ R.T + t
            ok = np.isfinite(B).all(axis=1)
            segs = [[np.asarray(a, float) @ R.T + t, np.asarray(b, float) @ R.T + t] for a, b in spokes]
            segs += [[B[i], B[i + 1]] for i in range(len(B) - 1) if ok[i] and ok[i + 1]]
            if segs:
                out[s.name] = np.asarray(segs, float)
        except Exception:                               # noqa: BLE001 — one sensor never takes the overlay down
            traceback.print_exc()
    return out
