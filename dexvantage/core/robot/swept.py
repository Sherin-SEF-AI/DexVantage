"""The ghosted swept volume of a joint trajectory (spec §13, viewport features): the robot's visual meshes drawn at a
few evenly spaced samples of a trajectory, see-through, so the volume the arm sweeps shows in the 3D view at once.

    ghost_mesh(model, cell_m=GHOST_CELL_M) -> (tri (M, 3, 3) float32 in each link's frame, link (M,) int32)
    swept_ghosts(model, traj, n=GHOST_POSES, cell_m=GHOST_CELL_M) -> {'tri' (n·M, 3, 3) float32 world, 'ghost' (n·M,)
        int32 index of the pose, 't' [s] the sampled times, 'triangles_per_pose', 'cell_m'}

The meshes are decimated once per compiled model by vertex clustering (core/scenegeom.decimate, Rossignac and Borrel
1993) link by link in the link's own frame, with a cell of GHOST_CELL_M (10 mm, a default: the ghost is a picture of
where the arm goes, not geometry anything is computed from), then posed by forward kinematics at each sampled time
(core/robot/trajectory.on_model: the trajectory mapped onto the model's joints, the others at home, clipped to the
limits). The decimation takes a few hundred milliseconds for a robot of 200 000 triangles, so the Tasks page runs it
as a background job the first time; posing is one matrix product per link and pose.

Known limits: a set of poses, not the continuous swept volume (motion between two ghosts is not drawn); the
decimated meshes are coarser than the cell near thin parts (a cable clamp, a finger) and can lose them; nothing is
measured from the ghosts (the cable study tests snags at its own sampled poses, core/cables.py).
"""
from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np

GHOST_POSES = 8               # a default: poses drawn along a trajectory
GHOST_CELL_M = 0.01           # a default: the vertex-clustering cell of the ghost meshes
MAX_GHOST_POSES = 64

_CACHE: OrderedDict = OrderedDict()
_LOCK = threading.Lock()


def ghost_mesh(model, cell_m: float = GHOST_CELL_M) -> tuple[np.ndarray, np.ndarray]:
    """The model's visual triangles decimated link by link in each link's own frame (cached per model and cell)."""
    from ..scenegeom import Occluders, decimate
    key = (id(model), float(cell_m))
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None and hit[0] is model:
            return hit[1], hit[2]
    tri_l = np.asarray(model.local_triangles())
    li = np.asarray(model.triangle_links())
    tris, links = [], []
    for k in np.unique(li) if len(li) else []:
        part = decimate(Occluders(tri_l[li == k], np.zeros(int((li == k).sum()), np.int32), ['link']), cell_m)
        tris.append(part.tris)
        links.append(np.full(len(part), int(k), np.int32))
    tri = np.concatenate(tris).astype(np.float32) if tris else np.zeros((0, 3, 3), np.float32)
    link = np.concatenate(links) if links else np.zeros(0, np.int32)
    with _LOCK:
        _CACHE[key] = (model, tri, link)
        while len(_CACHE) > 12:                        # as robot.model's cache of compiled models
            _CACHE.popitem(last=False)
    return tri, link


def is_cached(model, cell_m: float = GHOST_CELL_M) -> bool:
    with _LOCK:
        hit = _CACHE.get((id(model), float(cell_m)))
        return hit is not None and hit[0] is model


def swept_ghosts(model, traj: dict, n: int = GHOST_POSES, cell_m: float = GHOST_CELL_M) -> dict:
    """The ghost meshes at n evenly spaced times of the trajectory (its first and last sample included)."""
    from . import trajectory as TR
    n = int(max(1, min(int(n), MAX_GHOST_POSES)))
    t0, t1 = TR.duration(traj)
    ts = np.linspace(t0, t1, n) if n > 1 else np.array([t0])
    tri_l, link = ghost_mesh(model, cell_m)
    out = np.empty((n * len(tri_l), 3, 3), np.float32)
    for g, t in enumerate(ts):
        T = model.fk_array(model.clip(TR.on_model(traj, model.joint_names, model.home, float(t))))
        dst = out[g * len(tri_l):(g + 1) * len(tri_l)]
        for k in np.unique(link) if len(link) else []:
            sel = link == k
            dst[sel] = (tri_l[sel].reshape(-1, 3) @ T[k, :3, :3].T + T[k, :3, 3]).reshape(-1, 3, 3)
    return {'tri': out, 'ghost': np.repeat(np.arange(n, dtype=np.int32), len(tri_l)), 't': [float(t) for t in ts],
            'triangles_per_pose': int(len(tri_l)), 'cell_m': float(cell_m)}
