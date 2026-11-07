"""What the project's plan says about a recorded camera: which sensor a dataset camera is, its lens model at the
recorded size, where its optical frame is at a recorded joint state, how points of the robot and of the scene move
in its image (for the desync estimator), and which pixels the gripper's links cover (for the gripper check).

    match_cameras(project, keys, mapping=None) -> {key: (sensor uid | None, how)}
    PlanContext(project, joint_names) -> .camera(key, width, height, sensor_uid) -> PlanCamera | None
    PlanCamera.optical(Q) -> (N, 4, 4)                 the optical frame in the world at joint vectors Q
    PlanCamera.project_points(Q) -> (uv (N, P, 2), ok (N, P))   the sample points (robot surface, and the static scene
                                                       for a camera on a link) in the recorded image
    PlanCamera.gripper_mask(q) -> (H, W) bool          the gripper links' pixels at q (core/camrender's z-buffer of
                                                       the gripper's own triangles, so its links hide each other)

The plan's imager of a sensor is the one the MuJoCo scene records (core/sim/scene.py: coverage.pick_imager over the
library's imagers at the sensor's stream), scaled to the recorded image size with CameraModel.scaled.

A dataset camera is matched to a sensor by name: the dataset key with its 'observation.images.' prefix and any
bracketed part removed, lower case, non-alphanumerics as '_' (the rule core/sim/episodes.py names its cameras by),
first exactly, then when one name contains the other; a mapping {key: sensor uid} (the dataset entry's 'cameras' in
project.data) takes precedence. Joint states are matched to the robot by joint name, else by order when the counts
agree (said in the notes).

Known limits: the predicted image motion ignores occlusion (every sample point counts, seen or not); the gripper
mask is the gripper alone (an object in front of it is what the check is for); one robot (the primary) moves.
"""
from __future__ import annotations

import re

import numpy as np

N_ROBOT_POINTS = 1500             # default: robot surface points sampled for the predicted image motion
N_SCENE_POINTS = 1500             # default: static scene points sampled (cameras on a link)
SEED = 0


def norm_name(name: str) -> str:
    s = str(name or '')
    if s.startswith('observation.images.'):
        s = s[len('observation.images.'):]
    s = re.sub(r'\([^)]*\)', '', s).strip().lower()
    return re.sub(r'[^a-z0-9]+', '_', s).strip('_')


def match_cameras(project, keys, mapping: dict | None = None) -> dict:
    """{dataset camera key: (sensor uid or None, how matched)} over the active variant's enabled cameras and depth
    sensors (see the module docstring)."""
    sensors = [s for s in project.variant.sensors if s.enabled and s.kind in ('camera', 'depth')]
    by_uid = {s.uid: s for s in sensors}
    out, used = {}, set()
    for k in keys:
        u = (mapping or {}).get(k)
        if u and u in by_uid:
            out[k] = (u, 'chosen in the dataset settings')
            used.add(u)
    for k in keys:
        if k in out:
            continue
        nk = norm_name(k)
        hit = next((s for s in sensors if s.uid not in used and norm_name(s.name) == nk), None)
        how = 'by name'
        if hit is None and nk:
            hit = next((s for s in sensors if s.uid not in used and norm_name(s.name)
                        and (nk in norm_name(s.name) or norm_name(s.name) in nk)), None)
            how = 'by a name that contains the other'
        out[k] = (hit.uid, how) if hit is not None else (None, 'no sensor of the plan has this name')
        if hit is not None:
            used.add(hit.uid)
    return out


def _surface_points(tris: np.ndarray, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """n points spread over triangles by area: (points (n, 3), triangle index (n,))."""
    t = np.asarray(tris, float).reshape(-1, 3, 3)
    if not len(t) or n <= 0:
        return np.zeros((0, 3)), np.zeros(0, int)
    a = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    if a.sum() <= 0:
        return np.zeros((0, 3)), np.zeros(0, int)
    ix = rng.choice(len(t), n, p=a / a.sum())
    u = rng.random((n, 2))
    f = u.sum(axis=1) > 1
    u[f] = 1 - u[f]
    P = t[ix, 0] + u[:, :1] * (t[ix, 1] - t[ix, 0]) + u[:, 1:] * (t[ix, 2] - t[ix, 0])
    return P, ix


class PlanContext:
    """The project's primary robot and scene, shared by the cameras of one audit."""

    def __init__(self, project, joint_names=None):
        from .. import scenegeom as SG
        from ..robot.model import RobotModelError, get_model
        self.project = project
        self.notes: list = []
        self.robot = project.primary_robot
        self.model = None
        self.qmap = None
        if self.robot is not None:
            try:
                self.model = get_model(project, self.robot.uid)
            except RobotModelError as e:
                self.notes.append(f'The plan\'s robot cannot be posed: {e}')
        rng = np.random.default_rng(SEED)
        if self.model is not None:
            m = self.model
            self.qmap = self._joint_map(joint_names)
            if m.n_triangles:
                lt = np.asarray(m.local_triangles(), float)
                li = np.asarray(m.triangle_links(), int)
                P, ix = _surface_points(lt, N_ROBOT_POINTS, rng)
                self.robot_pts, self.robot_link = P, li[ix]
            else:
                self.robot_pts, self.robot_link = np.zeros((0, 3)), np.zeros(0, int)
            from ..sim.episodes import gripper_links
            try:
                self.gripper = list(gripper_links(self.robot, m))
            except Exception:                       # noqa: BLE001 — a model without a gripper frame
                self.gripper = []
        static = SG.scene_occluders(project, None, robots=False)
        self.scene_pts, _ = _surface_points(static.tris, N_SCENE_POINTS, rng)
        self._cams: dict = {}

    def _joint_map(self, names) -> np.ndarray | None:
        """Column of the recorded state for each model joint (-1: held at home), or None when nothing matches."""
        m = self.model
        names = list(names or [])
        idx = np.full(m.nq, -1, int)
        pos = {str(n): i for i, n in enumerate(names)}
        for j, n in enumerate(m.joint_names):
            if n in pos:
                idx[j] = pos[n]
        if (idx >= 0).any():
            miss = [m.joint_names[j] for j in range(m.nq) if idx[j] < 0]
            if miss:
                self.notes.append(f'The recorded joint states do not name {", ".join(miss)}; held at home.')
            return idx
        if len(names) == m.nq:
            self.notes.append(f'The recorded joint names do not match {self.robot.name}\'s; the {m.nq} values are '
                              'taken in the model\'s joint order.')
            return np.arange(m.nq)
        if names:
            self.notes.append(f'The recorded joint states ({len(names)} values) do not match {self.robot.name} '
                              f'({m.nq} joints), so the plan cannot be posed at them.')
        return None

    def q_model(self, Q) -> np.ndarray | None:
        """Recorded joint states (N, n) as the model's joint vectors (N, nq), or None."""
        if self.model is None or self.qmap is None:
            return None
        Q = np.atleast_2d(np.asarray(Q, float))
        out = np.tile(self.model.home, (len(Q), 1))
        ok = self.qmap >= 0
        out[:, ok] = Q[:, self.qmap[ok]]
        return out

    def camera(self, key: str, width: int, height: int, sensor_uid: str | None) -> 'PlanCamera | None':
        if not sensor_uid or self.model is None:
            return None
        ck = (key, width, height, sensor_uid)
        if ck not in self._cams:
            self._cams[ck] = PlanCamera(self, sensor_uid, width, height)
        return self._cams[ck]


class PlanCamera:
    def __init__(self, ctx: PlanContext, sensor_uid: str, width: int, height: int):
        from .. import library as LB
        from ..studies import coverage as CV
        p = ctx.project
        self.ctx = ctx
        self.sensor = next(s for s in p.variant.sensors if s.uid == sensor_uid)
        ims = LB.sensor_imagers(self.sensor, CV.sensor_mode(self.sensor, {}))
        if not ims:
            raise ValueError(f'{self.sensor.name} has no imager in its library entry.')
        self.imager, _why = CV.pick_imager(self.sensor, ims, {})
        cam = self.imager.camera
        self.stream = (int(cam.width), int(cam.height))
        self.cam = cam if (int(width), int(height)) == self.stream else cam.scaled(int(width), int(height))
        self.on_link = bool(self.sensor.mount.on_link) and self.sensor.mount.robot_uid == ctx.robot.uid
        self.T_po = self.sensor.mount.pose.matrix() @ self.imager.pose_in_body.matrix()
        self.link_i = ctx.model.link_index(self.sensor.mount.parent_frame) if self.on_link else -1
        if self.sensor.mount.on_link and not self.on_link:
            from ..scenegeom import sensor_world_pose
            self.T_static = sensor_world_pose(p, self.sensor).matrix() @ self.imager.pose_in_body.matrix()
        else:
            self.T_static = self.T_po
        self._mask_cache: dict = {}
        self._gripper_geom = None

    @property
    def name(self) -> str:
        return self.sensor.name

    def optical(self, Q) -> np.ndarray:
        """(N, 4, 4) optical frames in the world at recorded joint states Q (N, n)."""
        Qm = self.ctx.q_model(Q)
        N = len(Qm)
        if not self.on_link:
            return np.broadcast_to(self.T_static, (N, 4, 4)).copy()
        F = self.ctx.model.fk_batch(Qm)[:, self.link_i]
        return F @ self.T_po

    def project_points(self, Q) -> tuple[np.ndarray, np.ndarray]:
        """The sample points' pixels (N, P, 2) and in-image flags (N, P) at recorded joint states Q (N, n)."""
        c = self.ctx
        Qm = c.q_model(Q)
        F = c.model.fk_batch(Qm)                                     # (N, L, 4, 4)
        TT = F[:, c.robot_link]                                      # (N, P, 4, 4)
        W = np.einsum('npij,pj->npi', TT[..., :3, :3], c.robot_pts) + TT[..., :3, 3]
        if self.on_link and len(c.scene_pts):
            W = np.concatenate([W, np.broadcast_to(c.scene_pts, (len(Qm),) + c.scene_pts.shape)], axis=1)
            T = F[:, self.link_i] @ self.T_po
        else:
            T = np.broadcast_to(self.T_static, (len(Qm), 4, 4))
        R = T[:, :3, :3]
        rel = W - T[:, None, :3, 3]
        Pc = np.einsum('npi,nij->npj', rel, R)                       # world -> optical: R^T (x - o)
        uv, ok = self.cam.project(Pc.reshape(-1, 3))
        return uv.reshape(len(Qm), -1, 2), ok.reshape(len(Qm), -1)

    def gripper_mask(self, q) -> np.ndarray | None:
        """The gripper links' pixels at one recorded joint state q, or None when the plan has no gripper links. The
        render is cached by the gripper's pose relative to the camera (rounded to 0.5 mm and 0.003 rad), so a wrist
        camera whose gripper does not open or close renders once."""
        from .. import camrender as CR
        from ..frames import Pose
        c = self.ctx
        if not c.gripper:
            return None
        Qm = c.q_model(np.asarray(q, float).reshape(1, -1))
        if self._gripper_geom is None:
            g = CR.gather(c.project, {c.robot.uid: Qm[0]}, objects=False, surfaces=False)
            own = {f'robot:{c.robot.uid}:{n}' for n in c.gripper}
            ids = [i for i, o in enumerate(g.owners) if o in own]
            self._gripper_ids = ids
            self._gripper_geom = True
        F = c.model.fk_batch(Qm)[0]
        T = (F[self.link_i] @ self.T_po) if self.on_link else self.T_static
        rel = np.stack([np.linalg.inv(T) @ F[c.model.link_index(n)] for n in c.gripper])
        key = tuple(np.round(np.concatenate([np.ravel(rel[:, :3, 3]) / 5e-4, np.ravel(rel[:, :3, :3]) / 3e-3]))
                    .astype(np.int64).tolist())
        hit = self._mask_cache.get(key)
        if hit is not None:
            return hit
        geom = CR.gather(c.project, {c.robot.uid: Qm[0]}, objects=False, surfaces=False)
        keep = np.isin(geom.owner, self._gripper_ids)
        geom = CR.without_seat(geom.select(keep), c.project, self.sensor, [T[:3, 3]])
        if not len(geom):
            mask = np.zeros((self.cam.height, self.cam.width), bool)
        else:
            r = CR.render(geom, self.cam, Pose.from_matrix(T), floor=False)
            mask = (r.tri >= 0) & r.valid
        if len(self._mask_cache) > 64:
            self._mask_cache.clear()
        self._mask_cache[key] = mask
        return mask
