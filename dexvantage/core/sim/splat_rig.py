"""Robot-aligned 3D Gaussian Splatting (spec §9.3): a Gaussian scene of the robot cell fitted from posed images, the
robot's Gaussians attached to its links so the scene re-poses with the joint state q, rendered from any camera pose.
Optional: it needs PyTorch to fit (the splat extra; CUDA when there is one); rendering also works without torch
through the numpy reference rasteriser (small scenes).

    cap = synthetic_capture(project, n_views=24, qs=None, width=64, height=48)      # MuJoCo renders: synthetic
    cap = capture_from_sensors(project, qs)                                          # the rig's own cameras at q(t)
    rig, report = fit_splat_rig(project, cap, FitSettings(iterations=300), progress=None, cancel=None)
    out = rig.render(project, camera_model, T_world_optical, q)                     # {'rgb', 'alpha', ...}
    out = rig.render_sensor(project, sensor_uid, q, extrinsic_delta=None)            # novel pose replay, §8.5 knocks
    rig.save('cell.splat.npz')  ->  cell.splat.npz + cell.splat.json;  SplatRig.load('cell.splat.npz')

The approach is that of SplatSim (Qureshi et al., ICRA 2025, arXiv:2409.10161; key qureshi2025_splatsim) and Robo-GS
(Lou et al., ICRA 2025, arXiv:2408.14873; key lou2025_robogs): a 3DGS scene (Kerbl et al. 2023) whose robot Gaussians
are segmented per link, stored in the link's frame and moved by forward kinematics. Neither paper's code is used.

Capture
-------
A Capture is a list of views: an RGB image, its camera model (core/optics), its pose T_world_optical (the optical
frame in the world: the primary robot's base, REP-103) and the joint state q of every robot when it was taken, plus,
where available, a depth image (optical z, m) and a per-pixel owner segmentation (core/sim/render.py's seg and
seg_owner). For a real capture the poses come from the rig (a static camera's mount; a link-mounted camera's mount at
the logged q(t), scenegeom.sensor_world_pose) and the segmentation from rendering the robot model at the logged q
through the calibrated cameras (core/sim/render.py: the same renderer, so the masks are the model's, not a learned
segmenter's). synthetic_capture renders a bundled sample instead (MuJoCo, synthetic: marked so everywhere), from
views on a sphere round the robot looking at its bounding box.

Initialisation
--------------
Every view with depth is back-projected (every init_stride-th pixel); a point takes the owner of its pixel: a robot
link (its frame is the link's: the point goes into the link frame at the view's q, p_link = T_world_link(q)⁻¹ p) or
the world (frame −1: surfaces, objects, anything else). The points of each frame are thinned to voxel centroids
(voxel_m, grown by 20 % until the frame's share of max_gaussians fits; the budget is shared out by point count, the
robot's links together getting at least robot_share of it),
coloured by the mean of their pixels, shaped by core/splat_reproject.shapes_from_points, opacity INIT_OPACITY. A
capture without depth samples the links' visual meshes instead (mesh_points per link, coloured by the mesh's rgba),
and nothing for the world (the background colour is learnt).

Fitting
-------
Each step renders one training view: the robot Gaussians posed by the view's link transforms (core/robot/model.py,
MuJoCo forward kinematics at the view's q; precomputed per view), the world ones as they are, through the view's lens
model by core/splat_raster.py, against the image with 3DGS's loss 0.8 L1 + 0.2 (1 − SSIM); Adam on the centres (in
their frames), log-scales, rotations, opacity and colour logits and a background colour. Views k with
k mod holdout_every = holdout_every // 2 are held out and scored (PSNR and SSIM) before and after. A step may not move
a Gaussian more than max_shift_m from where it started or grow it beyond MAX_GROWTH times its first size (the depth
placed it; without this a few views let Gaussians stretch into the background). No densification or splitting (3DGS's
adaptive density control) is done; Gaussians left under 1 % opacity are removed. Iterations, a time budget, progress
and cancel follow core.jobs (progress(fraction, text); cancel() true stops and returns what was fitted, marked
cancelled). Seeds: numpy's default_rng(seed) for the view order, torch.manual_seed(seed); the CPU path repeats bit for
bit, the CUDA path does not (atomic adds in the backward pass).

Segmentation per link (segment_links)
-------------------------------------
- 'masks' (the default where the capture has segmentation): in every view each Gaussian's centre, posed at the
  view's q, is projected to its nearest pixel; when it lies on the surface that pixel sees (within DEPTH_TOL_M plus
  its own largest σ of the pixel's depth, defaults) it gets a vote for the owner the segmentation shows there (a
  link, or "not the robot"). A Gaussian takes the owner with most votes and keeps its label when no view voted.
  Chosen because it labels what the images show: a Gaussian that drifted off the mesh during fitting, or a cable or
  sticker the model does not have, still follows the pixels it sits on; the depth test leaves out hidden Gaussians
  and those a wrong label poses away from their surface. Measured on the samples (docs/models/splat.md), it has the
  lowest mislabel rate after a fit (1.0 % of SO-100's and 0.8 % of UR5e's Gaussians in frames that move), while
  'meshes' and 'both' undo more deliberately wrong labels (96 to 98 % against 71 to 72 %). A first version voted
  with each pixel's dominant Gaussian in a render: on SO-100 (24 views at 96 × 72) it left 44 % of the Gaussians
  without a vote and undid 49 % of the wrong labels.
- 'meshes': the link whose visual mesh is nearest to the Gaussian's centre at the capture's reference q (the first
  view's), within max_dist_m, else the world. The meshes are sampled at about one point per mesh_spacing_m² (plus
  their vertices), so the nearest-sample distance overestimates the surface distance by about that spacing. It needs
  no images, but mislabels where links meet each other or an object, and loses Gaussians the fit moved beyond
  max_dist_m.
- 'both': masks, then meshes for the Gaussians no pixel voted for.
A Gaussian that changes frame keeps its pose at the reference q. mislabel_rate compares labels with a truth;
docs/models/splat.md gives the rates measured on the bundled samples.

Files
-----
SplatRig.save(path) writes path (.npz: means, scales, quats (x, y, z, w), opacity, colours, frame, background) and the
sidecar path with .json (schema dexvantage.splat_rig/1: the links each frame index names, q_ref, the robots' file
digests, 'synthetic', the fit report and provenance). A robot Gaussian's mean and rotation are in its link frame; a
world one's in the world. SplatRig.load reads both. ply_bytes(project, q) is the scene posed at q as a standard 3DGS
.ply. Every render returns 'synthetic': True (a 3DGS render is never a real image), and a rig fitted from a synthetic
capture says so in its sidecar.

Known limits
------------
No densification, no view-dependent colour, no exposure or lighting model: a re-posed link shows the lighting it was
captured in (shadows and highlights move with the link); a pose far from every captured q shows surfaces no view saw
(holes, or Gaussians of the wrong side); objects the robot moves are world Gaussians (static); the masks' quality
bounds the segmentation (a calibration error shifts them); the fit report has no LPIPS (core/sim/splat_views.py
computes it, with AlexNet weights the user downloads). Reconstruction quality is measured, not assumed (fit report,
docs/models/splat.md).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .. import splat_raster as SR
from .. import splat_reproject as RP
from ..jobs import StudyError

SCHEMA = 'dexvantage.splat_rig/1'
INIT_OPACITY = 0.5                     # a DexVantage default (the depth places the Gaussians on surfaces)
MAX_GROWTH = 3.0                       # ported from Vantage's optimisation leash
PRUNE_OPACITY = 0.01
DEPTH_TOL_M, DEPTH_TOL_REL, DEPTH_TOL_SIGMA = 0.001, 0.0, 1.0   # defaults: a mask vote's depth tolerance
METHODS = ('masks', 'meshes', 'both')
LPIPS_NOTE = ('LPIPS is not in the fit report: splat views (core/sim/splat_views.py) computes it on held-out frames '
              'when AlexNet\'s weights are in the cache.')


# =========================================================================== capture
@dataclass
class CaptureView:
    """One posed image (module docstring, Capture)."""
    name: str
    rgb: np.ndarray                    # (H, W, 3) uint8
    camera: object                     # optics.CameraModel of the image
    T_world_optical: np.ndarray        # 4 × 4
    q: dict = field(default_factory=dict)          # {robot uid: joint vector}
    depth: np.ndarray | None = None    # (H, W) optical z in m, 0 = no hit
    seg: np.ndarray | None = None      # (H, W) int32 index into Capture.owners, −1 none
    sensor_uid: str | None = None


@dataclass
class Capture:
    views: list
    owners: list = field(default_factory=list)     # [('robot', uid, link) | ('object', uid) | ('surface', uid)]
    synthetic: bool = False
    source: dict = field(default_factory=dict)

    @property
    def q_ref(self) -> dict:
        return dict(self.views[0].q) if self.views else {}


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """T_world_optical of a camera at eye looking at target (optical z towards it, x right, y down; up the world's
    up direction on the image)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= max(np.linalg.norm(z), 1e-12)
    x = np.cross(z, np.asarray(up, float))
    if np.linalg.norm(x) < 1e-9:
        x = np.cross(z, [1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.c_[x, y, z]
    T[:3, 3] = eye
    return T


def sphere_poses(centre, radius: float, n: int, seed: int = 0, elev_deg=(10.0, 60.0)) -> list:
    """n camera poses on a sphere band round centre (elevations elev_deg, a golden-angle spiral in azimuth, ±5 % radius
    jitter from seed), each looking at centre."""
    rng = np.random.default_rng(seed)
    out = []
    lo, hi = (math.radians(e) for e in elev_deg)
    for k in range(int(n)):
        el = lo + (hi - lo) * ((k + 0.5) / max(n, 1))
        az = k * math.pi * (3.0 - math.sqrt(5.0)) + rng.uniform(-0.1, 0.1)
        r = radius * (1.0 + rng.uniform(-0.05, 0.05))
        eye = np.asarray(centre, float) + r * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az),
                                                        math.sin(el)])
        out.append(look_at(eye, centre))
    return out


def _robot_uid(project, robot_uid=None) -> str:
    if not project.robots:
        raise StudyError('A splat rig needs a robot: import one first (Robot ▸ Import robot…) or open a sample.')
    if robot_uid is None:
        return project.robots[0].uid
    if project.robot(robot_uid) is None:
        raise StudyError(f'There is no robot {robot_uid} in the project.')
    return robot_uid


def random_qs(project, n: int, seed: int = 0, spread_rad: float = 0.4, robot_uid=None) -> list:
    """[{robot uid: q}]: the home pose, then n − 1 poses drawn uniformly within ±spread_rad (a default) of it,
    clipped to the joint limits."""
    from ..robot.model import get_model
    uid = _robot_uid(project, robot_uid)
    m = get_model(project, uid)
    rng = np.random.default_rng(seed)
    out = [{uid: m.home.copy()}]
    for _ in range(max(int(n), 1) - 1):
        out.append({uid: m.clip(m.home + rng.uniform(-spread_rad, spread_rad, m.nq))})
    return out


def synthetic_capture(project, n_views: int = 24, qs=None, width: int = 64, height: int = 48,
                      hfov_deg: float = 55.0, seed: int = 0, radius_scale: float = 1.4, robot_uid=None,
                      renderer=None, progress=None, cancel=None) -> Capture:
    """A synthetic capture of the project's cell: n_views MuJoCo renders (core/sim/render.py: RGB, depth and the
    owner segmentation) through a pinhole of hfov_deg on width × height, from sphere_poses round the robot's bounding
    box at each view's q (qs: [{uid: q}], used in turn; default the home pose). radius = radius_scale × the box's
    diagonal (defaults). Marked synthetic."""
    from ..optics import CameraModel
    from ..robot.model import get_model
    from .render import Renderer
    uid = _robot_uid(project, robot_uid)
    m = get_model(project, uid)
    qs = qs or [{uid: m.home.copy()}]
    f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    cam = CameraModel(int(width), int(height), f, f, (width - 1) / 2.0, (height - 1) / 2.0)
    own = renderer is None
    R = renderer or Renderer(project)
    try:
        views = []
        lo, hi = m.bounds(m.home)
        centre = 0.5 * (lo + hi)
        radius = radius_scale * float(np.linalg.norm(hi - lo))
        poses = sphere_poses(centre, radius, n_views, seed)
        for k, T in enumerate(poses):
            if cancel is not None and cancel():
                break
            q = {u: np.asarray(v, float).copy() for u, v in qs[k % len(qs)].items()}
            img = R.render_pose(cam, T, q, ('rgb', 'depth', 'seg'))
            views.append(CaptureView(f'view{k:03d}', np.asarray(img['rgb'], np.uint8), cam, T, q,
                                     np.asarray(img['depth'], np.float32), np.asarray(img['seg'], np.int32)))
            if progress is not None:
                progress((k + 1) / len(poses), f'Rendering capture view {k + 1} of {len(poses)}')
        owners = list(R.build.owners)
        info = R.info
    finally:
        if own:
            R.close()
    return Capture(views, owners, True, {
        'kind': 'synthetic MuJoCo render (core/sim/render.py)', 'robot_uid': uid, 'n_views': len(views),
        'qs': [{u: [float(x) for x in v] for u, v in q.items()} for q in qs], 'width': int(width),
        'height': int(height), 'hfov_deg': float(hfov_deg), 'seed': int(seed), 'radius_m': radius,
        'renderer': info.get('renderer', ''), 'mujoco': info.get('mujoco', '')})


def capture_from_sensors(project, qs, sensor_uids=None, scale: float = 1.0, renderer=None) -> Capture:
    """A synthetic capture through the rig's own cameras: every camera or depth sensor (or those of sensor_uids) at
    every joint state of qs ([{uid: q}]; a link-mounted camera moves with it), rendered by MuJoCo at scale of its
    stream through its primary imager (the scene's camera). Poses: scenegeom.sensor_world_pose (the mount, at q for
    a link mount) composed with the imager's pose in the sensor body."""
    from ..scenegeom import sensor_world_pose
    from .render import Renderer
    sens = [s for s in project.variant.sensors if s.kind in ('camera', 'depth') and
            (sensor_uids is None or s.uid in sensor_uids)]
    if not sens:
        raise StudyError('The rig has no camera to capture from: add one, or use a synthetic capture.')
    own = renderer is None
    R = renderer or Renderer(project)
    views = []
    try:
        for j, q in enumerate(qs):
            jobs = [{'sensor_uid': s.uid, 'q': q, 'outputs': ('rgb', 'depth', 'seg'), 'scale': scale} for s in sens]
            for s, img in zip(sens, R.render(jobs)):
                T = sensor_world_pose(project, s, q).matrix() @ R.camera(s.uid).T_sensor_optical
                views.append(CaptureView(f'{s.name} · q{j}', np.asarray(img['rgb'], np.uint8), img['model'], T,
                                         {u: np.asarray(v, float).copy() for u, v in q.items()},
                                         np.asarray(img['depth'], np.float32), np.asarray(img['seg'], np.int32),
                                         s.uid))
        owners = list(R.build.owners)
    finally:
        if own:
            R.close()
    return Capture(views, owners, True, {'kind': 'synthetic MuJoCo render through the rig\'s cameras',
                                         'sensors': [s.uid for s in sens], 'n_q': len(qs), 'scale': float(scale)})


# =========================================================================== the rig
def robot_digest(project, uid: str) -> str:
    """SHA-256 of a robot's stored files and its record (base pose, joint records): the kinematics the rig was fitted
    with."""
    from ..robot import model as RM
    r = project.robot(uid)
    key = [list(map(str, RM._files_key(r, project.sources))), r.to_dict()]
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()


def _q_of(q, uid, m):
    v = (q or {}).get(uid)
    return m.home if v is None else np.asarray(v, float)


@dataclass
class SplatRig:
    """Gaussians, each in its own frame (frame[i] = −1: the world; k ≥ 0: links[k] = (robot uid, link name)),
    a learnt background colour, the capture's reference joint state, the robots' digests and the fit's info."""
    gaussians: RP.GaussianSet
    frame: np.ndarray
    links: list
    q_ref: dict = field(default_factory=dict)
    background: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    robots: dict = field(default_factory=dict)
    synthetic: bool = False
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        self.frame = np.asarray(self.frame, np.int32).reshape(-1)
        self.links = [tuple(x) for x in self.links]
        self.background = np.asarray(self.background, np.float32).reshape(3)
        self.q_ref = {u: np.asarray(v, float) for u, v in (self.q_ref or {}).items()}

    @property
    def n(self) -> int:
        return self.gaussians.n

    def check_robots(self, project):
        """Raise StudyError when a robot the rig is attached to is missing or its files differ from the fit's."""
        for uid, r in self.robots.items():
            if project.robot(uid) is None:
                raise StudyError(f'The splat rig was fitted to robot {r.get("name") or uid}, which is not in this '
                                 'project.')
            if r.get('digest') and r['digest'] != robot_digest(project, uid):
                raise StudyError(f'Robot {r.get("name") or uid} has changed since the splat rig was fitted (its files '
                                 'or joints differ): fit the rig again.')

    def link_transforms(self, project, q=None) -> np.ndarray:
        """(len(links) + 1, 4, 4): T_world_link of every link at q ({uid: q}, home for a robot q leaves out) by
        MuJoCo's forward kinematics (core/robot/model.py), the identity last (the world frame, index −1)."""
        from ..robot.model import get_model
        out = np.tile(np.eye(4), (len(self.links) + 1, 1, 1))
        fks = {}
        for k, (uid, ln) in enumerate(self.links):
            if uid not in fks:
                m = get_model(project, uid)
                fks[uid] = m.fk(_q_of(q, uid, m))
            out[k] = fks[uid][ln]
        return out

    def posed(self, project, q=None) -> RP.GaussianSet:
        """The Gaussians in the world at q: a robot Gaussian moved by its link's T_world_link(q)."""
        T = self.link_transforms(project, q)
        return self.gaussians.transformed(T[self.frame])        # frame −1 indexes the identity, last

    def render(self, project, cam, T_world_optical, q=None, backend: str = 'auto', device=None,
               want_index: bool = False, depth: bool = False) -> dict:
        """The scene at q through cam at T_world_optical: core/splat_reproject.render's dict plus 'synthetic': True."""
        out = RP.render(self.posed(project, q), cam, T_world_optical, backend, device, self.background, want_index,
                        depth)
        out['synthetic'] = True
        return out

    def render_sensor(self, project, sensor_uid: str, q=None, extrinsic_delta=None, scale: float = 1.0,
                      backend: str = 'auto', device=None) -> dict:
        """The scene through a rig sensor's imager (its RGB imager when it has one) at q, its mount perturbed by
        extrinsic_delta (frames.Pose or 4 × 4 in the sensor body frame: a proposed pose change or a §8.5 knock), at
        scale of the stream. Adds 'T_world_optical' and 'camera' to the render."""
        from ..scenegeom import sensor_world_pose
        s = project.variant.sensor(sensor_uid)
        if s is None:
            raise StudyError(f'There is no sensor {sensor_uid} in the active rig variant.')
        im = stream_imager(s)
        if im is None:
            raise StudyError(f'{s.name} has no imager to render.')
        Tb = sensor_world_pose(project, s, q).matrix()
        if extrinsic_delta is not None:
            Tb = Tb @ (extrinsic_delta.matrix() if hasattr(extrinsic_delta, 'matrix')
                       else np.asarray(extrinsic_delta, float))
        T = Tb @ im.pose_in_body.matrix()
        cam = im.camera
        if abs(scale - 1.0) > 1e-12:
            cam = cam.scaled(max(1, int(round(cam.width * scale))), max(1, int(round(cam.height * scale))))
        out = self.render(project, cam, T, q, backend, device)
        out.update(T_world_optical=T, camera=cam)
        return out

    def ply_bytes(self, project, q=None) -> bytes:
        what = 'synthetic capture' if self.synthetic else 'capture'
        return RP.ply_bytes(self.posed(project, q), f'DexVantage splat rig ({what}) posed at a joint state, robot base '
                                                    'frame, metres; renders are synthetic')

    # ---------------------------------------------------------------- files
    def sidecar(self) -> dict:
        return {'schema': SCHEMA, 'synthetic': bool(self.synthetic), 'units': 'metres',
                'quaternion_order': 'x, y, z, w', 'gaussians': self.n,
                'frames': 'frame -1: mean and rotation in the world (the primary robot base, REP-103); frame k: in '
                          'the frame of links[k] = [robot uid, link name]',
                'links': [list(x) for x in self.links],
                'q_ref': {u: [float(x) for x in v] for u, v in self.q_ref.items()},
                'robots': self.robots, 'background': [float(x) for x in self.background],
                'renders': 'synthetic (3D Gaussian Splatting renders, never real images)', 'info': self.info}

    def save(self, path) -> list[str]:
        """path (.npz; the extension is added when missing) and its .json sidecar; returns both paths."""
        path = str(path)
        if not path.endswith('.npz'):
            path += '.npz'
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        g = self.gaussians
        tmp = path[:-4] + '.part.npz'
        np.savez_compressed(tmp, means=g.means, scales=g.scales, quats=g.quats, opacity=g.opacity, colours=g.colours,
                            frame=self.frame, background=self.background)
        os.replace(tmp, path)
        side = path[:-4] + '.json'
        with open(side, 'w', encoding='utf-8') as fh:
            json.dump(self.sidecar(), fh, indent=1, default=_json_default)
        return [os.path.abspath(path), os.path.abspath(side)]

    @staticmethod
    def load(path) -> 'SplatRig':
        """A rig save wrote. Raises ValueError, saying why, for a file that is not one."""
        path = str(path)
        if not path.endswith('.npz'):
            path += '.npz'
        side = path[:-4] + '.json'
        name = os.path.basename(path)
        try:
            with open(side, encoding='utf-8') as fh:
                meta = json.load(fh)
        except (OSError, ValueError) as e:
            raise ValueError(f'{name} has no readable sidecar {os.path.basename(side)} ({e}).') from None
        if not isinstance(meta, dict) or meta.get('schema') != SCHEMA:
            raise ValueError(f'{os.path.basename(side)} is not a DexVantage splat rig ({SCHEMA}).')
        try:
            with np.load(path, allow_pickle=False) as z:
                g = RP.GaussianSet(z['means'], z['scales'], z['quats'], z['opacity'], z['colours'])
                frame, bg = z['frame'], z['background']
        except (OSError, ValueError, KeyError) as e:
            raise ValueError(f'{name} could not be read as a splat rig ({e}).') from None
        links = [tuple(x) for x in meta.get('links') or []]
        if len(frame) != g.n or (len(frame) and (frame.max() >= len(links) or frame.min() < -1)):
            raise ValueError(f'{name}: the Gaussians\' frames do not match the sidecar\'s links.')
        return SplatRig(g, frame, links, meta.get('q_ref') or {}, bg, dict(meta.get('robots') or {}),
                        bool(meta.get('synthetic')), dict(meta.get('info') or {}))


def stream_imager(sensor):
    """The imager a sensor's renders go through: its RGB imager (plucker.pick_imager) at the stream chosen for the
    sensor (Sensor.stream; its largest when none is chosen), the mode core/sim/scene.py renders the sensor at, so a
    splat render and the capture's MuJoCo views share one lens model (a 848 × 480 stream is not the 1280 × 720
    one scaled)."""
    from ..plucker import pick_imager, sensor_stream
    return pick_imager(sensor, sensor_stream(sensor))


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return str(o)


# =========================================================================== settings
@dataclass
class FitSettings:
    """The fit's settings; every figure is a DexVantage default (the learning rates follow Vantage's port of 3DGS's
    optimiser), not a measured optimum."""
    iterations: int = 300
    time_budget_s: float = 120.0
    seed: int = 0
    device: str = 'auto'               # auto: CUDA when present, else the CPU; cpu; cuda
    gpu_mem_gb: float = 3.0
    dtype: str = 'float32'
    init_stride: int = 1               # back-project every n-th pixel row and column
    voxel_m: float = 0.004             # the thinning voxel (grown until max_gaussians fits)
    max_gaussians: int = 20000
    robot_share: float = 0.5           # the robot's links get at least this share of max_gaussians
    mesh_points: int = 400             # per link, for a capture without depth
    holdout_every: int = 8             # 0: no held-out views
    optimise_positions: bool = True
    max_shift_m: float = 0.01
    lr_means_per_m: float = 1e-3       # × the scene's extent (m), decaying to 1 % over the iterations
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacity: float = 0.05
    lr_colours: float = 0.02
    lr_background: float = 0.02
    segment: str = 'auto'              # auto: 'masks' when the capture has segmentation, else 'meshes'; or METHODS
    max_dist_m: float = 0.01           # meshes: farther than this from every link's mesh: the world
    mesh_spacing_m: float = 0.002

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d) -> 'FitSettings':
        s = FitSettings()
        for k, v in (d or {}).items():
            if hasattr(s, k) and not isinstance(v, (dict, list)):
                cur = getattr(s, k)
                try:
                    setattr(s, k, type(cur)(v) if not isinstance(cur, bool) else bool(v))
                except (TypeError, ValueError):
                    pass
        return s


def _check(st: FitSettings, cap: Capture):
    if not cap.views:
        raise StudyError('The capture has no views: render or record some first.')
    if st.iterations < 0 or st.time_budget_s <= 0 or st.max_gaussians < 1:
        raise StudyError('Iterations must be zero or more, the time budget and the Gaussian budget positive.')
    if st.segment not in METHODS + ('auto', 'none'):
        raise StudyError(f'Unknown segmentation method {st.segment}: use masks, meshes, both or auto.')


# =========================================================================== initialisation
def _owner_frames(cap: Capture, links: list) -> np.ndarray:
    """owner index -> frame index (−1 for anything that is not a robot link), extending links."""
    lut = np.full(len(cap.owners) + 1, -1, np.int32)
    for i, o in enumerate(cap.owners):
        if o and o[0] == 'robot':
            key = (o[1], o[2])
            if key not in links:
                links.append(key)
            lut[i] = links.index(key)
    return lut


def init_rig(project, cap: Capture, st: FitSettings | None = None, robot_uid=None) -> SplatRig:
    """The rig before fitting (module docstring, Initialisation)."""
    from ..robot.model import get_model
    st = st or FitSettings()
    _check(st, cap)
    uid = _robot_uid(project, robot_uid)
    m = get_model(project, uid)
    links: list = [(uid, ln) for ln in m.link_names]
    lut = _owner_frames(cap, links)
    pts = {}                            # frame -> [points (k, 3) in the frame, colours (k, 3)]
    tmp = SplatRig(RP.GaussianSet.concat([]), np.zeros(0), links, cap.q_ref)
    have_depth = any(v.depth is not None and v.seg is not None for v in cap.views)
    for v in cap.views:
        if v.depth is None or v.seg is None:
            continue
        X, rc = RP.backproject(v.depth, v.camera, v.T_world_optical, st.init_stride)
        if not len(X):
            continue
        seg = np.asarray(v.seg)[rc[:, 0], rc[:, 1]]
        fr = np.where(seg >= 0, lut[np.clip(seg, 0, len(lut) - 1)], -1)
        col = np.asarray(v.rgb, float)[rc[:, 0], rc[:, 1]] / 255.0
        Tl = tmp.link_transforms(project, v.q)
        for f in np.unique(fr):
            sel = fr == f
            P = X[sel]
            if f >= 0:
                Ti = np.linalg.inv(Tl[f])
                P = P @ Ti[:3, :3].T + Ti[:3, 3]
            pts.setdefault(int(f), []).append((P, col[sel]))
    if not have_depth:
        for k, (u, ln) in enumerate(links):
            P, C = _mesh_samples(get_model(project, u), ln, st.mesh_points, st.seed + k)
            if len(P):
                pts.setdefault(k, []).append((P, C))
    if not pts:
        raise StudyError('The capture shows nothing to start from: no view has depth on any surface, and the robot '
                         'has no visual meshes.')
    counts = {f: sum(len(p) for p, _ in v) for f, v in pts.items()}
    total = sum(counts.values())
    shares = _budget_shares(counts, st.robot_share)
    sets, frames = [], []
    for f in sorted(pts):
        P = np.concatenate([p for p, _ in pts[f]])
        C = np.concatenate([c for _, c in pts[f]])
        budget = max(1, int(st.max_gaussians * shares[f]))
        Cen, vox, mc = RP.thin_to_budget(P, st.voxel_m, budget, C)
        sc, qu = RP.shapes_from_points(Cen, vox)
        sets.append(RP.GaussianSet(Cen, sc, qu, np.full(len(Cen), INIT_OPACITY), np.clip(mc, 0.0, 1.0)))
        frames.append(np.full(len(Cen), f, np.int32))
    g = RP.GaussianSet.concat(sets)
    fr = np.concatenate(frames)
    per = {('world' if f < 0 else links[f][1]): int((fr == f).sum()) for f in sorted(pts)}
    return SplatRig(g, fr, links, cap.q_ref, _background_guess(cap),
                    {uid: {'name': project.robot(uid).name, 'digest': robot_digest(project, uid)}}, cap.synthetic,
                    {'init': {'points': int(total), 'gaussians': g.n, 'from': 'depth' if have_depth else 'meshes',
                              'per_frame': per}})


def _budget_shares(counts: dict, robot_share: float) -> dict:
    """Each frame's share of the Gaussian budget: by point count, the robot's frames together given at least
    robot_share of it (the background's points outnumber the robot's in a wide view)."""
    rob = sum(c for f, c in counts.items() if f >= 0)
    world = counts.get(-1, 0)
    if not rob or not world:
        tot = max(rob + world, 1)
        return {f: c / tot for f, c in counts.items()}
    r = max(rob / (rob + world), min(max(robot_share, 0.0), 1.0))
    return {f: (r * c / rob if f >= 0 else 1.0 - r) for f, c in counts.items()}


def _background_guess(cap: Capture) -> np.ndarray:
    """The mean colour of the pixels with no depth (the renderer's background), else mid grey."""
    acc, n = np.zeros(3), 0
    for v in cap.views:
        if v.depth is None:
            continue
        sel = ~(np.asarray(v.depth) > 0)
        if sel.any():
            acc += np.asarray(v.rgb, float)[sel].sum(0) / 255.0
            n += int(sel.sum())
    return (acc / n if n else np.full(3, 0.5)).astype(np.float32)


def _mesh_samples(m, link: str, n: int, seed: int):
    """n points sampled on a link's visual meshes by area, in the link frame, with the meshes' colours."""
    meshes = [(V, F, c) for ln, V, F, c in m.visual_meshes() if ln == link and len(F)]
    if not meshes or n <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    tris = np.concatenate([V[F] for V, F, _ in meshes])
    cols = np.concatenate([np.repeat(np.asarray(c, float)[None, :3], len(F), 0) for _V, F, c in meshes])
    return _sample_triangles(tris, n, np.random.default_rng(seed), cols)


def _sample_triangles(tris, n: int, rng, cols=None):
    a = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    if a.sum() <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    k = rng.choice(len(tris), size=int(n), p=a / a.sum())
    r1, r2 = rng.random(int(n)), rng.random(int(n))
    s = np.sqrt(r1)
    P = (1 - s)[:, None] * tris[k, 0] + (s * (1 - r2))[:, None] * tris[k, 1] + (s * r2)[:, None] * tris[k, 2]
    return P, (np.full((len(P), 3), 0.5) if cols is None else cols[k])


# =========================================================================== fitting (torch)
class _TorchRig:
    def __init__(self, rig: SplatRig, dev, dtype, grad: bool, positions: bool):
        import torch
        self.rig, self.dev, self.dt = rig, dev, dtype

        def T(a, dt=None):
            return torch.tensor(np.asarray(a), dtype=dt or dtype, device=dev)
        g = rig.gaussians
        # a fit starts its colours and background inside [0.01, 0.99] (the logits' gradients vanish at 0 and 1); a
        # render that scores a fitted rig keeps them as fitted (clamping a learnt black background to 0.01 cost the
        # held-out SSIM 0.1 to 0.3 on the samples' dark backgrounds: tests/test_splat_views.py)
        lo = 0.01 if grad else 1e-6
        self.means = T(g.means).requires_grad_(grad and positions)
        self.log_s = torch.log(T(g.scales).clamp(min=1e-5)).requires_grad_(grad)
        self.quat = T(g.quats).requires_grad_(grad)
        self.op_logit = torch.logit(T(g.opacity).clamp(1e-4, 1 - 1e-4)).requires_grad_(grad)
        self.col_logit = torch.logit(T(g.colours).clamp(lo, 1 - lo)).requires_grad_(grad)
        self.bg_logit = torch.logit(T(rig.background).clamp(lo, 1 - lo)).requires_grad_(grad)
        f = np.where(rig.frame < 0, len(rig.links), rig.frame)
        self.frame = T(f, torch.long)

    def write_back(self):
        import torch
        with torch.no_grad():
            q = self.quat.detach()
            q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-12)
            self.rig.gaussians = RP.GaussianSet(
                self.means.detach().cpu().numpy(), torch.exp(self.log_s).cpu().numpy(), q.cpu().numpy(),
                torch.sigmoid(self.op_logit).cpu().numpy(), torch.sigmoid(self.col_logit).cpu().numpy())
            self.rig.background = torch.sigmoid(self.bg_logit).cpu().numpy().astype(np.float32)

    def render(self, Tl, cam, T_world_optical, want_index: bool = False):
        """Tl (L + 1, 4, 4) torch link transforms; returns splat's dict with 'rgb' (H, W, 3)."""
        import torch
        R = Tl[self.frame, :3, :3]
        t = Tl[self.frame, :3, 3]
        P = torch.einsum('nij,nj->ni', R, self.means) + t
        Rg = R @ SR.quat_to_rot(self.quat)
        Pc, Rc = RP.to_camera(P, Rg, T_world_optical)
        cov = SR.covariance(Rc, torch.exp(self.log_s))
        H, W = cam.height, cam.width
        bg = torch.sigmoid(self.bg_logit)[None].expand(H * W, 3)
        out = SR.splat(Pc, cov, torch.sigmoid(self.col_logit), torch.sigmoid(self.op_logit), cam, bg=bg,
                       want_index=want_index)
        out['rgb'] = out['image'].reshape(H, W, 3)
        return out


def held_out(k: int, every: int) -> bool:
    return every > 0 and k % every == every // 2


def _scores(trig: _TorchRig, views, Tls, idx) -> dict:
    import torch
    per = []
    with torch.no_grad():
        for k in idx:
            v = views[k]
            pred = trig.render(Tls[k], v.camera, v.T_world_optical)['rgb'].clamp(0, 1).cpu().numpy().astype(float)
            s = RP.image_scores(pred, np.asarray(v.rgb, float) / 255.0)
            per.append((s['psnr_db'], s['ssim']))
    if not per:
        return {'psnr_db': None, 'ssim': None, 'views': 0}
    return {'psnr_db': round(float(np.mean([p for p, _ in per])), 3),
            'ssim': round(float(np.mean([s for _, s in per])), 4), 'views': len(per)}


def fit_splat_rig(project, cap: Capture, settings: FitSettings | None = None, progress=None, cancel=None,
                  robot_uid=None, rig: SplatRig | None = None) -> tuple:
    """Fit a splat rig to a capture (module docstring): (SplatRig, report). settings: FitSettings (or a dict);
    progress(fraction, text) and cancel() -> bool as core.jobs; rig: start from this rig instead of init_rig's.
    Raises StudyError (plain words) for inputs it cannot work with, including a missing PyTorch."""
    st = settings if isinstance(settings, FitSettings) else FitSettings.from_dict(settings)
    _check(st, cap)
    if not SR.torch_ok():
        raise StudyError(SR.TORCH_MISSING.format(what='Fitting a splat rig'))
    import torch
    stop = cancel or (lambda: False)

    def prog(f, text):
        if progress is not None:
            progress(min(max(float(f), 0.0), 1.0), text)
    prog(0.0, 'Initialising the Gaussians from the capture')
    rig = rig or init_rig(project, cap, st, robot_uid)
    rng = np.random.default_rng(st.seed)
    torch.manual_seed(int(st.seed))
    dev = SR.device_for(st.device, st.gpu_mem_gb)
    dt = getattr(torch, st.dtype)
    views = cap.views
    Tls = [torch.tensor(rig.link_transforms(project, v.q), dtype=dt, device=dev) for v in views]
    train = [k for k in range(len(views)) if not held_out(k, st.holdout_every)]
    held = [k for k in range(len(views)) if held_out(k, st.holdout_every)]
    if not train:
        raise StudyError('Every view is held out: lower holdout_every or add views.')
    imgs = {k: torch.tensor(np.asarray(views[k].rgb, np.float32) / 255.0, dtype=dt, device=dev) for k in train}
    trig = _TorchRig(rig, dev, dt, True, st.optimise_positions)
    before = _scores(trig, views, Tls, held)
    extent = float(max(np.linalg.norm(np.ptp(rig.posed(project, rig.q_ref).means, axis=0)) if rig.n else 0.0, 1e-3))
    lr_means = st.lr_means_per_m * extent
    groups = [{'params': [trig.log_s], 'lr': st.lr_scales}, {'params': [trig.quat], 'lr': st.lr_quats},
              {'params': [trig.op_logit], 'lr': st.lr_opacity}, {'params': [trig.col_logit], 'lr': st.lr_colours},
              {'params': [trig.bg_logit], 'lr': st.lr_background}]
    if st.optimise_positions:
        groups.append({'params': [trig.means], 'lr': lr_means})
    opt = torch.optim.Adam(groups, eps=1e-15)
    with torch.no_grad():
        means0 = trig.means.detach().clone()
        log_s_max = trig.log_s.detach().clone() + math.log(MAX_GROWTH)
    t0 = time.perf_counter()
    losses, it, cut, cancelled, stopped = [], 0, 0, False, 'iterations'
    order = rng.permutation(train)
    while it < st.iterations:
        if stop():
            cancelled, stopped = True, 'cancelled'
            break
        el = time.perf_counter() - t0
        if el > st.time_budget_s:
            stopped = 'time budget'
            break
        if st.optimise_positions:
            groups[-1]['lr'] = lr_means * (0.01 ** (it / max(st.iterations, 1)))
        if it and it % len(train) == 0:
            order = rng.permutation(train)
        k = int(order[it % len(train)])
        v = views[k]
        out = trig.render(Tls[k], v.camera, v.T_world_optical)
        cut += out['cut']
        pred, tgt = out['rgb'], imgs[k]
        l1 = (pred - tgt).abs().mean()
        ss = SR.ssim_map(pred, tgt).mean()
        loss = 0.8 * l1 + 0.2 * (1 - ss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            trig.log_s.copy_(torch.minimum(trig.log_s, log_s_max))
            if st.optimise_positions:
                dm = trig.means - means0
                n_ = torch.linalg.norm(dm, dim=1, keepdim=True)
                trig.means.copy_(means0 + dm * torch.clamp(st.max_shift_m / n_.clamp(min=1e-12), max=1.0))
        if it % 10 == 0:
            lv = float(loss.detach())
            losses.append(round(lv, 5))
            prog(0.05 + 0.8 * min(it / max(st.iterations, 1), el / st.time_budget_s),
                 f'Fitting the splat rig · step {it} · loss {lv:.4f}')
        it += 1
    secs = time.perf_counter() - t0
    trig.write_back()
    keep = rig.gaussians.opacity >= PRUNE_OPACITY
    pruned = int((~keep).sum())
    if pruned:
        rig.gaussians = rig.gaussians.subset(keep)
        rig.frame = rig.frame[keep]
    report = {'iterations': it, 'seconds': round(secs, 3), 'device': str(dev), 'backend': 'torch',
              'dtype': st.dtype, 'gaussians': rig.n, 'pruned': pruned, 'train_views': len(train),
              'held_out_views': len(held), 'loss_first': losses[0] if losses else None,
              'loss_last': losses[-1] if losses else None, 'loss_curve': losses, 'tile_entries_cut': int(cut),
              'stopped_by': stopped, 'cancelled': cancelled, 'seed': int(st.seed), 'settings': st.to_dict(),
              'held_out_before': before, 'lpips': None, 'lpips_note': LPIPS_NOTE, 'extent_m': round(extent, 4),
              'deterministic': dev.type == 'cpu'}
    if not cancelled:
        method = st.segment if st.segment != 'auto' else ('masks' if any(v.seg is not None for v in cap.views)
                                                           else 'meshes')
        if method != 'none':
            prog(0.88, 'Segmenting the robot\'s Gaussians per link')
            labels, seg = segment_links(rig, project, cap, method, st.max_dist_m, st.mesh_spacing_m, device=dev)
            seg['changed'] = int((labels != rig.frame).sum())
            relabel(rig, project, labels)
            report['segmentation'] = seg
    trig2 = _TorchRig(rig, dev, dt, False, False)
    Tls = [torch.tensor(rig.link_transforms(project, v.q), dtype=dt, device=dev) for v in views]
    report['held_out_after'] = _scores(trig2, views, Tls, held)
    report['train_after'] = _scores(trig2, views, Tls, train[:: max(1, len(train) // 8)])
    report['synthetic_capture'] = bool(cap.synthetic)
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    rig.info.update(fit=report, capture=_capture_info(cap), provenance=_provenance(project))
    prog(1.0, 'Splat rig fitted' if not cancelled else 'Stopped: the rig holds what was fitted so far')
    return rig, report


def _capture_info(cap: Capture) -> dict:
    return {'views': len(cap.views), 'synthetic': bool(cap.synthetic), 'source': cap.source,
            'sizes': sorted({(v.camera.width, v.camera.height) for v in cap.views})}


def _provenance(project) -> dict:
    from .. import provenance as PV
    try:
        return {'project_sha256': PV.project_digest(project)['sha256'], 'environment': PV.scrub(PV.environment())}
    except Exception as e:                               # noqa: BLE001 — provenance is a record, never a failure
        return {'note': f'provenance not recorded: {e}'}


# =========================================================================== segmentation
def _centre_pixels(rig: SplatRig, project, v: CaptureView):
    """(flat pixel index (N,), −1 off the image or outside the model; optical z (N,)) of every Gaussian's centre in a
    view, the rig posed at the view's q (nearest pixel: OpenCV centres at integers)."""
    g = rig.posed(project, v.q)
    W2C = np.linalg.inv(np.asarray(v.T_world_optical, float))
    Pc = g.means.astype(float) @ W2C[:3, :3].T + W2C[:3, 3]
    uv, ok = v.camera.project(Pc)
    u = np.rint(np.where(ok, uv[:, 0], -1)).astype(np.int64)
    w = np.rint(np.where(ok, uv[:, 1], -1)).astype(np.int64)
    ok &= (u >= 0) & (u < v.camera.width) & (w >= 0) & (w < v.camera.height)
    return np.where(ok, w * v.camera.width + u, -1), Pc[:, 2]


def segment_links(rig: SplatRig, project, cap: Capture | None = None, method: str = 'masks',
                  max_dist_m: float = 0.01, mesh_spacing_m: float = 0.002, device=None,
                  depth_tol_m: float = DEPTH_TOL_M, depth_tol_rel: float = DEPTH_TOL_REL,
                  depth_tol_sigma: float = DEPTH_TOL_SIGMA) -> tuple:
    """Per-Gaussian frame labels (N,) (−1 the world, k links[k]) by method (module docstring) and a report
    {'method', 'voted', 'by_meshes', 'unchanged', 'per_frame'}. The rig is not changed (relabel applies labels).
    masks: each view votes for every Gaussian whose centre (posed at the view's q) projects onto a pixel with depth d
    and lies within depth_tol_m + depth_tol_rel · d + depth_tol_sigma · (its largest σ) of that depth (defaults):
    the Gaussian is on the surface the pixel sees, so a hidden one, or one posed elsewhere by a wrong label, does not
    vote."""
    if method not in METHODS:
        raise StudyError(f'Unknown segmentation method {method}: use masks, meshes or both.')
    labels = rig.frame.copy()
    rep = {'method': method, 'voted': 0, 'by_meshes': 0}
    voted = np.zeros(rig.n, bool)
    if method in ('masks', 'both'):
        if cap is None or not any(v.seg is not None and v.depth is not None for v in cap.views):
            raise StudyError('Segmentation by masks needs a capture with per-pixel segmentation and depth; use '
                             'meshes.')
        lut = _owner_frames(cap, rig.links)              # robot links the rig does not have yet are appended
        L = len(rig.links)
        votes = np.zeros((rig.n, L + 1), np.int64)
        smax = rig.gaussians.scales.max(1).astype(float)
        for v in cap.views:
            if v.seg is None or v.depth is None:
                continue
            pix, z = _centre_pixels(rig, project, v)
            seg = np.asarray(v.seg).reshape(-1)
            d = np.asarray(v.depth, float).reshape(-1)
            gi = np.flatnonzero(pix >= 0)
            px = pix[gi]
            tol = depth_tol_m + depth_tol_rel * d[px] + depth_tol_sigma * smax[gi]
            ok = (d[px] > 0) & (np.abs(z[gi] - d[px]) <= tol)
            gi, px = gi[ok], px[ok]
            fr = np.where(seg[px] >= 0, lut[np.clip(seg[px], 0, len(lut) - 1)], -1)
            np.add.at(votes, (gi, np.where(fr < 0, L, fr)), 1)
        voted = votes.sum(1) > 0
        best = votes.argmax(1)
        labels[voted] = np.where(best[voted] == L, -1, best[voted])
        rep['voted'] = int(voted.sum())
        rep['depth_tol'] = {'m': float(depth_tol_m), 'relative': float(depth_tol_rel), 'sigma': float(depth_tol_sigma)}
    if method in ('meshes', 'both'):
        todo = ~voted if method == 'both' else np.ones(rig.n, bool)
        if todo.any():
            labels[todo] = mesh_labels(rig, project, rig.posed(project, rig.q_ref).means[todo], max_dist_m,
                                       mesh_spacing_m)
        rep['by_meshes'] = int(todo.sum())
    rep['unchanged'] = int((labels == rig.frame).sum())
    rep['per_frame'] = {('world' if f < 0 else rig.links[f][1]): int((labels == f).sum()) for f in np.unique(labels)}
    rep['max_dist_m'] = float(max_dist_m) if method != 'masks' else None
    return labels, rep


def link_surface_samples(rig: SplatRig, project, q=None, spacing_m: float = 0.002, cap_per_link: int = 50000,
                         seed: int = 0):
    """(points (K, 3) world at q, frame index (K,)): every rig link's visual meshes sampled by area at about one
    point per spacing_m² (at most cap_per_link per link), plus their vertices."""
    from ..robot.model import get_model
    rng = np.random.default_rng(seed)
    T = rig.link_transforms(project, q)
    P, F = [], []
    meshes = {}
    for uid in {u for u, _ in rig.links}:
        for ln, V, Fc, _c in get_model(project, uid).visual_meshes():
            meshes.setdefault((uid, ln), []).append((V, Fc))
    for k, key in enumerate(rig.links):
        ms = meshes.get(key)
        if not ms:
            continue
        tris = np.concatenate([V[Fc] for V, Fc in ms if len(Fc)])
        area = float(0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1).sum())
        n = int(min(cap_per_link, max(1, area / spacing_m ** 2)))
        S, _ = _sample_triangles(tris, n, rng)
        S = np.concatenate([S, np.concatenate([V for V, _ in ms])])
        P.append(S @ T[k, :3, :3].T + T[k, :3, 3])
        F.append(np.full(len(S), k, np.int32))
    if not P:
        return np.zeros((0, 3)), np.zeros(0, np.int32)
    return np.concatenate(P), np.concatenate(F)


def mesh_labels(rig: SplatRig, project, points, max_dist_m: float = 0.01, spacing_m: float = 0.002,
                q=None) -> np.ndarray:
    """Frame labels of world points at q (default the rig's q_ref): the link whose mesh samples are nearest, within
    max_dist_m, else −1."""
    from scipy.spatial import cKDTree
    S, F = link_surface_samples(rig, project, rig.q_ref if q is None else q, spacing_m)
    P = np.asarray(points, float).reshape(-1, 3)
    if not len(S) or not len(P):
        return np.full(len(P), -1, np.int32)
    d, i = cKDTree(S).query(P, k=1)
    return np.where(d <= max_dist_m, F[i], -1).astype(np.int32)


def relabel(rig: SplatRig, project, labels) -> int:
    """Move the Gaussians whose label differs into their new frame, keeping their world pose at the rig's q_ref.
    Returns how many moved."""
    labels = np.asarray(labels, np.int32).reshape(-1)
    ch = labels != rig.frame
    if not ch.any():
        return 0
    T = rig.link_transforms(project, rig.q_ref)
    M = np.linalg.inv(T[labels[ch]]) @ T[rig.frame[ch]]           # new frame <- old frame, at q_ref
    moved = rig.gaussians.subset(ch).transformed(M)
    g = rig.gaussians
    g.means[ch], g.quats[ch] = moved.means, moved.quats
    rig.frame = labels.copy()
    return int(ch.sum())


def static_frames(rig: SplatRig, project, trials: int = 8, seed: int = 0) -> list:
    """The rig's link frames that do not move with q (the base and links fixed to it): their T_world_link is the same
    (within 1e-9) at the home pose and at `trials` random joint states. A Gaussian on one of them moves exactly as a
    world Gaussian does, so a label confusion among them and the world does not change any render."""
    T0 = rig.link_transforms(project, None)
    same = np.ones(len(rig.links), bool)
    for q in random_qs(project, trials + 1, seed, spread_rad=1.0, robot_uid=rig.links[0][0] if rig.links else None):
        T = rig.link_transforms(project, q)
        same &= np.abs(T[:-1] - T0[:-1]).reshape(len(rig.links), -1).max(1) < 1e-9
    return [int(k) for k in np.flatnonzero(same)]


def mislabel_rate(pred, truth, links=None, static=None) -> dict:
    """{'rate', 'wrong', 'n', 'robot_rate' (over the Gaussians the truth puts on the robot), 'moving_rate' (static
    frames, static_frames, counted as the world: the confusions that change a re-posed render; None without static),
    'per_frame': {frame: {'n', 'wrong'}}} of predicted frame labels against true ones."""
    p = np.asarray(pred).reshape(-1)
    t = np.asarray(truth).reshape(-1)
    n = len(t)
    wrong = p != t
    rob = t >= 0
    per = {}
    for f in np.unique(t):
        name = 'world' if f < 0 else (links[f][1] if links else int(f))
        per[name] = {'n': int((t == f).sum()), 'wrong': int((wrong & (t == f)).sum())}
    moving = None
    if static is not None:
        st = np.asarray(list(static) + [-1])
        pm, tm = np.where(np.isin(p, st), -1, p), np.where(np.isin(t, st), -1, t)
        moving = float((pm != tm).mean()) if n else None
    return {'rate': float(wrong.mean()) if n else None, 'wrong': int(wrong.sum()), 'n': int(n),
            'robot_rate': float(wrong[rob].mean()) if rob.any() else None, 'moving_rate': moving, 'per_frame': per}
