"""Multi-cell comparison: several robot cells (several labs' or benches' rigs: .dexv files, samples, or variants of one
project) side by side, on four questions.

1. **Figures of merit.** Every named metric asked for (core/metrics: the default is the quick figures, DEFAULT_METRICS,
   a default; 'all' is every metric of every study that reads no recorded dataset) evaluated on every cell, in one
   table with the best and worst cell per metric. With a tolerance Monte Carlo (core/tolerance.monte_carlo, n draws
   per cell, off by default) each difference from the reference cell is called meaningful when it exceeds what the
   build tolerances alone spread it by: |Δ| > k·√(σ_ref² + σ_cell²), k = 2 (MEANINGFUL_K, a default: about the 95 %
   two-sided band of a normal difference), σ the standard deviation of the metric over the cell's own draws. A figure
   stated only as a bound is not called either way. With core/stats installed, each such difference must also be
   significant over the draws (stats.compare_samples: the permutation test of the mean difference, the p-values
   Holm-adjusted across the table at α = 0.05, its defaults).
2. **The geometric gap** a policy trained on one cell meets on another. Ported from Vantage core/rig_gap.py (the Rig
   Contrastive Distance: a weighted mean of components in 0..1, the Hungarian camera matches, the heuristic transfer
   bands), with the automotive parts gone (viewing directions around a vehicle, the ground seen in bird's-eye view,
   the body's occlusion, nuScenes and KITTI calibration readers). Here each camera has a **role**: 'wrist' (on a link
   that a joint moves) or 'scene' (static: the world, a cage, a tripod, or a link no joint moves, such as the G1's
   torso). A wrist camera is compared in its robot's **contact frame** (origin between the fingertips, z the approach
   axis, x the closing axis: core/exporters/canonical_warp.contact_frame, at the home pose), so wrist cameras of two
   different arms compare as the policy sees them, relative to the fingers; a scene camera in the world frame (the
   primary robot's base). Cameras are matched one to one within a role (scipy's linear_sum_assignment on axis angle /
   90° + centre distance / 0.5 m + |ΔHFOV| / 90° + 1 when they ride on different arms; which pad comes first in a
   contact frame is arbitrary, so a wrist camera is also tried half a turn about the approach axis and the closer
   reading kept, said as 'closing_flipped'). Per matched pair: the
   **viewpoint angle** between the optical axes (2-argument arctangent of |z_a × z_b| and z_a · z_b, so exact near 0)
   and the full rotation between the optical frames (roll included), the **distance** between the centres, the
   distance to the target (wrist: to the fingertip centre; scene: to the task volume's centre) and its ratio, the
   horizontal **FOV** difference and ratio, the **resolution ratio** (pixels per degree at the image centre, f_x·π/180)
   and the **image-plane coverage of the task volume**: the share of the image whose rays pass through a task volume
   (rays through a grid of pixel centres at most 128 wide, unprojected through the lens, slab-tested against each
   volume's box) and the share of the volume's interior points the camera sees in its image. Components (each 0..1):
   viewpoint min(1, angle/90°) (weight 0.30), position min(1, distance/0.5 m) (0.20), resolution
   min(1, |log2 ratio|/2) (two octaves = 1; 0.15), coverage |Δ image share| (0.15), fov min(1, |ΔHFOV|/90°) (0.10),
   topology ½(|n_a − n_b|/max(n_a, n_b) + share of cameras without a match in their role) (0.10); the pair components
   are means over the matches. The total is their weighted mean; the bands (< 0.15 small, < 0.35 moderate, else
   large) are Vantage's heuristic bands, not calibrated against any policy's success. Identical cells give 0; a camera
   turned by θ about its centre gives a viewpoint angle of θ (the validation's checks).
3. **Dataset pooling** (core/cells_pool.py): whether the cells' recordings can be pooled for training, against the
   LeRobot v2.1 / v3.0 feature schema: camera keys and roles, sizes, frame rates, codecs, joint names, order and
   count, action space and units, gripper encoding, control rate; the mapping that makes them poolable (rename,
   resize, resample, reorder, re-encode, rescale) and what cannot be mapped.
4. **Drift of one design built several times.** Ported from Vantage core/fleet.py (measured minus planned per sensor,
   the ok / adjust / re-mount verdict at one and three tolerances, the robust z-score of a copy against the others'
   median, the systematic offset that says the plan should follow the build, the fleet-median variant) and the
   measurement idea of fleet_twin.py (offsets from calibration results, each with its σ); its Kalman filter, knock
   test and forecasts are not ported (a cell has a few calibrations, not a time series). The reference cell is the
   plan; another cell is a copy of it for each camera whose name it shares (case-insensitive). A copy's camera is
   **measured** when the cell holds a solved hand-eye session for it (project.calibration, core/calib: the latest by
   date, its optical frame X in the mount frame and its σ), else compared **as designed** (its own mount: a design
   difference, not drift). Offsets in the mount frame (a link for a wrist camera, the world for a static one): Δx Δy Δz
   in mm and the rotation vector of R_measured · R_planᵀ in degrees; the share of the tolerance is the larger of
   max|Δ| / the plan's position tolerance and the rotation angle / its rotation tolerance (the mount tolerance, ± 1 mm
   and ± 0.5° by default, labelled so). Across two or more measured copies: the median offset, the robust spread
   1.4826 · MAD (never below a quarter of the tolerance), outliers beyond a robust z of 3.5 (three or more copies),
   and a systematic offset when the median is beyond the tolerance while most copies sit within tolerance of the
   median: the build is consistent, the plan should follow (the suggestion adds an as-built variant).

The page's suggestion (Apply, one undo step): with the open project not the reference, mount its least-alike matched
camera at the reference camera's pose (relative to its fingertips for a wrist camera, in the world for a scene
camera); with the open project the reference and a systematic drift, add a variant at the copies' median.

Known limits: the gap is geometric (viewpoint, resolution, coverage), not appearance (lighting, backgrounds, lenses'
blur and colour) and not a predicted success rate; wrist cameras are related at the home pose and through the contact
frame the grasp code measures (an assumed one for a robot without pads, said in the notes); occlusion of the task
volume by the robot is not in the coverage share; scene cameras of two cells compare in their own world frames, so
cells whose robots stand differently relative to the table differ for that reason too; the drift compares one
calibration per copy and camera (the latest; earlier ones are listed), offsets by small-angle rotation vectors,
and copies matched by camera name; the metrics are only those their studies state (each study's own limits apply);
the tolerance Monte Carlo per cell costs n evaluations of every study named.
"""
from __future__ import annotations


import html
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from .jobs import Ctx, StudyError
from .metrics import delta_unit

SCHEMA = 'dexvantage.cells/1'
SID = 'cells'
DEFAULT_METRICS = ('coverage.covered', 'coverage.seen_twice', 'coverage.ppmm_p5', 'depth.valid', 'sync.skew_ms',
                   'bandwidth.payload_mbps', 'rig.camera_count', 'rig.sensor_count', 'rig.mass_g', 'rig.power_w')
# studies whose metrics 'all' leaves out: they read a recorded dataset, or run other studies many times
SKIP_FOR_ALL = ('audit', 'plan_vs_recording', 'silhouette_calib', 'handeye_solve', 'tolerance', 'experiments',
                'optimise', 'design_agent', 'requirements', 'variants_compare', SID)
WEIGHTS = {'viewpoint': 0.30, 'position': 0.20, 'resolution': 0.15, 'coverage': 0.15, 'fov': 0.10, 'topology': 0.10}
BANDS = ((0.15, 'small', 'good'), (0.35, 'moderate', 'warning'), (math.inf, 'large', 'serious'))
BAND_NOTE = ('a heuristic band on the index (< 0.15 small, < 0.35 moderate, else large), ported from Vantage; not '
             'calibrated against any policy\'s success')
ANGLE_SCALE_DEG = 90.0          # a viewpoint angle of 90° counts fully (component 1)
POSITION_SCALE_M = 0.5          # a default: centres 0.5 m apart count fully (Vantage: 2 m around a vehicle)
FOV_SCALE_DEG = 90.0
RES_OCTAVES = 2.0               # two octaves of resolution difference count fully (Vantage's scale)
MATCH_ARM_COST = 1.0            # matching a camera on one arm to one on another arm costs this much more
RAY_COLS, RAY_ROWS_MAX = 128, 128
INTERIOR = 9                    # interior points per volume edge for the in-view share
MEANINGFUL_K = 2.0              # a default: a difference beyond 2 σ of the tolerance spread is meaningful
STATS_RESAMPLES = 1999          # permutation and bootstrap resamples for core/stats over the draws (a default)
REMOUNT = 3.0                   # beyond this many tolerances: re-mount (Vantage core/fleet.py)
SIGMA_FLOOR = 0.25              # the robust spread is never below this share of the tolerance (Vantage)
OUTLIER_Z = 3.5                 # robust z beyond which a copy differs from the others (Vantage)
MIN_FLEET = 3                   # copies a camera needs before outliers are named (Vantage)
DEFAULT_TOL = (1.0, 0.5)        # core/project.Tolerance's defaults (mm, degrees)
VERDICT_SEV = {'ok': 'good', 'adjust': 'warning', 're-mount': 'serious'}


# ====================================================================== the cell set
@dataclass
class Cell:
    name: str
    project: object
    source: str                    # 'open project' | 'variant' | 'file' | 'sample' | 'given'
    label: str = ''                # a file's name (no folder), a sample's name, a variant's name

    def summary(self) -> dict:
        p = self.project
        return {'name': self.name, 'source': self.source, 'label': self.label, 'project': p.name,
                'variant': p.variant.name, 'robots': [r.name for r in p.robots],
                'cameras': sum(s.enabled and s.kind in ('camera', 'depth') for s in p.variant.sensors)}


def _load_path(path: str):
    from .project import Project, ProjectFormatError
    try:
        return Project.load(os.fspath(path))
    except ProjectFormatError as e:
        raise StudyError(f'The cell {os.path.basename(os.path.normpath(str(path)))} could not be opened: {e}') \
            from None
    except OSError as e:
        raise StudyError(f'The cell {os.path.basename(os.path.normpath(str(path)))} could not be read: '
                         f'{e.strerror or e}.') from None


def _load_sample(name: str):
    from .project import ProjectFormatError
    from .robot.samples import sample_project
    try:
        return sample_project(name)
    except ProjectFormatError as e:
        raise StudyError(str(e)) from None


def _with_variant(project, v):
    p = project.snapshot()
    p.active_uid = v.uid
    return p


def make_cells(items=(), open_project=None, variants: bool = False) -> list:
    """[Cell]: the open project (when given; its active variant), its other variants (variants=True), then each item:
    a core.project.Project, a dexvantage.Rig (anything with a .project), a path to a .dexv file or folder,
    'sample:<name>', or a dict {'path'} | {'sample'} | {'variant': uid or name of the open project's}."""
    from .project import Project
    out = []
    if open_project is not None:
        out.append(Cell(_name(open_project), open_project.snapshot(), 'open project', open_project.variant.name))
        if variants:
            for v in open_project.variants:
                if v.uid != open_project.active_uid:
                    out.append(Cell(f'{open_project.name} · {v.name}', _with_variant(open_project, v), 'variant',
                                    v.name))
    for it in items or ():
        if isinstance(it, Cell):
            out.append(it)
            continue
        if hasattr(it, 'project') and isinstance(getattr(it, 'project'), Project):
            it = it.project
        if isinstance(it, Project):
            out.append(Cell(_name(it), it.snapshot(), 'given', os.path.basename(it.path or '')))
            continue
        if isinstance(it, str):
            it = {'sample': it[len('sample:'):]} if it.startswith('sample:') else {'path': it}
        if isinstance(it, os.PathLike):
            it = {'path': os.fspath(it)}
        if not isinstance(it, dict):
            raise StudyError('A cell is a project, a .dexv path, "sample:<name>" or a variant of the open project.')
        if it.get('sample'):
            p = _load_sample(str(it['sample']))
            out.append(Cell(_name(p), p, 'sample', str(it['sample'])))
        elif it.get('path'):
            p = _load_path(str(it['path']))
            out.append(Cell(_name(p), p, 'file', os.path.basename(os.path.normpath(str(it['path'])))))
        elif it.get('variant'):
            if open_project is None:
                raise StudyError('A variant cell needs the open project.')
            v = open_project.variant_by_uid(it['variant']) or next(
                (x for x in open_project.variants if x.name == it['variant']), None)
            if v is None:
                raise StudyError(f'The open project has no variant "{it["variant"]}".')
            out.append(Cell(f'{open_project.name} · {v.name}', _with_variant(open_project, v), 'variant', v.name))
        else:
            raise StudyError('A cell entry names no path, sample or variant.')
    seen: dict = {}
    for c in out:
        k = c.name
        if k in seen:
            seen[k] += 1
            c.name = f'{k} ({seen[k]})'
        else:
            seen[k] = 1
    return out


def _name(p) -> str:
    return p.name if len(p.variants) == 1 else f'{p.name} · {p.variant.name}'


# ====================================================================== cameras of a cell
@dataclass
class CamRec:
    name: str
    uid: str
    role: str                     # 'wrist' | 'scene'
    arm: int | None               # the robot's index for a wrist camera
    robot: str
    frame: str                    # what T is relative to, in words
    T: np.ndarray                 # the optical frame in the role's frame (4 × 4)
    T_world: np.ndarray           # the optical frame in the world
    T_world_role: np.ndarray      # the role's frame in the world
    camera: object                # optics.CameraModel at the stream size
    imager: str
    model_id: str
    hfov: float
    vfov: float
    px_per_deg: float
    target_m: float | None
    coverage: dict
    locked: bool = False
    notes: list = field(default_factory=list)

    def row(self) -> dict:
        p = self.T[:3, 3]
        return {'name': self.name, 'uid': self.uid, 'role': self.role, 'arm': self.arm, 'robot': self.robot,
                'frame': self.frame, 'imager': self.imager, 'model_id': self.model_id,
                'position_m': [float(v) for v in p], 'axis': [float(v) for v in self.T[:3, 2]],
                'width': self.camera.width, 'height': self.camera.height, 'hfov_deg': self.hfov,
                'vfov_deg': self.vfov, 'px_per_deg': self.px_per_deg, 'target_m': self.target_m,
                'coverage': dict(self.coverage), 'notes': list(self.notes)}


def _volumes(project) -> list:
    vs = [v for v in project.workspace.volumes if not v.refine] or list(project.workspace.volumes)
    return [v.bounds() for v in vs]


def ray_box_hits(O, D, lo, hi) -> np.ndarray:
    """(N,) whether rays O + t·D (t ≥ 0) meet the axis-aligned box [lo, hi] (the slab test; a ray starting inside
    meets it)."""
    D = np.asarray(D, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        inv = 1.0 / D
        t1 = (np.asarray(lo, float) - O) * inv
        t2 = (np.asarray(hi, float) - O) * inv
    tmin = np.where(np.isnan(t1), -np.inf, np.minimum(t1, t2))
    tmax = np.where(np.isnan(t2), np.inf, np.maximum(t1, t2))
    # a ray parallel to a slab: inside it for every t when the origin is between the planes, else never
    par = D == 0
    inside = (O >= lo) & (O <= hi)
    tmin = np.where(par, np.where(inside, -np.inf, np.inf), tmin)
    tmax = np.where(par, np.where(inside, np.inf, -np.inf), tmax)
    t0 = np.max(tmin, axis=-1)
    t1_ = np.min(tmax, axis=-1)
    return (t1_ >= np.maximum(t0, 0.0)) & np.isfinite(t1_)


def image_grid(camera) -> np.ndarray:
    """(N, 2) pixel centres of a grid of at most RAY_COLS columns over the image (OpenCV centres)."""
    cols = min(camera.width, RAY_COLS)
    rows = max(1, min(RAY_ROWS_MAX, int(round(camera.height * cols / camera.width))))
    us = (np.arange(cols) + 0.5) * camera.width / cols - 0.5
    vs = (np.arange(rows) + 0.5) * camera.height / rows - 0.5
    U, V = np.meshgrid(us, vs)
    return np.stack([U.ravel(), V.ravel()], 1)


def task_coverage(camera, T_world, boxes) -> dict:
    """{'image_pct': share of the image whose rays pass through a task volume (rays of image_grid), 'in_view_pct':
    share of the volumes' interior points inside the image, 'pixels': the image share in pixels}."""
    if not boxes:
        return {'image_pct': None, 'in_view_pct': None, 'pixels': None, 'reason': 'the cell has no task volume'}
    uv = image_grid(camera)
    rays, ok = camera.unproject(uv, with_ok=True)
    R, t = T_world[:3, :3], T_world[:3, 3]
    D = np.where(ok[:, None], rays, 0.0) @ R.T
    hit = np.zeros(len(uv), bool)
    for lo, hi in boxes:
        hit |= ok & ray_box_hits(np.broadcast_to(t, D.shape), D, lo, hi)
    img = float(hit.mean())
    pts = []
    g = (np.arange(INTERIOR) + 0.5) / INTERIOR
    for lo, hi in boxes:
        X, Y, Z = np.meshgrid(*[lo[i] + g * (hi[i] - lo[i]) for i in range(3)], indexing='ij')
        pts.append(np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1))
    P = np.concatenate(pts)
    _, seen = camera.project((P - t) @ R)
    return {'image_pct': 100.0 * img, 'in_view_pct': 100.0 * float(seen.mean()),
            'pixels': int(round(img * camera.width * camera.height))}


def _static_link(project, sensor) -> bool:
    from .calib.records import _static_link as SL
    return SL(project, sensor.mount.robot_uid, sensor.mount.parent_frame)


def cameras(project) -> tuple[list, list]:
    """([CamRec], notes) of the active variant's enabled cameras and depth sensors (their picked imager, the RGB one
    of a depth camera, at the stream size), at the home pose."""
    from . import camrender as CR
    from . import plucker as PK
    from .exporters.canonical_warp import WarpError, contact_frame
    from .robot.model import RobotModelError, get_model
    out, notes = [], []
    boxes = _volumes(project)
    centre = None
    if boxes:
        lo = np.min([b[0] for b in boxes], axis=0)
        hi = np.max([b[1] for b in boxes], axis=0)
        centre = 0.5 * (lo + hi)
    frames: dict = {}
    order = {r.uid: k for k, r in enumerate(project.robots)}
    for s in project.variant.sensors:
        if not s.enabled or s.kind not in ('camera', 'depth'):
            continue
        try:
            im = PK.pick_imager(s, PK.sensor_stream(s))
        except Exception as e:                       # noqa: BLE001 — a library entry that cannot be read: said
            notes.append(f'{s.name} is left out: its imager could not be read ({e}).')
            continue
        if im is None:
            notes.append(f'{s.name} is left out: it has no imager.')
            continue
        try:
            Tw = CR.optical_pose(project, s, im).matrix()
        except (KeyError, RobotModelError, ValueError) as e:
            notes.append(f'{s.name} is left out: it cannot be posed ({e}).')
            continue
        cn = []
        wrist = s.mount.on_link and project.robot(s.mount.robot_uid) is not None and not _static_link(project, s)
        if wrist:
            rob = project.robot(s.mount.robot_uid)
            key = rob.uid
            if key not in frames:
                try:
                    T_wc, g = contact_frame(project, rob.uid)
                    frames[key] = (T_wc, f'the contact frame of {rob.name} (at home)',
                                   [f'{rob.name}: {g["rule"]}.'] if g.get('assumed') else [])
                except WarpError as e:
                    m = get_model(project, rob.uid)
                    frames[key] = (m.link_pose(m.home, s.mount.parent_frame),
                                   f'the link {s.mount.parent_frame} of {rob.name} (no contact frame: {e})', [])
            T_wr, ftext, fn = frames[key]
            cn += fn
            arm, rname = order.get(rob.uid), rob.name
            target = float(np.linalg.norm((np.linalg.inv(T_wr) @ Tw)[:3, 3]))
        else:
            T_wr, ftext = np.eye(4), 'the world (the primary robot\'s base)'
            arm, rname = None, ''
            target = None if centre is None else float(np.linalg.norm(Tw[:3, 3] - centre))
        cam = im.camera
        fov = cam.fov_deg()
        out.append(CamRec(s.name, s.uid, 'wrist' if wrist else 'scene', arm, rname, ftext, np.linalg.inv(T_wr) @ Tw,
                          Tw, T_wr, cam, im.name, s.model_id or '', float(fov['h']), float(fov['v']),
                          float(cam.fx * math.pi / 180.0), target, task_coverage(cam, Tw, boxes), bool(s.locked), cn))
    return out, notes


# ====================================================================== the geometric gap
def axis_angle_deg(Ta, Tb) -> float:
    """Angle (degrees) between two optical axes (z), from atan2(|a × b|, a · b): exact near 0."""
    a, b = np.asarray(Ta)[:3, 2], np.asarray(Tb)[:3, 2]
    return float(math.degrees(math.atan2(float(np.linalg.norm(np.cross(a, b))), float(a @ b))))


def rotation_deg(Ra, Rb) -> float:
    """Rotation angle (degrees) from Ra to Rb (atan2 of the sine and cosine parts, as Vantage core/fleet.py)."""
    M = np.asarray(Ra).T @ np.asarray(Rb)
    s = 0.5 * math.sqrt((M[2, 1] - M[1, 2]) ** 2 + (M[0, 2] - M[2, 0]) ** 2 + (M[1, 0] - M[0, 1]) ** 2)
    c = 0.5 * (M[0, 0] + M[1, 1] + M[2, 2] - 1.0)
    return float(math.degrees(math.atan2(s, c)))


FLIP = np.diag([-1.0, -1.0, 1.0, 1.0])     # half a turn about the contact frame's approach axis (z)


def _raw_cost(Ta, Tb, a: CamRec, b: CamRec) -> float:
    return (axis_angle_deg(Ta, Tb) / ANGLE_SCALE_DEG + float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]))
            / POSITION_SCALE_M + abs(a.hfov - b.hfov) / FOV_SCALE_DEG + (MATCH_ARM_COST if a.arm != b.arm else 0.0)
            + rotation_deg(Ta[:3, :3], Tb[:3, :3]) / 1800.0)


def aligned(a: CamRec, b: CamRec) -> tuple[np.ndarray, bool]:
    """(b's pose in its role frame, flipped): which pad comes first in a contact frame is arbitrary, so a wrist camera
    is compared with b's turned half a turn about the approach axis when that is the closer reading."""
    if a.role != 'wrist':
        return b.T, False
    Tf = FLIP @ b.T
    return (Tf, True) if _raw_cost(a.T, Tf, a, b) < _raw_cost(a.T, b.T, a, b) - 1e-12 else (b.T, False)


def _cost(a: CamRec, b: CamRec) -> float:
    Tb, _ = aligned(a, b)
    return _raw_cost(a.T, Tb, a, b)


def match(A: list, B: list) -> tuple[list, list, list]:
    """([(a, b)], unmatched of A, unmatched of B): one-to-one within each role (Hungarian, module docstring)."""
    from scipy.optimize import linear_sum_assignment
    pairs, ua, ub = [], [], []
    for role in ('wrist', 'scene'):
        ra = [c for c in A if c.role == role]
        rb = [c for c in B if c.role == role]
        if not ra or not rb:
            ua += ra
            ub += rb
            continue
        C = np.array([[_cost(a, b) for b in rb] for a in ra])
        ii, jj = linear_sum_assignment(C)
        pairs += [(ra[i], rb[j]) for i, j in zip(ii, jj)]
        ua += [c for k, c in enumerate(ra) if k not in set(ii)]
        ub += [c for k, c in enumerate(rb) if k not in set(jj)]
    return pairs, ua, ub


def _ratio(b, a):
    return None if a in (None, 0) or b is None else float(b) / float(a)


def pair_figures(a: CamRec, b: CamRec) -> dict:
    ia, ib = a.coverage.get('image_pct'), b.coverage.get('image_pct')
    va, vb = a.coverage.get('in_view_pct'), b.coverage.get('in_view_pct')
    Tb, flipped = aligned(a, b)
    return {'a': a.name, 'b': b.name, 'role': a.role, 'frame_a': a.frame, 'frame_b': b.frame,
            'cost': round(_cost(a, b), 6),
            'viewpoint_deg': axis_angle_deg(a.T, Tb), 'rotation_deg': rotation_deg(a.T[:3, :3], Tb[:3, :3]),
            'distance_m': float(np.linalg.norm(a.T[:3, 3] - Tb[:3, 3])), 'closing_flipped': flipped,
            'target_a_m': a.target_m, 'target_b_m': b.target_m, 'target_ratio': _ratio(b.target_m, a.target_m),
            'hfov_a_deg': a.hfov, 'hfov_b_deg': b.hfov, 'hfov_diff_deg': b.hfov - a.hfov,
            'vfov_diff_deg': b.vfov - a.vfov, 'fov_ratio': _ratio(b.hfov, a.hfov),
            'size_a': [a.camera.width, a.camera.height], 'size_b': [b.camera.width, b.camera.height],
            'px_per_deg_a': a.px_per_deg, 'px_per_deg_b': b.px_per_deg,
            'resolution_ratio': _ratio(b.px_per_deg, a.px_per_deg),
            'image_pct_a': ia, 'image_pct_b': ib,
            'image_pct_diff': None if ia is None or ib is None else ib - ia,
            'in_view_pct_a': va, 'in_view_pct_b': vb, 'arm_a': a.arm, 'arm_b': b.arm}


def gap(A: list, B: list, name_a: str = 'A', name_b: str = 'B') -> dict:
    """The geometric gap between two cells' cameras (module docstring): components, total, band, matches, findings."""
    pairs, ua, ub = match(A, B)
    figs = [pair_figures(a, b) for a, b in pairs]
    comp = {}
    if figs:
        comp['viewpoint'] = float(np.mean([min(1.0, f['viewpoint_deg'] / ANGLE_SCALE_DEG) for f in figs]))
        comp['position'] = float(np.mean([min(1.0, f['distance_m'] / POSITION_SCALE_M) for f in figs]))
        comp['resolution'] = float(np.mean([min(1.0, abs(math.log2(f['resolution_ratio'])) / RES_OCTAVES)
                                            if f['resolution_ratio'] else 1.0 for f in figs]))
        comp['coverage'] = float(np.mean([min(1.0, abs(f['image_pct_diff']) / 100.0)
                                          if f['image_pct_diff'] is not None else 0.0 for f in figs]))
        comp['fov'] = float(np.mean([min(1.0, abs(f['hfov_diff_deg']) / FOV_SCALE_DEG) for f in figs]))
    else:
        comp.update({k: (0.0 if not A and not B else 1.0) for k in WEIGHTS if k != 'topology'})
    na, nb = len(A), len(B)
    comp['topology'] = 0.5 * (abs(na - nb) / max(na, nb, 1) + (len(ua) + len(ub)) / max(na + nb, 1))
    comp = {k: float(np.clip(comp[k], 0.0, 1.0)) for k in WEIGHTS}
    total = float(sum(WEIGHTS[k] * comp[k] for k in WEIGHTS) / sum(WEIGHTS.values()))
    word, sev = next((w, s) for t, w, s in BANDS if total < t)
    f = [f'Gap index {total:.3f} ({word}: {BAND_NOTE}).']
    big = max(comp, key=comp.get)
    if comp[big] > 0:
        f.append(f'The largest component is {big} ({comp[big]:.2f}).')
    for x in sorted(figs, key=lambda x: -x['cost'])[:3]:
        if x['cost'] <= 1e-9:
            continue
        f.append(f'{x["a"]} ↔ {x["b"]} ({x["role"]}): axes {x["viewpoint_deg"]:.1f}° apart, centres '
                 f'{x["distance_m"] * 1000:.0f} mm apart, HFOV {x["hfov_diff_deg"]:+.1f}°'
                 + (f', {x["resolution_ratio"]:.2f}× the pixels per degree' if x['resolution_ratio'] else '')
                 + (f', task volume over {x["image_pct_a"]:.0f} % → {x["image_pct_b"]:.0f} % of the image'
                    if x['image_pct_diff'] is not None and abs(x['image_pct_diff']) >= 1 else '') + '.')
    if ua:
        f.append(f'{name_a} has cameras with no counterpart in {name_b}: {", ".join(c.name for c in ua)}.')
    if ub:
        f.append(f'{name_b} has cameras with no counterpart in {name_a}: {", ".join(c.name for c in ub)}.')
    return {'a': name_a, 'b': name_b, 'total': total, 'band': word, 'status': sev, 'components': comp,
            'weights': dict(WEIGHTS), 'matches': figs, 'unmatched_a': [c.name for c in ua],
            'unmatched_b': [c.name for c in ub], 'cameras': [na, nb], 'findings': f}


# ====================================================================== figures of merit
def metric_ids(spec) -> list:
    """The metric ids a setting names: None → DEFAULT_METRICS; 'all' → every metric outside SKIP_FOR_ALL."""
    from . import metrics as M
    M.load_metrics()
    if spec is None or spec == [] or spec == 'default':
        return list(DEFAULT_METRICS)
    if spec == 'all':
        return [m.id for m in sorted(M.METRICS.values(), key=lambda m: m.id) if m.study not in SKIP_FOR_ALL]
    if isinstance(spec, str):
        spec = [x.strip() for x in spec.split(',') if x.strip()]
    ids = list(dict.fromkeys(str(x) for x in spec))
    for i in ids:
        M.get(i)                                       # StudyError with the sentence for an unknown id
    return ids


def metrics_table(cells: list, ids: list, ref: int, draws: int = 0, seed: int = 1, ctx: Ctx | None = None,
                  evaluator=None) -> dict:
    """{'rows': [...], 'tolerance': {...}} (module docstring, part 1). evaluator(project, ids, ctx) -> the
    core.metrics.evaluate shape (tests pass their own)."""
    from . import jobs
    from . import metrics as M
    from . import tolerance as TL
    ctx = ctx or Ctx()
    cache = jobs.ResultCache(64)
    ev = evaluator or (lambda p, i, c: M.evaluate(p, i, cache=cache, ctx=c))
    vals, spread = [], []
    n = len(cells)
    for k, c in enumerate(cells):
        if ctx.cancelled():
            break
        ctx.report(k / max(1, n), f'Figures of {c.name}')
        inner = Ctx(cancel=ctx.cancel)
        vals.append(ev(c.project, ids, inner))
        if draws and draws >= 2:
            try:
                mc = TL.monte_carlo(c.project, ids, int(draws), int(seed), ctx=inner,
                                    evaluator=lambda p, i, cc: ev(p, i, cc))
                spread.append({m: {kk: r.get(kk) for kk in ('std', 'p5', 'p95', 'valid', 'values')}
                               for m, r in mc['metrics'].items()})
            except StudyError as e:
                spread.append({'_error': str(e)})
        else:
            spread.append(None)
    rows = []
    for mid in ids:
        m = M.get(mid)
        cells_v = [(v.get(mid) or {}) for v in vals]
        values = [r.get('value') for r in cells_v]
        bounds = [r.get('bound') for r in cells_v]
        plain = [(k, v) for k, (v, b) in enumerate(zip(values, bounds)) if v is not None and not b]
        best = worst = None
        if plain:
            hi = m.better == 'higher'
            best = max(plain, key=lambda kv: kv[1] if hi else -kv[1])[0]
            worst = min(plain, key=lambda kv: kv[1] if hi else -kv[1])[0]
            if values[best] == values[worst]:
                best = worst = None
        row = {'id': mid, 'title': m.title, 'unit': m.unit, 'better': m.better, 'study': m.study,
               'values': values, 'bounds': bounds, 'status': [r.get('status') for r in cells_v],
               'reasons': [r.get('reason') for r in cells_v], 'best': best, 'worst': worst,
               'stated_in': sum(v is not None for v in values), 'shared': sum(v is not None for v in values) >= 2,
               'spread': (max(v for _, v in plain) - min(v for _, v in plain)) if plain else None,
               'differences': []}
        for k in range(len(values)):
            if k == ref:
                continue
            row['differences'].append(_difference(values, bounds, spread, mid, ref, k, draws))
        if any(s for s in spread):
            row['tolerance'] = [None if not s or mid not in s else
                                {k: v for k, v in s[mid].items() if k != 'values'} for s in spread]
        rows.append(row)
    stats_used = _stats_tests(rows, spread, ref, [c.name for c in cells]) if draws and draws >= 2 else None
    assessed = [d for r in rows for d in r['differences'] if d['meaningful'] is not None]
    return {'rows': rows, 'ids': ids, 'reference': ref,
            'tolerance': {'draws': int(draws or 0), 'seed': int(seed), 'k': MEANINGFUL_K,
                          'rule': f'|Δ| > {MEANINGFUL_K:g} √(σ_ref² + σ_cell²) of the tolerance Monte Carlo '
                                  '(a default)' if draws else 'not assessed: no tolerance draws (set them to judge '
                                                              'whether a difference is more than the build spread)',
                          'errors': [s.get('_error') for s in spread if isinstance(s, dict) and s.get('_error')],
                          'stats': stats_used},
            'meaningful': sum(bool(d['meaningful']) for d in assessed), 'assessed': len(assessed)}


def _stats_tests(rows, spread, ref, names) -> dict | None:
    """With core/stats present: each difference's two-sample test over the cells' tolerance draws
    (stats.compare_samples: the permutation test of the mean difference, Hedges' g), the p-values Holm-adjusted across
    the table; a difference beyond the build spread is then meaningful only when it is also significant. None when
    core/stats is not installed (the two-sigma rule alone)."""
    try:
        from . import stats as ST
    except ImportError:
        return None
    tests = []
    for r in rows:
        for d in r['differences']:
            if d['meaningful'] is None:
                continue
            k = d['cell']
            xs = ((spread[ref] or {}).get(r['id']) or {}).get('values') or []
            ys = ((spread[k] or {}).get(r['id']) or {}).get('values') or []
            x = [v for v in xs if v is not None]
            y = [v for v in ys if v is not None]
            try:
                c = ST.compare_samples(x, y, r['better'], paired=False, n_boot=STATS_RESAMPLES,
                                       n_perm=STATS_RESAMPLES, names=(names[ref], names[k]))
            except Exception as e:                      # noqa: BLE001 — no test (constant or too few values): said
                d['test'] = {'p': None, 'why': str(e).split('\n')[0]}
                continue
            d['test'] = {'name': c['test_label'], 'p': c['p'], 'hedges_g': (c.get('standardised') or {}).get('g')}
            tests.append(d)
    if not tests:
        return {'used': True, 'tests': 0, 'correction': ST.CORRECTION}
    adj = ST.adjust([d['test']['p'] for d in tests], ST.CORRECTION, ST.ALPHA)
    for d, pa, rej in zip(tests, adj['adjusted'], adj['reject']):
        d['test'].update(p_adjusted=pa, significant=rej)
        if d['meaningful'] and not rej:
            d['meaningful'] = False
            d['why'] += f'; not significant over the draws (Holm-adjusted p {pa:.3g})'
    return {'used': True, 'tests': len(tests), 'correction': adj['label'], 'alpha': adj['alpha'],
            'rule': 'core/stats: permutation test of the mean over the draws, Holm-adjusted across the table'}


def _difference(values, bounds, spread, mid, ref, k, draws) -> dict:
    a, b = values[ref], values[k]
    out = {'cell': k, 'delta': None, 'sigma': None, 'z': None, 'meaningful': None, 'why': ''}
    if a is None or b is None:
        out['why'] = 'a figure is not stated'
        return out
    out['delta'] = b - a
    if bounds[ref] or bounds[k]:
        out['why'] = 'a figure is stated only as a bound'
        return out
    if not draws:
        out['why'] = 'no tolerance draws'
        return out
    sa, sb = (spread[ref] or {}).get(mid) or {}, (spread[k] or {}).get(mid) or {}
    if sa.get('std') is None or sb.get('std') is None:
        out['why'] = 'the tolerance draws state no spread for it'
        return out
    sd = math.sqrt(float(sa['std']) ** 2 + float(sb['std']) ** 2)
    out['sigma'] = sd
    if sd == 0:
        out['meaningful'] = out['delta'] != 0
        out['why'] = 'the tolerances do not move this figure' + ('' if out['delta'] else ', and it is the same')
        return out
    out['z'] = abs(out['delta']) / sd
    out['meaningful'] = out['z'] > MEANINGFUL_K
    out['why'] = (f'|Δ| is {out["z"]:.1f} σ of the build spread' + (' (more than the tolerances explain)'
                                                                   if out['meaningful'] else ' (within it)'))
    return out


# ====================================================================== drift of one design across built copies
def log_so3(R) -> np.ndarray:
    """The rotation vector (rad) of a rotation matrix, exact at small angles and up to π."""
    R = np.asarray(R, float)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 0.5
    s = float(np.linalg.norm(w))
    c = 0.5 * (np.trace(R) - 1.0)
    th = math.atan2(s, c)
    if s < 1e-12:
        if c > 0:
            return w                                   # th ≈ 0: w ≈ th · axis
        B = 0.5 * (R + np.eye(3))                      # th = π: the axis from R + I
        ax = B[:, int(np.argmax(np.diag(B)))]
        return ax / np.linalg.norm(ax) * math.pi
    return w / s * th


def exp_so3(w) -> np.ndarray:
    from .tolerance import exp_so3 as E
    return E(np.asarray(w, float))


def offset(X_plan, X_meas) -> np.ndarray:
    """[Δx, Δy, Δz (mm), ωx, ωy, ωz (degrees)]: measured minus planned in the mount frame; ω the rotation vector of
    R_meas · R_planᵀ (about the mount frame's axes)."""
    Xp, Xm = np.asarray(X_plan, float), np.asarray(X_meas, float)
    d = (Xm[:3, 3] - Xp[:3, 3]) * 1000.0
    w = np.degrees(log_so3(Xm[:3, :3] @ Xp[:3, :3].T))
    return np.r_[d, w]


def share_of_tolerance(d, tol) -> float:
    d = np.asarray(d, float)
    return float(max(np.max(np.abs(d[:3])) / tol[0], np.linalg.norm(d[3:]) / tol[1]))


def verdict_of(ratio) -> str:
    return 'ok' if ratio <= 1.0 + 1e-9 else 'adjust' if ratio <= REMOUNT + 1e-9 else 're-mount'


def robust(X) -> tuple[np.ndarray, np.ndarray]:
    """Median and robust standard deviation (1.4826 × median absolute deviation) of each column (Vantage)."""
    X = np.asarray(X, float)
    med = np.median(X, axis=0)
    return med, 1.4826 * np.median(np.abs(X - med), axis=0)


def _sessions(project, sensor) -> list:
    """The solved hand-eye sessions of this sensor, oldest first: [(date, session, camera entry)]."""
    cal = getattr(project, 'calibration', None) or {}
    out = []
    for s in cal.get('sessions') or []:
        if not isinstance(s, dict) or s.get('sensor_uid') != sensor.uid:
            continue
        r = s.get('result')
        if not isinstance(r, dict) or not r.get('ok'):
            continue
        cams = r.get('cameras') or {}
        e = cams.get(sensor.name) if sensor.name in cams else (next(iter(cams.values())) if len(cams) == 1 else None)
        if not isinstance(e, dict) or e.get('matrix') is None:
            continue
        X = np.asarray(e['matrix'], float)
        if X.shape != (4, 4) or not np.isfinite(X).all():
            continue
        out.append((str(r.get('applied_at') or s.get('created') or ''), s, e))
    out.sort(key=lambda x: x[0])
    return out


def _mount_X(project, sensor):
    """(robot uid, link, X): the imager's optical frame in its mount frame (core/calib/records.camera_mount)."""
    from .calib import records as RC
    im = RC.sensor_imager(sensor)
    return RC.camera_mount(project, sensor, im), im


def drift(cells: list, ref: int = 0) -> dict:
    """Drift of the reference cell's design across the other cells (module docstring, part 4)."""
    plan = cells[ref].project
    rows, consistency, notes = [], [], []
    for ps in plan.variant.sensors:
        if not ps.enabled or ps.kind not in ('camera', 'depth'):
            continue
        try:
            (pr, pl, Xp), _ = _mount_X(plan, ps)
        except Exception as e:                      # noqa: BLE001 — a camera that cannot be read: said, left out
            notes.append(f'{ps.name} is left out of the drift: {e}.')
            continue
        tol = (float(ps.mount.tolerance.position_mm), float(ps.mount.tolerance.rotation_deg))
        tol_default = tol == DEFAULT_TOL
        entries = []
        for k, c in enumerate(cells):
            cs = next((s for s in c.project.variant.sensors if s.name.strip().lower() == ps.name.strip().lower()
                       and s.kind in ('camera', 'depth')), None)
            if cs is None:
                continue
            try:
                (cr, cl, Xd), _ = _mount_X(c.project, cs)
            except Exception as e:                  # noqa: BLE001
                notes.append(f'{c.name}: {cs.name} cannot be read ({e}).')
                continue
            hist = _sessions(c.project, cs)
            if (cl or '') != (pl or ''):
                entries.append({'cell': k, 'name': c.name, 'sensor': cs.name, 'kind': 'not comparable',
                                'why': f'mounted on {cl or "a static frame"}, the plan on {pl or "a static frame"}'})
                continue
            if hist:
                date, sess, e = hist[-1]
                X, kind = np.asarray(e['matrix'], float), 'measured'
                src = f'hand-eye calibration "{sess.get("name", "")}"' + (f' ({date})' if date else '')
                sig = e.get('sigma')
            elif k == ref:
                continue                             # the plan against itself, unmeasured: nothing to say
            else:
                X, kind, src, sig = Xd, 'as designed', 'the cell\'s own mount (not measured)', None
            d = offset(Xp, X)
            ratio = share_of_tolerance(d, tol)
            within_sigma = None
            if sig:
                ps_mm = max(float(sig.get(a, 0) or 0) for a in ('x_mm', 'y_mm', 'z_mm'))
                ps_deg = max(float(sig.get(a, 0) or 0) for a in ('rx_deg', 'ry_deg', 'rz_deg'))
                within_sigma = bool(np.max(np.abs(d[:3])) <= 2 * ps_mm and np.linalg.norm(d[3:]) <= 2 * ps_deg)
            entries.append({'cell': k, 'name': c.name, 'sensor': cs.name, 'kind': kind, 'source': src,
                            'offset': [float(v) for v in d], 'offset_mm': float(np.linalg.norm(d[:3])),
                            'rotation_deg': float(np.linalg.norm(d[3:])), 'ratio': ratio,
                            'verdict': verdict_of(ratio), 'sigma': sig, 'within_2sigma': within_sigma,
                            'history': [{'date': h[0], 'session': h[1].get('name', ''),
                                         'offset': [float(v) for v in offset(Xp, np.asarray(h[2]['matrix'], float))]}
                                        for h in hist]})
        rows.append({'sensor': ps.name, 'uid': ps.uid, 'mount': pl or 'world', 'tolerance_mm': tol[0],
                     'tolerance_deg': tol[1], 'tolerance_default': tol_default, 'copies': entries})
        meas = [e for e in entries if e.get('kind') == 'measured']
        consistency.append(_consistency(ps, meas, tol))
    cells_measured = sorted({e['cell'] for r in rows for e in r['copies'] if e.get('kind') == 'measured'})
    worst = max((e for r in rows for e in r['copies'] if e.get('kind') == 'measured'), key=lambda e: e['ratio'],
                default=None)
    return {'plan': cells[ref].name, 'reference': ref, 'cameras': rows, 'consistency': consistency,
            'measured_cells': cells_measured, 'worst': None if worst is None else
            {'cell': worst['cell'], 'name': worst['name'], 'sensor': worst['sensor'], 'ratio': worst['ratio'],
             'verdict': worst['verdict']},
            'systematic': [c['sensor'] for c in consistency if c['systematic']], 'notes': notes,
            'frame': 'offsets in each camera\'s mount frame (its link, or the world for a static camera): Δx Δy Δz '
                     'in mm, then the rotation vector of R_measured · R_planᵀ in degrees'}


def _consistency(ps, meas: list, tol) -> dict:
    out = {'sensor': ps.name, 'uid': ps.uid, 'n': len(meas), 'median': None, 'sigma': None, 'median_ratio': None,
           'near_median': 0, 'systematic': False, 'outliers': [], 'z': {}}
    if len(meas) < 2:
        return out
    X = np.array([e['offset'] for e in meas])
    med, sig = robust(X)
    floor = np.array([tol[0]] * 3 + [tol[1]] * 3) * SIGMA_FLOOR
    spread = np.maximum(sig, floor)
    mr = share_of_tolerance(med, tol)
    near = sum(share_of_tolerance(np.asarray(e['offset']) - med, tol) <= 1.0 for e in meas)
    out.update(median=med.tolist(), sigma=sig.tolist(), median_ratio=mr, near_median=int(near),
               systematic=bool(mr > 1.0 and near > len(meas) / 2))
    if len(meas) >= MIN_FLEET:
        Z = (X - med) / spread
        names = ('x', 'y', 'z', 'rx', 'ry', 'rz')
        for e, z in zip(meas, Z):
            j = int(np.argmax(np.abs(z)))
            out['z'][e['name']] = float(abs(z[j]))
            if abs(z[j]) > OUTLIER_Z:
                out['outliers'].append({'cell': e['cell'], 'name': e['name'], 'component': names[j],
                                        'from_median': float(X[meas.index(e), j] - med[j]), 'z': float(z[j])})
    out['outliers'].sort(key=lambda o: -abs(o['z']))
    return out


def as_built_poses(plan, dr: dict) -> dict:
    """{sensor uid: mount pose dict} that puts each systematically offset camera of the plan at the copies' median."""
    from .frames import Pose
    out = {}
    for c in dr['consistency']:
        if not c['systematic']:
            continue
        s = plan.variant.sensor(c['uid'])
        if s is None:
            continue
        (robot, link, Xp), im = _mount_X(plan, s)
        med = np.asarray(c['median'], float)
        X = np.eye(4)
        X[:3, :3] = exp_so3(np.radians(med[3:])) @ Xp[:3, :3]
        X[:3, 3] = Xp[:3, 3] + med[:3] / 1000.0
        out[s.uid] = Pose.from_matrix(_mount_from_X(plan, s, im, X, robot, link)).to_dict()
    return out


def _mount_from_X(project, sensor, imager, X, robot, link) -> np.ndarray:
    """The sensor body pose in its mount's parent frame that puts the imager's optical frame at X (X in the frame
    core/calib/records.camera_mount uses: the moving link, else the world)."""
    from .robot.model import get_model
    T_bo = imager.pose_in_body.matrix()
    body = np.asarray(X, float) @ np.linalg.inv(T_bo)
    if robot or not sensor.mount.on_link:
        return body                                 # in the link (moving) or the world (static frames)
    m = get_model(project, sensor.mount.robot_uid)   # a link no joint moves: X was in the world
    return np.linalg.inv(m.link_pose(m.home, sensor.mount.parent_frame)) @ body


# ====================================================================== the suggestion
def _match_suggestion(cell0, recs0, gp: dict, ref_recs: list, ref_name: str) -> dict | None:
    """Mount the open cell's least-alike matched camera at the reference camera's pose in the role's frame."""
    from . import plucker as PK
    from .frames import Pose
    from .robot.model import get_model
    worst = max((m for m in gp['matches'] if m['cost'] > 1e-9), key=lambda m: m['cost'], default=None)
    if worst is None:
        return None
    a = next(r for r in ref_recs if r.name == worst['a'])
    b = next(r for r in recs0 if r.name == worst['b'])
    p = cell0.project
    s = p.variant.sensor(b.uid)
    im = PK.pick_imager(s, PK.sensor_stream(s))
    T_world_opt = b.T_world_role @ (FLIP @ a.T if worst.get('closing_flipped') else a.T)
    if s.mount.on_link:
        m = get_model(p, s.mount.robot_uid)
        T_wp = m.link_pose(m.home, s.mount.parent_frame)
    else:
        T_wp = np.eye(4)
    body = np.linalg.inv(T_wp) @ T_world_opt @ np.linalg.inv(im.pose_in_body.matrix())
    cur = s.mount.pose.matrix()
    move = float(np.linalg.norm(body[:3, 3] - cur[:3, 3]))
    turn = rotation_deg(cur[:3, :3], body[:3, :3])
    return {'kind': 'match_mount', 'sensor_uid': s.uid, 'sensor': s.name, 'locked': bool(s.locked),
            'reference_camera': a.name, 'reference': ref_name, 'role': a.role,
            'pose': Pose.from_matrix(body).to_dict(), 'move_m': move, 'turn_deg': turn,
            'T_world_optical': T_world_opt.tolist(),
            'text': (f'Mount {s.name} where {ref_name}\'s {a.name} sits'
                     + (' relative to its fingertips' if a.role == 'wrist' else ' in the world')
                     + f': {move * 1000:.0f} mm and {turn:.1f}° from its mount now; that pair\'s viewpoint gap is then '
                       'zero (the lenses may still differ).')}


# ====================================================================== the comparison
def compare(cells: list, reference: int = 0, metrics=None, tolerance_draws: int = 0, seed: int = 1,
            pooling: str = 'auto', ctx: Ctx | None = None, evaluator=None) -> dict:
    """The comparison document of a cell set (module docstring). cells: [Cell] (make_cells)."""
    from . import cells_pool as CP
    ctx = ctx or Ctx()
    if len(cells) < 2:
        raise StudyError('A cell set needs at least two cells: add another project (File ▸ Compare cells…), a '
                         'sample, or a variant of this project.')
    if pooling not in ('auto', 'plan', 'recorded'):
        raise StudyError('Pooling compares "auto" (a recorded dataset where the cell has one), "plan" or '
                         '"recorded".')
    try:
        ref = int(reference)
    except (TypeError, ValueError):
        raise StudyError('The reference cell is given by its number in the set (0 is the first).') from None
    if not 0 <= ref < len(cells):
        raise StudyError(f'The reference cell {ref} is not in the set (it has {len(cells)} cells, 0 to '
                         f'{len(cells) - 1}).')
    ids = metric_ids(metrics)
    names = [c.name for c in cells]
    notes = []
    ctx.report(0.02, 'Cameras')
    recs = []
    for c in cells:
        r, n = cameras(c.project)
        recs.append(r)
        notes += [f'{c.name}: {x}' for x in n]
    ctx.report(0.08, 'The geometric gap')
    gaps = [gap(recs[ref], recs[k], names[ref], names[k]) for k in range(len(cells))]
    matrix = [[gaps[j]['total'] if i == ref else (gaps[i]['total'] if j == ref else None)
               for j in range(len(cells))] for i in range(len(cells))]
    for i in range(len(cells)):
        for j in range(len(cells)):
            if matrix[i][j] is None:
                matrix[i][j] = 0.0 if i == j else gap(recs[i], recs[j], names[i], names[j])['total']
    ctx.report(0.12, 'Pooling')
    schemas = []
    for c in cells:
        try:
            schemas.append(CP.cell_schema(c.project, pooling))
        except StudyError as e:
            raise StudyError(f'{c.name}: {e}') from None
    pool = CP.pooling(schemas, names, ref)
    ctx.report(0.18, 'Drift')
    dr = drift(cells, ref)
    sub = Ctx(progress=lambda f, t='': ctx.report(0.2 + 0.78 * f, t), cancel=ctx.cancel)
    mt = metrics_table(cells, ids, ref, tolerance_draws, seed, sub, evaluator)
    others = [g for k, g in enumerate(gaps) if k != ref]
    worst = max(others, key=lambda g: g['total'])
    sugg = None
    if ref != 0:
        try:
            sugg = _match_suggestion(cells[0], recs[0], gaps[0], recs[ref], names[ref])
        except Exception as e:                       # noqa: BLE001 — no suggestion, said why
            notes.append(f'No suggestion: the camera could not be placed ({e}).')
    elif dr['systematic']:
        poses = as_built_poses(cells[0].project, dr)
        if poses:
            k = len(dr['measured_cells'])
            sugg = {'kind': 'as_built_variant', 'name': f'{cells[0].project.variant.name} as built (median of {k} '
                                                         f'cell{"" if k == 1 else "s"})', 'poses': poses,
                    'sensors': [cells[0].project.variant.sensor(u).name for u in poses],
                    'text': ('The copies are built consistently off the plan for '
                             + ', '.join(dr['systematic']) + ': add a variant with those cameras at the copies\' '
                             'median, so the plan follows the build (the plan itself is kept).')}
    ph = pool['counts']
    head = {'value': worst['total'], 'unit': '', 'status': worst['status'], 'word': f'{worst["band"]} gap',
            'cell': worst['b'],
            'text': f'Largest geometric gap to {names[ref]}: {worst["total"]:.3f} ({worst["b"]}). {pool["text"]}',
            'advice': (sugg or {}).get('text') or (
                'Choose another cell as the reference to get a placement suggestion for the open project.'
                if ref == 0 and worst['total'] > 0 else 'The cells are geometrically alike.'),
            'pooling': {'not_poolable': ph[CP.VERDICTS[2]], 'after_mapping': ph[CP.VERDICTS[1]],
                        'as_is': ph[CP.VERDICTS[0]]}}
    ctx.report(1.0, 'done')
    return {'schema': SCHEMA, 'cells': [dict(c.summary(), index=k, reference=k == ref, cameras_detail=[
                r.row() for r in recs[k]]) for k, c in enumerate(cells)],
            'reference': ref, 'headline': head,
            'gap': {'pairs': [g for k, g in enumerate(gaps) if k != ref], 'matrix': matrix, 'weights': dict(WEIGHTS),
                    'band_note': BAND_NOTE, 'scales': {'angle_deg': ANGLE_SCALE_DEG,
                                                       'position_m': POSITION_SCALE_M, 'fov_deg': FOV_SCALE_DEG,
                                                       'resolution_octaves': RES_OCTAVES},
                    'defaults': 'the weights, scales and bands are DexVantage\'s defaults: the bands and the angle, '
                                'FOV and resolution scales are Vantage\'s (core/rig_gap.py), the components and '
                                'their weights DexVantage\'s own; the position scale is 0.5 m for a robot cell'},
            'pooling': pool, 'drift': dr, 'metrics': mt, 'suggestion': sugg, 'notes': notes,
            'settings_used': {'reference': ref, 'metrics': ids, 'tolerance_draws': int(tolerance_draws or 0),
                              'seed': int(seed), 'pooling': pooling}}


def settings_of(project) -> dict:
    return dict((project.analysis.studies.get(SID) or {}) if project is not None else {})


def compare_project(project, ctx: Ctx | None = None, extra=None) -> dict:
    """The comparison the project's own cell set describes (project.analysis.studies['cells']: 'cells' [entries
    for make_cells], 'variants' (default True), 'reference', 'metrics', 'tolerance_draws', 'seed', 'pooling'), the open
    project being the first cell; extra: more cells given here (batch, API)."""
    st = settings_of(project)
    items = list(st.get('cells') or []) + list(extra or [])
    cells = make_cells(items, open_project=project, variants=bool(st.get('variants', True)))
    return compare(cells, st.get('reference', 0), st.get('metrics'), int(st.get('tolerance_draws') or 0),
                   int(st.get('seed', 1) or 1), str(st.get('pooling') or 'auto'), ctx)


# ====================================================================== outputs
def pooling_mapping(doc: dict) -> dict:
    """The pooling-mapping document (schema dexvantage.cells_pooling/1): the target and, per cell, its verdict, the
    steps that map its recording onto the target's features and what cannot be mapped."""
    p = doc['pooling']
    return {'schema': p['schema'], 'target': p['target'], 'poolable': p['poolable'], 'counts': p['counts'],
            'text': p['text'],
            'cells': [{k: c[k] for k in ('index', 'name', 'target', 'source', 'what', 'verdict', 'operations',
                                         'unmappable', 'warnings', 'camera_map', 'notes')} for c in p['cells']],
            'format': 'LeRobot v2.1 / v3.0 feature names (meta/info.json); an operation maps the cell\'s feature '
                      'onto the target\'s'}


STATUS_HEX = {'good': '#0ca30c', 'warning': '#c98500', 'serious': '#d4572a', 'critical': '#d03b3b'}


def _fmt(v, unit='', nd=3) -> str:
    if v is None:
        return '—'
    if isinstance(v, float):
        a = abs(v)
        s = f'{v:.{nd}g}' if a < 1e5 else f'{v:.3e}'
    else:
        s = str(v)
    return f'{s} {unit}'.strip()


def to_html(doc: dict, provenance: dict | None = None) -> str:
    """A standalone HTML page of the comparison (no scripts, no external files)."""
    e = html.escape
    cells = doc['cells']
    names = [c['name'] for c in cells]
    ref = doc['reference']
    h = doc['headline']
    css = ('body{font:14px/1.45 "IBM Plex Sans",system-ui,sans-serif;margin:24px;color:#1d2024;background:#fff}'
           'table{border-collapse:collapse;margin:10px 0 18px}th,td{border-bottom:1px solid #e3e5e8;padding:4px 8px;'
           'text-align:left;vertical-align:top}th{font-weight:600;color:#555}.m{font-family:"IBM Plex Mono",'
           'ui-monospace,monospace;white-space:nowrap}.best{color:#0a7a0a}.worst{color:#b42318}.muted{color:#777}'
           '.st{font-weight:600}h2{margin-top:28px}'
           '@media (prefers-color-scheme:dark){body{background:#16181b;color:#e6e8eb}th,td{border-color:#2c3036}'
           'th{color:#a0a6ad}.muted{color:#999}}')
    out = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
           f'content="width=device-width, initial-scale=1"><title>Cell comparison</title><style>{css}</style></head>'
           f'<body><h1>Cell comparison</h1><p>{len(cells)} cells; reference: {e(names[ref])}.</p>'
           f'<p class="st" style="color:{STATUS_HEX.get(h["status"], "#555")}">{e(h["word"])}: '
           f'<span class="m">{h["value"]:.3f}</span></p><p>{e(h["text"])}</p><p class="muted">{e(h["advice"])}</p>']
    out.append('<h2>Cells</h2><table><thead><tr><th>#</th><th>Cell</th><th>Source</th><th>Robots</th>'
               '<th>Cameras</th></tr></thead><tbody>')
    for c in cells:
        out.append(f'<tr><td class="m">{c["index"]}</td><td>{e(c["name"])}{" (reference)" if c["reference"] else ""}'
                   f'</td><td>{e(c["source"])}</td><td>{e(", ".join(c["robots"]))}</td>'
                   f'<td class="m">{c["cameras"]}</td></tr>')
    out.append('</tbody></table>')
    mt = doc['metrics']
    out.append(f'<h2>Figures of merit</h2><p class="muted">{e(mt["tolerance"]["rule"])}</p><table><thead><tr>'
               '<th>Metric</th>' + ''.join(f'<th>{e(n)}</th>' for n in names) + '<th>Differences from the reference'
               '</th></tr></thead><tbody>')
    for r in mt['rows']:
        tds = []
        for k, v in enumerate(r['values']):
            cls = 'best' if k == r['best'] else 'worst' if k == r['worst'] else ''
            b = {'at_least': '≥ ', 'at_most': '≤ '}.get(r['bounds'][k] or '', '')
            tip = e(r['reasons'][k] or '')
            tds.append(f'<td class="m {cls}" title="{tip}">{e(b + _fmt(v, r["unit"]))}</td>')
        diffs = '; '.join(f'{e(names[d["cell"]])}: {e(_fmt(d["delta"], delta_unit(r["unit"])))}'
                          + (' meaningful' if d['meaningful'] else ' within the spread' if d['meaningful'] is False
                             else '') for d in r['differences'] if d['delta'] is not None)
        out.append(f'<tr><td>{e(r["title"])}</td>{"".join(tds)}<td>{diffs or "—"}</td></tr>')
    out.append('</tbody></table><p class="muted">Green: best; red: worst (where the cells differ). — : not stated '
               '(hover for the reason).</p>')
    out.append(f'<h2>Geometric gap</h2><p class="muted">{e(doc["gap"]["band_note"])}; {e(doc["gap"]["defaults"])}.'
               '</p>')
    for g in doc['gap']['pairs']:
        out.append(f'<h3>{e(g["a"])} → {e(g["b"])}: <span class="m">{g["total"]:.3f}</span> ({e(g["band"])})</h3>'
                   '<table><thead><tr>' + ''.join(f'<th>{e(k)}</th>' for k in g['components']) + '</tr></thead>'
                   '<tbody><tr>' + ''.join(f'<td class="m">{v:.3f}</td>' for v in g['components'].values())
                   + '</tr></tbody></table>')
        if g['matches']:
            out.append('<table><thead><tr><th>Camera</th><th>Matched</th><th>Role</th><th>Viewpoint</th>'
                       '<th>Rotation</th><th>Distance</th><th>HFOV Δ</th><th>Resolution ratio</th>'
                       '<th>Task volume in image</th></tr></thead><tbody>')
            for m in g['matches']:
                out.append(f'<tr><td>{e(m["a"])}</td><td>{e(m["b"])}</td><td>{m["role"]}</td>'
                           f'<td class="m">{m["viewpoint_deg"]:.2f}°</td><td class="m">{m["rotation_deg"]:.2f}°</td>'
                           f'<td class="m">{m["distance_m"] * 1000:.1f} mm</td>'
                           f'<td class="m">{m["hfov_diff_deg"]:+.1f}°</td>'
                           f'<td class="m">{_fmt(m["resolution_ratio"])}</td>'
                           f'<td class="m">{_fmt(m["image_pct_a"], "%")} → {_fmt(m["image_pct_b"], "%")}</td></tr>')
            out.append('</tbody></table>')
        out.append('<ul>' + ''.join(f'<li>{e(x)}</li>' for x in g['findings']) + '</ul>')
    p = doc['pooling']
    out.append(f'<h2>Dataset pooling</h2><p>{e(p["text"])} Target: {e(p["target"]["name"])} '
               f'({e(p["target"]["what"])}).</p>')
    for c in p['cells']:
        if c['target']:
            continue
        out.append(f'<h3>{e(c["name"])}: {e(c["verdict"])}</h3><p class="muted">{e(c["what"])}</p>')
        if c['operations']:
            out.append('<table><thead><tr><th>Step</th><th>Feature</th><th>Detail</th><th>Why</th></tr></thead><tbody>')
            for o in c['operations']:
                det = {k: v for k, v in o.items() if k not in ('op', 'feature', 'why')}
                out.append(f'<tr><td>{e(o["op"])}</td><td class="m">{e(str(o.get("feature", "")))}</td>'
                           f'<td class="m">{e(json.dumps(det, ensure_ascii=False))}</td><td>{e(o["why"])}</td></tr>')
            out.append('</tbody></table>')
        if c['unmappable']:
            out.append('<p>Cannot be mapped:</p><ul>' + ''.join(f'<li><span class="m">{e(u["what"])}</span>: '
                                                                f'{e(u["why"])}</li>' for u in c['unmappable'])
                       + '</ul>')
        if c['warnings']:
            out.append('<ul class="muted">' + ''.join(f'<li>{e(w)}</li>' for w in c['warnings']) + '</ul>')
    dr = doc['drift']
    out.append(f'<h2>Drift of the design across the cells</h2><p class="muted">Plan: {e(dr["plan"])}. '
               f'{e(dr["frame"])}.</p>')
    rows = [(r, x) for r in dr['cameras'] for x in r['copies']]
    if rows:
        out.append('<table><thead><tr><th>Camera</th><th>Cell</th><th>Kind</th><th>Offset</th><th>Rotation</th>'
                   '<th>Share of tolerance</th><th>Verdict</th><th>Source</th></tr></thead><tbody>')
        for r, x in rows:
            if x.get('kind') == 'not comparable':
                out.append(f'<tr><td>{e(r["sensor"])}</td><td>{e(x["name"])}</td><td>not comparable</td>'
                           f'<td colspan="5">{e(x["why"])}</td></tr>')
                continue
            col = STATUS_HEX[VERDICT_SEV[x['verdict']]]
            out.append(f'<tr><td>{e(r["sensor"])}</td><td>{e(x["name"])}</td><td>{e(x["kind"])}</td>'
                       f'<td class="m">{x["offset_mm"]:.2f} mm</td><td class="m">{x["rotation_deg"]:.3f}°</td>'
                       f'<td class="m">{x["ratio"]:.2f}</td><td class="st" style="color:{col}">{e(x["verdict"])}</td>'
                       f'<td>{e(x["source"])}</td></tr>')
        out.append('</tbody></table>')
    else:
        out.append('<p>No other cell shares a camera name with the plan.</p>')
    for c in dr['consistency']:
        if c['systematic']:
            out.append(f'<p>{e(c["sensor"])}: the copies are built consistently off the plan (median at '
                       f'{c["median_ratio"]:.2f} of the tolerance, {c["near_median"]} of {c["n"]} near it).</p>')
        for o in c['outliers']:
            out.append(f'<p>{e(o["name"])}: {e(c["sensor"])} stands out from the other copies ({o["component"]} '
                       f'{o["from_median"]:+.2f} from their median, robust z {o["z"]:+.1f}).</p>')
    if doc.get('notes'):
        out.append('<h2>Notes</h2><ul class="muted">' + ''.join(f'<li>{e(n)}</li>' for n in doc['notes']) + '</ul>')
    if provenance:
        out.append('<h2>Provenance</h2><details><summary>How this was made</summary><pre class="m">'
                   + e(json.dumps(provenance, indent=1, ensure_ascii=False)) + '</pre></details>')
    return ''.join(out) + '</body></html>\n'


def write_files(doc: dict, project, out_dir: str, stem: str = 'cells') -> list:
    """<stem>.json (the comparison), <stem>_pooling.json (the pooling mapping) and <stem>_comparison.html in out_dir,
    the JSON files stamped with provenance; returns the file names."""
    from . import provenance as PV
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        raise StudyError(f'Could not create the folder {os.path.basename(out_dir) or out_dir}: {e.strerror or e}.') \
            from None
    clean = {k: v for k, v in doc.items() if k not in ('cached', 'revision', PV.KEY)}
    files = {f'{stem}.json': clean, f'{stem}_pooling.json': pooling_mapping(clean)}
    names = []
    try:
        for rel, d in files.items():
            st = PV.stamp(d, project, output={'option': SID, 'name': rel, 'in_folder': True})
            with open(os.path.join(out_dir, rel), 'w', encoding='utf-8') as fh:
                json.dump(st, fh, indent=1, ensure_ascii=False, allow_nan=False)
                fh.write('\n')
            names.append(rel)
        rel = f'{stem}_comparison.html'
        blk = PV.block(project, clean, output={'option': SID, 'name': rel, 'in_folder': True})
        with open(os.path.join(out_dir, rel), 'w', encoding='utf-8') as fh:
            fh.write(to_html(clean, blk))
        names.append(rel)
    except OSError as e:
        raise StudyError(f'Could not write into {os.path.basename(out_dir) or out_dir}: {e.strerror or e}.') from None
    return names


def jsonable(doc):
    """A deep copy with numpy scalars made plain (results are JSON)."""
    return json.loads(json.dumps(doc, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))


