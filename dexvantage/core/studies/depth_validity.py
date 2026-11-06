"""Depth validity (spec §7.2): where in the task volumes each depth sensor gives a depth reading, why it does not
elsewhere, and how noisy the reading is expected to be.

A voxel (core/voxels.py, its centre) is **depth-valid** for a depth sensor when all five conditions hold, tested in
this order (the first that fails is the voxel's reason, so the reasons add up to the invalid volume):

1. ``fov``: it is inside the field of view of every imager the technology needs: both imagers of a stereo pair
   (active or passive stereo: their overlap, which is the depth field of view of Intel's D400 datasheet §4.4,
   "Depth FOV = HFOV/2 + atan(tan(HFOV/2) − B/Z)"); the reference imager (the camera that sees the pattern) for
   structured light; the time-of-flight imager for ToF. The spec's wording for structured light, "the RGB and
   projector FOVs", is read as the pattern camera and the projector: the colour camera does not make the depth.
   For structured light the projector cone is part of this condition.
2. ``min_z`` / ``max_range``: its depth z (along the reference imager's optical axis, the D4xx depth convention) is
   at least the stated minimum Z for the chosen depth resolution (the entry's min_z table; a device without a
   per-resolution table uses its stated minimum range, labelled so) and at most the stated maximum range (per mode
   when the maker states one per mode). A figure the maker does not state is not tested, and the result says so.
3. ``occluded``: nothing in the scene (the robots posed at q, the task objects, the surfaces: core/scenegeom.py)
   hides it from **either** imager: one cube map (core/occlusion.py) per imager centre, so finger shadows and the
   parallax occlusion of a stereo pair (seen by one imager, hidden from the other) both count. The result splits
   the occluded volume into "hidden from one imager only" and "hidden from both".
4. ``projector``: it is inside the IR projector's cone, for active sensors that need its texture. The setting
   ``projector`` is 'auto' by default: on for active IR stereo when the device's projector and its FOV are stated
   (and always for structured light, where condition 1 already holds it), off for passive stereo and ToF (a ToF
   illuminator's field is not in the library). Set it to true or false per sensor.
5. ``material``: it is not on a task object tagged transparent, IR-absorbing or specular (TaskObject.materials;
   'glossy' and 'metallic' count as specular). A warning with its reason: the geometry is fine, the surface may
   still give no depth. IR-absorbing is a warning only for the active technologies (active IR stereo, structured
   light, ToF), whose depth comes from their own IR light. A voxel is "on" an object when it lies inside the
   object's shape grown by half the voxel's diagonal (a mesh: its bounding box so grown).

Volumes are voxel volumes (refined voxels are smaller), so every percentage is a percentage of the task volumes;
voxels whose centre lies inside a solid task object (box, cylinder or sphere) are left out of them, as in the
coverage study (no sensor measures a solid's inside; counted, they would all read "occluded").

**Expected depth noise.** For stereo (and structured light with a stated baseline), the first-order propagation of
a disparity error through z = f B / d gives

    σ_z = z² · σ_d / (f · B)

[Hartley and Zisserman 2004, §12; Gallup et al. 2008], with f the reference imager's focal length in pixels at the
chosen depth resolution (fx: disparity is horizontal), B the stated baseline and σ_d the matcher's subpixel disparity
error: the entry's stated subpixel_px, else the setting sigma_d_px, **0.08 px by default (a default, not a device
figure)**. The result gives the median and the 5th and 95th percentiles over the depth-valid voxels (volume
weighted) and compares the model with the maker's stated depth-error figure at the maker's distance: both are shown,
with the metric the maker uses (the entry's rms_error.metric: 'rms', 'median_abs', 'std_dev' or 'unspecified').
Only for an RMS figure is the σ_d that would reproduce it given: Intel's and Orbbec's stereo figures are bounds on a
plane-fit RMS over a stated share of the field of view at one resolution (Intel: "RMS error (or spatial noise) ≤ 2 %",
Table 4-15), not the same quantity as a per-point σ_z, and a median absolute error (Luxonis) or an undefined
"accuracy" (Stereolabs) is not an RMS at all, so no σ_d is matched to those. The comparison says how far apart they
are, not which is right. ToF has no stereo error model: the maker's figure is shown alone.

**Resolution.** Each sensor's depth resolution is a setting (studies['depth_validity']['sensors'][uid]['resolution'],
[w, h]); without it, the stream chosen for the sensor on the Sensors page (Sensor.stream) when that is a depth mode
of the device (the Inspector's min-Z and the Camera view use it), else the default: the entry's native depth stream,
its largest listed depth stream, else the reference imager's largest stream, else the largest row of its min-Z
table, else its native resolution. The result says which.

**Pose.** The study runs at the joint state given in its settings, q (robot uid -> joint values in
RobotModel.joint_names order), which the Depth page passes for the run from the 3D view's pose (view state, not
stored in the project); without q (batch mode, or pose = 'home') the robots are at their home pose. The result
records the pose it used.

Settings (project.analysis.studies['depth_validity']), all optional: sigma_d_px (0.08, default), cube_n (512),
backend ('auto'), q, pose ('home' ignores q), good_pct (80) and fair_pct (50) (the status words' thresholds,
defaults), sensors: {uid: {'resolution': [w, h], 'projector': 'auto' | true | false, 'sigma_d_px': x}}.

Known limits: the voxel centre stands for the voxel; occlusion is per cube-map texel (core/occlusion.py) from the
imagers' optical centres, and a sensor's own housing and other sensors' housings are not occluders; the projector's
own shadow is not modelled (only its cone), and the D4xx projector's position is a labelled assumption in the
library; the imagers at a depth stream take the stated depth FOV, which Intel states as the overlap at one distance
(0.2 m for the D405 and 2 m for the D415, D43x and D455 in the D400 datasheet, 4 m for the D457 in its own), so the
modelled overlap is narrower than the one the stated figure implies: by 2.9–3.6° for the D405 and 0.7–1.9° for the
others from min-Z outwards (Intel gives about ±3° unit to unit); min-Z is measured from the
optical centre, not the front glass (a few millimetres); materials are tags, not optics (no reflectance, no
multipath, no sunlight); σ_z is first order in σ_d / d and ignores calibration error, matcher bias and quantisation;
lens distortion is what the library states (pinhole unless calibrated).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .. import jobs
from ..frames import Pose
from ..jobs import StudyError

STUDY_ID = 'depth_validity'
DEFAULT_SIGMA_D_PX = 0.08        # default subpixel disparity error (px) when the maker states none: a DEFAULT
DEFAULT_CUBE_N = 512             # cube-map face size (texels)
DEFAULT_GOOD_PCT = 80.0          # status 'good' at or above this depth-valid share: a DEFAULT
DEFAULT_FAIR_PCT = 50.0          # status 'warning' at or above this, else 'critical': a DEFAULT
OVERLAY_PER_REASON = 3000        # invalid voxels kept per reason for the 3D overlay
CHUNK = 200_000

REASONS = ('fov', 'min_z', 'max_range', 'occluded', 'projector', 'material')
REASON_TITLE = {
    'valid': 'Depth-valid',
    'fov': 'Outside a field of view',
    'min_z': 'Closer than min-Z',
    'max_range': 'Beyond the maximum range',
    'occluded': 'Occluded for an imager',
    'projector': 'Outside the projector cone',
    'material': 'Material warning',
}
REASON_TEXT = {
    'fov': 'outside the field of view of an imager the depth needs (for stereo: not seen by both imagers)',
    'min_z': 'closer than the minimum Z at this resolution: the matcher cannot reach the disparity',
    'max_range': 'beyond the stated maximum range',
    'occluded': 'hidden from an imager by the robot, an object or a surface (finger shadows, stereo parallax)',
    'projector': 'outside the IR projector\'s cone, so it gets no projected texture',
    'material': 'on a surface tagged transparent, IR-absorbing or specular (a warning: see the reason)',
}
MATERIAL_REASON = {
    'transparent': 'transparent: the cameras and the IR pattern see through it or off it, so the depth is missing '
                   'or belongs to what is behind it',
    'ir_absorbing': 'IR-absorbing: it returns little of the device\'s own IR light (projector pattern or ToF '
                    'illumination), so the depth has holes',
    'specular': 'glossy or metallic: mirror reflections and highlights mislead stereo matching (and ToF by '
                'multipath)',
}
SPECULAR_TAGS = ('glossy', 'metallic', 'specular')
ACTIVE = ('active_ir_stereo', 'structured_light', 'tof')
STEREO = ('active_ir_stereo', 'passive_stereo')
TECH_TITLE = {'active_ir_stereo': 'active IR stereo', 'passive_stereo': 'passive stereo', 'tof': 'time of flight',
              'structured_light': 'structured light'}


class DepthSetupError(StudyError):
    """A sensor the study cannot analyse (no depth device, or its device lacks what the depth model needs)."""


# ====================================================================== the error model
def sigma_z(z, f_px: float, baseline_m: float, sigma_d_px: float):
    """Expected depth RMS error (m) at depth z (m): z² σ_d / (f B) (module docstring)."""
    z = np.asarray(z, float)
    return z * z * float(sigma_d_px) / (float(f_px) * float(baseline_m))


def sigma_d_equivalent(sigma_z_m: float, z_m: float, f_px: float, baseline_m: float) -> float:
    """The disparity error (px) for which the model gives sigma_z_m at z_m."""
    return float(sigma_z_m) * float(f_px) * float(baseline_m) / (float(z_m) ** 2)


def weighted_quantile(values, weights, qs) -> list:
    """Quantiles of values with weights (the voxels' volumes): the inverse of the weighted empirical CDF at the
    midpoints of each value's weight."""
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    if not len(v):
        return [None for _ in qs]
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return [float(np.interp(q, c, v)) for q in qs]


# ====================================================================== settings
def settings(project) -> dict:
    """The study's settings with the defaults filled in."""
    s = dict(project.analysis.studies.get(STUDY_ID) or {})
    out = {'sigma_d_px': DEFAULT_SIGMA_D_PX, 'cube_n': DEFAULT_CUBE_N, 'backend': 'auto', 'pose': None, 'q': {},
           'good_pct': DEFAULT_GOOD_PCT, 'fair_pct': DEFAULT_FAIR_PCT, 'sensors': {}}
    out.update({k: v for k, v in s.items() if v is not None})
    out['sigma_d_default'] = 'sigma_d_px' not in s
    return out


def sensor_settings(project, uid: str) -> dict:
    return dict((settings(project).get('sensors') or {}).get(uid) or {})


def device_title(entry) -> str:
    """The device's name, with its maker's name in front unless the name already carries the maker's last word
    ('RealSense Depth Camera D405' for maker 'Intel RealSense', not 'Intel RealSense RealSense ...')."""
    last = (entry.maker or '').split()[-1:] or ['']
    return entry.name if last[0].lower() in entry.name.lower() else entry.title


def _norm(s) -> str:
    return ' '.join(str(s or '').replace('_', ' ').replace('-', ' ').lower().split())


# ====================================================================== the sensor's depth set-up
def resolution_options(entry) -> list:
    """[(w, h, min_z m or None)] the depth modes of a library entry: its depth streams and its min-Z table's rows,
    largest first."""
    dp = entry.data.get('depth') or {}
    table = {(int(r['width']), int(r['height'])): r.get('m') for r in dp.get('min_z') or []}
    modes = {(int(s['width']), int(s['height'])) for s in entry.data.get('streams') or []
             if _norm(s.get('imager')) == 'depth'}
    modes |= set(table)
    if not modes:
        ref = entry.imager('depth') or {}
        if ref.get('resolution'):
            modes.add(tuple(int(v) for v in ref['resolution']))
    return sorted(((w, h, table.get((w, h))) for w, h in modes), key=lambda t: (-t[0] * t[1], -t[0]))


def native_depth_mode(entry) -> tuple:
    """((w, h), where it comes from): the entry's native depth stream (module docstring)."""
    ss = [s for s in entry.data.get('streams') or [] if _norm(s.get('imager')) == 'depth']
    if ss:
        s = max(ss, key=lambda s: (s['width'] * s['height'], s['width']))
        return (int(s['width']), int(s['height'])), 'the largest depth stream the maker lists'
    ref = entry.imager('depth') or {}
    ss = entry.streams(ref.get('name', '')) if ref else []
    if ss:
        s = max(ss, key=lambda s: (s['width'] * s['height'], s['width']))
        return (int(s['width']), int(s['height'])), 'the reference imager\'s largest listed stream'
    rows = (entry.data.get('depth') or {}).get('min_z') or []
    if rows:
        r = max(rows, key=lambda r: (r['width'] * r['height'], r['width']))
        return (int(r['width']), int(r['height'])), 'the largest resolution of the maker\'s min-Z table'
    if ref.get('resolution'):
        return tuple(int(v) for v in ref['resolution']), 'the reference imager\'s native resolution'
    from .. import library as L
    return tuple(L.DEFAULT_RESOLUTION), 'a default (the maker states no depth resolution)'


@dataclass
class DepthSetup:
    """What the depth model needs for one sensor, with where each figure comes from."""
    sensor: object
    entry: object
    technology: str
    resolution: tuple
    resolution_default: bool
    resolution_note: str
    ref: object                          # library.Imager: the depth (reference) imager
    needed: list                         # [library.Imager]: the imagers whose FOV and line of sight the depth needs
    projector: dict | None
    need_projector: bool
    projector_note: str
    min_z_m: float | None
    min_z_note: str
    max_range_m: float | None
    max_range_note: str
    baseline_m: float | None
    sigma_d_px: float
    sigma_d_note: str
    sigma_d_default: bool
    rms: dict | None
    notes: list = field(default_factory=list)

    @property
    def f_px(self) -> float:
        return float(self.ref.camera.fx)

    @property
    def has_error_model(self) -> bool:
        return self.technology != 'tof' and bool(self.baseline_m)


def depth_sensors(project) -> list:
    """The active variant's enabled sensors that have a depth device or are of kind 'depth'."""
    from .. import library as L
    return [s for s in project.variant.sensors if s.enabled and (s.kind == 'depth' or L.depth_spec(s) is not None)]


def setup(project, sensor, cfg: dict | None = None) -> DepthSetup:
    """The sensor's depth set-up from its library entry and the study settings; DepthSetupError says why not."""
    from .. import library as L
    st = settings(project)
    cfg = dict(sensor_settings(project, sensor.uid) if cfg is None else cfg)
    mid = sensor.model_id or ''
    entry = L.get(mid) if mid else None
    if entry is None:
        why = (f'its device "{mid}" is not in the library' if mid else 'no device is chosen for it')
        raise DepthSetupError(f'{sensor.name} is not analysed: {why}. Choose a depth device from the library.')
    dp = entry.data.get('depth')
    if not dp:
        raise DepthSetupError(f'{sensor.name} is not analysed: the {device_title(entry)} is not a depth device.')
    tech = dp.get('technology') or ''
    notes = []
    # ---- resolution
    opts = resolution_options(entry)
    native, why = native_depth_mode(entry)
    res, is_default = native, True
    note = f'{native[0]} × {native[1]}: the default, {why}'
    stream = tuple(int(v) for v in (getattr(sensor, 'stream', None) or ()))
    if len(stream) == 2 and stream != native and any((o[0], o[1]) == stream for o in opts):
        # the stream chosen for the sensor (Sensors page) is a depth mode: the Inspector's min-Z, the Camera view and
        # the frusta use it, so the study does too unless its own per-sensor setting says otherwise
        res, is_default = stream, False
        note = f'{stream[0]} × {stream[1]}: the stream chosen for {sensor.name} (Sensors page)'
    want = cfg.get('resolution')
    if want:
        try:
            w = (int(want[0]), int(want[1]))
        except (TypeError, ValueError, IndexError):
            w = None
        if w is not None and any((o[0], o[1]) == w for o in opts):
            res, is_default = w, w == native
            note = f'{w[0]} × {w[1]}: your setting' + (' (also the default)' if is_default else '')
        else:
            notes.append(f'The chosen resolution {want} is not a depth mode of the {device_title(entry)}; '
                         f'{res[0]} × {res[1]} is used.')
    # ---- imagers
    ref_im = entry.imager('depth')
    if ref_im is None:
        raise DepthSetupError(f'{sensor.name} is not analysed: the {device_title(entry)} entry names no reference '
                              'imager.')
    pair = dp.get('stereo_pair') if tech in STEREO else None
    names = list(pair) if pair else [ref_im['name']]
    if tech in STEREO and not pair:
        notes.append('The entry names no stereo pair: only the reference imager\'s field of view and line of sight '
                     'are tested.')
    try:
        ims = L.sensor_imagers(sensor, {n: res for n in names})
    except L.LibraryError as e:
        raise DepthSetupError(f'{sensor.name} is not analysed: {e}') from None
    by = {_norm(i.name): i for i in ims}
    ref = by.get(_norm(ref_im['name']))
    needed = [by[_norm(n)] for n in names if _norm(n) in by]
    if ref is None or len(needed) != len(names):
        raise DepthSetupError(f'{sensor.name} is not analysed: the {device_title(entry)} entry\'s imagers do not '
                              'match its depth block.')
    for i in needed:
        if not i.pose_stated:
            notes.append(f'{i.name}: its position in the device is an assumption (not stated by the maker).')
    # ---- projector
    pj = L.projector(sensor)
    mode = cfg.get('projector', 'auto')
    has_cone = bool(pj) and pj.get('hfov_deg') is not None and pj.get('vfov_deg') is not None
    if tech == 'structured_light':
        need, pnote = True, 'needed: structured light depth comes from the projected pattern'
    elif mode is True or mode is False:
        need = bool(mode)
        pnote = 'on: your setting' if need else 'off: your setting'
    elif tech == 'active_ir_stereo' and has_cone:
        need, pnote = True, ('on (the default for active IR stereo with a stated projector): voxels outside its cone '
                             'get no projected texture')
    elif tech == 'active_ir_stereo':
        need, pnote = False, 'not tested: the maker states no projector field of view'
    else:
        need, pnote = False, f'not needed ({TECH_TITLE.get(tech, tech)})'
    if need and not has_cone:
        need, pnote = False, 'not tested: the device has no projector with a stated field of view'
    if need and pj is not None and not pj.get('pose_stated'):
        pnote += '; its position in the device is an assumption (not stated by the maker)'
    # ---- range
    w, h = res
    mz, mz_note = None, ''
    rows = dp.get('min_z') or []
    row = next((r for r in rows if int(r.get('width', 0)) == w and int(r.get('height', 0)) == h), None)
    if row is not None and row.get('m') is not None:
        mz, mz_note = float(row['m']), f'stated for {w} × {h}'
    elif row is not None:
        mz_note = f'not stated by the maker for {w} × {h} (the table gives no figure): not tested'
    elif dp.get('min_range_m') is not None:
        mz = float(dp['min_range_m'])
        mz_note = 'the stated minimum range (the maker gives no per-resolution min-Z)'
    else:
        mz_note = f'not stated by the maker for {w} × {h}: not tested'
    mr, mr_note = None, ''
    mrow = next((r for r in dp.get('max_range_by_mode') or [] if int(r.get('width', 0)) == w
                 and int(r.get('height', 0)) == h), None)
    if mrow is not None and mrow.get('m') is not None:
        mr, mr_note = float(mrow['m']), f'stated for the {w} × {h} mode'
    elif dp.get('max_range_m') is not None:
        mr = float(dp['max_range_m'])
        mr_note = 'as stated' + (f' ("{dp["range_text"]}")' if dp.get('range_text') else '')
    else:
        mr_note = 'not stated by the maker: not tested'
    # ---- noise
    sub = dp.get('subpixel_px')
    if sub is not None:
        sd, sd_note, sd_def = float(sub), 'stated by the maker', False
    elif cfg.get('sigma_d_px') is not None:
        sd, sd_note, sd_def = float(cfg['sigma_d_px']), 'your setting for this sensor', False
    elif not st['sigma_d_default']:
        sd, sd_note, sd_def = float(st['sigma_d_px']), 'your setting', False
    else:
        sd, sd_note, sd_def = DEFAULT_SIGMA_D_PX, (f'a default ({DEFAULT_SIGMA_D_PX:g} px): the maker states no '
                                                   'subpixel accuracy'), True
    if not (sd > 0 and math.isfinite(sd)):
        raise DepthSetupError(f'The disparity error σ_d for {sensor.name} must be a positive number of pixels; it is '
                              f'{sd}.')
    return DepthSetup(sensor, entry, tech, res, is_default, note, ref, needed, pj, need, pnote, mz, mz_note, mr,
                      mr_note, dp.get('baseline_m'), sd, sd_note, sd_def, dp.get('rms_error'), notes)


# ====================================================================== geometry
def imager_world(body: Pose, imager) -> Pose:
    """The imager's optical frame in the world."""
    return body @ imager.pose_in_body


def blind_zone_local(ds: DepthSetup, per_side: int = 6):
    """The min-Z blind zone in the sensor body frame: (triangles (K, 3, 3), edge segments (S, 2, 3)), the part of the
    reference imager's field of view closer than min-Z (a pyramid from the optical centre to the plane z = min-Z);
    None when min-Z is not stated."""
    if ds.min_z_m is None:
        return None
    cam = ds.ref.camera
    W, H = cam.width, cam.height
    t = np.linspace(0.0, 1.0, per_side + 1)[:-1]
    e = 0.5
    border = np.concatenate([
        np.stack([-e + t * W, np.full_like(t, -e)], 1),
        np.stack([np.full_like(t, W - e), -e + t * H], 1),
        np.stack([W - e - t * W, np.full_like(t, H - e)], 1),
        np.stack([np.full_like(t, -e), H - e - t * H], 1)])
    rays = cam.unproject(border)
    good = np.isfinite(rays).all(axis=1) & (rays[:, 2] > 1e-6)
    rays = rays[good]
    if len(rays) < 3:
        return None
    far = rays * (ds.min_z_m / rays[:, 2:3])
    T = ds.ref.pose_in_body
    far_b = T.apply(far)
    apex = T.apply(np.zeros((1, 3)))[0]
    cap = T.apply(np.array([[0.0, 0.0, ds.min_z_m]]))[0]
    nxt = np.roll(far_b, -1, axis=0)
    sides = np.stack([np.broadcast_to(apex, far_b.shape), far_b, nxt], 1)
    caps = np.stack([np.broadcast_to(cap, far_b.shape), nxt, far_b], 1)
    tris = np.concatenate([sides, caps])
    k = len(far_b)
    corners = [0, per_side, 2 * per_side, 3 * per_side] if good.all() else list(range(0, k, max(1, k // 4)))
    segs = [[apex, far_b[i]] for i in corners] + [[far_b[i], far_b[(i + 1) % k]] for i in range(k)]
    return tris, np.asarray(segs, float)


def _projector_cone(P_world: np.ndarray, body: Pose, pj: dict) -> np.ndarray:
    """(N,) True inside the projector's rectangular cone (its stated horizontal and vertical FOV)."""
    T = (body @ pj['pose']).inverse()
    Q = T.apply(P_world)
    th = math.tan(math.radians(float(pj['hfov_deg'])) / 2)
    tv = math.tan(math.radians(float(pj['vfov_deg'])) / 2)
    z = Q[:, 2]
    with np.errstate(invalid='ignore', divide='ignore'):
        return (z > 0) & (np.abs(Q[:, 0]) <= th * z) & (np.abs(Q[:, 1]) <= tv * z)


def material_objects(project, technology: str) -> list:
    """[(object, [tags that warn], reason text)] of the task objects whose tags warn for this technology."""
    out = []
    for o in project.workspace.objects:
        tags = set(o.materials or ())
        warn = []
        if 'transparent' in tags:
            warn.append('transparent')
        if 'ir_absorbing' in tags and technology in ACTIVE:
            warn.append('ir_absorbing')
        if tags & set(SPECULAR_TAGS):
            warn.append('specular')
        if warn:
            out.append((o, warn, '; '.join(MATERIAL_REASON[w] for w in warn)))
    return out


def _on_object(o, P: np.ndarray, margin: np.ndarray, project) -> np.ndarray:
    """(N,) True where P lies inside the object's shape grown by margin (per point)."""
    from ..frames import quat_to_matrix
    from ..scenegeom import object_triangles
    s = np.r_[np.asarray(o.size, float).reshape(-1), np.zeros(3)][:3]
    R, t = quat_to_matrix(o.pose.quaternion), np.asarray(o.pose.position, float)
    L = (P - t) @ R
    if o.shape == 'box':
        return (np.abs(L) <= np.abs(s) / 2 + margin[:, None]).all(axis=1)
    if o.shape == 'cylinder':
        return (np.hypot(L[:, 0], L[:, 1]) <= abs(s[0]) + margin) & (np.abs(L[:, 2]) <= abs(s[1]) / 2 + margin)
    if o.shape == 'sphere':
        return np.linalg.norm(L, axis=1) <= abs(s[0]) + margin
    tri = object_triangles(o, project)
    if not len(tri):
        return np.zeros(len(P), bool)
    v = tri.reshape(-1, 3)
    lo, hi = v.min(axis=0), v.max(axis=0)
    return ((P >= lo - margin[:, None]) & (P <= hi + margin[:, None])).all(axis=1)


def evaluate(ds: DepthSetup, body: Pose, P, *, occluders=None, size=None, mats=(), project=None,
             n: int = DEFAULT_CUBE_N, backend: str = 'auto', cubes: dict | None = None) -> dict:
    """The five conditions at world points P (N, 3) for the sensor whose body frame is at body (world):
    {'z', 'in_fov', 'min_z_ok', 'max_ok', 'visible' (N, k) per needed imager, 'proj_ok', 'material' (N,) bitmask of
    mats, 'code' (N,) first failing reason (0 valid, 1 + REASONS index)}. occluders: core.scenegeom.Occluders (None:
    nothing occludes). cubes: a dict to reuse the cube maps in (keyed by imager name)."""
    from .. import occlusion as OC
    P = np.asarray(P, float).reshape(-1, 3)
    N = len(P)
    poses = [imager_world(body, im) for im in ds.needed]
    in_fov = np.ones(N, bool)
    z = None
    for im, T in zip(ds.needed, poses):
        Q = T.inverse().apply(P)
        _uv, ok = im.camera.project(Q)
        in_fov &= ok
        if im is ds.ref:
            z = Q[:, 2]
    if z is None:
        z = imager_world(body, ds.ref).inverse().apply(P)[:, 2]
    if ds.technology == 'structured_light' and ds.projector is not None and ds.need_projector:
        in_fov &= _projector_cone(P, body, ds.projector)
    min_ok = np.ones(N, bool) if ds.min_z_m is None else z >= ds.min_z_m
    max_ok = np.ones(N, bool) if ds.max_range_m is None else z <= ds.max_range_m
    vis = np.ones((N, len(ds.needed)), bool)
    if occluders is not None and len(occluders):
        idx = np.nonzero(in_fov)[0]
        for k, T in enumerate(poses):
            if not len(idx):
                break
            key = ds.needed[k].name
            cube = None if cubes is None else cubes.get(key)
            if cube is None:
                cube = OC.CACHE.get(occluders, T.position, n, backend)
                if cubes is not None:
                    cubes[key] = cube
            vis[idx, k] = cube.visible(P[idx])
    proj_ok = np.ones(N, bool)
    if ds.need_projector and ds.projector is not None and ds.technology != 'structured_light':
        proj_ok = _projector_cone(P, body, ds.projector)
    mat = np.zeros(N, np.int64)
    if mats:
        marg = (np.full(N, 0.0) if size is None else np.asarray(size, float).reshape(-1) * (math.sqrt(3) / 2))
        for j, (o, _tags, _why) in enumerate(mats):
            mat[_on_object(o, P, marg, project)] |= 1 << j
    conds = [in_fov, min_ok, max_ok, vis.all(axis=1), proj_ok, mat == 0]
    code = np.zeros(N, np.int8)
    for k in range(len(conds) - 1, -1, -1):
        code[~conds[k]] = k + 1
    return {'z': z, 'in_fov': in_fov, 'min_z_ok': min_ok, 'max_ok': max_ok, 'visible': vis, 'proj_ok': proj_ok,
            'material': mat, 'code': code}


# ====================================================================== the study
def _pose_q(project, st) -> tuple:
    """(q dict or None, {'kind', 'note', 'q'}) from the settings: q when given (unless pose is 'home'), else home."""
    from ..robot.model import RobotModelError, get_model
    given = st.get('q') or {}
    if st.get('pose') == 'home' or not given:
        return None, {'kind': 'home', 'q': {}, 'note': 'the robots at their home pose (the Depth page runs the study '
                                                       'at the pose on screen)'}
    q, bad = {}, []
    for uid, v in given.items():
        r = project.robot(uid)
        if r is None:
            continue
        try:
            m = get_model(project, uid)
        except RobotModelError:
            continue
        a = np.asarray(v, float).reshape(-1)
        if len(a) != m.nq or not np.isfinite(a).all():
            bad.append(r.name)
            continue
        q[uid] = m.clip(a)
    note = 'the joint state given with the run (the pose on screen when it was run from the Depth page)'
    if bad:
        note += f'; {", ".join(bad)} at home (the given joint values do not fit the robot)'
    return q, {'kind': 'current', 'note': note, 'q': {k: [float(x) for x in v] for k, v in q.items()}}


def _pct(x, total) -> float:
    return float(100.0 * x / total) if total > 0 else 0.0


def _status(pct: float, st) -> tuple:
    good, fair = float(st['good_pct']), float(st['fair_pct'])
    if pct >= good:
        return 'good', 'good'
    if pct >= fair:
        return 'warning', 'partial'
    return 'critical', 'poor'


def _fingertips(project, q, body: Pose, ds: DepthSetup) -> list:
    from ..robot.model import RobotModelError, get_model
    out = []
    T = imager_world(body, ds.ref).inverse()
    for r in project.robots:
        if not r.fingertip_frames:
            continue
        try:
            m = get_model(project, r.uid)
        except RobotModelError:
            continue
        qq = (q or {}).get(r.uid)
        qq = m.home if qq is None else qq
        for link in r.fingertip_frames:
            try:
                p = m.link_pose(qq, link)[:3, 3]
            except KeyError:
                continue
            Q = T.apply(p[None])
            _uv, ok = ds.ref.camera.project(Q)
            zz = float(Q[0, 2])
            out.append({'robot': r.name, 'link': link, 'z_m': zz, 'in_fov': bool(ok[0]),
                        'inside_min_z': bool(ds.min_z_m is not None and 0 < zz < ds.min_z_m)})
    return out


def _suggest(ds: DepthSetup, ev: dict, w: np.ndarray, total: float, tips: list = ()) -> dict | None:
    """A depth mode of the same aspect ratio (so the same field of view) with a smaller stated min-Z: the one that
    makes the most task volume depth-valid, estimated from the voxels that fail min-Z alone; when none gains volume,
    the largest one whose min-Z clears the fingertips that sit inside min-Z in view. Ties go to the larger
    resolution (the smaller σ_z penalty)."""
    if ds.min_z_m is None:
        return None
    others = ev['in_fov'] & ev['max_ok'] & ev['visible'].all(axis=1) & ev['proj_ok'] & (ev['material'] == 0)
    W, H = ds.resolution
    alts = [(w2, h2, m) for w2, h2, m in resolution_options(ds.entry)
            if m is not None and m < ds.min_z_m and (w2, h2) != (W, H) and abs(w2 / h2 - W / H) <= 0.01 * (W / H)]
    best = None
    for w2, h2, m in alts:
        gain = float(w[others & (ev['z'] < ds.min_z_m) & (ev['z'] >= m)].sum())
        if gain > 0 and (best is None or gain > best[0]):
            best = (gain, w2, h2, m)
    why = ''
    if best is None:
        near = [t['z_m'] for t in tips if t['in_fov'] and t['inside_min_z']]
        fit = [a for a in alts if near and a[2] <= min(near)]
        if not fit:
            return None
        w2, h2, m = fit[0]
        best = (0.0, w2, h2, m)
        why = f', which clears the fingertips at {min(near) * 1000:.0f} mm (inside min-Z now)'
    gain, w2, h2, m = best
    f2 = ds.f_px * w2 / W
    pct = _pct(gain, total)
    text = (f'Set {ds.sensor.name} to {w2} × {h2}: its min-Z is {m * 1000:.0f} mm instead of '
            f'{ds.min_z_m * 1000:.0f} mm' + (why or f', which makes about {pct:.1f} % more of the task volume '
                                                  'depth-valid'))
    if ds.has_error_model:
        text += f'; σ_z rises {W / w2:.2f}× (focal length about {f2:.0f} px instead of {ds.f_px:.0f} px)'
    return {'sensor_uid': ds.sensor.uid, 'sensor': ds.sensor.name, 'resolution': [w2, h2], 'min_z_m': m,
            'gain_pct': pct, 'fingertips': bool(why), 'text': text + '.'}


@jobs.study(STUDY_ID, 'Depth validity', quick=False, batch='depth-validity', needs_robot=False)
def depth_validity(project, ctx) -> dict:
    """Per depth sensor: the share of the task volumes that is depth-valid, the reasons for the rest, the expected
    depth noise against the maker's figure, the min-Z blind zone and the invalid voxels for the 3D view."""
    from .. import library as L
    from .. import scenegeom as SG
    from .. import voxels as VX
    from ..robot.model import RobotModelError
    st = settings(project)
    q, pose = _pose_q(project, st)
    sensors = depth_sensors(project)
    out_s, skipped = [], []
    eff = {k: st[k] for k in ('sigma_d_px', 'cube_n', 'backend', 'pose', 'good_pct', 'fair_pct')}
    eff['sigma_d_default'] = st['sigma_d_default']
    base = {'pose': pose, 'settings': eff, 'sensors': out_s, 'skipped': skipped, 'suggestion': None,
            'reasons': {k: REASON_TITLE[k] for k in ('valid',) + REASONS}}
    setups = []
    for s in sensors:
        try:
            ds = setup(project, s)
        except DepthSetupError as e:
            skipped.append({'uid': s.uid, 'name': s.name, 'why': str(e)})
            continue
        try:
            body = SG.sensor_world_pose(project, s, q)
        except (RobotModelError, KeyError, ValueError) as e:
            skipped.append({'uid': s.uid, 'name': s.name, 'why': f'{s.name} cannot be placed: {e}'})
            continue
        setups.append((ds, body))
    if not project.workspace.volumes:
        base.update(message='There is no task volume: add one (Workspace ▸ Add task volume) to see where the depth '
                            'is valid.', voxels={'count': 0}, rig={'valid_pct': None})
        return base
    if not setups:
        off = [x for x in project.variant.sensors if not x.enabled and (x.kind == 'depth' or L.depth_spec(x))]
        why = (('There is no depth sensor in use: ' + ', '.join(x.name for x in off[:3])
                + (' is' if len(off) == 1 else ' are') + ' switched off; tick "In use" on the Sensors and mounts '
                'page.') if not sensors and off else
               'There is no depth sensor in this rig: add one and choose its device from the library.'
               if not sensors else 'No depth sensor could be analysed: ' + ' '.join(k['why'] for k in skipped))
        base.update(message=why, voxels={'count': 0}, rig={'valid_pct': None})
        return base
    grid = VX.SparseGrid.from_project(project)            # StudyError (plain English) when it is too large
    ctx.report(0.02, f'{grid.count:,} voxels')
    # a voxel whose centre lies inside a solid task object (box, cylinder, sphere) is not space a sensor has to
    # measure: it is left out of the task volume, as in the coverage study (else every such voxel reads 'occluded')
    from .coverage import _inside_objects
    inside = _inside_objects(project, grid.centers.astype(np.float64))
    wts = np.where(inside, 0.0, grid.volume_m3)
    total = float(wts.sum())
    occ = SG.scene_occluders(project, q)
    any_valid = np.zeros(grid.count, bool)
    n, backend = int(st['cube_n']), str(st['backend'])
    best_sug = None
    for k, (ds, body) in enumerate(setups):
        if ctx.cancelled():
            break
        s = ds.sensor
        origins = [imager_world(body, im).position for im in ds.needed]
        occ_s = SG.sensor_occluders(project, s, q, origins=origins, base=occ)
        mats = material_objects(project, ds.technology)
        cubes: dict = {}
        evs = []
        for ch in grid.chunks(CHUNK):
            if ctx.cancelled():
                break
            evs.append(evaluate(ds, body, ch.centers.astype(np.float64), occluders=occ_s, size=ch.size, mats=mats,
                                project=project, n=n, backend=backend, cubes=cubes))
            ctx.report((k + ch.stop / max(grid.count, 1)) / len(setups), s.name)
        if len(evs) != len(list(grid.chunks(CHUNK))):
            break
        ev = {key: np.concatenate([e[key] for e in evs]) for key in evs[0]}
        code = ev['code']
        valid = code == 0
        geo_valid = (code == 0) | (code == REASONS.index('material') + 1)
        any_valid |= valid
        breakdown = {'valid': _pct(wts[valid].sum(), total)}
        for j, r in enumerate(REASONS):
            breakdown[r] = _pct(wts[code == j + 1].sum(), total)
        infov = ev['in_fov']
        vis = ev['visible']
        hid_any = infov & ~vis.all(axis=1)
        hid_all = infov & ~vis.any(axis=1)
        fails = {'fov': _pct(wts[~infov].sum(), total),
                 'min_z': _pct(wts[infov & ~ev['min_z_ok']].sum(), total),
                 'max_range': _pct(wts[infov & ~ev['max_ok']].sum(), total),
                 'occluded': _pct(wts[hid_any].sum(), total),
                 'projector': _pct(wts[infov & ~ev['proj_ok']].sum(), total),
                 'material': _pct(wts[infov & (ev['material'] != 0)].sum(), total)}
        occl = {'one_imager_pct': _pct(wts[hid_any & ~hid_all].sum(), total) if len(ds.needed) > 1 else 0.0,
                'all_imagers_pct': _pct(wts[hid_all].sum(), total)}
        per_vol = []
        for i, v in enumerate(grid.volumes):
            m = grid.volume_index == i
            per_vol.append({'uid': v.uid, 'name': v.name, 'valid_pct': _pct(wts[m & valid].sum(), wts[m].sum())})
        # ---- noise over the geometrically valid voxels
        zv, wv = ev['z'][geo_valid & ~inside], wts[geo_valid & ~inside]
        sz = None
        if ds.has_error_model and len(zv):
            sig = sigma_z(zv, ds.f_px, ds.baseline_m, ds.sigma_d_px)
            p5, med, p95 = weighted_quantile(sig, wv, (0.05, 0.5, 0.95))
            rel = weighted_quantile(sig / zv, wv, (0.5,))[0]
            sz = {'median_m': med, 'p5_m': p5, 'p95_m': p95, 'median_rel': rel, 'interval': '5th–95th percentile '
                  'over the depth-valid voxels (volume weighted)', 'voxels': int((geo_valid & ~inside).sum())}
        sz_why = '' if sz is not None else (
            'the stereo error model does not apply to a time-of-flight camera' if ds.technology == 'tof' else
            'the maker states no baseline' if not ds.baseline_m else 'no voxel is depth-valid')
        maker = None
        if ds.rms:
            # what the maker's figure measures (library rms_error.metric; missing: not defined by the maker). Only an
            # RMS figure is comparable with σ_z closely enough to solve for a matching σ_d.
            metric = ds.rms.get('metric') or 'unspecified'
            maker = {'text': ds.rms.get('text', ''), 'pct': ds.rms.get('pct'), 'at_m': ds.rms.get('at_m'),
                     'metric': metric, 'sigma_m': None, 'model_sigma_m': None, 'sigma_d_equiv_px': None}
            if ds.rms.get('pct') is not None and ds.rms.get('at_m'):
                zm = float(ds.rms['at_m'])
                maker['sigma_m'] = float(ds.rms['pct']) / 100.0 * zm     # the stated bound in metres, whatever metric
                if ds.has_error_model:
                    maker['model_sigma_m'] = float(sigma_z(zm, ds.f_px, ds.baseline_m, ds.sigma_d_px))
                    if metric == 'rms':
                        maker['sigma_d_equiv_px'] = sigma_d_equivalent(maker['sigma_m'], zm, ds.f_px,
                                                                       ds.baseline_m)
        mats_out = []
        for j, (o, tags, why) in enumerate(mats):
            on = (ev['material'] & (1 << j)) != 0
            mats_out.append({'uid': o.uid, 'object': o.name, 'tags': tags, 'reason': why,
                             'pct': _pct(wts[on & (code == REASONS.index('material') + 1)].sum(), total)})
        # ---- overlay: a sample of the invalid voxels per reason
        pts, rs = [], []
        for j in range(1, len(REASONS) + 1):
            idx = np.nonzero((code == j) & ~inside)[0]
            if len(idx) > OVERLAY_PER_REASON:
                idx = idx[np.linspace(0, len(idx) - 1, OVERLAY_PER_REASON).astype(np.int64)]
            pts.append(np.round(grid.centers[idx].astype(np.float64) * 1000).astype(np.int64))
            rs.append(np.full(len(idx), j, np.int64))
        tips = _fingertips(project, q, body, ds)
        sug = _suggest(ds, ev, wts, total, tips)
        if sug is not None and (best_sug is None or sug['gain_pct'] > best_sug['gain_pct']):
            best_sug = sug
        vp = _pct(wts[valid].sum(), total)
        sev, word = _status(vp, st)
        out_s.append({
            'uid': s.uid, 'name': s.name, 'slot': s.slot, 'model_id': ds.entry.id, 'title': device_title(ds.entry),
            'technology': ds.technology, 'technology_title': TECH_TITLE.get(ds.technology, ds.technology),
            'resolution': list(ds.resolution), 'resolution_default': ds.resolution_default,
            'resolution_note': ds.resolution_note,
            'min_z_m': ds.min_z_m, 'min_z_note': ds.min_z_note, 'max_range_m': ds.max_range_m,
            'max_range_note': ds.max_range_note, 'baseline_m': ds.baseline_m, 'f_px': ds.f_px,
            'sigma_d_px': ds.sigma_d_px, 'sigma_d_default': ds.sigma_d_default, 'sigma_d_note': ds.sigma_d_note,
            'projector': {'needed': ds.need_projector, 'note': ds.projector_note,
                          'hfov_deg': (ds.projector or {}).get('hfov_deg'),
                          'vfov_deg': (ds.projector or {}).get('vfov_deg'),
                          'pose_stated': bool((ds.projector or {}).get('pose_stated', False))},
            'imagers': [{'name': im.name, 'role': im.role, 'label': im.label, 'pose_stated': im.pose_stated,
                         'origin': [float(v) for v in o]} for im, o in zip(ds.needed, origins)],
            'valid_pct': vp, 'status': sev, 'status_word': word,
            'valid_geometric_pct': _pct(wts[geo_valid].sum(), total), 'valid_m3': float(wts[valid].sum()),
            'breakdown': breakdown, 'fails': fails, 'occlusion': occl, 'per_volume': per_vol,
            'sigma_z': sz, 'sigma_z_why': sz_why, 'maker_rms': maker, 'materials': mats_out,
            'fingertips': tips, 'suggestion': sug,
            'cube': {'n': n, 'backend': next(iter(cubes.values())).backend if cubes else '',
                     'seconds': float(sum(c.seconds for c in cubes.values()))},
            'overlay': {'points_mm': np.concatenate(pts).tolist(), 'reason': np.concatenate(rs).tolist(),
                        'voxel_mm': float(grid.voxel_m * 1000)},
            'world_pose': body.to_dict(), 'notes': list(ds.notes),
        })
    rig = _pct(wts[any_valid].sum(), total) if out_s else None
    base.update(voxels={'count': grid.count, 'voxel_mm': float(grid.voxel_m * 1000),
                        'refine_mm': float(grid.refine_m * 1000), 'message': grid.info.get('message', ''),
                        'volume_m3': total, 'inside_objects': int(inside.sum())},
                rig={'valid_pct': rig, 'status': _status(rig, st)[0] if rig is not None else 'warning',
                     'status_word': _status(rig, st)[1] if rig is not None else 'not run'},
                suggestion=best_sug, occluders={'triangles': int(len(occ)), 'notes': list(occ.notes)},
                message='')
    ctx.report(1.0, 'done')
    return base
