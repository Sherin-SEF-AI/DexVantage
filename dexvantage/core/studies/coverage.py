"""Coverage and resolution in the task volume (spec §7.1).

    @jobs.study('coverage', 'Coverage and resolution', quick=False, batch='coverage')   → --coverage OUT.json
    run(project, ctx) -> dict      the study (also through core.jobs.run_study('coverage', project))
    decode(doc) -> dict            the per-voxel arrays of a result as numpy arrays (centres, values per camera)
    status_for(pct, settings)      the headline's status and word for a share seen by two or more cameras

What it computes. Every task volume is voxelised as a sparse grid with adaptive refinement (core/voxels.py:
project.analysis.voxel_mm, 5 mm by default, refined to refine_mm, 2 mm by default, inside refinement regions and
around the task objects). For every voxel centre P and every camera c:

- in image: P, moved into the imager's optical frame (z forward, x right, y down) by the sensor's world pose (its
  mount, posed on its link at the joint state q) and the imager's pose in the sensor body (core/library), projects
  inside the image with the imager's lens model (core/optics CameraModel.project: in front, within the valid angle,
  u in [−0.5, W − 0.5], v in [−0.5, H − 0.5]).
- visible: in image and not occluded: nothing of the scene (the robots' visual meshes posed at q, the task objects,
  the tables; core/scenegeom.py) is nearer along the line of sight, by a cube map of distances built around the
  imager's optical centre (core/occlusion.py, face size cube_n = 512 by default, the GPU when there is one). The
  sensor's own seat on its mount link is left out by scenegeom.sensor_occluders' rule.
- pixels per millimetre: CameraModel.pixels_per_mm, the local linear pixel density √|det ∂(u, v)/∂(lateral)| of a
  small patch at P facing the camera (a surface tilted by α away from the line of sight has √cos α of it). It is
  the density of the lens model at P whether or not P is occluded; the statistics use visible voxels only.
- viewing angle: the angle between the optical axis and the line of sight to P (CameraModel.angle_to_axis).
- defocus: the thin-lens blur-disc diameter in pixels, b = (f/N) f |s − z| / (z (s − f)) / pixel
  (optics.blur_circle_px), with z the point's depth along the optical axis. It needs the f-number N, the focus
  distance s, and the focal length f in millimetres (from the pixel pitch: f = fx · pitch). The library states N for
  many imagers but no maker in it states a fixed-focus lens's focus distance, and a few give no pixel pitch, so the
  blur is computed only where every figure is known: the entry's, or one the user measured and entered in the
  study's settings ('optics', below), labelled as the user's. Otherwise it is "not computed" and the camera's record
  says which figure is missing. A blur up to coc_px (default optics.DEFAULT_COC_PX = 2 px, a default) counts as
  sharp. The pitch is taken as binned with the stream (pitch × native width / stream width), an assumption.
- redundancy: how many cameras see the voxel.

Which imager is "the camera": a camera sensor's first imager (its entry's reference imager); a depth sensor's RGB
imager when it has one (the colour image is what a policy is trained on), else its reference (depth) imager.
settings 'depth_imager': 'depth' uses the reference imager of every depth sensor, and settings 'imagers'
{sensor uid: imager name} picks one per sensor. The depth-validity study (§7.2) treats the stereo pair. Each camera
record names the imager used and why. Tactile sensors are contact sensors and are not cameras here; a sensor with no
library device gets the library's labelled default imager (90° × 60°, 1280 × 720 px, "no device chosen").

The joint state. A study gets only the project, so q is read from the study's settings,
project.analysis.studies['coverage']['q'] = {robot uid: [joint values in RobotModel.joint_names order]}; a robot not
named there is at its home pose. The Coverage page's Run passes the 3D view's current q in those settings (on the
copy of the project the study runs on, so the project is not changed), and batch mode uses whatever the saved
project's settings hold (home when nothing). The result records the q used and where it came from.

Summary. Fractions are of the task volume, each voxel weighted by its volume (so refined voxels do not count more
than the space they fill), leaving out voxels whose centre lies inside a task object (box, cylinder or sphere; a
solid's inside can be seen by no camera): seen by ≥ 1 and ≥ 2 cameras, by none, and resolved (seen at ≥
min_px_per_mm, default 1 px/mm, a default) by ≥ 1 and ≥ 2; per camera the share it sees, the share outside its image
and the share in its image but hidden, the volume-weighted median, 5th and 95th percentile of pixels per mm over the
voxels it sees, the median and largest viewing angle, and the blur; per task volume the same coverage shares. The
headline status of the share seen by two or more cameras is 'good' from good_pct (default 90 %), 'warning' from
warn_pct (60 %), 'serious' from serious_pct (30 %) and 'critical' below; all three are user settings with documented
defaults, not standards.

Settings (project.analysis.studies['coverage'], every key optional): q; voxel_mm, refine_mm (else
project.analysis'); good_pct, warn_pct, serious_pct; min_px_per_mm; coc_px; cube_n; backend ('auto', 'gl',
'numpy'); depth_imager ('rgb' or 'depth'); imagers {uid: name}; modes {uid: [w, h]} (a stream mode, else the
stream chosen for the sensor on the Sensors page, Sensor.stream, as the Camera view uses it, else the imager's
largest listed stream); optics {uid: {'focus_m', 'f_number', 'pixel_um', 'focal_mm'}} (the user's own
figures for the defocus); sensors [uid] (only these; default every enabled camera and depth sensor); chunk (voxels
per step, 100 000).

The result is JSON-able. The per-voxel arrays are quantised and stored zlib-compressed in base64 ('voxels', see
decode): voxel centres as integer multiples of half the refined edge from their volume's grid corner; per camera a
state (0 outside the image, 1 in the image but hidden, 2 visible), pixels per mm in steps of 0.01 px/mm (to
655.35), the viewing angle in steps of 0.5° and the blur in steps of 0.1 px (to 25.3 px; 25.4 means that or more).
The 480 000 voxels of the UR5e sample with its two cameras take 1.7 MB of JSON.

Validation: occlusion.coverage_vs_raycast compares the study's coverage on a scripted scene with an independent
path (pinhole projection written out from the stated FOV and exact ray casting, core/raycast.py), and
optics.coverage_density_fd compares its pixels per mm with finite differences of OpenCV's projectPoints
(dexvantage/validation/checks_coverage.py).

Known limits: visibility is judged at the voxel centre and per cube-map texel (a voxel within about a texel of a
silhouette may be judged either way); a voxel inside a robot link or a mesh object counts as unseen (only box,
cylinder and sphere objects are tested for their insides); the density is for a patch facing the camera (the task's
surfaces are not known); no sensor housings occlude; overlapping task volumes count their shared space once for each;
the defocus is a thin-lens model and is computed only from stated or entered figures; lighting, exposure, motion blur
and the depth sensor's own validity (min-Z, the stereo pair, the projector) are other studies (§7.2, §7.8, §7.9).
"""
from __future__ import annotations

import base64
import math
import time
import zlib

import numpy as np

from .. import jobs
from ..jobs import StudyError

DEFAULTS = {
    'good_pct': 90.0,          # default: share seen by >= 2 cameras that reads 'good'
    'warn_pct': 60.0,          # default: ... 'warning' from here (below it 'serious')
    'serious_pct': 30.0,       # default: ... 'serious' from here (below it 'critical')
    'min_px_per_mm': 1.0,      # default: a voxel is "resolved" by a camera at this density or more (1 mm ≈ 1 px)
    'coc_px': None,            # default optics.DEFAULT_COC_PX (2 px): blur up to this counts as sharp
    'cube_n': 512,             # default cube-map face size (texels)
    'backend': 'auto',
    'depth_imager': 'rgb',     # depth sensors: their RGB imager when they have one ('depth': the reference imager)
    'chunk': 100_000,
}
STATE_OUT, STATE_HIDDEN, STATE_SEEN = 0, 1, 2
PPMM_STEP = 0.01               # px/mm per stored unit (uint16)
ANGLE_STEP = 0.5               # degrees per stored unit (uint8; 255 = not in the image)
BLUR_STEP = 0.1                # px per stored unit (uint8; 254 = 25.4 px or more, 255 = not computed / not in image)
WORDS = {'good': 'well covered', 'warning': 'thin', 'serious': 'gaps', 'critical': 'mostly single or blind'}


# ====================================================================== settings
def settings_of(project) -> dict:
    """The study's settings with the defaults filled in; 'defaults' lists the keys left at their default."""
    from ..optics import DEFAULT_COC_PX
    given = dict(project.analysis.studies.get('coverage') or {})
    out = dict(DEFAULTS)
    out['coc_px'] = DEFAULT_COC_PX
    used_default = []
    for k in DEFAULTS:
        if given.get(k) is not None:
            out[k] = given[k]
        else:
            used_default.append(k)
    for k in ('good_pct', 'warn_pct', 'serious_pct', 'min_px_per_mm', 'coc_px'):
        try:
            out[k] = float(out[k])
        except (TypeError, ValueError):
            raise StudyError(f'The coverage setting {k} must be a number; it is {out[k]!r}.') from None
        if not math.isfinite(out[k]) or out[k] < 0:
            raise StudyError(f'The coverage setting {k} must be a positive number; it is {out[k]:g}.')
    try:
        out['cube_n'] = int(out['cube_n'])
        out['chunk'] = max(1000, int(out['chunk']))
    except (TypeError, ValueError):
        raise StudyError('The coverage settings cube_n and chunk must be whole numbers.') from None
    if not 16 <= out['cube_n'] <= 4096:
        raise StudyError(f'The cube-map face size must be 16 to 4096 texels; it is {out["cube_n"]}.')
    if out['backend'] not in ('auto', 'gl', 'numpy'):
        raise StudyError(f'The occlusion backend is auto, gl or numpy; it is {out["backend"]!r}.')
    if out['depth_imager'] not in ('rgb', 'depth'):
        raise StudyError(f'depth_imager is "rgb" or "depth"; it is {out["depth_imager"]!r}.')
    for k in ('q', 'imagers', 'modes', 'optics'):
        v = given.get(k)
        out[k] = dict(v) if isinstance(v, dict) else {}
    out['sensors'] = [str(u) for u in given.get('sensors') or []]
    out['voxel_mm'] = given.get('voxel_mm')
    out['refine_mm'] = given.get('refine_mm')
    out['defaults'] = used_default
    return out


def status_for(pct2: float, st: dict) -> tuple[str, str]:
    """(severity, word) of the share (%) of the task volume seen by two or more cameras."""
    if pct2 >= st['good_pct']:
        sev = 'good'
    elif pct2 >= st['warn_pct']:
        sev = 'warning'
    elif pct2 >= st['serious_pct']:
        sev = 'serious'
    else:
        sev = 'critical'
    return sev, WORDS[sev]


# ====================================================================== compact arrays
def encode_array(a: np.ndarray) -> dict:
    """{'dtype', 'shape', 'zlib_b64'}: the array's bytes (little-endian, C order) compressed and in base64."""
    a = np.ascontiguousarray(a)
    if a.dtype.byteorder == '>':
        a = a.astype(a.dtype.newbyteorder('<'))
    return {'dtype': a.dtype.str.lstrip('<|='), 'shape': list(a.shape),
            'zlib_b64': base64.b64encode(zlib.compress(a.tobytes(), 6)).decode('ascii')}


def decode_array(d: dict) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(d['zlib_b64']))
    return np.frombuffer(raw, dtype=np.dtype(d['dtype']).newbyteorder('<')).reshape(d['shape']).copy()


def decode(doc: dict) -> dict:
    """The per-voxel arrays of a coverage result: {'centers' (N, 3) m, 'size' (N,) m, 'volume_index' (N,),
    'inside' (N,) bool, 'redundancy' (N,), 'best_ppmm' (N,) px/mm, 'cameras': [{'state' (N,) uint8 (STATE_*),
    'ppmm' (N,) px/mm (0 outside the image), 'angle_deg' (N,) (NaN outside the image), 'blur_px' (N,) or None}]}."""
    vx = doc['voxels']
    ijk = decode_array(vx['ijk']).astype(np.float64)
    vi = decode_array(vx['volume_index']).astype(np.int32)
    org = np.asarray(vx['origins'], float).reshape(-1, 3)
    unit = float(vx['unit_m'])
    centers = org[vi] + ijk * unit if len(org) else np.zeros((0, 3))
    refined = decode_array(vx['refined']).astype(bool)
    size = np.where(refined, float(vx['refine_m']), float(vx['voxel_m']))
    cams = []
    for c in vx['cameras']:
        ang = decode_array(c['angle']).astype(np.float64)
        ang = np.where(ang == 255, np.nan, ang * ANGLE_STEP)
        blur = None
        if c.get('blur') is not None:
            b = decode_array(c['blur']).astype(np.float64)
            blur = np.where(b == 255, np.nan, b * BLUR_STEP)
        cams.append({'state': decode_array(c['state']), 'ppmm': decode_array(c['ppmm']).astype(np.float64) * PPMM_STEP,
                     'angle_deg': ang, 'blur_px': blur})
    return {'centers': centers, 'size': size, 'volume_index': vi, 'inside': decode_array(vx['inside']).astype(bool),
            'redundancy': decode_array(vx['redundancy']), 'best_ppmm':
                decode_array(vx['best_ppmm']).astype(np.float64) * PPMM_STEP, 'cameras': cams}


# ====================================================================== helpers
def _num(x, nd=6):
    """A JSON-safe float (None for NaN / inf / None)."""
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def _wfrac(mask, w) -> float | None:
    tot = float(w.sum())
    return float(w[mask].sum() / tot) if tot > 0 else None


def wpercentile(v, w, q: float) -> float | None:
    """The volume-weighted q-th percentile (0–100) of values v with weights w: the smallest v whose cumulative
    weight reaches q % (None when there are none)."""
    v, w = np.asarray(v, float), np.asarray(w, float)
    good = np.isfinite(v) & (w > 0)
    if not good.any():
        return None
    v, w = v[good], w[good]
    o = np.argsort(v, kind='stable')
    c = np.cumsum(w[o])
    i = int(np.searchsorted(c, q / 100.0 * c[-1] - 1e-15 * c[-1]))
    return float(v[o][min(i, len(v) - 1)])


def joint_state(project, st: dict, notes: list) -> tuple[dict, dict]:
    """({robot uid: q}, {robot uid: 'settings' | 'home'}) for the robots that can be posed."""
    from ..robot.model import RobotModelError, get_model
    qd, src = {}, {}
    given = st.get('q') or {}
    for r in project.robots:
        try:
            m = get_model(project, r.uid)
        except RobotModelError as e:
            notes.append(f'{r.name} could not be posed and is left out: {e}')
            continue
        v = given.get(r.uid)
        if v is None:
            qd[r.uid], src[r.uid] = m.home.copy(), 'home'
            continue
        a = np.asarray(v, float).reshape(-1)
        if len(a) != m.nq:
            raise StudyError(f'The joint state given for {r.name} has {len(a)} values; the robot has {m.nq} moving '
                             'joints. Run the study from the Coverage page, or give one value per joint.')
        if not np.isfinite(a).all():
            raise StudyError(f'The joint state given for {r.name} holds a value that is not a number.')
        c = m.clip(a)
        if not np.allclose(c, a):
            notes.append(f'The joint state given for {r.name} was outside its joint limits and was clipped to them.')
        qd[r.uid], src[r.uid] = c, 'settings'
    return qd, src


def _inside_objects(project, P: np.ndarray) -> np.ndarray:
    """(N,) True where a point lies inside a box, cylinder or sphere task object."""
    from ..frames import quat_to_matrix
    inside = np.zeros(len(P), bool)
    for o in project.workspace.objects:
        s = np.r_[np.abs(np.asarray(o.size, float).reshape(-1)), np.zeros(3)][:3]
        R = quat_to_matrix(o.pose.quaternion)
        L = (P - np.asarray(o.pose.position, float)) @ R          # object-frame coordinates
        if o.shape == 'box' and (s > 0).all():
            inside |= (np.abs(L) <= s / 2).all(axis=1)
        elif o.shape == 'cylinder' and s[0] > 0 and s[1] > 0:
            inside |= (np.hypot(L[:, 0], L[:, 1]) <= s[0]) & (np.abs(L[:, 2]) <= s[1] / 2)
        elif o.shape == 'sphere' and s[0] > 0:
            inside |= np.linalg.norm(L, axis=1) <= s[0]
    return inside


def _norm(s) -> str:
    return ' '.join(str(s or '').replace('_', ' ').replace('-', ' ').lower().split())


def sensor_mode(sensor, st: dict):
    """The stream mode (w, h) the sensor's imagers are modelled at: the study's 'modes' setting for it, else the
    stream chosen for the sensor (Sensor.stream, the Sensors page; the Camera view and the frusta use it too), else
    None (each imager's largest listed stream)."""
    m = (st.get('modes') or {}).get(sensor.uid)
    if not m:
        m = getattr(sensor, 'stream', None)
    if not m or len(m) != 2:
        return None
    return int(m[0]), int(m[1])


def pick_imager(sensor, imagers: list, st: dict):
    """(Imager, why) for a sensor's coverage camera (see the module docstring)."""
    from .. import library as LB
    want = (st.get('imagers') or {}).get(sensor.uid)
    if want:
        im = next((i for i in imagers if _norm(i.name) == _norm(want)), None)
        if im is None:
            raise StudyError(f'{sensor.name} has no imager named "{want}"; it has '
                             f'{", ".join(i.name for i in imagers)}.')
        return im, 'chosen in the settings'
    if len(imagers) == 1:
        return imagers[0], 'its only imager' if not imagers[0].is_default else 'the default imager (no device chosen)'
    entry = LB.get(sensor.model_id) if sensor.model_id else None
    ref = entry.reference_imager if entry is not None else None
    ref_im = next((i for i in imagers if ref is not None and i.name == ref.get('name')), imagers[0])
    is_depth = sensor.kind == 'depth' or (entry is not None and bool(entry.data.get('depth')))
    if is_depth and st.get('depth_imager', 'rgb') == 'rgb':
        rgb = next((i for i in imagers if i.role == 'rgb'), None)
        if rgb is not None:
            return rgb, ('its RGB imager (the colour image a policy is trained on; set depth_imager to "depth" for '
                         f'the reference imager, {ref_im.name})')
    if is_depth:
        return ref_im, 'its depth reference imager'
    return ref_im, 'its reference imager'


def _defocus(sensor, imager, st: dict) -> dict:
    """What the blur needs, from the entry and the user's figures: {'computed', 'why', 'f_mm', 'f_number',
    'focus_m', 'pixel_um' (at the stream), 'source': {figure: 'maker' | 'yours' | 'derived'}}."""
    from .. import library as LB
    entry = LB.get(sensor.model_id) if sensor.model_id else None
    im = entry.imager(imager.name) if entry is not None and not imager.is_default else None
    im = im or {}
    user = (st.get('optics') or {}).get(sensor.uid) or {}
    src, val = {}, {}
    for key, ek in (('f_number', 'f_number'), ('focus_m', 'focus_m'), ('pixel_um', 'pixel_um'),
                    ('focal_mm', 'focal_mm')):
        if user.get(key) is not None:
            val[key], src[key] = float(user[key]), 'yours'
        elif im.get(ek) is not None:
            val[key], src[key] = float(im[ek]), 'maker'
        else:
            val[key] = None
    cam = imager.camera
    native = im.get('resolution')
    pix = val['pixel_um']
    if pix is not None and src.get('pixel_um') == 'maker' and native and int(native[0]) != cam.width:
        pix = pix * float(native[0]) / cam.width          # ASSUMPTION: the stream bins or scales the full sensor
        src['pixel_um'] = 'derived'
    f_mm = val['focal_mm']
    if f_mm is None and pix is not None:
        f_mm = math.sqrt(cam.fx * cam.fy) * pix / 1000.0
        src['focal_mm'] = 'derived'
    out = {'f_mm': _num(f_mm), 'f_number': val['f_number'], 'focus_m': val['focus_m'], 'pixel_um': _num(pix),
           'source': src, 'computed': False, 'why': ''}
    missing = []
    if val['f_number'] is None:
        missing.append('the f-number')
    if val['focus_m'] is None:
        missing.append('the focus distance' + (' (the lens focuses automatically)' if im.get('focus') == 'auto'
                                               else ''))
    if pix is None and val['focal_mm'] is None:
        missing.append('the pixel pitch')
    elif pix is None:
        missing.append('the pixel pitch (to turn the blur into pixels)')
    if missing:
        who = 'the default imager (no device chosen) has no lens figures' if imager.is_default else \
            'the maker does not state ' + ' or '.join(missing)
        out['why'] = (f'Not computed: {who}. Enter measured figures for {sensor.name} in the coverage settings '
                      '(optics) to see the blur.')
        return out
    out['computed'] = True
    out['why'] = 'Computed from ' + ', '.join(f'{k.replace("_", " ")} ({v})' for k, v in sorted(src.items())) + '.'
    return out


# ====================================================================== the study
@jobs.study('coverage', 'Coverage and resolution', quick=False, batch='coverage', needs_robot=False)
def run(project, ctx: jobs.Ctx) -> dict:
    """Coverage and resolution of the task volumes by the rig's cameras (module docstring)."""
    from .. import library as LB
    from .. import occlusion as OC
    from .. import scenegeom as SG
    from .. import voxels as VX
    from ..optics import blur_circle_px, depth_of_field
    from ..robot.model import RobotModelError
    t_all = time.perf_counter()
    st = settings_of(project)
    notes: list = []
    ctx.report(0.0, 'Voxelising the task volumes')
    if not project.workspace.volumes:
        raise StudyError('There is no task volume to cover: add one (Workspace ▸ Add task volume).')
    sensors = [s for s in project.variant.sensors if s.enabled and s.kind in ('camera', 'depth')
               and (not st['sensors'] or s.uid in st['sensors'])]
    skipped = [s for s in project.variant.sensors if s.kind == 'tactile']
    if skipped:
        notes.append(f'{len(skipped)} tactile sensor{"s are" if len(skipped) != 1 else " is"} left out: tactile pads '
                     'are contact sensors, not cameras.')
    if not sensors:
        off = [x for x in project.variant.sensors if not x.enabled and x.kind in ('camera', 'depth')]
        raise StudyError('Coverage needs at least one camera or depth sensor in use: '
                         + (f'{off[0].name}{" and others are" if len(off) > 1 else " is"} switched off; tick "In use" '
                            'on the Sensors and mounts page, or add one (Sensors ▸ Add camera).' if off else
                            'add one (Sensors ▸ Add camera, or double-click a device in the Library).'))
    t0 = time.perf_counter()
    grid = VX.SparseGrid.from_project(project, st['voxel_mm'], st['refine_mm'])
    grid_s = time.perf_counter() - t0
    N = grid.count
    qd, qsrc = joint_state(project, st, notes)
    t0 = time.perf_counter()
    occ = SG.scene_occluders(project, qd)
    notes += list(occ.notes)
    occ_s = time.perf_counter() - t0
    P_all = grid.centers.astype(np.float64)
    inside = _inside_objects(project, P_all)
    w = grid.volume_m3.astype(np.float64)
    w_task = np.where(inside, 0.0, w)

    # ---------------------------------------------------------------- the cameras
    cams = []
    for s in sensors:
        try:
            ims = LB.sensor_imagers(s, sensor_mode(s, st))
        except LB.LibraryError as e:                   # one device that cannot be modelled: left out, said why
            notes.append(f'{s.name} is left out: {e}')
            continue
        if not ims:
            notes.append(f'{s.name} has no imager in its library entry and is left out.')
            continue
        if s.model_id and ims[0].is_default:                # a device id the library does not have (any more)
            notes.append(f'{s.name}: its device "{s.model_id}" is not in the library, so the default camera is used '
                         '(a default, not a device figure). Choose its device on the Sensors and mounts page.')
        im, why = pick_imager(s, ims, st)
        try:
            T_wb = SG.sensor_world_pose(project, s, qd).matrix()
        except (RobotModelError, KeyError) as e:
            notes.append(f'{s.name} could not be posed and is left out: {e}')
            continue
        T_wo = T_wb @ im.pose_in_body.matrix()
        cams.append({'sensor': s, 'imager': im, 'why': why, 'R': T_wo[:3, :3], 't': T_wo[:3, 3],
                     'defocus': _defocus(s, im, st)})
    if not cams:
        raise StudyError('None of the cameras could be placed: ' + ' '.join(notes[-3:]))
    C = len(cams)
    state = np.zeros((C, N), np.uint8)
    ppmm = np.zeros((C, N), np.float32)
    ang = np.full((C, N), np.nan, np.float32)
    blur = np.full((C, N), np.nan, np.float32)
    cube_s, vox_s, backend, renderer = 0.0, 0.0, '', ''
    n_chunks = max(1, math.ceil(N / st['chunk']))
    steps = C * (n_chunks + 1)
    step = 0
    cancelled = False
    coc_px = st['coc_px']
    for j, c in enumerate(cams):
        if ctx.cancelled():
            cancelled = True
            break
        s, im = c['sensor'], c['imager']
        cam = im.camera
        ctx.report(step / steps, f'{s.name}: occlusion map')
        t0 = time.perf_counter()
        occ_s_j = SG.sensor_occluders(project, s, qd, base=occ, origins=c['t'][None])
        try:
            cube = OC.CACHE.get(occ_s_j, c['t'], st['cube_n'], st['backend'])
        except OC.OcclusionError as e:
            raise StudyError(f'The occlusion map for {s.name} could not be built: {e}. Set the coverage backend to '
                             '"numpy" to use the software rasteriser.') from None
        cube_s += time.perf_counter() - t0
        backend, renderer = cube.backend, cube.renderer
        step += 1
        df = c['defocus']
        t0 = time.perf_counter()
        for ch in grid.chunks(st['chunk']):
            if ctx.cancelled():
                cancelled = True
                break
            P = ch.centers.astype(np.float64)
            Po = (P - c['t']) @ c['R']                       # optical frame: Rᵀ (P − t)
            _uv, ok = cam.project(Po)
            sl = slice(ch.start, ch.stop)
            idx = np.nonzero(ok)[0]
            if len(idx):
                seen = cube.visible(P[idx])
                stt = np.full(len(idx), STATE_HIDDEN, np.uint8)
                stt[seen] = STATE_SEEN
                state[j, sl][idx] = stt
                d = cam.pixels_per_mm(Po[idx])
                ppmm[j, sl][idx] = np.nan_to_num(d, nan=0.0)
                ang[j, sl][idx] = np.degrees(cam.angle_to_axis(Po[idx]))
                if df['computed']:
                    try:
                        b = blur_circle_px(df['f_mm'], df['f_number'], df['focus_m'], Po[idx, 2], df['pixel_um'])
                        blur[j, sl][idx] = np.asarray(b, float)
                    except ValueError as e:
                        df['computed'] = False
                        df['why'] = f'Not computed: {e}.'
            step_frac = (step + (ch.stop / max(N, 1))) / steps
            ctx.report(step_frac, f'{s.name}: {ch.stop:,} of {N:,} voxels')
        vox_s += time.perf_counter() - t0
        step += n_chunks
        if cancelled:
            break
        if ctx.partial:
            ctx.partial({'study': 'coverage', 'partial': True, 'cameras_done': j + 1, 'cameras': C,
                         'seen_fraction': {cams[k]['sensor'].name: _wfrac(state[k] == STATE_SEEN, w_task)
                                           for k in range(j + 1)}})

    # ---------------------------------------------------------------- the figures
    vis = state == STATE_SEEN
    red = vis.sum(axis=0).astype(np.uint8)
    res_ok = vis & (ppmm >= st['min_px_per_mm'])
    res_n = res_ok.sum(axis=0)
    best = np.where(vis, ppmm, 0.0).max(axis=0) if C else np.zeros(N)
    task = ~inside
    s_ge1, s_ge2 = _wfrac(red >= 1, w_task), _wfrac(red >= 2, w_task)
    summary = {
        'seen_ge1': _num(s_ge1), 'seen_ge2': _num(s_ge2), 'blind': _num(_wfrac(red == 0, w_task)),
        'seen_1_only': _num(_wfrac(red == 1, w_task)),
        'resolved_ge1': _num(_wfrac(res_n >= 1, w_task)), 'resolved_ge2': _num(_wfrac(res_n >= 2, w_task)),
        'best_ppmm_median': _num(wpercentile(best[task & (red >= 1)], w[task & (red >= 1)], 50), 4),
        'best_ppmm_p5': _num(wpercentile(best[task & (red >= 1)], w[task & (red >= 1)], 5), 4),
        'redundancy_mean': _num(float((red * w_task).sum() / max(w_task.sum(), 1e-300)), 4),
    }
    pct2 = 100.0 * (s_ge2 or 0.0)
    sev, word = status_for(pct2, st)
    summary.update({'status': sev, 'word': word, 'pct_ge2': _num(pct2, 3)})

    cam_docs = []
    for j, c in enumerate(cams):
        s, im = c['sensor'], c['imager']
        cam = im.camera
        m_seen = vis[j] & task
        m_in = (state[j] != STATE_OUT) & task
        df = c['defocus']
        sharp = None
        dof = (None, None)
        if df['computed'] and m_seen.any():
            b = blur[j][m_seen]
            sharp = _num(_wfrac(b <= coc_px, w[m_seen]))
            try:
                dof = depth_of_field(df['f_mm'], df['f_number'], df['focus_m'], coc_px * df['pixel_um'] / 1000.0)
            except ValueError:
                dof = (None, None)
        fov = cam.fov_deg()
        cam_docs.append({
            'sensor_uid': s.uid, 'name': s.name, 'kind': s.kind, 'slot': s.slot, 'model_id': s.model_id,
            'imager': im.name, 'imager_role': im.role, 'imager_why': c['why'], 'imager_label': im.label,
            'is_default': im.is_default, 'pose_stated': im.pose_stated, 'calibrated': im.calibrated,
            'mode': [cam.width, cam.height], 'lens_model': cam.model,
            'fov_deg': {k: _num(v, 3) for k, v in fov.items()},
            'position_m': [_num(v) for v in c['t']], 'axis': [_num(v) for v in c['R'][:, 2]],
            'on_link': bool(s.mount.on_link),
            'seen': _num(_wfrac(m_seen, w_task)), 'in_image': _num(_wfrac(m_in, w_task)),
            'outside_image': _num(_wfrac((state[j] == STATE_OUT) & task, w_task)),
            'hidden': _num(_wfrac((state[j] == STATE_HIDDEN) & task, w_task)),
            'resolved': _num(_wfrac(res_ok[j] & task, w_task)),
            'only_camera': _num(_wfrac(m_seen & (red == 1), w_task)),
            'ppmm_median': _num(wpercentile(ppmm[j][m_seen], w[m_seen], 50), 4),
            'ppmm_p5': _num(wpercentile(ppmm[j][m_seen], w[m_seen], 5), 4),
            'ppmm_p95': _num(wpercentile(ppmm[j][m_seen], w[m_seen], 95), 4),
            'angle_median_deg': _num(wpercentile(ang[j][m_seen], w[m_seen], 50), 3),
            'angle_max_deg': _num(float(np.nanmax(ang[j][m_seen])) if m_seen.any() else None, 3),
            'range_median_m': _num(wpercentile(np.linalg.norm(P_all[m_seen] - c['t'], axis=1), w[m_seen], 50), 4),
            'defocus': {**df, 'coc_px': coc_px, 'sharp': sharp, 'dof_near_m': _num(dof[0]),
                        'dof_far_m': _num(dof[1]),
                        'blur_median_px': _num(wpercentile(blur[j][m_seen], w[m_seen], 50), 3)
                        if df['computed'] else None},
        })

    vol_docs = []
    for i, v in enumerate(grid.volumes):
        mv = (grid.volume_index == i)
        wv = np.where(mv, w_task, 0.0)
        vol_docs.append({'uid': v.uid, 'name': v.name, 'voxels': int(mv.sum()), 'inside_objects':
                         int((mv & inside).sum()), 'volume_m3': _num(float(np.prod(v.size)), 9),
                         'seen_ge1': _num(_wfrac(red >= 1, wv)), 'seen_ge2': _num(_wfrac(red >= 2, wv)),
                         'blind': _num(_wfrac(red == 0, wv)),
                         'per_camera': [_num(_wfrac(vis[j], wv)) for j in range(C)]})
    summary['advice'] = advice(summary, cam_docs, st)

    # ---------------------------------------------------------------- compact per-voxel arrays
    voxels = _pack_voxels(grid, inside, red, best, state, ppmm, ang, blur, [c['defocus']['computed'] for c in cams])
    occ_info = {'triangles': int(len(occ)), 'owners': len(occ.owners), 'backend': backend, 'renderer': renderer,
                'cube_n': st['cube_n'], 'notes': list(occ.notes)}
    st_out = {k: v for k, v in st.items() if k not in ('q',)}
    st_out['voxel_mm'] = grid.voxel_m * 1000
    st_out['refine_mm'] = grid.refine_m * 1000
    return {
        'project': project.name, 'variant': project.variant.name,
        'q': {uid: [float(x) for x in v] for uid, v in qd.items()}, 'q_source': qsrc,
        'robots': {r.uid: r.name for r in project.robots},
        'settings': st_out,
        'grid': {'count': N, 'voxel_mm': _num(grid.voxel_m * 1000), 'refine_mm': _num(grid.refine_m * 1000),
                 'refined': int(grid.refined.sum()), 'inside_objects': int(inside.sum()),
                 'volume_m3': _num(float(w.sum()), 9), 'task_volume_m3': _num(float(w_task.sum()), 9),
                 'message': grid.info.get('message', '')},
        'summary': summary, 'cameras': cam_docs, 'volumes': vol_docs, 'occluders': occ_info,
        'voxels': voxels, 'notes': notes, 'partial': bool(cancelled),
        'timings': {'grid_seconds': round(grid_s, 4), 'occluders_seconds': round(occ_s, 4),
                    'cube_seconds': round(cube_s, 4), 'voxel_seconds': round(vox_s, 4),
                    'total_seconds': round(time.perf_counter() - t_all, 4)},
    }


def _pack_voxels(grid, inside, red, best, state, ppmm, ang, blur, blur_ok) -> dict:
    from .. import voxels as VX
    h, hr = grid.voxel_m, grid.refine_m
    unit = hr / 2.0
    origins = []
    for v in grid.volumes:
        d = VX.dims(v.size, h)
        origins.append((np.asarray(v.center, float) - d * h / 2).tolist())
    org = np.asarray(origins, float).reshape(-1, 3)
    if grid.count:
        ijk = np.rint((grid.centers.astype(np.float64) - org[grid.volume_index]) / unit)
    else:
        ijk = np.zeros((0, 3))
    dt = np.uint16 if ijk.size == 0 or ijk.max() < 65535 else np.uint32
    vt = np.uint8 if len(grid.volumes) < 256 else np.uint16
    q_ppmm = lambda a: np.clip(np.rint(np.nan_to_num(a, nan=0.0) / PPMM_STEP), 0, 65535).astype(np.uint16)  # noqa
    cams = []
    for j in range(len(state)):
        a = np.where(np.isfinite(ang[j]), np.clip(np.rint(ang[j] / ANGLE_STEP), 0, 254), 255).astype(np.uint8)
        b = None
        if blur_ok[j]:
            bb = blur[j]
            b = encode_array(np.where(np.isfinite(bb), np.clip(np.rint(bb / BLUR_STEP), 0, 254), 255).astype(np.uint8))
        cams.append({'state': encode_array(state[j]), 'ppmm': encode_array(q_ppmm(ppmm[j])), 'angle': encode_array(a),
                     'blur': b})
    return {'count': grid.count, 'encoding': 'numpy bytes, little-endian, zlib, base64 (coverage.decode)',
            'unit_m': unit, 'voxel_m': h, 'refine_m': hr, 'origins': origins,
            'ijk': encode_array(ijk.astype(dt)), 'volume_index': encode_array(grid.volume_index.astype(vt)),
            'refined': encode_array(grid.refined.astype(np.uint8)), 'inside': encode_array(inside.astype(np.uint8)),
            'redundancy': encode_array(red.astype(np.uint8)), 'best_ppmm': encode_array(q_ppmm(best)),
            'steps': {'ppmm': PPMM_STEP, 'angle_deg': ANGLE_STEP, 'blur_px': BLUR_STEP},
            'states': {'outside_image': STATE_OUT, 'hidden': STATE_HIDDEN, 'seen': STATE_SEEN},
            'cameras': cams}


def _pct(x) -> str:
    return '—' if x is None else f'{100 * x:.0f} %'


def advice(summary: dict, cams: list, st: dict) -> str:
    """One or two plain sentences: what the coverage means and the change that would help most."""
    s2, s1 = summary.get('seen_ge2') or 0.0, summary.get('seen_ge1') or 0.0
    if not cams:
        return 'No camera could be placed.'
    if 100 * s2 >= st['good_pct']:
        lo = min((c for c in cams if c['ppmm_p5'] is not None), key=lambda c: c['ppmm_p5'], default=None)
        tail = (f' The coarsest view is {lo["name"]}: 5 % of what it sees is below {lo["ppmm_p5"]:.2f} px/mm.'
                if lo is not None else '')
        return f'{_pct(s2)} of the task volume is seen by two or more cameras.' + tail
    worst = min(cams, key=lambda c: c['seen'] or 0.0)
    out, hid = worst['outside_image'] or 0.0, worst['hidden'] or 0.0
    text = (f'{_pct(s2)} of the task volume is seen by two or more cameras, {_pct(s1 - s2)} by one only and '
            f'{_pct(1 - s1)} by none. {worst["name"]} sees {_pct(worst["seen"])}: ')
    if len(cams) == 1:
        text += 'it is the only camera, so add a second camera that sees the task volume from another side.'
    elif out >= hid:
        text += (f'{_pct(out)} of the volume is outside its image, so aim it at the volume, move it back or choose a '
                 'wider lens.')
    else:
        text += (f'{_pct(hid)} is in its image but hidden by the scene at this pose, so move it to look past the '
                 'arm or the objects, or check the coverage at the other poses of the task.')
    return text
