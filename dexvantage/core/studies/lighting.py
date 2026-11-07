"""Lighting, glare and IR wash-out (spec §7.8): how much of what each camera sees of the object and the gripper is
blown out by specular highlights, how much valid depth an active-stereo sensor loses to ambient IR and to dark or
clear surfaces, over the gripper's working poses, and what to change.

    @jobs.study('lighting', 'Lighting and glare', batch='lighting')         → --lighting OUT.json
    run(project, ctx) -> dict                the study
    settings_of(project) -> dict             the settings with the labelled defaults filled in
    status_for(pct, st) -> (severity, word)  the headline's status for a saturated share (%)
    Evaluator(project, st, notes)            the machinery (for the page's previews, the checks and the tests)

Inputs. The lights (project.workspace.lights, core/lights.py: their stated luminance, luminous flux and IR irradiance,
each with its source; a light whose figures are not stated is named and left out), the materials (core/materials.py:
the task objects' tags; the robot's links take the setting robot_material, 'matte' by default, a DEFAULT, or
link_materials {robot uid: {link: tag}}; a surface its 'material'), and the rig's enabled cameras and depth sensors.

Working poses. With a grasp set in the project (settings poses 'auto' or 'grasps'), up to max_poses grasps (default 6)
evenly picked from the set (settings grasp_set, default the first) solved by the grasp study's inverse kinematics
(core/studies/grasp_visibility.solve_grasps, its settings and checks); the feasible solutions are the poses. Without
one (or poses 'home'), the joint state of the settings q, else home. Other robots stay at that joint state.

What is sampled. Points drawn uniformly by area (seeded) on each task object (object_points, default 400 each), on the
gripper's links (the links of the gripper frame's subtree: gripper_points, default 1200, carried with the links to each
pose) and on the surfaces (surface_points, default 500 each). Each point stands for its share of its part's area.

Visibility (core/occlusion). A point is seen by a camera when it projects inside the imager's image (its lens model at
its stream) and the camera's cube map of the scene posed at that pose (scenegeom.sensor_occluders: the camera's seat
left out; cube_n 512 texels) has nothing nearer. A point is lit by a light when the cube map at the light's centre has
nothing nearer, and by a window's sun when its beam passes the window's rectangle and core/raycast finds no triangle in
the way. A point on a closed part (an object, the gripper) whose outward normal faces away from the viewer is on the
part's far side: hidden (surfaces are one-sided sheets: either side shows). The others are lifted lift_mm (default
1 mm, a DEFAULT) along the normal turned towards the viewer and compared with the map with a slope-scaled tolerance,
eps + ¾ texel × distance × tan θ (θ from the normal, capped at 80°): an oblique surface changes depth across a texel,
and without it it shadows itself (shadow-map acne, Williams 1978). The check lighting.visibility_vs_raycast measures
the agreement with exact ray casting (0.66 % of the area on the UR5e sample with a can and two lights).

Radiance. Towards the camera, L = Σ_lights Σ_emitters f_r(n, ωi, ωo) E⊥ max(0, n · ωi) (the BRDF of core/materials.py;
an area light split into emitter patches, core/lights.py), in cd/m² (visible band, photometric).

Saturation. Two models; the result names the one used per camera:
- absolute, when the imager's sensitivity S (digital numbers per lux second at the sensor, unity gain) and its
  f-number N are stated (the library entry, or the settings sensors[uid]: sensitivity_dn_per_lux_s, bits, f_number)
  and an exposure time t is given (settings sensors[uid].exposure_s, or the entry): the image-plane illuminance
  E = (π / 4) T L cos⁴α / N² (the camera equation; T the lens transmittance, 1 by default: an ASSUMPTION, α the
  ray's angle from the optical axis), DN = S · g · t · E with the gain g (settings gain, default 1: unity), saturated
  where DN ≥ 2^bits − 1 (bits default 8). No library entry states S today, so this model runs only on your figures.
- relative (the default when any of those is missing; labelled "relative" everywhere): an auto-exposure that meters
  the visible scene sample's mean luminance L̄ as 18 % grey; ISO 12232's saturation-based speed rating puts that grey
  at 18 % / √2 = 12.7 % of saturation (half a stop of headroom for highlights), so a point saturates when
  L ≥ L̄ / 0.127 = 7.86 L̄ (sat_ratio, a DEFAULT from that definition). The glare index is L / L̄: a relative figure, not
  an absolute one.
The headline is the worst camera's saturated share of the visible object and gripper area (area-weighted), averaged
over the working poses; the worst pose is named. Surfaces (the table) are reported beside it, not in it.

Glare maps. Per camera, the image is divided into grid cells (default 32 × 24); each cell keeps the largest saturated
share of the visible area in it over the poses, and the largest glare index (relative) or DN / full scale (absolute).
The page draws them over the Camera view.

IR wash-out (active IR stereo and structured light). For each such depth sensor, at each pose, the points its reference
imager sees inside the projector's cone (its stated FOV) get the pattern's irradiance E_p = (P / Ω) cos θ / d², P the
projector's stated optical power (library projector.optical_power_w: Intel's D400 datasheet states 360 mW average for
the D415, D435, D435i and D455; null elsewhere: then wash-out is not computed and the result says so) spread uniformly
over its rectangular field Ω = 4 arcsin(sin(a/2) sin(b/2)), and the ambient IR E_a from the lights' and the sun's IR
figures (shadowed as above). The pattern's contrast against the ambient light, C = E_p / (E_p + E_a), is the same for
any Lambertian albedo; a point is washed out when C < contrast_min (default 0.5: ambient as bright as the pattern, a
DEFAULT). A point on a surface whose IR albedo is below ir_albedo_min (default 0.1, a DEFAULT) returns too little of
the pattern (dark or IR-absorbing); a point on a transmissive object the depth reads through (materials flag
depth_reads_through, per object in the settings 'materials') gets the depth of what is behind it. The predicted loss
of valid depth is the share of the visible object and gripper area with any of the three.

Recommendations, each with the predicted figure before and after (the same evaluation re-run with the change):
- a linear polariser on the worst camera (core/materials.polariser_factors: an ideal polariser passes half the diffuse
  light and k min(R_s, R_p) / (R_s + R_p) of the specular, turned at its best for every point, so the gain is an upper
  bound; in the relative model the auto-exposure makes up the lost half);
- moving the light that lights most saturated points: ±move_m (default 0.3 m) along world x and y and up, and dimming
  it to dim_to (default 50 %) (in the relative model dimming changes the result only through the other lights);
- tilting the worst camera ±tilt_deg (default 5° and 10°) about its optical x axis;
- for wash-out, turning the IR source that adds most ambient IR off or blinding it (dim 0).
The suggestion is the one that lowers the headline most (or the depth loss, when no camera saturates). Apply makes it
(one undo step): the polariser is a setting of this study, a light move or dim changes the light, a tilt the
camera's mount.

Settings (project.analysis.studies['lighting'], all optional): poses ('auto' | 'grasps' | 'home'), grasp_set,
max_poses, q, object_points, gripper_points, surface_points, rng_seed, lift_mm, cube_n (512), backend, n_area (4),
sat_ratio, lens_transmittance, contrast_min, ir_albedo_min, robot_material, link_materials, materials {object uid:
{'alpha', 'albedo', 'ir_albedo', 'depth_reads_through'}}, sensors {uid: {'exposure_s', 'gain',
'sensitivity_dn_per_lux_s', 'bits', 'f_number', 'polariser', 'polariser_k'}}, grid ([32, 24]), recommend (true),
move_m, dim_to, tilt_deg, good_pct (1 %), warn_pct (5 %), serious_pct (15 %) (glare status thresholds: defaults),
depth_good_pct (5 %), depth_warn_pct (15 %), depth_serious_pct (30 %) (depth-loss thresholds: defaults; when the
depth loss is the worse, it sets the headline's status and its word says both, e.g. 'low glare; depth washed out').

Validation (dexvantage/validation/checks_lighting.py): the BRDF's white furnace (quadrature) and energy (Monte Carlo),
reciprocity, the specular peak against its closed form at normal incidence, Schlick against the exact Fresnel
equations, the point-light inverse-square law against Monte Carlo over an area light, the split area light against
the closed-form irradiance of a rectangle, and the study's visibility against exact ray casting.

Known limits: direct light only (no inter-reflections, no ambient term: shadowed parts are darker than in a room, and
the relative model's metering is of the sampled object, gripper and surfaces, not the whole image); hard shadows from
each light's centre; the camera's spectral response is taken as photometric (V(λ)); the IR figures are taken as being
for the band the depth sensor's imagers see (its filter is not stated by the makers); the projector's field is uniform
over its stated FOV (the real dot pattern is not), its own shadowing is not modelled, and the D4xx lasers halve
their power above 60 °C (Intel), which is not modelled; passive stereo still matches on textured surfaces when the
pattern is washed out, so the loss is for textureless surfaces; the materials' roughness and albedos are defaults
(core/materials.py), metals are aluminium, and transmission is a flag; the polariser's gain is an upper bound; a
point is sampled, not a pixel (a highlight smaller than the sample spacing can be missed: raise the point counts);
visibility is per cube-map texel with a slope tolerance (points within about a texel of a silhouette or a contact can
be judged either way, and a point just behind an oblique edge can leak through);
the robot's link materials are not in its model file, so they are a setting.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np

from .. import jobs
from ..jobs import StudyError

SID = 'lighting'
ISO_GREY = 0.18 / math.sqrt(2.0)        # ISO 12232 saturation-based speed: 18 % grey at 12.7 % of saturation
DEFAULTS = {
    'poses': 'auto',
    'grasp_set': '',
    'max_poses': 6,
    'q': {},
    'object_points': 400,
    'gripper_points': 1200,
    'surface_points': 500,
    'rng_seed': 0,
    'lift_mm': 1.0,                    # default: a point is tested this far in front of its surface
    'cube_n': 512,                     # as the grasp study (a texel spans 3.9 mrad at a face centre)
    'backend': 'auto',
    'n_area': 4,
    'sat_ratio': 1.0 / ISO_GREY,       # default (relative model): saturation at 7.86 x the metered mean luminance
    'lens_transmittance': 1.0,         # ASSUMPTION (absolute model): no maker states it
    'contrast_min': 0.5,               # default: the pattern is washed out when ambient IR is as bright as it
    'ir_albedo_min': 0.1,              # default: surfaces returning less of the pattern than this give no depth
    'robot_material': 'matte',         # default: robot link material (model files carry no material tags)
    'link_materials': {},
    'materials': {},
    'sensors': {},
    'grid': [32, 24],
    'recommend': True,
    'move_m': 0.3,
    'dim_to': 0.5,
    'tilt_deg': [5.0, 10.0],
    'good_pct': 1.0,
    'warn_pct': 5.0,
    'serious_pct': 15.0,
    'depth_good_pct': 5.0,             # default: depth lost to wash-out, dark or clear surfaces, status thresholds
    'depth_warn_pct': 15.0,
    'depth_serious_pct': 30.0,
}
WORDS = {'good': 'low glare', 'warning': 'some glare', 'serious': 'glare', 'critical': 'heavy glare'}
DEPTH_WORDS = {'warning': 'some depth lost', 'serious': 'depth lost', 'critical': 'depth washed out'}
SEV_ORDER = {'good': 0, 'warning': 1, 'serious': 2, 'critical': 3}
ACTIVE = ('active_ir_stereo', 'structured_light')
MAX_OVERLAY_POINTS = 800


# ====================================================================== settings
def settings_of(project) -> dict:
    from ..project import MATERIAL_TAGS
    s = dict(project.analysis.studies.get(SID) or {})
    st = copy.deepcopy(DEFAULTS)
    st.update({k: v for k, v in s.items() if v is not None})
    st['defaults_used'] = sorted(k for k in DEFAULTS if k not in s)

    def num(k, lo, hi, integer=False):
        try:
            v = float(st[k])
        except (TypeError, ValueError):
            v = float(DEFAULTS[k])
        if not math.isfinite(v):
            v = float(DEFAULTS[k])
        v = min(max(v, lo), hi)
        st[k] = int(round(v)) if integer else v
    for k, lo, hi in (('max_poses', 1, 64), ('object_points', 10, 20000), ('gripper_points', 0, 50000),
                      ('surface_points', 0, 20000), ('cube_n', 32, 2048), ('n_area', 1, 16), ('rng_seed', 0, 2 ** 31)):
        num(k, lo, hi, True)
    for k, lo, hi in (('lift_mm', 0.0, 50.0), ('sat_ratio', 1.0, 1e6), ('lens_transmittance', 0.01, 1.0),
                      ('contrast_min', 0.0, 1.0), ('ir_albedo_min', 0.0, 1.0), ('move_m', 0.0, 5.0),
                      ('dim_to', 0.0, 1.0), ('good_pct', 0.0, 100.0), ('warn_pct', 0.0, 100.0),
                      ('serious_pct', 0.0, 100.0), ('depth_good_pct', 0.0, 100.0), ('depth_warn_pct', 0.0, 100.0),
                      ('depth_serious_pct', 0.0, 100.0)):
        num(k, lo, hi)
    if st['poses'] not in ('auto', 'grasps', 'home'):
        st['poses'] = 'auto'
    if st['robot_material'] not in MATERIAL_TAGS:
        st['robot_material'] = 'matte'
    g = st.get('grid')
    try:
        st['grid'] = [int(min(max(int(g[0]), 2), 256)), int(min(max(int(g[1]), 2), 256))]
    except (TypeError, ValueError, IndexError):
        st['grid'] = list(DEFAULTS['grid'])
    t = st.get('tilt_deg')
    t = t if isinstance(t, (list, tuple)) else [t]
    st['tilt_deg'] = [float(x) for x in t if isinstance(x, (int, float)) and math.isfinite(x) and 0 < abs(x) <= 45]
    for k in ('sensors', 'materials', 'link_materials', 'q'):
        if not isinstance(st.get(k), dict):
            st[k] = {}
    return st


def status_for(pct: float, st: dict) -> tuple[str, str]:
    if pct is None:
        return 'warning', 'not computed'
    if pct < st['good_pct']:
        sev = 'good'
    elif pct < st['warn_pct']:
        sev = 'warning'
    elif pct < st['serious_pct']:
        sev = 'serious'
    else:
        sev = 'critical'
    return sev, WORDS[sev]


def depth_status_for(pct: float, st: dict) -> str:
    """The severity of a depth loss (share of what an active depth sensor sees of the object and gripper)."""
    if pct < st['depth_good_pct']:
        return 'good'
    if pct < st['depth_warn_pct']:
        return 'warning'
    return 'serious' if pct < st['depth_serious_pct'] else 'critical'


def _num(x, nd=6):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def _rx(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _wsum(w, m) -> float:
    return float(np.sum(w[m])) if np.any(m) else 0.0


# ====================================================================== the cameras
class _Cam:
    """A camera's imager and its exposure model."""

    def __init__(self, project, sensor, st, notes):
        from .. import library as LB
        from . import coverage as CV
        self.sensor, self.uid, self.name = sensor, sensor.uid, sensor.name
        try:
            ims = LB.sensor_imagers(sensor, CV.sensor_mode(sensor, {}))
        except LB.LibraryError as e:
            raise StudyError(f'{sensor.name}: {e}') from None
        if not ims:
            raise StudyError(f'{sensor.name} has no imager to look through.')
        self.imager, self.why = CV.pick_imager(sensor, ims, {'depth_imager': 'rgb'})
        self.cam = self.imager.camera
        cfg = dict((st.get('sensors') or {}).get(sensor.uid) or {})
        self.cfg = cfg
        entry = LB.get(sensor.model_id) if sensor.model_id else None
        im = entry.imager(self.imager.name) if entry is not None else None
        im = im or {}

        def stated(key):
            v = im.get(key)
            if isinstance(v, dict):
                return v.get('value'), v.get('source') or 'the library entry'
            return (v, 'the library entry') if isinstance(v, (int, float)) else (None, '')
        self.f_number, fsrc = (cfg['f_number'], 'your setting') if cfg.get('f_number') else stated('f_number')
        sens, ssrc = ((cfg['sensitivity_dn_per_lux_s'], 'your setting') if cfg.get('sensitivity_dn_per_lux_s')
                      else stated('sensitivity'))
        sens_entry = im.get('sensitivity') if isinstance(im.get('sensitivity'), dict) else {}
        bits_v = cfg.get('bits') or sens_entry.get('bits')
        self.bits = int(bits_v) if bits_v else 8
        self.bits_default = not bits_v
        exp, esrc = (cfg['exposure_s'], 'your setting') if cfg.get('exposure_s') else stated('exposure_s')
        self.gain = float(cfg.get('gain') or 1.0)
        self.gain_default = not cfg.get('gain')
        self.sens, self.exposure = sens, exp
        missing = [w for w, v in (('the sensor\'s sensitivity (DN per lux second)', sens),
                                  ('the lens\'s f-number', self.f_number), ('the exposure time', exp)) if not v]
        self.absolute = not missing
        if self.absolute:
            self.mode_text = (f'absolute: sensitivity {float(sens):g} DN/(lx·s) ({ssrc}), f/{float(self.f_number):g} '
                              f'({fsrc}), exposure {float(exp) * 1000:g} ms ({esrc}), gain {self.gain:g}'
                              + (' (unity, a default)' if self.gain_default else '') + f', {self.bits} bits'
                              + (' (a default)' if self.bits_default else ''))
        else:
            self.mode_text = ('relative: ' + ', '.join(missing) + (' is' if len(missing) == 1 else ' are')
                              + ' not stated, so saturation is predicted for an auto-exposure that meters the scene '
                              'as 18 % grey (ISO 12232), not from the sensor\'s own figures')
        self.polariser = bool(cfg.get('polariser'))
        self.polariser_k = float(cfg.get('polariser_k') or 1.0)
        self.on_link = sensor.mount.on_link

    def optical(self, project, qd):
        from ..camrender import optical_pose
        from ..frames import quat_to_matrix
        T = optical_pose(project, self.sensor, self.imager, qd)
        return quat_to_matrix(T.quaternion), np.asarray(T.position, float)

    def mount_after_tilt(self, deg: float):
        """The sensor's mount pose after turning it deg about its imager's optical x axis (at the optical centre)."""
        from ..frames import Pose
        T_po = self.sensor.mount.pose.matrix() @ self.imager.pose_in_body.matrix()
        T2 = T_po.copy()
        T2[:3, :3] = T_po[:3, :3] @ _rx(deg)
        return Pose.from_matrix(T2 @ np.linalg.inv(self.imager.pose_in_body.matrix()))


class _Depth:
    """An active depth sensor's reference imager and projector."""

    def __init__(self, project, sensor, st, notes):
        from . import depth_validity as DV
        self.sensor, self.uid, self.name = sensor, sensor.uid, sensor.name
        self.ds = DV.setup(project, sensor)
        self.tech = self.ds.technology
        pj = self.ds.projector or {}
        self.pj = pj if pj else None
        self.power = pj.get('optical_power_w') if pj else None
        a, b = pj.get('hfov_deg'), pj.get('vfov_deg')
        self.fov = (a, b) if a and b else None
        self.omega = 4 * math.asin(math.sin(math.radians(a) / 2) * math.sin(math.radians(b) / 2)) if self.fov else None
        self.power_source = pj.get('optical_power_what') or ('the library entry' if self.power else '')


# ====================================================================== the evaluator
class Evaluator:
    """Everything the study computes, re-runnable with a change (a light moved, a polariser, a camera tilted)."""

    def __init__(self, project, st: dict, notes: list, ctx=None):
        from .. import lights as LT
        from .. import materials as MT
        self.project, self.st, self.notes, self.ctx = project, st, notes, ctx
        self.lights = [x for x in LT.lights_of(project)]
        self.MT = MT
        self.mats: list = []
        self._mat_key: dict = {}
        self.cams: list = []
        self.depths: list = []
        self.skipped: list = []
        for s in project.variant.sensors:
            if not s.enabled or s.kind not in ('camera', 'depth'):
                continue
            try:
                self.cams.append(_Cam(project, s, st, notes))
            except StudyError as e:
                self.skipped.append({'uid': s.uid, 'name': s.name, 'why': str(e)})
            from .. import library as LB
            from . import depth_validity as DV
            if LB.depth_spec(s) is None:
                continue
            try:
                d = _Depth(project, s, st, notes)
            except DV.DepthSetupError as e:
                self.skipped.append({'uid': s.uid, 'name': s.name, 'why': f'IR wash-out: {e}'})
                continue
            if d.tech in ACTIVE:
                self.depths.append(d)
            else:
                notes.append(f'{s.name}: no IR wash-out figure ({DV.TECH_TITLE.get(d.tech, d.tech)}: '
                             + ('no projector pattern to wash out).' if d.tech == 'passive_stereo' else
                                'its illuminator is not in the library).'))
        self.poses, self.pose_source = self._poses()
        self._sample()
        self._pose_cache: dict = {}
        self._shadow_cache: dict = {}
        self._em_cache: dict = {}
        self._vis_cache: dict = {}

    # ---------------------------------------------------------------- materials
    def _mat(self, tags, overrides=None) -> int:
        key = (tuple(sorted(tags)), repr(sorted((overrides or {}).items())))
        if key not in self._mat_key:
            self._mat_key[key] = len(self.mats)
            self.mats.append(self.MT.material_for(list(tags), overrides))
        return self._mat_key[key]

    # ---------------------------------------------------------------- poses
    def _poses(self):
        from . import coverage as CV
        st, p = self.st, self.project
        qd0, _src = CV.joint_state(p, st, self.notes)
        if st['poses'] == 'home' or not p.tasks.grasp_sets:
            if st['poses'] == 'grasps':
                self.notes.append('There is no grasp set, so the study runs at one pose: '
                                  + ('the joint state given' if st.get('q') else 'home') + '.')
            return [('the joint state given' if st.get('q') else 'home', qd0)], ('settings' if st.get('q')
                                                                                 else 'home')
        from .. import grasps as GR
        from . import grasp_visibility as GV
        gs = GR.get_set(p, st['grasp_set']) if st['grasp_set'] else GR.sets(p)[0]
        if gs is None:
            raise StudyError('The grasp set chosen in the lighting settings is not in the project any more: choose '
                             'another.')
        robot = p.robot(gs.get('robot_uid', ''))
        poses = list(gs.get('poses') or [])
        if robot is None or robot.uid not in qd0 or not poses:
            self.notes.append(f'The grasp set "{gs.get("name")}" cannot be posed (no robot or no grasps): the study '
                              'runs at home.')
            return [('home', qd0)], 'home'
        k = min(len(poses), st['max_poses'] * 3)
        sel = np.unique(np.linspace(0, len(poses) - 1, k).round().astype(int))
        gsel = dict(gs, poses=[poses[i] for i in sel])
        from ..robot.model import get_model
        model = get_model(p, robot.uid)
        try:
            link, tcp, targets = GR.ik_targets(p, gsel)
        except GR.GraspError as e:
            raise StudyError(str(e)) from None
        gst = GV.settings_of(p)
        b = GV.solve_grasps(p, robot, model, link, targets, tcp, gst, np.asarray(qd0[robot.uid], float), None,
                            self.notes)
        ok = np.flatnonzero(np.asarray(b.ok, bool))
        if not len(ok):
            self.notes.append(f'None of the {len(sel)} grasps tried from "{gs.get("name")}" is reachable: the study '
                              'runs at home.')
            return [('home', qd0)], 'home'
        pick = ok[np.unique(np.linspace(0, len(ok) - 1, min(st['max_poses'], len(ok))).round().astype(int))]
        out = []
        for i in pick:
            qd = {u: np.asarray(v, float).copy() for u, v in qd0.items()}
            qd[robot.uid] = np.asarray(b.q[i], float)
            out.append((f'grasp {int(sel[i]) + 1}', qd))
        self.grasp_set = gs.get('name', '')
        self.notes.append(f'Working poses: {len(out)} reachable grasps of "{gs.get("name")}" (of {len(sel)} tried, '
                          'evenly picked from the set).')
        return out, 'grasps'

    # ---------------------------------------------------------------- sample points
    def _sample(self):
        from .. import grasps as GR
        from ..scenegeom import object_triangles, surface_triangles
        st, p = self.st, self.project
        rng = np.random.default_rng([st['rng_seed'], 7])
        P, N, W, G, M, OWN = [], [], [], [], [], []
        self.parts = []                                    # [(owner text, name)]
        for o in p.workspace.objects:
            tri = np.asarray(object_triangles(o, p), float)
            if not len(tri):
                continue
            x, k = GR.sample_triangles(tri, st['object_points'], rng)
            n = _tri_normals(tri)[k]
            area = float(_tri_area(tri).sum())
            mi = self._mat(o.materials or ['matte'], (st['materials'] or {}).get(o.uid))
            P.append(x)
            N.append(n)
            W.append(np.full(len(x), area / len(x)))
            G.append(np.zeros(len(x), np.int8))
            M.append(np.full(len(x), mi, np.int32))
            OWN.append(np.full(len(x), len(self.parts), np.int32))
            self.parts.append((f'object:{o.uid}', o.name))
        if st['surface_points']:
            from ..project import MATERIAL_TAGS
            for j, s in enumerate(p.workspace.surfaces):
                if not isinstance(s, dict):
                    continue
                tri = np.asarray(surface_triangles(s), float)
                if not len(tri):
                    continue
                x, k = GR.sample_triangles(tri, st['surface_points'], rng)
                n = _tri_normals(tri)[k]
                tag = s.get('material') if s.get('material') in MATERIAL_TAGS else 'matte'
                mi = self._mat([tag], (st['materials'] or {}).get(str(s.get('uid') or j)))
                P.append(x)
                N.append(n)
                W.append(np.full(len(x), float(_tri_area(tri).sum()) / len(x)))
                G.append(np.full(len(x), 2, np.int8))
                M.append(np.full(len(x), mi, np.int32))
                OWN.append(np.full(len(x), len(self.parts), np.int32))
                self.parts.append((f'surface:{s.get("uid") or j}', str(s.get('name') or 'Surface')))
        self.fixed = (np.concatenate(P) if P else np.zeros((0, 3)), np.concatenate(N) if N else np.zeros((0, 3)),
                      np.concatenate(W) if W else np.zeros(0), np.concatenate(G) if G else np.zeros(0, np.int8),
                      np.concatenate(M) if M else np.zeros(0, np.int32),
                      np.concatenate(OWN) if OWN else np.zeros(0, np.int32))
        # the gripper: points in its links' frames, carried to each pose
        self.grip = None
        robot = None
        if self.pose_source == 'grasps':
            gs = (GR.get_set(p, st['grasp_set']) if st['grasp_set'] else GR.sets(p)[0]) or {}
            robot = p.robot(gs.get('robot_uid', ''))
        robot = robot or p.primary_robot
        if robot is None or not st['gripper_points']:
            return
        try:
            from ..robot.model import get_model
            model = get_model(p, robot.uid)
            frame = GR.gripper_frame(robot, model)
        except Exception as e:                           # noqa: BLE001 — no gripper to sample: said
            self.notes.append(f'The gripper is not sampled: {e}')
            return
        links = _subtree_links(model, frame)
        tl = model.triangle_links()
        sel = np.flatnonzero(np.isin(tl, links))
        if not len(sel):
            self.notes.append(f'The gripper ({frame} and the links below it) has no visual mesh to sample.')
            return
        tri = model.local_triangles()[sel]
        x, k = GR.sample_triangles(tri, st['gripper_points'], rng)
        n = _tri_normals(tri)[k]
        li = tl[sel][k]
        lm = (st.get('link_materials') or {}).get(robot.uid) or {}
        from ..project import MATERIAL_TAGS
        mats = np.array([self._mat([lm.get(model.link_names[i]) if lm.get(model.link_names[i]) in MATERIAL_TAGS
                                    else st['robot_material']]) for i in li], np.int32)
        area = float(_tri_area(tri).sum())
        own0 = len(self.parts)
        self.parts.append((f'robot:{robot.uid}:gripper', f'{robot.name} gripper'))
        self.grip = (robot.uid, model, x, n, li, np.full(len(x), area / len(x)), mats, own0)
        self.gripper_frame = frame
        if 'robot_material' in st['defaults_used'] and not lm:
            self.notes.append(f'The gripper\'s links are taken as {st["robot_material"]} (the default robot material:'
                              ' model files carry no material tags; set robot_material or link_materials).')

    # ---------------------------------------------------------------- per pose
    def pose(self, k: int) -> dict:
        if k in self._pose_cache:
            return self._pose_cache[k]
        from ..scenegeom import scene_occluders
        _label, qd = self.poses[k]
        P, N, W, G, M, OWN = self.fixed
        if self.grip is not None:
            uid, model, x, n, li, w, mats, own0 = self.grip
            T = model.fk_array(np.asarray(qd.get(uid, model.home), float))
            R, t = T[li, :3, :3], T[li, :3, 3]
            Pg = np.einsum('nij,nj->ni', R, x) + t
            Ng = np.einsum('nij,nj->ni', R, n)
            P = np.concatenate([P, Pg])
            N = np.concatenate([N, Ng])
            W = np.concatenate([W, w])
            G = np.concatenate([G, np.ones(len(x), np.int8)])
            M = np.concatenate([M, mats])
            OWN = np.concatenate([OWN, np.full(len(x), own0, np.int32)])
        occ = scene_occluders(self.project, qd)
        if occ.notes:
            for t_ in occ.notes:
                if t_ not in self.notes:
                    self.notes.append(t_)
        d = {'qd': qd, 'occ': occ, 'P': P, 'N': N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-300),
             'W': W, 'G': G, 'M': M, 'OWN': OWN}
        self._pose_cache[k] = d
        return d

    def _cube(self, occ, origin):
        from .. import occlusion as OC
        return OC.CACHE.get(occ, origin, self.st['cube_n'], self.st['backend'])

    def _visible(self, k: int, occ, origin, idx=None) -> np.ndarray:
        """(N,) the points of pose k seen from origin past occ. A point on a closed part (an object, the gripper)
        whose outward normal faces away from origin is on its far side: hidden. The others are lifted lift_mm along
        the normal turned towards origin and tested against the cube map with a slope-scaled tolerance: a surface
        seen at θ from its normal changes depth by up to ¾ texel × distance × tan θ between the texel's centre ray
        and the point's (θ capped at 80°), so the tolerance is eps + that; without it, oblique surfaces shadow
        themselves (shadow-map "acne", Williams 1978)."""
        from .. import occlusion as OC
        d = self.pose(k)
        P, N, G = d['P'], d['N'], d['G']
        v = np.asarray(origin, float) - P
        dist = np.maximum(np.linalg.norm(v, axis=1), 1e-12)
        cn = np.einsum('ij,ij->i', N, v) / dist
        front = (cn > 0) | (G == 2)                           # surfaces are one-sided sheets: either side shows
        n = N * np.where(cn >= 0, 1.0, -1.0)[:, None]
        c = np.clip(np.abs(cn), math.cos(math.radians(80.0)), 1.0)
        if not len(occ):
            return front
        cube = self._cube(occ, origin)
        Pl = P + self.st['lift_mm'] / 1000.0 * n
        eps = OC.EPS_M + 0.75 * cube.texel_rad * dist * np.sqrt(1.0 - c * c) / c
        return front & cube.visible(Pl, eps)

    def shadow(self, k: int, light: dict) -> np.ndarray:
        """(N,) True where the light's centre sees the point at pose k."""
        pos = tuple(np.round(np.asarray(light['pose']['position'], float), 6))
        key = (k, pos)
        if key not in self._shadow_cache:
            d = self.pose(k)
            o = np.asarray(pos, float)
            self._shadow_cache[key] = self._visible(k, d['occ'], o)
        return self._shadow_cache[key]

    def sun_lit(self, k: int, light: dict, direction) -> np.ndarray:
        from .. import lights as LT
        from .. import raycast as RC
        key = (k, 'sun', light['uid'], tuple(np.round(light['pose']['position'], 6)), tuple(direction))
        if key not in self._shadow_cache:
            d = self.pose(k)
            ok = LT.through_window(light, d['P'], direction)
            if ok.any() and len(d['occ']):
                n = d['N'][ok] * np.sign(d['N'][ok] @ np.asarray(direction, float))[:, None]
                O = d['P'][ok] + self.st['lift_mm'] / 1000.0 * n
                t = RC.first_hit(O, np.broadcast_to(np.asarray(direction, float), O.shape), d['occ'].tris)
                from ..frames import quat_to_matrix
                z = quat_to_matrix(light['pose']['quaternion'])[:, 2]
                s = ((O - np.asarray(light['pose']['position'], float)) @ z) / -float(np.dot(direction, z))
                ok[np.flatnonzero(ok)] = ~(t < s)
            self._shadow_cache[key] = ok
        return self._shadow_cache[key]

    def emitters(self, light: dict, band: str):
        from .. import lights as LT
        key = (repr(sorted((k, repr(v)) for k, v in light.items())), band)
        if key not in self._em_cache:
            self._em_cache[key] = LT.emitters(light, band, self.st['n_area'])
        return self._em_cache[key]

    # ---------------------------------------------------------------- radiance
    def radiance(self, k: int, idx, wo, n, lights=None, polariser=None):
        """(L (m,) cd/m² towards wo, per-light share (n_lights, m)) at the points idx of pose k with normals n turned
        to the viewer; polariser: k (an ideal linear polariser's transmission) or None."""
        from .. import lights as LT
        d = self.pose(k)
        lights = self.lights if lights is None else lights
        P = d['P'][idx]
        Mi = d['M'][idx]
        m = len(idx)
        per = np.zeros((len(lights), m))
        for j, lt in enumerate(lights):
            em = self.emitters(lt, 'vis')
            lit = self.shadow(k, lt)[idx] if em is not None or lt.get('sun') else None
            if em is not None:
                wi, E = LT.irradiance(em, P)
                per[j] += self._shade(wi, E, n, wo, Mi, polariser) * lit
            sb = LT.sun_beam(lt, 'vis')
            if sb is not None:
                dirn, Es = sb
                ok = self.sun_lit(k, lt, tuple(dirn))[idx]
                wi = np.broadcast_to(dirn, (1, m, 3))
                per[j] += self._shade(wi, np.full((1, m), Es), n, wo, Mi, polariser) * ok
        return per.sum(axis=0), per

    def _shade(self, wi, E, n, wo, Mi, polariser):
        """Σ over emitters of f_r E⊥ max(0, n · ωi) per point: wi (K, m, 3), E (K, m)."""
        out = np.zeros(wi.shape[1])
        cos = np.einsum('kmj,mj->km', wi, n)
        for mi in np.unique(Mi):
            s = Mi == mi
            mat = self.mats[int(mi)]
            K = wi.shape[0]
            nn = np.broadcast_to(n[s], (K,) + n[s].shape)
            ww = np.broadcast_to(wo[s], (K,) + wo[s].shape)
            fd, fs, cd = self.MT.brdf_terms(nn, wi[:, s], ww, mat, 'vis')
            if polariser is not None:
                kd, ks = self.MT.polariser_factors(cd, mat, 'vis', polariser)
                f = fd * kd + fs * ks
            else:
                f = fd + fs
            out[s] = np.sum(f * E[:, s] * np.maximum(cos[:, s], 0.0), axis=0)
        return out

    def ir_ambient(self, k: int, idx, n, lights=None) -> tuple[np.ndarray, np.ndarray]:
        """(E_a (m,) W/m² of ambient IR on the surfaces at points idx, per light (n_lights, m))."""
        from .. import lights as LT
        d = self.pose(k)
        lights = self.lights if lights is None else lights
        P = d['P'][idx]
        per = np.zeros((len(lights), len(idx)))
        for j, lt in enumerate(lights):
            em = self.emitters(lt, 'ir')
            if em is not None:
                wi, E = LT.irradiance(em, P)
                per[j] += np.sum(E * np.maximum(np.einsum('kmj,mj->km', wi, n), 0), axis=0) * self.shadow(k, lt)[idx]
            sb = LT.sun_beam(lt, 'ir')
            if sb is not None:
                dirn, Es = sb
                per[j] += Es * np.maximum(n @ dirn, 0) * self.sun_lit(k, lt, tuple(dirn))[idx]
        return per.sum(axis=0), per

    # ---------------------------------------------------------------- camera visibility
    def seen(self, k: int, cam: _Cam):
        """(idx of the points seen, uv (m, 2), cos α (m,), R, o, normals turned to the camera (m, 3), wo (m, 3))."""
        key = (k, cam.uid)
        if key in self._vis_cache:
            return self._vis_cache[key]
        from ..scenegeom import sensor_occluders
        d = self.pose(k)
        R, o = cam.optical(self.project, d['qd'])
        v = o - d['P']
        dist = np.maximum(np.linalg.norm(v, axis=1), 1e-12)
        wo = v / dist[:, None]
        n = d['N'] * np.sign(np.einsum('ij,ij->i', d['N'], wo))[:, None]
        occ = sensor_occluders(self.project, cam.sensor, d['qd'], base=d['occ'], origins=o[None])
        vis = self._visible(k, occ, o)
        rec = (vis, wo, n, R, o)
        self._vis_cache[key] = rec
        return rec

    def camera_eval(self, cam: _Cam, lights=None, polariser=None, tilt_deg: float = 0.0, keep_map=False):
        """Per pose: {'sat_pct', 'surface_sat_pct', 'visible_m2', ...}, plus the glare map when keep_map."""
        st = self.st
        gw, gh = st['grid']
        W_, H_ = cam.cam.width, cam.cam.height
        pol = polariser if polariser is not None else (cam.polariser_k if cam.polariser else None)
        out = []
        gsat = np.zeros(gw * gh)
        gidx = np.zeros(gw * gh)
        dom = {}
        sat_pts = None
        worst = -1.0
        for k in range(len(self.poses)):
            d = self.pose(k)
            vis, wo, n, R, o = self.seen(k, cam)
            if tilt_deg:
                R = R @ _rx(tilt_deg)
            Q = (d['P'] - o) @ R
            uv, inimg = cam.cam.project(Q)
            inimg = np.asarray(inimg, bool) & vis
            idx = np.flatnonzero(inimg)
            rec = {'pose': self.poses[k][0], 'visible_target_m2': 0.0, 'sat_pct': None, 'surface_sat_pct': None}
            if not len(idx):
                out.append(rec)
                continue
            L, per = self.radiance(k, idx, wo[idx], n[idx], lights, pol)
            w = d['W'][idx]
            g = d['G'][idx]
            if cam.absolute:
                cosa = Q[idx, 2] / np.maximum(np.linalg.norm(Q[idx], axis=1), 1e-12)
                Eimg = math.pi / 4 * st['lens_transmittance'] * L * cosa ** 4 / float(cam.f_number) ** 2
                dn = float(cam.sens) * cam.gain * float(cam.exposure) * Eimg
                full = 2 ** cam.bits - 1
                level = dn / full
                sat = dn >= full
                rec['mean_level'] = _num(float(np.sum(w * level) / max(np.sum(w), 1e-300)), 5)
            else:
                Lm = float(np.sum(w * L) / max(np.sum(w), 1e-300))
                level = L / Lm if Lm > 0 else np.zeros_like(L)
                sat = level >= st['sat_ratio'] if Lm > 0 else np.zeros(len(L), bool)
                rec['metered_cd_m2'] = _num(Lm, 5)
            tgt = g != 2
            vt = _wsum(w, tgt)
            rec['visible_target_m2'] = _num(vt, 8)
            rec['sat_pct'] = _num(100.0 * _wsum(w, tgt & sat) / vt, 4) if vt > 0 else None
            vs = _wsum(w, ~tgt)
            rec['surface_sat_pct'] = _num(100.0 * _wsum(w, ~tgt & sat) / vs, 4) if vs > 0 else None
            rec['peak_level'] = _num(float(level[tgt].max()) if tgt.any() else None, 4)
            if (sat & tgt).any():
                ls = self.lights if lights is None else lights
                for j in np.argmax(per[:, sat & tgt], axis=0):
                    u = ls[int(j)]['uid']
                    dom[u] = dom.get(u, 0) + 1
                parts = np.bincount(d['OWN'][idx][sat & tgt], weights=w[sat & tgt], minlength=len(self.parts))
                rec['parts'] = {self.parts[i][1]: _num(100.0 * parts[i] / vt, 3) for i in np.flatnonzero(parts)}
            if keep_map:
                ci = np.clip(((uv[idx, 0] + 0.5) / W_ * gw).astype(int), 0, gw - 1)
                cj = np.clip(((uv[idx, 1] + 0.5) / H_ * gh).astype(int), 0, gh - 1)
                cell = cj * gw + ci
                cw = np.bincount(cell, weights=w, minlength=gw * gh)
                cs = np.bincount(cell, weights=w * sat, minlength=gw * gh)
                with np.errstate(invalid='ignore', divide='ignore'):
                    share = np.where(cw > 0, cs / np.maximum(cw, 1e-300), 0.0)
                gsat = np.maximum(gsat, share)
                mx = np.zeros(gw * gh)
                np.maximum.at(mx, cell, level)
                gidx = np.maximum(gidx, mx)
                if rec['sat_pct'] is not None and rec['sat_pct'] > worst:
                    worst = rec['sat_pct']
                    sp = d['P'][idx][sat & tgt]
                    if len(sp) > MAX_OVERLAY_POINTS:
                        sp = sp[np.linspace(0, len(sp) - 1, MAX_OVERLAY_POINTS).astype(int)]
                    sat_pts = sp
            out.append(rec)
        vals = [r['sat_pct'] for r in out if r['sat_pct'] is not None]
        res = {'per_pose': out, 'sat_pct': _num(np.mean(vals), 4) if vals else None,
               'sat_pct_worst': _num(max(vals), 4) if vals else None,
               'worst_pose': (max((r for r in out if r['sat_pct'] is not None), key=lambda r: r['sat_pct'])['pose']
                              if vals else None),
               'dominant': dom}
        sv = [r['surface_sat_pct'] for r in out if r['surface_sat_pct'] is not None]
        res['surface_sat_pct'] = _num(np.mean(sv), 4) if sv else None
        if keep_map:
            res['glare_map'] = {'grid': [gw, gh], 'image': [W_, H_], 'sat': [round(float(x), 4) for x in gsat],
                                'level': [round(float(x), 4) for x in gidx],
                                'level_is': ('DN / full scale' if cam.absolute else
                                             'glare index L / L̄ (relative)')}
            res['sat_points_mm'] = (np.round(sat_pts * 1000).astype(int).tolist() if sat_pts is not None else [])
        return res

    # ---------------------------------------------------------------- IR wash-out
    def depth_eval(self, dz: _Depth, lights=None) -> dict:
        from ..frames import quat_to_matrix
        from ..scenegeom import sensor_occluders, sensor_world_pose
        from . import depth_validity as DV
        st = self.st
        ds = dz.ds
        out = []
        for k in range(len(self.poses)):
            d = self.pose(k)
            body = sensor_world_pose(self.project, dz.sensor, d['qd'])
            T = body @ ds.ref.pose_in_body
            R, o = quat_to_matrix(T.quaternion), np.asarray(T.position, float)
            v = o - d['P']
            wo = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
            n = d['N'] * np.sign(np.einsum('ij,ij->i', d['N'], wo))[:, None]
            occ = sensor_occluders(self.project, dz.sensor, d['qd'], base=d['occ'], origins=o[None])
            vis = self._visible(k, occ, o)
            _uv, inimg = ds.ref.camera.project((d['P'] - o) @ R)
            seen = np.asarray(inimg, bool) & vis
            if dz.pj is not None and dz.fov:
                seen &= DV._projector_cone(d['P'], body, dz.pj)
            idx = np.flatnonzero(seen)
            rec = {'pose': self.poses[k][0], 'loss_pct': None, 'washout_pct': None, 'dark_pct': None,
                   'through_pct': None, 'visible_target_m2': 0.0}
            g = d['G'][idx]
            tgt = g != 2
            w = d['W'][idx]
            vt = _wsum(w, tgt)
            if not len(idx) or vt <= 0:
                out.append(rec)
                continue
            nn = n[idx]
            Ea, per = self.ir_ambient(k, idx, nn, lights)
            mats = [self.mats[int(i)] for i in d['M'][idx]]
            alb = np.array([m.ir_albedo for m in mats])
            metal = np.array([m.metal for m in mats])
            thr = np.array([m.transmissive and m.depth_reads_through for m in mats])
            dark = (alb < st['ir_albedo_min']) & ~metal & ~thr
            Ep = None
            wash = np.zeros(len(idx), bool)
            if dz.power is not None and dz.omega:
                pp = body @ dz.pj['pose']
                dp = np.asarray(pp.position, float) - d['P'][idx]
                r2 = np.maximum(np.einsum('ij,ij->i', dp, dp), 1e-12)
                cosp = np.maximum(np.einsum('ij,ij->i', nn, dp / np.sqrt(r2)[:, None]), 0)
                Ep = float(dz.power) / dz.omega * cosp / r2
                with np.errstate(invalid='ignore', divide='ignore'):
                    C = np.where(Ep + Ea > 0, Ep / (Ep + Ea), 0.0)
                wash = (Ep > 0) & (C < st['contrast_min'])        # Ep = 0: the pattern cannot reach it (not light)
                rec['washout_pct'] = _num(100 * _wsum(w, tgt & wash & ~thr & ~dark) / vt, 4)
                rec['projector_w_m2_median'] = _num(float(np.median(Ep[tgt])), 6)
            rec['ambient_ir_w_m2_median'] = _num(float(np.median(Ea[tgt])), 6)
            rec['ambient_ir_w_m2_max'] = _num(float(np.max(Ea[tgt])), 6)
            rec['dark_pct'] = _num(100 * _wsum(w, tgt & dark) / vt, 4)
            rec['through_pct'] = _num(100 * _wsum(w, tgt & thr) / vt, 4)
            lost = thr | dark | wash
            rec['loss_pct'] = _num(100 * _wsum(w, tgt & lost) / vt, 4)
            rec['visible_target_m2'] = _num(vt, 8)
            if Ea.any():
                share = per[:, tgt].sum(axis=1)
                ls = lights if lights is not None else self.lights
                rec['ir_by_light'] = {ls[j]['uid']: _num(share[j] / max(share.sum(), 1e-300), 4)
                                      for j in range(len(ls)) if share[j] > 0}
            out.append(rec)

        def mean(key):
            v = [r[key] for r in out if r.get(key) is not None]
            return _num(np.mean(v), 4) if v else None
        res = {'per_pose': out, 'loss_pct': mean('loss_pct'), 'washout_pct': mean('washout_pct'),
               'dark_pct': mean('dark_pct'), 'through_pct': mean('through_pct'),
               'ambient_ir_w_m2_median': mean('ambient_ir_w_m2_median'),
               'projector_w_m2_median': mean('projector_w_m2_median')}
        by: dict = {}
        for r in out:
            for u, s in (r.get('ir_by_light') or {}).items():
                by[u] = by.get(u, 0.0) + s
        res['ir_by_light'] = {u: _num(s / max(len(out), 1), 4) for u, s in by.items()}
        return res


def _tri_area(tri) -> np.ndarray:
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def _tri_normals(tri) -> np.ndarray:
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)


def _subtree_links(model, frame: str) -> list:
    """Link indices of the gripper frame's link and every link below it in the model's tree."""
    li = model.link_index(frame)
    b0 = int(model._lb[li])
    par = np.asarray(model.m.body_parentid, int)
    inside = set()
    for b in range(len(par)):
        x = b
        while x > 0:
            if x == b0:
                inside.add(b)
                break
            x = int(par[x])
    if b0 == 0:
        inside = set(range(len(par)))
    return [i for i, b in enumerate(model._lb) if int(b) in inside]


# ====================================================================== the study
@jobs.study(SID, 'Lighting and glare', quick=False, batch='lighting')
def run(project, ctx: jobs.Ctx) -> dict:   # noqa: C901 — the study's steps in order, each said in the docstring
    """Lighting, glare and IR wash-out (module docstring)."""
    from .. import lights as LT
    from .. import occlusion as OC
    t0 = time.perf_counter()
    st = settings_of(project)
    notes: list = []
    ctx.report(0.0, 'Reading the lights')
    lights_all = LT.lights_of(project)
    if not any(s.enabled and s.kind in ('camera', 'depth') for s in project.variant.sensors):
        raise StudyError('The lighting study needs at least one camera or depth sensor in use: add one (Sensors ▸ '
                         'Add camera) or tick "In use" on the Sensors and mounts page.')
    light_rows = []
    for lt in lights_all:
        dsc = LT.describe(lt)
        light_rows.append({'uid': lt['uid'], 'name': lt['name'], 'kind': lt['kind'], 'enabled': lt['enabled'],
                           'position': [_num(x) for x in lt['pose']['position']], 'dim': lt['dim'],
                           'vis': dsc['vis'], 'ir': dsc['ir'], 'sun': dsc.get('sun'),
                           'library_id': lt.get('library_id', '')})
    base = {'settings': {k: v for k, v in st.items() if k != 'defaults_used'}, 'defaults_used': st['defaults_used'],
            'lights': light_rows, 'sat_ratio_source': ('ISO 12232 saturation-based speed: an 18 % grey metered '
                                                       'scene at 18 %/√2 = 12.7 % of saturation (half a stop of '
                                                       'headroom), read on Wikipedia "Film speed", 2026-11-07')}
    usable = [lt for lt in lights_all if lt['enabled'] and (LT.emitters(lt, 'vis') is not None or lt.get('sun')
                                                            or LT.emitters(lt, 'ir') is not None)]
    if not usable:
        why = ('There are no lights in the workspace' if not lights_all else
               'None of the lights has a stated luminance, luminous flux or IR irradiance')
        return {**base, 'cameras': [], 'depth': [], 'recommendations': [], 'suggestion': None, 'notes': notes,
                'headline': {'value': None, 'status': 'warning', 'word': 'no lights',
                             'text': f'{why}: add the lab\'s lights on the Lighting page (from the library or with '
                                     'the figures you measured), then run again.'}}
    ev = Evaluator(project, st, notes, ctx)
    ev.lights = [lt for lt in lights_all if lt['enabled']]
    ctx.report(0.1, f'{len(ev.poses)} working poses, {len(ev.cams)} cameras')
    if not ev.cams and not ev.depths:
        raise StudyError('None of the sensors in use could be modelled: ' + '; '.join(s['why'] for s in ev.skipped))
    cams_out = []
    for i, c in enumerate(ev.cams):
        if ctx.cancelled():
            break
        ctx.report(0.1 + 0.4 * i / max(len(ev.cams), 1), f'Glare: {c.name}')
        r = ev.camera_eval(c, keep_map=True)
        cams_out.append({'uid': c.uid, 'name': c.name, 'imager': c.imager.name, 'imager_why': c.why,
                         'imager_label': c.imager.label, 'absolute': c.absolute, 'mode': c.mode_text,
                         'polariser': c.polariser, 'slot': c.sensor.slot, **r,
                         'status': status_for(r['sat_pct'], st)[0], 'word': status_for(r['sat_pct'], st)[1]})
    depth_out = []
    for i, dz in enumerate(ev.depths):
        if ctx.cancelled():
            break
        ctx.report(0.5 + 0.15 * i / max(len(ev.depths), 1), f'IR wash-out: {dz.name}')
        r = ev.depth_eval(dz)
        if dz.power is None:
            pnote = ('The projector\'s optical power is not stated by the maker, so its pattern\'s irradiance and the '
                     'wash-out are not computed (the dark and clear surfaces are).')
        else:
            pnote = f'Projector: {dz.power * 1000:g} mW over {dz.fov[0]:g}° × {dz.fov[1]:g}° ({dz.power_source}).'
        depth_out.append({'uid': dz.uid, 'name': dz.name, 'technology': dz.tech,
                          'projector_w': dz.power, 'projector_fov_deg': list(dz.fov) if dz.fov else None,
                          'projector_note': pnote, **r})
    # ---------------------------------------------------------------- headline
    scored = [c for c in cams_out if c['sat_pct'] is not None]
    worst = max(scored, key=lambda c: c['sat_pct']) if scored else None
    if worst is not None:
        sev, word = status_for(worst['sat_pct'], st)
        rel = '' if worst['absolute'] else ' (relative model)'
        text = (f'{worst["sat_pct"]:.1f} % of the object and gripper area {worst["name"]} sees is in specular '
                f'saturation{rel}, {worst["sat_pct_worst"]:.1f} % at its worst pose ({worst["worst_pose"]}).')
        head = {'value': worst['sat_pct'], 'status': sev, 'word': word, 'camera': worst['uid'], 'text': text,
                'relative': not worst['absolute']}
    else:
        head = {'value': None, 'status': 'warning', 'word': 'nothing seen', 'camera': None, 'relative': True,
                'text': 'No camera sees the object or the gripper at the working poses.'}
    dep_worst = max((d for d in depth_out if d['loss_pct'] is not None), key=lambda d: d['loss_pct'], default=None)
    if dep_worst is not None:
        head['depth_loss_pct'] = dep_worst['loss_pct']
        head['depth_sensor'] = dep_worst['uid']
        head['text'] += (f' {dep_worst["name"]} is predicted to lose valid depth on {dep_worst["loss_pct"]:.1f} % of '
                         'what it sees of them' + ('' if dep_worst['washout_pct'] is not None else
                                                   ' (from dark and clear surfaces; wash-out not computed: the '
                                                   'projector\'s power is not stated)') + '.')
        dsev = depth_status_for(dep_worst['loss_pct'], st)
        if head['value'] is not None and SEV_ORDER[dsev] > SEV_ORDER.get(head['status'], 1):
            head['status'], head['word'] = dsev, f'{head["word"]}; {DEPTH_WORDS[dsev]}'
            head['status_from'] = 'depth'
    # ---------------------------------------------------------------- recommendations
    recs = []
    if st['recommend'] and not ctx.cancelled():
        ctx.report(0.65, 'Recommendations')
        recs = _recommend(ev, st, cams_out, depth_out, worst, dep_worst, ctx)
    sug = recs[0] if recs and (recs[0].get('delta') or 0) < 0 else None
    if sug is not None:
        head['action'] = sug['text']
    renderer = OC.gl_status()
    return {**base, 'headline': head, 'cameras': cams_out, 'depth': depth_out, 'recommendations': recs,
            'suggestion': sug, 'skipped': ev.skipped, 'notes': notes,
            'poses': {'source': ev.pose_source, 'labels': [p[0] for p in ev.poses], 'n': len(ev.poses)},
            'points': {'fixed': int(len(ev.fixed[0])), 'gripper': int(len(ev.grip[2])) if ev.grip else 0},
            'materials': [m.to_dict() for m in ev.mats],
            'renderer': {k: renderer.get(k) for k in ('renderer', 'backend', 'how') if k in renderer},
            'study_seconds': round(time.perf_counter() - t0, 3)}


def _dominant_light(ev: Evaluator, cam_res: dict | None, dep_res: dict | None) -> dict | None:
    if cam_res and cam_res.get('dominant'):
        uid = max(cam_res['dominant'], key=cam_res['dominant'].get)
    elif dep_res and dep_res.get('ir_by_light'):
        uid = max(dep_res['ir_by_light'], key=dep_res['ir_by_light'].get)
    else:
        return None
    return next((lt for lt in ev.lights if lt['uid'] == uid), None)


def _recommend(ev: Evaluator, st, cams_out, depth_out, worst, dep_worst, ctx) -> list:   # noqa: C901
    """Each candidate change with the predicted figure before and after (module docstring), best first."""
    recs = []
    cam = next((c for c in ev.cams if worst is not None and c.uid == worst['uid']), None)
    dz = next((d for d in ev.depths if dep_worst is not None and d.uid == dep_worst['uid']), None)

    def cam_after(**kw):
        return ev.camera_eval(cam, **kw)['sat_pct']

    def dep_after(lights):
        return ev.depth_eval(dz, lights)['loss_pct'] if dz is not None else None

    if cam is not None and worst['sat_pct'] and worst['sat_pct'] > 0:
        before = worst['sat_pct']
        if not cam.polariser:
            a = cam_after(polariser=1.0)
            if a is not None and a < before:
                ptext = (f'Fit a linear polariser to {cam.name}: saturation {before:.1f} % → {a:.1f} % (an ideal '
                         'polariser turned at its best: an upper bound on the gain).')
            else:
                ptext = (f'A linear polariser on {cam.name} would not lower its saturation here ({before:.1f} % → '
                         f'{a if a is not None else before:.1f} %): metals reflect both polarisations nearly '
                         'alike, and '
                         + ('the auto-exposure makes up the diffuse light it removes.' if not cam.absolute else
                            'it halves the diffuse light as well.'))
            recs.append({'kind': 'polariser', 'target': cam.uid, 'target_name': cam.name, 'figure': 'saturation',
                         'before': before, 'after': a, 'delta': _num(a - before, 4) if a is not None else None,
                         'text': ptext,
                         'cite': 'Malus\'s law and the Fresnel equations (Hecht, Optics, §8.2 and §4.6)',
                         'apply': {'section': 'analysis', 'sensor_uid': cam.uid, 'polariser': True}})
        lt = _dominant_light(ev, worst, None)
        if lt is not None:
            j = ev.lights.index(lt)
            best = None
            for ax, sgn, word in ((0, 1, '+x'), (0, -1, '−x'), (1, 1, '+y'), (1, -1, '−y'), (2, 1, 'up')):
                if ctx.cancelled():
                    break
                lt2 = copy.deepcopy(lt)
                lt2['pose']['position'][ax] += sgn * st['move_m']
                ls = list(ev.lights)
                ls[j] = lt2
                a = cam_after(lights=ls)
                if a is not None and (best is None or a < best[0]):
                    best = (a, word, lt2)
            if best is not None:
                a, word, lt2 = best
                where = 'up' if word == 'up' else f'along world {word}'
                recs.append({'kind': 'move_light', 'target': lt['uid'], 'target_name': lt['name'],
                             'figure': 'saturation', 'before': before, 'after': a, 'delta': _num(a - before, 4),
                             'text': f'Move {lt["name"]} {st["move_m"] * 100:.0f} cm {where} (it lights most of the '
                                     f'saturated points in {cam.name}): saturation {before:.1f} % → {a:.1f} %.',
                             'apply': {'section': 'workspace', 'light_uid': lt['uid'],
                                       'position': [float(x) for x in lt2['pose']['position']]}})
            lt3 = copy.deepcopy(lt)
            lt3['dim'] = float(lt.get('dim', 1.0)) * st['dim_to']
            ls = list(ev.lights)
            ls[j] = lt3
            a = cam_after(lights=ls)
            if a is not None:
                extra = ('' if cam.absolute or len(ev.lights) > 1 else
                         ' (the relative model\'s auto-exposure makes up for a dimmer single light)')
                recs.append({'kind': 'dim_light', 'target': lt['uid'], 'target_name': lt['name'],
                             'figure': 'saturation', 'before': before, 'after': a, 'delta': _num(a - before, 4),
                             'text': f'Dim {lt["name"]} to {lt3["dim"] * 100:.0f} %: saturation {before:.1f} % → '
                                     f'{a:.1f} %{extra}.',
                             'apply': {'section': 'workspace', 'light_uid': lt['uid'], 'dim': lt3['dim']}})
        for deg in st['tilt_deg']:
            for sgn in (1, -1):
                if ctx.cancelled():
                    break
                a = cam_after(tilt_deg=sgn * deg)
                if a is None:
                    continue
                recs.append({'kind': 'tilt', 'target': cam.uid, 'target_name': cam.name, 'figure': 'saturation',
                             'before': before, 'after': a, 'delta': _num(a - before, 4),
                             'text': f'Tilt {cam.name} {sgn * deg:+g}° (about its optical x axis): saturation '
                                     f'{before:.1f} % → {a:.1f} % (it then also sees a different part of the scene).',
                             'apply': {'section': 'variant', 'sensor_uid': cam.uid,
                                       'mount_pose': cam.mount_after_tilt(sgn * deg).to_dict()}})
    if dz is not None and dep_worst['washout_pct']:
        lt = _dominant_light(ev, None, dep_worst)
        if lt is not None:
            j = ev.lights.index(lt)
            lt2 = copy.deepcopy(lt)
            lt2['dim'] = 0.0
            ls = list(ev.lights)
            ls[j] = lt2
            a = dep_after(ls)
            before = dep_worst['loss_pct']
            if a is not None:
                what = 'blind or curtain' if lt['kind'] == 'window' else 'switch off'
                recs.append({'kind': 'remove_ir', 'target': lt['uid'], 'target_name': lt['name'],
                             'figure': 'depth loss', 'before': before, 'after': a, 'delta': _num(a - before, 4),
                             'text': f'{what.capitalize()} {lt["name"]} (most of the ambient IR {dz.name} sees): '
                                     f'predicted depth loss {before:.1f} % → {a:.1f} %.',
                             'apply': {'section': 'workspace', 'light_uid': lt['uid'], 'dim': 0.0}})
    sat_recs = sorted([r for r in recs if r['figure'] == 'saturation'], key=lambda r: (r['delta'] or 0))
    dep_recs = [r for r in recs if r['figure'] != 'saturation']
    return sat_recs + dep_recs if (sat_recs and (sat_recs[0]['delta'] or 0) < 0) else dep_recs + sat_recs
