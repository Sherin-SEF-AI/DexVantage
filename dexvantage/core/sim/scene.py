"""The project as a MuJoCo scene (spec §9.1): the robots at q, the task objects and surfaces, lights, and one MuJoCo
camera per sensor imager with the imager's intrinsics.

    build_scene(project, variant=None, objects=True, lights=True) -> SceneBuild
    save_scene(project, folder, variant=None) -> path of folder/scene.xml (+ folder/assets/, folder/scene_notes.txt)

SceneBuild
    spec            the mujoco.MjSpec (spec.to_xml() is the MJCF; .xml() the same), with .assets holding the robots'
                    mesh and texture files by name
    model           the compiled mujoco.MjModel (robot geoms the predictions do not see moved to group HIDDEN_GROUP)
    camera_names    {sensor_uid: MuJoCo camera name of the sensor's primary imager}
    imagers         {sensor_uid: (imager name, optics.CameraModel at its stream size)} of that imager
    cameras         {(sensor_uid, imager name): SceneCamera} for every imager of every sensor in use
    body_map        {MuJoCo body id: owner, and body name: owner}; owner ('robot', robot uid, link) | ('object', uid) |
                    ('surface', uid)
    geom_owner      (ngeom,) int32 index into owners for every geom that renders (-1: hidden, or not owned)
    owners          [owner] in index order (the Renderer's seg values index it)
    notes           what was left out or assumed, in words
    qpos(q)         the scene's qpos with each robot at q[robot uid] (its home pose when missing)

What is in the scene, and how it matches what the predictions see (core/scenegeom.py, core/camrender.py):
- **Robots**: each robot's stored MJCF or URDF (the files the project holds; nothing is read from disk), read as
  core/robot/model.py reads it (fusestatic off, so every link keeps its body; URDF visual geoms kept) and attached
  with mujoco.MjSpec.attach under a frame at the robot's base pose, its names prefixed 'robot<k>_'. Keyframes and the
  file's own world-body geoms (floors) are dropped (the predictions do not see them either; the robot's q is set
  through qpos). The geoms the predictions draw are the robot's visual geoms (RobotModel's rule: contype =
  conaffinity = 0 when the file has such geoms, else every geom; rgba alpha > 0; a type RobotModel can tessellate):
  every other robot geom is moved to group HIDDEN_GROUP, which MuJoCo's default visual options (geom groups 0-2 on,
  3-5 off) do not draw, so collision boxes never show in a render.
- **Task objects**: a static body 'object_<uid>' at the object's pose with a primitive of its exact shape (box with
  half-edges size/2; cylinder of radius size[0] and half-length size[1]/2 along its z axis; sphere of radius
  size[0]) or, for 'mesh', a mesh made of the project's triangles. The predictions tessellate cylinders and spheres
  (24 segments, at most 1.1 % of the radius inside the surface): a known, small difference at their silhouettes.
- **Surfaces**: a static body 'surface_<uid>' with a box whose top is at the surface's z (a slab down to z − thickness
  when the record has one, else SURFACE_SKIN_M thick, a default: the predictions use a rectangle of no thickness).
- **Materials**: each object's and surface's material tags map to a MuJoCo material by MATERIALS (defaults of
  DexVantage, not measured optics; MuJoCo renders visible light only, so 'ir_absorbing' is a dark matte surface and
  'transparent' a see-through colour whose depth and segmentation are still the surface's, as for an opaque one).
- **Lights**: the project's lights (project.workspace.lights, the lighting study's records, core/lights.py: a pose
  whose local +z is the way the emitting face looks, a kind, 'dim', 'enabled') when it has any, else DEFAULT_LIGHTS
  (labelled defaults), and MuJoCo's head light. A light that is off is left out; a point light is a MuJoCo point
  light; a spot a MuJoCo spot of cutoff = half its stated beam angle and exponent 0 (a uniform cone; 45°, MuJoCo's
  default, when no beam is stated); a panel, ring or window (a Lambertian area emitter) a spot of cutoff 90° and
  exponent 1 (the cos θ falloff of Lambert's law over the half space in front of the face), at the face's centre.
  The colour is MuJoCo's default diffuse (0.7) times 'dim' (MuJoCo's classic renderer has no photometric units, so
  the stated luminance and flux do not set it). Older records with 'position' and 'direction' (and 'kind'
  'directional') are read as given. Lights only change the RGB image; segmentation and depth do not depend on them.
- **Cameras**: one MuJoCo camera per imager of every enabled camera or depth sensor of the variant, named
  '<sensor>__<imager>' (sanitised), in the body of the link it is mounted on (or the world body for a static mount),
  at mount pose · imager pose in the body · diag(1, −1, −1) (MuJoCo's camera looks along its −z axis with +y up; the
  optical frame along +z with +y down). Intrinsics: resolution = the imager's stream size (W, H); sensorsize = W × H
  nominal pixels of PIXEL_PITCH_M (MuJoCo needs a length; only the ratios matter); focalpixel = (fx, fy) and
  principalpixel = ((W − 1)/2 − cx, (H − 1)/2 − cy): MuJoCo's principal offset moves the image of the optical axis to
  (W − 1)/2 − px, (H − 1)/2 − py (validation/mjrender.py, measured on renders; the scene check
  groundtruth.scene_intrinsics measures it again). MuJoCo cameras are pinhole: a lens with distortion gets its
  pinhole part (fx, fy, cx, cy) in the scene, and core/sim/render.py renders a wider pinhole (or cube faces) and
  remaps it through the lens model.

MuJoCo's camera attributes were read in the XML reference of the installed version: doc/XMLreference.rst of
github.com/google-deepmind/mujoco at tag 3.17.0, section body/camera, read 2026-11-07. fovy is the vertical field of
view in degrees ("the horizontal field of view is computed automatically given the window size"); sensorsize is "Size
of the camera sensor in length units. When specified, all intrinsic attributes become active and fovy is ignored";
focal / focalpixel are the focal length "in physical length units or in pixels, respectively. If both are specified,
the pixel value is used"; principal / principalpixel are the "Offset of the principal point (optical axis
intersection with the image plane) from the image center" (the sign is not stated there); resolution is "not used for
rendering since those dimensions are determined by the size of the rendering context" (it converts the pixel values
at compile time: MuJoCo 3.17.0 stores cam_intrinsic = focalpixel · sensorsize / resolution, checked in
tests/test_groundtruth.py). The near clipping plane is the model extent times visual/map znear: the scene sets
statistic/extent and znear so that the near plane is NEAR_M (5 mm, as core/occlusion.py and core/camrender.py).

Known limits: sensor housings are not in the scene (as in the predictions); robot meshes are the files' meshes, which
MuJoCo may re-process (it merges duplicate vertices and recomputes normals); equality constraints (a gripper's
linkage) are not solved: each joint takes its value from q, as in RobotModel; a mesh object with degenerate triangles
may be refused by MuJoCo's mesh compiler and is then left out with a note; materials are tags mapped to colours, not
measured reflectance.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

from ..frames import Pose, quat_from_matrix, quat_to_mujoco

HIDDEN_GROUP = 5                   # geom group MuJoCo's default visual options do not draw (groups 3-5 are off)
NEAR_M = 0.005                     # default near clipping plane (m), as core/occlusion.py NEAR_M
FAR_M = 50.0                       # far clipping plane (m)
PIXEL_PITCH_M = 3e-6               # nominal pixel pitch for MuJoCo's sensorsize (a length is needed; ratios matter)
SURFACE_SKIN_M = 0.001             # default thickness of a surface given without one (its top at z)
OPTICAL_TO_MUJOCO = np.diag([1.0, -1.0, -1.0])
OBJECT_RGBA = (0.78, 0.74, 0.66, 1.0)          # core/camrender DEFAULT_COLORS (a drawing default)
SURFACE_RGBA = (0.55, 0.53, 0.50, 1.0)
# material tags -> MuJoCo material attributes: DexVantage's defaults (not measured optics); applied in this order
MATERIALS = {
    'matte': {'specular': 0.0, 'shininess': 0.0, 'reflectance': 0.0},
    'glossy': {'specular': 0.8, 'shininess': 0.9},
    'metallic': {'specular': 1.0, 'shininess': 0.6, 'reflectance': 0.2, 'rgb': (0.75, 0.75, 0.78)},
    'black': {'rgb': (0.05, 0.05, 0.05)},
    'ir_absorbing': {'rgb': (0.15, 0.15, 0.15), 'specular': 0.0},
    'transparent': {'alpha': 0.3},
}
DEFAULT_LIGHTS = (                 # defaults of DexVantage when the project has no lights
    {'name': 'overhead (default)', 'pos': (0.0, 0.0, 3.0), 'dir': (0.0, 0.0, -1.0), 'diffuse': 0.6,
     'directional': True},
    {'name': 'front (default)', 'pos': (2.0, -1.0, 2.0), 'dir': (-0.67, 0.33, -0.67), 'diffuse': 0.3,
     'directional': True},
)


@dataclass
class SceneCamera:
    """One imager's MuJoCo camera: its name, the body it rides on, its optical frame in that body (4 × 4, the robot
    link's frame or the world) split as T_body_sensor (link offset · mount pose) · T_sensor_optical (the imager in
    the sensor body), and the imager's CameraModel at its stream size."""
    name: str
    sensor_uid: str
    sensor_name: str
    imager: str
    role: str
    body: str
    T_body_sensor: np.ndarray
    T_sensor_optical: np.ndarray
    model: object
    label: str = ''
    primary: bool = False

    def local_optical(self, delta=None) -> np.ndarray:
        """The optical frame in the camera's body, the sensor body perturbed by delta (frames.Pose or 4 × 4 in the
        sensor body frame: a mount slip) when given."""
        T = self.T_body_sensor
        if delta is not None:
            D = delta.matrix() if hasattr(delta, 'matrix') else np.asarray(delta, float).reshape(4, 4)
            T = T @ D
        return T @ self.T_sensor_optical


@dataclass
class SceneBuild:
    spec: object
    camera_names: dict = field(default_factory=dict)
    imagers: dict = field(default_factory=dict)
    body_map: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    cameras: dict = field(default_factory=dict)
    owners: list = field(default_factory=list)
    geom_owner: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    model: object = None
    robots: dict = field(default_factory=dict)     # uid -> {'prefix', 'name', 'map': [(robot adr, scene adr, n)]}
    defaults: list = field(default_factory=list)  # the labelled defaults the scene used

    def xml(self) -> str:
        return self.spec.to_xml()

    @property
    def spec_or_xml(self):
        return self.spec

    def camera(self, sensor_uid: str, imager: str | None = None) -> SceneCamera | None:
        """The SceneCamera of a sensor's imager (its primary imager when imager is None)."""
        if imager is None:
            return next((c for (u, _i), c in self.cameras.items() if u == sensor_uid and c.primary), None)
        return self.cameras.get((sensor_uid, imager)) or next(
            (c for (u, i), c in self.cameras.items() if u == sensor_uid and _norm(i) == _norm(imager)), None)

    def qpos(self, q: dict | None = None) -> np.ndarray:
        """qpos of the compiled scene with each robot at q[uid] (its RobotModel home when missing; not clipped, as
        RobotModel.fk poses it). Joints a robot has but q does not (free, ball) take the robot's keyframe."""
        from ..robot.model import get_model
        out = np.asarray(self.model.qpos0, float).copy()
        for uid, r in self.robots.items():
            rm = get_model(r['project'], uid)
            v = (q or {}).get(uid)
            v = rm.home if v is None else rm._q(v)
            d = rm._kin(v)
            for a_r, a_s, n in r['map']:
                out[a_s:a_s + n] = d.qpos[a_r:a_r + n]
        return out


def _norm(s) -> str:
    return re.sub(r'[^a-z0-9]', '', str(s or '').lower())


def _safe(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(s or '')).strip('_') or 'x'


def _mjquat(R) -> np.ndarray:
    return np.asarray(quat_to_mujoco(quat_from_matrix(np.asarray(R, float))), float)


def _frame_pq(T) -> tuple[np.ndarray, np.ndarray]:
    T = np.asarray(T, float)
    return T[:3, 3].copy(), _mjquat(T[:3, :3])


# ====================================================================== robots
def robot_spec(project, robot, prefix: str):
    """(mujoco.MjSpec of the robot's stored files, {asset name: bytes}, notes), read as core/robot/model.py compiles
    them (fusestatic off; URDF visual geoms kept), the file names flattened and prefixed so robots never share an
    asset name. Raises RobotModelError."""
    import xml.etree.ElementTree as ET

    import mujoco

    from ..robot import model as RM
    from ..robot.importer import mjcf_spec_files
    if robot.source.file not in project.sources:
        raise RM.RobotModelError(f'The model file of {robot.name} is not stored in the project.')
    kind = robot.source.kind or 'urdf'
    res = RM._Resolver(robot, project.sources)
    namer = RM._flat_namer()
    assets, notes = {}, []
    if kind == 'mjcf':
        incs = {}
        for k, rel in res.refs.items():
            if k.startswith('include:') and res.prefix + rel in project.sources:
                incs[k[len('include:'):]] = project.sources[res.prefix + rel]
        try:
            spec = mujoco.MjSpec.from_string(project.sources[robot.source.file].decode('utf-8'), include=incs)
        except Exception as e:                       # noqa: BLE001 — explained
            raise RM.RobotModelError(f'MuJoCo could not read the stored MJCF of {robot.name}: {RM._first_line(e)}.') \
                from None
        by_key, drop = {}, []
        for cat, el, attr, fn in mjcf_spec_files(spec):
            folder = spec.texturedir if cat == 'texture' else spec.meshdir
            key = res.key(cat, fn, folder)
            if key is None:
                if cat == 'texture':
                    if el not in drop:
                        drop.append(el)
                        notes.append(f'{robot.name}: texture {os.path.basename(fn)} is not stored, so it is drawn '
                                     'without it.')
                    continue
                raise RM.RobotModelError(f'The MJCF of {robot.name} names the {cat} {os.path.basename(fn)}, which is '
                                         'not stored in the project.')
            if key not in by_key:
                by_key[key] = prefix + namer(key)
                assets[by_key[key]] = project.sources[key]
            if attr == 'file':
                el.file = by_key[key]
            else:
                cf = list(el.cubefiles)
                cf[attr] = by_key[key]
                el.cubefiles = cf
        for t in drop:
            for mat in spec.materials:
                mat.textures = ['' if x == t.name else x for x in mat.textures]
            spec.delete(t)
    elif kind == 'urdf':
        try:
            root = ET.fromstring(project.sources[robot.source.file])
        except ET.ParseError as e:
            raise RM.RobotModelError(f'The stored URDF of {robot.name} is not valid XML ({e}).') from None
        by_key = {}
        for m in root.iter('mesh'):
            fn = m.get('filename')
            if not fn:
                continue
            key = res.key('mesh', fn)
            if key is None:
                raise RM.RobotModelError(f'The URDF of {robot.name} names mesh {os.path.basename(fn)}, which is not '
                                         'stored in the project.')
            if key not in by_key:
                by_key[key] = prefix + namer(key)
                assets[by_key[key]] = project.sources[key]
            m.set('filename', by_key[key])
        mj = root.find('mujoco')
        if mj is None:
            mj = ET.Element('mujoco')
            root.insert(0, mj)
        comp = mj.find('compiler')
        if comp is None:
            comp = ET.SubElement(mj, 'compiler')
        comp.set('discardvisual', 'false')
        comp.set('fusestatic', 'false')
        comp.set('strippath', 'false')
        for k in ('meshdir', 'assetdir', 'texturedir'):
            comp.attrib.pop(k, None)
        spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding='unicode'))
    else:
        raise RM.RobotModelError(f'{robot.name} comes from a {kind} file, which this build cannot compile yet.')
    spec.meshdir = ''
    spec.texturedir = ''
    spec.compiler.fusestatic = False
    for k in list(spec.keys):
        spec.delete(k)
    dropped = 0
    for g in list(spec.worldbody.geoms):
        spec.delete(g)
        dropped += 1
    if dropped:
        notes.append(f'{robot.name}: {dropped} geom(s) of its file\'s world body (a floor or a stand) are left out, '
                     'as in the predictions (they are not the robot\'s links).')
    return spec, assets, notes


def _match_options(child, parent):
    """Copy the parent's simulation options and sizes into the child before attaching, so MjSpec.attach has no
    conflict to report (rendering does not depend on them)."""
    for k in dir(parent.option):
        if k.startswith('_'):
            continue
        try:
            v = getattr(parent.option, k)
            if callable(v):
                continue
            setattr(child.option, k, v)
        except (AttributeError, TypeError, ValueError):
            pass
    for k in ('nkey', 'nuserdata', 'nuser_body', 'nuser_jnt', 'nuser_geom', 'nuser_site', 'nuser_cam',
              'nuser_tendon', 'nuser_actuator', 'nuser_sensor', 'memory', 'nstack', 'njmax', 'nconmax'):
        try:
            setattr(child, k, getattr(parent, k))
        except (AttributeError, TypeError, ValueError):
            pass


# ====================================================================== static content
def _material(spec, name: str, tags, rgba) -> str:
    rgb, alpha = list(rgba[:3]), float(rgba[3])
    att = {'specular': 0.5, 'shininess': 0.5, 'reflectance': 0.0}         # MuJoCo's own material defaults
    for t in [t for t in MATERIALS if t in set(tags or ())]:
        a = MATERIALS[t]
        if 'rgb' in a:
            rgb = list(a['rgb'])
        if 'alpha' in a:
            alpha = a['alpha']
        for k in ('specular', 'shininess', 'reflectance'):
            if k in a:
                att[k] = a[k]
    mat = spec.add_material(name=name)
    mat.rgba = rgb + [alpha]
    mat.specular, mat.shininess, mat.reflectance = att['specular'], att['shininess'], att['reflectance']
    return name


def _add_object(spec, project, o, notes) -> bool:
    import mujoco

    from ..scenegeom import _mesh_tris
    G = mujoco.mjtGeom
    s = np.r_[np.asarray(o.size, float).reshape(-1), np.zeros(3)][:3]
    R = o.pose.matrix()[:3, :3]
    body = spec.worldbody.add_body(name=f'object_{o.uid}', pos=np.asarray(o.pose.position, float), quat=_mjquat(R))
    mat = _material(spec, f'mat_object_{o.uid}', o.materials, OBJECT_RGBA)
    kw = dict(name=f'object_{o.uid}', material=mat, contype=0, conaffinity=0, group=0)
    if o.shape == 'box' and np.all(np.abs(s) > 0):
        body.add_geom(type=G.mjGEOM_BOX, size=np.abs(s) / 2, **kw)
    elif o.shape == 'cylinder' and abs(s[0]) > 0 and abs(s[1]) > 0:
        body.add_geom(type=G.mjGEOM_CYLINDER, size=[abs(s[0]), abs(s[1]) / 2, 0.0], **kw)
    elif o.shape == 'sphere' and abs(s[0]) > 0:
        body.add_geom(type=G.mjGEOM_SPHERE, size=[abs(s[0]), 0.0, 0.0], **kw)
    elif o.shape == 'mesh':
        t = _mesh_tris(project, o.mesh_key)
        if t is None or not len(t):
            notes.append(f'Object {o.name} is left out: its mesh is not in the project.')
            spec.delete(body)
            return False
        V = np.asarray(t, float).reshape(-1, 3)
        mesh = spec.add_mesh(name=f'mesh_object_{o.uid}')
        mesh.uservert = V.reshape(-1).tolist()
        mesh.userface = np.arange(len(V), dtype=np.int64).tolist()
        try:
            mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        except AttributeError:
            pass
        body.add_geom(type=G.mjGEOM_MESH, meshname=mesh.name, **kw)
    else:
        notes.append(f'Object {o.name} is left out: its {o.shape} has no size or mesh to draw.')
        spec.delete(body)
        return False
    return True


def _add_surface(spec, s: dict, k: int, notes, defaults) -> str | None:
    import mujoco
    sz = np.asarray(s.get('size') or (0.0, 0.0), float).reshape(-1)[:2]
    c = np.asarray(s.get('center') or (0.0, 0.0), float).reshape(-1)[:2]
    z = float(s.get('z', 0.0))
    if not (len(sz) == 2 and (sz > 0).all()):
        notes.append(f'Surface {s.get("name") or k} is left out: it has no size.')
        return None
    th = float(s.get('thickness') or 0.0)
    if th <= 0:
        th = SURFACE_SKIN_M
        defaults.append(f'surface {s.get("name") or k}: drawn {SURFACE_SKIN_M * 1000:g} mm thick, its top at z '
                        '(the predictions use a rectangle of no thickness)')
    uid = str(s.get('uid') or k)
    body = spec.worldbody.add_body(name=f'surface_{uid}', pos=[c[0], c[1], z - th / 2])
    tags = [s.get('material')] if isinstance(s.get('material'), str) else list(s.get('materials') or [])
    mat = _material(spec, f'mat_surface_{_safe(uid)}', tags, SURFACE_RGBA)
    body.add_geom(name=f'surface_{uid}', type=mujoco.mjtGeom.mjGEOM_BOX, size=[sz[0] / 2, sz[1] / 2, th / 2],
                  material=mat, contype=0, conaffinity=0, group=0)
    return uid


def _add_lights(spec, project, lights: bool, notes, defaults):
    import mujoco
    if not lights:
        return
    given = [x for x in (project.workspace.lights or []) if isinstance(x, dict)]
    if not given:
        for L in DEFAULT_LIGHTS:
            li = spec.worldbody.add_light(name=_safe(L['name']), pos=list(L['pos']), dir=list(L['dir']))
            li.diffuse = [L['diffuse']] * 3
            li.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
            li.castshadow = False
        defaults.append('lights: an overhead and a front directional light (defaults; the project has no lights)')
        return
    from ..lights import AREA_KINDS
    for k, L in enumerate(given):
        try:
            if L.get('enabled') is False:
                notes.append(f'Light {L.get("name") or k} is off, so it is not in the scene.')
                continue
            if isinstance(L.get('pose'), dict):
                P = Pose.from_dict(L['pose'])
                pos, R = P.position, P.matrix()[:3, :3]
                d = R[:, 2]                         # core/lights.py: the emitting face looks along local +z
            else:
                pos = np.asarray(L.get('position') or (0.0, 0.0, 2.0), float)
                d = np.asarray(L.get('direction') or (0.0, 0.0, -1.0), float)
            d = d / max(np.linalg.norm(d), 1e-12)
            li = spec.worldbody.add_light(name=f'light_{_safe(L.get("uid") or k)}', pos=pos.tolist(), dir=d.tolist())
            kind = str(L.get('kind') or 'spot')
            li.type = {'directional': mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                       'point': mujoco.mjtLightType.mjLIGHT_POINT}.get(kind, mujoco.mjtLightType.mjLIGHT_SPOT)
            if kind in AREA_KINDS:
                li.cutoff, li.exponent = 90.0, 1.0
            elif kind == 'spot':
                beam = (L.get('beam_deg') or {}).get('value') if isinstance(L.get('beam_deg'), dict) else None
                if isinstance(beam, (int, float)) and beam > 0:
                    li.cutoff, li.exponent = float(min(beam / 2.0, 90.0)), 0.0
            dif = L.get('diffuse', L.get('intensity'))
            if isinstance(dif, (int, float)):
                li.diffuse = [float(min(max(dif, 0.0), 1.0))] * 3
            dim = L.get('dim')
            if isinstance(dim, (int, float)) and 0.0 <= dim < 1.0:
                li.diffuse = [float(x) * float(dim) for x in li.diffuse]
            li.castshadow = False
        except (TypeError, ValueError, KeyError) as e:
            notes.append(f'Light {L.get("name") or k} is left out: {e}.')


# ====================================================================== cameras
def _intrinsic_attrs(cam_model) -> dict:
    """MuJoCo camera intrinsics reproducing a CameraModel's pinhole part at its own size (module docstring)."""
    W, H = cam_model.width, cam_model.height
    return {'resolution': [W, H], 'sensor_size': [W * PIXEL_PITCH_M, H * PIXEL_PITCH_M],
            'focal_pixel': [cam_model.fx, cam_model.fy],
            'principal_pixel': [(W - 1) / 2 - cam_model.cx, (H - 1) / 2 - cam_model.cy]}


def add_camera(body, name: str, T_body_optical, cam_model):
    """Add a MuJoCo camera to a spec body: its optical frame T_body_optical (4 × 4) in the body, the pinhole part of
    cam_model as its intrinsics (sensorsize, focalpixel, principalpixel, resolution)."""
    p, qm = _frame_pq(np.c_[np.r_[np.asarray(T_body_optical, float)[:3, :3] @ OPTICAL_TO_MUJOCO, np.zeros((1, 3))],
                            np.r_[np.asarray(T_body_optical, float)[:3, 3], 1.0]])
    cam = body.add_camera(name=name, pos=p, quat=qm)
    for k, v in _intrinsic_attrs(cam_model).items():
        setattr(cam, k, v)
    return cam


def _sensor_cameras(spec, project, variant, rob_bodies, notes, defaults) -> list:
    """[(SceneCamera)] for every imager of the variant's enabled camera and depth sensors, added to the spec."""
    from .. import library as LB
    from ..robot.model import RobotModelError, get_model
    from ..studies import coverage as CV
    out, used = [], set()
    for s in variant.sensors:
        if not s.enabled or s.kind not in ('camera', 'depth'):
            continue
        try:
            ims = LB.sensor_imagers(s, CV.sensor_mode(s, {}))
        except LB.LibraryError as e:
            notes.append(f'{s.name} has no camera in the scene: {e}')
            continue
        if not ims:
            notes.append(f'{s.name} has no imager in its library entry, so it has no camera in the scene.')
            continue
        if ims[0].is_default:
            defaults.append(f'{s.name}: the default camera (no library device), {ims[0].label}')
        prim, _why = CV.pick_imager(s, ims, {})
        if s.mount.on_link:
            r = project.robot(s.mount.robot_uid)
            if r is None or r.uid not in rob_bodies:
                notes.append(f'{s.name} is mounted on a robot that is not in the scene, so it has no camera.')
                continue
            try:
                rm = get_model(project, r.uid)
                li = rm.link_index(s.mount.parent_frame)
            except (RobotModelError, KeyError):
                notes.append(f'{s.name} is mounted on {s.mount.parent_frame}, which its robot does not have.')
                continue
            bname = rob_bodies[r.uid] + rm.body_link[int(rm._lb[li])]
            T_bs = rm._loff[li] @ s.mount.pose.matrix()
            body = spec.body(bname)
        else:
            bname, body = 'world', spec.worldbody
            T_bs = s.mount.pose.matrix()
        for im in ims:
            base = f'{_safe(s.name)}__{_safe(im.name)}'
            name, k = base, 2
            while name in used:
                name, k = f'{base}_{k}', k + 1
            used.add(name)
            T_so = im.pose_in_body.matrix()
            add_camera(body, name, T_bs @ T_so, im.camera)
            if im.camera.model != 'pinhole':
                notes.append(f'{s.name} · {im.name}: MuJoCo cameras are pinhole, so the scene\'s camera has the '
                             'lens\'s pinhole part (fx, fy, cx, cy); renders remap a wider pinhole through the lens '
                             'model.')
            out.append(SceneCamera(name, s.uid, s.name, im.name, im.role, bname, T_bs, T_so, im.camera, im.label,
                                   im is prim))
    return out


# ====================================================================== the scene
VIEW_CAMERA = 'dexvantage_view'    # a world-body camera the Renderer poses anywhere (render_pose); not saved


def build_scene(project, variant=None, objects: bool = True, lights: bool = True,
                view_camera: bool = False) -> SceneBuild:
    """The project (variant: a Variant or its uid; default the active one) as a compiled MuJoCo scene (module
    docstring). Robots that cannot be read are left out with a note. view_camera adds VIEW_CAMERA (for renders
    from any pose; save_scene leaves it out)."""
    import mujoco

    from ..robot.model import RobotModelError, get_model
    if variant is None:
        variant = project.variant
    elif isinstance(variant, str):
        variant = project.variant_by_uid(variant) or project.variant
    notes, defaults = [], []
    S = mujoco.MjSpec()
    S.modelname = _safe(project.name or 'dexvantage_scene')
    S.compiler.fusestatic = False
    assets, robots, rob_bodies = {}, {}, {}
    radius = 1.0
    for k, r in enumerate(project.robots):
        prefix = f'robot{k}_'
        try:
            rm = get_model(project, r.uid)
            child, a, n = robot_spec(project, r, prefix)
        except RobotModelError as e:
            notes.append(f'{r.name} is not in the scene: {e}')
            continue
        notes += n
        _match_options(child, S)
        pos, qm = _frame_pq(r.base.matrix())
        fr = S.worldbody.add_frame(pos=pos, quat=qm)
        try:
            S.attach(child, frame=fr, prefix=prefix)
        except Exception as e:                       # noqa: BLE001 — explained
            notes.append(f'{r.name} is not in the scene: MuJoCo could not attach it ({str(e).splitlines()[0]}).')
            continue
        assets.update(a)
        robots[r.uid] = {'prefix': prefix, 'name': r.name, 'project': project, 'map': []}
        rob_bodies[r.uid] = prefix
        try:
            lo, hi = rm.bounds()
            radius = max(radius, float(np.max(np.abs(np.r_[lo, hi]))))
        except Exception:                            # noqa: BLE001 — only sizes the extent
            pass
    obj_ok = []
    if objects:
        for o in project.workspace.objects:
            if _add_object(S, project, o, notes):
                obj_ok.append(o)
                radius = max(radius, float(np.max(np.abs(o.pose.position))) + float(np.max(np.abs(o.size))))
    surf = []
    for k, s in enumerate(project.workspace.surfaces):
        if isinstance(s, dict):
            uid = _add_surface(S, s, k, notes, defaults)
            if uid is not None:
                surf.append(uid)
    _add_lights(S, project, lights, notes, defaults)
    cams = _sensor_cameras(S, project, variant, rob_bodies, notes, defaults)
    if view_camera:
        S.worldbody.add_camera(name=VIEW_CAMERA, pos=[0.0, 0.0, 0.0])
    for c in cams:
        if c.body == 'world':
            radius = max(radius, float(np.max(np.abs(c.T_body_sensor[:3, 3]))))
    extent = float(np.ceil(2.0 * radius * 10) / 10)
    S.stat.extent = extent
    S.visual.map.znear = NEAR_M / extent
    S.visual.map.zfar = FAR_M / extent
    big = max([c.model.width for c in cams] + [640])
    S.visual.global_.offwidth = int(big)
    S.visual.global_.offheight = int(max([c.model.height for c in cams] + [480]))
    S.assets = assets
    try:
        m = S.compile()
    except Exception as e:                           # noqa: BLE001 — a mesh object MuJoCo refuses, explained
        msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
        bad = next((o for o in obj_ok if o.shape == 'mesh' and f'mesh_object_{o.uid}' in msg), None)
        if bad is None:
            raise RuntimeError(f'MuJoCo could not compile the scene: {msg}.') from None
        notes.append(f'Object {bad.name} is left out: MuJoCo refused its mesh ({msg}).')
        S.delete(S.body(f'object_{bad.uid}'))
        S.delete(S.mesh(f'mesh_object_{bad.uid}'))
        m = S.compile()
    build = SceneBuild(S, notes=notes, defaults=defaults, robots=robots)
    build.model = m
    _owners(build, project, m, surf)
    for c in cams:
        build.cameras[(c.sensor_uid, c.imager)] = c
        if c.primary:
            build.camera_names[c.sensor_uid] = c.name
            build.imagers[c.sensor_uid] = (c.imager, c.model)
    _joint_maps(build, project, m)
    return build


def _owners(build: SceneBuild, project, m, surfaces):
    """Body owners, the robot geoms the predictions do not draw moved to HIDDEN_GROUP (in the model and the spec),
    and the geom -> owner index table."""
    from ..robot import model as RM
    G_MESH = 7                                       # mujoco.mjtGeom.mjGEOM_MESH
    owners, index = [], {}

    def own(o) -> int:
        if o not in index:
            index[o] = len(owners)
            owners.append(o)
        return index[o]
    prefixes = {r['prefix']: uid for uid, r in build.robots.items()}
    links = {}
    for uid in build.robots:
        rm = RM.get_model(project, uid)
        links[uid] = {rm.body_link[b]: rm.link_names[i] for b, i in rm._body_own.items()}
    for b in range(1, m.nbody):
        name = m.body(b).name
        o = None
        for pre, uid in prefixes.items():
            if name.startswith(pre):
                ln = links[uid].get(name[len(pre):])
                if ln is not None:
                    o = ('robot', uid, ln)
                break
        if o is None and name.startswith('object_'):
            o = ('object', name[len('object_'):])
        if o is None and name.startswith('surface_'):
            o = ('surface', name[len('surface_'):])
        if o is not None:
            build.body_map[b] = o
            build.body_map[name] = o
    spec_geoms = {int(g.id): g for g in build.spec.geoms}
    gown = np.full(m.ngeom, -1, np.int32)
    body_robot = {}
    for b, o in build.body_map.items():
        if isinstance(b, int) and o[0] == 'robot':
            body_robot[b] = o[1]
    nonc = (m.geom_contype == 0) & (m.geom_conaffinity == 0)
    hidden = 0
    for uid in build.robots:
        gs = [g for g in range(m.ngeom) if body_robot.get(int(m.geom_bodyid[g])) == uid]
        has_visual = bool(np.any(nonc[gs])) if gs else False
        for g in gs:
            rgba = np.asarray(m.geom_rgba[g], float)
            mat = int(m.geom_matid[g])
            if mat >= 0 and np.allclose(rgba, RM._DEFAULT_RGBA):
                rgba = np.asarray(m.mat_rgba[mat], float)
            t = int(m.geom_type[g])
            drawable = t == G_MESH or RM.primitive_mesh(t, m.geom_size[g]) is not None
            if (nonc[g] or not has_visual) and rgba[3] > 0 and drawable:
                gown[g] = own(build.body_map[int(m.geom_bodyid[g])])
            else:
                m.geom_group[g] = HIDDEN_GROUP
                if g in spec_geoms:
                    spec_geoms[g].group = HIDDEN_GROUP
                hidden += 1
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        o = build.body_map.get(b)
        if o is not None and o[0] in ('object', 'surface'):
            gown[g] = own(o)
    for k in range(m.nsite):                         # sites and tendons are drawn by default: never in a render
        m.site_group[k] = HIDDEN_GROUP
    for k in range(m.ntendon):
        m.tendon_group[k] = HIDDEN_GROUP
    for x in list(build.spec.sites):
        x.group = HIDDEN_GROUP
    for x in list(build.spec.tendons):
        x.group = HIDDEN_GROUP
    if hidden:
        build.notes.append(f'{hidden} robot geom(s) the predictions do not draw (collision-only, invisible or a type '
                           f'they cannot draw) are in geom group {HIDDEN_GROUP}, which the renders leave out.')
    build.owners = owners
    build.geom_owner = gown


def _joint_maps(build: SceneBuild, project, m):
    from ..robot.model import get_model
    for uid, r in build.robots.items():
        rm = get_model(project, uid)
        mp, missing = [], 0
        for j in range(rm.m.njnt):
            nm = rm.m.joint(j).name
            if not nm:
                missing += 1
                continue
            try:
                js = m.joint(r['prefix'] + nm).id
            except KeyError:
                missing += 1
                continue
            t = int(rm.m.jnt_type[j])
            n = {0: 7, 1: 4}.get(t, 1)
            mp.append((int(rm.m.jnt_qposadr[j]), int(m.jnt_qposadr[js]), n))
        r['map'] = mp
        if missing:
            build.notes.append(f'{r["name"]}: {missing} unnamed joint(s) stay at the scene\'s qpos0.')


# ====================================================================== the saved scene
def save_scene(project, folder: str, variant=None) -> str:
    """Write the scene as a self-contained MJCF: folder/scene.xml, its mesh and texture files in folder/assets/, and
    folder/scene_notes.txt (what the scene holds, its defaults and notes). MuJoCo loads it with
    mujoco.MjModel.from_xml_path(path). Returns the path of scene.xml.

    MuJoCo's XML writer (MjSpec.to_xml, 3.17.0) writes numbers with 6 significant digits, so the file's camera
    poses and intrinsics are rounded to that (measured: within 6 × 10⁻⁷ m, 1.2 × 10⁻⁶ rad and 2 × 10⁻³ px of the
    project's; check formats.mjcf_scene); the in-memory scene the renders use is not rounded."""
    import mujoco
    b = build_scene(project, variant)
    os.makedirs(os.path.join(folder, 'assets'), exist_ok=True)
    for name, data in b.spec.assets.items():
        with open(os.path.join(folder, 'assets', name), 'wb') as f:
            f.write(data)
    b.spec.meshdir = 'assets'
    b.spec.texturedir = 'assets'
    xml = b.spec.to_xml()
    path = os.path.join(folder, 'scene.xml')
    tmp = path + '.part'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(xml)
    os.replace(tmp, path)
    mujoco.MjModel.from_xml_path(path)               # it must load as written
    v = project.variant_by_uid(variant) if isinstance(variant, str) else variant
    lines = [f'MuJoCo scene of the DexVantage project "{project.name}", rig variant "{(v or project.variant).name}"',
             '', 'Cameras (MuJoCo camera: sensor, imager, lens; MuJoCo cameras are pinhole):']
    for c in b.cameras.values():
        lines.append(f'  {c.name}: {c.sensor_name}, {c.imager}, {c.model.summary()}')
    lines += ['', f'Robot geoms the predictions do not draw are in geom group {HIDDEN_GROUP} (not drawn by default).',
              '', 'Defaults:'] + [f'  {d}' for d in b.defaults] + ['', 'Notes:'] + [f'  {n}' for n in b.notes]
    with open(os.path.join(folder, 'scene_notes.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return path
