"""The project: robots, sensors and their mounts, the workspace, tasks, variants, requirements and the design log, and
its file (.dexv).

A .dexv file is a zip archive:

    project.json      the project (schema docs/project.schema.json, written up in docs/project-format.md)
    meshes.npz        mesh arrays the project holds (robot link meshes, task object meshes), by key
    sources/...       the original model files the robots and meshes were read from, for provenance
    thumb.png         a picture of the 3D view when it was saved (optional)

project.json carries {"format": "dexvantage.project", "version": N}. Reading a file of an older version runs the
migrations in MIGRATIONS one version at a time; a newer version than this build knows is refused with a message that
names both versions.

Units are SI and frames follow REP-103 (core/frames.py). Every object that can be referred to has a uid: ten random
hex digits, a stable identity across edits and saves.

The robots' kinematic trees, the sensor library and the studies arrive in later phases (docs/BUILD_STATE.md). Phase 0
defines the containers and the file so the shell, the undo stack and the provenance blocks have something real to
work on.
"""
from __future__ import annotations

import copy
import datetime as _dt
import io
import json
import os
import secrets
import zipfile
from dataclasses import dataclass, field

import numpy as np

from .frames import Pose

FORMAT = 'dexvantage.project'
VERSION = 1
EXTENSION = '.dexv'
SECTIONS = ('meta', 'robots', 'workspace', 'tasks', 'variants', 'requirements', 'analysis')

SENSOR_KINDS = ('camera', 'depth', 'tactile')
VOLUME_KINDS = ('tabletop', 'shelf', 'bin', 'handover', 'custom')
MATERIAL_TAGS = ('matte', 'glossy', 'metallic', 'transparent', 'black', 'ir_absorbing')
MOUNT_STATIC = ('world', 'cage', 'tripod', 'table')


class ProjectFormatError(ValueError):
    """A project file DexVantage cannot read; the message says why in plain words."""


def new_uid() -> str:
    return secrets.token_hex(5)


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _pick(d, cls_fields: dict) -> dict:
    """The keys of d that a dataclass knows, with their defaults where d lacks them (unknown keys are dropped)."""
    d = d if isinstance(d, dict) else {}
    return {k: d.get(k, v) for k, v in cls_fields.items()}


# ====================================================================== robots
@dataclass
class RobotSource:
    kind: str = ''                 # 'urdf' | 'mjcf' | 'usd' | '' (none yet)
    file: str = ''                 # the file's name inside the archive's sources/ folder
    sha256: str = ''               # of the original file's bytes
    license: str = ''              # as recorded in THIRD_PARTY.md or stated by the user on import
    note: str = ''

    def to_dict(self):
        return dict(kind=self.kind, file=self.file, sha256=self.sha256, license=self.license, note=self.note)

    @staticmethod
    def from_dict(d):
        return RobotSource(**_pick(d, RobotSource().to_dict()))


@dataclass
class Robot:
    uid: str = field(default_factory=new_uid)
    name: str = 'Robot'
    source: RobotSource = field(default_factory=RobotSource)
    base: Pose = field(default_factory=Pose)          # the robot's base in the world frame (the primary's is identity)
    primary: bool = False
    # filled by the importers (Phase 1): plain records so the file stays readable without the kinematics code
    links: list = field(default_factory=list)         # [{'name', 'parent_joint', 'mesh_keys', 'collision'}]
    joints: list = field(default_factory=list)        # [{'name', 'type', 'parent', 'child', 'axis', 'origin',
                                                      #   'lower', 'upper', 'velocity', 'effort'}] radians / metres
    home: dict = field(default_factory=dict)          # joint name -> value
    gripper_frame: str = ''
    fingertip_frames: list = field(default_factory=list)
    contact_pads: list = field(default_factory=list)  # [{'link', 'name', 'faces' or 'box'}]
    locked: bool = False

    def to_dict(self):
        return {'uid': self.uid, 'name': self.name, 'source': self.source.to_dict(), 'base': self.base.to_dict(),
                'primary': self.primary, 'links': copy.deepcopy(self.links), 'joints': copy.deepcopy(self.joints),
                'home': dict(self.home), 'gripper_frame': self.gripper_frame,
                'fingertip_frames': list(self.fingertip_frames), 'contact_pads': copy.deepcopy(self.contact_pads),
                'locked': self.locked}

    @staticmethod
    def from_dict(d):
        d = d if isinstance(d, dict) else {}
        r = Robot(uid=str(d.get('uid') or new_uid()), name=str(d.get('name', 'Robot')))
        r.source = RobotSource.from_dict(d.get('source'))
        r.base = Pose.from_dict(d.get('base'))
        r.primary = bool(d.get('primary', False))
        r.links = list(d.get('links') or [])
        r.joints = list(d.get('joints') or [])
        r.home = {str(k): float(v) for k, v in (d.get('home') or {}).items()}
        r.gripper_frame = str(d.get('gripper_frame') or '')
        r.fingertip_frames = [str(x) for x in d.get('fingertip_frames') or []]
        r.contact_pads = list(d.get('contact_pads') or [])
        r.locked = bool(d.get('locked', False))
        return r


# ====================================================================== sensors and mounts
@dataclass
class Tolerance:
    position_mm: float = 1.0       # default, labelled so: the user sets their own
    rotation_deg: float = 0.5

    def to_dict(self):
        return {'position_mm': self.position_mm, 'rotation_deg': self.rotation_deg}

    @staticmethod
    def from_dict(d):
        return Tolerance(**{k: float(v) for k, v in _pick(d, Tolerance().to_dict()).items()})


@dataclass
class Mount:
    parent_frame: str = 'world'    # 'world', a static frame (cage, tripod, table) or a link name
    robot_uid: str = ''            # the robot whose link parent_frame names; '' for static frames
    pose: Pose = field(default_factory=Pose)          # the sensor body frame in the parent frame
    tolerance: Tolerance = field(default_factory=Tolerance)
    bracket: str = ''              # a bracket type id (Phase 6)

    @property
    def on_link(self) -> bool:
        return bool(self.robot_uid) and self.parent_frame not in MOUNT_STATIC

    def to_dict(self):
        return {'parent_frame': self.parent_frame, 'robot_uid': self.robot_uid, 'pose': self.pose.to_dict(),
                'tolerance': self.tolerance.to_dict(), 'bracket': self.bracket}

    @staticmethod
    def from_dict(d):
        d = d if isinstance(d, dict) else {}
        return Mount(str(d.get('parent_frame') or 'world'), str(d.get('robot_uid') or ''),
                     Pose.from_dict(d.get('pose')),
                     Tolerance.from_dict(d.get('tolerance')), str(d.get('bracket') or ''))


@dataclass
class Sensor:
    uid: str = field(default_factory=new_uid)
    name: str = 'Camera'
    kind: str = 'camera'           # SENSOR_KINDS
    model_id: str = ''             # a library entry (Phase 2)
    calibration: dict = field(default_factory=dict)   # the unit's own intrinsics, when calibrated
    mount: Mount = field(default_factory=Mount)
    slot: int = 0                  # its colour (theme CAMERA palette)
    locked: bool = False
    enabled: bool = True

    def to_dict(self):
        return {'uid': self.uid, 'name': self.name, 'kind': self.kind, 'model_id': self.model_id,
                'calibration': copy.deepcopy(self.calibration), 'mount': self.mount.to_dict(), 'slot': self.slot,
                'locked': self.locked, 'enabled': self.enabled}

    @staticmethod
    def from_dict(d):
        d = d if isinstance(d, dict) else {}
        kind = str(d.get('kind', 'camera'))
        return Sensor(str(d.get('uid') or new_uid()), str(d.get('name', 'Camera')),
                      kind if kind in SENSOR_KINDS else 'camera', str(d.get('model_id') or ''),
                      dict(d.get('calibration') or {}), Mount.from_dict(d.get('mount')), int(d.get('slot', 0)),
                      bool(d.get('locked', False)), bool(d.get('enabled', True)))


@dataclass
class Variant:
    """A whole-rig alternative: its sensors and their mounts."""
    uid: str = field(default_factory=new_uid)
    name: str = 'Rig A'
    sensors: list = field(default_factory=list)       # [Sensor]
    note: str = ''
    created: str = field(default_factory=_utc_now)

    def sensor(self, uid) -> Sensor | None:
        return next((s for s in self.sensors if s.uid == uid), None)

    def to_dict(self):
        return {'uid': self.uid, 'name': self.name, 'sensors': [s.to_dict() for s in self.sensors],
                'note': self.note, 'created': self.created}

    @staticmethod
    def from_dict(d):
        d = d if isinstance(d, dict) else {}
        return Variant(str(d.get('uid') or new_uid()), str(d.get('name', 'Rig A')),
                       [Sensor.from_dict(s) for s in d.get('sensors') or []], str(d.get('note') or ''),
                       str(d.get('created') or _utc_now()))


# ====================================================================== workspace
@dataclass
class Volume:
    uid: str = field(default_factory=new_uid)
    name: str = 'Tabletop'
    kind: str = 'tabletop'         # VOLUME_KINDS
    center: tuple = (0.45, 0.0, 0.15)                 # m, world frame
    size: tuple = (0.5, 0.6, 0.3)                     # m, an axis-aligned box in the world frame
    refine: bool = False           # a refinement region: the finer voxel size applies inside (Phase 2)

    def to_dict(self):
        return {'uid': self.uid, 'name': self.name, 'kind': self.kind, 'center': [float(v) for v in self.center],
                'size': [float(v) for v in self.size], 'refine': self.refine}

    @staticmethod
    def from_dict(d):
        d = d if isinstance(d, dict) else {}
        kind = str(d.get('kind', 'tabletop'))
        return Volume(str(d.get('uid') or new_uid()), str(d.get('name', 'Volume')),
                      kind if kind in VOLUME_KINDS else 'custom', tuple(float(v) for v in d.get('center', (0, 0, 0))),
                      tuple(max(0.0, float(v)) for v in d.get('size', (0.1, 0.1, 0.1))), bool(d.get('refine', False)))

    def bounds(self):
        c, s = np.asarray(self.center, float), np.asarray(self.size, float)
        return c - s / 2, c + s / 2


@dataclass
class TaskObject:
    uid: str = field(default_factory=new_uid)
    name: str = 'Object'
    shape: str = 'box'             # 'box' | 'cylinder' | 'sphere' | 'mesh'
    size: tuple = (0.05, 0.05, 0.05)                  # box edges, or (radius, height, -), (radius, -, -)
    pose: Pose = field(default_factory=Pose)
    mesh_key: str = ''             # into meshes.npz for shape 'mesh'
    materials: list = field(default_factory=lambda: ['matte'])   # MATERIAL_TAGS

    def to_dict(self):
        return {'uid': self.uid, 'name': self.name, 'shape': self.shape, 'size': [float(v) for v in self.size],
                'pose': self.pose.to_dict(), 'mesh_key': self.mesh_key, 'materials': list(self.materials)}

    @staticmethod
    def from_dict(d):
        d = d if isinstance(d, dict) else {}
        return TaskObject(str(d.get('uid') or new_uid()), str(d.get('name', 'Object')), str(d.get('shape', 'box')),
                          tuple(float(v) for v in d.get('size', (0.05, 0.05, 0.05))), Pose.from_dict(d.get('pose')),
                          str(d.get('mesh_key') or ''),
                          [m for m in (d.get('materials') or ['matte']) if m in MATERIAL_TAGS])


@dataclass
class Workspace:
    volumes: list = field(default_factory=list)       # [Volume]
    objects: list = field(default_factory=list)       # [TaskObject]
    lights: list = field(default_factory=list)        # [{'uid', 'name', 'kind', 'pose', ...}] (Phase 5)
    surfaces: list = field(default_factory=list)      # [{'uid', 'name', 'z', 'size', 'material'}]
    background: str = 'studio'

    def to_dict(self):
        return {'volumes': [v.to_dict() for v in self.volumes], 'objects': [o.to_dict() for o in self.objects],
                'lights': copy.deepcopy(self.lights), 'surfaces': copy.deepcopy(self.surfaces),
                'background': self.background}

    @staticmethod
    def from_dict(d):
        d = d if isinstance(d, dict) else {}
        return Workspace([Volume.from_dict(v) for v in d.get('volumes') or []],
                         [TaskObject.from_dict(o) for o in d.get('objects') or []],
                         list(d.get('lights') or []), list(d.get('surfaces') or []),
                         str(d.get('background') or 'studio'))


# ====================================================================== tasks, requirements, analysis
@dataclass
class Tasks:
    grasp_sets: list = field(default_factory=list)    # [{'uid', 'name', 'source', 'poses': [...]}] (Phase 3)
    trajectories: list = field(default_factory=list)  # [{'uid', 'name', 'robot_uid', 'joints', 't', 'q'}]
    teleop_devices: list = field(default_factory=list)

    def to_dict(self):
        return {'grasp_sets': copy.deepcopy(self.grasp_sets), 'trajectories': copy.deepcopy(self.trajectories),
                'teleop_devices': copy.deepcopy(self.teleop_devices)}

    @staticmethod
    def from_dict(d):
        d = d if isinstance(d, dict) else {}
        return Tasks(list(d.get('grasp_sets') or []), list(d.get('trajectories') or []),
                     list(d.get('teleop_devices') or []))


@dataclass
class Requirement:
    uid: str = field(default_factory=new_uid)
    name: str = ''
    metric: str = ''               # a study's metric id, e.g. 'grasp_visibility.score'
    op: str = '>='                 # '>=' | '<=' | '<' | '>' | '=='
    threshold: float = 0.0
    unit: str = ''
    study: str = ''
    is_default: bool = True        # the threshold is DexVantage's documented default, not the user's own figure
    note: str = ''

    def to_dict(self):
        return {'uid': self.uid, 'name': self.name, 'metric': self.metric, 'op': self.op,
                'threshold': float(self.threshold), 'unit': self.unit, 'study': self.study,
                'is_default': self.is_default, 'note': self.note}

    @staticmethod
    def from_dict(d):
        d = d if isinstance(d, dict) else {}
        op = str(d.get('op', '>='))
        return Requirement(str(d.get('uid') or new_uid()), str(d.get('name', '')), str(d.get('metric', '')),
                           op if op in ('>=', '<=', '<', '>', '==') else '>=', float(d.get('threshold', 0.0)),
                           str(d.get('unit', '')), str(d.get('study', '')), bool(d.get('is_default', True)),
                           str(d.get('note', '')))


@dataclass
class AnalysisSettings:
    voxel_mm: float = 5.0          # default coverage voxel (§7.1)
    refine_mm: float = 2.0         # inside refinement regions
    live: bool = True              # re-run quick studies after each edit (a view toggle: not part of the content hash)
    studies: dict = field(default_factory=dict)       # study id -> its own settings

    def to_dict(self):
        return {'voxel_mm': self.voxel_mm, 'refine_mm': self.refine_mm, 'live': self.live,
                'studies': copy.deepcopy(self.studies)}

    @staticmethod
    def from_dict(d):
        d = d if isinstance(d, dict) else {}
        return AnalysisSettings(float(d.get('voxel_mm', 5.0)), float(d.get('refine_mm', 2.0)),
                                bool(d.get('live', True)), dict(d.get('studies') or {}))


# ====================================================================== the project
class Project:
    def __init__(self, name: str = 'Untitled'):
        self.name = name
        self.path = ''                                 # where it was read from or saved to ('' = never saved)
        self.created = _utc_now()
        self.saved_at = ''
        self.robots: list[Robot] = []
        self.workspace = Workspace()
        self.tasks = Tasks()
        self.variants: list[Variant] = [Variant()]
        self.active_uid = self.variants[0].uid
        self.requirements: list[Requirement] = []
        self.analysis = AnalysisSettings()
        self.notes = ''
        self.log: list[dict] = []                      # the design log: [{'t', 'kind', 'text'}]
        self.meshes: dict[str, np.ndarray] = {}        # key -> array (meshes.npz)
        self.sources: dict[str, bytes] = {}            # file name -> original bytes (sources/)
        self.thumb: bytes = b''

    # ---------------------------------------------------------------- access
    @property
    def variant(self) -> Variant:
        v = self.variant_by_uid(self.active_uid)
        if v is None:
            v = self.variants[0]
            self.active_uid = v.uid
        return v

    def variant_by_uid(self, uid) -> Variant | None:
        return next((v for v in self.variants if v.uid == uid), None)

    @property
    def primary_robot(self) -> Robot | None:
        return next((r for r in self.robots if r.primary), self.robots[0] if self.robots else None)

    def robot(self, uid) -> Robot | None:
        return next((r for r in self.robots if r.uid == uid), None)

    def add_log(self, kind: str, text: str):
        self.log.append({'t': _utc_now(), 'kind': kind, 'text': text})
        del self.log[:-2000]

    def snapshot(self) -> 'Project':
        """A deep copy for a worker thread (arrays and source bytes are shared: they are never changed in place)."""
        p = Project.from_dict(self.to_dict())
        p.path, p.meshes, p.sources, p.thumb = self.path, dict(self.meshes), dict(self.sources), self.thumb
        return p

    # ---------------------------------------------------------------- sections (the undo stack's unit)
    def section(self, key: str):
        if key == 'meta':
            return {'name': self.name, 'notes': self.notes}
        if key == 'robots':
            return [r.to_dict() for r in self.robots]
        if key == 'workspace':
            return self.workspace.to_dict()
        if key == 'tasks':
            return self.tasks.to_dict()
        if key == 'variants':
            return {'variants': [v.to_dict() for v in self.variants], 'active_uid': self.active_uid}
        if key.startswith('variant:'):
            v = self.variant_by_uid(key.split(':', 1)[1])
            return v.to_dict() if v is not None else None
        if key == 'requirements':
            return [r.to_dict() for r in self.requirements]
        if key == 'analysis':
            return self.analysis.to_dict()
        raise KeyError(key)

    def restore(self, key: str, state):
        if key == 'meta':
            self.name, self.notes = state['name'], state['notes']
        elif key == 'robots':
            self.robots = [Robot.from_dict(d) for d in state]
        elif key == 'workspace':
            self.workspace = Workspace.from_dict(state)
        elif key == 'tasks':
            self.tasks = Tasks.from_dict(state)
        elif key == 'variants':
            self.variants = [Variant.from_dict(d) for d in state['variants']] or [Variant()]
            self.active_uid = state['active_uid']
        elif key.startswith('variant:'):
            uid = key.split(':', 1)[1]
            for i, v in enumerate(self.variants):
                if v.uid == uid:
                    if state is None:
                        del self.variants[i]
                    else:
                        self.variants[i] = Variant.from_dict(state)
                    break
            else:
                if state is not None:
                    self.variants.append(Variant.from_dict(state))
        elif key == 'requirements':
            self.requirements = [Requirement.from_dict(d) for d in state]
        elif key == 'analysis':
            self.analysis = AnalysisSettings.from_dict(state)
        else:
            raise KeyError(key)

    # ---------------------------------------------------------------- JSON
    def to_dict(self) -> dict:
        return {
            'format': FORMAT, 'version': VERSION,
            'name': self.name, 'notes': self.notes, 'created': self.created, 'saved_at': self.saved_at,
            'robots': [r.to_dict() for r in self.robots],
            'workspace': self.workspace.to_dict(),
            'tasks': self.tasks.to_dict(),
            'variants': [v.to_dict() for v in self.variants], 'active_uid': self.active_uid,
            'requirements': [r.to_dict() for r in self.requirements],
            'analysis': self.analysis.to_dict(),
            'log': list(self.log),
            'meshes': sorted(self.meshes), 'sources': sorted(self.sources),
        }

    @staticmethod
    def from_dict(d: dict) -> 'Project':
        d = migrate(d)
        p = Project(str(d.get('name') or 'Untitled'))
        p.notes = str(d.get('notes') or '')
        p.created = str(d.get('created') or p.created)
        p.saved_at = str(d.get('saved_at') or '')
        p.robots = [Robot.from_dict(r) for r in d.get('robots') or []]
        p.workspace = Workspace.from_dict(d.get('workspace'))
        p.tasks = Tasks.from_dict(d.get('tasks'))
        p.variants = [Variant.from_dict(v) for v in d.get('variants') or []] or [Variant()]
        p.active_uid = str(d.get('active_uid') or p.variants[0].uid)
        p.requirements = [Requirement.from_dict(r) for r in d.get('requirements') or []]
        p.analysis = AnalysisSettings.from_dict(d.get('analysis'))
        p.log = [e for e in d.get('log') or [] if isinstance(e, dict)]
        return p

    # ---------------------------------------------------------------- the file
    def save(self, path: str, thumb: bytes | None = None):
        """Write the .dexv archive atomically (a .part file renamed over the old one)."""
        if thumb is not None:
            self.thumb = thumb
        self.saved_at = _utc_now()
        doc = self.to_dict()
        tmp = path + '.part'
        with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('project.json', json.dumps(doc, indent=1, ensure_ascii=False) + '\n')
            if self.meshes:
                buf = io.BytesIO()
                np.savez_compressed(buf, **self.meshes)
                z.writestr('meshes.npz', buf.getvalue())
            for name, data in sorted(self.sources.items()):
                z.writestr(f'sources/{name}', data)
            if self.thumb:
                z.writestr('thumb.png', self.thumb)
        os.replace(tmp, path)
        self.path = path

    @staticmethod
    def load(path: str) -> 'Project':
        try:
            z = zipfile.ZipFile(path)
        except FileNotFoundError:
            raise ProjectFormatError(f'There is no file at {os.path.basename(path)}.') from None
        except zipfile.BadZipFile:
            raise ProjectFormatError(f'{os.path.basename(path)} is not a DexVantage project: a .dexv file is a zip '
                                     'archive, and this one is not.') from None
        with z:
            names = set(z.namelist())
            if 'project.json' not in names:
                raise ProjectFormatError(f'{os.path.basename(path)} has no project.json inside, so it is not a '
                                         'DexVantage project.')
            try:
                doc = json.loads(z.read('project.json').decode('utf-8'))
            except (UnicodeDecodeError, ValueError) as e:
                raise ProjectFormatError(f'The project.json inside {os.path.basename(path)} is damaged: {e}.') from None
            p = Project.from_dict(doc)
            if 'meshes.npz' in names:
                with np.load(io.BytesIO(z.read('meshes.npz')), allow_pickle=False) as npz:
                    p.meshes = {k: npz[k] for k in npz.files}
            p.sources = {n[len('sources/'):]: z.read(n) for n in names
                         if n.startswith('sources/') and not n.endswith('/')}
            p.thumb = z.read('thumb.png') if 'thumb.png' in names else b''
        p.path = path
        return p


# ====================================================================== migrations
# version n -> fn(doc) returning the doc at version n + 1. Version 1 is the first; none are needed yet.
MIGRATIONS: dict = {}


def migrate(doc: dict) -> dict:
    if not isinstance(doc, dict):
        raise ProjectFormatError('The project file holds no project (its JSON is not an object).')
    fmt = doc.get('format', FORMAT)
    if fmt != FORMAT:
        raise ProjectFormatError(f'This file is a "{fmt}" document, not a DexVantage project.')
    try:
        v = int(doc.get('version', VERSION))
    except (TypeError, ValueError):
        raise ProjectFormatError('The project file\'s version is not a number.') from None
    if v > VERSION:
        raise ProjectFormatError(f'This project was saved by a newer DexVantage (file version {v}); this build reads '
                                 f'up to version {VERSION}. Update DexVantage to open it.')
    doc = dict(doc)
    while v < VERSION:
        fn = MIGRATIONS.get(v)
        if fn is None:
            raise ProjectFormatError(f'No migration from project file version {v}.')
        doc = fn(doc)
        v += 1
        doc['version'] = v
    return doc


def blank_project(name: str = 'Untitled') -> Project:
    """An empty project with one rig variant and a default tabletop task volume (labelled a default)."""
    p = Project(name)
    p.workspace.volumes.append(Volume(name='Tabletop (default)'))
    p.add_log('project', 'New project')
    return p
