"""Named, settable quantities of a project ("factors"): the one way the tolerance Monte Carlo, designed experiments,
the optimiser, the design agent and the Python API address what they vary.

Paths (a sensor by its uid or, when unique, its name; the active variant unless variant= is given):

    sensor:<id>.x | .y | .z           mount position in the parent frame, metres
    sensor:<id>.roll | .pitch | .yaw  mount orientation in the parent frame, degrees (URDF rpy: R = Rz Ry Rx)
    sensor:<id>.dx | .dy | .dz        a shift of the mount along the parent frame's axes, metres (get() reads 0)
    sensor:<id>.droll | .dpitch | .dyaw
                                      a small rotation about the sensor body's own x, y, z axes, degrees (get() 0)
    sensor:<id>.model                 library id (categorical)
    sensor:<id>.parent                mount parent frame name (categorical)
    sensor:<id>.enabled               bool
    sensor:<id>.stream                [width, height] of the stream mode ([] = the largest)
    setting:<study>.<key>             project.analysis.studies[study][key]
    json:/a/b/0                       any JSON pointer into project.to_dict() (fallback; applied through from_dict)

    get(project, path)                         -> current value
    apply(project, {path: value, ...})         -> a NEW project (the input is never changed)
    describe(path, project=None)               -> {'path', 'title', 'unit', 'kind': 'continuous'|'categorical'|'bool'|
                                                  'list', 'sensor_uid'?, 'locked'?}
    FactorError                                a plain sentence: unknown path, unknown sensor, a locked sensor

Locked sensors are the user's: apply() refuses to change one unless allow_locked=True (the tolerance Monte Carlo
perturbs locked mounts too, because a locked mount is still built with tolerances). A json: pointer obeys the same
lock: a pointer edit that changes, removes or unlocks a locked sensor of any variant is refused.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from .frames import Pose, quat_from_rpy, quat_multiply, rpy_from_quat
from .jobs import StudyError

POSITION = ('x', 'y', 'z')
ANGLES = ('roll', 'pitch', 'yaw')
SHIFTS = ('dx', 'dy', 'dz')
TURNS = ('droll', 'dpitch', 'dyaw')
SENSOR_FIELDS = POSITION + ANGLES + SHIFTS + TURNS + ('model', 'parent', 'enabled', 'stream')


class FactorError(StudyError):
    """A factor path that cannot be read or set; the message says why in plain words."""


def _parse(path: str):
    if not isinstance(path, str) or ':' not in path:
        raise FactorError(f'"{path}" is not a factor path (it starts with sensor:, setting: or json:).')
    kind, rest = path.split(':', 1)
    if kind == 'sensor':
        sid, _, fld = rest.rpartition('.')
        if not sid or fld not in SENSOR_FIELDS:
            raise FactorError(f'"{path}" names no sensor quantity (one of: {", ".join(SENSOR_FIELDS)}).')
        return kind, sid, fld
    if kind == 'setting':
        study, _, key = rest.partition('.')
        if not study or not key:
            raise FactorError(f'"{path}" needs a study and a setting, as setting:<study>.<key>.')
        return kind, study, key
    if kind == 'json':
        if not rest.startswith('/'):
            raise FactorError(f'"{path}" needs a JSON pointer starting with /.')
        return kind, rest, None
    raise FactorError(f'"{path}" is not a factor path (it starts with sensor:, setting: or json:).')


def _variant(project, variant):
    if variant is None:
        return project.variant
    v = project.variant_by_uid(variant)
    if v is None:
        raise FactorError(f'The project has no variant {variant}.')
    return v


def find_sensor(project, ident: str, variant=None):
    sensors = _variant(project, variant).sensors
    hit = [s for s in sensors if s.uid == ident]
    if not hit:
        hit = [s for s in sensors if s.name == ident]
        if len(hit) > 1:
            raise FactorError(f'More than one sensor is called "{ident}": name it by its uid.')
    if not hit:
        raise FactorError(f'The rig has no sensor "{ident}".')
    return hit[0]


def _pointer(doc, ptr: str, value=None, set_=False):
    parts = [p.replace('~1', '/').replace('~0', '~') for p in ptr.lstrip('/').split('/')]
    cur = doc
    for i, p in enumerate(parts):
        last = i == len(parts) - 1
        if isinstance(cur, list):
            try:
                k = int(p)
                cur[k]
            except (ValueError, IndexError):
                raise FactorError(f'The project has nothing at {ptr}.') from None
        elif isinstance(cur, dict):
            if p not in cur and not (set_ and last):
                raise FactorError(f'The project has nothing at {ptr}.')
            k = p
        else:
            raise FactorError(f'The project has nothing at {ptr}.')
        if last:
            if set_:
                cur[k] = value
                return None
            return cur[k]
        cur = cur[k]


def get(project, path: str, variant=None):
    kind, a, b = _parse(path)
    if kind == 'sensor':
        s = find_sensor(project, a, variant)
        p = s.mount.pose
        if b in POSITION:
            return float(p.position[POSITION.index(b)])
        if b in ANGLES:
            return math.degrees(rpy_from_quat(p.quaternion)[ANGLES.index(b)])
        if b in SHIFTS + TURNS:
            return 0.0
        return {'model': s.model_id, 'parent': s.mount.parent_frame, 'enabled': s.enabled,
                'stream': list(s.stream)}[b]
    if kind == 'setting':
        st = project.analysis.studies.get(a, {})
        return copy.deepcopy(st.get(b))
    return copy.deepcopy(_pointer(project.to_dict(), a))


def _number(path: str, value) -> float:
    """value as a finite float, else FactorError in a sentence (a typo such as 'high', NaN or infinity)."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise FactorError(f'{path} takes a number, not {value!r}.') from None
    if not math.isfinite(x):
        raise FactorError(f'{path} takes a finite number, not {value!r}.')
    return x


def apply(project, values: dict, variant=None, allow_locked: bool = False):
    """A new project with every factor set (applied in the order given)."""
    from .project import Project
    if not isinstance(values, dict):
        raise FactorError('The factors to set are a mapping of path to value, e.g. {"sensor:Wrist.z": 0.3}.')
    json_edits = [(p, v) for p, v in values.items() if _parse(p)[0] == 'json']
    out = project.snapshot()
    for path, value in values.items():
        kind, a, b = _parse(path)
        if kind == 'json':
            continue
        if kind == 'setting':
            out.analysis.studies.setdefault(a, {})[b] = copy.deepcopy(value)
            continue
        s = find_sensor(out, a, variant)
        if s.locked and not allow_locked:
            raise FactorError(f'{s.name} is locked: unlock it to let this change it.')
        p = s.mount.pose
        if b in POSITION:
            pos = p.position.copy()
            pos[POSITION.index(b)] = _number(path, value)
            s.mount.pose = Pose(pos, p.quaternion)
        elif b in ANGLES:
            rpy = list(rpy_from_quat(p.quaternion))
            rpy[ANGLES.index(b)] = math.radians(_number(path, value))
            s.mount.pose = Pose(p.position, quat_from_rpy(*rpy))
        elif b in SHIFTS:
            pos = p.position.copy()
            pos[SHIFTS.index(b)] += _number(path, value)
            s.mount.pose = Pose(pos, p.quaternion)
        elif b in TURNS:
            axis = np.zeros(3)
            axis[TURNS.index(b)] = 1.0
            ang = math.radians(_number(path, value))
            dq = np.concatenate([axis * math.sin(ang / 2), [math.cos(ang / 2)]])
            s.mount.pose = Pose(p.position, quat_multiply(p.quaternion, dq))       # body-frame rotation
        elif b == 'model':
            from . import library
            if value not in (None, '') and library.get(str(value)) is None:
                raise FactorError(f'{path}: the library has no device with the id "{value}".')
            s.model_id = str(value or '')
        elif b == 'parent':
            s.mount.parent_frame = str(value)
        elif b == 'enabled':
            s.enabled = bool(value)
        elif b == 'stream':
            try:
                s.stream = [int(v) for v in (value or [])][:2]
            except (TypeError, ValueError):
                raise FactorError(f'{path} takes [width, height] in pixels (or [] for the largest), not '
                                  f'{value!r}.') from None
    if json_edits:
        doc = out.to_dict()
        for path, value in json_edits:
            _pointer(doc, _parse(path)[1], copy.deepcopy(value), set_=True)
        keep = out
        out = Project.from_dict(doc)
        out.path, out.meshes, out.sources, out.thumb = keep.path, keep.meshes, keep.sources, keep.thumb
        if not allow_locked:
            _refuse_locked_json(keep, out)
    return out


def locked_target(project, path: str, variant=None):
    """The locked sensor a factor path would change (a sensor: path of a locked sensor, or a json: pointer into one,
    /variants/i/sensors/j/...), else None. The searches use it to leave such a factor out up front."""
    try:
        kind, a, _b = _parse(path)
    except FactorError:
        return None
    if kind == 'sensor':
        try:
            s = find_sensor(project, a, variant)
        except FactorError:
            return None
        return s if s.locked else None
    if kind != 'json':
        return None
    parts = [p.replace('~1', '/').replace('~0', '~') for p in a.lstrip('/').split('/')]
    if parts[0] != 'variants':
        return None
    try:
        scope = project.variants if len(parts) == 1 else [project.variants[int(parts[1])]]
    except (ValueError, IndexError):
        return None
    if len(parts) > 2 and parts[2] != 'sensors':
        return None
    if len(parts) <= 3:                                  # a whole variant list, variant or sensor list
        return next((s for v in scope for s in v.sensors if s.locked), None)
    try:
        s = scope[0].sensors[int(parts[3])]
    except (ValueError, IndexError):
        return None
    return s if s.locked else None


def _refuse_locked_json(before, after):
    """FactorError when a json: pointer moved, re-modelled, removed or unlocked a locked sensor (of any variant):
    a pointer reaches the same fields as a sensor: path, so it obeys the same lock."""
    for v in before.variants:
        nv = after.variant_by_uid(v.uid)
        for s in v.sensors:
            if not s.locked:
                continue
            ns = nv.sensor(s.uid) if nv is not None else None
            if ns is None or ns.to_dict() != s.to_dict():
                raise FactorError(f'{s.name} is locked: unlock it to let this change it.')


def describe(path: str, project=None, variant=None) -> dict:
    kind, a, b = _parse(path)
    d = {'path': path}
    if kind == 'sensor':
        unit = 'm' if b in POSITION + SHIFTS else '°' if b in ANGLES + TURNS else ''
        k = ('continuous' if b in POSITION + ANGLES + SHIFTS + TURNS else 'bool' if b == 'enabled'
             else 'list' if b == 'stream' else 'categorical')
        name = a
        if project is not None:
            s = find_sensor(project, a, variant)
            name = s.name
            d.update(sensor_uid=s.uid, locked=s.locked)
        words = {'x': 'x', 'y': 'y', 'z': 'z', 'roll': 'roll', 'pitch': 'pitch', 'yaw': 'yaw', 'dx': 'shift x',
                 'dy': 'shift y', 'dz': 'shift z', 'droll': 'turn about its x', 'dpitch': 'turn about its y',
                 'dyaw': 'turn about its z', 'model': 'model', 'parent': 'mount frame', 'enabled': 'enabled',
                 'stream': 'stream mode'}
        d.update(title=f'{name} {words[b]}', unit=unit, kind=k)
    elif kind == 'setting':
        d.update(title=f'{a} setting {b}', unit='', kind='continuous')
    else:
        d.update(title=a, unit='', kind='continuous')
    return d
