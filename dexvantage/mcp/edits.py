"""Changes to a project as data: what edit_project and simulate_changes take, checked before anything changes.

    apply(project, changes=None, ops=None) -> Edit(project, lines, keys)
        changes   {factor path: value}: any settable quantity of core/factors.py (sensor:<id>.x|y|z|roll|pitch|yaw|
                  dx|dy|dz|droll|dpitch|dyaw|model|parent|enabled|stream, setting:<study>.<key>, json:/pointer)
        ops       [{'op': ...}] structural changes, applied first, in order:
                    add_sensor        model_id, name?, parent? ('world' or a link name), x y z (m)?, roll pitch yaw (°)?
                    remove_sensor     sensor (uid or name)
                    add_variant       name, copy? (default true: a copy of the active rig), activate? (default true)
                    activate_variant  variant (uid or name)
                    rename_variant    variant, name
        -> a NEW project (the input is never changed), one line per change in plain words, and the project sections
           it touches (the undo step's keys: 'meta', 'workspace', 'variant:<uid>', 'variants', 'analysis', ...)
    EditError     a plain sentence; nothing is changed

Locked items are the user's: apply() compares every locked sensor and robot of the project before and after and
refuses any change to one (moving, re-modelling, removing or unlocking it), whichever path made it. Locking is fine.
A model id must be a library entry; the sensor's kind follows the entry's. A parent frame must be a static frame
(world, cage, tripod, table) or a link of a robot of the project.

Ported from Vantage mcp/edits.py (operations as data, checked on a copy first, lines of what changed, the keys of the
undo step); the vocabulary is DexVantage's factor paths (core/factors.py).

Known limits: a copied variant keeps its sensors' uids (cables, policies and calibration name sensors by uid, so they
follow the copy); json:/ pointers are applied through Project.from_dict, which normalises what it does not know.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass

from ..core import factors as FX
from ..core.jobs import StudyError

OPS = ('add_sensor', 'remove_sensor', 'add_variant', 'activate_variant', 'rename_variant')
_NUMERIC = FX.POSITION + FX.ANGLES + FX.SHIFTS + FX.TURNS
_READ_ONLY_POINTERS = ('/format', '/version', '/log', '/created', '/saved_at', '/meshes', '/sources')


class EditError(StudyError):
    """Why a change cannot be made, in plain words; nothing was changed."""


@dataclass
class Edit:
    project: object
    lines: list
    keys: list


def _canon(o) -> str:
    return json.dumps(o, sort_keys=True, default=str, separators=(',', ':'))


def changed_keys(a, b) -> list:
    """The section keys (Project.section) whose content differs between projects a and b."""
    from ..core.project import SECTIONS
    keys = [k for k in SECTIONS if k != 'variants' and _canon(a.section(k)) != _canon(b.section(k))]
    ua, ub = [v.uid for v in a.variants], [v.uid for v in b.variants]
    if ua == ub and a.active_uid == b.active_uid:
        keys += [f'variant:{u}' for u in ub if _canon(a.section(f'variant:{u}')) != _canon(b.section(f'variant:{u}'))]
    else:
        keys.append('variants')
    return keys


def _find_variant(p, ident):
    ident = str(ident or '').strip()
    hit = [v for v in p.variants if v.uid == ident] or [v for v in p.variants if v.name == ident]
    if not hit:
        raise EditError(f'The project has no variant "{ident}" (it has: {", ".join(v.name for v in p.variants)}).')
    if len(hit) > 1:
        raise EditError(f'More than one variant is called "{ident}": name it by its uid.')
    return hit[0]


def _sensor(p, ident):
    try:
        return FX.find_sensor(p, str(ident or ''))
    except FX.FactorError as e:
        raise EditError(str(e)) from None


def _num(v, what):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise EditError(f'{what}: a number is expected, not {json.dumps(v, default=str)[:40]}.')
    return float(v)


def _parent(p, frame):
    """(parent_frame, robot_uid) for a frame name: a static frame or a link of one of the project's robots."""
    from ..core.project import MOUNT_STATIC
    frame = str(frame or 'world')
    if frame in MOUNT_STATIC:
        return frame, ''
    owners = [r for r in p.robots if any(lk.get('name') == frame for lk in r.links)]
    if not owners:
        raise EditError(f'"{frame}" is neither a static frame ({", ".join(MOUNT_STATIC)}) nor a link of a robot in '
                        'the project.')
    return frame, owners[0].uid


def _entry(model_id):
    from ..core import library as LB
    from ..core.project import SENSOR_KINDS
    e = LB.get(str(model_id or ''))
    if e is None:
        raise EditError(f'"{model_id}" is not in the library (get_sensor and the Library list the model ids).')
    if e.kind not in SENSOR_KINDS:
        raise EditError(f'{e.title} is a {e.kind} entry, not a sensor ({", ".join(SENSOR_KINDS)}).')
    return e


# =========================================================================== operations
def _op(p, op: dict, lines: list):
    if not isinstance(op, dict):
        raise EditError('Each operation is an object with "op".')
    kind = op.get('op')
    if kind not in OPS:
        raise EditError(f'"op" must be one of {", ".join(OPS)}.')
    if kind == 'add_sensor':
        from ..core.frames import Pose, quat_from_rpy
        from ..core.project import Mount, Sensor
        e = _entry(op.get('model_id'))
        frame, robot = _parent(p, op.get('parent'))
        pos = [_num(op.get(k, 0.0), k) for k in FX.POSITION]
        rpy = [math.radians(_num(op.get(k, 0.0), k)) for k in FX.ANGLES]
        n = len(p.variant.sensors)
        name = str(op.get('name') or f'{e.name} {n + 1}')
        s = Sensor(name=name, kind=e.kind, model_id=e.id, slot=n % 8,
                   mount=Mount(frame, robot, Pose(pos, quat_from_rpy(*rpy))))
        p.variant.sensors.append(s)
        lines.append(f'Added {name} ({e.title}) on {frame} at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) m')
    elif kind == 'remove_sensor':
        s = _sensor(p, op.get('sensor'))
        p.variant.sensors = [x for x in p.variant.sensors if x is not s]
        lines.append(f'Removed {s.name}')
    elif kind == 'add_variant':
        from ..core.project import Variant, new_uid
        name = str(op.get('name') or '').strip()
        if not name:
            raise EditError('add_variant needs a name.')
        if any(v.name == name for v in p.variants):
            raise EditError(f'A variant called "{name}" exists already.')
        if op.get('copy', True):
            v = Variant.from_dict({**copy.deepcopy(p.variant.to_dict()), 'uid': new_uid(), 'name': name,
                                   'note': f'a copy of {p.variant.name}'})
        else:
            v = Variant(name=name)
        p.variants.append(v)
        if op.get('activate', True):
            p.active_uid = v.uid
        lines.append(f'Added the variant {name}' + (' (a copy of the active rig)' if op.get('copy', True) else '')
                     + (', now active' if op.get('activate', True) else ''))
    elif kind == 'activate_variant':
        v = _find_variant(p, op.get('variant'))
        p.active_uid = v.uid
        lines.append(f'Made {v.name} the active variant')
    elif kind == 'rename_variant':
        v = _find_variant(p, op.get('variant'))
        name = str(op.get('name') or '').strip()
        if not name:
            raise EditError('rename_variant needs a name.')
        lines.append(f'Renamed the variant {v.name} to {name}')
        v.name = name


# =========================================================================== factor changes
def _fmt(v, unit=''):
    if isinstance(v, float):
        v = f'{v:.4g}'
    elif isinstance(v, (list, dict)):
        v = json.dumps(v, default=str)
    return f'{v}{" " + unit if unit else ""}'


def _check_value(p, path, value):
    kind, a, b = FX._parse(path)                       # noqa: SLF001 — the contract's own parser
    if kind == 'sensor':
        _sensor(p, a)
        if b in _NUMERIC:
            _num(value, path)
        elif b == 'model':
            _entry(value)
        elif b == 'parent':
            _parent(p, value)
        elif b == 'enabled' and not isinstance(value, bool):
            raise EditError(f'{path}: true or false is expected.')
        elif b == 'stream':
            if not (isinstance(value, list) and len(value) in (0, 2) and all(
                    isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in value)):
                raise EditError(f'{path}: [width, height] in pixels (or [] for the largest stream) is expected.')
    elif kind == 'setting':
        from ..core import jobs
        jobs.load_studies()
        if a not in jobs.STUDIES:
            raise EditError(f'{path}: there is no study {a} (run_study lists them).')
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            raise EditError(f'{path}: the value is not JSON.') from None
    elif kind == 'json':
        if any(a == r or a.startswith(r + '/') for r in _READ_ONLY_POINTERS):
            raise EditError(f'{path}: that part of the project is not edited from here.')


def _fix_up(p, path, value):
    """What factors.apply leaves to the caller: a new model's kind, a new parent's robot."""
    kind, a, b = FX._parse(path)                       # noqa: SLF001
    if kind != 'sensor':
        return
    s = _sensor(p, a)
    if b == 'model':
        s.kind = _entry(value).kind
    elif b == 'parent':
        s.mount.parent_frame, s.mount.robot_uid = _parent(p, value)


def _locked_check(before, after):
    """EditError when a locked sensor or robot of before differs (or is gone) in after."""
    for v in before.variants:
        nv = after.variant_by_uid(v.uid)
        for s in v.sensors:
            if not s.locked:
                continue
            ns = nv.sensor(s.uid) if nv is not None else None
            if ns is None or _canon(ns.to_dict()) != _canon(s.to_dict()):
                raise EditError(f'{s.name} is locked: it stays as the user set it (only the user unlocks it). '
                                'Nothing was changed.')
    for r in before.robots:
        if r.locked:
            nr = after.robot(r.uid)
            if nr is None or _canon(nr.to_dict()) != _canon(r.to_dict()):
                raise EditError(f'{r.name} is locked: it stays as the user set it. Nothing was changed.')


def apply(project, changes=None, ops=None) -> Edit:
    """The changes made on a copy of project (module docstring)."""
    if changes is None and ops is None:
        raise EditError('Give changes ({factor path: value}) or ops (a list of operations).')
    if changes is not None and not isinstance(changes, dict):
        raise EditError('changes is an object: {"sensor:Wrist camera.z": 0.12, ...}.')
    if ops is not None and not isinstance(ops, list):
        raise EditError('ops is a list of operations.')
    if not (changes or ops):
        raise EditError('There is nothing to change.')
    p = project.snapshot()
    lines: list = []
    for op in ops or []:
        _op(p, op, lines)
    if changes:
        for path, value in changes.items():
            try:
                _check_value(p, path, value)
            except FX.FactorError as e:
                raise EditError(str(e)) from None
        olds = {}
        for path in changes:
            try:
                olds[path] = FX.get(p, path)
            except FX.FactorError:
                olds[path] = None                    # a new setting or pointer
        try:
            q = FX.apply(p, dict(changes))
        except FX.FactorError as e:
            raise EditError(f'{e} Nothing was changed.') from None
        except (TypeError, ValueError) as e:
            raise EditError(f'A value does not fit its quantity ({e}). Nothing was changed.') from None
        for path, value in changes.items():
            _fix_up(q, path, value)
            d = FX.describe(path, q)
            new = FX.get(q, path) if FX._parse(path)[2] not in FX.SHIFTS + FX.TURNS else value   # noqa: SLF001
            if FX._parse(path)[2] in FX.SHIFTS + FX.TURNS:                                       # noqa: SLF001
                lines.append(f'{d["title"]}: by {_fmt(float(value), d.get("unit", ""))}')
            else:
                lines.append(f'{d["title"]}: {_fmt(olds[path], d.get("unit", ""))} → {_fmt(new, d.get("unit", ""))}')
        p = q
    _locked_check(project, p)
    keys = changed_keys(project, p)
    if not keys:
        lines.append('(the values were already so: nothing differs)')
    return Edit(p, lines, keys)


def title(lines) -> str:
    """The undo step's label from the lines."""
    if not lines:
        return 'change'
    t = lines[0] if len(lines) == 1 else f'{lines[0]} (+{len(lines) - 1} more)'
    return t[:120]


def commit(session, edit: Edit, base, changes=None, ops=None) -> str:
    """Make the edit on the session as one undo step; returns the step's label. base is the project the trial was
    made on: while the session's project has the same content in the sections the edit touches, the trial's sections
    are restored as they are (the same uids); otherwise the changes are made again on it."""
    def fn(p):
        e = edit
        if any(_canon(p.section(k)) != _canon(base.section(k)) for k in edit.keys):
            e = apply(p, changes, ops)                # the project changed meanwhile: made again on it
            if set(e.keys) - set(edit.keys):
                raise EditError('The project changed while the change was checked: call the tool again.')
        for k in edit.keys:
            p.restore(k, copy.deepcopy(e.project.section(k)))
    return session.edit(list(edit.keys), fn, title(edit.lines))
