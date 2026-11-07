"""Teleoperation devices from the hardware library (kind 'teleop', spec §6, §7.4): their stated figures, and the
teleop study's device records made from them.

A teleop entry (dexvantage/resources/library/teleop/<id>.json, schema library.schema.json "teleop", written up in
docs/models/teleop.md#library) holds what its maker or paper states, each figure {value, source: an index into the
entry's sources or null, quote}; a figure no source states is null ("not stated by the maker"). Nothing here fills a
gap. The block also says how the device drives a follower:

    device_kind   the study's kind: leader_arm | gello | vr | spacemouse | glove | hand_tracker
    mapping       joint (a leader arm, GELLO) or cartesian (a tracked position)
    leads         followers a source says the device leads joint to joint, as words matched against a robot's name
                  and model file ('*': any arm, GELLO being built for its target arm)
    same_joints   true when the source states the leader has the follower's joints; false when only the angles are
                  mapped one to one (ALOHA's WidowX leading a ViperX, the Koch leader's own parts)

device_from_library(project, library_id, follower_uid) makes a project.tasks.teleop_devices record (the shared
contract, core/studies/teleop.py clean_device) from an entry:

- a leader arm or GELLO whose entry leads the follower robot: leader = the follower's own model, joint to joint (the
  leader's comfortable box is then taken from the follower's joint ranges; stated when same_joints is true, a
  labelled ASSUMPTION otherwise);
- a leader arm whose entry does not lead that follower: leader = {'library_id'}: the study says the entry carries no
  kinematics and asks for the leader as a robot;
- a VR headset, space mouse, glove or hand tracker: cartesian, its volume the entry's stated workspace (no entry
  states one as a box or cylinder today, so the study says "not stated" until a measured volume is entered).

Every stated figure goes into the record's source dict with its URL, the date read and the quote, and the record keeps
library_id so the page can show the entry.

API
    entries() -> [library.Entry]                    the teleop entries
    entry_problems(d) -> [str]                      plain messages, [] when the teleop block is consistent
    figure(entry, key) -> (value, source text)      a figure and where it comes from ('not stated by the maker')
    figure_rows(entry) -> [dict]                    every figure with its label, unit, text and source (the page)
    leads(entry, robot) -> bool                     the entry is stated to lead this robot joint to joint
    device_from_library(project, library_id, follower_uid=None) -> dict   a clean teleop device record
    summary(entry) -> str                           one line of headline figures

Known limits: makers state few teleop figures and none states a tracking volume as a box or cylinder (a Leap Motion
cone, a headset's play area, a space mouse's cap travel), so library VR, glove, hand-tracker and space-mouse devices
reach the study without a comfortable region until the user enters a measured one; latencies are what each maker
calls latency (a glove's signal latency, a headset's photon-to-photon passthrough), not an end-to-end teleop delay;
"leads" matches words in the robot's name and model file, so a renamed robot is not recognised (choose the leader by
hand); a leader whose own joint ranges are not stated (same_joints false) is modelled with the follower's ranges, an
assumption the record says.
"""
from __future__ import annotations

import math
import re

from . import library

KINDS = ('leader_arm', 'gello', 'vr', 'spacemouse', 'glove', 'hand_tracker')
ARM_KINDS = ('leader_arm', 'gello')
MAPPINGS = ('joint', 'cartesian')
# key -> (label, unit, type)        type: int | num | text | volume
FIELDS = {
    'dof': ('Degrees of freedom', '', 'int'),
    'tracking_hz': ('Tracking or sample rate', 'Hz', 'num'),
    'latency_ms': ('Latency', 'ms', 'num'),
    'workspace': ('Workspace (box or cylinder, device frame)', 'm', 'volume'),
    'range_text': ('Range or field of view', '', 'text'),
    'reach_mm': ('Reach', 'mm', 'num'),
    'accuracy_mm': ('Accuracy', 'mm', 'num'),
    'position_resolution_mm': ('Position resolution', 'mm', 'num'),
    'joint_resolution_deg': ('Joint resolution', '°', 'num'),
    'feedback': ('Force or haptic feedback', '', 'text'),
}
NOT_STATED = library.NOT_STATED


def entries() -> list:
    return library.entries('teleop')


def _entry(library_id: str):
    e = library.get(library_id) if library_id else None
    return e if e is not None and e.kind == 'teleop' else None


def _norm(s) -> str:
    return re.sub(r'[\s_\-]+', '', str(s or '')).lower()


def _volume_problems(v) -> list:
    if not isinstance(v, dict) or v.get('kind') not in ('box', 'cylinder'):
        return ['must be {"kind": "box" | "cylinder", "center", "size"}']
    n = 3 if v['kind'] == 'box' else 2
    size, c = v.get('size'), v.get('center', [0, 0, 0])
    if not (isinstance(size, list) and len(size) == n and all(_is_num(x) and x > 0 for x in size)):
        return [f'size must be {n} positive lengths in metres']
    if not (isinstance(c, list) and len(c) == 3 and all(_is_num(x) for x in c)):
        return ['center must be [x, y, z] in metres']
    return []


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def entry_problems(d) -> list:
    """Plain messages for what is inconsistent in a teleop entry (the library schema checks the shape; this checks
    each figure against its field, that a stated figure names a source and quotes it, and that the device's kind,
    mapping and leads agree)."""
    if not isinstance(d, dict):
        return ['the entry must be a JSON object']
    t = d.get('teleop')
    if not isinstance(t, dict):
        return ['teleop: a teleop entry needs its teleop block']
    msgs = []
    kind, mapping = t.get('device_kind'), t.get('mapping')
    if kind not in KINDS:
        msgs.append(f'teleop.device_kind: must be one of {", ".join(KINDS)}')
    if mapping not in MAPPINGS:
        msgs.append('teleop.mapping: must be joint or cartesian')
    elif kind in KINDS and (mapping == 'joint') != (kind in ARM_KINDS):
        msgs.append('teleop.mapping: a leader arm or GELLO maps joint to joint; other devices map a tracked position '
                    '(cartesian)')
    if kind in ARM_KINDS:
        leads_ = t.get('leads')
        if not (isinstance(leads_, list) and leads_ and all(isinstance(x, str) and x.strip() for x in leads_)):
            msgs.append('teleop.leads: a leader arm names the followers its source says it leads (or "*")')
        if not isinstance(t.get('same_joints'), bool):
            msgs.append('teleop.same_joints: a leader arm says whether its source states the follower\'s joints')
    n_src = len(d.get('sources') or [])
    for k, (_label, _unit, typ) in FIELDS.items():
        fv = t.get(k)
        if fv is None:
            msgs.append(f'teleop.{k}: missing (write {{"value": null, "source": null}} when no source states it)')
            continue
        if not isinstance(fv, dict) or 'value' not in fv or 'source' not in fv:
            msgs.append(f'teleop.{k}: must be {{"value", "source"}}')
            continue
        v, s = fv['value'], fv['source']
        if v is None:
            if s is not None:
                msgs.append(f'teleop.{k}: a value no source states has no source index')
            continue
        if not (isinstance(s, int) and not isinstance(s, bool) and 0 <= s < n_src):
            msgs.append(f'teleop.{k}: a stated value needs the index of its source (0 to {n_src - 1})')
        if not str(fv.get('quote') or '').strip():
            msgs.append(f'teleop.{k}: a stated value quotes its source')
        if typ == 'int' and not (isinstance(v, int) and not isinstance(v, bool) and v > 0):
            msgs.append(f'teleop.{k}: must be a whole number above 0')
        elif typ == 'num' and not (_is_num(v) and v > 0):
            msgs.append(f'teleop.{k}: must be a number above 0')
        elif typ == 'text' and not (isinstance(v, str) and v.strip()):
            msgs.append(f'teleop.{k}: must be text')
        elif typ == 'volume':
            msgs += [f'teleop.{k}: {m}' for m in _volume_problems(v)]
    return msgs


def source_text(entry, idx, quote: str = '') -> str:
    srcs = entry.sources if hasattr(entry, 'sources') else list((entry or {}).get('sources') or [])
    if not isinstance(idx, int) or not 0 <= idx < len(srcs):
        return NOT_STATED
    s = srcs[idx]
    out = f'{s.get("title") or s.get("url")}, {s.get("url")}, read {s.get("read")}'
    return out + (f': {quote}' if quote else '')


def figure(entry, key: str):
    """(value, where it comes from): the stated value and its source with the quote, or (None, 'not stated by the
    maker' and what the entry says about it)."""
    fv = ((entry.data.get('teleop') or {}).get(key)) or {}
    v = fv.get('value')
    if v is None:
        q = str(fv.get('quote') or '').strip()
        return None, NOT_STATED + (f' ({q})' if q else '')
    return v, source_text(entry, fv.get('source'), str(fv.get('quote') or ''))


def value_text(key: str, v) -> str:
    if v is None:
        return '—'
    _label, unit, typ = FIELDS[key]
    if typ == 'volume':
        s = ' × '.join(f'{x:g}' for x in v['size'])
        return f'{v["kind"]} {s} m'
    if typ in ('int', 'num'):
        return f'{v:g}' + (f' {unit}' if unit else '')
    return str(v)


def figure_rows(entry) -> list:
    """Every teleop figure of an entry: [{'key', 'label', 'unit', 'value', 'text', 'source', 'stated'}], plus the
    entry's mass and interface (top-level figures), for the page."""
    rows = []
    for k, (label, unit, _typ) in FIELDS.items():
        v, where = figure(entry, k)
        rows.append({'key': k, 'label': label, 'unit': unit, 'value': v, 'text': value_text(k, v), 'source': where,
                     'stated': v is not None})
    first = source_text(entry, 0) if entry.sources else NOT_STATED
    for k, label, unit in (('mass_g', 'Mass', 'g'), ('interface', 'Interface', '')):
        v = entry.data.get(k)
        rows.append({'key': k, 'label': label, 'unit': unit, 'value': v,
                     'text': '—' if v is None else (f'{v:g} {unit}' if unit else str(v)),
                     'source': (first + ' (see the entry note)') if v is not None else NOT_STATED,
                     'stated': v is not None})
    return rows


def leads(entry, robot) -> bool:
    """Whether the entry's source says it leads this robot joint to joint: '*' leads any robot; otherwise one of its
    words appears in the robot's name, model file or source note (case, spaces, '-' and '_' ignored)."""
    words = list((entry.data.get('teleop') or {}).get('leads') or [])
    if '*' in words:
        return True
    if robot is None:
        return False
    src = getattr(robot, 'source', None)
    hay = _norm(' '.join([getattr(robot, 'name', ''), getattr(src, 'file', ''), getattr(src, 'note', '')]))
    return any(_norm(w) and _norm(w) in hay for w in words)


def summary(entry) -> str:
    """One short line of the entry's headline figures (stated ones only)."""
    parts = []
    for k in ('dof', 'tracking_hz', 'latency_ms', 'accuracy_mm'):
        v, _w = figure(entry, k)
        if v is not None:
            parts.append(('DoF ' if k == 'dof' else '') + value_text(k, v))
    return ', '.join(parts) if parts else 'no tracking figures stated'


def device_from_library(project, library_id: str, follower_uid: str | None = None) -> dict:
    """A project.tasks.teleop_devices record (core/studies/teleop.py clean_device) made from a library teleop entry,
    with every stated figure and its source in the record's source dict. Raises library.LibraryError when there is
    no such teleop entry."""
    from .studies import teleop as TS
    e = _entry(library_id)
    if e is None:
        raise library.LibraryError(f'"{library_id}" is not a teleop device in the library.')
    t = e.data['teleop']
    kind = t.get('device_kind') if t.get('device_kind') in TS.KINDS else 'leader_arm'
    fu = follower_uid or (project.primary_robot.uid if project.robots else '')
    robot = project.robot(fu) if fu else None
    n = sum(1 for d in project.tasks.teleop_devices if isinstance(d, dict) and d.get('library_id') == e.id) + 1
    src = {'device': f'library: {e.maker}, {e.name} ({e.id})'}
    for row in figure_rows(e):
        if row['stated'] and row['key'] not in ('mass_g', 'interface'):
            src[row['key']] = f'{row["text"]}: {row["source"]}'
    d = {'kind': kind, 'name': e.name + (f' {n}' if n > 1 else ''), 'follower_uid': fu, 'library_id': e.id,
         'mapping': {'kind': t.get('mapping') or ('joint' if kind in ARM_KINDS else 'cartesian'), 'scale': 1.0}}
    if kind in ARM_KINDS:
        if robot is not None and leads(e, robot):
            d['leader'] = {'robot_uid': fu}
            if t.get('same_joints'):
                src['leader'] = (f'the follower\'s own model, joint to joint: the source states the leader has the '
                                 f'follower\'s joints ({e.name})')
            else:
                src['leader'] = ('ASSUMPTION: the follower\'s own model, joint to joint. The source maps the joint '
                                 'angles one to one but does not state the leader\'s own joint ranges, so its '
                                 'comfortable box is taken from the follower\'s ranges.')
        else:
            d['leader'] = {'library_id': e.id}
            words = ', '.join(t.get('leads') or [])
            src['leader'] = (f'{e.name} is stated to lead {words or "no robot named"}; the follower here is not one '
                             'of them, so add the leader arm as a robot and choose it as the leader')
    else:
        ws, where = figure(e, 'workspace')
        if ws is not None:
            d['volume'] = ws
            src['volume'] = where
        else:
            src['volume'] = (f'not stated: {where}. Enter the tracking volume or bounds you measured, or a published '
                             'figure with its source')
    d['source'] = src
    return TS.clean_device(d, project)
