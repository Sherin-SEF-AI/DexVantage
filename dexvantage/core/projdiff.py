"""Project diff and review: what changed between two projects, in the project's own terms, and a three-way merge of
two edits of one project.

    diff(a, b, results_a=None, results_b=None, metric_ids=None) -> dict      a and b: Project objects or paths (.dexv
                                                                             files or unpacked folders)
    to_text(d) -> str        a plain-text table (batch --diff prints it)
    to_html(d) -> str        a standalone HTML page with the same table
    metric_deltas(results_a, results_b, ids=None) -> [row]       from study results already run (or cached)
    evaluate_deltas(a, b, ids, cache=None, ctx=None) -> [row]    runs the metrics' studies on both (core/metrics)
    merge3(base, ours, theirs) -> MergeResult                    MergeConflict (a plain sentence) when it cannot

The diff, schema 'dexvantage.diff/1':

    {'schema', 'a': {'name', 'file'}, 'b': {...}, 'identical': bool, 'counts': {section: n},
     'changes': [{'section', 'op': 'added' | 'removed' | 'changed', 'kind' ('Sensor', 'Robot', ...), 'item' (its
                  name), 'field' ('' for an added or removed item), 'path', 'before', 'after', 'delta'?, 'text'}],
     'metrics': [{'id', 'title', 'unit', 'better', 'study', 'a', 'b', 'delta', 'verdict' ('better' | 'worse' |
                  'same' | None), 'status_a', 'status_b'}]}

Sections compared: meta (name, notes), robots, variants, sensors (each variant's, matched as the variants are),
workspace (volumes, objects, lights, surfaces, background), tasks (grasp sets, trajectories, teleop devices),
requirements, policies, rig (hosts and their controllers, hubs, links, clocks and stream triggers), cables, calibration
(boards, sessions), settings (project.analysis: voxel sizes and each study's settings; the "live" view toggle is not
content), data (datasets, audit requirements) and files (mesh arrays, stored model files). The design log, the creation
and save times are not compared. Items with a uid are matched by uid, then by name (a name both projects hold once); a
mount or base pose that changed reports its shift in mm along the parent frame's axes and its turn in degrees
(frames.quat_angle). Numbers are equal when they differ by at most one part in 10¹² (a save and load can change the last
bit of a normalised quaternion).

The merge works on the project's JSON (Project.to_dict) section by section: a value one side changed and the other left
as in the base takes the changed side; lists of items with uids merge item by item (items the other side added are
appended in its order); other lists (a robot's joints, a trajectory's samples) and poses (position and orientation
together) are single values. A value both sides changed differently, or an item one side removed and the other changed,
is a conflict, and so is a merged reference that neither side left dangling (a sensor on a robot the other side removed;
a policy camera, cable or link naming a removed sensor; an active variant that is gone). Mesh arrays and stored model
files merge key by key. The design log keeps ours and appends their new entries. Any conflict refuses the whole merge
with a sentence that lists each one: nothing half-merged is returned.

Known limits: a renamed item that also changed uid (deleted and re-added) is reported as removed and added; list
values (joint tables, grasp poses, trajectory samples) are compared and merged whole, so the diff says how many
entries changed, not which grasp moved, and two edits of different entries of one list conflict; a merge can give two
items the same name (each side added one); metric deltas need results for both projects (run them, or pass
--diff-metrics to have batch mode run them).
"""
from __future__ import annotations

import copy
import hashlib
import html
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from .jobs import StudyError
from .metrics import delta_unit

SCHEMA = 'dexvantage.diff/1'
REL = 1e-12
MISSING = type('Missing', (), {'__repr__': lambda s: 'MISSING', '__bool__': lambda s: False})()
SECTIONS = ('meta', 'robots', 'variants', 'sensors', 'workspace', 'tasks', 'requirements', 'policies', 'rig', 'cables',
            'calibration', 'settings', 'data', 'files')
FIELD_NAMES = {'model_id': 'model', 'library_id': 'library entry', 'parent_frame': 'mount frame',
               'robot_uid': 'mount robot', 'sensor_uid': 'sensor', 'mesh_key': 'mesh', 'active_uid': 'active variant',
               'voxel_mm': 'coverage voxel (mm)', 'refine_mm': 'refinement voxel (mm)', 'practical_mbps':
               'practical rate (Mbit/s)', 'gripper_frame': 'gripper frame', 'fingertip_frames': 'fingertip frames',
               'contact_pads': 'contact pads', 'stream_ids': 'streams', 'cable_uid': 'cable'}


class MergeConflict(StudyError):
    """The two edits cannot be merged; .conflicts lists each conflicting place in words."""

    def __init__(self, text, conflicts=()):
        super().__init__(text)
        self.conflicts = list(conflicts)


# ====================================================================== values
def _isnum(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


def same(a, b) -> bool:
    """Deep equality of JSON-like values, numbers within one part in 10¹²."""
    if a is MISSING or b is MISSING:
        return a is b
    if _isnum(a) and _isnum(b):
        a, b = float(a), float(b)
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return a == b or abs(a - b) <= REL * max(1.0, abs(a), abs(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def _is_pose(v) -> bool:
    return isinstance(v, dict) and set(v) == {'position', 'quaternion'}


def _flat(d, prefix=()) -> dict:
    """{path tuple: value}: dicts recursed into; poses, lists and scalars are values."""
    out = {}
    if isinstance(d, dict) and not _is_pose(d):
        for k, v in d.items():
            if isinstance(v, dict) and not _is_pose(v) and v:
                out.update(_flat(v, prefix + (str(k),)))
            else:
                out[prefix + (str(k),)] = v
    else:
        out[prefix] = d
    return out


def _digest(v) -> str:
    return hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=False, default=str).encode('utf-8')).hexdigest()


def _record(v):
    """A value as the diff's JSON keeps it: small values as they are, long or nested lists as a count and a hash."""
    if v is MISSING:
        return None
    if isinstance(v, list) and (len(v) > 16 or any(isinstance(x, (list, dict)) for x in v)) and not _is_pose(v):
        return {'entries': len(v), 'sha256': _digest(v)}
    if isinstance(v, dict) and len(json.dumps(v, default=str)) > 400:
        return {'fields': len(v), 'sha256': _digest(v)}
    return copy.deepcopy(v)


def brief(v) -> str:
    """A short text of a value for the table."""
    if v is MISSING or v is None:
        return '—'
    if isinstance(v, bool):
        return 'yes' if v else 'no'
    if _isnum(v):
        return f'{float(v):.6g}'
    if isinstance(v, str):
        t = ' '.join(v.split())
        return f'"{t[:57]}…"' if len(t) > 60 else f'"{t}"'
    if _is_pose(v):
        p = v.get('position') or [0, 0, 0]
        return f'({", ".join(f"{float(x):.4g}" for x in p)}) m'
    if isinstance(v, list):
        if len(v) <= 4 and all(_isnum(x) or isinstance(x, str) for x in v):
            return '[' + ', '.join(brief(x).strip('"') for x in v) + ']'
        return f'{len(v)} entr{"y" if len(v) == 1 else "ies"}'
    if isinstance(v, dict):
        return f'{len(v)} field{"" if len(v) == 1 else "s"}'
    return str(v)


def pose_delta(a: dict, b: dict) -> dict:
    """{'shift_mm': [dx, dy, dz] in the parent frame, 'distance_mm', 'turn_deg'} between two pose dicts."""
    from .frames import quat_angle
    pa = np.asarray(a.get('position') or (0, 0, 0), float)
    pb = np.asarray(b.get('position') or (0, 0, 0), float)
    d = (pb - pa) * 1000.0
    ang = math.degrees(quat_angle(a.get('quaternion') or (0, 0, 0, 1), b.get('quaternion') or (0, 0, 0, 1)))
    return {'shift_mm': [float(x) for x in d], 'distance_mm': float(np.linalg.norm(d)), 'turn_deg': float(ang)}


def _field_name(path: tuple) -> str:
    return ' '.join(FIELD_NAMES.get(p, p.replace('_', ' ')) for p in path)


# ====================================================================== matching
def _label(x) -> str:
    if not isinstance(x, dict):
        return str(x)
    for k in ('name', 'metric', 'sensor_uid', 'uid'):
        if x.get(k):
            return str(x[k])
    return '(unnamed)'


def pairs(la, lb, key='uid', alt=('name',)) -> list:
    """[(a item or None, b item or None)]: matched by key, then by the first alt field both name uniquely; a's order,
    then b's new items."""
    la = [x for x in la or [] if isinstance(x, dict)]
    lb = [x for x in lb or [] if isinstance(x, dict)]
    used, out, rest = set(), [], []
    idx = {}
    for j, y in enumerate(lb):
        if y.get(key) not in (None, ''):
            idx.setdefault(y.get(key), j)
    for x in la:
        j = idx.get(x.get(key)) if x.get(key) not in (None, '') else None
        if j is not None and j not in used:
            used.add(j)
            out.append([x, lb[j]])
        else:
            out.append([x, None])
            rest.append(len(out) - 1)
    for f in alt:
        free_b = [j for j in range(len(lb)) if j not in used]
        nb: dict = {}
        for j in free_b:
            nb.setdefault(lb[j].get(f), []).append(j)
        na: dict = {}
        for i in rest:
            na.setdefault(out[i][0].get(f), []).append(i)
        still = []
        for i in rest:
            n = out[i][0].get(f)
            if n not in (None, '') and len(na.get(n, ())) == 1 and len(nb.get(n, ())) == 1:
                j = nb[n][0]
                used.add(j)
                out[i][1] = lb[j]
            else:
                still.append(i)
        rest = still
    out += [[None, lb[j]] for j in range(len(lb)) if j not in used]
    return [tuple(p) for p in out]


# ====================================================================== the diff
class _Diff:
    def __init__(self):
        self.changes = []

    def add(self, section, op, kind, item, fld='', path='', before=MISSING, after=MISSING, text='', delta=None):
        row = {'section': section, 'op': op, 'kind': kind, 'item': item, 'field': fld, 'path': path,
               'before': _record(before), 'after': _record(after), 'text': text}
        if delta is not None:
            row['delta'] = delta
        self.changes.append(row)

    def fields(self, section, kind, item, path, a, b, skip=('uid',), names=True):
        """One change per differing value of two items (names=False: field names as they are in the file)."""
        fa, fb = _flat(a), _flat(b)
        keys = list(fa) + [k for k in fb if k not in fa]
        for k in keys:
            if k and k[0] in skip:
                continue
            va, vb = fa.get(k, MISSING), fb.get(k, MISSING)
            if same(va, vb):
                continue
            # a dict empty on one side and holding values on the other: its values are the changes, not the dict
            if ({} in (va, vb)) and any(len(o) > len(k) and o[:len(k)] == k for o in keys):
                continue
            fname = _field_name(k) if names else ' '.join(k)
            p = path + ('.' + '.'.join(k) if k else '')
            if _is_pose(va) and _is_pose(vb):
                dl = pose_delta(va, vb)
                s = dl['shift_mm']
                text = (f'{kind} {item}: {fname} moved {dl["distance_mm"]:.1f} mm (x {s[0]:+.1f}, y {s[1]:+.1f}, '
                        f'z {s[2]:+.1f} mm) and turned {dl["turn_deg"]:.2f}°')
                self.add(section, 'changed', kind, item, fname, p, va, vb, text, dl)
            else:
                text = f'{kind} {item}: {fname} {brief(va)} → {brief(vb)}'
                self.add(section, 'changed', kind, item, fname, p, va, vb, text)

    def items(self, section, kind, path, la, lb, key='uid', alt=('name',), skip=('uid',), describe=None, inner=None):
        for x, y in pairs(la, lb, key, alt):
            lab = _label(y if y is not None else x)
            p = f'{path}[{(y or x).get(key) or lab}]'
            if x is None:
                extra = describe(y) if describe else ''
                self.add(section, 'added', kind, lab, '', p, MISSING, y, f'{kind} {lab} added{extra}')
            elif y is None:
                self.add(section, 'removed', kind, lab, '', p, x, MISSING, f'{kind} {lab} removed')
            else:
                if _label(x) != _label(y):
                    self.add(section, 'changed', kind, lab, 'name', p + '.name', _label(x), _label(y),
                             f'{kind} {_label(x)} renamed {lab}')
                self.fields(section, kind, lab, p, x, y, skip=tuple(skip) + ('name',) + tuple(inner or {}))
                for sub, fn in (inner or {}).items():
                    fn(lab, p, x.get(sub), y.get(sub))


def _as_project(x):
    from .project import Project
    if isinstance(x, Project):
        return x
    if isinstance(x, str):
        return Project.load(x)
    raise StudyError('A project to compare is a Project or the path of a .dexv file or unpacked folder.')


def _sensor_desc(s) -> str:
    m = (s.get('mount') or {})
    where = m.get('parent_frame') or 'world'
    return f' ({s.get("model_id") or "no device"} on {where})'


def _robots(D, la, lb):
    def joints(lab, p, ja, jb):
        D.items('robots', 'Joint', f'{p}.joints', ja, jb, key='name', alt=(), skip=(),
                describe=lambda j: f' ({j.get("type", "")}) on robot {lab}')

    def links(lab, p, ka, kb):
        D.items('robots', 'Link', f'{p}.links', ka, kb, key='name', alt=(), skip=(),
                describe=lambda k: f' on robot {lab}')
    D.items('robots', 'Robot', 'robots', la, lb, inner={'joints': joints, 'links': links},
            describe=lambda r: f' ({len(r.get("joints") or [])} joints)')


def _variants(D, da, db):
    va, vb = da.get('variants') or [], db.get('variants') or []
    many = len(va) > 1 or len(vb) > 1
    act_a = next((v for v in va if v.get('uid') == da.get('active_uid')), va[0] if va else None)
    act_b = next((v for v in vb if v.get('uid') == db.get('active_uid')), vb[0] if vb else None)
    for x, y in pairs(va, vb):
        lab = _label(y if y is not None else x)
        p = f'variants[{(y or x).get("uid")}]'
        if x is None:
            D.add('variants', 'added', 'Rig variant', lab, '', p, MISSING, y,
                  f'Rig variant {lab} added ({len(y.get("sensors") or [])} sensors)')
            continue
        if y is None:
            D.add('variants', 'removed', 'Rig variant', lab, '', p, x, MISSING, f'Rig variant {lab} removed')
            continue
        if _label(x) != _label(y):
            D.add('variants', 'changed', 'Rig variant', lab, 'name', p + '.name', _label(x), lab,
                  f'Rig variant {_label(x)} renamed {lab}')
        D.fields('variants', 'Rig variant', lab, p, x, y, skip=('uid', 'name', 'sensors', 'created'))
        pre = f'{lab} ▸ ' if many else ''
        for sa, sb in pairs(x.get('sensors'), y.get('sensors')):
            sl = pre + _label(sb if sb is not None else sa)
            sp = f'{p}.sensors[{(sb or sa).get("uid")}]'
            if sa is None:
                D.add('sensors', 'added', 'Sensor', sl, '', sp, MISSING, sb, f'Sensor {sl} added{_sensor_desc(sb)}')
            elif sb is None:
                D.add('sensors', 'removed', 'Sensor', sl, '', sp, sa, MISSING, f'Sensor {sl} removed')
            else:
                if _label(sa) != _label(sb):
                    D.add('sensors', 'changed', 'Sensor', sl, 'name', sp + '.name', _label(sa), _label(sb),
                          f'Sensor {pre}{_label(sa)} renamed {_label(sb)}')
                D.fields('sensors', 'Sensor', sl, sp, sa, sb, skip=('uid', 'name'))
    if act_a is not None and act_b is not None:
        matched = dict((id(x), y) for x, y in pairs(va, vb) if x is not None)
        partner_b = {id(y) for x, y in pairs(va, vb) if x is not None and y is not None}
        # said only when both active variants exist on both sides (an added or removed variant says it already)
        if matched.get(id(act_a)) is not None and id(act_b) in partner_b and matched.get(id(act_a)) is not act_b:
            D.add('variants', 'changed', 'Rig variant', _label(act_b), 'active variant', 'active_uid',
                  _label(act_a), _label(act_b), f'The active rig variant changed from {_label(act_a)} to '
                                                f'{_label(act_b)}')


def _keyed_sections(D, da, db):
    wa, wb = da.get('workspace') or {}, db.get('workspace') or {}
    for k, kind in (('volumes', 'Task volume'), ('objects', 'Task object'), ('lights', 'Light'),
                    ('surfaces', 'Surface')):
        D.items('workspace', kind, f'workspace.{k}', wa.get(k), wb.get(k))
    if not same(wa.get('background'), wb.get('background')):
        D.add('workspace', 'changed', 'Workspace', 'background', 'background', 'workspace.background',
              wa.get('background'), wb.get('background'),
              f'Workspace background {brief(wa.get("background"))} → {brief(wb.get("background"))}')
    ta, tb = da.get('tasks') or {}, db.get('tasks') or {}
    for k, kind in (('grasp_sets', 'Grasp set'), ('trajectories', 'Trajectory'), ('teleop_devices', 'Teleop device')):
        D.items('tasks', kind, f'tasks.{k}', ta.get(k), tb.get(k))
    D.items('requirements', 'Requirement', 'requirements', da.get('requirements'), db.get('requirements'),
            alt=('name', 'metric'))
    D.items('policies', 'Policy', 'policies', da.get('policies'), db.get('policies'))
    ra, rb = da.get('rig') or {}, db.get('rig') or {}

    def ctrls(lab, p, ca, cb):
        D.items('rig', 'Controller', f'{p}.controllers', ca, cb, describe=lambda c: f' on host {lab}')
    D.items('rig', 'Host', 'rig.hosts', ra.get('hosts'), rb.get('hosts'), inner={'controllers': ctrls})
    D.items('rig', 'Hub', 'rig.hubs', ra.get('hubs'), rb.get('hubs'))
    names = {s.get('uid'): s.get('name') for d in (da, db) for v in d.get('variants') or []
             for s in v.get('sensors') or []}
    la = [dict(x, name=names.get(x.get('sensor_uid')) or x.get('sensor_uid')) for x in ra.get('links') or []]
    lb = [dict(x, name=names.get(x.get('sensor_uid')) or x.get('sensor_uid')) for x in rb.get('links') or []]
    D.items('rig', 'Link', 'rig.links', la, lb, key='sensor_uid', alt=())
    sa, sb = ra.get('sync') or {}, rb.get('sync') or {}
    D.items('rig', 'Clock', 'rig.sync.clocks', sa.get('clocks'), sb.get('clocks'))
    D.fields('rig', 'Stream trigger', 'sync', 'rig.sync.streams', sa.get('streams') or {}, sb.get('streams') or {},
             skip=(), names=False)
    D.items('cables', 'Cable', 'cables', da.get('cables'), db.get('cables'))
    ca, cb = da.get('calibration') or {}, db.get('calibration') or {}
    D.items('calibration', 'Board', 'calibration.boards', ca.get('boards'), cb.get('boards'))
    D.items('calibration', 'Calibration session', 'calibration.sessions', ca.get('sessions'), cb.get('sessions'))
    aa = {k: v for k, v in (da.get('analysis') or {}).items() if k != 'live'}
    ab = {k: v for k, v in (db.get('analysis') or {}).items() if k != 'live'}
    D.fields('settings', 'Setting', 'analysis', 'analysis', aa, ab, skip=('studies',))
    st_a, st_b = aa.get('studies') or {}, ab.get('studies') or {}
    for sid in list(st_a) + [k for k in st_b if k not in st_a]:
        D.fields('settings', 'Study setting', sid, f'analysis.studies.{sid}', st_a.get(sid) or {},
                 st_b.get(sid) or {}, skip=(), names=False)
    xa, xb = da.get('data') or {}, db.get('data') or {}
    D.items('data', 'Dataset', 'data.datasets', xa.get('datasets'), xb.get('datasets'))
    D.fields('data', 'Audit requirement', 'audit', 'data.requirements', xa.get('requirements') or {},
             xb.get('requirements') or {}, skip=(), names=False)


def _array_sig(a) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha256(f'{a.dtype.str}{a.shape}'.encode() + a.tobytes()).hexdigest()


def _files(D, pa, pb):
    for what, ka, kb, sig in (('Mesh', pa.meshes, pb.meshes, _array_sig),
                              ('Model file', pa.sources, pb.sources, lambda v: hashlib.sha256(v).hexdigest())):
        for k in sorted(set(ka) | set(kb)):
            if k not in ka:
                D.add('files', 'added', what, k, '', f'{what.lower()}:{k}', MISSING, sig(kb[k]), f'{what} {k} added')
            elif k not in kb:
                D.add('files', 'removed', what, k, '', f'{what.lower()}:{k}', sig(ka[k]), MISSING,
                      f'{what} {k} removed')
            elif sig(ka[k]) != sig(kb[k]):
                D.add('files', 'changed', what, k, 'content', f'{what.lower()}:{k}', sig(ka[k]), sig(kb[k]),
                      f'{what} {k}: content changed')


def diff(a, b, results_a: dict | None = None, results_b: dict | None = None, metric_ids=None) -> dict:
    """What changed from project a to project b (Project objects or paths), and, with study results for both
    ({study id: result doc}), the metric deltas."""
    pa, pb = _as_project(a), _as_project(b)
    da, db = pa.to_dict(), pb.to_dict()
    D = _Diff()
    for k in ('name', 'notes'):
        if not same(da.get(k), db.get(k)):
            D.add('meta', 'changed', 'Project', db.get('name') or '', k, k, da.get(k), db.get(k),
                  f'Project {k} {brief(da.get(k))} → {brief(db.get(k))}')
    _robots(D, da.get('robots'), db.get('robots'))
    _variants(D, da, db)
    _keyed_sections(D, da, db)
    _files(D, pa, pb)
    order = {s: i for i, s in enumerate(SECTIONS)}
    changes = sorted(D.changes, key=lambda c: order.get(c['section'], 99))       # stable: within a section, found
    counts = {s: sum(1 for c in changes if c['section'] == s) for s in SECTIONS}
    out = {'schema': SCHEMA,
           'a': {'name': pa.name, 'file': os.path.basename(pa.path.rstrip('/\\')) if pa.path else ''},
           'b': {'name': pb.name, 'file': os.path.basename(pb.path.rstrip('/\\')) if pb.path else ''},
           'identical': not changes, 'counts': {s: n for s, n in counts.items() if n}, 'changes': changes,
           'metrics': []}
    if results_a and results_b:
        out['metrics'] = metric_deltas(results_a, results_b, metric_ids)
    return out


# ====================================================================== metric deltas
def _delta_row(m, ra: dict, rb: dict) -> dict:
    from . import metrics as MX
    va, vb = ra.get('value'), rb.get('value')
    row = {'id': m.id, 'title': m.title, 'unit': m.unit, 'better': m.better, 'study': m.study, 'a': va, 'b': vb,
           'delta': None, 'verdict': None, 'status_a': ra.get('status'), 'status_b': rb.get('status')}
    if va is not None and vb is not None:
        row['delta'] = vb - va
        row['verdict'] = ('same' if same(va, vb) else 'better' if MX.better_of(m.id, vb, va) else 'worse')
    for side, r in (('a', ra), ('b', rb)):
        if r.get('reason'):
            row[f'reason_{side}'] = r['reason']
    return row


def metric_deltas(results_a: dict, results_b: dict, ids=None) -> list:
    """Metric deltas from study results already run: every metric (or those of ids) whose study both result sets
    hold, read with core/metrics.read; a figure one side does not state gives no delta."""
    from . import metrics as MX
    MX.load_metrics()
    rows = []
    for mid in (ids if ids is not None else sorted(MX.METRICS)):
        m = MX.get(mid)
        if m.study in (results_a or {}) and m.study in (results_b or {}):
            rows.append(_delta_row(m, MX.read(m, results_a[m.study]), MX.read(m, results_b[m.study])))
    return rows


def evaluate_deltas(a, b, ids, cache=None, ctx=None) -> list:
    """Metric deltas computed: core/metrics.evaluate on both projects (each study run once per project)."""
    from . import metrics as MX
    pa, pb = _as_project(a), _as_project(b)
    ea = MX.evaluate(pa.snapshot(), list(ids), cache=cache, ctx=ctx)
    eb = MX.evaluate(pb.snapshot(), list(ids), cache=cache, ctx=ctx)
    return [_delta_row(MX.get(mid), ea[mid], eb[mid]) for mid in ids]


# ====================================================================== text and HTML
def _fmt(v, unit='') -> str:
    if v is None:
        return '—'
    return f'{v:.4g}' + (f' {unit}' if unit else '')


def _metric_text(r) -> str:
    if r['delta'] is None:
        return 'not stated on both sides'
    return f'{r["delta"]:+.4g} {delta_unit(r["unit"])} ({r["verdict"]})'.replace('  ', ' ')


def to_text(d: dict) -> str:
    """The diff as a plain-text table (section, item, change), then the metric deltas."""
    a, b = d['a'], d['b']
    head = (f'{a["name"]}' + (f' ({a["file"]})' if a['file'] else '') + ' → ' + f'{b["name"]}'
            + (f' ({b["file"]})' if b['file'] else ''))
    n = len(d['changes'])
    lines = [head, 'No differences.' if d['identical'] else f'{n} change{"" if n == 1 else "s"}: '
             + ', '.join(f'{k} {v}' for k, v in d['counts'].items()), '']
    if d['changes']:
        w1 = max(7, *(len(c['section']) for c in d['changes']))
        w2 = min(34, max(4, *(len(c['op']) for c in d['changes'])))
        lines.append(f'{"section":<{w1}}  {"op":<{w2}}  change')
        for c in d['changes']:
            lines.append(f'{c["section"]:<{w1}}  {c["op"]:<{w2}}  {c["text"]}')
    if d.get('metrics'):
        lines += ['', 'Figures of merit (a → b):']
        w = max(len(r['title']) for r in d['metrics'])
        for r in d['metrics']:
            lines.append(f'  {r["title"]:<{w}}  {_fmt(r["a"], r["unit"])} → {_fmt(r["b"], r["unit"])}  '
                         f'{_metric_text(r)}')
    return '\n'.join(lines) + '\n'


def to_html(d: dict) -> str:
    """A standalone HTML page of the diff (no scripts, no external files)."""
    e = html.escape
    a, b = d['a'], d['b']
    rows = ''.join(f'<tr class="{e(c["op"])}"><td>{e(c["section"])}</td><td>{e(c["op"])}</td><td>{e(c["kind"])}</td>'
                   f'<td>{e(c["item"])}</td><td>{e(c["field"])}</td><td class="m">{e(brief_record(c["before"]))}</td>'
                   f'<td class="m">{e(brief_record(c["after"]))}</td><td>{e(c["text"])}</td></tr>'
                   for c in d['changes'])
    mrows = ''.join(f'<tr><td>{e(r["title"])}</td><td class="m">{e(_fmt(r["a"], r["unit"]))}</td>'
                    f'<td class="m">{e(_fmt(r["b"], r["unit"]))}</td><td class="m">{e(_metric_text(r))}</td></tr>'
                    for r in d.get('metrics') or [])
    title = f'{a["name"]} → {b["name"]}'
    summary = ('No differences.' if d['identical'] else
               f'{len(d["changes"])} changes: ' + ', '.join(f'{k} {v}' for k, v in d['counts'].items()))
    css = ('body{font:14px/1.45 "IBM Plex Sans",system-ui,sans-serif;margin:24px;color:#1d2024;background:#fff}'
           'table{border-collapse:collapse;width:100%;margin:12px 0}th,td{border-bottom:1px solid #e3e5e8;'
           'padding:4px 8px;text-align:left;vertical-align:top}th{font-weight:600;color:#555}'
           '.m{font-family:"IBM Plex Mono",ui-monospace,monospace;white-space:nowrap}'
           'tr.added td:nth-child(2){color:#1a7f37}tr.removed td:nth-child(2){color:#b42318}'
           '@media (prefers-color-scheme:dark){body{background:#16181b;color:#e6e8eb}th,td{border-color:#2c3036}'
           'th{color:#a0a6ad}}')
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width, initial-scale=1"><title>Project diff</title><style>{css}</style></head>'
            f'<body><h1>{e(title)}</h1><p>{e(summary)}</p>')
    if d['changes']:
        page += ('<table><thead><tr><th>Section</th><th>Change</th><th>Kind</th><th>Item</th><th>Field</th>'
                 f'<th>Before</th><th>After</th><th>In words</th></tr></thead><tbody>{rows}</tbody></table>')
    if mrows:
        page += ('<h2>Figures of merit</h2><table><thead><tr><th>Metric</th><th>Before</th><th>After</th>'
                 f'<th>Change</th></tr></thead><tbody>{mrows}</tbody></table>')
    return page + '</body></html>\n'


def brief_record(v) -> str:
    """brief() of a value as the diff's JSON records it (long lists as their entry count)."""
    if isinstance(v, dict) and 'sha256' in v and ('entries' in v or 'fields' in v):
        n = v.get('entries', v.get('fields'))
        return f'{n} {"entries" if "entries" in v else "fields"}'
    return brief(v)


# ====================================================================== three-way merge
@dataclass
class MergeResult:
    project: object
    from_ours: list = field(default_factory=list)       # places only our side changed
    from_theirs: list = field(default_factory=list)     # places only their side changed
    notes: list = field(default_factory=list)


def _key(x):
    if not isinstance(x, dict):
        return None
    if x.get('uid'):
        return str(x['uid'])
    if x.get('sensor_uid'):
        return 'link:' + str(x['sensor_uid'])
    return None


def _keyed(*lists) -> bool:
    ls = [x for x in lists if x is not MISSING]
    if not any(ls) or not all(isinstance(x, list) for x in ls):
        return False
    return all(_key(i) is not None for x in ls for i in x) and all(len({_key(i) for i in x}) == len(x) for x in ls)


class _Merger:
    def __init__(self):
        self.conflicts, self.ours, self.theirs = [], [], []

    def merge(self, b, o, t, where):
        if same(o, t):
            return o
        if same(b, o):
            self.theirs.append(where)
            return t
        if same(b, t):
            self.ours.append(where)
            return o
        if all(isinstance(x, dict) for x in (o, t)) and (b is MISSING or isinstance(b, dict)) \
                and not (_is_pose(o) or _is_pose(t)):
            bb = b if isinstance(b, dict) else {}
            out = {}
            for k in list(o) + [k for k in t if k not in o]:
                r = self.merge(bb.get(k, MISSING), o.get(k, MISSING), t.get(k, MISSING), where + [str(k)])
                if r is not MISSING:
                    out[k] = r
            return out
        if isinstance(o, list) and isinstance(t, list) and (b is MISSING or isinstance(b, list)) and _keyed(b, o, t):
            bi = {_key(x): x for x in (b if isinstance(b, list) else [])}
            oi, ti = {_key(x): x for x in o}, {_key(x): x for x in t}
            out = []
            for k in [_key(x) for x in o] + [_key(x) for x in t if _key(x) not in oi]:
                lab = _label(oi.get(k) or ti.get(k) or bi.get(k))
                r = self.merge(bi.get(k, MISSING), oi.get(k, MISSING), ti.get(k, MISSING), where + [lab])
                if r is not MISSING:
                    out.append(r)
            return out
        if o is MISSING or t is MISSING:
            gone, kept = ('yours', 'theirs') if o is MISSING else ('theirs', 'yours')
            self.conflicts.append(f'{" ▸ ".join(where)} (removed in {gone}, changed in {kept})')
        else:
            self.conflicts.append(f'{" ▸ ".join(where)} (base {brief(b)}, yours {brief(o)}, theirs {brief(t)})')
        return o


def _dangling(doc: dict) -> set:
    """References in a project doc that name nothing: (what, place) pairs."""
    robots = {r.get('uid') for r in doc.get('robots') or []}
    sensors = {s.get('uid') for v in doc.get('variants') or [] for s in v.get('sensors') or []}
    out = set()
    for v in doc.get('variants') or []:
        for s in v.get('sensors') or []:
            ru = (s.get('mount') or {}).get('robot_uid')
            if ru and ru not in robots:
                out.add(f'sensor {s.get("name")} is mounted on a robot that is no longer in the project')
    for p in doc.get('policies') or []:
        for c in p.get('cameras') or []:
            if c not in sensors:
                out.add(f'policy {p.get("name")} takes a camera that is no longer in the project')
    for c in doc.get('cables') or []:
        if c.get('sensor_uid') and c['sensor_uid'] not in sensors:
            out.add(f'cable {c.get("name")} names a sensor that is no longer in the project')
    for k in (doc.get('rig') or {}).get('links') or []:
        if k.get('sensor_uid') and k['sensor_uid'] not in sensors:
            out.add('a host link names a sensor that is no longer in the project')
    if doc.get('active_uid') not in {v.get('uid') for v in doc.get('variants') or []}:
        out.add('the active rig variant is no longer in the project')
    return out


def _bin_merge(M, b: dict, o: dict, t: dict, sig, what):
    out = {}
    for k in list(o) + [k for k in t if k not in o] + [k for k in b if k not in o and k not in t]:
        sb = sig(b[k]) if k in b else None
        so = sig(o[k]) if k in o else None
        st = sig(t[k]) if k in t else None
        if so == st:
            pick = o.get(k)
        elif so == sb:
            pick = t.get(k)
        elif st == sb:
            pick = o.get(k)
        else:
            M.conflicts.append(f'{what} {k} (changed differently on both sides)')
            pick = o.get(k)
        if pick is not None:
            out[k] = pick
    return out


def merge3(base, ours, theirs) -> MergeResult:
    """Merge two edits (ours, theirs) of one project (base). MergeConflict lists every conflict in words."""
    from .project import Project
    pb, po, pt = _as_project(base), _as_project(ours), _as_project(theirs)
    db, do, dt = pb.to_dict(), po.to_dict(), pt.to_dict()
    M = _Merger()
    skip = ('format', 'version', 'created', 'saved_at', 'log', 'meshes', 'sources')
    doc = {k: do[k] for k in skip if k in do}
    for k in list(do) + [k for k in dt if k not in do]:
        if k in skip:
            continue
        r = M.merge(db.get(k, MISSING), do.get(k, MISSING), dt.get(k, MISSING), [k])
        if r is not MISSING:
            doc[k] = r
    seen = {json.dumps(e, sort_keys=True) for e in do.get('log') or []}
    base_log = {json.dumps(e, sort_keys=True) for e in db.get('log') or []}
    doc['log'] = list(do.get('log') or []) + [e for e in dt.get('log') or []
                                              if json.dumps(e, sort_keys=True) not in seen | base_log]
    meshes = _bin_merge(M, pb.meshes, po.meshes, pt.meshes, _array_sig, 'Mesh')
    sources = _bin_merge(M, pb.sources, po.sources, pt.sources, lambda v: hashlib.sha256(v).hexdigest(),
                         'Model file')
    new = _dangling(doc) - _dangling(do) - _dangling(dt)
    M.conflicts += sorted(f'the merge would leave a broken reference: {x}' for x in new)
    if M.conflicts:
        n = len(M.conflicts)
        listed = '; '.join(f'({i + 1}) {c}' for i, c in enumerate(M.conflicts[:12]))
        more = f'; and {n - 12} more' if n > 12 else ''
        raise MergeConflict(f'The two edits cannot be merged: {n} place{"s" if n > 1 else ""} '
                            f'{"were" if n > 1 else "was"} changed differently on both sides: {listed}{more}. Make the '
                            'change on one side only, or merge these by hand.', M.conflicts)
    p = Project.from_dict(doc)
    p.meshes, p.sources, p.thumb = meshes, sources, po.thumb
    p.created = po.created
    p.add_log('merge', f'Merged two edits from a common base: {len(M.ours)} change{"" if len(M.ours) == 1 else "s"} '
                       f'of ours, {len(M.theirs)} of theirs ({pt.name})')
    notes = []
    names = [s.name for s in p.variant.sensors]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        notes.append(f'Two sensors share a name after the merge ({", ".join(dup)}): each side added one. Rename one.')
    return MergeResult(p, M.ours, M.theirs, notes)
