"""Dataset pooling across cells (part of the multi-cell comparison, core/cells.py): can the recordings of several cells
be pooled into one training set, what mapping makes them poolable (rename, resize, resample, reorder, re-encode,
rescale) and what cannot be mapped.

Each cell's recording is described as a **schema** read from a LeRobot info.json (the LeRobot v2.1 / v3.0 feature
schema, core/datasets/lerobot.py, from the pinned LeRobot source):

- from a dataset the cell has recorded (project.data.datasets, opened with core/datasets: a LeRobot folder's own
  meta/info.json; for an RLDS folder or a ROS 2 bag, the first episode's cameras and joint names), or
- from the plan: the info.json the cell's LeRobot export would write (core/exporters/lerobot.info_json over
  rig_streams and state_names: one video feature per camera at its stream size, observation.state and action over the
  robots' actuated joints), with the figures the plan states besides (joint types and limits from the robot models,
  the policy's control rate, the streams' frame rates).

    plan_schema(project) -> schema         recorded_schema(path) -> schema         schema_from_info(info) -> schema
    pooling(schemas, names, target=0) -> {'target', 'cells': [{'verdict', 'operations', 'unmappable', ...}]}

A schema: {'source': 'plan' | 'recorded', 'format', 'version', 'fps': {'value', 'source'}, 'cameras': [{'key',
'role' ('wrist' | 'scene'), 'role_source', 'width', 'height', 'fps', 'codec', 'codec_source'}], 'state' and 'action'
({'key', 'names', 'count', 'units' (per joint 'rad' | 'm' | None: not stated), 'space'}), 'gripper' ({'joints',
'type', 'range' [lo, hi] or None}), 'control_hz' ({'value', 'source'})}.

Rules, cell against the target (the reference cell), each a mapping step or a reason it cannot be mapped:

- Cameras are paired by key, then by role in order (wrist with wrist, scene with scene). A paired camera under
  another key is renamed; another size is resized (the aspect ratio changing is said: pad or crop to keep the
  image undistorted); another frame rate is resampled (frames dropped or repeated); another codec re-encoded. A
  target camera without a counterpart cannot be mapped (that view does not exist in the cell); a camera the target
  does not have is dropped.
- The dataset's frame rate (its rows): another rate is resampled (interpolate the state; the actions as the policy
  needs them).
- State and action: another joint count cannot be mapped (another embodiment: pooling needs a policy that takes
  several embodiments, with padded or separate action spaces). The same names in another order are reordered (the
  index list is given); other names at the same count are taken to correspond by position, said as an assumption.
  Joints whose units differ (a revolute joint against a prismatic one, each target joint against the cell joint it
  maps to after a reorder, and only where both state a unit) cannot be mapped; units the dataset does not state are a
  warning (LeRobot stores no units), and values beyond ±2π on a revolute joint are flagged as probably degrees (a
  heuristic) with a deg→rad conversion step for the cell (rad→deg when it is the target that looks like degrees;
  neither, with a warning, when both do).
- Gripper encoding (joints named grip / finger / jaw): another joint type, or another range, is rescaled linearly
  from the cell's [closed, open] range to the target's when both ranges are stated (an assumption: a linear
  map of opening); a range not stated is a warning.
- Action space: a different space (joint position against end-effector, absolute against delta) cannot be mapped
  here. Control rate: another rate means resampling the actions (an action chunk spans another time).

Verdict per cell: 'poolable as is' (no step, nothing unmappable), 'poolable after mapping', 'not poolable'.

Known limits: the rules read the feature schema, not the data (a recorded dataset's values are read only for the
units heuristic and the gripper range, from its stored stats); joint correspondence by name or position, never by
kinematics; a camera's role in a recorded dataset comes from its key name (wrist / hand / gripper words), which a
dataset may not follow; image content (exposure, colour, viewpoint) is the geometric gap's business, not this
check's; the resample steps say what to do, not how well it preserves the motion; LeRobot's own multi-dataset
loader is not run here (the package is not a dependency).
"""
from __future__ import annotations

import math
import os
import re

import numpy as np

from .jobs import StudyError

SCHEMA = 'dexvantage.cells_pooling/1'
STATE_KEY, ACTION_KEY = 'observation.state', 'action'
GRIPPER_RE = re.compile(r'grip|finger|jaw', re.I)
WRIST_RE = re.compile(r'wrist|hand|gripper|eye_in_hand|palm', re.I)
DEG_LIMIT = 2 * math.pi + 0.1          # a revolute joint's stored values beyond this are probably degrees (heuristic)
VERDICTS = ('poolable as is', 'poolable after mapping', 'not poolable')
CODEC_PLAN_SOURCE = ("LeRobot's default encoder (libsvtav1, AV1), which DexVantage's LeRobot writer uses: a default, "
                     "not a recording")


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


# ====================================================================== schemas
def schema_from_info(info: dict, source: str = 'recorded') -> dict:
    """A schema from a LeRobot meta/info.json document (v2.1 or v3.0): its video and image features, the state and
    action features with their names, and the dataset's fps."""
    if not isinstance(info, dict):
        raise StudyError('The dataset description is not a LeRobot info.json object.')
    feats = info.get('features') or {}
    cams = []
    for k, ft in feats.items():
        if not isinstance(ft, dict) or ft.get('dtype') not in ('video', 'image'):
            continue
        vi = ft.get('info') or {}
        shape = ft.get('shape') or []
        names = ft.get('names') or []
        h = w = None
        if len(shape) == 3:
            if list(names)[:2] == ['channels', 'height'] or (shape[0] in (1, 3, 4) and shape[2] not in (1, 3, 4)):
                h, w = shape[1], shape[2]
            else:
                h, w = shape[0], shape[1]
        h = int(vi.get('video.height') or h) if (vi.get('video.height') or h) else None
        w = int(vi.get('video.width') or w) if (vi.get('video.width') or w) else None
        role = 'wrist' if WRIST_RE.search(k) else 'scene'
        cams.append({'key': k, 'role': role, 'role_source': 'the key name', 'width': w, 'height': h,
                     'fps': _num(vi.get('video.fps')), 'codec': vi.get('video.codec') if ft.get('dtype') == 'video'
                     else 'png', 'codec_source': 'info.json' if vi.get('video.codec') or ft.get('dtype') == 'image'
                     else None, 'dtype': ft.get('dtype')})

    def vec(key):
        ft = feats.get(key)
        if not isinstance(ft, dict):
            return {'key': key, 'names': [], 'count': 0, 'units': None, 'present': False}
        shape = ft.get('shape') or []
        n = int(shape[0]) if shape else 0
        names = ft.get('names')
        if isinstance(names, dict):                       # {'motors': [...]} as some datasets store them
            names = next((v for v in names.values() if isinstance(v, list)), [])
        names = [str(x) for x in names or []]
        return {'key': key, 'names': names if len(names) == n else [], 'count': n, 'units': None, 'present': True}
    fps = _num(info.get('fps'))
    state, action = vec(STATE_KEY), vec(ACTION_KEY)
    action['space'] = _action_space(state, action)
    return {'source': source, 'format': 'lerobot', 'version': str(info.get('codebase_version') or ''),
            'robot_type': info.get('robot_type'), 'fps': {'value': fps, 'source': 'info.json fps'},
            'cameras': sorted(cams, key=lambda c: c['key']), 'state': state, 'action': action,
            'gripper': _gripper(state['names'], None, None),
            'control_hz': {'value': fps, 'source': 'the dataset\'s row rate (LeRobot stores one action per row)'},
            'notes': []}


def _action_space(state: dict, action: dict) -> dict:
    names = action.get('names') or []
    if not action.get('present', True) or not action.get('count'):
        return {'value': None, 'source': 'no action feature'}
    if names and names == (state.get('names') or []):
        return {'value': 'joint position', 'source': 'the action names are the state\'s joint names'}
    low = ' '.join(names).lower()
    if names and re.search(r'delta|vel', low):
        return {'value': 'joint delta or velocity', 'source': 'the action names'}
    if names and re.search(r'\b(x|y|z|roll|pitch|yaw|eef|ee_)', low):
        return {'value': 'end effector', 'source': 'the action names'}
    return {'value': None, 'source': 'the dataset does not say what its actions are'}


def _gripper(names: list, types: list | None, ranges: list | None) -> dict:
    idx = [i for i, n in enumerate(names or []) if GRIPPER_RE.search(n)]
    if not idx:
        return {'joints': [], 'type': None, 'range': None, 'note': 'no joint named grip, finger or jaw'}
    i = idx[0]
    t = types[i] if types else None
    r = ranges[i] if ranges else None
    return {'joints': [names[j] for j in idx], 'type': t, 'range': r,
            'note': f'{len(idx)} gripper joint{"" if len(idx) == 1 else "s"} by name; the first is compared'}


def plan_schema(project) -> dict:
    """The schema of the LeRobot dataset the cell's rig would record (module docstring)."""
    from . import policies as PO
    from .exporters import lerobot as LE
    from .robot.model import RobotModelError, get_model
    streams = LE.rig_streams(project)
    names, notes = LE.state_names(project)
    types, ranges = [], []
    for r in project.robots:
        try:
            m = get_model(project, r.uid)
        except RobotModelError:
            continue
        for k in range(m.nq):
            types.append(m.joint_types[k])
            lo, hi = float(m.lower[k]), float(m.upper[k])
            ranges.append([lo, hi] if math.isfinite(lo) and math.isfinite(hi) else None)
    ctrl = {'value': None, 'source': 'no policy of the project states a control rate'}
    for pol in project.policies:
        try:
            f = PO.resolve(project, pol.uid)
        except StudyError:
            continue
        v = _num((f.get('control_hz') or (None, ''))[0])
        if v:
            ctrl = {'value': v, 'source': f'{pol.name}: {(f.get("control_hz") or (None, ""))[1]}'}
            break
    rates = [s['fps'] for s in streams if s.get('fps')]
    if ctrl['value'] and abs(ctrl['value'] - round(ctrl['value'])) < 1e-9:
        fps = {'value': float(round(ctrl['value'])), 'source': 'the policy\'s control rate'}
    elif rates:
        fps = {'value': float(round(max(rates))), 'source': 'the fastest camera stream'}
    else:
        fps = {'value': float(LE.DEFAULT_FPS), 'source': f'{LE.DEFAULT_FPS} fps, a default (nothing states a rate)',
               'default': True}
    info = LE.info_json('v3.0', int(fps['value']), None, names, streams)
    sc = schema_from_info(info, source='plan')
    by_key = {s['key']: s for s in streams}
    for c in sc['cameras']:
        s = by_key.get(c['key'])
        if s is None:
            continue
        static = not s['on_link'] or _static(project, s)
        c.update(role='scene' if static else 'wrist', role_source='the mount (wrist: on a link a joint moves)',
                 fps=s.get('fps'), codec=LE.VIDEO_DEFAULT['video.codec'], codec_source=CODEC_PLAN_SOURCE,
                 sensor=s['sensor'])
    units = ['m' if t == 'prismatic' else 'rad' for t in types] if len(types) == len(names) else None
    sc['state']['units'] = units
    sc['action']['units'] = units
    sc['action']['space'] = {'value': 'joint position', 'source': 'DexVantage\'s LeRobot export writes the actuated '
                                                                 'joints as the action'}
    sc['gripper'] = _gripper(names, types if len(types) == len(names) else None,
                             ranges if len(ranges) == len(names) else None)
    sc['fps'] = fps
    sc['control_hz'] = ctrl
    sc['notes'] = list(notes)
    sc['what'] = 'the recording the rig plans (its LeRobot export\'s info.json)'
    return sc


def _static(project, stream) -> bool:
    from .calib.records import _static_link
    s = project.variant.sensor(stream['sensor_uid'])
    return s is not None and s.mount.on_link and _static_link(project, s.mount.robot_uid, s.mount.parent_frame)


def recorded_schema(path: str) -> dict:
    """The schema of a recorded dataset (a LeRobot folder through its info.json; another format through its first
    episode); DatasetError (a StudyError) with a sentence when it cannot be read."""
    from .datasets import open_dataset
    with open_dataset(path) as ds:
        if getattr(ds, 'format', '') == 'lerobot':
            sc = schema_from_info(ds.info, source='recorded')
            try:
                st = ds.stats() or {}
            except Exception:                         # noqa: BLE001 — stats are optional: the heuristic is skipped
                st = {}
            _stats_hints(sc, st)
        else:
            refs = ds.episodes()
            if not refs:
                raise StudyError(f'{os.path.basename(os.path.normpath(path))} has no episodes to read a schema from.')
            e = ds.load(refs[0].index)
            cams = [{'key': k, 'role': 'wrist' if WRIST_RE.search(k) else 'scene', 'role_source': 'the key name',
                     'width': c.width, 'height': c.height, 'fps': _num(c.fps), 'codec': (c.meta or {}).get('codec'),
                     'codec_source': 'the stream' if (c.meta or {}).get('codec') else None}
                    for k, c in sorted(e.cameras.items())]
            q = np.asarray(e.q)
            n = int(q.reshape(len(e.t), -1).shape[1]) if q.size else 0
            state = {'key': 'joint state', 'names': list(e.joint_names), 'count': n, 'units': None, 'present': True}
            na = 0 if e.action is None else int(np.asarray(e.action).reshape(len(e.t), -1).shape[1])
            action = {'key': 'action', 'names': list(e.action_names), 'count': na, 'units': None,
                      'present': e.action is not None}
            action['space'] = _action_space(state, action)
            fps = _num(e.fps)
            sc = {'source': 'recorded', 'format': ds.format, 'version': ds.version, 'robot_type': e.robot_type,
                  'fps': {'value': fps, 'source': 'the episode\'s nominal rate'}, 'cameras': cams, 'state': state,
                  'action': action, 'gripper': _gripper(state['names'], None, None),
                  'control_hz': {'value': fps, 'source': 'the episode\'s nominal rate'}, 'notes': []}
        sc['what'] = f'the recorded dataset {os.path.basename(os.path.normpath(path))}'
        return sc


def _stats_hints(sc: dict, stats: dict):
    """The units heuristic and the gripper range from a LeRobot dataset's stored stats (min, max per joint)."""
    s = stats.get(STATE_KEY) if isinstance(stats, dict) else None
    if not isinstance(s, dict) or s.get('min') is None or s.get('max') is None:
        return
    lo = np.asarray(s['min'], float).reshape(-1)
    hi = np.asarray(s['max'], float).reshape(-1)
    names = sc['state']['names']
    if len(lo) != sc['state']['count']:
        return
    reach = np.maximum(np.abs(lo), np.abs(hi))
    big = [i for i in range(len(reach)) if reach[i] > DEG_LIMIT and not (names and GRIPPER_RE.search(names[i]))]
    if big:
        sc['state']['units_hint'] = {'value': 'deg', 'joints': [names[i] if names else str(i) for i in big],
                                     'source': f'stored values reach {float(reach[big].max()):.1f}, beyond ±2π: '
                                               'probably degrees (a heuristic; LeRobot stores no units)'}
    g = sc.get('gripper') or {}
    if g.get('joints') and names:
        i = names.index(g['joints'][0])
        g['range'] = [float(lo[i]), float(hi[i])]
        g['range_source'] = 'the dataset\'s stored min and max'


def cell_schema(project, mode: str = 'auto') -> dict:
    """The cell's schema: its first recorded dataset (mode 'auto' or 'recorded'), else (or with mode 'plan') its
    plan. A dataset that cannot be read falls back to the plan in 'auto' (said in the notes)."""
    ds = [d for d in (getattr(project, 'data', None) or {}).get('datasets') or [] if d.get('path')]
    why = ''
    if mode in ('auto', 'recorded') and ds:
        try:
            return recorded_schema(ds[0]['path'])
        except StudyError as e:
            if mode == 'recorded':
                raise
            why = f'Its dataset {ds[0].get("name") or os.path.basename(ds[0]["path"])} could not be read ({e}); ' \
                  'the plan is compared instead.'
    elif mode == 'recorded':
        raise StudyError(f'{project.name} has no recorded dataset to compare (Data ▸ Audit adds one).')
    sc = plan_schema(project)
    if why:
        sc['notes'].insert(0, why)
    return sc


# ====================================================================== the rules
def _pair_cameras(tc: list, cc: list) -> tuple[list, list, list]:
    """([(target camera, cell camera)], unpaired target cameras, unpaired cell cameras): by key, then by role in
    order."""
    pairs, used_t, used_c = [], set(), set()
    ck = {c['key']: j for j, c in enumerate(cc)}
    for i, t in enumerate(tc):
        j = ck.get(t['key'])
        if j is not None:
            pairs.append((i, j))
            used_t.add(i)
            used_c.add(j)
    for role in ('wrist', 'scene'):
        ts = [i for i, t in enumerate(tc) if i not in used_t and t['role'] == role]
        cs = [j for j, c in enumerate(cc) if j not in used_c and c['role'] == role]
        for i, j in zip(ts, cs):
            pairs.append((i, j))
            used_t.add(i)
            used_c.add(j)
    pairs.sort()
    return ([(tc[i], cc[j]) for i, j in pairs], [t for i, t in enumerate(tc) if i not in used_t],
            [c for j, c in enumerate(cc) if j not in used_c])


def _rate_differs(a, b) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) > 1e-6


def rules(target: dict, cell: dict) -> dict:
    """{'verdict', 'operations': [...], 'unmappable': [{'what', 'why'}], 'warnings': [...], 'camera_map': {cell key:
    target key}} of one cell's schema against the target's (module docstring)."""
    ops, bad, warn = [], [], []
    pairs, missing, extra = _pair_cameras(target['cameras'], cell['cameras'])
    cmap = {}
    for t, c in pairs:
        cmap[c['key']] = t['key']
        if c['key'] != t['key']:
            ops.append({'op': 'rename', 'feature': c['key'], 'to': t['key'],
                        'why': f'the same {t["role"]} view under another key'
                               + (' (paired by role)' if c['role'] == t['role'] else '')})
        if None in (c['width'], c['height'], t['width'], t['height']):
            warn.append(f'{c["key"]}: the image size is not stated, so a resize cannot be ruled out.')
        elif (c['width'], c['height']) != (t['width'], t['height']):
            aspect = abs(c['width'] / c['height'] - t['width'] / t['height']) > 1e-3
            ops.append({'op': 'resize', 'feature': c['key'], 'from': [c['width'], c['height']],
                        'to': [t['width'], t['height']], 'aspect_changes': aspect,
                        'why': 'another image size' + ('; the aspect ratio changes, so pad or crop to keep the image '
                                                       'undistorted' if aspect else '')})
        if _rate_differs(c.get('fps'), t.get('fps')):
            ops.append({'op': 'resample', 'feature': c['key'], 'from_hz': c['fps'], 'to_hz': t['fps'],
                        'why': 'another camera frame rate: frames dropped' if c['fps'] > t['fps'] else
                               'another camera frame rate: frames repeated (no new information)'})
        if c.get('codec') and t.get('codec') and c['codec'] != t['codec']:
            ops.append({'op': 'reencode', 'feature': c['key'], 'from': c['codec'], 'to': t['codec'],
                        'why': 'another video codec' + (' (the target\'s is a default, not a recording)'
                                                        if t.get('codec_source') == CODEC_PLAN_SOURCE else '')})
    for t in missing:
        bad.append({'what': t['key'], 'why': f'the target\'s {t["role"]} camera {t["key"]} has no counterpart in this '
                                             'cell: that view was not recorded (drop it from the pooled set, or '
                                             'mask it for this cell)'})
    for c in extra:
        ops.append({'op': 'drop', 'feature': c['key'], 'why': 'a camera the target does not have'})
    tf, cf = (target.get('fps') or {}).get('value'), (cell.get('fps') or {}).get('value')
    if _rate_differs(cf, tf):
        ops.append({'op': 'resample', 'feature': 'rows', 'from_hz': cf, 'to_hz': tf,
                    'why': 'another dataset frame rate: interpolate the state; resample the actions as the policy '
                           'needs them'})
    elif tf is None or cf is None:
        warn.append('A dataset frame rate is not stated, so the rows cannot be checked.')
    for key, what in ((STATE_KEY, 'state'), (ACTION_KEY, 'action')):
        ts, cs = target.get(what) or {}, cell.get(what) or {}
        if not ts.get('count') and not cs.get('count'):
            continue
        if ts.get('count') != cs.get('count'):
            bad.append({'what': key, 'why': f'{cs.get("count") or 0} {what} values against the target\'s '
                                            f'{ts.get("count") or 0}: another embodiment; pooling needs a policy '
                                            'that takes several embodiments (padded or separate action spaces)'})
            continue
        tn, cn = ts.get('names') or [], cs.get('names') or []
        order = None                                   # target joint i is cell joint order[i] (None: by position)
        if tn and cn and tn != cn:
            if sorted(tn) == sorted(cn) and len(set(tn)) == len(tn):
                order = [cn.index(n) for n in tn]
                ops.append({'op': 'reorder', 'feature': key, 'index': order,
                            'why': 'the same joints in another order: target[i] = cell[index[i]]'})
            else:
                ops.append({'op': 'rename', 'feature': key, 'by': 'position', 'from': cn, 'to': tn,
                            'why': 'other joint names at the same count: taken to correspond by position (an '
                                   'assumption: check them)'})
        elif not tn or not cn:
            warn.append(f'{key}: the joint names are not stated, so they are taken to correspond by position.')
        tu, cu = ts.get('units'), cs.get('units')
        hint, t_hint = cs.get('units_hint'), ts.get('units_hint')
        if tu and cu and len(tu) == len(cu):
            # joint i of the target against the cell joint it maps to (after the reorder), and only where both
            # units are stated: a unit not stated (None) is not "another kind"
            src = order or list(range(len(tu)))
            diff = [tn[i] if tn else str(i) for i in range(len(tu))
                    if tu[i] is not None and cu[src[i]] is not None and tu[i] != cu[src[i]]]
            if diff:
                bad.append({'what': key, 'why': f'joints of another kind (revolute against prismatic): '
                                                f'{", ".join(diff[:6])}'})
        elif (hint or t_hint) and what == 'state':
            if hint and t_hint:
                warn.append(f'{key}: both recordings look like degrees (stored values beyond ±2π): no conversion '
                            'between them, but check that both are in degrees.')
            else:                                      # the cell's values to the target's (a heuristic, said)
                if hint:
                    fr, to, joints, why = 'deg', 'rad', list(hint['joints']), hint['source']
                else:                                  # the target's joints named as the cell's joints they map to
                    src = order or list(range(len(tn) if tn else ts.get('count') or 0))
                    joints = []
                    for j in t_hint['joints']:
                        i = tn.index(j) if tn and j in tn else (int(j) if str(j).isdigit() else None)
                        if i is not None and i < len(src):
                            joints.append(cn[src[i]] if cn and src[i] < len(cn) else str(src[i]))
                    fr, to, why = 'rad', 'deg', f'the target\'s {t_hint["source"]}'
                ops.append({'op': 'convert', 'feature': key, 'joints': joints, 'from': fr, 'to': to, 'why': why})
                ops.append({'op': 'convert', 'feature': ACTION_KEY, 'joints': joints, 'from': fr, 'to': to,
                            'why': 'as the state (the same joints)'})
        elif what == 'state':
            warn.append('The units of a recording are not stated (LeRobot stores none): check that both store '
                        'radians and metres.')
    ta, ca = ((target.get('action') or {}).get('space') or {}).get('value'), \
        ((cell.get('action') or {}).get('space') or {}).get('value')
    if ta and ca and ta != ca:
        bad.append({'what': 'action space', 'why': f'{ca} against the target\'s {ta}: converting between action '
                                                   'spaces needs the robot\'s kinematics and is not a mapping step'})
    elif (target.get('action') or {}).get('count') and (not ta or not ca):
        warn.append('An action space is not stated, so it is taken to be the same.')
    tg, cg = target.get('gripper') or {}, cell.get('gripper') or {}
    if tg.get('joints') and cg.get('joints'):
        tr, cr = tg.get('range'), cg.get('range')
        if tr and cr:
            if tg.get('type') and cg.get('type') and tg['type'] != cg['type'] or \
                    max(abs(tr[0] - cr[0]), abs(tr[1] - cr[1])) > 1e-6:
                ops.append({'op': 'rescale', 'feature': 'gripper', 'joint': cg['joints'][0], 'to_joint':
                            tg['joints'][0], 'from_range': cr, 'to_range': tr, 'from_type': cg.get('type'),
                            'to_type': tg.get('type'),
                            'why': 'another gripper encoding: map the cell\'s [closed, open] range linearly onto the '
                                   'target\'s (an assumption: opening taken as linear in the joint)'})
        else:
            warn.append('A gripper range is not stated, so its encoding cannot be compared.')
    elif tg.get('joints') != cg.get('joints') and (tg.get('joints') or cg.get('joints')):
        warn.append('Only one of the two names a gripper joint (grip, finger or jaw), so the gripper encoding cannot '
                    'be compared.')
    th, ch = (target.get('control_hz') or {}).get('value'), (cell.get('control_hz') or {}).get('value')
    if _rate_differs(ch, th):
        ops.append({'op': 'resample', 'feature': 'actions', 'from_hz': ch, 'to_hz': th,
                    'why': 'another control rate: an action chunk spans another time; resample the actions'})
    verdict = VERDICTS[2] if bad else (VERDICTS[1] if ops else VERDICTS[0])
    return {'verdict': verdict, 'operations': ops, 'unmappable': bad, 'warnings': warn, 'camera_map': cmap}


def pooling(schemas: list, names: list, target: int = 0) -> dict:
    """The pooling verdict and mapping of every cell's schema against the target cell's (module docstring)."""
    if not schemas:
        raise StudyError('There is no cell to pool.')
    target = int(target) if 0 <= int(target) < len(schemas) else 0
    t = schemas[target]
    cells = []
    for k, (sc, name) in enumerate(zip(schemas, names)):
        r = rules(t, sc) if k != target else {'verdict': VERDICTS[0], 'operations': [], 'unmappable': [],
                                                'warnings': [], 'camera_map': {c['key']: c['key']
                                                                               for c in sc['cameras']}}
        cells.append({'index': k, 'name': name, 'target': k == target, 'source': sc['source'],
                      'what': sc.get('what', ''), **r, 'notes': list(sc.get('notes') or [])})
    n_bad = sum(c['verdict'] == VERDICTS[2] for c in cells)
    n_map = sum(c['verdict'] == VERDICTS[1] for c in cells)
    return {'schema': SCHEMA, 'target': {'index': target, 'name': names[target], **_brief(t)},
            'cells': cells, 'schemas': [_brief(s) for s in schemas],
            'counts': {v: sum(c['verdict'] == v for c in cells) for v in VERDICTS},
            'poolable': n_bad == 0, 'operations': sum(len(c['operations']) for c in cells),
            'text': ('Every cell pools as is.' if not n_bad and not n_map else
                     f'{len(cells) - n_bad} of {len(cells)} cells pool' + (' after mapping' if n_map else '')
                     + (f'; {n_bad} cannot be pooled' if n_bad else '') + '.')}


def _brief(sc: dict) -> dict:
    return {'source': sc['source'], 'what': sc.get('what', ''), 'format': sc.get('format'),
            'version': sc.get('version'), 'fps': sc.get('fps'), 'cameras': sc['cameras'], 'state': sc['state'],
            'action': sc['action'], 'gripper': sc.get('gripper'), 'control_hz': sc.get('control_hz')}
