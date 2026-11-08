"""The assistant's tools: what the in-app assistant and Claude over MCP (dexvantage.mcp) can read and change, over a
session (assistant/session.py: the window's project or a project file).

    TOOLS = {name: Tool(name, title, description, input_schema, fn, changes)}
    call(session, name, args) -> dict          the tool's JSON result; errors as {'error': 'a plain sentence'}
    validate_args(name, args) -> [problems]    the input checked against the tool's JSON Schema first
    input_schemas() -> {name: schema}

Every figure a tool returns comes from a study's result (core/jobs.py through the session's result cache, or the
result the window already holds for the design as it is now) or from the project and the cited library entries;
none is computed or filled in here. A study that refuses gives its own sentence as the error. Units: metres,
degrees (angles in the project are radians; the tools say which), percent, seconds; frames as core/frames.py
(world = the primary robot's base, REP-103: x forward, y left, z up).

What changes the project: edit_project (one undo step "Claude · …", returning the requirement metrics before and
after), undo_last_change (the assistant's own last step only). simulate_changes tries a change on a copy;
propose_plan hands a plan to the user to approve (the Copilot: the long studies, exports and edits it lists run only
after approval); run_study and the reading tools compute on a snapshot; export_files writes files to a folder;
show_in_app shows something in the window. Locked sensors and robots stay as the user set them: any change to one is
refused (mcp/edits.py).

Long studies (LONG: GPU renders and grasp sweeps, dataset reads; tens of seconds on the samples) run from a tool
only when the session allows it (the MCP server: the client asks the user before each call); the in-app assistant
proposes a plan instead, unless the page already holds a result for the design as it is now.

Ported from Vantage assistant/tools.py (the Tool registry, the schema check, errors as plain sentences, read tools
over a snapshot, simulate/propose), copilot_tools.py (propose_plan: a plan the user approves) and page_guide.py
(get_page_context). The tools themselves are DexVantage's studies.

Known limits: the briefs (assistant/figures.py) pick the figures a reader needs first; run_study with full=true
returns the whole result compacted. Studies run on the project as stored (the window's joint sliders are view state,
not project content), so a page run at another pose shows other figures; get_page_context reads what the page shows.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import secrets
import sys
import traceback
from dataclasses import dataclass

from ..core import jobs
from . import figures as F
from .session import NeedsWindow, SessionError

# Studies that take tens of seconds or more on the bundled samples (measured 2026-11-07: grasp_visibility 11.5 s on
# the SO-100 and 42.4 s on the UR5e; groundtruth 1.8-8.9 s with GPU renders; audit 9-15 s and plan_vs_recording
# 2.5-16 s reading a dataset; silhouette_calib 10-19 s; teleop 5-11 s; handeye_plan 2.5-6.9 s; the hand-eye solve
# and the multi-finger study render or sweep likewise; the Phase 9 studies run many studies each: tolerance 64 draws
# 81 s and optimise 105 s on the UR5e, design_agent 237 s, an experiment one run per row, 20-30 s for 40 runs on
# ALOHA 2): a plan the user approves runs them in the app.
LONG = ('grasp_visibility', 'groundtruth', 'audit', 'plan_vs_recording', 'silhouette_calib', 'teleop', 'handeye_plan',
        'handeye_solve', 'dex_occlusion', 'tolerance', 'experiments', 'optimise', 'design_agent',
        'cells',                       # the cell comparison: figures on every cell (core/cells.py)
        'sim2real',                    # the sim-to-real gap: renders and a silhouette calibration (5-6 s on the UR5e)
        'splat_views')                 # a 3D Gaussian Splatting fit: about a minute on a GPU (docs/models/splat.md)
PLAN_KINDS = ('study', 'export', 'edit')


class ToolError(Exception):
    """A plain sentence: why the tool cannot answer (nothing was changed)."""


@dataclass
class Tool:
    name: str
    title: str
    description: str
    input_schema: dict
    fn: object
    changes: bool = False            # it changes the project (or writes files, or shows in the window)


# =========================================================================== helpers
def _r(v, n=4):
    return F._round(v, n)                                # noqa: SLF001


def _deg(rad):
    return None if rad is None else _r(math.degrees(float(rad)), 3)


def _pose(pose) -> dict:
    from ..core.frames import rpy_from_quat
    r, p, y = rpy_from_quat(pose.quaternion)
    return {'position_m': [_r(x, 5) for x in pose.position], 'rpy_deg': [_deg(r), _deg(p), _deg(y)]}


def _entry(model_id):
    from ..core import library as LB
    return LB.get(model_id) if model_id else None


def _sensor(p, ident):
    from ..core import factors as FX
    try:
        return FX.find_sensor(p, str(ident or ''))
    except FX.FactorError as e:
        raise ToolError(f'{e} (get_overview lists the sensors).') from None


def _robot_of(p, ident=None):
    if not p.robots:
        raise ToolError('The project has no robot yet: import one (Robot ▸ Import robot…) or open a sample.')
    if not ident:
        return p.primary_robot
    hit = [r for r in p.robots if r.uid == ident] or [r for r in p.robots if r.name == ident]
    if not hit:
        raise ToolError(f'The project has no robot "{ident}" (it has: {", ".join(r.name for r in p.robots)}).')
    return hit[0]


def _studies():
    jobs.load_studies()
    return jobs.STUDIES


def _doc(session, project, sid: str, settings=None, use_last=True) -> dict:
    """Study sid's result on project: the window's own result for the design as it is now, else run (through the
    session's cache); refused (ToolError) for a long study the session may not run, or with the study's sentence."""
    info = _studies().get(sid)
    if info is None:
        raise ToolError(f'There is no study {sid} (run_study lists them).')
    if use_last and not settings:
        doc = session.last_result(sid)
        if doc is not None:
            return doc
    if sid in LONG and not session.allow_long:
        raise ToolError(f'{info.title} is long work (it takes tens of seconds or more): propose_plan a study step '
                        'for the user to approve, or ask the user to run it on its page.')
    if settings:
        project.analysis.studies[sid] = {**(project.analysis.studies.get(sid) or {}), **settings}
    try:
        return jobs.run_study(sid, project, cache=session.cache)
    except jobs.StudyError as e:
        raise ToolError(str(e)) from None


def _study_tool(sid: str):
    def fn(session, full=False, **_):
        doc = _doc(session, session.project(), sid)
        out = F.brief(sid, doc)
        if full:
            out['result'] = F.compact({k: v for k, v in doc.items() if k not in ('cached', 'revision')},
                                      items=12, max_depth=5)
        return out
    fn.__name__ = f'get_{sid}'
    return fn


# =========================================================================== reading
def get_overview(session, **_):
    from ..core import metrics as M
    p = session.project()
    sensors = []
    for s in p.variant.sensors:
        e = _entry(s.model_id)
        r = next((x for x in p.robots if x.uid == s.mount.robot_uid), None)
        sensors.append({'uid': s.uid, 'name': s.name, 'kind': s.kind, 'model_id': s.model_id or None,
                        'model': e.title if e is not None else None, 'parent': s.mount.parent_frame,
                        'robot': r.name if r is not None else None, **_pose(s.mount.pose), 'locked': s.locked,
                        'enabled': s.enabled, 'stream': list(s.stream) or 'largest', 'calibrated': bool(s.calibration)})
    st = _studies()
    M.load_metrics()
    return {
        'project': p.name, 'file': os.path.basename(p.path) if p.path else None, 'where': session.where(),
        'mode': session.mode, 'unsaved_changes': session.unsaved(),
        'robots': [{'uid': r.uid, 'name': r.name, 'primary': r.primary,                 # moving joints, as the
                    'joints': sum(j.get('type') in jobs.MOVING_JOINTS for j in r.joints),   # Outliner counts them
                    'fixed_or_held_joints': sum(j.get('type') not in jobs.MOVING_JOINTS for j in r.joints),
                    'links': len(r.links), 'gripper_frame': r.gripper_frame or None, 'locked': r.locked}
                   for r in p.robots],
        'variant': {'uid': p.variant.uid, 'name': p.variant.name},
        'variants': [{'uid': v.uid, 'name': v.name, 'sensors': len(v.sensors)} for v in p.variants],
        'sensors': sensors,
        'workspace': {'volumes': [{'uid': v.uid, 'name': v.name, 'kind': v.kind, 'center_m': list(v.center),
                                   'size_m': list(v.size)} for v in p.workspace.volumes],
                      'objects': [{'uid': o.uid, 'name': o.name} for o in p.workspace.objects],
                      'lights': len(p.workspace.lights)},
        'tasks': {'grasp_sets': [{'uid': g.get('uid'), 'name': g.get('name'), 'poses': len(g.get('poses') or [])}
                                 for g in p.tasks.grasp_sets if isinstance(g, dict)],
                  'trajectories': [{'uid': t.get('uid'), 'name': t.get('name')} for t in p.tasks.trajectories
                                   if isinstance(t, dict)],
                  'teleop_devices': len(p.tasks.teleop_devices)},
        'policies': [{'uid': x.uid, 'name': x.name, 'library_id': x.library_id} for x in p.policies],
        'requirements': [{'name': r.name, 'metric': r.metric, 'op': r.op, 'threshold': r.threshold, 'unit': r.unit,
                          'default': r.is_default} for r in p.requirements],
        'hosts': len(p.rig.get('hosts') or []), 'cables': len(p.cables),
        'datasets': [{'uid': d.get('uid'), 'name': d.get('name'), 'format': d.get('format')}
                     for d in (p.data or {}).get('datasets') or []],
        'studies': [{'id': k, 'title': v.title, 'long': k in LONG, 'needs_robot': v.needs_robot}
                    for k, v in st.items() if k != 'summary'],
        'metrics': len(M.METRICS),
        'frame': 'world = the primary robot\'s base, REP-103 (x forward, y left, z up), metres; mount rpy in degrees '
                 '(URDF convention R = Rz·Ry·Rx) in the parent frame',
    }


def get_robot(session, robot=None, **_):
    p = session.project()
    r = _robot_of(p, robot)
    joints = [{'name': j.get('name'), 'type': j.get('type'), 'parent': j.get('parent'), 'child': j.get('child'),
               'lower_rad': _r(j.get('lower'), 6), 'upper_rad': _r(j.get('upper'), 6),
               'lower_deg': _deg(j.get('lower')) if j.get('type') in ('revolute', 'continuous') else None,
               'upper_deg': _deg(j.get('upper')) if j.get('type') in ('revolute', 'continuous') else None,
               'velocity': _r(j.get('velocity')), 'effort': _r(j.get('effort'))} for j in r.joints]
    return {'uid': r.uid, 'name': r.name, 'primary': r.primary, 'locked': r.locked, 'base': _pose(r.base),
            'source': {'kind': r.source.kind, 'file': r.source.file, 'license': r.source.license,
                       'sha256': r.source.sha256, 'note': F.short(r.source.note, 400)},
            'joints': joints, 'links': [lk.get('name') for lk in r.links],
            'home_rad': {k: _r(v, 6) for k, v in r.home.items()}, 'gripper_frame': r.gripper_frame or None,
            'fingertip_frames': list(r.fingertip_frames),
            'contact_pads': [{'link': c.get('link'), 'name': c.get('name')} for c in r.contact_pads],
            'note': 'Joint limits, velocities and efforts are as the model file states them (radians, m/s or rad/s, '
                    'N or N·m); None where the file states none.'}


def get_sensor(session, sensor='', **_):
    p = session.project()
    s = _sensor(p, sensor)
    e = _entry(s.model_id)
    r = next((x for x in p.robots if x.uid == s.mount.robot_uid), None)
    out = {'uid': s.uid, 'name': s.name, 'kind': s.kind, 'locked': s.locked, 'enabled': s.enabled,
           'stream': list(s.stream) or 'largest', 'calibrated': bool(s.calibration),
           'mount': {'parent': s.mount.parent_frame, 'robot': r.name if r is not None else None, **_pose(s.mount.pose),
                     'tolerance': s.mount.tolerance.to_dict(), 'tolerance_note': 'the mount tolerance (a default '
                     'of 1 mm and 0.5° unless the user set it)', 'bracket': s.mount.bracket or None},
           'used_by_policies': [x.name for x in p.policies if not x.cameras or s.uid in x.cameras],
           'cables': [c.get('name') for c in p.cables if c.get('sensor_uid') == s.uid]}
    if e is None:
        out['model'] = None
        out['note'] = 'No library model is chosen: the studies use a default camera and say so.'
        return out
    d = e.data
    out['model'] = {'id': e.id, 'title': e.title, 'kind': e.kind, 'maker': e.maker, 'family': d.get('family'),
                    'interface': d.get('interface'), 'power_w': d.get('power_w'), 'mass_g': d.get('mass_g'),
                    'size_mm': d.get('size_mm'), 'imagers': F.compact(d.get('imagers'), items=6, max_depth=3),
                    'depth': F.compact(d.get('depth'), items=8, max_depth=3),
                    'streams': F.compact(d.get('streams'), items=8, max_depth=2),
                    'tactile': F.compact(d.get('tactile'), max_depth=2), 'sync': F.compact(d.get('sync'), max_depth=2),
                    'note': F.short(e.note, 600), 'custom': e.is_custom,
                    'not_stated': sorted(k for k in ('power_w', 'mass_g', 'size_mm', 'interface')
                                         if d.get(k) is None),
                    'sources': [{'title': x.get('title'), 'url': x.get('url'), 'read': x.get('read'),
                                 'covers': F.short(x.get('covers', ''), 200)} for x in e.sources]}
    out['note'] = ('Library figures are the maker\'s, as the cited sources state them; None is "not stated by the '
                   'maker".')
    return out


def get_page_context(session, **_):
    pc = session.page_context()
    if not pc:
        return {'page': None, 'note': 'No page context: there is no DexVantage window, or no analysis page is shown.'}
    return pc


def list_metrics(session, study=None, **_):
    from ..core import metrics as M
    M.load_metrics()
    rows = [{'id': m.id, 'title': m.title, 'unit': m.unit, 'better': m.better, 'study': m.study,
             'long': m.study in LONG, 'note': m.note} for m in M.METRICS.values() if not study or m.study == study]
    out = {'metrics': sorted(rows, key=lambda r: r['id'])}
    if not rows:
        out['note'] = 'No metrics are registered' + (f' for {study}' if study else '') + '.'
    return out


def _metric_rows(session, project, ids, use_last) -> dict:
    """{id: row} as core.metrics.evaluate gives them, each study run once, through _doc (so long studies and the
    window's own results are handled as everywhere else)."""
    from ..core import metrics as M
    M.load_metrics()
    out, docs = {}, {}
    for mid in ids:
        m = M.METRICS.get(mid)
        if m is None:
            out[mid] = {'value': None, 'status': 'unknown', 'reason': f'There is no metric called {mid} '
                                                                      '(list_metrics lists them).'}
            continue
        row = {'value': None, 'unit': m.unit, 'better': m.better, 'study': m.study, 'title': m.title}
        if m.study not in docs:
            try:
                docs[m.study] = ('ok', _doc(session, project.snapshot(), m.study, use_last=use_last))
            except ToolError as e:
                docs[m.study] = ('refused', str(e))
        kind, doc = docs[m.study]
        if kind == 'refused':
            row.update(status='refused', reason=doc)
        else:
            try:
                v = m.extract(doc)
            except (KeyError, TypeError, ValueError, IndexError):
                v = None
            row.update(value=None if v is None else _r(float(v), 6), status='ok' if v is not None else 'not stated')
        out[mid] = row
    return out


def evaluate_metrics(session, metrics=None, **_):
    if not metrics:
        raise ToolError('Name the metrics to evaluate (list_metrics lists them).')
    return {'metrics': _metric_rows(session, session.project(), list(metrics), use_last=True)}


def _passes(op, value, threshold):
    if value is None:
        return None
    return {'>=': value >= threshold, '<=': value <= threshold, '>': value > threshold, '<': value < threshold,
            '==': math.isclose(value, threshold, rel_tol=1e-9, abs_tol=1e-12)}.get(op)


def _before_after(session, p0, p1, extra, live_before=True) -> dict:
    ids = list(dict.fromkeys([r.metric for r in p0.requirements if r.metric] + list(extra or [])))
    if not ids:
        return {'metrics': {}, 'requirements': [],
                'note': 'The project has no requirements: name metrics (list_metrics) to compare before and after.'}
    before = _metric_rows(session, p0, ids, use_last=live_before)
    after = _metric_rows(session, p1, ids, use_last=False)
    reqs = []
    for r in p1.requirements:
        if not r.metric:
            continue
        b, a = before.get(r.metric, {}).get('value'), after.get(r.metric, {}).get('value')
        reqs.append({'name': r.name, 'metric': r.metric, 'op': r.op, 'threshold': r.threshold, 'unit': r.unit,
                     'default_threshold': r.is_default, 'before': b, 'after': a,
                     'pass_before': _passes(r.op, b, r.threshold), 'pass_after': _passes(r.op, a, r.threshold)})
    rows = {mid: {'title': after[mid].get('title'), 'unit': after[mid].get('unit'), 'better': after[mid].get('better'),
                  'before': before[mid].get('value'), 'after': after[mid].get('value'),
                  'status_before': before[mid].get('status'), 'status_after': after[mid].get('status'),
                  **({'reason': after[mid]['reason']} if after[mid].get('reason') else {})} for mid in ids}
    return {'metrics': rows, 'requirements': reqs}


# =========================================================================== changing
def _trial(session, changes, ops):
    from ..mcp import edits as E
    base = session.project()
    try:
        ed = E.apply(base, changes, ops)
    except E.EditError as e:
        raise ToolError(str(e)) from None
    return base, ed


def simulate_changes(session, changes=None, ops=None, metrics=None, **_):
    base, ed = _trial(session, changes, ops)
    out = {'would_change': ed.lines, 'sections': ed.keys, **_before_after(session, base, ed.project, metrics)}
    out['note'] = ' '.join(x for x in (out.get('note', ''), 'Nothing was changed: edit_project makes it (one undo '
                                                            'step).') if x)
    return out


def edit_project(session, changes=None, ops=None, metrics=None, **_):
    from ..mcp import edits as E
    base, ed = _trial(session, changes, ops)
    if not ed.keys:
        return {'changed': ed.lines, 'undo_step': None, 'note': 'Nothing differs, so no undo step was made.'}
    try:
        label = E.commit(session, ed, base, changes, ops)
    except E.EditError as e:
        raise ToolError(str(e)) from None
    out = {'changed': ed.lines, 'undo_step': label, **_before_after(session, base, session.project(), metrics)}
    saved = (' Saved to the file.' if session.mode == 'file' and not session.unsaved() else
             ' Not saved yet.' if session.unsaved() else '')
    out['note'] = ' '.join(x for x in (out.get('note', ''), 'undo_last_change takes it back.' + saved) if x)
    return out


def undo_last_change(session, **_):
    return {'result': session.undo()}


def run_study(session, study='', settings=None, full=False, **_):
    st = _studies()
    if not study:
        return {'studies': [{'id': k, 'title': v.title, 'long': k in LONG, 'needs_robot': v.needs_robot,
                             'batch_flag': f'--{v.batch}' if v.batch else None} for k, v in st.items()],
                'note': 'Name a study to run it; settings (optional) apply to this run only.'}
    if settings is not None and not isinstance(settings, dict):
        raise ToolError('settings is an object of the study\'s settings for this run.')
    doc = _doc(session, session.project(), study, settings=settings or None)
    out = F.brief(study, doc)
    if full:
        out['result'] = F.compact({k: v for k, v in doc.items() if k not in ('cached', 'revision')}, items=12,
                                  max_depth=5)
    if settings:
        out['note'] = 'The settings applied to this run only; edit_project with setting:<study>.<key> keeps them.'
    return out


def export_files(session, formats=None, folder=None, options=None, **_):
    from .. import workdir
    from ..core import exporters as X
    X.load_exporters()
    every = [{'id': i.id, 'title': i.title, 'needs_robot': i.needs_robot, 'description': i.description}
             for i in X.EXPORTERS.values()]
    if not formats:
        return {'formats': every, 'note': 'Name formats (ids) and a folder to write them.' if every else
                'No export formats are registered in this build.'}
    formats = X.resolve(formats)                        # aliases ('usd' is the Isaac export), as export() takes them
    unknown = [f for f in formats if f not in X.EXPORTERS]
    if unknown:
        raise ToolError(f'DexVantage has no export format {", ".join(unknown)} (it has: '
                        f'{", ".join(sorted(X.EXPORTERS)) or "none"}).')
    p = session.project()
    if not folder:
        folder = (os.path.join(os.path.dirname(p.path), f'{workdir.slug(p.name)}-export') if p.path else
                  workdir.project_dir(p, None, 'exports', workdir.slug(p.variant.name)))
    folder = os.path.abspath(os.path.expanduser(str(folder)))
    try:
        man = X.export(p, folder, list(formats), options=options if isinstance(options, dict) else None)
    except jobs.StudyError as e:
        raise ToolError(str(e)) from None
    return {'folder': folder, 'formats': {k: {kk: v.get(kk) for kk in ('title', 'status', 'reason', 'files', 'notes')
                                              if v.get(kk) is not None} for k, v in man['formats'].items()},
            'files': sorted(man['files']), 'manifest': 'export_manifest.json'}


SHOW = ('open_page', 'select', 'frame_all', 'message')


def show_in_app(session, action='', page=None, sensor=None, robot=None, text=None, **_):
    if action not in SHOW:
        raise ToolError(f'action must be one of {", ".join(SHOW)}.')
    req = {'action': action}
    if action == 'open_page':
        from .page_guide import GUIDES
        if not page:
            raise ToolError(f'Name the page (one of {", ".join(sorted(GUIDES))}).')
        req['page'] = str(page)
    elif action == 'select':
        p = session.project()
        if sensor:
            req['what'] = ('sensor', _sensor(p, sensor).uid)
        elif robot:
            req['what'] = ('robot', _robot_of(p, robot).uid)
        else:
            raise ToolError('Name the sensor or the robot to select.')
    elif action == 'message':
        if not text:
            raise ToolError('Give the text to show.')
        req['text'] = str(text)
    out = session.show(req)
    if isinstance(out, dict) and 'error' in out:
        raise ToolError(out['error'])
    return {'done': action, **{k: v for k, v in req.items() if k != 'action'}}


def _plan_step(session, p, i, step) -> dict:
    if not isinstance(step, dict) or step.get('kind') not in PLAN_KINDS:
        raise ToolError(f'Step {i + 1}: kind must be one of {", ".join(PLAN_KINDS)}.')
    kind = step['kind']
    out = {'kind': kind, 'label': F.short(step.get('label') or '', 120)}
    if kind == 'study':
        st = _studies()
        sid = step.get('study')
        if sid not in st:
            raise ToolError(f'Step {i + 1}: there is no study {sid} (run_study lists them).')
        if st[sid].needs_robot and not p.robots:
            raise ToolError(f'Step {i + 1}: {st[sid].title} needs a robot, and the project has none.')
        out.update(study=sid, settings=dict(step.get('settings') or {}), long=sid in LONG)
        out['label'] = out['label'] or st[sid].title
    elif kind == 'export':
        from ..core import exporters as X
        X.load_exporters()
        fm = X.resolve(step.get('formats') or [])
        bad = [f for f in fm if f not in X.EXPORTERS]
        if not fm or bad:
            raise ToolError(f'Step {i + 1}: name export formats that exist ({", ".join(sorted(X.EXPORTERS)) or "none"}'
                            ' are registered).')
        out.update(formats=fm)
        if step.get('folder'):
            out['folder'] = str(step['folder'])
        out['label'] = out['label'] or f'Export {", ".join(fm)}'
    else:
        from ..mcp import edits as E
        try:
            ed = E.apply(p, step.get('changes'), step.get('ops'))
        except E.EditError as e:
            raise ToolError(f'Step {i + 1}: {e}') from None
        out.update(changes=step.get('changes'), ops=step.get('ops'), would_change=ed.lines)
        out['label'] = out['label'] or E.title(ed.lines)
    return out


def propose_plan(session, title='', rationale='', steps=None, **_):
    if not title or not steps:
        raise ToolError('A plan needs a title and at least one step.')
    p = session.project()
    plan = {'schema': 'dexvantage.plan/1', 'plan': 'plan-' + secrets.token_hex(6), 'version': 1,
            'title': F.short(title, 120), 'rationale': F.short(rationale or '', 600),
            'project': p.name, 'steps': [_plan_step(session, p, i, s) for i, s in enumerate(steps[:20])],
            'status': 'proposed', 'created_utc': _dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}
    shown = session.propose(plan)
    return {'plan': plan, **(shown or {})}


# =========================================================================== the registry
_NONE = {'type': 'object', 'properties': {}, 'additionalProperties': False}


def _obj(props: dict, required=()) -> dict:
    s = {'type': 'object', 'properties': props, 'additionalProperties': False}
    if required:
        s['required'] = list(required)
    return s


_FULL = {'full': {'type': 'boolean', 'description': 'also return the whole result, compacted (default false)'}}
_CHANGES = {'type': 'object', 'additionalProperties': True,
            'description': 'factor path -> value (core/factors.py): sensor:<uid or name>.x|y|z (m in the '
                           'parent frame), .roll|.pitch|.yaw (degrees), .dx|.dy|.dz (a shift, m), '
                           '.droll|.dpitch|.dyaw (a turn about the sensor\'s own axes, degrees), .model (a library '
                           'id), .parent (world, cage, tripod, table or a link name), .enabled, .stream ([width, '
                           'height]); setting:<study>.<key>; json:/a/b (a JSON pointer into the project)'}
_OP = _obj({'op': {'type': 'string', 'enum': ['add_sensor', 'remove_sensor', 'add_variant', 'activate_variant',
                                              'rename_variant']},
            'model_id': {'type': 'string', 'description': 'add_sensor: a library id'},
            'name': {'type': 'string', 'description': 'add_sensor, add_variant, rename_variant: the name'},
            'parent': {'type': 'string', 'description': 'add_sensor: world, cage, tripod, table or a link name'},
            **{k: {'type': 'number', 'description': f'add_sensor: {k} ({"m" if k in "xyz" else "degrees"})'}
               for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')},
            'sensor': {'type': 'string', 'description': 'remove_sensor: uid or name'},
            'variant': {'type': 'string', 'description': 'activate_variant, rename_variant: uid or name'},
            'copy': {'type': 'boolean', 'description': 'add_variant: copy the active rig (default true)'},
            'activate': {'type': 'boolean', 'description': 'add_variant: make it active (default true)'}},
           ('op',))
_OPS = {'type': 'array', 'items': _OP, 'description': 'structural operations, applied first, in order'}
_METRICS = {'type': 'array', 'items': {'type': 'string'},
            'description': 'metric ids to compare besides the requirements\' (list_metrics)'}
_STEP = _obj({'kind': {'type': 'string', 'enum': list(PLAN_KINDS)},
              'label': {'type': 'string', 'description': 'a few words'},
              'study': {'type': 'string', 'description': 'study: the study id'},
              'settings': {'type': 'object', 'additionalProperties': True, 'description': 'study: its settings'},
              'formats': {'type': 'array', 'items': {'type': 'string'}, 'description': 'export: format ids'},
              'folder': {'type': 'string', 'description': 'export: the folder'},
              'changes': _CHANGES, 'ops': _OPS}, ('kind',))

_STUDY_TOOLS = [
    ('get_coverage', 'coverage', 'Coverage and resolution in the task volume: how much each camera sees (by two or '
                                 'more, by one only, by none), pixels per mm, hidden and outside-image shares.'),
    ('get_depth_validity', 'depth_validity', 'Depth validity: where each depth sensor returns valid depth (min-Z at '
                                             'the stream, range, projector cone, occlusion, materials).'),
    ('get_grasp_visibility', 'grasp_visibility', 'Grasp self-occlusion: the share of reachable grasps whose '
                                                 'fingertips stay visible, with its interval, per camera, and the '
                                                 'best move. Long work.'),
    ('get_teleop_overlap', 'teleop', 'Teleop overlap: the share of the task volume a teleop device leads without a '
                                     'dead zone. Long work.'),
    ('get_cable_flex', 'cables', 'Cable flex across joints: pulls, kinks and snags of each cable over the joint '
                                 'ranges (suggested routes when the project has no cable).'),
    ('get_bandwidth', 'bandwidth', 'Bandwidth, hosts and power: data rates, controller and link loads, bus power, '
                                   'disk rate and size.'),
    ('get_sync', 'time_sync', 'Time sync: each camera stream\'s skew against joint_states and the label error it '
                              'makes along the trajectories.'),
    ('get_vla_tokens', 'vla_tokens', 'VLA token resolution: how many ViT tokens a critical object and the fingertips '
                                     'cover in each policy input.'),
    ('get_action_chunk', 'action_chunk_sync', 'Action-chunk latency: how far the gripper close lands from contact '
                                              'given each policy\'s loop.'),
    ('get_handeye_plan', 'handeye_plan', 'Hand-eye pose plan: the planned calibration poses and the predicted '
                                         'uncertainty of the camera mount. Long work.'),
    ('get_audit', 'audit', 'Dataset audit: whether the recorded dataset meets the plan (drops, desync, mount slip), '
                           'per episode. Long work.'),
    ('get_sim2real', 'sim2real', 'Sim-to-real gap: a recorded episode rendered again through the rig and compared per '
                                 'camera (image gap, exposure, blur, noise, silhouette IoU, mount offset, latency, '
                                 'MMD), with the causes ranked and the change that closes each. Long work.'),
    ('get_campaign', 'campaign', 'Data-collection campaign plan: episodes, days, storage and the status against the '
                                 'budget, the counterbalanced schedule over operators and days, and the checklists.'),
    ('get_splat_views', 'splat_views', 'Splat views: a robot-aligned 3D Gaussian Splatting rig fitted to a recorded '
                                       'episode (or a MuJoCo render of the project, a synthetic demonstration), its '
                                       'reconstruction quality on held-out frames (PSNR, SSIM, LPIPS) and the target '
                                       'camera\'s current and proposed pose. Every splat render is synthetic. Needs '
                                       'the splat extra and a CUDA device. Long work.'),
]

TOOLS: dict = {}


def _add(t: Tool):
    TOOLS[t.name] = t


_add(Tool('get_overview', 'Read the project overview',
          'The project: robots, the active variant\'s sensors (uid, name, model, mount pose, locked), the variants, '
          'task volumes and objects, grasp sets, trajectories, policies, requirements, datasets, the studies that '
          'can run, and the conventions (units, frames). Call it first.', _NONE, get_overview))
_add(Tool('get_robot', 'Read a robot',
          'A robot as its model file states it: joints with limits, velocities and efforts, links, home pose, '
          'gripper and fingertip frames, contact pads, the source file and its licence.',
          _obj({'robot': {'type': 'string', 'description': 'uid or name (default: the primary robot)'}}), get_robot))
_add(Tool('get_sensor', 'Read a sensor',
          'A sensor of the active variant: its mount (parent frame, pose, tolerance, bracket), stream, lock, and its '
          'library model\'s figures (imagers, depth block, streams, power, mass) with their cited sources.',
          _obj({'sensor': {'type': 'string', 'description': 'uid or name'}}, ('sensor',)), get_sensor))
for _name, _sid, _desc in _STUDY_TOOLS:
    _add(Tool(_name, f'Read {_sid.replace("_", " ")}', _desc + ' Figures from the study\'s result; a study that cannot '
              'run says why.', _obj(dict(_FULL)), _study_tool(_sid)))
_add(Tool('get_page_context', 'Read the page on screen',
          'The page the user is on in the DexVantage window: what it is for, its headline as shown (figure, unit, '
          'status word), its status (never run, stale, done, running), what it still needs, its figures and the '
          'pages before and after it. Cite these figures when the user asks about the page.', _NONE,
          get_page_context))
_add(Tool('list_metrics', 'List the metrics',
          'The named figures of merit (id, title, unit, which is better, the study that yields it) that requirements, '
          'edit_project and evaluate_metrics use.',
          _obj({'study': {'type': 'string', 'description': 'only this study\'s metrics'}}), list_metrics))
_add(Tool('evaluate_metrics', 'Evaluate metrics',
          'The value of each named metric on the project as it is now, from its study (refused studies say why; a '
          'figure the result does not state is None).',
          _obj({'metrics': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1}}, ('metrics',)),
          evaluate_metrics))
_add(Tool('edit_project', 'Change the project',
          'Change the project as ONE undo step ("Claude · …"), checked first on a copy (a bad change changes nothing '
          'and says why): changes sets factor paths, ops adds or removes sensors and variants. Returns the lines '
          'changed and the requirement metrics (and any metrics named) before and after. Locked sensors and robots '
          'stay as the user set them.', _obj({'changes': _CHANGES, 'ops': _OPS, 'metrics': _METRICS}), edit_project,
          changes=True))
_add(Tool('simulate_changes', 'Try a change',
          'What a change would do, without making it: the lines it would change and the requirement metrics (and any '
          'metrics named) before and after. Same input as edit_project.',
          _obj({'changes': _CHANGES, 'ops': _OPS, 'metrics': _METRICS}), simulate_changes))
_add(Tool('propose_plan', 'Propose a plan',
          'A plan for the user to approve (the Copilot): study runs (the long ones belong here), exports and edits, '
          'in order. Each step is checked now; nothing runs until the user approves it in the window.',
          _obj({'title': {'type': 'string'}, 'rationale': {'type': 'string', 'description': 'one or two sentences'},
                'steps': {'type': 'array', 'items': _STEP, 'minItems': 1, 'maxItems': 20}}, ('title', 'steps')),
          propose_plan, changes=True))
_add(Tool('run_study', 'Run a study',
          'Run one study on the project as it is now and return its figures (without study: the list of studies). '
          'settings apply to this run only. Long studies need the user\'s approval in the app (propose_plan).',
          _obj({'study': {'type': 'string'}, 'settings': {'type': 'object', 'additionalProperties': True},
                **_FULL}), run_study))
_add(Tool('export_files', 'Export files',
          'Write the project in export formats (core/exporters) to a folder, with a manifest of SHA-256 digests and '
          'provenance. Without formats: the format ids.',
          _obj({'formats': {'type': 'array', 'items': {'type': 'string'}},
                'folder': {'type': 'string', 'description': 'default: next to the project file, else the working '
                                                            'folder'},
                'options': {'type': 'object', 'additionalProperties': True,
                            'description': 'per format id: its options'}}), export_files, changes=True))
_add(Tool('show_in_app', 'Show in the window',
          'Show something in the DexVantage window: open a page, select a sensor or robot, frame the scene, or show '
          'a message. Needs the window open.',
          _obj({'action': {'type': 'string', 'enum': list(SHOW)}, 'page': {'type': 'string'},
                'sensor': {'type': 'string'}, 'robot': {'type': 'string'}, 'text': {'type': 'string'}}, ('action',)),
          show_in_app, changes=True))
_add(Tool('undo_last_change', 'Undo the last change',
          'Undo the last change when it was the assistant\'s own (edit_project). A step the user made after it is '
          'never undone from here.', _NONE, undo_last_change, changes=True))


def get_methods_paragraph(session, studies=None, **_):
    """The publication kit's methods paragraph (core/publication.py) from the results the window holds for the
    design as it is now, plus the quick studies named (long ones need a plan)."""
    from ..core import publication as PB
    p = session.project()
    jobs.load_studies()
    results = {}
    for sid in jobs.STUDIES:
        doc = session.last_result(sid)
        if isinstance(doc, dict):
            results[sid] = {k: v for k, v in doc.items() if k not in ('cached', 'revision')}
    for sid in studies or []:
        if sid not in results:
            results[sid] = _doc(session, p, str(sid))
    try:
        kit = PB.build(p, {'results': results, 'studies': [], 'sections': ['methods']})
    except jobs.StudyError as e:
        raise ToolError(str(e)) from None
    st = kit['status']
    return {'methods_tex': kit['parts'].get('methods', ''), 'studies': sorted(results),
            'citations': kit['checks']['keys']['used'], 'numbers': st['numbers'], 'numbers_traced': st['traced'],
            'status': st['word'], 'note': 'Every number in the paragraph is read from the project, a library entry, '
            'a result or the software versions (provenance.json of an exported kit lists where each came from). '
            'Export the whole kit with export_files format paper_kit.'}


_add(Tool('get_methods_paragraph', 'Read the methods paragraph',
          'A methods paragraph in LaTeX for a paper (robot and model sources, every sensor with maker, model, '
          'stream, frame rate, field of view and mount pose, the studies with their settings and validation status, '
          'the software versions), with \\cite keys, from the results the window holds for the design as it is '
          'now and the quick studies named.',
          _obj({'studies': {'type': 'array', 'items': {'type': 'string'},
                            'description': 'study ids to run first when the window holds no result for them'}}),
          get_methods_paragraph))


def get_cells(session, full=False, **_):
    """The multi-cell comparison (core/cells.py): the window's result for the design as it is now, else the study run
    on the project's stored cell set (long work: it evaluates figures on every cell)."""
    p = session.project()
    doc = session.last_result('cells')
    if doc is None:
        if not session.allow_long:
            raise ToolError('The cell comparison is long work (it evaluates figures on every cell): propose_plan a '
                            'study step "cells" for the user to approve, or ask the user to run it on the Cells page '
                            '(File ▸ Compare cells…).')
        doc = _doc(session, p, 'cells')
    h = doc.get('headline') or {}
    pool = doc.get('pooling') or {}
    dr = doc.get('drift') or {}
    out = {'study': 'cells', 'headline': {k: h.get(k) for k in ('value', 'status', 'word', 'text', 'advice')},
           'cells': [{k: c.get(k) for k in ('index', 'name', 'source', 'robots', 'cameras', 'reference')}
                     for c in doc.get('cells') or []],
           'gap': [{'cell': g['b'], 'index': _r(g['total']), 'band': g['band'],
                    'components': {k: _r(v, 3) for k, v in g['components'].items()},
                    'matches': [{k: _r(m[k], 3) if isinstance(m[k], float) else m[k]
                                 for k in ('a', 'b', 'role', 'viewpoint_deg', 'distance_m', 'hfov_diff_deg',
                                           'resolution_ratio', 'image_pct_a', 'image_pct_b')}
                                for m in g['matches']]} for g in (doc.get('gap') or {}).get('pairs') or []],
           'pooling': {'text': pool.get('text'),
                       'cells': [{'name': c['name'], 'verdict': c['verdict'],
                                  'steps': [f'{o["op"]} {o.get("feature", "")}' for o in c['operations']],
                                  'cannot_map': [u['why'] for u in c['unmappable']]}
                                 for c in pool.get('cells') or [] if not c.get('target')]},
           'drift': {k: dr.get(k) for k in ('worst', 'systematic', 'measured_cells')},
           'metrics': [{'id': r['id'], 'unit': r['unit'], 'values': [_r(v) for v in r['values']],
                        'best': r['best'], 'worst': r['worst']} for r in (doc.get('metrics') or {}).get('rows') or []],
           'note': 'The gap index is 0 for identical cells and 1 for unrelated ones; its bands are a heuristic, not a '
                   'predicted success rate.'}
    if full:
        out['result'] = F.compact({k: v for k, v in doc.items() if k not in ('cached', 'revision')}, items=12,
                                  max_depth=5)
    return out


_add(Tool('get_cells', 'Read the cell comparison',
          'The multi-cell comparison (File ▸ Compare cells…): the cells, the geometric gap of each to the reference '
          'per camera role (viewpoint angle, distance, FOV, resolution ratio, task volume in the image), whether '
          'their recordings pool for training with the mapping steps and what cannot be mapped, the drift of built '
          'copies from the plan, and the figures of merit side by side. Long work when the window holds no result.',
          _obj(dict(_FULL)), get_cells))


def get_cluster_job(session, folder='', **_):
    """A cluster job folder's state (core/cluster): which shards are complete, incomplete, failed or missing, and,
    when the folder holds its gathered result, that result's headline (the study's own figures)."""
    from ..core import cluster as C
    if not folder:
        raise ToolError('Name the job folder (the folder --slurm or the Run on… dialog wrote).')
    try:
        st = C.status(os.path.expanduser(str(folder)))
    except jobs.StudyError as e:
        raise ToolError(str(e)) from None
    out = {'title': st['title'], 'shards': st['shards'], 'items': st['items'], 'items_done': st['items_done'],
           'missing': st['missing'], 'incomplete': st['incomplete'], 'failed': st['failed'],
           'failures': [r['error'] for r in st['rows'] if r.get('error')][:5], 'peak_rss_mb': st['peak_rss_mb'],
           'complete': not st['redo'] and st['items_done'] == st['items']}
    rp = os.path.join(os.path.expanduser(str(folder)), 'result.json')
    if os.path.isfile(rp):
        try:
            with open(rp, encoding='utf-8') as fh:
                h = (json.load(fh).get('headline') or {})
            out['headline'] = {k: h.get(k) for k in ('value', 'unit', 'status', 'word', 'text')}
        except (OSError, ValueError):
            out['headline'] = None
    return out


_add(Tool('get_cluster_job', 'Read a cluster job folder',
          'A Slurm job folder written by --slurm or the Run on… dialog: its shards (complete, incomplete, failed, '
          'missing), the items done, the first failures, the peak memory of a shard, and the gathered result\'s '
          'headline when --gather has written it.',
          _obj({'folder': {'type': 'string', 'description': 'the job folder'}}, ('folder',)), get_cluster_job))


def get_statistics(session, rates=None, power=None, compare='reference', full=False, **_):
    """The Statistics page's figures (core/studies/stats.py): with rates and/or power given, the study run on them
    (success counts and the power calculator are quick); else the window's result for the design as it is now, else
    the study on the project's saved statistics settings when they need no long work."""
    p = session.project()
    if rates or power:
        st = {'source': 'rates', 'compare': compare}
        if rates:
            st['rates'] = rates
        if power:
            st['power'] = power
        p.analysis.studies['stats'] = st
        try:
            doc = jobs.run_study('stats', p, cache=session.cache)
        except jobs.StudyError as e:
            raise ToolError(str(e)) from None
    else:
        doc = session.last_result('stats')
        if doc is None:
            src = (p.analysis.studies.get('stats') or {}).get('source') or 'rates'
            if src in ('variants', 'experiments') and not session.allow_long:
                raise ToolError('This comparison runs the tolerance Monte Carlo or an experiment (long work): '
                                'propose_plan a study step "stats" for the user to approve, or ask the user to run it '
                                'on the Statistics page (Analysis ▸ Statistics). Success counts given here (rates) '
                                'are quick.')
            try:
                doc = jobs.run_study('stats', p, cache=session.cache)
            except jobs.StudyError as e:
                raise ToolError(str(e)) from None
    h = doc.get('headline') or {}
    out = {'study': 'stats', 'source': doc.get('source_label'),
           'headline': {k: h.get(k) for k in ('value', 'unit', 'status', 'word', 'text')},
           'pairing': doc.get('pairing'), 'correction': (doc.get('correction') or {}).get('label'),
           'comparisons': [{'title': c.get('title'), 'test': c.get('test_label') or c.get('test'),
                            'why': c.get('why'), 'p': _r(c.get('p')), 'p_adjusted': _r(c.get('p_adjusted')),
                            'difference': _r((c.get('difference') or {}).get('value')),
                            'difference_ci': [_r(v) for v in (c.get('difference') or {}).get('ci') or []],
                            'unit': 'share (0-1)' if c.get('kind') == 'rates' else c.get('unit'),
                            'verdict': (c.get('verdict') or {}).get('text'), 'error': c.get('error')}
                           for c in doc.get('comparisons') or []],
           'assumptions': doc.get('assumptions'), 'out_of_scope': doc.get('out_of_scope')}
    pw = doc.get('power')
    if pw:
        out['power'] = {k: pw.get(k) for k in ('kind', 'n_per_group', 'total', 'power_at_n', 'method', 'test',
                                               'answer', 'planned', 'assumptions', 'note')}
    if doc.get('advice'):
        out['advice'] = doc['advice']
    if full:
        out['result'] = F.compact({k: v for k, v in doc.items() if k not in ('cached', 'revision')}, items=12,
                                  max_depth=5)
    return out


_RATE_ROW = _obj({'condition': {'type': 'string'}, 'successes': {'type': 'integer', 'minimum': 0},
                  'trials': {'type': 'integer', 'minimum': 1}}, ('condition', 'successes', 'trials'))
_POWER = {'type': 'object', 'additionalProperties': True,
          'description': 'the power calculator: kind rates (p1, p2 as shares 0-1), t2 or paired_t (d), or '
                         'demonstrations (demo_kind continuous with d, or binary with p1, p2; icc, per_operator, '
                         'attrition); method exact (default for rates), normal, arcsine or simulation; n: the '
                         'episodes planned per condition (its power is given); alpha (default 0.05), power (0.8)'}
_add(Tool('get_statistics', 'Read the statistics',
          'Statistics and power analysis (Analysis ▸ Statistics): compares success counts of conditions (policies, '
          'rigs) with Fisher\'s exact test, the risk difference and odds ratio with intervals, corrected across the '
          'comparisons, and says how many evaluation episodes or demonstrations per condition tell two rates or '
          'effects apart. With rates or power given it computes them (quick); else it reads the page\'s result '
          '(variants over the tolerance Monte Carlo, experiment or tracker runs). Each result says which test and '
          'why, and the assumptions.',
          _obj({'rates': {'type': 'array', 'items': _RATE_ROW, 'minItems': 2,
                          'description': 'conditions with their successes and trials'},
                'compare': {'type': 'string', 'enum': ['reference', 'pairs'],
                            'description': 'each against the first (default) or every pair'},
                'power': _POWER, **_FULL}), get_statistics))


def get_runs(session, study=None, tag=None, since=None, until=None, limit=20, **_):
    """The experiment tracker's runs (core/tracking) of the session's project: newest first, each with its study,
    status, headline, tags, note and stated figures with units."""
    from ..core import tracking as TR
    p = session.project()
    st = TR.settings_of(p)
    try:
        store = TR.store_for(p)
    except TR.TrackingError as e:
        return {'tracking_on': st['enabled'], 'runs': [], 'note': str(e)}
    if not store.exists():
        return {'tracking_on': st['enabled'], 'store': store.name, 'runs': [],
                'note': 'No run is logged yet' + ('' if st['enabled'] else ': tracking is off (turn it on on the '
                                                                          'Runs page)') + '.'}
    rows = store.runs(study=study, tag=tag, since=since, until=until)
    rows.reverse()
    n = max(1, min(int(limit or 20), 200))
    out = [{'run_id': r['run_id'], 'started_utc': r.get('started_utc'), 'study': r.get('study'),
            'kind': r.get('kind'), 'status': r.get('status'), 'reason': r.get('reason') or None,
            'headline': (r.get('headline') or {}).get('text'), 'tags': r.get('tags') or [], 'note': r.get('note'),
            'variant': (r.get('variant') or {}).get('name'), 'project_sha256': (r.get('project') or {}).get('sha256'),
            'rerun_of': r.get('rerun_of'), 'reproduce': (r.get('reproduce') or {}).get('verdict'),
            'figures': {k: {'value': m.get('value'), 'unit': m.get('unit'), 'status': m.get('status')}
                        for k, m in (r.get('metrics') or {}).items()}} for r in rows[:n]]
    return {'tracking_on': st['enabled'], 'store': store.name, 'total': len(rows), 'runs': out,
            **({'note': f'{len(rows) - n} older runs not listed (limit {n}).'} if len(rows) > n else {})}


_add(Tool('get_runs', 'Read the logged runs',
          'The experiment tracker (Analysis ▸ Runs): the runs logged for this project, newest first: each run\'s '
          'study, status, headline, tags, note, the variant and project content hash it ran on, and its figures '
          'with units (None where not stated). Filter by study, tag and date (since/until, ISO dates).',
          _obj({'study': {'type': 'string'}, 'tag': {'type': 'string'},
                'since': {'type': 'string', 'description': 'ISO date or time, e.g. 2026-11-01'},
                'until': {'type': 'string', 'description': 'ISO date or time (a date: the whole day)'},
                'limit': {'type': 'integer', 'description': 'at most this many runs (default 20, at most 200)'}}),
          get_runs))


def get_rig_cost(session, **_):
    """The rig cost from the cited library prices (core/prices.py, core/studies/rig_inventory.py)."""
    from ..core import prices as PR
    from ..core.studies import rig_inventory as RI
    inv = RI.inventory(session.project())
    s = PR.summary()
    return {'cost': inv['cost'],
            'parts': [{k: d.get(k) for k in ('name', 'model_id', 'device', 'price', 'currency', 'price_kind',
                                             'price_source', 'price_read')} for d in inv['devices']],
            'library': {'entries': s['entries'], 'priced': s['priced'], 'currencies': s['currencies']},
            'note': 'List prices on the date read, each from the page its library entry cites (no tax, shipping or '
                    'discount, no exchange rate). A cost with bound at_least is a lower bound: the parts in '
                    "'missing' state no price. Only the enabled sensors are counted."}


_add(Tool('get_rig_cost', 'Read the rig cost',
          'The rig cost added up from the cited list prices of the active variant\'s sensors (amount, currency, the '
          'store page and the date read for each), a lower bound with the parts that state no price when some do '
          'not, and how many library entries state a price. Nothing is estimated or converted.', _NONE,
          get_rig_cost))


def check_robot_file(session, path='', **_):
    """What a robot file (URDF, MJCF or USD) names and whether each referenced file is found
    (core/robot/filecheck.py: the import wizard's check, run before an import). Reads the file only."""
    from ..core.robot import filecheck as FC
    from ..core.robot import importer as I
    if not path:
        raise ToolError('Name the robot file (a .urdf, .xml MJCF or .usd/.usda/.usdc/.usdz file).')
    p = os.path.expanduser(str(path))
    try:
        rep = FC.check_references(p)
    except I.RobotImportError as e:
        raise ToolError(str(e)) from None
    items = rep.get('items') or []
    miss = [i for i in items if not i.get('found')]
    return {'file': os.path.basename(p), 'kind': rep.get('kind'), 'references': len(items),
            'found': len(items) - len(miss),
            'missing_required': [{'what': i['what'], 'ref': i['ref'], 'message': i['message']}
                                 for i in miss if i.get('required')][:40],
            'missing_optional': [{'what': i['what'], 'ref': i['ref']} for i in miss if not i.get('required')][:40],
            'note': 'A missing required file stops the import (Robot ▸ Import robot…, or dv.robot_io.add_robot); a '
                    'missing optional one (a texture, an unresolved USD layer) is left out and said in the notes.'}


_add(Tool('check_robot_file', 'Check a robot file',
          'Before importing a robot: the kind of a URDF, MJCF or USD file and every file it references (meshes, '
          'includes, textures, USD layers), each found or missing, with the importer\'s sentence for a missing one. '
          'Reads the file only; nothing is imported.',
          _obj({'path': {'type': 'string', 'description': 'the robot file'}}, ('path',)), check_robot_file))


def _only_variant(project, v, uid: str):
    """project with v as its one variant, under uid (so two variants of one project diff item by item)."""
    q = project.snapshot()
    keep = q.variant_by_uid(v.uid)
    keep.uid = uid
    q.variants, q.active_uid = [keep], uid
    return q


def get_project_diff(session, path='', variant='', metrics=None, **_):
    """What changed (core/projdiff.py, File ▸ Compare with…): from another project file to this project, or from the
    active variant to another variant of it; with metrics, each one's value on both sides and the delta. Reads only."""
    from ..core import metrics as M
    from ..core import projdiff as D
    from ..core.project import Project, ProjectFormatError
    if bool(path) == bool(variant):
        raise ToolError('Name either another project file (path) or another variant of this project (variant), '
                        'not both.' if path else 'Name another project file (path) or another variant of this '
                        'project (variant) to compare with.')
    here = session.project()
    if path:
        p = os.path.expanduser(str(path))
        if not os.path.isabs(p) and here.path:
            p = os.path.join(os.path.dirname(here.path), p)
        name = os.path.basename(p.rstrip('/\\'))
        if not os.path.exists(p):
            raise ToolError(f'There is no project at {name}: give a .dexv file or an unpacked project folder.')
        try:
            a = Project.load(p)
        except ProjectFormatError as e:                  # a plain sentence already
            raise ToolError(str(e)) from None
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise ToolError(f'Could not read {name} as a DexVantage project ({type(e).__name__}).') from None
        b, how = here, f'from {name} to this project'
    else:
        other = (here.variant_by_uid(str(variant)) or
                 next((v for v in here.variants if v.name == str(variant)), None))
        if other is None:
            raise ToolError(f'This project has no variant {variant!r} (it has: '
                            f'{", ".join(v.name for v in here.variants)}).')
        if other.uid == here.variant.uid:
            raise ToolError(f'{other.name} is the active variant: name another one to compare with.')
        act = here.variant.uid
        a, b = _only_variant(here, here.variant, act), _only_variant(here, other, act)
        how = f'from the active variant {here.variant.name} to {other.name}'
    try:
        doc = D.diff(a, b)
    except jobs.StudyError as e:
        raise ToolError(str(e)) from None
    ids = [metrics] if isinstance(metrics, str) else [str(m) for m in metrics or []]
    if ids:
        M.load_metrics()
        unknown = [m for m in ids if m not in M.METRICS]
        if unknown:
            raise ToolError(f'There is no metric {", ".join(unknown)} (list_metrics lists them).')
        sids = sorted({M.METRICS[m].study for m in ids})
        ra = {sid: _doc(session, a, sid, use_last=False) for sid in sids}
        rb = {sid: _doc(session, b, sid, use_last=bool(path)) for sid in sids}     # the window's own result
        doc['metrics'] = D.metric_deltas(ra, rb, ids)
    n = len(doc['changes'])
    return {'compared': how, 'identical': doc['identical'], 'counts': doc['counts'],
            'changes': [{k: c.get(k) for k in ('section', 'op', 'kind', 'item', 'field', 'text')}
                        for c in doc['changes'][:80]],
            'metrics': [{k: r.get(k) for k in ('id', 'title', 'unit', 'a', 'b', 'delta', 'verdict')}
                        | {'delta_unit': M.delta_unit(r.get('unit'))} for r in doc['metrics']],
            **({'note': f'{n - 80} more changes not listed.'} if n > 80 else {})}


_add(Tool('get_project_diff', 'Compare with another project or variant',
          'What changed, item by item (robots, sensors and their mounts, variants, task volumes, requirements, '
          'settings, files), from another project file (path: a .dexv file or an unpacked folder; a relative path '
          'is read beside this project) to this project, or from the active variant to another variant of this '
          'project (variant: its name or uid). With metrics (metric ids), each figure on both sides and its delta '
          '(delta_unit: a change of a percentage is in percentage points, pp). Reads only; nothing is changed.',
          _obj({'path': {'type': 'string', 'description': 'another project file or folder'},
                'variant': {'type': 'string', 'description': 'another variant of this project (name or uid)'},
                'metrics': {'type': 'array', 'items': {'type': 'string'},
                            'description': 'metric ids to compare on both sides (list_metrics lists them)'}}),
          get_project_diff))

def input_schemas() -> dict:
    return {n: t.input_schema for n, t in TOOLS.items()}


# =========================================================================== the schema check
_JSON_TYPES = {'string': (str,), 'boolean': (bool,), 'object': (dict,), 'array': (list,)}


def _check(schema, v, path, out):
    """Problems of v against a JSON Schema (the subset the tools use: type, enum, properties, required,
    additionalProperties, items, minItems, maxItems). Ported from Vantage assistant/tools.py."""
    t = schema.get('type')
    where = path or 'input'
    if t in ('number', 'integer'):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float) and not math.isfinite(v)):
            out.append(f'{where}: a {t} is expected, not {json.dumps(v, default=str)[:40]}')
            return
        if t == 'integer' and float(v) != int(v):
            out.append(f'{where}: an integer is expected')
            return
    elif t in _JSON_TYPES and not isinstance(v, _JSON_TYPES[t]):
        out.append(f'{where}: a{"n" if t[0] in "aeiou" else ""} {t} is expected, not {type(v).__name__}')
        return
    if 'enum' in schema and v not in schema['enum']:
        out.append(f'{where}: {json.dumps(v, default=str)[:40]} is not one of {", ".join(map(str, schema["enum"]))}')
        return
    if t == 'object' and isinstance(v, dict):
        props = schema.get('properties') or {}
        for k in schema.get('required', ()):
            if k not in v:
                out.append(f'{where}: {k} is missing')
        for k, x in v.items():
            if k in props:
                _check(props[k], x, f'{path}.{k}' if path else k, out)
            elif schema.get('additionalProperties') is False:
                out.append(f'{where}: there is no parameter {k!r}')
    if t == 'array' and isinstance(v, list):
        if 'minItems' in schema and len(v) < schema['minItems']:
            out.append(f'{where}: at least {schema["minItems"]} item(s) are expected')
        if 'maxItems' in schema and len(v) > schema['maxItems']:
            out.append(f'{where}: at most {schema["maxItems"]} items are allowed')
        if isinstance(schema.get('items'), dict):
            for i, x in enumerate(v[:500]):
                _check(schema['items'], x, f'{where}[{i}]', out)


def validate_args(name: str, args) -> list:
    t = TOOLS.get(name)
    if t is None:
        return [f'there is no tool {name!r}']
    out: list = []
    _check(t.input_schema, args, '', out)
    return out


# =========================================================================== running one
def call(session, name: str, args=None) -> dict:
    """Run tool name on the session: its JSON result, or {'error': 'a plain sentence'} (never a traceback)."""
    t = TOOLS.get(name)
    if t is None:
        return {'error': f'There is no tool {name!r}.'}
    args = {k: v for k, v in (args or {}).items() if v is not None} if isinstance(args, dict) else args
    problems = validate_args(name, args if isinstance(args, dict) else args)
    if problems:
        return {'error': 'The input does not match the tool: ' + '; '.join(problems[:6]) + '.'}
    try:
        return t.fn(session, **args)
    except (ToolError, SessionError, NeedsWindow, jobs.StudyError) as e:
        return {'error': str(e)}
    except Exception as e:                               # noqa: BLE001 — a tool never takes the caller down
        traceback.print_exc(file=sys.stderr)
        return {'error': f'{t.title} stopped with an unexpected error ({type(e).__name__}). The details are in the '
                         'log.'}
