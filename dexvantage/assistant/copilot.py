"""The Copilot: plans for long runs, approved by the user before anything runs.

The user says what they want ("check grasp visibility on every grasp, then export the LeRobot config"); the model
answers through tools only, and its one planning tool, propose_plan, hands the dock a Plan: one or more steps, each a
study (a registered core.jobs study with the settings it runs with) or an export (formats of core/exporters). Every
value says where it came from (the user's request, chosen by the Copilot, the page's default) and why; every step is
checked (the study exists, it has the robot it needs, the export formats exist) and carries its warnings and, where
this session has measured one, the time the study took when it last ran (never an invented estimate).

Nothing runs until the user approves. The plan card offers Approve, Edit (change one value: a new version, the diff
against the previous one shown, approval needed again) and Cancel. Approving writes an approval record (the plan, the
request, the provider and model, the resolved settings and their SHA-256, the time) into the project's Copilot file
(store.py: <data>/assistant/<project>/copilot.json). The dock then runs each step; before each one gate() refuses
unless an approved, intact record holds exactly that step's kind and settings. An approved plan is never changed: an
edit makes a new version. The model itself can never start a long study: the loop refuses run_study for any study
that is not quick (is_long) and tells the model to propose a plan instead.

Known limits: the record proves what was approved, not who approved it (anyone who can write the user's data folder
can write a record; verify() catches a record changed after approval, not a forged one). Run times are shown only when
this session has run the study before; there is no model of how long a study takes.

Ported from Vantage assistant/copilot.py: the same Plan, Step, record, verify and gate; the CARLA, campaign and
dataset kinds are replaced by DexVantage's studies and exports, and the records live beside the chats instead of in
the project file. No Qt here."""
from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import math
import re
import uuid
from dataclasses import dataclass, field

from . import store as ST

PLAN_SCHEMA = 'dexvantage.copilot_plan/1'
RECORD_SCHEMA = 'dexvantage.copilot_record/1'
MAX_RECORDS = 60
KINDS = ('study', 'export', 'edit')
KIND_LABEL = {'study': 'Study', 'export': 'Export', 'edit': 'Edit'}
SOURCES = {'request': 'from your request', 'default': 'default', 'copilot': 'chosen by the Copilot'}
STATUSES = ('draft', 'approved', 'running', 'done', 'failed', 'cancelled', 'rejected', 'superseded')

_PLAN_WORDS = re.compile(r'(^\s*/plan\b)|\b(run|runs|study|studies|sweep|export|exports|batch|simulate|render|audit|'
                         r'ground truth|experiment|optimi[sz]e)\b', re.IGNORECASE)


def is_plan_request(text: str) -> bool:
    """Does the message ask for something to be run or written (a Copilot turn: the model thinks harder)?"""
    return bool(_PLAN_WORDS.search(text or ''))


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).strftime('%Y-%m-%dT%H:%M:%SZ')


def canonical(o) -> bytes:
    return json.dumps(o, sort_keys=True, separators=(',', ':'), ensure_ascii=False, default=str).encode('utf-8')


def sha256(o) -> str:
    return hashlib.sha256(canonical(o)).hexdigest()


def same(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) \
            and not isinstance(b, bool):
        return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)
    return canonical(a) == canonical(b)


def parse_value(v):
    """A value the model wrote as JSON text ("0.5", "true", "[...]"); a bare word is taken as a string."""
    if not isinstance(v, str):
        return v
    t = v.strip()
    try:
        return json.loads(t)
    except ValueError:
        return t


# =========================================================================== studies
def studies() -> dict:
    from ..core import jobs
    jobs.load_studies()
    return jobs.STUDIES


def is_long(sid: str) -> bool:
    """Long work (tens of seconds or more on the samples, assistant/tools.py LONG, measured): only a plan the user
    approved runs it from the assistant. Without tools.py, every study that is not quick."""
    try:
        from .tools import LONG
    except ImportError:
        info = studies().get(sid)
        return info is not None and not info.quick
    return sid in LONG


def exporters() -> dict:
    from ..core import exporters as EX
    EX.load_exporters()
    return EX.EXPORTERS


def catalogue(project=None) -> dict:
    """What a plan can contain (the copilot_catalogue tool)."""
    st = [{'study': i.id, 'title': i.title, 'long': not i.quick, 'needs_robot': i.needs_robot,
           'settings_now': copy.deepcopy((project.analysis.studies.get(i.id) or {}) if project is not None else {})}
          for i in sorted(studies().values(), key=lambda x: x.id)]
    for row in st:
        row['long'] = is_long(row['study'])
    ex = [{'format': e.id, 'title': e.title, 'needs_robot': e.needs_robot} for e in
          sorted(exporters().values(), key=lambda x: x.id)]
    return {'kinds': {'study': 'Run one study with the settings given (merged into the page\'s stored settings for '
                               'this run only).',
                      'export': 'Write export formats into a new folder of the working folder.',
                      'edit': 'Change the project (changes: {factor path: value}, ops: structural operations, as '
                              'edit_project takes them) as one undo step when the step runs.'},
            'studies': st, 'export_formats': ex,
            'rule': 'Long studies (long: true) run only from a plan the user approves.'}


# =========================================================================== steps and plans
@dataclass
class Step:
    kind: str
    label: str = ''
    settings: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)      # key -> {'source', 'reason'}
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    estimate: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {'kind': self.kind, 'label': self.label, 'settings': copy.deepcopy(self.settings),
                'sources': copy.deepcopy(self.sources), 'errors': list(self.errors), 'warnings': list(self.warnings),
                'notes': list(self.notes), 'estimate': copy.deepcopy(self.estimate),
                'settings_sha256': step_sha256(self.kind, self.settings)}

    @staticmethod
    def from_dict(d) -> 'Step':
        d = d if isinstance(d, dict) else {}
        return Step(str(d.get('kind') or ''), str(d.get('label') or ''), copy.deepcopy(d.get('settings') or {}),
                    copy.deepcopy(d.get('sources') or {}), list(d.get('errors') or []), list(d.get('warnings') or []),
                    list(d.get('notes') or []), copy.deepcopy(d.get('estimate') or {}))


def step_sha256(kind, settings) -> str:
    return sha256({'kind': kind, 'settings': settings})


def plan_sha256(steps) -> str:
    return sha256([{'kind': s.kind if isinstance(s, Step) else s['kind'],
                    'settings': s.settings if isinstance(s, Step) else s['settings']} for s in steps])


@dataclass
class Plan:
    id: str
    version: int = 1
    title: str = ''
    rationale: str = ''
    request: list = field(default_factory=list)     # the user's words that led to it, in order
    steps: list = field(default_factory=list)       # Step
    status: str = 'draft'
    provider: str = ''
    model: str = ''
    created_utc: str = ''
    parent: int | None = None                       # the version it revises
    diff: list = field(default_factory=list)        # against the parent

    @property
    def ok(self) -> bool:
        return bool(self.steps) and all(s.ok for s in self.steps)

    @property
    def errors(self) -> list:
        return [f'Step {i + 1}: {e}' for i, s in enumerate(self.steps) for e in s.errors] + \
            ([] if self.steps else ['The plan has no steps.'])

    def settings_sha256(self) -> str:
        return plan_sha256(self.steps)

    def to_dict(self) -> dict:
        return {'schema': PLAN_SCHEMA, 'id': self.id, 'version': self.version, 'title': self.title,
                'rationale': self.rationale, 'request': list(self.request), 'status': self.status,
                'provider': self.provider, 'model': self.model, 'created_utc': self.created_utc,
                'parent': self.parent, 'diff': copy.deepcopy(self.diff),
                'steps': [s.to_dict() for s in self.steps], 'settings_sha256': self.settings_sha256()}

    @staticmethod
    def from_dict(d) -> 'Plan':
        d = d if isinstance(d, dict) else {}
        return Plan(str(d.get('id') or new_id()), int(d.get('version') or 1), str(d.get('title') or ''),
                    str(d.get('rationale') or ''), list(d.get('request') or []),
                    [Step.from_dict(s) for s in d.get('steps') or []], str(d.get('status') or 'draft'),
                    str(d.get('provider') or ''), str(d.get('model') or ''), str(d.get('created_utc') or ''),
                    d.get('parent'), copy.deepcopy(d.get('diff') or []))


def new_id() -> str:
    return 'P' + uuid.uuid4().hex[:7].upper()


def _values(raw) -> list:
    """[{'key', 'value', 'source', 'reason'}] from the model's values (a list of those, or a plain dict)."""
    if isinstance(raw, dict):
        return [{'key': k, 'value': v} for k, v in raw.items()]
    return [v for v in raw or [] if isinstance(v, dict) and v.get('key')]


def resolve_step(project, raw: dict, base: Step | None = None, last_seconds: dict | None = None) -> Step:
    """One step from propose_plan's input, checked. base: the previous version's step (a revision only says what
    changes)."""
    raw = raw if isinstance(raw, dict) else {}
    kind = str(raw.get('kind') or (base.kind if base else ''))
    st = Step(kind, str(raw.get('label') or (base.label if base else ''))[:120],
              copy.deepcopy(base.settings) if base and base.kind == kind else {},
              copy.deepcopy(base.sources) if base and base.kind == kind else {})
    if kind not in KINDS:
        st.errors.append(f'"{kind}" is not a kind of step (study, export or edit).')
        return st
    if kind == 'study':
        sid = str(raw.get('study') or st.settings.get('study') or '')
        st.settings['study'] = sid
        st.settings.setdefault('settings', {})
        if 'study' in raw:
            st.sources['study'] = {'source': raw.get('source') or 'request', 'reason': ''}
        info = studies().get(sid)
        if info is None:
            st.errors.append(f'DexVantage has no study "{sid}" (copilot_catalogue lists them).')
            return st
        st.label = st.label or info.title
        vals = _values(raw.get('values'))
        if isinstance(raw.get('settings'), dict):
            vals += [{'key': k, 'value': v, 'source': raw.get('source') or 'copilot'}
                     for k, v in raw['settings'].items()]
        for v in vals:
            key = str(v['key']).removeprefix('settings.')
            val = parse_value(v.get('value'))
            try:
                json.dumps(val)
            except (TypeError, ValueError):
                st.errors.append(f'The value of {key} is not plain JSON.')
                continue
            st.settings['settings'][key] = val
            st.sources[key] = {'source': v.get('source') if v.get('source') in SOURCES else 'copilot',
                               'reason': str(v.get('reason') or '')[:200]}
        check_step(project, st, last_seconds)
    elif kind == 'edit':
        for k in ('changes', 'ops'):
            if k in raw:
                st.settings[k] = copy.deepcopy(raw[k])
                st.sources[k] = {'source': raw.get('source') or 'copilot', 'reason': ''}
        st.settings.setdefault('changes', {})
        st.settings.setdefault('ops', [])
        check_step(project, st, last_seconds)
        st.label = st.label or (st.notes[0] if st.notes else 'Edit')
    else:
        fmts = raw.get('formats', st.settings.get('formats'))
        fmts = [str(f) for f in (fmts if isinstance(fmts, list) else [fmts] if fmts else [])]
        st.settings['formats'] = fmts
        opts = raw.get('options', st.settings.get('options') or {})
        st.settings['options'] = opts if isinstance(opts, dict) else {}
        st.label = st.label or 'Export ' + ', '.join(fmts)
        if 'formats' in raw:
            st.sources['formats'] = {'source': raw.get('source') or 'request', 'reason': ''}
        check_step(project, st, last_seconds)
    return st


def check_step(project, st: Step, last_seconds: dict | None = None):
    """The step's errors, warnings and estimate, from the project as it is now."""
    st.errors, st.warnings, st.notes, st.estimate = [], [], [], {}
    if st.kind == 'study':
        sid = st.settings.get('study', '')
        info = studies().get(sid)
        if info is None:
            st.errors.append(f'DexVantage has no study "{sid}" (copilot_catalogue lists them).')
            return
        if info.needs_robot and not project.robots:
            st.errors.append(f'{info.title} needs a robot: import one first or open a sample.')
        stored = project.analysis.studies.get(sid) or {}
        for k in st.settings.get('settings') or {}:
            if k not in stored:
                st.warnings.append(f'The page stores no value for "{k}" now: the study uses it only if it reads '
                                   'that setting.')
        st.estimate = {'long': not info.quick}
        sec = (last_seconds or {}).get(sid)
        if isinstance(sec, (int, float)) and math.isfinite(sec):
            st.estimate.update(last_run_s=round(float(sec), 2),
                               source='the time this study took when it last ran in this session (a measurement '
                                      'with other settings, not a prediction)')
        else:
            st.estimate['note'] = 'Not run in this session yet: no time is estimated.'
    elif st.kind == 'export':
        fmts = st.settings.get('formats') or []
        known = exporters()
        if not fmts:
            st.errors.append('An export step needs at least one format (copilot_catalogue lists them).')
        for f in fmts:
            e = known.get(f)
            if e is None:
                st.errors.append(f'DexVantage has no export format "{f}".')
            elif e.needs_robot and not project.robots:
                st.errors.append(f'{e.title} needs a robot: import one first or open a sample.')
        st.notes.append('The files go into a new folder of the working folder, named after the plan.')
    elif st.kind == 'edit':
        try:
            lines = edit_lines(project, st.settings)
        except Exception as e:              # noqa: BLE001 — EditError / FactorError: a plain sentence
            st.errors.append(str(e))
        else:
            st.notes += lines[:12]
            st.notes.append('Made as one undo step when the step runs (Edit ▸ Undo takes it back).')
    else:
        st.errors.append(f'"{st.kind}" is not a kind of step (study, export or edit).')


def edit_project_after(project, settings: dict):
    """The project after an edit step (a new project; mcp/edits.py) and its lines."""
    from ..mcp import edits as ME
    ed = ME.apply(project, settings.get('changes') or None, settings.get('ops') or None)
    if not ed.keys:
        raise ME.EditError('This edit changes nothing: the project already has those values.')
    return ed


def edit_lines(project, settings: dict) -> list:
    return list(edit_project_after(project, settings).lines)


def build(project, raw_steps, title='', rationale='', prev: Plan | None = None, request='', provider='', model='',
          revise=False, last_seconds: dict | None = None) -> Plan:
    """A plan from propose_plan's steps. revise (a follow-up): each step starts from the previous version's step at the
    same place, so the values only say what changes; the new version keeps the id."""
    steps = []
    for i, rs in enumerate(raw_steps or []):
        base = prev.steps[i] if (revise and prev is not None and i < len(prev.steps)) else None
        steps.append(resolve_step(project, rs, base, last_seconds))
    if prev is not None and revise:
        pid, ver = prev.id, prev.version + 1
        req = list(prev.request) + ([request] if request and request not in prev.request else [])
    else:
        pid, ver, req = new_id(), 1, ([request] if request else [])
    p = Plan(pid, ver, (title or (prev.title if prev is not None and revise else '') or 'Plan').strip()[:120],
             (rationale or (prev.rationale if prev is not None and revise else '')).strip()[:600], req, steps,
             'draft', provider, model, utc_now(), prev.version if prev is not None and pid == prev.id else None)
    if prev is not None and pid == prev.id:
        p.diff = diff(prev, p)
    return p


def _flat(d, prefix='') -> dict:
    out = {}
    for k, v in (d or {}).items():
        key = f'{prefix}{k}'
        if isinstance(v, dict) and v:
            out.update(_flat(v, key + '.'))
        else:
            out[key] = v
    return out


def diff(a: Plan, b: Plan) -> list:
    """What changed from version a to version b: [{'step', 'key', 'before', 'after'}], steps added or removed as
    {'step', 'added' | 'removed': kind}."""
    out = []
    for i in range(max(len(a.steps), len(b.steps))):
        sa = a.steps[i] if i < len(a.steps) else None
        sb = b.steps[i] if i < len(b.steps) else None
        if sa is None:
            out.append({'step': i, 'added': sb.kind, 'label': sb.label})
        elif sb is None:
            out.append({'step': i, 'removed': sa.kind, 'label': sa.label})
        elif sa.kind != sb.kind:
            out += [{'step': i, 'removed': sa.kind, 'label': sa.label}, {'step': i, 'added': sb.kind,
                                                                          'label': sb.label}]
        else:
            fa, fb = _flat(sa.settings), _flat(sb.settings)
            for k in sorted(set(fa) | set(fb)):
                if k not in fa or k not in fb or not same(fa[k], fb[k]):
                    out.append({'step': i, 'key': k, 'before': fa.get(k), 'after': fb.get(k)})
    return out


def edit_value(project, plan: Plan, step: int, key: str, value, last_seconds=None) -> Plan:
    """The user changed one value on the plan card: a new version (an approved plan is never changed), the value from
    the user (source request), checked again."""
    if not 0 <= step < len(plan.steps):
        raise IndexError(step)
    s = plan.steps[step]
    if s.kind == 'export' and key in ('formats', 'options'):
        raw = {key: value}
    else:
        raw = {'values': [{'key': key, 'value': value, 'source': 'request', 'reason': 'you changed it on the plan'}]}
    steps = [Step.from_dict(x.to_dict()) for x in plan.steps]
    steps[step] = resolve_step(project, raw, s, last_seconds)
    for i, x in enumerate(steps):
        if i != step:
            check_step(project, x, last_seconds)
    p = Plan(plan.id, plan.version + 1, plan.title, plan.rationale, list(plan.request), steps, 'draft', plan.provider,
             plan.model, utc_now(), plan.version)
    p.diff = diff(plan, p)
    return p


def revalidate(project, plan: Plan, last_seconds=None) -> Plan:
    out = Plan.from_dict(plan.to_dict())
    for s in out.steps:
        check_step(project, s, last_seconds)
    return out


# =========================================================================== records and the gate
def records_path(project) -> str:
    return ST.project_dir(project, 'copilot.json')


def records(project) -> list:
    doc = ST.read_json(records_path(project), {}) or {}
    rs = doc.get('plans') if isinstance(doc, dict) else None
    return [r for r in rs if isinstance(r, dict)] if isinstance(rs, list) else []


def save_records(project, rs: list):
    ST.write_json(records_path(project), {'schema': 'dexvantage.copilot_records/1',
                                          'project': getattr(project, 'name', ''), 'plans': rs[-MAX_RECORDS:]})


def find(rs: list, plan_id, version=None) -> dict | None:
    hit = [r for r in rs if r.get('id') == plan_id and (version is None or int(r.get('version') or 0) == int(version))]
    return hit[-1] if hit else None


def verify(rec: dict) -> bool:
    """The record still holds what was approved: its steps hash to the stored settings hash."""
    try:
        return rec.get('schema') == RECORD_SCHEMA and rec.get('status') in ('approved', 'running', 'done', 'failed',
                                                                            'cancelled') and \
            bool(rec.get('approved_utc')) and plan_sha256(rec['steps']) == rec.get('settings_sha256') and \
            all(step_sha256(s['kind'], s['settings']) == s.get('settings_sha256') for s in rec['steps'])
    except (KeyError, TypeError):
        return False


def record(plan: Plan, provider='', model='') -> dict:
    d = plan.to_dict()
    return {'schema': RECORD_SCHEMA, 'id': plan.id, 'version': plan.version, 'title': plan.title,
            'rationale': plan.rationale, 'request': list(plan.request), 'provider': provider or plan.provider,
            'model': model or plan.model,
            'steps': [{k: s[k] for k in ('kind', 'label', 'settings', 'sources', 'warnings', 'notes', 'estimate',
                                         'settings_sha256')} for s in d['steps']],
            'settings_sha256': d['settings_sha256'], 'approved_utc': utc_now(), 'status': 'approved', 'runs': []}


def approve(project, plan: Plan, provider='', model='', last_seconds=None) -> tuple[dict | None, list]:
    """(the record, []) when the plan can be approved: it is checked again now; else (None, the errors)."""
    if plan.status != 'draft':
        return None, [f'This plan is {plan.status}: an approved plan cannot be approved again (edit it for a new '
                      'version).']
    again = revalidate(project, plan, last_seconds)
    if not again.ok:
        return None, again.errors
    return record(again, provider, model), []


def add_record(project, rec: dict) -> dict:
    """Store an approval record (newest last; earlier approved versions of the same plan never run are superseded)."""
    rs = records(project)
    for r in rs:
        if r.get('id') == rec['id'] and r.get('status') == 'approved' and not r.get('runs'):
            r['status'] = 'superseded'
    rs.append(copy.deepcopy(rec))
    save_records(project, rs)
    return rs[-1]


def update_record(project, rec: dict):
    rs = records(project)
    for i, r in enumerate(rs):
        if r.get('id') == rec['id'] and int(r.get('version') or 0) == int(rec['version']):
            rs[i] = copy.deepcopy(rec)
            break
    else:
        rs.append(copy.deepcopy(rec))
    save_records(project, rs)


def ticket(rec: dict, step: int) -> dict:
    return {'plan': rec['id'], 'version': int(rec['version']), 'step': int(step)}


def gate(rs: list, tk, kind: str, settings: dict) -> tuple[bool, str]:
    """May this step run? Only when an approved, intact record of that plan version holds a step of this kind with
    exactly these settings. (ok, why not)."""
    if not isinstance(tk, dict) or not tk.get('plan'):
        return False, 'No approved plan: a long study runs only from a plan you approve.'
    rec = find(rs, tk.get('plan'), tk.get('version'))
    if rec is None:
        return False, f'No approval record of plan {tk.get("plan")} v{tk.get("version")} for this project.'
    if not verify(rec):
        return False, (f'The approval record of plan {rec.get("id")} v{rec.get("version")} does not verify (it was '
                       'changed after approval or never approved).')
    i = int(tk.get('step', -1))
    if not 0 <= i < len(rec['steps']):
        return False, f'Plan {rec["id"]} has no step {i + 1}.'
    s = rec['steps'][i]
    if s['kind'] != kind:
        return False, f'Step {i + 1} of plan {rec["id"]} is a {KIND_LABEL.get(s["kind"], s["kind"]).lower()} step.'
    if not same(s['settings'], settings):
        fa, fb = _flat(s['settings']), _flat(settings)
        keys = [k for k in sorted(set(fa) | set(fb)) if not same(fa.get(k), fb.get(k))]
        return False, (f'The settings differ from the approved plan ({", ".join(keys[:4])}): approve the changed plan '
                       'first.')
    return True, ''


def start_run(rec: dict, via='app') -> dict:
    run = {'started_utc': utc_now(), 'finished_utc': None, 'via': via, 'status': 'running',
           'steps': [{'status': 'pending', 'message': ''} for _ in rec['steps']]}
    rec.setdefault('runs', []).append(run)
    rec['status'] = 'running'
    return run


def finish_run(rec: dict, status: str):
    run = (rec.get('runs') or [None])[-1]
    if run is not None:
        run['status'] = status
        run['finished_utc'] = utc_now()
    rec['status'] = status


def run_step(project, rs: list, tk: dict, ctx=None, cache=None, out_dir: str | None = None) -> dict:
    """Run one approved step headless (the tests, batch use): gate first, then the study (its settings merged into a
    snapshot's stored settings for this run) or the export. Returns a short summary; raises core.jobs.StudyError."""
    from ..core import jobs
    rec = find(rs, tk.get('plan'), tk.get('version'))
    if rec is None or not 0 <= int(tk.get('step', -1)) < len(rec.get('steps') or []):
        raise jobs.StudyError(gate(rs, tk, 'study', {})[1])
    s = rec['steps'][int(tk['step'])]
    ok, why = gate(rs, tk, s['kind'], s['settings'])
    if not ok:
        raise jobs.StudyError(why)
    if s['kind'] == 'edit':
        ed = edit_project_after(project, s['settings'])
        return {'kind': 'edit', 'lines': ed.lines, 'keys': ed.keys, 'project': ed.project}
    if s['kind'] == 'study':
        snap = project.snapshot()
        sid = s['settings']['study']
        snap.analysis.studies[sid] = {**(snap.analysis.studies.get(sid) or {}), **(s['settings'].get('settings') or {})}
        doc = jobs.run_study(sid, snap, ctx, cache)
        return {'kind': 'study', 'study': sid, 'seconds': doc.get('seconds'), 'doc': doc}
    from ..core import exporters as EX
    if not out_dir:
        raise jobs.StudyError('An export step needs a folder to write into.')
    man = EX.export(project, out_dir, s['settings']['formats'], s['settings'].get('options') or {}, ctx)
    return {'kind': 'export', 'formats': {k: v.get('status') for k, v in man['formats'].items()},
            'files': len(man['files'])}


def for_model(plan: Plan) -> dict:
    """The plan as the model is told it."""
    return {'plan': plan.id, 'version': plan.version, 'status': plan.status, 'title': plan.title, 'ok': plan.ok,
            'steps': [{'step': i + 1, 'kind': s.kind, 'label': s.label, 'settings': s.settings, 'ok': s.ok,
                       'errors': s.errors, 'warnings': s.warnings, 'notes': s.notes, 'estimate': s.estimate}
                      for i, s in enumerate(plan.steps)],
            'changed_from_previous_version': plan.diff[:40],
            'note': 'Shown to the user as a plan card with Approve, Edit and Cancel. Nothing runs until they approve.'}
