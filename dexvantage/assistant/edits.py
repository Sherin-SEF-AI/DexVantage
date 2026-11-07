"""Edit proposals: the assistant never changes the project itself. It proposes a change, the dock shows the diff
(each quantity before and after), and the user accepts it (one undo step, Edit ▸ Undo takes it back) or discards it
(a 'rejected' memory note is written so it is not proposed again).

A proposal comes from one of two places:
- propose_changes: named factors (core/factors.py paths: sensor:<id>.z, setting:<study>.<key>, ...) and their new
  values. The proposal is checked by applying it to a copy (a locked sensor, an unknown path or a bad value is a
  plain sentence back to the model), and applied again to the project as it is when the user accepts it.
- a changing tool of assistant/tools.py (Tool.changes): the tool runs on a copy of the project, and what it changed
  becomes the proposal, section by section. If the project changed after the proposal was made, accepting it is
  refused (ask again), so an edit made meanwhile is never overwritten.

apply() makes the change on a project in place, one project section at a time (Project.section / restore, the undo
stack's unit), and returns the section keys, so the dock wraps it in ProjectStore.edit(keys, ...) as one undo step.
No Qt here."""
from __future__ import annotations

import copy
import json
import uuid
from dataclasses import dataclass, field

from ..core import factors as F

SECTIONS = ('meta', 'robots', 'workspace', 'tasks', 'variants', 'requirements', 'analysis', 'rig', 'cables',
            'policies', 'calibration', 'data')
MAX_ROWS = 60


def _canon(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(',', ':'), default=str)


@dataclass
class Proposal:
    id: str
    title: str
    reason: str = ''
    values: dict = field(default_factory=dict)       # factor path -> value (propose_changes); {} for a tool's change
    ops: list = field(default_factory=list)          # structural operations (mcp/edits.py), applied first
    rows: list = field(default_factory=list)         # [{'factor'|'path', 'title', 'unit', 'before', 'after'}]
    keys: list = field(default_factory=list)         # project sections it changes
    source: str = 'propose_changes'                  # or 'tool:<name>'
    status: str = 'pending'                          # pending | applied | discarded | undone
    base_digest: str = ''                            # the project's content when it was made (a tool's change)
    after_sections: dict = field(default_factory=dict)   # a tool's change: section key -> its state after
    undo_index: int | None = None                    # the undo stack's index after applying (the dock's Undo)

    def to_dict(self) -> dict:
        return {'id': self.id, 'title': self.title, 'reason': self.reason, 'values': copy.deepcopy(self.values),
                'ops': copy.deepcopy(self.ops), 'rows': copy.deepcopy(self.rows), 'keys': list(self.keys),
                'source': self.source,
                'status': self.status}

    def for_model(self) -> dict:
        return {'proposal': self.id, 'title': self.title, 'status': self.status, 'changes': self.rows[:MAX_ROWS],
                'note': 'Shown to the user with Accept and Discard; nothing is changed until they accept it.'}


def new_id() -> str:
    return 'E' + uuid.uuid4().hex[:7].upper()


def digest(project) -> str:
    return _canon({k: project.section(k) for k in SECTIONS})


def section_keys(before, after) -> list:
    """The section keys whose content differs (a variant edited in place is its variant:<uid> key; adding, removing
    or switching variants is 'variants')."""
    keys = []
    for k in SECTIONS:
        if k == 'variants':
            same_list = [v.uid for v in before.variants] == [v.uid for v in after.variants] and \
                before.active_uid == after.active_uid
            if not same_list:
                keys.append('variants')
                continue
            for v in after.variants:
                if _canon(before.section(f'variant:{v.uid}')) != _canon(after.section(f'variant:{v.uid}')):
                    keys.append(f'variant:{v.uid}')
            continue
        if _canon(before.section(k)) != _canon(after.section(k)):
            keys.append(k)
    return keys


def _fmt_value(v):
    if isinstance(v, float):
        return round(v, 6)
    return v


def factor_rows(before, after, values: dict) -> list:
    rows = []
    for path in values:
        d = F.describe(path, before)
        try:
            b = F.get(before, path)
        except F.FactorError:
            b = None
        relative = path.startswith('sensor:') and path.rpartition('.')[2] in F.SHIFTS + F.TURNS
        a = values[path] if relative else F.get(after, path)        # a shift or turn reads 0: show the amount
        rows.append({'factor': path, 'title': d.get('title', path), 'unit': d.get('unit', ''),
                     'before': _fmt_value(b), 'after': _fmt_value(a)})
    return rows


def _flat(o, prefix='') -> dict:
    out = {}
    if isinstance(o, dict) and o:
        for k, v in o.items():
            out.update(_flat(v, f'{prefix}/{k}'))
    elif isinstance(o, list) and o and all(isinstance(x, (dict, list)) for x in o):
        for i, v in enumerate(o):
            out.update(_flat(v, f'{prefix}/{i}'))
    else:
        out[prefix or '/'] = o
    return out


def section_rows(before, after, keys) -> list:
    rows = []
    for k in keys:
        fa, fb = _flat(before.section(k) if k != 'variants' else before.section('variants')), \
            _flat(after.section(k) if k != 'variants' else after.section('variants'))
        for p in sorted(set(fa) | set(fb)):
            if _canon(fa.get(p)) != _canon(fb.get(p)):
                rows.append({'path': f'{k}{p}', 'title': f'{k}{p}', 'unit': '', 'before': fa.get(p),
                             'after': fb.get(p)})
                if len(rows) >= MAX_ROWS:
                    return rows
    return rows


def _engine(project, values: dict, ops):
    """The project after the change and the lines that say it: mcp/edits.py (factor paths, structural ops, every
    locked item checked) when it is there, else core/factors.py."""
    try:
        from ..mcp import edits as ME
    except ImportError:
        ME = None
    if ME is not None:
        try:
            ed = ME.apply(project, values or None, ops or None)
        except ME.EditError as e:
            raise F.FactorError(str(e)) from None
        return ed.project, list(ed.lines)
    if ops:
        raise F.FactorError('Structural changes (ops) are not available in this build: change factors only.')
    return F.apply(project, values), []


def from_factors(project, title: str, changes, reason: str = '', ops=None) -> Proposal:
    """A proposal from [{'factor', 'value'}] (or {factor: value}) and optional structural ops; raises
    factors.FactorError in plain words."""
    if isinstance(changes, dict):
        values = dict(changes)
    else:
        values = {}
        for c in changes or []:
            if not isinstance(c, dict) or not c.get('factor'):
                raise F.FactorError('Each change needs a factor path and a value.')
            values[str(c['factor'])] = c.get('value')
    ops = [o for o in ops or [] if isinstance(o, dict)]
    if not values and not ops:
        raise F.FactorError('The proposal changes nothing: give at least one factor and its new value.')
    after, lines = _engine(project, values, ops)
    keys = section_keys(project, after)
    if not keys:
        raise F.FactorError('Those values are what the project has already: nothing would change.')
    rows = factor_rows(project, after, values) if not ops else []
    if ops:                                     # sensors added or removed: the engine's own lines say it
        rows = [{'path': f'change {i + 1}', 'title': ln, 'unit': '', 'before': None, 'after': None}
                for i, ln in enumerate(lines)]
    return Proposal(new_id(), (title or 'Proposed change').strip()[:120], (reason or '').strip()[:600], values=values,
                    rows=rows, keys=keys, ops=ops)


def from_projects(before, after, title: str, source: str, reason: str = '') -> Proposal | None:
    """A proposal from what a tool changed on a copy of the project (None when it changed nothing). before is a
    snapshot (Project.snapshot) of the project as it was, so its digest compares with a later snapshot's."""
    keys = section_keys(before, after)
    if not keys:
        return None
    return Proposal(new_id(), (title or 'Proposed change').strip()[:120], reason, values={},
                    rows=section_rows(before, after, keys), keys=keys, source=source, base_digest=digest(before),
                    after_sections={k: copy.deepcopy(after.section(k)) for k in keys})


def apply(project, prop: Proposal) -> list:
    """Make the change on project in place (call it inside ProjectStore.edit). Returns the section keys changed;
    raises factors.FactorError when it cannot be applied any more."""
    if prop.values or prop.ops:
        after, _ = _engine(project, prop.values, prop.ops)
        keys = section_keys(project, after)
        for k in keys:
            project.restore(k, copy.deepcopy(after.section(k)))
        return keys
    if prop.base_digest and digest(project.snapshot()) != prop.base_digest:
        raise F.FactorError('The project changed after this proposal was made: ask the assistant again.')
    for k in prop.keys:
        project.restore(k, copy.deepcopy(prop.after_sections[k]))
    return list(prop.keys)


def keys_for(project, prop: Proposal) -> list:
    """The section keys an apply() would touch now (what ProjectStore.edit snapshots for undo)."""
    if prop.values or prop.ops:
        return section_keys(project, _engine(project, prop.values, prop.ops)[0])
    return list(prop.keys)
