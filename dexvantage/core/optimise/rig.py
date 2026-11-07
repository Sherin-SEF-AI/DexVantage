"""The optimiser (spec §7.11): NSGA-II over the current rig's factors (core/factors) with the objectives and the
requirements of core/metrics, locked sensors never changed.

    settings_of(project) -> dict                    project.analysis.studies['optimise'] with the labelled defaults
    default_factors(project, st) -> [Factor]        what the search varies when the settings name no factors
    optimise(project, st=None, ctx=None) -> dict    the JSON-able result (also the 'optimise' study)
    accept(project, doc, label) -> list             set the option's factors on the active variant (inside one
                                                    store.edit('variant', ...) for one undo step)
    option_project(project, doc, label) -> Project  a copy with the option applied (Preview)

What is varied (the defaults, each labelled; settings 'factors' replaces them with your own list of
{'path', 'lo', 'hi'} or {'path', 'choices'}): for every enabled, unlocked camera, depth or tactile sensor of the active
variant a shift of its mount along the parent frame's x, y and z within ±move_m (default 0.05 m) and a turn about the
sensor's own y and z axes (pitch and yaw of the view) within ±turn_deg (default 15°); with allow_remove (default on)
whether it is enabled, so the camera count can fall; with swap_models (default off) its device, among the library
entries of the same kind. The current rig is the search's first design (every shift 0, every device as it is).

Objectives (settings 'objectives', default every one of core/optimise/objectives.DEFAULT_OBJECTIVES): a metric the
studies do not state for the current rig is left out, with the study's reason (a rig with no hosts has no bandwidth
margin; one without a task object no VLA object tokens); the cost is left out unless every device the search can
choose has a stated price. Constraints: every requirement of the project (and settings 'constraints'), as
constrained domination: a design that misses a requirement ranks behind every design that meets them all.

Ported from Vantage core/optimiser.py (the shape: the current layout as the first candidate, the Pareto front of the
feasible candidates, else of those closest to feasible, the changes said per camera, candidates labelled, locked
cameras kept) with its single-objective evolution replaced by NSGA-II and its automotive objectives (blind ground,
pixel density) by DexVantage's metrics.

Known limits: as core/optimise/__init__.py.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass

import numpy as np

from .. import factors as FA
from .. import jobs
from . import nsga2 as N
from . import objectives as OB
from . import search as S

SID = 'optimise'
DEFAULTS = dict(S.COMMON, move_m=0.05, turn_deg=15.0, allow_remove=True, swap_models=False, factors=[])
SENSOR_KINDS = ('camera', 'depth', 'tactile')


@dataclass
class Factor:
    path: str
    title: str
    unit: str
    lo: float = 0.0
    hi: float = 0.0
    choices: list | None = None
    nominal: object = 0.0             # the current rig's value (its gene in the first design)
    sensor_uid: str = ''

    def var(self) -> N.Var:
        return N.Var(self.path, self.lo, self.hi, self.choices)

    def to_dict(self):
        d = {'path': self.path, 'title': self.title, 'unit': self.unit, 'nominal': self.nominal,
             'sensor_uid': self.sensor_uid}
        if self.choices is not None:
            d['choices'] = list(self.choices)
        else:
            d.update(lo=self.lo, hi=self.hi)
        return d


def settings_of(project, given: dict | None = None) -> dict:
    g = dict(project.analysis.studies.get(SID) or {}) if given is None else dict(given)
    st = S.common_settings(g, DEFAULTS)
    st['move_m'] = S._float(st['move_m'], DEFAULTS['move_m'], 0.0, 1.0)
    st['turn_deg'] = S._float(st['turn_deg'], DEFAULTS['turn_deg'], 0.0, 180.0)
    st['factors'] = [f for f in st['factors'] if isinstance(f, dict) and f.get('path')] \
        if isinstance(st['factors'], list) else []
    return st


def _same_kind_models(kind: str) -> list:
    from .. import library as LB
    lib_kind = {'camera': 'camera', 'depth': 'depth', 'tactile': 'tactile'}.get(kind)
    return [e.id for e in LB.entries(lib_kind)] if lib_kind else []


def default_factors(project, st) -> tuple[list, list]:
    """([Factor], notes): the shifts, turns, enabled and (optionally) device of every enabled, unlocked sensor."""
    out, notes = [], []
    for s in project.variant.sensors:
        if s.locked:
            notes.append(f'{s.name} is locked: the search leaves it as it is.')
            continue
        if not s.enabled or s.kind not in SENSOR_KINDS:
            continue
        m, t = st['move_m'], st['turn_deg']
        if m > 0:
            for ax in ('dx', 'dy', 'dz'):
                d = FA.describe(f'sensor:{s.uid}.{ax}', project)
                out.append(Factor(d['path'], d['title'], 'm', -m, m, None, 0.0, s.uid))
        if t > 0:
            for ax in ('dpitch', 'dyaw'):
                d = FA.describe(f'sensor:{s.uid}.{ax}', project)
                out.append(Factor(d['path'], d['title'], '°', -t, t, None, 0.0, s.uid))
        if st['allow_remove'] and s.kind in ('camera', 'depth'):
            out.append(Factor(f'sensor:{s.uid}.enabled', f'{s.name} kept', '', choices=[True, False], nominal=True,
                              sensor_uid=s.uid))
        if st['swap_models']:
            ch = _same_kind_models(s.kind)
            ch = ([s.model_id] if s.model_id else []) + [c for c in ch if c != s.model_id]
            if len(ch) > 1:
                out.append(Factor(f'sensor:{s.uid}.model', f'{s.name} device', '', choices=ch,
                                  nominal=ch[0] if s.model_id else ch[0], sensor_uid=s.uid))
    return out, notes


def given_factors(project, items) -> tuple[list, list]:
    """The settings' own factor list, checked: a path that cannot be read, or one on a locked sensor, is left out and
    said why."""
    out, notes = [], []
    for it in items:
        path = str(it.get('path'))
        try:
            d = FA.describe(path, project)
            nominal = FA.get(project, path)
        except FA.FactorError as e:
            notes.append(f'Factor {path} left out: {e}')
            continue
        if d.get('locked') or FA.locked_target(project, path) is not None:   # a json: pointer into one too
            notes.append(f'Factor {path} left out: the sensor is locked, so the search never changes it.')
            continue
        if isinstance(it.get('choices'), list) and it['choices']:
            ch = list(it['choices'])
            if nominal in ch:
                ch.remove(nominal)
                ch.insert(0, nominal)
            out.append(Factor(path, d['title'], d['unit'], choices=ch, nominal=ch[0], sensor_uid=d.get('sensor_uid',
                                                                                                       '')))
            continue
        try:
            lo, hi = float(it.get('lo')), float(it.get('hi'))
        except (TypeError, ValueError):
            notes.append(f'Factor {path} left out: it needs numbers lo and hi, or a list of choices.')
            continue
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi < lo:
            notes.append(f'Factor {path} left out: its range {lo:g} to {hi:g} is not a range.')
            continue
        nom = float(nominal) if isinstance(nominal, (int, float)) and not isinstance(nominal, bool) else lo
        out.append(Factor(path, d['title'], d['unit'], lo, hi, None, min(max(nom, lo), hi), d.get('sensor_uid', '')))
    return out, notes


def locked_state(project) -> dict:
    """{uid: the locked sensor's dict} of the active variant: what no design may change."""
    return {s.uid: s.to_dict() for s in project.variant.sensors if s.locked}


def rig_fingerprint(project) -> str:
    from .. import provenance as PV
    return PV.sha256(PV.canonical(project.variant.to_dict()))[:16]


def _key(values) -> tuple:
    return tuple(round(v, 6) if isinstance(v, float) else (tuple(v) if isinstance(v, list) else v) for v in values)


def _changes(project, factors, values) -> list:
    """Plain words for what an option changes, per sensor."""
    by = {}
    for f, v in zip(factors, values):
        if f.choices is not None:
            if v != f.nominal:
                by.setdefault(f.sensor_uid, []).append(('choice', f, v))
        elif abs(float(v) - float(f.nominal)) > (5e-4 if f.unit == 'm' else 0.05):
            by.setdefault(f.sensor_uid, []).append(('num', f, v))
    out = []
    names = {s.uid: s.name for s in project.variant.sensors}
    for uid, items in by.items():
        name = names.get(uid, uid or 'setting')
        if any(f.path.endswith('.enabled') and v is False for _k, f, v in items):
            out.append(f'{name}: removed')
            continue
        bits = []
        for kind, f, v in items:
            what = f.path.rpartition('.')[2]
            if kind == 'choice':
                if what == 'model':
                    from .. import library as LB
                    e = LB.get(v)
                    bits.append(f'device {e.title if e is not None else v}')
                elif what != 'enabled':
                    bits.append(f'{what} {v}')
            elif what in ('dx', 'dy', 'dz'):
                bits.append(f'{(float(v) - float(f.nominal)) * 100:+.1f} cm along {what[1]}')
            elif what in ('dpitch', 'dyaw'):
                bits.append(f'{"tilted" if what == "dpitch" else "panned"} {float(v) - float(f.nominal):+.1f}°')
            else:
                bits.append(f'{f.title} {float(v):.4g}{f.unit}')
        if bits:
            out.append(f'{name}: {", ".join(bits)}')
    return out


def _price_gaps(project, factors) -> list:
    """Device ids the search can put on the rig whose library entry states no price."""
    ids = {s.model_id for s in project.variant.sensors if s.enabled}
    for f in factors:
        if f.path.endswith('.model') and f.choices:
            ids |= set(f.choices)
    return sorted(i or '(no device)' for i in ids if not OB.price_stated(i))


def optimise(project, st: dict | None = None, ctx: jobs.Ctx | None = None,  # noqa: C901 — the steps in order
             prefetch=None, cache=None) -> dict:
    """Run the optimiser on a snapshot of project (see the module docstring); the JSON-able result. prefetch(base,
    factors, rows, evaluator) and cache (a core.jobs.ResultCache for the evaluator) are core/cluster's hooks: a
    generation's designs scored in parallel into the evaluator's memo before NSGA-II asks for them one by one."""
    OB.ensure_metrics()
    ctx = ctx or jobs.Ctx()
    st = st or settings_of(project)
    t0 = time.perf_counter()
    base = project.snapshot()
    notes = []
    if st['factors']:
        factors, n1 = given_factors(base, st['factors'])
    else:
        factors, n1 = default_factors(base, st)
    notes += n1
    if not factors:
        raise jobs.StudyError('There is nothing to optimise: every sensor is locked or disabled, or the factors named '
                              'cannot be set. Unlock a sensor, or name the factors in the optimiser settings.')
    unknown = [m for m in st['objectives'] if m not in OB.M.METRICS]
    dropped = [{'id': m, 'title': m, 'reason': 'there is no metric of that name'} for m in unknown]
    objs = [m for m in st['objectives'] if m in OB.M.METRICS]
    if OB.COST in objs:
        gaps = _price_gaps(base, factors)
        if gaps:
            objs.remove(OB.COST)
            dropped.append({'id': OB.COST, 'title': OB.M.METRICS[OB.COST].title,
                            'reason': f'no price in US dollars is stated for {", ".join(gaps[:6])}'
                                      f'{" …" if len(gaps) > 6 else ""} (the library entry gives none, or gives it '
                                      'in another currency: add a USD price to a custom entry; nothing is guessed '
                                      'or converted)'})
    if OB.CAMERAS in objs and not any(f.path.endswith(('.enabled',)) for f in factors):
        objs.remove(OB.CAMERAS)
        dropped.append({'id': OB.CAMERAS, 'title': OB.M.METRICS[OB.CAMERAS].title,
                        'reason': 'no camera can be removed, so the count is the same for every design'})
    constraints, skipped = OB.constraint_rows(base, st['constraints'])
    locked = locked_state(base)
    ev = S.Evaluator(objs, constraints, st['screen'], cancel=ctx.cancel, cache=cache)
    nominal = [f.nominal for f in factors]
    ctx.report(0.0, 'scoring the current rig')
    try:
        cur = ev.score(base.snapshot(), key=_key(nominal))
    except S._Cancelled:
        raise jobs.StudyError('Cancelled before the current rig was scored.') from None
    keep = []
    for m in objs:
        if cur.status.get(m) == 'ok':
            keep.append(m)
        else:
            why = cur.reasons.get(m) or 'the study states no figure for the current rig'
            dropped.append({'id': m, 'title': OB.M.METRICS[m].title, 'reason': why.rstrip('.')})
    if not keep:
        raise jobs.StudyError('No objective can be measured on this rig: ' + '; '.join(
            f'{d["title"]}: {d["reason"]}' for d in dropped) + '.')
    objs = keep
    ev.objectives = objs
    ev.memo.clear()
    cur = ev.make(cur.figures, cur.status, cur.reasons, cur.bounds)
    ev.memo[_key(nominal)] = cur
    archive = []

    def evaluate(values):
        k = _key(values)
        try:
            p = FA.apply(base, dict(zip([f.path for f in factors], values)))
        except FA.FactorError:                        # a locked sensor (a json: pointer into one): not a design
            return np.full(len(objs), S.NOT_STATED), np.r_[np.zeros(len(constraints)), 1e6]
        if locked_state(p) != locked:                 # the invariant every design keeps
            raise AssertionError('a design changed a locked sensor')
        try:
            sc = ev.score(p, key=k)
        except S._Cancelled:
            return None
        extra = 0.0
        if not any(s.enabled and s.kind in ('camera', 'depth') for s in p.variant.sensors):
            extra = 1.0                                # a rig without a camera is not a design
        if k not in {a['key'] for a in archive[-1:]}:
            archive.append({'key': k, 'values': list(values), 'score': sc, 'extra': extra})
        g = np.concatenate([sc.g, [extra]])
        return sc.f, g

    problem = N.Problem([f.var() for f in factors], len(objs), evaluate, n_con=len(constraints) + 1)

    def prog(frac, text):
        ctx.report(0.02 + 0.93 * frac, text)
    res = N.run(problem, pop=st['population'], budget=st['budget'], seed=st['seed'], eta_c=st['eta_c'],
                eta_m=st['eta_m'], pc=st['pc'], progress=prog, cancel=ctx.cancel, initial=[nominal],
                prefetch=None if prefetch is None else (lambda rows: prefetch(base, factors, rows, ev)))
    infos = [S.metric_info(m) for m in objs]
    entries = []
    seen = set()
    for a in archive:
        if a['key'] in seen:
            continue
        seen.add(a['key'])
        sc = a['score']
        entries.append({'key': a['key'], 'values': a['values'], 'f': sc.f.tolist(), 'cv': sc.cv + a['extra'],
                        'figures': {m: sc.figures.get(m) for m in objs + [c['metric'] for c in constraints]},
                        'violations': sc.violations, 'feasible': sc.cv + a['extra'] <= 0.0})
    cur_entry = {'f': cur.f.tolist(), 'figures': dict(cur.figures), 'violations': cur.violations,
                 'feasible': cur.cv <= 0.0, 'values': nominal}
    others = [e for e in entries if e['key'] != _key(nominal)]
    chosen = S.choose(others, len(objs), st['options'],
                      reference={'f': cur_entry['f'], 'cv': cur.cv})
    for i, o in enumerate(chosen):
        o['label'] = chr(ord('A') + i)
        o['changes'] = _changes(base, factors, o['values'])
        o['summary'] = ('; '.join(o['changes']) if o['changes'] else 'No change to the rig')
    if st['confirm'] and chosen and not ctx.cancelled():
        ctx.report(0.96, 'scoring the options at the project\'s own settings')
        for o in chosen:
            try:
                p = FA.apply(base, dict(zip([f.path for f in factors], o['values'])))
                sc = ev.score(p, screen={})
            except S._Cancelled:
                break
            o['screened'] = dict(o['figures'])
            o['figures'] = {m: sc.figures.get(m) for m in objs + [c['metric'] for c in constraints]}
            o['violations'] = sc.violations
            o['f'] = sc.f.tolist()
            o['feasible'] = sc.cv <= 0.0
            o['confirmed'] = True
    for o in chosen:
        o['explanation'] = S.explain(o, chosen, cur_entry, infos, constraints)
        o['dominates_current'] = S.dominates(o['f'], cur_entry['f']) and (o['feasible'] or not cur_entry['feasible'])
        o.pop('key', None)
    head = S.headline(chosen, cur_entry, constraints, res.cancelled, len(objs))
    for o in chosen + [cur_entry]:
        o['f'] = S.finite_or_none(o['f'])
    ctx.report(1.0, '')
    return {
        'kind': 'optimise', 'project': project.name, 'variant': base.variant.name, 'variant_uid': base.variant.uid,
        'rig_fingerprint': rig_fingerprint(base),
        'settings': {k: copy.deepcopy(v) for k, v in st.items()},
        'objectives': infos, 'dropped': dropped, 'constraints': constraints, 'skipped_requirements': skipped,
        'factors': [f.to_dict() for f in factors], 'locked': [s.name for s in base.variant.sensors if s.locked],
        'current': cur_entry, 'options': chosen,
        'evaluated': [{'figures': [e['figures'].get(m) for m in objs], 'feasible': e['feasible']} for e in entries],
        'search': {'algorithm': 'NSGA-II', 'evaluations': res.evaluations, 'budget': st['budget'],
                   'population': st['population'], 'generations': res.generations, 'seed': st['seed'],
                   'studies_scored': ev.scored, 'reused': ev.reused, 'cache_hits': ev.cache.hits,
                   'seconds': round(time.perf_counter() - t0, 2),
                   'stopped': 'cancelled' if res.cancelled else 'budget'},
        'headline': head, 'notes': notes + _screen_note(st),
        'partial': bool(res.cancelled),
    }


def _screen_note(st) -> list:
    if not st['screen']:
        return ['Every design was scored at the project\'s own study settings.']
    bits = '; '.join(f'{sid}: ' + ', '.join(f'{k} {v}' for k, v in s.items()) for sid, s in st['screen'].items())
    note = f'Designs were scored at screening settings ({bits}; defaults, settings "screen").'
    if not st['confirm']:
        note += ' Turn on confirm to score the designs shown again at the project\'s own settings.'
    return [note]


def option(doc: dict, label: str) -> dict:
    o = next((o for o in doc.get('options') or [] if o.get('label') == label), None)
    if o is None:
        raise jobs.StudyError(f'The optimiser result has no design {label}.')
    return o


def option_values(doc: dict, label: str) -> dict:
    o = option(doc, label)
    return dict(zip([f['path'] for f in doc['factors']], o['values']))


def option_project(project, doc: dict, label: str):
    """A copy of project with design `label` applied (for Preview); the project itself is not changed."""
    return FA.apply(project, option_values(doc, label))


def accept(project, doc: dict, label: str) -> list:
    """Set design `label`'s factors on the active variant, in place (run inside store.edit('variant', ...)); the
    sensors changed. Raises StudyError when the rig changed since the search (its figures would not hold)."""
    if doc.get('variant_uid') != project.variant.uid or doc.get('rig_fingerprint') != rig_fingerprint(project):
        raise jobs.StudyError('The rig changed since the search ran, so its designs no longer start from it: run the '
                              'optimiser again.')
    q = option_project(project, doc, label)
    before = {s.uid: s.to_dict() for s in project.variant.sensors}
    project.variant.sensors = q.variant.sensors
    st = dict(project.analysis.studies.get(SID) or {})
    acc = list(st.get('accepted') or [])[-39:]
    acc.append({'design': label, 'variant_uid': project.variant.uid, 'date': time.strftime('%Y-%m-%d'),
                'figures': option(doc, label)['figures']})
    st['accepted'] = acc
    project.analysis.studies[SID] = st
    return [s.name for s in project.variant.sensors if before.get(s.uid) != s.to_dict()]
