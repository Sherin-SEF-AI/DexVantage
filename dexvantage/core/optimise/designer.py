"""The design agent (spec §7.11): the project's requirements in, a Pareto set of complete rigs out, each explained;
nothing applied until the user accepts one.

    settings_of(project) -> dict                   project.analysis.studies['design_agent'] with the labelled defaults
    sites(project, st) -> [Site]                   where a camera can go: static frames and robot links
    design(project, st=None, ctx=None) -> dict     the JSON-able result (also the 'design_agent' study)
    option_variant(project, doc, label) -> Variant the design as a new variant (not added)
    accept(project, doc, label) -> Variant         add it and make it active (inside one
                                                   store.edit(['variants', 'analysis'], ...) for one undo step)

A rig is up to max_cameras (default 3) camera slots, each with genes: on or off; its device (a categorical choice of
the candidate devices: settings 'models', default every camera and depth entry of the library); its mount site (a
categorical choice of the sites) and three numbers in [0, 1] that place it on that site:

- a static site (settings 'static_frames', default the cage; a static frame is fixed in the world, as the samples'
  overhead and tripod cameras are): the camera looks at the centre of the first task volume (else the primary
  robot's gripper at home) from azimuth −180° to 180°, elevation elevation_deg (default 20° to 85°) and distance
  distance_m (default 0.45 to 1.2 m), defaults;
- a link site (settings 'links', default each robot's gripper frame): the camera rides on the link, looking at a point
  3 cm beyond the fingertips (the mean of the robot's fingertip frames at home, in the link frame; the approach axis
  runs from the link origin to that point) from wrist_back_m (default 2 to 15 cm) behind the fingertips and
  wrist_radial_m (default 4 to 12 cm) off the approach axis, at any angle around it, defaults.

Locked sensors of the active variant are in every rig, unchanged; the other current sensors are left out (a design is
a complete rig) unless keep_unlocked. Each new sensor gets the bracket type core/brackets.suggest_kind gives (none
where no type fits). The objectives and requirements are the optimiser's (core/optimise/rig.py, objectives.py); a
rig with no camera is infeasible. The current rig is scored too, as the reference the explanations compare with.

Ported from Vantage core/rig_designer.py (the pattern: requirements as constraints, objectives the studies' figures,
cost from stated prices only, a seeded search budgeted in evaluations, the options spread along the front and each
explained, the current rig as the reference, accepting adds the design as a new variant with the decision recorded)
without its automotive parts (reference vehicle rigs, lidars and radars, the perception model, the LLM brief).

Known limits: as core/optimise/__init__.py; besides, mount sites are points on a sphere around the task volume and a
cone around the gripper's approach axis, not a search of every surface of the cell or the robot; the camera's roll
about its view axis is not searched (the image's up points away from the approach axis, or up in the world); the
stream mode is the device's largest.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .. import jobs
from ..frames import Pose
from ..project import MOUNT_STATIC, Mount, Sensor, Variant, new_uid
from . import nsga2 as N
from . import objectives as OB
from . import search as S

SID = 'design_agent'
DEFAULTS = dict(S.COMMON, budget=120, population=16, max_cameras=3, models=[], static_frames=['cage'], links=[],
                distance_m=[0.45, 1.2], elevation_deg=[20.0, 85.0], wrist_back_m=[0.02, 0.15],
                wrist_radial_m=[0.04, 0.12], keep_unlocked=False)
WRIST_LOOK_BEYOND_M = 0.03            # the wrist camera looks this far beyond the fingertips (a default)


@dataclass
class Site:
    key: str
    title: str
    parent_frame: str
    robot_uid: str = ''
    target: np.ndarray = field(default_factory=lambda: np.zeros(3))      # in the parent frame
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))   # link: the approach axis
    kind: str = 'static'          # 'static' | 'link'

    def to_dict(self):
        return {'key': self.key, 'title': self.title, 'parent_frame': self.parent_frame, 'robot_uid': self.robot_uid,
                'target': [float(v) for v in self.target], 'axis': [float(v) for v in self.axis], 'kind': self.kind}


def _range(v, default, lo, hi):
    if isinstance(v, (list, tuple)) and len(v) == 2:
        a, b = S._float(v[0], default[0], lo, hi), S._float(v[1], default[1], lo, hi)
        if b >= a:
            return [a, b]
    return list(default)


def settings_of(project, given: dict | None = None) -> dict:
    g = dict(project.analysis.studies.get(SID) or {}) if given is None else dict(given)
    st = S.common_settings(g, DEFAULTS)
    st['max_cameras'] = S._int(st['max_cameras'], DEFAULTS['max_cameras'], 1, 8)
    st['models'] = [str(m) for m in st['models']] if isinstance(st['models'], list) else []
    st['static_frames'] = [f for f in (st['static_frames'] if isinstance(st['static_frames'], list) else [])
                           if f in MOUNT_STATIC] or ([] if isinstance(g.get('static_frames'), list) and
                                                     not g['static_frames'] else ['cage'])
    st['links'] = [str(x) for x in st['links']] if isinstance(st['links'], list) else []
    st['distance_m'] = _range(st['distance_m'], DEFAULTS['distance_m'], 0.05, 5.0)
    st['elevation_deg'] = _range(st['elevation_deg'], DEFAULTS['elevation_deg'], -89.0, 89.0)
    st['wrist_back_m'] = _range(st['wrist_back_m'], DEFAULTS['wrist_back_m'], -0.2, 0.5)
    st['wrist_radial_m'] = _range(st['wrist_radial_m'], DEFAULTS['wrist_radial_m'], 0.0, 0.5)
    return st


def look_at(position, target, up=(0.0, 0.0, 1.0)) -> Pose:
    """The sensor body pose (REP-103 body: x along the view, z up) at position looking at target."""
    from ..robot.samples import look_at as LA
    return LA(np.asarray(position, float), np.asarray(target, float), up)


def task_target(project) -> np.ndarray:
    """The world point static cameras look at: the first task volume's centre, else the primary robot's gripper (or
    its base) at home."""
    if project.workspace.volumes:
        return np.asarray(project.workspace.volumes[0].center, float)
    r = project.primary_robot
    if r is not None:
        try:
            from ..robot.model import get_model
            m = get_model(project, r.uid)
            T = m.link_pose(m.home, r.gripper_frame) if r.gripper_frame else m.base
            return np.asarray(T[:3, 3], float)
        except Exception:                        # noqa: BLE001 — a robot that cannot be posed: the origin
            pass
    return np.zeros(3)


def sites(project, st) -> tuple[list, list]:
    """([Site], notes)."""
    out, notes = [], []
    tgt = task_target(project)
    for fr in st['static_frames']:
        out.append(Site(f'static:{fr}', f'the {fr}', fr, '', tgt.copy(), np.array([0.0, 0.0, 1.0]), 'static'))
    from ..robot.model import RobotModelError, get_model
    for r in project.robots:
        links = st['links'] or ([r.gripper_frame] if r.gripper_frame else [])
        if not links:
            notes.append(f'{r.name} has no gripper frame, so no camera is placed on it (name links in the design '
                         'agent settings).')
            continue
        try:
            m = get_model(project, r.uid)
        except RobotModelError as e:
            notes.append(f'{r.name} cannot be posed, so no camera is placed on it: {e}')
            continue
        for ln in links:
            if ln not in m.link_names:
                notes.append(f'{r.name} has no link {ln}: left out of the mount sites.')
                continue
            T = m.link_pose(m.home, ln)
            Ti = np.linalg.inv(T)
            tips = [f for f in r.fingertip_frames if f in m.link_names]
            if tips:
                P = np.array([(Ti @ m.link_pose(m.home, f))[:3, 3] for f in tips])
                tip = P.mean(axis=0)
            else:
                tip = np.zeros(3)
            n = float(np.linalg.norm(tip))
            axis = tip / n if n > 0.01 else np.array([0.0, 0.0, 1.0])
            if n <= 0.01:
                tip = axis * 0.10                  # no fingertip frames: 10 cm along the link's z (a default)
                notes.append(f'{r.name} names no fingertip frames on {ln}: the wrist camera looks 10 cm along the '
                             'link\'s z axis (a default).')
            out.append(Site(f'link:{r.uid}:{ln}', f'{ln} of {r.name}', ln, r.uid, tip, axis, 'link'))
    return out, notes


def place(site: Site, a: float, b: float, c: float, st) -> Pose:
    """The sensor's mount pose in the site's parent frame for the three placement genes (see the module docstring)."""
    if site.kind == 'static':
        az = math.radians(-180.0 + 360.0 * a)
        el = math.radians(st['elevation_deg'][0] + (st['elevation_deg'][1] - st['elevation_deg'][0]) * b)
        d = st['distance_m'][0] + (st['distance_m'][1] - st['distance_m'][0]) * c
        pos = site.target + d * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        up = (0.0, 0.0, 1.0) if abs(math.degrees(el)) < 80 else (math.cos(az), math.sin(az), 0.0)
        return look_at(pos, site.target, up)
    ax = site.axis / max(np.linalg.norm(site.axis), 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(ax[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(ax, ref)
    u /= np.linalg.norm(u)
    v = np.cross(ax, u)
    phi = 2.0 * math.pi * a
    back = st['wrist_back_m'][0] + (st['wrist_back_m'][1] - st['wrist_back_m'][0]) * b
    rad = st['wrist_radial_m'][0] + (st['wrist_radial_m'][1] - st['wrist_radial_m'][0]) * c
    radial = math.cos(phi) * u + math.sin(phi) * v
    pos = site.target - back * ax + rad * radial
    return look_at(pos, site.target + WRIST_LOOK_BEYOND_M * ax, tuple(radial))


def candidate_models(st) -> tuple[list, list]:
    from .. import library as LB
    notes = []
    if st['models']:
        ok = [m for m in st['models'] if LB.get(m) is not None and LB.get(m).kind in ('camera', 'depth')]
        bad = [m for m in st['models'] if m not in ok]
        if bad:
            notes.append(f'Not camera or depth devices of the library, left out: {", ".join(bad)}.')
        return ok, notes
    return [e.id for e in LB.entries('depth')] + [e.id for e in LB.entries('camera')], notes


def _slot_vars(k: int, models, site_list) -> list:
    return [N.Var(f'slot{k}.on', choices=[True, False]), N.Var(f'slot{k}.model', choices=list(models)),
            N.Var(f'slot{k}.site', choices=[s.key for s in site_list]), N.Var(f'slot{k}.a', 0.0, 1.0),
            N.Var(f'slot{k}.b', 0.0, 1.0), N.Var(f'slot{k}.c', 0.0, 1.0)]


def _short(e) -> str:
    """A device's short name: its name without the maker's words."""
    n = e.name
    for w in e.maker.split():
        n = n.replace(w, '').strip()
    return n or e.name


def build(project, values, models, site_list, st, keep) -> tuple[list, list]:
    """(new sensors, their descriptions) for decoded genes (6 per slot)."""
    from .. import brackets as B
    from .. import library as LB
    by = {s.key: s for s in site_list}
    sensors, desc = [], []
    slot0 = max([s.slot for s in keep] + [-1]) + 1
    for k in range(len(values) // 6):
        on, mid, skey, a, b, c = values[6 * k:6 * k + 6]
        if not on:
            continue
        e = LB.get(mid)
        site = by[skey]
        pose = place(site, float(a), float(b), float(c), st)
        name = f'{_short(e)} on {site.parent_frame}'
        s = Sensor(uid=f'design{k:02d}', name=name, kind='depth' if e.kind == 'depth' else 'camera', model_id=mid,
                   mount=Mount(site.parent_frame, site.robot_uid, pose), slot=slot0 + len(sensors))
        s.mount.bracket = ''
        if e.data.get('size_mm'):                # a bracket is built around the stated body size
            try:
                kind, _why = B.suggest_kind(project, s)
                s.mount.bracket = kind if kind and kind != 'none' else ''
            except Exception:                    # noqa: BLE001 — a bracket guess must not stop the design
                s.mount.bracket = ''
        sensors.append(s)
        if site.kind == 'static':
            az = -180.0 + 360.0 * float(a)
            el = st['elevation_deg'][0] + (st['elevation_deg'][1] - st['elevation_deg'][0]) * float(b)
            d = st['distance_m'][0] + (st['distance_m'][1] - st['distance_m'][0]) * float(c)
            desc.append(f'{e.title} on {site.title}, {d:.2f} m from the task volume, {el:.0f}° up, azimuth {az:.0f}°')
        else:
            back = st['wrist_back_m'][0] + (st['wrist_back_m'][1] - st['wrist_back_m'][0]) * float(b)
            rad = st['wrist_radial_m'][0] + (st['wrist_radial_m'][1] - st['wrist_radial_m'][0]) * float(c)
            desc.append(f'{e.title} on {site.title}, {rad * 100:.0f} cm off the approach axis and '
                        f'{back * 100:.0f} cm behind the fingertips')
    return sensors, desc


def _variant_with(project, sensors, keep, name='Design (preview)') -> Variant:
    return Variant(name=name, sensors=[copy.deepcopy(s) for s in keep] + list(sensors))


def design(project, st: dict | None = None, ctx: jobs.Ctx | None = None) -> dict:  # noqa: C901 — the steps in order
    """Run the design agent on a snapshot of project; the JSON-able result (see the module docstring)."""
    OB.ensure_metrics()
    ctx = ctx or jobs.Ctx()
    st = st or settings_of(project)
    t0 = time.perf_counter()
    base = project.snapshot()
    models, notes = candidate_models(st)
    if not models:
        raise jobs.StudyError('The design agent has no camera to choose: the library has no camera or depth device '
                              'among the models named in its settings.')
    site_list, n2 = sites(base, st)
    notes += n2
    if not site_list:
        raise jobs.StudyError('The design agent has nowhere to put a camera: name a static frame (cage, tripod, table '
                              'or world) or a robot link in its settings.')
    keep = [s for s in base.variant.sensors if s.locked or st['keep_unlocked']]
    locked = {s.uid: s.to_dict() for s in keep if s.locked}
    dropped = [{'id': m, 'title': m, 'reason': 'there is no metric of that name'}
               for m in st['objectives'] if m not in OB.M.METRICS]
    objs = [m for m in st['objectives'] if m in OB.M.METRICS]
    if OB.COST in objs:
        gaps = [m for m in models if not OB.price_stated(m)]
        gaps += [s.model_id or s.name for s in keep if s.enabled and not OB.price_stated(s.model_id)]
        if gaps:
            objs.remove(OB.COST)
            dropped.append({'id': OB.COST, 'title': OB.M.METRICS[OB.COST].title,
                            'reason': f'no price in US dollars is stated for {len(gaps)} of the devices it can choose '
                                      f'(for example {", ".join(gaps[:3])}): the library entries give none, or give '
                                      'it in another currency; add USD prices to custom entries, or name only priced '
                                      'devices. Nothing is guessed or converted'})
    if not objs:
        raise jobs.StudyError('The design agent has no objective to rank rigs by: name metrics in its settings.')
    constraints, skipped = OB.constraint_rows(base, st['constraints'])
    ev = S.Evaluator(objs, constraints, st['screen'], cancel=ctx.cancel)
    ctx.report(0.0, 'scoring the current rig')
    try:
        cur = ev.score(base.snapshot())
    except S._Cancelled:
        raise jobs.StudyError('Cancelled before the current rig was scored.') from None
    var = []
    for k in range(st['max_cameras']):
        var += _slot_vars(k, models, site_list)
    archive = []

    def evaluate(values):
        sensors, desc = build(base, values, models, site_list, st, keep)
        p = base.snapshot()
        v = _variant_with(p, sensors, keep)
        p.variants.append(v)
        p.active_uid = v.uid
        if {s.uid: s.to_dict() for s in v.sensors if s.locked} != locked:
            raise AssertionError('a design changed a locked sensor')
        n_cam = sum(s.enabled and s.kind in ('camera', 'depth') for s in v.sensors)
        key = tuple(round(x, 6) if isinstance(x, float) else x for x in values)
        try:
            sc = ev.score(p, key=key)
        except S._Cancelled:
            return None
        extra = 0.0 if n_cam else 1.0
        archive.append({'key': key, 'values': list(values), 'score': sc, 'extra': extra,
                        'sensors': [s.to_dict() for s in sensors], 'desc': desc})
        return sc.f, np.concatenate([sc.g, [extra]])

    problem = N.Problem(var, len(objs), evaluate, n_con=len(constraints) + 1)
    # first designs: one camera of each site (the first candidate device, placed mid-range), then random ones
    initial = []
    for s in site_list[:st['population']]:
        row = []
        for k in range(st['max_cameras']):
            row += [k == 0, models[0], s.key, 0.5 if s.kind == 'link' else 0.375, 0.5, 0.5]
        initial.append(row)
    res = N.run(problem, pop=st['population'], budget=st['budget'], seed=st['seed'], eta_c=st['eta_c'],
                eta_m=st['eta_m'], pc=st['pc'], progress=lambda f, t: ctx.report(0.02 + 0.93 * f, t),
                cancel=ctx.cancel, initial=initial)
    # an objective no design states is left out of the comparison (with the study's reason)
    stated = [i for i, m in enumerate(objs) if any(a['score'].figures.get(m) is not None for a in archive)]
    for i, m in enumerate(objs):
        if i not in stated:
            why = next((a['score'].reasons.get(m) for a in archive if a['score'].reasons.get(m)), '') or \
                'the study states no figure for any design'
            dropped.append({'id': m, 'title': OB.M.METRICS[m].title, 'reason': why.rstrip('.')})
    kept = [objs[i] for i in stated]
    if not kept:
        raise jobs.StudyError('No objective could be measured on any design: ' + '; '.join(
            f'{d["title"]}: {d["reason"]}' for d in dropped) + '.')
    infos = [S.metric_info(m) for m in kept]
    entries, seen = [], set()
    for a in archive:
        if a['key'] in seen:
            continue
        seen.add(a['key'])
        sc = a['score']
        f = [sc.f[i] for i in stated]
        entries.append({'key': a['key'], 'values': a['values'], 'f': f, 'cv': sc.cv + a['extra'],
                        'figures': {m: sc.figures.get(m) for m in kept + [c['metric'] for c in constraints]},
                        'violations': sc.violations, 'feasible': sc.cv + a['extra'] <= 0.0,
                        'sensors': a['sensors'], 'desc': a['desc']})
    cur_entry = {'f': [cur.f[i] for i in stated], 'figures': dict(cur.figures), 'violations': cur.violations,
                 'feasible': cur.cv <= 0.0, 'sensors': [s.to_dict() for s in base.variant.sensors],
                 'summary': f'The current rig, {base.variant.name}'}
    chosen = S.choose(entries, len(kept), st['options'],
                      reference={'f': cur_entry['f'], 'cv': cur.cv})
    for i, o in enumerate(chosen):
        o['label'] = chr(ord('A') + i)
        n = len(o['sensors'])
        locked_words = f' plus the {len(keep)} kept sensor{"s" * (len(keep) != 1)}' if keep else ''
        o['summary'] = (f'{n} camera{"s" * (n != 1)}{locked_words}: ' + '; '.join(o['desc'])) if n else \
            f'No new camera{locked_words}'
    if st['confirm'] and chosen and not ctx.cancelled():
        ctx.report(0.96, 'scoring the options at the project\'s own settings')
        for o in chosen:
            p = base.snapshot()
            v = Variant(name='confirm', sensors=[copy.deepcopy(s) for s in keep] +
                        [Sensor.from_dict(d) for d in o['sensors']])
            p.variants.append(v)
            p.active_uid = v.uid
            try:
                sc = ev.score(p, screen={})
            except S._Cancelled:
                break
            o['screened'] = dict(o['figures'])
            o['figures'] = {m: sc.figures.get(m) for m in kept + [c['metric'] for c in constraints]}
            o['violations'] = sc.violations
            o['f'] = [sc.f[objs.index(m)] for m in kept]
            o['feasible'] = sc.cv <= 0.0 and bool(o['sensors'] or keep)
            o['confirmed'] = True
    for o in chosen:
        o['explanation'] = S.explain(o, chosen, cur_entry, infos, constraints)
        o['dominates_current'] = S.dominates(o['f'], cur_entry['f']) and (o['feasible'] or not cur_entry['feasible'])
        o.pop('key', None)
        o.pop('desc', None)
    head = S.headline(chosen, cur_entry, constraints, res.cancelled, len(kept))
    for o in chosen + [cur_entry]:
        o['f'] = S.finite_or_none(o['f'])
    ctx.report(1.0, '')
    return {
        'kind': 'design_agent', 'project': project.name, 'variant': base.variant.name,
        'variant_uid': base.variant.uid,
        'settings': {k: copy.deepcopy(v) for k, v in st.items()},
        'objectives': infos, 'dropped': dropped, 'constraints': constraints, 'skipped_requirements': skipped,
        'models': models, 'sites': [s.to_dict() for s in site_list],
        'kept': [s.name for s in keep], 'locked': [s.name for s in keep if s.locked],
        'current': cur_entry, 'options': chosen,
        'evaluated': [{'figures': [e['figures'].get(m) for m in kept], 'feasible': e['feasible']} for e in entries],
        'search': {'algorithm': 'NSGA-II', 'evaluations': res.evaluations, 'budget': st['budget'],
                   'population': st['population'], 'generations': res.generations, 'seed': st['seed'],
                   'studies_scored': ev.scored, 'reused': ev.reused, 'cache_hits': ev.cache.hits,
                   'seconds': round(time.perf_counter() - t0, 2),
                   'stopped': 'cancelled' if res.cancelled else 'budget'},
        'headline': head, 'notes': notes + _notes(st, keep, base),
        'partial': bool(res.cancelled),
    }


def _notes(st, keep, base) -> list:
    from .rig import _screen_note
    out = _screen_note(st)
    dropped = [s.name for s in base.variant.sensors if s not in keep]
    if dropped:
        out.append(f'The current rig\'s unlocked sensors ({", ".join(dropped)}) are not in the designs: each design is '
                   'a complete rig (turn on keep_unlocked to design around them).')
    return out


def option(doc: dict, label: str) -> dict:
    o = next((o for o in doc.get('options') or [] if o.get('label') == label), None)
    if o is None:
        raise jobs.StudyError(f'The design agent result has no design {label}.')
    return o


def option_variant(project, doc: dict, label: str, name: str | None = None) -> Variant:
    """Design `label` as a new Variant (fresh uids): the project's locked (and, with keep_unlocked, other) sensors as
    they are now, and the design's sensors."""
    o = option(doc, label)
    keep_all = bool((doc.get('settings') or {}).get('keep_unlocked'))
    keep = [copy.deepcopy(s) for s in project.variant.sensors if s.locked or keep_all]
    new = []
    for d in o['sensors']:
        s = Sensor.from_dict(d)
        s.uid = new_uid()
        new.append(s)
    figs = '; '.join(f'{i["title"]} {S.fmt(o["figures"].get(i["id"]), i["unit"])}' for i in doc['objectives'])
    return Variant(name=name or f'Design {label} (design agent)', sensors=keep + new,
                   note=f'Design agent, design {label}: {figs}. {o.get("summary", "")}')


def accept(project, doc: dict, label: str, why: str = '') -> Variant:
    """Add design `label` as a new variant and make it active, and record the decision in the settings and the design
    log (run inside store.edit(['variants', 'analysis'], ...) for one undo step). Returns the new Variant."""
    v = option_variant(project, doc, label)
    taken = {x.name for x in project.variants}
    base, k = v.name, 2
    while v.name in taken:
        v.name, k = f'{base} ({k})', k + 1
    project.variants.append(v)
    project.active_uid = v.uid
    st = dict(project.analysis.studies.get(SID) or {})
    acc = list(st.get('accepted') or [])[-39:]
    acc.append({'design': label, 'variant': v.name, 'variant_uid': v.uid, 'date': time.strftime('%Y-%m-%d'),
                'why': str(why)[:300], 'figures': option(doc, label)['figures']})
    st['accepted'] = acc
    project.analysis.studies[SID] = st
    others = [o['label'] for o in doc.get('options') or [] if o['label'] != label]
    project.add_log('decision', f'Accepted design {label} of the design agent as {v.name}'
                    + (f' ({why})' if why else '') + (f'; not taken: {", ".join(others)}' if others else ''))
    return v
