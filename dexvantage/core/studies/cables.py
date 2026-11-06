"""Cable flex across joints (spec §7.5): for every cable in project.cables, the length each run between two attachment
points needs over the full range of the joints it crosses and along the project's trajectories, the slack loop per
joint, the bend radius against the cable's stated minimum bend radius times a factor, and the pull, snag and bend
flags; then the harness sheet (cable type, length, slack loop per joint, connectors, strain relief) as JSON, CSV and
HTML.

Ported from Vantage core/harness.py: the harness-sheet idea and its export (to_doc / to_csv / files / export), the
findings with a severity and a plain message, the "good" finding when nothing is wrong, the Ramer–Douglas–Peucker
simplify used for the exported polylines. Stripped: the vehicle body surface, its routing graph and entry points,
the link-reach limits and the RF and power checks (automotive), replaced by the articulated-joint model in
core/cables.py.

What it does, per cable (core/cables.py has the geometry and its equations):
1. Each run between consecutive attachments crosses the moving joints on the kinematic tree's path between their
   links. Each crossed joint is swept over its full range (the model file's limits; an unlimited joint one turn
   around home, said so) with the other joints at home (sweep_samples samples, default 181), and every trajectory of
   the robot is played (its samples, with poses added so no joint moves more than max_step_deg, default 2°,
   between two).
2. The run's length at each pose: the routed arc model when it crosses one revolute joint, else the straight chord.
   Its bend radius estimate: the smaller of the route's bend on the joint cylinder and the slack loop's radius as a
   circular arc of the laid length over the run's length (core/cables.py bend_estimate).
   The slack a joint needs is the largest minus the smallest length over its sweep; the suggested laid length of a
   run is its largest length over every pose plus a service loop (loop_m, default 0.05 m).
3. Pull: a run laid shorter than it needs at some pose (its length_to_next_m), or a cable whose provided length is
   shorter than its runs need together. Bend: the slack loop's smallest bend radius over the poses is below the
   cable's stated minimum bend radius × bend_factor (default 1.5, a DexVantage margin, not a maker's figure; the
   moving / dynamic figure is used when the maker states one). Snag: at home or along a trajectory the slack loop's
   polyline passes through one of the robot's links (outside a seat of seat_m, default 0.02 m, around each
   attachment), tested by ray casting against the links' colliding triangles.
4. Apply sets every run's laid length to its suggested one (one undo step); a project without cables gets a
   suggested route for each link-mounted sensor (labelled a suggestion), which Apply adds.

Settings (project.analysis.studies['cables'], every one a labelled default): bend_factor 1.5, loop_m 0.05,
sweep_samples 181, seat_m 0.02, max_step_deg 2, max_poses 3000 per trajectory, points 49 per run polyline,
suggest_when_empty true, joint_ranges 'model' (sweep each crossed joint over the model file's limits; 'trajectories'
sweeps it over the span the robot's trajectories use, home included, so the cable is sized for the task rather than
for every turn the joint can make).

Known limits: the cable is a stiffness-free curve (no sag, torsion or friction) and its slack loop a circular bow
outwards from the joint, so the bend radius is an estimate; a run across several joints is a straight chord (add an
attachment per link); the sweep moves one joint at a time (combinations come only from the trajectories); the wrap
radius is one cylinder about the axis; an unlimited joint cannot be crossed by a cable at all and is swept one turn
only; snags are tested at sampled poses of the trajectories (and at home), not over the continuous swept volume,
and only against the cable's own robot (not other robots, objects or tables); the run from the last attachment to
the host is not modelled; connectors and strain relief are the user's entries, not checked.
"""
from __future__ import annotations

import csv
import html
import io
import json
import math
import os

import numpy as np

from .. import cables as CB
from ..jobs import Ctx, StudyError, study

SCHEMA = 'dexvantage.cables/1'
DEFAULTS = {
    'bend_factor': 1.5,          # default: allowed bend radius = stated minimum × this (a margin for flexing)
    'loop_m': 0.05,              # default: service loop added to a run's largest length
    'sweep_samples': 181,        # default: samples over each crossed joint's range
    'seat_m': 0.02,              # default: around each attachment the cable may touch its link (the clamp)
    'max_step_deg': 2.0,         # default: poses added along trajectories so no joint moves more than this
    'max_poses': 3000,           # default: poses per trajectory at most (evenly thinned beyond)
    'points': 49,                # default: points per run polyline
    'suggest_when_empty': True,  # default: a project without cables gets a suggested route per link-mounted sensor
    'joint_ranges': 'model',     # default: sweep each crossed joint over the model file's limits ('trajectories': over
                                 # the span the project's trajectories of that robot use, home included)
}
JOINT_RANGES = ('model', 'trajectories')
LIMITS = {'bend_factor': (0.1, 100.0), 'loop_m': (0.0, 5.0), 'sweep_samples': (3, 5001), 'seat_m': (0.0, 0.5),
          'max_step_deg': (0.05, 90.0), 'max_poses': (10, 100000), 'points': (9, 401)}
SEV = ('good', 'warning', 'serious', 'critical')
WORDS = {'good': 'clear', 'warning': 'check', 'serious': 'at risk', 'critical': 'pulls'}
KIND_SEV = {'pull': 'critical', 'snag': 'serious', 'bend': 'serious', 'unset': 'warning', 'unknown': 'warning',
            'setup': 'warning', 'endless': 'serious'}
ROUND_UP_M = 0.005              # suggested laid lengths are rounded up to this (a practical cut length)
UNLIMITED_HALF_RANGE = math.pi  # an unlimited joint is swept this far either side of home


def _worst(*sevs) -> str:
    return max(sevs or ('good',), key=SEV.index)


def _r(v, d=4):
    if v is None:
        return None
    v = float(v)
    return round(v, d) + 0.0 if math.isfinite(v) else None


# ====================================================================== settings
def settings_of(project) -> dict:
    """The study's settings with the defaults filled in; 'defaults' lists the keys left at their default."""
    given = dict(project.analysis.studies.get('cables') or {})
    out, used = dict(DEFAULTS), []
    for k, d in DEFAULTS.items():
        v = given.get(k)
        if v is None:
            used.append(k)
            continue
        if isinstance(d, bool):
            out[k] = bool(v)
            continue
        if k == 'joint_ranges':
            if v not in JOINT_RANGES:
                raise StudyError(f'The cable setting joint_ranges must be one of {", ".join(JOINT_RANGES)}; it is '
                                 f'{v!r}.')
            out[k] = v
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            raise StudyError(f'The cable setting {k} must be a number; it is {v!r}.') from None
        lo, hi = LIMITS[k]
        if not math.isfinite(x) or not lo <= x <= hi:
            raise StudyError(f'The cable setting {k} must be between {lo:g} and {hi:g}; it is {x:g}.')
        out[k] = int(round(x)) if isinstance(d, int) else x
    out['defaults'] = used
    return out


# ====================================================================== poses
def trajectory_poses(model, traj: dict, home, max_step_deg: float, max_poses: int) -> tuple:
    """(Q (K, nq), t (K,), note) along a trajectory: its samples, with poses added between two samples so no revolute
    joint moves more than max_step_deg (no prismatic one more than 10 mm), thinned evenly to max_poses."""
    from ..robot import trajectory as TR
    t = np.asarray(traj.get('t') or [], float)
    if len(t) == 0:
        return np.zeros((0, model.nq)), np.zeros(0), ''
    Qs = np.array([TR.on_model(traj, model.joint_names, home, float(x)) for x in t]).reshape(len(t), model.nq)
    rev = np.array([jt != 'prismatic' for jt in model.joint_types])
    step = np.where(rev, math.radians(max_step_deg), 0.01)
    ts = [t[0]]
    for k in range(len(t) - 1):
        n = int(np.ceil(np.max(np.abs(Qs[k + 1] - Qs[k]) / step))) if model.nq else 1
        n = max(1, min(n, 10000))
        ts += list(t[k] + (t[k + 1] - t[k]) * np.arange(1, n + 1) / n)
    ts = np.asarray(ts)
    note = ''
    if len(ts) > max_poses:
        note = (f'Trajectory "{traj.get("name", "")}": {len(ts)} poses thinned evenly to {max_poses} (max_poses); '
                'snags between them are not tested.')
        ts = ts[np.unique(np.round(np.linspace(0, len(ts) - 1, max_poses)).astype(int))]
    Q = np.array([TR.on_model(traj, model.joint_names, home, float(x)) for x in ts]).reshape(len(ts), model.nq)
    return np.clip(Q, model.lower, model.upper), ts, note


class _Run:
    """One run between two attachments on one robot model."""

    def __init__(self, model, a: dict, b: dict, wrap_radius):
        self.model = model
        self.la, self.lb = model.link_index(a['link']), model.link_index(b['link'])
        self.pa, self.pb = np.asarray(a['point'], float), np.asarray(b['point'], float)
        self.crossed = CB.crossed_joints(model, a['link'], b['link'])
        self.kind = CB.route_kind(model, self.crossed)
        home = model.home
        T, anc, ax = CB.posed(model, home[None])
        self.axis_j = next((j for j, _s in self.crossed if model.joint_types[j] != 'prismatic'), None)
        self.rho = None
        self.rho_default = wrap_radius is None
        self.theta_home = 0.0
        if self.kind == 'wrap':
            j, _side = self.crossed[0]
            A, B = CB.world_points(T, self.la, self.pa), CB.world_points(T, self.lb, self.pb)
            cyl = CB.cylindrical(A, B, anc[:, j], ax[:, j])
            self.da, self.db = float(cyl['da'][0]), float(cyl['db'][0])
            self.theta_home = float(cyl['angle'][0])
            self.rho = float(CB.default_wrap_radius(self.da, self.db)) if wrap_radius is None else \
                min(float(wrap_radius), self.da, self.db)
            self.rho_clipped = wrap_radius is not None and float(wrap_radius) > min(self.da, self.db) + 1e-12

    def evaluate(self, T, anc, ax, Q) -> dict:
        """Lengths (K,), ends, and what route() needs, at the posed joint vectors Q (K, nq)."""
        A, B = CB.world_points(T, self.la, self.pa), CB.world_points(T, self.lb, self.pb)
        out = {'A': A, 'B': B, 'O': None, 'u': None, 'wraps': np.zeros(len(A), bool),
               'bend_route': np.full(len(A), np.inf)}
        if self.kind == 'wrap':
            j, side = self.crossed[0]
            cyl = CB.cylindrical(A, B, anc[:, j], ax[:, j])
            theta = self.theta_home + side * (Q[:, j] - self.model.home[j])
            w = CB.wrap_length(cyl['da'], cyl['db'], cyl['zb'] - cyl['za'], theta, self.rho)
            out.update(L=w['L'], cyl=cyl, theta=theta, O=anc[:, j], u=ax[:, j], wraps=w['wraps'],
                       turns=np.abs(theta) / (2 * np.pi), bend_route=w['bend_m'])
        else:
            out['L'] = np.linalg.norm(B - A, axis=1)
            if self.axis_j is not None:
                out['O'], out['u'] = anc[:, self.axis_j], ax[:, self.axis_j]
        return out

    def route(self, ev: dict, n: int) -> np.ndarray:
        if self.kind == 'wrap':
            return CB.wrap_polyline(ev['cyl'], ev['theta'], self.rho, n)
        return CB.chord_polyline(ev['A'], ev['B'], n)

    def loop(self, ev: dict, s, n: int) -> np.ndarray:
        """(K, n, 3) the slack loop of laid length s at each pose."""
        P = self.route(ev, n)
        return CB.bulge(P, s, CB.outward(P, ev['O'], ev['u']))


def _take(ev: dict, idx) -> dict:
    out = {}
    for k, v in ev.items():
        if isinstance(v, dict):
            out[k] = _take(v, idx)
        elif isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] > 0:
            out[k] = v[idx]
        else:
            out[k] = v
    return out


def _concat(evs: list) -> dict:
    out = {}
    for k in evs[0]:
        vs = [e[k] for e in evs]
        if isinstance(vs[0], dict):
            out[k] = _concat(vs)
        elif isinstance(vs[0], np.ndarray):
            out[k] = np.concatenate(vs)
        else:
            out[k] = vs[0]
    return out


# ====================================================================== one cable
def _sensor(project, uid):
    for v in project.variants:
        s = v.sensor(uid)
        if s is not None:
            return s
    return None


def _flag(kind, message, **where) -> dict:
    return {'kind': kind, 'severity': KIND_SEV[kind], 'message': message, **where}


def analyse_cable(project, cable: dict, st: dict, ctx: Ctx | None = None, frac=(0.0, 1.0)) -> dict:
    """The study's result for one cable record (see the module docstring)."""
    from ..robot.model import RobotModelError, get_model
    c = CB.clean_cable(cable)
    sensor = _sensor(project, c['sensor_uid']) if c['sensor_uid'] else None
    mb = c['min_bend_radius_m']
    out = {'uid': c['uid'], 'name': c['name'], 'sensor_uid': c['sensor_uid'],
           'sensor': sensor.name if sensor is not None else '', 'type': c['type'],
           'type_label': CB.TYPE_LABEL.get(c['type'], 'Cable'), 'library_id': c['library_id'],
           'diameter_m': c['diameter_m'], 'min_bend_radius': mb, 'bend_factor': st['bend_factor'],
           'allowed_bend_m': None if mb is None else _r(mb['value'] * st['bend_factor'], 5),
           'length_m': c['length_m'], 'connectors': c['connectors'], 'strain_relief': c['strain_relief'],
           'note': c['note'], 'segments': [], 'flags': [], 'notes': [], 'status': 'warning',
           'required_m': None, 'suggested_m': None, 'laid_m': None, 'robot_uid': ''}
    atts = c['attachments']
    if c['sensor_uid'] and sensor is None:
        out['notes'].append('Its sensor is not in the project any more.')
    if len(atts) < 2:
        out['flags'].append(_flag('setup', f'{c["name"]} has {len(atts)} attachment point'
                                           f'{"" if len(atts) == 1 else "s"}: add at least two (Suggest '
                                           'attachments places them along the arm).'))
        return out
    ruids = {a['robot_uid'] for a in atts}
    if len(ruids) != 1 or project.robot(next(iter(ruids))) is None:
        out['flags'].append(_flag('setup', f'The attachments of {c["name"]} must all be on one robot of the '
                                           'project; pick their links again.'))
        return out
    ruid = next(iter(ruids))
    out['robot_uid'] = ruid
    try:
        model = get_model(project, ruid)
    except RobotModelError as e:
        out['flags'].append(_flag('setup', f'The robot of {c["name"]} cannot be compiled: {e}'))
        return out
    bad = [a['link'] for a in atts if a['link'] not in model.link_names]
    if bad:
        out['flags'].append(_flag('setup', f'{model.name} has no link named {bad[0]}: pick the attachment\'s link '
                                           'again.'))
        return out
    n = int(st['points'])
    home = model.home
    # the poses: home, each crossed joint's sweep (built per run), the trajectories
    trajs = [t for t in project.tasks.trajectories if isinstance(t, dict) and t.get('robot_uid') == ruid]
    TQ, Tt, Tname = [], [], []
    for tr in trajs:
        Q, ts, note = trajectory_poses(model, tr, home, st['max_step_deg'], int(st['max_poses']))
        if note:
            out['notes'].append(note)
        TQ.append(Q)
        Tt += list(ts)
        Tname += [tr.get('name', 'trajectory')] * len(ts)
    TQ = np.concatenate(TQ) if TQ else np.zeros((0, model.nq))
    Q0 = np.concatenate([home[None], TQ])                       # home + trajectory poses (snag poses)
    P0 = CB.posed(model, Q0)
    if not trajs:
        out['notes'].append('No trajectory of this robot in the project: snags are tested at home only (import a '
                            'trajectory under Tasks to test the motion).')
    by_traj = st['joint_ranges'] == 'trajectories' and len(TQ) > 0
    if st['joint_ranges'] == 'trajectories' and not by_traj:
        out['notes'].append('Joint ranges are set to the trajectories, but this robot has none in the project: the '
                            'model file\'s limits are used.')
    grids = CB.link_grids(model)
    f0, f1 = frac
    nseg = len(atts) - 1
    for i in range(nseg):
        if ctx is not None:
            if ctx.cancelled():
                break
            ctx.report(f0 + (f1 - f0) * i / nseg, f'{c["name"]}: run {i + 1} of {nseg}')
        a, b = atts[i], atts[i + 1]
        run = _Run(model, a, b, c['wrap_radius_m'])
        seg = {'index': i, 'from': {'link': a['link'], 'point': [_r(v) for v in a['point']]},
               'to': {'link': b['link'], 'point': [_r(v) for v in b['point']]}, 'route': run.kind,
               'joints': [], 'flags': [], 'notes': []}
        if run.kind == 'wrap':
            seg.update(wrap_radius_m=_r(run.rho), wrap_radius_default=run.rho_default,
                       radius_from_axis_m=[_r(run.da), _r(run.db)], angle_home_deg=_r(math.degrees(run.theta_home), 2))
            if run.rho_clipped:
                seg['notes'].append(f'The wrap radius is limited to the nearer attachment\'s distance from the axis, '
                                    f'{run.rho * 1000:.1f} mm.')
            if run.rho < 1e-3:
                seg['notes'].append('An attachment lies on the joint axis: the cable would bend round the axis itself. '
                                    'Move the attachment off the axis.')
        elif run.kind == 'chord' and len(run.crossed) > 1:
            names = ', '.join(model.joint_names[j] for j, _s in run.crossed)
            seg['notes'].append(f'This run crosses {len(run.crossed)} joints ({names}): its length is the straight '
                                'chord, short of a cable laid along the arm when it folds. Add an attachment on each '
                                'link between them.')
        elif run.kind == 'rigid':
            seg['notes'].append('No moving joint between these links: the run keeps its length.')
        # sweeps
        evs, tags = [], []
        for j, _side in run.crossed:
            lo, hi = float(model.lower[j]), float(model.upper[j])
            jname = model.joint_names[j]
            jrec = {'name': jname, 'type': model.joint_types[j], 'range': [_r(lo, 5), _r(hi, 5)],
                    'range_from': 'the model file\'s limits'}
            if by_traj:
                lo = float(min(home[j], TQ[:, j].min()))
                hi = float(max(home[j], TQ[:, j].max()))
                jrec.update(range=[_r(lo, 5), _r(hi, 5)], range_from=f'the project\'s trajectories ({len(TQ)} poses, '
                                                                     'home included)')
            elif (model.joint_types[j] != 'prismatic' and math.isfinite(lo) and math.isfinite(hi)
                  and hi - lo > 2 * math.pi - 1e-3):
                seg['notes'].append(f'{jname} turns {math.degrees(hi - lo):.0f}° in the model file, and the cable is '
                                    'sized for all of it, so its loop at home is long. If the task uses less, set the '
                                    'joint ranges to the trajectories.')
            if not by_traj and not (math.isfinite(lo) and math.isfinite(hi)):
                lo, hi = home[j] - UNLIMITED_HALF_RANGE, home[j] + UNLIMITED_HALF_RANGE
                jrec.update(range=[_r(lo, 5), _r(hi, 5)], range_from='unlimited in the model file: one turn around '
                                                                     'home')
                seg['flags'].append(_flag('endless', f'{jname} has no limit in the model file: a cable across it '
                                                     'winds up without end. Limit its range or route the cable '
                                                     'through a slip ring.', joint=jname))
            Q = np.tile(home, (int(st['sweep_samples']), 1))
            Q[:, j] = np.linspace(lo, hi, int(st['sweep_samples']))
            T, anc, ax = CB.posed(model, Q)
            ev = run.evaluate(T, anc, ax, Q)
            ev['T'] = T
            L = ev['L']
            k_lo, k_hi = int(np.argmin(L)), int(np.argmax(L))
            jrec.update(length_min_m=_r(L[k_lo]), length_max_m=_r(L[k_hi]), at_min=_r(Q[k_lo, j], 5),
                        at_max=_r(Q[k_hi, j], 5), slack_m=_r(L[k_hi] - L[k_lo]),
                        slack_loop_m=_r(L[k_hi] - L[k_lo] + st['loop_m']),
                        unit='m' if model.joint_types[j] == 'prismatic' else 'rad')
            if run.kind == 'wrap':
                jrec['wrap_turns_max'] = _r(float(np.max(ev['turns'])), 3)
            seg['joints'].append(jrec)
            evs.append(ev)
            tags += [('sweep', jname, float(q)) for q in Q[:, j]]
        T0, anc0, ax0 = P0
        ev0 = run.evaluate(T0, anc0, ax0, Q0)
        ev0['T'] = T0
        evs.append(ev0)
        tags += [('home', '', 0.0)] + [('trajectory', nm, float(t)) for nm, t in zip(Tname, Tt)]
        ev = _concat(evs)
        L = ev['L']
        kmax = int(np.argmax(L))
        Lmax, Lmin = float(L.max()), float(L.min())
        sugg = math.ceil((Lmax + st['loop_m']) / ROUND_UP_M - 1e-9) * ROUND_UP_M
        laid = a['length_to_next_m']
        s = laid if laid is not None else sugg
        seg.update(length_home_m=_r(ev0['L'][0]), length_min_m=_r(Lmin), length_max_m=_r(Lmax),
                   longest_at=_where(tags[kmax]), suggested_m=_r(sugg), laid_m=_r(laid), used_m=_r(s),
                   slack_m=_r(Lmax - Lmin))
        if laid is None:
            seg['flags'].append(_flag('unset', f'Run {i + 1} ({a["link"]} → {b["link"]}) has no laid length yet: '
                                               f'the suggested {sugg:.3f} m is used. Apply sets it.'))
        elif laid < Lmax - 1e-9:
            seg['flags'].append(_flag('pull', f'Run {i + 1} ({a["link"]} → {b["link"]}) is laid {laid:.3f} m but '
                                              f'needs {Lmax:.3f} m {_where_text(tags[kmax])}: the cable pulls. '
                                              f'Lay {sugg:.3f} m (Apply).', at=_where(tags[kmax])))
        # bend: the route's own bend on the joint cylinder and the slack loop (a circular arc of the laid length over
        # the run's length), the smaller of the two, at every pose
        R = CB.bend_estimate(ev['bend_route'], L, s)
        loops = run.loop(ev, s, n)
        kb = int(np.argmin(R))
        rmin = float(R[kb])
        seg.update(bend_radius_min_m=_r(rmin, 5), bend_at=_where(tags[kb]))
        if out['allowed_bend_m'] is not None and rmin < out['allowed_bend_m']:
            seg['flags'].append(_flag('bend', f'Run {i + 1} bends to {rmin * 1000:.1f} mm {_where_text(tags[kb])}, '
                                              f'below the allowed {out["allowed_bend_m"] * 1000:.1f} mm (the stated '
                                              f'minimum × {st["bend_factor"]:g}). '
                                      + ('It bends round the joint at the wrap radius: move the attachments further '
                                         'from the joint axis (the wrap radius is at most the nearer one\'s '
                                         'distance from it).' if ev['bend_route'][kb] <= rmin + 1e-12 else
                                         'Its slack loop is tight there: lay less slack or move the attachments '
                                         'apart.'), at=_where(tags[kb])))
        # snag: home and the trajectories
        k0 = len(L) - len(Q0)
        ev_s = _take(ev, slice(k0, len(L)))
        Ps = loops[k0:]
        hits = CB.snag_hits(model, T0, Ps, ev_s['A'], ev_s['B'], st['seat_m'], grids)
        snagged = [k for k, h in enumerate(hits) if h]
        seg['snag_poses'] = len(snagged)
        seg['snag_poses_tested'] = len(hits)
        if snagged:
            k = snagged[0]
            tg = tags[k0 + k]
            links = sorted({h[0] for h in hits[k]})
            seg['snag'] = {'at': _where(tg), 'links': links, 'point': [_r(v) for v in hits[k][0][1]]}
            what = 'at home' if tg[0] == 'home' else f'on "{tg[1]}" at t = {tg[2]:.2f} s'
            seg['flags'].append(_flag('snag', f'Run {i + 1}\'s slack loop passes through {", ".join(links)} {what} '
                                              f'({len(snagged)} of {len(hits)} poses tested). Move an attachment or '
                                              'lay less slack.', at=_where(tg), links=links))
        seg['polyline_home'] = [[_r(v) for v in p] for p in simplify(loops[k0], 0.0005)]
        seg['status'] = _worst(*[f['severity'] for f in seg['flags']])
        out['segments'].append(seg)
    segs = out['segments']
    if segs:
        req = sum(sg['length_max_m'] for sg in segs)
        sug = sum(sg['suggested_m'] for sg in segs)
        used = sum(sg['used_m'] for sg in segs)
        out.update(required_m=_r(req), suggested_m=_r(sug), laid_m=_r(used))
        if c['length_m'] is not None:
            if c['length_m'] < req - 1e-9:
                out['flags'].append(_flag('pull', f'{c["name"]} is {c["length_m"]:.2f} m long but its runs need '
                                                  f'{req:.3f} m at their longest, before the run to the host: it '
                                                  'pulls. Choose a longer cable.'))
            elif c['length_m'] < used - 1e-9:
                out['flags'].append(_flag('unset', f'{c["name"]} is {c["length_m"]:.2f} m long: enough for its runs '
                                                   f'at their longest ({req:.3f} m) but not for the laid lengths '
                                                   f'with their loops ({used:.3f} m).'))
    if mb is None:
        out['flags'].append(_flag('unknown', f'The minimum bend radius of {c["name"]} is not stated: bends are not '
                                             'checked. Choose a library cable or enter the maker\'s figure.'))
    elif mb.get('kind') == 'static':
        out['notes'].append('The minimum bend radius is the maker\'s fixed (static) figure; a cable that flexes at a '
                            'joint may need more (the library\'s cables that state both give 2.5 times the fixed '
                            'figure for moving use): raise the bend factor if the maker says so.')
    out['status'] = _worst(*[f['severity'] for f in out['flags']],
                           *[sg['status'] for sg in segs])
    return out


def _where(tag) -> dict:
    kind, name, v = tag
    if kind == 'sweep':
        return {'pose': 'sweep', 'joint': name, 'value': _r(v, 5)}
    if kind == 'trajectory':
        return {'pose': 'trajectory', 'trajectory': name, 't_s': _r(v, 4)}
    return {'pose': 'home'}


def _where_text(tag) -> str:
    kind, name, v = tag
    if kind == 'sweep':
        return f'with {name} at {math.degrees(v):.1f}°'
    if kind == 'trajectory':
        return f'on "{name}" at t = {v:.2f} s'
    return 'at home'


def simplify(P, tol=0.004) -> np.ndarray:
    """The polyline P (n, 3) with the points within tol of the line through their neighbours dropped
    (Ramer–Douglas–Peucker): the drawn and exported path. Ported from Vantage core/harness.py."""
    P = np.asarray(P, float)
    if len(P) < 3:
        return P
    keep = np.zeros(len(P), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(P) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = P[i], P[j]
        ab = b - a
        L2 = float(ab @ ab)
        Qp = P[i + 1:j] - a
        if L2 < 1e-18:
            d = np.linalg.norm(Qp, axis=1)
        else:
            t = np.clip(Qp @ ab / L2, 0.0, 1.0)
            d = np.linalg.norm(Qp - t[:, None] * ab, axis=1)
        k = int(np.argmax(d))
        if d[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return P[keep]


# ====================================================================== suggestions
def suggested_cables(project) -> tuple[list, list]:
    """(cable records, notes): a suggested cable per enabled link-mounted sensor of the active variant."""
    from ..robot.model import RobotModelError, get_model
    out, notes = [], []
    for s in project.variant.sensors:
        if not s.enabled or not s.mount.on_link or project.robot(s.mount.robot_uid) is None:
            continue
        try:
            model = get_model(project, s.mount.robot_uid)
            atts, nn = CB.suggest_attachments(model, s)
        except (RobotModelError, KeyError) as e:
            notes.append(f'No route suggested for {s.name}: {e}')
            continue
        notes += nn
        if len(atts) >= 2:
            c = CB.new_cable(f'{s.name} cable (suggested)', s.uid)
            c['attachments'] = atts
            c['note'] = ('Suggested route: one attachment per link from the sensor to the base, at the middle of each '
                         'link off its surface. Check the points against the real robot.')
            out.append(c)
    return out, notes


def apply_suggested(cables: list, doc: dict) -> list:
    """The cable records with every run's laid length set to the study's suggested one (and, when the study ran on
    suggested cables, those cables added). Returns the new list; the input is not changed."""
    new = [CB.clean_cable(c) for c in cables]
    by = {c['uid']: c for c in new}
    if doc.get('suggested') and not new:
        new = [CB.clean_cable(c) for c in doc.get('suggested_cables') or []]
        for c in new:
            c['name'] = c['name'].replace(' (suggested)', '')
        by = {c['uid']: c for c in new}
    for rc in doc.get('cables') or []:
        c = by.get(rc.get('uid'))
        if c is None:
            continue
        for sg in rc.get('segments') or []:
            i = sg.get('index')
            if isinstance(i, int) and i < len(c['attachments']) - 1 and sg.get('suggested_m'):
                c['attachments'][i]['length_to_next_m'] = float(sg['suggested_m'])
    return new


def cable_polylines(project, q=None, points: int | None = None) -> list:
    """[{'uid', 'name', 'sensor_uid', 'points' (K, 3) world m, 'radius_m'}]: each cable of project.cables laid along
    its attachments at the joint state q ({robot uid: joint vector}; a robot not in it, or with a vector of the wrong
    length, at home), every run drawn as its slack loop at its laid length (length_to_next_m, else the route's length
    there plus loop_m, as the page draws it). radius_m is half the stated diameter, 0 when not stated. A cable whose
    attachments span robots or name a link the model does not have is left out (the study reports it)."""
    from ..robot.model import RobotModelError, get_model
    st = settings_of(project)
    n = int(points or st['points'])
    out = []
    for raw in getattr(project, 'cables', None) or []:
        c = CB.clean_cable(raw)
        atts = c['attachments']
        if len(atts) < 2 or len({a['robot_uid'] for a in atts}) != 1 or project.robot(atts[0]['robot_uid']) is None:
            continue
        ruid = atts[0]['robot_uid']
        try:
            model = get_model(project, ruid)
        except RobotModelError:
            continue
        if any(a['link'] not in model.link_names for a in atts):
            continue
        qv = (q or {}).get(ruid) if isinstance(q, dict) else None
        qv = model.home if qv is None or len(qv) != model.nq else np.asarray(qv, float)
        T, anc, ax = CB.posed(model, qv[None])
        parts = []
        for i in range(len(atts) - 1):
            run = _Run(model, atts[i], atts[i + 1], c['wrap_radius_m'])
            ev = run.evaluate(T, anc, ax, qv[None])
            s = atts[i]['length_to_next_m']
            if s is None:
                s = float(ev['L'][0]) + st['loop_m']
            P = run.loop(ev, float(s), n)[0]
            parts.append(P if not parts else P[1:])
        dia = c.get('diameter_m')
        out.append({'uid': c['uid'], 'name': c['name'], 'sensor_uid': c.get('sensor_uid') or '',
                    'points': np.concatenate(parts), 'radius_m': 0.5 * float(dia) if dia else 0.0})
    return out


# ====================================================================== the study
@study('cables', 'Cable flex across joints', quick=False, batch='cables', needs_robot=True)
def cables_study(project, ctx: Ctx) -> dict:
    """Cable length, slack, bend and snags across the joints, and the harness sheet."""
    st = settings_of(project)
    cables = [CB.clean_cable(c) for c in getattr(project, 'cables', []) or []]
    notes, suggested, sugg = [], False, []
    if not cables and st['suggest_when_empty']:
        sugg, nn = suggested_cables(project)
        notes += nn
        if sugg:
            suggested = True
            cables = sugg
            notes.insert(0, 'The project has no cables yet: these are suggested routes, one per link-mounted sensor '
                            '(not in the project until you Apply).')
    res = []
    for k, c in enumerate(cables):
        if ctx.cancelled():
            break
        res.append(analyse_cable(project, c, st, ctx, (k / max(1, len(cables)), (k + 1) / max(1, len(cables)))))
    ctx.report(1.0, 'done')
    doc = {'settings': st, 'suggested': suggested, 'notes': notes, 'cables': res,
           'suggested_cables': sugg if suggested else [], 'frame': 'world: the primary robot\'s base, REP-103, m',
           'summary': summarise(res, st), 'sheet': sheet_rows(res)}
    return doc


def summarise(res: list, st: dict) -> dict:
    n = len(res)
    sev = _worst(*[r['status'] for r in res]) if res else 'warning'
    clear = sum(r['status'] == 'good' for r in res)
    counts = {}
    for r in res:
        for f in r['flags'] + [f for sg in r['segments'] for f in sg['flags']]:
            counts[f['kind']] = counts.get(f['kind'], 0) + 1
    if not res:
        advice = ('There are no cables to check: add one for a sensor on a robot link (New cable, then Suggest '
                  'attachments).')
    else:
        worst, owner, segflag = None, None, False
        for r in res:
            seg_flags = [f for sg in r['segments'] for f in sg['flags']]
            for f in seg_flags + r['flags']:
                if worst is None or SEV.index(f['severity']) > SEV.index(worst['severity']):
                    worst, owner, segflag = f, r, f in seg_flags
        # a run's flag says "Run 4 …": name its cable, or the headline does not say which cable it means
        advice = (f'{owner["name"]}: {worst["message"]}' if segflag and owner.get('name') else worst['message']) \
            if worst is not None else (
            f'Every cable keeps its length, loop and bend radius over the joint ranges and trajectories '
            f'(bend factor {st["bend_factor"]:g}).')
    return {'cables': n, 'clear': clear, 'status': sev, 'word': WORDS[sev], 'flags': counts, 'advice': advice}


# ====================================================================== the harness sheet
CSV_COLUMNS = ('cable', 'sensor', 'type', 'library_id', 'diameter_mm', 'min_bend_radius_mm', 'min_bend_kind',
               'min_bend_source', 'allowed_bend_mm', 'cable_length_m', 'run', 'from_link', 'to_link', 'route',
               'joints', 'slack_loop_per_joint_mm', 'length_min_m', 'length_max_m', 'suggested_m', 'laid_m',
               'bend_radius_min_mm', 'snag_poses', 'connectors', 'strain_relief', 'status', 'flags')


def _mm(v, d=1):
    return '' if v is None else f'{v * 1000:.{d}f}'


def sheet_rows(res: list) -> list:
    """One row per run of each cable (a cable without runs gets one row): the harness sheet."""
    rows = []
    for r in res:
        mb = r['min_bend_radius'] or {}
        base = {'cable': r['name'], 'sensor': r['sensor'], 'type': r['type_label'], 'library_id': r['library_id'],
                'diameter_mm': _mm(r['diameter_m'], 2), 'min_bend_radius_mm': _mm(mb.get('value')),
                'min_bend_kind': mb.get('kind', ''), 'min_bend_source': mb.get('source', CB.NOT_STATED),
                'allowed_bend_mm': _mm(r['allowed_bend_m']),
                'cable_length_m': '' if r['length_m'] is None else f'{r["length_m"]:.3f}',
                'connectors': ' / '.join(r['connectors']), 'strain_relief': r['strain_relief']}
        cflags = [f['message'] for f in r['flags']]
        if not r['segments']:
            rows.append({**{k: '' for k in CSV_COLUMNS}, **base, 'status': r['status'], 'flags': ' | '.join(cflags)})
            continue
        for sg in r['segments']:
            rows.append({**base, 'run': sg['index'] + 1, 'from_link': sg['from']['link'],
                         'to_link': sg['to']['link'], 'route': sg['route'],
                         'joints': ', '.join(j['name'] for j in sg['joints']),
                         'slack_loop_per_joint_mm': '; '.join(f'{j["name"]} {_mm(j["slack_loop_m"])}'
                                                              for j in sg['joints']),
                         'length_min_m': f'{sg["length_min_m"]:.3f}', 'length_max_m': f'{sg["length_max_m"]:.3f}',
                         'suggested_m': f'{sg["suggested_m"]:.3f}',
                         'laid_m': '' if sg['laid_m'] is None else f'{sg["laid_m"]:.3f}',
                         'bend_radius_min_mm': _mm(sg['bend_radius_min_m']), 'snag_poses': sg['snag_poses'],
                         'status': sg['status'], 'flags': ' | '.join([f['message'] for f in sg['flags']] + cflags)})
            cflags = []
    return rows


def to_csv(doc: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(CSV_COLUMNS)
    for row in doc.get('sheet') or sheet_rows(doc.get('cables') or []):
        w.writerow([row.get(k, '') for k in CSV_COLUMNS])
    return buf.getvalue()


def to_html(doc: dict, project_name: str = '') -> str:
    """The harness sheet as a standalone HTML page (no scripts, no external files)."""
    e = html.escape
    s = doc.get('summary') or {}
    st = doc.get('settings') or {}
    head = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Harness sheet</title><style>'
            'body{font:14px/1.45 "IBM Plex Sans",system-ui,sans-serif;margin:24px;color:#1d1d1b;background:#fff}'
            'table{border-collapse:collapse;margin:8px 0 20px}th,td{border:1px solid #ccc;padding:4px 8px;'
            'vertical-align:top;text-align:left}td.n{font-family:"IBM Plex Mono",monospace;text-align:right}'
            '.good{color:#0a7d0a}.warning{color:#9a6a00}.serious{color:#b4532c}.critical{color:#b02a2a}'
            'small{color:#666}</style></head><body>')
    out = [head, f'<h1>Harness sheet{" · " + e(project_name) if project_name else ""}</h1>',
           f'<p class="{e(s.get("status", ""))}"><b>{s.get("clear", 0)} of {s.get("cables", 0)} cables clear '
           f'· {e(s.get("word", ""))}</b> — {e(s.get("advice", ""))}</p>']
    defaults = set(st.get('defaults') or [])
    out.append('<p><small>Settings: ' + ', '.join(
        f'{k} {st[k]:g}' + (' (default)' if k in defaults else '') for k in ('bend_factor', 'loop_m', 'seat_m',
                                                                              'max_step_deg') if k in st)
        + '. Lengths in metres, radii in millimetres.</small></p>')
    for n in doc.get('notes') or []:
        out.append(f'<p><small>{e(n)}</small></p>')
    for r in doc.get('cables') or []:
        mb = r.get('min_bend_radius') or {}
        clen = '—' if r['length_m'] is None else f'{r["length_m"]:.3f} m'
        out.append(f'<h2>{e(r["name"])} <span class="{e(r["status"])}">· {e(WORDS[r["status"]])}</span></h2>')
        out.append('<table><tr><th>Sensor</th><th>Type</th><th>Library entry</th><th>Diameter</th>'
                   '<th>Minimum bend radius</th><th>Cable length</th><th>Connectors</th><th>Strain relief</th></tr>'
                   f'<tr><td>{e(r["sensor"] or "—")}</td><td>{e(r["type_label"])}</td><td>{e(r["library_id"] or "—")}'
                   f'</td><td class="n">{_mm(r["diameter_m"], 2) or "—"} mm</td><td>'
                   + (f'{_mm(mb.get("value"))} mm ({e(mb.get("kind", ""))}) × {r["bend_factor"]:g} = '
                      f'{_mm(r["allowed_bend_m"])} mm<br><small>{e(mb.get("source", ""))}</small>' if mb else
                      'not stated')
                   + f'</td><td class="n">{clen}'
                   f'</td><td>{e(" / ".join(r["connectors"]) or "—")}</td><td>{e(r["strain_relief"] or "—")}</td>'
                   '</tr></table>')
        if r['segments']:
            out.append('<table><tr><th>Run</th><th>From</th><th>To</th><th>Joints crossed</th>'
                       '<th>Slack loop per joint (mm)</th><th>Length min–max (m)</th><th>Suggested (m)</th>'
                       '<th>Laid (m)</th><th>Smallest bend (mm)</th><th>Status</th></tr>')
            for sg in r['segments']:
                laid = '—' if sg['laid_m'] is None else f'{sg["laid_m"]:.3f}'
                loops = '<br>'.join(f'{e(j["name"])}: {_mm(j["slack_loop_m"])}' for j in sg['joints']) or '—'
                out.append(f'<tr><td class="n">{sg["index"] + 1}</td><td>{e(sg["from"]["link"])}</td>'
                           f'<td>{e(sg["to"]["link"])}</td><td>{e(", ".join(j["name"] for j in sg["joints"]) or "—")}'
                           f'</td><td class="n">{loops}</td><td class="n">{sg["length_min_m"]:.3f}–'
                           f'{sg["length_max_m"]:.3f}</td><td class="n">{sg["suggested_m"]:.3f}</td>'
                           f'<td class="n">{laid}</td>'
                           f'<td class="n">{_mm(sg["bend_radius_min_m"])}</td>'
                           f'<td class="{e(sg["status"])}">{e(WORDS[sg["status"]])}</td></tr>')
            out.append('</table>')
        msgs = [f['message'] for sg in r['segments'] for f in sg['flags']] + [f['message'] for f in r['flags']]
        if msgs:
            out.append('<ul>' + ''.join(f'<li>{e(m)}</li>' for m in msgs) + '</ul>')
        for n in r['notes'] + [n for sg in r['segments'] for n in sg['notes']]:
            out.append(f'<p><small>{e(n)}</small></p>')
    out.append('<p><small>Known limits: the cable is a stiffness-free curve whose slack loop bulges outwards from the '
               'joint, so the bend radius is an estimate; a run across several joints is a straight chord; snags are '
               'tested at sampled poses against the cable\'s own robot only; the run to the host is not '
               'included.</small></p></body></html>\n')
    return ''.join(out)


def files(doc: dict, project=None) -> list:
    """[(name, bytes)]: cables.json (with provenance when the project is given), cables.csv, cables.html."""
    from .. import provenance as PV
    d = {k: v for k, v in doc.items() if k not in ('cached', 'revision')}
    if project is not None:
        d = PV.stamp(d, project, output={'option': 'cables', 'name': 'cables.json', 'in_folder': True})
    return [('cables.json', (json.dumps(d, indent=1, ensure_ascii=False) + '\n').encode('utf-8')),
            ('cables.csv', to_csv(doc).encode('utf-8')),
            ('cables.html', to_html(doc, getattr(project, 'name', '')).encode('utf-8'))]


def export(doc: dict, project, out_dir) -> list[str]:
    """Write cables.json, cables.csv and cables.html into out_dir; returns their paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(doc, project):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths
