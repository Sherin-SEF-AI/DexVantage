"""Brackets and clearance (spec §7.7, §10 Brackets checks): a printable bracket for each sensor of the active variant
whose mount names a bracket type (Mount.bracket), meshed into a closed STL with its dimensions and print notes, and
checked for clearance to the robot's links over the joint range and to the cable paths.

    @jobs.study('brackets', 'Brackets and clearance', quick=False, batch='brackets')   -> --brackets OUT.json

The generator, the clearance check and their models are core/brackets.py (written up in docs/models/brackets.md).
Settings, project.analysis.studies['brackets'] (every number a labelled default unless the user sets it):

    sensors     {sensor uid: BracketSpec fields}: per-sensor options (pattern, side, standoff, wall, ...)
    samples     joint states sampled within the limits per robot, after dropping self-collisions (default 48)
    seed        their seed (default 0)
    margin_mm   the clearance below which a link counts as too close (default 2 mm)
    q           {robot uid: joint vector}: the pose on screen, added to the samples (passed by the page per run)
    mesh        False skips meshing (design and clearance only)

The result per sensor: the bracket kind and status (good, warning, critical) with the reason, the dimensions, the
mesh check (closed, consistently wound, Euler characteristic, volume), the holes with their cited sources, the
clearance per link and to cables, the share of the camera's image the bracket covers, print and assembly notes, and a
display mesh (vertex-clustered, millimetres in the bracket frame) with T_parent_bracket for drawing. Sensors without a
bracket type get a suggested one (core/brackets.suggest_kind), which the page's Apply sets in one undo step.

Known limits: as core/brackets.py: the camera body is a box of the stated size placed by an assumption; joint states
are sampled, not exhaustively searched (48 states find a collision that a sixth of the joint space shows with
probability 1 − (5/6)^48 > 99.98 %, a rarer one less surely); the clearance is a lower bound at mesh points, so a thin
triangle edge through the bracket can be missed; brackets are not load-tested; the display mesh is decimated (the
exported STL is not).
"""
from __future__ import annotations

import math

import numpy as np

from .. import brackets as B
from .. import jobs

STUDY_ID = 'brackets'
DEFAULTS = {'samples': 48, 'seed': 0, 'margin_mm': 2.0, 'mesh': True}
DISPLAY_CELL_MM = 1.5                  # vertex clustering cell of the display mesh (drawing only)
DISPLAY_MAX = 6000                     # triangles


def settings_of(project) -> dict:
    given = dict(project.analysis.studies.get(STUDY_ID) or {})
    out = dict(DEFAULTS)
    used_default = []
    for k, v in DEFAULTS.items():
        g = given.get(k)
        if g is None:
            used_default.append(k)
            continue
        if isinstance(v, bool):
            out[k] = bool(g)
            continue
        try:
            x = float(g)
        except (TypeError, ValueError):
            raise jobs.StudyError(f'The brackets setting {k} must be a number; it is {g!r}.') from None
        if not math.isfinite(x) or x < 0:
            raise jobs.StudyError(f'The brackets setting {k} must be a positive number; it is {g!r}.')
        out[k] = int(x) if isinstance(v, int) else x
    out['samples'] = min(max(int(out['samples']), 1), 2000)
    out['q'] = dict(given.get('q') or {}) if isinstance(given.get('q'), dict) else {}
    out['sensors'] = dict(given.get('sensors') or {}) if isinstance(given.get('sensors'), dict) else {}
    out['defaults'] = used_default
    return out


def display_mesh(V, F, cell: float = DISPLAY_CELL_MM, cap: int = DISPLAY_MAX) -> np.ndarray:
    """(K, 3, 3) triangles of V, F thinned by vertex clustering (for drawing only), the cell grown until at most cap."""
    V, F = np.asarray(V, float), np.asarray(F, np.int64)
    c = cell
    while True:
        key = np.floor(V / c).astype(np.int64)
        _u, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        k = int(inv.max()) + 1
        mean = np.stack([np.bincount(inv, V[:, j], minlength=k) for j in range(3)], axis=1)
        mean /= np.bincount(inv, minlength=k)[:, None]
        ids = inv[F]
        good = (ids[:, 0] != ids[:, 1]) & (ids[:, 1] != ids[:, 2]) & (ids[:, 0] != ids[:, 2])
        tri = mean[ids[good]]
        if len(tri) <= cap or c > 50:
            return tri
        c *= 1.4


def status_word(sev: str) -> str:
    return {'good': 'ready', 'warning': 'check', 'critical': 'collides'}.get(sev, sev)


@jobs.study(STUDY_ID, 'Brackets and clearance', quick=False, batch='brackets', needs_robot=False)
def run(project, ctx: jobs.Ctx) -> dict:
    st = settings_of(project)
    sensors = [s for s in project.variant.sensors if s.enabled]
    if not sensors:
        raise jobs.StudyError('There is no sensor in this rig: add one (Sensors) before designing its bracket.')
    rows, suggestion = [], []
    todo = []
    for s in sensors:
        spec = B.spec_for(project, s)
        if s.uid in st['sensors']:
            d = dict(st['sensors'][s.uid])
            d['kind'] = spec.kind
            if not d.get('pattern'):
                d['pattern'] = spec.pattern
            spec = B.BracketSpec.from_dict(d)
        if not (s.mount.bracket or '').strip():
            kind, why = B.suggest_kind(project, s)
            if kind:
                suggestion.append({'sensor_uid': s.uid, 'sensor': s.name, 'kind': kind, 'label': B.TYPES[kind],
                                   'why': why})
                msg = f'No bracket type is chosen: {B.lc(B.TYPES[kind])} suggested ({why}).'
            else:
                msg = f'No bracket type is chosen: {why}.'
            rows.append({'uid': s.uid, 'name': s.name, 'kind': '', 'label': 'Not chosen', 'status': 'unset',
                         'message': msg, 'flags': [], 'notes': []})
            continue
        todo.append((s, spec))
    for i, (s, spec) in enumerate(todo):
        if ctx.cancelled():
            break
        ctx.report(i / max(len(todo), 1), s.name)

        def prog(f, i=i, name=s.name):
            ctx.report((i + 0.6 * f) / max(len(todo), 1), name)
        row = B.build(project, s, spec, mesh_it=st['mesh'], samples=st['samples'], seed=st['seed'],
                      q_now=st['q'], margin_mm=st['margin_mm'], progress=prog, cancel=ctx.cancelled)
        mesh = row.pop('_mesh', None)
        row.pop('_design', None)
        if mesh is not None:
            tri = display_mesh(*mesh)
            row['display'] = {'tris_mm': np.round(tri, 2).reshape(-1).tolist(), 'triangles': int(len(tri)),
                              'cell_mm': DISPLAY_CELL_MM, 'note': 'thinned for drawing; the STL export is not'}
        rows.append(row)
    ctx.report(1.0, '')
    built = [r for r in rows if r['status'] in ('good', 'warning', 'critical')]
    ok = [r for r in built if r['status'] == 'good']
    if any(r['status'] == 'critical' for r in rows):
        sev = 'critical'
    elif any(r['status'] in ('warning', 'unset') for r in rows):
        sev = 'warning'
    else:
        sev = 'good'
    if not built and any(r['status'] == 'unset' for r in rows):
        text = 'No sensor has a bracket type yet: ' + ('Apply sets the suggested ones.' if suggestion else
                                                       'choose one per sensor.')
    else:
        worst = next((r for r in rows if r['status'] == 'critical'), None) or \
            next((r for r in rows if r['status'] == 'warning'), None)
        text = f'{worst["name"]}: {worst["message"]}' if worst else 'Every bracket is closed, printable and clear.'
    return {
        'sensors': B.clean(rows),
        'rig': {'brackets': len(built), 'ready': len(ok), 'status': sev, 'status_word': status_word(sev),
                'text': text, 'unset': sum(r['status'] == 'unset' for r in rows)},
        'suggestion': suggestion or None,
        'settings': {k: st[k] for k in ('samples', 'seed', 'margin_mm', 'mesh')},
        'defaults': st['defaults'],
        'pose': {'q': {k: [float(x) for x in np.asarray(v, float).reshape(-1)] for k, v in st['q'].items()},
                 'note': 'the pose on screen' if st['q'] else 'the robots\' home pose (no pose given)'},
        'sources': [B.source(k) for k in B.SOURCES],
        'material': B.MATERIAL,
    }
