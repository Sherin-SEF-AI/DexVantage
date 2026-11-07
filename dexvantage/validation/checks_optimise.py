"""Checks of the optimiser and the design agent (core/optimise, spec §7.11) against independent references.

- optimise.zdt1_igd, optimise.zdt2_igd: NSGA-II (population 100, 25 000 evaluations, 30 genes, SBX η 15, polynomial
  mutation η 20, the defaults) on ZDT1 and ZDT2 against their analytic Pareto fronts (f2 = 1 − √f1 and 1 − f1², g = 1;
  Zitzler, Deb and Thiele 2000), 1 000 points evenly spaced in f1. Reported: the worst inverted generational distance
  (IGD) over three seeds (one in a quick run). Tolerance 0.01: 1 % of the fronts' extent in f1 and f2 (both span 0 to
  1). For scale the detail gives the IGD of an ideal 100-point set spaced evenly along the analytic front (about
  0.0037 for ZDT1): the tolerance is about 2.7 times that floor, so it allows crowding distance's uneven spacing but
  fails a front offset by more than 0.01 or with a gap of more than a few per cent of its length.
- optimise.constrained: NSGA-II with constrained domination on Deb's CONSTR problem (two constraints that cut the
  unconstrained front; Deb et al. 2002, Table V) against its analytic front, worked out from the constraints and
  written out in core/optimise/nsga2.constr_front (f2 = 1/f1 for f1 ≥ 2/3, (7 − 9 f1)/f1 for 7/18 ≤ f1 < 2/3).
  Reported: the worst IGD over the seeds (population 100, 10 000 evaluations); every point of the final front must
  satisfy both constraints, re-evaluated here from the genes (not from the optimiser's violation figures). Tolerance
  0.05 in objective units: f2 spans 1 to 9 on this front, so 0.05 is 0.6 % of its range; the detail gives the even
  100-point floor (about 0.020, the front being long in f2).
- optimise.locks: the rig optimiser on random rigs (2 to 5 sensors on static frames, each locked with probability
  0.5, moves up to 0.5 m and 45°, devices swapped on every other rig, and on every third rig factor lists that name the
  locked sensors) with a cheap test metric (distance from the nearest camera to a point, camera count, height) that
  this check registers and removes. Every project the search builds is caught as core.factors.apply returns it and
  its locked sensors compared, field by field, with the original's. Reported: the number of designs that changed a
  locked sensor. Tolerance 0.
- optimise.seed_reproducible: two runs with the same seed must give identical results: NSGA-II on ZDT1 (the archive
  of every evaluation) and the rig optimiser on a test rig (the options' factor values and figures); a different seed
  must give a different archive. Reported: the largest absolute difference between the same-seed runs. Tolerance 0.
- optimise.figures_reproduce: the optimiser on the UR5e sample (contact visibility, depth-valid coverage, camera
  count; 12 evaluations): every option's figures against the studies run again, without the cache, on the option
  applied to the project by core.factors (at the same screening settings). Reported: the largest difference. Tolerance
  10⁻⁹ (the same deterministic studies; any difference means the figures shown are not the design's).
"""
from __future__ import annotations

import numpy as np

from .suite import Outcome, check, figure, series

REFS_NSGA = ('deb2002_nsga2', 'deb_agrawal1995_sbx', 'deb_goyal1996_mutation')


def _even_floor(ref: np.ndarray, n: int = 100) -> float:
    """The IGD of n points spaced evenly by arc length along a dense reference front (the best an n-point set can
    do)."""
    from ..core.optimise import nsga2 as N
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(ref, axis=0), axis=1))]
    t = np.linspace(0.0, d[-1], n)
    pts = np.stack([np.interp(t, d, ref[:, k]) for k in range(ref.shape[1])], axis=1)
    return N.igd(pts, ref)


def _zdt(ctx, kind):
    from ..core.optimise import nsga2 as N
    ref = N.zdt_front(kind)
    seeds = (0,) if ctx.quick else (0, 1, 2)
    igds, fronts = [], []
    for s in seeds:
        r = N.run(N.benchmark_problem(kind), pop=100, budget=25000, seed=int(ctx.rng(f'optimise.{kind}', s)
                                                                             .integers(0, 2 ** 31 - 1)))
        F = r.F[r.front()]
        igds.append(N.igd(F, ref))
        fronts.append(F)
    F = fronts[int(np.argmax(igds))]
    o = np.argsort(F[:, 0])
    fig = figure(f'NSGA-II on {kind.upper()} against the analytic front', 'f1', 'f2',
                 [series('analytic front', ref[::10, 0], ref[::10, 1]),
                  series('NSGA-II front (worst seed)', F[o, 0], F[o, 1], kind='points')])
    return Outcome(max(igds), {'igd_per_seed': [round(x, 6) for x in igds], 'front_points': [len(f) for f in fronts],
                               'evaluations': 25000, 'population': 100, 'genes': 30,
                               'even_100_point_floor': round(_even_floor(ref), 6)}, fig)


@check('optimise.zdt1_igd', 'optimise', 'NSGA-II on ZDT1 against the analytic front',
       'inverted generational distance of the final front (worst seed)', 'the analytic ZDT1 front f2 = 1 − √f1',
       0.01, 'IGD (objective units)', refs=REFS_NSGA + ('zitzler2000_zdt', 'coello2004_igd'))
def zdt1(ctx):
    return _zdt(ctx, 'zdt1')


@check('optimise.zdt2_igd', 'optimise', 'NSGA-II on ZDT2 against the analytic front',
       'inverted generational distance of the final front (worst seed)', 'the analytic ZDT2 front f2 = 1 − f1²',
       0.01, 'IGD (objective units)', refs=REFS_NSGA + ('zitzler2000_zdt', 'coello2004_igd'))
def zdt2(ctx):
    return _zdt(ctx, 'zdt2')


@check('optimise.constrained', 'optimise', 'Constrained domination on CONSTR against its analytic front',
       'inverted generational distance of the final front (worst seed); every point feasible',
       'the CONSTR front worked out from its constraints', 0.05, 'IGD (objective units)',
       refs=REFS_NSGA + ('deb2001_moo', 'coello2004_igd'))
def constrained(ctx):
    from ..core.optimise import nsga2 as N
    ref = N.constr_front()
    seeds = (0,) if ctx.quick else (0, 1, 2)
    igds, worst_g, F_keep = [], 0.0, None
    for s in seeds:
        r = N.run(N.benchmark_problem('constr'), pop=100, budget=10000,
                  seed=int(ctx.rng('optimise.constrained', s).integers(0, 2 ** 31 - 1)))
        fr = r.front()
        X = r.X[fr]
        # the constraints again, from the genes: x2 + 9 x1 >= 6 and -x2 + 9 x1 >= 1
        g = np.maximum(0.0, np.maximum(6.0 - X[:, 1] - 9.0 * X[:, 0], 1.0 + X[:, 1] - 9.0 * X[:, 0]))
        worst_g = max(worst_g, float(g.max()))
        F = np.stack([X[:, 0], (1.0 + X[:, 1]) / X[:, 0]], axis=1)          # objectives again, from the genes
        igds.append(N.igd(F, ref))
        F_keep = F if F_keep is None or igds[-1] >= max(igds[:-1] or [0]) else F_keep
    o = np.argsort(F_keep[:, 0])
    fig = figure('NSGA-II on CONSTR against its analytic front', 'f1', 'f2',
                 [series('analytic front', ref[::10, 0], ref[::10, 1]),
                  series('NSGA-II front (worst seed)', F_keep[o, 0], F_keep[o, 1], kind='points')])
    return Outcome(max(igds), {'igd_per_seed': [round(x, 6) for x in igds], 'largest_violation': worst_g,
                               'even_100_point_floor': round(_even_floor(ref), 6), 'evaluations': 10000},
                   fig, ok=worst_g <= 0.0)


# ====================================================================== the rig searches, on a test metric
def _test_metric():
    """Register a cheap study and three metrics on it (removed by _drop_test_metric)."""
    from ..core import jobs
    from ..core import metrics as M
    from ..core.optimise import objectives as OB
    from ..core.scenegeom import sensor_world_pose
    target = np.array([0.5, 0.0, 0.3])

    @jobs.study('_check_optimise_geo', 'Optimiser check geometry', quick=False)
    def _geo(project, ctx):
        cams = [s for s in project.variant.sensors if s.enabled and s.kind in ('camera', 'depth')]
        P = [sensor_world_pose(project, s).position for s in cams]
        return {'dist': min(float(np.linalg.norm(p - target)) for p in P) if P else None, 'cams': len(cams),
                'height': max([float(p[2]) for p in P] or [0.0])}
    OB.ensure_metrics()
    M.metric('_check.dist', 'Distance', 'm', 'lower', study='_check_optimise_geo')(lambda d: d['dist'])
    M.metric('_check.cams', 'Cameras', 'count', 'lower', study='_check_optimise_geo')(lambda d: d['cams'])
    M.metric('_check.height', 'Height', 'm', 'higher', study='_check_optimise_geo')(lambda d: d['height'])


def _drop_test_metric():
    from ..core import jobs
    from ..core import metrics as M
    jobs.STUDIES.pop('_check_optimise_geo', None)
    for k in ('_check.dist', '_check.cams', '_check.height'):
        M.METRICS.pop(k, None)


def _random_rig(rng, k):
    from ..core.frames import Pose
    from ..core.project import Mount, Project, Sensor
    p = Project(f'Locks {k}')
    for i in range(int(rng.integers(2, 6))):
        p.variant.sensors.append(Sensor(name=f'S{i}', locked=bool(rng.random() < 0.5),
                                        mount=Mount(str(rng.choice(['cage', 'world', 'tripod'])), '',
                                                    Pose(rng.uniform(-1, 1, 3) + [0, 0, 1.0]))))
    if all(s.locked for s in p.variant.sensors):
        p.variant.sensors[0].locked = False
    return p


@check('optimise.locks', 'optimise', 'Locked sensors are never changed by a design',
       'designs whose locked sensors differ from the originals', 'the locked sensors\' own fields, compared',
       0, 'designs', refs=('deb2002_nsga2',))
def locks(ctx):
    from ..core import factors as FA
    from ..core.optimise import rig as R
    _test_metric()
    rng = ctx.rng('optimise.locks')
    bad, designs, cases = 0, 0, 6 if ctx.quick else 20
    real = FA.apply
    try:
        for k in range(cases):
            p = _random_rig(rng, k)
            locked = {s.uid: s.to_dict() for s in p.variant.sensors if s.locked}
            factors = []
            if k % 3 == 0:
                factors = [{'path': f'sensor:{u}.x', 'lo': -1, 'hi': 1} for u in locked] + \
                    [{'path': f'sensor:{s.uid}.z', 'lo': 0, 'hi': 2} for s in p.variant.sensors if not s.locked]
            st = R.settings_of(p, {'objectives': ['_check.dist', '_check.cams', '_check.height'], 'budget': 24,
                                   'population': 6, 'move_m': 0.5, 'turn_deg': 45, 'swap_models': bool(k % 2),
                                   'screen': {}, 'seed': k, 'factors': factors})
            seen = []

            def spy(project, values, *a, _seen=seen, **kw):
                out = real(project, values, *a, **kw)
                _seen.append({s.uid: s.to_dict() for s in out.variant.sensors if s.locked})
                return out
            FA.apply = spy
            try:
                R.optimise(p, st)
            finally:
                FA.apply = real
            designs += len(seen)
            bad += sum(x != locked for x in seen)
    finally:
        FA.apply = real
        _drop_test_metric()
    return Outcome(float(bad), {'rigs': cases, 'designs_built': designs})


@check('optimise.seed_reproducible', 'optimise', 'The same seed gives the same search',
       'largest difference between two runs with the same seed', 'the first run', 0, 'objective units',
       refs=('deb2002_nsga2',))
def seed_reproducible(ctx):
    from ..core.frames import Pose
    from ..core.optimise import nsga2 as N
    from ..core.optimise import rig as R
    from ..core.project import Mount, Project, Sensor
    p = N.benchmark_problem('zdt1', 10)
    a, b = N.run(p, pop=20, budget=600, seed=11), N.run(p, pop=20, budget=600, seed=11)
    c = N.run(p, pop=20, budget=600, seed=12)
    err = float(np.abs(a.archive_F - b.archive_F).max())
    differs = not np.array_equal(a.archive_X, c.archive_X)
    _test_metric()
    try:
        q = Project('Seeds')
        q.variant.sensors = [Sensor(name='A', mount=Mount('cage', '', Pose(np.array([1.0, 0.0, 1.0])))),
                             Sensor(name='B', mount=Mount('world', '', Pose(np.array([0.0, 1.0, 1.0]))))]
        st = R.settings_of(q, {'objectives': ['_check.dist', '_check.cams'], 'budget': 24, 'population': 6,
                               'move_m': 0.3, 'screen': {}, 'seed': 4})
        r1, r2 = R.optimise(q, st), R.optimise(q, st)
        same = [o['values'] for o in r1['options']] == [o['values'] for o in r2['options']] and \
            [o['figures'] for o in r1['options']] == [o['figures'] for o in r2['options']]
    finally:
        _drop_test_metric()
    return Outcome(err if same else float('inf'), {'nsga2_archive_rows': len(a.archive_F),
                                                    'rig_options_identical': same,
                                                    'other_seed_differs': differs}, ok=differs)


@check('optimise.figures_reproduce', 'optimise', 'Each option\'s figures are its design\'s',
       'largest difference between an option\'s figures and the studies run again on it',
       'the studies run again, without the cache, on the option applied', 1e-9, 'metric units',
       refs=('deb2002_nsga2',), cost='study')
def figures_reproduce(ctx):
    from ..core import metrics as M
    from ..core.optimise import objectives as OB
    from ..core.optimise import rig as R
    from ..core.optimise import search as S
    from ..core.robot.samples import sample_project
    OB.ensure_metrics()
    p = sample_project('ur5e')
    ids = ['grasp.contact_visibility', 'depth.valid', 'rig.camera_count']
    st = R.settings_of(p, {'objectives': ids, 'budget': 8 if ctx.quick else 12, 'population': 4, 'seed': 2})
    doc = R.optimise(p, st)
    worst, rows = 0.0, []
    for o in doc['options']:
        q = S.Evaluator(ids, [], st['screen']).screened(R.option_project(p, doc, o['label']))
        got = M.evaluate(q, ids)                               # no cache
        for m in ids:
            a, b = o['figures'].get(m), got[m]['value']
            d = 0.0 if a is None and b is None else (float('inf') if a is None or b is None else abs(a - b))
            worst = max(worst, d)
        rows.append({'label': o['label'], 'figures': o['figures']})
    return Outcome(worst, {'options': rows, 'evaluations': doc['search']['evaluations']}, ok=bool(rows))
