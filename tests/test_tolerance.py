"""Monte Carlo tolerances (core/tolerance.py, core/studies/tolerance.py), the perturbation export
(core/exporters/perturbations.py), their validation checks and the Tolerances page.

Adapted from Vantage tests/test_core.py test_tolerance_and_worst_single: the run returns one value per draw, the
sensitivity (here the rank-correlation contributions) is not empty, and a figure that is best at the nominal design
never beats it under perturbation."""
import json
import math
import os

import numpy as np
import pytest

from conftest import pump
from dexvantage.core import factors as F
from dexvantage.core import jobs
from dexvantage.core import metrics as M
from dexvantage.core import tolerance as TL
from dexvantage.core.project import Requirement
from dexvantage.core.robot.samples import sample_project
from dexvantage.core.studies import tolerance as TS
from dexvantage.validation.checks_tolerance import tol_rig

TEST_STUDY = '_t_tolfig'


@pytest.fixture
def figures():
    """A quick test study and three metrics read from the mounts: x of the first sensor (mm, linear), −(Δy)² of the
    second (best at its nominal y), and a figure the study does not state."""
    @jobs.study(TEST_STUDY, 'Test figures', quick=True)
    def _study(project, ctx):
        st = project.analysis.studies.get(TEST_STUDY) or {}
        a, b = project.variant.sensors[0], project.variant.sensors[1]
        dy = float(b.mount.pose.position[1]) - float(st.get('y0', 0.0))
        return {'x_mm': 1000 * float(a.mount.pose.position[0]), 'peak': -(1000 * dy) ** 2}

    M.metric('_t.x', 'Test x', 'mm', 'higher', study=TEST_STUDY)(lambda d: d['x_mm'])
    M.metric('_t.peak', 'Test peak', 'mm²', 'higher', study=TEST_STUDY)(lambda d: d['peak'])
    M.metric('_t.none', 'Test unstated', '', 'lower', study=TEST_STUDY)(lambda d: None)
    yield
    jobs.STUDIES.pop(TEST_STUDY, None)
    for k in ('_t.x', '_t.peak', '_t.none'):
        M.METRICS.pop(k, None)


def _sample(figures_on=True):
    p = sample_project('so100')
    b = p.variant.sensors[1]
    p.analysis.studies[TEST_STUDY] = {'y0': float(b.mount.pose.position[1])}
    return p


# ====================================================================== the model
def test_bounds_are_the_plus_minus_tolerance_and_scale():
    p = sample_project('so100')
    b = TL.bounds(p)
    assert len(b) == 2
    for row in b.values():
        assert row['source'] == TL.MOUNT and row['position_m'] == [[-0.001, 0.001]] * 3
        assert row['rotation_deg'] == [[-0.5, 0.5]] * 3
    b2 = TL.bounds(p, scale=2.0)
    assert next(iter(b2.values()))['position_m'][0] == [-0.002, 0.002]
    with pytest.raises(jobs.StudyError):
        TL.sources(p, distribution='triangular')


def test_draws_stay_within_bounds_and_normal_has_the_truncated_spread():
    p = sample_project('so100')
    rng = np.random.default_rng(3)
    for dist in ('uniform', 'normal'):
        srcs = TL.sources(p, distribution=dist)
        D = np.array([TL.draw_source(srcs[0], rng) for _ in range(4000)])
        assert np.all(np.abs(D[:, :3]) <= 0.001 + 1e-15) and np.all(np.abs(D[:, 3:]) <= 0.5 + 1e-12)
        sd = D[:, 0].std()
        if dist == 'uniform':
            assert abs(sd / (0.001 / math.sqrt(3)) - 1) < 0.05
        else:
            assert abs(sd / (0.0005 * math.sqrt(0.7737)) - 1) < 0.05


def test_perturb_gives_a_new_project_and_moves_locked_mounts():
    p = sample_project('so100')
    p.variant.sensors[0].locked = True
    before = [s.mount.pose.position.copy() for s in p.variant.sensors]
    q = TL.perturb(p, np.random.default_rng(1))
    assert q is not p
    for s, b0 in zip(p.variant.sensors, before):
        assert np.array_equal(s.mount.pose.position, b0)          # the input is never changed
    moved = [np.abs(s.mount.pose.position - b0).max() for s, b0 in zip(q.variant.sensors, before)]
    assert all(0 < m <= 0.001 + 1e-12 for m in moved)            # the locked one too
    with pytest.raises(F.FactorError):                            # factors still refuse it without allow_locked
        F.apply(p, {f'sensor:{p.variant.sensors[0].uid}.dx': 0.001})


def test_decompose_xyz_inverts_the_factor_turns():
    from dexvantage.core.frames import rot_x, rot_y, rot_z
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b, c = rng.uniform(-1.2, 1.2, 3)
        R = rot_x(a) @ rot_y(b) @ rot_z(c)
        assert np.allclose(TL.decompose_xyz(R), (a, b, c), atol=1e-12)
    w = rng.normal(size=3) * 0.3
    from scipy.spatial.transform import Rotation
    assert np.allclose(TL.exp_so3(w), Rotation.from_rotvec(w).as_matrix(), atol=1e-14)


def test_calibration_covariance_source_and_its_stale_fallback():
    p, uids, C = tol_rig()
    srcs = {s.uid: s for s in TL.sources(p)}
    assert srcs[uids['C']].kind == TL.CALIB and srcs[uids['A']].kind == TL.MOUNT
    b = TL.bounds(p)[uids['C']]
    assert b['source'] == TL.CALIB and '2 σ' in b['meaning']
    assert all(hi > 0 and lo == -hi for lo, hi in b['position_m'])
    q = F.apply(p, {f'sensor:{uids["C"]}.dx': 0.01})               # moved after its calibration
    s = {x.uid: x for x in TL.sources(q)}[uids['C']]
    assert s.kind == TL.MOUNT and 'moved' in s.note and s.position_mm == 5.0


def test_spearman_and_wilson():
    from scipy import stats
    rng = np.random.default_rng(2)
    x = np.round(rng.normal(size=60), 1)
    y = x + rng.normal(size=60)
    assert abs(TL.spearman(x, y) - stats.spearmanr(x, y).statistic) < 1e-12
    assert TL.spearman([1, 1, 1], [1, 2, 3]) is None
    lo, hi = TL.wilson(19, 20)
    assert 0.75 < lo < 0.95 < hi <= 1.0


def test_monte_carlo_distributions_shares_and_contributions(figures):
    p = _sample()
    a = p.variant.sensors[0]
    x0 = 1000 * float(a.mount.pose.position[0])
    p.requirements = [Requirement(name='x high', metric='_t.x', op='>=', threshold=x0 - 0.5),
                      Requirement(name='peak', metric='_t.peak', op='>=', threshold=-0.25, is_default=False)]
    res = TL.monte_carlo(p, ['_t.x', '_t.peak', '_t.none'], 200, 7)
    mx, mp = res['metrics']['_t.x'], res['metrics']['_t.peak']
    assert len(mx['values']) == 200 and mx['valid'] == 200 and res['done'] == 200
    assert abs(mx['nominal'] - x0) < 1e-9 and abs(mx['mean'] - x0) < 0.15
    assert mx['p5'] <= mx['p50'] <= mx['p95'] and mx['pessimistic']['quantile'] == 'P5'
    share = mx['requirements'][0]['share']
    assert share == sum(v >= x0 - 0.5 for v in mx['values']) / 200 and 0.6 < share < 0.85
    assert mx['contributions'][0]['sensor_uid'] == a.uid and mx['contributions'][0]['significant']
    assert mx['contributions'][0]['components']['dx']['rho'] > 0.9
    # a figure best at the nominal design never beats it, and its contribution shows through |Δ| (ρ_abs)
    assert max(mp['values']) <= mp['nominal'] + 1e-9
    top = mp['contributions'][0]
    assert top['sensor_uid'] == p.variant.sensors[1].uid and top['components']['dy']['rho_abs'] < -0.9
    assert res['metrics']['_t.none']['valid'] == 0 and res['metrics']['_t.none']['no_value'] == 200
    again = TL.monte_carlo(p, ['_t.x'], 200, 7)
    assert again['draws'] == res['draws'] and again['metrics']['_t.x']['values'] == mx['values']
    assert TL.monte_carlo(p, ['_t.x'], 200, 8)['draws'] != res['draws']
    json.dumps(res)                                              # JSON-able


def test_monte_carlo_stops_when_cancelled(figures):
    p = _sample()
    calls = {'n': 0}

    def cancel():
        calls['n'] += 1
        return calls['n'] > 5
    res = TL.monte_carlo(p, ['_t.x'], 50, 0, ctx=jobs.Ctx(cancel=cancel))
    assert res['cancelled'] and 0 < res['done'] < 50 and len(res['metrics']['_t.x']['values']) == res['done']


def test_tolerance_study_headline_status_and_suggestion(figures):
    p = _sample()
    a = p.variant.sensors[0]
    x0 = 1000 * float(a.mount.pose.position[0])
    p.requirements = [Requirement(name='x high', metric='_t.x', op='>=', threshold=x0 - 0.5)]
    p.analysis.studies['tolerance'] = {'n': 80, 'seed': 1}
    doc = jobs.run_study('tolerance', p)
    h = doc['headline']
    assert h['metric'] == '_t.x' and h['quantile'] == 'P5' and h['value'] == doc['metrics']['_t.x']['p5']
    sev, word = TS.status_of(h['share'])
    assert (h['status'], h['word']) == (sev, word) and h['status'] in ('serious', 'warning', 'critical')
    sg = doc['suggestion']
    assert sg['sensor_uid'] == a.uid and sg['field'] == 'position_mm' and sg['to'] == 0.5
    assert doc['metric_source'] == "the project's requirements" and 'n' not in doc['defaults']
    assert 'distribution' in doc['defaults']
    json.dumps(doc)
    # a locked mount is never suggested
    a.locked = True
    doc = jobs.run_study('tolerance', p)
    assert doc['suggestion'] is None or doc['suggestion']['sensor_uid'] != a.uid
    # settings and inputs it cannot work with: plain sentences
    p.analysis.studies['tolerance'] = {'n': 1}
    with pytest.raises(jobs.StudyError, match='draws'):
        jobs.run_study('tolerance', p)
    p.analysis.studies['tolerance'] = {'metrics': ['nope.nothing']}
    with pytest.raises(jobs.StudyError, match='no metric'):
        jobs.run_study('tolerance', p)
    p.analysis.studies['tolerance'] = {'distribution': 'cauchy'}
    with pytest.raises(jobs.StudyError, match='uniform or normal'):
        jobs.run_study('tolerance', p)
    q = _sample()
    q.variant.sensors[:] = []
    q.analysis.studies['tolerance'] = {'metrics': ['_t.x'], 'n': 4}
    with pytest.raises(jobs.StudyError):
        jobs.run_study('tolerance', q)


def test_without_requirements_it_takes_the_default_figures_and_says_so(figures):
    p = _sample()
    ids, why = TS.metric_ids(p, TS.settings_of(p))
    M.load_metrics()
    assert ids == [m for m in TS.DEFAULT_METRICS if m in M.METRICS] and 'default' in why
    saved = {m: M.METRICS.pop(m) for m in TS.DEFAULT_METRICS if m in M.METRICS}
    try:                                    # no default metric registered: every quick study's metrics
        p.analysis.studies['tolerance'] = {'n': 12}
        doc = jobs.run_study('tolerance', p)
        assert '_t.x' in doc['metrics'] and 'quick study' in doc['metric_source']
        assert doc['headline']['word'] in ('no requirement', 'no value')
    finally:
        M.METRICS.update(saved)


def test_real_coverage_metric_through_the_study():
    M.load_metrics()
    if 'coverage.covered' not in M.METRICS:
        pytest.skip('core/metric_defs.py does not define coverage.covered')
    p = sample_project('so100')
    p.requirements = [Requirement(name='seen', metric='coverage.covered', op='>=', threshold=90.0)]
    p.analysis.studies['tolerance'] = {'n': 4}
    doc = jobs.run_study('tolerance', p)
    m = doc['metrics']['coverage.covered']
    assert m['valid'] == 4 and m['nominal'] is not None and doc['headline']['metric'] == 'coverage.covered'


def test_batch_flag_does_not_take_reproduce_tolerance():
    import argparse

    from dexvantage import batch
    ap = argparse.ArgumentParser()
    batch.add_arguments(ap)
    a = ap.parse_args(['--tolerance-mc', 'out.json', '--tolerance', '1e-6'])
    assert a.study__tolerance == 'out.json' and a.tolerance == ['1e-6']


# ====================================================================== the export
def test_perturbations_exporter(tmp_path):
    from dexvantage.core import exporters as X
    p, uids, _C = tol_rig()
    man = X.export(p, str(tmp_path), ['perturbations'], options={'perturbations': {'n': 7, 'seed': 3}})
    row = man['formats']['perturbations']
    assert row['status'] == 'ok', row
    assert set(row['files']) <= set(man['files'])
    doc = json.load(open(tmp_path / 'perturbations' / 'perturbations.json'))
    assert doc['n'] == 7 and doc['seed'] == 3 and 'provenance' in doc and len(doc['draws']) == 7
    Z = np.load(tmp_path / 'perturbations' / 'perturbations.npz')
    assert Z['d'].shape == (7, 3, 6) and Z['T_world_optical'].shape == (7, 3, 4, 4)
    assert list(Z['sensor_uids']) == [s['uid'] for s in doc['sensors']]
    E = TL.perturbed_extrinsics(p, 7, 3)
    assert np.array_equal(E['d'], Z['d'])
    il = doc['draws'][0][0]['isaac_lab']
    assert il['convention'] == 'ros' and il['rot_wxyz'] == [il['rot_xyzw'][3]] + il['rot_xyzw'][:3]
    # the body delta reproduces the moved mount (core/sim/render extrinsic_delta convention)
    s0 = p.variant.sensor(uids['A'])
    q = TL.apply_draw(p, {s0.uid: Z['d'][2, 0]})
    assert np.allclose(s0.mount.pose.matrix() @ Z['delta_body'][2, 0], q.variant.sensor(s0.uid).mount.pose.matrix(),
                       atol=1e-12)
    gs = doc['draws'][0][0]['gs']
    assert np.allclose(np.array(gs['rotation']), Z['T_world_optical'][0, 0][:3, :3], atol=1e-8)
    bad = X.export(p, str(tmp_path / 'b'), ['perturbations'], options={'perturbations': {'n': 0}})
    assert bad['formats']['perturbations']['status'] == 'refused'


def test_perturbations_on_a_link_mounted_camera(tmp_path):
    from dexvantage.core.exporters import perturbations as PX
    p = sample_project('so100')
    res = PX.write(p, str(tmp_path), {'n': 3})
    A = res['arrays']
    assert np.isfinite(A['T_world_optical']).all() and any('Link-mounted' in n for n in res['notes'])


# ====================================================================== validation
def test_tolerance_checks_pass_quick():
    from dexvantage.validation import suite as VS
    VS.load_checks()
    ids = [s for s in VS.CHECKS if s.model == 'tolerance']
    assert {s.id for s in ids} == {'tolerance.linear_propagation', 'tolerance.sampler_ks',
                                   'tolerance.seed_reproducible', 'tolerance.spearman', 'tolerance.export_roundtrip'}
    ctx = VS.Ctx(None, None, None, VS.ValidationSettings(quick=True))
    try:
        for sp in ids:
            c = VS.run_one(sp, ctx)
            assert c.passed, (c.id, c.error, c.tolerance, c.exception)
            from dexvantage.validation.references import REFERENCES
            assert all(r in REFERENCES for r in sp.refs)
    finally:
        ctx.close()


# ====================================================================== the page
def test_tolerance_page(win, app, tmp_path, figures):
    from dexvantage.ui.viewport3d import LineDraw
    win.store.set_project(_sample())
    win.ctx.analysis.wait(20)
    p = win.store.project
    a = p.variant.sensors[0]
    x0 = 1000 * float(a.mount.pose.position[0])
    p.requirements = [Requirement(name='x high', metric='_t.x', op='>=', threshold=x0 - 0.5)]
    p.analysis.studies['tolerance'] = {'n': 40, 'seed': 2}
    win.open_page('tolerance')
    pump(app, 0.05)
    page = win.pages['tolerance']
    assert len(page._rows) == 2
    page.run()
    win.ctx.analysis.wait(120)
    pump(app, 0.1)
    assert page.doc is not None and page.head.figure.text() != '—'
    assert page.table.topLevelItemCount() == 1 and '±' in page.table.topLevelItem(0).text(2)
    assert page.b_apply.isEnabled() and page.chart.rows
    items = page.overlays(win.viewport)
    assert items and all(isinstance(i, LineDraw) for i in items)
    page.preview()
    assert page._preview and len(page.overlays(win.viewport)) == 2       # as it is, and tightened
    n0 = win.store.undo.count()
    page.apply()
    assert win.store.undo.count() == n0 + 1 and win.store.project.variant.sensors[0].mount.tolerance.position_mm == 0.5
    assert page.set_tolerance(a.uid, 'rotation_deg', 0.2)                # the editor: one undo step each
    assert win.store.undo.count() == n0 + 2
    pos = page._rows[a.uid][2]
    pos.committed.emit(0.3)
    assert win.store.undo.count() == n0 + 3 and win.store.project.variant.sensors[0].mount.tolerance.position_mm == 0.3
    win.store.undo.undo()
    win.store.undo.undo()
    win.store.undo.undo()
    assert win.store.project.variant.sensors[0].mount.tolerance.position_mm == 1.0
    man = page.export_perturbations(str(tmp_path / 'pert'), block=True)
    assert man and os.path.exists(tmp_path / 'pert' / 'perturbations' / 'perturbations.npz')
    assert page.export_perturbations(str(tmp_path / 'pert2')) is True      # the button's way: a worker thread
    for _ in range(200):
        pump(app, 0.02)
        if page.b_perturb.isEnabled():
            break
    assert os.path.exists(tmp_path / 'pert2' / 'export_manifest.json')
    path = page.export(str(tmp_path / 'tol.json'))
    assert path and 'provenance' in json.load(open(path))
    win.ctx.analysis.wait(60)


def test_headline_ties_go_to_the_smallest_margin_and_unanswered_requirements_say_so():
    def m(p5, sd, thr, share, reason=None):
        row = {'title': 'T', 'unit': '%', 'nominal': 90.0, 'mean': None if p5 is None else p5 + sd, 'std': sd,
               'pessimistic': {'quantile': 'P5', 'value': p5},
               'requirements': [{'uid': f'r{thr}', 'name': f'R{thr}', 'op': '>=', 'threshold': thr,
                                 'is_default': True, 'of': 10, 'share': share, 'share_ci95': [0.7, 1.0]}]}
        if reason:
            row['reason'] = reason
        return row
    doc = {'metrics': {'a': m(95.0, 1.0, 90.0, 1.0), 'b': m(91.0, 1.0, 90.0, 1.0)}}
    h = TS.headline(doc)
    assert h['metric'] == 'b' and (h['status'], h['word']) == ('good', 'holds')
    doc = {'metrics': {'a': m(None, None, 90.0, None, reason='It needs a depth sensor.')}}
    h = TS.headline(doc)
    assert h['word'] == 'no value' and h['status'] == 'critical' and 'depth sensor' in h['text']
