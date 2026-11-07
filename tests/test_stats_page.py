"""The Statistics page (ui/pages/statistics.py): success counts entered in the table (one undo step), the headline
with its status word, which test and why, the power calculator with its curve, Apply (one undo step) and Preview,
the experiment and variant sources, Export with provenance, and the UI-thread budget."""
import json
import time

from conftest import pump


def _wait(win, app, timeout=120):
    win.ctx.analysis.wait(timeout)
    pump(app, 0.1)


def _page(win, app):
    assert 'statistics' in win.pages
    page = win.pages['statistics']
    win.open_page('statistics')
    pump(app, 0.05)
    return page


def _wait_power(page, app, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        pump(app, 0.05)
        if page._power is not None or page.pw_answer.text():
            return
    raise AssertionError('the power calculation did not finish')


def test_rates_table_headline_apply_preview_export(app, win, tmp_path):
    page = _page(win, app)
    assert page.head.status.text.text() == 'not run yet'
    u = win.store.undo
    n0 = u.index()
    t0 = time.perf_counter()
    assert page.set_rates([('ACT', 14, 20), ('Diffusion', 19, 20)])
    assert time.perf_counter() - t0 < 0.1                       # the study runs on a worker thread
    assert u.index() == n0 + 1
    _wait(win, app)
    st = win.store.project.analysis.studies['stats']
    assert st['source'] == 'rates' and [r['successes'] for r in st['rates']] == [14, 19]
    assert page.head.status.text.text() == 'no clear difference'
    assert page.table.topLevelItemCount() == 1
    it = page.table.topLevelItem(0)
    assert it.text(5) == 'Fisher\'s exact test' and 'pp' in it.text(3) and it.text(8) == 'no clear difference'
    assert 'Fisher' in page.why.text() and 'Holm' in page.why.text()
    assert 'independent trial' in page.assumptions.text() and 'Sequential' in page.assumptions.text()
    assert page.b_apply.isEnabled() and page.b_preview.isEnabled()
    # Preview: the plan shown in the power calculator, not stored
    page.preview()
    _wait_power(page, app)
    page._power = None
    page._compute_power(block=True)
    assert page.pw_answer.text().startswith('Preview')
    assert win.store.project.analysis.studies['stats'].get('power') is None
    # Apply: one undo step, the plan stored
    n1 = u.index()
    sg = page.apply()
    assert sg and u.index() == n1 + 1
    pw = win.store.project.analysis.studies['stats']['power']
    assert pw['n'] == sg['n'] and pw['p1'] == 0.7 and pw['p2'] == 0.95
    page._compute_power(block=True)
    assert f'The plan of {sg["n"]}' in page.pw_answer.text() and page.pw_curve.series
    u.undo()
    assert win.store.project.analysis.studies['stats'].get('power') is None
    u.redo()
    # a bad row is refused in words, nothing stored
    n2 = u.index()
    page.set_rates([('A', 30, 20), ('B', 1, 2)])
    assert 'between 0 and the trials' in page.rates_note.text() and u.index() == n2
    # Export: the result with provenance and the power calculator as shown
    page.set_rates([('ACT', 14, 20), ('Diffusion', 19, 20)])
    _wait(win, app)
    page._compute_power(block=True)
    path = page.export(str(tmp_path / 'stats.json'))
    doc = json.loads(open(path, encoding='utf-8').read())
    assert doc['study'] == 'stats' and 'provenance' in doc and doc['power']['n_per_group'] > 0
    assert doc['comparisons'][0]['test'] == 'fisher'


def test_power_calculator_edits_are_undo_steps(app, win):
    page = _page(win, app)
    u = win.store.undo
    n0 = u.index()
    page._set_power({'kind': 'paired_t'}, 'paired')
    page._set_power({'d': 0.4, 'n': 30}, 'd')
    assert u.index() == n0 + 2
    page._compute_power(block=True)
    assert 'pairs' in page.pw_answer.text() and page._power['kind'] == 'paired_t'
    assert page.pw_curve.series and page.pw_curve.markers
    t0 = time.perf_counter()
    page._fill()
    assert time.perf_counter() - t0 < 0.1
    page._set_power({'kind': 'demonstrations', 'icc': 0.1, 'per_operator': 11, 'attrition': 0.2}, 'demos')
    page._compute_power(block=True)
    assert page._power['design_effect'] > 1.9 and 'Kish' in page.pw_assume.text()
    assert page.pw_icc.isEnabled() and not page.pw_p1.isEnabled()


def test_experiment_source_from_the_last_result(app, win):
    from test_stats import _factorial_doc
    page = _page(win, app)
    win.ctx.analysis.results['experiments'] = _factorial_doc()
    page._set({'source': 'experiments'}, 'source')
    page._fill()
    assert page.ex_factor.count() == 2 and 'Full factorial' in page.ex_note.text()
    page.ex_factor.setCurrentIndex(page.ex_factor.findData('sensor:Cam.model'))
    page._set_experiment({'mode': 'factor', 'factor': 'sensor:Cam.model'}, 'factor')
    page.run()
    _wait(win, app)
    doc = page.doc
    assert doc and doc['source'] == 'experiments' and doc['paired'] and 'other factors' in doc['pairing']
    assert doc['settings']['experiment']['result'] == '(the experiment result in memory)'
    it = page.table.topLevelItem(0)
    # three pairs: the smallest two-sided sign-flip p is 2/8, so even a constant difference is not clear
    assert page.table.topLevelItemCount() == 1 and it.text(8) == 'no clear difference' and it.text(6) == '0.250'
    assert it.text(4) == '—'                     # every pair differs by exactly 2: d_z is undefined, said


def test_variants_source_runs_the_monte_carlo(app, win):
    from test_stats import _two_variant_project
    page = _page(win, app)
    p = _two_variant_project()
    p.variants[1].sensors[1].mount.pose.position[2] += 0.06
    win.store.set_project(p)
    _wait(win, app)
    page._set({'source': 'variants', 'variants': {'b': p.variants[1].uid, 'n': 4,
                                                  'metrics': ['coverage.covered']}}, 'variants')
    page._fill()
    assert page.var_b.currentData() == p.variants[1].uid and page.stack.currentIndex() == 1
    page.run()
    _wait(win, app, 180)
    doc = page.doc
    assert doc and doc['source'] == 'variants' and 'paired by seed' in doc['pairing']
    assert page.table.topLevelItemCount() == 1 and 'permutation' in page.table.topLevelItem(0).text(5)
    assert page.head.status.text.text() in ('clear difference', 'no clear difference')


def test_tracker_source_says_why_without_a_saved_project(app, win):
    page = _page(win, app)
    page._set({'source': 'tracker'}, 'source')
    page._fill()
    assert 'not saved' in page.tr_note.text() or 'run store' in page.tr_note.text()
