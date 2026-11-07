"""The Runs page (ui/pages/runs.py): Apply turns tracking on in one undo step; long studies the window finishes are
logged off the UI thread (quick ones only when asked, a cached result not twice, a refusal as a failed run); the
table and its filters; compare with deltas; re-run; tags; export; and the UI-thread budget."""
import os
import time

from conftest import pump
from dexvantage.core import tracking as TR
from dexvantage.core.project import Project, blank_project


def _saved(tmp_path, sample=None):
    if sample:
        from dexvantage.core.robot.samples import sample_project
        p = sample_project(sample)
    else:
        p = blank_project('Runs cell')
    path = str(tmp_path / 'cell.dexv')
    p.save(path)
    return Project.load(path)


def _settle(win, app, page, rounds=4):
    for _ in range(rounds):
        win.ctx.analysis.wait(120)
        page.wait(120)
        pump(app, 0.05)


def _word(page):
    return page.head.state['word']


def test_runs_page_logs_compares_reruns_and_exports(app, win, tmp_path):
    win.store.set_project(_saved(tmp_path, 'so100'))
    page = win.pages['runs']
    _settle(win, app, page)
    win.open_page('runs')
    page.refresh()
    _settle(win, app, page)
    assert _word(page) == 'tracking off' and page.b_apply.isEnabled()
    u = win.store.undo
    n0 = u.index()
    page.apply()
    assert u.index() == n0 + 1 and TR.enabled(win.store.project) and not page.b_apply.isEnabled()
    u.undo()
    assert not TR.enabled(win.store.project) and page.c_on.isChecked() is False
    u.redo()
    assert TR.enabled(win.store.project) and page.c_on.isChecked()
    _settle(win, app, page)
    assert _word(page) == 'no runs yet'

    win.ctx.analysis.run('rig')                                       # a long study: logged
    _settle(win, app, page)
    assert page.table.topLevelItemCount() == 1 and _word(page) == 'tracking on'
    win.ctx.analysis.run('rig')                                       # from the cache: not logged twice
    _settle(win, app, page)
    assert page.table.topLevelItemCount() == 1
    win.ctx.analysis.run('bandwidth')                                 # a quick study: not logged by default
    _settle(win, app, page)
    assert page.table.topLevelItemCount() == 1
    assert page.set_setting('quick', True) and not page.set_setting('quick', True)
    win.ctx.analysis.run('bandwidth')
    _settle(win, app, page)
    assert page.table.topLevelItemCount() == 2

    sid = win.store.project.variant.sensors[0].uid                    # one camera fewer, then the rig again
    win.store.edit('variant', lambda p: setattr(p.variant, 'sensors', [s for s in p.variant.sensors
                                                                       if s.uid != sid]), 'Remove a camera')
    _settle(win, app, page)
    win.ctx.analysis.run('rig')
    _settle(win, app, page)
    st = TR.store_for(win.store.project)
    rigs = [r['run_id'] for r in st.runs(study='rig')]
    assert len(rigs) == 2 and st.get(rigs[0])['project']['sha256'] != st.get(rigs[1])['project']['sha256']

    page.select_ids(rigs)
    assert page.b_compare.isEnabled() and not page.b_rerun.isEnabled()
    page.compare_selected()
    _settle(win, app, page)
    row = next(r for r in page.compared['rows'] if r['key'] == 'rig.camera_count')
    assert row['values'] == [2.0, 1.0] and row['deltas'][1] == -1.0 and row['best'] == 1     # fewer is better
    assert page.cmp.topLevelItemCount() == len(page.compared['rows'])
    assert any(d['path'] == 'project.sha256' for d in page.compared['differences'])

    page.select_ids([rigs[1]])
    page.rerun_selected()
    _settle(win, app, page)
    reruns = [r for r in st.runs() if r.get('rerun_of') == rigs[1]]
    assert len(reruns) == 1 and reruns[0]['reproduce']['verdict'] == 'reproduces', page.last_message
    assert page.last_message.startswith('Re-run: Reproduces')

    page.select_ids([rigs[0]])
    page.edit_tags('baseline, v1')
    page.edit_note('the two-camera rig')
    _settle(win, app, page)
    assert st.get(rigs[0])['tags'] == ['baseline', 'v1'] and st.get(rigs[0])['note'] == 'the two-camera rig'
    page.f_tag.setText('baseline')
    assert page.table.topLevelItemCount() == 1
    page.f_tag.setText('')
    page.f_since.setText('2999-01-01')
    assert page.table.topLevelItemCount() == 0
    page.f_since.setText('not a date')
    assert 'not a date' in page.l_sel.text()
    page.f_since.setText('')

    out = tmp_path / 'export'
    page.export_runs(str(out))
    _settle(win, app, page)
    assert (out / 'runs.csv').is_file() and (out / 'mlruns').is_dir() and (out / 'wandb' / 'replay_wandb.py').is_file()
    assert page.last_message.startswith(f'Exported {len(st.runs())} runs')     # quick on: the live re-runs too
    path = page.export(str(tmp_path / 'runs.json'))                   # the summary with provenance
    assert path and os.path.isfile(path)


def test_refusal_is_logged_as_failed_and_unsaved_says_so(app, win, tmp_path):
    page = win.pages['runs']
    win.store.edit('analysis', lambda p: p.analysis.studies.__setitem__('tracking', {'enabled': True}),
                   'tracking on')
    page.refresh()
    _settle(win, app, page)
    assert _word(page) == 'not saved' and 'save it' in page.head.state['explain']
    win.store.set_project(_saved(tmp_path))
    win.store.edit('analysis', lambda p: p.analysis.studies.__setitem__('tracking', {'enabled': True}),
                   'tracking on')
    _settle(win, app, page)
    win.ctx.analysis.run('grasp_visibility')                          # needs a robot: refused at once
    _settle(win, app, page)
    rows = TR.store_for(win.store.project).runs()
    assert len(rows) == 1 and rows[0]['status'] == 'failed' and 'robot' in rows[0]['reason']
    assert 'Traceback' not in rows[0]['reason']
    assert page.table.topLevelItemCount() == 1 and page.table.topLevelItem(0).text(3) == 'failed'


def test_ui_thread_budget_with_many_runs(app, win, tmp_path):
    from dexvantage.core.studies.tracking import row
    from dexvantage.core.tracking._stress import synthetic
    page = win.pages['runs']
    rows = []
    for i in range(2000):
        rec, _ = synthetic(1, i)
        rec['started_utc'] = f'2026-11-07T{(i // 3600) % 24:02d}:{(i // 60) % 60:02d}:{i % 60:02d}.000Z'
        rows.append(row(rec))
    doc = {'enabled': True, 'runs': rows, 'counts': {'runs': 2000},
           'headline': {'value': 2000, 'unit': 'runs', 'status': 'good', 'word': 'tracking on', 'text': ''}}
    t0 = time.perf_counter()
    page.show_result(doc)
    dt = time.perf_counter() - t0
    assert page.table.topLevelItemCount() == page.MAX_SHOWN and dt < 0.1, dt
    t0 = time.perf_counter()
    page.f_tag.setText('w1')
    assert time.perf_counter() - t0 < 0.1
