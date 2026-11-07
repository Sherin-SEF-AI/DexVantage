"""The Campaign page (ui/pages/campaign.py, Data ▸ Campaign): the headline with its status word, settings edits as one
undo step each (refused in words when unreadable), Apply writing the plan (one undo step), Preview, Export of the
files, the spec file round trip and the UI-thread budget."""
import json
import time

from conftest import pump
from dexvantage.core import campaign as CP


def _wait(win, app, timeout=60):
    win.ctx.analysis.wait(timeout)
    pump(app, 0.05)


def _page(win, app):
    from dexvantage.core.robot.samples import sample_project
    win.store.set_project(sample_project('ur5e'))
    _wait(win, app)
    assert 'campaign' in win.pages
    page = win.pages['campaign']
    win.open_page('campaign')
    page.run()
    _wait(win, app)
    return page


def test_page_headline_edits_and_apply(app, win, tmp_path):
    page = _page(win, app)
    assert page.doc is not None and page.doc['study'] == 'campaign'
    assert page.head.unit.text().startswith('episodes over ') and 'GB' in page.head.unit.text()
    assert page.head.status.text.text() == 'no budget set'
    assert page.form_labels['reset_s'].text() == 'Reset between episodes (s) (default)'
    u = win.store.undo
    n0 = u.index()
    assert page.set_field('operators', 'Ana, Ben') and u.index() == n0 + 1
    _wait(win, app)
    assert page.doc['operators'] == ['Ana', 'Ben']
    assert page.form_labels['operators'].text() == 'Operators'
    assert not page.set_field('reset_s', 'soon') and u.index() == n0 + 1
    assert 'not a number' in page.note.text()
    assert not page.set_field('day_start', '25:00') and 'time of day' in page.note.text()
    assert page.set_field('budget.disk_gb', '1') and u.index() == n0 + 2
    _wait(win, app)
    assert page.head.status.text.text() == 'over budget' and page.budget.topLevelItemCount() == 1
    assert page.set_field('budget.disk_gb', '') and u.index() == n0 + 3      # blank: no budget again
    _wait(win, app)
    assert 'disk_gb' not in win.store.project.analysis.studies['campaign'].get('budget', {})
    # Apply: one undo step; the plan is in the project; undo takes it out
    assert page.b_apply.isEnabled()
    k = u.index()
    t0 = time.perf_counter()
    doc = page.apply()
    ms = (time.perf_counter() - t0) * 1000
    assert doc is not None and u.index() == k + 1 and ms < 100, ms
    ap = CP.applied_plan(win.store.project)
    assert ap['digest'] == CP.plan_digest(page.doc) and not page.b_apply.isEnabled()
    u.undo()
    assert CP.applied_plan(win.store.project) is None
    u.redo()
    assert CP.applied_plan(win.store.project) is not None
    page.preview()
    assert 'already holds this plan' in win.msg_label.text()
    # the sessions, a checklist, the files
    assert page.sessions.topLevelItemCount() == len(page.doc['sessions'])
    page.show_session(page.doc['sessions'][-1]['index'])
    assert '☐' in page.checklist.text()
    paths = page.export_files(str(tmp_path))
    assert [p.rsplit('.', 1)[1] for p in paths] == ['csv', 'ics', 'html']
    out = page.export(str(tmp_path / 'campaign.json'))
    assert json.load(open(out))['provenance']
    ctx = page.page_context()
    assert ctx['headline']['word'] and ctx['study'] == 'campaign'


def test_spec_round_trip_through_the_page(app, win, tmp_path):
    page = _page(win, app)
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps({'schema': CP.SPEC_SCHEMA, 'operators': ['X', 'Y', 'Z'],
                                'lighting': [{'name': 'Bright'}, {'name': 'Dim'}]}))
    u = win.store.undo
    n0 = u.index()
    assert page.open_spec(str(path)) and u.index() == n0 + 1
    _wait(win, app)
    assert len(page.doc['conditions']) == 2 and page.doc['operators'] == ['X', 'Y', 'Z']
    assert page.edits['operators'].text() == 'X, Y, Z'
    out = page.save_spec(str(tmp_path / 'saved.json'))
    saved = json.load(open(out))
    assert saved['schema'] == CP.SPEC_SCHEMA and saved['lighting'][1]['name'] == 'Dim'
    path.write_text(json.dumps({'variants': ['Nope']}))
    assert not page.open_spec(str(path)) and u.index() == n0 + 1
    assert 'rig variant "Nope"' in page.note.text()


def test_the_page_is_in_the_data_menu(win):
    from dexvantage.ui.pages import PAGES
    assert PAGES['campaign'].menu == 'Data' and PAGES['campaign'].title == 'Campaign'
