"""The Requirements page (ui/pages/requirements.py): the editable list (one undo step per edit), the scorecard with
status words, the variants compare with the best variant per requirement, Apply (one undo step), Preview, the 3D
overlay, YAML save and import, Export, and the UI-thread budget."""
import json
import time

from conftest import pump
from dexvantage.core import requirements as RQ
from dexvantage.core.project import Variant


def _wait(win, app, timeout=120):
    win.ctx.analysis.wait(timeout)
    pump(app, 0.05)


def _cheap(page, win):
    """Requirements whose studies are quick: camera count, controllers over budget, sync skew."""
    reqs = [RQ.clean({'name': 'One camera at most', 'metric': 'rig.camera_count', 'op': '<=', 'threshold': 1}),
            RQ.clean({'metric': 'bandwidth.over_budget', 'op': '<=', 'threshold': 0}),
            RQ.clean({'metric': 'sync.skew_ms', 'op': '<', 'threshold': 2})]
    win.store.edit('requirements', lambda p: setattr(p, 'requirements', reqs), 'Set requirements')
    return reqs


def test_page_list_edits_are_one_undo_step_each(app, win, tmp_path):
    from dexvantage.core.robot.samples import sample_project
    win.store.set_project(sample_project('ur5e'))
    _wait(win, app)
    assert 'requirements' in win.pages
    page = win.pages['requirements']
    win.open_page('requirements')
    assert page.head.status.text.text() == 'no requirements' and not page.b_del.isEnabled()
    u = win.store.undo
    n0 = u.index()
    assert page.add_defaults() == 5 and u.index() == n0 + 1
    assert page.table.topLevelItemCount() == 5 and page.table.topLevelItem(0).text(1) == '≥ 95 % (default)'
    assert page.add_defaults() == 0 and u.index() == n0 + 1               # nothing left to add: no step
    u.undo()
    assert win.store.project.requirements == [] and page.table.topLevelItemCount() == 0
    u.redo()
    assert len(win.store.project.requirements) == 5
    # change a default's threshold: one step, and it is the user's own figure now
    r = win.store.project.requirements[1]
    page._uid = r.uid
    page._fill_editor()
    n1 = u.index()
    assert page.set_field('threshold', '1.5') and u.index() == n1 + 1
    r = win.store.project.requirements[1]
    assert r.threshold == 1.5 and not r.is_default and '(default)' not in page.table.topLevelItem(1).text(1)
    assert not page.set_field('threshold', 'fast') and u.index() == n1 + 1     # refused in words
    assert 'not a number' in page.note.text()
    assert page.set_field('op', '<=') and win.store.project.requirements[1].op == '<='
    assert page.set_field('name', 'Skew') and page.table.topLevelItem(1).text(0) == 'Skew'
    u.undo()
    assert win.store.project.requirements[1].name != 'Skew'
    # add and remove one
    k = u.index()
    page.add('rig.camera_count')
    assert len(win.store.project.requirements) == 6 and u.index() == k + 1
    page.remove()
    assert len(win.store.project.requirements) == 5 and u.index() == k + 2
    # save as YAML and import it back (the --check format)
    path = str(tmp_path / 'req.yaml')
    assert page.save_yaml(path) == path
    assert [x.metric for x in RQ.load(path)] == [x.metric for x in win.store.project.requirements]
    assert page.import_file(path) == 5 and len(win.store.project.requirements) == 10
    bad = tmp_path / 'bad.yaml'
    bad.write_text('requirements: [{metric: nope, threshold: 1}]')
    assert page.import_file(str(bad)) == 0 and 'does not have' in page.note.text()


def test_page_scorecard_variants_apply_preview(app, win, tmp_path):
    from dexvantage.core.robot.samples import sample_project
    win.store.set_project(sample_project('ur5e'))
    _wait(win, app)
    page = win.pages['requirements']
    win.open_page('requirements')
    _cheap(page, win)
    assert page.head.status.text.text() == 'not run yet'
    page.run()
    _wait(win, app)
    assert page.doc is not None and page.doc['verdict'] == 'fail'
    st = [page.table.topLevelItem(i).text(4) for i in range(3)]
    assert st == ['fail', 'not evaluated', 'fail']                      # 2 cameras; no host; skew at least 54 ms
    assert page.table.topLevelItem(2).text(2).startswith('≥ 54.4')
    assert page.head.figure.text() == '0 / 3' and page.head.status.text.text() == 'not ready'
    assert 'no host controller' in page.table.topLevelItem(1).toolTip(4)
    assert page.overlays(win.viewport)                                  # each sensor marked
    # an edit makes the scorecard stale: said, and the edited row is "not run"
    page._uid = win.store.project.requirements[0].uid
    page.set_field('threshold', '3')
    assert 'Run again' in page.head.explain.text() and page.table.topLevelItem(0).text(4) == 'not run'
    win.store.undo.undo()
    # a second variant with one camera: it passes the camera count
    a = win.store.project.variant

    def add_variant(p):
        v = Variant.from_dict(json.loads(json.dumps(a.to_dict())))
        v.uid, v.name = 'b' * 10, 'One camera'
        v.sensors = v.sensors[:1]
        p.variants.append(v)
    win.store.edit('variants', add_variant, 'Add a variant')
    assert not page.b_apply.isEnabled()
    page.compare()
    _wait(win, app)
    d = page.vdoc
    assert d is not None and [v['name'] for v in d['variants']] == [a.name, 'One camera']
    assert page.matrix.topLevelItemCount() == 3 and page.matrix.columnCount() == 3
    assert d['best'][d['requirements'][0]['uid']] == ['b' * 10]
    assert 'best' in page.matrix.topLevelItem(0).text(2) and 'pass' in page.matrix.topLevelItem(0).text(2)
    assert d['ranking'][0] == 'b' * 10 and page.v_rank.text().startswith('Ranking by requirements passed: One camera')
    # Preview, then Apply: one undo step
    assert page.b_apply.isEnabled() and page.b_preview.isEnabled()
    n_plain = len(page.overlays(win.viewport))
    page.preview()
    assert len(page.overlays(win.viewport)) > n_plain
    n0 = win.store.undo.index()
    t0 = time.perf_counter()
    assert page.apply()
    dt = time.perf_counter() - t0
    assert win.store.project.active_uid == 'b' * 10 and win.store.undo.index() == n0 + 1
    assert dt < 0.1, dt                                                 # the UI-thread budget
    win.store.undo.undo()
    assert win.store.project.active_uid == a.uid
    # a new variant from the active one: one undo step, made active, its sensors keeping their uids
    n0 = win.store.undo.index()
    v = page.copy_variant()
    assert win.store.undo.index() == n0 + 1 and win.store.project.active_uid == v.uid
    assert v.name == f'{a.name} (2)' and [s.uid for s in v.sensors] == [s.uid for s in a.sensors]
    win.store.undo.undo()
    assert len(win.store.project.variants) == 2 and win.store.project.active_uid == a.uid
    # export with provenance
    out = page.export(str(tmp_path / 'card.json'))
    doc = json.load(open(out))
    assert doc['schema'] == 'dexvantage.requirements/1' and doc['provenance']['study'] == 'requirements'


def test_page_refresh_is_within_the_ui_budget(app, win):
    from dexvantage.core.robot.samples import sample_project
    win.store.set_project(sample_project('so100'))
    _wait(win, app)
    page = win.pages['requirements']
    win.store.edit('requirements', lambda p: setattr(p, 'requirements', RQ.default_requirements() * 4), 'Many')
    t0 = time.perf_counter()
    page.refresh()
    assert time.perf_counter() - t0 < 0.1
