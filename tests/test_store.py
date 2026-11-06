from conftest import pump

from dexvantage.core.project import Volume, blank_project
from dexvantage.ui.store import AnalysisController, ProjectStore


def _add(store, name):
    return store.edit('workspace', lambda p: p.workspace.volumes.append(Volume(name=name)), f'Added {name}')


def test_edit_undo_redo():
    st = ProjectStore(blank_project())
    seen = []
    st.changed.connect(lambda s: seen.append(s))
    _add(st, 'Shelf')
    assert [v.name for v in st.project.workspace.volumes][-1] == 'Shelf'
    st.undo.undo()
    assert 'Shelf' not in [v.name for v in st.project.workspace.volumes]
    st.undo.redo()
    assert [v.name for v in st.project.workspace.volumes][-1] == 'Shelf'
    assert seen == [{'workspace'}] * 3
    assert st.project.log[-1]['text'] == 'Added Shelf'


def test_drag_merges_into_one_step_and_esc_cancels():
    st = ProjectStore(blank_project())
    uid = st.project.workspace.volumes[0].uid

    def move(x):
        def fn(p):
            next(v for v in p.workspace.volumes if v.uid == uid).center = (x, 0, 0)
        st.edit('workspace', fn, 'Moved', merge_key=('vol', uid))
    st.dragging = True
    for x in (0.1, 0.2, 0.3):
        move(x)
    assert st.undo.count() == 1
    assert st.cancel_gesture()
    st.dragging = False
    assert st.project.workspace.volumes[0].center == (0.45, 0.0, 0.15) and st.undo.count() == 0


def test_variant_key_targets_the_active_variant():
    st = ProjectStore(blank_project())
    st.edit('variant', lambda p: setattr(p.variant, 'name', 'Rig B'), 'Renamed')
    st.undo.undo()
    assert st.project.variant.name == 'Rig A'


def test_selection_is_dropped_when_its_object_goes():
    st = ProjectStore(blank_project())
    _add(st, 'Bin')
    uid = st.project.workspace.volumes[-1].uid
    st.select(('volume', uid))
    st.undo.undo()
    assert st.sel == ()


def test_controller_runs_after_edits_and_caches(app):
    st = ProjectStore(blank_project())
    an = AnalysisController(st)
    got = []
    # the summary's results only: quick studies of later phases (bandwidth) run live after edits too
    an.finished.connect(lambda sid, doc: got.append(doc) if sid == 'summary' else None)
    an.request(force=True)
    assert an.wait(5)
    _add(st, 'Shelf')
    assert an.wait(5)
    pump(app, 0.05)
    assert len(got) >= 2 and len(got[-1]['volumes']) == 2 and not an.stale
    an.run('summary')
    assert an.wait(5)
    pump(app, 0.05)
    assert got[-1]['cached']


def test_controller_marks_stale_when_live_is_off(app):
    st = ProjectStore(blank_project())
    st.project.analysis.live = False
    an = AnalysisController(st)
    _add(st, 'Shelf')
    assert an.stale
