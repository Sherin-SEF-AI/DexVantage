import json

import pytest
from PyQt6.QtGui import QKeySequence

from conftest import pump
from dexvantage.core import provenance as PV

MENUS = ['File', 'Edit', 'View', 'Robot', 'Sensors', 'Workspace', 'Tasks', 'Analysis', 'Calibrate', 'Simulate', 'Data',
         'Export', 'Window', 'Help']


def test_menus_docks_and_shortcuts(win):
    assert [a.text().replace('&', '') for a in win.menuBar().actions()] == MENUS
    assert set(win.docks) == {'outliner', 'library', 'inspector', 'camview', 'bottom', 'assistant'}
    sc = {k: win.actions_by_id[k].shortcut().toString() for k in ('palette', 'find', 'shortcuts')}
    assert sc == {'palette': 'Ctrl+K', 'find': 'Ctrl+F', 'shortcuts': '?'}


def test_later_phase_commands_are_disabled_and_say_when(win):
    later = [a for a in win.actions_by_id.values() if (a.data() or {}).get('phase')]
    assert len(later) >= 1                # the placeholders are enabled as their phases land (Phase 10: export)
    for a in later:
        assert not a.isEnabled() and f'Phase {a.data()["phase"]}' in a.toolTip()


def test_summary_headline_after_start(win):
    doc = win.ctx.analysis.results['summary']
    assert doc['volumes'][0]['voxels_dense'] == 720_000
    assert win.bottom.summary.head.figure.text() == '720 000'


def test_add_camera_undo_redo_and_delete(app, win):
    st = win.store
    win.add_static_camera()
    s = st.project.variant.sensors[-1]
    assert st.sel == ('sensor', s.uid) and s.mount.parent_frame == 'tripod'
    st.undo.undo()
    assert not st.project.variant.sensors
    st.undo.redo()
    st.select(('sensor', st.project.variant.sensors[0].uid))
    win.delete_selection()
    assert not st.project.variant.sensors


def test_camera_on_tripod_looks_at_the_task_volume(win):
    import numpy as np
    from dexvantage.core.frames import quat_to_matrix
    win.add_static_camera()
    s = win.store.project.variant.sensors[-1]
    fwd = quat_to_matrix(s.mount.pose.quaternion)[:, 0]          # body x is the optical axis
    to = np.asarray(win.store.project.workspace.volumes[0].center) - s.mount.pose.position
    assert fwd @ (to / np.linalg.norm(to)) == pytest.approx(1.0, abs=1e-9)


def test_inspector_edit_is_one_undo_step(app, win):
    st = win.store
    win.add_volume()
    from dexvantage.ui.widgets import ScrubSpinBox
    fields = win.inspector.findChildren(ScrubSpinBox)
    assert len(fields) == 6
    n = st.undo.count()
    fields[0].set_value(0.6)
    fields[0].committed.emit(0.6)
    pump(app, 0.02)
    assert st.undo.count() == n + 1 and st.project.workspace.volumes[-1].center[0] == pytest.approx(0.6)


def test_workspaces_switch_save_and_reset(app, win):
    win.workspace('audit')
    # the Data audit workspace opens on the page registered for it (the Dataset audit page)
    assert not win.docks['library'].isVisible() and win.bottom.tabs.currentWidget() is win.pages['audit']
    win.docks['inspector'].hide()
    win.save_workspace()
    win.workspace('placement')
    win.workspace('audit')
    assert not win.docks['inspector'].isVisible()
    win.reset_workspace()
    assert win.docks['inspector'].isVisible()


def test_palette_lists_commands_and_objects(win):
    from dexvantage.ui.dialogs.palette import CommandPalette
    win.add_static_camera()
    cp = CommandPalette(win.ctx, win, 'obj ')
    texts = [it['text'] for it in cp.items]
    assert 'Add camera on a tripod' in texts and 'Camera 1' in texts and 'Tabletop (default)' in texts
    assert cp.scope == 'objects'
    cp.deleteLater()


def test_dialogs_build(win):
    from dexvantage.ui.dialogs.misc import AboutDialog, FramesDialog, ShortcutsDialog
    sd = ShortcutsDialog(win)
    rows = [r[2] for r in sd.rows]
    assert QKeySequence('Ctrl+K').toString(QKeySequence.SequenceFormat.NativeText) in rows
    for d in (sd, FramesDialog(win), AboutDialog(win)):
        d.deleteLater()


def test_save_open_round_trip(tmp_path, win):
    win.add_object()
    path = str(tmp_path / 'cell.dexv')
    assert win._write(path)
    assert win.store.undo.isClean()
    win.new_project()
    assert win.open_project(path)
    assert win.store.project.workspace.objects[0].name == 'Box 1'
    assert win.store.project.thumb[:4] == b'\x89PNG'


def test_export_summary_carries_provenance(tmp_path, win):
    path = win.export_summary(str(tmp_path / 's.json'))
    doc = json.loads(open(path).read())
    assert PV.verify(doc)['ok'] is True and doc['provenance']['study'] == 'summary'


def test_render_image_is_high_resolution(win):
    img = win.viewport.render_image(scale=2.0)
    assert img.width() == 2 * win.viewport.width()


def test_light_theme_and_back(app, win):
    from PyQt6.QtWidgets import QApplication
    from dexvantage.ui.theme import THEME, TOKENS
    win.set_theme('light')
    assert THEME.name == 'light' and QApplication.palette().window().color().name() == TOKENS['light']['panel']
    win.set_theme('dark')
    assert QApplication.palette().window().color().name() == TOKENS['dark']['panel']


def test_validation_job_runs_in_the_background(app, win, tmp_path):
    # the job's plumbing, on two cheap models (the whole quick run takes minutes since Phases 4, 5 and 8)
    assert win.validate_models(folder=str(tmp_path), open_report=False, quick=True, models=['frames', 'formats'])
    for _ in range(3000):               # up to 60 s
        pump(app, 0.02)
        if 'validation' not in win._jobs:
            break
    assert (tmp_path / 'validation.html').exists()
    assert PV.verify(json.loads((tmp_path / 'validation.json').read_text()))['ok'] is True
