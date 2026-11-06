"""Phase 2 product review: what a researcher meets on the samples, fixed and kept fixed.

Batch output that cannot be written is said in a sentence (exit 2), a coverage result keyed by robot uid reproduces;
the custom device form refuses figures the lens model cannot show (NaN, 0, a pinhole at 180° or more) in the form's
own words; a custom device the optics cannot model leaves that sensor out of coverage and depth validity with the
reason instead of stopping the study; study failures are plain sentences (page, status bar) that leave the status bar
once the study runs; coverage notes are shown, one camera gets the right advice, a device missing from the library
and switched-off sensors are said; the coverage slice stays visible at the slider's end; the Camera view draws the
slice and the other sensors' min-Z zones; Export says why nothing happens before a run; the Inspector does not link a
source that is not a URL and gives the position its unit.
"""
import copy
import os
import subprocess
import sys

import pytest

from conftest import pump

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def _env():
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    env.pop('PYTHONPATH', None)
    return env


def test_batch_output_that_cannot_be_written_is_a_sentence_and_exit_2(tmp_path):
    out = tmp_path / 'no-such-folder' / 'cov.json'
    r = subprocess.run([sys.executable, '-m', 'dexvantage', '--sample', 'so100', '--depth-validity', str(out)],
                       cwd=ROOT, env=_env(), capture_output=True, text=True, timeout=300)
    assert r.returncode == 2, r.stderr
    assert 'Traceback' not in r.stderr
    assert 'could not write' in r.stderr and 'Choose a folder' in r.stderr


def test_reproduce_compares_results_keyed_by_uid_by_place():
    """Coverage keys q and robots by robot uid, which is new in every sample project: compare by order."""
    from dexvantage.core import provenance as PV
    a = {'q': {'0123456789': [0.5, 1.0]}, 'robots': {'0123456789': 'UR5e'}, 'cameras': [{'sensor_uid': 'aaaaaaaaaa'}]}
    b = {'q': {'abcdef0123': [0.5, 1.0]}, 'robots': {'abcdef0123': 'UR5e'}, 'cameras': [{'sensor_uid': 'bbbbbbbbbb'}]}
    rep = PV.compare(a, b)
    assert rep['verdict'] == 'reproduces', rep
    c = copy.deepcopy(b)
    c['q']['abcdef0123'][1] = 2.0
    assert PV.compare(a, c)['verdict'] == 'does_not_reproduce'


# ---------------------------------------------------------------- the custom device form
@pytest.fixture
def dialog(win):
    from dexvantage.ui.dialogs.custom_device import CustomDeviceDialog
    d = CustomDeviceDialog(win.ctx, win, kind='depth')
    d.editor('maker').setText('Acme')
    d.editor('name').setText('Stereo One')
    d.editor('src_url').setText('measured by me with a checkerboard')
    for k, v in (('hfov', '87'), ('vfov', '58'), ('resolution', '1280, 720'), ('baseline_mm', '50')):
        d.set_given(k, True)
        d.editor(k).setText(v)
    assert d._check() == []
    yield d
    d.deleteLater()


@pytest.mark.parametrize('key,text,words', [
    ('hfov', 'nan', 'not a finite number'), ('hfov', 'inf', 'not a finite number'),
    ('hfov', '0', 'must be more than 0'), ('baseline_mm', '0', 'must be more than 0'),
    ('hfov', '200', 'a pinhole lens shows less than 180°'), ('vfov', '180', 'a pinhole lens shows less than 180°'),
])
def test_custom_device_refuses_figures_the_lens_cannot_show(dialog, key, text, words):
    dialog.editor(key).setText(text)
    errs = dialog._check()
    assert errs and words in errs[0], errs
    assert not dialog.b_save.isEnabled()
    assert '[' not in errs[0] and '_deg' not in errs[0]       # the form's words, not a schema path


def test_custom_device_fisheye_may_be_wider_than_180(dialog):
    dialog.set_value('lens_model', 'equidistant')
    dialog.editor('hfov').setText('200')
    dialog.set_given('vfov', False)
    assert dialog._check() == []


def test_schema_messages_use_the_form_words():
    from dexvantage.ui.dialogs.custom_device import _plain
    assert _plain('imagers[1].hfov_deg: 400.0 is greater than the maximum of 360') == \
        'Horizontal FOV: must be at most 360'
    assert _plain('depth.baseline_m: 0.0 is less than or equal to the minimum of 0') == \
        'Stereo baseline: must be more than 0'


def test_custom_device_save_confirmation_stays(dialog):
    from dexvantage.core import library
    try:
        e = dialog.save()
        assert e is not None
        assert dialog.msg.text().startswith('Saved ')
    finally:
        library.remove_custom('acme-stereo-one')


# ---------------------------------------------------------------- a device the optics cannot model
@pytest.fixture
def bad_device():
    from dexvantage.core import library
    d = copy.deepcopy(library.get('intel-realsense-d435').data)
    d['id'], d['name'] = 'review-too-wide', 'Too wide'
    for im in d['imagers']:
        im['hfov_deg'] = 200.0                # pinhole: cannot be modelled
        im['fov_modes'] = []
    library.add_custom(d)
    yield d['id']
    library.remove_custom(d['id'])


def test_unmodellable_device_is_left_out_with_its_reason(bad_device):
    from dexvantage.core import jobs, library
    from dexvantage.core.robot.samples import sample_project
    from dexvantage.core.studies import coverage, depth_validity  # noqa: F401
    p = sample_project('ur5e')
    s = p.variant.sensors[1]
    s.model_id = bad_device
    with pytest.raises(library.LibraryError, match='cannot be modelled'):
        library.sensor_imagers(s)
    snap = p.snapshot()
    snap.analysis.studies['coverage'] = {'voxel_mm': 20, 'refine_mm': 20, 'backend': 'numpy', 'cube_n': 64}
    snap.analysis.studies['depth_validity'] = {'backend': 'numpy', 'cube_n': 64}
    snap.analysis.voxel_mm = 20
    cov = jobs.run_study('coverage', snap)
    assert [c['name'] for c in cov['cameras']] == ['Wrist D405']
    assert any('Overhead D435 is left out' in n and 'cannot be modelled' in n for n in cov['notes'])
    dv = jobs.run_study('depth_validity', snap)
    assert [x['name'] for x in dv['sensors']] == ['Wrist D405']
    assert any('cannot be modelled' in k['why'] for k in dv['skipped'])


# ---------------------------------------------------------------- study messages
def test_failure_text_is_plain():
    from dexvantage.ui.store import failure_text
    t, ok = failure_text('StudyError: There is no task volume to cover.\nTraceback (most recent call last): …')
    assert ok and t == 'There is no task volume to cover.'
    t, ok = failure_text('ZeroDivisionError: division by zero\nTraceback …', 'The coverage study')
    assert not ok and 'Traceback' not in t and 'division' not in t and 'log' in t


def test_coverage_messages_say_what_to_do():
    from dexvantage.core import jobs
    from dexvantage.core.project import blank_project
    from dexvantage.core.robot.samples import sample_project
    from dexvantage.core.studies import coverage, depth_validity  # noqa: F401
    p = sample_project('ur5e')
    for s in p.variant.sensors:
        s.enabled = False
    with pytest.raises(jobs.StudyError, match='switched off; tick "In use"'):
        jobs.run_study('coverage', p.snapshot())
    assert 'switched off' in jobs.run_study('depth_validity', p.snapshot())['message']
    # one camera: add a second one (not "aim it at the volume" when it sees all of it)
    from dexvantage.ui.rigedit import new_sensor
    b = blank_project('One camera')
    b.variant.sensors.append(new_sensor(b))
    b.analysis.studies['coverage'] = {'backend': 'numpy', 'cube_n': 64}
    b.analysis.voxel_mm = 20
    doc = jobs.run_study('coverage', b.snapshot())
    assert 'it is the only camera, so add a second camera' in doc['summary']['advice']
    # a device id the library does not have: the default camera, said
    p = sample_project('ur5e')
    p.variant.sensors[1].model_id = 'not-a-device'
    p.analysis.studies['coverage'] = {'backend': 'numpy', 'cube_n': 64}
    p.analysis.voxel_mm = 20
    doc = jobs.run_study('coverage', p.snapshot())
    assert any('"not-a-device" is not in the library' in n for n in doc['notes'])


# ---------------------------------------------------------------- the window
def _run(win, app, pid, sid):
    win.open_page(pid)
    win.pages[pid].run()
    win.ctx.analysis.wait(120)
    pump(app, 0.1)
    return win.pages[pid]


@pytest.fixture
def ur5e_win(app, settings):
    from dexvantage.core.robot.samples import sample_project
    from dexvantage.ui.mainwindow import MainWindow
    p = sample_project('ur5e')
    p.analysis.voxel_mm = 20
    p.analysis.studies['coverage'] = {'cube_n': 128}
    p.analysis.studies['depth_validity'] = {'cube_n': 128}
    w = MainWindow(p, soft_3d=True, settings=settings)
    w.resize(1400, 900)
    w.show()
    w.ctx.analysis.wait(10)
    pump(app, 0.05)
    yield w
    w.cancel_jobs()
    w.store.undo.setClean()
    w.hide()
    w.deleteLater()
    pump(app, 0.02)


def test_failure_leaves_the_status_bar_when_the_study_runs(ur5e_win, app):
    w, st = ur5e_win, ur5e_win.ctx.store
    st.edit('workspace', lambda p: p.workspace.volumes.clear(), 'no volume')
    page = _run(w, app, 'coverage', 'coverage')
    assert page.head.figure.text() == '—' and 'StudyError' not in page.head.explain.text()
    assert w.msg_label.isVisible() and w.msg_label.text().startswith('Coverage and resolution could not run: ')
    st.undo.undo()
    _run(w, app, 'coverage', 'coverage')
    assert not w.msg_label.isVisible() or 'could not run' not in w.msg_label.text()


def test_slice_end_notes_and_camera_view_overlays(ur5e_win, app):
    w, st = ur5e_win, ur5e_win.ctx.store
    page = _run(w, app, 'coverage', 'coverage')
    assert page.doc is not None and not page.notes_label.isVisible()
    page.slider.setValue(1000)
    ax, x = page.slice_position()
    assert page.slice_mask(ax, x).any() and page.overlays(None)
    over = st.project.variant.sensors[1]
    st.select(('sensor', over.uid))
    pump(app, 0.05)
    w.camview.render_sync()
    ov = w.camview.view.page_overlays()
    assert any(lbl.startswith('slice z = ') for _p, _c, lbl in ov)
    _run(w, app, 'depth', 'depth_validity')
    w.camview.view.refresh_overlays()
    ov = w.camview.view.page_overlays()
    assert any(lbl == 'min-Z zone · Wrist D405' and len(p) for p, _c, lbl in ov)


def test_export_before_a_run_says_why(win):
    page = win.pages['coverage']
    assert page.doc is None
    assert page.export() is None
    assert 'Run it first' in win.msg_label.text()


def test_inspector_source_link_and_position_unit(win):
    from PyQt6.QtWidgets import QLabel

    from dexvantage.core import library
    from dexvantage.ui import rigedit as RE
    from dexvantage.ui.widgets import ScrubSpinBox
    d = copy.deepcopy(library.get('intel-realsense-d405').data)
    d['id'], d['name'] = 'review-measured', 'Measured'
    d['sources'] = [{'url': 'measured by me with a checkerboard', 'read': '2026-11-06'}] + d['sources']
    library.add_custom(d)
    try:
        RE.add_sensor(win.ctx.store, 'review-measured')
        labels = [lb.text() for lb in win.inspector.findChildren(QLabel)]
        mine = [t for t in labels if 'measured by me' in t]
        assert mine and all('<a ' not in t for t in mine)
        linked = [t for t in labels if '<a href="https://' in t]
        assert linked and all('style="color:' in t for t in linked)
        boxes = win.inspector.findChildren(ScrubSpinBox)
        assert boxes and all(b.unit_label == 'm' for b in boxes[:3])
    finally:
        library.remove_custom('review-measured')


def test_inspector_partial_dash_tooltip(win):
    from PyQt6.QtWidgets import QLabel

    from dexvantage.core import library
    from dexvantage.ui import rigedit as RE
    d = copy.deepcopy(library.get('intel-realsense-d405').data)
    d['id'], d['name'] = 'review-no-dfov', 'No diagonal'
    for im in d['imagers']:
        im['dfov_deg'] = None
    library.add_custom(d)
    try:
        RE.add_sensor(win.ctx.store, 'review-no-dfov')
        fov = [lb for lb in win.inspector.findChildren(QLabel) if lb.text().endswith('× —')]
        assert fov and library.NOT_STATED in fov[0].toolTip()
    finally:
        library.remove_custom('review-no-dfov')
