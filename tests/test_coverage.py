"""The coverage study (core/studies/coverage.py), its page (ui/pages/coverage.py), its batch flag and its checks."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from dexvantage.core import jobs
from dexvantage.core.frames import Pose
from dexvantage.core.project import Mount, Project, Sensor, TaskObject, Volume
from dexvantage.core.robot.model import get_model
from dexvantage.core.robot.samples import _site_in_body, look_at, sample_project
from dexvantage.core.studies import coverage as CV

from conftest import pump

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


@pytest.fixture(scope='module')
def ur5e_doc():
    p = sample_project('ur5e')
    return p, jobs.run_study('coverage', p)


@pytest.fixture(scope='module')
def so100_doc():
    p = sample_project('so100')
    return p, jobs.run_study('coverage', p)


def _static_project(n_cams=2, objects=()):
    p = Project('Coverage test')
    c = np.array([0.5, 0.0, 0.1])
    p.workspace.volumes.append(Volume(name='V', center=tuple(c), size=(0.2, 0.2, 0.2)))
    for k in range(n_cams):
        eye = c + [0.0, 0.0, 0.6] if k == 0 else c + [0.6, 0.3, 0.3]
        p.variant.sensors.append(Sensor(name=f'C{k}', slot=k, mount=Mount('cage', '', look_at(eye, c,
                                                                                             up=(1.0, 0, 0)))))
    p.workspace.objects += list(objects)
    p.analysis.voxel_mm = p.analysis.refine_mm = 10.0
    return p


# ---------------------------------------------------------------------- the study on the samples
def test_ur5e_result_is_json_and_consistent(ur5e_doc):
    p, doc = ur5e_doc
    text = json.dumps(doc, allow_nan=False)
    assert len(text) < 4_000_000                         # compact enough to keep in memory for the page
    s = doc['summary']
    for k in ('seen_ge1', 'seen_ge2', 'blind', 'resolved_ge1', 'resolved_ge2'):
        assert 0.0 <= s[k] <= 1.0
    assert s['seen_ge1'] >= s['seen_ge2'] >= s['resolved_ge2']
    assert abs(s['seen_ge1'] + s['blind'] - 1.0) < 1e-5
    assert abs(s['pct_ge2'] - 100 * s['seen_ge2']) < 1e-3
    assert s['status'] in ('good', 'warning', 'serious', 'critical') and s['word'] and s['advice']
    assert doc['q_source'] == {p.robots[0].uid: 'home'}
    assert doc['grid']['count'] == 480_000
    names = [c['name'] for c in doc['cameras']]
    assert names == ['Wrist D405', 'Overhead D435']
    for c in doc['cameras']:
        assert c['imager'] == 'RGB' and 'RGB imager' in c['imager_why']   # depth sensors: the RGB imager, said so
        assert abs(c['seen'] + c['hidden'] + c['outside_image'] - 1.0) < 1e-5
        assert c['ppmm_p5'] <= c['ppmm_median'] <= c['ppmm_p95']
        assert not c['defocus']['computed'] and 'Not computed' in c['defocus']['why']
        assert 'focus distance' in c['defocus']['why']
    over = doc['cameras'][1]
    assert over['in_image'] == 1.0 and over['seen'] > 0.5            # the overhead camera frames the whole volume
    assert doc['volumes'][0]['seen_ge2'] == s['seen_ge2']


def test_decode_matches_summary(ur5e_doc):
    _p, doc = ur5e_doc
    d = CV.decode(doc)
    n = doc['grid']['count']
    assert d['centers'].shape == (n, 3) and d['size'].shape == (n,)
    w = d['size'] ** 3 * ~d['inside']
    red = d['redundancy']
    assert abs(w[red >= 2].sum() / w.sum() - doc['summary']['seen_ge2']) < 1e-6
    for j, c in enumerate(doc['cameras']):
        seen = d['cameras'][j]['state'] == CV.STATE_SEEN
        assert abs(w[seen].sum() / w.sum() - c['seen']) < 1e-6
        assert np.all(d['cameras'][j]['ppmm'][seen] > 0)
        assert np.all(np.isnan(d['cameras'][j]['angle_deg'][d['cameras'][j]['state'] == CV.STATE_OUT]))
    lo, hi = sample_project('ur5e').workspace.volumes[0].bounds()
    assert np.all(d['centers'] > lo - 1e-6) and np.all(d['centers'] < hi + 1e-6)


def test_wrist_camera_sees_in_front_of_the_gripper(ur5e_doc):
    """Golden-ish: the UR5e wrist D405 looks at the pinch point; the space just beyond it is in view and clear."""
    p, doc = ur5e_doc
    m = get_model(p, p.robots[0].uid)
    pinch = (m.link_pose(m.home, 'gripper_base') @ np.r_[_site_in_body(m, 'gripper_pinch', 'gripper_base')[:3],
                                                          1.0])[:3]
    cam = np.array(doc['cameras'][0]['position_m'])
    d = CV.decode(doc)
    ahead = pinch + 0.06 * (pinch - cam) / np.linalg.norm(pinch - cam)
    near = np.linalg.norm(d['centers'] - ahead, axis=1) < 0.02
    assert near.sum() > 100
    assert (d['cameras'][0]['state'][near] == CV.STATE_SEEN).mean() > 0.95
    # and closer to the camera than the pinch point (behind it, at the flange) it sees nothing of the volume
    behind = d['cameras'][0]['state'][np.linalg.norm(d['centers'] - cam, axis=1) < 0.05]
    assert not (behind == CV.STATE_SEEN).any()


def test_so100_default_imagers_are_labelled(so100_doc):
    _p, doc = so100_doc
    for c in doc['cameras']:
        assert c['is_default'] and 'no device chosen' in c['imager']
        assert 'default' in c['imager_label'].lower()
        assert 'no lens figures' in c['defocus']['why']
    third = doc['cameras'][1]
    assert third['in_image'] == 1.0 and third['seen'] > 0.8


# ---------------------------------------------------------------------- joint state, settings, errors
def test_q_from_settings_moves_the_wrist_camera():
    p = sample_project('ur5e')
    r = p.robots[0]
    m = get_model(p, r.uid)
    q = m.home.copy()
    q[0] += 0.4
    p.analysis.studies['coverage'] = {'q': {r.uid: q.tolist()}, 'voxel_mm': 10.0, 'refine_mm': 10.0}
    doc = jobs.run_study('coverage', p)
    assert doc['q_source'][r.uid] == 'settings'
    assert np.allclose(doc['q'][r.uid], m.clip(q))
    home = sample_project('ur5e')
    home.analysis.studies['coverage'] = {'voxel_mm': 10.0, 'refine_mm': 10.0}
    d0 = jobs.run_study('coverage', home)
    moved = np.linalg.norm(np.subtract(doc['cameras'][0]['position_m'], d0['cameras'][0]['position_m']))
    assert moved > 0.05
    assert doc['cameras'][1]['position_m'] == d0['cameras'][1]['position_m']     # the overhead camera stays
    p.analysis.studies['coverage'] = {'q': {r.uid: [0.0, 1.0]}}
    with pytest.raises(jobs.StudyError, match='moving joints'):
        jobs.run_study('coverage', p)


def test_plain_errors():
    p = _static_project(0)
    with pytest.raises(jobs.StudyError, match='camera or depth sensor'):
        jobs.run_study('coverage', p)
    p = _static_project(1)
    p.workspace.volumes = []
    with pytest.raises(jobs.StudyError, match='no task volume'):
        jobs.run_study('coverage', p)
    p = _static_project(1)
    p.analysis.studies['coverage'] = {'good_pct': 'lots'}
    with pytest.raises(jobs.StudyError, match='good_pct'):
        jobs.run_study('coverage', p)
    p = _static_project(1)
    big = Volume(name='Big', size=(1.5, 1.5, 1.2))
    p.workspace.volumes = [big]
    p.analysis.voxel_mm = 1.0
    with pytest.raises(jobs.StudyError, match=r'2\.7 × 10⁹ voxels'):
        jobs.run_study('coverage', p)


def test_inside_objects_left_out_and_occlusion_by_objects():
    box = TaskObject(name='Block', shape='box', size=(0.1, 0.1, 0.1), pose=Pose(np.array([0.5, 0.0, 0.05])))
    p = _static_project(1, [box])
    doc = jobs.run_study('coverage', p)
    d = CV.decode(doc)
    ins = d['inside']
    assert ins.sum() > 0 and doc['grid']['inside_objects'] == int(ins.sum())
    assert not (d['cameras'][0]['state'][ins] == CV.STATE_SEEN).any()
    # the voxels inside the block are left out of the shares
    w = d['size'] ** 3 * ~ins
    assert abs(w[d['redundancy'] >= 1].sum() / w.sum() - doc['summary']['seen_ge1']) < 1e-6


def test_defocus_with_your_figures():
    p = _static_project(1)
    uid = p.variant.sensors[0].uid
    p.variant.sensors[0].model_id = 'intel-realsense-d435'
    p.variant.sensors[0].kind = 'depth'
    p.analysis.studies['coverage'] = {'optics': {uid: {'f_number': 2.0, 'focus_m': 0.6, 'pixel_um': 1.4}}}
    doc = jobs.run_study('coverage', p)
    df = doc['cameras'][0]['defocus']
    assert df['computed'] and df['source']['focus_m'] == 'yours' and df['source']['f_number'] == 'yours'
    assert df['blur_median_px'] is not None and df['sharp'] is not None
    assert df['dof_near_m'] < 0.6 < df['dof_far_m']
    d = CV.decode(doc)
    b = d['cameras'][0]['blur_px']
    assert b is not None and np.nanmin(b) < 0.5          # points near the focus distance are sharp


def test_cancel_returns_a_partial_result():
    p = sample_project('ur5e')
    calls = []
    ctx = jobs.Ctx(progress=lambda f, t: calls.append(f), cancel=lambda: len(calls) > 2)
    doc = jobs.run_study('coverage', p, ctx)
    assert doc['cancelled'] and doc['partial']
    json.dumps(doc, allow_nan=False)


def test_encode_roundtrip():
    a = (np.arange(1000, dtype=np.uint16) * 7) % 65535
    assert np.array_equal(CV.decode_array(CV.encode_array(a)), a)
    assert CV.wpercentile([1.0, 2.0, 3.0], [1.0, 1.0, 2.0], 50) == 2.0
    assert CV.wpercentile([], [], 50) is None
    st = CV.settings_of(Project())
    assert CV.status_for(95, st) == ('good', CV.WORDS['good'])
    assert CV.status_for(10, st)[0] == 'critical'
    assert set(st['defaults']) >= {'good_pct', 'warn_pct', 'serious_pct', 'min_px_per_mm'}


# ---------------------------------------------------------------------- the page
def test_page_renders_offscreen_and_runs_with_the_live_q(app, win, tmp_path):
    assert 'coverage' in win.pages
    assert win.actions_by_id['coverage'].isEnabled()
    assert win.open_sample('ur5e')
    page = win.pages['coverage']
    win.open_page('coverage')
    pump(app, 0.05)
    assert page.head.status.text.text() == 'not run yet'
    r = win.store.project.robots[0]
    m = get_model(win.store.project, r.uid)
    q = m.home.copy()
    q[0] += 0.2
    win.store.set_q(r.uid, q)
    page.run()
    assert win.ctx.analysis.wait(120)
    pump(app, 0.1)
    doc = page.doc
    assert doc is not None and doc['q_source'][r.uid] == 'settings'
    assert np.allclose(doc['q'][r.uid], win.store.q[r.uid])
    assert 'q' not in (win.store.project.analysis.studies.get('coverage') or {})   # the project is not changed
    assert page.head.figure.text() not in ('', '—')
    assert page.head.status.text.text() in CV.WORDS.values()
    assert page.cams.topLevelItemCount() == 2 and page.vols.topLevelItemCount() == 1
    assert page.cams.topLevelItem(0).text(9) == 'not computed'
    # the slice overlay is drawn while the page is shown
    items = page.overlays(win.viewport)
    assert items and len(items[0].tri) > 100 and items[0].colors.shape == (len(items[0].tri), 4)
    assert win.page_overlays(win.viewport)
    page.axis.set_value('x')
    page.axis.changed.emit('x')
    page.slider.setValue(250)
    assert page.slice_position()[0] == 'x' and page.overlays(win.viewport)
    page.layer.setCurrentIndex(page.layer.findData('best'))
    assert page.legend.mode == 'scale'
    page.layer.setCurrentIndex(page.layer.findData('cam:0'))
    assert page.overlays(win.viewport)
    page.show3d.setChecked(False)
    assert page.overlays(win.viewport) == []
    win.viewport.grab()
    # thresholds are user settings (undoable), labelled default until changed
    sb, dl = page._spin['good_pct']
    assert dl.text() == 'default'
    sb.committed.emit(20.0)
    assert win.store.project.analysis.studies['coverage']['good_pct'] == 20.0
    assert 'yours' in dl.text()
    assert page.stale_note.text() == ''                  # a threshold re-reads the status; no new run is needed
    assert page.head.status.text.text() == CV.WORDS['good']
    win.store.undo.undo()
    assert 'good_pct' not in (win.store.project.analysis.studies.get('coverage') or {})
    # moving the arm marks the result stale
    q2 = q.copy()
    q2[1] += 0.1
    win.store.set_q(r.uid, q2)
    assert 'moved' in page.stale_note.text()
    win.store.edit('workspace', lambda p: setattr(p.workspace.volumes[0], 'size', (0.3, 0.3, 0.3)), 'Resized')
    assert 'project changed' in page.stale_note.text()
    win.store.undo.undo()
    assert 'cube map' in page.explain_text()
    out = page.export(str(tmp_path / 'coverage.json'))
    exp = json.loads(open(out, encoding='utf-8').read())
    assert 'provenance' in exp and exp['summary'] == doc['summary']


def test_page_explains_a_failed_run(app, win):
    page = win.pages['coverage']
    win.store.project.variant.sensors = []
    page.run()
    assert win.ctx.analysis.wait(30)
    pump(app, 0.05)
    assert page.head.status.text.text() == 'could not run'
    assert 'StudyError' not in page.head.explain.text() and 'camera' in page.head.explain.text()


# ---------------------------------------------------------------------- batch and validation
def test_batch_flag_writes_coverage(tmp_path):
    out = tmp_path / 'cov.json'
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    r = subprocess.run([sys.executable, '-m', 'dexvantage', '--sample', 'so100', '--coverage', str(out)],
                       capture_output=True, text=True, env=env, cwd=ROOT, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    doc = json.loads(out.read_text())
    assert doc['study'] == 'coverage' and 'provenance' in doc and doc['summary']['seen_ge1'] > 0.5
    assert str(tmp_path) not in out.read_text()


def test_validation_checks_pass_quick():
    from dexvantage.validation import suite
    suite.load_checks()
    specs = {s.id: s for s in suite.CHECKS}
    st = suite.ValidationSettings(quick=True)
    ctx = suite.Ctx(None, None, None, st)
    try:
        for cid in ('occlusion.coverage_vs_raycast', 'optics.coverage_density_fd'):
            res = suite.run_one(specs[cid], ctx)
            assert res.passed, (cid, res.error, res.exception, res.detail)
    finally:
        ctx.close()
