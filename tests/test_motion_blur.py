"""Motion blur and rolling shutter (core/studies/motion_blur.py), its page (ui/pages/motion_blur.py), its validation
checks (validation/checks_blur.py), the library's timing figures and the UR5e sample's scripted pick."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from dexvantage.core import jobs
from dexvantage.core.studies import motion_blur as MB

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


@pytest.fixture(scope='module')
def ur5e():
    from dexvantage.core.robot.samples import sample_project
    return sample_project('ur5e')


@pytest.fixture(scope='module')
def doc(ur5e):
    return jobs.run_study('motion_blur', ur5e)


def _cam(doc, sensor, imager):
    return next(c for c in doc['cameras'] if c['sensor'] == sensor and c['imager'] == imager)


# ====================================================================== the model
def test_projection_jacobian_against_differences():
    from dexvantage.core.optics import CameraModel
    rng = np.random.default_rng(0)
    for cam in (CameraModel.from_fov(640, 480, 80.0, None),
                CameraModel(640, 480, 400.0, 410.0, 319.5, 239.5, 'brown', (-0.2, 0.05, 0.001, -0.001, 0.0))):
        P = np.c_[rng.uniform(-0.3, 0.3, (20, 2)), rng.uniform(0.5, 2.0, 20)]
        J = MB.projection_jacobian(cam, P)
        h = 1e-7
        for k in range(3):
            d = np.zeros(3)
            d[k] = h
            fd = (cam.project(P + d)[0] - cam.project(P - d)[0]) / (2 * h)
            assert np.allclose(J[:, :, k], fd, rtol=1e-5, atol=1e-3)


def test_image_velocity_closed_forms():
    """A point moving parallel to the image plane: u̇ = f·v/Z; the camera moving the other way gives the same; a
    camera turning about its y axis moves a point on the axis at f·ω."""
    from dexvantage.core.optics import CameraModel
    cam = CameraModel.from_fov(1280, 720, 90.0, None)
    f = cam.fx
    P = np.array([[0.1, -0.05, 0.8]])
    _uv, ud, ok, z = MB.image_velocity(cam, np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3), P, [[0.2, 0.0, 0.0]])
    assert ok[0] and z[0] == pytest.approx(0.8) and ud[0] == pytest.approx([f * 0.2 / 0.8, 0.0])
    _uv, ud2, _ok, _z = MB.image_velocity(cam, np.eye(3), np.zeros(3), [-0.2, 0.0, 0.0], np.zeros(3), P,
                                          np.zeros((1, 3)))
    assert ud2[0] == pytest.approx(ud[0])
    _uv, ud3, _ok, _z = MB.image_velocity(cam, np.eye(3), np.zeros(3), np.zeros(3), [0.0, 0.5, 0.0],
                                          [[0.0, 0.0, 1.0]], np.zeros((1, 3)))
    assert np.linalg.norm(ud3[0]) == pytest.approx(f * 0.5)
    # behind the camera: not counted
    _uv, _ud, ok4, _z = MB.image_velocity(cam, np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3),
                                          [[0.0, 0.0, -1.0]], np.zeros((1, 3)))
    assert not ok4[0]


def test_rolling_capture_against_row_scan():
    """Linear image motion: the closed form against scanning the rows one by one."""
    rng = np.random.default_rng(1)
    H, Tr = 480, 0.02
    uv = np.c_[rng.uniform(0, 640, 30), rng.uniform(0, H - 1, 30)]
    ud = rng.uniform(-3000, 3000, (30, 2))
    got, ts = MB.rolling_capture(uv, ud, H, Tr)
    rows = np.arange(H)
    t_rows = Tr * rows / (H - 1)
    for j in range(len(uv)):
        if not np.isfinite(ts[j]) or ts[j] > Tr:
            continue
        d = uv[j, 1] + ud[j, 1] * t_rows - rows
        s = np.flatnonzero(np.sign(d[:-1]) != np.sign(d[1:]))
        if not len(s):
            continue
        i = s[0]
        a = d[i] / (d[i] - d[i + 1])
        t = t_rows[i] + a * (t_rows[i + 1] - t_rows[i])
        assert ts[j] == pytest.approx(t, abs=1e-12)
        assert got[j] == pytest.approx(uv[j] + ud[j] * t, abs=1e-9)
    same, t0 = MB.rolling_capture(uv, ud, H, 0.0)                  # a global shutter: nothing moves
    assert np.array_equal(same, uv) and not t0.any()


def test_settings_defaults_and_clamping(ur5e):
    st = MB.settings_of(ur5e)
    assert st['smear_limit_px'] == 1.0 and 'smear_limit_px' in st['defaults']
    p = ur5e.snapshot()
    p.analysis.studies['motion_blur'] = {'smear_limit_px': -3, 'step_s': 'x', 'max_segments': 3.7,
                                         'exposure_s': {'a': {'value': 0.002, 'source': 's'}, 'b': {'value': -1},
                                                        'c': 5}}
    st = MB.settings_of(p)
    assert st['smear_limit_px'] == MB.LIMITS['smear_limit_px'][0] and st['step_s'] == MB.DEFAULTS['step_s']
    assert st['max_segments'] == 4 and list(st['exposure_s']) == ['a']


# ====================================================================== the library and the sample
def test_library_timing_figures_are_cited_and_valid():
    from dexvantage.core import library as LB
    d435, d405 = LB.get('intel-realsense-d435'), LB.get('intel-realsense-d405')
    for e in (d435, d405):
        assert LB.validate(e.data) == []
    ir = d435.imager('IR left')
    assert ir['exposure_s']['min'] == pytest.approx(0.001) and ir['exposure_s']['max'] == pytest.approx(0.166)
    assert 'Table 4-23' in ir['exposure_s']['what'] and d435.imager('RGB')['exposure_s'] is None
    assert all(im['readout_s'] is None for im in d435.imagers + d405.imagers)       # no readout time is stated
    assert d435.data['sync']['hardware_trigger'] is True and d435.data['sync']['imagers'] == ['IR left', 'IR right']
    assert d405.data['sync']['hardware_trigger'] is False and 'D405 does not support' in d405.data['sync']['what']
    assert all(im['exposure_s'] is None for im in d405.imagers)


def test_ur5e_sample_has_a_scripted_pick(ur5e):
    from dexvantage.core import grasps as GR
    from dexvantage.core.robot.model import get_model
    (tr,) = ur5e.tasks.trajectories
    m = get_model(ur5e, tr['robot_uid'])
    assert tr['name'] == 'Scripted pick (sample)' and tr['joints'] == list(m.joint_names) and len(tr['t']) == 6
    opening = GR.pad_opening(ur5e, tr['robot_uid'], np.asarray(tr['q'], float))
    assert opening[0] > 0.05 and opening[-1] < 0.02                  # the gripper closes on the part
    vol = ur5e.workspace.volumes[0]
    lo, hi = vol.bounds()
    g = GR.gripper_geometry(ur5e, tr['robot_uid'])
    for q in tr['q'][1:]:
        T = m.link_pose(np.asarray(q), g['frame'])
        p = T[:3, :3] @ g['tcp'] + T[:3, 3]
        assert np.all(p >= lo - 1e-6) and np.all(p <= hi + 0.06)     # inside the volume (lifted to its top)


# ====================================================================== the study
def test_study_on_the_ur5e_sample(doc):
    text = json.dumps(doc, allow_nan=False)
    assert doc['study'] == 'motion_blur' and doc['source'] == 'trajectories' and '/home/' not in text
    assert {c['name'] for c in doc['cameras']} == {'Wrist D405 · IR left', 'Wrist D405 · RGB',
                                                   'Overhead D435 · IR left', 'Overhead D435 · RGB'}
    (tr,) = doc['trajectories']
    assert len(tr['segments']) == 5 and tr['segments'][1]['status'] == 'good'     # the arm holds still 1.0-1.5 s
    # every exposure is the frame-period bound, so a segment over the limit "may be blurred", like the headline
    assert {sg['word'] for sg in tr['segments'] if sg['status'] == 'serious'} == {'may be blurred'}
    assert tr['segments'][1]['word'] == 'sharp'
    for c in doc['cameras']:
        assert c['seen'] and c['exposure']['kind'] == 'upper bound'
        assert c['exposure']['value'] == pytest.approx(1 / 30, abs=1e-9)          # the frame period at 30 fps
        assert c['smear_px'] == pytest.approx(c['peak_speed_px_s'] * c['exposure']['value'], rel=1e-3)
        assert c['max_exposure_s'] == pytest.approx(1.0 / c['peak_speed_px_s'], rel=1e-3)
        assert c['frames'] and c['frames'][0]['vectors']
    rgb = _cam(doc, 'Overhead D435', 'RGB')
    assert rgb['shutter'] == 'rolling' and rgb['skew_bound'] and rgb['readout']['kind'] == 'upper bound'
    assert rgb['skew_px'] == pytest.approx(rgb['peak_speed_px_s'] / 30, rel=1e-3)
    assert _cam(doc, 'Wrist D405', 'IR left')['skew_px'] == 0.0                    # global shutter
    ir = _cam(doc, 'Overhead D435', 'IR left')
    assert ir['exposure']['stated_min'] == pytest.approx(0.001)
    # the wrist camera sees the scene stream past, faster than the fixed camera sees the gripper
    assert _cam(doc, 'Wrist D405', 'IR left')['peak_speed_px_s'] > ir['peak_speed_px_s']
    s = doc['summary']
    assert s['status'] == 'serious' and s['smear_bound'] and doc['smear_limit_default']
    sg = doc['suggestion']
    assert sg['kind'] == 'exposure' and {ch['sensor'] for ch in sg['changes']} == {'Wrist D405', 'Overhead D435'}
    assert all(ch['smear_after_px'] <= 1.0 for ch in sg['changes']) and 'global-shutter' in sg['text']


def test_your_exposure_and_apply(ur5e, doc):
    p = ur5e.snapshot()
    MB.apply_suggestion(p, doc['suggestion'])
    d2 = jobs.run_study('motion_blur', p)
    for c in d2['cameras']:
        assert c['exposure']['kind'] == 'yours' and 'suggestion' in c['exposure']['source']
        assert c['smear_px'] <= 1.0 + 1e-9 and c['status'] == 'good'
    assert d2['summary']['status'] == 'good'
    # the rolling shutter still skews: the suggestion says so but has no exposure to change
    assert d2['suggestion']['kind'] == 'advice' and 'rolling shutter' in d2['suggestion']['text']
    # an exposure shorter than the maker states is refused as advice
    q = ur5e.snapshot()
    q.analysis.studies['motion_blur'] = {'smear_limit_px': 0.01}
    d3 = jobs.run_study('motion_blur', q)
    assert any('shortest exposure the maker states' in a for a in d3['suggestion']['advice'])


def test_grasp_approaches_without_a_trajectory():
    from dexvantage.core.robot.samples import sample_project
    p = sample_project('so100')
    assert not p.tasks.trajectories
    d = jobs.run_study('motion_blur', p)
    assert d['source'] == 'grasp approaches' and d['trajectories'] and all(t['synthetic'] for t in d['trajectories'])
    assert any('grasp approaches' in n for n in d['notes'])
    for c in d['cameras']:                                       # no library device: no frame rate, no exposure
        assert c['exposure']['kind'] == 'unknown' and c['smear_px'] is None and c['readout']['kind'] == 'unknown'
    assert d['summary']['word'] == 'exposure not set' and d['summary']['max_exposure_s'] > 0
    p.tasks.grasp_sets = []
    d = jobs.run_study('motion_blur', p)
    assert d['source'] == 'none' and d['summary']['word'] == 'no motion'


def test_refusals_are_sentences(ur5e):
    p = ur5e.snapshot()
    p.variant.sensors = []
    with pytest.raises(jobs.StudyError, match='no camera'):
        jobs.run_study('motion_blur', p)
    from dexvantage.core.project import blank_project
    with pytest.raises(jobs.StudyError, match='needs a robot'):
        jobs.run_study('motion_blur', blank_project())


def test_batch_flag(tmp_path):
    assert jobs.STUDIES['motion_blur'].batch == 'motion-blur'
    out = tmp_path / 'blur.json'
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    r = subprocess.run([sys.executable, '-m', 'dexvantage', '--sample', 'ur5e', '--motion-blur', str(out)],
                       capture_output=True, text=True, env=env, cwd=ROOT, timeout=300)
    assert r.returncode == 0, r.stderr
    d = json.loads(out.read_text())
    assert d['study'] == 'motion_blur' and 'provenance' in d


# ====================================================================== validation
def test_validation_checks_pass():
    from dexvantage.validation import checks_blur  # noqa: F401 — registers the checks
    from dexvantage.validation import suite as S
    ctx = S.Ctx(None, None, None, S.ValidationSettings(quick=True))
    for spec in [s for s in S.CHECKS if s.id.startswith('blur.')]:
        out = spec.fn(ctx)
        assert out.ok and out.error <= (out.tolerance or spec.tolerance), (spec.id, out.error, out.detail)
    assert {s.id for s in S.CHECKS if s.id.startswith('blur.')} == {'blur.image_velocity_fd', 'blur.smear_analytic',
                                                                    'blur.rolling_shutter_bruteforce'}


# ====================================================================== the page
def test_page_runs_applies_undoes_and_draws(app, win):
    from conftest import pump
    from dexvantage.core.robot.samples import sample_project
    win.store.set_project(sample_project('ur5e'))
    win.ctx.analysis.wait(60)
    assert 'motion_blur' in win.pages
    win.open_page('motion_blur')
    page = win.pages['motion_blur']
    page.run()
    win.ctx.analysis.wait(120)
    pump(app, 0.1)
    assert page.doc is not None and page.doc['study'] == 'motion_blur'
    assert page.head.figure.text().startswith('up to ') and page.head.status is not None
    assert page.cams.topLevelItemCount() == 4 and page.segs.topLevelItemCount() == 5
    assert page.b_apply.isEnabled() and page.overlays(win.viewport)
    # a worst frame: the robot is posed there and the Camera view gets its smear vectors
    assert page.show_frame(0)
    c, f = page._frame
    assert np.allclose(win.store.q[f['robot_uid']], f['q'], atol=1e-6)
    from dexvantage.core import library as LB
    ims = LB.sensor_imagers(win.store.project.variant.sensor(c['sensor_uid']))
    cam = next(im.camera for im in ims if im.name == c['imager'])
    polys, scale = page.frame_vectors(cam)
    assert polys and all(p.shape == (2, 2) for p in polys) and scale >= 1.0
    win.camview.view.sensor_uid, win.camview.view.imager_name = c['sensor_uid'], c['imager']
    assert page.camview_overlays(cam, None)
    # Apply: one undo step
    n = win.store.undo.count()
    assert page.apply()
    assert win.store.undo.count() == n + 1
    assert set(win.store.project.analysis.studies['motion_blur']['exposure_s']) == {
        s.uid for s in win.store.project.variant.sensors}
    win.ctx.analysis.wait(120)
    pump(app, 0.1)
    win.store.undo.undo()
    assert 'exposure_s' not in (win.store.project.analysis.studies.get('motion_blur') or {})
    # your exposure for one camera
    s = win.store.project.variant.sensors[1]
    assert page.set_exposure(s.uid, 0.002, 'the driver setting')
    assert win.store.project.analysis.studies['motion_blur']['exposure_s'][s.uid] == {
        'value': 0.002, 'source': 'the driver setting'}
    assert 'Known limits' in page.explain_text()
    win.ctx.analysis.wait(120)
