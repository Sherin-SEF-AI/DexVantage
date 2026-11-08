"""Targetless link-silhouette calibration (core/calib/render_calib.py), its study, page and checks: the geometry and
the edge features against closed forms, recovery of a known mount slip from MuJoCo renders of the UR5e sample's
overhead camera, the observability test (a static robot, a robot too small), the refusals, the drift over a small
synthetic LeRobot dataset, the study and its batch flag, and the page (Apply in one undo step). Small: 320 × 180 px,
6–8 frames."""
import json
import math
import os

import numpy as np
import pytest

from conftest import pump
from dexvantage.core import jobs
from dexvantage.core.calib import render_calib as RC
from dexvantage.core.datasets.episode import CameraStream, Episode
from dexvantage.core.robot.model import get_model
from dexvantage.core.robot.samples import sample_project
from dexvantage.core.sim.episodes import _delta

FAST = {'max_frames': 6, 'bootstrap': 4, 'outer_iters': 3, 'inner_iters': 6, 'device': 'auto'}
SLIP_P = [0.004, -0.003, 0.0]           # 5 mm
SLIP_R = [0.6, 0.0, 0.8]                # 1°
W, H = 320, 180


@pytest.fixture(scope='module')
def ur5e():
    return sample_project('ur5e')


@pytest.fixture(scope='module')
def renderer(ur5e):
    from render_support import require
    require()                               # renders MuJoCo scenes
    from dexvantage.core.sim.render import Renderer
    with Renderer(ur5e) as R:
        yield R


def motion(model, n, amp=0.4, seed=0):
    """Joint vectors around home, every joint moving."""
    lo = np.where(np.isfinite(model.lower), model.lower, -np.pi)
    hi = np.where(np.isfinite(model.upper), model.upper, np.pi)
    k = np.arange(n)[:, None]
    Q = model.home + amp * np.sin(0.35 * k + np.arange(model.nq)[None, :] + seed) * np.minimum(1.0, (hi - lo) / 4)
    return np.clip(Q, lo, hi)


def rendered_episode(project, R, sensor, Q, delta=None, noise=1.0, seed=0) -> Episode:
    """An in-memory episode: the frames rendered through the mount perturbed by delta (sensor body), noise added."""
    rid = project.robots[0].uid
    cam = R.camera(sensor.uid).model.scaled(W, H)
    views = [{'sensor_uid': sensor.uid, 'q': {rid: q}, 'outputs': ('rgb',), 'camera_model': cam,
              'extrinsic_delta': delta} for q in Q]
    rng = np.random.default_rng(seed)
    imgs = [np.clip(o['rgb'] + rng.normal(0, noise, o['rgb'].shape), 0, 255).astype(np.uint8)
            for o in R.render(views)]
    t = np.arange(len(Q)) / 10.0

    def dec(idx):
        for i in (range(len(imgs)) if idx is None else idx):
            yield i, imgs[i]
    cs = CameraStream('observation.images.overhead_d435', W, H, 10.0, t.copy(), _decode=dec)
    m = get_model(project, rid)
    return Episode('e', 0, t, np.asarray(Q, float), list(m.joint_names), cameras={cs.key: cs})


def overhead(p):
    return next(s for s in p.variant.sensors if not s.mount.on_link)


# ---------------------------------------------------------------------- geometry and features
def test_se3_and_frames():
    xi = np.array([0.01, -0.02, 0.003, 0.1, -0.2, 0.05])
    assert np.allclose(RC.log_se3(RC.exp_se3(xi)), xi, atol=1e-12)
    D = RC.exp_se3(xi)
    assert np.allclose(RC.compose(D, np.zeros(6)), D)
    # the same as the synthetic episodes' mount-slip delta (rotation vector in degrees)
    assert np.allclose(RC.exp_se3(np.r_[SLIP_P, np.radians(SLIP_R)]), _delta(SLIP_P, SLIP_R), atol=1e-12)
    Q = np.array([[0, 0], [0.001, 0], [1, 0], [1, 1], [0.5, 0.5]], float)
    ch = RC.choose_frames(Q, 3)
    assert len(ch) == 3 and 2 in ch and 3 in ch and len(set(ch)) == 3
    assert RC.choose_frames(Q, 10) == [0, 1, 2, 3, 4] and RC.choose_frames(Q, 2, [0, 1]) == [0, 1]
    assert RC.pose_clusters(Q, math.radians(5)) == [0, 0, 1, 2, 3]


def test_edges_subpixel_and_directional():
    img = np.zeros((60, 80, 3), np.uint8)
    img[20:40, 30:50] = 200                          # a square: edges between pixels 29|30, 49|50, 19|20, 39|40
    e = RC.Edges.image(img, [50, 150])
    assert e.n_edges > 0
    # points at x = 29.5 with a horizontal normal lie on the left edge: residual 0
    uv = np.array([[29.5, 25.0], [29.5, 30.0], [31.0, 30.0]])
    nrm = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    r, n, ok, _d = e.residuals(uv, 3.0, nrm)
    assert ok.all() and np.abs(r[:2]).max() < 0.1 and abs(abs(r[2]) - 1.5) < 0.1
    # directional: a point near the left edge with a vertical normal matches no vertical edge's pixels within 1 px
    r, n, ok, d = e.residuals(np.array([[29.5, 30.0]]), 1.0, np.array([[0.0, 1.0]]))
    assert not ok[0]
    # the mask boundary sits half a pixel out of the mask
    F = np.zeros((60, 80), bool)
    F[20:40, 30:50] = True
    m = RC.Edges.mask(F)
    r, n, ok, _d = m.residuals(np.array([[29.5, 30.0], [40.0, 39.5]]), 3.0, np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert ok.all() and np.abs(r).max() < 0.05
    # IoU and the median foreground
    a = np.zeros((10, 10), bool)
    a[:5] = True
    assert RC.iou(a, a) == 1.0 and RC.iou(a, ~a) == 0.0 and RC.iou(np.zeros_like(a), np.zeros_like(a)) is None
    bg = np.full((20, 20, 3), 100, np.uint8)
    imgs = [bg.copy() for _ in range(5)]
    imgs[2][5:10, 5:10] = 200
    fg = RC.foreground(imgs, 25)
    assert fg[2][5:10, 5:10].all() and fg[2].sum() == 25 and not fg[0].any()
    assert RC.foreground(imgs[:2], 25) is None


def test_jacobian_against_finite_differences(ur5e, renderer):
    s = overhead(ur5e)
    prob = RC._Problem.__new__(RC._Problem)
    sc = renderer.camera(s.uid)
    prob.T_bs, prob.T_so = sc.T_body_sensor, sc.T_sensor_optical
    cam = sc.model.scaled(W, H)
    X = np.array([[0.1, 0.4, 0.1], [-0.1, 0.5, 0.2], [0.0, 0.3, 0.0]])
    D = RC.exp_se3(np.array([0.01, 0.002, -0.003, 0.01, -0.02, 0.005]))
    uv, J = prob.jacobian(cam, D, X)
    for a in range(6):
        xi = np.zeros(6)
        xi[a] = 1e-6
        up, _ = prob.project(cam, RC.compose(D, xi), X)
        um, _ = prob.project(cam, RC.compose(D, -xi), X)
        assert np.allclose((up - um) / 2e-6, J[:, :, a], rtol=1e-4, atol=1e-3)


# ---------------------------------------------------------------------- recovery
def test_recovers_known_slip(ur5e, renderer):
    s = overhead(ur5e)
    m = get_model(ur5e, ur5e.robots[0].uid)
    D = _delta(SLIP_P, SLIP_R)
    ep = rendered_episode(ur5e, renderer, s, motion(m, 24), D)
    doc = RC.estimate(ep, ur5e, s.uid, settings=FAST, renderer=renderer)
    json.dumps(doc, allow_nan=True)
    assert doc['observable'], doc['why']
    est = RC.exp_se3(np.r_[doc['delta']['position_m'], np.radians(doc['delta']['rotation_deg'])])
    from dexvantage.core.calib.axxb import transform_error
    et, er = transform_error(est, D)
    sc = doc['scale']
    # within 1 pixel of the working image (0.32 px of position and 0.4 px of rotation measured, 2026-11-07)
    assert 1000 * et < sc['mm_per_px'] and math.degrees(er) < sc['deg_per_px'], (et, er, sc)
    assert doc['fit']['iou_after'] >= doc['fit']['iou_before'] - 0.01
    assert len(doc['frames_used']) == 6 and doc['sigma']['hessian'] and doc['sigma']['bootstrap']
    assert all(x > 0 for x in doc['sigma']['position_mm']) and doc['overlay']['png_base64']
    assert doc['overlay']['before'] and doc['overlay']['after']
    assert doc['render']['renders'] > 0 and doc['mount']['parent_frame'] == s.mount.parent_frame
    # the pose is the mount perturbed by ΔT: T_world_sensor · ΔT · T_sensor_optical
    from dexvantage.core.frames import Pose
    sc_ = renderer.camera(s.uid)
    T = sc_.T_body_sensor @ est @ sc_.T_sensor_optical
    assert np.allclose(Pose.from_dict(doc['pose']).matrix(), T, atol=1e-9)


def test_observability_static_and_small(ur5e, renderer):
    s = overhead(ur5e)
    m = get_model(ur5e, ur5e.robots[0].uid)
    Q = np.tile(m.home, (8, 1))
    ep = rendered_episode(ur5e, renderer, s, Q)
    doc = RC.estimate(ep, ur5e, s.uid, settings=dict(FAST, coarse_search=False), renderer=renderer)
    assert not doc['observable'] and 'too static' in doc['why'] and doc['observability']['distinct_poses'] == 1
    ep = rendered_episode(ur5e, renderer, s, motion(m, 12))
    doc = RC.estimate(ep, ur5e, s.uid, settings=dict(FAST, min_robot_frac=0.95, coarse_search=False),
                      renderer=renderer)
    assert not doc['observable'] and 'too small' in doc['why']
    lam = doc['observability']['min_eigenvalue']
    assert lam is not None and lam > 0 and doc['observability']['weakest_direction'].startswith('the camera')


def test_refusals(ur5e, renderer):
    wrist = next(s for s in ur5e.variant.sensors if s.mount.on_link)
    m = get_model(ur5e, ur5e.robots[0].uid)
    ep = rendered_episode(ur5e, renderer, overhead(ur5e), motion(m, 4))
    for args, words in (((ep, ur5e, wrist.uid), 'static cameras'), ((ep, ur5e, 'nope'), 'not in this rig'),
                        ((42, ur5e, overhead(ur5e).uid), 'needs a dataset')):
        with pytest.raises(RC.CalibError) as e:
            RC.estimate(*args, renderer=renderer)
        assert words in str(e.value) and 'Traceback' not in str(e.value)
    with pytest.raises(RC.CalibError) as e:
        RC.settings_of(ur5e, {'colour': 1})
    assert 'no setting colour' in str(e.value)
    with pytest.raises(RC.CalibError) as e:
        RC.estimate(ep, ur5e, overhead(ur5e).uid, settings={'camera': 'observation.images.x'}, renderer=renderer)
    assert 'no camera observation.images.x' in str(e.value)
    st = RC.settings_of(ur5e)
    assert st['need_mm'] == 5.0 and 'need_mm' in st['defaults'] and st['levels'] == [0.25, 0.5, 1.0]


# ---------------------------------------------------------------------- a synthetic LeRobot dataset, the study
@pytest.fixture(scope='module')
def dataset(ur5e, tmp_path_factory):
    from render_support import require
    require()                               # renders MuJoCo scenes
    pytest.importorskip('pyarrow', reason='pyarrow (the lerobot extra) is not installed')
    pytest.importorskip('av', reason='av (the lerobot extra) is not installed')
    from dexvantage.core.sim import episodes as SE
    faults = {k: False for k in SE.FAULT_KINDS}
    faults['mount_slip'] = {'from_episode': 1, 'position_m': SLIP_P, 'rotation_deg': SLIP_R}
    root = str(tmp_path_factory.mktemp('rcal') / 'ur5e')
    SE.generate(ur5e, root, {'episodes': 2, 'frames': 40, 'size': [W, H], 'seed': 4, 'faults': faults})
    return root


def test_drift_and_study(ur5e, dataset):
    from dexvantage.core.datasets import open_dataset
    s = overhead(ur5e)
    with open_dataset(dataset) as ds:
        dr = RC.drift(ds, ur5e, s.uid, settings=FAST)
        truth = ds.truth()                       # read only after the estimate
    assert [d['episode'] for d in dr['episodes']] == [0, 1]
    st = dr['steps'][0]
    assert st['from'] == 0 and st['to'] == 1 and st['step'], st
    assert abs(st['norm_mm'] - 5.0) < 2.0 and abs(st['angle_deg'] - 1.0) < 0.3
    f = next(x for x in truth['episodes']['1']['faults'] if x['kind'] == 'mount_slip')
    assert f['camera'] == dr['episodes'][1]['camera_key']
    # the study over the same dataset, from the project's settings
    snap = ur5e.snapshot()
    snap.analysis.studies['silhouette_calib'] = dict(FAST, dataset=dataset)
    doc = jobs.run_study('silhouette_calib', snap)
    json.dumps(doc, allow_nan=False)
    assert doc['sensor_uid'] == s.uid and len(doc['episodes']) == 2 and doc['headline']['episode'] == 1
    assert doc['word'] in ('mount moved', 'uncertain', 'mount holds', 'not observable')
    assert doc['apply'] and abs(doc['apply']['change_mm'] - 5.0) < 2.0 and doc['dataset']['format'] == 'lerobot'
    assert 'overlay' in doc['estimate'] and 'overlay' not in doc['episodes'][0]
    assert any('moved between episodes 0 and 1' in n for n in doc['notes'])
    assert jobs.STUDIES['silhouette_calib'].batch == 'silhouette-calib'
    # refusals in words
    snap.analysis.studies['silhouette_calib'] = {}
    with pytest.raises(jobs.StudyError) as e:
        jobs.run_study('silhouette_calib', snap)
    assert 'Choose a recorded dataset' in str(e.value)
    snap.analysis.studies['silhouette_calib'] = {'dataset': os.path.join(dataset, 'nothing-here')}
    with pytest.raises(jobs.StudyError) as e:
        jobs.run_study('silhouette_calib', snap)
    assert 'not there (nothing at that path)' in str(e.value)


def test_checks_registered():
    from dexvantage.validation import suite as S
    from dexvantage.validation import references as RF
    import dexvantage.validation.checks_render_calib as CR       # noqa: F401
    ids = {c.id: c for c in S.CHECKS}
    for cid in ('handeye.silhouette_recovery', 'handeye.silhouette_vs_board'):
        assert cid in ids and ids[cid].model == 'handeye' and ids[cid].unit == 'px'
        assert all(r in RF.REFERENCES for r in ids[cid].refs)


# ---------------------------------------------------------------------- the page
def test_page(app, win, ur5e, dataset):
    assert 'silhouette' in win.pages
    assert win.open_sample('ur5e')
    page = win.pages['silhouette']
    win.open_page('silhouette')
    pump(app, 0.05)
    assert page.head.status.text.text() == 'not run yet'
    assert page.sensor_box.count() == 1 and page.sensor_box.currentText() == 'Overhead D435'
    page.choose_dataset(dataset)
    assert win.store.project.analysis.studies['silhouette_calib']['dataset'] == dataset
    assert page.dataset_label.text() == os.path.basename(dataset)
    snap = win.store.project.snapshot()
    snap.analysis.studies['silhouette_calib'] = dict(FAST, dataset=dataset)
    doc = jobs.run_study('silhouette_calib', snap)
    page._finished('silhouette_calib', doc)
    pump(app, 0.05)
    assert page.head.status.text.text() == doc['word'] and 'mm' in page.head.figure.text()
    assert page.table.topLevelItemCount() == 2 and not page.thumb.pixmap().isNull()
    assert page.b_apply.isEnabled() and page.overlays(win.viewport)
    page.preview()
    s = win.store.project.variant.sensor(doc['sensor_uid'])
    before = s.mount.pose.to_dict()
    n0 = win.store.undo.count()
    assert page.apply()
    assert win.store.undo.count() == n0 + 1
    from dexvantage.core.frames import Pose
    now = win.store.project.variant.sensor(doc['sensor_uid']).mount.pose
    assert np.allclose(now.matrix(), Pose.from_dict(doc['apply']['pose']).matrix())
    win.store.undo.undo()
    assert np.allclose(win.store.project.variant.sensor(doc['sensor_uid']).mount.pose.matrix(),
                       Pose.from_dict(before).matrix())
    # the Camera view, showing the calibrated camera, outlines the robot through the recovered mount
    from dexvantage.core import camrender as CR
    from dexvantage.core import library as LB
    p = win.store.project
    im = LB.sensor_imagers(s)[0]
    cm = im.camera.scaled(W, H)
    pose = CR.optical_pose(p, s, im, dict(win.store.q))
    win.camview.view.sensor_uid = ''
    assert page.camview_overlays(cm, pose) == []                 # another camera: nothing
    win.camview.view.sensor_uid = s.uid
    out = []
    for _ in range(300):
        out = page.camview_overlays(cm, pose)
        if out:
            break
        pump(app, 0.05)
    assert out and out[0][2] == 'The robot through the recovered mount' and len(out[0][0]) >= 1
    exp = page.export(os.path.join(os.path.dirname(dataset), 'sil.json'))
    assert 'provenance' in json.loads(open(exp, encoding='utf-8').read())
    page._replaced()
    pump(app, 0.2)
