"""The canonical view warp (core/exporters/canonical_warp.py), its study (core/studies/canonical_view.py), its exporter
'canonical_warp', the batch flag and the Canonical view page."""
import json
import os

import cv2
import numpy as np
import pytest

from dexvantage.core import exporters as X
from dexvantage.core import jobs
from dexvantage.core.exporters import canonical_warp as CW
from dexvantage.core.frames import Pose
from dexvantage.core.project import blank_project
from dexvantage.core.robot.samples import sample_project
from dexvantage.validation import checks_warp as CK

from conftest import pump


def _merged():
    """The UR5e sample with the SO-100 arm and its wrist camera added: two embodiments in one project."""
    p = sample_project('ur5e')
    a = sample_project('so100')
    r = a.robots[0]
    r.primary = False
    r.base = Pose(np.array([0.0, -0.8, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))
    p.robots.append(r)
    p.sources.update(a.sources)
    p.variant.sensors.append(a.variant.sensors[0])
    return p


def _pair(lens=False, seed=0):
    return CK._pairs(np.random.default_rng(seed), lens)[0]


def test_homography_is_exact_on_the_plane_and_the_lut_matches_its_inverse():
    A, B, w = _pair()
    P = np.c_[np.random.default_rng(1).uniform(-0.04, 0.04, (200, 2)), np.zeros(200)]
    xa = CK._pinhole(A.K, CK._to_cam(A, P))
    xb = CK._pinhole(B.K, CK._to_cam(B, P))
    h = np.c_[xb, np.ones(len(xb))] @ w.H.T
    assert np.abs(h[:, :2] / h[:, 2:] - xa).max() < 1e-9
    assert w.H[2, 2] == pytest.approx(1.0)
    uv, ok = CW.lut_at(w, xa)
    assert ok.any() and np.abs(uv[ok] - xb[ok]).max() < 1e-9
    # the convention: n_B points at camera B (d_B > 0) and the plane's points satisfy n_B · X + d_B = 0
    XB = CK._to_cam(B, P)
    assert w.d_B > 0 and np.abs(XB @ w.n_B + w.d_B).max() < 1e-12


def test_parallax_vanishes_on_the_plane_and_for_a_shared_centre():
    A, B, w = _pair(lens=True)
    uv = CW.pixel_grid(A.camera, 40).reshape(-1, 2)
    D0, ok0 = CW.parallax_at(w, uv, 0.0)
    assert ok0.any() and np.nanmax(np.abs(D0[ok0])) < 1e-6
    D, ok = CW.parallax_at(w, uv, -0.1)
    assert np.nanmax(np.linalg.norm(D[ok], axis=1)) > 1.0
    # the source camera at the canonical camera's centre (rotated): no parallax at any depth
    T = A.T_contact_cam.copy()
    T[:3, :3] = T[:3, :3] @ Pose(np.zeros(3), np.array([0.0, 0.0, np.sin(0.1), np.cos(0.1)])).matrix()[:3, :3]
    w2 = CW.make_warp(CW.View('B', B.camera, T), A)
    assert w2.baseline_m < 1e-12
    D2, ok2 = CW.parallax_at(w2, uv, -0.15)
    assert ok2.any() and np.nanmax(np.abs(D2[ok2])) < 1e-6


def test_parallax_is_largest_at_an_end_of_the_depth_range():
    A, B, w = _pair(seed=3)
    uv = CW.pixel_grid(A.camera, 64).reshape(-1, 2)
    hs = np.linspace(-0.12, 0.03, 16)
    mags = np.stack([np.linalg.norm(CW.parallax_at(w, uv, h)[0], axis=1) for h in hs])
    ok = np.isfinite(mags).all(axis=0)
    ends = np.maximum(mags[0, ok], mags[-1, ok])
    assert (mags[:, ok].max(axis=0) <= ends + 1e-9).all()


def test_remap_maps_and_the_grid_sample_grid():
    A, B, w = _pair()
    mx, my, ok = CW.remap_maps(w)
    assert mx.dtype == np.float32 and mx.shape == (A.camera.height, A.camera.width)
    assert (mx[~ok] == CW.INVALID).all() and ok.mean() > 0.1
    g = CW.grid_sample_grid(w, mx, my, ok)
    assert g.shape == mx.shape + (2,)
    W = B.camera.width
    assert np.allclose(g[..., 0][ok], (2 * mx[ok].astype(np.float64) + 1) / W - 1, atol=1e-6)
    assert (g[~ok] == CW.GRID_INVALID).all()


def test_the_validation_checks_pass():
    from dexvantage.validation.suite import CHECKS, Ctx, ValidationSettings, load_checks, run_one
    load_checks()
    ctx = Ctx(blank_project('v'), None, None, ValidationSettings(quick=True))
    try:
        for cid in ('vla.warp_plane_exact', 'vla.warp_lut_lens_cv2', 'vla.warp_parallax_bruteforce',
                    'vla.warp_lut_vs_warpperspective', 'formats.canonical_warp_files'):
            spec = next(s for s in CHECKS if s.id == cid)
            c = run_one(spec, ctx)
            assert c.passed, (cid, c.error, c.exception, c.detail)
    finally:
        ctx.close()


def test_views_of_the_ur5e_sample():
    p = sample_project('ur5e')
    wrist = next(s for s in p.variant.sensors if s.name == 'Wrist D405')
    over = next(s for s in p.variant.sensors if s.name == 'Overhead D435')
    vw = CW.view_of_sensor(p, wrist.uid)
    vo = CW.view_of_sensor(p, over.uid)
    assert vw.rigid and not vo.rigid and any('not fixed' in n for n in vo.notes)
    assert vw.camera.width == 1280 and np.allclose(vw.T_contact_cam[3], [0, 0, 0, 1])
    # the camera looks at the fingertips: the TCP is in front of it
    tcp_cam = np.linalg.inv(vw.T_contact_cam) @ np.array([0, 0, 0, 1.0])
    assert tcp_cam[2] > 0


def test_study_on_the_ur5e_sample_is_jsonable_and_explained():
    p = sample_project('ur5e')
    d = jobs.run_study(CW.SID, p)
    json.dumps(d, allow_nan=False)
    h = d['headline']
    assert h['unit'] == 'px' and h['word'] in CW.WORDS.values() and h['status'] in CW.WORDS
    assert d['depth_range']['default'] and d['depth_range']['h_lo_m'] < 0 < d['depth_range']['h_hi_m']
    assert d['depth_range']['pad_reach_m'] is not None
    assert 0 < d['coverage']['share'] <= 1 and len(d['homography']) == 3
    assert d['parallax']['at_lo']['max_px'] >= d['parallax']['at_lo']['p95_px']
    assert d['suggestion'] and d['suggestion']['parallax_after_px'] == 0.0


def test_two_embodiments_default_pair_and_the_suggestion_removes_the_parallax():
    p = _merged()
    so_cam = p.variant.sensors[-1]
    d = jobs.run_study(CW.SID, p)
    assert d['source']['sensor_uid'] == so_cam.uid and d['canonical']['name'] == 'Wrist D405'
    assert d['source']['robot'] == 'SO-ARM100' and d['canonical']['robot'].startswith('UR5e')
    assert d['headline']['value'] > 1.0
    sg = d['suggestion']
    assert sg['sensor_uid'] == so_cam.uid and sg['parent_frame'] == 'Fixed_Jaw'
    so_cam.mount.pose = Pose.from_dict(sg['pose'])
    d2 = jobs.run_study(CW.SID, p)
    assert d2['relative']['baseline_m'] < 1e-9
    assert d2['headline']['value'] < 1e-6 and d2['headline']['word'] == 'aligned'
    assert d2['suggestion'] is None


def test_settings_offset_flip_depth_range_and_refusals():
    p = _merged()
    base = CW.analyse(p)
    p.analysis.studies[CW.SID] = {'plane_offset_m': 0.01, 'closing_sign_source': -1, 'depth_range_m': [-0.05, 0.01]}
    d = CW.analyse(p)
    assert d['plane']['offset_m'] == 0.01 and not d['plane']['offset_default']
    assert d['depth_range']['rule'] == 'your setting' and d['depth_range']['h_lo_m'] == -0.05
    assert not np.allclose(d['homography'], base['homography'])
    p.analysis.studies[CW.SID] = {'depth_range_m': [0.1, -0.1]}
    with pytest.raises(CW.WarpError, match='lower to its upper'):
        CW.analyse(p)
    cam = p.variant.sensors[0].uid
    p.analysis.studies[CW.SID] = {'source': cam, 'canonical': cam}
    with pytest.raises(CW.WarpError, match='same camera'):
        CW.analyse(p)
    with pytest.raises(jobs.StudyError, match='needs a robot'):
        jobs.run_study(CW.SID, blank_project('empty'))
    one = sample_project('so100')
    one.variant.sensors = one.variant.sensors[:1]
    with pytest.raises(CW.WarpError, match='second camera'):
        CW.analyse(one)
    nov = sample_project('so100')
    nov.workspace.volumes = []
    with pytest.raises(CW.WarpError, match='task volume'):
        CW.analyse(nov)


def test_camera_in_the_plane_is_refused():
    A, B, _w = _pair()
    T = B.T_contact_cam.copy()
    T[2, 3] = 0.0
    with pytest.raises(CW.WarpError, match='lies in the contact plane'):
        CW.make_warp(CW.View('B', B.camera, T), A)


def test_export_files_round_trip_and_a_loaded_canonical_view(tmp_path):
    p = _merged()
    man = X.export(p, str(tmp_path), ['canonical_warp'])
    row = man['formats']['canonical_warp']
    assert row['status'] == 'ok', row
    assert all(r in man['files'] for r in row['files'])
    f = {os.path.basename(r).rsplit('_', 1)[-1]: str(tmp_path / r) for r in row['files']}
    assert set(f) == {'map1.npy', 'map2.npy', 'remap.yml', 'grid.npy', 'valid.npy', 'parallax.npy', 'view.json',
                      'homography.json'}
    m1, m2 = np.load(f['map1.npy']), np.load(f['map2.npy'])
    img = (np.arange(720 * 1280) % 251).astype(np.uint8).reshape(720, 1280)
    out = cv2.remap(img, m1, m2, cv2.INTER_LINEAR)
    assert out.shape == (720, 1280)
    fs = cv2.FileStorage(f['remap.yml'], cv2.FILE_STORAGE_READ)
    assert np.array_equal(fs.getNode('map1').mat(), m1)
    fs.release()
    h = json.load(open(f['homography.json'], encoding='utf-8'))
    assert h['schema'] == CW.SCHEMA and 'provenance' in h and 'convention' in h
    assert h['parallax_full_resolution']['max_px'] >= h['parallax_full_resolution']['p95_px']
    par = np.load(f['parallax.npy'])
    assert par.dtype == np.float32 and par.shape == m1.shape
    # the canonical view file loads back as a virtual canonical camera: the same homography
    view = json.load(open(f['view.json'], encoding='utf-8'))
    q = _merged()
    q.analysis.studies[CW.SID] = {'canonical': view}
    d = CW.analyse(q)
    assert d['canonical']['virtual'] and np.allclose(d['homography'], h['homography'], atol=1e-9)
    with pytest.raises(CW.WarpError, match='formats'):
        CW.export_warp(p, str(tmp_path / 'x'), {'formats': ['png']})


def test_export_refuses_without_a_robot(tmp_path):
    man = X.export(blank_project('none'), str(tmp_path), ['canonical_warp'])
    assert man['formats']['canonical_warp']['status'] == 'refused'
    assert 'robot' in man['formats']['canonical_warp']['reason']


def test_overlay_geometry():
    p = _merged()
    w, _st = CW.build(p)
    g = CW.world_geometry(w)
    assert g['plane'].shape == (5, 3) and g['source'] and g['canonical']
    tcp = w.source.T_world_contact[:3, 3]
    assert np.linalg.norm(g['plane'][:4].mean(axis=0) - tcp) < 1e-9
    lines = CW.warped_grid(w, 'canonical')
    assert lines and all(x.shape[1] == 2 for x in lines)
    cam = w.canonical.camera
    allp = np.concatenate(lines)
    assert (allp[:, 0] >= -0.5).all() and (allp[:, 0] <= cam.width - 0.5).all()
    assert CW.warped_grid(w, 'source')


def test_self_test_line():
    e_h, e_lut = CW.self_test()
    assert e_h < 1e-9 and e_lut < 1e-6


def test_batch_flag(tmp_path):
    from test_batch import _run
    out = tmp_path / 'cv.json'
    r = _run('--sample', 'ur5e', '--canonical-view', str(out))
    assert r.returncode == 0, r.stderr
    d = json.load(open(out, encoding='utf-8'))
    assert d['study'] == CW.SID and 'provenance' in d and d['headline']['unit'] == 'px'


def test_page(app, win, tmp_path):
    from dexvantage.core import camrender as CR
    from dexvantage.ui.pages import canonical_view as PG
    assert 'canonical_view' in win.pages
    assert win.open_sample('ur5e')
    a = sample_project('so100')
    r = a.robots[0]
    r.primary = False
    r.base = Pose(np.array([0.0, -0.8, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))

    def add(pr):
        pr.robots.append(r)
        pr.sources.update(a.sources)
        pr.variant.sensors.append(a.variant.sensors[0])
    win.store.edit(['robots', 'sources', 'variant'], add, 'add the SO-100')
    page = win.pages['canonical_view']
    win.open_page('canonical_view')
    page.activate()
    assert page.src_box.count() == 3 and page.can_box.count() == 3
    page.run()
    assert win.ctx.analysis.wait(120)
    pump(app, 0.1)
    d = page.doc
    assert d is not None and page.head.figure.text() not in ('', '—')
    assert page.head.status.text.text() in CW.WORDS.values()
    assert page.tab.topLevelItemCount() == 2 and page.map.pixmap() is not None
    assert page.b_apply.isEnabled()
    from dexvantage.ui.viewport3d import LineDraw
    items = page.overlays(win.viewport)
    assert len(items) > 3 and all(isinstance(x, LineDraw) for x in items)
    src = win.store.project.variant.sensor(d['source']['sensor_uid'])
    can = win.store.project.variant.sensor(d['canonical']['sensor_uid'])
    from dexvantage.core import plucker as PK
    for s, want in ((can, 'warped here'), (src, 'reads')):
        im = PK.pick_imager(s, PK.sensor_stream(s))
        pose = CR.optical_pose(win.store.project, s, im, win.store.q)
        win.camview.view.sensor_uid = s.uid
        ov = page.camview_overlays(im.camera, pose)
        assert ov and want in ov[0][2] and all(x.shape[1] == 2 for x in ov[0][0])
    # a setting is one undo step and re-runs the study
    n = win.store.undo.count()
    page.s_offset.committed.emit(5.0)
    assert win.store.undo.count() == n + 1
    assert win.store.project.analysis.studies[CW.SID]['plane_offset_m'] == pytest.approx(0.005)
    assert win.ctx.analysis.wait(120)
    pump(app, 0.1)
    assert page.doc['plane']['offset_m'] == pytest.approx(0.005)
    # Apply: one undo step, the parallax goes to zero
    page.preview()
    assert page._preview
    n = win.store.undo.count()
    page.apply()
    assert win.store.undo.count() == n + 1
    assert win.ctx.analysis.wait(120)
    pump(app, 0.1)
    assert page.doc['headline']['value'] < 1e-6 and not page.b_apply.isEnabled()
    win.store.undo.undo()
    # exports: the JSON with provenance and the maps through the exporter
    out = page.export(str(tmp_path / 'cv.json'))
    assert 'provenance' in json.load(open(out, encoding='utf-8'))
    man = page.export_maps(str(tmp_path / 'maps'), wait=True)
    assert man['formats']['canonical_warp']['status'] == 'ok'
    assert os.path.isfile(tmp_path / 'maps' / 'export_manifest.json')
    # a canonical view file as the canonical camera
    view = next(r for r in man['formats']['canonical_warp']['files'] if r.endswith('canonical_view.json'))
    assert page.load_view(str(tmp_path / 'maps' / view)) is not None
    assert page.can_box.currentData() == PG.LOADED
    assert win.ctx.analysis.wait(120)
    pump(app, 0.1)
    assert page.doc['canonical']['virtual']
