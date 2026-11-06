"""Hand-eye calibration: boards (ChArUco, AprilGrid), printing at true scale, detection and PnP on rendered images,
the closed-form AX = XB / AX = ZB solvers, the bundle adjustment (eye-in-hand, eye-to-hand, multi-camera, two arms,
intrinsics, joint offsets, encoder noise, outliers) and its covariance, and the validation checks' registration."""
from __future__ import annotations

import json
import math
import re
import struct
import zlib

import numpy as np
import pytest

from dexvantage.core.calib import axxb, boards as B, detect as D, handeye as HE, synthetic as S


# =========================================================================== boards
def test_charuco_points_are_opencvs_chessboard_corners():
    b = B.CharucoBoard(9, 7, 0.03, 0.022)
    P = b.points()
    assert P.shape == (48, 3) and b.n_points == 48
    np.testing.assert_allclose(P, np.asarray(b.cv().getChessboardCorners(), float))
    # inner corners, row-major, one square apart, origin at the pattern's top-left outer corner
    np.testing.assert_allclose(P[0], [0.03, 0.03, 0.0])
    np.testing.assert_allclose(P[1] - P[0], [0.03, 0.0, 0.0])
    np.testing.assert_allclose(P[8] - P[0], [0.0, 0.03, 0.0])
    assert b.size_m == pytest.approx((0.27, 0.21))


@pytest.mark.parametrize('kw, words', [
    (dict(squares_x=5, squares_y=4, square_m=0.03, marker_m=0.03), 'smaller than the square'),
    (dict(squares_x=30, squares_y=30, square_m=0.03, marker_m=0.02, dictionary='DICT_4X4_50'), 'needs 450 markers'),
    (dict(squares_x=5, squares_y=4, square_m=0.03, marker_m=0.02, dictionary='DICT_NOPE'), 'no ArUco dictionary'),
    (dict(squares_x=2, squares_y=4, square_m=0.03, marker_m=0.02), 'at least 3'),
    (dict(squares_x=5, squares_y=4, square_m=-1, marker_m=0.02), 'positive'),
])
def test_charuco_refuses_bad_definitions_in_words(kw, words):
    with pytest.raises(B.BoardError, match=words):
        B.CharucoBoard(**kw)


def test_dictionary_names_accept_either_spelling():
    assert B.CharucoBoard(5, 4, 0.03, 0.02, 'dict_5x5_100').dictionary == 'DICT_5X5_100'
    assert B.CharucoBoard(5, 4, 0.03, 0.02, 'DICT_APRILTAG_36H11').dictionary == 'DICT_APRILTAG_36h11'
    assert 'DICT_5X5_100' in B.dictionary_names()


def test_aprilgrid_points_follow_kalibrs_layout():
    g = B.AprilGrid(6, 5, 0.04, 0.3)
    P = g.points()
    assert P.shape == (120, 3) and g.n_points == 120
    pitch = 1.3 * 0.04
    # Kalibr: index = R · 2 · cols + C, x = (C // 2) pitch + (C % 2) tag, likewise y
    for idx in (0, 1, 2, 13, 119):
        R, C = divmod(idx, 12)
        assert P[idx] == pytest.approx([(C // 2) * pitch + (C % 2) * 0.04, (R // 2) * pitch + (R % 2) * 0.04, 0.0])
    # tag 0 at the bottom left, ids along the columns first; corners in OpenCV's order (tl, tr, br, bl)
    tl, tr, br, bl = P[g.tag_point_ids(0)]
    assert bl[:2] == pytest.approx([0, 0]) and br[:2] == pytest.approx([0.04, 0])
    assert tr[:2] == pytest.approx([0.04, 0.04]) and tl[:2] == pytest.approx([0, 0.04])
    assert P[g.tag_point_ids(1)][3][:2] == pytest.approx([pitch, 0.0])          # to the right of tag 0
    assert P[g.tag_point_ids(6)][3][:2] == pytest.approx([0.0, pitch])          # the row above
    with pytest.raises(KeyError):
        g.tag_point_ids(30)
    assert B.AprilGrid(2, 2, 0.04, first_id=10).tag_point_ids(13)[3] == 2 * 4 + 2
    with pytest.raises(B.BoardError, match='36h11 family has 587'):
        B.AprilGrid(30, 20, 0.02)


def test_board_records_round_trip():
    for b in (B.CharucoBoard(7, 5, 0.04, 0.03, 'DICT_6X6_250', legacy=True), B.AprilGrid(4, 3, 0.05, 0.25, 7)):
        d = json.loads(json.dumps(b.to_dict()))
        assert B.board_from_dict(d) == b
    with pytest.raises(B.BoardError, match='unknown board kind'):
        B.board_from_dict({'kind': 'checker'})


def test_rasterise_covers_exactly_the_black_area():
    """Exact area coverage: the ink summed over the image is the black area in pixels, to rounding."""
    b = B.CharucoBoard(5, 4, 0.03, 0.022)
    img = B.rasterise(b, 10000.0, 0.0)                 # 10 px per mm: 150 × 120 mm
    assert img.shape == (1200, 1500)
    area = sum((c - a) * (d - bb) for a, bb, c, d in b.page_rects()) * 10000.0 ** 2
    ink = (255.0 - img.astype(float)).sum() / 255.0
    assert ink == pytest.approx(area, rel=2e-3)


def _png_chunks(data: bytes) -> dict:
    out, i = {}, 8
    while i < len(data):
        n = struct.unpack('>I', data[i:i + 4])[0]
        t = data[i + 4:i + 8]
        out.setdefault(t, []).append(data[i + 8:i + 8 + n])
        crc = struct.unpack('>I', data[i + 8 + n:i + 12 + n])[0]
        assert crc == zlib.crc32(t + data[i + 8:i + 8 + n]) & 0xFFFFFFFF
        i += 12 + n
    return out


def test_png_is_true_scale_and_detectable(tmp_path):
    import cv2
    b = B.CharucoBoard(7, 5, 0.03, 0.022)
    path = B.save_board(b, str(tmp_path / 'board.png'), dpi=200)
    data = open(path, 'rb').read()
    ch = _png_chunks(data)
    w, h = struct.unpack('>II', ch[b'IHDR'][0][:8])
    ppm_x, ppm_y, unit = struct.unpack('>IIB', ch[b'pHYs'][0])
    assert unit == 1 and ppm_x == ppm_y == round(200 / 0.0254)
    # the pattern plus a one-square margin on each side, at 200 dpi
    assert w == math.ceil((0.21 + 0.06) * 200 / 0.0254 - 1e-9) and h == math.ceil((0.15 + 0.06) * 200 / 0.0254 - 1e-9)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert img.shape == (h, w)
    det = D.detect(img, b)
    assert det is not None and det.n == b.n_points
    # the printed corners sit where board_points says, at the file's scale (pixel centres at integers)
    s = 200 / 0.0254
    expect = (b.points()[det.ids, :2] + 0.03) * s - 0.5
    assert np.abs(det.corners - expect).max() < 0.15


def test_aprilgrid_png_is_detectable(tmp_path):
    import cv2
    g = B.AprilGrid(4, 3, 0.03)
    path = B.save_board(g, str(tmp_path / 'grid.png'), dpi=150)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    det = D.detect(img, g)
    assert det is not None and det.n == g.n_points and det.markers == 12
    s = 150 / 0.0254
    page = g.board_to_page(g.points()[det.ids])
    expect = (page + 0.03) * s - 0.5
    assert np.abs(det.corners - expect).max() < 0.3


def test_pdf_page_is_the_board_at_true_size(tmp_path):
    g = B.AprilGrid(6, 6, 0.088, 0.3)                      # Kalibr's wiki example
    path = B.save_board(g, str(tmp_path / 'grid.pdf'))
    data = open(path, 'rb').read()
    assert data.startswith(b'%PDF-1.4') and data.rstrip().endswith(b'%%EOF')
    box = [float(v) for v in re.search(rb'/MediaBox \[([^\]]+)\]', data).group(1).split()]
    x0, y0, x1, y1 = g.page_extent()
    pt = 72 / 0.0254
    assert box[2] == pytest.approx((x1 - x0 + 2 * 0.088) * pt, abs=1e-3)
    assert box[3] == pytest.approx((y1 - y0 + 2 * 0.088) * pt, abs=1e-3)
    assert data.count(b' re f') == len(g.page_rects())
    # the cross-reference table points at each object
    xref = int(re.search(rb'startxref\n(\d+)', data).group(1))
    offs = [int(m) for m in re.findall(rb'(\d{10}) 00000 n', data[xref:])]
    for i, o in enumerate(offs, 1):
        assert data[o:o + 12].startswith(f'{i} 0 obj'.encode())
    assert b'Print at 100 %' in data


def test_save_board_refuses_other_formats(tmp_path):
    with pytest.raises(B.BoardError, match='.png or .pdf'):
        B.save_board(B.AprilGrid(2, 2, 0.03), str(tmp_path / 'x.svg'))


# =========================================================================== detection and PnP
def _check_view(board, cam, T, max_rms=0.2):
    img = S.render_view(board, cam, T, noise=1.0, seed=3)
    det, pnp = D.detect_and_solve(img, board, cam)
    assert det is not None and pnp is not None
    P = det.points @ T[:3, :3].T + T[:3, 3]
    uv, _ = cam.project(P)
    e = np.linalg.norm(det.corners - uv, axis=1)
    assert np.sqrt(np.mean(e ** 2)) < max_rms
    t, r = axxb.transform_error(pnp.T_cam_board, T)
    assert t < 1e-3 and math.degrees(r) < 0.2
    assert pnp.rms_px == pytest.approx(np.sqrt(np.mean(pnp.errors_px ** 2)))
    return det, pnp


def test_charuco_on_a_rendered_view():
    b = B.CharucoBoard(9, 7, 0.03, 0.022)
    cam = S.synthetic_camera()
    det, pnp = _check_view(b, cam, S.look_at_board(0.45, (25.0, -15.0), (0.05, 0.02), 20.0, board=b))
    assert det.n == 48 and pnp.n == 48
    json.dumps(det.to_dict())
    json.dumps(pnp.to_dict())


def test_aprilgrid_on_a_rendered_view():
    g = B.AprilGrid(6, 5, 0.04)
    _check_view(g, S.synthetic_camera(), S.look_at_board(0.5, (-20.0, 25.0), board=g))


def test_detection_and_pnp_through_a_fisheye():
    """A Kannala-Brandt lens: the board near the image edge, 50° off the axis, where a pinhole PnP would fail."""
    from dexvantage.core.optics import CameraModel
    cam = CameraModel(800, 800, 260.0, 260.0, 399.5, 399.5, 'kb', (0.02, -0.01, 0.002, -0.0005))
    b = B.CharucoBoard(7, 5, 0.04, 0.03)
    T = S.look_at_board(0.4, (0.0, 35.0), (math.tan(math.radians(40)), 0.0), board=b)
    _check_view(b, cam, T, max_rms=0.3)


def test_partly_visible_charuco_keeps_the_right_ids():
    b = B.CharucoBoard(9, 7, 0.03, 0.022)
    cam = S.synthetic_camera()
    T = S.look_at_board(0.35, (0.0, 0.0), (0.45, 0.0), board=b)          # the right part is off the image
    img = S.render_view(b, cam, T)
    det = D.detect(img, b, cam)
    assert det is not None and 6 <= det.n < 48
    uv, _ = cam.project(det.points @ T[:3, :3].T + T[:3, 3])
    assert np.abs(det.corners - uv).max() < 0.5


def test_nothing_found_on_a_blank_image_and_bad_input_said():
    assert D.detect(np.full((240, 320), 128, np.uint8), B.CharucoBoard(5, 4, 0.03, 0.02)) is None
    assert D.detect(np.full((240, 320, 3), 200, np.uint8), B.AprilGrid(3, 3, 0.03)) is None
    with pytest.raises(D.DetectError, match='grey or colour'):
        D.to_gray(np.zeros((4, 4, 2), np.uint8))
    assert D.to_gray(np.ones((3, 3), float)).max() == 255
    assert D.solve_pnp(np.zeros((3, 3)), np.zeros((3, 2)), S.synthetic_camera()) is None


# =========================================================================== closed-form solvers
def _rand_T(rng, s=1.0):
    from scipy.spatial.transform import Rotation
    return axxb.T_of(Rotation.from_rotvec(rng.normal(size=3) * s).as_matrix(), rng.normal(size=3) * 0.3)


@pytest.mark.parametrize('method', axxb.HAND_EYE_METHODS)
def test_hand_eye_methods_are_exact_on_exact_motions(method):
    rng = np.random.default_rng(4)
    X, Y = _rand_T(rng), _rand_T(rng)                    # camera in gripper, board in base
    G = [_rand_T(rng) for _ in range(8)]
    C = [axxb.inv_T(X) @ axxb.inv_T(g) @ Y for g in G]
    t, r = axxb.transform_error(axxb.hand_eye_numpy(G, C, method), X)
    assert t < 1e-12 and r < 1e-12


def test_eye_to_hand_with_inverted_motions_gives_the_camera_in_the_base():
    """OpenCV's documented use for a fixed camera: pass base-to-gripper; the result is camera-to-base."""
    rng = np.random.default_rng(5)
    Xc, Zb = _rand_T(rng), _rand_T(rng)                  # camera in base, board in gripper
    Gs = [_rand_T(rng) for _ in range(8)]                # gripper in base
    C = [axxb.inv_T(Xc) @ g @ Zb for g in Gs]
    for m in axxb.HAND_EYE_METHODS:
        t, r = axxb.transform_error(axxb.hand_eye_numpy([axxb.inv_T(g) for g in Gs], C, m), Xc)
        assert t < 1e-10 and r < 1e-10, m


@pytest.mark.parametrize('method', axxb.ROBOT_WORLD_METHODS)
def test_robot_world_methods_are_exact(method):
    rng = np.random.default_rng(6)
    X, Z = _rand_T(rng), _rand_T(rng)
    Bs = [_rand_T(rng) for _ in range(8)]
    A = [Z @ b @ axxb.inv_T(X) for b in Bs]              # A X = Z B
    Xe, Ze = axxb.robot_world_numpy(A, Bs, method)
    assert axxb.transform_error(Xe, X)[0] < 1e-12 and axxb.transform_error(Ze, Z)[0] < 1e-12


def test_closed_form_refuses_degenerate_motions_in_words():
    rng = np.random.default_rng(7)
    X, Y = _rand_T(rng), _rand_T(rng)
    with pytest.raises(axxb.SolverError, match='at least 3 poses'):
        axxb.hand_eye_numpy([np.eye(4)] * 2, [np.eye(4)] * 2)
    # rotations about one axis only
    G = [axxb.T_of(axxb.exp_so3([0, 0, a]), [a, 0, 0]) for a in (0.1, 0.5, 0.9, 1.3)]
    C = [axxb.inv_T(X) @ axxb.inv_T(g) @ Y for g in G]
    with pytest.raises(axxb.SolverError, match='two different axes'):
        axxb.hand_eye_numpy(G, C, 'park')
    with pytest.raises(axxb.SolverError, match='unknown hand-eye method'):
        axxb.hand_eye_numpy(G, C, 'magic')


def test_backend_is_reported():
    rng = np.random.default_rng(8)
    X, Y = _rand_T(rng), _rand_T(rng)
    G = [_rand_T(rng) for _ in range(5)]
    C = [axxb.inv_T(X) @ axxb.inv_T(g) @ Y for g in G]
    T, bk = axxb.hand_eye(G, C, 'park')
    assert bk == ('opencv' if axxb.opencv_has_hand_eye() else 'numpy')
    assert axxb.transform_error(T, X)[0] < 1e-9
    if not axxb.opencv_has_hand_eye():
        with pytest.raises(axxb.SolverError, match='no calibrateHandEye'):
            axxb.hand_eye_opencv(G, C)


# =========================================================================== bundle adjustment
def _within(res, truth, names, k=5.0, mm=1.0):
    for name in names:
        T = res.X.get(name, res.Z.get(name))
        C = res.cov_X.get(name, res.cov_Z.get(name))
        ref = truth['X'].get(name, truth['Z'].get(name))
        e = HE.component_errors(T, ref)
        s = np.sqrt(np.diag(C))
        assert np.all(np.abs(e) <= k * s), (name, e / s)
        assert 1000 * np.linalg.norm(e[3:]) < mm, name


@pytest.mark.parametrize('kind', ['eye_in_hand', 'eye_to_hand'])
def test_ba_recovers_the_truth(kind):
    prob, truth, _ = S.scene(kind, n_poses=15, noise_px=0.3, seed=12)
    assert prob.mode() == kind
    res = HE.solve(prob)
    assert res.ok, res.message
    _within(res, truth, list(truth['X']) + list(truth['Z']), mm=2.0)
    assert res.rms_px == pytest.approx(0.3 * math.sqrt(2), rel=0.15)
    assert res.sigma_px == pytest.approx(0.3, rel=0.15)
    assert all(v['used'] for v in res.views) and len(res.views) == 15
    chosen = [m for m in res.initial['methods'] if m.get('chosen')]
    assert len(chosen) == 1
    expect = set(axxb.HAND_EYE_METHODS) | set(axxb.ROBOT_WORLD_METHODS)
    assert {m['method'] for m in res.initial['methods']} <= expect
    assert len(res.initial['methods']) >= 6


def test_ba_two_static_cameras_and_two_arms():
    prob, truth, _ = S.scene('eye_to_hand', n_poses=12, noise_px=0.3, seed=13, two_cameras=True)
    res = HE.solve(prob)
    assert res.ok and set(res.X) == {'static', 'static 2'}
    _within(res, truth, ['static', 'static 2', 'board'], mm=2.0)
    prob, truth, _ = S.two_arm_scene(n_poses=12, noise_px=0.3, seed=14)
    assert prob.mode() == 'general'
    res = HE.solve(prob)
    assert res.ok, res.message
    _within(res, truth, ['wrist a', 'board b'])
    assert {m['method'] for m in res.initial['methods']} == {'shah', 'li'}   # AX = XB does not apply
    assert 'second arm' in res.frames['board b']


def test_ba_estimates_intrinsics():
    cam = S.synthetic_camera(model='brown', dist=(-0.12, 0.05, 0.0005, -0.0003, 0.0))
    guess = S.synthetic_camera(f=470.0, model='brown', dist=(0.0,) * 5)
    prob, truth, _ = S.scene('eye_in_hand', n_poses=20, noise_px=0.3, seed=15, camera=cam,
                             estimate_intrinsics=True, intrinsics_guess={'wrist': guess})
    res = HE.solve(prob)
    est = res.intrinsics['wrist']
    m, s = est['model'], np.array(est['sigma'])
    got = np.r_[m.fx, m.fy, m.cx, m.cy, m.dist]
    want = np.r_[cam.fx, cam.fy, cam.cx, cam.cy, cam.dist]
    assert np.all(np.abs(got - want) <= 5 * s)
    assert abs(m.fx - 480.0) < 1.0 and est['names'][:4] == ['fx', 'fy', 'cx', 'cy']
    _within(res, truth, ['wrist', 'board'])


def test_ba_joint_offsets_and_what_the_views_cannot_see():
    offs = [0.3, -0.4, 0.25, 0.5, -0.3, 0.2] + [0.0] * 8
    prob, truth, _ = S.scene('eye_in_hand', n_poses=20, noise_px=0.3, seed=16, two_cameras=True,
                             joint_offsets_deg={'ur5e': offs})
    plain = HE.solve(prob)
    res = HE.solve(prob, HE.Settings(joint_offsets=('ur5e',)))
    assert res.ok and res.rms_px < plain.rms_px
    jo = res.joint_offsets['ur5e']
    assert jo['joints'][:6] == prob.robots['ur5e'].joint_names[:6] and len(jo['joints']) == 6   # not the fingers
    # the first joint trades off against the board's yaw, the last against the camera's roll: held by the prior
    assert jo['determined'] == [False, True, True, True, True, False]
    assert jo['sigma_deg'][0] == pytest.approx(HE.JOINT_OFFSET_PRIOR_DEG, rel=0.05)
    for v, s, d, t in zip(jo['value_deg'], jo['sigma_deg'], jo['determined'], offs):
        if d:
            assert abs(v - t) <= 5 * s


def test_ba_with_encoder_noise_uses_per_pose_joints():
    prob, truth, _ = S.scene('eye_to_hand', n_poses=12, noise_px=0.3, seed=17, encoder_noise_deg=0.02)
    st = HE.Settings(encoder_sigma_deg=0.02)
    res = HE.solve(prob, st)
    assert res.ok
    _within(res, truth, ['static', 'board'])
    # the sparse pattern: each pose's joint columns touch only that pose's views, so they share column groups
    obs = prob.observations
    init = HE.initialise(prob, obs, st)
    m = HE._Model(prob, st, obs, init['X'], init['Z'], 0.3)
    n_chain = len(m.kin.chain['ur5e'])
    assert m.L.n == 12 + 12 * n_chain
    assert len(m.groups) == 12 + n_chain
    # the grouped Jacobian equals a column-by-column central difference
    x = np.random.default_rng(1).normal(0, 1e-4, m.L.n)
    J = m.jacobian(x)
    h = 1e-6
    Jf = np.stack([(m.residual(x + h * e) - m.residual(x - h * e)) / (2 * h) for e in np.eye(m.L.n)], 1)
    assert np.abs(J - Jf).max() < 1e-6 * max(1.0, np.abs(Jf).max())
    assert np.all(J[~m.S] == 0)


def test_ba_drops_outlier_views():
    prob, truth, _ = S.scene('eye_in_hand', n_poses=15, noise_px=0.3, seed=18)
    for k in (2, 9):
        o = prob.observations[k]
        o.uv = o.uv + np.array([4.0, -3.0]) + np.random.default_rng(k).normal(0, 3.0, o.uv.shape)
    res = HE.solve(prob)
    dropped = [v for v in res.views if not v['used']]
    assert {v['pose'] for v in dropped} == {prob.observations[2].pose, prob.observations[9].pose}
    assert all('dropped in round' in v['note'] for v in dropped)
    _within(res, truth, ['wrist', 'board'])
    assert '13 of 15 views used' in res.message


def test_covariance_scales_with_the_noise():
    base, truth, _ = S.scene('eye_in_hand', n_poses=12, noise_px=0.0, seed=19)
    a = HE.solve(S.with_noise(base, 0.2, 1))
    b = HE.solve(S.with_noise(base, 0.8, 1))
    ratio = np.sqrt(np.diag(b.cov_X['wrist'])) / np.sqrt(np.diag(a.cov_X['wrist']))
    assert np.all(np.abs(ratio - 4.0) < 0.6)


def test_result_is_json_with_units():
    prob, truth, _ = S.scene('eye_in_hand', n_poses=8, noise_px=0.3, seed=20)
    res = HE.solve(prob)
    d = json.loads(json.dumps(res.to_dict()))
    assert d['schema'] == 'dexvantage.handeye/1' and d['mode'] == 'eye_in_hand'
    cam = d['cameras']['wrist']
    assert set(cam['sigma']) == {'x_mm', 'y_mm', 'z_mm', 'rx_deg', 'ry_deg', 'rz_deg'}
    assert 'wrist_3_link' in cam['frame'] and np.array(cam['covariance']).shape == (6, 6)
    assert d['boards']['board']['frame'].endswith('in the world frame')
    assert d['settings']['defaults']['outlier_k'] == HE.OUTLIER_K
    assert res.sigma('wrist')['x_mm'] > 0


def test_unsolvable_problems_are_explained():
    prob, truth, _ = S.scene('eye_in_hand', n_poses=5, noise_px=0.3, seed=21)
    empty = HE.Problem(prob.cameras, prob.boards, prob.robots, prob.joints, [])
    res = HE.solve(empty)
    assert not res.ok and 'no views' in res.message
    static = HE.Problem([HE.CameraSpec('c', prob.cameras[0].camera)], prob.boards, prob.robots, prob.joints,
                        [HE.Observation('c', 'board', 0, o.ids, o.uv) for o in prob.observations])
    res = HE.solve(static)
    assert not res.ok and 'fixed in the world' in res.message
    few = HE.Problem(prob.cameras, prob.boards, prob.robots, prob.joints, prob.observations[:2])
    res = HE.solve(few)
    assert not res.ok and 'not linked' in res.message
    bad = HE.Problem(prob.cameras, prob.boards, prob.robots, prob.joints,
                     [HE.Observation('wrist', 'board', 99, prob.observations[0].ids, prob.observations[0].uv)])
    assert 'robot pose 99' in HE.solve(bad).message


# =========================================================================== validation registration
def test_handeye_checks_are_registered():
    from dexvantage.validation import suite
    suite.load_checks()
    ids = {s.id for s in suite.CHECKS if s.model == 'handeye'}
    assert {'handeye.closed_form_exact', 'handeye.closed_form_noisy', 'handeye.ba_truth', 'handeye.ba_covariance_mc',
            'handeye.axzb_two_arms', 'handeye.charuco_render', 'handeye.aprilgrid_render'} <= ids
    from dexvantage.validation.references import REFERENCES
    for s in suite.CHECKS:
        if s.model == 'handeye':
            assert all(k in REFERENCES for k in s.refs), s.id


def test_closed_form_exact_check_passes():
    from dexvantage.validation import suite
    suite.load_checks()
    spec = next(s for s in suite.CHECKS if s.id == 'handeye.closed_form_exact')
    ctx = suite.Ctx(None, None, None, suite.ValidationSettings(quick=True))
    try:
        c = suite.run_one(spec, ctx)
    finally:
        ctx.close()
    assert c.passed, (c.error, c.exception)
