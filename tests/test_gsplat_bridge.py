"""The 3D Gaussian Splatting bridge (core/gsplat_bridge.py), ported from Vantage's tests/test_gsplat_bridge.py: the
trainers' camera parameters reproduce DexVantage's lens (against OpenCV itself, whose pixel centres are at integers:
the trainers' are half a pixel further), the axis conventions on a known pose (Nerfstudio's OpenGL camera-to-world,
COLMAP's OpenCV world-to-camera), and the files of Nerfstudio and COLMAP written for a small capture.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from dexvantage.core import gsplat_bridge as GB
from dexvantage.core.frames import OPTICAL_TO_BODY
from dexvantage.core.optics import CameraModel
from dexvantage.core.sim.splat_rig import Capture, CaptureView, look_at

cv2 = pytest.importorskip('cv2', reason='OpenCV is not installed: the exported images are not read back')


def test_fisheye_params_reproduce_the_lens():
    cam = CameraModel(320, 240, 120.0, 121.0, 160.2, 118.7, 'kb', (0.03, -0.01, 0.002, 0.0))
    p = GB.camera_params(cam)
    assert p['colmap'] == 'OPENCV_FISHEYE' and p['nerfstudio'] == 'OPENCV_FISHEYE' and p['approx_px'] == 0.0
    rng = np.random.default_rng(1)
    D = rng.normal(size=(300, 3))
    D[:, 2] = np.abs(D[:, 2]) + 0.5
    uv, ok = cam.project(D)
    K = np.array([[p['fx'], 0, p['cx'] - 0.5], [0, p['fy'], p['cy'] - 0.5], [0, 0, 1]])
    pts, _ = cv2.fisheye.projectPoints(D.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K,
                                       np.array([p['k1'], p['k2'], p['k3'], p['k4']]))
    assert np.abs(pts[ok, 0] - uv[ok]).max() < 1e-6


def test_brown_params_reproduce_the_lens():
    cam = CameraModel(320, 240, 260.0, 255.0, 161.0, 118.0, 'brown', (-0.2, 0.05, 0.001, 0.0005, 0.0))
    p = GB.camera_params(cam)
    assert p['colmap'] == 'OPENCV' and len(p['colmap_params']) == 8 and p['nerfstudio'] == 'OPENCV'
    D = np.array([[0.1, -0.05, 1.0], [-0.3, 0.2, 1.0]])
    uv, _ = cam.project(D)
    K = np.array([[p['fx'], 0, p['cx'] - 0.5], [0, p['fy'], p['cy'] - 0.5], [0, 0, 1]])
    ref, _ = cv2.projectPoints(D, np.zeros(3), np.zeros(3), K, np.array([p['k1'], p['k2'], p['p1'], p['p2'],
                                                                           p['k3']]))
    assert np.abs(ref[:, 0] - uv).max() < 1e-6
    rational = CameraModel(320, 240, 260.0, 255.0, 161.0, 118.0, 'brown', (-0.2, 0.05, 0, 0, 0.01, 0.1, 0, 0))
    q = GB.camera_params(rational)
    assert q['colmap'] == 'FULL_OPENCV' and len(q['colmap_params']) == 12 and q['nerfstudio'] is None
    es = GB.camera_params(CameraModel(320, 240, 100, 100, 159.5, 119.5, 'equisolid'))
    assert es['colmap'] == 'OPENCV_FISHEYE' and 0 < es['approx_px'] < 2.0


def test_axis_conventions_on_a_known_pose():
    T = np.eye(4)
    T[:3, :3] = np.c_[[-1, 0, 0], [0, 0, -1], [0, -1, 0]]           # looking along world −y, level: x right, y down
    T[:3, 3] = (1.0, 2.0, 3.0)
    assert np.allclose(T[:3, :3] @ [0, 0, 1], [0, -1, 0])                        # optical z (forward) = world −y
    G = GB.c2w_opengl(T)
    assert np.allclose(-G[:3, 2], [0, -1, 0]) and np.allclose(G[:3, 1], [0, 0, 1])   # looks down −Z, +Y up
    assert np.allclose(G[:3, 3], [1, 2, 3])
    R, t = GB.w2c_opencv(T)
    assert np.allclose(R @ [1.0, -3.0, 3.0] + t, [0, 0, 5], atol=1e-12)           # 5 m straight ahead
    assert np.allclose(GB.c2w_opengl(np.eye(4))[:3, :3], np.diag([1, -1, -1]))
    assert np.allclose(OPTICAL_TO_BODY @ [0, 0, 1], [1, 0, 0])


def _capture(n=4, cam=None):
    cam = cam or CameraModel(16, 12, 14.0, 14.0, 7.5, 5.5)
    views = []
    for k in range(n):
        a = 2 * np.pi * k / n
        T = look_at([np.cos(a), np.sin(a), 0.5], [0, 0, 0.1])
        rgb = np.full((12, 16, 3), 40 * k, np.uint8)
        views.append(CaptureView(f'v{k}', rgb, cam, T, {'r1': np.zeros(2)}))
    return Capture(views, [], True, {'kind': 'test'})


def test_export_files_parse(tmp_path):
    cap = _capture()
    paths = GB.export_nerfstudio(cap, str(tmp_path))
    assert all(p.startswith(str(tmp_path)) for p in paths)
    doc = json.loads((tmp_path / 'nerfstudio' / 'transforms.json').read_text())
    assert doc['camera_model'] == 'OPENCV' and len(doc['frames']) == 4 and doc['dexvantage']['synthetic'] is True
    for k in ('fl_x', 'fl_y', 'cx', 'cy', 'w', 'h', 'k1', 'k2', 'k3', 'k4', 'p1', 'p2', 'transform_matrix'):
        assert all(k in f for f in doc['frames']), k
    f = doc['frames'][1]
    assert f['cx'] == pytest.approx(8.0) and (tmp_path / 'nerfstudio' / f['file_path']).is_file()
    G = np.array(f['transform_matrix'])
    assert np.allclose(G, GB.c2w_opengl(cap.views[1].T_world_optical), atol=1e-8)
    img = cv2.imread(str(tmp_path / 'nerfstudio' / f['file_path']))
    assert img.shape == (12, 16, 3) and int(img[0, 0, 0]) == 40
    pts = np.array([[0.0, 0.0, 0.1], [0.05, 0.0, 0.1]])
    GB.export_colmap(cap, str(tmp_path), pts, np.array([[1.0, 0, 0], [0, 1.0, 0]]))
    sp = tmp_path / 'colmap' / 'sparse' / '0'
    cams = [ln for ln in (sp / 'cameras.txt').read_text().splitlines() if ln and not ln.startswith('#')]
    assert len(cams) == 4 and all(ln.split()[1] == 'PINHOLE' and len(ln.split()) == 8 for ln in cams)
    imgs = [ln for ln in (sp / 'images.txt').read_text().splitlines() if not ln.startswith('#')]
    data = [ln for ln in imgs[::2] if ln]
    assert len(data) == 4 and all(len(ln.split()) == 10 for ln in data)
    w, x, y, z, tx, ty, tz = (float(v) for v in data[0].split()[1:8])
    from scipy.spatial.transform import Rotation
    R = Rotation.from_quat([x, y, z, w]).as_matrix()
    Pc = R @ pts[0] + [tx, ty, tz]
    uv, ok = cap.views[0].camera.project(Pc[None])
    assert ok[0] and np.allclose(uv[0], [7.5, 5.5], atol=1e-6)                   # the look-at target, centred
    p3 = [ln for ln in (sp / 'points3D.txt').read_text().splitlines() if ln and not ln.startswith('#')]
    assert len(p3) == 2 and p3[0].split()[4:7] == ['255', '0', '0']


def test_mixed_lenses_are_refused_for_nerfstudio(tmp_path):
    cap = _capture(2)
    cap.views[1].camera = CameraModel(16, 12, 10.0, 10.0, 7.5, 5.5, 'kb', (0.0,) * 4)
    with pytest.raises(ValueError, match='mixes perspective and fisheye'):
        GB.export_nerfstudio(cap, str(tmp_path), write_images=False)


def test_gsplat_version_is_reported_or_none():
    v = GB.gsplat_version()
    assert v is None or isinstance(v, str)
