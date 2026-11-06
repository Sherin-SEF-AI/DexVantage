"""Board corner detection, sub-pixel refinement and PnP with the per-view reprojection RMS.

Ported from Vantage core/calib.py (find_board's grey conversion, _pose_fisheye's PnP on lens-undistorted rays and its
per-view reprojection error), with the chessboard search replaced by the ChArUco and AprilGrid detectors of OpenCV's
aruco module and the vehicle-camera code (unit records, the board pre-check, the rectified fisheye search) removed.

Detection:
- ChArUco: cv2.aruco.CharucoDetector.detectBoard (markers found, then the chessboard corners between them
  interpolated and refined by OpenCV's cornerSubPix). A second cornerSubPix pass was tried and left out: on rendered
  boards (validation/checks_handeye.py's scene) it made the corners worse at 25–40° of tilt (0.09 → 0.10–0.13 px).
- AprilGrid: cv2.aruco.ArucoDetector with the DICT_APRILTAG_36h11 dictionary, markerBorderBits = 2 (Kalibr's 2-bit
  black border) and contour corner refinement, then cv2.cornerSubPix at each tag corner with a half-window of
  SUBPIX_GAP_FRACTION of the gap between tags in pixels (2 to SUBPIX_MAX_HALF px; both defaults), so the window stays
  inside the gap. With Kalibr's corner squares the tag corners are chessboard-like X corners, which cornerSubPix
  locates without bias (measured on rendered grids with 1 grey level of noise: 0.08–0.12 px RMS); without them they
  are L corners (0.13–0.26 px). OpenCV's CORNER_REFINE_APRILTAG is not used: on the same renders with OpenCV 5.0.0
  its corners sat about +0.5 px off in both axes (a half-pixel convention difference), and it lost tags next to the
  corner squares. A corner that the sub-pixel step moves by more than SUBPIX_MAX_SHIFT px (a default) keeps the
  detector's position. A view needs APRILGRID_MIN_TAGS tags (Kalibr's default 4); a ChArUco view needs MIN_POINTS
  corners.

PnP (solve_pnp): the corners are turned into rays by the camera model (any core.optics model, so fisheye lenses
work), the planar board pose is solved on the normalised plane z = 1 (cv2.solvePnPGeneric with IPPE, the solution
with the smaller reprojection error kept), refined by Levenberg-Marquardt on that plane (cv2.solvePnPRefineLM), then
once more in pixels through the camera model (scipy.optimize.least_squares), so the reported error is the pixel
reprojection error the hand-eye solver also minimises: per corner |uv_detected − project(T · P)| and the view's RMS
√(mean e²).

Known limits: corners must lie in front of the camera (a ray at 90° or more off the axis of a wide fisheye has no
point on the normalised plane and is left out of the initial pose; it is used in the pixel refinement). OpenCV's
interpolation of ChArUco corners uses no lens model unless the camera is a pinhole or Brown-Conrady model (then its
K and D are passed); a strongly distorted fisheye view may lose corners near the image edge. The detector's own
sub-pixel accuracy is measured on rendered images by validation/checks_handeye.py, not on real prints.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .boards import APRILGRID_BORDER_BITS, APRILGRID_DICT, APRILGRID_MIN_TAGS, AprilGrid, Board, CharucoBoard, \
    get_dictionary

MIN_POINTS = 6              # a default: corners a ChArUco view needs to count (PnP needs 4; 6 leaves a check)
SUBPIX_GAP_FRACTION = 0.33  # a default: AprilGrid cornerSubPix half-window as a share of the gap between tags, px
SUBPIX_MAX_HALF = 8         # a default: px, the largest half-window
SUBPIX_MAX_SHIFT = 2.5      # a default: px, a larger move by the sub-pixel step is not accepted


class DetectError(ValueError):
    """An image that cannot be looked at (wrong type or shape); said in plain words."""


def to_gray(img) -> np.ndarray:
    """A uint8 grey image from grey, RGB or RGBA input (uint8, or float in [0, 1])."""
    import cv2
    a = np.asarray(img)
    if a.dtype != np.uint8:
        if a.dtype.kind == 'f':
            a = np.clip(np.nan_to_num(a) * (255.0 if a.max(initial=0) <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        else:
            a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        return a
    if a.ndim == 3 and a.shape[2] == 1:
        return a[..., 0]
    if a.ndim == 3 and a.shape[2] == 3:
        return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    if a.ndim == 3 and a.shape[2] == 4:
        return cv2.cvtColor(a, cv2.COLOR_RGBA2GRAY)
    raise DetectError(f'an image must be grey or colour (H × W, H × W × 3 or × 4), got shape {a.shape}')


@dataclass
class Detection:
    """The board corners found in one image: point ids into board.points(), pixel positions (OpenCV convention:
    pixel centres at integers)."""
    board: Board
    ids: np.ndarray
    corners: np.ndarray
    image_size: tuple                       # (width, height)
    markers: int = 0                        # markers (ChArUco) or tags (AprilGrid) decoded
    note: str = ''

    @property
    def n(self) -> int:
        return int(len(self.ids))

    @property
    def points(self) -> np.ndarray:
        return self.board.points()[self.ids]

    def to_dict(self) -> dict:
        return {'board': self.board.to_dict(), 'ids': [int(i) for i in self.ids],
                'corners': [[float(u), float(v)] for u, v in self.corners], 'image_size': list(self.image_size),
                'markers': int(self.markers), 'note': self.note}


def _subpix(gray, corners, half: np.ndarray) -> np.ndarray:
    import cv2
    out = corners.astype(np.float32).copy()
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 60, 1e-4)
    for h in np.unique(half):
        sel = half == h
        pts = out[sel].reshape(-1, 1, 2).copy()
        cv2.cornerSubPix(gray, pts, (int(h), int(h)), (-1, -1), crit)
        out[sel] = pts.reshape(-1, 2)
    moved = np.linalg.norm(out - corners, axis=1) > SUBPIX_MAX_SHIFT
    out[moved] = corners[moved]
    return out.astype(np.float64)


def detect_charuco(gray: np.ndarray, board: CharucoBoard, camera=None) -> Detection | None:
    import cv2
    cp = cv2.aruco.CharucoParameters()
    if camera is not None and camera.model in ('pinhole', 'brown'):
        K, D, _ = camera.to_opencv()
        cp.cameraMatrix = K
        cp.distCoeffs = np.asarray(D, float).reshape(1, -1)
    dp = cv2.aruco.DetectorParameters()
    det = cv2.aruco.CharucoDetector(board.cv(), cp, dp)
    cc, cid, mc, mid = det.detectBoard(gray)
    nm = 0 if mid is None else len(mid)
    if cid is None or len(cid) < MIN_POINTS:
        return None
    ids = np.asarray(cid).ravel().astype(int)
    uv = np.asarray(cc, float).reshape(-1, 2)
    order = np.argsort(ids)
    ids, uv = ids[order], uv[order]
    h, w = gray.shape
    return Detection(board, ids, uv, (w, h), nm)


def detect_aprilgrid(gray: np.ndarray, board: AprilGrid, refine: bool = True) -> Detection | None:
    import cv2
    dp = cv2.aruco.DetectorParameters()
    dp.markerBorderBits = APRILGRID_BORDER_BITS
    dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    det = cv2.aruco.ArucoDetector(get_dictionary(APRILGRID_DICT), dp)
    corners, mid, _rej = det.detectMarkers(gray)
    if mid is None:
        return None
    ids, uv, half = [], [], []
    tags = 0
    for c, i in zip(corners, np.asarray(mid).ravel()):
        try:
            pid = board.tag_point_ids(int(i))
        except KeyError:
            continue                                      # a tag that is not part of this grid
        tags += 1
        q = np.asarray(c, float).reshape(4, 2)
        side = float(np.mean(np.linalg.norm(q - np.roll(q, 1, axis=0), axis=1)))
        ids.append(pid)
        uv.append(q)
        half += [int(np.clip(math.floor(SUBPIX_GAP_FRACTION * board.spacing * side), 2, SUBPIX_MAX_HALF))] * 4
    if tags < APRILGRID_MIN_TAGS:
        return None
    ids = np.concatenate(ids)
    uv = np.concatenate(uv)
    if refine:
        uv = _subpix(gray, uv, np.array(half))
    order = np.argsort(ids)
    h, w = gray.shape
    return Detection(board, ids[order], uv[order], (w, h), tags)


def detect(image, board: Board, camera=None, refine: bool = True) -> Detection | None:
    """The board's corners in an image (None when too few are found: MIN_POINTS ChArUco corners, or
    APRILGRID_MIN_TAGS tags). camera: the lens model (optional; helps OpenCV interpolate ChArUco corners on a
    distorted pinhole or Brown-Conrady lens). refine: the AprilGrid sub-pixel step (ChArUco corners are always
    OpenCV's refined ones)."""
    gray = to_gray(image)
    if isinstance(board, CharucoBoard):
        return detect_charuco(gray, board, camera)
    if isinstance(board, AprilGrid):
        return detect_aprilgrid(gray, board, refine)
    raise DetectError(f'unknown board type {type(board).__name__}')


# =========================================================================== PnP
@dataclass
class PnPResult:
    """A board pose from one view: T_cam_board (4 × 4, board frame → camera optical frame), the per-corner
    reprojection errors (px) and their RMS."""
    T_cam_board: np.ndarray
    rms_px: float
    errors_px: np.ndarray
    n: int
    note: str = ''
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        from ..frames import Pose
        return {'pose_cam_board': Pose.from_matrix(self.T_cam_board).to_dict(), 'rms_px': float(self.rms_px),
                'max_px': float(np.max(self.errors_px)) if len(self.errors_px) else None, 'n': int(self.n),
                'note': self.note}


def _T(rvec, tvec) -> np.ndarray:
    import cv2
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]
    T[:3, 3] = np.asarray(tvec, float).ravel()
    return T


def reprojection_errors(camera, T_cam_board, points, corners) -> np.ndarray:
    """Per corner pixel distance between the detected corner and the board point projected through the camera
    (inf where the projection is undefined)."""
    P = np.asarray(points, float) @ T_cam_board[:3, :3].T + T_cam_board[:3, 3]
    uv, _ok = camera.project(P)
    e = np.linalg.norm(uv - np.asarray(corners, float), axis=1)
    return np.where(np.isfinite(e), e, np.inf)


def solve_pnp(points, corners, camera) -> PnPResult | None:
    """The board pose from board points (N, 3) and their pixels (N, 2) through a core.optics.CameraModel; None when
    fewer than 4 corners have a ray or the solve fails."""
    import cv2
    from scipy.optimize import least_squares
    P = np.asarray(points, float).reshape(-1, 3)
    uv = np.asarray(corners, float).reshape(-1, 2)
    rays = camera.unproject(uv)
    good = np.isfinite(rays).all(axis=1) & (rays[:, 2] > 1e-6)
    if good.sum() < 4:
        return None
    xn = rays[good, :2] / rays[good, 2:3]
    obj = P[good].astype(np.float64)
    cands = []
    try:
        n, rvs, tvs, _ = cv2.solvePnPGeneric(obj, xn, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
        cands += list(zip(rvs, tvs))
    except cv2.error:
        pass
    if not cands:
        ok, rv, tv = cv2.solvePnP(obj, xn, np.eye(3), None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        cands = [(rv, tv)]
    best = None
    for rv, tv in cands:
        try:
            rv, tv = cv2.solvePnPRefineLM(obj, xn, np.eye(3), None, rv.copy(), tv.copy())
        except cv2.error:
            pass
        e = reprojection_errors(camera, _T(rv, tv), P, uv)
        s = float(np.sqrt(np.mean(np.minimum(e, 1e6) ** 2)))
        if best is None or s < best[0]:
            best = (s, np.asarray(rv, float).ravel(), np.asarray(tv, float).ravel())
    _, rv, tv = best

    def res(x):
        T = _T(x[:3], x[3:])
        Q = P @ T[:3, :3].T + T[:3, 3]
        p, _ = camera.project(Q)
        r = (p - uv).ravel()
        return np.where(np.isfinite(r), r, 1e3)
    try:
        sol = least_squares(res, np.r_[rv, tv], method='lm', xtol=1e-12, ftol=1e-12)
        x = sol.x
    except Exception:                                    # noqa: BLE001 — keep the OpenCV pose
        x = np.r_[rv, tv]
    T = _T(x[:3], x[3:])
    e = reprojection_errors(camera, T, P, uv)
    if not np.isfinite(e).all():
        return None
    return PnPResult(T, float(np.sqrt(np.mean(e ** 2))), e, int(len(e)))


def detect_and_solve(image, board: Board, camera, refine: bool = True) -> tuple:
    """(Detection or None, PnPResult or None) for one image."""
    det = detect(image, board, camera, refine)
    if det is None:
        return None, None
    return det, solve_pnp(det.points, det.corners, camera)


def rms(errors) -> float:
    e = np.asarray(errors, float)
    return float(math.sqrt(np.mean(e ** 2))) if len(e) else float('nan')
