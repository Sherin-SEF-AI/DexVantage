"""The per-episode checks of the dataset auditor (spec §8.2), from the recorded data alone: timestamps, pixels and
joint states. The project's plan, when it has the recorded cameras, adds what the data alone cannot say (how the
robot should move in each image, where the gripper should be); nothing is ever read from a synthetic dataset's
ground-truth sidecar here.

    requirements_of(project, given=None) -> dict          the pass/fail limits, labelled defaults filled in
    audit_episode(ds, index, req, plan=None, ...) -> dict  every check of one episode, JSON-able

Per camera of the episode (CameraStream):

1. Timing. The nominal period P = 1 / the stream's stated fps (else the median spacing). A spacing d ≥ 1.5 P between
   stored frames is a gap of round(d / P) − 1 missing frames: a dropped frame while d < usb_gap_s (default 0.1 s), a
   USB reset or stall from usb_gap_s on. Each frame's slot n_k = Σ round(d / P); the stamps are fitted as
   t_k = a + P' n_k (least squares), the measured rate is 1 / P' and the jitter the robust spread of the residuals
   (1.4826 × their median absolute deviation, MAD). Out-of-order or repeated stamps are counted. With device
   timestamps (hw_timestamps) the host-minus-device time is fitted against the device time: its slope is the clock
   drift (ppm) and its residual spread the transport jitter.
2. Duplicated frames: a frame whose pixels hash (BLAKE2b) like the previous frame's is an exact duplicate; one whose
   mean absolute change e_k = mean |I_k − I_{k−1}| is under dup_ratio (default 0.3) of the local median change (the
   eight frames around it, itself and the next left out: the next carries the double step) while that median is at
   least DUP_MIN_DN above the episode's still-scene floor (its 5th percentile of e) is a near duplicate: a stale
   frame re-encoded by a lossy codec is not bit-identical, so equality alone misses it.
3. Exposure and gain steps: per pair of frames the median over pixels (both values in 16…200, every second pixel) of
   ln((I_k + 1) / (I_{k−1} + 1)): a global gain multiplies every pixel, so the median is ln(gain) while the moving
   part of the scene stays a minority. |median| > ln(1 + exposure_step) (default 15 %) is a step; a step is paired
   with a later one of opposite sign into a run of changed frames. Pairs across a timestamp gap are not tested.
   When the episode records an exposure or gain column for the camera (Episode.meta['columns'], a name containing
   'exposure' or 'gain' and the camera's name), its relative changes over exposure_step are used instead.
4. Motion blur: the variance of the Laplacian (Pech-Pacheco et al. 2000), normalised by the mean squared intensity
   (so a gain change does not read as blur), over the pixels that move (|I_{k+1} − I_{k−1}| > MOTION_DN, dilated
   3 × 3), against the same over the two neighbouring frames (k ± 1) and over the next two out (k ± 2), the smaller
   of the two means: a ratio under blur_ratio (default 0.85) is a blurred frame. The second pair is there because
   LeRobot's encoder settings (g = 2) make every other frame a keyframe, and the frames between keep less fine detail
   (measured: about 0.6 of their neighbours' Laplacian energy on synthetic episodes), so a frame is compared with the
   frames of its own kind too. Cameras on the robot are not tested by default (blur_on_moving_cameras): their whole
   view changes from frame to frame, and the codec's loss of detail on fast motion reads as blur (measured on
   synthetic wrist cameras: dozens of false alarms per episode). Frames without enough moving pixels
   (MOTION_MIN_FRAC of the image) are not tested; the joint speed at each blurred frame is reported (blur while the
   arm is still points at focus or exposure, not motion).
5. Camera-joint desync: the time offset δ that makes the image motion line up with the joints'. Observed: e_k (2.,
   over the pairs not flagged by 1–4). Predicted, with the plan: the linearised brightness-constancy change at each
   sample point of the robot's surface (and, for a camera on a link, of the static scene), |∇I_{k−1}(u) · Δu|, Δu the
   point's image displacement between the two frames' times shifted by δ, the robot posed at the recorded joint
   states (interpolated linearly) through the plan's camera; averaged over the points. Without a plan: the joint
   speed |q̇| at the pair's mid time + δ times the spacing. δ maximises Spearman's rank correlation of observed and
   predicted over a grid (4 ms steps within ±max_lag_s), refined by a parabola through the peak: a rank
   correlation needs only a monotone relation between the two, which is the robust fit (frame differences saturate
   and depend on texture, so a linear fit would be biased). σ_δ is the spread of the peak over a moving-block
   bootstrap (Künsch 1989; blocks of 10 pairs, 40 resamples). Drift: δ(t) = δ0 + d (t − t̄) fitted jointly, the
   pair (δ0, d) that maximises the same rank correlation over a grid (d within ±30 000 ppm in 1000 ppm steps), σ_d by
   the same bootstrap on the grid around the peak, but at least σ_δ √12 / T (T the episode's span: a drift cannot be
   known better than the offset itself over the episode; on clean 8 s synthetic episodes the bootstrap alone said
   about 1700 ppm where the estimates scattered by about 5000 ppm: the errors are the model's, not the noise's).
   The method is time-delay estimation by cross-correlation (Knapp and
   Carter 1976) with the rank correlation in place of the product moment.
6. Gripper not visible (cameras on the robot whose view the plan says holds the gripper): the plan's gripper mask
   (core/audit/plan.py: core/camrender's z-buffer of the gripper links alone, at the recorded joint state, through
   the plan's lens and mount) is compared with the image inside it: the zero-mean normalised cross-correlation of the
   frame's pixels in the mask with the median of the frames around it (±GRIP_HALF frames; the gripper is fixed in a
   wrist camera's view, so the median is its appearance). NCC under gripper_ncc (default 0.5) means something else
   is there: the gripper is hidden. Frames where the mask covers less than GRIP_MIN_FRAC of the image are reported
   as "out of view by the plan", not tested. NCC is gain-invariant, so an exposure step does not read as hidden.

Per episode: action-observation alignment, the row lag k (−MAX_LAG_ROWS … +MAX_LAG_ROWS, refined by a parabola)
minimising the RMS of action[t] − q[t + k] over the joints both name (by name, else by order when the counts agree);
a recorded action is normally the command for a later state, so k ≥ 0; the requirement is a lag within
action_lag_rows (default 0 … 3 rows).

Every requirement is a labelled default the user can change (project.data['requirements']); each verdict is
pass / fail with the value and the limit. An episode that fails only on what can be corrected afterwards (a constant
camera offset or a drift: re-stamp the frames) is 'fix'; any other failure is 'drop' (drop it or re-record it).

Known limits: duplicates of a still scene cannot be told from stillness (they are only tested while the scene
changes); LeRobot stores camera timestamps on the nominal grid (its frames are decoded at the rows' times), so drops,
gaps and jitter in a LeRobot recording that were resampled away before writing do not show (a resampled drop shows as
a duplicate); the desync estimate needs motion that varies in speed; on synthetic textureless scenes it was measured
within about ±12 ms of the truth (docs/models/audit.md), so offsets under about 25 ms are not resolved; the drift is
only resolved when it moves the offset by more than its σ within one episode (real oscillators drift tens of ppm,
which an episode cannot show; across a campaign the per-episode offsets can); motion blur is relative to the
neighbouring frames, so a whole episode blurred alike is not flagged; the gripper check needs the plan's mount and
lens to be close to the real ones (a mount slip moves the mask); exposure from pixels needs most of the image to be
static between two frames.
"""
from __future__ import annotations

import hashlib
import math
import time

import numpy as np

from ..jobs import StudyError

# ---------------------------------------------------------------- requirements (every one a labelled default)
REQUIREMENTS = {
    # id: (default, unit, title, what is compared)
    'max_dropped_frames': (0, 'frames', 'Dropped frames', 'dropped frames per camera'),
    'max_duplicate_frames': (0, 'frames', 'Duplicated frames', 'duplicated frames per camera'),
    'fps_tolerance_pct': (1.0, '%', 'Frame rate', 'measured frame rate against the stated one'),
    'max_jitter_ms': (1.0, 'ms', 'Timestamp jitter', 'robust spread of the stamps about a regular grid'),
    'max_gaps': (0, 'gaps', 'USB resets and gaps', 'gaps of usb_gap_s or longer per camera'),
    'max_offset_ms': (25.0, 'ms', 'Camera-joint offset', '|offset| of the image motion against the joints'),
    'max_drift_ppm': (2000.0, 'ppm', 'Clock drift', '|drift| of the offset within the episode'),
    'max_exposure_changes': (0, 'steps', 'Exposure and gain steps', 'automatic exposure or gain steps per camera'),
    'max_blur_frames': (0, 'frames', 'Motion-blurred frames', 'blurred frames per camera'),
    'max_gripper_hidden_frames': (0, 'frames', 'Gripper hidden', 'frames where the gripper is hidden in a wrist '
                                  'camera'),
    'action_lag_rows': ([0, 3], 'rows', 'Action-observation alignment', 'row lag of the action against the state'),
    'max_mount_shift_deg': (0.5, '°', 'Mount slip (rotation)', 'rotation of a mount from its first estimate'),
    'max_mount_shift_mm': (2.0, 'mm', 'Mount slip (position)', 'shift of a mount from its first estimate'),
}
SETTINGS = {
    # detection settings (labelled defaults)
    'usb_gap_s': 0.1,             # a gap of this long or longer is a USB reset or stall, shorter a dropped frame
    'dup_ratio': 0.3,             # near duplicate: change under this share of the local median change
    'exposure_step': 0.15,        # a global brightness step over 15 %
    'blur_ratio': 0.85,           # normalised Laplacian energy under this share of the neighbours'
    'gripper_ncc': 0.5,           # gripper hidden below this correlation with its own appearance
    'max_lag_s': 0.25,            # desync searched within ± this
    'analysis_width': 320,        # frames are analysed at most this wide (grey)
    'blur_on_moving_cameras': 0,  # 1: test blur on cameras on the robot too (0: not; see the known limits)
}
FIXABLE = ('max_offset_ms', 'max_drift_ppm')
DUP_SIGMA_PX = 1.5                # frames are low-passed (Gaussian σ, px) before the change is measured
MOTION_DN = 8.0                   # a pixel moves when its grey value changes by more than this between k−1 and k+1
MOTION_MIN_FRAC = 0.002           # blur is tested on frames with at least this share of moving pixels
GRIP_HALF = 15                    # frames either side whose median is the gripper's appearance
GRIP_MIN_FRAC = 0.002             # the plan's gripper mask must cover this share of the image to be tested
SYNC_WIDTH = 160                  # the desync estimate works at this width at most
SYNC_MAX_PAIRS = 600              # … on at most this many frame pairs (spread evenly)
SYNC_GRID_HZ = 240.0              # the robot is posed on this time grid (linear between)
SYNC_BOOT = 40                    # bootstrap resamples for σ
SYNC_BLOCK = 10                   # pairs per bootstrap block
SYNC_MIN_PAIRS = 30
DUP_MIN_PX = 1.0                  # a duplicate is only tested where the plan's points move this far (px)
DUP_MIN_SPEED_FRAC = 0.1          # … or, without a plan, where the joints move this share of their top speed
SYNC_STEP_S = 0.004               # the lag grid (s)
SYNC_DRIFT_MAX_PPM = 30000.0      # drift searched within ± this (ppm)
SYNC_DRIFT_STEP_PPM = 1000.0      # … in steps of this
MAX_LAG_ROWS = 10


def requirements_of(project=None, given: dict | None = None) -> dict:
    """The requirements and detection settings: project.data['requirements'] then given over the defaults, with
    'defaults' listing the keys left at their default. Junk values raise StudyError with a sentence."""
    stored = {}
    if project is not None:
        stored = dict((getattr(project, 'data', None) or {}).get('requirements') or {})
    out, defaults = {}, []
    src = dict(stored, **(given or {}))
    for k, (d, _u, _t, _w) in REQUIREMENTS.items():
        out[k], defaults = _req_value(k, src, d, defaults)
    for k, d in SETTINGS.items():
        out[k], defaults = _req_value(k, src, d, defaults)
    unknown = [k for k in src if k not in REQUIREMENTS and k not in SETTINGS]
    if unknown:
        raise StudyError(f'There is no audit requirement called {unknown[0]}; they are '
                         f'{", ".join(list(REQUIREMENTS) + list(SETTINGS))}.')
    out['defaults'] = defaults
    return out


def _req_value(k, src, d, defaults):
    if k not in src or src[k] is None:
        return (list(d) if isinstance(d, list) else d), defaults + [k]
    v = src[k]
    if isinstance(d, list):
        if not (isinstance(v, (list, tuple)) and len(v) == 2):
            raise StudyError(f'The requirement {k} must be two numbers [low, high]; it is {v!r}.')
        try:
            lo, hi = float(v[0]), float(v[1])
        except (TypeError, ValueError):
            raise StudyError(f'The requirement {k} must be two numbers [low, high]; it is {v!r}.') from None
        if lo > hi:
            raise StudyError(f'The requirement {k} has its low end above its high end ({lo:g} > {hi:g}).')
        return [lo, hi], defaults
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise StudyError(f'The requirement {k} must be a number; it is {v!r}.') from None
    if not math.isfinite(x) or x < 0:
        raise StudyError(f'The requirement {k} must be zero or more; it is {v!r}.')
    return (int(x) if isinstance(d, int) and x == int(x) else x), defaults


# ---------------------------------------------------------------- frames
def _grey(img: np.ndarray, width: int) -> np.ndarray:
    import cv2
    g = img if img.ndim == 2 else cv2.cvtColor(np.ascontiguousarray(img[..., :3]), cv2.COLOR_RGB2GRAY)
    if g.shape[1] > width:
        h = max(2, int(round(g.shape[0] * width / g.shape[1])))
        g = cv2.resize(g, (width, h), interpolation=cv2.INTER_AREA)
    return g


def load_frames(stream, width: int, cancel=None) -> tuple[np.ndarray, list, tuple]:
    """(grey frames (K, h, w) uint8 at most width wide, BLAKE2b digests of the full-size RGB, full size (W, H))."""
    from ..datasets import DatasetError
    g, hashes, size = [], [], None
    try:
        for _i, img in stream.frames():
            if cancel is not None and cancel():
                break
            a = np.ascontiguousarray(img)
            size = (int(a.shape[1]), int(a.shape[0]))
            hashes.append(hashlib.blake2b(a.tobytes(), digest_size=12).hexdigest())
            g.append(_grey(a, width))
    except DatasetError:
        raise
    except Exception as e:                                    # noqa: BLE001 — a damaged video: say so
        raise StudyError(f'The frames of camera {stream.key} could not be decoded: {str(e).splitlines()[0]}.') \
            from None
    if not g:
        return np.zeros((0, 2, 2), np.uint8), [], size or (stream.width or 0, stream.height or 0)
    return np.stack(g), hashes, size


# ---------------------------------------------------------------- 1. timing
def timing(stream, fps_episode: float | None, req: dict) -> dict:
    ts = np.asarray(stream.timestamps, float)
    M = len(ts)
    fps_nom = stream.fps or fps_episode
    d = np.diff(ts)
    out = {'frames': int(M), 'fps_stated': float(fps_nom) if fps_nom else None, 'gaps': [], 'drops': [],
           'usb_gaps': [], 'out_of_order': int(np.sum(d <= 0)) if M > 1 else 0}
    if M < 3:
        out.update(fps_measured=None, jitter_ms=None, period_s=None, note='Too few frames to test the timing.')
        return out
    pos = d[d > 0]
    P = 1.0 / float(fps_nom) if fps_nom else float(np.median(pos))
    out['period_s'] = P
    steps = np.maximum(np.rint(d / P), 1.0)
    for k in np.flatnonzero(d >= 1.5 * P):
        miss = int(steps[k]) - 1
        g = {'frame': int(k + 1), 'after_s': float(ts[k]), 'gap_s': float(d[k]), 'missing': miss}
        out['gaps'].append(g)
        (out['usb_gaps'] if d[k] >= req['usb_gap_s'] else out['drops']).append(g)
    out['dropped_frames'] = int(sum(g['missing'] for g in out['drops']))
    out['gap_frames'] = int(sum(g['missing'] for g in out['usb_gaps']))
    n = np.concatenate([[0.0], np.cumsum(steps)])
    A = np.column_stack([np.ones(M), n])
    (a, Pm), *_ = np.linalg.lstsq(A, ts, rcond=None)
    r = ts - (a + Pm * n)
    out['fps_measured'] = float(1.0 / Pm) if Pm > 0 else None
    out['jitter_ms'] = float(1.4826 * np.median(np.abs(r - np.median(r))) * 1000.0)
    out['max_residual_ms'] = float(np.max(np.abs(r)) * 1000.0)
    hw = getattr(stream, 'hw_timestamps', None)
    if hw is not None and len(hw) == M:
        h = np.asarray(hw, float)
        y = ts - h
        B = np.column_stack([np.ones(M), h - h[0]])
        (c0, c1), *_ = np.linalg.lstsq(B, y, rcond=None)
        rr = y - B @ np.array([c0, c1])
        out['hw'] = {'drift_ppm': float(c1 * 1e6), 'offset_s': float(c0),
                     'jitter_ms': float(1.4826 * np.median(np.abs(rr - np.median(rr))) * 1000.0)}
    return out


# ---------------------------------------------------------------- 2. duplicates
def duplicates(g: np.ndarray, hashes: list, req: dict, moving: np.ndarray, gap_before: np.ndarray) -> dict:
    import cv2
    K = len(g)
    if K < 3:
        return {'exact': [], 'near': [], 'frames': [], 'energy': [], 'identical_still': 0}
    gb = np.stack([cv2.GaussianBlur(x.astype(np.float32), (0, 0), DUP_SIGMA_PX) for x in g])
    e = np.abs(np.diff(gb, axis=0)).mean(axis=(1, 2)).astype(float)      # e[k-1]: the change into frame k
    same = [k for k in range(1, K) if hashes[k] == hashes[k - 1]]
    exact = [k for k in same if moving[k] and not gap_before[k]]
    near = []
    for k in range(1, K):
        if k in exact or not moving[k] or gap_before[k]:
            continue
        nb = [e[i] for i in range(max(0, k - 5), min(len(e), k + 4)) if i not in (k - 1, k)
              and not gap_before[i + 1]]
        if len(nb) < 3:
            continue
        ref = float(np.median(nb))
        if ref > 0 and e[k - 1] < req['dup_ratio'] * ref:
            near.append(k)
    return {'exact': exact, 'near': near, 'frames': sorted(exact + near), 'energy': e.tolist(),
            'identical_still': int(len(same) - len(exact))}


# ---------------------------------------------------------------- 3. exposure
def exposure(g: np.ndarray, gap_before: np.ndarray, req: dict, columns: dict | None = None, key: str = '') -> dict:
    thr = math.log1p(req['exposure_step'])
    col = _exposure_column(columns, key)
    K = len(g)
    if col is not None:
        name, v = col
        v = np.asarray(v, float)
        rel = np.abs(np.diff(v)) / np.maximum(np.abs(v[:-1]), 1e-12)
        steps = [{'frame': int(k + 1), 'log_ratio': float(math.log(max(v[k + 1], 1e-12) / max(v[k], 1e-12)))
                  if v[k] > 0 and v[k + 1] > 0 else None} for k in np.flatnonzero(rel > req['exposure_step'])]
        return {'source': f'the recorded column {name}', 'steps': steps, 'runs': _runs(steps, len(v)),
                'log_ratio': []}
    lr = np.full(K, np.nan)
    if K >= 2:
        a = g[:-1, ::2, ::2].astype(np.float32)
        b = g[1:, ::2, ::2].astype(np.float32)
        ok = (a >= 16) & (a <= 200) & (b >= 16) & (b <= 200)
        r = np.log((b + 1.0) / (a + 1.0))
        for k in range(K - 1):
            m = ok[k]
            if gap_before[k + 1] or m.sum() < 50:
                continue
            lr[k + 1] = float(np.median(r[k][m]))
    steps = [{'frame': int(k), 'log_ratio': float(lr[k])} for k in range(1, K)
             if np.isfinite(lr[k]) and abs(lr[k]) > thr]
    return {'source': 'the frames\' brightness', 'steps': steps, 'runs': _runs(steps, K),
            'log_ratio': [None if not np.isfinite(x) else round(float(x), 4) for x in lr]}


def _exposure_column(columns, key):
    if not columns:
        return None
    from .plan import norm_name
    nk = norm_name(key)
    for name, v in columns.items():
        n = str(name).lower()
        if ('exposure' in n or 'gain' in n) and (not nk or nk in norm_name(name) or len(columns) == 1):
            arr = np.asarray(v, float)
            if arr.ndim == 1 and len(arr) > 1:
                return name, arr
    return None


def _runs(steps: list, K: int) -> list:
    """Pairs a step with the next of opposite sign: [{'start', 'end' (exclusive), 'log_ratio'}]; a step not undone
    runs to the end."""
    runs, i = [], 0
    s = [x for x in steps if x.get('log_ratio') is not None]
    while i < len(s):
        a = s[i]
        j = next((m for m in range(i + 1, len(s)) if np.sign(s[m]['log_ratio']) != np.sign(a['log_ratio'])), None)
        runs.append({'start': a['frame'], 'end': s[j]['frame'] if j is not None else K, 'log_ratio': a['log_ratio']})
        i = (j + 1) if j is not None else len(s)
    return runs


# ---------------------------------------------------------------- 4. blur
def blur(g: np.ndarray, req: dict) -> dict:
    import cv2
    K = len(g)
    ratio = np.full(K, np.nan)
    if K < 3:
        return {'frames': [], 'ratio': []}
    f = g.astype(np.float32)
    L = np.stack([cv2.Laplacian(x, cv2.CV_32F, ksize=3) for x in f])
    L2 = L * L
    I2 = f * f
    kern = np.ones((3, 3), np.uint8)
    npx = g.shape[1] * g.shape[2]
    for k in range(1, K - 1):
        M = (np.abs(f[k + 1] - f[k - 1]) > MOTION_DN).astype(np.uint8)
        if M.sum() < MOTION_MIN_FRAC * npx:
            continue
        M = cv2.dilate(M, kern).astype(bool)

        def s(i):
            return float(L2[i][M].mean()) / max(float(I2[i][M].mean()), 1e-6)
        refs = [0.5 * (s(k - 1) + s(k + 1))]
        if 2 <= k < K - 2:
            refs.append(0.5 * (s(k - 2) + s(k + 2)))
        ref = min(refs)
        if ref > 0:
            ratio[k] = s(k) / ref
    frames = [int(k) for k in range(K) if np.isfinite(ratio[k]) and ratio[k] < req['blur_ratio']]
    return {'frames': frames, 'ratio': [None if not np.isfinite(x) else round(float(x), 4) for x in ratio]}


# ---------------------------------------------------------------- 5. desync
def _interp_q(ep, t: np.ndarray) -> np.ndarray:
    tt = np.clip(t, ep.t[0], ep.t[-1])
    return np.stack([np.interp(tt, ep.t, ep.q[:, j]) for j in range(ep.q.shape[1])], axis=1)


def _ranks(x: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return rankdata(x, axis=-1)


def _spearman_rows(Pm: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Spearman's ρ of each row of Pm (L, n) with e (n,)."""
    re = _ranks(e)
    re = re - re.mean()
    rp = _ranks(Pm)
    rp = rp - rp.mean(axis=1, keepdims=True)
    den = np.sqrt((rp * rp).sum(axis=1) * (re * re).sum())
    with np.errstate(invalid='ignore', divide='ignore'):
        return (rp @ re) / np.where(den > 0, den, np.nan)


def _peak(lags: np.ndarray, rho: np.ndarray) -> tuple[float | None, float]:
    if not np.isfinite(rho).any():
        return None, float('nan')
    k = int(np.nanargmax(rho))
    x = float(lags[k])
    if 0 < k < len(lags) - 1 and np.all(np.isfinite(rho[k - 1:k + 2])):
        y0, y1, y2 = rho[k - 1], rho[k], rho[k + 1]
        den = y0 - 2 * y1 + y2
        h = (lags[k + 1] - lags[k - 1]) / 2
        if den < 0:
            x += float(np.clip(0.5 * (y0 - y2) / den, -1, 1) * h)
    return x, float(rho[k])


class _Predictor:
    """The predicted image change of frame pairs at a lag (module docstring, 5), on frames gs at the sync size."""

    def __init__(self, ep, ts, gs, cam, scale):
        self.ts, self.ep, self.cam, self.gs, self.scale = np.asarray(ts, float), ep, cam, gs, scale
        self.pairs = []
        if cam is not None:
            t0, t1 = float(ts[0]) - 0.35, float(ts[-1]) + 0.35
            self.grid0 = t0
            n = int(math.ceil((t1 - t0) * SYNC_GRID_HZ)) + 2
            tg = t0 + np.arange(n) / SYNC_GRID_HZ
            UV, OK = cam.project_points(_interp_q(ep, tg))
            OK = OK & np.isfinite(UV).all(axis=2)
            keep = OK.any(axis=0)                                    # points that are ever in the image
            self.UV = np.nan_to_num(UV[:, keep] * scale).astype(np.float32)
            self.OK = OK[:, keep]
            self.h, self.w = gs.shape[1:]
        self.v = np.linalg.norm(np.gradient(ep.q, ep.t, axis=0), axis=1) if len(ep.t) > 2 else None

    def use(self, pairs):
        """Set the frame pairs [(a, b)] the next calls predict."""
        import cv2
        self.pairs = list(pairs)
        self.a = np.array([p[0] for p in self.pairs], int)
        self.b = np.array([p[1] for p in self.pairs], int)
        if self.cam is not None:
            f = self.gs[self.a].astype(np.float32)
            self.gx = np.stack([cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3) / 8.0 for x in f])   # (n, h, w)
            self.gy = np.stack([cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3) / 8.0 for x in f])
        return self

    def _at(self, t):
        f = (np.asarray(t) - self.grid0) * SYNC_GRID_HZ
        i0 = np.clip(np.floor(f).astype(int), 0, len(self.UV) - 2)
        w = np.clip(f - i0, 0.0, 1.0)[:, None, None].astype(np.float32)
        uv = self.UV[i0] * (1 - w) + self.UV[i0 + 1] * w
        ok = self.OK[i0] & self.OK[i0 + 1]
        return uv, ok

    def _uv(self, lag):
        ua, oa = self._at(self.ts[self.a] + lag)
        ub, ob = self._at(self.ts[self.b] + lag)
        return ua, ub, oa & ob

    def __call__(self, lag: float) -> np.ndarray:
        if self.cam is None:
            mid = 0.5 * (self.ts[self.a] + self.ts[self.b]) + lag
            return np.interp(mid, self.ep.t, self.v) * (self.ts[self.b] - self.ts[self.a])
        ua, ub, ok = self._uv(lag)
        d = ub - ua
        u = np.clip(np.rint(ua[..., 0]).astype(np.int32), 0, self.w - 1)
        v = np.clip(np.rint(ua[..., 1]).astype(np.int32), 0, self.h - 1)
        n = np.arange(len(self.a))[:, None]
        gg = np.abs(self.gx[n, v, u] * d[..., 0] + self.gy[n, v, u] * d[..., 1]) * ok
        cnt = ok.sum(axis=1)
        return np.where(cnt > 0, gg.sum(axis=1) / np.maximum(cnt, 1), 0.0)

    def moving(self, K: int) -> np.ndarray:
        """(K,) True where the scene should change into frame k: the plan's sample points move at least DUP_MIN_PX
        (at the recorded size) between frames k − 1 and k, else (no plan) the joint speed is at least
        DUP_MIN_SPEED_FRAC of the episode's largest."""
        out = np.zeros(K, bool)
        if K < 2:
            return out
        a = np.arange(K - 1)
        b = a + 1
        if self.cam is not None:
            self.a, self.b = a, b
            ua, ub, ok = self._uv(0.0)
            dd = np.linalg.norm(np.where(ok[..., None], ub - ua, 0.0), axis=2)
            cnt = ok.sum(axis=1)
            disp = np.where(cnt > 0, dd.sum(axis=1) / np.maximum(cnt, 1), 0.0) / max(self.scale, 1e-9)
            out[1:] = disp >= DUP_MIN_PX
        elif self.v is not None and np.max(self.v) > 0:
            sp = np.interp(0.5 * (self.ts[a] + self.ts[b]), self.ep.t, self.v)
            out[1:] = sp >= DUP_MIN_SPEED_FRAC * float(np.max(self.v))
        return out


def sync_frames(g: np.ndarray, full) -> tuple[np.ndarray, float]:
    """The frames at the desync size (at most SYNC_WIDTH wide) and their scale from the recorded size."""
    import cv2
    if g.shape[2] > SYNC_WIDTH:
        h = max(2, int(round(g.shape[1] * SYNC_WIDTH / g.shape[2])))
        g = np.stack([cv2.resize(x, (SYNC_WIDTH, h), interpolation=cv2.INTER_AREA) for x in g])
    return g, g.shape[2] / float(full[0] or g.shape[2])


def desync(ep, stream, gs: np.ndarray, bad: np.ndarray, plan_cam, req: dict, pred, rng=None) -> dict:
    """The camera's offset against the joints (module docstring, 5)."""
    ts = np.asarray(stream.timestamps, float)
    g = gs
    K = len(g)
    method = ('the plan\'s predicted image change (linearised brightness constancy at points on the robot'
              + (' and the scene' if plan_cam is not None and plan_cam.on_link else '') + ')'
              if plan_cam is not None else 'the joint speed (no camera of the plan matches this one)')
    out = {'offset_ms': None, 'sigma_ms': None, 'drift_ppm': None, 'drift_sigma_ppm': None, 'rho': None,
           'reference': method, 'pairs': 0}
    if K < SYNC_MIN_PAIRS + 1 or len(ep.t) < 3:
        out['note'] = 'Too few frames or joint states to estimate the offset.'
        return out
    pairs = [(k - 1, k) for k in range(1, K) if not bad[k] and not bad[k - 1]]
    if len(pairs) > SYNC_MAX_PAIRS:
        sel = np.linspace(0, len(pairs) - 1, SYNC_MAX_PAIRS).round().astype(int)
        pairs = [pairs[i] for i in sel]
    if len(pairs) < SYNC_MIN_PAIRS:
        out['note'] = 'Too few clean frame pairs to estimate the offset.'
        return out
    gi = g.astype(np.int16)
    e = np.array([float(np.abs(gi[b] - gi[a]).mean()) for a, b in pairs])
    pred.use(pairs)
    L = float(req['max_lag_s'])
    lags = np.round(np.arange(-L, L + 1e-9, SYNC_STEP_S), 6)
    Pm = np.stack([pred(x) for x in lags])                       # (lags, pairs)
    rho = _spearman_rows(Pm, e)
    off, rmax = _peak(lags, rho)
    if off is None:
        out['note'] = 'The image does not change with the joints, so the offset cannot be estimated.'
        return out
    rng = rng if rng is not None else np.random.default_rng(0)
    n = len(e)

    def resample(m):
        nb = max(1, int(math.ceil(m / SYNC_BLOCK)))
        starts = rng.integers(0, max(1, m - SYNC_BLOCK + 1), nb)
        return np.concatenate([np.arange(s_, min(m, s_ + SYNC_BLOCK)) for s_ in starts])[:m]
    boot = []
    for _b in range(SYNC_BOOT):
        idx = resample(n)
        pk, _ = _peak(lags, _spearman_rows(Pm[:, idx], e[idx]))
        if pk is not None:
            boot.append(pk)
    sig = float(np.std(boot, ddof=1)) if len(boot) > 2 else None
    out.update(offset_ms=off * 1000.0, sigma_ms=None if sig is None else max(sig * 1000.0, 0.5), rho=rmax,
               pairs=int(n), curve={'lag_ms': [round(float(x) * 1000, 1) for x in lags[::2]],
                                    'rho': [None if not np.isfinite(r) else round(float(r), 4) for r in rho[::2]]})
    # drift: the offset a line in time, δ(t) = δ0 + d (t − t̄), fitted jointly over (δ0, d)
    tau = np.array([0.5 * (ts[a_] + ts[b_]) for a_, b_ in pairs])
    tau = tau - tau.mean()
    if n >= 2 * SYNC_MIN_PAIRS and np.ptp(tau) > 0:
        cols = np.arange(n)

        def joint(d_grid, l_idx, sel):
            best = (-np.inf, None, None)
            for d in d_grid:
                ii = np.clip(np.rint((lags[l_idx][:, None] + d * tau[None, sel] - lags[0]) / SYNC_STEP_S)
                             .astype(int), 0, len(lags) - 1)
                r = _spearman_rows(Pm[ii, cols[None, sel]], e[sel])
                k = int(np.nanargmax(r)) if np.isfinite(r).any() else None
                if k is not None and r[k] > best[0]:
                    best = (float(r[k]), float(lags[l_idx][k]), float(d))
            return best
        dmax = SYNC_DRIFT_MAX_PPM * 1e-6
        d_grid = np.linspace(-dmax, dmax, 2 * int(SYNC_DRIFT_MAX_PPM / SYNC_DRIFT_STEP_PPM) + 1)
        all_l = np.arange(len(lags))
        r0, l0, d0 = joint(d_grid, all_l, cols)
        if l0 is not None and abs(d0) >= dmax - 0.5 * SYNC_DRIFT_STEP_PPM * 1e-6:
            out['drift_note'] = ('The drift estimate ran to the edge of the search (±30 000 ppm): not resolved.')
            l0 = None
        if l0 is not None:
            near = all_l[np.abs(lags - l0) <= 0.012]
            dl = d_grid[np.abs(d_grid - d0) <= 6 * SYNC_DRIFT_STEP_PPM * 1e-6]
            bd = []
            for _b in range(SYNC_BOOT // 2):
                sel = resample(n)
                _r, _l, d_ = joint(dl, near, sel)
                if d_ is not None:
                    bd.append(d_)
            sd = float(np.std(bd, ddof=1)) if len(bd) > 2 else None
            out['drift_ppm'] = d0 * 1e6
            # a drift cannot be known better than the offset's own σ over the episode's span (a line through two
            # ends that are each uncertain by σ: σ √12 / T for points spread evenly, their errors taken as one)
            floor = (sig or 0.0) * math.sqrt(12.0) / max(float(np.ptp(tau)), 1e-9)
            out['drift_sigma_ppm'] = max(floor, 0.0 if sd is None else sd) * 1e6
            out['drift_rho'] = r0
            out['offset_mid_ms'] = l0 * 1000.0
    return out


# ---------------------------------------------------------------- 6. gripper
def gripper(ep, stream, g: np.ndarray, plan_cam, req: dict, full_size, cancel=None) -> dict:
    import cv2
    out = {'tested': False, 'hidden': [], 'out_of_view': [], 'ncc': [],
           'method': 'the plan\'s gripper mask (core/camrender z-buffer of the gripper links at the recorded joint '
                     'state, the plan\'s lens and mount) against the image: zero-mean normalised cross-correlation '
                     'with the median of the frames around it'}
    if plan_cam is None or not plan_cam.on_link or not plan_cam.ctx.gripper:
        out['note'] = 'Not a camera on the robot with a gripper in the plan.'
        return out
    K = len(g)
    ts = np.asarray(stream.timestamps, float)
    Q = _interp_q(ep, ts)
    h, w = g.shape[1:]
    ncc = np.full(K, np.nan)
    masks = {}
    for k in range(K):
        if cancel is not None and cancel():
            break
        m = plan_cam.gripper_mask(Q[k])
        if m is None:
            out['note'] = 'The plan has no gripper links.'
            return out
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        masks[k] = m
    if not masks:
        return out
    out['tested'] = True
    f = g.astype(np.float32)
    for k, m in masks.items():
        if m.mean() < GRIP_MIN_FRAC:
            out['out_of_view'].append(int(k))
            continue
        lo, hi = max(0, k - GRIP_HALF), min(K, k + GRIP_HALF + 1)
        ref = np.median(f[lo:hi][:, m], axis=0)
        a = f[k][m] - f[k][m].mean()
        b = ref - ref.mean()
        den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
        ncc[k] = float((a * b).sum() / den) if den > 0 else 0.0
    out['hidden'] = [int(k) for k in range(K) if np.isfinite(ncc[k]) and ncc[k] < req['gripper_ncc']]
    out['ncc'] = [None if not np.isfinite(x) else round(float(x), 4) for x in ncc]
    out['mask_frac'] = float(np.mean([m.mean() for m in masks.values()]))
    return out


# ---------------------------------------------------------------- action alignment
def action_alignment(ep) -> dict:
    if ep.action is None or len(ep.t) < 4:
        return {'lag_rows': None, 'note': 'The episode records no action.'}
    A = np.asarray(ep.action, float)
    Q = np.asarray(ep.q, float)
    an, qn = list(ep.action_names or []), list(ep.joint_names or [])
    common = [n for n in an if n in qn]
    if common:
        ia = [an.index(n) for n in common]
        iq = [qn.index(n) for n in common]
        how = f'{len(common)} joints matched by name'
    elif A.shape[1] == Q.shape[1]:
        ia = iq = list(range(Q.shape[1]))
        how = 'matched by order (the names differ)'
    else:
        return {'lag_rows': None, 'note': f'The action ({A.shape[1]} values) and the state ({Q.shape[1]}) do not '
                                          'name the same joints.'}
    A, Q = A[:, ia], Q[:, iq]
    N = len(A)
    ks = np.arange(-MAX_LAG_ROWS, MAX_LAG_ROWS + 1)
    rms = np.full(len(ks), np.nan)
    for i, k in enumerate(ks):
        lo, hi = max(0, -k), min(N, N - k)
        if hi - lo < 3:
            continue
        rms[i] = float(np.sqrt(np.mean((A[lo:hi] - Q[lo + k:hi + k]) ** 2)))
    if not np.isfinite(rms).any():
        return {'lag_rows': None, 'note': 'The episode is too short to align the action.'}
    j = int(np.nanargmin(rms))
    lag = float(ks[j])
    if 0 < j < len(ks) - 1 and np.isfinite(rms[j - 1:j + 2]).all():
        y0, y1, y2 = rms[j - 1] ** 2, rms[j] ** 2, rms[j + 1] ** 2
        den = y0 - 2 * y1 + y2
        if den > 0:
            lag += float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
    dt = float(np.median(np.diff(ep.t)))
    return {'lag_rows': lag, 'lag_s': lag * dt, 'rms_at_lag': float(rms[j]), 'rms_at_zero': float(rms[MAX_LAG_ROWS]),
            'joints': how}


# ---------------------------------------------------------------- one episode
def _req(rid, value, ok, unit=None, camera=None, detail=''):
    d, u, title, what = REQUIREMENTS[rid]
    return {'id': rid, 'title': title, 'camera': camera, 'value': value, 'pass': bool(ok), 'unit': unit or u,
            'what': what, 'detail': detail}


READABLE = 'readable'             # the requirement every episode meets by being read (no limit to set)


def empty_reason(ep) -> str | None:
    """Why a loaded episode holds nothing to audit (no joint rows and no camera), else None."""
    if len(ep.t) or ep.cameras:
        return None
    warn = [str(w) for w in (ep.meta or {}).get('warnings') or []]
    return 'it holds no joint states and no camera frames' + (f' ({"; ".join(warn[:2])})' if warn else '')


def unreadable_episode(index: int, ep_id: str, reason: str) -> dict:
    """The record of an episode that cannot be audited: verdict drop, flag 'unreadable', the reason in words."""
    r = {'id': READABLE, 'title': 'Episode readable', 'camera': None, 'value': None, 'pass': False, 'unit': '',
         'what': 'the episode\'s files can be read and hold data', 'detail': reason}
    return {'index': int(index), 'id': str(ep_id), 'rows': 0, 'duration_s': 0.0, 'task': '', 'fps': None,
            'cameras': {}, 'requirements': [r], 'notes': [f'The episode cannot be audited: {reason}.'],
            'action': {'lag_rows': None, 'note': reason}, 'verdict': 'drop', 'flags': ['unreadable'],
            'seconds': 0.0, 'error': reason}


def audit_episode(ds, index: int, req: dict, plan=None, cam_map: dict | None = None, *, progress=None,
                  cancel=None, per_frame: bool = True) -> dict:
    """Every per-episode check of episode index (module docstring). plan: a plan.PlanContext or None; cam_map:
    {camera key: sensor uid or None} (plan.match_cameras). An episode that cannot be read (a damaged file) or holds
    nothing gives unreadable_episode(...) (verdict drop) instead of stopping the campaign."""
    from ..datasets import DatasetError
    t0 = time.perf_counter()
    try:
        ep = ds.load(index)
    except DatasetError as e:
        return unreadable_episode(index, str(index), str(e).rstrip('.'))
    why = empty_reason(ep)
    if why is not None:
        return unreadable_episode(index, ep.id, why)
    out = {'index': int(index), 'id': ep.id, 'rows': int(len(ep.t)), 'duration_s': float(ep.duration_s),
           'task': ep.task, 'fps': ep.fps, 'cameras': {}, 'requirements': [], 'notes': []}
    if (ep.meta or {}).get('time_base') == 'steps':
        out['notes'].append('The dataset records no times: times are step counts, so timing figures are in steps.')
    columns = (ep.meta or {}).get('columns') or {}
    keys = list(ep.cameras)
    rng = np.random.default_rng([20261107, int(index)])
    for ci, key in enumerate(keys):
        if cancel is not None and cancel():
            break
        cs = ep.cameras[key]
        if progress:
            progress(ci / max(1, len(keys)), f'episode {index}: {key}')
        tm = timing(cs, ep.fps, req)
        g, hashes, full = load_frames(cs, int(req['analysis_width']), cancel)
        K = len(g)
        if K != tm['frames']:
            out['notes'].append(f'{key}: {K} frames decoded, {tm["frames"]} timestamps; the frame checks use the '
                                'decoded frames.')
        uid = (cam_map or {}).get(key)
        pc = None
        if plan is not None and uid and plan.qmap is not None:      # None: the joints do not match (plan.notes)
            try:
                pc = plan.camera(key, int(full[0]), int(full[1]), uid)
            except Exception as e:                    # noqa: BLE001 — the plan side is optional; say why
                out['notes'].append(f'{key}: the plan\'s camera cannot be used ({e}).')
        gap_before = np.zeros(K, bool)
        for gp in tm['gaps']:
            if gp['frame'] < K:
                gap_before[gp['frame']] = True
        gs, sscale = sync_frames(g, full)
        pred = _Predictor(ep, cs.timestamps[:K], gs, pc, sscale)
        dup = duplicates(g, hashes, req, pred.moving(K), gap_before)
        ex = exposure(g, gap_before, req, columns, key)
        if pc is not None and pc.on_link and not req['blur_on_moving_cameras']:
            bl = {'frames': [], 'ratio': [None] * K, 'skipped': 'a camera on the robot: its whole view moves, so a '
                  'frame cannot be compared with its neighbours (blur_on_moving_cameras is off, a default)'}
        else:
            bl = blur(g, req)
        bad = gap_before.copy()
        for k in dup['frames']:
            bad[k] = True
            if k + 1 < K:
                bad[k + 1] = True
        for s in ex['steps']:
            if s['frame'] < K:
                bad[s['frame']] = True
        for k in bl['frames']:
            bad[k] = True
        sy = desync(ep, cs, gs, bad, pc, req, pred, rng)
        gr = gripper(ep, cs, g, pc, req, full, cancel)
        speed = None
        if len(ep.t) > 2 and K:
            v = np.linalg.norm(np.gradient(ep.q, ep.t, axis=0), axis=1)
            speed = np.interp(np.asarray(cs.timestamps, float)[:K], ep.t, v)
        blur_frames = [{'frame': k, 'ratio': bl['ratio'][k], 'joint_speed': None if speed is None else
                        float(speed[k])} for k in bl['frames']]
        cam = {'key': key, 'sensor_uid': uid, 'sensor': pc.name if pc is not None else None,
               'size': list(full) if full else None, 'timing': tm,
               'duplicates': {'exact': dup['exact'], 'near': dup['near'], 'frames': dup['frames'],
                              'identical_still': dup['identical_still']},
               'exposure': {k: v for k, v in ex.items() if k != 'log_ratio'},
               'blur': {'frames': blur_frames, 'skipped': bl.get('skipped')},
               'sync': {k: v for k, v in sy.items()}, 'gripper': {k: v for k, v in gr.items() if k != 'ncc'}}
        if per_frame:
            ts = np.asarray(cs.timestamps, float)[:K]
            cam['per_frame'] = {
                't': [round(float(x), 5) for x in ts],
                'change_dn': [None] + [round(float(x), 3) for x in dup['energy']] if K else [],
                'sharpness_ratio': bl['ratio'], 'gain_log_ratio': ex.get('log_ratio') or [],
                'gripper_ncc': gr.get('ncc') or [],
                'gap_before_s': [round(float(gp['gap_s']), 5) if gp else None for gp in
                                 [next((x for x in tm['gaps'] if x['frame'] == k), None) for k in range(K)]],
                'duplicate': [k in set(dup['frames']) for k in range(K)],
                'blur': [k in set(bl['frames']) for k in range(K)],
                'exposure_changed': _in_runs(ex['runs'], K),
                'gripper_hidden': [k in set(gr['hidden']) for k in range(K)],
            }
        out['cameras'][key] = cam
        _camera_requirements(out['requirements'], cam, req)
    out['action'] = action_alignment(ep)
    lag = out['action'].get('lag_rows')
    lo, hi = req['action_lag_rows']
    out['requirements'].append(_req('action_lag_rows', lag, lag is None or lo - 0.5 <= lag <= hi + 0.5,
                                    detail=out['action'].get('note') or out['action'].get('joints', '')))
    fails = [r for r in out['requirements'] if not r['pass']]
    out['verdict'] = 'pass' if not fails else ('fix' if all(r['id'] in FIXABLE for r in fails) else 'drop')
    out['flags'] = sorted({FLAG[r['id']] for r in fails})
    out['seconds'] = round(time.perf_counter() - t0, 3)
    return out


FLAG = {'max_dropped_frames': 'drop', 'max_duplicate_frames': 'duplicate', 'fps_tolerance_pct': 'fps',
        'max_jitter_ms': 'jitter', 'max_gaps': 'usb_reset', 'max_offset_ms': 'desync', 'max_drift_ppm': 'drift',
        'max_exposure_changes': 'exposure_change', 'max_blur_frames': 'blur',
        'max_gripper_hidden_frames': 'gripper_hidden', 'action_lag_rows': 'action_lag',
        'max_mount_shift_deg': 'mount_slip', 'max_mount_shift_mm': 'mount_slip', READABLE: 'unreadable'}


def _in_runs(runs, K) -> list:
    m = [False] * K
    for r in runs:
        for k in range(max(0, int(r['start'])), min(K, int(r['end']))):
            m[k] = True
    return m


def _camera_requirements(reqs: list, cam: dict, req: dict):
    key = cam['key']
    tm = cam['timing']
    reqs.append(_req('max_dropped_frames', tm.get('dropped_frames', 0),
                     tm.get('dropped_frames', 0) <= req['max_dropped_frames'], camera=key))
    nd = len(cam['duplicates']['frames'])
    reqs.append(_req('max_duplicate_frames', nd, nd <= req['max_duplicate_frames'], camera=key))
    if tm.get('fps_measured') and tm.get('fps_stated'):
        err = 100.0 * abs(tm['fps_measured'] - tm['fps_stated']) / tm['fps_stated']
        reqs.append(_req('fps_tolerance_pct', err, err <= req['fps_tolerance_pct'], camera=key,
                         detail=f'{tm["fps_measured"]:.3f} fps measured, {tm["fps_stated"]:g} stated'))
    if tm.get('jitter_ms') is not None:
        reqs.append(_req('max_jitter_ms', tm['jitter_ms'], tm['jitter_ms'] <= req['max_jitter_ms'], camera=key))
    ng = len(tm['usb_gaps'])
    reqs.append(_req('max_gaps', ng, ng <= req['max_gaps'], camera=key,
                     detail=', '.join(f'{g["gap_s"] * 1000:.0f} ms after {g["after_s"]:.2f} s'
                                      for g in tm['usb_gaps'])))
    sy = cam['sync']
    if sy.get('offset_ms') is not None:
        sig = sy.get('sigma_ms') or 0.0
        big = abs(sy['offset_ms']) > req['max_offset_ms'] and abs(sy['offset_ms']) > 3 * sig
        reqs.append(_req('max_offset_ms', sy['offset_ms'], not big, camera=key,
                         detail=f'± {sig:.1f} ms (1 σ, bootstrap); ρ {sy["rho"]:.2f} against {sy["reference"]}'))
    if sy.get('drift_ppm') is not None:
        sig = sy.get('drift_sigma_ppm') or 0.0
        big = abs(sy['drift_ppm']) > req['max_drift_ppm'] and abs(sy['drift_ppm']) > 3 * sig
        reqs.append(_req('max_drift_ppm', sy['drift_ppm'], not big, camera=key, detail=f'± {sig:.0f} ppm (1 σ)'))
    ne = len(cam['exposure']['steps'])
    reqs.append(_req('max_exposure_changes', ne, ne <= req['max_exposure_changes'], camera=key,
                     detail=cam['exposure']['source']))
    if not cam['blur'].get('skipped'):
        nb = len(cam['blur']['frames'])
        reqs.append(_req('max_blur_frames', nb, nb <= req['max_blur_frames'], camera=key))
    if cam['gripper'].get('tested'):
        nh = len(cam['gripper']['hidden'])
        reqs.append(_req('max_gripper_hidden_frames', nh, nh <= req['max_gripper_hidden_frames'], camera=key))
