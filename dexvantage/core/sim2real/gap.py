"""The sim-to-real gap of a recorded episode, camera by camera: the episode rendered again in MuJoCo through the
project's rig (core/sim/render.py: each camera's lens, its mount, the robot at the recorded joint states), compared
with the recorded frames, and the gap explained by the corrections that would close it.

    settings_of(project, given=None) -> dict          the settings, labelled defaults filled in
    analyse(project, dataset, settings=None, *, progress=None, cancel=None) -> dict    the study's document
    apply_change(project, change) -> str              make one suggestion's change on a project (in place); returns
                                                      the store section it touches ('variant' or 'analysis' or
                                                      'workspace')
    describe_change(project, change) -> str           what apply_change would do, in words (the page's Preview)
    appearance_of(project, sensor_uid) -> dict        the render's camera response stored for a sensor

Per camera (every enabled camera and depth sensor of the active variant the episode records, matched by
core/calib/render_calib.match_stream: the stream's sensor_uid, else its name), at a working width (default 320 px;
frames resized with area averaging, the render made at the recording's size up to render_max_width, default 640 px,
then resized the same way, so both carry the same resampling):

1. Latency (camera content against the joint states; a latency the time-sync settings already hold for the camera,
   latency_s, is the plan's: each frame is rendered at its stamp minus it, and the search finds what is left): at
   lag_frames frames (default 8, those where the joints move fastest) the robot is rendered at the frame's stamp + τ
   for τ on a grid (±lag_max_s, default 150 ms, in lag_step_s
   steps, default 10 ms); the score at τ is the mean over the frames of the zero-mean normalised cross-correlation of
   the gradient magnitudes (Sobel of the 1 px Gaussian-smoothed grey image) of the recorded frame and the render,
   over the pixels the robot covers at some τ (dilated 5 px) for a static camera, every pixel for a camera on the
   robot. Gradient magnitudes and a normalised correlation do not depend on gain or offset, so an exposure gap does
   not move the peak. The peak is refined by a parabola through its neighbours; σ_τ is the spread of the peak over
   200 bootstrap resamples of the frames (seeded). τ > 0: the image shows the robot later than its stamp says.
   Not observable when the joints hardly move, the peak is at the end of the grid, or σ_τ is over 2 grid steps.
2. Mount offset (static cameras, setting mount on): the silhouette calibration (core/calib/render_calib.estimate, its
   settings mount_settings over the project's silhouette_calib ones; defaults 12 frames, 10 bootstrap resamples) on
   the episode with the camera's stamps moved by τ when the latency is observable (a desync otherwise reads as a mount
   error: render_calib's known limit). Its ΔT (sensor body frame) is applied as Renderer's extrinsic_delta. When the
   mount is observable the latency search (1) runs again through the recovered mount (a mount offset biases it; the
   first pass is kept as first_pass_s).
3. The gap, on frames frames (default 8, farthest apart in joint space) rendered with each combination of the
   geometric corrections found (latency, mount): the image difference d = mean |recorded − rendered| over pixels,
   channels and frames (DN), after the photometric corrections of the combination: blur (a Gaussian σ fitted so the
   render's sharpness equals the recording's; only when the recording's sharpness is under blur_min_ratio, default
   0.9, of the render's) then exposure (per channel gain and offset that line up the histograms' quantiles,
   measures.fit_affine).
   Each correction's share of the gap is its Shapley value over the combinations (measures.shapley: what it removes,
   averaged over every order of applying them): the shares add up to d(none) − d(all). The render first gets the
   camera response already stored for the sensor (settings appearance: gain, offset_dn, blur_px, noise_dn), so after
   Apply the gap it closed is gone.
4. Photometric figures (recording against the render with the geometric corrections): mean per channel, RMS
   contrast, Earth mover's distance of each channel's histogram (DN), sharpness (variance of the Laplacian less its
   noise share), noise σ (Immerkær 1996, edge pixels left out), the gain and offset fitted (after the blur
   correction when one is fitted), and the brightness change at mid grey: g · 128 + o − 128 DN, log₂((g · 128 + o) /
   128) stops.
5. Geometric figures: the robot silhouette IoU, predicted against observed, before (as planned) and after every
   correction. No reader exposes segmentation masks, so the robot is segmented by background difference
   (render_calib.foreground: pixels more than fg_threshold, default 25 DN, from the frames' median, opened 3 × 3),
   the same rule on the recorded and on the rendered frames: the median holds the robot where it rests, so both masks
   are the robot where it moves, and they are compared like for like. Static cameras only (a camera on the robot has
   no static background).
6. Depth (depth sensors): the predicted valid-pixel share (rendered depth within the device's stated minimum Z at the
   stream and maximum range, each when stated) and, when the episode carries a decoded depth stream for the camera,
   the measured share and the depth error on the robot (measures.depth_gap). DexVantage's readers list recorded depth
   streams but do not decode them, so the measured side is "not measured" with the streams named.
7. Timing: the stamps' jitter (core/audit/episode.timing: robust spread about a regular grid), the latency τ ± σ_τ,
   and both against the time-sync study's prediction for the stream (core/studies/plan_vs_recording.sync_prediction:
   the predicted offset is the negated skew interval).
8. Distribution distance: kernel MMD (measures.mmd_test: RBF on 32 × 24 thumbnails, median-heuristic bandwidth,
   200 permutations) between the recorded and the rendered frames, as planned and with every correction. It is not the
   FID: that needs a trained Inception network's features over thousands of images, which DexVantage does not ship.

Causes and actions, ranked over the cameras by the DN they remove: latency (Apply enters the measured latency, stamp
minus exposure = −τ, in the time-sync settings; the Time sync page), mount (Apply writes the recovered mount; the
Silhouette calibration page), exposure (Apply sets the render's per-channel gain and offset for the camera; the
recording side is the camera's exposure or gain, in stops), lighting (the same gain within lighting_tol, default 10 %,
in every camera with an exposure cause: the scene's light, not one camera's exposure; Apply scales the project's
lights' dim when it has lights, the scale fits and the cameras share no offset of min_reduction_dn or more (a
scale gives a gain, not an offset), else the cameras' render gains; the Lighting page), blur (Apply sets
the render's blur; on the recording side focus or exposure time, the Motion blur page), noise (σ_real² − σ_render²,
Apply sets the render's noise; not ranked: noise adds to the image difference rather than closing it). A cause under
min_share_pct (default 5 %) of its camera's gap, or removing under min_reduction_dn (default 1 DN), gets no action.
The headline is the largest d(none) over the cameras, with a status against gap_good_dn, gap_warn_dn, gap_serious_dn
(defaults 4, 8, 16 DN: DexVantage defaults, not standards).

Known limits: the render is MuJoCo's OpenGL shading of the project's model of the cell (materials and lights are
DexVantage's defaults unless set), so a real recording differs by much that no correction here models (textures,
shadows, cables, the background): the corrections explain the part of the gap they can, and the rest is "not
explained"; one episode per run; the latency search assumes a constant offset over the episode (no drift) and needs
the robot to move; the mount offset is the silhouette calibration's, with its own limits (static cameras, a robot that
looks like its model); a static camera's observed silhouette is the moving part of the robot only; the exposure fit is
global per channel (no vignetting or tone curve); the blur is one Gaussian (no motion blur direction); depth is
compared only when a reader decodes it; Shapley shares are exact for the corrections tried, but corrections interact
(a mount fitted under a wrong exposure), so a share is not a measurement of a physical cause.
"""
from __future__ import annotations

import base64
import copy
import dataclasses
import math
import os

import numpy as np

from ..jobs import StudyError
from . import measures as MS

SID = 'sim2real'
DEFAULTS = {
    'dataset': None,               # the recorded dataset (path); else the audit's / project.data's
    'episode': None,               # episode index (None: the first)
    'cameras': None,               # sensor uids (None: every recorded camera)
    'frames': 8,                   # default: frames compared per camera
    'width': 320,                  # default: working width (px)
    'render_max_width': 640,       # default: renders at the recording's size up to this width (px)
    'lag_frames': 8,               # default: frames of the latency search
    'lag_max_s': 0.15,             # default: latency searched within ± this (s)
    'lag_step_s': 0.01,            # default: latency grid step (s)
    'mount': True,                 # default: run the silhouette calibration on static cameras
    'mount_settings': None,        # render_calib settings over the project's (None: MOUNT_DEFAULTS)
    'fg_threshold': 25.0,          # default: background-difference threshold (DN)
    'edge_pct': 90.0,              # default: the noise estimate leaves out pixels above this gradient percentile
    'blur_min_ratio': 0.9,         # default: a blur is fitted when the recording's sharpness is under this share
    'lighting_tol': 0.1,           # default: cameras whose gains agree within this share point at the lighting
    'min_share_pct': 5.0,          # default: a cause under this share of its camera's gap gets no action
    'min_reduction_dn': 1.0,       # default: ... or removing under this much of it (DN)
    'gap_good_dn': 4.0,            # default: status thresholds of the headline (DN)
    'gap_warn_dn': 8.0,
    'gap_serious_dn': 16.0,
    'mmd_size': [32, 24],          # default: thumbnail size of the MMD's features (px)
    'mmd_permutations': 200,       # default: permutations of the MMD test
    'thumbs': 3,                   # default: frames kept as thumbnails per camera
    'thumb_width': 240,            # default: thumbnail width (px)
    'appearance': None,            # {sensor uid: {'gain', 'offset_dn', 'blur_px', 'noise_dn'}}: the render's response
    'device': 'auto',
    'seed': 0,
}
MOUNT_DEFAULTS = {'max_frames': 12, 'bootstrap': 10}
NUMERIC = ('frames', 'width', 'render_max_width', 'lag_frames', 'lag_max_s', 'lag_step_s', 'fg_threshold',
           'edge_pct', 'blur_min_ratio', 'lighting_tol', 'min_share_pct', 'min_reduction_dn', 'gap_good_dn',
           'gap_warn_dn', 'gap_serious_dn', 'mmd_permutations', 'thumbs', 'thumb_width', 'seed')
INTEGER = ('frames', 'width', 'render_max_width', 'lag_frames', 'mmd_permutations', 'thumbs', 'thumb_width', 'seed')
WORDS = {'good': 'close to real', 'warning': 'small gap', 'serious': 'large gap', 'critical': 'very large gap'}
FACTOR_TITLES = {'latency': 'Camera latency', 'mount': 'Mount offset', 'exposure': 'Exposure and colour',
                 'blur': 'Blur', 'noise': 'Noise', 'lighting': 'Lighting'}
PAGES = {'latency': ('time_sync', 'Time sync'), 'mount': ('silhouette', 'Silhouette calibration'),
         'exposure': ('sim2real', 'Sim-to-real gap'), 'blur': ('motion_blur', 'Motion blur'),
         'noise': ('sim2real', 'Sim-to-real gap'), 'lighting': ('lighting', 'Lighting')}
BOOT = 200                         # bootstrap resamples of the latency peak
MIN_SPEED = 0.05                   # rad/s: below this at every latency frame the latency is not observable
DILATE_PX = 5                      # the latency mask's dilation (px)
RENDER_CHUNK = 48                  # views per render call
FID_NOTE = ('FID is not computed: it needs a trained Inception network\'s features over thousands of images '
            '(torch and its weights are not part of DexVantage). The kernel MMD on thumbnails is shown instead; it '
            'is not comparable with FID values.')


class GapError(StudyError):
    """The sim-to-real gap cannot be measured on this input; the message says why."""


def _signed(x: float, nd: int = 0) -> str:
    """A signed figure as shown ('+3'; never '-0': a value that rounds to zero is written '0')."""
    t = f'{x:+.{nd}f}'
    return t[1:] if float(t) == 0 else t


# ====================================================================== settings
def settings_of(project, given: dict | None = None) -> dict:
    stored = dict(project.analysis.studies.get(SID) or {}) if project is not None else {}
    src = dict(stored, **(given or {}))
    unknown = [k for k in src if k not in DEFAULTS and k not in ('defaults', 'fingerprint')]
    if unknown:
        raise GapError(f'The sim-to-real gap has no setting {unknown[0]}; it has {", ".join(DEFAULTS)}.')
    st, used = copy.deepcopy(DEFAULTS), []
    for k, v in src.items():
        if k in ('defaults', 'fingerprint') or v is None:
            continue
        if k in NUMERIC:
            try:
                x = float(v)
            except (TypeError, ValueError):
                raise GapError(f'The setting {k} must be a number; it is {v!r}.') from None
            if not math.isfinite(x) or (x < 0 and k != 'seed'):
                raise GapError(f'The setting {k} must be a number of zero or more; it is {v!r}.')
            st[k] = int(round(x)) if k in INTEGER else x
        elif k == 'mount':
            st[k] = bool(v)
        elif k in ('appearance', 'mount_settings'):
            if not isinstance(v, dict):
                raise GapError(f'The setting {k} must be a dictionary.')
            st[k] = copy.deepcopy(v)
        elif k == 'cameras':
            if isinstance(v, str):
                v = [v]
            st[k] = [str(x) for x in v]
        elif k == 'mmd_size':
            try:
                w, h = int(v[0]), int(v[1])
            except (TypeError, ValueError, IndexError):
                raise GapError('The setting mmd_size must be [width, height] in pixels.') from None
            st[k] = [max(2, w), max(2, h)]
        else:
            st[k] = v
        used.append(k)
    if st['frames'] < 3:
        raise GapError('The setting frames must be at least 3 (the background difference needs a median).')
    if st['lag_step_s'] <= 0:
        raise GapError('The setting lag_step_s must be above zero.')
    st['appearance'] = st['appearance'] or {}
    st['mount_settings'] = dict(MOUNT_DEFAULTS, **(st['mount_settings'] or {}))
    st['defaults'] = [k for k in DEFAULTS if k not in used and k not in ('dataset', 'episode', 'cameras',
                                                                          'appearance', 'device')]
    return st


def appearance_of(project, sensor_uid: str) -> dict:
    a = ((project.analysis.studies.get(SID) or {}).get('appearance') or {}).get(sensor_uid) or {}
    return dict(a) if isinstance(a, dict) else {}


def _apply_appearance(rgb, app: dict, rng) -> np.ndarray:
    """The render's stored camera response: blur, then gain and offset per channel, then Gaussian noise."""
    a = np.asarray(rgb, np.float32)
    if not app:
        return a
    a = MS.blur(a, float(app.get('blur_px') or 0.0))
    g = np.asarray(app.get('gain') or [1.0, 1.0, 1.0], np.float32).reshape(-1)
    o = np.asarray(app.get('offset_dn') or [0.0, 0.0, 0.0], np.float32).reshape(-1)
    a = MS.apply_affine(a, g if len(g) == 3 else np.repeat(g[:1], 3), o if len(o) == 3 else np.repeat(o[:1], 3))
    n = float(app.get('noise_dn') or 0.0)
    if n > 0:
        a = np.clip(a + rng.normal(0.0, n, a.shape).astype(np.float32), 0.0, 255.0)
    return a


# ====================================================================== helpers
def _resize(img, w, h, nearest=False):
    import cv2
    a = np.asarray(img)
    if a.shape[1] == w and a.shape[0] == h:
        return a
    if nearest:
        return cv2.resize(a, (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(a.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)


def _grad(img) -> np.ndarray:
    import cv2
    g = cv2.GaussianBlur(MS.to_grey(img), (0, 0), 1.0)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))


def _ncc(a, b) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else 0.0


def _parabola(y, k) -> float:
    """Sub-step offset of the peak at k from a parabola through y[k-1], y[k], y[k+1] (0 at an end)."""
    if k <= 0 or k >= len(y) - 1:
        return 0.0
    a, b, c = y[k - 1], y[k], y[k + 1]
    den = a - 2 * b + c
    return float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) if den < 0 else 0.0


def _jpeg(img, width: int) -> str:
    import cv2
    a = np.clip(np.rint(np.asarray(img, np.float32)), 0, 255).astype(np.uint8)
    h = max(2, int(round(a.shape[0] * width / a.shape[1])))
    a = cv2.resize(a, (width, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(a, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf.tobytes()).decode('ascii') if ok else ''


def _num(x, nd=4):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


class _Cam:
    """One camera's renders at the working size."""

    def __init__(self, R, sensor, stream, cam_full, ep, model, pairs, ruid, st):
        self.R, self.sensor, self.stream = R, sensor, stream
        self.ep, self.model, self.pairs, self.ruid = ep, model, pairs, ruid
        W0, H0 = cam_full.width, cam_full.height
        w = min(int(st['width']), W0)
        self.w, self.h = w, max(2, int(round(H0 * w / W0)))
        rw = min(int(st['render_max_width']), W0)
        self.cam_render = cam_full if rw == W0 else cam_full.scaled(rw, max(2, int(round(H0 * rw / W0))))
        self.cam_work = cam_full.scaled(self.w, self.h)
        self.robot_ids = np.array([i for i, o in enumerate(R.build.owners) if o and o[0] == 'robot'], np.int32)
        self.renders = 0

    def q(self, times):
        from ..calib import render_calib as RC
        return RC.q_at(self.ep, self.model, self.pairs, np.asarray(times, float))

    def render(self, times, delta=None, outputs=('rgb', 'seg'), full=True, cancel=None) -> list:
        """[{rgb (h, w, 3) float32, robot (h, w) bool, depth (h, w) or None}] at the working size."""
        cam = self.cam_render if full else self.cam_work
        Q = self.q(times)
        views = [{'sensor_uid': self.sensor.uid, 'q': {self.ruid: q}, 'outputs': tuple(outputs), 'camera_model': cam,
                  'extrinsic_delta': delta} for q in Q]
        out = []
        for a in range(0, len(views), RENDER_CHUNK):
            if cancel is not None and cancel():
                raise GapError('The sim-to-real gap was cancelled.')
            for img in self.R.render(views[a:a + RENDER_CHUNK]):
                d = {'rgb': _resize(img['rgb'], self.w, self.h).astype(np.float32) if 'rgb' in img else None,
                     'robot': _resize(np.isin(img['seg'], self.robot_ids).astype(np.uint8), self.w, self.h,
                                      nearest=True).astype(bool) if 'seg' in img else None,
                     'depth': _resize(img['depth'], self.w, self.h, nearest=True) if 'depth' in img else None}
                out.append(d)
        self.renders += len(views)
        return out


# ====================================================================== the parts
def _latency(C: _Cam, reals: dict, idx: list, st: dict, rng, cancel, delta=None) -> dict:
    ts = np.asarray(C.stream.timestamps, float)
    lags = np.arange(-st['lag_max_s'], st['lag_max_s'] + 0.5 * st['lag_step_s'], st['lag_step_s'])
    Q = C.q(ts)
    dt = np.gradient(ts) if len(ts) > 1 else np.ones(1)
    speed = np.max(np.abs(np.gradient(Q, axis=0) / np.where(dt > 0, dt, 1.0)[:, None]), axis=1) \
        if len(ts) > 1 else np.zeros(1)
    top = float(np.max(speed[idx])) if idx else 0.0
    if top < MIN_SPEED:
        return {'observable': False, 'why': 'the robot hardly moves at the frames searched', 'latency_s': None,
                'sigma_s': None, 'frames': idx}
    times = np.concatenate([ts[i] + lags for i in idx])
    rends = C.render(times, delta, outputs=('rgb', 'seg'), full=False, cancel=cancel)
    import cv2
    L = len(lags)
    score = np.zeros((len(idx), L))
    k = np.ones((2 * DILATE_PX + 1, 2 * DILATE_PX + 1), np.uint8)
    for n, i in enumerate(idx):
        sub = rends[n * L:(n + 1) * L]
        gr = _grad(_resize(reals[i], C.w, C.h))
        if C.sensor.mount.on_link:
            mask = np.ones(gr.shape, bool)
        else:
            u = np.zeros(gr.shape, np.uint8)
            for r in sub:
                u |= r['robot'].astype(np.uint8)
            mask = cv2.dilate(u, k) > 0
            if mask.sum() < 50:
                mask = np.ones(gr.shape, bool)
        for j, r in enumerate(sub):
            score[n, j] = _ncc(gr[mask], _grad(r['rgb'])[mask])
    curve = score.mean(0)
    j = int(np.argmax(curve))
    tau = float(lags[j] + _parabola(curve, j) * st['lag_step_s'])
    boots = []
    for _ in range(BOOT):
        s = score[rng.integers(0, len(idx), len(idx))].mean(0)
        b = int(np.argmax(s))
        boots.append(lags[b] + _parabola(s, b) * st['lag_step_s'])
    sig = float(np.std(boots))
    out = {'latency_s': tau, 'sigma_s': sig, 'frames': [int(i) for i in idx],
           'curve': [[_num(x, 4), _num(y, 5)] for x, y in zip(lags, curve)], 'peak_score': _num(curve[j], 5),
           'grid_s': [float(lags[0]), float(lags[-1]), float(st['lag_step_s'])], 'renders': len(times)}
    if j == 0 or j == L - 1:
        out.update(observable=False, why=f'the best lag is at the end of the search (±{st["lag_max_s"] * 1000:g} ms): '
                                         'widen lag_max_s')
    elif sig > 2 * st['lag_step_s']:
        out.update(observable=False, why=f'the peak is not sharp (σ {sig * 1000:.0f} ms over the frames)')
    else:
        out.update(observable=True, why='')
    return out


def _mount(C: _Cam, project, ep, tau: float | None, st: dict, report, cancel) -> dict:
    from ..calib import render_calib as RC
    s = C.sensor
    if s.mount.on_link:
        return {'observable': False, 'why': f'{s.name} rides on {s.mount.parent_frame}: the mount of a camera on the '
                                            'robot comes from the hand-eye solve, not the silhouettes'}
    if not st['mount']:
        return {'observable': False, 'why': 'the mount search is off (setting mount)'}
    cs = C.stream
    e2 = ep
    if tau is not None:
        ns = dataclasses.replace(cs, timestamps=np.asarray(cs.timestamps, float) + float(tau))
        e2 = dataclasses.replace(ep, cameras={**ep.cameras, cs.key: ns})
    stored = {k: v for k, v in (project.analysis.studies.get('silhouette_calib') or {}).items()
              if k not in ('dataset', 'sensor', 'episodes', 'episode', 'camera')}
    cal = dict(stored, **st['mount_settings'], camera=cs.key, device=st['device'])
    try:
        doc = RC.estimate(e2, project, s.uid, settings=cal, renderer=C.R, progress=report, cancel=cancel)
    except StudyError as e:
        return {'observable': False, 'why': str(e).rstrip('.')}
    from ..frames import Pose
    out = {'observable': bool(doc.get('observable')), 'why': doc.get('why', ''),
           'delta': doc['delta'],                   # an axis the frames do not constrain has no σ (None, not NaN)
           'sigma': {k: [float(x) if x is not None and math.isfinite(float(x)) else None for x in doc['sigma'][k]]
                     for k in ('position_mm', 'rotation_deg')},
           'fit': {k: doc['fit'].get(k) for k in ('iou_before', 'iou_after', 'rms_px')},
           'frames_used': doc.get('frames_used'), 'time_shift_s': tau,
           'renders': (doc.get('render') or {}).get('renders')}
    D = RC.exp_se3(np.r_[doc['delta']['position_m'], np.radians(doc['delta']['rotation_deg'])])
    out['D'] = D
    cur = s.mount.pose.matrix()
    new = Pose.from_dict(doc['mount']['pose']).matrix()
    d = np.linalg.inv(cur) @ new
    out['apply'] = {'kind': 'mount', 'sensor_uid': s.uid, 'pose': doc['mount']['pose'],
                    'change_mm': float(np.linalg.norm(d[:3, 3]) * 1000),
                    'change_deg': math.degrees(float(np.linalg.norm(RC.log_se3(d)[3:])))}
    return out


def _timing(project, stream, ep, uid, imager, lat: dict, pred_cache: dict, t_plan: float = 0.0) -> dict:
    """Jitter and the latency (the time-sync settings' planned one plus the one found) against the prediction."""
    from ..audit import episode as AE
    from ..studies import plan_vs_recording as PR
    try:
        t = AE.timing(stream, ep.fps, AE.requirements_of(project))
    except StudyError:
        t = AE.timing(stream, ep.fps, AE.requirements_of(None))
    out = {'frames': t.get('frames'), 'fps_stated': t.get('fps_stated'), 'fps_measured': t.get('fps_measured'),
           'jitter_ms': t.get('jitter_ms'), 'dropped_frames': t.get('dropped_frames'),
           'latency_ms': None if lat.get('latency_s') is None else (lat['latency_s'] + t_plan) * 1000,
           'planned_latency_ms': t_plan * 1000,
           'latency_sigma_ms': None if lat.get('sigma_s') is None else lat['sigma_s'] * 1000,
           'latency_observable': bool(lat.get('observable'))}
    if 'pred' not in pred_cache:
        pred_cache['pred'] = PR._plan_sync(project)                     # noqa: SLF001 — the shared prediction
    offs = [{'offset_ms': out['latency_ms'], 'sigma_ms': out['latency_sigma_ms']}] if lat.get('observable') else []
    pr = PR.sync_prediction(pred_cache['pred'], uid, imager, offs)
    out['plan'] = pr
    if pr.get('predicted_offset_ms'):
        lo, hi = pr['predicted_offset_ms']
        out['predicted_spread_ms'] = float(hi - lo)
    return out


def _depth(C: _Cam, rend: list, ep) -> dict:
    from .. import library as LB
    s = C.sensor
    dp = LB.depth_spec(s) or {}
    zmax = dp.get('max_range_m')
    zmax = float(zmax) if isinstance(zmax, (int, float)) else None
    st_ = getattr(s, 'stream', None)
    zmin = LB.min_z(s, int(st_[0]), int(st_[1])) if isinstance(st_, (list, tuple)) and len(st_) == 2 else None
    preds = [r['depth'] for r in rend if r.get('depth') is not None]
    out = {'z_min_m': zmin, 'z_max_m': zmax,
           'range_note': ('minimum Z ' + ('as stated for the stream' if zmin is not None else 'not stated by the maker')
                          + '; maximum range ' + ('as stated' if zmax is not None else 'not stated by the maker'))}
    if preds:
        ok = [np.isfinite(p) & (p > 0) & ((p >= zmin) if zmin is not None else True) &
              ((p <= zmax) if zmax is not None else True) for p in preds]
        out['predicted_valid_pct'] = float(100 * np.mean([o.mean() for o in ok]))
    meas = _recorded_depth(ep, C.stream)
    if meas is None:
        listed = list(((ep.meta or {}).get('depth_streams') or {}).keys())
        out['measured'] = None
        out['why'] = ('the recording lists depth streams (' + ', '.join(listed[:4]) + ') but DexVantage\'s readers '
                      'do not decode them' if listed else 'the recording has no depth stream for this camera')
        return out
    D, idx_ = meas
    rows = []
    for r, i in zip(rend, C._idx):
        if i in D and r.get('depth') is not None:
            m = _resize(np.asarray(D[i], np.float32), C.w, C.h, nearest=True)
            rows.append(MS.depth_gap(r['depth'], m, r['robot'], zmin, zmax))
    if rows:
        out['measured_valid_pct'] = float(np.mean([x['measured_valid_pct'] for x in rows]))
        out['valid_gap_pct'] = out['measured_valid_pct'] - out.get('predicted_valid_pct', 0.0)
        errs = [x['robot_error_mm'] for x in rows if x.get('robot_error_mm')]
        out['robot_error_mm'] = {k: float(np.median([e[k] for e in errs])) for k in errs[0]} if errs else None
    return out


def _recorded_depth(ep, stream):
    """({frame index: depth (H, W) m}, key) when the episode carries a decoded depth stream for this camera (a stream
    whose meta says {'depth_of': key, 'depth_scale_m': s}), else None. No reader of DexVantage provides one yet; a
    plugin reader can."""
    for k, cs in (ep.cameras or {}).items():
        if (cs.meta or {}).get('depth_of') == stream.key:
            sc = float((cs.meta or {}).get('depth_scale_m') or 1.0)
            return {i: np.asarray(im, np.float32).reshape(im.shape[:2]) * sc for i, im in cs.frames()}, k
    return None


# ====================================================================== one camera
def _camera(R, project, ep, sensor, stream, model, pairs, ruid, st, rng, pred_cache, report, cancel) -> dict:
    from ..calib import render_calib as RC
    notes = []
    scam = R.camera(sensor.uid)
    ts = np.asarray(stream.timestamps, float)
    if len(ts) < 3:
        raise GapError(f'Camera {stream.key} has {len(ts)} frames; the gap needs at least 3.')
    if stream.width is None or stream.height is None:
        for _i, im in stream.frames([0]):
            stream.height, stream.width = np.asarray(im).shape[:2]
    cam_full = RC.camera_model_of(stream, scam, notes)
    if (stream.width, stream.height) != (cam_full.width, cam_full.height):
        cam_full = cam_full.scaled(int(stream.width), int(stream.height))
    # a latency the time-sync settings already hold (stamp minus exposure, e): the plan renders at stamp − e
    from ..studies import time_sync as TS
    known = TS.settings_of(project)['latency_s'].get(sensor.uid) or {}
    t_plan = -float(known['value']) if known.get('value') is not None else 0.0
    if t_plan:
        stream = dataclasses.replace(stream, timestamps=ts + t_plan)
        ep = dataclasses.replace(ep, cameras={**ep.cameras, stream.key: stream})
        ts = ts + t_plan
        notes.append(f'The time-sync settings hold a latency of {-t_plan * 1000:+.1f} ms for this camera (stamp minus '
                     'exposure): the plan renders each frame that much off its stamp, and the latency found is what '
                     'is left.')
    C = _Cam(R, sensor, stream, cam_full, ep, model, pairs, ruid, st)
    Qall = RC.q_at(ep, model, pairs, ts)
    idx_p = RC.choose_frames(Qall, int(st['frames']))
    dt = np.gradient(ts)
    sp = np.max(np.abs(np.gradient(Qall, axis=0) / np.where(dt > 0, dt, 1.0)[:, None]), axis=1)
    order = sorted(range(len(ts)), key=lambda i: -sp[i])
    gap = max(1, len(ts) // (3 * max(1, int(st['lag_frames']))))
    idx_l = []
    for i in order:
        if all(abs(i - j) >= gap for j in idx_l):
            idx_l.append(i)
        if len(idx_l) >= int(st['lag_frames']):
            break
    idx_l = sorted(idx_l)
    want = sorted(set(idx_p) | set(idx_l))
    reals_full = {int(i): np.asarray(im, np.uint8) for i, im in stream.frames(want)}
    missing = [i for i in want if i not in reals_full]
    if missing:
        raise GapError(f'{len(missing)} frames of {stream.key} could not be decoded.')
    reals = {i: _resize(im, C.w, C.h).astype(np.float32) for i, im in reals_full.items()}
    del reals_full
    app = dict((st.get('appearance') or {}).get(sensor.uid) or {})
    out = {'sensor_uid': sensor.uid, 'sensor': sensor.name, 'key': stream.key, 'kind': sensor.kind,
           'on_link': bool(sensor.mount.on_link), 'size': [int(stream.width), int(stream.height)],
           'working_size': [C.w, C.h], 'frames': [int(i) for i in idx_p], 'appearance': app, 'notes': notes,
           'imager': scam.imager}
    # 1. latency
    report(0.1, f'{sensor.name}: latency')
    lat = _latency(C, reals, idx_l, st, rng, cancel)
    out['latency'] = {k: v for k, v in lat.items()}
    tau = lat['latency_s'] if lat.get('observable') else None
    # 2. mount
    report(0.35, f'{sensor.name}: mount (silhouette calibration)')
    mnt = _mount(C, project, ep, tau, st, report=None, cancel=cancel)
    D = mnt.pop('D', None) if mnt.get('observable') else None
    mnt.pop('D', None)
    out['mount'] = mnt
    if D is not None:                   # the latency again through the recovered mount (the offset biased it)
        lat2 = _latency(C, reals, idx_l, st, rng, cancel, D)
        lat2['first_pass_s'] = lat.get('latency_s')
        lat2['through'] = 'the mount the silhouette calibration recovered'
        lat = lat2
        out['latency'] = dict(lat)
        tau = lat['latency_s'] if lat.get('observable') else None
    # 3. the render sets
    report(0.7, f'{sensor.name}: rendering the compared frames')
    geo = [f for f, ok in (('latency', tau is not None), ('mount', D is not None)) if ok]
    outs = ('rgb', 'seg', 'depth') if sensor.kind == 'depth' else ('rgb', 'seg')
    C._idx = idx_p
    sets = {}
    for r in range(len(geo) + 1):
        import itertools
        for S in itertools.combinations(geo, r):
            S = frozenset(S)
            times = ts[idx_p] + (tau if 'latency' in S else 0.0)
            rend = C.render(times, D if 'mount' in S else None, outs, full=True, cancel=cancel)
            arng = np.random.default_rng([int(st['seed']), 7, len(S)])
            for x in rend:
                x['rgb'] = _apply_appearance(x['rgb'], app, arng)
            sets[S] = rend
    allg = frozenset(geo)
    R_all, R_none = sets[allg], sets[frozenset()]
    real_l = [reals[i] for i in idx_p]
    # photometric figures against the geometry-corrected render
    sims = [x['rgb'] for x in R_all]
    nr = [MS.noise_sigma(r, st['edge_pct']) for r in real_l]
    ns = [MS.noise_sigma(s, st['edge_pct']) for s in sims]
    sh_r = [MS.sharpness(r, n) for r, n in zip(real_l, nr)]
    sh_s = [MS.sharpness(s, n) for s, n in zip(sims, ns)]
    ratio = (sum(sh_r) / sum(sh_s)) if sum(sh_s) > 0 else None
    sigma_b = MS.blur_to_match(sims, sum(sh_r)) if ratio is not None and ratio < st['blur_min_ratio'] else 0.0
    g, o = MS.fit_affine([MS.blur(x, sigma_b) for x in sims] if sigma_b > 0 else sims, real_l)
    pr, ps = MS.photometric(real_l), MS.photometric(sims)
    emd = MS.emd_channels([np.clip(np.rint(x), 0, 255).astype(np.uint8) for x in real_l],
                          [np.clip(np.rint(x), 0, 255).astype(np.uint8) for x in sims])
    mid = float(np.mean(g * 128.0 + o))             # mid grey (128 DN) of the render, as the recording shows it
    out['photometric'] = {
        'recorded': pr, 'rendered': ps, 'mean_gap_dn': pr['mean'] - ps['mean'],
        'contrast_ratio': (pr['contrast'] / ps['contrast']) if ps['contrast'] > 0 else None,
        'emd_dn': emd, 'gain': [float(x) for x in g], 'offset_dn': [float(x) for x in o],
        'fitted_after_blur': sigma_b > 0,
        'mid_grey_change_dn': mid - 128.0, 'exposure_stops': math.log2(mid / 128.0) if mid > 0 else None,
        'sharpness': {'recorded': float(np.median(sh_r)), 'rendered': float(np.median(sh_s)), 'ratio': ratio},
        'blur_px': sigma_b, 'blur_px_full': sigma_b * stream.width / C.w,
        'noise_dn': {'recorded': float(np.median(nr)), 'rendered': float(np.median(ns)),
                     'gap': float(math.sqrt(max(0.0, np.median(nr) ** 2 - np.median(ns) ** 2)))},
        'compared_on': 'the render with the geometric corrections found' if geo else 'the render as planned'}
    # the image difference over the combinations
    players = geo + ['exposure'] + (['blur'] if sigma_b > 0 else [])

    def corrected(S):
        rs = [x['rgb'] for x in sets[frozenset(S) & allg]]
        if 'blur' in S:
            rs = [MS.blur(x, sigma_b) for x in rs]
        if 'exposure' in S:
            gg, oo = MS.fit_affine(rs, real_l)
            rs = [MS.apply_affine(x, gg, oo) for x in rs]
        return rs

    def dist(S):
        return float(np.mean([MS.mad(a, b) for a, b in zip(real_l, corrected(S))]))
    d0, d_all = dist(frozenset()), dist(frozenset(players))
    phi = MS.shapley(players, dist)
    out['gap'] = {'none_dn': d0, 'all_dn': d_all, 'explained_pct': 100 * (d0 - d_all) / d0 if d0 > 0 else None,
                  'shapley_dn': {k: float(v) for k, v in phi.items()},
                  'alone_dn': {k: d0 - dist(frozenset([k])) for k in players}, 'players': players}
    # geometry: silhouette IoU
    if sensor.mount.on_link:
        out['silhouette'] = {'iou_before': None, 'iou_after': None, 'method': None,
                             'why': 'a camera on the robot has no static background to segment the robot by'}
    else:
        def fg(imgs):
            return RC.foreground([np.clip(np.rint(x), 0, 255).astype(np.uint8) for x in imgs], st['fg_threshold'])
        F = fg(real_l)

        def siou(imgs):
            P_ = fg(imgs)
            if not F or not P_:
                return None
            v = [x for x in (MS.iou(a, b) for a, b in zip(F, P_)) if x is not None]
            return float(np.mean(v)) if v else None
        out['silhouette'] = {'iou_before': siou([x['rgb'] for x in R_none]),
                             'iou_after': siou(corrected(frozenset(players))),
                             'method': 'background difference against the median frame, the same rule on the '
                                       'recorded and on the rendered frames (the dataset stores no masks): the robot '
                                       'where it moves', 'fg_threshold': st['fg_threshold']}
    # depth
    if sensor.kind == 'depth':
        out['depth'] = _depth(C, R_all, ep)
    # timing
    out['latency']['planned_s'] = t_plan
    out['timing'] = _timing(project, stream, ep, sensor.uid, scam.imager, lat, pred_cache, t_plan)
    # distribution distance
    fx = [MS.features(x, st['mmd_size']) for x in real_l]
    planned = [x['rgb'] for x in R_none]
    best = corrected(frozenset(players))
    mb = MS.mmd_test(fx, [MS.features(x, st['mmd_size']) for x in planned], None, st['mmd_permutations'],
                     int(st['seed']))
    ma = MS.mmd_test(fx, [MS.features(x, st['mmd_size']) for x in best], mb.get('sigma'), st['mmd_permutations'],
                     int(st['seed']))
    out['mmd'] = {'planned': mb, 'corrected': ma, 'features': f'{st["mmd_size"][0]} × {st["mmd_size"][1]} RGB '
                  'thumbnails in [0, 1]', 'not_fid': True}
    # thumbnails
    k = max(0, min(int(st['thumbs']), len(idx_p)))
    pick = [int(round(x)) for x in np.linspace(0, len(idx_p) - 1, k)] if k else []
    out['thumbs'] = [{'frame': int(idx_p[j]), 'time_s': float(ts[idx_p[j]]),
                      'real': _jpeg(real_l[j], int(st['thumb_width'])),
                      'planned': _jpeg(planned[j], int(st['thumb_width'])),
                      'corrected': _jpeg(best[j], int(st['thumb_width'])),
                      'mad_planned_dn': MS.mad(real_l[j], planned[j]), 'mad_corrected_dn': MS.mad(real_l[j], best[j])}
                     for j in dict.fromkeys(pick)]
    out['renders'] = C.renders + int(mnt.get('renders') or 0)
    out['causes'] = _causes(out, st, app, ep)
    return out


def _compose_affine(app, g, o) -> dict:
    g0 = np.asarray(app.get('gain') or [1.0, 1.0, 1.0], float)
    o0 = np.asarray(app.get('offset_dn') or [0.0, 0.0, 0.0], float)
    return {'gain': [float(x) for x in np.asarray(g) * g0], 'offset_dn': [float(x) for x in np.asarray(g) * o0 + o]}


def _causes(c: dict, st: dict, app: dict, ep) -> list:
    """The camera's causes: one per correction tried, with its share and its action (module docstring)."""
    gap = c['gap']
    d0 = gap['none_dn']
    rows = []
    ph = c['photometric']
    for f, v in gap['shapley_dn'].items():
        share = 100 * v / d0 if d0 > 0 else 0.0
        row = {'factor': f, 'title': FACTOR_TITLES[f], 'camera': c['key'], 'sensor_uid': c['sensor_uid'],
               'sensor': c['sensor'], 'reduction_dn': float(v), 'share_pct': float(share),
               'page': PAGES[f][0], 'page_title': PAGES[f][1]}
        if f == 'latency':
            tau = c['latency']['latency_s']
            row['figure'] = f'{_signed(tau * 1000)} ms ± {c["latency"]["sigma_s"] * 1000:.0f} ms'
            row['text'] = (f'The frames show the robot {abs(tau) * 1000:.0f} ms {"later" if tau > 0 else "earlier"} '
                           'than their stamps say. Enter the measured latency in the time-sync settings, or re-stamp '
                           'the recording.')
            total = float(tau) + float(c['latency'].get('planned_s') or 0.0)
            row['apply'] = {'kind': 'latency', 'sensor_uid': c['sensor_uid'], 'value_s': -total,
                            'source': f'measured by the sim-to-real gap study (stamp minus exposure, relative to '
                                      f'joint_states), episode {ep.index}'}
        elif f == 'mount':
            m = c['mount']
            row['figure'] = f'{m["apply"]["change_mm"]:.1f} mm, {m["apply"]["change_deg"]:.2f}°'
            row['text'] = ('The camera sits off its planned mount (the silhouette calibration\'s estimate). Apply '
                           'writes the recovered mount to the sensor.')
            row['apply'] = dict(m['apply'])
        elif f == 'exposure':
            stops = ph['exposure_stops']
            row['figure'] = (f'gain {", ".join(f"{x:.2f}" for x in ph["gain"])}, offset '
                             f'{", ".join(_signed(x) for x in ph["offset_dn"])} DN')
            word = 'brighter' if (stops or 0) > 0 else 'darker'
            row['text'] = (f'At mid grey the recording is {abs(ph["mid_grey_change_dn"]):.0f} DN '
                           f'({abs(stops or 0):.2f} stops) {word} than the render, with the gain, offset and colour '
                           'balance shown. Apply sets the render\'s camera response to match; on the camera, '
                           f'{"lower" if (stops or 0) > 0 else "raise"} the exposure or gain to match the plan.')
            row['apply'] = {'kind': 'appearance', 'sensor_uid': c['sensor_uid'],
                            'set': _compose_affine(app, ph['gain'], ph['offset_dn'])}
        elif f == 'blur':
            row['figure'] = f'σ {ph["blur_px_full"]:.2f} px'
            row['text'] = ('The recording is blurrier than the render (sharpness ratio '
                           f'{ph["sharpness"]["ratio"]:.2f}). Apply blurs the render to match; on the camera, check '
                           'the focus, or shorten the exposure if the robot is moving.')
            row['apply'] = {'kind': 'appearance', 'sensor_uid': c['sensor_uid'],
                            'set': {'blur_px': float(math.hypot(float(app.get('blur_px') or 0.0), ph['blur_px']))}}
        if share < st['min_share_pct'] or v < st['min_reduction_dn']:
            row['apply'] = None
            row['text'] = (f'Removes {max(v, 0.0):.1f} DN ({max(share, 0.0):.0f} % of this camera\'s gap), under '
                           f'{st["min_reduction_dn"]:g} DN or {st["min_share_pct"]:g} %: nothing to change.')
        rows.append(row)
    nz = ph['noise_dn']
    if nz['gap'] > 0.5:
        rows.append({'factor': 'noise', 'title': FACTOR_TITLES['noise'], 'camera': c['key'],
                     'sensor_uid': c['sensor_uid'], 'sensor': c['sensor'], 'reduction_dn': None, 'share_pct': None,
                     'figure': f'σ {nz["recorded"]:.1f} DN recorded, {nz["rendered"]:.1f} DN rendered',
                     'text': 'The recording is noisier than the render (noise adds to the image difference rather '
                             'than closing it, so it is not ranked). Apply adds the missing noise to the render.',
                     'page': PAGES['noise'][0], 'page_title': PAGES['noise'][1],
                     'apply': {'kind': 'appearance', 'sensor_uid': c['sensor_uid'],
                               'set': {'noise_dn': float(math.hypot(float(app.get('noise_dn') or 0.0),
                                                                    nz['gap']))}}})
    return rows


def _rank(project, cams: list, st: dict) -> list:
    """Every camera's causes in one list, largest reduction first; a common exposure gain becomes one lighting cause
    (module docstring)."""
    rows = [r for c in cams for r in c['causes']]
    ex = [r for r in rows if r['factor'] == 'exposure' and r.get('apply')]
    if len(ex) >= 2:
        gains = [float(np.mean(next(c for c in cams if c['sensor_uid'] == r['sensor_uid'])['photometric']['gain']))
                 for r in ex]
        gm = float(np.mean(gains))
        if gm > 0 and all(abs(x / gm - 1) <= st['lighting_tol'] for x in gains):
            lights = [L for L in (project.workspace.lights or []) if isinstance(L, dict) and L.get('enabled', True)]
            dims = [float(L.get('dim', 1.0) if isinstance(L.get('dim'), (int, float)) else 1.0) for L in lights]
            offs = [float(np.mean(next(c for c in cams if c['sensor_uid'] == r['sensor_uid'])['photometric']
                                  .get('offset_dn') or [0.0])) for r in ex]
            om = float(np.mean(offs))
            # a scale of the lights gives a gain only: a constant offset (min_reduction_dn or more) is not one
            offset = abs(om) >= st['min_reduction_dn']
            fits = bool(lights) and not offset and all(d * gm <= 1.0 + 1e-9 for d in dims)
            red = float(sum(r['reduction_dn'] for r in ex) / len(ex))
            row = {'factor': 'lighting', 'title': FACTOR_TITLES['lighting'], 'camera': ', '.join(r['camera'] for r in
                                                                                                    ex),
                   'sensor_uid': None, 'sensor': ', '.join(r['sensor'] for r in ex), 'reduction_dn': red,
                   'share_pct': float(np.mean([r['share_pct'] for r in ex])), 'page': PAGES['lighting'][0],
                   'page_title': PAGES['lighting'][1],
                   'figure': f'gain {gm:.2f}' + (f', offset {_signed(om)} DN' if offset else '') +
                             f' in {len(ex)} cameras', 'cameras': [r['sensor_uid'] for r in ex]}
            seen = (f'Every camera sees the scene {gm:.2f}× as bright as the render' if not offset else
                    f'Every camera differs from its render in the same way (gain {gm:.2f}, offset {_signed(om)} DN)')
            if fits:
                row['text'] = (f'{seen}: the room\'s light differs from the project\'s. Apply scales the project\'s '
                               'lights by that factor.')
                row['apply'] = {'kind': 'lights', 'scale': gm}
            else:
                why = ('the project has no lights (MuJoCo\'s default lights are used): add the room\'s lights on the '
                       'Lighting page' if not lights else
                       'a scale of the lights gives a gain, not a constant offset' if offset else
                       'the lights cannot be made that much brighter (dim at most 1)')
                row['text'] = (f'{seen}: the room\'s light differs from the project\'s, but {why}. Apply sets each '
                               'camera\'s render response.')
                row['apply'] = {'kind': 'appearance_many', 'changes': [r['apply'] for r in ex]}
            rows = [r for r in rows if r not in ex] + [row]
    ranked = sorted([r for r in rows if r['reduction_dn'] is not None], key=lambda r: -r['reduction_dn'])
    return ranked + [r for r in rows if r['reduction_dn'] is None]


# ====================================================================== the analysis
def _open(dataset):
    from ..datasets.episode import Dataset, DatasetError, open_dataset
    if isinstance(dataset, Dataset):
        return dataset, False
    try:
        return open_dataset(str(dataset)), True
    except DatasetError as e:
        raise GapError(str(e)) from None


def analyse(project, dataset, settings: dict | None = None, *, progress=None, cancel=None) -> dict:
    """The sim-to-real gap of one episode of dataset (a path or an open Dataset) against the project (module
    docstring). Raises GapError (a plain sentence)."""
    from ..calib import render_calib as RC
    st = settings_of(project, settings)
    report = progress or (lambda f, t='': None)
    cancel = cancel or (lambda: False)
    if not project.robots:
        raise GapError('The sim-to-real gap needs a robot to render: import one first or open a sample.')
    ds, own = _open(dataset)
    try:
        refs = ds.episodes()
        if not refs:
            raise GapError('The dataset has no episodes.')
        idx = refs[0].index if st['episode'] is None else int(st['episode'])
        if idx not in [r.index for r in refs]:
            raise GapError(f'The dataset has no episode {idx}; it has {refs[0].index} to {refs[-1].index}.')
        report(0.0, f'Loading episode {idx}')
        ep = ds.load(idx)
        if not len(ep.t) or np.asarray(ep.q).ndim != 2 or not len(ep.q):
            raise GapError(f'Episode {idx} has no joint states, so the robot cannot be posed at them.')
        ruid, model, pairs, rnotes = RC.robot_map(project, ep)
        sensors = [s for s in project.variant.sensors if s.enabled and s.kind in ('camera', 'depth')]
        if st['cameras']:
            sensors = [s for s in sensors if s.uid in st['cameras'] or s.name in st['cameras']]
        matched, not_rec = [], []
        for s in sensors:
            try:
                matched.append((s, RC.match_stream(ep, s)))
            except StudyError:
                not_rec.append(s.name)
        if not matched:
            raise GapError('No camera of the rig is in the recording (matched by the stream\'s sensor or its name): '
                           'it records ' + ', '.join(ep.cameras or {'nothing'}) + '.')
        dinfo = {'name': os.path.basename(os.path.normpath(str(getattr(ds, 'path', '') or dataset))),
                 'format': ds.format, 'version': ds.version, 'episodes': len(refs)}
        from ..sim.render import Renderer
        from ...validation.mjrender import RenderUnavailable
        try:
            R = Renderer(project, device=st['device'])
        except (RuntimeError, RenderUnavailable) as e:
            raise GapError(f'The MuJoCo scene could not be built or rendered: {e}') from None
        cams, failed = [], []
        rng = np.random.default_rng([int(st['seed']), 11])
        pred_cache = {}
        try:
            for k, (s, cs) in enumerate(matched):
                sub = (lambda a, b: (lambda f, t='': report(a + (b - a) * f, t)))(0.03 + 0.95 * k / len(matched),
                                                                                  0.03 + 0.95 * (k + 1) / len(matched))
                try:
                    cams.append(_camera(R, project, ep, s, cs, model, pairs, ruid, st, rng, pred_cache, sub, cancel))
                except RenderUnavailable as e:
                    raise GapError(f'MuJoCo cannot render here: {e}') from None
                except GapError:
                    raise
                except StudyError as e:
                    failed.append({'sensor': s.name, 'why': str(e)})
            info = R.info
        finally:
            R.close()
    finally:
        if own:
            ds.close()
    if not cams:
        raise GapError('No camera could be compared: ' + '; '.join(f'{f["sensor"]}: {f["why"]}' for f in failed))
    causes = _rank(project, cams, st)
    worst = max(cams, key=lambda c: c['gap']['none_dn'])
    g0 = worst['gap']['none_dn']
    sev = 'good' if g0 <= st['gap_good_dn'] else 'warning' if g0 <= st['gap_warn_dn'] else 'serious' \
        if g0 <= st['gap_serious_dn'] else 'critical'
    top = next((c for c in causes if c.get('apply')), None)
    if top is None:
        advice = ('The render matches the recording within what the corrections explain; the rest is the scene '
                  'itself (textures, shadows, objects the project does not model).' if sev != 'good' else
                  'The render matches the recording: no correction is needed.')
    else:
        whose = 'each camera\'s' if top['factor'] == 'lighting' else \
            top['camera'].replace('observation.images.', '') + '\'s'
        advice = (f'{top["title"]} explains most ({top["share_pct"]:.0f} % of {whose} gap, '
                  f'{top["reduction_dn"]:.1f} DN): {top["text"]}')
    summary = {'gap_dn': g0, 'camera': worst['key'], 'sensor': worst['sensor'], 'status': sev, 'word': WORDS[sev],
               'remaining_dn': worst['gap']['all_dn'], 'explained_pct': worst['gap']['explained_pct'],
               'top': {k: top.get(k) for k in ('factor', 'title', 'camera', 'sensor', 'share_pct', 'reduction_dn',
                                               'figure', 'page', 'page_title')} if top else None,
               'advice': advice}
    notes = list(rnotes) + [f'{f["sensor"]} was not compared: {f["why"]}' for f in failed]
    if not_rec:
        notes.append('Not in the recording: ' + ', '.join(not_rec) + '.')
    return {'dataset': dinfo, 'episode': int(idx), 'summary': summary, 'causes': causes, 'cameras': cams,
            'not_recorded': not_rec, 'apply': top.get('apply') if top else None, 'renderer': info,
            'fid': {'computed': False, 'why': FID_NOTE}, 'notes': notes,
            'settings': {k: v for k, v in st.items() if k not in ('defaults',)}, 'defaults': list(st['defaults'])}


# ====================================================================== Apply
def sections_of(change: dict) -> list:
    k = (change or {}).get('kind')
    return {'mount': ['variant'], 'lights': ['workspace']}.get(k, ['analysis'])


def apply_change(project, change: dict) -> list:
    """Make one suggestion's change on project (in place); returns the store sections it touches. A locked sensor is
    refused (GapError), as everywhere."""
    from ..frames import Pose
    k = (change or {}).get('kind')
    if k == 'mount':
        s = project.variant.sensor(change['sensor_uid'])
        if s is None:
            raise GapError('The camera is not in this rig any more.')
        if s.locked:
            raise GapError(f'{s.name} is locked: unlock it on the Sensors and mounts page to apply the mount.')
        s.mount.pose = Pose.from_dict(change['pose'])
    elif k == 'latency':
        d = dict(project.analysis.studies.get('time_sync') or {})
        lat = dict(d.get('latency_s') or {})
        lat[change['sensor_uid']] = {'value': float(change['value_s']), 'source': str(change.get('source') or '')}
        d['latency_s'] = lat
        project.analysis.studies['time_sync'] = d
    elif k in ('appearance', 'appearance_many'):
        d = dict(project.analysis.studies.get(SID) or {})
        appd = copy.deepcopy(d.get('appearance') or {})
        for ch in (change['changes'] if k == 'appearance_many' else [change]):
            cur = dict(appd.get(ch['sensor_uid']) or {})
            cur.update(copy.deepcopy(ch['set']))
            appd[ch['sensor_uid']] = cur
        d['appearance'] = appd
        project.analysis.studies[SID] = d
    elif k == 'lights':
        sc = float(change['scale'])
        for L in project.workspace.lights or []:
            if isinstance(L, dict) and L.get('enabled', True):
                dim = float(L['dim']) if isinstance(L.get('dim'), (int, float)) else 1.0
                L['dim'] = float(min(1.0, max(0.0, dim * sc)))
    else:
        raise GapError('There is no change to apply.')
    return sections_of(change)


def describe_change(project, change: dict) -> str:
    k = (change or {}).get('kind')

    def name(uid):
        s = project.variant.sensor(uid) if uid else None
        return s.name if s is not None else str(uid)
    if k == 'mount':
        return (f'Apply would move {name(change["sensor_uid"])} {change["change_mm"]:.2f} mm and '
                f'{change["change_deg"]:.3f}° to the mount the silhouettes give.')
    if k == 'latency':
        return (f'Apply would enter a measured latency of {_signed(change["value_s"] * 1000)} ms (stamp minus '
                f'exposure) for {name(change["sensor_uid"])} in the time-sync settings.')
    if k == 'appearance':
        return f'Apply would set the render\'s camera response of {name(change["sensor_uid"])}: ' + \
            ', '.join(f'{a} {v if not isinstance(v, list) else [round(x, 3) for x in v]}' for a, v in
                      change['set'].items()) + '.'
    if k == 'appearance_many':
        return 'Apply would set the render\'s camera response of ' + ', '.join(
            name(c['sensor_uid']) for c in change['changes']) + ' (gain and offset).'
    if k == 'lights':
        return f'Apply would scale every enabled light of the project by {change["scale"]:.2f} (its dim, at most 1).'
    return 'Nothing to change.'
