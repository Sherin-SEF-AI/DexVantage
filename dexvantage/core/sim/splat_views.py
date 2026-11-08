"""Splat views (spec §9.3, Phase 12): a robot-aligned 3D Gaussian Splatting rig (core/sim/splat_rig.py) fitted to a
recorded LeRobot episode (or to a MuJoCo render of the project, for demonstration), then used to

- show what a proposed camera pose would see, before the bracket is printed (the before/after preview, novel camera
  pose replay of a recorded episode at its logged q(t) to a video and per-frame images: replay), and
- render an episode through the perturbed extrinsics of §8.5 (core/tolerance.py, core/exporters/perturbations.py) as
  N extra copies, written as a LeRobot dataset whose every render is marked synthetic (augment),

with the reconstruction quality measured on frames the fit did not use: PSNR, SSIM and LPIPS on held-out views and on
held-out joint states.

    settings_of(project, given=None) -> dict          the settings, labelled defaults filled in ('defaults' lists them)
    analyse(project, settings=None, *, progress=None, cancel=None) -> dict     the study's document (splat_views)
    get_rig(project, settings=None, *, progress=None, cancel=None) -> (SplatRig, info)    fit, or the cached fit
    replay(project, out_dir, settings=None, *, progress=None, cancel=None) -> dict          novel camera pose replay
    augment(project, out_dir, settings=None, *, progress=None, cancel=None) -> dict         perturbation augmentation
    lpips_state() -> 'ready' | 'no package' | 'no weights';  fetch_lpips_weights() -> path  (network, with consent)
    requirement() -> str | None                       None when the splat extra is installed, else the sentence

Source of the rig (setting source; 'auto' takes the episode when a dataset is named, else the demonstration):
- 'episode': views from one episode of a recorded dataset (core/datasets: LeRobot v2.1 and v3.0, and the other
  readers). Every enabled camera of the active rig variant that the episode records (core/calib/render_calib
  match_stream) gives views_per_camera frames (default 24), chosen farthest apart in joint space
  (render_calib.choose_frames), resized to a working width (default 160 px, area averaging). Each view's pose is the
  camera's mount at the frame's logged joint state (scenegeom.sensor_world_pose: a wrist camera moves with q), its
  lens the dataset's intrinsics when stored, else the project's imager (render_calib.camera_model_of); its depth and
  per-pixel owner masks are MuJoCo renders of the project's model at that q (core/sim/render.py), as splat_rig's
  module docstring describes for a real capture. The episode must give at least min_views (default 8) camera poses
  that are view_sep_m (2 cm) or view_sep_deg (5°) apart from each other (defaults; greedy count): a static camera is
  one pose however long it records. A dataset DexVantage generated (core/sim/episodes.py: it carries
  dexvantage_truth.json) is marked synthetic.
- 'demo': splat_rig.synthetic_capture of the project: demo_views MuJoCo renders (default 48) at demo_width ×
  demo_height (160 × 120) from a sphere of poses round the robot at demo_q_states joint states (3: home and 2 drawn
  within ±demo_spread_rad, 0.4 rad). A demonstration of the method on the project's model, labelled synthetic
  everywhere, not a measurement of a real cell.
- 'rig': a rig saved before (setting rig_file); its quality is the one its own fit report states.

Fitting: splat_rig.fit_splat_rig with iterations (default 1500), time_budget_s (120 s), max_gaussians (20 000), seed
and segment ('auto'); the fitted rig is kept in the user's cache folder (XDG_CACHE_HOME/DexVantage/splat), keyed by
the project's content, its robots (the rig is attached to them), the source and these settings, so replay and
augment reuse it. The fit runs on CUDA: device 'auto' refuses when there is no CUDA device, 'cpu' fits on the
processor (slow; small scenes).

Quality (frames not used in fitting; all on the same working size):
- held-out views: views k with k mod holdout_every = holdout_every // 2 (splat_rig.held_out; default every 8th):
  rendered at the view's pose and q and compared with the image: PSNR (10 log10(1 / MSE), images in [0, 1]), SSIM
  (Wang et al. 2004: 11 × 11 Gaussian window, σ 1.5, K1 0.01, K2 0.03, splat_raster.ssim_np) and LPIPS (Zhang et al.
  2018, the lpips package, AlexNet features with its v0.1 linear layers, images scaled to [−1, 1]). Also the PSNR over
  the robot's pixels only (the model's mask).
- held-out joint states: 'demo' renders holdout_q new joint states (default 4, drawn as above with another seed) from
  new poses, MuJoCo against the rig; 'episode' takes, per camera, the holdout_q frames whose joint vector is farthest
  (max-norm) from every frame of that camera used in fitting, the hardest frames of the episode. Not measured for
  'rig'.
LPIPS needs AlexNet's ImageNet weights (torchvision's IMAGENET1K_V1, 244 MB, from download.pytorch.org): they are
used from the cache (XDG_CACHE_HOME/DexVantage/torch) when there, downloaded only with lpips_download true or
fetch_lpips_weights (the Splat views page asks first); without them LPIPS is None and the document says why.

The headline is the held-out views' PSNR with a status against psnr_good_db, psnr_warn_db, psnr_serious_db (defaults
25, 20, 15 dB: DexVantage defaults, not standards). The preview: preview_frames frames (default 3) of the target
camera, the current mount against the proposed pose (setting target_sensor, default the first camera on a link, else
the first camera; target_offset_m and target_rpy_deg in the sensor body frame, a proposed pose change; or target_pose
{'frame': 'world' or a link of the robot, 'position', 'quaternion' (x, y, z, w)}: the sensor body's new pose), with
the recorded frame beside them when the episode records that camera.

Every splat image is a render, never a photograph: the document, every file replay and augment write and every image
the page shows are marked synthetic, and their paths carry 'synthetic'. Inside the files too: each PNG frame has a
tEXt Comment and each MP4 (the replay's, and every video of the augmented dataset) a container title and comment
saying so (RENDER_NOTE), so a file copied out of its folder still says what it is; and a dataset augment wrote is
synthetic when DexVantage opens it again (splat_marked).

References: qureshi2025_splatsim, lou2025_robogs, kerbl2023_3dgs, wang2004_ssim, zhang2018_lpips.

Known limits: those of splat_rig (no densification, no view-dependent colour, no lighting model: a re-posed link keeps
the shading it was captured with; a joint state far from every captured one shows surfaces no view saw; objects the
robot moves stay where they were captured; masks from the model, so a calibration error shifts them); the poses of
an episode's views are the rig's planned mounts at the logged q, so a mount that differs from the plan (measure it with
the silhouette calibration first) or a latency between the images and the joint states blurs the fit; the background
has depth only where the project models a surface, elsewhere the learnt background colour shows; quality falls for a
proposed camera far from every view of the capture (a wrist camera moved 10 cm sees regions no view saw), on specular
and transparent surfaces and in regions no view saw; LPIPS needs the AlexNet weights (a download); one episode per
fit.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import shutil
import time

import numpy as np

from ..jobs import StudyError

SID = 'splat_views'
SCHEMA_REPLAY = 'dexvantage.splat_replay/1'
SCHEMA_AUGMENT = 'dexvantage.splat_augment/1'
SYNTHETIC_TAG = 'synthetic'
REPLAY_FOLDER = 'splat_replay_synthetic'
AUGMENT_FOLDER = 'splat_augment_synthetic'
MANIFEST = 'dexvantage_splat_manifest.json'
SOURCES = ('auto', 'episode', 'demo', 'rig')
DEVICES = ('auto', 'cuda', 'cpu')
MAX_AUGMENT = 100
DEFAULTS = {
    'source': 'auto',               # auto: the episode when a dataset is named, else the demonstration
    'dataset': None,                # the recorded dataset (path); else the audit's / project.data's
    'episode': None,                # episode index (None: the first)
    'cameras': None,                # sensor uids or names (None: every recorded camera)
    'rig_file': None,               # a saved rig (.npz) for source 'rig'
    'views_per_camera': 24,         # default: frames per camera used as views (episode)
    'width': 160,                   # default: working width of an episode's views (px)
    'min_views': 8,                 # default: distinct camera poses an episode must give
    'view_sep_m': 0.02,             # default: two poses are distinct beyond this distance ...
    'view_sep_deg': 5.0,            # ... or this angle
    'demo_views': 48,               # default: views of the demonstration capture
    'demo_width': 160,              # default: its image size (px)
    'demo_height': 120,
    'demo_q_states': 3,             # default: its joint states (home and drawn ones)
    'demo_spread_rad': 0.4,         # default: the drawn joint states lie within ± this of home
    'holdout_q': 4,                 # default: held-out joint states scored (demo: in all; episode: per camera)
    'holdout_every': 8,             # default: every n-th view is held out of the fit
    'iterations': 1500,             # default: fit steps
    'time_budget_s': 120.0,         # default: the fit stops after this
    'max_gaussians': 20000,         # default
    'segment': 'auto',              # default: per-link segmentation (splat_rig METHODS, or auto)
    'device': 'auto',               # auto: CUDA, refused without it; cuda; cpu (slow)
    'seed': 0,
    'target_sensor': None,          # the camera to move (uid or name; default the first on a link, else the first)
    'target_offset_m': [0.0, 0.0, 0.0],      # the proposed change in the sensor body frame (m) ...
    'target_rpy_deg': [0.0, 0.0, 0.0],       # ... and rotation (roll, pitch, yaw, degrees)
    'target_pose': None,            # or the sensor body's new pose {'frame', 'position', 'quaternion' (x, y, z, w)}
    'preview_frames': 3,            # default: frames of the before/after preview
    'thumb_width': 240,             # default: preview image width (px)
    'lpips': 'auto',                # auto: LPIPS when its weights are in the cache; off
    'lpips_download': False,        # default: never download the AlexNet weights unasked
    'psnr_good_db': 25.0,           # default: status thresholds of the headline (dB)
    'psnr_warn_db': 20.0,
    'psnr_serious_db': 15.0,
    'replay_scale': 1.0,            # default: replay at this share of the target camera's stream size
    'replay_max_frames': None,      # default: every state row of the episode
    'codec': 'auto',                # LeRobot writer's codecs: auto (AV1, else H.264), av1, h264
    'augment_n': 4,                 # default: perturbed copies of the episode
    'augment_seed': 0,
    'augment_scale': 1.0,           # default: the §8.5 tolerances scaled by this
    'augment_distribution': 'uniform',
    'augment_version': 'v3.0',      # LeRobot codebase version written
    'augment_frame_scale': 1.0,     # default: augmented frames at this share of the recorded size
}
NUMERIC = ('views_per_camera', 'width', 'min_views', 'view_sep_m', 'view_sep_deg', 'demo_views', 'demo_width',
           'demo_height', 'demo_q_states', 'demo_spread_rad', 'holdout_q', 'holdout_every', 'iterations',
           'time_budget_s', 'max_gaussians', 'seed', 'preview_frames', 'thumb_width', 'psnr_good_db', 'psnr_warn_db',
           'psnr_serious_db', 'replay_scale', 'augment_n', 'augment_seed', 'augment_scale', 'augment_frame_scale')
INTEGER = ('views_per_camera', 'width', 'min_views', 'demo_views', 'demo_width', 'demo_height', 'demo_q_states',
           'holdout_q', 'holdout_every', 'iterations', 'max_gaussians', 'seed', 'preview_frames', 'thumb_width',
           'augment_n', 'augment_seed')
FIRST = ('source', 'device', 'iterations', 'max_gaussians', 'holdout_every', 'seed', 'lpips', 'width')  # shown first
WORDS = {'good': 'close to the held-out frames', 'warning': 'visible differences', 'serious': 'large differences',
         'critical': 'not usable'}
INSTALL = 'pip install "dexvantage[splat]" --extra-index-url https://download.pytorch.org/whl/cu130'
LPIPS_URL = 'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth'
LPIPS_FILE = 'alexnet-owt-7be5be79.pth'
LPIPS_MB = 244                      # 244 408 911 bytes, measured 2026-11-08
LPIPS_ABOUT = ('LPIPS (Zhang et al. 2018) compares AlexNet features; it needs AlexNet\'s ImageNet weights '
               f'(torchvision\'s IMAGENET1K_V1, {LPIPS_MB} MB, from download.pytorch.org, distributed by the PyTorch '
               'project; DexVantage does not bundle them).')


# =========================================================================== availability
def requirement() -> str | None:
    """None when PyTorch (the splat extra) imports; else the sentence saying the splat extra is not installed."""
    import importlib.util

    from ... import extras
    try:
        if importlib.util.find_spec('torch') is not None:     # cheap: the page asks without importing torch
            return None
    except (ImportError, ValueError):
        pass
    return extras.missing('splat', f'Splat views need PyTorch: the splat extra is not installed ({INSTALL}).',
                          ', adding --extra-index-url https://download.pytorch.org/whl/cu130 for PyTorch\'s CUDA build')


def _need_torch():
    why = requirement()
    if why:
        raise StudyError(why)


def _device(st: dict) -> str:
    from .. import splat_raster as SR
    d = st['device']
    if d == 'cpu':
        return 'cpu'
    if not SR.cuda_ok():
        raise StudyError('There is no CUDA device here, and the splat fit runs on a CUDA GPU: set the device to cpu '
                         'to fit a small scene on the processor (slow), or run on a machine with an NVIDIA GPU.')
    return 'cuda'


def cache_dir(*parts) -> str:
    base = os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache')
    return os.path.join(base, 'DexVantage', *parts)


# =========================================================================== settings
def settings_of(project, given: dict | None = None) -> dict:
    """The study's settings: the project's (analysis.studies['splat_views']) laid over DEFAULTS, then given; 'defaults'
    lists the keys left at their default. Raises StudyError for a value it cannot use."""
    st = dict(DEFAULTS)
    user = {**((project.analysis.studies.get(SID) or {}) if project is not None else {}), **(given or {})}
    for k, v in user.items():
        if k in DEFAULTS and v is not None:
            st[k] = v
    for k in NUMERIC:
        try:
            v = float(st[k])
        except (TypeError, ValueError):
            raise StudyError(f'The splat setting {k} must be a number; it is {st[k]!r}.') from None
        if not math.isfinite(v) or v < 0:
            raise StudyError(f'The splat setting {k} must be zero or more; it is {st[k]!r}.')
        st[k] = int(round(v)) if k in INTEGER else v
    if st['source'] not in SOURCES:
        raise StudyError(f'The splat source must be one of {", ".join(SOURCES)}; it is {st["source"]!r}.')
    if st['device'] not in DEVICES:
        raise StudyError(f'The splat device must be auto, cuda or cpu; it is {st["device"]!r}.')
    if st['lpips'] not in ('auto', 'off'):
        raise StudyError('The splat setting lpips must be auto or off.')
    if st['iterations'] < 1 or st['max_gaussians'] < 1 or st['time_budget_s'] <= 0:
        raise StudyError('The splat fit needs at least one iteration, a positive time budget and Gaussian budget.')
    if st['width'] < 16 or st['demo_width'] < 16 or st['demo_height'] < 12:
        raise StudyError('The splat working images must be at least 16 × 12 px.')
    if not 1 <= st['augment_n'] <= MAX_AUGMENT:
        raise StudyError(f'Augmentation writes 1 to {MAX_AUGMENT} perturbed copies; it was asked for '
                         f'{st["augment_n"]}.')
    if st['replay_scale'] <= 0 or st['augment_frame_scale'] <= 0:
        raise StudyError('The replay and augmentation image scales must be above zero.')
    for k in ('target_offset_m', 'target_rpy_deg'):
        try:
            a = np.asarray(st[k], float).reshape(3)
        except (TypeError, ValueError):
            raise StudyError(f'The splat setting {k} must be three numbers.') from None
        if not np.isfinite(a).all():
            raise StudyError(f'The splat setting {k} must be three finite numbers.')
        st[k] = [float(x) for x in a]
    if st['target_pose'] is not None and not isinstance(st['target_pose'], dict):
        raise StudyError('The splat target pose must be {"frame", "position", "quaternion"}.')
    if st['replay_max_frames'] is not None:
        try:
            st['replay_max_frames'] = max(1, int(st['replay_max_frames']))
        except (TypeError, ValueError):
            raise StudyError('replay_max_frames must be a whole number.') from None
    st['defaults'] = sorted(k for k in DEFAULTS if k not in user or user[k] is None)
    return st


def _fit_settings(st: dict, device: str):
    from . import splat_rig as SRG
    return SRG.FitSettings(iterations=st['iterations'], time_budget_s=st['time_budget_s'], seed=st['seed'],
                           device=device, max_gaussians=st['max_gaussians'], holdout_every=st['holdout_every'],
                           segment=st['segment'])


def dataset_path(project, st: dict) -> str:
    """The recorded dataset: the settings' 'dataset', else the one the project names (core/studies/audit
    dataset_path_for: the audit's, then project.data's). Raises StudyError when none is named or it is missing."""
    from ..studies import audit as AU
    p = st.get('dataset')
    if p:
        from ..datasets.hub import is_hub_ref
        if not os.path.exists(str(p)) and not is_hub_ref(str(p)):
            raise StudyError(f'The dataset {os.path.basename(os.path.normpath(str(p)))} is not there (nothing at that '
                             'path): choose it again, or check the path.')
        return str(p)
    return AU.dataset_path_for(project, SID)


def _source(project, st: dict) -> tuple:
    """(kind, dataset path or None)."""
    src = st['source']
    if src == 'rig' or (src == 'auto' and st.get('rig_file')):
        if not st.get('rig_file'):
            raise StudyError('Source rig needs a rig file (rig_file): one a splat fit saved (.npz).')
        return 'rig', None
    if src == 'demo':
        return 'demo', None
    try:
        path = dataset_path(project, st)
    except StudyError as e:
        if src == 'auto' and str(e).startswith('Choose a dataset'):
            return 'demo', None
        if src == 'episode' and str(e).startswith('Choose a dataset'):
            raise StudyError('Choose the recorded LeRobot dataset to fit the splat rig to (Dataset… on the Splat '
                             'views page, or --dataset PATH), or use the demonstration source.') from None
        raise
    return 'episode', path


# =========================================================================== small helpers
def _even(n: float) -> int:
    return max(2, int(round(n)) // 2 * 2)


def _resize(img, w, h, nearest=False):
    import cv2
    if img.shape[1] == w and img.shape[0] == h:
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)


def _jpeg(img, width: int) -> str:
    import cv2
    a = np.clip(np.asarray(img, float), 0, 255).astype(np.uint8)
    h, w = a.shape[:2]
    if w != width:
        a = cv2.resize(a, (int(width), max(1, int(round(h * width / w)))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(a, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode('ascii') if ok else ''


def _u8(rgb01) -> np.ndarray:
    return np.clip(np.rint(np.asarray(rgb01, float) * 255.0), 0, 255).astype(np.uint8)


def _num(x, nd=4):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def distinct_views(Ts, sep_m: float, sep_deg: float) -> int:
    """Greedy count of camera poses (4 × 4) each at least sep_m or sep_deg from every pose counted before it."""
    kept = []
    for T in Ts:
        T = np.asarray(T, float)
        new = True
        for K in kept:
            d = float(np.linalg.norm(T[:3, 3] - K[:3, 3]))
            c = (np.trace(K[:3, :3].T @ T[:3, :3]) - 1.0) / 2.0
            a = math.degrees(math.acos(min(1.0, max(-1.0, c))))
            if d < sep_m and a < sep_deg:
                new = False
                break
        if new:
            kept.append(T)
    return len(kept)


def _status(st: dict, psnr) -> str:
    if psnr is None:
        return 'warning'
    if psnr >= st['psnr_good_db']:
        return 'good'
    if psnr >= st['psnr_warn_db']:
        return 'warning'
    return 'serious' if psnr >= st['psnr_serious_db'] else 'critical'


# =========================================================================== LPIPS
def lpips_weights_path() -> str:
    return cache_dir('torch', 'hub', 'checkpoints', LPIPS_FILE)


def lpips_state() -> str:
    """'ready' (the lpips package and the AlexNet weights are here), 'no package' or 'no weights'."""
    import importlib.util
    try:
        if importlib.util.find_spec('lpips') is None or importlib.util.find_spec('torchvision') is None:
            return 'no package'
    except (ImportError, ValueError):
        return 'no package'
    return 'ready' if os.path.isfile(lpips_weights_path()) else 'no weights'


def lpips_note(state: str | None = None) -> str:
    state = state or lpips_state()
    if state == 'ready':
        return 'LPIPS: AlexNet features with the lpips package\'s v0.1 linear layers (images scaled to [−1, 1]).'
    if state == 'no package':
        return ('LPIPS is not computed: the lpips package is not installed (it comes with the splat extra, '
                f'{INSTALL}).')
    return ('LPIPS is not computed: AlexNet\'s weights are not in the cache. ' + LPIPS_ABOUT +
            ' Get them on the Splat views page (it asks first), or with dv.splat.fetch_lpips_weights().')


def fetch_lpips_weights(progress: bool = False) -> str:
    """Download AlexNet's ImageNet weights for LPIPS into the cache (network; checked against the file's SHA-256
    prefix 7be5be79 in its name). Returns the path. Raises StudyError (a sentence) when it cannot."""
    _need_torch()
    path = lpips_weights_path()
    if os.path.isfile(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        import torch.hub
        torch.hub.download_url_to_file(LPIPS_URL, path, hash_prefix='7be5be79', progress=progress)
    except Exception as e:                               # noqa: BLE001 — a network or disk error: one sentence
        try:
            os.remove(path)
        except OSError:
            pass
        raise StudyError(f'The AlexNet weights for LPIPS could not be downloaded ({type(e).__name__}: {e}).') from None
    return path


class _Lpips:
    """lpips.LPIPS(net='alex') with its backbone loaded from the cached AlexNet weights (never downloaded here)."""

    def __init__(self, device):
        import warnings

        import lpips
        import torch
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = lpips.LPIPS(net='alex', pnet_rand=True, verbose=False)
            sd = torch.load(lpips_weights_path(), map_location='cpu', weights_only=True)
        net = m.net.state_dict()
        new = {}
        for k in net:                                    # sliceN.I.weight <- features.I.weight
            i = k.split('.', 1)[1]
            src = f'features.{i}'
            if src not in sd:
                raise StudyError('The cached AlexNet weights do not have the layers LPIPS needs; delete '
                                 f'{LPIPS_FILE} from the cache and get them again.')
            new[k] = sd[src]
        m.net.load_state_dict(new)
        self.m = m.to(device).eval()
        self.dev = device

    def __call__(self, a, b) -> float:
        import torch
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(a, np.float32) * 2 - 1, device=self.dev).permute(2, 0, 1)[None]
            y = torch.as_tensor(np.asarray(b, np.float32) * 2 - 1, device=self.dev).permute(2, 0, 1)[None]
            return float(self.m(x, y).reshape(-1)[0])


def _lpips_for(st: dict, device):
    """(_Lpips or None, note)."""
    if st['lpips'] == 'off':
        return None, 'LPIPS is not computed: the setting lpips is off.'
    s = lpips_state()
    if s == 'no weights' and st['lpips_download']:
        fetch_lpips_weights()
        s = lpips_state()
    if s != 'ready':
        return None, lpips_note(s)
    return _Lpips(device), lpips_note('ready')


# =========================================================================== captures
def _demo_capture(project, st: dict, progress, cancel):
    from . import splat_rig as SRG
    qs = SRG.random_qs(project, st['demo_q_states'], st['seed'], st['demo_spread_rad'])
    cap = SRG.synthetic_capture(project, n_views=st['demo_views'], qs=qs, width=st['demo_width'],
                                height=st['demo_height'], seed=st['seed'], progress=progress, cancel=cancel)
    cap.source['label'] = 'a MuJoCo render of the project (a demonstration: synthetic)'
    return cap


class _Episode:
    """One episode of a dataset matched to the project: its robot, the rig cameras it records."""

    def __init__(self, project, path: str, st: dict):
        from ..calib import render_calib as RC
        from ..datasets.episode import DatasetError, open_dataset
        try:
            self.ds = open_dataset(str(path))
        except DatasetError as e:
            raise StudyError(str(e)) from None
        refs = self.ds.episodes()
        if not refs:
            self.ds.close()
            raise StudyError('The dataset has no episodes.')
        idx = refs[0].index if st['episode'] is None else int(st['episode'])
        if idx not in [r.index for r in refs]:
            self.ds.close()
            raise StudyError(f'The dataset has no episode {idx}; it has {refs[0].index} to {refs[-1].index}.')
        self.index = idx
        self.ep = self.ds.load(idx)
        if not len(self.ep.t) or np.asarray(self.ep.q).ndim != 2 or not len(self.ep.q):
            self.ds.close()
            raise StudyError(f'Episode {idx} has no joint states, so the robot cannot be posed at them.')
        self.ruid, self.model, self.pairs, self.notes = RC.robot_map(project, self.ep)
        sensors = [s for s in project.variant.sensors if s.enabled and s.kind in ('camera', 'depth')]
        if st['cameras']:
            sensors = [s for s in sensors if s.uid in st['cameras'] or s.name in st['cameras']]
        self.matched, self.not_recorded = [], []
        for s in sensors:
            try:
                self.matched.append((s, RC.match_stream(self.ep, s)))
            except StudyError:
                self.not_recorded.append(s.name)
        try:
            truth = self.ds.truth() if hasattr(self.ds, 'truth') else None
        except Exception:                                # noqa: BLE001 — no truth file: a recording
            truth = None
        splat = splat_marked(getattr(self.ds, 'info', None))
        self.synthetic = (bool(truth) or splat or
                          any((c.meta or {}).get('synthetic') for c in self.ep.cameras.values()))
        kind = 'splat renders' if splat else ('DexVantage synthetic dataset' if self.synthetic else None)
        self.info = {'name': os.path.basename(os.path.normpath(str(path))), 'format': self.ds.format,
                     'version': self.ds.version, 'episodes': len(refs), 'episode': int(idx),
                     'rows': int(len(self.ep.t)), 'duration_s': _num(self.ep.duration_s, 3),
                     'synthetic': self.synthetic, 'synthetic_kind': kind}

    def q_at(self, times) -> np.ndarray:
        from ..calib import render_calib as RC
        return RC.q_at(self.ep, self.model, self.pairs, np.asarray(times, float))

    def close(self):
        try:
            self.ds.close()
        except Exception:                                # noqa: BLE001
            pass


def _episode_capture(project, E: _Episode, st: dict, R, progress, cancel):
    """(Capture, held-out-q views [CaptureView], notes)."""
    from ..calib import render_calib as RC
    from ..scenegeom import sensor_world_pose
    from . import splat_rig as SRG
    if not E.matched:
        raise StudyError('No camera of the rig is in the recording (matched by the stream\'s sensor or its name): it '
                         'records ' + ', '.join(E.ep.cameras or {'nothing'}) + '.')
    views, hold, notes, per_cam = [], [], [], {}
    n_cam = len(E.matched)
    for ci, (s, cs) in enumerate(E.matched):
        if cancel is not None and cancel():
            raise StudyError('The splat fit was cancelled.')
        sc = R.camera(s.uid)
        cam_full = RC.camera_model_of(cs, sc, notes)
        ts = np.asarray(cs.timestamps, float)
        if not len(ts):
            notes.append(f'{s.name}: the stream has no frames.')
            continue
        Q = E.q_at(ts)
        pick = list(RC.choose_frames(Q, min(st['views_per_camera'], len(ts))))
        chosen = set(pick)
        rest = [i for i in range(len(ts)) if i not in chosen]
        far = []
        if rest and st['holdout_q']:
            D = np.abs(Q[rest][:, None, :] - Q[pick][None, :, :]).max(-1).min(1)
            far = [rest[i] for i in np.argsort(-D)[: st['holdout_q']]]
        want = sorted(set(pick) | set(far))
        frames = dict(cs.frames(want))
        w = min(int(st['width']), cam_full.width)
        h = max(2, int(round(cam_full.height * w / cam_full.width)))
        cam = cam_full.scaled(w, h)
        jobs = [{'sensor_uid': s.uid, 'q': {E.ruid: Q[i]}, 'outputs': ('depth', 'seg'), 'camera_model': cam}
                for i in pick]
        imgs = R.render(jobs) if jobs else []
        for i, img in zip(pick, imgs):
            q = {E.ruid: Q[i].copy()}
            T = sensor_world_pose(project, s, q).matrix() @ sc.T_sensor_optical
            rgb = _resize(np.asarray(frames[i], np.uint8), w, h)
            views.append(SRG.CaptureView(f'{s.name} · frame {i}', rgb, cam, T, q,
                                         np.asarray(img['depth'], np.float32), np.asarray(img['seg'], np.int32),
                                         s.uid))
        for i in far:
            q = {E.ruid: Q[i].copy()}
            T = sensor_world_pose(project, s, q).matrix() @ sc.T_sensor_optical
            hold.append(SRG.CaptureView(f'{s.name} · frame {i}', _resize(np.asarray(frames[i], np.uint8), w, h),
                                        cam, T, q, None, None, s.uid))
        per_cam[s.uid] = {'sensor': s.name, 'stream': cs.key, 'views': len(pick), 'held_out_q': len(far),
                          'size': [w, h], 'on_link': bool(s.mount.on_link)}
        if progress is not None:
            progress((ci + 1) / n_cam, f'Capture views of {s.name}')
    # interleave the cameras' views so every n-th held-out view samples each camera
    order = sorted(range(len(views)), key=lambda k: (k % max(1, st['views_per_camera']), k))
    views = [views[k] for k in order]
    n_dist = distinct_views([v.T_world_optical for v in views], st['view_sep_m'], st['view_sep_deg'])
    if n_dist < st['min_views']:
        raise StudyError(f'The episode has too few distinct views to fit a splat rig: {n_dist} camera pose'
                         f'{"" if n_dist == 1 else "s"} at least {st["view_sep_m"] * 100:g} cm or '
                         f'{st["view_sep_deg"]:g}° apart, where {st["min_views"]} are needed (DexVantage defaults). '
                         'A static camera is one pose however long it records: a camera on the robot\'s wrist that '
                         'moves through the episode, or more cameras, give more views; or use the demonstration '
                         'source (a MuJoCo render of the project).')
    kind = E.info.get('synthetic_kind')
    cap = SRG.Capture(views, list(R.build.owners), E.synthetic,
                      {'kind': 'recorded episode' + (f' ({kind}, synthetic)' if kind else ''),
                       'dataset': E.info, 'cameras': per_cam, 'distinct_views': n_dist,
                       'label': f'episode {E.index} of {E.info["name"]}'})
    return cap, hold, notes + list(E.notes)


# =========================================================================== the rig
def _rig_key(project, st: dict, kind: str, path: str | None) -> str:
    from .. import provenance as PV
    from ..studies import audit as AU
    from . import splat_rig as SRG
    keys = ('episode', 'cameras', 'views_per_camera', 'width', 'demo_views', 'demo_width', 'demo_height',
            'demo_q_states', 'demo_spread_rad', 'holdout_every', 'iterations', 'time_budget_s', 'max_gaussians',
            'segment', 'device', 'seed')
    # the project without this study's own settings: those the fit reads are in keys (the dataset by its
    # fingerprint), and the rest (the proposed pose, LPIPS, the thresholds, replay and augmentation) do not change
    # the fit, so a new proposed pose, replay or augment reuses the rig instead of fitting it again
    if (project.analysis.studies or {}).get(SID) is not None:
        project = project.snapshot()
        project.analysis.studies.pop(SID, None)
    d = {'schema': SRG.SCHEMA, 'project': PV.project_digest(project)['sha256'], 'kind': kind,
         'robots': [r.uid for r in project.robots],      # the rig is attached to these robots (their uids)
         'dataset': AU.fingerprint(path) if path else None, **{k: st[k] for k in keys}}
    return hashlib.sha256(PV.canonical(d)).hexdigest()[:32]


def get_rig(project, settings: dict | None = None, *, progress=None, cancel=None, keep_capture: bool = False) -> tuple:
    """(SplatRig, info): the rig of the settings' source, from the cache when it was fitted before with the same
    inputs, else fitted now (and cached). info: {'kind', 'cached', 'path', 'fit', 'capture', 'episode', 'notes',
    'synthetic'} plus, with keep_capture, the capture and the held-out-q views ('_capture', '_hold')."""
    from . import splat_rig as SRG
    st = settings if isinstance(settings, dict) and 'defaults' in settings else settings_of(project, settings)
    _need_torch()
    report = progress or (lambda f, t='': None)
    kind, path = _source(project, st)
    if kind == 'rig':
        try:
            rig = SRG.SplatRig.load(st['rig_file'])
        except ValueError as e:
            raise StudyError(str(e)) from None
        rig.check_robots(project)
        return rig, {'kind': 'rig', 'cached': True, 'path': os.path.abspath(str(st['rig_file'])),
                     'fit': (rig.info or {}).get('fit'), 'capture': (rig.info or {}).get('capture'), 'episode': None,
                     'notes': ['The rig was loaded from its file; its quality is the one its fit report states.'],
                     'synthetic': bool(rig.synthetic)}
    device = _device(st)
    key = _rig_key(project, st, kind, path)
    rp = cache_dir('splat', f'{key}.npz')
    need_cap = keep_capture or not os.path.isfile(rp)
    cap, hold, notes, E, dinfo = None, [], [], None, None
    if need_cap:
        from ...validation.mjrender import RenderUnavailable
        from .render import Renderer
        try:
            if kind == 'demo':
                report(0.0, 'Rendering the demonstration capture (MuJoCo)')
                cap = _demo_capture(project, st, lambda f, t='': report(0.1 * f, t), cancel)
            else:
                E = _Episode(project, path, st)
                dinfo = E.info
                try:
                    R = Renderer(project)
                except (RuntimeError, RenderUnavailable) as e:
                    raise StudyError(f'The MuJoCo scene could not be built or rendered: {e}') from None
                try:
                    cap, hold, notes = _episode_capture(project, E, st, R, lambda f, t='': report(0.1 * f, t),
                                                        cancel)
                finally:
                    R.close()
        except RenderUnavailable as e:
            raise StudyError(f'MuJoCo cannot render here: {e}') from None
        finally:
            if E is not None:
                E.close()
    cached = os.path.isfile(rp)
    if cached:
        try:
            rig = SRG.SplatRig.load(rp)
            rig.check_robots(project)
        except (ValueError, StudyError):
            cached = False
    if not cached:
        if cap is None:                                       # the cache file went bad after the check above
            return get_rig(project, st, progress=progress, cancel=cancel, keep_capture=True)
        report(0.1, 'Fitting the splat rig')
        rig, fit = SRG.fit_splat_rig(project, cap, _fit_settings(st, device),
                                     lambda f, t='': report(0.1 + 0.75 * f, t), cancel)
        if fit.get('cancelled'):
            raise StudyError('The splat fit was cancelled; nothing was kept.')
        rig.info['source'] = {'kind': kind, 'label': cap.source.get('label'), 'dataset': dinfo}
        try:
            rig.save(rp)
        except OSError as e:
            notes.append(f'The fitted rig could not be kept in the cache ({e.strerror or e}).')
    info = {'kind': kind, 'cached': cached, 'path': rp if os.path.isfile(rp) else None,
            'fit': (rig.info or {}).get('fit'), 'capture': (rig.info or {}).get('capture'),
            'episode': dinfo or ((rig.info or {}).get('source') or {}).get('dataset'), 'notes': notes,
            'synthetic': bool(rig.synthetic), 'device': device, 'dataset_path': path}
    if keep_capture:
        info['_capture'], info['_hold'] = cap, hold
    return rig, info


# =========================================================================== the target camera
def target_sensor(project, st: dict):
    from .splat_rig import stream_imager
    cams = [s for s in project.variant.sensors if s.enabled and s.kind in ('camera', 'depth')]
    want = st.get('target_sensor')
    if want:
        s = next((x for x in cams if want in (x.uid, x.name)), None)
        if s is None:
            raise StudyError(f'There is no enabled camera {want} in the active rig variant to move.')
    else:
        s = next((x for x in cams if x.mount.on_link), cams[0] if cams else None)
        if s is None:
            raise StudyError('The rig has no camera to move: add one first (Sensors).')
    im = stream_imager(s)
    if im is None:
        raise StudyError(f'{s.name} has no imager to render.')
    return s, im


def _delta(st: dict) -> np.ndarray:
    from ..frames import Pose, quat_from_rpy
    r = [math.radians(x) for x in st['target_rpy_deg']]
    return Pose(st['target_offset_m'], quat_from_rpy(*r)).matrix()


def target_pose_fn(project, st: dict, sensor, imager, proposed: bool):
    """fn(q) -> T_world_optical of the target camera at joint state q: the current mount, or the proposed pose."""
    from ..frames import Pose
    from ..scenegeom import sensor_world_pose
    T_bo = imager.pose_in_body.matrix()
    tp = st.get('target_pose') if proposed else None
    if tp:
        try:
            P = Pose(tp.get('position', (0, 0, 0)), tp.get('quaternion', (0, 0, 0, 1))).matrix()
        except (TypeError, ValueError):
            raise StudyError('The splat target pose needs a position (3 numbers) and a quaternion (x, y, z, w).') \
                from None
        frame = str(tp.get('frame') or 'world')
        if frame == 'world':
            return lambda q: P @ T_bo
        from ..robot.model import get_model
        rob = next((r for r in project.robots if frame in get_model(project, r.uid).link_names), None)
        if rob is None:
            raise StudyError(f'The target pose\'s frame {frame} is neither the world nor a link of the robot.')
        m = get_model(project, rob.uid)

        def on_link(q):
            qq = (q or {}).get(rob.uid)
            return m.link_pose(m.home if qq is None else np.asarray(qq, float), frame) @ P @ T_bo
        return on_link
    D = _delta(st) if proposed else np.eye(4)
    return lambda q: sensor_world_pose(project, sensor, q).matrix() @ D @ T_bo


def _proposal_words(st: dict) -> str:
    tp = st.get('target_pose')
    if tp:
        p = [round(float(x), 4) for x in tp.get('position', (0, 0, 0))]
        return f'the sensor body moved to {p} m in {tp.get("frame") or "world"}'
    o, r = st['target_offset_m'], st['target_rpy_deg']
    if not any(o) and not any(r):
        return 'no pose change entered: the proposed camera is the current one'
    return (f'moved by ({", ".join(f"{1000 * x:+.0f}" for x in o)}) mm and turned by '
            f'({", ".join(f"{x:+.1f}" for x in r)})° (roll, pitch, yaw) in its own frame')


# =========================================================================== quality
def _score(rig, project, views, dev, lp, robot_ids=None) -> dict:
    from .. import splat_reproject as RP
    per = []
    for v in views:
        out = rig.render(project, v.camera, v.T_world_optical, v.q, backend='torch', device=dev)
        pred = np.clip(out['rgb'], 0, 1)
        tgt = np.asarray(v.rgb, float) / 255.0
        s = RP.image_scores(pred, tgt)
        row = {'view': v.name, 'psnr_db': _num(s['psnr_db'], 3), 'ssim': _num(s['ssim'], 4),
               'lpips': _num(lp(pred, tgt), 4) if lp is not None else None, 'robot_psnr_db': None}
        if robot_ids is not None and v.seg is not None:
            mask = np.isin(v.seg, robot_ids)
            if mask.any():
                row['robot_psnr_db'] = _num(RP.image_scores(pred, tgt, mask)['psnr_db'], 3)
        per.append(row)

    def mean(k):
        xs = [r[k] for r in per if r[k] is not None]
        return _num(np.mean(xs), 4) if xs else None
    return {'psnr_db': mean('psnr_db'), 'ssim': mean('ssim'), 'lpips': mean('lpips'),
            'robot_psnr_db': mean('robot_psnr_db'), 'views': len(per), 'per_view': per}


def _torch_device(device: str):
    import torch
    from .. import splat_raster as SR
    return SR.device_for('cuda') if device == 'cuda' else torch.device('cpu')


def _quality(project, rig, info: dict, st: dict, dev, report) -> dict:
    from . import splat_rig as SRG
    cap = info.get('_capture')
    if cap is None:
        f = info.get('fit') or {}
        return {'measured': False, 'held_out_views': f.get('held_out_after'), 'before_fit': f.get('held_out_before'),
                'held_out_q': None, 'lpips_note': 'Not measured again: the rig was loaded from its file; these are its '
                                                  'fit report\'s figures (no LPIPS).'}
    lp, note = _lpips_for(st, dev)
    robot_ids = np.array([i for i, o in enumerate(cap.owners) if o and o[0] == 'robot'], np.int32)
    held = [v for k, v in enumerate(cap.views) if SRG.held_out(k, st['holdout_every'])]
    train = [v for k, v in enumerate(cap.views) if not SRG.held_out(k, st['holdout_every'])]
    report(0.88, 'Scoring the held-out views')
    q_held = _score(rig, project, held, dev, lp, robot_ids)
    q_train = _score(rig, project, train[:: max(1, len(train) // 8)], dev, None, robot_ids)
    report(0.92, 'Scoring the held-out joint states')
    hold = info.get('_hold') or []
    how = 'the episode\'s frames farthest in joint space from every frame used in fitting'
    if info['kind'] == 'demo' and st['holdout_q']:
        qn = SRG.random_qs(project, st['holdout_q'] + 1, st['seed'] + 101, st['demo_spread_rad'])[1:]
        novel = SRG.synthetic_capture(project, n_views=st['holdout_q'], qs=qn, width=st['demo_width'],
                                      height=st['demo_height'], seed=st['seed'] + 202)
        hold = novel.views
        how = (f'{st["holdout_q"]} joint states drawn within ±{st["demo_spread_rad"]:g} rad of home with another seed, '
               'from new poses, against MuJoCo\'s renders')
    q_q = _score(rig, project, hold, dev, lp, robot_ids) if hold else None
    if q_q is not None:
        q_q['how'] = how
    f = info.get('fit') or {}
    return {'measured': True, 'held_out_views': q_held, 'held_out_q': q_q, 'train_views': q_train,
            'before_fit': f.get('held_out_before'), 'lpips_note': note,
            'how': f'views k with k mod {st["holdout_every"]} = {st["holdout_every"] // 2} were left out of the fit '
                   '(splat_rig.held_out); joint states: ' + how}


# =========================================================================== preview
def _preview(project, rig, st: dict, info: dict, dev) -> dict:
    """The target camera, current mount against the proposed pose, at preview_frames frames."""
    s, im = target_sensor(project, st)
    cam = im.camera
    w = min(cam.width, max(st['thumb_width'] * 2, 160))
    cam = cam.scaled(_even(w), _even(cam.height * w / cam.width))
    cur = target_pose_fn(project, st, s, im, False)
    new = target_pose_fn(project, st, s, im, True)
    frames, rec = [], {}
    path = info.get('dataset_path')
    qs = []
    if info['kind'] == 'episode' and path:
        E = _Episode(project, path, st)
        try:
            t = np.asarray(E.ep.t, float)
            n = max(1, st['preview_frames'])
            rows = sorted(set(int(round(x)) for x in np.linspace(0, len(t) - 1, n)))
            Q = E.q_at(t[rows])
            qs = [({E.ruid: Q[k]}, float(t[r] - t[0]), int(r)) for k, r in enumerate(rows)]
            from ..calib import render_calib as RC
            try:
                cs = RC.match_stream(E.ep, s)
                ts = np.asarray(cs.timestamps, float)
                idx = [int(np.argmin(np.abs(ts - t[r]))) for r in rows]
                got = dict(cs.frames(sorted(set(idx))))
                rec = {r: got[i] for r, i in zip(rows, idx) if i in got}
            except StudyError:
                rec = {}
        finally:
            E.close()
    else:
        from . import splat_rig as SRG
        q0 = SRG.random_qs(project, max(1, st['preview_frames']), st['seed'], st['demo_spread_rad'])
        qs = [(q, None, None) for q in q0]
    for q, t, r in qs:
        a = rig.render(project, cam, cur(q), q, backend='torch', device=dev)['rgb']
        b = rig.render(project, cam, new(q), q, backend='torch', device=dev)['rgb']
        row = {'time_s': _num(t, 3), 'row': r, 'current': _jpeg(_u8(a), st['thumb_width']),
               'proposed': _jpeg(_u8(b), st['thumb_width']), 'synthetic': True}
        if r in rec:
            row['recorded'] = _jpeg(rec[r], st['thumb_width'])
        frames.append(row)
    return {'sensor_uid': s.uid, 'sensor': s.name, 'imager': getattr(im, 'name', ''), 'on_link': bool(s.mount.on_link),
            'proposal': _proposal_words(st), 'offset_m': st['target_offset_m'], 'rpy_deg': st['target_rpy_deg'],
            'pose': st.get('target_pose'), 'size': [cam.width, cam.height], 'frames': frames, 'synthetic': True}


# =========================================================================== the study
def analyse(project, settings: dict | None = None, *, progress=None, cancel=None) -> dict:
    """The splat_views study's document (module docstring)."""
    st = settings_of(project, settings)
    _need_torch()
    if not project.robots:
        raise StudyError('Splat views need a robot: import one first (Robot ▸ Import robot…) or open a sample.')
    report = progress or (lambda f, t='': None)
    rig, info = get_rig(project, st, progress=report, cancel=cancel, keep_capture=True)
    dev = _torch_device(info.get('device') or ('cuda' if st['device'] != 'cpu' else 'cpu'))
    qual = _quality(project, rig, info, st, dev, report)
    report(0.95, 'Rendering the before/after preview')
    try:
        prev = _preview(project, rig, st, info, dev)
    except StudyError as e:
        prev = {'why': str(e), 'frames': [], 'synthetic': True}
    hv = qual.get('held_out_views') or {}
    psnr = hv.get('psnr_db')
    sev = _status(st, psnr)
    synth = bool(info['synthetic'])
    src = {'kind': info['kind'], 'synthetic': synth,
           'label': {'demo': 'a MuJoCo render of the project (a demonstration, synthetic)',
                     'episode': f'episode {(info.get("episode") or {}).get("episode")} of '
                                f'{(info.get("episode") or {}).get("name")}',
                     'rig': 'a saved splat rig'}[info['kind']], 'dataset': info.get('episode')}
    if info['kind'] == 'episode' and synth:
        kind = (info.get('episode') or {}).get('synthetic_kind')
        src['label'] += (' (splat renders: synthetic, not a recording)' if kind == 'splat renders'
                         else ' (a DexVantage synthetic dataset)')
    lp = hv.get('lpips')
    advice = ('Use Write replay… for a video of the episode through the proposed camera, and Write augmented… for '
              'perturbed copies of the episode (every render marked synthetic).')
    if sev in ('serious', 'critical'):
        advice = ('The rig reproduces the held-out frames poorly: more views (a moving wrist camera, more cameras), '
                  'more iterations or a measured mount (the silhouette calibration) help; renders from it are not '
                  'a reliable preview.')
    fit = info.get('fit') or {}
    text = (f'Held-out views: PSNR {psnr:.1f} dB, SSIM {hv.get("ssim", 0) or 0:.3f}'
            + (f', LPIPS {lp:.3f}' if lp is not None else ', LPIPS not computed')
            + f' ({hv.get("views", 0)} views the fit did not use; {src["label"]}).') if psnr is not None else \
        'The held-out quality was not measured.'
    head = {'value': psnr, 'unit': 'dB', 'status': sev, 'word': WORDS[sev] + (' (synthetic capture)' if synth else ''),
            'text': text, 'advice': advice, 'ssim': hv.get('ssim'), 'lpips': lp,
            'held_out_q_psnr_db': (qual.get('held_out_q') or {}).get('psnr_db')}
    notes = list(info.get('notes') or [])
    if synth:
        made = ('splat renders of an earlier fit' if (info.get('episode') or {}).get('synthetic_kind') ==
                'splat renders' else 'MuJoCo renders')
        notes.append(f'The capture is synthetic ({made}): a real recording adds pose, calibration and exposure '
                     'errors and will score lower.')
    notes.append('Every splat image is a render, never a photograph: the replay and augmentation files are marked '
                 'synthetic.')
    return {'headline': head, 'source': src, 'synthetic': True, 'synthetic_capture': synth,
            'rig': {'gaussians': rig.n, 'links': len(rig.links), 'cached': bool(info['cached']),
                    'file': os.path.basename(info['path']) if info.get('path') else None,
                    'device': fit.get('device'), 'fit_seconds': fit.get('seconds'), 'iterations': fit.get('iterations'),
                    'stopped_by': fit.get('stopped_by'), 'segmentation': (fit.get('segmentation') or {}).get('method'),
                    'deterministic': fit.get('deterministic')},
            'capture': {k: v for k, v in (info.get('capture') or {}).items() if k != 'source'},
            'quality': qual, 'preview': prev, 'notes': notes,
            'references': ['qureshi2025_splatsim', 'lou2025_robogs', 'kerbl2023_3dgs', 'wang2004_ssim',
                           'zhang2018_lpips'],
            'settings': {k: st[k] for k in FIRST + tuple(k for k in st if k not in FIRST and k != 'defaults')},
            'defaults': list(st['defaults'])}


# =========================================================================== files
RENDER_NOTE = ('synthetic: a DexVantage splat render (robot-aligned 3D Gaussian Splatting of a fitted rig), never a '
               'real image')


def splat_marked(info) -> bool:
    """True when a LeRobot info.json (a dataset's header) carries augment's marks: 'dexvantage': {'synthetic': true}
    or a video feature's 'dexvantage.synthetic' (_mark_info). Reopened in DexVantage, such a dataset is synthetic."""
    if not isinstance(info, dict):
        return False
    if isinstance(info.get('dexvantage'), dict) and info['dexvantage'].get('synthetic'):
        return True
    feats = info.get('features') if isinstance(info.get('features'), dict) else {}
    return any(isinstance(f, dict) and isinstance(f.get('info'), dict) and f['info'].get('dexvantage.synthetic')
               for f in feats.values())


def _write_png(path: str, rgb) -> None:
    """An RGB uint8 image as a PNG whose tEXt chunks say it is a splat render (Software, Comment: RENDER_NOTE),
    so a frame copied out of its folder still says so (PNG 1.2: a tEXt chunk is keyword, NUL, Latin-1 text, CRC-32
    over type and data)."""
    import struct
    import zlib

    import cv2
    ok, buf = cv2.imencode('.png', cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2BGR))
    if not ok:
        raise StudyError(f'The frame {os.path.basename(path)} could not be encoded as PNG.')
    data = buf.tobytes()
    end = 8 + 12 + struct.unpack('>I', data[8:12])[0]          # the signature, then the IHDR chunk
    text = b''
    for key, value in (('Software', 'DexVantage'), ('Comment', RENDER_NOTE)):
        body = key.encode('latin-1') + b'\0' + value.encode('latin-1')
        text += struct.pack('>I', len(body)) + b'tEXt' + body + struct.pack('>I', zlib.crc32(b'tEXt' + body))
    with open(path, 'wb') as fh:
        fh.write(data[:end] + text + data[end:])


def png_text(path: str) -> dict:
    """The tEXt chunks of a PNG file as {keyword: text} (what _write_png wrote)."""
    import struct
    out = {}
    with open(path, 'rb') as fh:
        data = fh.read()
    i = 8
    while i + 8 <= len(data):
        n, kind = struct.unpack('>I', data[i:i + 4])[0], data[i + 4:i + 8]
        if kind == b'tEXt':
            key, _, val = data[i + 8:i + 8 + n].partition(b'\0')
            out[key.decode('latin-1')] = val.decode('latin-1')
        if kind == b'IEND':
            break
        i += 12 + n
    return out


def _mark_video(path: str) -> None:
    """Copy an MP4's packets unchanged (no re-encoding) into a file whose container metadata says the video is a
    splat render (title and comment: RENDER_NOTE), and keep it only when every frame's presentation time is the same
    and every frame decodes."""
    from ..datasets import lerobot as L
    av = L._av()
    tmp = path[:-4] + '.part.mp4'
    try:
        with av.open(path) as src, av.open(tmp, 'w', options={'movflags': 'faststart'}) as dst:
            dst.metadata['title'] = 'DexVantage splat render (synthetic)'
            dst.metadata['comment'] = RENDER_NOTE
            si = src.streams.video[0]
            so = dst.add_stream_from_template(si, opaque=True)   # no codec opened: AV1 decodes by libdav1d
            for p in src.demux(si):
                if p.dts is None:                         # the demuxer's flush packet
                    continue
                p.stream = so
                dst.mux(p)
        a, ta = L.video_pts(path)
        b, tb = L.video_pts(tmp)
        same = len(a) == len(b) and np.allclose(a * float(ta), b * float(tb), rtol=0, atol=1e-9)
        if not same or not L._decodes(tmp, len(b)):
            raise StudyError(f'The video {os.path.basename(path)} changed when its synthetic mark was added; '
                             'nothing was kept.')
        os.replace(tmp, path)
    except StudyError:
        raise
    except Exception as e:                               # noqa: BLE001 — a muxer error: one sentence
        raise StudyError(f'The synthetic mark could not be added to the video {os.path.basename(path)} '
                         f'({type(e).__name__}: {e}).') from None
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _file_rows(root: str, skip=()) -> list:
    out = []
    for d, dirs, files in os.walk(root):
        dirs.sort()
        for f in sorted(files):
            p = os.path.join(d, f)
            rel = os.path.relpath(p, root).replace(os.sep, '/')
            if rel in skip:
                continue
            out.append({'path': rel, 'bytes': os.path.getsize(p), 'sha256': _sha256(p)})
    return out


def _fresh(folder: str):
    if os.path.exists(folder):
        if not os.path.isdir(folder):
            raise StudyError(f'{os.path.basename(folder)} exists and is not a folder.')
        if os.listdir(folder):
            raise StudyError(f'The folder {os.path.basename(folder)} is not empty: choose another output folder (each '
                             'replay or augmentation writes a new one).')
    try:
        os.makedirs(folder, exist_ok=True)
    except OSError as e:
        raise StudyError(f'Could not create the folder {os.path.basename(folder)}: {e.strerror or e}.') from None


def _rig_summary(rig, info) -> dict:
    f = info.get('fit') or {}
    return {'gaussians': rig.n, 'source': info['kind'], 'synthetic_capture': bool(info['synthetic']),
            'held_out_after': f.get('held_out_after'), 'fit_seconds': f.get('seconds'), 'device': f.get('device'),
            'file': os.path.basename(info['path']) if info.get('path') else None}


def _fps(ep) -> float:
    if ep.fps:
        return float(ep.fps)
    t = np.asarray(ep.t, float)
    return float(round(1.0 / np.median(np.diff(t)))) if len(t) > 1 else 30.0


def _write_video(path: str, key: str, read_frame, ts, fps: float, codec: str) -> dict:
    """Encode the frames (read_frame(i) -> HxWx3 uint8, i = 0..len(ts)-1) at their timestamps as the LeRobot writer
    does (core/datasets/lerobot: the same encoder options, AV1 re-encoded with one thread when it does not decode)
    and read it back: every frame must decode."""
    from ..datasets import lerobot as L
    from ..datasets.episode import CameraStream, DatasetError
    ts = np.asarray(ts, float)
    try:
        vcodec, label = L.available_codec(codec)
    except DatasetError as e:
        raise StudyError(str(e)) from None
    regular = L._regular(ts, fps)
    img0 = read_frame(0)
    cs = CameraStream(key, int(img0.shape[1]), int(img0.shape[0]), fps, ts,
                      _decode=lambda idx: ((i, read_frame(i)) for i in range(len(ts))))
    try:
        L._encode_episode(path, key, 0, cs, ts, fps, vcodec, 'v3.0', regular, {})
    except DatasetError as e:
        raise StudyError(str(e)) from None
    if not L._decodes(path, len(ts)):
        raise StudyError(f'The video {os.path.basename(path)} does not decode after writing; write it with the H.264 '
                         'codec instead.')
    _mark_video(path)
    return {'codec': label, 'encoder': vcodec, 'frames': int(len(ts)), 'fps': fps, 'read_back': True,
            'regular': regular}


def replay(project, out_dir: str, settings: dict | None = None, *, progress=None, cancel=None) -> dict:
    """Novel camera pose replay (module docstring): the recorded episode at its logged q(t) through the proposed
    target camera, written to out_dir/splat_replay_synthetic/: <camera>_replay_synthetic.mp4, frames/
    frame_NNNNNN_synthetic.png (each file's own metadata says synthetic too) and the manifest (SHA-256 of every file,
    provenance). Returns the manifest."""
    import cv2

    from .. import provenance as PV
    from ...workdir import slug
    st = settings_of(project, settings)
    report = progress or (lambda f, t='': None)
    stop = cancel or (lambda: False)
    kind, path = _source(project, st)
    if kind != 'episode' and not path:
        try:
            path = dataset_path(project, st)
        except StudyError:
            raise StudyError('Replay needs a recorded LeRobot episode to re-render: choose the dataset first '
                             '(Dataset… on the Splat views page, or --dataset PATH).') from None
    rig, info = get_rig(project, st, progress=lambda f, t='': report(0.4 * f, t), cancel=cancel)
    dev = _torch_device(info.get('device') or ('cpu' if st['device'] == 'cpu' else 'cuda'))
    s, im = target_sensor(project, st)
    cam = im.camera
    cam = cam.scaled(_even(cam.width * st['replay_scale']), _even(cam.height * st['replay_scale']))
    fn = target_pose_fn(project, st, s, im, True)
    E = _Episode(project, path, st)
    try:
        t = np.asarray(E.ep.t, float)
        rows = np.arange(len(t))
        if st['replay_max_frames']:
            rows = rows[: st['replay_max_frames']]
        Q = E.q_at(t[rows])
        fps = _fps(E.ep)
        dinfo = E.info
    finally:
        E.close()
    root = os.path.join(os.path.abspath(out_dir), REPLAY_FOLDER)
    _fresh(root)
    fdir = os.path.join(root, 'frames')
    os.makedirs(fdir)
    names = []
    t0 = time.perf_counter()
    try:
        for k, q in enumerate(Q):
            if stop():
                raise StudyError('The replay was cancelled; nothing was kept.')
            qq = {E.ruid: q}
            img = _u8(rig.render(project, cam, fn(qq), qq, backend='torch', device=dev)['rgb'])
            n = f'frame_{k:06d}_{SYNTHETIC_TAG}.png'
            _write_png(os.path.join(fdir, n), img)
            names.append(n)
            if k % 5 == 0 or k == len(Q) - 1:
                report(0.4 + 0.5 * (k + 1) / len(Q), f'Rendering frame {k + 1} of {len(Q)}')
        render_s = time.perf_counter() - t0
        report(0.92, 'Writing the video')
        ts = t[rows] - t[rows][0]

        def read(i):
            return cv2.cvtColor(cv2.imread(os.path.join(fdir, names[i]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        vname = f'{slug(s.name)}_replay_{SYNTHETIC_TAG}.mp4'
        video = _write_video(os.path.join(root, vname), f'observation.images.{slug(s.name)}', read, ts, fps,
                             st['codec'])
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    T0 = fn({E.ruid: Q[0]})
    doc = {'schema': SCHEMA_REPLAY, 'synthetic': True,
           'renders': 'synthetic (3D Gaussian Splatting renders of the fitted rig, never real images)',
           'source': {'dataset': dinfo, 'rows': int(len(rows)), 'fps': fps},
           'camera': {'sensor_uid': s.uid, 'sensor': s.name, 'proposal': _proposal_words(st),
                      'offset_m': st['target_offset_m'], 'rpy_deg': st['target_rpy_deg'], 'pose': st.get('target_pose'),
                      'width': cam.width, 'height': cam.height, 'lens': cam.summary(),
                      'T_world_optical_first_frame': [[round(float(x), 9) for x in r] for r in T0]},
           'video': {'file': vname, **video}, 'frames_folder': 'frames', 'frames': len(names),
           'render_seconds': round(render_s, 3), 'frames_per_s': round(len(names) / render_s, 2) if render_s else None,
           'rig': _rig_summary(rig, info),
           'conventions': {'frames': 'the OpenCV optical frame (z forward, x right, y down) in the world, the primary '
                                     'robot base (REP-103); quaternion (x, y, z, w)',
                           'q': 'the episode\'s state rows, joints matched to the model (core/datasets/joints)'}}
    doc['files'] = _file_rows(root, skip=(MANIFEST,))
    doc = PV.stamp(doc, project, study=SID)
    with open(os.path.join(root, MANIFEST), 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False, allow_nan=False)
    doc['root'] = root
    report(1.0, 'Replay written')
    return doc


def augment(project, out_dir: str, settings: dict | None = None, *, progress=None, cancel=None) -> dict:
    """Perturbation augmentation (module docstring): augment_n copies of the episode, each camera the rig records
    re-rendered through its mount perturbed by one draw of §8.5's tolerances (core/tolerance.perturbed_extrinsics),
    written as a LeRobot dataset in out_dir/splat_augment_synthetic/ with every render marked synthetic: the info.json
    'dexvantage' block and each video feature's info, each MP4's container title and comment, and the manifest (draws,
    SHA-256 of every file, provenance). Returns the manifest."""
    from .. import provenance as PV
    from .. import tolerance as TL
    from ..datasets import lerobot as L
    from ..datasets.episode import CameraStream, DatasetError, Episode
    from .episodes import intrinsics
    st = settings_of(project, settings)
    report = progress or (lambda f, t='': None)
    stop = cancel or (lambda: False)
    try:
        path = dataset_path(project, st)
    except StudyError:
        raise StudyError('Augmentation needs a recorded LeRobot episode to re-render: choose the dataset first '
                         '(Dataset… on the Splat views page, or --dataset PATH).') from None
    rig, info = get_rig(project, st, progress=lambda f, t='': report(0.3 * f, t), cancel=cancel)
    dev = _torch_device(info.get('device') or ('cpu' if st['device'] == 'cpu' else 'cuda'))
    E = _Episode(project, path, st)
    try:
        if not E.matched:
            raise StudyError('No camera of the rig is in the recording, so there is nothing to re-render.')
        A = TL.perturbed_extrinsics(project, st['augment_n'], st['augment_seed'], st['augment_scale'],
                                    st['augment_distribution'])
        col = {u['uid']: j for j, u in enumerate(A['sensors'])}
        ep = E.ep
        t0 = float(np.asarray(ep.t, float)[0])
        fps = _fps(ep)
        cams = []
        for s, cs in E.matched:
            if s.uid not in col:
                continue
            from .splat_rig import stream_imager
            im = stream_imager(s)
            W0 = int(cs.width or im.camera.width)
            H0 = int(cs.height or im.camera.height)
            cam = im.camera.scaled(_even(W0 * st['augment_frame_scale']), _even(H0 * st['augment_frame_scale']))
            ts = np.asarray(cs.timestamps, float) - t0
            cams.append((s, cs, im, cam, np.maximum(ts, 0.0), E.q_at(np.asarray(cs.timestamps, float))))
        if not cams:
            raise StudyError('None of the recorded cameras has a mount tolerance to perturb.')
        root = os.path.join(os.path.abspath(out_dir), AUGMENT_FOLDER)
        _fresh(root)
        from ..scenegeom import sensor_world_pose
        total = st['augment_n'] * sum(len(c[4]) for c in cams)
        done = [0]

        def stream(i, s, cs, im, cam, ts, Q):
            D = A['delta_body'][i, col[s.uid]]
            T_bo = im.pose_in_body.matrix()

            def decode(indices):
                for k in (range(len(ts)) if indices is None else indices):
                    if stop():
                        raise StudyError('The augmentation was cancelled; nothing was kept.')
                    q = {E.ruid: Q[k]}
                    T = sensor_world_pose(project, s, q).matrix() @ D @ T_bo
                    img = _u8(rig.render(project, cam, T, q, backend='torch', device=dev)['rgb'])
                    done[0] += 1
                    if done[0] % 10 == 0:
                        report(0.3 + 0.65 * done[0] / max(total, 1), f'Rendering {done[0]:,} of {total:,} frames')
                    yield k, img
            Tpo = A['T_parent_optical'][i, col[s.uid]]
            ex = None
            if np.isfinite(Tpo).all():
                from ..frames import quat_from_matrix
                ex = {'frame': s.mount.parent_frame if s.mount.on_link else 'world',
                      'pose': {'position': [float(x) for x in Tpo[:3, 3]],
                               'quaternion': [float(x) for x in quat_from_matrix(Tpo[:3, :3])]}}
            return CameraStream(cs.key, cam.width, cam.height, cs.fps or fps, ts, None, intrinsics(cam), ex,
                                {'synthetic': True, 'splat_render': True, 'sensor_uid': s.uid, 'perturbation': i},
                                decode)
        eps = []
        for i in range(st['augment_n']):
            eps.append(Episode(f'episode_{i:06d}', i, np.asarray(ep.t, float) - t0, np.asarray(ep.q, float),
                               list(ep.joint_names), ep.action, list(ep.action_names), fps, ep.robot_type, ep.task,
                               {c[1].key: stream(i, *c) for c in cams}, {'synthetic': True}))
        try:
            summary = L.write_lerobot(root, eps, st['augment_version'], round(fps), ep.robot_type or None,
                                      codec=st['codec'], overwrite=False)
        except DatasetError as e:
            shutil.rmtree(root, ignore_errors=True)
            raise StudyError(str(e)) from None
        except BaseException:
            shutil.rmtree(root, ignore_errors=True)
            raise
        dinfo = E.info
    finally:
        E.close()
    try:
        _mark_info(root, info, dinfo)
        for d, _dirs, files in os.walk(os.path.join(root, 'videos')):
            for f in sorted(files):
                if f.endswith('.mp4'):
                    _mark_video(os.path.join(d, f))
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    draws = [[{'sensor_uid': c[0].uid, 'sensor': c[0].name, 'camera': c[1].key,
               'd': [round(float(x), 9) for x in A['d'][i, col[c[0].uid]]]} for c in cams]
             for i in range(st['augment_n'])]
    doc = {'schema': SCHEMA_AUGMENT, 'synthetic': True,
           'renders': 'synthetic (3D Gaussian Splatting renders of the fitted rig through perturbed extrinsics, never '
                      'real images)',
           'source': dinfo, 'episodes': st['augment_n'], 'cameras': [c[1].key for c in cams],
           'lerobot': {k: summary.get(k) for k in ('version', 'codec', 'fps', 'frames')},
           'perturbations': {'n': A['n'], 'seed': A['seed'], 'scale': A['scale'], 'distribution': A['distribution'],
                             'sources': A['sources'], 'components': A['components'],
                             'component_units': A['component_units'], 'draws': draws,
                             'convention': 'episode i is draw i; d as core/exporters/perturbations.py writes it'},
           'rig': _rig_summary(rig, info)}
    doc['files'] = _file_rows(root, skip=(MANIFEST,))
    doc = PV.stamp(doc, project, study=SID)
    with open(os.path.join(root, MANIFEST), 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False, allow_nan=False)
    doc['root'] = root
    report(1.0, 'Augmented dataset written')
    return doc


def _mark_info(root: str, info: dict, dinfo: dict):
    """meta/info.json: a 'dexvantage' block saying every video is a splat render, and the same in each video
    feature's info (LeRobot's DatasetInfo warns about keys it does not know and leaves them out; the file keeps
    them)."""
    from ..datasets import lerobot as L
    p = os.path.join(root, L.INFO_PATH)
    with open(p, encoding='utf-8') as fh:
        meta = json.load(fh)
    meta['dexvantage'] = {'synthetic': True, 'kind': 'splat render (robot-aligned 3D Gaussian Splatting)',
                          'source_dataset': dinfo.get('name'), 'source_episode': dinfo.get('episode'),
                          'capture_synthetic': bool(info.get('synthetic')), 'manifest': MANIFEST}
    for ft in (meta.get('features') or {}).values():
        if isinstance(ft, dict) and ft.get('dtype') == 'video':
            ft.setdefault('info', {})['dexvantage.synthetic'] = True
    tmp = p + '.part'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=4, ensure_ascii=False)
    os.replace(tmp, p)
