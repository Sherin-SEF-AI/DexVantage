"""Test recordings with KNOWN sim-to-real gaps: the synthetic episode generator (core/sim/episodes.py) renders the
project's robot through its cameras, and the gaps are injected on the way to the LeRobot files, so the sim-to-real
gap study (core/sim2real/gap.py) can be checked against what was put in.

    GAPS                                       the gap kinds and what each does
    make_gap_dataset(project, folder, gaps, settings=None, *, progress=None) -> dict
        the summary of core/sim/episodes.generate plus 'gaps' (what was injected, per camera key); the truth sidecar
        (dexvantage_truth.json) carries the same under 'sim2real_gaps'.

gaps (every one optional; 'camera' a dataset camera key or a sensor uid, default every camera for the photometric
gaps, the first static camera for the mount, the first camera for the latency):
- brightness_dn: an offset added to every pixel (DN), after
- gain: a factor on every pixel value (an exposure difference of log₂ gain stops), after
- blur_px: a Gaussian blur of σ pixels of the stored image (a focus blur);
- noise_dn: extra Gaussian pixel noise (σ, DN, seeded) on top of the generator's own noise_dn;
- mount: {'position_m': [x, y, z], 'rotation_deg': [rx, ry, rz]}: the generator's mount_slip fault from episode 0
  (the camera renders through T_body_sensor · ΔT, ΔT in the sensor's body frame, a rotation vector in degrees);
- latency_s: the generator's desync fault (each frame shows the robot latency_s later than its stamp).
Order on each frame: rendered (with the mount and latency), blur, gain, brightness, noise, rounded and clipped to
8 bits; then the LeRobot writer's video codec (as for any synthetic episode). Every other fault of the generator is
off unless settings name it.

Known limits: the photometric gaps are applied to the rendered image, not to the light (a gain also scales shadows,
a real exposure change also changes motion blur); blur is a stationary Gaussian; the codec removes part of the
injected noise and adds its own blur, so the recovered figures include the codec's share (docs/models/sim2real.md).
"""
from __future__ import annotations

import copy
import os
import shutil
import tempfile

import numpy as np

from ..jobs import StudyError

GAPS = ('brightness_dn', 'gain', 'blur_px', 'noise_dn', 'mount', 'latency_s', 'camera')


def _checked_gaps(gaps) -> dict:
    """The gaps with every figure a finite number in its range, or StudyError saying which one is not."""
    import math
    if gaps is not None and not isinstance(gaps, dict):
        raise StudyError('The gaps are a dict, e.g. {"brightness_dn": 20, "blur_px": 1.5}.')
    gaps = dict(gaps or {})
    bad = [k for k in gaps if k not in GAPS]
    if bad:
        raise StudyError(f'There is no gap called {bad[0]}; the gaps are {", ".join(GAPS)}.')

    def num(v, what, lo=None):
        try:
            x = float(v)
        except (TypeError, ValueError):
            x = math.nan
        if isinstance(v, bool) or not math.isfinite(x) or (lo is not None and x < lo):
            raise StudyError(f'The gap {what} must be a number' + (f' of at least {lo:g}' if lo is not None else '') +
                             f'; it is {v!r}.')
        return x
    for k, lo in (('brightness_dn', None), ('gain', 1e-6), ('blur_px', 0.0), ('noise_dn', 0.0), ('latency_s', None)):
        if gaps.get(k) is not None:
            gaps[k] = num(gaps[k], k, lo)
    m = gaps.get('mount')
    if m is not None:
        if not isinstance(m, dict) or set(m) - {'position_m', 'rotation_deg'}:
            raise StudyError('The gap mount is {"position_m": [x, y, z], "rotation_deg": [rx, ry, rz]}.')
        for k in ('position_m', 'rotation_deg'):
            v = m.get(k, [0, 0, 0])
            if not isinstance(v, (list, tuple)) or len(v) != 3:
                raise StudyError(f'The gap mount.{k} is three numbers; it is {v!r}.')
            m = dict(m, **{k: [num(x, f'mount.{k}') for x in v]})
        gaps['mount'] = m
    return gaps


def _cam_key(cams: list, sensors: dict, ident):
    """The dataset key of a camera given by key or sensor uid (None: None)."""
    if ident is None:
        return None
    for c in cams:
        if c['key'] == ident or c['sensor_uid'] == ident:
            return c['key']
    raise StudyError(f'There is no camera {ident} in the synthetic dataset; it has '
                     f'{", ".join(c["key"] for c in cams)}.')


def make_gap_dataset(project, folder: str, gaps: dict | None = None, settings: dict | None = None, *,
                     progress=None) -> dict:
    """Render a synthetic LeRobot dataset with the gaps injected (module docstring) into folder (new or empty)."""
    import cv2

    from ..datasets import lerobot as L
    from ..datasets.episode import CameraStream, Episode
    from ..sim import episodes as SE
    gaps = _checked_gaps(gaps)
    root = os.path.abspath(os.fspath(folder))          # refused before the renders, not after them
    if os.path.exists(root) and (not os.path.isdir(root) or os.listdir(root)):
        raise StudyError(f'{os.path.basename(root)} is not an empty folder: choose a new or empty folder for the '
                         'recording.')
    st = dict(settings or {})
    faults = {k: False for k in SE.FAULT_KINDS}
    faults.update(st.pop('faults', None) or {})
    # which cameras: plan the generator's cameras first (no rendering)
    plan = SE.plan(project, dict(st, faults=faults))
    cams = [{'key': c.key, 'sensor_uid': c.sensor_uid, 'on_link': c.on_link} for c in plan.cameras]
    cam = gaps.get('camera')
    phot_keys = [_cam_key(cams, {}, cam)] if cam is not None else [c['key'] for c in cams]
    injected = {}
    if gaps.get('mount') is not None:
        mk = _cam_key(cams, {}, cam) if cam is not None else \
            next((c['key'] for c in cams if not c['on_link']), cams[0]['key'])
        m = gaps['mount']
        faults['mount_slip'] = {'from_episode': 0, 'camera': mk,
                                'position_m': [float(x) for x in m.get('position_m', [0, 0, 0])],
                                'rotation_deg': [float(x) for x in m.get('rotation_deg', [0, 0, 0])]}
        injected.setdefault(mk, {})['mount'] = copy.deepcopy(faults['mount_slip'])
    if gaps.get('latency_s'):
        lk = _cam_key(cams, {}, cam) if cam is not None else cams[0]['key']
        faults['desync'] = {'offset_s': float(gaps['latency_s']), 'camera': lk}
        injected.setdefault(lk, {})['latency_s'] = float(gaps['latency_s'])
    phot = {k: float(gaps[k]) for k in ('brightness_dn', 'gain', 'blur_px', 'noise_dn') if gaps.get(k) is not None}
    for k in phot_keys:
        if phot:
            injected.setdefault(k, {}).update(phot)
    tmp = tempfile.mkdtemp(prefix='dexvantage-gaps-')
    try:
        s = SE.generate(project, os.path.join(tmp, 'clean'), dict(st, faults=faults), keep_frames=True,
                        progress=progress)
        truth = copy.deepcopy(s['truth'])
        gen = truth['generator']
        names = list(gen['robot']['joint_names'])
        seed = int(gen['seed'])

        def treat(img, key, e, i):
            if key not in phot_keys or not phot:
                return img
            a = np.asarray(img, np.float32)
            if phot.get('blur_px'):
                a = cv2.GaussianBlur(a, (0, 0), phot['blur_px'], borderType=cv2.BORDER_REPLICATE)
            a = a * phot.get('gain', 1.0) + phot.get('brightness_dn', 0.0)
            if phot.get('noise_dn'):
                a = a + np.random.default_rng([seed, 509, e, i, sum(map(ord, key))]).normal(
                    0.0, phot['noise_dn'], a.shape)
            return np.clip(np.rint(a), 0, 255).astype(np.uint8)
        eps = []
        for e, arr in enumerate(s['arrays']):
            et = truth['episodes'][str(e)]
            cs = {}
            for key, c in arr['cameras'].items():
                frames = [treat(f, key, e, i) for i, f in enumerate(c['frames'])]

                def dec(idx, frames=frames):
                    for i in (range(len(frames)) if idx is None else idx):
                        yield int(i), frames[int(i)]
                tc = truth['cameras'][key]
                h, w = frames[0].shape[:2]
                cs[key] = CameraStream(key, w, h, float(tc['fps']), np.asarray(c['timestamps'], float), None,
                                       tc['intrinsics'], et['extrinsics'][key],
                                       {'synthetic': True, 'sensor_uid': tc['sensor_uid']}, dec)
            eps.append(Episode(f'episode_{e:06d}', e, np.asarray(arr['t'], float), np.asarray(arr['q'], float),
                               names, np.asarray(arr['action'], float), names, float(gen['fps']),
                               gen['robot']['name'], f'{et["motion"]["name"]} (synthetic, gaps injected)', cs,
                               {'synthetic': True}))
        truth['sim2real_gaps'] = {'injected': injected, 'order': 'rendered (mount, latency), blur, gain, brightness, '
                                                                 'noise, 8 bits, then the video codec'}
        out = L.write_lerobot(folder, eps, s['version'], s['fps'], gen['robot']['name'],
                              codec=st.get('codec') or 'auto', truth=truth)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {'root': out['root'], 'version': s['version'], 'episodes': s['episodes'], 'cameras': s['cameras'],
            'gaps': injected, 'renderer': s['renderer'], 'truth_file': L.TRUTH_FILE}
