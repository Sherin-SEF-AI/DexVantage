"""The campaign checks of the dataset auditor (spec §8.2): every episode audited (core/audit/episode.py), then what
only the whole campaign shows, mount slip above all, and the campaign's report: pass or fail per requirement, the
episodes to drop or re-record.

    audit_dataset(path_or_dataset, project, req=None, settings=None, *, progress=None, cancel=None) -> dict

Mount slip: one estimate of each camera's mount per episode, then the series over the episodes (in recording order,
on a time axis of days when the dataset records when each episode was taken, else the episode's place) explained by
steps against gradual drift by the model selection ported from Vantage core/online_calib.py (core/audit/slip.py).
Three sources (settings 'mount_source': 'auto' (default: a board when one is found, else registration), 'board',
'silhouette' (the link-silhouette fit, else registration: it costs MuJoCo renders, about 3 s per episode and camera on
the SO-100 sample, so it runs only when chosen) or 'registration'):

1. Board re-checks: when the project has a calibration board (project.calibration boards) and it is found in an
   episode's frames (core/calib/detect.detect_and_solve on up to BOARD_FRAMES frames spread over the episode), the
   camera's optical frame follows from the board's placement: a static camera T_world_cam = T_world_board ·
   T_cam_board⁻¹; a camera on a link T_link_cam = F_link(q)⁻¹ · T_world_board · T_cam_board⁻¹, the board's pose
   from its record (a board on a link is posed at q). The episode's estimate is the chordal mean of the frames'.
2. The robot's own links (§8.4): for a static camera, core/calib/render_calib.estimate (the targetless link-silhouette
   fit against MuJoCo renders) gives the mount correction ΔT against the plan with its σ; called only through that
   API, and skipped with a note when it is not installed or refuses (too small or too static a robot).
3. Image registration (when neither applies, and for cameras on a link): the episode's static image, the per-pixel
   median of up to REG_FRAMES frames, over the pixels that stay put within the episode (temporal standard deviation
   under STABLE_DN; for a static camera the background, for a wrist camera the gripper parts fixed in its view), is
   registered to the first episode's by a pure rotation of the camera: each reference pixel's ray r (the plan's lens
   model, any projection) is looked up in the episode's image at project(Aᵀ r), A = Exp(ω) the rotation of the
   optical frame, with a gain and an offset for exposure; ω by Gauss-Newton (numerical Jacobian, Huber weights with
   a scale of at least REG_NOISE_DN; Lucas and Kanade 1981; Baker and Matthews 2004) on a three-level pyramid, over
   the static pixels whose grey gradient exceeds REG_MIN_GRAD (the others carry no information). σ is the spread over a
   block bootstrap of the image (REG_BOOT resamples of 16 × 16 px blocks). It is relative to the first episode and
   measures rotation only (a translation of millimetres moves a background a metre away by under a pixel). The rotation
   is reported in the sensor's body frame, ΔR = R_so A R_soᵀ, as a rotation vector in degrees (the convention of
   render_calib's 'delta' and of the synthetic episodes' mount_slip fault).

Each axis (body x, y, z rotation, and position when the source gives it) goes through slip.classify: a step (the
mount moved and stayed: a slip; the episodes after it need the new extrinsics), an excursion (it moved and came back),
a gradual drift (a slope per day or per episode), or steady. A shift beyond max_mount_shift_deg / mm (defaults
0.5° and 2 mm) from the first estimate fails the requirement for that episode.

Campaign summary: per requirement the number of episodes (and cameras) that pass; each episode's verdict
(pass, fix: only a camera offset or drift fails, which re-stamping corrects; drop: drop or re-record it); the camera
offsets over the campaign (a per-episode offset that changes from episode to episode is a clock that is not
re-synchronised).

Known limits: registration needs a static background (or fixed gripper parts) with texture, measures rotation only and
relative to the first episode; a slip within an episode is seen as an episode between two levels (its estimate is
the median); the silhouette fit is for static cameras only and costs MuJoCo renders per episode; a board must be
placed as its record says; episodes are taken in index order unless the dataset records their times.
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..jobs import StudyError
from . import episode as AE
from . import slip as SL

SETTINGS = {
    'episodes': None,              # indices to audit (None: all)
    'max_episodes': None,          # at most this many (None: all)
    'mount_source': 'auto',        # 'auto' | 'board' | 'silhouette' | 'registration' | 'none'
    'per_frame': True,             # keep the per-frame scores (for the FiftyOne export and the page)
    'min_step_deg': 0.1,           # default: mount steps smaller than this are not reported
    'min_step_mm': 0.5,            # default: … in position
}
SOURCES = ('auto', 'board', 'silhouette', 'registration', 'none')
BOARD_FRAMES = 8                   # frames per episode searched for a board
REG_FRAMES = 60                    # frames per episode in the median image
REG_WIDTH = 320                    # registration at most this wide
STABLE_DN = 6.0                    # a pixel is static when its temporal standard deviation is under this (grey levels)
REG_BOOT = 20                      # bootstrap resamples for the registration's σ
REG_BLOCK = 16                     # px
REG_LEVELS = (0.25, 0.5, 1.0)      # pyramid scales, coarse to fine
REG_MIN_GRAD = 1.0                 # only pixels with a grey gradient over this (per px) are used
REG_NOISE_DN = 2.0                 # the Huber scale never falls below this (grey levels)


def settings_of(project=None, given: dict | None = None) -> dict:
    stored = {}
    if project is not None:
        stored = dict((project.analysis.studies.get('audit') or {}))
    out = dict(SETTINGS)
    for k, v in dict(stored, **(given or {})).items():
        if k in SETTINGS and v is not None:
            out[k] = v
    if out['mount_source'] not in SOURCES:
        raise StudyError(f'The mount source must be one of {", ".join(SOURCES)}; it is {out["mount_source"]!r}.')
    return out


# ====================================================================== mount estimates
def _rotvec_R(w) -> np.ndarray:
    w = np.asarray(w, float)
    a = float(np.linalg.norm(w))
    if a < 1e-15:
        return np.eye(3)
    k = w / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def _R_rotvec(R) -> np.ndarray:
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    a = math.acos(c)
    if a < 1e-12:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if math.sin(a) < 1e-9:                                   # near 180°: the axis from the symmetric part
        w, V = np.linalg.eigh((R + R.T) / 2)
        return V[:, -1] * a
    return v / (2 * math.sin(a)) * a


def static_image(stream, frames_n: int, width: int, exclude=()) -> tuple[np.ndarray, np.ndarray] | None:
    """(median grey image, static mask) over up to frames_n frames spread over the episode."""
    M = stream.n_frames
    if M < 3:
        return None
    idx = sorted(set(np.linspace(0, M - 1, min(frames_n, M)).round().astype(int).tolist()) - set(exclude))
    g = [AE._grey(np.asarray(im), width) for _i, im in stream.frames(idx)]
    if len(g) < 3:
        return None
    G = np.stack(g).astype(np.float32)
    return np.median(G, axis=0), G.std(axis=0) < STABLE_DN


class Registration:
    """A camera's static image in one episode registered to its reference (module docstring, 3)."""

    def __init__(self, cam, ref: np.ndarray, ref_mask: np.ndarray):
        import cv2
        self.cv2 = cv2
        h, w = ref.shape
        self.cam = cam.scaled(w, h) if (cam.width, cam.height) != (w, h) else cam
        self.ref = ref
        self.mask = ref_mask
        self.levels = []
        for s in REG_LEVELS:
            cm = self.cam.scaled(max(8, int(round(w * s))), max(8, int(round(h * s)))) if s != 1.0 else self.cam
            W, H = cm.width, cm.height
            r = cv2.resize(ref, (W, H), interpolation=cv2.INTER_AREA) if s != 1.0 else ref
            m = cv2.resize(ref_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool) \
                if s != 1.0 else ref_mask
            gx = cv2.Sobel(r.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3) / 8.0
            gy = cv2.Sobel(r.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3) / 8.0
            m = m & (np.hypot(gx, gy) > REG_MIN_GRAD)
            st = 1 if s < 1.0 else 2
            vv, uu = np.mgrid[0:H:st, 0:W:st]
            keep = m[vv, uu]
            vv, uu = vv[keep], uu[keep]
            border = (uu > 2) & (uu < W - 3) & (vv > 2) & (vv < H - 3)
            uu, vv = uu[border], vv[border]
            rays, ok = cm.unproject(np.stack([uu, vv], axis=1).astype(float), with_ok=True)
            self.levels.append({'cam': cm, 'ref': r.astype(np.float32), 'u': uu[ok], 'v': vv[ok], 'rays': rays[ok]})

    def _sample(self, img, L, w):
        A = _rotvec_R(w)
        uv, ok = L['cam'].project(L['rays'] @ A)              # rows: Aᵀ r
        uv = np.nan_to_num(uv, nan=-10.0).astype(np.float32)
        vals = self.cv2.remap(img, uv[:, 0].reshape(-1, 1), uv[:, 1].reshape(-1, 1), self.cv2.INTER_LINEAR,
                              borderMode=self.cv2.BORDER_CONSTANT, borderValue=-1.0).reshape(-1)
        return vals, ok & (vals >= 0)

    def _solve(self, img_levels, w0, idx_sel=None, iters=(8, 8, 8)):
        w = np.array(w0, float)
        last = None
        for L, img, n_it in zip(self.levels, img_levels, iters):
            ref = L['ref'][L['v'], L['u']]
            sel = idx_sel if (idx_sel is not None and L is self.levels[-1]) else np.arange(len(ref))
            h = math.radians(0.05)
            for _ in range(n_it):
                f0, ok0 = self._sample(img, L, w)
                J = []
                oks = ok0.copy()
                for a in range(3):
                    e = np.zeros(3)
                    e[a] = h
                    fp, okp = self._sample(img, L, w + e)
                    fm, okm = self._sample(img, L, w - e)
                    J.append((fp - fm) / (2 * h))
                    oks &= okp & okm
                J = np.stack(J, axis=1)
                cnt = np.bincount(sel, minlength=len(ref))
                m = (cnt > 0) & oks
                if m.sum() < 30:
                    return None
                # photometric gain and offset: ref ≈ a f + b, solved jointly (linearised)
                X = np.column_stack([J[m], f0[m], np.ones(m.sum())])
                y = ref[m]
                rep = cnt[m].astype(float)
                wt = rep.copy()
                for _k in range(3):
                    sw = np.sqrt(wt)
                    beta, *_ = np.linalg.lstsq(X * sw[:, None], (y - 0.0) * sw, rcond=None)
                    a_, b_ = beta[3], beta[4]
                    r = y - (a_ * f0[m] + b_)
                    s = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), REG_NOISE_DN)
                    wt = rep * AE_huber(np.abs(r) / s)
                # the step: ref − (a f0 + b) ≈ a J dw
                Xs = J[m] * a_
                sw = np.sqrt(wt)
                dw, *_ = np.linalg.lstsq(Xs * sw[:, None], r * sw, rcond=None)
                w = w + dw
                last = (Xs, r, wt)
                if np.linalg.norm(dw) < math.radians(0.002):
                    break
        return w, last

    def solve(self, img: np.ndarray, mask: np.ndarray, rng) -> dict | None:
        cv2 = self.cv2
        h, w = self.ref.shape
        if img.shape != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        imgs = [cv2.resize(img, (L['cam'].width, L['cam'].height), interpolation=cv2.INTER_AREA).astype(np.float32)
                if (L['cam'].width, L['cam'].height) != (w, h) else img.astype(np.float32) for L in self.levels]
        res = self._solve(imgs, np.zeros(3))
        if res is None:
            return None
        w_opt, _last = res
        # block bootstrap of the finest level
        L = self.levels[-1]
        blk = (L['v'] // REG_BLOCK) * 10000 + (L['u'] // REG_BLOCK)
        ub = np.unique(blk)
        boot = []
        for _b in range(REG_BOOT):
            pick = rng.choice(ub, len(ub), replace=True)
            counts = {}
            for p in pick:
                counts[p] = counts.get(p, 0) + 1
            sel = np.concatenate([np.repeat(np.flatnonzero(blk == p), c) for p, c in counts.items()])
            r2 = self._solve(imgs, w_opt, sel, iters=(0,) * (len(self.levels) - 1) + (3,))
            if r2 is not None:
                boot.append(r2[0])
        sig = np.std(np.stack(boot), axis=0, ddof=1) if len(boot) > 2 else np.full(3, np.nan)
        return {'omega_opt': w_opt, 'sigma_opt': sig, 'pixels': int(len(L['u']))}


def AE_huber(a, k=1.345):
    return np.where(a <= k, 1.0, k / np.maximum(a, 1e-12))


def _board_estimates(project, ep, key, plan_cam) -> dict | None:
    """The camera's optical frame (in its link or the world) from a calibration board in its frames, or None."""
    from ..calib import detect as DT
    from ..calib import records as RC
    recs = (getattr(project, 'calibration', None) or {}).get('boards') or []
    if not recs or plan_cam is None:
        return None
    cs = ep.cameras[key]
    M = cs.n_frames
    if M == 0:
        return None
    idx = np.linspace(0, M - 1, min(BOARD_FRAMES, M)).round().astype(int).tolist()
    ts = np.asarray(cs.timestamps, float)
    Ts = []
    for rec in recs:
        try:
            board = RC.board_of(rec)
            robot_uid, link, Z = RC.board_mount(rec)
        except Exception:                                   # noqa: BLE001 — a board record that cannot be used
            continue
        for i, img in cs.frames(idx):
            _det, pnp = DT.detect_and_solve(np.asarray(img), board, plan_cam.cam)
            if pnp is None:
                continue
            q = AE._interp_q(ep, np.array([ts[i]]))
            T_wb = Z
            if robot_uid:
                F = plan_cam.ctx.model.fk_batch(plan_cam.ctx.q_model(q))[0]
                T_wb = F[plan_cam.ctx.model.link_index(link)] @ Z
            T_wc = T_wb @ np.linalg.inv(pnp.T_cam_board)
            if plan_cam.on_link:
                F = plan_cam.ctx.model.fk_batch(plan_cam.ctx.q_model(q))[0]
                T_wc = np.linalg.inv(F[plan_cam.link_i]) @ T_wc
            Ts.append(T_wc)
        if Ts:
            break
    if not Ts:
        return None
    from ..calib.axxb import mean_transform
    T = mean_transform(np.stack(Ts))
    # as a correction of the mount: T_po' = T_po · D_opt, D in the body: R_so D_opt R_soᵀ
    T_po = plan_cam.T_po if plan_cam.on_link else plan_cam.T_static
    D_opt = np.linalg.inv(T_po) @ T
    R_so = plan_cam.imager.pose_in_body.matrix()[:3, :3]
    rot_body = R_so @ _R_rotvec(D_opt[:3, :3])
    pos_body = R_so @ D_opt[:3, 3]
    spread = [np.degrees(np.linalg.norm(_R_rotvec(T[:3, :3].T @ x[:3, :3]))) for x in Ts]
    return {'source': 'board', 'rotation_deg': np.degrees(rot_body).tolist(), 'position_mm': (pos_body * 1000).tolist(),
            'sigma_deg': [float(np.std(spread) / max(1, math.sqrt(len(Ts))) + 0.01)] * 3,
            'sigma_mm': [None] * 3, 'frames': len(Ts), 'absolute': True}


def _silhouette(ds, index, project, uid) -> dict | None:
    try:
        from ..calib import render_calib as RCAL
    except ImportError:
        return None
    if not hasattr(RCAL, 'estimate'):
        return None
    doc = RCAL.estimate(ds.load(index), project, uid)
    if not doc.get('observable', True) and doc.get('why'):
        return {'source': 'silhouette', 'error': str(doc.get('why'))}
    d, s = doc.get('delta') or {}, doc.get('sigma') or {}
    return {'source': 'silhouette', 'rotation_deg': [float(x) for x in d.get('rotation_deg', [np.nan] * 3)],
            'position_mm': [float(x) * 1000 for x in d.get('position_m', [np.nan] * 3)],
            'sigma_deg': [float(x) if x is not None else None for x in s.get('rotation_deg', [None] * 3)],
            'sigma_mm': [float(x) if x is not None else None for x in s.get('position_mm', [None] * 3)],
            'frames': len(doc.get('frames_used') or []), 'absolute': True}


def mount_series(ds, project, plan, cam_map: dict, episodes: list, st: dict, req: dict, *, times=None,
                 progress=None, cancel=None, exclude: dict | None = None) -> dict:
    """{camera key: {'sensor', 'source', 'episodes': [{'episode', 'rotation_deg', 'sigma_deg', ...}], 'axes':
    {axis: slip.classify result}, 'events', 'shift_deg', 'shift_mm'}} (module docstring)."""
    out = {}
    rng = np.random.default_rng(20261107)
    src = st['mount_source']
    if src == 'none' or not episodes:
        return out
    for key, uid in cam_map.items():
        if cancel is not None and cancel():
            break
        pc = None
        if plan is not None and uid:
            try:
                ep0 = ds.load(episodes[0])
                cs0 = ep0.cameras.get(key)
                if cs0 is None:
                    continue
                w0 = cs0.width or REG_WIDTH
                h0 = cs0.height or REG_WIDTH
                pc = plan.camera(key, int(w0), int(h0), uid)
            except Exception:                               # noqa: BLE001
                pc = None
        series, notes, used = [], [], None
        reg, ref_set = None, False
        for n, e in enumerate(episodes):
            if cancel is not None and cancel():
                break
            if progress:
                progress(n / max(1, len(episodes)), f'mount of {key}, episode {e}')
            ep = ds.load(e)
            cs = ep.cameras.get(key)
            if cs is None:
                continue
            est = None
            if src in ('auto', 'board'):
                try:
                    est = _board_estimates(project, ep, key, pc)
                except Exception as err:                    # noqa: BLE001 — a board check that fails: said
                    notes.append(f'episode {e}: the board re-check failed ({err}).')
            if est is None and src == 'silhouette' and pc is not None and not pc.on_link:
                try:
                    est = _silhouette(ds, e, project, uid)
                except Exception as err:                    # noqa: BLE001 — render_calib refused: say why
                    notes.append(f'episode {e}: the link-silhouette fit was not used ({str(err).splitlines()[0]}).')
                    est = None
                if est is not None and est.get('error'):
                    notes.append(f'episode {e}: the link-silhouette fit says {est["error"]}')
                    est = None
            if est is None and src in ('auto', 'registration', 'silhouette'):
                cam = pc.cam if pc is not None else _fallback_cam(cs)
                if cam is None:
                    notes.append('No lens model for this camera (no plan camera matches it and the dataset stores '
                                 'no intrinsics), so its mount is not registered.')
                    break
                si = static_image(cs, REG_FRAMES, REG_WIDTH, (exclude or {}).get((e, key), ()))
                if si is None:
                    continue
                img, mask = si
                if not ref_set:
                    reg = Registration(cam, img, mask)
                    ref_set = True
                    est = {'source': 'registration', 'rotation_deg': [0.0, 0.0, 0.0], 'sigma_deg': [0.0, 0.0, 0.0],
                           'position_mm': [None] * 3, 'sigma_mm': [None] * 3, 'reference': True,
                           'pixels': int(mask.sum()), 'absolute': False}
                else:
                    r = reg.solve(img, mask, rng)
                    if r is None:
                        notes.append(f'episode {e}: too few static pixels to register.')
                        continue
                    R_so = (pc.imager.pose_in_body.matrix()[:3, :3] if pc is not None else np.eye(3))
                    est = {'source': 'registration', 'rotation_deg': np.degrees(R_so @ r['omega_opt']).tolist(),
                           'sigma_deg': np.degrees(np.abs(R_so) @ r['sigma_opt']).tolist(),
                           'position_mm': [None] * 3, 'sigma_mm': [None] * 3, 'pixels': r['pixels'],
                           'absolute': False}
            if est is None:
                continue
            used = used or est['source']
            est['episode'] = int(e)
            est['t'] = float(times[n]) if times is not None else float(n)
            series.append(est)
        doc = {'sensor_uid': uid, 'sensor': pc.name if pc is not None else None, 'source': used,
               'episodes': series, 'notes': notes, 'axes': {}, 'events': []}
        if len(series) >= 2:
            _classify_series(doc, st, req)
        out[key] = doc
    return out


def _fallback_cam(cs):
    """The stream's own intrinsics when the dataset stores them, else None (registration then needs the plan)."""
    from ..optics import CameraModel
    if not (cs.intrinsics and cs.intrinsics.get('K') and cs.width and cs.height):
        return None
    K = np.asarray(cs.intrinsics['K'], float)
    return CameraModel(int(cs.width), int(cs.height), K[0, 0], K[1, 1], K[0, 2], K[1, 2])


AXIS_NAMES = ('roll (body x)', 'pitch (body y)', 'yaw (body z)')


def _classify_series(doc: dict, st: dict, req: dict):
    S = doc['episodes']
    t = np.array([s['t'] for s in S], float)
    first = S[0]
    for kind, unit, mstep in (('rotation_deg', '°', st['min_step_deg']), ('position_mm', 'mm', st['min_step_mm'])):
        sk = 'sigma_deg' if kind == 'rotation_deg' else 'sigma_mm'
        for a in range(3):
            y = np.array([np.nan if s[kind][a] is None else s[kind][a] for s in S], float)
            sg = np.array([np.nan if (s.get(sk) or [None] * 3)[a] is None else s[sk][a] for s in S], float)
            if not np.isfinite(y).sum() >= 2:
                continue
            # the reference episode of a registration has σ 0 by definition: give it the median σ of the others
            if np.isfinite(sg).any():
                med = float(np.nanmedian(np.where(sg > 0, sg, np.nan))) if np.any(sg > 0) else 0.01
                sg = np.where(~np.isfinite(sg) | (sg <= 0), med, sg)
            else:
                sg = np.full(len(y), 0.05 if kind == 'rotation_deg' else 0.5)
            c = SL.classify(t, y, sg, min_step=mstep, min_amp=mstep, cov_name='time')
            name = ('rotation ' if kind == 'rotation_deg' else 'position ') + 'xyz'[a]
            c.update(unit=unit, axis=name, episodes=[S[i]['episode'] for i in c['index']])
            doc['axes'][name] = c
            for ev in c['events']:
                ev = dict(ev, axis=name, unit=unit)
                if 'j' in ev:
                    ev['episode'] = S[c['index'][ev['j']]]['episode']
                if 'j_end' in ev:
                    ev['episode_end'] = S[c['index'][ev['j_end']]]['episode']
                doc['events'].append(ev)
    # shift of each episode from the first estimate, against the requirement
    for s in S:
        r = [x if x is not None else 0.0 for x in s['rotation_deg']]
        r0 = [x if x is not None else 0.0 for x in first['rotation_deg']]
        s['shift_deg'] = float(np.linalg.norm(np.subtract(r, r0)))
        if all(x is not None for x in s.get('position_mm') or [None]) and \
                all(x is not None for x in first.get('position_mm') or [None]):
            s['shift_mm'] = float(np.linalg.norm(np.subtract(s['position_mm'], first['position_mm'])))
        else:
            s['shift_mm'] = None
    kinds = {e['kind'] for e in doc['events']}
    short = bool(doc['axes']) and all(c['kind'] == 'too short' for c in doc['axes'].values())
    doc['kind'] = ('slip' if 'step' in kinds else 'excursion' if 'excursion' in kinds else
                   'gradual' if 'gradual' in kinds else 'too short' if short else 'steady')


# ====================================================================== the campaign
def _times(ds, refs) -> tuple[list, str]:
    """Each episode's recording time in days from the first when the dataset says, else its order."""
    t = []
    for r in refs:
        v = (r.meta or {}).get('t0_ns')
        t.append(None if v is None else float(v) / 86400e9)
    if all(x is not None for x in t) and t:
        t0 = min(t)
        return [x - t0 for x in t], 'days'
    return [float(i) for i in range(len(refs))], 'episodes'


def audit_dataset(path_or_dataset, project, req: dict | None = None, settings: dict | None = None, *,
                  progress=None, cancel=None) -> dict:
    """Audit every episode of a dataset and the campaign (module docstring). JSON-able."""
    from ..datasets import Dataset, DatasetError, open_dataset
    from . import plan as AP
    t0 = time.perf_counter()
    st = settings_of(project, settings)
    req = req if req is not None else AE.requirements_of(project)
    own = not isinstance(path_or_dataset, Dataset)
    ds = open_dataset(path_or_dataset) if own else path_or_dataset
    report = progress or (lambda f, t='': None)
    try:
        refs = ds.episodes()
        if not refs:
            raise StudyError('The dataset has no episodes.')
        idx = [r.index for r in refs]
        if st['episodes'] is not None:
            want = {int(i) for i in st['episodes']}
            refs = [r for r in refs if r.index in want]
            if not refs:
                raise StudyError('None of the episodes asked for is in the dataset.')
        if st['max_episodes']:
            refs = refs[:int(st['max_episodes'])]
        ep0, first_err = None, None
        for r in refs:                                 # the first episode that reads and holds data sets the scene
            try:
                e0 = ds.load(r.index)
            except DatasetError as e:
                first_err = first_err or f'episode {r.id}: {str(e).rstrip(".")}'
                continue
            why = AE.empty_reason(e0)
            if why is None:
                ep0 = e0
                break
            first_err = first_err or f'episode {r.id}: {why}'
        if ep0 is None:
            raise StudyError(f'No episode of {_ds_name(ds) or "the dataset"} can be audited ({first_err}). The '
                             'files may be damaged or the recording empty.')
        plan = AP.PlanContext(project, ep0.joint_names) if project is not None and project.robots else None
        mapping = _dataset_mapping(project, ds)
        cam_match = AP.match_cameras(project, list(ep0.cameras), mapping) if project is not None else \
            {k: (None, 'no project') for k in ep0.cameras}
        cam_map = {k: v[0] for k, v in cam_match.items()}
        episodes = []
        n = len(refs)
        for k, r in enumerate(refs):
            if cancel is not None and cancel():
                break
            sub = (lambda f, t='', k=k: report(0.85 * (k + f) / n, t))
            episodes.append(AE.audit_episode(ds, r.index, req, plan, cam_map, progress=sub, cancel=cancel,
                                             per_frame=bool(st['per_frame'])))
        times, time_unit = _times(ds, refs[:len(episodes)])
        exclude = {}
        for e in episodes:
            for key, c in e['cameras'].items():
                bad = set(c['duplicates']['frames']) | {b['frame'] for b in c['blur']['frames']}
                for r_ in c['exposure']['runs']:
                    bad |= set(range(int(r_['start']), int(r_['end'])))
                exclude[(e['index'], key)] = bad
        readable = [e['index'] for e in episodes if not e.get('error')]
        mounts = mount_series(ds, project, plan, cam_map, readable, st, req, times=[
            t for t, e in zip(times, episodes) if not e.get('error')] if times is not None else None,
                              progress=lambda f, t='': report(0.85 + 0.13 * f, t), cancel=cancel, exclude=exclude)
        _mount_requirements(episodes, mounts, req)
        out = summarise(episodes, mounts, req, st)
        out.update(dataset={'format': ds.format, 'version': ds.version, 'episodes_total': len(idx),
                            'episodes_audited': len(episodes), 'name': _ds_name(ds),
                            **({'source': ds.source} if isinstance(getattr(ds, 'source', None), dict) else {})},
                   cameras={k: {'sensor_uid': v[0], 'matched': v[1]} for k, v in cam_match.items()},
                   time_unit=time_unit, plan_notes=list(plan.notes) if plan is not None else
                   ['No robot in the project: the checks that need the plan (desync against predicted image motion, '
                    'the gripper mask) use the joint speed or are skipped.'],
                   requirements_used={k: v for k, v in req.items()}, settings=st, seconds=round(
                       time.perf_counter() - t0, 2), cancelled=bool(cancel is not None and cancel()))
        report(1.0, 'done')
        return out
    finally:
        if own:
            ds.close()


def _ds_name(ds) -> str:
    import os
    return os.path.basename(os.path.normpath(getattr(ds, 'path', '') or '')) or ''


def _dataset_mapping(project, ds) -> dict:
    """The camera mapping stored for this dataset in project.data, if any."""
    import os
    if project is None:
        return {}
    for d in (getattr(project, 'data', None) or {}).get('datasets') or []:
        p = d.get('path') or ''
        if p and os.path.normpath(p) == os.path.normpath(getattr(ds, 'path', '') or ''):
            return dict(d.get('cameras') or {})
    return {}


def _mount_requirements(episodes: list, mounts: dict, req: dict):
    by = {e['index']: e for e in episodes}
    for key, m in mounts.items():
        for s in m['episodes']:
            e = by.get(s['episode'])
            if e is None:
                continue
            ok = s.get('shift_deg') is None or s['shift_deg'] <= req['max_mount_shift_deg']
            e['requirements'].append(AE._req('max_mount_shift_deg', s.get('shift_deg'), ok, camera=key,
                                             detail=f'{m["source"]}, from episode {m["episodes"][0]["episode"]}'))
            if s.get('shift_mm') is not None:
                e['requirements'].append(AE._req('max_mount_shift_mm', s['shift_mm'],
                                                 s['shift_mm'] <= req['max_mount_shift_mm'], camera=key))
    for e in episodes:
        fails = [r for r in e['requirements'] if not r['pass']]
        mount_only = fails and all(r['id'] in ('max_mount_shift_deg', 'max_mount_shift_mm') for r in fails)
        e['verdict'] = 'pass' if not fails else ('fix' if all(r['id'] in AE.FIXABLE for r in fails) else
                                                 'recalibrate' if mount_only else 'drop')
        e['flags'] = sorted({AE.FLAG[r['id']] for r in fails})


VERDICT_TEXT = {'pass': 'meets every requirement', 'fix': 'fixable: re-stamp the camera (offset or drift)',
                'recalibrate': 'usable with new extrinsics (the mount moved)', 'drop': 'drop or re-record'}


def summarise(episodes: list, mounts: dict, req: dict, st: dict) -> dict:
    per_req = {}
    for e in episodes:
        for r in e['requirements']:
            d = per_req.setdefault(r['id'], {'id': r['id'], 'title': r['title'], 'unit': r['unit'], 'checked': 0,
                                             'failed': 0, 'episodes_failed': [], 'worst': None,
                                             'limit': req.get(r['id']), 'default': r['id'] in req['defaults']})
            d['checked'] += 1
            if not r['pass']:
                d['failed'] += 1
                if e['index'] not in d['episodes_failed']:
                    d['episodes_failed'].append(e['index'])
            v = r['value']
            if isinstance(v, (int, float)) and v is not None and math.isfinite(v):
                d['worst'] = v if d['worst'] is None or abs(v) > abs(d['worst']) else d['worst']
    if AE.READABLE in per_req:                    # only unreadable episodes carry the row; every episode was tried
        per_req[AE.READABLE]['checked'] = len(episodes)
    order = list(AE.REQUIREMENTS) + [AE.READABLE]
    table =[per_req[k] for k in order if k in per_req]
    for d in table:
        d['pass'] = d['failed'] == 0
    verdicts = {k: [e['index'] for e in episodes if e['verdict'] == k] for k in VERDICT_TEXT}
    n = len(episodes)
    good = len(verdicts['pass'])
    offsets = {}
    for e in episodes:
        for key, c in e['cameras'].items():
            sy = c['sync']
            offsets.setdefault(key, []).append({'episode': e['index'], 'offset_ms': sy.get('offset_ms'),
                                                'sigma_ms': sy.get('sigma_ms')})
    slips = [{'camera': k, 'kind': m.get('kind'), 'events': m['events'], 'source': m['source']}
             for k, m in mounts.items() if m.get('kind') in ('slip', 'excursion', 'gradual')]
    frac = good / n if n else None
    word, sev = ('ready', 'good') if n and good == n else ('mostly usable', 'warning') if frac and frac >= 0.8 else \
        ('needs work', 'serious') if frac and frac >= 0.5 else ('not usable', 'critical')
    advice = _advice(verdicts, table, slips)
    return {'summary': {'episodes': n, 'pass': good, 'pass_pct': 100.0 * frac if n else None, 'status': sev,
                        'word': word, 'advice': advice, 'verdicts': verdicts,
                        'drop_or_rerecord': verdicts['drop'], 'fix': verdicts['fix'],
                        'recalibrate': verdicts['recalibrate']},
            'requirements': table, 'episodes': episodes, 'mounts': mounts, 'offsets': offsets, 'slips': slips,
            'verdict_text': VERDICT_TEXT}


def _advice(verdicts, table, slips) -> str:
    parts = []
    if verdicts['drop']:
        parts.append(f'Drop or re-record episode{"s" if len(verdicts["drop"]) > 1 else ""} '
                     f'{_list(verdicts["drop"])}')
    if verdicts['fix']:
        parts.append(f're-stamp the cameras of episode{"s" if len(verdicts["fix"]) > 1 else ""} '
                     f'{_list(verdicts["fix"])} (a constant offset or drift)')
    for s in slips:
        ev = next((e for e in s['events'] if e['kind'] == 'step'), None)
        if ev is not None:
            parts.append(f'{s["camera"]} moved at episode {ev.get("episode")} ({ev["axis"]} '
                         f'{ev["size"]:+.2f}{ev["unit"]}): calibrate it again from there')
        elif s['kind'] == 'gradual':
            ev = next((e for e in s['events'] if e['kind'] == 'gradual'), None)
            if ev is not None:
                u = '°' if 'rot' in ev['axis'] else ' mm'
                parts.append(f'{s["camera"]} drifts ({ev["axis"]} {ev["slope"]:+.3f}{u} per unit of time): check its '
                             'bracket')
    worst = [d for d in table if not d['pass']]
    if not parts and not worst:
        return 'Every episode meets every requirement.'
    if not parts:
        parts.append(f'{worst[0]["title"]} fails in {worst[0]["failed"]} checks')
    s = '; '.join(parts) + '.'
    return s[0].upper() + s[1:]


def _list(xs) -> str:
    xs = list(xs)
    if len(xs) > 8:
        return ', '.join(str(x) for x in xs[:8]) + f' and {len(xs) - 8} more'
    return ', '.join(str(x) for x in xs[:-1]) + (' and ' if len(xs) > 1 else '') + str(xs[-1])
