"""The dataset auditor against ground truth (core/audit, spec §8.2) and plan against recording (§8.3).

A seeded synthetic campaign (core/sim/episodes.py: the SO-100 sample, six episodes of 240 rows at 30 fps, 320 × 240
px, each fault kind injected into two episodes so the other four are clean for it, mount slip from episode 3) is
audited from its data alone (open_dataset; the sidecar dexvantage_truth.json is never read by the auditor), then the
findings are compared with the sidecar (score()):

- audit.fault_recall: per fault kind, the share of injected (episode, camera) faults the auditor finds (recall) and
  the share of clean (episode, camera) pairs it flags with that kind (false-positive rate). The error figure is the
  largest shortfall of a kind's recall below its floor (RECALL_FLOOR) and of its false-positive rate above
  FP_CEILING; the check passes at 0. The floors are what the measurement supports, stated per kind with why
  (FLOOR_WHY): kinds the estimator cannot resolve at the injected size are reported with floor 0, never claimed.
- audit.desync_offset: the desync estimate of each static camera against its true mean offset (the sidecar's content
  time minus stamp, averaged over the stored frames), the largest absolute error in ms against DESYNC_TOL_MS
  (20 ms: two thirds of a 30 fps frame period, the resolution a frame-difference signal supports; the estimator's
  largest error on these episodes was measured at 16.9 ms, with σ 9–20 ms, docs/models/audit.md). Cameras on the
  robot are reported, not counted (their predicted image motion comes from the scene's few edges: measured biases of
  up to about 50 ms on textureless synthetic scenes; their σ says so and keeps them from failing the requirement).
- audit.slip_vs_vantage: the step-against-gradual classifier (core/audit/slip.py, ported from Vantage
  core/online_calib.py) against Vantage's original on the same series: the original's outputs on twelve seeded series
  (steps, a returning shift, a slope, pure noise, an outlier) were recorded once from the Vantage checkout
  (scripts/record_vantage_slip.py; VANTAGE_SLIP.json beside this module: Vantage commit, date) and are compared
  term by term: the same steps at the same places, the same slope terms, and the fitted levels, sizes and slopes
  within 1e-9 (both are the same weighted least squares; a difference beyond rounding is a porting error). This is a
  check of the port against its original, not an independent validation of the method: the classifier's independent
  check is the mount_slip row of audit.fault_recall (the injected slip found from the data alone).
- audit.plan_vs_recording: the plan-against-recording comparison on the same campaign: the cameras matched to the
  plan, their recorded resolution against the plan's stream scaled to the recording, and the mount difference on the
  slipped camera against the sidecar's true delta: |Δω_recovered − Δω_true|, the difference of the two rotation
  vectors (deg, both in the sensor's body frame), so a rotation about the wrong axis counts (comparing only their
  magnitudes, as this check first did, read 0.020° where the vectors are 0.20° apart). Tolerance PLAN_TOL_DEG (0.3°):
  the registration fits a pure rotation, and the slip also moves the camera 3 mm; seen at the scene's distance from the
  SO-100 third-person camera (0.56 m to the task volume's centre) 3 mm subtends 0.31°, which a rotation-only fit can
  absorb in part.
"""
from __future__ import annotations

import json
import math
import os
import tempfile

import numpy as np

from .suite import Outcome, check

SETTINGS = {'episodes': 6, 'frames': 240, 'size': [320, 240], 'seed': 7,
            'faults': {'drop': {'episodes': [0, 3]}, 'duplicate': {'episodes': [1, 4]},
                       'desync': {'episodes': [0, 2]}, 'drift': {'episodes': [1, 5]},
                       'jitter': {'episodes': [2, 3]}, 'exposure_change': {'episodes': [3, 4]},
                       'blur': {'episodes': [0, 5]}, 'gripper_hidden': {'episodes': [1, 2]},
                       'usb_reset': {'episodes': [4, 5]}, 'mount_slip': {'from_episode': 3}}}
KINDS = ('drop', 'duplicate', 'desync', 'drift', 'jitter', 'exposure_change', 'blur', 'gripper_hidden',
         'usb_reset', 'mount_slip')
RECALL_FLOOR = {'drop': 1.0, 'usb_reset': 1.0, 'jitter': 1.0, 'exposure_change': 1.0, 'gripper_hidden': 1.0,
                'mount_slip': 1.0, 'duplicate': 0.5, 'desync': 0.5, 'blur': 0.5, 'drift': 0.0}
FP_CEILING = {k: 0.0 for k in KINDS}
FP_CEILING.update({'blur': 0.25, 'desync': 0.0, 'drift': 0.0, 'duplicate': 0.1})
FLOOR_WHY = {
    'drop': 'a missing slot is a timestamp gap of two periods: exact',
    'usb_reset': 'a gap of 0.25 s: exact',
    'jitter': '3 ms Gaussian stamps against a 1 ms requirement: far apart',
    'exposure_change': 'a 1.4 gain step is ln 1.4 = 0.34 against a 0.14 threshold',
    'gripper_hidden': 'a flat painted occluder over the plan\'s gripper mask: no correlation left',
    'mount_slip': 'a 1.5° rotation moves the background 4–6 px at 320 × 240',
    'duplicate': 'a stale frame re-encoded by the codec (g = 2) is 30–40 dB from its original, so only the drop in '
                 'change while the plan says the scene moves shows it; on the third-person camera the robot is small '
                 'and the drop is near the codec\'s noise',
    'desync': '50 ms against the 25 ms requirement and a σ of 10–20 ms on static cameras; cameras on the robot are '
              'not resolved (their σ is large)',
    'blur': 'the synthetic blur averages one frame period at the fastest motion: a 2–3 px smear of a small robot, '
            'sharpness ratios 0.64–0.91 against the 0.85 threshold',
    'drift': '5000 ppm over 8 s moves the offset by 40 ms, about the estimator\'s own scatter: not resolved, '
             'reported only',
}
DESYNC_TOL_MS = 20.0
PLAN_TOL_DEG = 0.3                 # module docstring: the slip's 3 mm at 0.56 m subtends 0.31°


# ====================================================================== scoring against the sidecar
def _flagged(cam: dict, kind: str) -> set:
    """Frames (or {-1} for a whole-camera finding) the auditor reports for one kind on one camera."""
    tm = cam['timing']
    if kind == 'drop':
        return {g['frame'] for g in tm['drops']}
    if kind == 'usb_reset':
        return {g['frame'] for g in tm['usb_gaps']}
    if kind == 'duplicate':
        return set(cam['duplicates']['frames'])
    if kind == 'blur':
        return {b['frame'] for b in cam['blur']['frames']}
    if kind == 'gripper_hidden':
        return set(cam['gripper'].get('hidden') or [])
    if kind == 'exposure_change':
        out = set()
        for r in cam['exposure']['runs']:
            out |= set(range(int(r['start']), int(r['end'])))
        return out
    return set()


def _req_fail(ep: dict, rid: str, key: str) -> bool:
    return any(r['id'] == rid and r['camera'] == key and not r['pass'] for r in ep['requirements'])


REQ_OF = {'desync': 'max_offset_ms', 'drift': 'max_drift_ppm', 'jitter': 'max_jitter_ms'}


def score(result: dict, truth: dict) -> dict:
    """{kind: {'injected', 'found', 'recall', 'clean', 'false', 'fp_rate', 'frames_injected', 'frames_found'}} of an
    audit result (core/audit/campaign.audit_dataset) against a synthetic dataset's sidecar."""
    out = {k: {'injected': 0, 'found': 0, 'clean': 0, 'false': 0, 'frames_injected': 0, 'frames_found': 0,
               'missed': [], 'false_at': []} for k in KINDS}
    keys = list(truth['cameras'])
    for ep in result['episodes']:
        te = truth['episodes'][str(ep['index'])]
        for key in keys:
            cam = ep['cameras'].get(key)
            if cam is None:
                continue
            for kind in KINDS:
                if kind == 'mount_slip':
                    continue
                inj = [f for f in te['faults'] if f['kind'] == kind and f['camera'] == key]
                d = out[kind]
                if kind in REQ_OF:
                    hit = _req_fail(ep, REQ_OF[kind], key)
                else:
                    fl = _flagged(cam, kind)
                    hit = bool(fl)
                if inj:
                    d['injected'] += 1
                    if kind in REQ_OF:
                        ok = hit
                    else:
                        want = set()
                        for f in inj:
                            want |= set(f.get('frames') or [])
                        near = {x + o for x in fl for o in (-1, 0, 1)}
                        got = want & near
                        d['frames_injected'] += len(want)
                        d['frames_found'] += len(got)
                        ok = bool(got)
                    d['found'] += int(ok)
                    if not ok:
                        d['missed'].append([ep['index'], key])
                else:
                    d['clean'] += 1
                    if hit:
                        d['false'] += 1
                        d['false_at'].append([ep['index'], key])
    # mount slip: per camera over the campaign
    slip_cams = {}
    for e, te in truth['episodes'].items():
        for f in te['faults']:
            if f['kind'] == 'mount_slip':
                slip_cams[f['camera']] = int(f['from_episode'])
    d = out['mount_slip']
    for key in keys:
        m = (result.get('mounts') or {}).get(key) or {}
        steps = [ev for ev in m.get('events') or [] if ev['kind'] in ('step', 'gradual')]
        if key in slip_cams:
            d['injected'] += 1
            ok = any(ev['kind'] == 'step' and abs(int(ev.get('episode', -99)) - slip_cams[key]) <= 1 for ev in steps)
            d['found'] += int(ok)
            if not ok:
                d['missed'].append([None, key])
        else:
            d['clean'] += 1
            if steps:
                d['false'] += 1
                d['false_at'].append([None, key])
    for d in out.values():
        d['recall'] = d['found'] / d['injected'] if d['injected'] else None
        d['fp_rate'] = d['false'] / d['clean'] if d['clean'] else None
    return out


def desync_errors(result: dict, truth: dict) -> list:
    """[(episode, camera, estimate ms, true mean offset ms, σ ms, on the robot)] for every camera with an estimate."""
    rows = []
    for ep in result['episodes']:
        te = truth['episodes'][str(ep['index'])]
        for key, cam in ep['cameras'].items():
            sy = cam['sync']
            tm = te['timing'].get(key)
            if sy.get('offset_ms') is None or tm is None:
                continue
            true = 1000.0 * float(np.mean(np.asarray(tm['content_time_s']) - np.asarray(tm['stamp_s'])))
            rows.append((ep['index'], key, float(sy['offset_ms']), true, sy.get('sigma_ms'),
                         bool(truth['cameras'][key].get('wrist'))))
    return rows


# ====================================================================== the campaign (shared, made once per run)
_CACHE: dict = {}


def campaign(ctx=None) -> tuple:
    """(audit result, truth, dataset folder, project) of the seeded campaign, generated once per process: written,
    copied without its sidecar, the copy audited, and only then the sidecar read."""
    quick = bool(getattr(ctx, 'quick', False))
    if _CACHE.get('quick') == quick and 'c' in _CACHE and os.path.isdir(_CACHE['c'][2]):   # a run's folder goes
        return _CACHE['c']
    import shutil
    from ..core.audit import campaign as AC
    from ..core.robot.samples import sample_project
    from ..core.sim import episodes as SE
    project = sample_project('so100')
    base = ctx.tmp() if ctx is not None and hasattr(ctx, 'tmp') else tempfile.mkdtemp(prefix='dexvantage-audit-')
    root = os.path.join(base, 'campaign')
    SE.generate(project, root, settings_for(ctx), overwrite=True)
    blind = os.path.join(base, 'blind')
    shutil.rmtree(blind, ignore_errors=True)
    shutil.copytree(root, blind)
    os.remove(os.path.join(blind, 'dexvantage_truth.json'))
    res = AC.audit_dataset(blind, project, settings={'per_frame': False})
    with open(os.path.join(root, 'dexvantage_truth.json'), encoding='utf-8') as fh:
        truth = json.load(fh)
    _CACHE['c'] = (res, truth, blind, project)
    _CACHE['quick'] = quick
    return _CACHE['c']


def settings_for(ctx=None) -> dict:
    """The campaign's settings: SETTINGS, with 120 rows per episode (4 s) instead of 240 in a quick run. Every fault
    is still injected in the same episodes; the floors and ceilings are unchanged (measured to hold at both sizes,
    2026-11-07)."""
    return dict(SETTINGS, frames=120) if getattr(ctx, 'quick', False) else dict(SETTINGS)


# ====================================================================== checks
@check('audit.fault_recall', 'data', 'Auditor: every injected fault kind found from the data alone',
       'per fault kind, recall and false-positive rate over (episode, camera) pairs; error = the largest shortfall '
       'against the stated floors and ceilings',
       'the ground-truth sidecar of a seeded synthetic campaign (core/sim/episodes.py), read only after the audit',
       0.0, 'share', refs=('lerobot_dataset_format', 'pech_pacheco2000', 'huber1964'), cost='study')
def fault_recall(ctx):
    res, truth, _root, _p = campaign(ctx)
    sc = score(res, truth)
    worst = 0.0
    rows = {}
    for k, d in sc.items():
        r, f = d['recall'], d['fp_rate']
        short = max(0.0, RECALL_FLOOR[k] - (r if r is not None else 1.0))
        over = max(0.0, (f if f is not None else 0.0) - FP_CEILING[k])
        worst = max(worst, short, over)
        rows[k] = {'recall': r, 'fp_rate': f, 'injected': d['injected'], 'clean': d['clean'],
                   'frames': f'{d["frames_found"]}/{d["frames_injected"]}' if d['frames_injected'] else None,
                   'floor': RECALL_FLOOR[k], 'fp_ceiling': FP_CEILING[k], 'why': FLOOR_WHY[k],
                   'missed': d['missed'], 'false_at': d['false_at']}
    return Outcome(worst, {'kinds': rows, 'settings': settings_for(ctx), 'seconds_audit': res.get('seconds')},
                   note='Recall and false-positive rate per fault kind; floors are what the method supports at the '
                        'injected sizes (FLOOR_WHY), drift is reported, not claimed.')


@check('audit.desync_offset', 'data', 'Auditor: camera-joint offset against the injected offset',
       'largest |estimated − true mean offset| over the static cameras of the campaign (ms)',
       'the sidecar\'s per-frame content time minus stamp (the injected desync, drift and jitter combined)',
       DESYNC_TOL_MS, 'ms', refs=('knapp_carter1976', 'spearman1904', 'kunsch1989'), cost='study')
def desync_offset(ctx):
    res, truth, _root, _p = campaign(ctx)
    rows = desync_errors(res, truth)
    static = [r for r in rows if not r[5]]
    err = max((abs(r[2] - r[3]) for r in static), default=float('nan'))
    return Outcome(err, {'cameras': [{'episode': r[0], 'camera': r[1], 'estimate_ms': r[2], 'true_ms': r[3],
                                      'sigma_ms': r[4], 'on_robot': r[5]} for r in rows],
                         'on_robot_max_error_ms': max((abs(r[2] - r[3]) for r in rows if r[5]), default=None)},
                   note='Static cameras are counted; cameras on the robot are listed (not resolved on these scenes).')


@check('audit.slip_vs_vantage', 'data', 'Mount slip: step against gradual flex, against Vantage\'s original',
       'largest difference of the fitted level, step sizes and slopes, with the same terms chosen',
       'Vantage core/online_calib.py select_model on the same twelve series, its outputs recorded once',
       1e-9, 'units of the series', refs=('kass_raftery1995',))
def slip_vs_vantage(ctx):
    from ..core.audit import slip as SL
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VANTAGE_SLIP.json')
    with open(path, encoding='utf-8') as fh:
        rec = json.load(fh)
    worst, bad, rows = 0.0, [], []
    for s in rec['series']:
        m = SL.select_model(s['t'], s['y'], s['sigma'], {k: v for k, v in (s.get('covs') or {}).items()},
                            1.0, s['margin'], s['min_seg'], s['max_terms'], s['huber_k'], s['min_step'], s['min_amp'],
                            s['scale_floor'])
        mine = {'steps': [[st['j'], st['size']] for st in m['steps']], 'flex': [[f['cov'], f['slope']]
                                                                               for f in m['flex']],
                'level': m['level']}
        want = s['vantage']
        same = [x[0] for x in mine['steps']] == [x[0] for x in want['steps']] and \
            [x[0] for x in mine['flex']] == [x[0] for x in want['flex']]
        d = abs(mine['level'] - want['level'])
        if same:
            for a, b in zip(mine['steps'], want['steps']):
                d = max(d, abs(a[1] - b[1]))
            for a, b in zip(mine['flex'], want['flex']):
                d = max(d, abs(a[1] - b[1]))
        else:
            bad.append(s['name'])
            d = math.inf
        worst = max(worst, d)
        rows.append({'series': s['name'], 'terms_same': same, 'max_difference': d, 'vantage': want, 'port': mine})
    return Outcome(worst, {'series': rows, 'recorded': rec['recorded']}, ok=not bad,
                   note='Vantage\'s outputs were recorded from its checkout (commit and date in the detail).')


@check('audit.plan_vs_recording', 'data', 'Plan against recording: cameras, resolution and the slipped mount',
       '|recovered mount rotation change − the sidecar\'s true mount slip| (deg): the difference of the two rotation '
       'vectors in the sensor\'s body frame, with every camera matched',
       'the sidecar\'s cameras (sensor uid, size) and its mount_slip delta', PLAN_TOL_DEG, '°',
       refs=('lucas_kanade1981', 'baker_matthews2004'),
       cost='study')
def plan_vs_recording(ctx):
    from ..core.studies import plan_vs_recording as PR
    res, truth, root, project = campaign(ctx)
    doc = PR.compare(project, root, audit=res)
    errs, ok = [], True
    for key, c in truth['cameras'].items():
        row = next((r for r in doc['cameras'] if r['key'] == key), None)
        if row is None or row.get('sensor_uid') != c['sensor_uid']:
            ok = False
            continue
    slip = None
    for te in truth['episodes'].values():
        for f in te['faults']:
            if f['kind'] == 'mount_slip':
                slip = f
    if slip is not None:
        row = next((r for r in doc['cameras'] if r['key'] == slip['camera']), None)
        got = (row or {}).get('mount', {}).get('change_deg')
        if got is None:
            ok = False
        else:
            want = np.asarray(slip['delta']['rotation_deg'], float)
            got = np.asarray(got, float)
            errs.append(float(np.linalg.norm(got - want)))
            mag = abs(float(np.linalg.norm(got)) - float(np.linalg.norm(want)))
    err = max(errs) if errs else float('nan')
    detail = {'cameras': doc['cameras'], 'summary': doc['summary']}
    if errs:
        detail.update({'recovered_deg': got.tolist(), 'true_deg': want.tolist(), 'magnitude_difference_deg': mag})
    return Outcome(err, detail, ok=ok,
                   note='The rotation vectors are compared, not only their sizes. The registration fits a pure '
                        'rotation and the slip also moves the camera 3 mm, which at the scene\'s 0.56 m subtends up to '
                        '0.31°: the tolerance states that.')
