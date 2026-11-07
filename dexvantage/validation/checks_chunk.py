"""Checks of the action-chunk latency study (core/studies/action_chunk_sync.py, spec §7.13) and of the visual-to-
tactile blind gap (core/studies/dex_hand_tactile.py part 2, spec §7.15) against independent references.

- chunk.loop_vs_simulation: the close-timing model (the sum of the loop's terms, with the uniform phases of the
  camera, the policy loop and the control ticks) against an event-by-event simulation of a point moving at constant
  speed towards a contact, seen by a camera with the same exposure, readout and transfer: frames at their own clock,
  the newest frame ready when inference starts, its timestamp (host arrival or mid-exposure), the chunk index the
  ideal imitator chooses for the close (rounded to whole steps), the first control tick after the chunk is ready,
  the skipped actions and the command latency. The simulation never adds the model's terms; it plays the events.
  Reported: the largest of |mean(simulated) − mean(model)| and how far any simulated close falls outside the model's
  interval, in mm at 0.1 m/s. Tolerance 0.05 mm: about five standard errors of the simulated mean (20 000 trials,
  spread about 1.4 mm); the simulated spread must also match the model's (the standard deviation of its independent
  uniform phases) within 3 %, so a model that is merely loose fails.
- chunk.ensemble_closed_form: ActEnsembler (ACT's all_time_actions buffer, step by step) on a point moving at constant
  speed whose every chunk holds the point where it was seen, and ensemble_mean_age, against the closed form of the
  exponentially weighted average written out here: with r = e^(−m) and n predictions, the lag in steps is
  (n − 1) − S1/S0, S0 = (1 − rⁿ)/(1 − r), S1 = r(1 − n rⁿ⁻¹ + (n − 1) rⁿ)/(1 − r)² ((n − 1)/2 for m = 0), and the
  ensembled vector against Σ wᵢ aᵢ / Σ wᵢ with the weights from the formula; i = 0 is the oldest (ACT's code).
  Tolerance 1e-9 steps (floating point).
- dexhand.blind_gap_bruteforce: the sensory blind gap of the handoff (coarse distances, then the refine spacing only
  inside intervals where a fingertip's seen state changes) against a brute-force scan of every distance at the refine
  spacing, on the dexhand sample with a wall that hides the fingertips from the head camera over the last part of
  the approach. Same kernel, same distances where the two meet, so the difference can come only from a short seen or
  hidden stretch the coarse samples skip. Tolerance: one refine step (2 mm here).
"""
from __future__ import annotations

import math

import numpy as np

from .suite import Outcome, check, figure, series

V_MS = 0.1                         # m/s: the point's speed in the loop check


def _scenario(k: int) -> dict:
    """Loop figures for the check (typed in here; any plausible values serve, they are not device figures)."""
    base = {'exp': 0.008, 'readout': 0.012, 'transfer': 0.005, 'fps': 30.0, 'tau_i': 0.073, 'hz': 50.0,
            'cmd': 0.020, 'cmd_rec': 0.020, 'delta': 0.040, 'stamp': 'host_receive', 'skip': 0, 'extra': 0.0}
    variants = [{}, {'stamp': 'mid_exposure'}, {'skip': 2, 'cmd_rec': 0.010, 'hz': 20.0, 'tau_i': 0.1},
                {'fps': 90.0, 'tau_i': 0.010, 'delta': 0.0, 'extra': 0.004, 'stamp': 'mid_exposure'}]
    return dict(base, **variants[k])


def _model(sc: dict):
    from ..core.studies import action_chunk_sync as A
    dt = 1.0 / sc['hz']
    terms = [A._term('exposure_half', sc['exp'] / 2, sc['exp'] / 2, 'user', ''),
             A._term('readout', sc['readout'], sc['readout'], 'user', ''),
             A._term('transfer', sc['transfer'], sc['transfer'], 'user', ''),
             A._term('queue', sc['extra'], sc['extra'] + 1 / sc['fps'], 'default', ''),
             A._term('inference', sc['tau_i'], sc['tau_i'], 'policy', ''),
             A._term('tick', 0.0, dt, 'default', ''),
             A._term('command', sc['cmd'], sc['cmd'], 'user', '')]
    sch = {'dt': dt, 'hz': sc['hz'], 's': 1, 'H': 100, 'ensemble': False, 'm': None}
    st = {'command_rec_ms': sc['cmd_rec'] * 1000, 'skip_steps': sc['skip']}
    te = A.timing_error({'terms': terms}, sch, st, sc['stamp'], sc['delta'])
    return te['e_s'], te['e_sd_s']


def _simulate(sc: dict, rng, n: int) -> np.ndarray:
    """Close time − contact time (s) for n trials, event by event (module docstring)."""
    dt, Tf = 1.0 / sc['hz'], 1.0 / sc['fps']
    ready_after = sc['exp'] / 2 + sc['readout'] + sc['transfer']        # mid-exposure to arrival at the host
    phi_cam = rng.uniform(0, Tf, n)                                      # the camera's clock
    phi_tick = rng.uniform(0, dt, n)                                     # the controller's clock
    s = 100.0 + rng.uniform(0, 1.0, n)                                   # inference starts (the policy's clock)
    look = s - sc['extra']                                               # the frame taken: newest ready by then
    k = np.floor((look - ready_after - phi_cam) / Tf)
    t_mid = phi_cam + k * Tf                                             # its mid-exposure
    t_stamp = t_mid + ready_after if sc['stamp'] == 'host_receive' else t_mid
    t_c = s + rng.uniform(0.3, 0.6, n)                                   # when the moving point reaches the contact
    # the ideal imitator: in training, the image stamped t came with the action stamped t + delta, and the close
    # command was sent cmd_rec before contact; it predicts the point reaches the contact at t_c
    j = np.round((t_c - sc['cmd_rec'] - t_stamp - sc['delta']) / dt)
    ready = s + sc['tau_i']
    t0 = phi_tick + np.ceil((ready - phi_tick) / dt) * dt                # first control tick after the chunk is ready
    t_close = t0 + (j - sc['skip']) * dt + sc['cmd']
    return t_close - t_c


@check('chunk.loop_vs_simulation', 'timing', 'Action-chunk loop against a discrete-time simulation of a moving point',
       'the largest of |mean(simulated) − mean(model)| and the simulated closes outside the model\'s interval, '
       'position error of a point at 0.1 m/s, over four loop scenarios (host and mid-exposure stamps, skipped '
       'actions, a command latency that differs from the recording\'s)',
       'event-by-event simulation of frames, inference starts, chunk indexing and control ticks (20 000 trials each)',
       5e-5, 'm', refs=('zhao2023_act', 'act_code'))
def _loop(ctx):
    rows, worst, ok = [], 0.0, True
    xs, ym, ys = [], [], []
    for k in range(4):
        sc = _scenario(k)
        (lo, mid, hi), sd = _model(sc)
        e = _simulate(sc, ctx.rng('chunk.loop_vs_simulation', k), 20000)
        mean_err = float(abs(e.mean() - mid) * V_MS)
        out = float(max(0.0, lo - e.min(), e.max() - hi) * V_MS)
        sd_ratio = float(e.std() / sd)
        ok = ok and abs(sd_ratio - 1.0) <= 0.03
        worst = max(worst, mean_err, out)
        rows.append({'scenario': k, 'stamp': sc['stamp'], 'skip': sc['skip'], 'model_s': [lo, mid, hi],
                     'simulated_s': [float(e.min()), float(e.mean()), float(e.max())], 'mean_err_m': mean_err,
                     'outside_m': out, 'sd_ratio': sd_ratio})
        xs.append(k)
        ym.append(mid * 1000)
        ys.append(e.mean() * 1000)
    return Outcome(worst, {'scenarios': rows, 'speed_m_s': V_MS}, ok=ok,
                   figure=figure('Close-timing error: model against simulation', 'scenario', 'mean error, ms',
                                 [series('model', xs, ym, 'points'), series('simulation', xs, ys, 'points')]),
                   note='The simulation plays the events (frames on their own clock, the newest frame ready at '
                        'inference start, the chunk index rounded to whole steps, the next control tick); the model '
                        'adds the terms with their ranges.')


def _closed_lag(m: float, n: int) -> float:
    if n <= 1:
        return 0.0
    if abs(m) < 1e-12:
        return (n - 1) / 2.0
    r = math.exp(-m)
    s0 = (1 - r ** n) / (1 - r)
    s1 = r * (1 - n * r ** (n - 1) + (n - 1) * r ** n) / (1 - r) ** 2
    return (n - 1) - s1 / s0


@check('chunk.ensemble_closed_form', 'timing', 'Temporal ensembling against the closed-form weighted average',
       'largest difference, in control steps, between the lag of ACT\'s step-by-step ensemble of a moving point (and '
       'ensemble_mean_age) and the geometric-series closed form; and between the ensembled vector and Σ wᵢ aᵢ / Σ wᵢ',
       'the closed form of exp(−m·i) weights, i = 0 the oldest (ACT code)', 1e-9, 'steps',
       refs=('zhao2023_act', 'act_code', 'lerobot_act_ensembler'))
def _ensemble(ctx):
    from ..core.studies import action_chunk_sync as A
    rng = ctx.rng('chunk.ensemble_closed_form')
    worst, rows = 0.0, []
    for m, H in ((0.01, 100), (0.0, 50), (-0.1, 30), (0.2, 16), (1.5, 8), (0.05, 1)):
        ens = A.ActEnsembler(H, m)
        lag = None
        for t in range(3 * H + 2):                               # a point at x = t (one unit per step)
            ens.add(t, np.full((H, 1), float(t)))
            a = float(ens.action(t)[0])
            if t >= H - 1:
                lag = t - a
        ref = _closed_lag(m, H)
        mod = A.ensemble_mean_age(m, H, 1)
        # the vector: random predictions for one step, ACT order (the oldest first)
        P = rng.normal(size=(H, 3))
        ens2 = A.ActEnsembler(H, m)
        for i in range(H):                                       # chunk i issued at step i covers step H − 1
            c = np.zeros((H, 3))
            c[H - 1 - i] = P[i]
            ens2.add(i, c)
        got = ens2.action(H - 1)
        r = math.exp(-m)
        w = np.array([r ** i for i in range(H)])
        want = (w[:, None] * P).sum(axis=0) / w.sum()
        err = max(abs(lag - ref), abs(mod - ref), float(np.abs(got - want).max()))
        worst = max(worst, err)
        rows.append({'m': m, 'H': H, 'lag_steps': lag, 'closed_form_steps': ref, 'mean_age_steps': mod,
                     'error': err})
    return Outcome(worst, {'cases': rows},
                   figure=figure('Ensembled lag: ACT buffer against the closed form', 'm', 'lag, steps',
                                 [series('closed form', [r['m'] for r in rows], [r['closed_form_steps'] for r in rows],
                                         'points'),
                                  series('ACT buffer', [r['m'] for r in rows], [r['lag_steps'] for r in rows],
                                         'points')]),
                   note='i = 0 is the oldest prediction, as in ACT\'s code; a negative m weights the newest more.')


def blind_gap_scene(quick: bool = False):
    """The dexhand sample with only its head camera and a wall (a task object; objects occlude along the approach)
    between it and the fingertips: a vertical plate 40 % of the way from the fingertips to the camera whose top edge
    is where the line of sight to the patches' centre passes when the hand is 20 mm before contact, so the head camera
    loses some fingertips over the last part of the approach and keeps others; the first approach is the fully closed
    power posture."""
    from ..core import grasps as GK
    from ..core.frames import Pose
    from ..core.project import TaskObject
    from ..core.robot import samples as S
    from ..core.studies import dex_hand_tactile as D
    p = S.sample_project('dexhand')
    head = next(s for s in p.variant.sensors if s.name.startswith('Head'))
    st = dict(p.analysis.studies[D.STUDY_ID])
    st.update({'sensors': [head.uid], 'handoff_max_mm': 40.0, 'handoff_step_mm': 10.0,
               'handoff_refine_mm': 4.0 if quick else 2.0})
    p.analysis.studies[D.STUDY_ID] = st
    S_ = D.settings_of(p)
    robot, model, hand, cases, cams, qd = D._setup(p, S_, [])
    c = D._handoff_cases(cases, S_)[0]
    ap = D._Approach(p, robot, model, hand, cams, S_, qd, [c])
    tip = np.concatenate([v['points'] for v in GK.contact_patches(p, robot.uid, c['q']).values()]).mean(axis=0)
    cam = np.asarray(head.mount.pose.position, float)
    t20 = tip - 0.020 * ap.a[0]                        # the patches' centre 20 mm before contact
    x = tip[0] + 0.4 * (cam[0] - tip[0])
    lam = (x - t20[0]) / (cam[0] - t20[0])
    top = t20[2] + lam * (cam[2] - t20[2])
    h = 0.20
    p.workspace.objects.append(TaskObject(name='Wall (check)', shape='box', size=(0.004, 0.40, h),
                                          pose=Pose(np.array([x, tip[1], top - h / 2]), np.array([0, 0, 0, 1.0]))))
    return p


@check('dexhand.blind_gap_bruteforce', 'dexhand', 'Sensory blind gap against a brute-force scan of the approach',
       'largest difference between the blind gap (coarse distances, refined inside transitions) and a scan of every '
       'distance at the refine spacing, over the fingertips of one approach on the dexhand sample with a wall',
       'brute-force scan of the approach with the same visibility kernel', 2e-3, 'm', refs=('moller1997',),
       cost='study')
def _blind_gap(ctx):
    from ..core.studies import dex_hand_tactile as D
    p = blind_gap_scene(ctx.quick)
    out = D.blind_gap_check(p)
    tol = out['refine_m']
    return Outcome(out['error_m'], out, tolerance=tol,
                   figure=figure('Blind gap per fingertip: adaptive against brute force', 'fingertip', 'gap, mm',
                                 [series('adaptive', list(range(len(out['fingers']))),
                                         [g * 1000 for g in out['adaptive_m']], 'points'),
                                  series('brute force', list(range(len(out['fingers']))),
                                         [g * 1000 for g in out['brute_m']], 'points')]),
                   note=f'{out["evaluations_adaptive"]} visibility evaluations against {out["evaluations_brute"]} for '
                        'the scan; the tolerance is one refine step.')
