"""Action-chunk horizon latency and temporal alignment (spec §7.13): how late or early an action-chunked policy (ACT,
Diffusion Policy, π0 flow matching) closes the gripper on the contact point, given the rig's sensor-to-action loop
and the arm's approach speed, and what to write into the training config to correct it.

    @jobs.study('action_chunk_sync', 'Action-chunk latency', quick=False, batch='action-chunk') → --action-chunk OUT
    run(project, ctx) -> dict                  the study (also core.jobs.run_study('action_chunk_sync', project))
    settings_of(project) -> dict               the settings with the labelled defaults filled in
    ensemble_weights(m, n) -> (n,)             ACT's w_i = exp(−m·i), i = 0 the OLDEST prediction
    ensemble_mean_age(m, H, s=1) -> steps      the weighted mean age (in control steps) of the ensembled action
    ensemble_n_eff(m, H, s=1) -> float         the effective number of predictions averaged, (Σw)² / Σw²
    ActEnsembler(H, m)                         ACT's all_time_actions buffer, step by step (the official code)
    loop_terms(...), timing_error(...)         the loop and the close-timing error, term by term
    apply_recommendations(project, rec)        what the page's Apply does (one undo step there)

**The loop** (per policy and per camera it takes; every term has a value, an origin and a source):

    age of the image when the policy starts on it = τ_e/2 + τ_r + τ_x + τ_q
    the first action reaches the arm               + τ_i + τ_w + τ_c

- τ_e/2: half the exposure. An image shows the scene at the middle of its exposure, so half the exposure has passed
  when the exposure ends. τ_r: the readout (the last row read after the exposure; for a rolling shutter that is the
  row read last, an upper bound for the rows above it). Both come from the camera's library entry (an imager's
  exposure_s / readout_s, or exposure_ms / readout_ms, as a number or {value, source}) or the user's figure with its
  source (settings figures {sensor uid: {exposure_ms, readout_ms, transfer_ms: {value, source}}}). No entry in the
  library states them today; a figure not stated is said and left out (the loop is then "at least").
- τ_x: the transfer of one frame of the policy's stream over its link: frame bits × (1 + protocol overhead) / the
  bandwidth left to it, from the bandwidth study (core/studies/bandwidth.py, §7.6): the lower of the link's capacity
  less the sensor's other streams, and the controller's capacity less everything else it carries (those streams take
  their share continuously). A sensor not plugged into a controller has no transfer figure; a link over budget backs
  frames up without bound and is refused as a figure.
- τ_q: host queueing. The policy takes the newest frame when it starts; the camera and the policy loop run on
  independent clocks, so the newest frame's age is uniform over one frame period, 0 to 1/fps (mean 1/(2 fps)), plus
  queue_extra_ms (default 0 ms). A labelled default model, not a measurement.
- τ_i: one inference, from the policy entry (core/policies.resolve: cited or measured). A policy whose inference time
  no source states is refused with a sentence that says so and where to enter a measurement.
- τ_w: the chunk's first action waits for the next control tick, uniform over one control period Δt = 1/control_hz
  (the same independent-clocks assumption).
- τ_c: command latency, from the command leaving the host to the arm acting on it: command_ms when the user sets it
  with its source, else one control period (a labelled default).

**The close-timing error.** The policy is modelled as an ideal imitator of its demonstrations (an assumption, the
model's core): it closes the gripper on the frame that, in the training data, came with the close command. In the
recorded data the close command stamped t_c − c_rec (c_rec: the follower's command latency while recording, default
the same τ_c) made the gripper close at contact t_c, and the training pairs the image stamped t with the action
stamped t + δ (δ: the proprioceptive delay offset of the training config, default 0, as LeRobot's delta_timestamps
start at 0). An image stamped on arrival at the host ('host_receive', the default) carries the age a = τ_e/2 + τ_r +
τ_x; an image stamped at mid-exposure (a hardware timestamp, 'mid_exposure'; the default when the time-sync settings
put the stream on a camera hardware clock) carries none. At deployment the close reaches the gripper

    e = (τ_e/2 + τ_r + τ_x − a) + τ_q + τ_i + τ_w + ρ + (τ_c − c_rec) − δ − k·Δt

after contact (e > 0: late, the hand pushes on past the contact point, the OVERSHOOT; e < 0: early, the fingers close
in the air, the AIR-GRASP), where ρ, uniform over ±Δt/2, is the close landing on a whole control step and k the
actions the executor skips at the start of a chunk (skip_steps, default 0). The interval [low–high] is the sum of
the terms' ranges, the nominal value the sum of their means. Spatially, the arm follows the demonstrated motion, so
the error is how far the tool centre is from the contact point when the fingers close: along a trajectory that
closes the gripper at t_c, with P(t) its tool centre,

    overshoot = |P(t_c + e_high) − P(t_c)| (e_high > 0),   air-grasp = |P(t_c + e_low) − P(t_c)| (e_low < 0)

(the worst closing stroke; a hand that stops before it closes loses nothing to timing), and without a trajectory
v · max(0, e_high) and v · max(0, −e_low) at the approach speed v.

**Chunks and temporal ensembling** change how old the information behind an executed action is (the reaction age),
not the timing above. A new chunk comes every s = max(E, ⌈τ_i/Δt⌉) control steps (E: the actions executed per chunk,
from the entry; asynchronous inference). Executed open loop (Diffusion Policy's receding horizon, π0), action k of a
chunk rests on an image L + k·Δt old, k = 0 … s − 1. With temporal ensembling (ACT; E = 1), the executed action is the
weighted average of every prediction for this step, w_i = exp(−m·i) where **i = 0 is the OLDEST prediction**: the
official ACT code (imitate_episodes.py) stacks the chunks in all_time_actions by the step they were issued at and
weights actions_for_curr_step with exp(−k·arange(n)), k = 0.01, so row 0, the oldest chunk, gets the largest
weight; LeRobot's ACTTemporalEnsembler says "w₀ is the oldest action", "positive gives more weight to older actions",
"negative gives more weight to newer actions". The mean age is L + Δt · Σ w_i (n − 1 − i) / Σ w_i (with s = 1, n = H;
with s > 1 averaged over the step's place in the cycle). The reaction distance v × that age is how far the hand
travels before an action that has seen a change at the contact (the object not where earlier frames put it) lands
(along a trajectory, |P(t_c) − P(t_c − age)|): it bounds the overshoot when the contact comes sooner than predicted
and the air-grasp when it comes later.

**Recommendations** (what Apply stores on the policy, Policy.recommendations, one undo step; the result and the
page's export carry them; writing them into a LeRobot training config is the exporters' work, §9.2, not built yet):
- the proprioceptive delay offset δ that centres the close-timing interval of the policy's cameras on contact,
  rounded to whole control steps (delta_timestamps are multiples of 1/fps; the dataset is taken to be recorded at
  the control rate), with the predicted error before and after;
- for an ensembling policy, the ensemble weight m (ACT's convention) that averages the most predictions (the largest
  n_eff = (Σw)² / Σw², a measure of smoothing) while the reaction distance stays within reaction_tolerance_mm
  (default 10 mm); when no m in m_range (default −1 to 1) does, the most averaging within 10 % of the shortest
  reaction reachable, and the advice to cut the latency or the approach speed. A policy
  that executes its chunks open loop gets no m (its source says it uses none, or does not say).

**The approach** comes from the project's joint trajectories: every closing stroke of the gripper (core/grasps.py
closure_events on pad_opening) is a contact at its onset t_c, with the tool centre's path P(t) (forward kinematics at
the samples, linear between them) and its speed |J(q) q̇| (RobotModel.jacobian at the tool centre, q̇ from the
samples) averaged over approach_window_s (default 0.25 s) before the onset, v of the fastest stroke. Without a
trajectory that closes the gripper, or when the user sets approach_speed_m_s with its source: a constant speed v
(default 0.10 m/s, a default, not a measurement). **The overshoot map**: every trajectory sample t_k as if the
fingers closed there (|P(t_k + e) − P(t_k)|) and the grasp sets' grasp centres (at v), in cells of cell_m (default
3 cm) over the task volumes, each cell the largest overshoot and air-grasp of its points.

Status: the worst policy's larger of overshoot and air-grasp against tolerance_mm (default 5 mm): good up to it,
warning up to 2×, serious up to 4×, critical beyond (defaults, not standards).

Validation (validation/checks_chunk.py): chunk.loop_vs_simulation runs an event-by-event simulation of a moving point
seen by a camera with the same exposure, readout, transfer, independent camera, policy and control clocks, chunk
indexing and command latency, and compares the closing position's mean and range with this model;
chunk.ensemble_closed_form compares ActEnsembler on a point moving at constant speed, and ensemble_mean_age, with the
geometric-series closed form of the weighted average.

Known limits: the policy is an ideal imitator, so its learned errors (and the smoothing ensembling buys against
them) are not modelled, only timing; arm and gripper commands are taken to reach the hardware with the same latency,
and the arm to follow the demonstrated motion exactly (servo tracking lag and the gripper's own closing time are not
added: enter them in command_ms); the clocks are independent and uniform phases are assumed (a hardware-triggered rig
whose policy loop is locked to the camera has a narrower interval); the time-sync study's clock offsets and jitter
are not added here; transfer assumes the bandwidth left to a stream is used at its full rate; the trajectory is held
at its first and last sample beyond its ends; grasp centres in the map use one speed v; a policy that
waits for a whole chunk before inferring again (synchronous LeRobot) pauses the arm, which is not modelled.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .. import jobs
from ..jobs import StudyError

STUDY_ID = 'action_chunk_sync'
DEFAULTS = {
    'policies': [],                  # policy uids; [] = every policy of the project
    'approach_speed_m_s': 0.10,      # default when no trajectory closes the gripper (not a measurement)
    'approach_speed_source': '',     # where the user's approach speed comes from
    'approach_window_s': 0.25,       # default: the window before a closing stroke the speed is averaged over
    'command_ms': None,              # None: one control period (a default)
    'command_source': '',
    'command_rec_ms': None,          # None: the same as the deployment's command latency (a default)
    'queue_extra_ms': 0.0,           # default: no queueing beyond the newest-frame wait
    'image_stamp': None,             # None: 'host_receive' (default), 'mid_exposure' on a camera hardware clock
    'skip_steps': 0,                 # default: the executor starts a chunk at its first action (LeRobot)
    'tolerance_mm': 5.0,             # default: the error at contact that reads 'good'
    'reaction_tolerance_mm': 10.0,   # default: the reaction distance the recommended m keeps within
    'cell_m': 0.03,                  # default: overshoot map cell
    'm_range': [-1.0, 1.0],          # default: the ensemble weights searched
    'm_step': 0.005,
    'figures': {},                   # sensor uid -> {'exposure_ms' | 'readout_ms' | 'transfer_ms': {value, source}}
    'delay_offset_s': {},            # policy uid -> the training config's current offset (default 0)
}
LIMITS = {'approach_speed_m_s': (1e-6, 20.0), 'approach_window_s': (0.001, 10.0), 'queue_extra_ms': (0.0, 10000.0),
          'skip_steps': (0, 1000), 'tolerance_mm': (0.01, 1000.0),
          'reaction_tolerance_mm': (0.01, 10000.0), 'cell_m': (0.005, 1.0), 'm_step': (1e-4, 1.0)}
STAMPS = ('host_receive', 'mid_exposure')
WORDS = {'good': 'on contact', 'warning': 'slightly off', 'serious': 'off contact', 'critical': 'misses contact'}
REF_ACT_CODE = ('ACT official code, imitate_episodes.py (temporal_agg: all_time_actions, actions_for_curr_step, '
                'exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step))), k = 0.01), '
                'https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/imitate_episodes.py, '
                'read 2026-11-07')
REF_LEROBOT_ENSEMBLER = ('LeRobot ACTTemporalEnsembler docstring: "wᵢ = exp(-temporal_ensemble_coeff * i) where w₀ is '
                         'the oldest action"; positive gives more weight to older actions, negative to newer ones, '
                         'https://github.com/huggingface/lerobot/blob/d490e472a5d1a207cfe0255caaaf74b6d66e4656/src/'
                         'lerobot/policies/act/modeling_act.py, read 2026-11-07')
REF_ACT_PAPER = ('T. Z. Zhao, V. Kumar, S. Levine, C. Finn, "Learning Fine-Grained Bimanual Manipulation with '
                 'Low-Cost Hardware" (ACT), RSS 2023, https://arxiv.org/abs/2304.13705')
TERM_TITLES = {'exposure_half': 'Half the exposure', 'readout': 'Readout', 'transfer': 'Transfer',
               'queue': 'Host queueing', 'inference': 'Policy inference', 'tick': 'Wait for the control tick',
               'command': 'Command latency'}


# ====================================================================== settings
def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def settings_of(project) -> dict:
    """The settings with the defaults filled in; 'defaults' lists the keys left at their default."""
    given = dict(project.analysis.studies.get(STUDY_ID) or {})
    out = dict(DEFAULTS)
    out['defaults'] = [k for k in DEFAULTS if k not in given or given.get(k) is None]
    for k, v in given.items():
        if v is not None or k not in DEFAULTS:
            out[k] = v
    for k, (lo, hi) in LIMITS.items():
        v = _f(out[k])
        if v is None or v < lo or v > hi:
            raise StudyError(f'The action-chunk setting {k} must be a number from {lo:g} to {hi:g}; got {out[k]!r}.')
        out[k] = int(v) if k == 'skip_steps' else v
    for k in ('command_ms', 'command_rec_ms'):
        if out[k] is not None:
            v = _f(out[k])
            if v is None or v < 0 or v > 10000:
                raise StudyError(f'The {k.replace("_", " ")} must be 0 to 10000 ms; got {out[k]!r}.')
            out[k] = v
    if out['image_stamp'] is not None and out['image_stamp'] not in STAMPS:
        raise StudyError(f'The image timestamp must be one of {", ".join(STAMPS)}; got "{out["image_stamp"]}".')
    try:
        lo, hi = (float(v) for v in out['m_range'])
    except (TypeError, ValueError):
        raise StudyError('The ensemble weight range must be two numbers (low, high).') from None
    if not (lo < hi) or abs(lo) > 50 or abs(hi) > 50:
        raise StudyError('The ensemble weight range must run from a lower to a higher number within ±50.')
    out['m_range'] = [lo, hi]
    out['figures'] = dict(out.get('figures') or {})
    out['delay_offset_s'] = dict(out.get('delay_offset_s') or {})
    out['policies'] = [str(x) for x in out.get('policies') or []]
    return out


# ====================================================================== temporal ensembling (ACT's convention)
def ensemble_weights(m: float, n: int) -> np.ndarray:
    """ACT's weights w_i = exp(−m·i), i = 0 … n − 1, where i = 0 is the OLDEST prediction (module docstring)."""
    return np.exp(-float(m) * np.arange(int(n), dtype=float))


def _cycle(H: int, s: int):
    """[(phase p, n predictions covering the step)] for chunks every s steps of horizon H."""
    H, s = int(H), max(int(s), 1)
    return [(p, (H - 1 - p) // s + 1) for p in range(min(s, H))]


def ensemble_mean_age(m: float, H: int, s: int = 1) -> float:
    """The weighted mean age, in control steps since the newest chunk's image, of the ensembled action: chunks every
    s steps, the predictions for a step from n chunks of ages p + j·s (j = 0 the newest), weighted by ACT's
    exp(−m·i) with i = 0 the oldest; averaged over the step's place p in the cycle."""
    out = []
    for p, n in _cycle(H, s):
        w = ensemble_weights(m, n)
        age = p + (n - 1 - np.arange(n)) * s            # i = 0 is the oldest: age p + (n − 1)·s
        out.append(float((w * age).sum() / w.sum()))
    return float(np.mean(out))


def ensemble_n_eff(m: float, H: int, s: int = 1) -> float:
    """The effective number of predictions the ensemble averages, (Σw)² / Σw², averaged over the cycle."""
    out = []
    for _p, n in _cycle(H, s):
        w = ensemble_weights(m, n)
        out.append(float(w.sum() ** 2 / (w * w).sum()))
    return float(np.mean(out))


class ActEnsembler:
    """ACT's temporal aggregation as the official code does it: chunk t (H actions for steps t … t + H − 1) is written
    into row t of all_time_actions; the action for step t is the average of column t over the rows that hold it,
    weighted exp(−m·arange(n)) after normalising, row order = issue order (the oldest first)."""

    def __init__(self, H: int, m: float):
        self.H, self.m = int(H), float(m)
        self.rows: dict = {}                          # issue step -> (H, dim) chunk

    def add(self, t: int, chunk):
        self.rows[int(t)] = np.asarray(chunk, float).reshape(self.H, -1)

    def action(self, t: int) -> np.ndarray:
        t = int(t)
        col = [self.rows[r][t - r] for r in sorted(self.rows) if 0 <= t - r < self.H]
        if not col:
            raise ValueError(f'No chunk covers step {t}.')
        a = np.stack(col)
        w = np.exp(-self.m * np.arange(len(a)))
        w = w / w.sum()
        for r in [r for r in self.rows if t - r >= self.H - 1]:     # a chunk that no later step needs
            del self.rows[r]
        return (a * w[:, None]).sum(axis=0)


# ====================================================================== library figures
def _figure(v):
    """(seconds or None, source text) of a library figure written in seconds (number or {value, source})."""
    if isinstance(v, dict):
        x = _f(v.get('value'))
        return x, str(v.get('source') or '')
    return _f(v), ''


def camera_timing(entry, imager_name: str, key: str):
    """(seconds or None, source) of 'exposure' or 'readout' in the library entry: the imager's <key>_s or <key>_ms,
    else the entry's 'timing' block; None when the entry does not state it."""
    if entry is None:
        return None, 'no device chosen'
    data = entry.data
    blocks = [im for im in data.get('imagers') or [] if im.get('name') == imager_name] + [data.get('timing') or {}]
    for b in blocks:
        for k, scale in ((f'{key}_s', 1.0), (f'{key}_ms', 1e-3)):
            if b.get(k) is not None:
                x, src = _figure(b[k])
                if x is not None:
                    srcs = entry.sources
                    cite = (f'{srcs[0].get("title") or srcs[0].get("url")}, read {srcs[0].get("read")}' if srcs
                            else entry.title)
                    return x * scale, src or f'{entry.title} library entry ({cite})'
    return None, 'not stated by the maker'


def _user_ms(st, uid, key):
    f = (st['figures'].get(uid) or {}).get(key)
    if isinstance(f, dict) and _f(f.get('value')) is not None and _f(f.get('value')) >= 0:
        return _f(f['value']) / 1000.0, f'your figure: {f.get("source") or "no source given"}'
    return None, ''


def _blur_figure(project, uid, key):
    """(seconds or None, source) of your exposure or readout for a sensor as the motion blur study stores it
    (project.analysis.studies['motion_blur'][key][uid] = {'value' (s), 'source'}), so one figure serves both studies."""
    f = ((project.analysis.studies.get('motion_blur') or {}).get(key) or {}).get(uid)
    if isinstance(f, dict) and _f(f.get('value')) is not None and _f(f.get('value')) > 0:
        return _f(f['value']), f'your figure (motion blur page): {f.get("source") or "no source given"}'
    return None, ''


def _term(key, lo, hi, origin, source, value=None):
    mid = None if lo is None else (value if value is not None else 0.5 * (lo + hi))
    return {'key': key, 'title': TERM_TITLES.get(key, key), 'value_s': mid, 'lo_s': lo, 'hi_s': hi, 'origin': origin,
            'source': source}


def _stream_of(row, imager_name):
    streams = [s for s in row.get('streams') or [] if s.get('width')]
    if not streams:
        return None
    norm = lambda t: ''.join(c for c in str(t).lower() if c.isalnum())   # noqa: E731
    by = [s for s in streams if norm(s['imager']) == norm(imager_name)]
    if not by:
        by = [s for s in streams if any(w in norm(s['imager']) for w in ('rgb', 'color', 'colour'))]
    return (by or streams)[0]


def transfer_term(row, crow_by, imager_name):
    """The τ_x term of one sensor from the bandwidth study's row (module docstring)."""
    if row is None:
        return _term('transfer', None, None, 'not_stated', 'the sensor is not in the bandwidth budget'), None
    s = _stream_of(row, imager_name)
    if s is None or s.get('bits_per_pixel') is None:
        why = 'the stream\'s bits per pixel are not stated, so its frame size is not known'
        return _term('transfer', None, None, 'not_stated', why), s
    frame_bits = s['width'] * s['height'] * s['bits_per_pixel'] * (1.0 + row['overhead'])
    stream_wire = (s['payload_mbps'] or 0.0) * (1.0 + row['overhead'])
    c = crow_by.get(row.get('controller_uid'))
    if c is None or row.get('link_capacity_mbps') is None:
        why = ('the sensor is not plugged into a controller (bandwidth page)' if c is None else
               f'the link capacity is not stated ({row.get("link_capacity_source") or "enter it"})')
        return _term('transfer', None, None, 'not_stated', why), s
    own_others = max((row['wire_mbps'] or 0.0) - stream_wire, 0.0)
    avail_link = row['link_capacity_mbps'] - own_others
    avail = avail_link
    if c.get('capacity_mbps'):
        avail = min(avail, c['capacity_mbps'] - (c['load_mbps'] - stream_wire))
    if avail <= stream_wire * (1 + 1e-9):
        return _term('transfer', None, None, 'not_stated',
                     f'{c["label"]} is over budget: frames back up without bound (bandwidth page)'), s
    t = frame_bits / (avail * 1e6)
    return _term('transfer', t, t, 'study',
                 f'{s["width"]} × {s["height"]} × {s["bits_per_pixel"]} bit × (1 + {row["overhead"]:g} overhead) '
                 f'at the {avail:.0f} Mbit/s left to it on {c["label"]} (bandwidth study; '
                 f'{row["link_capacity_source"]})'), s


# ====================================================================== the loop
def policy_scheme(project, pol) -> dict:
    """The policy's timing figures (core/policies.resolve) and its execution scheme; StudyError says what is missing."""
    from .. import policies as PO
    r = PO.resolve_detail(project, pol.uid)
    name = pol.name

    def need(k, what):
        v = r[k]['value']
        if v is None:
            raise StudyError(f'{name}: its {what} is not stated by its source ({PO.title_of(pol)}), so its loop '
                             'cannot be timed: enter a measured figure on the Policies page.')
        return v
    tau_i = need('inference_ms', 'inference time') / 1000.0
    hz = need('control_hz', 'control rate')
    H = int(need('action_chunk', 'action chunk H'))
    m = r['temporal_ensemble_m']['value']
    rec = pol.recommendations or {}
    m_src = r['temporal_ensemble_m']['source']
    if isinstance(rec.get('temporal_ensemble_m'), dict) and _f(rec['temporal_ensemble_m'].get('value')) is not None:
        m, m_src = float(rec['temporal_ensemble_m']['value']), 'the applied recommendation (Policy.recommendations)'
    E = r['execute_steps']['value']
    notes = []
    numeric_m = m is not None and m is not False and _f(m) is not None
    if E is None:
        if not numeric_m:
            raise StudyError(f'{name}: how many actions it executes per chunk is not stated by its source, so its '
                             'reaction cannot be timed: enter "Executed per chunk" on the Policies page.')
        E = 1
        notes.append(f'{name}: executed per chunk is not stated; with temporal ensembling the official ACT code '
                     'queries every step (query_frequency = 1), so 1 is used.')
    E = int(E)
    ensemble = numeric_m and E == 1
    if numeric_m and E != 1:
        notes.append(f'{name}: temporal ensembling needs a new chunk every step and this policy executes {E}; it is '
                     'modelled open loop.')
    dt = 1.0 / float(hz)
    s = max(E, int(math.ceil(tau_i / dt - 1e-9)))
    if s > H:
        notes.append(f'{name}: a chunk of {H} actions runs out before the next one is ready ({s} steps): the arm '
                     'stops between chunks, which this model does not include.')
    return {'tau_i': tau_i, 'tau_i_source': r['inference_ms']['source'], 'hardware': r['inference_ms'].get('hardware'),
            'hz': float(hz), 'hz_source': r['control_hz']['source'], 'dt': dt, 'H': H, 'E': E, 's': s,
            'ensemble': ensemble, 'm': float(m) if numeric_m else (False if m is False else None), 'm_source': m_src,
            'H_source': r['action_chunk']['source'], 'E_source': r['execute_steps']['source'], 'notes': notes}


def _stamp_of(project, sensor_uid, st) -> tuple[str, str]:
    if st.get('image_stamp'):
        return st['image_stamp'], 'your setting'
    sync = (getattr(project, 'rig', None) or {}).get('sync') or {}
    sm = (sync.get('streams') or {}).get(sensor_uid) or {}
    clocks = {c.get('uid'): c for c in sync.get('clocks') or [] if isinstance(c, dict)}
    ck = clocks.get(sm.get('clock')) or {}
    if ck.get('kind') == 'camera_hw':
        name = ck.get('name') or 'a camera hardware clock'
        return 'mid_exposure', (f'the time-sync settings put this stream on {name}; its timestamps are taken to '
                                'be at mid-exposure (an assumption: check the maker\'s convention)')
    return 'host_receive', 'default: stamped on arrival at the host'


def loop_terms(project, sensor, imager_name, sch: dict, st: dict, bw_row, crow_by) -> dict:
    """The deployment loop's terms for one camera of a policy (module docstring)."""
    from .. import library as LB
    entry = LB.get(sensor.model_id) if sensor.model_id else None
    terms = []
    for key, tkey in (('exposure', 'exposure_half'), ('readout', 'readout')):
        x, src = _user_ms(st, sensor.uid, f'{key}_ms')
        origin = 'user'
        if x is None:                                   # the figure set on the motion blur page (shared with time sync)
            x, src = _blur_figure(project, sensor.uid, f'{key}_s')
        if x is None:
            x, src = camera_timing(entry, imager_name, key)
            origin = 'library' if x is not None else 'not_stated'
        if x is not None and key == 'exposure':
            terms.append(_term(tkey, x / 2, x / 2, origin, f'half of {x * 1000:.3g} ms: {src}'))
        else:
            terms.append(_term(tkey, x, x, origin, src))
    x, src = _user_ms(st, sensor.uid, 'transfer_ms')
    stream = None
    if x is not None:
        terms.append(_term('transfer', x, x, 'user', src))
        stream = _stream_of(bw_row, imager_name) if bw_row else None
    else:
        t, stream = transfer_term(bw_row, crow_by, imager_name)
        terms.append(t)
    fps = _f((stream or {}).get('fps'))
    extra = st['queue_extra_ms'] / 1000.0
    if fps:
        q = _term('queue', extra, extra + 1.0 / fps, 'default',
                  f'default model: the newest frame is 0 to 1/fps = {1000 / fps:.3g} ms old when the policy starts '
                  f'(independent clocks) at {fps:g} fps' + (f', plus {extra * 1000:g} ms' if extra else ''))
    else:
        q = _term('queue', None, None, 'not_stated', 'the camera\'s frame rate is not known (no stream with a rate)')
    terms.append(q)
    terms.append(_term('inference', sch['tau_i'], sch['tau_i'], 'policy', sch['tau_i_source']))
    terms.append(_term('tick', 0.0, sch['dt'], 'default', f'default model: 0 to one control period '
                       f'({sch["dt"] * 1000:.3g} ms at {sch["hz"]:g} Hz) until the next control tick'))
    if st['command_ms'] is not None:
        c = st['command_ms'] / 1000.0
        terms.append(_term('command', c, c, 'user', f'your figure: {st.get("command_source") or "no source given"}'))
    else:
        terms.append(_term('command', sch['dt'], sch['dt'], 'default', 'default: one control period'))
    return {'terms': terms, 'stream': stream, 'fps': fps,
            'missing': [t['title'] for t in terms if t['lo_s'] is None]}


def _sum(terms, keys):
    sel = [t for t in terms if t['key'] in keys and t['lo_s'] is not None]
    return (sum(t['lo_s'] for t in sel), sum(t['value_s'] for t in sel), sum(t['hi_s'] for t in sel))


def timing_error(lt: dict, sch: dict, st: dict, stamp: str, delta: float) -> dict:
    """The close-timing error e (s): low, nominal, high, and the reaction ages (module docstring)."""
    terms = lt['terms']
    by = {t['key']: t for t in terms}
    img = _sum(terms, ('exposure_half', 'readout', 'transfer'))
    age_stamp = img if stamp == 'host_receive' else (0.0, 0.0, 0.0)
    c = by['command']['value_s']
    c_rec = c if st['command_rec_ms'] is None else st['command_rec_ms'] / 1000.0
    k = st['skip_steps'] * sch['dt']
    rest = _sum(terms, ('queue', 'inference', 'tick'))
    dt = sch['dt']
    e = tuple(img[i] - age_stamp[i] + rest[i] + (c - c_rec) - delta - k for i in range(3))
    e = (e[0] - dt / 2, e[1], e[2] + dt / 2)                       # ρ: the close lands on a whole step
    L = _sum(terms, tuple(TERM_TITLES))
    missing_sys = [by[k]['title'] for k in ('queue',) if by[k]['lo_s'] is None]
    if stamp == 'mid_exposure':
        missing_sys += [by[k]['title'] for k in ('exposure_half', 'readout', 'transfer') if by[k]['lo_s'] is None]
    widths = [t['hi_s'] - t['lo_s'] for t in terms if t['key'] in ('queue', 'tick') and t['lo_s'] is not None] + [dt]
    sd = math.sqrt(sum(w * w for w in widths) / 12.0)              # independent uniform phases
    return {'e_s': e, 'e_sd_s': sd, 'loop_s': L, 'c_rec_s': c_rec, 'stamp_age_s': age_stamp[1],
            'missing_for_timing': missing_sys}


def reaction_age(L, sch, m=None) -> tuple:
    """(mean, high) age in seconds of the information behind an executed action (module docstring)."""
    dt, s = sch['dt'], sch['s']
    if sch['ensemble'] or m is not None:
        a = ensemble_mean_age(sch['m'] if m is None else m, sch['H'], s) * dt
        return L[1] + a, L[2] + a
    return L[1] + (s - 1) * dt / 2, L[2] + (s - 1) * dt


# ====================================================================== the approach speed and the map
def _tcp_samples(project, tr):
    from .. import grasps as GK
    from ..robot import trajectory as TR
    from ..robot.model import get_model
    model = get_model(project, tr.get('robot_uid', ''))
    geom = GK.gripper_geometry(project, tr.get('robot_uid', ''))
    ts = np.asarray([float(t) for t in tr.get('t') or []])
    Q = np.array([model.clip(TR.on_model(tr, model.joint_names, model.home, t)) for t in ts])
    link, tcp = geom['frame'], np.asarray(geom['tcp'], float)
    i = model.link_index(link)
    T = model.fk_batch(Q)[:, i]
    P = T[:, :3, :3] @ tcp + T[:, :3, 3]
    Qd = np.gradient(Q, ts, axis=0) if len(ts) > 1 else np.zeros_like(Q)
    V = np.array([model.jacobian(Q[k], link, tcp)[:3] @ Qd[k] for k in range(len(ts))])
    return ts, Q, P, V


def approach_speed(project, st: dict, notes: list) -> dict:
    """{'v', 'origin', 'source', 'events': [...], 'points': [(p, speed)]} (module docstring)."""
    from .. import grasps as GK
    events, points, tracks = [], [], []
    for tr in project.tasks.trajectories or []:
        try:
            ts, Q, P, V = _tcp_samples(project, tr)
        except Exception as e:  # noqa: BLE001 — a trajectory that cannot be posed is said and left out
            notes.append(f'The trajectory {tr.get("name", "")} is left out: {e}')
            continue
        sp = np.linalg.norm(V, axis=1)
        tracks.append((ts, P))
        points += [(P[k], float(sp[k]), len(tracks) - 1, float(ts[k])) for k in range(len(ts))]
        try:
            ev = GK.closure_events(GK.pad_opening(project, tr.get('robot_uid', ''), Q))
        except GK.GraspError:
            continue
        for k0, _k1, _drop in ev:
            win = (ts <= ts[k0]) & (ts >= ts[k0] - st['approach_window_s'])
            v = float(sp[win].mean()) if win.any() else float(sp[k0])
            events.append({'trajectory': tr.get('name', ''), 't': float(ts[k0]), 'speed_m_s': v,
                           'position': [float(x) for x in P[k0]], 'track': len(tracks) - 1})
    user = 'approach_speed_m_s' not in st['defaults']
    if user:
        out = {'v': st['approach_speed_m_s'], 'origin': 'user',
               'source': f'your figure: {st.get("approach_speed_source") or "no source given"}'}
    elif events:
        best = max(events, key=lambda e: e['speed_m_s'])
        out = {'v': best['speed_m_s'], 'origin': 'trajectory',
               'source': f'the fastest of {len(events)} closing stroke{"s" if len(events) != 1 else ""}: '
                         f'{best["trajectory"]} at {best["t"]:.2f} s, the tool centre\'s mean speed over the '
                         f'{st["approach_window_s"]:g} s before it (|J(q) q̇|)'}
    else:
        out = {'v': st['approach_speed_m_s'], 'origin': 'default',
               'source': f'default {st["approach_speed_m_s"]:g} m/s, not a measurement: no trajectory closes the '
                         'gripper; set your demonstrations\' approach speed'}
    gpts = []
    for gs in GK.sets(project):
        try:
            _link, _tcp, T = GK.ik_targets(project, gs)
        except Exception:  # noqa: BLE001 — a set without its robot is left out of the map
            continue
        gpts += [(T[k, :3, 3], None) for k in range(len(T))]
    out.update(events=events, points=points, grasp_points=gpts, tracks=tracks,
               along=bool(events) and out['origin'] == 'trajectory')
    return out


def displacement(ts, P, t0: float, dt: float) -> float:
    """|P(t0 + dt) − P(t0)| along a sampled path (linear between samples, held beyond its ends)."""
    def at(t):
        return np.array([np.interp(t, ts, P[:, i]) for i in range(3)])
    return float(np.linalg.norm(at(t0 + dt) - at(t0)))


def spatial(sp: dict, e: float) -> float:
    """How far (m) the tool centre is from the contact point when the fingers close e seconds after (e > 0) or
    before (e < 0) contact: the worst closing stroke along its trajectory, else the constant speed v × |e|."""
    if sp.get('along'):
        return max(displacement(*sp['tracks'][ev['track']], ev['t'], e) for ev in sp['events'])
    return sp['v'] * abs(e)


def _bounds(project, pts):
    vols = project.workspace.volumes
    if vols:
        lo = np.min([np.asarray(v.center) - np.asarray(v.size) / 2 for v in vols], axis=0)
        hi = np.max([np.asarray(v.center) + np.asarray(v.size) / 2 for v in vols], axis=0)
        return lo, hi, True
    if not pts:
        return None, None, False
    P = np.array([p for p, _ in pts])
    return P.min(axis=0), P.max(axis=0) + 1e-9, False


def overshoot_map(project, sp: dict, e_lo: float, e_hi: float, cell: float) -> dict:
    """Cells over the task volumes: the largest overshoot and air-grasp (m) of the points in each."""
    pts = [(p, s) for p, s, _k, _t in sp['points']] + [(p, sp['v']) for p, _ in sp['grasp_points']]
    lo, hi, in_vol = _bounds(project, pts)
    if lo is None:
        return {'cells': [], 'cell_m': cell, 'points': 0, 'bounds': None}
    acc: dict = {}
    items = [(p, s, k, t) for p, s, k, t in sp['points']] + [(p, sp['v'], None, None) for p, _ in sp['grasp_points']]
    for p, s, k_, t in items:
        p = np.asarray(p, float)
        if in_vol and (np.any(p < lo - 1e-9) or np.any(p > hi + 1e-9)):
            continue
        key = tuple(np.floor((p - lo) / cell).astype(int))
        if k_ is not None:
            ts, P = sp['tracks'][k_]
            o = displacement(ts, P, t, e_hi) if e_hi > 0 else 0.0
            a = displacement(ts, P, t, e_lo) if e_lo < 0 else 0.0
        else:
            o, a = s * max(0.0, e_hi), s * max(0.0, -e_lo)
        c = acc.setdefault(key, [0.0, 0.0, 0.0, 0])
        c[0], c[1], c[2], c[3] = max(c[0], o), max(c[1], a), max(c[2], s), c[3] + 1
    cells = [{'center': [float(x) for x in lo + (np.asarray(k) + 0.5) * cell], 'overshoot_m': v[0],
              'air_grasp_m': v[1], 'speed_m_s': v[2], 'n': v[3]} for k, v in sorted(acc.items())]
    return {'cells': cells, 'cell_m': cell, 'points': int(sum(c['n'] for c in cells)),
            'bounds': [[float(x) for x in lo], [float(x) for x in hi]]}


# ====================================================================== the study
def status_for(err_mm: float, st: dict) -> str:
    t = st['tolerance_mm']
    if err_mm <= t + 1e-12:
        return 'good'
    if err_mm <= 2 * t:
        return 'warning'
    if err_mm <= 4 * t:
        return 'serious'
    return 'critical'


def _r(x, nd=6):
    return None if x is None or not math.isfinite(float(x)) else round(float(x), nd)


def recommend_m(sch, L, sp, tol_m, st, m_now) -> dict | None:
    """The ensemble weight m (module docstring) and the predicted change, for an ensembling policy."""
    if not sch['ensemble']:
        return None
    lo, hi = st['m_range']
    grid = np.unique(np.r_[np.arange(lo, hi + 1e-12, st['m_step']), m_now])
    reach = np.array([spatial(sp, -reaction_age(L, sch, m)[0]) for m in grid])
    neff = np.array([ensemble_n_eff(m, sch['H'], sch['s']) for m in grid])
    met = bool((reach <= tol_m + 1e-12).any())
    ok = reach <= (tol_m + 1e-12 if met else 1.1 * reach.min() + 1e-12)
    k = int(np.flatnonzero(ok)[np.argmax(neff[ok])])
    k_now = int(np.argmin(np.abs(grid - m_now)))
    step = max(1, len(grid) // 200)
    return {'m': _r(grid[k]), 'met': met, 'from_m': _r(m_now),
            'reaction_m': _r(reach[k]), 'from_reaction_m': _r(reach[k_now]),
            'n_eff': _r(neff[k], 3), 'from_n_eff': _r(neff[k_now], 3),
            'curve': {'m': [_r(x) for x in grid[::step]], 'reaction_m': [_r(x) for x in reach[::step]],
                      'n_eff': [_r(x, 3) for x in neff[::step]]}}


def _policy_result(project, pol, st, bw, crow_by, sp, notes) -> dict:  # noqa: C901 — the loop, term by term
    from .. import library as LB
    from .. import policies as PO
    from .coverage import pick_imager, sensor_mode
    sch = policy_scheme(project, pol)
    notes += sch['notes']
    cams = [s for s in PO.policy_cameras(project, pol) if s.enabled and s.kind in ('camera', 'depth')]
    if not cams:
        raise StudyError(f'{pol.name} takes no camera of the active rig that is in use: choose its cameras on the '
                         'Policies page.')
    rows_by = {r['uid']: r for r in bw.get('sensors') or []}
    rec = pol.recommendations or {}
    if isinstance(rec.get('delay_offset_s'), dict) and _f(rec['delay_offset_s'].get('value')) is not None:
        delta, d_src = float(rec['delay_offset_s']['value']), 'the applied recommendation (Policy.recommendations)'
    elif _f(st['delay_offset_s'].get(pol.uid)) is not None:
        delta, d_src = float(st['delay_offset_s'][pol.uid]), 'your setting'
    else:
        delta, d_src = 0.0, 'default 0: LeRobot\'s delta_timestamps start at 0'
    cam_rows = []
    for s in cams:
        try:
            ims = LB.sensor_imagers(s, sensor_mode(s, {}))
            im, _why = pick_imager(s, ims, {})
            iname = im.name
        except (LB.LibraryError, StudyError, IndexError):
            iname = ''
        lt = loop_terms(project, s, iname, sch, st, rows_by.get(s.uid), crow_by)
        stamp, stamp_src = _stamp_of(project, s.uid, st)
        te = timing_error(lt, sch, st, stamp, delta)
        e = te['e_s']
        ra = reaction_age(te['loop_s'], sch)
        cam_rows.append({
            'uid': s.uid, 'name': s.name, 'imager': iname, 'slot': s.slot,
            'stream': None if lt['stream'] is None else lt['stream'].get('label'),
            'terms': lt['terms'], 'missing': lt['missing'], 'missing_for_timing': te['missing_for_timing'],
            'loop_s': [_r(x) for x in te['loop_s']], 'stamp': stamp, 'stamp_source': stamp_src,
            'stamp_age_s': _r(te['stamp_age_s']), 'command_rec_s': _r(te['c_rec_s']),
            'error_s': [_r(x) for x in e], 'error_sd_s': _r(te['e_sd_s']),
            'error_m': [_r(math.copysign(spatial(sp, x), x)) for x in e],
            'overshoot_m': _r(spatial(sp, e[2]) if e[2] > 0 else 0.0),
            'air_grasp_m': _r(spatial(sp, e[0]) if e[0] < 0 else 0.0),
            'reaction_s': [_r(x) for x in ra], 'reaction_m': [_r(spatial(sp, -x)) for x in ra]})
    e_lo = min(c['error_s'][0] for c in cam_rows)
    e_hi = max(c['error_s'][2] for c in cam_rows)
    worst = max(max(c['overshoot_m'], c['air_grasp_m']) for c in cam_rows)
    # the delay offset: centre [e_lo, e_hi] on contact, in whole control steps
    dt = sch['dt']
    centre = 0.5 * (e_lo + e_hi)
    d_new = round((delta + centre) / dt) * dt
    shift = d_new - delta
    lo2, hi2 = e_lo - shift, e_hi - shift
    worst2 = max(spatial(sp, hi2) if hi2 > 0 else 0.0, spatial(sp, lo2) if lo2 < 0 else 0.0)
    tol = st['reaction_tolerance_mm'] / 1000.0
    L_worst = max((c for c in cam_rows), key=lambda c: c['reaction_s'][0])
    Lw = tuple(x for x in L_worst['loop_s'])
    m_rec = recommend_m(sch, Lw, sp, tol, st, sch['m']) if sch['ensemble'] else None
    over = max(c['overshoot_m'] for c in cam_rows)
    air = max(c['air_grasp_m'] for c in cam_rows)
    kind = 'overshoot' if over >= air else 'air_grasp'
    missing = sorted({t for c in cam_rows for t in c['missing']})
    rec_doc = {
        'delay_offset_s': {'value': _r(d_new), 'from': _r(delta), 'from_source': d_src, 'steps': int(round(d_new / dt)),
                           'error_s': [_r(lo2), _r(0.5 * (lo2 + hi2)), _r(hi2)],
                           'from_error_s': [_r(e_lo), _r(centre), _r(e_hi)],
                           'worst_m': _r(worst2), 'from_worst_m': _r(worst), 'changes': abs(shift) > 1e-12},
        'temporal_ensemble_m': m_rec,
    }
    return {
        'uid': pol.uid, 'name': pol.name, 'library_id': pol.library_id, 'entry': PO.title_of(pol),
        'scheme': {k: sch[k] for k in ('H', 'E', 's', 'hz', 'dt', 'ensemble', 'm', 'tau_i')} | {
            'H_source': sch['H_source'], 'E_source': sch['E_source'], 'm_source': sch['m_source'],
            'hz_source': sch['hz_source'], 'inference_source': sch['tau_i_source'], 'hardware': sch['hardware'],
            'mean_age_steps': _r(ensemble_mean_age(sch['m'], sch['H'], sch['s']) if sch['ensemble']
                                 else (sch['s'] - 1) / 2, 4),
            'n_eff': _r(ensemble_n_eff(sch['m'], sch['H'], sch['s']), 3) if sch['ensemble'] else 1.0},
        'delay_offset_s': _r(delta), 'delay_offset_source': d_src,
        'cameras': cam_rows, 'error_s': [_r(e_lo), _r(centre), _r(e_hi)],
        'overshoot_m': _r(over), 'air_grasp_m': _r(air), 'worst_m': _r(worst), 'worst_kind': kind,
        'reaction_m': L_worst['reaction_m'], 'missing': missing,
        'status': status_for(worst * 1000, st), 'recommendations': rec_doc,
        'map': overshoot_map(project, sp, e_lo, e_hi, st['cell_m']),
        'map_after': overshoot_map(project, sp, lo2, hi2, st['cell_m']),      # with the recommended offset (Preview)
    }


@jobs.study(STUDY_ID, 'Action-chunk latency', quick=False, batch='action-chunk', needs_robot=False)
def run(project, ctx: jobs.Ctx) -> dict:
    """The loop, the close-timing error and the recommendations per policy (module docstring)."""
    from . import bandwidth as BW
    t0 = time.perf_counter()
    st = settings_of(project)
    pols = [p for p in project.policies if not st['policies'] or p.uid in st['policies']]
    if not project.policies:
        raise StudyError('There is no policy in the project: add the policy the data is for on the Policies page '
                         '(ACT, Diffusion Policy, π0…).')
    if not pols:
        raise StudyError('None of the policies chosen for the action-chunk study is in the project any more.')
    notes: list = []
    ctx.report(0.05, 'The bandwidth budget')
    bw = BW.budget(project)
    crow_by = {c['uid']: c for c in bw.get('controllers') or []}
    ctx.report(0.2, 'The approach speed')
    sp = approach_speed(project, st, notes)
    rows, refused = [], []
    for k, pol in enumerate(pols):
        if ctx.cancelled():
            break
        ctx.report(0.3 + 0.65 * k / len(pols), pol.name)
        try:
            rows.append(_policy_result(project, pol, st, bw, crow_by, sp, notes))
        except StudyError as e:
            refused.append({'uid': pol.uid, 'name': pol.name, 'reason': str(e)})
    if not rows:
        raise StudyError(' '.join(r['reason'] for r in refused))
    worst = max(rows, key=lambda r: r['worst_m'])
    status = worst['status']
    word = WORDS[status] if status == 'good' else ('closes late' if worst['worst_kind'] == 'overshoot'
                                                   else 'closes early')
    rd = worst['recommendations']['delay_offset_s']
    if rd['from_worst_m'] < 5e-5 and sp['along']:
        e = worst['error_s']
        advice = (f'In the demonstrations the hand is still when the gripper closes, so {worst["name"]}\'s close '
                  f'({e[0] * 1000:+.0f} to {e[2] * 1000:+.0f} ms from contact) lands on the contact point. '
                  + (f'A delay offset of {rd["value"] * 1000:.0f} ms centres it on contact all the same.'
                     if rd['changes'] else ''))
    elif rd['changes']:
        advice = (f'Train {worst["name"]} with a proprioceptive delay offset of {rd["value"] * 1000:.0f} ms '
                  f'({rd["steps"]} control step{"s" if abs(rd["steps"]) != 1 else ""}; now '
                  f'{rd["from"] * 1000:.0f} ms): the worst error at contact goes from {rd["from_worst_m"] * 1000:.1f} '
                  f'to {rd["worst_m"] * 1000:.1f} mm' + (' along the demonstrated approach.' if sp['along'] else
                                                         f' at {sp["v"]:.3g} m/s.'))
    else:
        advice = (f'{worst["name"]}\'s delay offset ({rd["value"] * 1000:.0f} ms) already centres the close on '
                  'contact within one control step.')
    mr = worst['recommendations']['temporal_ensemble_m']
    if mr:
        advice += (f' Ensemble weight m = {mr["m"]:g} (now {mr["from_m"]:g}; ACT\'s convention, i = 0 the oldest): '
                   f'the reaction distance goes from {mr["from_reaction_m"] * 1000:.1f} to '
                   f'{mr["reaction_m"] * 1000:.1f} mm' + ('.' if mr['met'] else
                                                          f', still above {st["reaction_tolerance_mm"]:g} mm: '
                                                          'shorten the loop or slow the approach.'))
    if any(r['missing'] for r in rows):
        notes.append('Figures not stated (the loop is "at least" without them): ' + '; '.join(
            f'{r["name"]}: {", ".join(r["missing"])}' for r in rows if r['missing']) + '. Camera exposure and readout '
            'come from the library entry or your measured figure (study settings "figures").')
    return {
        'policies': rows, 'refused': refused,
        'approach': {**{k: sp[k] for k in ('v', 'origin', 'source', 'along')},
                     'events': [{k: v for k, v in ev.items() if k != 'track'} for ev in sp['events']]},
        'headline': {'worst_mm': _r(worst['worst_m'] * 1000, 3), 'policy_uid': worst['uid'], 'policy': worst['name'],
                     'kind': worst['worst_kind'], 'status': status, 'status_word': word,
                     'overshoot_mm': _r(worst['overshoot_m'] * 1000, 3),
                     'air_grasp_mm': _r(worst['air_grasp_m'] * 1000, 3),
                     'reaction_mm': _r(worst['reaction_m'][0] * 1000, 3)},
        'advice': advice,
        'settings': {k: v for k, v in st.items()},
        'references': [REF_ACT_PAPER, REF_ACT_CODE, REF_LEROBOT_ENSEMBLER],
        'notes': notes,
        'timing': {'seconds': _r(time.perf_counter() - t0, 4)},
    }


# ====================================================================== Apply
def recommendation_records(row: dict, date: str) -> dict:
    """The Policy.recommendations records of one policy's result row."""
    out = {}
    d = row['recommendations']['delay_offset_s']
    out['delay_offset_s'] = {'value': d['value'], 'unit': 's', 'steps': d['steps'],
                             'source': f'action-chunk study ({date}): centres the predicted close on contact',
                             'predicted_worst_m': d['worst_m'], 'from_worst_m': d['from_worst_m']}
    m = row['recommendations']['temporal_ensemble_m']
    if m:
        out['temporal_ensemble_m'] = {'value': m['m'], 'convention': 'w_i = exp(-m i), i = 0 the oldest (ACT)',
                                      'source': f'action-chunk study ({date}): the most averaging within the '
                                                'tolerance' if m['met'] else
                                      f'action-chunk study ({date}): the shortest reaction (tolerance not met)',
                                      'predicted_reaction_m': m['reaction_m'], 'from_reaction_m': m['from_reaction_m']}
    return out


def apply_recommendations(project, doc: dict, policy_uid: str | None = None) -> list:
    """Store the result's recommendations on the policies (every policy, or one); returns the names changed."""
    import datetime as _dt
    date = _dt.date.today().isoformat()
    names = []
    for row in doc.get('policies') or []:
        if policy_uid and row['uid'] != policy_uid:
            continue
        pol = next((p for p in project.policies if p.uid == row['uid']), None)
        if pol is None:
            continue
        pol.recommendations = {**(pol.recommendations or {}), **recommendation_records(row, date)}
        names.append(pol.name)
    return names
