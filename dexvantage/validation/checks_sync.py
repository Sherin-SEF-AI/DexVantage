"""Checks of the time sync study (core/studies/time_sync.py) against independent references.

- sync.stamp_interval_sim: (1) the USB frame timing the study reads off the bandwidth study's topology, against the
  frame sizes and the USB 3.2 Gen 1 capacity typed in again here (848 × 480 × 16 bit × 1.05 at 3600 Mbit/s; the
  wait behind one frame of every other stream on the controller); (2) the stamp-error interval of every camera of the
  UR5e sample under five configurations (free-running and stamped on arrival; the same with a fixed exposure; a
  hardware trigger; a device clock with stated offset, jitter and drift; a measured latency) against a discrete
  simulation of each frame: the rows exposed and read one by one, the frame queued behind a random set of other
  frames on the bus, the host seeing the transfer end at the next 125 µs bus interval from a random phase, a device
  clock read at a random instant of the exposure. Every simulated error must lie in the study's interval, and the
  simulated errors must reach across it (each term is drawn at its ends a third of the time).
- sync.action_label_sim: the action-label error (the largest end-effector displacement over the skew, from the
  study's 1 ms forward-kinematics path) against a discrete simulation: a 30 fps camera with random phases along the
  UR5e sample's scripted pick, each frame stamped with the skew, its label the TCP at the stamp and its truth the TCP
  at the exposure, both by forward kinematics at those instants.
"""
from __future__ import annotations

import numpy as np

from .suite import Outcome, check

BITS = 848 * 480 * 16 * 1.05          # one 848 × 480 16-bit frame with the 5 % overhead default, bits
GEN1_MBPS = 450 * 8.0                 # USB 3.2 §4.4.11: "450 MB/sec or less", typed in again
BUS_S = 125e-6                        # the bus interval (Microsoft Learn; USB 2.0 §5.8.4)


def _project(ctx):
    p = ctx.cache.get('sync.ur5e')
    if p is None:
        from ..core.project import rig_from_dict
        from ..core.robot.samples import sample_project
        p = sample_project('ur5e')
        ids = ['depth|848x480|30|Z16', 'RGB|848x480|30|YUY2']
        p.rig = rig_from_dict({'hosts': [{'uid': 'h1', 'name': 'Host', 'library_id': 'usb-3.2-gen1-host',
                                          'controllers': [{'uid': 'c1', 'name': 'USB 3', 'kind': 'usb3'}]}],
                               'hubs': [], 'links': [{'sensor_uid': s.uid, 'stream_ids': ids, 'via': 'c1'}
                                                     for s in p.variant.sensors]})
        ctx.cache['sync.ur5e'] = p
    return p


def _draw(rng, lo, hi, n):
    """n draws over [lo, hi]: a third at lo, a third at hi, a third uniform between."""
    k = rng.integers(0, 3, n)
    u = rng.uniform(lo, hi, n)
    return np.where(k == 0, lo, np.where(k == 1, hi, u))


def _simulate(rng, c, cfg, clock, n, transfer, st):
    """Simulated stamp errors (n,) of camera imager c under stream config cfg (module docstring)."""
    H = c['imager'].camera.height
    ex = c['exposure']
    if ex['kind'] == 'yours':
        E = np.full(n, ex['value'])
    else:
        E = _draw(rng, ex.get('stated_min') or 0.0, ex['value'], n)
    fr = c['frame_readout']
    Tr = _draw(rng, 0.0, fr['value'], n) if fr['kind'] == 'upper bound' else np.full(n, fr['value'])
    mid = (H - 1) / 2.0
    rows = np.array([np.floor(mid), np.ceil(mid)])
    out = np.empty(n)
    for i in range(n):
        if c['shutter'] == 'rolling':
            starts = rows * Tr[i] / (H - 1)
            true = float(np.mean(starts + E[i] / 2))           # the middle row's exposure centre
            last_read = (H - 1) * Tr[i] / (H - 1) + E[i]        # the last row is read at the end of its exposure
        else:
            true = E[i] / 2                                     # every row exposed together
            last_read = E[i] + Tr[i]                            # then read row by row
        trig = cfg.get('trigger')
        if trig == 'hardware':
            stamp = 0.0                                         # the trigger edge starts the first row's exposure
        elif clock is not None and clock['kind'] == 'camera_hw':
            at = rng.choice([true - E[i] / 2, true + E[i] / 2, rng.uniform(true - E[i] / 2, true + E[i] / 2)])
            stamp = float(at)
        elif st['latency_s'].get(c['sensor'].uid):
            stamp = true + st['latency_s'][c['sensor'].uid]['value']
        else:
            others = transfer['others_bits']
            ahead = others[rng.random(len(others)) < rng.choice([0.0, 1.0, 0.5])]
            done = last_read + BITS / (transfer['link_mbps'] * 1e6) + float(ahead.sum()) / (GEN1_MBPS * 1e6)
            phase = rng.choice([0.0, BUS_S * (1 - 1e-12), rng.uniform(0, BUS_S)])
            k = np.ceil((done - phase) / BUS_S - 1e-12)
            stamp = k * BUS_S + phase
            if stamp < done:
                stamp += BUS_S
        if clock is not None and clock['kind'] != 'host':
            off = clock['offset_s']['value']
            j = clock['jitter_s']['value']
            d = clock['drift_ppm']['value'] * 1e-6 * st['episode_s']
            stamp += off + float(_draw(rng, -j, j, 1)[0]) + float(_draw(rng, min(0.0, d), max(0.0, d), 1)[0])
        out[i] = stamp - true
    return out


@check('sync.stamp_interval_sim', 'timing', 'Stamp-error interval against a simulation of the frame timeline',
       'largest amount (s) by which a simulated stamp error falls outside the study\'s interval, or by which the '
       'study\'s USB transfer times differ from the frame sizes and capacity typed in again; the simulated errors '
       'must also span at least 90 % of each interval',
       'a frame simulated row by row (exposure and readout), queued behind a random set of other frames on a USB 3.2 '
       'Gen 1 bus (450 MB/s), seen at the next 125 µs bus interval; device clocks read at a random instant of the '
       'exposure with offset, jitter and drift drawn over their stated ranges', 1e-9, 's',
       refs=('usb32_spec', 'usb20_spec', 'ms_usb_isoch', 'liang2008'), cost='study')
def _stamp_sim(ctx):
    from ..core.studies import motion_blur as MB
    from ..core.studies import time_sync as TS
    p = _project(ctx)
    rng = ctx.rng('sync.stamp_interval_sim')
    cams = MB.imaging(p, MB.settings_of(p))
    xfer = TS.transfer_times(p, cams)
    worst, rows, ok = 0.0, [], True
    # (1) USB frame timing from the topology: two sensors, two 848 × 480 16-bit streams each, one Gen 1 controller
    for c in cams:
        x = xfer[c['key']]
        want_min = BITS / (GEN1_MBPS * 1e6)
        want_max = 4 * BITS / (GEN1_MBPS * 1e6)
        e = max(abs(x['min_s'] - want_min), abs((x['max_s'] or np.inf) - want_max))
        worst = max(worst, e)
        rows.append({'camera': c['name'], 'transfer_s': [x['min_s'], x['max_s']], 'typed_in_s': [want_min, want_max]})
    # (2) the stamp interval under five configurations
    n = 300 if ctx.quick else 2000
    hw = {'uid': 'k1', 'name': 'Camera clock', 'kind': 'camera_hw', 'offset_s': {'value': 0.0015, 'source': 'check'},
          'jitter_s': {'value': 0.0002, 'source': 'check'}, 'drift_ppm': {'value': -20.0, 'source': 'check'}}
    for label, st_extra, cfg, clock in (
            ('free, host arrival', {}, {'clock': '', 'trigger': 'free'}, None),
            ('fixed exposure 2 ms', {'exposure_s': 0.002}, {'clock': '', 'trigger': 'free'}, None),
            ('hardware trigger', {}, {'clock': '', 'trigger': 'hardware'}, None),
            ('device clock', {}, {'clock': 'k1', 'trigger': 'free'}, hw),
            ('measured latency 7 ms', {'latency': 0.007}, {'clock': '', 'trigger': 'free'}, None)):
        q = p.snapshot()
        if 'exposure_s' in st_extra:
            q.analysis.studies['motion_blur'] = {'exposure_s': {s.uid: {'value': st_extra['exposure_s'],
                                                                         'source': 'check'} for s in q.variant.sensors}}
        if 'latency' in st_extra:
            q.analysis.studies['time_sync'] = {'latency_s': {s.uid: {'value': st_extra['latency'], 'source': 'check'}
                                                             for s in q.variant.sensors}}
        st = TS.settings_of(q)
        ccams = MB.imaging(q, MB.settings_of(q))
        clocks = {hw['uid']: hw} if clock else {}
        for c in ccams:
            sup, _w = TS._trigger_support(c['sensor'], c['imager'].name)
            if cfg['trigger'] == 'hardware' and sup is not True:
                continue
            b = TS.stamp_budget(c, cfg, clocks, xfer[c['key']], st, sup)
            others = np.full(3, BITS)
            sim = _simulate(rng, c, cfg, clock, n, {'others_bits': others, 'link_mbps': GEN1_MBPS}, st)
            out = max(0.0, b.lo - float(sim.min()), float(sim.max()) - b.hi)
            width = b.hi - b.lo
            span = (float(sim.max()) - float(sim.min())) / width if width > 0 else 1.0
            worst = max(worst, out)
            ok = ok and span >= 0.9
            rows.append({'config': label, 'camera': c['name'], 'interval_s': [b.lo, b.hi],
                         'simulated_s': [float(sim.min()), float(sim.max())], 'outside_s': out,
                         'span': round(span, 4)})
    return Outcome(worst, {'frames_per_case': n, 'rows': rows}, ok=ok,
                   note='Both sides add the same physical terms, so a stamp outside the interval is a term the study '
                        'misses or a wrong sign; 10⁻⁹ s is rounding. The span condition shows the interval is not '
                        'wider than the timeline allows (each term is drawn at its ends a third of the time, so the '
                        'extremes are reached).')


@check('sync.action_label_sim', 'timing', 'Action-label error against a simulated moving end effector',
       'largest relative difference between the study\'s action-label error and the largest error in a simulation '
       'that stamps 30 fps frames with the skew',
       'frames at random phases along the UR5e sample\'s scripted pick; the label is the TCP at the stamp, the truth '
       'the TCP at the exposure, each by forward kinematics at that instant (no grid, no interpolation)', 0.02,
       'fraction', refs=('mujoco_todorov2012',), cost='study')
def _label_sim(ctx):
    from ..core import grasps as GR
    from ..core.robot import trajectory as TR
    from ..core.robot.model import get_model
    from ..core.studies import time_sync as TS
    p = _project(ctx)
    tr = p.tasks.trajectories[0]
    m = get_model(p, tr['robot_uid'])
    g = GR.gripper_geometry(p, tr['robot_uid'])
    tcp = np.r_[np.asarray(g['tcp'], float), 1.0]
    li = m.link_index(g['frame'])
    rng = ctx.rng('sync.action_label_sim')
    times, P = TS.ee_path(p, tr, TS.DEFAULTS['grid_s'])
    t0, t1 = TR.duration(tr)

    def tcp_at(t):
        q = TR.on_model(tr, m.joint_names, m.home, float(min(max(t, t0), t1)))
        return (m.fk_array(q)[li] @ tcp)[:3]

    phases = 6 if ctx.quick else 40
    worst, rows = 0.0, []
    for delta in (-0.020, -0.002, 0.003, 0.015, 0.040):
        pred, _t = TS.label_error(times, P, delta)
        sim = 0.0
        for _ in range(phases):
            f0 = t0 + float(rng.uniform(0, 1 / 30))
            for tk in np.arange(f0, t1, 1 / 30):
                sim = max(sim, float(np.linalg.norm(tcp_at(tk + delta) - tcp_at(tk))))
        err = abs(pred - sim) / pred
        worst = max(worst, err)
        rows.append({'skew_s': delta, 'study_m': pred, 'simulated_m': sim, 'error': err})
    return Outcome(worst, {'phases': phases, 'fps': 30, 'trajectory': tr['name'], 'rows': rows},
                   note='The study takes the largest displacement over every instant (a 1 ms path, linear between); '
                        'the simulation only sees the frames it samples, so it can fall a little short of the largest '
                        'displacement, and the path\'s chords differ from the arc by far less than a micrometre. 2 % '
                        'allows the sampling; a skew applied with the wrong sign or scale shows as tens of per cent.')
