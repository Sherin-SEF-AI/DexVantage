"""Time sync (spec §7.10): how far each camera's frame timestamps are from the instant the frame was exposed, against
the robot's joint_states, and what that does to the action labels of a dataset: the end-effector displacement during
the skew at the trajectory's speed.

    @jobs.study('time_sync', 'Time sync', batch='time-sync') → --time-sync OUT.json
    run(project, ctx) -> dict              the study (also core.jobs.run_study('time_sync', project))
    settings_of(project) -> dict           settings with their defaults (DEFAULTS) and which are defaults
    sync_config(project) -> dict           project.rig['sync'] normalised (core/project.py sync_from_dict)
    stamp_budget(...)                      one stream's stamp error interval and its components
    label_error(times, P, delta)           the largest end-effector displacement over a time offset delta
    suggest(result)                        the change the page's Apply makes (or None)

**Clocks and streams** live in project.rig['sync'] (core/project.py sync_from_dict): clocks {'uid', 'name', 'kind'
camera_hw | host | ptp | ntp | robot, 'offset_s', 'jitter_s', 'drift_ppm'}, each figure {'value', 'source'} or null
(not stated), and per stream (a sensor uid, or 'joint_states') its clock and trigger (free | hardware | sync_cable).
A stream with no entry is stamped by the host when its frame arrives, free-running: the default, said in the result.
Times are on the host's time base; a clock's offset is its reading minus the host's.

**Stamp error of a camera frame** e = stamp − the instant the frame was exposed, taken as the middle of the exposure
of its middle row (Liang et al. 2008; as Vantage core/timing.py). Each source of error is an interval [low, high]:

- Stamped by the host on arrival (no clock, or a 'host' clock), free or on a sync cable: the frame is complete when
  its last row is read, E + T_r after the first row starts exposing (exposure E, readout T_r), so it is
  T_r/2 + E/2 after the middle row's exposure centre for a rolling shutter and E/2 + T_r for a global one (rows
  exposed together, then read); then it crosses USB: T_x from bits / C (the link's capacity, nothing else on the bus)
  to (bits + the other frames on the same controller) / C_controller (every other device's frame queued first),
  bits = width × height × bits per pixel × (1 + overhead) of the stream, capacities and overheads from the bandwidth
  study's topology (core/studies/bandwidth.py: project.rig hosts, controllers, hubs and links); and the host sees the
  transfer end at the next bus interval, up to 125 µs later ("For high speed and SuperSpeed, the bus interval is a
  microframe. The length of a microframe is 125 microseconds", Microsoft, How to transfer data to USB isochronous
  endpoints, 2024-01-17, read 2026-11-07; USB 2.0 §5.8.4: 125 µs microframes). The device's own processing between
  readout and sending (a depth camera computes depth first) is not stated by any maker in the library: the interval
  is then a lower bound ("incomplete"), unless you enter the stream's measured arrival latency (settings 'latency_s'
  {uid: {'value', 'source'}}: stamp minus exposure, e.g. from the driver's metadata), which replaces the readout,
  transfer and processing terms.
- Stamped by the device's own clock (a 'camera_hw', 'ptp' or 'ntp' clock): the instant of the exposure the stamp marks
  is not stated: ± E/2; plus the clock's figures.
- Hardware trigger ('hardware', only where the maker supports one: the library entry's 'sync' block, cited): the
  frame starts on a trigger edge whose time is the stamp, on the stream's clock: e = −(E/2 + T_r/2) for a rolling
  shutter, −E/2 for a global one; transfer and processing no longer count. The trigger-to-exposure delay is not
  stated (not counted, said).
- A clock's figures, when its source states them: offset_s (a constant), jitter_s (±), drift_ppm (the clock walks
  drift × 10⁻⁶ × episode_s over an episode, episode_s 60 s by default). PTP and NTP figures are never assumed: enter
  what your servo reports (ptp4l's offset, chronyc tracking), with its source; a figure not entered is "not stated"
  and makes the budget incomplete.
- The exposure and the readout come from the motion blur study's imaging (core/studies/motion_blur.py imaging: your
  figures, the library's, or the frame period as an upper bound, each labelled). An exposure that is not fixed
  (auto-exposure) moves the stamp by its range: from the shortest the maker states (0 when not stated) to the bound.

**joint_states** has its own entry: stamped by the host on arrival from the robot controller (the default: that
latency is not stated, so not counted, said), or by a 'robot', 'ptp' or 'ntp' clock with its figures; settings
'latency_s' {'joint_states': ...} takes a measured latency. A frame's label is joint_states at the frame's stamp,
interpolated (pairing 'interpolate', the default) or the nearest sample (pairing 'nearest': ± half the sample period
1/(2 r), r = settings 'joint_rate_hz' {'value', 'source'}, the controller's stated or measured rate).

**Skew** of a camera stream against joint_states = e_camera − e_joint: the interval [e_c,low − e_j,high, e_c,high −
e_j,low]; the worst |skew| is its largest magnitude, the spread its width (what is left if a constant latency is
subtracted). Two cameras free-running are also out of phase by up to half the longer frame period when frames are
matched by timestamp (a sync cable or a common hardware trigger removes that): the camera pairs table.

**Action-label error** = the end-effector (the TCP, core/grasps.py gripper_geometry) displacement during the skew:
max over the trajectory of |p(t + δ) − p(t)| for δ at both ends of the skew interval, p from forward kinematics
(core/robot/model.py fk) of the trajectory's q(t) on a 1 ms grid (linear between grid points). The motions are the
motion blur study's (the trajectories, else the grasp approaches).

**Status**: the worst skew against skew_limit_s (2 ms, a DexVantage default, not a standard): within it "in sync", up
to twice "tight", beyond "out of sync"; a budget with terms not stated says "at least". The recommendation: a
hardware trigger for the worst stream when its maker supports one (with the predicted skew), else a sync cable for
camera pairs out of phase, else measuring the latency; Apply sets the trigger in project.rig['sync'] (one undo step).

Validation (dexvantage/validation/checks_sync.py, model 'timing'): the stamp-error interval against a discrete
simulation of a frame's timeline row by row (exposure, readout, transfer behind other frames on a shared bus, bus
interval quantisation) with random phases and exposures; the action-label error against a discrete simulation of
the moving end effector sampled at a camera's frame times with the stated skew.

Known limits: the processing time inside a device is not stated by the makers, so a host-stamped budget is a lower
bound until a measured latency is entered; USB transfers are modelled as the stream's frame at the link's capacity,
queued behind one frame of every other device on its controller (isochronous reservations, driver buffering and the
host's interrupt moderation are not modelled); clock figures are what you enter (no servo simulation; Vantage's clock
tree is automotive); the trigger-to-exposure delay of a hardware trigger is not counted; the TCP is the end effector
(an object held off-centre moves differently when the gripper turns); other joints stay at home.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from .. import jobs

STUDY_ID = 'time_sync'
DEFAULTS = {
    'skew_limit_s': 0.002,          # the spec's "2 ms sync skew" (a DexVantage default, not a standard)
    'episode_s': 60.0,              # one episode: the time a drifting clock walks over
    'grid_s': 0.001,                # forward kinematics grid for the action-label error
}
LIMITS = {'skew_limit_s': (1e-6, 10.0), 'episode_s': (0.1, 86400.0), 'grid_s': (1e-4, 0.1)}
PAIRINGS = ('interpolate', 'nearest')
BUS_INTERVAL_S = 125e-6             # Microsoft Learn, isochronous transfers; USB 2.0 §5.8.4 (module docstring)
BUS_INTERVAL_SOURCE = ('USB bus interval: "For high speed and SuperSpeed, the bus interval is a microframe. The length '
                       'of a microframe is 125 microseconds" (Microsoft, How to transfer data to USB isochronous '
                       'endpoints, dated 2024-01-17, read 2026-11-07); USB 2.0 §5.8.4 (125 µs microframes)')
WORDS = {'good': 'in sync', 'warning': 'tight', 'serious': 'out of sync', 'unknown': 'not known'}
REFERENCES = [
    'C.-K. Liang, L.-W. Chang and H. H. Chen, "Analysis and compensation of rolling shutter effect", IEEE Trans. '
    'Image Processing 17(8), 1323-1330, 2008 (row timing).',
    'Microsoft, "How to transfer data to USB isochronous endpoints", Windows driver documentation, 2024-01-17, '
    'https://learn.microsoft.com/windows-hardware/drivers/usbcon/transfer-data-to-isochronous-endpoints (read '
    '2026-11-07).',
    'Universal Serial Bus Specification, Revision 2.0, April 27, 2000 (§5.8.4, Table 5-10: 125 µs microframes).',
    'RealSense, D400 Series Datasheet, document 337029-017, rev. 023, March 2026 (sections 3.7.4 and 7.13: external '
    'trigger and multi-camera hardware sync).',
]


# ====================================================================== settings
def _fig(v):
    if not isinstance(v, dict):
        return None
    try:
        x = float(v.get('value'))
    except (TypeError, ValueError):
        return None
    return {'value': x, 'source': str(v.get('source') or '')} if math.isfinite(x) else None


def settings_of(project) -> dict:
    raw = dict(project.analysis.studies.get(STUDY_ID) or {})
    st, defaults = {}, []
    for k, dv in DEFAULTS.items():
        v = raw.get(k)
        try:
            x = float(v)
            ok = math.isfinite(x) and not isinstance(v, bool)
        except (TypeError, ValueError):
            ok = False
        if v is None or not ok:
            st[k] = dv
            defaults.append(k)
        else:
            lo, hi = LIMITS[k]
            st[k] = min(max(x, lo), hi)
    p = str(raw.get('pairing') or 'interpolate')
    st['pairing'] = p if p in PAIRINGS else 'interpolate'
    if 'pairing' not in raw:
        defaults.append('pairing')
    jr = _fig(raw.get('joint_rate_hz'))
    st['joint_rate_hz'] = jr if jr is not None and jr['value'] > 0 else None
    st['latency_s'] = {str(k): f for k, v in (raw.get('latency_s') or {}).items()
                       if (f := _fig(v)) is not None} if isinstance(raw.get('latency_s'), dict) else {}
    st['defaults'] = defaults
    return st


def sync_config(project) -> dict:
    from ..project import sync_from_dict
    rig = project.rig if isinstance(getattr(project, 'rig', None), dict) else {}
    return sync_from_dict(rig.get('sync'))


# ====================================================================== intervals
class Budget:
    """An interval [lo, hi] (s) built from labelled terms; missing terms make it incomplete."""

    def __init__(self):
        self.terms: list = []
        self.missing: list = []

    def add(self, name: str, lo: float, hi: float, source: str):
        lo, hi = float(min(lo, hi)), float(max(lo, hi))
        self.terms.append({'term': name, 'low_s': lo, 'high_s': hi, 'source': source})

    def miss(self, text: str):
        self.missing.append(text)

    @property
    def lo(self) -> float:
        return float(sum(t['low_s'] for t in self.terms))

    @property
    def hi(self) -> float:
        return float(sum(t['high_s'] for t in self.terms))

    def to_dict(self) -> dict:
        return {'low_s': _r(self.lo, 9), 'high_s': _r(self.hi, 9), 'complete': not self.missing,
                'terms': [{**t, 'low_s': _r(t['low_s'], 9), 'high_s': _r(t['high_s'], 9)} for t in self.terms],
                'not_stated': list(self.missing)}


def _r(x, nd=6):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def _clock_terms(b: Budget, clock: dict | None, st: dict):
    """A clock's stated figures as terms of b (module docstring)."""
    if clock is None or clock['kind'] == 'host':
        return
    nm = clock['name']
    for key, label in (('offset_s', 'offset'), ('jitter_s', 'jitter'), ('drift_ppm', 'drift')):
        f = clock.get(key)
        if f is None:
            b.miss(f'{nm}: {label} not stated')
            continue
        src = f['source'] or 'your figure, no source given'
        v = float(f['value'])
        if key == 'offset_s':
            b.add(f'{nm} offset', v, v, src)
        elif key == 'jitter_s':
            b.add(f'{nm} jitter', -abs(v), abs(v), src)
        else:
            w = v * 1e-6 * st['episode_s']
            b.add(f'{nm} drift over {st["episode_s"]:g} s', min(0.0, w), max(0.0, w), src)


def _exposure_range(c) -> tuple:
    """(E_lo, E_hi, fixed, source) of a camera imager (motion_blur.imaging), or (None, None, False, why)."""
    ex = c['exposure']
    if ex['value'] is None:
        return None, None, False, ex['source']
    if ex['kind'] == 'yours':
        return ex['value'], ex['value'], True, ex['source']
    lo = ex.get('stated_min') or 0.0
    return min(lo, ex['value']), ex['value'], False, (ex['source'] + '; auto-exposure from '
                                                      + (f'{lo * 1000:g} ms (stated)' if lo else '0 (no minimum '
                                                                                                'stated)'))


def _readout_range(c) -> tuple:
    """(T_lo, T_hi, source) of the frame's readout (motion_blur.imaging 'frame_readout'), or (None, None, why)."""
    ro = c.get('frame_readout') or c['readout']
    if ro['value'] is None:
        return None, None, ro['source']
    if ro['kind'] == 'upper bound':
        return 0.0, ro['value'], ro['source']
    return ro['value'], ro['value'], ro['source']


def stamp_budget(c: dict, cfg: dict, clocks: dict, transfer: dict, st: dict, trigger_ok) -> Budget:
    """The stamp error interval of camera imager c (motion_blur.imaging) under stream config cfg {'clock',
    'trigger'}; transfer {'min_s', 'max_s' (None: not bounded), 'source'}; trigger_ok True / False / None (the
    maker's statement). Module docstring."""
    b = Budget()
    uid = c['sensor'].uid
    clock = clocks.get(cfg.get('clock') or '')
    trig = cfg.get('trigger') or 'free'
    e_lo, e_hi, _fixed, e_src = _exposure_range(c)
    r_lo, r_hi, r_src = _readout_range(c)
    rolling = c['shutter'] == 'rolling'
    glob = c['shutter'] == 'global'
    if trig == 'hardware' and trigger_ok is False:
        b.miss('the maker states this imager takes no hardware trigger, so it free-runs: counted as stamped on '
               'arrival')
        trig = 'free'
    if trig == 'hardware':
        if trigger_ok is None:
            b.miss('whether this imager takes a hardware trigger is not stated by the maker')
        if e_lo is None:
            b.miss('exposure not known')
        else:
            b.add('trigger edge to the exposure centre', -e_hi / 2, -e_lo / 2, e_src)
        if rolling:
            if r_lo is None:
                b.miss('readout not known')
            else:
                b.add('to the middle row (rolling shutter)', -r_hi / 2, -r_lo / 2, r_src)
        b.miss('trigger-to-exposure delay (not stated by the maker; not counted)')
        _clock_terms(b, clock, st)
        return b
    if clock is not None and clock['kind'] in ('camera_hw', 'ptp', 'ntp', 'robot'):
        if e_lo is None:
            b.miss('exposure not known')
        else:
            b.add('instant of the exposure the device stamp marks (not stated)', -e_hi / 2, e_hi / 2, e_src)
        _clock_terms(b, clock, st)
        return b
    meas = st['latency_s'].get(uid)
    if meas is not None:
        b.add('measured arrival latency', meas['value'], meas['value'], meas['source'] or 'your figure')
    else:
        if e_lo is None:
            b.miss('exposure not known')
        else:
            b.add('exposure centre to the end of the exposure', e_lo / 2, e_hi / 2, e_src)
        if rolling or glob:
            if r_lo is None:
                b.miss(f'readout time not known ({r_src})')
            elif rolling:
                b.add('middle row to the last row read (rolling shutter)', r_lo / 2, r_hi / 2, r_src)
            else:
                b.add('readout after the exposure (global shutter)', r_lo, r_hi, r_src)
        else:
            b.miss('shutter type not stated')
        if transfer.get('min_s') is None:
            b.miss(transfer.get('source') or 'USB transfer time not known')
        else:
            hi = transfer['max_s'] if transfer.get('max_s') is not None else transfer['min_s']
            b.add('USB transfer of the frame', transfer['min_s'], hi, transfer['source'])
            if transfer.get('max_s') is None:
                b.miss(transfer.get('max_why') or 'queueing behind other devices not bounded')
        b.add('USB bus interval (the host sees the end of the transfer)', 0.0, BUS_INTERVAL_S, BUS_INTERVAL_SOURCE)
        b.miss('processing inside the device before it sends the frame (not stated by the maker)')
    _clock_terms(b, clock, st)
    return b


def joint_budget(cfg: dict, clocks: dict, st: dict) -> Budget:
    b = Budget()
    clock = clocks.get(cfg.get('clock') or '')
    meas = st['latency_s'].get('joint_states')
    if meas is not None:
        b.add('measured joint_states latency', meas['value'], meas['value'], meas['source'] or 'your figure')
    elif clock is None or clock['kind'] == 'host':
        b.miss('the robot controller to host latency of joint_states (not stated; enter a measured one)')
    _clock_terms(b, clock, st)
    if st['pairing'] == 'nearest':
        r = st['joint_rate_hz']
        if r is None:
            b.miss('the joint_states rate (pairing with the nearest sample needs it)')
        else:
            h = 0.5 / r['value']
            b.add(f'nearest joint_states sample at {r["value"]:g} Hz', -h, h, r['source'] or 'your figure')
    return b


# ====================================================================== USB frame timing
def transfer_times(project, cams) -> dict:
    """key -> {'min_s', 'max_s' (None: not bounded), 'source', 'max_why'} from the bandwidth study's topology."""
    from . import bandwidth as BW
    try:
        bw = BW.budget(project)
    except Exception as e:                               # noqa: BLE001 — explained in the result
        return {c['key']: {'min_s': None, 'max_s': None, 'source': f'the bandwidth study could not run: {e}'}
                for c in cams}
    rows = {r['uid']: r for r in bw['sensors']}
    ctrls = {c['uid']: c for c in bw['controllers']}

    def frame_bits(r, s):
        bpp = s.get('bits_per_pixel')
        if bpp is None or not s.get('width'):
            return None
        return s['width'] * s['height'] * bpp * (1.0 + r['overhead'])

    out = {}
    for c in cams:
        uid = c['sensor'].uid
        r = rows.get(uid)
        entry = None
        from .. import library as LB
        if c['sensor'].model_id:
            entry = LB.get(c['sensor'].model_id)
        if r is None or entry is None:
            out[c['key']] = {'min_s': None, 'max_s': None, 'source': 'no library device: the stream\'s size and '
                                                                     'format are not known'}
            continue
        mine = next((s for s in r['streams'] if (entry.imager(s['imager']) or {}).get('name') == c['imager'].name),
                    None)
        bits = frame_bits(r, mine) if mine else None
        if bits is None:
            out[c['key']] = {'min_s': None, 'max_s': None,
                             'source': 'the stream\'s bits per pixel are not stated, so its transfer time is not known'}
            continue
        if r['bus'] != 'usb':
            out[c['key']] = {'min_s': None, 'max_s': None,
                             'source': f'a {r["bus"].upper()} device: its link timing is not modelled'}
            continue
        cap = r.get('link_capacity_mbps')
        gen = r.get('usb_generation')
        src = r.get('link_capacity_source') or ''
        if not cap and gen in BW.CAPACITY_MBPS:
            cap, src = BW.CAPACITY_MBPS[gen], f'the device\'s own {BW.GEN_TITLE[gen]}: {BW.CAPACITY_WHY[gen]}'
        if not cap:
            out[c['key']] = {'min_s': None, 'max_s': None, 'source': 'the USB link capacity is not known'}
            continue
        t_min = bits / (cap * 1e6)
        ctrl = ctrls.get(r.get('controller_uid') or '')
        rec = {'min_s': t_min, 'max_s': None, 'bits': bits,
               'source': f'{bits / 8e6:.3f} MB per frame (with {r["overhead"] * 100:g} % overhead) at '
                         f'{cap:g} Mbit/s ({src})'}
        if ctrl is None:
            rec['max_why'] = (f'{c["sensor"].name} is not assigned to a host controller (Bandwidth page): the wait '
                              'behind other devices is not bounded')
        elif not ctrl.get('capacity_mbps'):
            rec['max_why'] = f'{ctrl["label"]} has no capacity figure: the wait behind other devices is not bounded'
        else:
            others, unknown = bits, []
            for u in ctrl['sensors']:
                ro = rows.get(u)
                if ro is None:
                    continue
                for s in ro['streams']:
                    if u == uid and s is mine:
                        continue
                    fb = frame_bits(ro, s)
                    if fb is None:
                        unknown.append(ro['name'])
                    else:
                        others += fb
            if unknown:
                rec['max_why'] = (f'{", ".join(sorted(set(unknown)))} on the same controller send streams without a '
                                  'stated rate: the wait behind them is not bounded')
            else:
                rec['max_s'] = others / (ctrl['capacity_mbps'] * 1e6)
                rec['source'] += (f'; at most after one frame of every other stream on {ctrl["label"]} '
                                  f'({(others - bits) / 8e6:.3f} MB) at {ctrl["capacity_mbps"]:g} Mbit/s')
        out[c['key']] = rec
    return out


# ====================================================================== the action-label error
def ee_path(project, traj: dict, grid_s: float) -> tuple:
    """(times (K,), TCP positions (K, 3) world) of a trajectory on a grid of grid_s, by forward kinematics."""
    from .. import grasps as GR
    from ..robot import trajectory as TR
    from ..robot.model import get_model
    m = get_model(project, traj['robot_uid'])
    geom = GR.gripper_geometry(project, traj['robot_uid'])
    t0, t1 = TR.duration(traj)
    n = max(2, int(math.ceil((t1 - t0) / grid_s)) + 1)
    times = np.linspace(t0, t1, n)
    tcp = np.r_[np.asarray(geom['tcp'], float), 1.0]
    i = m.link_index(geom['frame'])
    P = np.empty((n, 3))
    for k, t in enumerate(times):
        q = TR.on_model(traj, m.joint_names, m.home, float(t))
        P[k] = (m.fk_array(q)[i] @ tcp)[:3]
    return times, P


def label_error(times, P, delta: float) -> tuple:
    """(the largest |p(t + δ) − p(t)| over t in the path (m), the t where it is): p linear between path samples and
    held before the first and after the last."""
    times = np.asarray(times, float)
    P = np.asarray(P, float)
    tq = np.clip(times + float(delta), times[0], times[-1])
    Q = np.stack([np.interp(tq, times, P[:, k]) for k in range(3)], axis=1)
    d = np.linalg.norm(Q - P, axis=1)
    j = int(np.argmax(d))
    return float(d[j]), float(times[j])


# ====================================================================== the study
def _status(worst: float | None, limit: float) -> str:
    if worst is None:
        return 'unknown'
    if worst <= limit + 1e-15:
        return 'good'
    return 'warning' if worst <= 2 * limit else 'serious'


def _trigger_support(sensor, imager_name: str):
    """(True / False / None, the maker's words) for a hardware trigger on that imager (library 'sync')."""
    from .. import library as LB
    e = LB.get(sensor.model_id) if sensor.model_id else None
    if e is None:
        return None, 'no library device'
    s = e.data.get('sync')
    if not isinstance(s, dict) or s.get('hardware_trigger') is None:
        return None, f'{e.title}: not stated by the maker'
    ims = [str(x) for x in s.get('imagers') or []]
    if s['hardware_trigger'] and ims and imager_name not in ims:
        return False, f'{e.title}: only {", ".join(ims)} take the trigger ({s.get("what", "")})'
    return bool(s['hardware_trigger']), f'{e.title}: {s.get("what", "")}'


def _cable_support(sensor):
    from .. import library as LB
    e = LB.get(sensor.model_id) if sensor.model_id else None
    s = (e.data.get('sync') if e is not None else None) or {}
    return s.get('sync_cable') if isinstance(s, dict) else None


def analyse(project, ctx: jobs.Ctx | None = None, sync_override: dict | None = None) -> dict:
    """The study's result without the job wrapper; sync_override replaces project.rig['sync'] (for predictions)."""
    from ..project import sync_from_dict
    from . import motion_blur as MB
    ctx = ctx or jobs.Ctx()
    st = settings_of(project)
    cfg = sync_from_dict(sync_override) if sync_override is not None else sync_config(project)
    clocks = {c['uid']: c for c in cfg['clocks']}
    mst = MB.settings_of(project)
    cams = MB.imaging(project, mst)
    if not cams:
        raise jobs.StudyError('There is no camera in the rig to time: add a camera or depth sensor and switch it on.')
    notes = []
    if not cfg['clocks'] and not cfg['streams']:
        notes.append('No clocks or triggers are set (Time sync page): every stream is taken as stamped by the host on '
                     'arrival and free-running, the default.')
    for k, v in cfg['streams'].items():
        if v['clock'] and v['clock'] not in clocks:
            notes.append(f'A stream names a clock that is not in the rig ({v["clock"]}): it is taken as host '
                         'arrival.')
    ctx.report(0.05, 'USB frame timing')
    xfer = transfer_times(project, cams)
    jcfg = cfg['streams'].get('joint_states', {'clock': '', 'trigger': 'free'})
    jb = joint_budget(jcfg, clocks, st)
    # motions and their end-effector paths
    trajs, source, mnotes = MB.motions(project, mst)
    notes += mnotes
    paths = []
    for k, tr in enumerate(trajs):
        if ctx.cancelled():
            break
        try:
            times, P = ee_path(project, tr, st['grid_s'])
        except Exception as e:                           # noqa: BLE001 — a robot that cannot be posed: said
            notes.append(f'{tr.get("name", "A trajectory")} is left out of the action-label error: {e}')
            continue
        dt = np.diff(times)
        v = np.linalg.norm(np.diff(P, axis=0), axis=1) / np.where(dt > 0, dt, 1.0)
        paths.append((tr, times, P, float(v.max()) if len(v) else 0.0))
        ctx.report(0.1 + 0.6 * (k + 1) / max(1, len(trajs)), tr.get('name', ''))
    limit = st['skew_limit_s']
    streams = []
    for c in cams:
        s = c['sensor']
        scfg = cfg['streams'].get(s.uid, {'clock': '', 'trigger': 'free'})
        tsup, twhat = _trigger_support(s, c['imager'].name)
        b = stamp_budget(c, scfg, clocks, xfer[c['key']], st, tsup)
        lo, hi = b.lo - jb.hi, b.hi - jb.lo
        worst = max(abs(lo), abs(hi))
        complete = not b.missing and not jb.missing
        errs = []
        for tr, times, P, vmax in paths:
            e_lo, t_lo = label_error(times, P, lo)
            e_hi, t_hi = label_error(times, P, hi)
            e_sp, _t = label_error(times, P, (hi - lo) / 2)
            e, t, dl = (e_lo, t_lo, lo) if e_lo >= e_hi else (e_hi, t_hi, hi)
            a = [float(np.interp(t, times, P[:, k])) for k in range(3)]
            tb = min(max(t + dl, times[0]), times[-1])
            bpt = [float(np.interp(tb, times, P[:, k])) for k in range(3)]
            errs.append({'trajectory_uid': tr['uid'], 'trajectory': tr.get('name', ''), 'error_m': _r(e, 7),
                         't': _r(t, 4), 'skew_s': _r(dl, 9), 'spread_error_m': _r(e_sp, 7),
                         'peak_speed_m_s': _r(vmax, 5), 'tcp_at_exposure': [_r(x, 6) for x in a],
                         'tcp_at_stamp': [_r(x, 6) for x in bpt]})
        werr = max(errs, key=lambda x: x['error_m'] or 0.0) if errs else None
        status = _status(worst, limit)
        if status == 'good' and not complete:
            status = 'warning'
        streams.append({'key': c['key'], 'sensor_uid': s.uid, 'sensor': s.name, 'imager': c['imager'].name,
                        'name': c['name'], 'slot': s.slot, 'fps': c['fps'], 'shutter': c['shutter'],
                        'clock': _clock_name(scfg['clock'], clocks, s.uid in cfg['streams']),
                        'clock_uid': scfg['clock'], 'trigger': scfg['trigger'],
                        'configured': s.uid in cfg['streams'],
                        'hardware_trigger': tsup, 'hardware_trigger_what': twhat, 'sync_cable': _cable_support(s),
                        'stamp': b.to_dict(), 'skew_low_s': _r(lo, 9), 'skew_high_s': _r(hi, 9),
                        'worst_skew_s': _r(worst, 9), 'spread_s': _r(hi - lo, 9), 'complete': complete,
                        'status': status, 'word': _word(status, complete),
                        'transfer': {k: (_r(v, 9) if isinstance(v, float) else v) for k, v in xfer[c['key']].items()},
                        'label_error': errs, 'label_error_m': werr['error_m'] if werr else None,
                        'label_error_worst': werr})
    pairs = _pairs(streams, cams, cfg)
    res = {'settings': copy.deepcopy(st), 'skew_limit_s': limit, 'skew_limit_default': 'skew_limit_s' in st['defaults'],
           'sync': copy.deepcopy(cfg),
           'joint_states': {**jb.to_dict(), 'clock': _clock_name(jcfg['clock'], clocks, 'joint_states' in
                                                                 cfg['streams']), 'pairing': st['pairing']},
           'streams': streams, 'pairs': pairs, 'motions': {'source': source, 'trajectories': [
               {'uid': tr['uid'], 'name': tr.get('name', ''), 'robot_uid': tr['robot_uid'],
                'synthetic': bool(tr.get('synthetic')), 'peak_tcp_speed_m_s': _r(v, 5),
                'tcp_path': [[_r(x, 5) for x in q] for q in P[np.linspace(0, len(P) - 1, min(len(P), 200))
                                                              .round().astype(int)]]}
               for tr, _t, P, v in paths]},
           'notes': notes, 'references': list(REFERENCES), 'units': 'seconds; metres'}
    res['summary'] = _summary(res)
    ctx.report(0.8, 'what would help')
    res['suggestion'] = None if sync_override is not None else suggest(res, project)
    return res


def _clock_name(uid: str, clocks: dict, configured: bool) -> str:
    if uid and uid in clocks:
        return clocks[uid]['name']
    if uid:
        return 'a clock not in the rig (host arrival)'
    return 'host arrival' if configured else 'host arrival (default)'


def _word(status: str, complete: bool) -> str:
    if complete:
        return WORDS[status]
    return WORDS[status] if status == 'serious' else 'incomplete'     # serious with terms missing: the figure is '≥'


def _pairs(streams, cams, cfg) -> list:
    """Camera pairs (one row per two sensors): the phase between free-running frames plus the stamp differences."""
    out = []
    by = {}
    for s, c in zip(streams, cams):
        by.setdefault(s['sensor_uid'], (s, c))
    keys = list(by)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            (a, ca), (b, cb) = by[keys[i]], by[keys[j]]
            locked = (a['trigger'] == b['trigger'] == 'sync_cable') or (
                a['trigger'] == b['trigger'] == 'hardware' and a['clock_uid'] == b['clock_uid'])
            if locked or not (ca['fps'] and cb['fps']):
                ph = 0.0 if locked else None
            else:
                ph = 0.5 / min(ca['fps'], cb['fps'])
            if ph is None:
                out.append({'a': a['name'], 'b': b['name'], 'phase_s': None, 'worst_s': None, 'locked': locked,
                            'note': 'frame rate not known'})
                continue
            d_lo = a['stamp']['low_s'] - b['stamp']['high_s']
            d_hi = a['stamp']['high_s'] - b['stamp']['low_s']
            worst = ph + max(abs(d_lo), abs(d_hi))
            out.append({'a': a['name'], 'b': b['name'], 'a_uid': a['sensor_uid'], 'b_uid': b['sensor_uid'],
                        'phase_s': _r(ph, 9), 'stamp_difference_s': [_r(d_lo, 9), _r(d_hi, 9)],
                        'worst_s': _r(worst, 9), 'locked': locked,
                        'complete': a['stamp']['complete'] and b['stamp']['complete'],
                        'cable_possible': bool(_both(a['sync_cable'], b['sync_cable']))})
    return out


def _both(x, y):
    return x is True and y is True


def _summary(res) -> dict:
    ss = res['streams']
    worst = max(ss, key=lambda s: s['worst_skew_s'] or 0.0)
    order = {'serious': 3, 'warning': 2, 'unknown': 1, 'good': 0}
    status = max((s['status'] for s in ss), key=lambda x: order[x])
    lim = res['skew_limit_s']
    incomplete = [s['key'] for s in ss if not s['complete']]
    word = WORDS[status]
    if incomplete and status == 'serious':
        pass                                    # out of sync already; the figure is a lower bound (shown with ≥)
    elif incomplete and status == 'warning' and (worst['worst_skew_s'] or 0) <= lim:
        word = 'incomplete'
    return {'status': status, 'word': word, 'worst': worst['key'], 'worst_name': worst['name'],
            'worst_skew_s': worst['worst_skew_s'], 'skew_interval_s': [worst['skew_low_s'], worst['skew_high_s']],
            'label_error_m': worst['label_error_m'], 'label_error_worst': worst['label_error_worst'],
            'complete': not incomplete, 'incomplete': incomplete,
            'over_limit': [s['key'] for s in ss if (s['worst_skew_s'] or 0) > lim + 1e-15]}


def suggest(res: dict, project=None) -> dict | None:
    """The change that brings the worst stream's skew down most: a hardware trigger where the maker supports one
    ({'kind': 'trigger', 'sensor_uid', 'imager', 'before', 'after' (the predicted worst skew and label error)}), a
    sync cable between free-running cameras whose makers both support one ({'kind': 'sync_cable', 'sensor_uids'}),
    or {'kind': 'advice'}; None when every stream is in sync and its budget complete."""
    lim = res['skew_limit_s']
    ss = sorted(res['streams'], key=lambda s: -(s['worst_skew_s'] or 0.0))
    if all(s['status'] == 'good' for s in ss):
        return None
    for s in ss:
        if s['trigger'] == 'hardware' or s['hardware_trigger'] is not True:
            continue
        if project is None:
            break
        cfg = copy.deepcopy(res['sync'])
        cur = cfg['streams'].get(s['sensor_uid'], {'clock': '', 'trigger': 'free'})
        cfg['streams'][s['sensor_uid']] = {'clock': cur['clock'], 'trigger': 'hardware'}
        try:
            after = analyse(project, None, cfg)
        except Exception:                               # noqa: BLE001 — no prediction: no suggestion of this kind
            break
        a = next((x for x in after['streams'] if x['key'] == s['key']), None)
        if a is None:
            break
        text = (f'Trigger {s["sensor"]} in hardware ({s["hardware_trigger_what"]}): the stamp becomes the trigger '
                f'edge, so the USB transfer and the device\'s processing leave the budget. Predicted worst skew of '
                f'{s["name"]} against joint_states {_ms(s["worst_skew_s"])} → {_ms(a["worst_skew_s"])}'
                + (f', action-label error {_mm(s["label_error_m"])} → {_mm(a["label_error_m"])}'
                   if s['label_error_m'] is not None else '') + '. What is left is half the exposure (and half the '
                'readout of a rolling shutter): with a fixed exposure (the Motion blur page) it is a constant that '
                'can be subtracted. The trigger signal must come from a clock the host knows (the host itself, or a '
                'PTP-disciplined trigger source); its delay to the exposure is not stated and not counted.')
        return {'kind': 'trigger', 'sensor_uid': s['sensor_uid'], 'sensor': s['sensor'], 'key': s['key'],
                'clock': cur['clock'],
                'before': {'worst_skew_s': s['worst_skew_s'], 'label_error_m': s['label_error_m']},
                'after': {'worst_skew_s': a['worst_skew_s'], 'label_error_m': a['label_error_m'],
                          'complete': a['complete']}, 'text': text}
    for p in res['pairs']:
        if not p.get('locked') and p.get('cable_possible') and (p['worst_s'] or 0) > lim:
            return {'kind': 'sync_cable', 'sensor_uids': [p['a_uid'], p['b_uid']],
                    'text': f'Connect {p["a"]} and {p["b"]} with a sync cable (both makers support one): their frames '
                            f'are then exposed together instead of up to {_ms(p["phase_s"])} apart.'}
    w = ss[0]
    why = ('its maker states it has no hardware trigger' if w['hardware_trigger'] is False else
           'its maker does not state a hardware trigger' if w['hardware_trigger'] is None else
           'it is already triggered')
    text = (f'{w["name"]}: {why} ({w["hardware_trigger_what"]}). Measure the stream\'s arrival latency (stamp minus '
            'exposure, from the driver\'s frame metadata) and enter it with its source, so a constant delay can be '
            'subtracted; or stamp the frames on a clock disciplined by PTP and enter its figures.')
    return {'kind': 'advice', 'text': text}


def _ms(x) -> str:
    return '—' if x is None else f'{x * 1000:.3g} ms'


def _mm(x) -> str:
    return '—' if x is None else f'{x * 1000:.3g} mm'


def apply_suggestion(project, sug: dict) -> None:
    """Make the suggested trigger or sync cable in project.rig['sync'] (the caller makes it one undo step)."""
    from ..project import rig_from_dict, sync_from_dict
    rig = rig_from_dict(project.rig)
    cfg = sync_from_dict(rig.get('sync'))
    if sug.get('kind') == 'trigger':
        cfg['streams'][sug['sensor_uid']] = {'clock': sug.get('clock') or '', 'trigger': 'hardware'}
    elif sug.get('kind') == 'sync_cable':
        for u in sug['sensor_uids']:
            cur = cfg['streams'].get(u, {'clock': '', 'trigger': 'free'})
            cfg['streams'][u] = {'clock': cur['clock'], 'trigger': 'sync_cable'}
    else:
        return
    rig['sync'] = cfg
    project.rig = rig_from_dict(rig)


@jobs.study('time_sync', 'Time sync', quick=False, batch='time-sync', needs_robot=True)
def run(project, ctx: jobs.Ctx) -> dict:
    """Stamp skew and jitter per stream against joint_states and the action-label error (module docstring)."""
    return analyse(project, ctx)
