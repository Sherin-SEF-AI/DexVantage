"""Bandwidth, hosts and power (spec §7.6, §6 Hosts): whether the rig's host controllers, hubs and links carry every
stream, how much one episode and one campaign record, whether the recorder writes fast enough, and whether each
bus-powered USB device gets the current it states.

    @jobs.study('bandwidth', 'Bandwidth, hosts and power', batch='bandwidth', needs_robot=False) → --bandwidth OUT.json
    run(project, ctx) -> dict          the study (also core.jobs.run_study('bandwidth', project))
    budget(project, settings=None)     the same result without the job wrapper
    stream_id / parse_stream_id        a stream's id in project.rig links: 'imager|WxH|fps|format'
    sensor_streams(sensor, entry, ids, settings)   the streams a sensor sends and their data rates
    suggest(result)                    the change the page's Apply makes (or None)

Ported from Vantage core/recording.py (data-rate rows, the disk budget, link loads and their findings) and
core/power.py (devices against what feeds them, findings sorted worst first), with the vehicle's GMSL ECU, lidars,
radars, DC-DC converter, alternator and fuses stripped; the host topology (USB controllers, hubs, PCIe and GMSL
deserialisers) and USB bus power are new.

**The topology** lives in project.rig (core/project.py rig_from_dict): hosts, each with its controllers (kind usb3,
usb2, pcie or gmsl, and the controller's practical bandwidth when a source states it); hubs, each plugged into a
controller or another hub, bus-powered or powered; and links, one per sensor cable: the sensor, the streams it sends
(empty: the default streams below) and the controller or hub it plugs into. A sensor of the active variant with no
link is "not assigned": its data still counts for the disk, not for any controller.

**Data rate of a stream** (decimal units, Mbit/s = 10⁶ bit/s, MB = 10⁶ bytes, as links and disks are sold):

    payload = width × height × fps × bits per pixel / 10⁶        (bytes per pixel = bits per pixel / 8)
    on the wire = payload × (1 + overhead)

with width, height, the frame rate and the bits per pixel of the stream mode as the library entry states them
(core/library, every figure cited). A stream whose bits per pixel the maker does not state (MJPEG, the Luxonis
sensor modes whose output format depends on the on-device pipeline, the DIGIT tactile camera, the GelSight Mini, which
lists no stream at all) has no data rate: it is "not stated", the budget says it is incomplete, and the user can
enter a measured rate for the sensor (settings 'measured' {uid: {'mbps', 'source'}}). Tactile sensors are counted as
any other sensor. Extra streams the library does not list (an IMU, audio) go in settings 'extra' {uid: [{'name',
'mbps', 'source'}]}.

overhead is the share the class and link protocol add to the pixels (video class payload headers, frame gaps, CSI-2
packet headers over GMSL) on top of what the capacity figure below already takes off. No maker in the library states
it, so it is a **labelled default per interface**, 5 % (DEFAULT_OVERHEAD, with the reasoning in OVERHEAD_WHY), and
a user figure with its source replaces it (settings 'overhead' {bus: {'value', 'source'}}).

**Default streams.** A link with no stream ids sends, for a depth sensor, its 'depth' stream and its RGB stream when
the device lists them (each sensor computes depth on board), else one stream of every imager (a ZED sends both
images); for a camera or tactile sensor its first imager's stream. The size is the stream chosen for the sensor
(Sensor.stream) when that imager streams it, else its largest; the format the first listed at that size; the rate the
listed one nearest to default_fps, 30 fps (a default). The result labels each stream "default selection".

**Capacity.** A controller's capacity is its practical_mbps when a source states it (a measurement, the board's
manual). Otherwise, for USB, the effective bandwidth the USB specifications state as an upper bound, a figure no real
controller exceeds and many fall short of:

- USB 3.2 Gen 1x1 (5 Gbit/s, "USB 3.0", "USB 3.1 Gen 1"): "the raw throughput is 500 MB/sec. Accounting for flow
  control, packet framing and protocol overheads reduces the effective bandwidth down to 450 MB/sec or less"; Gen 2x1
  (10 Gbit/s): "approximately 1.1 GB/sec or less"; "the effective bandwidth is doubled" for x2 (USB 3.2 Revision 1.1,
  §4.4.11 Efficiency): 3600, 8800 and 17 600 Mbit/s.
- USB 2.0 high speed (480 Mbit/s): at most 13 bulk transactions of 512 bytes per 125 µs microframe, 53 248 000
  bytes/s, "does not include the overhead associated with bit stuffing" (USB 2.0 §5.8.4, Table 5-10): 425.984 Mbit/s.

A usb3 controller's generation comes from its host's library entry (its stated per-port signalling rate: 5000, 10 000
or 20 000 Mbit/s, USB-IF); without one, Gen 1, the lowest, is assumed and said. A host whose interface names USB 3
ports of several generations (the Jetson AGX Orin developer kit: Gen 2 and Gen 1) cannot say which one a controller
serves: the lowest named is assumed and said (host_usb3_generations). A PCIe or GMSL controller without a
stated figure is not checked (the GMSL2 link rate is in the serialiser's and deserialiser's datasheets, which the
library could not read; enter it). A sensor's link carries all its streams (depth + RGB + IMU on one cable) and its
capacity is the lower of the device's own USB generation (parsed from the entry's interface: "USB 3.1 Gen 1" is Gen
1) and its controller's. A hub's uplink carries everything below it, at its controller's generation.

Every load is a share of its capacity; above margin_warn (80 %, a default) it is "tight" (warning), above 100 % "over
budget" (critical). The margin is capacity − load in Mbit/s.

**Flags**: a controller or a link over budget or tight; a hub's uplink over budget; a hub on a shared controller
(the controller also carries other devices or hubs, so everything on the hub shares the controller's bandwidth with
them); a USB device on a GMSL or PCIe controller or a GMSL device on a USB one; a link to a controller or hub that is
not in the rig, a hub loop; sensors not assigned; data rates or power figures not stated.

**Disk.** recorded = Σ payload of the variant's sensors in use × record_ratio (1.0: recorded as streamed, a default;
enter your encoder's measured ratio) × (1 + container_overhead), container_overhead 3 % (message headers and
indexes, a default as in Vantage); per episode recorded × episode_s (60 s, a default) and per campaign × episodes
(100, a default). The recorder must sustain recorded × write_headroom (1.3, a default) against its stated sustained
write (settings 'recorder' {'name', 'write_mb_s', 'capacity_gb', 'source'}; none of the library's hosts states a disk,
so without your figure the write check is "not stated"). With capacity_gb, the episodes that fit on it.

**Power.** USB bus power from the specifications: a unit load is 100 mA for USB 2.0 and a high-power device draws up
to five (500 mA) (USB 2.0 §7.2.1); for Enhanced SuperSpeed single-lane operation a unit load is 150 mA and a
high-power device up to six (900 mA), 250 mA and 1500 mA for two-lane operation (USB 3.2 Table 3-1, §11.4.1). A
device's current is its stated power (the entry's power_w, as the maker states it: often a maximum) at the nominal
5 V (the source is 4.75 V to 5.50 V, a device must work from 4.00 V: §11.4.2; a regulated device draws more current
at a lower voltage, which is not modelled). A device on a root port or a powered hub port (a self-powered hub "shall
ensure that it is able to provide at least six unit loads for each exposed downstream port", USB 2.0: five) is
checked against the high-power allowance. A bus-powered hub shares its own upstream allowance among everything below
it (USB 3.2 §10.14: "able to supply any power not used by the hub electronics ... with any split across the exposed
downstream ports"); a USB 2.0 bus-powered hub's ports "can supply only one unit load per port" (100 mA). The hub's
own electronics are not stated anywhere here and count 0 mA (said). A sensor listed in settings 'external_power' has
its own supply and is not counted; PCIe and GMSL devices are not checked (power over coax comes from the deserialiser
board, whose budget is not stated).

Settings (project.analysis.studies['bandwidth'], every key optional; DEFAULTS): margin_warn, default_fps, overhead,
measured, extra, record_ratio, container_overhead, episode_s, episodes, write_headroom, recorder, external_power.

The result is JSON-able: sensors (streams, rates, link, power), controllers, hubs, disk, power, findings worst first,
the headline and the suggestion.

Validation (dexvantage/validation/checks_bandwidth.py, model 'bandwidth'): the stream data rate against the rates
makers state for a mode (the NanEyeC's serial rates), the specification-derived capacities and bus-power allowances
against the figures each specification states separately, and the whole budget against an independent re-computation
on seeded random topologies.

Known limits: the capacity of a controller without a measured figure is the specification's upper bound, which real
controllers and drivers often miss (several ports can share one controller, which the maker rarely states: lsusb
-t shows it); isochronous USB transfers have their own per-endpoint limits (USB 2.0 Table 5-5), not modelled; the
overhead is a default, not a measurement; compressed streams (MJPEG, H.264) have no stated rate and need a
measurement; stream modes the device cannot run at the same time (Intel's simultaneous-stream tables) are not
checked; a 10 Gbit/s host is taken as Gen 2x1 (Gen 1x2, 7200 Mbit/s effective, is not told apart); bus power uses
the maker's stated figure at 5 V, not the inrush or the drop along the cable; hub
electronics count 0 mA; USB Type-C current (1.5 A, 3 A) and USB Power Delivery are not modelled; the GMSL2 link rate
and the deserialisers' power-over-coax budgets are not in the library; the disk check is a sustained rate, not the
burst behaviour of an SSD's cache.
"""
from __future__ import annotations

import copy
import math
import re

from .. import jobs

STUDY_ID = 'bandwidth'
SEVERITIES = ('critical', 'serious', 'warning', 'note', 'good')
WORDS = {'good': 'within budget', 'warning': 'tight', 'serious': 'problems', 'critical': 'over budget'}

# ---------------------------------------------------------------------- figures the specifications state (cited)
REF_USB32 = ('Universal Serial Bus 3.2 Specification, Revision 1.1, June 2022 (USB 3.0 Promoter Group), '
             'https://www.usb.org/sites/default/files/usb_32_202206_0.zip, read 2026-11-06')
REF_USB20 = ('Universal Serial Bus Specification, Revision 2.0, April 27, 2000, '
             'https://www.usb.org/sites/default/files/usb_20_20250603.zip, read 2026-11-06')
USB2_BULK_PACKETS_PER_MICROFRAME = 13          # USB 2.0 Table 5-10, 512-byte payloads
USB2_BULK_PAYLOAD_BYTES = 512
USB2_MICROFRAMES_PER_S = 8000                  # 125 µs microframes
USB2_BULK_BYTES_S_STATED = 53_248_000          # USB 2.0 Table 5-10 "Max Bandwidth (bytes/second)" at 512 bytes
USB3_EFFECTIVE_MB_S = {'gen1': 450.0, 'gen2': 1100.0}    # USB 3.2 §4.4.11: "450 MB/sec or less", "approximately 1.1
#                                                          GB/sec or less"; "doubled in Gen 2x2 operation"
SIGNALLING_MBPS = {'usb2': 480.0, 'gen1': 5000.0, 'gen2': 10000.0, 'gen2x2': 20000.0}   # USB 3.2 Table 3-1; USB-IF
CAPACITY_MBPS = {
    'usb2': USB2_BULK_PACKETS_PER_MICROFRAME * USB2_BULK_PAYLOAD_BYTES * USB2_MICROFRAMES_PER_S * 8 / 1e6,
    'gen1': USB3_EFFECTIVE_MB_S['gen1'] * 8.0,
    'gen2': USB3_EFFECTIVE_MB_S['gen2'] * 8.0,
    'gen2x2': 2.0 * USB3_EFFECTIVE_MB_S['gen2'] * 8.0,
}
CAPACITY_WHY = {
    'usb2': ('USB 2.0 high speed: 13 bulk transactions of 512 bytes per microframe, 53 248 000 bytes/s, without '
             'bit stuffing (USB 2.0 Table 5-10), an upper bound'),
    'gen1': ('USB 3.2 Gen 1x1: "450 MB/sec or less" effective after flow control, packet framing and protocol '
             'overheads (USB 3.2 §4.4.11), an upper bound'),
    'gen2': ('USB 3.2 Gen 2x1: "approximately 1.1 GB/sec or less" effective (USB 3.2 §4.4.11), an upper bound'),
    'gen2x2': ('USB 3.2 Gen 2x2: the Gen 2x1 effective figure "doubled in Gen 2x2 operation" (USB 3.2 §4.4.11), '
               'an upper bound'),
}
GEN_TITLE = {'usb2': 'USB 2.0', 'gen1': 'USB 3.2 Gen 1', 'gen2': 'USB 3.2 Gen 2', 'gen2x2': 'USB 3.2 Gen 2x2'}
GEN_RANK = {'usb2': 0, 'gen1': 1, 'gen2': 2, 'gen2x2': 3}
UNIT_LOAD_MA = {'usb2': 100.0, 'usb3': 150.0, 'usb3x2': 250.0}      # USB 2.0 §7.2.1; USB 3.2 Table 3-1, §11.4.1
HIGH_POWER_UNITS = {'usb2': 5, 'usb3': 6, 'usb3x2': 6}
VBUS_NOMINAL_V = 5.0                           # USB 3.2 §11.4.2: "The nominal 5 V source is 4.75 V to 5.50 V"
VBUS_SOURCE_V = (4.75, 5.50)
VBUS_MIN_DEVICE_V = 4.00                       # USB 3.2 §11.4.2: devices work "with as little as 4.00 V"

# ---------------------------------------------------------------------- labelled defaults
DEFAULT_OVERHEAD = {'usb': 0.05, 'gmsl': 0.05, 'pcie': 0.05, 'other': 0.05}
OVERHEAD_WHY = {
    'usb': ('a default, not a stated figure: the USB capacity used here already takes off the link\'s flow control, '
            'framing and protocol overheads (USB 3.2 §4.4.11; USB 2.0 Table 5-10 does not count bit stuffing), and '
            'the 5 % covers what the camera\'s own class protocol adds (video class payload headers, frame gaps), '
            'which no maker in the library states'),
    'gmsl': ('a default, not a stated figure: CSI-2 packet headers and line blanking carried over the serial link; '
             'the serialiser datasheets that state them were not read'),
    'pcie': 'a default, not a stated figure: the frame grabber\'s own protocol, which no source here states',
    'other': 'a default, not a stated figure: the interface\'s own protocol, which no source here states',
}
DEFAULTS = {
    'margin_warn': 0.8,          # default: a link or controller busier than this share reads "tight"
    'default_fps': 30.0,         # default: the frame rate of a default stream (the listed rate nearest to it)
    'record_ratio': 1.0,         # default: recorded as streamed (enter your encoder's measured ratio)
    'container_overhead': 0.03,  # default: message headers and indexes, a share of what is recorded (as Vantage)
    'episode_s': 60.0,           # default episode length
    'episodes': 100,             # default episodes per campaign
    'write_headroom': 1.3,       # default: the recorder must sustain this many times the data rate (as Vantage)
}
LIMITS = {'margin_warn': (0.05, 1.0), 'default_fps': (0.1, 10000.0), 'record_ratio': (0.0001, 10.0),
          'container_overhead': (0.0, 1.0), 'episode_s': (0.1, 86400.0), 'episodes': (1, 10_000_000),
          'write_headroom': (1.0, 10.0)}
ASSUMPTIONS = (
    'Data rates are the stream modes as the makers state them (width × height × fps × bits per pixel); a stream '
    'whose bits per pixel are not stated has no rate until you enter a measured one.',
    'Protocol overhead is a default per interface (5 %), not a stated figure.',
    'A controller without a measured practical figure is compared with the USB specification\'s effective bandwidth, '
    'an upper bound real controllers often miss; several ports can share one controller.',
    'Bus power is the maker\'s stated power at the nominal 5 V; hub electronics count 0 mA.',
    'Episode length, episode count, recording ratio, container overhead and write headroom are defaults until you '
    'set them.',
)


# ====================================================================== small helpers
def _num(v, default=None):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _pos(v):
    v = _num(v)
    return v if v is not None and v > 0 else None


def _norm(s) -> str:
    return re.sub(r'[\s_-]+', ' ', str(s or '')).strip().lower()


def _r(v, d=3):
    return None if v is None or not math.isfinite(float(v)) else round(float(v), d)


def settings_of(project) -> dict:
    """The study's settings with the defaults filled in; 'defaults' lists the keys left at their default."""
    given = dict(project.analysis.studies.get(STUDY_ID) or {})
    st = dict(DEFAULTS)
    used = []
    for k, (lo, hi) in LIMITS.items():
        v = _num(given.get(k))
        if v is None:
            used.append(k)
            continue
        st[k] = min(max(v, lo), hi)
    st['episodes'] = int(round(st['episodes']))
    ov = {}
    for bus, val in (given.get('overhead') or {}).items() if isinstance(given.get('overhead'), dict) else ():
        x = _num(val.get('value') if isinstance(val, dict) else val)
        if bus in DEFAULT_OVERHEAD and x is not None and 0 <= x <= 5:
            ov[bus] = {'value': x, 'source': str(val.get('source') or '') if isinstance(val, dict) else ''}
    st['overhead'] = ov
    meas = {}
    for uid, m in (given.get('measured') or {}).items() if isinstance(given.get('measured'), dict) else ():
        x = _num(m.get('mbps') if isinstance(m, dict) else m)
        if x is not None and x >= 0:
            meas[str(uid)] = {'mbps': x, 'source': str(m.get('source') or '') if isinstance(m, dict) else ''}
    st['measured'] = meas
    extra = {}
    for uid, rows in (given.get('extra') or {}).items() if isinstance(given.get('extra'), dict) else ():
        out = []
        for e in rows if isinstance(rows, list) else []:
            x = _num(e.get('mbps')) if isinstance(e, dict) else None
            if x is not None and x >= 0:
                out.append({'name': str(e.get('name') or 'Extra stream'), 'mbps': x,
                            'source': str(e.get('source') or '')})
        if out:
            extra[str(uid)] = out
    st['extra'] = extra
    rec = given.get('recorder') if isinstance(given.get('recorder'), dict) else {}
    st['recorder'] = {'name': str(rec.get('name') or ''), 'write_mb_s': _pos(rec.get('write_mb_s')),
                      'capacity_gb': _pos(rec.get('capacity_gb')), 'source': str(rec.get('source') or '')}
    ext = given.get('external_power')
    st['external_power'] = [str(u) for u in ext] if isinstance(ext, list) else []
    st['defaults'] = used
    return st


# ====================================================================== interfaces
def usb_generation(text) -> str | None:
    """The fastest USB generation an interface text names ('USB 3.1 Gen 1' → 'gen1', 'USB 3.0' → 'gen1', 'USB 3.2 Gen
    2x2' → 'gen2x2', 'USB 2.0' → 'usb2'); None when it names USB without a generation, or no USB."""
    t = str(text or '').lower()
    if 'usb' not in t and 'superspeed' not in t:
        return None
    found = []
    if re.search(r'gen\s*2\s*x\s*2|20\s*gb', t):
        found.append('gen2x2')
    if re.search(r'gen\s*2(?!\s*x)|10\s*gb', t):
        found.append('gen2')
    if re.search(r'gen\s*1|usb\s*3(\.0)?\b|usb3(?!\.\d\s*gen\s*2)|superspeed|5\s*gb', t):
        found.append('gen1')
    if re.search(r'usb\s*2', t):
        found.append('usb2')
    return max(found, key=GEN_RANK.get) if found else None


def device_bus(entry) -> str:
    """'usb', 'gmsl' or 'other' for a library entry by its interface text ('other' for analog and serial pads)."""
    t = str((entry.data.get('interface') if entry is not None else '') or '').lower()
    if 'gmsl' in t:
        return 'gmsl'
    if 'usb' in t:
        return 'usb'
    if 'pcie' in t or 'pci express' in t:
        return 'pcie'
    return 'other'


def controller_bus(kind: str) -> str:
    return 'usb' if kind in ('usb3', 'usb2') else kind


def host_generation(host_entry) -> str | None:
    """A host entry's USB generation from its stated per-port signalling rate (USB-IF: 5, 10, 20 Gbit/s)."""
    if host_entry is None:
        return None
    h = host_entry.data.get('host') or {}
    sm = _num(h.get('stated_mbps'))
    if sm is not None and h.get('type') in ('usb_controller', 'compute'):
        for g in ('gen2x2', 'gen2', 'gen1'):
            if abs(sm - SIGNALLING_MBPS[g]) < 1e-6:
                return g
    return usb_generation(host_entry.data.get('interface'))


def host_usb3_generations(host_entry) -> list:
    """The USB 3 generations a host entry's interface text names explicitly ('USB 3.2 Gen 2 / Gen 1' → ['gen1',
    'gen2']), lowest first; empty when it names none. A host with ports of several generations (the Jetson AGX Orin
    developer kit has Gen 2 and Gen 1 ports) cannot give one controller's generation."""
    if host_entry is None:
        return []
    t = str(host_entry.data.get('interface') or '').lower()
    found = set()
    for m in re.finditer(r'gen\s*([12])(\s*x\s*2)?\b', t):
        found.add('gen2x2' if m.group(1) == '2' and m.group(2) else f'gen{m.group(1)}')
    return sorted(found, key=GEN_RANK.get)


# ====================================================================== streams
def stream_id(stream: dict, fps) -> str:
    """'imager|WxH|fps|format' (format '' when the maker states none)."""
    w, h = int(stream['width']), int(stream['height'])
    return f'{stream["imager"]}|{w}x{h}|{float(fps):g}|{stream.get("format") or ""}'


def parse_stream_id(sid: str):
    """(imager, width, height, fps, format) or None when it is not a stream id."""
    parts = str(sid or '').split('|')
    if len(parts) != 4:
        return None
    m = re.fullmatch(r'\s*(\d+)\s*x\s*(\d+)\s*', parts[1])
    fps = _pos(parts[2])
    if not m or fps is None or not parts[0].strip():
        return None
    return parts[0].strip(), int(m.group(1)), int(m.group(2)), fps, parts[3].strip()


def _groups(entry) -> dict:
    """{normalised imager label: [streams]} in the entry's order."""
    out: dict = {}
    for s in entry.data.get('streams') or []:
        if isinstance(s, dict) and s.get('imager') and s.get('width') and s.get('height'):
            out.setdefault(_norm(s['imager']), []).append(s)
    return out


def _fps_list(s) -> list:
    return [float(f) for f in (s.get('fps') or []) if _pos(f)]


def _nearest_fps(fl, target) -> float | None:
    if not fl:
        return None
    return min(fl, key=lambda f: (abs(f - target), -f))


def default_stream_ids(sensor, entry, default_fps: float) -> list:
    """The streams a sensor sends when its link names none (module docstring)."""
    if entry is None:
        return []
    groups = _groups(entry)
    if not groups:
        return []
    keys = list(groups)
    if sensor.kind == 'depth':
        if 'depth' in groups:
            rgb = next((k for k in keys if k != 'depth' and (entry.imager(k) or {}).get('role') == 'rgb'
                        or k == 'rgb'), None)
            chosen = ['depth'] + ([rgb] if rgb else [])
        else:
            chosen = keys
    else:
        chosen = keys[:1]
    want = tuple(int(v) for v in (getattr(sensor, 'stream', None) or ())[:2])
    out = []
    for k in chosen:
        ss = groups[k]
        same = [s for s in ss if (s['width'], s['height']) == want] if len(want) == 2 else []
        if not same:
            big = max(ss, key=lambda s: (s['width'] * s['height'], s['width']))
            same = [s for s in ss if (s['width'], s['height']) == (big['width'], big['height'])]
        s = same[0]
        fps = _nearest_fps(_fps_list(s), default_fps)
        if fps is not None:
            out.append(stream_id(s, fps))
    return out


def resolve_stream(entry, sid: str):
    """(stream dict, fps, '') for a stream id the entry lists, else (None, None, why)."""
    p = parse_stream_id(sid)
    if p is None:
        return None, None, f'"{sid}" is not a stream id (imager|WxH|fps|format)'
    if entry is None:
        return None, None, 'the sensor has no library device'
    im, w, h, fps, fmt = p
    for s in entry.data.get('streams') or []:
        if (_norm(s.get('imager')) == _norm(im) and int(s.get('width', 0)) == w and int(s.get('height', 0)) == h
                and str(s.get('format') or '') == fmt):
            fl = _fps_list(s)
            if not any(abs(f - fps) < 1e-9 for f in fl):
                return None, None, (f'{im} {w} × {h} {fmt or ""} is listed at {", ".join(f"{f:g}" for f in fl)} fps, '
                                    f'not {fps:g}').replace('  ', ' ')
            return s, fps, ''
    return None, None, f'{entry.title} lists no {im} stream {w} × {h}{" " + fmt if fmt else ""}'


def stream_payload_mbps(width, height, fps, bits_per_pixel):
    """width × height × fps × bits per pixel / 10⁶ (Mbit/s); None when bits per pixel is not stated."""
    if bits_per_pixel is None:
        return None
    return float(width) * float(height) * float(fps) * float(bits_per_pixel) / 1e6


def sensor_streams(sensor, entry, ids, st) -> dict:
    """The streams a sensor sends: {'streams': [...], 'default': bool, 'payload_mbps' (None when any stream's rate is
    not stated and no measured figure is given), 'rate_source', 'not_stated': [stream labels], 'problems'}."""
    default = not ids
    if default:
        ids = default_stream_ids(sensor, entry, st['default_fps'])
    rows, problems, missing = [], [], []
    for sid in ids:
        s, fps, why = resolve_stream(entry, sid)
        if s is None:
            problems.append(why)
            continue
        bpp = s.get('bits_per_pixel')
        mb = stream_payload_mbps(s['width'], s['height'], fps, bpp)
        label = f'{s["imager"]} {s["width"]} × {s["height"]} @ {fps:g} fps' + (f' {s["format"]}' if s.get('format')
                                                                                else '')
        if mb is None:
            missing.append(label)
        rows.append({'id': stream_id(s, fps), 'label': label, 'imager': s['imager'], 'width': int(s['width']),
                     'height': int(s['height']), 'fps': fps, 'format': s.get('format'), 'bits_per_pixel': bpp,
                     'bytes_per_pixel': None if bpp is None else bpp / 8.0, 'payload_mbps': mb,
                     'default': default})
    for e in st['extra'].get(sensor.uid, []):
        rows.append({'id': f'extra|{e["name"]}', 'label': f'{e["name"]} (your figure'
                     + (f': {e["source"]}' if e['source'] else ', no source given') + ')', 'imager': e['name'],
                     'width': None, 'height': None, 'fps': None, 'format': None, 'bits_per_pixel': None,
                     'bytes_per_pixel': None, 'payload_mbps': e['mbps'], 'default': False, 'extra': True})
    meas = st['measured'].get(sensor.uid)
    if meas is not None:
        total, src = meas['mbps'], 'measured' + (f': {meas["source"]}' if meas['source'] else ' (no source given)')
    elif entry is None:
        total, src = None, 'no device chosen'
    elif missing or not rows:
        total, src = None, 'not stated'
    else:
        total, src = sum(r['payload_mbps'] for r in rows), 'computed from the stated stream modes'
    return {'streams': rows, 'default': default, 'payload_mbps': total, 'rate_source': src, 'not_stated': missing,
            'problems': problems}


# ====================================================================== topology
def _index(rig):
    ctrl = {}
    for h in rig.get('hosts') or []:
        for c in h.get('controllers') or []:
            ctrl[c['uid']] = (h, c)
    hubs = {b['uid']: b for b in rig.get('hubs') or []}
    return ctrl, hubs


def root_of(uid, ctrl, hubs):
    """(controller uid or None, [hub uids from the node up], problem text). A loop or a dangling uid is a problem."""
    path, seen, cur = [], set(), uid
    while True:
        if cur in ctrl:
            return cur, path, ''
        b = hubs.get(cur)
        if b is None:
            return None, path, (f'"{cur}" is not a controller or hub of the rig' if cur else 'plugged into nothing')
        if cur in seen:
            return None, path, 'the hubs form a loop'
        seen.add(cur)
        path.append(cur)
        cur = b.get('upstream') or ''


# ====================================================================== the budget
def hub_label(name: str) -> str:
    """'Hub 1' for a hub named '1' or 'Hub 1'."""
    name = str(name or '').strip()
    return name if name.lower().startswith('hub') else f'Hub {name}'.strip()


def _finding(out, sev, kind, message, uids=()):
    out.append({'severity': sev, 'kind': kind, 'message': message, 'uids': [u for u in uids if u]})


def _status(load, warn):
    if load is None:
        return 'unknown'
    return 'critical' if load > 1.0 + 1e-12 else 'warning' if load > warn + 1e-12 else 'good'


def _fmt_mbps(x) -> str:
    return '—' if x is None else f'{x:,.0f} Mbit/s' if x >= 100 else f'{x:,.1f} Mbit/s'


def budget(project, settings: dict | None = None) -> dict:
    """The study's result (module docstring) for the active variant."""
    from .. import library as LB
    st = settings or settings_of(project)
    rig = project.rig if isinstance(getattr(project, 'rig', None), dict) else {}
    ctrl, hubs = _index(rig)
    findings: list = []
    notes: list = []
    var = project.variant
    sensors = [s for s in var.sensors if s.enabled]
    off = [s for s in var.sensors if not s.enabled]
    if off:
        notes.append(f'{len(off)} sensor{"s" if len(off) != 1 else ""} switched off ({", ".join(s.name for s in off)}) '
                     'left out.')
    in_var = {s.uid for s in var.sensors}
    all_uids = {s.uid for v in project.variants for s in v.sensors}
    links: dict = {}
    for k in rig.get('links') or []:
        u = k['sensor_uid']
        if u not in all_uids:
            notes.append(f'A link names a sensor that is in no variant ({u}); it is ignored.')
            continue
        if u not in in_var:
            continue                                   # another variant's sensor: its own link
        if u in links:
            s = var.sensor(u)
            _finding(findings, 'warning', 'topology', f'{s.name} has more than one link; only the first is used.', [u])
            continue
        links[u] = k
    # ---- per sensor: streams, rate, link
    rows = []
    for s in sensors:
        entry = LB.get(s.model_id) if s.model_id else None
        link = links.get(s.uid)
        ss = sensor_streams(s, entry, (link or {}).get('stream_ids') or [], st)
        bus = device_bus(entry) if entry is not None else 'other'
        ov = st['overhead'].get(bus)
        overhead = ov['value'] if ov else DEFAULT_OVERHEAD[bus if bus in DEFAULT_OVERHEAD else 'other']
        osrc = (f'your figure: {ov["source"] or "no source given"}' if ov else
                OVERHEAD_WHY[bus if bus in OVERHEAD_WHY else 'other'])
        pay = ss['payload_mbps']
        r = {'uid': s.uid, 'name': s.name, 'kind': s.kind, 'slot': s.slot, 'model_id': s.model_id,
             'title': entry.title if entry is not None else 'no device chosen',
             'interface': (entry.data.get('interface') if entry is not None else '') or '',
             'bus': bus, 'usb_generation': usb_generation(entry.data.get('interface')) if entry is not None else None,
             'streams': ss['streams'], 'default_streams': ss['default'], 'rate_source': ss['rate_source'],
             'not_stated': ss['not_stated'], 'stream_problems': ss['problems'],
             'payload_mbps': pay, 'overhead': overhead, 'overhead_source': osrc,
             'wire_mbps': None if pay is None else pay * (1.0 + overhead),
             'linked': link is not None, 'via': (link or {}).get('via', ''),
             'cable_uid': (link or {}).get('cable_uid', ''),
             'controller_uid': None, 'hub_path': [], 'link_capacity_mbps': None, 'link_capacity_source': '',
             'link_load': None, 'link_margin_mbps': None, 'link_status': 'unknown',
             'power_w': _num(entry.data.get('power_w')) if entry is not None else None,
             'draw_ma': None, 'allowance_ma': None, 'power_status': 'unknown', 'power_note': ''}
        for why in ss['problems']:
            _finding(findings, 'warning', 'stream', f'{s.name}: a stream on its link is not used: {why}.', [s.uid])
        if link is not None:
            root, path, prob = root_of(link['via'], ctrl, hubs)
            if prob:
                _finding(findings, 'serious', 'topology', f'{s.name} is plugged into something the rig does not have '
                                                          f'({prob}): plug it into a controller or hub again.', [s.uid])
            r['controller_uid'], r['hub_path'] = root, path
        rows.append(r)
    # ---- controllers
    host_entries = {}
    for h in rig.get('hosts') or []:
        host_entries[h['uid']] = LB.get(h.get('library_id')) if h.get('library_id') else None
    crow = []
    for cu, (h, c) in ctrl.items():
        he = host_entries.get(h['uid'])
        gen = None
        if c['kind'] == 'usb2':
            gen = 'usb2'
        elif c['kind'] == 'usb3':
            gen = host_generation(he)
            named = host_usb3_generations(he)
            if gen in (None, 'usb2'):
                gen = 'gen1'
                gen_note = ('generation not known from the host' + (' entry' if he is not None else ' (no library '
                            'entry)') + ': Gen 1, the lowest, is assumed')
            elif len(named) > 1:
                gen = named[0]
                gen_note = (f'the host entry ({he.title}) has USB 3 ports of ' + ' and '.join(
                    GEN_TITLE[g] for g in reversed(named)) + f': {GEN_TITLE[gen]}, the lowest, is assumed; enter this '
                    'controller\'s practical bandwidth with its source if it serves the faster ports')
            else:
                gen_note = f'{GEN_TITLE[gen]} from the host entry ({he.title})'
        if c['kind'] == 'usb2':
            gen_note = 'USB 2.0'
        elif c['kind'] in ('pcie', 'gmsl'):
            gen_note = ''
        if c.get('practical_mbps'):
            cap = float(c['practical_mbps'])
            csrc = 'stated: ' + (c.get('source') or 'your figure, no source given')
            stated = True
        elif gen is not None:
            cap, csrc, stated = CAPACITY_MBPS[gen], CAPACITY_WHY[gen], False
        else:
            cap, csrc, stated = None, ('not stated: enter the practical bandwidth with its source (the GMSL2 link '
                                       'rate is in the serialiser and deserialiser datasheets)' if c['kind'] == 'gmsl'
                                       else 'not stated: enter the practical bandwidth with its source'), False
        under = [r for r in rows if r['controller_uid'] == cu]
        known = [r['wire_mbps'] for r in under if r['wire_mbps'] is not None]
        load_mbps = sum(known)
        incomplete = len(known) < len(under)
        load = load_mbps / cap if cap else None
        crow.append({'uid': cu, 'name': c['name'], 'host_uid': h['uid'], 'host_name': h['name'],
                     'label': f'{h["name"]} · {c["name"]}', 'kind': c['kind'], 'generation': gen,
                     'generation_note': gen_note, 'capacity_mbps': cap, 'capacity_source': csrc,
                     'capacity_stated': stated, 'load_mbps': load_mbps, 'margin_mbps': None if cap is None else
                     cap - load_mbps, 'load': load, 'status': _status(load, st['margin_warn']),
                     'incomplete': incomplete, 'sensors': [r['uid'] for r in under],
                     'direct_sensors': [r['uid'] for r in under if not r['hub_path']],
                     'hubs': [b for b in hubs if root_of(b, ctrl, hubs)[0] == cu]})
    cby = {c['uid']: c for c in crow}
    # ---- links (per sensor cable): the lower of the device's generation and the controller's
    for r in rows:
        c = cby.get(r['controller_uid'])
        if c is None:
            continue
        cbus = controller_bus(c['kind'])
        if r['bus'] in ('usb', 'gmsl', 'pcie') and cbus != r['bus']:
            _finding(findings, 'serious', 'mismatch',
                     f'{r["name"]} has a {r["bus"].upper()} interface ({r["interface"]}) but is plugged into '
                     f'{c["label"]}, a {c["kind"].upper()} controller: it cannot connect there.', [r['uid']])
            continue
        if cbus == 'usb':
            dg = r['usb_generation']
            cg = c['generation']
            g = cg if dg is None else min((dg, cg), key=GEN_RANK.get) if cg else dg
            r['link_capacity_mbps'] = CAPACITY_MBPS[g]
            r['link_capacity_source'] = (f'{GEN_TITLE[g]} (' + ('the device states no USB generation, the '
                                         "controller's is used" if dg is None else
                                         f'device {GEN_TITLE[dg]}, controller {GEN_TITLE[cg]}')
                                         + f'): {CAPACITY_WHY[g]}')
            if c['capacity_stated'] and c['capacity_mbps'] < r['link_capacity_mbps']:
                r['link_capacity_mbps'] = c['capacity_mbps']
                r['link_capacity_source'] = f'the controller\'s stated figure ({c["capacity_source"]})'
        else:
            r['link_capacity_source'] = ('not stated: the per-camera GMSL2 link rate is not in the library'
                                         if cbus == 'gmsl' else 'not stated')
        if r['link_capacity_mbps'] and r['wire_mbps'] is not None:
            r['link_load'] = r['wire_mbps'] / r['link_capacity_mbps']
            r['link_margin_mbps'] = r['link_capacity_mbps'] - r['wire_mbps']
            r['link_status'] = _status(r['link_load'], st['margin_warn'])
    # ---- hubs: uplink load, shared controller
    hrow = []
    for bu, b in hubs.items():
        root, path, prob = root_of(bu, ctrl, hubs)
        below = [r for r in rows if bu in r['hub_path']]
        known = [r['wire_mbps'] for r in below if r['wire_mbps'] is not None]
        c = cby.get(root)
        cap = None
        if c is not None and controller_bus(c['kind']) == 'usb':
            cap = c['capacity_mbps'] if c['capacity_stated'] else CAPACITY_MBPS[c['generation']]
        load_mbps = sum(known)
        load = load_mbps / cap if cap else None
        shared = False
        if c is not None:
            mine = {r['uid'] for r in below}
            others = [u for u in c['sensors'] if u not in mine]
            other_hubs = [h for h in c['hubs'] if h != bu and bu not in root_of(h, ctrl, hubs)[1]
                          and h not in path]
            shared = bool(others or other_hubs)
        hrow.append({'uid': bu, 'name': b['name'], 'powered': b['powered'], 'upstream': b['upstream'],
                     'controller_uid': root, 'problem': prob, 'sensors': [r['uid'] for r in below],
                     'load_mbps': load_mbps, 'uplink_capacity_mbps': cap, 'load': load,
                     'status': _status(load, st['margin_warn']), 'incomplete': len(known) < len(below),
                     'shared_controller': shared, 'draw_ma': None, 'available_ma': None, 'power_status': 'unknown'})
        if prob:
            _finding(findings, 'serious', 'topology', f'{hub_label(b["name"])} is plugged into something the rig '
                                                      f'does not have ({prob}).', [bu])
        elif c is not None and controller_bus(c['kind']) != 'usb':
            _finding(findings, 'serious', 'mismatch', f'{hub_label(b["name"])} is a USB hub on {c["label"]}, a '
                                                      f'{c["kind"].upper()} controller.', [bu])
    _power(rows, crow, hrow, hubs, cby, st, findings)
    disk = _disk(rows, st, findings)
    _bandwidth_findings(rows, crow, hrow, st, findings, ctrl)
    order = {s: i for i, s in enumerate(SEVERITIES)}
    findings.sort(key=lambda f: order.get(f['severity'], 9))
    worst = next((f['severity'] for f in findings if f['severity'] in ('critical', 'serious', 'warning')), 'good')
    loads = [(c['load'], c['label'], 'controller') for c in crow if c['load'] is not None]
    loads += [(r['link_load'], f'{r["name"]}\'s link', 'link') for r in rows if r['link_load'] is not None]
    loads += [(h['load'], hub_label(h['name']), 'hub') for h in hrow if h['load'] is not None]
    busiest = max(loads, default=None)
    incomplete = [r['uid'] for r in rows if r['payload_mbps'] is None]
    word = WORDS.get(worst, worst)
    if worst == 'warning' and (busiest is None or busiest[0] <= st['margin_warn'] + 1e-12):
        word = 'see findings'                          # a warning that is not a load above the margin
    if not crow:
        word = 'no hosts'
    elif incomplete and worst == 'good':
        worst, word = 'warning', 'incomplete'
    res = {
        'variant': var.name, 'variant_uid': var.uid,
        'units': 'decimal: Mbit/s = 1e6 bit/s, MB = 1e6 bytes, GB = 1e9 bytes; current mA at the nominal 5 V',
        'settings': {k: v for k, v in st.items()},
        'sensors': rows, 'controllers': crow, 'hubs': hrow, 'disk': disk,
        'power': {'rows': [{'uid': r['uid'], 'name': r['name'], 'draw_ma': r['draw_ma'],
                            'allowance_ma': r['allowance_ma'], 'status': r['power_status'], 'note': r['power_note']}
                           for r in rows],
                  'hubs': [{'uid': h['uid'], 'name': h['name'], 'powered': h['powered'], 'draw_ma': h['draw_ma'],
                            'available_ma': h['available_ma'], 'status': h['power_status']} for h in hrow],
                  'vbus_nominal_v': VBUS_NOMINAL_V},
        'findings': findings, 'notes': notes, 'assumptions': list(ASSUMPTIONS),
        'references': [REF_USB32, REF_USB20],
        'headline': {'busiest_load_pct': None if busiest is None else _r(busiest[0] * 100, 2),
                     'busiest': None if busiest is None else busiest[1],
                     'busiest_kind': None if busiest is None else busiest[2],
                     'status': worst, 'status_word': word, 'incomplete_sensors': incomplete,
                     'unassigned_sensors': [r['uid'] for r in rows if not r['linked']],
                     'total_payload_mbps': _r(sum(r['payload_mbps'] or 0.0 for r in rows), 3)},
    }
    res['suggestion'] = suggest(res, project)
    return res


def _power(rows, crow, hrow, hubs, cby, st, findings):
    """USB bus power (module docstring), filling draw_ma / allowance_ma / power_status on rows and hubs."""
    ext = set(st['external_power'])
    hby = {h['uid']: h for h in hrow}

    def flavour(c, r=None):
        if c['kind'] == 'usb2' or (r is not None and r['usb_generation'] == 'usb2'):
            return 'usb2'
        two = c['generation'] == 'gen2x2' and (r is None or r['usb_generation'] == 'gen2x2')
        return 'usb3x2' if two else 'usb3'

    def high(fl):
        return UNIT_LOAD_MA[fl] * HIGH_POWER_UNITS[fl]

    for r in rows:
        c = cby.get(r['controller_uid'])
        if r['uid'] in ext:
            r['power_status'], r['power_note'] = 'good', 'its own supply (settings external_power): not counted'
            r['draw_ma'] = 0.0
            continue
        if c is None or controller_bus(c['kind']) != 'usb' or r['bus'] != 'usb':
            r['power_note'] = ('not checked: not on a USB controller' if c is not None else
                               'not checked: not assigned to a controller')
            continue
        if r['power_w'] is None:
            r['power_note'] = 'not stated by the maker: the draw is not checked'
            _finding(findings, 'note', 'power', f'{r["name"]}: its maker states no power figure, so its bus power is '
                                                'not checked.', [r['uid']])
            continue
        r['draw_ma'] = r['power_w'] / VBUS_NOMINAL_V * 1000.0
    for h in hrow:                                     # bus-powered hubs draw what is below them, deepest first
        h['draw_ma'] = 0.0 if h['powered'] else None

    def hub_draw(bu, seen=()):
        if bu in seen:
            return 0.0
        h = hby[bu]
        if h['powered']:
            return 0.0                                 # a self-powered hub may draw one unit load: not stated, 0
        tot = 0.0
        for r in rows:
            if r['via'] == bu and r['draw_ma'] is not None:
                tot += r['draw_ma']
        for b2, hb in hubs.items():
            if hb.get('upstream') == bu and b2 in hby:
                tot += hub_draw(b2, seen + (bu,))
        h['draw_ma'] = tot
        return tot

    for bu in hby:
        hub_draw(bu)

    def allowance(via, c, r=None):
        """The current available at the port 'via' feeds (a controller's root port or a hub's port)."""
        if via in cby:
            return high(flavour(c, r)), 'a root port: a high-power port'
        h = hby.get(via)
        if h is None:
            return None, ''
        if h['powered']:
            return high(flavour(c, r)), f'a port of powered hub ({hub_label(h["name"])})'
        if flavour(c) == 'usb2':
            return UNIT_LOAD_MA['usb2'], f'a port of bus-powered USB 2.0 hub ({hub_label(h["name"])}): one unit load'
        return None, f'a port of bus-powered hub ({hub_label(h["name"])}), which shares its own allowance'

    for h in hrow:
        c = cby.get(h['controller_uid'])
        if c is None or controller_bus(c['kind']) != 'usb' or h['problem']:
            continue
        a, _why = allowance(h['upstream'], c)
        if h['powered']:
            h['power_status'] = 'good'
            continue
        if a is None:                                  # a bus-powered hub below a bus-powered USB 3 hub: its draw
            h['power_status'] = 'shared'               # counts in that hub's, which is checked
            continue
        h['available_ma'] = a
        if a is not None and h['draw_ma'] is not None:
            h['power_status'] = 'critical' if h['draw_ma'] > a + 1e-9 else 'good'
            if h['power_status'] == 'critical':
                _finding(findings, 'critical', 'power',
                         f'Bus-powered {hub_label(h["name"])}: the devices on it state {h["draw_ma"]:,.0f} mA, '
                         f'more than the {a:,.0f} mA its upstream port gives (its own electronics not counted). '
                         'Use a powered hub or move a device to another port.', [h['uid']] + h['sensors'])
    for r in rows:
        c = cby.get(r['controller_uid'])
        if r['draw_ma'] is None or c is None or r['uid'] in ext:
            continue
        a, why = allowance(r['via'], c, r)
        if a is None:
            h = hby.get(r['via'])
            r['allowance_ma'] = None if h is None else h['available_ma']
            r['power_status'] = 'good' if h is None or h['power_status'] != 'critical' else 'critical'
            r['power_note'] = why
            continue
        r['allowance_ma'] = a
        r['power_note'] = why
        if r['draw_ma'] > a + 1e-9:
            r['power_status'] = 'critical'
            _finding(findings, 'critical', 'power',
                     f'{r["name"]} states {r["power_w"]:g} W, {r["draw_ma"]:,.0f} mA at 5 V, more than the '
                     f'{a:,.0f} mA of {why}. Give it its own supply, or a powered hub.', [r['uid']])
        else:
            r['power_status'] = 'good'
    checked = [r for r in rows if r['allowance_ma'] and r['draw_ma'] is not None and r['uid'] not in ext]
    if checked and all(r['power_status'] == 'good' for r in checked):
        b = max(checked, key=lambda r: r['draw_ma'] / r['allowance_ma'])
        _finding(findings, 'good', 'power', f'Every bus-powered USB device is within its port\'s current: the '
                                            f'largest share, {b["name"]}, states {b["draw_ma"]:,.0f} of '
                                            f'{b["allowance_ma"]:,.0f} mA.')


def _disk(rows, st, findings) -> dict:
    recorded = [r for r in rows if r['payload_mbps'] is not None]
    payload = sum(r['payload_mbps'] for r in recorded)
    mbps = payload * st['record_ratio'] * (1.0 + st['container_overhead'])
    mb_s = mbps / 8.0
    ep_gb = mb_s * st['episode_s'] / 1000.0
    camp_gb = ep_gb * st['episodes']
    rec = st['recorder']
    need = mb_s * st['write_headroom']
    out = {'payload_mbps': payload, 'recorded_mbps': mbps, 'recorded_mb_s': mb_s, 'gb_per_hour': mb_s * 3.6,
           'episode_s': st['episode_s'], 'episodes': st['episodes'], 'gb_per_episode': ep_gb,
           'gb_per_campaign': camp_gb, 'tb_per_campaign': camp_gb / 1000.0, 'write_needed_mb_s': need,
           'write_mb_s': rec['write_mb_s'], 'write_source': rec['source'], 'recorder': rec['name'],
           'write_load': None if rec['write_mb_s'] is None else need / rec['write_mb_s'],
           'capacity_gb': rec['capacity_gb'],
           'episodes_per_disk': (None if not rec['capacity_gb'] or ep_gb <= 0 else
                                 math.floor(rec['capacity_gb'] / ep_gb)),
           'incomplete': [r['uid'] for r in rows if r['payload_mbps'] is None]}
    if not rows:
        return out
    what = (f'{mb_s:,.1f} MB/s recorded: {ep_gb:,.2f} GB per {st["episode_s"]:g} s episode, '
            f'{camp_gb / 1000:,.2f} TB per campaign of {st["episodes"]:,} episodes')
    if rec['write_mb_s'] is None:
        _finding(findings, 'note', 'disk', f'{what}. The recorder\'s sustained write speed is not stated: enter it '
                                           'with its source to check the disk keeps up.')
    elif need > rec['write_mb_s'] + 1e-9:
        _finding(findings, 'critical', 'disk', f'The recorder cannot keep up: {mb_s:,.1f} MB/s × '
                                               f'{st["write_headroom"]:g} headroom needs {need:,.1f} MB/s sustained, '
                                               'it writes '
                                               f'{rec["write_mb_s"]:,.0f} MB/s. Use a faster disk, compress, or lower '
                                               'a frame rate.')
    else:
        _finding(findings, 'good', 'disk', f'{what}; the recorder keeps up ({need:,.1f} of '
                                           f'{rec["write_mb_s"]:,.0f} MB/s sustained write needed).')
    if out['episodes_per_disk'] is not None and out['episodes_per_disk'] < st['episodes']:
        _finding(findings, 'warning', 'disk', f'One {rec["capacity_gb"]:,.0f} GB disk holds '
                                              f'{out["episodes_per_disk"]:,} episodes, fewer than the '
                                              f'campaign\'s {st["episodes"]:,}.')
    return out


def _bandwidth_findings(rows, crow, hrow, st, findings, ctrl):
    warn = st['margin_warn'] * 100
    for c in crow:
        names = ', '.join(r['name'] for r in rows if r['uid'] in c['sensors']) or 'nothing'
        if c['capacity_mbps'] is None:
            if c['sensors']:
                _finding(findings, 'warning', 'controller', f'{c["label"]} carries {_fmt_mbps(c["load_mbps"])} '
                                                            f'({names}), but its bandwidth is {c["capacity_source"]}.',
                         [c['uid']])
            continue
        if c['status'] == 'critical':
            _finding(findings, 'critical', 'controller',
                     f'{c["label"]} is over budget: {_fmt_mbps(c["load_mbps"])} of {_fmt_mbps(c["capacity_mbps"])} '
                     f'({c["load"] * 100:.0f} %) from {names}. Move a sensor to another controller or lower a '
                     'resolution or frame rate.', [c['uid']] + c['sensors'])
        elif c['status'] == 'warning':
            _finding(findings, 'warning', 'controller',
                     f'{c["label"]} is tight: {c["load"] * 100:.0f} % of {_fmt_mbps(c["capacity_mbps"])}, above the '
                     f'{warn:.0f} % margin ({_fmt_mbps(c["margin_mbps"])} left).', [c['uid']] + c['sensors'])
        if c['incomplete']:
            _finding(findings, 'warning', 'controller', f'{c["label"]}: some sensors on it have no stated data rate, '
                                                        'so its load is a lower bound.', [c['uid']])
    for r in rows:
        if r['link_status'] == 'critical':
            what = ' + '.join(s['imager'] for s in r['streams']) or 'its streams'
            _finding(findings, 'critical', 'link',
                     f'{r["name"]}\'s link is over its limit: {what} send {_fmt_mbps(r["wire_mbps"])} over '
                     f'{_fmt_mbps(r["link_capacity_mbps"])} ({r["link_load"] * 100:.0f} %). Lower a resolution or '
                     'frame rate, or drop a stream.', [r['uid']])
        elif r['link_status'] == 'warning':
            _finding(findings, 'warning', 'link',
                     f'{r["name"]}\'s link is tight: {r["link_load"] * 100:.0f} % of '
                     f'{_fmt_mbps(r["link_capacity_mbps"])}, above the {warn:.0f} % margin.', [r['uid']])
    for h in hrow:
        if h['status'] == 'critical':
            _finding(findings, 'critical', 'hub', f'{hub_label(h["name"])}\'s uplink is over its limit: '
                                                  f'{_fmt_mbps(h["load_mbps"])} of '
                                                  f'{_fmt_mbps(h["uplink_capacity_mbps"])}.',
                     [h['uid']] + h['sensors'])
        if h['shared_controller']:
            _finding(findings, 'warning', 'hub', f'{hub_label(h["name"])} is on a shared controller: everything on '
                                                 'it shares the controller\'s bandwidth with the other devices there. '
                                                 'Give the hub a controller of its own if the margin is small.',
                     [h['uid']])
    loose = [r for r in rows if not r['linked']]
    if loose and ctrl:
        _finding(findings, 'warning', 'unassigned', f'Not plugged into any controller: '
                                                    f'{", ".join(r["name"] for r in loose)}. Assign '
                                                    f'{"them" if len(loose) > 1 else "it"} on the topology editor.',
                 [r['uid'] for r in loose])
    elif loose:
        _finding(findings, 'warning', 'unassigned', 'The rig has no host yet: add one (or Apply the suggestion) to '
                                                    'check the controllers, links and bus power.')
    for r in rows:
        if r['payload_mbps'] is None:
            why = ('it has no library device' if r['rate_source'] == 'no device chosen' else
                   'the maker states no bits per pixel for ' + (', '.join(r['not_stated']) or 'any stream'))
            _finding(findings, 'warning', 'rate', f'{r["name"]}: no data rate ({why}). Enter a measured rate for it.',
                     [r['uid']])
    known = [c for c in crow if c['load'] is not None]
    if known and all(c['status'] == 'good' for c in known):
        b = max(known, key=lambda c: c['load'])
        _finding(findings, 'good', 'controller', f'Every controller is under {warn:.0f} %: the busiest, {b["label"]}, '
                                                 f'carries {_fmt_mbps(b["load_mbps"])} of '
                                                 f'{_fmt_mbps(b["capacity_mbps"])} ({b["load"] * 100:.0f} %).')


# ====================================================================== the suggestion (Apply)
def suggest(res: dict, project=None) -> dict | None:
    """The change Apply makes: {'kind', 'text', 'moves': [{'sensor_uid', 'to'}], 'add_host': bool}.

    - no host: add one host (no library entry) with one USB 3 controller, labelled an assumption, and plug every USB
      sensor into it;
    - sensors not assigned: plug each (largest data rate first) into the compatible controller with the most room;
    - a controller over the margin: move the one sensor whose move brings the busiest load lowest, when that helps."""
    rows, crow = res['sensors'], res['controllers']
    warn = res['settings']['margin_warn']
    if not rows:
        return None
    if not crow:
        usb = [r for r in rows if r['bus'] == 'usb']
        if not usb:
            return None
        return {'kind': 'add_host', 'add_host': True, 'moves': [{'sensor_uid': r['uid'], 'to': ''} for r in usb],
                'text': ('Add a host with one USB 3 controller (an assumption: set your machine\'s controllers) and '
                         f'plug {", ".join(r["name"] for r in usb)} into it.')}
    room = {c['uid']: (c['capacity_mbps'] - c['load_mbps']) if c['capacity_mbps'] else None for c in crow}
    cb = {c['uid']: c for c in crow}

    def fits(r, c):
        return r['bus'] == controller_bus(c['kind']) or (r['bus'] == 'other' and controller_bus(c['kind']) == 'usb'
                                                        and r['model_id'] == '')
    loose = sorted([r for r in rows if not r['linked']], key=lambda r: -(r['wire_mbps'] or 0.0))
    moves = []
    for r in loose:
        cands = [c for c in crow if fits(r, c)]
        if not cands:
            continue
        c = max(cands, key=lambda c: (room[c['uid']] is not None, room[c['uid']] or 0.0))
        moves.append({'sensor_uid': r['uid'], 'to': c['uid']})
        if room[c['uid']] is not None:
            room[c['uid']] -= r['wire_mbps'] or 0.0
    if moves:
        names = ', '.join(f'{next(r["name"] for r in rows if r["uid"] == m["sensor_uid"])} into '
                          f'{cb[m["to"]]["label"]}' for m in moves)
        return {'kind': 'assign', 'add_host': False, 'moves': moves, 'text': f'Plug {names}.'}
    over = [c for c in crow if c['load'] is not None and c['load'] > warn + 1e-12]
    if not over:
        return None
    busy = max(over, key=lambda c: c['load'])
    best = None
    for r in rows:
        if r['controller_uid'] != busy['uid'] or r['wire_mbps'] is None:
            continue
        for c in crow:
            if c['uid'] == busy['uid'] or not c['capacity_mbps'] or not fits(r, c):
                continue
            new_busy = (busy['load_mbps'] - r['wire_mbps']) / busy['capacity_mbps']
            new_c = (c['load_mbps'] + r['wire_mbps']) / c['capacity_mbps']
            peak = max(new_busy, new_c)
            if peak < busy['load'] - 1e-9 and (best is None or peak < best[0]):
                best = (peak, r, c)
    if best is None:
        return None
    peak, r, c = best
    return {'kind': 'move', 'add_host': False, 'moves': [{'sensor_uid': r['uid'], 'to': c['uid']}],
            'text': (f'Move {r["name"]} from {busy["label"]} to {c["label"]}: the busiest load falls from '
                     f'{busy["load"] * 100:.0f} % to {peak * 100:.0f} %.')}


def apply_suggestion(project, sug: dict) -> None:
    """Make a suggestion's change on project.rig (the page wraps it in one undo step)."""
    from ..project import new_uid, rig_from_dict
    rig = copy.deepcopy(rig_from_dict(project.rig))
    if sug.get('add_host'):
        cu = new_uid()
        rig['hosts'].append({'uid': new_uid(), 'name': 'Host (assumed)', 'library_id': '',
                             'controllers': [{'uid': cu, 'name': 'USB 3 controller (assumed)', 'kind': 'usb3',
                                              'practical_mbps': None, 'source': ''}]})
        moves = [dict(m, to=cu) for m in sug.get('moves') or []]
    else:
        moves = sug.get('moves') or []
    for m in moves:
        link = next((k for k in rig['links'] if k['sensor_uid'] == m['sensor_uid']), None)
        if link is None:
            rig['links'].append({'sensor_uid': m['sensor_uid'], 'stream_ids': [], 'via': m['to'], 'cable_uid': ''})
        else:
            link['via'] = m['to']
    project.rig = rig_from_dict(rig)


# ====================================================================== the study
def _jsonable(doc):
    """Round floats for the result and drop non-finite numbers (None)."""
    if isinstance(doc, dict):
        return {k: _jsonable(v) for k, v in doc.items()}
    if isinstance(doc, (list, tuple)):
        return [_jsonable(v) for v in doc]
    if isinstance(doc, float):
        return doc if math.isfinite(doc) else None
    return doc


@jobs.study('bandwidth', 'Bandwidth, hosts and power', batch='bandwidth', needs_robot=False)
def run(project, ctx: jobs.Ctx) -> dict:
    """Bandwidth, hosts and power of the active variant (module docstring)."""
    ctx.report(0.0, 'Data rates')
    res = budget(project)
    if not project.variant.sensors:                # a live study: an empty rig is a state, not an error
        res['headline']['status_word'] = 'no sensors'
        res['message'] = ('There is no sensor in this rig: add one (Sensors ▸ Add camera, or double-click a device in '
                          'the Library) and choose its device.')
    ctx.report(1.0, '')
    return _jsonable(res)
