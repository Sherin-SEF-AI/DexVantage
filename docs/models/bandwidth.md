# Bandwidth, hosts and power {#bandwidth}

Module: `dexvantage/core/studies/bandwidth.py` (study `bandwidth`, batch flag `--bandwidth OUT.json`). Page:
`dexvantage/ui/pages/bandwidth.py` (Analysis ▸ Bandwidth, hosts and power). Checks:
`dexvantage/validation/checks_bandwidth.py` (model `bandwidth`). Tests: `tests/test_bandwidth.py`. Spec §7.6 and §6
(Hosts); the bandwidth checks of §10.

Ported from Vantage `core/recording.py` (data-rate rows, the disk budget, link loads and their findings) and
`core/power.py` (devices against what feeds them, findings worst first). Vantage's GMSL ECU, lidars, radars, DC-DC
converter, alternator, fuses and power cables were stripped; the host topology and USB bus power are new.

The study is cheap (no geometry, a few milliseconds) and runs live after every edit.

## The host topology (`project.rig`)

A project section of its own (`core/project.py` `rig_from_dict`; schema `docs/project.schema.json`; one undo step per
edit through the `rig` section):

```json
{"hosts": [{"uid": "…", "name": "Workstation", "library_id": "usb-3.2-gen2-host",
            "controllers": [{"uid": "…", "name": "USB 3 controller 1", "kind": "usb3",
                             "practical_mbps": null, "source": ""}]}],
 "hubs":  [{"uid": "…", "name": "Hub 1", "powered": false, "upstream": "<controller or hub uid>"}],
 "links": [{"sensor_uid": "…", "stream_ids": ["depth|848x480|30|Z16"], "via": "<controller or hub uid>",
            "cable_uid": ""}]}
```

- `kind` is `usb3`, `usb2`, `pcie` or `gmsl`. `practical_mbps` is the controller's practical bandwidth when a source
  states it (a measurement, a board manual), with the source in `source`; `null` means not stated.
- A link is one sensor cable. `stream_ids` name the streams it carries as `imager|WxH|fps|format`, each a stream mode
  of the sensor's library entry at one of its listed rates; empty means the default streams (below).
- The rig is project-wide; links name sensors by uid, so each variant's sensors keep their own links. A sensor of the
  active variant with no link is "not assigned": its data counts for the disk, not for a controller.
- The content hash (`core/provenance.py`) includes the rig only when it holds something, so projects without one keep
  their hashes.

## Data rate of a stream

Decimal units throughout (Mbit/s = 10⁶ bit/s, MB = 10⁶ bytes, GB = 10⁹ bytes), as links and disks are sold.

    payload  = width × height × fps × bits per pixel / 10⁶          (bytes per pixel = bits per pixel / 8)
    on wire  = payload × (1 + overhead)

Width, height, the listed frame rates and the bits per pixel come from the library entry's `streams` (each entry cites
its datasheet). A stream whose bits per pixel the maker does not state has **no data rate**: MJPEG streams, the
Luxonis sensor modes (the format on the link depends on the on-device pipeline), the DIGIT tactile camera, and the
GelSight Mini (no stream listed at all). The study says so, the budget is marked incomplete (a controller's load is
then a lower bound), and a measured figure can be entered per sensor:

    project.analysis.studies['bandwidth']['measured'] = {sensor_uid: {'mbps': 74.0, 'source': 'usbtop, 2026-11-06'}}

Streams the library does not list (an IMU, audio) are added per sensor in `extra` (`[{'name', 'mbps', 'source'}]`).
Tactile sensors are counted like any other sensor.

**Protocol overhead.** No maker in the library states the share its class protocol adds (video class payload headers,
frame gaps; CSI-2 packet headers and blanking over GMSL). It is a **labelled default per interface, 5 %**
(`DEFAULT_OVERHEAD`, reasoning in `OVERHEAD_WHY`): for USB, the capacity figure below already takes off the link's
flow control, framing and protocol overheads, so the 5 % stands only for the class protocol on top. A user figure with
its source (`overhead {bus: {'value', 'source'}}`) replaces it.

**Default streams.** A depth sensor sends its `depth` stream and its RGB stream when its entry lists them (it computes
depth on board); a depth sensor without a depth stream (a ZED) sends one stream of every imager; a camera or tactile
sensor its first imager's stream. The size is the sensor's chosen stream (`Sensor.stream`, from the Sensors page) when
that imager streams it, else the imager's largest; the format the first listed at that size; the rate the listed one
nearest `default_fps` (30 fps, a default; ties go to the higher rate). Each such stream is labelled "default".

## Capacity

| Interface | Capacity used without a measured figure | Source |
|---|---|---|
| USB 2.0 high speed (480 Mbit/s) | 425.984 Mbit/s: 13 bulk transactions × 512 bytes per 125 µs microframe = 53 248 000 bytes/s, "does not include the overhead associated with bit stuffing" | USB 2.0, §5.8.4, Table 5-10 |
| USB 3.2 Gen 1x1 (5 Gbit/s; "USB 3.0", "USB 3.1 Gen 1") | 3600 Mbit/s: "the raw throughput is 500 MB/sec. Accounting for flow control, packet framing and protocol overheads reduces the effective bandwidth down to 450 MB/sec or less" | USB 3.2, §4.4.11 |
| USB 3.2 Gen 2x1 (10 Gbit/s) | 8800 Mbit/s: "approximately 1.1 GB/sec or less" | USB 3.2, §4.4.11 |
| USB 3.2 Gen 2x2 (20 Gbit/s) | 17 600 Mbit/s: "The effective bandwidth is doubled in Gen 2x2 operation" | USB 3.2, §4.4.11 |
| PCIe, GMSL | not stated: not checked until you enter a figure | the GMSL2 link rate is in the serialiser and deserialiser datasheets (ADI MAX96717 / MAX96712), which could not be fetched |

Each is an **upper bound**: the specification says "or less", and real controllers, drivers and hub trees often fall
short; several ports can share one controller, which makers rarely state (`lsusb -t` shows it). A measured
`practical_mbps` always replaces it.

- A `usb3` controller's generation comes from its host's library entry (its stated per-port signalling rate, 5000,
  10 000 or 20 000 Mbit/s per USB-IF); without one Gen 1, the lowest, is assumed and the result says so.
  A host whose interface names USB 3 ports of several generations (the Jetson AGX Orin developer kit's "USB 3.2 Gen
  2 / Gen 1") cannot say which one a controller serves: the lowest named is assumed and said; enter the controller's
  practical bandwidth for a faster port. A 10 Gbit/s rate is taken as Gen 2x1: Gen 1x2 (two 5 Gbit/s lanes, "The
  effective bandwidth is doubled in Gen 1x2 operation", 7200 Mbit/s) is not told apart.
- A sensor's **link** carries all its streams (depth + RGB + IMU on one cable). Its capacity is the lower of the
  device's own USB generation (parsed from the entry's `interface`: "USB 3.1 Gen 1" is Gen 1, "USB 3.0" is Gen 1 as
  USB-IF says) and its controller's, or the controller's measured figure when that is lower. A device that states no
  generation ("USB") takes the controller's.
- A **hub's uplink** carries everything below it, at its controller's capacity.
- **Load** = on-wire rate / capacity; **margin** = capacity − on-wire rate (Mbit/s). Above `margin_warn` (80 %, a
  default) the status is "tight" (warning), above 100 % "over budget" (critical).

## Flags

| Finding | Severity |
|---|---|
| A controller, a sensor link or a hub uplink over 100 % | critical |
| … above the 80 % margin | warning |
| A hub on a shared controller: the controller also carries other devices or hubs | warning |
| A USB device on a GMSL or PCIe controller, a GMSL device on a USB one, a USB hub on a non-USB controller | serious |
| A link or hub plugged into a uid the rig does not have; hubs in a loop | serious |
| A sensor linked twice (the first link is used) or a stream id the entry does not list | warning |
| Sensors not assigned; a sensor without a stated data rate; a controller whose load is a lower bound | warning |
| A device or bus-powered hub over its port's current | critical |
| A power figure not stated; the recorder's write speed not stated | note |

The headline is the busiest controller, link or hub uplink as a share of its capacity, with the worst finding's
status word ("within budget", "tight", "problems", "over budget"; "see findings" when the worst finding is a warning
that is not a load above the margin; "incomplete" when everything is fine but some rates are not stated; "no hosts").

## Disk

    recorded (Mbit/s) = Σ payload × record_ratio × (1 + container_overhead)
    per episode (GB)  = recorded / 8 × episode_s / 1000
    per campaign      = per episode × episodes
    write needed      = recorded / 8 × write_headroom   (MB/s, against the recorder's stated sustained write)

`record_ratio` 1.0 (recorded as streamed), `container_overhead` 3 % (message headers and indexes, Vantage's default),
`episode_s` 60 s, `episodes` 100 and `write_headroom` 1.3 (Vantage's default) are labelled defaults. The recorder
(`recorder {'name', 'write_mb_s', 'capacity_gb', 'source'}`) is the user's figure with its source: none of the
library's hosts states a disk, so without it the write check is "not stated". With `capacity_gb`, the episodes that
fit on one disk.

## Bus power

| Port | Current available | Source |
|---|---|---|
| USB 2.0 root port or powered hub port | 500 mA: "a unit load is defined to be 100 mA", a high-power device "up to five unit loads" | USB 2.0 §7.2.1; USB 3.2 Table 3-1 "low (100 mA)/high (500 mA)" |
| USB 3 root port or powered hub port, one lane | 900 mA: "a unit load for Enhanced SuperSpeed for single-lane operation is 150 mA", high-power "up to six unit loads" | USB 3.2 §11.4.1, Table 3-1 "high (900 mA)" |
| USB 3 two-lane (Gen 2x2 device on a Gen 2x2 host) | 1500 mA | USB 3.2 Table 3-1 |
| Port of a bus-powered USB 2.0 hub | 100 mA: "External ports in a bus-powered hub can supply only one unit load per port" | USB 2.0 §7.2.1 |
| Ports of a bus-powered USB 3 hub | shared: the hub supplies "any power not used by the hub electronics … with any split across the exposed downstream ports" | USB 3.2 §10.14 (hub configuration) |

A device's current is its stated power (`power_w`, as the maker states it, often a maximum) at the nominal 5 V
(USB 3.2 §11.4.2: "The nominal 5 V source is 4.75 V to 5.50 V"; devices must work from 4.00 V). A bus-powered hub
draws the sum of everything below it from its own port; a powered hub draws nothing counted (the specification lets
it draw one unit load; that is not stated anywhere here). Hub electronics count 0 mA and the result says so. Sensors in
`external_power` have their own supply; PCIe and GMSL devices are not checked (power over coax comes from the
deserialiser board, whose budget is not stated).

## The suggestion (Apply)

1. No host: add "Host (assumed)" with one "USB 3 controller (assumed)" and plug every USB sensor into it — labelled
   an assumption in its names; set your machine's controllers.
2. Sensors not assigned: plug each, largest rate first, into the compatible controller with the most room.
3. A controller above the margin: move the one sensor whose move brings the higher of the two loads lowest, when that
   helps.

Each is one undo step on the `rig` section. Preview draws the move as a dashed line in the 3D view.

## Validation (measured)

| Check | Reference | Tolerance | Measured |
|---|---|---|---|
| `bandwidth.stated_mode_rates` | ams OSRAM NanEyeC datasheet: "75 Mbps at 58 Hz" (SEIM), "63 Mbps at 49 Hz" (LVDS), the only stream data rates a maker in the library states | the payload must not exceed the stated serial rate (0) and be at least half of it | 0: payload 59.392 Mbit/s = 79.2 % of 75, 50.176 Mbit/s = 79.6 % of 63; implied framing 26.3 % and 25.6 % (reported, not judged); the study's on-the-wire figure with its default 5 % overhead is 16.9 % and 16.4 % below the stated rates (reported) |
| `bandwidth.spec_figures` | USB 2.0 Table 5-10 (53 248 000 bytes/s), USB 3.2 §4.4.11 (450 MB/s, 1.1 GB/s, doubled for x2), USB 3.2 Table 3-1 (500, 900, 1500 mA) against the study's derivations from other statements (transactions per microframe, unit loads × count) | 10⁻¹² relative | 0 |
| `bandwidth.budget_recompute` | the budget written out again in the check from the library's JSON files (own stream lookup, hub walk, a table of device generations, USB rules typed in again), 40 seeded random topologies (8 in a quick run) with every controller kind, powered and bus-powered hubs, eleven devices | 10⁻⁹ relative | 3.9 × 10⁻¹⁶ over 752 figures |

The stated-rate check is weak by nature: the NanEyeC's figures are serial rates that include framing, so they bound
the payload rather than equal it. No USB or GMSL maker in the library states a data rate for a stream mode, so no
other device can be checked against a stated rate; the arithmetic itself is checked by the re-computation.
The one stated figure also disagrees with the default overhead: the NanEyeC's own serial framing takes about 26 %,
five times the 5 % default. The default is meant for what a USB or GMSL link adds on top of the pixels, so it is not
changed; for a link that carries a sensor's raw serial stream, enter the stated rate as a measured figure.

## Known limits

- A controller without a measured figure is compared with the specification's upper bound; real controllers often
  deliver less, and several ports can share one controller (makers rarely state which).
- Isochronous USB transfers have their own per-endpoint limits (USB 2.0 Table 5-5), not modelled; the USB 2.0
  figure is the bulk limit without bit stuffing.
- The protocol overhead is a default, not a measurement.
- Compressed streams (MJPEG, H.264) and pipeline-dependent formats have no stated rate and need a measurement.
- Stream modes a device cannot run at the same time (Intel's simultaneous-stream tables) are not checked.
- Bus power uses the maker's stated figure at 5 V: not the inrush, not the drop along the cable, not a regulated
  device's higher current at 4.00 V; hub electronics count 0 mA; USB Type-C current (1.5 A, 3 A) and USB Power
  Delivery are not modelled.
- The GMSL2 link rate and the deserialisers' power-over-coax budgets are not in the library.
- The disk check is a sustained rate, not an SSD's behaviour once its write cache is full; the recorder's figure is
  the user's.

## References

- USB 3.0 Promoter Group, *Universal Serial Bus 3.2 Specification*, Revision 1.1, June 2022: Table 3-1, §4.4.11,
  §10.14, §11.4.1, §11.4.2. https://www.usb.org/sites/default/files/usb_32_202206_0.zip, read 2026-11-06.
- *Universal Serial Bus Specification*, Revision 2.0, April 27, 2000: Table 5-10, §7.2.1.
  https://www.usb.org/sites/default/files/usb_20_20250603.zip, read 2026-11-06.
- USB-IF, *USB 3.2 Specification Language Usage Guidelines* (signalling rates; cited in the library's host entries).
- ams OSRAM, NanEyeC datasheet (as cited in the library entry `ams-osram-naneyec`).
