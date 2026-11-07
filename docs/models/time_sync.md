# Time sync {#time_sync}

Module: `dexvantage/core/studies/time_sync.py` (study id `time_sync`, batch flag `--time-sync OUT.json`, or `--sync OUT.json`, the specification's name). Settings in
`project.rig['sync']` (`core/project.py` `sync_from_dict`, both copies of the project schema) and
`project.analysis.studies['time_sync']`. Page: `dexvantage/ui/pages/time_sync.py` (Analysis ▸ Time sync, Placement
workspace). Checks: `dexvantage/validation/checks_sync.py` (model `timing`). Tests: `tests/test_time_sync.py`. Spec
§7.10. Written for DexVantage. Vantage's `core/clock_tree.py` simulates an automotive GNSS/PTP clock tree; nothing of
it is used here.

## Clocks and streams (`project.rig['sync']`)

    {'clocks':  [{'uid', 'name', 'kind': camera_hw | host | ptp | ntp | robot,
                  'offset_s', 'jitter_s', 'drift_ppm'}],        each {'value', 'source'} or null (not stated)
     'streams': {sensor uid | 'joint_states': {'clock': clock uid ('' = host arrival), 'trigger': free | hardware | sync_cable}}}

The section is only written when it holds something, so an untouched rig keeps its old form. A stream with no entry
is stamped by the host when its frame arrives and runs free; this is the default, and the result says so. Times are
on the host's time base. A clock's offset is its reading minus the host's.

## Stamp error of a camera frame

`e = stamp − the instant the frame was exposed`, taken as the middle of the middle row's exposure (Liang et al. 2008).
Each term is an interval `[low, high]`, and the budget is their sum:

| Stamping | Terms |
|---|---|
| Host arrival (no clock or a `host` clock), free or on a sync cable | `E/2` (exposure centre to its end) + `T_r/2` (rolling shutter: the middle row to the last) or `T_r` (global shutter: rows read after the exposure) + USB transfer `[bits/C_link, (bits + one frame of every other stream on the controller)/C_controller]` + bus interval `[0, 125 µs]` |
| Device clock (`camera_hw`, `ptp`, `ntp`) | `±E/2`: which instant of the exposure the stamp marks is not stated |
| Hardware trigger (only where the maker states one) | `−E/2`, and `−T_r/2` for a rolling shutter: the stamp is the trigger edge. Transfer and processing no longer count |
| Measured latency (settings `latency_s`) | the measured figure replaces exposure, readout, transfer and processing |
| Any clock but the host's | its offset (a constant), `±jitter`, and drift `× 10⁻⁶ × episode_s` (60 s by default) |

Each input has a stated source:
- **Exposure E and readout T_r:** from the motion blur study's `imaging`. That is your figure, else the library
  entry's, else the frame period as an upper bound. An exposure that is not fixed (auto-exposure) spans from the
  shortest the maker states (0 when none is stated) to that bound.
- **USB transfer:** from the bandwidth study's topology. Bits are `width × height × bits per pixel × (1 + overhead)`,
  with the 5 % overhead default. `C_link` is the lower of the device's and the controller's USB generation, at the
  USB 3.2 §4.4.11 effective rate or the controller's stated practical figure. When a sensor is not assigned to a
  controller, only the lower end is known, and the budget is flagged incomplete.
- **Bus interval:** "For high speed and SuperSpeed, the bus interval is a microframe. The length of a microframe is
  125 microseconds" (Microsoft, *How to transfer data to USB isochronous endpoints*, 2024-01-17, read 2026-11-07;
  USB 2.0 §5.8.4).
- **Processing inside the device:** not stated by any maker in the library (a depth camera computes depth before it
  sends). A host-stamped budget is therefore a **lower bound**, said "at least" or "incomplete", until you enter a
  measured latency.
- **PTP and NTP figures:** never assumed. Enter what your servo reports (for example ptp4l's offset or `chronyc
  tracking`) with its source. A figure not entered is "not stated".
- **Hardware trigger support:** from the library entry's `sync` block (`hardware_trigger`, `sync_cable`, the imagers
  it reaches, the maker's words). For the D435, only the depth sensor takes the external trigger (D400 datasheet
  §3.7.4); for the D405, "D405 does not support the hardware sync signal" (§7.13). Setting a trigger on an imager the
  maker says takes none counts that imager as free-running, and says so. The page refuses the setting.

## joint_states

joint_states has its own budget:
- **Stamped by the host on arrival** (the default): the latency from the robot controller to the host is not stated.
- **Stamped by a `robot`, `ptp` or `ntp` clock:** that clock's figures.
- **Measured:** a figure in `latency_s['joint_states']`.

A frame's label is joint_states at the frame's stamp. With `pairing` `interpolate` (the default) the label is
interpolated. With `nearest` it is the nearest sample, which adds `±1/(2r)`; `r` is `joint_rate_hz`, with its source.

## Skew, pairs and the action-label error

    skew = e_camera − e_joint = [e_c,low − e_j,high,  e_c,high − e_j,low]

- **Worst skew:** the interval's largest magnitude.
- **Spread:** the interval's width. That is what remains after a constant latency is subtracted.
- **Camera pairs:** two free-running cameras matched by timestamp are also up to half the longer frame period apart.
  A shared sync cable, or a hardware trigger on the same clock, removes that.

The **action-label error** is the end effector's largest displacement during the skew:

    max over t of |p(t + δ) − p(t)|,  δ at both ends of the skew interval

`p` is the TCP (`grasps.gripper_geometry`). It comes from forward kinematics (`RobotModel.fk_array`) of `q(t)` on a
1 ms grid, linear between grid points. The motions are the motion blur study's.

## Status and recommendation

The status compares the worst skew with `skew_limit_s` (2 ms by default; the spec's figure, a DexVantage default, not
a standard):
- within the limit: "in sync";
- up to twice the limit: "tight";
- beyond that: "out of sync";
- when a term is not stated: "out of sync" with the figure shown as a lower bound (≥), or "incomplete".

The recommendation takes the first of these that applies:
1. A hardware trigger for the stream with the worst skew whose maker supports one. The predicted skew and
   action-label error come from running the study again with the trigger set. Apply sets it in `project.rig['sync']`
   (one undo step).
2. A sync cable between two free-running cameras that both support one.
3. Advice: measure the arrival latency, or stamp the frames on a PTP-disciplined clock.

## Validation (model `timing`, measured 2026-11-07, full run)

| Check | Reference | Measured | Tolerance |
|---|---|---|---|
| `sync.stamp_interval_sim` | the transfer times against the frame size and the USB 3.2 Gen 1 capacity typed in again; the stamp interval against a row-by-row simulation of 2 000 frames per case (free-running, fixed exposure, hardware trigger, device clock with offset, jitter and drift, measured latency): frames queued behind a random set of other frames on the bus and seen at the next 125 µs boundary from a random phase | 3.5 × 10⁻¹⁸ s outside the interval; every simulated span at least 99.97 % of the interval | 10⁻⁹ s; span ≥ 90 % |
| `sync.action_label_sim` | 30 fps frames at 40 random phases along the UR5e sample's scripted pick, stamped with skews of −20, −2, 3, 15 and 40 ms; label and truth by forward kinematics at those instants | 2.2 × 10⁻⁴ (relative) | 2 % |

## Worked example

The UR5e sample is in [../guide/time_sync.md](../guide/time_sync.md).

## Known limits

- **Device processing:** the makers do not state the processing time inside a device. A host-stamped budget is a
  lower bound until you enter a measured latency.
- **USB model:** a USB transfer is modelled as the stream's frame at the link's capacity, queued behind at most one
  frame of every other device on its controller. Isochronous reservations, driver buffering and the host's interrupt
  moderation are not modelled.
- **Clocks:** clock figures are what you enter; no servo is simulated.
- **Trigger delay:** the delay from a hardware trigger to the exposure is not stated and not counted.
- **End effector:** the TCP stands for the end effector, so an object held off-centre moves differently when the
  gripper turns.
- **Other joints:** the robot's other joints stay at home.
