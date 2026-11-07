# Time sync: a worked example on the UR5e sample

Is each image labelled with the joint state of the moment it was taken? The Time sync page adds up how far each
camera's frame stamps are from the instant of exposure, against the robot's joint_states. It turns that skew into the
**action-label error**: how far the end effector moves in that time along your trajectory. The model, its sources and
its validation are in [../models/time_sync.md](../models/time_sync.md).

## 1. Open the sample and run

1. **File ▸ Open sample ▸ UR5e** (the wrist D405, the overhead D435 and the 4.5 s scripted pick).
2. **Analysis ▸ Time sync** opens the page (Placement workspace). Press **Run** (0.2 s).

No clocks or triggers are set, so every stream is taken as stamped by the host when its frame arrives, and
free-running. That is the default, and the notes say so. The headline reads **≥ 54.43 ms · out of sync**.

The budget of the Wrist D405 · IR left stream lists each term with its source:

| Term | Low (ms) | High (ms) | From |
|---|---|---|---|
| exposure centre to the end of the exposure | 0 | 16.7 | no exposure set: up to the frame period at 30 fps; no minimum stated |
| readout after the exposure (global shutter) | 0 | 33.3 | readout not stated by Intel: at most the frame period |
| USB transfer of the frame | 4.30 | 4.30 | 1.94 MB per frame at 3600 Mbit/s (the device's own USB 3.2 Gen 1) |
| USB bus interval | 0 | 0.125 | 125 µs microframes (Microsoft Learn; USB 2.0 §5.8.4) |

Two terms are **not counted**. The D405 is not assigned to a host controller on the Bandwidth page, so its wait
behind other devices is not bounded. The processing inside the device before it sends the frame is not stated by
Intel. That is why the figure says "≥": the real skew can only be larger. joint_states has its own unstated term, the
robot controller's latency to the host.

At the pick's peak TCP speed (0.17 m/s), the skew moves the end effector by up to **9.19 mm**: the action-label error,
shown in the Label error column. The 3D view draws the TCP's path and joins the TCP at the exposure to the TCP at the
stamp at the worst moment. **Show worst moment** poses the robot there, and the Camera view draws the same segment.

## 2. The hardware trigger

The suggestion reads: *Trigger Overhead D435 in hardware.* It quotes Intel: "For D435, D435i, D435f and D435if, only
the depth sensor can be synced by an external trigger" (§3.7.4). It predicts the D435 IR stream's worst skew falling
from 54.4 ms to 16.7 ms, and its label error from 9.19 mm to 2.82 mm. What is left is half the exposure. With a fixed
exposure (the Motion blur page) that becomes a constant you can subtract.

**Apply** sets the trigger in `project.rig['sync']` (one undo step) and runs again. The D435's RGB stream stays
free-running: Intel's statement covers only the depth sensor, and the budget says so. The next suggestion concerns the
D405. Intel states it "does not support the hardware sync signal" (§7.13), so the advice is to measure its arrival
latency, or to stamp on a PTP-disciplined clock. The page refuses a hardware trigger on the D405 with Intel's words.

## 3. Make the budget complete

- **Bandwidth page:** add a host and plug both cameras into its USB 3 controller. The transfer term then becomes
  1.90 to 7.60 ms for two 848 × 480 16-bit streams per camera on one USB 3.2 Gen 1 controller: the stream's frame
  alone, up to behind one frame of every other stream.
- **Motion blur page:** set the exposure (for example 1 ms).
- **Measured latencies:** enter each stream's arrival latency in the settings `latency_s`, for example from the
  RealSense frame metadata (the frame timestamp against the arrival time), with its source, and joint_states' too.
  With these, every stream's budget is complete.
- **Clocks:** for a PTP or NTP clock, enter the offset, jitter and drift your servo reports, with their source. The
  study never assumes them.
- **Labels:** choose interpolated labels (the default), or the nearest joint_states sample with its rate in Hz and
  its source.

## The camera pair

The pairs table shows the wrist and overhead cameras. Both are free-running at 30 fps, so frames matched by timestamp
can be up to 16.7 ms apart, on top of the difference in their stamp errors. A sync cable would lock them, but Intel
states the D405 has none ("Sync cable possible: no").

## Batch

    dexvantage --sample ur5e --time-sync sync.json          # 0.7 s with start-up; CPU only

The SO-100 sample's cameras have no library device. With no frame rate, the budget has only the bus interval and
says **incomplete**; choose the cameras' devices on the Sensors page.
