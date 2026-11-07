# Motion blur and rolling shutter: a worked example on the UR5e sample

Will the images be sharp while the arm moves? The Motion blur page follows each camera's view along the project's
trajectories. It reports how far the image smears during one exposure and how far a rolling shutter skews it. It also
gives the longest exposure that keeps the smear within your limit. The model, its equations and its validation are in
[../models/motion_blur.md](../models/motion_blur.md).

## 1. Open the sample and run

1. **File ▸ Open sample ▸ UR5e**. The rig has a wrist D405 on the flange and an overhead D435 on the cage. The
   sample also carries a 4.5 s scripted pick (*Scripted pick (sample)*, a sample motion, not a recording). From home
   the arm lowers over 1 s, holds for 0.5 s, closes the 2F-85 over 0.5 s, lifts for 1 s and carries the part sideways
   for 1.5 s.
2. **Analysis ▸ Motion blur and rolling shutter** opens the page (Placement workspace). Press **Run** (0.3 s).

The headline reads **up to 38.04 px · may be blurred**: the wrist D405's IR left imager at t = 3.03 s, seeing the task
volume. "Up to" is there because no exposure is set, so the study uses the longest exposure the stream allows: the
frame period at 30 fps, 33.3 ms. That is an upper bound, not an Intel figure. The table:

| Camera | Shutter | Exposure (ms) | Image speed (px/s) | Smear (px) | Skew (px) | Longest exposure (ms) |
|---|---|---|---|---|---|---|
| Wrist D405 · IR left | global | 33.3 (at most) | 1 141 | 38.04 | 0 | 0.876 |
| Wrist D405 · RGB | global | 33.3 (at most) | 1 051 | 35.04 | 0 | 0.951 |
| Overhead D435 · IR left | global | 33.3 (at most) | 146 | 4.86 | 0 | 6.85 |
| Overhead D435 · RGB | rolling | 33.3 (at most) | 312 | 10.40 | up to 10.40 | 3.21 |

The wrist camera rides on the flange, so the whole scene streams past it: its twist J(q)·q̇ turns it at the wrist's
angular velocity. The overhead camera stands still and sees only the gripper move (its worst point is the left pad).
The D435's RGB camera has a rolling shutter. Intel does not state its readout time, so the skew is bounded by the
frame period: "up to 10.40 px".

The segments table follows the motion:
- lowering (0–1 s): 35.0 px;
- the hold (1.0–1.5 s): 0 px, **sharp**;
- closing (1.5–2.0 s): 16.0 px, the pads moving in the wrist camera's view;
- lifting (2–3 s): 27.0 px;
- carrying (3.0–4.5 s): 38.0 px.

The 3D view draws the TCP's path in those segments' status colours.

## 2. Look at the worst frame

Choose the Wrist D405 · IR left row, pick its first worst frame and press **Show frame**. The robot is posed at
t = 3.03 s, and the Camera view switches to that camera and imager. It draws each point's smear vector, lengthened
when short; the legend says by how much.

## 3. Apply the suggestion

The suggestion reads: *Set Wrist D405: exposure at most 0.87 ms (smear up to 38.04 px → 0.99 px); Overhead D435:
exposure at most 3.2 ms (smear up to 10.40 px → 1.00 px).* It adds that the D435's rolling shutter skews the image
by up to 10.40 px, and that a global-shutter camera has none; the library's are the D405, D455 and D457.

**Apply** records these exposures as yours in one undo step, and the study runs again. Every camera is now
**sharp**, within 1 px (the default limit, a DexVantage default, not a standard). The skew is not changed by the
exposure, so the rolling-shutter advice stays.

Set the same exposure on the cameras, for example in the RealSense viewer or your driver's parameters. A shorter
exposure needs more light or gain; the Lighting page treats that. Intel states the D435 depth imagers' manual exposure
as 1 to 166 ms (datasheet Table 4-23), so 3.2 ms is inside its range. For the D405 the datasheet's exposure table does
not name the module, so no range is checked.

## 4. Your own figures

- **Your exposure:** pick a camera, type the exposure in ms and where it comes from (for example "driver setting,
  2026-11-07"), then press **Set exposure** (one undo step). **Clear** goes back to the bound.
- **Smear limit:** your tolerance in pixels (default 1 px).
- **Readout:** the study takes a measured rolling-shutter readout from the settings `readout_s` (`{sensor uid:
  {value, source}}`).

## The SO-100 sample: no trajectory, no device

The SO-100 sample has no trajectory, so the study uses 8 approaches of its grasp set: 0.10 m along the approach at
0.10 m/s, both defaults, solved by IK. Its cameras have no library device, so there is no frame rate and no exposure.
The headline says **exposure not set**. The page still gives the image speed (up to 1 559 px/s for the wrist camera)
and the longest exposure for 1 px (0.64 ms).

## Batch

    dexvantage --sample ur5e --motion-blur blur.json        # 0.8 s with start-up; CPU only (MuJoCo kinematics)

The result is JSON with a provenance block; `--reproduce blur.json` runs it again and compares.
