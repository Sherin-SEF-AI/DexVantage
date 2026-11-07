# Teleop overlap: a worked example on the SO-100 sample

The model, its equations and its validation are in `docs/models/teleop.md`.

## 1. Open the sample and the page

    dexvantage --sample so100

Choose the Teleop workspace (Window ▸ Teleop workspace) or Analysis ▸ Teleop overlap. The sample carries one device,
**SO-100 leader arm (sample)**: a leader arm whose leader is the follower's own model, mapped joint to joint. That is
how the SO-100 kit is built and calibrated (LeRobot's SO-100 guide, read 2026-11-07; the device's source says so).

Press **Run**. The study runs in the background (about 7 s).

## 2. Read the headline

    96.7 %   clear
    96.7 % of the task volume can be teleoperated without a dead zone. 3.3 % is outside the leader's comfortable
    workspace. Move SO-ARM100's base +25 mm in y, +25 mm in z: teleoperable 96.7 % → 100.0 % …

- **Comfortable** is the part of the task volume the follower reaches with every joint inside 0.8 of its range (the
  comfortable share is a default, under Settings). Here 96.7 %: the lowest corner of the tabletop volume needs the
  shoulder or wrist beyond that share.
- **Teleoperable** removes the dead zones. The sample has none at the defaults: at the comfortable poses no voxel
  crosses the singularity or self-collision thresholds, and the comfortable box itself keeps every joint 10 % of its
  range from its stops.
- The follower line says what the follower alone does: it reaches 100 % of the volume, and with its own IK
  (nearest home) Wrist_Pitch sits within 2.5 % of its stop in 19.4 % of it, because the file's home already has the
  wrist pitched to 1.57 rad against a 1.66 rad stop. A leader arm avoids this; a cartesian device does not (step 4).

## 3. Apply the suggestion

**Preview** draws the move from the arm's base; **Apply** makes it in one undo step. The SO-100 is the primary robot,
whose base is the world frame, so the move is applied as the opposite move of the task volume, the table, the tripod
camera and anything else fixed in the world (the wrist camera rides on the arm and stays). Run again: 100 %. Undo
(Ctrl+Z) puts everything back.

## 4. Add a VR device

Under Teleop devices choose **VR tracking** and **Add**. Its volume reads *Not stated*: no headset maker publishes a
tracking volume, and DexVantage does not guess one. Run now and the device is listed as "not stated" with a note
saying what to enter.

Enter the box you measured for your operator (here 0.30 × 0.20 × 0.12 m centred on 0, −0.24, 0.06 m, the sample's task
volume, scale 1, no offset) and write where it comes from in **Source** ("measured, 2026-11-07"). Each field is one
undo step. Run:

| Device | Comfortable | Teleoperable | Dead zones |
|---|---|---|---|
| SO-100 leader arm (sample) | 96.7 % | 96.7 % | 0 |
| VR tracking 1 | 100 % | 80.6 % | 19.4 %, 1 zone |

The headline (through either device) reads 98.1 %. The dead zone is **joint limit**, 1 453 cm³: "Wrist_Pitch is at
its stop". Click it in the Dead zones list and the arm takes the pose at its worst voxel; the 3D view shows the zone's
voxels in the joint-limit colour, the Camera view its box. A cartesian retargeting solves the follower's own IK, which
here pins the wrist; the leader arm's joint mapping does not.

The suggestion now moves the base +50 mm in y and z: teleoperable 98.1 % → 100 %, dead zones 1 453 → 63 cm³. Applied
and run again it measures 100 %, and the VR device's dead zones 63 cm³, as predicted (the VR volume is placed in the
follower's base frame, so it moves with the base; the leader arm covers the rest).

## 5. Two arms

With a second robot and a device for it (Robot ▸ Import, then Add with that robot as the follower), the page adds the
bimanual figures. Two SO-100s facing each other across the sample's volume (bases 0.48 m apart) both reach all of it
and both are teleoperable over 93.3 %; at 60 sampled handover voxels with the grippers 100 mm apart, the handover
quality is 15 % and the smallest arm-to-arm margin 0.1 mm (median 28.5 mm): the wrists meet. **Pose the worst
handover** shows both arms there.

## 6. Batch and export

    dexvantage --sample so100 --teleop teleop.json

writes the same result (exit 0; a project the study cannot work with exits 2 with one plain sentence). On the page,
**Export…** writes it with its provenance block. Per voxel, `devices[i].state` holds 0 teleoperable, 1 outside the
comfortable workspace, 2–5 a dead zone (unreachable, self-collision, singularity, joint limit) and 9 not stated, in
the order of `grid.centers`.

## Settings

All are your settings with labelled defaults: voxel size 25 mm, comfortable share 0.8, singular below 1/κ = 0.1, near
a limit below 0.05 of the half-range, self-collision margin 10 mm, suggestion search 2 voxels each way (more in the
project file: `jacobian`, `ik_rule`, `seeds`, `handover_gap_mm`, `arm_margin_mm`, `handover_samples`,
`min_column_mm`).
