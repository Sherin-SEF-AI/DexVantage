# Depth validity: a worked example

Where in the task volume does each depth camera give a depth reading, and why not elsewhere? This walk-through uses
the UR5e + Robotiq 2F-85 sample, whose "Wrist D405" rides on the flange and whose "Overhead D435" hangs from the
cage. The model and its checks are in `docs/models/depth.md`. The figures below are what this build computes on the
sample at the home pose; they move with the robot's pose and your edits.

## 1. Open the sample and run the study

    dexvantage --sample ur5e

Analysis ▸ Depth validity opens the page. **Run** passes the 3D view's joint state to the study (the sample opens at
its home pose). In batch mode the same study runs at the home pose:

    dexvantage --sample ur5e --depth-validity depth.json

The headline reads **84.6 %, good**: that much of the 0.50 × 0.40 × 0.30 m tabletop volume has valid depth from at
least one of the two cameras (good from 80 %, partial from 50 %: defaults, shown in the headline's tooltip). Per
sensor: Wrist D405 45.8 %, Overhead D435 61.3 %.

## 2. Read the breakdown

With **Sensor** set to Wrist D405, the chart gives the first condition each voxel fails:

| | % of the task volume |
|---|---|
| Depth-valid | 45.8 |
| Outside a field of view | 36.6 |
| Occluded for an imager | 17.6 |
| Closer than min-Z, beyond the range, outside the projector cone, material | 0.0 |

The D405 looks down at the gripper from 70 mm off the tool axis, so a third of the tabletop is simply outside its
84° × 58° depth field of view (both of its imagers must see a point). Of the occluded 17.6 %, the line under the chart
splits off 7.2 % hidden from one imager only: the fingers' shadows and the stereo parallax, which a single camera
model would miss. The D405 is passive (no projector), so condition 4 does not apply; the D435's projector cone does
(Sensors table: "needed").

Tick the boxes under **Invalid voxels in the 3D view** to see the failing voxels coloured by reason (the field of
view box is off by default, since those are most of the invalid voxels and the least surprising).

## 3. Read the min-Z blind zone

In the 3D view a translucent pyramid in each camera's colour sits in front of it: the part of its field of view
nearer than min-Z, where the stereo matcher cannot reach the disparity. At the default resolution, 1280 × 720 (the
D405's largest depth stream, labelled "(default)" in **Resolution**), its depth is 100 mm (Intel's Table 4-12).
Move the arm (Robot page sliders, or IK drag) and the D405's zone moves with the wrist; the D435's stays put. View ▸
Depth min-Z zones hides them.

The status line under the chart answers the question a wrist camera raises: are the fingertips beyond min-Z? Here
they are: "Fingertips beyond Wrist D405's min-Z (100 mm): gripper_right_pad 114 mm, gripper_left_pad 114 mm." A
14 mm margin.

## 4. Move the camera and watch it change

Select Wrist D405 (Outliner) and change its **Position** in the Inspector. The mount pose is in the `wrist_3_link`
frame and the camera looks along (0, 0.873, −0.489) there, so moving along that axis keeps its aim.

**30 mm closer**, Position (0, 0.157, 0.055): the zone now swallows the finger pads. Run again:

- "Inside Wrist D405's min-Z (100 mm): gripper_right_pad at 84 mm, gripper_left_pad at 84 mm. The fingertips have no
  depth there", a warning.
- Depth-valid 41.4 % (more of the table is out of view from closer in).
- The headline offers: "Set Wrist D405 to 848 × 480: its min-Z is 70 mm instead of 100 mm, which clears the
  fingertips at 84 mm (inside min-Z now); σ_z rises 1.51× (focal length about 471 px instead of 711 px)." **Preview**
  draws that zone dashed; **Apply** sets the resolution (one undo step) and runs again: the fingertips clear, the
  zone shrinks to 70 mm, and the median σ_z goes from 0.67 mm to 1.02 mm.

**30 mm back** instead (Edit ▸ Undo, then Position (0, 0.105, 0.085)): the fingertips sit at 144 mm, the zone is
well clear of them, and depth-valid rises to 51.4 % (rig 86.2 %) because the wider view takes in more of the table.
The cost is noise: σ_z median 0.86 mm instead of 0.77 mm, since σ_z grows with z².

## 5. Read the noise against the maker

The table **Expected depth error against the maker's figure** gives, for the D405 at 1280 × 720: f 710.8 px, baseline
18.0 mm, σ_d 0.08 px (default: Intel states no subpixel accuracy), σ_z median 0.77 mm [0.25–1.39] (5th–95th
percentile over the depth-valid voxels), the model at Intel's distance, 1.6 mm at 0.5 m, against Intel's "≤ 5.0 mm
(1 % at 0.5 m)", and the σ_d that would match Intel's figure, 0.256 px. Intel's figure is an upper bound on a
plane-fit RMS over 80 % of the view under its own test conditions, so the two are not the same quantity; if you have
measured your unit, put its σ_d in the settings (`sigma_d_px`) and the table follows it.

## What to try next

- Add a task object (Workspace ▸ Add box object) and tag it transparent or glossy (its `materials` list in the
  project; the Inspector shows the tags but has no field to change them yet): its surface shows as a material
  warning with the reason, and the IR-absorbing tag warns for the active D435 but not the passive D405.
- Change the D435's resolution and compare its min-Z (280 mm at 1280 × 720, 105 mm at 424 × 240) with how far it is
  from the table.
- `dexvantage --validate out/validation --models depth` runs the four depth checks.
