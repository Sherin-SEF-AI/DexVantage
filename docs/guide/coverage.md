# Coverage and resolution: a worked example on the UR5e sample

How much of the task volume do the cameras see, from how many angles, and with how many pixels per millimetre? The
Coverage page answers this for the pose on screen. The model, its definitions and its validation are in
[../models/coverage.md](../models/coverage.md).

## 1. Open the sample and run

1. **File ▸ Open sample ▸ UR5e and Robotiq 2F-85 with a wrist D405 and an overhead D435**. The rig is "Wrist D405" on the flange and "Overhead D435" on the
   cage, over a 0.50 × 0.40 × 0.30 m tabletop task volume. These are sample placements; the lenses are the library
   entries from Intel's D400 series datasheet.
2. **Analysis ▸ Coverage and resolution** opens the page in the Analysis dock (Placement workspace).
3. Press **Run**. The study voxelises the volume (480 000 voxels at 5 mm), builds an occlusion cube map around each
   camera with the arm at the pose in the 3D view, and fills the page in about a second.

At the home pose the headline reads:

    26.4 % of the task volume · mostly single or blind (status: critical)

with the sentence "26 % of the task volume is seen by two or more cameras, 65 % by one only and 8 % by none. Wrist
D405 sees 46 %: 40 % of the volume is outside its image, so aim it at the volume, move it back or choose a wider
lens."

The status word comes from your thresholds: good from 90 %, warning from 60 %, serious from 30 % (the defaults,
labelled "default" under **Thresholds**; change them there, one undo step each).

## 2. Read the camera table

| Camera | Imager | Sees | Outside image | Hidden | px/mm median | px/mm 5 % | Angle median | Blur median |
|---|---|---|---|---|---|---|---|---|
| Wrist D405 | RGB | 46 % | 40 % | 14 % | 2.13 | 1.53 | 24° | not computed |
| Overhead D435 | RGB | 72 % | 0 % | 28 % | 1.68 | 1.44 | 13° | not computed |

- Both are depth cameras; the study uses their **RGB imager** (hover over "RGB" for the lens model and why). Set
  `depth_imager` to `depth` in the study settings to judge them by the left IR imager instead: the overhead camera
  then drops to 0.79 px/mm (its IR stream is 1280 × 720, its colour stream 1920 × 1080).
- The overhead camera frames the whole volume (0 % outside its image) but the arm hides 28 % of it at this pose.
- The wrist camera sees the space in front of the gripper at about 2 px/mm, but 40 % of the volume is outside its
  84° × 53° image.
- **Blur: not computed.** Intel states no focus distance and no pixel pitch for these imagers. Hover over the cell
  for the missing figures; if you have measured them, enter them in the settings (`optics`) and run again.

## 3. Look at the slice

Under **Slice in the 3D view**, choose the axis (x, y or z) and drag the slider. The 3D view draws one layer of
voxels at that height, coloured by **Cameras that see it** (red: none; three blues for 1, 2 and 3 or more), by
**Best pixels per mm**, or by one camera's pixels per mm (log scale from the 1 px/mm "resolved" threshold, grey
below it or where that camera does not see). At z = 0.09 m the red patches are the arm's shadow from the overhead
camera, where the wrist camera does not look either.

## 4. Try another pose

Open the **Robot** tab and lower `shoulder_lift_joint` by 20° (from −90° to −110°): the arm rises. The Coverage page
says "The robot has moved since this run: Run again for the coverage at this pose." Press **Run**:

    59.9 % of the task volume · thin (status: warning)

The arm now hides only 4 % of the volume from the overhead camera (96 % seen), the wrist camera sees 64 % at a median
1.52 px/mm, and 99.9 % of the volume is seen by at least one camera. Raising the wrist further (shoulder −30°, elbow
+30°) gives the overhead camera all of the volume, but the wrist camera then sees only 9 % and two-camera coverage
falls to 8.8 %: the two cameras trade off against each other, and the task's real poses decide.

## 5. Export and batch

**Export…** writes the result as JSON with a provenance block (the project's content hash, the versions, the time).
It holds the joint state used (`q`, `q_source`), the summary, every camera's figures and the per-voxel arrays
(decode them with `dexvantage.core.studies.coverage.decode`).

From the command line, at the joint state saved in the project's coverage settings (home when none):

    dexvantage --sample ur5e --coverage coverage.json
    dexvantage my-rig.dexv --coverage coverage.json

## Known limits

Coverage is for one pose per run. Pixels per millimetre is for a patch facing the camera. Voxels inside the robot
count as unseen. No sensor housings occlude. Depth validity (min-Z, the stereo pair, the projector) is its own
study. See the model page for the full list.
