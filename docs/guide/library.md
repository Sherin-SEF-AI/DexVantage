# The hardware library: a worked example

The Library panel (left dock) lists every camera, depth camera and tactile sensor DexVantage knows from a maker's
datasheet, product page or paper, each figure with its source and the date it was read. The entry format, the gaps
policy and the data-consistency checks are in [docs/models/library.md](../models/library.md).

## 1. Find a device

1. **File ▸ Open sample ▸ UR5e**.
2. In the Library panel type `zed x one` in the search box. Two entries remain: *Stereolabs ZED X One GS, wide lens
   (2.2 mm)* and its narrow-lens twin. The grey column reads `110° · 1928 × 1208` - the horizontal FOV and the native
   image of the first imager.
3. Hover the wide one. The tooltip lists every figure, "—" for those the maker does not state, and the two sources:
   the ZED X One GS datasheet (rev. 2.3) and the store page, both "read 2026-11-07". The note (in the Inspector once the
   sensor is placed) says what the sources disagree on: for the narrow lens the datasheet gives 73° × 45° × 87° and
   the store page 80° × 52° × 91°; the entry uses the datasheet.

## 2. Put it on the robot

1. Double-click the entry (or drag it into the 3D view). A sensor of that model is added to the rig, looking at the task
   volume, as one undo step.
2. In the Inspector, set what it is mounted on to the UR5e's `wrist_3_link` to use it as a global-shutter wrist
   camera (the Inspector's "Mounted on" row then reads `wrist_3_link (link)`).
3. Open the **Camera view**. The library builds the camera from the stated figures: a pinhole 1920 × 1200 px (the
   largest listed stream) with 110° × 80°, labelled "as stated for 1928 × 1208". Its focal lengths come out as
   fx = 672.2 px and fy = 715.1 px - not equal, because a distortion-free pinhole with square pixels cannot show both
   stated angles. The stated focal length says the same thing another way: 2.2 mm / 3 µm = 733 px, which as a pinhole
   would show only 105.5° across the image; the lens shows 110°, about 8 % of barrel distortion. Calibrate a unit and
   store its K and D in the sensor's calibration (docs/models/library.md, "From a sensor to cameras") before trusting
   pixel positions near the edges.

## 3. A depth camera variant and a tactile pad

- Search `d436`. The D436 (added in the RealSense datasheet revision 023) is the D435's depth module with a
  global-shutter OV9782 colour camera. Its RGB imager reads "derived from the stated 1280 × 800 FOV by a centred crop":
  the datasheet streams colour only up to 1280 × 720 while the store page says 1280 × 800, and the entry's note says
  which it assumed. Its RGB position is marked as an assumption (not stated by the maker).
- Search `xela`. The uSkin Patch uSPa 44 is a contact sensor: `pad 23 × 25 mm`, 16 taxels, no imagers. It is placed on
  a finger pad rather than looking at the scene, and the multi-finger occlusion and tactile hand-off study counts its
  pad coverage. It lists no stream (its readings are not images), so its data rate has to be entered for the
  bandwidth budget.

## 4. Check the library itself

Run the library's data-consistency checks from a terminal:

    dexvantage --validate out/validation --models library --quick

On the shipped entries (2026-11-07) they report:

    pass  library.sensor_geometry            0.0122 / 0.02 fraction
    pass  library.fov_shape                  2.92 / 3 deg
    pass  library.fov_focal                  0.477 / 3 deg

`out/validation/validation.html` lists, per check, the entries checked and the six imagers whose maker's angles
contradict each other (their entries carry a `figures_conflict` statement saying so, and the library uses their
horizontal and vertical angles). A custom device you add (Sensors ▸ New custom device…) is not part of these checks;
the same bounds apply to it, so a diagonal FOV far wider than the horizontal and vertical allow is worth a second look.
