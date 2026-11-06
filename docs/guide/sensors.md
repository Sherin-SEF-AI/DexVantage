# Sensors and mounts: a worked example on the UR5e sample

Choose each sensor's device from the datasheet library, mount it on a static frame or a robot link, set its pose and
stream, and see what it sees. The image model and its validation are in [../models/camview.md](../models/camview.md);
the library's entries in [../models/library.md](../models/library.md).

## 1. Look through the wrist camera

1. **File ▸ Open sample ▸ UR5e and Robotiq 2F-85 with a wrist D405 and an overhead D435**.
2. Select **Wrist D405** in the Outliner. The **Camera view** (right, under the Inspector) shows its image: the
   2F-85's fingers at the bottom, the table and the dashed edges of the task volume beyond. The line under the image
   reads `RGB · 1280 × 720 px · 84° × 53° · …`: the D405's colour stream through a pinhole lens fitted to Intel's
   stated FOV (hover over the line for where each figure comes from).
3. Pick **IR left** or **IR right** in the combo above the image: the stereo pair, 18 mm apart.
4. Move a joint (Robot ▸ Joint sliders, or the pose scrubber). The image follows the arm; while you drag it is drawn
   at half size and sharpens when you let go.
5. In the 3D view each sensor shows its imagers' outlines, from the same lens models; the selected sensor's reaches
   further. The overhead D435 draws three: IR left (with the lines from its centre), IR right and the wider RGB.

## 2. Read the device's figures

The **Inspector** lists the device's figures for the selected sensor: the imager and its stream, the FOV as stated
and as modelled, the shutter, pixel size, baseline, minimum Z at that stream, range, interface, power and mass, and
links to the sources with the date each was read. A figure the maker does not state shows "—"; hover over it: "not
stated by the maker" (the D405's pixel size, for one). Nothing is filled in.

## 3. Change the rig on the Sensors and mounts page

**Sensors ▸ Sensors and mounts** opens the page in the Analysis dock. Its table lists the rig's sensors; select one
to edit it below. Each change is one undo step (Ctrl+Z):

- **Device:** any camera, depth or tactile entry of the library (or no device: the default camera, labelled so).
- **Mounted on:** World, Cage, Tripod or Table (static), or any link of any robot (`UR5e + Robotiq 2F-85 ▸
  wrist_3_link`). With **Keep its place** ticked the sensor stays where it is and its pose is recomputed in the new
  frame: mount the overhead D435 on `wrist_3_link` and it rides with the wrist from where it hangs now.
- **Position** (mm) and **roll, pitch, yaw** (degrees, URDF order) of the sensor body in that frame; type `450 mm` or
  `0.45 m`, or drag the axis letter.
- **Stream:** the device's stream modes from its datasheet (`848 × 480 @ 5, 15, 30, 60, 90 fps · depth, IR left…`).
  The Camera view and the studies use that image size; "largest listed" is the default.
- **In use:** switch a sensor off to keep it in the rig but out of the image, the frusta and the studies.

**Add camera**, **Duplicate** and **Remove** work on the selection.

## 4. Add a device from the Library

Open **Depth sensors** in the Library (left) and hover over **Intel RealSense Depth Camera D455**: every figure, "—" where Intel
states none, and the sources. Double-click it (or drag it into the 3D view, over the floor where you want it): a
D455 on a tripod 0.75 m up, looking at the task volume, joins the rig and is selected.

## 5. A device of your own

**Sensors ▸ New custom device…** (or the last line of the Library) opens a form. Fill the maker, the name and a source
(a datasheet's URL, or "measured by me, …" with how), then tick **given** for each figure you have and type it; leave
the rest unticked: they are stored as not given. **Start from a library device** copies an entry's figures and
sources as a starting point. Errors are shown under the form in plain words and block saving. **Save to my library**
writes the device to your library (`DexVantage/library` in your data folder); it appears in the Library at once,
marked "(custom)", and can be used like any other device.
