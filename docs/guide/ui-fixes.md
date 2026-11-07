# Responsiveness, bounded figures and custom devices: a worked example

The model and the measurements are in [docs/models/ui-fixes.md](../models/ui-fixes.md). This walk-through uses the
bundled samples.

## 1. Edits stay quick on the two-arm samples

1. File ▸ Open sample ▸ ALOHA 2.
2. Drag a camera in the 3D view or type a new x in the Inspector. Each edit now costs about 4 ms on the UI thread
   (it was 32-39 ms): the 30-odd analysis pages that are not shown wait, and refresh once when you open them.
3. Open the Sensors page: it shows the rig as it is now, after all the edits, in one refresh.

The assistant reads a page as you would see it, shown or not: asking it about the Sensors page while the Summary is
open replays what the page held first.

## 2. Adding the G1 from the robot catalogue

Library ▸ Robots ▸ Unitree G1 ▸ Import…, read the licence, tick "I accept", Import. The model compiles, and its
bounds are computed, on a worker; adding it to the scene takes about 11 ms on the UI thread (it took 171 ms).

## 3. A refused study in the log

Open a new project with no camera and run Coverage. The page says "could not run" with the sentence; the log has one
line, `coverage refused: …`, and the traceback only when DEBUG logging is on.

## 4. An experiment where a study states only a bound

Some figures are stated as bounds: the motion-blur smear is "at most" a value when the exposure comes from the
stream's maximum, the sync skew "at most" when a readout time is not stated. In an experiment over such a figure:

* the runs table shows a bounded response as "≤ 3.1";
* the analysis (means, main effects, Sobol indices) leaves bounded values out and counts them under "censored";
* the best run is the one known to be best; when a bounded run could still beat it, the headline says the pick is
  not settled and names how many runs could;
* when the bounds do not decide whether the best run beats the rig as it is, the headline word is "not settled" and
  Apply is not offered. Give the figures the study lacks (the exposure, the readout time) to settle it.

## 5. The canonical view on the Panda pair

File ▸ Open sample ▸ Panda bimanual, then Export ▸ Canonical view. The headline reads **larger than the image
(bound 11 878 px)**, critical, "large parallax": the two wrist cameras are mirrored about the fingertips, so points off
the contact plane land far outside the 1280 px canonical image. The advice says the same in words and suggests moving
the source camera to the canonical pose (Apply, one undo step).

## 6. A custom light, cable, policy or teleop device

Library ▸ New custom device…, Kind: Light. Fill the maker, the name and the source ("measured by me with a lux
meter at 1 m" counts), choose the type (ring), tick and type only the figures you have (beam angle 40°). Every figure
left unticked is stored as not stated. Save to my library: the entry's note starts "User-entered custom device." and
the Lighting page can use it at once. A teleop leader arm also asks which followers it leads and whether it has
their joints; a policy asks for its input size, crop rule, chunk and timings.

## 7. Surfaces in the Outliner

On the Workspace page, Add surface. The Outliner lists it under Workspace; click it there and the Workspace page
selects it for editing, the Inspector shows its height and size, and Delete removes it (one undo step).

## 8. The readiness gate

    dexvantage cell.dexv --check requirements.yaml

prints the table and, last, `Scorecard → /full/path/to/scorecard.json (no output file given: scorecard.json in the
current folder)`. Give a second argument to write elsewhere.
