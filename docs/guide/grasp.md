# Grasp self-occlusion: a worked example on the UR5e sample

At the moment the fingers close, can the cameras see the fingertips and the object? A policy learns contact only from
what its cameras show. The Grasp self-occlusion page answers this for a whole grasp set: it solves the arm for every
grasp, asks every camera which parts of each fingertip's contact patch it sees, and reports the share of grasps where
your requirement holds, the Contact Visibility Score, with its interval. The model, its definitions, its validation
and the measured timings are in [../models/grasp.md](../models/grasp.md).

## 1. Make a grasp set

1. **File ▸ Open sample ▸ UR5e and Robotiq 2F-85 with a wrist D405 and an overhead D435**. The rig is "Wrist D405" on
   the flange and "Overhead D435" on the cage above a 0.50 × 0.40 × 0.30 m tabletop task volume (sample placements).
2. The sample already carries the set "Tabletop grasps (sample)": the same 200 grasps as below, so you can go
   straight to step 2. To make it yourself, open the **Tasks** page, delete that set, and under the sampling form tick **Top-down pinch**, **Side grasp** and **Angled**, leave
   **Count** at 200 and **Seed** at 0, and press **Apply**. The set "Sampled: top-down pinch, side grasp, angled (200)"
   appears in the table and as gripper glyphs in the 3D view. Grasp sets are described in
   [grasps.md](grasps.md); you can also import a CSV or JSON of gripper poses or take a trajectory.

## 2. Run the study

1. **Analysis ▸ Grasp self-occlusion** opens the page (Grasp study workspace). The **Grasp set** box shows the set,
   **Cameras** reads "All cameras in use", the requirement "Every fingertip", IK "Nearest to home", and the numbers
   below are labelled "default".
2. Press **Run**. On a recent GPU the 200 grasps take about 25 s: inverse kinematics, the cube maps at contact and at
   2.5 and 5 cm before it, "what to move" and the gallery. The note under the page says how fast it went and on which
   renderer.

The headline reads

    100.0 [97.2–100] % of reachable grasps · well seen

with "100 % of the 133 reachable grasps have every fingertip at least 50 % visible in at least one camera. … Every
resample gives the same score, so the bootstrap interval is a point; the figure shows Wilson's 95 % score interval
instead, 97.2 % to 100.0 %." The 100 % is real, not a weak test: the wrist D405 sees both fingertips in every grasp;
the overhead D435 alone sees them in only 35 % (the camera table). Hover over the
figure for the counts, the interval's method and the thresholds (good from 90 %, warning from 70 %, serious from
40 %: DexVantage defaults, not standards).

Under the settings: "133 of 200 grasps reachable without a collision; unreachable: 64 obstacle collision, 2 out of
reach, 1 no convergence; most obstacle contacts: wrist_1_link × Table (sample) (49), …". The sampler puts the grasp
centre 30 mm above the table; for a side grasp that low, the UR5e's wrist hits the table. That is about the task, not
the cameras: unreachable grasps are left out of the score (the export also gives the share of all grasps).

## 3. Read the camera table

| Camera | Fingertips visible | Both ≥ threshold | Pre-contact | Seen | Outside image | Behind own finger | Hidden by scene |
|---|---|---|---|---|---|---|---|
| Wrist D405 | 95 %, 95 % | 100 % | 95 % | 95 % | 0 % | 0 % | 5 % |
| Overhead D435 | 42 %, 42 % | 35 % | 33 % | 42 % | 0 % | 25 % | 33 % |

- The wrist camera looks down the gap between the fingers and sees both pads in every grasp: it alone carries the
  score.
- The overhead camera sees 42 % of the contact points. A quarter are on the far side of their own finger (the pad's
  face is turned away from it), and a third are hidden by the scene, mostly by the gripper itself above the pads.
- "Pre-contact" is the weaker fingertip's visible share on the last 5 cm of the approach.

## 4. One camera at a time

What if the wrist camera is not there (a different gripper, or a dataset recorded without it)? Choose **Cameras ▸
Overhead D435** (one undo step) and press **Run** again:

    34.6 [26–43] % of reachable grasps · blind at contact

with "35 % of the 133 reachable grasps have every fingertip at least 50 % visible in at least one camera." Wilson's
interval is 27.0 to 43.0 %. Now:

- **Worst grasps** shows six renders from the D435, all top-down grasps with score 0: seen from above, a top-down
  grasp puts the gripper between the camera and the pads. Contact points are drawn green (visible), red (hidden by
  the scene), orange (behind their own finger) and grey (outside the image). Click a render, or a row of the grasp
  table, to put the robot in that grasp's pose; the Camera view then shows it live.
- The 3D view shows every grasp centre coloured by its score: blue where the requirement holds, orange where it does
  not (darker further from it), grey where the arm cannot reach. The Camera view draws the same centres as crosses.
- **What to move** ranks the D435 moved by 2 cm along each world axis or turned by 5° (defaults). The best is "move
  Overhead D435 −2 cm along x of world" at +2 points, but its interval reaches below zero, and the advice says it may
  be noise: small moves of this camera do not change what a top-down grasp hides. A camera at the side of the table,
  looking at the pinch in profile, would see both fingertips; add one (Sensors ▸ Add camera) and run again.
- **Preview** draws where the best move would put the camera; **Apply** makes it (one undo step) and marks the result
  for a new run.

Set **Cameras** back to "All cameras in use" (or undo) before going on.

## 5. Change the requirement

- **Fingertip visible from** (default 50 %) is the share of a patch that must be seen; **Cameras needed** (default 1)
  how many cameras must see it. Changing them, the requirement (any fingertip, the object region, or both) or the IK
  rule needs a new Run; changing **Good from**, **Warning from** or **Serious from** only re-reads the status.
- Other settings live in `project.analysis.studies['grasp_visibility']` (see the model page): the pre-contact
  distance, the lift of the contact location in front of the pad (2 mm), the cube-map size, the q-bucket size, the
  bootstrap resamples and seed, the step of "what to move".

## 6. Export and batch

**Export…** writes the result with provenance: every grasp (id, kind, reachable and why, score, joint vector, each
fingertip's visible share per camera, pre-contact), the cameras, the moves and the gallery.

Without the window, on a saved project that has a grasp set:

    dexvantage cell.dexv --grasp grasp.json

writes the same JSON. On a project without a grasp set the command stops with "There is no grasp set to study: make one
on the Tasks page …" and exit code 2. Large sets (above 2,000 grasps) run in the background in the window and show
partial results; 2,000 grasps on this sample took 112 s on an RTX 5080 and 131 s on the integrated Radeon of a
Ryzen 9 9900X (the model page has the breakdown).

## Known limits

Visibility is judged per cube-map texel, so a point at an edge may go either way; the fingers stay at their home
opening; the contact patch comes from the collision geometry; IK is a local method and obstacles are checked at
contact only; "what to move" moves one camera along one axis at a time on a subset of 250 grasps. The full list is on
the model page.
