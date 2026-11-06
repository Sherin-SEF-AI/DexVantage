# Cable flex across joints: a worked example on the SO-100 sample

Will the wrist camera's cable survive the arm's joint ranges? The Cable flex page answers this for every cable in the
project. For each run between two attachment points it reports the length the cable needs as the joints move, the
slack loop each joint needs, the tightest bend against the cable's stated minimum, and whether the loop catches on a
link. The model, its equations and its validation are in [../models/cables.md](../models/cables.md).

## 1. Open the sample and run

1. **File ▸ Open sample ▸ SO-100**. The rig has a wrist camera on the fixed jaw (on a bracket, a sample placement)
   and a third-person camera on a tripod. The project has no cables yet.
2. **Analysis ▸ Cable flex across joints** opens the page in the Analysis dock (Placement workspace).
3. Press **Run** (0.3 s). With no cables in the project, the study suggests a route for each sensor mounted on a
   link. For the wrist camera that is six attachment points: the camera's mount on `Fixed_Jaw`, then one point each
   on `Wrist_Pitch_Roll`, `Lower_Arm`, `Upper_Arm`, `Rotation_Pitch` and `Base`. Each run crosses one joint. The
   page says these are suggestions and are not in the project until you Apply.

The headline reads **0 of 1 cables clear · at risk**. The flags are:
- **check**: the minimum bend radius is not stated, so bends are not checked; and no run has a laid length yet, so the
  suggested ones are used;
- **at risk**: run 3's slack loop (`Lower_Arm → Upper_Arm`, across the elbow) passes through `Fixed_Jaw` and
  `Lower_Arm` at home.

## 2. Apply, then choose the cable

4. Press **Apply** (one undo step). The suggested cable is added to the project, and every run's laid length is set
   to its suggested length: the largest length over the joint's range plus a 0.05 m service loop (a default),
   rounded up to 5 mm.
5. Under **Library cable**, choose **Basler Cable GMSL, FAKRA-Z, 1x F/F**. The diameter (3.45 mm), the connectors
   (FAKRA-Z female at both ends) and the minimum bend radius come with it. The radius is the maker's moving figure,
   34.5 mm, and the source line under it cites Basler's page and the date it was read. Set **Provided length** to
   3 m (Basler sells 3, 5, 10 and 15 m).
6. Press **Run** again. The **Runs** table now reads:

| Run | Joint (range from the model file) | Slack loop per joint | Length min–max | Laid | Smallest bend | Status |
|---|---|---|---|---|---|---|
| 1. Fixed_Jaw → Wrist_Pitch_Roll | Wrist_Roll (±2.79 rad) | 140 mm | 0.048–0.138 m | 0.190 m | 24.7 mm | at risk |
| 2. Wrist_Pitch_Roll → Lower_Arm | Wrist_Pitch (±1.66 rad) | 177 mm | 0.065–0.192 m | 0.245 m | 38.1 mm | at risk |
| 3. Lower_Arm → Upper_Arm | Elbow (−0.174 to 3.14 rad) | 283 mm | 0.087–0.319 m | 0.370 m | 70.2 mm | at risk |
| 4. Upper_Arm → Rotation_Pitch | Pitch (−3.32 to 0.174 rad) | 254 mm | 0.078–0.282 m | 0.335 m | 58.3 mm | clear |
| 5. Rotation_Pitch → Base | Rotation (±1.92 rad) | 210 mm | 0.051–0.211 m | 0.265 m | 50.8 mm | at risk |

The allowed bend radius is 34.5 mm × 1.5 = 51.8 mm. The bend factor of 1.5 is a DexVantage default margin for
flexing, labelled "default" under **Settings**. The flags say what to do:
- "Run 1 bends to 24.7 mm with Wrist_Roll at 159.9°, below the allowed 51.8 mm (the stated minimum × 1.5). It bends
  round the joint at the wrap radius: move the attachments further from the joint axis (the wrap radius is at most
  the nearer one's distance from it)." The suggested point on `Wrist_Pitch_Roll` is only 23.5 mm from the roll axis.
- Run 2 bends at 38.1 mm, also at its wrap radius.
- Run 5: "Its slack loop is tight there: lay less slack or move the attachments apart." It bends to 50.8 mm at
  Rotation −90.5°, just under the allowed radius.
- Run 3 still snags at home.

The cable needs 1.14 m between its first and last attachment at the runs' longest, and 1.41 m as laid. The 3 m cable
is long enough, so there is no pull. The run to the host is extra and not modelled.

**With the bend factor at 1.0** (the maker's figure with no margin; set it under Settings, one undo step), only run 1
is still flagged for its bend. The page shows which setting you used and labels it "yours (default 1.5)".

## 3. Test a motion

7. Import a trajectory for the arm under **Tasks** (a CSV of t and joint values). This example used a "reach and
   roll" over 4 s: Rotation −1 → 0 → 1 rad, Elbow 1.57 → 0.6 → 1.57 rad and Wrist_Roll −1.57 → 0 → 1.57 rad. Press
   **Run**. Snags are now tested at home and at 91 poses along it (2° steps, a default). Run 1 snags at
   t = 1.73 s; run 2 at 20 of the 92 poses, first at t = 1.42 s; run 3 at 81. Runs 4 and 5 stay clear.
8. The 3D view draws each run's loop at the pose on screen, in its status colour. Drag the joint sliders or play the
   trajectory to see where it catches. The Camera view draws the cables too, so you can see whether a cable crosses
   the wrist camera's own image. **Preview** draws the loops with the suggested laid lengths, dashed.

## 4. Fix and export

- Edit an attachment point in the **Attachment points** table: its link, its x, y, z in the link frame (mm), and the
  cable laid to the next point. Each edit is one undo step. Moving the `Wrist_Pitch_Roll` point further from the
  roll axis raises run 1's wrap radius. **Use the sensor's position** puts the selected point at the camera's mount.
- A larger **Wrap radius** (a guide ring or a spiral wrap round the joint) helps only up to the nearer attachment's
  distance from the axis. Set the attachments first.
- **Export…** writes the harness sheet. It can be JSON (with provenance), CSV (one row per run: cable type,
  library entry, minimum bend radius with its source, slack loop per joint, laid lengths, connectors, strain relief
  and status) or HTML (the same content as tables) for the people who build the rig. In batch:
  `python -m dexvantage project.dexv --cables cables.json`.

## The UR5e sample: what a full-turn joint means for a cable

On the UR5e sample, the wrist D405's suggested cable with the VA Imaging active USB 3.0 cable (50 mm moving radius,
75 mm allowed) needs 4.14 m between flange and base at the runs' longest. That is because the UR5e's model file
gives five joints a range of ±2π. Run 6 (across the shoulder pan) needs 0.78 m of slack on its own, and run 4 (the
elbow, ±π) needs 1.28 m. A 3 m cable pulls. The wrist and shoulder-lift runs bend to 37–65 mm, round wrap radii of 36–62 mm, below the
allowed 75 mm. The study is right:
a cable fixed on the outside of the arm cannot follow a joint that turns twice round. Limit the joints' ranges in the
controller (the study reads the model file's limits; import a model with your ranges), or route through the arm.
