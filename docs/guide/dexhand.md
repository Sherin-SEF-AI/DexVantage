# Multi-finger occlusion: a worked example on the dexhand sample

A dexterous hand hides its own fingertips, and which finger hides which changes with the grasp. The Multi-finger
occlusion page counts, for each fingertip and each camera, how much of the fingertip's contact patch the camera does
not see, and what is in the way. The model, its rules and its validation are in
[../models/dexhand.md](../models/dexhand.md).

## 1. Open the sample

**File ▸ Open sample ▸ UR5e and Shadow Hand with a palm, a dorsal wrist and a head camera** (or
`dexvantage --sample dexhand`). The UR5e carries the right Shadow Hand on its flange, pointing down, palm towards
+x. Three cameras sit at sample placements (not published figures), each with a library device:

- **Palm micro camera**: an OmniVision OVM6948 (0.65 mm module, 120° diagonal view) set into the palm, tilted 25°
  towards the fingers;
- **Dorsal wrist camera**: an e-con Systems See3CAM_24CUG on the back of the hand's forearm, 65 mm off its axis,
  angled to look past the knuckles;
- **Head camera**: an Intel RealSense D435 (its RGB imager) 0.55 m in front of the palm and 0.4 m above it.

The fingertips (the five distal links) and their contact pads (the palmar side of each distal link's mesh) are
already tagged; the Robot page shows them. The sample's grasp postures are sweeps between the hand file's own
keyframes: power (open hand to "grasp sphere"), precision (to "two finger pinch") and tripod (to "three finger
pinch"). They are sample sweeps, not measured synergies, and the page says so.

## 2. Run the posture sweep

1. **Analysis ▸ Multi-finger occlusion.** Cases: **Grasp postures**, **Arm as on screen**.
2. Press **Run** (about 5 s: 13 cases, 3 cameras, 640 patch points, exact ray casting).

The headline reads **93.8 % · fingertips seen**: in 93.8 % of the fingertip-cases at least half of the fingertip's
patch (a default) is seen by some camera. Every fingertip is seen in 69.2 % of the cases. Hover the figure for the
thresholds (90, 60 and 30 %, defaults).

## 3. Read the matrix

The heatmap has a row per finger and a column per camera; the colour runs away from the background as more is hidden (darker in the light theme, lighter in the dark one). Each cell gives the share of the
patch the camera does not see over the 13 cases, and what hides most:

- the **Dorsal wrist camera** column is almost black (97–100 %), "by palm" or "by itself": it looks at the back of
  the hand, the pads face away from it and the palm is in the way;
- the **Palm micro camera** misses 43–80 % "outside the image": its 120° view cannot hold the spread fingers of an
  open hand;
- the **Head camera** sees the four fingers well (14–37 % not seen) but the thumb hardly at all (84 % not seen, by
  itself: the thumb's pad faces the fingers, away from the camera).

Click the Head camera column. **Who hides whom** splits its losses by owner: the first finger loses 37 %, 24 points of it
to the thumb and 12 to its own back. Hover a cell for the exact split.

## 4. Compare the candidate mounts

The table ranks the cameras by the share of fingertip-cases each sees alone: Head camera first (75.4 %), Palm micro
camera (49.2 %), Dorsal wrist camera last (0 %). No single camera sees every fingertip often, which is why the rig's
figure is higher than any one of them.

## 5. Look at one case

In **Sweep**, the rows are the cases and the cells the not-seen share for the chosen camera. Choose
"Power grasp (sphere) · 100 % closed" and press **Show this case**: the 3D view takes the posture and draws each patch
point, seen by the chosen camera in its colour and not seen in grey. Open the Camera view of the Palm micro camera:
the patch points are drawn there too, each finger in its colour, with "First finger · 84 % seen" in the legend. In
this posture the fingers curl in front of the palm camera, and it sees 76–88 % of each patch.

## 6. What to move

The headline's line suggests turning the Palm micro camera 10° left about its optical centre: the share of patch
points seen by any camera rises from 81.0 % to 81.8 % (the fingertip-cases stay at 93.8 %). **Preview** draws the
suggested optical axis dashed beside the current one; **Apply** turns the camera (one undo step) and runs again.

## 7. Over a grasp set

Make a grasp set on the Tasks page (for example top-down, side and angled, 30 grasps, seed 1), then choose Cases:
**Grasp set** and the set, and Run. Each grasp is solved by IK with the hand closed in the power posture (self-
collision not checked by default: the closed fingers touch in their convex hulls). On that set all 30 grasps are
reached (12 s); the Palm micro camera sees every fingertip in 100 % of them, the Head camera in 40.7 %.

## Batch

    dexvantage --sample dexhand --dex-occlusion out/dexhand.json

writes the result with provenance. The settings live in `project.analysis.studies['dex_occlusion']` (cases' source,
postures, steps, grasp set, thresholds, backend; see the module docstring).
