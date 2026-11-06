# VLA token resolution: a worked example on the UR5e sample

Check how many of π0's ViT tokens a 50 mm cube and the gripper's pads cover in each camera, see the patch grid and
the crop in the Camera view, follow the suggestion, and export Plücker ray maps. The model and its validation are in
[../models/vla.md](../models/vla.md).

## 1. Set up

1. **File ▸ Open sample ▸ UR5e and Robotiq 2F-85 with a wrist D405 and an overhead D435**.
2. Add a task object (**Workspace ▸ Add box object**, then set its size and position in the Inspector): a box, 50 × 50 × 50 mm, at x = −0.134, y = 0.492, z = 0.025 m
   (the centre of the sample's tabletop volume, resting on the floor plane).
3. On the **Policies** page remove the sample's own "π0 (sample)" and add **π0 (pi0)**: input 224 × 224, *resized
   with padding*, ViT patch 14 px, all from its cited sources. On the **Tasks** page delete the sample's grasp set
   "Tabletop grasps (sample)" for now (step 2 runs without one; section 3 samples a set again).

## 2. Run at the home pose

Open **Analysis ▸ VLA token resolution** (it is the Grasp study workspace's page) and press **Run**. With no grasp
set the arm stands at the pose on screen; the notes say so. At the sample's home pose the headline reads
**4 tokens · too few tokens**: the cube in the Overhead D435 touches 4 tokens but spans only 0.6 × 0.6 of them.

The mapping line for **Overhead D435** reads: crop box *u* −0.5 … 1919.5, *v* −0.5 … 1079.5 px (1920 × 1080, the
whole stream); scale 0.1167 × 0.1167 input px per sensor px; padding 44 %; patch 14 px → 16 × 16 = 256. A 16:9 image
padded into a square leaves 7 of the 16 token rows black. Hover over the line for each figure's source and the
equation. For the **Wrist D405** (1280 × 720 at 0.175 input px per sensor px) the cube spans 1.1 × 1.4 tokens:
also below the 2 × 2 default.

## 3. At the grasp poses

On the **Tasks** page sample a grasp set (top-down pinch, 20 grasps), then Run again. Twelve grasps are counted,
evenly through the set (a default), each at pre-grasp (100 mm back along the approach, a default) and contact. The
wrist camera is posed by IK for each; at those poses the cube spans 2.4 × 2.7 tokens at the worst one (12 to 27
tokens touched), and the warning for the D405 goes. The overhead camera does not move: still 0.6 × 0.6. The
fingertip table: each pad touches 6 tokens in the wrist camera and 1 to 2 in the overhead one.

## 4. The overlays

Show the **Camera view** of the Overhead D435 while the page is shown: the crop boundary (the whole image), the
14-px patch grid over the rows the input keeps, and the cells covering the cube. The 3D view draws those cells'
footprints on the cube (each token's boundary cast from the camera onto it); untick *Show token footprints in 3D*
to hide them.

## 5. Act on it

The suggestion reads: *Set Overhead D435's stream to 640 × 480 (now 1920 × 1080): for π0 the worst warned object
then spans 0.9 tokens on its narrower side instead of 0.6.* **Preview** says it; **Apply** sets the stream (one undo
step, Ctrl+Z restores it) and runs again. A 4:3 stream wastes less of the square input on padding, but 0.9 is still
below 2: move the overhead camera closer, choose a narrower lens, or mark the cube not critical if the policy does
not need it from that view (the *Critical objects* boxes; each change is one undo step). The smallest span (default
2 tokens) and the pre-grasp distance are your settings, labelled when they differ from the defaults.

## 6. Export

**Export…** writes the result with its provenance. **Export Plücker rays…** asks for a folder and writes, per camera
π0 takes, `plucker_0_pi0_<camera>.npy` (file names keep letters, digits, `-` and `_`; 224 × 224 × 6: *d* then
*m* = *o* × *d*, in the robot base frame), `.safetensors` (the same plus `valid`, `u`, `v`, `origin`, `rotation`) and
`.json` (the lens, the crop box in sensor pixels, the frame and the provenance), at the pose on screen. The padded rows are zero with `valid = false`.

In batch: `dexvantage project.dexv --vla-tokens tokens.json` (a project without a policy exits with the sentence
"There is no policy in the project: add one on the Policies page …").
