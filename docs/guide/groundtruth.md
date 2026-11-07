# Rendered against predicted: a worked example

Do the grasp study's fingertip-visibility predictions hold up in an image? This walk-through renders the UR5e + Robotiq
2F-85 sample in MuJoCo and compares. The model, the equations and the checks are in `docs/models/groundtruth.md`. The
figures are what this build computes on the sample (RTX 5080 through EGL, MuJoCo 3.17.0); they change with your edits.

## 1. Open the sample and run the comparison

    dexvantage --sample ur5e

Simulate ▸ Rendered against predicted visibility… opens the page; **Run** starts it in the background (about 7 s:
inverse kinematics for 50 grasps, about 180 rendered frames, the depth study's voxels). In batch mode:

    dexvantage --sample ur5e --groundtruth groundtruth.json

The headline reads **92.8 %, agrees**: of the 20 992 (camera, contact point) pairs in view, over the home pose, the
32 of the 50 sampled grasps the arm reaches and 8 evenly spaced frames of the sample's scripted pick trajectory, the
grasp study and MuJoCo agree on 92.8 % (agrees from 90 %, mostly agrees from 75 %: defaults, in the tooltip). The
sample of 50 grasps is a default, drawn with seed 0; change it in **Grasps**. The 8 trajectory frames are a default too
(`trajectory_frames`; 0 leaves the trajectories out); the cases table lists them as "Scripted pick (sample) at 2.25 s"
and so on, and on their own they agree on 98.9 % (the grasps alone 91.1 %).

## 2. Read the confusion matrix

| | Rendered visible | Rendered hidden |
|---|---|---|
| Predicted visible | 11 120 | 1 457 |
| Predicted hidden | 64 | 8 351 |

The IoU of the visible-point sets is 88.0 %. Almost every disagreement is one way: the grasp study calls the point
visible and the render does not. The table under it says why:

| Why | Points | |
|---|---|---|
| At an edge | 1 076 | the pixel is within a pixel of a segmentation boundary |
| Rule | 356 | the two occlusion tests agree; the prediction tests the contact location 2 mm in front of the pad, the render the pad face (a pad seen edge-on) |
| Occlusion | 89 | the cube map and the render disagree about what is in front |

So the cube maps and MuJoCo agree on what hides what (89 points, 0.4 %); the rest is how "visible" is defined for a
pad seen edge-on. The camera table shows where: the Wrist D405 (pads face on) agrees on 96.4 %, the Overhead D435 (pads
seen from above, nearly edge-on) on 89.1 %. The grasp study's requirement (every fingertip at 50 % in one camera) gives
the same verdict on all 32 grasps, predicted or rendered.

## 3. Look at the worst disagreements

The thumbnails pair the prediction's Camera view render (left) with MuJoCo's RGB (right) for the poses and cameras
with the most disagreements; each contact point is green where that side calls it visible and orange where hidden,
ringed where the two differ. Click a row of **Poses** to put the robot there: the 3D view shows its contact points
(green both visible, orange predicted only, blue rendered only, grey both hidden) and the Camera view outlines MuJoCo's
segmentation of what it shows, rendered in the background at the view's own pose.

**Preview** jumps to the pose with the most disagreements. **Apply** appears when occlusion disagreements exceed 1 % of
the points (a default): it doubles the grasp study's cube-map size (one undo step). On the sample it stays off.

## 4. Depth validity against rendered depth

| Sensor | Voxels agree | IoU | Surfaces agree | Noise |
|---|---|---|---|---|
| Wrist D405 | 99.89 % | 99.77 % | 99.72 % | σ_z = z² σ_d / (f B), σ_d 0.08 px (a default), f 710.8 px, B 18 mm |
| Overhead D435 | 99.66 % | 99.45 % | 91.61 % | σ_d 0.08 px (a default), f 674.4 px, B 50 mm |

The voxels are the depth study's own (480 000 of the tabletop volume): its cube-map occlusion against the rendered
depth with the noise model added. The surfaces column tests the points the D435 actually sees; on the table 0.9 m away
a cube-map texel spans more depth than the study's 1 mm tolerance, so the study calls part of the table hidden from
itself. Inside the task volume the surfaces agree on 99.6 %.

## 5. Save the scene for MuJoCo

Simulate ▸ Build the MuJoCo scene… (or **Save the MuJoCo scene…** on the page) writes a folder:

    scene.xml          the robots at their files' geometry, the table, lights, one camera per imager
    assets/            the robots' meshes
    scene_notes.txt    each camera with its lens model, the defaults and what was left out

    python -c "import mujoco; m = mujoco.MjModel.from_xml_path('scene.xml'); print(m.ncam)"     # 6

The cameras carry the imagers' intrinsics (sensorsize, focalpixel, principalpixel, resolution); a lens with distortion
gets its pinhole part, since MuJoCo cameras are pinhole.

## 6. The SO-100 and the dexhand samples

`dexvantage --sample so100 --groundtruth gt.json` reads **84.2 %, mostly agrees** (22 905 points over 45 poses; edge
1 440, rule 2 081, occlusion 95; the requirement agrees on all 44 grasps): its jaws' contact patches come from collision
geometry that is not everywhere on the visual mesh, so more points differ by rule. The dexhand sample has no grasp set,
so only its head camera's depth is compared: 99.58 % of the voxels agree.
