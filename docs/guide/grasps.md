# Grasp sets: a worked example on the UR5e and SO-100 samples

The grasp self-occlusion study asks, for many grasps, whether the cameras see the fingertips and the object at
contact. Its input is a grasp set: gripper poses at contact with their approach direction. The Tasks page makes,
imports, lists and deletes them. The model, the rules and their validation are in
[../models/grasps.md](../models/grasps.md).

## 1. Sample a set on the UR5e

1. **File ▸ Open sample ▸ UR5e and Robotiq 2F-85 with a wrist D405 and an overhead D435.**
2. Open **Tasks** (Tasks ▸ Tasks). The sample comes with one set, "Tabletop grasps (sample)" (200 grasps: top-down,
   side and angled, seed 0), so the headline reads `200 grasps in 1 set`. To follow these steps from an empty list,
   select it and press **Delete set**; the headline then reads `0 grasps · none yet`.
3. Under **Sample a grasp set**: robot "UR5e + Robotiq 2F-85", task volume "Tabletop (sample)", tick **Top-down
   pinch**, **Side grasp** and **Angled**, 90 grasps, seed 1.
4. Press **Preview**: the grasps are drawn in the 3D view as dashed glyphs (a stem along the approach, a palm, two
   fingers ending at the grasp centre; blue top-down, orange side, green angled) and in the Camera view. Nothing is
   added yet.
5. Press **Add set**. The headline reads `90 grasps in 1 set · ready`; the table lists
   "Sampled: top-down pinch, side grasp, angled (90)", source `sampled`, 90 poses, 30 of each kind. **Edit ▸ Undo**
   removes it in one step. A set of 1 000 grasps or more is sampled on a worker thread (the status bar shows
   "Sampling … grasps"); the preview or the new set appears when it is done.

The status bar says why the grasps sit where they do: "No task object is in Tabletop (sample): grasp points are on
its support surface, 30 mm up." The sample's volume holds no objects, so the grasp centres are 30 mm (a default)
above the table, spread over the volume's footprint. The selected set's note gives the gripper geometry used: the
grasp centre (TCP) 130.7 mm out along `gripper_base`'s z axis, between the 2F-85's pads, measured from their inner
faces at home.

What the UR5e can reach (IK with 8 seeds, self-collision checked, `ik.solve_batch(..., tcp=…)` through
`grasps.ik_targets`): top-down 30 of 30, side 28 of 30 (2 "no convergence"), angled 29 of 30 (1 "self-collision").

## 2. Grasp an object

Add a box with **Workspace ▸ Add object** inside the task volume and sample again. The table's note now reads
"Grasp points on 1 task object in Tabletop (sample)": top-down grasps close across the box's narrower side, their
centre half the pads' length (18.5 mm) below its top; side and angled grasps centre on the box; each pose records
the object's uid and `width_m`, the box's extent along the closing axis.

## 3. The SO-100

Open the SO-100 sample and sample the same three kinds. The gripper frame is `Fixed_Jaw`; the TCP sits between the
jaws' inner faces, 82.9 mm along the jaw, with the jaws 21.8 mm apart at home. The arm has five joints, so IK reaches
the grasp centre first and turns the jaw as far as it can: top-down 26 of 30 reached (orientation exact for 1), side
5 of 30 (14 "joint limit", 11 "out of reach"), angled 24 of 30. A side grasp needs a sixth joint, which the SO-100
does not have; keep to top-down and angled sets for it.

## 4. Import a file

**Import file…** reads CSV or JSON for the robot chosen in the form. A minimal CSV:

    id,x,y,z,qx,qy,qz,qw,ax,ay,az,kind,object
    a,-0.10,0.50,0.03,1,0,0,0,0,0,-1,top_down,
    b,-0.20,0.45,0.05,0.7071068,0,0,0.7071068,,,,,cup

Positions are metres, the quaternion is the gripper frame's (x, y, z, w) in the world; a row without ax, ay, az gets
the gripper's approach axis turned by its orientation. A mistake is named with its line and column, for example
"Could not import grasps.csv: Line 3, column y: "abc" is not a number. Write positions in metres and the quaternion
as four numbers (x, y, z, w)." A file in millimetres is imported with the note "1 position is more than 5 m from the
origin: the file must be in metres, not millimetres."

## 5. From a trajectory

With a trajectory in the project (**Robot ▸ Import trajectory…**), pick it next to **Add from trajectory**.

- **Where it closes** (the default) adds the grasps the trajectory performs: one each time the opening between the
  contact pads falls by 5 mm or more (a default), at the moment the fingers start to close, approaching the way the
  tool centre came in over its last 20 mm. On the UR5e sample with the scripted pick of the validation check
  (`validation/checks_grasps.scripted_pick`: lower, close, lift, open, move, close) the set holds two grasps, at
  1.25 s and 3.50 s, and the message says "the gripper closes 2 times in the trajectory". Run the grasp study on it to
  see whether the cameras see the fingertips at those moments.
- **Every N samples** adds the gripper frame at samples 0, N, 2N, …, each with its time.

Tick **Swept volume** to see where the arm goes along the chosen trajectory: the robot drawn see-through at 8 evenly
spaced times (a default), fainter early and stronger late (`core/robot/swept.py`; the first time for a robot its
meshes are simplified as a background job, a few hundred milliseconds).

## 6. Export and delete

**Export…** writes the selected set as JSON with the provenance block. **Delete set** removes it in one undo step.

## In scripts

    from dexvantage.core import grasps
    from dexvantage.core.robot.samples import sample_project
    p = sample_project('ur5e')
    gs = grasps.sample(p, p.robots[0].uid, ['top_down', 'angled'], 500, rng_seed=1)
    grasps.add_set(p, gs)
    tips = grasps.contact_points(p, p.robots[0].uid, q)        # {'gripper_right_pad': (128, 3), ...}
    approach = grasps.pre_contact_poses(gs['poses'][0], [0.02, 0.04, 0.06])
    picks = grasps.from_trajectory(p, traj_uid, mode='closures')  # one grasp per closing of the gripper
