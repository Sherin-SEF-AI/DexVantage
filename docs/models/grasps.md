# Grasp sets {#grasps}

Module: `dexvantage/core/grasps.py`. Page: `dexvantage/ui/pages/tasks.py` (Tasks). Validation:
`dexvantage/validation/checks_grasps.py`. Tests: `tests/test_grasps.py`. Spec §7.3 (grasp sets), §13 (Tasks page).

A grasp set is the input of the grasp self-occlusion study: a list of gripper poses at contact, each with the
direction the gripper moves into the grasp. This module is the only code that writes or reads them.

## The record

`project.tasks.grasp_sets` holds plain records (they round-trip through `project.json` unchanged):

    {'uid', 'name', 'source': 'sampled' | 'imported' | 'trajectory' | 'dataset', 'robot_uid',
     'frame': the link the poses are for (default Robot.gripper_frame, else the model's last link),
     'poses': [{'id': str, 'pose': {'position': [x, y, z], 'quaternion': [x, y, z, w]},
                'approach': [x, y, z], 'kind': 'top_down' | 'side' | 'angled' | 'handoff' | 'imported' | 'trajectory',
                'object_uid': str or '', 'width_m': float or null, 't': float or null}],
     'settings': {how it was made: the sampler's options and which are defaults, the seed, the file, the trajectory,
                  the gripper geometry used, notes}}

`pose` is the gripper frame in the world at contact (metres; quaternion x, y, z, w as `core/frames.py`); `approach`
is a world unit vector. `width_m` is an upper bound of the object's width along the closing axis (its bounding box)
or null; `t` is the trajectory time or null.

API:

| Function | Returns |
|---|---|
| `sample(project, robot_uid, kinds, n, rng_seed, volume_uid=None, handover_uid=None, name=None, tcp=None, approach_axis=None, closing_axis=None, **options)` | a sampled set (not added) |
| `import_file(path, project=None, robot_uid='', name=None)` | an imported set (CSV or JSON) |
| `from_trajectory(project, traj_uid, every=1, name=None, mode='every', close_mm=5, approach_mm=20)` | mode `every`: the gripper frame at every Nth sample; mode `closures`: one grasp where the gripper closes |
| `pad_opening(project, robot_uid, Q)`, `closure_events(opening)` | the gripper's opening at joint vectors; its closing strokes |
| `add_set(project, gs)`, `remove_set(project, uid)`, `get_set`, `sets`, `counts` | the project's records |
| `grasp_matrix(grasp)`, `pre_contact_poses(grasp, distances_m)` | 4 × 4 at contact; backed off along −approach |
| `contact_patches(project, robot_uid, q)` | `{fingertip: {'points', 'normals', 'link', 'rule'}}` (world) |
| `contact_points(project, robot_uid, q)` | `{fingertip: (K, 3)}` world points on the contact pads |
| `object_points(project, object_uid, n, rng_seed=0)` | (n, 3) points on a task object's surface, by area |
| `gripper_geometry(project, robot_uid)` | TCP, approach and closing axes in the gripper frame, and the rule used |
| `ik_targets(project, gs)` | `(link, tcp, targets)` for `ik.solve_batch(..., tcp=tcp)` |

Every function raises `grasps.GraspError` with a plain sentence for inputs it cannot use.

## Contact pads

`contact_patches` decides which surface of each fingertip touches the object, once at the robot's home pose, holds
it in the link's frame and carries it with the link at any q. The rule:

1. **Tagged pads** (`Robot.contact_pads`, the import wizard's pad tagger): a box `{'center', 'size'}` in the link
   frame contributes the face whose outward normal points most nearly at the centroid of the pads on other links
   (12 × 12 points at cell centres); a pad with no pad on another link uses its whole box; a pad given as `'faces'`
   (triangles in the link frame) contributes `PAD_POINTS` area-weighted points on them.
2. **Otherwise the fingertip frames' geometry**: each fingertip link's collision geoms (its visual meshes if it has
   none). Its triangles' outward normals are taken away from each geom's own centroid (exact for the convex pads and
   boxes of the samples). A triangle is a contact face when it is
   - within `PAD_LENGTH_M` = 40 mm (a default) of the finger's distal end along the approach axis (the line from the
     gripper frame's origin to the fingertips' centroid),
   - facing the other fingertips' centroid within `PAD_FACING_DEG` = 45° (a default), its normal measured against
     that direction with its approach component removed, and
   - uncovered: a ray from its centre towards the other fingertips meets none of the same finger's triangles
     (`core/raycast.py`).
   When nothing passes, the rule drops the third test, then the second, then the first, and the patch's `rule` text
   says which. `PAD_POINTS` = 128 points (a default) are drawn on the kept triangles by area, from a generator seeded
   per fingertip, so a patch is the same every time.

On the samples:

- **UR5e + Robotiq 2F-85** (`gripper_right_pad`, `gripper_left_pad`): each pad link has two collision boxes, 22 mm
  wide and 18.75 mm long, stacked to 37.5 mm. The rule keeps exactly the inner face of both boxes on each pad (one
  plane, normals towards the other pad).
- **SO-100** (`Fixed_Jaw`, `Moving_Jaw`): the inner faces of the jaws' distal collision boxes and hull, mean normals
  1.8° from opposite. Both patches span 58 mm along the approach (a triangle that reaches into the
  40 mm band is kept whole) and end within 2.3 mm of each other.

## The gripper geometry

The samplers place the grasp centre, not the gripper frame, on their targets. `gripper_geometry` measures it from the
pads at home, in the gripper frame:

    TCP       t = mean of the pads' centroids (each the area-weighted centroid of its contact faces)
    closing   c = (n₀ − n₁)/|n₀ − n₁| for two pads whose mean inward normals n₀, n₁ oppose within 60°,
              else (C₁ − C₀)/|C₁ − C₀|; for more than two pads the principal axis of their centroids
    approach  a = (t − (t·c) c) / |…|, the direction from the frame origin to the TCP across the closing axis;
              the frame's z axis when that is under 1 mm long

A robot without fingertips or pads gets t = 0, a = z, c = x and the set's notes say these are assumed. The sampler's
`tcp`, `approach_axis` and `closing_axis` override them. Measured:

| Robot | Frame | TCP (mm) | Approach | Closing | Pad gap at home | Pads along approach |
|---|---|---|---|---|---|---|
| UR5e + 2F-85 | gripper_base | (0.0, 0.0, 130.7) | (0, 0, 1) | (0, −1, 0) | 85.4 mm | 37.0 mm |
| SO-100 | Fixed_Jaw | (−0.2, −82.9, 0.7) | (−0.003, −1.000, 0.008) | (−1.000, 0.003, 0.000) | 21.8 mm | 58.1 mm |

The 2F-85's TCP is the pads' centre, 14.3 mm short of Menagerie's `pinch` site (at the pads' tips) on the same axis.

A grasp's gripper pose follows from its TCP target p, approach â and closing ĉ (world unit vectors, orthogonal) by
matching the bases B_l = [c, a, c × a] (gripper frame) and B_w = [ĉ, â, ĉ × â] (world):

    R = B_w B_lᵀ,    position = p − R t

`ik_targets` gives the TCP frame G·[I | t] and the offset to pass as `tcp=` to `core.robot.ik.solve_batch`, so a
position-first solve (the SO-100's five joints) puts the grasp centre, not the frame's origin, on the target. The TCP
is the one the set was made with (`settings['gripper']['tcp']`, which records a sampler's `tcp` override) when it
names the same frame, else the one measured now (an imported set).

## The samplers

`n` grasps go round-robin over the kinds asked for; one `numpy.random.default_rng(rng_seed)` draws everything, so
the same seed and settings give the same set. World z is up (REP-103); "heading" φ is the azimuth of the horizontal
part of the approach.

**Target points.** With task objects whose bounding-box centre is inside the task volume, each grasp picks one
uniformly and uses its world bounding box (centre c, extents e). Without objects: x, y uniform over the volume's
footprint; the support height is the highest table surface (`workspace.surfaces`, a rectangle at height z) under the
point within the volume's height range, else the volume's floor; the TCP is `height_m` (default 30 mm) above it,
capped at the volume's top.

| Kind | Approach â | Closing ĉ | TCP on an object |
|---|---|---|---|
| top_down (top-down pinch) | (0, 0, −1) | heading uniform over 180° (volume points); across the narrower horizontal side of the object's box, turned by up to ±`yaw_jitter_deg` (15°) | top − min(`depth_m`, e_z/2) above the centre; `depth_m` defaults to half the pads' length along the approach |
| side | (cos φ, sin φ, 0) | (−sin φ, cos φ, 0) | the centre |
| angled | (cos φ sin θ, sin φ sin θ, −cos θ), θ = `angled_tilt_deg` (45°) from vertical | (−sin φ, cos φ, 0) | the centre |
| handoff (bimanual handoff) | (cos φ, sin φ, 0) | (−sin φ, cos φ, 0) | targets uniform in the handover volume, in mid-air |

For side and angled grasps φ is the heading from the robot's base to the target, turned by a uniform amount within
±`spread_deg` (60°): the gripper moves outward, as an arm reaching from its base does. For handoff φ is the heading
from this robot's base to the partner's base (the other robot nearest the handover volume; without one, through the
volume's centre), within ±`handoff_spread_deg` (20°); the partner's uid is in the settings. The handover volume is
`handover_uid` or the first volume of kind `handover`; without one, asking for handoff grasps is refused with what to
add.

Every option is a DexVantage default, not a standard; `settings.options` records the values used and
`settings.defaults` lists the ones left at their default. Sampling 2 000 grasps takes 25 ms on either sample (one
core of an AMD Ryzen 9 9900X).

## Imported files

**CSV**: a header row naming at least `id, x, y, z, qx, qy, qz, qw` (metres; the gripper frame's quaternion x, y, z,
w in the world), optionally `ax, ay, az` (approach), `kind`, `object` (or `object_uid`), `width_m` and `t`; columns in
any order, case ignored; commas, semicolons or tabs (a semicolon or tab file may write decimal commas); blank lines
and `#` lines skipped. **JSON**: a list of objects with `id` and either `pose: {position, quaternion}`, `position` and
`quaternion`, or the CSV's keys, with the same optional keys (`approach` as a list); or a grasp set record with
`poses`. A row without an approach gets the gripper's approach axis turned by its orientation (the frame's z axis
when no robot is given; the set's settings say which).

Refused, each with the line (CSV) or item (JSON) and the column: a cell that is not a finite number, a missing
required column, a quaternion of zero length, an approach of zero length or given in part, an unknown kind, a blank
id, a row with the wrong number of values, the same id twice, an empty file, more than 100 000 grasps. Normalised
with a note: quaternions not of unit length. Noted: positions more than 5 m from the origin ("the file must be in
metres").

## From a trajectory

Two modes (the Tasks page's **Where it closes** and **Every N samples**). In both the trajectory is played on the
robot's model (joints it does not name at home; values clipped to the limits, as playing it does), the gripper
frame's world pose comes from forward kinematics, the kind is `trajectory` and `t` the sample's time.

**Every N samples** (`mode='every'`, `every=N`): samples 0, N, 2N, …, the approach the gripper's approach axis there.

**Where it closes** (`mode='closures'`, the page's default): the grasps the trajectory performs.

- The opening at each sample (`pad_opening`): with the contact pads' centroids c_i carried with their links and c̄
  their mean, opening = 2 · mean_i |c_i − c̄|, which for two pads is the distance between them (the 2F-85 on the UR5e
  sample: 85.4 mm open).
- The closing strokes (`closure_events`): with Δ_k the change of opening from sample k to k + 1, a stroke starts at a
  fall Δ_k < −FLAT (0.1 mm, a default), runs while no Δ exceeds +FLAT (a pause is part of it) and ends at its last
  fall. It counts when the opening fell by at least `close_mm` (5 mm, a default) over it.
- The grasp is the gripper frame at the stroke's onset (the arm in place, the fingers still open, as the grasp study
  poses them); the approach is the unit vector from the TCP at the last sample at least `approach_mm` (20 mm, a
  default) away from the onset's TCP, to the onset's TCP; without such a sample (the trajectory starts at the grasp)
  the gripper's approach axis, said in the set's notes.
- The set's settings list each stroke: its onset and end times, how far the opening fell and the opening at the
  onset, and which of `close_mm` and `approach_mm` are defaults. A trajectory in which the gripper never closes that
  far is refused with its opening range; a robot with fewer than two fingertips or pads tagged is refused (use every
  N samples).

## Validation

`validation/checks_grasps.py`, measured on 2026-11-06 (full run; quick run the same to the digits shown):

| Check | Reference | Tolerance | Measured |
|---|---|---|---|
| `grasp.pads_vs_mujoco_boxes` | the 2F-85's pad box geoms from MuJoCo's compiled data (`geom_xpos`, `geom_xmat`, `geom_size` after `mj_kinematics`); the faces facing each other at home, carried to 40 random arm and finger joint vectors | 1 × 10⁻⁶ m | 3.6 × 10⁻¹¹ m; normals on the inner face 80 of 80 |
| `grasp.tcp_vs_pinch_site` | Menagerie's `gripper_pinch` site, on the gripper's axis | 0.5° (and 1 mm off-axis) | 0.0°, 7 × 10⁻¹⁷ m off-axis |
| `grasp.sampler_closed_form` | the samplers' definitions written out again; poses read back with SciPy's `Rotation` | 1 × 10⁻⁹ | 4.4 × 10⁻¹⁶ (1 500 grasps) |
| `grasp.object_points_box` | a rotated box's closed-form signed distance; each face's share of points against its share of area (4σ binomial) | 1 × 10⁻⁹ m | 9.7 × 10⁻¹⁷ m; worst face 0.81σ |
| `grasp.trajectory_closures` | the opening against the distance between the 2F-85 pads' inner-face centres built from MuJoCo's compiled box geoms, along a scripted pick (90 samples: lower, close, lift, open, move, close, open, a 0.01 rad twitch); the strokes found against the sample ranges the script closes over | 1 × 10⁻⁶ m, strokes exact | 6.5 × 10⁻¹⁰ m; strokes (1.25–1.75 s, 3.50–3.75 s) as scripted, the twitch (0.6 mm) not a grasp (measured 2026-11-07) |

Why these tolerances: the meshes are float32 (3 × 10⁻⁸ m at half a metre) and everything else double, so 10⁻⁶ m
catches a wrong face (8 mm away) or frame by four orders of magnitude; the pads are mirror images about the pinch
site's axis, so 0.5° (1.3 mm over 145 mm) is rounding plus mesh export; the closed forms involve nothing but double
precision.

## Known limits

- The samplers do not check reach or collisions: the grasp study's IK does and reports why a grasp is infeasible. On
  the SO-100 (five joints) most side grasps are unreachable by construction (5 of 30 in the worked example).
- Object grasps use the object's axis-aligned world bounding box, not its shape: a grasp on a rotated or concave
  object can miss it, and `width_m` is an upper bound.
- The closing axis is fixed by the kind (a side grasp always closes horizontally); there is no sampler for grasps
  along an object's own axes or from a grasp-quality model.
- The contact patch and the TCP are measured at the home pose. A gripper whose home is closed and asymmetric gets a
  biased TCP; the 2F-85's linkage joints are independent entries of q (BUILD_STATE), so a q that moves one finger
  joint alone carries its pad where the real linkage cannot.
- The inner-face rule takes outward normals away from each geom's centroid: right for convex geoms, possibly wrong
  for faces of a strongly concave mesh used as a fingertip's collision geometry.
- `'dataset'` sets arrive with the dataset readers (Phase 8). An imported row's approach is not checked against its
  orientation.
- Grasps found in a trajectory are its closing strokes: a close without an object (a test close, a regrasp) is a
  grasp too, and fingers that move less than 0.1 mm between two samples are not seen to close. The grasp is posed
  with the fingers at their home opening, as every set is.
