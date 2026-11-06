# Multi-finger occlusion {#dexhand}

Modules: `dexvantage/core/dexhand.py` (fingers, contact pads, grasp postures), `dexvantage/core/studies/dex_hand_tactile.py`
(the study, part 1 of spec §7.14 and the extension point of §7.15), `dexvantage/core/robot/dexhand_sample.py` (the
`dexhand` sample). Page: `dexvantage/ui/pages/dexhand.py` (Analysis ▸ Multi-finger occlusion). Validation:
`dexvantage/validation/checks_dexhand.py` (model `dexhand`). Tests: `tests/test_dexhand.py`. Bundling:
`scripts/bundle_dexhand.py`. Batch: `dexvantage project.dexv --dex-occlusion OUT.json`.

A dexterous hand hides its own fingertips. Which finger hides which fingertip's contact patch from which camera
changes with the grasp, so a mount that sees a two-finger pinch well can be blind in a power grasp. The study counts
it, per fingertip and camera, over a sweep of grasp postures or a grasp set, and compares the candidate mounts.

## The hand model

**The bundled hand.** MuJoCo Menagerie's licences differ per model; every dexterous hand folder's own `LICENSE` was read
(Menagerie commit `98a70c5`, 2026-11-06; THIRD_PARTY.md lists them all). The Shadow Hand E3M5 (`shadow_hand`,
Apache-2.0, 24 joints, 3.4 MB) is bundled byte for byte: it is redistributable, small, has the most joints of the
candidates and carries named postures of its own (`keyframes.xml`). The Allegro Hand (`wonik_allegro`, BSD-2-Clause),
LEAP Hand (MIT), Shadow DEX-EE, TetherIA Aero Hand and Sharpa Wave (Apache-2.0) and the G1 with hands (`unitree_g1`,
BSD-3-Clause, 39.7 MB; its README does not name the hand model, so it is not called Dex3 here) are redistributable
too and download on request (Robot ▸ Import from MuJoCo Menagerie). Inspire and Ability hands have no Menagerie
folder.

`ur5e_shadow_hand.xml` puts the hand on the UR5e's flange the way MuJoCo does it:
`MjSpec.attach(hand, site="attachment_site", prefix="hand_")`, then `MjSpec.to_xml()`. The prefix is needed (both
files name a mesh `forearm_0`). One change is made: the hand file turns its forearm by `quat="0 1 0 1"` so the
stand-alone hand lies along x; here that is identity, so the hand continues the flange axis. The script checks the
hand's triangles, in the forearm frame, against the hand file compiled alone: largest difference 3 × 10⁻¹⁶ m.

**Fingers** (`dexhand.structure`). Each tagged fingertip frame ends one finger; the finger's links are the chain of
bodies from the tip up to, not including, the first body that also carries another fingertip. The palm is the
lowest common ancestor of the fingertips; every other link (wrist, forearm, arm) is "wrist and arm". On the sample:
first, middle, ring finger 4 links each, little finger 5 (with its metacarpal), thumb 5 (to `thbase`). A tip that is
itself an ancestor of another tip (a parallel gripper's fixed jaw, as on the SO-100) owns its own link, so its contact
patch counts for it rather than for the palm. A finger's short name is its tip link's name without the tips' common
prefix and suffix, cut only at a word boundary (`gripper_right_pad`, `gripper_left_pad` give `right`, `left`); its
title spells out Shadow Robot's two-letter names and otherwise is the tip's name without the prefix ("Right pad",
"Fixed jaw").

**Contact pads** (`dexhand.pad_faces`). A fingertip's pad is the palmar side of its link's collision mesh: the
triangles whose outward normal n lies within 45° (a default) of the flexion direction

    v = s · a × (c − o)

with a, o the axis and anchor of the nearest joint at or above the tip link, c the mesh centroid and s the sign of
the joint's larger travel (Shadow's distal joints run 0 to 1.57 rad, so s = +1). Outward normals come from the mesh
winding, flipped as a whole when the signed volume is negative. The sample stores the pads as `Robot.contact_pads`
faces (0.1 mm rounding, collapsed triangles dropped), so `core/grasps.py contact_patches` (its rule 1) samples 128
points on each by area, and every study that reads contact patches sees the same pads.

**Grasp postures** (`dexhand.Synergy`, `posture`). A posture is a straight line in joint space,
q(s) = q_open + s (q_closed − q_open), s ∈ [0, 1], on the joints it names, clipped to the limits. For the Shadow Hand
both ends are the model folder's own keyframes: power = "open hand" → "grasp sphere", precision = "open hand" → "two
finger pinch", tripod = "open hand" → "three finger pinch". The keyframes are the model's; the straight line between
them is a sample sweep, not a measured synergy, and no published synergy basis is used. A hand without postures gets
`default_synergies`: every finger joint closed 70 % (a default) of the way from home to the end of its larger travel
(power), or only the thumb and the finger nearest it (precision). Both are labelled sample postures in the result.

## The study

For every case (the open hand, then each posture at closures 1/4, 1/2, 3/4, 1, or every reachable grasp of a grasp
set with the hand closed in the power posture), every camera c and every patch point P:

- **in view:** P projects inside the image through the imager's lens model (`CameraModel.project`), from its optical
  centre o at its pose at the case's joint state;
- **first hit and its owner:** the occluders (robots' visual meshes and tables; task objects only with
  `objects_occlude`, off by default because a fingertip touching an object is hidden by it by definition; the
  camera's own seat left out, `scenegeom.sensor_occluders`) are split into owner groups g: each finger, the palm,
  wrist and arm, task objects, the rest. With d_g(P) the distance from o to group g's first triangle along the ray to P,

      d(P) = min_g d_g(P),   owner(P) = argmin_g d_g(P),   seen(P) ⇔ in view ∧ |P − o| ≤ d(P) + ε

  with ε = 1 mm (the occlusion module's default). d_g is found by exact ray casting (`core/raycast.py first_hit`, one
  grid per group, the ray started 5 mm out: the default backend `raycast`) or read from one cube map per group
  (`auto`, `gl`, `numpy`): a z-buffer of a union keeps the nearest of its parts, so min_g d_g equals the whole
  scene's cube map exactly.

Per finger f and camera c the matrix entry is the share of f's patch not seen by c, averaged over the cases, split
into "hidden by g" for each group and "outside the image"; the largest part is shown as what hides it. A fingertip
counts as seen in a case from 50 % of its patch (`min_visible_pct`, a default), by one camera, or by the rig (a point
seen by any camera). The headline is the share of fingertip-cases the rig sees; status good ≥ 90 %, warning ≥ 60 %,
serious ≥ 30 %, else critical (defaults). The mount comparison ranks the cameras by the share of fingertip-cases each
sees alone, with the share of cases in which it sees every fingertip, the share of patch points outside its image or
hidden, its worst finger and what hides most.

**What to move.** Each unlocked camera is turned by ±10° (a default) in pitch and in yaw about its optical centre and
the rig's figure is counted again. Turning about the optical centre changes only which points are in view: the
first hits are reused, so the search costs no ray casting. The best turn that raises the headline (or, equal, the
rig's mean patch share by 0.5 points or more) is the suggestion; Apply sets that mount pose in one undo step.

**Part 2** (§7.15, the visual-to-tactile handoff and the blind gap, Phase 5) belongs in the same module:
`tactile_handoff()` is its entry point (it says it arrives in Phase 5), `visibility_at(project, q, settings)` is the
part 1 kernel for one joint state, to be called along each approach (`grasps.pre_contact_poses`), and the result
keeps a `handoff` key (null) with the fields `HANDOFF_FIELDS` part 2 will fill.

## Measured on the dexhand sample

Synergy sweep at the home arm pose, 13 cases × 3 cameras × 640 patch points, exact ray casting: 4.9 s (2.7 cases/s;
the cube-map path on the GPU, 286 maps: 3.9 s). The rig sees **93.8 %** of fingertip-cases ("good"), every fingertip
in 69.2 % of the cases; the thumb is seen least (62.4 % of its patch by any camera).

| Not seen (% of the patch) | Palm micro camera | Dorsal wrist camera | Head camera |
|---|---|---|---|
| First finger | 43.0 (outside the image) | 99.6 (palm) | 36.7 (thumb) |
| Middle finger | 61.0 (outside) | 99.3 (palm) | 14.8 (itself) |
| Ring finger | 80.3 (outside) | 99.1 (itself) | 13.7 (itself) |
| Little finger | 71.1 (outside) | 97.5 (itself) | 19.4 (itself) |
| Thumb | 51.9 (outside) | 98.0 (palm) | 83.7 (itself) |

The dorsal wrist camera looks at the back of the hand: the pads face away from it and the palm is in the way. The
palm camera's 120° diagonal view misses the spread fingers of the open hand; in a power grasp the fingers curl in
front of it (on a 30-grasp set with the hand closed in the power posture it sees every fingertip in 100 % of the
grasps, the head camera in 40.7 %).

Cube maps against exact ray casting: the cube-map path (n = 512) differs from ray casting on this sample by up to
6.0 points (thumb, head camera, 77.8 % against 83.7 % not seen) and 4.4 points (ring finger, head camera), in both
directions. A patch point lies on its own surface; seen from 0.7 m a texel is 2.7 mm wide, and on a curved or grazing
fingertip the texel-centre ray meets the surface a few millimetres in front of or behind the point. That is why ray
casting is the default.

## Validation

| Check | Reference | Tolerance | Measured |
|---|---|---|---|
| `dexhand.matrix_vs_analytic`: matrix entries (2 pads × 2 cameras × owners) on a scripted two-finger scene with a palm plate | central projection of finger B's top face onto finger A's pad (camera above) and of the palm plate's edge (low side camera), written out in the check: A hidden by B 44 %, A hidden by the palm 20 %, B's downward pad hidden by B itself 100 %, seen 100 % from the side | 10⁻³ of a pad | 0 on both paths (ray casting; cube maps at n = 2048) |
| `dexhand.owner_vs_bruteforce`: owner of the first hit per point | all-pairs Möller–Trumbore keeping the nearest triangle's index, written out in the check (no grid, nothing shared with `core/raycast.py`); 30 random scenes, 4 500 points | 10⁻³ | 0 |
| `dexhand.cube_owner_union`: min over the group cube maps against one cube map of the scene, sample at the power grasp, 3 cameras | `core/occlusion.py CubeMap` of the union | 10⁻⁶ m | 2.0 × 10⁻⁸ m (GPU, float32) |

Every pad point of the scripted scene is 0.5 mm from a shadow edge on the pad plane (1.7 mrad from the upper camera,
against a texel of 1 mrad at n = 2048), so neither path has an ambiguous point and both must match the closed form
point for point.

## References

Möller and Trumbore 1997 (ray–triangle intersection); Greene 1986 (the cube map); Hartley and Zisserman 2004
(central projection). MuJoCo Menagerie `shadow_hand` (Apache-2.0) and `universal_robots_ur5e` (BSD-3-Clause) at
commit `98a70c5`, https://github.com/google-deepmind/mujoco_menagerie, read 2026-11-06.

## Known limits

The postures are straight lines between two joint vectors (the Shadow Hand's ends are its own keyframes), not
measured synergies; the Shadow Hand's tendon-coupled middle and distal joints are independent entries of q, so an
interpolated posture may couple them differently from the real hand. The pad is the whole palmar side of the distal
link, a fixed point set carried with the link, not the contact area on a given object. The arm's joints stay fixed
during the posture sweep; grasp-set IK holds the hand at one posture for every grasp and does not check
self-collision by default (neighbouring fingers of a closed hand touch in the convex hulls the collision test uses).
With a cube-map backend, visibility is per texel (silhouettes and grazing own surfaces, above). No camera housing
occludes. The re-aim search tries single 10° turns of one camera, not moves or combinations. The hand's mass (the
forearm alone is 3 kg in the file) is not checked against the UR5e's payload. The flexion direction of a fingertip
is taken as its joint's larger travel, which is wrong for a joint whose extension range is the larger. Lighting,
blur and the tactile handoff (Phase 5) are other studies.
