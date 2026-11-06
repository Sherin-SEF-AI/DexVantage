# Grasp self-occlusion {#grasp}

Module: `dexvantage/core/studies/grasp_visibility.py` (study id `grasp_visibility`, batch flag `--grasp OUT.json`).
Page: `dexvantage/ui/pages/grasp.py` (Analysis ▸ Grasp self-occlusion, Grasp study workspace). Checks:
`dexvantage/validation/checks_grasp.py` (`grasp.visibility_vs_raycast`, `grasp.bootstrap_coverage`,
`grasp.qbucket_vs_exact`). Tests: `tests/test_grasp_visibility.py`. Timing script: `scripts/measure_grasp.py`. Spec
§7.3 and the §16 performance budget. Builds on [grasps.md](grasps.md) (grasp sets, contact patches, the TCP),
[ik.md](ik.md), [occlusion.md](occlusion.md), [optics.md](optics.md) and [camview.md](camview.md).

The question: at the moment of grasp, can the rig's cameras see the fingertips and the object, and on the way in? A
policy trained on images can only learn contact it can see.

## What it computes

**1. Inverse kinematics.** The grasp set (`project.tasks.grasp_sets`, read through `core/grasps.py`; settings
`grasp_set`, default the first set) holds gripper-frame poses at contact. `grasps.ik_targets` turns each into a target
for the grasp centre (TCP), and `ik.solve_batch` solves them all at once: damped least squares with joint-limit
avoidance and `ik_seeds` seeds (default 8), retrying unsolved targets from fresh seeds (see [ik.md](ik.md)). A
solution must be free of self-collision (`ik`'s own test) and of the project's obstacles by `collision_margin_m`
(default 0 m): `collision.obstacle_distance` against the task objects, the tables and the static geoms of the robot's
file. A grasp whose solution hits an obstacle is solved again twice, each time from seeds drawn uniformly within the
limits and keeping the solution nearest such a draw (`nearest_q0`), so that another IK branch can be found; if none
is clear, it is infeasible with the reason `obstacle collision`. Infeasible grasps are counted by reason
(`out of reach`, `joint limit`, `orientation unreachable`, `self-collision`, `no convergence`, `obstacle collision`),
and the result names the link and obstacle of each obstacle hit (`ik.obstacle_pairs`). Among several solutions,
`rule` picks one: `nearest_home` (default) or `most_manipulable`. Joints that do not move the gripper (its fingers,
another arm) stay at the joint state of the settings `q` (else home).

Optional backends (`ik_backend`): `pinocchio` (`pip install pin`) and `curobo` (NVIDIA cuRobo, CUDA). They are
imported only when asked for, only for a robot read from a URDF (the file is in the project), and every solution they
return is checked again with DexVantage's forward kinematics (target within 10 × the IK tolerance) and self-collision
test. Without the package the study says how to install it or to use `dls`. Neither package was installed where this
was built, so these two paths are untested here.

**2. Contact points.** Each fingertip's contact patch (`grasps.contact_patches`: the robot's tagged pads, else the inner
faces of the fingertip links' collision geoms, 128 points per pad) is held in its link's frame and carried to each
solution q. Because the patch comes from the collision geometry and the cameras see the visual meshes, each point is
first moved onto the visual surface in front of it: a ray from `snap_mm` (default 5 mm) outside the point, back along
its normal, against the robot's visual meshes at home. On the UR5e sample no point moves; on the SO-100, 3 of the fixed
jaw's 128 points and 14 of the moving jaw's do. For a grasp on a task object, the object's grasp region is the part of
`object_points` (400 points drawn on its surface by area, seeded) within `region_radius_m` (default 30 mm) of the grasp
centre (the 8 nearest when fewer lie within it).

**3. Visibility.** What is tested is the *contact location*: each patch point P lifted `lift_mm` (default 2 mm) along
its outward normal n, just in front of the pad's face, where the object meets it. From a camera with optical centre o
and rotation R, the point is visible when

| Test | Definition |
|---|---|
| in image | `CameraModel.project(Rᵀ (P − o))` is ok: in front, within the lens model's valid angle, inside the image (the imager's library model at its stream) |
| not hidden | nothing of the scene is nearer along the line of sight to P + lift · n, less ε: `CubeMap.visible`, ε = `eps_mm` (default 1 mm), with the cube map of distances around o of the robots' visual meshes posed at q, the task objects and the tables (the camera's seat on its mount link left out by `scenegeom.sensor_occluders`' rule) |
| facing (optional) | with `max_view_angle_deg` set (off by default): the angle between n and o − P is at most that |

Why the lift. The two inner faces of a parallel gripper face each other, so no camera outside the gap sees both of them
head-on, and a camera looking at the pinch from the side (in profile) sees both at a grazing angle. Testing the faces
themselves (an earlier draft required the face to turn within 80° of the camera) called every profile view blind: on
the SO-100 sample, 0 % of grasps had both fingertips at 50 % visible. Testing the location in front of the face asks
what an image shows: a camera behind a finger has that finger in the way (the segment to the lifted point crosses the
pad), a camera in profile sees both contact locations. The object region is tested the same way (lifted along the
object's normals).

The camera of a sensor is chosen as in coverage (`coverage.pick_imager`): a depth sensor's RGB imager when it has one.
Settings `sensors` limits the study to some cameras; `modes` and `imagers` as in coverage.

**Culling.** Each map draws only the triangles that can lie between o and the points: those whose bounding sphere (per
link, then per triangle; centre and radius held in the link's frame) meets the cone from o around the lifted points,
widened by 2° and 2 cm. The test is conservative: with w = C − o, the cone's axis a and half-angle θ, a sphere of
radius r is kept when |w − (w · a) a| cos θ − (w · a) sin θ ≤ r (the distance from its centre to the cone's surface
when it is within θ + 90° of the axis; beyond that the nearest point is the apex, which is never nearer) and
|w| − r ≤ the cone's length. A triangle outside the cone cannot hide any of the points, so a culled map answers as the
full scene's does for them; the validation check compares with ray casting of the full scene. A camera with no point
in its image draws no map.

**4. The q-bucket cache.** The joint state is quantised: the grasp robot's joint vector divided by `q_bucket_deg`
(default 0.1°; prismatic joints by `q_bucket_mm`, default 0.5 mm) and rounded is the bucket key. Every grasp or
pre-contact pose in a bucket is evaluated at the bucket's first joint vector: its maps, its camera poses and its
contact points (world-fixed points, the object region, keep their own positions). A map is reused when the points lie
inside the cone it was drawn for (else it is drawn again). Grasps are evaluated in a seeded random order with the
members of a bucket together, so partial results are an unbiased sample and a bucket's map is used while it is fresh
(an LRU of 256 MB of maps). Why not reuse a map for a slightly different q: a map holds the pads' own surfaces, and at
another q the pad points move by up to about 2.6 mm against them, more than ε; an early version did that and the
reported visible share of a patch changed by up to 75 points. Quantising q keeps the points and the map consistent.
On sampled sets almost every grasp has its own bucket (hit rate 0 % on the samples); duplicates and the still phases
of a trajectory reuse.

**5. Pre-contact.** The gripper backed off along −approach by `precontact_cm` (default 5 cm) in `precontact_steps`
equal steps (default 2: 2.5 and 5 cm). Each step is solved by IK from the previous one (rule `nearest_q0`, one seed
plus home, so the arm does not change branch; self-collision checked, obstacles not), and the same visibility measured
there.

## Measures and the score

Per grasp and camera: each fingertip's visible fraction (the share of its patch's points visible; the points are spread
evenly by area, so this is the visible share of the patch), the object region's visible fraction, and both at each
pre-contact step.

**The requirement** (`requirement`, `min_visible`, `min_cameras`, `same_camera`; defaults `all_fingertips`, 50 %, 1,
true: one camera sees every fingertip at 50 % or more; DexVantage defaults, not standards) gives each grasp a score:

| requirement | per camera c | over cameras |
|---|---|---|
| `all_fingertips` | s_c = min over fingertips of the visible fraction | the `min_cameras`-th largest s_c |
| `any_fingertip` | s_c = max over fingertips | the `min_cameras`-th largest |
| `object` | s_c = the object region's fraction | the `min_cameras`-th largest |
| `fingertips_and_object` | s_c = min(min over fingertips, object) | the `min_cameras`-th largest |

With `same_camera` false (`all_fingertips`, `fingertips_and_object`), each fingertip takes its own `min_cameras`-th
best camera and the score is the weakest of those. The requirement holds when score ≥ `min_visible`.

**The Contact Visibility Score** (CVS) is the share of the feasible grasps where it holds (for the object requirements,
of those with an object region), CVS = k / n. Its interval is the percentile bootstrap [Efron 1979] at `ci` (default
95 %) from `bootstrap` resamples (default 2000) seeded by `rng_seed`: the resampled mean of a 0/1 indicator of n
grasps is Binomial(n, p̂)/n exactly, so that is drawn directly instead of resampling the grasps (the same
distribution). Wilson's score interval [Wilson 1927],

    (p̂ + z²/2n ± z √(p̂(1 − p̂)/n + z²/4n²)) / (1 + z²/n),

is given beside it. When every grasp passes (or none does) the bootstrap interval is a point, and the page says so and
shows Wilson's. Also reported: the CVS of all grasps (infeasible ones failing), the both-fingertips-visible rate per
camera and for the rig, the object's visibility, the pre-contact visibility per step, and per camera why contact
points are not seen: outside the image, behind their own finger (n · (o − P) ≤ 0, the face turned away), or hidden by
the scene. The headline's status is `good` from `good_pct` (90 %), `warning` from `warn_pct` (70 %), `serious` from
`serious_pct` (40 %), else `critical` (defaults, not standards).

## Outputs

- **Per-grasp table** (`grasps`, `decode(doc)`): id, kind, object, TCP, feasible and reason, score, meets,
  manipulability, q (float32, NaN when infeasible), the fingertip fractions (N, C, F), the object fractions (N, C) and
  the pre-contact fractions (S, N, C, F); arrays zlib-compressed in base64 as coverage stores voxels.
- **Heatmap over the task volume**: the page draws each grasp centre as a small octahedron coloured by score on a
  diverging ramp centred on `min_visible` (blue at or above, orange below, grey unreachable), and the Camera view the
  same centres as crosses by pass, fail, unreachable.
- **Worst-case gallery**: Camera view renders (`core/camrender.render`, `gallery_px` wide, default 320) of the
  `gallery` (default 6) lowest-scoring feasible grasps from every camera (up to 4), each contact point marked by state.
- **What to move**: on a seeded subset of `sens_grasps` feasible grasps (default 250), one camera at a time moved by
  ±`sens_step_m` (default 2 cm) along each axis of its mount's parent frame (the world for a fixed mount, the link for
  one on a robot) or turned by ±`sens_step_deg` (default 5°) about its optical x (tilt), y (pan) or z (roll) axis
  (the turn is about the optical centre, written back as a mount pose: mount′ = mount · T_bo · Rot · T_bo⁻¹). The CVS
  on the subset is recomputed with only that camera changed; moves are ranked by ΔCVS (paired; a paired bootstrap
  interval of 1000 resamples) and then by the mean change of the score. A turn reuses the maps (they are aligned with
  the world, and the centre does not move); a move draws new ones at the new centre. The best move (the first with
  ΔCVS > 0, else the first that raises the mean score) carries its new mount pose: the page's Apply.
- **Throughput**: grasps per second overall and of the visibility step, maps drawn and per second, the cache hit rate,
  the renderer, and the timing of each phase. Above `BACKGROUND_GRASPS` (2,000) grasps the study hands partial
  results to the page (`ctx.partial`, the running CVS and throughput).

## Performance budget (measured)

`scripts/measure_grasp.py --n 2000` on the UR5e sample: 2,000 grasps sampled by `grasps.sample` (top-down, side and
angled, seed 0), default settings (cube maps of 512 texels, two pre-contact steps, sensitivity on 250 grasps, a
gallery of 6). 1,322 grasps are reachable (the rest mostly bring the wrist into the table); 10,808 cube maps are drawn.
Measured 2026-11-06 with the GPU through EGL in the study's own process, numpy 2 on Python 3.14:

| Renderer (EGL device) | Total | IK | Pre-contact IK | Visibility | What to move | Gallery | Grasps/s | Maps/s |
|---|---|---|---|---|---|---|---|---|
| NVIDIA GeForce RTX 5080 (EGL device 0) | 111.6 s | 13.2 s | 1.0 s | 68.8 s | 27.3 s | 1.1 s | 17.9 | 171 |
| AMD Radeon Graphics, the Ryzen 9 9900X's integrated GPU (radeonsi, EGL device 2) | 131.3 s | 13.3 s | 0.9 s | 83.7 s | 31.8 s | 1.3 s | 15.2 | 132 |
| RTX 5080, `cube_n` 256 | 85.5 s | 13.3 s | 0.9 s | 51.1 s | 18.7 s | 1.1 s | 23.4 | 283 |
| RTX 5080, `sensitivity` off | 83.6 s | 13.2 s | 1.0 s | 68.0 s | — | 1.2 s | 23.9 | 173 |

The two GPUs give identical results (the same 1,322 grasps, CVS 100 %). The integrated GPU is only 18 % slower: the
time goes into each map's upload, draw and read-back of 6 × 512² floats and the numpy work around it (`occlusion.py`),
not into the triangles. In the window, PyOpenGL may already be loaded for the display, and then the maps are drawn in a
worker process (`occlusion._GLWorkerClient`); forcing that path (`PYOPENGL_PLATFORM=glx`, 300 grasps, sensitivity off)
measured 100 maps/s against 168 in process, so expect about 1.5 × the visibility time there. Without a GPU
(`backend: numpy`) the reference rasteriser draws 7 maps/s (20 grasps, sensitivity off: 14.4 s), about 25 × slower:
use a smaller set. Knobs, in the order they help: `sensitivity` off or a smaller `sens_grasps`; `cube_n` 256 (a texel
of 7.8 mrad at a face centre: 6 mm at the overhead camera's 0.8 m, so check the validation figures before relying on
it); `precontact_steps` 1.

## Validation

| Check | Reference | Tolerance | Measured (seed 20261027, full run, RTX 5080) |
|---|---|---|---|
| `grasp.visibility_vs_raycast` | exact ray casting (`raycast.visible_exact`) of the full scene from `scenegeom.sensor_occluders`: no culling, no cube map, no q-bucket; UR5e sample with a 50 mm box, 10 feasible grasps of 40 sampled, both cameras, pads and object region, all 7,892 lifted contact locations | 10⁻³ of the points away from an edge (and under 2 % overall) | 0 away from an edge; 71 (0.90 %) within two texels of an edge |
| `grasp.bootstrap_coverage` | the known p of synthetic binomial samples (n = 100, 300, 1000; p = 0.3, 0.7, 0.9; 1000 seeded samples each); Wilson's interval on the same samples | coverage within 0.01 + 3 standard errors (0.031) of 95 %; mean endpoint gap to Wilson ≤ 0.02 | largest coverage gap 0.017 (bootstrap 0.939 to 0.967, Wilson 0.935 to 0.966 over the nine cases); mean endpoint gap at most 0.013 (n = 100), 0.0012 to 0.0015 at n = 1000 |
| `grasp.qbucket_vs_exact` | the same visibility evaluated at each joint vector with maps drawn there; UR5e sample, 12 pairs of joint vectors drawn in one 0.1° bucket | mean difference of a patch's visible fraction ≤ 0.02; largest ≤ 0.15; the cache was used | mean 0.0024, largest 0.055 (7 of 128 points); over four seeds the mean ranged 0.0005 to 0.008 and the largest reached 0.07 |

The edge test of the first check moves each point by two texels' angle within the plane of its own surface (not across
the line of sight, as `raycast.near_silhouette` does: that moves a point on a surface off it, and flagged 239 of 256 pad
points as near an edge). The q-bucket check's largest differences come from an occluder's edge crossing a patch: a
row of the 2F-85 pad's grid (about 11 points) flips at once. That is a real change of the view between two joint
vectors 0.1° apart, so the tolerance holds the mean.

## On the samples

With the defaults, 200 grasps (top-down, side and angled, seed 0): the UR5e sample's Wrist D405 sees 95 % of each pad
in every reachable grasp (133 of 200; 64 put the wrist into the table), CVS 100 % [100–100] (Wilson 97.2–100 %). The
Overhead D435 alone (`sensors` set to it) scores 34.6 % [26.3–42.9] (Wilson 27.0–43.0 %): it sees 42 % of the contact
points, 25 % are behind their own finger and 33 % hidden by the scene, mostly the gripper above the pads in top-down
grasps. The SO-100 sample: 120 of 200 reachable (47 joint limit, 33 out of reach), CVS 100 %; its third-person camera
sees 50 % of the contact points (38 % are behind their own finger). The worked example is in [../guide/grasp.md](../guide/grasp.md).

## Settings

`project.analysis.studies['grasp_visibility']`, every key optional, defaults in `grasp_visibility.DEFAULTS`:
`grasp_set`, `q`, `rule`, `ik_backend`, `ik_seeds`, `collision_margin_m`, `requirement`, `min_visible`,
`min_cameras`, `same_camera`, `precontact_cm`, `precontact_steps`, `region_radius_m`, `object_points`, `lift_mm`,
`max_view_angle_deg`, `snap_mm`, `eps_mm`, `cube_n`, `backend`, `q_bucket_deg`, `q_bucket_mm`, `bootstrap`, `ci`,
`rng_seed`, `good_pct`, `warn_pct`, `serious_pct`, `sensitivity`, `sens_grasps`, `sens_step_m`, `sens_step_deg`,
`gallery`, `gallery_px`, `sensors`, `modes`, `imagers`, `depth_imager`, `limit` (evaluate at most this many grasps,
evenly picked), `chunk`. The result's `settings.defaults` lists the keys left at their default. The page's Run passes
the 3D view's joint state for the robots the grasps do not move.

## Known limits

- Visibility is per cube-map texel: a point within about a texel of an edge may be judged either way (0.9 % of the
  points in the check; a texel is 3.9 mrad at a face centre at 512).
- The fingers stay at their home opening; they are not closed to the object's width. The 2F-85's linkage joints are
  independent entries of q (core/robot/model.py).
- The contact patch comes from the collision geometry, moved onto the visual mesh only within `snap_mm`; the tested
  location is `lift_mm` in front of it, a convention, not a contact model.
- The object's grasp region is a ball around the grasp centre on the object's surface, not the true contact footprint.
- IK is a local method: a reachable grasp can be reported infeasible (`no convergence`). A 5-joint arm (the SO-100) is
  solved position-first, so its orientation may differ from the grasp's.
- Obstacles are checked at contact, not along the approach; the approach is a straight line along `approach`.
- No sensor housings occlude (scenegeom); materials, lighting and motion blur are other studies.
- What to move is one camera and one axis at a time on a subset (interactions between moves are not explored); a
  change within its interval's reach of zero may be noise, and the page says so.
- The pinocchio and cuRobo backends are untested here (not installed).
