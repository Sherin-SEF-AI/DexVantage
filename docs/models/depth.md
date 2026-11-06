# Depth validity {#depth}

Module: `dexvantage/core/studies/depth_validity.py` (study `depth_validity`, batch flag `--depth-validity OUT.json`).
Page: `dexvantage/ui/pages/depth.py` (Analysis ▸ Depth validity). Checks: `dexvantage/validation/checks_depth.py`
(model `depth`). Tests: `tests/test_depth_validity.py`. Spec §7.2 and the depth checks of §10.

It builds on the library (`core/library`: the device's imagers, depth block and projector, `docs/models/library.md`),
the lens models (`core/optics`, `docs/models/optics.md`), the sparse voxels (`core/voxels.py`, `docs/models/voxels.md`)
and the articulated occlusion (`core/scenegeom.py`, `core/occlusion.py`, `docs/models/occlusion.md`).

## The five conditions

A voxel (its centre) is **depth-valid** for a depth sensor when all five hold. They are tested in this order and the
first that fails is the voxel's reason, so the reasons and the valid share add up to 100 % of the task volumes. Shares
are volume weighted (refined voxels are smaller). Voxels whose centre lies inside a solid task object (box, cylinder or
sphere) are left out of the task volumes, as in the coverage study: no sensor measures a solid's inside, and counted
they would all read "occluded" (`voxels.inside_objects` in the result counts them).

| # | Reason id | Condition | Where its figures come from |
|---|---|---|---|
| 1 | `fov` | Inside the field of view of every imager the technology needs: **both** imagers of a stereo pair (active or passive); the pattern camera and the projector cone for structured light; the ToF imager for time of flight | `library.sensor_imagers` at the chosen depth resolution (a pinhole fitted to the stated depth FOV unless the unit is calibrated) |
| 2 | `min_z`, `max_range` | Depth z (along the reference imager's optical axis, Intel's depth convention) at least min-Z for the chosen resolution and at most the maximum range | the entry's `min_z` table (else its stated `min_range_m`, labelled "not per resolution"); `max_range_by_mode` for the mode, else `max_range_m` |
| 3 | `occluded` | Seen by **every** needed imager: no robot link, task object or surface between the voxel and the imager's optical centre | one cube map per imager centre (`occlusion.CACHE`), from `scenegeom.sensor_occluders` at q |
| 4 | `projector` | Inside the IR projector's cone, for active sensors that need its texture | the entry's `projector` (horizontal and vertical FOV and pose) |
| 5 | `material` | Not on a task object tagged `transparent`, `ir_absorbing` or specular (`glossy`, `metallic`) | `TaskObject.materials` |

Notes on each:

1. **Stereo overlap.** Requiring both imagers is the overlap of their frusta. Intel's D400 datasheet (§4.4) gives its
   horizontal extent at distance Z as `Depth FOV = HFOV/2 + atan(tan(HFOV/2) − B/Z)` with the left imager as the
   reference; the check `depth.fov_overlap_formula` compares the two. The left edge of the overlap is Intel's
   "invalid depth band" (§4.5): near the camera, the left part of the depth image has no match in the right image.
   The spec says "the RGB and projector FOVs" for structured light; the camera that sees the pattern makes the depth,
   so the reference imager of the entry is used (no structured-light device is in the library yet).
2. **Range.** A figure the maker does not state is not tested, and the result says so (for example the D405 at
   640 × 480, which Table 4-12 gives as "-"). OAK's `min_z_extended` (extended-disparity mode) is not used.
3. **Occlusion from either imager** covers finger shadows and the parallax occlusion of a stereo pair: a voxel seen
   by one imager and hidden from the other has no depth. The result splits the occluded volume into "hidden from one
   imager only" and "hidden from every imager". The mount link stays an occluder (a wrist camera sees its own
   gripper); triangles of it within 25 mm of an imager centre are its seat (`scenegeom.OWN_RADIUS_M`, a default).
4. **Projector need** is the per-sensor setting `projector`: `'auto'` (default) turns it on for active IR stereo when
   the device's projector and its FOV are stated, and always for structured light (where it is part of condition 1);
   off for passive stereo and ToF (a ToF illuminator's field is not in the library). The D4xx projector's position in
   the device is not stated by Intel; the library places it at a labelled assumption, and the page says so.
5. **Materials** are warnings with a reason: the geometry is fine, the surface may still give no depth.
   IR-absorbing warns only for the active technologies (active IR stereo, structured light, ToF), whose depth comes
   from their own IR light. A voxel is "on" an object inside the object's shape grown by half the voxel's diagonal
   (a mesh: its bounding box so grown). The result gives each object's share of the volume that is otherwise
   depth-valid, and the depth-valid share with and without the warnings (`valid_pct`, `valid_geometric_pct`).

## Expected depth error

For stereo, z = f B / d; a disparity error σ_d propagates to first order as

    σ_z = z² · σ_d / (f · B)

[Hartley and Zisserman 2004, §12; Gallup et al. 2008]. f is the reference imager's horizontal focal length in pixels
at the chosen depth resolution (disparity is horizontal), B the stated baseline and σ_d the matcher's subpixel
disparity error: the entry's `subpixel_px` when the maker states it, else the setting `sigma_d_px`, **0.08 px by
default (a default, not a device figure; no D4xx entry states one)**. The result reports the median and the 5th and
95th percentiles over the depth-valid voxels (volume weighted), written `median [p5–p95]`, and σ_z / z.

**Against the maker.** Where the entry states an RMS figure as a percentage at a distance, the page shows both: the
maker's bound (for example the D405's "RMS error ≤ 1 % at ≤ 0.5 m", Intel Table 4-15, so ≤ 5 mm at 0.5 m) and the
model at that distance, with the σ_d that would make the model match the maker. They are not the same quantity:
Intel's figure is a plane-fit RMS over 80 % of the field of view at HD resolution under its test conditions (a
textured target in room light for the D405, a white wall with the projector for the others), an upper bound; the
model is a per-point σ for an ideal matcher. On the UR5e sample at the default 0.08 px the model gives 1.56 mm for the
D405 at 0.5 m (the bound is 5 mm; σ_d = 0.26 px would match it) and 9.5 mm for the D435 at 2 m (bound 40 mm; σ_d =
0.34 px). ToF has no stereo error model: the maker's figure is shown alone.

Not every maker's figure is an RMS. The entry's `rms_error.metric` says what it is: `rms` (Intel, and Orbbec's
Gemini 2 and 335/336 "RMS distance ... to a fitted plane"), `median_abs` (Luxonis: "median ... absolute depth
error"), `std_dev` (Orbbec Femto Bolt, in millimetres) or `unspecified` (Stereolabs: "Depth Accuracy" with no metric
defined; also any entry without the field). The page shows the figure with its metric, and the σ_d that would match
it only for `rms`; for the others that column reads "not an RMS".

## Settings

`project.analysis.studies['depth_validity']`, all optional:

| Key | Default | Meaning |
|---|---|---|
| `sensors[uid].resolution` | the native depth stream | `[w, h]`, one of the device's depth modes (its depth streams and min-Z rows). Without it, the stream chosen for the sensor on the Sensors page (`Sensor.stream`) when that is a depth mode (the Inspector's min-Z and the Camera view use it), else the default: the largest listed depth stream, else the reference imager's largest stream, else the largest min-Z row, else its native resolution; the result says which |
| `sensors[uid].projector` | `'auto'` | `true` / `false` overrides the projector need |
| `sensors[uid].sigma_d_px`, `sigma_d_px` | 0.08 px (a default) | the disparity error when the maker states none |
| `cube_n` | 512 | cube-map face size |
| `backend` | `'auto'` | the occlusion backend (`'gl'`, `'numpy'`) |
| `q`, `pose` | home | joint state per robot uid; the Depth page passes the 3D view's joint state with each run (view state: not stored). `pose: 'home'` ignores q |
| `good_pct`, `fair_pct` | 80 %, 50 % (defaults) | the status words: good from 80 %, partial from 50 %, else poor |

## What the result holds

Per sensor: the resolution and whether it is the default, min-Z and maximum range with where they come from, f,
B, σ_d (and whether it is the default), the projector need and why, each needed imager's label (what its camera model
is and which of its figures are assumptions), the depth-valid share, the breakdown by first failing reason, the
share failing each condition among the voxels in view, the occlusion split, per-volume shares, σ_z, the maker's
figure beside the model, the material warnings, the depth of the robots' fingertip frames against min-Z, a sample of
up to 3000 invalid voxels per reason for the 3D view, and its world pose. For the rig: the share depth-valid for at
least one sensor (the page's headline). A suggestion (Apply on the page): the depth mode of the same aspect ratio
with a smaller stated min-Z that makes the most task volume depth-valid (estimated from the voxels that fail min-Z
alone), or, when none gains volume, the largest one whose min-Z clears fingertips that sit inside min-Z; ties go to
the larger resolution because σ_z grows as the focal length in pixels shrinks.

## The min-Z blind zone in the 3D view

`blind_zone_local(setup)` is the part of the reference imager's field of view nearer than min-Z: a pyramid from the
optical centre to the plane z = min-Z, its edges sampled on the image border. The Depth page registers it as an
overlay source (`ctx.overlay_sources`), so it is drawn on every page, translucent in the sensor's colour, from the
sensor's world pose at the window's joint state: a wrist camera's zone moves with the wrist. View ▸ Depth min-Z
zones turns it off. No zone is drawn when min-Z is not stated for the chosen resolution.

## Validation

`dexvantage --validate out/validation --models depth` (or Help ▸ Validate the models…). Measured with the default
seed on the discrete GPU's EGL backend (an NVIDIA GeForce RTX 5080), full run:

| Check | Reference | Tolerance | Measured |
|---|---|---|---|
| `depth.sigma_z_monte_carlo` | Monte Carlo: a rectified pair written out in the check, Gaussian noise on the right-image match, linear (DLT) triangulation, 40 000 trials per case; D405, D435, D455 geometry, σ_d 0.08 and 0.3 px, four depths from min-Z to the maximum range (24 cases) | 4.5 standard errors of a standard deviation (1/√(2(n − 1))) plus 3 (σ_d/d)², the second-order term the model leaves out: 0.0183 | 0.0070 (largest relative difference) |
| `depth.d400_tables` | Intel's D400 datasheet Tables 3-49 and 4-12 and D457 datasheet Tables 3-19 and 4-4, read 2026-11-06, typed again in the check: min-Z at six resolutions (through the study's own lookup), baseline, depth FOV HD and VGA, projector FOV; D405, D415, D435, D435i, D455, D457 | 0 figures | 0 of 92 differ |
| `depth.fov_overlap_formula` | Intel §4.4, Depth FOV = HFOV/2 + atan(tan(HFOV/2) − B/Z), with the model's own left-imager HFOV; D405, D435, D455 at 1280 × 720 and 848 × 480, Z from min-Z to 2 m | 0.01° (the sweep step is 0.0005°) | 0.00094° |
| `depth.finger_shadow_raycast` | `raycast.visible_exact` (Möller–Trumbore through a uniform grid) from each imager centre; UR5e sample, wrist D405 and overhead D435, three random arm poses with the gripper partly closed, points in the D405's frustum (30 to 600 mm) and in the task volume | 1e-3 of the point-imager pairs away from a silhouette (and under 2 % in all); points seen by one imager only must occur | 0 of 242 850 away from a silhouette; 0.16 % within two texels of one; 8675 points seen by one imager only |

The table check is a second reading of the same documents, not a second source: it catches transcription errors and
a broken resolution lookup, not an error in the datasheet.

## Known limits

The voxel centre stands for the voxel. Occlusion is per cube-map texel from each imager's optical centre (points
within about a texel of a silhouette may be judged either way, `docs/models/occlusion.md`); no sensor housing is an
occluder. The projector's own shadow is not modelled, only its cone, and the D4xx projector's position is a labelled
assumption. The imagers at a depth stream take the stated depth FOV, which Intel states as the overlap at one
distance (0.2 m for the D405 and 2 m for the D415, D43x and D455 in the D400 datasheet, 4 m for the D457 in its own),
so the modelled overlap is narrower than the one the stated figure implies: by 2.9–3.6° for the D405 and 0.7–1.9° for
the others from min-Z outwards; Intel gives about ±3° unit to unit. min-Z is measured from the optical
centre, not the front glass (Intel's depth start point is a few millimetres away). Materials are tags, not optics: no
reflectance, multipath, sunlight or IR wash-out (§7.8). σ_z is first order in σ_d / d and leaves out calibration
error, matcher bias, quantisation and the matcher's dependence on texture. Lens distortion is what the library
states (a pinhole unless the unit is calibrated). The Camera view overlay of the depth-valid mask is not drawn yet.

## References

R. Hartley and A. Zisserman, Multiple View Geometry in Computer Vision, 2nd ed., 2004. D. Gallup, J.-M. Frahm,
P. Mordohai and M. Pollefeys, "Variable baseline/resolution stereo", CVPR 2008. RealSense, D400 Series Datasheet
337029-017 rev. 023 (March 2026), https://www.realsenseai.com/wp-content/uploads/2026/03/RealSense-D400-Series-Datasheet-Mar-2026.pdf;
D457 Datasheet 337029-011 rev. 006 (August 2025), https://realsenseai.com/wp-content/uploads/2025/08/RealSense-D457-Datasheet-Aug_2025.pdf;
both read 2026-11-06.
