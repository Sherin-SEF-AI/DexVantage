# VLA token resolution and Plücker ray maps

`dexvantage/core/studies/vla_tokens.py` (study `vla_tokens`, batch `--vla-tokens OUT.json`),
`dexvantage/core/plucker.py` (ray maps, the safetensors writer), the page `dexvantage/ui/pages/vla.py`, the checks
`dexvantage/validation/checks_vla.py` (model `vla`). Spec §7.12.

A vision-language-action policy resizes or crops each camera stream to a fixed input (224 × 224 for π0 and OpenVLA,
256 × 256 for Octo, 512 × 512 for SmolVLA) and slices it into ViT patches (14 or 16 px). An object that covers one or
two patches gives the policy almost nothing to condition on. The study counts, per policy and per camera the policy
takes, the tokens each task object and each fingertip covers at the pre-grasp and contact poses of a grasp set, and
warns when a critical object spans fewer than 2 × 2 tokens (a default) or falls outside the crop. The input
resolution, crop rule, patch and token count come from the policy (core/policies.py `resolve`: the library entry,
cited from the model card, paper or official code, or the user's measured figure; docs/models/policies.md).

## The image → input mapping

Each camera is modelled at its stream (`Sensor.stream`, else the imager's largest listed stream) through its RGB
imager (else its first; the Camera view's rule) and its lens model (`optics.CameraModel`). The policy's crop rule
gives a `CropGeometry` (core/policies.py, validated by `policies.crop_geometry` against each rule's own code): the
crop box in the sensor, `src_rect = (x0, y0, x1, y1)` in pixel-edge coordinates (OpenCV pixel *u* spans *x* = *u* to
*u* + 1), the rectangle it fills in the input, `dst_rect`, and the scales *s*ₓ, *s*ᵧ (input pixels per sensor pixel):

    x_in = dst_x0 + (u + ½ − x0) · sx,      y_in = dst_y0 + (v + ½ − y0) · sy

Every camera record of the result carries this mapping (`mapping`: `crop_box_sensor_px` in OpenCV pixel
coordinates, `scale_input_per_sensor_px`, `content_rect_input_px`, the padding share and the equation written out).
On the UR5e sample with π0 (`resize_pad`), the Overhead D435's 1920 × 1080 colour stream is used whole
(*u* −0.5 … 1919.5) at 0.1167 input px per sensor px, filling rows 49 to 175 of the 224 × 224 input: 43.75 % of the
input, and so of its tokens, is black padding.

## The token grid

The input is sliced into square cells of `patch_px`: cols = ⌊*w* / *p*⌋, rows = ⌊*h* / *p*⌋ (a strided patch
embedding drops a remainder). A policy whose entry states no patch but states `tokens_per_image` *T* gets the cell
*c* = √(*w h* / *T*) when that is a whole number of pixels dividing both sides; it is labelled *derived from tokens per
image* (it assumes the tokens tile the input on a square grid). The library's policies:

| Policy | Input | Grid | From |
|---|---|---|---|
| π0, π0-FAST, OpenVLA | 224 × 224 | 14 px, 16 × 16 | the patch |
| Octo | 256 × 256 | 16 px, 16 × 16 | the patch |
| SmolVLA | 512 × 512 | 16 px, 32 × 32 (64 tokens of 64 px after the pixel shuffle, reported beside) | the patch |
| ACT | 640 × 480 | 32 px, 20 × 15 | derived (300 tokens) |
| GR00T N1 | 224 × 224 | 28 px, 8 × 8 | derived (64 tokens) |
| Diffusion Policy | 288 × 216 | none: pixels only, said so | neither stated |

## The silhouette and the counts

A task object's world triangles (`scenegeom.object_triangles`) are projected through the lens and the mapping and
rasterised at input resolution: an input pixel is covered when its centre lies inside a projected triangle (three
edge functions with the triangle's winding, edges inclusive). A lens that bends straight lines (Brown-Conrady, the
fisheyes) is handled by subdividing each triangle into four at its edge midpoints until every projected edge midpoint
lies within `raster_tol_px` (default 0.01 input px) of the midpoint of its projected ends, at most `max_subdiv`
levels (default 8). A triangle with a corner behind a pinhole camera or beyond the valid angle is split the same way;
what is still invalid at the last level is dropped and named in the notes. The raster covers the whole camera image
mapped into input coordinates, so the part of an object the camera sees but the crop cuts away is measured.

Per object, camera and pose: **tokens touched** (cells with at least one covered pixel), **tokens at least half
covered**, covered input pixels, the **extent** (the silhouette's bounding box in input pixels divided by the cell:
width × height in tokens), the share **outside the crop** (of what the camera sees of it), in view or not.

**Warnings** (critical objects only: `critical_objects`, default every task object):

- *spans fewer than 2 × 2 tokens* (serious): its extent is below `min_tokens_side` (default 2, a default, not a
  standard) along either side at its worst pose. The extent, not the touched count, decides: a 3 px object straddling
  a cell corner touches 4 tokens but spans 0.2 × 0.2.
- *partly cropped* (warning): more than `crop_warn_fraction` (default 1 %) of what the camera sees of it is cut away.
- *falls outside the crop* (serious): the camera sees it, the input does not.
- *not in the input* (critical): no camera of the policy has it in its input at any pose counted.

**Fingertips**: the contact pads' sample points (`grasps.contact_points`, 128 area-weighted points per pad by
default): tokens touched are the cells holding a point inside the input; the extent is the points' bounding box.

## The poses

With a grasp set (`grasp_set`, else the project's first), up to `max_grasps` (default 12) grasps evenly through the
set, each at **contact** (the grasp's gripper pose) and **pre-grasp** (backed off `pre_grasp_m` along the approach,
default 100 mm). The pads are carried rigidly from home with the gripper frame. A camera mounted on that robot moves
with it: the arm is posed by IK of the grasp centre (`ik.solve_batch` with `grasps.ik_targets`' tool offset, 4 seeds,
collision checks off); a pose IK does not reach is left out for that camera and counted in the notes. Without a grasp
set the robots stand at home, or at the joint state passed in the run's settings (the page passes the pose on
screen), and the notes say so.

## Headline and suggestion

The headline is the fewest tokens a critical object touches in a camera that has it in its input (over every policy,
camera and pose), with its extent; the status is the worst warning (*resolved*, *partly cropped*, *too few tokens*,
*not in the input*). With no critical object counted at all (no task object, as on the samples) the headline is the
fewest tokens on a fingertip and the status is a warning, *no task object*, never *resolved*. The suggestion tries each stream mode the camera's device lists (`entry.streams(imager)`) at the
warned objects' worst poses and proposes the one under which the worst warned object's narrower extent grows most:
a 4:3 stream padded into a square input wastes less of it than a 16:9 one. Apply sets `Sensor.stream` (one undo step).

## Plücker ray maps

Ported from Vantage `core/plucker.py` (the per-pixel field of origin *o*, unit direction *d* through the exact lens
and moment *m* = *o* × *d*, channel order `dm` or `md`, one file per camera plus provenance); here the grid is the
policy's input, not a stride grid over the sensor, and the frame is the robot base, not ISO 8855.

Input pixel (row *j*, column *i*) has its centre at input edge coordinates (*i* + ½, *j* + ½);
`CropGeometry.to_source` maps it to sensor edge coordinates (*x*, *y*), the sensor pixel is (*u*, *v*) = (*x* − ½,
*y* − ½), `CameraModel.unproject` gives the optical ray, and the imager's pose at *q* turns it into the frame:
`base` (default: the base of `robot_uid`, default the project's first robot, `Robot.base`), `world`, or `camera`
(*o* = 0, so *m* = 0). Valid: inside the crop's content (not padding), inside the sensor image and within the lens's
valid angle. Invalid pixels are zero in all six channels with `valid = false`. The mapping is separable, so `u` (*w*,)
and `v` (*h*,) are stored as vectors.

`export_plucker(project, out_dir, policy_uid, q=None, frame='base', dtype='float32', order='dm')` writes, per camera
the policy takes, `plucker_<policy>_<camera>.npy` ((*h*, *w*, 6)), `.safetensors` (tensors `plucker`, `valid`, `u`,
`v`, `origin`, `rotation`; metadata schema, frame, order, convention) and `.json` (the convention, the lens, the crop
with its sensor-pixel box, the share valid and padded, *q*, and the provenance block).

**safetensors** is written here (the package is optional): 8 bytes of header length *N* (unsigned little-endian
64-bit), *N* bytes of UTF-8 JSON starting with `{` and padded with spaces to a multiple of 8, mapping each name to
`{"dtype", "shape", "data_offsets": [begin, end]}` relative to the byte buffer, plus `__metadata__` (strings only);
then the buffer, little-endian row-major, no holes [safetensors README "Format" and `safetensors/src/tensor.rs`,
https://github.com/huggingface/safetensors, read 2026-11-06]. `read_safetensors` refuses holes, overlaps, trailing
bytes and offsets that do not match the shape.

## Validation (measured 2026-11-06, default seed)

| Check | Reference | Tolerance | Measured |
|---|---|---|---|
| `vla.patch_counts_raycast` | every input pixel's centre mapped back through the crop, unprojected, ray-cast against the object (core/raycast.py), tokens counted with a plain loop; pinhole, Brown-Conrady (k1 = −0.28), Kannala-Brandt fisheye; `resize_pad`, `center_crop_resize`, `resize`, `center_crop`, `resize_pad_top_left`; 45 boxes, cylinders and spheres | 0 tokens | 0 tokens; 0 of 84 586 covered pixels differ |
| `vla.plucker_moment_cv2` | the crop rectangle written out again, `cv2.undistortPoints` (`cv2.fisheye.undistortPoints`) from `to_opencv`, run to 10⁻¹⁴, the base-frame pose composed in the check, *m* = *o* × *d*; also *m* · *d* and the moment of *o* + 2.7 *d* | 10⁻⁶ | 1.3 × 10⁻¹⁵ |
| `vla.safetensors_roundtrip` | a parser written from the public format (and the `safetensors` package when installed: not in `.venv`) | 0, bit-exact | 0 |

The reference of the first check is sensitive: with subdivision off (0 levels, seed 1) 162 pixels differ on the distorted
lenses and one token count is off by one; at 0.05 px and 6 levels a few pixel centres within 0.05 px of an edge
differ. OpenCV's undistortion returns *x*/*z*, *y*/*z*, which has no ray at or beyond 90° and is ill-conditioned close
to it (0.018 apart at 89° on the scripted fisheye), so the ray check compares rays within 85° of the axis and reports
how many it left out.

## Known limits

Silhouettes are not occlusion-tested (an object hidden behind the gripper still counts its tokens: the grasp
self-occlusion study treats visibility). Objects are their triangle models (cylinders and spheres tessellated as
scenegeom does). Tokens are counted on the inference-time crop (training augmentation is not modelled) and on a
square cell grid (other merging schemes are not modelled); interpolation filters are not modelled, so a token touched
by a sliver of an object may carry little of it. Octo's 128 × 128 wrist input is listed by the Policies page but not
applied per camera role. The fingertip pads are carried rigidly from home (the finger opening at home), and IK picks
one arm configuration per grasp. A ray is that of its pixel's centre (no footprint averaging); ray maps are not
masked by the scene; a wrist camera's map is for the pose it was exported at.
