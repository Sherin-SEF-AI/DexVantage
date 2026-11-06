# Coverage and resolution in the task volume {#coverage}

Module: `dexvantage/core/studies/coverage.py` (study id `coverage`, batch flag `--coverage OUT.json`). Page:
`dexvantage/ui/pages/coverage.py` (Analysis ▸ Coverage and resolution, Placement workspace). Checks:
`dexvantage/validation/checks_coverage.py` (`occlusion.coverage_vs_raycast`, `optics.coverage_density_fd`). Tests:
`tests/test_coverage.py`. Spec §7.1. Builds on [voxels.md](voxels.md), [occlusion.md](occlusion.md),
[optics.md](optics.md) and [library.md](library.md).

## What it computes

Every task volume is voxelised as a sparse grid with adaptive refinement (`voxels.SparseGrid.from_project`: 5 mm by
default, refined to at most 2 mm inside refinement regions and within 20 mm of the task objects; both sizes are
`project.analysis` settings). For every voxel centre **P** and every camera *c*:

| Figure | Definition |
|---|---|
| in image | P in the imager's optical frame (z forward, x right, y down), `Rᵀ (P − t)` with R, t the imager's world pose (the sensor's mount, posed on its link at q, composed with the imager's pose in the sensor body from the library), projects inside the image: `CameraModel.project` ok (in front, within the lens model's valid angle, u ∈ [−0.5, W − 0.5], v ∈ [−0.5, H − 0.5]) |
| visible | in image, and the cube map of distances around the imager's optical centre has nothing nearer along the line of sight (`occlusion.CubeMap.visible`, ε = 1 mm, face size 512 by default) |
| pixels per mm | `CameraModel.pixels_per_mm(P)`: √\|det ∂(u, v)/∂(lateral)\| for a small patch at P facing the camera, so a 1 mm feature spans that many pixels; a surface tilted by α away from the line of sight has √cos α of it. For a pinhole lens it is √(fx fy) / (R² cos³θ)^½ / 1000 at range R and θ off the axis |
| viewing angle | the angle between the optical axis and the line of sight (`CameraModel.angle_to_axis`) |
| defocus | the thin-lens blur-disc diameter in pixels, `b = (f/N) f |s − z| / (z (s − f)) / p` (`optics.blur_circle_px`), z the point's depth along the axis, s the focus distance, N the f-number, f the focal length, p the pixel pitch |
| redundancy | the number of cameras for which the voxel is visible |

**Which imager is the camera.** A camera sensor's first imager (its entry's reference imager). A depth sensor's RGB
imager when it has one, because the colour image is what a policy is trained on; otherwise its reference (depth)
imager. Setting `depth_imager: "depth"` uses the reference imager of every depth sensor; `imagers: {sensor uid: name}`
picks one per sensor. The result names the imager used and why (`cameras[i].imager`, `imager_why`). The stereo pair
belongs to depth validity (§7.2). Tactile pads are contact sensors and are left out (the result says how many).
A sensor without a library device gets the library's labelled default imager (90° × 60°, 1280 × 720 px pinhole,
"no device chosen"); the page marks it "(default)".

**The joint state.** A study gets only the project, so q comes from the study's settings,
`project.analysis.studies['coverage']['q'] = {robot uid: [values in RobotModel.joint_names order]}`; a robot not named
there is at its home pose. The page's **Run** passes the 3D view's current q in those settings for that run only (on
the copy of the project the study runs on: `AnalysisController.run(sid, settings)`), so the coverage is for the pose
on screen and the project is not changed. Batch mode uses whatever the saved settings hold (home when nothing). The
result records the q used (`q`) and where it came from (`q_source`: `settings` or `home`); values outside the joint
limits are clipped, with a note; a vector of the wrong length is refused with a plain message.

**Defocus needs four figures**: the f-number, the focus distance, and the focal length in millimetres (from the pixel
pitch, f = √(fx fy) · p). The library states N for many imagers, but no maker in it states the focus distance of a
fixed-focus lens, and the RealSense entries state no pixel pitch, so on the samples the blur is "not computed" and
each camera's record says which figure is missing. You can enter measured figures per sensor in the settings,
`optics: {sensor uid: {"focus_m", "f_number", "pixel_um", "focal_mm"}}`; the result marks each figure's source
(`maker`, `yours`, or `derived`). The pitch is taken as binned with the stream (pitch × native width / stream width),
an assumption. A blur up to `coc_px` (2 px, `optics.DEFAULT_COC_PX`, a default) counts as sharp, and the depth of
field for that circle is reported (`dof_near_m`, `dof_far_m`).

## The summary

Shares are of the **task volume**: each voxel weighted by its volume, so refined voxels count for the space they
fill, not for their number. Voxels whose centre lies inside a box, cylinder or sphere task object are left out (no
camera can see inside a solid); `grid.inside_objects` counts them.

| Key | Meaning |
|---|---|
| `summary.seen_ge1`, `seen_ge2`, `blind`, `seen_1_only` | shares seen by ≥ 1, ≥ 2, 0 and exactly 1 camera |
| `summary.resolved_ge1`, `resolved_ge2` | shares seen at ≥ `min_px_per_mm` (1 px/mm, a default) by ≥ 1 and ≥ 2 cameras |
| `summary.best_ppmm_median`, `best_ppmm_p5` | the best camera's pixels per mm over the seen volume (volume-weighted median and 5th percentile) |
| `summary.pct_ge2`, `status`, `word`, `advice` | the headline: % seen by ≥ 2, its status and a sentence on what to change |
| `cameras[i].seen`, `outside_image`, `hidden`, `only_camera`, `resolved` | per camera (seen + hidden + outside = 1) |
| `cameras[i].ppmm_median`, `ppmm_p5`, `ppmm_p95`, `angle_median_deg`, `angle_max_deg`, `range_median_m` | over the voxels it sees |
| `cameras[i].defocus` | `computed`, `why`, the figures and their sources, `blur_median_px`, `sharp`, `dof_near_m`, `dof_far_m` |
| `volumes[k]` | per task volume: voxels, `seen_ge1`, `seen_ge2`, `blind`, `per_camera` |

**Status thresholds** (user settings, with documented defaults; not standards): the share seen by two or more cameras
reads `good` from `good_pct` = 90 %, `warning` from `warn_pct` = 60 %, `serious` from `serious_pct` = 30 %, and
`critical` below. The page edits them as one undo step each and labels each "default" until it is changed.

**Advice.** When the volume is well covered, the sentence names the coarsest view (lowest 5th percentile px/mm).
Otherwise it names the camera that sees least, and whether more of the volume lies outside its image (aim it, move it
back or widen the lens) or is hidden by the scene at this pose (move it to look past the arm or the objects).

## The per-voxel arrays

`voxels` holds compact per-voxel arrays, each numpy bytes (little-endian), zlib-compressed and base64-encoded
(`coverage.encode_array` / `decode_array`; `coverage.decode(doc)` returns them all as numpy arrays):

| Array | Encoding |
|---|---|
| `ijk` | voxel centre as integer multiples of `unit_m` (half the refined edge) from its volume's grid corner `origins[volume]` (uint16; uint32 when needed) |
| `volume_index`, `refined`, `inside` | the voxel's volume, whether it is refined (edge `refine_m`, else `voxel_m`), whether it lies inside an object |
| `redundancy` | cameras that see it (uint8) |
| `best_ppmm` | the best visible camera's px/mm in steps of 0.01 (uint16, up to 655.35) |
| `cameras[i].state` | 0 outside the image, 1 in the image but hidden, 2 visible |
| `cameras[i].ppmm` | px/mm in steps of 0.01 (0 outside the image) |
| `cameras[i].angle` | viewing angle in steps of 0.5° (255: outside the image) |
| `cameras[i].blur` | blur in steps of 0.1 px (254: 25.4 px or more; 255: not computed or outside the image); `null` when not computed |

The UR5e sample's 480 000 voxels with its two cameras take 1.7 MB of JSON.

## Computation

The scene's occluders are built once for q (`scenegeom.scene_occluders`), each camera's by the mount-link rule
(`scenegeom.sensor_occluders` with the imager's centre as origin), and each camera's cube map through the shared
`occlusion.CACHE` (the GPU when there is one, `backend` setting otherwise). The voxels are then processed in chunks
of 100 000 (`chunk` setting), with progress after every chunk, a partial result (each finished camera's seen share)
after every camera, and a cancel check before every chunk; a cancelled run returns what it has with `partial: true`.
The UR5e sample (480 000 voxels, two cameras, 193 000 occluder triangles) takes about 0.8 s with the GPU backend;
the SO-100 sample (57 600 voxels) 0.1 s.

## Settings

All in `project.analysis.studies['coverage']`, every key optional: `q`; `voxel_mm`, `refine_mm` (else
`project.analysis`'); `good_pct`, `warn_pct`, `serious_pct`; `min_px_per_mm`; `coc_px`; `cube_n`; `backend`
(`auto`, `gl`, `numpy`); `depth_imager` (`rgb`, `depth`); `imagers`; `modes` (`{uid: [w, h]}`, a stream mode, else
the stream chosen for the sensor on the Sensors page, `Sensor.stream`, as the Camera view uses it, else the imager's
largest listed stream); `optics`; `sensors` (only these uids; default every enabled camera and depth
sensor); `chunk`. The result's `settings.defaults` lists the keys left at their default.

## Validation

| Check | Reference | Tolerance | Measured (full run, seed default) |
|---|---|---|---|
| `occlusion.coverage_vs_raycast` | pinhole projection written out from the stated FOV (fx = (W/2)/tan(hfov/2), REP-103 rotation typed in again), an inside-the-object test written out again, and exact ray casting (`raycast.visible_exact`) in place of the cube maps | 0.005 of the volume on every share; per voxel < 2 % disagreement, ≤ 1e-3 of the occlusion answers farther than two texels from a silhouette, ≤ 1e-4 of the in-image answers | largest share difference 8.8 × 10⁻⁴ (seen by ≥ 2); 0.41 % of voxel-camera answers differ, none farther than two texels from a silhouette, no in-image differences (335 100 voxels at 5 mm refined to 2.5 mm, 3 cameras, GPU backend) |
| `optics.coverage_density_fd` | √\|det J\| of `cv2.projectPoints` differentiated across a patch facing the camera (central differences, 0.1 mm) | 1 % relative | 0.65 % (6 000 voxels, 3 cameras); the stored 0.01 px/mm step is the whole of it |

The scripted scene: a 0.30 × 0.30 × 0.20 m task volume on a table, a box and a cylinder standing in it, two static
cameras with the default imager and one with the Intel RealSense D435 entry (its RGB imager). The two paths share
only the scene's triangles.

## Known limits

Visibility is judged at the voxel centre, and per cube-map texel (0.22° at a face centre at n = 512), so a voxel
within about a texel of a silhouette may be judged either way. A voxel inside a robot link or a mesh object counts as
unseen (only box, cylinder and sphere objects are tested for their insides). Pixels per millimetre is for a patch
facing the camera, because the task's surfaces are not known. No sensor housings occlude. Overlapping task volumes
count their shared space once for each. The defocus is a thin-lens model, computed only from stated or entered
figures, with the pitch assumed binned with the stream. Coverage is for one joint state per run; a trajectory or a
grasp set needs a run per pose (Phase 3). Lighting, exposure, motion blur and the depth sensor's own validity
(min-Z, the stereo pair, the projector) are other studies (§7.2, §7.8, §7.9). The Camera view overlay of §7.1 is
not drawn yet. An exported result made with the page's live q cannot be re-run by `--reproduce` at that q, because
q is not stored in the project (the result records it).
