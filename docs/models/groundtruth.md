# MuJoCo ground truth: rendered against predicted {#groundtruth}

Modules: `dexvantage/core/sim/scene.py` (the scene), `dexvantage/core/sim/render.py` (renders through each lens),
`dexvantage/core/studies/groundtruth.py` (study `groundtruth`, batch flag `--groundtruth OUT.json`).
Page: `dexvantage/ui/pages/groundtruth.py` (Simulate ▸ Rendered against predicted visibility…; Simulate ▸ Build the
MuJoCo scene… writes the scene folder). Checks: `dexvantage/validation/checks_groundtruth.py` (model `groundtruth`).
Tests: `tests/test_groundtruth.py`. Self-test line `groundtruth`. Throughput: `scripts/measure_groundtruth.py`.
Spec §9.1 ("this is DexVantage's CARLA").

The renders reuse `dexvantage/validation/mjrender.py` (the EGL worker process the Phase 2 checks use), extended with a
worker that stays up (`mjrender.Session`), a per-job camera pose and intrinsics, and the EGL device choice.

## The scene

`build_scene(project, variant=None, objects=True, lights=True)` returns a `SceneBuild`: the `mujoco.MjSpec`, the
compiled model, `camera_names` and `imagers` (each sensor's primary imager: the RGB imager of a depth sensor, the one
the grasp study uses), `cameras` (every imager of every sensor in use), `body_map` (body id or name → owner) and the
geom → owner table the segmentation is read through.

| Part | How it is built | How it matches the predictions |
|---|---|---|
| Robots | the stored MJCF/URDF read as `core/robot/model.py` reads it (fusestatic off), attached with `MjSpec.attach` under a frame at the robot's base, names prefixed `robot<k>_`; keyframes and the file's world-body geoms dropped | only the geoms RobotModel draws are drawn (visual-only geoms when the file has them, alpha > 0, a type it tessellates); the others, and every site and tendon, are moved to geom group 5, which MuJoCo's default visual options do not draw. Body poses equal `RobotModel.fk` (test: 0 m) |
| Task objects | a static body per object with its exact primitive (box, cylinder, sphere) or a mesh of the project's triangles | the predictions tessellate cylinders and spheres (24 segments, ≤ 1.1 % of the radius inside) |
| Surfaces | a box whose top is at the surface's z (its thickness, else 1 mm: a labelled default) | the predictions use a zero-thickness rectangle |
| Materials | tags → MuJoCo materials (`scene.MATERIALS`, DexVantage defaults: glossy specular 0.8, metallic grey, black, IR-absorbing dark matte, transparent alpha 0.3) | RGB only; segmentation and depth do not depend on them |
| Lights | the project's lights as the lighting study defines them (`core/lights.py`: the face looks along the light's local +z; a light that is off is left out; point → point light, spot → spot of half its stated beam angle, panel, ring and window → spot of cutoff 90° and exponent 1, Lambert's cos θ; MuJoCo's default colour 0.7 times the dim; MuJoCo's classic renderer has no photometric units), else an overhead and a front directional light (labelled defaults), and MuJoCo's head light | RGB only |
| Cameras | one per imager, in the mount link's body (or the world), at mount · imager · diag(1, −1, −1) | the imager's `CameraModel` at its stream size (below) |

**Camera intrinsics, verified in the installed MuJoCo's reference.** Read in `doc/XMLreference.rst` of
github.com/google-deepmind/mujoco at tag **3.17.0** (the installed version), section body/camera, on 2026-11-07:
`fovy` is the vertical field of view in degrees and "the horizontal field of view is computed automatically given the
window size"; `sensorsize` is the "Size of the camera sensor in length units. When specified, all intrinsic attributes
become active and fovy is ignored"; `focal` / `focalpixel` are the focal length "in physical length units or in
pixels, respectively. If both are specified, the pixel value is used"; `principal` / `principalpixel` are the "Offset
of the principal point (optical axis intersection with the image plane) from the image center" (the direction is not
stated); `resolution` "is not used for rendering since those dimensions are determined by the size of the rendering
context". `visual/map/znear` is the near plane as a fraction of the model extent. The scene therefore sets, per
camera at its W × H stream:

    resolution = (W, H)
    sensorsize = (W, H) · 3 µm            (a nominal pixel pitch: MuJoCo needs a length, only ratios matter)
    focalpixel = (fx, fy)
    principalpixel = ((W − 1)/2 − cx, (H − 1)/2 − cy)
    statistic/extent and visual/map/znear so that the near plane is 5 mm (as core/occlusion.py and camrender)

MuJoCo 3.17.0 compiles this to `cam_intrinsic = focalpixel · sensorsize / resolution` (measured in
`tests/test_groundtruth.py`; setting `intrinsic` directly compiles to zero, so the scene does not). The principal
offset's sign, (W − 1)/2 − px, is the one `mjrender.camera_model` measured on renders in Phase 2; the checks below
confirm it again through fiducials and per-pixel agreement. MuJoCo cameras are pinhole: a lens with distortion has its
pinhole part (fx, fy, cx, cy) in the scene and the note says so.

`save_scene(project, folder)` writes `scene.xml`, `assets/` (the robots' mesh and texture files) and
`scene_notes.txt` (the cameras with their lens models, the defaults, the notes); the result loads with
`mujoco.MjModel.from_xml_path` and holds no path of this machine.

## Renders through the lens model

`Renderer(project, variant=None, device='auto')` is a context manager around one `mjrender.Session`; `.render(views)`
takes `[{'sensor_uid', 'q', 'outputs', 'scale', 'extrinsic_delta', 'imager', 'camera_model'}]` and returns per view
`rgb` (H × W × 3 uint8), `depth` (metres along the optical z, 0 = no hit), `range` (metres along the pixel's ray),
`seg` (owner index, −1 = nothing), `seg_owner` and the `CameraModel` the image follows. `extrinsic_delta` perturbs the
mount (T_body_sensor · Δ) for a synthetic mount slip. `.info` names the renderer, GL, MuJoCo and device.

The lens decides the path (`render.lens_plan`):

- **pinhole**: the scene camera at the asked size, no resampling.
- **wide** (every pixel ray within 60° of the axis: Brown-Conrady, a mild fisheye): one pinhole base image in the same
  optical frame that covers every pixel's ray. For each lens pixel (u, v), r = `CameraModel.unproject(u, v)` and the
  base pixel is (fx_b r_x / r_z + cx_b, fy_b r_y / r_z + cy_b); `cv2.remap` samples it (nearest for segmentation and
  depth, bilinear for RGB). The lens's distortion is applied exactly at each pixel centre; nothing is fitted.
- **cube** (beyond 60°, including fisheyes past 180°): the 90° cube faces around the optical frame that the rays need;
  each pixel takes the face its ray leaves through.

The base images are at least `OVERSAMPLE` = 2 (a default) times as dense, in pixels per radian, as the lens image at
its densest, up to 4096 px a side (2048 for a cube face); a cap that bites is noted. The base's intrinsics are set per
render through `mjrender`'s `camera_intrinsic` (cam_sensorsize and cam_intrinsic in pixel units, which MuJoCo reads at
every render: measured in the tests). Depth along a base camera's axis a becomes the lens's optical z and range with the
pixel's own ray: range = d / (r · a), z = range · r_z.

## The study

**(a) Contact visibility.** A seeded sample of the grasp set (50 grasps, a default), the home pose, and
`trajectory_frames` (8, a default) evenly spaced frames, both ends included, of every joint trajectory of the same robot
(`trajectory.on_model`: joints a trajectory does not name stay at the grasp study's joint state). A project with
trajectories and no grasp set compares home and the trajectory frames. Each grasp is
solved and evaluated by the grasp study itself (`grasp_visibility.solve_grasps`, `snapped_pads`, `pad_world`,
`Evaluator.states` at the grasp's own q, no q-bucket cache), with the grasp study's settings. A contact point is
**predicted visible** when it projects into the image and the contact location (the point lifted 2 mm along the pad's
normal) is not hidden in the camera's cube map. It is **rendered visible** when it projects into the image, the
segmentation at its nearest pixel is its own finger, and the rendered depth there is within `depth_tol_mm` (5 mm, a
default) of the point's depth. "Its own finger" is the pad link and every link welded to it (`groundtruth.finger_links`):
the Robotiq 2F-85's pad link is a collision box with no visual geometry; its silicone pad and follower are welded to
it. Over the (camera, point) pairs in view:

    confusion: TP both visible, FP predicted visible only, FN rendered visible only, TN both hidden
    agreement = (TP + TN) / n,   IoU of the visible sets = TP / (TP + FP + FN)

per pose, per kind of pose (`by_kind`: home, grasp, trajectory), per camera and in total, and, over the grasps only, the
grasp study's requirement (its own `grasp_score`) on the predicted and on
the rendered fractions. Each disagreeing point gets a reason: **edge** (its pixel's 3 × 3 neighbourhood holds two
owners), **rule** (the render's own occlusion test at the lifted point, rendered range ≥ its distance − eps, agrees with
the prediction: the two definitions differ, typically a pad seen edge-on) or **occlusion** (the cube map and the
render disagree about what is in front).

**(b) Depth validity.** For every depth sensor the depth study can analyse, at home, every imager the depth needs is
rendered and the reference depth gets the sensor's noise model, z̃ = z + σ_z(z) n with σ_z = z² σ_d / (f B) (the
study's own `sigma_z`, σ_d and baseline; seeded). On the study's own voxels (the headline): field of view, min-Z, range,
projector and materials are the same functions on both sides; occlusion is the study's cube maps against the renders
(a voxel is hidden from the reference imager when the noisy depth at its pixel is nearer than it by more than 1 mm, the
study's eps, and from another imager when the rendered range is). On the rendered surfaces (reported beside it): the
study's evaluate at each rendered surface point against the noisy point, with a left-right consistency check (the other
imager's rendered range within 5 mm + 3 σ_z, a default).

Status: agrees from 90 %, mostly agrees from 75 %, else disagrees (defaults, not standards).

## Measured (2026-11-07, RTX 5080 through EGL, MuJoCo 3.17.0)

`dexvantage --sample <s> --groundtruth gt.json`, default settings (50 grasps, seed 0):

| Sample | Contact agreement | IoU | Points in view (poses) | FP / FN | Why (edge / rule / occlusion) | Requirement same verdict | Run |
|---|---|---|---|---|---|---|---|
| UR5e + 2F-85 | **91.3 %** | 86.3 % | 16 896 (33) | 1 423 / 51 | 1 061 / 324 / 89 | 32 of 32 | 7.3 s |
| SO-100 | **84.2 %** | 78.1 % | 22 905 (45) | 3 461 / 155 | 1 440 / 2 081 / 95 | 44 of 44 | 3.9 s |
| dexhand | — (no grasp set) | | | | | | 1.4 s |

With the trajectory frames (added 2026-11-07; the UR5e sample's scripted pick, 8 frames): the UR5e's total over 41
poses is **92.8 %** (IoU 88.0 %, 20 992 points in view; occlusion away from an edge 89 points, 0.4 %), of which the
grasps 91.1 %, home 98.0 % and the trajectory frames 98.9 % (4 096 points: along the scripted pick the pads face the
cameras). This is the figure the check `groundtruth.ur5e_contacts` now measures. The SO-100 and dexhand samples have no
trajectory, so their rows above are unchanged.

Per camera on the UR5e (grasps and home): Wrist D405 96.1 % (its pads are seen face on), Overhead D435 86.5 % (from above the pads are
nearly edge-on, so most of its 1 093 false positives are edge and rule cases). Where the render shows the finger and the
prediction says visible, the rendered depth minus the point's depth has median −0.04 mm (5–95 %: −10.2 to 0.6 mm; the
tail is the finger's own edge in front of a grazing point). Occlusion disagreements away from an edge are 0.5 % (UR5e)
and 0.4 % (SO-100) of the points: the cube maps and the renders agree about what is in front.

The disagreement is mostly **definitional**, and it is reported, not tuned away: the grasp study asks whether the place
where the object meets the pad (2 mm in front of it) is in view, the render whether the pad face itself shows. On the
SO-100 the jaws' contact patches come from collision geometry that does not lie on the visual mesh everywhere (the grasp
study moves 2 and 11 % of the points onto it), and its rule disagreements are the larger share.

| Sample · sensor | Voxels agree | Voxel IoU | Disagreements (predicted → rendered) | Surfaces agree (inside the task volume) | median σ_z |
|---|---|---|---|---|---|
| UR5e · Wrist D405 | 99.89 % | 99.77 % | occluded → valid 266, valid → occluded 244 (of 480 000) | 99.72 % (99.94 %) | 1.1 mm |
| UR5e · Overhead D435 | 99.66 % | 99.45 % | valid → occluded 848, occluded → valid 780 | 91.61 % (99.61 %) | 2.4 mm |
| dexhand · Head camera | 99.58 % | 99.51 % | valid → occluded 206, occluded → valid 161 | 91.44 % (97.67 %) | 0.9 mm |

The voxel disagreements sit within a texel or a σ_z of a surface. On the rendered surfaces the overhead D435 reads
91.6 %: points exactly on the table, 0.9 m away, where a cube-map texel (3.9 mrad, about 3.5 mm there) spans more depth
than the study's 1 mm tolerance, so the study calls part of the table hidden from itself (a speckle over the table in
the page's masks). Inside the task volume, where the study is used, 99.6 %.
The dexhand head camera reads 91.4 % on its rendered surfaces (re-measured 2026-11-07, RTX 5080) with the same pattern:
8 499 of its 8 595 disagreements are pixels the study calls occluded (hidden from the other imager) where the render
sees valid depth, on surfaces outside the task volume; the other way, 96. Inside the task volume 97.7 % (5 240
pixels; 106 of its 122 disagreements are predicted invalid where the render sees valid depth). This is consistent
with the texel-depth explanation above but has not been traced pixel by pixel for this camera; it is a known
limit, not tuned away.

**Validation checks** (`--validate out --models groundtruth`, full run 13 s):

| Check | Measured | Tolerance | Why |
|---|---|---|---|
| `groundtruth.lens_remap_fiducials` | 0.20 px (quick 0.19) | 0.4 px | six lens models, 35 spheres each, rendered mask centroid against the mask CameraModel predicts; MuJoCo's sphere tessellation 0.10 px plus nearest sampling of a base twice as dense (≤ ¼ px at a boundary). Measured 0.33 px with OVERSAMPLE = 1. |
| `groundtruth.scene_intrinsics` | 8.8 × 10⁻⁵ px | 10⁻³ px | MuJoCo's float32 camera fields (6 × 10⁻⁸ relative); a principal offset with the wrong sign is off by twice the offset |
| `groundtruth.scene_vs_camrender` | 2.2 × 10⁻⁵ of 2.3 M pixels | 10⁻³ | per-pixel owner, MuJoCo against core/camrender, every camera of the three samples; only edge-rule pixels differ |
| `groundtruth.ur5e_contacts` | 1 − 0.928 = 0.072 (0.087 before the trajectory frames) | 0.15 (floor 85 %) | the grasps, home and the 8 trajectory frames; plus occlusion disagreements ≤ 2 % (measured 0.4 %), on the UR5e and on the SO-100 (0.4 %); the SO-100's own agreement, 84.2 %, is under the floor and is reported in the detail, not held to it (the floor was set on the UR5e) |
| `groundtruth.ur5e_depth` | 1 − 0.998 = 0.0022 | 0.01 (floor 99 %) | the study's voxels against the noisy renders |

## Throughput

`python scripts/measure_groundtruth.py --device N`: the UR5e sample, 120 views (the 1280 × 720 D405 and the
1920 × 1080 D435, alternately) at random joint states, segmentation + depth + RGB, 8 views per call, after a warm-up.

| Device | GPU (OpenGL renderer) | Renderer frames/s | Views/s end to end |
|---|---|---|---|
| EGL device 0 | NVIDIA GeForce RTX 5080 | 76–99 | 14.7–17.5 |
| EGL device 2 | AMD Radeon (Ryzen 9 9900X integrated, radeonsi) | 68 | 14.2 |
| EGL device 1 | — (MuJoCo cannot open a PLATFORM_DEVICE display on it: the Renderer says so in words) | | |
| RTX 5080, lens swapped for Brown-Conrady (wide path, base capped at 4096 px) | | 12 | 2.4 |
| RTX 5080, lens swapped for an equidistant fisheye (5 cube faces of 2048 px) | | 19 | 0.7 (16 views) |

`device='auto'` lets MuJoCo take the first EGL device that initialises (device 0 here), or the one `DEXVANTAGE_EGL`
names. End to end is bounded by moving full-size images between the processes (about 23 MB per 1920 × 1080 view), not by
the GPU. The renders were measured while other work shared the machine; the spread is the range of two runs.

## Known limits

The prediction leaves out the camera's seat on its mount link (the 25 mm rule of `scenegeom`) and the render does not;
MuJoCo draws cylinders and spheres as smooth primitives, the predictions as 24-gons; one pixel decides each contact
point, so a point within a pixel of a silhouette can go either way (counted as "edge"); the noise model is first order
and Gaussian (no matcher bias, no holes from missing texture, no multipath); the projector pattern and IR materials are
not rendered (the study's own rules on both sides); depth is compared at the home pose; renders are geometric (MuJoCo's
shading: no exposure, noise, blur or rolling shutter); a lens rendered through cube faces moves a lot of data between
processes and is slow; equality constraints (a gripper's linkage) are not solved, as in RobotModel; the scene file
written by Build the MuJoCo scene… carries MuJoCo's XML writer's 6 significant digits (camera poses within 6 × 10⁻⁷ m
and 1.2 × 10⁻⁶ rad, intrinsics within 2 × 10⁻³ px of the project's, measured by `formats.mjcf_scene` when the file is
loaded back from disk); the renders use the scene in memory, which is not rounded.
