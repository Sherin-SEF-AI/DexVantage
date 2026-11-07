# Exporters, part A: MuJoCo, ROS 2, calibration, masks and wrappers

Spec §9.2 and §11. The exporters write the project in the formats other tools read. Each registers with
`@exporter` in `dexvantage/core/exporters/` and runs through `core.exporters.export(project, out_dir, formats,
options)`, which writes `export_manifest.json` (each format's status, files and notes, the SHA-256 of every file,
the project's provenance) and stamps every JSON file it lists with the provenance block, except the files a format
marks `unstamped` (a LeRobot or FiftyOne dataset keeps its JSON as that tool defines it).

| Format id | Module | Files (under the output folder) |
|---|---|---|
| `mjcf` | `mjcf.py` | `mjcf/cameras_snippet.xml`, `mjcf/scene/scene.xml` + `assets/` + `scene_notes.txt` |
| `urdf` | `ros2.py` | `ros2/rig.urdf`, `ros2/meshes/*.stl`, `ros2/static_transforms.launch.py`, `ros2/tf_tree.yaml` |
| `opencv_yaml` | `calibration.py` | `calibration/opencv/<sensor>__<imager>.yaml`, `calibration/kalibr/camchain.yaml`, `calibration/cameras.json` |
| `masks` | `masks.py` | `masks/self/*.png` (+ `_owner.png`), `masks/atlas/*`, `masks/depth/*.png` (+ `_valid.png`), `masks/masks.json` |
| `plucker` | `bridges.py` | `plucker/plucker_<policy>_<camera>.npy`, `.safetensors`, `.json` (core/plucker.py) |
| `brackets` | `bridges.py` | `brackets/<sensor>.stl` + `.stl.json`, `.step` with gmsh |
| `fiftyone` | `bridges.py` | `fiftyone/` (core/audit/report.export_fiftyone after an audit of `options.dataset`) |
| `lerobot_dataset` | `bridges.py` | `lerobot_dataset/` (core/sim/episodes.generate) |

Batch: `dexvantage PROJECT.dexv --export OUT_DIR --formats mjcf,urdf,... [--export-options JSON]`; an unknown
format name is refused in one line before anything is written (exit 2); exit 1 when some formats are refused and the
others written, 0 when all are written. In the window: File ▸ Export… (also Export ▸ Export to MuJoCo, Isaac, ROS 2,
LeRobot…), a window job; the dialog then shows the manifest.

## Frames and the pose of an export

A sensor's body frame is `<sensor>_link` (REP-103: x forward, y left, z up) and each imager's optical frame is
`<sensor>_<imager>_optical_frame` (REP-103's optical frame: z forward, x right, y down); the default imager of a
sensor without a device is named by its role (`rgb`). A sensor on a robot link hangs from that link's frame
(DexVantage's link frame: the MuJoCo body of the robot's file, or the URDF link); a sensor on a static frame (cage,
tripod, table) hangs from `world`, the primary robot's base, because static mount frames have no transforms of their
own (`core/scenegeom.sensor_world_pose`). Every export takes one joint state (`options.q`, {robot uid: joint values};
default every robot at home; the dialog offers "the pose on screen"): it matters for world poses of cameras on links,
the Kalibr chain between a moving and a fixed camera, and the masks.

## MJCF (`mjcf`)

**Snippets.** For every body carrying a sensor, the elements to paste inside `<body name="…">` of the robot's own
file (its body names, no prefix), and a block for `<worldbody>` (the project's world). Each sensor gets
`<site name="<sensor>_mount">` at its body frame (MuJoCo's default site type and size, a drawing aid) and each imager
a `<camera>` at T_body_optical · diag(1, −1, −1) (MuJoCo's camera looks along −z with +y up) with

    resolution = (W, H),  sensorsize = (W, H) · 3 µm (nominal: only the ratios matter),
    focalpixel = (fx, fy),  principalpixel = ((W − 1)/2 − cx, (H − 1)/2 − cy)

exactly as `core/sim/scene.py` sets them (its docstring cites MuJoCo 3.17.0's XML reference, and the principal-point
sign was measured on renders, `validation/mjrender.py`). Numbers are written with 17 significant digits (Python's
`repr`). A static sensor is in the project's world frame, which is the primary robot's file frame; for another
robot's file move it by the inverse of that robot's base (the notes say so). MuJoCo cameras are pinhole: a distorted
lens gets its pinhole part and the notes say so.

**Scene.** `core/sim/scene.save_scene`, the scene the ground-truth renders use (robots, objects, tables, lights,
every imager), written by MuJoCo's own XML writer (6 significant digits) with its mesh and texture files.

## ROS 2 (`urdf`)

The URDF is written from the MuJoCo model DexVantage compiles (`core/robot/model.py`), so MJCF robots export too.
Each MuJoCo body is a link at DexVantage's link frame. MuJoCo moves a body with a hinge j (anchor p and axis a in the
body frame, reference qpos0) as

    X_body(q) = X_parent · T0 · Trans(p) · Rot(a, q − qpos0) · Trans(−p)

(mj_kinematics; T0 the body's pos and quat), and a slide as X_parent · T0 · Trans(a (q − qpos0)). URDF moves a child
as X_parent · origin · Rot(axis, q) about the child frame's origin [urdfdom joint and pose parsers], so

    origin = T0 · Trans(p) · Rot(a, −qpos0)       (a slide: T0 · Trans(−a qpos0))

and, when p ≠ 0 (or a body has several joints), an extra link `<body>__<joint>` carries the joint and a fixed joint
with origin Trans(−p) brings the body back to its own frame. A body with free or ball joints is fixed at its home
pose (DexVantage holds them there). A URDF link MuJoCo merged into its parent is a fixed joint at its recorded
offset. With several robots every link and joint is prefixed `<robot>_`.

`origin rpy` is urdfdom's fixed-axis roll, pitch, yaw, R = Rz(yaw) Ry(pitch) Rx(roll) (`urdf_model/pose.h`
`setFromRPY`). The writer takes pitch = atan2(−R20, hypot(R00, R10)), yaw = atan2(R10, R00), and the roll from
Ry(pitch)ᵀ Rz(yaw)ᵀ R, which stays within rounding at and near ±90° of pitch (asin loses half the digits there, and
an ill-conditioned yaw near the lock is absorbed by the roll; `test_rpy_writer_keeps_precision_near_gimbal_lock`).

Joint limits are DexVantage's (the robot's records or the file's ranges); without finite limits a hinge is
`continuous`; a slide without limits is refused (URDF cannot write it). urdfdom refuses a `<limit>` without `effort`
and `velocity` (`urdf_parser/src/joint.cpp`); when the robot's records do not state them they are written as 0 and
the notes say so: set them before giving the file to a controller or a planner. No figure is invented: no inertials
(DexVantage does not use mass, and not every file states it) and no collision geometry; each link's visual meshes
are one binary STL in metres in its frame.

`static_transforms.launch.py`: one `tf2_ros` `static_transform_publisher` Node per sensor frame (parent →
`<sensor>_link`, `<sensor>_link` → each optical frame), with the program's flags `--x --y --z --qx --qy --qz --qw
--frame-id --child-frame-id` (ros2/geometry2 at commit 183399c, read 2026-11-07), for use beside your own robot
description. `tf_tree.yaml`: every sensor frame with its parent, kind, whether it moves with a robot link, the static
transform and its world pose at the export's joint state.

## Calibration (`opencv_yaml`)

Per imager an OpenCV YAML written by `cv2.FileStorage` (`image_width`, `image_height`, `camera_name`,
`camera_matrix`, `distortion_model` in ROS's names, `distortion_coefficients`, `opencv_functions`, `lens_model`,
`exact`, `approximation_error_px`, `unmapped_share`, `parent_frame`, `camera_frame`, `T_parent_camera`,
`T_world_camera`, `joint_state`); a Kalibr `camchain.yaml` (`cam0…`: `camera_model`, `intrinsics [fu fv pu pv]`,
`distortion_model`, `distortion_coeffs`, `resolution`, `rostopic` (a suggestion), and from cam1 on `T_cn_cnm1`,
which "takes cam0 to cam1 coordinates" for cam1 [Kalibr wiki "YAML formats", read 2026-11-07]); and `cameras.json`
with every imager's `CameraModel.to_dict`, OpenCV's K and D, both mappings, the frames and the extrinsics.

**Lens mapping.**

| DexVantage | OpenCV | Kalibr |
|---|---|---|
| pinhole | plumb_bob, zeros (exact) | pinhole / none (exact) |
| Brown-Conrady, 5 terms | plumb_bob (exact) | radtan [k1 k2 p1 p2] (exact when k3 = 0, else fitted) |
| Brown-Conrady, 8 terms | rational_polynomial (exact) | radtan, fitted |
| Kannala-Brandt | fisheye (exact) | equidistant [k1..k4] (exact) |
| equidistant | fisheye, zeros (exact) | equidistant, zeros (exact) |
| equisolid, stereographic | fisheye, least-squares KB (no exact form) | equidistant, least-squares KB |

Kalibr's radtan is OpenCV's Brown-Conrady with k3 = 0 and its equidistant is OpenCV's fisheye (aslam_cv
`RadialTangentialDistortion` and `EquidistantDistortion` at Kalibr commit 1f60227). A Brown lens with k3 or rational
terms has its radtan terms fitted by linear least squares: with (x, y) = (X/Z, Y/Z) of the lens's rays at a 41 × 41
grid of pixels and (xd, yd) their distorted coordinates,

    xd − x = k1 x r² + k2 x r⁴ + 2 p1 x y + p2 (r² + 2x²),  yd − y = k1 y r² + k2 y r⁴ + p1 (r² + 2y²) + 2 p2 x y

is linear in (k1, k2, p1, p2). The radial fisheyes go to Kannala-Brandt by `CameraModel.kb_approximation`. Every
approximation states its error: the largest distance between where the lens and the written model put the same ray,
over the 41 × 41 grid, and the share of pixels whose rays the written model cannot map. OpenCV's and Kalibr's
fisheye compute θ from X/Z, so they take only rays in front of the camera: a lens whose image reaches 90° off the
axis is said to have no exact equivalent there.

Pixel convention: DexVantage, OpenCV and ROS put pixel centres at integers; Kalibr's projection is u = fu·x + pu
(aslam_cv `PinholeProjection`) and its documentation states no pixel-centre convention, so the same numbers are
written.

## Masks (`masks`)

**Self-occlusion.** Per camera (each sensor's primary imager by default; `imagers: "all"` for every imager) the
Camera view's renderer (`core/camrender.py`, each pixel-centre ray through the imager's own lens) of the robots at q,
the task objects and the tables, with the camera's seat on its mount link left out by the occlusion study's rule
(`scenegeom.sensor_occluders`, OWN_RADIUS_M = 25 mm, a default). A pixel is 255 when the first surface on its ray is
a robot link (any robot), else 0. The 16-bit owner image holds 1 + the index into `masks.json`'s `owners`
(`robot:<uid>:<link>`, `object:<uid>`, `surface:<uid>`), 0 for nothing, so a policy can mask one link. `scale`
(default 1) renders at a share of the stream size (the lens scaled by `CameraModel.scaled`).

**Atlas.** Over a grasp set (`grasp_set`, default the first), the grasp study's inverse kinematics
(`grasp_visibility.solve_grasps` with its settings) poses the robot; up to 4 × `atlas_grasps` grasps are solved
(evenly picked) and `atlas_grasps` of the reachable ones kept (8, a default), each camera rendered at `atlas_scale`
(0.25, a default): one tiled PNG per camera (row-major) and a frequency PNG (the share of those grasps in which the
robot covers each pixel, 0–255).

**Depth validity.** Per depth sensor, every pixel of the depth (reference) imager at the depth resolution takes the
first surface on its ray, and that point is tested by the depth validity study's own `evaluate()`; the pixel's code
is the first condition it fails (0 valid, 1 fov, 2 min_z, 3 max_range, 4 occluded, 5 projector, 6 material) and 255
when the ray meets nothing the project models (the floor is not geometry in DexVantage). The other imagers' lines of
sight use the study's cube maps at 2048 texels per face (a default; at the study's 512, a texel spans more depth than
the 1 mm tolerance on far grazing surfaces and 4.4 % of the overhead D435's pixels read occluded); the reference
imager sees each pixel's point by construction.

## Wrappers

`plucker` calls `core/plucker.export_plucker` for a policy (default the first with a camera in use) at the export's
pose. `brackets` builds each sensor's bracket (`core/brackets.build`, the Brackets page's settings) and writes STL
with its sidecar; with `step: true` and gmsh installed (`pip install "dexvantage[step]"`, gmsh 4.15.2) each triangle
becomes a planar face of one OpenCASCADE solid (points, shared lines, `addSurfaceLoop(sewing=True)`), written as
STEP: a faceted solid, not parametric CAD; above `step_max_triangles` (20 000, a default) it is skipped with a note,
and `step_res_mm` meshes the STEP on a coarser grid (measured on the UR5e wrist flange adapter: 119 092 triangles at
0.5 mm; with the STEP at 2 mm the bracket export took 6.8 s in all and the STEP is 16.7 MB). Without gmsh the request is refused in one sentence.
`fiftyone` audits `options.dataset` (per-frame scores on) and writes the FiftyOne export; `lerobot_dataset` generates
synthetic episodes (`core/sim/episodes.generate`, its settings as options) into `lerobot_dataset/`.

## Validation (measured 2026-11-07, `formats` model)

| Check | Reference | Tolerance | Measured |
|---|---|---|---|
| `formats.mjcf_snippets` | snippets parsed by MuJoCo, attached to each body of the robot's own model; camrender.optical_pose | 10⁻⁹ m, 10⁻⁹ rad, 10⁻³ px | 3.1 × 10⁻¹⁷ m, 7.7 × 10⁻¹⁶ rad, 8.8 × 10⁻⁵ px (float32 cam_intrinsic) |
| `formats.mjcf_scene` | the saved scene from disk (Phase 4 check) | 2 × 10⁻⁶ m | 5.8 × 10⁻⁷ m |
| `formats.urdf_roundtrip` | ElementTree walk with urdfdom's RPY quaternion and Rodrigues written again; MuJoCo's URDF compiler | 10⁻⁹ m, 10⁻⁹ rad, no structural problem | 1.1 × 10⁻¹⁵ m, 1.3 × 10⁻¹⁵ rad |
| `formats.ros2_tf` | launch file read with `ast`, tree with PyYAML, composed on RobotModel link poses | 10⁻⁹ m | 2.8 × 10⁻¹⁷ m, 7.5 × 10⁻¹⁶ rad |
| `formats.opencv_yaml` | cv2.FileStorage read-back; cv2.projectPoints / fisheye.projectPoints, seven lens models | 0 (bit for bit); 10⁻⁶ px exact mappings | 0; 5.1 × 10⁻¹³ px; approximations 3.7 × 10⁻⁹ px (equisolid) and 1.6 × 10⁻⁴ px (stereographic), within their stated 4.8 × 10⁻⁹ and 3.7 × 10⁻⁴ px |
| `formats.kalibr_camchain` | PyYAML read-back; chain from optical poses; ConfigReader's model rules | 0 | 0 |
| `formats.masks_vs_occlusion` | exact ray casting (core/raycast.py) on the occlusion study's occluders; its cube map (n = 2048); depth conditions per pixel by ray casting | 10⁻³ of pixels; cube 0.5 %; depth 1 % | 0 of 187 200; 0.07 %; 0.25 % |
| `formats.bracket_step` (with gmsh) | OpenCASCADE's STEP reader and mass properties against the mesh's divergence-theorem volume | 10⁻⁶ relative | 3.6 × 10⁻¹³ |

The self-test line "exporters" exports the UR5e sample as URDF and OpenCV YAML, compiles the URDF in MuJoCo and
compares every optical frame at a random joint state (about 0.5 s).

## Known limits

- One joint state per export: a camera on a link has its world pose, Kalibr chain and masks at that pose only.
- URDF: no inertials, no collision geometry, equality constraints and mimic joints not written; effort and velocity
  0 when not stated; one colour per link.
- MJCF snippets: a URDF link MuJoCo merged into its parent is reached through the parent body; MuJoCo cameras are
  pinhole (a distorted lens's distortion is not in the file).
- Calibration: no camera-IMU transforms (DexVantage models no IMUs); Kalibr's omni, ds and eucm models are not
  fitted, so a fisheye past 90° has no exact Kalibr or OpenCV form; rostopic names are suggestions.
- Masks are geometric (visual meshes; no cables, housings, blur or anti-aliasing); a distorted lens's mask edges move
  by up to half a pixel of the renderer's pinhole base image; the atlas poses only the grasp's robot, fingers at
  their home opening; depth codes test the surface at the pixel-centre ray.
- STEP is a faceted solid (a face per triangle), large and slow above a few tens of thousands of triangles.
- The FiftyOne wrapper re-runs the audit (not cached between exports); LeRobot's own loader and FiftyOne's importer
  were not run (not installed).
