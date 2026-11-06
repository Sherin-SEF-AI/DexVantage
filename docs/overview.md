# DexVantage: models and methods

Every model DexVantage uses, written out with its equations, references and limits. Each phase adds its models here.

## Frames and units {#frames}

Units are SI. The world frame is the primary robot's base frame (REP-103: x forward, y left, z up). A camera's
optical frame is REP-103's `_optical_frame`: z along the optical axis, x right, y down. For a camera link whose x axis
is the optical axis, a point's optical coordinates are

    p_opt = B · p_body,   B = [[0, −1, 0], [0, 0, −1], [1, 0, 0]].

Rotations are unit quaternions q = (x, y, z, w) with w ≥ 0 (one of the two signs, so equal rotations compare equal).
The matrix is the standard one [Shoemake 1985]; the inverse uses Shepperd's method [Shepperd 1978], taking the
largest of the four diagonal combinations so half turns stay accurate. URDF's rpy is R = R_z(yaw) · R_y(pitch) ·
R_x(roll) (fixed axes x, y, z). MuJoCo writes (w, x, y, z); the conversion happens at the boundary. The angle between
two rotations is 2·atan2(|v|, |w|) of q₁⁻¹ q₂, which keeps full precision for small angles.

**Validation:** six checks against scipy's Rotation and MuJoCo's `mju_quat2Mat` (worst error 3 × 10⁻¹⁴).
**Known limits:** rpy is singular at pitch ±90° (the yaw takes the whole turn there).

## The project summary and the dense-voxel limit {#summary}

A dense axis-aligned grid over a box of edges (a, b, c) at voxel size s has N = ⌈a/s⌉·⌈b/s⌉·⌈c/s⌉ voxels. At 1 mm,
1.5 × 1.5 × 1.2 m needs 2.7 × 10⁹ voxels, so the coverage studies (Phase 2) use 5 mm by default and refine to 1–2 mm
only inside refinement regions. The summary reports N at the chosen size and at 1 mm, and marks a dense grid above
5 × 10⁷ voxels as not feasible (the limit is a documented default of this build, not a standard).

## Robots {#robots}

A URDF or MJCF robot is read with every file it names into the project and compiled by MuJoCo from memory; joint
limits and the home pose are the file's own. The UR5e + Robotiq 2F-85 and SO-100 samples are bundled MuJoCo
Menagerie models, after their licences were read. See [models/robots.md](models/robots.md).

## Robot kinematics {#kinematics}

MuJoCo poses the links: T_link(q) from its tree of body transforms and joint motions, and the world Jacobian of a
point from `mj_jac`. An independent product-of-exponentials chain checks it,

    T(q) = e^[S₁]θ₁ · e^[S₂]θ₂ ⋯ e^[Sₙ]θₙ · M        [Lynch and Park 2017]

with its own XML reader and twist exponentials (`core/robot/kinematics.py`), and both engines are checked against
the closed-form DH tables of the UR5e and the Panda: see [models/kinematics.md](models/kinematics.md).

## Inverse kinematics, manipulability and collision distances {#ik}

Damped least squares with error-dependent damping [Wampler 1986; Sugihara 2011], joint-limit avoidance in the null
space [Liégeois 1977], several seeds and a choice of solution; Yoshikawa's manipulability w = √det(J Jᵀ) and the
condition number σ_max/σ_min; signed distances between collision geoms from MuJoCo with a bounding-sphere prefilter.
See [models/ik.md](models/ik.md).

## The hardware library {#library}

One JSON file per device (`dexvantage/resources/library/<kind>/<id>.json`, schema `library.schema.json`): depth
cameras, cameras, tactile sensors and hosts, every figure copied from a cited source (URL or document, title, the
date read, what it covers) and `null` where the source is silent, shown as "—" with the tooltip "not stated by the
maker". An offset the maker does not give (an RGB imager's position in a D435, a projector's) is placed as a
labelled assumption (`pose_stated: false`). `library.sensor_imagers(sensor)` turns a sensor's device into imagers:
each a pose in the device body frame and a lens model fitted to the stated FOV at the chosen stream (a centred crop
for other aspect ratios, a square-pixel pinhole from a diagonal-only FOV; both assumptions the imager's label says),
or the unit's own calibration. A sensor with no device gets a labelled default camera (pinhole 1280 × 720 px,
90° × 60°). 40 entries in Phase 2. See [models/library.md](models/library.md).

## Lens models, pixel density and defocus {#optics}

`CameraModel` (pixels, OpenCV conventions: pixel centres at integers): pinhole; Brown-Conrady with 5 or 8 terms
[Brown 1966]; Kannala-Brandt, OpenCV's fisheye θd = θ(1 + k₁θ² + k₂θ⁴ + k₃θ⁶ + k₄θ⁸) [Kannala and Brandt 2006];
equidistant, equisolid and stereographic. Every model has a valid angle beyond which it is not one-to-one, and
points past it are never projected. Pixel density at a point is √|det J| of the image Jacobian over a patch facing the
camera (f/Z on a pinhole's axis; × √cos α for a tilted surface), and defocus is the thin-lens blur disc
b = A f |s − z| / (z (s − f)) [Ray 2002]. Ported from Vantage `core/lens.py`. See
[models/optics.md](models/optics.md).

## Occlusion by the posed robot {#occlusion}

Around each imager's optical centre, a cube map of distances to the nearest occluder (the robots' visual meshes
posed at q, the task objects, the tables): six 90° faces of n × n texels (default 512), built by a z-buffer on the GPU
(EGL) or by an exact numpy rasteriser (the ray through each texel centre against each triangle; the depth is the
exact plane distance along that ray). A point is visible when its distance is at most its texel's depth plus 1 mm (a
default); near-plane clipping at 5 mm keeps a sensor from being blinded by its own seat. Exact ray casting
(Möller-Trumbore over a uniform grid) is the independent reference. Ported from Vantage `core/occlusion.py` (the
cube-map idea and face basis). See [models/occlusion.md](models/occlusion.md).

## Sparse voxels {#voxels}

Each task volume is covered by cubic voxels of 5 mm (a default), split into k³ voxels of at most 2 mm (a default)
inside refinement regions and around the task objects. A dense 1 mm grid over 1.5 × 1.5 × 1.2 m is 2.7 × 10⁹ voxels;
a grid over 5 × 10⁷ voxels is refused with a message saying so. See [models/voxels.md](models/voxels.md).

## Coverage and resolution {#coverage}

For every voxel and camera: in the image (projected by the imager's lens model), visible (the cube map), pixels per
millimetre, the viewing angle, the defocus blur where every lens figure is stated or entered, and redundancy (how
many cameras see it). Shares are volume-weighted; the headline is the share seen by two or more cameras, with status
thresholds 90 / 60 / 30 % (defaults). See [models/coverage.md](models/coverage.md).

## Depth validity {#depth}

A voxel is depth-valid for a depth sensor when it is (1) inside every needed imager's FOV (both of a stereo pair),
(2) beyond min-Z for the chosen resolution (the maker's table) and within the stated range, (3) seen by every needed
imager (finger shadows and stereo parallax), (4) inside the projector cone where texture is needed, and (5) not on a
surface tagged transparent, IR-absorbing or specular (a warning). The expected error is σ_z = z² σ_d / (f B)
[Hartley and Zisserman 2004], σ_d = 0.08 px by default (labelled), shown beside the maker's RMS statement. See
[models/depth.md](models/depth.md).

## The Camera view {#camview}

The selected sensor's image through its imager's lens model: an exact per-pixel z-buffer of the posed robots,
objects, tables and floor, with the task volumes' visible edges and the analysis page's overlays; rendered on a
worker thread, at half size while dragging. See [models/camview.md](models/camview.md).

## Performance {#performance}

No UI-thread step over 100 ms on the samples with the shader renderer (largest: 68 ms median, 96 ms worst, for a UR5e
frame after a joint change on an integrated GPU). See [models/performance.md](models/performance.md).

## Provenance {#provenance}

As Vantage: a block under `provenance` (schema `dexvantage.provenance/1`) with the version, a SHA-256 of the package
sources, library versions, the project's content hash and per-section hashes, the study and its settings, seeds,
the batch command with paths reduced to file names, and the SHA-256 of the result without its time stamps.
`--reproduce` re-runs a result and compares every number. No path, user or host name is ever written.

## Validation {#formats}

`dexvantage --validate DIR` or Help ▸ Validate the models. Each check compares one model quantity with an independent
reference and passes when the error is at most its stated tolerance. Phase 0: 8 checks (frames; the project file
against its JSON Schema through the jsonschema package; save and load identical). Phase 1: 10 kinematics checks (FK
and Jacobians against an independent chain, DH closed forms and finite differences; IK round trips; the 2R
manipulability closed form; analytic sphere and capsule distances). Phase 2: 22 more, 40 in all — optics 10
(OpenCV's `projectPoints` and `fisheye.projectPoints`, round trips, FOV fits, finite differences, the hyperfocal
formulae, MuJoCo-rendered fiducials), occlusion 7 (exact ray casting, MuJoCo segmentation on scripted scenes and on
the UR5e, the Camera view and the coverage study against ray casting), depth 4 (σ_z against Monte Carlo, Intel's
min-Z and FOV tables, the stereo-overlap formula, finger shadows against ray casting from both imagers) and the
project-obstacle distance. With no project given, the project-dependent checks run on the bundled SO-100 and UR5e
samples.

## Model documents

| Document | Models |
|---|---|
| [models/robots.md](models/robots.md) | robot import, the MuJoCo kinematics model, bundled models and samples |
| [models/kinematics.md](models/kinematics.md) | product-of-exponentials forward kinematics, Jacobians, DH tables |
| [models/ik.md](models/ik.md) | inverse kinematics, manipulability, collision distances |
| [models/library.md](models/library.md) | the hardware library: entry format, sources and dates, the gaps policy, from a sensor to imagers, custom devices |
| [models/optics.md](models/optics.md) | lens models and their valid angles, FOV fitting, pixel density, viewing angle, thin-lens defocus, MuJoCo fiducial check |
| [models/occlusion.md](models/occlusion.md) | the scene's occluders at q, articulated cube-map occlusion (numpy and GPU), exact ray casting, decimation, timings |
| [models/voxels.md](models/voxels.md) | sparse voxel grids with adaptive refinement, the dense-grid limit |
| [models/camview.md](models/camview.md) | the Camera view: a sensor's simulated image through its lens model, lens-model frusta (worked example: [guide/sensors.md](guide/sensors.md)) |
| [models/coverage.md](models/coverage.md) | coverage and resolution of the task volume: visibility, pixels per mm, viewing angle, defocus, redundancy (worked example: [guide/coverage.md](guide/coverage.md)) |
| [models/depth.md](models/depth.md) | depth validity: the five conditions, min-Z zones, σ_z against the maker's RMS figure (worked example: [guide/depth.md](guide/depth.md)) |
| [models/performance.md](models/performance.md) | the UI-thread budget: measured timings on the samples and what was changed to meet it |
