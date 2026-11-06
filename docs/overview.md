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

## Grasp sets {#grasps}

A grasp set (`core/grasps.py`, `project.tasks.grasp_sets`) is a list of gripper poses at contact, each with the
direction the gripper moves in. It is sampled over a task volume (top-down pinch, side, angled, bimanual handoff),
imported from CSV or JSON, or taken from a trajectory. The grasp centre and the gripper's approach and closing axes are
measured from the contact pads at home, not typed in: the robot's tagged pads, else the inner faces of each
fingertip's collision geometry (faces within 40 mm of the tip facing the other fingertips within 45°, not covered by
the same finger). On the UR5e sample this gives the inner faces of the 2F-85's pads, 130.7 mm out along the gripper
base's z axis. **Validation:** the pads against MuJoCo's compiled pad boxes 3.6 × 10⁻¹¹ m (10⁻⁶ m), the grasp centre
against Menagerie's `pinch` site 0.0° (0.5°), the samplers against their definitions written out again 4.4 × 10⁻¹⁶,
object surface points against a box's closed form 9.7 × 10⁻¹⁷ m. **Known limits:** samplers do not check reach or
collisions (the study's IK does); object grasps use the object's axis-aligned box, so the width is an upper bound;
dataset grasp sets wait for the Phase 8 readers. See [models/grasps.md](models/grasps.md).

## Grasp self-occlusion {#grasp}

For every grasp of a set: batched damped-least-squares IK for the grasp centre (several seeds, joint-limit avoidance,
self-collision and the project's obstacles; the rule nearest home or most manipulable; infeasible grasps counted by
reason; pinocchio and cuRobo as optional backends, every solution re-checked by DexVantage's own FK). At each
solution the contact patches ride with their links; each patch point, lifted 2 mm along its normal (a default: the
inner faces of a parallel gripper face each other, so the faces themselves are blind from every side view), is
visible from a camera when it projects inside the image through the lens model and nothing in the posed scene's cube
map is nearer. Cube maps are cached by q-bucket (0.1°, a default). Per grasp, the requirement (default: one camera sees
every fingertip at 50 % or more) holds or not; the **Contact Visibility Score** is CVS = k / n over the reachable
grasps, with a percentile bootstrap 95 % interval [Efron 1979] (drawn as Binomial(n, p̂)/n, the exact distribution
of the resampled mean) and Wilson's score interval [Wilson 1927] beside it. Pre-contact visibility over the last 5 cm,
the both-fingertips rate, a worst-case gallery and "what to move" (each camera moved ±2 cm and turned ±5°, one at a
time) complete the result. **Performance (§16):** 2 000 grasps on the UR5e sample in 111.6 s on an NVIDIA GeForce RTX
5080 (EGL device 0) and 131.3 s on the Ryzen 9 9900X's integrated Radeon, 17.9 and 15.2 grasps/s, 10 808 cube maps;
85.5 s with 256-texel maps, 83.6 s without the sensitivity sweep. **Validation:** visibility against exact ray casting
of the full scene, 0 of 7 892 points away from an edge (10⁻³); the bootstrap's coverage against the known p of
binomial samples, within 0.017 of 95 % (0.031); the q-bucket against evaluation at each grasp's own q, mean 0.0024 of
a patch (0.02). **Known limits:** the fingers stay at their home opening; obstacles are not checked along the approach;
the 2F-85's linkage joints are independent entries of q. See [models/grasp.md](models/grasp.md).

## Policies {#policies}

A project names the policies its data is for (`core/policies.py`, `project.policies`): a policy library entry (8:
OpenVLA 7B, π0, π0-FAST, SmolVLA base, ACT, Diffusion Policy, Octo-Base, GR00T N1 2B), the cameras it takes and the
user's own figures over the entry's. Every figure (input resolution, crop rule, patch, tokens per image, control rate,
action horizon, inference time) is cited from the model card, the paper or the official code at a pinned commit, with
a quote; a figure no source states is null and shown as not stated. The crop geometry maps a camera pixel (u, v) to
the input, x_in = dst_x0 + (u + ½ − x0)·s_x, for six rules (resize with padding, centre crop then resize, plain resize,
centre crop, top-left padding, and GR00T's fractional crop). **Validation:** each rule's pipeline written out again
from its source code and run through OpenCV's resize, 2.5 × 10⁻¹³ px over 1.4 million input pixels (10⁻⁶ px). **Known
limits:** which camera fills which slot (base or wrist) is not modelled; Diffusion Policy's step from a 16:9 stream to
320 × 240 is not stated and is modelled as a stretch, labelled. See [models/policies.md](models/policies.md).

## VLA token resolution and Plücker ray maps {#vla}

Per policy and camera: the camera at its stream through its lens model, the policy's crop geometry, and a token grid
of the stated patch (cols = ⌊w/p⌋, rows = ⌊h/p⌋), or a cell derived from tokens per image when that tiles the input on
a square grid (ACT 32 px, GR00T 28 px, labelled derived). Each task object's and fingertip's triangles are projected
(subdivided on distorting lenses) and rasterised at input resolution; the study counts the tokens they cover at the
pre-grasp and contact poses of a grasp set and warns below 2 × 2 tokens (a default) or when more than half is cropped
away. Plücker ray maps (o × d, d) in the robot base frame are exported per input pixel as `.npy` or `.safetensors`
(ported from Vantage `core/plucker.py`; DexVantage's own safetensors writer). **Validation:** token counts against
ray casting every input pixel's centre, 0 tokens (0); Plücker moments against OpenCV's undistortion, 1.3 × 10⁻¹⁵
(10⁻⁶); the safetensors round trip bit-exact. **Known limits:** silhouettes are not occlusion-tested (the grasp study
does visibility); Octo's 128 × 128 wrist input is not applied per camera role. See [models/vla.md](models/vla.md).

## Multi-finger occlusion {#dexhand}

For a dexterous hand (the `dexhand` sample: the Shadow Hand on a UR5e, Apache-2.0 and BSD-3-Clause), over grasp
postures swept between the hand file's own keyframes or over a grasp set: per fingertip patch point P and camera, the
first hit along the ray to P and its owner (each finger, the palm, the arm, objects, the rest), d(P) = min_g d_g(P),
seen ⇔ in view ∧ |P − o| ≤ d(P) + 1 mm. The finger × camera matrix gives the share of each patch not seen and what
hides most; the mount comparison ranks a palm micro camera, a dorsal wrist camera and a head camera; "what to move"
turns each camera ±10° about its optical centre (no ray casting needed). **Validation:** the matrix against a central
projection written out in closed form, 0 (10⁻³); owners against all-pairs Möller–Trumbore, 0 (10⁻³); the union of
group cube maps against one cube map, 2.0 × 10⁻⁸ m (10⁻⁶ m). **Known limits:** the postures are sample sweeps, not
measured synergies; the Shadow Hand's tendon coupling is not enforced between keyframes; part 2 (§7.15, the tactile
handoff) is Phase 5. See [models/dexhand.md](models/dexhand.md).

## Bandwidth, hosts and power {#bandwidth}

Each stream's payload is width × height × fps × bits per pixel / 10⁶ Mbit/s from the library's stream modes, on the
wire × (1 + overhead), the overhead a labelled default of 5 % per interface; a stream whose maker states no bits per
pixel has no rate until a measured one is entered. The host topology (`project.rig`: hosts, controllers, hubs, which
sensor plugs where) gives each controller and link its load against its capacity (USB 2.0: 53.248 MB/s from Table 5-10;
USB 3.2: 450 MB/s and 1.1 GB/s from §4.4.11, doubled for x2), the margin and flags (a controller over budget, a hub on
a shared controller); disk per episode and campaign; bus power against USB 3.2 Table 3-1 (900 mA) and USB 2.0 §7.2.1
(500 mA). Ported from Vantage `core/recording.py` and `core/power.py`. **Validation:** the NanEyeC's stated serial
rates bound its payload (79 %); the USB figures against their derivations, 0; the budget re-computed independently
over 40 random topologies, 3.9 × 10⁻¹⁶. **Known limits:** GMSL link rates and deserialiser power are not stated in the
library; USB Type-C current levels and Power Delivery are not modelled. See [models/bandwidth.md](models/bandwidth.md).

## Cable flex across joints {#cables}

A cable runs between attachments on links. Across one revolute joint it is a belt over a pulley: tangent, arc round a
wrap cylinder of radius ρ (default the nearer attachment's distance from the axis), tangent, with the height
difference unrolled, L = √(L_plane² + Δz²), L_plane = t_a + t_b + ρ(|θ| − α_a − α_b) when it wraps; the angle is not
folded to half a turn, so a joint turned further winds the cable. Each crossed joint is swept over its limits and
trajectories are played at 2°: slack, bend radius ρ/cos²φ (a helix [do Carmo 1976]) against the cable's stated minimum
× a factor, pull, and snags (the slack loop through a link). A harness sheet lists cable, length, slack per joint,
connectors and strain relief (ported from Vantage `core/harness.py`). **Validation:** the arc against the circle, helix
and belt closed forms, 1.7 × 10⁻¹⁶ m (10⁻⁹ m); helix bend 4.9 × 10⁻⁵ (5 × 10⁻³); loop arc 4.4 × 10⁻⁴ (5 × 10⁻³);
snags against brute force, 0 poses. **Known limits:** runs across several joints are straight chords; snags are tested
at home and trajectory poses only, against the cable's own robot. See [models/cables.md](models/cables.md).

## Brackets and mounts {#brackets}

Watertight, printable STL from signed-distance solids meshed by marching tetrahedra (ported from Vantage
`core/meshing.py` and `core/brackets.py`): wrist-flange adapters (ISO 9409-1-50-4-M6 from UR's drawing; other sizes
from the designation), 2020 and 4040 T-slot clamps, a pan-tilt neck plate, palm and dorsal sleeves for micro cameras;
the camera's stated mounting holes where its datasheet gives them (RealSense D405, D435, D435i). Clearance: the
bracket and camera body against every link but the mount link at home, on screen and 48 seeded joint states, and
against cable paths when there are cables. **Validation:** edges, winding, Euler characteristic and volume written out
again, 0 defects on all six kinds; the flange disc's volume against its closed form, 0.18 % (1 %); hole centres fitted
on the mesh against the pattern, 4.2 × 10⁻⁵ mm (0.05 mm). **Known limits:** STEP export is not done (STL with a JSON
sidecar); other ISO 9409-1 sizes lack centring and dowel figures. See [models/brackets.md](models/brackets.md).

## Hand-eye calibration {#handeye}

Boards (ChArUco, AprilGrid), sub-pixel detection and PnP (ported from Vantage `core/targets.py`, `core/calib.py`);
closed-form AX = XB (Tsai-Lenz, Park-Martin, Horaud-Dornaika, Andreff, Daniilidis) and AX = ZB (Shah, Li) written from
the papers in NumPy, used because OpenCV 5.0.0's Python bindings lack `calibrateHandEye` and
`calibrateRobotWorldHandEye` (OpenCV's are used where present); then one sparse bundle adjustment over the corners,
the joint readings and optionally the intrinsics and joint zero offsets, multi-camera and multi-arm, with its
covariance. The planner picks 15–25 poses (default 20) from candidates on a hemisphere around the board, IK-feasible,
collision-free and with the board in view, greedily D-optimal in the Fisher information of X and Z, and predicts σ of
X and Z in mm and degrees; the coach says which pose to take next. **Validation:** the closed forms on exact poses
1.8 × 10⁻¹⁵ m (10⁻⁹ m); with 0.3 px noise 5.9 mm (15 mm); the adjustment 3.1 σ from the truth (5 σ); its covariance
against 40 Monte Carlo draws within the χ² band; AX = ZB with two arms 3.5 mm (15 mm); rendered ChArUco and AprilGrid
corners 0.113 and 0.118 px (0.25 px); the planner's predicted σ against the adjustment's, ln ratio 0.0087 (0.095).
**Known limits:** no comparison with OpenCV's own solvers has run (not in this OpenCV); Tsai-Lenz is singular near
180° motions; time offsets between images and joint readings are not modelled. See [models/handeye.md](models/handeye.md)
and its [planner section](models/handeye.md#handeye-planner).

## Performance {#performance}

No UI-thread step over 100 ms on the samples with the shader renderer (largest: 68 ms median, 96 ms worst, for a UR5e
frame after a joint change on an integrated GPU). See [models/performance.md](models/performance.md). The Phase 3,
6 and 7 pages add no step over 100 ms (largest: 67 ms, the events after Apply on the multi-finger occlusion page;
`scripts/measure_ui_phase3.py`). The grasp study's 2 000-grasp budget is in [Grasp self-occlusion](#grasp).

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
project-obstacle distance. Phases 3, 6 and 7: 32 more, 72 in all — grasp 7, VLA tokens and policies 4,
multi-finger occlusion 3, bandwidth 3, cables 4, brackets 3, hand-eye 8 (each summarised in its section above). With
no project given, the project-dependent checks run on the bundled SO-100 and UR5e samples (the multi-finger checks
on the dexhand sample).

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
| [models/grasps.md](models/grasps.md) | grasp sets: the record, contact pads, the gripper geometry, the samplers, imports (worked example: [guide/grasps.md](guide/grasps.md)) |
| [models/grasp.md](models/grasp.md) | grasp self-occlusion: batched IK, contact visibility, the Contact Visibility Score and its intervals, the q-bucket cache, the 2 000-grasp budget (worked example: [guide/grasp.md](guide/grasp.md)) |
| [models/policies.md](models/policies.md) | policies: the fields, the cited library, the crop geometry (worked example: [guide/policies.md](guide/policies.md)) |
| [models/vla.md](models/vla.md) | VLA token resolution, crop warnings and Plücker ray maps (worked example: [guide/vla.md](guide/vla.md)) |
| [models/dexhand.md](models/dexhand.md) | the multi-finger occlusion matrix and the dexhand sample (worked example: [guide/dexhand.md](guide/dexhand.md)) |
| [models/bandwidth.md](models/bandwidth.md) | data rates, the host topology, capacities, disk and bus power (worked example: [guide/bandwidth.md](guide/bandwidth.md)) |
| [models/cables.md](models/cables.md) | cable flex: the routed arc, slack loops, bends, snags, the harness sheet (worked example: [guide/cables.md](guide/cables.md)) |
| [models/brackets.md](models/brackets.md) | brackets: the solids, the mesher, ISO 9409-1, the clearance check (worked example: [guide/brackets.md](guide/brackets.md)) |
| [models/handeye.md](models/handeye.md) | hand-eye calibration: boards, detection, solvers, the bundle adjustment, the planner and the coach (worked example: [guide/handeye.md](guide/handeye.md)) |
| [models/performance.md](models/performance.md) | the UI-thread budget: measured timings on the samples and what was changed to meet it |
