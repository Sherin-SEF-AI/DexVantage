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
measured synergies; the Shadow Hand's tendon coupling is not enforced between keyframes. See
[models/dexhand.md](models/dexhand.md).

**Visual-to-tactile handoff (§7.15, part 2 of the same study, result key `handoff`).** Along a straight approach of
each contact case (the hand in its contact posture, moved back along the approach), the share of each fingertip's
contact patch some camera sees is computed at coarse distances and refined inside every seen-to-hidden transition; at
contact, the share a tactile pad's footprint covers. The **sensory blind gap** is the distance before contact over
which a contact is neither seen nor touched (flagged above 10 mm, a default). On the dexhand sample every fingertip is
seen up to contact (gap 0 mm), and no tactile sensor is mounted, so nothing is touched. **Validation:** the gap
against a brute-force scan of every distance at the refine spacing, 0 m (2 mm). **Known limits:** the fingers do not
close during the approach; touch is judged by geometry, not force. See [models/handoff.md](models/handoff.md).

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

## MuJoCo ground truth: rendered against predicted {#groundtruth}

`core/sim/scene.py` builds one MuJoCo scene from the project (each robot from its stored files at its base pose,
objects, surfaces, material colours, default lights, one camera per imager whose intrinsics are set from its lens
model); Simulate ▸ Build the MuJoCo scene… writes it as `scene.xml` with its assets. `core/sim/render.py` renders
segmentation, depth and RGB through every lens model (pinhole directly; Brown-Conrady and Kannala-Brandt through an
oversampled base image remapped, wide lenses through cube faces) in an EGL worker process. The `groundtruth` study
(`--groundtruth`) compares the predictions with the renders: the grasp study's contact visibility (each contact point
visible in the render when the finger's segment owns its pixel within the depth tolerance) and the depth-validity
study's voxels against rendered depth with a first-order noise model. **Measured** (2026-11-07, RTX 5080): contact
agreement 92.8 % on the UR5e over 41 poses with its scripted pick (IoU 88.0 %; the grasps alone 91.1 %), 84.2 % on the
SO-100; the grasp-level verdict the same on 32 of 32 and 44
of 44 grasps; depth 99.8 % of voxels (UR5e), 99.6 % (dexhand). Most disagreements are within a pixel of a
segmentation edge or follow from the definition (the study tests 2 mm in front of the pad, the render the pad face);
occlusion disagreements away from an edge are 0.5 % (UR5e) and 0.4 % (SO-100). **Throughput:** 14.7–17.5 views/s end
to end on the RTX 5080 (76–99 renderer frames/s), 14.2 on the integrated Radeon, for the 1280 × 720 D405 and the
1920 × 1080 D435 with segmentation, depth and RGB. **Validation:** six lens models' rendered fiducials against the
lens model, 0.20 px (0.4 px); the compiled intrinsics 8.8 × 10⁻⁵ px (10⁻³); per-pixel owner against the Camera view's
renderer 2.2 × 10⁻⁵ of 2.3 M pixels (10⁻³); UR5e contacts 0.072 (0.15); UR5e depth 0.0022 (0.01). **Known limits:**
the prediction leaves out the camera's seat on its mount, MuJoCo draws smooth primitives, depth is compared at home
only, distorted lenses are slow (0.7–2.4 views/s). See [models/groundtruth.md](models/groundtruth.md).

## Synthetic LeRobot episodes {#synth_episodes}

`core/sim/episodes.py` (Simulate ▸ Generate synthetic episodes…, `--synth-episodes OUT_DIR [--synth-settings JSON]`)
moves the project's robot along its trajectories (or grasp approaches solved by IK with minimum-jerk timing), renders
every camera through its own lens model with the ground-truth renderer and writes a LeRobot v2.1 or v3.0 dataset with
seeded, known faults (dropped and duplicated frames, camera-joint desync, clock drift, jitter, exposure steps,
motion blur, a hidden gripper, mount slip, USB gaps) and their ground truth in `dexvantage_truth.json`. The auditor,
the silhouette calibration and the data links are tested on it. **Throughput:** 124–138 camera frames/s at 320 px wide
on the RTX 5080 including AV1 encoding (`--synth-episodes`, 4 SO-100 episodes of 120 rows and two cameras in 7.4 s).
**Validation:** two episodes with every fault written as v2.1 and v3.0 and read back, 0 mismatches. **Known limits:**
MuJoCo shading, no sensor noise model beyond one grey level; the gripper occluder is painted in the image; fault sizes
are defaults chosen to show in a short episode (drift 5000 ppm), not measurements. See
[models/synth_episodes.md](models/synth_episodes.md).

## Teleop overlap and dead zones {#teleop}

Per teleop device in `project.tasks.teleop_devices` (a leader arm or GELLO joint to joint, or a VR or space-mouse
volume mapped by scale and offset): a task-volume voxel is **comfortable** when the follower's IK converges with its
joints inside scale × the leader's comfortable box (the central 0.8 of each range, a default) or the mapped device
volume contains it; it is a **dead zone** when the operator is comfortable but the follower reaches badly (not at all,
near a singularity by 1/κ of the tool Jacobian, near a joint limit, close to itself). The suggestion moves the
follower's base on a lattice to maximise the teleoperable share; with two arms, the overlap and the handover quality.
The SO-100 sample carries its leader arm (the LeRobot SO-100 guide: the leader is the follower's own model): 96.7 %
teleoperable, "clear", moving the base +25 mm in y and z gives 100 %. **Validation:** w and 1/κ against the 2R
closed form 1.4 × 10⁻¹⁶ m² (10⁻¹²); against a finite-difference Jacobian of the independent chain 3.8 × 10⁻⁹ (10⁻⁶);
the leader's comfortable workspace against brute-force sampling, 0 missed (0.05); a device volume's mapping against
Monte Carlo, 0.19 in units of three standard errors (1). **Known limits:** reach is position-only with a free orientation; the
comfortable box is a default, not a human-factors model; VR and space-mouse volumes are what the user states (no maker
publishes a tracking volume). See [models/teleop.md](models/teleop.md).

## Lighting, glare and IR wash-out {#lighting}

Lights live in `project.workspace.lights` (panel, ring, window, point, spot; three Elgato entries in the light
library, every figure cited, a gap stays null). A GGX microfacet BRDF with Smith shadowing and Schlick's Fresnel per
material tag (`core/materials.py`) gives each camera's radiance at sampled object, gripper and table points, with
light and camera visibility from cube maps. Saturation is **relative** unless the user enters a sensor's sensitivity,
f-number and exposure: an auto-exposure meters the scene as 18 % grey (ISO 12232), so a point saturates at 7.86 × the
metered mean. For active-stereo sensors, ambient IR against the projector's pattern (Intel's stated optical power and
FOV) gives the share of valid depth washed out, plus dark and clear surfaces. Recommendations (a polariser, moving or
dimming a light, tilting the camera, blinding a window) each show the figure before and after; the headline's status
is the worse of glare and depth loss. **Validation:** GGX normalised 1.4 × 10⁻¹³ (10⁻⁶); weak white furnace by quadrature 8.1 × 10⁻¹²
(10⁻⁶; the Monte Carlo estimate, 0.0065 from 1, inside its 4 standard-error band); no material's directional albedo
over 1; reciprocity 1.3 × 10⁻¹⁴; the specular peak's
closed form 2.2 × 10⁻¹³; Schlick against exact Fresnel 0.036 (0.04); inverse square against Monte Carlo 0.0033
(0.005); area lights against the closed form 0.022 (0.025); visibility against exact ray casting 0.0066 of the area
(0.015). **Known limits:** direct light only, hard shadows, no measured beam profiles or IES files, the camera's
spectral response taken as photometric, no library camera states a sensitivity. See
[models/lighting.md](models/lighting.md).

## Motion blur, time sync and action-chunk latency {#timing}

**Motion blur and rolling shutter** (`--motion-blur`): along each trajectory (or grasp approach), the image velocity
of what each imager sees (points of the task volume, objects and the gripper, projected through the lens at q(t)),
the smear in one exposure and the rolling-shutter skew over the readout; without a set exposure the frame period is
an upper bound, labelled. On the UR5e sample's scripted pick: up to 38.04 px on the wrist D405, "may be blurred" (an
upper bound); at most
0.88 ms keeps it to 1 px. **Time sync** (`--time-sync`, `project.rig['sync']`): each stream's stamp error interval
(exposure centre, readout, USB transfer behind the other frames on its controller, the bus interval) by clock and
trigger, the skew against the joint states and the action-label error it causes (the TCP's motion in that time); a
term no maker states makes the figure a lower bound (≥). UR5e: ≥ 54.43 ms, "out of sync", 9.2 mm of label error; the
suggestion is the D435's hardware trigger. **Action-chunk latency** (`--action-chunk`, §7.13): the policy (from
`core/policies.resolve`, cited inference time and horizon) as an ideal imitator; the sensor-to-action loop (half the
exposure, readout, transfer, queueing, inference, the control tick, command latency) gives when the gripper closes
relative to the demonstrated contact, the overshoot or air-grasp distance and an overshoot map over the task volume,
and recommends a delay offset and a temporal-ensemble weight (stored on the policy for the LeRobot exporter). A
policy whose inference time no source states is refused in one sentence. **Validation:** image velocity against
central differences 2.8 × 10⁻⁹ (10⁻⁵); smear analytic 2.4 × 10⁻¹¹; rolling shutter against a row-by-row readout
0.024 (0.05); stamp intervals against a frame-timeline simulation 3.5 × 10⁻¹⁸ s (10⁻⁹); action-label error against
forward kinematics at random phases 2.2 × 10⁻⁴ (0.02); the loop against an event simulation (20 000 trials)
2.9 × 10⁻⁵ m (5 × 10⁻⁵); the ensemble weights against the closed form 3.3 × 10⁻¹³. **Known limits:** no library entry
states a readout time or a device's processing latency (bounds until measured); first order in the exposure; the
policy's learned errors are not modelled. See [models/motion_blur.md](models/motion_blur.md),
[models/time_sync.md](models/time_sync.md) and [models/action_chunk.md](models/action_chunk.md).

## Dataset readers, the auditor and plan against recording {#data}

Readers turn LeRobot v2.1 and v3.0 (`core/datasets/lerobot.py`, also a writer; the format read from the LeRobot
source at commit d490e47 and tag v0.3.3), ROS 2 bags in MCAP or SQLite (`core/datasets/bag.py` over
`core/rosbag.py`, ported from Vantage) and RLDS TFRecords (`core/datasets/rlds.py`) into one Episode shape. The
**auditor** (`--audit OUT.json --dataset PATH`, Data ▸ Audit a dataset…) checks each episode from the data alone:
dropped and duplicated frames, frame rate, jitter, USB gaps, camera-joint offset (cross-correlation of image motion
with the plan's predicted motion), clock drift, exposure steps, motion blur, the gripper hidden in the wrist camera,
and mount slip across the campaign (step against gradual flex, ported from Vantage `core/online_calib.py`), with a
verdict per episode (pass, fix, recalibrate, drop), a JSON and HTML report and a FiftyOne export with per-frame tags.
**Plan against recording** (`--plan-vs-recording`) compares each recorded camera with the plan: resolution, frame
rate, mount, sync and the grasp study's visibility against what the frames show. **Measured** on a seeded synthetic
SO-100 campaign (6 episodes × 240 rows, every fault in 2 episodes, audited without its sidecar): recall 1.0 with no
false positives for drops, USB gaps, jitter, exposure steps, the hidden gripper and mount slip; duplicates and desync
0.5; blur 1.0 with a false-positive rate of 0.1; in-episode drift at 5000 ppm not resolved (reported, not claimed).
**Validation:** recall and false-positive floors per fault kind, 0 shortfall; desync on static cameras 16.9 ms (20
ms); the slip classifier against Vantage's recorded outputs 0 (10⁻⁹); the plan's mount difference against the true
slip 0.20° (0.3°), the rotation vectors compared; LeRobot v2.1 and v3.0 round trip read independently with pyarrow and
ffmpeg, 0 mismatches; MCAP
round trip and TFRecord CRC-32C, 0. **Known limits:** LeRobot's own loader and FiftyOne's importer have not been run
(neither is installed); desync on cameras on the robot is poorly resolved on textureless scenes; drift shows across a
campaign, not within an episode. See [models/datasets.md](models/datasets.md) and [models/audit.md](models/audit.md).

## Targetless link-silhouette calibration {#render_calib}

A static camera's mount is estimated from the robot's own silhouette in recorded frames (`core/calib/render_calib.py`,
`--silhouette-calib OUT.json --dataset PATH`, Calibrate ▸ Silhouette calibration): the robot is rendered at the logged
joint states through the camera's lens model, and the mount is fitted coarse to fine to the frames' edges (a distance
transform) and a median-background foreground mask, with a bootstrap σ and an observability test; run per episode it
shows the mount's drift. **Measured** on synthetic episodes (640 × 360, AV1, a 5 mm and 1° slip): the UR5e overhead
D435 within 1.3 mm and 0.03°; the SO-100 third-person camera 2.4 mm and 0.30° (the robot covers 4 % of the image),
the slip between episodes found to about 1 mm. **Validation:** recovery of the slip 1.66 px of the working image (2
px); against the board-based bundle adjustment of the same rig 1.65 px (2 px). **Known limits:** the robot must look
like its model; intrinsics and joint times are taken as exact (a desync biases the mount); σ can understate the error
along the lateral-shift-against-yaw direction for a small robot. See [models/render_calib.md](models/render_calib.md).

## Data links: grasps and hand-eye captures from datasets {#datalinks}

`core/datasets/joints.py` matches a recording's joints to the robot model (exact name, the name without LeRobot's
prefixes and suffixes, then by order when the counts agree). Tasks ▸ From a dataset… makes a grasp set from each
time the gripper closes in an episode (`core/grasps.from_dataset`); Hand-eye solve ▸ From a LeRobot dataset or ROS 2
bag… makes a capture session from the frames with the board in view (`core/calib/capture.capture_dataset`).
**Validation:** grasps at a scripted pick's closings 6.2 × 10⁻⁸ m (10⁻⁶); a hand-eye solve from a rendered dataset
0.046 mm from the true mount (0.2 mm). **Known limits:** no clock offset is estimated (only a given one applied); a
dataset's own intrinsics are compared, never used. See [models/datalinks.md](models/datalinks.md).

## Performance {#performance}

No UI-thread step over 100 ms on the samples with the shader renderer (largest: 68 ms median, 96 ms worst, for a UR5e
frame after a joint change on an integrated GPU). See [models/performance.md](models/performance.md). The Phase 3,
6 and 7 pages add no step over 100 ms (largest: 67 ms, the events after Apply on the multi-finger occlusion page;
`scripts/measure_ui_phase3.py`). The grasp study's 2 000-grasp budget is in [Grasp self-occlusion](#grasp). The
Phase 4, 5 and 8 pages add none either (largest: 80.8 ms, the events after Apply on the multi-finger page with the
handoff; every new page's Apply under 30 ms; `scripts/measure_ui_phase458.py`).

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
multi-finger occlusion 3, bandwidth 3, cables 4, brackets 3, hand-eye 8 (each summarised in its section above).
Phases 4, 5 and 8: 39 more, 113 in all — MuJoCo ground truth 6 (with the synthetic-episode round trip), teleop 4,
lighting 9, timing 7, the tactile handoff 1, data 4, dataset formats 4 (with the MJCF scene loaded by MuJoCo on its
own), data links 2, the silhouette calibration 2.
With no project given, the project-dependent checks run on the bundled SO-100 and UR5e samples (the multi-finger
checks on the dexhand sample). The self-test runs the quick run's cheap checks (`cheap_only`); `--validate` runs them
all (full run 258 s, peak memory 4.8 GB).

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
| [models/swept.md](models/swept.md) | the ghosted swept volume of a trajectory on the Tasks page |
| [models/handoff.md](models/handoff.md) | the visual-to-tactile handoff and the sensory blind gap (worked example: [guide/handoff.md](guide/handoff.md)) |
| [models/groundtruth.md](models/groundtruth.md) | the MuJoCo scene, renders through lens models, rendered against predicted visibility and depth (worked example: [guide/groundtruth.md](guide/groundtruth.md)) |
| [models/synth_episodes.md](models/synth_episodes.md) | synthetic LeRobot episodes with known faults and their ground truth (worked example: [guide/synth_episodes.md](guide/synth_episodes.md)) |
| [models/teleop.md](models/teleop.md) | teleop devices, the comfortable workspace, dead zones, bimanual overlap (worked example: [guide/teleop.md](guide/teleop.md)) |
| [models/lighting.md](models/lighting.md) | light sources, the microfacet materials, relative and absolute saturation, IR wash-out (worked example: [guide/lighting.md](guide/lighting.md)) |
| [models/motion_blur.md](models/motion_blur.md) | image velocity, smear and rolling-shutter skew (worked example: [guide/motion_blur.md](guide/motion_blur.md)) |
| [models/time_sync.md](models/time_sync.md) | clocks, triggers, stamp error intervals and the action-label error (worked example: [guide/time_sync.md](guide/time_sync.md)) |
| [models/action_chunk.md](models/action_chunk.md) | the sensor-to-action loop of a chunked policy, overshoot and air-grasp, delay offset and ensemble weight (worked example: [guide/action_chunk.md](guide/action_chunk.md)) |
| [models/datasets.md](models/datasets.md) | LeRobot v2.1 and v3.0 read and write, ROS 2 bags, RLDS, the Episode shape (worked examples: [guide/datasets.md](guide/datasets.md), [guide/bags_rlds.md](guide/bags_rlds.md)) |
| [models/audit.md](models/audit.md) | the dataset auditor, plan against recording, the FiftyOne export (worked example: [guide/audit.md](guide/audit.md)) |
| [models/render_calib.md](models/render_calib.md) | targetless link-silhouette calibration and mount drift (worked example: [guide/render_calib.md](guide/render_calib.md)) |
| [models/datalinks.md](models/datalinks.md) | joint matching, grasp sets and hand-eye captures from recorded datasets (worked examples: [guide/grasps.md](guide/grasps.md), [guide/handeye.md](guide/handeye.md)) |
| [models/performance.md](models/performance.md) | the UI-thread budget: measured timings on the samples and what was changed to meet it |
