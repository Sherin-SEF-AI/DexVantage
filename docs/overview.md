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

**References (checks):** `mujoco_xml`, `rep103`, `scipy_rotation`, `shepperd1978`, `shoemake1985`, `urdf` ([full entries](#references)).

## The project summary and the dense-voxel limit {#summary}

A dense axis-aligned grid over a box of edges (a, b, c) at voxel size s has N = ⌈a/s⌉·⌈b/s⌉·⌈c/s⌉ voxels. At 1 mm,
1.5 × 1.5 × 1.2 m needs 2.7 × 10⁹ voxels, so the coverage studies (Phase 2) use 5 mm by default and refine to 1–2 mm
only inside refinement regions. The summary reports N at the chosen size and at 1 mm, and marks a dense grid above
5 × 10⁷ voxels as not feasible (the limit is a documented default of this build, not a standard).

## Robots {#robots}

A URDF or MJCF robot is read with every file it names into the project and compiled by MuJoCo from memory; joint
limits and the home pose are the file's own. The UR5e + Robotiq 2F-85, SO-100, UR5e + Shadow Hand, ALOHA 2, bimanual
Panda and G1 samples are bundled MuJoCo Menagerie models, after their licences were read ([the new samples](#samples)).
See [models/robots.md](models/robots.md) (worked example: [guide/robots.md](guide/robots.md)).

## Robot kinematics {#kinematics}

MuJoCo poses the links: T_link(q) from its tree of body transforms and joint motions, and the world Jacobian of a
point from `mj_jac`. An independent product-of-exponentials chain checks it,

    T(q) = e^[S₁]θ₁ · e^[S₂]θ₂ ⋯ e^[Sₙ]θₙ · M        [Lynch and Park 2017]

with its own XML reader and twist exponentials (`core/robot/kinematics.py`), and both engines are checked against
the closed-form DH tables of the UR5e and the Panda: see [models/kinematics.md](models/kinematics.md).

**Validation:** 11 checks (`checks_kinematics.py`, `checks_ik.py`, `checks_obstacles.py`). Measured in the validation
run of 2026-11-08 (239 of 239 checks pass): MuJoCo's forward kinematics against the product-of-exponentials chain 8.7 ×
10⁻¹⁵ m (10⁻⁹ m); the UR5e and Panda DH tables 5.6 × 10⁻¹⁶ and 6.3 × 10⁻¹⁶ m (10⁻⁹ m); the analytic Jacobian against
finite differences 4.4 × 10⁻¹⁰ m/rad (10⁻⁷) and against MuJoCo's `mj_jac` 8.8 × 10⁻¹⁵ m/rad (10⁻⁹); IK round trips 99.7
µm and, position only, 93.0 µm (100 µm); the 2R manipulability closed form 1.1 × 10⁻¹⁶ m² (10⁻¹²); sphere and capsule
distances 1.1 × 10⁻¹⁶ m (10⁻⁹ m); a moved and rotated obstacle 2.6 × 10⁻⁷ m (10⁻⁶ m). **Known limits:** free and ball
joints are held at home and are not part of q; URDF mimic joints move independently; URDF planar joints and MJCF
`<replicate>`, `<attach>`, `<composite>` and `<flexcomp>` are refused in a plain sentence; the DH checks validate the
engines' conventions, not a vendor file's geometry (the bundled Menagerie UR5e's flange is up to 1.3 mm from the DH
flange).

**References (checks):** `craig2005`, `dh1955`, `ericson2005`, `franka_dh`, `liegeois1977`, `lynch_park2017`, `mujoco_geomdistance`, `mujoco_todorov2012`, `mujoco_xml`, `nakamura1986`, `spong2006`, `sugihara2011`, `ur_dh`, `urdf`, `wampler1986`, `yoshikawa1985` ([full entries](#references)).

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
90° × 60°). 40 entries in Phase 2; 138 hardware entries since Phases 9 and 10 (33
cameras, 33 depth cameras, 10 tactile sensors, 18 hosts and capture cards, 19 policies, 6 lights, 6 cables, 13
teleop devices in the new `teleop` kind) and 53 robots in the `robot` kind ([the robot catalogue](#catalogue)).
**Equations:** the sensor geometry check uses w = p·W, h = p·H, d = p·√(W² + H²) (pixel pitch p, native resolution W ×
H) and the pinhole FOV 2·atan(p·N / 2f) of a stated focal length f; a ViT policy input of w × h px with patch p and
token merge s gives (w/p)(h/p)/s² tokens. **Validation:** 8 checks (`checks_library.py`, `checks_library_systems.py`,
`checks_policies.py`, `checks_prices.py`). Measured in the validation run of 2026-11-08 (239 of 239 checks pass): pixel
pitch × resolution against the stated active area 1.2 % (2 %: sizes are rounded to 0.1 mm); stated FOVs within the
bounds of a radially symmetric lens, 2.92° at most (3°), and against the pinhole FOV of the stated focal length 0.48°
(3°); stated totals against count × the per-port figure 0.09 % (0.5 %); token grids exact; teleop blocks, prices and the
rig cost 0 problems. **Known limits:** figures are as stated on the date read (a maker can revise them); a gap stays
null, so a study cannot use that figure until the user measures it; an FOV for a stream the maker does not state is a
centred crop, an assumption the imager says. See
[models/library.md](models/library.md) (worked examples: [guide/library.md](guide/library.md),
[guide/library_systems.md](guide/library_systems.md)).

**References (checks):** `dosovitskiy2021`, `hartley2004`, `intel_d400`, `iso4217`, `kannala_brandt2006`, `library_host_sources`, `library_policy_sources`, `library_teleop_sources`, `nist_sp811` ([full entries](#references)).

## Lens models, pixel density and defocus {#optics}

`CameraModel` (pixels, OpenCV conventions: pixel centres at integers): pinhole; Brown-Conrady with 5 or 8 terms
[Brown 1966]; Kannala-Brandt, OpenCV's fisheye θd = θ(1 + k₁θ² + k₂θ⁴ + k₃θ⁶ + k₄θ⁸) [Kannala and Brandt 2006];
equidistant, equisolid and stereographic. Every model has a valid angle beyond which it is not one-to-one, and
points past it are never projected. Pixel density at a point is √|det J| of the image Jacobian over a patch facing the
camera (f/Z on a pinhole's axis; × √cos α for a tilted surface), and defocus is the thin-lens blur disc
b = A f |s − z| / (z (s − f)) [Ray 2002]. Ported from Vantage `core/lens.py`. See
[models/optics.md](models/optics.md).

**Validation:** 10 checks (`checks_optics.py`, `checks_optics_mujoco.py`). Measured in the validation run of 2026-11-08
(239 of 239 checks pass): against OpenCV's `projectPoints` 1.8 × 10⁻¹² px and `fisheye.projectPoints` 5.7 × 10⁻¹³ px
(10⁻⁶ px); project(unproject(pixel)) 4.7 × 10⁻¹³ px; FOV fits 2.3 × 10⁻¹² °; pixel density against √|det J| 8.2 × 10⁻⁹
relative (10⁻⁶) and the coverage study's figure against `cv2.projectPoints` differences 0.0065 (0.01); depth of field
and the blur disc 2.9 × 10⁻¹³ and 1.1 × 10⁻¹³ relative; MuJoCo-rendered fiducials 0.075 px for mask centroids (0.1 px)
and 0.17 px for centres (0.3 px). **Known limits:** thin lens only (no pupil magnification, diffraction, aberrations or
focus breathing); no skew, thin-prism or tilt terms; a model fitted from a stated FOV reproduces the stated angles, and
its distortion between them is a model until a calibration replaces it; the MuJoCo check covers pinhole cameras only
(MuJoCo renders no distortion).

**References (checks):** `brown1966`, `hartley2004`, `hecht2017`, `kannala_brandt2006`, `miyamoto1964`, `mujoco_camera`, `mujoco_renderer`, `opencv_calib3d`, `opencv_fisheye`, `ray2002` ([full entries](#references)).

## Occlusion by the posed robot {#occlusion}

Around each imager's optical centre, a cube map of distances to the nearest occluder (the robots' visual meshes
posed at q, the task objects, the tables): six 90° faces of n × n texels (default 512), built by a z-buffer on the GPU
(EGL) or by an exact numpy rasteriser (the ray through each texel centre against each triangle; the depth is the
exact plane distance along that ray). A point is visible when its distance is at most its texel's depth plus 1 mm (a
default); near-plane clipping at 5 mm keeps a sensor from being blinded by its own seat. Exact ray casting
(Möller-Trumbore over a uniform grid) is the independent reference. Ported from Vantage `core/occlusion.py` (the
cube-map idea and face basis). See [models/occlusion.md](models/occlusion.md).

**Equations:** with the cube map centred at the optical centre c, a point p is visible when |p − c| ≤ D(texel(p)) + ε (ε
= 1 mm, a default), D the texel's stored distance; a texel ray d meets the triangle (a₀, a₁, a₂) (c at the origin) at t
= (a₀ · n) / (n · d), n = (a₁ − a₀) × (a₂ − a₀), kept when t ≥ the 5 mm near plane [Möller and Trumbore 1997].
**Validation:** 7 checks (`checks_occlusion.py`, `checks_occlusion_mujoco.py`, `checks_camview.py`,
`checks_coverage.py`). Measured in the validation run of 2026-11-08 (239 of 239 checks pass): texels whose distance
differs from the exact first hit by more than 1 mm 4.2 × 10⁻⁶ (10⁻³); against exact ray casting 2.5 × 10⁻⁵ of random
points (10⁻³) and 0.18 % on the SO-100 and UR5e samples (1.45 %, the measured share of points within half a texel
diagonal of a silhouette); MuJoCo segmentation of a box scene 0 and of the UR5e 2.5 × 10⁻⁵ of the target pixels away
from silhouettes (10⁻³); the Camera view 1.4 × 10⁻⁵ (10⁻³); the coverage shares 8.8 × 10⁻⁴ of the volume (5 × 10⁻³).
**Known limits:** one centre per map (a stereo pair needs one per imager); visibility is decided per texel (0.22° at n =
512), so a point within about a texel of a silhouette may go either way; MuJoCo's equality constraints (the 2F-85
linkage) are not applied; cables, cloth and sensor housings do not occlude.

**References (checks):** `amanatides_woo1987`, `greene1986`, `moller1997`, `mujoco_camera`, `mujoco_renderer`, `mujoco_todorov2012` ([full entries](#references)).

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

**Validation:** 4 checks (`checks_depth.py`). Measured in the validation run of 2026-11-08 (239 of 239 checks pass): σ_z
against a Monte Carlo of disparity noise 0.70 % relative (1.83 %, the Monte Carlo's band); the library's min-Z and FOV
figures against Intel's D400 tables, 0 differ; the stereo-overlap formula against brute force 9.4 × 10⁻⁴ ° (0.01°);
finger shadows against ray casting from both imagers 0 (10⁻³). **Known limits:** the voxel centre stands for the voxel;
the projector's own shadow is not modelled (its position in a D4xx is a labelled assumption); Intel states the depth FOV
at one distance, so the modelled overlap is narrower by 0.7–3.6°; min-Z is measured from the optical centre, not the
glass; materials are tags, not optics; σ_z is first order and leaves out calibration error and matcher bias.

**References (checks):** `amanatides_woo1987`, `gallup2008`, `greene1986`, `hartley2004`, `intel_d400`, `intel_d457`, `moller1997` ([full entries](#references)).

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
time) complete the result. **Performance (§16):** measured 2026-11-06, 2 000 grasps on the UR5e sample in 111.6 s on an NVIDIA
GeForce RTX 5080 (EGL device 0) and 131.3 s on the Ryzen 9 9900X's integrated Radeon, 17.9 and 15.2 grasps/s, 10 808
cube maps; 85.5 s with 256-texel maps, 83.6 s without the sensitivity sweep. Measured again at the Phase 11 gate
(2026-11-08, RTX 5080, a loaded machine) on every sample: 114.2 s on the UR5e and 182.1 s on ALOHA 2, which misses
this project's 180 s default ([performance.md](performance.md)). **Validation:** visibility against exact ray casting
of the full scene, 0 of 7 892 points away from an edge (10⁻³); the bootstrap's coverage against the known p of
binomial samples, within 0.017 of 95 % (0.031); the q-bucket against evaluation at each grasp's own q, mean 0.0024 of
a patch (0.02). **Known limits:** the fingers stay at their home opening; obstacles are not checked along the approach;
the 2F-85's linkage joints are independent entries of q. See [models/grasp.md](models/grasp.md).

**References (checks):** `amanatides_woo1987`, `efron1979`, `greene1986`, `lerobot_dataset_format`, `moller1997`, `mujoco_todorov2012`, `mujoco_xml`, `rep103`, `scipy_rotation`, `wilson1927` ([full entries](#references)).

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

**References (checks):** `amanatides_woo1987`, `brown1966`, `hartley2004`, `irani_anandan1996`, `isaac_gr00t`, `kannala_brandt2006`, `lerobot`, `moller1997`, `opencv_calib3d`, `opencv_fisheye`, `opencv_remap`, `opencv_resize`, `opencv_warp_perspective`, `openpi`, `plucker1865`, `safetensors`, `torchvision_centercrop` ([full entries](#references)).

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

**References (checks):** `greene1986`, `hartley2004`, `moller1997` ([full entries](#references)).

## Bandwidth, hosts and power {#bandwidth}

Each stream's payload is width × height × fps × bits per pixel / 10⁶ Mbit/s from the library's stream modes, on the
wire × (1 + overhead), the overhead a labelled default of 5 % per interface; a stream whose maker states no bits per
pixel has no rate until a measured one is entered. The host topology (`project.rig`: hosts, controllers, hubs, which
sensor plugs where) gives each controller and link its load against its capacity (USB 2.0: 53.248 MB/s from Table 5-10;
USB 3.2: 450 MB/s and 1.1 GB/s from §4.4.11, doubled for x2), the margin and flags (a controller over budget, a hub on
a shared controller); disk per episode and campaign; bus power against USB 3.2 Table 3-1 (900 mA) and USB 2.0 §7.2.1
(500 mA). Ported from Vantage `core/recording.py` and `core/power.py`. **Equations:** payload = W × H × fps × bits per
pixel / 10⁶ Mbit/s; on wire = payload × (1 + overhead); load = Σ on wire / capacity per controller and link; margin =
capacity − Σ on wire; recorded = Σ payload × record ratio × (1 + container overhead). **Validation:** the NanEyeC's
stated serial
rates bound its payload (79 %); the USB figures against their derivations, 0; the budget re-computed independently
over 40 random topologies, 3.9 × 10⁻¹⁶. **Known limits:** GMSL link rates and deserialiser power are not stated in the
library; USB Type-C current levels and Power Delivery are not modelled. See [models/bandwidth.md](models/bandwidth.md).

**References (checks):** `naneyec`, `usb20_spec`, `usb32_spec` ([full entries](#references)).

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

**References (checks):** `amanatides_woo1987`, `docarmo1976`, `moller1997`, `scipy_brentq` ([full entries](#references)).

## Brackets and mounts {#brackets}

Watertight, printable STL from signed-distance solids meshed by marching tetrahedra (ported from Vantage
`core/meshing.py` and `core/brackets.py`): wrist-flange adapters (ISO 9409-1-50-4-M6 from UR's drawing; other sizes
from the designation), 2020 and 4040 T-slot clamps, a pan-tilt neck plate, palm and dorsal sleeves for micro cameras;
the camera's stated mounting holes where its datasheet gives them (RealSense D405, D435, D435i). Clearance: the
bracket and camera body against every link but the mount link at home, on screen and 48 seeded joint states, and
against cable paths when there are cables. **Equations:** the mesh volume V = Σ a · (b × c) / 6 over the triangles (a,
b, c) with outward normals and the Euler characteristic V − E + F = 2 − 2g per shell (g the through holes); the flange
disc in closed form π t (R² − r_c² − 4 r_b² − r_p²) + π h (r_s² − r_c²) − π h (r_r² − r_c²). **Validation:** edges,
winding, Euler characteristic and volume written out
again, 0 defects on all six kinds; the flange disc's volume against its closed form, 0.18 % (1 %); hole centres fitted
on the mesh against the pattern, 4.2 × 10⁻⁵ mm (0.05 mm). **Known limits:** STEP export is not done (STL with a JSON
sidecar); other ISO 9409-1 sizes lack centring and dowel figures. See [models/brackets.md](models/brackets.md).

**References (checks):** `euler_poincare`, `iso9409_1`, `treece1999`, `ur5e_flange` ([full entries](#references)).

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
180° motions; time offsets between images and joint readings are not modelled. See
[models/handeye.md](models/handeye.md)
and its [planner section](models/handeye.md#handeye-planner).

**References (checks):** `andreff1999`, `barrow1977`, `borgefors1988`, `canny1986`, `daniilidis1999`, `efron1979`, `garrido2014`, `hartley2004`, `horaud_dornaika1995`, `kalibr`, `lerobot_dataset_format`, `li2010`, `liang_zeger1986`, `liu2010_fdcm`, `mujoco_camera_317`, `opencv_calib3d`, `opencv_charuco`, `opencv_distance_transform`, `opencv_handeye`, `park_martin1994`, `shah2013`, `triggs2000`, `tsai_lenz1989`, `wang_olson2016` ([full entries](#references)).

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
1920 × 1080 D435 with segmentation, depth and RGB. **Equations:** each MuJoCo camera is compiled from the imager's lens
model with resolution = (W, H), focalpixel = (fx, fy) and principalpixel = ((W − 1)/2 − cx, (H − 1)/2 − cy); a contact
point counts as visible in the render when its pixel belongs to the finger's segment and |z_render − z_point| is within
the depth tolerance. **Validation:** six lens models' rendered fiducials against the
lens model, 0.20 px (0.4 px); the compiled intrinsics 8.8 × 10⁻⁵ px (10⁻³); per-pixel owner against the Camera view's
renderer 2.2 × 10⁻⁵ of 2.3 M pixels (10⁻³); UR5e contacts 0.072 (0.15); UR5e depth 0.0022 (0.01). **Known limits:**
the prediction leaves out the camera's seat on its mount, MuJoCo draws smooth primitives, depth is compared at home
only, distorted lenses are slow (0.7–2.4 views/s). See [models/groundtruth.md](models/groundtruth.md).

**References (checks):** `intel_d400`, `kannala_brandt2006`, `lerobot_dataset_format`, `mujoco_camera_317`, `mujoco_renderer`, `opencv_remap` ([full entries](#references)).

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
teleoperable, "clear", moving the base +25 mm in y and z gives 100 %. **Equations:** the leader's comfortable workspace
W = { FK_f(q) : q ∈ scale · C }, C the central box of the leader's ranges; a device volume maps by p_world = B_f · O ·
(s · p_d); manipulability w = √det(J Jᵀ) [Yoshikawa 1985] and the inverse condition number 1/κ = σ_min / σ_max of the
tool Jacobian. **Validation:** w and 1/κ against the 2R
closed form 1.4 × 10⁻¹⁶ m² (10⁻¹²); against a finite-difference Jacobian of the independent chain 3.8 × 10⁻⁹ (10⁻⁶);
the leader's comfortable workspace against brute-force sampling, 0 missed (0.05); a device volume's mapping against
Monte Carlo, 0.19 in units of three standard errors (1). **Known limits:** reach is position-only with a free
orientation; the
comfortable box is a default, not a human-factors model; VR and space-mouse volumes are what the user states (no maker
publishes a tracking volume). See [models/teleop.md](models/teleop.md).

**References (checks):** `lerobot_so100`, `lynch_park2017`, `yoshikawa1985` ([full entries](#references)).

## Lighting, glare and IR wash-out {#lighting}

Lights live in `project.workspace.lights` (panel, ring, window, point, spot; three Elgato entries in the light
library, every figure cited, a gap stays null). A GGX microfacet BRDF with Smith shadowing and Schlick's Fresnel per
material tag (`core/materials.py`) gives each camera's radiance at sampled object, gripper and table points, with
light and camera visibility from cube maps. Saturation is **relative** unless the user enters a sensor's sensitivity,
f-number and exposure: an auto-exposure meters the scene as 18 % grey (ISO 12232), so a point saturates at 7.86 × the
metered mean. For active-stereo sensors, ambient IR against the projector's pattern (Intel's stated optical power and
FOV) gives the share of valid depth washed out, plus dark and clear surfaces. Recommendations (a polariser, moving or
dimming a light, tilting the camera, blinding a window) each show the figure before and after; the headline's status
is the worse of glare and depth loss. **Equations:** irradiance E = I(θ) max(0, n · ωᵢ) / r² [McCluney 2014]; reflected
radiance from f_r = f_d + f_s with f_s = F(ωᵢ · h) G(ωᵢ, ωₒ) D(h) / (4 |n · ωᵢ| |n · ωₒ|), D(h) = α² / (π cos⁴θ_h (α² +
tan²θ_h)²) (GGX [Walter et al. 2007]), G Smith's shadowing, F = F₀ + (1 − F₀)(1 − cos θ)⁵ [Schlick 1994].
**Validation:** GGX normalised 1.4 × 10⁻¹³ (10⁻⁶); weak white furnace by quadrature 8.1 × 10⁻¹²
(10⁻⁶; the Monte Carlo estimate, 0.0065 from 1, inside its 4 standard-error band); no material's directional albedo
over 1; reciprocity 1.3 × 10⁻¹⁴; the specular peak's
closed form 2.2 × 10⁻¹³; Schlick against exact Fresnel 0.036 (0.04); inverse square against Monte Carlo 0.0033
(0.005); area lights against the closed form 0.022 (0.025); visibility against exact ray casting 0.0066 of the area
(0.015). **Known limits:** direct light only, hard shadows, no measured beam profiles or IES files, the camera's
spectral response taken as photometric, no library camera states a sensitivity. See
[models/lighting.md](models/lighting.md).

**References (checks):** `amanatides_woo1987`, `ashikhmin2000`, `hecht2017`, `heitz2014`, `howell_catalog`, `mcluney2014`, `mcpeak2015`, `moller1997`, `schlick1994`, `schott_bk7`, `scipy_quad`, `sultanova2009`, `veach1997`, `walter2007`, `williams1978` ([full entries](#references)).

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

**References (checks):** `act_code`, `hartley2004`, `kannala_brandt2006`, `lerobot_act_ensembler`, `liang2008`, `lynch_park2017`, `ms_usb_isoch`, `mujoco_todorov2012`, `usb20_spec`, `usb32_spec`, `zhao2023_act` ([full entries](#references)).

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
**Equations:** with P = 1 / the stated fps, a spacing d ≥ 1.5 P is a gap of round(d/P) − 1 frames; the stamps are fitted
t_k = a + P′ n_k by least squares (measured rate 1/P′, jitter 1.4826 · MAD of the residuals); in-campaign drift δ(t) =
δ₀ + d (t − t̄). **Validation:** recall and false-positive floors per fault kind, 0 shortfall; desync on static cameras
16.9 ms (20
ms); the slip classifier against Vantage's recorded outputs 0 (10⁻⁹); the plan's mount difference against the true
slip 0.20° (0.3°), the rotation vectors compared; LeRobot v2.1 and v3.0 round trip read independently with pyarrow and
ffmpeg, 0 mismatches; MCAP
round trip and TFRecord CRC-32C, 0. **Known limits:** LeRobot's own loader and FiftyOne's importer have not been run
(neither is installed); desync on cameras on the robot is poorly resolved on textureless scenes; drift shows across a
campaign, not within an episode. See [models/datasets.md](models/datasets.md) and [models/audit.md](models/audit.md).

**References (checks):** `baker_matthews2004`, `huber1964`, `kass_raftery1995`, `knapp_carter1976`, `kunsch1989`, `lerobot_dataset_format`, `lucas_kanade1981`, `pech_pacheco2000`, `spearman1904` ([full entries](#references)).

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

## Metrics, requirements and the readiness scorecard {#requirements}

`core/metric_defs.py` names the figures of merit of every study (`<area>.<figure>`, e.g. `grasp.contact_visibility`,
`sync.skew_ms`, `canonical_view.parallax_p95`): each reads a number the study already computed, `None` ("not stated")
with the study's reason when it does not state it, and `at_least` / `at_most` when the study states only a bound.
A requirement is a metric, an operator and a threshold (`project.requirements`, or a YAML file); the scorecard
evaluates them on the active variant with a signed margin (pass, fail, not evaluated: a bound decides only in its
direction), and `--check REQUIREMENTS.yaml` is the CI gate (exit 0 pass, 1 fail or not evaluated, 2 error;
`defaults` checks the labelled defaults). The variants compare evaluates every requirement on every variant;
`--reproduce` re-runs any result from its provenance and compares every number; `--run-plan` runs studies, checks,
exports and reproductions in order. Ported from Vantage `core/requirements.py` (the margin, the verdict, the gate),
without its priorities. On the UR5e sample the labelled defaults fail (camera to joint-state skew at least 54.4 ms
against 2 ms; cable pulls and snags; the blind gap); its scorecard reproduces exactly. **Equations:** margin m = v − t
for ≥ and >, m = t − v for ≤ and <, m = −|v − t| for ==; a requirement passes when m ≥ −ε (≥, ≤, ==) or m > ε (>, <), ε
= 10⁻⁹ · max(1, |t|). **Validation:** 8 exit-code
cases written out from §11, 0 wrong; bound decisions against a Monte Carlo over the true figures each bound allows,
0 of 4 000 disagree; 14 metric figures against the results read by hand, 0 difference; reproduction round trip and
the detection of a 10⁻⁶ change, 0 missed. **Known limits:** no priorities (every requirement decides); thresholds
compare with the study's rounded figure; `rig.cost` reads "not stated" until a library entry states a price (none
does). See [models/requirements.md](models/requirements.md) (worked example:
[guide/requirements.md](guide/requirements.md)).

**References (checks):** `fips180_4`, `metropolis_ulam1949`, `posix_exit`, `pyyaml`, `rfc8259` ([full entries](#references)).

## Monte Carlo tolerances and the perturbation export {#tolerance}

Ported from Vantage `core/tolerance.py`. Each mount's tolerance is a ± bound per axis (shifts along the mount parent
frame's axes, turns about the sensor's own x, y, z), drawn uniformly (a labelled default) or as a normal truncated at
2σ; a sensor with an applied hand-eye result draws from its 6 × 6 calibration covariance instead, converted exactly
into mount changes. Each draw moves the project through `core/factors.py` and re-evaluates the named metrics, giving
each metric's distribution, the share of draws that meet each requirement (with its Wilson interval) and the
components that drive it (rank correlation). `--tolerance-mc OUT.json` (`--tolerance` is `--reproduce`'s option);
the perturbation export (§8.5, format `perturbations`) writes the same draws as per-sensor extrinsic deltas for
augmentation and Isaac randomisation. On the UR5e sample (64 draws, the labelled default metrics) the worst motion
smear reads 38.6 px, "no requirement". **Equations:** a draw moves the mount by p′ = p + d (d along the parent frame's
axes) and turns it by R′ = R · R_x(δroll) · R_y(δpitch) · R_z(δyaw); a uniform draw is U(−b, b), a normal one N(0,
(b/2)²) truncated at ±b (standard deviation 0.4398 b). **Validation:** the Monte Carlo σ of a linear metric against the
GUM law of
propagation, 0.0060 relative (4 standard errors, 0.020); 30 sampler marginals against `scipy.stats.kstest`, largest
D 0.0235 (critical 0.037); the seed reproduces exactly; Spearman against SciPy 5.6 × 10⁻¹⁷; the export against the
moved project 1.5 × 10⁻⁹ m or rad (10⁻⁸). **Known limits:** components and sensors are independent (a shared bracket
that bends is not modelled); intrinsics, joint encoders and link lengths are not drawn; n is bounded by the slowest
study. See [models/tolerance.md](models/tolerance.md) (worked example: [guide/tolerance.md](guide/tolerance.md)).

**References (checks):** `isaaclab_camera_cfg`, `jcgm100_2008`, `jcgm101_2008`, `kolmogorov1933`, `scipy_rotation`, `scipy_stats`, `smirnov1948`, `spearman1904` ([full entries](#references)).

## Designed experiments {#experiments}

Ported from Vantage `core/experiments.py`. Factors are `core/factors.py` paths (a mount coordinate, a device, a
stream, a study setting); designs are full factorial, two-level fractional factorial at resolution III or IV, Latin
hypercube, scrambled Sobol and Saltelli; responses are metric ids. The analysis gives first-order and total Sobol
indices (Jansen estimators) with bootstrap percentile intervals, main effects with Welch intervals, and the best run
against the rig as it is. `--experiment DESIGN.json` (schema `dexvantage.experiment/1`) or the page's saved design
(`--experiment-saved`). **Equations:** Jansen's estimators [Saltelli et al. 2010]: V = Var(f(A) ∪ f(B)), S_i = (V − ½
mean_j (f(B)_j − f(AB_i)_j)²) / V, S_Ti = ½ mean_j (f(A)_j − f(AB_i)_j)² / V; a main effect d = ȳ_last − ȳ_first with
Welch's interval d ± t_{ν, 1−α/2} · se. **Validation:** Ishigami's analytic Sobol indices within 0.0008 (2/√N, 0.031);
Latin
hypercube strata exact; Sobol discrepancy against `scipy.stats.qmc` 3.4 × 10⁻¹¹; a Sobol set's discrepancy 0.027 of
random's expectation (0.5); exact main effects of a linear model on a factorial and a resolution IV fraction; 20 of 20
fractions have their stated resolution (brute force). **Known limits:** Sobol indices assume independent factors and
settle slowly (read them with their intervals); main-effect intervals treat the other factors as noise; the best run
is a sampled point, not an optimum. See [models/experiments.md](models/experiments.md) (worked example:
[guide/experiments.md](guide/experiments.md)).

**References (checks):** `box_hunter2005`, `efron1993`, `helsel2012`, `hickernell1998`, `ishigami1990`, `jansen1999`, `mckay1979`, `saltelli2002`, `saltelli2010`, `scipy_qmc`, `sobol1993` ([full entries](#references)).

## Optimiser and design agent (NSGA-II) {#optimise}

NSGA-II written from Deb et al. 2002 in NumPy (fast non-dominated sort, crowding distance, bounded SBX and polynomial
mutation, uniform operators for categorical genes, Deb's constrained domination); the patterns of Vantage
`core/optimiser.py` and `core/rig_designer.py` (the current layout as the first candidate, locked cameras kept,
options explained, accepting adds a variant). Objectives are metric ids; requirements become normalised constraints;
an objective the studies do not state is left out with their reason, and cost only when every candidate device states
a price. The optimiser ("Improve this rig", `--optimise`) moves mounts within a box and a cone; the design agent
("Design new rigs", `--design-agent`) chooses devices, mount sites and bracket types. Designs are scored at
screening settings (the grasp study on 40 grasps, labelled defaults) unless `confirm` re-scores the shown ones.
**Equations:** x dominates y when fᵢ(x) ≤ fᵢ(y) for every objective i and fⱼ(x) < fⱼ(y) for one j (with Deb's
constrained domination: a feasible design beats an infeasible one, two infeasible ones compare by total violation); the
crowding distance is Σᵢ (fᵢ(next) − fᵢ(previous)) / (fᵢ,max − fᵢ,min) along each front. **Validation:** ZDT1 and ZDT2
against their analytic fronts, IGD 0.0051 and 0.0053 (0.01); CONSTR 0.021 (0.05) with no
constraint violated; 0 of 480 designs changed a locked sensor; the seed reproduces; the options' figures re-run
without the cache, 0 difference. **Known limits:** an approximate front within a budget of evaluations (about 0.8 s
per design on the UR5e sample, 2 s for the design agent); the design agent's sites are a sphere around the task
volume and a cone around the gripper, without camera roll or stream choice. See
[models/optimise.md](models/optimise.md) (worked example: [guide/optimise.md](guide/optimise.md)).

**References (checks):** `coello2004_igd`, `deb2001_moo`, `deb2002_nsga2`, `deb_agrawal1995_sbx`, `deb_goyal1996_mutation`, `zitzler2000_zdt` ([full entries](#references)).

## Exporters {#exporters}

Each format registers with `@exporter` in `core/exporters/`; `core.exporters.export` writes `export_manifest.json`
(each format's status, files and notes, the SHA-256 of every file, the project's provenance) and stamps every JSON
file with the provenance block (except those a target parser refuses extra keys in, listed as `unstamped`). Formats:
`mjcf` (camera and site snippets for the robot's own MJCF, and the full MuJoCo scene), `urdf` (a URDF built from the
compiled model, the sensor links and optical frames, a tf2 static-transform launch file), `opencv_yaml` (OpenCV
FileStorage per imager and a Kalibr camchain), `masks` (self-occlusion, the mask atlas over a grasp set, depth
validity), `plucker`, `brackets` (STL, STEP with the optional gmsh), `isaac` (USD camera prims, Isaac Lab v3.0.0 and
v2.3.2 camera configs with the tolerance study's randomisation bounds), `lerobot` (the dataset features, train
configs for ACT and Diffusion Policy with the action-chunk study's delay offset and ensemble weight, OpenVLA's
entry), `report` (HTML and PDF of the results with the scorecard and provenance), `perturbations`, `canonical_warp`,
`fiftyone` and `lerobot_dataset`. File ▸ Export… (a window job; the report reuses the results the window has
computed) or `--export OUT_DIR --formats …`. **Validation:** MJCF snippets through MuJoCo 3.1 × 10⁻¹⁷ m; URDF through
urdfdom's RPY rule and MuJoCo's URDF compiler 1.1 × 10⁻¹⁵ m; the tf tree 2.8 × 10⁻¹⁷ m; OpenCV YAML read back bit for
bit and through `projectPoints` 5.1 × 10⁻¹³ px; Kalibr exact; masks against exact ray casting 0 of 187 200 pixels;
USD through pxr's `Gf.Camera` 2.6 × 10⁻⁵ px (10⁻³, float32); Isaac Lab configs 1.4 × 10⁻¹⁶ m; LeRobot export and the
report parsed back, 0 mismatches. **Known limits:** one joint state per export; URDF has no inertials or collision
geometry (effort and velocity 0 where not stated); Isaac Lab, Isaac Sim, LeRobot and Kalibr were not run (configs are
checked against the pinned sources' fields). See [models/exporters_a.md](models/exporters_a.md) and
[models/exporters_b.md](models/exporters_b.md) (worked examples: [guide/exporters_a.md](guide/exporters_a.md),
[guide/exporters_b.md](guide/exporters_b.md)).

## Cross-embodiment canonical view warp {#canonical_warp}

Maps robot B's wrist image to robot A's canonical viewpoint (§7.16) through the fingertip contact plane, measured
from each gripper's pads (origin at the tool centre point, z the approach axis, x the closing axis):
H = K_A (R − t n_Bᵀ/d_B) K_B⁻¹, and a full remap LUT through both lenses. Off the plane the error is the residual
parallax, reported (P95 and a map) over the task volume's depth range along the approach axis. Study
`canonical_view` (`--canonical-view`), format `canonical_warp`. On the UR5e sample (Wrist D405 to Overhead D435) the
default depth range gives 675 px, "large parallax": the range is a pessimistic bound. **Validation:** plane points
2.3 × 10⁻¹³ px (10⁻⁹); the LUT against `cv2.projectPoints` and `fisheye.projectPoints` 2.3 × 10⁻¹³ px; parallax
against brute force 2.8 × 10⁻¹³ px; the LUT through `cv2.remap` against `cv2.warpPerspective` 1 grey level on 16
pixels (1, the quantisation bound: zero margin). **Known limits:** exact only on the contact plane; occlusion is not
modelled; the default depth range is a bound, not a measurement. See
[models/canonical_warp.md](models/canonical_warp.md)
(worked example: [guide/canonical_warp.md](guide/canonical_warp.md)).

## Rerun bridge {#rerun}

`core/bridges/rerun.py` logs the robot over its trajectory, camera frusta and pinholes, min-Z blind zones, task
volumes and audit flags per episode to an `.rrd` file (`--rerun OUT.rrd [--rerun-audit AUDIT.json]`) or a live
viewer (Export ▸ Stream to Rerun…), with the optional `rerun-sdk==0.42.0`. **Validation:**
`formats.rerun_rrd_readback` reads the file back with rerun's own reader (only when rerun-sdk is installed): 83
entities, none missing; intrinsics 4.8 × 10⁻⁵ px, imager poses 7.7 × 10⁻¹⁰ m, the gripper link 1.3 × 10⁻⁸ m against
MuJoCo (10⁻⁴: float32). **Known limits:** a fisheye's frustum is approximate in Rerun's pinhole; one trajectory per
robot; contact pads, objects and study results are not logged; the live viewer was exercised only through a fake
module. See [models/rerun.md](models/rerun.md) (worked example: [guide/rerun.md](guide/rerun.md)).

## The assistant's tools and the MCP server {#assistant}

Ported from Vantage `assistant/tools.py`, `copilot_tools.py`, `page_guide.py` and `mcp/`. 30 tools with closed JSON
Schemas (reads of the project and the library with their sources, every study's figures, metrics, edits that
simulate or apply as one undo step, plans, exports, the page on screen); errors are plain sentences. Nothing is
computed in the tool layer: figures come from study results, the project or a library entry. `python -m
dexvantage.mcp` is a stdio MCP server (MCP specification 2025-11-25) on a project file, or attached to an open
window. **Validation:** 14 tool figures against the study results read again, 3.65 × 10⁻⁵ relative (the briefs'
4-digit rounding, 5 × 10⁻⁴); the server's replies and 41 tool schemas against MCP's examples and the JSON Schema
2020-12 meta-schema, 0 nonconformities; edit then undo, content hash identical. **Known limits:** one request at a
time (a long study blocks the next call); no progress notifications. See [models/mcp.md](models/mcp.md) (worked
example: [guide/mcp.md](guide/mcp.md)).

**References (checks):** `json_schema`, `jsonrpc2`, `mcp_2025_11_25` ([full entries](#references)).

## The Assistant dock and the Copilot {#copilot}

Window ▸ Assistant (Ctrl+J): Claude (the official SDK, pinned in the `assistant` extra) or Groq, keys only in the
system keyring; each question goes with the page on screen. Three rules the loop enforces: every number in an answer
is checked against the figures the tools and the page returned (unmatched numbers are shown to the user), a changing
tool comes back as a proposal applied as one undo step, and long studies run only from a plan the user approves
(recorded with its SHA-256). Chats and memory notes are kept per project. Ported from Vantage's assistant and dock.
**Validation:** the number check on 28 hand-labelled answers, 0 mismatches; the plan gate on 12 records, 0 wrong; a
proposal applied and undone, 0 fields differ. **Known limits:** the post-check matches values, not meanings; no
request was sent to a live API while building (the tests use a fake provider). See
[models/assistant_dock.md](models/assistant_dock.md) (worked example:
[guide/assistant_dock.md](guide/assistant_dock.md)).

**References (checks):** `fips180_4` ([full entries](#references)).

## The ALOHA 2, bimanual Panda and G1 samples {#samples}

From MuJoCo Menagerie (commit 98a70c5, each folder's own licence read: ALOHA BSD-3-Clause, Franka Panda Apache-2.0,
Unitree G1 BSD-3-Clause): `--sample aloha` (both arms as robots of their own, the four ALOHA cameras from `aloha.xml`
and `scene_parts.xml`, WidowX leaders), `--sample panda-bimanual` (two Pandas, GELLO leaders, sample camera
placements), `--sample g1` (the G1's upper body with hands, legs held at `stand`, the head D435 on `torso_link`;
meshes simplified within 0.5 mm). **Validation:** the ALOHA arms against `aloha.xml` 4.6 × 10⁻⁷ m and its cameras
5.1 × 10⁻⁷ m (10⁻⁶: six-digit XML); the G1's upper body against `g1_with_hands.xml` 0 m, its head camera
2.2 × 10⁻¹⁶ m; both Pandas 0 m. **Known limits:** the ALOHA gripper's equality constraint is not applied; the G1 does
not step, squat or bend at the waist; synthetic episodes move one robot. See [models/samples.md](models/samples.md)
(worked example: [guide/samples.md](guide/samples.md)).

**References (checks):** `aloha2_2024`, `menagerie_98a70c5`, `mujoco_camera`, `mujoco_todorov2012`, `unitree_g1_urdf` ([full entries](#references)).

## The robot catalogue {#catalogue}

53 robots in the Library (kind `robot`, one JSON per robot): the maker's figures (DoF, reach, payload,
repeatability, mass, control rate, interfaces, joint ranges), each with its source's index and own words, `null`
where no source states it; and the model (Menagerie folder, MJCF file, commit, licence), imported with that model's
licence shown. 22 entries have no maker figure (pages that render by script, 404s), each note says which.
**Validation:** `robots.catalogue_dof` recounts the joints of every bundled model the catalogue names against its
records, 0 disagreements. **Known limits:** the check counts joints, not geometry or limits; the catalogue's joint
ranges are the maker's, for comparison (an imported robot keeps its model file's own); a maker's page can change after
the date it was read, which each entry records. See [models/robots.md](models/robots.md#catalogue) (worked example:
[guide/robots.md](guide/robots.md)).

**References (checks):** `mujoco_menagerie`, `mujoco_todorov2012`, `robot_datasheets` ([full entries](#references)).

## Project diff, merge, the Git folder form, citations and figures {#versioning}

`core/projdiff.py` lists every change between two projects (meta, robots and joints, variants and sensors with pose
deltas in mm and degrees, workspace, tasks, requirements, policies, rig, cables, calibration, settings, files), with
metric deltas when results exist, and merges three ways (refusing conflicts). `core/projfolder.py` writes a project
as a Git-friendly folder (`.dexv.d`, one JSON per section) and a stable text for `git diff` (`--textconv`).
`core/cite.py` writes BibTeX for every source a project depends on (`--cite`); `core/figures.py` writes publication
figures (SVG or PDF, one or two columns, a colour-blind-safe palette) from a chart's data (`--figure`, Export figure…
on page charts). **Validation:** 24 seeded edits of nine kinds, 0 missed or extra; pose deltas exact; 8 of 8 merge
cases; a folder round trip byte-identical (66 files); 235 BibTeX entries parse with no repeated key; SVG data
recovered from the tick labels within 4.7 × 10⁻⁵ of the axis range; the PDF page within 0.36 pt. **Known limits:**
lists (grasp poses, joint tables) are compared and merged whole; text widths in figures are estimated. See
[models/versioning.md](models/versioning.md) and [models/publication.md](models/publication.md) (worked examples:
[guide/versioning.md](guide/versioning.md), [guide/publication.md](guide/publication.md)).

**References (checks):** `bibtex_btxdoc`, `git_textconv`, `ieee_figure_size`, `json_schema`, `pdf_32000`, `scipy_rotation`, `svg11`, `tol_colours`, `zip_appnote` ([full entries](#references)).

## Python API and plugins {#api}

`import dexvantage as dv`: `dv.open`, `dv.sample`, `dv.new` return a `Rig` (robots, sensors, variants, factors,
`run(study, **settings)`, `metrics`, `export`, `check`, `synth_episodes`, `to_frame` with the optional pandas);
`dexvantage/plugin_sdk.py` lets a package add studies, metrics, exporters, readers, library folders and pages
through the `dexvantage.plugins` entry point (`--list-plugins`, `--no-plugins`, Help ▸ Plugins…), rolled back and
explained when a plugin fails. Neither computes a figure of its own (no validation model); the plugin template's
angle is checked in its own test against a closed form. See [guide/api.md](guide/api.md) and
[guide/plugins.md](guide/plugins.md).

## The workspace check {#workspace}

`core/workspace_check.py` (Workspace ▸ Workspace): whether each task object lies inside a task volume, with a
one-step suggestion of a volume around the objects outside. **Equations:** an object with box [lo, hi] is inside a
volume of centre c and size s when lo ≥ c − s/2 − τ and hi ≤ c + s/2 + τ on each axis, τ = 1 mm (a default); a turned
box's half-extents are |R| h [Arvo 1990]. **Validation:** `workspace.object_extent` (an object's box
against the exact box of its true shape: 0.861 of the tessellation bound over 300 random objects, tolerance 1) and
`workspace.inside_classification` (0 of 240 objects misplaced). **Known limits:** volumes are axis-aligned boxes and the
inside test uses the object's
axis-aligned box. See [models/workspace.md](models/workspace.md) (worked example:
[guide/workspace.md](guide/workspace.md)).

**References (checks):** `arvo1990`, `scipy_rotation` ([full entries](#references)).

## Experiment tracker {#tracking}

`core/tracking/` logs every study run, designed experiment, optimiser run, tolerance Monte Carlo and readiness
scorecard as a **run** (schema `dexvantage.run/1`: start and end in UTC, the project's content hash and variant, the
study's settings and seeds, every figure with unit and status read through `core/metrics`, the result file and its
provenance, tags and a note) in a run store beside the project (JSON lines plus one folder per run, atomic renames
and a file lock). It is off unless the project turns it on (Analysis ▸ Runs, `analysis.studies['tracking']`) or
batch mode gets `--track DIR`; the setting is left out of the content hash. Runs are compared (deltas against the
first, the settings that differ and, with replicates, a significance note from `core/stats`, paired by seed),
re-run through `core/reproduce` and exported as CSV, Parquet, an MLflow 3.17 file store and a W&B replay script.
Cluster runs (`--jobs`, `--gather`, `dv.cluster`) are logged too. **Validation:** 5 checks, tolerance 0 (storage and
file formats are exact or wrong): 0 of 443 fields differ after a round trip, 8 writer processes × 100 runs lose none,
7 of 7 crash cases recover, 0 of 8 240 table cells differ, 0 of 80 fields differ read back by MLflow's own
`MlflowClient`. **Known limits:** a run keeps the project's hash, not the project; MLflow artifact URIs are
absolute; single figures are compared by their difference only. See [models/tracking.md](models/tracking.md)
(worked example: [guide/tracking.md](guide/tracking.md)).

**References (checks):** `json_lines`, `mlflow_client_3172`, `mlflow_filestore_3172`, `parquet_format`, `posix_rename_flock`, `rfc4180` ([full entries](#references)).

## Statistics and power analysis {#stats}

`core/stats.py` (NumPy and SciPy): Wilson and Clopper-Pearson intervals, BCa bootstrap intervals, exact or Monte
Carlo permutation tests (paired and unpaired), Welch and paired t, Mann-Whitney and Wilcoxon, Cohen's d and Hedges'
g with noncentral-t intervals, Cliff's delta, the risk difference (Newcombe) and odds ratio (Woolf), Fisher's exact
and z tests, Holm, Bonferroni, Benjamini-Hochberg and -Yekutieli corrections, and power and sample size for two
proportions (normal, arcsine, exact, simulation) and t tests. The study `stats` compares success counts, two variants
over the tolerance Monte Carlo, an experiment's factor levels or tracker runs; every comparison names its test and
why, and a difference that is not clear comes with the smallest difference the data detect. The campaign planner and
the cell comparison use it. **Equations:** Fisher's exact test from the hypergeometric P(X = x) = C(n₁, x) C(n₂, m − x)
/ C(n₁ + n₂, m); the Wilson interval (p̂ + z²/2n ± z √(p̂(1 − p̂)/n + z²/4n²)) / (1 + z²/n) [Wilson 1927]; Welch's df
(v₁ + v₂)² / (v₁²/(n₁ − 1) + v₂²/(n₂ − 1)), vᵢ = sᵢ²/nᵢ. **Validation:** 13 checks against SciPy (to 10⁻¹⁵), Benjamini
and Hochberg (1995)'s
worked example, Cohen (1992) Table 2 (within 1 per group), the G*Power 3.1 manual (4.6 × 10⁻⁷) and simulated
coverage and power (within 1.6 standard errors). **Known limits:** fixed-sample tests only (no sequential testing);
episodes independent; power for rates and normal data. See [models/stats.md](models/stats.md) (worked example:
[guide/stats.md](guide/stats.md)).

**References (checks):** `benjamini_hochberg1995`, `benjamini_yekutieli2001`, `cliff1993`, `clopper_pearson1934`, `cohen1988`, `cohen1992`, `efron1993`, `fleiss2003`, `gpower31_manual`, `holm1979`, `newcombe1998`, `scipy_stats_tests`, `steiger_fouladi1997`, `wilson1927` ([full entries](#references)).

## Data-collection campaign planner {#campaign}

`core/campaign.py`: the conditions (tasks, rig variants, lighting presets, object sets and teleop devices, crossed or
listed), demonstrations per condition (entered, or from `core/stats`'s two-proportion power analysis), operators
counterbalanced by a Williams Latin square, sessions and days within labelled fatigue limits, storage from the
bandwidth model, the status against the budget, and a schedule exported as CSV, an RFC 5545 calendar and an HTML
checklist (`--campaign SPEC.json`, Data ▸ Campaign, `dv.campaign`). **Equations:** storage GB = Σ payload (Mbit/s) ×
ratio × (1 + container overhead) × episode s / 8 / 1000 over the enabled sensors; a Williams square of order C has the
first row 0, 1, C−1, 2, C−2, … and row i = (first row + i) mod C [Williams 1949]. **Validation:** 6 checks: Williams
squares of
orders 1–12 by brute force (0 violations), 40 seeded campaigns' balance from their CSV (0), 383 sessions within their
limits, storage against the bandwidth study (0), 34 calendar events through a strict RFC 5545 reader (0), and the
normal-approximation sample size against exact power by enumeration (0.033 at most, tolerance 0.04). **Known
limits:** episodes of fixed length; conditions not balanced across days; a compressed format's storage is the raw
size until a measured ratio is entered. See [models/campaign.md](models/campaign.md) (worked example:
[guide/campaign.md](guide/campaign.md)).

**References (checks):** `fleiss2003`, `fleiss_tytun_ury1980`, `rfc5545`, `scipy_stats`, `williams1949` ([full entries](#references)).

## Sim-to-real gap {#sim2real}

`core/sim2real/` renders one recorded episode again in MuJoCo through the rig (each camera's lens and mount, the
robot at the recorded joint states) and compares it with the recording camera by camera: brightness and contrast,
per-channel histogram distance, an exposure gain and offset, sharpness and blur, noise (Immerkær 1996), the robot's
silhouette overlap, the mount offset (through the silhouette calibration), latency and a kernel MMD between frame
sets. Shapley shares rank the causes, and each comes with the change that closes it (Apply, one undo step).
**Equations:** for a set S of gaps closed in the render, d(S) = mean |recorded − f_S(render_S)| in grey levels; each
cause's Shapley share φᵢ = Σ_{S ⊆ P∖{i}} |S|! (n − |S| − 1)! / n! · (d(S) − d(S ∪ {i})), so Σ φᵢ = d(∅) − d(P) [Shapley
1953]. **Validation:** 7 checks: SciPy's Wasserstein distance (1.7 × 10⁻¹³ DN), MMD against a hand case and the Gaussian
closed form, known noise (1.2 % against 5 %) and blur (0.0005 px), Shapley by brute force over all orders, and four
UR5e recordings with one injected gap each, each recovered and ranked first (worst 0.68 of its tolerance). **Known
limits:** the render is MuJoCo's shading of the project's model (textures, shadows and background are what is
left); one episode and a constant latency; no FID (no torch). See [models/sim2real.md](models/sim2real.md) (worked
example: [guide/sim2real.md](guide/sim2real.md)).

**References (checks):** `gretton2012`, `immerkaer1996`, `lerobot_dataset_format`, `mujoco_todorov2012`, `pech_pacheco2000`, `rubner2000`, `scipy_stats`, `scipy_wasserstein`, `shapley1953`, `tai_yang2008` ([full entries](#references)).

## Robot-aligned 3D Gaussian Splatting (splat views) {#splat}

`core/sim/splat_rig.py` fits a 3D Gaussian Splatting scene [kerbl2023_3dgs] of the cell to posed images and attaches
the robot's Gaussians to its links, so the scene re-poses with q (the approach of SplatSim [qureshi2025_splatsim] and
Robo-GS [lou2025_robogs]; neither paper's code is used). `core/sim/splat_views.py` (the **Splat views** page, the
`splat_views` study, `--splat`, `dv.splat`) fits it to a recorded LeRobot episode, each view posed at its mount at the
logged q, or to a MuJoCo render of the project for demonstration; it previews what a proposed camera pose would see,
replays the episode through it to a video, and writes perturbed copies (§8.5) as a LeRobot dataset. Every render is
marked synthetic. Optional: the `splat` extra (PyTorch, lpips) and a CUDA device. **Equations:** a Gaussian's image
footprint is Σ₂ = J R S Sᵀ Rᵀ Jᵀ + 0.3 I (J the lens projection's Jacobian, the EWA approximation [zwicker2001_ewa]),
αᵢ = oᵢ exp(−½ dᵀ Σ₂⁻¹ d), C = Σ cᵢ αᵢ Πⱼ₍ⱼ<ᵢ₎ (1 − αⱼ); a robot Gaussian at μ_link is drawn at T_world_link(q) μ_link;
the quality on held-out frames is PSNR = 10 log10(1 / MSE), SSIM [wang2004_ssim] and LPIPS [zhang2018_lpips].
**Validation:** 7 checks: one Gaussian's footprint against its closed form (6.2 × 10⁻¹⁵), the torch rasteriser against
the numpy reference on the CPU (2.0 × 10⁻¹⁴) and on CUDA (1.3 × 10⁻⁵, left out without a CUDA device), stereo
reprojection against the closed-form disparity (1.8 × 10⁻¹³ px), re-posing against the camera moved the opposite way
(5.9 × 10⁻⁷, float32 storage), SSIM and PSNR of constant images against their closed forms and SSIM against
scikit-image. **Known limits:** every quality figure is from synthetic captures (no real recording is in the
repository); no densification, view-dependent colour or lighting model; quality falls for camera poses far from the
captured views (a wrist camera seen only from a sphere of views scores 17 dB) and for joint states far from the
capture; the views' poses are the planned mounts at the logged q; LPIPS needs AlexNet's weights (a download, asked
first). See [models/splat.md](models/splat.md) (worked example: [guide/splat.md](guide/splat.md)).

**References (checks):** `kerbl2023_3dgs`, `lou2025_robogs`, `qureshi2025_splatsim`, `skimage_ssim`, `wang2004_ssim`, `zwicker2001_ewa` ([full entries](#references)).

## Publication kit {#paper_kit}

`core/publication.py` (Export ▸ Paper kit…, the `paper_kit` export format, `dv.publication`): a methods paragraph,
booktabs LaTeX tables (sensors, results with their intervals, the scorecard, the experiment, variants, the statistical
comparisons with their intervals and corrected p-values, the cell comparison and the tracked runs, each when its
result is present), SVG and PDF figures at one- and two-column widths, a BibTeX file and a reproducibility appendix
with a run plan. Every number is traced to the project, a library entry, a result or the software environment, and
checked again. **Validation:** 4 checks, each at 0: every number of the paragraph found in the source data, every
`\cite` key defined, the LaTeX structure of 10 files, and 60 table cells against their sources. **Known limits:** no
LaTeX engine on the build machine (the compile test runs where one is installed); the paragraph says what the project
and results state, nothing more. See [models/paper_kit.md](models/paper_kit.md) (worked example:
[guide/paper_kit.md](guide/paper_kit.md)).

**References (checks):** `bibtex_btxdoc`, `booktabs_fear2020`, `latex2e_unofficial`, `siunitx_wright2026` ([full entries](#references)).

## Cluster sweeps {#cluster}

`core/cluster/` runs the independent parts of long work on several processes: a designed experiment's runs, a
tolerance Monte Carlo's draws, each optimiser generation's designs and any list of (project, study, settings) jobs,
in a spawn process pool (`--jobs N`, Run on… ▸ This machine) or as a Slurm job array (`--slurm FOLDER`,
`--run-shard`, `--gather`), with a shared on-disk result cache. The gathered result is the in-process one.
**Validation:** 4 checks at 0: a 6-run experiment and a 6-draw Monte Carlo gathered from separate shard processes
against the serial runs, the shard partition over m ≤ 300 and N ≤ 40, three processes writing the cache at once, and
the `#SBATCH` directives through `bash -n`. **Known limits:** Slurm only, a shared filesystem, like machines for
machine-dependent figures; the design agent is not distributed. See [models/cluster.md](models/cluster.md) (worked
example: [guide/cluster.md](guide/cluster.md)).

**References (checks):** `flock2`, `nfs5`, `rename2`, `slurm_job_array`, `slurm_sbatch` ([full entries](#references)).

## Multi-cell comparison {#cells}

`core/cells.py` and `core/cells_pool.py` (File ▸ Compare cells…, `--cells`, `dv.cells.compare`): several robot cells
(other projects, samples, variants) side by side: the figures of merit with the best and worst cell (a difference
meaningful only beyond the build spread of a tolerance Monte Carlo and, with `core/stats`, a Holm-adjusted
permutation test), a geometric gap index per camera role (ported from Vantage `core/rig_gap.py`), whether the cells'
recordings pool for training against the LeRobot feature schema, and the drift of built copies from the plan.
**Equations:** a difference between two cells is meaningful when |Δ| > k · √(σ_ref² + σ_cell²), k = 2 (a default), σ
from the tolerance Monte Carlo of each cell. **Validation:** 6 checks: identical cells give 0, a known rotation and
shift come back to 10⁻¹³, the coverage share
against an exact polygon clip (0.0144, tolerance 0.02), three LeRobot datasets with known differences (0), and known
drift offsets (10⁻¹⁵). **Known limits:** the gap is geometric, not appearance or success, and its bands are a
heuristic; pooling reads the schema, not the data. See [models/cells.md](models/cells.md) (worked example:
[guide/cells.md](guide/cells.md)).

**References (checks):** `andrew1979`, `hampel1974`, `kay_kajiya1986`, `lerobot_dataset_format`, `scipy_rotation`, `sutherland_hodgman1974`, `vantage_rig_gap` ([full entries](#references)).

## Robot import and export: USD, URDF inertials, OpenVLA, Isaac {#robot_io}

The USD robot import (spec §13; `core/robot/usd_import.py`, UsdPhysics to MJCF, through the import wizard and
`dv.robot_io`), a small UsdPhysics writer for the round trip, URDF exports with inertials that MuJoCo loads as
written, OpenVLA state and action encodings chosen from the rig, and Isaac exports with an off-centre principal
point. **Validation:** 5 checks: the URDF's sensor frames through MuJoCo's URDF compiler (10⁻⁶ m), three robots
written to USD and imported back over 50 random states, a third-party USD UR5e (the newton-assets file) against the
sample (2.6 × 10⁻⁸ m), the OpenVLA rules of six rigs (0 mismatches) and the Isaac intrinsics (4.8 × 10⁻⁵ px).
**Known limits:** one articulation per USD file; drives, loops and couplings are not imported; joint limits round to
UsdPhysics' 32-bit floats. See [models/robot_io.md](models/robot_io.md) (worked example:
[guide/robot_io.md](guide/robot_io.md)).

**References (checks):** `isaaclab_camera_v232`, `isaaclab_camera_v300`, `isaaclab_intrinsics`, `mujoco_menagerie`, `mujoco_urdf`, `newton_assets_ur5e`, `openusd_camera`, `openusd_physics`, `openvla_materialize`, `openvla_oft`, `openvla_oxe`, `urdf`, `urdfdom` ([full entries](#references)).

## Prices and the rig cost {#prices}

81 of the 191 built-in library entries state a cited list price (USD, the maker's store or official price list, the
page and date read); `rig.cost` adds up the enabled sensors' prices (a lower bound when some state none), and the
optimiser's cost objective works where every candidate states a price. **Validation:** `library.prices_sound` (every
built-in price positive, ISO 4217, sourced over https and read within two years: 0 broken) and
`library.rig_cost_sum` (the rig cost against the stated prices added up again from the entry files, tolerance
0.005 USD). **Known
limits:** hosts, lights, cables and robots are priced in the library but not added to the rig total; prices go
stale. See [models/library.md](models/library.md#prices) (worked example: [guide/prices.md](guide/prices.md)).

## UI responsiveness and the review round's fixes {#ui_fixes}

Hidden analysis pages hold the store's and the analysis controller's signals until they are shown and replay them
coalesced (`ui/pages/_gate.py`): an edit on the two-arm samples costs 3.6–5.5 ms on the UI thread instead of
28–39 ms. Adding the G1 is 10–13 ms (171 ms before). Experiments censor figures stated only as bounds
(`experiments.censored_best`: 0 contradictions in 300 random designs), and the canonical view says when its parallax
bound is larger than the image. See [models/ui-fixes.md](models/ui-fixes.md) (worked example:
[guide/ui-fixes.md](guide/ui-fixes.md)).

## Performance {#performance}

No UI-thread step over 100 ms on the samples with the shader renderer (largest: 68 ms median, 96 ms worst, for a UR5e
frame after a joint change on an integrated GPU). See [models/performance.md](models/performance.md). The Phase 3,
6 and 7 pages add no step over 100 ms (largest: 67 ms, the events after Apply on the multi-finger occlusion page;
`scripts/measure_ui_phase3.py`). The grasp study's 2 000-grasp budget is in [Grasp self-occlusion](#grasp). The
Phase 4, 5 and 8 pages add none either (largest: 80.8 ms, the events after Apply on the multi-finger page with the
handoff; every new page's Apply under 30 ms; `scripts/measure_ui_phase458.py`).
The Phase 9 and 10 pages add none either (largest: 48.5 ms, Apply on the Tolerances page; every page shows within
17 ms; `scripts/measure_ui_phase910.py`); File ▸ Open sample builds the project on a window job. The research tools
part 2 pages add none either on ALOHA 2 and the G1 (largest: 19.8 ms, the events after the cell comparison arrives;
every new page shows within 18 ms, Apply 11–14 ms; `scripts/measure_ui_rt2.py`), and hidden pages now hold their
signals until shown, so an edit costs 6–7 ms with every page built ([models/ui-fixes.md](models/ui-fixes.md)).

## Provenance {#provenance}

As Vantage: a block under `provenance` (schema `dexvantage.provenance/1`) with the version, a SHA-256 of the package
sources, library versions, the project's content hash and per-section hashes, the study and its settings, seeds,
the batch command with paths reduced to file names, and the SHA-256 of the result without its time stamps.
`--reproduce` re-runs a result and compares every number. No path, user or host name is ever written.

## Project and export formats {#formats}

The `.dexv` project file (a zip of `project.json`, `meshes.npz` and the stored model files, written atomically;
[project-format.md](project-format.md), `project.schema.json`, JSON Schema draft 2020-12), the dataset formats
DexVantage reads and writes (LeRobot v2.1 and v3.0, ROS 2 bags in MCAP and SQLite, RLDS TFRecords: [Dataset
readers](#data)) and the export formats ([Exporters](#exporters)). A format is right or wrong, so most checks have
tolerance 0; geometric round trips through another library carry that library's precision. **Validation:** 19 checks
(`checks_formats.py`, `formats_bags.py`, `formats_exports.py`, `formats_isaac_lerobot.py`, `checks_rerun.py`,
`checks_warp.py`). Measured in the validation run of 2026-11-08 (239 of 239 checks pass): a fully populated
`project.json` against its schema through jsonschema, 0 violations; save and load, 0 sections differ; the LeRobot round
trip, MCAP and TFRecord CRC-32C against the published vectors, 0 mismatches; the MuJoCo scene's cameras loaded by MuJoCo
5.8 × 10⁻⁷ m (2 × 10⁻⁶ m, six-digit XML) and the snippets 3.1 × 10⁻¹⁷ m; URDF 1.1 × 10⁻¹⁵ m and tf2 2.8 × 10⁻¹⁷ m (10⁻⁹
m); OpenCV YAML through `cv2.FileStorage` and the Kalibr camchain exact; masks against the occlusion model 0 pixels; the
bracket STEP volume through gmsh 3.6 × 10⁻¹³ relative; Isaac USD prims through pxr's `Gf.Camera` 2.6 × 10⁻⁵ px (10⁻³,
float32) and the Isaac Lab configs 1.4 × 10⁻¹⁶ m; the LeRobot export and the report parsed back, 0 mismatches; the Rerun
read-back 4.8 × 10⁻⁵ (10⁻⁴); the canonical warp files 1.3 × 10⁻¹⁵ px. **Known limits:** LeRobot's own loader, Isaac Lab,
Isaac Sim and Kalibr were not run (their configs are checked against the pinned sources' fields); the Rerun and STEP
checks need their optional extras; a project of a newer version is refused, not read.

**References (checks):** `act_repo`, `amanatides_woo1987`, `apache_parquet`, `crc_catalogue`, `diffusion_policy_repo`, `ffmpeg_cli`, `gmsh`, `greene1986`, `html_living_standard`, `isaaclab_camera_v232`, `isaaclab_camera_v300`, `json_schema`, `kalibr_yaml`, `lerobot_dataset_format`, `lerobot_train_config`, `mcap_python`, `mcap_spec`, `moller1997`, `mujoco_camera_317`, `mujoco_todorov2012`, `mujoco_urdf`, `mujoco_xml`, `omg_xtypes_cdr`, `opencv_calib3d`, `opencv_filestorage`, `opencv_fisheye`, `openusd_camera`, `openvla_oxe`, `pdf_32000`, `poppler_pdftotext`, `python_ast`, `pyyaml`, `rep103`, `rerun_sdk`, `rfc3720`, `rosbag2_mcap`, `step_ap`, `tf2_static_publisher`, `tfrecord_format`, `urdf`, `urdfdom`, `zip_appnote` ([full entries](#references)).

## The validation suite {#validation}

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
Phases 9 and 10, the samples, the catalogue and research tools part 1: 64 more, 177 in all — robot catalogue 1,
library data 6, tolerance 5, experiments 6, optimise 6, formats 11 more (the exporters' round trips and the Rerun
read-back), the canonical warp 4, assistant 3, requirements 5, copilot 3, versioning 7, samples 5 (each summarised
in its section above).
Research tools part 2 and the review round's fixes: 55 more, 232 in all (the two workspace checks came with the
Phases 9 and 10 reviews) — tracking 5, stats 13, campaign 6, sim2real 7, publication 4, cluster 4, cells 6,
robot_io 5, library 2 (prices), experiments 1 (`experiments.censored_best`) (each summarised in its section above).
With no project given, the project-dependent checks run on the bundled SO-100 and UR5e samples (the multi-finger
checks on the dexhand sample). The self-test runs the quick run's cheap checks (`cheap_only`, 167 of them);
`--validate` runs them all (239 checks with the splat model's 7, all passing: the Phase 12 gate's full run, alone on
this machine, 425 s in the suite and 428 s for the process, peak 5.62 GB, 2026-11-08, timed in
[performance.md](performance.md)). Without a CUDA device `splat.raster_torch_cuda` is left out, and without PyTorch
both torch rasteriser checks are, naming the extra (the other five run on numpy).

## Model documents

| Document | Models |
|---|---|
| [models/robots.md](models/robots.md) | robot import, the MuJoCo kinematics model, bundled models and samples, [the robot catalogue](models/robots.md#catalogue) (worked example: [guide/robots.md](guide/robots.md)) |
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
| [performance.md](performance.md) | the spec §16 budgets measured on one machine: the 2,000-grasp study on every sample, coverage and depth, cube maps, UI-thread events, start-up, import and a full validation run (`scripts/benchmarks.py`) |
| [validation.md](validation.md) | every validation check with its reference, tolerance and the value a full run measured, the agreement with MuJoCo renders first (`scripts/gen_validation_doc.py`) |
| [models/requirements.md](models/requirements.md) | named metrics, requirements, the readiness scorecard and CI gate, variants compare, reproduction and plans (worked example: [guide/requirements.md](guide/requirements.md)) |
| [models/tolerance.md](models/tolerance.md) | Monte Carlo mount and calibration tolerances, requirement shares, the perturbation export (worked example: [guide/tolerance.md](guide/tolerance.md)) |
| [models/experiments.md](models/experiments.md) | designed experiments: factorial, fractional, Latin hypercube, Sobol and Saltelli designs; Sobol indices and main effects (worked example: [guide/experiments.md](guide/experiments.md)) |
| [models/optimise.md](models/optimise.md) | NSGA-II, objectives and constraints, the optimiser and the design agent (worked example: [guide/optimise.md](guide/optimise.md)) |
| [models/exporters_a.md](models/exporters_a.md) | exporters part A: MJCF, URDF and tf, OpenCV and Kalibr calibration, masks, Plücker, brackets, FiftyOne and LeRobot dataset wrappers (worked example: [guide/exporters_a.md](guide/exporters_a.md)) |
| [models/exporters_b.md](models/exporters_b.md) | exporters part B: Isaac Lab and Isaac Sim cameras, LeRobot train configs and policy codes, the HTML and PDF report (worked example: [guide/exporters_b.md](guide/exporters_b.md)) |
| [models/canonical_warp.md](models/canonical_warp.md) | the cross-embodiment canonical view warp: contact frame, homography, remap LUT, residual parallax (worked example: [guide/canonical_warp.md](guide/canonical_warp.md)) |
| [models/rerun.md](models/rerun.md) | the Rerun bridge: what is logged, the read-back check (worked example: [guide/rerun.md](guide/rerun.md)) |
| [models/mcp.md](models/mcp.md) | the assistant's tool layer, sessions, edits as data, plans and the MCP server (worked example: [guide/mcp.md](guide/mcp.md)) |
| [models/assistant_dock.md](models/assistant_dock.md) | the Assistant dock: providers, the tool loop, the number check, proposals and the Copilot's plan gate (worked example: [guide/assistant_dock.md](guide/assistant_dock.md)) |
| [models/samples.md](models/samples.md) | the ALOHA 2, bimanual Panda and G1 samples: sources, licences, what was changed and how it was checked (worked example: [guide/samples.md](guide/samples.md)) |
| [models/versioning.md](models/versioning.md) | project diff, three-way merge, the unpacked folder and the git textconv (worked example: [guide/versioning.md](guide/versioning.md)) |
| [models/publication.md](models/publication.md) | BibTeX of a project's sources and publication figures (worked example: [guide/publication.md](guide/publication.md)) |
| [models/workspace.md](models/workspace.md) | the workspace check: task objects inside task volumes (worked example: [guide/workspace.md](guide/workspace.md)) |
| [models/tracking.md](models/tracking.md) | the experiment tracker: runs, the run store, comparisons, re-runs, CSV, Parquet, MLflow and W&B exports (worked example: [guide/tracking.md](guide/tracking.md)) |
| [models/stats.md](models/stats.md) | statistics and power analysis: intervals, permutation and rank tests, effect sizes, corrections, power and sample size (worked example: [guide/stats.md](guide/stats.md)) |
| [models/campaign.md](models/campaign.md) | the data-collection campaign planner: conditions, demonstrations, Williams-square schedules, storage, calendar export (worked example: [guide/campaign.md](guide/campaign.md)) |
| [models/sim2real.md](models/sim2real.md) | the sim-to-real gap: a recorded episode rendered again, per-camera measures, causes and their changes (worked example: [guide/sim2real.md](guide/sim2real.md)) |
| [models/splat.md](models/splat.md) | robot-aligned 3D Gaussian Splatting: the rasteriser, the rig that re-poses with q, splat views (novel camera pose replay, perturbation augmentation) and the measured reconstruction quality (worked example: [guide/splat.md](guide/splat.md)) |
| [models/paper_kit.md](models/paper_kit.md) | the publication kit: methods paragraph, tables, figures, BibTeX and appendix, every number traced (worked example: [guide/paper_kit.md](guide/paper_kit.md)) |
| [models/cluster.md](models/cluster.md) | cluster sweeps: the process pool, Slurm job arrays, gather, the shared result cache (worked example: [guide/cluster.md](guide/cluster.md)) |
| [models/cells.md](models/cells.md) | the multi-cell comparison: figures of merit, the geometric gap, dataset pooling, drift (worked example: [guide/cells.md](guide/cells.md)) |
| [models/robot_io.md](models/robot_io.md) | USD robot import, URDF inertials, OpenVLA encodings, the Isaac principal point (worked example: [guide/robot_io.md](guide/robot_io.md)) |
| [models/ui-fixes.md](models/ui-fixes.md) | hidden pages that wait until shown, censored experiment figures, the review round's fixes (worked example: [guide/ui-fixes.md](guide/ui-fixes.md)) |
| [models/library.md#prices](models/library.md#prices) | cited prices and the rig cost (worked example: [guide/prices.md](guide/prices.md)) |
| [guide/api.md](guide/api.md), [guide/plugins.md](guide/plugins.md) | the Python and notebook API (`import dexvantage as dv`) and the plugin SDK |

## References {#references}

Every source the validation checks cite, by key (`dexvantage/validation/references.py`; the model documents cite more).

- `act_code`: ACT official code, imitate_episodes.py, temporal_agg (all_time_actions rows by issue step, exp_weights = np.exp(-k * np.arange(n)), k = 0.01: row 0, the oldest chunk, weighs most), https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/imitate_episodes.py, read 2026-11-07.
- `act_repo`: T. Z. Zhao et al., ACT, https://github.com/tonyzhaozh/act at commit 742c753c0d4a5d87076c8f69e5628c79a8cc5488: constants.py (task configs: camera_names) and imitate_episodes.py (temporal aggregation, k = 0.01), read 2026-11-07.
- `aloha2_2024`: ALOHA 2 Team, "ALOHA 2: An Enhanced Low-Cost Hardware for Bimanual Teleoperation", 2024, https://arxiv.org/html/2405.02292v1 (four RealSense D405 cameras at 848 x 480; the MuJoCo Menagerie model "precisely" models all cameras and robot positions; read 2026-11-07).
- `amanatides_woo1987`: J. Amanatides and A. Woo, "A fast voxel traversal algorithm for ray tracing", Eurographics '87, 3–10, 1987.
- `andreff1999`: N. Andreff, R. Horaud and B. Espiau, "On-line hand-eye calibration", Second International Conference on 3-D Digital Imaging and Modeling (3DIM), 430–436, 1999.
- `andrew1979`: A. M. Andrew, "Another efficient algorithm for convex hulls in two dimensions", Information Processing Letters 9(5), 216–219, 1979 (the monotone chain).
- `apache_parquet`: Apache Parquet file format, https://parquet.apache.org/docs/file-format/ (read here through pyarrow.parquet, https://arrow.apache.org/docs/python/parquet.html).
- `arvo1990`: J. Arvo, "Transforming axis-aligned bounding boxes", in A. S. Glassner (ed.), Graphics Gems, Academic Press, 548–550, 1990.
- `ashikhmin2000`: M. Ashikhmin and P. Shirley, "An anisotropic Phong BRDF model", Journal of Graphics Tools 5(2), 25–32, 2000 (the energy-coupled diffuse term, eq. 5).
- `baker_matthews2004`: S. Baker and I. Matthews, "Lucas-Kanade 20 years on: a unifying framework", International Journal of Computer Vision 56(3), 221–255, 2004.
- `barrow1977`: H. G. Barrow, J. M. Tenenbaum, R. C. Bolles and H. C. Wolf, "Parametric correspondence and chamfer matching: two new techniques for image matching", Proc. 5th IJCAI, 659–663, 1977.
- `benjamini_hochberg1995`: Y. Benjamini and Y. Hochberg, "Controlling the false discovery rate: a practical and powerful approach to multiple testing", Journal of the Royal Statistical Society B 57(1), 289-300, 1995: section 3.2, p. 295, the 15 ordered p-values of Neuhaus et al. (1992); Bonferroni at 0.05 rejects 3, Hochberg 3, the FDR procedure at q* = 0.05 rejects 4 (read 2026-11-07 from a scan of the paper).
- `benjamini_yekutieli2001`: Y. Benjamini and D. Yekutieli, "The control of the false discovery rate in multiple testing under dependency", Annals of Statistics 29(4), 1165-1188, 2001 (the factor c(m) = sum of 1/i).
- `bibtex_btxdoc`: BibTeX documentation btxdoc ("Using BibTeX" on CTAN), https://mirrors.ctan.org/biblio/bibtex/base/btxdoc.pdf, listed at https://ctan.org/pkg/bibtex, read 2026-11-07.
- `booktabs_fear2020`: S. Fear, "Publication quality tables in LaTeX", booktabs package documentation, version 1.61803398, CTAN, 2020 (section 2: "Never, ever use vertical rules"; "Never use double rules"; "Put the units in the column heading"), https://ctan.org/pkg/booktabs, read 2026-11-07.
- `borgefors1988`: G. Borgefors, "Hierarchical chamfer matching: a parametric edge matching algorithm", IEEE Transactions on Pattern Analysis and Machine Intelligence 10(6), 849–865, 1988.
- `box_hunter2005`: G. E. P. Box, J. S. Hunter and W. G. Hunter, Statistics for Experimenters: Design, Innovation, and Discovery, 2nd ed., Wiley, 2005 (two-level factorials and their fractions, resolution).
- `brown1966`: D. C. Brown, "Decentering distortion of lenses", Photogrammetric Engineering 32(3), 444–462, 1966.
- `canny1986`: J. Canny, "A computational approach to edge detection", IEEE Transactions on Pattern Analysis and Machine Intelligence 8(6), 679–698, 1986.
- `cliff1993`: N. Cliff, "Dominance statistics: ordinal analyses to answer ordinal questions", Psychological Bulletin 114(3), 494-509, 1993.
- `clopper_pearson1934`: C. J. Clopper and E. S. Pearson, "The use of confidence or fiducial limits illustrated in the case of the binomial", Biometrika 26(4), 404-413, 1934.
- `coello2004_igd`: C. A. Coello Coello and M. Reyes Sierra, "A study of the parallelization of a coevolutionary multi-objective evolutionary algorithm", MICAI 2004, LNAI 2972, 688–697 (inverted generational distance).
- `cohen1988`: J. Cohen, Statistical Power Analysis for the Behavioral Sciences, 2nd ed., Lawrence Erlbaum, 1988 (effect sizes d and h, the power tables).
- `cohen1992`: J. Cohen, "A power primer", Psychological Bulletin 112(1), 155-159, 1992: Table 2, p. 158, N per group for power 0.80 at alpha 0.01, 0.05 and 0.10; row 1 (mean difference, d = 0.2, 0.5, 0.8): 586 95 38 / 393 64 26 / 310 50 20; row 5 (P difference, h = 0.2, 0.5, 0.8): 584 93 36 / 392 63 25 / 309 49 19 (read 2026-11-07 from a scan of the paper).
- `craig2005`: J. J. Craig, Introduction to Robotics: Mechanics and Control, 3rd ed., Pearson Prentice Hall, 2005 (modified DH link transform Rx(α) Tx(a) Rz(θ) Tz(d)).
- `crc_catalogue`: G. Cook, Catalogue of parametrised CRC algorithms, CRC-32/ISCSI (CRC-32C, Castagnoli): poly 0x1edc6f41, init and xorout 0xffffffff, reflected, check 0xe3069283, https://reveng.sourceforge.io/crc-catalogue/17plus.htm, read 2026-11-07.
- `daniilidis1999`: K. Daniilidis, "Hand-eye calibration using dual quaternions", The International Journal of Robotics Research 18(3), 286–298, 1999.
- `deb2001_moo`: K. Deb, Multi-Objective Optimization Using Evolutionary Algorithms, Wiley, 2001 (constrained test problems).
- `deb2002_nsga2`: K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast and elitist multiobjective genetic algorithm: NSGA-II", IEEE Transactions on Evolutionary Computation 6(2), 182–197, 2002 (the fast non-dominated sort, crowding distance, constrained domination and the CONSTR problem).
- `deb_agrawal1995_sbx`: K. Deb and R. B. Agrawal, "Simulated binary crossover for continuous search space", Complex Systems 9(2), 115–148, 1995.
- `deb_goyal1996_mutation`: K. Deb and M. Goyal, "A combined genetic adaptive search (GeneAS) for engineering design", Computer Science and Informatics 26(4), 30–45, 1996 (polynomial mutation).
- `dh1955`: J. Denavit and R. S. Hartenberg, "A kinematic notation for lower-pair mechanisms based on matrices", Journal of Applied Mechanics 22(2), 215–221, 1955.
- `diffusion_policy_repo`: C. Chi et al., Diffusion Policy, https://github.com/real-stanford/diffusion_policy at commit 5ba07ac6661db573af695b419a7947ecb704690f: config/task/real_pusht_image.yaml (shape_meta), config/train_diffusion_unet_real_image_workspace.yaml (n_latency_steps), dataset/real_pusht_image_dataset.py, read 2026-11-07.
- `docarmo1976`: M. P. do Carmo, Differential Geometry of Curves and Surfaces, Prentice-Hall, 1976 (§1-5: the helix (a cos s/c, a sin s/c, b s/c), c² = a² + b², has curvature a / c²).
- `dosovitskiy2021`: A. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021 (the ViT patch grid: (H/P)(W/P) patches).
- `efron1979`: B. Efron, "Bootstrap methods: another look at the jackknife", The Annals of Statistics 7(1), 1–26, 1979 (the percentile bootstrap interval).
- `efron1993`: B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap, Chapman & Hall, 1993 (percentile intervals).
- `ericson2005`: C. Ericson, Real-Time Collision Detection, Morgan Kaufmann, 2005 (§5.1.9, closest points of two line segments).
- `euler_poincare`: The Euler–Poincaré formula for a closed orientable surface of genus g: V − E + F = 2 − 2g per connected shell (e.g. M. de Berg et al., Computational Geometry, 3rd ed., Springer, 2008, §2.2).
- `ffmpeg_cli`: FFmpeg command-line tools ffmpeg and ffprobe (rawvideo rgb24 output, frame pts_time), https://ffmpeg.org/ffmpeg.html and https://ffmpeg.org/ffprobe.html.
- `fips180_4`: NIST FIPS PUB 180-4, Secure Hash Standard (SHS), August 2015 (SHA-256), as implemented by Python's hashlib, https://csrc.nist.gov/pubs/fips/180-4/upd1/final (read 2026-11-07).
- `fleiss2003`: J. L. Fleiss, B. Levin and M. C. Paik, Statistical Methods for Rates and Proportions, 3rd ed., Wiley, 2003 (sample size for comparing two independent proportions).
- `fleiss_tytun_ury1980`: J. L. Fleiss, A. Tytun and H. K. Ury, "A simple approximation for calculating sample sizes for comparing independent proportions", Biometrics 36(2), 343-346, 1980, https://doi.org/10.2307/2529990 (read 2026-11-07).
- `flock2`: Linux man-pages, flock(2) (NFS clients emulate flock with fcntl byte-range locks since Linux 2.6.12; SMB: not propagated before 5.5, mandatory after), https://man7.org/linux/man-pages/man2/flock.2.html, read 2026-11-07.
- `franka_dh`: Franka Emika, "Robot and interface specifications", Denavit–Hartenberg parameters of the Panda (Craig's convention), source/control_parameters.rst of https://github.com/frankarobotics/docs at commit 476a2ed2c1f457b87ba2fbb9213ebcd13825382d (2021-11-11), read 2026-11-06; the page https://frankaemika.github.io/docs/control_parameters.html returned HTTP 404 that day, and its successor https://frankarobotics.github.io/docs/robot_specifications.html gives the same figures for the Franka Research 3.
- `gallup2008`: D. Gallup, J.-M. Frahm, P. Mordohai and M. Pollefeys, "Variable baseline/resolution stereo", IEEE CVPR 2008 (stereo depth error growing as z² / (b f)).
- `garrido2014`: S. Garrido-Jurado, R. Muñoz-Salinas, F. J. Madrid-Cuevas and M. J. Marín-Jiménez, "Automatic generation and detection of highly reliable fiducial markers under occlusion", Pattern Recognition 47(6), 2280–2292, 2014.
- `git_textconv`: Git documentation, gitattributes(5), "Performing text diffs of binary files" (textconv: "The program should take a single argument, the name of a file to convert, and produce the resulting text on stdout"), https://git-scm.com/docs/gitattributes, read 2026-11-07.
- `gmsh`: C. Geuzaine and J.-F. Remacle, "Gmsh: a 3-D finite element mesh generator with built-in pre- and post-processing facilities", International Journal for Numerical Methods in Engineering 79(11), 1309–1331, 2009; gmsh 4.15.2 Python API (occ.addPlaneSurface, addSurfaceLoop with sewing, importShapes, getMass), https://gmsh.info.
- `gpower31_manual`: G*Power 3.1 manual (Heinrich-Heine-Universitaet Duesseldorf, June 1, 2023), https://www.psychologie.hhu.de/fileadmin/redaktion/Fakultaeten/Mathematisch-Naturwissenschaftliche_Fakultaet/Psychologie/AAP/gpower/GPowerManual.pdf: section 19.3, p. 48 (matched pairs, two-sided alpha 0.05, N = 50: d_z = 0.421637 gives power 0.832114, d_z = 0.2828427 gives 0.500352); section 20.3, p. 50 (one sample: d = 0.625, one-sided alpha 0.05, power 0.95 needs N = 30, actual power 0.955144; d = 0.1, two-sided alpha 0.01, power 0.90 needs N = 1492, actual power 0.900169) (read 2026-11-07).
- `greene1986`: N. Greene, "Environment mapping and other applications of world projections", IEEE Computer Graphics and Applications 6(11), 21–29, 1986 (the cube map).
- `gretton2012`: A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schölkopf and A. Smola, "A Kernel Two-Sample Test", Journal of Machine Learning Research 13(25), 723–773, 2012, https://www.jmlr.org/papers/v13/gretton12a.html (read 2026-11-07; the unbiased MMD² of Lemma 6, eq. 3; the median heuristic, section 8).
- `hampel1974`: F. R. Hampel, "The influence curve and its role in robust estimation", Journal of the American Statistical Association 69(346), 383–393, 1974 (the median absolute deviation, 1.4826 × MAD).
- `hartley2004`: R. Hartley and A. Zisserman, Multiple View Geometry in Computer Vision, 2nd ed., Cambridge University Press, 2004.
- `hecht2017`: E. Hecht, Optics, 5th ed., Pearson, 2017 (the Gaussian thin-lens equation; the Fresnel equations, §4.6; Malus's law, §8.2).
- `heitz2014`: E. Heitz, "Understanding the masking-shadowing function in microfacet-based BRDFs", Journal of Computer Graphics Techniques 3(2), 48–107, 2014 (the weak white furnace test).
- `helsel2012`: D. R. Helsel, Statistics for Censored Environmental Data Using Minitab and R, 2nd edition, Wiley, 2012 (ISBN 978-0-470-47988-9; substituting a value for a censored one is not an estimate), https://www.wiley.com/en-us/Statistics+for+Censored+Environmental+Data+Using+Minitab+and+R%2C+2nd+Edition-p-9780470479889 (read 2026-11-07).
- `hickernell1998`: F. J. Hickernell, "A generalized discrepancy and quadrature error bound", Mathematics of Computation 67(221), 299–322, 1998 (the centred L2 discrepancy).
- `holm1979`: S. Holm, "A simple sequentially rejective multiple test procedure", Scandinavian Journal of Statistics 6(2), 65-70, 1979.
- `horaud_dornaika1995`: R. Horaud and F. Dornaika, "Hand-eye calibration", The International Journal of Robotics Research 14(3), 195–210, 1995.
- `howell_catalog`: J. R. Howell, A Catalog of Radiation Heat Transfer Configuration Factors, configuration B-3 (plane element parallel to a rectangle, its normal through a corner), http://www.thermalradiation.net/indexCat.html.
- `html_living_standard`: WHATWG HTML Living Standard, https://html.spec.whatwg.org/, parsed here with the Python standard library's html.parser (https://docs.python.org/3/library/html.parser.html).
- `huber1964`: P. J. Huber, "Robust estimation of a location parameter", The Annals of Mathematical Statistics 35(1), 73–101, 1964.
- `ieee_figure_size`: IEEE Author Center, "Resolution and Size" ("One column width: 3.5 inches, 88.9 millimeters, or 21 picas"; "Two columns width: 7.16 inches, 182 millimeters, or 43 picas"), https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/resolution-and-size/, read 2026-11-07.
- `immerkaer1996`: J. Immerkær, "Fast Noise Variance Estimation", Computer Vision and Image Understanding 64(2), 300–302, 1996, doi:10.1006/cviu.1996.0060 (bibliographic record and abstract read 2026-11-07; the full text is paywalled, the estimator's factors are derived again in core/sim2real/measures.py).
- `intel_d400`: RealSense, "RealSense Product Family D400 Series Datasheet", document 337029-017, revision 023, March 2026 (Table 3-49 depth camera SKU properties, §4.4 depth FOV at distance, Table 4-12 minimum-Z depth, Table 4-15 depth quality), https://www.realsenseai.com/wp-content/uploads/2026/03/RealSense-D400-Series-Datasheet-Mar-2026.pdf, read 2026-11-06.
- `intel_d457`: RealSense, "RealSense Product Family D457 Datasheet", document 337029-011, revision 006, August 2025 (Table 3-19 depth camera D457 properties, Table 4-4 minimum-Z depth), https://realsenseai.com/wp-content/uploads/2025/08/RealSense-D457-Datasheet-Aug_2025.pdf, read 2026-11-06.
- `irani_anandan1996`: M. Irani and P. Anandan, "Parallax geometry of pairs of points for 3D scene analysis", ECCV 1996, LNCS 1064, 17–30, doi:10.1007/BFb0015520 (the plane-plus-parallax decomposition; bibliographic details read at dblp.org, 2026-11-07).
- `isaac_gr00t`: NVIDIA, Isaac-GR00T, branch n1-release, gr00t/data/transform/video.py VideoCrop and gr00t/experiment/data_config.py, https://github.com/NVIDIA/Isaac-GR00T (commit 755876a), read 2026-11-06.
- `isaaclab_camera_cfg`: Isaac Lab, source/isaaclab/isaaclab/sensors/camera/camera_cfg.py, CameraCfg.OffsetCfg (pos, rot, convention "ros"; rot (w, x, y, z) at tag v2.3.0, (x, y, z, w) at tag v3.0.0 and main commit d4be263), https://github.com/isaac-sim/IsaacLab, read 2026-11-07.
- `isaaclab_camera_v232`: Isaac Lab tag v2.3.2 (commit 37ddf626871758333d6ed89cf64ad702aef127d0), the same files: CameraCfg.OffsetCfg rot (w, x, y, z), TiledCameraCfg, PinholeCameraCfg without OpenCV distortion fields, https://github.com/isaac-sim/IsaacLab, read 2026-11-07.
- `isaaclab_camera_v300`: Isaac Lab tag v3.0.0 (commit e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4), source/isaaclab/isaaclab/sensors/camera/camera_cfg.py (CameraCfg, OffsetCfg rot (x, y, z, w), convention "ros"), tiled_camera_cfg.py (deprecated), camera.py (_initialize_intrinsics), sim/spawners/sensors/sensors_cfg.py (PinholeCameraCfg, OpenCvPinholeDistortionCfg, OpenCvFisheyeDistortionCfg) and sensors.py (spawn_camera, omni:lensdistortion attributes), https://github.com/isaac-sim/IsaacLab, read 2026-11-07.
- `isaaclab_intrinsics`: Isaac Lab v3.0.0 (commit e3f7d9f) sensors/camera/camera.py L828-857 and L879-904 (_read_authored_opencv_intrinsics, _initialize_intrinsics: the authored OpenCV fx, fy, cx, cy rule; without them cx = width / 2) and sim/spawners/sensors/sensors.py L162-165 (aperture offsets ignored); v2.3.2 (commit 37ddf62) camera.py L589-599 (cx = width / 2), https://github.com/isaac-sim/IsaacLab (read 2026-11-07).
- `ishigami1990`: T. Ishigami and T. Homma, "An importance quantification technique in uncertainty analysis for computer models", Proc. First International Symposium on Uncertainty Modeling and Analysis (ISUMA '90), 398–403, IEEE, 1990.
- `iso4217`: ISO 4217 Codes for the representation of currencies, list one (current currency and funds code list), SIX Financial Information as the maintenance agency, published 2026-09-17, https://www.six-group.com/dam/download/financial-information/data-center/iso-currrency/lists/list-one.xml (read 2026-11-07).
- `iso9409_1`: ISO 9409-1:2004, Manipulating industrial robots — Mechanical interfaces — Part 1: Plates; public preview (clauses 4 and 5, Figure 1), https://cdn.standards.iteh.ai/samples/36578/348a837664f444bc83e2902c7a5acf4c/ISO-9409-1-2004.pdf, read 2026-11-06.
- `jansen1999`: M. J. W. Jansen, "Analysis of variance designs for model output", Computer Physics Communications 117(1–2), 35–43, 1999.
- `jcgm100_2008`: JCGM 100:2008, Evaluation of measurement data — Guide to the expression of uncertainty in measurement (GUM), clause 5.1.2 and 5.2.2, eq. 10 and 13 (the law of propagation of uncertainty), https://www.bipm.org/en/committees/jc/jcgm/publications.
- `jcgm101_2008`: JCGM 101:2008, Evaluation of measurement data — Supplement 1 to the GUM: propagation of distributions using a Monte Carlo method, https://www.bipm.org/en/committees/jc/jcgm/publications.
- `json_lines`: JSON Lines (one JSON value per line, UTF-8, newline-terminated), https://jsonlines.org/ (read 2026-11-07).
- `json_schema`: JSON Schema, draft 2020-12, https://json-schema.org/draft/2020-12.
- `jsonrpc2`: JSON-RPC 2.0 Specification (JSON-RPC Working Group, 2010), https://www.jsonrpc.org/specification.
- `kalibr`: ethz-asl/kalibr: the "Calibration targets" wiki page (AprilGrid tagSize "edge to edge", tagSpacing the ratio of the gap to tagSize, 0.3 in the example), aslam_cv/aslam_cameras_april/src/GridCalibrationTargetAprilgrid.cpp and its header (point order, blackTagBorder 2, minTagsForValidObs 4) and kalibr_create_target_pdf (tag ids row by row from the bottom left, corner squares), https://github.com/ethz-asl/kalibr, read 2026-11-06.
- `kalibr_yaml`: ethz-asl/kalibr wiki, "YAML formats" (camchain.yaml: camera_model, intrinsics [fu fv pu pv], distortion_model, distortion_coeffs, resolution, rostopic, T_cn_cnm1) and "Supported models"; kalibr_common/ConfigReader.py and aslam_cv RadialTangentialDistortion / EquidistantDistortion at commit 1f60227, https://github.com/ethz-asl/kalibr, read 2026-11-07.
- `kannala_brandt2006`: J. Kannala and S. S. Brandt, "A generic camera model and calibration method for conventional, wide-angle, and fish-eye lenses", IEEE Transactions on Pattern Analysis and Machine Intelligence 28(8), 1335–1340, 2006.
- `kass_raftery1995`: R. E. Kass and A. E. Raftery, "Bayes factors", Journal of the American Statistical Association 90(430), 773–795, 1995 (BIC differences over 6: strong evidence).
- `kay_kajiya1986`: T. L. Kay and J. T. Kajiya, "Ray tracing complex scenes", SIGGRAPH '86, Computer Graphics 20(4), 269–278, 1986 (the slab test of a ray against a box).
- `kerbl2023_3dgs`: B. Kerbl, G. Kopanas, T. Leimkühler and G. Drettakis, "3D Gaussian Splatting for real-time radiance field rendering", ACM Transactions on Graphics 42(4), 2023; reference code utils/camera_utils.py camera_to_JSON, https://github.com/graphdeco-inria/gaussian-splatting (main, commit 54c035f), read 2026-11-07.
- `knapp_carter1976`: C. H. Knapp and G. C. Carter, "The generalized correlation method for estimation of time delay", IEEE Transactions on Acoustics, Speech, and Signal Processing 24(4), 320–327, 1976.
- `kolmogorov1933`: A. N. Kolmogorov, "Sulla determinazione empirica di una legge di distribuzione", Giornale dell'Istituto Italiano degli Attuari 4, 83–91, 1933.
- `kunsch1989`: H. R. Künsch, "The jackknife and the bootstrap for general stationary observations", The Annals of Statistics 17(3), 1217–1241, 1989 (the moving-block bootstrap).
- `latex2e_unofficial`: LaTeX2e: An unofficial reference manual (June 2026), "Environments" ("The environment-name at the beginning must exactly match that at the end"), https://latexref.xyz/Environments.html, read 2026-11-07.
- `lerobot`: Hugging Face, LeRobot, src/lerobot/policies/common/vla_utils.py resize_with_pad (left and top padding), https://github.com/huggingface/lerobot (commit d490e47), read 2026-11-06.
- `lerobot_act_ensembler`: Hugging Face, LeRobot, src/lerobot/policies/act/modeling_act.py ACTTemporalEnsembler ("w_0 is the oldest action"), https://github.com/huggingface/lerobot/blob/d490e472a5d1a207cfe0255caaaf74b6d66e4656/src/lerobot/policies/act/modeling_act.py, read 2026-11-07.
- `lerobot_dataset_format`: Hugging Face, LeRobot dataset format: v3.0 from src/lerobot/datasets/{utils, dataset_metadata, dataset_writer, io_utils, compute_stats, video_utils}.py, src/lerobot/scripts/convert_dataset_v21_to_v30.py and docs/source/lerobot-dataset-v3.mdx at commit d490e472a5d1a207cfe0255caaaf74b6d66e4656; v2.1 from the same files at tag v0.3.3 (commit b883328e6c95681ca90a18b102e4ae5e1f91e2bf), https://github.com/huggingface/lerobot, read 2026-11-07.
- `lerobot_so100`: Hugging Face, LeRobot documentation, "SO-100" (the leader arm is built from the follower's parts through the wrist and both are calibrated to the same joint positions), https://huggingface.co/docs/lerobot/so100, read 2026-11-07.
- `lerobot_train_config`: Hugging Face LeRobot at commit d490e472a5d1a207cfe0255caaaf74b6d66e4656: src/lerobot/configs/{train, default, policies, types}.py (TrainPipelineConfig, DatasetConfig, PreTrainedConfig, FeatureType), policies/act/configuration_act.py, policies/diffusion/configuration_diffusion.py, datasets/factory.py (resolve_delta_timestamps), https://github.com/huggingface/lerobot, read 2026-11-07.
- `li2010`: A. Li, L. Wang and D. Wu, "Simultaneous robot-world and hand-eye calibration using dual-quaternions and Kronecker product", International Journal of the Physical Sciences 5(10), 1530–1536, 2010.
- `liang2008`: C.-K. Liang, L.-W. Chang and H. H. Chen, "Analysis and compensation of rolling shutter effect", IEEE Transactions on Image Processing 17(8), 1323–1330, 2008.
- `liang_zeger1986`: K.-Y. Liang and S. L. Zeger, "Longitudinal data analysis using generalized linear models", Biometrika 73(1), 13–22, 1986 (the cluster-robust "sandwich" covariance).
- `library_host_sources`: The makers' specification pages and datasheets each host library entry cites in its sources (dexvantage/resources/library/host/*.json), with the date each was read.
- `library_policy_sources`: The papers, model cards and official code each policy library entry cites in its sources (dexvantage/resources/library/policy/*.json), with the date each was read.
- `library_teleop_sources`: The makers' pages and the papers each teleop library entry cites in its sources (dexvantage/resources/library/teleop/*.json), with the date each was read.
- `liegeois1977`: A. Liégeois, "Automatic supervisory control of the configuration and behavior of multibody mechanisms", IEEE Transactions on Systems, Man, and Cybernetics 7(12), 868–871, 1977.
- `liu2010_fdcm`: M.-Y. Liu, O. Tuzel, A. Veeraraghavan and R. Chellappa, "Fast directional chamfer matching", IEEE CVPR 2010, 1696–1703 (edges matched within orientation channels).
- `lou2025_robogs`: H. Lou, Y. Liu, Y. Pan, Y. Geng, J. Chen, W. Ma, C. Li, L. Wang, H. Feng, L. Shi, L. Luo and Y. Shi, "Robo-GS: A physics consistent spatial-temporal model for robotic arm with hybrid representation", ICRA 2025, arXiv:2408.14873, https://arxiv.org/abs/2408.14873 (read 2026-11-08).
- `lucas_kanade1981`: B. D. Lucas and T. Kanade, "An iterative image registration technique with an application to stereo vision", IJCAI 1981, 674–679.
- `lynch_park2017`: K. M. Lynch and F. C. Park, Modern Robotics: Mechanics, Planning, and Control, Cambridge University Press, 2017 (product of exponentials, chapter 4; space Jacobian, chapter 5).
- `mcap_python`: Foxglove, mcap Python package (mcap.reader.make_reader with validate_crcs), MIT licence, https://github.com/foxglove/mcap/tree/main/python/mcap, version 1.6.0 used 2026-11-07.
- `mcap_spec`: Foxglove, MCAP file format specification (records, chunks, message indexes, summary, CRC-32 fields), https://mcap.dev/spec, read 2026-11-07.
- `mckay1979`: M. D. McKay, R. J. Beckman and W. J. Conover, "A comparison of three methods for selecting values of input variables in the analysis of output from a computer code", Technometrics 21(2), 239–245, 1979.
- `mcluney2014`: W. R. McCluney, Introduction to Radiometry and Photometry, 2nd ed., Artech House, 2014 (the inverse-square law, Lambertian sources).
- `mcp_2025_11_25`: Model Context Protocol specification, version 2025-11-25: basic/lifecycle, basic/transports (stdio), server/tools, https://modelcontextprotocol.io/specification/2025-11-25 (read 2026-11-07).
- `mcpeak2015`: K. M. McPeak et al., "Plasmonic films can easily be better: rules and recipes", ACS Photonics 2, 326–333, 2015 (aluminium n, k; via the refractiveindex.info database, CC0, read 2026-11-07).
- `menagerie_98a70c5`: MuJoCo Menagerie at commit 98a70c569c4c1b6da467a20012f766ee1f8dc521, folders aloha, franka_emika_panda and unitree_g1, https://github.com/google-deepmind/mujoco_menagerie (read 2026-11-07).
- `metropolis_ulam1949`: N. Metropolis and S. Ulam, "The Monte Carlo method", Journal of the American Statistical Association 44(247), 335–341, 1949.
- `miyamoto1964`: K. Miyamoto, "Fish eye lens", Journal of the Optical Society of America 54(8), 1060–1061, 1964.
- `mlflow_client_3172`: MLflow 3.17.2, mlflow.tracking.MlflowClient over a file: tracking URI (the package mlflow-skinny 3.17.2, an independent reader of the mlruns/ layout), https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html (read 2026-11-07).
- `mlflow_filestore_3172`: MLflow 3.17.2, mlflow/store/tracking/file_store.py (FileStore: the mlruns/ layout, meta.yaml, metrics/params/tags files) with mlflow/entities/run_info.py, run_status.py, source_type.py and mlflow/utils/validation.py at the same tag, https://github.com/mlflow/mlflow/blob/v3.17.2/mlflow/store/tracking/file_store.py (read 2026-11-07).
- `moller1997`: T. Möller and B. Trumbore, "Fast, minimum storage ray-triangle intersection", Journal of Graphics Tools 2(1), 21–28, 1997.
- `ms_usb_isoch`: Microsoft, "How to transfer data to USB isochronous endpoints", Windows driver documentation, 2024-01-17: "For high speed and SuperSpeed, the bus interval is a microframe. The length of a microframe is 125 microseconds", https://learn.microsoft.com/windows-hardware/drivers/usbcon/transfer-data-to-isochronous-endpoints (read 2026-11-07).
- `mujoco_camera`: MuJoCo XML reference, body/camera (looks along its −z axis, +x right, +y up; fovy, focal, focalpixel, principal, principalpixel, sensorsize, resolution) and visual/quality (offsamples, numslices), doc/XMLreference.rst of https://github.com/google-deepmind/mujoco (main branch), read 2026-11-06.
- `mujoco_camera_317`: MuJoCo XML reference at tag 3.17.0 (the installed version), body/camera: fovy, resolution, sensorsize ("When specified, all intrinsic attributes become active and fovy is ignored"), focal / focalpixel, principal / principalpixel ("Offset of the principal point ... from the image center"), and visual/map znear, zfar; doc/XMLreference.rst of https://github.com/google-deepmind/mujoco/blob/3.17.0/doc/XMLreference.rst, read 2026-11-07.
- `mujoco_geomdistance`: MuJoCo API reference, mj_geomDistance (the signed distance between two geoms, negative when they overlap), https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-geomdistance.
- `mujoco_menagerie`: MuJoCo Menagerie (Google DeepMind), a collection of MuJoCo models with per-model licences, https://github.com/google-deepmind/mujoco_menagerie at commit 98a70c569c4c1b6da467a20012f766ee1f8dc521, read 2026-11-07.
- `mujoco_renderer`: MuJoCo Python bindings, mujoco.Renderer (offscreen OpenGL rendering of RGB, depth as the distance from the camera plane, and segmentation as (object id, object type)), with MUJOCO_GL=egl, https://mujoco.readthedocs.io/en/stable/python.html#rendering.
- `mujoco_todorov2012`: E. Todorov, T. Erez and Y. Tassa, "MuJoCo: A physics engine for model-based control", IEEE/RSJ IROS 2012, 5026–5033.
- `mujoco_urdf`: MuJoCo XML reference and modeling guide, URDF extensions: the <mujoco><compiler …/></mujoco> element of a URDF (fusestatic, discardvisual, boundmass), read through mujoco.MjModel at 3.17.0, https://mujoco.readthedocs.io/en/stable/modeling.html#urdf-extensions.
- `mujoco_xml`: MuJoCo XML reference: quat attributes are written (w, x, y, z), https://mujoco.readthedocs.io/en/stable/XMLreference.html.
- `nakamura1986`: Y. Nakamura and H. Hanafusa, "Inverse kinematic solutions with singularity robustness for robot manipulator control", ASME Journal of Dynamic Systems, Measurement, and Control 108(3), 163–171, 1986.
- `naneyec`: ams OSRAM, NanEyeC miniature camera module datasheet (frame rate and serial data rate over SEIM and LVDS), https://look.ams-osram.com/m/19863b4335e67f1b/original/NanEyeC-Miniature-Camera-Module.pdf, as recorded in the library entry ams-osram-naneyec (read 2026-11-06).
- `newcombe1998`: R. G. Newcombe, "Interval estimation for the difference between independent proportions: comparison of eleven methods", Statistics in Medicine 17(8), 873-890, 1998 (method 10).
- `newton_assets_ur5e`: newton-physics/newton-assets at commit ac1f172b6f53915f9b81aa9ccd765ae792cc264c, folder universal_robots_ur5e/usd_structured (BSD-3-Clause, ROS Industrial Consortium): written by the mujoco-usd-converter v0.5.0 (https://github.com/newton-physics/mujoco-usd-converter) from MuJoCo Menagerie universal_robots_ur5e/ur5e.xml at commit da76818e269b82289eba39808e2fb91d679d6994, https://github.com/newton-physics/newton-assets (read 2026-11-07).
- `nfs5`: Linux man-pages, nfs(5) (nolock: locks exclude only applications on the same client; close-to-open cache consistency), https://man7.org/linux/man-pages/man5/nfs.5.html, read 2026-11-07.
- `nist_sp811`: NIST Special Publication 811, Guide for the Use of the International System of Units (SI), Appendix B.8: pound (avoirdupois) = 4.535 924 E-01 kg (exactly 0.453 592 37 kg), https://www.nist.gov/pml/special-publication-811, read 2026-11-07.
- `omg_xtypes_cdr`: Object Management Group, DDS-XTypes 1.3, section 7.4 (XCDR1 "plain CDR" serialisation: encapsulation header, alignment of primitives to their size), https://www.omg.org/spec/DDS-XTypes/1.3.
- `opencv_calib3d`: OpenCV, "Camera Calibration and 3D Reconstruction" (calib3d): the pinhole camera model with Brown-Conrady radial, tangential and rational distortion (k1, k2, p1, p2, k3, k4, k5, k6) and cv2.projectPoints, https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html.
- `opencv_charuco`: OpenCV objdetect aruco module: CharucoBoard, CharucoDetector, ArucoDetector and the predefined dictionaries (DICT_APRILTAG_36h11 among them), https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html.
- `opencv_distance_transform`: OpenCV imgproc, cv2.distanceTransform and cv2.distanceTransformWithLabels (DIST_L2 with a 5 × 5 mask, DIST_LABEL_PIXEL), https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html.
- `opencv_filestorage`: OpenCV core, cv::FileStorage (XML, YAML and JSON persistence of matrices), https://docs.opencv.org/4.x/da/d56/classcv_1_1FileStorage.html (HTTP 403 on 2026-11-07; the installed OpenCV 5.0.0 bindings were used).
- `opencv_fisheye`: OpenCV, "Fisheye camera model" (calib3d fisheye): θd = θ (1 + k1 θ² + k2 θ⁴ + k3 θ⁶ + k4 θ⁸) and cv2.fisheye.projectPoints, https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html.
- `opencv_handeye`: OpenCV calib3d, calibrateHandEye and calibrateRobotWorldHandEye (conventions: gripper-to-base and target-to-camera in, camera-to-gripper out; world-to-camera and base-to-gripper in, base-to-world and gripper-to-camera out), https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html.
- `opencv_remap`: OpenCV imgproc, cv2.remap (dst(x, y) = src(map_x(x, y), map_y(x, y)), nearest and bilinear interpolation), https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html.
- `opencv_resize`: OpenCV, cv2.resize with INTER_LINEAR (bilinear, pixel centres aligned), https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html.
- `opencv_warp_perspective`: OpenCV imgproc, cv2.warpPerspective (dst(x, y) = src(M⁻¹ (x, y)) unless WARP_INVERSE_MAP is set; sample positions in 1/INTER_TAB_SIZE = 1/32 px), the docstring of the installed OpenCV 5.0.0, read 2026-11-07; https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html returned HTTP 403 that day.
- `openpi`: Physical Intelligence, openpi, src/openpi/shared/image_tools.py resize_with_pad and src/openpi/models/model.py, https://github.com/Physical-Intelligence/openpi (commit 215abfb), read 2026-11-06.
- `openusd_camera`: OpenUSD UsdGeomCamera (focalLength, horizontalAperture, verticalAperture and their offsets, clippingRange; a camera looks along -Z with +Y up) and Gf.Camera / GfFrustum (ComputeViewMatrix, ComputeProjectionMatrix), https://openusd.org/release/api/class_usd_geom_camera.html, as implemented by usd-core 26.8 (pxr), read 2026-11-07.
- `openusd_physics`: OpenUSD UsdPhysics schemas (PhysicsRigidBodyAPI, PhysicsMassAPI, PhysicsArticulationRootAPI, PhysicsRevoluteJoint / PrismaticJoint / FixedJoint: body0, body1, localPos0/1, localRot0/1, axis, lowerLimit and upperLimit in degrees or distance), https://openusd.org/release/api/usd_physics_page_front.html, as implemented by usd-core 26.8 (pxr), read 2026-11-07.
- `openvla_materialize`: OpenVLA at commit c8f03f48af692657d3060c19588038c7220e9af9, prismatic/vla/datasets/rlds/oxe/materialize.py L32-33 (only EEF_POS and EEF_R6 actions) and configs.py L33-50 (StateEncoding, ActionEncoding), https://github.com/openvla/openvla (read 2026-11-07).
- `openvla_oft`: OpenVLA-OFT at commit e4287e94541f459edc4feabc4e181f537cd569a8, prismatic/vla/datasets/rlds/oxe/materialize.py L32-33 and L43-45 (JOINT_POS_BIMANUAL accepted, all 14 values absolute) and configs.py L681-687 (the ALOHA entries), https://github.com/moojink/openvla-oft (read 2026-11-07).
- `openvla_oxe`: OpenVLA, https://github.com/openvla/openvla at commit c8f03f48af692657d3060c19588038c7220e9af9: prismatic/vla/datasets/rlds/oxe/configs.py (OXE_DATASET_CONFIGS: image_obs_keys, depth_obs_keys, state_obs_keys, StateEncoding, ActionEncoding), read 2026-11-07.
- `park_martin1994`: F. C. Park and B. J. Martin, "Robot sensor calibration: solving AX = XB on the Euclidean group", IEEE Transactions on Robotics and Automation 10(5), 717–721, 1994.
- `parquet_format`: Apache Parquet file format (read here through pyarrow.parquet, an independent reader), https://parquet.apache.org/docs/file-format/.
- `pdf_32000`: ISO 32000-1:2008, Document management — Portable document format — Part 1: PDF 1.7 (§7.5 file structure, §7.4.4 FlateDecode, §9.6.2.2 the standard 14 fonts); Adobe's public copy, https://opensource.adobe.com/dc-acrobat-sdk-docs/pdfstandards/PDF32000_2008.pdf.
- `pech_pacheco2000`: J. L. Pech-Pacheco, G. Cristóbal, J. Chamorro-Martínez and J. Fernández-Valdivia, "Diatom autofocusing in brightfield microscopy: a comparative study", ICPR 2000, vol. 3, 314–317 (the variance of the Laplacian as a focus measure).
- `plucker1865`: J. Plücker, "On a new geometry of space", Philosophical Transactions of the Royal Society of London 155, 725–791, 1865 (line coordinates: direction d and moment m = o × d, m · d = 0).
- `poppler_pdftotext`: Poppler pdftotext, the PDF text extractor of the poppler-utils package, https://poppler.freedesktop.org/.
- `posix_exit`: The Open Group Base Specifications Issue 7, 2018 edition (IEEE Std 1003.1-2017), exit(): "The value of status may be 0, EXIT_SUCCESS, EXIT_FAILURE, or any other value", https://pubs.opengroup.org/onlinepubs/9699919799/functions/exit.html (read 2026-11-07); the meaning of 1 and 2 is DexVantage's (spec §11).
- `posix_rename_flock`: POSIX.1-2017 rename() ("a link named new shall remain visible to other threads throughout the renaming operation and refer either to the file referred to by new or old before the operation began"; RATIONALE: "the action of the function be atomic") and Linux flock(2) advisory locks, https://pubs.opengroup.org/onlinepubs/9699919799/functions/rename.html, https://man7.org/linux/man-pages/man2/flock.2.html (read 2026-11-07).
- `python_ast`: Python standard library, ast (parse, literal_eval) and xml.etree.ElementTree, https://docs.python.org/3/library/ast.html.
- `pyyaml`: PyYAML (yaml.safe_load), https://pyyaml.org/wiki/PyYAMLDocumentation.
- `qureshi2025_splatsim`: M. N. Qureshi, S. Garg, F. Yandun, D. Held, G. Kantor and A. Silwal, "SplatSim: Zero-shot Sim2Real transfer of RGB manipulation policies using Gaussian Splatting", ICRA 2025, arXiv:2409.10161, https://splatsim.github.io (read 2026-11-08).
- `ray2002`: S. F. Ray, Applied Photographic Optics, 3rd ed., Focal Press, 2002 (depth of field and hyperfocal distance of a thin lens).
- `rename2`: Linux man-pages, rename(2) ("If newpath already exists, it will be atomically replaced, so that there is no point at which another process attempting to access newpath will find it missing"; on NFS a failed rename may have been done), https://man7.org/linux/man-pages/man2/rename.2.html, read 2026-11-07.
- `rep103`: ROS REP 103, Standard Units of Measure and Coordinate Conventions (T. Foote, M. Purvis, 2010), https://www.ros.org/reps/rep-0103.html.
- `rerun_sdk`: Rerun, rerun-sdk 0.42.0 for Python (RecordingStream, the archetypes Transform3D, Pinhole, Mesh3D, Boxes3D, TextLog, Scalars, and the reader rerun.chunk.RrdReader): the installed wheel's source and https://rerun.io/docs/reference/types/archetypes/pinhole (version 0.42.0), read 2026-11-07.
- `rfc3720`: J. Satran et al., RFC 3720, Internet Small Computer Systems Interface (iSCSI), April 2004, appendix B.4 CRC examples (CRC32C test vectors), https://www.rfc-editor.org/rfc/rfc3720.txt, read 2026-11-07.
- `rfc4180`: Y. Shafranovich, Common Format and MIME Type for Comma-Separated Values (CSV) Files, RFC 4180, October 2005, https://www.rfc-editor.org/rfc/rfc4180 (read here through Python's csv module).
- `rfc5545`: B. Desruisseaux (ed.), Internet Calendaring and Scheduling Core Object Specification (iCalendar), RFC 5545, September 2009 (§3.1 content lines and folding, §3.3.11 TEXT, §3.6 and §3.6.1 required properties, §3.8.7.2 DTSTAMP in UTC), https://www.rfc-editor.org/rfc/rfc5545 (read 2026-11-07).
- `rfc8259`: T. Bray (ed.), RFC 8259, The JavaScript Object Notation (JSON) Data Interchange Format, December 2017.
- `robot_datasheets`: The makers' datasheets and product pages each robot catalogue entry cites in its sources (dexvantage/resources/library/robot/*.json), with the date each was read.
- `rosbag2_mcap`: ROS 2 rosbag2 storage plugins rosbag2_storage_mcap (profile ros2, ros2msg schemas, cdr messages) and rosbag2_storage_sqlite3 (topics and messages tables), https://github.com/ros2/rosbag2.
- `rubner2000`: Y. Rubner, C. Tomasi and L. J. Guibas, "The Earth Mover's Distance as a Metric for Image Retrieval", International Journal of Computer Vision 40(2), 99–121, 2000.
- `safetensors`: Hugging Face, safetensors, README "Format" (8-byte little-endian header length, UTF-8 JSON header with dtype, shape and data_offsets, __metadata__, no holes, little-endian row-major) and safetensors/src/tensor.rs (dtype names, header padded with spaces to 8 bytes), https://github.com/huggingface/safetensors, read 2026-11-06.
- `saltelli2002`: A. Saltelli, "Making best use of model evaluations to compute sensitivity indices", Computer Physics Communications 145(2), 280–297, 2002.
- `saltelli2010`: A. Saltelli, P. Annoni, I. Azzini, F. Campolongo, M. Ratto and S. Tarantola, "Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index", Computer Physics Communications 181(2), 259–270, 2010 (Table 2: the estimators).
- `schlick1994`: C. Schlick, "An inexpensive BRDF model for physically-based rendering", Computer Graphics Forum 13(3), 233–246, 1994 (the Fresnel approximation, eq. 15).
- `schott_bk7`: SCHOTT optical glass catalogue 2017-01-20b, N-BK7 Sellmeier coefficients (via the refractiveindex.info database, CC0, read 2026-11-07).
- `scipy_brentq`: SciPy, scipy.optimize.brentq (Brent's root finder, an independent solver), https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html.
- `scipy_qmc`: SciPy, scipy.stats.qmc (Sobol with Owen scrambling, discrepancy), https://docs.scipy.org/doc/scipy/reference/stats.qmc.html.
- `scipy_quad`: SciPy, scipy.integrate.quad (adaptive quadrature, QUADPACK), https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html.
- `scipy_rotation`: SciPy, scipy.spatial.transform.Rotation (an independent implementation of rotation conversions), https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html.
- `scipy_stats`: SciPy, scipy.stats (kstest, uniform, truncnorm, norm, spearmanr: independent implementations), https://docs.scipy.org/doc/scipy/reference/stats.html.
- `scipy_stats_tests`: SciPy 1.18, scipy.stats: binomtest(...).proportion_ci (wilson, exact), bootstrap (percentile, BCa), permutation_test, ttest_ind (equal_var=False), ttest_rel, mannwhitneyu, wilcoxon, fisher_exact, contingency.odds_ratio (kind="sample"): independent implementations, https://docs.scipy.org/doc/scipy/reference/stats.html (read 2026-11-07).
- `scipy_wasserstein`: SciPy, scipy.stats.wasserstein_distance (the first Wasserstein distance between two 1-D distributions, an independent implementation), https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html.
- `shah2013`: M. Shah, "Solving the robot-world/hand-eye calibration problem using the Kronecker product", Journal of Mechanisms and Robotics 5(3), 031007, 2013.
- `shapley1953`: L. S. Shapley, "A value for n-person games", in H. W. Kuhn and A. W. Tucker (eds.), Contributions to the Theory of Games II, Annals of Mathematics Studies 28, Princeton University Press, 307–317, 1953.
- `shepperd1978`: S. W. Shepperd, "Quaternion from rotation matrix", Journal of Guidance and Control 1(3), 223–224, 1978.
- `shoemake1985`: K. Shoemake, "Animating rotation with quaternion curves", SIGGRAPH '85, Computer Graphics 19(3), 245–254, 1985.
- `siunitx_wright2026`: J. Wright, "siunitx - A comprehensive (SI) units package", package manual, version 3.6.7, CTAN, 2026 (section 3.4: literal unit input such as \unit{kg.m/s^2}; \num, \qty, \percent, \degree), https://ctan.org/pkg/siunitx, read 2026-11-07.
- `skimage_ssim`: scikit-image 0.26.0, skimage.metrics.structural_similarity (gaussian_weights, sigma 1.5, use_sample_covariance False and data_range to match Wang et al. 2004), https://scikit-image.org/docs/stable/api/skimage.metrics.html (read 2026-11-08).
- `slurm_job_array`: SchedMD, Slurm Workload Manager, "Job Array Support", version 26.05 (--array ranges, comma lists and the % throttle; SLURM_ARRAY_TASK_ID; %A and %a; MaxArraySize default 1001), https://slurm.schedmd.com/job_array.html, read 2026-11-07.
- `slurm_sbatch`: SchedMD, Slurm Workload Manager, "sbatch", version 26.05 (--cpus-per-task, --mem in mebibytes with K, M, G, T, --time formats, --parsable, --dependency=afterany, SLURM_SUBMIT_DIR; #SBATCH directives are read up to the first non-comment, non-whitespace line), https://slurm.schedmd.com/sbatch.html, read 2026-11-07.
- `smirnov1948`: N. Smirnov, "Table for estimating the goodness of fit of empirical distributions", The Annals of Mathematical Statistics 19(2), 279–281, 1948.
- `sobol1993`: I. M. Sobol', "Sensitivity estimates for nonlinear mathematical models", Mathematical Modelling and Computational Experiments 1(4), 407–414, 1993.
- `spearman1904`: C. Spearman, "The proof and measurement of association between two things", The American Journal of Psychology 15(1), 72–101, 1904 (the rank correlation).
- `spong2006`: M. W. Spong, S. Hutchinson and M. Vidyasagar, Robot Modeling and Control, Wiley, 2006 (standard DH link transform Rz(θ) Tz(d) Tx(a) Rx(α)).
- `steiger_fouladi1997`: J. H. Steiger and R. T. Fouladi, "Noncentrality interval estimation and the evaluation of statistical models", in L. L. Harlow, S. A. Mulaik and J. H. Steiger (eds.), What If There Were No Significance Tests?, Erlbaum, 221-257, 1997.
- `step_ap`: ISO 10303-21, Industrial automation systems and integration: product data representation and exchange, clear text encoding of the exchange structure (STEP), as written and read by OpenCASCADE through gmsh.
- `sugihara2011`: T. Sugihara, "Solvability-unconcerned inverse kinematics by the Levenberg–Marquardt method", IEEE Transactions on Robotics 27(5), 984–991, 2011.
- `sultanova2009`: N. Sultanova, S. Kasarova and I. Nikolov, "Dispersion properties of optical polymers", Acta Physica Polonica A 116, 585–587, 2009 (PMMA; via the refractiveindex.info database, CC0, read 2026-11-07).
- `sutherland_hodgman1974`: I. E. Sutherland and G. W. Hodgman, "Reentrant polygon clipping", Communications of the ACM 17(1), 32–42, 1974.
- `svg11`: W3C, Scalable Vector Graphics (SVG) 1.1 (Second Edition), W3C Recommendation 16 August 2011, https://www.w3.org/TR/SVG11/, read 2026-11-07.
- `tai_yang2008`: S.-C. Tai and S.-M. Yang, "A fast method for image noise estimation using Laplacian operator and adaptive edge detection", 3rd International Symposium on Communications, Control and Signal Processing (ISCCSP 2008), 1077–1081, doi:10.1109/ISCCSP.2008.4537384 (abstract read 2026-11-07: Sobel edge pixels left out of Immerkær's estimate).
- `tf2_static_publisher`: ros2/geometry2 at commit 183399c, tf2_ros/src/static_transform_broadcaster_program.cpp (--x --y --z --qx --qy --qz --qw --frame-id --child-frame-id), https://github.com/ros2/geometry2, read 2026-11-07.
- `tfrecord_format`: TensorFlow TFRecord format: uint64 length, uint32 masked CRC-32C of the length, data, uint32 masked CRC-32C of the data; mask = rotate right 15 + 0xa282ead8 (xla/tsl/lib/io/record_writer.h and xla/tsl/lib/hash/crc32c.h, https://github.com/openxla/xla; https://www.tensorflow.org/tutorials/load_data/tfrecord), read 2026-11-07.
- `tol_colours`: P. Tol, Paul Tol's Notes, "Qualitative colour schemes" (the bright scheme: #4477AA, #EE6677, #228833, #CCBB44, #66CCEE, #AA3377, #BBBBBB), https://sronpersonalpages.nl/~pault/ (page dated 16 August 2026), read 2026-11-07.
- `torchvision_centercrop`: torchvision.transforms.functional.center_crop (crop_top = int(round((image_height − crop_height) / 2.0)), likewise the left), torchvision/transforms/functional.py of https://github.com/pytorch/vision (main branch), read 2026-11-06.
- `treece1999`: G. M. Treece, R. W. Prager and A. H. Gee, "Regularised marching tetrahedra: improved iso-surface extraction", Computers & Graphics 23(4), 583–598, 1999.
- `triggs2000`: B. Triggs, P. McLauchlan, R. Hartley and A. Fitzgibbon, "Bundle adjustment — a modern synthesis", Vision Algorithms: Theory and Practice, LNCS 1883, 298–372, Springer, 2000.
- `tsai_lenz1989`: R. Y. Tsai and R. K. Lenz, "A new technique for fully autonomous and efficient 3D robotics hand/eye calibration", IEEE Transactions on Robotics and Automation 5(3), 345–358, 1989.
- `unitree_g1_urdf`: Unitree Robotics, unitree_ros robots/g1_description/g1_29dof_rev_1_0.urdf at commit 7c40519e02d7dd16c06b25fe3fa3b67fdeb7cd74, joint "d435_joint" (torso_link to d435_link: xyz 0.0576235 0.01753 0.42987 m, rpy 0 0.8307767239493009 0 rad), https://github.com/unitreerobotics/unitree_ros (read 2026-11-07).
- `ur5e_flange`: Universal Robots, UR5e datasheet, drawing "Tool flange for UR5e" (EN ISO-9409-1-50-4-M6), https://www.universal-robots.com/manuals/latest/en/datasheets/ur5e/, read 2026-11-06.
- `ur_dh`: Universal Robots, "DH Parameters for calculations of kinematics and dynamics", UR5e/UR7e table, https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/, read 2026-11-06.
- `urdf`: ROS URDF XML specification, the <origin rpy="…"> element: roll, pitch, yaw about fixed x, y, z axes, http://wiki.ros.org/urdf/XML/joint.
- `urdfdom`: ros/urdfdom at commit bfcf29f (urdf_parser/src/joint.cpp: <limit> needs effort and velocity) and ros/urdfdom_headers (urdf_model/pose.h, Rotation::setFromRPY: fixed-axis roll, pitch, yaw), https://github.com/ros/urdfdom, read 2026-11-07.
- `usb20_spec`: Universal Serial Bus Specification, Revision 2.0, April 27, 2000 (Table 5-10 high-speed bulk transaction limits, §7.2.1 classes of devices), https://www.usb.org/sites/default/files/usb_20_20250603.zip, read 2026-11-06.
- `usb32_spec`: USB 3.0 Promoter Group, Universal Serial Bus 3.2 Specification, Revision 1.1, June 2022 (Table 3-1 bus power, §4.4.11 efficiency, §10.14 hub configuration and §11.4 power distribution), https://www.usb.org/sites/default/files/usb_32_202206_0.zip, read 2026-11-06.
- `vantage_rig_gap`: Vantage core/rig_gap.py (the Rig Contrastive Distance: components in 0..1, their weights, the Hungarian camera match, the heuristic transfer bands) and core/fleet.py (measured minus planned, the ok / adjust / re-mount verdict, the robust z-score, the systematic offset), the read-only reference implementation DexVantage ports from.
- `veach1997`: E. Veach, Robust Monte Carlo Methods for Light Transport Simulation, PhD thesis, Stanford University, 1997 (multiple importance sampling, chapter 9).
- `walter2007`: B. Walter, S. R. Marschner, H. Li and K. E. Torrance, "Microfacet models for refraction through rough surfaces", Eurographics Symposium on Rendering 2007, 195–206 (the GGX distribution, eq. 33; Smith G1, eq. 34; the microfacet BRDF, eq. 20; sampling, eqs. 35–36).
- `wampler1986`: C. W. Wampler, "Manipulator inverse kinematic solutions based on vector formulations and damped least-squares methods", IEEE Transactions on Systems, Man, and Cybernetics 16(1), 93–101, 1986.
- `wang2004_ssim`: Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli, "Image quality assessment: from error visibility to structural similarity", IEEE Transactions on Image Processing 13(4), 600–612, 2004.
- `wang_olson2016`: J. Wang and E. Olson, "AprilTag 2: Efficient and robust fiducial detection", IEEE/RSJ IROS 2016, 4193–4198.
- `williams1949`: E. J. Williams, "Experimental designs balanced for the estimation of residual effects of treatments", Australian Journal of Scientific Research, Series A 2(2), 149-168, 1949, https://doi.org/10.1071/CH9490149 (read 2026-11-07).
- `williams1978`: L. Williams, "Casting curved shadows on curved surfaces", SIGGRAPH '78, Computer Graphics 12(3), 270–274, 1978 (shadow maps and their depth bias).
- `wilson1927`: E. B. Wilson, "Probable inference, the law of succession, and statistical inference", Journal of the American Statistical Association 22(158), 209–212, 1927 (the score interval of a proportion).
- `yoshikawa1985`: T. Yoshikawa, "Manipulability of robotic mechanisms", The International Journal of Robotics Research 4(2), 3–9, 1985.
- `zhao2023_act`: T. Z. Zhao, V. Kumar, S. Levine and C. Finn, "Learning fine-grained bimanual manipulation with low-cost hardware" (ACT, temporal ensembling w_i = exp(-m i)), RSS 2023, https://arxiv.org/abs/2304.13705.
- `zip_appnote`: PKWARE, APPNOTE.TXT, .ZIP File Format Specification (read here through Python's zipfile).
- `zitzler2000_zdt`: E. Zitzler, K. Deb and L. Thiele, "Comparison of multiobjective evolutionary algorithms: empirical results", Evolutionary Computation 8(2), 173–195, 2000 (ZDT1 and ZDT2 and their analytic fronts, g = 1).
- `zwicker2001_ewa`: M. Zwicker, H. Pfister, J. van Baar and M. Gross, "EWA volume splatting", IEEE Visualization 2001, 29–36 (the local affine approximation of the projection of a Gaussian).
