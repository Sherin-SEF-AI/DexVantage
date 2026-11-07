# Teleop overlap and follower dexterity

Spec §7.4. Module `dexvantage/core/studies/teleop.py` (study `teleop`, batch flag `--teleop OUT.json`, or `--teleop-overlap OUT.json`, the specification's name), page
`dexvantage/ui/pages/teleop.py` (Analysis ▸ Teleop overlap, the Teleop workspace), checks
`dexvantage/validation/checks_teleop.py`.

The question: where in the task volume can a teleoperator drive the follower robot, and where is the operator
comfortable while the follower reaches badly (a **dead zone**: near a singularity, a joint limit or itself, or not at
all)?

## Teleop devices

`project.tasks.teleop_devices` holds one record per device (the Phase 5 shared contract; `teleop.clean_device`
normalises it):

| Field | Meaning |
|---|---|
| `kind` | `leader_arm`, `gello`, `vr`, `spacemouse`, `glove` or `hand_tracker` (the last two cartesian like VR: their volume is the wrist tracker's or the tracker's own) |
| `follower_uid` | the robot of the project the device drives |
| `leader` | `{'robot_uid'}`: a robot of the project whose kinematics the leader has (the follower's own uid: a leader of the same model); `{'library_id'}`: a library teleop entry that is not stated to lead this follower (it carries figures, no kinematics: the study says so and asks for the leader as a robot); `null` |
| `library_id` | optional: the library teleop entry the device was made from (below) |
| `mapping` | `kind` `joint` or `cartesian`, `scale`, `offset` (a pose of the device frame in the follower's base frame), `joint_map` {leader joint: follower joint} |
| `volume` | VR tracking volume or space-mouse bounds in the device frame: a box (`size` = three edges) or a cylinder (`size` = [radius, height], axis z); `null` = not stated |
| `source` | where each device figure comes from (a URL and the date read, or "measured") |

Nothing is invented. A VR headset's tracking volume or a space mouse's bounds are not published as volumes by their
makers; a new VR or space-mouse device has `volume: null`, and the study says the device's workspace is not stated
until the user enters a measured or cited one. A leader arm or GELLO leads with a robot model of the project; for the
SO-100 kit and GELLO that model is the follower's own:

- LeRobot's SO-100 guide (https://huggingface.co/docs/lerobot/so100, read 2026-11-07) builds the leader arm with the
  follower's assembly steps 1–23 (through the wrist; a handle and trigger replace the gripper) and calibrates both
  arms "to ensure that the leader and follower arms have the same position values when they are in the same physical
  position". The SO-100 sample carries this device (`SO-100 leader arm (sample)`).
- GELLO (Wu et al. 2023, arXiv:2309.13037): "Given a target robot arm, we build a GELLO controller device that has the
  same kinematic structure as the target arm". Its joint ranges and link lengths are not stated in the abstract; the
  follower's model is used.

## Teleop devices from the library {#library}

The library kind `teleop` (`dexvantage/resources/library/teleop/*.json`, `core/teleop_devices.py`, schema block
`teleop`) holds 13 devices, every figure read on 2026-11-07 from the source it names, each `{value, source, quote}`
and null when no source states it. The Teleop page's **Add from library** makes a device record from one
(`teleop_devices.device_from_library(project, library_id, follower_uid)`, one undo step); the plain **Add** works as
before.

| Entry | Kind | Stated figures (source) | Not stated |
|---|---|---|---|
| GELLO | gello, joints of the target arm (`leads` "*") | joint resolution 0.088° (12-bit encoders; paper) | DoF (the target arm's), rate, latency, mass |
| SO-100 leader arm | leader arm, same joints as an SO-100 | 6 motors (LeRobot docs) | mass, rate, latency |
| SO-101 leader arm | leader arm, same joints as an SO-101 | 6 axes, gear ratios (LeRobot docs) | mass, rate, latency |
| Koch v1.1 leader arm | leader arm, angles mapped one to one (own parts) | 6 XL330-M077-T servos, 5 V (README) | its joint ranges, mass, rate |
| ALOHA leader (WidowX-250 6DOF) | leader arm, angles mapped to a ViperX ("approximately scaled versions") | 6 DoF, reach 650 mm, accuracy 5–8 mm (Trossen); teleoperation at 50 Hz (ALOHA paper) | latency, mass |
| SpaceMouse Compact | space mouse | 6 DoF, 480 g, 77 × 77 × 54 mm | cap travel, rate |
| Meta Quest 3, 3S | VR | 515 g, 514 g (excluding the facial interface); hand tracking by 4 IR + 2 RGB cameras | tracking volume, tracking rate, latency |
| Apple Vision Pro (M5) | VR | 12 ms photon-to-photon (R1 chip), 750–800 g | hand-tracking rate, volume |
| MANUS Quantum Metagloves | glove | 120 Hz sample rate, ≤ 7.5 ms signal latency, 138 g, 15 m wireless | accuracy figure, wrist volume |
| Rokoko Smartgloves II | glove | 250 Hz IMU rate, 0.47 mm wrist accuracy at fingertip distance, 7 sensors | latency, mass |
| HaptX Gloves G1 | glove | 36 tracked DoF, 0.3 mm rms resolution, ~1 lb, 135 tactile actuators per hand | latency ("no perceivable latency"), rate |
| Leap Motion Controller 2 | hand tracker | 115 fps camera maximum, 10–110 cm, 160° × 160°, 29 g, 850 nm | latency, accuracy |

How a record is made from an entry (nothing is invented):

- **Leader arms and GELLO** map joint to joint. When the entry is stated to lead the follower robot (`leads`: words
  matched against the robot's name and model file, ignoring case, spaces, "-" and "_"; GELLO's "*" leads any arm), the
  leader is the follower's own model. With `same_joints` true (SO-100, SO-101, GELLO) the source states the leader has
  the follower's joints; with false (Koch v1.1, ALOHA's WidowX) only the angles are mapped one to one, and the leader's
  comfortable box is taken from the follower's ranges, an ASSUMPTION written into the record's `source.leader`. A
  leader not stated to lead the follower gets `{'library_id'}`, and the study asks for the leader arm as a robot.
- **VR headsets, space mice, gloves and hand trackers** are cartesian. Their volume is the entry's stated
  `workspace`; no maker states one as a box or cylinder (the Leap Motion's is a 160° cone from 10 to 110 cm, a headset's
  play area is not a tracking volume, a glove's wrist is tracked by another system), so the record's volume is null and
  the study says "not stated" until a measured or cited volume is entered on the page.
- Every stated figure goes into the record's `source` with its URL, the date read and the quote; the record keeps
  `library_id`, the page shows which figures the maker does not state, and the study's device rows carry the id.

## The comfortable workspace in the follower's frame

**Joint mapping** (leader arm, GELLO). For each `joint_map` entry (default: joints of the same name),
`q_f[j_f] = scale · q_l[j_l]`; follower joints the map leaves out are held at home. The leader is comfortable inside
the central `comfort_fraction` of each joint's range (default **0.8**, a DexVantage default; not a human-factors
measurement): `[mid − f·(hi − lo)/2, mid + f·(hi − lo)/2]`, an unlimited revolute joint `home ± f·π`. The comfortable
workspace in the follower's frame is

    W = { FK_f(q) : q ∈ scale · C }       C = the leader's comfortable box

A task voxel is comfortable when the follower's damped-least-squares IK (`core/robot/ik.py solve_batch`, position
target, orientation free) converges with the joint bounds replaced by `scale · C`. The bounds are deliberately not cut
to the follower's own limits: a mapped range that runs past a follower stop leaves the follower at the stop while the
leader is comfortable, a joint-limit dead zone.

**Cartesian mapping** (VR, space mouse, or an arm mapped by its tool position). A device-frame point `p_d` maps to the
world as

    p_world = B_f · O · (s · p_d)          B_f the follower's base, O the offset pose, s the scale

and back by `p_d = (B_f · O)⁻¹ p_world / s`. A voxel is comfortable when its centre maps back inside the device's box
or cylinder, or, for an arm, when the leader's IK (inside its comfortable box) reaches `p_d` in the leader's base frame.
The follower then solves its own IK.

## The follower's measures

The task volumes are voxelised with `core/voxels.py` (`SparseGrid`, `voxel_mm` default **25 mm**, no refinement:
every voxel is an IK solve), plus `search_voxels` (default 2) layers round each volume for the suggestion. At each
voxel centre the follower's tool point (gripper frame and TCP from `core/grasps.gripper_geometry`) is solved by IK
(position only; 4 seeds, 4 × 4 fresh seeds for a target none solves; the solution nearest home, as a retargeting IK
started from home finds it; self-collision avoided). At the solution, with J the tool point's Jacobian over the
chain's joints (MuJoCo `mj_jac`; the position rows by default, `jacobian: 'full'` for all six):

    w   = Π σᵢ(J)                  = √det(J Jᵀ) when J has no more rows than columns   [Yoshikawa 1985]
                                   = √det(Jᵀ J) otherwise (a planar arm in its plane)
    1/κ = σ_min / σ_max            0 singular, 1 isotropic
    d_lim = min_j 2·min(q_j − lo_j, hi_j − q_j)/(hi_j − lo_j)
            over the chain's joints whose Jacobian column exceeds min_column_mm (1 mm/rad): a wrist roll about the
            tool axis cannot block a position
    d_self = the self-collision distance (core/robot/collision.py) over link pairs the chain's joints move against
             each other (the 2F-85's own linkage, 3.6 mm apart at every arm pose, is left out), capped at 100 mm

The full Jacobian mixes metres and radians, so its κ depends on the length unit (the usual caveat of Yoshikawa's
measure); the position default is unit-consistent and matches the position-only reach.

## Dead zones

A voxel the leader reaches comfortably is a dead zone when (labelled defaults, the user's settings):

| Cause | Test | Default |
|---|---|---|
| unreachable | no follower IK solution | — |
| self-collision | `d_self < self_margin_mm` | 10 mm |
| singularity | `1/κ < min_inverse_condition` | 0.1 (κ > 10) |
| joint limit | `d_lim < min_limit_distance` | 0.05 (within 2.5 % of the range from a stop) |

A voxel's first cause in that order names it. Connected dead-zone voxels of one cause (6-neighbours,
`scipy.ndimage.label`) form a zone, listed with its volume, its box and the follower's joint values at its worst voxel
(the smallest distance, 1/κ or `d_lim`; for unreachable zones the voxel nearest the zone's centre). **Teleoperable**
= comfortable and no cause. The headline is the share of task voxels teleoperable through at least one device;
status "clear" at 90 % or more, "gaps" from 50 %, "poor" below (labelled bands).

## Bimanual

With two or more follower robots among the devices, for each pair: the share of the task volume both reach, the share
teleoperable by both, and a handover test at the voxels of `handover` task volumes (else up to `handover_samples`,
default 60, of the voxels both reach, seed 0): each arm reaches a point `handover_gap_mm` (100 mm) apart along the line
between their bases; the handover quality is the share where both reach without a dead-zone cause and keep
`arm_margin_mm` (20 mm) apart. The arm-to-arm margin is the smallest distance between the two arms' colliding-geometry
vertices (about 4 000 per arm, `scipy.spatial.cKDTree`): an upper bound of the true distance by up to the vertex
spacing, and a margin near zero may be an overlap.

## The suggestion

A follower's dexterity map and its devices' comfortable regions are rigid with its base, so moving the base by whole
voxels `s·h` shifts them over the task voxels: the teleoperable share after the move is read from the grid at
`index − s` without new IK. A cartesian device's mapping offset shifts only its comfortable region. Every move within
±`search_voxels` per axis is scored; the one that most raises the headline (fewer dead-zone voxels, then the smaller
move, break ties) is suggested with the predicted share and dead-zone volume. Apply makes it in one undo step
(`apply_suggestion`): a non-primary robot's base moves; the primary robot's base is the world frame, so its move is
applied as the opposite move of the task volumes, objects, surfaces, lights, fixed sensors and other robots (hand-eye
board placements are not moved). A locked fixed sensor blocks that, with a sentence saying why.

## Validation

`validation/checks_teleop.py`, measured 2026-11-07 (full run, default seed, CPU only: the study renders nothing):

| Check | Reference | Result | Tolerance |
|---|---|---|---|
| `teleop.manipulability_2r` | w = \|l₁ l₂ sin q₂\| and 1/κ from the eigenvalues of J Jᵀ of the analytic 2R Jacobian, at 1 000 random q (singular poses included) and the study's IK solutions of 250 reachable points | w 1.4 × 10⁻¹⁶ m², 1/κ 2.4 × 10⁻¹³; 250 of 250 solved | 10⁻¹² m² (rounding of products of order 0.1); 1/κ 10⁻⁹ |
| `teleop.jacobian_fd` | central differences (step 10⁻⁶ rad) of the tool point posed by the independent PoE chain (no MuJoCo) | SO-100 3.8 × 10⁻⁹, UR5e 2.1 × 10⁻⁹ relative (60 random q each) | 10⁻⁶ (truncation ~10⁻¹²; rounding of the difference ~10⁻¹⁰/10⁻⁶) |
| `teleop.leader_workspace` | 120 000 uniform joint vectors of the SO-100's comfortable box posed by the PoE chain; the voxel centres a sample passes within 5 mm of | 0 of 274 such voxels missed; 0 of 920 comfortable voxels without a valid pose (inside the box, PoE tool point within 0.2 mm) | 0.05 (voxel centres within 5 mm of the boundary; local-IK misses) |
| `teleop.cartesian_mapping` | points drawn in the device frame (box and cylinder, random rotated offset, scale 0.4–2.5, turned base) and the analytic volume s³·V | 0 misclassified; Monte Carlo volume 0.19 and 0.019 of 3 standard errors; round trip 4.4 × 10⁻¹⁶ m | 1 (3 σ) |

The leader-workspace check first ran with one round of fresh seeds per failed target (as many as the first round): 8 of
274 voxels were missed, and a 32-seed search found all 8 inside the box (local-IK misses, none on the boundary). The
study now gives a failed target 4 × seeds fresh seeds; the check reports the same split every run.

## Performance

Measured on the samples (batch, CPU, times include start-up): SO-100 sample (480 task voxels at 25 mm, 1 728 points
with the search layers) 7.4 s; with a second, VR device 7.2 s; UR5e sample with no device 12.2 s, with a leader arm
about 18–26 s (7 680 points, two IK passes). A bimanual pair of SO-100s with a 60-voxel handover test 15 s. Most of the
time is the IK (MuJoCo Jacobians, one per seed and iteration). On the page every UI step stays under 3 ms (showing a
UR5e result with every voxel a dead zone: 1.1 ms; its 46 080 overlay triangles: 2.4 ms); the study runs on the
analysis controller's worker.

## Known limits

Reach is position-only with a free orientation (a task that needs a given gripper orientation can be harder); the IK
is local, so a voxel every seed misses reads as unreachable or outside the comfortable workspace; the comfortable
fraction of the joint ranges is a default, not a human-factors model, and the comfortable workspace is a set of tool
positions, not the operator's posture or speed; dexterity is first order at one pose; the suggestion searches lattice
translations only (no rotations, no finer steps) and its prediction holds only to the voxel size; obstacles and other
robots are not avoided by the reach (self-collision is); arm-to-arm distances are between mesh vertices (an upper
bound); the SO-100's joint limits are Menagerie's approximate ones, so its comfortable box inherits that; the VR and
space-mouse figures are whatever the user states, and a device without them is left out. Library devices: no maker
states a tracking volume as a box or cylinder, so a library VR, glove, hand-tracker or space-mouse device reaches the
study without a comfortable region until one is entered; the stated rates and latencies (a glove's sample rate and
signal latency, a headset's photon-to-photon passthrough) are carried as figures with their sources and are not used
by the reach model; a leader whose own joint ranges are not stated (Koch v1.1, ALOHA's WidowX) is given the
follower's ranges, an assumption its record says; `leads` matches words of the robot's name and model file, so a
renamed robot is not recognised (choose the leader by hand).

## References

- T. Yoshikawa, "Manipulability of robotic mechanisms", IJRR 4(2), 1985 (`yoshikawa1985`).
- K. M. Lynch and F. C. Park, Modern Robotics, 2017 (the PoE chain of the checks, `lynch_park2017`).
- Hugging Face LeRobot, SO-100 guide, https://huggingface.co/docs/lerobot/so100, read 2026-11-07 (`lerobot_so100`).
- P. Wu et al., GELLO, arXiv:2309.13037, 2023 (`gello2023`).
- The library teleop entries' sources, each with the date read (`library_teleop_sources`; listed in each entry and in
  [library.md](library.md#library-systems)).
