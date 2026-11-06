# DexVantage

Camera, depth and teleop planning for robot-learning data collection.

DexVantage is a desktop application for robotics R&D teams who build manipulation and humanoid robots and collect
data for robot-learning policies (ACT, Diffusion Policy, VLA models such as OpenVLA, π0 and GR00T). It answers, before
a rig is built and again after it records:

- where the wrist, head, chest, overhead and third-person cameras go so the fingertips, the object and the contact are
  visible at the moment of grasp;
- whether depth is valid where the policy needs it (minimum range, finger shadows, glossy or transparent surfaces);
- whether the teleoperator can reach poses the follower cannot, and where cables, USB/GMSL bandwidth and time sync
  break;
- whether a recorded dataset (LeRobot, RLDS / Open X-Embodiment, ROS 2 MCAP) meets the plan.

It is the sibling of Vantage, the sensor-rig planner for autonomous-driving data collection, with the same design
language and engineering standards: every number comes from a model that a validation check compares with an
independent reference, and every export carries a provenance block.

> **Status: Phases 0–3, 6 and 7 of 12.** Robots import from URDF, MJCF and MuJoCo Menagerie, with forward and
> inverse kinematics and three bundled samples; a datasheet library of cameras, depth cameras, tactile sensors, hosts,
> cables and robot-learning policies; sensors on any link with a Camera view through each lens model; coverage, depth
> validity, grasp self-occlusion with the Contact Visibility Score, VLA token resolution, the multi-finger occlusion
> matrix, bandwidth and power, cable flex, printable brackets and hand-eye calibration. Phases 4 (MuJoCo ground
> truth), 5 (teleop, lighting, blur and sync) and 8 (data) are next. See [docs/BUILD_STATE.md](docs/BUILD_STATE.md).

## Install (from source)

| Platform | Steps |
|---|---|
| Linux (x86_64, arm64 / Jetson) | `python3 -m venv .venv && .venv/bin/pip install -e .[test]` |
| macOS (Apple silicon) | `python3 -m venv .venv && .venv/bin/pip install -e .[test]` |
| Windows | `py -m venv .venv && .venv\Scripts\pip install -e .[test]` |

Python 3.10 or newer. Packaged bundles (PyInstaller) arrive in Phase 11.

## Quick start

```bash
dexvantage                          # the window, with an empty project
dexvantage --sample ur5e            # the UR5e + Robotiq 2F-85 sample (or so100 for the SO-100 arm)
dexvantage --sample so100 --analyse -   # a sample's summary as JSON, in batch mode
dexvantage --sample ur5e --coverage cov.json        # coverage and resolution of the task volume
dexvantage --sample ur5e --depth-validity dv.json   # depth validity of the wrist D405 and overhead D435
dexvantage --sample ur5e --grasp grasp.json         # grasp self-occlusion: the Contact Visibility Score
dexvantage --sample ur5e --vla-tokens vla.json      # ViT tokens per object and fingertip for the sample's π0
dexvantage --sample dexhand --dex-occlusion d.json  # the finger × camera occlusion matrix of a Shadow Hand
dexvantage --sample ur5e --bandwidth bw.json        # data rates, controller margins, disk and bus power
dexvantage --sample ur5e --cables cables.json       # cable slack, bends and snags across the joints
dexvantage --sample ur5e --brackets br.json         # printable brackets and their clearance
dexvantage --sample ur5e --handeye-plan plan.json   # a hand-eye calibration pose plan with predicted σ
dexvantage --self-test              # check this installation without a window
dexvantage cell.dexv --analyse -    # the project summary as JSON, with provenance
dexvantage --validate report/       # the validation suite and its HTML report
python -m pytest tests              # the tests (offscreen)
```

## What works now

| Area | State | Measured |
|---|---|---|
| Window shell: 14 menus, docks, six workspaces, command palette (Ctrl+K), Find (Ctrl+F), shortcuts (?) | done | — |
| 3D viewport: Vantage's GLSL renderer and orbit-about-cursor navigation, robot-cell scale | done | renders headless through EGL |
| `.dexv` project file, JSON Schema, migrations, undo/redo | done | round trip: 0 differences |
| Provenance (`dexvantage.provenance/1`), `--reproduce` | done | — |
| Robot import: URDF, MJCF (meshes, includes, textures stored in the project), import wizard and taggers | done | — |
| MuJoCo Menagerie import with each model's own licence shown and recorded | done | — |
| Forward kinematics (MuJoCo) checked by an independent product-of-exponentials chain and DH closed forms | done | agreement 8.7 × 10⁻¹⁵ m on both samples; DH 6.3 × 10⁻¹⁶ m |
| Inverse kinematics (damped least squares, limits, seeds, IK drag) | done | 150 of 150 round trips within 0.1 mm; about 4 ms per target |
| Manipulability, self-collision distances | done | 2R closed form 1.1 × 10⁻¹⁶ m²; sphere and capsule distances 1.1 × 10⁻¹⁶ m |
| Joint sliders, pose scrubber, trajectories from CSV | done | — |
| Samples: SO-100, UR5e + Robotiq 2F-85, UR5e + Shadow Hand (MuJoCo Menagerie, licences checked) | done | — |
| Hardware library from datasheets: 21 depth cameras, 9 cameras, 2 tactile sensors, 8 hosts; every figure with its source and date, gaps "not stated by the maker" | done | 40 of 40 entries validate against the schema |
| Lens models: pinhole, Brown-Conrady, Kannala-Brandt, equidistant, equisolid, stereographic; pixel density and defocus | done | against OpenCV `projectPoints` 1.8 × 10⁻¹² px, `fisheye.projectPoints` 5.7 × 10⁻¹³ px; MuJoCo-rendered fiducials 0.17 px |
| Sensors and mounts: any robot link, pose in mm and degrees, stream modes, custom devices | done | — |
| Articulated cube-map occlusion against the posed robot (GPU through EGL, or numpy) | done | 0.19 % of 480 000 points disagree with exact ray casting, all at silhouettes; MuJoCo segmentation on the UR5e 99.81 %; 13–16 ms per map on an RTX 5080 |
| Coverage and resolution study (`--coverage`): visibility, px/mm, viewing angle, defocus, redundancy on sparse voxels | done | UR5e sample, 480 000 voxels: 0.6 s; against exact ray casting 0.09 % of the volume |
| Depth validity study (`--depth-validity`): stereo FOV overlap, min-Z per resolution, finger shadows, projector cone, materials, σ_z | done | Intel's min-Z and FOV tables exact; σ_z against Monte Carlo 0.7 %; UR5e sample 0.34 s |
| Camera view: the selected sensor's image through its lens model, posed with the robot | done | UR5e image 78 ms on a worker thread; UI thread 0.2 ms |
| UI-thread budget (no step over 100 ms on the samples) | met | worst: a UR5e frame after a joint change, 68 ms median, 96 ms worst ([performance](docs/models/performance.md)) |
| Grasp sets: sampled, imported, from a trajectory; contact pads measured from the gripper | done | pads against MuJoCo's pad boxes 3.6 × 10⁻¹¹ m |
| Grasp self-occlusion (`--grasp`): batched IK, contact visibility, Contact Visibility Score with bootstrap and Wilson intervals, gallery, what to move | done | 2 000 grasps in 111.6 s on an RTX 5080 (131.3 s on an integrated Radeon); against exact ray casting 0 of 7 892 points away from an edge |
| Policies: 8 cited entries (OpenVLA, π0, π0-FAST, SmolVLA, ACT, Diffusion Policy, Octo, GR00T N1), crop rules | done | crop geometry against each rule's own code 2.5 × 10⁻¹³ px |
| VLA token resolution (`--vla-tokens`), patch grid in the Camera view, Plücker ray-map export | done | token counts against ray casting: 0 tokens off; Plücker moments 1.3 × 10⁻¹⁵ |
| Multi-finger occlusion matrix (`--dex-occlusion`) | done | against a closed-form projection: 0 |
| Bandwidth, host topology, disk and USB power (`--bandwidth`) | done | budget re-computed independently: 3.9 × 10⁻¹⁶ |
| Cable flex across joints (`--cables`), harness sheet | done | routed arc against closed forms 1.7 × 10⁻¹⁶ m; snags against brute force: 0 poses |
| Brackets (`--brackets`): ISO 9409-1 adapters, T-slot clamps, pan-tilt, palm and dorsal sleeves; watertight STL; clearance | done | 0 mesh defects; ISO holes 4.2 × 10⁻⁵ mm; flange volume 0.18 % |
| Hand-eye calibration: ChArUco and AprilGrid, AX = XB and AX = ZB, bundle adjustment, pose planner, coach (`--handeye-plan`, `--handeye-solve`) | done | rendered corners 0.11 px; covariance within the Monte Carlo χ² band; planner σ against the adjustment's within 0.9 % |
| Validation suite | 72 checks | 72 of 72 pass |
| MuJoCo ground truth, teleop, lighting, blur, sync, data audit, optimisation, exporters, assistant | Phases 4, 5, 8–12 | — |

## Known limits

Studies of later phases (MuJoCo ground truth, teleop, lighting, blur, sync, data audit) are in the menus, disabled,
with the phase that brings them. Grasps: the fingers stay at their home opening and obstacles are not checked along
the approach; the dexterous-hand postures are sample sweeps, not measured synergies. Hand-eye: OpenCV 5.0.0 has no
`calibrateHandEye` bindings, so DexVantage's own solvers have not been compared with OpenCV's. Sensors: most makers state no lens distortion, so built-in imagers are pinhole
models fitted to the stated FOV until you calibrate a unit; defocus is computed only where a lens's focus distance
and f-number are stated or entered; sensor housings do not occlude; the Camera view has no study overlays yet.
Robots: xacro must be expanded to URDF
first and USD is not read; mimic joints and the 2F-85's linkage move independently; free and ball joints are held at
the home pose; the gripper frame must be a link. ALOHA 2, the bimanual Panda and the G1 samples arrive in Phase 11.

## Licence and citation

GPL-3.0-or-later (see [LICENSE](LICENSE)); third-party components in [THIRD_PARTY.md](THIRD_PARTY.md). Cite with
[CITATION.cff](CITATION.cff).
