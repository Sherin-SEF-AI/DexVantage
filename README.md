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

> **Status: Phases 0–10 of 12, with research tools parts 1 and 2.** Robots import from URDF, MJCF and MuJoCo Menagerie, with
> forward and inverse kinematics, six bundled samples (SO-100, UR5e + Robotiq 2F-85, UR5e + Shadow Hand, ALOHA 2, two
> Franka Pandas, the Unitree G1 upper body) and a catalogue of 53 robots; a datasheet library of 138 cameras, depth
> cameras, tactile sensors, hosts and capture cards, cables, lights, teleop devices and robot-learning policies;
> sensors on any link with a Camera view through each lens model; coverage, depth validity, grasp self-occlusion with
> the Contact Visibility Score, VLA token resolution, the multi-finger occlusion matrix and the tactile handoff,
> bandwidth and power, cable flex, printable brackets and hand-eye calibration; MuJoCo ground truth and synthetic
> LeRobot episodes; teleop dead zones, lighting and glare, motion blur, time sync and action-chunk latency; LeRobot,
> ROS 2 bag and RLDS readers, the dataset auditor, plan against recording and targetless silhouette calibration;
> requirements and the readiness scorecard (a CI gate), Monte Carlo tolerances, designed experiments, an NSGA-II
> optimiser and design agent; exporters to MuJoCo, ROS 2, OpenCV and Kalibr, Isaac Lab, LeRobot and an HTML/PDF
> report, the canonical view warp and the Rerun bridge; the Assistant dock and an MCP server; a Python API, plugins,
> project diff and merge, a Git folder form, citations and publication figures; an experiment tracker (MLflow and W&B
> exports), statistics and power analysis, a data-collection campaign planner, the sim-to-real gap, a publication kit,
> cluster sweeps (process pool and Slurm), a multi-cell comparison, the USD robot import and cited prices. Packaging
> and the website (Phase 11) are next. See [docs/BUILD_STATE.md](docs/BUILD_STATE.md).

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
dexvantage --sample ur5e --check defaults sc.json  # the readiness scorecard: exit 0 pass, 1 fail, 2 error
dexvantage --sample ur5e --tolerance-mc tol.json   # Monte Carlo mount tolerances
dexvantage --sample ur5e --optimise opt.json       # NSGA-II: a Pareto front of better rigs
dexvantage --sample ur5e --export out/ --formats mjcf,urdf,opencv_yaml,isaac,lerobot,report
dexvantage --sample aloha --coverage cov.json      # the new samples: aloha, panda-bimanual, g1
dexvantage a.dexv --diff b.dexv                    # what changed between two projects
python -m dexvantage.mcp cell.dexv                 # the MCP server for an MCP client
dexvantage cell.dexv --coverage cov.json --track runs/   # log the run in the experiment tracker
dexvantage --stats spec.json                       # statistics and power analysis from a spec
dexvantage --sample aloha --campaign spec.json     # a data-collection campaign plan (CSV, .ics, HTML)
dexvantage --sample so100 --experiment design.json --jobs 4    # a designed experiment on 4 processes
dexvantage --cells sample:so100 sample:ur5e sample:aloha       # several robot cells side by side
dexvantage --sample ur5e --export out/ --formats paper_kit     # methods, tables, figures, BibTeX
python -c "import dexvantage as dv; print(dv.sample('ur5e').run('coverage').headline)"
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
| Samples: SO-100, UR5e + Robotiq 2F-85, UR5e + Shadow Hand, ALOHA 2, bimanual Panda, Unitree G1 upper body (MuJoCo Menagerie, licences checked) | done | ALOHA arms against `aloha.xml` 4.6 × 10⁻⁷ m; G1 against `g1_with_hands.xml` 0 m |
| Robot catalogue: 53 robots with the maker's figures and sources, imported with each model's licence | done | joint counts against the bundled models: 0 disagreements |
| Hardware library from datasheets: 33 depth cameras, 33 cameras, 10 tactile sensors, 18 hosts and capture cards, 19 policies, 6 lights, 6 cables, 13 teleop devices; every figure with its source and date, gaps "not stated by the maker" | done | 138 of 138 entries validate against the schema |
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
| MuJoCo ground truth (`--groundtruth`): the project as a MuJoCo scene, renders through every lens model, rendered against predicted visibility and depth | done | contact points agree 92.8 % (UR5e), 84.2 % (SO-100); depth 99.8 %; 14.7–17.5 views/s on an RTX 5080 |
| Synthetic LeRobot v2.1 / v3.0 episodes with known faults and ground truth (`--synth-episodes`) | done | written and read back: 0 mismatches; about 130 frames/s at 320 px |
| Teleop overlap and dead zones (`--teleop`): leader arms, GELLO, VR and space-mouse volumes, bimanual handover | done | 2R closed form 1.4 × 10⁻¹⁶ m²; leader workspace against brute force: 0 missed |
| Lighting, glare and IR wash-out (`--lighting`): light library, GGX materials, glare map in the Camera view | done | white furnace by quadrature 8.1 × 10⁻¹²; area lights 2.2 % from the closed form; visibility against ray casting 0.66 % |
| Motion blur and rolling shutter (`--motion-blur`), time sync (`--time-sync`) | done | image velocity against finite differences 2.8 × 10⁻⁹; rolling shutter against a row-by-row readout 2.4 % |
| Action-chunk latency (`--action-chunk`): loop terms, overshoot map, delay offset and ensemble weight | done | against an event simulation 2.9 × 10⁻⁵ m |
| Visual-to-tactile handoff and the sensory blind gap (part of `--dex-occlusion`) | done | against a brute-force scan: 0 m |
| Dataset readers: LeRobot v2.1 and v3.0 (read and write), ROS 2 bags (MCAP, SQLite), RLDS | done | round trips read independently (pyarrow, ffmpeg): 0 mismatches |
| Dataset auditor (`--audit`), plan against recording (`--plan-vs-recording`), FiftyOne export | done | synthetic campaign: recall 1.0 with no false positives for 6 of 10 fault kinds; desync 16.9 ms |
| Targetless link-silhouette calibration (`--silhouette-calib`) | done | slip recovered within 1.66 px; against a board solve 1.65 px |
| Grasp sets and hand-eye captures from recorded datasets | done | hand-eye from a rendered dataset 0.046 mm from the truth |
| Requirements and the readiness scorecard (`--check`), variants compare, `--reproduce`, `--run-plan` | done | bound decisions against a Monte Carlo: 0 of 4 000 disagree |
| Monte Carlo tolerances (`--tolerance-mc`) and the perturbation export | done | σ against the GUM law of propagation 0.6 %; samplers against `kstest` D 0.024 |
| Designed experiments (`--experiment`): factorial, fractional, Latin hypercube, Sobol, Saltelli; Sobol indices | done | Ishigami's analytic indices within 0.0008 |
| Optimiser and design agent (`--optimise`, `--design-agent`): NSGA-II over metric objectives | done | ZDT1, ZDT2 IGD 0.005 (0.01); CONSTR 0.021 (0.05) |
| Exporters (`--export`): MJCF, URDF and tf, OpenCV and Kalibr, masks, Plücker, brackets, Isaac Lab and USD, LeRobot and policy configs, report, perturbations, canonical warp | done | URDF through MuJoCo 1.1 × 10⁻¹⁵ m; USD through pxr 2.6 × 10⁻⁵ px; masks against ray casting 0 pixels |
| Canonical view warp (`--canonical-view`), Rerun bridge (`--rerun`) | done | plane points 2.3 × 10⁻¹³ px; `.rrd` read back 4.8 × 10⁻⁵ px |
| Assistant dock (Claude or Groq, keys in the keyring, number check, proposals, approved plans) and MCP server (39 tools) | done | MCP conformance 0 nonconformities; number check 0 of 25 wrong |
| Python API (`import dexvantage as dv`), plugin SDK, project diff and merge, Git folder form, BibTeX, publication figures | done | folder round trip byte-identical; 8 of 8 merge cases |
| Experiment tracker (`--track`, Analysis ▸ Runs): runs with settings, figures and provenance; compare, re-run; CSV, Parquet, MLflow, W&B | done | 8 writers × 100 runs: 0 lost; read back by MLflow's client: 0 of 80 fields differ |
| Statistics and power analysis (`--stats`, Analysis ▸ Statistics) | done | against SciPy 8.9 × 10⁻¹⁶; G*Power manual 4.6 × 10⁻⁷; Cohen (1992) Table 2 within 1 per group |
| Data-collection campaign planner (`--campaign`, Data ▸ Campaign): Williams-square schedules, storage, calendar | done | 40 seeded campaigns: 0 balance violations; 34 calendar events through RFC 5545: 0 |
| Sim-to-real gap (`--sim2real`, Data ▸ Sim-to-real gap) | done | four injected gaps recovered and ranked first (worst 0.68 of tolerance) |
| Publication kit (`--formats paper_kit`, Export ▸ Paper kit…) | done | every number traced: 0 of 60 table cells off |
| Cluster sweeps (`--jobs N`, `--slurm`, `--run-shard`, `--gather`) | done | gathered shards against the serial run: 0 values differ |
| Multi-cell comparison (`--cells`, File ▸ Compare cells…) | done | known rotation and shift 3.6 × 10⁻¹⁴ °, 2.8 × 10⁻¹⁴ mm |
| USD robot import, URDF inertials, OpenVLA encodings, Isaac principal point | done | third-party USD UR5e against the sample 2.6 × 10⁻⁸ m |
| Cited prices: 81 of 191 library entries, `rig.cost` | done | rig cost against the entries added up again: 0 USD |
| Validation suite | 232 checks | 232 of 232 pass |
| Packaging, website, Gaussian splatting | Phases 11–12 | — |

## Known limits

Exporters: Isaac Lab, Isaac Sim, LeRobot, Kalibr and FiftyOne were not run on DexVantage's files (their formats
follow the cited sources). Assistant: no request was sent to a live API while building it (the tests use a fake
provider). Prices: 81 library entries state a cited list price; the rest read "not stated", and the rig cost
counts the sensors only. Research tools: the tracker keeps a run's project hash, not the project; statistics are
fixed-sample tests (no sequential testing); the sim-to-real render is MuJoCo's shading of the project's model;
cluster sweeps run on Slurm only. Lighting: saturation is relative to an auto-exposure until you
enter a sensor's sensitivity (no maker in the library states one). Timing: no maker states a readout time or a
device's processing latency, so smear, skew and sync figures are bounds until you enter measured ones. Data: LeRobot's
own loader and FiftyOne's importer have not been run on DexVantage's files. Grasps: the fingers stay at their home opening and obstacles are not checked along
the approach; the dexterous-hand postures are sample sweeps, not measured synergies. Hand-eye: OpenCV 5.0.0 has no
`calibrateHandEye` bindings, so DexVantage's own solvers have not been compared with OpenCV's. Sensors: most makers state no lens distortion, so built-in imagers are pinhole
models fitted to the stated FOV until you calibrate a unit; defocus is computed only where a lens's focus distance
and f-number are stated or entered; sensor housings do not occlude.
Robots: xacro must be expanded to URDF
first and a USD file holds one articulation (drives, loops and couplings are not imported); mimic joints and the 2F-85's linkage move independently; free and ball joints are held at
the home pose; the gripper frame must be a link. The G1 sample's legs are held at its stand pose.

## Licence and citation

GPL-3.0-or-later (see [LICENSE](LICENSE)); third-party components in [THIRD_PARTY.md](THIRD_PARTY.md). Cite with
[CITATION.cff](CITATION.cff).
