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

> **Status: Phase 1 of 12 (robots and kinematics).** Robots import from URDF, MJCF and MuJoCo Menagerie, with
> forward and inverse kinematics, joint sliders, IK drag and two bundled samples; sensors and the studies arrive phase
> by phase. See [docs/BUILD_STATE.md](docs/BUILD_STATE.md) for what is done and what is next.

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
dexvantage --self-test              # check this installation without a window
dexvantage cell.dexv --analyse -    # the project summary as JSON, with provenance
dexvantage --validate report/       # the validation suite and its HTML report
python -m pytest tests              # the tests (offscreen)
```

## What works now

| Area | State | Measured |
|---|---|---|
| Window shell: 13 menus, docks, six workspaces, command palette (Ctrl+K), Find (Ctrl+F), shortcuts (?) | done | — |
| 3D viewport: Vantage's GLSL renderer and orbit-about-cursor navigation, robot-cell scale | done | renders headless through EGL |
| `.dexv` project file, JSON Schema, migrations, undo/redo | done | round trip: 0 differences |
| Provenance (`dexvantage.provenance/1`), `--reproduce` | done | — |
| Robot import: URDF, MJCF (meshes, includes, textures stored in the project), import wizard and taggers | done | — |
| MuJoCo Menagerie import with each model's own licence shown and recorded | done | — |
| Forward kinematics (MuJoCo) checked by an independent product-of-exponentials chain and DH closed forms | done | agreement 8.7 × 10⁻¹⁵ m on both samples; DH 6.3 × 10⁻¹⁶ m |
| Inverse kinematics (damped least squares, limits, seeds, IK drag) | done | 150 of 150 round trips within 0.1 mm; about 4 ms per target |
| Manipulability, self-collision distances | done | 2R closed form 1.1 × 10⁻¹⁶ m²; sphere and capsule distances 1.1 × 10⁻¹⁶ m |
| Joint sliders, pose scrubber, trajectories from CSV | done | — |
| Samples: SO-100, UR5e + Robotiq 2F-85 (MuJoCo Menagerie, licences checked) | done | — |
| Validation suite | 18 checks | 18 of 18 pass; worst relative to tolerance: IK round trip 0.0997 mm of 0.1 mm |
| Sensors, coverage, grasp visibility, … | Phases 2–12 | — |

## Known limits

There is no sensor library or study yet beyond the project summary (dense voxel counts of the task volumes);
commands for them are in the menus, disabled, with the phase that brings them. Robots: xacro must be expanded to URDF
first and USD is not read; mimic joints and the 2F-85's linkage move independently; free and ball joints are held at
the home pose; the gripper frame must be a link. ALOHA 2, the bimanual Panda and the G1 samples arrive in Phase 11.

## Licence and citation

GPL-3.0-or-later (see [LICENSE](LICENSE)); third-party components in [THIRD_PARTY.md](THIRD_PARTY.md). Cite with
[CITATION.cff](CITATION.cff).
