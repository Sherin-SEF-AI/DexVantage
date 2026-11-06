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

> **Status: Phase 0 of 12 (foundation).** The application shell, the project file, provenance and the validation
> machinery work; robots, sensors and the studies arrive phase by phase. See [docs/BUILD_STATE.md](docs/BUILD_STATE.md)
> for what is done and what is next.

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
| Validation suite | 8 checks | worst error 3.1 × 10⁻¹⁴ against scipy and MuJoCo |
| Robots, sensors, coverage, grasp visibility, … | Phases 1–12 | — |

## Known limits

Phase 0 has no robot import, sensor library or studies yet; commands for them are in the menus, disabled, with the
phase that brings them. The only study is the project summary (dense voxel counts of the task volumes).

## Licence and citation

GPL-3.0-or-later (see [LICENSE](LICENSE)); third-party components in [THIRD_PARTY.md](THIRD_PARTY.md). Cite with
[CITATION.cff](CITATION.cff).
