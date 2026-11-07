# The Workspace page, the design log and the spec's batch names

This chapter covers what the Phases 9 and 10 spec-completeness review added: the Workspace page (spec §13), notes
in the design log, another variant drawn beside the active one, the larger library in the Library panel, the
Canonical view's own depth range (§7.16), and the batch names of §11.

## Worked example: a cup outside the task volume (SO-100)

1. `dexvantage --sample so100`, then **Workspace ▸ Workspace**. The headline reads **1 volume · ready**: the sample
   has one task volume, "Tabletop (sample)" (300 × 200 × 120 mm), a table surface and no task objects.
2. **Add box object** puts a 60 mm box on the volume's floor (one undo step). In the editor below the list, set its
   shape to *cylinder*, its size to radius 40 mm and height 100 mm, its position to x 450, y 50, z 50 mm and tick
   *Glossy* under Materials (each change one undo step; the lighting and depth studies read the tags).
3. The headline turns to **objects outside**: "1 of 1 objects lie outside every task volume (Box 1)". The 3D view
   outlines the selected object; the Camera view draws the same outline in the selected camera's image.
4. **Preview** draws the suggested volume dashed; **Apply** grows "Tabletop (sample)" to hold the cup with a 20 mm
   margin (a DexVantage default; the floor stays at z = 0): the volume becomes 660 × 449 × 120 mm, from
   (−0.15, −0.339, 0) to (0.51, 0.11, 0.12) m, and the headline reads **ready**. Undo puts the old volume back.
5. **Add surface** adds a 1 × 1 m table top at z = 0; set its height, size, centre, thickness (0: a rectangle) and
   material. Lights are listed here and edited on the Lighting page (the button opens it).
6. **Export…** writes the workspace and the check as JSON with provenance.

The check is `core/workspace_check.py` (the model, its two validation checks and their measured agreement:
[models/workspace.md](../models/workspace.md)).

## Notes and decisions in the design log

The Analysis dock's last tab, **Design log**, lists every edit (newest first). Type a note in its field, choose
*Note* or *Decision* and press Enter (or **Add**): it is kept in the project file (`project.log`, kind `note` or
`decision`) and the project is marked as changed. **Show** filters the list: everything, notes and decisions, or edits.

## Another variant beside the active one

On **Analysis ▸ Requirements and variants**, **New variant from this one** copies the active variant. Move a camera
in the copy, then choose it under **Show in 3D**: its cameras are drawn as dashed frusta beside the active variant's,
with a dashed line from each sensor to its counterpart of the same name. *Only the active variant* turns it off.

## The larger library in the Library panel

The Library panel now has groups for **Teleop devices**, **Policies**, **Lights** and **Cables** next to the
sensors and hosts. They are not placed in the scene: double-click opens the page that uses them and adds the entry
there (a teleop device on the Teleop page, driving the primary robot; a policy on the Policies page; a light on the
Lighting page, above the task volume), each one undo step; a cable is chosen per cable run on the Cables page. The
**Mounts** group opens the Brackets page (ISO 9409-1 flange adapter, T-slot clamps, pan-tilt, palm and dorsal
mounts).

## The Canonical view's depth range (§7.16)

By default the residual parallax is measured over a bound from the task volume and the pads' reach. Tick **Own depth
range** on **Export ▸ Canonical view** and give the two distances from the contact plane (negative: beyond it;
positive: towards the source camera; the upper end is held short of the camera's near limit). The setting is
`depth_range_m` of the study; unticking returns to the default (each one undo step).

## The specification's batch names (§11)

| Specification | DexVantage |
|---|---|
| `--grasp-study grasps.json` | the same (also `--grasp`) |
| `--cable-flex out.json` | the same (also `--cables`) |
| `--mujoco-validate out/` | the same: a folder (one that exists, or a path ending in `/`) gets `out/groundtruth.json`; also `--groundtruth OUT.json` |
| `--handeye-solve captures/` | `--handeye-solve OUT.json --captures captures/` (the joint states are `joint_states.csv` or `.json` in the folder, or `--joints FILE`) |
| `--export out/ --formats mjcf,usd,…` | `usd` is the Isaac Lab and Isaac Sim export (`isaac`) |
| `--formats …,fiftyone` | with `--dataset PATH`, or the dataset the project's audit uses |

Every study flag takes a folder as well as a file: `dexvantage --sample so100 --vla-tokens out/` writes
`out/vla-tokens.json`.

Example (a simulated capture, so the true mount offset is known):

    dexvantage --sample ur5e --handeye-solve solve.json --captures cap/images --joints cap/joint_states.csv

`tests/test_spec_completeness.py` renders such a capture of ten planned poses with MuJoCo and solves it in batch:
the solved change matches the simulated offset within 0.3 mm.
