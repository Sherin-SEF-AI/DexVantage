# Optimise: improve a rig, or design new ones

Analysis ▸ Optimise. The model, its equations, references and validation are in `docs/models/optimise.md`.

The page has two modes:

- **Improve this rig** (the optimiser): moves, tilts and pans the rig's unlocked cameras a little, switches cameras off
  and, if you allow it, swaps their devices, looking for designs that do better on the objectives within your
  requirements.
- **Design new rigs** (the design agent): builds complete rigs from the library (up to three new cameras by default) on
  the cage around the task volume and on the robot's gripper, for your requirements.

Both run NSGA-II, an evolutionary multi-objective search, in the background (Run; Stop keeps what has been scored).
Locked sensors are never changed. Nothing changes until you press **Accept** on a design, which is one undo step.

## Worked example: the UR5e sample

Open File ▸ Open sample ▸ UR5e and Robotiq 2F-85 (or `dexvantage --sample ur5e`). Add one requirement, the
specification's default contact visibility of at least 95 % (`grasp.contact_visibility >= 95`, labelled a default).
The figures below were measured on 2026-11-07 with the default settings (seed 0).

### Improve this rig

Run with the defaults (96 evaluations, population 12, shifts up to 5 cm, turns up to 15°, cameras may be removed). It
took 105 s. The headline reads **2 designs, better rig found**.

Left out, each with its reason under the table: the cost (no price is stated for the D405 or the D435: the built-in
library entries carry no prices, and none is guessed), the bandwidth margin (the sample has no host controller) and
the VLA object tokens (no task object is marked critical). The search therefore ranks four objectives: contact
visibility, depth-valid task volume, cameras and sensors without a ready bracket.

| Design | Contact visibility | Depth-valid | Cameras | Not ready | What it changes |
|---|---|---|---|---|---|
| current | 100.0 % | 84.6 % | 2 | 2 | — |
| A | 100.0 % | 96.4 % | 2 | 2 | Wrist D405 up 4.8 cm, tilted −4.3°, panned −7.9°; Overhead D435 3.7 cm along −x, up 2.4 cm |
| B | 100.0 % | 77.5 % | 1 | 1 | Wrist D405 moved and turned; Overhead D435 removed |

Design A's explanation: "Against the current rig it has 11.8 points more depth-valid task volume (96.4 %), and is no
worse on the other objectives. ... Against design B it trades one more camera and one more sensor without a ready
bracket for 18.9 points more depth-valid task volume. It meets every requirement (1)."

Select A, press **Preview**: its cameras are drawn dashed in the 3D view (and marked in the Camera view). Press
**Accept**: the rig moves to design A in one undo step. The Depth validity study then reads the same 96.4 % at the robot's home
pose (the validation check optimise.figures_reproduce re-runs the studies on each option and finds no difference). The
search scores the grasp study at screening settings (40 grasps); turn on Confirm to re-score the designs shown at the
project's own settings.

### Design new rigs

Switch to Design new rigs and Run with the defaults (120 evaluations, population 16, up to 3 cameras, every camera and
depth device of the library, the cage and the gripper as mount sites). It took 237 s and offers **5 designs, better
rig found**. The current rig's two sensors are not in the designs (each design is a complete rig; lock a sensor to keep
it, or set keep_unlocked).

| Design | Contact visibility | Depth-valid | Cameras | Not ready | The rig |
|---|---|---|---|---|---|
| current | 100.0 % | 84.6 % | 2 | 2 | Wrist D405 + overhead D435 |
| A | 100.0 % | 99.1 % | 3 | 3 | D457 on the cage (1.15 m, 28° up); OV2312 and D415 on the gripper |
| B | 100.0 % | 98.9 % | 2 | 2 | D457 on the cage (1.15 m, 30° up); OAK-D Pro on the gripper, 7 cm off its axis |
| C | 100.0 % | 9.9 % | 2 | 1 | ZED 2i and OVM6948 on the gripper |
| D | 100.0 % | 6.3 % | 1 | 1 | D415 on the gripper |
| E | 100.0 % | not stated | 1 | 0 | OVM6948 on the gripper |

B is the design that beats the current rig outright: as many cameras, 14.2 points more depth-valid task volume. A adds
a third camera for 0.2 points more. E has no depth sensor, so its depth-valid figure is "not stated" (it ranks last on
that objective), and the explanation says so. Accept adds the design as a new variant ("Design B (design agent)") and
makes it active; the variant menu or Undo returns to the sample rig.

With the default budgets the front is coarse: a larger budget (Evaluations) or another seed finds more designs. The
same seed and settings give the same designs.

## Batch

    dexvantage --sample ur5e --optimise opt.json
    dexvantage project.dexv --design-agent rigs.json --optimise-settings '{"budget": 300, "seed": 4}'

`--optimise-settings` takes JSON text or a .json file and is merged over the project's settings
(`project.analysis.studies['optimise']` or `['design_agent']`) for that run. The result is the page's (objectives,
what was left out and why, constraints, every design scored, the options with their explanations) with its provenance
block. Settings: `objectives` (metric ids), `budget`, `population`, `seed`, `options`, `confirm`, `constraints`,
`screen`; the optimiser's `move_m`, `turn_deg`, `allow_remove`, `swap_models`, `factors` (your own list of
`{"path": "sensor:<id>.x", "lo": .., "hi": ..}` or `{"path": .., "choices": [..]}`); the design agent's
`max_cameras`, `models`, `static_frames`, `links`, `distance_m`, `elevation_deg`, `wrist_back_m`, `wrist_radial_m`,
`keep_unlocked`. Every number not set is a labelled default.
