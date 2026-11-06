# Third-party components

Checked 2026-11-06. Only redistributable assets are bundled; anything else is downloaded on first use with the
licence shown and the user's consent (spec §1.2).

## Bundled

| Component | Source | Licence | Version | Where |
|---|---|---|---|---|
| IBM Plex Sans, IBM Plex Mono (Regular, Medium, SemiBold) | https://github.com/IBM/plex | SIL Open Font License 1.1 (`dexvantage/resources/fonts/OFL.txt`) | IBM Plex Sans 3.005, IBM Plex Mono 2.004 (the fonts' own version records; the files Vantage ships) | `dexvantage/resources/fonts/` |
| Code ported from Vantage (theme, widgets, charts, colour maps, icons and icon data, renderer, viewport, navigation, dialogs, palette, store, app, batch, self-test, sysenv, workdir, math utilities, provenance, validation machinery; none from Vantage's Apache-2.0 `vantage/runtime`) | https://github.com/Sherin-SEF-AI/Vantage | GPL-3.0-or-later | Vantage 0.8.1 | marked in each module's docstring |

## Runtime dependencies (installed by pip, not bundled in the source tree)

| Package | Licence |
|---|---|
| PyQt6 | GPL-3.0-only (or Riverbank's commercial licence) |
| NumPy | BSD-3-Clause, with parts under 0BSD, MIT, Zlib and CC0-1.0 (its package metadata) |
| SciPy | BSD-3-Clause |
| PyOpenGL | BSD-3-Clause-style (its `license.txt`) |
| OpenCV (opencv-python) | Apache-2.0 |
| MuJoCo | Apache-2.0 |
| PyYAML | MIT |
| certifi | MPL-2.0 |
| jsonschema (tests) | MIT |
| pinocchio (optional extra `pin`, not installed by default) | BSD-2-Clause (repository licence, as GitHub reports it) |

## Robot models

MuJoCo Menagerie licences differ per model, so each model's own `LICENSE` file was read (2026-11-06) and is kept
beside the bundled files. Source: https://github.com/google-deepmind/mujoco_menagerie, checked out at commit
`98a70c569c4c1b6da467a20012f766ee1f8dc521` (2026-11-02); the column "Folder commit" is the last commit that changed
the model's folder (checked in the checkout: the folder's tree at that commit equals the one at `98a70c5` and differs
from its parent's). The files were copied by `scripts/bundle_menagerie.py`, which says exactly what it changes.

### Bundled (`dexvantage/resources/robots/`)

| Model | Menagerie folder | Licence (its own LICENSE) | Folder commit | Checked | Bundled files | Changes |
|---|---|---|---|---|---|---|
| SO-ARM100 (The Robot Studio) | `trs_so_arm100` | Apache-2.0 (`trs_so_arm100/LICENSE`) | `71f066ad0be9cd271f7ed58c030243ef157af9f4` | 2026-11-06 | `so_arm100.xml`, `assets/*.stl`, `LICENSE` (3.1 MB) | none |
| Universal Robots UR5e | `universal_robots_ur5e` | BSD-3-Clause, © 2018 ROS Industrial Consortium (`universal_robots_ur5e/LICENSE`) | `fb36886b3023db08c198264e1e9956ab202c3e3b` | 2026-11-06 | `ur5e.xml`, `assets/*.stl`, `LICENSE` (6.1 MB) | the 20 OBJ meshes (31.0 MB) rewritten as binary STL with the same triangles (MuJoCo-compiled triangles compared: largest difference 0 m); the `<mesh file>` names in `ur5e.xml` changed to `.stl`, with a comment saying so |
| Robotiq 2F-85 | `robotiq_2f85` | BSD-2-Clause, © 2013 ROS-Industrial (`robotiq_2f85/LICENSE`) | `71f066ad0be9cd271f7ed58c030243ef157af9f4` | 2026-11-06 | `2f85.xml`, `assets/*.stl`, `LICENSE` (3.2 MB) | none |
| UR5e + Robotiq 2F-85 (combined) | the two above | BSD-3-Clause and BSD-2-Clause (both LICENSE files) | as above | 2026-11-06 | `ur5e_robotiq_2f85.xml` | written by MuJoCo: `MjSpec.attach(gripper, site="attachment_site", prefix="gripper_")` then `MjSpec.to_xml()`; mesh paths point into the two folders |

Sizes are decimal megabytes (10⁶ bytes). The two folders marked "none" are byte for byte the checkout's files
(compared with `diff -r` on 2026-11-06). Not bundled: each folder's gallery picture, README, CHANGELOG and `scene.xml`;
no folder has a NOTICE file. Menagerie's READMEs describe the models as "simplified" descriptions derived from public
URDFs (ROS-Industrial's for the UR5e and the 2F-85, The Robot Studio's for the SO-ARM100) and say that the SO-ARM100's
joint limits are approximate ([docs/models/robots.md](docs/models/robots.md#known-limits)). Apache-2.0 and the BSD licences allow
redistribution with the licence text kept, which the `LICENSE` files beside the models do; they are compatible with
DexVantage's GPL-3.0-or-later.

### Checked, not bundled

| Model | Where | Licence | Commit | Checked | Why not bundled |
|---|---|---|---|---|---|
| SO-101 (The Robot Studio) | Menagerie `robotstudio_so101` | Apache-2.0 (`robotstudio_so101/LICENSE`) | `3abca2e302625cec9429c49806121ea2855cef81` | 2026-11-06 | its meshes are 18.2 MB; Robot ▸ Import from MuJoCo Menagerie downloads it with its licence shown |
| SO-100 and SO-101 (The Robot Studio's own files) | https://github.com/TheRobotStudio/SO-ARM100 (`Simulation/SO100`, `Simulation/SO101`: URDF and MJCF) | Apache-2.0 (repository licence, as GitHub reports it) | `a758567c3978dfeefe282ede0500085a48fe8f78` | 2026-11-06 | Menagerie's copy is bundled instead |

### Downloaded on request

Robot ▸ Import from MuJoCo Menagerie (`dexvantage/core/robot/menagerie.py`) downloads any other Menagerie model into
`XDG_DATA_HOME/DexVantage/menagerie/<model>/<commit>/`, shows that model's own LICENSE and imports it only after the
user accepts it. The robot records the licence name, the commit and the date accepted (`RobotSource.license`), and
the project keeps the LICENSE text with the model files. Nothing downloaded this way is bundled.
