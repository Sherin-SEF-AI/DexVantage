# Third-party components

Checked 2026-11-06. Only redistributable assets are bundled; anything else is downloaded on first use with the
licence shown and the user's consent (spec §1.2).

## Bundled

| Component | Source | Licence | Version | Where |
|---|---|---|---|---|
| IBM Plex Sans, IBM Plex Mono (Regular, Medium, SemiBold) | https://github.com/IBM/plex | SIL Open Font License 1.1 (`dexvantage/resources/fonts/OFL.txt`) | IBM Plex Sans 3.005, IBM Plex Mono 2.004 (the fonts' own version records; the files Vantage ships) | `dexvantage/resources/fonts/` |
| Code ported from Vantage (theme, widgets, charts, colour maps, icons and icon data, renderer, viewport, navigation, dialogs, palette, store, app, batch, self-test, sysenv, workdir, math utilities, provenance, validation machinery; Phase 2: the lens projection maths of `core/optics/lens.py` from Vantage `core/lens.py`, the cube-map idea, face basis and plane depth of `core/occlusion.py` from Vantage `core/occlusion.py`, the Camera view widget `ui/camview.py` from Vantage `ui/camview.py`; Phases 3, 6 and 7: `core/plucker.py` from Vantage `core/plucker.py`, `core/meshing.py` and `core/brackets.py` from Vantage `core/meshing.py` and `core/brackets.py`, the board points, grey conversion, PnP and outlier rule of `core/calib/boards.py` and `core/calib/detect.py` from Vantage `core/targets.py` and `core/calib.py`, the harness sheet, findings and `simplify` of `core/cables.py` / `core/studies/cables.py` from Vantage `core/harness.py`, the data-rate rows, disk budget and power findings of `core/studies/bandwidth.py` from Vantage `core/recording.py` and `core/power.py`; ideas only, no code copied verbatim: the Fisher-information planner of `core/calib/planner.py` from Vantage `core/observability.py` and the next-best-view coach of `core/calib/coach.py` from Vantage `core/calib_coach.py`; none from Vantage's Apache-2.0 `vantage/runtime`; Phases 4, 5 and 8: `core/rosbag.py` from Vantage `core/rosbag.py` (the CDR codec, the MCAP writer and both readers; the automotive box-marker helper dropped), `core/audit/slip.py` from Vantage `core/online_calib.py` (`select_model`, `refine_step`, the Huber weights and the event rule; the vehicle covariates replaced by time); `dexvantage/validation/VANTAGE_SLIP.json` holds Vantage's own outputs of `select_model` on twelve seeded series, recorded once at Vantage commit `89eeab8` by `scripts/record_vantage_slip.py`) | https://github.com/Sherin-SEF-AI/Vantage | GPL-3.0-or-later | Vantage 0.8.1 | marked in each module's docstring |

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
| pyarrow (optional extra `lerobot`: LeRobot Parquet files) | Apache-2.0 |
| PyAV `av` (optional extra `lerobot`: LeRobot AV1 and H.264 videos) | BSD-3-Clause (its package metadata); its wheel carries FFmpeg's and SVT-AV1's shared libraries, whose licence terms the wheel's metadata does not state: read them before redistributing an installation |
| zstandard, lz4 (optional extra `bags`: compressed MCAP chunks) | BSD-3-Clause; BSD (package classifiers) |
| mcap, protobuf (tests only: reference readers for the bag and RLDS checks; extra `test`) | MIT; BSD-3-Clause |

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
| Shadow Hand E3M5, right (Shadow Robot Company) | `shadow_hand` | Apache-2.0, © 2022 Shadow Robot Company Ltd (`shadow_hand/LICENSE`) | `71f066ad0be9cd271f7ed58c030243ef157af9f4` | 2026-11-06 | `right_hand.xml`, `keyframes.xml` (the folder's named hand postures), the 13 `assets/*.obj` meshes `right_hand.xml` uses, `LICENSE` (3.4 MB) | none: byte for byte (compared with `diff -r`); the folder's `mounting_plate.obj`, used by no model file, `left_hand.xml`, the scenes, README, CHANGELOG and gallery picture are not bundled |
| UR5e + Shadow Hand (combined) | `universal_robots_ur5e` and `shadow_hand` | BSD-3-Clause and Apache-2.0 (both LICENSE files) | as above | 2026-11-06 | `ur5e_shadow_hand.xml` | written by MuJoCo with `scripts/bundle_dexhand.py`: `MjSpec.attach(hand, site="attachment_site", prefix="hand_")` then `MjSpec.to_xml()`; before attaching, the hand file's forearm rotation `quat="0 1 0 1"` is set to identity so the hand continues the flange axis (the file's top comment says so); the hand's triangles in its forearm frame match the hand file's within 3 × 10⁻¹⁶ m |

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

Dexterous hands (spec §6; Menagerie checkout at `98a70c5`, each folder's own LICENSE read 2026-11-06; sizes are the
whole folder). One hand is bundled, the Shadow Hand (24 joints, with named postures of its own in `keyframes.xml`,
which the multi-finger occlusion study sweeps); the others are redistributable too and download on request with
Robot ▸ Import from MuJoCo Menagerie.

| Model | Where | Licence (its own LICENSE) | Folder commit | Checked | Why not bundled |
|---|---|---|---|---|---|
| Wonik Robotics Allegro Hand V3 | Menagerie `wonik_allegro` (`right_hand.xml`, `left_hand.xml`; 16 joints) | BSD-2-Clause, © 2016 SimLab (`wonik_allegro/LICENSE`) | `71f066ad0be9cd271f7ed58c030243ef157af9f4` | 2026-11-06 | redistributable (2.6 MB); one sample hand is bundled, the Shadow Hand |
| Unitree G1 with hands | Menagerie `unitree_g1` (`g1_with_hands.xml`: a three-finger hand per side with 7 joints, thumb 3, index 2, middle 2; the README does not name the hand model, so it is not called Dex3 here) | BSD-3-Clause, © 2016-2023 Unitree Robotics (`unitree_g1/LICENSE`) | `98a70c569c4c1b6da467a20012f766ee1f8dc521` | 2026-11-06 | 39.7 MB; downloads on request |
| LEAP Hand | Menagerie `leap_hand` | MIT, © 2023 Ananye Agarwal (`leap_hand/LICENSE`) | `71f066ad0be9cd271f7ed58c030243ef157af9f4` | 2026-11-06 | 15.6 MB; downloads on request |
| Shadow DEX-EE | Menagerie `shadow_dexee` | Apache-2.0 (`shadow_dexee/LICENSE`) | `fb36886b3023db08c198264e1e9956ab202c3e3b` | 2026-11-06 | 10.9 MB; downloads on request |
| TetherIA Aero Hand Open | Menagerie `tetheria_aero_hand_open` | Apache-2.0 (`tetheria_aero_hand_open/LICENSE`) | `fb36886b3023db08c198264e1e9956ab202c3e3b` | 2026-11-06 | 9.1 MB; downloads on request |
| Sharpa Wave | Menagerie `sharpa_wave` | Apache-2.0 (`sharpa_wave/LICENSE`) | `98a70c569c4c1b6da467a20012f766ee1f8dc521` | 2026-11-06 | 15.8 MB; downloads on request |
| Inspire hands, Ability Hand (PSYONIC) | not in Menagerie: no folder for them at `98a70c5` | — | — | 2026-11-06 | no official model file in Menagerie; a maker's own URDF can be imported with Robot ▸ Import robot… after reading its licence |

### Downloaded on request

Robot ▸ Import from MuJoCo Menagerie (`dexvantage/core/robot/menagerie.py`) downloads any other Menagerie model into
`XDG_DATA_HOME/DexVantage/menagerie/<model>/<commit>/`, shows that model's own LICENSE and imports it only after the
user accepts it. The robot records the licence name, the commit and the date accepted (`RobotSource.license`), and
the project keeps the LICENSE text with the model files. Nothing downloaded this way is bundled.

## Hardware library sources (cited, not bundled)

The hardware library (`dexvantage/resources/library/`, 40 entries in Phase 2, 51 after Phases 3, 6 and 7) holds figures copied from the makers'
public datasheets, product pages and papers, each entry with its source URL or document, title and the date read
(2026-11-06 for every built-in entry). Only the figures are stored, written as DexVantage's own JSON; no datasheet,
drawing, image or text of a maker is bundled, and nothing is downloaded at run time. The sources by maker:

| Maker | Where the figures were read |
|---|---|
| Intel RealSense (D405, D415, D435, D435i, D455, D457) | "RealSense Product Family D400 Series Datasheet" Rev 023 (March 2026) and the D457 datasheet Rev 006, from realsenseai.com; store.realsenseai.com product pages |
| Stereolabs (ZED 2i, ZED Mini, ZED X, ZED X Mini) | stereolabs.com product pages and the datasheets they link (cdn.sanity.io) |
| Luxonis (OAK-D, OAK-D Pro, OAK-D SR, OAK-D Lite) | docs.luxonis.com and shop.luxonis.com; camera positions from the design values in Luxonis' `depthai-boards` calibration files on GitHub (commit `bfa8f36`), cited as figures only |
| Orbbec (Femto Bolt, Gemini 2, Gemini 335, Gemini 336) | orbbec.com product pages, doc.orbbec.com and the datasheets they link |
| e-con Systems, Leopard Imaging, Basler, Teledyne FLIR, OmniVision, ams OSRAM | each maker's product page or datasheet |
| GelSight (Mini), Meta (DIGIT) | gelsight.com; Lambeta et al. 2020 (arXiv) and Meta's `digit-interface` repository at commit `87a28bb` (figures only) |
| NVIDIA (Jetson), USB-IF | nvidia.com specifications tables; usb.org |

The library schema (`dexvantage/resources/library.schema.json`) and every entry are DexVantage's own work under its
GPL-3.0-or-later licence. Datasheet figures are facts restated with their source; the makers' documents keep their
own terms and are reached through the cited URLs.

## Phases 3, 6 and 7: cited sources (figures only, nothing bundled)

Policy library (`dexvantage/resources/library/policy/`, 8 entries). Every figure carries the index of its source and
a quote of what the source says; a figure no source states is null. Code is cited at a pinned commit. All read
2026-11-06.

| Entry | Sources |
|---|---|
| π0, π0-FAST | Black et al., arXiv:2410.24164; Pertsch et al., arXiv:2501.09747; Beyer et al. (PaliGemma), arXiv:2407.07726; openpi (official code, Apache-2.0) at commit `215abfb` |
| OpenVLA 7B | Kim et al., arXiv:2406.09246; the `openvla/openvla-7b` model card, `config.json` and `preprocessor_config.json` on Hugging Face; the `openvla/openvla` README |
| SmolVLA base | Shukor et al., arXiv:2506.01844; the `lerobot/smolvla_base` model card and `config.json`; LeRobot (Apache-2.0) at commit `d490e47`; the SmolVLM2-500M `config.json` |
| ACT | Zhao et al., arXiv:2304.13705; `tonyzhaozh/act` at commit `742c753`; LeRobot's ACT configuration at commit `d490e47` |
| Diffusion Policy (CNN) | Chi et al., arXiv:2303.04137 |
| Octo-Base | Octo Model Team, arXiv:2405.12213; `octo-models/octo` at commit `241fb35` |
| GR00T N1 2B | NVIDIA, arXiv:2503.14734; the `nvidia/GR00T-N1-2B` model card; `NVIDIA/Isaac-GR00T` (branch n1-release) at commit `755876a` |

The crop-rule check (`policies.crop_geometry`) writes each rule's pipeline out again from the cited code (openpi
`resize_with_pad`, LeRobot's left-top variant, torchvision `center_crop`, GR00T `VideoCrop`); no code of those
projects is copied into DexVantage.

Cable library (`dexvantage/resources/library/cable/`, 3 entries): Basler product documentation "Cable GMSL, FAKRA-Z,
1x F/F" (docs.baslerweb.com); VA Imaging datasheet "VA-CABLE-USB3-xxm-HF" (hs.va-imaging.com); Newnex product page
"High Flex Active USB3 Extension Cable" (newnex.com). Read 2026-11-06.

Brackets: the ISO 9409-1-50-4-M6 flange pattern from Universal Robots' UR5e datasheet drawing "Tool flange for UR5e"
(universal-robots.com) and the public preview of ISO 9409-1:2004 (clauses 4 and 5, Figure 1; cdn.standards.iteh.ai);
the 2020 and 4040 B-type T-slot profile figures from motedis.es product pages; the RealSense D405, D435 and D435i
mounting holes from Intel's D400 datasheet (added to those library entries with their source). Read 2026-11-06. No
drawing or standard text is bundled.

Bandwidth and power: USB 2.0 (Table 5-10, §7.2.1) and USB 3.2 (§4.4.11, Table 3-1) specifications from usb.org; the
ams OSRAM NanEyeC datasheet's stated serial rates. Read 2026-11-06.

Hand-eye calibration: the AprilGrid layout facts (tag spacing, the corner order) read from Kalibr's documentation and
source on GitHub (ethz-asl/kalibr, BSD-3-Clause) on 2026-11-06; no Kalibr code was copied. ChArUco boards follow
OpenCV's layout (OpenCV, Apache-2.0, already a runtime dependency). The closed-form solvers are written from the
papers (Tsai-Lenz 1989, Park-Martin 1994, Horaud-Dornaika 1995, Andreff 1999, Daniilidis 1999, Shah 2013, Li 2010;
`validation/references.py`).

Plücker export: the `.safetensors` files are written by DexVantage's own writer from the public format description
(huggingface/safetensors README); the `safetensors` package is used only to read them back in the check when it is
installed (Apache-2.0; not installed in `.venv`, not a dependency).

## Phases 4, 5 and 8: cited sources and formats (figures only, nothing bundled)

Light library (`dexvantage/resources/library/light/`, 3 entries: Elgato Key Light, Key Light Air MK.2, Ring Light):
the figures each Elgato product page and its technical specifications state (elgato.com, read 2026-11-07); none
states an IR figure, so none is entered. RealSense projector optical power and wavelength (D415, D435, D435i, D455):
Table 3-17 of the D400 Series Datasheet (Rev 023); the D435's and D405's exposure ranges and trigger support: Table
4-23 and sections 3.7.4 and 7.13 of the same datasheet (read 2026-11-07). Refractive indices in `core/materials.py`:
tabulated measurements and Sellmeier fits from the refractiveindex.info database (public domain, CC0; the files are
listed in the module). The relative-exposure model's 18 % grey and 12.7 % of saturation: ISO 12232 as summarised by
Wikipedia "Film speed" (the standard itself was not read). The SO-100 leader arm of the sample: the LeRobot SO-100
guide (huggingface.co/docs/lerobot/so100, read 2026-11-07).

Formats, read from their sources; no code of these projects is copied into DexVantage: LeRobot dataset v3.0 at
commit `d490e47` and v2.1 at tag v0.3.3 (`b883328`) (huggingface/lerobot, Apache-2.0); the MCAP specification and the
ROS 2 message definitions (`sensor_msgs`, `std_msgs`; ros2/common_interfaces, Apache-2.0); RLDS and the TFRecord
framing with CRC-32C (google-research/rlds and TensorFlow, Apache-2.0); the FiftyOne legacy dataset layout read by
`LegacyFiftyOneDatasetImporter` (voxel51/fiftyone develop, Apache-2.0, read 2026-11-07). LeRobot's own loader and
FiftyOne's importer have not been run against DexVantage's files (neither is installed).

Synthetic episodes and the ground-truth scenes are generated from the project's own bundled robots (licences above);
nothing new is bundled.

## Phases 3, 6 and 7 additions to runtime use

No new runtime dependency. Optional and imported only when asked for and installed: pinocchio (`pip install pin`,
BSD-2-Clause) and NVIDIA cuRobo (NVIDIA's own licence; read it before installing) as batch-IK backends of the grasp
study; neither is installed in `.venv`, and their code paths have not run.

## Phase 2 additions to runtime use

No new runtime dependency. The occlusion module's GPU backend uses PyOpenGL (already listed) through EGL when it is
available; jsonschema (already listed, MIT) validates library entries when installed, else a smaller built-in check
runs.
