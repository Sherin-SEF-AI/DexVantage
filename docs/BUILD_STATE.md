# Build state

The resume point for the next session: what is done, what is next, open issues, and the decisions taken. Updated at
the end of every phase.

## Phase 1: robots and kinematics (done, 2026-11-06)

**Review gate**
Final gate, after the three reviews (truthfulness and licences, kinematics correctness, researcher workflows):
- Tests: **506 passed**, 0 failed, 0 skipped (`python -m pytest tests`, offscreen, memory-capped, 32 s). Phase 1 added
  174: test_robot_core 20, test_kinematics 35, test_ik 18, test_samples 18, test_menagerie 33, test_robot_ui 11,
  test_phase1_integration 9, test_phase1_review 18, test_robot_review 12.
- Validation: **18 of 18 checks pass** (frames 6, kinematics 10, formats 2; full run 3 s). Kinematics, measured with
  the default seed: FK MuJoCo against the independent PoE chain 8.7 × 10⁻¹⁵ m (both samples and five robots written in
  the check, 200 random q each, every link); UR5e standard DH 5.6 × 10⁻¹⁶ m; Panda modified DH 6.3 × 10⁻¹⁶ m;
  Jacobian against finite differences 4.4 × 10⁻¹⁰; Jacobian against `mj_jac` 8.8 × 10⁻¹⁵ (link origins and a random
  point fixed in each link); IK round trip 150/150, worst 0.0997 mm of 0.1 mm (the solver's own convergence
  tolerance); IK position-only 150/150, worst 0.093 mm; 2R manipulability 1.1 × 10⁻¹⁶ m²; sphere and capsule distances
  1.1 × 10⁻¹⁶ m (both against 10⁻⁹ m). `kinematics.fk_pinocchio` is registered only when pinocchio is installed and
  has not run (not in `.venv`).
- `dexvantage --self-test`: **10 of 10 ok** (new: robots, both samples import and the two FK engines agree within
  4 × 10⁻¹⁵ m).
- `ruff check dexvantage tests scripts`: clean.
- `dexvantage --sample so100` and `--sample ur5e` open in the window (also File ▸ Open sample);
  `python -m dexvantage --sample so100|ur5e --analyse -` exits 0.
- Demo: `scripts/demo-phase1.sh`. Screenshots: `python scripts/screens.py` → `out/screens/` (`sample-so100-*`,
  `sample-ur5e-*`: main window, Robot page, the wizard's six steps; 13 s).

**Review fixes (Phase 1)**
- Truthfulness: the Menagerie licence recogniser named GPL/LGPL/AGPL, Clear BSD, CC0, CC BY, OFL, BSD-4 and Boost
  wrongly (it now reads the licence named first and its version; checked on the official texts and all 70 Menagerie
  licences at 98a70c5); SO-ARM100 joint limits are marked approximate (Menagerie's README says so); the bundled UR5e's
  rounded link offsets are documented (flange up to 1.3 mm from UR's DH after best alignment, 0.9 mm RMS);
  `THIRD_PARTY.md` sizes, font versions, ported-code list and runtime licences corrected.
- Kinematics: IK `nearest_q0` wrapped unlimited joints around home instead of q0; the IK reach bound for slide joints
  could reject reachable targets; the half-turn rotation error was off by up to 1 mrad; wrongly shaped joint batches
  were silently regrouped (now `RobotModel.batch_q` refuses them); MJCF `fusestatic` dropped jointless bodies; the
  `mj_jac` check now tests offset points and fails instead of crashing on reader mismatches.
- Workflows: File ▸ Open sample; robots and trajectories can be deleted (one undo step); the Inspector shows robots and
  trajectories; trajectories are listed under Tasks; xacro and link-less URDFs and COLLADA meshes are refused with
  what to do; `--reproduce` no longer reports a sample's variant as differing from itself (`variant_digest` replaces
  outside ids by their place); CSVs with semicolons, tabs and decimal commas read; misleading messages and plurals
  fixed.

**Done**
- `core/robot/`: `importer.py` (URDF, MJCF, every referenced file stored with a manifest), `model.py` (MuJoCo
  kinematics model and shared cache), `kinematics.py` (independent PoE FK and Jacobian, own XML reader), `ik.py`,
  `manipulability.py`, `collision.py`, `filecheck.py`, `trajectory.py`, `samples.py`, `menagerie.py`.
- UI: the import wizard (`ui/dialogs/import_wizard.py`), the Menagerie dialog, the Robot page with joint sliders and
  read-outs (`ui/pages/robot.py`), IK drag (`ui/ikdrag.py`), the pose scrubber (`ui/scrubber.py`), trajectory import;
  robots drawn and link-mounted sensors posed at q (`ui/scene.py`, `ui/store.py`).
- Bundled Menagerie models (`dexvantage/resources/robots/`, 12.4 MB) and the two samples; `--sample` in the window and
  in batch mode.
- Validation: `checks_kinematics.py`, `checks_ik.py`; the suite's built-in project is now the two samples.
- Docs: `docs/models/robots.md`, `kinematics.md`, `ik.md`, linked from `docs/overview.md`; the user guide's Robots
  chapter with the SO-100 and UR5e tours (`docs/user-guide.md`); `docs/guide/samples.md`; robot records in
  `docs/project-format.md`; README and CHANGELOG.

**Decisions (Phase 1)**
1. **Two kinematics engines.** MuJoCo is the primary engine; the product-of-exponentials chain shares no code with it
   or with `core/frames.py`, so the FK and Jacobian checks compare independent implementations.
2. **Only the joints that move the target link are solved by IK** (`manipulability.chain_mask`): a gripper's fingers
   and the other arm of a bimanual robot stay put. Arms with fewer than six such joints are solved position-first.
3. **IK drag uses the rule `nearest_q0`** (the solution nearest the current pose), so dragging never jumps between
   elbow-up and elbow-down.
4. **The project summary counts the joints in q** (revolute, continuous, prismatic); it counted fixed joints before
   (the UR5e sample read 21 instead of 14). The Outliner does the same.
5. **The Robot page's headline** flags "near a limit" only for joints that move the gripper frame (an open gripper
   rests at its stops by design), and "singular" when the condition number exceeds 10¹² (rank-deficient to double
   precision; a numerical bound, not a tuning default). A planar arm therefore always reads "singular": its tip cannot
   move out of the plane.
6. **Framing:** the frame button fits the robots, task volumes and sensors (the 1.2 m empty-cell box is used only
   while a project has no robot or volume); the SO-100 filled a tenth of the view before.
7. **Samples tag their gripper joints** (`"gripper": true`), as the wizard's joint tagger does; the wizard's
   gripper-frame guess never picks a link that a gripper joint moves (it guessed `Moving_Jaw` and
   `gripper_left_silicone_pad` before; now `Fixed_Jaw` and `gripper_base`, the samples' own choices).
8. **The UR5e's meshes are bundled as binary STL** converted from Menagerie's OBJ (31.0 MB of text; the triangles are
   identical when compiled by MuJoCo). BSD-3-Clause allows modified redistribution; the change is recorded in
   `THIRD_PARTY.md` and at the top of `ur5e.xml`.
9. **Sample camera placements are labelled as sample placements** (names, variant note, project notes); the sensors
   have no library model until Phase 2.
10. **Validation's built-in project** is the SO-100 sample with the UR5e sample's robot added, so the project-dependent
    kinematics checks cover the bundled robots without a project being opened.

**Spec corrections (Phase 1)**
1. **Panda DH source (§10, the DH checks):** Franka's "Robot and interface specifications" page now returns HTTP 404.
   The Panda's figures come from that page's source file in the Franka docs repository at commit 476a2ed2c1
   (2021-11-11), which matches the current Franka Research 3 page (`references.py`, `docs/models/kinematics.md`).
2. **§14.2 "the official model" of the SO-100/SO-101:** the bundled SO-100 is MuJoCo Menagerie's `trs_so_arm100`
   (Apache-2.0). The SO-101 (`robotstudio_so101`, Apache-2.0) is not bundled because its meshes are 18 MB; it imports
   through Robot ▸ Import from MuJoCo Menagerie.
3. **§13 fingertip tagger:** fingertips and the gripper frame can only be links; MJCF sites (such as the 2F-85's
   `pinch` site) are not frames of `RobotModel` yet.
4. **§14.3 "a wrist RealSense D405 and an overhead D435":** placed and named, but their figures (lens, depth range,
   minimum Z) come with the Phase 2 datasheet library; no figure was typed in.

## Phase 0: foundation (done, 2026-11-06)

**Review gate**
- Tests: **332 passed**, 0 failed, 0 skipped (`python -m pytest tests`, offscreen, 7.6 s). 255 of them are the
  per-icon markup and render checks.
- Validation: **8 of 8 checks pass** (frames 6, formats 2); worst error 3.1 × 10⁻¹⁴ (URDF rpy round trip).
- `dexvantage --self-test`: 9 of 9 ok (project, frames, mujoco, opencv, fonts, opengl, main window, software 3d,
  validation).
- `ruff check dexvantage tests scripts`: clean.
- Demo: `scripts/demo-phase0.sh`. Screenshots: `python scripts/screens.py` → `out/screens/` (not committed).

**Done**
- Repository, `.venv`, `pyproject.toml`, GPL-3.0-or-later (decided with the user in Phase 0), `THIRD_PARTY.md`,
  `CITATION.cff`, `.gitignore`.
- Ported from Vantage: `ui/theme.py` (unchanged), `ui/widgets.py`, `ui/charts.py`, `ui/colormaps.py`, `ui/icons.py`,
  `ui/icon_data.py` (+8 robot icons), `ui/navigation.py`, `ui/render_gl.py`, `ui/viewport3d.py`, `ui/dialogs/base.py`,
  `ui/dialogs/palette.py`, `core/mathutil.py`, `core/provenance.py`, `validation/` (suite, report, references,
  CLI), `workdir.py`, `sysenv.py`, the self-test runner and its EGL context.
- New: `core/frames.py`, `core/project.py` (.dexv + schema), `core/jobs.py` (studies, cache, summary),
  `ui/store.py` (undo + controller), `ui/scene.py`, `ui/panels.py`, `ui/context.py`, `ui/mainwindow.py`,
  `ui/dialogs/misc.py` (shortcuts, coordinate frames, about), `app.py`, `batch.py`, `selftest.py`.


## Next: Phase 2 (sensors and coverage)

- The datasheet library (cameras, depth sensors, tactile pads) with source URLs and dates; give "Wrist D405" and
  "Overhead D435" in the UR5e sample their model_id from Intel's datasheets, and the SO-100 sample's cameras theirs.
- Mounts and lens models; articulated cube-map occlusion against the posed robot meshes (`RobotModel.posed_triangles`),
  sparse voxels, coverage, the Camera view and depth validity.
- The optics, occlusion and depth checks; project obstacles (tables) in the collision distances.

## Spec additions (2026-11-06)

The user added advanced capabilities and ecosystem bridges during Phase 0. They are now in the specification
(sections and phases):

| Capability | Spec | Phase |
|---|---|---|
| VLA ViT-patch token resolution, crop warnings, Plücker ray-map export | §7.12 | 3 |
| Multi-finger self-occlusion matrix for dexterous hands | §7.14 | 3 |
| Action-chunk latency and temporal-ensemble alignment (air-grasp, overshoot map) | §7.13 | 5 |
| Visual-to-tactile handoff, sensory blind gap | §7.15 | 5 |
| Palm and dorsal micro-camera mounts | §7.7 | 6 |
| LeRobot v2.1 and v3.0 readers and writers | §8.2, §9.1 | 4 (writer), 8 (reader) |
| Targetless link-silhouette extrinsic check (render and compare) | §8.4 | 8 |
| FiftyOne tag export | §9.4 | 8 |
| Extrinsic perturbation export for augmentation and Isaac randomisation | §8.5 | 9 |
| Cross-embodiment canonical view warp | §7.16 | 10 |
| Rerun bridge; Isaac Lab camera configs | §9.4, §9.2 | 10 |
| Menagerie one-click import with per-model licence | §9.4 | 1 |
| cuRobo / pinocchio batch-IK backends | §7.3 | 3 (optional) |
| Robot-aligned 3D Gaussian Splatting view augmentation | §9.3 | 12 (optional, CUDA) |

Rule added with them (§1.1.6): policy inference times, input resolutions, crop rules and patch sizes are cited from
the model card or paper, or measured by the user; never constants in the code.

## Decisions and spec corrections

1. **Licence:** GPL-3.0-or-later, as Vantage (asked once, answered in Phase 0).
2. **Icon stroke:** the spec says "the same 1.5 px stroke style"; Vantage's icons use a 1.25 px stroke on a 16 px
   grid. The robot icons use 1.25 px so they match Vantage's, which is the intent of "the same style".
3. **Qt 6.12 regression:** a style-sheeted `QAbstractSpinBox` with no buttons gets a 1 px line edit, so number fields
   show nothing. Fixed in `ScrubSpinBox._fit_line_edit` (a no-op on earlier Qt). Vantage has the same code and will
   show the same symptom under Qt 6.12; Vantage was not changed (read-only).
4. **Exit codes:** validation and batch use 0 pass, 1 fail, 2 error, as §11 sets for `--check` (Vantage uses 6 for a
   failed validation).
5. **Quaternion angle:** `quat_angle` uses 2·atan2(|v|, |w|) of the relative rotation instead of 2·acos(|q₁·q₂|),
   which loses half the digits near zero; the frames checks reach 10⁻¹⁵ instead of 10⁻⁸.
6. **Provenance:** Vantage's vehicle-specific settings tables and mesh digest were removed; the content hash covers
   DexVantage's sections, mesh arrays by dtype, shape and bytes, and source files by SHA-256.
7. **Mounts:** a mount is "on a link" only when it names both a robot and a frame that is not a static one.
8. **The specification file:** the file the spec's instructions point to was empty on disk when this phase started;
   it was rewritten from the user's message with the additions above. It lives outside this repository.

## Environment of this phase

Python 3.14.4, PyQt6 6.12.0 (Qt 6.12.0), NumPy 2.5.4, SciPy 1.18.1, OpenCV 5.0.0, MuJoCo 3.17.0, Linux x86_64.
Headless EGL picked the integrated GPU (radeonsi) through the default display; the discrete GPU is reachable as an
EGL device. Phase 3's performance budget must say which GPU it measured.

## Open issues

- The GL viewport is only exercised headless (EGL) and through the QPainter fallback in tests; the on-screen
  QOpenGLWidget path needs a desktop check at each review gate (Phase 1: robots in the main view and in the wizard's
  embedded preview have only been drawn headless).
- `site/` (website) and `packaging/` are Phase 11.
- `kinematics.fk_pinocchio` has never run: pinocchio is not installed in `.venv` (optional extra).
- The 2F-85's eight linkage hinges and URDF mimic joints are independent entries of q; MuJoCo's equality constraints
  are not applied by kinematics. At the UR5e sample's home pose two non-adjacent linkage links are 3.6 mm apart, so a
  self-collision margin above that reports them as touching.
- Free and ball joints are held at the home keyframe and are not part of q.
- The SO-ARM100's joint limits are approximate (Menagerie's own note); its "near a limit" flags inherit that. The
  bundled UR5e's link offsets are rounded to the millimetre, so its flange is up to 1.3 mm (0.9 mm RMS) from UR's
  DH geometry; the DH check uses UR's table, not the bundled file. Both are documented in `docs/models/robots.md`.
- IK drag moves the frame's position only (orientation free); there is no option yet to hold the orientation.
- Environment distances cover only the static geometry in the robot's own file, not project obstacles (Phase 2).
- The Robot page installs IK drag and the Import trajectory command on the window from its constructor (a hidden side
  effect); they could move into `ui/mainwindow.py`. Registering it with workspace "placement" makes the Placement
  workspace open on the Robot tab.
- The Menagerie download shows progress in its dialog, not the window's job bar, and has no cancel button.
- `get_model` serialises the robot's records to JSON on every call for its cache key; a large humanoid called several
  times per frame may want a revision-based key.
- The Camera view's empty-state text overlaps its title when the right dock is squeezed short (seen with a tall
  Analysis dock); a Phase 0 placeholder that Phase 2 replaces.
- `MUJOCO_LOG.TXT` in the repository root holds MjSpec attach warnings from building the combined UR5e file; it is not
  tracked and can be deleted.
