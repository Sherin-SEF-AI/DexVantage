# Build state

The resume point for the next session: what is done, what is next, open issues, and the decisions taken. Updated at
the end of every phase.

## Phase 2: sensors and coverage (done, 2026-11-06)

**Review gate**
Final gate, after the three reviews (truthfulness, correctness, researcher workflows), re-run 2026-11-06:
- Tests: **1106 passed**, 0 failed, 0 skipped (`python -m pytest tests`, offscreen, memory-capped at 7 GB, 198 s).
  Phase 2 added 600: test_library 101, test_library_depth_entries 92, test_library_other_entries 138, test_optics 48,
  test_occlusion 20, test_voxels 8, test_coverage 14, test_depth_validity 28, test_sensors_ui 19, test_checks_mujoco
  10, test_obstacles 9, test_phase2_integration 9; the reviews' regression tests test_phase2_review_truth 21,
  test_phase2_review_correctness 64, test_phase2_product_review 19.
- Validation: **40 of 40 checks pass** (`--validate out/validation`, full run 48 s; quick run 7.3 s): frames 6,
  kinematics 11, optics 10, occlusion 7, depth 4, formats 2. Phase 2's, measured: OpenCV `projectPoints`
  1.8 × 10⁻¹² px and `fisheye.projectPoints` 5.7 × 10⁻¹³ px (tolerance 10⁻⁶); coverage pixel density against OpenCV
  finite differences 0.65 % (1 %); MuJoCo-rendered fiducials, centres 0.167 px (0.3) and masks 0.075 px (0.1); cube
  map against exact ray casting 2.5 × 10⁻⁵ of points away from silhouettes (10⁻³), on both samples 0.184 % (1.45 %,
  the measured share within half a texel of a silhouette); texel depth 4.2 × 10⁻⁶ (10⁻³); MuJoCo segmentation on
  scripted spheres 0 and on the UR5e 2.45 × 10⁻⁵ away from silhouettes (10⁻³); the Camera view against ray casting
  1.4 × 10⁻⁵ (10⁻³); coverage against an independent path 8.8 × 10⁻⁴ of the volume (5 × 10⁻³); σ_z against Monte
  Carlo 0.70 % (1.83 %); Intel's min-Z and FOV tables exact; the stereo-overlap formula 0.0009° (0.01°); finger
  shadows from both imagers against ray casting 0 (10⁻³); project-obstacle distance 2.6 × 10⁻⁷ m (10⁻⁶).
- `dexvantage --self-test`: **12 of 12 ok** in 9 s (new: "library", 40 entries valid, the UR5e sample's D405 and
  D435 resolve; "occlusion", a cube map of a scripted scene against exact ray casting, 165 ms, names the backend: gl
  on the RTX 5080 here, numpy with `DEXVANTAGE_NO_GL=1`; its validation line runs the quick suite, 40 of 40).
- `ruff check dexvantage tests scripts`: clean.
- Every library JSON validates against `library.schema.json` with jsonschema (40 files).
- Batch, both samples, all exit 0: `--analyse` (0.3 s, 0.4 s), `--coverage` (SO-100 0.6 s, 47 % seen by two or more
  cameras; UR5e 1.3 s, 26 %) and `--depth-validity` (SO-100 0.3 s, "There is no depth sensor in this rig: add one and
  choose its device from the library."; UR5e 1.2 s, Wrist D405 45.8 %, Overhead D435 61.3 % depth-valid, both
  maker figures RMS).
- Hygiene: every tracked and untracked file (binaries included) grepped for personal paths, user and host names,
  e-mail addresses, keys (`sk-ant-`, `sk-`, AWS, GitHub, Slack, private-key blocks) and AI attribution lines: none.
  The only hits are the rules and tests that name the patterns, the Copilot plan (the app's own feature) and the
  optional `anthropic` extra.
- UI-thread budget (§16): no step over 100 ms on the samples with the shader renderer; measured by
  `scripts/measure_ui.py`, recorded in `docs/models/performance.md`.
- Demo: `scripts/demo-phase2.sh` (self-test, validation report, both studies on both samples in batch, the UI
  timings, the screenshots, then the UR5e sample in the window).
- Screenshots: `python scripts/screens.py` → `out/screens/phase2-*` (Sensors, Coverage and Depth pages on both
  samples; the Camera view of the Wrist D405, Overhead D435 and the SO-100's cameras; the Library panel; the custom
  device dialog, blank and from the D435; the Camera view's empty state, normal and squeezed). Looked at and fixed:
  the Camera view legend was unreadable over the floor grid (now on a translucent plate), its info line said "Default
  camera (no device chosen)" twice, the Library list cut every Intel entry to "Intel RealSense Depth …" (now elided in
  the middle), the coverage page's slice combo was clipped. The Phase 0 empty-state overlap is gone (the empty state
  now only clips its last line when the dock is squeezed to about 110 px). At the final gate (38 screenshots, 28 s,
  none NOT-FINISHED) `phase2-ur5e-depth-page.png` and `phase2-so100-coverage-page.png` were looked at again: the
  Depth page reads 84.6 % "good" with the per-sensor breakdown and the D405's fingertip min-Z line; the Coverage page
  reads 47.3 % "gaps" with its advice, and the Camera view now carries the coverage slice (z = 0.060 m).

**Review fixes (Phase 2)**
- Truthfulness (every cited source fetched again on 2026-11-06 and compared field by field; all 40 entries hold
  up): the Depth study treated every maker's depth-error figure as an RMS. Each entry's `rms_error` now names its
  `metric` (Intel and Orbbec Gemini RMS, Femto Bolt standard deviation, Luxonis median absolute error, Stereolabs
  unspecified; an entry without one counts as unspecified), and σ_d is matched to the maker's figure only when it is an
  RMS; the Depth page says what the figure is. Notes corrected (the D405's stated range wording, the OAK-D SR's note
  no longer quotes a datasheet that does not exist). `optics.from_fov` also checks the pinhole closed form from fx,
  fy, cx, cy.
- Correctness (six bugs, each with a regression test that fails on the old code): coverage modelled every camera at
  its largest stream instead of the stream chosen for the sensor (`Sensor.stream`, 2.26× the pixels per mm of an
  848 × 480 stream); depth validity's resolution and min-Z did not follow that stream either; the GL worker client
  did not follow the worker's least-recently-used order, so after a reuse it could ask the worker to draw a dropped
  geometry; the result cache did not miss when a custom device changed under its id; depth validity counted the
  inside of a solid object as "occluded" (79 % where coverage said 91 %; both now share one task volume). Checked
  and found right: imager poses and stereo baseline signs, link-mounted poses at q, both rasterisers, the ray caster,
  voxel refinement and chunking.
- Workflows (both samples, offscreen with a 5 ms UI-thread watchdog): batch output that cannot be written is a
  sentence and exit 2; `--reproduce` of a coverage result keyed by robot uid compares by place; the custom device form
  refuses NaN, 0, inf, negative figures and a pinhole at 180° or more in its own words (a fisheye may exceed 180°); a
  custom device the optics cannot model is left out of the studies with its reason instead of stopping them; study
  failures are plain sentences that leave the status bar once the study runs; coverage notes are shown, one camera
  gets the right advice, a device missing from the library and switched-off sensors are said; the coverage slice
  stays visible at the slider's end; the Camera view draws the slice and the other sensors' min-Z zones; Export says
  why nothing happens before a run; the Inspector does not link a source that is not a URL and gives positions their
  unit.

**Integration fixes (Phase 2)**
- Performance: the 3D view's frame after a joint change took 385 ms on the UR5e (shader renderer, integrated GPU)
  and 108 ms on the SO-100, because the renderer re-merged 580 000 corners for smooth normals at every q (256 ms) and
  posing used a 3 × 3 product per triangle (33 ms). Now 68 ms [96 worst] and 27 ms: normals once per model in link
  frames, rotated at q (`RobotModel.pose_vectors`, `MeshDraw.vnormals`, `render_gl.vertex_normals(group=)`); one
  matrix product per link run in `posed_triangles` (identical numbers); bounds per column, no reorder copy of an
  all-opaque mesh. Occlusion and the Camera view pose faster too.
- The SO-100's wrist camera looked through its own fixed jaw (39 % of the volume hidden from it, 13 % seen); it now
  sits on a bracket 60 mm behind the jaws on the wrist side (11 % hidden, 54 % seen; two or more cameras see 47 %
  instead of 13 %). Its project note no longer says "no library model yet (Phase 2)"; the samples guide and the user
  guide no longer say the UR5e's cameras lack a lens or depth model.
- Library: `Entry.title` said "Intel RealSense RealSense Depth Camera D405"; a camera stating only a diagonal FOV
  (OmniVision OVM6948) got the 90° × 60° default instead of a square-pixel pinhole fitted to its 120° diagonal.
- `tests/test_gui.py::test_validation_job_runs_in_the_background` waited 8 s for a quick run that now takes 7–8 s;
  it waits up to 60 s.

**Done**
- `core/library` (40 entries: 21 depth, 9 camera, 2 tactile, 8 host; schema `resources/library.schema.json`),
  `core/optics` (six lens models, ported from Vantage `core/lens.py`), `core/scenegeom.py`, `core/occlusion.py`
  (GPU through EGL and numpy), `core/raycast.py`, `core/voxels.py`, `core/camrender.py`; studies
  `core/studies/coverage.py` (`--coverage`) and `depth_validity.py` (`--depth-validity`); project obstacles in
  `core/robot/collision.py`.
- UI: the Camera view (`ui/camview.py`), the Sensors and mounts page (`ui/pages/sensors.py`, `ui/rigedit.py`), the
  Coverage and Depth validity pages, the Library panel, the Inspector's device figures, Sensors ▸ New custom
  device… (`ui/dialogs/custom_device.py`), frusta from lens models, min-Z zones that ride with the wrist.
- Validation: `checks_optics.py`, `checks_optics_mujoco.py`, `checks_occlusion.py`, `checks_occlusion_mujoco.py`,
  `checks_camview.py`, `checks_coverage.py`, `checks_depth.py`, `checks_obstacles.py`, `mjrender.py`.
- Self-test lines "library" and "occlusion"; `scripts/screens.py` Phase 2 screens; `scripts/measure_ui.py`.
- Docs: `docs/models/library.md`, `optics.md`, `occlusion.md`, `voxels.md`, `coverage.md`, `depth.md`, `camview.md`,
  `performance.md`, linked from `docs/overview.md` (which now has a section per Phase 2 model, the anchors the
  validation report links to); `docs/guide/sensors.md`, `coverage.md`, `depth.md`; the user guide's chapter 4
  "Sensors and coverage" with the SO-100 and UR5e tours; README, CHANGELOG, THIRD_PARTY (ported modules, the
  library's cited sources).

**Decisions (Phase 2)**
1. **Pixel convention:** OpenCV's (pixel centres at integers, image [−0.5, W − 0.5]); Vantage used i + ½. Every lens
   model, the Camera view and the MuJoCo checks use it.
2. **One JSON file per device**, figures or null, every entry with sources (URL, title, date read, what it covers).
   An offset the maker does not give is placed as an assumption with `pose_stated: false` and an "ASSUMPTION" line in
   the note; the UI says "offset assumed".
3. **Lens models of built-in imagers are pinhole fits to the stated FOV** (no maker in the library states
   distortion coefficients). A stream of another aspect ratio is a centred crop of the native image; a diagonal-only
   FOV gives a square-pixel pinhole. Each is labelled on the imager.
4. **Coverage judges a depth sensor by its RGB imager** (the colour image is what a policy trains on); setting
   `depth_imager: 'depth'` switches. Depth validity treats the stereo pair (both imagers, both cube maps).
5. **Coverage and depth validity are `quick=False`**: they run on Run, a resolution change or Apply, not after every
   edit (0.6 s and 0.34 s on the UR5e sample with the GPU, seconds with numpy).
6. **The joint state q is view state**: pages pass it in the run's settings (`AnalysisController.run(sid, settings)`)
   on a snapshot; the result records q and where it came from; the project is not changed.
7. **Occlusion backend 'auto'**: the discrete GPU's EGL device first, a software renderer only when nothing else
   works (it still beats numpy), numpy otherwise; near plane 5 mm and visibility epsilon 1 mm are labelled defaults.
   The sensor's seat on its mount link (triangles within 25 mm, a default) is not an occluder; the rest of the link
   is (a wrist camera sees its own gripper).
8. **Smooth normals in link frames** (integration): computed once per robot model, rotated at each q; vertices of
   two links are never merged, so shading does not change as the arm moves.
9. **The SO-100's cameras keep no device**: no library entry is stated by its maker as an SO-100 camera; the
   labelled default camera is used and the project note says so.
10. **MuJoCo's principal-point sign** (`principalpixel`) was read off a render on MuJoCo 3.17.0 (the XML reference
    does not state it); `validation/mjrender.py` and the optics docs say so, and the Phase 9 MJCF exporter must use it.
11. **Materials:** `project.MATERIAL_TAGS` has no 'specular'; 'glossy' and 'metallic' are read as specular in depth
    validity.

**Spec corrections (Phase 2)**
1. **§6 "the minimum-Z at each resolution (stored as a table)":** only Intel (and Luxonis, partly) state min-Z per
   resolution. Stereolabs (all ZED) and Orbbec's Gemini 2 state one device-wide minimum depth; their `min_z` table is
   empty and the stated `min_range_m` is used, labelled "not per resolution". Nothing was filled in per resolution.
2. **§6 "the depth FOV and the RGB FOV with their extrinsic offset":** Intel does not state the RGB-to-depth offset of
   the D415, D435, D435i, D455 or D457 (the D405's colour comes from its left imager), nor the projector's position;
   Orbbec does not publish the Femto Bolt's RGB offset or tilt; Gemini 2 gives only sideways offsets on a drawing.
   These are labelled assumptions until a unit is calibrated. The Gemini 335/336 offset is stated (Table 4-8-1).
3. **§6 lens figures:** no maker in the library states distortion coefficients (Intel: "Distortion ≤ 1.5 %") or the
   stereo matcher's subpixel accuracy; σ_d is the labelled default 0.08 px (§7.2 already allows it).
4. **§6 tactile "piezoresistive fingertip arrays where published":** none was found with figures published by its
   maker (Weiss DSA 9205 and PSYONIC searches found only second-hand figures); no entry.
5. **§6 "Library ▸ New custom device…":** the window has no Library menu; the command is Sensors ▸ New custom device…
   and in the Library panel's right-click menu.
6. **§7.2 condition (1) "the RGB and projector FOVs for structured light":** the camera that sees the pattern makes
   the depth, so the entry's reference (depth) imager and the projector cone are used.
7. **§7.2 depth FOV:** Intel states the depth FOV as the stereo overlap at one distance (datasheet §4.4); the
   modelled overlap of the two imagers is up to about 3° narrower near the camera, within Intel's ±3° unit spread.
8. **§14.2 "one wrist camera and one third-person camera" with library models:** no maker states an SO-100 camera;
   the placements are labelled sample placements on the default camera (decision 9).

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


## Next: Phase 3 (grasp self-occlusion, VLA tokens, multi-finger matrix)

- Grasp sets, batched IK and batched occlusion; the Contact Visibility Score with its CI; galleries, sensitivity and
  "what to move"; the 2 000-grasp performance budget measured on a named GPU (§7.3).
- VLA token resolution (ViT patch grid), crop warnings and Plücker ray-map export (§7.12), from cited model cards.
- The multi-finger self-occlusion matrix for dexterous hands (§7.14).
- Throughput groundwork from Phase 2: about 49 ms of the 84 ms per joint vector was posing and the geometry digest
  (posing is now 12 ms); upload each link's mesh once and pass link transforms as uniforms, for both the occlusion
  cube maps and the 3D view.
- The Camera view overlays of the studies (coverage, depth validity) that Phase 2 left out.

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
EGL device. Phase 2's occlusion backend picks EGL device 0, the NVIDIA GeForce RTX 5080 (driver 595.99.02); the
self-test's OpenGL line and `scripts/measure_ui.py` use the default display (radeonsi). Phase 3's performance budget
must say which GPU it measured.

## Open issues

Phase 2:
- **Camera view study overlays are outlines only**: the Coverage page draws its slice outline and the Depth page the
  other sensors' min-Z zones (product review); the depth-valid mask, the finger shadows in the sensor's own image
  (§4) and the coverage heatmap projected into the image are not drawn. §7.1's isosurfaces are not done; coverage
  slices are TriDraw quads (SceneProvider has no `points(vp)` for PointDraw voxels).
- From the truthfulness review: the LI-OV2312-GMSL2-110H's only stream (1600 × 1300 at 30 fps) is the USB 3.0 kit's
  rate, the GMSL2 rate is not stated (the note says so, but a bandwidth study would read it as the GMSL rate); the
  Basler daA1440-220um's stream carries only 227.7 fps (minimum exposure, Device Link Throughput Limit off; the
  default 98.9 fps is in the note only); the D415 datasheet gives two projector FOVs (Table 3-49 67° × 41° × 75°,
  Table 3-16 64° × 41° × 72°), the entry and `depth.d400_tables` use 3-49 and the note gives both; the depth FOV is
  used as each IR imager's FOV and then intersected again, so the overlap is counted twice (2.9–3.6° for the D405,
  0.7–1.9° for the others; documented as a known limit, needs the rectified imager FOV Intel does not state); the
  custom device dialog has no error-metric field (custom figures count as "unspecified" and a copy of a built-in
  keeps the original's metric); two checks declare a tolerance other than the one they apply
  (`occlusion.samples_vs_raycast` declares 0.01 and applies the measured half-texel band, now 0.0145;
  `depth.sigma_z_monte_carlo` declares 0.02 and applies 0.0183); not fetched again: the Luxonis per-sensor pages, the
  Intel store pages other than the D405's, the ZED 2i and ZED Mini store pages and the Leopard product pages (all
  cross-checks; the primary documents were re-read).
- From the correctness review: the depth study's suggested resolution is the mode with the most depth-valid volume,
  which tends to be the lowest resolution (a smaller min-Z only adds volume) even when 848 × 480 gains nearly as much;
  a rule weighing the gain against σ_z may serve better. The seat rule leaves out only the mount link's triangles, so
  a sensor against the next link (a gripper base modelled as its own link) is protected only by the 5 mm near plane
  (no sample camera has anything within 3 cm). GPU depth is float32 with sub-texel vertex snapping (about 10⁻⁴
  relative at n = 16, under 10⁻⁵ at n = 512): backends can disagree at silhouettes and slivers. The Gemini 335/336 RGB
  offset reads Table 4-8-1 ("X 14.00, Z 2.02 mm from the left IR") in the left IR's optical frame, giving body
  (+2.02, −14, 0) mm; the sign convention was not re-checked against the datasheet. Static mount frames (cage,
  tripod, table) have no transforms of their own, so a pose on them is read in the world frame.
- From the workflow review: mounting a sensor on a link keeps its place in the world by default (a D435 moved from a
  tripod to the SO-100's Lower_Arm hangs 0.5 m from the arm; untick "Keep its place" to keep the pose numbers); a
  "place at the link origin" option would help. No global `sys.excepthook` (as Vantage): an unhandled exception in a
  Qt slot ends in PyQt6's default abort, not an explained message (predates Phase 2; none hit). The Inspector's Model
  row cuts long device names (full name in the tooltip), the Library's last row reads "New c…vice…", and the
  Inspector's position fields with their "m" label are crowded at the default dock width.
- CHANGELOG.md does not yet list the review fixes above (this file does; `docs/models/depth.md` documents the
  error metric).
- **Defocus is never computed for built-in entries**: no maker states a fixed-focus lens's focus distance and the
  RealSense entries state no pixel pitch. Measured figures go in the study settings (`optics`), with no form for them.
- `--reproduce` of a coverage or depth result made at the page's live q cannot re-run at that q (q is view state);
  the result records it. `--reproduce` could apply the recorded settings.
- Not modelled: sensor housings as occluders, the projector's own shadow, min-Z from the front glass (it is from the
  optical centre), OAK `min_z_extended` and Gemini `min_z_disparity_256`, USB 2.0 stream modes, simultaneous-stream
  limits and IMUs; a ZED 1080p stream is a crop whose FOV the loader takes as the full one (no per-mode FOV stated).
- Library data gaps: the NanEyeC's 120° FOV names no axis (default shown); the Basler dart and FLIR Blackfly S are
  sold without a lens (default FOV until the user enters one); the RealSense T265 (stated Kannala-Brandt) is not in
  (its datasheet could not be fetched); no USB 2.0 host entry (the usb.org page read does not state 480 Mbit/s); the
  MAX96712 link rate and lane count are null (analog.com timed out); e-con's datasheets behind download pages were
  not read. Entries carry keys the loader does not read (`ideal_range_m`, `range_text`, `min_range_m` outside depth
  validity, `max_range_by_mode`, `depth_start_mm`, projector `wavelength_nm`, stream `what`): the shared contract
  should standardise or drop them. The host fields' meaning for Jetson entries (ports, lanes, stated_mbps) is not
  defined in `docs/models/library.md`.
- The QPainter fallback 3D view (no OpenGL) takes 0.16 s per SO-100 frame and 0.59 s per UR5e frame; a decimated
  mesh for that path is the fix. The shader renderer still poses and uploads the whole mesh at each joint change
  (68 ms median on the UR5e, integrated GPU).
- `occlusion.samples_vs_raycast` takes about 20 s in a full run (both backends on 24 maps); two points of 80 000 in
  the full occlusion run disagree with ray casting away from a silhouette (within tolerance; not investigated); the
  radeonsi n = 256 wrist timing (58 ms) is from one quick run.
- In a GUI process the GPU occlusion context is made current per build under a lock; builds run on worker threads;
  building on the Qt GL thread was not tested.
- Sensors page: not registered with workspace "placement" (the Placement workspace opens on the Robot tab); mount
  tolerance is shown, not edited; per-imager streams are not editable (a stream applies to every imager that streams
  that size); a sensor's unit calibration is cleared when its device changes (undo restores it); the Inspector's
  Position row is in metres while the page edits in mm.
- Project format: `TaskObject` mesh storage accepts three layouts of `project.meshes[key]` and surfaces gained
  optional `center` and `thickness`; `docs/project-format.md` and the schema do not record them yet. The Inspector
  shows material tags but cannot edit them.
- Optics: a Brown model's first build takes about 14 ms (its valid-angle search; cached after); `pixels_per_metre` is
  the geometric mean (no worst direction); `pixel_rays`/`unproject` return NaN for pixels with no ray (fisheye corners).
- Collision: other robots are not obstacles, a robot's base is not checked, concave meshes are taken as their hulls;
  IK does not avoid obstacles (`docs/models/ik.md` says so).
- The Camera view and the occlusion GPU path are timed headless only; the on-screen GL views need a desktop check.

Earlier phases:

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
- The Robot page installs IK drag and the Import trajectory command on the window from its constructor (a hidden side
  effect); they could move into `ui/mainwindow.py`. Registering it with workspace "placement" makes the Placement
  workspace open on the Robot tab.
- The Menagerie download shows progress in its dialog, not the window's job bar, and has no cancel button.
- `get_model` serialises the robot's records to JSON on every call for its cache key; a large humanoid called several
  times per frame may want a revision-based key.
- `MUJOCO_LOG.TXT` in the repository root holds MjSpec attach warnings from building the combined UR5e file; it is not
  tracked and can be deleted.
