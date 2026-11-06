# Build state

The resume point for the next session: what is done, what is next, open issues, and the decisions taken. Updated at
the end of every phase.

## Phases 3, 6 and 7: grasps, VLA tokens and dexterous hands; hardware engineering; calibration (done, 2026-11-06)

Built in parallel by separate agents (grasp sets; the grasp study; policies; VLA tokens; the multi-finger study and
the dexhand sample; bandwidth; cables; brackets; hand-eye solvers; hand-eye planning, recording and the coach), then
integrated, then reviewed four times (truthfulness, correctness, a researcher's workflows, spec completeness).

**Final gate (after the four reviews, 2026-11-07)**
- Tests: **1 538 passed**, 0 failed, 0 skipped (`python -m pytest tests`, offscreen, memory-capped at 7 GB, 757 s).
  The reviews added 45 regression tests: test_phase3_product_review 22, test_spec_review_p367 11,
  test_truthfulness_review 7, test_review_phase367 5.
- Validation: **74 of 74 checks pass** (`--validate out/validation`, full run 127 s). New since the review gate:
  `grasp.trajectory_closures` (grasps found where a scripted pick closes the gripper, 6.5 × 10⁻¹⁰ m against 10⁻⁶ m)
  and `handeye.sessions_joint` (two sessions in one bundle adjustment, 1.28 σ from the truth against 5 σ).
  `handeye.numpy_vs_opencv` still does not register (OpenCV 5.0.0); see spec correction 2.
- `dexvantage --self-test`: **15 of 15 ok** (43 s; its quick validation run 74 of 74 in 40 s).
- `ruff check dexvantage tests scripts`: clean.
- Batch, every Phase 3/6/7 flag on all three samples (times include start-up, RTX 5080 through EGL). Results exit 0;
  refusals are one plain sentence and exit 2.
  - SO-100: `--grasp` 10.2 s, `--vla-tokens` 0.5 s, `--dex-occlusion` 1.0 s (the two jaws), `--bandwidth` 0.3 s,
    `--cables` 0.7 s, `--brackets` 0.4 s, `--handeye-plan` 2.9 s (position-first planning for the 5-joint arm).
  - UR5e: `--grasp` 28.6 s, `--vla-tokens` 0.7 s, `--dex-occlusion` 2.4 s, `--bandwidth` 0.7 s, `--cables` 1.1 s,
    `--brackets` 0.7 s, `--handeye-plan` 6.8 s.
  - dexhand: `--dex-occlusion` 6.3 s, `--bandwidth` 0.8 s, `--cables` 1.2 s, `--brackets` 0.7 s, `--handeye-plan`
    3.9 s; `--grasp` and `--vla-tokens` refuse (the sample has no grasp set and no policy, by design).
  - `--handeye-solve` refuses on every sample (no capture session is stored; "Simulate a capture…" on the page makes
    one).
- Screenshots: `python scripts/screens.py` (58 images, 131 s, none unfinished). Looked at again:
  `phase3-ur5e-cables-page.png` (0 of 1 cables clear, "at risk", the headline names the cable and the run),
  `phase3-ur5e-handeye-solve.png` (σ 0.00304° about the camera's z axis, "meets the requirement", 12 simulated views,
  RMS 0.123 px, the covariance table), `phase3-ur5e-brackets-preview.png` (0/2 brackets ready, "check": the D405's
  1/4-20 hole position is not dimensioned by Intel; the wrist adapter's −2.0 mm clearance to forearm_link is shown as a
  finding). Nothing needed fixing.
- Hygiene: every tracked and untracked file (374, binaries included) grepped for personal paths, user and host names,
  e-mail addresses, keys (`sk-ant-`, `sk-`, AWS, GitHub, Slack, private-key blocks) and AI attribution lines: none.
  The only hits are the rules and tests that name the patterns and the optional `anthropic` extra (Phase 10).

**Review fixes (Phases 3, 6 and 7)**
- Truthfulness (every Phase 3/6/7 figure re-read at its primary source; all 8 policy entries hold up): a host whose
  USB ports are of mixed generations (Jetson AGX Orin dev kit: Gen 1 and Gen 2) was taken at its fastest; now the
  lowest, and the page says so (`bandwidth.host_usb3_generations`; "PCIe Gen4" is no longer read as USB). The
  bracket mass density cites Prusament PETG's data sheet (1.27 g/cm³); the T-slot suggestion no longer claims a
  survey; ISO 9409-1 and T-slot figures re-checked against their sources; the NanEyeC check reports that the 5 %
  overhead default disagrees with the maker's serial rate (about 26 % framing). The hand-eye solvers were compared
  once with OpenCV 4.10.0 (`docs/models/handeye.md`).
- Correctness (five bugs): grasp IK targets used the pads' TCP instead of the one the set was sampled with (69 mm off
  on the UR5e with a TCP override); the grasp study's occluder digest did not follow a moved robot base (stale cube
  maps); the result cache key did not follow policy and host library entries; a negative wrap radius was used as given
  (now the default); joint-state files in degrees converted prismatic joints too.
- Workflows (14 problems, on all three samples): the multi-finger study named pads "righ"/"lef" and stopped on the
  SO-100 ("Fixed has no contact patch"); the hand-eye planner found no pose for the 5-joint SO-100 (now
  position-first); number fields showed "0.00 mm" for figures not stated (now "—"); a zero-length cable run made NaN;
  the cables headline did not name the cable; the VLA headline said "resolved" in green with no task object counted
  (now "no task object", warning); bracket STL export meshed on the UI thread before checking the folder (now a
  window job); the multi-finger headline named the best mount twice; `--sample` help omitted dexhand; large grasp sets
  are sampled on a worker and added without a second normalising copy; policy messages keep acronyms; a ViT patch
  larger than the input is explained instead of counted as 0 tokens; simulated views are not flagged as moving. New:
  Calibrate ▸ Hand-eye solve ▸ "Simulate a capture…" renders a capture with a known offset that the solve recovers.
- Spec completeness: brackets never checked the project's cables (an import of a module that does not exist;
  every project said "no cable paths"); now `core/studies/cables.cable_polylines(project, q)` lays each cable at a
  pose and brackets use it. Added: grasps where the gripper closes in a trajectory (§7.3), several hand-eye sessions in
  one bundle adjustment, a fixed camera located through a board a wrist camera sees (§8.1, the solve page applies
  every camera as one step), the ghosted swept volume of a trajectory on the Tasks page (§13,
  `core/robot/swept.py`, `docs/models/swept.md`).

**Review gate (integration, 2026-11-06)**
- Tests: **1 493 passed**, 0 failed, 0 skipped (`python -m pytest tests`, offscreen, memory-capped at 7 GB, 639 s). Phases 3, 6 and 7
  added: test_policies 89, test_grasps 50, test_calib 41, test_bandwidth 37, test_calib_planner 27, test_vla_tokens
  24, test_brackets 23, test_grasp_visibility 23, test_dexhand 22, test_cables 19, test_calib_ui 5,
  test_phase3_integration 5 (1 493 collected; Phase 2 ended at 1 106).
- Validation: **72 of 72 checks pass** (`--validate out/validation`, full run 98 s; quick run 32 s): frames 6,
  kinematics 11, optics 10, occlusion 7, depth 4, **grasp 7, hand-eye 8, bandwidth 3, brackets 3, VLA tokens and
  policies 4, multi-finger 3, cables 4**, formats 2. Measured: grasp visibility against exact ray casting of the
  full scene 0 of 7 892 lifted contact points away from an edge (10⁻³); bootstrap coverage within 0.017 of 95 %
  (0.031); q-bucket against evaluation at each grasp's own q, mean 0.0024 of a patch (0.02); pads against MuJoCo's
  compiled pad boxes 3.6 × 10⁻¹¹ m (10⁻⁶ m); TCP against Menagerie's pinch site 0.0° (0.5°); samplers 4.4 × 10⁻¹⁶;
  object points 9.7 × 10⁻¹⁷ m; crop geometry against each rule's own code 2.5 × 10⁻¹³ px (10⁻⁶); token counts against
  ray casting 0 tokens (0); Plücker moments against OpenCV undistortion 1.3 × 10⁻¹⁵ (10⁻⁶); safetensors bit-exact;
  the finger matrix against a closed-form projection 0 (10⁻³); hit owners against all-pairs Möller–Trumbore 0
  (10⁻³); group cube maps against one map 2.0 × 10⁻⁸ m (10⁻⁶ m); data rates against the NanEyeC's stated serial
  rates (payload 79 % of them; bound 0); USB figures 0 (10⁻¹²); the budget re-computed independently 3.9 × 10⁻¹⁶
  (10⁻⁹); routed arc 1.7 × 10⁻¹⁶ m (10⁻⁹ m); helix bend 4.9 × 10⁻⁵ (5 × 10⁻³); loop arc 4.4 × 10⁻⁴ (5 × 10⁻³); snags
  against brute force 0 poses (0); bracket meshes 0 defects on all six kinds; flange disc volume 0.18 % (1 %); ISO
  9409-1 hole centres 4.2 × 10⁻⁵ mm (0.05 mm); hand-eye closed forms exact 1.8 × 10⁻¹⁵ m (10⁻⁹ m), noisy 5.9 mm (15
  mm), the adjustment 3.1 σ from the truth (5 σ), its covariance 0.56 of the Monte Carlo χ² band (1), AX = ZB with
  two arms 3.5 mm (15 mm), rendered ChArUco 0.113 px and AprilGrid 0.118 px (0.25 px), planner σ against the
  adjustment's ln ratio 0.0087 (0.095). `handeye.numpy_vs_opencv` registers only when OpenCV has `calibrateHandEye`;
  OpenCV 5.0.0 does not, so it has never run (spec correction 2).
- `dexvantage --self-test`: **15 of 15 ok** (34 s). New lines: "grasp" (20 grasps on the UR5e sample, 20
  feasible, CVS 100 % [100–100], 40 cube maps on the RTX 5080, 0.6 s), "brackets" (the ISO 9409-1-50-4-M6 disc,
  76 008 triangles, watertight, one shell, 21.0 cm³, 0.26 s), "handeye" (Park-Martin on 12 exact poses, 1.1 × 10⁻¹⁶ m
  and 6.6 × 10⁻¹⁷ rad, 0.02 s). Its validation line runs the quick suite (72 of 72).
- `ruff check dexvantage tests scripts`: clean.
- Batch, every new flag on the UR5e sample, all exit 0 (times include start-up, on the RTX 5080 through EGL):
  `--grasp` 24.7 s (200 grasps, 133 reachable: 64 obstacle collisions with the table, 2 out of reach, 1 no
  convergence; CVS 100 % [100–100], Wilson 97.2–100 %), `--vla-tokens` 0.7 s (π0; no task object in the sample, so
  the fingertips are counted: fewest 1 token, gripper_left_pad in the Overhead D435 at contact), `--bandwidth` 0.6 s
  (no host yet: the suggestion adds one USB 3 controller, then 68 %), `--cables` 0.9 s (suggested routes; the
  UR5e's ±2π limits make every external cable pull or bend, reported as findings), `--brackets` 0.6 s (no bracket
  types yet: the suggestion), `--handeye-plan` 6.5 s (20 poses, σ meets the requirement); `--dex-occlusion` on the
  dexhand sample 5.5 s (93.8 % of fingertip-cases seen, every fingertip seen in 69.2 % of the cases, the thumb worst at
  62.4 %). SO-100: `--grasp` 10.4 s (86 of 100 top-down pinches reachable, CVS 100 %), `--vla-tokens` 0.5 s (SmolVLA).
- **Performance budget (§16)**, measured with `scripts/measure_grasp.py --n 2000` on the UR5e sample (2 000 grasps,
  top-down, side and angled, seed 0; 1 322 reachable; 10 808 cube maps of 512 texels; sensitivity on 250 grasps,
  gallery of 6): **111.6 s on an NVIDIA GeForce RTX 5080** (EGL device 0, driver 595.99.02; IK 13.2 s, pre-contact IK
  1.0 s, visibility 68.8 s, what to move 27.3 s, gallery 1.1 s; 17.9 grasps/s, 171 maps/s) and **131.3 s on the
  Ryzen 9 9900X's integrated AMD Radeon** (radeonsi, EGL device 2; 15.2 grasps/s). 256-texel maps: 85.5 s;
  sensitivity off: 83.6 s. Without a GPU the numpy rasteriser draws 7 maps/s (about 25 × slower). About 5 of the 6 ms
  per map go to upload, six draws and the read-back of 6 × 512² floats (`core/occlusion.py`), which is why the
  integrated GPU is only 18 % slower. Hand-eye planner on the UR5e sample: 6–7 s (5 s of it 133 lazy occlusion
  checks). Multi-finger study on the dexhand sample: 4.6 s.
- UI-thread budget: no step over 100 ms on the samples (`scripts/measure_ui_phase3.py`, offscreen, idle machine, the
  3D view's paint left out as in Phase 2; table in `docs/models/performance.md`). Largest: on the dexhand sample the
  events after Apply 67.3 ms, Apply 60.7 ms, the Camera view overlay 53.2 ms; on the UR5e the hand-eye solve's Apply
  46.6 ms, the events after the Tasks page's Apply 46.4 ms, the brackets Apply 38.9 ms; every page shows in under
  11 ms. Under load (the full suite running alongside) the brackets Apply once took 146 ms. Building the UR5e window
  takes 0.82 s once (0.31 s in Phase 2: fourteen pages built at start). The hand-eye Import button ran the board
  detection on the UI thread (230 ms for 12 images; seconds for a real capture): now a window job.
- Screenshots: `python scripts/screens.py --only phase3` → `out/screens/phase3-*` (20 images, 102 s): the Tasks page
  with a set previewed, the Policies page and its crop outline in the Camera view, the grasp page after a run and its
  gallery, the VLA page and its patch grid in the Camera view, the bandwidth page before and after its suggested host
  (topology and margins), the cables page, the brackets page and the bracket in the 3D view, the hand-eye plan and
  solve pages (a synthetic capture rendered at 12 planned poses, imported and solved), the multi-finger matrix on the
  dexhand sample. Looked at and fixed: the grasp gallery spread its thumbnails over three wide columns (now one row per
  grasp, packed left); the VLA page's policy and camera combos were cut to "π0 (sam" and "Wrist D4", and its headline
  said "1 tokens" and repeated the advice; the bracket page's standoff, tilt and wall fields were invisible (plain
  QDoubleSpinBox under Qt 6.12) and their labels stayed when the fields were hidden; every number field glued its
  unit to the number ("0.30px"); the Bandwidth and Cables pages showed 0 for figures not stated (now "—", the wrap
  radius "default"); the Camera view legend ran past the image (now elided); the multi-finger matrix's legend said
  "darker is more hidden" while the dark theme's ramp makes more hidden lighter; the hand-eye session combo was cut;
  the board place's rotation fields had no names.

**Integration fixes**
- `ik.solve_batch` with one start per target (q0 of shape (N, nq)) always raised ("got 42 joint values"): it clipped
  the stack with `RobotModel.clip`. Two agents worked around it; fixed in `ik.py`, with a regression test.
- The UR5e and SO-100 samples carry a grasp set and a policy (decision 4), so `--grasp` and `--vla-tokens` run on
  them; tests and checks that sample their own set now start from the sample without them.
- The window has a Tasks menu (the Tasks and Policies pages registered there and fell back to Analysis); the Grasp
  study and Calibration workspaces no longer say "(Phase 3)" / "(Phase 7)".
- Memory: the full suite was killed by its 7 GB cap at 21 %. Each sample opening compiled its own MuJoCo model
  (the cache key held the robot's uid) and the model cache kept 32 of them (about 35 MB each); the digest cache kept
  up to 4 096 file blobs alive; windows closed by tests were never deleted (`deleteLater` needs the event loop).
  Now: compiled models are keyed by file content (mesh keys re-prefixed per robot), at most 12 models and 256 MB of
  digested bytes; the test fixtures let background work finish, run the deferred deletes and drop the caches after
  each module. Deleting windows exposed four lambdas connected to the global `THEME.changed` (scene, Outliner,
  Library, main window) that outlived their window and aborted the process on the next theme change; they are bound
  slots now (PyQt drops those with their object).
- The hand-eye Import button detected the board in every image on the UI thread (seconds for a capture); it now runs
  as a window job and adds the session when done (`import_capture(background=True)`; the synchronous call remains for
  scripts and tests).
- Five checks declared no unit (`grasp.bootstrap_coverage`, `grasp.sampler_closed_form`, `cables.helix_bend`,
  `cables.loop_arc`, `vla.safetensors_roundtrip`); `tests/test_store.py` took the live bandwidth study's result for
  the summary's.
- Self-test lines "grasp", "brackets", "handeye"; `scripts/screens.py --only phase3`; `scripts/measure_ui_phase3.py`.
- Docs: overview sections for every new model (anchors grasps, grasp, policies, vla, dexhand, bandwidth, cables,
  brackets, handeye, the ones the validation report links); user-guide chapters 5–7 and the batch flags;
  `docs/guide/samples.md` lists the samples' sets and policies; the grasps, grasp, policies and VLA guides say how to
  start from the sample without them; THIRD_PARTY (ported modules, the policy, cable, bracket, USB and Kalibr sources,
  the optional IK backends); README; CHANGELOG.

**Done**
- Phase 3: `core/grasps.py` and the Tasks page; `core/studies/grasp_visibility.py` (`--grasp`) and the Grasp
  self-occlusion page; `core/policies.py`, the policy library (8 entries) and the Policies page;
  `core/studies/vla_tokens.py` (`--vla-tokens`), `core/plucker.py` and the VLA page; `core/dexhand.py`,
  `core/studies/dex_hand_tactile.py` (`--dex-occlusion`, part 1 of §7.14), the dexhand sample
  (`core/robot/dexhand_sample.py`, the Shadow Hand bundled) and the Multi-finger occlusion page.
- Phase 6: `core/studies/bandwidth.py` (`--bandwidth`, `project.rig`) and its page with the topology editor;
  `core/cables.py`, `core/studies/cables.py` (`--cables`, `project.cables`), 3 cable entries and the Cables page;
  `core/meshing.py`, `core/brackets.py`, `core/studies/brackets.py` (`--brackets`) and the Brackets page.
- Phase 7: `core/calib/` (boards, detect, axxb, handeye, synthetic, planner, coach, capture, records),
  `core/studies/handeye.py` (`--handeye-plan`, `--handeye-solve`, `project.calibration`) and the two Calibrate pages.
- Validation: `checks_grasp.py`, `checks_grasps.py`, `checks_policies.py`, `checks_vla.py`, `checks_dexhand.py`,
  `checks_bandwidth.py`, `checks_cables.py`, `checks_brackets.py`, `checks_handeye.py`.
- Docs: `docs/models/` and `docs/guide/` grasps, grasp, policies, vla, dexhand, bandwidth, cables, brackets, handeye.

**Decisions (Phases 3, 6 and 7)**
1. **Contact visibility is measured at the contact location**, each pad point lifted 2 mm along its normal (a
   labelled default), by occlusion alone; an obliquity rule is optional and off. The inner faces of a parallel gripper
   face each other, so testing the faces themselves made every side-on view blind (0 % on the SO-100).
2. **Batch flags** follow the extension point's names: `--grasp` and `--cables` (the spec names `--grasp-study` and
   `--cable-flex`; spec correction 1).
3. **Hand-eye closed forms in NumPy**, written from the papers, because OpenCV 5.0.0's Python bindings carry only the
   method constants; OpenCV's functions are used automatically where present and every result names its backend.
4. **The samples carry a grasp set and a policy** (sample choices, labelled "(sample)"): UR5e 200 grasps (top-down,
   side, angled, seed 0) and π0; SO-100 100 top-down pinches and SmolVLA (its paper evaluates on SO-100 rigs). The
   poses are sampled once per process and given each opening's uids.
5. **Policy figures** are values with a source index and a quote; a figure no source states is null. The token cell
   is derived from tokens per image only when that tiles the input on a square grid, and labelled derived.
6. **Bandwidth**: protocol overhead 5 % per interface (a labelled default); a stream without stated bits per pixel has
   no rate until a measured one is entered; decimal units.
7. **Cables**: the routed-arc angle is not folded to half a turn (a joint turned further winds the cable); runs over
   several joints are straight chords with a note.
8. **Brackets** export STL with a JSON sidecar; the clearance check drops sampled joint states where the robot hits
   itself.
9. **The robot-model cache is keyed by file content**, so two robots read from the same files share one compile.
10. **Hand-eye requirement default** 0.5 mm and 0.05° (labelled); the sample's predicted σ is 20–50 times tighter.

**Spec corrections (Phases 3, 6 and 7)**
1. **§11 flag names:** `--grasp-study` is `--grasp` and `--cable-flex` is `--cables` (the study registry's batch
   names); the MCP names (`get_grasp_visibility`, `get_cable_flex`, …) arrive with the MCP server in Phase 10.
2. **§8.1 / §10 "validate against OpenCV's calibrateHandEye (Tsai, Park, Horaud, Andreff, Daniilidis) and
   calibrateRobotWorldHandEye":** OpenCV 5.0.0's Python bindings do not have them. The solvers are checked against
   synthetic ground truth (exact and noisy) and the comparison check registers itself only when the functions exist, so
   it does not run in the project's own environment. It was run once (2026-11-07) from a scratch environment with
   `opencv-python-headless` 4.14.0 over the project's packages: `--validate --models handeye` passed 9 of 9, with
   `handeye.numpy_vs_opencv` at 3.17e-15 m (tolerance 1e-9 m, exact poses). At the truthfulness review they were compared once with OpenCV 4.10.0 loaded from a
   scratch folder (`docs/models/handeye.md`): Park-Martin, Horaud-Dornaika, Daniilidis and Shah agree to rounding;
   Tsai-Lenz differs by 0.1–0.4 mm under noise (not traced); Andreff and Li re-solve the translation and differ by
   several times the pose noise (closer to the truth than OpenCV's).
3. **§7.3 "contact visibility = the fraction of the contact patch visible":** measured 2 mm in front of the patch
   (decision 1).
4. **§7.14 "Shadow, Allegro, Inspire, Ability, Unitree Dex3":** Inspire and Ability hands have no Menagerie model;
   Menagerie's G1 with hands does not name its hand, so it is not called Dex3. The Shadow Hand is bundled (Apache-2.0);
   Allegro, LEAP, DEX-EE, Aero and Sharpa Wave download on request.
5. **§7.6 GMSL:** the MAX96717 datasheet could not be fetched; GMSL link rates and deserialiser power-over-coax are
   "not stated" until the user enters them. USB Type-C current levels and Power Delivery are not modelled (the Type-C
   specification was over the fetch size limit).
6. **§7.12 Diffusion Policy:** the paper does not state how a 16:9 stream becomes 320 × 240; it is modelled as a
   stretch and labelled an assumption. Octo's 128 × 128 wrist input is stored but not applied per camera role.
7. **§7.7 / §9 STEP export through gmsh:** not done; STL plus a JSON sidecar.
8. **§7 "each study has an assistant tool and an MCP tool … and an exporter":** the assistant and MCP arrive in Phase
   10; the new studies export JSON with provenance (cables also CSV and HTML, brackets STL, VLA Plücker maps).
   Plücker maps are not yet a `--export plucker` batch format (Phase 9 exporters).
9. **§7.3 "taken from a recorded dataset":** grasp sets from datasets wait for the Phase 8 readers; hand-eye capture
   from ROS 2 topics and LeRobot episodes likewise ("arrives in Phase 8").

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


## Next: Phases 4, 5 and 8

- **Phase 4, MuJoCo ground truth:** the scene builder, renders with distortion remap, predicted against rendered
  visibility and depth (the grasp study's contact visibility is the first candidate), synthetic LeRobot episodes.
- **Phase 5, teleop, lighting, blur and sync:** teleop overlap and dead zones, glare and IR, motion blur and rolling
  shutter, time sync, action-chunk latency (§7.13, reads the policies' cited inference times and horizons) and the
  visual-to-tactile handoff (§7.15, part 2 of `core/studies/dex_hand_tactile.py`: `tactile_handoff`,
  `visibility_at` and the result's `handoff` key are in place).
- **Phase 8, data:** LeRobot, RLDS and MCAP readers, the auditor, plan against recording, targetless link-silhouette
  calibration, FiftyOne tags; then grasp sets from datasets and hand-eye capture from ROS 2 and LeRobot.
- Throughput (both the grasp study and the planner): read back only the cube-map texels the points need, or draw
  several maps per pass, in `core/occlusion.py`; upload each link's mesh once and pass link transforms as uniforms.

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

Phases 3, 6 and 7:
- **Grasp study:** the fingers stay at their home opening (not closed to the object); obstacles are not checked along
  the approach (pre-contact IK checks self-collision only); the 2F-85's linkage joints are independent entries of q
  (contact patches and the grasp centre are measured at home); the q-bucket cache rarely hits on sampled sets
  (0 % on the samples; it helps on duplicates and still trajectory phases), and a coarser bucket raises the worst
  case (0.25°: up to 11 % of a patch); pinocchio and cuRobo backends are written but untested (not installed); on both
  samples the score is 100 % because the wrist camera sees both pads (the Overhead D435 alone: 34.6 % [26–43]).
- **Grasp sets:** samplers check neither reach nor collisions; object grasps use the object's axis-aligned box
  (width an upper bound); the Outliner's Tasks group counts grasp sets but lists only trajectories.
- **VLA tokens:** silhouettes are not occlusion-tested; with a grasp set, static cameras and other robots use the
  on-screen q; IK there checks no collisions (4 seeds); the `safetensors` package is not in `.venv`, so only the
  check's own parser reads the files back. The UR5e sample has no task object, so its headline counts fingertips.
- **Policies:** the Camera view does not tell a page which sensor it draws (pages read
  `ctx.window.camview.view.sensor_uid`); the Library panel has no Policies or Cables group.
- **Multi-finger:** sample posture sweeps, not measured synergies; the Shadow Hand's tendon coupling is not enforced
  between keyframes; in grasp-set mode IK checks no self-collision by default (neighbouring fingers of a closed hand
  touch in their convex hulls); the dexhand sample stores about 4 900 pad triangles in the project (370 kB of JSON);
  the cube-map backends differ from exact ray casting by up to 6 points of a patch (ray casting is the default).
- **Bandwidth:** only the NanEyeC states a data rate for a stream mode (a serial rate including framing), so the
  stated-rate check bounds the payload only; the topology editor has no drag and drop and hubs cannot be renamed in the UI; per-link stream
  lists are editable only in project data; isochronous endpoint limits and Intel's simultaneous-stream tables are not
  checked; a ZED 2i defaults to 2208 × 1242 at 15 fps.
- **Cables:** snags are tested at home and trajectory poses only, against the cable's own robot (not other robots,
  objects or tables); suggested attachments are a heuristic that can still snag or over-bend; `core/cables.py` uses
  private members of `RobotModel` (a public joint anchor and axis accessor in `model.py` would be cleaner); the
  headline names the cable, then the run by its index ("Run 4").
- **Brackets:** STEP export not done; ISO 9409-1 sizes other than 50-4-M6 lack centring and dowel figures; the cable
  clearance (now read from the project's cables, spec review) is checked at home and the pose on screen, not over the
  joint sweep; the clearance sampling takes about 17 s on the UR5e
  (83 of 144 drawn states dropped as self-collisions of the gripper linkage); the self-test line meshes the disc only.
- **Hand-eye:** compared with OpenCV's solvers only once, by hand (spec correction 2); Tsai-Lenz is singular for a camera rotation
  near 180° in the static frame; end to end from rendered images the error is about 2–2.5 × the predicted σ (corner
  errors are not independent; treat σ as a lower bound); AprilGrid corner order checked only on DexVantage's own
  boards (not Kalibr's `kalibr_create_target_pdf`, which turns the code 180°); robot bases exact, no link-length or
  axis errors, no time offsets; capture takes the nearest joint sample; sessions store corner pixels in the project
  (about 25 kB per 25 images); the planner covers one camera and one board, poses but not the motion between them
  are collision-checked; writing refined intrinsics on Apply has no UI test; the quick validation run grew to about
  40 s (`tests/test_gui.py::test_validation_job_runs_in_the_background` waits up to 60 s).
- **Window:** building the UR5e window takes 0.82 s (0.31 s in Phase 2), fourteen pages built eagerly; Apply steps
  that edit the rig rebuild the Sensors and Policies editors (40–65 ms, past 100 ms on a loaded machine); the Analysis
  dock's tab bar no longer fits its tabs at 1680 px (it scrolls).
- **Tests:** the full suite peaks at about 4.7 GB (after test_validation) under its 7 GB cap; new test modules should not keep windows
  or samples at module scope.
- **From the four reviews (still open):**
  - Hand-eye: `handeye.numpy_vs_opencv` compares exact poses only and never fills its "noisy" detail; it should also
    compare noisy poses with per-method tolerances (Andreff and Li differ from OpenCV by design). It cannot register
    on OpenCV 5.0.0 anyway. Eye-to-hand Apply writes a camera mounted on a link back as a world-mounted pose
    (detaching it from its link): looks intended, to be confirmed by the lead engineer. The planner still plans one
    camera and one board (multi-camera problems are solved, not planned); poses are merged by joint vector only, so a
    static camera seeing a static board gets no coach advice.
  - Bandwidth: no per-controller USB generation (a mixed-generation host assumes the lowest; enter `practical_mbps`
    for the faster ports; a field needs project, schema and UI changes); USB 2.0 devices on a USB 3 controller are
    counted against the SuperSpeed capacity (conservative); the 5 % overhead default disagrees with the NanEyeC's
    stated serial framing (about 26 %), reported, not changed.
  - Grasps: "from a recorded dataset" waits for Phase 8; trajectory closures with no object (a test close, a
    regrasp) count as grasps. 10 000-grasp sets stall the UI 150 ms–1 s after Add set (worker threads holding the
    GIL in the project hash, JSON encoding and Camera view rendering; at 2 000 grasps every step is within 100 ms);
    a cheaper per-study cache key (the store revision) or process workers would fix it. Export on study pages hashes
    the project on the UI thread (`PV.stamp`, about 190 ms with 10 000 grasps): it should be a window job.
  - VLA: a suggested stream change is scored for one policy only (it may worsen another policy on the same camera);
    the Plücker export's "base" frame defaults to the project's first robot even when the policy's cameras ride on
    another; Plücker maps are not yet a `--export plucker` batch format.
  - Cables: the suggested routes are "at risk" on every sample (a slack loop through a link at home): on the UR5e
    this follows from its ±2π limits; on the SO-100 and dexhand the suggested attachments or loops may need work.
    Snags are sampled poses, not the continuous swept volume; the run from the last attachment to the host is not
    modelled.
  - Not re-fetched at the truthfulness review: openpi's `Pi0FASTConfig` action_horizon defaults and Isaac-GR00T's
    VideoCrop code (the paper-stated figures in those entries were verified).
  - Window: the Camera view dock gets about 100 px of image when the Analysis dock is tall, and its legend then
    covers the image. Simulated captures have no batch flag. The swept-volume ghosts are not drawn in the Camera view.
  - Docs: `docs/models/swept.md` is not yet linked from `docs/overview.md`, and the user guide does not yet describe
    the simulated capture, position-first planning for 5-joint arms, trajectory closures or joint sessions.

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
