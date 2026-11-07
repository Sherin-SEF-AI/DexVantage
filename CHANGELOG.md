# Changelog

All notable changes to DexVantage. The format follows Keep a Changelog; versions follow semantic versioning.

## [Unreleased]

### Phases 4, 5 and 8: MuJoCo ground truth; teleop, lighting, blur and sync; data
- MuJoCo ground truth (`core/sim/scene.py`, `core/sim/render.py`, `core/studies/groundtruth.py`, `--groundtruth`,
  Simulate ▸ Rendered against predicted visibility…, Simulate ▸ Build the MuJoCo scene…): the project as one MuJoCo
  scene, renders through every lens model (distortion by remapping an oversampled base or cube faces) in an EGL worker,
  and the grasp study's contact visibility and the depth study's voxels compared with them: 92.8 % of contact points
  agree on the UR5e sample (over its grasps, home and scripted pick), 84.2 % on the SO-100, depth 99.8 %.
  `validation/mjrender.py` gained a persistent session,
  per-job camera pose and intrinsics and the EGL device choice.
- Synthetic LeRobot episodes (`core/sim/episodes.py`, Simulate ▸ Generate synthetic episodes…, `--synth-episodes
  OUT_DIR [--synth-settings JSON]`): v2.1 or v3.0 datasets rendered from the project with seeded faults and their
  ground truth (`dexvantage_truth.json`).
- Teleop overlap and dead zones (`core/studies/teleop.py`, `--teleop`, the Teleop overlap page takes over the
  placeholder command): leader arms and GELLO joint to joint, VR and space-mouse volumes by scale and offset, the
  comfortable workspace, dead zones by cause, a base move suggestion, bimanual overlap and handover; the SO-100 sample
  carries its leader arm.
- Lighting, glare and IR wash-out (`core/materials.py`, `core/lights.py`, `core/studies/lighting.py`, `--lighting`):
  lights in `project.workspace.lights` (3 cited Elgato entries in a new `light` library kind), a GGX microfacet BRDF,
  relative (ISO 12232 auto-exposure) or absolute saturation, IR wash-out of active-stereo depth (the RealSense
  projectors' stated optical power added), recommendations with their figures before and after, a glare map in the
  Camera view.
- Motion blur and rolling shutter (`core/studies/motion_blur.py`, `--motion-blur`) and time sync
  (`core/studies/time_sync.py`, `--time-sync`, `project.rig['sync']` clocks and triggers): smear and skew per imager
  along trajectories, stamp error intervals, the action-label error, suggested exposures and hardware triggers; the
  RealSense D405 and D435 entries gained their stated exposure ranges and trigger support. The UR5e sample gained a
  scripted pick trajectory.
- Action-chunk latency (`core/studies/action_chunk_sync.py`, `--action-chunk`, §7.13): the policy's sensor-to-action
  loop term by term with sources, overshoot and air-grasp at the demonstrated contact, an overshoot map, a delay offset
  and temporal-ensemble weight stored on the policy. It uses exposures set on the motion blur page.
- Visual-to-tactile handoff (§7.15, part 2 of `core/studies/dex_hand_tactile.py`): the share of each contact patch
  seen against the distance to contact, tactile coverage at contact, the sensory blind gap; a chart on the
  multi-finger page.
- Datasets (`core/datasets/lerobot.py`, `bag.py`, `rlds.py`; `core/rosbag.py` ported from Vantage): LeRobot v2.1 and
  v3.0 read and write, ROS 2 bags (MCAP and SQLite) and RLDS TFRecords read into one Episode shape.
- The dataset auditor (`core/audit/`, `core/studies/audit.py`, `--audit OUT.json --dataset PATH`, Data ▸ Audit a
  dataset…): per-episode timing, duplicates, desync, drift, exposure, blur, the hidden gripper, campaign mount slip
  (ported from Vantage `core/online_calib.py`), verdicts, JSON and HTML reports and a FiftyOne export. Plan against
  recording (`core/studies/plan_vs_recording.py`, `--plan-vs-recording`).
- Targetless link-silhouette calibration (`core/calib/render_calib.py`, `core/studies/silhouette_calib.py`,
  `--silhouette-calib OUT.json --dataset PATH`, Calibrate ▸ Silhouette calibration): a static camera's mount and its
  drift from the robot's silhouette in recorded frames.
- Data links: grasp sets from a recorded dataset (Tasks ▸ From a dataset…) and hand-eye capture sessions from a
  LeRobot dataset or ROS 2 bag (`core/datasets/joints.py` matches joints to the model).
- Validation: 39 new checks, 113 in all (ground truth 6, teleop 4, lighting 9, timing 7, the handoff 1, data 4,
  formats 4, data links 2, silhouette 2). Self-test lines "teleop", "lighting", "timing", "datasets", "audit"; the
  self-test's validation line runs the quick run's cheap checks (`ValidationSettings.cheap_only`); the validation run
  frees process caches between models (peak 4.8 GB instead of over 6 GB).
- `scripts/screens.py --only phase458`, `scripts/measure_ui_phase458.py`, `scripts/measure_groundtruth.py`,
  `scripts/measure_synth.py`, `scripts/record_vantage_slip.py`.

### Changed (Phases 4, 5 and 8 integration)
- Four validation checks declared no unit (two teleop, the device mapping, the slip comparison); a test now checks
  every registered check's unit, reference and quantity without running it. `test_validation` runs every check in
  quick mode (the full run is `dexvantage --validate`), and the test fixtures free the process caches after each
  module, so the full suite passes under its 7 GB cap.
- Fixed an abort: a background Camera view render on the silhouette page finishing after the project was replaced
  raised inside a Qt slot; the silhouette and ground-truth pages now drop late results.
- The action-chunk study runs on request, not live: on a project without a policy its refusal took the status bar
  after every edit.
- The time sync study says "out of sync" with the figure as a lower bound (≥) instead of "at least out of sync".
- The lighting headline's status follows a large depth loss as well as glare ("low glare; depth washed out"), with
  labelled default thresholds (5, 15, 30 %).
- `--dataset` also feeds `--silhouette-calib`; a refusal with the same reason for every episode says it once.
- Teleop: an empty dead-zone list says why; ground truth: the worst-disagreement thumbnails are packed left; plan
  against recording: the visibility table's headers are not cut; the handoff and action-chunk curves keep their y label
  below the title.
- The quick validation run is shorter: the audit campaign uses 120 rows per episode and the silhouette checks shorter
  episodes and 10 bootstrap resamples (tolerances unchanged); the suite's MODELS lost the unused `sim` entry and
  `timing` names the action-chunk module.

### Fixed (Phases 4, 5 and 8 reviews and final gate)
- Truthfulness: the Elgato lights' stated masses (1.3 kg, 735 g, 1.07 kg) were null; the projector power is the laser
  maximum (D400 datasheet Table 4-23); the motion-blur word over an upper bound is "may be blurred" (the headline and,
  since the final gate, the trajectory's segment table); the ground-truth advice no longer calls voxels "surface
  pixels"; `lighting.white_furnace` is checked by quadrature (8.1 × 10⁻¹² against 10⁻⁶) with the Monte Carlo estimate
  inside its band; `audit.plan_vs_recording` compares rotation vectors (0.20° against 0.3°), not their sizes.
- The result cache served the old result when a recorded dataset changed at the same path: the audit,
  plan-against-recording
  and silhouette results now key on the dataset's fingerprint, and the key follows lights, teleop devices, the rig's
  sync block and the data section. Tests pin the lens remap direction, LeRobot frame seeking, CDR alignment, CRC-32C,
  ACT's ensembling order and the blind gap.
- Bad inputs are explained: damaged LeRobot files and bags, an empty bag (it audited as "ready"), one damaged episode
  (dropped, the audit goes on), a recording of another robot (plan against recording crashed), a `StudyError`
  subclass on a page, an unexpected batch error, `--dataset` alone (it opened the window), a dataset path with
  nothing there. A long status message no longer widens the window; the plan preview names sensors; the silhouette
  page follows Apply and undo.
- Added for the spec: `--teleop-overlap` and `--sync` aliases; `--audit-html OUT.html` and `--fiftyone OUT_DIR` with
  `--audit`; the check `formats.mjcf_scene`; the recorded camera-joint offset against the time-sync prediction in plan
  against recording; object materials on the Lighting page; LeRobot datasets streamed from the Hugging Face Hub
  (`core/datasets/hub.py`, public datasets, cached per commit).
- The audit's "Episode readable" row counts every episode ("2 of 3", not "0 of 1"); "1 gap", not "1 gaps"; the
  action-chunk Apply is disabled when its recommendation is already stored.
- Tests: `test_validation`'s in-process run of every check runs first, so the full suite (1 886 tests) passes under
  its 7 GB cap (it was killed at 95 %, peak 7.4 GB).

### Phases 3, 6 and 7: grasps, VLA tokens, dexterous hands; hardware engineering; hand-eye calibration
- Grasp sets (`core/grasps.py`, the Tasks page): sampled over a task volume (top-down pinch, side, angled, bimanual
  handoff), imported from CSV or JSON, or taken from a trajectory; contact pads and the grasp centre measured from the
  gripper's geometry. The UR5e and SO-100 samples now carry a grasp set and a policy, so every study runs as they open.
- Grasp self-occlusion (`core/studies/grasp_visibility.py`, `--grasp`, the hero study of spec §7.3): batched IK with
  reasons for infeasible grasps, contact visibility per fingertip and camera through cube maps cached by q-bucket,
  the Contact Visibility Score with its bootstrap 95 % interval and Wilson's interval, pre-contact visibility, a
  worst-case gallery and "what to move". 2 000 grasps on the UR5e sample in 111.6 s on an NVIDIA GeForce RTX 5080
  (131.3 s on the integrated Radeon); optional pinocchio and cuRobo IK backends (untested here).
- Policies (`core/policies.py`, `project.policies`, the Policies page): 8 cited entries (OpenVLA 7B, π0, π0-FAST,
  SmolVLA base, ACT, Diffusion Policy, Octo-Base, GR00T N1 2B), every figure with its source and quote, null where no
  source states it; user overrides with their source; six crop rules.
- VLA token resolution (`core/studies/vla_tokens.py`, `--vla-tokens`): tokens per task object and fingertip at
  pre-grasp and contact, crop warnings, the patch grid in the Camera view; Plücker ray-map export as `.npy` or
  `.safetensors` (`core/plucker.py`, ported from Vantage).
- Multi-finger occlusion (`core/studies/dex_hand_tactile.py`, `--dex-occlusion`): the finger × camera matrix and
  mount comparison; the `dexhand` sample (MuJoCo Menagerie's Shadow Hand, Apache-2.0, on the UR5e).
- Bandwidth, hosts and power (`core/studies/bandwidth.py`, `--bandwidth`, `project.rig`): data rates from stated
  stream modes, the host topology editor, controller and link margins, disk and USB bus power.
- Cable flex across joints (`core/cables.py`, `core/studies/cables.py`, `--cables`, `project.cables`): the routed
  arc model, slack, bend, pull and snag flags, the harness sheet; 3 cited cable entries.
- Brackets (`core/meshing.py`, `core/brackets.py`, ported from Vantage; `core/studies/brackets.py`, `--brackets`):
  watertight STL for ISO 9409-1 flange adapters, 2020/4040 T-slot clamps, pan-tilt plates, palm and dorsal sleeves,
  with the clearance check; mounting holes added to the RealSense D405, D435 and D435i entries from Intel's datasheet.
- Hand-eye calibration (`core/calib/`, the Calibrate menu's two pages, `--handeye-plan`, `--handeye-solve`,
  `project.calibration`): ChArUco and AprilGrid boards, detection and PnP, seven closed-form solvers written from the
  papers (OpenCV 5.0.0's bindings have none), one sparse bundle adjustment with covariance, the D-optimal pose planner,
  recording from image folders and joint-state files, and the coach.
- The window: a Tasks menu (Tasks, Policies); the Grasp study and Calibration workspaces open their pages.
- Validation: 32 new checks, 72 in all, 72 of 72 pass (grasp 7, VLA tokens and policies 4, multi-finger 3,
  bandwidth 3, cables 4, brackets 3, hand-eye 8).
- `--self-test`: "grasp" (a 20-grasp run on the UR5e sample), "brackets" (one watertight STL), "handeye" (a synthetic
  AX = XB solve).
- `scripts/screens.py --only phase3` (every new page), `scripts/measure_ui_phase3.py` (no UI-thread step over 100 ms),
  `scripts/measure_grasp.py` (the 2 000-grasp budget), `scripts/bundle_dexhand.py`.
- Docs: `docs/models/grasps.md`, `grasp.md`, `policies.md`, `vla.md`, `dexhand.md`, `bandwidth.md`, `cables.md`,
  `brackets.md`, `handeye.md` and their worked examples in `docs/guide/`; overview sections and user-guide chapters 5
  to 7.

### Fixed (Phases 3, 6 and 7 integration)
- `ik.solve_batch` with one start per target (q0 of shape (N, nq)) raised: it clipped the stack with
  `RobotModel.clip`, which takes one joint vector.
- Memory: robot models compiled from the same files are shared whatever the robot's uid (each opening of a sample
  compiled its own), the model cache holds 12 models, and the digest cache at most 256 MB of file bytes.
- Number fields: a gap between the number and its unit ("0.30 px", not "0.30px"); the Brackets page's standoff,
  tilt and wall fields were invisible (a plain spin box under Qt 6.12, decision 3 of Phase 0); the Camera view's
  legend is elided to the image.
- The hand-eye Import button detects the board on a worker thread (it blocked the window for seconds).
- Lambdas on the global theme signal outlived their window; closing a window and changing the theme then aborted.
- Pages: the grasp gallery packs each grasp's views in a row; the VLA headline names a fingertip count in the
  singular; "not stated" figures on the Bandwidth and Cables pages show "—" instead of 0; the multi-finger matrix's
  legend follows the theme (lighter is more hidden in the dark theme).

### Added (Phases 3, 6 and 7 reviews)
- Grasp sets from a trajectory take the grasps where the gripper closes (`grasp.trajectory_closures` check).
- Several hand-eye sessions solve in one bundle adjustment, which locates a fixed camera through a board a wrist
  camera sees; the solve page applies every camera as one undo step (`handeye.sessions_joint` check).
- The Tasks page draws a trajectory's swept volume as ghosts (`core/robot/swept.py`, `docs/models/swept.md`).
- Calibrate ▸ Hand-eye solve ▸ "Simulate a capture…" renders a capture with a known offset, so the loop can be tried
  without a rig.
- The hand-eye planner works on 5-joint arms such as the SO-100 (position first).
- Validation: 74 checks, 74 of 74 pass. Tests: 1 538 pass.

### Fixed (Phases 3, 6 and 7 reviews)
- Grasp IK targets use the TCP the set was sampled with (they used the pads' TCP: 69 mm off on the UR5e with a TCP
  override).
- The grasp study's cached occluders follow a moved robot base; the result cache misses when a policy or host
  library entry changes.
- Brackets check clearance to the project's cables (the check never ran: it imported a module that does not exist).
- Bandwidth: a host with mixed USB generations is taken at its lowest and the page says so; "PCIe Gen4" is not read
  as a USB generation.
- Cables: a negative wrap radius means the default; a zero-length run no longer makes NaN; the headline names the
  cable.
- Hand-eye: joint-state files in degrees leave prismatic joints in metres.
- Multi-finger: gripper pads are named at word boundaries ("right", not "righ"), the SO-100's fixed jaw has its links
  (the study stopped with "Fixed has no contact patch"), the headline names the best mount once.
- VLA: the headline is a warning ("no task object") when no task object is counted, not a green "resolved"; a ViT
  patch larger than the input is explained instead of counted as 0 tokens; policy messages keep acronyms.
- Number fields show "—" for figures not stated (they showed "0.00 mm").
- Bracket STL export checks the folder first and meshes as a window job; large grasp sets are sampled on a worker.
- `--sample`'s help lists the dexhand sample.
- Sources: the bracket mass density cites Prusament PETG's data sheet; the T-slot suggestion makes no survey claim;
  the NanEyeC check reports where the 5 % overhead default disagrees with the maker's framing. The hand-eye solvers
  were compared once with OpenCV 4.10.0 (`docs/models/handeye.md`).

### Phase 2: sensors and coverage
- The hardware library (`core/library`, schema `resources/library.schema.json`): 40 entries from makers' datasheets
  and product pages, each with its source URL, title and the date read (2026-11-06): 21 depth cameras (Intel
  RealSense D405, D415, D435, D435i, D455, D457; Stereolabs ZED 2i, ZED Mini, ZED X, ZED X Mini, with separate
  narrow-lens entries; Luxonis OAK-D, OAK-D Pro, OAK-D SR, OAK-D Lite; Orbbec Femto Bolt, Gemini 2, Gemini 335,
  Gemini 336), 9 cameras (e-con Systems, Leopard Imaging, Basler, Teledyne FLIR, OmniVision, ams OSRAM), 2 tactile
  sensors (GelSight Mini, DIGIT) and 8 hosts (USB 3.2 host ports, a GMSL2 deserialiser, Jetson modules and kits).
  A figure the source does not state is null ("—", "not stated by the maker"); assumed offsets are labelled.
  Custom devices (Sensors ▸ New custom device…) in the user's own library folder.
- Lens models (`core/optics`, ported from Vantage `core/lens.py`): pinhole, Brown-Conrady (5 and 8 terms),
  Kannala-Brandt, equidistant, equisolid, stereographic, each with its valid angle; FOV fitting; pixel density,
  viewing angle and thin-lens defocus.
- Articulated occlusion (`core/occlusion.py`, cube maps around each imager, GPU through EGL or an exact numpy
  rasteriser), the scene at q (`core/scenegeom.py`), exact ray casting (`core/raycast.py`), sparse voxels with
  adaptive refinement (`core/voxels.py`).
- Studies: coverage and resolution (`--coverage`) and depth validity (`--depth-validity`), each with its page.
- The Camera view (`ui/camview.py`, `core/camrender.py`): the selected sensor's image through its lens model, on a
  worker thread; frusta in the 3D view drawn from the same lens models.
- The Sensors and mounts page (device, mount on any robot link, pose in mm and degrees, stream, in use), the Library
  panel (double-click or drag a device into the view), the Inspector's device figures with their sources.
- Project obstacles (tables, task objects) in the robot's collision distances.
- The samples: the UR5e's Wrist D405 and Overhead D435 take their figures from the Intel entries. The SO-100's wrist
  camera moved from the back of the fixed jaw, where the jaw hid 39 % of the task volume from it, to a bracket on the
  wrist side looking past the jaw tips (11 % hidden).
- Validation: 22 new checks, 40 in all, 40 of 40 pass (optics against OpenCV and MuJoCo-rendered fiducials;
  occlusion against exact ray casting and MuJoCo segmentation; depth against Monte Carlo, Intel's tables and ray
  casting from both imagers; project-obstacle distances).
- `--self-test`: "library" (every entry validates; the UR5e sample's sensors resolve) and "occlusion" (a cube map
  against exact ray casting, naming the backend).
- `scripts/screens.py` adds the Phase 2 pages, the Camera view, the Library panel and the custom device dialog;
  `scripts/measure_ui.py` measures the UI-thread budget ([docs/models/performance.md](docs/models/performance.md)).
- Docs: `docs/models/library.md`, `optics.md`, `occlusion.md`, `voxels.md`, `coverage.md`, `depth.md`,
  `camview.md`, `performance.md`; `docs/guide/sensors.md`, `coverage.md`, `depth.md`; the user guide's Sensors and
  coverage chapter with the SO-100 and UR5e tours.

### Changed (Phase 2)
- The 3D view's frame after a joint change: 385 ms → 68 ms on the UR5e (shader renderer, integrated GPU). Smooth
  normals are computed once per robot model in link frames and rotated at each q; `RobotModel.posed_triangles` poses
  each link's triangles with one matrix product (35 ms → 12 ms, identical numbers).
- `Entry.title` says a word the maker's and the device's names share once ("Intel RealSense Depth Camera D405").
- A camera entry that states only a diagonal FOV gets a square-pixel pinhole fitted to it (labelled), not the default.
- The Camera view's legend sits on a translucent plate; the Library list elides names in the middle so model numbers
  stay visible.

### Phase 1: robots and kinematics
- Robot import (`core/robot/importer.py`): URDF and MJCF with every mesh, include and texture they name, stored in the
  project byte for byte with a reference manifest, so a robot compiles again from memory on any machine. Limits,
  velocities, efforts and the home pose are the file's own; a figure the file does not state is "—". Millimetre
  meshes are caught with the fix in the message. xacro must be expanded first; USD is not read yet.
- The MuJoCo kinematics model (`core/robot/model.py`): FK, world Jacobians, visual and collision meshes posed at q,
  a shared thread-safe model cache.
- An independent product-of-exponentials FK and Jacobian (`core/robot/kinematics.py`) with its own URDF/MJCF reader.
- Inverse kinematics (`core/robot/ik.py`): damped least squares with error-dependent damping, joint-limit avoidance,
  seeds, solution rules (nearest home, most manipulable, nearest the current pose), position-only mode for arms with
  fewer than six joints, plain failure reasons. About 4 ms per target on the samples (single thread).
- Manipulability (Yoshikawa, condition number, distance to limits) and collision distances (MuJoCo signed distance,
  self and static environment) in `core/robot/manipulability.py` and `collision.py`.
- The import wizard (File, Meshes, Scale and units, Joints, Gripper and pads, Summary) with a live preview; the
  joint, gripper-frame, fingertip and contact-pad taggers.
- The Robot page: joint sliders in degrees or millimetres, the gripper frame's pose, manipulability, condition
  number and self-collision with a status word (clear, near a limit, singular, in collision).
- IK drag of the end effector with live feasibility (green "reachable", or red with the solver's reason).
- Trajectory import from CSV and the pose scrubber (scrub, play at real time, step).
- Link-mounted sensors posed at q; the 3D view draws the robots and frames them.
- MuJoCo Menagerie import with each model's own licence shown and accepted before import, and recorded.
- Bundled after the licence check: Menagerie `trs_so_arm100` (Apache-2.0), `universal_robots_ur5e` (BSD-3-Clause,
  meshes stored as binary STL) and `robotiq_2f85` (BSD-2-Clause). Samples `--sample so100` and `--sample ur5e`, in
  the window and in batch mode.
- Validation: 10 kinematics checks (FK against the independent chain on the samples and five written robots; UR5e and
  Panda DH closed forms; Jacobians against finite differences and `mj_jac`; IK round trips; 2R manipulability; sphere
  and capsule distances). 18 of 18 pass. With no project given, the suite checks the bundled samples.
- `--self-test`: a "robots" line (both samples import; MuJoCo and the product-of-exponentials FK agree).
- Docs: `docs/models/robots.md`, `kinematics.md`, `ik.md`; the user guide's Robots chapter with the SO-100 and UR5e
  tours.
- 468 tests.

### Changed
- The project summary's `joints` counts the joints in q (revolute, continuous, prismatic), not fixed joints.

### Phase 0: foundation
- Ported from Vantage: the design tokens, QSS and IBM Plex fonts (unchanged), widgets, charts, colour maps, icons,
  the GLSL renderer, the 3D viewport and its navigation, the dialog frame, the command palette, provenance, the
  validation registry and report, the working-folder and environment helpers. Each module's docstring names its origin.
- Robot icons in the same stroke style: robot arm, gripper, depth sensor, tactile pad, task volume, grasp, joint,
  teleop. A new app icon.
- The viewport, renderer and scale bar sized for a robot cell (near plane 5 mm, zoom to 3 cm, millimetre scale bar).
- `core/frames.py`: REP-103 frames, quaternions (x, y, z, w), URDF rpy, MuJoCo's (w, x, y, z), poses.
- `core/project.py`: the `.dexv` project (zip: project.json, meshes.npz, sources/, thumb.png) with robots, sensors and
  mounts, the workspace, tasks, variants, requirements, analysis settings and the design log; JSON Schema in
  `docs/project.schema.json`; version checks and the migration hook; clear messages for files it cannot read.
- `core/jobs.py`: studies with progress, cancel and partial results, cached by content digest; the project summary
  study (dense voxel counts and the §7.1 feasibility limit).
- The main window: Outliner, Library, Inspector, Camera view, Assistant column, Analysis dock, pose scrubber; 13 menus;
  six saved and resettable workspaces; undoable edits (add a camera on a tripod, a task volume, a box object; move,
  resize, delete); save, open, recent files; a high-resolution render; summary export with provenance; Help ▸
  Coordinate frames, Keyboard shortcuts, Validate the models, About.
- Batch mode: `--analyse`, `--validate`, `--reproduce`; `--self-test` (9 checks).
- Validation: 8 checks (frames against scipy and MuJoCo, the project file against its schema and a round trip).
- 332 tests.

### Fixed
- Number fields showed no text under Qt 6.12 (a style-sheeted spin box without buttons got a 1 px line edit); the
  widget now gives the line edit its width back.
