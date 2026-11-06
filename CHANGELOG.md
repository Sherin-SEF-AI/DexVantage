# Changelog

All notable changes to DexVantage. The format follows Keep a Changelog; versions follow semantic versioning.

## [Unreleased]

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
