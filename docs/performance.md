# Performance

Measured by `scripts/benchmarks.py` on one machine; every figure below comes from the JSON files it wrote to `out/bench/` and this page is made from them (`python scripts/benchmarks.py --doc-only`). Spec §16 asks for two budgets: no UI-thread work over 100 ms, and a 2,000-grasp study that finishes in a measured, documented time on a mid-range GPU. Only the GPU below has been measured, and this page does not class it as mid-range or otherwise: the figures are one machine's, a slower GPU or CPU scales them, and the machine was not idle (the load average is given beside each part). The UI timings of the earlier phases, and what was changed to meet the budget then, are in [models/performance.md](models/performance.md).

Measured on 2026-11-07 to 2026-11-08 (UTC).

## The machine

| | |
|---|---|
| CPU | AMD Ryzen 9 9900X 12-Core Processor (12 cores, 24 threads) |
| GPU (the occlusion and render backend) | NVIDIA GeForce RTX 5080/PCIe/SSE2 through EGL device 0 |
| NVIDIA driver, memory | NVIDIA GeForce RTX 5080: driver 595.99.02, 16303 MiB |
| RAM | 32.1 GB |
| OS | Ubuntu 26.04.1 LTS, Linux 7.0.0-42-generic, x86_64 |
| Python | 3.14.4 (CPython) |
| Packages | numpy 2.5.4, scipy 1.18.1, mujoco 3.17.0, opencv 5.0.0, PyOpenGL 3.1.10, Qt 6.12.0, PyQt6 6.12.0, dexvantage 0.1.0rc1 |
| BLAS | scipy-openblas 0.3.34.106.0 |
| Source | commit a668e4900e with local changes |

## Budgets

| Budget | Where it comes from | Measured | Result |
|---|---|---|---|
| 2,000-grasp study, so100: at most 180 s | spec §16 (the time is this project's default; why below) | 55.1 s | pass |
| 2,000-grasp study, ur5e: at most 180 s | spec §16 (the time is this project's default; why below) | 114.2 s | pass |
| 2,000-grasp study, dexhand: at most 180 s | spec §16 (the time is this project's default; why below) | 68.8 s | pass |
| 2,000-grasp study, aloha: at most 180 s | spec §16 (the time is this project's default; why below) | 182.1 s | FAIL |
| 2,000-grasp study, panda-bimanual: at most 180 s | spec §16 (the time is this project's default; why below) | 79.9 s | pass |
| 2,000-grasp study, g1: at most 180 s | spec §16 (the time is this project's default; why below) | 45.4 s | pass |
| No UI-thread step over 100 ms, so100 | spec §16 | slowest 79.6 ms (a stall, motion_blur: the study running and its result arriving (outside DexVantage's Python code (Qt's own work: layout, paint or delivering queued signals))); 0 over | pass |
| No UI-thread step over 100 ms, ur5e | spec §16 | slowest 37.1 ms (coverage: events while the study runs and its result arrives (watchdog stall)); 0 over | pass |
| No UI-thread step over 100 ms, dexhand | spec §16 | slowest 71.6 ms (a stall, coverage: the study running and its result arriving (at dexvantage/core/project.py:121 to_dict < dexvantage/ui/pages/coverage.py:253 inputs_key < dexvantage/ui/pages/coverage.py:346 show_result)); 0 over | pass |
| No UI-thread step over 100 ms, aloha | spec §16 | slowest 33.5 ms (grasp_visibility: events while the study runs and its result arrives (watchdog stall)); 0 over | pass |
| No UI-thread step over 100 ms, panda-bimanual | spec §16 | slowest 55.9 ms (coverage: events while the study runs and its result arrives (watchdog stall)); 0 over | pass |
| No UI-thread step over 100 ms, g1 | spec §16 | slowest 61.9 ms (grasp: the result arrives (show_result)); 0 over | pass |

The grasp budget of 180 s is a default chosen for this project, not a figure from the specification (which asks for a measured, documented time): spec §18 asks that a researcher sees which grasps lose fingertip visibility, moves a camera and sees the score change within ten minutes, which runs the grasp study at least twice beside the rest of that tour; three minutes a run leaves four for the rest.

## A 2,000-grasp study on each sample

2 000 grasps sampled on the primary arm of each sample (`core/grasps.sample`, top-down, side and angled in turn, seed 0), then the grasp self-occlusion study with its default settings: IK for every grasp, the pre-contact path, cube maps per camera and joint state, "what to move" (sensitivity, 12 one-at-a-time moves per camera on up to 250 grasps) and the gallery. Each sample in a fresh process; the peak is the process's largest resident set.

| Sample | Arm | Cameras | Feasible grasps | Study (s) | Grasps/s | IK (s) | Visibility (s) | What to move (s) | Cube maps | Maps/s | Peak memory (MB) | Load before → after |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| so100 | SO-ARM100 | 2 | 1 194 of 2 000 | 55.1 | 36.3 | 13.1 | 27.9 | 11.9 | 9 388 | 308 | 1 074 | 5.8 → 5.5 |
| ur5e | UR5e + Robotiq 2F-85 | 2 | 1 322 of 2 000 | 114.2 | 17.5 | 13.1 | 71.9 | 26.8 | 10 808 | 174 | 1 147 | 5.5 → 5.1 |
| dexhand | UR5e + Shadow Hand | 3 | 113 of 2 000 | 68.8 | 29.1 | 45.8 | 7.9 | 13.1 | 2 159 | 168 | 1 377 | 5.1 → 4.7 |
| aloha | ALOHA 2 left arm | 4 | 1 135 of 2 000 | 182.1 | 11.0 | 12.8 | 116.8 | 48.6 | 17 530 | 172 | 1 585 | 4.7 → 8.2 |
| panda-bimanual | Panda left (sample cell) | 3 | 1 004 of 2 000 | 79.9 | 25.0 | 21.7 | 38.3 | 17.0 | 13 527 | 324 | 1 168 | 8.2 → 5.7 |
| g1 | Unitree G1 upper body | 1 | 32 of 2 000 | 45.4 | 44.0 | 42.2 | 1.0 | 1.3 | 293 | 180 | 1 200 | 5.7 → 8.5 |

Cube maps rendered by NVIDIA GeForce RTX 5080/PCIe/SSE2 at 512 texels a face. Grasps/s is the study's own figure (grasps over the whole run).

### The change made to the grasp study

The first measurement missed the budget on the ALOHA 2 sample, and a profile showed a share of the time in the grasp study's own culling: every cube map tested each triangle of a link against the cone around the points (two links of 94 730 triangles each), and each "what to move" step recomputed the moved camera's seat mask once per grasp. The study now tests runs of 256 consecutive triangles first and only the triangles of the runs that meet the cone (the triangles kept are exactly the per-triangle test's: `tests/test_grasp_cull.py`), and keeps the seat masks of the last 16 poses (`core/studies/grasp_visibility.py`). Measured with the same 2,000-grasp study (seed 0, default settings) run with the grasp module of the last commit and then with the working tree, back to back in fresh processes (aloha, 2026-11-08):

| | Study (s) | Visibility (s) | What to move (s) | Cube maps | Load before → after |
|---|---|---|---|---|---|
| Before | 213.2 | 126.3 | 70.2 | 17 530 | 5.1 → 7.8 |
| After | 181.0 | 116.5 | 47.8 | 17 530 | 7.8 → 7.7 |

The study's result (its summary and the maps drawn) is the same before and after.

### Where the time goes (aloha)

The slowest sample over the budget, run again under cProfile (`--only profile`): 196.9 s profiled (the profiler adds its own overhead; 195.8 s for the study in that run). The DexVantage functions with the most time in all:

| Function | Calls | Time in all (s) | Own time (s) |
|---|---|---|---|
| `dexvantage/core/jobs.py:155 run_study` | 1 | 195.8 | 0.0 |
| `dexvantage/core/studies/grasp_visibility.py:1086 run` | 1 | 195.8 | 0.3 |
| `dexvantage/core/studies/grasp_visibility.py:667 states` | 19 600 | 171.7 | 0.8 |
| `dexvantage/core/studies/grasp_visibility.py:642 cube` | 17 530 | 165.5 | 0.1 |
| `dexvantage/core/studies/grasp_visibility.py:626 build` | 17 530 | 164.9 | 0.2 |
| `dexvantage/core/occlusion.py:658 build` | 17 530 | 105.8 | 7.7 |
| `dexvantage/core/occlusion.py:608 render` | 17 530 | 71.2 | 0.1 |
| `dexvantage/core/occlusion.py:420 render` | 17 530 | 71.1 | 27.0 |
| `dexvantage/core/studies/grasp_visibility.py:582 occluders` | 17 530 | 57.8 | 21.7 |
| `dexvantage/core/studies/grasp_visibility.py:1421 _sensitivity` | 1 | 52.0 | 0.1 |
| `dexvantage/core/studies/grasp_visibility.py:434 cone_mask` | 67 276 | 34.8 | 20.4 |
| `dexvantage/core/robot/ik.py:410 solve_batch` | 11 | 18.9 | 0.0 |

Of the 196.9 s profiled: drawing the cube maps (`core/occlusion.py` `CubeMap.build`) 105.8 s (54 %); posing and culling the triangles for each map (`Evaluator.occluders`) 57.8 s (29 %); the IK (`ik.solve_batch`) 18.9 s (10 %). What is left in the grasp study's own code is posing each kept triangle on the CPU for every map; the larger share is the shared occlusion code (`core/occlusion.py`, not changed here): its GL backend's per-map work in Python, the read-back of the six faces and the far-plane distance over every vertex (the own times below). Drawing the posed links on the GPU from their link transforms, as `docs/models/occlusion.md` proposes, would remove both the posing and most of the upload.

The functions with the most time of their own (any library):

| Function | Calls | Own time (s) | Time in all (s) |
|---|---|---|---|
| `dexvantage/core/occlusion.py:420 render` | 17 530 | 27.0 | 71.1 |
| `~:0 <method 'reduce' of 'numpy.ufunc' objects>` | 467 589 | 22.4 | 22.4 |
| `dexvantage/core/studies/grasp_visibility.py:582 occluders` | 17 530 | 21.7 | 57.8 |
| `dexvantage/core/studies/grasp_visibility.py:434 cone_mask` | 67 276 | 20.4 | 34.8 |
| `.venv/lib/python3.14/site-packages/OpenGL/GL/images.py:340 glReadPixels` | 17 530 | 20.0 | 25.2 |
| `dexvantage/core/studies/grasp_visibility.py:381 _in_cone` | 187 142 | 9.6 | 14.3 |
| `dexvantage/core/occlusion.py:658 build` | 17 530 | 7.7 | 105.8 |
| `~:0 <method 'astype' of 'numpy.ndarray' objects>` | 436 541 | 7.1 | 7.1 |
| `.venv/lib/python3.14/site-packages/numpy/linalg/_linalg.py:2615 norm` | 251 122 | 6.3 | 27.5 |
| `.venv/lib/python3.14/site-packages/OpenGL/wrapper.py:782 wrapperCall` | 157 764 | 5.7 | 7.0 |

## Coverage and depth validity on every sample

Each study in a fresh process with the sample's own settings: the first run (it builds the robot model, the scene's occluders and the cube maps) and a repeat in the same process (the compiled model and cube maps then come from the process caches).

| Sample | Study | Cameras | First run (s) | Repeat (s) | Result | Peak memory (MB) |
|---|---|---|---|---|---|---|
| so100 | coverage | 2 | 0.33 | 0.075 | 94.5 % of the task volume seen | 401 |
| so100 | depth_validity | 2 | 0.06 | 0.000 | There is no depth sensor in this rig: add one and choose its device from the library. | 187 |
| ur5e | coverage | 2 | 1.04 | 0.591 | 91.9 % of the task volume seen | 699 |
| ur5e | depth_validity | 2 | 0.92 | 0.541 | 2 depth sensors | 730 |
| dexhand | coverage | 3 | 0.74 | 0.257 | 98.9 % of the task volume seen | 615 |
| dexhand | depth_validity | 3 | 0.72 | 0.064 | 1 depth sensor | 583 |
| aloha | coverage | 4 | 1.14 | 0.774 | 100.0 % of the task volume seen | 913 |
| aloha | depth_validity | 4 | 1.47 | 0.815 | 4 depth sensors | 939 |
| panda-bimanual | coverage | 3 | 1.77 | 1.330 | 100.0 % of the task volume seen | 682 |
| panda-bimanual | depth_validity | 3 | 1.47 | 1.111 | 3 depth sensors | 716 |
| g1 | coverage | 1 | 0.46 | 0.195 | 96.6 % of the task volume seen | 718 |
| g1 | depth_validity | 1 | 0.49 | 0.191 | 1 depth sensor | 736 |

## Occlusion cube-map build

One cube map (six faces) of the scene seen from each camera at the home pose, `core/occlusion.CubeMap.build`; the median of 10 builds (GPU) or 3 (NumPy) after a first build that uploads the geometry. GPU: NVIDIA GeForce RTX 5080/PCIe/SSE2.

| Sample | Camera | Triangles | gl 256 (ms) | gl 512 (ms) | numpy 256 (ms) | numpy 512 (ms) | GPU first build, 512 (ms) |
|---|---|---|---|---|---|---|---|
| so100 | Wrist camera (sample placement) | 60 684 | 5.0 | 13.9 | 357.1 | 1 129.8 | 25.2 |
| so100 | Third-person camera (sample placement) | 60 684 | 5.2 | 8.1 | 80.6 | 140.9 | 9.2 |
| ur5e | Wrist D405 | 193 115 | 14.2 | 15.8 | 623.6 | 985.3 | 20.4 |
| ur5e | Overhead D435 | 193 115 | 11.9 | 14.4 | 228.5 | 276.2 | 16.6 |
| dexhand | Palm micro camera (sample placement) | 178 791 | 10.1 | 13.6 | 452.1 | 1 648.0 | 15.2 |
| dexhand | Dorsal wrist camera (sample placement) | 179 521 | 14.0 | 17.2 | 543.2 | 1 504.2 | 19.9 |
| dexhand | Head camera (sample placement) | 179 521 | 16.2 | 23.8 | 303.0 | 509.8 | 22.9 |
| aloha | cam_high (overhead) | 331 394 | 23.5 | 26.6 | 517.8 | 664.8 | 28.2 |
| aloha | cam_low (worms-eye) | 331 394 | 28.8 | 30.4 | 771.9 | 1 166.4 | 33.6 |
| aloha | cam_left_wrist | 329 600 | 32.8 | 37.5 | 2 945.7 | 10 137.5 | 44.2 |
| aloha | cam_right_wrist | 329 600 | 34.4 | 29.5 | 2 539.9 | 7 792.2 | 31.2 |
| panda-bimanual | Wrist D405, left (sample placement) | 269 778 | 17.0 | 20.3 | 710.0 | 1 677.8 | 18.8 |
| panda-bimanual | Wrist D405, right (sample placement) | 269 778 | 22.8 | 24.3 | 794.7 | 1 714.2 | 26.7 |
| panda-bimanual | Front D435 (sample placement) | 269 778 | 21.5 | 25.7 | 414.8 | 383.5 | 31.3 |
| g1 | Head D435 (Unitree URDF d435_link) | 216 312 | 17.1 | 23.1 | 533.9 | 1 510.5 | 23.8 |

## UI-thread events

Each sample opened in a fresh process (offscreen), then every study its pages offer run from its page on the worker thread, as a user starts it. Timed on the UI thread: showing each page, the result arriving (show_result) and the page's 3D overlays (median of 3, the largest in the table); an event-loop watchdog (a 5 ms timer, `scripts/measure_ui_phase3.py`) records the longest stall while the study runs and its result arrives, so a slow handler no step names still shows. Offscreen the 3D view falls back to QPainter; its paint is left out, as in the earlier UI timings (a desktop draws it with the shader renderer).

| Sample | Studies run | Slowest UI-thread step | Its time (ms) | Over budget | Building the window (ms) | Show to first paint (ms) |
|---|---|---|---|---|---|---|
| so100 | 18 | a stall, motion_blur: the study running and its result arriving (outside DexVantage's Python code (Qt's own work: layout, paint or delivering queued signals)) | 79.6 | 0 | 1 577 | 506 |
| ur5e | 18 | coverage: events while the study runs and its result arrives (watchdog stall) | 37.1 | 0 | 1 561 | 1 218 |
| dexhand | 18 | a stall, coverage: the study running and its result arriving (at dexvantage/core/project.py:121 to_dict < dexvantage/ui/pages/coverage.py:253 inputs_key < dexvantage/ui/pages/coverage.py:346 show_result) | 71.6 | 0 | 1 688 | 1 280 |
| aloha | 18 | grasp_visibility: events while the study runs and its result arrives (watchdog stall) | 33.5 | 0 | 2 001 | 1 865 |
| panda-bimanual | 18 | coverage: events while the study runs and its result arrives (watchdog stall) | 55.9 | 0 | 1 527 | 1 576 |
| g1 | 18 | grasp: the result arrives (show_result) | 61.9 | 0 | 1 757 | 1 188 |

Building the window and its first paint are start-up (the window is not yet on screen); they are listed here and in the start-up table below, not counted against the 100 ms budget, as in the earlier UI timings. "Over budget" counts the steps and the stalls above 100 ms after the first paint.

The stalls of 50 ms or more on each sample, with where the UI thread was (a thread samples its Python stack while a 5 ms timer is late; the innermost DexVantage frames first):

- **so100**: 80 ms motion_blur: the study running and its result arriving — `outside DexVantage's Python code (Qt's own work: layout, paint or delivering queued signals)`; 52 ms lighting: after showing the page — `outside DexVantage's Python code (Qt's own work: layout, paint or delivering queued signals)`
- **ur5e**: none
- **dexhand**: 72 ms coverage: the study running and its result arriving — `dexvantage/core/project.py:121 to_dict < dexvantage/ui/pages/coverage.py:253 inputs_key < dexvantage/ui/pages/coverage.py:346 show_result` (cycle collector 27 ms); 56 ms dex_occlusion: the study running and its result arriving — `outside DexVantage's Python code (Qt's own work: layout, paint or delivering queued signals)`
- **aloha**: none
- **panda-bimanual**: 56 ms coverage: the study running and its result arriving — `dexvantage/core/studies/coverage.py:178 decode_array < dexvantage/core/studies/coverage.py:202 decode < dexvantage/ui/pages/coverage.py:345 show_result`
- **g1**: none

The five slowest timed UI-thread steps on each sample:

- **so100**: motion_blur: events while the study runs and its result arrives (watchdog stall) 78.8 ms; lighting: events after showing (watchdog stall) 51.8 ms; action_chunk: show the page 36.6 ms; grasp_visibility: events while the study runs and its result arrives (watchdog stall) 29.7 ms; canonical_view: show the page 27.8 ms
- **ur5e**: coverage: events while the study runs and its result arrives (watchdog stall) 37.1 ms; cables: events while the study runs and its result arrives (watchdog stall) 30.7 ms; grasp_visibility: events while the study runs and its result arrives (watchdog stall) 27.7 ms; coverage: the result arrives (show_result) 26.9 ms; depth_validity: events while the study runs and its result arrives (watchdog stall) 23.0 ms
- **dexhand**: coverage: events while the study runs and its result arrives (watchdog stall) 71.5 ms; action_chunk: events after showing (watchdog stall) 70.6 ms; dex_occlusion: events while the study runs and its result arrives (watchdog stall) 56.4 ms; requirements: 3D overlays for one frame 50.9 ms; dexhand: 3D overlays for one frame 48.7 ms
- **aloha**: grasp_visibility: events while the study runs and its result arrives (watchdog stall) 33.5 ms; grasp: the result arrives (show_result) 33.0 ms; coverage: events while the study runs and its result arrives (watchdog stall) 32.1 ms; coverage: the result arrives (show_result) 28.2 ms; time_sync: events while the study runs and its result arrives (watchdog stall) 23.7 ms
- **panda-bimanual**: coverage: events while the study runs and its result arrives (watchdog stall) 55.9 ms; bandwidth: show the page 55.7 ms; coverage: the result arrives (show_result) 42.7 ms; grasp: the result arrives (show_result) 30.7 ms; grasp_visibility: events while the study runs and its result arrives (watchdog stall) 28.9 ms
- **g1**: grasp: the result arrives (show_result) 61.9 ms; handeye_plan: events while the study runs and its result arrives (watchdog stall) 23.0 ms; grasp_visibility: events while the study runs and its result arrives (watchdog stall) 18.2 ms; coverage: events while the study runs and its result arrives (watchdog stall) 17.4 ms; action_chunk: show the page 13.8 ms

Studies that refused on a sample (their refusal is the result the page shows; the events were still timed): so100 action_chunk_sync ("StudyError: SmolVLA (sample): its inference time is not stated by its source (SmolVLA base), so its loop cannot be timed: enter a measured figure on the Policies page."); dexhand action_chunk_sync ("StudyError: There is no policy in the project: add the policy the data is for on the Policies page (ACT, Diffusion Policy, π0…)."); dexhand grasp_visibility ("StudyError: There is no grasp set to study: make one on the Tasks page (sample the task volume, import a file or take a trajectory)."); dexhand vla_tokens ("StudyError: There is no policy in the project: add one on the Policies page (its input resolution, crop rule and ViT patch come from its model card)."); g1 action_chunk_sync ("StudyError: GR00T N1.5 (sample): its control rate is not stated by its source (GR00T N1.5 3B), so its loop cannot be timed: enter a measured figure on the Policies page."); g1 canonical_view ("StudyError: The canonical view warp needs a second camera (the canonical view) besides the source: add one, or load a canonical view file on the Canonical view page.").

Not run in this part: audit (needs a recorded dataset); campaign (needs a campaign spec); cells (needs a second cell); experiments (needs factors and responses chosen on its page (timed in Phase 9 with a Saltelli design)); groundtruth (renders every camera with MuJoCo (timed by the validation run below)); handeye_solve (needs captured board images); optimise (runs the coverage and grasp studies for every design (minutes; its page was timed in Phase 9)); design_agent (runs the optimiser in rounds (minutes)); plan_vs_recording (needs a recorded dataset); variants_compare (needs a second variant); silhouette_calib (needs a recorded dataset); sim2real (needs a recorded dataset); stats (needs success counts); tolerance (runs the studies once per Monte Carlo draw (minutes; its page was timed in Phase 9)); tracking (needs a run store).

The longest worker runs (the study on its thread, the UI responsive meanwhile): panda-bimanual teleop 136.1 s; g1 teleop 42.3 s; aloha teleop 40.6 s; aloha grasp_visibility 30.9 s; ur5e grasp_visibility 24.6 s; aloha dex_occlusion 11.8 s.

## Start-up to first paint

`dexvantage.app.main` started in a fresh process as `dexvantage [--sample NAME]` (offscreen, throw-away settings), timed from the moment the parent starts the process to the first paint event of the main window; 3 runs each.

| Project | First paint, median (s) | Largest (s) | Interpreter up (s) | Application made (s) | Peak memory (MB) |
|---|---|---|---|---|---|
| blank | 1.50 | 1.54 | 0.03 | 0.17 | 281 |
| so100 | 1.72 | 1.89 | 0.03 | 0.33 | 520 |
| ur5e | 1.97 | 1.98 | 0.03 | 0.53 | 858 |
| dexhand | 2.14 | 2.21 | 0.03 | 0.44 | 722 |
| aloha | 2.37 | 2.46 | 0.04 | 0.96 | 976 |
| panda-bimanual | 1.86 | 1.97 | 0.03 | 0.56 | 771 |
| g1 | 2.08 | 2.10 | 0.03 | 0.74 | 827 |

## Import time

`import dexvantage as dv` in a fresh interpreter from the source tree (no bundle), 7 runs, medians. The package resolves its API names on first use, so the import itself loads almost nothing; the first API name loads `dexvantage.api`, and a first use opens a sample as a Rig.

| Step | Time (ms) | Modules loaded after it |
|---|---|---|
| `import dexvantage as dv` | 0.06 | 83 |
| the first API name (`dv.Rig`) | 72.0 | 213 |
| a first use (`dv.sample('so100')`) | 153.4 | 316 |
| the whole process (an empty interpreter takes 12.4) | 261.1 | |

Qt is imported by the import: no; after the first use: Qt no, MuJoCo yes, OpenCV no, SciPy no.

The slowest modules `dexvantage.api` imports directly (`python -X importtime`, cumulative): dexvantage.core.jobs 70.4 ms, dexvantage.api_research 2.3 ms, typing 1.3 ms, html 0.6 ms, json 0.6 ms, dexvantage.core 0.1 ms, dexvantage.api_cluster 0.1 ms, dexvantage 0.0 ms.

## Splat views: fit time, render rate and GPU memory

The splat views study (`core/sim/splat_views.py`) with its default settings on a MuJoCo render of each sample (a synthetic demonstration capture), in a fresh process: the fit (1,500 steps at most, a 120 s budget, a 20,000-Gaussian budget, 48 views at 160 × 120 px, defaults), then 40 renders of the fitted rig through the target camera (the wrist camera) at its stream size at 40 joint states, after 3 warm-up renders. GPU: NVIDIA GeForce RTX 5080, PyTorch 2.14.1+cu130 (CUDA 13.0), the plain-torch rasteriser (the study fits and renders through it whether or not gsplat is installed: [models/splat.md](models/splat.md)). GPU memory is what PyTorch allocated and reserved at its peak (`torch.cuda.max_memory_allocated` and `max_memory_reserved`), not what the driver reports for the process.

| Sample | Gaussians | Fit (s) | ms per step | Study (s) | Fit GPU peak, allocated / reserved (MB) | Render size | Render fps | ms per frame | Render GPU peak, allocated / reserved (MB) | Process peak (MB) | Load before → after |
|---|---|---|---|---|---|---|---|---|---|---|---|
| so100 | 16 939 | 60.0 | 40.0 | 64.3 | 807 / 1 642 | 1280 × 720 | 26.6 | 37.5 | 748 / 1 776 | 2 683 | 3.8 → 3.2 |
| ur5e | 18 670 | 62.2 | 41.5 | 66.8 | 807 / 1 562 | 1280 × 720 | 23.7 | 42.1 | 749 / 1 697 | 2 910 | 3.2 → 3.0 |

The study's time includes the MuJoCo capture, the held-out scoring and the before/after preview besides the fit. A render includes posing the Gaussians by forward kinematics and copying the image back to the CPU; the replay and the augmentation also encode video.

## A full validation run

`dexvantage --validate out/validation` in a fresh process: 428 s for the process, 425 s in the suite, 239 of 239 checks passed over 38 models, peak memory 5 622 MB (load 5.0 → 7.6). The checks and their measured agreement are in [validation.md](validation.md).

## How to reproduce

```
cd <repository>
systemd-run --user --scope -q -p MemoryMax=7G env QT_QPA_PLATFORM=offscreen \
    .venv/bin/python scripts/benchmarks.py --egl 'EGL device 0'
.venv/bin/python scripts/gen_validation_doc.py out/validation/validation.json   # docs/validation.md
```

`--only grasp,ui` runs some parts, `--samples ur5e,aloha` some samples, `--doc-only` remakes this page from `out/bench/*.json`. `--egl` picks the EGL device or renderer of the GPU cube maps (DEXVANTAGE_EGL); each child runs offscreen with throw-away settings folders.

## Known limits

- One machine, one run of each figure (start-up and import take the median of a few); the machine was shared with other work, so the times are upper bounds of what it does idle.
- Peak memory is the child process's largest resident set (`ru_maxrss`); a validation check's own child processes (cluster shards, MuJoCo renders in a separate process) are not added to it.
- The UI timings are offscreen with the QPainter 3D fallback's paint left out; a desktop session adds the shader renderer's frames, timed in Phase 2 ([models/performance.md](models/performance.md)): 7.8 ms (SO-100) and 11.9 ms (UR5e) median for a steady frame, and for a frame after a joint change 26.5 ms and 67.6 ms median, 95.5 ms worst (UR5e, integrated GPU). The QPainter fallback itself (no OpenGL, or `--software-3d`) takes 160 ms and 586 ms a frame there, over the budget.
- The 2,000-grasp set is sampled, not a task's own grasps; a set with more infeasible grasps is faster.

