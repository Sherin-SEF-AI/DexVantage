# Changelog

All notable changes to DexVantage. The format follows Keep a Changelog; versions follow semantic versioning.

## [Unreleased]

### Research tools part 2 and fixes
- The experiment tracker (`core/tracking/`, Analysis ▸ Runs, `--track DIR`, `--runs`, `--runs-export`, `dv.runs`,
  `dv.track`, `dv.run_store`, the assistant's `get_runs`): every study run, designed experiment, optimiser run,
  tolerance Monte Carlo, scorecard, statistics, campaign and cell comparison logged as a run (settings, seeds,
  figures with units and status, result and provenance) in a run store beside the project, safe for several writers;
  compare (with a `core/stats` significance note paired by seed), re-run and export as CSV, Parquet, an MLflow 3.17
  file store and a W&B replay script (optional extra `tracking`). Opt-in; the setting is left out of the content hash.
- Statistics and power analysis (`core/stats.py`, `core/statcompare.py`, Analysis ▸ Statistics, `--stats SPEC.json`,
  `--stats-saved`, `dv.stats`, `get_statistics`): Wilson and Clopper-Pearson intervals, BCa bootstrap, exact and Monte
  Carlo permutation tests, t and rank tests, effect sizes with intervals, Fisher's exact and z tests, multiplicity
  corrections, power and sample size, each result naming its test and why.
- The data-collection campaign planner (`core/campaign.py`, Data ▸ Campaign, `--campaign SPEC.json`,
  `--campaign-saved`, the `campaign` exporter, `dv.campaign`, `get_campaign`): conditions, demonstrations from a power
  analysis (through `core/stats`), operators counterbalanced by Williams squares, sessions and days, storage, budget,
  CSV, iCalendar and HTML checklists.
- The sim-to-real gap (`core/sim2real/`, Data ▸ Sim-to-real gap, `--sim2real OUT.json --dataset PATH`,
  `rig.sim2real`, `rig.gap_dataset`, `get_sim2real`): a recorded episode rendered again through the rig, per-camera
  photometric, geometric and distribution measures, Shapley-ranked causes and the change that closes each.
- The publication kit (`core/publication.py`, Export ▸ Paper kit…, `--export … --formats paper_kit`,
  `dv.publication`, `get_methods_paragraph`): a methods paragraph, booktabs tables (now also the statistical
  comparisons with their intervals, the cell comparison and the tracked runs when present), SVG and PDF figures,
  BibTeX and a reproducibility appendix, every number traced and checked.
- Cluster sweeps (`core/cluster/`, `--jobs N`, `--slurm FOLDER`, `--run-shard`, `--gather`, `--cluster-studies`,
  `--optimise-continue`, Run on… on the Designed experiments and Tolerances pages, `dv.cluster`,
  `get_cluster_job`): a spawn process pool and Slurm job arrays with a shared result cache; the gathered result is the
  in-process one.
- The multi-cell comparison (`core/cells.py`, `core/cells_pool.py`, File ▸ Compare cells…, `--cells`,
  `--cells-compare`, the `cells` exporter, `dv.cells.compare`, `get_cells`): figures of merit side by side, a
  geometric gap index (ported from Vantage `core/rig_gap.py`), dataset pooling against the LeRobot schema, drift.
- Robot import and export: the USD robot import (spec §13, `core/robot/usd_import.py`, the import wizard,
  `dv.robot_io`), a UsdPhysics writer, URDF inertials that MuJoCo loads as written, OpenVLA encodings from the rig,
  the Isaac off-centre principal point; the bundled newton-assets UR5e (BSD-3-Clause) as a third-party USD reference.
- Prices: 81 of the 191 built-in library entries carry a cited list price (`core/prices.py`, `dv.prices`,
  `dv.price`, `get_rig_cost`); `rig.cost` and the optimiser's cost objective are figures on the samples.
- UI: hidden analysis pages hold their signals until shown (`ui/pages/_gate.py`): an edit costs 3.6–5.5 ms instead
  of 28–39 ms on the two-arm samples; adding the G1 is 10–13 ms (171 ms before); a refused study logs one line;
  custom devices of every library kind; surfaces in the Outliner. Experiments censor figures stated only as bounds.
- Validation: models `tracking` 5, `stats` 13, `campaign` 6, `sim2real` 7, `publication` 4, `cluster` 4, `cells` 6,
  `robot_io` 5, two `library` price checks and `experiments.censored_best` (232 checks in all); self-test lines for
  each area.
- Integration: the tracker logs cluster runs (`--jobs`, a complete `--gather`, `dv.cluster`), `--stats`,
  `--campaign`, `--cells` and the steps of a `--run-plan`, and the Runs page logs while it is hidden (the page gate
  held its signals, so a result finished on another page was not logged until the Runs page was shown, and then only
  if no edit came between); the paper kit's statistics, cells and runs tables; the experiment best run's bound and
  settled flag in the tracker; `--jobs 0` was taken as no `--jobs` (now refused in one sentence); the occlusion
  cube-map cache keys its origin exactly (a map for an origin within 0.1 mm was reused, so a long-lived process
  differed from fresh shards in the last digits of a coverage figure); the report's "all studies" leaves out the
  cell comparison, the statistics and the sim-to-real gap (they run other studies many times or render); the
  assistant's `check_robot_file` tool (33 tools; the MCP server lists 39); page guides for the six new pages; a
  spawned cluster worker reported its parent's peak memory (Linux keeps `ru_maxrss` across exec; now `VmHWM`); runs
  started in the same millisecond keep the store's order; the LeRobot reader retries single-threaded the frames the
  threaded AV1 decoder silently drops; the paper kit's runs table shows each run's headline figure.
- Screens and UI: `scripts/screens.py --only research2`, `scripts/measure_ui_rt2.py` (0 UI-thread steps over 100 ms
  on ALOHA 2 and the G1); the run store's path on one line, the Statistics page's source panel sized to its content
  and its Preview shown in the calculator, no "-0" figures on the sim-to-real page, the paper kit's sections in
  columns, the import wizard's file line clear of Browse and a USD robot named after its stage, the campaign's unused
  demonstration fields disabled; the Tasks page builds the robot's model and pads on a worker when it offers Apply
  (a cold Apply took about 200 ms on the UI thread on ALOHA 2).

### Added (research tools part 2 reviews)
- `rig.add_variant(name, {factors}, from_variant=None)`: a second variant from Python (a copy of the active one,
  sensors keep their uids, the factors set on the copy only).
- The part 1 research tools from Python (`dexvantage/api_research.py`, no Qt import): `dv.diff`, `dv.merge`,
  `dv.bibtex`, `dv.figures`, `dv.figure`, `dv.export_runs`.
- A plugin's validation checks (`PluginAPI.check`; `validation/suite.py` `PLUGIN_MODELS`): each plugin model runs
  with the built-in checks, cites its references in the report and BibTeX, and goes when the plugin is unloaded or
  fails.
- The research workflow example `examples/03_research_workflow.py` (a jupytext percent-format notebook);
  `docs/models/research-api.md`, `docs/models/product-review.md` and their guides.

### Fixed (research tools part 2 reviews and final gate)
- Statistics: Fisher's exact test named its one-sided alternatives the other way round from the rest of
  `core/stats`; a tolerance run's samples dropped draws without a value, so seeded runs were paired off by one, and
  the tracker's significance test refused a list holding a missing draw; a zero difference read "+0 % [+0, +0]";
  a p-value read "1 p" in the Runs table and the paper kit; a run compared with itself was compared.
- Cluster: a short write in a shard's output lost two lines; the shared cache never replaced a corrupt entry;
  `dv.cluster.*(jobs=0)` and other non-counts ran serially (refused now, as `--jobs 0`); a pool started from a
  script's unguarded top-level code is refused in one sentence.
- Tracker: two deletes of one run at once raised a raw OSError; `dv.stats.*` on a rig, `dv.campaign.plan`,
  `dv.cells.compare` and `dv.cluster.gather` were not logged while their batch forms were; a logged designed
  experiment had no duration.
- Cells: pooling compared joint kinds by position after a reorder, took a unit not stated as another kind and
  ignored a target in degrees; the page added a missing file as a cell, and then the comparison refused to run.
- Optimiser and prices: the cost objective was kept when a device is priced in another currency; a price without a
  currency counted as the currency "None".
- Sim-to-real: a mount σ the frames do not constrain was NaN (the tracker refused the run, the result was not valid
  JSON); a brightness offset was read as a light gain, and Apply scaled the lights; `rig.gap_dataset` raised a raw
  ValueError on a bad gap and refused a folder that is not empty only after rendering every frame.
- Campaign: an unknown spec key was ignored silently (a mistyped power analysis gave the default 50
  demonstrations); "1 sessions" and the HTML schedule's plurals; a compression ratio above 1 (20 for 0.05) is
  named: the ratio is the encoded size over the raw size.
- USD import: pxr's C++ names and full temporary paths in the error; the reader's own words now.
- Statistics page: the Variants, Experiment runs and Tracker runs panels spread their controls over the height, and
  the figures field had no label.
- The headline of a study without a headline metric shows its unit and quantile.
- `scripts/screens.py` removes its temporary XDG, capture and export folders when it ends.

### Phases 9 and 10, samples, catalogue, library, research tools part 1
- Shared contracts (additive): named metrics (`core/metrics.py`, `@metric`, `evaluate`, `better_of`) with 53
  definitions over every study (`core/metric_defs.py`, including `canonical_view.parallax_p95` and
  `canonical_view.coverage`); settable project quantities (`core/factors.py`: `sensor:<id>.x|…|model|stream`,
  `setting:<study>.<key>`, `json:/pointer`; locked sensors refused); the exporter registry
  (`core/exporters/__init__.py`, `@exporter`, `export` with a manifest of SHA-256s and provenance, `unstamped` files).
- Requirements and the readiness scorecard (`core/requirements.py`, `--check REQUIREMENTS.yaml|defaults`, exit 0, 1,
  2; `--scorecard`, `--variants-compare`, `--rig-inventory`), the Requirements and variants page, `--reproduce` of any
  result (`core/reproduce.py`, `--reproduce-out`) and `--run-plan` (`core/run_plan.py`).
- Monte Carlo tolerances (`core/tolerance.py`, ported from Vantage; `--tolerance-mc`; the Tolerances page) from mount
  ± bounds or a hand-eye calibration's covariance, with requirement shares and drivers; the extrinsic perturbation
  export (`perturbations`, §8.5).
- Designed experiments (`core/experiments.py`, ported from Vantage; `--experiment DESIGN.json`, `--experiment-saved`;
  the Designed experiments page): full and fractional factorial, Latin hypercube, Sobol, Saltelli; Sobol indices with
  bootstrap intervals, main effects, the best run.
- The optimiser and the design agent (`core/optimise/`, NSGA-II in NumPy; `--optimise`, `--design-agent`,
  `--optimise-settings`; the Optimise page): Pareto fronts over metric objectives with requirements as constraints.
- Exporters: `mjcf`, `urdf` (with a tf2 launch file), `opencv_yaml` (and Kalibr), `masks`, `plucker`, `brackets`
  (STEP with the optional `step` extra), `fiftyone`, `lerobot_dataset`, `isaac` (USD prims; Isaac Lab v3.0.0 and
  v2.3.2 camera configs), `lerobot` (features, ACT and Diffusion Policy train configs with the delay offset and
  ensemble weight, OpenVLA), `report` (HTML and PDF), `perturbations`, `canonical_warp`; File ▸ Export… (takes over
  the last placeholder command; the report reuses the window's results) and `--export OUT --formats … --export-options`.
- The cross-embodiment canonical view warp (§7.16; `--canonical-view`; Export ▸ Canonical view) and the Rerun bridge
  (§9.4; `--rerun OUT.rrd`, Export ▸ Stream to Rerun…; optional extra `rerun`).
- The assistant's tool layer (30 tools) and the MCP server (`python -m dexvantage.mcp`, attaching to an open window);
  the Assistant dock (Window ▸ Assistant, Ctrl+J; Claude or Groq; keys in the system keyring only; a number check on
  every answer, proposals as one undo step, long studies only from an approved plan; optional extra `assistant`).
- Samples `--sample aloha|panda-bimanual|g1` (MuJoCo Menagerie, licences read and recorded); a robot catalogue of 53
  robots in the Library (kind `robot`); 44 more cameras, depth cameras and tactile sensors; 10 more hosts and
  capture cards, 11 policies, 3 lights and 3 cables; a new `teleop` kind with 13 devices, added from the Teleop page.
- Research tools part 1: the Python and notebook API (`import dexvantage as dv`), the plugin SDK
  (`dexvantage/plugin_sdk.py`, `--list-plugins`, `--no-plugins`, Help ▸ Plugins…, a template in
  `examples/plugin-template/`), project diff and three-way merge (`--diff`, `--diff-html`, `--merge`, File ▸ Compare
  with…), the Git folder form (`--unpack`, `--pack`, `--textconv`, File ▸ Save as folder (for Git)…), BibTeX
  (`--cite`) and publication figures (`--figure`, Export figure… on page charts).
- Validation: models `robots`, `library`, `tolerance`, `experiments`, `optimise`, `assistant`, `requirements`,
  `copilot`, `versioning`, `samples` and new `formats` and `vla` checks; self-test lines for each new area.
- Fixes at integration: the brackets study crashed when a sensor with a bracket type had a library device that states
  no size (now refused in one sentence); four tolerance checks declared no unit; the Export dialog did not pose the
  Isaac export at the on-screen joint state; File ▸ Open sample built the project on the UI thread (now a window
  job); the variants compare said "from before the last edit" right after a compare; the LeRobot reader decodes a
  packet the frame-threaded AV1 decoder refuses once more single-threaded (an intermittent failure in the full
  suite on files that decode cleanly).
- Memory: two glibc malloc arenas (`core/memlimit.py`, in `app.main` and the tests), so memory freed by worker
  threads goes back to the system; the test suite's quick validation of every check runs one model per child
  process. The full suite's peak fell from over 7.5 GB to 4.85 GB, `--validate`'s from 6.7 to 5.4 GB.
- Removed: the placeholder Assistant panel; the unpinned `claude` and `keyring` extras (the pinned `assistant` extra
  replaces them).

### Added (Phases 9 and 10 reviews)
- Spec §11 names as aliases: `--grasp-study`, `--cable-flex`, `--mujoco-validate out/` (a study given a folder
  writes `<folder>/<flag>.json`), `--formats usd` (the Isaac export), `--export … fiftyone --dataset`, and
  `--handeye-solve OUT --captures DIR [--joints FILE]` (capture from a folder of board images in batch).
- A Workspace page with a workspace check and its one-step suggestion (`core/workspace_check.py`, validation model
  `workspace`, 2 checks, a self-test line); design-log notes with a filter; a depth-range field on the Canonical view
  page; Teleop devices, Policies, Lights, Cables and Mounts groups in the Library panel; another variant beside the
  active one on the Requirements page; variant ghosts in the 3D view.

### Fixed (Phases 9 and 10 reviews and final gate)
- Truthfulness (sources re-read on 2026-11-07): the Alvium 1800 C-158c and U-240c streams' pixel format and bits per
  pixel are null (their maximum frame rates cannot be RGB8 rates over their links); Diffusion Policy (CNN)'s 0.1 s is
  for 10 DDIM iterations where the paper's Table 7 lists 16; the Franka Research 3 and Flexiv model folders' licence
  texts are recorded as they are (Apache-2.0 followed by a BSD text); the Isaac and OpenVLA exports' notes state their
  limits.
- Correctness: a `json:` factor pointer could move, re-model, delete or unlock a locked sensor (refused now, also in
  the experiments and the optimiser); the tolerance Monte Carlo and the optimiser counted a figure stated only as a
  bound as meeting a requirement; NSGA-II's crowding favoured designs whose figure is not stated; `--reproduce` of a
  scorecard for a non-active variant re-ran the active one; the Plücker "base" frame defaulted to the first robot, not
  the primary one; the optimiser's refused-design path returned a constraint vector of the wrong length.
- Workflows: bad factor values, unknown devices and malformed streams are refused in a sentence; the API raises
  `DexVantageError` throughout and `rig.save` to a folder writes the unpacked form; `--reproduce` of a file naming no
  study and `--optimise-settings '[1]'` explain themselves; an undecided bound shows no margin; a small Saltelli design
  says its indices are rough; the MCP server and the assistant open the new samples and treat the tolerance, experiment
  and optimiser runs as long work; the Panda sample's Diffusion Policy takes its two wrist views; the Menagerie import
  compiles off the UI thread and keeps the catalogue's name; analysis pages open readable; empty selectors say why;
  "No warnings."; a user-set mount tolerance is not labelled "(default)"; "1 sensor"; the assistant counts moving
  joints as the Outliner does; the brackets export's refusal names the menu.
- Throw-away folders (the self-test's and the tests' settings, the tests' XDG folders, the audit check's fallback
  folder, the plugin template's test folders) are removed when their process or test ends; two days of runs had left
  about 12 000 folders (4 GB) in the system temp folder, enough to fill it during a full test run.
- The assistant's and MCP's export tools take the `usd` alias, as `--formats` does.
- A disabled, ticked checkbox drew an empty box (the Requirements page's read-only "DexVantage default").
- Final gate: 3 482 tests pass (1 skipped), 179 of 179 validation checks, 37 of 37 self-test lines.

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
