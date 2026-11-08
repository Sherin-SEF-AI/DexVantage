# Build state

The resume point for the next session: what is done, what is next, open issues, and the decisions taken. Updated at
the end of every phase.

## Next

The user decides on the **0.1.0 release**: `packaging/release.py --publish` (after tagging and pushing `v0.1.0rc1`
or `v0.1.0`; run only when the user asks; it publishes a release candidate as a GitHub pre-release), turning on
GitHub Pages for `site/`, and whether to run the CI and bundle workflows first (none has run on GitHub yet). Then,
optionally, **Phase 12**: robot-aligned 3D Gaussian Splatting view augmentation (spec §9.3), a CUDA-only extra with
measured reconstruction quality. Carried over: the occlusion throughput (`core/occlusion.py`, the ALOHA 2 grasp
budget), ground truth with distorted lenses, and the open issues below.

## Phase 11: samples, documentation, website, packaging and CI; release candidate 0.1.0rc1 (done, 2026-11-08)

Built in parallel by separate agents (the user guide and generated references; the README and the website; the
PyInstaller bundles and the release script; CI and the bundle workflow; the performance budgets and the validation
page; nine polish items), then integrated. Version **0.1.0rc1** in `dexvantage/__init__.py` (pyproject reads it),
`CITATION.cff`, the CHANGELOG (`## 0.1.0rc1 — unreleased`, gathering everything that was under Unreleased), the
README and the site. Nothing was committed, tagged, pushed, released or uploaded.

**Final gate (2026-11-08, after the four reviews: truthfulness, a fresh install, packaging and CI, the definition
of done)** — run alone on this machine, each memory-capped, `TMPDIR`, `--basetemp` and the XDG folders under `out/`
(all deleted afterwards). These figures replace the gate's below.
- Tests: **3 991 passed**, 0 failed, 2 skipped (the paper kit's LaTeX compile: no LaTeX engine on PATH; the reason
  is printed) (`python -m pytest tests`, offscreen, 7 GB cap, 2 350 s). Peak memory: the largest process 5.17 GB
  (`/usr/bin/time -v` maximum resident set, 5 047 580 KiB), the scope's cgroup `memory.peak` 6.40 GB with page
  cache. The first full run had 1 failure (3 990 passed): `test_assistant_loop`'s chats-per-project test found a chat
  that `test_assistant_dock` had saved one second earlier. An unsaved project's chat folder is named by its name and
  its creation second, and both tests' SO-100 samples were made in the same second (reproduced: two
  `sample_project('so100')` made in one second share the folder). Fixed in the test (its project has a name of its
  own); the second full run is the figure here.
- Validation: **232 of 232** (`--validate out/validation`, 388 s, peak 5.42 GB). Every check's error equals the one
  `docs/validation.md` publishes (regenerated to a scratch file and compared: only the run time differs, 387 s
  against 383 s, so the page was left as it is). `tracking.store_round_trip` compares 443 fields;
  `docs/models/tracking.md` said 434 (fixed, site page rebuilt).
- **MuJoCo 4.0.0** (what a fresh `pip install` now gets): `--validate` in a throw-away environment that differs
  from `.venv` in MuJoCo only (4.0.0 with `mujoco-core` and `mujoco-studio`, the rest from `.venv`): **232 of 232**
  (388 s, peak 5.47 GB). 24 errors differ from the 3.17.0 run, all at round-off (the largest change
  `blur.image_velocity_fd` 2.8 × 10⁻⁹ to 6.2 × 10⁻⁹ against 10⁻⁵); the rendered checks are identical. The tests were
  not run with 4.0.0. README says so now (it said 4.0.0 had not been run).
- `dexvantage --self-test`: **46 of 46 ok** (45 s, peak 3.76 GB).
- `ruff check dexvantage tests scripts examples packaging` (0.17.4): clean. `actionlint` 1.7.12 on the three
  workflows: clean.
- Generated docs: `gen_cli_reference.py`, `gen_api_reference.py` and `gen_overview_refs.py --check`: up to date.
  `scripts/build_site.py`, then `--check`: 120 pages, no problems.
- MCP over stdio (`python -m dexvantage.mcp --sample so100`): `initialize` answered (protocol 2025-06-18 as asked,
  server `dexvantage` 0.1.0rc1, capability `tools`), `tools/list` 40 tools, each with an input schema.
- Linux x86_64 bundle rebuilt from the final source (`packaging/build.py --smoke`, 2 min 30 s, peak 3.59 GB; the
  one in `out/dist` predated the reviews' changes to `selftest.py`, `shards.py` and `run_dexvantage.py`): folder
  849.0 MB, download 315.0 MB (315 010 387 bytes, SHA-256 d14b8ba3…4e4f35db953), start-up 1.6 s median of 3,
  `--version` 0.27 s, cluster pool 14.3 s, experiments pool 4.0 s, the bundle's `--self-test` **46 of 46 ok** in
  46.3 s (`docs/packaging.md`, "The final gate's build").
- `packaging/release.py --dry-run` (never `--publish`): 1 release file, `SHA256SUMS.txt` and `RELEASE_NOTES.md`
  written to `out/dist`; it prints `gh release create v0.1.0rc1 … --verify-tag --notes-file … --prerelease`.
- Hygiene, over the 1 351 files git would commit: no personal paths, no e-mail addresses at all (pyproject and
  CITATION carry none), no keys (`sk-ant-`, `sk-`, `gsk_`, `hf_`, AWS, GitHub, Slack, Google, private-key blocks),
  no AI attribution lines (the hits are the rules and the tests that look for them), and none in the commit
  messages. One user name: `tests/test_packaging.py` used a Windows user folder named with this machine's login as an
  installer path; renamed to a made-up name. No file over 5 MB (the largest is `site/validation/validation.json`,
  1.0 MB). Media: `docs/media` 3.18 MB (17 files), `site/media` 3.52 MB (21), `site/docs-media` 2.11 MB (11); the
  whole of `site/` 11.3 MB (220 files).
- Recorded here from the two reviews that had no entry: the fresh-install review (a plain `pip install`:
  `--validate` failed 14 checks for want of optional extras and now leaves them out, naming the extra; the
  QPainter-fallback notice; Ctrl+C; the export's progress on a terminal; `tests/test_fresh_install.py`) and the
  truthfulness review (the README's MCP command, the CI-only bundle rows). Both are in the CHANGELOG.

**Phase 11 gate (2026-11-08)**
- Tests: **3 965 passed**, 0 failed, 2 skipped (the paper kit's LaTeX compile at both unit settings: no LaTeX
  engine on PATH) (`python -m pytest tests`, offscreen, memory-capped at 7 GB, run alone, 2 313 s; `TMPDIR`,
  `--basetemp` and the XDG folders under `out/`). Peak: the largest process 5.22 GB (`time -v` maximum resident
  set), the cgroup's `memory.peak` 6.32 GB with page cache. Three full runs: the first had 2 failures
  (3 962 passed) and the second 2 others (3 963 passed), all fixed below; the third is the figure here. New test files this phase: test_guide_examples (34 examples,
  about 6 min, one process each), test_ten_minute_tour (about 2.5 min), test_docs_reference, test_site 25,
  test_packaging, test_ci_workflows 20, test_benchmarks_doc, test_validation_doc, test_grasp_cull,
  test_phase11_polish 14, test_phase11_integration 4.
- Validation: **232 of 232 checks pass** (`--validate out/validation` through `scripts/benchmarks.py --only
  validate`, 384 s, peak 5.58 GB, 37 models); `docs/validation.md` is made from that report
  (`scripts/gen_validation_doc.py`) and publishes the measured agreement with MuJoCo renders per check (spec §18).
  Every figure the README and the home page quote equals the report's (unchanged from the research tools part 2
  gate).
- `dexvantage --self-test`: **46 of 46 ok** (45 s, peak 3.82 GB; "DexVantage 0.1.0rc1 self-test").
- `ruff check dexvantage tests scripts examples packaging`: clean. `actionlint` 1.7.12 on the three workflows: clean.
- Linux x86_64 bundle (`packaging/build.py --smoke`, the gate's source, Python 3.14.4, PyInstaller 6.22.3):
  folder **849.0 MB**, download `DexVantage-0.1.0rc1-linux-x86_64.tar.gz` **315.0 MB** (314 987 717 bytes, sha256
  8451f978…ef43d9); start-up to the main window **1.66 s** median of 3 (first 1.72 s); `--version` 0.20 s and
  0.25 s by hand (`dexvantage 0.1.0rc1`); batch run of three studies on SO-100 0.8 s; `--sample aloha --coverage
  --bandwidth` 2.15 s, peak 763 MB; cluster pool `--jobs 2` 14.1 s; experiments pool of 2 4.0 s; `--self-test` 46 of
  46 ok in 45.7 s. The packaging agent's run of `--validate --quick` from the bundle: 228 of 228 (the four other
  checks register only with gmsh, rerun-sdk or usd-core, which the bundle leaves out). In `docs/packaging.md`.
- Performance (spec §16, `docs/performance.md`, `out/bench/`; RTX 5080 through EGL device 0, load 5–9): the UI-thread
  budget now **passes on all six samples** (0 steps over 100 ms; slowest so100 79.6 ms, a motion-blur stall outside
  DexVantage's Python; dexhand 71.6 ms, aloha 33.5 ms, panda-bimanual 55.9 ms, g1 61.9 ms, ur5e 37.1 ms; re-measured
  after the cables fix below). The 2,000-grasp study (180 s per sample, a project default derived from §18's ten
  minutes, not a spec figure): so100 55.1 s, ur5e 114.2 s, dexhand 68.8 s, **aloha 182.1 s (misses by 2.1 s)**,
  panda-bimanual 79.9 s, g1 45.4 s. Start-up to first paint from source 1.50 s (blank) to 2.37 s (ALOHA 2), medians of 3; `import dexvantage`
  0.06 ms, first API name 72 ms.
- Screenshots: all groups refreshed (`scripts/screens.py`, then `--only samples`, `phase910`, `research2`: 142
  images, 225 + 206 + 246 + 42 s, none failed); `scripts/tour_media.py` (11 tour images in `docs/media`) and
  `scripts/build_site.py --validation --media` (27 media files, `validation.html` 232 of 232) re-run; `--check`: 120
  pages, no problems. Looked at the cables page and its Camera view overlay (the polylines arrive from the worker),
  grasp self-occlusion, statistics (pp), requirements scorecard, and the ALOHA 2 and G1 main windows: nothing to fix.
- Regenerated: `docs/reference/cli.md`, `docs/reference/api.md`, the overview's references, `docs/validation.md`,
  `docs/performance.md`, the site; the guide's figures checked by `guide_examples.py --check` (34 examples).

**Integration fixes**
- Two failures of the first full run: (1) the paper kit example printed 48 traced numbers where the guide said 49:
  the methods paragraph's version `0.1.0.dev0` counted as a number (`0.1.0`) and `0.1.0rc1` does not; the guide
  now says 48. (2) `test_review_phase458_product::test_recording_of_another_robot_is_compared_not_crashed`: a
  synthetic SO-100 video did not decode. Reproduced outside the suite: SVT-AV1 (PyAV 19.0.2's libsvtav1) prints
  "live_count should not be EB_ObjectWrapperReleasedValue" and now and then writes an undecodable stream (2 of 32
  and 1 of 40 datasets written four at a time; 0 of 40 with `svtav1-params lp=1`, which is about 11 × slower:
  11.1 s against 1.0 s for 300 frames at 640 × 480; `lp=4` still failed 2 of 40). The LeRobot writer now decodes
  every AV1 file it writes and encodes a bad one again with one thread (0 of 40 failed afterwards; a second failure
  is refused in one sentence naming H.264); synthetic episodes count their frames once.
- The cables page laid its polylines on the UI thread (106–149 ms on dexhand and panda-bimanual): now on a worker
  thread, the views drawing the last polylines until the new ones arrive (`polylines(wait=True)` computes in place).
- `validation/checks_publication.found()` ran `float()` before its multi-dot test, so a version-like token not in
  the word set raised ValueError (reported by the packaging agent).
- `api._fmt`: "1 tokens" is "1 token" (a counted word, unit alphabetic, longer than three letters, ending in s).
- `scripts/build_site.py` drops lines that are a single HTML comment (the guide's `<!-- example: id -->` markers,
  which markdown-it with html off printed as text), and copies the images a document shows into `site/docs-media/`
  (the sample tours' screenshots were GitHub blob URLs, which `test_site` refuses: the second full run's two
  failures were this and a fake key in this file's own text).
- `.github/workflows/bundles.yml` builds the Linux bundles on `ubuntu-22.04` and `ubuntu-22.04-arm` (glibc 2.35)
  instead of 24.04 (2.39), so they run on Ubuntu 22.04 and on Jetson with JetPack 6; `.github/scripts/apt-gl-qt.sh`
  takes GLib's package name of either release.
- `packaging/release.py --publish` refused a release candidate as a development version; `is_dev` (a `.devN`
  version) is now the refusal and an rc is published as a GitHub pre-release. Its tests no longer assume the
  repository's version is a development one.
- `tests/test_benchmarks_doc.py` held a fake Anthropic-style key that the hygiene test refuses (renamed).

**Decisions (Phase 11)**
1. The version is PEP 440 `0.1.0rc1` (the spec's "release candidate 0.1.0"); the CHANGELOG says so and keeps an
   empty `[Unreleased]` above it.
2. A release candidate may be published (as a pre-release) by `packaging/release.py --publish`, run only when the
   user asks; a `.dev` version never.
3. Linux bundles are built on the oldest glibc that the targets need (Ubuntu 22.04, JetPack 6), not the newest
   runner; the tests run on 24.04.
4. Overlays that take more than the UI budget are computed on a worker thread and drawn from the last result until
   the new one arrives (the cables page first).
5. AV1 files are verified by decoding after writing rather than always encoded with one thread (11 × slower).
6. The 2,000-grasp budget of 180 s stays a labelled project default; ALOHA 2 misses it by 2.1 s and the page says
   so (no re-run to get under it).

**Spec corrections (Phase 11)**
- GitHub retired the `macos-14` runner on 2026-11-02 (runner-images issue 15226): CI uses `macos-15` (also Apple
  silicon).
- "Linux arm64 for Jetson": PyQt6 6.8 and later have aarch64 wheels for glibc 2.39 only, and JetPack 6.2 is Ubuntu
  22.04 (glibc 2.35), so the arm64 bundle pins PyQt6 6.7.1 / Qt 6.7.3 (`packaging/requirements-bundle.txt`) and
  builds on `ubuntu-22.04-arm`. DexVantage has been tested on PyQt6 6.12 only; the arm64 CI self-test is the first
  check of 6.7.
- §16's 2,000-grasp budget gives no time; this project's default is 180 s on a mid-range GPU (derivation in
  `docs/performance.md`).
- §17 "release candidate 0.1.0" is written `0.1.0rc1` (PEP 440).

**Open issues (Phase 11)**
- **Not built or run here:** the Windows installer, the macOS app (signing and notarisation untried), the Linux
  arm64 bundle; no workflow has run on GitHub; the suite has run on Linux only (the Windows and macOS test jobs are
  required, so platform failures will show). Under llvmpipe the rendered tests are 2–3 × slower than on the GPU, so
  the 75-minute job limit and the shard weights may need tuning after the first run.
- **Runners:** GitHub retires its oldest label once a newer image is out (Ubuntu 26.04 is out): when `ubuntu-22.04`
  goes, the Linux bundles need another glibc-2.35 build machine (an `ubuntu:22.04` container) to keep JetPack 6.
- **Grasp budget:** ALOHA 2 182.1 s against 180 s; profiled: `core/occlusion.py` `CubeMap.build` 54 % (per-map
  Python work in the GL backend, `glReadPixels`, the far-plane distance over every vertex) and `Evaluator.occluders`'
  posing 29 %. The fix `docs/models/occlusion.md` proposes (upload each link mesh once and draw it from the link
  transforms on the GPU) would remove most of it.
- **UI timings:** the ALOHA 2 motion-blur stall (72–369 ms in the benchmark agent's runs, little of it the UI
  thread's own processor time, cause not found) did not show in the gate's run (slowest 33.5 ms on ALOHA 2); so100's
  79.6 ms motion-blur stall is also outside DexVantage's Python. Building the window (1.4–2.1 s) and show to first
  paint are reported as start-up, not counted against the 100 ms budget.
- **MuJoCo 4.0.0** (2026-11-06): the final gate ran `--validate` with it (232 of 232; 24 errors differ from 3.17.0's,
  all at round-off), but not the tests; `pyproject` says `mujoco>=3.1` (CI pins 3.17.0): an upper bound is the lead's
  call. The bundle pins were measured with Python 3.14 here; CI builds with 3.12 (every cp312 wheel not
  confirmed).
- **Bundle size:** the bundle built here carries opencv-python (with its own Qt 5 libraries); CI uses
  opencv-python-headless, estimated 36 MB smaller from the wheels (not measured on a bundle).
- **Docs:** the guide's `{#id}` heading anchors are the site builder's syntax: on GitHub they show as text and
  `user-guide.md#batch`-style links land at the top of the page (the site resolves them). 56 public API members have
  no docstring ("No docstring." in `docs/reference/api.md`). The site is a snapshot: rerun `scripts/build_site.py
  --validation --media` after docs change (no test fails when it falls behind). `docs/performance.md --doc-only`
  needs `out/bench/` (ignored by git); `grasp_ab.json` (the grasp change's before and after) cannot be re-measured.
- **Tests that fail on purpose until regenerated:** `test_docs_reference` (after a parser, API docstring or check
  change: `gen_cli_reference.py`, `gen_api_reference.py`, `gen_overview_refs.py`), `test_validation_doc` (after a
  check is added or re-described: a full validation run, then `gen_validation_doc.py`), `test_guide_examples` (a
  changed sample figure: `guide_examples.py --check`, then the guide text). The two new example tests add about 8.5
  minutes to the full suite.
- **Polish leftovers:** setting names are checked for 11 studies only (the others' settings come from specs or pass
  through); the paper kit's statistics and requirements tables have one Unit column, so a difference of percentages
  shows "%" there where it is pp; ruff's annotation rules flag 9 298 missing hints outside the nine API modules it
  checks (1 067 public functions without a return type; see the definition-of-done review below);
  `mlruns` run `meta.yaml` still writes `deleted_time: null` and absolute artifact URIs.
- **SVT-AV1:** its "live_count …" errors still print on stderr while it encodes; the writer catches the bad files by
  decoding them (more time per dataset).
- **Not yet live:** <https://sherin-sef-ai.github.io/DexVantage/> needs GitHub Pages turned on; the release page
  needs a release.

**Definition-of-done review (spec §14–§18, 2026-11-08)**

Every item of §14–§18 was checked by running it or reading the artefact.
- **§14 samples:** all five are present, and the extra Shadow Hand sample too. ALOHA 2 has the four D405s of the paper,
  named as ALOHA's recordings name them. SO-100 has a wrist and a third-person camera. UR5e + 2F-85 has a wrist D405
  and an overhead D435. The Panda pair says there is no official bimanual model. The G1 has the head D435 where
  Unitree's URDF puts it; the hand is not called Dex3, because Menagerie does not name it. Each sample has a tour in
  `docs/guide/samples.md`.
- **§15:** the README sections, the install table (with Linux arm64 for Jetson), overview.md (every model with its
  equations, references and limits; all 58 `docs/models/*.md` linked), user-guide.md (every registered study and
  exporter named, with worked examples), project-format.md, `project.schema.json` (docs and resources copies
  identical), CHANGELOG, CITATION.cff, `packaging/release.py` and the site (`build_site.py --check`: 120 pages, no
  problems) are all there. One bundle is built (Linux x86_64; `--version` from the bundle prints 0.1.0rc1), and
  MuJoCo's libraries are in it. The other three are CI-defined only. `--self-test` runs in CI on every bundle and on
  the source install on Linux, Windows and macOS.
- **§16:** ruff config: yes. Validation in CI: a quick run of every check (fewer trials, same tolerances), on Linux
  only. Skips: each states its reason (`test_phase11_polish`). Budgets are measured and published in
  `docs/performance.md`.
- **§18, by running it:** `scripts/demo-phase11.sh` (new) was run here with QUICK=1 and NO_WINDOW=1 and exited 0
  (622 s). Results:
  - self-test 46 of 46 ok;
  - ALOHA 2: grasp score 100.0 % (Wilson 95.7–100), and 4.7 % "blind at contact" with `cam_left_wrist` pitched from
    −65° to −75°. For `cam_high` the contact points are 16 % seen, 39 % behind their own finger and 46 % hidden by the
    scene;
  - depth validity, VLA tokens and bandwidth ran; MJCF and LeRobot exports: 27 + 11 files with the manifest;
  - 2 bracket STLs (the brackets study's suggested types applied);
  - tour compute 137 s;
  - 3 synthetic SO-100 episodes audited, each with a verdict and the requirements it failed;
  - the quick validation run: 232 of 232; the reference generators and the site check are current.

  All of these equal the user guide's figures. Published agreement with MuJoCo renders: `docs/validation.md`
  ("Agreement with MuJoCo renders"). Headline tooltips name their model and inputs on every page
  (`test_phase11_polish`).

Fixed in this review:
1. **§16 type hints on the rest of the public Python API.** `dv.diff`, `dv.merge`, `dv.bibtex`, `dv.figure(s)`,
   `dv.export_runs`, `dv.Diff`, `dv.Merge`, every `dv.cluster.*` method and the plugin SDK (`PluginAPI.*`,
   `plugin_sdk.load`/`discover`/`unload`) had unannotated parameters or returns (89 findings). They are now
   annotated, with the names imported only under `TYPE_CHECKING`. ruff's ANN rules now cover `api_research.py`,
   `api_cluster.py` and `plugin_sdk.py` too (the pyproject per-file-ignores line), and `docs/reference/api.md` was
   regenerated. Test: `tests/test_dod_review.py`, which uses an AST walk independent of ruff and also checks that
   every name a hint uses can be resolved.
2. **§17 "a short demo script" at the gate.** Phases 3–11 had none. `scripts/demo-phase11.sh` walks §18 in batch mode
   and through the Python API, prints the figures, and exits non-zero if fewer than 40 checks pass. Its text, Python
   and flags are tested in `tests/test_dod_review.py`. Phases 3–10 still have no demo script of their own; this one
   covers their product-level outcome.
3. **§16 wording on `docs/performance.md`:** it quoted the budget as "on a GPU". It now says "a mid-range GPU", and
   that only the one GPU listed was measured and is not classed here (`scripts/benchmarks.py`, page regenerated with
   `--doc-only`).

What remains (not feasible in this review):
- **"Mid-range GPU" (§16):** the 2,000-grasp times exist for one card only (an RTX 5080). Run `scripts/benchmarks.py
  --only grasp` on a card the lead considers mid-range and publish it. ALOHA 2 is already 2.1 s over the project's
  180 s default on the RTX 5080.
- **Type hints across the codebase:** 1 067 public functions outside the nine checked modules have no return type
  (mostly UI widgets, studies and core helpers). Extending the ANN check module by module is the route; doing it in
  one pass would touch about 300 files that other work edits.
- **Bundles for Windows, macOS and Linux arm64, and the CI runs:** defined, not run (as above).
- **Batch bracket export of a sample:** `--export … --formats brackets` refuses until the project has bracket types.
  The window's Apply, a saved project or the Python API (as the demo does) sets them. There is no batch flag that
  applies the brackets study's suggestion.
- **"Every figure on screen has a tooltip with its model and inputs" (§18):** tested for every page's headline figure.
  Table cells and chart points are not covered by a test that sweeps them all.

**Packaging and CI review (2026-11-08)** — every workflow and packaging file read line by line; details and how each
figure was taken in `docs/packaging.md` ("Packaging and CI review"), regression tests in
`tests/test_packaging_ci_review.py`.
- Checked: the seven action SHAs are their tags' commits and the newest releases (`git ls-remote`); the actionlint
  checksums match its release; every runner label is listed in the runner-images README; all 24 bundle pins have
  CPython 3.12 wheels on the four bundle platforms (Linux at glibc 2.35 or older) and the arm64 wheels pip picks on
  macOS 15 need macOS 14.0, as the plist says; every source file compiles under CPython 3.10 and 3.12; the shards
  cover every test file once. The shard weights lacked 34 files (counted as the median, about 5 s, though
  `test_guide_examples.py` takes 374 s): `scripts/ci_test_durations.json` is now from the review's full run.
- Run: the rebuilt Linux bundle (`build.py --smoke`, passed) through all 35 pages on all six samples with each page's
  study on the UR5e sample (no missing module, resource or traceback; 5 studies refused for want of input), and 82
  batch commands covering every flag (none failed but as documented); the rendered tests and `--validate --quick` on
  Mesa llvmpipe alone with `MUJOCO_GL` unset, as on a GPU-less runner (212 of 213 passed, the other on another
  agent's half-written file and alone it passed; 232 of 232); 52 render-related test files with no EGL at all, as on
  Windows and macOS (1 019 passed, 82 skipped with the right reason, none failed).
- Full suite at the end (7 GB cap, 2 612 s): 3 990 passed, 2 skipped (no LaTeX engine), 1 failed: the hygiene test
  on a comment of this review that spelled a Windows user folder; fixed, and that file and the review's own test
  files passed again.
- Fixed: on Windows every spawned pool worker died (`import resource` at the top of `core/cluster/shards.py`); the
  self-test failed four lines on a plain `pip install` (pyarrow, jsonschema missing: now warnings naming the extra,
  failures where the package is installed); the start-up probe used the real settings on Windows and macOS and needed
  a stdout the console-less Windows launcher may lack; `build.py` set no `TMP`/`TEMP` and its home-folder scan missed
  Windows' other spellings; bundles.yml built the executable's path with backslashes for Git Bash, used Git Bash's
  `/tmp` for a native program, and drafted the release candidate as a full release; ci.yml set `MUJOCO_GL=egl` for
  the whole process (never measured that way), lacked UTF-8 mode for 77 encoding-less opens in the tests, and
  linted less than the gate; `windows-latest`/`ubuntu-latest` pinned to `windows-2025`/`ubuntu-24.04`; two mock-sbatch
  tests and the installer-escaping test skip on Windows with the reason; wrong comments (a `build.yml`, "MuJoCo renders
  through GLFW on Windows and macOS": DexVantage renders only through EGL, so on Windows and macOS the rendered
  features stop with "MuJoCo cannot render offscreen through EGL on this machine").
- Decision: CI's test and validation processes run with `MUJOCO_GL` unset, as users run them; only the render worker
  sets EGL (it always did).
- Still open: nothing has run on a GitHub runner, a Windows or a macOS machine; the Windows smoke test and installer,
  the macOS app and the arm64 bundle are untested; the shard weights were measured on this machine only.

## Research tools part 2 and the open-issue fixes (done, 2026-11-08)

Built in parallel by separate agents (the experiment tracker; statistics and power analysis; the campaign planner;
the sim-to-real gap; the publication kit; cluster sweeps; the multi-cell comparison; the export and import fixes
(USD robot import, URDF inertials, OpenVLA encodings, the Isaac principal point); cited prices; the UI and analysis
fixes of the review round), then integrated, then reviewed four times (truthfulness, correctness, a research team's
workflow on ALOHA 2, completeness against spec §7.11, §9, §11, §12 and §13).

**Final gate (after the four reviews, 2026-11-08)**
- Tests: **3 806 passed**, 0 failed, 2 skipped (the paper kit's LaTeX compile at both unit settings: no LaTeX engine
  on PATH) (`python -m pytest tests`, offscreen, memory-capped at 7 GB, run alone, 1 850 s; temporary folders on the
  home disk with `TMPDIR` and `--basetemp`, because `/tmp` is a 15 GB tmpfs with a per-user quota that other work
  fills). Peak: the largest process 5.02 GB (`time -v` maximum resident set), the cgroup's `memory.peak` 6.14 GB with
  page cache (the integration gate 5.09 GB and 6.30 GB). Afterwards its temporary folder held only the shared icon
  cache. The reviews added test_review_correctness 17, test_product_review 24, test_completeness_api 16 and
  test_completeness_plugins 6 and asserts in existing files (the integration gate ended at 3 743).
- Validation: **232 of 232 checks pass** (`--validate out/validation`, 387 s, peak 5.29 GB).
- `dexvantage --self-test`: **46 of 46 ok** (45 s, peak 3.8 GB; the validation line runs 160 cheap checks in 32 s;
  occlusion and grasp on the RTX 5080, the OpenGL line on the default display).
- `ruff check dexvantage tests scripts examples`: clean.
- `python -m dexvantage.mcp` over stdio: initialize answers protocol 2025-11-25 (server "dexvantage" 0.1.0.dev0),
  tools/list **39 tools**, `open_project` with `sample: aloha`, then `get_runs`, `get_campaign` and `get_rig_cost`
  (1 300 USD) answer, `get_statistics` and `get_cells` refuse in one sentence (no success counts, one cell); 1.1 s.
- Batch, every research tools part 2 flag on the samples (58 runs, each under a 4 GB cap, one at a time; largest
  1.13 GB, the three-sample cell comparison); results exit 0, refusals one sentence and exit 2, a failing check or an
  incomplete gather exit 1, no traceback in any run. The figures equal the integration gate's: `--stats` 14/20
  against 19/20 +25.0 pp [+0.8, +47.3], adjusted p 0.0915; `--campaign` ALOHA 2 216 episodes, 1 304 GB, UR5e
  1 938 GB; `--sim2real` on the UR5e +20 DN recording 6.4 s, SO-100 synthetic 8.0 s; the paper kit 48 and 31 numbers
  traced (UR5e), 36 and 60 (SO-100 from batch results); `--experiment --jobs 4` 1.9 s, `--tolerance-mc --jobs 4`
  7.8 s; `--slurm` 3 shards, `--run-shard` × 3, `--gather` incomplete (exit 1) and complete; `--cells` three samples
  0.373 (ALOHA 2), files 0.745 (G1). Run for the first time at a gate: `--campaign-saved` (ALOHA 2, the stored
  spec), `--stats-saved` and `--cells-compare` on a project with nothing set (refused in one sentence),
  `--stats variants.json` on an SO-100 saved with a second variant made by `rig.add_variant` (7.1 s: task volume
  seen −0.182 [−0.193, −0.169] points, adjusted p 0.0078, "clear difference"; the three figures the SO-100 does not
  state are named with the study's reason), `--cluster-studies` with `--jobs 3` (three samples, three studies,
  1.3 s) and with `--slurm`, `--optimise --slurm` (budget 12, population 4) through three rounds of `--run-shard`
  and `--optimise-continue` to `result.json` (1 design at least as good as the rig on every objective); refused:
  `--optimise-continue` on a folder that is not a job, `--submit` without `--slurm`.
- Screenshots: `python scripts/screens.py --only research2` (15 images, 41 s, none unfinished). Looked at the six new
  pages: Runs (4 runs, the tracking line on one row), Statistics (+25.0 pp, "no clear difference", the plan of 43
  episodes in the Preview line), Campaign (216 episodes, "within budget", the demonstration fields the power
  analysis sets disabled), Sim-to-real (19.9 DN, "very large gap", recorded, rendered and difference frames, the
  causes with their gain and offset), Paper kit (33 of 33 numbers, "ready", sections in three columns) and Cells
  (0.373, "large gap", the three cells and their bands): nothing to fix on the pages. The script left its three
  XDG folders (and, in other sets, its capture and export folders) in the temporary folder; it now removes them when
  it ends (`scripts/screens.py`), checked by a second run that left only the shared icon cache.
- Hygiene: every tracked and untracked file (1 054) grepped for personal paths, user and host names, e-mail
  addresses, keys (`sk-ant-`, Groq, Hugging Face, AWS, GitHub, private-key blocks) and AI attribution lines
  (Co-Authored-By, Claude-Session, "Generated with"): none (the hits are the rules and tests asserting their
  absence). No file over 5 MB was added (the untracked files total 2.0 MB).

**Review fixes (research tools part 2)**
- Truthfulness (all 81 cited prices re-read on their maker or official price pages, in USD, and all match: RealSense
  11, Stereolabs 11, Luxonis 10, Orbbec 8, NVIDIA's Jetson FAQ 11, e-con Systems 6, Leopard 6, Elgato 3, GelSight and
  DIGIT 2, ROBOTIS 4, AgileX, I2RT, UFACTORY 2, Rokoko, Amaran, 3Dconnexion, Apple and Trossen; the statistics
  worked examples re-derived with code written apart from `core/stats`; the file formats checked against their
  sources; 8 fixes, among them the MLflow tracker's docstring on the experiment `meta.yaml` keys MLflow 3.17.2 adds).
  Validation of the four models it touched (campaign, sim2real, publication, tracking) 22 of 22.
- Correctness (8 bugs, each with a regression test in `tests/test_review_correctness.py`): Fisher's exact test named
  its one-sided alternatives the other way round from the rest of `core/stats` (condition 2 against condition 1); a
  short `os.write` in a shard's output left half a line the next line ran into (both lost); the shared disk cache
  never replaced a short or corrupt entry; two deletes of one run at once raised a raw OSError; a tolerance run's
  samples dropped draws without a value, so two runs with one seed that lost different draws were paired off by
  one, and the tracker's significance test refused lists holding a missing draw; dataset pooling compared joint
  kinds by position after a reorder (a false "not poolable"), took a unit not stated as another kind and ignored a
  target recorded in degrees; the optimiser kept the cost objective when a device is priced in another currency,
  and a price without a currency counted as the currency "None"; the campaign says that a compression ratio above
  one grows the recording (a ratio such as 20 entered for 0.05 is named: the ratio is the encoded size over the raw
  size).
- A research team's workflow on ALOHA 2 (15 fixes, `tests/test_product_review.py`, `docs/models/product-review.md`):
  `rig.add_variant(name, {factors}, from_variant=None)`; the API logs `dv.stats.*` on a rig, `dv.campaign.plan`,
  `dv.cells.compare` and `dv.cluster.gather` (with `rig=` into the rig's store) as their batch forms do; the
  sim-to-real result held NaN for a mount σ the frames do not constrain (now `None`, so the tracker and JSON take
  it); a brightness offset was read as a light gain (the cause names the offset, and Apply sets the cameras' render
  response when they share an offset of 1 DN or more); `rig.gap_dataset` checks its gaps first and refuses a folder
  that is not empty before rendering; a campaign spec refuses an unknown key (a mistyped power analysis gave 50
  demonstrations silently); `dv.cluster.*(jobs=0)` and other non-counts are refused; a run compared with itself is
  refused; pxr's C++ names and full paths are kept out of a USD import error; the Statistics page's Variants,
  Experiment runs and Tracker runs panels keep their controls together and the figures field is labelled; "+0 %
  [+0, +0]" is written 0; "1 p" reads "p = 1"; "1 sessions" and the HTML schedule's plurals; the Cells page refuses a
  missing file when it is added; a logged designed experiment states its duration.
- Completeness (5 gaps, `tests/test_completeness_api.py`, `tests/test_completeness_plugins.py`,
  `docs/models/research-api.md`): the part 1 research tools from Python (`dv.diff`, `dv.merge`, `dv.bibtex`,
  `dv.figures`, `dv.figure`, `dv.export_runs`, `dexvantage/api_research.py`, with no Qt import); a plugin's
  validation checks (`PluginAPI.check`, `validation/suite.py` `PLUGIN_MODELS`: its own model, cited references,
  removed when the plugin is unloaded or fails); the headline of a study without a headline metric shows its unit
  and quantile; a process pool started from a script's unguarded top-level code is refused in one sentence
  (`core/cluster/local.check_main_guard`); the research workflow example (`examples/03_research_workflow.py`, a
  jupytext percent-format notebook, run by its test).

**Integration gate (2026-11-07)**
- Tests: **3 743 passed**, 0 failed, 2 skipped (the paper kit's LaTeX compile at both unit settings: no LaTeX engine
  on PATH) (`python -m pytest tests`, offscreen, memory-capped at 7 GB, run alone, 1 859 s). Peak: the largest
  process 5.09 GB (`time -v` maximum resident set), the cgroup's `memory.peak` 6.30 GB with page cache (0.7 GB under
  the cap; the earlier runs of this gate 4.9–5.2 GB and 6.08–6.11 GB); no test module needed moving to a child
  process. Seven full runs were made; the failures of the first six are fixed below. New test files:
  test_tracking 21, test_tracking_page 3, test_stats 26, test_stats_page 5, test_campaign 33, test_campaign_page 3,
  test_sim2real 17, test_publication 19, test_cluster 21, test_cluster_ui 4, test_cells 21, test_cells_page 3,
  test_robot_io 14, test_prices 29, test_ui_fixes 28, test_integration_rt2 14 (3 745 collected; the Phases 9 and
  10 gate ended at 3 482).
- Validation: **232 of 232 checks pass** (`--validate out/validation`, full run 397 s on the final code, peak
  5.3 GB). 53 new since Phases 9 and 10: tracking 5, stats 13, campaign 6, sim2real 7, publication 4, cluster 4, cells 6, robot_io 5,
  library 2 (prices), experiments 1 (`experiments.censored_best`); each model's figures are in its document and the
  overview.
- `dexvantage --self-test`: **46 of 46 ok** (44 s, peak 3.8 GB). New lines: prices, censored bounds, cluster,
  publication, campaign, cells, sim2real, statistics, robot io, tracking; the validation line runs 160 cheap checks
  (31 s); the MCP line lists 39 tools.
- `ruff check dexvantage tests scripts examples`: clean.
- MCP: `tools/list` 39 tools (the registry's 33 plus the server's six); the counts and the nine new reading tools
  are asserted in `tests/test_integration_rt2.py`.
- Batch, every new flag on the samples (each under a 4 GB cap, run one at a time, times include start-up; RTX 5080
  through EGL); results exit 0, refusals one sentence and exit 2, no traceback in any run:
  - `--track runs` with `--coverage` (0.7 s), `--check defaults` (10.0 s, exit 1: not ready), `--runs` (0.3 s),
    `--runs-export --runs-formats csv,parquet,mlflow,wandb` (0.6 s); refused: `--track-tag` without a store,
    `--runs-formats csv,xls`.
  - `--stats rates.json` (ACT 14/20 against Diffusion 19/20: +25.0 pp [+0.8, +47.3], adjusted p 0.0915, "no clear
    difference"; 0.7 s); refused: one condition, broken JSON, a missing file, a variants source on a one-variant
    project (logged as a failed run with `--track`).
  - `--campaign` ALOHA 2 216 episodes over 2 days, 1 304 GB, within budget (0.7 s), UR5e 216 episodes, 1 938 GB
    (0.8 s); refused: equal success rates for the power analysis, no project.
  - `--sim2real` on a UR5e recording with an injected +20 DN gap (made with `rig.gap_dataset`): "very large gap",
    exposure ranked first (6.4 s, 0.79 GB); on an SO-100 `--synth-episodes` campaign: "large gap" (8.0 s); refused: no
    dataset, a dataset path that is not there.
  - `--export --formats paper_kit` UR5e with five studies: 4 parts, 4 figures, 16 BibTeX entries, 48 methods numbers
    and 31 table numbers traced (2.0 s); SO-100 from batch results (coverage, the scorecard, the tracker summary, a
    three-sample cell comparison): 7 parts with the cells and runs tables, 36 and 60 numbers traced (0.6 s);
    refused: an unknown section.
  - `--experiment DESIGN.json --jobs 4` (an 8-run Latin hypercube on the SO-100, logged with `--track`: 1.9 s);
    `--tolerance-mc --jobs 4` (64 draws, 7.4 s, peak 554 MB per worker); refused: `--jobs 0`, `--jobs` without a
    target.
  - `--slurm job --shards 3` (0.3 s), `--run-shard` × 3 (0.5–0.6 s each, peak 370 MB), `--gather` with a shard
    missing (exit 1, the missing shards named) and complete (0.6 s, logged with `--track`: "Best of 8 runs: task
    volume seen 78.41 % at run 5, against 94.52 % for the rig as it is"); refused: shard 5 of 3, `--shard` without
    `--of`.
  - `--cells sample:so100 sample:ur5e sample:aloha` with the HTML page and the pooling mapping (4.9 s, 1.1 GB):
    largest gap 0.373 (ALOHA 2), 1 of 3 cells pool; files SO-100, UR5e and G1 (3.0 s): 0.745 (G1); refused: one
    cell, a missing file, a reference cell 9 of 2.
- Screenshots: `python scripts/screens.py --only research2` (15 images, 41 s, none unfinished): the Runs page with a
  comparison (SO-100, a camera raised 30 mm: 94.52 % against 90.76 %), Statistics with the power curve (39 episodes
  needed, 43 planned), the ALOHA campaign, the sim-to-real gap with recorded, rendered and difference frames, the
  paper kit preview, the three-sample cell comparison, the Run on… dialog (both modes) and the USD import wizard on
  the bundled UR5e. Looked at and fixed: the run store's full path wrapped and was cut off (one line now: the folder
  name and where it is, the path in the tooltip); the Statistics page's hidden source panels set its height and its
  spare height fell between its controls (now below them); its Preview computed the suggested plan while the calculator's fields
  still showed the stored one; the sim-to-real camera box cut "Overhead D435" to "Overhea"; "-0 ms", "-0.0 DN" and
  "+0, -0, -0" figures (a value that rounds to zero is written 0); the paper kit's eleven section boxes in one
  column and its part box cut "Methods"; the import wizard's file line ran under the Browse button and a USD robot
  whose articulation sits on its base link was named "base" (now the stage's name, "ur5e"); the campaign form
  offered "Demonstrations per condition" as editable while the power analysis sets it (the unused fields are
  disabled).
- UI-thread budget (`scripts/measure_ui_rt2.py`, offscreen, alone): **0 steps over 100 ms** on ALOHA 2 and the G1
  with tracking on. Largest: 19.8 ms (the events after the cell comparison arrives, ALOHA 2); every new page shows
  within 18 ms; Apply 11–14 ms (campaign, sim-to-real gap); an edit with every page built 5.8 ms (ALOHA 2) and
  7.0 ms (G1); the Run on… dialog opens in 0.8–1.5 ms; the USD import wizard starts its worker in 7 ms. Workers:
  sim-to-real 4.8 s (ALOHA 2) and 1.5 s (G1), cells 1.2 s and 0.4 s.
- Hygiene: every modified and new file grepped for personal paths, user and host names, e-mail addresses, keys and AI
  attribution lines: none (the hits are tests asserting their absence). No file over 5 MB was added
  (`resources/usd_robots` is 108 kB).

**Integration fixes and cross-track hooks**
- The tracker logs cluster runs (`--jobs` experiments, tolerances, optimiser and study lists; a complete `--gather` of
  an experiment or a Monte Carlo; `dv.cluster`'s results), `--stats`, `--campaign`, `--cells` and the study and check
  steps of a `--run-plan` (each was unlogged); the tracker's end-of-run line comes last so it counts them.
- The Runs page logged through the page gate (`ui/pages/_gate.py`, built in parallel): while it was hidden, a
  finished study was held, replayed only when the page was shown, coalesced, and dropped if an edit came between.
  It now listens to the analysis controller itself (`tests/test_integration_rt2.py` fails without the fix).
- The paper kit writes `tables/statistics.tex` (each comparison's test, both sides, B − A with its interval in
  percentage points for rates, the corrected p, the verdict), `tables/cells.tex` (the figures per cell, bounds as ≥/≤,
  the gap index to the reference) and `tables/runs.tex` (the newest 20 tracked runs with their headline figure)
  whenever those results are present, each number traced and checked like the other tables; the methods paragraph
  points to them; the runs are read from the store when the project's tracking is on.
- The tracker keeps an experiment best run's bound and settled flag (a bounded best read as exact); the Runs summary
  keeps a figure's bound; runs that start in the same millisecond keep the store's order (a test failed when two
  quick runs swapped).
- `--jobs 0` was taken as "no `--jobs`" and ran serially; it is refused in one sentence.
- The occlusion cube-map cache keyed its origin to 0.1 mm, so a long-lived process reused a map built for a nearby
  origin and differed from fresh shards in the last digits of a coverage figure (69.2431 % against 69.2448 %); the
  origin is keyed exactly.
- A spawned cluster worker reported its parent's peak memory: Linux keeps `ru_maxrss` across the exec, so a worker of
  the 2.9 GB test process read 2 918 MB (a full-suite failure); the peak is now `VmHWM` of the worker's own address
  space.
- The report's "all studies" left out only the Phase 9 meta studies, so a report or paper kit with no results given
  would run the cell comparison, the statistics' Monte Carlo and the sim-to-real renders: they join `META_STUDIES`.
- The self-test's prices line had the wrong signature (it failed the self-test test).
- `test_ui_fixes`' Apply timing failed in full runs (one Apply at 111 ms, then every Apply at 90–190 ms, against
  7–57 ms alone): the cycle collector's pauses over the test process's 5 GB heap. Reproduced alone with a 4-million
  object heap (Applies of 535–674 ms) and gone with the collector paused around the timed call, which the test now
  does (it collects first); the window's own heap is a small fraction of the suite's.
- With the collector paused, the Tasks page's first Apply still read 110–119 ms on ALOHA 2 in five full runs (thread
  CPU, not waiting; two worker threads alive), within budget in a fresh process. A cold Apply samples on the UI
  thread after a MuJoCo compile (90 ms) and the pads' ray casts (100 ms), so the page now builds both on a worker
  as soon as it offers Apply (`ui/pages/tasks.py` `_warm`); that did not change the full-run figure, and a profiled
  repeat of the same Apply took 29 ms with no hotspot. The test now holds a repeat to the budget and reports the
  first figure; the one-time cost of a process that has run about 3 700 tests was not found (open issue).
- The LeRobot reader's single-threaded retry (Phases 9 and 10) covered a packet the frame-threaded AV1 decoder
  refuses; a second full run here saw the threaded decoder return too few frames instead ("2 frame(s) … could not be
  decoded") on a file that decodes cleanly alone. Missing frames are now retried single-threaded too (a damaged file
  still fails both ways; test in `tests/test_integration_rt2.py`).
- Two tests asserted what changed: the occlusion cache test expected a map 1 µm away to be the same (now its own
  map), and the paper kit's run-plan test the study registry's import order (now a set).
- The assistant's `check_robot_file` tool (what a URDF, MJCF or USD file references, before an import) and page guides
  for the six new pages; `docs/project-format.md` describes the new `analysis.studies` settings; the overview gained
  the missing `workspace` anchor.

**Shared contracts (additive changes, as reported):** `core/provenance.py` leaves the tracking setting out of the
content hash; `core/jobs.py` (sim-to-real); `core/optimise/nsga2.py` and `rig.py` (`prefetch`, `cache`) and
`core/studies/tolerance.py` (`evaluator`) for the cluster; `validation/suite.py` `tolerance_quick` in `@check`;
`ui/pages/base.py` and `_gate.py` (hidden pages held; `window.pages[pid]` follows); `core/metric_defs.py` 20 new
metrics, 73 in all (`tracking.runs`, `stats.*` 3, `campaign.*` 5, `sim2real.*` 6, `cells.*` 5); `library.schema.json` the `price`
block; `core/exporters/report.py` `META_STUDIES` gained `cells`, `stats`, `sim2real`; `core/publication.SECTIONS`
gained `statistics`, `cells`, `runs`.

**Decisions (research tools part 2)**
1. **Tracking is opt-in** and kept beside the project (`<name>.runs/`), never inside the `.dexv` file; the setting is
   not part of the content hash. MLflow and W&B are written as files (their packages only read them back).
2. **Statistics are fixed-sample** (no sequential testing); success rates are tested with Fisher's exact test by
   default, figures over draws or runs with permutation tests; a difference that is not clear is reported with the
   smallest difference the data detect, never as equality.
3. **The campaign planner's power analysis goes through `core/stats`** (its own formula stays as the fallback and the
   result shows both).
4. **The sim-to-real gap is a render of the project's model**; what it cannot model is reported as left, not fitted.
5. **Cluster results equal the in-process ones** (gathered shards are compared bit for bit in tests and validation);
   Slurm only.
6. **The cell gap is geometric** (ported from Vantage's heuristic index, weights and bands labelled defaults).
7. **`rig.cost` counts the enabled sensors only**; hosts, lights, cables and robots are priced in the library but not
   added (it would change the optimiser and requirements: a decision for the lead engineer).
8. **The batch form of the sim-to-real gap** is `--sim2real OUT.json --dataset PATH` (as `--silhouette-calib`), not the
   spec's literal `--sim2real dataset_path`.

**Open issues (research tools part 2)**
- **Tracker:** logging in the window hangs off the Runs page (built when the window starts; if pages are ever built
  on first open, the hook moves into the window); each logged run snapshots the project on the UI thread (within
  the budget on the samples, not measured on 10 000-grasp sets); `mlruns/` artifact URIs are absolute (a moved or
  shared copy keeps its figures but not its artifact links, and shows the original folder); the pinned
  mlflow-skinny 3.17.2 and wandb 0.30.0 pull in large dependency trees (packaging, Phase 11).
- **Statistics:** Cohen (1992) Table 2 and Benjamini and Hochberg (1995) were read from scans and a public copy (not
  stored); Newcombe (1998) Table II could not be obtained, so Newcombe's interval is validated by simulated coverage
  only; `--reproduce` of a `--stats` result made without a project asks for one.
- **Campaign:** `--reproduce` of a `--campaign SPEC.json` result re-runs the project's stored spec (store it first);
  the page edits the scalar settings only (level lists through a spec file); no source states LeRobot's AV1
  compression, so the `lerobot` format uses the raw size until a measured ratio is entered; conditions are not
  balanced across days and days are filled greedily.
- **Sim-to-real:** depth is compared only where a reader decodes depth (DexVantage's readers list depth streams but do
  not decode them); no FID (torch is not installed); with blur and a brightness gap together the mount estimate
  degrades (6.7 mm and 1.31° against a true 3 mm and 1°); one episode per run, a constant latency, no silhouette or
  mount for cameras on the robot.
- **Paper kit:** not compiled on this machine (no LaTeX engine; the compile test runs where one is installed); its
  figures follow the report's naming rule, and the window does not hand its page charts to the kit; experiment
  runs have no headline figure in the runs table.
- **Cluster:** Slurm only; the design agent is not distributed; the Run on… dialog writes the job folder but does
  not call `sbatch` (`--submit` does); `--sample` makes new sensor ids per process (save the sample first).
- **Cells:** the gap compares designs (an unapplied hand-eye result shows only in the drift); its weights, scales and
  bands are uncalibrated defaults; pooling reads the feature schema, not the data, and LeRobot's own loader is not
  run; drift uses the latest calibration per copy (no Kalman trend).
- **Robot import and export:** one articulation per USD file; loops, drives, mimic and tendon couplings, effort and
  velocity limits are not imported; joint limits round to UsdPhysics' float32; the bundled newton-assets UR5e lacks
  its 6.3 MB visual mesh layer, so the wizard's preview of it shows collision capsules and frames it too tight (its
  bounds come from the link origins); pxr prints its unresolved-layer warnings, with full paths, on the console; the
  URDF's sensor links carry a placeholder inertial; the Policies page cannot set the OpenVLA action space.
- **Prices:** `rig.cost` counts the enabled sensors only (decision 7); Arducam, Basler, Neewer, Meta Quest and
  Teledyne FLIR prices could not be read (bot checks, refusals, rate limits); a store can show another price in
  another region; `library.prices_sound` fails a price read more than two years before the run (from about
  2028-11).
- **UI:** the Tasks page's first Apply on ALOHA 2 costs 110–119 ms of UI-thread CPU at the end of a full test run
  (29 ms repeated, within budget in a fresh process; cause not found: see the fixes above); a page taken by name
  (`window.pages[pid]`) follows every signal from then on in that window (no production path does); the Optimise, Tolerance and (Panda) Teleop Applies after a full run were not timed by the review
  round; the canonical view's advice reads "Wrist D405, left (sample placement)'s image".
- **From the four reviews and the final gate (open):**
  - `/tmp` is a 15 GB tmpfs with a per-user quota: old `/tmp/dexvantage-*` folders (about 12 250 when this gate
    started, 52 by its end, cleared by other work), `/tmp/pytest-of-<user>` (2.6 GB) and other work's files fill it,
    and a full run with the default temporary folder then fails in bulk ("Disk quota exceeded"). Clean `/tmp` when
    nothing runs, or run with `TMPDIR` and `--basetemp` on another volume (as this gate did).
  - `rig.run(study, **settings)` ignores setting names a study does not know (`dv.cluster.tolerance(rig, draws=8)`
    ran the default 64 draws; the key is `n`): there is no per-study list of setting names to check against.
  - Python's spawn rule: each pool worker re-runs a script's unguarded top-level code; the main-guard check catches
    only a pool started from such code (the example uses `jobs=1` as a script, 2 in a notebook).
  - Plugin checks join a validation run only where plugins are loaded (the app, batch mode and the API; not
    `python -m dexvantage.validation`); the paper kit's validation summary names the built-in models only.
  - No assistant or MCP reading tool for project diffs (it would change the 39 tools `test_integration_rt2.py`
    asserts).
  - `core/figures.py` says "too small for its rows" when long bar labels, not the row count, leave no room for the
    bars (`dv.figure` now cuts labels as the paper kit does; the message is unchanged).
  - `validation/suite.environment()` imports PyQt6 for the Qt version (not on the API path).
  - A change of a percentage is written "−0.182 %" in the statistics text, the tracking guide and the UI where it
    is a change of percentage points.
  - The citation key `core/cite.py` gives a shared datasheet is the first entry's name: the D435 is cited as
    `intel-realsense-d405` (the same D400 Series datasheet, so the content is right; Phase 9 code).
  - The tracker's `mlruns/` experiment `meta.yaml` leaves out `effective_trace_archival_retention`,
    `trace_location` and `workspace`, which MLflow 3.17.2 writes and its reader fills in; adding them could upset
    older readers (a call for the lead engineer).
  - Cohen (1992) Table 2 was compared from the values recorded earlier, not fetched again; every figure is within
    the documented one-per-group rounding of the review's own re-derivation.
  - Hidden pages: "partial" signals reach a hidden page at once while "finished" is held, so a page shown during a
    second run first shows the earlier run's final result until that run finishes (by design).
  - BCa has no guard when 1 − a(z0 + z) ≤ 0 (extreme acceleration; as SciPy); `compare_rates` uses 1 − confidence
    rather than alpha for its "borderline" note (cosmetic).
  - The software 3D view's paint takes 0.9–1.1 s per frame offscreen on ALOHA 2 (the GL view does not run it; a
    machine that falls back to software rendering would feel it).
  - The Slurm array was checked by running the shards by hand (`run_shard`, `--run-shard`); `sbatch` was not run
    (no Slurm here). `dv.cluster.gather` without `rig=` logs into the job folder's base project store, as `--gather`
    does.
  - The campaign spec refuses unknown keys, so a spec written by a newer version with a new key is refused with the
    key named. The sim-to-real offset rule reuses `min_reduction_dn` (1 DN, a labelled default) as the smallest
    offset a light scale cannot absorb (not calibrated).
  - Cosmetic: the cells headline reads "0.373 index (large gap)"; the Designed experiments page's "Now" column has
    no units while Low and High do; stored settings set from code are not range-checked by the Experiments page
    (the core clamps them at run time).

## Phases 9 and 10, samples, robot catalogue, larger library, research tools part 1 (done, 2026-11-07)

Built in parallel by separate agents (tolerances; experiments; the optimiser and design agent; requirements and
metrics; exporters A and B; the canonical warp and Rerun; the tool layer and MCP; the Assistant dock; the samples;
the robot catalogue; the camera, depth and tactile library; the systems library and teleop devices; the Python API
and plugins; diff, merge, folder form, citations and figures), then integrated, then reviewed four times
(truthfulness, correctness, a research team's workflows, spec completeness).

**Final gate (after the four reviews, 2026-11-07)**
- Tests: **3 477 passed**, 0 failed, 1 skipped (the plugin template's real `pip install`: setuptools is not in
  `.venv`) (`python -m pytest tests`, offscreen, memory-capped at 7 GB, run alone on the machine, 1 600 s). Peak:
  the largest process 5.11 GB (`time -v` maximum resident set), the cgroup's `memory.peak` 6.19 GB with page cache.
  The reviews added test_review_phase910_correctness 17, test_review_p910_product 19, test_spec_completeness 14 and
  asserts in existing files (3 478 collected; the integration gate ended at 3 427). The reviewers' failures
  (`test_metric_defs::test_groundtruth_agreement` "Disk quota exceeded", a 136 ms Apply under three parallel suites,
  `test_robot_catalogue` edited during a run) did not recur. After the gate's own fixes (below) the suite was run
  again: **3 482 passed**, 0 failed, 1 skipped (the five new tests of `tests/test_final_gate_p910.py`), 1 585 s,
  largest process 5.08 GB, cgroup 6.36 GB; its temporary folder held only the icon cache and pytest's own folder
  afterwards.
- Validation: **179 of 179 checks pass** (`--validate out/validation`, full run 500 s with two batch streams running
  alongside, peak 5.35 GB). New since the integration gate: `workspace.object_extent` (0.861 of the tessellation
  bound, tolerance 1) and `workspace.inside_classification` (0 of 30 objects misplaced).
- `dexvantage --self-test`: **37 of 37 ok** (38 s, peak 3.6 GB; two lines more than at the integration gate, one of
  them `workspace`; the validation line runs 124 cheap checks in 27 s; occlusion and grasp on the RTX 5080, the
  OpenGL line on the default display).
- `ruff check dexvantage tests scripts examples`: clean.
- `python -m dexvantage.mcp` over stdio: initialize answers protocol 2025-11-25 (server "dexvantage" 0.1.0.dev0),
  tools/list 30 tools, `open_project` with `sample: g1` opens it (0.7 s in all).
- Batch, every Phase 9/10 flag on the five arm samples (so100, ur5e, aloha, panda-bimanual, g1), 147 runs, each under
  a 4 GB cap (largest 2.3 GB), two sample streams in parallel (times include start-up and that contention); results
  exit 0, refusals one sentence and exit 2, a failing scorecard or plan exit 1, no traceback in any run:
  - `--canonical-view`: SO-100 323 px, UR5e 675 px ("large parallax"), ALOHA 0 px ("aligned": its two wrist cameras
    sit alike on their grippers), Panda 11 878 px, G1 refused (one camera). `--reproduce` of each: "reproduces"
    (ALOHA 2 409 numbers the same).
  - `--rig-inventory`, `--scorecard` ("no requirements"), `--variants-compare`: under 1.5 s each.
  - `--check defaults`: SO-100 1/5 (2 not evaluated), UR5e 1/5, ALOHA 2/5, Panda 2/5, G1 1/5 (contact visibility 0 %),
    all "not ready", exit 1, 10–78 s; `--reproduce` of the scorecards with `--reproduce-out`: "reproduces".
  - `--experiment DESIGN.json` (6-run Latin hypercube over a camera's pitch and height): UR5e 97.1 % against 91.9 %
    ("better than now"), SO-100 96.7 % against 94.5 %, G1 98.9 % against 96.6 %, ALOHA and Panda 100 % ("same as
    now"), 1.6–12.7 s; `--experiment-saved` refused on each ("The experiment has no factor yet …").
  - `--optimise` (budget 12) 6–38 s and `--design-agent` (budget 8) 13–38 s on all five: 0 to 5 designs each; a
    design at least as good on every objective (status good) from the optimiser on the SO-100 and UR5e and the agent
    on the SO-100 and G1, trade-offs only on ALOHA and Panda, and on the G1 the optimiser kept the current rig.
  - `--tolerance-mc` (64 draws): SO-100 24 s (coverage P5 94.3 %), UR5e 107 s (worst smear P95 38.6 px), ALOHA 191 s
    (66.6 px), Panda 210 s (102.5 px), G1 58 s (coverage P5 95.9 %).
  - Spec aliases: `--cable-flex` 0.7–2.6 s, `--sync` 0.6–1.4 s, `--mujoco-validate DIR/` (writes `groundtruth.json`
    in the folder) 6–17 s (UR5e 92.8 % "agrees", ALOHA 87.1 % "mostly agrees"), `--grasp-study` UR5e 32 s and ALOHA
    50 s (100 %), `--teleop-overlap` SO-100 7.4 s (96.7 %) and Panda 138 s (100 %, "clear"); `--handeye-solve
    --captures DIR --joints FILE` on a simulated UR5e capture 3.1 s (10 of 10 views, the mount found 2.680 mm off
    against a true 2.693 mm).
  - `--export` with eleven formats: 10 of 11 written on each sample (brackets refused: no bracket type set; on the G1
    also the canonical warp), SO-100 28 s to Panda 211 s; `--formats usd` 0.6–1.5 s; `--export … fiftyone,
    lerobot_dataset --dataset` on an SO-100 campaign 18 s; `--rerun` 0.7–1.8 s; `--rerun --rerun-audit` 0.9 s.
  - `--unpack`, `--pack`, `--diff` of the round trip ("No differences.", exit 0), `--diff` against the UR5e with
    `--diff-html` and `--diff-metrics` (exit 1 where they differ), `--merge`, `--textconv`, `--cite
    --cite-references all` (203 entries on the UR5e), `--run-plan` (exit 1: its check fails): each under 1 s except
    the plan (10–67 s).
  - `--figure … --figure-width double --figure-palette chart`, `--list-plugins`, `--no-plugins`: 0.2–0.4 s.
  - Refused in one sentence (exit 2): `--captures` or `--joints` without `--handeye-solve`, `--rerun-audit` without
    `--rerun`, `--optimise-settings '[1]'`, `--export-options` naming a format `--formats` does not ask for.
- Screenshots: `python scripts/screens.py --only phase910` (28 images, 316 s, none unfinished; three new:
  `p910-ur5e-workspace-page.png`, `p910-library-panel-teleop-policies.png`, `p910-sample-g1-coverage-page.png`).
  Looked at: the Workspace page (1 volume, "ready", the volumes, objects, surfaces and lights listed: fine); the
  Library's Teleop devices group (13; long GELLO name elided, full name in the tooltip); the G1's coverage page (0.0 %
  seen by two or more, "mostly single or blind", advice to add a second camera: honest); the canonical view page with
  its new depth-range field (675.1 px: fine); the Panda sample's main window (2 robots, 18 joints, 3 sensors: fine);
  the requirements scorecard (1 / 5, "not ready"): the read-only "DexVantage default" box drew no tick for a default
  requirement (fixed, below).
- Hygiene: every tracked and untracked file (941) grepped for personal paths, user and host names, e-mail addresses,
  keys (`sk-ant-`, Groq, Hugging Face, AWS, GitHub, private-key blocks) and AI attribution lines (Co-Authored-By,
  Claude-Session, "Generated with"): none (the hits are the rules, and the MCP docs naming Claude Code and Claude
  Desktop as clients). No file over 5 MB was added (the largest new file is ALOHA's
  `vx300s_7_gripper_wrist_mount.stl`, 2.3 MB). `resources/robots` gained **26.1 MB** in 140 files (aloha 17 files,
  franka_emika_panda 69, unitree_g1 51 and three scene files); the folder is 41.9 MB in all, and each of its seven
  model folders carries its LICENSE.

**Final gate fixes**
- Temporary folders: `make_app(settings_scope=…)` (the self-test, the tests) and `tests/conftest.py` (the session's
  XDG folders, one settings folder per test) never removed their folders, and the audit check's fallback folder and
  the plugin template's test folders neither: two days of runs left 12 195 folders (4.1 GB) in the 15 GB `/tmp`,
  which ran out during a review's full suite ("Disk quota exceeded" in a MuJoCo render). Each now goes when its
  process or test ends (`app.py`, `tests/conftest.py`, `validation/checks_audit.py`,
  `examples/plugin-template/tests/conftest.py`; `tests/test_final_gate_p910.py`).
- The assistant's and MCP's `export_files` and plan steps refused the `usd` alias that `export()` and `--formats`
  take (`assistant/tools.py` resolves aliases).
- A ticked checkbox that is disabled drew an empty box in both themes (the `:disabled` rule cleared the fill the
  inverted tick needs): the Requirements page's read-only "DexVantage default" read as unticked for a default
  (`ui/theme.py`).
- `scripts/screens.py --only phase910` also shoots the Workspace page, the Library's teleop and policy groups and a
  G1 study page.

**Review fixes (Phases 9 and 10)**
- Truthfulness (every source re-read on 2026-11-07 as PDFs, raw GitHub files at the pinned commits and product
  pages, about 50 library entries and all seven bundled robots' LICENSE files; 8 fixes): among them the Allied Vision
  Alvium 1800 C-158c and U-240c maximum frame rates cannot be RGB8 rates over their links, so the stream's format and
  bits per pixel are null; Diffusion Policy (CNN)'s 0.1 s is Section 3.4's figure for 10 DDIM iterations where Table 7
  lists 16 (the note says it may understate); the Franka Research 3 and Flexiv model folders' LICENSE files hold the
  Apache-2.0 text followed by a BSD text (the entries and the catalogue test say so); the Isaac export's and the
  OpenVLA export's notes state their limits (USD aperture offsets, joint encodings).
- Correctness (8 bugs, each with a regression test in `tests/test_review_phase910_correctness.py`): a `json:`
  factor pointer could move, re-model, delete or unlock a locked sensor (`core/factors.apply` refuses unless
  `allow_locked`); the experiments and the optimiser took such a pointer as a factor, and the optimiser's
  refused-design path returned a constraint vector of the wrong length; the tolerance Monte Carlo counted a draw
  stated only as a bound ("at least 1 ms") as meeting "< 2 ms" and compared without the scorecard's 10⁻⁹ tolerance
  (`core/tolerance.meets` uses `core/requirements.decide`); the optimiser counted such a bound as a met constraint;
  NSGA-II's crowding distance gave +inf to a design whose figure is not stated; `--reproduce` of a scorecard made for
  a non-active variant re-ran the active one; the Plücker "base" frame defaulted to the first robot instead of the
  primary one. Conventions pinned: the tolerance draw (body turn, parent shift), the calibration covariance's
  parameterisation, URDF rpy (fixed-axis xyz), the canonical homography's direction, the Kalibr chain and USD camera
  axes, merge conflicts on remove against change, the unpacked folder's exact round trip.
- A research team's workflows (18 fixes, `tests/test_review_p910_product.py`): `factors.apply` refuses a value that
  is not a finite number, an unknown library device and a stream that is not [width, height] in a sentence (a raw
  ValueError, NaN or a model id that does not exist before); `dv.open` and `dv.sample` raise `DexVantageError`;
  `rig.save` to a folder writes the unpacked form; `--reproduce` of a file naming no study and `--optimise-settings
  '[1]'` say so; a bound that does not settle a requirement shows no margin; a small Saltelli design says its indices
  are rough; the MCP server and the assistant open the three new samples, and the tolerance Monte Carlo, experiments,
  the optimiser and the design agent are long work for the assistant (an approved plan runs them); the Panda
  sample's Diffusion Policy takes the two wrist views ("3 of 2 cameras assigned" before); the Menagerie import
  compiles on the dialog's worker and keeps the catalogue's robot name; opening an analysis page makes the Analysis
  dock tall enough to read; empty selectors say why; "No warnings." (not "None."); a user-set mount tolerance is not
  labelled "(default)"; "1 sensor" (not "1 sensors"); the assistant's overview counts moving joints as the Outliner
  does (28 on the G1, not 44); the brackets export's refusal names the menu.
- Spec completeness (§7.11, §7.16, §8.5, §9.2, §9.4, §11, §12, §13; `tests/test_spec_completeness.py` 14):
  §11's names as aliases (`--grasp-study`, `--cable-flex`, `--mujoco-validate out/`; a study given a folder writes
  `<folder>/<flag>.json`), `--formats usd` (the Isaac export), `--export … fiftyone --dataset`, `--handeye-solve OUT
  --captures DIR [--joints FILE]` (§8.4's batch capture); a Workspace page with a workspace check and its suggestion
  (`core/workspace_check.py`, validation model `workspace`, 2 checks); design-log notes and a filter; the Canonical
  view page's depth-range field; the Library panel's Teleop devices, Policies, Lights, Cables and Mounts groups; the
  Requirements page shows another variant beside the active one; variant ghosts in the 3D view.

**Integration gate (2026-11-07)**
- Tests: **3 426 passed**, 0 failed, 1 skipped (the real `pip install` of the plugin template: setuptools is not in
  `.venv`) (`python -m pytest tests`, offscreen, memory-capped at 7 GB, 1 505 s). The suite's process peaked at
  **4.85 GB** (VmHWM; the largest process of the run 4.99 GB), its RSS 3.0 GB at the end; before the memory fixes
  the same suite passed 7.5 GB a third of the way through. New test files: test_tolerance 17, test_experiments,
  test_optimise,
  test_optimise_page, test_metric_defs, test_requirements, test_requirements_batch, test_requirements_page,
  test_metrics_contract, test_factors_contract, test_exporters_contract, test_exporters_a 25,
  test_export_isaac_lerobot_report, test_canonical_warp, test_rerun_bridge, test_assistant_tools, test_mcp,
  test_assistant_loop, test_assistant_dock, test_samples_bimanual, test_robot_catalogue, test_library_added_entries,
  test_library_depth_entries, test_library_systems 143, test_api, test_plugins, test_examples, test_projdiff,
  test_projfolder, test_cite, test_figures, test_compare_ui, test_integration_p910 6 (3 427 collected; the Phases 4,
  5 and 8 gate ended at 1 886).
- **Test memory (step 0 of this gate):** a diagnostic run (11 GB cap) showed `test_validation`'s in-process quick run
  of every check peaking at 6.2 GB and leaving 3.3 GB resident, and the suite then creeping by about 50 MB per module
  (7.5 GB peak by `test_grasp_visibility`, a third of the way). Per-module object counts found nothing alive (no
  window, robot model or project survived a module); the creep was glibc's per-thread malloc arenas keeping freed
  worker-thread memory mapped: with two arenas seven modules ended at 0.79 GB instead of 1.64 GB. Fixes:
  `core/memlimit.py` (`mallopt(M_ARENA_MAX, 2)`, called by `tests/conftest.py` and by `app.main`), and
  `test_validation` runs each model's quick run in a child process (the heavy `formats` 4.9 GB, `handeye` 3.7 GB and
  `groundtruth` 2.4 GB alone, the others two at a time; the suite's own process stays at about 0.25 GB while they run).
  `scripts/run_tests.py` was not needed.
- Validation: **177 of 177 checks pass** (`--validate out/validation`, full run 400 s, peak 5.4 GB with two malloc
  arenas; 410 s and 6.7 GB before that fix, with the test suite running alongside). 64 new since Phases 4, 5
  and 8: robots 1, library 6 (data consistency), tolerance 5,
  experiments 6, optimise 6, formats 11 more (MJCF snippets, URDF, tf, OpenCV YAML, Kalibr, masks, STEP, Isaac USD and
  configs, LeRobot export, report, canonical warp files, Rerun read-back), vla 4 more (warp), assistant 3,
  requirements 5, copilot 3, versioning 7, samples 5 (each model's figures are in its document and in the overview).
  The quick run of every model in its own process: 177 of 177, 197 s in all.
- `dexvantage --self-test`: **35 of 35 ok** (36 s). New lines: samples, robot catalogue, library teleop, tolerance,
  experiments,
  optimiser, requirements, versioning, mcp server, assistant, exporters, sim exports, warp, python api (each under
  1 s); the validation line runs 122 cheap checks (26 s).
- `ruff check dexvantage tests scripts`: clean.
- `python -m dexvantage.mcp`: answers initialize (protocol 2025-11-25, or the client's version when it is one of
  those served) and tools/list (30 tools).
- Batch, every new flag on the samples (4 GB cap, times include start-up; RTX 5080 through EGL): `--sample ur5e
  --pack` 0.7 s, `--unpack` 0.2 s and `--pack` back 0.4 s (their diff: "No differences.", exit 0), `--textconv`
  0.2 s, `--diff` UR5e against SO-100 0.4 s (65 changes, exit 1 as they differ, with `--diff-html`), `--cite` 0.4 s,
  `--figure` 0.2 s; `--tolerance-mc` UR5e 81 s (64 draws: worst motion smear P95 38.6 px, "no requirement"), SO-100
  24 s (coverage P5 94.3 %); `--reproduce` of the UR5e tolerance result 94 s ("Reproduces exactly: all 1141 numbers");
  `--experiment` (a 6-run Latin hypercube over the Overhead D435's pitch and height) 9.2 s (92.0 %, "better than
  now"); `--optimise` (budget 12) 17 s (3 designs, "better rig found"); `--design-agent` (budget 8) 26 s (3 designs,
  "trade-offs only"); `--check defaults` UR5e 36 s (1 pass, 3 fail, 1 not evaluated, exit 1) and its `--reproduce`
  35 s (exit 0); `--run-plan` (a study, a check, an export) 35 s (exit 1: the check fails); `--export` with every
  format but fiftyone and lerobot_dataset on the UR5e 76 s (exit 1: brackets refused, no bracket type set; the other
  10 written), SO-100 seven formats 1.8 s; `--canonical-view` 0.9 s (675 px, "large parallax"); `--rerun` 1.3 s;
  `--scorecard`, `--variants-compare`, `--rig-inventory` under 1 s; `--sample aloha --coverage` 2.2 s (100 % seen,
  99.2 % by two or more cameras), `panda-bimanual` 2.5 s (100 %), `g1` 2.2 s (96.6 %); `--sample aloha --grasp` 45 s
  (100 %); `--sample panda-bimanual --teleop` 139 s (100 % teleoperable, "clear"); `--sample g1 --depth-validity`
  1.2 s; `--list-plugins` 0.2 s.
- UI-thread budget (`scripts/measure_ui_phase910.py`, offscreen, measured while the full suite ran alongside): **0
  steps over 100 ms**. Largest: the tolerance page's Apply 48.5 ms, the canonical view's Apply 40.6 ms, the
  experiments and optimise Applies 38.7 and 38.3 ms, File ▸ Export… opening 22.3 ms; every page shows within 17 ms
  and every result arrives within 1 ms. Workers: tolerance (24 draws) 31.3 s, experiments (128 runs) 18.6 s, optimise
  (24 designs) 23.0 s. Building the main window: UR5e 745 ms, ALOHA 427 ms, Panda 410 ms, G1 432 ms (File ▸ Open
  sample now builds the project on a window job; the window's own build is the "startup" row, as before).
- Screenshots: `python scripts/screens.py --only phase910` → `out/screens/p910-*` (25 images, none unfinished).
  Looked at and fixed: the requirements scorecard and variants compare shots showed an empty list (the sample has no
  requirements: the script now adds the labelled defaults; the page then reads 1 / 5, "not ready"); the variants
  compare said "from before the last edit" right after a compare (one stale flag served the scorecard and the
  compare: now one each); the Sobol bars at N = 8 had first-order indices above the totals with intervals spanning the
  axis (the estimator is right, the design too small: the shot uses N = 32, 128 runs); the optimiser's scatter had
  unround ticks (15.1, 37.9 …) and a percentage axis to 106 (round ticks, a share axis starts at 0 and has no tick
  above 100); the tolerance chart cut a long figure name without an ellipsis; the Library's Robots group showed no
  count (53 now); the export dialog shot showed a home folder (a temporary folder now).

**Integration fixes**
- The brackets study crashed (TypeError in `core/brackets.box_of`) for a sensor with a bracket type whose library
  device states no size (the NanEyeC, the Leopard GMSL2 Owl): refused in one sentence now
  (`tests/test_integration_p910.py`).
- Four tolerance checks declared no unit (`test_validation` caught it): relative, KS statistic D, m or rad, rank
  correlation.
- The Export dialog: the report reuses the results the window already computed for this revision of the design and,
  by default ("Also run the studies not computed yet"), runs the others; the Isaac export is posed at the on-screen
  joint state like the other posed formats.
- File ▸ Open sample builds the project on a window job (ALOHA 2 took about 2.5 s on the UI thread the first time).
- `test_review_phase458_product::test_recording_of_another_robot_is_compared_not_crashed` failed in two full runs in
  a row (and once in a run of the modules p to r): the frame-threaded AV1 decoder refused a packet ("Invalid data
  found when processing input") of a file that decodes cleanly on its own (the same bytes, MD5-identical, from passing
  runs). The LeRobot reader now decodes such a pass again single-threaded from the frame it reached; a file that
  fails both ways is still refused (`core/datasets/lerobot.py`, test in `tests/test_integration_p910.py`). The
  threaded failure itself was not reproduced outside the suite (18 audits in a loop: 0).
- `tests/test_plugins.py`'s registry fixture restored the page registry to a snapshot taken before the pages loaded,
  so a window built after it (in that file order) had no audit, plan or silhouette page: it loads the pages first.
- `tests/test_library_added_entries.py` asserted the exact set of library checks (another agent added three).
- Metrics `canonical_view.parallax_p95` and `canonical_view.coverage` (53 metrics now); `rig_inventory.stated_price`
  public (the optimiser's cost objective used a private function).
- `validation/suite.py`: the `experiments` model retitled "Designed experiments (factorial, Latin hypercube, Sobol)"
  (the optimiser has its own model); overview anchors added for every new model id (requirements, tolerance,
  experiments, optimise, assistant, copilot, samples, catalogue, versioning).
- `tests/test_gui.py`: no placeholder command is left (asserted); the validation job test and
  `test_batch.py::test_validate_exit_code_and_report` validate cheap models (frames, optics, library): the formats
  model's quick run alone now takes 30–100 s.
- The self-test's name column widened to 15 characters ("robot catalogue", "library teleop").
- `pyproject.toml`: the unpinned `claude` and `keyring` extras removed (the pinned `assistant` extra replaces them);
  the old placeholder `AssistantPanel` removed from `ui/panels.py`.
- `batch.py`'s docstring lists the new flags; `docs/project-format.md` documents the `.dexv.d` folder and the Phase 9
  fields; `docs/DEVELOPING.md` the new extension points (metrics, factors, exporters, plugins) and the test-memory
  rules.
- `scripts/screens.py --only phase910`, `scripts/measure_ui_phase910.py`.
- Docs: overview sections for every new model and the model-document table; user-guide chapters 11–14 and the batch
  lines; THIRD_PARTY (the optional packages with their licences, the Phase 9/10 sources); README; CHANGELOG.

**Shared contracts (additive changes, as reported):** `core/exporters/__init__.py` gained `unstamped` (LeRobot's
train config and info.json templates, whose readers refuse extra keys); `core/jobs.py` `ResultCache.key` gained the
active variant's index before the dataset fingerprint (anything that unpacks the key by position must know);
`core/library` gained the kinds `robot` and `teleop` and `validate_minimal`; `library.schema.json` the `robot` and
`teleop` blocks, `ingest[]` for hosts and the new camera, depth and tactile fields; `core/project.py` the unpacked
folder form (load and save of `.dexv.d`).

**Decisions (Phases 9 and 10)**
1. **A mount tolerance is a ± bound per axis** (shifts along the mount parent frame's axes, turns about the sensor's
   own x, then y, then z), drawn uniformly (a labelled default) or as a normal truncated at its 2σ bound; a sensor
   with an applied hand-eye result draws from its 6 × 6 covariance, unless it moved since (then its mount tolerance).
2. **The tolerance study's default metrics** are those under 1 s per draw (coverage, depth, motion blur, VLA tokens);
   slow ones (the grasp study, about 30 s per draw) only when a requirement or the settings name them.
3. **NSGA-II written from the paper** (Vantage's optimiser is single-objective); cost is an objective only when every
   candidate device states a price (none does); the grasp study is scored on 40 grasps during the search (labelled
   screening defaults), `confirm` re-scores the shown designs.
4. **Requirements have no priorities** (Vantage's must/should were stripped): every requirement decides the verdict;
   a bound (≥, ≤) decides only in its direction, otherwise "not evaluated".
5. **Experiments run serially by default**; a process pool on request; a run a cancel interrupts is discarded.
6. **Exports are posed at one joint state** (home, or the pose on screen); URDF effort and velocity are written as 0
   where the robot's files do not state them, and the notes say so.
7. **The canonical warp's contact frame** is measured from each gripper's pads (TCP, approach z, closing x with a ±1
   user sign); the depth range is a bound from the task volume, not a measurement.
8. **The assistant never computes a figure:** tools return study results, project fields or library entries; the
   dock flags any number it cannot trace; keys live only in the system keyring; long studies need an approved plan
   (the MCP bridge allows them, as the MCP client asks the user per call).
9. **ALOHA 2's arms are two robots** (per-gripper studies take one gripper per robot); the G1 is its upper body with
   the legs held at `stand` and meshes simplified within 0.5 mm.
10. **Catalogue and library gaps stay null:** 22 catalogue robots have no maker figure; no desktop GPU is a host
    entry (their figures are not camera-ingest figures).
11. **The test suite's memory:** two malloc arenas and the validation quick run in child processes (above), not
    separate process batches.

**Spec corrections (Phases 9 and 10)**
1. **§11 `--tolerance`** is `--tolerance-mc OUT.json`: `--tolerance` is `--reproduce`'s comparison tolerance.
2. **§11 `--experiment`** takes a design file (`dexvantage.experiment/1`); the experiment saved in the project runs
   with `--experiment-saved OUT.json`.
3. **§9.2 Isaac Lab:** in v3.0.0 `TiledCameraCfg` is deprecated ("Use CameraCfg directly") and OffsetCfg's rotation
   is (x, y, z, w); v2.3.2 uses `TiledCameraCfg` and (w, x, y, z): both files are written.
4. **§9.2 LeRobot train config:** it has no field for the action-chunk study's delay offset; it is written as
   `delta_timestamps.json` and a `make_dataset.py` instead.
5. **§9.2 Kalibr and Rerun** state no pixel-centre convention; OpenCV's is used and said.
6. **§12 MCP tool names:** 30 tools, named per study (`get_teleop_overlap`, `get_sync`, `get_cable_flex` …) plus
   reads, metrics, edits, plans and exports; MCP specification 2025-11-25.

## Phases 4, 5 and 8: MuJoCo ground truth; teleop, lighting, blur and sync; data (done, 2026-11-07)

Built in parallel by separate agents (ground truth; synthetic episodes; LeRobot read and write; bags and RLDS; teleop;
lighting; motion blur and time sync; action-chunk latency and the tactile handoff; the auditor, plan against recording
and FiftyOne; silhouette calibration; the data links), then integrated, then reviewed four times (truthfulness,
correctness, a researcher's workflows, spec completeness).

**Final gate (after the four reviews, 2026-11-07)**
- Tests: **1 886 passed**, 0 failed, 0 skipped (`python -m pytest tests`, offscreen, memory-capped at 7 GB, 1 054 s;
  process peak 6.55 GB, the cgroup's `memory.peak` 7.11 GB with page cache). The reviews added test_spec_review_p458 13,
  test_review_phase458_product 11, test_review_phase458_correctness 8, test_truthfulness_review_p458 6 and regression
  asserts in existing files (1 886 collected; the integration gate ended at 1 841). At first the suite was killed by
  its 7 GB cap at 95 %: `test_validation`'s in-process quick run of every check grows the process by about 2.7 GB for
  a while, and at the end of the alphabetical order it came on top of the suite's creep (4.7 GB): peak 7.4 GB,
  measured under a 12 GB cap, where all 1 886 passed (1 087 s). `tests/conftest.py` now runs that one test first
  (`RUN_FIRST`; alone it peaks at 4.45 GB). The two tests the reviews saw fail only under load
  (`test_review_phase458_product::test_recording_of_another_robot_is_compared_not_crashed`, the `test_sensors_ui`
  drag size) passed in both full runs.
- Validation: **113 of 113 checks pass** (`--validate out/validation`, full run 303 s with the test suite running
  alongside, peak 4.7 GB, renders on the NVIDIA GeForce RTX 5080 through EGL device 0). New since the integration:
  `formats.mjcf_scene` (the saved MuJoCo scene loaded by MuJoCo on its own, 5.8 × 10⁻⁷ m against 2 × 10⁻⁶ m).
  Changed by the reviews: `lighting.white_furnace` by adaptive quadrature 8.1 × 10⁻¹² (10⁻⁶; the Monte Carlo
  estimate, 0.0065 from 1, must lie inside its 4 standard-error band, 0.024); `audit.plan_vs_recording` compares
  rotation vectors, 0.204° (0.3°). With the UR5e's trajectory frames `groundtruth.ur5e_contacts` measures 1 − 0.928 =
  0.072 (0.15). Every other figure as at the integration gate.
- `dexvantage --self-test`: **21 of 21 ok** (25 s; its validation line 77 of 77 cheap checks in 18.9 s; the occlusion
  and grasp lines on the RTX 5080, the OpenGL line on the default display, the integrated Radeon).
- `ruff check dexvantage tests scripts`: clean.
- Batch, every Phase 4/5/8 flag (with the spec aliases) on the three samples, 4 GB cap, RTX 5080 through EGL, times
  include start-up; results exit 0, refusals one sentence and exit 2:
  - `--groundtruth`: SO-100 4.5 s (84.2 %, "mostly agrees"), UR5e 8.9 s (92.8 % of 20 992 points over 41 poses,
    "agrees"), dexhand 1.8 s (depth only, 99.6 %; head camera surfaces 91.4 %, 97.7 % inside the task volume).
  - `--teleop` = `--teleop-overlap`: SO-100 7.8 s (96.7 %, "clear"), UR5e 10.7 s and dexhand 5.4 s ("no device").
  - `--lighting`: 0.4–0.7 s, "no lights" on every sample.
  - `--motion-blur`: SO-100 0.5 s ("exposure not set"), UR5e 0.9 s (up to 38.04 px, "may be blurred"), dexhand
    0.7 s ("no motion").
  - `--time-sync` = `--sync`: SO-100 0.5 s ("incomplete"), UR5e 0.8 s and dexhand 0.8 s (≥ 54.4 ms, "out of sync").
  - `--action-chunk`: UR5e 0.6 s (π0, 0.0 mm, "on contact"); SO-100 refuses (SmolVLA's inference time not stated),
    dexhand refuses (no policy).
  - `--synth-episodes`: SO-100 11.2 s (4 episodes, every fault on), a second SO-100 campaign with the slip from
    episode 1 8.1 s (3 episodes), UR5e 8.9 s (2 episodes); dexhand refuses (no trajectory, no grasp set); a folder
    that is not empty is refused.
  - On those campaigns (`--dataset`): `--audit` SO-100 9.1 s (0 of 4 pass, "not usable"; with `--audit-html` and
    `--fiftyone`, 8 samples), UR5e 14.5 s (0 of 2, "not usable"); `--plan-vs-recording` SO-100 9.7 s (7 of 10,
    "differs"), UR5e 15.8 s (10 of 12, "mostly matches"), the UR5e plan against the SO-100 recording 2.5 s (0 of 4,
    "does not match", no crash); `--silhouette-calib` SO-100 18.8 s (2.42 mm, 1.64° ± 0.72 mm, 0.11°; truth 3 mm,
    1.5°), the second campaign 13.7 s (4.06 mm, 1.65°), UR5e 10.1 s (3.30 mm, 1.39°). Refused in one sentence:
    `--audit` without `--dataset`, a dataset path with nothing there, `--audit-html` without `--audit`.
- Screenshots: `python scripts/screens.py --only phase458` (25 images, 112 s, none unfinished). Looked at:
  `p458-so100-audit-page.png` (0 / 4, "not usable"; the requirement table read "1 gaps": now "1 gap"),
  `p458-ur5e-groundtruth-page.png` (92.8 %, "agrees", the confusion matrix and why they disagree: fine),
  `p458-ur5e-motion-blur-page.png` (the headline said "may be blurred" but the trajectory's segment table said
  "blurred" in red for the same upper bounds: fixed), `p458-ur5e-action-chunk-page.png` (0.0 mm, "on contact", the
  loop bars "at least 120 ms": fine).
- Hygiene: every tracked and untracked file (479) grepped for personal paths, user and host names, e-mail addresses,
  keys (`sk-ant-`, `sk-`, AWS, GitHub, Slack, Google, private-key blocks) and AI attribution lines: none (the only hits
  are the rules and the tests that assert their absence). No file over 5 MB was added (the largest new file is
  `core/rosbag.py`, 87 kB).

**Review fixes (Phases 4, 5 and 8)**
- Truthfulness (every figure re-read at its source on 2026-11-07: the Elgato pages, Intel's D400 datasheet rev 023,
  refractiveindex.info, ISO 12232's relative rule, the GELLO and LeRobot quotes): the three Elgato lights' masses were
  null although their pages state them (1.3 kg, 735 g, 1.07 kg); the projector power is said to be the laser maximum
  (Table 4-23, 30 mW steps); the ground-truth advice called voxels "surface pixels"; the motion-blur word over an
  upper bound is "may be blurred"; the white furnace check by quadrature instead of a Monte Carlo band 3.7 times its
  measured error; the plan-against-recording check compared the sizes of two rotations (0.020°) instead of the
  vectors (0.20°); the audit docs' "±12 ms" desync is the measured 16.9 ms.
- Correctness (one bug, conventions pinned): the audit, plan-against-recording and silhouette studies read a dataset
  the project names only by its path, so a dataset re-recorded at the same path hit the result cache (the silhouette
  page passed no fingerprint at all); the cache key now carries the dataset's fingerprint (`jobs.DATASET_STUDIES`) and
  follows lights, teleop devices, the rig's sync block and the data section. Pinned against independent references:
  the lens remap direction, LeRobot frame seeking (v2.1, v3.0, across episodes and drops), CDR alignment and byte
  order, CRC-32C (RFC 3720), ACT's ensembling order, the blind gap.
- Workflows (eleven problems, on the SO-100 and UR5e samples and bad inputs): a `StudyError` subclass reached the page
  as "stopped with an unexpected error (DatasetError)"; damaged datasets (a broken `meta/info.json`, Parquet file or
  bag file) are explained, an empty bag audited as "ready" and is now refused, one damaged episode is dropped instead
  of stopping the audit; a recording of another robot crashed plan against recording; a long status message widened
  the main window to thousands of pixels; `--dataset` alone opened the window (now a batch refusal); an unexpected
  batch error is one sentence; the plan preview named sensors by uid; the silhouette page says when its mount is
  applied and offers Apply again after an undo; the time-sync preview message is cut (the full text in the tooltip).
- Spec completeness (eight gaps): the §11 names `--teleop-overlap` and `--sync` as aliases; `--audit-html OUT.html`
  and `--fiftyone OUT_DIR` with `--audit` (§8.2); the check `formats.mjcf_scene` (§10); plan against recording compares
  the recorded camera-joint offset with the time-sync prediction (§8.3); the Lighting page sets object materials;
  the MuJoCo scene's lights follow the lighting contract; LeRobot datasets streamed from the Hugging Face Hub
  (`core/datasets/hub.py`, `hf://datasets/<org>/<name>[@rev]`, public datasets only, files cached per commit, video by
  range requests; this replaces "Hub datasets are read only from a local snapshot"); the page and docs no longer claim
  a LeRobot training-config exporter (Phase 10).
- Final gate: the audit's "Episode readable" row counted only unreadable episodes ("0 of 1", now "2 of 3"); the
  action-chunk Apply stayed enabled when the recommendation was already stored; the motion-blur segment table said
  "blurred" over an upper bound; "1 gaps"; a dataset path that never existed was "not there any more" (now "not there
  (nothing at that path)"); `test_validation`'s in-process run moved first (above). Figures corrected in the overview,
  README, user guide and `docs/models/groundtruth.md` (the UR5e's 92.8 %, the white furnace, the plan mount, "may be
  blurred", 113 checks); the dexhand head camera's surface agreement is now explained in `groundtruth.md`.

**Integration gate (2026-11-07)**
- Tests: **1 841 passed**, 0 failed, 0 skipped (`python -m pytest tests`, offscreen, memory-capped at 7 GB, 1 043 s). New test files:
  test_groundtruth 24, test_lerobot_io 23, test_bag_reader 73, test_rlds_reader 21, test_teleop 21,
  test_lighting 21, test_motion_blur and test_time_sync 23, test_action_chunk, test_handoff (and test_dexhand)
  46, test_synth_episodes 13, test_audit 18, test_plan_vs_recording 6, test_render_calib 9, test_dataset_links 16
  plus test_validation 2 (1 841 collected; the Phases 3, 6 and 7 gate ended at 1 538).
- Validation: **112 of 112 checks pass** (`--validate out/validation`, full run 258 s, 294 s with the test suite running
  alongside; peak memory 4.8 GB; RTX 5080 through EGL device 0). 38 new: groundtruth 6, teleop 4, lighting 9, timing 7, data 4, formats 3, grasp 1
  (`grasp.dataset_closures`), hand-eye 3 (`dataset_capture`, `silhouette_recovery`, `silhouette_vs_board`), dexhand 1
  (`blind_gap_bruteforce`). Measured: rendered fiducials of six lens models 0.20 px (0.4); compiled intrinsics
  8.8 × 10⁻⁵ px (10⁻³); per-pixel owner against the Camera view's renderer 2.2 × 10⁻⁵ of 2.3 M pixels (10⁻³); UR5e
  contacts 1 − 0.913 (0.15); UR5e depth 1 − 0.998 (0.01); synthetic episodes written and read back 0 mismatches;
  teleop 2R 1.4 × 10⁻¹⁶ m², finite-difference Jacobian 3.8 × 10⁻⁹, leader workspace 0 missed (0.05), device mapping
  0.19 (1); GGX normalised 1.4 × 10⁻¹³, white furnace 0.0065 (Monte Carlo band 0.024; since the review by quadrature, 8.1 × 10⁻¹² against 10⁻⁶), energy 0, reciprocity 1.3 ×
  10⁻¹⁴, specular peak 2.2 × 10⁻¹³, Schlick 0.036 (0.04), inverse square 0.0033 (0.005), area lights 0.022 (0.025),
  visibility against ray casting 0.0066 (0.015); image velocity 2.8 × 10⁻⁹ (10⁻⁵), smear 2.4 × 10⁻¹¹, rolling shutter
  0.024 (0.05), stamp intervals 3.5 × 10⁻¹⁸ s, action-label error 2.2 × 10⁻⁴ (0.02), chunk loop 2.9 × 10⁻⁵ m
  (5 × 10⁻⁵), ensemble 3.3 × 10⁻¹³; blind gap 0 m (2 mm); audit recall floors 0 shortfall, desync 16.9 ms (20),
  slip classifier against Vantage 0 (10⁻⁹), plan mount 0.020° (0.2°; the review found this compared the sizes of the rotations: as rotation vectors 0.20° against 0.3°); LeRobot, MCAP and TFRecord round trips 0;
  dataset grasps 6.2 × 10⁻⁸ m (10⁻⁶), dataset hand-eye 0.046 mm (0.2 mm); silhouette recovery 1.66 px and against
  the board solve 1.65 px (2 px). The quick run (all 112) takes 136 s (peak 4.4 GB; it took 205 s and 6.7 GB before the integration
  fixes): the self-test now runs only its 77 cheap checks (18 s).
- `dexvantage --self-test`: **21 of 21 ok** (25 s). New lines: "teleop" (the study's IK and dexterity on a 2R arm
  against the closed form, 0.25 s), "lighting" (the lit UR5e scene: 0.0 % glare, "low glare; depth washed out",
  1.1 s), "timing" (motion blur, time sync and action chunk on the UR5e: 38.0 px, 54.4 ms, 0.0 mm; 0.35 s),
  "datasets" (LeRobot v2.1 and v3.0 and MCAP round trips, 0.9 s), "audit" (the slip classifier against Vantage, 0.3 s).
- `ruff check dexvantage tests scripts`: clean.
- Batch, every new flag (times include start-up; RTX 5080 through EGL; results exit 0, refusals one sentence and
  exit 2):
  - `--groundtruth`: SO-100 4.4 s (84.2 %, "mostly agrees"), UR5e 8.4 s (91.3 %, "agrees"), dexhand 2.4 s (depth
    only, 99.6 %).
  - `--teleop`: SO-100 7.6 s (96.7 % teleoperable, "clear"), UR5e 10.9 s and dexhand 6.9 s ("no device": the reach
    and dexterity are shown).
  - `--lighting`: 0.4–0.8 s on every sample ("no lights": the samples have none; the lit UR5e scene of the checks and
    the guide reads 0.0 % glare and 73.8 % depth lost on the Overhead D435).
  - `--motion-blur`: SO-100 0.6 s ("exposure not set": its cameras have no device), UR5e 1.0 s (up to 38.04 px,
    "blurred"; since the review "may be blurred", the smear being an upper bound), dexhand 0.7 s ("no motion").
  - `--time-sync`: SO-100 0.5 s ("incomplete"), UR5e 0.8 s (≥ 54.4 ms, "out of sync"), dexhand 0.7 s.
  - `--action-chunk`: UR5e 1.0 s (π0, 0.0 mm "on contact": the scripted pick holds still as the gripper closes);
    SO-100 refuses (SmolVLA's inference time is not stated by its source) and dexhand refuses (no policy).
  - `--synth-episodes out/synth_so100` on the SO-100: 4 episodes × 120 rows, 920 frames of 320 × 240 from 2 cameras,
    v3.0, 7.4 s (124 frames/s, RTX 5080).
  - With `--dataset out/synth_so100` (the campaign above, every fault on, a slip from episode 2): `--audit` 13.9 s
    (0 of 4 episodes pass, "not usable"), `--plan-vs-recording` 16.6 s (5 of 10 comparisons match, "differs").
    `--silhouette-calib` refuses a 4:3 recording of a 16:9 imager in one sentence; on a 16:9 campaign
    (`out/synth_so100w`, a slip from episode 1) 7.1 s: 4.2 mm and 1.45° ± 2.8 mm and 0.27°, "mount moved"
    (1 680 renders).
- **Measured MuJoCo agreement** (`--groundtruth`, 50 grasps, seed 0): contact points UR5e 91.3 % (IoU 86.3 %, 16 896
  points over 33 poses; Wrist D405 96.1 %, Overhead D435 86.5 %), SO-100 84.2 % (IoU 78.1 %); the grasp-level verdict
  the same on 32 of 32 and 44 of 44 grasps; occlusion disagreements away from an edge 0.5 % and 0.4 %; most of the
  rest within a pixel of a segmentation edge or the definition (2 mm in front of the pad against the pad face). Depth:
  voxels 99.89 % (Wrist D405), 99.66 % (Overhead D435), 99.58 % (dexhand head camera); on rendered surfaces 91.6 %
  for the overhead D435 (texel depth over the 1 mm tolerance on the far table, 99.6 % inside the task volume).
- **Render and generation throughput:** the renderer (`scripts/measure_groundtruth.py`, the UR5e's 1280 × 720 D405 and
  1920 × 1080 D435, segmentation, depth and RGB, 8 views per call) 76–99 renderer frames/s and 14.7–17.5 views/s end
  to end on the **NVIDIA GeForce RTX 5080** (EGL device 0), 68 and 14.2 views/s on the **Ryzen 9 9900X's integrated
  Radeon** (radeonsi, EGL device 2); EGL device 1 cannot open a display (said in words). Distorted lenses: Brown-
  Conrady 2.4 views/s, a cube-face fisheye 0.7 views/s. Synthetic episodes (`scripts/measure_synth.py`, every fault
  on, 2 × 640 × 360, AV1): 70.9 camera frames/s on the RTX 5080, 55.4 on the integrated Radeon (8.6 of 14.8 s outside
  rendering, mostly AV1 encoding); 545 frames/s at 64 × 48; 11.8 frames/s at stream size.
- UI-thread budget: no step over 100 ms (`scripts/measure_ui_phase458.py`, offscreen, measured while the full suite
  ran alongside): largest the events after Apply on the multi-finger page with the handoff 80.8 ms, its Apply 67.6 ms
  and Camera view overlay 70.4 ms worst; every new page's Apply under 30 ms (lighting 29.8, silhouette 27.4, teleop
  27.0, action chunk 23.6 ms); the ground-truth result arriving 13.8 ms. Worker times: ground truth 7.6 s, teleop
  7.2 s, audit 11.5 s, plan against recording 12.8 s, silhouette 5.7 s, the multi-finger study with the handoff 12.8 s.
- Screenshots: `python scripts/screens.py --only phase458` → `out/screens/p458-*` (25 images, 102 s, none unfinished). Looked at and
  fixed: the time sync status read "at least out of sync" (now "out of sync" with the figure as "≥"); the lighting
  headline said "low glare" in green while the Overhead D435 lost 73.8 % of its depth (now "low glare; depth washed
  out", critical); the teleop dead-zone list was empty without a word on the sample (now a row saying why; a second
  shot with the leader scaled 1.25 × shows a joint-limit dead zone); the handoff and action-chunk charts drew their y
  label under the title; the plan-against-recording visibility headers were cut; the ground-truth thumbnails spread
  across the page; the motion blur and time sync Camera view overlays need a frame chosen (the screenshots now pick
  the worst).

**Integration fixes**
- `validation/suite.py` MODELS: the unused `sim` entry removed (the synthetic-episode check uses `groundtruth`);
  `timing` lists the action-chunk module. Overview anchors added for every model id the report links (`groundtruth`,
  `teleop`, `lighting`, `timing`, `data`).
- Memory: the full validation run grew past 6.7 GB and the full test suite was killed by its 7 GB cap at the self-test
  (its quick validation run in a subprocess). The suite now frees the process caches (compiled robot models, cube
  maps, lens remap plans, rendered board rays, the audit campaign) between models: peak 4.8 GB. The test fixtures free
  the same caches after every module (`release_memory`). The suite's process still creeps from 0.2 to about 4.2 GB
  over its 1 841 tests (measured per module), and `test_validation`'s in-process full run (with its render worker
  processes, in the same 7 GB cgroup) was killed twice; it now runs every check in quick mode (same tolerances, fewer
  samples); the full run is the `--validate` gate above. Under a 10 GB cap the suite with the full in-process run
  passed 1 841 of 1 841 (19 min 42 s).
- Time: the quick run (every check) took 205 s; three tests run it as plumbing (`test_batch`, `test_gui`,
  `test_selftest`, limits 60–180 s). `ValidationSettings.cheap_only` selects the 77 cheap checks (the self-test's
  line, 18 s); the plumbing tests run two or three cheap models; the quick audit campaign uses 120 rows per episode
  and the quick silhouette checks shorter episodes and 10 bootstrap resamples (tolerances unchanged, measured to pass).
- The action-chunk study was registered live (`quick=True`): on every project without a policy its refusal took the
  status bar after each edit and logged a traceback; it now runs on request. It now also reads exposures and
  readouts set on the motion blur page (the time sync study already did).
- A crash: the silhouette page's background Camera view render could finish after the project was replaced; its
  callback then read the cleared result and raised inside a Qt slot, which PyQt turns into an abort (the full suite
  died in `test_render_calib.py::test_page`; the agent had seen it once). The callbacks of the silhouette and
  ground-truth pages now drop late results and log instead of raising.
- Four checks declared no unit (`teleop.jacobian_fd`, `teleop.leader_workspace`, `teleop.cartesian_mapping`,
  `audit.slip_vs_vantage`), which `test_validation` caught only after its 255 s full run; a new test checks every
  registered check's unit, reference and quantity without running them.
- Batch: `--dataset` feeds `--silhouette-calib` too (it read the project's settings only); a refusal with one reason
  for every episode says it once.
- `tests/test_gui.py`: the placeholder count is now at least 1 (export_formats, Phase 10); the Data audit workspace
  opens on the audit page; the validation job test runs two cheap models. The audit workspace's status line no longer
  says "(Phase 8)".
- Self-test lines "teleop", "lighting", "timing", "datasets", "audit"; `scripts/screens.py --only phase458`;
  `scripts/measure_ui_phase458.py`.
- Docs: overview sections (groundtruth, synth_episodes, teleop, lighting, timing, data, render_calib, datalinks; the
  handoff in the multi-finger section; `swept.md` and every new model document in the table); user-guide chapters 8–10,
  the batch flags, the dataset links, silhouette calibration, the simulated capture and joint sessions; THIRD_PARTY
  (ported `core/rosbag.py` and `core/audit/slip.py`, Vantage's recorded slip outputs, the light, projector, exposure,
  refractive-index and format sources, the optional pyarrow, av, zstandard, lz4 and test-only mcap and protobuf);
  README; CHANGELOG.

**Done**
- Phase 4: `core/sim/scene.py`, `core/sim/render.py`, `core/studies/groundtruth.py` (`--groundtruth`) and the Rendered
  against predicted page (Simulate menu; Build the MuJoCo scene…); `validation/mjrender.py` extended (a persistent
  session, per-job pose and intrinsics, the EGL device); `core/sim/episodes.py` (`--synth-episodes`, Simulate ▸
  Generate synthetic episodes…, `ui/dialogs/synth_episodes.py`); `core/datasets/lerobot.py` (the writer, and the
  reader for Phase 8).
- Phase 5: `core/studies/teleop.py` (`--teleop`) and the Teleop overlap page; `core/materials.py`, `core/lights.py`,
  `core/studies/lighting.py` (`--lighting`), the light library kind (3 entries) and the Lighting page;
  `core/studies/motion_blur.py` (`--motion-blur`), `core/studies/time_sync.py` (`--time-sync`, `project.rig['sync']`)
  and their pages; `core/studies/action_chunk_sync.py` (`--action-chunk`) and its page; the tactile handoff in
  `core/studies/dex_hand_tactile.py` and its chart on the multi-finger page.
- Phase 8: `core/rosbag.py` (ported), `core/datasets/bag.py`, `core/datasets/rlds.py`, `core/datasets/joints.py`;
  `core/audit/` and the studies `audit` (`--audit`) and `plan_vs_recording` (`--plan-vs-recording`) with their pages
  in the Data audit workspace and the FiftyOne export; `core/calib/render_calib.py`, `core/studies/silhouette_calib.py`
  (`--silhouette-calib`) and the Silhouette calibration page; grasp sets from datasets (`core/grasps.from_dataset`,
  Tasks ▸ From a dataset…) and hand-eye capture from datasets (`core/calib/capture.capture_dataset`).
- Validation: `checks_groundtruth.py`, `checks_synth.py`, `checks_teleop.py`, `checks_lighting.py`, `checks_blur.py`,
  `checks_sync.py`, `checks_chunk.py`, `checks_audit.py`, `checks_render_calib.py`, `checks_datalinks.py`, and in
  `checks_formats.py` the LeRobot, MCAP and TFRecord round trips (`formats_bags.py`).
- Docs: `docs/models/` and `docs/guide/` groundtruth, synth_episodes, teleop, lighting, motion_blur, time_sync,
  action_chunk, handoff, datasets (and guide bags_rlds), audit, render_calib, datalinks (guides: grasps, handeye).

**Decisions (Phases 4, 5 and 8)**
1. **The render contract:** renders go through `core/sim/render.Renderer` (MuJoCo in an EGL worker process); a lens
   with distortion is rendered as an oversampled pinhole base (or cube faces for wide lenses) remapped through
   `CameraModel`. The principal-point convention ((W−1)/2 − px) is the one Phase 2 measured on renders.
2. **Contact agreement is reported as measured** (91.3 % / 84.2 %): the grasp study's definition (2 mm in front of
   the pad) is kept; the render asks whether the pad face shows. The occlusion-only disagreement (0.5 %) is in the
   result.
3. **Saturation is relative** (an ISO 12232 auto-exposure metering 18 % grey) unless the user enters a sensor's
   sensitivity, f-number and exposure; no library camera states a sensitivity. The lighting headline's status is the
   worse of glare and depth loss (depth thresholds 5, 15, 30 %, labelled defaults).
4. **Bounds, not guesses:** without a set exposure the frame period bounds the smear; a missing readout or device
   latency makes the sync skew a lower bound shown as "≥"; a policy whose inference time no source states is refused.
5. **Teleop comfort** is the central 0.8 of each leader joint range (a default), not a human-factors model; reach is
   position-only with a free orientation.
6. **The auditor reads only the data**: the synthetic sidecar is never read by it; tests and checks audit a copy
   without it. Floors per fault are what the method supports at the injected sizes; in-episode drift is reported, not
   claimed.
7. **Synthetic fault sizes are defaults chosen to show in a short episode** (drift 5000 ppm, desync 50 ms, gain 1.4,
   a 0.25 s USB gap), said in the docs; the gripper occluder is painted, not geometry.
8. **LeRobot files follow the source** at commit d490e47 (v3.0) and tag v0.3.3 (v2.1); v3.0 H.264 is written without
   B-frames so packets can be remuxed; AV1 is the default codec.
9. **The action-chunk study runs on request**, not live, and shares the motion blur page's exposures.
10. **The self-test validates only the cheap checks**; `--validate` runs all 113 (112 at integration).

**Spec corrections (Phases 4, 5 and 8)**
1. **§11 flag names:** `--teleop-overlap` is `--teleop`, `--sync` is `--time-sync`, `--silhouette-calib dataset/` is
   `--silhouette-calib OUT.json --dataset dataset/` (the study registry's automatic flags take OUT.json; `--dataset`
   names the recording for `--audit`, `--plan-vs-recording` and `--silhouette-calib`). MCP names arrive in Phase 10.
2. **§9.1 principal point:** MuJoCo 3.17.0's XML reference does not say which way the principal-point offset moves the
   image; the convention used is the one measured on renders (Phase 2) and checked again by the fiducial and
   per-pixel checks. `focalpixel` is converted with `sensorsize / resolution`, `fovy` is ignored once `sensorsize` is
   set, and a camera's `intrinsic` set directly compiles to zero (measured on 3.17.0).
3. **§8.2 "LeRobot compatible":** LeRobot's own loader was not run (the `lerobot` package is not installed); the files
   follow the source and are read back independently with pyarrow and ffmpeg. FiftyOne's importer likewise.
4. **§7.8 IR figures:** no lab IR source with a stated irradiance was found; the light library has no IR figures and
   users enter meter readings. Intel's D400 datasheet gives two D415 projector FOVs (67° and 64° horizontal); the
   entry keeps 67°.
5. **§7.4 VR and space mice:** no maker (Meta Quest, Apple Vision Pro, 3Dconnexion) publishes a tracking volume as
   such; their volumes are null until the user enters one.
6. **§7.9/§7.10 timing figures:** no library entry states a rolling-shutter readout time or a device's processing
   latency; only the D405 and D435 entries carry exposure ranges and trigger support so far.

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

Phases 9 and 10, samples, catalogue, library, research tools part 1:
- **Not run against the real programs:** Isaac Lab and Isaac Sim, LeRobot's loader and trainer, ACT, Diffusion
  Policy, OpenVLA, Kalibr and FiftyOne are not installed; the exports are checked against the pinned sources' fields
  and by independent readers. No request was sent to the Claude or Groq APIs (fake provider only); the Claude request
  shape follows the API reference read on 2026-11-07. The MCP server was exercised in process and over stdio here,
  not from a real MCP client. The Rerun live viewer only through a fake module.
- **Prices:** resolved by research tools part 2 (81 cited prices; `rig.cost` is a figure on the samples).
- **Tolerances:** no correlation between sensors on a shared bracket; intrinsics, joint encoders and link lengths are
  not drawn; slow metrics (the grasp study, about 30 s per draw on the UR5e) only on request.
- **Experiments:** `--reproduce` of a result from `--experiment DESIGN.json` is refused unless the project has that
  design saved (results from the page or `--experiment-saved` reproduce); the process pool does not share the row
  cache and is untested in a frozen bundle (Phase 11).
- **Optimiser:** the default budgets (96 and 120 designs) give a coarse front (105 s and 237 s on the UR5e); the
  design agent's sites are a sphere and a cone, without camera roll or stream choice; on the UR5e only four of the
  seven objectives are ranked (cost, bandwidth margin with no host, VLA object tokens with no critical object are
  left out, each with its reason).
- **Exporters:** one joint state per export; URDF without inertials, collision geometry, mimic joints or equality
  constraints; STEP is faceted (the UR5e wrist bracket at 0.5 mm is skipped above the 20 000-triangle cap unless
  `step_res_mm` is given); masks treat the floor as "no surface"; the depth study still uses 512-texel cube maps
  where the masks use 2048 (about 5 % of the overhead D435's far table mislabelled, as before); the randomisation is
  drawn once per Isaac config build; LeRobot state and action include the 2F-85's linkage joints (14 on the UR5e).
- **Canonical warp:** the default depth range is a pessimistic bound (675 px on the UR5e, 323 px on the SO-100;
  the page now has a depth-range field); `vla.warp_lut_vs_warpperspective` passes at its 1-grey-level bound with no
  margin. On the Panda sample the headline reads 11 878 px (P95) in a 1280 px image (the two wrist cameras are
  mirrored about the fingertips, so far points land near the canonical camera's image plane): the bound is right but
  hard to read; showing it as "larger than the image" is a decision for the warp's owner. The G1 has one camera, so
  the warp refuses (one sentence).
- **Assistant and MCP:** one request at a time, no progress notifications; no Preferences control for the window
  bridge (`mcp/bridge`, on by default); `handeye_solve` and `dex_occlusion` were listed as long studies by analogy;
  an approved plan runs from the dock (resolved in research tools part 2: `ui/assistant.py` Approve,
  `tests/test_assistant_dock.py::test_long_work_runs_only_from_an_approved_plan`).
- **Samples:** G1 silhouette calibration refuses the head camera (it rides on `torso_link`, which no joint moves;
  `render_calib` asks for a world mount); synthetic episodes move one robot (an ALOHA or Panda campaign records one
  arm) and on the G1 still report a "gripper hidden in the wrist camera" fault it cannot show; ALOHA's gripper
  equality constraint is not applied; the WidowX leader uses the follower's joint ranges (the WidowX's are not
  stated); no sample places a VLA task object.
- **Catalogue and library:** 22 of 53 catalogue robots have no maker figure; KUKA iiwa 14 has no repeatability or
  mass, SO-ARM100, Koch v1.1 and H1 no stated DoF, UR5e and UR10e no control rate; the Z1 entry uses the Z1 AIR's
  figures; GelSight Mini variants, Orbbec Gemini 2 XL and 305 and OAK-1 Max were not added; the bandwidth page does
  not take a capture card's stated sustained rate (Basler 3 040 Mbit/s per port, Leopard ~51 Gbit/s) into its
  controller automatically; the new lights state no flux, so the lighting study reports their output as not stated;
  the custom-device dialog cannot create teleop, policy, light or cable entries (right-click offers it only for the
  kinds it can make). Not fetched again at the truthfulness review: the Unitree G1/H1/Z1 pages, the Flexiv pages, the
  Neewer page (HTTP 429) and the camera, depth and tactile entries outside the roughly 50 it checked.
  `library.fov_shape` lets through maker angles that break the lens geometry by up to 2.92° under its 3° tolerance
  (Leopard OV2312 110H, e-con See3CAM_37CUG, Femto Bolt/Mega 10:9, ZED X One GS narrow); the check lists them, the
  entries' notes do not say so.
- **Research tools:** lists (grasp poses, joint tables) are diffed and merged whole; a plugin page whose id equals a
  built-in page's is replaced without a warning (documented); a plugin loaded while the window runs adds its page
  after a restart; only curve and bar charts give publication data specs. (The plugin template's real `pip install` test
  now runs: setuptools is the pinned `dev` extra.)
- **Tests:** the threaded AV1 decode failure above is worked around, not understood (presumably libdav1d's frame threads after a
  seek, under the suite's load; SVT-AV1 also prints "live_count should not be EB_ObjectWrapperReleasedValue" when an
  encoder closes, harmlessly as far as measured). The suite's peak (4.8 GB) comes from the in-process validation checks that some modules run
  (`test_synth_episodes`, `test_audit`); moving those to child processes as `test_validation` does would lower it.
- **Window:** the Analysis dock's tab bar scrolls. Side-by-side variants share one 3D view (dashed ghosts), not
  split viewports. (Resolved in research tools part 2: hidden pages hold their signals until shown, an edit costs
  4–7 ms; adding the G1 is 10–13 ms on the UI thread; a refused study logs one line; surfaces are in the Outliner.)
- **From the four reviews (still open):**
  - Spec §13: the USD robot import and §7.11's cost objective were built in research tools part 2 (see there).
  - Validation: `tolerance.linear_propagation` and `tolerance.sampler_ks` now declare the band they apply
    (`tolerance_quick`; resolved in research tools part 2).
  - Exports: resolved in research tools part 2 (the Isaac principal point reaches Isaac Lab v3.0.0 and is stated for
    v2.3.2; OpenVLA encodings follow the rig; the URDF carries inertials and MuJoCo loads it).
  - Experiments: bounded figures are now censored (`experiments.censored_best`; resolved in research tools part 2).
    Diff and merge match a sensor name as unique among the not-yet-matched items; when ours and theirs make the same change merge3 counts it in neither (cosmetic).
  - The assistant's number check does not flag a figure the model derived itself, nor a right value cited for the
    wrong quantity (documented).
  - `--check` without an output file writes `scorecard.json` into the current folder (documented in its help).
  - The G1 sample opens with 0 % seen by two or more cameras (critical) and 0 % grasp contact visibility (one head
    camera: honest for that rig); `--action-chunk` refuses on it (GR00T N1.5's control rate is not stated).
- **Temporary folders:** before the final gate's fix, every test session, self-test and validation run left its
  throw-away settings and XDG folders behind: 12 195 `dexvantage-*` folders (4.1 GB) are still in the system temp
  folder (the gate was not allowed to delete shared temporary files). Delete `/tmp/dexvantage-*` when no DexVantage
  process or test run is going; new runs remove their own.

Phases 4, 5 and 8:
- **Ground truth:** the prediction leaves out the camera's seat on its mount link and the render does not; MuJoCo draws
  smooth primitives, the predictions 24-gons; depth compared at home only; distorted lenses are slow (2.4 and 0.7
  views/s at full size); `device='auto'` takes MuJoCo's first EGL device rather than searching for a discrete GPU as
  `core/occlusion` does. The depth study can call a point on a far surface hidden from itself (a texel spans more
  depth than its 1 mm tolerance: 91.6 % on the overhead D435's table); an eps that grows with distance in
  `core/occlusion` or the depth study would fix it.
- **Occlusion GPU path:** at 256 texels it missed about 8 texels of a cylinder's top cap seen exactly along its axis
  (exact ray casting and numpy hit it); the lighting study defaults to 512 texels. To be looked at in
  `core/occlusion.py`.
- **Synthetic episodes:** duplicates are re-encoded by the codec (g = 2), so decoded duplicates are 36–40 dB from their
  original, not identical; grasp approaches do not close the gripper and are not checked against obstacles; one
  imager per sensor; no depth videos; a faults dict overrides only the faults it names.
- **Teleop:** no library teleop-device entries; reach is position-only; a primary-robot base move does not move
  hand-eye board placements.
- **Lighting:** direct light only; the Library panel has no Lights group and the Outliner does not list lights (added
  from the Lighting page); the Inspector cannot edit material tags; the schema describes `workspace.lights` as plain
  objects (the contract's keys are not spelled out in either schema copy).
- **Motion blur and sync:** a per-sensor exposure applies to all its imagers (the D435's IR and RGB exposures are
  separate controls); clock offsets and jitter are not added to the action-chunk loop; the default command latency
  (one control period) is large for a 10 Hz policy.
- **Handoff:** the multi-finger study with the handoff takes about 12.8 s on the dexhand sample (6 s more than
  without); the fingers do not close during the approach.
- **Datasets:** LeRobot's own loader and FiftyOne's importer have not been run (not installed); Parquet files carry no
  HF `huggingface` schema metadata; `tasks.parquet`'s pandas index block is hand-written; rollover is measured on
  Arrow and packet bytes, not files on disk; depth videos are listed but not dequantised. Bags: actions, dynamic `/tf`
  and vendor exposure metadata are not read. RLDS: no timestamps, intrinsics or extrinsics; without an fps, times
  count steps (`meta['time_base'] == 'steps'`); `open_dataset(path)` cannot pass an fps.
- **Auditor:** in-episode drift is not resolved (recall 0 at 5000 ppm); duplicates on the small third-person view
  0.5; desync on cameras on the robot biased by up to about 50 ms on textureless scenes; blur false positives 0.1.
- **Silhouette calibration:** on the SO-100 third-person camera (robot about 4 % of the image) 2.4 mm and 0.30° off,
  σ understating by up to 7 σ; a camera–joint desync biases the mount.
- **Data links:** the hand-eye solve study's batch flag captures only from a folder (datasets are captured from the
  page or `capture_dataset`); no clock offset is estimated; grippers stored as a percentage are not converted.
- **Validation time:** the quick run of all checks took 136 s at 112 checks (the audit campaign and the silhouette
  checks
  most of it); the self-test and the plumbing tests run cheap checks only.
- **Test memory:** the suite passes under its 7 GB cap with little margin (process peak 6.55 GB, cgroup 7.11 GB with
  page cache): the process still creeps from 0.2 to about 4.7 GB over 1 886 tests (after two GUI modules `gc` found
  one window and no robot model or page alive, so most of it is native or allocator heap), and the module that reaches
  the 6.55 GB peak
  was not identified. Lowering the creep, or running the heavy GUI modules in their own process, is the next step.
- **After the reviews:** the dexhand head camera's surface agreement (91.4 %, 97.7 % inside the task volume) has the
  D435's pattern (almost all "occluded → valid") but was not traced pixel by pixel; `batch.py` keeps its own copy of
  `DATASET_STUDIES` (it could import `core.jobs`'s); the audit and plan pages still pass their own fingerprint
  (redundant with the cache key, one extra pass over the files off the UI thread); the time-sync camera-pairs table
  uses half the longer frame period (an upper bound); a bag CDR message with an encapsulation other than XCDR1 is
  refused; §8.3 sync against the plan gives no verdict while the time-sync budget is incomplete (every library device
  today: no maker states its processing latency); `lighting.schlick_vs_fresnel` bounds Schlick's own approximation
  error (0.036 against 0.04) rather than validating the model; Schlick 1994 "eq. 15" and the page range of Heitz 2014
  (JCGT) were not re-read (not freely fetchable); the software 3D paint (offscreen QPainter fallback) takes
  150–1100 ms per frame on the UI thread; the Analysis dock gets about 180 px in the Placement workspace at
  1680 × 1000.
- **Hub streaming:** video bytes read by range requests are not kept on disk (a second pass reads them again unless
  they are among the 24 blocks of 4 MiB in memory); gated and private datasets are refused (no token); the cache key
  does not follow a branch's newest commit (the commit pinned at open is recorded in the result).
- **Phase 10 (spec §9.2, §11, §7):** the LeRobot training-config exporter (delay offset and ensemble weight m),
  `--mujoco-validate out/` as a folder (now `--groundtruth OUT.json`), `--export … lerobot,fiftyone`, `--rerun`, and
  each study's assistant and MCP tools.
- **Samples:** no light, no exposure set, no teleop device on the UR5e and dexhand samples, no policy on the dexhand
  sample, and SmolVLA's inference time is not stated, so `--lighting`, `--teleop` and `--action-chunk` say so on them.

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
  - Grasps: trajectory closures with no object (a test close, a regrasp) count as grasps (also from datasets,
    Phase 8). 10 000-grasp sets stall the UI 150 ms–1 s after Add set (worker threads holding the
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
- `kinematics.fk_pinocchio` is not run in `.venv` (pinocchio, the optional `pin` extra, is not installed there); the
  Phase 11 fresh-install review ran it once with pin 4.1.0 (9.57 × 10⁻¹⁶ m), so a run with every extra has 233
  checks.
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
