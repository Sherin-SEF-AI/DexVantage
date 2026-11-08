# Packaging: the downloadable app

DexVantage ships as a PyInstaller bundle for each platform. The bundle is a folder with Python, Qt, NumPy, SciPy,
MuJoCo, OpenCV, PyAV (FFmpeg) and pyarrow inside, plus two launchers:

- `DexVantage` (`DexVantage.exe`) opens the window. On Windows it has no console.
- `dexvantage-cli` (`dexvantage-cli.exe`) runs batch mode, `--self-test`, `--version` and `--validate`, and shows
  errors in the terminal.

The two launchers share one copy of everything else. Each launcher is 22.8 MB on Linux (`ls -l` of the 0.1.0rc1
build), so the second one costs 22.8 MB; two bundles would cost the whole 849 MB
again.

| Platform | What you download | Install | Built |
|---|---|---|---|
| Linux x86_64 | `DexVantage-<version>-linux-x86_64.tar.gz` | `./install.sh` for your user only, no root | here; by CI (not run yet) |
| Linux arm64 (Jetson) | `DexVantage-<version>-linux-arm64.tar.gz` | `./install.sh` | by CI only (not run yet) |
| Windows x64 | `DexVantage-<version>-windows-x64-setup.exe` (and a `.zip`) | per-user installer, no administrator rights | by CI only (not run yet) |
| macOS (Apple silicon) | `DexVantage-<version>-macos-arm64.zip` (`DexVantage.app`) | drag to Applications | by CI only (not run yet) |

Everything here is ported from Vantage `packaging/`. The automotive parts are gone. The style of the spec, the build
script, the installers and the release script is kept.

## Files

| File | What it does |
|---|---|
| `packaging/dexvantage.spec` | The PyInstaller spec: what goes in, what stays out, and the checks that stop a broken build. |
| `packaging/run_dexvantage.py` | The entry script. It calls `multiprocessing.freeze_support()` first, runs `-m dexvantage.…` worker launches, then calls `dexvantage.app.main()`. |
| `packaging/build.py` | Builds, smoke-tests (`--smoke`, `--full-smoke`) and packs the bundle for this platform. |
| `packaging/release.py` | Reads the version, CHANGELOG notes, data files and licences. Makes the release: checksums and notes. It prints the `gh` command and runs it only with `--publish`. |
| `packaging/requirements-bundle.txt` | The pinned versions CI builds the bundles from. |
| `packaging/make_icon.py` | Writes `icons/dexvantage.ico` and `icons/dexvantage.icns` with the same drawing as `scripts/make_icon.py`. |
| `packaging/linux/` | `install.sh`, `uninstall.sh`, the `.desktop` entry and the `.dexv` MIME type. |
| `packaging/windows/dexvantage.iss` | The Inno Setup script for the per-user installer. |
| `packaging/macos/` | `Info.plist` (the spec reads it), `entitlements.plist` and `sign_and_notarise.sh`. |
| `dexvantage/extras.py` | Which optional extras the bundle carries, and the sentence the app adds when one is missing. |
| `.github/workflows/bundles.yml` | CI: builds and smoke-tests every platform. On a version tag it makes a draft release. |

## Building on Linux x86_64

    .venv/bin/pip install -e ".[bundle,lerobot,bags,assistant]"   # or: pip install -r packaging/requirements-bundle.txt
    .venv/bin/python packaging/build.py --smoke          # bundle, smoke test, tar.gz into out/dist (ignored by git)
    .venv/bin/python packaging/build.py --full-smoke     # the same, plus --validate --quick from the bundle

PyInstaller is pinned in the `bundle` extra (`pyinstaller==6.22.3`, `pyinstaller-hooks-contrib==2026.9`). The build
writes into `out/build` and `out/dist`.

### What goes into the bundle

- **The `dexvantage` package.** `collect_submodules('dexvantage')` takes every module, because studies, pages,
  exporters, readers and checks are imported by name. `module_collection_mode={'dexvantage': 'pyz+py'}` puts in the
  compiled modules and the `.py` files too. Three things need the `.py` files:
  - the folder scans that find those modules (`pkgutil.iter_modules` over the package's folders);
  - the provenance `code_sha256` (SHA-256 over the `.py` files);
  - `--reproduce`.

  The spec stops with an error if any module of the package is missing from the bundle.
- **Resources.** `release.data_files()` takes every file of the package that is not Python:
  - fonts and icons;
  - the library JSON and `library.schema.json`;
  - `project.schema.json`;
  - the bundled robots and USD robots;
  - JSON files that sit next to modules.

  `LICENSE`, `CITATION.cff` and `THIRD_PARTY.md` go at the top. `core/cite.py` reads `CITATION.cff` from the folder
  above the package. `tests/test_packaging.py` checks that every `resources/` subfolder is listed.
- **MuJoCo.** The analysis finds `libmujoco` and its extension modules. The spec adds three things by hand:
  - the plugin libraries in `mujoco/plugin`, which `mujoco/__init__.py` loads from that folder;
  - the OpenGL backends `mujoco.egl`, `mujoco.osmesa`, `mujoco.glfw` and `mujoco.cgl`, which MuJoCo picks by
    `MUJOCO_GL` at run time, and the PyOpenGL platform modules they load by name (EGL, GLX, OSMesa, WGL);
  - the `glfw` package's `libglfw`. `import mujoco` loads MuJoCo's GLFW backend on Windows (and on macOS when CGL
    fails), so the bundle needs it even though DexVantage never renders through it: DexVantage renders MuJoCo scenes
    only through EGL, in a worker process (`validation/mjrender.py`), and EGL is a Linux API. On Windows and macOS
    the rendered features (ground truth, synthetic episodes, the silhouette calibration, the sim-to-real gap and the
    rendered validation checks) stop with the message "MuJoCo cannot render offscreen through EGL on this machine",
    and the tests that need a render skip with that reason (`tests/render_support.py`).

  MuJoCo's 3.17.0 wheel ships no EGL or OSMesa library. PyOpenGL loads those from the system (a GPU driver or Mesa)
  with ctypes, so they are never bundled. The bundle carries no `libEGL` or `libGL`, which is right for Jetson: there
  NVIDIA's own EGL must be used.

  There is no PyInstaller hook for MuJoCo, in PyInstaller 6.22.3 or in hooks-contrib 2026.9. These MuJoCo parts are
  left out because DexVantage does not use them:
  - the interactive viewer and `_simulate`;
  - the experimental studio (41 MB);
  - the Filament renderer;
  - sysid and MJX.
- **Qt.** PyQt6's own hook collects:
  - the platform plugins: xcb, wayland, offscreen, cocoa and windows;
  - the image formats, the SVG icon engine and the TLS backends.

  Qt modules DexVantage never imports are excluded, and so are Qt's translations. On Linux the spec also drops the
  GTK 3 platform theme and the 30 libraries only it loads, the same as Vantage. A bundled GTK loads the user's own GTK
  modules and themes.
- **Video and Parquet.** hooks-contrib's hooks collect PyAV with its FFmpeg libraries (`av.libs`) and pyarrow. Some
  pyarrow parts are excluded: Flight, the S3, GCS, Azure and HDFS filesystems, ORC, Substrait, CUDA, and its C++
  headers and Cython sources. On Linux the spec also drops the nine libraries that only those parts load. DexVantage
  reads and writes local Parquet only.
- **Data and metadata.** SciPy's hook leaves out `scipy/stats/_sobol_direction_numbers.npz`, which the Sobol designs
  need, so the spec adds it. It also adds the `importlib.metadata` of mujoco, numpy, PyQt6 and PyQt6-Qt6, which the
  provenance block and the publication kit read.
- **Licences.** The licence files of every bundled distribution go into `licenses/<distribution>/`. A notice,
  `licenses/THIRD-PARTY.txt`, lists each component with its version and licence. This is ported from Vantage.

### What is kept out, and why

- **Pillow.** It is pulled in only by `av.video.frame.to_image`, which DexVantage never calls.
- **Optional SDK integrations.** Pygments, Google auth, the OpenTelemetry SDK and exporters, `requests` and `fsspec`
  come from `collect_submodules('anthropic')` and pyarrow's optional imports on a well-stocked build machine. The
  Anthropic SDK needs only `opentelemetry-api`.
- **DexVantage's own install metadata.** A development build has an editable install, and its `direct_url.json`
  names the checkout's path. Any `direct_url.json` is dropped.
- **A home-folder check.** `build.py` scans the text files of the finished bundle for the build machine's home
  folder and fails if it finds it (DEVELOPING.md rule 7).

The first smoke test found two defects, and the spec now fixes both. The SciPy Sobol file was missing. Without the
mujoco metadata, the publication check crashed. These failures are why `--smoke` is part of every build.

### Optional extras: which ship in the bundle

The bundle carries the extras whose features are part of the main workflow and leaves out the large ones that a
feature can do without.

The sizes below are installed sizes, measured with `python packaging/release.py extras` on 2026-11-08 (Linux x86_64,
Python 3.14). Each figure is the sum of each distribution's `RECORD` files.
- For a bundled extra, it is what the extra adds to the core requirements.
- For an extra that is left out, it is what the extra would add to the bundle as built. Shared dependencies are
  counted once.

| Extra | For | In the bundle | Size (MB) | Why |
|---|---|---|---|---|
| `lerobot` (pyarrow, av) | LeRobot Parquet and AV1/MP4 video | yes | 308.0 | the auditor, plan against recording, the sim-to-real gap and the LeRobot exporter |
| `bags` (zstandard, lz4) | compressed ROS 2 bags | yes | 28.6 | small; a compressed bag cannot be read at all without it |
| `assistant` (anthropic, keyring) | the Assistant dock | yes | 51.4 | small; the dock should work in the download (it still needs the user's key) |
| `usd` (usd-core) | USD import, pxr-written Isaac prims | no | 217.6 | the Isaac exporter writes `cameras.usda` as text without it; import URDF or MJCF instead |
| `rerun` (rerun-sdk) | the Rerun bridge | no | 554.4 | the largest; the Rerun viewer is a separate install anyway |
| `step` (gmsh) | bracket STEP export | no | 109.4 | STL brackets work without it |
| `research` (pandas) | `Result.to_frame()` in the Python API | no | 75.1 | the bundle has no Python prompt |
| `tracking` (mlflow-skinny, wandb) | reading the tracker's exports back | no | 160.6 | the tracker writes the MLflow store and W&B script without them |
| `pin` (pinocchio) | a second IK backend | no | — | not installed where measured |

Small validation references also ship when they are installed: jsonschema, mcap and protobuf. With them,
`--validate` runs the same checks in the bundle as from source.

When a feature needs an extra the bundle leaves out, its own "is not installed" message gets one more sentence from
`dexvantage.extras.bundle_note`. For example: *This DexVantage download leaves out the Rerun bridge (it would add
about 554 MB): run DexVantage from a pip installation to use it.* Each feature's own message already names the
`pip install "dexvantage[…]"` command. From source or a
pip installation the sentence is empty. The features that add it are:
- the Rerun bridge;
- the USD import, in the importer and the wizard;
- the Isaac exporter's USD note;
- the bracket STEP export.

### Processes in a frozen app

DexVantage starts processes in two ways, and both work in the bundle.

1. **Spawned pools.** The cluster's local pool (`core/cluster/local.py`) and the designed experiments' pool
   (`core/experiments.py`) use multiprocessing's spawn start method. In a bundle the child is the bundle's own
   executable, started with `--multiprocessing-fork`. Its resource tracker is started with `-c "from
   multiprocessing.resource_tracker import main…"`. PyInstaller's multiprocessing runtime hook makes
   `freeze_support()` recognise both on every platform. `run_dexvantage.py` calls it first, before the argument parser
   sees those arguments.
2. **Worker modules.** The MuJoCo render worker, the occlusion GPU worker, a cluster shard and the MCP batch runner
   are started as `[sys.executable, '-m', 'dexvantage.…']`. In a bundle `sys.executable` is the launcher, so
   `run_dexvantage.py` runs the module itself, the way `python -m` would. It runs DexVantage's own modules only and
   refuses any other in one sentence. `tests/test_packaging.py` checks that every worker launch in the code uses this
   form.

### Measured on this machine (Linux x86_64)

Measured on 2026-11-08 on the development machine: Ubuntu 26.04.1, kernel 7.0, glibc 2.43 (`ldd --version`), Ryzen 9
9900X, RTX 5080, 32.1 GB.
The build used Python 3.14.4, PyInstaller 6.22.3 and opencv-python 5.0.0.93 from the development environment. All
runs used `QT_QPA_PLATFORM=offscreen` and throw-away XDG folders.

| What | Measured | How |
|---|---|---|
| Bundle folder | 849.0 MB (10⁶ bytes) | `du -sb out/dist/DexVantage` |
| Download (`.tar.gz`) | 315.0 MB | the file `build.py` wrote |
| Largest parts | opencv-python 195.1 MB (cv2 74.2 + its libraries 120.9), av 130.9 MB (25.8 + FFmpeg libraries 105.1), pyarrow 104.4, PyQt6 101.0, scipy 73.4, dexvantage 51.5 (robots and library 41 MB, sources 7 MB), numpy 41.9, mujoco 22.0 | `du -sb` per folder of `_internal` |
| Start-up to the main window | 2.70 s median of 5 (2.26–2.92); from source 2.18 s (1.91–2.49) | `DEXVANTAGE_STARTUP_PROBE=1`, wall time from launch until the event loop runs with the main window shown; bundle and source runs alternated |
| `dexvantage-cli --version` | 0.26–0.30 s (source 0.19–0.22 s) | wall time, 3 runs each |
| Batch run of three studies on SO-100 | 1.0 s | `--sample so100 --coverage … --depth-validity … --bandwidth …` |
| Cluster local pool, 2 spawned workers | 17.6 s, peak 637 MB per worker | `--sample so100 --tolerance-mc tol.json --jobs 2` (64 draws) |
| Experiments pool, 2 spawned workers | 6.2 s | `--sample ur5e --experiment design.json`, 4 LHS runs with `"execution": {"workers": 2}`; the result records "process pool of 2" |
| `--self-test` | passed, 46 of 46 lines ok, 66.7 s | every line must be ok except OpenGL (`BUNDLE_MAY_WARN`); here OpenGL was ok too |
| `--validate --quick` | 228 of 228 passed, 401.9 s | from the bundle. The source registers 232 checks; the other 4 register only when their reference library is installed (`formats.bracket_step`: gmsh; `formats.rerun_rrd_readback`: rerun-sdk; `robot_io.usd_roundtrip`, `robot_io.usd_third_party`: usd-core), and those extras are left out of the bundle |

The machine was shared with other jobs while these figures were taken (load average 11–15), so the times are
upper bounds for an idle machine. `build.py --smoke` writes the same figures to `out/dist/smoke-report.json` on every
build.

**The release candidate build (0.1.0rc1, the Phase 11 gate, 2026-11-08).** Built again from the gate's source with
`packaging/build.py --smoke` (same machine and environment, load average about 6):

| What | Measured | How |
|---|---|---|
| Bundle folder | 849.0 MB | `du -sb out/dist/DexVantage` |
| Download | `DexVantage-0.1.0rc1-linux-x86_64.tar.gz`, 315.0 MB (314 987 717 bytes) | the file `build.py` wrote |
| Start-up to the main window | 1.66 s median of 3 (first 1.72 s) | `build.py --smoke` |
| `dexvantage-cli --version` | 0.20 s (smoke), 0.25 s (by hand) | prints `dexvantage 0.1.0rc1` |
| Batch run of three studies on SO-100 | 0.8 s | `build.py --smoke` |
| Batch run on ALOHA 2 (coverage and bandwidth) | 2.15 s, peak 763 MB | `--sample aloha --coverage … --bandwidth …`, `/usr/bin/time` |
| Cluster local pool, 2 workers | 14.1 s | `build.py --smoke` |
| Experiments pool, 2 workers | 4.0 s | `build.py --smoke` |
| `--self-test` | passed, 46 of 46 lines ok, 45.7 s | `build.py --smoke` |

**The final gate's build (2026-11-08, after the four Phase 11 reviews).** Built again with `packaging/build.py
--smoke` from the final source (Python 3.14.4, glibc 2.43, the same machine), because the reviews changed
`selftest.py`, `core/cluster/shards.py` and `run_dexvantage.py` after the build above. This is the archive in
`out/dist` that `packaging/release.py --dry-run` checksummed:

| What | Measured | How |
|---|---|---|
| Bundle folder | 849.0 MB (849 007 085 bytes) | `du -sb out/dist/DexVantage` |
| Download | `DexVantage-0.1.0rc1-linux-x86_64.tar.gz`, 315.0 MB (315 010 387 bytes), SHA-256 `d14b8ba39b53407c46636f0ae36828d1e4017412087b18fc2719e4e4f35db953` | the file `build.py` wrote |
| Start-up to the main window | 1.6 s median of 3 (first 1.6 s) | `build.py --smoke` |
| `dexvantage-cli --version` | 0.27 s | `build.py --smoke` |
| Batch run of three studies on SO-100 | 0.8 s | `build.py --smoke` |
| Cluster local pool, 2 workers | 14.3 s | `build.py --smoke` |
| Experiments pool, 2 workers | 4.0 s | `build.py --smoke` |
| `--self-test` | passed, 46 of 46 lines ok, 46.3 s | `build.py --smoke` |
| The whole build and smoke test | 2 min 30 s, peak 3.59 GB | `/usr/bin/time -v`, memory-capped at 7 GB |

## Windows (not built here)

`packaging/windows/dexvantage.iss` is an Inno Setup 6 script with `PrivilegesRequired=lowest`. It installs for the
user into `%LOCALAPPDATA%\Programs\DexVantage`, with no administrator rights. It adds:
- a Start menu entry;
- a "DexVantage self-test" shortcut (`cmd /k` with quoting that survives a path with spaces);
- an optional desktop icon;
- the `.dexv` file type under `HKCU`.

The setup also offers an all-users install, which then asks for administrator rights. An update replaces
`{app}\_internal`.

`build.py` runs `iscc` when Inno Setup is on `PATH` or in its default folder. It also writes a `.zip` of the folder.
The installer is not signed. SmartScreen shows "Windows protected your PC"; choose **More info ▸ Run anyway**.

## macOS, Apple silicon (not built or signed here)

The spec's `BUNDLE` step makes `DexVantage.app` from `packaging/macos/Info.plist`. It adds the version, the icon and
the executable.

`LSMinimumSystemVersion` is 14.0, because the newest arm64 wheels the bundle is built from need macOS 14 (PyPI, read
2026-11-08):
- pyqt6 6.12.0 is `macosx_14_0_universal2`;
- numpy 2.5.4 and scipy 1.18.1 ship `macosx_14_0_arm64` builds, which pip prefers on macOS 14 and later.

`LSBackgroundOnly` is false. Without it, PyInstaller marks the app background-only, because COLLECT takes the console
flag of the last launcher. The plist also declares the `.dexv` document type.

Signing and notarisation are documented, not run. They need the maintainer's Developer ID certificate, and none is
in this repository. `packaging/macos/sign_and_notarise.sh APP "Developer ID Application: NAME (TEAMID)" PROFILE`
does it in three steps, as in Apple's "Notarizing macOS software before distribution":
1. **Sign** every Mach-O file inside the app, deepest first, then the app itself. Each one gets the hardened runtime
   (`--options runtime`), a secure timestamp and `packaging/macos/entitlements.plist`.
2. **Notarise.** Zip the app with `ditto` and submit it with `xcrun notarytool submit … --wait`.
3. **Staple.** Run `xcrun stapler staple`, check with `spctl --assess`, then zip again for the release.

`PROFILE` is a keychain profile, made once with `xcrun notarytool store-credentials`.

The entitlements turn off library validation and allow unsigned executable memory (ctypes callbacks). They have not
been tried on a signed build yet. When the first signed build runs, try removing each one and keep only what the app
needs.

An unsigned app shows "cannot be opened because Apple cannot check it for malicious software". Open it anyway from
**System Settings ▸ Privacy & Security ▸ Open Anyway**.

## Linux arm64 (NVIDIA Jetson): what differs

Checked on PyPI on 2026-11-08 (`https://pypi.org/pypi/<name>/<version>/json`), and on NVIDIA's JetPack pages:

- **MuJoCo 3.17.0** has `manylinux_2_27`/`manylinux_2_28` aarch64 wheels for CPython 3.12 to 3.15. They install on
  glibc 2.28 and later.
- **PyQt6 and Qt.** PyQt6 6.12.0 and PyQt6-Qt6 6.12.0 have aarch64 wheels for `manylinux_2_39` only, so they need
  glibc 2.39. PyQt6-Qt6 6.8.1 is also `manylinux_2_39` on aarch64. The newest aarch64 wheels for glibc 2.28 are PyQt6
  6.7.1 and PyQt6-Qt6 6.7.3. On x86_64, 6.12.0 is `manylinux_2_34`.
- **JetPack versions.** JetPack 6.2 uses an "Ubuntu 22.04 based root file system" (glibc 2.35). JetPack 7 is "built
  on Linux Kernel 6.8 and Ubuntu 24.04 LTS" (glibc 2.39), for Jetson Orin and Thor (developer.nvidia.com/embedded/
  jetpack-sdk-62 and developer.nvidia.com/embedded/jetpack, read 2026-11-08). PyQt6 6.12 therefore installs on
  JetPack 7 but not on JetPack 6. `requirements-bundle.txt` pins 6.7.1 / 6.7.3 on aarch64.

  DexVantage declares `PyQt6>=6.5` but has been tested on 6.12 only. The arm64 CI job's self-test is the check for
  6.7.
- **Other wheels.** These all have aarch64 manylinux_2_28 (or older) wheels: av 19.0.2 (cp312-abi3), pyarrow 26.0.1,
  numpy 2.5.4, scipy 1.18.1, opencv-python-headless 5.0.0.93, glfw 2.11.0, zstandard 0.25.0, lz4 4.4.5, PyQt6-sip
  13.13.0 and pyinstaller 6.22.3.
- **Where to build.** A PyInstaller bundle runs only on a glibc at least as new as the machine that built it. One
  arm64 bundle for JetPack 6 and 7 must therefore be built on Ubuntu 22.04 arm64 (GitHub's `ubuntu-22.04-arm`
  runner). The same holds for x86_64 bundles meant to run on Ubuntu 22.04. `.github/workflows/bundles.yml` therefore
  builds the Linux bundles on `ubuntu-22.04` and `ubuntu-22.04-arm` (glibc 2.35), so they run on Ubuntu 22.04 and
  later and on JetPack 6 and 7; the tests still run on `ubuntu-24.04`. GitHub lists both 22.04 labels (runner-images
  README, read 2026-11-08) and retires the oldest label once a newer image is out, which Ubuntu 26.04 now is: when
  22.04 goes, the arm64 bundle needs another glibc-2.35 build machine (a container such as `ubuntu:22.04`) to keep
  JetPack 6.
- **Rendering.** On a Jetson, MuJoCo's EGL backend and the occlusion cube maps use NVIDIA's Tegra EGL through the
  system's libglvnd. The bundle carries no EGL or GL library, so the system's is used. CUDA is not used by the bundle.

## Releasing (only when the maintainer says so)

    python packaging/release.py                    # build + smoke test, SHA256SUMS.txt, RELEASE_NOTES.md, print gh command
    python packaging/release.py --no-build --dist out/dist   # the same for artefacts already built (e.g. from CI)
    python packaging/release.py --dry-run --dist DIR --out DIR2   # no build; checksums and notes of what is in DIR
    python packaging/release.py --publish ...      # the maintainer only: runs the printed gh release create

`release.py` writes:
- `SHA256SUMS.txt`, over every artefact in `--dist` (`.tar.gz`, `.zip`, `-setup.exe`, `.AppImage`, `.dmg`), in
  `sha256sum`'s format;
- `RELEASE_NOTES.md`, the CHANGELOG entry of the version.

It then prints the `gh release create v<version> … --verify-tag --notes-file …` command, with `--prerelease` for a
dev or rc version. It runs that command only with `--publish`.

`--publish` refuses a development version (`.devN`); a release candidate such as `0.1.0rc1` may be published, as
a GitHub pre-release. It also refuses a version with no CHANGELOG entry of its own. A `--dry-run` of a `.dev`
version takes the `[Unreleased]` notes and says so in the first line.

`python packaging/release.py notes --tag vX.Y.Z` prints the notes for CI and fails when the tag and the version
disagree.

CI's `bundles.yml` builds every platform on a version tag and makes a draft release. It never publishes.

## Packaging and CI review (2026-11-08)

A review of every workflow and packaging file, line by line, after the Phase 11 gate. What was checked, and how:

| What | How | Result |
|---|---|---|
| Action pins | `git ls-remote` of each action's repository, 2026-11-08 | all seven SHAs are the commits their tags name, and each is the newest release |
| actionlint download | its v1.7.12 `checksums.txt`, fetched again | all six pinned SHA-256 match |
| Runner labels | the runner-images README, 2026-11-08 | all listed; `windows-latest` and `ubuntu-latest` replaced by `windows-2025` and `ubuntu-24.04` (a `-latest` label moves to a new image over one to two months) |
| Bundle pins have wheels | PyPI JSON of each pin in `requirements-bundle.txt`, CPython 3.12 (what CI builds with) | all 24 have wheels for Linux x86_64 and arm64 (glibc 2.35 or older), Windows x64 and macOS arm64 |
| macOS minimum | the newest macOS tag of the arm64 wheel pip picks on macOS 15, per pin | 14.0 (pyqt6, numpy, scipy, av), as `LSMinimumSystemVersion` says |
| CI pins have wheels | the same for `.github/ci-constraints.txt`, CPython 3.12 and 3.14 | all have wheels on the test platforms; `usd-core` and `gmsh` have none for Linux arm64, where CI runs no tests |
| Oldest Python | every source file compiled by CPython 3.10.20 and 3.12.13 | no syntax error |
| Shards | `scripts/ci_shards.py --list` | every test file in exactly one shard (141 when checked, 144 at the end: other reviews added files). The duration table lacked 34 files, which counted as the median (about 5 s) although `test_guide_examples.py` takes 374 s; it is now taken from the review's full run (3 993 tests, 2 610 s), and the five shards beside `test_validation`'s are 468 s each on this machine |
| CI's Linux renderer | the rendered tests (20 files, `DEXVANTAGE_TEST_RENDER=require`) and `--validate --quick` with Mesa alone (`__EGL_VENDOR_LIBRARY_FILENAMES` naming Mesa's vendor file, `LIBGL_ALWAYS_SOFTWARE=1`), `MUJOCO_GL` unset | the probe reports llvmpipe; 212 of 213 tests passed (the other failed on another agent's half-written file and passed alone), 20 min 48 s; validation 232 of 232, 10 min 34 s; the bundle's `--self-test` passed (OpenGL, occlusion and grasp on llvmpipe) |
| No renderer (as on Windows and macOS) | 52 render-related test files with no EGL vendor (`__EGL_VENDOR_LIBRARY_FILENAMES` naming no file), `DEXVANTAGE_TEST_RENDER=auto` | 1 019 passed, 82 skipped, none failed (27 min 38 s): no test renders without the marker or a gated fixture. The skips say why: 74 "needs MuJoCo's offscreen renderer (EGL)" with the worker's reason, 6 "no OpenGL for the GPU backend here" (the occlusion GPU backend's own tests), 2 no LaTeX engine. Tests in the same files that do not render ran. (A first run stopped once with a Qt abort in a window fixture while an earlier test's statistics thread was still running; the rerun did not repeat it.) |
| The bundle, every page | the rebuilt bundle (below) running a walk script through its own `-m` dispatch: the 35 pages opened on all six samples offscreen, each page's study run on the UR5e sample and its result exported | 29 studies ran, 24 with a result; the other 5 refused in a plain sentence for want of input (a capture session, a dataset, a factor, two conditions, two cells); no missing module, resource or traceback |
| The bundle, every batch flag | 82 commands: every study flag on SO-100 (the multi-finger one on the dexhand), three studies on each other sample, synthetic episodes and the dataset studies on them, every export format alone and together, `--check`, `--reproduce`, `--run-plan`, `--unpack`, `--pack`, `--textconv`, `--diff`, `--merge`, `--cite`, `--figure`, `--campaign`, `--stats`, `--track`, `--runs-export`, `--cells`, `--experiment` (serial and `--jobs 2`), `--tolerance-mc --jobs 2`, `--slurm` with both `--run-shard`s and `--gather`, `--cluster-studies --jobs 2`, `--optimise --jobs 2`, `--validate --models optics` | none printed a traceback, a missing module or a missing file; the non-zero exits are the documented ones (a failing `--check`, a `--diff` that finds changes, a refused input, the Rerun bridge left out) |
| Rebuilt bundle | `build.py --smoke` into a scratch folder | passed: start-up 2.13 s median of 3, cluster pool 19.1 s, experiments pool 7.7 s, self-test 46 ok in 82 s (load average about 15) |

Fixed:
- **Windows: every spawned pool worker failed.** `core/cluster/shards.py` imported the POSIX-only `resource` module at
  its top, and the pool's task imports it in each worker, so `--jobs N` (and the bundle smoke test's
  `--tolerance-mc --jobs 2`) would stop with `ModuleNotFoundError` on Windows. It now imports `resource` where it is
  used, and on Windows reads the peak working set (`K32GetProcessMemoryInfo`).
- **`--self-test` failed on a plain `pip install`.** Four lines (datasets, sim exports, tracking, validation) failed
  without pyarrow or jsonschema, which are optional extras, so README's first command after `pip install -e .`, and
  CI's oldest-Python job, failed. Such a line is now a warning that names the package and the extra
  (`selftest.OPTIONAL_PACKAGES`), and only when the package is really absent; the bundles carry those packages, so
  there the line must still be ok.
- **The start-up measurement used the real settings** on Windows (the registry) and macOS (the preferences), which the
  smoke test's XDG folders do not redirect; it now uses throw-away settings. It also writes its line to a file, as the
  console-less Windows launcher may have no stdout. The smoke test sets `TMP` and `TEMP` (Windows' names) as well as
  `TMPDIR`.
- **The home-folder scan** in `build.py` also matches forward slashes and JSON's doubled backslashes (Windows).
- **bundles.yml:** the self-test step names the executable with forward slashes and makes its temporary folder with
  Python (a Windows path, not Git Bash's `/tmp`); the draft release of a pre-release (0.1.0rc1) is marked
  `--prerelease`, so publishing it does not make it the repository's latest release.
- **ci.yml:** no process-wide `MUJOCO_GL=egl` (the render worker sets it itself; the suite and validation had only
  been measured without it); Python's UTF-8 mode (`PYTHONUTF8=1`), because 77 opens in the tests name no encoding;
  ruff over `examples` and `packaging` too, as the gate ran it.
- **Tests that cannot run on Windows** say so: the mock-sbatch tests (Slurm is Linux; the mock is a bash script) and
  the Linux installer's escaping test.
- **Comments:** `requirements-bundle.txt` named a `build.yml` that does not exist; the spec, the requirements and this
  page said MuJoCo renders through GLFW on Windows and macOS, but DexVantage renders only through EGL (above).

The regression tests are in `tests/test_packaging_ci_review.py`.

## Known limits

- Only the Linux x86_64 bundle has been built and run. Windows, macOS and Linux arm64 are built only by CI, and their
  spec branches are untested until CI runs:
  - the Windows icon and the Inno Setup step;
  - the macOS `BUNDLE` step, the plist and signing;
  - the PyQt6 6.7 pin on arm64.
- The Linux bundle built here takes opencv-python, the development environment's OpenCV, which carries its own Qt 5
  libraries. CI builds from `requirements-bundle.txt` with opencv-python-headless. The headless 5.0.0.93 wheel
  unpacks to 159.5 MB, against 195.8 MB for opencv-python as installed (measured from the wheel and site-packages).
  A CI bundle should therefore be about 36 MB smaller. That figure is an estimate from the wheel, not a measured
  bundle.
- A bundle runs only on a glibc at least as new as the build machine's. This one was built on glibc 2.43 and needs
  it: `objdump -T` over the bundle finds `GLIBC_2.43` symbol versions in the `libglib-2.0` and `libmvec` it copied
  from the build machine (and `GLIBC_2.42` in `libtinfo` and Python's `termios`), so it does not run on Ubuntu 24.04
  or 22.04. The glibc 2.35 figure in the README's install table holds for the bundles CI builds on Ubuntu 22.04,
  none of which has been built yet.
- The pyarrow prune and the GTK theme drop run on Linux only, because that is where they were measured. Windows and
  macOS keep all of pyarrow's libraries.
- The bundle's start-up time is measured to the first event-loop turn with the main window shown (offscreen). A
  real display adds the first paint.
