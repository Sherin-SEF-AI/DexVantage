# Build state

The resume point for the next session: what is done, what is next, open issues, and the decisions taken. Updated at
the end of every phase.

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

**Next: Phase 1 (robots and kinematics)**
- URDF and MJCF import through MuJoCo's compiler (USD optional), mesh check, joint/fingertip/contact-pad taggers.
- An independent product-of-exponentials FK; IK (damped least squares, joint-limit avoidance, seeds); collision
  distances; manipulability.
- MuJoCo Menagerie import with the per-model licence shown and recorded (spec §9.4).
- Joint sliders, the pose scrubber driving `q`, IK drag; link-mounted sensors posed at `q`.
- Samples: SO-100/SO-101 and UR5e + Robotiq 2F-85, after the licence check.
- Kinematics checks: FK against closed-form DH (UR5e, Panda), Jacobian against finite differences, IK round trips,
  2R manipulability, pinocchio when installed.

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
  QOpenGLWidget path needs a desktop check at each review gate.
- `site/` (website) and `packaging/` are Phase 11.
