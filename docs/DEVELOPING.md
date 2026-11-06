# Developing DexVantage

Read this before changing anything. The full product specification lives outside the repository (the build prompt);
`docs/BUILD_STATE.md` says what is done, what is next and the decisions taken.

## Rules that are never broken

1. **No invented figures.** Every hardware or robot figure (resolution, FOV, baseline, min-Z, shutter, data rate,
   power, mass, joint limits, link masses, policy inference times, VLA input sizes) comes from the maker's public
   datasheet, an official model file, a model card or paper (cited), or a user measurement. Record the source URL or
   document and the date read in the entry's `note`. A figure the source does not state is `None`, shown as "—" with
   the tooltip "not stated by the maker". Never fill a gap with a guess. If a web page cannot be fetched, leave the
   figure `None` and say so; do not reconstruct it from memory.
2. **Defaults are labelled as defaults** (thresholds such as 95 % contact visibility, σ_d = 0.08 px) — user settings
   with a documented default, never "industry standards".
3. **Every model has a validation check** comparing it with an **independent** reference (a closed form written out
   again, another library, a published table, brute-force ray casting, Monte Carlo, MuJoCo renders) with a stated
   tolerance and why. Checks live in `dexvantage/validation/checks_<area>.py` and register with `@check(...)`; the
   suite discovers them. Measured agreement is reported honestly, including when it is poor.
4. **Uncertainty is shown** as `value [low–high]` with the interval named, wherever inputs are uncertain.
5. **Known limits**: every study module docstring and its docs section has a "Known limits" paragraph.
6. **Licences**: check before bundling any robot model, mesh, dataset or asset; record source URL, licence,
   version/commit and date checked in `THIRD_PARTY.md`. Bundle only redistributable assets; MuJoCo Menagerie
   licences differ **per model** — read each model's LICENSE. Non-redistributable assets download on first use with
   the licence shown and consent.
7. **No personal paths** (`/home/…`, `/Users/…`), user names, host names, e-mail addresses or keys in any file.
   Vantage (the read-only reference implementation at `../vantage`) is referred to as "ported from Vantage
   `core/x.py`". Never modify, commit to or push the Vantage repository.
8. **Never** read or write the user's real config, keyring or API keys; no real LLM API calls in tests (fake
   providers). Don't kill processes you did not start. No system-Python installs: use `.venv/bin/pip`.
9. **No git commits, pushes, tags or releases** from agents — the lead engineer commits. Never add any AI
   attribution (no Co-Authored-By trailers).
10. **Plain English in the UI**, sentence case, short; numbers in IBM Plex Mono with units; say what a number means
    and what to do about it. No emojis, no marketing words. Status uses the four STATUS colours *with a word*.
11. **Errors are explained, never a raw traceback** in the UI (log it with `traceback.print_exc()`); studies raise
    `core.jobs.StudyError("plain sentence")` for inputs they cannot work with.

## Running things

    cd <repo>; .venv/bin/python -m pytest -q tests/test_x.py         # one file while developing
    systemd-run --user --scope -q -p MemoryMax=7G env QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q tests
    QT_QPA_PLATFORM=offscreen .venv/bin/python -m dexvantage --self-test
    QT_QPA_PLATFORM=offscreen .venv/bin/python -m dexvantage --validate out/validation
    .venv/bin/ruff check dexvantage tests scripts

Tests run offscreen with throw-away XDG folders (tests/conftest.py provides `app`, `win`, `settings`, `pump`).
Heavy scripts run memory-capped as above. Generated output goes to `out/` (ignored).

## Frames and units

SI units; joints in radians (shown in degrees); poses as position + quaternion (x, y, z, w) (`core/frames.py`);
world = primary robot base, REP-103 (x forward, y left, z up); camera optical frame z forward, x right, y down;
MuJoCo quaternions are (w, x, y, z) — convert with `frames.quat_to_mujoco` / `quat_from_mujoco`.

## Extension points (add features without editing shared files)

| What | Where | How |
|---|---|---|
| A study | `dexvantage/core/studies/<name>.py` | `@jobs.study('id', 'Title', quick=False, batch='flag-name', needs_robot=True)` on `fn(project, ctx) -> dict`; use `ctx.report(f, text)`, `ctx.partial(doc)`, `ctx.cancelled()`; JSON-able output; settings in `project.analysis.studies['id']`. The batch flag `--flag-name OUT.json` appears automatically. |
| An analysis page | `dexvantage/ui/pages/<name>.py` | subclass `pages.base.StudyPage` (STUDY, TITLE, build_body, show_result, apply/preview, overlays(vp)), then `register('id', 'Title', Cls, menu='Analysis', icon=..., action_id=<existing placeholder id or None>, workspace=...)`. The window adds the tab and the menu command. Keep constructors cheap. |
| 3D scene content | `ctx.overlay_sources.append(fn)` | `fn(vp) -> [LineDraw | TriDraw]` (ui/viewport3d.py). Page overlays come from `StudyPage.overlays(vp)`. |
| A validation check | `dexvantage/validation/checks_<area>.py` | `@check(id, model, title, quantity, reference, tolerance, unit, refs=(...))` returning `Outcome(error, detail, ...)`; model ids are in `validation/suite.py` MODELS; references in `validation/references.py` (add keys there). |
| A self-test line | `dexvantage/selftest.py` CHECKS | add `(name, fn)`; keep it fast (< 2 s). |
| Docs | `docs/models/<area>.md` | the model, equations, references, validation result, known limits; the overview links them. |
| User guide | `docs/guide/<area>.md` | a worked example on a sample project. |

Shared files (`ui/mainwindow.py`, `core/project.py`, `batch.py`, `ui/scene.py`, `ui/store.py`) change only when an
extension point cannot do the job; keep such edits small and say what you changed.

## Style

Python ≥ 3.10, type hints on public functions, ruff line length 120, numpy vectorised maths, no new heavy
dependencies in the core (optional extras in `pyproject.toml`, imported lazily with a clear message when missing).
Port from Vantage where the specification says so: copy, strip the automotive assumptions, keep and adapt its tests,
note the origin in the module docstring.
