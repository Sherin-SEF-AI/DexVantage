# Contributing to DexVantage

Thank you for helping. DexVantage is a research tool: people plan real camera rigs and read real figures from it, so
the bar is that every number it shows can be traced to a source or a measurement. Read
[`docs/DEVELOPING.md`](docs/DEVELOPING.md) before changing anything; this page is the short version for a pull
request.

## Set up

    git clone https://github.com/Sherin-SEF-AI/DexVantage.git
    cd DexVantage
    python -m venv .venv
    .venv/bin/pip install -e ".[test]"            # Linux only: DexVantage supports no other system

Add the extras your change touches (`lerobot`, `bags`, `usd`, `rerun`, `step`, `research`, `tracking`, `assistant`,
`dev`; `pyproject.toml` says what each is for). Tests that need a missing extra skip and name it. CI installs the
versions in `.github/ci-constraints.txt`.

On Linux without a desktop, install Mesa's EGL and the Qt libraries: `.github/scripts/apt-gl-qt.sh` lists them
(Ubuntu 24.04 names; on 22.04 it installs `libglib2.0-0` in place of `libglib2.0-0t64`).

## Run the tests

    QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q tests/test_x.py      # the file you changed
    .venv/bin/ruff check dexvantage tests scripts examples packaging
    QT_QPA_PLATFORM=offscreen .venv/bin/python -m dexvantage --self-test
    QT_QPA_PLATFORM=offscreen .venv/bin/python -m dexvantage --validate out/validation --quick

- The whole suite in one process peaked at 6.4 GB with its render workers (measured on a full run, see
  `docs/BUILD_STATE.md`), so run it under a memory cap (`docs/DEVELOPING.md`). CI runs it in six shards
  (`python scripts/ci_shards.py --shards 6 --index N --run -- -q`); you can run a shard the same way.
- Tests use throw-away settings, data and cache folders (`tests/conftest.py`): no test reads your own configuration,
  keyring or API keys, and no test touches the network. Keep it that way: fake providers, local fixtures.
- Tests that render MuJoCo scenes are marked `@pytest.mark.mujoco_render`. They need EGL (Linux with a GPU driver, or
  Mesa's llvmpipe without a GPU); elsewhere they skip and say why. `DEXVANTAGE_TEST_RENDER=skip` runs the suite as
  on a machine without a renderer, `DEXVANTAGE_TEST_RENDER=require` makes a missing renderer a failure (CI on
  Linux). A new test that renders, directly or through a fixture (synthetic episodes, the ground-truth study, the
  rendered validation checks), gets the marker.
- No skipped test without a stated reason, and no failing test.
- `python scripts/get_actionlint.py` fetches the pinned actionlint (checksum-verified, into `out/tools`); then
  `tests/test_ci_workflows.py` also lints the workflows with it.

## The rules a pull request must keep

1. **No invented figures.** A hardware or robot figure (resolution, FOV, baseline, min-Z, data rate, power, mass,
   joint limits, inference times, input sizes) comes from the maker's datasheet, an official model file, a model card
   or a paper, or a measurement; the entry's `note` records the source and the date read. A figure the source does
   not state is `None` (shown as "—", "not stated by the maker"). Never fill a gap with a guess.
2. **Defaults are labelled as defaults**, never as "industry standards".
3. **Every model has a validation check** against an independent reference with a stated tolerance and why
   (`dexvantage/validation/checks_<area>.py`). Report the measured agreement honestly, also when it is poor.
4. **Uncertainty is shown** as `value [low–high]` with the interval named.
5. **Known limits**: a study's docstring and its docs section say what it does not model.
6. **Licences**: check before bundling any model, mesh, dataset or asset, and record it in `THIRD_PARTY.md`.
   MuJoCo Menagerie licences differ per model.
7. **No personal paths, user names, host names, e-mail addresses or keys** in any file.
8. **Errors are explained in plain sentences**, never a raw traceback in the UI (log it; studies raise
   `core.jobs.StudyError("plain sentence")`).
9. **Plain English in the UI**: sentence case, short, numbers with units, no marketing words.
10. **No AI attribution** in commits, pull requests or files.

Style: Python 3.10 or newer, type hints on public functions, ruff (line length 120), vectorised numpy. New heavy
dependencies go into an optional extra, imported lazily with a clear message when missing. The extension points
(studies, pages, checks, metrics, exporters, plugins) are in `docs/DEVELOPING.md`; shared files such as
`ui/mainwindow.py` and `core/project.py` change only when an extension point cannot do the job.

## Pull requests

- One change per pull request, with tests. Say what you measured and how, and paste the test and validation counts.
- Update the docs (`docs/models/<area>.md`, `docs/guide/<area>.md`) and `CHANGELOG.md` under `[Unreleased]`.
- CI must pass: ruff and actionlint, the test shards on Ubuntu 24.04 (Python 3.12 and 3.14), the self-test and the
  quick validation run on each, and the Python 3.10 install check. DexVantage supports Linux only, so CI runs no
  Windows or macOS job.
- Releases are made by the maintainer only. A version tag builds the bundles and a **draft** release
  (`.github/workflows/bundles.yml`); the maintainer decides whether to publish it.

## Reporting a problem

Use the issue templates. For a wrong figure, give the source that disagrees (a datasheet page, a model file, a
measurement and how it was made). Never paste API keys, tokens or private data into an issue.
