# Publication kit (paper kit)

Modules: `dexvantage/core/publication.py` (the kit), `dexvantage/core/exporters/paper_kit.py` (the `paper_kit`
export format), `dexvantage/ui/pages/publication.py` (Export ▸ Paper kit…), `dv.publication` in
`dexvantage/api.py`, the assistant tool `get_methods_paragraph` (`dexvantage/assistant/tools.py`). Validation:
`dexvantage/validation/checks_publication.py` (model `publication`). Built on `core/cite.py` (BibTeX entries),
`core/figures.py` (SVG and PDF figures) and the report exporter (`core/exporters/report.py`: the results gathered,
the named figures read), described in [publication.md](publication.md).

## What the kit holds

| File | What |
|---|---|
| `methods.tex` | the methods paragraph: the software (DexVantage version and commit, MuJoCo, Python), each robot with its model source, commit and licence, each enabled sensor (maker and model, stream size and frame rate, field of view, mount site and pose), the studies with the settings the project stores for them and the validation status of each study's model, with `\cite{}` keys |
| `tables/sensors.tex` | the sensors: device, kind, stream, rate, FOV, mount, position |
| `tables/results.tex` | every named figure (core/metric_defs.py) of every result, with its unit and, where the result states one, its interval (the grasp study's bootstrap interval; P5 to P95 of a tolerance Monte Carlo) |
| `tables/scorecard.tex` | the requirements scorecard (value, threshold, margin, status) |
| `tables/experiments.tex` | an experiment's factors, its responses (runs, mean, SD, min, max) and its Sobol indices with their intervals |
| `tables/variants.tex` | the variants against the requirements (a Variants compare result), else each variant's sensors |
| `tables/statistics.tex` | the Statistics study's comparisons (when its result is present): the test, each side's figure (a rate in per cent, else the mean), B − A with its interval (percentage points for rates), the corrected p-value, the unit and the verdict word |
| `tables/cells.tex` | the multi-cell comparison (when present): each figure of merit per cell (a bound written ≥ or ≤, "not stated" where a cell does not state it) and the geometric gap index of each cell to the reference |
| `tables/runs.tex` | the experiment tracker's runs (when the 'tracking' result is present, or the project's tracking is on and its store holds runs): the newest 20 (a default), each with its start, study, status and headline figure |
| `figures/<id>-single.pdf/.svg`, `-double` | each result's per-item chart (the report's rule) and an experiment's Sobol indices, at 88.9 mm and 182 mm |
| `refs.bib` | exactly the entries the `.tex` files cite (core/cite.py, references `all`, filtered) |
| `appendix.tex` | the run plan's steps and the figures each regenerates, the seeds, the software versions, each result's SHA-256 |
| `reproduce/run_plan.json` | a `--run-plan` file: one study step per result (with its stored settings), the scorecard check, then this export again |
| `main.tex` | `\input`s the parts and includes the figures; `pdflatex main; bibtex main; pdflatex main` twice |
| `provenance.json` | the kit's contents, the trace of every number of the paragraph, the checks and their outcome, the provenance block |

## Every number traced

The paragraph is written through a `Writer` that records, for each number, where it is read from: a path into the
project's JSON (`project.variants[0].sensors[1].mount.pose.position[2]`), a library entry
(`library.intel-realsense-d405.streams[0].fps[2]`), a result, the software environment, the validation report, the
check registry (`registry.timing`: the number of registered checks), or a count (the length of a list there).
`verify_numbers` resolves every recorded place again and accepts a written number only when the source value
rounded to the digits shown equals it, `|v − t| ≤ ½·10^−d`, and requires every number of the paragraph to have such
an entry. Units are never converted in the prose: positions are metres and orientations unit quaternions (x, y, z,
w), as the project stores them, rounded to 3 and 4 decimals (DexVantage defaults); stated device figures are written
as the library states them (4 significant digits at most). Table rows end with `% dv-src: [...]`, one source per
number cell, and `verify_tables` checks each cell the same way (a named figure through `core.metrics.read`).

Frame rates and stream sizes are those the bandwidth model takes for the sensor: the rig's link when it names
streams, else its labelled default selection (the largest stream of the chosen imager at the listed rate nearest the
default 30 fps), and the paragraph says which. A field of view is the one the maker states for the first stream's imager (the sentence names that stream when it
lists several, so a depth camera's depth FOV is not read as its RGB stream's):
the stated mode of the stream's aspect when the stream is not the native image, else the imager's own for its native
image (said with the native size). A sensor with no library device is described by DexVantage's default camera, with
"DexVantage defaults, not a device's figures".

**Validation status.** Each study maps to the validation models that check it (`STUDY_MODELS`, e.g. coverage →
occlusion and optics). Given a validation report (`validation.json` of Help ▸ Validate or `--validate`), the
paragraph says how many of each model's checks passed; without one it says how many checks are registered and cites
up to two of their references. A check module that fails to import is named instead of a count.

**Tables** follow booktabs' guidelines: no vertical rules, no double rules, units in the column heading where a
column has one unit (Fear, booktabs v1.61803398 documentation, §2, read 2026-11-07). With the setting `siunitx`
(default off, a DexVantage default) numbers are `\num{…}`, quantities `\qty{…}{…}` and units `\unit{…}` with
siunitx's macros for %, °, m, mm, s, ms, g, W and its literal input for the rest (Mbit/s, px/mm; Wright, siunitx 3.6.7
manual §3.4, read 2026-11-07); without it `\dvnum{…}`, which each file defines as `\ensuremath{#1}`, so only booktabs
and graphicx are needed.

## Settings

`project.analysis.studies['publication']` (the page's Apply keeps them; options override): `sections` (methods,
sensors, results, scorecard, experiments, variants, figures, appendix; default all), `siunitx` (False), `widths`
(single and double), `pdf_figures` (True), `bibliography` (`bibtex` with the plain style, or `biblatex` with biber),
`title`. Export options besides: `results`, `results_dir`, `studies` (as the report: without results every study but
the meta-studies runs first), `validation`, `figure_specs` (more core/figures specs, such as a page chart's).

## Validation (model `publication`)

Measured 2026-11-07 on the SO-100 sample with the results of rig inventory, bandwidth, coverage, time sync and
motion blur run in the check, and a grasp result written in the check (so the interval column is exercised):

| Check | Reference (independent) | Tolerance | Measured |
|---|---|---|---|
| `publication.methods_numbers` | every number of `methods.tex`, found with a pattern written in the check, looked for in the source data read there: `Project.to_dict`, the library JSON files on disk, the results, `importlib.metadata` versions, the check registry, list lengths | 0 numbers not found (≥ 20 numbers) | 0 of 37 (a membership test over a large pool: it catches a number with no source anywhere, not one read from the wrong place, which `verify_numbers`' per-number trace and the tests' planted numbers cover) |
| `publication.bibtex_keys` | `\cite{}` keys read in the check against refs.bib read with checks_versioning's brace-counting parser | 0 | 0 (12 cited, 12 defined) |
| `publication.latex_structure` | a character scan written in the check: braces, `\begin`/`\end` order (verbatim skipped), `\input` and `\includegraphics` targets, no `|` column rule and no `\hline` | 0 | 0 problems in 10 files (both unit settings) |
| `publication.table_numbers` | each `\num`/`\dvnum` cell against `core.metrics.read` on the result or the value at its path read in the check | 0 cells beyond ½ unit of the last digit (or 5 × 10⁻⁴ of the value) | 0 of 60 cells in 30 rows |

The tests (`tests/test_publication.py`) do the same on the SO-100 and UR5e samples, with a number planted in the
paragraph and a table value changed to show that both are caught. **Compile:** no LaTeX engine (pdflatex, lualatex,
tectonic) was on PATH on the build machine, so the compile test skipped and only the structural check ran; where one
is installed the test runs `pdflatex`, `bibtex` and `pdflatex` twice on the kit of both unit settings and requires
`main.pdf`.

## Known limits

The paragraph describes what the project and the results state and nothing else: no task description, no claim about
the data the rig will record, no interpretation of a figure. Stream rates and sizes are the bandwidth model's
selection, not a recording's. A field of view is the maker's for the native image or the stated mode of that aspect,
not re-derived for a cropped stream (the studies' camera models do that). Only each result's per-item chart (chosen
by the report's naming rule, so not necessarily its most important figure) and an experiment's Sobol indices are
drawn, plus specs passed in. Characters outside Latin-1 with no TeX equivalent in `TEX_UNICODE` become `?`. The
LaTeX check is structural; the kit is compiled only where an engine is installed. Classic BibTeX styles do not know
`@software` and `urldate` (the date is also in `note`); choose biblatex for them. The headless API reads the Qt
version from the package metadata rather than importing Qt; the PDF figures need Qt's PDF writer (offscreen is
enough), without it only the SVGs are written and `main.tex` says so in a box.
