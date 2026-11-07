# UI responsiveness and the review round's analysis fixes

What changed in the review round of 2026-11-07 on the UI and analysis side, the rules each fix follows, how it was
measured and what it does not do. The worked example is in [the guide](../guide/ui-fixes.md).

## Hidden pages wait until they are shown

**Model.** The window builds every analysis page through `ui/pages/_gate.build_page(cls, ctx)`. The page gets a
view of the context whose store and analysis controller are the real ones except for six signals:
`store.changed`, `pose_changed`, `selection_changed`, `project_replaced`, `analysis.finished` and `failed`. A
per-page gate relays them at once while the page is shown (or while the window itself is not shown: scripts and the
self-test read pages of an unshown window), and holds them while it is hidden. On the page's Show event the held
signals are replayed in their order, coalesced:

* consecutive `changed(sections)` calls are joined into one with the union of the sections;
* only the last `finished` or `failed` of each study is kept (a later one replaces an earlier one, at the later
  position, so a change followed by a fresh result still ends with the result);
* only the last pose per robot, the last selection and the last project replacement are kept.

The page then ends in the state it would have reached by following every signal. `page_context()`, `apply()`,
`preview()`, `export()` and `explain()` replay what is held before they run (StudyPage wraps them, in subclasses
too), so the assistant and the MCP bridge read a hidden page up to date; `StudyPage.flush_pending()` does it for any
other reader. A page taken by name, `window.pages[pid]` (a `PageDict`), follows every signal from then on: a script
or a test that holds a page reads it as it goes. The window's own code paths use `pages.get`, `in`, `values()` and
`items()`. A page class with `DEFER_HIDDEN = False` is never held.

**Adding a robot.** The Menagerie and robot-catalogue import compiled the model on a worker but then compiled it
again on the UI thread (the shared model cache is keyed by the robot as it is stored, and the robot was renamed and
made primary there), and the frame button took the mesh bounds of every robot three times. The worker now names
the robot and sets its primary flag as the import will, builds its kinematic model and its home-pose bounds into
`core/robot/model`'s shared cache, and `RobotModel.bounds(q)` keeps the last pose's bounds.

**Measured** (offscreen, `QT_QPA_PLATFORM=offscreen`, RTX 5080 machine under the other agents' load, the software
3D view's paint left out because it stands in for the app's GL view in tests). "Edit" is one `store.edit` nudging a
sensor 1 mm, synchronous; "event" the longest single `processEvents()` while the studies finish.

| Sample | Edit before | Edit after | Longest event after an edit | Apply offered (after) | Add the G1, UI thread |
|---|---|---|---|---|---|
| ALOHA 2 | 32-39 ms | 3.7-5.4 ms | 1.4-2.6 ms | Tasks 12 ms, Bandwidth 11 ms, Runs 7 ms | 171 ms before (77 ms compile + 78 ms bounds); 10.6 ms after |
| Panda bimanual | 32-35 ms | 3.7-4.0 ms | 0.9-1.2 ms | Tasks 13 ms, Bandwidth 14 ms, Runs 8 ms | 10.4 ms |
| G1 | 28-30 ms | 3.6-5.5 ms | 1.7-2.5 ms | Tasks 14 ms, Bandwidth 10 ms, Runs 6 ms | 12.7 ms |

"Before" is the same build with every page following every signal (`DEFER_HIDDEN = False`), which is what the
window did. The longest event after Tasks' Apply is 31-40 ms (the Tasks page itself rebuilding its lists, shown).
In a second pass every page's own study was run first (Run, up to 120 s each; Optimise, Tolerance and on the
Panda pair Teleop took longer and were cancelled) and every Apply then offered was timed, synchronous, with the
longest event after it:

| Sample | Applies offered | Slowest Apply (sync) | Longest event after an Apply |
|---|---|---|---|
| ALOHA 2 | 13 (tasks, grasp, dexhand, action chunk, bandwidth, cables, hand-eye plan, brackets, motion blur, campaign, optimise, tolerance, runs) | 17.3 ms (cables) | 30.1 ms (tasks) |
| Panda bimanual | 17 (the above, plus depth, ground truth, time sync and canonical view) | 57.4 ms (tasks) | 38.2 ms (tasks) |
| G1 | 12 | 23.8 ms (tasks) | 44.0 ms (tasks) |

Every one is under the 100 ms budget. The other pages offer no Apply on these samples (their studies give no
suggestion there, or refuse).

**Validation.** `tests/test_ui_fixes.py` times an edit (median of five), the longest event while the studies finish
and every Apply the pages offer on ALOHA 2, the Panda pair and the G1, and adding the G1 through the Menagerie
dialog, against stated margins: an edit under 50 ms (half the 100 ms budget, about ten times what is measured, for
a loaded machine), any single event, any Apply and the G1's UI-thread part under 100 ms (the budget). The gate's
ordering and coalescing are checked on a stub, and a hidden Sensors page is checked to hold four edits and to show
the right table once shown.

**Known limits.** A page built directly (`SomePage(ctx)`) is not gated. The first show of a page after many edits
costs one refresh of that page on the UI thread (not one per edit). Signals other than the six (progress, partial
results, the theme) still reach hidden pages. A page taken with `window.pages[pid]` is never held again in that
window. The timings exclude the 3D view's own drawing.

## A refused study logs one line

A study that refuses its inputs (`StudyError` and the other explained errors, `ui/store.EXPLAINED_ERRORS`) is logged
by `ui/store.log_failure` as one INFO line with its sentence (`dexvantage.ui.store`, "coverage refused: Add a camera
first."); its traceback is logged only at DEBUG. An unexpected error is logged at ERROR with its traceback. Nothing
is printed to stderr by the window for a refusal any more (it printed the whole message with `print`). The page and
the status bar show the sentence as before.

## Experiments: figures stated only as bounds are censored

A study may state a figure only as a bound: `{'value': v, 'bound': 'at_least' | 'at_most'}` (core/metrics.read).
The experiments' analysis used v as if it were exact. The rule now (core/experiments.py, `CENSORED_RULE`):

* **Statistics.** A bounded value is censored: it is left out of the mean, standard deviation, minimum, maximum,
  main effects and Sobol indices (as a missing value is) and counted per response
  (`analysis[metric]['censored'] = {'at_least': k, 'at_most': k}`), with a note. Substituting the bound would bias
  every statistic toward it (Helsel 2012).
* **The best run.** In the "higher is better" sense (lower-is-better figures negated), each run has a known figure
  and a best possible figure: exact v gives (v, v); a bound on the side of better (at least, for higher-is-better)
  gives (v, +∞); a bound on the other side gives (none, v). The best run has the best known figure; a run with no
  known figure is not eligible. The pick is **settled** when no other run's best possible figure exceeds it;
  otherwise the runs that could beat it are given as `contenders`.
* **Against the rig as it is.** `improves` is True when the best run's known figure beats the rig's best possible
  one, False when the best run's best possible figure does not beat the rig's known one, and None otherwise ("not
  settled": the headline says so and Apply is not offered). This is the rule `core/requirements.decide` applies to a
  requirement and that the tolerance Monte Carlo and the optimiser use for a bound that does not settle a test.

The runs table shows a bounded response as "≥ v" or "≤ v". **Validation:** `experiments.censored_best` (new,
`validation/checks_censored.py`) draws, for 60 (quick) or 300 random designs of 12 runs with 10, 30 or 60 % of the
figures bounded, true figures consistent with every statement (the intervals' corners included) and counts the
statements they contradict: a settled pick that is not best, an `improves` of True or False that does not hold, a
None the bounds would have decided. Tolerance 0 (a decision rule). Measured: 0 contradictions in 60 (quick) and in
300 designs (15 and 69 settled picks, 49 and 249 decided verdicts, 11 and 51 withheld). The same check fails (61 contradictions) when a bound is read as exact.
**Known limits:** censoring drops information (a run that is "at least 9" counts for nothing in the means); no
Kaplan-Meier or Turnbull estimate is made.

## The canonical view's headline past the image

When the residual parallax (P95 at the worse depth extreme) exceeds the canonical camera's image width, the study's
headline carries `beyond_image: true` and `image_width_px`, its advice starts "The residual parallax is larger than
…'s image (bound N px P95 against a W px wide image)", and the page shows **larger than the image** with
"(bound 11 878 px)" beside it, with the status word (critical, as before). The figure in the result is unchanged.

## Custom devices of every library kind

Library ▸ New custom device… now makes teleop devices, policies, lights and cables as well as cameras, depth
cameras, tactile sensors and hosts, with the schema fields of each kind's block (`resources/library.schema.json`):
a teleop device's kind, the followers it leads and whether it has their joints, its rate, latency, reach, accuracy,
resolutions, range and feedback (the mapping follows the kind; the workspace volume is not entered here); a policy's
every `core/policies.FIELDS` entry and the hardware its inference time is for; a light's type and photometric figures;
a cable's type, diameter, bend radii, cycles, lengths, connectors, data rate and ratings. A figure left unticked is
null ("not stated"). A ticked teleop or policy figure is stored as `{value, source: 0, quote: "user-entered …"}`
and the note starts "User-entered custom device." A teleop entry is checked with `core/teleop_devices.entry_problems`
and a policy with `core/policies.entry_problems` before the schema, so the messages are plain. Every built-in teleop,
policy, light and cable entry starts a valid custom entry (tested). The Library's right-click offers "Start a custom
device from this one…" for every kind the dialog makes.

## Surfaces in the Outliner

The Outliner's Workspace group lists the support surfaces (size and height); selecting one selects `('surface',
uid)` in the store (`ProjectStore.selected()` returns its dict, and an undo keeps the selection), which the
Workspace page follows (and the other way round); the Inspector shows its height, size and material with "Edit on the
Workspace page"; Delete removes it (one undo step).

## Tolerances declared as applied

`tolerance.linear_propagation` and `tolerance.sampler_ks` apply a band set by the number of draws. They declared
0.05 and applied 0.052 / 0.020 and 0.061 / 0.037 (quick / full). The suite now takes `tolerance_quick` in `@check`
(additive: `Spec.tolerance_quick`, `suite.declared_tolerance(spec, quick)`, used when a check returns no tolerance of
its own; `validation.catalogue()` lists the tolerance of the run its settings describe, plus `tolerance_full` and
`tolerance_quick`). The two checks declare 4 √(2 / 4n) = 0.0200 / 0.0516 and the critical KS D at α = 0.001 / 30 =
0.0371 / 0.0606 (full / quick), the bands they apply. The KS check's docstring said 18 marginals; it tests 30 (two
distributions × two mounts × six components, and the calibrated sensor's six), which the applied band always used.

## FOV notes

The library entries whose stated angles break the lens-geometry bounds of `library.fov_shape` by more than 2° (all
within its 3° tolerance) say so in their notes, with the figure: Leopard LI-OV2312 110H (2.92°), e-con See3CAM_37CUG
(2.85°), Orbbec Femto Bolt and Femto Mega in their 10:9 mode (2.78°), ZED X One GS narrow (2.25°), and ZED X and ZED X
Mini narrow (2.09°). Each note names the bound broken, the angle against its limit and the check. A test keeps every
such entry's note in step with the check.

## Where --check writes

`--check REQUIREMENTS` without an output file still writes `scorecard.json` into the current folder (as documented);
it now prints the absolute path and says that it is the default.

## The plugin template's install test

`setuptools==84.2.1` (MIT) is declared as the `dev` extra in `pyproject.toml` and installed in `.venv` offline from
pip's own cache: the cached wheel's SHA-256 (ca140b45…89c2) equals the one PyPI lists for
`setuptools-84.2.1-py3-none-any.whl` (https://pypi.org/pypi/setuptools/84.2.1/json, read 2026-11-07; released
2026-10-20), installed with `--no-index --require-hashes`. `tests/test_plugins.py`'s real `pip install --target`
test of `examples/plugin-template` now runs instead of skipping (13 passed).

## References

* D. R. Helsel, *Statistics for Censored Environmental Data Using Minitab and R*, 2nd edition, Wiley, 2012
  (`helsel2012`).
* JCGM 100:2008 and the Kolmogorov-Smirnov references of the tolerance checks, as before.
