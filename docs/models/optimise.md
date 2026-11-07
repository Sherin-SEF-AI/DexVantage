# Optimiser and design agent

Spec §7.11. Code: `dexvantage/core/optimise/` (`nsga2.py`, `objectives.py`, `search.py`, `rig.py`, `designer.py`),
the studies `optimise` and `design_agent` (`core/studies/optimise.py`), the page Analysis ▸ Optimise
(`ui/pages/optimise.py`), the checks `validation/checks_optimise.py` (model id `optimise`).

Ported from Vantage `core/optimiser.py` (the current layout as the first candidate, the Pareto front of the feasible
candidates else of those closest to feasible, the changes said per camera, locked cameras kept) and
`core/rig_designer.py` (requirements as constraints, cost from stated prices only, a seeded search budgeted in
evaluations, options spread along the front and each explained, the current rig as the reference, accepting adds a
variant and records the decision). Vantage's single-objective evolution is replaced by NSGA-II; its automotive
objectives (blind ground, pixel density, perception range), reference vehicle rigs, lidars, radars and LLM brief are
not ported.

## The search: NSGA-II

Deb, Pratap, Agarwal and Meyarivan (2002), in NumPy, over mixed genes:

- **Continuous genes** (mount shifts in metres, turns in degrees, the design agent's placement numbers in [0, 1]):
  simulated binary crossover (SBX, Deb and Agrawal 1995) in its bounded form, probability `pc` = 0.9 per pair and 0.5
  per gene, distribution index η_c = 15; polynomial mutation (Deb and Goyal 1996), bounded, probability 1/n per gene,
  η_m = 20. These are the values of the NSGA-II paper's experiments and are labelled defaults (settings `pc`, `eta_c`,
  `eta_m`).
- **Categorical genes** (a camera kept or not, its device, its mount site): uniform crossover (each gene swapped with
  probability 0.5 when the pair crosses) and mutation to another choice drawn uniformly, probability 1/n.
- **Selection**: binary tournament on the crowded comparison (lower front, then larger crowding distance); the next
  population is parents and offspring sorted into fronts, filled front by front, the last front cut by crowding
  distance.
- **Constraints** by constrained domination (Deb 2002, §VI): feasible beats infeasible, the smaller total violation
  beats the larger, Pareto dominance between feasible designs.
- **Budget** in evaluations (the last generation is cut to fit); **seed**: one `numpy` generator, so a run is
  reproducible exactly; **cancel** between evaluations (the designs scored so far are kept and the result says
  "cancelled"); **progress** per evaluation.

Every objective is minimised internally (a higher-is-better metric is negated). A figure a study does not state for a
design is +∞ for sorting only (the design ranks below every design that states it); it is shown as "not stated" and
never replaced by a number. Along an objective it does not state, a design takes no part in the crowding distance (the
extremes are those of the stated values), so the crowded comparison never prefers it to a stated design.

## Objectives and constraints

Objectives are metric ids of `core/metric_defs.py` (the vocabulary the requirements scorecard, tolerance and experiments
share). The default list is the specification's:

| Objective | Metric | Unit | Better |
|---|---|---|---|
| Contact visibility | `grasp.contact_visibility` | % | higher |
| Depth-valid coverage | `depth.valid` | % | higher |
| Camera count | `rig.camera_count` | count | lower |
| Cost | `rig.cost` | USD | lower |
| Bandwidth margin | `bandwidth.margin` | Mbit/s | higher |
| Bracket printability | `brackets.not_ready` | count | lower |
| VLA token coverage | `vla.min_tokens` | tokens | higher |

An objective is left out, with its reason in the result, when the current rig's study does not state it (the
optimiser; a rig with no host has no bandwidth margin), when no design states it (the design agent), or, for the
cost, when any device the search can choose has no price in its library entry (no built-in entry states one, so the
cost is left out on every sample; a custom entry with a `price` block brings it in). Nothing is guessed.

Constraints are the project's requirements (`project.requirements`: metric, `>=`/`>`/`<=`/`<`/`==`, threshold) plus
settings `constraints`. The violation of one is its shortfall divided by max(|threshold|, a unit floor: 1 % for
percentages, 1 for counts, 0.1 ms, 1 mm…), so requirements in different units weigh alike; a figure not stated
counts as a violation of 1 (it cannot be shown to hold), and so does a figure stated only as a bound that does not
settle the test (the scorecard's "not evaluated"). A requirement whose metric is unknown is listed and not used.
A rig without a camera is infeasible.

## Scoring a design

Each design is a project (`core.factors.apply` returns a new one; the design agent adds a variant) scored by
`core.metrics.evaluate`, one study at a time (a study that stops with an unexpected error leaves its metrics
"refused" with a sentence instead of stopping the search). The studies run at **screening settings** (labelled
defaults, settings `screen`; `{}` uses the project's own): the grasp study on 40 grasps without sensitivity,
gallery or pre-contact steps and with 200 bootstrap resamples (0.5–1 s instead of about 40 s on the UR5e sample), the
brackets study on 12 joint samples without meshing. `confirm` scores the options shown again at the project's own
settings. Identical designs are scored once (a memo), and study results are kept between runs in a shared result
cache (256 results).

## The optimiser (Improve this rig)

Factors (`core.factors` paths; settings `factors` replaces them): for every enabled, unlocked camera, depth or tactile
sensor, `dx`, `dy`, `dz` within ±`move_m` (default 0.05 m) along its parent frame and `dpitch`, `dyaw` within
±`turn_deg` (default 15°) about its own axes; with `allow_remove` (default on) `enabled`; with `swap_models` (default
off) `model` among the library's devices of its kind. Locked sensors have no factor; a factor in the settings that
names one is refused with a note, and every design built is checked to leave the locked sensors' fields unchanged (an
assertion in the search, a validation check). The current rig is the first design. Accept sets the design's factors on
the active variant (one undo step) and refuses when the rig changed since the search ran.

## The design agent (Design new rigs)

A rig is up to `max_cameras` (default 3) slots; each slot's genes are: on or off, the device (settings `models`,
default every camera and depth entry of the library), the mount site, and three numbers in [0, 1] placing it there.

- **Static site** (settings `static_frames`, default the cage): looking at the first task volume's centre from
  azimuth −180° to 180°, elevation 20° to 85° and distance 0.45 to 1.2 m (defaults `elevation_deg`, `distance_m`).
- **Link site** (settings `links`, default each robot's gripper frame): riding on the link, looking at a point 3 cm
  beyond the fingertips (the mean of the fingertip frames at home, in the link frame; the approach axis runs from the
  link origin to it) from 2 to 15 cm behind the fingertips and 4 to 12 cm off the axis, at any angle around it
  (defaults `wrist_back_m`, `wrist_radial_m`).

Locked sensors are in every rig unchanged; the other current sensors are left out (each design is a complete rig)
unless `keep_unlocked`. Each new sensor whose device states its size gets the bracket type
`core/brackets.suggest_kind` gives. Accept adds the design as a new variant, makes it active and records the decision
in the settings and the design log (one undo step).

## Options and explanations

The options are up to `options` (default 5) designs of the first front of every design scored, with the current rig
taking part in the dominance comparison (a design the current rig dominates is not offered), the same figures counted
once, the best design on each objective first and then by crowding distance. Each option's explanation is built from
its figures only: what it changes or what the rig is, what it gains and gives up against the current rig (with the
differences), on which objectives it is the only best of the set, what it trades against its nearest neighbour on the
front (objectives scaled by their range over the options), which requirements it meets or misses and which figures
are not stated.

The headline is the number of designs offered with a status word: **better rig found** (a feasible design at least as
good as the current rig on every objective and better on one), **trade-offs only**, **current rig is best**, **misses
the requirements** (none feasible), with "(cancelled)" when stopped.

## Validation (measured 2026-11-07)

`dexvantage --validate OUT --models optimise`, 6 of 6 pass, 40.7 s:

| Check | Reference | Measured | Tolerance and why |
|---|---|---|---|
| `optimise.zdt1_igd` | analytic ZDT1 front f2 = 1 − √f1 (Zitzler, Deb and Thiele 2000) | IGD 0.00511 worst of 3 seeds (0.00511, 0.00475, 0.00488); even 100-point floor 0.00372 | 0.01: 1 % of the front's extent; about 2.7 times the best a 100-point set can do |
| `optimise.zdt2_igd` | analytic ZDT2 front f2 = 1 − f1² | IGD 0.00531 (0.00492, 0.00496, 0.00531); floor 0.00373 | 0.01, as above |
| `optimise.constrained` | CONSTR's front worked out from its constraints (f2 = 1/f1 for f1 ≥ 2/3, (7 − 9 f1)/f1 for 7/18 ≤ f1 < 2/3) | IGD 0.0212 (0.0210, 0.0212, 0.0209); floor 0.0203; largest constraint violation of the front 0, re-evaluated from the genes | 0.05: 0.6 % of f2's range (1 to 9) |
| `optimise.locks` | the locked sensors' own fields | 0 of 480 designs on 20 random rigs changed a locked sensor | 0 |
| `optimise.seed_reproducible` | the first run | 0 difference (NSGA-II archive of 600 evaluations; the rig optimiser's options); another seed differs | 0 |
| `optimise.figures_reproduce` | the studies run again without the cache on each option applied (UR5e sample) | 0 | 10⁻⁹ |

IGD is the inverted generational distance (Coello Coello and Reyes Sierra 2004): the mean distance from 1 000 points
of the analytic front to the nearest point found. NSGA-II ran with population 100, 25 000 evaluations (10 000 for
CONSTR) and the default operators. The self-test line `optimiser` runs ZDT1 at 6 000 evaluations (IGD 0.0316 against
0.05, 0.2 s) and checks the seed.

## Known limits

- The front is an approximation from a seeded evolutionary search within a budget of evaluations, not a proof of
  optimality; with the page's default budgets (96 and 120 designs) it is a coarse one. NSGA-II's crowding distance
  spreads well for two or three objectives and less well beyond (many-objective problems want reference directions,
  NSGA-III).
- Each design is scored by the studies themselves, so the search inherits their modelling limits; at the screening
  settings the grasp study's figure carries the sampling error of 40 grasps (use `confirm`, or `screen: {}`).
- A figure a study does not state ranks a design last on that objective, and a requirement it does not state counts
  as missed.
- The cost is the sum of stated device prices only (no cables, hosts, brackets, lenses sold separately, taxes); no
  built-in library entry states a price.
- The optimiser moves mounts within a box and turns within a cone of the current pose; it does not move a camera to
  another link. The design agent's mount sites are a sphere around the task volume and a cone around the gripper's
  approach axis, not every surface of the cell or the robot; the camera's roll about its view axis is not searched;
  the stream mode is the device's largest.
- The budget is in evaluations, not seconds: about 0.8 s per design on the UR5e sample for the optimiser and 2 s for
  the design agent (whose designs carry more studies' work, e.g. new brackets).
