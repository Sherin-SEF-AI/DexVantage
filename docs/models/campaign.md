# Data-collection campaign planner

Research tools part 2. Code: `dexvantage/core/campaign.py` (the model), `dexvantage/core/studies/campaign.py` (the
study `campaign`, batch flag `--campaign-saved OUT.json`), `dexvantage/batch_campaign.py` (`--campaign SPEC.json
[--campaign-out OUT.json]`), `dexvantage/core/exporters/campaign.py` (the exporter `campaign`),
`dexvantage/ui/pages/campaign.py` (Data ▸ Campaign), `dv.campaign` in `dexvantage/api.py`, the assistant's
`get_campaign`. Checks: `dexvantage/validation/checks_campaign.py` (model `campaign`, 6 checks). Ported in part from
Vantage `core/simcampaign.py` (a seeded design with each condition list shuffled per seed) and `core/fleet.py`
(settings clamped to limits, dates parsed and normalised); nothing automotive is kept.

## Inputs

The spec lives in `project.analysis.studies['campaign']`, or in a spec file for batch mode (schema
`dexvantage.campaign_spec/1`). Every key is optional; each default below is a DexVantage default and the result lists
the keys left at their default (`defaults`).

| Key | Default | Meaning |
|---|---|---|
| `tasks` | one per coordinated motion of `project.tasks.trajectories` (trajectories of different robots over the same time span are one bimanual motion), else one per grasp set, else "Task" | `[{name, trajectory, episode_s}]` |
| `variants` | the active variant | rig variants (uid or name) |
| `lighting` | one preset, "As designed" (the lights as they are) | `[{name, lights: {light uid or name: dim 0..1}}]` (0 = off) |
| `object_sets` | one set, every task object | `[{name, objects: [uid or name]}]` |
| `devices` | one level, every teleop device | `[{name, devices: [uid or name]}]` |
| `conditions` | the full factorial of the levels (at most 500) | an explicit list `[{task, variant, lighting, objects, device}]` by level name |
| `operators` | `["Operator 1"]` | names, or a count |
| `demos` | `{mode: entered, per_condition: 50}` | or `{mode: power, p_base, p_target, alpha: 0.05, power: 0.8, continuity: true}` |
| `keep_rate` | 1 (every episode kept) | your audit pass rate: what is recorded is the target / keep_rate, rounded up |
| `episode_s` | each task's trajectory duration, else 60 s | seconds per episode (every task) |
| `reset_s` | 30 s | between episodes |
| `changeover_min` | 5 min | setting a condition up at the start of a session and at each change of condition |
| `calibration` | `{every: day, minutes: 10, check_poses: 5}` | the calibration check (`day` or `session`) |
| `session_max_min` | 90 min | operator fatigue limit: the longest session |
| `operator_sessions_per_day` | 2 | operator fatigue limit |
| `sessions_per_day`, `break_min` | 4, 15 min | the rig |
| `day_start`, `day_hours` | 09:00, 8 h | the day |
| `start_date`, `weekdays_only` | none (days numbered), true | dates |
| `rounds` | `auto` | rounds per operator (below) |
| `seed` | 0 | shuffles the design |
| `format`, `compression_ratio` | `raw`, none | `raw`, `lerobot` or `rosbag`; `{value, source}` measured encoded / raw |
| `budget` | none | `{disk_gb, days, upload_mbps, upload_hours_per_day, operator_hours}` |

## The design

C conditions, O operators. Each operator records each condition m times in B rounds; in each round the operator
records every condition once, as a block of k = m / B consecutive episodes, in the order of one row of a **Williams
balanced Latin square** (Williams 1949):

    first row  0, 1, C−1, 2, C−2, 3, …            row i = (first row + i) mod C
    C odd:     those C rows and their C mirror images (2C rows)

In the C rows of an even order every condition is at every position once and follows every other condition exactly
once; in the 2C rows of an odd order every ordered pair is adjacent exactly twice. The S = O × B sequences take the
square's rows in turn (row s mod L, L the square's row count); the seed shuffles which condition each symbol stands
for and which sequence (operator, round) gets which row. Hence:

- every condition is recorded B times by every operator (always);
- every condition is at every position of a round equally often when S is a multiple of C;
- every ordered pair is adjacent equally often (first-order carry-over balance) when S is a multiple of L.

`rounds: auto` takes the smallest B ≤ m with S a multiple of L, else of C, else B = 1; the result's `design.text`
says which balance holds and how many sequences full balance needs. m is rounded up to a multiple of B (the notes say
how many more episodes that records).

**Sessions.** An operator's episodes are packed in order into sessions of at most `session_max_min` (whole minutes,
rounded up; a session's length is its changeovers, its episodes with their resets, and the calibration check when it
is per session). An episode never straddles two sessions. The packing is then repeated with the shortest limit that
needs no more sessions (bisection: in-order packing never needs more sessions for a longer limit), so no session is a
short remainder.

**Days.** Filled in passes: each pass offers one session to every operator who has one left, in an order rotated by
one operator each day, while the day has a slot (`sessions_per_day`), the operator is under
`operator_sessions_per_day`, and the session ends by `day_start + day_hours` (after the daily calibration check and a
`break_min` break after each session). The first session of each day carries the calibration check when it is daily.

## Storage

Per episode, for the condition's variant (core/studies/bandwidth.py, the bandwidth study's own figures):

    GB = Σ payload_mbps (enabled sensors) × ratio × (1 + container_overhead) × episode_s / 8 / 1000

with payload_mbps = width × height × fps × bits per pixel of the stated stream modes (or your measured rate),
container_overhead the bandwidth study's setting (3 %, a default), ratio 1 for `raw` and `rosbag` (raw image messages)
and, for `lerobot`, your measured `compression_ratio`. LeRobot encodes video with AV1 (libsvtav1, yuv420p, g 2, crf 30,
preset 12 at v3.0; `core/datasets/lerobot.py` ENCODER_DEFAULTS, from LeRobot's configs/video.py); no source states the
size that gives (it depends on the content), so without your ratio the size is the raw size and the result says so. A
sensor without a stated rate adds nothing: the storage is then "at least" (metric bound `at_least`).

## Demonstrations from a power analysis

The per-group size to tell success rate p1 from p2 with a two-sided test at α with power 1 − β (Fleiss, Levin & Paik
2003, normal approximation):

    n = [z_{1−α/2} √(2 p̄ q̄) + z_{1−β} √(p1 q1 + p2 q2)]² / (p2 − p1)²,      p̄ = (p1 + p2) / 2,  q = 1 − p

and with the continuity correction (the default; Fleiss, Tytun & Ury 1980): n_c = n / 4 · (1 + √(1 + 4 / (n |p2 −
p1|)))². 0.5 against 0.7 at α 0.05 and 80 % power: n = 93 without, 103 with the correction. The figure comes from
core/stats.py's `n_two_proportions` (its normal method, the same formula, with the power it reaches at n) when that
module is there; the formula in `power_n` is the fallback, and the result keeps both (`n`, `formula_n`).

## Budget and status

Each budget figure given is compared with the plan: storage against `disk_gb`, days against `days`, the busiest
day's upload time (its GB × 8000 / upload_mbps / 3600 h) against `upload_hours_per_day` (24 h without one), the
busiest operator's hours against `operator_hours`. Within: good; above 90 % of the limit (TIGHT, a default): warning,
"tight"; over: critical, "over budget". No figure given: warning, "no budget set". A storage that is only "at least"
turns a good status into "storage incomplete". The advice says what to change for each figure over or near its limit.

## Outputs

The result (`dexvantage.campaign/1`): `conditions` (each with its levels, episode length and its source, GB per
episode, recorded and expected kept episodes), `design`, `sequences`, `sessions` (day, date, start and end, operator,
blocks, GB and the checklist), `days`, `totals`, `budget`, `headline` ("N episodes over D days, S GB" with its status
word), `advice`, `notes`, `assumptions`, `defaults`, `references`.

The **checklist** of a session: the calibration check (the hand-eye plan's camera and board, `check_poses` of its
planned poses, its requirement `need_mm` and `need_deg`, labelled defaults when unset; the last solved session); per
item one line when every condition of the session shares it, else one per condition: the rig variant, the lighting
preset (each light on at its dim or off), the object set, the teleop devices, each camera's exposure, gain and
polariser as the motion-blur and lighting studies' settings hold them (what their Apply writes; "not set"
otherwise); the disk to free; the blocks to record.

Exports: CSV (one row per block), iCalendar (RFC 5545: one VEVENT per session with UID, DTSTAMP in UTC, floating
DTSTART and DTEND, SUMMARY and the checklist as DESCRIPTION; TEXT escaped, CRLF, lines folded at 75 octets without
splitting a UTF-8 sequence), a printable HTML page (one day per printed page). Apply writes `compact_plan` (the
schedule without the checklists, its digest) into `analysis.studies['campaign']['plan']`, one undo step; the study
ignores that key.

Metrics: `campaign.episodes` (headline), `campaign.days`, `campaign.storage_gb` (bound `at_least` when incomplete),
`campaign.rig_hours`, `campaign.kept_per_condition`.

## Validation (measured 2026-11-07, `checks_campaign.py`)

| Check | Reference | Tolerance (why) | Measured |
|---|---|---|---|
| `campaign.williams_square` | the definition, counted by brute force, orders 1–12 | 0 (a combinatorial property) | 0 violations |
| `campaign.schedule_balance` | counts from the exported CSV, parsed with `csv`, on 40 seeded random campaigns (16 quick) | 0 (exact by construction where the design says so) | 0; 35 of 40 exact in position, 27 in carry-over |
| `campaign.session_limits` | each campaign's limits, re-checked from the session times | 0 (hard limits) | 0 in 383 sessions |
| `campaign.storage_vs_bandwidth` | the bandwidth study's `gb_per_episode` and `gb_per_campaign` (UR5e raw; dex-hand LeRobot at ratio 0.08): a consistency check, since the campaign reads them; and, independently, GB per episode added up again in the check from each selected stream's width × height × fps and the library file's bits per pixel | 10⁻⁹ relative (the same arithmetic in another order) | 0 |
| `campaign.ics_roundtrip` | a strict RFC 5545 reader written in the check (8 calendars with `;`, `,`, `\`, non-ASCII) | 0 (a format round trip) | 0 in 34 events |
| `campaign.power_exact` | exact power of the pooled z-test (and the Yates-corrected one) by enumeration over both binomials (SciPy) | 0.04 power (the normal approximation is furthest off near 0 or 1) | 0.033 (0.8 against 0.95: 0.833 for 80 %, an overshoot); the largest shortfall below the target power is 0.0047 (0.4 against 0.6, 80 %, with the correction); 0.016 in the quick run |

Self-test line `campaign` (Williams squares, 16 campaigns' balance and limits, 34 calendar events read back): 0.5 s.

## Known limits

Every episode is taken to last episode_s + reset_s (real demonstrations vary, fail and are re-done; the keep rate
covers discarded episodes on average, not their time); a trajectory's duration is the robot's scripted motion, which
teleoperation may be slower than; operators are taken as equally fast; the fatigue limits are defaults, not ergonomics
findings; position and carry-over balance are exact only for the numbers of sequences the design states, and the
balance is of blocks, not single episodes; conditions are not balanced across days (operators rotate their order;
a day holds what fits); days are filled greedily, which is not always the fewest days when sessions differ; a
compressed format's storage is the raw size without a measured ratio; a sensor without a stated data rate adds
nothing; upload is a steady rate without protocol overhead; times are local without a time zone (floating calendar
times, RFC 5545 §3.3.5); holidays are not known; the power analysis is for two independent proportions, not several
conditions at once, paired designs or learning curves. `--reproduce` of a `--campaign SPEC.json` result re-runs the
project's stored spec, not the file's (store the spec with Open spec… or Apply to reproduce it).

## References

- E. J. Williams, "Experimental designs balanced for the estimation of residual effects of treatments", Australian
  Journal of Scientific Research, Series A 2(2), 149–168, 1949, https://doi.org/10.1071/CH9490149 (read 2026-11-07).
- J. L. Fleiss, B. Levin and M. C. Paik, Statistical Methods for Rates and Proportions, 3rd ed., Wiley, 2003.
- J. L. Fleiss, A. Tytun and H. K. Ury, "A simple approximation for calculating sample sizes for comparing
  independent proportions", Biometrics 36(2), 343–346, 1980, https://doi.org/10.2307/2529990 (read 2026-11-07).
- B. Desruisseaux (ed.), Internet Calendaring and Scheduling Core Object Specification (iCalendar), RFC 5545,
  September 2009, https://www.rfc-editor.org/rfc/rfc5545 (read 2026-11-07).
