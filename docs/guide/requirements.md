# Requirements, the readiness scorecard and the variants compare

A worked example on the UR5e + Robotiq 2F-85 sample (a wrist D405 and an overhead D435). The model, the equations and
the measured validation are in `docs/models/requirements.md`.

## 1. The defaults on the sample, from the command line

    dexvantage --sample ur5e --check defaults scorecard.json

prints (29 s on the RTX 5080, most of it the grasp study):

    Readiness scorecard · Wrist D405 + overhead D435 (sample)
    Requirement                          Metric                    Value       Threshold         Margin     Status
    Contact visibility                   grasp.contact_visibility  100 %       ≥ 95 % (default)  +5 %       pass
    Camera to joint-state skew           sync.skew_ms              ≥ 54.43 ms  < 2 ms (default)  −52.43 ms  fail
    No controller over budget            bandwidth.over_budget     —           ≤ 0 (default)     —          not evaluated
    No cable pull or snag                cables.pull_snag          ≥ 3         ≤ 0 (default)     −3         fail
    No sensory blind gap before contact  dexhand.blind_gap_mm      ≥ 50 mm     ≤ 0 mm (default)  −50 mm     fail
      No controller over budget: The rig has no host controller: add the host on the Bandwidth page.
    Verdict: not ready (1 pass, 3 fail, 1 not evaluated)

and exits 1. Read it row by row:

- **Contact visibility** passes: 100 % of the reachable grasps have every fingertip at least half visible.
- **Skew** fails although the figure is a bound: the D405's clock figures are not all stated, so the time-sync study
  can only say "at least 54.4 ms", and at least 54.4 ms is never under 2 ms. Had the bound been "at least 1 ms", the
  row would read "not evaluated" with the sentence saying which figures are missing.
- **Controllers over budget** is not evaluated: the sample has no host. Add one on the Bandwidth page (Apply adds the
  suggested one) and run again.
- **Cable pulls and snags**: the suggested wrist cable's slack loop passes through the upper arm at three checked
  poses; "≥" because the cable's length and bend radius are not stated.
- **Blind gap**: the parallel gripper's pads are not seen for the last 50 mm of the approach (the whole scan, hence
  "at least"); with a tactile sensor or a camera that sees the pads it closes.

The defaults are DexVantage's starting points, labelled "(default)", not standards: write your own file.

## 2. Your own requirements file

    format: dexvantage.requirements
    version: 1
    requirements:
    - {name: Contact visibility, metric: grasp.contact_visibility, comparison: '>=', threshold: 90, unit: '%'}
    - {name: Skew, metric: sync.skew_ms, comparison: '<', threshold: 5}
    - {name: Two cameras at most, metric: rig.camera_count, comparison: '<=', threshold: 2}

`dexvantage --sample ur5e --check mine.yaml card.json` evaluates it; every metric id, with its unit and study, is in
the table of `docs/models/requirements.md` (or `dexvantage.core.requirements.catalogue()`). A threshold in the wrong
unit, an unknown metric or a broken file stop the gate with one sentence and exit 2. In CI:

    dexvantage rig.dexv --check requirements.yaml card.json || exit $?

## 3. The Requirements page

Analysis ▸ Requirements and variants (the Review workspace). **Add the defaults** puts the five defaults in the list;
pick a metric and **Add** for your own. Select a row to edit its name, comparison, threshold and note below the table:
each edit is one undo step, and changing a default's threshold unticks "DexVantage default". **Run** evaluates the
scorecard in the background: the headline reads, on the sample, "1 / 5 pass · not ready"; hover a status for the
reason. **Save as YAML…** writes the list as a file for `--check`; **Import…** adds a file's requirements.

## 4. Comparing variants

Press **New variant from this one** (a copy of the active variant, made active; its sensors keep their wiring),
turn the overhead D435 off in the copy on the Sensors and mounts page, and press **Compare variants**: the
table shows every requirement against each variant, the best per requirement marked "best", and the ranking by
requirements passed. With requirement "Cameras ≤ 1" the copy ranks first; **Preview** draws its sensors in the 3D view
next to the active ones, **Use the best variant** makes it active (one undo step).

## 5. Reproducing and plans

    dexvantage --reproduce card.json

re-runs the scorecard with the requirements recorded in the file, on the sample its command names, and prints
"Reproduces exactly: all 24 numbers are the same." A changed number is listed with its path, the recorded and the
re-run value, and the exit code is 1. A plan runs several steps and writes one summary:

    [{"study": "bandwidth", "settings": {"margin_warn": 0.5}, "out": "bw.json"},
     {"check": "requirements.yaml", "out": "card.json"},
     {"reproduce": "last-week.json"}]

    dexvantage rig.dexv --run-plan plan.json out/

`--reproduce out/plan-summary.json` later reproduces every step's result with that step's settings.
