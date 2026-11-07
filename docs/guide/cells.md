# Comparing cells: a worked example on the UR5e sample

Three benches were built from the UR5e + Robotiq 2F-85 sample's design, and a fourth lab runs an SO-100. This compares
them on their figures of merit, the geometric gap a policy trained on one meets on another, whether their recordings
pool for training, and how far the built copies drifted from the design. The model is in `docs/models/cells.md`.
Figures below are from runs on 2026-11-07.

The cells:

- **UR5e + Robotiq 2F-85 sample**: the design, open in the window (cell 0, the reference).
- **UR5e bench B**: the overhead D435 was mounted tilted 10° about its optical frame's x axis (the image's horizontal
  axis; the body frame's x is the optical axis itself), and its wrist D405 was calibrated
  on the Hand-eye page (a solved session, not applied) 2.5 mm off along the link's x and 0.3° about its z.
- **UR5e bench C**: as designed; its wrist D405 calibrated 0.2 mm and 0.1 mm off.
- **SO-100 sample**: another lab's arm (`sample:so100`).

## 1. Build the cell set

Open the UR5e sample (`dexvantage --sample ur5e`), then File ▸ **Compare cells…** (the command after Compare with…).
**Add cells…** takes the benches' `.dexv` files; the sample box adds the SO-100. Each addition is one undo step and
is kept with the project (`project.analysis.studies['cells']`). Leave the reference at `0: UR5e + Robotiq 2F-85
sample`, the figures at the quick set (the default) and, for this run, the tolerance spread at 8 draws.

## 2. Run

**Run** works in the background (17 s here: the 8 tolerance draws re-run coverage on every cell; without draws the
same set takes about 3 s). The headline reads **0.298**, status **moderate gap**:

> Largest geometric gap to UR5e + Robotiq 2F-85 sample: 0.298 (SO-100 sample). 3 of 4 cells pool; 1 cannot be pooled.

The gap table:

| Cell | Gap index | Band | Largest component |
|---|---|---|---|
| UR5e bench B | 0.023 | small | viewpoint 0.06 |
| UR5e bench C | 0.000 | small | — |
| SO-100 sample | 0.298 | moderate | position 0.53 |

Selecting bench B shows its matched cameras: the wrist D405 identical (the gap compares the designs; the calibration
is in the drift below), the overhead D435 **10.00°** apart on its axis (rotation 10.00°, distance 0 mm), the task
volume over 38.7 % of its image as designed against 30.2 % as built. Against the SO-100 the wrist cameras are 6.5°
apart on their axes and 31 mm apart relative to the fingertips, but their images are rolled 179.9° against each other
(the rotation column); the scene cameras are 57.5° and 1.29 m apart, the SO-100's has 21° less HFOV and 0.46× the
pixels per degree, and its table fills 9.5 % of the image against 38.7 %.

## 3. Figures side by side

With 8 draws per cell (± 1 mm, ± 0.5° per mount, the labelled defaults), a star marks a difference beyond
2 √(σ_ref² + σ_cell²) of the build spread that is also significant over the draws (core/stats' permutation test,
Holm-adjusted across the table):

| Figure | Design | Bench B | Bench C | SO-100 |
|---|---|---|---|---|
| Task volume seen | 91.90 % | 81.62 % * | 91.90 % | 94.52 % * (best) |
| Seen by two or more cameras | 26.44 % | 19.77 % * | 26.44 % | 47.30 % * (best) |

Bench B's turned overhead camera costs 10.3 points of coverage (18 σ of the build spread: not something the mount
tolerances explain; Holm-adjusted p 0.006); bench C's figures are the design's (Δ 0, adjusted p 1, not
meaningful). 4 of the 6 differences are meaningful.

## 4. Pooling

The pooling table says the three UR5e benches pool **as is** (the same cameras, keys, sizes, rates, codec, joints and
control rate in their LeRobot features). The SO-100 is **not poolable**: 6 state and action values against 14 (another
embodiment); the steps that would map the rest are listed (rename both cameras, resize the scene camera 1280 × 720 →
1920 × 1080, resample the rows and actions 30 → 50 Hz, rescale the gripper from the Jaw's range to the 2F-85 driver
joint's). **Export…** writes `cells_pooling.json` with these steps per cell.

## 5. Drift of the built copies

| Camera | Cell | Kind | Offset | Rotation | Share of tolerance | Verdict |
|---|---|---|---|---|---|---|
| Wrist D405 | bench B | measured | 2.50 mm | 0.300° | 2.50 | adjust |
| Wrist D405 | bench C | measured | 0.22 mm | 0.000° | 0.20 | ok |
| Overhead D435 | bench B | as designed | 0.00 mm | 10.000° | 20.00 | re-mount |
| Overhead D435 | bench C | as designed | 0.00 mm | 0.000° | 0.00 | ok |

The wrist rows come from the calibrations; the overhead rows compare the benches' designs (no calibration of that
camera), so bench B's "re-mount" says its file places the camera 10° off the plan. With two measured copies there is
no outlier test (three are needed) and the copies' median (1.35 mm) is beyond the tolerance but neither copy is near
it, so the build is not called systematic and there is no as-built variant to add.

## 6. Apply

With the reference the open project, the page has no placement to suggest (it says to choose another reference).
Open bench B instead, add the design as a cell and choose it as the reference: the suggestion reads

> Mount Overhead D435 where UR5e + Robotiq 2F-85 sample's Overhead D435 sits in the world: 0 mm and 10.0° from its
> mount now; that pair's viewpoint gap is then zero (the lenses may still differ).

**Preview** draws the pose as a dashed frustum; **Apply** moves the camera (one undo step).

## The same from the command line and Python

    dexvantage --sample ur5e --cells benchB.dexv benchC.dexv sample:so100 --cells-out cells.json \
        --cells-pooling cells_pooling.json --cells-html cells.html --cells-settings '{"tolerance_draws": 8}'

    import dexvantage as dv
    res = dv.cells.compare(dv.sample('ur5e'), 'benchB.dexv', 'benchC.dexv', 'sample:so100', tolerance_draws=8)
    res.headline['value'], res['pooling']['text']
    res.to_frame('cells'); res.metric_rows(); res.gap_rows(); res.save_files('out/')

`--cells-compare OUT.json` runs the study on the cell set stored in the project. The export format `cells` writes
the same three files with the other exports (`--export out/ --formats cells`).
