# Tolerances: a worked example on the SO-100 sample

This carries the SO-100 sample's two mount tolerances (the wrist camera on the fixed jaw, the third-person camera on
its tripod, both at the labelled defaults ± 1 mm, ± 0.5°) through a coverage requirement, tightens the mount that
matters, and exports perturbed extrinsics for training. The model is in `docs/models/tolerance.md`. Figures below are
from a run on 2026-11-07 at the default settings (64 draws, seed 0, uniform within the bounds).

## 1. A requirement to judge by

    dexvantage --sample so100

The study carries the figures your requirements name. Add one for the share of the task volume seen,
`coverage.covered >= 94 %` (the sample's nominal is 94.52 %). Without requirements the study carries the headline
figures of coverage, depth validity, motion blur and VLA tokens (a default) and has no threshold to judge them by.

## 2. Run

Open Analysis ▸ Tolerances and press **Run** (about 9 s: each of the 64 draws re-runs the coverage study on the moved
rig). The headline reads **94.34 %**, status **P5 · holds**:

> Task volume seen (>= 94.00 %, a default): 100 % of 64 draws meet it [94–100 %, Wilson 95 %]; P5 94.34 %, nominal
> 94.52 %.

The table shows the figure ± spread: mean 94.54 % ± 0.103 % (σ), [P5–P95 94.34–94.70 %]. Quick-study headlines on
the other pages still show the design as it is (94.52 %); this page is where the spread shows.

The contribution chart says the wrist camera carries 77 % of the spread (largest |ρ| 0.49, against a noise level of
2 / √63 = 0.25) and the third-person camera 23 % (|ρ| 0.31). For the wrist camera, the rotation part is the larger
(Σρ² 0.44 against 0.33 for position).

## 3. Tighten and run again

**Preview** draws the wrist camera's 64 perturbed frusta (to scale, out to the task volume) in its colour and, in
green, the same draws at the tightened tolerance. **Apply** sets its rotation tolerance to ± 0.25° (one undo step).
Run again: P5 94.39 %, mean 94.53 % ± 0.077 %, and now the wrist camera's position part dominates; the next
suggestion is ± 0.5 mm. The per-sensor editor below the chart sets any mount's ± mm and ± ° directly, one undo step
each.

A calibrated sensor (Calibrate ▸ Solve hand-eye, then Apply) draws from its calibration covariance instead of its
mount tolerance; the editor's "Drawn from" column says which, and a sensor moved after its calibration goes back to
its mount tolerance.

## 4. Export perturbed extrinsics

**Export perturbations…** writes 100 draws (seed 0; the `n_export` setting changes the count) to
`perturbations/perturbations.json` (287 kB) and `perturbations/perturbations.npz` (85 kB), with an
`export_manifest.json` holding their SHA-256 and provenance. In a script:

    from dexvantage.core import exporters as X
    X.export(project, 'out', ['perturbations'], options={'perturbations': {'n': 500, 'seed': 1}})

For Isaac Lab, each draw's `isaac_lab` entry is a `CameraCfg.OffsetCfg` in the camera's mount frame (the wrist
camera's link `Fixed_Jaw`, the world for the tripod): `pos`, `rot_xyzw` (Isaac Lab v3.0 and later) or `rot_wxyz`
(v2.x), `convention="ros"`. Pick a draw per environment at reset to randomise the mounts. For the 3DGS renderer, the
`gs` entry is the camera centre and camera-to-world rotation at the robot's home pose (OpenCV axes). For DexVantage's
own MuJoCo renders, pass `delta_body` as the view's `extrinsic_delta`.

The same seed gives the same draws, in the export and in the study.

## Batch

    dexvantage project.dexv --tolerance-mc tolerance.json

writes the study's result (every draw's figures and mount changes) as JSON with provenance.
