# Designed experiments: a worked example on the UR5e sample

Which of your placement choices moves a figure, and by how much? The Designed experiments page varies a few
quantities of the rig together. It evaluates the studies at every run and reports the main effects and the Sobol
indices, with their intervals. It also gives the best run it found, against the rig as it is. The model, the
estimators and their validation are in [../models/experiments.md](../models/experiments.md).

The figures below were measured on the development machine on 2026-11-07. Your figures depend on the seed and on the
studies' versions.

## 1. Open the sample and the page

1. **File ▸ Open sample ▸ UR5e**. The rig has a wrist D405 on the flange and an overhead D435 on the cage. The D435
   looks straight down (pitch 90°).
2. Open **Analysis ▸ Designed experiments**.

## 2. Pick the factors

Pick each quantity from the factor list, then press **Add factor**. The list offers every quantity of the rig
(`core/factors.py`):
- each sensor's x, y, z, roll, pitch and yaw;
- the shifts dx, dy, dz and the small turns droll, dpitch, dyaw about the sensor's own axes;
- its model, and whether it is enabled;
- the numeric and on/off settings of the studies.

A new factor starts at ±20 mm or ±5° around the current value. That is a DexVantage default; edit **Low** and
**High** in the row below the table. A locked sensor is listed as "(locked)" and is refused: unlock it to vary it.

For this example:

| Factor | Path | Range |
|---|---|---|
| Overhead D435, turn about its y | `sensor:Overhead D435.dpitch` | −5° to +5° |
| Overhead D435, shift z | `sensor:Overhead D435.dz` | −0.05 to +0.05 m |
| Wrist D405, turn about its y | `sensor:Wrist D405.dpitch` | −5° to +5° |

Use the turns (`dpitch`) here, not the absolute `pitch`. An absolute pitch range of −5° to +5° would turn the
overhead camera from looking down to looking at the horizon: a first try at this example did just that. It gave
40.9–52.2 % of the task volume seen, against 91.9 % now.

## 3. Pick the responses and the design

1. Tick **Task volume seen** (`coverage.covered`), **Depth-valid task volume** (`depth.valid`) and **Task volume seen
   by two or more cameras** (`coverage.seen_twice`).
2. Leave **Best run by** at the first response.
3. Under **Design**, choose **Latin hypercube** (the default), n = 16 and seed 1. The count reads **16 runs**.
4. Press **Run**. Each run evaluates the coverage and depth-validity studies, about 1 s on the UR5e sample, so this
   takes about 16 s. The progress shows in the page's note; **Cancel run** stops it and keeps the runs done.

## 4. Read the result

The headline reads **92.8 % · better than now**: *Best of 16 runs: task volume seen 92.8 % at run 15, against 91.9 %
for the rig as it is.* Run 15 sets:
- the overhead D435's dpitch to −2.51°;
- its dz to +0.042 m;
- the wrist D405's dpitch to −4.01°.

The runs table lists the best run first, in green. The **Main effects** chart (binned: last quantile bin minus first,
with Welch intervals) gives, for task volume seen:

| Factor | Effect (low to high) | 95 % interval |
|---|---|---|
| Overhead D435 dpitch | −1.10 % | −5.23 to 3.03 |
| Overhead D435 dz | +3.59 % | 0.02 to 7.15 |
| Wrist D405 dpitch | −2.50 % | −3.13 to −1.87 |

With 16 runs only the wrist tilt's effect is clearly away from zero. The overhead camera's height comes close; its
tilt's interval spans zero. The 3D view draws each run's position of the overhead D435 as a small cross, darker where
more of the task volume was seen. The best run is ringed in green and joined to where the camera is now; a turn moves
the orientation, so the crosses show the dz shifts.

## 5. What drives the variance: Sobol indices

Choose **Sobol indices (Saltelli sampling)**, n = 64 and **Processes** 3, then press **Run**. That is
64 × (3 + 2) = 320 runs. In a pool of three processes they took 194.9 s, about 1.6 times quicker than serial (each
process loads DexVantage and the robot once). The **Sobol indices** chart gives, for task volume seen:

| Factor | First order S1 [95 %] | Total ST [95 %] |
|---|---|---|
| Overhead D435 dpitch | 0.22 [−0.01–0.44] | 0.31 [0.21–0.45] |
| Overhead D435 dz | 0.19 [−0.11–0.43] | 0.14 [0.08–0.21] |
| Wrist D405 dpitch | 0.58 [0.43–0.70] | 0.61 [0.41–0.82] |

The wrist camera's tilt drives about 60 % of the variance in what the rig sees over these ranges. The overhead
camera's tilt drives a third, part of it through interactions (ST above S1). Its height drives the least. The dz
estimate's S1 is above its ST: that is the estimators' spread at N = 64, and their intervals overlap. Read the
intervals before small differences. With few base samples the notes say so: when an estimate falls outside 0 to 1, or
an interval is more than 0.5 wide, a note gives the range of the estimates and the widest interval (on ALOHA 2 with
N = 8, the overhead camera's tilt read S1 = 0.84 and ST = 1.52); raise n until the intervals are narrow enough to
rank the factors.

The depth-valid task volume tells a different story:
- Overhead D435 dpitch: ST = 0.000 [0.000–0.000]. A ±5° tilt of the overhead D435 does not change its depth-valid
  share at all (61.28 % at −5°, 0° and +5°, checked directly). The task volume stays inside its depth field of view
  and range; it only changes at a 20° tilt (45.34 %).
- Overhead D435 dz: ST = 0.34 [0.26–0.45].
- Wrist D405 dpitch: ST = 0.65 [0.46–0.88].

The best of the 320 runs sees 93.11 % (run 319: the overhead D435's dpitch +0.86°, dz −0.018 m; the wrist D405's
dpitch −4.95°).

## 6. Apply, preview, export

- **Preview** names the best run's values in the status bar and rings that run in the 3D view. It changes nothing.
- **Apply** sets the best run's factors on the rig as one undo step (Edit ▸ Undo restores it).
- **Export…** writes the result as JSON with provenance, and the runs as CSV beside it. The JSON records the CSV's
  SHA-256.
- **Explain** gives the designs, the estimators and the defaults.

The best run is the best point the design sampled, not an optimum. The optimiser searches; the experiment tells you
which quantities are worth searching over.

## 7. The same in batch mode

Save the design as `design.json`:

```json
{
  "schema": "dexvantage.experiment/1",
  "name": "Overhead and wrist camera tilt",
  "factors": [
    {"path": "sensor:Overhead D435.dpitch", "lo": -5, "hi": 5},
    {"path": "sensor:Overhead D435.dz", "lo": -0.05, "hi": 0.05},
    {"path": "sensor:Wrist D405.dpitch", "lo": -5, "hi": 5}
  ],
  "responses": ["coverage.covered", "depth.valid", "coverage.seen_twice"],
  "objective": "coverage.covered",
  "design": {"method": "lhs", "n": 16, "seed": 1}
}
```

Then run it:

    dexvantage --sample ur5e --experiment design.json
    Designed experiment (Latin hypercube, 16 runs): better than now → design.results.json, design.results.csv

`--experiment-out OUT.json` chooses the result's name; the CSV goes beside it. `--experiment-saved OUT.json` runs the
experiment saved in a project instead (the page saves it there). A design that cannot run exits with code 2 and a
sentence. For example, *There is no metric called no.such.metric.*, or *Wrist D405 z: the sensor is locked; unlock it
to vary it.*

## Other designs

- **Full factorial**: every combination of the levels. A continuous factor takes 2 levels by default, or its own
  "factorial levels".
- **Fractional factorial**: two levels per factor, at resolution IV by default (main effects free of two-factor
  interactions) or III (fewer runs; main effects aliased with two-factor interactions). The notes give the
  generators and the resolution reached. Seven factors take 16 runs at IV and 8 at III.
- **Sobol sequence**: a space-filling set of n points. Choose n as a power of two.

Categorical factors (a sensor's model, with levels such as `intel-realsense-d405, intel-realsense-d435`) and on/off
factors (`enabled`) go in every design. A fraction needs them at two levels.
