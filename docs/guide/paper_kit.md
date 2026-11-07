# Writing the paper kit

A worked example on the UR5e sample. The rules are in [docs/models/paper_kit.md](../models/paper_kit.md).

## 1. In the window

Open the UR5e sample (File ▸ Open sample), run the studies you want to report (Coverage, Depth validity, Time sync,
Bandwidth…), then Export ▸ Paper kit…. The Publication page opens in the Analysis dock:

- **Sections**: tick what the kit holds (methods paragraph, sensor table, results table, scorecard, experiment
  results, variant comparison, statistics, cell comparison, tracked runs, figures, reproducibility appendix). The
  statistics, cells and runs tables are written when their results are present (run the Statistics or Cells page, or
  turn tracking on for the runs).
- **Options**: siunitx (`\num`, `\qty`, `\unit`) or plain numbers, one and/or two column figures, PDF figures beside
  the SVGs, BibTeX or biblatex, a title, and a validation report (`validation.json` from Help ▸ Validate) for the
  number of checks each model passed.
- **Refresh preview** builds the kit in the background from the results on screen (no study runs). The headline
  reads, for example, `48 / 48 numbers`, **ready**: every number of the methods paragraph traces to the project, a
  library entry, a result or the software versions, every citation has its BibTeX entry, the TeX is balanced. A
  problem turns it to **check** with the first problem named.
- **Preview** rebuilds it with the choices without keeping them; **Apply** keeps them in the project (one undo step),
  so a batch export writes the same kit; **Export…** writes it into a folder (tick "Run the studies not computed
  yet" to fill the gaps first).

## 2. In batch mode

    dexvantage --sample ur5e --export out/ --formats paper_kit \
        --export-options '{"paper_kit": {"studies": ["rig", "bandwidth", "coverage", "depth_validity", "time_sync"]}}'
    paper_kit: 24 files
      · 4 TeX part(s), 4 figure(s) at single and double column width, 16 BibTeX entries; ready: All 48 numbers in the
        methods paragraph and 30 in the tables trace to the project or a result; every citation has its BibTeX entry
        and the LaTeX is balanced.

(measured 2026-11-07; the studies took about 2 s). Give `"results_dir": "results/"` instead of `studies` to report
batch result files you already have, and `"siunitx": true`, `"widths": ["single"]` or `"sections": [...]` to choose.

## 3. What comes out

`out/paper_kit/methods.tex` starts:

> We planned the sensing rig with DexVantage~\cite{dexvantage} version \texttt{0.1.0.dev0} (commit …), which loads
> the robot models with MuJoCo~\cite{mujoco_todorov2012} version 3.17.0 under Python 3.14.4.
>
> UR5e + Robotiq 2F-85 is the primary robot, whose base is the world frame (x forward, y left, z up), modelled from
> the MuJoCo Menagerie models `universal_robots_ur5e`, `robotiq_2f85` at commit `98a70c569c4c`~\cite{…} (licence:
> BSD-3-Clause … and BSD-2-Clause …).
>
> The rig "Wrist D405 + overhead D435 (sample)" holds 2 sensors (depth); … Wrist D405 is an Intel RealSense Depth
> Camera D405~\cite{intel-realsense-d405,intel-realsense-d405-2} passive stereo depth camera with a stereo baseline of
> 0.018 m; the studies take its depth stream at 1280 × 720 px and 30 fps (Z16), its RGB stream at 1280 × 720 px and 30
> fps (YUY2) (the stream DexVantage selects by default when the rig names none), with a field of view of 84° × 58° for
> its depth stream (horizontal by vertical, as the maker states for streams of this aspect). It is mounted on link `wrist_3_link` of
> UR5e + Robotiq 2F-85 at position (0.000, 0.131, 0.070) m and orientation (x, y, z, w) = (−0.1785, 0.1785, 0.6842,
> 0.6842) in that link's frame. …

then one sentence per study, e.g. "Coverage and resolution (study `coverage`) ran with the DexVantage defaults
otherwise; the model "Occlusion by the posed robot" has 7 registered checks; the model "Lens projection and defocus"
has 10 registered checks~\cite{…}" (with a validation report: "passed 7 of 7 checks").

Every number there is in `provenance.json`'s `trace` with the place it was read from, for example
`{"text": "0.018", "kind": "number", "path": ["library", "intel-realsense-d405", "depth", "baseline_m"]}`.

`tables/results.tex` is a booktabs table of every named figure (Task volume seen 91.9 %, Depth-valid task volume
84.61 %, Worst camera to joint-state skew ≥ 54.43 ms …), each row ending with the source of its numbers:

    Coverage and resolution & Task volume seen & \dvnum{91.9} & \% & --- &  \\ % dv-src: [["metric", "coverage", "coverage.covered"]]

Compile with `pdflatex main`, `bibtex main`, `pdflatex main` twice (or choose biblatex and run biber). Copy what you
need into your paper: each table file defines `\dvnum` itself, so it compiles on its own with booktabs (and siunitx
when you chose it).

## 4. Reproducing it

`out/paper_kit/reproduce/run_plan.json` runs every study the kit reports with its stored settings, the scorecard,
and then this export again:

    dexvantage cell.dexv --run-plan out/paper_kit/reproduce/run_plan.json again/

The appendix lists the plan's steps with the figures each regenerates, the seeds, the software versions and each
result's SHA-256.

## 5. From Python

    import dexvantage as dv
    rig = dv.sample('ur5e')
    res = {s: rig.run(s) for s in ('rig', 'bandwidth', 'coverage')}
    print(dv.publication.methods(rig, res))          # the paragraph (TeX)
    dv.publication.tables(rig, res)['results']       # a table
    dv.publication.kit(rig, res)['status']           # {'word': 'ready', 'numbers': …, 'traced': …}
    dv.publication.export(rig, 'out/', res, siunitx=True)

The assistant's `get_methods_paragraph` tool returns the same paragraph from the results the window holds.
