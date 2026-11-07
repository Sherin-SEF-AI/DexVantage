"""The multi-cell comparison study (research tools part 2): the open project and the cells its settings name
(core/cells.py: figures of merit side by side, the geometric gap per camera role, dataset pooling against the
LeRobot feature schema, drift of the design across built copies).

    @jobs.study('cells', 'Multi-cell comparison', quick=False, batch='cells-compare')  → --cells-compare OUT.json
    (batch --cells A.dexv B.dexv … compares files given on the command line: dexvantage/batch_cells.py)

Settings (project.analysis.studies['cells'], every one optional): 'cells' (a list of {'path': a .dexv file or
folder} | {'sample': name} | {'variant': uid or name}, or the same as strings, 'sample:<name>' for a sample),
'variants' (default true: the open project's other variants are cells too), 'reference' (the cell the others are
compared with, by number; default 0, the open project), 'metrics' (a list of metric ids, 'all', or the default quick
figures, core/cells.DEFAULT_METRICS), 'tolerance_draws' (default 0, off: with n ≥ 2 the tolerance Monte Carlo says
which differences exceed the build spread), 'seed' (default 1), 'pooling' ('auto' (default): a cell's recorded
dataset where it has one, else its plan; 'plan'; 'recorded'). The page passes '_fingerprint' (the cells' files'
sizes and times) so a changed file misses the result cache.

The result (JSON-able): the cells, the headline (the largest gap index to the reference, with its band as the status
word; the pooling verdict counts), the figures table, the gap per cell (components, matched cameras, findings) and
the matrix of every pair, the pooling verdict and mapping per cell, the drift per camera and copy, the suggestion
(Apply on the page) and notes.

Known limits: those of core/cells.py and core/cells_pool.py (docs/models/cells.md): a geometric gap, not appearance
or success; pooling read from the feature schema, not the data; drift from the latest calibration per copy; cells
loaded from files are read when the study runs (a file changed since is read again only when its size or time
changes).
"""
from __future__ import annotations

from .. import cells as CE
from .. import jobs


@jobs.study(CE.SID, 'Multi-cell comparison', quick=False, batch='cells-compare')
def cells_study(project, ctx: jobs.Ctx) -> dict:
    """The comparison of the open project with the cells its settings name (module docstring)."""
    ctx.report(0.01, 'Reading the cells')
    return CE.jsonable(CE.compare_project(project, ctx))
