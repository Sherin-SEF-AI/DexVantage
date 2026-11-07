"""The multi-cell comparison export (core/cells.py): the comparison as JSON, the pooling mapping JSON (schema
dexvantage.cells_pooling/1: per cell its verdict, the rename / resize / resample / reorder / re-encode / rescale
steps onto the target's LeRobot features, and what cannot be mapped) and a standalone comparison HTML page.

    @exporter('cells', 'Multi-cell comparison (pooling mapping JSON, comparison HTML)')
    options: cells (more cells besides those the project's settings name: paths, 'sample:<name>'), and any key of
             the study's settings (reference, metrics, tolerance_draws, seed, pooling, variants), laid over the
             project's

Files (under out_dir/cells/): cells.json, cells_pooling.json, cells_comparison.html (the JSON files stamped with
provenance; the page carries its provenance block in a closed section).

Known limits: those of the study (core/studies/cells.py); a cell set of one cell is refused.
"""
from __future__ import annotations

import os

from .. import cells as CE
from . import exporter

FOLDER = 'cells'


@exporter('cells', 'Multi-cell comparison (pooling mapping JSON, comparison HTML)', needs_robot=False,
          description='The open project against other cells: figures of merit, the geometric gap per camera role, '
                      'the dataset pooling mapping and drift, as JSON and an HTML page.')
def export_cells(project, out_dir: str, options: dict, ctx) -> dict:
    p = project.snapshot()
    st = dict(p.analysis.studies.get(CE.SID) or {})
    extra = list(options.get('cells') or [])
    st.update({k: v for k, v in options.items() if k != 'cells'})
    p.analysis.studies[CE.SID] = st
    doc = CE.jsonable(CE.compare_project(p, ctx, extra=extra))
    names = CE.write_files(doc, project, os.path.join(out_dir, FOLDER))
    return {'files': [f'{FOLDER}/{n}' for n in names],
            'notes': [doc['headline']['text'], 'The gap bands are a heuristic, not calibrated against any policy\'s '
                                               'success.'] + [n for n in doc.get('notes') or []][:5],
            'unstamped': []}
