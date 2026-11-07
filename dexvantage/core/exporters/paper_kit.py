"""The paper kit (core/publication.py): a methods paragraph, LaTeX tables, figures as PDF and SVG at one- and
two-column widths, refs.bib, a main.tex that \\input's them, a reproducibility appendix with its run plan, and
provenance.json.

    @exporter('paper_kit', 'Paper kit (LaTeX methods, tables, figures, BibTeX)')
    options: those of core.publication.build: results ({study id: result} or [result]), results_dir (a folder of
             result JSON files; a relative folder not found from the current folder is looked for beside the export
             folder, as the kit's own run plan writes it), studies ('all' or [study id] to run now; default 'all'
             when neither results nor results_dir is given, as the report), sections, siunitx, widths, pdf_figures,
             bibliography, title, validation (validation.json or its document), figure_specs

Files (under out_dir/paper_kit/): main.tex, methods.tex, appendix.tex, tables/<name>.tex, figures/<id>-<width>.svg
and .pdf, refs.bib, reproduce/run_plan.json, provenance.json (what the kit holds, the trace of every number in the
methods paragraph, the checks and their outcome; stamped with the provenance block like every exported JSON).

Known limits: those of core/publication.py; the kit is compiled only where a LaTeX engine is installed (the tests
say which ran); the PDF figures need Qt's PDF writer (offscreen is enough), without it only the SVGs are written.
"""
from __future__ import annotations

import os

from . import exporter


@exporter('paper_kit', 'Paper kit (LaTeX methods, tables, figures, BibTeX)', needs_robot=False,
          description='A methods paragraph with every number traced to the project or a result, booktabs tables '
                      '(sensors, results with intervals, scorecard, experiment, variants), PDF and SVG figures at one '
                      'and two column widths, refs.bib, main.tex and a reproducibility appendix with its run plan.')
def export_paper_kit(project, out_dir: str, options: dict, ctx) -> dict:
    from .. import publication as PB
    o = dict(options or {})
    rd = o.get('results_dir')
    if rd and not os.path.isabs(str(rd)) and not os.path.isdir(str(rd)):
        beside = os.path.join(out_dir, str(rd))
        if os.path.isdir(beside):
            o['results_dir'] = beside
    res = PB.write(project, out_dir, o, ctx)
    return {'files': res['files'], 'notes': res['notes']}
