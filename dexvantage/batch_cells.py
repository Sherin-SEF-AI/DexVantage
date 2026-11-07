"""Batch mode for the multi-cell comparison (core/cells.py):

    dexvantage --cells A.dexv B.dexv [C.dexv …] [--cells-out OUT.json] [--cells-pooling POOL.json]
               [--cells-html PAGE.html] [--cells-settings JSON]
    dexvantage A.dexv --cells B.dexv …          the project given first is the first cell (and the reference)

A cell is a .dexv file or folder, or sample:<name> (dexvantage --help lists the samples). --cells-settings takes the
study's settings as JSON text or a .json file ({"reference": 1, "metrics": "all", "tolerance_draws": 16,
"pooling": "plan", "variants": false}); the project's own stored cell set (project.analysis.studies['cells']) is
used too. Without an output flag the comparison is written to standard output. Exit codes: 0 done, 2 an input that
cannot be used (a file that cannot be read, a set of one cell, bad settings), said in one sentence.
"""
from __future__ import annotations

import json
import os
import sys

EXIT_OK, EXIT_ERROR = 0, 2


def add_arguments(g):
    g.add_argument('--cells', nargs='+', metavar='CELL', help='compare several robot cells: .dexv files or folders, '
                   'or sample:<name>; with a project given, it is the first cell (core/cells.py)')
    g.add_argument('--cells-out', metavar='OUT.json', help='with --cells: the comparison as JSON ("-" for stdout, '
                   'the default)')
    g.add_argument('--cells-pooling', metavar='POOL.json', help='with --cells: the pooling mapping (rename, resize, '
                   'resample, reorder … per cell, and what cannot be mapped)')
    g.add_argument('--cells-html', metavar='PAGE.html', help='with --cells: the comparison as a standalone HTML page')
    g.add_argument('--cells-settings', metavar='JSON', help='with --cells: the study\'s settings as JSON text or a '
                   '.json file, e.g. {"reference": 1, "metrics": "all", "tolerance_draws": 16}')


def wanted(args) -> bool:
    return any(getattr(args, k, None) for k in ('cells', 'cells_out', 'cells_pooling', 'cells_html',
                                                 'cells_settings'))


def _err(text) -> int:
    print(f'dexvantage: {text}', file=sys.stderr)
    return EXIT_ERROR


def _settings(text) -> dict:
    from .core.jobs import StudyError
    try:
        if not str(text).lstrip().startswith(('{', '[')):
            with open(text, encoding='utf-8') as fh:
                text = fh.read()
        d = json.loads(text)
    except (OSError, ValueError) as e:
        raise StudyError(f'--cells-settings could not be read as JSON ({e}).') from None
    if not isinstance(d, dict):
        raise StudyError('--cells-settings must be a JSON object, e.g. {"reference": 1}.')
    return d


def _write_json(doc, dest):
    text = json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False) + '\n'
    if dest in (None, '', '-'):
        sys.stdout.write(text)
        return
    tmp = dest + '.part'
    with open(tmp, 'w', encoding='utf-8') as fh:
        fh.write(text)
    os.replace(tmp, dest)


def run(args, project) -> int:
    from .core import cells as CE
    from .core import provenance as PV
    from .core.jobs import StudyError
    from .core.project import blank_project
    if not getattr(args, 'cells', None):
        return _err('--cells-out, --cells-pooling, --cells-html and --cells-settings go with --cells A.dexv B.dexv …')
    items = list(args.cells)
    try:
        st = _settings(args.cells_settings) if getattr(args, 'cells_settings', None) else {}
        if project is None:
            first, items = items[0], items[1:]
            cells0 = CE.make_cells([first])
            base = cells0[0].project
        else:
            base = project.snapshot()
        base.analysis.studies[CE.SID] = {**(base.analysis.studies.get(CE.SID) or {}), **st}
        doc = CE.jsonable(CE.compare_project(base, extra=items))
    except StudyError as e:
        return _err(str(e))
    except Exception as e:                              # noqa: BLE001 — a sentence here, the traceback in a file
        from .batch import _unexpected
        return _unexpected('The cell comparison', e)
    doc = {'schema': CE.SCHEMA, 'study': CE.SID, **{k: v for k, v in doc.items() if k != 'schema'}}
    who = project if project is not None else (base if base is not None else blank_project('No project'))
    out = getattr(args, 'cells_out', None)
    try:
        _write_json(PV.stamp(doc, who, output=PV.batch_output(doc, out, option='cells_out')), out)
        if getattr(args, 'cells_pooling', None):
            pm = CE.pooling_mapping(doc)
            _write_json(PV.stamp(pm, who, output=PV.batch_output(pm, args.cells_pooling, option='cells_pooling')),
                        args.cells_pooling)
        if getattr(args, 'cells_html', None):
            blk = PV.block(who, doc, output=PV.batch_output(doc, args.cells_html, option='cells_html'))
            with open(args.cells_html, 'w', encoding='utf-8') as fh:
                fh.write(CE.to_html(doc, blk))
    except OSError as e:
        return _err(f'could not write the comparison: {e.strerror or e}. Choose a folder that exists and that you can '
                    'write to.')
    h = doc['headline']
    if out not in (None, '', '-'):
        print(f'Cells: {len(doc["cells"])}; {h["word"]} {h["value"]:.3f} ({h["cell"]}); {doc["pooling"]["text"]} '
              f'→ {out}')
    from . import batch_track
    batch_track.record(CE.SID, doc, who, out if out not in (None, '', '-') else None)   # --track: a run
    return EXIT_OK
