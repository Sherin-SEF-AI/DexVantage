"""Batch mode for statistics and power analysis (core/stats.py, core/statcompare.py, core/studies/stats.py):

    dexvantage [PROJECT.dexv] --stats SPEC.json [--stats-out OUT.json]

runs the statistics spec (schema dexvantage.stats/1, docs/models/stats.md: success counts, samples, two variants over
the tolerance Monte Carlo, an experiment's runs, tracker runs, and the power calculator) laid over the project's
saved statistics settings (or with no project: success counts, samples and the power calculator need none) and writes
OUT.json (default SPEC.results.json beside the spec) with provenance. It prints the headline and each comparison's
verdict. Exit 0 when it ran (whatever the verdict: a difference that is not clear is a result, not a failure), 2 when
the spec cannot be analysed (one sentence on stderr). --stats-saved OUT.json (core/jobs) runs the settings saved in
the project.
"""
from __future__ import annotations

import os
import sys


def add_arguments(g):
    from .core import provenance as PV
    PV.INPUT_ARGS.setdefault('stats', 'json')           # the spec file is hashed as an input
    g.add_argument('--stats', metavar='SPEC.json', help='statistics and power analysis from a spec (schema '
                   'dexvantage.stats/1, docs/models/stats.md): compare success counts, samples, two variants over '
                   'the tolerance Monte Carlo, experiment or tracker runs; the power calculator; writes the result '
                   'as JSON with provenance')
    g.add_argument('--stats-out', metavar='OUT.json', help='with --stats: the result file (default SPEC.results.json '
                   'beside the spec; "-" for stdout)')


def wanted(args) -> bool:
    return bool(getattr(args, 'stats', None) or getattr(args, 'stats_out', None))


def run(args, project) -> int:
    from .batch import EXIT_ERROR, EXIT_OK, _err, _unexpected, _write
    from .core import jobs
    from .core import provenance as PV
    from .core.studies import stats as SS
    if not getattr(args, 'stats', None):
        return _err('--stats-out goes with --stats SPEC.json')
    out = getattr(args, 'stats_out', None) or SS.default_output(args.stats)
    if out != '-' and os.path.abspath(out) == os.path.abspath(args.stats):
        return _err('the output would overwrite the spec: choose another name with --stats-out')
    try:
        doc, snap = SS.run_spec(args.stats, project)
    except jobs.StudyError as e:
        if project is not None:
            from . import batch_track
            batch_track.record('stats', None, project, reason=str(e))      # --track: a failed run
        return _err(str(e))
    except Exception as e:                              # noqa: BLE001 — a sentence here, the traceback in a file
        return _unexpected('The statistics', e)
    doc = {k: v for k, v in doc.items() if k != 'cached'}
    doc = PV.stamp(doc, snap, output=PV.batch_output(doc, out, option='stats'))
    try:
        _write(doc, out)
    except SystemExit as e:
        return int(e.code)
    from . import batch_track
    batch_track.record('stats', doc, snap, out)              # --track (or the project's tracking): a run
    if out != '-':
        h = doc['headline']
        print(f'Statistics ({doc["source_label"].lower()}): {h["word"]} → {out}')
        for c in doc.get('comparisons') or []:
            print(f'  · {c["verdict"]["text"]}')
        if doc.get('power'):
            print(f'  · Power: {doc["power"].get("answer", "")}')
        sys.stdout.flush()
    return EXIT_OK if doc.get('headline') else EXIT_ERROR
