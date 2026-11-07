"""Batch mode's CI gate, reproduction and plans (spec §11), added to batch.py's parser:

    dexvantage PROJECT.dexv --check requirements.yaml [OUT.json]   the readiness scorecard: prints a table, writes the
                                                                   JSON with provenance (default OUT: scorecard.json
                                                                   in the current folder); "defaults" checks the
                                                                   labelled defaults
    dexvantage [PROJECT.dexv] --reproduce RESULT.json [--reproduce-out REPORT.json] [--tolerance ...]
                                                                   re-run what wrote RESULT.json and compare every
                                                                   number (core/reproduce.py)
    dexvantage PROJECT.dexv --run-plan plan.json [OUT_DIR]         studies, checks, exports and reproductions in order,
                                                                   one JSON summary (core/run_plan.py)

Exit codes, the same for all three: 0 pass (reproduces; every step done and passed), 1 fail (a requirement fails or is
not evaluated; the numbers do not reproduce; a check of the plan fails), 2 error (a file that cannot be read, an
unknown metric or study, no project, a step that could not run). Errors are one sentence on stderr.
"""
from __future__ import annotations

import os
import sys

EXIT_OK, EXIT_FAIL, EXIT_ERROR = 0, 1, 2


def add_arguments(g):
    from .core import provenance as PV
    g.add_argument('--check', nargs='+', metavar=('REQUIREMENTS.yaml', 'OUT.json'),
                   help='the readiness scorecard CI gate: evaluate a requirements file ("defaults" for the labelled '
                        'defaults), print the table and write the JSON with provenance (default scorecard.json); '
                        'exit 0 pass, 1 fail, 2 error')
    g.add_argument('--reproduce-out', metavar='REPORT.json', help='with --reproduce: also write the reproduction '
                   'report (every number compared) as JSON')
    g.add_argument('--run-plan', nargs='+', metavar=('PLAN.json', 'OUT_DIR'),
                   help='run the studies, checks, exports and reproductions a plan lists, in order, and write one '
                        'JSON summary (core/run_plan.py); outputs go under OUT_DIR (default: the plan\'s folder)')
    PV.INPUT_ARGS.setdefault('check', 'first')          # the requirements file and the plan are hashed as inputs
    PV.INPUT_ARGS.setdefault('run_plan', 'first')


def wanted(args) -> bool:
    return bool(getattr(args, 'check', None) or getattr(args, 'run_plan', None)
                or getattr(args, 'reproduce_out', None))


def _err(text) -> int:
    print(f'dexvantage: {text}', file=sys.stderr)
    return EXIT_ERROR


def _write(doc, dest) -> int | None:
    import json
    text = json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False) + '\n'
    if dest == '-':
        sys.stdout.write(text)
        return None
    try:
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        with open(dest + '.part', 'w', encoding='utf-8') as fh:
            fh.write(text)
        os.replace(dest + '.part', dest)
    except OSError as e:
        return _err(f'could not write {dest}: {e.strerror or e}. Choose a folder that exists and that you can write '
                    'to.')
    return None


def run(args, project) -> int:
    """--check and --run-plan (after batch.py's own options); the worst exit code."""
    code = EXIT_OK
    if getattr(args, 'reproduce_out', None) and not getattr(args, 'reproduce', None):
        return _err('--reproduce-out goes with --reproduce RESULT.json')
    if getattr(args, 'check', None):
        code = max(code, check(args, project))
    if getattr(args, 'run_plan', None):
        code = max(code, run_plan(args, project))
    return code


def check(args, project) -> int:
    from .core import jobs
    from .core import provenance as PV
    from .core import requirements as RQ
    vals = list(args.check)
    if len(vals) > 2:
        return _err('--check takes a requirements file and, optionally, the JSON file to write')
    src, dest = vals[0], (vals[1] if len(vals) > 1 else 'scorecard.json')
    if project is None:
        return _err('--check needs a project file (or --sample)')
    try:
        reqs = RQ.load(src)
    except RQ.RequirementsError as e:
        return _err(str(e))
    try:
        import time
        t0 = time.perf_counter()
        card = RQ.check(project.snapshot(), reqs)
    except jobs.StudyError as e:
        return _err(str(e))
    doc = {'schema': RQ.SCHEMA, 'study': 'requirements', **card, 'seconds': round(time.perf_counter() - t0, 4),
           'cancelled': False}
    doc = PV.stamp(doc, project, output=PV.batch_output(doc, dest, option='check'))
    print(RQ.table(card))
    bad = _write(doc, dest)
    if bad is not None:
        return bad
    if dest != '-':                      # say where: without an output file it is the current folder's scorecard.json
        dflt = '' if len(vals) > 1 else ' (no output file given: scorecard.json in the current folder)'
        print(f'Scorecard → {os.path.abspath(dest)}{dflt}')
    from . import batch_track
    batch_track.record('requirements', doc, project, dest, kind='check')     # --track (core/tracking)
    for e in card['errors']:
        print(f'dexvantage: {e["message"]}', file=sys.stderr)
    return RQ.exit_code(card)


def reproduce(args, project) -> int:
    from .core import provenance as PV
    from .core import reproduce as RP
    try:
        tol = PV.parse_tolerances(args.tolerance)
    except ValueError as e:
        return _err(str(e))
    if not os.path.isfile(args.reproduce):
        return _err(f'{args.reproduce} is not there (nothing at that path)')
    rep = RP.reproduce_file(args.reproduce, project, tol, getattr(args, 'dataset', None))
    for line in RP.lines(rep):
        print(line)
    if getattr(args, 'reproduce_out', None):
        bad = _write(PV.stamp(rep, None), args.reproduce_out)
        if bad is not None:
            return bad
    return RP.exit_code(rep)


def run_plan(args, project) -> int:
    from .core import provenance as PV
    from .core import run_plan as RN
    vals = list(args.run_plan)
    if len(vals) > 2:
        return _err('--run-plan takes a plan file and, optionally, the output folder')
    if project is None:
        return _err('--run-plan needs a project file (or --sample)')
    try:
        plan = RN.load(vals[0])
    except RN.PlanError as e:
        return _err(str(e))
    plan_dir = os.path.dirname(os.path.abspath(vals[0]))
    out_dir = vals[1] if len(vals) > 1 else plan_dir
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        return _err(f'could not make {out_dir}: {e.strerror or e}')
    summary, code = RN.run(project, plan, out_dir, plan_dir, getattr(args, 'dataset', None),
                           progress=lambda f, t: print(t))
    dest = os.path.join(out_dir, str(plan.get('summary') or 'plan-summary.json'))
    doc = PV.stamp(summary, project, output={'option': '--run-plan', 'name': os.path.basename(dest),
                                             'in_folder': True})
    for r in summary['steps']:
        extra = r.get('verdict_text') or r.get('verdict') or ''
        print(f'  {r["index"]:>2}  {r["kind"]:9s} {str(r["what"])[:40]:40s} {r["status"]:18s} '
              f'{r.get("out") or ""}' + (f'  ({extra})' if extra else '') +
              (f'\n      {r["message"]}' if r.get('message') and r['exit'] else ''))
    bad = _write(doc, dest)
    if bad is not None:
        return bad
    print(f'Plan "{plan["name"]}": {summary["verdict"]} → {dest}')
    return code
