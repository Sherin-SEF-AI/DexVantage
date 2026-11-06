"""Batch mode: DexVantage without a window.

    dexvantage PROJECT.dexv --analyse summary.json      the project summary (task volumes, voxel counts)
    dexvantage [PROJECT.dexv] --validate OUT_DIR        the validation suite and its report (JSON, CSV, HTML, figures)
    dexvantage PROJECT.dexv --reproduce RESULT.json     re-run what produced RESULT.json and compare every number

'-' as an output writes to standard output. Exit codes: 0 done (validation: every check passed; reproduce: the numbers
reproduce), 1 a failed check or a result that does not reproduce, 2 an error that stopped the run (a file that cannot
be read, a bad option). The study flags of the specification (§11: --grasp-study, --depth-validity, --check …) are
added by the phases that build their studies.

Ported from Vantage batch.py (the structure and the provenance handling).
"""
from __future__ import annotations

import json
import os
import sys

EXIT_OK, EXIT_FAIL, EXIT_ERROR = 0, 1, 2


def add_arguments(ap):
    g = ap.add_argument_group('batch mode (no window)')
    g.add_argument('--analyse', metavar='OUT.json', help='write the project summary study as JSON ("-" for stdout)')
    g.add_argument('--validate', metavar='OUT_DIR', help='run the validation suite and write its report')
    g.add_argument('--models', help='with --validate: comma-separated model ids (default: all)')
    g.add_argument('--quick', action='store_true', help='with --validate: fewer Monte Carlo trials (a smoke run)')
    g.add_argument('--reproduce', metavar='RESULT.json', help='re-run the study that wrote RESULT.json and compare')
    g.add_argument('--tolerance', action='append', metavar='[PATTERN=]VALUE[%]',
                   help='with --reproduce: allowed differences, e.g. 1e-9 or voxels*=0')


def wanted(args) -> bool:
    return bool(args.analyse or args.validate or args.reproduce)


def _load(args):
    from .core.project import Project, ProjectFormatError
    if not args.path:
        return None
    try:
        return Project.load(args.path)
    except ProjectFormatError as e:
        raise SystemExit(_err(str(e)))


def _err(text) -> int:
    print(f'dexvantage: {text}', file=sys.stderr)
    return EXIT_ERROR


def _write(doc, dest):
    text = json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False) + '\n'
    if dest in (None, '', '-'):
        sys.stdout.write(text)
        return
    tmp = dest + '.part'
    with open(tmp, 'w', encoding='utf-8') as fh:
        fh.write(text)
    os.replace(tmp, dest)


def _summary(project, dest) -> dict:
    from .core import jobs, provenance as PV
    doc = jobs.run_study('summary', project.snapshot())
    doc = {k: v for k, v in doc.items() if k not in ('cached',)}
    return PV.stamp(doc, project, output=PV.batch_output(doc, dest, option='analyse'))


def run(args) -> int:
    from .core import provenance as PV
    parser = getattr(args, '_parser', None)
    try:
        project = _load(args)
    except SystemExit as e:
        return int(e.code)
    if args.reproduce:
        return _reproduce(args, project)
    ctx = PV.begin_batch(args, parser) if parser is not None else None
    try:
        if project is not None:
            PV.note_project(project)
        code = EXIT_OK
        if args.analyse:
            if project is None:
                return _err('--analyse needs a project file')
            doc = _summary(project, args.analyse)
            _write(doc, args.analyse)
            if args.analyse != '-':
                vols = doc['volumes']
                print(f'{project.name}: {len(vols)} task volumes, '
                      f'{sum(v["voxels_dense"] for v in vols):,} voxels at {doc["voxel_mm"]:g} mm → {args.analyse}')
        if args.validate:
            from . import validation as VA
            from .validation.suite import qt_app
            st = {'quick': bool(args.quick)}
            if args.models:
                bad = [m for m in args.models.split(',') if m.strip() not in VA.MODEL_IDS]
                if bad:
                    return _err(f'unknown model id(s) {", ".join(bad)}; known: {", ".join(VA.MODEL_IDS)}')
                st['models'] = args.models
            qt_app()
            res = VA.run(project, settings=VA.ValidationSettings.from_dict(st))
            for c in res.checks:
                err = f'{c.error:.3g}' if c.error == c.error else 'n/a'
                print(f'{"pass" if c.passed else "FAIL"}  {c.id:34s} {err} / {c.tolerance:g} {c.unit}'
                      + (f'  ({c.exception})' if c.exception else ''))
            os.makedirs(args.validate, exist_ok=True)
            PV.stamp_paths(VA.export(res, project, args.validate), project, args.validate)
            print(f'{res.passed}/{len(res.checks)} checks passed; report in {args.validate}')
            code = max(code, VA.exit_code(res))
        return code
    finally:
        if ctx is not None:
            PV.end_batch()


def _reproduce(args, project) -> int:
    from .core import provenance as PV
    try:
        with open(args.reproduce, encoding='utf-8') as fh:
            recorded = json.load(fh)
    except (OSError, ValueError) as e:
        return _err(f'{args.reproduce} could not be read: {e}')
    prov = recorded.get(PV.KEY) or {}
    study = prov.get('study')
    try:
        tol = PV.parse_tolerances(args.tolerance)
    except ValueError as e:
        return _err(str(e))
    if study != 'summary':
        rep = PV.report(os.path.basename(args.reproduce), recorded, None, tol,
                        error=f'this build cannot re-run the study "{study}" yet')
    elif project is None:
        return _err('--reproduce needs the project file the result was made from')
    else:
        doc = _summary(project, '-')
        rep = PV.report(os.path.basename(args.reproduce), recorded, doc, tol, now_provenance=doc[PV.KEY],
                        project_now=PV.project_digest(project))
    print(rep['message'] if 'message' in rep else rep.get('verdict_text', ''))
    for r in rep.get('reasons') or []:
        print(f'  · {r}')
    return EXIT_OK if rep['verdict'] in ('reproduces', 'reproduces_within_tolerance') else EXIT_FAIL
